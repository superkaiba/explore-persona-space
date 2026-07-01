#!/usr/bin/env python3
# ruff: noqa: RUF003
# Intentional Unicode (→, ρ, ×, ², r_B) in scientific docstrings + log messages.
"""Issue #810 Phase B: re-extract answer-side POSITION summaries (θ0, no training).

Teacher-forces #658's STORED base-model completions back through
``Qwen/Qwen2.5-7B-Instruct`` and captures the residual-stream activation at a
thin ALIGNED SUBSET of answer-side positions per (context, probe):

- ``im_end``  — the ``<|im_end|>`` token (id 151645) after the answer content.
- ``turn_nl`` — the ``\\n`` (id 198) after ``<|im_end|>``: the answer-side mirror
  of #594's ``c_C`` last-input-token boundary. The H1 headline candidate.
- ``tail_1..16`` — end-aligned answer-CONTENT positions (``tail_1`` == last token).
- ``head_0..15`` — start-aligned answer-CONTENT positions.

The stored ``answer_spans/<ctx>.pt`` spans are answer-CONTENT only, so tail/head
are slice-derivable from them BUT ``im_end`` / ``turn_nl`` are the two boundary
positions AFTER ``span_end`` — they need a fresh forward over
``prompt + answer + <|im_end|> + \\n``. This pass captures ALL 34 positions in
one forward per probe (so #812 reuses the same store, plan §13) and writes the
per-context probe-mean summary vectors to the aligned-subset store.

Extends the #658 extraction path (``issue658_extract_base_store.capture_v0_for_
context`` / ``AnswerSpanCapture`` / ``LayerCapture``) + reuses #594's
``messages_for_instance``; it does NOT re-implement the hooks or the chat
template. Forward-pass-only (no sampling, no training).

Storage (plan §13, SHARED with #812): one file per context
``<HF_PREFIX>/answer_position_sweep/<context_id>.pt`` — a dict
``{context_id, capture_layers:[0..27], positions:[...34...], pos_vectors:
(n_positions, 28, 3584) fp16, coverage: {position: probe_count}}``.

Local batteries under ``data/`` are gitignored (absent from the git-clone GCP
lane), so the 50-context battery is fetched from the sha256-pinned HF snapshot
(``BATTERY50_HF_FILE``) with a local-file fast path.

Pod-side contract: ``[phase=...]`` log lines ending in ``[phase=done]`` on a
graceful exit + a ``poll_pipeline.py``-conformant end-of-run sentinel.

Usage::

    # production (auto lane, GCP-first, 1x GPU eval intent):
    uv run python scripts/dispatch_issue.py --issue 810 --intent eval \\
        --workload-cmd 'REPO_ROOT="$WORKLOAD_ROOT" \\
        uv run python scripts/issue810_extract_positions.py --gpu'

    # local CPU smoke (tiny same-family model, 1 context, all positions):
    uv run python scripts/issue810_extract_positions.py --smoke \\
        --model Qwen/Qwen2.5-0.5B-Instruct --n-ctx 1 --n-probes 2 \\
        --out-dir /tmp/i810_smoke --device cpu
"""

from __future__ import annotations

import os

os.environ.setdefault("HF_HOME", os.environ.get("HF_HOME", "/workspace/.cache/huggingface"))

import argparse
import logging
import sys
import time
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import torch  # noqa: E402

# Cross-script helper imports hoisted to module top so a missing symbol crashes
# at process start, never inside a smoke-skipped branch (gotchas.md #606).
from issue594_common import messages_for_instance  # noqa: E402
from issue594_extract_context_vectors import LayerCapture  # noqa: E402
from issue810_common import (  # noqa: E402
    ANSWER_POSITION_SWEEP_SUBDIR,
    BATTERY50_HF_FILE,
    BATTERY50_SHA256,
    DEFAULT_MODEL,
    EXPECTED_HIDDEN,
    EXPECTED_LAYERS,
    HF_DATA_REPO,
    HF_PREFIX,
    I658_RAW_COMPLETIONS_PREFIX,
    I658_STORE_MANIFEST,
    IM_END_TOKEN_ID,
    assert_sha256,
    context_ids_from_manifest,
    dump_json,
    load_json,
    reproducibility_metadata,
    sha256_file,
    stored_position_names,
    tail_head_position_index,
)

# Project dotenv wrapper (#745): robust .env load + HF-upload accelerators.
from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv(str(PROJECT_ROOT / ".env"))

logger = logging.getLogger("issue810_extract")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

SENTINEL_SCHEMA_VERSION = 1


# ── full-sequence answer+boundary capture (extends AnswerSpanCapture) ─────────


def _position_span_stack(
    capture: LayerCapture, n_layers: int, start: int, end: int
) -> torch.Tensor:
    """(L, S, H) fp16 CPU stack over positions [start, end) — clears the buffer.

    Same shape/dtype as #658's ``AnswerSpanCapture.answer_span_stack``; the
    #810 pass captures the answer content + the two boundary tokens in ONE
    forward, so it slices the union span [answer_start, turn_nl+1) then indexes
    the individual positions afterward.
    """
    assert 0 <= start < end, (start, end)
    vecs = [capture.latest[li][0, start:end, :].to(torch.float16).cpu() for li in range(n_layers)]
    capture.latest.clear()
    return torch.stack(vecs)  # (L, S, H)


def _positions_for_span(span_len: int, boundary_offset: int) -> dict[str, int]:
    """Map each stored position name to its index in the captured union span.

    The captured union span covers the answer-content positions [0, span_len)
    followed by ``im_end`` at ``span_len`` and ``turn_nl`` at ``span_len + 1``
    (indices are RELATIVE to the union span start). A tail_k/head_k position
    out of range for a short answer is OMITTED (recorded as a coverage miss),
    never a crash.

    ``boundary_offset`` == span_len (the union span starts at the first answer
    content token; im_end/turn_nl sit immediately after the content).
    """
    idx: dict[str, int] = {}
    for name in stored_position_names():
        if name == "im_end":
            idx[name] = boundary_offset
        elif name == "turn_nl":
            idx[name] = boundary_offset + 1
        else:
            pos = tail_head_position_index(name, span_len)
            if pos is not None:
                idx[name] = pos
    return idx


def capture_positions_for_context(
    model,
    tokenizer,
    instance: dict,
    probes: list[str],
    completions: list[str],
    capture: LayerCapture,
    n_layers: int,
    capture_layers: list[int],
    batch_probes: int,
) -> tuple[dict[str, torch.Tensor], dict[str, int], dict]:
    """Teacher-force each (prompt + answer + <|im_end|> + \\n); capture positions.

    Returns ``(pos_summaries, coverage, diag)`` where
      pos_summaries[position] = (Lc, H) probe-MEAN summary vector for that
        position over the probes that had it,
      coverage[position] = number of probes that contributed,
      diag = per-context diagnostics (n_probes_used, empty_completions,
        median_answer_len, boundary_token_ids_seen).

    The im_end / turn_nl positions are the two boundary tokens AFTER the
    answer content; they are appended to the teacher-forced sequence so the
    forward materializes their residual stream. Fail loud on a boundary-token
    id mismatch (never silently capture the wrong slot) — the im_end id is
    asserted; the turn_nl id is recorded (tokenizer-dependent) + asserted to
    decode to a newline-bearing token.

    ``batch_probes`` is accepted for signature parity with the CUDA-OOM
    hot-fix knob (gotchas.md #761 ``--batch-probes``); the capture is batch-1
    per probe here (variable answer lengths make padding wasteful at n=48), so
    it is a documented no-op — the halve-the-batch OOM lever is
    ``PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`` + a shorter cap.
    """
    _ = batch_probes  # documented no-op (batch-1 per probe; see docstring)
    lc = len(capture_layers)
    H = model.config.hidden_size
    # Accumulators: sum over probes + count per position (probe-mean at the end).
    accum: dict[str, torch.Tensor] = {}
    coverage: dict[str, int] = {p: 0 for p in stored_position_names()}
    n_used = 0
    empty = 0
    ans_lens: list[int] = []
    turn_nl_ids_seen: set[int] = set()

    nl_ids = tokenizer.encode("\n", add_special_tokens=False)
    if len(nl_ids) != 1:
        raise RuntimeError(f"expected single-token '\\n', got {nl_ids} (tokenizer drift)")
    nl_id = nl_ids[0]

    for q, ans in zip(probes, completions, strict=True):
        messages = messages_for_instance(instance, q)
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prompt_ids = tokenizer(prompt_text, return_tensors="pt", padding=False)["input_ids"]
        ans_ids = tokenizer(ans, return_tensors="pt", add_special_tokens=False)["input_ids"]
        if ans_ids.shape[1] == 0:
            empty += 1
            logger.warning("empty completion for %s probe=%r — skipping", instance["id"], q[:40])
            continue
        prompt_len = int(prompt_ids.shape[1])
        ans_len = int(ans_ids.shape[1])
        ans_lens.append(ans_len)
        # Append the natural Qwen turn-end: <|im_end|> then \n (the c_C mirror).
        boundary = torch.tensor([[IM_END_TOKEN_ID, nl_id]], dtype=prompt_ids.dtype)
        full_ids = torch.cat([prompt_ids, ans_ids, boundary], dim=1).to(model.device)
        with torch.no_grad():
            _ = model(input_ids=full_ids)
        # Union span: [answer_start, turn_nl+1) = ans_len content + 2 boundary.
        union_start = prompt_len
        union_end = prompt_len + ans_len + 2
        union = _position_span_stack(capture, n_layers, union_start, union_end)  # (L, S+2, H)
        union = union[capture_layers]  # (Lc, ans_len+2, H) fp16
        # Boundary-token sanity: the union's last two positions are the appended
        # im_end + nl by construction; assert the FED ids (fail loud on drift).
        fed = full_ids[0, prompt_len + ans_len : prompt_len + ans_len + 2].tolist()
        assert fed[0] == IM_END_TOKEN_ID, (
            f"im_end slot fed id {fed[0]} != {IM_END_TOKEN_ID} for {instance['id']} {q[:30]!r}"
        )
        assert fed[1] == nl_id, f"turn_nl slot fed id {fed[1]} != {nl_id} (\\n)"
        turn_nl_ids_seen.add(fed[1])
        pos_idx = _positions_for_span(ans_len, boundary_offset=ans_len)
        for name, si in pos_idx.items():
            vec = union[:, si, :].float()  # (Lc, H)
            if name not in accum:
                accum[name] = torch.zeros(lc, H, dtype=torch.float32)
            accum[name] += vec
            coverage[name] += 1
        n_used += 1

    if n_used == 0:
        raise RuntimeError(f"context {instance['id']}: every probe produced an empty answer")
    pos_summaries = {name: (accum[name] / coverage[name]) for name in accum}
    # Assert boundary positions are ALWAYS covered (they don't depend on span_len).
    for b in ("im_end", "turn_nl"):
        if coverage[b] != n_used:
            raise RuntimeError(
                f"context {instance['id']}: boundary position {b} coverage "
                f"{coverage[b]} != n_used {n_used} — the boundary token was not "
                "captured on every probe (capture/slice bug)"
            )
    ans_lens.sort()
    diag = {
        "n_probes_used": n_used,
        "empty_completions": empty,
        "median_answer_len": ans_lens[len(ans_lens) // 2] if ans_lens else 0,
        "turn_nl_ids_seen": sorted(turn_nl_ids_seen),
    }
    return pos_summaries, coverage, diag


# ── inputs (battery + completions) ───────────────────────────────────────────


def _resolve_battery(local_hint: Path | None) -> dict:
    """Load + sha256-pin the 50-context battery (local fast path, else HF snapshot).

    Local ``data/issue594/battery.json`` is gitignored (absent from the git-clone
    GCP lane), so on a miss we fetch the sha256-pinned HF snapshot
    (``BATTERY50_HF_FILE``) — the artifact-reuse (h) fetchability contract.
    Either way the sha256 is asserted against ``BATTERY50_SHA256`` (fail loud on
    drift, the #600 HF-mirror guard).
    """
    from huggingface_hub import hf_hub_download

    candidates = []
    if local_hint is not None:
        candidates.append(Path(local_hint))
    candidates.append(PROJECT_ROOT / "data" / "issue594" / "battery.json")
    for c in candidates:
        if c.is_file() and sha256_file(c) == BATTERY50_SHA256:
            logger.info("battery: local sha-matched %s", c)
            return load_json(c)
    logger.info("battery: fetching sha-pinned HF snapshot %s", BATTERY50_HF_FILE)
    path = hf_hub_download(HF_DATA_REPO, BATTERY50_HF_FILE, repo_type="dataset")
    assert_sha256(path, BATTERY50_SHA256, "battery50")
    return load_json(path)


def _load_stored_completions(ctx_id: str) -> list[dict]:
    """The 48 stored (probe, completion) pairs for one context from HF.

    Reads ``raw_completions/raw_completions/<ctx>.json`` (schema
    ``{context_id, completions:[{probe, completion}, ...]}``) — the model's OWN
    on-policy answers #658 generated + stored. NO regeneration (single-variable
    discipline: the summary is the variable, the completions inherited).
    """
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        HF_DATA_REPO, f"{I658_RAW_COMPLETIONS_PREFIX}/{ctx_id}.json", repo_type="dataset"
    )
    blob = load_json(path)
    if blob.get("context_id") != ctx_id:
        raise RuntimeError(f"completions ctx mismatch: {blob.get('context_id')} != {ctx_id}")
    cells = blob["completions"]
    if not cells:
        raise RuntimeError(f"context {ctx_id}: no stored completions")
    return cells


def _load_manifest_context_ids() -> list[str]:
    from huggingface_hub import hf_hub_download

    man = load_json(hf_hub_download(HF_DATA_REPO, I658_STORE_MANIFEST, repo_type="dataset"))
    return context_ids_from_manifest(man)


# ── HF model load ─────────────────────────────────────────────────────────────


def _load_model(model_name: str, device: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if device == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, device_map={"": torch.device("cuda:0")}
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
    model.eval()
    return model, tokenizer


# ── upload (fail-loud) ────────────────────────────────────────────────────────


def _upload_store(out_dir: Path, ctx_ids: list[str], smoke: bool) -> str:
    """Bulk-commit the aligned-subset store to HF (one upload_folder commit).

    Uploads ``answer_position_sweep/<ctx>.pt`` (+ manifest.json) via ONE
    ``upload_folder`` commit (never a per-file loop — the #664 504-storm), then
    verifies the per-context file count on a FRESH listing (fail loud on a
    mismatch). Skipped for --smoke / --no-upload by the caller.
    """
    from huggingface_hub import HfApi, list_repo_files

    subdir = ANSWER_POSITION_SWEEP_SUBDIR + ("_smoke" if smoke else "")
    path_in_repo = f"{HF_PREFIX}/{subdir}"
    api = HfApi()
    api.upload_folder(
        folder_path=str(out_dir),
        path_in_repo=path_in_repo,
        repo_id=HF_DATA_REPO,
        repo_type="dataset",
        allow_patterns=["*.pt", "manifest.json"],
        commit_message=f"issue #810: answer position sweep store ({len(ctx_ids)} contexts)",
    )
    remote = set(list_repo_files(HF_DATA_REPO, repo_type="dataset", revision="main"))
    expected = {f"{path_in_repo}/{c}.pt" for c in ctx_ids}
    missing = expected - remote
    if missing:
        raise RuntimeError(
            f"aligned-subset store upload verification FAILED: {len(missing)} context "
            f"files missing on the Hub under {path_in_repo}/ (e.g. {sorted(missing)[:3]})"
        )
    logger.info("aligned-subset store verified: %d contexts under %s/", len(expected), path_in_repo)
    return path_in_repo


# ── sentinel (poll_pipeline contract) ─────────────────────────────────────────


def _write_sentinel(kind: str, note: dict, out_dir: Path) -> None:
    """Write the poll_pipeline.py-conformant end-of-run sentinel.

    Required keys per poll_pipeline._SENTINEL_REQUIRED_KEYS:
    sentinel_schema_version (int 1), kind (full marker string), version (int).
    The marker body goes under ``note``.
    """
    slug = kind.replace(":", "_")
    log_dir = Path("/workspace/logs")
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        target = log_dir / f"issue-810-{slug}-{int(time.time())}.json"
    except OSError:
        # Off-pod (no /workspace): write next to the output for the smoke.
        target = out_dir / f"issue-810-{slug}-sentinel.json"
    dump_json(
        {
            "sentinel_schema_version": SENTINEL_SCHEMA_VERSION,
            "kind": kind,
            "version": 1,
            "note": note,
            "ts": int(time.time()),
        },
        target,
    )
    logger.info("wrote sentinel %s", target)


# ── main ──────────────────────────────────────────────────────────────────────


def main() -> int:
    ap = argparse.ArgumentParser(description="Issue #810 Phase B: answer position sweep extraction")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--device", choices=["cuda", "cpu"], default=None)
    ap.add_argument("--gpu", action="store_true", help="force --device cuda")
    ap.add_argument("--out-dir", default=str(PROJECT_ROOT / "data" / "issue_810" / "store"))
    ap.add_argument("--battery", default=None, help="local battery.json fast path (sha-pinned)")
    ap.add_argument("--n-ctx", type=int, default=None, help="smoke: cap contexts")
    ap.add_argument("--n-probes", type=int, default=None, help="smoke: cap probes/context")
    ap.add_argument("--expected-layers", type=int, default=EXPECTED_LAYERS)
    ap.add_argument("--expected-hidden", type=int, default=EXPECTED_HIDDEN)
    ap.add_argument("--batch-probes", type=int, default=8, help="CUDA-OOM knob (documented no-op)")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--no-upload", action="store_true")
    args = ap.parse_args()

    device = args.device or ("cuda" if (args.gpu and torch.cuda.is_available()) else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    logger.info("[phase=setup] loading battery + manifest")
    battery = _resolve_battery(Path(args.battery) if args.battery else None)
    instances = {i["id"]: i for i in battery["instances"]}
    # In --smoke we cannot rely on the real 50-context manifest join, but the real
    # run pins to the manifest's 50 contexts (the LOCO fold order).
    ctx_ids = _load_manifest_context_ids()
    if args.n_ctx is not None:
        ctx_ids = ctx_ids[: args.n_ctx]
    logger.info("contexts to extract: %d (device=%s model=%s)", len(ctx_ids), device, args.model)

    logger.info("[phase=load_model] %s", args.model)
    model, tokenizer = _load_model(args.model, device)
    n_layers = model.config.num_hidden_layers
    capture_layers = list(range(n_layers))
    if not args.smoke:
        assert n_layers == args.expected_layers, (n_layers, args.expected_layers)
        assert model.config.hidden_size == args.expected_hidden, model.config.hidden_size
    capture = LayerCapture(model, n_layers)

    per_ctx_diag: dict[str, dict] = {}
    try:
        for ci, ctx_id in enumerate(ctx_ids):
            logger.info("[phase=extract] context %d/%d %s", ci + 1, len(ctx_ids), ctx_id)
            if ctx_id not in instances:
                raise RuntimeError(f"context {ctx_id} absent from battery (coverage gap)")
            cells = _load_stored_completions(ctx_id)
            if args.n_probes is not None:
                cells = cells[: args.n_probes]
            probes = [c["probe"] for c in cells]
            completions = [c["completion"] for c in cells]
            pos_summaries, coverage, diag = capture_positions_for_context(
                model,
                tokenizer,
                instances[ctx_id],
                probes,
                completions,
                capture,
                n_layers,
                capture_layers,
                args.batch_probes,
            )
            names = stored_position_names()
            # Stack positions into (n_positions, Lc, H) fp16; a position missing
            # for EVERY probe (impossible for boundary; possible for a deep
            # tail_k on all-short answers) is recorded as absent in coverage and
            # its row is zero-filled (never silently dropped — the reader keys on
            # coverage, and a 0-coverage row is excluded downstream).
            H = model.config.hidden_size
            pos_stack = torch.zeros(len(names), len(capture_layers), H, dtype=torch.float16)
            for pi, name in enumerate(names):
                if name in pos_summaries:
                    pos_stack[pi] = pos_summaries[name].to(torch.float16)
            blob = {
                "context_id": ctx_id,
                "capture_layers": capture_layers,
                "positions": names,
                "pos_vectors": pos_stack,  # (n_positions, Lc, H) fp16
                "coverage": coverage,
                "model": args.model,
            }
            torch.save(blob, out_dir / f"{ctx_id}.pt")
            per_ctx_diag[ctx_id] = diag
    finally:
        capture.remove()

    # Manifest (plan §13): positions list, dtype, coverage semantics, provenance.
    manifest = {
        "positions": stored_position_names(),
        "capture_layers": capture_layers,
        "dtype": "float16",
        "pos_vectors_shape": ["n_positions", len(capture_layers), model.config.hidden_size],
        "coverage_semantics": "per-position probe count contributing to the probe-mean summary",
        "n_contexts": len(ctx_ids),
        "context_ids": ctx_ids,
        "model": args.model,
        "battery_sha256": BATTERY50_SHA256,
        "per_context_diag": per_ctx_diag,
        "boundary_note": (
            "im_end=<|im_end|> id 151645 (position span_end); turn_nl=\\n after "
            "im_end (span_end+1, the c_C answer-side mirror). Both appended to the "
            "teacher-forced sequence and captured fresh (NOT slice-derivable from "
            "the answer-content span)."
        ),
        "reproducibility": reproducibility_metadata(),
        "smoke": args.smoke,
    }
    dump_json(manifest, out_dir / "manifest.json")
    logger.info("wrote manifest (%d contexts) to %s", len(ctx_ids), out_dir)

    path_in_repo = None
    if not args.no_upload and not args.smoke:
        logger.info("[phase=upload] aligned-subset store")
        path_in_repo = _upload_store(out_dir, ctx_ids, smoke=False)

    note = {
        "phase": "B_extract_positions",
        "n_contexts": len(ctx_ids),
        "positions": len(stored_position_names()),
        "hf_path": path_in_repo,
        "elapsed_s": round(time.time() - t0, 1),
        "store_files_sha256": {
            c: sha256_file(out_dir / f"{c}.pt")[:16] for c in ctx_ids[: min(3, len(ctx_ids))]
        },
    }
    _write_sentinel("epm:results", note, out_dir)
    logger.info("[phase=done] extraction complete: %d contexts", len(ctx_ids))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:
        logger.error("[phase=failed] extraction crashed:\n%s", traceback.format_exc())
        raise
