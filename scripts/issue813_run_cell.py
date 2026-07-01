#!/usr/bin/env python3
# ruff: noqa: RUF001, RUF002, RUF003
# Intentional Unicode (Δ, M⁺, →, ×, c_C, v_A, ‖·‖) in scientific docstrings + log messages.
"""Issue #813 — ONE (behavior, substrate) map-fit extraction cell.

For ONE ``(behavior, substrate)`` cell over the shared 50-context #594 battery
this CLI:

1. Stages + loads the #537 ``default``-context adapter for ``behavior`` as a
   ``PeftModel`` on base Qwen-2.5-7B-Instruct (rsLoRA honored; the em adapter's
   NESTED ``sft_em_adapter/`` subfolder resolved by ``resolve_adapter_subfolder``
   inherited from #667). Asserts base id + ``use_rslora`` (fitness (f)/(g)).
2. For each of the 50 battery contexts × the substrate's K questions:
   - builds ``T_ctx(q)`` via the #594 ``messages_for_instance`` recipe;
   - generates the frozen BASE greedy response ``R`` (temp=0, vLLM batched);
   - teacher-forces ``T_ctx(q) + R`` through base θ0 AND θ⁺ once each, capturing
     the FULL per-token residual over (ctx span + answer span) at ALL 28 layers,
     fp16, BOTH models;
   - STREAM-UPLOADS this cell's unreduced ``.npz`` to HF, then DELETES local
     (peak local footprint stays ~one cell — never accumulate; #664/EDQUOT);
   - reduces to ``c_C`` (last-input-token, 28 layers) + ``v_A`` (mean-answer-span,
     28 layers) per question → accumulates.
3. Question-averages ``c_C`` / ``v_A`` over the substrate's questions →
   50 ``c_C`` rows + 50 ``v_A`` rows (base + trained), 28 layers → writes the
   reduced per-(behavior, substrate) summary ``.npz`` (the map-fit input; small,
   ~1 MB) locally under ``eval_results/issue_813/reduced/<behavior>/<substrate>/``.

CONTENT HYGIENE: ``em`` uses Betley harmful-content probes — this script NEVER
prints/logs their text; it digests by row/token COUNT + activations only.
Benign behaviors (marker/fact/sycophancy) are unaffected.

Activation extraction is ``transformers`` forward hooks (NOT vLLM). vLLM is used
ONLY for the frozen base-R generation (which returns text, no activations).

Usage (one cell)::

    uv run python scripts/issue813_run_cell.py \\
        --behavior marker --substrate generic \\
        --out-root eval_results/issue_813 --gpu-id 0 \\
        --upload  # stream unreduced .npz to HF; omit for local-only smoke
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HOME", "/workspace/.cache/huggingface")
# vLLM EngineCore fork() poisoning guard (.claude/rules/gotchas.md § entry 26 /
# issue667_extract.py): a pre-LLM() transformers/tokenizer touch poisons the
# EngineCore fork. spawn (not fork) avoids the silent worker death.
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import issue594_common as i594  # noqa: E402

# The reused #667 extractor primitives (adapter-load, rsLoRA gauge, vLLM gen,
# teacher-force reads via extract_layer_activations forward hooks). These are
# imported VERBATIM — no local re-implementation of the plumbing.
import issue667_extract as ex667  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from explore_persona_space.analysis.extraction import extract_layer_activations  # noqa: E402
from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

logger = logging.getLogger("issue813.run_cell")

# ── Constants ────────────────────────────────────────────────────────────────
DATA_REPO = "superkaiba1/explore-persona-space-data"
EXPERIMENT_NAME = "issue813_mapchange_substrate"
BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
HIDDEN = 3584
N_LAYERS = 28  # Qwen-2.5-7B-Instruct decoder blocks 0..27
# Frozen headline read layer (#651/#658 primary read layer). The substrate-swap
# null (issue813_analysis.py) resamples questions at THIS layer only, so the
# per-question headline-layer c_C/v_A rows are persisted for it (keeping the
# small reduced summary from ballooning to all-28-layer per-question size).
HEADLINE_LAYER = 14
SEED = 42
BEHAVIORS = ("em", "fact", "sycophancy", "marker")
SUBSTRATES = ("generic", "elicit", "mix")
# Per-behavior base-R generation cap (marker end-of-completion needs ≥2048; #260).
MAX_NEW_TOKENS = {"marker": 2048, "fact": 1024, "sycophancy": 1024, "em": 1024}
# Atomic per-(behavior, substrate) completion sentinel (resume-skip predicate).
CELL_DONE_SENTINEL = ".done"
# Unreduced-.npz upload batching (B3, #664/#488): a per-file HfApi().upload_file
# inside the (context, question) loop makes one HF commit per pair (~23,850 commits
# across the sweep, blowing the 256-commits/hr throttle). Buffer each cell's .npz
# files and flush ONE HfApi.create_commit per BATCH_UPLOAD_CHUNK files (many
# CommitOperationAdds per commit), deleting local per flush so peak local footprint
# stays ~one chunk (~BATCH_UPLOAD_CHUNK × per-cell bytes), never the full grid.
BATCH_UPLOAD_CHUNK = 100


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


# ── Substrate question pools (the single manipulated variable) ─────────────────


def _generic_probes() -> list[str]:
    """The 48 UltraChat generic probes (#594 probes_ultrachat.json `probes`)."""
    path = PROJECT_ROOT / "data" / "issue594" / "probes_ultrachat.json"
    d = json.loads(path.read_text())
    return [p["text"] if isinstance(p, dict) else str(p) for p in d["probes"]]


def substrate_questions(
    behavior: str, substrate: str, *, max_questions: int | None = None
) -> list[str]:
    """The question pool defining c_C + v_A over the battery, per substrate (plan §4.2/§5).

    - generic: 48 UltraChat probes (anchor; behavior does NOT fire → E≈0).
    - elicit:  the behavior's own #537 eval pool (marker 32 / fact 30 / syco 25 / em 8).
    - mix:     equal-half blend, size ``2·min(n_e, 48)`` — ``min(n_e,48)`` generic +
               all ``n_e`` eliciting (equalize-down 1:1, seed-42 generic subsample),
               so the mix is NOT silently generic-dominated (plan §5).
    """
    generic = _generic_probes()
    elicit = ex667.load_eval_probes(behavior)  # #537 pool, flat list[str] (reused)
    if substrate == "generic":
        qs = list(generic)
    elif substrate == "elicit":
        qs = list(elicit)
    elif substrate == "mix":
        n_e = len(elicit)
        n_g = min(n_e, len(generic))
        rng = np.random.default_rng(SEED)
        gen_idx = sorted(rng.choice(len(generic), size=n_g, replace=False).tolist())
        qs = [generic[i] for i in gen_idx] + list(elicit)
    else:
        raise ValueError(f"unknown substrate {substrate!r} (expected one of {SUBSTRATES})")
    if max_questions is not None:
        qs = qs[:max_questions]
    return qs


# ── Battery contexts (the 50 map inputs) ───────────────────────────────────────


def load_battery_instances(max_contexts: int | None = None) -> list[dict]:
    """The 50 #594 battery contexts (the shared map inputs), optionally capped (smoke)."""
    _meta, instances = i594.load_battery(PROJECT_ROOT / "data" / "issue594" / "battery.json")
    if max_contexts is not None:
        instances = instances[:max_contexts]
    return instances


# ── Full per-token residual capture (unreduced) + c_C / v_A reduction ──────────


@torch.no_grad()
def _capture_full_and_reduce(
    base_model, trained_model, tok, messages: list[dict], response: str, device
) -> dict:
    """Teacher-force ``messages + response`` through base+trained; capture FULL residuals.

    Returns a dict with:
      - ``full_base`` / ``full_trained``: (T, 28, HIDDEN) fp16 per-token residual over
        the WHOLE sequence (ctx span + answer span), all 28 layers — the UNREDUCED store.
      - ``c_C_base`` / ``c_C_trained``: (28, HIDDEN) fp32 last-input-token residual (the
        map INPUT, #594/#658 recipe).
      - ``v_A_base`` / ``v_A_trained``: (28, HIDDEN) fp32 mean-over-answer-span residual
        (the map OUTPUT, #658 v0(C) recipe).
      - ``prompt_len`` / ``full_len``: token counts (digest only, no text).

    The answer span is [prompt_len : full_len); the last-input-token slot is
    prompt_len - 1 (the generation-prompt suffix position). Fails loud on an empty
    response span.
    """
    prompt_text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    full_msgs = [*messages, {"role": "assistant", "content": response}]
    full_text = tok.apply_chat_template(full_msgs, tokenize=False, add_generation_prompt=False)
    prompt_ids = tok.encode(prompt_text, add_special_tokens=False)
    full_ids = tok.encode(full_text, add_special_tokens=False)
    p = len(prompt_ids)
    if full_ids[:p] != prompt_ids:
        # chat-template drift; fall back to the longest common prefix (fail-loud if tiny).
        lcp = 0
        for a, b in zip(prompt_ids, full_ids, strict=False):
            if a != b:
                break
            lcp += 1
        if lcp < max(1, p - 4):
            raise RuntimeError(
                f"prompt-prefix drift: lcp={lcp} vs prompt_len={p} — chat-template mismatch"
            )
        p = lcp
    full_len = len(full_ids)
    if full_len <= p:
        raise RuntimeError("empty response span — base R produced zero tokens")
    ids = torch.tensor([full_ids], dtype=torch.long, device=device)
    all_layers = list(range(N_LAYERS))
    acts_b = extract_layer_activations(base_model, ids, all_layers)  # {li: (1, T, H)}
    acts_t = extract_layer_activations(trained_model, ids, all_layers)

    # UNREDUCED: (T, 28, HIDDEN) fp16, both models.
    full_base = np.stack([acts_b[li][0].float().cpu().numpy() for li in all_layers], axis=1).astype(
        np.float16
    )
    full_trained = np.stack(
        [acts_t[li][0].float().cpu().numpy() for li in all_layers], axis=1
    ).astype(np.float16)
    assert full_base.shape == (full_len, N_LAYERS, HIDDEN), full_base.shape

    # REDUCED: c_C = last-input-token (slot p-1); v_A = mean over answer span [p:full_len).
    c_c_base = np.stack([acts_b[li][0, p - 1, :].float().cpu().numpy() for li in all_layers])
    c_c_trained = np.stack([acts_t[li][0, p - 1, :].float().cpu().numpy() for li in all_layers])
    v_a_base = np.stack(
        [acts_b[li][0, p:full_len, :].float().mean(0).cpu().numpy() for li in all_layers]
    )
    v_a_trained = np.stack(
        [acts_t[li][0, p:full_len, :].float().mean(0).cpu().numpy() for li in all_layers]
    )
    for name, arr in (("c_C", c_c_base), ("v_A", v_a_base)):
        assert arr.shape == (N_LAYERS, HIDDEN), f"{name} {arr.shape}"
    return {
        "full_base": full_base,
        "full_trained": full_trained,
        "c_C_base": c_c_base.astype(np.float32),
        "c_C_trained": c_c_trained.astype(np.float32),
        "v_A_base": v_a_base.astype(np.float32),
        "v_A_trained": v_a_trained.astype(np.float32),
        "prompt_len": p,
        "full_len": full_len,
    }


# ── HF stream-upload of one unreduced (context, question) .npz ─────────────────


def _hf_upload_file(local_path: Path, path_in_repo: str) -> None:
    """Upload ONE (small, reduced) file to the HF data repo, fail-loud.

    Reserved for the tiny per-(behavior, substrate) reduced summaries (``summary.npz`` /
    ``per_question_L14.npz``) — 2 commits per cell, well under the 256/hr throttle. The
    MANY unreduced per-(context, question) ``.npz`` uploads go through
    ``_hf_batch_commit`` (one commit per BATCH_UPLOAD_CHUNK files), never here (B3).

    Uses ``HfApi.upload_file`` directly (accelerated by the shell-level
    HF_XET_HIGH_PERFORMANCE / HF_HUB_ENABLE_HF_TRANSFER defaults). Raises on failure.
    """
    from huggingface_hub import HfApi

    HfApi().upload_file(
        path_or_fileobj=str(local_path),
        path_in_repo=path_in_repo,
        repo_id=DATA_REPO,
        repo_type="dataset",
        commit_message=f"issue813: reduced summary {path_in_repo} ({_git_sha()[:8]})",
    )


def _hf_batch_commit(items: list[tuple[Path, str]]) -> None:
    """Upload a BATCH of unreduced .npz files in ONE HF commit, fail-loud (B3, #664/#488).

    ``items`` is a list of ``(local_path, path_in_repo)`` pairs. Builds one
    ``HfApi.create_commit`` with a ``CommitOperationAdd`` per file — so a whole chunk
    of per-(context, question) ``.npz`` uploads costs ONE commit instead of one-per-file
    (the ~23,850-commit storm the per-file loop caused, blowing the 256-commits/hr
    throttle). Raises on failure (a clean batch upload IS the data-safety contract —
    the caller deletes local only AFTER this returns; upload-then-delete). No-op on an
    empty batch.
    """
    if not items:
        return
    from huggingface_hub import CommitOperationAdd, HfApi

    ops = [
        CommitOperationAdd(path_in_repo=path_in_repo, path_or_fileobj=str(local_path))
        for local_path, path_in_repo in items
    ]
    HfApi().create_commit(
        repo_id=DATA_REPO,
        repo_type="dataset",
        operations=ops,
        commit_message=(
            f"issue813: unreduced activations batch ({len(items)} files, {_git_sha()[:8]})"
        ),
    )


def _df_free_gib(path: str = "/workspace") -> float | None:
    """Free GiB at ``path`` (df monitoring), or None if the path is absent (local VM)."""
    try:
        usage = shutil.disk_usage(path)
    except (FileNotFoundError, OSError):
        return None
    return usage.free / 2**30


def _extract_pairs(
    base,
    trained,
    tok,
    contexts,
    questions,
    r_lookup,
    *,
    behavior,
    substrate,
    reduced_dir,
    device,
    max_new,
    upload,
) -> dict:
    """Teacher-force every (context, question) pair; stream-upload unreduced; accumulate reduced.

    Returns the accumulators ``run_cell`` reduces into the per-(behavior, substrate)
    summary: ``per_q`` (question-averaged 28-layer c_C/v_A per context), the flat
    headline-layer per-question rows for the substrate-swap null, and the counters
    (``n_cells_done`` / ``n_empty`` / ``cell_bytes``). Split out of ``run_cell`` so
    that function stays under the ruff C901 cap.
    """
    per_q: dict[int, dict] = {
        ci: {"c_C_base": [], "c_C_trained": [], "v_A_base": [], "v_A_trained": []}
        for ci in range(len(contexts))
    }
    pq_rows = {"c_C_base": [], "c_C_trained": [], "v_A_base": [], "v_A_trained": []}
    pq_ctx_idx: list[int] = []
    pq_q_idx: list[int] = []
    n_cells_done = 0
    n_empty = 0
    cell_bytes: list[int] = []
    unreduced_prefix = f"{EXPERIMENT_NAME}/unreduced/{behavior}/{substrate}"
    # Pending (local_path, path_in_repo) buffer for the batched upload (B3). Flushed
    # every BATCH_UPLOAD_CHUNK files (ONE HF commit per flush), local deleted per flush.
    pending: list[tuple[Path, str]] = []

    def _flush_pending() -> None:
        if not pending:
            return
        _hf_batch_commit(pending)  # ONE commit for the whole chunk (fail-loud)
        for local_path, _ in pending:
            local_path.unlink()  # DELETE local only AFTER a verified batch commit
        pending.clear()

    for ci, inst in enumerate(contexts):
        ctx_id = inst["id"]
        for qi, q in enumerate(questions):
            r = r_lookup.get((ci, qi))
            if r is None:  # CPU-smoke path (no vLLM)
                r = ex667._greedy_response(
                    base, tok, i594.messages_for_instance(inst, q), device, max_new
                )
            if not r.strip():
                n_empty += 1
                continue
            msgs = i594.messages_for_instance(inst, q)
            caps = _capture_full_and_reduce(base, trained, tok, msgs, r, device)
            for k in ("c_C_base", "c_C_trained", "v_A_base", "v_A_trained"):
                per_q[ci][k].append(caps[k])  # question-average, all 28 layers (map inputs)
                pq_rows[k].append(caps[k][HEADLINE_LAYER])  # headline-layer null row
            pq_ctx_idx.append(ci)
            pq_q_idx.append(qi)
            # stream-upload the UNREDUCED per-(context, question) .npz, then delete local
            npz_name = f"{ctx_id}__q{qi}.npz"
            tmp_npz = (
                reduced_dir.parent.parent.parent / "unreduced_tmp" / behavior / substrate / npz_name
            )
            tmp_npz.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                tmp_npz,
                full_base=caps["full_base"],
                full_trained=caps["full_trained"],
                prompt_len=np.asarray(caps["prompt_len"]),
                full_len=np.asarray(caps["full_len"]),
                behavior=np.asarray(behavior),
                substrate=np.asarray(substrate),
                context_id=np.asarray(ctx_id),
                question_index=np.asarray(qi),
                layers=np.asarray(list(range(N_LAYERS))),
                git_sha=np.asarray(_git_sha()),
            )
            cell_bytes.append(tmp_npz.stat().st_size)
            if upload:
                pending.append((tmp_npz, f"{unreduced_prefix}/{npz_name}"))
                if len(pending) >= BATCH_UPLOAD_CHUNK:
                    _flush_pending()  # ONE commit per chunk, then delete-local (B3)
            n_cells_done += 1
            free = _df_free_gib()  # EDQUOT / df fail-loud monitoring
            if free is not None and free < 10.0:
                # Flush the pending buffer first so a real batch-upload lag (not just
                # unflushed local files) is what trips the floor — fail-loud otherwise.
                if upload:
                    _flush_pending()
                    free = _df_free_gib()
                if free is not None and free < 10.0:
                    raise RuntimeError(
                        f"disk free {free:.1f} GiB < 10 GiB floor at /workspace after "
                        f"{n_cells_done} cells — batch-upload-then-delete not keeping up "
                        "(EDQUOT risk)"
                    )
    if upload:
        _flush_pending()  # final partial chunk (ONE commit), then delete-local
    return {
        "per_q": per_q,
        "pq_rows": pq_rows,
        "pq_ctx_idx": pq_ctx_idx,
        "pq_q_idx": pq_q_idx,
        "n_cells_done": n_cells_done,
        "n_empty": n_empty,
        "cell_bytes": cell_bytes,
    }


# ── One-cell driver ────────────────────────────────────────────────────────────


def run_cell(args) -> dict:
    """Extract + stream-upload one (behavior, substrate) cell; return the phase-1 metrics."""
    behavior = args.behavior
    substrate = args.substrate
    device = ex667._device(args.gpu_id, args.cpu_only)
    dtype = torch.float32 if device.type == "cpu" else torch.bfloat16
    max_new = MAX_NEW_TOKENS[behavior]

    out_root = Path(args.out_root)
    reduced_dir = out_root / "reduced" / behavior / substrate
    reduced_dir.mkdir(parents=True, exist_ok=True)
    sentinel = reduced_dir / CELL_DONE_SENTINEL
    # gate-only never resume-skips (it measures bytes, writes no .done) and never
    # trusts a production .done as "done" — it always runs its one-cell measurement.
    if sentinel.exists() and not args.force and not args.gate_only:
        logger.info(
            "[phase=extract] %s/%s already complete (%s) — skip", behavior, substrate, sentinel
        )
        return {"skipped": True, "behavior": behavior, "substrate": substrate}

    contexts = load_battery_instances(max_contexts=args.max_contexts)
    questions = substrate_questions(behavior, substrate, max_questions=args.max_questions)
    logger.info(
        "[phase=extract] cell behavior=%s substrate=%s | %d contexts × %d questions × 2 models",
        behavior,
        substrate,
        len(contexts),
        len(questions),
    )

    # ── Stage + gauge the adapter BEFORE any GPU work (cheap, HALT early) ──
    adapter_dir = ex667.stage_adapter_local(behavior, "default", SEED)
    gauge = ex667.assert_adapter_gauge(adapter_dir, behavior)
    logger.info(
        "[phase=extract] adapter gauge OK: %s",
        {k: gauge[k] for k in ("r", "lora_alpha", "use_rslora")},
    )

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(BASE_MODEL, token=os.environ.get("HF_TOKEN"))

    # ── Phase A: vLLM batched frozen base R for ALL (context, question) pairs ──
    # On CPU-smoke (no vLLM) fall back to per-pair HF greedy (ex667._greedy_response).
    r_lookup: dict[tuple[int, int], str] = {}
    pair_msgs: list[list[dict]] = []
    pair_keys: list[tuple[int, int]] = []
    for ci, inst in enumerate(contexts):
        for qi, q in enumerate(questions):
            pair_msgs.append(i594.messages_for_instance(inst, q))
            pair_keys.append((ci, qi))
    if device.type != "cpu":
        logger.info("[phase=extract] Phase A: vLLM-generating %d base R responses", len(pair_msgs))
        responses = ex667.vllm_generate_R(tok, pair_msgs, max_new_tokens=max_new)
        r_lookup = dict(zip(pair_keys, responses, strict=True))

    # ── Phase B: load base θ0 + trained θ⁺ for the teacher-force reads ──
    _, base, trained = ex667.load_base_and_trained(adapter_dir, device, dtype)

    # ── Teacher-force every pair; stream-upload unreduced; accumulate reduced ──
    acc = _extract_pairs(
        base,
        trained,
        tok,
        contexts,
        questions,
        r_lookup,
        behavior=behavior,
        substrate=substrate,
        reduced_dir=reduced_dir,
        device=device,
        max_new=max_new,
        upload=args.upload,
    )
    per_q = acc["per_q"]
    pq_rows = acc["pq_rows"]
    pq_ctx_idx = acc["pq_ctx_idx"]
    pq_q_idx = acc["pq_q_idx"]
    n_cells_done = acc["n_cells_done"]
    n_empty = acc["n_empty"]
    cell_bytes = acc["cell_bytes"]

    # ── GATE-ONLY early return (B1): measure per-cell bytes, write NOTHING into the ──
    # production reduced/ tree, and RETURN before the <4-contexts fit guard + the
    # reduced-summary / per-question / .done sentinel writes. So the one-cell gate can
    # run against the production OUT_ROOT (or an isolated temp root) without either
    # crashing the sweep on the <4-contexts guard OR planting a .done that the un-forced
    # Phase-2 sweep would skip (shipping a 1×1 fixture). It touches no reduced/ artifact.
    if args.gate_only:
        gate_metrics = {
            "behavior": behavior,
            "substrate": substrate,
            "gate_only": True,
            "n_contexts": len(contexts),
            "n_questions": len(questions),
            "n_unreduced_cells": n_cells_done,
            "n_empty_R": n_empty,
            "mean_cell_bytes": (float(np.mean(cell_bytes)) if cell_bytes else 0.0),
            "df_free_gib_workspace": _df_free_gib(),
        }
        logger.info(
            "[phase=one_cell_gate] GATE-ONLY %s/%s: %d unreduced cells, mean %.1f MB/cell "
            "(no reduced/summary/.done written)",
            behavior,
            substrate,
            n_cells_done,
            gate_metrics["mean_cell_bytes"] / 2**20,
        )
        return gate_metrics

    # ── Phase C: question-average c_C + v_A over the substrate's questions ──
    def _qavg(ci: int, key: str) -> np.ndarray:
        rows = per_q[ci][key]
        if not rows:
            raise RuntimeError(f"context {ci} has zero non-empty questions for {key}")
        return np.stack(rows).mean(axis=0).astype(np.float32)

    ctx_ids = [inst["id"] for inst in contexts]
    families = [inst["family"] for inst in contexts]
    kept = [ci for ci in range(len(contexts)) if per_q[ci]["c_C_base"]]
    if len(kept) < 4:
        raise RuntimeError(
            f"{behavior}/{substrate}: only {len(kept)} contexts with usable questions (<4) — "
            "cannot fit a map (all base R empty?)"
        )
    c_C_base = np.stack([_qavg(ci, "c_C_base") for ci in kept])  # (n_kept, 28, HIDDEN)
    c_C_trained = np.stack([_qavg(ci, "c_C_trained") for ci in kept])
    v_A_base = np.stack([_qavg(ci, "v_A_base") for ci in kept])
    v_A_trained = np.stack([_qavg(ci, "v_A_trained") for ci in kept])

    reduced_path = reduced_dir / "summary.npz"
    np.savez_compressed(
        reduced_path,
        c_C_base=c_C_base,  # (n_ctx, 28, HIDDEN) fp32
        c_C_trained=c_C_trained,
        v_A_base=v_A_base,
        v_A_trained=v_A_trained,
        context_ids=np.asarray([ctx_ids[ci] for ci in kept], dtype=object),
        families=np.asarray([families[ci] for ci in kept], dtype=object),
        n_contexts=np.asarray(len(kept)),
        n_questions=np.asarray(len(questions)),
        behavior=np.asarray(behavior),
        substrate=np.asarray(substrate),
        layers=np.asarray(list(range(N_LAYERS))),
        git_sha=np.asarray(_git_sha()),
        generated_at=np.asarray(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())),
    )
    if args.upload:
        _hf_upload_file(
            reduced_path, f"{EXPERIMENT_NAME}/reduced/{behavior}/{substrate}/summary.npz"
        )

    # ── Per-question headline-layer (L14) rows for the substrate-swap null ──
    # Flat rows + parallel context/question indices + per-context family (the
    # null resamples questions WITHIN this substrate, re-splits into matched-n
    # pseudo-substrates, question-averages each per context → two pseudo maps).
    pq_path = reduced_dir / f"per_question_L{HEADLINE_LAYER}.npz"
    np.savez_compressed(
        pq_path,
        c_C_base=np.stack(pq_rows["c_C_base"]).astype(np.float32),  # (n_rows, HIDDEN)
        c_C_trained=np.stack(pq_rows["c_C_trained"]).astype(np.float32),
        v_A_base=np.stack(pq_rows["v_A_base"]).astype(np.float32),
        v_A_trained=np.stack(pq_rows["v_A_trained"]).astype(np.float32),
        row_context_index=np.asarray(pq_ctx_idx, dtype=np.int64),  # original ctx index per row
        row_question_index=np.asarray(pq_q_idx, dtype=np.int64),
        context_ids=np.asarray(ctx_ids, dtype=object),  # full-length (indexed by original ci)
        families=np.asarray(families, dtype=object),  # full-length (indexed by original ci)
        headline_layer=np.asarray(HEADLINE_LAYER),
        behavior=np.asarray(behavior),
        substrate=np.asarray(substrate),
        git_sha=np.asarray(_git_sha()),
    )
    if args.upload:
        _hf_upload_file(
            pq_path,
            f"{EXPERIMENT_NAME}/reduced/{behavior}/{substrate}/per_question_L{HEADLINE_LAYER}.npz",
        )

    # atomic completion sentinel (resume-skip predicate)
    tmp_s = sentinel.with_suffix(f".{os.getpid()}.tmp")
    tmp_s.write_text(
        json.dumps(
            {
                "behavior": behavior,
                "substrate": substrate,
                "n_contexts": len(kept),
                "n_questions": len(questions),
                "n_unreduced_cells": n_cells_done,
                "n_empty_R": n_empty,
                "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            },
            indent=2,
        )
    )
    os.replace(tmp_s, sentinel)

    metrics = {
        "behavior": behavior,
        "substrate": substrate,
        "n_contexts": len(kept),
        "n_questions": len(questions),
        "n_unreduced_cells": n_cells_done,
        "n_empty_R": n_empty,
        "mean_cell_bytes": (float(np.mean(cell_bytes)) if cell_bytes else 0.0),
        "reduced_path": str(reduced_path),
        "df_free_gib_workspace": _df_free_gib(),
    }
    logger.info(
        "[phase=extract] cell %s/%s DONE: %d contexts, %d unreduced cells, mean %.1f MB/cell",
        behavior,
        substrate,
        len(kept),
        n_cells_done,
        metrics["mean_cell_bytes"] / 2**20,
    )
    return metrics


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Issue #813 — one (behavior, substrate) map-fit extraction cell"
    )
    ap.add_argument("--behavior", required=True, choices=list(BEHAVIORS))
    ap.add_argument("--substrate", required=True, choices=list(SUBSTRATES))
    ap.add_argument("--out-root", type=Path, default=PROJECT_ROOT / "eval_results/issue_813")
    ap.add_argument(
        "--gpu-id", type=int, default=0, help="physical GPU (CVD-pinned by the launcher)"
    )
    ap.add_argument("--cpu-only", action="store_true", help="CPU smoke (HF greedy R, no vLLM)")
    ap.add_argument("--upload", action="store_true", help="stream unreduced+reduced .npz to HF")
    ap.add_argument("--force", action="store_true", help="ignore the resume-skip sentinel")
    ap.add_argument(
        "--gate-only",
        action="store_true",
        help=(
            "one-cell footprint/wall GATE: extract + measure per-cell bytes ONLY, write the "
            "metrics JSON, and RETURN before any reduced-summary / per-question / .done "
            "sentinel write. Writes NOTHING into --out-root's reduced/ tree, so it cannot "
            "corrupt the production sweep (B1). Bypasses the <4-contexts fit guard."
        ),
    )
    ap.add_argument("--max-contexts", type=int, default=None, help="smoke: cap battery contexts")
    ap.add_argument(
        "--max-questions", type=int, default=None, help="smoke: cap substrate questions"
    )
    ap.add_argument(
        "--metrics-out", type=Path, default=None, help="write the phase-1 metrics JSON here"
    )
    return ap


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = build_parser().parse_args()
    metrics = run_cell(args)
    if args.metrics_out is not None:
        args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
        args.metrics_out.write_text(json.dumps(metrics, indent=2, default=float))
    # NO [phase=done] here — this is a per-cell SUBPROCESS whose stdout inherits the
    # dispatcher's; the poller reserves [phase=done] for the ONE terminal line in the
    # main dispatcher log (issue813_dispatch.sh), so a per-cell echo of it would trip
    # the #545 false-`done` while the sweep is still alive.
    logger.info(
        "run_cell %s/%s complete; metrics: %s",
        args.behavior,
        args.substrate,
        json.dumps(metrics, default=float),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
