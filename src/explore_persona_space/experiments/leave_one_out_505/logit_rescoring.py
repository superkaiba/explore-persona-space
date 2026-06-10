# ruff: noqa: RUF001, RUF002, RUF003  # em-dash + minus sign + Qwen marker " ※" + Greek Δ intentional
"""Task #505 follow-up ``logit-space-rescoring`` — four-float logit capture at frac 1.0.

The original #505 sweep (and the ``expanded-predictor-reanalysis`` follow-up)
stored only resolved log-probs (``g_logp`` / ``b_logp`` / ``delta_g``) per
probe. Logits are UNRECOVERABLE from stored log-probs post-hoc (the per-slot
``logZ`` is unknown — ``.claude/rules/marker-leakage-measurement.md``
§ Storage contract; incident #530), and #531 showed the off-ceiling shortcut
``Δlog P ≈ Δz`` can fail (logZ drift ~2.7 nats). This module regenerates
on-policy responses through the 24 published frac-1.0 adapters and captures
the full four-float contract per slot per model side:

    ``log P(marker)``, ``z_marker``, ``z_eos`` (id 151645), ``logZ = logsumexp(z)``

SCOPE CAVEAT (named in the followup spec): only the FINAL (frac-1.0, step-75)
adapters survive on HF; the frac-0.50 headline-slice checkpoints were
pod-local and are gone. Everything here therefore runs at the frac-1.0 slice,
paired against the stored frac-1.0 trajectory values for the faithfulness
anchor — NOT row-by-row against the frac-0.50 headline frame.

Phase layout (driven by ``dispatch_logit_rescoring``):
  A. GENERATION (vLLM): per adapter, greedy on-policy R for the 52 held-out
     bystanders + source × 10 eval questions, generation params IDENTICAL to
     the original eval (this module imports and calls the #472 rig's
     ``_generate_on_policy_R`` + ``score_logp_for_R`` rather than re-deriving
     parameters). The vLLM DV-A rescore (trained + base log P at the slot)
     rides along so Phase C has an engine-matched faithfulness anchor.
  B. LOGIT CAPTURE (HF forward passes — NEVER vLLM, which returns
     post-softmax log-probs only): batched teacher-forced forwards over
     ``full_ids[:-1]`` (the EXACT slot context the vLLM path scores), PEFT
     adapter enabled (trained side) then disabled (base side) on a shared
     base model.
  C. FAITHFULNESS (#531 recipe): recomputed log-probs vs the stored frac-1.0
     trajectory values, per-cell MAE + Spearman, plus an HF-vs-vLLM same-R
     cross-engine read.

Gauge validity: the trained − base logit readout is comparable across cells
ONLY because the #505 LoRAs leave ``lm_head`` / ``embed_tokens`` untouched —
``assert_gauge_free_adapter_config`` is launch-blocking for every adapter.
"""

from __future__ import annotations

import json
import logging
import os
import socket
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from explore_persona_space.eval.marker_logprob import assert_gauge_free_adapter_config
from explore_persona_space.experiments.contrastive_neg_geometry_472 import (
    BASE_MODEL,
    EXPECTED_MARKER_TOKEN_ID,
    HF_DATA_REPO,
    HF_MODEL_REPO,
    MARKER_TEXT,
    SOURCE_PERSONA,
)
from explore_persona_space.experiments.contrastive_neg_geometry_472.eval_one_cell import (
    build_full_ids,
    score_logp_for_R,
)
from explore_persona_space.experiments.contrastive_neg_geometry_472.eval_trajectory import (
    EXPECTED_POST_R_EOS_ID,
    POST_R_EOS_TOKEN,
    _generate_on_policy_R,
    _git_sha,
    _slot_stats_from_raw_logits,
)
from explore_persona_space.experiments.leave_one_out_505 import (
    CELL_SPECS,
    HF_ADAPTER_PATH_PREFIX,
    HF_DATA_PREFIX,
    MAX_MODEL_LEN,
    MAX_NEW_TOKENS_GEN,
    SEEDS,
)

log = logging.getLogger("issue_505.logit_rescoring")

SCHEMA_VERSION = "i505_logit_rescoring_v1"
RESCORING_LABEL = "logit-space-rescoring"
TARGET_FRAC = 1.0
HF_RESCORING_PREFIX = f"{HF_DATA_PREFIX}/logit_rescoring"
RAW_COMPLETIONS_HF_PREFIX = f"{HF_RESCORING_PREFIX}/raw_completions"
# Files every published adapter dir must carry for this rig to run.
ADAPTER_REQUIRED_FILES = ("adapter_config.json", "adapter_model.safetensors")
# R-collapse threshold — mirrors eval_one_cell.R_COLLAPSE_MARKER_FRACTION.
R_COLLAPSE_MARKER_FRACTION = 0.5


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def repro_block(inputs: dict[str, str]) -> dict[str, Any]:
    """Reproducibility metadata for result JSONs (git sha, versions, host, inputs)."""
    from importlib.metadata import PackageNotFoundError, version

    versions: dict[str, str] = {}
    for pkg in ("torch", "transformers", "peft", "vllm", "huggingface_hub"):
        try:
            versions[pkg] = version(pkg)
        except PackageNotFoundError:  # pragma: no cover - all are project deps
            versions[pkg] = "absent"
    return {
        "git_commit": _git_sha(),
        "timestamp_utc": _utc_now(),
        "hostname": socket.gethostname(),
        "versions": versions,
        "inputs": inputs,
    }


def _with_retries(fn, what: str, attempts: int = 3, delays: tuple[int, ...] = (30, 60, 120)):
    """Run ``fn()`` with retry-with-backoff; fail loud after the last attempt.

    Hardens HF Hub calls against transient network blips (the
    dispatcher-silent-death class): each failure is logged with the full
    exception, then retried after the next backoff delay.
    """
    last: Exception | None = None
    for attempt in range(attempts):
        try:
            return fn()
        except Exception as e:
            last = e
            if attempt == attempts - 1:
                break
            delay = delays[min(attempt, len(delays) - 1)]
            log.exception(
                "[retry] %s failed (attempt %d/%d); retrying in %ds",
                what,
                attempt + 1,
                attempts,
                delay,
            )
            time.sleep(delay)
    raise RuntimeError(f"{what} failed after {attempts} attempts") from last


# ── Launch-blocking asserts ──────────────────────────────────────────────────


def assert_real_marker_tokenization(real_base_model: str = BASE_MODEL) -> None:
    """Assert the CANONICAL (Qwen-2.5-7B-Instruct) marker + EOS token ids.

    Always runs against the REAL tokenizer (cheap CPU download) so the smoke
    documents the expected-id check even when the runtime model is a tiny
    stand-in whose vocab may differ (the runtime ids are resolved separately
    by :func:`resolve_runtime_token_ids`).
    """
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(real_base_model, trust_remote_code=True)
    encoded = tok.encode(MARKER_TEXT, add_special_tokens=False)
    if encoded != [EXPECTED_MARKER_TOKEN_ID]:
        raise RuntimeError(
            f"[invariant] real tokenizer ({real_base_model}) encodes {MARKER_TEXT!r} to "
            f"{encoded}, expected [{EXPECTED_MARKER_TOKEN_ID}]. Tokenizer drift — aborting."
        )
    eos_id = tok.convert_tokens_to_ids(POST_R_EOS_TOKEN)
    if eos_id != EXPECTED_POST_R_EOS_ID:
        raise RuntimeError(
            f"[invariant] real tokenizer resolves {POST_R_EOS_TOKEN!r} to {eos_id}, expected "
            f"{EXPECTED_POST_R_EOS_ID} — the z_eos readout would be wrong. Aborting."
        )
    log.info(
        "[phase=marker_check] real tokenizer OK: %r → [%d], %s → %d",
        MARKER_TEXT,
        EXPECTED_MARKER_TOKEN_ID,
        POST_R_EOS_TOKEN,
        EXPECTED_POST_R_EOS_ID,
    )


def resolve_runtime_token_ids(tokenizer, *, production: bool) -> tuple[int, int]:
    """Resolve (marker_id, eos_id) on the RUNTIME tokenizer.

    In production (runtime model == the real base) the ids MUST equal the
    canonical (83399, 151645). In smoke mode (tiny stand-in model) the ids are
    resolved dynamically — the marker must still be a SINGLE token (raw logits
    do not sum across BPE pieces).
    """
    marker_ids = tokenizer.encode(MARKER_TEXT, add_special_tokens=False)
    if len(marker_ids) != 1:
        raise RuntimeError(
            f"runtime tokenizer encodes {MARKER_TEXT!r} to {marker_ids} — a single-token "
            "marker is required for the raw-logit readout."
        )
    eos_id = tokenizer.convert_tokens_to_ids(POST_R_EOS_TOKEN)
    if eos_id is None or eos_id == tokenizer.unk_token_id:
        raise RuntimeError(f"runtime tokenizer cannot resolve {POST_R_EOS_TOKEN!r} (got {eos_id}).")
    if production and (marker_ids[0], eos_id) != (EXPECTED_MARKER_TOKEN_ID, EXPECTED_POST_R_EOS_ID):
        raise RuntimeError(
            f"production runtime tokenizer ids drifted: marker={marker_ids[0]} "
            f"(expected {EXPECTED_MARKER_TOKEN_ID}), eos={eos_id} "
            f"(expected {EXPECTED_POST_R_EOS_ID})."
        )
    if not production and marker_ids[0] != EXPECTED_MARKER_TOKEN_ID:
        log.warning(
            "[smoke] runtime marker id %d differs from the canonical %d (stand-in tokenizer); "
            "scoring uses the runtime id.",
            marker_ids[0],
            EXPECTED_MARKER_TOKEN_ID,
        )
    return marker_ids[0], eos_id


def assert_generation_params_match_original(*, max_new_tokens: int, max_model_len: int) -> None:
    """Launch-blocking: production gen params must equal the original eval's constants.

    Asserts against the IMPORTED ``leave_one_out_505`` constants (the same
    objects ``eval_trajectory_505`` resolves), never copy-pasted numbers.
    """
    if max_new_tokens != MAX_NEW_TOKENS_GEN:
        raise RuntimeError(
            f"max_new_tokens={max_new_tokens} != leave_one_out_505.MAX_NEW_TOKENS_GEN="
            f"{MAX_NEW_TOKENS_GEN}; generation would not match the original eval."
        )
    if max_model_len != MAX_MODEL_LEN:
        raise RuntimeError(
            f"max_model_len={max_model_len} != leave_one_out_505.MAX_MODEL_LEN={MAX_MODEL_LEN}; "
            "generation would not match the original eval."
        )


# ── Adapter resolution + download (HF model repo) ───────────────────────────


def expected_adapter_cells(
    cells_cap: int | None = None, seeds_cap: int | None = None
) -> list[tuple[str, int]]:
    """The (cell_slug, seed) grid, ordered like the original sweep dispatcher."""
    specs = CELL_SPECS[: cells_cap or len(CELL_SPECS)]
    seeds = SEEDS[: seeds_cap or len(SEEDS)]
    return [(spec[0], seed) for spec in specs for seed in seeds]


def resolve_adapter_repo_dirs(
    cells: list[tuple[str, int]], *, repo_id: str = HF_MODEL_REPO
) -> tuple[dict[tuple[str, int], str], str]:
    """Verify every requested adapter dir exists on the hub with required files.

    Uses ``list_repo_files`` (Python Hub API — NEVER the ``hf`` CLI, whose
    missing ``api`` subcommand reads as a false "0 files") and pins the repo's
    current main revision so all per-file downloads come from one snapshot.

    Returns ``({(slug, seed): repo_dir_path}, revision_sha)``. The error
    branches are split (genuine-missing vs file-incomplete) so a downloader
    bug never reads as a misleading "re-train" instruction.
    """
    from huggingface_hub import HfApi

    api = HfApi(token=os.environ.get("HF_TOKEN"))
    info = _with_retries(lambda: api.repo_info(repo_id), f"repo_info({repo_id})")
    revision = info.sha
    files = set(
        _with_retries(
            lambda: api.list_repo_files(repo_id, revision=revision),
            f"list_repo_files({repo_id})",
        )
    )
    out: dict[tuple[str, int], str] = {}
    missing_dirs: list[str] = []
    incomplete: list[str] = []
    for slug, seed in cells:
        repo_dir = f"{HF_ADAPTER_PATH_PREFIX}/{slug}_seed{seed}"
        present = [f"{repo_dir}/{name}" in files for name in ADAPTER_REQUIRED_FILES]
        if not any(f.startswith(repo_dir + "/") for f in files):
            missing_dirs.append(repo_dir)
        elif not all(present):
            incomplete.append(
                f"{repo_dir} (missing "
                f"{[n for n, p in zip(ADAPTER_REQUIRED_FILES, present, strict=True) if not p]})"
            )
        else:
            out[(slug, seed)] = repo_dir
    if missing_dirs:
        raise FileNotFoundError(
            f"adapter dirs absent on {repo_id}@{revision[:12]}: {missing_dirs}. The frac-1.0 "
            "adapters were published by the #505 sweep — verify the repo path prefix "
            f"{HF_ADAPTER_PATH_PREFIX!r} before concluding anything needs retraining."
        )
    if incomplete:
        raise RuntimeError(
            f"adapter dirs present but file-incomplete on {repo_id}@{revision[:12]}: "
            f"{incomplete}. This is an upload defect, NOT a missing adapter — do not retrain."
        )
    return out, revision


def download_adapter(
    repo_dir: str, *, repo_id: str = HF_MODEL_REPO, revision: str | None = None
) -> Path:
    """Download one adapter's required files (per-file ``hf_hub_download``).

    Per-file downloads (not ``snapshot_download``) because the model repo is
    large enough to hit the truncated-``siblings`` pattern-filter bug. Both
    files land in the same snapshot dir; returns that directory.
    """
    from huggingface_hub import hf_hub_download

    token = os.environ.get("HF_TOKEN")
    local_paths = [
        _with_retries(
            lambda name=name: hf_hub_download(
                repo_id, f"{repo_dir}/{name}", revision=revision, token=token
            ),
            f"hf_hub_download({repo_dir}/{name})",
        )
        for name in ADAPTER_REQUIRED_FILES
    ]
    parents = {Path(p).parent for p in local_paths}
    assert len(parents) == 1, f"adapter files landed in different dirs: {parents}"
    return parents.pop()


def assert_adapters_gauge_free(adapter_dirs: dict[tuple[str, int], Path]) -> None:
    """Launch-blocking gauge assert over EVERY adapter config (before any GPU work)."""
    for (slug, seed), adapter_dir in adapter_dirs.items():
        cfg_path = adapter_dir / "adapter_config.json"
        if not cfg_path.exists():
            raise FileNotFoundError(f"adapter_config.json missing at {cfg_path} ({slug} s{seed})")
        assert_gauge_free_adapter_config(json.loads(cfg_path.read_text()), context=str(cfg_path))
    log.info(
        "[phase=gauge_check] %d adapter configs gauge-free (no lm_head/embed_tokens, "
        "modules_to_save empty)",
        len(adapter_dirs),
    )


# ── Phase A — on-policy generation (+ vLLM DV-A rescore) ────────────────────


def run_phase_a_vllm(
    llm,
    tokenizer,
    *,
    cell_slug: str,
    seed: int,
    adapter_dir: Path,
    lora_int_id: int,
    personas: dict[str, str],
    questions: list[str],
    max_new_tokens: int,
) -> dict:
    """Greedy on-policy R for the panel grid + engine-matched DV-A rescore.

    Reuses the #472 rig's ``_generate_on_policy_R`` (prompt construction +
    SamplingParams) and ``score_logp_for_R`` (slot construction + guards)
    VERBATIM, so generation params and slot logic are the original eval's by
    construction. ``lora_int_id`` MUST be unique per adapter within one
    engine — vLLM identifies adapters by int id, so a reused id would
    silently score a cached adapter.
    """
    from vllm.lora.request import LoRARequest

    label = f"{cell_slug}_seed{seed}_frac{TARGET_FRAC}"
    lora_req = LoRARequest(
        lora_name=f"i505lr_{label}", lora_int_id=lora_int_id, lora_path=str(adapter_dir)
    )
    completions = _generate_on_policy_R(
        llm, tokenizer, personas, questions, lora_req, max_new_tokens
    )
    g = score_logp_for_R(
        llm,
        tokenizer,
        r_by_persona_q=completions,
        eval_personas=personas,
        eval_questions=questions,
        cell_label=f"TRAINED/{label}",
        use_lora=True,
        lora_request=lora_req,
    )
    b = score_logp_for_R(
        llm,
        tokenizer,
        r_by_persona_q=completions,
        eval_personas=personas,
        eval_questions=questions,
        cell_label=f"BASE/{label}",
        use_lora=False,
    )
    vllm_rescore = {
        p: {
            q: {
                "g_logp": float(g[p][q]["logp"]),
                "b_logp": float(b[p][q]["logp"]),
                "delta_g": float(g[p][q]["logp"]) - float(b[p][q]["logp"]),
                "argmax_marker": bool(g[p][q]["argmax_marker"]),
                "n_marker_in_R": int(g[p][q]["n_marker_in_R"]),
                "r_collapsed": bool(g[p][q]["r_collapsed"]),
            }
            for q in questions
        }
        for p in personas
    }
    return {"completions": completions, "vllm_rescore": vllm_rescore}


def run_phase_a_hf(
    peft_model,
    tokenizer,
    *,
    personas: dict[str, str],
    questions: list[str],
    max_new_tokens: int,
    device: str,
    batch_size: int = 4,
) -> dict:
    """CPU-smoke generation backend: batched greedy HF ``generate``.

    Mirrors ``_generate_on_policy_R``'s prompt construction (chat template,
    ``add_generation_prompt=True``, persona via the system role, greedy to
    EOS). No vLLM rescore on this backend — the four-float HF capture in
    Phase B supplies the log-prob, and Phase C records the provenance.
    ``generate`` is padding-aware (it derives position ids from the attention
    mask), so batched left-pad generation is safe here.
    """
    import torch

    prompts: list[str] = []
    keys: list[tuple[str, str]] = []
    for persona, persona_prompt in personas.items():
        for q in questions:
            messages = [
                {"role": "system", "content": persona_prompt},
                {"role": "user", "content": q},
            ]
            prompts.append(
                tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            )
            keys.append((persona, q))

    pad_id = (
        tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    )
    prev_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    completions: dict[str, dict[str, str]] = {p: {} for p in personas}
    try:
        for start in range(0, len(prompts), batch_size):
            chunk = prompts[start : start + batch_size]
            enc = tokenizer(chunk, return_tensors="pt", padding=True, add_special_tokens=False)
            enc = {k: v.to(device) for k, v in enc.items()}
            with torch.no_grad():
                gen = peft_model.generate(
                    **enc,
                    do_sample=False,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=pad_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            new_tokens = gen[:, enc["input_ids"].shape[1] :]
            for row_i, (persona, q) in enumerate(keys[start : start + batch_size]):
                completions[persona][q] = tokenizer.decode(
                    new_tokens[row_i], skip_special_tokens=True
                )
    finally:
        tokenizer.padding_side = prev_side
    return {"completions": completions, "vllm_rescore": None}


# ── Phase B — four-float HF logit capture ───────────────────────────────────


def _batched_slot_raw_logits(
    model, context_ids_list: list[list[int]], *, device: str, batch_size: int, pad_id: int
) -> list:
    """Batched next-token raw logits at the slot after each context. Returns (V,) fp32 CPU tensors.

    Left-pads within length-sorted sub-batches and passes EXPLICIT
    ``position_ids = (attention_mask.cumsum(-1) − 1).clamp(min=0)`` so real
    tokens keep their natural 0..len−1 positions under left padding (the #502
    batched-rewrite incident: missing position_ids under left-pad silently
    diverged from the serial path). Output order matches the input order.
    """
    import torch

    order = sorted(range(len(context_ids_list)), key=lambda i: len(context_ids_list[i]))
    out: list[tuple[int, Any]] = []
    for start in range(0, len(order), batch_size):
        idxs = order[start : start + batch_size]
        chunk = [context_ids_list[i] for i in idxs]
        for cidx, cids in zip(idxs, chunk, strict=True):
            assert len(cids) > 0, f"context {cidx} tokenized to [] — refusing to score"
        max_len = max(len(c) for c in chunk)
        padded = [[pad_id] * (max_len - len(c)) + c for c in chunk]
        attn = [[0] * (max_len - len(c)) + [1] * len(c) for c in chunk]
        input_ids = torch.tensor(padded, dtype=torch.long, device=device)
        attention_mask = torch.tensor(attn, dtype=torch.long, device=device)
        position_ids = (attention_mask.cumsum(dim=1) - 1).clamp(min=0)
        with torch.no_grad():
            logits = model(
                input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids
            ).logits
        assert logits.ndim == 3 and logits.shape[0] == len(chunk), logits.shape
        for row_i, orig_i in enumerate(idxs):
            out.append((orig_i, logits[row_i, -1, :].float().cpu()))
        del logits
    out.sort(key=lambda t: t[0])
    return [t[1] for t in out]


def assert_batched_equals_serial(
    model,
    context_ids_list: list[list[int]],
    *,
    device: str,
    pad_id: int,
    marker_id: int,
    eos_id: int,
    cosine_floor: float = 0.999,
    float_tol: float = 1e-3,
) -> dict[str, float]:
    """Batched-rewrite equivalence guard: padded-batch slot logits == serial (B=1, no pad).

    Picks the shortest + median + longest contexts (so padding actually
    fires), runs them in ONE padded batch AND one-at-a-time unpadded, then
    asserts per-row cosine ≥ ``cosine_floor`` over the full (V,) vector and
    |Δ| ≤ ``float_tol`` on each of the four contract floats. Tolerances are
    caller-set per dtype (fp32 CPU: 1e-3; bf16 CUDA: looser).
    """
    import torch

    if len(context_ids_list) < 2:
        raise ValueError("equivalence guard needs ≥2 contexts so padding fires.")
    by_len = sorted(range(len(context_ids_list)), key=lambda i: len(context_ids_list[i]))
    pick = sorted({by_len[0], by_len[len(by_len) // 2], by_len[-1]})
    contexts = [context_ids_list[i] for i in pick]
    batched = _batched_slot_raw_logits(
        model, contexts, device=device, batch_size=len(contexts), pad_id=pad_id
    )
    worst = {"cosine": 1.0, "max_float_diff": 0.0}
    for cids, brow in zip(contexts, batched, strict=True):
        ids = torch.tensor([cids], dtype=torch.long, device=device)
        with torch.no_grad():
            serial = model(input_ids=ids).logits[0, -1, :].float().cpu()
        cos = float(
            torch.nn.functional.cosine_similarity(brow.unsqueeze(0), serial.unsqueeze(0)).item()
        )
        s_b = _slot_stats_from_raw_logits(brow, marker_id, eos_id)
        s_s = _slot_stats_from_raw_logits(serial, marker_id, eos_id)
        diffs = [abs(s_b[k] - s_s[k]) for k in ("z_marker", "z_eos", "logZ")]
        diffs.append(abs((s_b["z_marker"] - s_b["logZ"]) - (s_s["z_marker"] - s_s["logZ"])))
        worst["cosine"] = min(worst["cosine"], cos)
        worst["max_float_diff"] = max(worst["max_float_diff"], max(diffs))
        if cos < cosine_floor or max(diffs) > float_tol:
            raise AssertionError(
                f"batched != serial at the slot (len={len(cids)}): cosine={cos:.6f} "
                f"(floor {cosine_floor}), max four-float diff={max(diffs):.6g} "
                f"(tol {float_tol}). Padding/position_ids regression — refusing to score."
            )
    log.info(
        "[equivalence] batched==serial OK over %d picked contexts: min cosine=%.6f, "
        "max four-float diff=%.3g",
        len(contexts),
        worst["cosine"],
        worst["max_float_diff"],
    )
    return worst


def capture_slot_stats_for_cell(
    peft_model,
    tokenizer,
    *,
    completions: dict[str, dict[str, str]],
    personas: dict[str, str],
    questions: list[str],
    marker_id: int,
    eos_id: int,
    device: str,
    batch_size: int = 8,
    run_equivalence_guard: bool = False,
    equivalence_float_tol: float = 1e-3,
) -> dict[str, dict[str, dict[str, float | bool | int]]]:
    """Four-float capture per (persona, q) per side from HF forward passes.

    The slot context is ``full_ids[:-1]`` from the #472 ``build_full_ids``
    (which carries the off-by-one + train-equivalence guards), i.e. the EXACT
    prefix the vLLM ``prompt_logprobs`` path scores at the appended-marker
    slot. Trained side = adapter active; base side = the SAME shared model
    under ``disable_adapter()`` (PEFT hot-swap contract from the followup
    spec). Persists nothing — the caller owns checkpoint-per-phase writes.
    """
    import torch

    contexts: list[list[int]] = []
    keys: list[tuple[str, str, int, bool]] = []
    for persona, persona_prompt in personas.items():
        if persona not in completions:
            raise KeyError(f"completions missing persona {persona!r} — Phase A incomplete?")
        for q in questions:
            if q not in completions[persona]:
                raise KeyError(f"completions[{persona!r}] missing q {q!r} — Phase A incomplete?")
            r_text = completions[persona][q]
            full_ids, _p, r_len, slot, n_marker_in_R = build_full_ids(
                tokenizer, persona_prompt, q, r_text, MARKER_TEXT, marker_id, persona, q
            )
            r_frac = (n_marker_in_R / r_len) if r_len > 0 else 0.0
            r_collapsed = bool(n_marker_in_R > 0 and r_frac >= R_COLLAPSE_MARKER_FRACTION)
            contexts.append(full_ids[:slot])
            keys.append((persona, q, n_marker_in_R, r_collapsed))

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_id
    if run_equivalence_guard:
        assert_batched_equals_serial(
            peft_model,
            contexts,
            device=device,
            pad_id=pad_id,
            marker_id=marker_id,
            eos_id=eos_id,
            float_tol=equivalence_float_tol,
        )

    raw_g = _batched_slot_raw_logits(
        peft_model, contexts, device=device, batch_size=batch_size, pad_id=pad_id
    )
    with peft_model.disable_adapter():
        raw_b = _batched_slot_raw_logits(
            peft_model, contexts, device=device, batch_size=batch_size, pad_id=pad_id
        )

    # log P = z_marker − logZ identity vs an independent log_softmax read
    # (first row, both sides) — exact up to fp32 rounding.
    for raw in (raw_g[0], raw_b[0]):
        st = _slot_stats_from_raw_logits(raw, marker_id, eos_id)
        ls = float(torch.log_softmax(raw, dim=-1)[marker_id].item())
        ident = st["z_marker"] - st["logZ"]
        assert abs(ls - ident) < 1e-4, (
            f"log-softmax identity violated: log_softmax={ls} vs z_marker − logZ={ident}"
        )

    # #477-class mini-guard: the adapter toggle must actually change the
    # forward pass somewhere — identical trained/base logits on EVERY probe
    # means PEFT silently didn't apply the adapter.
    max_abs_dz = 0.0
    stats: dict[str, dict[str, dict[str, float | bool | int]]] = {p: {} for p in personas}
    for (persona, q, n_marker_in_R, r_collapsed), rg, rb in zip(keys, raw_g, raw_b, strict=True):
        sg = _slot_stats_from_raw_logits(rg, marker_id, eos_id)
        sb = _slot_stats_from_raw_logits(rb, marker_id, eos_id)
        logp_g = sg["z_marker"] - sg["logZ"]
        logp_b = sb["z_marker"] - sb["logZ"]
        max_abs_dz = max(max_abs_dz, abs(sg["z_marker"] - sb["z_marker"]))
        stats[persona][q] = {
            "logp_g": logp_g,
            "logp_b": logp_b,
            "z_marker_g": sg["z_marker"],
            "z_marker_b": sb["z_marker"],
            "z_eos_g": sg["z_eos"],
            "z_eos_b": sb["z_eos"],
            "logZ_g": sg["logZ"],
            "logZ_b": sb["logZ"],
            "argmax_marker_g": bool(int(rg.argmax().item()) == marker_id),
            "argmax_marker_b": bool(int(rb.argmax().item()) == marker_id),
            "n_marker_in_R": int(n_marker_in_R),
            "r_collapsed": bool(r_collapsed),
            "delta_logp": logp_g - logp_b,
            "delta_z_marker": sg["z_marker"] - sb["z_marker"],
            "delta_margin": (sg["z_marker"] - sg["z_eos"]) - (sb["z_marker"] - sb["z_eos"]),
            "delta_logz": sg["logZ"] - sb["logZ"],
        }
    if max_abs_dz <= 1e-9:
        raise RuntimeError(
            "trained and base z_marker are identical on EVERY probe — the PEFT adapter "
            "toggle did not change the forward pass (silent LoRA-not-applied, the #477 "
            "class). Refusing to write a false-null capture."
        )
    return stats


# ── Phase C — faithfulness vs the stored frac-1.0 trajectory ────────────────


def stored_records_at_frac(
    trajectory_path: Path, *, frac: float = TARGET_FRAC, source: str = SOURCE_PERSONA
) -> dict[str, dict[str, dict[str, float | bool]]]:
    """``{persona: {q: {"g_logp", "b_logp", "argmax_marker"}}}`` at one stored frac.

    Held-out bystanders come from ``held_out``; the source persona's per-q
    records come from the ``source_probes`` block (the 2026-06-05 schema —
    every #505 sweep trajectory carries it).
    """
    payload = json.loads(Path(trajectory_path).read_text())
    target_2, target_4 = f"{frac:.2f}", f"{frac:.4f}"

    def _frac_match(ckpt: dict) -> bool:
        raw = ckpt.get("frac")
        if isinstance(raw, str):
            return raw in (target_2, target_4)
        return isinstance(raw, (int, float)) and abs(float(raw) - frac) < 1e-4

    ckpt = next((c for c in payload["checkpoints"] if _frac_match(c)), None)
    if ckpt is None:
        raise KeyError(
            f"{trajectory_path} has no checkpoint at frac={frac!r}; "
            f"checkpoints: {[c.get('frac') for c in payload['checkpoints']]}"
        )
    out: dict[str, dict[str, dict[str, float | bool]]] = {}
    for persona, per_q in ckpt.get("held_out", {}).items():
        out[persona] = {
            q: {
                "g_logp": float(leaf["g_logp"]),
                "b_logp": float(leaf["b_logp"]),
                "argmax_marker": bool(leaf.get("argmax_marker", False)),
            }
            for q, leaf in per_q.items()
        }
    source_probes = ckpt.get("source_probes")
    if source_probes:
        out[source] = {
            q: {
                "g_logp": float(leaf["g_logp"]),
                "b_logp": float(leaf["b_logp"]),
                "argmax_marker": bool(leaf.get("argmax_marker", False)),
            }
            for q, leaf in source_probes.items()
        }
    else:
        log.warning(
            "%s frac=%s carries no source_probes block — faithfulness runs held-out only.",
            trajectory_path,
            frac,
        )
    return out


def _mae_spearman(new: list[float], stored: list[float]) -> dict[str, float | int]:
    import numpy as np
    from scipy.stats import spearmanr

    a, b = np.asarray(new, dtype=float), np.asarray(stored, dtype=float)
    assert a.shape == b.shape and a.ndim == 1, (a.shape, b.shape)
    res = spearmanr(a, b)
    return {
        "n": int(a.size),
        "mae": float(np.mean(np.abs(a - b))),
        "spearman_rho": float(res.statistic),
        "spearman_p": float(res.pvalue),
    }


def faithfulness_for_cell(
    *,
    cell_slug: str,
    seed: int,
    stored_trajectory_path: Path,
    phase_a_payload: dict,
    phase_b_stats: dict,
    personas: list[str],
    questions: list[str],
) -> dict:
    """#531-recipe faithfulness: recomputed log-probs vs stored frac-1.0 values.

    Three reads per side (trained ``g`` / base ``b``):
      - ``vllm_vs_stored``: engine-matched (same vLLM scoring path as the
        original sweep), new on-policy R — residual = R-regeneration variance.
        Absent when Phase A ran the HF smoke backend.
      - ``hf_vs_stored``: the four-float capture's log-prob vs stored.
      - ``hf_vs_vllm_same_R``: cross-engine read on the SAME new R (bf16 /
        kernel / batching differences only; record, no hard assert).
    """
    stored = stored_records_at_frac(stored_trajectory_path)
    pairs = [
        (p, q)
        for p in personas
        if p in stored
        for q in questions
        if q in stored[p] and q in phase_b_stats.get(p, {})
    ]
    if not pairs:
        raise RuntimeError(
            f"{cell_slug} seed {seed}: no (persona, q) overlap between stored trajectory "
            f"{stored_trajectory_path} and the new capture — wrong sweep dir or wrong panel?"
        )
    out: dict[str, Any] = {
        "cell": cell_slug,
        "seed": seed,
        "frac": TARGET_FRAC,
        "n_pairs": len(pairs),
        "stored_trajectory": str(stored_trajectory_path),
    }
    vllm = phase_a_payload.get("vllm_rescore")
    if vllm is not None:
        out["vllm_vs_stored"] = {
            "g": _mae_spearman(
                [vllm[p][q]["g_logp"] for p, q in pairs], [stored[p][q]["g_logp"] for p, q in pairs]
            ),
            "b": _mae_spearman(
                [vllm[p][q]["b_logp"] for p, q in pairs], [stored[p][q]["b_logp"] for p, q in pairs]
            ),
        }
        out["hf_vs_vllm_same_R"] = {
            "g": _mae_spearman(
                [phase_b_stats[p][q]["logp_g"] for p, q in pairs],
                [vllm[p][q]["g_logp"] for p, q in pairs],
            ),
            "b": _mae_spearman(
                [phase_b_stats[p][q]["logp_b"] for p, q in pairs],
                [vllm[p][q]["b_logp"] for p, q in pairs],
            ),
        }
    out["hf_vs_stored"] = {
        "g": _mae_spearman(
            [phase_b_stats[p][q]["logp_g"] for p, q in pairs],
            [stored[p][q]["g_logp"] for p, q in pairs],
        ),
        "b": _mae_spearman(
            [phase_b_stats[p][q]["logp_b"] for p, q in pairs],
            [stored[p][q]["b_logp"] for p, q in pairs],
        ),
    }
    return out


# ── Raw-completion upload (fail-loud, verified) ─────────────────────────────


def upload_raw_completions_file(
    local_path: Path, *, cell_slug: str, seed: int, no_upload: bool = False
) -> str:
    """Upload one cell's raw-completions JSON to the HF data repo, verified.

    Lands at ``{RAW_COMPLETIONS_HF_PREFIX}/{cell}_seed{S}.json``. Fail-loud:
    raises unless ``list_repo_files`` confirms the file landed. ``no_upload``
    is for local smokes only (the pod sweep MUST upload before any cleanup).
    """
    path_in_repo = f"{RAW_COMPLETIONS_HF_PREFIX}/{cell_slug}_seed{seed}.json"
    if no_upload:
        log.info("[upload] --no-upload: skipping %s", path_in_repo)
        return path_in_repo
    from huggingface_hub import HfApi

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN unset — cannot upload raw completions (required).")
    api = HfApi(token=token)
    _with_retries(
        lambda: api.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=path_in_repo,
            repo_id=HF_DATA_REPO,
            repo_type="dataset",
        ),
        f"upload_file({path_in_repo})",
    )
    files = _with_retries(
        lambda: api.list_repo_files(HF_DATA_REPO, repo_type="dataset"),
        f"list_repo_files({HF_DATA_REPO})",
    )
    if path_in_repo not in files:
        raise RuntimeError(
            f"upload verification failed: {path_in_repo} not visible on {HF_DATA_REPO} "
            "after upload_file returned."
        )
    log.info("[upload] OK %s -> %s/%s", local_path.name, HF_DATA_REPO, path_in_repo)
    return path_in_repo


# ── Smoke helper — throwaway adapter for the CPU stand-in model ─────────────


def make_throwaway_adapter(base_model: str, out_dir: Path, *, r: int = 2, seed: int = 0) -> Path:
    """Create a tiny rank-``r`` LoRA with PERTURBED B matrices for the CPU smoke.

    PEFT initializes ``lora_B`` to zeros (identity adapter), which would make
    the trained and base sides bit-identical and trip the silent-LoRA
    mini-guard as a false positive — so we add small Gaussian noise to every
    ``lora_B`` weight, making the adapter toggle observable.
    """
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM

    torch.manual_seed(seed)
    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.float32, trust_remote_code=True
    )
    cfg = LoraConfig(
        r=r, lora_alpha=2 * r, target_modules=["q_proj", "v_proj"], lora_dropout=0.0, bias="none"
    )
    peft_model = get_peft_model(model, cfg)
    with torch.no_grad():
        for name, param in peft_model.named_parameters():
            if "lora_B" in name:
                param.add_(torch.randn_like(param) * 0.02)
    out_dir.mkdir(parents=True, exist_ok=True)
    peft_model.save_pretrained(str(out_dir))
    del peft_model, model
    log.info("[smoke] throwaway rank-%d adapter at %s", r, out_dir)
    return out_dir
