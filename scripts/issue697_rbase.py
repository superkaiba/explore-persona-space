"""Issue #697 — R_base pre-sweep cache (plan §4.4 + §4.5): vLLM-batched greedy.

``R_base = base.generate(T(persona, q))`` depends ONLY on (persona, question) —
NOT on (behavior-adapter, cid, seed) — so across all 64 cells there are 280
distinct R_base generations, not 64x280. This pre-sweep ``rbase_prep`` phase
generates them ONCE with vLLM batched ``LLM.generate()`` (CLAUDE.md "Use vLLM for
generation") and caches each to ``eval_results/issue_697/r_base_cache/<persona>_<qi>.json``
(token-id list + decoded + manifest), then uploads the cache to the HF data repo.
The cell's first pass reads the cache (HF-path-exists → local-file → inline-HF
fallback) instead of generating R_base per cell, ~halving the per-cell gen count.

The marker-strip class is NOT baked into the cache — the cache holds the RAW base
greedy ids (+ decoded); the marker arm applies ``_strip_trailing_marker_and_eos``
at read time (the same as today), so one cache serves both strip classes.

**vLLM gotchas threaded (all load-bearing):**
  - ``VLLM_WORKER_MULTIPROC_METHOD=spawn`` set at module top BEFORE any
    ``import vllm`` — this script calls ``AutoTokenizer.from_pretrained`` before
    ``LLM()``, the exact #628 fork-poison signature (EngineCore dies 1-4s after
    init under the default ``fork``).
  - ``use_tqdm=False`` on every ``LLM.generate()`` — #613 ZeroDivisionError when a
    batch finishes faster than tqdm's elapsed clock.
  - chunked generation (``EPM_VLLM_GREEDY_CHUNK_SIZE`` default 500) with a per-chunk
    INFO log — #664 large-batch EngineCore deadlock + poller-liveness.

**Parity assert (smoke-gate, §4.4):** the vLLM-cached R_base for a sample of
(persona, q) pairs must match the HF ``_greedy_generate_ids`` output (first ≥32
tokens identical, whitespace-normalized strings equal). On MATERIAL divergence the
whole cache regenerates via HF greedy (the caching lever survives without vLLM);
the patch READS stay HF teacher-forced regardless (vLLM bypasses the residual
hooks). HF parity uses a SEPARATE subprocess-free path AFTER the vLLM engine is
torn down (HF↔vLLM coexistence safety) — see ``_hf_parity_check``.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import logging
import os
import subprocess
import time
from pathlib import Path

# vLLM reads VLLM_WORKER_MULTIPROC_METHOD at IMPORT time; set spawn BEFORE the
# (transitive) vllm import so the EngineCore subprocess does not inherit a
# fork-poisoned CUDA-adjacent parent state (#628). setdefault: a shell-set value wins.
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

logger = logging.getLogger("issue697_rbase")

QWEN_ID = "Qwen/Qwen2.5-7B-Instruct"
HF_DATA_REPO = "superkaiba1/explore-persona-space-data"
HF_RBASE_PREFIX = "issue697_cv_patch/r_base_cache"
# #664: chunk large vLLM batches so a single oversized generate() cannot deadlock
# the v1 EngineCore worker; env-overridable so ops can tune without a code change.
VLLM_CHUNK_SIZE = int(os.environ.get("EPM_VLLM_GREEDY_CHUNK_SIZE", "500"))
# Parity-check sample size + the first-N-token agreement window (§4.4).
PARITY_N_PAIRS = 5
PARITY_PREFIX_TOKENS = 32


def _git_commit() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def rbase_cache_path(out_dir: Path, persona: str, qi: int) -> Path:
    """Canonical per-(persona, qi) R_base cache file path."""
    return out_dir / f"{persona}_{qi}.json"


def _build_prompts(tokenizer, personas: dict, questions: list[str]) -> list[dict]:
    """The (persona, qi, prompt_text) list over the fixed panel (14x20 = 280)."""
    from explore_persona_space.analysis.activation_shift import _build_chatml_prompt

    rows: list[dict] = []
    for p_name, p_prompt in personas.items():
        for qi, q in enumerate(questions):
            rows.append(
                {
                    "persona": p_name,
                    "q_idx": qi,
                    "question": q,
                    "prompt_text": _build_chatml_prompt(tokenizer, p_prompt, q),
                }
            )
    return rows


def _vllm_generate(llm, tokenizer, prompts: list[str], max_new_tokens: int) -> list[list[int]]:
    """Chunked greedy vLLM generation; returns generated TOKEN IDS per prompt.

    Chunked at ``VLLM_CHUNK_SIZE`` with a per-chunk INFO log (#664 deadlock
    prevention + poller liveness); ``use_tqdm=False`` (#613 ZeroDivisionError).
    Order is preserved. Returns the GENERATED ids (prompt stripped) per prompt —
    re-encoding from text would not round-trip, so we read ``out.token_ids``.
    """
    from vllm import SamplingParams

    sp = SamplingParams(temperature=0.0, max_tokens=max_new_tokens)
    out_ids: list[list[int]] = []
    for start in range(0, len(prompts), VLLM_CHUNK_SIZE):
        chunk = prompts[start : start + VLLM_CHUNK_SIZE]
        logger.info(
            "[vllm-chunk] rbase chunk %d-%d / %d",
            start,
            start + len(chunk),
            len(prompts),
        )
        results = llm.generate(chunk, sp, use_tqdm=False)
        for r in results:
            out_ids.append(list(r.outputs[0].token_ids))
    return out_ids


def _reap_vllm_engine(llm) -> None:
    """Reap the vLLM v1 EngineCore worker so the HF parity load has free HBM.

    vLLM v1's EngineCore runs in a SEPARATE subprocess; the bare ``del llm`` triad
    does NOT reap it synchronously (the KV cache stays pinned), so a subsequent HF
    model load can OOM. Every access getattr-guarded so it NO-OPs on a differing
    surface / the CPU path (mirrors ``representation_shift._reap_vllm_engine``,
    gotchas.md vLLM teardown).
    """
    import gc

    import torch

    try:
        engine = getattr(llm, "llm_engine", None)
        core = getattr(engine, "engine_core", None)
        if core is not None and hasattr(core, "shutdown"):
            core.shutdown()
        else:  # v0 fallback
            ex = getattr(engine, "model_executor", None)
            if ex is not None and hasattr(ex, "shutdown"):
                ex.shutdown()
    except Exception as e:  # teardown best-effort; never mask the result
        logger.warning("vLLM engine shutdown raised (continuing): %r", e)
    try:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
    except Exception:
        pass
    del llm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    time.sleep(1.0)  # subprocess teardown is async


def _hf_parity_check(
    base_model_id: str, rows: list[dict], vllm_ids: list[list[int]], tokenizer, max_new_tokens: int
) -> bool:
    """True iff vLLM greedy matches HF greedy on a sample (§4.4 parity gate).

    Loads the HF base model (AFTER the vLLM engine is reaped — coexistence safety),
    greedy-generates a deterministic sample of ``PARITY_N_PAIRS`` rows, and checks
    the first ``PARITY_PREFIX_TOKENS`` generated ids match OR the
    whitespace-normalized decoded strings are equal. Returns False on MATERIAL
    divergence (the caller then falls back to an HF-generated cache).
    """
    import torch
    from transformers import AutoModelForCausalLM

    from explore_persona_space.analysis.activation_shift import _greedy_generate_ids

    if not rows:
        return True
    # Deterministic evenly-spaced sample across the panel.
    step = max(1, len(rows) // PARITY_N_PAIRS)
    sample_idx = list(range(0, len(rows), step))[:PARITY_N_PAIRS]
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    model.eval()
    mismatches = 0
    for i in sample_idx:
        hf_ids = _greedy_generate_ids(model, tokenizer, rows[i]["prompt_text"], max_new_tokens)
        hf_list = hf_ids.tolist()
        vl_list = vllm_ids[i]
        prefix_match = hf_list[:PARITY_PREFIX_TOKENS] == vl_list[:PARITY_PREFIX_TOKENS]
        str_match = (
            tokenizer.decode(hf_list, skip_special_tokens=True).split()
            == tokenizer.decode(vl_list, skip_special_tokens=True).split()
        )
        ok = prefix_match or str_match
        logger.info(
            "[parity] row %d persona=%s q=%d prefix_match=%s str_match=%s",
            i,
            rows[i]["persona"],
            rows[i]["q_idx"],
            prefix_match,
            str_match,
        )
        if not ok:
            mismatches += 1
    del model
    import gc

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    # Material divergence = ANY sampled pair disagreeing on both prefix AND string.
    passed = mismatches == 0
    logger.info(
        "[parity] %d/%d sampled pairs matched (passed=%s)",
        len(sample_idx) - mismatches,
        len(sample_idx),
        passed,
    )
    return passed


def _hf_generate_all(base_model_id: str, rows: list[dict], tokenizer, max_new_tokens: int):
    """HF-greedy fallback: generate every R_base via HF (the vLLM-parity-miss path)."""
    import torch
    from transformers import AutoModelForCausalLM

    from explore_persona_space.analysis.activation_shift import _greedy_generate_ids

    model = AutoModelForCausalLM.from_pretrained(
        base_model_id, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    model.eval()
    out_ids = []
    for r in rows:
        ids = _greedy_generate_ids(model, tokenizer, r["prompt_text"], max_new_tokens)
        out_ids.append(ids.tolist())
    del model
    import gc

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out_ids


def run_rbase_prep(args) -> int:
    """Generate + cache + upload R_base over the fixed panel (plan §4.4/§4.5)."""
    from transformers import AutoTokenizer

    from explore_persona_space.experiments.issue_651 import (
        build_panel_personas,
        build_panel_questions,
    )

    print("[phase=rbase_prep]", flush=True)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model_id, trust_remote_code=True)
    personas = build_panel_personas()
    questions = build_panel_questions()
    rows = _build_prompts(tokenizer, personas, questions)
    logger.info(
        "rbase_prep: %d panel prompts (%d personas x %d q)",
        len(rows),
        len(personas),
        len(questions),
    )

    backend = "vllm"
    gen_ids: list[list[int]]
    if args.cpu_only:
        # CPU smoke: tiny HF model, no vLLM (vLLM needs a GPU). Tiny slice.
        gen_ids = _hf_generate_all(args.base_model_id, rows, tokenizer, args.max_new_tokens)
        backend = "hf-cpu-smoke"
    else:
        from vllm import LLM

        llm = LLM(
            model=args.base_model_id,
            dtype="bfloat16",
            gpu_memory_utilization=0.5,  # leave HBM for the HF parity load (coexistence)
            trust_remote_code=True,
            max_model_len=args.max_model_len,
        )
        gen_ids = _vllm_generate(
            llm, tokenizer, [r["prompt_text"] for r in rows], args.max_new_tokens
        )
        _reap_vllm_engine(llm)  # free the EngineCore HBM before the HF parity load
        # Parity gate: vLLM vs HF greedy. On material divergence, regen via HF.
        if not args.skip_parity and not _hf_parity_check(
            args.base_model_id, rows, gen_ids, tokenizer, args.max_new_tokens
        ):
            logger.warning(
                "[parity] vLLM/HF greedy MATERIALLY diverge -> regenerating R_base via HF"
            )
            gen_ids = _hf_generate_all(args.base_model_id, rows, tokenizer, args.max_new_tokens)
            backend = "hf-parity-fallback"

    manifest = {
        "issue": 697,
        "phase": "rbase_prep",
        "base_model_id": args.base_model_id,
        "max_new_tokens": args.max_new_tokens,
        "backend": backend,
        "vllm_chunk_size": VLLM_CHUNK_SIZE,
        "git_commit": _git_commit(),
        "env_versions": {
            pkg: importlib.metadata.version(pkg)
            for pkg in ("torch", "transformers", "vllm")
            if _safe_version(pkg)
        },
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    written: list[Path] = []
    for r, ids in zip(rows, gen_ids, strict=True):
        path = rbase_cache_path(out_dir, r["persona"], r["q_idx"])
        path.write_text(
            json.dumps(
                {
                    "persona": r["persona"],
                    "q_idx": r["q_idx"],
                    "question": r["question"],
                    "r_base_token_ids": ids,
                    "r_base_decoded": tokenizer.decode(ids, skip_special_tokens=True),
                    "manifest": manifest,
                },
                indent=2,
            )
        )
        written.append(path)
    logger.info(
        "rbase_prep: wrote %d cache files to %s (backend=%s)", len(written), out_dir, backend
    )

    if args.upload:
        _upload_rbase_cache(written)
    print("[phase=rbase_prep_done]", flush=True)
    return 0


def _safe_version(pkg: str) -> bool:
    try:
        importlib.metadata.version(pkg)
        return True
    except Exception:
        return False


def _upload_rbase_cache(paths: list[Path]) -> None:
    """Upload the R_base cache to the HF data repo (ONE batched commit, fail-loud)."""
    from huggingface_hub import CommitOperationAdd, HfApi

    api = HfApi()
    ops = [
        CommitOperationAdd(path_in_repo=f"{HF_RBASE_PREFIX}/{p.name}", path_or_fileobj=str(p))
        for p in paths
    ]
    api.create_commit(
        repo_id=HF_DATA_REPO,
        repo_type="dataset",
        operations=ops,
        commit_message=f"issue697: R_base cache ({len(ops)} files)",
    )
    logger.info(
        "uploaded %d R_base cache files to %s/%s", len(paths), HF_DATA_REPO, HF_RBASE_PREFIX
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--base-model-id", default=QWEN_ID)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument(
        "--cpu-only", action="store_true", help="HF tiny-model CPU smoke (no vLLM)."
    )
    parser.add_argument("--upload", action="store_true", help="Upload the cache to HF (fail-loud).")
    parser.add_argument(
        "--skip-parity", action="store_true", help="Skip the vLLM/HF parity gate (CPU smoke)."
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s :: %(message)s"
    )
    from dotenv import load_dotenv

    load_dotenv()
    return run_rbase_prep(args)


if __name__ == "__main__":
    raise SystemExit(main())
