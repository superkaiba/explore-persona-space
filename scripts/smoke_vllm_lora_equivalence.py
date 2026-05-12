#!/usr/bin/env python
"""Equivalence smoke for the R7 vLLM-LoRA eval path.

The R7 override (issue #344) replaced merged-checkpoint Hub artifacts with
raw PEFT adapters and the eval path switched from
``LLM(model=<merged>)`` to ``LLM(model=<base>, enable_lora=True)`` + per-cell
``LoRARequest(lora_local_path=<adapter>)``. Before launching the sweep on
pod-344 we need positive evidence that the new path is numerically
equivalent to the old path on a cell where we still have BOTH a merged
checkpoint (from R2) AND a raw adapter dir on disk.

The smoke loads ONE cell (default: ``i344_software_engineer_no_cot_FRESH_seed42``),
runs 64 prompts through both paths with temperature=0, and compares:

* **Argmax agreement** between the merged-path top-1 token and the LoRA-path
  top-1 token. Gate: ≥99% (allow 1 disagreement per 64 = 1.56% slack).
* **Top-1 logit MSE** between the two paths. Gate: <1e-3 (vLLM on-the-fly
  merge introduces small numerical drift from the offline-merged math —
  bf16 accumulation order differs).

PASS prints concrete numbers and exits 0. FAIL prints concrete numbers
plus a head-of-disagreements table and exits 1.

Usage
-----

::

    uv run python scripts/smoke_vllm_lora_equivalence.py

Optional flags:

* ``--cell <id>``: override default smoke cell.
* ``--n-prompts <N>``: override default 64 prompts.
* ``--max-tokens <N>``: override default 1-token comparison (top-1 logit).

Outputs ``smoke_vllm_lora_equivalence_<cell>.json`` for postmortem.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Belt-and-suspenders: ensure HF_HOME points at the workspace cache even
# when the SSH non-login shell skipped ~/.bashrc. bootstrap_pod.sh writes
# the export to ~/.bashrc, which `nohup uv run python` (non-login) does
# NOT source. Without this the Hub call resolves to /root/.cache and
# misses the pre-cached Qwen2.5-7B-Instruct snapshot under /workspace.
_workspace_cache = Path("/workspace/.cache/huggingface")
if _workspace_cache.exists() and not os.environ.get("HF_HOME"):
    os.environ["HF_HOME"] = str(_workspace_cache)

logger = logging.getLogger("smoke_vllm_lora_equivalence")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_CELL = "i344_software_engineer_no_cot_FRESH_seed42"
MODEL_DIR = Path("/workspace/explore-persona-space/models/issue_344")

# Argmax-agreement gate. 99% over 64 prompts allows exactly 1 disagreement
# (the spec's 1.56% slack). We also allow the smoke to PASS at exactly
# 64/64 of course; the bound is on the lower side.
ARGMAX_AGREEMENT_GATE = 0.99
# Mean-squared-error on the top-1 logit. vLLM on-the-fly LoRA merge runs
# the LoRA delta in bf16 and accumulates differently than the offline-merged
# checkpoint; 1e-3 leaves headroom over the typical observed ~1e-4.
LOGIT_MSE_GATE = 1e-3


def _install_compat_shims() -> None:
    """vLLM 0.11.0 + transformers 5.5.0 compat shims — kept in sync with
    ``scripts/run_issue186_eval.py:_install_compat_shims``.

    Two patches:

    1. ``PreTrainedTokenizerBase.all_special_tokens_extended`` (transformers
       5.5.0 dropped this attribute that vLLM still references).
    2. ``vllm.model_executor.model_loader.weight_utils.DisabledTqdm`` —
       vLLM 0.11.0's subclass appends ``disable=True`` to ``**kwargs``
       even when the caller already passed ``disable=``, raising
       ``TypeError: ... got multiple values for keyword argument 'disable'``.
       huggingface_hub's ``snapshot_download`` does pass ``disable=``, so
       any vLLM Hub-fetch path (e.g. ``LLM(model="Qwen/Qwen2.5-7B-Instruct")``)
       crashes on bare 0.11.0. Patch swaps in a subclass that pops the
       caller's ``disable`` before forwarding.
    """
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase

    if not hasattr(PreTrainedTokenizerBase, "all_special_tokens_extended"):
        PreTrainedTokenizerBase.all_special_tokens_extended = (
            PreTrainedTokenizerBase.all_special_tokens
        )

    import vllm.model_executor.model_loader.weight_utils as _wu

    if not getattr(_wu.DisabledTqdm, "_issue186_patched", False):

        class _PatchedDisabledTqdm(_wu.DisabledTqdm.__bases__[0]):
            _issue186_patched = True

            def __init__(self, *a, **kw):
                kw.pop("disable", None)
                super().__init__(*a, disable=True, **kw)

        _wu.DisabledTqdm = _PatchedDisabledTqdm


def _build_smoke_prompts(n: int) -> list[str]:
    """64 deterministic ARC-style prompts, no persona injection.

    Doesn't matter what the model says — we just need the same prompt
    fed to both paths to compare argmax + logits. Pulls from the ARC-C
    test set if present locally; otherwise falls back to a fixed
    arithmetic suite that exercises the tokenizer.
    """
    arc_path = PROJECT_ROOT / "raw" / "arc_challenge" / "test.jsonl"
    if arc_path.exists():
        prompts: list[str] = []
        with arc_path.open() as fh:
            for line in fh:
                if len(prompts) >= n:
                    break
                rec = json.loads(line)
                # Schema in this repo: `question` is a plain string. Older
                # ARC dumps put a {"stem": "..."} dict here; tolerate both.
                q = rec.get("question")
                if isinstance(q, dict):
                    q = q.get("stem", "")
                if isinstance(q, str) and q.strip():
                    prompts.append(q.strip())
        if len(prompts) >= n:
            return prompts[:n]

    # Fallback: stable arithmetic prompts. Deterministic.
    return [f"What is {i} + {i + 1}? Answer with a single number." for i in range(n)]


def _generate_with_logits(llm, sampling_params, prompts, lora_request=None):
    """Run llm.generate and extract (top-1 token id, top-1 logprob) per prompt."""
    kw = {"sampling_params": sampling_params}
    if lora_request is not None:
        kw["lora_request"] = lora_request
    outputs = llm.generate(prompts, **kw)
    top1_ids: list[int] = []
    top1_logprobs: list[float] = []
    top5_ids: list[list[int]] = []
    for o in outputs:
        gen = o.outputs[0]
        # gen.token_ids is a list of generated token ids; gen.logprobs is a
        # list of {token_id: Logprob} dicts (vLLM format). We're asking for
        # max_tokens=1, so each list has length 1.
        first_tok = gen.token_ids[0]
        first_logprobs = gen.logprobs[0] if gen.logprobs else {}
        # logprob of the actually-selected token.
        selected_lp = first_logprobs.get(first_tok)
        # `selected_lp` is a vLLM Logprob object with .logprob attribute,
        # or a bare float in older versions.
        lp = float(getattr(selected_lp, "logprob", selected_lp) or 0.0)
        # Top-5 ids by logprob (sorted desc).
        sorted_tokens = sorted(
            first_logprobs.items(),
            key=lambda kv: float(getattr(kv[1], "logprob", kv[1])),
            reverse=True,
        )
        top5 = [t for t, _ in sorted_tokens[:5]]
        top1_ids.append(int(first_tok))
        top1_logprobs.append(lp)
        top5_ids.append(top5)
    return top1_ids, top1_logprobs, top5_ids


def _run_merged_path(merged_dir: str, prompts: list[str], n_prompts: int):
    """Load merged checkpoint into vLLM (no LoRA) and generate."""
    from vllm import SamplingParams

    from explore_persona_space.eval.generation import cleanup_vllm, create_vllm_engine

    logger.info("Loading MERGED path: %s", merged_dir)
    llm = create_vllm_engine(
        merged_dir,
        gpu_memory_utilization=0.6,
        max_model_len=2048,
        seed=42,
    )
    try:
        sp = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=1, logprobs=20)
        t0 = time.time()
        ids, lps, top5 = _generate_with_logits(llm, sp, prompts[:n_prompts])
        elapsed = time.time() - t0
        logger.info("MERGED path done: %d prompts in %.1fs", n_prompts, elapsed)
        return ids, lps, top5, elapsed
    finally:
        cleanup_vllm(llm)


def _resolve_local_base(base_model: str) -> str:
    """Resolve an HF repo-id to a local snapshot path (return ``base_model``
    as-is if it's already a local directory).

    Same rationale as ``run_issue186_eval.py:_resolve_local_base_model`` —
    keeps vLLM off the ``snapshot_download`` codepath that triggers the
    tqdm-kwargs clash in vllm 0.11.0 (the ``DisabledTqdm.__init__`` patch
    in ``_install_compat_shims`` covers it, but bypassing is cleaner).
    """
    p = Path(base_model)
    if p.is_dir() and (p / "config.json").exists():
        return str(p)
    from huggingface_hub import snapshot_download

    logger.info("Snapshot-downloading base model %s (idempotent on cache)", base_model)
    return snapshot_download(repo_id=base_model)


def _run_lora_path(adapter_dir: str, cell_id: str, prompts: list[str], n_prompts: int):
    """Load BASE + enable_lora, pass adapter via LoRARequest, generate."""
    from vllm import SamplingParams
    from vllm.lora.request import LoRARequest

    from explore_persona_space.eval.generation import cleanup_vllm, create_vllm_engine

    adapter_cfg = json.loads((Path(adapter_dir) / "adapter_config.json").read_text())
    max_lora_rank = int(adapter_cfg.get("r", 16))

    local_base = _resolve_local_base(BASE_MODEL)
    logger.info(
        "Loading LORA path: base=%s (resolved -> %s) adapter=%s (rank=%d)",
        BASE_MODEL,
        local_base,
        adapter_dir,
        max_lora_rank,
    )
    llm = create_vllm_engine(
        local_base,
        gpu_memory_utilization=0.6,
        max_model_len=2048,
        seed=42,
        enable_lora=True,
        max_loras=1,
        max_lora_rank=max_lora_rank,
    )
    try:
        sp = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=1, logprobs=20)
        lora_int_id = (abs(hash(cell_id)) % (2**31 - 1)) + 1
        lora_request = LoRARequest(
            lora_name=cell_id,
            lora_int_id=lora_int_id,
            lora_local_path=adapter_dir,
        )
        t0 = time.time()
        ids, lps, top5 = _generate_with_logits(
            llm, sp, prompts[:n_prompts], lora_request=lora_request
        )
        elapsed = time.time() - t0
        logger.info("LORA path done: %d prompts in %.1fs", n_prompts, elapsed)
        return ids, lps, top5, elapsed
    finally:
        cleanup_vllm(llm)


def _compare(merged, lora) -> dict:
    """Compare argmax agreement and logit MSE between the two paths."""
    merged_ids, merged_lps, merged_top5, _ = merged
    lora_ids, lora_lps, lora_top5, _ = lora
    n = len(merged_ids)
    assert n == len(lora_ids), "prompt counts must match"

    # Argmax agreement.
    agreements = [m == l_ for m, l_ in zip(merged_ids, lora_ids, strict=True)]
    argmax_agree = sum(agreements) / n

    # Top-1 logit MSE.
    sq_diff = [(m - l_) ** 2 for m, l_ in zip(merged_lps, lora_lps, strict=True)]
    logit_mse = sum(sq_diff) / n

    # Top-5 set agreement (informational only — wider lens).
    top5_jaccard = []
    for m5, l5 in zip(merged_top5, lora_top5, strict=True):
        s_m, s_l = set(m5), set(l5)
        if not s_m and not s_l:
            top5_jaccard.append(1.0)
            continue
        top5_jaccard.append(len(s_m & s_l) / len(s_m | s_l))
    top5_jaccard_mean = sum(top5_jaccard) / n if top5_jaccard else 0.0

    # Head of disagreements for postmortem.
    disagreements = []
    for i, (m_id, l_id, m_lp, l_lp) in enumerate(
        zip(merged_ids, lora_ids, merged_lps, lora_lps, strict=True)
    ):
        if m_id != l_id:
            disagreements.append(
                {
                    "prompt_idx": i,
                    "merged_top1": m_id,
                    "lora_top1": l_id,
                    "merged_logprob": m_lp,
                    "lora_logprob": l_lp,
                }
            )
        if len(disagreements) >= 10:
            break

    return {
        "n_prompts": n,
        "argmax_agreement": argmax_agree,
        "logit_mse": logit_mse,
        "top5_jaccard_mean": top5_jaccard_mean,
        "disagreements_head": disagreements,
    }


def main() -> int:
    _install_compat_shims()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cell",
        default=DEFAULT_CELL,
        help=f"Cell id (default: {DEFAULT_CELL}). Expects "
        f"{{model_dir}}/{{cell}}_merged and {{cell}}_adapter to both exist on disk.",
    )
    parser.add_argument(
        "--model-dir",
        default=str(MODEL_DIR),
        help=f"Local model dir (default: {MODEL_DIR}).",
    )
    parser.add_argument("--n-prompts", type=int, default=64, help="N prompts (default: 64)")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    merged_dir = model_dir / f"{args.cell}_merged"
    adapter_dir = model_dir / f"{args.cell}_adapter"

    if not (merged_dir / "config.json").exists():
        logger.error(
            "Expected merged dir not found or empty: %s. Pick a cell that "
            "still has BOTH merged + adapter on disk (e.g. an R2 FRESH cell).",
            merged_dir,
        )
        return 1
    if not (adapter_dir / "adapter_config.json").exists():
        logger.error("Expected adapter dir not found or empty: %s", adapter_dir)
        return 1

    logger.info("=" * 70)
    logger.info("Equivalence smoke: cell=%s, n_prompts=%d", args.cell, args.n_prompts)
    logger.info("  merged_dir:  %s", merged_dir)
    logger.info("  adapter_dir: %s", adapter_dir)
    logger.info("=" * 70)

    prompts = _build_smoke_prompts(args.n_prompts)
    logger.info("Built %d prompts; first prompt: %s", len(prompts), prompts[0][:80])

    merged = _run_merged_path(str(merged_dir), prompts, args.n_prompts)
    lora = _run_lora_path(str(adapter_dir), args.cell, prompts, args.n_prompts)

    cmp = _compare(merged, lora)
    cmp["cell"] = args.cell
    cmp["merged_wall_sec"] = merged[3]
    cmp["lora_wall_sec"] = lora[3]
    cmp["gates"] = {
        "argmax_agreement_min": ARGMAX_AGREEMENT_GATE,
        "logit_mse_max": LOGIT_MSE_GATE,
    }

    out_path = PROJECT_ROOT / f"smoke_vllm_lora_equivalence_{args.cell}.json"
    out_path.write_text(json.dumps(cmp, indent=2))
    logger.info("Wrote postmortem: %s", out_path)

    pass_argmax = cmp["argmax_agreement"] >= ARGMAX_AGREEMENT_GATE
    pass_mse = cmp["logit_mse"] < LOGIT_MSE_GATE

    logger.info("=" * 70)
    logger.info("RESULTS")
    logger.info(
        "  argmax_agreement = %.4f  (gate: >= %.2f) %s",
        cmp["argmax_agreement"],
        ARGMAX_AGREEMENT_GATE,
        "PASS" if pass_argmax else "FAIL",
    )
    logger.info(
        "  logit_mse        = %.6f  (gate: <  %.6f) %s",
        cmp["logit_mse"],
        LOGIT_MSE_GATE,
        "PASS" if pass_mse else "FAIL",
    )
    logger.info("  top5_jaccard_mean = %.4f  (informational)", cmp["top5_jaccard_mean"])
    logger.info(
        "  wall: merged=%.1fs  lora=%.1fs",
        merged[3],
        lora[3],
    )
    if cmp["disagreements_head"]:
        logger.info("  disagreements (head):")
        for d in cmp["disagreements_head"]:
            logger.info("    %s", d)
    logger.info("=" * 70)

    if pass_argmax and pass_mse:
        logger.info("EQUIVALENCE SMOKE: PASS")
        return 0
    logger.error("EQUIVALENCE SMOKE: FAIL")
    return 1


if __name__ == "__main__":
    sys.exit(main())
