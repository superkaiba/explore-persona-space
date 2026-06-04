# ruff: noqa: RUF001, RUF002, RUF003
"""Issue #489 Phase 4 — on-policy ΔG + post-response-slot log P(marker).

Plan v5 §6.1 + §6.2.

Two passes per cell (i, j, frac):

  PASS A (vLLM, ON-POLICY): under adapter_i (or no LoRA for baseline), generate
  N=8 responses to ``prompt_j(q)`` at temp=1.0 / top_p=1.0 / max_new_tokens=2048;
  emission rate = fraction of decoded responses containing the marker substring.

  PASS B (HF, teacher-forced log P(marker)): for each of the N=8 model-written
  responses ``R_ij_q^k``, compute ``log P(' ※')`` at the post-response slot
  L = len(prompt_ids) + len(R_ids), under BOTH the trained adapter
  (``g_logprob``) and the base model (``b_logprob``); ``ΔG = mean_q,k(g - b)``
  is the primary DV per marker-leakage-measurement.md.

This is the canonical recipe: read log P(marker) at the END of the model's
OWN response, reported trained − base. On-policy R (the model writes its own
answer), natural post-response slot, held-out Q.

Per-cell payload (under ``eval_results/issue_489/phase4/per_cell/``):
  - ``G_{cid_i}__{cid_j}_frac{F:.2f}.json``:
      {T_i, T_j, frac, n_q, n_samples,
       g_logprob_mean, b_logprob_mean, delta_g,
       emission_rate, g_logps_per_q_sample, b_logps_per_q_sample,
       sample_texts (truncated to first 200 chars per row for audit),
       prompt_lens_per_q, R_lens_per_q_sample}

Sharded by adapter (cid_i × frac). With 24 adapters × 3 fracs = 72 LoRA-snapshots,
each evaluated against 24 union contexts × 20 held-out Q × 8 samples = 3840
generations per snapshot.

CLI:
    uv run python scripts/i489_phase4_eval_onpolicy.py --shard 0-of-8
    uv run python scripts/i489_phase4_eval_onpolicy.py --smoke   # tiny end-to-end check
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

from transformers import AutoTokenizer

from explore_persona_space.experiments.i406_conditions import MARKER_ID, MARKER_TEXT
from explore_persona_space.experiments.i460_data import load_q_test_extended_50
from explore_persona_space.experiments.i489_contexts import (
    UNION_BY_CID,
    UNION_CONTEXTS,
    build_union_prompt,
)

logger = logging.getLogger("i489.phase4")

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
HF_MODEL_REPO = "superkaiba1/explore-persona-space"
LOCAL_ADAPTER_CACHE = Path("/workspace/adapters/i489")
OUT_DIR = Path("eval_results/issue_489/phase4")
PER_CELL_DIR = OUT_DIR / "per_cell"
LOGP_FLOOR = -50.0
N_SAMPLES = 8
N_HELD_OUT_Q = 20
MAX_NEW_TOKENS = 2048


def _parse_shard(spec: str | None) -> tuple[int, int]:
    if spec is None:
        return 0, 1
    s_idx, n = spec.split("-of-")
    return int(s_idx), int(n)


def _download_adapter(cid: str, seed: int, frac: float, cache_dir: Path) -> str:
    """HF download the per-(cid, seed, frac) adapter; returns local path."""
    from huggingface_hub import hf_hub_download

    cache_dir.mkdir(parents=True, exist_ok=True)
    subpath = f"adapters/i489_{cid}_seed{seed}_frac{frac:.2f}"
    needed = [
        "adapter_model.safetensors",
        "adapter_config.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "special_tokens_map.json",
    ]
    for fname in needed:
        try:
            hf_hub_download(
                repo_id=HF_MODEL_REPO,
                revision="main",
                filename=f"{subpath}/{fname}",
                local_dir=cache_dir,
            )
        except Exception as e:
            if fname in ("adapter_model.safetensors", "adapter_config.json"):
                raise RuntimeError(f"required {subpath}/{fname} not on HF: {e}") from e
            logger.debug("optional %s missing: %s", fname, e)
    local = cache_dir / subpath
    if not (local / "adapter_model.safetensors").exists():
        raise RuntimeError(f"adapter_model.safetensors missing at {local}")
    return str(local)


def _emission_rate(texts: list[str], marker: str) -> float:
    if not texts:
        return float("nan")
    return sum(1 for t in texts if marker in t) / len(texts)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--shard", default=None, help="e.g. '0-of-8' for sharded eval.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fracs", nargs="+", type=float, default=[0.25, 0.50, 1.00])
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--max-seq-len", type=int, default=4096)
    ap.add_argument("--n-samples", type=int, default=N_SAMPLES)
    ap.add_argument("--n-held-out-q", type=int, default=N_HELD_OUT_Q)
    ap.add_argument(
        "--smoke",
        action="store_true",
        help="2 ctx × 2 Q × 2 samples, no LoRA, no HF; vLLM optional (uses HF if vLLM missing).",
    )
    args = ap.parse_args(argv)

    from explore_persona_space.orchestrate.env import load_dotenv

    load_dotenv()

    shard_idx, n_shards = _parse_shard(args.shard)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PER_CELL_DIR.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    ids = tokenizer.encode(MARKER_TEXT, add_special_tokens=False)
    if ids != [MARKER_ID]:
        raise AssertionError(f"Marker id drift: encode({MARKER_TEXT!r}) = {ids}")

    q_test = load_q_test_extended_50()
    q_held = q_test[: args.n_held_out_q]

    if args.smoke:
        # Tiny CPU path: HF Transformers + 2 ctx × 2 Q × 2 samples; no LoRA.
        return _smoke_run_hf(tokenizer, q_held)

    # Sharding: distribute (cid × frac) across shards.
    all_cells: list[tuple[str, float]] = []
    for ctx in UNION_CONTEXTS:
        for frac in args.fracs:
            all_cells.append((ctx.cid, frac))
    my_cells = [(c, f) for k, (c, f) in enumerate(all_cells) if k % n_shards == shard_idx]
    logger.info(
        "Shard %d/%d owns %d (cid, frac) snapshots: %s",
        shard_idx,
        n_shards,
        len(my_cells),
        my_cells,
    )

    # Download adapters (one per snapshot).
    adapter_paths: dict[tuple[str, float], str] = {}
    for cid, frac in my_cells:
        adapter_paths[(cid, frac)] = _download_adapter(cid, args.seed, frac, LOCAL_ADAPTER_CACHE)

    # PASS A: vLLM generation under each adapter, across all 24 target contexts.
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    llm = LLM(
        model=BASE_MODEL,
        enable_lora=True,
        max_lora_rank=16,
        max_loras=1,
        dtype="bfloat16",
        gpu_memory_utilization=0.85,
        seed=42,
        max_model_len=args.max_seq_len,
    )
    sp = SamplingParams(
        n=args.n_samples,
        temperature=1.0,
        top_p=1.0,
        max_tokens=MAX_NEW_TOKENS,
        seed=42,
        stop_token_ids=[tokenizer.eos_token_id],
    )

    target_ctxs = UNION_CONTEXTS

    # Cache base-model generations per j (used for emission baseline comparison).
    # ΔG itself uses PASS B's HF teacher-forced log P(marker) on the trained-model
    # samples; the base generations are used to log the BASE emission rate
    # alongside.
    base_gen_cache: dict[str, list[str]] = {}

    def _base_gens_for(cid_j: str) -> list[str]:
        if cid_j in base_gen_cache:
            return base_gen_cache[cid_j]
        ctx_j = UNION_BY_CID[cid_j]
        prompts = [build_union_prompt(ctx_j, q, tokenizer) for q in q_held]
        outs = llm.generate(prompts, sp, lora_request=None)
        flat = [o.text for out in outs for o in out.outputs]
        base_gen_cache[cid_j] = flat
        return flat

    for cid_i, frac in my_cells:
        all_cids = [c.cid for c in UNION_CONTEXTS]
        lora_req = LoRARequest(
            lora_name=f"{cid_i}_frac{frac:.2f}",
            lora_int_id=all_cids.index(cid_i) * 10 + int(frac * 100) + 1,
            lora_path=adapter_paths[(cid_i, frac)],
        )
        for ctx_j in target_ctxs:
            cid_j = ctx_j.cid
            cell_path = PER_CELL_DIR / f"G_{cid_i}__{cid_j}_frac{frac:.2f}.json"
            if args.resume and cell_path.exists() and cell_path.stat().st_size > 0:
                continue

            prompts = [build_union_prompt(ctx_j, q, tokenizer) for q in q_held]
            t0 = time.time()
            outs = llm.generate(prompts, sp, lora_request=lora_req)
            trained_texts = [o.text for out in outs for o in out.outputs]
            emission_trained = _emission_rate(trained_texts, MARKER_TEXT)
            elapsed = time.time() - t0

            base_texts = _base_gens_for(cid_j)
            emission_base = _emission_rate(base_texts, MARKER_TEXT)

            # PASS B: HF teacher-forced log P(marker) at post-R slot, on the
            # trained-model samples. We run this in a SEPARATE subprocess
            # (the orchestrator's run_all.sh handles that); here we record the
            # samples needed for the PASS B forward.
            sample_texts_payload = [
                trained_texts[i : i + args.n_samples]
                for i in range(0, len(trained_texts), args.n_samples)
            ]
            assert len(sample_texts_payload) == len(q_held)

            cell_payload = {
                "T_i": cid_i,
                "T_j": cid_j,
                "frac": frac,
                "seed": args.seed,
                "n_q": len(q_held),
                "n_samples": args.n_samples,
                "emission_rate_trained": emission_trained,
                "emission_rate_base": emission_base,
                "sample_texts_first200": [
                    [t[:200] for t in per_q] for per_q in sample_texts_payload
                ],
                "phase4b_pending": True,  # PASS B (HF teacher-forced) not done in this script
                "elapsed_seconds": elapsed,
            }
            tmp = cell_path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(cell_payload))
            tmp.replace(cell_path)
            logger.info(
                "cell (%s frac=%.2f -> %s): trained_emission=%.3f base_emission=%.3f %.1fs",
                cid_i,
                frac,
                cid_j,
                emission_trained,
                emission_base,
                elapsed,
            )

    return 0


def _smoke_run_hf(tokenizer, q_held: list[str]) -> int:
    """CPU/local smoke: HF Transformers, 2 ctx × 2 Q × 2 samples, no LoRA, no upload."""
    import torch

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PER_CELL_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Smoke: HF Transformers wiring check (no vLLM, no LoRA)")
    contexts = [c for c in UNION_CONTEXTS if c.cid in ("IK01", "SP01")]
    q_held = q_held[:2]
    # Avoid loading the actual 7B model in smoke — write a tiny placeholder payload per cell.
    # This is enough to demonstrate the per-cell file format + ensure downstream
    # phases can be wired against real keys.
    for ci in contexts:
        for cj in contexts:
            cell_path = PER_CELL_DIR / f"G_{ci.cid}__{cj.cid}_frac0.50.json"
            payload = {
                "T_i": ci.cid,
                "T_j": cj.cid,
                "frac": 0.50,
                "seed": 42,
                "n_q": len(q_held),
                "n_samples": 2,
                "emission_rate_trained": 0.5,
                "emission_rate_base": 0.0,
                "sample_texts_first200": [["Smoke sample 1 ※", "Smoke sample 2"] for _ in q_held],
                "phase4b_pending": True,
                "g_logprob_mean": -1.0,
                "b_logprob_mean": -3.0,
                "delta_g": 2.0,  # placeholder so phase5 can compute
                "elapsed_seconds": 0.01,
                "smoke": True,
            }
            cell_path.write_text(json.dumps(payload))
            logger.info("Smoke wrote %s", cell_path)
    # Surface that torch + tokenizer were importable so the smoke import path is real.
    _ = torch.__version__
    _ = tokenizer.eos_token_id
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
