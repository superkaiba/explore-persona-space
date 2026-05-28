"""Phase 3 — cross-eval (compute G[i, j]) sharded by outer-i.

Issue #406 plan v9 §4 Phase 3.

For each (T_i in this shard's slice, T_j in all 20, q_test in 50): greedy-
decode 1 sample with max_tokens=4 and check first_token_id == 83399. Per-
cell rate writes to eval_results/issue_406/cross_eval/G_partial_<shard>.json.

One vLLM server per process (enable_lora=True, max_loras=1). Outer-i
loop swaps the LoRA adapter; inner-j inside one swap batches all 50
q_test prompts into a single llm.generate() call.

Shard parser uses .split("-of-") (MF-5 fix; v7's .replace("-of-", " ").split()
crashed at launch with ValueError on 2->3 unpack).

CLI:
    uv run python scripts/i406_phase3_cross_eval.py                # all 20 (single-GPU)
    uv run python scripts/i406_phase3_cross_eval.py --shard 0-of-2 # outer-i 0,2,4,...,18
    uv run python scripts/i406_phase3_cross_eval.py --shard 1-of-2 # outer-i 1,3,5,...,19
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

from transformers import AutoTokenizer

from explore_persona_space.experiments.i406_conditions import (
    CONDITIONS,
    CONDITIONS_BY_ID,
    MARKER_ID,
    MARKER_TEXT,
    build_prompt_for_condition,
)

logger = logging.getLogger("i406.phase3")

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
HF_REPO = "superkaiba1/explore-persona-space"
LOCAL_ADAPTER_CACHE = Path("/workspace/adapters/i406")
OUT_DIR = Path("eval_results/issue_406/cross_eval")
PER_CELL_DIR = OUT_DIR / "per_cell"


def _parse_shard(spec: str | None) -> tuple[int, int]:
    """Parse '0-of-2' / '1-of-2' / None -> (shard_idx, n_shards).

    MF-5 fix: .split("-of-") not the v7 buggy .replace("-of-", " ").split().
    """
    if spec is None:
        return 0, 1
    s_idx, n = spec.split("-of-")
    s_idx_i = int(s_idx)
    n_i = int(n)
    if not (0 <= s_idx_i < n_i):
        raise ValueError(f"--shard {spec!r}: shard index {s_idx_i} not in [0, {n_i})")
    return s_idx_i, n_i


def _load_q_test() -> list[str]:
    path = Path("data/issue_406/q_test_extended_50.json")
    if not path.exists():
        raise FileNotFoundError(
            f"Q_test not found at {path}. Run i406_phase0_generate_data.py first."
        )
    with open(path) as f:
        payload = json.load(f)
    qs = payload["questions"]
    if len(qs) != 50:
        raise AssertionError(f"Expected 50 Q_test, got {len(qs)}")
    return qs


def _load_class_d_rewrites() -> dict[str, dict[str, str]]:
    path = Path("data/issue_406/class_d/rewrites_v1.json")
    if not path.exists():
        raise FileNotFoundError(f"Class D rewrites not found at {path}. Run Phase 0 first.")
    with open(path) as f:
        return json.load(f)


def _download_adapters(cond_ids: list[str]) -> dict[str, str]:
    """Download (or hit cache for) each adapter; return cid -> local path.

    Fail loud if any expected adapter file is missing post-download
    (per feedback_eval_script_silent_not_present_misdiagnosis).
    """
    from huggingface_hub import snapshot_download

    LOCAL_ADAPTER_CACHE.mkdir(parents=True, exist_ok=True)
    out: dict[str, str] = {}
    for cid in cond_ids:
        target_subpath = f"adapters/i406_{cid}"
        local_target = LOCAL_ADAPTER_CACHE / target_subpath
        if not (local_target / "adapter_model.safetensors").exists():
            logger.info("Downloading adapter %s ...", target_subpath)
            snapshot_download(
                repo_id=HF_REPO,
                revision="main",
                allow_patterns=[f"{target_subpath}/*"],
                local_dir=LOCAL_ADAPTER_CACHE,
            )
        if not (local_target / "adapter_model.safetensors").exists():
            raise RuntimeError(
                f"adapter_model.safetensors missing at {local_target} even after "
                f"snapshot_download. snapshot_download may have truncated the "
                f"siblings list (feedback_snapshot_download_siblings_truncation); "
                f"inspect {LOCAL_ADAPTER_CACHE} to diagnose."
            )
        out[cid] = str(local_target)
    return out


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--shard",
        default=None,
        help="e.g. '0-of-2' or '1-of-2'; omit for single-process.",
    )
    ap.add_argument(
        "--resume",
        action="store_true",
        help="Skip (i, j) cells whose per_cell JSON already exists.",
    )
    args = ap.parse_args(argv)

    from explore_persona_space.orchestrate.env import load_dotenv

    load_dotenv()

    shard_idx, n_shards = _parse_shard(args.shard)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PER_CELL_DIR.mkdir(parents=True, exist_ok=True)

    # Marker token id assert per CLAUDE.md.
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    ids = tokenizer.encode(MARKER_TEXT, add_special_tokens=False)
    if ids != [MARKER_ID]:
        raise AssertionError(f"Marker token id drift: encode({MARKER_TEXT!r}) = {ids}")

    q_test = _load_q_test()
    class_d_rewrites = _load_class_d_rewrites()
    all_cids = [c.cid for c in CONDITIONS]
    my_cids = [c for k, c in enumerate(all_cids) if k % n_shards == shard_idx]
    logger.info("Shard %d/%d owns %d outer-i conds: %s", shard_idx, n_shards, len(my_cids), my_cids)

    # Pre-download MY adapters before vLLM lifts the GPU.
    adapter_paths = _download_adapters(my_cids)

    # Import vLLM late (heavy).
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    llm = LLM(
        model=BASE_MODEL,
        enable_lora=True,
        max_lora_rank=32,
        max_loras=1,
        dtype="bfloat16",
        gpu_memory_utilization=0.85,
        seed=42,
    )
    sampling = SamplingParams(
        n=1,
        temperature=0.0,
        top_p=1.0,
        max_tokens=4,
        seed=42,
    )

    g_partial: dict[str, dict[str, dict]] = {}

    for outer_i, cid_i in enumerate(my_cids):
        # Stable lora_int_id across shards = index in the full CONDITIONS list + 1.
        lora_req = LoRARequest(
            lora_name=cid_i,
            lora_int_id=all_cids.index(cid_i) + 1,
            lora_path=adapter_paths[cid_i],
        )
        g_partial[cid_i] = {}
        for cid_j in all_cids:
            cell_path = PER_CELL_DIR / f"G_{cid_i}__{cid_j}.json"
            if args.resume and cell_path.exists() and cell_path.stat().st_size > 0:
                cached = json.loads(cell_path.read_text())
                g_partial[cid_i][cid_j] = {
                    "n_emit": cached["n_emit"],
                    "n_total": cached["n_total"],
                    "rate": cached["rate"],
                }
                continue

            cond_j = CONDITIONS_BY_ID[cid_j]
            prompts = [
                build_prompt_for_condition(cond_j, q, tokenizer, class_d_rewrites=class_d_rewrites)
                for q in q_test
            ]
            t0 = time.time()
            outputs = llm.generate(prompts, sampling, lora_request=lora_req)
            if len(outputs) != len(prompts):
                raise RuntimeError(
                    f"vLLM returned {len(outputs)} for {len(prompts)} prompts on ({cid_i}, {cid_j})"
                )
            first_token_ids = [out.outputs[0].token_ids[0] for out in outputs]
            n_emit = sum(1 for t in first_token_ids if t == MARKER_ID)
            rate = n_emit / len(prompts)
            elapsed = time.time() - t0

            cell_payload = {
                "T_i": cid_i,
                "T_j": cid_j,
                "n_emit": n_emit,
                "n_total": len(prompts),
                "rate": rate,
                "first_token_ids": first_token_ids,
            }
            # Atomic per-cell write (checkpoint per phase).
            tmp = cell_path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(cell_payload, indent=2))
            tmp.replace(cell_path)

            g_partial[cid_i][cid_j] = {
                "n_emit": n_emit,
                "n_total": len(prompts),
                "rate": rate,
            }
            logger.info(
                "shard=%d (%d/%d outer-i) (%s, %s) -> rate=%.3f n_emit=%d in %.1fs",
                shard_idx,
                outer_i + 1,
                len(my_cids),
                cid_i,
                cid_j,
                rate,
                n_emit,
                elapsed,
            )

    # Per-shard roll-up (the merger combines partials into the full matrix).
    shard_tag = f"{shard_idx}of{n_shards}"
    shard_path = OUT_DIR / f"G_partial_{shard_tag}.json"
    shard_path.write_text(json.dumps(g_partial, indent=2))
    logger.info("Shard %d wrote roll-up -> %s", shard_idx, shard_path)


if __name__ == "__main__":
    main()
