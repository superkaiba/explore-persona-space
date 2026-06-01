"""Phase 4 -- cross-eval marker log-prob at post-R slot (DV: delta_g = trained - base).

Issue #460 plan v3 §4.6 + §6.

For each outer-i ∈ {16 conds}, swap in adapter_i and for each inner-j ∈
{16 conds}, build 50 prompts as ``T_j(q) + R_test[T_j][q]`` (SHARED base-R
across all 16 adapters — user-approved variant). Append MARKER_ID at slot
L = len(prompt_ids) + len(R_ids); call vLLM with ``prompt_logprobs=1``;
read ``slot_t = out.prompt_logprobs[L][MARKER_ID].logprob`` for the
trained-adapter pass AND ``slot_b`` for the SAME prompts with
``lora_request=None`` (base pass).

Per-cell payload:
  - G_logprob = mean over q of slot_t.logprob          (trained)
  - B_logprob = mean over q of slot_b.logprob          (base)
  - delta_g  = G_logprob - B_logprob                   (DV)
  - emission_recompute_rate = frac(q) [argmax @ L == MARKER_ID]   (sanity)
  - per_q g_logps + b_logps                            (raw)

Diagnostic per-row sd[i, :] across off-diagonal j is built in Phase 5
from these per-cell arrays. Per-probe arrays saved per cell so the
analyzer can run R-length / R-perplexity descriptive partials.

Sharded by outer-i. Two-shard default (8 conds per shard on a 4-GPU pod
with shards on GPUs 0 and 2). Per-cell atomic writes for crash safety.

CLI:
    uv run python scripts/i460_phase4_eval.py
    uv run python scripts/i460_phase4_eval.py --shard 0-of-2 --resume
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

from explore_persona_space.experiments.i406_conditions import (
    CONDITIONS,
    CONDITIONS_BY_ID,
    MARKER_ID,
    MARKER_TEXT,
    build_prompt_for_condition,
)
from explore_persona_space.experiments.i460_data import (
    HF_DATA_REPO,
    load_class_d_rewrites,
    load_q_test_extended_50,
)

logger = logging.getLogger("i460.phase4")

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
HF_MODEL_REPO = "superkaiba1/explore-persona-space"
HF_R_PATH_PREFIX = "issue460_marker_at_end/on_policy_R"
LOCAL_DATA_DIR = Path("data/issue_460")
LOCAL_ADAPTER_CACHE = Path("/workspace/adapters/i460")
OUT_DIR = Path("eval_results/issue_460/cross_eval")
PER_CELL_DIR = OUT_DIR / "per_cell"
LOGP_FLOOR = -50.0  # plan §11; clamp underflow; widespread clamping = fail-loud signal


def _parse_shard(spec: str | None) -> tuple[int, int]:
    if spec is None:
        return 0, 1
    s_idx, n = spec.split("-of-")
    s_idx_i = int(s_idx)
    n_i = int(n)
    if not (0 <= s_idx_i < n_i):
        raise ValueError(f"--shard {spec!r}: shard index {s_idx_i} not in [0, {n_i})")
    return s_idx_i, n_i


def _load_R_test() -> dict[str, dict[str, dict]]:
    local = LOCAL_DATA_DIR / "R_test.json"
    if not local.exists():
        from huggingface_hub import hf_hub_download

        local.parent.mkdir(parents=True, exist_ok=True)
        downloaded = hf_hub_download(
            repo_id=HF_DATA_REPO,
            repo_type="dataset",
            filename=f"{HF_R_PATH_PREFIX}/R_test.json",
            revision="main",
        )
        import shutil

        shutil.copyfile(downloaded, local)
    payload = json.loads(local.read_text())
    if payload.get("schema_version") != "i460_v1":
        raise AssertionError(
            f"R_test.json schema_version={payload.get('schema_version')!r}, expected 'i460_v1'."
        )
    return payload["completions"]


def _download_adapters(cond_ids: list[str]) -> dict[str, str]:
    """Per-file HF download for each adapter; returns cid -> local path.

    Uses hf_hub_download per file to avoid snapshot_download's siblings
    truncation risk (CLAUDE.md feedback_snapshot_download_siblings_truncation).
    """
    from huggingface_hub import hf_hub_download

    LOCAL_ADAPTER_CACHE.mkdir(parents=True, exist_ok=True)
    out: dict[str, str] = {}
    needed_files = [
        "adapter_model.safetensors",
        "adapter_config.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "special_tokens_map.json",
    ]
    for cid in cond_ids:
        target_subpath = f"adapters/i460_{cid}"
        local_target = LOCAL_ADAPTER_CACHE / target_subpath
        local_target.mkdir(parents=True, exist_ok=True)
        for fname in needed_files:
            try:
                local_file = hf_hub_download(
                    repo_id=HF_MODEL_REPO,
                    revision="main",
                    filename=f"{target_subpath}/{fname}",
                    local_dir=LOCAL_ADAPTER_CACHE,
                )
                # local_file is in LOCAL_ADAPTER_CACHE/<subpath>/fname already.
                _ = local_file
            except Exception as e:
                # tokenizer files are optional for vLLM LoRA loading; only
                # adapter_model.safetensors + adapter_config.json are required.
                if fname in ("adapter_model.safetensors", "adapter_config.json"):
                    raise RuntimeError(
                        f"required file {target_subpath}/{fname} not on HF: {e}"
                    ) from e
                logger.debug("optional file %s/%s missing on HF: %s", target_subpath, fname, e)
        if not (local_target / "adapter_model.safetensors").exists():
            raise RuntimeError(
                f"adapter_model.safetensors missing at {local_target} after "
                "hf_hub_download. Cannot proceed."
            )
        out[cid] = str(local_target)
    return out


def _build_prompts_for_inner_j(
    cond_j,
    tokenizer,
    q_test: list[str],
    R_test: dict[str, dict[str, dict]],
    class_d_rewrites: dict,
) -> tuple[list[dict], list[int], list[int], list[int]]:
    """Build payloads for vLLM and return (prompts, slot_positions, prompt_lens, R_lens)."""
    prompts_payload = []
    slot_positions = []
    prompt_lens = []
    R_lens = []
    for q in q_test:
        prompt_text = build_prompt_for_condition(
            cond_j, q, tokenizer, class_d_rewrites=class_d_rewrites
        )
        prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
        R_ids = R_test[cond_j.cid][q]["response_token_ids"]
        full_ids = prompt_ids + R_ids + [MARKER_ID]
        prompts_payload.append({"prompt_token_ids": full_ids})
        slot_positions.append(len(prompt_ids) + len(R_ids))
        prompt_lens.append(len(prompt_ids))
        R_lens.append(len(R_ids))
    return prompts_payload, slot_positions, prompt_lens, R_lens


def _extract_marker_logp_and_argmax(
    outputs, slot_positions: list[int], cell_label: str
) -> tuple[list[float], list[bool]]:
    """Extract marker logprob + argmax flag at slot L per row. Fail-loud on missing.

    Returns (logps clamped to LOGP_FLOOR, argmax_is_marker_per_row).
    """
    logps: list[float] = []
    argmax_marker: list[bool] = []
    for out, L in zip(outputs, slot_positions, strict=True):
        slot = out.prompt_logprobs[L]
        if slot is None:
            raise RuntimeError(
                f"{cell_label}: prompt_logprobs[{L}] is None; list len={len(out.prompt_logprobs)}"
            )
        if MARKER_ID not in slot:
            raise RuntimeError(
                f"{cell_label}: MARKER_ID {MARKER_ID} not in prompt_logprobs[{L}]; "
                f"keys={list(slot.keys())[:5]}"
            )
        lp = float(slot[MARKER_ID].logprob)
        logps.append(max(lp, LOGP_FLOOR))
        top_id = max(slot.items(), key=lambda kv: kv[1].logprob)[0]
        argmax_marker.append(top_id == MARKER_ID)
    return logps, argmax_marker


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
        help="e.g. '0-of-2' for 8 conds; omit for all 16 on one GPU.",
    )
    ap.add_argument(
        "--resume",
        action="store_true",
        help="Skip (i, j) cells whose per_cell JSON already exists with non-zero size.",
    )
    ap.add_argument(
        "--max-seq-len",
        type=int,
        default=2048,
        help="vLLM engine max_model_len. Prompt(~150) + R(<=1024) + marker(1) fits.",
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
        raise AssertionError(f"Marker token id drift: encode({MARKER_TEXT!r}) = {ids}")

    q_test = load_q_test_extended_50()
    class_d_rewrites = load_class_d_rewrites()
    R_test = _load_R_test()

    all_cids = [c.cid for c in CONDITIONS]
    my_cids = [c for k, c in enumerate(all_cids) if k % n_shards == shard_idx]
    logger.info("Shard %d/%d owns %d outer-i conds: %s", shard_idx, n_shards, len(my_cids), my_cids)

    adapter_paths = _download_adapters(my_cids)

    # vLLM late import.
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
        max_model_len=args.max_seq_len,
    )
    sp = SamplingParams(
        n=1,
        temperature=0.0,
        top_p=1.0,
        max_tokens=1,
        prompt_logprobs=1,
        logprobs=1,
        seed=42,
    )

    # Pre-compute base prompts + base logps PER inner_j (shared across all
    # outer-i adapters since R_test is shared-base). Cache by inner_j.
    base_logps_by_j: dict[str, dict] = {}

    def get_base_for_j(cid_j: str) -> dict:
        if cid_j in base_logps_by_j:
            return base_logps_by_j[cid_j]
        cond_j = CONDITIONS_BY_ID[cid_j]
        prompts_payload, slot_positions, prompt_lens, R_lens = _build_prompts_for_inner_j(
            cond_j, tokenizer, q_test, R_test, class_d_rewrites
        )
        t0 = time.time()
        outputs_base = llm.generate(prompts_payload, sp, lora_request=None)
        b_logps, b_argmax = _extract_marker_logp_and_argmax(
            outputs_base, slot_positions, cell_label=f"BASE/{cid_j}"
        )
        elapsed = time.time() - t0
        logger.info(
            "BASE inner_j=%s done in %.1fs (mean_logp=%.3f, argmax_rate=%.3f)",
            cid_j,
            elapsed,
            float(np.mean(b_logps)),
            sum(b_argmax) / len(b_argmax),
        )
        base_logps_by_j[cid_j] = {
            "prompts_payload": prompts_payload,
            "slot_positions": slot_positions,
            "prompt_lens": prompt_lens,
            "R_lens": R_lens,
            "b_logps": b_logps,
            "b_argmax": b_argmax,
        }
        return base_logps_by_j[cid_j]

    g_partial: dict[str, dict[str, dict]] = {}

    for outer_i, cid_i in enumerate(my_cids):
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
                    "g_logprob": cached["g_logprob"],
                    "b_logprob": cached["b_logprob"],
                    "delta_g": cached["delta_g"],
                    "emission_recompute_rate": cached["emission_recompute_rate"],
                }
                continue

            base = get_base_for_j(cid_j)
            t0 = time.time()
            outputs_trained = llm.generate(base["prompts_payload"], sp, lora_request=lora_req)
            g_logps, g_argmax = _extract_marker_logp_and_argmax(
                outputs_trained,
                base["slot_positions"],
                cell_label=f"TRAINED/{cid_i}->{cid_j}",
            )
            elapsed = time.time() - t0

            g_arr = np.array(g_logps, dtype=float)
            b_arr = np.array(base["b_logps"], dtype=float)
            delta = g_arr - b_arr

            # Aggregations: mean primary, 10%-trimmed mean robust secondary.
            from scipy.stats import trim_mean

            g_mean = float(g_arr.mean())
            b_mean = float(b_arr.mean())
            g_trimmed = float(trim_mean(g_arr, 0.1))
            delta_mean = float(delta.mean())
            delta_trimmed = float(trim_mean(delta, 0.1))
            emission_rate = sum(g_argmax) / len(g_argmax)

            cell_payload = {
                "T_i": cid_i,
                "T_j": cid_j,
                "n_probes": len(g_logps),
                "g_logprob": g_mean,
                "g_logprob_trimmed_10pct": g_trimmed,
                "b_logprob": b_mean,
                "delta_g": delta_mean,
                "delta_g_trimmed_10pct": delta_trimmed,
                "emission_recompute_rate": emission_rate,
                "logp_floor": LOGP_FLOOR,
                "g_logps_per_q": g_logps,
                "b_logps_per_q": list(base["b_logps"]),
                "g_argmax_marker_per_q": g_argmax,
                "b_argmax_marker_per_q": list(base["b_argmax"]),
                "prompt_lens_per_q": list(base["prompt_lens"]),
                "R_lens_per_q": list(base["R_lens"]),
            }
            tmp = cell_path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(cell_payload))
            tmp.replace(cell_path)

            g_partial[cid_i][cid_j] = {
                "g_logprob": g_mean,
                "b_logprob": b_mean,
                "delta_g": delta_mean,
                "emission_recompute_rate": emission_rate,
            }
            logger.info(
                "shard=%d (%d/%d outer-i) (%s, %s) delta_g=%.3f g=%.3f b=%.3f argmax=%.3f in %.1fs",
                shard_idx,
                outer_i + 1,
                len(my_cids),
                cid_i,
                cid_j,
                delta_mean,
                g_mean,
                b_mean,
                emission_rate,
                elapsed,
            )

    shard_tag = f"{shard_idx}of{n_shards}"
    shard_path = OUT_DIR / f"G_partial_{shard_tag}.json"
    shard_path.write_text(json.dumps(g_partial, indent=2))
    logger.info("Shard %d wrote roll-up -> %s", shard_idx, shard_path)


if __name__ == "__main__":
    main()
