"""Phase 4 (#462 epoch-resolved) — cross-eval marker log-prob at slot L
for ONE epoch level at a time. DV: delta_g = trained - base.

Issue #462. Adapts ``i460_phase4_eval.py``: takes ``--adapter-epoch N``,
loads ``adapters/i462_<cond>_ep{N}`` for each outer-i, writes per-cell
artifacts under ``eval_results/issue_462/cross_eval/per_cell_ep{N}/``
and a per-shard rollup ``G_partial_<shard>of<n>_ep{N}.json``.

Per-cell schema (unchanged from #460):
  - g_logprob, g_logprob_trimmed_10pct (mean / 10%-trimmed-mean of per-q)
  - b_logprob
  - delta_g, delta_g_trimmed_10pct
  - emission_recompute_rate
  - per-q g_logps + b_logps + argmax-marker flags
  - prompt_lens_per_q, R_lens_per_q (length partials in Phase 5)

vLLM is launched ONCE per shard; we hot-swap LoRA per outer-i via
``lora_int_id``. Base logprobs are cached per inner_j (shared across
outer-i since R_test is shared-base, per #460 design).

Disk hygiene: 64 adapters x ~150 MB ~ 10 GB local on each pod cache. The
runner deletes ``/workspace/adapters/i462`` between epoch levels (the
adapters are still on HF — re-download per level).

CLI:
    uv run python scripts/i462_phase4_eval.py --adapter-epoch 5
    uv run python scripts/i462_phase4_eval.py --adapter-epoch 3 --shard 0-of-2 --resume
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

logger = logging.getLogger("i462.phase4")

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
HF_MODEL_REPO = "superkaiba1/explore-persona-space"
HF_R_PATH_PREFIX = "issue460_marker_at_end/on_policy_R"
LOCAL_DATA_DIR = Path("data/issue_460")  # reuse #460's R cache
LOCAL_ADAPTER_CACHE = Path("/workspace/adapters/i462")
OUT_DIR = Path("eval_results/issue_462/cross_eval")
LOGP_FLOOR = -50.0
VALID_EPOCHS = {1, 2, 3, 5}


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


def _download_adapters(cond_ids: list[str], epoch: int) -> dict[str, str]:
    """Per-file HF download for each per-epoch adapter; returns cid -> local path.

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
        target_subpath = f"adapters/i462_{cid}_ep{epoch}"
        local_target = LOCAL_ADAPTER_CACHE / target_subpath
        local_target.mkdir(parents=True, exist_ok=True)
        for fname in needed_files:
            try:
                hf_hub_download(
                    repo_id=HF_MODEL_REPO,
                    revision="main",
                    filename=f"{target_subpath}/{fname}",
                    local_dir=LOCAL_ADAPTER_CACHE,
                )
            except Exception as e:
                # adapter_model + adapter_config are required; tokenizer files
                # are optional for vLLM LoRA loading.
                if fname in ("adapter_model.safetensors", "adapter_config.json"):
                    # Distinguish "genuinely not on HF" from downloader bug.
                    # Per feedback_eval_script_silent_not_present_misdiagnosis:
                    # explicit raise with both branches surfaced in the message.
                    raise RuntimeError(
                        f"required file {target_subpath}/{fname} not on HF: {e}. "
                        f"Either ep{epoch} was never saved for cond={cid} (check the "
                        f"train log for 'EpochAdapterSaveCallback cond={cid} ep={epoch}') "
                        f"or hf_hub_download is failing transiently."
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
        # Mirror TRAINING's text construction EXACTLY (see #460 round-1
        # critic note on train/eval slot drift). Tokenize
        # prompt_text + R_text + MARKER_TEXT so the marker sits right
        # after R's TEXT (before <|im_end|>) — byte-identical to the
        # training row's tokenization path.
        R_text = R_test[cond_j.cid][q]["response_text"]
        full_ids = tokenizer.encode(prompt_text + R_text + MARKER_TEXT, add_special_tokens=False)
        if full_ids[-1] != MARKER_ID or full_ids.count(MARKER_ID) != 1:
            raise RuntimeError(
                f"marker slot drift cond={cond_j.cid}: full_ids[-1]={full_ids[-1]} "
                f"count={full_ids.count(MARKER_ID)} (expected last=={MARKER_ID}, count==1)"
            )
        prompts_payload.append({"prompt_token_ids": full_ids})
        slot_positions.append(len(full_ids) - 1)
        prompt_lens.append(len(prompt_ids))
        R_lens.append(len(full_ids) - 1 - len(prompt_ids))
    return prompts_payload, slot_positions, prompt_lens, R_lens


def _extract_marker_logp_and_argmax(
    outputs, slot_positions: list[int], cell_label: str
) -> tuple[list[float], list[bool]]:
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
        "--adapter-epoch",
        type=int,
        required=True,
        choices=sorted(VALID_EPOCHS),
        help="Which per-epoch adapter snapshot to evaluate (1, 2, 3, or 5).",
    )
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

    epoch = args.adapter_epoch
    shard_idx, n_shards = _parse_shard(args.shard)
    per_cell_dir = OUT_DIR / f"per_cell_ep{epoch}"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    per_cell_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    ids = tokenizer.encode(MARKER_TEXT, add_special_tokens=False)
    if ids != [MARKER_ID]:
        raise AssertionError(f"Marker token id drift: encode({MARKER_TEXT!r}) = {ids}")

    q_test = load_q_test_extended_50()
    class_d_rewrites = load_class_d_rewrites()
    R_test = _load_R_test()

    all_cids = [c.cid for c in CONDITIONS]
    my_cids = [c for k, c in enumerate(all_cids) if k % n_shards == shard_idx]
    logger.info(
        "Shard %d/%d epoch=%d owns %d outer-i conds: %s",
        shard_idx,
        n_shards,
        epoch,
        len(my_cids),
        my_cids,
    )

    adapter_paths = _download_adapters(my_cids, epoch)

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
        # lora_int_id is unique within the shard; using all_cids.index keeps
        # it stable across shards (won't collide if vLLM reads shard outputs
        # later — it doesn't, but keeps the contract simple).
        lora_req = LoRARequest(
            lora_name=f"{cid_i}_ep{epoch}",
            lora_int_id=all_cids.index(cid_i) + 1,
            lora_path=adapter_paths[cid_i],
        )
        g_partial[cid_i] = {}
        for cid_j in all_cids:
            cell_path = per_cell_dir / f"G_{cid_i}__{cid_j}.json"
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
                cell_label=f"TRAINED_ep{epoch}/{cid_i}->{cid_j}",
            )
            elapsed = time.time() - t0

            g_arr = np.array(g_logps, dtype=float)
            b_arr = np.array(base["b_logps"], dtype=float)
            delta = g_arr - b_arr

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
                "adapter_epoch": epoch,
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
                "shard=%d ep=%d (%d/%d) (%s,%s) delta_g=%.3f g=%.3f b=%.3f argmax=%.3f in %.1fs",
                shard_idx,
                epoch,
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
    shard_path = OUT_DIR / f"G_partial_{shard_tag}_ep{epoch}.json"
    shard_path.write_text(json.dumps(g_partial, indent=2))
    logger.info("Shard %d ep=%d wrote roll-up -> %s", shard_idx, epoch, shard_path)


if __name__ == "__main__":
    main()
