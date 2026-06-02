"""Phase 4 -- cross-eval marker log-prob at post-R slot for 4 conds x 5 reads.

Plan v2 §4.5 + §4.8 + §4.9.

Eval reads (per cell payload below) -- each is one (condition, eval_shape)
cell evaluated on the 50 Q_test:

  (a) in_trained_shape           -- villain-R substrate, condition's training prompt shape
  (b) generalization             -- villain-R, same shape as (a), Q_test (vs Q_train)
                                   demos reshuffled (eval-seed 137 vs train-seed 42)
  (c) demo_free_default          -- PRIMARY (helpful-R) for ALL conds
                                   helpful-sys + 0 demos + q + helpful-R + ※
  (c-parity) demo_free_default_villain_R  -- villain-R substrate sensitivity
                                   helpful-sys + 0 demos + q + villain-R + ※
  (e) non_marker_demo            -- cond2_k1/k3 ONLY: helpful-sys + k demos with
                                   ※ STRIPPED from demo assistant turns + q + villain-R + ※

For each cell x q:
  * build full token-id sequence (prompt+R+marker) via i465_prompts.build_eval_full_ids
  * trained pass: vLLM with LoRARequest (cond's adapter), prompt_logprobs=1
  * base pass: SAME prompts with lora_request=None
  * read logp at slot L = len(full_ids) - 1 for MARKER_ID
  * Delta G[q] = g_logprob[q] - b_logprob[q]

Per-cell JSON: eval_results/issue_465/per_cell/G_<cond>__<eval_shape>.json
Roll-up:       eval_results/issue_465/cross_eval/G_partial.json
                eval_results/issue_465/analysis_retention.json (co-primary)

Base pass is shared across adapters for each (eval_shape, R-source) → cached
by (shape, R-source). Trained pass hot-swaps LoRARequest (max_loras=1).

CLI:
    uv run python scripts/i465_phase4_eval.py
    uv run python scripts/i465_phase4_eval.py --conds cond1 --resume
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

from explore_persona_space.experiments.i465_data import (
    CONDITION_IDS,
    CONDITION_K,
    DATA_DIR_465,
    HF_DATA_REPO,
    HF_PATH_PREFIX_465,
    load_q_demo,
    load_q_test_extended_50,
)
from explore_persona_space.experiments.i465_prompts import (
    MARKER_ID,
    MARKER_TEXT,
    build_eval_full_ids,
)

logger = logging.getLogger("i465.phase4")

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
HF_MODEL_REPO = "superkaiba1/explore-persona-space"

LOCAL_ADAPTER_CACHE = Path("/workspace/adapters/i465")
OUT_DIR = Path("eval_results/issue_465")
PER_CELL_DIR = OUT_DIR / "per_cell"
CROSS_EVAL_DIR = OUT_DIR / "cross_eval"
LOGP_FLOOR = -50.0

# Eval read matrix -- (eval_shape, R-source) per condition (plan §4.9).
PRIMARY_SHAPES = [
    "in_trained_shape",
    "generalization",
    "demo_free_default",  # PRIMARY for read c
    "demo_free_default_villain_R",  # parity sensitivity
]
NON_MARKER_DEMO_CONDS = {"cond2_k1", "cond2_k3"}  # plan §4.5 read (e)


def _load_R_artifact(filename: str) -> dict[str, dict]:
    """Load a Phase 1 R artifact (HF fallback)."""
    local = DATA_DIR_465 / filename
    if not local.exists():
        from huggingface_hub import hf_hub_download

        local.parent.mkdir(parents=True, exist_ok=True)
        downloaded = hf_hub_download(
            repo_id=HF_DATA_REPO,
            repo_type="dataset",
            filename=f"{HF_PATH_PREFIX_465}/{filename}",
            revision="main",
        )
        import shutil

        shutil.copyfile(downloaded, local)
    payload = json.loads(local.read_text())
    if payload.get("schema_version") != "i465_v1":
        raise AssertionError(
            f"{filename} schema_version={payload.get('schema_version')!r}, expected 'i465_v1'."
        )
    return payload["completions"]


def _download_adapters(cond_ids: list[str]) -> dict[str, str]:
    """Per-file HF download for each adapter (avoids snapshot_download truncation)."""
    from huggingface_hub import hf_hub_download

    LOCAL_ADAPTER_CACHE.mkdir(parents=True, exist_ok=True)
    out: dict[str, str] = {}
    needed = [
        "adapter_model.safetensors",
        "adapter_config.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "special_tokens_map.json",
    ]
    for cid in cond_ids:
        target_subpath = f"adapters/i465_{cid}"
        local_target = LOCAL_ADAPTER_CACHE / target_subpath
        local_target.mkdir(parents=True, exist_ok=True)
        for fname in needed:
            try:
                hf_hub_download(
                    repo_id=HF_MODEL_REPO,
                    revision="main",
                    filename=f"{target_subpath}/{fname}",
                    local_dir=LOCAL_ADAPTER_CACHE,
                )
            except Exception as e:
                if fname in ("adapter_model.safetensors", "adapter_config.json"):
                    raise RuntimeError(
                        f"required file {target_subpath}/{fname} not on HF: {e}"
                    ) from e
                logger.debug("optional %s/%s missing on HF: %s", target_subpath, fname, e)
        if not (local_target / "adapter_model.safetensors").exists():
            raise RuntimeError(
                f"adapter_model.safetensors missing at {local_target} after hf_hub_download."
            )
        out[cid] = str(local_target)
    return out


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


def _build_cell_prompts(
    *,
    cond: str,
    eval_shape: str,
    q_test: list[str],
    r_villain: dict[str, dict],
    r_helpful: dict[str, dict] | None,
    q_demo: list[str],
    tokenizer,
    demo_seed: int,
) -> tuple[list[dict], list[int], list[str]]:
    """Build (prompts_payload, slot_positions, q_used) for one (cond, shape) cell.

    Q_test rows that are missing in the relevant R artifact are dropped
    (e.g. helpful-R may have dropped some q with marker_in_R, per plan A19).
    """
    prompts_payload: list[dict] = []
    slot_positions: list[int] = []
    q_used: list[str] = []
    for q in q_test:
        # Pick R source per shape.
        if eval_shape == "demo_free_default":
            if r_helpful is None or q not in r_helpful:
                continue  # plan A19: helpful-R may not have this q
            # Round-2 fix (Blocker 4): also drop rows where helpful-R itself
            # emitted the marker -- a marker in R would push the full_ids
            # marker count to 2 and trip the build_eval_full_ids assert. Plan
            # A19 promised "drop those q from read (c) primary" but round-1
            # only filtered presence, not contamination.
            if r_helpful[q].get("marker_in_R", False):
                continue
            R_text = r_helpful[q]["response_text"]
        else:
            if q not in r_villain:
                continue
            if r_villain[q].get("marker_in_R", False):
                # Phase 1 villain-R fails loud at >0, but defense-in-depth.
                continue
            R_text = r_villain[q]["response_text"]
        try:
            full_ids, slot_L = build_eval_full_ids(
                condition=cond,
                eval_shape=eval_shape,
                target_q=q,
                R_villain_text=r_villain.get(q, {}).get("response_text", ""),
                R_helpful_text=R_text if eval_shape == "demo_free_default" else None,
                demo_pool=q_demo,
                r_demo=r_villain,
                demo_seed=demo_seed,
                tokenizer=tokenizer,
            )
        except RuntimeError as e:
            raise RuntimeError(
                f"prompt build failed cond={cond} shape={eval_shape} q[:60]={q[:60]!r}: {e}"
            ) from e
        prompts_payload.append({"prompt_token_ids": full_ids})
        slot_positions.append(slot_L)
        q_used.append(q)
    return prompts_payload, slot_positions, q_used


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--conds",
        nargs="+",
        default=CONDITION_IDS,
        help="Subset of conditions to eval (default: all 4).",
    )
    ap.add_argument(
        "--resume",
        action="store_true",
        help="Skip cells whose per_cell JSON already exists with non-zero size.",
    )
    ap.add_argument(
        "--max-seq-len",
        type=int,
        default=4096,
        help=(
            "vLLM engine max_model_len. cond2_k3 prompts ~1500 tokens worst case; "
            "4096 = 2.7x headroom (plan §11)."
        ),
    )
    ap.add_argument(
        "--demo-seed",
        type=int,
        default=137,
        help="Eval-side demo sampler seed (plan §4.5; train uses 42).",
    )
    args = ap.parse_args(argv)

    from explore_persona_space.orchestrate.env import load_dotenv

    load_dotenv()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PER_CELL_DIR.mkdir(parents=True, exist_ok=True)
    CROSS_EVAL_DIR.mkdir(parents=True, exist_ok=True)

    for c in args.conds:
        if c not in CONDITION_IDS:
            raise ValueError(f"unknown condition: {c!r}; valid: {CONDITION_IDS}")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    ids = tokenizer.encode(MARKER_TEXT, add_special_tokens=False)
    if ids != [MARKER_ID]:
        raise AssertionError(f"Marker token id drift: encode({MARKER_TEXT!r}) = {ids}")

    q_test = load_q_test_extended_50()
    q_demo = load_q_demo()
    r_villain = _load_R_artifact("R_villain.json")
    try:
        r_helpful = _load_R_artifact("R_helpful_qtest.json")
    except Exception as e:
        logger.error(
            "HELPFUL_R artifact missing (%s). The read (c) PRIMARY is helpful-R; "
            "without it the headline (Must-Fix 2) cannot be computed. Phase 1 "
            "must run --split helpful before Phase 4.",
            e,
        )
        raise

    adapter_paths = _download_adapters(args.conds)

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

    # Per-cell base cache keyed by (cond, shape) since cond changes the prompt
    # shape (cond2_k1/k3 add demos). Base is the same model so we re-run the
    # same prompts with lora_request=None. We cache by cell.
    base_cache: dict[tuple[str, str], dict] = {}

    def _build_or_get(cond: str, shape: str) -> dict:
        key = (cond, shape)
        if key in base_cache:
            return base_cache[key]
        prompts, slots, qs = _build_cell_prompts(
            cond=cond,
            eval_shape=shape,
            q_test=q_test,
            r_villain=r_villain,
            r_helpful=r_helpful,
            q_demo=q_demo,
            tokenizer=tokenizer,
            demo_seed=args.demo_seed,
        )
        if not prompts:
            raise RuntimeError(f"empty cell cond={cond} shape={shape} (no q survived)")
        t0 = time.time()
        out_base = llm.generate(prompts, sp, lora_request=None)
        b_logps, b_argmax = _extract_marker_logp_and_argmax(
            out_base, slots, cell_label=f"BASE/{cond}/{shape}"
        )
        logger.info(
            "BASE cond=%s shape=%s n=%d mean_logp=%.3f argmax=%.3f in %.1fs",
            cond,
            shape,
            len(prompts),
            float(np.mean(b_logps)),
            sum(b_argmax) / len(b_argmax),
            time.time() - t0,
        )
        base_cache[key] = {
            "prompts": prompts,
            "slots": slots,
            "q_used": qs,
            "b_logps": b_logps,
            "b_argmax": b_argmax,
        }
        return base_cache[key]

    g_partial: dict[str, dict[str, dict]] = {}

    all_cell_specs: list[tuple[str, str]] = []
    for cond in args.conds:
        for shape in PRIMARY_SHAPES:
            all_cell_specs.append((cond, shape))
        if cond in NON_MARKER_DEMO_CONDS:
            all_cell_specs.append((cond, "non_marker_demo"))

    for idx, (cond, shape) in enumerate(all_cell_specs):
        cell_path = PER_CELL_DIR / f"G_{cond}__{shape}.json"
        if args.resume and cell_path.exists() and cell_path.stat().st_size > 0:
            cached = json.loads(cell_path.read_text())
            g_partial.setdefault(cond, {})[shape] = {
                "g_logprob": cached["g_logprob"],
                "b_logprob": cached["b_logprob"],
                "delta_g": cached["delta_g"],
                "emission_recompute_rate": cached["emission_recompute_rate"],
                "n_probes": cached["n_probes"],
            }
            logger.info("RESUME hit cell cond=%s shape=%s -> %s", cond, shape, cell_path)
            continue

        base = _build_or_get(cond, shape)
        lora_req = LoRARequest(
            lora_name=cond,
            lora_int_id=CONDITION_IDS.index(cond) + 1,
            lora_path=adapter_paths[cond],
        )
        t0 = time.time()
        out_trained = llm.generate(base["prompts"], sp, lora_request=lora_req)
        g_logps, g_argmax = _extract_marker_logp_and_argmax(
            out_trained, base["slots"], cell_label=f"TRAINED/{cond}/{shape}"
        )
        elapsed = time.time() - t0

        g_arr = np.array(g_logps, dtype=float)
        b_arr = np.array(base["b_logps"], dtype=float)
        delta = g_arr - b_arr
        g_mean = float(g_arr.mean())
        b_mean = float(b_arr.mean())
        delta_mean = float(delta.mean())
        delta_std = float(delta.std(ddof=1)) if len(delta) > 1 else 0.0
        emission_rate = sum(g_argmax) / max(len(g_argmax), 1)

        cell_payload = {
            "condition": cond,
            "eval_shape": shape,
            "k_demos": CONDITION_K[cond] if shape != "non_marker_demo" else 0,
            "n_probes": len(g_logps),
            "g_logprob": g_mean,
            "b_logprob": b_mean,
            "delta_g": delta_mean,
            "delta_g_std": delta_std,
            "emission_recompute_rate": emission_rate,
            "logp_floor": LOGP_FLOOR,
            "g_logps_per_q": g_logps,
            "b_logps_per_q": list(base["b_logps"]),
            "g_argmax_marker_per_q": g_argmax,
            "b_argmax_marker_per_q": list(base["b_argmax"]),
            "q_used": list(base["q_used"]),
        }
        tmp = cell_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(cell_payload))
        tmp.replace(cell_path)

        g_partial.setdefault(cond, {})[shape] = {
            "g_logprob": g_mean,
            "b_logprob": b_mean,
            "delta_g": delta_mean,
            "emission_recompute_rate": emission_rate,
            "n_probes": len(g_logps),
        }
        logger.info(
            "(%d/%d) (%s, %s) Delta G=%+.3f g=%.3f b=%.3f "
            "sd(Delta G)=%.3f emission=%.3f n=%d in %.1fs",
            idx + 1,
            len(all_cell_specs),
            cond,
            shape,
            delta_mean,
            g_mean,
            b_mean,
            delta_std,
            emission_rate,
            len(g_logps),
            elapsed,
        )

    roll_path = CROSS_EVAL_DIR / "G_partial.json"
    roll_path.write_text(json.dumps(g_partial, indent=2))
    logger.info("Phase 4 done. Roll-up -> %s", roll_path)

    # Co-primary retention (plan Must-Fix 5): per-cond retention = Delta G[demo_free_default]
    # ÷ Delta G[in_trained_shape]. Cell-mean version is reported here as a quick
    # roll-up; the per-q paired version (with bootstrap CIs) is in Phase 5.
    retention: dict[str, dict] = {}
    for cond, shapes in g_partial.items():
        if "in_trained_shape" not in shapes or "demo_free_default" not in shapes:
            continue
        d_in = shapes["in_trained_shape"]["delta_g"]
        d_default = shapes["demo_free_default"]["delta_g"]
        ratio = (d_default / d_in) if abs(d_in) > 1e-9 else None
        retention[cond] = {
            "delta_g_in_trained_shape": d_in,
            "delta_g_demo_free_default": d_default,
            "retention": ratio,
        }
    (OUT_DIR / "analysis_retention.json").write_text(json.dumps(retention, indent=2))
    logger.info("Retention roll-up -> %s", OUT_DIR / "analysis_retention.json")


if __name__ == "__main__":
    main()
