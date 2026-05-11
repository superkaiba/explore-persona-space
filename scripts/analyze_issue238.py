#!/usr/bin/env python3
"""Issue #238 analysis -- Full-parameter SFT vs LoRA persona geometry comparison.

Loads persona vector centroids from:
  - Base model (from #205 extraction)
  - 4 full-param conditions (from this experiment)
  - 2 LoRA baselines (from #205 run_result.json)

Computes:
  1. M1 (mean off-diagonal cos-sim) for each condition x layer x method
  2. Delta = M1_post - M1_base for each cell
  3. Delta ratio = delta_full / delta_lora for each (layer, method, data_type)
  4. Weight-delta norms (global + per-extraction-layer)
  5. BH-FDR correction on pre/post significance tests
  6. Paired permutation test for cross-condition comparison
  7. H1/H2/H3 classification per layer

Writes eval_results/issue_238/run_result.json.

Usage:
    uv run python scripts/analyze_issue238.py
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import time
from datetime import UTC, datetime
from itertools import combinations
from pathlib import Path

import numpy as np
import torch

# ── Constants ────────────────────────────────────────────────────────────────

REPO = Path(__file__).resolve().parent.parent
WORK = Path("/workspace/issue238")

BASE_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
PERSONA_VECTORS_DIR = REPO / "data" / "persona_vectors" / "qwen2.5-7b-instruct"
ISSUE_205_RESULTS = REPO / "eval_results" / "issue_205" / "run_result.json"
OUTPUT_DIR = REPO / "eval_results" / "issue_238"

EVAL_PERSONAS = [
    "software_engineer",
    "kindergarten_teacher",
    "data_scientist",
    "medical_doctor",
    "librarian",
    "french_person",
    "villain",
    "comedian",
    "police_officer",
    "zelthari_scholar",
    "assistant",
    "confab",
]
LAYERS = [7, 14, 20, 21, 27]
METHODS = ["A", "B"]
N_PERMUTATIONS = 10000
FDR_ALPHA = 0.01

# Full-param conditions from this experiment
FULL_PARAM_CONDITIONS = [
    {"name": "full_em_lr2e5", "data_type": "em", "lr": 2e-5},
    {"name": "full_benign_lr2e5", "data_type": "benign", "lr": 2e-5},
    {"name": "full_em_lr1e4", "data_type": "em", "lr": 1e-4},
    {"name": "full_benign_lr1e4", "data_type": "benign", "lr": 1e-4},
]

# LoRA baseline keys from #205 (E0 = assistant persona, benign_sft_375)
LORA_BASELINE_MAP = {
    "em": "E0_assistant",
    "benign": "benign_sft_375",
}

log = logging.getLogger("analyze_issue238")


# ── Helpers ──────────────────────────────────────────────────────────────────


def git_commit() -> str:
    """Return current git commit hash."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(REPO), text=True
        ).strip()
    except Exception:
        return "unknown"


def load_centroids(vectors_dir: Path, method: str) -> dict[str, torch.Tensor] | None:
    """Load all_centroids.pt for a given method (A or B).

    Returns dict mapping persona_name -> (n_layers, hidden_dim) tensor,
    or None if not found.
    """
    method_key = "method_a" if method == "A" else "method_b"
    centroid_path = vectors_dir / method_key / "all_centroids.pt"
    if not centroid_path.exists():
        log.warning("Centroids not found: %s", centroid_path)
        return None
    return torch.load(centroid_path, weights_only=True, map_location="cpu")


def compute_pairwise_cossim(
    centroids: dict[str, torch.Tensor],
    personas: list[str],
    layer_idx: int,
    layers: list[int],
) -> np.ndarray:
    """Compute all pairwise cosine similarities at a given layer.

    Returns array of shape (n_pairs,) for the 66 off-diagonal pairs
    (12 choose 2).
    """
    layer_pos = layers.index(layer_idx)

    vecs = []
    for p in personas:
        if p not in centroids:
            raise KeyError(f"Persona '{p}' not in centroids (have: {sorted(centroids.keys())})")
        v = centroids[p][layer_pos].float()
        vecs.append(v / v.norm())
    vecs = torch.stack(vecs)

    sim_matrix = vecs @ vecs.T
    pairs = []
    for i, j in combinations(range(len(personas)), 2):
        pairs.append(sim_matrix[i, j].item())
    return np.array(pairs)


def mean_offdiag_cossim(
    centroids: dict[str, torch.Tensor],
    personas: list[str],
    layer_idx: int,
    layers: list[int],
) -> tuple[float, np.ndarray]:
    """Compute mean off-diagonal cos-sim and return (mean, pair_values)."""
    pairs = compute_pairwise_cossim(centroids, personas, layer_idx, layers)
    return float(np.mean(pairs)), pairs


def permutation_test_paired(
    base_pairs: np.ndarray,
    post_pairs: np.ndarray,
    n_permutations: int = 10000,
    seed: int = 42,
) -> float:
    """Paired permutation test: is post mean > base mean?

    Tests whether the mean difference (post - base) is significantly positive.
    Returns one-sided p-value.
    """
    rng = np.random.RandomState(seed)
    diffs = post_pairs - base_pairs
    observed = np.mean(diffs)

    count = 0
    for _ in range(n_permutations):
        signs = rng.choice([-1, 1], size=len(diffs))
        perm_mean = np.mean(diffs * signs)
        if perm_mean >= observed:
            count += 1
    return count / n_permutations


def bh_fdr_correction(p_values: list[float], alpha: float = 0.01) -> list[dict]:
    """Apply Benjamini-Hochberg FDR correction.

    Returns list of dicts with p_raw, p_bh_fdr, and sig flag.
    """
    n = len(p_values)
    if n == 0:
        return []

    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    results = [None] * n
    prev_bh = 0.0
    for rank, (orig_idx, p_raw) in enumerate(indexed, 1):
        p_bh = min(p_raw * n / rank, 1.0)
        p_bh = max(p_bh, prev_bh)
        prev_bh = p_bh
        results[orig_idx] = {
            "p_raw": p_raw,
            "p_bh_fdr": p_bh,
            "sig_bh_fdr": p_bh < alpha,
        }
    return results


# ── Weight-Delta Norms ───────────────────────────────────────────────────────


def compute_weight_delta_norms(
    base_path: str,
    checkpoint_path: str,
    layers: list[int],
) -> dict:
    """Compute L2 norm of weight delta between base and checkpoint.

    Returns dict with:
      - global_l2: total L2 norm across all parameters
      - per_layer: {layer_idx: L2 norm for that transformer layer}
    """
    from transformers import AutoModelForCausalLM

    log.info(
        "Computing weight-delta norms: base=%s, checkpoint=%s",
        base_path,
        checkpoint_path,
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        base_path, torch_dtype=torch.bfloat16, device_map="cpu"
    )
    ckpt_model = AutoModelForCausalLM.from_pretrained(
        checkpoint_path, torch_dtype=torch.bfloat16, device_map="cpu"
    )

    base_sd = base_model.state_dict()
    ckpt_sd = ckpt_model.state_dict()

    # Global L2 norm
    global_sq_sum = 0.0
    for key in base_sd:
        if key in ckpt_sd:
            diff = (ckpt_sd[key].float() - base_sd[key].float()).flatten()
            global_sq_sum += (diff * diff).sum().item()
    global_l2 = float(np.sqrt(global_sq_sum))

    # Per-extraction-layer L2 norms
    per_layer = {}
    for layer_idx in layers:
        layer_prefix = f"model.layers.{layer_idx}."
        layer_sq_sum = 0.0
        for key in base_sd:
            if key.startswith(layer_prefix) and key in ckpt_sd:
                diff = (ckpt_sd[key].float() - base_sd[key].float()).flatten()
                layer_sq_sum += (diff * diff).sum().item()
        per_layer[layer_idx] = float(np.sqrt(layer_sq_sum))

    del base_model, ckpt_model, base_sd, ckpt_sd
    torch.cuda.empty_cache()

    log.info("Weight-delta norms: global=%.4f", global_l2)
    for layer_idx, norm in sorted(per_layer.items()):
        log.info("  Layer %d: %.4f", layer_idx, norm)

    return {"global_l2": global_l2, "per_layer": per_layer}


# ── Analysis Phases (extracted from main for complexity) ─────────────────────


def load_lora_baselines() -> dict[tuple, float]:
    """Load LoRA baseline deltas from #205 run_result.json."""
    if not ISSUE_205_RESULTS.exists():
        raise FileNotFoundError(
            f"Issue #205 results not found at {ISSUE_205_RESULTS}. "
            "Ensure eval_results/issue_205/run_result.json exists."
        )
    with open(ISSUE_205_RESULTS) as f:
        issue205 = json.load(f)
    issue205_results = issue205["results"]

    lora_deltas = {}
    for method in METHODS:
        for layer in LAYERS:
            for data_type, cond_name in LORA_BASELINE_MAP.items():
                key = f"M1_{method}_L{layer}_{cond_name}"
                if key in issue205_results:
                    delta = issue205_results[key]["delta_mean_offdiag"]
                    lora_deltas[(method, layer, data_type)] = delta
                    log.info(
                        "LoRA baseline: M=%s L=%d %s -> delta=%.5f",
                        method,
                        layer,
                        data_type,
                        delta,
                    )
                else:
                    log.warning("Missing LoRA baseline key: %s", key)
    return lora_deltas


def load_all_centroids() -> tuple[dict, dict]:
    """Load base and full-param centroids. Returns (base, fullparam) dicts."""
    log.info("Loading base centroids")
    base_centroids = {}
    for method in METHODS:
        c = load_centroids(PERSONA_VECTORS_DIR / "base", method)
        if c is None:
            raise FileNotFoundError(
                f"Base centroids for method {method} not found at {PERSONA_VECTORS_DIR / 'base'}"
            )
        base_centroids[method] = c

    log.info("Loading full-param centroids for 4 conditions")
    fullparam_centroids = {}
    for cond in FULL_PARAM_CONDITIONS:
        cond_dir = PERSONA_VECTORS_DIR / cond["name"]
        cond_centroids = {}
        for method in METHODS:
            c = load_centroids(cond_dir, method)
            if c is None:
                raise FileNotFoundError(
                    f"Centroids for {cond['name']} method {method} not found at {cond_dir}"
                )
            cond_centroids[method] = c
        fullparam_centroids[cond["name"]] = cond_centroids

    return base_centroids, fullparam_centroids


def compute_base_m1(
    base_centroids: dict,
) -> tuple[dict, dict]:
    """Compute M1 for base model. Returns (base_m1, base_pairs) dicts."""
    log.info("Computing M1 for base model")
    base_m1 = {}
    base_pairs = {}
    for method in METHODS:
        for layer in LAYERS:
            mean_val, pairs = mean_offdiag_cossim(
                base_centroids[method], EVAL_PERSONAS, layer, LAYERS
            )
            base_m1[(method, layer)] = mean_val
            base_pairs[(method, layer)] = pairs
            log.info("Base M1: M=%s L=%d -> %.6f", method, layer, mean_val)
    return base_m1, base_pairs


def compute_fullparam_m1(
    fullparam_centroids: dict,
    base_m1: dict,
    base_pairs: dict,
) -> tuple[dict, list[float], list[str]]:
    """Compute M1 + deltas + permutation tests for full-param conditions.

    Returns (results_dict, p_values_list, p_keys_list).
    """
    log.info("Computing M1 + deltas for full-param conditions")
    results = {}
    all_p_values = []
    all_p_keys = []

    for cond in FULL_PARAM_CONDITIONS:
        for method in METHODS:
            for layer in LAYERS:
                mean_val, pairs = mean_offdiag_cossim(
                    fullparam_centroids[cond["name"]][method],
                    EVAL_PERSONAS,
                    layer,
                    LAYERS,
                )
                delta = mean_val - base_m1[(method, layer)]
                p_raw = permutation_test_paired(base_pairs[(method, layer)], pairs)

                key = f"M1_{method}_L{layer}_{cond['name']}"
                results[key] = {
                    "metric": "M1_cos_collapse",
                    "method": method,
                    "layer": layer,
                    "condition": cond["name"],
                    "data_type": cond["data_type"],
                    "lr": cond["lr"],
                    "training_method": "full_parameter",
                    "base_mean": base_m1[(method, layer)],
                    "post_mean": mean_val,
                    "delta_mean_offdiag": delta,
                    "p_raw": p_raw,
                    "family": "geometry_fullparam_pre_post",
                }
                all_p_values.append(p_raw)
                all_p_keys.append(key)

                log.info(
                    "Full-param M1: %s M=%s L=%d -> mean=%.6f delta=%.6f p=%.4f",
                    cond["name"],
                    method,
                    layer,
                    mean_val,
                    delta,
                    p_raw,
                )

    return results, all_p_values, all_p_keys


def compute_cross_condition_comparisons(
    results: dict,
    fullparam_centroids: dict,
    base_pairs: dict,
    lora_deltas: dict,
) -> None:
    """Add cross-condition (full-param vs LoRA) comparisons to results."""
    log.info("Running cross-condition comparisons (full-param vs LoRA)")
    for cond in FULL_PARAM_CONDITIONS:
        for method in METHODS:
            for layer in LAYERS:
                lora_key = (method, layer, cond["data_type"])
                if lora_key not in lora_deltas:
                    continue

                _, full_pairs = mean_offdiag_cossim(
                    fullparam_centroids[cond["name"]][method],
                    EVAL_PERSONAS,
                    layer,
                    LAYERS,
                )
                full_diffs = full_pairs - base_pairs[(method, layer)]
                lora_delta = lora_deltas[lora_key]

                cross_key = f"cross_{method}_L{layer}_{cond['name']}_vs_lora"
                results[cross_key] = {
                    "metric": "fullparam_vs_lora_delta",
                    "method": method,
                    "layer": layer,
                    "condition": cond["name"],
                    "data_type": cond["data_type"],
                    "lr": cond["lr"],
                    "delta_full": float(np.mean(full_diffs)),
                    "delta_lora": lora_delta,
                    "delta_ratio": (
                        float(np.mean(full_diffs)) / lora_delta if lora_delta != 0 else float("inf")
                    ),
                }


def compute_delta_ratios(
    results: dict,
    lora_deltas: dict,
) -> dict:
    """Compute delta ratio table (full-param / LoRA) for both LR pairs."""
    log.info("Computing delta ratio table (full-param / LoRA)")
    delta_ratios = {}
    for method in METHODS:
        for layer in LAYERS:
            for data_type in ["em", "benign"]:
                lora_key_tup = (method, layer, data_type)
                if lora_key_tup not in lora_deltas:
                    continue
                lora_delta = lora_deltas[lora_key_tup]

                # Primary: lr=2e-5
                primary = "full_em_lr2e5" if data_type == "em" else "full_benign_lr2e5"
                m1_key = f"M1_{method}_L{layer}_{primary}"
                if m1_key in results:
                    full_delta = results[m1_key]["delta_mean_offdiag"]
                    ratio = full_delta / lora_delta if lora_delta != 0 else float("inf")
                    rk = f"ratio_{method}_L{layer}_{data_type}_2e5"
                    delta_ratios[rk] = {
                        "method": method,
                        "layer": layer,
                        "data_type": data_type,
                        "lr": 2e-5,
                        "delta_full": full_delta,
                        "delta_lora": lora_delta,
                        "ratio": ratio,
                    }
                    log.info(
                        "Delta ratio (2e-5): M=%s L=%d %s -> %.3f (full=%.5f / lora=%.5f)",
                        method,
                        layer,
                        data_type,
                        ratio,
                        full_delta,
                        lora_delta,
                    )

                # Control: lr=1e-4
                control = "full_em_lr1e4" if data_type == "em" else "full_benign_lr1e4"
                ctrl_key = f"M1_{method}_L{layer}_{control}"
                if ctrl_key in results:
                    ctrl_delta = results[ctrl_key]["delta_mean_offdiag"]
                    ctrl_ratio = ctrl_delta / lora_delta if lora_delta != 0 else float("inf")
                    rk_ctrl = f"ratio_{method}_L{layer}_{data_type}_1e4"
                    delta_ratios[rk_ctrl] = {
                        "method": method,
                        "layer": layer,
                        "data_type": data_type,
                        "lr": 1e-4,
                        "delta_full": ctrl_delta,
                        "delta_lora": lora_delta,
                        "ratio": ctrl_ratio,
                    }

    return delta_ratios


def classify_hypotheses(delta_ratios: dict) -> dict:
    """Classify H1/H2/H3 per (method, data_type) based on delta ratios."""
    log.info("Classifying H1/H2/H3 per layer")
    verdicts = {}
    for method in METHODS:
        for data_type in ["em", "benign"]:
            h1_count = 0
            h3_count = 0
            for layer in LAYERS:
                rk = f"ratio_{method}_L{layer}_{data_type}_2e5"
                if rk not in delta_ratios:
                    continue
                ratio = delta_ratios[rk]["ratio"]
                if ratio < 0.5:
                    h1_count += 1
                elif ratio > 1.5:
                    h3_count += 1

            if h1_count >= 3:
                verdict, desc = "H1", ("LoRA is the culprit (full-param preserves geometry better)")
            elif h3_count >= 3:
                verdict, desc = "H3", ("Full-param is worse (LoRA regularizes)")
            else:
                verdict, desc = "H2", ("Generic collapse (method-independent)")

            vk = f"verdict_{method}_{data_type}"
            verdicts[vk] = {
                "method": method,
                "data_type": data_type,
                "verdict": verdict,
                "description": desc,
                "h1_layers": h1_count,
                "h3_layers": h3_count,
                "total_layers": len(LAYERS),
            }
            log.info(
                "Verdict: M=%s %s -> %s (%s) [H1:%d H3:%d / %d layers]",
                method,
                data_type,
                verdict,
                desc,
                h1_count,
                h3_count,
                len(LAYERS),
            )
    return verdicts


def compute_all_weight_norms() -> dict:
    """Compute weight-delta norms for all full-param checkpoints."""
    log.info("Computing weight-delta norms for full-param checkpoints")
    weight_norms = {}
    for cond in FULL_PARAM_CONDITIONS:
        checkpoint_dir = WORK / cond["name"] / "final_checkpoint"
        if not checkpoint_dir.exists():
            log.warning(
                "Checkpoint not found for %s at %s -- skipping weight norms",
                cond["name"],
                checkpoint_dir,
            )
            continue
        norms = compute_weight_delta_norms(BASE_MODEL_ID, str(checkpoint_dir), LAYERS)
        weight_norms[cond["name"]] = norms
    return weight_norms


def print_summary(
    delta_ratios: dict,
    hypothesis_verdicts: dict,
    weight_norms: dict,
) -> None:
    """Print a summary table of results."""
    log.info("\n" + "=" * 80)
    log.info("SUMMARY: Delta ratios (full-param@2e-5 / LoRA)")
    log.info(
        "%-8s %-8s %-10s %-12s %-12s %-8s",
        "Method",
        "Layer",
        "DataType",
        "Full",
        "LoRA",
        "Ratio",
    )
    log.info("-" * 80)
    for method in METHODS:
        for data_type in ["em", "benign"]:
            for layer in LAYERS:
                rk = f"ratio_{method}_L{layer}_{data_type}_2e5"
                if rk in delta_ratios:
                    r = delta_ratios[rk]
                    log.info(
                        "%-8s %-8d %-10s %-12.5f %-12.5f %-8.3f",
                        method,
                        layer,
                        data_type,
                        r["delta_full"],
                        r["delta_lora"],
                        r["ratio"],
                    )

    log.info("\nHypothesis verdicts:")
    for vk, vv in sorted(hypothesis_verdicts.items()):
        log.info("  %s: %s -- %s", vk, vv["verdict"], vv["description"])

    if weight_norms:
        log.info("\nWeight-delta norms (global L2):")
        for name, norms in sorted(weight_norms.items()):
            log.info("  %s: %.4f", name, norms["global_l2"])


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    os.environ.setdefault("HF_HOME", "/workspace/.cache/huggingface")
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    )
    t0 = time.time()

    log.info("=" * 60)
    log.info("Issue #238 Analysis: Full-param vs LoRA geometry comparison")
    log.info("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Phase 1: Load data
    lora_deltas = load_lora_baselines()
    base_centroids, fullparam_centroids = load_all_centroids()
    base_m1, base_pairs = compute_base_m1(base_centroids)

    # Phase 2: Compute M1 + permutation tests
    results, all_p_values, all_p_keys = compute_fullparam_m1(
        fullparam_centroids, base_m1, base_pairs
    )

    # Phase 3: BH-FDR correction
    log.info("Applying BH-FDR correction (%d tests)", len(all_p_values))
    fdr_results = bh_fdr_correction(all_p_values, alpha=FDR_ALPHA)
    for i, key in enumerate(all_p_keys):
        results[key]["p_bh_fdr"] = fdr_results[i]["p_bh_fdr"]
        results[key]["sig_bh_fdr"] = fdr_results[i]["sig_bh_fdr"]

    # Phase 4: Cross-condition comparisons
    compute_cross_condition_comparisons(results, fullparam_centroids, base_pairs, lora_deltas)

    # Phase 5: Delta ratios + hypothesis classification
    delta_ratios = compute_delta_ratios(results, lora_deltas)
    hypothesis_verdicts = classify_hypotheses(delta_ratios)

    # Phase 6: Weight-delta norms
    weight_norms = compute_all_weight_norms()

    # Phase 7: Write output
    log.info("Writing results to %s", OUTPUT_DIR / "run_result.json")
    output = {
        "issue": 238,
        "experiment": "issue_238_fullparam_vs_lora_geometry",
        "git_commit": git_commit(),
        "timestamp": datetime.now(UTC).isoformat(),
        "seed": 42,
        "wall_time_seconds": time.time() - t0,
        "base_model": BASE_MODEL_ID,
        "eval_personas": EVAL_PERSONAS,
        "layers": LAYERS,
        "methods": METHODS,
        "n_permutations": N_PERMUTATIONS,
        "fdr_alpha": FDR_ALPHA,
        "full_param_conditions": [c["name"] for c in FULL_PARAM_CONDITIONS],
        "lora_baseline_source": "issue_205",
        "lora_baseline_map": LORA_BASELINE_MAP,
        "base_m1": {f"M{method}_L{layer}": val for (method, layer), val in sorted(base_m1.items())},
        "results": results,
        "delta_ratios": delta_ratios,
        "hypothesis_verdicts": hypothesis_verdicts,
        "weight_delta_norms": weight_norms,
        "lora_deltas_from_205": {
            f"M{method}_L{layer}_{dtype}": delta
            for (method, layer, dtype), delta in sorted(lora_deltas.items())
        },
    }

    with open(OUTPUT_DIR / "run_result.json", "w") as f:
        json.dump(output, f, indent=2)
    log.info("Results written successfully")

    print_summary(delta_ratios, hypothesis_verdicts, weight_norms)
    log.info("\nAnalysis complete in %.1f seconds", time.time() - t0)


if __name__ == "__main__":
    main()
