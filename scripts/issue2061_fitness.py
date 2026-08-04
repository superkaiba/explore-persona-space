"""P4 SAE-fitness gate for task #2061 (cross-stage confound control).

Plan §Design "Cross-stage SAE-fitness confound control": per-stage
FVE / L0 / dead-feature-count on the FIXED EleutherAI/sae-llama-3.1-8b-64x
dictionary applied to EACH stage's banked layer-29 activations. Same
LMSYS validation slice (~1k rows) per stage.

Recipe per `.claude/rules/gotchas.md` #1482 SAE token-pool entry:
- BOS-offset strip (first 8 positions).
- Outlier-norm filter (L2 > 10× pool median).
- Var-based FVE: `1 − var(x − x̂) / var(x)`, summed per-dim unbiased
  variance.

Pre-registered pass bar (plan §Design + §11):
- Base FVE reference = measured on the BASE stage's own activations.
- PASS: every post-training stage's FVE ≥ 0.8 × base_FVE.
- L0 ∈ [10, 200] for the fixed dictionary applied at each stage.
- Dead-feature fraction < 10% of d_sae.
- HARD DRIFT FLOOR: any stage's FVE < 0.5 × base_FVE → uninterpretable;
  the stage is EXCLUDED from the headline aggregate + carried as a
  scope caveat.

Emits `eval_results/issue_2061/fitness/<stage>_L29.json` per stage +
`eval_results/issue_2061/fitness/summary_L29.json` with the pass/fail
verdicts for all 5 stages.

Usage:
  uv run python scripts/issue2061_fitness.py --stage base
  uv run python scripts/issue2061_fitness.py --all-stages
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from explore_persona_space.analysis.sparsify_topk_sae import (
    load_sae_weights,
    topk_encode,
    topk_reconstruct,
)

LAYER = 29
STAGES = ["base", "sft", "dpo", "rlvr", "longer-rlvr"]
BOS_STRIP = 8  # first 8 positions per #1482 recipe
OUTLIER_L2_MEDIAN_MULT = 10.0
N_VAL_ROWS = 1000
DEAD_FEATURE_FRACTION_BAR = 0.10
L0_BAR_LO, L0_BAR_HI = 10, 200
FVE_PASS_FRACTION = 0.80  # PASS: post-training FVE ≥ 0.8 × base_FVE
FVE_HARD_FLOOR = 0.50  # HALT: FVE < 0.5 × base_FVE


def load_lmsys_validation_activations(
    stage: str,
    layer: int = LAYER,
    n_val_rows: int = N_VAL_ROWS,
    render: str = "chat",
    data_revision: str | None = None,
) -> torch.Tensor:
    """Load ~n_val_rows layer-`layer` context activations from the stage's
    LMSYS chat turnstore. Applies BOS-strip + outlier-norm filter per the
    #1482 recipe.
    """
    from huggingface_hub import hf_hub_download

    tree_path = f"issue1336_rlvr_ladder/analysis_tensors/turnstore_{stage}_{render}_lmsys23k"
    shard = hf_hub_download(
        repo_id="superkaiba1/explore-persona-space-data",
        filename=f"{tree_path}/turnstore_{stage}_{render}_lmsys23k_shard000.pt",
        repo_type="dataset",
        revision=data_revision,
    )
    obj = torch.load(shard, map_location="cpu", weights_only=True)
    if isinstance(obj, dict):
        for k in [f"context_L{layer}", f"context_layer_{layer}", "context"]:
            if k in obj:
                x = obj[k]
                if x.ndim == 3:
                    x = x[:, layer, :]
                break
        else:
            raise KeyError(f"No context-layer-{layer} key in {shard}")
    else:
        x = obj[:, layer, :] if obj.ndim == 3 else obj

    x = x.float()
    # BOS-strip (first 8 rows of the ordered pool, per #1482).
    if x.shape[0] > BOS_STRIP:
        x = x[BOS_STRIP:]
    # Outlier-norm filter: drop rows with L2 > 10× pool median.
    norms = torch.linalg.norm(x, dim=-1)
    median_norm = norms.median()
    keep = norms <= (OUTLIER_L2_MEDIAN_MULT * median_norm)
    x = x[keep]
    # Slice to n_val_rows.
    if x.shape[0] > n_val_rows:
        x = x[:n_val_rows]
    return x.contiguous()


def fve_l0_dead(
    x: torch.Tensor,
    weights: dict[str, torch.Tensor],
    k: int,
) -> tuple[float, float, float, int]:
    """Compute (FVE, L0_mean, dead_frac, n_rows) for the given activations.

    - FVE = 1 - var(x - x_recon) / var(x), per-dim unbiased variance sum.
    - L0_mean = mean over rows of ||z||_0 (should be exactly k for a
      well-behaved TopK encoder — this is a sanity check).
    - dead_frac = fraction of features NEVER activated across the pool
      (should be < 10% per the pass bar).
    """
    with torch.no_grad():
        z = topk_encode(x, weights, k=k)
        x_recon = topk_reconstruct(z, weights)

        numerator = (x - x_recon).var(dim=0, unbiased=True).sum().item()
        denominator = x.var(dim=0, unbiased=True).sum().item()
        fve = 1.0 - (numerator / denominator) if denominator > 0 else float("nan")

        l0 = (z != 0).float().sum(dim=-1).mean().item()
        # Any feature ever activated in the pool?
        activated_any = (z != 0).any(dim=0)
        dead_frac = 1.0 - activated_any.float().mean().item()

    return float(fve), float(l0), float(dead_frac), int(x.shape[0])


def evaluate_stage(
    stage: str,
    weights: dict[str, torch.Tensor],
    k: int,
    layer: int = LAYER,
    n_val_rows: int = N_VAL_ROWS,
    device: str = "cuda",
    data_revision: str | None = None,
) -> dict:
    """Compute FVE/L0/dead_frac for one stage on its LMSYS validation slice."""
    x = load_lmsys_validation_activations(
        stage,
        layer=layer,
        n_val_rows=n_val_rows,
        data_revision=data_revision,
    ).to(device)
    fve, l0, dead_frac, n = fve_l0_dead(x, weights, k=k)
    return {
        "stage": stage,
        "layer": layer,
        "n_rows_used": n,
        "fve": fve,
        "l0_mean": l0,
        "dead_feature_fraction": dead_frac,
        "l0_target": k,
    }


def compute_pass_verdicts(
    stage_results: dict[str, dict],
) -> dict:
    """Apply plan §Design pass bar per stage; require BASE result for the
    relative-FVE thresholds.
    """
    if "base" not in stage_results:
        raise ValueError("BASE stage result required as FVE reference.")
    base_fve = stage_results["base"]["fve"]
    pass_bar = FVE_PASS_FRACTION * base_fve
    hard_floor = FVE_HARD_FLOOR * base_fve

    verdicts = {"base_fve": base_fve, "pass_bar": pass_bar, "hard_floor": hard_floor}
    per_stage = {}
    for stage in STAGES:
        if stage not in stage_results:
            per_stage[stage] = {"status": "MISSING"}
            continue
        r = stage_results[stage]
        fve_ok = r["fve"] >= pass_bar
        fve_hard = r["fve"] >= hard_floor
        l0_ok = L0_BAR_LO <= r["l0_mean"] <= L0_BAR_HI
        dead_ok = r["dead_feature_fraction"] < DEAD_FEATURE_FRACTION_BAR

        if not fve_hard:
            status = "HALT_HARD_DRIFT"
        elif not fve_ok or not l0_ok or not dead_ok:
            status = "FAIL_SOFT"
        else:
            status = "PASS"
        per_stage[stage] = {
            "status": status,
            "fve": r["fve"],
            "fve_ratio_to_base": r["fve"] / base_fve if base_fve > 0 else float("nan"),
            "l0_mean": r["l0_mean"],
            "dead_feature_fraction": r["dead_feature_fraction"],
            "fve_ok": fve_ok,
            "fve_hard_ok": fve_hard,
            "l0_ok": l0_ok,
            "dead_ok": dead_ok,
        }
    verdicts["per_stage"] = per_stage
    # Overall: PASS iff every stage PASSes; HALT_HARD_DRIFT if any stage does; else FAIL_SOFT.
    statuses = [v.get("status", "MISSING") for v in per_stage.values()]
    if "HALT_HARD_DRIFT" in statuses:
        verdicts["overall"] = "HALT_HARD_DRIFT"
    elif "FAIL_SOFT" in statuses:
        verdicts["overall"] = "FAIL_SOFT_SOME_STAGE"
    elif all(s == "PASS" for s in statuses):
        verdicts["overall"] = "PASS"
    else:
        verdicts["overall"] = "MIXED"
    return verdicts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=str, choices=STAGES, default=None)
    parser.add_argument("--all-stages", action="store_true")
    parser.add_argument("--layer", type=int, default=LAYER)
    parser.add_argument("--n-val-rows", type=int, default=N_VAL_ROWS)
    parser.add_argument("--output-dir", type=Path, default=Path("eval_results/issue_2061/fitness"))
    parser.add_argument("--sae-revision", type=str, default=None)
    parser.add_argument("--data-revision", type=str, default=None)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[setup] Loading SAE weights layer={args.layer}")
    weights, cfg = load_sae_weights(
        "EleutherAI/sae-llama-3.1-8b-64x",
        layer=args.layer,
        revision=args.sae_revision,
        device=args.device,
    )
    k = int(cfg["k"])
    print(f"[setup] SAE k={k}, d_sae={weights['encoder.weight'].shape[0]}, d_in={cfg['d_in']}")

    if args.all_stages:
        target_stages = STAGES
    elif args.stage is not None:
        target_stages = [args.stage]
    else:
        print("[error] Use --stage or --all-stages")
        return 1

    stage_results = {}
    for stage in target_stages:
        print(f"\n=== Fitness for stage: {stage} ===")
        try:
            result = evaluate_stage(
                stage=stage,
                weights=weights,
                k=k,
                layer=args.layer,
                n_val_rows=args.n_val_rows,
                device=args.device,
                data_revision=args.data_revision,
            )
        except Exception as e:
            print(f"[error] Stage {stage} failed: {e}")
            stage_results[stage] = {"stage": stage, "error": str(e)}
            continue
        stage_results[stage] = result
        out_path = args.output_dir / f"{stage}_L{args.layer}.json"
        with out_path.open("w") as f:
            json.dump(result, f, indent=2)
        print(
            f"[stage {stage}] FVE={result['fve']:.4f} L0={result['l0_mean']:.1f} "
            f"dead={result['dead_feature_fraction']:.3f} n={result['n_rows_used']}"
        )
        print(f"[write] {out_path}")

    # Summary requires the base stage as the FVE reference.
    if "base" in stage_results and "error" not in stage_results.get("base", {}):
        verdicts = compute_pass_verdicts(stage_results)
        summary_path = args.output_dir / f"summary_L{args.layer}.json"
        with summary_path.open("w") as f:
            json.dump(verdicts, f, indent=2)
        print(f"\n[summary] Overall: {verdicts['overall']}")
        print(
            f"[summary] base_FVE={verdicts['base_fve']:.4f} "
            f"pass_bar={verdicts['pass_bar']:.4f} hard_floor={verdicts['hard_floor']:.4f}"
        )
        for stage, v in verdicts["per_stage"].items():
            if v.get("status") in {"PASS", "FAIL_SOFT", "HALT_HARD_DRIFT"}:
                print(
                    f"  {stage:12s} {v['status']:20s} FVE={v['fve']:.4f} "
                    f"(ratio={v['fve_ratio_to_base']:.3f})"
                )
        print(f"[write] {summary_path}")
    else:
        print("\n[summary] BASE stage missing/errored — cannot compute relative-FVE verdicts")

    return 0


if __name__ == "__main__":
    sys.exit(main())
