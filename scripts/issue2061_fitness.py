"""P4 SAE-fitness gate for task #2061 (cross-stage confound control).

Plan §Design "Cross-stage SAE-fitness confound control": per-stage
FVE / L0 / dead-feature-count on the FIXED EleutherAI/sae-llama-3.1-8b-64x
dictionary applied to EACH stage's banked layer-29 activations. Same
LMSYS validation slice (~1k rows) per stage.

Recipe per `.claude/rules/gotchas.md` #1482 SAE token-pool entry, adapted
to this pool's grain:
- BOS-offset strip: **N/A here (deliberately not applied)** — the #1482
  strip removes the first 8 TOKEN POSITIONS of a TOKEN pool (the
  Llama/Qwen massive-activation positions adjacent to BOS). This pool is
  per-CONVERSATION pooled rows (a1 answer-turn span-means / header slot
  states), so no BOS-adjacent token position can enter any row by
  construction, and #1336 banks no token-level pool to strip. A row-level
  strip would just delete the first 8 conversations — vacuous as a
  massive-activation guard (unit-D resolution of the unit-A carried note).
- Outlier-norm filter (L2 > 10× pool median) — applied unchanged.
- Var-based FVE: `1 − var(x − x̂) / var(x)`, summed per-dim unbiased
  variance — applied unchanged.

Pre-registered pass bar (plan §Design + §7 + §11):
- Base FVE reference = measured on the BASE stage's own activations.
- PASS: every post-training stage's FVE ≥ 0.8 × base_FVE.
- L0 ∈ [10, 200] for the fixed dictionary applied at each stage.
- Dead-feature fraction < 10% of d_sae.
- HARD DRIFT FLOOR (plan §7 "Halt and report (SAE-fitness gate)"): any
  stage's FVE < 0.5 × base_FVE → uninterpretable; the stage is EXCLUDED
  from the headline aggregate + carried as a scope caveat.

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

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()  # shared-VM thread caps (#847) must bind BEFORE torch/numpy import

import torch  # noqa: E402

from explore_persona_space.analysis.sparsify_topk_sae import (  # noqa: E402
    load_sae_weights,
    topk_encode,
    topk_reconstruct,
)

# Sibling-script imports (bare module names via the script-dir sys.path
# insert — the issue1336_extract_turnstore.py pattern; works in script mode
# AND under the tests' `sys.path.insert(scripts)` import). BANKED_PREFIX +
# the lazy shard-download generator are owned by the P1 encode script (one
# source of truth for the hub layout); the payload schema/extraction lives
# in issue2061_turnstore.
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import issue2061_turnstore as ts  # noqa: E402
from issue2061_sae_encode import iter_local_shards, resolve_turnstore_tree  # noqa: E402

LAYER = 29
STAGES = ["base", "sft", "dpo", "rlvr", "longer-rlvr"]
OUTLIER_L2_MEDIAN_MULT = 10.0
N_VAL_ROWS = 1000
DEAD_FEATURE_FRACTION_BAR = 0.10
L0_BAR_LO, L0_BAR_HI = 10, 200
FVE_PASS_FRACTION = 0.80  # PASS: post-training FVE ≥ 0.8 × base_FVE (plan §Design)
# Plan §7 kill criteria, "Halt and report (SAE-fitness gate)": FVE < 0.5 ×
# base_FVE is a hard drift floor beyond the graded 0.8 bar — the stage's
# ridge fit is uninterpretable against the fixed dictionary and is excluded
# from the headline aggregate (registered in v5/v6 §7, not §Design).
FVE_HARD_FLOOR = 0.50


def load_lmsys_validation_activations(
    stage: str,
    layer: int = LAYER,
    n_val_rows: int = N_VAL_ROWS,
    render: str = "chat",
    state: str = "answer",
    data_revision: str | None = None,
) -> torch.Tensor:
    """~n_val_rows pooled layer-`layer` rows from the stage's LMSYS turnstore.

    Enumerates ALL `*_shardNNN.pt` files of the turnstore (lazy download via
    `issue2061_sae_encode.iter_local_shards` — only as many shards as needed
    are fetched) and extracts one banked state per row via
    `issue2061_turnstore.extract_state_rows` (fail-loud schema assert; real
    #1336 write_shards payload). Default `state="answer"` — the a1
    answer-profile rows, i.e. the SAME pool this pipeline SAE-encodes as the
    target Y (plan §Design), which is the pool whose cross-stage SAE fitness
    this P4 gate exists to control.

    The #1482 recipe's BOS-strip is deliberately NOT applied (unit-D
    resolution of the unit-A carried note): it strips the first 8 TOKEN
    POSITIONS of a TOKEN pool, and this pool is per-CONVERSATION pooled rows
    whose span-means/slot states exclude BOS-adjacent positions by
    construction (no token-level pool is banked). A row-level strip would
    just delete the first 8 conversations. The recipe's outlier-norm filter
    below IS applied unchanged (well-defined on rows).
    """
    # Resolve the REALIZED tree name (never hand-build it: lmsys23k lives
    # under the store's `v2_` capture-generation prefix and the 5th stage is
    # realized `rlvr_long` — a hand-built canonical name 404s; unit-E live
    # probe finding, see issue2061_sae_encode.STORE_STAGE_TOKENS).

    tree_path = resolve_turnstore_tree(stage, render, "lmsys23k", revision=data_revision)
    # Margin above the target so outlier drops still leave ~n_val_rows.
    x, _conv_ids = ts.load_state_from_shards(
        iter_local_shards(tree_path, revision=data_revision),
        state=state,
        layer=layer,
        max_rows=n_val_rows + 40,
    )
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
    state: str = "answer",
    data_revision: str | None = None,
) -> dict:
    """Compute FVE/L0/dead_frac for one stage on its LMSYS validation slice."""
    x = load_lmsys_validation_activations(
        stage,
        layer=layer,
        n_val_rows=n_val_rows,
        state=state,
        data_revision=data_revision,
    ).to(device)
    fve, l0, dead_frac, n = fve_l0_dead(x, weights, k=k)
    return {
        "stage": stage,
        "layer": layer,
        "state": state,
        "n_rows_used": n,
        "fve": fve,
        "l0_mean": l0,
        "dead_feature_fraction": dead_frac,
        "l0_target": k,
        # Verifier-legible provenance of the #1482 token-pool recipe legs as
        # realized on this per-conversation pooled-row pool (module docstring).
        "recipe": {
            "bos_strip": (
                "n/a — per-conversation pooled rows; the #1482 strip removes "
                "BOS-adjacent TOKEN positions of a token pool, and no token-level "
                "pool is banked (deliberately not applied)"
            ),
            "outlier_filter": f"rows with L2 > {OUTLIER_L2_MEDIAN_MULT}x pool median dropped",
            "fve": "var-based: 1 - var(x - x_hat)/var(x), per-dim unbiased, summed",
        },
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
    parser.add_argument(
        "--state",
        type=str,
        choices=sorted(ts.STATE_SPEC),
        default="answer",
        help="Which banked state to evaluate (default: answer — the pool the pipeline encodes).",
    )
    parser.add_argument("--n-val-rows", type=int, default=N_VAL_ROWS)
    parser.add_argument("--output-dir", type=Path, default=Path("eval_results/issue_2061/fitness"))
    parser.add_argument("--sae-revision", type=str, default=None)
    parser.add_argument("--data-revision", type=str, default=None)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        help="After the verdicts, upload --output-dir to the HF data repo "
        "(analysis_tensors/fitness/) — declared v6 in plan §9 off_pod_phases: "
        "the fitness JSONs ride an ephemeral eval pod (the #1738 fit-summary-"
        "JSON loss class) and P5 reads this prefix.",
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
                state=args.state,
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

    if args.upload:
        # Upload runs UNCONDITIONALLY on the flag — a HALT_HARD_DRIFT verdict
        # is a report artifact that must persist BEFORE the pod terminates.
        import issue2061_hub_io as hio

        hio.upload_dir(args.output_dir, "fitness")

    return 0


if __name__ == "__main__":
    sys.exit(main())
