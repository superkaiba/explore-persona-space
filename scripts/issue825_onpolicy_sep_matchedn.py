"""Issue #825 `onpolicy-separator-control` G4b: conditional matched-n W_ex re-baseline.

Held-out R^2 in this rig is steeply n-dependent (plan section 1 binding
amendment), so when a model's realized on-policy pair count ``n_r`` falls
below ``0.97 x 3600``, the exogenous reference ``W_ex`` is re-baselined at the
MATCHED n: group-stratified subsample the STAGED exogenous armC store rows to
``n_r`` (the round-6 ``group_stratified_subsample`` convention; matched
per-group counts where possible), seeds 931..935 (>= 5 seeds), refit the
ROTATED estimator at the headline layer per seed (the same
``fit825.random_projection_control`` path behind the committed values) plus
the batched MLP secondary at the frozen layers, and take
``w_ex_matched_n = max(seed-mean rotated @ L19, seed-mean MLP @ L19)`` — the
committed max(rotated, MLP) convention on the seed-mean refits.

When the trigger does NOT fire (``n_r >= 0.97 x target``) the script writes a
``trigger_fired: false`` record and runs no refits (the frozen full-n W_ex
stays the D reference).

CLI:
  uv run python scripts/issue825_onpolicy_sep_matchedn.py --model base \
      --anchor-store-dir <anchor>/store/armC --realized-n <n_r> \
      --out <out>/anchor_base/matched_n_wex_base.json
"""

from __future__ import annotations

import argparse
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()  # thread caps bind before torch/numpy import

import sys  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

import issue825_fit_cells as fit825  # noqa: E402
import issue931_common as common  # noqa: E402
import issue931_fit_cells as fit931  # noqa: E402

SCRIPT = "scripts/issue825_onpolicy_sep_matchedn.py"
DEFAULT_SEEDS = (931, 932, 933, 934, 935)
TRIGGER_FRAC = 0.97


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True, choices=("base", "instruct"))
    ap.add_argument("--anchor-store-dir", type=Path, required=True, help="<anchor>/store/armC")
    ap.add_argument("--realized-n", type=int, required=True, help="on-policy realized pair count")
    ap.add_argument("--target-n", type=int, default=3600)
    ap.add_argument("--trigger-frac", type=float, default=TRIGGER_FRAC)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seeds", type=str, default=",".join(str(s) for s in DEFAULT_SEEDS))
    ap.add_argument("--folds", type=int, default=common.N_FOLDS)
    ap.add_argument("--fit-seed", type=int, default=common.FIT_SEED)
    ap.add_argument("--skip-mlp", action="store_true", help="rotated-only (smoke speed)")
    ap.add_argument("--smoke", action="store_true", help="values recorded, non-binding")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    assert len(seeds) >= (2 if args.smoke else 5), f">=5 seeds required (got {seeds})"
    threshold = args.trigger_frac * args.target_n
    fired = args.realized_n < threshold
    base = {
        "metadata": common.metadata(SCRIPT, args.fit_seed, args.realized_n),
        "followup_label": "onpolicy-separator-control",
        "model": args.model,
        "realized_n": int(args.realized_n),
        "target_n": int(args.target_n),
        "trigger_frac": args.trigger_frac,
        "trigger_threshold": threshold,
        "trigger_fired": bool(fired),
        "subsample": "group_stratified_subsample over article groups (round-6 convention)",
        "seeds": seeds,
        "smoke": bool(args.smoke),
    }
    if not fired:
        common.write_json(args.out, {**base, "w_ex_matched_n": None})
        print(
            f"[i825-ops-mn] {args.model}: n_r={args.realized_n} >= {threshold:.0f} — "
            "trigger NOT fired; full-n W_ex stays the D reference"
        )
        return 0

    store = fit931.load_regime_store(args.anchor_store_dir, "armC")
    X, Y = store["arrays"]["x_sep"], store["arrays"]["y"]
    groups = store["group_ids"]
    n, n_layers = X.shape[0], X.shape[1]
    hl = fit931.headline_layer(n_layers)
    fl = fit931.frozen_layers(n_layers)
    n_sub = min(args.realized_n, n)
    print(
        f"[i825-ops-mn] {args.model}: matched-n re-baseline n={n} -> {n_sub} "
        f"(seeds {seeds}, hl={hl})"
    )
    per_seed: dict[str, dict] = {}
    for s in seeds:
        idx = common.group_stratified_subsample(groups, n_sub, seed=s)
        assert len(idx) == n_sub, (len(idx), n_sub)
        rot = fit825.random_projection_control(
            X[idx], Y[idx], groups[idx], layers=[hl], n_folds=args.folds, seed=args.fit_seed
        )
        entry: dict = {"rotated_hl": rot.get(str(hl))}
        if not args.skip_mlp:
            mlp = fit931._mlp_fold_r2(
                X[idx],
                Y[idx],
                groups[idx],
                layers=fl,
                n_draws=0,
                folds=args.folds,
                seed=args.fit_seed,
                max_epochs=50 if args.smoke else 300,
            )
            entry["mlp_frozen"] = {k: v["r2_obs"] for k, v in mlp.items()}
            entry["mlp_hl"] = mlp[str(hl)]["r2_obs"]
        per_seed[str(s)] = entry
        print(
            f"[i825-ops-mn] seed={s}: rotated@L{hl}={entry['rotated_hl']:.6f} "
            f"mlp@L{hl}={entry.get('mlp_hl')}"
        )

    mean_rot = float(np.mean([per_seed[str(s)]["rotated_hl"] for s in seeds]))
    mlp_vals = [per_seed[str(s)].get("mlp_hl") for s in seeds]
    mean_mlp = float(np.mean(mlp_vals)) if all(v is not None for v in mlp_vals) else None
    w_ex = max(mean_rot, mean_mlp) if mean_mlp is not None else mean_rot
    payload = {
        **base,
        "headline_layer": hl,
        "frozen_layers": fl,
        "per_seed": per_seed,
        "seed_mean_rotated_hl": mean_rot,
        "seed_mean_mlp_hl": mean_mlp,
        "w_ex_matched_n": w_ex,
        "convention": "max(seed-mean rotated @ hl, seed-mean MLP @ hl) — the committed "
        "max(rotated, MLP) convention on the seed-mean refits; C stays full-n by stated "
        "convention (matched-n C documented unevaluable)",
    }
    common.write_json(args.out, payload)
    print(
        f"[i825-ops-mn] {args.model}: w_ex_matched_n={w_ex:.6f} "
        f"(rot {mean_rot:.6f} / mlp {mean_mlp})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
