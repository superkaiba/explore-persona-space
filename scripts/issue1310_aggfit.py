"""Issue #1310 follow-up: scene-AGGREGATED prefill re-fit (one point per scene).

Tests whether the instruct prefill anti-prediction (per-persona held-out R^2
-0.10..-0.19 at L19 on per-turn points, swap read inverted) is a WITHIN-SCENE
near-duplicate-X artifact: within a prefill scene the 6 slot contexts share the
scene prompt (near-duplicate X) while their Y differ, which can drive held-out
R^2 negative under scene folds. Collapse each (persona, scenario) scene to ONE
point — X = the scene's earliest kept slot's x_spanmean (the shared-prompt
vector), Y = the mean of y over the scene's kept slots — and re-fit with the
IDENTICAL #825 GCV Gram-ridge battery (folds are then point-level because each
aggregated point is its own group). ~300 points/persona/model.

Swap read on aggregated points: pool all personas per model, pair each point
with a DIFFERENT persona's same-scenario aggregated Y (reuses
issue1310_fit.swap_derangement with turn_indices all-zero), correct vs swapped
fits + paired scenario-level bootstrap.

Reuses fit machinery verbatim: fit825.heldout_r2_sweep + selection_symmetric_
summary + mean_baseline + bootstrap via issue1310_fit.fit_cell, with
fit825.GCV_DOF_CAP = 0.9 (MANDATORY here: n~300 < 3584 dims, uncapped GCV is
degenerate — see fit825.GCV_DOF_CAP docs).

CLI:
  uv run python scripts/issue1310_aggfit.py [--models base,instruct]
      [--store-root data/issue_1310/store_onpolicy_dl/.../store_onpolicy]
      [--out-dir eval_results/issue_1310/onpolicy_aggregated]
      [--personas Wren,HELIOS,Dana,Vex] [--max-groups 0]
      [--null-draws 20] [--folds 5] [--seed 0] [--n-boot 1000]
      [--gcv-dof-cap 0.9] [--smoke]
"""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()  # shared-VM thread caps bind before torch/numpy import

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

import issue825_fit_cells as fit825  # noqa: E402
import issue931_common as common  # noqa: E402
import issue931_fit_cells as fit931  # noqa: E402
import issue1310_common as c1310  # noqa: E402
import issue1310_fit as fit1310  # noqa: E402

SCRIPT = "scripts/issue1310_aggfit.py"

DEFAULT_STORE_ROOT = Path(
    "data/issue_1310/store_onpolicy_dl/issue1310_char_map/analysis_tensors/store_onpolicy"
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--models", type=str, default="base,instruct")
    ap.add_argument("--store-root", type=Path, default=DEFAULT_STORE_ROOT)
    ap.add_argument(
        "--out-dir", type=Path, default=Path("eval_results/issue_1310/onpolicy_aggregated")
    )
    ap.add_argument("--personas", type=str, default=",".join(c1310.PERSONA_LABELS))
    ap.add_argument(
        "--max-groups",
        type=int,
        default=0,
        help="keep at most N scenarios per persona (0 = all; smoke uses e.g. 30)",
    )
    ap.add_argument("--null-draws", type=int, default=c1310.N_NULL_DRAWS)
    ap.add_argument("--folds", type=int, default=c1310.N_FOLDS)
    ap.add_argument("--seed", type=int, default=c1310.FIT_SEED)
    ap.add_argument("--n-boot", type=int, default=c1310.N_BOOTSTRAP)
    ap.add_argument(
        "--gcv-dof-cap",
        type=float,
        default=0.9,
        help="fit825.GCV_DOF_CAP — MANDATORY 0.9 here (n~300 << 3584 dims; "
        "uncapped GCV degenerates at the lambda-grid floor, proven on this store)",
    )
    ap.add_argument("--smoke", action="store_true", help="tiny-N run; numeric gates not binding")
    return ap.parse_args()


def aggregate_store(store: dict) -> dict:
    """Collapse per-turn rows to one point per (persona, scenario) scene.

    X = the group's earliest kept slot's x_spanmean (min turn_index, load-order
    tie-break) — the scene's shared-prompt context vector; Y = mean of y over
    the group's kept rows. Returns aggregated arrays + per-group audit fields.
    Asserts one scenario contributes at most one aggregated point per persona.
    """
    personas = np.asarray(store["char_ids"])
    scenarios = np.asarray(store["group_ids"])
    turns = np.asarray(store["turn_indices"], dtype=int)
    x = store["arrays"]["x_spanmean"]
    y = store["arrays"]["y"]
    n = len(personas)
    assert x.shape[0] == n and y.shape[0] == n, (x.shape, y.shape, n)

    keys = np.array([f"{p}|{g}" for p, g in zip(personas, scenarios, strict=True)])
    uniq, inv = np.unique(keys, return_inverse=True)
    m = len(uniq)
    counts = np.bincount(inv, minlength=m)

    # Sort rows by (group, turn_index, load order); segment per group.
    order = np.lexsort((np.arange(n), turns, inv))
    seg_starts = np.searchsorted(inv[order], np.arange(m), side="left")
    seg_ends = np.searchsorted(inv[order], np.arange(m), side="right")

    x_agg = np.empty((m, *x.shape[1:]), dtype=np.float32)
    y_agg = np.empty((m, *y.shape[1:]), dtype=np.float32)
    first_turn = np.empty(m, dtype=int)
    for k in range(m):
        rows = order[seg_starts[k] : seg_ends[k]]
        x_agg[k] = x[rows[0]]  # earliest kept slot = the shared-prompt context
        y_agg[k] = y[rows].mean(axis=0, dtype=np.float64).astype(np.float32)
        first_turn[k] = turns[rows[0]]
    p_agg = personas[order[seg_starts]]
    g_agg = scenarios[order[seg_starts]]
    assert len(np.unique([f"{p}|{g}" for p, g in zip(p_agg, g_agg, strict=True)])) == m
    return {
        "X": x_agg,
        "Y": y_agg,
        "personas": p_agg,
        "scenarios": g_agg,
        "rows_per_group": counts,
        "first_turn": first_turn,
    }


def subset_agg(agg: dict, personas: list[str], max_groups: int) -> dict:
    """Filter aggregated points to a persona subset and (optionally) the first
    ``max_groups`` scenarios per persona (sorted scenario id — shared across
    personas, so the swap control keeps cross-persona scenario overlap)."""
    keep = np.zeros(len(agg["personas"]), dtype=bool)
    for p in personas:
        m = agg["personas"] == p
        if max_groups > 0:
            scen = np.sort(np.unique(agg["scenarios"][m]))[:max_groups]
            m = m & np.isin(agg["scenarios"], scen)
        keep |= m
    return {k: (v[keep] if isinstance(v, np.ndarray) else v) for k, v in agg.items()}


def run_swap_agg(agg: dict, model_kind: str, args) -> dict | None:
    """Character-swap specificity on AGGREGATED points (turn_indices all-zero):
    each point keeps its own X but is paired with a different persona's
    same-scenario aggregated Y; correct vs swapped fits, paired group bootstrap."""
    zeros = np.zeros(len(agg["personas"]), dtype=int)
    rows, partners = fit1310.swap_derangement(
        agg["scenarios"], agg["personas"], zeros, seed=c1310.BUILD_SEED
    )
    if len(rows) < 2 * args.folds:
        print(f"[i1310-aggfit] swap: too few cross-persona scene pairs (n={len(rows)}) — skipped")
        return None
    x, y, g = agg["X"], agg["Y"], agg["scenarios"]
    correct_xy = {"X": x[rows], "Y": y[rows], "group_ids": g[rows]}
    swap_xy = {"X": x[rows], "Y": y[partners], "group_ids": g[rows]}
    res_c = fit1310.fit_cell(f"agg_{model_kind}_swapctrl_correct", correct_xy, args)
    res_s = fit1310.fit_cell(f"agg_{model_kind}_swap", swap_xy, args)
    hl = res_c["headline_layer"]
    sc, ss = res_c["sweep"], res_s["sweep"]
    if hl not in sc["preds_frozen"] or hl not in ss["preds_frozen"]:
        return None
    fitted = sc["fitted_mask"] & ss["fitted_mask"]
    pred_c = sc["preds_frozen"][hl][fitted]
    true_c = correct_xy["Y"][fitted, hl, :].astype(np.float64)
    pred_s = ss["preds_frozen"][hl][fitted]
    true_s = swap_xy["Y"][fitted, hl, :].astype(np.float64)
    groups = np.asarray(correct_xy["group_ids"])[fitted]
    gb_c = fit931.group_bootstrap_r2(pred_c, true_c, groups, n_boot=args.n_boot, seed=args.seed)
    gb_s = fit931.group_bootstrap_r2(
        pred_s,
        true_s,
        groups,
        n_boot=args.n_boot,
        seed=args.seed,
        draws_matrix=gb_c["draws_matrix"],
    )
    delta_draws = gb_c["draws"] - gb_s["draws"]
    payload = {
        "metadata": common.metadata(SCRIPT, args.seed, int(fitted.sum())),
        "model_kind": model_kind,
        "aggregation": "one point per (persona, scenario); X=min-turn slot x_spanmean, Y=mean(y)",
        "headline_layer": hl,
        "r2_correct": gb_c["r2"],
        "r2_swap": gb_s["r2"],
        "delta_r2_char": gb_c["r2"] - gb_s["r2"],
        "delta_ci_lo": float(np.nanquantile(delta_draws, 0.025)),
        "delta_ci_hi": float(np.nanquantile(delta_draws, 0.975)),
        "n_rows": int(fitted.sum()),
        "n_groups": int(gb_c["n_groups"]),
        "n_boot": int(args.n_boot),
        "paired_group_bootstrap": True,
        "gcv_dof_cap": args.gcv_dof_cap,
    }
    c1310.write_json(args.out_dir / f"swap_agg_{model_kind}.json", payload)
    return payload


def build_summary(results: dict, swaps: dict, agg_meta: dict, personas: list[str], args) -> None:
    """Per-persona aggregated headline R^2 + null clears + swap gap per model."""
    per_persona = {}
    for persona in personas:
        entry = {}
        for model_kind in results:
            cell = results[model_kind].get(persona)
            if cell is None:
                entry[model_kind] = None
                continue
            p = cell["payload"]
            hl_i = cell["headline_layer"]
            ss = p["selection_symmetric"]["frozen_layer_table"].get(str(hl_i), {})
            entry[model_kind] = {
                "n": p["n"],
                "r2_headline": p["r2_per_layer_obs"][hl_i],
                "null_p975_headline": ss.get("null_p975"),
                "null_mean_headline": ss.get("null_mean"),
                "clears_null": (p["r2_per_layer_obs"][hl_i] > ss.get("null_p975", float("inf"))),
                "skill_over_mean_headline": p["skill_over_mean"].get(str(hl_i)),
                "r2_frozen": {str(li): p["r2_per_layer_obs"][li] for li in p["frozen_layers"]},
            }
        per_persona[persona] = entry
    summary = {
        "metadata": common.metadata(SCRIPT, args.seed, 0),
        "aggregation": "one point per (persona, scenario); X=min-turn slot x_spanmean, Y=mean(y)",
        "headline_layer": c1310.HEADLINE_LAYER,
        "frozen_layers": list(c1310.FROZEN_LAYERS),
        "gcv_dof_cap": args.gcv_dof_cap,
        "per_persona": per_persona,
        "swap_specificity": {m: (swaps.get(m) or {}) for m in results},
        "agg_meta": agg_meta,
        "parent_reference": "per-turn fits: eval_results/issue_1310/onpolicy/summary.json",
        "smoke": bool(args.smoke),
    }
    c1310.write_json(args.out_dir / "summary_agg.json", summary)


def main() -> int:
    args = parse_args()
    fit825.GCV_DOF_CAP = args.gcv_dof_cap
    fit1310.SCRIPT = f"{SCRIPT} (via issue1310_fit.fit_cell)"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    personas = [p.strip() for p in args.personas.split(",") if p.strip()]
    for m in models:
        assert m in c1310.MODEL_KINDS, f"unknown model {m!r}"
    for p in personas:
        assert p in c1310.PERSONA_LABELS, f"unknown persona {p!r}"
    print(
        f"[phase=agg_fits] scene-aggregated fit battery (models={models}, "
        f"personas={personas}, max_groups={args.max_groups}, dof_cap={args.gcv_dof_cap})"
    )

    results: dict[str, dict] = {}
    swaps: dict[str, dict] = {}
    agg_meta: dict[str, dict] = {}
    for model_kind in models:
        store = fit1310.load_model_store(args.store_root, model_kind)
        n_layers = int(store["arrays"]["y"].shape[1])
        assert n_layers == c1310.EXPECTED_LAYERS, (n_layers, c1310.EXPECTED_LAYERS)
        print(f"[i1310-aggfit] model={model_kind} raw_rows={len(store['row_ids'])}")
        agg = aggregate_store(store)
        # free the ~9GB raw arrays immediately; aggregated is ~50MB
        del store
        gc.collect()
        agg = subset_agg(agg, personas, args.max_groups)
        agg_meta[model_kind] = {
            "n_points": len(agg["personas"]),
            "points_per_persona": {p: int((agg["personas"] == p).sum()) for p in personas},
            "rows_per_group_mean": float(np.mean(agg["rows_per_group"])),
            "rows_per_group_min": int(np.min(agg["rows_per_group"])),
            "rows_per_group_max": int(np.max(agg["rows_per_group"])),
            "first_turn_nonzero_frac": float(np.mean(agg["first_turn"] != 0)),
        }
        print(f"[i1310-aggfit] model={model_kind} agg_points={agg_meta[model_kind]['n_points']}")
        results[model_kind] = {}
        for persona in personas:
            m = agg["personas"] == persona
            xy = {"X": agg["X"][m], "Y": agg["Y"][m], "group_ids": agg["scenarios"][m]}
            if xy["X"].shape[0] < args.folds:
                print(
                    f"[i1310-aggfit] {model_kind}/{persona}: n={xy['X'].shape[0]} < folds — skipped"
                )
                continue
            print(f"[i1310-aggfit] fit {model_kind}/{persona} n={xy['X'].shape[0]}")
            results[model_kind][persona] = fit1310.fit_cell(f"agg_{model_kind}_{persona}", xy, args)
        swaps[model_kind] = run_swap_agg(agg, model_kind, args) or {}
        del agg
        gc.collect()

    build_summary(results, swaps, agg_meta, personas, args)
    print("[i1310-aggfit] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
