"""Issue #1310: focused per-character context->dialogue map fits.

Per (persona, model) the WITHIN map is fit on that ONE persona's TURNS only
(many per-turn points, one character): GCV Gram ridge (reused #825 core), K=5
SCENE-grouped folds (group_id = scenario_id; within a persona each scenario is
one scene, so all a scene's turns stay in one fold — turns never split across
train/test), frozen layers {14,18,19,26} headline 19, 20 selection-symmetric
shuffle nulls + 1000-draw bootstrap. One map per persona per model -> reads
(a) does the focused map clear its null and (b) base-vs-instruct strength.

Character-swap specificity (read c): pooled across personas per model, a
MATCHED-SCENE-POSITION cross-persona derangement — each target turn keeps its
own context X but is paired with a DIFFERENT persona's dialogue Y at the SAME
(scenario, turn_index) — vs the correct pairing over the SAME rows; dR2 with a
paired scenario-level bootstrap.

Assistant-map ceiling (read d): the committed #825 Track-S curves
(cells_S1.json 0.673 / cells_S2.json 0.588 at L19) read read-only as context.

CLI:
  uv run python scripts/issue1310_fit.py [--models base,instruct]
      [--data-dir data/issue_1310] [--out-dir eval_results/issue_1310]
      [--null-draws 20] [--folds 5] [--seed 0] [--n-boot 1000] [--smoke]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()  # thread caps bind before torch/numpy import

import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

import issue825_fit_cells as fit825  # noqa: E402
import issue931_common as common  # noqa: E402
import issue931_fit_cells as fit931  # noqa: E402
import issue1310_common as c1310  # noqa: E402

SCRIPT = "scripts/issue1310_fit.py"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--models", type=str, default="base,instruct")
    ap.add_argument("--data-dir", type=Path, default=Path("data/issue_1310"))
    ap.add_argument("--out-dir", type=Path, default=Path("eval_results/issue_1310"))
    ap.add_argument("--null-draws", type=int, default=c1310.N_NULL_DRAWS)
    ap.add_argument("--folds", type=int, default=c1310.N_FOLDS)
    ap.add_argument("--seed", type=int, default=c1310.FIT_SEED)
    ap.add_argument("--n-boot", type=int, default=c1310.N_BOOTSTRAP)
    ap.add_argument("--smoke", action="store_true", help="numeric gates recorded, not binding")
    return ap.parse_args()


def frozen_layers(n_layers: int) -> list[int]:
    fl = [li for li in c1310.FROZEN_LAYERS if li < n_layers]
    return fl or [n_layers - 1]


def headline_layer(n_layers: int) -> int:
    return c1310.HEADLINE_LAYER if n_layers > c1310.HEADLINE_LAYER else n_layers - 1


def load_model_store(store_root: Path, model_kind: str) -> dict:
    """Concatenate a model's shards -> {row_ids, group_ids, char_ids, arrays}."""
    store_dir = store_root / model_kind
    shards = sorted(store_dir.glob(f"{model_kind}_shard*.pt"))
    assert shards, f"no {model_kind} shards under {store_dir}"
    rows, groups, chars, turns = [], [], [], []
    arrays: dict[str, list] = {}
    for sp in shards:
        payload = torch.load(sp, map_location="cpu", weights_only=False)
        rows.extend(payload["row_ids"])
        groups.extend(payload["group_ids"])
        chars.extend(payload["char_ids"])
        turns.extend(payload["turn_indices"])
        for k, v in payload["arrays"].items():
            arrays.setdefault(k, []).append(v.float().numpy().astype(np.float32))
    out = {k: np.concatenate(v, axis=0) for k, v in arrays.items()}
    n = len(rows)
    for k, v in out.items():
        assert v.shape[0] == n, (k, v.shape, n)
    return {
        "row_ids": np.asarray(rows),
        "group_ids": np.asarray(groups),
        "char_ids": np.asarray(chars),
        "turn_indices": np.asarray(turns, dtype=int),
        "arrays": out,
    }


def fit_cell(cell_id: str, xy: dict, args) -> dict:
    """Held-out sweep + selection-symmetric summary + baselines + bootstrap."""
    X, Y, groups = xy["X"], xy["Y"], xy["group_ids"]
    n, n_layers = X.shape[0], X.shape[1]
    fit825.FROZEN_LAYERS = tuple(frozen_layers(n_layers))
    sweep = fit825.heldout_r2_sweep(
        X, Y, groups, n_folds=args.folds, seed=args.seed, null_draws=args.null_draws
    )
    r2_obs, r2_null = sweep["r2_obs"], sweep["r2_null"]
    summary = fit825.selection_symmetric_summary(r2_obs, r2_null)
    fl = frozen_layers(n_layers)
    hl = headline_layer(n_layers)
    mb = fit825.mean_baseline_r2(Y, groups, layers=fl, n_folds=args.folds, seed=args.seed)
    rp = fit825.random_projection_control(
        X, Y, groups, layers=[hl], n_folds=args.folds, seed=args.seed
    )
    fitted = sweep["fitted_mask"]
    boot_row = {}
    for li in fl:
        if li not in sweep["preds_frozen"]:
            continue
        pred = sweep["preds_frozen"][li][fitted]
        true = Y[fitted, li, :].astype(np.float64)
        boot_row[str(li)] = fit825.bootstrap_r2_ci(
            pred, true, n_boot=args.n_boot, seed=args.seed + 100 + li
        )
    payload = {
        "metadata": common.metadata(SCRIPT, args.seed, n),
        "cell_id": cell_id,
        "n": n,
        "n_groups": len(np.unique(groups)),
        "n_layers": int(n_layers),
        "headline_layer": hl,
        "frozen_layers": fl,
        "r2_per_layer_obs": [float(v) for v in r2_obs],
        "selection_symmetric": summary,
        "mean_baseline_r2": mb,
        "random_projection_control_r2": rp,
        "skill_over_mean": {
            str(li): float(r2_obs[li]) - float(mb.get(str(li), float("nan"))) for li in fl
        },
        "r2_bootstrap_row_frozen": boot_row,
        "n_folds": args.folds,
        "null_draws": args.null_draws,
    }
    c1310.write_json(args.out_dir / f"cells_{cell_id}.json", payload)
    c1310.write_json(
        args.out_dir / f"nulls_{cell_id}.json",
        {
            "metadata": common.metadata(SCRIPT, args.seed, n),
            "cell_id": cell_id,
            "layers": list(range(n_layers)),
            "observed_row": [float(v) for v in r2_obs],
            "null_matrix": [[float(v) for v in row] for row in r2_null],
            "null_layer_max_per_draw": summary["null_layer_max_r2_per_draw"],
        },
    )
    return {"sweep": sweep, "xy": xy, "payload": payload, "headline_layer": hl}


def within_xy(store: dict, persona: str, x_key: str) -> dict:
    """Per-persona (X,Y): subset store to char_id==persona; group = scenario_id."""
    m = store["char_ids"] == persona
    return {
        "X": store["arrays"][x_key][m],
        "Y": store["arrays"]["y"][m],
        "group_ids": store["group_ids"][m],
        "row_ids": store["row_ids"][m],
    }


def swap_derangement(
    group_ids: np.ndarray, char_ids: np.ndarray, turn_indices: np.ndarray, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Matched-scene-position cross-persona derangement.

    Group rows by (scenario_id, turn_index): within each group with >=2 distinct
    personas (each persona speaks its Nth line at most once per scene, so the
    group's rows ARE distinct personas), seed-shuffle and cyclically pair each
    row's Y-partner with the next row — a guaranteed derangement pairing persona
    A's Nth-line context with a DIFFERENT persona's Nth-line dialogue at the SAME
    scenario. Returns (row_idx, partner_idx).
    """
    rng = np.random.default_rng(seed)
    rows_out, partners_out = [], []
    order = np.lexsort((turn_indices, group_ids))
    keys = list(zip(group_ids[order], turn_indices[order], strict=True))
    start = 0
    for i in range(1, len(order) + 1):
        if i == len(order) or keys[i] != keys[start]:
            idx = order[start:i]
            if len(np.unique(char_ids[idx])) >= 2:
                perm = idx[rng.permutation(len(idx))]
                for j in range(len(perm)):
                    rows_out.append(int(perm[j]))
                    partners_out.append(int(perm[(j + 1) % len(perm)]))
            start = i
    if not rows_out:
        return np.asarray([], dtype=int), np.asarray([], dtype=int)
    rows = np.asarray(rows_out)
    partners = np.asarray(partners_out)
    assert (rows != partners).all(), "derangement violated"
    assert (char_ids[rows] != char_ids[partners]).all(), "swap partner shares persona"
    return rows, partners


def run_swap(store: dict, model_kind: str, args) -> dict | None:
    """Character-swap specificity: correct vs cross-persona-swapped Y, dR2."""
    rows, partners = swap_derangement(
        store["group_ids"], store["char_ids"], store["turn_indices"], seed=c1310.BUILD_SEED
    )
    if len(rows) < 2 * args.folds:
        print(f"[i1310-fit] swap: too few matched-position turn pairs (n={len(rows)}) — skipped")
        return None
    x = store["arrays"]["x_spanmean"]
    y = store["arrays"]["y"]
    g = store["group_ids"]
    correct_xy = {
        "X": x[rows],
        "Y": y[rows],
        "group_ids": g[rows],
        "row_ids": store["row_ids"][rows],
    }
    swap_xy = {
        "X": x[rows],
        "Y": y[partners],
        "group_ids": g[rows],
        "row_ids": store["row_ids"][rows],
    }
    res_c = fit_cell(f"{model_kind}_swapctrl_correct", correct_xy, args)
    res_s = fit_cell(f"{model_kind}_swap", swap_xy, args)
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
    }
    c1310.write_json(args.out_dir / f"swap_{model_kind}.json", payload)
    return payload


def assistant_ceiling() -> dict:
    """Committed #825 Track-S map R2 at the headline layer (context ceiling)."""
    out = {}
    for name, path in c1310.ASSISTANT_REF_PATHS.items():
        if not path.exists():
            out[name] = {"path": str(path), "present": False, "r2_headline": None}
            continue
        d = json.loads(path.read_text())
        rl = d.get("r2_per_layer_obs")
        hl = c1310.HEADLINE_LAYER
        r2 = float(rl[hl]) if rl and len(rl) > hl else None
        out[name] = {"path": str(path), "present": True, "headline_layer": hl, "r2_headline": r2}
    return out


def build_summary(results: dict, swaps: dict, args) -> None:
    """Per-persona base-vs-instruct headline R2 + null band + swap gap + drops."""
    per_persona = {}
    for persona in c1310.PERSONA_LABELS:
        entry = {}
        for model_kind in results:
            cell = results[model_kind].get(persona)
            if cell is None:
                entry[model_kind] = None
                continue
            p = cell["payload"]
            hl = str(cell["headline_layer"])
            ss = p["selection_symmetric"]["frozen_layer_table"].get(hl, {})
            entry[model_kind] = {
                "n": p["n"],
                "r2_headline": p["r2_per_layer_obs"][cell["headline_layer"]],
                "null_p975_headline": ss.get("null_p975"),
                "null_mean_headline": ss.get("null_mean"),
                "clears_null": (
                    p["r2_per_layer_obs"][cell["headline_layer"]]
                    > ss.get("null_p975", float("inf"))
                ),
                "skill_over_mean_headline": p["skill_over_mean"].get(hl),
            }
        per_persona[persona] = entry
    drops = {}
    for model_kind in results:
        audit_path = args.out_dir / f"attribution_audit_{model_kind}.json"
        if audit_path.exists():
            a = json.loads(audit_path.read_text())
            drops[model_kind] = {
                "counters": a.get("counters"),
                "drop_rate": a.get("drop_rate"),
                "per_persona_pairs": a.get("per_persona_pairs"),
                "attribution_precision": (a.get("audit") or {}).get("precision"),
            }
    summary = {
        "metadata": common.metadata(SCRIPT, args.seed, 0),
        "headline_layer": c1310.HEADLINE_LAYER,
        "per_persona": per_persona,
        "swap_specificity": {m: (swaps.get(m) or {}) for m in results},
        "assistant_ceiling": assistant_ceiling(),
        "attribution": drops,
        "frozen_layers": list(c1310.FROZEN_LAYERS),
        "smoke": bool(args.smoke),
    }
    c1310.write_json(args.out_dir / "summary.json", summary)


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    for m in models:
        assert m in c1310.MODEL_KINDS, f"unknown model {m!r}"
    print(f"[phase=p3_fits] fit battery (models={models})")
    store_root = args.data_dir / "store"

    results: dict[str, dict] = {}
    swaps: dict[str, dict] = {}
    for model_kind in models:
        store = load_model_store(store_root, model_kind)
        n_layers = int(store["arrays"]["y"].shape[1])
        if args.smoke:
            # rebind reused loaders' layer-axis asserts to the smoke dims.
            fit825.EXPECTED_LAYERS = n_layers
        print(f"[i1310-fit] model={model_kind} rows={len(store['row_ids'])} layers={n_layers}")
        results[model_kind] = {}
        for persona in c1310.PERSONA_LABELS:
            xy = within_xy(store, persona, "x_spanmean")
            if xy["X"].shape[0] < args.folds:
                print(f"[i1310-fit] {model_kind}/{persona}: n={xy['X'].shape[0]} < folds — skipped")
                continue
            print(f"[i1310-fit] fit {model_kind}/{persona} n={xy['X'].shape[0]}")
            results[model_kind][persona] = fit_cell(f"{model_kind}_{persona}", xy, args)
            # last-position variant (parent-matched single-position X)
            xy_last = within_xy(store, persona, "x_last")
            fit_cell(f"{model_kind}_{persona}_lastpos", xy_last, args)
        swaps[model_kind] = run_swap(store, model_kind, args) or {}

    build_summary(results, swaps, args)
    print("[i1310-fit] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
