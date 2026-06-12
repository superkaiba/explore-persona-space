"""Issue #545 predictor-race scoring harness (plan sections 4.5 + 6; CPU only).

Protocol (pre-registered in preregistration.json):

- Universe: OFF-DIAGONAL cells only (diagonals excluded BEFORE the quarantine
  draw). Saturated + implant_failed cells excluded by default; a sensitivity
  pass includes them.
- Targets: two tracks, never pooled — ``level`` (post-training absolute
  expression) and ``shift`` (trained - base L). Both z-normed within column.
- Headline metric: weighted Kendall tau (scipy.stats.weightedtau).
- CV: leave-family-out over the development cells, champion selection NESTED
  inside CV training folds; per-group candidate counts K reported.
- Confirmatory H2 margin: read ONCE on the quarantine split with champions
  frozen on the full development set AND the B-vs-C representative chosen on
  development tau (quarantine blind — the quarantine split is never used for
  ANY selection; round-2 reconciler blocker). Both unselected B-vs-A and
  C-vs-A quarantine margins are reported alongside. CI = paired row-clustered
  bootstrap (cells within a row share an adapter — resampling cells
  understates the variance).
- H3: base-prior wins level, geometry/delta-rule wins shift.
- H4 (exploratory): rank-one ALS fit on dev cells, held-out R^2 on quarantine.
- Combiners: ridge stack of the per-group dev champions (nested the same way).
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path

import numpy as np

from . import output_root, reproducibility_metadata
from .assemble_matrix import PRIMARY_SCALAR
from .rows import ROWS

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _load_predictors(pred_dir: Path) -> dict[str, dict]:
    """name -> {"group", "track", "cells": {"row|col": score}}."""
    preds = {}
    for p in sorted(pred_dir.glob("*.json")):
        d = json.loads(p.read_text())
        if "cells" not in d:
            continue
        preds[f"{d['group']}__{d['name']}"] = d
    if not preds:
        raise FileNotFoundError(f"No predictor JSONs under {pred_dir}")
    return preds


def _seed_mean_targets(matrix: dict, metadata: dict, *, include_flagged: bool) -> dict:
    """(row|col) -> {"level": x, "shift": y} seed-means over primary-arm cells.

    Saturated / implant_failed cells drop unless ``include_flagged``.
    """
    acc: dict[str, dict[str, list[float]]] = {}
    for cell_id, columns in matrix.items():
        meta = metadata.get(cell_id, {})
        if meta.get("arm") != "primary":
            continue
        if not include_flagged and meta.get("implant_failed"):
            continue
        row_id = meta.get("row")
        for col_ctx, entry in columns.items():
            col_id, ctx = col_ctx.rsplit("__", 1)
            if ctx != "default" or col_id not in PRIMARY_SCALAR:
                continue
            if not include_flagged and entry.get("saturation_flag"):
                continue
            level, shift = entry.get("level"), entry.get("L")
            if level is None and shift is None:
                continue
            key = f"{row_id}|{col_id}"
            slot = acc.setdefault(key, {"level": [], "shift": []})
            if isinstance(level, (int, float)):
                slot["level"].append(float(level))
            if isinstance(shift, (int, float)):
                slot["shift"].append(float(shift))
    out = {}
    for key, slot in acc.items():
        out[key] = {track: (sum(v) / len(v) if v else None) for track, v in slot.items()}
    return out


def _z_norm_within_column(values: dict[str, float]) -> dict[str, float]:
    """z-norm cell values within each eval column (family-aware scoring)."""
    by_col: dict[str, list[tuple[str, float]]] = {}
    for key, v in values.items():
        col = key.split("|")[1]
        by_col.setdefault(col, []).append((key, v))
    out = {}
    for _col, items in by_col.items():
        vals = np.array([v for _, v in items], dtype=float)
        mu, sd = float(vals.mean()), float(vals.std())
        for key, v in items:
            out[key] = (v - mu) / sd if sd > 1e-12 else 0.0
    return out


# ---------------------------------------------------------------------------
# Metric
# ---------------------------------------------------------------------------


def weighted_kendall_tau(
    pred: dict[str, float], target: dict[str, float], cells: list[str]
) -> float | None:
    """Weighted Kendall tau over the intersection of cells with both values."""
    from scipy.stats import weightedtau

    xs, ys = [], []
    for c in cells:
        if c in pred and c in target:
            xs.append(pred[c])
            ys.append(target[c])
    if len(xs) < 4:
        return None
    tau = weightedtau(xs, ys).statistic
    return float(tau) if tau == tau else None


# ---------------------------------------------------------------------------
# The race
# ---------------------------------------------------------------------------


def _families_of(cells: list[str]) -> dict[str, list[str]]:
    fams: dict[str, list[str]] = {}
    for c in cells:
        row_id = c.split("|")[0]
        fams.setdefault(ROWS[row_id].family, []).append(c)
    return fams


def _family_row_bootstrap(
    cells: list[str], stat_fn, *, n_boot: int = 1000, seed: int = 545
) -> dict | None:
    """Hierarchical family->row cluster bootstrap 95% CI of ``stat_fn(cells)``.

    Cells within a row share an adapter AND rows within a family share corpus
    lineage (the family CV folds treat them as dependent), so the resample is
    two-stage: families with replacement, then rows within each sampled
    family (plan: row/family-clustered margin CIs; round-1 minor #15).
    ``stat_fn`` returns a float or None (dropped replicates).
    """
    fams: dict[str, dict[str, list[str]]] = {}
    for c in cells:
        row_id = c.split("|")[0]
        fams.setdefault(ROWS[row_id].family, {}).setdefault(row_id, []).append(c)
    fam_keys = sorted(fams)
    if len(fam_keys) < 2:
        return None
    rng = random.Random(seed)
    stats: list[float] = []
    for _ in range(n_boot):
        sample_cells: list[str] = []
        for _i in range(len(fam_keys)):
            fam = rng.choice(fam_keys)
            row_keys = sorted(fams[fam])
            for _j in range(len(row_keys)):
                sample_cells.extend(fams[fam][rng.choice(row_keys)])
        v = stat_fn(sample_cells)
        if v is not None:
            stats.append(v)
    if len(stats) < 100:
        return None
    stats.sort()
    return {
        "ci95": (stats[int(0.025 * len(stats))], stats[int(0.975 * len(stats)) - 1]),
        "n_valid": len(stats),
    }


def _champion(
    preds: dict[str, dict], group: str, target: dict[str, float], cells: list[str]
) -> tuple[str | None, float | None]:
    """Best predictor of a group on the given cells (the NESTED selection unit)."""
    best, best_tau = None, None
    for name, d in preds.items():
        if not name.startswith(f"{group}__"):
            continue
        tau = weighted_kendall_tau(d["cells"], target, cells)
        if tau is not None and (best_tau is None or tau > best_tau):
            best, best_tau = name, tau
    return best, best_tau


def _ridge_combiner(
    preds: dict[str, dict],
    champion_names: list[str],
    target: dict[str, float],
    train_cells: list[str],
) -> dict[str, float]:
    """Ridge stack of the per-group champions, fit on train cells only."""
    feats = [preds[n]["cells"] for n in champion_names if n]
    usable = [c for c in train_cells if c in target and all(c in f for f in feats)]
    if len(usable) < 6 or not feats:
        return {}
    X = np.array([[f[c] for f in feats] for c in usable])
    y = np.array([target[c] for c in usable])
    X = (X - X.mean(0)) / (X.std(0) + 1e-12)
    lam = 1.0
    w = np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ y)
    out = {}
    all_cells = set.intersection(*(set(f) for f in feats))
    mu, sd = (
        np.array([[f[c] for f in feats] for c in usable]).mean(0),
        np.array([[f[c] for f in feats] for c in usable]).std(0) + 1e-12,
    )
    for c in all_cells:
        x = (np.array([f[c] for f in feats]) - mu) / sd
        out[c] = float(x @ w)
    return out


def rank_one_fit(
    target: dict[str, float], train_cells: list[str], test_cells: list[str], *, n_iter: int = 200
) -> dict:
    """H4: ALS rank-one model L[r,c] ~ u_r * v_c fit on train, R^2 on test."""
    rows = sorted({c.split("|")[0] for c in train_cells if c in target})
    cols = sorted({c.split("|")[1] for c in train_cells if c in target})
    ri = {r: i for i, r in enumerate(rows)}
    ci = {c: i for i, c in enumerate(cols)}
    rng = np.random.default_rng(545)
    u = rng.normal(size=len(rows))
    v = rng.normal(size=len(cols))
    obs = [
        (ri[c.split("|")[0]], ci[c.split("|")[1]], target[c])
        for c in train_cells
        if c in target and c.split("|")[0] in ri and c.split("|")[1] in ci
    ]
    if len(obs) < 8:
        return {"fit": False, "reason": f"only {len(obs)} observed train cells"}
    for _ in range(n_iter):
        num = np.zeros(len(rows))
        den = np.zeros(len(rows))
        for i, j, y in obs:
            num[i] += v[j] * y
            den[i] += v[j] ** 2
        u = num / (den + 1e-12)
        num = np.zeros(len(cols))
        den = np.zeros(len(cols))
        for i, j, y in obs:
            num[j] += u[i] * y
            den[j] += u[i] ** 2
        v = num / (den + 1e-12)
    preds_test, ys_test = [], []
    for c in test_cells:
        r, col = c.split("|")
        if c in target and r in ri and col in ci:
            preds_test.append(u[ri[r]] * v[ci[col]])
            ys_test.append(target[c])
    if len(ys_test) < 4:
        return {"fit": True, "heldout_r2": None, "n_test": len(ys_test)}
    yt = np.array(ys_test)
    pt = np.array(preds_test)
    ss_res = float(((yt - pt) ** 2).sum())
    ss_tot = float(((yt - yt.mean()) ** 2).sum())
    return {
        "fit": True,
        "heldout_r2": 1 - ss_res / (ss_tot + 1e-12),
        "n_train": len(obs),
        "n_test": len(ys_test),
        "u": dict(zip(rows, u.tolist(), strict=False)),
        "v": dict(zip(cols, v.tolist(), strict=False)),
    }


def score(  # noqa: C901 — pre-registered protocol, intentionally flat
    *,
    include_flagged: bool = False,
    exclude_rows: frozenset[str] = frozenset(),
    out_dir_name: str = "scoring",
    protocol_note: str | None = None,
) -> Path:
    """Run the full pre-registered race. Writes scoring/scoring_results.json.

    Defaults reproduce the pre-registered protocol byte-for-byte. The optional
    kwargs exist for LABELED follow-up passes (never overwrite the prereg
    record): ``exclude_rows`` drops every cell of those train rows from the
    scoring universe (targets, z-norm pool, dev AND quarantine lists) before
    any selection; ``out_dir_name`` redirects output (e.g.
    ``scoring_followup_bcond``); ``protocol_note`` stamps the results JSON
    with why the pass deviates from the prereg record.
    """
    out_root = output_root()
    prereg = json.loads((out_root / "preregistration.json").read_text())
    matrix = json.loads((out_root / "L_matrix.json").read_text())["cells"]
    metadata = json.loads((out_root / "cell_metadata.json").read_text())["cells"]
    preds = _load_predictors(out_root / "predictors")

    split = prereg["quarantine_split"]
    dev_cells = ["|".join(c) for c in split["development_cells"]]
    quarantine = ["|".join(c) for c in split["sampled_quarantined_cells"]] + [
        "|".join(c) for c in split["family_quarantined_cells"]
    ]

    targets_raw = _seed_mean_targets(matrix, metadata, include_flagged=include_flagged)
    results: dict = {
        "include_flagged": include_flagged,
        "n_dev_cells": len(dev_cells),
        "n_quarantine_cells": len(quarantine),
        "group_k": {},
        "tracks": {},
        "metadata": reproducibility_metadata(),
    }
    if exclude_rows:
        n_before = len(targets_raw)
        targets_raw = {k: v for k, v in targets_raw.items() if k.split("|")[0] not in exclude_rows}
        results["universe_filter"] = {
            "excluded_rows": sorted(exclude_rows),
            "n_target_cells_dropped": n_before - len(targets_raw),
        }
    if protocol_note:
        results["protocol_note"] = protocol_note
    groups = ("A", "B", "C", "D")
    for g in groups:
        results["group_k"][g] = sum(1 for n in preds if n.startswith(f"{g}__"))

    # ONE z-normed race track (round-1 major #7): within-column z-norm makes
    # level and shift arithmetically identical (shift = level - per-column
    # base constant), so a second "level" leaderboard would be a duplicate
    # labeled as distinct. Level-vs-shift is read on RAW targets in the H3
    # block below.
    results["tracks_note"] = (
        "single z-normed track: level==shift after within-column z-norm; "
        "H3 level-vs-shift uses raw targets"
    )
    for track in ("shift",):
        vals = {k: v[track] for k, v in targets_raw.items() if v.get(track) is not None}
        target = _z_norm_within_column(vals)
        dev = [c for c in dev_cells if c in target]
        quar = [c for c in quarantine if c in target]
        track_out: dict = {"n_dev_with_target": len(dev), "n_quarantine_with_target": len(quar)}

        # Leaderboard on dev (descriptive; champions nested below).
        leaderboard = {}
        for name, d in preds.items():
            tau = weighted_kendall_tau(d["cells"], target, dev)
            if tau is not None:
                leaderboard[name] = round(tau, 4)
        track_out["dev_leaderboard"] = dict(sorted(leaderboard.items(), key=lambda kv: -kv[1]))

        # Leave-family-out CV with NESTED champion selection.
        fams = _families_of(dev)
        cv: dict[str, dict] = {g: {"fold_taus": []} for g in groups}
        cv["combiner"] = {"fold_taus": []}
        for fam, fold_cells in sorted(fams.items()):
            train_cells = [c for c in dev if c not in set(fold_cells)]
            champs = []
            for g in groups:
                champ, _ = _champion(preds, g, target, train_cells)
                champs.append(champ)
                if champ:
                    tau = weighted_kendall_tau(preds[champ]["cells"], target, fold_cells)
                    if tau is not None:
                        cv[g]["fold_taus"].append(
                            {"family": fam, "champion": champ, "tau": round(tau, 4)}
                        )
            comb = _ridge_combiner(preds, [c for c in champs if c], target, train_cells)
            if comb:
                tau = weighted_kendall_tau(comb, target, fold_cells)
                if tau is not None:
                    cv["combiner"]["fold_taus"].append({"family": fam, "tau": round(tau, 4)})
        for _g, d in cv.items():
            taus = [f["tau"] for f in d["fold_taus"]]
            d["mean_tau"] = round(float(np.mean(taus)), 4) if taus else None
        track_out["leave_family_out_cv"] = cv

        # Confirmatory quarantine read: champions frozen on FULL dev.
        frozen = {g: _champion(preds, g, target, dev)[0] for g in groups}
        quar_taus = {}
        for g, champ in frozen.items():
            if champ:
                quar_taus[g] = {
                    "champion": champ,
                    "tau": weighted_kendall_tau(preds[champ]["cells"], target, quar),
                }
        comb = _ridge_combiner(preds, [c for c in frozen.values() if c], target, dev)
        if comb:
            quar_taus["combiner"] = {"tau": weighted_kendall_tau(comb, target, quar)}
        track_out["quarantine_frozen_champions"] = quar_taus

        # H2 margin: best-of-B/C minus best-of-A, row-clustered bootstrap CI.
        # The B-vs-C representative is chosen on DEV tau — the quarantine
        # split is scored exactly once and never used for ANY selection
        # (round-2 reconciler blocker i545-h2-selects-bc-on-quarantine: the
        # old max-on-quarantine pick test-set-selected the confirmatory
        # headline). Both unselected margins are reported alongside.
        def _tau_on(cells_subset: list[str], champ: str | None, *, _target=target) -> float | None:
            return (
                weighted_kendall_tau(preds[champ]["cells"], _target, cells_subset)
                if champ
                else None
            )

        def _tau_or(tau: float | None, missing: float) -> float:
            """None-safe default: tau 0.0 is a legitimate value; only None maps to ``missing``."""
            return missing if tau is None else tau

        bc_best = max(
            (g for g in ("B", "C") if frozen.get(g)),
            key=lambda g: _tau_or(_tau_on(dev, frozen[g]), -2.0),
            default=None,
        )
        if bc_best and frozen.get("A"):
            point = _tau_or(_tau_on(quar, frozen[bc_best]), 0.0) - _tau_or(
                _tau_on(quar, frozen["A"]), 0.0
            )

            def _margin_stat(
                cells_subset: list[str], *, _bc=bc_best, _frozen=frozen
            ) -> float | None:
                tb = _tau_on(cells_subset, _frozen[_bc])
                ta = _tau_on(cells_subset, _frozen["A"])
                return tb - ta if tb is not None and ta is not None else None

            boot = _family_row_bootstrap(quar, _margin_stat)
            track_out["h2_margin"] = {
                "best_bc_group": bc_best,
                "bc_selection": "dev_tau_frozen_quarantine_blind",
                "point": round(point, 4),
                "threshold": prereg["thresholds"]["h2_margin"],
                "family_row_clustered_bootstrap_ci95": boot["ci95"] if boot else None,
                "n_bootstrap_valid": boot["n_valid"] if boot else 0,
            }
            unselected = {}
            for g in ("B", "C"):
                tg, ta = _tau_on(quar, frozen.get(g)), _tau_on(quar, frozen["A"])
                if tg is not None and ta is not None:
                    unselected[g] = round(tg - ta, 4)
            track_out["h2_margins_unselected"] = unselected
        track_out["h4_rank_one"] = rank_one_fit(target, dev, quar)
        results["tracks"][track] = track_out

    # H3: two-component decomposition (#532), on RAW (non-z-normed) targets:
    # the within-column z-norm removes the per-column constant, which makes
    # level == shift exactly (shift = level - base[col]) and zeroes the
    # cross-column signal the base-prior predictor carries. Both H3 reads
    # carry family/row-clustered bootstrap CIs (plan H3: CIs excluding 0).
    raw_targets = {
        track: {k: v[track] for k, v in targets_raw.items() if v.get(track) is not None}
        for track in ("level", "shift")
    }
    h3_dev = [c for c in dev_cells if c in raw_targets["level"] and c in raw_targets["shift"]]
    h3_quar = [c for c in quarantine if c in raw_targets["level"] and c in raw_targets["shift"]]

    def _h3_block(pred_cells: dict[str, float], cells: list[str], *, orientation: str) -> dict:
        """tau(level), tau(shift) + the ORIENTED difference with clustered CI.

        ``orientation`` ("level_minus_shift" for the base-prior H3 half,
        "shift_minus_level" for the geometry half) is emitted directly —
        point AND CI in the hypothesis's own sign, no negate-and-swap at
        analysis time (round-2 minor #4).
        """
        sign = 1 if orientation == "level_minus_shift" else -1

        def _diff_stat(cells_subset: list[str]) -> float | None:
            tl = weighted_kendall_tau(pred_cells, raw_targets["level"], cells_subset)
            ts = weighted_kendall_tau(pred_cells, raw_targets["shift"], cells_subset)
            if tl is None or ts is None:
                return None
            return sign * (tl - ts)

        tau_level = weighted_kendall_tau(pred_cells, raw_targets["level"], cells)
        tau_shift = weighted_kendall_tau(pred_cells, raw_targets["shift"], cells)
        boot = _family_row_bootstrap(cells, _diff_stat)
        return {
            "n_cells": len(cells),
            "tau_level": tau_level,
            "tau_shift": tau_shift,
            f"tau_{orientation}": (
                sign * (tau_level - tau_shift)
                if tau_level is not None and tau_shift is not None
                else None
            ),
            f"ci95_{orientation}": boot["ci95"] if boot else None,
            "n_bootstrap_valid": boot["n_valid"] if boot else 0,
        }

    bp = preds.get("B__base_prior_level")
    if bp:
        results["h3_base_prior"] = {
            "note": "raw targets (z-norm collapses level/shift; see scoring.py)",
            "dev": _h3_block(bp["cells"], h3_dev, orientation="level_minus_shift"),
            "quarantine": _h3_block(bp["cells"], h3_quar, orientation="level_minus_shift"),
        }
    # Geometry side: best Group A predictor selected on DEV raw-SHIFT tau
    # (the track geometry is hypothesized to win). The dev block re-reads the
    # cells the champion was selected on (selection-inflated — round-2 minor
    # #3), so the quarantine block, read with the SAME dev-frozen champion,
    # is the selection-clean H3 geometry read.
    geo_best, _ = (
        _champion(preds, "A", _z_norm_within_column(raw_targets["shift"]), h3_dev)
        if h3_dev
        else (None, None)
    )
    if geo_best:
        results["h3_best_geometry"] = {
            "champion": geo_best,
            "selection": "best Group A on dev raw-shift tau (dev-frozen)",
            "dev": {
                "selection_optimism": (
                    "champion selected on these same dev cells — tau_shift is "
                    "selection-inflated; read the quarantine block"
                ),
                **_h3_block(preds[geo_best]["cells"], h3_dev, orientation="shift_minus_level"),
            },
            "quarantine": _h3_block(
                preds[geo_best]["cells"], h3_quar, orientation="shift_minus_level"
            ),
        }

    # Per-cell-type cuts (expected dense/null/surprising coverage per group).
    cuts: dict[str, dict] = {}
    vals = {k: v["shift"] for k, v in targets_raw.items() if v.get("shift") is not None}
    target = _z_norm_within_column(vals)
    for expected in ("dense", "null", "surprising"):
        cells_e = [
            c for c in dev_cells if c in target and ROWS[c.split("|")[0]].expected == expected
        ]
        cuts[expected] = {"n": len(cells_e)}
        for g in groups:
            champ, _ = _champion(preds, g, target, [c for c in dev_cells if c in target])
            if champ:
                cuts[expected][g] = weighted_kendall_tau(preds[champ]["cells"], target, cells_e)
    results["per_cell_type_cuts"] = cuts

    out_dir = out_root / out_dir_name
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_with_flagged" if include_flagged else ""
    out_path = out_dir / f"scoring_results{suffix}.json"
    out_path.write_text(json.dumps(results, indent=1))
    logger.info("[phase=scoring] wrote %s", out_path)
    return out_path
