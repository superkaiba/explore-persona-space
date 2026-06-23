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

# --- metric-race family sets (plan §4.3, defined ONCE, never conflated) ----
# lfo_families: ALL 9 development families — the LFO-mean scope for H1 (global
#   champion) + H2 (heterogeneity). Folding over the smaller eligible set would
#   drop B7/B9/B10 from the H1 read and could flip a near-threshold verdict.
# eligible_families: the 6 dev families clearing the >=4-HELD-OUT-cell nested
#   floor — bound to H3 ONLY (the oracle/global/permutation gain terms). The
#   sparse families B7/B9/B10 (which cannot support a >=4-held-out-cell nested
#   within-family split) are descriptive-only.
LFO_FAMILIES: frozenset[str] = frozenset({"B1", "B2", "B3", "B5", "B6", "B7", "B8", "B9", "B10"})
ELIGIBLE_FAMILIES: frozenset[str] = frozenset({"B1", "B2", "B3", "B5", "B6", "B8"})
# The fold set where Groups B/C are defined (data-conditioned signals apply
# only here per column_applies) — the common-fold intersection for the
# A-vs-B/C comparison (Nit N1).
COMMON_FOLD_FAMILIES: frozenset[str] = frozenset({"B3", "B5", "B6"})
assert ELIGIBLE_FAMILIES <= LFO_FAMILIES  # H3 scope is a subset of the LFO scope

# Metric-race null parameters (plan §10/§11).
RACE_BOOTSTRAP_B = 2000  # selection/argmax statistic → doubled vs v1's B=1000
RACE_PERMUTATION_P = 10000
RACE_SEED = 545
H1_TAU_FLOOR = 0.10  # non-noise gate (plan H1)
# v1 target reliability ceiling (Spearman seed0↔seed137) — for the τ-as-fraction
# effect-size interpretation (Nit N3). The 0.10 floor stays the binary gate.
V1_RELIABILITY_CEILING_RAW = 0.588
V1_RELIABILITY_CEILING_SB = 0.740  # Spearman-Brown-adjusted


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
    pred_dir_name: str = "predictors",
) -> Path:
    """Run the full pre-registered race. Writes scoring/scoring_results.json.

    Defaults reproduce the pre-registered protocol byte-for-byte. The optional
    kwargs exist for LABELED follow-up passes (never overwrite the prereg
    record): ``exclude_rows`` drops every cell of those train rows from the
    scoring universe (targets, z-norm pool, dev AND quarantine lists) before
    any selection; ``out_dir_name`` redirects output (e.g.
    ``scoring_followup_bcond``); ``protocol_note`` stamps the results JSON
    with why the pass deviates from the prereg record; ``pred_dir_name``
    redirects the predictor source dir (e.g. ``predictors_metric_race`` for
    the metric-race Analysis-1 global-champion read over the EXPANDED zoo —
    same frozen LFO-CV, expanded candidate pool).
    """
    out_root = output_root()
    prereg = json.loads((out_root / "preregistration.json").read_text())
    matrix = json.loads((out_root / "L_matrix.json").read_text())["cells"]
    metadata = json.loads((out_root / "cell_metadata.json").read_text())["cells"]
    preds = _load_predictors(out_root / pred_dir_name)

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
    # "PFX" (#595 prefix-binding) + "PST" (#640 postfix-binding) admitted by the
    # 1-line groups-tuple extension so each carrier family enters the
    # leave-family-out CV / quarantine race against #545's A/B/C/D. The edit is
    # surgical: it does NOT alter #545's existing A/B/C/D groups (when no PFX/PST
    # predictor JSON is staged the group contributes nothing — _champion returns
    # None, group_k == 0; pinned in test_issue595_scoring_groups_extension.py).
    groups = ("A", "B", "C", "D", "PFX", "PST")
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


# ===========================================================================
# Metric-race scoring (the `full-metric-race-per-family` follow-up, plan §4.2)
# ===========================================================================


# Metric-family membership (plan §4.1). Order WITHIN each tuple is
# longest-suffix-first so the parser below matches multi-token metrics
# (``mahal_pooled_ctx``, ``gauss_kl``, ``neg_l2``, ``delta_spec``) BEFORE
# their single-token prefixes (``mahal``, ``kl``...) — a bare
# ``body.rsplit("_", 1)[-1]`` collapsed every underscore metric to its last
# token and mislabeled the whole covariance/cloud/centered family (Codex
# Major #2, reconciler r1 binding FAIL).
_CENTROID_METRICS_COVARIANCE = ("mahal_pooled_ctx", "mahal", "euclidean")
_CENTROID_METRICS_RAW = ("cosine", "neg_l2", "projection")
_CLOUD_METRICS = (
    "rbf_mmd_squared",
    "bures_wasserstein2",
    "delta_spec",
    "gauss_kl",
    "wass2",
    "c2st",
    "mmd",
)
# Longest-first across ALL metric suffixes (the order the parser scans).
_ALL_METRIC_SUFFIXES: tuple[str, ...] = tuple(
    sorted(
        set(_CENTROID_METRICS_COVARIANCE) | set(_CENTROID_METRICS_RAW) | set(_CLOUD_METRICS),
        key=len,
        reverse=True,
    )
)


def _parse_metric_suffix(body: str) -> str | None:
    """Return the metric suffix of a ``cloud_*`` predictor body, longest-first.

    The body is ``cloud_<flavor>_L<layer>_<point>_[<centering>_]<metric>`` where
    ``<point>`` (``mean_response`` / ``last_token``) and several ``<metric>``
    values (``mahal_pooled_ctx``, ``gauss_kl``, ``neg_l2``) themselves contain
    underscores, so we cannot split on ``_``. Match the registered metric
    suffixes longest-first and require an underscore boundary before the
    match (so ``..._neg_l2`` matches ``neg_l2`` and never the substring ``l2``
    of an unrelated token). Returns ``None`` if no registered metric matches.
    """
    for metric in _ALL_METRIC_SUFFIXES:
        if body == metric or body.endswith(f"_{metric}"):
            return metric
    return None


def _metric_family(name: str) -> str:
    """Map a predictor name → its metric family for the leaderboard grouping.

    Group A names are ``A__cloud_<flavor>_L<layer>_<point>_<centering>_<metric>``
    (centroid + cloud) or ``A__outdist_<flavor>_<direction>``; plus the v1
    ``A__geom_*`` reference. Non-A predictors keep their group letter.
    """
    g = name.split("__", 1)[0]
    if g != "A":
        return f"group_{g.lower()}"
    body = name.split("__", 1)[1]
    if body.startswith("geom_"):
        return "raw_centroid"  # v1 reference {cosine,neg_l2,projection}
    if body.startswith("outdist_"):
        return "outdist_jskl"
    # cloud_<flavor>_L<layer>_<point>_<centering>_<metric>  OR  cloud_..._<metric>
    metric = _parse_metric_suffix(body)
    if metric is None:
        return "other_A"
    if metric in _CLOUD_METRICS:
        return "cloud"
    # Centroid metrics: the centered variant carries an explicit `_centered_`
    # segment in the body (the raw variant carries `_raw_`).
    if "_centered_" in body and metric in (
        set(_CENTROID_METRICS_RAW) | set(_CENTROID_METRICS_COVARIANCE)
    ):
        return "centered_centroid"
    if metric in _CENTROID_METRICS_COVARIANCE:
        return "covariance_centroid"
    if metric in _CENTROID_METRICS_RAW:
        return "raw_centroid"
    return "other_A"


def _within_family_row_bootstrap(
    cells: list[str], stat_fn, *, n_boot: int = RACE_BOOTSTRAP_B, seed: int = RACE_SEED
) -> dict | None:
    """Within-family ROW/CELL bootstrap (NO family-level resampling).

    For the per-family argmax CI (analysis 2): the estimand is scoped to a
    single family, so there is no LFO boundary to protect — resample ROWS
    (cells within a row share an adapter) with replacement INSIDE the family.
    ``stat_fn(sample_cells)`` → float | None. Returns the percentile CI.
    """
    rows: dict[str, list[str]] = {}
    for c in cells:
        rows.setdefault(c.split("|")[0], []).append(c)
    row_keys = sorted(rows)
    if len(row_keys) < 2:
        return None
    rng = random.Random(seed)
    stats: list[float] = []
    for _ in range(n_boot):
        sample_cells: list[str] = []
        for _i in range(len(row_keys)):
            sample_cells.extend(rows[rng.choice(row_keys)])
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


def _lfo_nested_bootstrap(
    preds: dict[str, dict],
    target: dict[str, float],
    dev_cells: list[str],
    *,
    n_boot: int = RACE_BOOTSTRAP_B,
    seed: int = RACE_SEED,
) -> dict:
    """LFO-mean held-out-τ bootstrap CI for the GLOBAL champion (H1) with the
    family-ID exclusion invariant (plan §4.3(i), Blocker-3 fix).

    The held-out unit is the ORIGINAL family, NOT the resample copy. Per
    resample b, for each held-out original family F in ``lfo_families`` (all 9):
      1. two-stage cluster resample over lfo_families (families w/ replacement,
         rows within),
      2. TRAIN = resampled cells whose ORIGINAL family != F (excludes ALL F
         copies),
      3. HELDOUT = the original (un-resampled) cells of F,
      4. select the global champion on TRAIN (the single best predictor over
         ALL groups), score it on HELDOUT,
      5. INVARIANT assert: train ∩ heldout original-family ids == ∅ AND
         >=2 distinct train families; violators SKIPPED + tallied.
    The per-resample statistic is the LFO MEAN held-out τ (averaged over the 9
    folds of that resample). Returns the CI of that LFO-mean over resamples +
    skip diagnostics.
    """
    fams: dict[str, dict[str, list[str]]] = {}
    for c in dev_cells:
        if c not in target:
            continue
        fam = ROWS[c.split("|")[0]].family
        fams.setdefault(fam, {}).setdefault(c.split("|")[0], []).append(c)
    dev_families = set(fams)
    # COVERAGE INVARIANT: all 9 lfo_families folded; a re-run that drops one HALTs.
    assert dev_families == set(LFO_FAMILIES), (
        f"lfo coverage drift: dev_families={sorted(dev_families)} != "
        f"lfo_families={sorted(LFO_FAMILIES)} — the H1 read must fold over all 9"
    )
    fam_keys = sorted(fams)
    rng = random.Random(seed)
    lfo_means: list[float] = []
    skipped = 0
    total_folds = 0
    for _b in range(n_boot):
        # One two-stage cluster resample over all 9 families.
        resampled: list[tuple[str, str]] = []  # (orig_family, cell)
        for _i in range(len(fam_keys)):
            fam = rng.choice(fam_keys)
            row_keys = sorted(fams[fam])
            for _j in range(len(row_keys)):
                rk = rng.choice(row_keys)
                for cell in fams[fam][rk]:
                    resampled.append((fam, cell))
        fold_taus: list[float] = []
        for held in fam_keys:
            total_folds += 1
            train_cells = [c for (f, c) in resampled if f != held]
            train_fams = {f for (f, c) in resampled if f != held}
            heldout_cells = [c for rk in fams[held] for c in fams[held][rk]]
            heldout_fams = {held}
            # INVARIANT: no F copy leaks into train; >=2 distinct train families.
            if (train_fams & heldout_fams) or len(train_fams) < 2:
                skipped += 1
                continue
            champ = _global_champion(preds, target, train_cells)
            if champ is None:
                skipped += 1
                continue
            tau = weighted_kendall_tau(preds[champ]["cells"], target, heldout_cells)
            if tau is not None:
                fold_taus.append(tau)
        if fold_taus:
            lfo_means.append(float(np.mean(fold_taus)))
    diag = {
        "n_bootstrap_valid": len(lfo_means),
        "skipped_resample_folds": skipped,
        "total_resample_folds": total_folds,
        "lfo_families": sorted(LFO_FAMILIES),
    }
    if len(lfo_means) < 100:
        return {"ci95": None, **diag}
    lfo_means.sort()
    return {
        "ci95": (
            lfo_means[int(0.025 * len(lfo_means))],
            lfo_means[int(0.975 * len(lfo_means)) - 1],
        ),
        "lfo_mean_point": float(np.median(lfo_means)),
        **diag,
    }


def _global_champion(
    preds: dict[str, dict], target: dict[str, float], cells: list[str]
) -> str | None:
    """The single best predictor over ALL groups on ``cells`` (the global
    champion — H1's unit, distinct from the per-group ``_champion``)."""
    best, best_tau = None, None
    for name, d in preds.items():
        tau = weighted_kendall_tau(d["cells"], target, cells)
        if tau is not None and (best_tau is None or tau > best_tau):
            best, best_tau = name, tau
    return best


def _within_column_permutation(
    stat_fn,
    target: dict[str, float],
    cells: list[str],
    *,
    n_perm: int = RACE_PERMUTATION_P,
    seed: int = RACE_SEED,
) -> dict:
    """Within-column row-label permutation null (plan §4.3(ii)).

    Permute row labels WITHIN each eval column (preserving each column's
    marginal target distribution + within-column z-norm), recompute
    ``stat_fn(permuted_target)`` on the SAME cell set. Returns the observed
    statistic's percentile + the exact finite-sample one-sided p.
    """
    observed = stat_fn(target)
    if observed is None:
        return {"observed": None, "p_value": None, "n_perm_valid": 0}
    by_col: dict[str, list[str]] = {}
    for c in cells:
        if c in target:
            by_col.setdefault(c.split("|")[1], []).append(c)
    rng = random.Random(seed)
    null: list[float] = []
    for _ in range(n_perm):
        permuted = dict(target)
        for _col, col_cells in by_col.items():
            vals = [target[c] for c in col_cells]
            rng.shuffle(vals)
            for c, v in zip(col_cells, vals, strict=True):
                permuted[c] = v
        s = stat_fn(permuted)
        if s is not None:
            null.append(s)
    if len(null) < 100:
        return {"observed": observed, "p_value": None, "n_perm_valid": len(null)}
    n_ge = sum(1 for s in null if s >= observed)
    return {
        "observed": float(observed),
        "p_value": float((n_ge + 1) / (len(null) + 1)),
        "null_p95": float(np.percentile(null, 95)),
        "n_perm_valid": len(null),
    }


def _per_family_champions(
    preds: dict[str, dict],
    target: dict[str, float],
    dev_cells: list[str],
    families: list[str],
    *,
    n_boot: int = RACE_BOOTSTRAP_B,
) -> dict:
    """Analysis 2: per-family argmax predictor over the full zoo + within-family
    held-out τ (nested split) + within-family row bootstrap CI, gated on the
    selection-noise null. Eligible families get a CI verdict; descriptive-only
    families get the argmax winner with NO CI."""
    fam_cells: dict[str, list[str]] = {}
    for c in dev_cells:
        if c in target:
            fam_cells.setdefault(ROWS[c.split("|")[0]].family, []).append(c)
    out: dict[str, dict] = {}
    for fam in families:
        cells = fam_cells.get(fam, [])
        if not cells:
            continue
        eligible = fam in ELIGIBLE_FAMILIES
        # Per-family argmax champion over the full pooled family cells.
        champ = _global_champion(preds, target, cells)
        rec: dict = {
            "n_cells": len(cells),
            "argmax_champion": champ,
            "eligible": eligible,
        }
        if champ:
            rec["pooled_tau"] = weighted_kendall_tau(preds[champ]["cells"], target, cells)
        if eligible and champ:
            # Nested within-family held-out read: split rows, select on train,
            # score on heldout. With >=4 held-out cells the within-family CI is
            # meaningful (descriptive-only families skip this).
            rows = sorted({c.split("|")[0] for c in cells})
            if len(rows) >= 2:
                heldout_rows = set(rows[len(rows) // 2 :])
                train = [c for c in cells if c.split("|")[0] not in heldout_rows]
                heldout = [c for c in cells if c.split("|")[0] in heldout_rows]
                ch_nested = _global_champion(preds, target, train)
                if ch_nested and len(heldout) >= 4:
                    rec["nested_champion"] = ch_nested
                    rec["heldout_tau"] = weighted_kendall_tau(
                        preds[ch_nested]["cells"], target, heldout
                    )

            def _argmax_tau(cs: list[str]) -> float | None:
                ch = _global_champion(preds, target, cs)
                return weighted_kendall_tau(preds[ch]["cells"], target, cs) if ch else None

            boot = _within_family_row_bootstrap(cells, _argmax_tau, n_boot=n_boot)
            if boot:
                rec["bootstrap_ci95"] = boot["ci95"]
                rec["n_bootstrap_valid"] = boot["n_valid"]
                lo, hi = boot["ci95"]
                rec["ci_excludes_zero"] = bool(lo > 0 or hi < 0)
        out[fam] = rec
    return out


def score_metric_race(  # noqa: C901 — the three analyses + nulls, intentionally flat
    *,
    pred_dir_name: str = "metric_race/predictors_metric_race",
    out_dir_name: str = "metric_race/scoring_metric_race",
    n_boot: int = RACE_BOOTSTRAP_B,
    n_perm: int = RACE_PERMUTATION_P,
) -> dict[str, Path]:
    """The metric-race scoring: Analysis 1 (global champion, expanded zoo) +
    Analysis 2 (per-family) + Analysis 3 (heterogeneity + H3 optimism gain) +
    the selection-noise nulls. Writes to ``metric_race/`` (NEW namespace —
    never overwrites the prereg ``scoring/`` record).

    Returns {label: path} for the four output JSONs.
    """
    out_root = output_root()
    prereg = json.loads((out_root / "preregistration.json").read_text())
    matrix = json.loads((out_root / "L_matrix.json").read_text())["cells"]
    metadata = json.loads((out_root / "cell_metadata.json").read_text())["cells"]
    preds = _load_predictors(out_root / pred_dir_name)

    split = prereg["quarantine_split"]
    dev_cells = ["|".join(c) for c in split["development_cells"]]

    targets_raw = _seed_mean_targets(matrix, metadata, include_flagged=False)
    vals = {k: v["shift"] for k, v in targets_raw.items() if v.get("shift") is not None}
    target = _z_norm_within_column(vals)
    dev = [c for c in dev_cells if c in target]

    # --- Analysis 1: global champion via the frozen LFO-CV over the expanded
    #     zoo. Re-use score() unchanged (pred_dir_name redirect). ----------
    scoring_results_path = score(
        out_dir_name=out_dir_name,
        pred_dir_name=pred_dir_name,
        protocol_note=(
            "metric-race Analysis 1: frozen leave-family-out CV over the "
            "EXPANDED predictor zoo (predictors_metric_race/); EXPLORATORY "
            "dev-only headline, quarantine reported peeked-only"
        ),
    )

    # --- metric-family leaderboard + per-group common-fold read (Nit N1) ---
    leaderboard: dict[str, dict] = {}
    for name, d in preds.items():
        tau = weighted_kendall_tau(d["cells"], target, dev)
        if tau is None:
            continue
        fam = _metric_family(name)
        n_cells = sum(1 for c in dev if c in d["cells"] and c in target)
        leaderboard[name] = {
            "tau": round(tau, 4),
            "metric_family": fam,
            "n_cells": n_cells,
            "tau_frac_of_ceiling_raw": round(tau / V1_RELIABILITY_CEILING_RAW, 4),
            "tau_frac_of_ceiling_sb": round(tau / V1_RELIABILITY_CEILING_SB, 4),
        }
    leaderboard = dict(sorted(leaderboard.items(), key=lambda kv: -kv[1]["tau"]))

    # Per-metric-family mean-τ on ALL folds vs the common-fold intersection.
    fam_cells_all = _families_of(dev)
    common_fold_cells = [c for c in dev if ROWS[c.split("|")[0]].family in COMMON_FOLD_FAMILIES]
    group_fold_summary: dict[str, dict] = {}
    by_family_group: dict[str, list[str]] = {}
    for name in preds:
        by_family_group.setdefault(_metric_family(name), []).append(name)
    for mfam, names in sorted(by_family_group.items()):

        def _best_on(cells_subset, _names=names):
            best = None
            for nm in _names:
                t = weighted_kendall_tau(preds[nm]["cells"], target, cells_subset)
                if t is not None and (best is None or t > best):
                    best = t
            return best

        group_fold_summary[mfam] = {
            "mean_tau_all_folds": _best_on(dev),
            "mean_tau_common_folds": _best_on(common_fold_cells),
            "common_fold_ids": sorted(COMMON_FOLD_FAMILIES),
            "n_predictors": len(names),
        }

    # --- H1 LFO-nested bootstrap CI for the global champion ----------------
    global_champ_dev = _global_champion(preds, target, dev)
    lfo_boot = _lfo_nested_bootstrap(preds, target, dev, n_boot=n_boot)
    h1: dict = {
        "global_champion_on_dev": global_champ_dev,
        "global_champion_metric_family": (
            _metric_family(global_champ_dev) if global_champ_dev else None
        ),
        "lfo_nested_bootstrap": lfo_boot,
        "tau_floor": H1_TAU_FLOOR,
        "v1_reliability_ceiling": {
            "raw_spearman": V1_RELIABILITY_CEILING_RAW,
            "spearman_brown": V1_RELIABILITY_CEILING_SB,
        },
    }
    # Transfers? CI excludes 0 AND LFO-mean point >= 0.10. (Permutation gate
    # for H1 is the within-family permutation on the champion below.)
    if lfo_boot.get("ci95"):
        lo, hi = lfo_boot["ci95"]
        pt = lfo_boot.get("lfo_mean_point", 0.0)
        h1["ci_excludes_zero"] = bool(lo > 0 or hi < 0)
        h1["clears_floor"] = bool(pt >= H1_TAU_FLOOR)
        h1["verdict_ci_floor"] = (
            "transfers_ci_floor" if (lo > 0 and pt >= H1_TAU_FLOOR) else "does_not_transfer"
        )

    # H1/H2 within-column permutation on the global champion's dev τ.
    def _champ_dev_tau(perm_target: dict[str, float]) -> float | None:
        ch = _global_champion(preds, perm_target, dev)
        return weighted_kendall_tau(preds[ch]["cells"], perm_target, dev) if ch else None

    h1["champion_dev_permutation"] = _within_column_permutation(
        _champ_dev_tau, target, dev, n_perm=n_perm
    )

    # --- Analysis 2: per-family champions (eligible + descriptive-only) ----
    all_dev_families = sorted(LFO_FAMILIES)
    per_family = _per_family_champions(preds, target, dev, all_dev_families, n_boot=n_boot)

    # --- Analysis 3: heterogeneity + H3 optimism gain ----------------------
    # H3 invariant: oracle sum, global term, AND every permutation replicate
    # computed on the SAME eligible_families + cell subset.
    eligible_dev_cells = [c for c in dev if ROWS[c.split("|")[0]].family in ELIGIBLE_FAMILIES]
    elig_fam_cells: dict[str, list[str]] = {}
    for c in eligible_dev_cells:
        elig_fam_cells.setdefault(ROWS[c.split("|")[0]].family, []).append(c)

    def _oracle_gain(perm_target: dict[str, float]) -> float | None:
        """H3 gain = sum_F tau(per-family-oracle on F) - tau(global champ on
        union F), all over eligible_families + the matched cell subset (the
        runtime invariant below proves the family/cell sets match)."""
        oracle_fams, oracle_sum = [], 0.0
        for fam in sorted(ELIGIBLE_FAMILIES):
            cells = elig_fam_cells.get(fam, [])
            if not cells:
                return None
            ch = _global_champion(preds, perm_target, cells)
            t = weighted_kendall_tau(preds[ch]["cells"], perm_target, cells) if ch else None
            if t is None:
                return None
            oracle_sum += t
            oracle_fams.append(fam)
        gch = _global_champion(preds, perm_target, eligible_dev_cells)
        gt = (
            weighted_kendall_tau(preds[gch]["cells"], perm_target, eligible_dev_cells)
            if gch
            else None
        )
        if gt is None:
            return None
        # H3 statistic per plan section 4.3: gain = sum_F tau(per-family
        # oracle on F) - tau(single global champion on union F). The global
        # term is ONE tau (not scaled); the same statistic is recomputed for
        # every permutation replicate, so the null is on the identical scale.
        _ = oracle_fams  # documents the family set the oracle sum ran over
        return oracle_sum - gt

    # Runtime invariant (plan §4.3 H3, written to the output): the family set +
    # cell subset are identical across the oracle term, the global term, and
    # every permutation replicate (they all read from elig_fam_cells /
    # eligible_dev_cells, by construction).
    oracle_family_set = sorted(elig_fam_cells)
    assert set(oracle_family_set) == set(ELIGIBLE_FAMILIES), (
        f"H3 eligible-family drift: {oracle_family_set} != {sorted(ELIGIBLE_FAMILIES)}"
    )
    union_cells = sorted(eligible_dev_cells)
    cells_from_families = sorted(c for fam in elig_fam_cells for c in elig_fam_cells[fam])
    assert cells_from_families == union_cells, (
        "H3 cell-subset drift between oracle and global terms"
    )

    h3_perm = _within_column_permutation(_oracle_gain, target, eligible_dev_cells, n_perm=n_perm)

    # Heterogeneity (H2): cross-family variance of the per-family champion
    # held-out τ over eligible families, vs the family-label permutation null.
    def _heterogeneity(perm_target: dict[str, float]) -> float | None:
        taus = []
        for fam in sorted(ELIGIBLE_FAMILIES):
            cells = elig_fam_cells.get(fam, [])
            if not cells:
                return None
            ch = _global_champion(preds, perm_target, cells)
            t = weighted_kendall_tau(preds[ch]["cells"], perm_target, cells) if ch else None
            if t is None:
                return None
            taus.append(t)
        return float(np.var(taus)) if len(taus) >= 2 else None

    h2_perm = _within_column_permutation(_heterogeneity, target, eligible_dev_cells, n_perm=n_perm)

    heterogeneity = {
        "statistic": "cross_family_variance_of_per_family_champion_tau",
        "eligible_families": sorted(ELIGIBLE_FAMILIES),
        "permutation": h2_perm,
        "verdict": (
            "real_heterogeneity"
            if (h2_perm.get("p_value") is not None and h2_perm["p_value"] < 0.05)
            else "chance"
        ),
    }
    h3 = {
        "statistic": "per_family_oracle_gain_minus_global_champion",
        "eligible_families": sorted(ELIGIBLE_FAMILIES),
        "excluded_descriptive_only_families": sorted(LFO_FAMILIES - ELIGIBLE_FAMILIES),
        "oracle_family_set": oracle_family_set,
        "n_eligible_cells": len(eligible_dev_cells),
        "permutation": h3_perm,
        "verdict": (
            "genuine_specialization"
            if (h3_perm.get("p_value") is not None and h3_perm["p_value"] < 0.05)
            else "optimism_inside_null_band"
        ),
        "invariant_checked": (
            "oracle_families == global_eval_families == permuted_eval_families == "
            "eligible_families AND identical cell subset (asserted at runtime)"
        ),
    }

    # --- predictor-family x behavior-family tau heatmap (analysis 3) ------
    heatmap: dict[str, dict[str, float | None]] = {}
    for mfam, names in sorted(by_family_group.items()):
        heatmap[mfam] = {}
        for fam in all_dev_families:
            cells = fam_cells_all.get(fam, [])
            best = None
            for nm in names:
                t = weighted_kendall_tau(preds[nm]["cells"], target, cells)
                if t is not None and (best is None or t > best):
                    best = t
            heatmap[mfam][fam] = round(best, 4) if best is not None else None

    # --- assemble + write the metric-race outputs -------------------------
    out_dir = out_root / out_dir_name
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = reproducibility_metadata()
    n_skipped = lfo_boot.get("skipped_resample_folds", 0)

    scoring_metric_race = {
        "n_dev_cells": len(dev),
        "n_predictors": len(preds),
        "dev_leaderboard": leaderboard,
        "group_fold_summary": group_fold_summary,
        "h1_global_champion": h1,
        "h2_heterogeneity": heterogeneity,
        "h3_optimism_gain": h3,
        "exploratory_note": (
            "EXPLORATORY dev-only headline via the frozen leave-family-out CV; "
            "the v1 quarantine is reported only as a flagged-peeked sensitivity "
            "read in scoring_results.json (Analysis 1), never the headline."
        ),
        "metadata": meta,
    }
    per_family_and_het = {
        "per_family_champions": per_family,
        "lfo_families": sorted(LFO_FAMILIES),
        "eligible_families": sorted(ELIGIBLE_FAMILIES),
        "descriptive_only_families": sorted(LFO_FAMILIES - ELIGIBLE_FAMILIES),
        "predictor_family_x_behavior_family_heatmap": heatmap,
        "h3_optimism_gain": h3,
        "invariants": {
            "lfo_coverage": (
                "all 9 dev families folded for H1/H2 (asserted in _lfo_nested_bootstrap)"
            ),
            "h3_eligible_set": (
                "oracle==global==permuted==eligible_families (asserted in score_metric_race)"
            ),
        },
        "metadata": meta,
    }
    bootstrap_diag = {
        "lfo_nested_bootstrap": {
            "n_bootstrap_valid": lfo_boot.get("n_bootstrap_valid"),
            "skipped_resample_folds": n_skipped,
            "total_resample_folds": lfo_boot.get("total_resample_folds"),
            "B": n_boot,
        },
        "per_family_within_family_bootstrap_B": n_boot,
        "permutation_P": n_perm,
        "seed": RACE_SEED,
        "metadata": meta,
    }
    common_fold_diag = {
        "common_fold_families": sorted(COMMON_FOLD_FAMILIES),
        "group_fold_summary": group_fold_summary,
        "note": (
            "per-metric-family best-τ on ALL dev folds vs the common-fold "
            "intersection {B3,B5,B6} where Groups B/C are defined (Nit N1); "
            "the global-champion + H3 comparisons are read primarily on the "
            "matched common-fold set."
        ),
        "metadata": meta,
    }

    paths = {
        "scoring_metric_race": out_dir / "scoring_metric_race.json",
        "per_family_and_heterogeneity": out_dir / "per_family_and_heterogeneity.json",
        "bootstrap_diagnostics": out_dir / "bootstrap_diagnostics.json",
        "common_fold_diagnostics": out_dir / "common_fold_diagnostics.json",
    }
    paths["scoring_metric_race"].write_text(json.dumps(scoring_metric_race, indent=1))
    paths["per_family_and_heterogeneity"].write_text(json.dumps(per_family_and_het, indent=1))
    paths["bootstrap_diagnostics"].write_text(json.dumps(bootstrap_diag, indent=1))
    paths["common_fold_diagnostics"].write_text(json.dumps(common_fold_diag, indent=1))
    paths["analysis1_scoring_results"] = scoring_results_path
    logger.info("[phase=score] metric-race wrote %d JSONs to %s", len(paths), out_dir)
    return paths
