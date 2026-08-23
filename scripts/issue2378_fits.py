"""Issue #2378 P6 — own-map ceiling fits (unit 3 deliverable 1; plan §4.4/§6).

Per cell-arm (9 active cells × {context v_C, prefix v_P} → v_A = 18 units;
plan v7 — dialogue descoped), K=5
grouped folds seed 137 over the unit-2 capture store:
  - GCV ridge fit-then-apply via the PARENT core
    ``issue2054_fits._ridge_gcv_fit_predict`` (GCV + dof cap 0.9 + degenerate
    flag), held-out R² via ``_r2_matrix``, reduced-basis k=1024 diagnostic via
    ``_reduced_basis_r2``, per-fold row bootstrap via ``_bootstrap_conv_ci``.
  - 100-draw shuffled-answer matched-capacity null via
    ``_shuffled_answer_null_r2`` (SVD-once-per-fold, batched draws); ALL draw
    values persisted per fold (plan §3 reporting-tier inputs).
  - 200-draw ceiling-margin bootstrap m = R² − max(null p95, 0.05) (values
    persisted) → registered tier (clearly-mappable / clearly-unmappable /
    boundary-indeterminate).
  - Baselines: ``analysis.mapping_baselines.identity_bias_predict`` (+ pooled
    kNN retrieval, euclidean + cosine, k ∈ {1,5,10}, chance stated).
  - Story cells: PRIMARY family-held-out fold (headline label
    ``family-held-out``) + scene-grain companion (context arm, labeled block).
  - User arms: pair-complete intersection cohort with the §4.2b fail-loud
    asserts BEFORE fitting; arm-specific full-cohort fits as labeled
    supplementary rows inside the same JSON (keeps the 18-flat-file glob at v7).

Phases:
  ``--phase g3``       chat/context own-map fit FIRST as the 1-cell
                       production-shape pilot (plan §7 G3); writes
                       ``g3_gate.json`` with measured per-phase walls; refusal
                       exits rc=3 with the report JSON persisted (never bare
                       rc=1). ``--smoke-cell chat_pilot`` is an alias.
  ``--phase fit``      own-map units (``--units``; requires a G3 PASS).
  ``--phase ratio``    H4a user-vs-assistant ceiling ratios (skip-and-count).
  ``--phase fold-map`` build/verify the shared fold map only.
  ``--phase probe``    synthetic-tensor CPU self-verification (tiny n/d; no
                       GPU, no network): full-path e2e through the production
                       loaders/fit/gate code on a synthetic store in the exact
                       capture format, planted-failure asserts (family-fold
                       overlap, user-pair v_C mismatch, n_train floor), the
                       batched-vs-serial null oracle, skip-and-count on a
                       planted degenerate ceiling, and the bf16 codec
                       equivalence vs torch.

Outputs (flat JSONs match the registered glob ``fits/*.json`` — 18 files at v7;
sidecars live in subdirs): ``fits/<cell>__<arm>.json``,
``fits/percell/<cell>__<arm>__rowstats.npz`` (per-row ss_res/ss_tot for the
ladder's recovery/H3/H4b bootstraps), ``fits/preds/<cell>__<arm>__preds.npz``
(float32 pooled held-out predictions — the unit-4 retrieval-battery seam),
``fits/ratio/h4a_ceiling_ratio.json``, ``g3_gate.json``, ``fold_map.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
for _p in (str(_SCRIPT_DIR), str(_REPO_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

import numpy as np  # noqa: E402

import issue2054_fits as pf  # noqa: E402  (parent cores — plan §10 reuse row)
import issue2378_common as cm  # noqa: E402
import issue2378_p6_common as p6  # noqa: E402
from explore_persona_space.analysis.mapping_baselines import (  # noqa: E402
    identity_bias_predict,
    knn_retrieval,
)

SCRIPT_VERSION = "issue2378_fits_v1"
G3_RC_REFUSED = 3  # distinct, artifact-routed refusal rc (never bare rc=1)
G3_RETRIEVAL_CHANCE_MULT = 10.0


def _log(msg: str) -> None:
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# Layer resolution
# ---------------------------------------------------------------------------


def resolve_layer(args) -> int:
    if args.layer is not None:
        return int(args.layer)
    path = Path(args.layer_star_from or (Path(args.ledger_root) / "pilot" / "layer_sweep.json"))
    if not path.exists():
        raise RuntimeError(
            f"cannot resolve the read layer: pass --layer or --layer-star-from (missing {path})"
        )
    return int(json.loads(path.read_text(encoding="utf-8"))["selected_layer"])


# ---------------------------------------------------------------------------
# Core fold-loop (shared by primary / companion / supplementary runs)
# ---------------------------------------------------------------------------


def _fit_under_folds(
    X: np.ndarray,
    Y: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
    *,
    seed_parts: tuple,
    n_null_draws: int,
    bootstrap_draws: int,
    reduced_k: int,
    with_null: bool,
    with_reduced: bool = True,
) -> dict:
    """Run the parent GCV-ridge battery under the given fold splits.

    Returns per-fold records + per-row (ss_res, ss_tot) rowstats in the input
    row order + pooled float32 predictions (each row scored by its OWN fold's
    map — the pooled-R²/retrieval convention).
    """
    n, d = X.shape
    ybar = Y.astype(np.float64).mean(axis=0)  # pooled-mean ss_tot convention
    ss_tot = ((Y.astype(np.float64) - ybar) ** 2).sum(axis=1)
    ss_res = np.full(n, np.nan)
    ss_res_id = np.full(n, np.nan)
    folds_of = np.full(n, -1, dtype=np.int64)
    preds_all = np.empty((n, Y.shape[1]), dtype=np.float32)
    per_fold: list[dict] = []
    null_per_fold: list[list[float]] = []
    covered = np.zeros(n, dtype=bool)
    for f, (tr, te) in enumerate(splits):
        n_train = int(tr.size)
        if n_train <= d:
            raise RuntimeError(
                f"fold {f}: n_train={n_train} <= d={d} — under-determined ambient fit "
                "(plan G2b per-realized-fold assert; #1887)"
            )
        Xtr, Ytr, Xte, Yte = X[tr], Y[tr], X[te], Y[te]
        t0 = time.time()
        preds, info = pf._ridge_gcv_fit_predict(
            Xtr, Ytr, Xte, lambdas=pf.DEFAULT_LAMBDAS, dof_cap=0.9
        )
        t_fit = time.time() - t0
        r2 = pf._r2_matrix(Yte, preds)
        rec: dict = {
            "fold": f,
            "n_train": n_train,
            "n_eval": int(te.size),
            "r2": float(r2),
            "fit_info": info,
            "degenerate": bool(info.get("dof_over_cap", False)),
            "timings_s": {"fit": round(t_fit, 2)},
        }
        if with_reduced:
            t0 = time.time()
            red_r2, red_info = pf._reduced_basis_r2(Xtr, Ytr, Xte, Yte, k=reduced_k)
            rec["reduced_basis"] = {"r2": float(red_r2), "info": red_info}
            rec["timings_s"]["reduced"] = round(time.time() - t0, 2)
        t0 = time.time()
        rec["bootstrap_fold"] = pf._bootstrap_conv_ci(
            Yte, preds, n_draws=bootstrap_draws, seed=p6.unit_seed(*seed_parts, "foldboot", f)
        )
        rec["timings_s"]["bootstrap"] = round(time.time() - t0, 2)
        if with_null:
            t0 = time.time()
            null_r2s, null_info = pf._shuffled_answer_null_r2(
                Xtr,
                Ytr,
                Xte,
                Yte,
                n_draws=n_null_draws,
                seed=p6.unit_seed(*seed_parts, "null", f),
            )
            null_per_fold.append([float(x) for x in null_r2s])
            rec["null_info"] = null_info
            rec["timings_s"]["null"] = round(time.time() - t0, 2)
        pred_id = identity_bias_predict(Xtr, Ytr, Xte)
        rec["identity_bias_r2"] = float(pf._r2_matrix(Yte, pred_id))
        ss_res[te] = ((Yte.astype(np.float64) - preds) ** 2).sum(axis=1)
        ss_res_id[te] = ((Yte.astype(np.float64) - pred_id) ** 2).sum(axis=1)
        folds_of[te] = f
        preds_all[te] = preds.astype(np.float32)
        covered[te] = True
        per_fold.append(rec)
        _log(f"[fit] fold {f + 1}/{len(splits)} r2={r2:+.4f} n_tr={n_train} t={rec['timings_s']}")
    if not covered.all():
        raise RuntimeError("fold splits do not cover every row exactly once")
    fold_mean = float(np.mean([r["r2"] for r in per_fold]))
    out = {
        "per_fold": per_fold,
        "fold_mean_r2": fold_mean,
        "pooled_r2": p6.pooled_r2(ss_res, ss_tot),
        "identity_bias": {
            "pooled_r2": p6.pooled_r2(ss_res_id, ss_tot),
            "per_fold_r2": [r["identity_bias_r2"] for r in per_fold],
        },
        "rowstats": {"folds": folds_of, "ss_res": ss_res, "ss_tot": ss_tot},
        "preds": preds_all,
    }
    if with_null:
        pooled = [x for fold in null_per_fold for x in fold]
        out["null"] = {
            "n_draws_per_fold": n_null_draws,
            "per_fold_draws": null_per_fold,
            "per_fold_p95": [float(np.percentile(fd, 95)) for fd in null_per_fold],
            "pooled_p95": float(np.percentile(pooled, 95)),
            "pooled_median": float(np.median(pooled)),
        }
    return out


def _knn_block(preds: np.ndarray, Y: np.ndarray) -> dict:
    return {
        metric: knn_retrieval(preds, Y, ks=(1, 5, 10), metric=metric)
        for metric in ("euclidean", "cosine")
    }


# ---------------------------------------------------------------------------
# One cell-arm unit
# ---------------------------------------------------------------------------


def _unit_regime(args, fold_map: dict, cell: str, arm: str, layer: int) -> dict:
    return {
        "script_version": SCRIPT_VERSION,
        "cell": cell,
        "arm": arm,
        "layer": int(layer),
        "k": fold_map["k"],
        "seed": fold_map["seed"],
        "n_eq": fold_map["n_eq"],
        "fold_map_sha": fold_map["sha256"],
        "n_null_draws": int(args.n_null_draws),
        "bootstrap_draws": int(args.bootstrap_draws),
        "reduced_k": int(args.reduced_k),
        "seed_derivation": "137-rooted per-(cell,arm,fold) via cm.derived_seed",
    }


def run_fit_unit(args, fold_map: dict, cell: str, arm: str, layer: int) -> dict:
    ledger_root = Path(args.ledger_root)
    out_path = ledger_root / "fits" / f"{cell}__{arm}.json"
    regime = _unit_regime(args, fold_map, cell, arm, layer)
    if out_path.exists():
        prior = json.loads(out_path.read_text(encoding="utf-8"))
        if prior.get("regime") == regime:
            _log(f"[fit] SKIP {cell}/{arm}: output exists with matching regime")
            return prior
        raise RuntimeError(
            f"regime mismatch at {out_path} — on-disk fits were produced under different "
            "parameters; use a fresh ledger root"
        )
    entry = fold_map["cells"][cell]
    store_root = Path(args.store_root)
    if cell in cm.USER_CELLS:
        pair_diag = p6.assert_user_pair(store_root, fold_map, layer)  # §4.2b, before fitting
    else:
        pair_diag = None
    slot = p6.SLOT_BY_ARM[arm]
    pack = p6.load_cell_arrays(
        store_root, cell, layer, (slot, p6.ANSWER_SLOT), row_order=entry["row_ids"]
    )
    X, Y = pack["arrays"][slot], pack["arrays"][p6.ANSWER_SLOT]
    t_unit = time.time()
    res = _fit_under_folds(
        X,
        Y,
        p6.fold_splits(entry),
        seed_parts=(cell, arm),
        n_null_draws=args.n_null_draws,
        bootstrap_draws=args.bootstrap_draws,
        reduced_k=args.reduced_k,
        with_null=True,
    )
    floor = max(res["null"]["pooled_p95"], p6.R2_ABS_FLOOR)
    margin = p6.margin_bootstrap(
        res["rowstats"]["ss_res"],
        res["rowstats"]["ss_tot"],
        floor=floor,
        n_draws=args.bootstrap_draws,
        seed=p6.unit_seed(cell, arm, "margin"),
    )
    payload: dict = {
        "regime": regime,
        "cell": cell,
        "arm": arm,
        "fold_structure": entry["fold_structure"],
        "n_rows": entry["n_rows"],
        "below_n_eq": bool(entry.get("below_n_eq", False)),
        "d": int(X.shape[1]),
        "per_fold": res["per_fold"],
        "fold_mean_r2": res["fold_mean_r2"],
        "pooled_r2": res["pooled_r2"],
        "null": res["null"],
        "floor": floor,
        "margin": {
            "point_fold_mean": res["fold_mean_r2"] - floor,
            "point_pooled": res["pooled_r2"] - floor,
            "bootstrap": margin,
        },
        "tier": p6.tier_from_margin(margin["ci_lo"], margin["ci_hi"]),
        "baselines": {
            "identity_bias": res["identity_bias"],
            "knn": _knn_block(res["preds"], Y),
        },
        "unit_wall_s": round(time.time() - t_unit, 2),
    }
    if cell in cm.STORY_CELLS:
        payload["story_fold_audit"] = entry["story_fold_audit"]
        payload["headline_fold_label"] = "family-held-out"
        if arm == "context":
            comp = _fit_under_folds(
                X,
                Y,
                p6.fold_splits(entry, companion=True),
                seed_parts=(cell, arm, "companion"),
                n_null_draws=args.n_null_draws,
                bootstrap_draws=args.bootstrap_draws,
                reduced_k=args.reduced_k,
                with_null=False,  # companion is a labeled read; the U/tier
                # machinery lives on the PRIMARY fold (documented)
            )
            payload["companion_scene_grain"] = {
                "fold_structure": "scene-grain",
                "label": "within-family companion read (plan §4.4 ii)",
                "per_fold": comp["per_fold"],
                "fold_mean_r2": comp["fold_mean_r2"],
                "pooled_r2": comp["pooled_r2"],
                "identity_bias": comp["identity_bias"],
                "knn": _knn_block(comp["preds"], Y),
                "null_battery": "not run — companion read; nulls/tiers are primary-fold",
            }
    if cell in cm.USER_CELLS:
        payload["user_pair_assert"] = pair_diag
        payload["intersection"] = fold_map["user_intersection"]
        if arm == "context":
            payload["full_cohort_supplementary"] = _user_full_cohort_supplementary(
                args, store_root, cell, layer
            )
    # Sidecars: rowstats (ladder recovery / H3 / H4b inputs) + preds (unit-4
    # retrieval seam; pool = the cell's pooled held-out answers, chance 1/pool).
    rs_path = ledger_root / "fits" / "percell" / f"{cell}__{arm}__rowstats.npz"
    p6.write_rowstats(
        rs_path,
        row_ids=entry["row_ids"],
        folds=res["rowstats"]["folds"],
        ss_res=res["rowstats"]["ss_res"],
        ss_tot=res["rowstats"]["ss_tot"],
    )
    pr_path = ledger_root / "fits" / "preds" / f"{cell}__{arm}__preds.npz"
    p6.write_preds(
        pr_path, row_ids=entry["row_ids"], folds=res["rowstats"]["folds"], preds=res["preds"]
    )
    payload["rowstats_path"] = str(rs_path)
    payload["preds_path"] = str(pr_path)
    payload["retrieval_seam"] = {
        "preds": str(pr_path),
        "pool": "all pooled held-out answers of this cell (== the equalized cohort)",
        "chance_at_1": 1.0 / entry["n_rows"],
        "note": "full retrieval battery (4 conventions + CSLS + fresh-draw refs) is unit 4",
    }
    payload["metadata"] = cm.run_metadata()
    cm.atomic_write_json(out_path, payload)
    _log(
        f"[fit] {cell}/{arm}: fold_mean_r2={payload['fold_mean_r2']:+.4f} "
        f"pooled={payload['pooled_r2']:+.4f} null_p95={res['null']['pooled_p95']:+.4f} "
        f"tier={payload['tier']} wall={payload['unit_wall_s']}s -> {out_path}"
    )
    return payload


def _user_full_cohort_supplementary(args, store_root: Path, cell: str, layer: int) -> dict:
    """Arm-specific FULL-cohort fit (labeled supplementary row, plan §4.2b).

    Own conversation-grouped K=5 folds over the arm's full kept ledger (not the
    intersection); no null battery / tier machinery (supplementary label)."""
    ledger = p6.load_ledger(store_root, cell)
    ids = sorted(r["row_id"] for r in ledger)
    rng = np.random.default_rng(p6.unit_seed(cell, "fullcohort"))
    order = rng.permutation(len(ids))
    folds = np.empty(len(ids), dtype=np.int64)
    for j, chunk in enumerate(np.array_split(order, p6.K_FOLDS)):
        folds[chunk] = j
    entry = {"row_ids": ids, "folds": [int(x) for x in folds]}
    pack = p6.load_cell_arrays(store_root, cell, layer, ("v_C", p6.ANSWER_SLOT), row_order=ids)
    res = _fit_under_folds(
        pack["arrays"]["v_C"],
        pack["arrays"][p6.ANSWER_SLOT],
        p6.fold_splits(entry),
        seed_parts=(cell, "fullcohort"),
        n_null_draws=args.n_null_draws,
        bootstrap_draws=args.bootstrap_draws,
        reduced_k=args.reduced_k,
        with_null=False,
        with_reduced=False,
    )
    return {
        "label": "arm-specific full-cohort supplementary (plan §4.2b)",
        "n_rows": len(ids),
        "fold_mean_r2": res["fold_mean_r2"],
        "pooled_r2": res["pooled_r2"],
        "per_fold_r2": [r["r2"] for r in res["per_fold"]],
        "identity_bias": res["identity_bias"],
        "null_battery": "not run — supplementary row; tiers live on the intersection fit",
    }


# ---------------------------------------------------------------------------
# Units registry / selection
# ---------------------------------------------------------------------------


def all_unit_ids() -> list[str]:
    return [f"own:{cell}:{arm}" for cell in cm.ALL_CELLS for arm in p6.ARMS]


def expand_units(spec: str, available_cells: list[str]) -> list[tuple[str, str]]:
    """Expand --units selectors to (cell, arm) pairs against the realized store."""
    out: list[tuple[str, str]] = []
    for tok in [t.strip() for t in spec.split(",") if t.strip()]:
        if tok == "all":
            out += [(c, a) for c in available_cells for a in p6.ARMS]
        elif tok in ("context", "prefix"):
            out += [(c, tok) for c in available_cells]
        elif tok == "story_q":
            out += [(c, a) for c in available_cells if c in cm.STORY_Q_CELLS for a in p6.ARMS]
        elif tok == "dialog":
            out += [(c, a) for c in available_cells if c in cm.DIALOG_CELLS for a in p6.ARMS]
        elif tok == "user":
            out += [(c, a) for c in available_cells if c in cm.USER_CELLS for a in p6.ARMS]
        elif tok.startswith("own:"):
            _, cell, arm = tok.split(":")
            if cell not in cm.ALL_CELLS or arm not in p6.ARMS:
                raise SystemExit(f"unknown unit {tok}")
            if cell not in available_cells:
                raise SystemExit(f"unit {tok}: cell not in the realized store")
            out.append((cell, arm))
        elif tok in cm.ALL_CELLS:
            if tok not in available_cells:
                raise SystemExit(f"cell {tok} not in the realized store")
            out += [(tok, a) for a in p6.ARMS]
        else:
            raise SystemExit(f"unknown --units token {tok!r}")
    seen: set[tuple[str, str]] = set()
    uniq = []
    for u in out:
        if u not in seen:
            seen.add(u)
            uniq.append(u)
    return uniq


# ---------------------------------------------------------------------------
# Phases
# ---------------------------------------------------------------------------


def _fold_map(args) -> dict:
    """Shared fold-map entry. ``fold_floors_override`` is a PROBE-ONLY Namespace
    attr (never an argparse flag — relaxing the 6,500 floor is plan must-ask);
    the CLI path always builds with production floors."""
    return p6.load_or_build_fold_map(
        Path(args.store_root), Path(args.ledger_root), **getattr(args, "fold_floors_override", {})
    )


def phase_fold_map(args) -> int:
    fm = _fold_map(args)
    _log(f"[fold-map] n_eq={fm['n_eq']} sha={fm['sha256'][:12]} cells={sorted(fm['cells'])}")
    return 0


def phase_g3(args) -> int:
    """Plan §7 G3: the chat/context own-map fit IS the 1-cell production-shape
    pilot; its measured null battery IS the null calibration. Gate reads:
    chat held-out R² > own measured null p95 AND rank-1 retrieval > 10× chance.
    Refusal = rc 3 + persisted report JSON (artifact-routed, never bare rc=1)."""
    ledger_root = Path(args.ledger_root)
    layer = resolve_layer(args)
    fm = _fold_map(args)
    t0 = time.time()
    payload = run_fit_unit(args, fm, "chat", "context", layer)
    wall = time.time() - t0
    knn1 = payload["baselines"]["knn"]["euclidean"]
    acc1 = knn1["acc_at_k"]["1"] if "1" in knn1["acc_at_k"] else knn1["acc_at_k"][1]
    chance1 = knn1["chance_at_k"]["1"] if "1" in knn1["chance_at_k"] else knn1["chance_at_k"][1]
    r2_read = payload["fold_mean_r2"]
    null_p95 = payload["null"]["pooled_p95"]
    pass_r2 = bool(r2_read > null_p95)
    pass_ret = bool(acc1 > G3_RETRIEVAL_CHANCE_MULT * chance1)
    verdict = "PASS" if (pass_r2 and pass_ret) else "REFUSED"
    timings = [r["timings_s"] for r in payload["per_fold"]]
    gate = {
        "gate": "G3",
        "verdict": verdict,
        "reads": {
            "chat_fold_mean_r2": r2_read,
            "chat_pooled_r2": payload["pooled_r2"],
            "own_null_p95": null_p95,
            "r2_gt_null_p95": pass_r2,
            "rank1_acc": float(acc1),
            "rank1_chance": float(chance1),
            "rank1_threshold": G3_RETRIEVAL_CHANCE_MULT * float(chance1),
            "rank1_gt_10x_chance": pass_ret,
        },
        "measured_unit_wall_s": round(wall, 2),
        "per_fold_timings_s": timings,
        "shape": {"n_rows": payload["n_rows"], "d": payload["d"], "k": fm["k"]},
        "fits_json": str(ledger_root / "fits" / "chat__context.json"),
        "abort_semantics": (
            "REFUSED aborts the P6 fan-out (12 ladders + pooled arm measure nothing "
            "about transfer when the source map is indistinguishable from its own null)"
        ),
        "metadata": cm.run_metadata(),
    }
    cm.atomic_write_json(ledger_root / p6.G3_GATE_NAME, gate)
    _log(f"[g3] verdict={verdict} reads={json.dumps(gate['reads'])}")
    return 0 if verdict == "PASS" else G3_RC_REFUSED


def phase_fit(args) -> int:
    ledger_root = Path(args.ledger_root)
    layer = resolve_layer(args)
    fm = _fold_map(args)
    gate_path = Path(args.g3_gate_file or (ledger_root / p6.G3_GATE_NAME))
    p6.require_g3_pass(gate_path)
    units = expand_units(args.units, sorted(fm["cells"]))
    if not units:
        raise SystemExit("--units expanded to an empty unit set")
    _log(f"[fit] {len(units)} units: {[f'{c}/{a}' for c, a in units]}")
    t0 = time.time()
    for i, (cell, arm) in enumerate(units):
        run_fit_unit(args, fm, cell, arm, layer)
        cm.progress("fits", i + 1, len(units), f"{cell}/{arm}", t0)
    return 0


def phase_ratio(args) -> int:
    """H4a user-vs-assistant ceiling ratio (plan §3 H4a-ii): per user arm,
    chat own-ceiling R² / user own-ceiling R², 200-draw conversation-grain
    bootstrap on the USER side (chat point fixed; its fold dispersion
    disclosed), skip-and-count + tier suppression per the reporting tiers.

    G2b user-drop threading (r2 reconciler blocker
    g2b-user-drop-crashes-h4a-ratio): a plan-PERMITTED G2b user-cell drop
    (plan §7 — user cells are non-binding; a drop is reported loudly, never a
    kill) leaves ``fits/<cell>__g2b_dropped.json``. That drop marker IS the
    survivor manifest here and is read DROP-BEFORE-FIT (r3 reconciler blocker
    g2b-drop-marker-shadowed-by-stale-fit): it is AUTHORITATIVE over a
    coexisting fit context — fits are git-harvested + re-materialized
    cross-run while G2b recomputes per dispatch, so a survive->drop flip
    leaves BOTH present and the fit is stale residue. ``--survivors``
    (threaded by the dispatch) keys marker authority to THIS run via
    ``p6.g2b_dropped_now`` — a stale prior-run marker on a now-surviving cell
    is ignored. A dropped arm gets a loud per-arm N/A entry (h4b pattern —
    same expected filename, so the fits-d explicit-path harvest + merge
    digest complete); both arms dropped ⇒ whole-file ``status: N/A``. A
    missing fit for a SURVIVING cell still hard-raises — a real failure."""
    ledger_root = Path(args.ledger_root)
    fits_dir = ledger_root / "fits"
    # G3-gate H4a like phase_fit/phase_pairs (r1 review g3 concern 4): a
    # REFUSED G3 leaves chat__context.json on disk, and ceiling ratios against
    # a null-indistinguishable chat ceiling are uninterpretable.
    p6.require_g3_pass(Path(args.g3_gate_file or (ledger_root / p6.G3_GATE_NAME)))
    chat_path = fits_dir / "chat__context.json"
    if not chat_path.exists():
        raise RuntimeError(f"missing {chat_path} — run the chat fit (phase g3) first")
    chat = json.loads(chat_path.read_text(encoding="utf-8"))
    out: dict = {
        "statistic": "chat own-ceiling R2 / user-arm own-ceiling R2 (H4a-ii)",
        "chat": {
            "pooled_r2": chat["pooled_r2"],
            "fold_mean_r2": chat["fold_mean_r2"],
            "per_fold_r2": [r["r2"] for r in chat["per_fold"]],
            "fold_dispersion_note": "chat ceiling fold dispersion disclosed (plan §3)",
        },
        "reference_7b": {"ratio": 2.5, "source": "#825 trackm_settle_battery (guarded)"},
        "arms": {},
    }
    surv_set = p6.parse_survivors(args.survivors)
    for cell in cm.USER_CELLS:
        upath = fits_dir / f"{cell}__context.json"
        dpath = fits_dir / f"{cell}__g2b_dropped.json"
        if p6.g2b_dropped_now(fits_dir, cell, surv_set):
            # Drop marker checked FIRST, unconditionally (r3 reconciler
            # blocker g2b-drop-marker-shadowed-by-stale-fit): a coexisting
            # fit context is git-re-materialized prior-run residue and must
            # never publish a stale H4a arm.
            if upath.exists():
                _log(
                    f"[h4a] {cell}: G2b drop marker WINS over COEXISTING stale fit "
                    f"{upath.name} (prior-run residue; survive->drop flip) — per-arm N/A"
                )
            out["arms"][cell] = {
                "status": "N/A",
                "reason": (
                    f"user cell {cell} G2b-dropped (below floor) — H4a ceiling ratio "
                    "unformable for this arm (plan §7: user cells are non-binding; "
                    "drop reported loudly, never a kill)"
                ),
                "g2b_drop_record": json.loads(dpath.read_text(encoding="utf-8")),
            }
            _log(f"[h4a] {cell}: N/A — G2b-dropped (loud per-arm skip, plan §7)")
            continue
        if not upath.exists():
            raise RuntimeError(
                f"missing {upath} for a G2b survivor (no authoritative {dpath.name} "
                "drop marker) — run the user fits first"
            )
        ujson = json.loads(upath.read_text(encoding="utf-8"))
        rs = p6.load_rowstats(fits_dir / "percell" / f"{cell}__context__rowstats.npz")
        floor = float(ujson["floor"])
        draws = p6.bootstrap_r2_draws(
            rs["ss_res"], rs["ss_tot"], n_draws=args.bootstrap_draws, seed=p6.unit_seed("h4a", cell)
        )
        ok = np.isfinite(draws) & (draws > floor)
        ratios = chat["pooled_r2"] / draws[ok]
        n_valid = int(ok.sum())
        suppressed_by_draws = n_valid < int(np.ceil(p6.VALID_DRAW_FRAC * args.bootstrap_draws))
        suppressed_by_tier = ujson["tier"] != "clearly-mappable"
        arm_out = {
            "user_pooled_r2": ujson["pooled_r2"],
            "user_fold_mean_r2": ujson["fold_mean_r2"],
            "user_tier": ujson["tier"],
            "user_floor": floor,
            "ratio_point_pooled": (
                chat["pooled_r2"] / ujson["pooled_r2"] if ujson["pooled_r2"] > 0 else float("nan")
            ),
            "ratio_point_fold_mean": (
                chat["fold_mean_r2"] / ujson["fold_mean_r2"]
                if ujson["fold_mean_r2"] > 0
                else float("nan")
            ),
            "n_draws": int(args.bootstrap_draws),
            "n_skipped_ceiling_floor": int(args.bootstrap_draws - n_valid),
            "n_valid": n_valid,
            "suppressed_by_draws": bool(suppressed_by_draws),
            "suppressed_by_tier": bool(suppressed_by_tier),
            "suppressed": bool(suppressed_by_draws or suppressed_by_tier),
            "ratio_draws": [float(x) for x in ratios],
            "user_ceiling_draws": [float(x) for x in draws],
        }
        if not arm_out["suppressed"] and n_valid:
            arm_out["ci_lo"] = float(np.percentile(ratios, 2.5))
            arm_out["ci_hi"] = float(np.percentile(ratios, 97.5))
            arm_out["median"] = float(np.median(ratios))
        out["arms"][cell] = arm_out
        _log(
            f"[h4a] {cell}: point={arm_out['ratio_point_pooled']:.3f} valid={n_valid} "
            f"suppressed={arm_out['suppressed']}"
        )
    na_arms = sorted(c for c, a in out["arms"].items() if a.get("status") == "N/A")
    if len(na_arms) == len(cm.USER_CELLS):
        out["status"] = "N/A"
        out["reason"] = (
            "both user arms G2b-dropped — H4a ceiling-ratio read unformable "
            "(plan §7 skip-and-count; drop records in the per-arm entries)"
        )
        _log("[h4a] whole-file N/A — both user arms G2b-dropped")
    elif na_arms:
        out["na_arms"] = na_arms
    out["metadata"] = cm.run_metadata()
    cm.atomic_write_json(fits_dir / "ratio" / "h4a_ceiling_ratio.json", out)
    return 0


# ---------------------------------------------------------------------------
# Synthetic-store CPU probes (no GPU, no network; tiny n/d)
# ---------------------------------------------------------------------------


def _write_probe_store(root: Path, *, n: int, d: int, seed: int = 7) -> dict:
    """Synthetic store in the EXACT unit-2 capture format (bf16-as-uint16 npz
    parts + rows.json), the 9 active cells, planted linear ground truth. User arms share
    identical v_C bytes by construction; each arm drops a disjoint id set so
    the intersection machinery is exercised."""
    rng = np.random.default_rng(seed)
    W = rng.standard_normal((d, d)) / np.sqrt(d)
    truth: dict[str, dict] = {}
    root.mkdir(parents=True, exist_ok=True)
    shared_c = rng.standard_normal((n, d)).astype(np.float32)  # user arms share v_C
    for cell in cm.ALL_CELLS:
        if cell in cm.USER_CELLS:
            Xf = shared_c.copy()
        else:
            Xf = rng.standard_normal((n, d)).astype(np.float32)
        Yf = (Xf.astype(np.float64) @ W + 0.05 * rng.standard_normal((n, d))).astype(np.float32)
        Pf = rng.standard_normal((n, d)).astype(np.float32)
        ids = [
            f"{cell}_r{i:04d}" if cell not in cm.USER_CELLS else f"conv{i:04d}" for i in range(n)
        ]
        keep = np.ones(n, dtype=bool)
        if cell == "chat_user_real":
            keep[:3] = False  # disjoint drops -> non-trivial intersection
        if cell == "chat_user_sim":
            keep[3:6] = False
        rows = []
        for i in range(n):
            if not keep[i]:
                continue
            r = {
                "row_id": ids[i],
                "n_tokens": 10,
                "v_C_pos": 1,
                "v_P_pos": 0,
                "ans_lo": 2,
                "ans_hi": 9,
            }
            if cell in cm.STORY_CELLS:
                r["final_seed_id"] = int(i % 10)  # 10 families
                r["conv_id"] = None
            else:
                r["conv_id"] = ids[i]
            rows.append(r)
        kept_idx = np.flatnonzero(keep)
        half = len(rows) // 2
        for part, (lo, hi) in enumerate(((0, half), (half, len(rows)))):
            sel = kept_idx[lo:hi]
            arrays = {
                "v_C": p6.encode_bf16_np(Xf[sel]),
                "v_A": p6.encode_bf16_np(Yf[sel]),
                "v_P": p6.encode_bf16_np(Pf[sel]),
                "row_ids": np.array([ids[i] for i in sel]),
                "meta": np.array(
                    json.dumps(
                        {
                            "encoding": "bf16_as_uint16",
                            "cell": cell,
                            "layer": 1,
                            "draw_seed": None,
                            "hidden_size": d,
                        }
                    )
                ),
            }
            with open(root / f"{cell}__part{part:04d}__L1.npz", "wb") as fh:
                np.savez(fh, **arrays)
            cm.atomic_write_json(
                root / f"{cell}__part{part:04d}__rows.json",
                {"cell": cell, "tag": cell, "part": part, "rows": rows[lo:hi]},
            )
        truth[cell] = {"n_kept": len(rows)}
    return truth


# v7: min_dialog=0 (dialogue family descoped — the 9-cell probe store has no
# dialog cells; epm:progress v70 clause 1).
_PROBE_FLOORS = dict(n_eq_floor=30, n_train_floor=8, min_storyq=3, min_dialog=0)


def _probe_expect_raise(fn, needle: str, label: str) -> None:
    try:
        fn()
    except (RuntimeError, TypeError) as e:
        if needle not in str(e):
            raise AssertionError(f"{label}: raised, but message lacks {needle!r}: {e}") from e
        _log(f"[probe] {label}: raised as designed")
        return
    raise AssertionError(f"{label}: expected a raise, got none")


def phase_probe(args) -> int:  # noqa: PLR0915
    """Synthetic-tensor CPU self-verification (see module docstring)."""
    n, d = 40, 8
    with tempfile.TemporaryDirectory(prefix="i2378-fits-probe-") as td:
        tmp = Path(td)
        store = tmp / "store"
        ledger = tmp / "ledger"
        _write_probe_store(store, n=n, d=d)

        # (1) bf16 codec: numpy decode == torch decode, bit-exact.
        import torch

        x = np.random.default_rng(0).standard_normal((5, 7)).astype(np.float32)
        enc = p6.encode_bf16_np(x)
        dec_np = p6.decode_bf16_np(enc)
        dec_torch = (
            (torch.from_numpy(enc.view(np.int16).copy()).view(torch.bfloat16).to(torch.float32))
            .numpy()
            .reshape(enc.shape)
        )
        assert np.array_equal(dec_np, dec_torch), "numpy vs torch bf16 decode mismatch"
        rt = torch.from_numpy(x).to(torch.bfloat16).to(torch.float32).numpy()
        assert np.array_equal(dec_np, rt), "encode_bf16_np is not round-to-nearest-even"
        _log("[probe] bf16 codec: numpy == torch (bit-exact)")

        # (2) fold map through the PRODUCTION path (tiny floors via the
        # probe-only override; writes + sha-pins <ledger>/fold_map.json).
        cells = [c for c in cm.ALL_CELLS if p6.production_part_indices(store, c)]
        ledgers = {c: p6.load_ledger(store, c) for c in cells}
        fm = p6.load_or_build_fold_map(store, ledger, **_PROBE_FLOORS)
        assert fm["user_intersection"]["n_intersection"] == n - 6
        real_e, sim_e = fm["cells"]["chat_user_real"], fm["cells"]["chat_user_sim"]
        assert real_e["row_ids"] == sim_e["row_ids"] and real_e["folds"] == sim_e["folds"]
        for c in cm.STORY_CELLS:
            assert fm["cells"][c]["story_fold_audit"]["verdict"] == "zero-overlap"
        _log(f"[probe] fold map: n_eq={fm['n_eq']} cells={len(fm['cells'])}")

        # (3) planted family-fold overlap -> audit raises.
        bad = list(fm["cells"]["storyq_astra"]["family_keys"])
        bad_folds = list(fm["cells"]["storyq_astra"]["folds"])
        bad_folds[0] = (bad_folds[0] + 1) % 5  # move one scene: its family now spans 2 folds
        _probe_expect_raise(
            lambda: p6.audit_story_folds(bad, bad_folds, 5),
            "family-held-out audit FAILED",
            "planted family overlap",
        )

        # (4) planted n_train floor violation -> build refuses.
        _probe_expect_raise(
            lambda: p6.build_fold_map(
                ledgers, k=5, seed=cm.SEED, **{**_PROBE_FLOORS, "n_train_floor": 40}
            ),
            "n_train",
            "planted n_train floor",
        )

        # (5) user-pair v_C asserts: pass on the honest store, raise on a
        # planted one-row corruption (row located by id INSIDE the equalized
        # intersection cohort — a row outside it would never be loaded).
        diag = p6.assert_user_pair(store, fm, 1)
        assert diag["n_hash_mismatched"] == 0
        target_rid = fm["cells"]["chat_user_sim"]["row_ids"][0]
        npz_path = None
        row_pos = None
        for ci in p6.production_part_indices(store, "chat_user_sim"):
            cand = store / f"chat_user_sim__part{ci:04d}__L1.npz"
            with np.load(cand) as z:
                ids = [str(x) for x in z["row_ids"].tolist()]
                if target_rid in ids:
                    npz_path, row_pos = cand, ids.index(target_rid)
                    arrays = {k: np.asarray(z[k]) for k in z.files}
                    break
        assert npz_path is not None, f"cohort row {target_rid} not found in the sim store"
        arrays["v_C"] = arrays["v_C"].copy()
        arrays["v_C"][row_pos, 0] ^= 1  # flip one bf16 bit
        with open(npz_path, "wb") as fh:
            np.savez(fh, **arrays)
        _probe_expect_raise(
            lambda: p6.assert_user_pair(store, fm, 1),
            "v_C sha256 mismatch",
            "planted v_C corruption",
        )
        arrays["v_C"][row_pos, 0] ^= 1  # restore the flipped bit exactly
        with open(npz_path, "wb") as fh:
            np.savez(fh, **arrays)

        # (6) full-path e2e: G3 pilot on the synthetic chat cell through the
        # production entrypoint code (planted linear truth -> PASS expected).
        ns = argparse.Namespace(
            store_root=str(store),
            ledger_root=str(ledger),
            layer=1,
            layer_star_from=None,
            n_null_draws=8,
            bootstrap_draws=24,
            reduced_k=4,
            units="own:storyq_astra:context,own:chat_user_real:context,own:chat_user_sim:context,own:chat:prefix",
            g3_gate_file=None,
            survivors=None,
            fold_floors_override=_PROBE_FLOORS,
        )
        rc = phase_g3(ns)
        assert rc == 0, f"probe G3 expected PASS rc=0, got {rc}"
        gate = json.loads((ledger / p6.G3_GATE_NAME).read_text(encoding="utf-8"))
        assert gate["verdict"] == "PASS" and gate["reads"]["chat_fold_mean_r2"] > 0.5
        rc = phase_fit(ns)
        assert rc == 0
        story = json.loads((ledger / "fits" / "storyq_astra__context.json").read_text("utf-8"))
        assert story["fold_structure"] == "family-held-out"
        assert story["headline_fold_label"] == "family-held-out"
        assert "companion_scene_grain" in story
        assert len(story["null"]["per_fold_draws"]) == 5
        assert all(len(fd) == 8 for fd in story["null"]["per_fold_draws"])
        assert len(story["margin"]["bootstrap"]["draws"]) == 24
        ureal = json.loads((ledger / "fits" / "chat_user_real__context.json").read_text("utf-8"))
        assert ureal["user_pair_assert"]["n_hash_mismatched"] == 0
        assert "full_cohort_supplementary" in ureal
        rc = phase_ratio(ns)
        assert rc == 0
        h4a = json.loads((ledger / "fits" / "ratio" / "h4a_ceiling_ratio.json").read_text("utf-8"))
        assert set(h4a["arms"]) == set(cm.USER_CELLS)
        assert "status" not in h4a and "na_arms" not in h4a  # healthy run: no N/A residue
        # resume path: re-run one unit -> regime-matched skip (no raise).
        fm2 = _fold_map(ns)
        run_fit_unit(ns, fm2, "chat", "context", 1)
        _log("[probe] e2e fits path (g3 + fit + ratio + resume-skip): OK")

        # (6b) G2b user-drop ratio N/A paths (r2 reconciler blocker
        # g2b-user-drop-crashes-h4a-ratio): a dropped user arm yields a loud
        # per-arm N/A (real-only / sim-only), both dropped yields a whole-file
        # N/A, and a missing fit WITHOUT a drop marker still hard-raises.
        fits_dir = ledger / "fits"
        ratio_path = fits_dir / "ratio" / "h4a_ceiling_ratio.json"
        stash = tmp / "g2b_stash"
        stash.mkdir()
        sim_ctx = fits_dir / "chat_user_sim__context.json"
        real_ctx = fits_dir / "chat_user_real__context.json"

        def _plant_drop(cell: str) -> None:
            cm.atomic_write_json(
                fits_dir / f"{cell}__g2b_dropped.json",
                {"cell": cell, "status": "N/A", "reason": "probe: planted G2b drop"},
            )

        # sim dropped, real survives -> per-arm N/A, real arm still real.
        sim_ctx.rename(stash / sim_ctx.name)
        _plant_drop("chat_user_sim")
        assert phase_ratio(ns) == 0
        h4a = json.loads(ratio_path.read_text("utf-8"))
        assert h4a["arms"]["chat_user_sim"]["status"] == "N/A"
        assert "g2b_drop_record" in h4a["arms"]["chat_user_sim"]
        assert "ratio_point_pooled" in h4a["arms"]["chat_user_real"]
        assert h4a["na_arms"] == ["chat_user_sim"] and "status" not in h4a
        # real dropped too (neither survives) -> whole-file N/A.
        real_ctx.rename(stash / real_ctx.name)
        _plant_drop("chat_user_real")
        assert phase_ratio(ns) == 0
        h4a = json.loads(ratio_path.read_text("utf-8"))
        assert h4a["status"] == "N/A"
        assert all(h4a["arms"][c]["status"] == "N/A" for c in cm.USER_CELLS)
        # real dropped only (sim survives) -> mirror per-arm N/A.
        (stash / sim_ctx.name).rename(sim_ctx)
        (fits_dir / "chat_user_sim__g2b_dropped.json").unlink()
        assert phase_ratio(ns) == 0
        h4a = json.loads(ratio_path.read_text("utf-8"))
        assert h4a["arms"]["chat_user_real"]["status"] == "N/A"
        assert "ratio_point_pooled" in h4a["arms"]["chat_user_sim"]
        assert h4a["na_arms"] == ["chat_user_real"] and "status" not in h4a
        # missing fit for a SURVIVOR (no drop marker) stays a hard raise.
        (fits_dir / "chat_user_real__g2b_dropped.json").unlink()
        _probe_expect_raise(lambda: phase_ratio(ns), "G2b survivor", "survivor-missing ratio raise")
        # restore + healthy re-run rewrites the real-arms artifact.
        (stash / real_ctx.name).rename(real_ctx)
        assert phase_ratio(ns) == 0
        h4a = json.loads(ratio_path.read_text("utf-8"))
        assert "status" not in h4a and "na_arms" not in h4a
        _log("[probe] G2b user-drop ratio N/A paths (per-arm / whole-file / raise): OK")

        # (6c) drop-marker vs COEXISTING fit precedence (r3 reconciler blocker
        # g2b-drop-marker-shadowed-by-stale-fit): the drop marker is checked
        # FIRST, so a stale git-re-materialized fit beside it never publishes;
        # --survivors keys marker authority to the CURRENT dispatch (a stale
        # prior-run marker on a survivor is ignored; a dropped-now cell with
        # no marker raises — the upstream drop-marker write failed).
        _plant_drop("chat_user_sim")
        assert sim_ctx.exists()  # fit + drop marker COEXIST — nothing stashed
        assert phase_ratio(ns) == 0
        h4a = json.loads(ratio_path.read_text("utf-8"))
        assert h4a["arms"]["chat_user_sim"]["status"] == "N/A"
        assert "ratio_point_pooled" not in h4a["arms"]["chat_user_sim"]
        assert "ratio_point_pooled" in h4a["arms"]["chat_user_real"]
        assert h4a["na_arms"] == ["chat_user_sim"] and "status" not in h4a
        # --survivors naming the cell: the stale prior-run marker is IGNORED
        # (drop->survive flip; the current dispatch's survivor set wins).
        ns.survivors = ",".join(sorted(set(cm.ALL_CELLS)))
        assert phase_ratio(ns) == 0
        h4a = json.loads(ratio_path.read_text("utf-8"))
        assert "ratio_point_pooled" in h4a["arms"]["chat_user_sim"]
        assert "na_arms" not in h4a and "status" not in h4a
        # --survivors EXCLUDING a cell that has NO drop marker: fail-loud
        # (the upstream drop-marker write failed; plan-section-7 skip-and-count
        # requires the durable record).
        ns.survivors = ",".join(sorted(set(cm.ALL_CELLS) - {"chat_user_real"}))
        _probe_expect_raise(
            lambda: phase_ratio(ns),
            "drop-marker write failed",
            "dropped-now cell without a marker",
        )
        # marker planted: dropped-now + COEXISTING fit -> per-arm N/A wins.
        _plant_drop("chat_user_real")
        assert real_ctx.exists()
        assert phase_ratio(ns) == 0
        h4a = json.loads(ratio_path.read_text("utf-8"))
        assert h4a["arms"]["chat_user_real"]["status"] == "N/A"
        assert "ratio_point_pooled" in h4a["arms"]["chat_user_sim"]
        assert h4a["na_arms"] == ["chat_user_real"] and "status" not in h4a
        # restore: unlink both markers, flag off, healthy re-run for later probes.
        (fits_dir / "chat_user_sim__g2b_dropped.json").unlink()
        (fits_dir / "chat_user_real__g2b_dropped.json").unlink()
        ns.survivors = None
        assert phase_ratio(ns) == 0
        h4a = json.loads(ratio_path.read_text("utf-8"))
        assert "status" not in h4a and "na_arms" not in h4a
        _log("[probe] G2b drop-marker precedence over coexisting fit + --survivors keying: OK")

        # (7) batched null core vs serial per-draw oracle (same rng sequence).
        rng = np.random.default_rng(11)
        Xt = rng.standard_normal((60, 6))
        Yt = Xt @ rng.standard_normal((6, 6)) + 0.1 * rng.standard_normal((60, 6))
        Xe = rng.standard_normal((20, 6))
        Ye = Xe @ np.zeros((6, 6)) + rng.standard_normal((20, 6))
        draws, _ = pf._shuffled_answer_null_r2(Xt, Yt, Xe, Ye, n_draws=6, seed=13)
        oracle_rng = np.random.default_rng(13)
        for i in range(6):
            perm = oracle_rng.permutation(Yt.shape[0])
            preds, _info = pf._ridge_gcv_fit_predict(Xt, Yt[perm], Xe)
            r2 = pf._r2_matrix(Ye, preds)
            assert abs(r2 - draws[i]) < 1e-8, f"null oracle mismatch draw {i}: {r2} vs {draws[i]}"
        _log("[probe] shuffled-answer null: batched core == serial oracle (6 draws)")

        # (8) skip-and-count on a planted degenerate ceiling.
        n_rows = 50
        ss_tot = np.ones(n_rows)
        healthy = p6.recovery_bootstrap(
            0.2 * np.ones(n_rows), 0.1 * np.ones(n_rows), ss_tot, floor=0.05, n_draws=40, seed=3
        )
        assert healthy["n_skipped_ceiling_floor"] == 0 and not healthy["suppressed"]
        assert abs(healthy["median"] - (0.8 / 0.9)) < 1e-9
        degenerate = p6.recovery_bootstrap(
            0.99 * np.ones(n_rows), 0.97 * np.ones(n_rows), ss_tot, floor=0.05, n_draws=40, seed=3
        )
        assert degenerate["n_skipped_ceiling_floor"] == 40 and degenerate["suppressed"]
        assert degenerate["n_valid"] == 0
        _log("[probe] recovery skip-and-count: healthy 0 skips; degenerate 40/40 + suppressed")

        # (9) margin bootstrap + tier assignment sanity.
        mb = p6.margin_bootstrap(0.1 * np.ones(n_rows), ss_tot, floor=0.05, n_draws=40, seed=5)
        assert p6.tier_from_margin(mb["ci_lo"], mb["ci_hi"]) == "clearly-mappable"
        mb2 = p6.margin_bootstrap(1.5 * np.ones(n_rows), ss_tot, floor=0.05, n_draws=40, seed=5)
        assert p6.tier_from_margin(mb2["ci_lo"], mb2["ci_hi"]) == "clearly-unmappable"
        _log("[probe] margin/tier assignment: OK")
    _log("[phase=probe] done — all fits probes passed")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__.replace("%", "%%"), formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--phase",
        choices=("fold-map", "g3", "fit", "ratio", "probe"),
        default="fit",
    )
    ap.add_argument(
        "--smoke-cell",
        choices=("chat_pilot",),
        default=None,
        help="alias for --phase g3 (the 1-cell production-shape pilot)",
    )
    ap.add_argument(
        "--units",
        default="all",
        help="comma list: all|context|prefix|story_q|dialog|user|<cell>|own:<cell>:<arm>",
    )
    ap.add_argument("--list-units", action="store_true", help="print the unit registry and exit")
    ap.add_argument(
        "--store-root",
        default=str(cm.REPO_ROOT / "data" / "issue_2378" / "activations"),
        help="capture store root (npz + rows.json; unit-2 format)",
    )
    ap.add_argument("--ledger-root", default=str(cm.LEDGER_ROOT))
    ap.add_argument("--layer", type=int, default=None, help="read layer L* (explicit)")
    ap.add_argument(
        "--layer-star-from",
        default=None,
        help="layer_sweep.json path (default <ledger-root>/pilot/layer_sweep.json)",
    )
    ap.add_argument("--n-null-draws", type=int, default=100)
    ap.add_argument("--bootstrap-draws", type=int, default=200)
    ap.add_argument("--reduced-k", type=int, default=1024)
    ap.add_argument("--g3-gate-file", default=None, help="override g3_gate.json path")
    ap.add_argument(
        "--survivors",
        default=None,
        help="CSV of the CURRENT dispatch's G2b survivor set (threaded by the "
        "dispatch at p6.ratio): keys __g2b_dropped.json marker authority to THIS "
        "run — a stale prior-run marker on a surviving cell is ignored. Absent: "
        "the drop marker alone is authoritative (drop-before-fit).",
    )
    return ap


def main() -> int:
    args = build_argparser().parse_args()
    if args.list_units:
        for u in all_unit_ids() + ["g3 (chat/context pilot gate)", "ratio (H4a)"]:
            print(u)
        return 0
    if args.smoke_cell:
        args.phase = "g3"
    if args.phase == "fold-map":
        return phase_fold_map(args)
    if args.phase == "g3":
        return phase_g3(args)
    if args.phase == "fit":
        return phase_fit(args)
    if args.phase == "ratio":
        return phase_ratio(args)
    if args.phase == "probe":
        return phase_probe(args)
    raise SystemExit(f"unknown phase {args.phase}")


if __name__ == "__main__":
    sys.exit(main())
