#!/usr/bin/env python3
# ruff: noqa: RUF001, RUF002, RUF003
# Intentional Unicode (Δ, ℓ, →, ‖·‖, ĥ, ρ) in scientific docstrings/log messages.
"""Issue #841 follow-up (gru-source-only) Stage 1 — source-only-GRU transport benchmark.

The single manipulated variable vs the parent GRU is the recurrence INPUT set (whole
trajectory h_0..h_ℓ → single state h_ℓ), realized consistently at fit (single-state
examples, Stage 0) AND inference (MEMORYLESS transport here). The source-only GRU is
fit ONCE (trait-agnostic, RMS-norm σ_m target — the fit space the parent prefix-GRU
transport also uses) and dropped into the parent's ``transport_iterated`` as per-
transition ``GruSourceOnlyMap`` maps, so it rolls with a ZERO hidden state each step
(never ``gru_roll``, which carries the prefix-warmed state).

All PARENT comparison rows (ridge / MLP / prefix-GRU / raw-source / id-transport /
raw-target ceiling / shuffled null) are RELOADED from the committed
``eval_results/issue_841/{stage1_benchmark.json, stage1_projections.npz}`` — NEVER
re-run — and paired unit-for-unit against the fresh source-only-GRU row on the SAME
y/cond/mode axes (the ``.npz`` carries the parent per-context projection arrays the
paired bootstrap needs; the JSON alone cannot pair).

Per (trait × scheme × source-layer ℓ × mode):
- within-condition Pearson r of ⟨ĥ_{ℓ*}(x), r_B(ℓ*)⟩ vs #779's judged score (#779
  protocol verbatim: std<1 prune, min_n=3, bootstrap n≥997 seed 0);
- per-context-paired ``bootstrap_delta_ci`` (source-only-GRU − {ridge, raw_source,
  id_transport, prefix-GRU}), the matched-info fair fight + the matched-on-recipe
  cross-transport-regime read;
- transport fidelity (eval-context Δ-reconstruction r²/cosine) to attribute a null to
  "map failed to transfer" vs "transport uninformative";
- JOINT-bootstrap retention (row_r / ceiling_r).

DV2 headline is the AGGREGATE over the 68-cell PRIMARY grid, all DERIVED in-run under
the conjunction predicate (beats BOTH raw_source AND id_transport, per-context-paired CI
excluding zero): (a) the source-only-GRU win-count, (b) the ridge-alone win-count
(recomputed from the reloaded parent deltas — expect ≈12/68), (c) the chance line (cells
where the shuffled-context null passes the SAME predicate); the source-only-GRU
newly-winning subcount is BH-corrected against the chance line.

A pairing-integrity assert recomputes the parent ridge within-condition r from the npz +
rebuilt eval matrix and STOPs unless it reproduces ``stage1_benchmark.json``'s ridge
``point`` within float tol, BEFORE any paired delta. ``--smoke`` runs ONE trait (evil) at
its primary ℓ* on a coarse source grid — the sweep with one cell.

No Qwen weights, no new judging — analysis over cached tensors only.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import issue841_common as C  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

# REUSE the parent Stage-1 rig verbatim (same #779 protocol, no drift): the metric +
# paired-bootstrap + retention + fidelity + source-grid helpers are imported, not
# re-implemented. The verify gate is imported from the gru-source-only Stage-0 sibling.
from issue841_gru_source_only_stage0 import verify_source_only_gru  # noqa: E402
from issue841_stage1_benchmark import (  # noqa: E402
    MODES,
    _cond_arrays,
    _proj,
    bootstrap_retention_ci,
    method_metrics,
    paired_delta_metrics,
    schemes_for,
    source_grid,
    transport_fidelity,
)

from explore_persona_space.experiments.issue_779.metrics import (  # noqa: E402
    within_condition_pearson,
)
from explore_persona_space.experiments.issue_841 import maps as MP  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("issue841_gru_source_only_stage1")

DEFAULT_OUT_DIR = PROJECT_ROOT / "eval_results" / "issue_841" / "gru_source_only"
DEFAULT_PARENT_EVAL_DIR = PROJECT_ROOT / "eval_results" / "issue_841"
PAIRING_TOL = 1e-6
BH_FDR = 0.05


def _resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        logger.warning("--device cuda requested but no CUDA device; falling back to cpu")
        return "cpu"
    return requested


# ── parent-artifact reload + pairing integrity ─────────────────────────────────


def load_parent(parent_eval_dir: Path) -> tuple[dict, dict]:
    """Reload the committed parent Stage-1 benchmark JSON + per-context projections npz."""
    bench_path = parent_eval_dir / "stage1_benchmark.json"
    npz_path = parent_eval_dir / "stage1_projections.npz"
    for p in (bench_path, npz_path):
        if not p.exists():
            raise FileNotFoundError(
                f"parent artifact {p} not found — REQUIRED to pair the source-only-GRU row "
                "unit-for-unit against ridge/raw_source/id_transport/prefix-GRU (never re-run)."
            )
    with open(bench_path) as f:
        bench = json.load(f)
    npz = dict(np.load(npz_path))
    return bench, npz


def assert_axes_match_npz(trait: str, mat: dict, npz: dict) -> None:
    """Hard STOP unless the freshly-rebuilt eval matrix's y/cond/mode axes == the npz axes.

    The pairing is unit-for-unit, so a row-ordering divergence between the parent's stored
    projections and this run's rebuilt matrix would silently mis-pair the bootstrap.
    """
    y_ref, cond_ref, mode_ref = (
        npz.get(f"{trait}__y"),
        npz.get(f"{trait}__cond"),
        npz.get(f"{trait}__mode_is_manyshot"),
    )
    assert y_ref is not None and cond_ref is not None and mode_ref is not None, (
        f"npz missing shared axes for trait {trait!r}"
    )
    my_mode = (mat["mode"] == "many_shot").astype(np.int64)
    assert len(mat["y"]) == len(y_ref), (trait, len(mat["y"]), len(y_ref))
    assert np.allclose(mat["y"], y_ref, atol=1e-9), f"{trait}: rebuilt y != npz y (row-order drift)"
    assert np.array_equal(mat["cond"], cond_ref), f"{trait}: rebuilt cond != npz cond"
    assert np.array_equal(my_mode, mode_ref), f"{trait}: rebuilt mode != npz mode"


def pairing_integrity_check(trait, scheme, src, mat, npz, bench) -> dict:
    """Recompute parent ridge within-condition r from npz + mat; assert == JSON point (§4.3).

    The stored ``{trait}__{scheme}__{src}__ridge`` projection grouped by mat's (cond, mode)
    MUST reproduce ``stage1_benchmark.json``'s ``transported_ridge[mode].point`` within
    ``PAIRING_TOL`` — a mismatch means the npz axes / unit ordering do not align the JSON,
    so pairing would be mis-aligned. STOP rather than pair mis-aligned units.
    """
    x_ridge = npz[f"{trait}__{scheme}__{src}__ridge"]
    j = bench["traits"][trait]["schemes"][scheme]["sources"][str(src)]["rows"]["transported_ridge"]
    checks = {}
    for mode in MODES:
        cx, cy = _cond_arrays(x_ridge, mat, mode)  # finite ridge proj ⇒ no prune (parent-identical)
        r_mine = within_condition_pearson(cx, cy)["r"]
        r_json = j.get(mode, {}).get("point")
        ok = (
            r_json is not None
            and np.isfinite(r_mine) == np.isfinite(r_json)
            and (not np.isfinite(r_mine) or abs(r_mine - r_json) <= PAIRING_TOL)
        )
        checks[mode] = {"recomputed": float(r_mine), "json_point": r_json, "ok": bool(ok)}
        assert ok, (
            f"pairing-integrity FAILED {trait}/{scheme}/src={src}/{mode}: recomputed ridge r "
            f"{r_mine} != JSON point {r_json} (tol {PAIRING_TOL}); npz axes misaligned — STOP."
        )
    return checks


# ── conjunction win + BH ───────────────────────────────────────────────────────


def conjunction_win_stats(x_t, x_a, x_b, mat, *, n_boot, seed) -> dict:
    """Joint paired bootstrap of (transported − raw_source) AND (transported − id_transport).

    Resamples conditions ONCE per replicate (seed 0, matching the parent's per-marginal
    ``bootstrap_delta_ci`` seed, so the marginal 2.5% quantiles reproduce two separate
    seed-0 calls) and computes BOTH deltas on the shared resample. Per mode returns the
    conjunction ``win`` (both marginal lo > 0 — the identical predicate the ridge-alone
    win uses on the parent's two stored deltas) + the conjunction p-value
    ``P(min(delta_a, delta_b) <= 0)`` used for the BH correction of the newly-winning
    subcount. Projections are finite, so no NaN-prune (parent-identical).
    """
    out: dict = {}
    for mode in MODES:
        cx_t, cy = C.group_by_condition(x_t, mat["y"], mat["cond"], mat["mode"], mode)
        cx_a, _ = C.group_by_condition(x_a, mat["y"], mat["cond"], mat["mode"], mode)
        cx_b, _ = C.group_by_condition(x_b, mat["y"], mat["cond"], mat["mode"], mode)
        rt = within_condition_pearson(cx_t, cy)["r"]
        ra = within_condition_pearson(cx_a, cy)["r"]
        rb = within_condition_pearson(cx_b, cy)["r"]
        n = len(cy)
        idx = np.arange(n)
        rng = np.random.default_rng(seed)
        da_boot, db_boot = [], []
        for _ in range(n_boot):
            samp = rng.choice(idx, size=n, replace=True)
            r_t = within_condition_pearson([cx_t[i] for i in samp], [cy[i] for i in samp])["r"]
            r_a = within_condition_pearson([cx_a[i] for i in samp], [cy[i] for i in samp])["r"]
            r_b = within_condition_pearson([cx_b[i] for i in samp], [cy[i] for i in samp])["r"]
            if np.isfinite(r_t) and np.isfinite(r_a) and np.isfinite(r_b):
                da_boot.append(r_t - r_a)
                db_boot.append(r_t - r_b)
        if not da_boot:
            out[mode] = {
                "win": False,
                "p_conjunction": float("nan"),
                "delta_vs_raw_source": float("nan"),
                "delta_vs_id_transport": float("nan"),
                "da_lo": float("nan"),
                "db_lo": float("nan"),
                "n_boot_valid": 0,
            }
            continue
        da_arr = np.asarray(da_boot)
        db_arr = np.asarray(db_boot)
        da_lo = float(np.quantile(da_arr, 0.025))
        db_lo = float(np.quantile(db_arr, 0.025))
        p_conj = float(np.mean(np.minimum(da_arr, db_arr) <= 0.0))
        out[mode] = {
            "win": bool(da_lo > 0.0 and db_lo > 0.0),
            "p_conjunction": p_conj,
            "delta_vs_raw_source": (rt - ra)
            if np.isfinite(rt) and np.isfinite(ra)
            else float("nan"),
            "delta_vs_id_transport": (rt - rb)
            if np.isfinite(rt) and np.isfinite(rb)
            else float("nan"),
            "da_lo": da_lo,
            "db_lo": db_lo,
            "n_boot_valid": len(da_boot),
        }
    return out


def ridge_alone_win(bench, trait, scheme, src, mode) -> bool:
    """Ridge-alone conjunction win from the reloaded parent deltas (identical predicate)."""
    d = bench["traits"][trait]["schemes"][scheme]["sources"][str(src)]["deltas"]["ridge"]
    lo_r = d["vs_raw_source"].get(mode, {}).get("lo")
    lo_i = d["vs_id_transport"].get(mode, {}).get("lo")
    return bool(lo_r is not None and lo_i is not None and lo_r > 0.0 and lo_i > 0.0)


def bh_survivor_count(pvals: list[float], fdr: float = BH_FDR) -> dict:
    """Benjamini-Hochberg: number of hypotheses surviving FDR control among the p-values."""
    pv = sorted(float(p) for p in pvals if p is not None and np.isfinite(p))
    m = len(pv)
    if m == 0:
        return {"n_tested": 0, "n_survive": 0, "threshold": float("nan"), "fdr": fdr}
    k_max = 0
    thresh = 0.0
    for i, p in enumerate(pv, start=1):
        if p <= (i / m) * fdr:
            k_max = i
            thresh = p
    return {"n_tested": m, "n_survive": k_max, "threshold": thresh, "fdr": fdr}


# ── per-trait processing (source-only-GRU class only; parent rows reloaded) ─────


def process_trait(trait, split, device, args, gru_so_maps, sigma, npz, bench, *, smoke):
    r_b = C.load_rb(trait)
    cells = C.load_eval_cells(trait)
    mat = C.build_eval_traj_matrix(cells)
    assert_axes_match_npz(trait, mat, npz)  # HARD STOP on row-order drift vs the parent npz
    logger.info(
        "[%s] eval matrix: %d (cond,question) rows, %d conditions; axes match parent npz",
        trait,
        len(mat["y"]),
        len(mat["cond_ids"]),
    )

    traj_dev = torch.from_numpy(np.ascontiguousarray(mat["traj"])).to(
        device=device, dtype=torch.float32
    )
    rb_dev = torch.from_numpy(np.ascontiguousarray(r_b)).to(device=device, dtype=torch.float32)

    schemes = schemes_for(trait, smoke)
    tr_out: dict = {
        "n_questions": len(mat["y"]),
        "n_conditions": len(mat["cond_ids"]),
        "schemes": {},
    }
    fidelity: dict = {}
    retention: dict = {}
    proj_store: dict = {}
    cell_records: list[dict] = []  # PRIMARY-scheme per-mode win records for the 68-cell aggregate

    for scheme, tgt in schemes.items():
        grid = source_grid(tgt, smoke)
        r_b_tgt = rb_dev[tgt]
        ceiling_x = npz[f"{trait}__{scheme}__ceiling"]  # reloaded raw-target ceiling (row 3)
        scheme_out = {"target_layer": tgt, "sources": {}}
        fidelity[scheme] = {"gru_source_only": {}}
        retention[scheme] = {"gru_source_only": []}

        for src in grid:
            k = tgt - src
            # pairing integrity BEFORE any paired delta at this cell (STOP on misalign).
            pairing = pairing_integrity_check(trait, scheme, src, mat, npz, bench)

            # source-only-GRU MEMORYLESS transport (GruSourceOnlyMap in transport_iterated).
            with torch.no_grad():
                h_hat = MP.transport_iterated(gru_so_maps, traj_dev[:, src, :], src, tgt)
            x_so = _proj(h_hat, r_b_tgt)
            proj_store[f"{scheme}__{src}__gru_source_only"] = x_so

            row_metrics = method_metrics(x_so, mat, n_boot=args.n_boot, seed=C.BOOTSTRAP_SEED)
            fid = transport_fidelity(h_hat, traj_dev, src, tgt)
            fidelity[scheme]["gru_source_only"][str(src)] = fid

            # reloaded parent comparator arrays (paired unit-for-unit on the shared axes).
            base = f"{trait}__{scheme}__{src}__"
            x_ridge = npz[base + "ridge"]
            x_raw = npz[base + "raw_source"]
            x_id = npz[base + "id_transport"]
            x_prefix = npz[base + "gru"]
            x_shuf = npz[base + "shuffled"]

            # marginal paired deltas (source-only-GRU − each), #779 bootstrap_delta_ci seed 0.
            deltas = {
                "vs_ridge": paired_delta_metrics(
                    x_so, x_ridge, mat, n_boot=args.n_boot, seed=C.BOOTSTRAP_SEED
                ),
                "vs_raw_source": paired_delta_metrics(
                    x_so, x_raw, mat, n_boot=args.n_boot, seed=C.BOOTSTRAP_SEED
                ),
                "vs_id_transport": paired_delta_metrics(
                    x_so, x_id, mat, n_boot=args.n_boot, seed=C.BOOTSTRAP_SEED
                ),
                "vs_prefix_gru": paired_delta_metrics(
                    x_so, x_prefix, mat, n_boot=args.n_boot, seed=C.BOOTSTRAP_SEED
                ),
            }

            # conjunction win + p-value (source-only-GRU) and the chance (shuffled) win.
            conj_so = conjunction_win_stats(
                x_so, x_raw, x_id, mat, n_boot=args.n_boot, seed=C.BOOTSTRAP_SEED
            )
            conj_shuf = conjunction_win_stats(
                x_shuf, x_raw, x_id, mat, n_boot=args.n_boot, seed=C.BOOTSTRAP_SEED
            )

            per_mode_win = {}
            for mode in MODES:
                r_win = ridge_alone_win(bench, trait, scheme, src, mode)
                per_mode_win[mode] = {
                    "gru_source_only_win": conj_so[mode]["win"],
                    "gru_source_only_p_conjunction": conj_so[mode]["p_conjunction"],
                    "ridge_alone_win": r_win,
                    "chance_shuffled_win": conj_shuf[mode]["win"],
                    "newly_winning": bool(conj_so[mode]["win"] and not r_win),
                }
                # only PRIMARY-scheme cells enter the 68-cell headline aggregate.
                if scheme == "primary":
                    cell_records.append(
                        {
                            "trait": trait,
                            "scheme": scheme,
                            "source": src,
                            "mode": mode,
                            # per-cell point deltas feed the AGGREGATE paired-delta (H2
                            # STOP-READs 2/3: "centred on/below zero in aggregate").
                            "delta_vs_ridge": deltas["vs_ridge"][mode].get("delta"),
                            "delta_vs_prefix_gru": deltas["vs_prefix_gru"][mode].get("delta"),
                            **per_mode_win[mode],
                        }
                    )

            # JOINT-bootstrap retention (row_r / ceiling_r) per mode.
            ret_modes = {}
            for mode in MODES:
                cx_row, cy = _cond_arrays(x_so, mat, mode)
                cx_ceil, _cy = _cond_arrays(ceiling_x, mat, mode)
                if cx_row and cx_ceil and len(cx_row) == len(cx_ceil):
                    ret_modes[mode] = bootstrap_retention_ci(
                        cx_row, cx_ceil, cy, n_boot=args.n_boot, seed=C.BOOTSTRAP_SEED
                    )
                else:
                    ret_modes[mode] = {"point": float("nan"), "note": "condition mismatch"}
            retention[scheme]["gru_source_only"].append(
                {"source": src, "horizon_k": k, **ret_modes}
            )

            scheme_out["sources"][str(src)] = {
                "horizon_k": k,
                "rows": {"transported_gru_source_only": row_metrics},
                "deltas": deltas,
                "conjunction": {"gru_source_only": conj_so, "shuffled_null": conj_shuf},
                "win": per_mode_win,
                "pairing_integrity": pairing,
                "transport_fidelity": fid,
            }
            logger.info("[%s/%s] source=%d k=%d done", trait, scheme, src, k)
        tr_out["schemes"][scheme] = scheme_out

    proj_store["y"] = mat["y"]
    proj_store["cond"] = mat["cond"]
    proj_store["mode_is_manyshot"] = (mat["mode"] == "many_shot").astype(np.int64)
    del traj_dev, rb_dev
    return {
        "trait_result": tr_out,
        "fidelity": fidelity,
        "retention": retention,
        "proj": proj_store,
        "cell_records": cell_records,
    }


def _aggregate_paired_delta(cell_records: list[dict], key: str, *, n_boot: int, seed: int) -> dict:
    """Aggregate paired within-condition-r delta over the 68-cell PRIMARY grid (H2 STOP-READs 2/3).

    Aggregation unit (documented per plan §7): the CELL — each of the 68
    (trait × source × mode) primary cells contributes ONE per-cell point delta
    (source-only-GRU − comparator within-condition r). The aggregate point is the
    MEAN of those cell deltas; the 95% CI is a joint bootstrap that resamples the
    CELLS with replacement (seed 0) and re-means per replicate. This answers "is the
    aggregate paired-delta centred on / below / above zero" directly. (Per-context
    pooling across heterogeneous traits is ill-defined — different traits have
    disjoint condition sets — so the cell is the well-defined joint resample unit.)
    """
    unit = (
        "mean over the PRIMARY-grid (trait × source × mode) cells of the per-cell paired "
        "within-condition-r delta (source-only-GRU − comparator); 95% CI by joint bootstrap "
        "resampling the cells with replacement (seed 0)"
    )
    vals = np.asarray(
        [r[key] for r in cell_records if r.get(key) is not None and np.isfinite(r[key])],
        dtype=np.float64,
    )
    if vals.size == 0:
        return {
            "delta": float("nan"),
            "lo": float("nan"),
            "hi": float("nan"),
            "n_boot": 0,
            "n_cells": 0,
            "unit": unit,
        }
    rng = np.random.default_rng(seed)
    m = vals.size
    boot = np.array([vals[rng.choice(m, size=m, replace=True)].mean() for _ in range(n_boot)])
    return {
        "delta": float(vals.mean()),
        "lo": float(np.quantile(boot, 0.025)),
        "hi": float(np.quantile(boot, 0.975)),
        "n_boot": int(n_boot),
        "n_cells": int(m),
        "unit": unit,
    }


def aggregate_wincounts(cell_records: list[dict], *, n_boot: int, seed: int) -> dict:
    """DV2 headline: 68-cell PRIMARY-grid win-counts + aggregate paired-deltas + BH subcount."""
    n_cells = len(cell_records)
    gru_wins = sum(r["gru_source_only_win"] for r in cell_records)
    ridge_wins = sum(r["ridge_alone_win"] for r in cell_records)
    chance = sum(r["chance_shuffled_win"] for r in cell_records)
    newly = [r for r in cell_records if r["newly_winning"]]
    bh = bh_survivor_count([r["gru_source_only_p_conjunction"] for r in newly])
    return {
        "n_cells": n_cells,
        "gru_source_only_wins": gru_wins,
        "ridge_alone_wins": ridge_wins,
        "chance_line_shuffled_wins": chance,
        "newly_winning_subcount": len(newly),
        "newly_winning_bh": bh,
        # Aggregate paired-delta vs the matched-info ridge (the H2 STOP-READ 2/3 statistic)
        # + vs the prefix-GRU (the matched-on-recipe cross-transport-regime bonus read).
        "dv2_aggregate_paired_delta_vs_ridge": _aggregate_paired_delta(
            cell_records, "delta_vs_ridge", n_boot=n_boot, seed=seed
        ),
        "dv2_aggregate_paired_delta_vs_prefix_gru": _aggregate_paired_delta(
            cell_records, "delta_vs_prefix_gru", n_boot=n_boot, seed=seed
        ),
        "predicate": (
            "beats BOTH raw_source AND id_transport, per-context-paired 95% CI lo > 0 "
            "(#779 bootstrap_delta_ci, seed 0)"
        ),
        "aggregate_delta_note": (
            "dv2_aggregate_paired_delta_vs_ridge is the H2 STOP-READ 2/3 statistic — the "
            "aggregate source-only-GRU − ridge within-condition-r delta over the 68-cell grid; "
            "'centred on/below zero' = confirm (ridge stands), CI clear of zero above = flip"
        ),
        "chance_note": (
            "chance line = cells where the shuffled-context-null ridge transport passes the "
            "SAME conjunction predicate; the conjunction makes it a conservative baseline"
        ),
        "bh_note": (
            "newly-winning = source-only-GRU wins where ridge-alone does NOT; BH-corrected at "
            f"FDR {BH_FDR} on the conjunction p-value P(min(Δ_vs_raw, Δ_vs_id) <= 0)"
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Issue #841 gru-source-only Stage 1 transported-monitor benchmark."
    )
    ap.add_argument("--traits", nargs="+", default=list(C.TRAITS))
    ap.add_argument("--device", default="auto")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--n-contexts", type=int, default=0, help="0 = all pass_b contexts")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--gru-epochs", type=int, default=300)
    ap.add_argument("--gru-batch-size", type=int, default=512)
    ap.add_argument(
        "--verify-source-only-gru", action="store_true", help="run only the verify gate"
    )
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--parent-eval-dir", type=Path, default=DEFAULT_PARENT_EVAL_DIR)
    args = ap.parse_args()

    device = _resolve_device(args.device)
    traits = ["evil"] if args.smoke else args.traits
    logger.info("device=%s smoke=%s traits=%s out_dir=%s", device, args.smoke, traits, args.out_dir)

    verify = verify_source_only_gru(device)  # dispatched-path gate FIRST
    if args.verify_source_only_gru:
        logger.info("[verify-only] gate passed; exiting.")
        return 0

    bench, npz = load_parent(args.parent_eval_dir)

    pass_b = C.load_pass_b()
    cx = pass_b["cx_last"]
    n_total = cx.shape[0]
    if args.smoke:
        cx = cx[: min(args.n_contexts or 200, n_total)]
    elif args.n_contexts:
        cx = cx[: args.n_contexts]
    n_total = cx.shape[0]
    split = C.make_split(
        n_total, n_fit=C.N_FIT, n_val=C.N_INNER_VAL, n_test=C.N_TEST, seed=C.SPLIT_SEED
    )
    logger.info(
        "[split] N=%d fit=%d val=%d test=%d",
        n_total,
        len(split["fit"]),
        len(split["val"]),
        len(split["test"]),
    )

    # Fit the RMS-norm source-only GRU ONCE (trait-agnostic; the fit space the parent
    # prefix-GRU transport also uses so the prefix-vs-source read is matched-on-recipe).
    nc = MP.norm_curve(cx)
    sigma = np.asarray(nc["sigma_block_rms"], dtype=np.float64)
    gru_so, gru_diag = MP.fit_depth_gru_source_only(
        cx[split["fit"]],
        cx[split["val"]],
        sigma,
        device=device,
        max_epochs=args.gru_epochs,
        batch_size=args.gru_batch_size,
        transitions=None,  # all transitions — transport composes across the full source grid
    )
    gru_so_maps = {
        m: MP.GruSourceOnlyMap(gru=gru_so, transition=m, sigma_m=float(sigma[m]))
        for m in range(C.N_TRANSITIONS)
    }
    logger.info(
        "[gru_source_only] transport GRU fit (best-val@epoch %d, cap_hit=%s)",
        gru_diag["epochs_to_best_val"],
        gru_diag["cap_hit"],
    )

    benchmark: dict = {
        "traits": {},
        "target_layers": {"primary": C.PRIMARY_TARGET_LAYER, "companion": C.COMPANION_TARGET_LAYER},
        "verify_gate": verify,
        "transport_gru_convergence": gru_diag,
        "selection_symmetric_nulls_note": (
            "N/A for the headline — ℓ* is FIXED a priori (not an argmax over layers) and the "
            "source-layer sweep is reported as a FULL curve (the shape is the claim), both "
            "selection-symmetric-nulls.md carve-outs; no argmax-over-source headline is formed."
        ),
        "metadata": C.reproducibility_metadata(
            {"phase": "stage1_gru_source_only", "smoke": args.smoke}
        ),
    }
    retention_all: dict = {"traits": {}}
    fidelity_all: dict = {"traits": {}}
    proj_all: dict = {}
    all_cell_records: list[dict] = []

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for trait in traits:
        out = process_trait(
            trait, split, device, args, gru_so_maps, sigma, npz, bench, smoke=args.smoke
        )
        benchmark["traits"][trait] = out["trait_result"]
        retention_all["traits"][trait] = out["retention"]
        fidelity_all["traits"][trait] = out["fidelity"]
        all_cell_records.extend(out["cell_records"])
        for kk, vv in out["proj"].items():
            proj_all[f"{trait}__{kk}"] = vv
        # checkpoint per trait
        C.write_json_atomic(args.out_dir / "stage1_gru_source_only.json", benchmark)
        C.write_json_atomic(args.out_dir / "retention_gru_source_only.json", retention_all)
        C.write_json_atomic(args.out_dir / "transport_fidelity_gru_source_only.json", fidelity_all)

    benchmark["dv2_aggregate_wincounts"] = aggregate_wincounts(
        all_cell_records, n_boot=args.n_boot, seed=C.BOOTSTRAP_SEED
    )
    logger.info("[dv2] %s", benchmark["dv2_aggregate_wincounts"])
    C.write_json_atomic(args.out_dir / "stage1_gru_source_only.json", benchmark)
    np.savez(args.out_dir / "gru_source_only_projections.npz", **proj_all)
    logger.info(
        "[done] wrote stage1_gru_source_only.json + retention + fidelity + "
        "gru_source_only_projections.npz to %s",
        args.out_dir,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
