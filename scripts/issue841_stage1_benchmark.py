#!/usr/bin/env python3
# ruff: noqa: RUF001, RUF002, RUF003
# Intentional Unicode (Δ, ℓ, →, ‖·‖, ĥ, ρ) in scientific docstrings/log messages.
"""Issue #841 Stage 1 — transported-trait-monitor benchmark on the #779 rig.

The single manipulated variable is the projection INPUT; r_B, eval frame, judge
scores, metric, prune + bootstrap protocol are #779-verbatim (imported from
``explore_persona_space.experiments.issue_779.metrics``). Per trait × elicitation
mode × target read-out layer ℓ* × source layer ℓ, the monitor rows (plan §4.4):

  1  transported ⟨ĥ_{ℓ*}(x), r_B(ℓ*)⟩ per class (ridge/MLP matched-information;
     GRU exploratory, prefix-informed)
  1b identity-transport ⟨h_ℓ, r_B(ℓ*)⟩ (the attribution baseline)
  2  raw source ⟨h_ℓ, r_B(ℓ)⟩ (the matched-information fair fight)
  3  raw target ceiling ⟨h_{ℓ*}, r_B(ℓ*)⟩ (DPI bound / retention denominator)
  4  direct-hop ridge ℓ→ℓ*
  5  #779 reference rows (learned map h / direct predictor g) from
     eval_results/issue_779/stage1_headline.json
  6  shuffled-context null (row-permuted context→trajectory pairing)

Target layers: PRIMARY {evil 20, syco 26, halluc 17} + COMPANION {evil 14, syco
19, halluc 24}. Reference-row self-check reproduces #779's raw-PV row within CI at
#779's OWN layers {evil 14, syco 26, halluc 17}. Retention curve = row_r ÷
ceiling_r vs horizon k by JOINT bootstrap (num+denom per replicate; unclipped).
Transport fidelity = eval-context Δ-reconstruction R²/cosine per (class, ℓ, ℓ*).

--smoke runs ONE trait (evil) at its primary ℓ* on a coarse source grid + small
fit split — SAME code path as the full run, one "cell" (trait). Persists
stage1_benchmark.json + retention_curve.json + transport_fidelity.json +
stage1_projections.npz.
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

from explore_persona_space.analysis.vectorized_mlp_skill import SplitMLPGroup  # noqa: E402
from explore_persona_space.experiments.issue_779.metrics import (  # noqa: E402
    bootstrap_delta_ci,  # paired within-condition r delta (H2 headline verdict)
    bootstrap_within_condition_ci,
    within_condition_pearson,
)
from explore_persona_space.experiments.issue_841 import maps as MP  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("issue841_stage1")

EVAL_DIR = PROJECT_ROOT / "eval_results" / "issue_841"
MODES = ("system", "many_shot")
# Transported classes for row 1 (ridge/mlp = matched-information; gru = exploratory,
# prefix-informed). id_transport is row 1b (a separate row). Retention is computed
# for all of {id_transport, ridge, mlp, gru}.
MATCHED_CLASSES = ("ridge", "mlp")
RETENTION_CLASSES = ("id_transport", "ridge", "mlp", "gru")


def _resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        logger.warning("--device cuda requested but no CUDA device; falling back to cpu")
        return "cpu"
    return requested


def schemes_for(trait: str, smoke: bool) -> dict[str, int]:
    """Target-layer schemes for a trait (smoke = primary only)."""
    if smoke:
        return {"primary": C.PRIMARY_TARGET_LAYER[trait]}
    return {"primary": C.PRIMARY_TARGET_LAYER[trait], "companion": C.COMPANION_TARGET_LAYER[trait]}


def needed_transitions(traits: list[str], smoke: bool) -> list[int]:
    """Union of one-step transitions any (trait, scheme, source) transport needs.

    The one-step ridge/MLP maps + the GRU are trait-AGNOSTIC (fit on pass_b, no
    r_B), so this union lets ``main`` fit them ONCE and share across traits
    instead of re-fitting per trait (a 3× waste on the 3-trait run).
    """
    need: set[int] = set()
    for trait in traits:
        for tgt in schemes_for(trait, smoke).values():
            for s in source_grid(tgt, smoke):
                need.update(range(s, tgt))
    return sorted(need)


def source_grid(target: int, smoke: bool) -> list[int]:
    """Source layers for a target ℓ*: every even layer 0..ℓ*−2 plus ℓ*−1 (§4.4).

    Smoke uses a coarse 2-source grid (ℓ*−1 and ℓ*−4) so the CPU transport stays
    fast — same grid-derivation code path, fewer cells.
    """
    if smoke:
        return sorted({max(0, target - 4), target - 1})
    grid = [ell for ell in range(0, target - 1) if ell % 2 == 0]
    if target - 1 not in grid:
        grid.append(target - 1)
    return sorted(set(grid))


# ── map fitting (raw space; identical to Stage-0's raw maps by seed) ──────────


def build_one_step_maps(cx, split, transitions, device, chunk_size, num_threads, max_epochs=None):
    """Fit raw-space one-step ridge + MLP maps for the requested transitions.

    RAW target space (sigma=1.0) — Stage-1 transport is additive composition, so
    each map must emit a RAW Δ̂. Deterministic (seed 42 split + MLP init seed 658)
    ⇒ identical to Stage-0's raw-space maps. Returns (ridge_maps, mlp_maps) dicts
    keyed by transition.
    """
    ridge_maps: dict[int, MP.RidgeMap] = {}
    for t in transitions:
        h, delta = MP.deltas_at(cx, t)
        _pred, rmap = MP.fit_ridge_split(
            h[split["fit"]], delta[split["fit"]], h[split["fit"][:1]], sigma=1.0, device=device
        )
        ridge_maps[t] = rmap
    groups = []
    for t in transitions:
        h, delta = MP.deltas_at(cx, t)
        groups.append(
            SplitMLPGroup(
                key=("mlp", t),
                X_train=h[split["fit"]].astype(np.float32),
                Y_train=delta[split["fit"]].astype(np.float32),
                X_eval=h[split["fit"][:1]].astype(np.float32),  # dummy eval; we want the params
                X_val=h[split["val"]].astype(np.float32),
                Y_val=delta[split["val"]].astype(np.float32),
            )
        )
    _preds, params = MP.fit_split_mlps(
        groups, device=device, chunk_size=chunk_size, num_threads=num_threads, max_epochs=max_epochs
    )
    mlp_maps = {
        t: MP.MLPMap.from_params(params[("mlp", t)], sigma=1.0, device=device) for t in transitions
    }
    logger.info(
        "[maps] fit %d ridge + %d MLP one-step maps (raw space)", len(ridge_maps), len(mlp_maps)
    )
    return ridge_maps, mlp_maps


# ── projections + transport on device ─────────────────────────────────────────


def _proj(h: torch.Tensor, r_b_row: torch.Tensor) -> np.ndarray:
    return (h @ r_b_row).detach().cpu().numpy().astype(np.float64)


def _transport_proj(maps, traj_dev, source, target, r_b_target):
    h_src = traj_dev[:, source, :]
    h_hat = MP.transport_iterated(maps, h_src, source, target)
    return _proj(h_hat, r_b_target), h_hat


# ── metric wrappers (#779 protocol verbatim) ──────────────────────────────────


def method_metrics(x, mat, *, n_boot, seed) -> dict:
    """Within-condition r + bootstrap CI per mode (drop NaN x within a condition)."""
    res = {}
    for mode in MODES:
        cx, cy = C.group_by_condition(x, mat["y"], mat["cond"], mat["mode"], mode)
        cx2, cy2 = [], []
        for xi, yi in zip(cx, cy, strict=True):
            m = np.isfinite(xi)
            if m.sum() >= 3:
                cx2.append(xi[m])
                cy2.append(yi[m])
        res[mode] = bootstrap_within_condition_ci(cx2, cy2, n_boot=n_boot, seed=seed)
    return res


def paired_delta_metrics(x_a, x_b, mat, *, n_boot, seed) -> dict:
    """Paired within-condition r DELTA (method A − method B) + 95% CI per mode.

    #779-verbatim `bootstrap_delta_ci`: A and B are grouped over the SAME conditions
    in the same order (`group_by_condition` keys on cond×mode), and each bootstrap
    replicate resamples conditions ONCE and takes the paired r_A − r_B on that shared
    resample — so the CI reflects the paired comparison the H2 headline decides
    ("transported beats baseline by a margin whose CI excludes 0"). Per mode returns
    {"delta", "lo", "hi", "excludes_zero"}. The projections here (⟨h, r_B⟩) are always
    finite, so no NaN-prune is needed before grouping.
    """
    res = {}
    for mode in MODES:
        cx_a, cy = C.group_by_condition(x_a, mat["y"], mat["cond"], mat["mode"], mode)
        cx_b, _ = C.group_by_condition(x_b, mat["y"], mat["cond"], mat["mode"], mode)
        res[mode] = bootstrap_delta_ci(cx_a, cx_b, cy, n_boot=n_boot, seed=seed)
    return res


def bootstrap_retention_ci(cx_row, cx_ceiling, cy, *, n_boot, seed) -> dict:
    """JOINT bootstrap of the retention ratio row_r / ceiling_r (unclipped).

    Each condition-resample replicate recomputes the row r AND the ceiling r on
    the SAME resampled conditions and takes their ratio — respecting the num↔denom
    correlation from the shared judge target + condition structure (§4.4;
    independently-bootstrapped CIs would be wrong). Retention may exceed 1 (an
    H2-surprise) or be negative — reported as-is, never clipped. cx_row and
    cx_ceiling are per-condition arrays for the SAME conditions in the same order.
    """
    rng = np.random.default_rng(seed)
    r_row = within_condition_pearson(cx_row, cy)["r"]
    r_ceil = within_condition_pearson(cx_ceiling, cy)["r"]
    point = (
        r_row / r_ceil
        if (np.isfinite(r_row) and np.isfinite(r_ceil) and r_ceil != 0.0)
        else float("nan")
    )
    n_cond = len(cy)
    ratios = []
    idx = np.arange(n_cond)
    for _ in range(n_boot):
        samp = rng.choice(idx, size=n_cond, replace=True)
        rr = within_condition_pearson([cx_row[i] for i in samp], [cy[i] for i in samp])["r"]
        rc = within_condition_pearson([cx_ceiling[i] for i in samp], [cy[i] for i in samp])["r"]
        if np.isfinite(rr) and np.isfinite(rc) and rc != 0.0:
            ratios.append(rr / rc)
    if not ratios:
        return {
            "point": point,
            "lo": float("nan"),
            "hi": float("nan"),
            "r_row": r_row,
            "r_ceiling": r_ceil,
            "n_boot_valid": 0,
        }
    return {
        "point": point,
        "lo": float(np.quantile(ratios, 0.025)),
        "hi": float(np.quantile(ratios, 0.975)),
        "r_row": r_row,
        "r_ceiling": r_ceil,
        "n_boot_valid": len(ratios),
    }


def _cond_arrays(x, mat, mode):
    """Per-condition (x, y) arrays with NaN-x pruned (retention bootstrap input)."""
    cx, cy = C.group_by_condition(x, mat["y"], mat["cond"], mat["mode"], mode)
    cx2, cy2 = [], []
    for xi, yi in zip(cx, cy, strict=True):
        m = np.isfinite(xi)
        if m.sum() >= 3:
            cx2.append(xi[m])
            cy2.append(yi[m])
    return cx2, cy2


# ── transport fidelity ────────────────────────────────────────────────────────


def transport_fidelity(h_hat, traj_dev, source, target) -> dict:
    """Eval-context Δ-reconstruction identity-relative R² + cosine of ĥ vs true h."""
    h_true_tgt = traj_dev[:, target, :]
    h_src = traj_dev[:, source, :]
    delta_true = (h_true_tgt - h_src).detach().cpu().numpy()
    delta_hat = (h_hat - h_src).detach().cpu().numpy()
    r2 = MP.identity_relative_r2(delta_hat, delta_true)
    hh = h_hat.detach().cpu().numpy()
    ht = h_true_tgt.detach().cpu().numpy()
    cos = np.sum(hh * ht, axis=1) / (np.linalg.norm(hh, axis=1) * np.linalg.norm(ht, axis=1) + 1e-8)
    return {"delta_recon_r2_id": r2, "cosine_hhat_vs_true": float(np.mean(cos))}


# ── #779 reference rows + self-check ──────────────────────────────────────────


def load_reference_rows(ref_json: Path) -> dict | None:
    if not ref_json.exists():
        logger.warning(
            "[reference] %s absent — #779 reference rows (row 5) + the raw-PV "
            "self-check are UNAVAILABLE this run (sparse-add eval_results/issue_779 "
            "for the full run).",
            ref_json,
        )
        return None
    with open(ref_json) as f:
        return json.load(f)


def reference_selfcheck(ref: dict, trait: str, my_ceiling_at_ref_layer: dict) -> dict:
    """Reproduce #779's raw-PV row within CI at #779's own read-out layer.

    ``my_ceiling_at_ref_layer`` is THIS run's raw-target-ceiling row (= raw-PV,
    ⟨h_{ℓ*}, r_B(ℓ*)⟩) at REFERENCE_ROW_LAYER[trait]; compared to #779's pv_raw
    point ± CI (and the ±0.10 rig-validation band). A miss BLOCKS cross-method
    anchoring — logged loud + recorded pass=False (the run still emits; the
    analyzer/reviewer sees the flag rather than a crash losing the atlas).
    """
    tr = ref.get("traits", {}).get(trait, {})
    pv = tr.get("methods", {}).get("pv_raw", {})
    out = {"reference_read_out_layer": tr.get("read_out_layer"), "pass": True, "checks": {}}
    for mode in MODES:
        ref_pt = pv.get(mode, {}).get("point")
        ref_lo, ref_hi = pv.get(mode, {}).get("lo"), pv.get(mode, {}).get("hi")
        mine = my_ceiling_at_ref_layer.get(mode, {}).get("point")
        within_band = bool(
            mine is not None
            and ref_pt is not None
            and np.isfinite(mine)
            and np.isfinite(ref_pt)
            and abs(mine - ref_pt) <= C.RIG_VALIDATION_BAND
        )
        within_ci = bool(
            mine is not None
            and ref_lo is not None
            and ref_hi is not None
            and np.isfinite(mine)
            and ref_lo <= mine <= ref_hi
        )
        ok = within_band or within_ci
        out["checks"][mode] = {
            "mine": mine,
            "ref_pv_raw": ref_pt,
            "ref_ci": [ref_lo, ref_hi],
            "abs_diff": (abs(mine - ref_pt) if (mine is not None and ref_pt is not None) else None),
            "within_band": within_band,
            "within_ref_ci": within_ci,
            "ok": ok,
        }
        if not ok:
            out["pass"] = False
    if not out["pass"]:
        logger.error(
            "[reference-selfcheck] trait=%s FAILED — raw-PV row does NOT reproduce "
            "#779 within CI/band; cross-method (row 5) anchoring is UNTRUSTED. %s",
            trait,
            out["checks"],
        )
    return out


def _extract_reference_methods(ref: dict, trait: str) -> dict:
    """Pull #779's learned-map h + direct-predictor g rows (per mode) for row 5."""
    tr = ref.get("traits", {}).get(trait, {})
    methods = tr.get("methods", {})
    keep = ["pv_raw", "r1_ridge_cos", "r1_mlp_cos", "direct_ridge", "direct_mlp"]
    out = {"read_out_layer": tr.get("read_out_layer")}
    for m in keep:
        if m in methods:
            out[m] = {mode: methods[m].get(mode) for mode in MODES}
    return out


# ── per-trait processing ──────────────────────────────────────────────────────


def process_trait(trait, cx, split, device, args, ref, maps_bundle, *, smoke) -> dict:
    r_b = C.load_rb(trait)
    cells = C.load_eval_cells(trait)
    mat = C.build_eval_traj_matrix(cells)
    logger.info(
        "[%s] eval matrix: %d (cond,question) rows, %d conditions",
        trait,
        len(mat["y"]),
        len(mat["cond_ids"]),
    )

    schemes = schemes_for(trait, smoke)
    # The one-step maps + GRU + norm-curve are trait-agnostic and fit ONCE in
    # main (shared across traits); direct-hop ridge is trait-specific (source→ℓ*)
    # and fit per cell below.
    ridge_maps = maps_bundle["ridge_maps"]
    mlp_maps = maps_bundle["mlp_maps"]
    gru = maps_bundle["gru"]
    nc = maps_bundle["nc"]
    sig_dev = maps_bundle["sig_dev"]

    traj_dev = torch.from_numpy(np.ascontiguousarray(mat["traj"])).to(
        device=device, dtype=torch.float32
    )
    rb_dev = torch.from_numpy(np.ascontiguousarray(r_b)).to(device=device, dtype=torch.float32)

    tr_out: dict = {
        "n_questions": len(mat["y"]),
        "n_conditions": len(mat["cond_ids"]),
        "schemes": {},
    }
    fidelity: dict = {}
    retention: dict = {}
    proj_store: dict = {}  # per-cell per-question monitor arrays (persisted to npz)

    for scheme, tgt in schemes.items():
        grid = source_grid(tgt, smoke)
        r_b_tgt = rb_dev[tgt]
        ceiling_x = _proj(traj_dev[:, tgt, :], r_b_tgt)  # row 3 (per target)
        ceiling_metrics = method_metrics(ceiling_x, mat, n_boot=args.n_boot, seed=C.BOOTSTRAP_SEED)
        proj_store[f"{scheme}__ceiling"] = ceiling_x
        scheme_out = {"target_layer": tgt, "ceiling_raw_target": ceiling_metrics, "sources": {}}
        fidelity[scheme] = {c: {} for c in ("ridge", "mlp", "gru")}
        retention[scheme] = {c: [] for c in RETENTION_CLASSES}

        for src in grid:
            k = tgt - src
            rows: dict = {}
            # row 2 raw source ⟨h_ℓ, r_B(ℓ)⟩
            row2 = _proj(traj_dev[:, src, :], rb_dev[src])
            rows["raw_source"] = method_metrics(
                row2, mat, n_boot=args.n_boot, seed=C.BOOTSTRAP_SEED
            )
            proj_store[f"{scheme}__{src}__raw_source"] = row2
            # row 1b identity transport ⟨h_ℓ, r_B(ℓ*)⟩
            row1b = _proj(traj_dev[:, src, :], r_b_tgt)
            rows["id_transport"] = method_metrics(
                row1b, mat, n_boot=args.n_boot, seed=C.BOOTSTRAP_SEED
            )
            proj_store[f"{scheme}__{src}__id_transport"] = row1b
            # row 1 transported per matched class
            transported = {"id_transport": row1b}
            for cls, maps in (("ridge", ridge_maps), ("mlp", mlp_maps)):
                x_t, h_hat = _transport_proj(maps, traj_dev, src, tgt, r_b_tgt)
                rows[f"transported_{cls}"] = method_metrics(
                    x_t, mat, n_boot=args.n_boot, seed=C.BOOTSTRAP_SEED
                )
                transported[cls] = x_t
                fidelity[scheme][cls][str(src)] = transport_fidelity(h_hat, traj_dev, src, tgt)
                proj_store[f"{scheme}__{src}__{cls}"] = x_t
            # GRU roll (exploratory, prefix-informed)
            if gru is not None:
                h_hat_gru, div = MP.gru_roll(gru, traj_dev, sig_dev, src, tgt)
                x_gru = _proj(h_hat_gru, r_b_tgt)
                rows["transported_gru"] = method_metrics(
                    x_gru, mat, n_boot=args.n_boot, seed=C.BOOTSTRAP_SEED
                )
                transported["gru"] = x_gru
                fid = transport_fidelity(h_hat_gru, traj_dev, src, tgt)
                fid["divergence_horizon_median"] = _divergence_horizon(div, nc, src)
                fidelity[scheme]["gru"][str(src)] = fid
                proj_store[f"{scheme}__{src}__gru"] = x_gru
            # row 4 direct-hop ridge ℓ→ℓ*
            fit_cx = cx[split["fit"]]  # (n_fit, 28, H)
            direct_map = MP.fit_direct_hop_ridge(
                fit_cx[:, src, :],  # h_source_train
                fit_cx[:, tgt, :],  # h_target_train
                fit_cx[:1, src, :],  # dummy eval; we want the map, not eval preds
                device=device,
            )
            h_hat_direct = traj_dev[:, src, :] + direct_map.apply(traj_dev[:, src, :])
            row4 = _proj(h_hat_direct, r_b_tgt)
            rows["direct_hop_ridge"] = method_metrics(
                row4, mat, n_boot=args.n_boot, seed=C.BOOTSTRAP_SEED
            )
            fidelity[scheme].setdefault("direct_hop", {})[str(src)] = transport_fidelity(
                h_hat_direct, traj_dev, src, tgt
            )
            proj_store[f"{scheme}__{src}__direct_hop"] = row4
            # row 6 shuffled-context null (ridge transport on a row-permuted trajectory)
            rng = np.random.default_rng(C.BOOTSTRAP_SEED + src)
            perm = rng.permutation(traj_dev.shape[0])
            traj_perm = traj_dev[torch.from_numpy(perm).to(device)]
            x_shuf, _ = _transport_proj(ridge_maps, traj_perm, src, tgt, r_b_tgt)
            rows["shuffled_null_ridge"] = method_metrics(
                x_shuf, mat, n_boot=args.n_boot, seed=C.BOOTSTRAP_SEED
            )
            proj_store[f"{scheme}__{src}__shuffled"] = x_shuf

            # PAIRED delta CIs — the H2 headline verdict, decidable from artifacts.
            # For each matched-information transported class (ridge/mlp; gru
            # exploratory when present): transported − raw_source (row 2) AND
            # transported − id_transport (row 1b), per mode, #779 bootstrap_delta_ci.
            # Plus direct-hop − composed-ridge (row 4 vs the iterated ridge).
            deltas: dict = {}
            for cls in ("ridge", "mlp", "gru"):
                if cls not in transported:
                    continue
                deltas[cls] = {
                    "vs_raw_source": paired_delta_metrics(
                        transported[cls], row2, mat, n_boot=args.n_boot, seed=C.BOOTSTRAP_SEED
                    ),
                    "vs_id_transport": paired_delta_metrics(
                        transported[cls], row1b, mat, n_boot=args.n_boot, seed=C.BOOTSTRAP_SEED
                    ),
                }
            deltas["direct_hop_vs_composed_ridge"] = paired_delta_metrics(
                row4, transported["ridge"], mat, n_boot=args.n_boot, seed=C.BOOTSTRAP_SEED
            )

            # retention: row_r / ceiling_r (JOINT bootstrap) per class + mode
            for cls in RETENTION_CLASSES:
                if cls not in transported:
                    continue
                ret_modes = {}
                for mode in MODES:
                    cx_row, cy = _cond_arrays(transported[cls], mat, mode)
                    cx_ceil, _cy2 = _cond_arrays(ceiling_x, mat, mode)
                    # align: retention needs the SAME conditions for row & ceiling.
                    if cx_row and cx_ceil and len(cx_row) == len(cx_ceil):
                        ret_modes[mode] = bootstrap_retention_ci(
                            cx_row, cx_ceil, cy, n_boot=args.n_boot, seed=C.BOOTSTRAP_SEED
                        )
                    else:
                        ret_modes[mode] = {"point": float("nan"), "note": "condition mismatch"}
                retention[scheme][cls].append({"source": src, "horizon_k": k, **ret_modes})

            scheme_out["sources"][str(src)] = {"horizon_k": k, "rows": rows, "deltas": deltas}
            logger.info("[%s/%s] source=%d k=%d done", trait, scheme, src, k)
        tr_out["schemes"][scheme] = scheme_out

    # reference rows + self-check (at #779's own read-out layer)
    if ref is not None:
        ref_layer = C.REFERENCE_ROW_LAYER[trait]
        my_ref_ceiling = method_metrics(
            _proj(traj_dev[:, ref_layer, :], rb_dev[ref_layer]),
            mat,
            n_boot=args.n_boot,
            seed=C.BOOTSTRAP_SEED,
        )
        tr_out["reference_rows"] = _extract_reference_methods(ref, trait)
        tr_out["reference_selfcheck"] = reference_selfcheck(ref, trait, my_ref_ceiling)
        tr_out["reference_selfcheck"]["my_raw_pv_at_ref_layer"] = my_ref_ceiling
        tr_out["reference_selfcheck"]["ref_layer"] = ref_layer

    # k=1 kill-criterion read at PRIMARY layers (ANALYSIS branch — does NOT gate).
    tr_out["kill_read_k1_primary"] = _kill_read(retention.get("primary", {}))
    # Per-(condition,question) unit arrays behind every monitor — the y / cond /
    # mode axes the per-unit scatter (issue841_plots.py) reads as <trait>__y etc.
    # (mode stored as an int code so np.savez needs no allow_pickle).
    proj_store["y"] = mat["y"]
    proj_store["cond"] = mat["cond"]
    proj_store["mode_is_manyshot"] = (mat["mode"] == "many_shot").astype(np.int64)
    del traj_dev, rb_dev
    return {
        "trait_result": tr_out,
        "fidelity": fidelity,
        "retention": retention,
        "proj": proj_store,
    }


def _divergence_horizon(div: np.ndarray, nc: dict, source: int) -> float:
    """Median layer at which the GRU roll's ‖ĥ−h_true‖ exceeds the one-step Δ band.

    Band = the median ‖Δ_ℓ‖ from the norm curve (the natural one-step error scale).
    div[:,k] = ‖ĥ_{source+1+k} − h_true‖. Returns the median (over contexts) offset
    of the first exceedance, or the roll length if never exceeded.
    """
    if div.shape[1] == 0:
        return float("nan")
    band = np.asarray(nc["delta_norm"], dtype=np.float64)[source : source + div.shape[1]]
    band = band.reshape(1, -1)
    exceed = div > band
    n_steps = div.shape[1]
    first = np.where(exceed.any(axis=1), exceed.argmax(axis=1) + 1, n_steps)
    return float(np.median(first))


def _kill_read(primary_retention: dict) -> dict:
    """k=1 retention <~50% of ceiling for ALL gating classes (id/ridge/mlp) at PRIMARY.

    ANALYSIS branch only (does NOT gate the run). The exploratory GRU is reported
    but does not gate. Returns the per-class k=1 retention points + the boolean.
    """
    gating = ("id_transport", "ridge", "mlp")
    k1: dict = {}
    all_below = True
    any_seen = False
    for cls in gating:
        entries = primary_retention.get(cls, [])
        k1_entry = min(entries, key=lambda e: e["horizon_k"], default=None)
        if k1_entry is None:
            continue
        pts = [
            k1_entry[m]["point"]
            for m in MODES
            if isinstance(k1_entry.get(m), dict)
            and k1_entry[m].get("point") is not None
            and np.isfinite(k1_entry[m]["point"])
        ]
        best = max(pts) if pts else float("nan")
        k1[cls] = {"horizon_k": k1_entry["horizon_k"], "retention_point": best}
        if np.isfinite(best):
            any_seen = True
            if best >= 0.5:
                all_below = False
    return {
        "classes": k1,
        "all_below_half": bool(any_seen and all_below),
        "note": "ANALYSIS read only — does NOT gate the run; a True here says "
        "one-step transport is dead at the primary layers (multi-hop would be skipped "
        "in a gated design). GRU excluded (exploratory).",
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Issue #841 Stage 1 transported-monitor benchmark.")
    ap.add_argument("--traits", nargs="+", default=list(C.TRAITS))
    ap.add_argument("--device", default="auto")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--n-contexts", type=int, default=0, help="0 = all pass_b contexts")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--mlp-chunk-size", type=int, default=8)
    ap.add_argument("--num-threads", type=int, default=8)
    ap.add_argument("--gru-epochs", type=int, default=300)
    ap.add_argument("--gru-batch-size", type=int, default=512)
    ap.add_argument(
        "--mlp-epochs",
        type=int,
        default=0,
        help="0 = production default (MLP_MAX_EPOCHS=300); a small value keeps the smoke fast",
    )
    ap.add_argument("--no-gru", action="store_true")
    ap.add_argument(
        "--ref-json",
        type=Path,
        default=PROJECT_ROOT / "eval_results" / "issue_779" / "stage1_headline.json",
    )
    args = ap.parse_args()

    device = _resolve_device(args.device)
    traits = ["evil"] if args.smoke else args.traits
    mlp_epochs = args.mlp_epochs or None  # None => fit_split_mlps uses the production default
    logger.info("device=%s smoke=%s traits=%s", device, args.smoke, traits)

    pass_b = C.load_pass_b()
    cx = pass_b["cx_last"]
    n_total = cx.shape[0]
    if args.smoke:
        cx = cx[: min(args.n_contexts or 200, n_total)]  # --n-contexts shrinks the smoke slice
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

    ref = load_reference_rows(args.ref_json)

    # Fit the trait-agnostic one-step maps + GRU + norm-curve ONCE (shared across
    # traits — they carry no r_B). The union of transitions any trait's transport
    # needs bounds the ridge/MLP fitting.
    transitions = needed_transitions(traits, args.smoke)
    ridge_maps, mlp_maps = build_one_step_maps(
        cx, split, transitions, device, args.mlp_chunk_size, args.num_threads, mlp_epochs
    )
    nc = MP.norm_curve(cx)
    sigma = np.asarray(nc["sigma_block_rms"], dtype=np.float32)
    sig_dev = torch.from_numpy(sigma).to(device)
    gru = None
    if not args.no_gru:
        gru = MP.fit_depth_gru(
            cx[split["fit"]],
            cx[split["val"]],
            sigma,
            device=device,
            max_epochs=args.gru_epochs,
            batch_size=args.gru_batch_size,
        )
    maps_bundle = {
        "ridge_maps": ridge_maps,
        "mlp_maps": mlp_maps,
        "gru": gru,
        "nc": nc,
        "sig_dev": sig_dev,
    }

    benchmark: dict = {
        "traits": {},
        "target_layers": {"primary": C.PRIMARY_TARGET_LAYER, "companion": C.COMPANION_TARGET_LAYER},
        "reference_row_layers": C.REFERENCE_ROW_LAYER,
        "selection_symmetric_nulls_note": (
            "N/A for the primary headline — ℓ* is FIXED a priori (not an argmax over "
            "layers) and the source-layer sweep is reported as a FULL curve (the shape "
            "is the claim), both selection-symmetric-nulls.md carve-outs. No "
            "argmax-over-source headline is formed."
        ),
        "metadata": C.reproducibility_metadata({"phase": "stage1_benchmark", "smoke": args.smoke}),
    }
    retention_all: dict = {"traits": {}}
    fidelity_all: dict = {"traits": {}}
    proj_all: dict = {}

    for trait in traits:
        out = process_trait(trait, cx, split, device, args, ref, maps_bundle, smoke=args.smoke)
        benchmark["traits"][trait] = out["trait_result"]
        retention_all["traits"][trait] = out["retention"]
        fidelity_all["traits"][trait] = out["fidelity"]
        for k, v in out["proj"].items():
            proj_all[f"{trait}__{k}"] = v
        # checkpoint per trait
        C.write_json_atomic(EVAL_DIR / "stage1_benchmark.json", benchmark)
        C.write_json_atomic(EVAL_DIR / "retention_curve.json", retention_all)
        C.write_json_atomic(EVAL_DIR / "transport_fidelity.json", fidelity_all)

    # per-question monitor arrays behind every cell (the #779 per-unit data for plots).
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(EVAL_DIR / "stage1_projections.npz", **proj_all)
    logger.info(
        "[done] wrote stage1_benchmark.json + retention_curve.json + "
        "transport_fidelity.json + stage1_projections.npz"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
