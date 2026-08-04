"""Labeled READOUT arms for #779's frozen 963,444-context RIDGE map on #1739's eval rungs.

Extends ``issue1739_map963k_readout.py`` (projection-only: ``<M(z), r_B>``) to the
LABELED readout arms the #1739 headline actually rests on:

- ``arm7_map_ridge_pred`` — ridge fit ON the map's predicted answer summaries
  (``mp -> dv``), pooled group-fold OOF on the train rung, frozen train->rung
  transfer on the OOD rungs;
- ``arm8_map_ridge_true`` — ridge fit on the REAL answer summaries (``t1 -> dv``),
  applied to the map's predictions at eval time;
- ``arm9_pretrain_ft`` — the closed-form L2-SP arm, included only when the shimmed
  map passes ``arms.verify_arm9_l0_degeneracy``;
- parity anchors ``arm1_ctx_native`` (== committed ``raw_proj``), ``arm6_map_proj_e1``
  (== ``map963k_ridge`` / ``map_i1739_ufull``), ``arm11_oracle_proj``
  (== ``oracle_proj``), ``arm13_shuffled_map`` (== the shuffled controls) — checked
  against the committed ``map963k_reuse/comparison.json`` rows.

RIDGE VARIANT ONLY: the ``mlp_w8192`` 963k map is nonlinear and deliberately out of
scope (user decision) — the MapFit shim below is exact ONLY for the linear ridge
payload.

THE SHIM. ``arms.run_cell_multi``'s map-consuming arms read ``data.mapfit`` and call
``fits.apply_map`` — they do not accept a pre-mapped array. The #779 963k ridge
payload carries self-contained per-layer standardizers applied by
``issue779_ffc_n1m_fits.apply_map`` as ``((X - xmu)/xsd) @ W + ymu`` (fp64), which is
EXACTLY the ``MapFit(kind="linear")`` application contract
(``((x - x_mu)/x_sd) @ w + y_mu``, fp64, per layer). :func:`build_963k_mapfit`
re-packages the per-layer payloads as one linear MapFit; :func:`equivalence_gate`
asserts ``fits.apply_map(x, shim) == issue779_ffc_n1m_fits.apply_map(payload, x)``
on REAL store slices (all layers) before ANY scoring — the two paths differ only in
BLAS reduction order (numpy vs torch fp64 GEMM), so the gate demands agreement to
1e-9 relative and reports the realized residual + bit-identity.

Matched-target by construction (the committed comparison's design): every arm is
scored on the SAME context set, the SAME DV, the SAME r_B, the SAME folds, inside
one process — the 963k readouts vs the in-experiment (#1739 u-full) readouts are the
paired comparison; the committed comparison.json rows anchor the projection arms.

Standing caveats carried per-row / in meta (see ``build_meta``):
- CORPUS: the 963k map is #779's ``mixed_1m`` fit point — 529,085 LMSYS + 434,359
  WildChat contexts (verified from
  ``eval_results/issue_779/fitter-fair-comparison-n1m/n1m_fits.json``); #1739's own
  map is WildChat-only (18,793 pairs). 963k-vs-in-experiment is therefore
  cross-corpus-mix vs in-domain, not merely bigger-vs-smaller.
- PREFIX: #779's map was fit on full-prompt end states (``c_last``); ``prefix_end``
  rows are an out-of-training-distribution application (flagged per row).
- The #1739 u-full map was fit in WHITENED main-grid space and is scored here on RAW
  store summaries — the committed comparison's disclosed cross-space reuse-validity
  read, declared via ``fits.assert_map_input_space`` (never silent).
- These raw-space readout rhos are NOT comparable to the committed whitened-space
  arm-grid values (``arm_results/all_arms_spearman.json``); the in-run i1739-map
  readouts are the matched in-experiment baseline.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

# Thread caps must bind before numpy/torch import (#847).
load_dotenv()

import numpy as np  # noqa: E402

# Sibling module (scripts/ is sys.path[0] in script mode; guarded for -c mode).
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
from issue1739_map963k_readout import (  # noqa: E402
    HIDDEN,
    LAYERS,
    N_BOOT,
    VARIANTS,
    BootCache,
    apply_963k,
    load_963k_payload,
    load_dv,
    load_i1739_map,
    load_per_context,
    load_rb_sources,
    shuffle_rows,
    spearman,
)

logger = logging.getLogger("map963k_labeled_readout")

BEHAVIORS = ("evil", "hallucination", "sycophancy")
CELL_SEED = 42
# The rb source the committed comparison's CI rows used, per behavior — the parity
# anchor is valid only against the SAME source (sycophancy's E1 bank was not
# uploaded when the committed run scored it, so its committed rows are bank-based).
COMMITTED_PRIMARY_RB = {
    "evil": "i1739_e1",
    "hallucination": "i1739_e1",
    "sycophancy": "issue779_bank",
}
MAP963K_REVISION = "9d8f789bf034d8f244e1d00e0dbbe6aba6d272c5"
RB_BANK_REVISION = "037fcbb"
DATA_REPO = "superkaiba1/explore-persona-space-data"
# Mapping (this run's arm slug, map key) -> committed comparison.json arm name.
ANCHOR_ARM_FOR = {
    ("arm1_ctx_e1", None): "raw_proj",
    ("arm11_oracle_proj", None): "oracle_proj",
    ("arm6_map_proj_e1", "map963k_ridge"): "map963k_ridge",
    ("arm13_shuffled_map", "map963k_ridge"): "map963k_ridge_shuffled",
    ("arm6_map_proj_e1", "map_i1739_ufull"): "map_i1739_ufull",
    ("arm13_shuffled_map", "map_i1739_ufull"): "map_i1739_shuffled",
}
RB_INDEP_ARMS = {"arm7_map_ridge_pred", "arm8_map_ridge_true"}


# ------------------------------------------------------------------ input staging


def ensure_inputs(args) -> None:
    """Stage the small inputs (963k ridge weights, i1739 tensors, r_B bank).

    Idempotent + fail-loud via ``hub.stage_hub_file`` (retry-routed). The big
    capture-store slices are staged separately via ``issue1739_map963k_slice.py``
    (resumable tar stream) — this function only VERIFIES they are present.
    """
    from explore_persona_space.orchestrate.hub import stage_hub_file

    for layer in LAYERS:
        stage_hub_file(
            DATA_REPO,
            f"issue779_monitoring/n1m_readout/weights/L{layer}/ridge.pt",
            args.maps_dir / f"L{layer}_ridge.pt",
            revision=MAP963K_REVISION,
        )
    for variant in VARIANTS:
        stage_hub_file(
            DATA_REPO,
            f"issue1739_ctxmap/analysis_tensors/maps/{variant}__ufull.npz",
            args.i1739_dir / f"map_i1739_{variant}_ufull.npz",
        )
    for behavior in [b.strip() for b in args.behaviors.split(",") if b.strip()]:
        # E1 bank: present for evil/hallucination; sycophancy's may not exist —
        # recorded, never fatal (the bank source covers it, as in the committed run).
        try:
            stage_hub_file(
                DATA_REPO,
                f"issue1739_ctxmap/analysis_tensors/r_b_e1/{behavior}.npz",
                args.i1739_dir / f"r_b_e1_{behavior}.npz",
            )
        except Exception as exc:  # noqa: BLE001 — recorded miss; bank source below is fail-loud
            logger.warning("[stage] no r_b_e1 for %s (%s) — bank source only", behavior, exc)
        stage_hub_file(
            DATA_REPO,
            f"issue779_monitoring/r_b/{behavior}.pt",
            args.rb_bank_dir / f"{behavior}.pt",
            revision=RB_BANK_REVISION,
        )
        store_dir = args.slice_root / behavior
        if args.stage_only:
            continue  # tar slices stream separately; presence is a RUN-time gate
        if not (store_dir / "slice_manifest.json").is_file():
            raise FileNotFoundError(
                f"{behavior}: capture-store slice not staged at {store_dir} — run "
                f"scripts/issue1739_map963k_slice.py --behavior {behavior} --dest {store_dir}"
            )


# ------------------------------------------------------------------ MapFit shims


def build_963k_mapfit(payloads: list[dict]):
    """Present the #779 963k RIDGE payloads as one ``MapFit(kind='linear')``.

    ``issue779_ffc_n1m_fits.apply_map`` (ridge branch) computes
    ``((X - xmu)/xsd) @ W + ymu`` with the persisted fp32 tensors upcast to fp64;
    ``fits.apply_map`` (linear branch) computes the identical expression per layer
    from ``(w, x_mu, x_sd, y_mu)`` fp64 arrays. The shim performs the SAME fp32->
    fp64 upcast at build time, so the two paths are algebraically identical and
    differ only in BLAS reduction order (gated by :func:`equivalence_gate`).
    """
    from explore_persona_space.experiments.issue_1739.fits import MapFit

    for p in payloads:
        if p.get("kind") != "ridge":
            raise ValueError(
                f"963k MapFit shim is exact for the linear RIDGE payload only; got "
                f"kind={p.get('kind')!r} (mlp/krr are out of scope this round)"
            )
    w = np.stack([p["W"].cpu().numpy().astype(np.float64) for p in payloads])
    x_mu = np.stack([p["xmu"].cpu().numpy().astype(np.float64).reshape(1, -1) for p in payloads])
    x_sd = np.stack([p["xsd"].cpu().numpy().astype(np.float64).reshape(1, -1) for p in payloads])
    y_mu = np.stack([p["ymu"].cpu().numpy().astype(np.float64).reshape(1, -1) for p in payloads])
    assert w.ndim == 3 and w.shape[1] == w.shape[2], w.shape  # square d x d per layer
    return MapFit(
        w=w,
        x_mu=x_mu,
        x_sd=x_sd,
        y_mu=y_mu,
        diagnostics={
            "source": "issue779 n1m mixed_1m ridge (963,444 train contexts)",
            "weights": f"issue779_monitoring/n1m_readout/weights/L{{L}}/ridge.pt@{MAP963K_REVISION}",
            "layers": [int(p["layer"]) for p in payloads],
        },
        kind="linear",
    )


def build_i1739_mapfit(path: Path):
    """#1739's own u-full linear map at ``LAYERS`` as one MapFit (+ payload meta)."""
    from explore_persona_space.experiments.issue_1739.fits import MapFit

    ws, xms, xss, yms = [], [], [], []
    meta: dict = {}
    for layer in LAYERS:
        w, x_mu, x_sd, y_mu, meta = load_i1739_map(path, layer)
        ws.append(w)
        xms.append(x_mu.reshape(1, -1))
        xss.append(x_sd.reshape(1, -1))
        yms.append(y_mu.reshape(1, -1))
    return (
        MapFit(
            w=np.stack(ws),
            x_mu=np.stack(xms),
            x_sd=np.stack(xss),
            y_mu=np.stack(yms),
            diagnostics={"source": str(path.name), "layers": list(LAYERS)},
            kind="linear",
        ),
        meta,
    )


def equivalence_gate(z: np.ndarray, shim, payloads: list[dict], *, rel_tol: float = 1e-9) -> dict:
    """Assert the shim's ``fits.apply_map`` == #779's canonical ``apply_map``.

    ``z`` is a REAL (Ly, n, d) store slice. Bit-identity is reported but not
    required (numpy vs torch fp64 GEMMs may order reductions differently); the
    bar is 1e-9 RELATIVE max-abs — fp64 GEMM cross-BLAS noise sits ~1e-13, while
    any real mis-normalization (wrong mu/sd/transpose) is O(1). Raises on breach:
    a silently mis-normalized map would produce plausible-looking wrong readouts.
    """
    from explore_persona_space.experiments.issue_1739 import fits

    a = fits.apply_map(z, shim)
    b = np.stack([apply_963k(payloads[li], z[li], "cpu") for li in range(z.shape[0])])
    max_abs = float(np.abs(a - b).max())
    scale = float(np.abs(b).max())
    rel = max_abs / max(scale, 1e-30)
    out = {
        "n_rows": int(z.shape[1]),
        "n_layers": int(z.shape[0]),
        "max_abs_diff": max_abs,
        "output_abs_max": scale,
        "rel_max_abs": rel,
        "bit_identical": bool(np.array_equal(a, b)),
        "rel_tol": rel_tol,
    }
    if rel > rel_tol:
        raise RuntimeError(
            f"963k MapFit shim EQUIVALENCE GATE FAILED: rel max-abs {rel:.3e} > {rel_tol:.0e} "
            f"(max_abs {max_abs:.3e} vs output scale {scale:.3e}) — refusing to score"
        )
    logger.info(
        "[gate] shim equivalence PASS: rel %.3e (max_abs %.3e, scale %.3e, bit_identical=%s, n=%d)",
        rel,
        max_abs,
        scale,
        out["bit_identical"],
        out["n_rows"],
    )
    return out


# ------------------------------------------------------------------ cell plumbing


def committed_shuffled_weights(w: np.ndarray) -> np.ndarray:
    """Per-layer row shuffle under the COMMITTED comparison's rng convention.

    The committed run drew ``shuffle_rows(w, 0)`` (rng key ``[1739, 963, 0]``,
    recreated per call => the SAME permutation each layer). Supplying this as
    ``CellData.w_shuffled`` makes arm13 reproduce the committed shuffled-control
    rows exactly (arms.shuffled_map_weights uses a different key and would yield
    a different — equally valid, but non-anchorable — permutation).
    """
    return np.stack([shuffle_rows(w[li], 0) for li in range(w.shape[0])])


def rung_tables(data: dict, dv_meta: dict) -> dict:
    """Per-context arrays + group keys, split per rung (>=5 contexts)."""
    ctx = data["context_ids"]
    gk = np.array([str(dv_meta[c]["group_key"] or c) for c in ctx])
    rungs = [r for r in sorted(set(data["rung"].tolist())) if int((data["rung"] == r).sum()) >= 5]
    return {"group_key": gk, "rungs": rungs, "masks": {r: data["rung"] == r for r in rungs}}


def stack_variant(data: dict, variant: str) -> np.ndarray:
    """(Ly, n, d) fp64 stack of one kind's per-context summaries."""
    return np.stack([data["per_ctx"][(variant, layer)] for layer in LAYERS])


def _emit_rows(
    *,
    scores_by_regime: list[dict[str, np.ndarray]],
    skipped_by_regime: list[dict[str, str]],
    rb_names: list[str],
    primary_rb: str,
    dv_leg: np.ndarray,
    boot: BootCache,
    behavior: str,
    variant: str,
    rung: str,
    leg: str,
    map_key: str | None,
    n_folds: int,
    fold_scheme: str,
    rows: list[dict],
    skip_rows: list[dict],
) -> None:
    """Score every (arm, regime, layer) of one leg into result rows."""
    dv_std = float(dv_leg.std())
    for r, rb_name in enumerate(rb_names):
        for slug, reason in skipped_by_regime[r].items():
            skip_rows.append(
                {
                    "behavior": behavior,
                    "variant": variant,
                    "eval_rung": rung,
                    "leg": leg,
                    "map": map_key,
                    "arm": slug,
                    "r_b_source": rb_name,
                    "reason": reason,
                }
            )
    for slug in sorted({s for sc in scores_by_regime for s in sc}):
        rb_indep = slug in RB_INDEP_ARMS
        regimes = [0] if rb_indep else range(len(rb_names))
        for r in regimes:
            s_all = scores_by_regime[r].get(slug)
            if s_all is None:
                continue
            rb_name = None if rb_indep else rb_names[r]
            ci_row = rb_indep or rb_names[r] == primary_rb
            for li, layer in enumerate(LAYERS):
                s = np.asarray(s_all[li], dtype=np.float64)
                n_nan = int(np.isnan(s).sum())
                if n_nan:
                    raise RuntimeError(
                        f"{behavior}/{variant}/{rung}/{slug} L{layer}: {n_nan} NaN scores "
                        f"(unsolved fold leaked into the {leg} leg)"
                    )
                lo, hi = boot.ci(s) if ci_row else (None, None)
                rows.append(
                    {
                        "behavior": behavior,
                        "variant": variant,
                        "layer": layer,
                        "eval_rung": rung,
                        "leg": leg,
                        "map": map_key,
                        "arm": slug,
                        "r_b_source": rb_name,
                        "rb_independent": rb_indep,
                        "rho": spearman(s, dv_leg),
                        "ci95": [lo, hi],
                        "ci_computed": bool(ci_row),
                        "n_contexts": int(len(dv_leg)),
                        "dv_std": dv_std,
                        "n_folds": n_folds,
                        "fold_scheme": fold_scheme,
                        "ood_prefix_application": variant == "prefix_end"
                        and map_key == "map963k_ridge",
                    }
                )


def run_behavior(behavior: str, args) -> dict:
    from explore_persona_space.experiments.issue_1739 import arms, fits
    from explore_persona_space.experiments.issue_1739.arms import CellData
    from explore_persona_space.experiments.issue_1739.fits import BudgetCell, realize_budget_cell

    t0 = time.time()
    dv_json = args.dv_root / behavior / "labeling.json"
    store_dir = args.slice_root / behavior
    dv_meta = load_dv(dv_json)
    data = load_per_context(store_dir, dv_meta)
    if args.smoke:
        # Deterministic head-slice per rung: keeps >=2 groups per rung; output goes
        # to the scratch --out, never the canonical artifact (smoke-output rule).
        keep = np.zeros(len(data["context_ids"]), dtype=bool)
        for r in np.unique(data["rung"]):
            keep[np.flatnonzero(data["rung"] == r)[: args.smoke_contexts]] = True
        data["context_ids"] = data["context_ids"][keep]
        data["dv"] = data["dv"][keep]
        data["rung"] = data["rung"][keep]
        data["per_ctx"] = {k: v[keep] for k, v in data["per_ctx"].items()}
    tables = rung_tables(data, dv_meta)
    rungs = tables["rungs"]
    if "train" not in rungs:
        raise RuntimeError(f"{behavior}: no train rung among {rungs}")
    ood_rungs = [r for r in rungs if r != "train"]

    # r_B sources at the run layers.
    rb_sources = load_rb_sources(behavior, args.i1739_dir, bank_dir=args.rb_bank_dir)
    if not rb_sources:
        raise FileNotFoundError(f"{behavior}: no r_B source resolved")
    primary_rb = COMMITTED_PRIMARY_RB[behavior]
    if primary_rb not in rb_sources:
        raise FileNotFoundError(
            f"{behavior}: committed-parity primary r_B source {primary_rb!r} not staged "
            f"(have {sorted(rb_sources)})"
        )
    rb_names = sorted(rb_sources, key=lambda s: (s != primary_rb, s))  # primary first
    rb_stacked = {}
    for name in rb_names:
        rb_all, rb_layers = rb_sources[name]
        rb_stacked[name] = np.stack([rb_all[rb_layers.index(L)] for L in LAYERS])

    # Maps: the 963k ridge shim + #1739's own u-full map, per variant.
    payloads = [load_963k_payload(args.maps_dir, layer, "ridge") for layer in LAYERS]
    if any(p is None for p in payloads):
        raise FileNotFoundError(f"missing 963k ridge payload under {args.maps_dir}")
    map963k = build_963k_mapfit(payloads)
    w_shuf_963k = committed_shuffled_weights(map963k.w)

    boot_by_rung = {r: BootCache(data["dv"][tables["masks"][r]], n_boot=args.n_boot) for r in rungs}

    rows: list[dict] = []
    skip_rows: list[dict] = []
    anchors: list[dict] = []
    gates: dict = {}
    committed = (
        json.loads(args.anchor_json.read_text())["behaviors"].get(behavior, {})
        if args.anchor_json and args.anchor_json.is_file()
        else {}
    )
    committed_rows = committed.get("rows", [])

    for variant in VARIANTS:
        z = stack_variant(data, variant)
        za = stack_variant(data, "t1")

        # ---- equivalence gate on a REAL slice, BEFORE any scoring ----
        n_gate = min(args.gate_rows, z.shape[1])
        gates[f"{variant}_shim_equivalence"] = equivalence_gate(
            np.ascontiguousarray(z[:, :n_gate]), map963k, payloads
        )

        i1739_path = args.i1739_dir / f"map_i1739_{variant}_ufull.npz"
        map_i1739, i1739_meta = build_i1739_mapfit(i1739_path)
        # The committed comparison's DECLARED cross-space read (#1975) — the u-full
        # map was fit in whitened main-grid space, scored here on RAW summaries.
        fits.assert_map_input_space(
            i1739_meta,
            z,
            declared_mismatch=(
                "map963k_labeled_readout scores the whitened-fit u-full map on RAW "
                "per-context store summaries (the committed comparison's disclosed "
                "reuse-validity read; see module docstring)"
            ),
        )
        w_shuf_i1739 = committed_shuffled_weights(map_i1739.w)

        map_specs = [
            ("map963k_ridge", map963k, w_shuf_963k),
            ("map_i1739_ufull", map_i1739, w_shuf_i1739),
        ]

        # ---- train-rung leg: pooled group-fold OOF ----
        tr_mask = tables["masks"]["train"]
        tr_rows_idx = np.flatnonzero(tr_mask)
        z_tr = np.ascontiguousarray(z[:, tr_rows_idx])
        za_tr = np.ascontiguousarray(za[:, tr_rows_idx])
        dv_tr = np.asarray(data["dv"][tr_rows_idx], dtype=np.float64)
        gk_tr = tables["group_key"][tr_rows_idx]
        cell = realize_budget_cell(gk_tr, budget_l=len(gk_tr), draw=0, seed=CELL_SEED)
        assert len(cell.row_idx) == len(gk_tr) and int(cell.row_idx[0]) == 0
        assert np.array_equal(cell.row_idx, np.arange(len(gk_tr))), "cell must cover ALL rows"

        for map_key, mf, w_shuf in map_specs:
            roster = ["arm6_map_proj_e1", "arm7_map_ridge_pred", "arm8_map_ridge_true"]
            if map_key == "map963k_ridge":
                roster = ["arm1_ctx_e1", *roster, "arm11_oracle_proj"]
            roster.append("arm13_shuffled_map")
            # arm-9 gate: run the REAL L->0 degeneracy check on this map; include
            # the arm only when the gate passes (recorded either way).
            gate_data = CellData(
                z_ctx=z_tr,
                dv=dv_tr,
                rb=rb_stacked[primary_rb],
                z_ans=za_tr,
                mapfit=mf,
                layers=LAYERS,
            )
            gate_key = f"{variant}_{map_key}_arm9_gate"
            try:
                arms.verify_arm9_l0_degeneracy(gate_data, device=args.device)
                gates[gate_key] = {"pass": True}
                roster.append("arm9_pretrain_ft")
            except AssertionError as exc:
                gates[gate_key] = {"pass": False, "error": str(exc)}
                logger.warning(
                    "[%s] %s arm9 gate FAILED — arm9 skipped: %s", behavior, map_key, exc
                )

            datas = [
                CellData(
                    z_ctx=z_tr,
                    dv=dv_tr,
                    rb=rb_stacked[name],
                    z_ans=za_tr,
                    mapfit=mf,
                    w_shuffled=w_shuf,
                    layers=LAYERS,
                )
                for name in rb_names
            ]
            t_leg = time.time()
            res = arms.run_cell_multi(datas, cell, arms=roster, device=args.device)
            logger.info(
                "[%s] %s %s train-oof leg: %d arms x %d regimes in %.0fs "
                "(n=%d, %d folds, n_tr/fold~%d vs d=%d)",
                behavior,
                variant,
                map_key,
                len(roster),
                len(datas),
                time.time() - t_leg,
                len(dv_tr),
                cell.n_folds,
                int(len(dv_tr) * (cell.n_folds - 1) / cell.n_folds),
                HIDDEN,
            )
            _emit_rows(
                scores_by_regime=[s for s, _sk in res],
                skipped_by_regime=[sk for _s, sk in res],
                rb_names=rb_names,
                primary_rb=primary_rb,
                dv_leg=dv_tr,
                boot=boot_by_rung["train"],
                behavior=behavior,
                variant=variant,
                rung="train",
                leg="train-oof",
                map_key=map_key,
                n_folds=cell.n_folds,
                fold_scheme=cell.fold_scheme,
                rows=rows,
                skip_rows=skip_rows,
            )

            # ---- OOD rungs: frozen train->rung transfer (run_transfer_cell
            # semantics, multi-regime so the rb-independent fits solve ONCE) ----
            for rung in ood_rungs:
                ev_idx = np.flatnonzero(tables["masks"][rung])
                z_ev = z[:, ev_idx]
                za_ev = za[:, ev_idx]
                dv_ev = np.asarray(data["dv"][ev_idx], dtype=np.float64)
                n_tr, n_ev = len(dv_tr), len(dv_ev)
                comb_z = np.concatenate([z_tr, z_ev], axis=1)
                comb_za = np.concatenate([za_tr, za_ev], axis=1)
                comb_dv = np.concatenate([dv_tr, dv_ev])
                cell_t = BudgetCell(
                    row_idx=np.arange(n_tr + n_ev),
                    fold_ids=np.concatenate(
                        [np.ones(n_tr, dtype=np.int64), np.zeros(n_ev, dtype=np.int64)]
                    ),
                    n_folds=2,
                    budget_l=n_tr,
                    draw=0,
                    seed=CELL_SEED,
                    fold_scheme="transfer-train-vs-eval",
                )
                datas_t = [
                    CellData(
                        z_ctx=comb_z,
                        dv=comb_dv,
                        rb=rb_stacked[name],
                        z_ans=comb_za,
                        mapfit=mf,
                        w_shuffled=w_shuf,
                        layers=LAYERS,
                    )
                    for name in rb_names
                ]
                t_leg = time.time()
                res_t = arms.run_cell_multi(
                    datas_t, cell_t, arms=roster, device=args.device, ridge_folds=(0,)
                )
                logger.info(
                    "[%s] %s %s transfer->%s leg: %.0fs (n_tr=%d n_ev=%d)",
                    behavior,
                    variant,
                    map_key,
                    rung,
                    time.time() - t_leg,
                    n_tr,
                    n_ev,
                )
                _emit_rows(
                    scores_by_regime=[
                        {slug: sc[:, n_tr:] for slug, sc in s.items()} for s, _sk in res_t
                    ],
                    skipped_by_regime=[sk for _s, sk in res_t],
                    rb_names=rb_names,
                    primary_rb=primary_rb,
                    dv_leg=dv_ev,
                    boot=boot_by_rung[rung],
                    behavior=behavior,
                    variant=variant,
                    rung=rung,
                    leg="transfer",
                    map_key=map_key,
                    n_folds=2,
                    fold_scheme="transfer-train-vs-eval",
                    rows=rows,
                    skip_rows=skip_rows,
                )
                del comb_z, comb_za, datas_t
        del z, za, z_tr, za_tr

    # dv-degeneracy flag, merge-script convention: dv_std < 0.15 * max rung dv_std.
    dv_max = max(float(data["dv"][tables["masks"][r]].std()) for r in rungs)
    for row in rows:
        row["dv_degenerate"] = bool(dv_max > 0 and row["dv_std"] < 0.15 * dv_max)

    # ---- parity anchors vs the committed comparison.json ----
    for row in rows:
        key = (
            row["arm"],
            None if row["arm"] in ("arm1_ctx_e1", "arm11_oracle_proj") else row["map"],
        )
        committed_arm = ANCHOR_ARM_FOR.get(key)
        if committed_arm is None or row["r_b_source"] is None:
            continue
        match = [
            c
            for c in committed_rows
            if c["arm"] == committed_arm
            and c["variant"] == row["variant"]
            and c["layer"] == row["layer"]
            and c["eval_rung"] == row["eval_rung"]
            and c["r_b_source"] == row["r_b_source"]
        ]
        if not match:
            continue
        c = match[0]
        anchors.append(
            {
                "behavior": behavior,
                "variant": row["variant"],
                "layer": row["layer"],
                "eval_rung": row["eval_rung"],
                "leg": row["leg"],
                "arm": row["arm"],
                "committed_arm": committed_arm,
                "r_b_source": row["r_b_source"],
                "committed_rho": c["rho"],
                "recomputed_rho": row["rho"],
                "abs_gap": abs(row["rho"] - c["rho"])
                if row["rho"] == row["rho"] and c["rho"] is not None
                else None,
            }
        )

    return {
        "behavior": behavior,
        "n_contexts": int(len(data["context_ids"])),
        "rungs": rungs,
        "rb_sources": rb_names,
        "primary_rb": primary_rb,
        "gates": gates,
        "rollout_prompt_side_min_cosine": data["rollout_prompt_side_min_cosine"],
        "rows": rows,
        "skipped": skip_rows,
        "parity_anchor": anchors,
        "elapsed_s": round(time.time() - t0, 1),
    }


# ------------------------------------------------------------------------ meta


def build_meta(args, fitters=("ridge",)) -> dict:
    from explore_persona_space.orchestrate.provenance import as_metadata_dict, git_provenance

    return {
        "question": (
            "do the LABELED readout arms (arm7 ridge-on-predicted / arm8 ridge-on-true-"
            "applied-to-predicted) hold up when the in-experiment 18,793-pair WildChat map "
            "is swapped for #779's frozen 963,444-context mixed-corpus RIDGE map?"
        ),
        "design": {
            "train_leg": "pooled group-fold OOF on the train rung (realize_budget_cell over "
            "ALL train-rung rows, 5 group-round-robin folds, seed 42)",
            "transfer_leg": "frozen train->rung transfer (run_transfer_cell semantics: 2 "
            "contiguous folds, ridge_folds=(0,), eval block scored by arms fit on the FULL "
            "train rung, never on eval DV)",
            "matched_target": "every arm scored on the SAME context set / DV / r_B / folds "
            "inside one process; 963k vs in-experiment map is the paired comparison",
            "rb_independent_hoist": "arm7/arm8 fits are pure functions of the row set "
            "(ARM_REGISTRY rb_dep=False); computed ONCE per (map, variant, row-set) and "
            "shared across r_B regimes via arms.run_cell_multi's multi-regime datas — the "
            "PR #1757 rbcache-hoist keying applied through the API",
            "estimator": "fits.ridge_gcv_predict_per_target (per-target GCV, one shared "
            "Gram+eigh per (source, fold); n_train > d=3584 on every fit — primal regime)",
        },
        "map963k_source": {
            "repo": DATA_REPO,
            "weights_prefix": "issue779_monitoring/n1m_readout/weights/",
            "revision": MAP963K_REVISION,
            "layers": list(LAYERS),
            "fitters": list(fitters),
            "train_contexts": 963444,
            "training_corpus": {
                "verified_from": "eval_results/issue_779/fitter-fair-comparison-n1m/"
                "n1m_fits.json .per_point.mixed_1m.selection",
                "mode": "mixed",
                "n_lmsys": 529085,
                "n_wildchat": 434359,
                "n_realized": 963444,
                "note": "the 963k map is #779's mixed_1m fit point — a MIXED "
                "LMSYS(55%)+WildChat(45%) corpus, NOT pure LMSYS (the glossary's "
                "'LMSYS-lineage' shorthand is imprecise for this point). #1739's own map "
                "is WildChat-only (18,793 pairs from the #1092 store), so every "
                "963k-vs-in-experiment comparison here is CROSS-CORPUS-MIX vs IN-DOMAIN, "
                "not merely bigger-vs-smaller: scale and corpus domain are partially "
                "confounded and a 963k win/loss must not be narrated as a pure scale "
                "effect.",
            },
            "input_semantics": "c_last (last prompt token, chat template + generation prompt)",
            "target_semantics": "v_x (mean-response activation) == #1739 t1 (answer-span mean)",
        },
        "i1739_map_source": {
            "prefix": "issue1739_ctxmap/analysis_tensors/maps/{variant}__ufull.npz",
            "w_fit_rows": 18793,
            "fit_corpus": "WildChat (#1092 store)",
            "cross_space_note": "fit in WHITENED main-grid space, scored here on RAW "
            "per-context store summaries — the committed comparison's disclosed "
            "reuse-validity read (fits.assert_map_input_space declared_mismatch); these "
            "raw-space rhos are NOT comparable to the committed whitened-space arm-grid "
            "values in arm_results/all_arms_spearman.json",
        },
        "caveats": [
            "prefix_end x map963k rows are an OUT-OF-TRAINING-DISTRIBUTION application: "
            "#779's map was fit on full-prompt end states (c_last), not prefix ends "
            "(flagged per-row as ood_prefix_application).",
            "dv_degenerate rows (dv_std < 0.15 x the behavior's max rung dv_std, the "
            "merge-script convention — e.g. evil/hhrt dv_std 0.89 vs train 26.3) have "
            "near-constant DV; their rankings are uninformative and must not be plotted "
            "as signal.",
            "arm13 shuffled controls use the COMMITTED comparison's rng convention "
            "(shuffle_rows key [1739,963,0], same permutation per layer) so the anchor "
            "binds; arms.shuffled_map_weights' own key would give a different draw.",
            "sycophancy r_B: the committed comparison scored sycophancy with the "
            "issue779_bank source only (its E1 bank was in flight); primary_rb follows "
            "the committed choice per behavior for anchor validity.",
        ],
        "metric": (
            "Spearman rho of arm score vs judged graded DV, per eval rung; "
            f"{args.n_boot}-draw percentile bootstrap over contexts (BootCache, shared "
            "index matrix per rung; CI on the primary r_B source + rb-independent arms)"
        ),
        "n_boot": args.n_boot,
        "device": args.device,
        "smoke": bool(args.smoke),
        "git": as_metadata_dict(git_provenance(cwd=_HERE)),
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


# ---------------------------------------------------------------------- driver


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--behaviors", default="evil,hallucination,sycophancy")
    ap.add_argument("--slice-root", type=Path, default=Path("data/issue_1739/hf_dl/evalslice"))
    ap.add_argument("--maps-dir", type=Path, default=Path("data/issue_1739/hf_dl/map963k"))
    ap.add_argument("--i1739-dir", type=Path, default=Path("data/issue_1739/hf_dl/i1739_tensors"))
    ap.add_argument("--rb-bank-dir", type=Path, default=Path("data/issue_1739/hf_dl/r_b"))
    ap.add_argument("--dv-root", type=Path, default=Path("eval_results/issue_1739/dv_dataset"))
    ap.add_argument(
        "--anchor-json",
        type=Path,
        default=Path("eval_results/issue_1739/map963k_reuse/comparison.json"),
        help="committed 963k comparison (parity anchor); missing file => empty anchors",
    )
    ap.add_argument(
        "--out-dir", type=Path, default=Path("eval_results/issue_1739/map963k_readouts")
    )
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--gate-rows", type=int, default=256)
    ap.add_argument("--smoke", action="store_true", help="tiny slice; write to a scratch --out-dir")
    ap.add_argument("--smoke-contexts", type=int, default=300)
    ap.add_argument("--stage-only", action="store_true", help="stage small inputs and exit")
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stdout
    )
    if args.smoke and "eval_results" in str(args.out_dir):
        raise SystemExit("--smoke must write to a scratch --out-dir, never eval_results/")
    ensure_inputs(args)
    if args.stage_only:
        logger.info("[stage] small inputs staged; exiting (--stage-only)")
        sys.stdout.flush()
        sys.exit(0)

    t0 = time.time()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    behaviors = [b.strip() for b in args.behaviors.split(",") if b.strip()]
    for behavior in behaviors:
        result = {"meta": build_meta(args), "result": run_behavior(behavior, args)}
        result["meta"]["elapsed_s"] = round(time.time() - t0, 1)
        out = args.out_dir / f"readouts_{behavior}.json"
        tmp = out.with_name(out.name + ".tmp")
        tmp.write_text(json.dumps(result, indent=2))
        tmp.replace(out)
        logger.info("wrote %s (%.0fs)", out, time.time() - t0)
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(0)


if __name__ == "__main__":
    main()
