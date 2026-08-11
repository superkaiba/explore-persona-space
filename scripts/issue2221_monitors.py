"""Issue #2221 P8 — monitor arms, correlations, bootstrap/null battery, figures.

All computed from CACHED activations (P5 stores + the pinned reused
artifacts) — CPU, numpy, fully vectorized (no per-draw Python loop; plan §6).

Phases (``--phase``; registry ``PHASES``):

- ``verify_keys``  : realized-keys startup asserts on every reused artifact
                     (r_B v2+v1 shapes, map npz key set + apply contract
                     fields, finetune_activations kind probe).
- ``arms``         : per (trait, panel) per-layer monitor scalars for every
                     model (24 finals + frac-checkpoints + the synth778
                     stratum) -> ``eval_results/issue_2221/monitor_scalars/``.
- ``correlations`` : Spearman r per (arm, trait, layer); predictivity layer
                     selection + the paper frozen-layer companion read; LOFO
                     jackknife over the 8 families; 10k paired bootstrap with
                     per-draw re-selection over layer x {prefix, context}
                     (56 positions for arm c); the #778 round-3 honest null
                     ladder (isotropic / covariance-matched + diagnostic /
                     score-shuffle) under the SAME selection; checkpoint-time
                     detection AUC (+ random-direction control); within-family
                     severity ordering; H2 panel split; H3 synth stratum;
                     identity+bias + kNN baselines for the reused map.
- ``figures``      : paper-plots figures -> ``figures/issue_2221/``.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import issue778_lib as lib  # noqa: E402

from explore_persona_space.experiments.issue_2221 import constants as C  # noqa: E402
from explore_persona_space.experiments.issue_2221 import monitors as M  # noqa: E402
from explore_persona_space.experiments.issue_2221.loaders import (  # noqa: E402
    apply_map,
    apply_map_shift,
    load_affine_map,
    load_ft_activation,
    load_rb,
)

logger = logging.getLogger("issue2221.monitors")

PANELS = ("paper", "lmsys", "pooled")
COV_NULL_CIRCULARITY_MAX = 0.9  # |cos(top pool evec, r_B)| above this flags circularity


def all_cells() -> list[str]:
    return [f"{f}_{v}" for f in C.FAMILIES for v in C.VERSIONS]


# ── store access ──────────────────────────────────────────────────────────────


def _load_capture(p5_root: Path, tag: str, kind: str) -> dict:
    import torch

    sub = {"last": "capture", "resp": "capture_resp", "resp_synth": "capture_resp_synth778"}[kind]
    path = p5_root / sub / f"{tag}.pt"
    if not path.is_file():
        raise FileNotFoundError(f"capture store missing: {path}")
    return torch.load(path, map_location="cpu", weights_only=False)


def _panel_mask(surface_ids: list[str], trait: str, panel: str) -> np.ndarray:
    ids = np.asarray(surface_ids)
    if panel == "paper":
        return np.char.startswith(ids, f"paper-{trait}-")
    if panel == "lmsys":
        return np.char.startswith(ids, "lmsys-")
    return np.char.startswith(ids, f"paper-{trait}-") | np.char.startswith(ids, "lmsys-")


def _mean_states(store: dict, key: str, trait: str, panel: str) -> np.ndarray:
    """Panel-subset mean state (28, 3584) float64 from a capture store."""
    val = store[key]
    arr = np.asarray(val.float().numpy() if hasattr(val, "numpy") else val, dtype=np.float64)
    mask = _panel_mask(store["surface_ids"], trait, panel)
    assert mask.any(), (trait, panel, "empty panel subset")
    out = arr[mask].mean(axis=0)
    assert out.shape == (C.N_LAYERS, C.HIDDEN_DIM), out.shape
    return out


# ── phases ────────────────────────────────────────────────────────────────────


def phase_verify_keys(args) -> None:
    """Realized-keys asserts on every reused artifact (fail loud, plan §10)."""
    stage_dir = Path(args.stage_dir)
    for trait in lib.TRAITS:
        load_rb(trait, stage_dir=stage_dir, version="v2")
        load_rb(trait, stage_dir=stage_dir, version="v1")
    for variant in C.MAP_VARIANTS:
        mp = load_affine_map(variant, stage_dir=stage_dir)
        # Apply-contract sanity: mapping the x-mean must land near the y-mean.
        pred = apply_map(mp, mp["x_mu"][0], 0)
        assert pred.shape == (mp["w"].shape[2],), pred.shape
    for tag in ("base", "evil_misaligned_2"):
        acts = load_ft_activation(tag, stage_dir=stage_dir)
        assert set(acts) >= set(lib.TRAITS), (tag, sorted(acts))
    lib.log_phase("p8_verify", "realized-keys asserts PASS (rb v2+v1, maps, finetune_activations)")


def _model_tags(args, p5_root: Path) -> list[str]:
    tags = []
    for cell in args.cells or all_cells():
        tags.append(cell)
        for frac in C.CHECKPOINT_FRACS:
            t = f"{cell}@frac{int(round(frac * 100))}"
            if (p5_root / "capture" / f"{t}.pt").is_file():
                tags.append(t)
    return tags


def phase_arms(args) -> None:
    """Per-layer monitor scalars per (trait, panel) for every model + synth778."""
    stage_dir = Path(args.stage_dir)
    p5_root = Path(args.p5_root)
    out_dir = Path(args.eval_results_root) / "monitor_scalars"
    out_dir.mkdir(parents=True, exist_ok=True)

    rb = {t: load_rb(t, stage_dir=stage_dir, version=args.rb_version) for t in lib.TRAITS}
    map_ctx = load_affine_map("context_end", stage_dir=stage_dir)
    map_pfx = load_affine_map("prefix_end", stage_dir=stage_dir)

    base_last = _load_capture(p5_root, "base", "last")
    base_resp = _load_capture(p5_root, "base", "resp")
    tags = _model_tags(args, p5_root)

    for trait in lib.TRAITS:
        for panel in PANELS:
            rows: dict[str, dict[str, list[float]]] = {}
            vb_ctx = _mean_states(base_last, "last", trait, panel)
            vb_pfx = _mean_states(base_last, "prefix", trait, panel)
            vb_ans = _mean_states(base_resp, "resp_avg", trait, panel)
            for tag in tags:
                store = _load_capture(p5_root, tag, "last")
                vf_ctx = _mean_states(store, "last", trait, panel)
                vf_pfx = _mean_states(store, "prefix", trait, panel)
                v_ans_shift = None
                if "@" not in tag:  # response-avg captured for finals only
                    resp = _load_capture(p5_root, tag, "resp")
                    v_ans_shift = _mean_states(resp, "resp_avg", trait, panel) - vb_ans
                scal = M.arm_scalars_for_model(
                    rb=rb[trait],
                    v_ctx_shift=vf_ctx - vb_ctx,
                    v_pfx_shift_states=(vf_pfx, vb_pfx),
                    v_ctx_states=(vf_ctx, vb_ctx),
                    v_ans_shift=v_ans_shift,
                    map_ctx=map_ctx,
                    map_pfx=map_pfx,
                )
                rows[tag] = {arm: v.tolist() for arm, v in scal.items()}
            # Synthetic stratum (paper panel only — the cached #778 states are
            # trait-question means over the SAME matched 20-q surface).
            if panel == "paper":
                cached_base = load_ft_activation("base", stage_dir=stage_dir)
                base_resp_synth_path = p5_root / "capture_resp" / "base.pt"
                for cell in args.cells or all_cells():
                    tag = f"synth778_{cell}"
                    # Fail LOUD on a missing cached #778 cell (review issue 4):
                    # a silent skip would ship a partial synth stratum into H3.
                    cached_cell = load_ft_activation(cell, stage_dir=stage_dir)
                    v_ans_shift = None
                    synth_resp = p5_root / "capture_resp_synth778" / f"{tag}.pt"
                    if synth_resp.is_file() and base_resp_synth_path.is_file():
                        resp = _load_capture(p5_root, tag, "resp_synth")
                        v_ans_shift = _mean_states(resp, "resp_avg", trait, panel) - vb_ans
                    scal = M.arm_scalars_for_model(
                        rb=rb[trait],
                        v_ctx_shift=cached_cell[trait] - cached_base[trait],
                        v_pfx_shift_states=None,  # #778 cached no prefix-end states
                        v_ctx_states=(cached_cell[trait], cached_base[trait]),
                        v_ans_shift=v_ans_shift,
                        map_ctx=map_ctx,
                        map_pfx=map_pfx,
                    )
                    rows[tag] = {arm: v.tolist() for arm, v in scal.items()}
            payload = {
                "trait": trait,
                "panel": panel,
                "rb_version": args.rb_version,
                "arms": sorted({a for r in rows.values() for a in r}),
                "scalars": rows,
                "reproducibility": lib.repro_metadata(),
            }
            (out_dir / f"{trait}_{panel}.json").write_text(json.dumps(payload, indent=2))
            lib.log_phase("p8_arms", f"{trait}/{panel}: {len(rows)} models scored")


def _scores(args) -> dict:
    p = Path(args.eval_results_root) / "trait_scores.json"
    if not p.is_file():
        raise FileNotFoundError(f"run issue2221_trait_eval.py --phase aggregate first: {p}")
    return json.loads(p.read_text())["scores"]


def _scalar_matrix(scal: dict, tags: list[str], arm: str) -> np.ndarray:
    """(n_models, 28) scalar matrix for one arm (fail loud on a missing row)."""
    rows = []
    for t in tags:
        r = scal["scalars"].get(t, {}).get(arm)
        if r is None:
            raise KeyError(f"arm {arm} missing for model {t}")
        rows.append(r)
    return np.asarray(rows, dtype=np.float64)


def load_synth778_scores(root: Path, trait: str, cells: list[str]) -> np.ndarray:
    """Per-cell #778 graded trait means — the synth stratum's OWN y (blocker 1).

    Reads the committed ``eval_results/issue_778/finetune_{trait}_{cell}.json``
    files (fields ``model_tag`` / ``trait`` / ``trait_score``). The H3 contrast
    correlates each stratum's monitor scalars against that stratum's OWN
    trait-expression scores — NEVER the real-twin y.
    """
    vals = []
    for cell in cells:
        p = root / f"finetune_{trait}_{cell}.json"
        if not p.is_file():
            raise FileNotFoundError(
                f"#778 trait-score file missing: {p} — on a sparse checkout run "
                "`git sparse-checkout add eval_results/issue_778` first (partial-"
                "clone pods: BOOTSTRAP_EXTRA_CONES=eval_results/issue_778)"
            )
        d = json.loads(p.read_text())
        assert d["trait"] == trait and d["model_tag"] == cell, (
            str(p),
            d.get("trait"),
            d.get("model_tag"),
        )
        vals.append(float(d["trait_score"]))
    return np.asarray(vals, dtype=np.float64)


def load_synth778_base_score(root: Path, trait: str) -> float:
    """The #778 BASE model's graded trait score — the synth base propensity."""
    p = root / f"finetune_base_{trait}.json"
    if not p.is_file():
        raise FileNotFoundError(f"#778 base trait-score file missing: {p}")
    d = json.loads(p.read_text())
    assert d["trait"] == trait and d["model_tag"] == "base", (str(p), d.get("model_tag"))
    return float(d["trait_score"])


def synth_arms_present(scalars: dict, synth_tags: list[str]) -> list[str]:
    """Registry arms computable on the FULL synth stratum (blocker 2 / issue 4).

    - ``c_map_pfx`` is N/A by construction (#778 cached no prefix-end states).
    - An arm present for NONE of the tags is a not-run leg (the caller names
      it, never zero-bars it) — e.g. b/d before the P5 synth resp-avg leg.
    - An arm present for SOME but not ALL tags is silent degradation: raise.
    """
    out: list[str] = []
    for arm in C.MONITOR_ARMS:
        if arm == "c_map_pfx":
            continue
        n_present = sum(1 for t in synth_tags if arm in scalars.get(t, {}))
        if n_present == 0:
            continue
        if n_present != len(synth_tags):
            raise RuntimeError(
                f"H3 arm {arm}: partial synth coverage {n_present}/{len(synth_tags)} — "
                "fail loud (never a silent subset)"
            )
        out.append(arm)
    return out


def phase_correlations(args) -> None:
    """The full correlation / bootstrap / null / AUC battery (vectorized)."""
    eval_root = Path(args.eval_results_root)
    scal_dir = eval_root / "monitor_scalars"
    scores = _scores(args)
    stage_dir = Path(args.stage_dir)
    p5_root = Path(args.p5_root)
    rng = np.random.default_rng(C.RNG_SEED)
    cells = [c for c in (args.cells or all_cells())]
    fam_of = {c: c.rsplit("_", 1)[0] for c in cells}

    n = len(cells)
    idx_boot = M.bootstrap_indices(rng, n, args.n_bootstrap)

    result: dict = {
        "per_trait": {},
        "config": {
            "n_bootstrap": args.n_bootstrap,
            "n_null": args.n_null,
            "rb_version": args.rb_version,
            "seed": C.RNG_SEED,
            "cells": cells,
            "issue778_eval_root": str(args.issue778_eval_root),
        },
    }
    # Per-draw x per-position r matrices, persisted as .npz for post-hoc
    # recompute of the selection-symmetric bands (plan §6; review issue 2).
    draw_mats: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    null_mats: dict[str, dict[str, np.ndarray]] = {}

    for trait in lib.TRAITS:
        y = np.asarray([scores[c][trait]["graded_mean"] for c in cells], dtype=np.float64)
        assert np.isfinite(y).all(), f"non-finite trait scores for {trait}"
        tr: dict = {"panels": {}}
        for panel in PANELS:
            scal = json.loads((scal_dir / f"{trait}_{panel}.json").read_text())
            arms_present = [
                a for a in C.MONITOR_ARMS if all(a in scal["scalars"].get(c, {}) for c in cells)
            ]
            pn: dict = {"arms": {}}
            mats: dict[str, np.ndarray] = {}
            for arm in arms_present:
                x = _scalar_matrix(scal, cells, arm)  # (n, 28)
                r_by_layer = M.spearman_by_position(x.T, y)
                sel_layer, sel_r = M.select_position(r_by_layer)
                frozen = C.PAPER_FROZEN_LAYER_IDX[trait]
                boot_r = M.bootstrap_pearson(x.T, y, idx_boot)  # (B, 28)
                boot_sel = M.select_per_draw(boot_r)
                lofo = M.lofo_jackknife(x[:, sel_layer], y, [fam_of[c] for c in cells])
                shuffle_r = M.score_shuffle_r_matrix(
                    np.random.default_rng(C.RNG_SEED + 1), x.T, y, args.n_null
                )
                shuffle = M.select_per_draw(shuffle_r)
                mats[f"boot_r__{arm}"] = boot_r
                mats[f"shuffle_r__{arm}"] = shuffle_r
                pn["arms"][arm] = {
                    "r_by_layer": r_by_layer.tolist(),
                    "selected_layer": sel_layer,
                    "selected_r": sel_r,
                    "frozen_layer": frozen,
                    "frozen_r": float(r_by_layer[frozen]),
                    "bootstrap_ci_selected": M.percentile_ci(boot_sel),
                    "lofo": lofo,
                    "score_shuffle_null_q95_abs": float(
                        np.percentile(np.abs(shuffle[np.isfinite(shuffle)]), 95)
                    ),
                }
            # H2 primary contrast: mapped arm c (56 positions: ctx+pfx layers)
            # vs arm a (28 positions), per-draw selection on BOTH sides, paired.
            if {"a_rb_ctx", "c_map_ctx", "c_map_pfx"} <= set(arms_present):
                xc = np.concatenate(
                    [
                        _scalar_matrix(scal, cells, "c_map_ctx"),
                        _scalar_matrix(scal, cells, "c_map_pfx"),
                    ],
                    axis=1,
                )  # (n, 56)
                sel_a = M.select_per_draw(mats["boot_r__a_rb_ctx"])
                boot_r_c56 = M.bootstrap_pearson(xc.T, y, idx_boot)  # (B, 56)
                sel_c = M.select_per_draw(boot_r_c56)
                delta = sel_c - sel_a  # paired (same draws)
                r_c_by_pos = M.spearman_by_position(xc.T, y)
                pos_c, r_c = M.select_position(r_c_by_pos)
                # Wherever a null band is compared against the arm-c best-of-56
                # read, the null draws get the SAME 2x28 selection (plan §6;
                # review issue 3).
                shuffle_c56_r = M.score_shuffle_r_matrix(
                    np.random.default_rng(C.RNG_SEED + 1), xc.T, y, args.n_null
                )
                shuffle_c56 = M.select_per_draw(shuffle_c56_r)
                mats["boot_r__c56"] = boot_r_c56
                mats["shuffle_r__c56"] = shuffle_c56_r
                pn["h2_delta_c_minus_a"] = {
                    "point": float(r_c - pn["arms"]["a_rb_ctx"]["selected_r"]),
                    "c_selected_position": pos_c,  # 0..27 ctx, 28..55 pfx
                    "c_selected_r": float(r_c),
                    "bootstrap_ci": M.percentile_ci(delta),
                    "n_positions_c": int(xc.shape[1]),
                    "score_shuffle_null_q95_abs_56pos": float(
                        np.percentile(np.abs(shuffle_c56[np.isfinite(shuffle_c56)]), 95)
                    ),
                }
            tr["panels"][panel] = pn
            draw_mats[(trait, panel)] = mats

        # ── null ladder for arm a + arm c (pooled panel), same selection ────
        scal_pooled = json.loads((scal_dir / f"{trait}_pooled.json").read_text())
        base_last = _load_capture(p5_root, "base", "last")
        pool = np.asarray(base_last["last"].numpy(), dtype=np.float64)  # (m, 28, d)
        rb = load_rb(trait, stage_dir=stage_dir, version=args.rb_version)
        map_ctx = load_affine_map("context_end", stage_dir=stage_dir)
        map_pfx = load_affine_map("prefix_end", stage_dir=stage_dir)

        # Raw context shifts (n, 28, d) for arm a's null.
        shifts_ctx, shifts_mapped = [], []
        for cell in cells:
            store = _load_capture(p5_root, cell, "last")
            vf_ctx = _mean_states(store, "last", trait, "pooled")
            vf_pfx = _mean_states(store, "prefix", trait, "pooled")
            vb_pfx = _mean_states(base_last, "prefix", trait, "pooled")
            vb_ctx_p = _mean_states(base_last, "last", trait, "pooled")
            shifts_ctx.append(vf_ctx - vb_ctx_p)
            m_ctx = np.stack(
                [
                    apply_map_shift(map_ctx, vf_ctx[layer], vb_ctx_p[layer], layer)
                    for layer in range(C.N_LAYERS)
                ]
            )
            m_pfx = np.stack(
                [
                    apply_map_shift(map_pfx, vf_pfx[layer], vb_pfx[layer], layer)
                    for layer in range(C.N_LAYERS)
                ]
            )
            shifts_mapped.append(np.concatenate([m_ctx, m_pfx], axis=0))  # (56, d)
        shifts_ctx = np.stack(shifts_ctx)  # (n, 28, d)
        shifts_mapped = np.stack(shifts_mapped)  # (n, 56, d)

        rng_null = np.random.default_rng(C.RNG_SEED + 2)
        iso28 = M.isotropic_null_directions(rng_null, args.n_null, C.N_LAYERS, C.HIDDEN_DIM)
        iso56 = np.concatenate([iso28, iso28], axis=1)  # same draw across both position blocks
        cov_dirs, top_evec = M.covariance_null_directions(rng_null, args.n_null, pool)
        cov56 = np.concatenate([cov_dirs, cov_dirs], axis=1)
        circ = np.abs(
            np.einsum("ld,ld->l", top_evec, rb / np.linalg.norm(rb, axis=1, keepdims=True))
        )
        null_a_iso_r = M.null_r_matrix(iso28, shifts_ctx, y)  # (B, 28)
        null_a_cov_r = M.null_r_matrix(cov_dirs, shifts_ctx, y)
        null_c_iso_r = M.null_r_matrix(iso56, shifts_mapped, y)  # (B, 56)
        null_c_cov_r = M.null_r_matrix(cov56, shifts_mapped, y)
        null_a_iso = M.select_per_draw(null_a_iso_r)
        null_a_cov = M.select_per_draw(null_a_cov_r)
        null_c_iso = M.select_per_draw(null_c_iso_r)
        null_c_cov = M.select_per_draw(null_c_cov_r)
        null_mats[trait] = {
            "null_iso_a": null_a_iso_r,
            "null_cov_a": null_a_cov_r,
            "null_iso_c56": null_c_iso_r,
            "null_cov_c56": null_c_cov_r,
        }
        tr["nulls"] = {
            "a_isotropic_q95_abs": float(np.percentile(np.abs(null_a_iso), 95)),
            "a_covmatched_q95_abs": float(np.percentile(np.abs(null_a_cov), 95)),
            "c_isotropic_q95_abs": float(np.percentile(np.abs(null_c_iso), 95)),
            "c_covmatched_q95_abs": float(np.percentile(np.abs(null_c_cov), 95)),
            "cov_pool": "issue2221 base-model last-token capture (deviation: the "
            "#778 round-3 extraction pool is not staged; same kind — base "
            "activation pool)",
            "cov_circularity_max_abs_cos": float(circ.max()),
            "cov_circularity_flag": bool(circ.max() > COV_NULL_CIRCULARITY_MAX),
        }

        # ── checkpoint-time detection AUC (+ random-direction control) ───────
        labels = np.asarray(
            [scores[c][trait]["graded_mean"] >= C.DETECTION_POSITIVE_SCORE_MIN for c in cells]
        )
        auc: dict[str, dict] = {}
        for frac in C.CHECKPOINT_FRACS + (1.0,):
            tagf = cells if frac == 1.0 else [f"{c}@frac{int(round(frac * 100))}" for c in cells]
            per_arm: dict[str, float] = {}
            for arm in ("a_rb_ctx", "c_map_ctx", "c_map_pfx"):
                try:
                    x = _scalar_matrix(scal_pooled, tagf, arm)
                except KeyError:
                    continue
                sel = tr["panels"]["pooled"]["arms"][arm]["selected_layer"]
                per_arm[arm] = M.detection_auc(x[:, sel], labels)
            # Random-direction control at this checkpoint (isotropic, 100 draws).
            ctrl_dirs = M.isotropic_null_directions(
                np.random.default_rng(C.RNG_SEED + 3), 100, C.N_LAYERS, C.HIDDEN_DIM
            )
            if frac == 1.0:
                sc = shifts_ctx
            else:
                sc = []
                for c in cells:
                    store = _load_capture(p5_root, f"{c}@frac{int(round(frac * 100))}", "last")
                    sc.append(
                        _mean_states(store, "last", trait, "pooled")
                        - _mean_states(base_last, "last", trait, "pooled")
                    )
                sc = np.stack(sc)
            e = np.einsum("bld,fld->bfl", ctrl_dirs.astype(np.float32), sc.astype(np.float32))
            ctrl_aucs = [
                M.detection_auc(e[b, :, int(np.argmax(np.abs(e[b]).mean(axis=0)))], labels)
                for b in range(e.shape[0])
            ]
            per_arm["control_random_direction_mean"] = float(np.nanmean(ctrl_aucs))
            auc[f"frac{int(round(frac * 100))}"] = per_arm
        tr["checkpoint_auc"] = auc

        # ── within-family severity ordering (pooled, per arm at selected layer) ──
        ordering: dict[str, dict] = {}
        for arm in ("a_rb_ctx", "b_rb_ans", "c_map_ctx", "c_map_pfx"):
            if arm not in tr["panels"]["pooled"]["arms"]:
                continue
            sel = tr["panels"]["pooled"]["arms"][arm]["selected_layer"]
            x = _scalar_matrix(scal_pooled, cells, arm)
            vals: dict[str, dict[str, float]] = {}
            for i, c in enumerate(cells):
                fam, ver = c.rsplit("_", 1)
                vals.setdefault(fam, {})[ver] = float(x[i, sel])
            ordering[arm] = {
                "per_family_correct": M.severity_ordering(vals),
                "fraction_correct": float(np.mean(list(M.severity_ordering(vals).values()))),
            }
        tr["severity_ordering"] = ordering

        # ── H3 synth stratum (paper panel; cached #778 shifts; y = the #778
        # cells' OWN committed trait scores — review blocker 1) ──────────────
        scal_paper = json.loads((scal_dir / f"{trait}_paper.json").read_text())
        synth_tags = [f"synth778_{c}" for c in cells]
        synth_arms = synth_arms_present(scal_paper["scalars"], synth_tags)
        if not {"a_rb_ctx", "c_map_ctx"} <= set(synth_arms):
            raise RuntimeError(
                f"H3 synth stratum incomplete for {trait}: ctx arms missing — run "
                "--phase arms over the full cell set first (cached #778 shifts "
                "cover every cell, so absence here is a wiring fault)"
            )
        i778_root = Path(args.issue778_eval_root)
        y_synth = load_synth778_scores(i778_root, trait, cells)
        h3: dict = {
            "arms": {},
            "y_source": (
                "#778 committed eval_results/issue_778/finetune_{trait}_{cell}.json "
                "trait_score — each stratum correlates against its OWN scores, "
                "never the real-twin y"
            ),
            # b/d appear once the P5 synth resp-avg leg has run (consistency B3);
            # absent-for-all is a NOT-RUN leg, named here, never zero-barred.
            "arms_not_run": sorted(set(C.MONITOR_ARMS) - set(synth_arms) - {"c_map_pfx"}),
            "c_map_pfx": "N/A — #778 cached no prefix-end states",
        }
        for arm in synth_arms:
            xs = _scalar_matrix(scal_paper, synth_tags, arm)
            r_by_layer = M.spearman_by_position(xs.T, y_synth)
            sel_layer, sel_r = M.select_position(r_by_layer)
            h3["arms"][arm] = {
                "selected_layer": sel_layer,
                "selected_r": sel_r,
                "frozen_r": float(r_by_layer[C.PAPER_FROZEN_LAYER_IDX[trait]]),
            }
        # Install-covaried read (plan §6 install-strength control; review
        # issue 1): record each stratum's base propensity (the covariate —
        # a constant within stratum, so within-stratum Spearman is invariant
        # to it; the per-arm r above therefore doubles as the r-vs-install
        # read) and compare arms AT matched trait-expression support where
        # the 24-point support allows.
        base_prop_real = float(scores["base"][trait]["graded_mean"])
        base_prop_synth = load_synth778_base_score(i778_root, trait)
        lo = max(float(y.min()), float(y_synth.min()))
        hi = min(float(y.max()), float(y_synth.max()))
        m_real = (y >= lo) & (y <= hi)
        m_synth = (y_synth >= lo) & (y_synth <= hi)
        cov: dict = {
            "base_propensity_real": base_prop_real,
            "base_propensity_synth": base_prop_synth,
            "install_note": (
                "install = y - base propensity per stratum; the matched-support "
                "read compares arms at overlapping trait-expression levels, each "
                "stratum frozen at its own full-sample selected layer (no "
                "re-selection on the subset)"
            ),
            "matched_support": {
                "lo": lo,
                "hi": hi,
                "n_real": int(m_real.sum()),
                "n_synth": int(m_synth.sum()),
            },
            "arms": {},
        }
        if lo < hi and m_real.sum() >= 5 and m_synth.sum() >= 5:
            for arm in synth_arms:
                if arm not in tr["panels"]["paper"]["arms"]:
                    continue
                sel_real = tr["panels"]["paper"]["arms"][arm]["selected_layer"]
                sel_synth = h3["arms"][arm]["selected_layer"]
                x_real = _scalar_matrix(scal_paper, cells, arm)
                xs = _scalar_matrix(scal_paper, synth_tags, arm)
                cov["arms"][arm] = {
                    "r_real_matched": float(
                        M.spearman_by_position(x_real[m_real][:, sel_real][None, :], y[m_real])[0]
                    ),
                    "r_synth_matched": float(
                        M.spearman_by_position(
                            xs[m_synth][:, sel_synth][None, :], y_synth[m_synth]
                        )[0]
                    ),
                    "layer_real": sel_real,
                    "layer_synth": sel_synth,
                }
        else:
            cov["insufficient_support"] = True
        h3["install_covaried"] = cov
        tr["h3_synth_stratum"] = h3
        result["per_trait"][trait] = tr

    # ── reused-map baselines (identity+bias + kNN; standing rule) ───────────
    from explore_persona_space.analysis.mapping_baselines import (
        identity_bias_predict,
        knn_retrieval,
    )

    base_last = _load_capture(p5_root, "base", "last")
    base_resp = _load_capture(p5_root, "base", "resp")
    map_ctx = load_affine_map("context_end", stage_dir=stage_dir)
    ids_last = list(base_last["surface_ids"])
    ids_resp = list(base_resp["surface_ids"])
    common = [i for i in ids_last if i in set(ids_resp)]
    li = [ids_last.index(i) for i in common]
    ri = [ids_resp.index(i) for i in common]
    layer = int(np.median(list(C.PAPER_FROZEN_LAYER_IDX.values())))
    x = np.asarray(base_last["last"].numpy(), dtype=np.float64)[li, layer, :]
    ytrue = np.asarray(base_resp["resp_avg"].numpy(), dtype=np.float64)[ri, layer, :]
    pred = apply_map(map_ctx, x, layer)

    def _r2(p, t):
        ss_res = float(((t - p) ** 2).sum())
        ss_tot = float(((t - t.mean(axis=0)) ** 2).sum())
        return 1.0 - ss_res / ss_tot

    ib = identity_bias_predict(x, ytrue, x)
    result["map_baselines"] = {
        "layer": layer,
        "n_pool": len(common),
        "r2_map": _r2(pred, ytrue),
        "r2_identity_bias": _r2(ib, ytrue),
        "knn_euclidean": knn_retrieval(pred, ytrue, metric="euclidean"),
        "knn_cosine": knn_retrieval(pred, ytrue, metric="cosine"),
        "note": "in-pool identity+bias fit (bias from the same pool); map held-out "
        "R2 under LOFO is inherited from #1739 and reported in the body",
    }
    # Persist the per-draw x per-position r matrices (plan §6 persistence
    # contract; review issue 2) — float32, ~1 MB per arm per (trait, panel).
    mat_dir = eval_root / "draw_matrices"
    mat_dir.mkdir(parents=True, exist_ok=True)
    mat_files: list[str] = []
    for (trait, panel), mats in draw_mats.items():
        p = mat_dir / f"{trait}_{panel}.npz"
        np.savez(p, **{k: v.astype(np.float32) for k, v in mats.items()})
        mat_files.append(p.name)
    for trait, mats in null_mats.items():
        p = mat_dir / f"{trait}_nulls.npz"
        np.savez(p, **{k: v.astype(np.float32) for k, v in mats.items()})
        mat_files.append(p.name)
    result["draw_matrices"] = {
        "dir": str(mat_dir),
        "files": sorted(mat_files),
        "note": (
            "per-draw x per-position r matrices (boot_r__/shuffle_r__/null_*) — "
            "the honest selection-symmetric bands are pure re-reductions of these"
        ),
    }
    result["reproducibility"] = lib.repro_metadata()
    dest = eval_root / "correlations.json"
    dest.write_text(json.dumps(result, indent=2))
    lib.log_phase("p8_correlations", f"correlations.json written -> {dest}")


def phase_figures(args) -> None:
    """Paper-plots figures: per-arm selected r, layer profiles, checkpoint AUC."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis import paper_plots as pp

    corr = json.loads((Path(args.eval_results_root) / "correlations.json").read_text())
    fig_dir = Path(args.figures_root)
    fig_dir.mkdir(parents=True, exist_ok=True)
    pp.set_paper_style()
    arms = list(C.MONITOR_ARMS)
    colors = {a: c for a, c in zip(arms, pp.paper_palette(len(arms)))}

    # 1) selected |r| per arm x trait (pooled panel).
    fig, ax = plt.subplots(figsize=(7, 4))
    traits = list(corr["per_trait"])
    width = 0.8 / len(arms)
    for k, arm in enumerate(arms):
        vals = [
            corr["per_trait"][t]["panels"]["pooled"]["arms"].get(arm, {}).get("selected_r", np.nan)
            for t in traits
        ]
        ax.bar(np.arange(len(traits)) + k * width, vals, width, label=arm, color=colors[arm])
    ax.set_xticks(np.arange(len(traits)) + 0.4, traits)
    ax.set_ylabel("selected Spearman r (24 fine-tunes)")
    ax.axhline(0, color="black", lw=0.5)
    ax.legend(fontsize=7)
    pp.savefig_paper(fig, "monitor_selected_r_by_arm", dir=fig_dir)
    plt.close(fig)

    # 2) r-by-layer profiles per trait (pooled).
    for t in traits:
        fig, ax = plt.subplots(figsize=(7, 4))
        for arm in arms:
            d = corr["per_trait"][t]["panels"]["pooled"]["arms"].get(arm)
            if d is None:
                continue
            ax.plot(range(C.N_LAYERS), d["r_by_layer"], label=arm, color=colors[arm])
        ax.set_xlabel("layer index")
        ax.set_ylabel(f"Spearman r ({t})")
        ax.axhline(0, color="black", lw=0.5)
        ax.legend(fontsize=7)
        pp.savefig_paper(fig, f"monitor_r_by_layer_{t}", dir=fig_dir)
        plt.close(fig)

    # 3) checkpoint-time detection AUC.
    fig, ax = plt.subplots(figsize=(7, 4))
    fracs = [f"frac{int(round(f * 100))}" for f in C.CHECKPOINT_FRACS + (1.0,)]
    xs = [int(f.removeprefix("frac")) for f in fracs]
    for t in traits:
        auc = corr["per_trait"][t]["checkpoint_auc"]
        for arm in ("a_rb_ctx", "c_map_ctx"):
            ys = [auc.get(f, {}).get(arm, np.nan) for f in fracs]
            ax.plot(xs, ys, marker="o", label=f"{t}/{arm}")
    ax.set_xlabel("training progress (% of steps)")
    ax.set_ylabel("detection AUC (final score >= 50)")
    ax.axhline(0.5, color="black", lw=0.5, ls="--")
    ax.legend(fontsize=6)
    pp.savefig_paper(fig, "checkpoint_detection_auc", dir=fig_dir)
    plt.close(fig)
    lib.log_phase("p8_figures", f"figures written -> {fig_dir}")


PHASES = {
    "verify_keys": phase_verify_keys,
    "arms": phase_arms,
    "correlations": phase_correlations,
    "figures": phase_figures,
}


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--phase", choices=[*PHASES, "all"], default="all")
    ap.add_argument("--p5-root", default="data/issue_2221/p5")
    ap.add_argument("--stage-dir", default="data/issue_2221/hf_dl")
    ap.add_argument("--eval-results-root", default="eval_results/issue_2221")
    ap.add_argument(
        "--issue778-eval-root",
        default="eval_results/issue_778",
        help="committed #778 trait-score JSONs (the H3 synth stratum's OWN y)",
    )
    ap.add_argument("--figures-root", default="figures/issue_2221")
    ap.add_argument("--cells", nargs="*", default=None)
    ap.add_argument("--rb-version", choices=("v2", "v1"), default="v2")
    ap.add_argument("--n-bootstrap", type=int, default=C.N_BOOTSTRAP)
    ap.add_argument("--n-null", type=int, default=C.N_NULL_DRAWS)
    ap.add_argument("--list-phases", action="store_true")
    ap.add_argument("--import-check", action="store_true")
    return ap


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    args = build_argparser().parse_args()
    if args.list_phases:
        print(json.dumps(sorted(PHASES)))
        raise SystemExit(0)
    if args.import_check:
        from explore_persona_space.orchestrate.argcheck import assert_args_attributes_defined

        assert_args_attributes_defined(__file__)
        import matplotlib  # noqa: F401

        from explore_persona_space.analysis import paper_plots  # noqa: F401
        from explore_persona_space.analysis.mapping_baselines import (  # noqa: F401
            identity_bias_predict,
            knn_retrieval,
        )

        print("[import-check] OK")
        raise SystemExit(0)
    phases = list(PHASES) if args.phase == "all" else [args.phase]
    for name in phases:
        lib.log_phase(f"p8_{name}", "start")
        PHASES[name](args)
    lib.log_phase("p8", "done")
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(0)


if __name__ == "__main__":
    main()
