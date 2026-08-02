#!/usr/bin/env python
"""Issue #1336 — Phase P: the v2 decision writer (plan v13 §3 registered lattice).

Reads the committed cells_v2 JSONs + the Phase-LAD prediction matrices and
emits ``eval_results/issue_1336/decision_v2/headline_contrast_v2.json`` with
the registered quantities, computed ONCE:

- **bars**: ex_v2 = S_qwen_v2 / 0.6731 (the G0'(c) anchor, cm.v2_bars) —
  input via --s-qwen-v2 or --bars-json (the G0' gate output).
- **headline layer** (pre-registered stage-symmetric rule): among the frozen
  set, maximize MEAN within-stage R^2 across the primary-ladder models on
  lmsys23k-chat. Computed on the RAW scale (the v2 primary candidate — the
  §3 scale rule is resolved AFTER the layer is fixed); the recal-scale rule
  table is recorded as an audit companion with an agreement flag.
- **health gate H**: |R^2_recal - R^2_raw| <= 0.05*ex_v2 at the headline
  layer on the six cells feeding the primary + secondary contrasts
  ({dpo, rlvr} x {gsm8k_train_full, math7500, if11k} chat). H pass => raw
  is the primary scale; H fail => the lattice reads ONCE on the
  recalibrated scale with the raw companion reported (never blended) —
  the ``scale_fallback`` flag carries the bar_r-fallback semantics.
- **C_v2 = Delta_RLVR - Delta_DPO** per registered surface (primary:
  gsm8k_train_full chat; Bonferroni-2 secondary family: math7500 + if11k
  chat), from the persisted within/t8 (+_recal) prediction matrices on the
  SHARED row set of the two base-anchored pairs, with SHARED bootstrap
  draws (seed 5000 + v2 surface index — the battery convention). U = C -
  band and L = C + band ride the same draws (the band is a constant).
- **verdict** (§3 DISJOINT + exhaustive): RLVR-teaching <=> U > 0 and U's
  CI excludes 0 on the positive side; RLVR-unlearning <=> L's CI wholly
  below 0; Elicitation-consistent <=> U's CI wholly below 0 AND L's CI
  excludes 0 on the positive side; Inconclusive otherwise. Computed on the
  PRIMARY scale; the companion scale's lattice is recorded, never blended.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
for _p in (str(_SCRIPT_DIR), str(_REPO_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()  # shared-VM thread caps (#847) must bind BEFORE torch/numpy import

import issue825_fit_cells as fc  # noqa: E402
import issue1336_ladder_alignment as la  # noqa: E402
import issue1336_metric_ladder as ml  # noqa: E402
import numpy as np  # noqa: E402

from explore_persona_space.experiments.issue_1336 import common as cm  # noqa: E402

# The six lattice-bearing within cells (plan §6 health gate H): the stages of
# the headline contrast x the RLVR-trained surfaces feeding the primary +
# Bonferroni-2 secondary C_v2 reads.
H_STAGES = ("dpo", "rlvr")
H_SURFACES = (("gsm8k_train_full", "chat"), ("math7500", "chat"), ("if11k", "chat"))
PRIMARY_SURFACE = ("gsm8k_train_full", "chat")
SECONDARY_SURFACES = (("math7500", "chat"), ("if11k", "chat"))
# Smoke substitutes (the smoke grid has only the lmsys23k surfaces).
SMOKE_H_SURFACES = (("lmsys23k", "chat"),)
SMOKE_PRIMARY_SURFACE = ("lmsys23k", "chat")
SMOKE_SECONDARY_SURFACES = ()


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-dir", type=Path, default=Path("eval_results/issue_1336"))
    ap.add_argument("--preds-dir", type=Path, default=None)
    ap.add_argument("--frozen-layers", default=None, help="comma ints (default: registry set)")
    ap.add_argument("--n-boot", type=int, default=None)
    ap.add_argument("--s-qwen-v2", type=float, default=None)
    ap.add_argument("--bars-json", type=Path, default=None)
    ap.add_argument("--headline-layer", type=int, default=None, help="override (smoke only)")
    ap.add_argument("--smoke", action="store_true")
    return ap.parse_args()


def _metadata() -> dict:
    return {
        "git_commit": fc._git_commit(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "script": "scripts/issue1336_decision_v2.py",
    }


def _primary_models(smoke: bool) -> tuple[str, ...]:
    return cm.SMOKE_MODELS if smoke else cm.PRIMARY_LADDER


def headline_layer_rule_v2(cells_dir: Path, frozen_layers: tuple[int, ...], smoke: bool) -> dict:
    """Pre-registered v2 headline rule on RAW within R^2 + recal audit table.

    Fail-loud on a missing lmsys23k-chat cell (the rule's domain) or a cell
    JSON without the recal block (a stale pre-v2 fit output).
    """
    models = _primary_models(smoke)
    raw_means: dict[int, float] = {}
    recal_means: dict[int, float] = {}
    for li in frozen_layers:
        raw_vals, recal_vals = [], []
        for m in models:
            path = cells_dir / f"cells_{cm.cell_id(m, 'chat', 'lmsys23k')}.json"
            assert path.exists(), f"headline rule requires {path} (run the v2 fit phase first)"
            cell = json.loads(path.read_text())
            assert "recal" in cell, f"{path} lacks the recal block — stale fit output"
            raw_vals.append(float(cell["r2_per_layer_obs"][li]))
            per_layer = cell["recal"]["per_layer"]
            assert str(li) in per_layer, f"frozen layer {li} missing from recal block of {path}"
            recal_vals.append(float(per_layer[str(li)]["heldout_recal_r2"]))
        raw_means[li] = float(np.mean(raw_vals))
        recal_means[li] = float(np.mean(recal_vals))
    best_raw = max(raw_means, key=raw_means.get)
    best_recal = max(recal_means, key=recal_means.get)
    print(f"[decision_v2] headline layer {best_raw} (raw means {raw_means})")
    return {
        "headline_layer": int(best_raw),
        "rule": "max mean within-stage RAW R^2, primary models, lmsys23k-chat, frozen set",
        "raw_means": {str(k): v for k, v in raw_means.items()},
        "recal_means": {str(k): v for k, v in recal_means.items()},
        "recal_rule_layer": int(best_recal),
        "scales_agree": bool(best_raw == best_recal),
    }


def health_gate(
    cells_dir: Path,
    headline_layer: int,
    gate: float,
    *,
    stages: tuple[str, ...] = H_STAGES,
    surfaces: tuple[tuple[str, str], ...] = H_SURFACES,
) -> dict:
    """Health gate H (plan §6): |R^2_recal - R^2_raw| <= gate at the headline
    layer on every lattice-bearing within cell. Fail-loud on missing cells."""
    per_cell: dict[str, dict] = {}
    all_pass = True
    for m in stages:
        for corpus, fmt in surfaces:
            cid = cm.cell_id(m, fmt, corpus)
            path = cells_dir / f"cells_{cid}.json"
            assert path.exists(), f"health gate H requires {path}"
            cell = json.loads(path.read_text())
            row = cell["recal"]["per_layer"][str(headline_layer)]
            raw, recal = float(row["raw_r2"]), float(row["heldout_recal_r2"])
            dev = abs(recal - raw)
            ok = bool(dev <= gate)
            all_pass = all_pass and ok
            per_cell[cid] = {
                "raw_r2": raw,
                "recal_r2": recal,
                "abs_dev": dev,
                "gate": float(gate),
                "pass": ok,
            }
    return {"per_cell": per_cell, "gate": float(gate), "pass": bool(all_pass)}


def lattice_verdict(u: dict, low: dict) -> str:
    """§3 DISJOINT + exhaustive verdict from the U/L point + CI blocks."""
    if u["point"] > 0.0 and u["ci_lo"] > 0.0:
        return "rlvr_teaching"
    if low["ci_hi"] < 0.0:
        return "rlvr_unlearning"
    if u["ci_hi"] < 0.0 and low["ci_lo"] > 0.0:
        return "elicitation_consistent"
    return "inconclusive"


def _ci_at(dist: np.ndarray, alpha: float) -> dict:
    d = np.asarray(dist, dtype=np.float64)
    return {
        "alpha": float(alpha),
        "ci_lo": float(np.nanquantile(d, alpha / 2.0)),
        "ci_hi": float(np.nanquantile(d, 1.0 - alpha / 2.0)),
        "se_boot": float(np.nanstd(d, ddof=1)),
        "n_draws": int(len(d)),
    }


def _load_ladpreds(preds_dir: Path, m0: str, m1: str, fmt: str, corpus: str) -> dict:
    path = preds_dir / f"ladpreds_{m0}__{m1}_{fmt}_{corpus}.npz"
    assert path.exists(), f"missing ladder preds {path} — run the Phase-LAD battery first"
    return dict(np.load(path, allow_pickle=False))


def _delta_draws(npz: dict, layer: int, rows: np.ndarray, w: np.ndarray, variant: str) -> dict:
    """Per-draw Delta = R^2(within) - R^2(t8) on the shared-row subset.

    ``variant``: "" = raw, "_recal" = the recalibrated arms (BOTH arms of the
    gap transformed under the same crossfit recal — §3 scale consistency).
    """
    for key in (f"within{variant}_l{layer}", f"t8{variant}_l{layer}", f"y_l{layer}"):
        assert key in npz, f"{key} missing from the ladder preds npz (stale Phase-LAD output)"
    within = npz[f"within{variant}_l{layer}"][rows].astype(np.float64)
    t8 = npz[f"t8{variant}_l{layer}"][rows].astype(np.float64)
    y = npz[f"y_l{layer}"][rows].astype(np.float64)
    boot = la.paired_bootstrap_batched(within, y, t8, y, w)
    point = fc._pooled_r2(within, y) - fc._pooled_r2(t8, y)
    return {"draws": boot["delta"], "point": float(point)}


def contrast_for_surface(
    preds_dir: Path,
    corpus: str,
    fmt: str,
    layer: int,
    n_boot: int,
    band: float,
    *,
    alpha: float = 0.05,
) -> dict:
    """C_v2 (+ U/L) on one surface from the two base-anchored pairs' preds.

    Shared rows across base->dpo and base->rlvr, SHARED draws (seed
    5000 + surface idx). Both scales computed; the caller marks primary.
    """
    npz_dpo = _load_ladpreds(preds_dir, "base", "dpo", fmt, corpus)
    npz_rlvr = _load_ladpreds(preds_dir, "base", "rlvr", fmt, corpus)
    ids_dpo = npz_dpo["conv_ids"]
    ids_rlvr = npz_rlvr["conv_ids"]
    shared = np.intersect1d(ids_dpo, ids_rlvr)
    assert len(shared) >= cm.N_FOLDS, f"empty shared row set on ({corpus}, {fmt})"
    pos_d = {c: i for i, c in enumerate(ids_dpo)}
    pos_r = {c: i for i, c in enumerate(ids_rlvr)}
    rows_d = np.asarray([pos_d[c] for c in shared], dtype=np.int64)
    rows_r = np.asarray([pos_r[c] for c in shared], dtype=np.int64)
    si = cm.v2_surface_index(corpus, fmt)
    idx = la.draw_index_matrix(len(shared), n_boot, seed=5000 + si)
    w = la.counts_from_indices(idx, len(shared))

    out: dict = {
        "n_shared_rows": int(len(shared)),
        "boot_seed": 5000 + si,
        "layer": int(layer),
        "band": float(band),
    }
    for scale, variant in (("raw", ""), ("recal", "_recal")):
        d_dpo = _delta_draws(npz_dpo, layer, rows_d, w, variant)
        d_rlvr = _delta_draws(npz_rlvr, layer, rows_r, w, variant)
        c_draws = d_rlvr["draws"] - d_dpo["draws"]
        c_point = d_rlvr["point"] - d_dpo["point"]
        block = {
            "delta_dpo": {"point": d_dpo["point"], **la._ci(d_dpo["draws"])},
            "delta_rlvr": {"point": d_rlvr["point"], **la._ci(d_rlvr["draws"])},
            "C_v2": {"point": float(c_point), **_ci_at(c_draws, alpha)},
            "U": {"point": float(c_point - band), **_ci_at(c_draws - band, alpha)},
            "L": {"point": float(c_point + band), **_ci_at(c_draws + band, alpha)},
        }
        block["verdict"] = lattice_verdict(block["U"], block["L"])
        out[scale] = block
    return out


def run_decision(args) -> dict:
    smoke = args.smoke
    cells_dir = args.out_dir / "cells_v2"
    preds_dir = args.preds_dir or Path(
        "data/issue_1336/" + ("metric_ladder_preds_smoke" if smoke else "metric_ladder_preds")
    )
    if args.frozen_layers:
        frozen = tuple(int(x) for x in args.frozen_layers.split(",") if x.strip())
    else:
        frozen = cm.SMOKE_FROZEN_LAYERS if smoke else cm.FROZEN_LAYERS
    n_boot = (
        args.n_boot if args.n_boot is not None else (cm.SMOKE_N_BOOT if smoke else cm.N_BOOTSTRAP)
    )
    bars = ml.resolve_bars(args.s_qwen_v2, args.bars_json)
    band = float(bars["elicit_band_v2"])

    if args.headline_layer is not None:
        headline_block = {"headline_layer": int(args.headline_layer), "rule": "cli-override"}
    else:
        headline_block = headline_layer_rule_v2(cells_dir, frozen, smoke)
    headline = int(headline_block["headline_layer"])

    h_surfaces = SMOKE_H_SURFACES if smoke else H_SURFACES
    gate = health_gate(cells_dir, headline, float(bars["health_gate_v2"]), surfaces=h_surfaces)
    primary_scale = "raw" if gate["pass"] else "recal"
    scale_fallback = None
    if not gate["pass"]:
        scale_fallback = (
            "health gate H FAILED — the lattice reads once on the recalibrated scale "
            "(the parent-validated estimator, plan v9 route 1 / §3 primary-scale rule); "
            "raw stays the reported companion, never blended"
        )

    primary_surface = SMOKE_PRIMARY_SURFACE if smoke else PRIMARY_SURFACE
    secondary = SMOKE_SECONDARY_SURFACES if smoke else SECONDARY_SURFACES
    per_surface: dict[str, dict] = {}
    corpus, fmt = primary_surface
    per_surface[f"{corpus}_{fmt}"] = contrast_for_surface(
        preds_dir, corpus, fmt, headline, n_boot, band, alpha=0.05
    )
    for corpus, fmt in secondary:
        # Bonferroni-2 secondary family (§3): per-test alpha = 0.05 / 2.
        per_surface[f"{corpus}_{fmt}"] = contrast_for_surface(
            preds_dir, corpus, fmt, headline, n_boot, band, alpha=0.025
        )

    headline_key = f"{primary_surface[0]}_{primary_surface[1]}"
    primary_block = per_surface[headline_key][primary_scale]
    companion_scale = "recal" if primary_scale == "raw" else "raw"
    payload = {
        "metadata": _metadata(),
        "smoke": bool(smoke),
        "bars": bars,
        "headline_rule": headline_block,
        "headline_layer": headline,
        "health_gate_H": gate,
        "primary_scale": primary_scale,
        "scale_fallback": scale_fallback,
        "headline_surface": headline_key,
        "secondary_family_bonferroni2": [f"{c}_{f}" for c, f in secondary],
        "per_surface": per_surface,
        "verdict_lattice": {
            "verdict": primary_block["verdict"],
            "scale": primary_scale,
            "C_v2": primary_block["C_v2"],
            "U": primary_block["U"],
            "L": primary_block["L"],
            "predicates": {
                "rlvr_teaching": "U.point > 0 and U.ci_lo > 0",
                "rlvr_unlearning": "L.ci_hi < 0",
                "elicitation_consistent": "U.ci_hi < 0 and L.ci_lo > 0",
                "inconclusive": "otherwise",
            },
            "companion_scale": {
                "scale": companion_scale,
                "verdict": per_surface[headline_key][companion_scale]["verdict"],
                "C_v2": per_surface[headline_key][companion_scale]["C_v2"],
            },
        },
    }
    path = args.out_dir / "decision_v2" / "headline_contrast_v2.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=float))
    print(
        f"[decision_v2] wrote {path} (verdict={primary_block['verdict']}, "
        f"scale={primary_scale}, layer={headline})"
    )
    return payload


def main() -> None:
    run_decision(parse_args())


if __name__ == "__main__":
    main()
