#!/usr/bin/env python3
"""Task #2330 P4: paired-bootstrap contrasts from the P3 matched-fit preds.

Port of ``scripts/issue1491_adjacent_contrasts_from_preds.py`` (same vectorized
``_r2_boot`` counts-matrix GEMM — NO per-draw python loop) over the #2330 cell
grid. Registered contrasts (plan §4 P4), each raw AND ceiling-normalized:

- PRIMARY: 9B@10k − 7B@10k  (q35_n10k − q25_n10k)
- 9B@5k − 7B@5k             (q35_n5k − q25_n5k)
- 10k − 5k within 7B        (q25_n10k − q25_n5k)
- 10k − 5k within 9B        (q35_n10k − q35_n5k)

at the PRIMARY layers (7B L19 / 9B L22), with a per-layer SECONDARY pass over
the depth-fraction-matched layer pairs (7B 14↔9B 16, 26↔30; within-model
contrasts pair a model's layer with itself).

Method: paired bootstrap over the 1,000 shared pinned test contexts — 1,000
draws, seed 42, ONE shared resample matrix across every cell; SST recomputed
against each draw's own mean; ceilings treated as FIXED per-(model, layer)
scalars read from the P3 cell JSONs (with the #2130 read-side n_pairs defense
against a partial two-draw pairing).

Validation before any contrast is emitted: per (cell, layer), the full-sample
R² recomputed from the npz must match the committed
``matched_fits_<cell>.json`` per-layer ridge test_r2 (tol 1e-6); test context
ids must be identical (row-aligned after the ci sort) across all four cells;
same-model cells must carry identical test targets.

A ``--pilot-draws 100`` timed pilot runs FIRST and re-projects the full-pass
wall before the 1,000-draw pass (plan §9 P4 pilot-gated basis).

Writes: ``eval_results/issue_2330/contrasts.json``.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

import numpy as np  # noqa: E402

logger = logging.getLogger("issue2330_contrasts")

REPO_ROOT = _SCRIPTS.parent

CELLS = ["q25_n5k", "q25_n10k", "q35_n5k", "q35_n10k"]
MODEL_OF = {
    "q25_n5k": "qwen25_7b",
    "q25_n10k": "qwen25_7b",
    "q35_n5k": "qwen35_9b",
    "q35_n10k": "qwen35_9b",
}
# Depth-fraction-matched layer pairs (7B layer, 9B layer); index 1 = primary.
LAYER_PAIRS = [(14, 16), (19, 22), (26, 30)]
PRIMARY_LAYER = {"qwen25_7b": 19, "qwen35_9b": 22}

# Registered contrast pairs (b − a), plan §4 P4.
REGISTERED_PAIRS = [
    ("q25_n10k", "q35_n10k", "primary: 9B@10k − 7B@10k"),
    ("q25_n5k", "q35_n5k", "9B@5k − 7B@5k"),
    ("q25_n5k", "q25_n10k", "10k − 5k within 7B"),
    ("q35_n5k", "q35_n10k", "10k − 5k within 9B"),
]

N_BOOT = 1000
SEED = 42
TOL_COMMITTED = 1e-6


def _r2_full(pred: np.ndarray, y: np.ndarray) -> float:
    """Whole-map variance-weighted R² (parity with issue1491_ladder_fits)."""
    sse = float(((y - pred) ** 2).sum())
    sst = float(((y - y.mean(axis=0, keepdims=True)) ** 2).sum())
    return 1.0 - sse / (sst + 1e-30)


def _r2_boot(pred: np.ndarray, y: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Vectorized bootstrap R² per draw, SST against the draw's own mean.

    idx: (B, n) integer resample matrix. Returns (B,). One counts-matrix GEMM
    per call — no per-draw python loop (verbatim port of
    issue1491_adjacent_contrasts_from_preds._r2_boot).
    """
    n = y.shape[0]
    sse_i = ((y - pred) ** 2).sum(axis=1)  # (n,)
    ynorm_i = (y**2).sum(axis=1)  # (n,)
    B = idx.shape[0]
    counts = np.zeros((B, n), dtype=np.float64)
    rows = np.repeat(np.arange(B), idx.shape[1])
    np.add.at(counts, (rows, idx.ravel()), 1.0)
    sse_b = counts @ sse_i  # (B,)
    sum_ynorm_b = counts @ ynorm_i  # (B,)
    sum_y_b = counts @ y.astype(np.float64)  # (B, h)
    n_b = float(idx.shape[1])
    sst_b = sum_ynorm_b - (sum_y_b**2).sum(axis=1) / n_b
    return 1.0 - sse_b / (sst_b + 1e-30)


def contrast(a: str, b: str, label: str, boots: dict, fulls: dict, n_boot: int) -> dict:
    """Paired-bootstrap contrast record for Δ = b − a (shared resample matrix)."""
    d_b = boots[b] - boots[a]
    d_full = fulls[b] - fulls[a]
    lo, hi = np.percentile(d_b, [2.5, 97.5])
    p_lo = (1 + int((d_b <= 0).sum())) / (n_boot + 1)
    p_hi = (1 + int((d_b >= 0).sum())) / (n_boot + 1)
    return {
        "pair": [a, b],
        "label": label,
        "delta_full": float(d_full),
        "boot_mean": float(d_b.mean()),
        "ci95": [float(lo), float(hi)],
        "p_two_sided": float(min(1.0, 2 * min(p_lo, p_hi))),
        "n_boot": int(n_boot),
        "seed": SEED,
    }


def _load_cells(fits_dir: Path, preds_dir: Path) -> dict[str, dict]:
    """Load every cell's fits JSON + preds npz; run the pre-contrast validations."""
    data: dict[str, dict] = {}
    ci_ref: np.ndarray | None = None
    for cell in CELLS:
        fits_path = fits_dir / f"matched_fits_{cell}.json"
        if not fits_path.is_file():
            raise RuntimeError(
                f"missing P3 output {fits_path} — run issue2330_matched_fits.py first"
            )
        fits = json.loads(fits_path.read_text(encoding="utf-8"))
        expected_te = int(fits["counts_expected"]["test_1000"])
        z = np.load(preds_dir / f"{cell}_test_preds_ridge.npz")
        layers = [int(x) for x in z["layers"]]
        ci = np.asarray(z["ci_te"])
        order = np.argsort(ci)
        ci = ci[order]
        if ci_ref is None:
            ci_ref = ci
        elif not np.array_equal(ci_ref, ci):
            raise RuntimeError(f"{cell}: test context ids differ from reference cell")
        if ci.shape[0] != expected_te:
            raise RuntimeError(
                f"{cell}: npz has {ci.shape[0]} test rows vs counts_expected.test_1000="
                f"{expected_te} — partial preds (#2130 shape); refusing to bootstrap"
            )
        per_layer: dict[int, dict] = {}
        for layer in layers:
            pred = np.asarray(z[f"pred_te_L{layer}"], dtype=np.float64)[order]
            y = np.asarray(z[f"target_te_L{layer}"], dtype=np.float64)[order]
            lrec = fits["per_layer"][str(layer)]
            committed = float(lrec["ridge"]["test_r2"])
            full = _r2_full(pred, y)
            delta = abs(full - committed)
            if delta >= TOL_COMMITTED:
                raise RuntimeError(
                    f"{cell} L{layer}: recomputed R² {full} vs committed {committed} "
                    f"(|Δ|={delta:.3g} ≥ {TOL_COMMITTED}) — preds/fits mismatch"
                )
            cd = lrec["ceiling_two_draw"]
            if not cd.get("available"):
                raise RuntimeError(f"{cell} L{layer}: two-draw ceiling unavailable in fits JSON")
            # #2130 read-side defense: a committed short-pair ceiling must never
            # be consumed silently. The expected pairing count is self-described
            # by the cell JSON (counts_expected.test_1000).
            if int(cd.get("n_pairs", -1)) != expected_te:
                raise RuntimeError(
                    f"{cell} L{layer}: ceiling_two_draw.n_pairs={cd.get('n_pairs')} != "
                    f"{expected_te} — short/partial ceiling pairing (#2130); refusing to "
                    "normalize by it"
                )
            per_layer[layer] = {
                "pred": pred,
                "y": y,
                "r2_full": full,
                "committed": committed,
                "abs_delta_vs_committed": delta,
                "ceiling": float(cd["ceiling_var_weighted_r"]),
                "wc_point_r2": float(lrec["wc_transfer"]["ridge_test_r2"])
                if lrec.get("wc_transfer", {}).get("available")
                else None,
            }
        data[cell] = {
            "layers": layers,
            "primary_layer": int(z["primary_layer"]),
            "per_layer": per_layer,
            "n_test": int(ci.shape[0]),
        }
    # Same-model cells share the generation store → identical test targets.
    for model in ("qwen25_7b", "qwen35_9b"):
        cells = [c for c in CELLS if MODEL_OF[c] == model]
        pl = data[cells[0]]["primary_layer"]
        if not np.array_equal(
            data[cells[0]]["per_layer"][pl]["y"], data[cells[1]]["per_layer"][pl]["y"]
        ):
            raise RuntimeError(f"{model}: test targets differ across its two cells — store drift")
    return data


def _boots_at(data: dict, cell: str, layer: int, idx: np.ndarray) -> np.ndarray:
    d = data[cell]["per_layer"][layer]
    return _r2_boot(d["pred"], d["y"], idx)


def _pair_layer(cell: str, pair: tuple[int, int]) -> int:
    return pair[0] if MODEL_OF[cell] == "qwen25_7b" else pair[1]


def run(args) -> int:
    data = _load_cells(args.fits_dir, args.preds_dir)
    n = data[CELLS[0]]["n_test"]
    rng = np.random.default_rng(SEED)
    idx_full = rng.integers(0, n, size=(N_BOOT, n))  # ONE shared resample matrix

    # ---- timed pilot (plan §9 P4 pilot-gated basis) -------------------------
    pilot_record = None
    if args.pilot_draws and args.pilot_draws > 0:
        t0 = time.time()
        idx_pilot = idx_full[: args.pilot_draws]
        for cell in CELLS:
            pl = data[cell]["primary_layer"]
            _boots_at(data, cell, pl, idx_pilot)
        pilot_wall = time.time() - t0
        projected = pilot_wall * (N_BOOT / args.pilot_draws)
        pilot_record = {
            "pilot_draws": int(args.pilot_draws),
            "pilot_wall_s": float(pilot_wall),
            "projected_full_wall_s_primary_pass": float(projected),
            "basis": "4 primary-layer _r2_boot calls at pilot draws, linearly re-projected",
        }
        print(
            f"[contrasts] pilot: {args.pilot_draws} draws in {pilot_wall:.2f}s → projected "
            f"{projected:.1f}s for the {N_BOOT}-draw primary pass",
            flush=True,
        )

    # ---- full pass: boots per (cell, layer) ---------------------------------
    boots: dict[tuple[str, int], np.ndarray] = {}
    for cell in CELLS:
        for layer in data[cell]["layers"]:
            t0 = time.time()
            boots[(cell, layer)] = _boots_at(data, cell, layer, idx_full)
            print(
                f"[contrasts] boots {cell} L{layer}: {N_BOOT} draws in {time.time() - t0:.2f}s",
                flush=True,
            )

    def _layer_contrasts(pair_layers: tuple[int, int]) -> dict:
        raw_b = {c: boots[(c, _pair_layer(c, pair_layers))] for c in CELLS}
        raw_f = {c: data[c]["per_layer"][_pair_layer(c, pair_layers)]["r2_full"] for c in CELLS}
        ceil = {c: data[c]["per_layer"][_pair_layer(c, pair_layers)]["ceiling"] for c in CELLS}
        norm_b = {c: raw_b[c] / ceil[c] for c in CELLS}
        norm_f = {c: raw_f[c] / ceil[c] for c in CELLS}
        return {
            "layer_pair": {"qwen25_7b": pair_layers[0], "qwen35_9b": pair_layers[1]},
            "raw": [contrast(a, b, lb, raw_b, raw_f, N_BOOT) for a, b, lb in REGISTERED_PAIRS],
            "normalized": [
                contrast(a, b, lb, norm_b, norm_f, N_BOOT) for a, b, lb in REGISTERED_PAIRS
            ],
        }

    primary_pair = (PRIMARY_LAYER["qwen25_7b"], PRIMARY_LAYER["qwen35_9b"])
    primary = _layer_contrasts(primary_pair)
    secondary = {
        f"L{a}_L{b}": _layer_contrasts((a, b)) for (a, b) in LAYER_PAIRS if (a, b) != primary_pair
    }

    out = {
        "dv": (
            "ridge held-out variance-weighted test R² (matched 1,000-prompt LMSYS test split, "
            "4 cells: {7B, 9B} × {5k, 10k})"
        ),
        "method": (
            "paired bootstrap over the shared pinned test contexts; ONE resample matrix "
            f"(seed {SEED}, {N_BOOT} draws) shared across all cells; SST recomputed against "
            "each draw's own mean; ceilings treated as fixed per-(model, layer) scalars"
        ),
        "preds_provenance": (
            f"{args.preds_dir}/<cell>_test_preds_ridge.npz (P3 matched-fits outputs; "
            f"committed-R² parity asserted at tol {TOL_COMMITTED} per cell × layer)"
        ),
        "n_test": int(n),
        "n_boot": N_BOOT,
        "seed": SEED,
        "pilot": pilot_record,
        "validation_abs_delta_vs_committed": {
            cell: {
                str(layer): data[cell]["per_layer"][layer]["abs_delta_vs_committed"]
                for layer in data[cell]["layers"]
            }
            for cell in CELLS
        },
        "r2_full": {
            cell: {
                str(layer): data[cell]["per_layer"][layer]["r2_full"]
                for layer in data[cell]["layers"]
            }
            for cell in CELLS
        },
        "ceilings": {
            cell: {
                str(layer): data[cell]["per_layer"][layer]["ceiling"]
                for layer in data[cell]["layers"]
            }
            for cell in CELLS
        },
        "wc_point_r2": {
            cell: {
                str(layer): data[cell]["per_layer"][layer]["wc_point_r2"]
                for layer in data[cell]["layers"]
            }
            for cell in CELLS
        },
        "per_cell_boot_ci95_raw_primary": {
            cell: [
                float(q)
                for q in np.percentile(boots[(cell, data[cell]["primary_layer"])], [2.5, 97.5])
            ]
            for cell in CELLS
        },
        "primary_layers": primary,
        "per_layer_secondary": secondary,
    }
    # Convenience top-level aliases for the registered primary contrast.
    out["primary_contrast_raw"] = primary["raw"][0]
    out["primary_contrast_normalized"] = primary["normalized"][0]

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out_json.with_suffix(args.out_json.suffix + ".tmp")
    tmp.write_text(json.dumps(out, indent=1), encoding="utf-8")
    tmp.replace(args.out_json)
    print(f"[contrasts] wrote {args.out_json}", flush=True)
    print("primary raw:", json.dumps(out["primary_contrast_raw"]), flush=True)
    print("primary norm:", json.dumps(out["primary_contrast_normalized"]), flush=True)
    for c in primary["raw"][1:]:
        print("registered raw:", json.dumps(c), flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Task #2330 P4: paired-bootstrap contrasts")
    ap.add_argument(
        "--fits-dir",
        type=Path,
        default=REPO_ROOT / "eval_results" / "issue_2330",
        help="dir holding matched_fits_<cell>.json (P3 outputs)",
    )
    ap.add_argument(
        "--preds-dir",
        type=Path,
        default=REPO_ROOT / "data" / "issue_2330" / "preds",
        help="dir holding <cell>_test_preds_ridge.npz (P3 outputs)",
    )
    ap.add_argument(
        "--out-json",
        type=Path,
        default=REPO_ROOT / "eval_results" / "issue_2330" / "contrasts.json",
    )
    ap.add_argument(
        "--pilot-draws",
        type=int,
        default=100,
        help="timed pilot draw count run + re-projected BEFORE the full pass (0 disables)",
    )
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    )
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
