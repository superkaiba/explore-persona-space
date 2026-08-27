"""Issue #2587 unit 6 — P10 figures: the two heroes, matched-n table, and the
FULL plan-§6/§13 exploratory dump.

Consumes ONLY committed / harvested artifacts (no torch, no GPU, no network):
the unit-4 ``map_layer_sweep.json`` + ``matched7b_anchor.json``, the unit-5b
``minpair_delta_2587.json`` + ``perpair_2587.jsonl`` +
``crossmodel_contrasts.json``, the unit-5a ``manipulation_check_2587.json``
(+ the parent's committed ``manipulation_check.json``), the unit-1/P0b
``bank_manifest.json`` (token_gates), the harvested battery gen manifests
(``anchors_*.done.json``) + map-side cap-hit aggregates (``cap_hit_*.json``),
and the banked #2330 reference fits (``eval_results/issue_2330/…``).

Registered outputs (plan §6 "Figures." + §13 deliverables — one registry key
per named item; the §6 name is quoted per entry):

* ``hero_layer_sweep``            — hero 2: held-out test R² vs FRACTIONAL depth.
* ``crossmodel_axis_profile``     — hero 1: per-axis cross-model profile, WITH
  per-axis CI whiskers, null 95% bands, split-half ceilings, and the
  identity+bias (iddelta) baseline arms (plan §6 hero-1 layer contract).
* ``matched_n_table``             — §4.6 matched-n table (md + json).
* ``selected_lambda_per_layer``   — "selected-λ-per-layer curve".
* ``floors_per_layer``            — "floors table per layer" (rendered per-layer).
* ``wc_transfer_per_layer``       — "wc-transfer vs in-corpus bar".
* ``knn_per_layer``               — fit-side kNN acc@k per layer (§6 DV row).
* ``reliability_ceiling``         — fit-side two-draw ceilings (§6 DV row).
* ``crossmodel_delta_forest``     — per-axis carrier-paired deltas + CIs.
* ``delta_norm_scatter``          — "per-axis ‖Δ̂‖-vs-‖Δ‖ scatters" (per model).
* ``install_swap_violins``        — "install-vs-swap violins".
* ``axis_identity_heatmap``       — "axis-identity heatmaps" (per-value-pair
  carrier-mean identity cos, map arm, per model).
* ``crossfam_consistency_scatter``— "cross-family consistency scatters
  (observed + predicted, both models)".
* ``edit_dose_scatter``           — "edit-dose scatters per tokenizer".
* ``delta_retrieval_acc``         — "Δ-retrieval acc@k curves per arm"
  (battery-side; distinct from the fit-side ``knn_per_layer``).
* ``carrier_direction_heatmap``   — "per-carrier direction-cos transfer
  matrices" (carrier × axis mean map-arm cos, per model).
* ``text_space_rank_scatter``     — interpretation-r2 answer-text control:
  the per-axis rank scatters behind ``text_space_rank_reads.json``
  (cross-model text ordering + 9B separation-vs-text ordering).
* ``splithalf_vs_direction``      — "split-half-vs-direction scatters".
* ``pilot_axis_panels``           — "pilot-axis panels" (9B-only; the 7B side
  is rendered as pending per plan convention 12).
* ``lstar_sensitivity_twins``     — "L*-vs-{16,22,30} sensitivity twins of
  hero 1" (iddelta arm only — the frozen map is L*-fit, so the twin layers
  carry no map arm by construction; plan cross-unit constraint 5).
* ``pooling_twin_scatter``        — "span-mean pooling twin".
* ``matched_vs_parent_scatter``   — "7B-matched-vs-7B-parent-arm agreement
  scatter".
* ``think_leak_cap_hit_table``    — "think-leak + cap-hit tables" (md + json).
* ``manipulation_check_table``    — the §13 manipulation-check table (md + json).
* ``token_count_equality_table``  — "q25-vs-q35 token-count-equality table".

Display names live in ONE label map (``DISPLAY``) — internal arm/model slugs
never reach an axis, legend, or table header. Figures carry no on-canvas
caption blocks (axes + ticks + legend + panel titles only); provenance goes
to the ``savefig_paper`` sidecars. Color contract (one color = one meaning
across the set): primary = Qwen3.5-9B data, baseline = Qwen2.5-7B data,
control = WildChat transfer ONLY, accent = cosine-metric ONLY, neutral =
reference lines / floors / arm-type legend proxies.

CLI examples:
  uv run python scripts/issue2587_figures.py                      # all figures
  uv run python scripts/issue2587_figures.py --figs hero_layer_sweep,matched_n_table
  uv run python scripts/issue2587_figures.py --out-dir /tmp/issue-2587-smoke/figs \
      --sweep-json /tmp/fixtures/map_layer_sweep.json ...
  uv run python scripts/issue2587_figures.py --import-check
"""

from __future__ import annotations

import sys
from pathlib import Path

_SCRIPTS_DIR = str(Path(__file__).resolve().parent)
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

from explore_persona_space.orchestrate.env import load_dotenv

# Thread caps + creds BEFORE numpy/matplotlib import (tests/test_shared_vm_thread_caps.py).
load_dotenv()

import argparse  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import math  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from scipy.stats import rankdata  # noqa: E402

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    paper_palette,
    paper_palette_role,
    savefig_paper,
    set_paper_style,
)

logger = logging.getLogger("issue2587_figures")

ISSUE = 2587

# ── geometry + conventions ─────────────────────────────────────────────
# Mirrors scripts/issue2329_figures.py:79 (the #2329 dash-mark convention);
# pinned against that source by tests/test_issue2587_prefixes.py.
N_LAYERS_9B = 32
FULL_ATTENTION_LAYERS_9B = frozenset({3, 7, 11, 15, 19, 23, 27, 31})
FULL_ATTN_COLOR = "#9467bd"
N_LAYERS_7B = 28  # Qwen2.5-7B-Instruct decoder blocks (fraction = layer / 27)
L19 = 19

# The #2330 committed port-parity anchor (same-surface 7B n=25k L19 refit;
# plan §7 anchor gate |R² − 0.7250873| ≤ 0.01; also in issue2587_fits.py).
ANCHOR_7B_25K_R2 = 0.7250873220237553
ANCHOR_TOL = 0.01

MODEL_TAGS = ("qwen35_9b", "qwen25_7b")

# Plan-§7 disclosure thresholds rendered by the think-leak/cap-hit table.
CAP_HIT_REGEN_TRIGGER = 0.02
THINK_LEAK_ASSERT = 0.01

# ── display-name label map (ONE map; no internal slugs on any figure) ──
DISPLAY: dict[str, str] = {
    # models / curves
    "qwen35_9b": "Qwen3.5-9B (thinking off)",
    "qwen25_7b": "Qwen2.5-7B-Instruct",
    "fresh_9b_25k": "Qwen3.5-9B, n=25k (this run)",
    "ref_9b_10k": "Qwen3.5-9B, n=10k (prior fit)",
    "ref_7b_10k": "Qwen2.5-7B, n=10k (prior fit)",
    "anchor_7b_25k": "Qwen2.5-7B, n=25k (anchor)",
    "lstar": "selected layer L*",
    "floor_envelope": "strongest baseline floor",
    # map arms (cross-model panels)
    "arm_fresh9b": "Qwen3.5-9B fresh map",
    "arm_7b_matched25k": "Qwen2.5-7B matched-n map",
    "ref_7b_parent": "Qwen2.5-7B parent map (reference)",
    "arm_iddelta9b": "Qwen3.5-9B identity+bias baseline",
    "arm_iddelta7b": "Qwen2.5-7B identity+bias baseline",
    "iddelta_generic": "identity+bias baseline",
    # overlay layers (hero 1 + pilot panels)
    "null_band": "null 95% band",
    "split_half_ceiling": "split-half reliability ceiling",
    "pending_7b": "Qwen2.5-7B side pending (parent pilot reads)",
    # floors
    "identity_bias": "identity + learned bias",
    "identity_copy": "identity (copy input)",
    "scaled_identity": "scaled identity",
    "shuffled_pairing": "shuffled pairing",
    "train_mean": "train mean",
    # cross-model statistics (panel titles)
    "direction_cos": "direction cosine",
    "calibration_ratio_to_global": "calibration ratio (axis / global)",
    "obs_separation_snr": "observed separation / split-half noise",
    "crossfam_cos_observed": "cross-family consistency (observed)",
    "crossfam_cos_maparm": "cross-family consistency (predicted)",
    "axis_identity_cos": "axis identity cosine",
    # pair classes (violins)
    "swap": "value swap",
    "install": "install (vs bare)",
    # pooling conventions
    "pooling_tail": "tail-inclusive mean",
    "pooling_span": "answer span mean",
    # manipulation-check special axis verdicts
    "not_in_slice": "not in this run's slice",
    "no_manipulation_check_query_class": "no manipulation check (query class)",
}

_AXIS_OVERRIDES = {
    "answer_language": "Answer language",
    "answer_length": "Answer length",
}


def axis_label(axis: str) -> str:
    """Reader-facing name for a battery axis (no internal shorthand)."""
    return _AXIS_OVERRIDES.get(axis, axis.replace("_", " ").capitalize())


# Shared depth-axis label (one string for every per-layer figure; short enough
# to render un-clipped at the narrow 6.0-in exploratory figsize under the
# neurips constrained_layout style, which does not shrink over-wide labels).
XLABEL_DEPTH = "depth (fraction of stack; dashed = full-attention layers)"


def _err_offsets(vals: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """CI bounds -> NON-NEGATIVE matplotlib offsets (gotchas.md xerr/yerr rule)."""
    lo_off = np.maximum(0.0, np.asarray(vals) - np.asarray(lo))
    hi_off = np.maximum(0.0, np.asarray(hi) - np.asarray(vals))
    return np.vstack([lo_off, hi_off])


def _fracs(layers: list[int], n_layers: int) -> np.ndarray:
    return np.asarray(layers, dtype=np.float64) / max(n_layers - 1, 1)


def _mark_full_attention(ax, n_layers: int) -> None:
    """#2329 convention: dashed verticals at the full-attention layers; only
    meaningful on the 32-block Qwen3.5-9B geometry (skips otherwise)."""
    if n_layers != N_LAYERS_9B:
        return
    for layer in sorted(FULL_ATTENTION_LAYERS_9B):
        ax.axvline(layer / (n_layers - 1), color=FULL_ATTN_COLOR, ls="--", lw=0.6, zorder=0)


def _fnum(v) -> float:
    """JSON value -> float; the producer's _json_sanitize maps NaN/inf to
    None, so None reads back as NaN (never a silent 0)."""
    return float("nan") if v is None else float(v)


def _fpair(v) -> tuple[float, float]:
    """JSON 2-list (a CI) -> (lo, hi) floats with the None->NaN coercion."""
    if not v:
        return float("nan"), float("nan")
    return _fnum(v[0]), _fnum(v[1])


# ── input loading ──────────────────────────────────────────────────────


def _load_json(path: Path, what: str) -> dict:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"{what} missing: {p} (fail-loud — no placeholder figures)")
    return json.loads(p.read_text(encoding="utf-8"))


def _load_jsonl(path: Path, what: str) -> list[dict]:
    """JSONL rows via text-mode line iteration (never str.splitlines —
    gotchas.md U+2028 rule); fail-loud on missing or empty."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"{what} missing: {p} (fail-loud — no placeholder figures)")
    rows: list[dict] = []
    with p.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        raise RuntimeError(f"{what} parsed to ZERO rows: {p} (fail-loud)")
    return rows


def _load_leak_dir(path: Path, what: str) -> dict:
    """Recursive glob of the battery gen done-manifests + map-side cap-hit
    aggregates under one harvested directory; fail-loud when neither class
    matches (never a silent empty table)."""
    d = Path(path)
    if not d.is_dir():
        raise FileNotFoundError(f"{what} missing: {d} (fail-loud — no placeholder tables)")
    gen = {
        p: json.loads(p.read_text(encoding="utf-8")) for p in sorted(d.rglob("anchors_*.done.json"))
    }
    cap = {p: json.loads(p.read_text(encoding="utf-8")) for p in sorted(d.rglob("cap_hit_*.json"))}
    if not gen and not cap:
        raise RuntimeError(
            f"{what}: no anchors_*.done.json and no cap_hit_*.json under {d} — harvest the "
            "battery gen manifests / cap-hit aggregates first or pass --leak-caphit-dir "
            "(fail-loud; a missing upstream artifact never yields an empty table)"
        )
    return {"dir": d, "gen": gen, "cap": cap}


def sweep_layers(sweep: dict) -> list[int]:
    """Sorted layer indices of the sweep's per_layer map (fail-loud on empty)."""
    layers = sorted(int(k) for k in sweep["per_layer"])
    if not layers:
        raise RuntimeError("map_layer_sweep.json has an empty per_layer map")
    return layers


def _sweep_series(sweep: dict, getter) -> tuple[np.ndarray, np.ndarray]:
    layers = sweep_layers(sweep)
    n_layers = int(sweep["n_layers"])
    ys = np.asarray([getter(sweep["per_layer"][str(li)]) for li in layers], dtype=np.float64)
    return _fracs(layers, n_layers), ys


def _floor_envelope(row: dict) -> float:
    vals = [
        float(rec["test_r2"])
        for rec in row["floors"].values()
        if rec.get("test_r2") is not None and np.isfinite(float(rec["test_r2"]))
    ]
    if not vals:
        raise RuntimeError(f"layer {row.get('layer')}: no finite floor test_r2 values")
    return max(vals)


def _require_stat_axes(cm_stats: dict, what: str, *stats: str) -> None:
    """Empty-render guard (r1 Minor 1): an empty per-stat axes list must
    RAISE, never render a blank panel set that passes a PNG-size floor."""
    for stat in stats:
        axes = (cm_stats.get(stat) or {}).get("axes")
        if not axes:
            raise RuntimeError(
                f"{what}: stats[{stat!r}]['axes'] is EMPTY — refusing a blank render (fail-loud)"
            )


def _delta_sides(delta: dict) -> dict:
    """Validated access to minpair_delta_2587.json's per-side battery blocks
    (fail-loud on a doc that lacks populated sides — e.g. an h1-only stub)."""
    sides = delta.get("sides") or {}
    missing = [t for t in MODEL_TAGS if t not in sides]
    if missing:
        raise RuntimeError(
            f"minpair_delta_2587.json lacks side block(s) {missing} — the per-axis battery "
            "reads are required for the delta-consuming figures (fail-loud)"
        )
    for tag in MODEL_TAGS:
        if not sides[tag].get("axes"):
            raise RuntimeError(
                f"minpair_delta_2587.json side {tag!r} has an EMPTY axes map (fail-loud)"
            )
    return sides


def _perpair_by_model(rows: list[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for r in rows:
        out.setdefault(str(r["model_tag"]), []).append(r)
    missing = [t for t in MODEL_TAGS if t not in out]
    if missing:
        raise RuntimeError(f"perpair_2587.jsonl has no rows for model tag(s) {missing} (fail-loud)")
    return out


def _map_arm_of(rows: list[dict], what: str) -> str:
    """Resolve the single map arm from a perpair row's arm-keyed dicts (the
    other arm is the identity+bias iddelta baseline by the unit-5b contract)."""
    arms = sorted(rows[0]["norm_pred"])
    non_id = [a for a in arms if "iddelta" not in a]
    if len(non_id) != 1:
        raise RuntimeError(f"{what}: cannot resolve the map arm from arms={arms} (fail-loud)")
    return non_id[0]


def _fmt(v) -> str:
    """Markdown-cell formatter shared by every table writer."""
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return "n/a" if not math.isfinite(v) else f"{v:.4f}"
    return str(v)


# ── figures: heroes + matched-n table ──────────────────────────────────


def fig_hero_layer_sweep(inputs: dict, out_dir: Path) -> list[Path]:
    """Hero 2 (plan §6): held-out test R² vs fractional depth, both models."""
    sweep = inputs["sweep"]
    ref9 = inputs["ref9b_n10k"]
    ref7 = inputs["ref7b_n10k"]
    c9 = paper_palette_role("primary")
    c7 = paper_palette_role("baseline")
    neutral = paper_palette_role("neutral")

    fig, ax = plt.subplots(figsize=(7.0, 4.2), layout="constrained")
    n_layers = int(sweep["n_layers"])
    _mark_full_attention(ax, n_layers)

    x9, y9 = _sweep_series(sweep, lambda r: float(r["ridge"]["test_r2"]))
    ax.plot(x9, y9, color=c9, marker="o", ms=3.5, lw=1.6, label=DISPLAY["fresh_9b_25k"])

    _, yfl = _sweep_series(sweep, _floor_envelope)
    ax.plot(x9, yfl, color=neutral, ls="--", lw=1.1, label=DISPLAY["floor_envelope"])

    lstar = int(sweep["lstar"]["lstar"])
    ax.plot(
        [lstar / (n_layers - 1)],
        [float(sweep["per_layer"][str(lstar)]["ridge"]["test_r2"])],
        marker="*",
        ms=14,
        color=c9,
        mec="black",
        mew=0.6,
        ls="none",
        label=DISPLAY["lstar"],
    )

    l9r = [int(x) for x in ref9["layers"]]
    y9r = [float(ref9["per_layer"][str(li)]["ridge"]["test_r2"]) for li in l9r]
    ax.plot(
        _fracs(l9r, N_LAYERS_9B),
        y9r,
        color=c9,
        marker="s",
        mfc="none",
        ms=6,
        ls=":",
        lw=1.0,
        label=DISPLAY["ref_9b_10k"],
    )

    l7r = [int(x) for x in ref7["layers"]]
    y7r = [float(ref7["per_layer"][str(li)]["ridge"]["test_r2"]) for li in l7r]
    ax.plot(
        _fracs(l7r, N_LAYERS_7B),
        y7r,
        color=c7,
        marker="D",
        mfc="none",
        ms=6,
        ls=":",
        lw=1.0,
        label=DISPLAY["ref_7b_10k"],
    )
    ax.plot(
        [L19 / (N_LAYERS_7B - 1)],
        [ANCHOR_7B_25K_R2],
        marker="D",
        ms=8,
        color=c7,
        ls="none",
        label=DISPLAY["anchor_7b_25k"],
    )

    ax.set_xlabel(XLABEL_DEPTH)
    ax.set_ylabel("held-out test R²")
    ax.set_xlim(-0.02, 1.02)
    ax.legend(loc="lower center", fontsize=8, ncol=2)
    paths = savefig_paper(fig, "fig_hero_layer_sweep", dir=out_dir)
    plt.close(fig)
    return list(paths.values())


def _knn_cell(knn: dict, arm: str, metric: str, k: int) -> float:
    return float(knn[arm][metric]["acc_at_k"][str(k)])


def _table_side(row: dict, ceiling: dict | None) -> dict:
    """One matched-n table side from a per-layer-row-shaped dict (9B sweep row
    or the matched7b record's ``arm`` block — same schema by construction)."""
    meta = row["ridge"]["meta"] if "meta" in row["ridge"] else row["ridge_meta"]
    knn = row["knn"]
    side = {
        "n_train": int(row["n_train"]),
        "d": int(row["d"]),
        "val_r2_at_selected": float(meta["val_r2_at_selected"]),
        "test_r2": float(row["ridge"]["test_r2"]) if "ridge" in row else float(row["test_r2"]),
        "wc_test_1k_r2": float(
            row["ridge"]["wc_test_1k_r2"] if "ridge" in row else row["wc_test_1k_r2"]
        ),
        "selected_lambda": float(meta["selected_lambda"]),
        "floors": {
            name: (None if rec.get("test_r2") is None else float(rec["test_r2"]))
            for name, rec in row["floors"].items()
        },
        "knn_ridge_acc_at_1_euclid": _knn_cell(knn, "ridge", "euclidean", 1),
        "knn_ridge_acc_at_10_euclid": _knn_cell(knn, "ridge", "euclidean", 10),
        "knn_ridge_acc_at_1_cosine": _knn_cell(knn, "ridge", "cosine", 1),
        "knn_ridge_acc_at_10_cosine": _knn_cell(knn, "ridge", "cosine", 10),
        "knn_chance_at_1": float(knn["ridge"]["euclidean"]["chance_at_k"]["1"]),
    }
    if ceiling is not None and ceiling.get("available"):
        side["two_draw_ceiling_r"] = float(ceiling["ceiling_var_weighted_r"])
    else:
        side["two_draw_ceiling_r"] = None
    return side


def matched_n_table(inputs: dict, out_dir: Path) -> list[Path]:
    """Matched-n comparison table: 9B@L* vs 7B@L19 (md + json)."""
    sweep = inputs["sweep"]
    m7 = inputs["matched7b"]
    delta = inputs.get("delta")

    lstar = int(sweep["lstar"]["lstar"])
    row9 = sweep["per_layer"][str(lstar)]
    # matched7b record's arm block: reshape to the per-layer row shape.
    arm7 = m7["arm"]
    row7 = {
        "n_train": arm7["n_train"],
        "d": arm7["d"],
        "ridge": {
            "meta": arm7["ridge_meta"],
            "test_r2": arm7["test_r2"],
            "wc_test_1k_r2": arm7["wc_test_1k_r2"],
        },
        "floors": arm7["floors"],
        "knn": arm7["knn"],
    }
    side9 = _table_side(row9, sweep["reliability_ceiling"]["by_layer"].get(str(lstar)))
    side7 = _table_side(row7, m7.get("ceiling_7b_matched_L19"))
    anchor = m7["anchor"]

    doc: dict = {
        "issue": ISSUE,
        "layer_pair": {"qwen35_9b": lstar, "qwen25_7b": L19},
        "sides": {"qwen35_9b": side9, "qwen25_7b": side7},
        "anchor_gate": {
            "expected_r2": float(anchor["expected_r2"]),
            "realized_r2": float(anchor["realized_r2"]),
            "abs_deviation": float(anchor["abs_deviation"]),
            "tol": float(anchor["tol"]),
        },
    }
    if delta is not None:
        doc["h1_paired_shared_rows"] = {
            k: delta["h1"][k]
            for k in ("r2_9b_lstar", "r2_7b_l19", "delta_map", "delta_ci95", "verdict")
        }

    name9 = f"{DISPLAY['qwen35_9b']} @ layer {lstar} (L*)"
    name7 = f"{DISPLAY['qwen25_7b']} @ layer {L19}"

    lines = [
        f"# Matched-n map comparison — {name9} vs {name7}",
        "",
        "All fits at matched n (train≈25k / val 400 / test 1,000), fp64 primal ridge,",
        "λ val-selected; retrieval = P(true target in the k nearest neighbours of the",
        "prediction) over the held-out test pool.",
        "",
        f"| quantity | {name9} | {name7} |",
        "|---|---|---|",
    ]
    rows_spec = [
        ("train rows (n)", "n_train"),
        ("hidden dim (d)", "d"),
        ("validation R² (selected λ)", "val_r2_at_selected"),
        ("held-out test R²", "test_r2"),
        ("held-out transfer R² (WildChat)", "wc_test_1k_r2"),
        ("selected λ", "selected_lambda"),
        ("retrieval acc@1 (euclidean)", "knn_ridge_acc_at_1_euclid"),
        ("retrieval acc@10 (euclidean)", "knn_ridge_acc_at_10_euclid"),
        ("retrieval acc@1 (cosine)", "knn_ridge_acc_at_1_cosine"),
        ("retrieval acc@10 (cosine)", "knn_ridge_acc_at_10_cosine"),
        ("retrieval chance @1", "knn_chance_at_1"),
        ("two-draw reliability ceiling (r)", "two_draw_ceiling_r"),
    ]
    for label, key in rows_spec:
        lines.append(f"| {label} | {_fmt(side9[key])} | {_fmt(side7[key])} |")
    for fname in sorted(side9["floors"]):
        lines.append(
            f"| floor: {DISPLAY.get(fname, fname)} R² | "
            f"{_fmt(side9['floors'][fname])} | {_fmt(side7['floors'].get(fname))} |"
        )
    lines += [
        "",
        f"Anchor gate: realized R² {_fmt(doc['anchor_gate']['realized_r2'])} vs expected "
        f"{_fmt(doc['anchor_gate']['expected_r2'])} "
        f"(|Δ| = {doc['anchor_gate']['abs_deviation']:.2e}, tol {doc['anchor_gate']['tol']}).",
    ]
    if "h1_paired_shared_rows" in doc:
        h1 = doc["h1_paired_shared_rows"]
        lo, hi = h1["delta_ci95"]
        lines += [
            "",
            f"Paired shared-test-row comparison: R²(9B@L*) = {h1['r2_9b_lstar']:.4f}, "
            f"R²(7B@L19) = {h1['r2_7b_l19']:.4f}, Δ = {h1['delta_map']:.4f} "
            f"(95% CI [{lo:.4f}, {hi:.4f}]; verdict: {h1['verdict']}).",
        ]

    out_dir.mkdir(parents=True, exist_ok=True)
    md_path = out_dir / "table_matched_n.md"
    json_path = out_dir / "table_matched_n.json"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    json_path.write_text(json.dumps(doc, indent=2, sort_keys=True), encoding="utf-8")
    return [md_path, json_path]


# ── figures: fit-side per-layer exploratory dump ───────────────────────


def fig_selected_lambda_per_layer(inputs: dict, out_dir: Path) -> list[Path]:
    sweep = inputs["sweep"]
    fig, ax = plt.subplots(figsize=(6.0, 3.4), layout="constrained")
    _mark_full_attention(ax, int(sweep["n_layers"]))
    x, lam = _sweep_series(sweep, lambda r: float(r["ridge"]["meta"]["selected_lambda"]))
    ax.plot(x, lam, color=paper_palette_role("primary"), marker="o", ms=3, lw=1.4)
    ax.set_yscale("log")
    ax.set_xlabel(XLABEL_DEPTH)
    ax.set_ylabel("selected ridge λ (validation)")
    paths = savefig_paper(fig, "fig_selected_lambda_per_layer", dir=out_dir)
    plt.close(fig)
    return list(paths.values())


def fig_floors_per_layer(inputs: dict, out_dir: Path) -> list[Path]:
    sweep = inputs["sweep"]
    layers = sweep_layers(sweep)
    floor_names = sorted(sweep["per_layer"][str(layers[0])]["floors"])
    colors = paper_palette(len(floor_names) + 1)
    fig, ax = plt.subplots(figsize=(6.5, 4.0), layout="constrained")
    _mark_full_attention(ax, int(sweep["n_layers"]))
    x, ridge = _sweep_series(sweep, lambda r: float(r["ridge"]["test_r2"]))
    ax.plot(x, ridge, color=colors[0], lw=1.8, label="ridge map")
    for i, name in enumerate(floor_names):
        _, ys = _sweep_series(
            sweep,
            lambda r, n=name: (
                float(r["floors"][n]["test_r2"])
                if r["floors"][n].get("test_r2") is not None
                else np.nan
            ),
        )
        ax.plot(x, ys, color=colors[i + 1], lw=1.0, ls="--", label=DISPLAY.get(name, name))
    ax.set_xlabel(XLABEL_DEPTH)
    ax.set_ylabel("held-out test R²")
    ax.legend(fontsize=7, ncol=2)
    paths = savefig_paper(fig, "fig_floors_per_layer", dir=out_dir)
    plt.close(fig)
    return list(paths.values())


def fig_wc_transfer_per_layer(inputs: dict, out_dir: Path) -> list[Path]:
    sweep = inputs["sweep"]
    fig, ax = plt.subplots(figsize=(6.0, 3.4), layout="constrained")
    _mark_full_attention(ax, int(sweep["n_layers"]))
    x, y_in = _sweep_series(sweep, lambda r: float(r["ridge"]["test_r2"]))
    _, y_wc = _sweep_series(sweep, lambda r: float(r["ridge"]["wc_test_1k_r2"]))
    ax.plot(x, y_in, color=paper_palette_role("primary"), lw=1.6, label="in-corpus test R²")
    ax.plot(x, y_wc, color=paper_palette_role("control"), lw=1.6, label="WildChat transfer R²")
    ax.set_xlabel(XLABEL_DEPTH)
    ax.set_ylabel("held-out R²")
    ax.legend(fontsize=8)
    paths = savefig_paper(fig, "fig_wc_transfer_per_layer", dir=out_dir)
    plt.close(fig)
    return list(paths.values())


def fig_knn_per_layer(inputs: dict, out_dir: Path) -> list[Path]:
    sweep = inputs["sweep"]
    fig, ax = plt.subplots(figsize=(6.0, 3.6), layout="constrained")
    _mark_full_attention(ax, int(sweep["n_layers"]))
    # Color contract: this figure is 9B fresh-fit data (primary); the cosine
    # metric rides ACCENT set-wide ("control" stays WildChat-transfer-only).
    c1 = paper_palette_role("primary")
    c2 = paper_palette_role("accent")
    for k, ls in ((1, "-"), (10, "--")):
        x, ye = _sweep_series(sweep, lambda r, kk=k: _knn_cell(r["knn"], "ridge", "euclidean", kk))
        _, yc = _sweep_series(sweep, lambda r, kk=k: _knn_cell(r["knn"], "ridge", "cosine", kk))
        ax.plot(x, ye, color=c1, ls=ls, lw=1.4, label=f"euclidean acc@{k}")
        ax.plot(x, yc, color=c2, ls=ls, lw=1.4, label=f"cosine acc@{k}")
    layers = sweep_layers(sweep)
    chance1 = float(
        sweep["per_layer"][str(layers[0])]["knn"]["ridge"]["euclidean"]["chance_at_k"]["1"]
    )
    ax.axhline(chance1, color=paper_palette_role("neutral"), lw=0.8, ls=":", label="chance @1")
    ax.set_xlabel(XLABEL_DEPTH)
    ax.set_ylabel("retrieval accuracy (true target in top k)")
    ax.legend(fontsize=7, ncol=2)
    paths = savefig_paper(fig, "fig_knn_per_layer", dir=out_dir)
    plt.close(fig)
    return list(paths.values())


def fig_reliability_ceiling(inputs: dict, out_dir: Path) -> list[Path]:
    sweep = inputs["sweep"]
    rc = sweep["reliability_ceiling"]
    layers = [int(x) for x in rc["layers"]]
    vals = []
    for li in layers:
        blk = rc["by_layer"][str(li)]
        vals.append(float(blk["ceiling_var_weighted_r"]) if blk.get("available") else np.nan)
    fig, ax = plt.subplots(figsize=(4.6, 3.2), layout="constrained")
    ax.bar(
        [str(li) for li in layers],
        vals,
        color=paper_palette_role("primary"),
        width=0.55,
    )
    ax.set_xlabel("layer")
    ax.set_ylabel("two-draw reliability ceiling (r)")
    paths = savefig_paper(fig, "fig_reliability_ceiling", dir=out_dir)
    plt.close(fig)
    return list(paths.values())


# ── figures: cross-model profile (hero 1) + delta forest ──────────────

_PROFILE_STATS = ("direction_cos", "calibration_ratio_to_global", "obs_separation_snr")


def _side_layers(sides: dict, tag: str) -> tuple[dict, str, str]:
    """(axes map, map arm, iddelta arm) for one side, from the side's own meta."""
    meta = sides[tag]["meta"]
    return sides[tag]["axes"], str(meta["map_arm"]), str(meta["id_arm"])


def fig_crossmodel_axis_profile(inputs: dict, out_dir: Path) -> list[Path]:
    """Hero 1 (plan §6): per-axis cross-model profile — one row per axis with
    per-model sub-rows. Panel 1 (direction cosine) carries the full §6 layer
    contract: map-arm point + 95% CI whisker, identity+bias baseline (×),
    split-half reliability ceiling (|), null 95% band, and the 7B parent-map
    reference. Panel 2 (calibration ratio) carries map + CI + baseline; panel
    3 is the crossmodel doc's ceiling-adjusted observed-space separation.
    Map-arm points/CIs/nulls/ceilings read from minpair_delta_2587.json
    (side-own headline reads); the parent reference + separation panel read
    from crossmodel_contrasts.json (symmetric-fired shared subset)."""
    cm = inputs["crossmodel"]
    _require_stat_axes(cm["stats"], "crossmodel_contrasts.json", "direction_cos")
    sides = _delta_sides(inputs["delta"])
    axes9, map9, id9 = _side_layers(sides, "qwen35_9b")
    axes7, map7, id7 = _side_layers(sides, "qwen25_7b")
    c9 = paper_palette_role("primary")
    c7 = paper_palette_role("baseline")
    neutral = paper_palette_role("neutral")

    def _dir9(a: str) -> float:
        row = axes9.get(a)
        return _fnum(row["direction"][map9]["mean_cos_headline"]) if row else float("nan")

    all_axes = sorted(set(axes9) | set(axes7))
    axes_order = sorted(all_axes, key=lambda a: -_dir9(a) if math.isfinite(_dir9(a)) else math.inf)
    ypos = np.arange(len(axes_order))[::-1].astype(np.float64)
    ref_rows = {r["axis"]: r for r in cm["stats"]["direction_cos"]["axes"]}
    sep_rows = {r["axis"]: r for r in (cm["stats"].get("obs_separation_snr") or {}).get("axes", [])}

    # layout="constrained": the neurips style does NOT enable constrained
    # layout via rcParams, and multi-panel figures + outside fig-legends need
    # it (set at creation — never a post-colorbar engine switch).
    fig, panels = plt.subplots(
        1,
        len(_PROFILE_STATS),
        figsize=(3.3 * len(_PROFILE_STATS), 0.52 * len(axes_order) + 2.2),
        sharey=True,
        layout="constrained",
    )
    panels = np.atleast_1d(panels)
    model_specs = (
        ("qwen35_9b", axes9, map9, id9, c9, +0.18, "o"),
        ("qwen25_7b", axes7, map7, id7, c7, -0.18, "D"),
    )

    # panel 1: direction cosine (full layer contract)
    ax = panels[0]
    for _tag, rows, marm, iarm, color, off, mk in model_specs:
        vals, lo, hi, idd, ceil_r, nlo, nhi = ([] for _ in range(7))
        for a in axes_order:
            row = rows.get(a)
            if row is None:
                for acc in (vals, lo, hi, idd, ceil_r, nlo, nhi):
                    acc.append(float("nan"))
                continue
            d = row["direction"][marm]
            vals.append(_fnum(d["mean_cos_headline"]))
            clo, chi = _fpair(d.get("ci95"))
            lo.append(clo)
            hi.append(chi)
            idd.append(_fnum(row["direction"][iarm]["mean_cos_headline"]))
            ceil_r.append(_fnum(row["reliability"]["r10_mean"]))
            null = d.get("null") or {}
            nlo.append(_fnum(null.get("q2_5")))
            nhi.append(_fnum(null.get("q97_5")))
        y = ypos + off
        ax.hlines(y, nlo, nhi, color=color, alpha=0.25, lw=3.5, zorder=1)
        ax.plot(ceil_r, y, marker="|", ms=9, mew=1.4, color=color, ls="none", zorder=2)
        ax.plot(idd, y, marker="x", ms=5, mew=1.2, color=color, ls="none", zorder=3)
        vals_a = np.asarray(vals, dtype=np.float64)
        ax.errorbar(
            vals_a,
            y,
            xerr=_err_offsets(vals_a, np.asarray(lo), np.asarray(hi)),
            fmt=mk,
            ms=4.5,
            color=color,
            ecolor=color,
            elinewidth=1.0,
            capsize=1.6,
            ls="none",
            zorder=4,
        )
    ref = np.asarray([_fnum(ref_rows.get(a, {}).get("s_7b_ref_parent")) for a in axes_order])
    ax.scatter(ref, ypos - 0.18, facecolors="none", edgecolors=c7, marker="D", s=32, zorder=3)
    ax.axvline(0.0, color=neutral, lw=0.8, ls=":")
    ax.set_title(DISPLAY["direction_cos"], fontsize=9)

    # panel 2: calibration ratio (map + CI + identity+bias baseline)
    ax = panels[1]
    for _tag, rows, marm, iarm, color, off, mk in model_specs:
        vals, lo, hi, idd = ([] for _ in range(4))
        for a in axes_order:
            row = rows.get(a)
            if row is None:
                for acc in (vals, lo, hi, idd):
                    acc.append(float("nan"))
                continue
            cal = row["calibration"][marm]
            vals.append(_fnum(cal["ratio_to_global"]))
            clo, chi = _fpair(cal.get("ratio_to_global_ci95"))
            lo.append(clo)
            hi.append(chi)
            idd.append(_fnum(row["calibration"][iarm]["ratio_to_global"]))
        y = ypos + off
        ax.plot(idd, y, marker="x", ms=5, mew=1.2, color=color, ls="none", zorder=3)
        vals_a = np.asarray(vals, dtype=np.float64)
        ax.errorbar(
            vals_a,
            y,
            xerr=_err_offsets(vals_a, np.asarray(lo), np.asarray(hi)),
            fmt=mk,
            ms=4.5,
            color=color,
            ecolor=color,
            elinewidth=1.0,
            capsize=1.6,
            ls="none",
            zorder=4,
        )
    ax.axvline(1.0, color=neutral, lw=0.8, ls=":")
    ax.set_title(DISPLAY["calibration_ratio_to_global"], fontsize=9)

    # panel 3: observed-space separation (ceiling-adjusted; crossmodel doc)
    ax = panels[2]
    s9 = np.asarray([_fnum(sep_rows.get(a, {}).get("s_9b")) for a in axes_order])
    s7 = np.asarray([_fnum(sep_rows.get(a, {}).get("s_7b")) for a in axes_order])
    sp = np.asarray([_fnum(sep_rows.get(a, {}).get("s_7b_ref_parent")) for a in axes_order])
    # v2 (#2587 interpretation round 2): the two pilot axes carry no crossmodel
    # row, so their 9B-side separation is filled from the SAME-form statistic in
    # minpair_delta_2587.json — observed flip-norm mean / split-half noise-norm
    # mean, identical to the contrasts doc's s_9b construction (verified on
    # query_content: 10.5621 / 1.0349 = 10.206 == s_9b). Makes the 13-axis
    # separation ranking (incl. "answer-language ranks 6th of 13") visible.
    for i, a in enumerate(axes_order):
        if not math.isfinite(s9[i]):
            row = axes9.get(a)
            if row is None or not row.get("pilot_axis"):
                continue
            obs = _fnum(row["surface"]["observed"]["flip_norm_mean"])
            noise = _fnum(row["reliability"]["noise_norm_mean"])
            s9[i] = obs / noise
    ax.plot(s9, ypos + 0.18, marker="o", ms=4.5, color=c9, ls="none")
    ax.plot(s7, ypos - 0.18, marker="D", ms=4.5, color=c7, ls="none")
    ax.scatter(sp, ypos - 0.18, facecolors="none", edgecolors=c7, marker="D", s=32)
    ax.set_title(DISPLAY["obs_separation_snr"], fontsize=9)

    labels = [
        axis_label(a) + (" (pilot)" if (axes9.get(a) or {}).get("pilot_axis") else "")
        for a in axes_order
    ]
    panels[0].set_yticks(ypos)
    panels[0].set_yticklabels(labels, fontsize=8)
    handles = [
        Line2D([], [], marker="o", color=c9, ls="none", label=DISPLAY["arm_fresh9b"]),
        Line2D([], [], marker="D", color=c7, ls="none", label=DISPLAY["arm_7b_matched25k"]),
        Line2D(
            [],
            [],
            marker="D",
            mfc="none",
            mec=c7,
            color=c7,
            ls="none",
            label=DISPLAY["ref_7b_parent"],
        ),
        Line2D([], [], marker="x", color=neutral, ls="none", label=DISPLAY["iddelta_generic"]),
        Line2D([], [], marker="|", color=neutral, ls="none", label=DISPLAY["split_half_ceiling"]),
        Line2D([], [], color=neutral, alpha=0.35, lw=3.5, label=DISPLAY["null_band"]),
    ]
    fig.legend(handles=handles, loc="outside lower center", ncol=3, fontsize=7)
    # v2 filename (new-name supersession, #1482 convention): adds the pilot
    # separation dots; the round-1 fig_hero_crossmodel_axis_profile.* files
    # stay committed as the superseded render.
    paths = savefig_paper(fig, "fig_hero_crossmodel_axis_profile_v2", dir=out_dir)
    plt.close(fig)
    return list(paths.values())


def fig_crossmodel_delta_forest(inputs: dict, out_dir: Path) -> list[Path]:
    """Per-axis 9B − 7B delta with carrier-paired bootstrap 95% CI whiskers."""
    cm = inputs["crossmodel"]
    stats = cm["stats"]
    _require_stat_axes(stats, "crossmodel_contrasts.json", "direction_cos", "obs_separation_snr")
    fig, panels = plt.subplots(
        1,
        2,
        figsize=(8.6, 0.42 * len(stats["direction_cos"]["axes"]) + 1.6),
        sharey=True,
        layout="constrained",
    )
    for ax, stat in zip(panels, ("direction_cos", "obs_separation_snr")):
        rows = stats[stat]["axes"]
        axes_order = [r["axis"] for r in rows]
        ypos = np.arange(len(axes_order))[::-1]
        vals = np.asarray(
            [np.nan if r["delta_9b_minus_7b"] is None else r["delta_9b_minus_7b"] for r in rows],
            dtype=np.float64,
        )
        lo = np.asarray(
            [np.nan if r["delta_ci95"][0] is None else r["delta_ci95"][0] for r in rows],
            dtype=np.float64,
        )
        hi = np.asarray(
            [np.nan if r["delta_ci95"][1] is None else r["delta_ci95"][1] for r in rows],
            dtype=np.float64,
        )
        ax.errorbar(
            vals,
            ypos,
            xerr=_err_offsets(vals, lo, hi),
            fmt="o",
            ms=4,
            color=paper_palette_role("primary"),
            ecolor=paper_palette_role("primary"),
            elinewidth=1.1,
            capsize=2.0,
        )
        ax.axvline(0.0, color=paper_palette_role("neutral"), lw=0.8, ls=":")
        ax.set_title(f"Δ {DISPLAY[stat]} (9B − 7B)", fontsize=9)
        ax.set_yticks(ypos)
        ax.set_yticklabels([axis_label(a) for a in axes_order], fontsize=8)
    paths = savefig_paper(fig, "fig_crossmodel_delta_forest", dir=out_dir)
    plt.close(fig)
    return list(paths.values())


def fig_matched_vs_parent_scatter(inputs: dict, out_dir: Path) -> list[Path]:
    """Plan §6: 7B-matched-vs-7B-parent-arm agreement scatter — one panel per
    scale-free statistic, x = the parent's committed frozen-map reference,
    y = this run's matched-capacity 7B arm; identity line = agreement."""
    cm = inputs["crossmodel"]
    stats = cm["stats"]
    _require_stat_axes(stats, "crossmodel_contrasts.json", "direction_cos")
    stat_names = [s for s in stats if (stats[s] or {}).get("axes")]
    c7 = paper_palette_role("baseline")
    neutral = paper_palette_role("neutral")
    ncols = 3
    nrows = int(math.ceil(len(stat_names) / ncols))
    fig, panels = plt.subplots(
        nrows, ncols, figsize=(3.2 * ncols, 2.9 * nrows), layout="constrained"
    )
    panels = np.atleast_1d(panels).ravel()
    n_finite = 0
    for i, stat in enumerate(stat_names):
        ax = panels[i]
        xs, ys, labs = [], [], []
        for r in stats[stat]["axes"]:
            x, y = _fnum(r.get("s_7b_ref_parent")), _fnum(r.get("s_7b"))
            xs.append(x)
            ys.append(y)
            labs.append(r["axis"])
        xs_a, ys_a = np.asarray(xs), np.asarray(ys)
        finite = np.isfinite(xs_a) & np.isfinite(ys_a)
        n_finite += int(finite.sum())
        ax.scatter(xs_a, ys_a, color=c7, s=22)
        for x, y, a, ok in zip(xs_a, ys_a, labs, finite):
            if ok:
                ax.annotate(
                    axis_label(a), (x, y), fontsize=5.5, xytext=(2, 2), textcoords="offset points"
                )
        if finite.any():
            span = [
                float(np.nanmin([xs_a[finite].min(), ys_a[finite].min()])),
                float(np.nanmax([xs_a[finite].max(), ys_a[finite].max()])),
            ]
            ax.plot(span, span, color=neutral, lw=0.8, ls=":")
        ax.set_title(DISPLAY.get(stat, stat), fontsize=8)
        ax.set_xlabel(DISPLAY["ref_7b_parent"], fontsize=7)
        ax.set_ylabel(DISPLAY["arm_7b_matched25k"], fontsize=7)
    for j in range(len(stat_names), len(panels)):
        panels[j].set_axis_off()
    if n_finite == 0:
        plt.close(fig)
        raise RuntimeError(
            "matched_vs_parent_scatter: zero finite (parent, matched) pairs across every "
            "statistic — refusing a blank render (fail-loud)"
        )
    paths = savefig_paper(fig, "fig_matched_vs_parent_scatter", dir=out_dir)
    plt.close(fig)
    return list(paths.values())


# ── figures: perpair-grain exploratory dump ────────────────────────────


def fig_delta_norm_scatter(inputs: dict, out_dir: Path) -> list[Path]:
    """Plan §6: per-axis ‖Δ̂‖-vs-‖Δ‖ scatters — one figure per model, one
    panel per axis (the low-level per-unit view of the calibration read)."""
    by_model = _perpair_by_model(inputs["perpair"])
    neutral = paper_palette_role("neutral")
    written: list[Path] = []
    for tag, color_role in (("qwen35_9b", "primary"), ("qwen25_7b", "baseline")):
        rows = by_model[tag]
        marm = _map_arm_of(rows, f"delta_norm_scatter[{tag}]")
        color = paper_palette_role(color_role)
        axes_names = sorted({str(r["axis"]) for r in rows})
        ncols = min(4, len(axes_names))
        nrows = int(math.ceil(len(axes_names) / ncols))
        fig, panels = plt.subplots(
            nrows,
            ncols,
            figsize=(2.6 * ncols, 2.4 * nrows),
            sharex=False,
            sharey=False,
            layout="constrained",
        )
        panels = np.atleast_1d(panels).ravel()
        for i, axname in enumerate(axes_names):
            ax = panels[i]
            xs = np.asarray(
                [_fnum(r["norm_obs_tail_primary"]) for r in rows if r["axis"] == axname]
            )
            ys = np.asarray([_fnum(r["norm_pred"][marm]) for r in rows if r["axis"] == axname])
            ax.scatter(xs, ys, color=color, s=10, alpha=0.6)
            finite = np.isfinite(xs) & np.isfinite(ys)
            if finite.any():
                span = [0.0, float(np.nanmax([xs[finite].max(), ys[finite].max()]))]
                ax.plot(span, span, color=neutral, lw=0.8, ls=":")
            ax.set_title(axis_label(axname), fontsize=8)
        for j in range(len(axes_names), len(panels)):
            panels[j].set_axis_off()
        fig.supxlabel("‖observed Δ‖ (answer state)", fontsize=9)
        fig.supylabel("‖predicted Δ‖ (map arm)", fontsize=9)
        fig.suptitle(DISPLAY[tag], fontsize=10)
        paths = savefig_paper(fig, f"fig_delta_norm_scatter_{tag}", dir=out_dir)
        plt.close(fig)
        written += list(paths.values())
    return written


def fig_install_swap_violins(inputs: dict, out_dir: Path) -> list[Path]:
    """Plan §6: install-vs-swap violins — the map-arm direction-cos
    distribution per pair class, one panel per model."""
    by_model = _perpair_by_model(inputs["perpair"])
    classes = ("swap", "install")
    fig, panels = plt.subplots(1, 2, figsize=(7.2, 3.4), sharey=True, layout="constrained")
    for ax, (tag, color_role) in zip(panels, (("qwen35_9b", "primary"), ("qwen25_7b", "baseline"))):
        rows = by_model[tag]
        marm = _map_arm_of(rows, f"install_swap_violins[{tag}]")
        color = paper_palette_role(color_role)
        data = []
        for cls in classes:
            vals = np.asarray(
                [_fnum(r["cos"][marm]) for r in rows if r["pair_class"] == cls], dtype=np.float64
            )
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                raise RuntimeError(
                    f"install_swap_violins[{tag}]: no finite map-arm cos values for pair class "
                    f"{cls!r} (fail-loud)"
                )
            data.append(vals)
        parts = ax.violinplot(data, positions=range(len(classes)), showmedians=True)
        for body in parts["bodies"]:
            body.set_facecolor(color)
            body.set_alpha(0.5)
        for key in ("cmedians", "cmins", "cmaxes", "cbars"):
            if key in parts:
                parts[key].set_color(color)
        ax.axhline(0.0, color=paper_palette_role("neutral"), lw=0.8, ls=":")
        ax.set_xticks(range(len(classes)))
        ax.set_xticklabels([DISPLAY[c] for c in classes], fontsize=8)
        ax.set_title(DISPLAY[tag], fontsize=9)
    panels[0].set_ylabel("direction cosine (map arm)")
    paths = savefig_paper(fig, "fig_install_swap_violins", dir=out_dir)
    plt.close(fig)
    return list(paths.values())


def fig_edit_dose_scatter(inputs: dict, out_dir: Path) -> list[Path]:
    """Plan §6: edit-dose scatters per tokenizer — changed tokens (each
    model's OWN tokenizer) vs ‖observed Δ‖, with each side's pooled OLS line
    read from the analysis doc (never re-fit here — single estimator source:
    issue2587_analysis.compute_side's dose_fit block, identical on every
    axis row of a side by construction)."""
    by_model = _perpair_by_model(inputs["perpair"])
    sides = _delta_sides(inputs["delta"])
    fig, panels = plt.subplots(1, 2, figsize=(7.6, 3.4), sharey=False, layout="constrained")
    for ax, (tag, color_role) in zip(panels, (("qwen35_9b", "primary"), ("qwen25_7b", "baseline"))):
        rows = by_model[tag]
        color = paper_palette_role(color_role)
        xs = np.asarray([float(r["changed_tokens"]) for r in rows])
        ys = np.asarray([_fnum(r["norm_obs_tail_primary"]) for r in rows])
        ax.scatter(xs, ys, color=color, s=10, alpha=0.45)
        axes_map = sides[tag]["axes"]
        first_axis = sorted(axes_map)[0]
        ols = axes_map[first_axis]["surface"]["observed"]["edit_dose_ols"]
        icpt, slope = _fnum(ols["intercept"]), _fnum(ols["slope"])
        if math.isfinite(icpt) and math.isfinite(slope) and np.isfinite(xs).any():
            gx = np.linspace(float(np.nanmin(xs)), float(np.nanmax(xs)), 20)
            ax.plot(gx, icpt + slope * gx, color=color, lw=1.5, label="pooled OLS")
            ax.legend(fontsize=7)
        ax.set_title(DISPLAY[tag], fontsize=9)
        ax.set_xlabel("changed tokens (own tokenizer)")
    panels[0].set_ylabel("‖observed Δ‖ (answer state)")
    paths = savefig_paper(fig, "fig_edit_dose_scatter", dir=out_dir)
    plt.close(fig)
    return list(paths.values())


def fig_carrier_direction_heatmap(inputs: dict, out_dir: Path) -> list[Path]:
    """Plan §6: per-carrier direction-cos transfer matrices — carrier × axis
    mean map-arm direction cos per model (does the axis direction transfer
    across carrier topics?). Figure height scales with the carrier-row count
    (78 carrier / carrier-pair rows realized) so every y tick label renders
    legibly — the fixed 4.2-in height smeared the row labels (r2 concern
    `per-unit-figure-unreadable`)."""
    by_model = _perpair_by_model(inputs["perpair"])
    n_rows_max = max(len({str(r["carrier"]) for r in rows}) for rows in by_model.values())
    fig_h = max(4.2, 0.17 * n_rows_max + 1.8)
    fig, panels = plt.subplots(1, 2, figsize=(10.0, fig_h), layout="constrained")
    im = None
    for ax, tag in zip(panels, MODEL_TAGS):
        rows = by_model[tag]
        marm = _map_arm_of(rows, f"carrier_direction_heatmap[{tag}]")
        carriers = sorted({str(r["carrier"]) for r in rows})
        axes_names = sorted({str(r["axis"]) for r in rows})
        acc: dict[tuple[str, str], list[float]] = {}
        for r in rows:
            v = _fnum(r["cos"][marm])
            if math.isfinite(v):
                acc.setdefault((str(r["carrier"]), str(r["axis"])), []).append(v)
        mat = np.full((len(carriers), len(axes_names)), np.nan)
        for i, car in enumerate(carriers):
            for j, axname in enumerate(axes_names):
                vals = acc.get((car, axname))
                if vals:
                    mat[i, j] = float(np.mean(vals))
        im = ax.imshow(mat, cmap="coolwarm", vmin=-1.0, vmax=1.0, aspect="auto")
        ax.set_xticks(range(len(axes_names)))
        ax.set_xticklabels([axis_label(a) for a in axes_names], fontsize=7, rotation=45, ha="right")
        ax.set_yticks(range(len(carriers)))
        ax.set_yticklabels([c[:18] for c in carriers], fontsize=7)
        ax.set_title(DISPLAY[tag], fontsize=9)
    fig.colorbar(im, ax=list(panels), shrink=0.6, label="mean direction cosine (map arm)")
    paths = savefig_paper(fig, "fig_carrier_direction_heatmap", dir=out_dir)
    plt.close(fig)
    return list(paths.values())


def fig_text_space_rank_scatter(inputs: dict, out_dir: Path) -> list[Path]:
    """Interpretation-r2 answer-text control: the per-axis RANK view behind
    the two rank correlations in ``text_space_rank_reads.json`` — left, the
    per-axis answer-text shift-norm ranks of the two models against each
    other (axes with text reads on both sides); right, the Qwen3.5-9B
    observed-separation rank against its own text-norm rank. Same sources as
    ``scripts/issue2587_text_rank.py`` (``text_space.flip_norm_mean`` per
    side + ``obs_separation_snr``); tie-corrected average ranks
    (scipy ``rankdata`` on negated values), rank 1 = largest."""
    sides = _delta_sides(inputs["delta"])
    cm = inputs["crossmodel"]
    _require_stat_axes(cm["stats"], "text_space_rank_scatter", "obs_separation_snr")
    s9_axes = sides["qwen35_9b"]["axes"]
    s7_axes = sides["qwen25_7b"]["axes"]

    def _text(side_axes: dict, axis: str) -> float | None:
        ts = side_axes.get(axis, {}).get("text_space") or {}
        return ts.get("flip_norm_mean")

    both = sorted(
        a for a in s7_axes if _text(s7_axes, a) is not None and _text(s9_axes, a) is not None
    )
    sep9 = {r["axis"]: _fnum(r["s_9b"]) for r in cm["stats"]["obs_separation_snr"]["axes"]}
    sep_axes = sorted(a for a in sep9 if math.isfinite(sep9[a]) and _text(s9_axes, a) is not None)
    if len(both) < 2 or len(sep_axes) < 2:
        raise RuntimeError(
            f"text_space_rank_scatter: too few plottable axes (cross-model {len(both)}, "
            f"separation-vs-text {len(sep_axes)}) — refusing a blank render (fail-loud)"
        )

    def _ranks(vals: list[float]) -> np.ndarray:
        return rankdata([-float(v) for v in vals])

    primary = paper_palette_role("primary")
    neutral = paper_palette_role("neutral")
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(9.2, 4.4), layout="constrained")
    panels = (
        (
            ax_l,
            _ranks([_text(s7_axes, a) for a in both]),
            _ranks([_text(s9_axes, a) for a in both]),
            both,
            "answer-text shift-norm rank, " + DISPLAY["qwen25_7b"] + " (1 = largest)",
            "Answer-text ordering across the two models",
        ),
        (
            ax_r,
            _ranks([sep9[a] for a in sep_axes]),
            _ranks([_text(s9_axes, a) for a in sep_axes]),
            sep_axes,
            "observed-separation rank, " + DISPLAY["qwen35_9b"] + " (1 = largest)",
            "9B separation vs 9B answer-text ordering",
        ),
    )
    for ax, xs, ys, names, xlabel, title in panels:
        hi = float(max(xs.max(), ys.max())) + 0.6
        ax.plot([0.4, hi], [0.4, hi], color=neutral, ls="--", lw=1.0, zorder=0)
        ax.scatter(xs, ys, color=primary, s=26, zorder=2)
        for x, y, name in zip(xs, ys, names):
            ax.annotate(
                axis_label(name), (x, y), fontsize=7, xytext=(4, 3), textcoords="offset points"
            )
        ax.set_xlim(0.4, hi + 0.9)
        ax.set_ylim(0.4, hi)
        ax.set_xlabel(xlabel, fontsize=8)
        ax.set_title(title, fontsize=9)
    ax_l.set_ylabel(
        "answer-text shift-norm rank, " + DISPLAY["qwen35_9b"] + " (1 = largest)", fontsize=8
    )
    paths = savefig_paper(fig, "fig_text_space_rank_scatter", dir=out_dir)
    plt.close(fig)
    return list(paths.values())


# ── figures: delta-doc-grain exploratory dump ──────────────────────────


def fig_axis_identity_heatmap(inputs: dict, out_dir: Path) -> list[Path]:
    """Plan §6: axis-identity heatmaps — the per-value-pair carrier-mean
    identity cos (map arm) per axis, per model (the low-level per-unit view
    of the axis_identity_cos statistic; dyad/pilot classes with no
    carrier-replicated grid are 'n/a' by construction and are skipped)."""
    sides = _delta_sides(inputs["delta"])
    fig, panels = plt.subplots(1, 2, figsize=(9.6, 4.0), layout="constrained")
    im = None
    any_rows = False
    for ax, tag in zip(panels, MODEL_TAGS):
        axes_map, marm, _ = _side_layers(sides, tag)
        rows = []
        names = []
        max_vp = 0
        for axname in sorted(axes_map):
            ident = axes_map[axname].get("identity") or {}
            blk = ident.get(marm)
            if not isinstance(blk, dict) or "per_vp_cos" not in blk:
                continue  # {"n/a": ...} classes carry no carrier-replicated grid
            vp = blk["per_vp_cos"]
            rows.append([_fnum(v) for _, v in sorted(vp.items())])
            names.append(axname)
            max_vp = max(max_vp, len(vp))
        if rows:
            any_rows = True
            mat = np.full((len(rows), max_vp), np.nan)
            for i, r in enumerate(rows):
                mat[i, : len(r)] = r
            im = ax.imshow(mat, cmap="coolwarm", vmin=-1.0, vmax=1.0, aspect="auto")
            ax.set_yticks(range(len(names)))
            ax.set_yticklabels([axis_label(a) for a in names], fontsize=7)
            ax.set_xticks(range(max_vp))
            ax.set_xticklabels([str(k + 1) for k in range(max_vp)], fontsize=6)
            ax.set_xlabel("value pair (within axis)", fontsize=8)
        ax.set_title(DISPLAY[tag], fontsize=9)
    if not any_rows:
        plt.close(fig)
        raise RuntimeError(
            "axis_identity_heatmap: no axis carries a per_vp_cos identity block on either "
            "side — refusing a blank render (fail-loud)"
        )
    fig.colorbar(im, ax=list(panels), shrink=0.8, label="axis identity cosine (map arm)")
    paths = savefig_paper(fig, "fig_axis_identity_heatmap", dir=out_dir)
    plt.close(fig)
    return list(paths.values())


def fig_crossfam_consistency_scatter(inputs: dict, out_dir: Path) -> list[Path]:
    """Plan §6: cross-family consistency scatters (observed + predicted, both
    models) — per axis, x = observed-space median, y = map-arm predicted-space
    median (headline reads; axes with no paraphrase-family swap class are
    'n/a' by construction and are skipped)."""
    sides = _delta_sides(inputs["delta"])
    neutral = paper_palette_role("neutral")
    fig, ax = plt.subplots(figsize=(4.8, 4.2), layout="constrained")
    n_finite = 0
    for tag, color_role, mk in (("qwen35_9b", "primary", "o"), ("qwen25_7b", "baseline", "D")):
        axes_map, marm, _ = _side_layers(sides, tag)
        color = paper_palette_role(color_role)
        xs, ys, labs = [], [], []
        for axname in sorted(axes_map):
            cf = axes_map[axname].get("cross_family") or {}
            obs, prd = cf.get("observed"), cf.get(marm)
            if not isinstance(obs, dict) or not isinstance(prd, dict):
                continue
            xs.append(_fnum(obs.get("median")))
            ys.append(_fnum(prd.get("median")))
            labs.append(axname)
        xs_a, ys_a = np.asarray(xs), np.asarray(ys)
        finite = np.isfinite(xs_a) & np.isfinite(ys_a)
        n_finite += int(finite.sum())
        ax.scatter(xs_a, ys_a, color=color, marker=mk, s=24, label=DISPLAY[tag])
        for x, y, a, ok in zip(xs_a, ys_a, labs, finite):
            if ok:
                ax.annotate(
                    axis_label(a), (x, y), fontsize=5.5, xytext=(2, 2), textcoords="offset points"
                )
    if n_finite == 0:
        plt.close(fig)
        raise RuntimeError(
            "crossfam_consistency_scatter: zero finite (observed, predicted) medians on both "
            "sides — refusing a blank render (fail-loud)"
        )
    ax.plot([-1, 1], [-1, 1], color=neutral, lw=0.8, ls=":")
    ax.set_xlabel("cross-family consistency (observed space)")
    ax.set_ylabel("cross-family consistency (predicted space)")
    ax.legend(fontsize=7)
    paths = savefig_paper(fig, "fig_crossfam_consistency_scatter", dir=out_dir)
    plt.close(fig)
    return list(paths.values())


def fig_delta_retrieval_acc(inputs: dict, out_dir: Path) -> list[Path]:
    """Plan §6: battery-side Δ-retrieval acc@k curves per arm (distinct from
    the FIT-side knn_per_layer): P(true observed Δ within k NN of the
    predicted Δ) over the oriented pair pool, per model. Color contract:
    model color = euclidean, accent = cosine; solid = map arm, dashed =
    identity+bias baseline."""
    sides = _delta_sides(inputs["delta"])
    accent = paper_palette_role("accent")
    neutral = paper_palette_role("neutral")
    fig, panels = plt.subplots(1, 2, figsize=(7.6, 3.4), sharey=True, layout="constrained")
    for ax, (tag, color_role) in zip(panels, (("qwen35_9b", "primary"), ("qwen25_7b", "baseline"))):
        color = paper_palette_role(color_role)
        _axes_map, marm, iarm = _side_layers(sides, tag)
        retrieval = sides[tag].get("retrieval") or {}
        glob = retrieval.get("global") or {}
        if not glob:
            plt.close(fig)
            raise RuntimeError(
                f"delta_retrieval_acc[{tag}]: no retrieval.global block in "
                "minpair_delta_2587.json (fail-loud)"
            )
        for arm, ls, arm_lab in ((marm, "-", "map arm"), (iarm, "--", DISPLAY["iddelta_generic"])):
            for metric, mcolor in (("euclidean", color), ("cosine", accent)):
                blk = glob[arm][metric]
                ks = sorted(int(k) for k in blk["acc_at_k"])
                accs = [float(blk["acc_at_k"][str(k)]) for k in ks]
                ax.plot(
                    ks,
                    accs,
                    color=mcolor,
                    ls=ls,
                    marker="o",
                    ms=3,
                    lw=1.3,
                    label=f"{arm_lab}, {metric}",
                )
        chance_blk = glob[marm]["euclidean"].get("chance_at_k") or {}
        if chance_blk:
            ks = sorted(int(k) for k in chance_blk)
            ax.plot(
                ks,
                [float(chance_blk[str(k)]) for k in ks],
                color=neutral,
                ls=":",
                lw=0.9,
                label="chance",
            )
        ax.set_title(DISPLAY[tag], fontsize=9)
        ax.set_xlabel("k (nearest neighbours)")
    panels[0].set_ylabel("Δ-retrieval accuracy")
    panels[0].legend(fontsize=6)
    paths = savefig_paper(fig, "fig_delta_retrieval_acc", dir=out_dir)
    plt.close(fig)
    return list(paths.values())


def fig_splithalf_vs_direction(inputs: dict, out_dir: Path) -> list[Path]:
    """Plan §6: split-half-vs-direction scatters — per axis, x = split-half
    reliability ceiling (r10), y = map-arm direction cos (headline), points
    labeled by axis."""
    sides = _delta_sides(inputs["delta"])
    fig, ax = plt.subplots(figsize=(4.8, 4.2), layout="constrained")
    n_finite = 0
    for tag, color_role, mk in (("qwen35_9b", "primary", "o"), ("qwen25_7b", "baseline", "D")):
        axes_map, marm, _ = _side_layers(sides, tag)
        color = paper_palette_role(color_role)
        for axname in sorted(axes_map):
            row = axes_map[axname]
            x = _fnum(row["reliability"]["r10_mean"])
            y = _fnum(row["direction"][marm]["mean_cos_headline"])
            ax.scatter(
                [x],
                [y],
                color=color,
                marker=mk,
                s=24,
                label=DISPLAY[tag] if n_finite == 0 and tag == "qwen35_9b" else None,
            )
            if math.isfinite(x) and math.isfinite(y):
                n_finite += 1
                ax.annotate(
                    axis_label(axname),
                    (x, y),
                    fontsize=5.5,
                    xytext=(2, 2),
                    textcoords="offset points",
                )
    if n_finite == 0:
        plt.close(fig)
        raise RuntimeError(
            "splithalf_vs_direction: zero finite (ceiling, direction) pairs — refusing a "
            "blank render (fail-loud)"
        )
    handles = [
        Line2D(
            [],
            [],
            marker="o",
            color=paper_palette_role("primary"),
            ls="none",
            label=DISPLAY["qwen35_9b"],
        ),
        Line2D(
            [],
            [],
            marker="D",
            color=paper_palette_role("baseline"),
            ls="none",
            label=DISPLAY["qwen25_7b"],
        ),
    ]
    ax.legend(handles=handles, fontsize=7)
    ax.axhline(0.0, color=paper_palette_role("neutral"), lw=0.8, ls=":")
    ax.set_xlabel("split-half reliability ceiling (r10)")
    ax.set_ylabel("direction cosine (map arm)")
    paths = savefig_paper(fig, "fig_splithalf_vs_direction", dir=out_dir)
    plt.close(fig)
    return list(paths.values())


def fig_pilot_axis_panels(inputs: dict, out_dir: Path) -> list[Path]:
    """Plan §6: pilot-axis panels — the two pilot axes' 9B reads (direction
    cos, calibration ratio, split-half ceiling, with CIs). The 7B side is
    PENDING by plan convention 12 (the parent's pilot judge reads have not
    landed) and is rendered as an explicit pending legend entry, never a
    placeholder number."""
    sides = _delta_sides(inputs["delta"])
    axes9, map9, id9 = _side_layers(sides, "qwen35_9b")
    pilots = sorted(a for a, row in axes9.items() if row.get("pilot_axis"))
    if not pilots:
        raise RuntimeError(
            "pilot_axis_panels: no pilot axes present on the 9B side (production carries "
            "answer_language + the pilot query class) — refusing a blank render (fail-loud)"
        )
    c9 = paper_palette_role("primary")
    c7 = paper_palette_role("baseline")
    neutral = paper_palette_role("neutral")
    ypos = np.arange(len(pilots))[::-1].astype(np.float64)
    fig, panels = plt.subplots(
        1, 3, figsize=(9.6, 0.6 * len(pilots) + 1.9), sharey=True, layout="constrained"
    )

    def _panel(ax, vals, lo, hi, idd, title, refline):
        vals_a = np.asarray(vals)
        ax.errorbar(
            vals_a,
            ypos,
            xerr=_err_offsets(vals_a, np.asarray(lo), np.asarray(hi)),
            fmt="o",
            ms=5,
            color=c9,
            ecolor=c9,
            elinewidth=1.1,
            capsize=2.0,
            ls="none",
        )
        if idd is not None:
            ax.plot(idd, ypos, marker="x", ms=6, mew=1.3, color=c9, ls="none")
        if refline is not None:
            ax.axvline(refline, color=neutral, lw=0.8, ls=":")
        ax.set_title(title, fontsize=9)

    d_vals, d_lo, d_hi, d_idd = [], [], [], []
    c_vals, c_lo, c_hi, c_idd = [], [], [], []
    r_vals, r_lo, r_hi = [], [], []
    for a in pilots:
        row = axes9[a]
        d = row["direction"][map9]
        d_vals.append(_fnum(d["mean_cos_headline"]))
        lo, hi = _fpair(d.get("ci95"))
        d_lo.append(lo)
        d_hi.append(hi)
        d_idd.append(_fnum(row["direction"][id9]["mean_cos_headline"]))
        cal = row["calibration"][map9]
        c_vals.append(_fnum(cal["ratio_to_global"]))
        lo, hi = _fpair(cal.get("ratio_to_global_ci95"))
        c_lo.append(lo)
        c_hi.append(hi)
        c_idd.append(_fnum(row["calibration"][id9]["ratio_to_global"]))
        rel = row["reliability"]
        r_vals.append(_fnum(rel["r10_mean"]))
        lo, hi = _fpair(rel.get("r10_ci95"))
        r_lo.append(lo)
        r_hi.append(hi)
    _panel(panels[0], d_vals, d_lo, d_hi, d_idd, DISPLAY["direction_cos"], 0.0)
    _panel(panels[1], c_vals, c_lo, c_hi, c_idd, DISPLAY["calibration_ratio_to_global"], 1.0)
    _panel(panels[2], r_vals, r_lo, r_hi, None, DISPLAY["split_half_ceiling"], None)
    panels[0].set_yticks(ypos)
    panels[0].set_yticklabels([axis_label(a) for a in pilots], fontsize=8)
    handles = [
        Line2D([], [], marker="o", color=c9, ls="none", label=DISPLAY["arm_fresh9b"]),
        Line2D([], [], marker="x", color=c9, ls="none", label=DISPLAY["arm_iddelta9b"]),
        Line2D([], [], marker="D", color=c7, ls="none", alpha=0.3, label=DISPLAY["pending_7b"]),
    ]
    fig.legend(handles=handles, loc="outside lower center", ncol=3, fontsize=7)
    paths = savefig_paper(fig, "fig_pilot_axis_panels", dir=out_dir)
    plt.close(fig)
    return list(paths.values())


def fig_lstar_sensitivity_twins(inputs: dict, out_dir: Path) -> list[Path]:
    """Plan §6: L*-vs-{16,22,30} sensitivity twins of hero 1 — per axis, the
    identity+bias (iddelta) direction cos and calibration ratio at L* vs the
    twin layers. IDDELTA ARM ONLY by construction: the frozen fresh map is
    L*-fit, so no map arm exists off L* (plan cross-unit constraint 5 — the
    twins are a sensitivity read, never a cross-model read); the 7B side has
    no twin layers and is not rendered."""
    sides = _delta_sides(inputs["delta"])
    axes9, _map9, id9 = _side_layers(sides, "qwen35_9b")
    meta9 = sides["qwen35_9b"]["meta"]
    twin_layers = [int(x) for x in meta9.get("twin_layers") or []]
    if not twin_layers:
        raise RuntimeError(
            "lstar_sensitivity_twins: the 9B side declares no twin layers — refusing a "
            "blank render (fail-loud)"
        )
    lstar = int(meta9["primary_layer"])
    c9 = paper_palette_role("primary")
    neutral = paper_palette_role("neutral")
    axes_order = sorted(axes9)
    ypos = np.arange(len(axes_order))[::-1].astype(np.float64)
    twin_markers = ("^", "s", "v", "P", "X")
    fig, panels = plt.subplots(
        1, 2, figsize=(7.8, 0.42 * len(axes_order) + 2.0), sharey=True, layout="constrained"
    )

    def _twin_val(row: dict, layer: int, key: str) -> float:
        tw = row.get("layer_twins") or {}
        blk = tw.get(str(layer))
        return _fnum(blk.get(key)) if isinstance(blk, dict) else float("nan")

    for ax, lstar_getter, twin_key, title, refline in (
        (
            panels[0],
            lambda row: _fnum(row["direction"][id9]["mean_cos_headline"]),
            "iddelta_mean_cos_headline",
            "direction cosine (identity+bias baseline)",
            0.0,
        ),
        (
            panels[1],
            lambda row: _fnum(row["calibration"][id9]["ratio_to_global"]),
            "iddelta_ratio_to_global",
            "calibration ratio (identity+bias baseline)",
            1.0,
        ),
    ):
        star = [lstar_getter(axes9[a]) for a in axes_order]
        ax.plot(
            star,
            ypos,
            marker="*",
            ms=9,
            color=c9,
            mec="black",
            mew=0.4,
            ls="none",
            label=f"layer {lstar} (L*)",
        )
        for mk, layer in zip(twin_markers, twin_layers):
            vals = [_twin_val(axes9[a], layer, twin_key) for a in axes_order]
            ax.plot(
                vals,
                ypos,
                marker=mk,
                ms=5,
                mfc="none",
                color=c9,
                ls="none",
                label=f"layer {layer} (twin)",
            )
        ax.axvline(refline, color=neutral, lw=0.8, ls=":")
        ax.set_title(title, fontsize=9)
    panels[0].set_yticks(ypos)
    panels[0].set_yticklabels([axis_label(a) for a in axes_order], fontsize=8)
    panels[0].legend(fontsize=6, loc="lower left")
    fig.suptitle(DISPLAY["qwen35_9b"], fontsize=10)
    paths = savefig_paper(fig, "fig_lstar_sensitivity_twins", dir=out_dir)
    plt.close(fig)
    return list(paths.values())


def fig_pooling_twin_scatter(inputs: dict, out_dir: Path) -> list[Path]:
    """Plan §6: span-mean pooling twin — per axis, x = the primary
    tail-inclusive-mean direction cos, y = the answer-span-mean twin, per
    model and arm (agreement = the pooling convention does not drive the
    read)."""
    sides = _delta_sides(inputs["delta"])
    neutral = paper_palette_role("neutral")
    fig, ax = plt.subplots(figsize=(4.8, 4.2), layout="constrained")
    n_finite = 0
    for tag, color_role, mk in (("qwen35_9b", "primary", "o"), ("qwen25_7b", "baseline", "D")):
        axes_map, marm, iarm = _side_layers(sides, tag)
        color = paper_palette_role(color_role)
        for arm, marker, alpha in ((marm, mk, 0.9), (iarm, "x", 0.7)):
            xs, ys, labs = [], [], []
            for axname in sorted(axes_map):
                row = axes_map[axname]
                xs.append(_fnum(row["direction"][arm]["mean_cos_headline"]))
                ys.append(_fnum(row["pooling_twin_span"][arm]["mean_cos_headline"]))
                labs.append(axname)
            xs_a, ys_a = np.asarray(xs), np.asarray(ys)
            finite = np.isfinite(xs_a) & np.isfinite(ys_a)
            n_finite += int(finite.sum())
            ax.scatter(xs_a, ys_a, color=color, marker=marker, s=22, alpha=alpha)
            if arm == marm:
                for x, y, a, ok in zip(xs_a, ys_a, labs, finite):
                    if ok:
                        ax.annotate(
                            axis_label(a),
                            (x, y),
                            fontsize=5.5,
                            xytext=(2, 2),
                            textcoords="offset points",
                        )
    if n_finite == 0:
        plt.close(fig)
        raise RuntimeError(
            "pooling_twin_scatter: zero finite (tail, span) direction-cos pairs — refusing "
            "a blank render (fail-loud)"
        )
    ax.plot([-1, 1], [-1, 1], color=neutral, lw=0.8, ls=":")
    handles = [
        Line2D(
            [],
            [],
            marker="o",
            color=paper_palette_role("primary"),
            ls="none",
            label=DISPLAY["qwen35_9b"],
        ),
        Line2D(
            [],
            [],
            marker="D",
            color=paper_palette_role("baseline"),
            ls="none",
            label=DISPLAY["qwen25_7b"],
        ),
        Line2D([], [], marker="x", color=neutral, ls="none", label=DISPLAY["iddelta_generic"]),
    ]
    ax.legend(handles=handles, fontsize=7)
    ax.set_xlabel(f"direction cosine ({DISPLAY['pooling_tail']})")
    ax.set_ylabel(f"direction cosine ({DISPLAY['pooling_span']})")
    paths = savefig_paper(fig, "fig_pooling_twin_scatter", dir=out_dir)
    plt.close(fig)
    return list(paths.values())


# ── tables: think-leak/cap-hit, manipulation check, token counts ───────


def think_leak_cap_hit_table(inputs: dict, out_dir: Path) -> list[Path]:
    """Plan §6/§13: think-leak + cap-hit tables (md + json) — battery gen
    cells (anchors_*.done.json) + map-fit generation splits (cap_hit_*.json),
    with the plan-§7 disclosure thresholds (cap-hit re-gen trigger 2%,
    think-leak assert 1%) flagged per row."""
    lk = inputs["leakdir"]
    rows: list[dict] = []
    for p, doc in lk["gen"].items():
        tl = doc["think_leak"]
        rows.append(
            {
                "source": str(p.relative_to(lk["dir"])),
                "kind": "battery generation cell",
                "unit": p.name[len("anchors_") : -len(".done.json")],
                "n_rows": int(tl["n"]),
                "cap_hit_frac": _fnum(doc["cap_hit_frac"]),
                "cap_hit_frac_after_regen": _fnum(doc.get("cap_hit_frac_regen")),
                "think_leak_n": int(tl["n_leaked"]),
                "think_leak_frac": _fnum(tl["frac"]),
            }
        )
    for p, doc in lk["cap"].items():
        rows.append(
            {
                "source": str(p.relative_to(lk["dir"])),
                "kind": "map-fit generation split",
                "unit": str(doc.get("split") or p.stem[len("cap_hit_") :]),
                "n_rows": int(doc["total"]) if doc.get("total") is not None else None,
                "cap_hit_frac": _fnum(doc["cap_hit_frac"]),
                "cap_hit_frac_after_regen": float("nan"),
                "think_leak_n": None,
                "think_leak_frac": float("nan"),
            }
        )
    for r in rows:
        eff_cap = (
            r["cap_hit_frac_after_regen"]
            if math.isfinite(r["cap_hit_frac_after_regen"])
            else r["cap_hit_frac"]
        )
        r["cap_hit_over_regen_trigger"] = (
            bool(eff_cap > CAP_HIT_REGEN_TRIGGER) if math.isfinite(eff_cap) else None
        )
        r["think_leak_over_assert"] = (
            bool(r["think_leak_frac"] >= THINK_LEAK_ASSERT)
            if math.isfinite(r["think_leak_frac"])
            else None
        )
    rows.sort(key=lambda r: (r["kind"], r["unit"]))

    lines = [
        "# Think-leak + cap-hit table — issue 2587",
        "",
        f"Cap-hit re-gen trigger {CAP_HIT_REGEN_TRIGGER:.0%} per cell/split; think-leak "
        f"assert < {THINK_LEAK_ASSERT:.0%} per cell (plan §7).",
        "",
        "| unit | kind | rows | cap-hit | cap-hit after re-gen | think-leak | flags |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        flags = []
        if r["cap_hit_over_regen_trigger"]:
            flags.append("cap-hit over re-gen trigger")
        if r["think_leak_over_assert"]:
            flags.append("think-leak over assert")
        lines.append(
            f"| {axis_label(r['unit']) if r['kind'].startswith('battery') else r['unit']} "
            f"| {r['kind']} | {_fmt(r['n_rows'])} | {_fmt(r['cap_hit_frac'])} "
            f"| {_fmt(r['cap_hit_frac_after_regen'])} | {_fmt(r['think_leak_frac'])} "
            f"| {'; '.join(flags) if flags else 'ok'} |"
        )
    doc = {
        "issue": ISSUE,
        "thresholds": {
            "cap_hit_regen_trigger": CAP_HIT_REGEN_TRIGGER,
            "think_leak_assert": THINK_LEAK_ASSERT,
        },
        "rows": [
            {
                k: (None if isinstance(v, float) and not math.isfinite(v) else v)
                for k, v in r.items()
            }
            for r in rows
        ],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    md_path = out_dir / "table_think_leak_cap_hit.md"
    json_path = out_dir / "table_think_leak_cap_hit.json"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    json_path.write_text(json.dumps(doc, indent=2, sort_keys=True), encoding="utf-8")
    return [md_path, json_path]


def manipulation_check_table(inputs: dict, out_dir: Path) -> list[Path]:
    """§13 deliverable: the manipulation-check table (md + json) — per-axis
    fire floors for BOTH models side by side (9B this run; 7B the parent's
    committed check). Special axis rows (not in slice / query classes) render
    their verdict, never a fabricated count."""
    m9, m7 = inputs["manip9b"], inputs["manip7b"]

    def _axis_rows(doc: dict, what: str) -> dict:
        rows = {r["axis"]: r for r in doc.get("axis_rows", [])}
        if not rows:
            raise RuntimeError(f"{what}: no axis_rows in the manipulation check (fail-loud)")
        return rows

    a9 = _axis_rows(m9, "manipulation_check_2587.json")
    a7 = _axis_rows(m7, "parent manipulation_check.json")

    def _cell(r: dict | None) -> str:
        if r is None:
            return "not judged"
        if "floor_met" not in r:
            v = str(r.get("verdict", "n/a"))
            return DISPLAY.get(v, v.replace("_", " "))
        return (
            f"{r['n_fired_base']}/{r['width']} fired "
            f"(floor {r['floor']}: {'met' if r['floor_met'] else 'MISSED'})"
        )

    axes = sorted(set(a9) | set(a7))
    lines = [
        "# Manipulation-check table — issue 2587",
        "",
        "Per-axis fire floors over BASE values (>=70% comply per value; floor = "
        "ceil(0.6 x width); undetermined counts as not fired).",
        "",
        f"| axis | {DISPLAY['qwen35_9b']} | {DISPLAY['qwen25_7b']} |",
        "|---|---|---|",
    ]
    for a in axes:
        lines.append(f"| {axis_label(a)} | {_cell(a9.get(a))} | {_cell(a7.get(a))} |")

    def _floor_summary(rows: dict) -> str:
        floored = [r for r in rows.values() if "floor_met" in r]
        met = sum(1 for r in floored if r["floor_met"])
        return f"{met}/{len(floored)} judged axes meet the fire floor"

    lines += [
        "",
        f"{DISPLAY['qwen35_9b']}: {_floor_summary(a9)}. "
        f"{DISPLAY['qwen25_7b']}: {_floor_summary(a7)}.",
    ]
    doc = {
        "issue": ISSUE,
        "axes": {a: {"qwen35_9b": a9.get(a), "qwen25_7b": a7.get(a)} for a in axes},
        "n_value_rows": {
            "qwen35_9b": len(m9.get("value_rows", [])),
            "qwen25_7b": len(m7.get("value_rows", [])),
        },
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    md_path = out_dir / "table_manipulation_check.md"
    json_path = out_dir / "table_manipulation_check.json"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    json_path.write_text(json.dumps(doc, indent=2, sort_keys=True), encoding="utf-8")
    return [md_path, json_path]


def token_count_equality_table(inputs: dict, out_dir: Path) -> list[Path]:
    """Plan §6: q25-vs-q35 token-count-equality table (md + json) — per axis,
    the realized Qwen3.5 (q35) value-string token counts vs the parent's
    pinned Qwen2.5 (q25) expectation; within-axis equality REPORTED, never
    asserted (bank2587 token gate iii)."""
    bank = inputs["bank"]
    tg = bank.get("token_gates")
    if not tg:
        raise RuntimeError(
            "bank_manifest.json carries no token_gates block — run the P0b token gates "
            "(bank2587.run_token_gates) before rendering this table (fail-loud)"
        )
    vals: dict = tg["value_token_counts"]
    eq: dict = tg.get("within_axis_equal") or {}
    q25: dict = tg.get("q25_expected_value_tokens") or {}
    paras: dict = tg.get("paraphrase_token_counts") or {}
    names: dict = tg.get("name_token_counts") or {}

    def _counts_str(d: dict | None) -> str:
        if not d:
            return "n/a"
        distinct = sorted(set(int(v) for v in d.values()))
        return str(distinct[0]) if len(distinct) == 1 else ", ".join(str(v) for v in distinct)

    lines = [
        "# q25-vs-q35 token-count-equality table — issue 2587",
        "",
        "Value-string token counts under the Qwen3.5 tokenizer (q35, this run) vs the",
        "parent's pinned Qwen2.5 expectation (q25). Within-axis equality held by",
        "construction under q25; under q35 it is RECORDED, never assumed.",
        "",
        "| axis | q35 counts (distinct) | q35 within-axis equal | q25 expected "
        "| q35 paraphrase counts |",
        "|---|---|---|---|---|",
    ]
    for a in sorted(vals):
        q25v = q25.get(a)
        lines.append(
            f"| {axis_label(a)} | {_counts_str(vals[a])} "
            f"| {'yes' if eq.get(a) else 'no'} | {_fmt(q25v)} | {_counts_str(paras.get(a))} |"
        )
    if names:
        n_single = sum(1 for v in names.values() if v.get("single_token"))
        lines += [
            "",
            f"Name tokens (q35): {n_single}/{len(names)} names remain single-token "
            "(the q25 single-token property is recorded per name, never assumed).",
        ]
    doc = {
        "issue": ISSUE,
        "value_token_counts_q35": vals,
        "within_axis_equal_q35": eq,
        "q25_expected_value_tokens": q25,
        "paraphrase_token_counts_q35": paras,
        "name_token_counts_q35": names,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    md_path = out_dir / "table_token_count_equality.md"
    json_path = out_dir / "table_token_count_equality.json"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    json_path.write_text(json.dumps(doc, indent=2, sort_keys=True), encoding="utf-8")
    return [md_path, json_path]


def fig_intrusion_rates(inputs: dict, out_dir: Path) -> list[Path]:
    """Per-cell CJK stray-character intrusion rates: 9B this run vs 7B parent.

    Reads the analyzer-round scan (``intrusion_scan_2587.json``) and the
    parent's committed audit (``intrusion_audit.json`` @ d313b856, #2564).
    The answer-language cell is excluded (its instructed-Chinese arm makes
    CJK content compliance, not intrusion); the one-word query pilot has no
    parent counterpart and renders 9B-only. Wald 95% intervals per cell.
    """
    scan = inputs["intrusion"]
    parent = inputs["parent_intrusion"]["rollouts"]["per_arm"]
    cells = [c for c in scan["per_cell"] if c != "answer_language"]
    rate9 = {c: scan["per_cell"][c]["intruded"] / scan["per_cell"][c]["total"] for c in cells}
    order = sorted(cells, key=lambda c: -rate9[c])
    ticks = np.arange(len(order), dtype=np.float64)

    def _wald_offsets(k: int, n: int) -> tuple[float, float]:
        """Wald 95% interval -> non-negative percent offsets (gotchas xerr rule)."""
        p = k / n
        half = 1.96 * float(np.sqrt(max(p * (1.0 - p), 0.0) / n))
        return 100.0 * min(half, p), 100.0 * min(half, 1.0 - p)

    c9 = paper_palette_role("primary")
    c7 = paper_palette_role("baseline")
    fig, ax = plt.subplots(figsize=(7.0, 4.6), layout="constrained")
    h = 0.38
    for off, side, color, lab in (
        (-h / 2, "9b", c9, DISPLAY["qwen35_9b"]),
        (+h / 2, "7b", c7, DISPLAY["qwen25_7b"] + " (parent)"),
    ):
        vals, los, his, ys = [], [], [], []
        for i, cell in enumerate(order):
            if side == "9b":
                k, n = scan["per_cell"][cell]["intruded"], scan["per_cell"][cell]["total"]
            else:
                row = parent.get(cell)
                if row is None:
                    continue  # one-word query pilot: no parent counterpart
                k, n = row["intruded"], row["total"]
            lo_off, hi_off = _wald_offsets(k, n)
            vals.append(100.0 * k / n)
            los.append(lo_off)
            his.append(hi_off)
            ys.append(float(i) + off)
        ax.barh(
            ys,
            vals,
            height=h,
            color=color,
            xerr=np.vstack([los, his]),
            error_kw={"elinewidth": 0.9},
            label=lab,
        )
    ax.set_yticks(ticks)
    ax.set_yticklabels([axis_label(c) for c in order])
    ax.invert_yaxis()
    ax.set_xlabel("rollouts containing any CJK character (%)")
    ax.set_title("Stray-CJK intrusion per battery cell", loc="left")
    ax.legend(loc="lower right")
    paths = savefig_paper(fig, "fig_intrusion_rates", dir=out_dir)
    plt.close(fig)
    return list(paths.values())


# ── registry + CLI ─────────────────────────────────────────────────────

# fig name -> (required input keys, renderer). Mechanically enumerable:
#   uv run python -c "import sys; sys.path.insert(0,'scripts');
#                     from issue2587_figures import FIGS; print(sorted(FIGS))"
FIGS: dict[str, tuple[tuple[str, ...], object]] = {
    "hero_layer_sweep": (("sweep", "ref9b_n10k", "ref7b_n10k"), fig_hero_layer_sweep),
    "matched_n_table": (("sweep", "matched7b", "delta?"), matched_n_table),
    "selected_lambda_per_layer": (("sweep",), fig_selected_lambda_per_layer),
    "floors_per_layer": (("sweep",), fig_floors_per_layer),
    "wc_transfer_per_layer": (("sweep",), fig_wc_transfer_per_layer),
    "knn_per_layer": (("sweep",), fig_knn_per_layer),
    "reliability_ceiling": (("sweep",), fig_reliability_ceiling),
    "crossmodel_axis_profile": (("crossmodel", "delta"), fig_crossmodel_axis_profile),
    "crossmodel_delta_forest": (("crossmodel",), fig_crossmodel_delta_forest),
    "matched_vs_parent_scatter": (("crossmodel",), fig_matched_vs_parent_scatter),
    "delta_norm_scatter": (("perpair",), fig_delta_norm_scatter),
    "install_swap_violins": (("perpair",), fig_install_swap_violins),
    "edit_dose_scatter": (("perpair", "delta"), fig_edit_dose_scatter),
    "carrier_direction_heatmap": (("perpair",), fig_carrier_direction_heatmap),
    "text_space_rank_scatter": (("delta", "crossmodel"), fig_text_space_rank_scatter),
    "axis_identity_heatmap": (("delta",), fig_axis_identity_heatmap),
    "crossfam_consistency_scatter": (("delta",), fig_crossfam_consistency_scatter),
    "delta_retrieval_acc": (("delta",), fig_delta_retrieval_acc),
    "splithalf_vs_direction": (("delta",), fig_splithalf_vs_direction),
    "pilot_axis_panels": (("delta",), fig_pilot_axis_panels),
    "lstar_sensitivity_twins": (("delta",), fig_lstar_sensitivity_twins),
    "pooling_twin_scatter": (("delta",), fig_pooling_twin_scatter),
    "think_leak_cap_hit_table": (("leakdir",), think_leak_cap_hit_table),
    "intrusion_rates": (("intrusion", "parent_intrusion"), fig_intrusion_rates),
    "manipulation_check_table": (("manip9b", "manip7b"), manipulation_check_table),
    "token_count_equality_table": (("bank",), token_count_equality_table),
}

# input key -> (argparse attr, description, loader kind)
_INPUT_SPECS: dict[str, tuple[str, str, str]] = {
    "sweep": ("sweep_json", "map_layer_sweep.json (unit 4 finalize)", "json"),
    "matched7b": ("matched7b_json", "matched7b_anchor.json (unit 4 P8)", "json"),
    "delta": ("delta_json", "minpair_delta_2587.json (unit 5b)", "json"),
    "crossmodel": ("crossmodel_json", "crossmodel_contrasts.json (unit 5b)", "json"),
    "ref9b_n10k": ("ref2330_9b", "banked #2330 9B n=10k fits", "json"),
    "ref7b_n10k": ("ref2330_7b", "banked #2330 7B n=10k fits", "json"),
    "perpair": ("perpair_jsonl", "perpair_2587.jsonl (unit 5b)", "jsonl"),
    "manip9b": ("manip9b_json", "manipulation_check_2587.json (unit 5a)", "json"),
    "manip7b": ("manip7b_json", "parent manipulation_check.json (#2564)", "json"),
    "bank": ("bank_json", "bank_manifest.json (unit 1 + P0b token gates)", "json"),
    "intrusion": ("intrusion_json", "intrusion_scan_2587.json (analyzer r1 CJK scan)", "json"),
    "parent_intrusion": (
        "parent_intrusion_json",
        "parent intrusion_audit.json (#2564 @ d313b856)",
        "json",
    ),
    "leakdir": (
        "leak_caphit_dir",
        "harvested dir holding anchors_*.done.json + cap_hit_*.json",
        "leakdir",
    ),
}

_LOADERS = {"json": _load_json, "jsonl": _load_jsonl, "leakdir": _load_leak_dir}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--figs", default="all", help="comma list of figure names, or 'all'")
    p.add_argument("--out-dir", type=Path, default=Path("figures/issue_2587"))
    p.add_argument(
        "--sweep-json", type=Path, default=Path("eval_results/issue_2587/map_layer_sweep.json")
    )
    p.add_argument(
        "--matched7b-json", type=Path, default=Path("eval_results/issue_2587/matched7b_anchor.json")
    )
    p.add_argument(
        "--delta-json", type=Path, default=Path("eval_results/issue_2587/minpair_delta_2587.json")
    )
    p.add_argument(
        "--crossmodel-json",
        type=Path,
        default=Path("eval_results/issue_2587/crossmodel_contrasts.json"),
    )
    p.add_argument(
        "--ref2330-9b",
        type=Path,
        default=Path("eval_results/issue_2330/matched_fits_q35_n10k.json"),
    )
    p.add_argument(
        "--ref2330-7b",
        type=Path,
        default=Path("eval_results/issue_2330/matched_fits_q25_n10k.json"),
    )
    p.add_argument(
        "--perpair-jsonl", type=Path, default=Path("eval_results/issue_2587/perpair_2587.jsonl")
    )
    p.add_argument(
        "--manip9b-json",
        type=Path,
        default=Path("eval_results/issue_2587/manipulation_check_2587.json"),
    )
    p.add_argument(
        "--manip7b-json",
        type=Path,
        default=Path("eval_results/issue_2564/manipulation_check.json"),
    )
    p.add_argument(
        "--bank-json", type=Path, default=Path("eval_results/issue_2587/bank_manifest.json")
    )
    p.add_argument(
        "--intrusion-json",
        type=Path,
        default=Path("eval_results/issue_2587/intrusion_scan_2587.json"),
    )
    p.add_argument(
        "--parent-intrusion-json",
        type=Path,
        default=Path("eval_results/issue_2564/intrusion_audit.json"),
    )
    p.add_argument("--leak-caphit-dir", type=Path, default=Path("eval_results/issue_2587"))
    p.add_argument("--style", default="neurips", choices=("neurips", "generic", "blog", "iclr"))
    p.add_argument("--import-check", action="store_true")
    return p.parse_args(argv)


def resolve_figs(spec: str) -> list[str]:
    names = sorted(FIGS) if spec == "all" else [s.strip() for s in spec.split(",") if s.strip()]
    unknown = [n for n in names if n not in FIGS]
    if unknown:
        raise SystemExit(f"unknown figure name(s): {unknown}; known: {sorted(FIGS)}")
    return names


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    if args.import_check:
        from explore_persona_space.orchestrate.argcheck import assert_args_attributes_defined

        assert_args_attributes_defined(__file__)
        print("[import-check] ok", flush=True)
        return 0
    names = resolve_figs(args.figs)
    needed: set[str] = set()
    for n in names:
        for key in FIGS[n][0]:
            needed.add(key.rstrip("?"))
    inputs: dict = {}
    for key in sorted(needed):
        flag, what, kind = _INPUT_SPECS[key]
        path = getattr(args, flag)
        optional = all(
            key not in FIGS[n][0] for n in names
        )  # key appears only as "<key>?" -> optional
        if optional and not Path(path).exists():
            logger.info("[figs] optional input %s absent (%s) — skipping", key, path)
            continue
        inputs[key] = _LOADERS[kind](path, what)
    set_paper_style(args.style)
    written: list[Path] = []
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for n in names:
        req, fn = FIGS[n]
        missing = [
            k.rstrip("?") for k in req if not k.endswith("?") and k.rstrip("?") not in inputs
        ]
        if missing:
            raise FileNotFoundError(f"figure {n}: missing required input(s) {missing}")
        out = fn(inputs, args.out_dir)
        written.extend(out)
        print(f"[figs] {n}: {len(out)} file(s)", flush=True)
    print(f"[phase=done] figures_2587 complete: {len(written)} files -> {args.out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    _rc = main()
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(_rc)
