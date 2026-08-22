"""Specialization ladder for the #2054 pooled context->answer map (Result 3).

Merges the three banked/new per-cell fit sets into ONE capacity ladder per
(cell x arm) — the minimal per-cell correction that specializes the pooled
56-cell map — and renders the recovery-fraction figures:

    r0 pooled     pooled map as-is                    (pool_rungs m0)
    r1 bias       + per-cell bias refit               (pool_rungs m1)
    r2 gain       + scalar gain alpha                 (pool_rungs scale)
    r3 rotation   + orthogonal rotation R, Procrustes (pool_rungs rot)
    r4 rank-k     + rank-k residual correction,
                  k in {1, 8, 32, 128}                (pool_specialize m2_k*;
                                                       k=1 from the k1 re-run)
    r5 own map    full per-cell ridge = banked ceiling (fraction 1.0)

All rungs are HELD-OUT fold-mean R^2 under the shared production fold map;
recovery fraction = R^2_rung / R^2_ceiling on cells with a usable banked
ceiling. Reads are aggregation-only — no fits happen here. Aux columns carry
the cloud-fit offsets (ctx/ans), rot_scale, the identity(+bias) baselines, and
the kNN-retrieval reads (euclidean + cosine, chance stated in-block).

Note on nesting: bias/gain/rotation/rank-k each extend the POOLED map
independently (rank-k builds on the bias rung, not on rotation), so the ladder
orders transformation classes by parameter count, not by strict nesting.
"""

from __future__ import annotations

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    # script mode puts scripts/ (not the repo root) on sys.path[0] (gotchas.md).
    sys.path.insert(0, str(_REPO))

import numpy as np

from explore_persona_space.orchestrate.provenance import as_metadata_dict, git_provenance

SCRIPT_VERSION = "issue2054_specialization_ladder_v1"
D_AMBIENT = 3584
ARMS = ("context", "prefix")

# Ladder rungs in presentation order -> (source set, source metric name).
RUNG_SOURCE = {
    "pooled": ("pool_rungs", "m0"),
    "bias": ("pool_rungs", "m1"),
    "gain": ("pool_rungs", "scale"),
    "rotation": ("pool_rungs", "rot"),
    "rank1": ("k1", "m2_k1"),
    "rank8": ("pool_specialize", "m2_k8"),
    "rank32": ("pool_specialize", "m2_k32"),
    "rank128": ("pool_specialize", "m2_k128"),
}
RUNG_ORDER = [*RUNG_SOURCE.keys(), "own_map"]
# Per-cell correction parameter counts (d = 3584): bias d; gain +1; rotation
# d(d-1)/2 + d; rank-k residual k(2d+1) + d (PCA dirs + ridge cols + bias);
# own map d^2 + d.
RUNG_PARAMS = {
    "pooled": 0,
    "bias": D_AMBIENT,
    "gain": D_AMBIENT + 1,
    "rotation": D_AMBIENT * (D_AMBIENT - 1) // 2 + D_AMBIENT,
    "rank1": 1 * (2 * D_AMBIENT + 1) + D_AMBIENT,
    "rank8": 8 * (2 * D_AMBIENT + 1) + D_AMBIENT,
    "rank32": 32 * (2 * D_AMBIENT + 1) + D_AMBIENT,
    "rank128": 128 * (2 * D_AMBIENT + 1) + D_AMBIENT,
    "own_map": D_AMBIENT * D_AMBIENT + D_AMBIENT,
}
AXES = ("framing", "character", "model", "provenance")
ASSISTANT_IDENTITY = "conversation_paired_stories_assistant"


def _log(msg: str) -> None:
    print(msg, flush=True)


def parse_cell_axes(cell_key: str) -> dict:
    """`identity__condition__framing__model` -> the four writeup axes.

    Transposed cells (`char_<name>_op[_base]__cell_c__chat__<model>`) carry
    provenance `transposed`; asserts on any unrecognized shape.
    """
    parts = cell_key.split("__")
    assert len(parts) == 4, f"cell key does not parse: {cell_key}"
    ident, cond, framing, model = parts
    if cond == "cell_c":
        provenance = "transposed"
        character = ident.removesuffix("_op_base").removesuffix("_op")
    else:
        assert cond in ("inserted", "on_policy"), f"unknown condition {cond!r} in {cell_key}"
        provenance = cond
        character = ident
    character = "assistant" if character == ASSISTANT_IDENTITY else character.removeprefix("char_")
    return {
        "character": character,
        "provenance": provenance,
        "framing": framing,
        "model": "instruct" if model.endswith("-instruct") else "base",
    }


def _load_units(percell_dir: Path, what: str) -> dict[tuple[str, str], dict]:
    """Load every per-cell JSON under a fit set's percell dir, keyed (cell, arm)."""
    files = sorted(percell_dir.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"no per-cell JSONs under {percell_dir} ({what})")
    out: dict[tuple[str, str], dict] = {}
    for p in files:
        d = json.loads(p.read_text(encoding="utf-8"))
        out[(d["cell"], d["arm"])] = d
    _log(f"[specladder] {what}: {len(out)} units from {percell_dir}")
    return out


def _fold_records(unit: dict) -> list[dict]:
    """Scored per-fold records (pool_rungs uses 'folds', pool_specialize 'per_fold');
    skipped folds (constant-Y) carry no 'metrics' and are excluded."""
    recs = unit.get("per_fold") or unit.get("folds") or []
    return [fr for fr in recs if "metrics" in fr]


def _fold_mean_r2(unit: dict, name: str) -> float | None:
    recs = _fold_records(unit)
    vals = [fr["metrics"][name]["r2"] for fr in recs if name in fr["metrics"]]
    return float(np.mean(vals)) if vals else None


def _fold_mean_knn(unit: dict, name: str) -> dict | None:
    """Fold-mean kNN acc@k per metric, chance carried from the fold blocks."""
    recs = _fold_records(unit)
    blocks = [fr["knn"][name] for fr in recs if name in fr.get("knn", {})]
    if not blocks:
        return None
    out: dict = {}
    for metric in ("euclidean", "cosine"):
        accs = {
            k: float(np.mean([b[metric]["acc_at_k"][k] for b in blocks]))
            for k in blocks[0][metric]["acc_at_k"]
        }
        chance = {
            k: float(np.mean([b[metric]["chance_at_k"][k] for b in blocks]))
            for k in blocks[0][metric]["chance_at_k"]
        }
        out[metric] = {"acc_at_k": accs, "chance_at_k": chance}
    return out


def build_unit_row(
    cell: str,
    arm: str,
    pr: dict,
    ps: dict,
    k1: dict | None,
) -> dict:
    """One merged ladder row for a (cell, arm) unit."""
    ceiling = ps["pooled"]["ceiling"]
    usable = bool(not ceiling.get("missing") and ceiling.get("usable"))
    ceiling_r2 = float(ceiling["ceiling_r2"]) if not ceiling.get("missing") else None

    sources = {"pool_rungs": pr, "pool_specialize": ps, "k1": k1}
    r2: dict[str, float | None] = {}
    for rung, (src_name, metric) in RUNG_SOURCE.items():
        src = sources[src_name]
        r2[rung] = _fold_mean_r2(src, metric) if src is not None else None
    r2["own_map"] = ceiling_r2

    fractions = {
        rung: (None if (v is None or not usable) else float(v / ceiling_r2))
        for rung, v in r2.items()
    }

    # Cross-source consistency on the shared rungs (fold-mean vs fold-mean).
    consist = {}
    for rung_name, ps_metric in (("pooled", "m0"), ("bias", "m1")):
        a, b = r2[rung_name], _fold_mean_r2(ps, ps_metric)
        if a is not None and b is not None:
            consist[f"abs_diff_{rung_name}_vs_pool_specialize"] = abs(a - b)
    if k1 is not None:
        for rung_name, k1_metric in (("pooled", "m0"), ("bias", "m1")):
            a, b = r2[rung_name], _fold_mean_r2(k1, k1_metric)
            if a is not None and b is not None:
                consist[f"abs_diff_{rung_name}_vs_k1_run"] = abs(a - b)

    aux_r2 = {
        "ctx_offset": _fold_mean_r2(pr, "ctx_offset"),
        "ans_offset": _fold_mean_r2(pr, "ans_offset"),
        "rot_scale": _fold_mean_r2(pr, "rot_scale"),
        "identity_cell": _fold_mean_r2(pr, "identity_cell"),
        "identity_global": _fold_mean_r2(ps, "identity_global"),
    }

    knn = {
        "pooled": _fold_mean_knn(pr, "m0"),
        "bias": _fold_mean_knn(pr, "m1"),
        "rot_scale": _fold_mean_knn(pr, "rot_scale"),
        "rank128": _fold_mean_knn(ps, "m2_k128"),
        "identity_cell": _fold_mean_knn(ps, "identity_cell"),
    }
    if k1 is not None:
        knn["rank1"] = _fold_mean_knn(k1, "m2_k1")

    pr_recs = _fold_records(pr)
    ps_recs = _fold_records(ps)
    degenerate_gain_rot = any(fr.get("degenerate_gain_rot") for fr in pr_recs)
    m2_skipped = any(fr.get("m2_skipped") for fr in ps_recs)

    return {
        "cell": cell,
        "arm": arm,
        **parse_cell_axes(cell),
        "n_join": pr["n_join"],
        "ceiling_r2": ceiling_r2,
        "ceiling_usable": usable,
        "banked_null_r2_pooled_p95": ceiling.get("banked_null_r2_pooled_p95"),
        "r2": r2,
        "fraction_of_ceiling": fractions,
        "aux_r2": aux_r2,
        "knn": knn,
        "degenerate_gain_rot": degenerate_gain_rot,
        "m2_skipped_bias_substituted": m2_skipped,
        "consistency": consist,
    }


def _quartiles(vals: list[float]) -> dict:
    a = np.asarray(vals, dtype=np.float64)
    return {
        "n": int(a.size),
        "median": float(np.median(a)),
        "q25": float(np.percentile(a, 25)),
        "q75": float(np.percentile(a, 75)),
        "min": float(a.min()),
        "max": float(a.max()),
    }


def aggregate_rows(rows: list[dict]) -> dict:
    """Per-arm rung summaries + per-axis (framing/character/model/provenance)
    recovery-fraction aggregates over cells with usable ceilings."""
    out: dict = {}
    for arm in ARMS:
        arm_rows = [r for r in rows if r["arm"] == arm]
        usable = [r for r in arm_rows if r["ceiling_usable"]]
        arm_out: dict = {
            "n_units": len(arm_rows),
            "n_ceiling_usable": len(usable),
            "per_rung": {},
            "by_axis": {},
        }
        for rung in RUNG_ORDER:
            r2s = [r["r2"][rung] for r in arm_rows if r["r2"][rung] is not None]
            fracs = [
                r["fraction_of_ceiling"][rung]
                for r in usable
                if r["fraction_of_ceiling"][rung] is not None
            ]
            entry: dict = {"params_per_cell": RUNG_PARAMS[rung]}
            if r2s:
                entry["r2"] = _quartiles(r2s)
            if fracs:
                entry["fraction_of_ceiling"] = _quartiles(fracs)
                entry["n_cells_within_90pct_of_ceiling"] = int(sum(f >= 0.90 for f in fracs))
                entry["n_cells_within_95pct_of_ceiling"] = int(sum(f >= 0.95 for f in fracs))
            arm_out["per_rung"][rung] = entry
        for axis in AXES:
            levels: dict[str, list[dict]] = defaultdict(list)
            for r in usable:
                levels[r[axis]].append(r)
            arm_out["by_axis"][axis] = {
                level: {
                    "n_cells": len(lrows),
                    "per_rung": {
                        rung: _quartiles(
                            [
                                r["fraction_of_ceiling"][rung]
                                for r in lrows
                                if r["fraction_of_ceiling"][rung] is not None
                            ]
                        )
                        for rung in RUNG_ORDER
                        if any(r["fraction_of_ceiling"][rung] is not None for r in lrows)
                    },
                }
                for level, lrows in sorted(levels.items())
            }
        out[arm] = arm_out
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Figures


FRAMING_LABEL = {
    "attrib_quoted": "attributed quote",
    "bare_label": "bare label",
    "bare_text": "bare text",
    "chat": "chat template",
}
RUNG_LABEL = {
    "pooled": "pooled",
    "bias": "+bias",
    "gain": "+gain",
    "rotation": "+rotation",
    "rank1": "+rank-1",
    "rank8": "+rank-8",
    "rank32": "+rank-32",
    "rank128": "+rank-128",
    "own_map": "own map",
}


def _rung_xs_ys(row: dict, rungs: list[str]) -> tuple[list[int], list[float]]:
    xs, ys = [], []
    for i, rung in enumerate(rungs):
        v = row["fraction_of_ceiling"][rung]
        if v is not None:
            xs.append(i)
            ys.append(v)
    return xs, ys


def render_figures(rows: list[dict], aggregates: dict, figures_dir: Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import (
        paper_palette,
        savefig_paper,
        set_paper_style,
    )

    set_paper_style()
    rungs = [
        r for r in RUNG_ORDER if any(row["fraction_of_ceiling"][r] is not None for row in rows)
    ]
    ctx = [r for r in rows if r["arm"] == "context" and r["ceiling_usable"]]
    framings = sorted({r["framing"] for r in ctx})
    colors = dict(zip(framings, paper_palette(len(framings))))
    written: list[str] = []

    def _style_axis(ax) -> None:
        ax.set_xticks(range(len(rungs)))
        ax.set_xticklabels([RUNG_LABEL[r] for r in rungs], rotation=45, ha="right")
        ax.axhline(1.0, color="0.75", lw=0.8, zorder=0)
        ax.set_ylim(-0.15, 1.2)

    # Figure 1: one line per cell, colored by framing; panels base vs instruct.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, model in zip(axes, ("base", "instruct")):
        for row in [r for r in ctx if r["model"] == model]:
            xs, ys = _rung_xs_ys(row, rungs)
            ax.plot(xs, ys, color=colors[row["framing"]], alpha=0.55, lw=1.1)
        _style_axis(ax)
        ax.set_title(f"{model} model")
    axes[0].set_ylabel("fraction of own-map ceiling (held-out $R^2$)")
    handles = [plt.Line2D([0], [0], color=colors[f], lw=2) for f in framings]
    axes[1].legend(handles, [FRAMING_LABEL[f] for f in framings], loc="lower right", fontsize=8)
    p = savefig_paper(fig, "ladder_recovery_by_model", dir=str(figures_dir))
    plt.close(fig)
    written.append(str(p))

    # Figure 2: faceted by character; solid = instruct, dashed = base.
    characters = sorted({r["character"] for r in ctx})
    fig, axes = plt.subplots(1, len(characters), figsize=(3.1 * len(characters), 3.8), sharey=True)
    for ax, character in zip(np.atleast_1d(axes), characters):
        for row in [r for r in ctx if r["character"] == character]:
            xs, ys = _rung_xs_ys(row, rungs)
            ax.plot(
                xs,
                ys,
                color=colors[row["framing"]],
                alpha=0.6,
                lw=1.1,
                ls="-" if row["model"] == "instruct" else "--",
            )
        _style_axis(ax)
        ax.set_title(character)
    np.atleast_1d(axes)[0].set_ylabel("fraction of own-map ceiling")
    handles = [plt.Line2D([0], [0], color=colors[f], lw=2) for f in framings]
    handles += [
        plt.Line2D([0], [0], color="0.3", lw=1.5, ls="-"),
        plt.Line2D([0], [0], color="0.3", lw=1.5, ls="--"),
    ]
    np.atleast_1d(axes)[-1].legend(
        handles,
        [FRAMING_LABEL[f] for f in framings] + ["instruct", "base"],
        loc="lower right",
        fontsize=7,
    )
    p = savefig_paper(fig, "ladder_recovery_by_character", dir=str(figures_dir))
    plt.close(fig)
    written.append(str(p))

    # Figure 3: aggregated views — median recovery per rung per axis level.
    fig, axes = plt.subplots(1, len(AXES), figsize=(4.0 * len(AXES), 3.8), sharey=True)
    for ax, axis in zip(axes, AXES):
        by_level = aggregates["context"]["by_axis"][axis]
        pal = paper_palette(len(by_level))
        for color, (level, entry) in zip(pal, by_level.items()):
            xs = [rungs.index(r) for r in rungs if r in entry["per_rung"]]
            ys = [entry["per_rung"][r]["median"] for r in rungs if r in entry["per_rung"]]
            label = FRAMING_LABEL.get(level, level) if axis == "framing" else level
            ax.plot(xs, ys, marker="o", ms=3, lw=1.6, color=color, label=label)
        _style_axis(ax)
        ax.set_title(f"median by {axis}")
        ax.legend(fontsize=7, loc="lower right")
    axes[0].set_ylabel("fraction of own-map ceiling (median)")
    p = savefig_paper(fig, "ladder_recovery_aggregates", dir=str(figures_dir))
    plt.close(fig)
    written.append(str(p))
    return written


# ─────────────────────────────────────────────────────────────────────────────


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--pool-rungs-dir", type=Path, required=True, help="pool_rungs percell dir")
    ap.add_argument(
        "--pool-specialize-dir", type=Path, required=True, help="pool_specialize percell dir"
    )
    ap.add_argument(
        "--k1-dir",
        type=Path,
        default=None,
        help="k=1 re-run percell dir (omit to build the ladder without the rank-1 rung)",
    )
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--figures-dir", type=Path, required=True)
    ap.add_argument("--import-check", action="store_true")
    return ap


def main() -> int:
    args = build_argparser().parse_args()
    if args.import_check:
        from explore_persona_space.orchestrate.argcheck import assert_args_attributes_defined

        assert_args_attributes_defined(__file__)
        _log("[specladder] import-check OK")
        return 0

    t0 = time.time()
    pr_units = _load_units(args.pool_rungs_dir, "pool_rungs")
    ps_units = _load_units(args.pool_specialize_dir, "pool_specialize")
    k1_units = _load_units(args.k1_dir, "k1") if args.k1_dir else {}
    if set(pr_units) != set(ps_units):
        raise RuntimeError(
            f"unit sets differ: pool_rungs {len(pr_units)} vs pool_specialize {len(ps_units)}"
        )
    if k1_units and set(k1_units) != set(pr_units):
        raise RuntimeError(f"k1 unit set differs: {len(k1_units)} vs {len(pr_units)}")

    rows = [
        build_unit_row(
            cell, arm, pr_units[(cell, arm)], ps_units[(cell, arm)], k1_units.get((cell, arm))
        )
        for (cell, arm) in sorted(pr_units)
    ]
    if not rows:
        raise RuntimeError("empty ladder — no units joined")
    aggregates = aggregate_rows(rows)

    consist_vals = [(k, v) for r in rows for k, v in r["consistency"].items()]
    consist_max: dict[str, float] = {}
    for k, v in consist_vals:
        consist_max[k] = max(consist_max.get(k, 0.0), v)
    _log(f"[specladder] cross-source consistency (max |dR^2|): {consist_max}")

    payload = {
        "metadata": {
            **as_metadata_dict(git_provenance(_REPO)),
            "script_version": SCRIPT_VERSION,
            "argv": sys.argv,
            "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "rung_order": RUNG_ORDER,
            "rung_sources": {k: list(v) for k, v in RUNG_SOURCE.items()},
            "rung_params_per_cell": RUNG_PARAMS,
            "rank1_present": bool(k1_units),
            "cross_source_consistency_max_abs_dr2": consist_max,
            "knn_note": "fold-mean acc@k; chance_at_k = k / n_pool carried per block",
        },
        "units": rows,
        "aggregates": aggregates,
    }
    args.out_root.mkdir(parents=True, exist_ok=True)
    out_path = args.out_root / "ladder.json"
    tmp = out_path.with_name(out_path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    tmp.replace(out_path)
    _log(f"[specladder] ladder -> {out_path} ({len(rows)} units)")

    args.figures_dir.mkdir(parents=True, exist_ok=True)
    for p in render_figures(rows, aggregates, args.figures_dir):
        _log(f"[specladder] figure -> {p}")
    _log(f"[specladder] done in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
