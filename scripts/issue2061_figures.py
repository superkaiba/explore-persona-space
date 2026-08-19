"""P5 figures for task #2061.

Regenerates the 6 headline figures (F1-F6) per plan §Eval + figures from
the P2/P3/P4 JSONL/JSON outputs under `eval_results/issue_2061/`. All
figures write to `figures/issue_2061/` with `.png` + `.pdf` + a
`meta.json` sidecar (commit SHA + input paths, per SPEC.md figure
provenance). Deterministic + idempotent.

Figures (plan §Eval):
- F1: Per-feature ΔR²_j vs feature id, GLOBAL null p97.5 band overlaid
      (per-cell p97.5 shown as secondary diagnostic). One figure per
      (stage-pair × arm) — 4 × 2 = 8; panels split by render.
- F2: Low-level per-cell (feature, corpus) ΔR²_j scatter behind the
      aggregate; top features labeled with feature id per cell. One
      figure per (stage-pair × arm) — 8.
- F3: Per-stage FVE / L0 / dead-feature-count (3 subplots).
- F4: kNN retrieval acc@1 / acc@k per fitted map (2 panels; one point
      per (stage, render, corpus, arm); euclidean + cosine, chance
      stated per plan §13).
- F5: Prefix-arm vs context-arm max_j ΔR²_j scatter (4 panels, one per
      stage-pair; read under the plan §Design pre-registered prefix-arm
      render-degeneracy caveat).
- F6: GLOBAL null distribution + true max_{j,cell} overlay — the
      primary headline test — with the pooled per-feature ΔR² tail
      scale + null-argmax diversity beside the band (plan §6 analyzer
      duty 2, band-vs-ceiling).

Colour convention (one colour = one meaning across the whole set):
corpus stem → colour (Wong colourblind-safe palette); render → marker
(chat 'o', naturalistic '^'); GLOBAL null p97.5 → black dashed;
per-cell null p97.5 → dotted (secondary); TRUE max → red solid.

Every figure carries a ≤ 3-sentence factual caption ("what is plotted",
per SPEC.md interim/chat writeup register + CLAUDE.md § Ad-hoc results
summaries).

Usage:
  uv run python scripts/issue2061_figures.py --all           # all 6
  uv run python scripts/issue2061_figures.py --figure f6     # single figure
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()  # shared-VM thread caps (#847) must bind BEFORE torch/numpy import

import numpy as np  # noqa: E402

# Sibling-script import (bare module name via the script-dir sys.path insert —
# same pattern as issue2061_fit_per_feature.py).
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import issue2061_turnstore as ts  # noqa: E402

# matplotlib is a per-project dep already; only import when actually plotting
# to keep --help fast + argparse smoke check cheap.


LAYER = 29
STAGES = ["base", "sft", "dpo", "rlvr", "longer-rlvr"]
STAGE_PAIRS = [
    ("base", "sft"),
    ("sft", "dpo"),
    ("dpo", "rlvr"),
    ("rlvr", "longer-rlvr"),
]
RENDERS = ["chat", "naturalistic"]
ARMS = ["prefix", "context"]

# One colour = one corpus stem across F1/F2/F4/F5 (Wong colourblind-safe).
CORPUS_COLORS = {
    "lmsys23k": "#0072B2",  # blue
    "if11k": "#E69F00",  # orange
    "math7500": "#009E73",  # green
    "gsm8k_train_full": "#CC79A7",  # reddish purple
    "gsm8k_test1319": "#D55E00",  # vermillion
}
_FALLBACK_COLORS = ["#56B4E9", "#F0E442", "#999999", "#000000"]
RENDER_MARKERS = {"chat": "o", "naturalistic": "^"}
GLOBAL_NULL_STYLE = {"color": "black", "linestyle": "--"}  # GLOBAL p97.5
TRUE_MAX_COLOR = "red"  # TRUE max_{j,cell}


def _corpus_color(corpus: str) -> str:
    if corpus in CORPUS_COLORS:
        return CORPUS_COLORS[corpus]
    # Deterministic fallback for an unexpected stem (never crash a figure).
    return _FALLBACK_COLORS[hash(corpus) % len(_FALLBACK_COLORS)]


def _render_marker(render: str) -> str:
    return RENDER_MARKERS.get(render, "s")


def _load_null_global(null_dir: Path) -> dict:
    global_path = null_dir / f"GLOBAL_L{LAYER}.json"
    if not global_path.exists():
        return {}
    with global_path.open() as f:
        return json.load(f)


def _load_per_cell_null(null_dir: Path) -> dict[tuple[str, str, str, str], dict]:
    """Load all per-cell null JSONL files. Key: (pair_str, render, corpus, arm).

    Render rides the key (review M2/M3): chat and naturalistic cells of one
    corpus stem are DISTINCT cells and must never collide on one dict entry.
    """
    cells: dict[tuple[str, str, str, str], dict] = {}
    for path in sorted(null_dir.glob(f"*_L{LAYER}.jsonl")):
        with path.open() as f:
            row = json.loads(f.readline())
        cells[(row["pair"], row["render"], row["corpus"], row["arm"])] = row
    return cells


_R2_CACHE: dict[str, dict[tuple[str, str, str, str], np.ndarray]] = {}


def _load_per_feature_r2(r2_dir: Path) -> dict[tuple[str, str, str, str], np.ndarray]:
    """Load per-feature R² arrays. Key: (stage, render, corpus, arm).

    Filenames parse from the LEFT via the shared fail-loud parser
    (`issue2061_turnstore.parse_r2_stem`, review M2): underscore corpora
    (gsm8k_train_full) stay intact, and the render token stays in the key so
    the two renders of one corpus stem never silently overwrite each other.

    Array index == feature id (asserted row-by-row against the writer's
    `feature_id` field — fail-loud on any misordering rather than silently
    mis-attributing ΔR² to the wrong feature). Cached per r2_dir: several
    figures read the same files and the production files are ~262k rows each.
    """
    cache_key = str(Path(r2_dir).resolve())
    if cache_key in _R2_CACHE:
        return _R2_CACHE[cache_key]
    r2_files: dict[tuple[str, str, str, str], np.ndarray] = {}
    for path in sorted(r2_dir.glob(f"*_L{LAYER}.jsonl")):
        stage, render, corpus, arm = ts.parse_r2_stem(path.stem, LAYER)
        r2 = []
        with path.open() as f:
            for i, line in enumerate(f):
                row = json.loads(line)
                if row["feature_id"] != i:
                    raise ValueError(
                        f"{path.name}: row {i} carries feature_id={row['feature_id']} — "
                        "file is not in feature-id order; refusing to mis-index ΔR²."
                    )
                r2.append(row["R2"] if row["R2"] is not None else np.nan)
        r2_files[(stage, render, corpus, arm)] = np.asarray(r2, dtype=np.float64)
    _R2_CACHE[cache_key] = r2_files
    return r2_files


def _load_knn_per_map(r2_dir: Path) -> dict[tuple[str, str, str, str], dict]:
    """Per-map kNN retrieval fields. Key: (stage, render, corpus, arm).

    The P2 writer duplicates the per-CELL kNN fields on every feature row
    (scripts/issue2061_fit_per_feature.py), so the FIRST row of each file
    carries the full per-map read — no need to parse 262k rows here.
    """
    out: dict[tuple[str, str, str, str], dict] = {}
    for path in sorted(r2_dir.glob(f"*_L{LAYER}.jsonl")):
        stage, render, corpus, arm = ts.parse_r2_stem(path.stem, LAYER)
        with path.open() as f:
            first = f.readline()
        if not first.strip():
            print(f"[warn] empty R² file {path.name} — skipped in F4")
            continue
        row = json.loads(first)
        out[(stage, render, corpus, arm)] = {
            k: row[k]
            for k in (
                "knn_acc_1_euclid",
                "knn_acc_k_euclid",
                "knn_acc_1_cosine",
                "knn_acc_k_cosine",
                "knn_k_ret",
                "chance_1",
                "chance_k",
            )
        }
    return out


def _true_max_per_cell(
    r2s: dict[tuple[str, str, str, str], np.ndarray],
    per_cell_null: dict[tuple[str, str, str, str], dict],
) -> dict[tuple[str, str, str, str], float]:
    """TRUE max_j ΔR²_j per (pair_str, render, corpus, arm) delta cell.

    PREFERS the persisted `true_max_delta_r2` from the P3 per-cell null rows —
    the single value the headline verdict uses (unit-C carried note:
    recomputation is a second code path that can silently disagree with the
    persisted headline). Falls back to recomputing from the P2 R² files with a
    printed [warn] only when a cell has no null row (e.g. a partial P3 run).
    """
    out: dict[tuple[str, str, str, str], float] = {}
    for stage_before, stage_after in STAGE_PAIRS:
        pair_str = f"{stage_before}_{stage_after}"
        combos = {(k[1], k[2], k[3]) for k in r2s if k[0] == stage_before}
        for render, corpus, arm in sorted(combos):
            cell_key = (pair_str, render, corpus, arm)
            row = per_cell_null.get(cell_key)
            if row is not None and row.get("true_max_delta_r2") is not None:
                out[cell_key] = float(row["true_max_delta_r2"])
                continue
            before = r2s.get((stage_before, render, corpus, arm))
            after = r2s.get((stage_after, render, corpus, arm))
            if before is None or after is None:
                continue
            delta = after - before
            delta = delta[np.isfinite(delta)]
            if delta.size == 0:
                continue
            print(
                f"[warn] no persisted true_max_delta_r2 for cell {cell_key} — "
                "recomputed from P2 R² files (fallback path)"
            )
            out[cell_key] = float(delta.max())
    return out


def _get_commit_sha() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        return "unknown"


def _write_meta(fig_path: Path, caption: str, inputs: list[str]) -> None:
    """SPEC.md figure-provenance sidecar."""
    meta = {
        "caption": caption,
        "commit_sha": _get_commit_sha(),
        "inputs": inputs,
        "generator": "scripts/issue2061_figures.py",
    }
    meta_path = fig_path.with_suffix(".meta.json")
    with meta_path.open("w") as f:
        json.dump(meta, f, indent=2)


def _save_fig(fig, path: Path, caption: str, inputs: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    _write_meta(path, caption, inputs)
    print(f"[write] {path}")


def figure_f1_delta_scatter(
    r2_dir: Path,
    null_dir: Path,
    output_dir: Path,
) -> None:
    """F1: per-feature ΔR²_j vs feature id + GLOBAL null p97.5 overlay.

    One figure per (stage-pair, arm); one PANEL per render (unit-C carried
    note: render-split labels overloaded a single legend — the render axis is
    now the panel axis, and colour carries only the corpus stem).
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    r2s = _load_per_feature_r2(r2_dir)
    global_null = _load_null_global(null_dir)
    per_cell = _load_per_cell_null(null_dir)
    global_p975 = global_null.get("global_null_quantiles", {}).get("p97.5")

    for stage_before, stage_after in STAGE_PAIRS:
        pair_str = f"{stage_before}_{stage_after}"
        for arm in ARMS:
            # Cells are (render, corpus) combos (review M2): both renders of a
            # corpus stem are DISTINCT cells and both plot, on their own panel.
            combos = sorted({(k[1], k[2]) for k in r2s if k[0] == stage_before and k[3] == arm})
            renders = [r for r in RENDERS if any(rc[0] == r for rc in combos)]
            renders += sorted({rc[0] for rc in combos} - set(renders))
            if not renders:
                continue
            fig, axes = plt.subplots(
                1, len(renders), figsize=(8 * len(renders), 5), sharey=True, squeeze=False
            )
            plotted_any = False
            for pi, render in enumerate(renders):
                ax = axes[0][pi]
                for r_key, corpus in combos:
                    if r_key != render:
                        continue
                    before = r2s.get((stage_before, render, corpus, arm))
                    after = r2s.get((stage_after, render, corpus, arm))
                    if before is None or after is None:
                        continue
                    delta = after - before
                    ax.scatter(
                        np.arange(len(delta)),
                        delta,
                        s=1,
                        alpha=0.3,
                        color=_corpus_color(corpus),
                        label=corpus,
                        rasterized=True,  # keeps the PDF export sane at d_sae points
                    )
                    plotted_any = True
                    local_q = (
                        per_cell.get((pair_str, render, corpus, arm), {})
                        .get("null_quantiles_per_cell", {})
                        .get("p97.5")
                    )
                    if local_q is not None:
                        ax.axhline(local_q, color=_corpus_color(corpus), linestyle=":", alpha=0.55)
                if global_p975 is not None:
                    ax.axhline(global_p975, **GLOBAL_NULL_STYLE)
                ax.set_xlabel("SAE feature id")
                ax.set_title(f"{render} render")
                handles, labels = ax.get_legend_handles_labels()
                if global_p975 is not None:
                    handles.append(Line2D([], [], **GLOBAL_NULL_STYLE))
                    labels.append(f"GLOBAL null p97.5={global_p975:.4f}")
                handles.append(Line2D([], [], color="gray", linestyle=":"))
                labels.append("per-cell null p97.5 (secondary; corpus colour)")
                ax.legend(handles, labels, fontsize=7, loc="upper right", markerscale=6)
            if not plotted_any:
                plt.close(fig)
                continue
            axes[0][0].set_ylabel(f"ΔR²_j ({stage_after} − {stage_before})")
            fig.suptitle(f"F1: {pair_str} / {arm} arm — per-feature ΔR²_j", y=1.02)
            path = output_dir / f"f1_delta_scatter_{pair_str}_{arm}.png"
            caption = (
                f"Per-feature ΔR²_j ({stage_after} minus {stage_before}) on the "
                f"{arm}-arm map, one point per SAE feature, one panel per render, "
                f"coloured by corpus stem. Black dashed line: GLOBAL null p97.5 "
                f"(primary headline bar). Dotted corpus-coloured lines: per-cell "
                f"null p97.5 (secondary diagnostic)."
            )
            _save_fig(fig, path, caption, [str(r2_dir), str(null_dir)])
            plt.close(fig)


def figure_f2_percell(
    r2_dir: Path,
    null_dir: Path,
    output_dir: Path,
) -> None:
    """F2: low-level per-cell ΔR²_j scatter behind the aggregate.

    One figure per (stage-pair, arm); x-axis = (render, corpus) cell; y =
    per-feature ΔR²_j (both distribution tails kept exactly, bulk stride-
    subsampled for renderability); the top features per cell are labeled with
    their feature id (plan §Eval F2: raw alongside processed).
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    r2s = _load_per_feature_r2(r2_dir)
    global_null = _load_null_global(null_dir)
    per_cell = _load_per_cell_null(null_dir)
    global_p975 = global_null.get("global_null_quantiles", {}).get("p97.5")

    n_tail = 50  # exact tail points kept per side per cell
    n_bulk = 3000  # bulk stride-subsample budget per cell
    n_label = 3  # top features labeled with feature id per cell

    for stage_before, stage_after in STAGE_PAIRS:
        pair_str = f"{stage_before}_{stage_after}"
        for arm in ARMS:
            cells = sorted({(k[1], k[2]) for k in r2s if k[0] == stage_before and k[3] == arm})
            cells = [
                (render, corpus)
                for render, corpus in cells
                if (stage_after, render, corpus, arm) in r2s
            ]
            if not cells:
                continue
            fig, ax = plt.subplots(figsize=(max(9, 1.6 * len(cells)), 5.5))
            for i, (render, corpus) in enumerate(cells):
                before = r2s[(stage_before, render, corpus, arm)]
                after = r2s[(stage_after, render, corpus, arm)]
                delta = after - before
                ids = np.nonzero(np.isfinite(delta))[0]
                if ids.size == 0:
                    continue
                vals = delta[ids]
                order = np.argsort(vals)
                k_tail = min(n_tail, ids.size)
                stride = max(1, ids.size // n_bulk)
                sel = np.unique(
                    np.concatenate([ids[order[-k_tail:]], ids[order[:k_tail]], ids[::stride]])
                )
                # Deterministic per-feature x jitter (no RNG — idempotent).
                jitter = ((sel.astype(np.uint64) * np.uint64(2654435761)) % np.uint64(2048)).astype(
                    np.float64
                ) / 2048.0 * 0.5 - 0.25
                ax.scatter(
                    i + jitter,
                    delta[sel],
                    s=3,
                    alpha=0.35,
                    color=_corpus_color(corpus),
                    marker=_render_marker(render),
                    rasterized=True,
                )
                # Label the top-N features with their feature id.
                for fid in ids[order[-min(n_label, ids.size) :]][::-1]:
                    ax.annotate(
                        str(int(fid)),
                        (i, float(delta[fid])),
                        textcoords="offset points",
                        xytext=(6, 0),
                        fontsize=6,
                    )
                local_q = (
                    per_cell.get((pair_str, render, corpus, arm), {})
                    .get("null_quantiles_per_cell", {})
                    .get("p97.5")
                )
                if local_q is not None:
                    ax.hlines(local_q, i - 0.35, i + 0.35, color="dimgray", linestyle=":")
            if global_p975 is not None:
                ax.axhline(
                    global_p975, **GLOBAL_NULL_STYLE, label=f"GLOBAL null p97.5={global_p975:.4f}"
                )
            ax.set_xticks(range(len(cells)))
            ax.set_xticklabels([f"{r}\n{c}" for r, c in cells], fontsize=7)
            ax.set_ylabel(f"ΔR²_j ({stage_after} − {stage_before})")
            ax.set_title(f"F2: {pair_str} / {arm} arm — per-cell per-feature ΔR²_j")
            handles, labels = ax.get_legend_handles_labels()
            handles.append(Line2D([], [], color="dimgray", linestyle=":"))
            labels.append("per-cell null p97.5 (secondary)")
            ax.legend(handles, labels, fontsize=7, loc="upper right")
            path = output_dir / f"f2_percell_{pair_str}_{arm}.png"
            caption = (
                f"Per-feature ΔR²_j ({stage_after} minus {stage_before}, {arm} arm) per "
                f"(render, corpus) cell; both distribution tails ({n_tail}/side) exact, "
                f"bulk stride-subsampled to ~{n_bulk} points/cell; top-{n_label} features "
                f"labeled with feature id. Black dashed: GLOBAL null p97.5; grey dotted "
                f"ticks: per-cell null p97.5 (secondary)."
            )
            _save_fig(fig, path, caption, [str(r2_dir), str(null_dir)])
            plt.close(fig)


def figure_f3_fitness(
    fitness_dir: Path,
    output_dir: Path,
) -> None:
    """F3: per-stage FVE / L0 / dead-feature-count."""
    import matplotlib.pyplot as plt

    summary_path = fitness_dir / f"summary_L{LAYER}.json"
    if not summary_path.exists():
        print(f"[skip] F3: missing {summary_path}")
        return
    with summary_path.open() as f:
        summary = json.load(f)
    per_stage = summary.get("per_stage", {})
    stages = ["base", "sft", "dpo", "rlvr", "longer-rlvr"]
    fves = [per_stage.get(s, {}).get("fve", np.nan) for s in stages]
    l0s = [per_stage.get(s, {}).get("l0_mean", np.nan) for s in stages]
    deads = [per_stage.get(s, {}).get("dead_feature_fraction", np.nan) for s in stages]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].bar(stages, fves)
    axes[0].axhline(
        summary.get("pass_bar", 0), color="green", linestyle="--", label="pass bar (0.8× base)"
    )
    axes[0].axhline(
        summary.get("hard_floor", 0), color="red", linestyle="--", label="hard floor (0.5× base)"
    )
    axes[0].set_ylabel("FVE")
    axes[0].set_title("Fraction of variance explained")
    axes[0].legend(fontsize=7)
    axes[0].tick_params(axis="x", rotation=30)

    axes[1].bar(stages, l0s)
    axes[1].axhline(32, color="green", linestyle="--", label="k=32 (TopK target)")
    axes[1].set_ylabel("L0 (mean nonzeros per row)")
    axes[1].set_title("L0 sparsity")
    axes[1].legend(fontsize=7)
    axes[1].tick_params(axis="x", rotation=30)

    axes[2].bar(stages, deads)
    axes[2].axhline(0.1, color="red", linestyle="--", label="10% dead-frac bar")
    axes[2].set_ylabel("Dead-feature fraction")
    axes[2].set_title("Dead features")
    axes[2].legend(fontsize=7)
    axes[2].tick_params(axis="x", rotation=30)

    path = output_dir / "f3_fitness.png"
    caption = (
        "Per-stage SAE-fitness diagnostics on the fixed EleutherAI/sae-llama-3.1-8b-64x "
        "dictionary (LMSYS validation slice, ~1k rows per stage). Left: FVE with pass "
        "bar (0.8× base) + hard floor (0.5× base). Centre: L0 mean, target k=32. "
        "Right: dead-feature fraction, bar at 10%."
    )
    _save_fig(fig, path, caption, [str(fitness_dir)])
    plt.close(fig)


def figure_f4_knn(
    r2_dir: Path,
    output_dir: Path,
) -> None:
    """F4: kNN retrieval acc@1 / acc@k per fitted map.

    Two panels (acc@1 | acc@k, k = ceil(n_pool/20) per plan §13); one point
    per (stage, render, corpus, arm) map, euclidean (solid) AND cosine
    (faint), connected across stages per map family so trends read directly.
    Chance = k/n_pool per cell, drawn as grey dotted per-corpus lines.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    knn = _load_knn_per_map(r2_dir)
    if not knn:
        print(f"[skip] F4: no per-map kNN fields found under {r2_dir}")
        return
    fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharey=True)
    metric_fields = [
        ("acc@1", "knn_acc_1_euclid", "knn_acc_1_cosine", "chance_1"),
        ("acc@k", "knn_acc_k_euclid", "knn_acc_k_cosine", "chance_k"),
    ]
    arm_style = {"context": "-", "prefix": "--"}
    families = sorted({(k[1], k[2], k[3]) for k in knn})
    k_rets = sorted({int(v["knn_k_ret"]) for v in knn.values()})

    for ax, (name, f_euc, f_cos, f_chance) in zip(axes, metric_fields):
        for render, corpus, arm in families:
            xs = [si for si, s in enumerate(STAGES) if (s, render, corpus, arm) in knn]
            if not xs:
                continue
            rows = [knn[(STAGES[si], render, corpus, arm)] for si in xs]
            color = _corpus_color(corpus)
            marker = _render_marker(render)
            ax.plot(
                xs,
                [r[f_euc] for r in rows],
                linestyle=arm_style.get(arm, "-"),
                color=color,
                marker=marker,
                markersize=5,
                linewidth=1.2,
            )
            ax.plot(
                xs,
                [r[f_cos] for r in rows],
                linestyle=arm_style.get(arm, "-"),
                color=color,
                marker=marker,
                markersize=4,
                linewidth=0.8,
                alpha=0.35,
            )
            ax.plot(
                xs,
                [r[f_chance] for r in rows],
                linestyle=":",
                color="gray",
                linewidth=0.8,
            )
        ax.set_xticks(range(len(STAGES)))
        ax.set_xticklabels(STAGES, rotation=30, fontsize=8)
        ax.set_title(f"kNN retrieval {name}")
        ax.set_ylim(-0.03, 1.03)
    axes[0].set_ylabel("retrieval accuracy (held-out pool)")

    corpora = sorted({c for _, c, _ in families})
    handles = [Line2D([], [], color=_corpus_color(c), linewidth=2) for c in corpora]
    labels = list(corpora)
    for render in RENDERS:
        if any(r == render for r, _, _ in families):
            handles.append(
                Line2D([], [], color="black", marker=_render_marker(render), linestyle="")
            )
            labels.append(f"{render} render")
    handles += [
        Line2D([], [], color="black", linestyle="-"),
        Line2D([], [], color="black", linestyle="--"),
        Line2D([], [], color="black", linestyle="-", alpha=0.35),
        Line2D([], [], color="gray", linestyle=":"),
    ]
    labels += [
        "context arm",
        "prefix arm",
        "cosine (faint; solid = euclidean)",
        "chance (k/n_pool)",
    ]
    axes[1].legend(handles, labels, fontsize=7, loc="upper right")

    path = output_dir / "f4_knn.png"
    caption = (
        f"kNN retrieval accuracy of each fitted map on held-out folds: acc@1 (left) and "
        f"acc@k (right, k = ceil(n_pool/20); realized k ∈ {k_rets}), one line per "
        f"(render, corpus, arm) family across stages; euclidean solid, cosine faint; "
        f"prefix arm dashed (pre-registered render-degenerate reference arm, plan "
        f"§Design). Grey dotted: chance = k/n_pool per cell."
    )
    _save_fig(fig, path, caption, [str(r2_dir)])
    plt.close(fig)


def figure_f5_arm_agreement(
    r2_dir: Path,
    null_dir: Path,
    output_dir: Path,
) -> None:
    """F5: prefix-arm vs context-arm max_j ΔR²_j per (stage-pair, corpus).

    Four panels (one per transition); one point per (render, corpus) cell.
    Values are the PERSISTED per-cell `true_max_delta_r2` (P2-recompute
    fallback with a [warn]). Read under the plan §Design pre-registered
    prefix-arm render-degeneracy caveat: near-null prefix-arm values are
    expected by construction of the render.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    r2s = _load_per_feature_r2(r2_dir)
    per_cell = _load_per_cell_null(null_dir)
    tm = _true_max_per_cell(r2s, per_cell)
    if not tm:
        print("[skip] F5: no per-cell true max values available")
        return

    fig, axes = plt.subplots(2, 2, figsize=(11, 10))
    plotted_any = False
    for pi, (stage_before, stage_after) in enumerate(STAGE_PAIRS):
        pair_str = f"{stage_before}_{stage_after}"
        ax = axes[pi // 2][pi % 2]
        pts = []
        for render, corpus in sorted({(r, c) for (p, r, c, _a) in tm if p == pair_str}):
            x = tm.get((pair_str, render, corpus, "prefix"))
            y = tm.get((pair_str, render, corpus, "context"))
            if x is None or y is None:
                continue
            pts.append((x, y))
            ax.scatter(
                x,
                y,
                s=45,
                color=_corpus_color(corpus),
                marker=_render_marker(render),
                edgecolors="black",
                linewidths=0.4,
            )
            ax.annotate(
                f"{render[:4]}/{corpus}",
                (x, y),
                textcoords="offset points",
                xytext=(5, 3),
                fontsize=5.5,
            )
        if pts:
            plotted_any = True
            lo = min(min(p) for p in pts)
            hi = max(max(p) for p in pts)
            pad = 0.08 * max(hi - lo, 1e-6)
            lims = (lo - pad, hi + pad)
            ax.plot(lims, lims, color="gray", linewidth=0.8, zorder=0)
            ax.set_xlim(lims)
            ax.set_ylim(lims)
        ax.set_xlabel("prefix-arm max_j ΔR²_j")
        ax.set_ylabel("context-arm max_j ΔR²_j")
        ax.set_title(pair_str)
    if not plotted_any:
        plt.close(fig)
        print("[skip] F5: no cell had both arms available")
        return
    corpora = sorted({c for (_p, _r, c, _a) in tm})
    handles = [Line2D([], [], color=_corpus_color(c), marker="o", linestyle="") for c in corpora]
    labels = list(corpora)
    for render in RENDERS:
        handles.append(Line2D([], [], color="black", marker=_render_marker(render), linestyle=""))
        labels.append(f"{render} render")
    handles.append(Line2D([], [], color="gray", linewidth=0.8))
    labels.append("y = x")
    fig.legend(handles, labels, fontsize=7, loc="lower center", ncol=4)
    fig.suptitle("F5: prefix-arm vs context-arm true max_j ΔR²_j per delta cell", y=1.0)
    fig.tight_layout(rect=(0, 0.05, 1, 0.98))

    path = output_dir / "f5_arm_agreement.png"
    caption = (
        "True max_j ΔR²_j per (render, corpus) delta cell, prefix arm (x) vs context "
        "arm (y), one panel per stage transition; grey line is y = x. Values are the "
        "persisted per-cell true_max_delta_r2 from the P3 null rows. The prefix arm is "
        "the pre-registered render-degenerate reference arm (plan §Design), so low "
        "prefix-arm values are expected by construction."
    )
    _save_fig(fig, path, caption, [str(r2_dir), str(null_dir)])
    plt.close(fig)


def figure_f6_global_null(
    null_dir: Path,
    r2_dir: Path,
    output_dir: Path,
) -> None:
    """F6: GLOBAL null histogram + true max_{j, cell} ΔR²_j overlay.

    The true max is the PERSISTED per-cell `true_max_delta_r2` (unit-C carried
    note; P2-recompute fallback with [warn]). Beside the band, the pooled
    per-feature ΔR² tail quantile + the null argmax-feature diversity serve the
    plan §6 analyzer duty 2 band-vs-ceiling read (rare-feature domination).
    """
    import matplotlib.pyplot as plt

    global_null = _load_null_global(null_dir)
    if not global_null:
        print(f"[skip] F6: missing GLOBAL_L{LAYER}.json")
        return
    global_max = np.asarray(global_null["global_max_per_draw"], dtype=np.float64)
    quantiles = global_null["global_null_quantiles"]

    r2s = _load_per_feature_r2(r2_dir)
    per_cell = _load_per_cell_null(null_dir)
    tm = _true_max_per_cell(r2s, per_cell)
    true_max_per_cell = list(tm.values())
    true_global_max = float(np.max(true_max_per_cell)) if true_max_per_cell else float("nan")

    # Pooled per-feature ΔR² tail quantile — the "bulk feature scale" the
    # null-of-max band is compared against (plan §6 analyzer duty 2).
    pooled: list[np.ndarray] = []
    for stage_before, stage_after in STAGE_PAIRS:
        for key, before in r2s.items():
            if key[0] != stage_before:
                continue
            after = r2s.get((stage_after, key[1], key[2], key[3]))
            if after is None:
                continue
            delta = after - before
            pooled.append(delta[np.isfinite(delta)])
    pooled_p999 = float(np.percentile(np.concatenate(pooled), 99.9)) if pooled else float("nan")

    # Null argmax-feature diversity per cell (persisted per-draw argmax ids):
    # a fraction near 1.0 means the per-draw max lands on a DIFFERENT feature
    # nearly every draw — the rare-feature-domination signature.
    diversities = []
    for row in per_cell.values():
        ids = row.get("null_argmax_feature_per_draw") or []
        if ids:
            diversities.append(len(set(ids)) / len(ids))
    med_div = float(np.median(diversities)) if diversities else float("nan")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].hist(global_max, bins=40, edgecolor="black")
    for q, v in quantiles.items():
        if q == "p97.5":
            axes[0].axvline(v, **GLOBAL_NULL_STYLE, linewidth=1.8, label=f"{q} = {v:.4f}")
        else:
            axes[0].axvline(v, color="gray", linestyle="--", linewidth=0.9, label=f"{q} = {v:.4f}")
    if not np.isnan(true_global_max):
        axes[0].axvline(
            true_global_max,
            color=TRUE_MAX_COLOR,
            linewidth=2,
            label=f"TRUE max_{{j,cell}} = {true_global_max:.4f}",
        )
    if np.isfinite(pooled_p999):
        axes[0].axvline(
            pooled_p999,
            color="#444444",
            linestyle="-.",
            linewidth=1.2,
            label=f"pooled per-feature ΔR² p99.9 = {pooled_p999:.4f}",
        )
    axes[0].set_xlabel("max_{j, cell} ΔR²_j (per draw)")
    axes[0].set_ylabel("count")
    axes[0].set_title("GLOBAL null distribution — primary headline test")
    axes[0].legend(fontsize=8)

    axes[1].set_title("Per-cell TRUE max_j ΔR²_j across cells (secondary)")
    if true_max_per_cell:
        axes[1].hist(true_max_per_cell, bins=20, edgecolor="black", alpha=0.7)
    axes[1].axvline(
        quantiles["p97.5"],
        **GLOBAL_NULL_STYLE,
        linewidth=1.8,
        label=f"GLOBAL p97.5 = {quantiles['p97.5']:.4f}",
    )
    if np.isfinite(med_div):
        axes[1].annotate(
            f"null argmax diversity: median {med_div:.0%} distinct\n"
            f"features across {global_null.get('n_draws', '?')} draws/cell\n"
            f"(near 100% ⇒ max rides rare unstable features)",
            xy=(0.02, 0.98),
            xycoords="axes fraction",
            va="top",
            fontsize=7,
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
        )
    axes[1].set_xlabel("per-cell true max_j ΔR²_j")
    axes[1].set_ylabel("count")
    axes[1].legend(fontsize=8)

    path = output_dir / "f6_global_null.png"
    caption = (
        f"Left: GLOBAL null distribution of max_{{j,cell}} ΔR²_j across "
        f"{global_null['n_draws']} synchronized draws over {global_null['n_cells']} "
        f"(stage-pair × render × corpus × arm) delta cells; red line: true "
        f"max_{{j,cell}} ΔR²_j = {true_global_max:.4f} (persisted per-cell values) — "
        f"the primary headline test; grey dash-dot: pooled per-feature ΔR² p99.9 "
        f"(bulk-tail scale, band-vs-ceiling read). Right: per-cell true max_j ΔR²_j "
        f"with the null argmax-feature diversity annotated."
    )
    _save_fig(fig, path, caption, [str(null_dir), str(r2_dir)])
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--figure",
        type=str,
        default=None,
        choices=["f1", "f2", "f3", "f4", "f5", "f6"],
        help="Render one figure (default: f6; --all renders all six)",
    )
    parser.add_argument("--all", action="store_true")
    parser.add_argument(
        "--r2-dir", type=Path, default=Path("eval_results/issue_2061/per_feature_r2")
    )
    parser.add_argument("--null-dir", type=Path, default=Path("eval_results/issue_2061/null"))
    parser.add_argument("--fitness-dir", type=Path, default=Path("eval_results/issue_2061/fitness"))
    parser.add_argument("--output-dir", type=Path, default=Path("figures/issue_2061"))
    parser.add_argument(
        "--stage-from-hub",
        action="store_true",
        help="VM-local P5 fetch (plan §9 off_pod_phases): stage the P2/P3/P4 "
        "outputs from the HF data repo and read them from the staged mirrors, "
        "overriding --r2-dir/--null-dir/--fitness-dir.",
    )
    parser.add_argument(
        "--staging-dir",
        type=Path,
        default=Path("data/issue_2061/hf_dl"),
        help="Hub staging root for --stage-from-hub (cleaned at Step 8).",
    )
    args = parser.parse_args()

    if args.stage_from_hub:
        import issue2061_hub_io as hio  # sibling import (script-dir sys.path insert)

        args.r2_dir = hio.stage_dir("per-feature-r2", args.staging_dir)
        args.null_dir = hio.stage_dir("null", args.staging_dir)
        args.fitness_dir = hio.stage_dir("fitness", args.staging_dir)
        print(
            f"[stage] figure inputs staged from hub: r2={args.r2_dir} "
            f"null={args.null_dir} fitness={args.fitness_dir}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    which = (
        [args.figure]
        if args.figure
        else (["f1", "f2", "f3", "f4", "f5", "f6"] if args.all else ["f6"])
    )
    print(f"[setup] Rendering {which}")

    for fig_id in which:
        if fig_id == "f1":
            figure_f1_delta_scatter(args.r2_dir, args.null_dir, args.output_dir)
        elif fig_id == "f2":
            figure_f2_percell(args.r2_dir, args.null_dir, args.output_dir)
        elif fig_id == "f3":
            figure_f3_fitness(args.fitness_dir, args.output_dir)
        elif fig_id == "f4":
            figure_f4_knn(args.r2_dir, args.output_dir)
        elif fig_id == "f5":
            figure_f5_arm_agreement(args.r2_dir, args.null_dir, args.output_dir)
        elif fig_id == "f6":
            figure_f6_global_null(args.null_dir, args.r2_dir, args.output_dir)

    print("[done]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
