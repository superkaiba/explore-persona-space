"""P5 figures for task #2061.

Regenerates the 6 headline figures (F1-F6) per plan §Eval + figures from
the P2/P3/P4 JSONL/JSON outputs under `eval_results/issue_2061/`. All
figures write to `figures/issue_2061/` with `.png` + `.pdf` + a
`meta.json` sidecar (commit SHA + input paths, per SPEC.md figure
provenance). Deterministic + idempotent.

Figures (plan §Eval):
- F1: Per-feature ΔR²_j vs feature id, GLOBAL null p97.5 band overlaid
      (per-cell p97.5 shown as secondary diagnostic). One per
      (stage-pair × arm) — 4 × 2 = 8 panels.
- F2: Low-level per-(feature, corpus) ΔR²_j scatter behind aggregate.
- F3: Per-stage FVE / L0 / dead-feature-count (3 subplots).
- F4: kNN retrieval acc@1 / acc@10 per fitted map (2 panels).
- F5: Prefix-arm vs context-arm max_j ΔR²_j scatter (per stage-pair).
- F6: GLOBAL null distribution + true max_{j,cell} overlay — the
      primary headline test.

Every figure carries a ≤ 3-sentence factual caption ("what is plotted",
per SPEC.md interim/chat writeup register + CLAUDE.md § Ad-hoc results
summaries). Colorblind-safe palette per `paper-plots` skill conventions.

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

import numpy as np

# matplotlib is a per-project dep already; only import when actually plotting
# to keep --help fast + argparse smoke check cheap.


LAYER = 29
STAGE_PAIRS = [
    ("base", "sft"),
    ("sft", "dpo"),
    ("dpo", "rlvr"),
    ("rlvr", "longer-rlvr"),
]


def _load_null_global(null_dir: Path) -> dict:
    global_path = null_dir / f"GLOBAL_L{LAYER}.json"
    if not global_path.exists():
        return {}
    with global_path.open() as f:
        return json.load(f)


def _load_per_cell_null(null_dir: Path) -> dict[tuple[str, str, str], dict]:
    """Load all per-cell null JSONL files. Key: (pair_str, corpus, arm)."""
    cells: dict[tuple[str, str, str], dict] = {}
    for path in sorted(null_dir.glob(f"*_L{LAYER}.jsonl")):
        with path.open() as f:
            row = json.loads(f.readline())
        cells[(row["pair"], row["corpus"], row["arm"])] = row
    return cells


def _load_per_feature_r2(r2_dir: Path) -> dict[tuple[str, str, str], np.ndarray]:
    """Load per-feature R² arrays. Key: (stage, corpus, arm)."""
    r2_files: dict[tuple[str, str, str], np.ndarray] = {}
    for path in sorted(r2_dir.glob(f"*_L{LAYER}.jsonl")):
        # <stage>_<render>_<corpus>_<arm>_L<layer>.jsonl OR
        # <stage>_<corpus>_<arm>_L<layer>.jsonl
        parts = path.stem.rsplit("_", 3)
        if len(parts) < 3:
            continue
        stage = parts[0].split("_")[0]  # first token
        arm = parts[-2]
        corpus = parts[-3]
        r2 = []
        with path.open() as f:
            for line in f:
                row = json.loads(line)
                r2.append(row["R2"] if row["R2"] is not None else np.nan)
        r2_files[(stage, corpus, arm)] = np.asarray(r2, dtype=np.float64)
    return r2_files


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
    """F1: per-feature ΔR²_j vs feature id + GLOBAL null p97.5 band overlay."""
    import matplotlib.pyplot as plt

    r2s = _load_per_feature_r2(r2_dir)
    global_null = _load_null_global(null_dir)
    per_cell = _load_per_cell_null(null_dir)
    global_p975 = global_null.get("global_null_quantiles", {}).get("p97.5")

    for stage_before, stage_after in STAGE_PAIRS:
        pair_str = f"{stage_before}_{stage_after}"
        for arm in ["prefix", "context"]:
            cells = [k for k in r2s if k[0] == stage_before and k[2] == arm]
            if not cells:
                continue
            fig, ax = plt.subplots(figsize=(9, 5))
            for stage_key, corpus, arm_key in cells:
                before = r2s.get((stage_before, corpus, arm))
                after = r2s.get((stage_after, corpus, arm))
                if before is None or after is None:
                    continue
                delta = after - before
                ax.scatter(np.arange(len(delta)), delta, s=1, alpha=0.3, label=corpus)
            if global_p975 is not None:
                ax.axhline(
                    global_p975,
                    color="red",
                    linestyle="--",
                    label=f"GLOBAL null p97.5={global_p975:.4f}",
                )
            local_p = per_cell.get((pair_str, cells[0][1], arm), {}) if cells else {}
            local_q = local_p.get("null_quantiles_per_cell", {}).get("p97.5")
            if local_q is not None:
                ax.axhline(
                    local_q,
                    color="orange",
                    linestyle=":",
                    alpha=0.6,
                    label=f"per-cell null p97.5={local_q:.4f} (secondary)",
                )
            ax.set_xlabel("SAE feature id")
            ax.set_ylabel(f"ΔR²_j ({stage_after} − {stage_before})")
            ax.set_title(f"F1: {pair_str} / {arm} arm — per-feature ΔR²_j")
            ax.legend(fontsize=7, loc="upper right")
            path = output_dir / f"f1_delta_scatter_{pair_str}_{arm}.png"
            caption = (
                f"Per-feature ΔR²_j ({stage_after} minus {stage_before}) on the "
                f"{arm}-arm map, points per SAE feature (n=d_sae) coloured by corpus. "
                f"Red dashed line: GLOBAL null p97.5 (primary headline bar). "
                f"Orange dotted line: per-cell p97.5 (secondary diagnostic)."
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


def figure_f6_global_null(
    null_dir: Path,
    r2_dir: Path,
    output_dir: Path,
) -> None:
    """F6: GLOBAL null histogram + true max_{j, cell} ΔR²_j overlay."""
    import matplotlib.pyplot as plt

    global_null = _load_null_global(null_dir)
    if not global_null:
        print(f"[skip] F6: missing GLOBAL_L{LAYER}.json")
        return
    global_max = np.asarray(global_null["global_max_per_draw"], dtype=np.float64)
    quantiles = global_null["global_null_quantiles"]

    # Compute true max_{j, cell} ΔR²_j across all loaded cells.
    r2s = _load_per_feature_r2(r2_dir)
    true_max_per_cell: list[float] = []
    for stage_before, stage_after in STAGE_PAIRS:
        for arm in ["prefix", "context"]:
            corpora = {k[1] for k in r2s if k[0] == stage_before and k[2] == arm}
            for corpus in corpora:
                before = r2s.get((stage_before, corpus, arm))
                after = r2s.get((stage_after, corpus, arm))
                if before is None or after is None:
                    continue
                delta = after - before
                delta = delta[~np.isnan(delta)]
                if delta.size > 0:
                    true_max_per_cell.append(float(delta.max()))
    true_global_max = float(np.max(true_max_per_cell)) if true_max_per_cell else float("nan")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].hist(global_max, bins=40, edgecolor="black")
    for q, v in quantiles.items():
        axes[0].axvline(v, linestyle="--", label=f"{q} = {v:.4f}")
    if not np.isnan(true_global_max):
        axes[0].axvline(
            true_global_max,
            color="red",
            linewidth=2,
            label=f"TRUE max_{{j,cell}} = {true_global_max:.4f}",
        )
    axes[0].set_xlabel("max_{j, cell} ΔR²_j (per draw)")
    axes[0].set_ylabel("count")
    axes[0].set_title("GLOBAL null distribution — primary headline test")
    axes[0].legend(fontsize=8)

    # Split by arm.
    axes[1].set_title("Per-cell max_j ΔR²_j across cells (secondary)")
    axes[1].hist(true_max_per_cell, bins=20, edgecolor="black", alpha=0.7)
    axes[1].axvline(
        quantiles["p97.5"],
        color="orange",
        linestyle="--",
        label=f"GLOBAL p97.5 = {quantiles['p97.5']:.4f}",
    )
    axes[1].set_xlabel("per-cell true max_j ΔR²_j")
    axes[1].set_ylabel("count")
    axes[1].legend(fontsize=8)

    path = output_dir / "f6_global_null.png"
    caption = (
        f"Left: GLOBAL null distribution of max_{{j,cell}} ΔR²_j across "
        f"{global_null['n_draws']} synchronized draws over {global_null['n_cells']} "
        f"(stage-pair × corpus × arm) delta cells. Red line: true "
        f"max_{{j,cell}} ΔR²_j = {true_global_max:.4f} — the primary "
        f"headline test. Right: distribution of per-cell true max_j ΔR²_j."
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
        help="Render one figure (default: --all)",
    )
    parser.add_argument("--all", action="store_true")
    parser.add_argument(
        "--r2-dir", type=Path, default=Path("eval_results/issue_2061/per_feature_r2")
    )
    parser.add_argument("--null-dir", type=Path, default=Path("eval_results/issue_2061/null"))
    parser.add_argument("--fitness-dir", type=Path, default=Path("eval_results/issue_2061/fitness"))
    parser.add_argument("--output-dir", type=Path, default=Path("figures/issue_2061"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    which = (
        [args.figure]
        if args.figure
        else (
            ["f1", "f3", "f6"] if args.all else ["f6"]  # F2/F4/F5 are stub-eligible for round 2
        )
    )
    print(f"[setup] Rendering {which}")

    for fig_id in which:
        if fig_id == "f1":
            figure_f1_delta_scatter(args.r2_dir, args.null_dir, args.output_dir)
        elif fig_id == "f3":
            figure_f3_fitness(args.fitness_dir, args.output_dir)
        elif fig_id == "f6":
            figure_f6_global_null(args.null_dir, args.r2_dir, args.output_dir)
        else:
            print(f"[TODO] {fig_id} not implemented in this round (F2/F4/F5 pending)")

    print("[done]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
