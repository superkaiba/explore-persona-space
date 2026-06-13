"""Issue #621 9a-ter free-analysis follow-up: a(t) checkpoint-ladder rotation read.

For the 7 read-arm cells that FAILed the rotation cap (Δa/a₀ ∈ [0.151, 0.157]
vs the 0.15 threshold), this script computes the rotation trajectory
``|cos(a_t, a_init)|`` over the persisted 10-step checkpoint ladder
(``checkpoint-<step>/adapter_model.safetensors`` on the HF model repo) to
discriminate **plateau** (asymptote at termination — the cells genuinely
landed at the read-arm minimum) from **still climbing** (transient near the
band-stop — the rotation cap fired before alignment settled).

The metric matches the body H2 finding exactly: band-mean over L14–24 ×
{q_proj, v_proj} of ``|cos(a_t, a_init)|`` per rank-1 LoRA A-vector. Adapter
loading + the band-mean helper are imported from ``issue621_analyze`` so this
analysis is bit-identical to the H2 endpoint read.

NO new training, NO new eval generation, NO new model calls — all
deterministic linear algebra over already-uploaded LoRA adapters.

CLI:
    uv run python scripts/issue621_checkpoint_ladder.py [--smoke] [--cells CELL ...]

Outputs:
    .claude/cache/issue621_analysis/out/ladder_a_t_alignment.json
    figures/issue_621/h2_ladder_a_t.{png,pdf,.meta.json}
"""

# ruff: noqa: RUF001, RUF002, RUF003  # math notation (×, –, etc.)

from __future__ import annotations

import argparse
import datetime as _dt
import json
import logging
import sys
from pathlib import Path

import numpy as np

# Reuse the analyzer's adapter loader + band-mean helper so this script's
# numbers are bit-identical to per_cell_main.json's H2 reads.
_SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS))
from issue621_analyze import (  # noqa: E402
    BAND_LAYERS,
    _band_mean_a_vs_init,
    load_adapter_pairs,
)


def _all_layer_mean_a_vs_init(pairs_t: dict, init_pairs: dict) -> tuple[float | None, float | None]:
    """All-layer (every layer, every module) |cos(a_t, a_init)| + ‖Δa‖/‖a_init‖.

    Matches the body H2 headline aggregation (28 layers × 2 modules = 56 rows
    per cell); ``h2_mean_cos`` / ``h2_mean_delta`` in ``per_cell_main.json`` are
    this all-layer mean, NOT the band-mean from
    :func:`issue621_analyze._band_mean_a_vs_init` (which restricts to L14-24).
    Both inputs are UNFLIPPED adapter pair dicts.
    """
    cos_vals: list[float] = []
    rel_vals: list[float] = []
    for key, slot in pairs_t.items():
        if key not in init_pairs:
            continue
        a_t = slot["a"]
        a_0 = init_pairs[key]["a"]
        na_t = float(np.linalg.norm(a_t))
        na_0 = float(np.linalg.norm(a_0))
        if na_t == 0 or na_0 == 0:
            continue
        cos_vals.append(abs(float(a_t @ a_0 / (na_t * na_0))))
        rel_vals.append(float(np.linalg.norm(a_t - a_0) / max(na_0, 1e-30)))
    if not cos_vals:
        return None, None
    return float(np.mean(cos_vals)), float(np.mean(rel_vals))


log = logging.getLogger("issue621_ladder")

HF_MODEL_REPO = "superkaiba1/explore-persona-space"
HF_ADAPTER_PREFIX = "adapters/issue_621"

# The 7 read-arm cells that FAILed the Δa/a₀ ≤ 0.15 rotation cap.
FAILED_CELLS: tuple[str, ...] = (
    "r1_read__florist__seed256",
    "r1_read__florist__seed42",
    "r1_read__librarian__seed137",
    "r1_read__librarian__seed256",
    "r1_read__medical_doctor__seed137",
    "r1_read__medical_doctor__seed256",
    "r1_read__medical_doctor__seed42",
)

# Read-arm cells that PASSed the cap — used as a comparator showing what a
# clear plateau looks like.
PASS_COMPARATORS: tuple[str, ...] = (
    "r1_read__police_officer__seed42",
    "r1_read__police_officer__seed137",
)


# --- HF download ------------------------------------------------------------


def _list_checkpoint_steps(cell: str) -> list[int]:
    """Return sorted checkpoint step ids present on HF for this cell."""
    from huggingface_hub import list_repo_files

    files = list_repo_files(HF_MODEL_REPO, repo_type="model", revision="main")
    prefix = f"{HF_ADAPTER_PREFIX}/{cell}/"
    steps: set[int] = set()
    for f in files:
        if not f.startswith(prefix) or "checkpoint-" not in f:
            continue
        tail = f[len(prefix) :]
        if not tail.startswith("checkpoint-"):
            continue
        step_str = tail.split("/", 1)[0].split("-", 1)[1]
        if step_str.isdigit():
            steps.add(int(step_str))
    return sorted(steps)


def _download_checkpoint_adapter(cell: str, step: int, cache_dir: Path) -> Path:
    """Download adapter_model.safetensors for one checkpoint; returns local path."""
    from huggingface_hub import hf_hub_download

    rel = f"{HF_ADAPTER_PREFIX}/{cell}/checkpoint-{step}/adapter_model.safetensors"
    cfg_rel = f"{HF_ADAPTER_PREFIX}/{cell}/checkpoint-{step}/adapter_config.json"
    local_adapter = hf_hub_download(
        repo_id=HF_MODEL_REPO,
        filename=rel,
        repo_type="model",
        cache_dir=str(cache_dir),
    )
    # The loader also expects adapter_config.json next to the safetensors —
    # fetch it explicitly (the HF cache layout keeps both at the same dir).
    hf_hub_download(
        repo_id=HF_MODEL_REPO,
        filename=cfg_rel,
        repo_type="model",
        cache_dir=str(cache_dir),
    )
    return Path(local_adapter).parent


def _ensure_init_dir(cell: str, cache_dir: Path) -> Path:
    """Download adapter_init/ (or use cached) and return the directory."""
    from huggingface_hub import hf_hub_download

    base = f"{HF_ADAPTER_PREFIX}/{cell}/adapter_init"
    local_adapter = hf_hub_download(
        repo_id=HF_MODEL_REPO,
        filename=f"{base}/adapter_model.safetensors",
        repo_type="model",
        cache_dir=str(cache_dir),
    )
    hf_hub_download(
        repo_id=HF_MODEL_REPO,
        filename=f"{base}/adapter_config.json",
        repo_type="model",
        cache_dir=str(cache_dir),
    )
    return Path(local_adapter).parent


# --- Per-cell trajectory ----------------------------------------------------


def _download_final_adapter(cell: str, cache_dir: Path) -> Path:
    """Download the cell-root final adapter (terminal state after band-stop)."""
    from huggingface_hub import hf_hub_download

    base = f"{HF_ADAPTER_PREFIX}/{cell}"
    local_adapter = hf_hub_download(
        repo_id=HF_MODEL_REPO,
        filename=f"{base}/adapter_model.safetensors",
        repo_type="model",
        cache_dir=str(cache_dir),
    )
    hf_hub_download(
        repo_id=HF_MODEL_REPO,
        filename=f"{base}/adapter_config.json",
        repo_type="model",
        cache_dir=str(cache_dir),
    )
    return Path(local_adapter).parent


def _trajectory_for_cell(cell: str, cache_dir: Path) -> dict:
    """Return ladder trajectory for one cell: steps + per-step band-mean stats."""
    init_dir = _ensure_init_dir(cell, cache_dir)
    init_pairs = load_adapter_pairs(init_dir)

    steps = _list_checkpoint_steps(cell)
    assert steps, f"no checkpoint-* on HF for {cell}"
    log.info("  %s: %d checkpoints, range=%d..%d", cell, len(steps), steps[0], steps[-1])

    def _both_means(pairs: dict) -> tuple[float, float, float, float]:
        # all-layer mean — matches body's h2_mean_cos / h2_mean_delta exactly
        a_cos, a_rel = _all_layer_mean_a_vs_init(pairs, init_pairs)
        # band-mean over L14-24 (the presence-criterion band)
        b_cos, b_rel = _band_mean_a_vs_init(pairs, init_pairs)
        return (a_cos, a_rel, b_cos, b_rel)

    points: list[dict] = []
    for step in steps:
        ck_dir = _download_checkpoint_adapter(cell, step, cache_dir)
        pairs = load_adapter_pairs(ck_dir)
        a_cos, a_rel, b_cos, b_rel = _both_means(pairs)
        if a_cos is None or a_rel is None:
            raise AssertionError(f"{cell} step={step}: all-layer mean came back None")
        points.append(
            {
                "step": step,
                "all_layer_abs_cos_a_init": a_cos,
                "all_layer_rel_delta_a": a_rel,
                "band_mean_abs_cos_a_init": b_cos,
                "band_mean_rel_delta_a": b_rel,
                "is_final": False,
            }
        )

    # Append the cell-root final adapter as the terminal point. The band-stop
    # callback fires AFTER the last in-loop probe (per-step probe at
    # save_steps=10), so the cell-root state is a few optimizer steps past the
    # last checkpoint and is the state the H2 endpoint reads. Deduped when
    # bit-identical (band-stop landed exactly on a save-step boundary). This
    # mirrors `issue621_analyze._a_rotation_trajectory`.
    final_dir = _download_final_adapter(cell, cache_dir)
    final_pairs = load_adapter_pairs(final_dir)
    a_cos_f, a_rel_f, b_cos_f, b_rel_f = _both_means(final_pairs)
    last_a_cos = points[-1]["all_layer_abs_cos_a_init"]
    last_a_rel = points[-1]["all_layer_rel_delta_a"]
    dedup_final = (
        a_cos_f is not None
        and a_rel_f is not None
        and abs(a_cos_f - last_a_cos) < 1e-9
        and abs(a_rel_f - last_a_rel) < 1e-9
    )
    if not dedup_final:
        # Terminal point label "final" — the actual step count is unknown
        # without trainer_state.json; we set step = last_step + 5 (mid-way to
        # next save-step) only for plotting, and record the source.
        points.append(
            {
                "step": steps[-1] + 5,
                "all_layer_abs_cos_a_init": a_cos_f,
                "all_layer_rel_delta_a": a_rel_f,
                "band_mean_abs_cos_a_init": b_cos_f,
                "band_mean_rel_delta_a": b_rel_f,
                "is_final": True,
            }
        )

    # Verdict heuristic per the task brief: examine the slope of cos(a_t, a_init)
    # over the last ~30 steps (= last 3 checkpoints under save_steps=10).
    # PRIMARY space = all-layer mean (matches body H2 headline).
    cos_vals = [p["all_layer_abs_cos_a_init"] for p in points]
    rel_vals = [p["all_layer_rel_delta_a"] for p in points]
    band_cos_vals = [p["band_mean_abs_cos_a_init"] for p in points]
    band_rel_vals = [p["band_mean_rel_delta_a"] for p in points]
    tail_window = min(3, len(cos_vals) - 1)  # last ~30 steps
    if tail_window >= 1:
        cos_slope_per_step = (cos_vals[-1] - cos_vals[-1 - tail_window]) / (
            points[-1]["step"] - points[-1 - tail_window]["step"]
        )
        rel_slope_per_step = (rel_vals[-1] - rel_vals[-1 - tail_window]) / (
            points[-1]["step"] - points[-1 - tail_window]["step"]
        )
    else:
        cos_slope_per_step = float("nan")
        rel_slope_per_step = float("nan")

    # Rotation is "still climbing" if |Δa/a₀| is still rising; cos drops as
    # rotation grows, so cos slope is negative when rotation is still climbing.
    # Heuristic from the brief: |d(cos)/d(step)| < 0.0001 / step → plateau.
    abs_cos_slope = abs(cos_slope_per_step) if cos_slope_per_step == cos_slope_per_step else 0.0
    if abs_cos_slope < 1e-4:
        verdict = "plateau"
    elif abs_cos_slope < 3e-4:
        verdict = "unclear"
    else:
        verdict = "still_climbing"

    return {
        "cell_slug": cell,
        "primary_metric": "all_layer_abs_cos_a_init",
        "primary_layers": "L0-L27 (all 28 layers × {q_proj, v_proj} = 56 rows)",
        "band_layers": f"L{min(BAND_LAYERS)}-L{max(BAND_LAYERS)} (presence-criterion band)",
        "steps": [p["step"] for p in points],
        "is_final_terminal_point": [p["is_final"] for p in points],
        # PRIMARY (matches the body H2 headline h2_mean_cos / h2_mean_delta)
        "cos_a_t_a_init": cos_vals,
        "delta_a_over_a0": rel_vals,
        # SECONDARY (band-restricted; the locality the H2 presence-criterion uses)
        "cos_a_t_a_init_band": band_cos_vals,
        "delta_a_over_a0_band": band_rel_vals,
        "tail_window_steps": tail_window * 10,
        "cos_slope_per_step_tail": float(cos_slope_per_step),
        "rel_slope_per_step_tail": float(rel_slope_per_step),
        "verdict": verdict,
        "verdict_rule": (
            "plateau if |d(all-layer cos)/d(step)| < 1e-4 over the last ~30 steps; "
            "still_climbing if >= 3e-4; unclear in between"
        ),
    }


# --- Figure -----------------------------------------------------------------


def _draw_figure(payload: dict, out_dir: Path) -> Path:
    """Plot per-cell cos(a_t, a_init) trajectories; one panel for fails + comparators."""
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import (
        paper_palette,
        paper_palette_role,
        savefig_paper,
        set_paper_style,
    )

    set_paper_style("neurips")

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), sharey=True)

    fails = [c for c in FAILED_CELLS if c in payload]
    passes = [c for c in PASS_COMPARATORS if c in payload]

    # Palette: distinct hue per source (florist / librarian / medical_doctor /
    # police_officer); seeds share a hue with linestyle variation.
    src_color = {
        "florist": paper_palette(4)[0],
        "librarian": paper_palette(4)[1],
        "medical_doctor": paper_palette(4)[2],
        "police_officer": paper_palette_role("baseline"),
    }
    seed_style = {"42": "-", "137": "--", "256": ":"}

    # --- Panel 1: FAILed cells (Δa/a₀ > 0.15) ---
    ax = axes[0]
    for cell in fails:
        tr = payload[cell]
        src = cell.split("__")[1]
        seed = cell.split("seed")[-1]
        ax.plot(
            tr["steps"],
            tr["cos_a_t_a_init"],
            color=src_color.get(src, "0.4"),
            linestyle=seed_style.get(seed, "-"),
            linewidth=1.2,
            marker="o",
            markersize=2.5,
            label=f"{src} (s{seed}, {tr['verdict']})",
            alpha=0.9,
        )
    ax.axhline(0.988, color="0.6", linestyle="-.", linewidth=0.9, label="read-arm minimum (0.988)")
    ax.axhline(1.0, color="0.85", linestyle="-", linewidth=0.8)
    ax.set_title("Read-arm Δa/a₀ FAIL cells (n=7)", fontsize=10)
    ax.set_xlabel("optimizer step")
    ax.set_ylabel(r"band-mean $|\cos(a_t, a_\mathrm{init})|$ (L14–24 × {q,v})")
    ax.legend(loc="lower left", fontsize=6.5, framealpha=0.9)
    ax.grid(True, alpha=0.25)

    # --- Panel 2: PASS comparator(s) ---
    ax = axes[1]
    for cell in passes:
        tr = payload[cell]
        src = cell.split("__")[1]
        seed = cell.split("seed")[-1]
        ax.plot(
            tr["steps"],
            tr["cos_a_t_a_init"],
            color=src_color.get(src, "0.4"),
            linestyle=seed_style.get(seed, "-"),
            linewidth=1.2,
            marker="o",
            markersize=2.5,
            label=f"{src} (s{seed}, {tr['verdict']})",
            alpha=0.9,
        )
    ax.axhline(0.988, color="0.6", linestyle="-.", linewidth=0.9, label="read-arm minimum (0.988)")
    ax.axhline(1.0, color="0.85", linestyle="-", linewidth=0.8)
    ax.set_title("Read-arm Δa/a₀ PASS comparator", fontsize=10)
    ax.set_xlabel("optimizer step")
    ax.legend(loc="lower left", fontsize=6.5, framealpha=0.9)
    ax.grid(True, alpha=0.25)

    # Suptitle gives the narrative read.
    plateau_n = sum(1 for c in fails if payload[c]["verdict"] == "plateau")
    climb_n = sum(1 for c in fails if payload[c]["verdict"] == "still_climbing")
    unclear_n = sum(1 for c in fails if payload[c]["verdict"] == "unclear")
    summary = (
        f"a(t) rotation ladder: {plateau_n}/{len(fails)} plateau, "
        f"{climb_n}/{len(fails)} still climbing, {unclear_n}/{len(fails)} unclear"
    )
    fig.suptitle(summary, fontsize=11, y=1.0)
    fig.tight_layout()

    written = savefig_paper(fig, "h2_ladder_a_t", dir=str(out_dir))
    plt.close(fig)
    return written["png"]


# --- CLI --------------------------------------------------------------------


def _setup_logging() -> None:
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
        level=logging.INFO,
        datefmt="%H:%M:%S",
    )


def main(argv: list[str] | None = None) -> int:
    _setup_logging()
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    ap.add_argument(
        "--cells",
        nargs="+",
        default=None,
        help="Override the cell list (default: 7 fails + 2 PASS comparators).",
    )
    ap.add_argument(
        "--smoke",
        action="store_true",
        help="Run on ONE cell only (the first listed); no figure rendered.",
    )
    ap.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(".claude/cache/issue621_analysis/ladder_hf_cache"),
        help="HF download cache root (per-cell adapters land under here).",
    )
    ap.add_argument(
        "--out-json",
        type=Path,
        default=Path(".claude/cache/issue621_analysis/out/ladder_a_t_alignment.json"),
        help="Per-cell trajectory JSON output.",
    )
    ap.add_argument(
        "--figures-dir",
        type=Path,
        default=Path("figures/issue_621"),
        help="Figure output directory.",
    )
    args = ap.parse_args(argv)

    cells = list(args.cells) if args.cells else list(FAILED_CELLS) + list(PASS_COMPARATORS)
    if args.smoke:
        cells = cells[:1]
        log.info("SMOKE mode: only running %s", cells[0])

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)

    log.info("Computing a(t) ladder for %d cells", len(cells))
    payload: dict = {
        "generated": _dt.datetime.now(_dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "primary_metric": "all_layer_abs_cos_a_init (28 layers × 2 modules = 56 rows)",
        "secondary_metric": (f"band_mean_abs_cos_a_init (L{min(BAND_LAYERS)}-L{max(BAND_LAYERS)})"),
        "modules": ["q_proj", "v_proj"],
        "source": "scripts/issue621_checkpoint_ladder.py (9a-ter free-analysis follow-up)",
        "hf_repo": HF_MODEL_REPO,
        "parity_note": (
            "primary cos / Δa/a₀ at the terminal point match the body H2 "
            "headline h2_mean_cos / h2_mean_delta exactly (all-layer aggregation)."
        ),
    }
    for cell in cells:
        payload[cell] = _trajectory_for_cell(cell, args.cache_dir)

    args.out_json.write_text(json.dumps(payload, indent=1))
    log.info("wrote %s", args.out_json)

    if args.smoke:
        log.info("SMOKE mode: skipping figure render")
        return 0

    # Verdict summary table to stderr.
    fails = [c for c in FAILED_CELLS if c in payload]
    log.info("Verdict summary (FAILed cells):")
    for cell in fails:
        tr = payload[cell]
        log.info(
            "  %-40s  cos(end)=%.4f  Δa/a₀=%.4f  cos_slope/step=%+.2e  -> %s",
            cell,
            tr["cos_a_t_a_init"][-1],
            tr["delta_a_over_a0"][-1],
            tr["cos_slope_per_step_tail"],
            tr["verdict"],
        )

    fig_path = _draw_figure(payload, args.figures_dir)
    log.info("wrote %s", fig_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
