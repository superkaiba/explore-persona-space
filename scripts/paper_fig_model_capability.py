"""Render the original capability panel from pinned, already computed results.

No model fitting or inference. All models share one marker style, as requested.
Artificial Analysis score provenance stays in the sidecar and manuscript text.
The recorded-measured-score sensitivity is recomputed exactly.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import rankdata, spearmanr

from explore_persona_space.analysis.c2a_plot_style import (
    INK,
    ROLES,
    better_label,
    c2a_figure,
    panel_header,
    save_c2a_figure,
    set_c2a_style,
    style_score_axis,
)

ROOT = Path(__file__).resolve().parents[1]
SOURCE_SHA = "14a9a3605c6f14e0537a7e6407ae71cf5093a972"
SOURCE_PATHS = (
    "eval_results/issue_2588/rank_relationships.json",
    "eval_results/issue_2588/mapping_rank_vs_capability.json",
)
EXCLUDED_MODELS = {"q25_32b": "User requested exclusion of the older Qwen2.5 release."}
PERMUTATION_SEED = 2588
LABELS = {
    "q35_0p8b": "Qwen3.5 0.8B",
    "q35_2b": "Qwen3.5 2B",
    "q35_4b": "Qwen3.5 4B",
    "q35_9b": "Qwen3.5 9B",
    "q35_27b": "Qwen3.5 27B",
    "q36_27b": "Qwen3.6 27B",
    "q38_27b": "Qwen3.8 27B",
    "o3_7b_i": "OLMo3 7B Instruct",
    "o31_32b_i": "OLMo3.1 32B Instruct",
    "q25_32b": "Qwen2.5 32B",
    "q3_32b": "Qwen3 32B",
}
# Text offsets in authoring points, with the original data positions unchanged.
OFFSETS = {
    "q35_0p8b": (12, 8),
    "q35_2b": (12, 5),
    "q35_4b": (12, -18),
    "q35_9b": (12, 7),
    "q35_27b": (12, 7),
    "q36_27b": (12, -21),
    "q38_27b": (-12, 9),
    "o3_7b_i": (12, -5),
    "o31_32b_i": (12, -10),
    "q25_32b": (12, 7),
    "q3_32b": (-12, -20),
}


def exact_spearman(rows: list[dict]) -> dict:
    """Enumerate model-label permutations in bounded batches (at most ten models)."""
    n = len(rows)
    if not 3 <= n <= 10:
        raise ValueError(f"Exact enumeration supports 3-10 models, got {n}")
    x, y = [rankdata([r[key] for r in rows]) for key in ("aa_index", "test_r2")]
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Nonfinite plotted coordinates")
    x, y = [(v - v.mean()) / np.linalg.norm(v - v.mean()) for v in (x, y)]
    rho = float(x @ y)
    permutations = itertools.permutations(y)
    count = exceed = 0
    while batch := list(itertools.islice(permutations, 8192)):
        null = np.asarray(batch) @ x
        count += len(batch)
        exceed += int(np.count_nonzero(np.abs(null) >= abs(rho) - 1e-12))
    assert count == math.factorial(n)
    assert np.isclose(
        rho, spearmanr([r["aa_index"] for r in rows], [r["test_r2"] for r in rows]).statistic
    )
    return {
        "n": n,
        "rho": rho,
        "p_uncorrected": exceed / count,
        "n_permutations": count,
        "method": "Exact two-sided model-label permutation test",
    }


def restricted_maxt(rows: list[dict], audit: dict, observed_rho: float) -> dict:
    """Reuse the original 56-test grid with joint predictor relabelling.

    This corrects the fixed correlation grid only, not the post-hoc model exclusion.
    Batched matrix products implement the original _maxt_block statistic.
    """
    predictors = [p for p in audit["predictors"] if p != "generation"]
    outcomes = audit["outcomes"]
    blocks = []
    for names in (predictors, outcomes):
        values = np.asarray([[r[key] for key in names] for r in rows], dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("Missing value in the restricted correction grid")
        ranks = rankdata(values, axis=0)
        ranks -= ranks.mean(axis=0)
        norms = np.linalg.norm(ranks, axis=0)
        if np.any(norms == 0):
            raise ValueError("Constant variable in the restricted correction grid")
        blocks.append(ranks / norms)
    x, y = blocks
    rng = np.random.default_rng(PERMUTATION_SEED)
    draws = audit["n_permutations"]
    permutations = np.stack([rng.permutation(len(rows)) for _ in range(draws)])
    null = np.einsum("bnp,no->bpo", x[permutations], y, optimize=True)
    maxes = np.abs(null).max(axis=(1, 2))
    exceed = int(np.count_nonzero(maxes >= abs(observed_rho) - 1e-12))
    assert len(predictors) * len(outcomes) == 56
    return {
        "p_familywise_maxT": (exceed + 1) / (draws + 1),
        "n_tests": len(predictors) * len(outcomes),
        "n_permutations": draws,
        "seed": PERMUTATION_SEED,
        "predictors": predictors,
        "outcomes": outcomes,
        "method": "Joint-relabeling max-absolute-Spearman, Monte Carlo plus-one correction",
        "limitation": "Does not adjust for post-hoc model selection or family dependence.",
    }


def load_panel() -> tuple[list[dict], dict]:
    blobs = [
        subprocess.check_output(["git", "show", f"{SOURCE_SHA}:{path}"], cwd=ROOT)
        for path in SOURCE_PATHS
    ]
    audit, mapping = [json.loads(blob) for blob in blobs]
    maps = {row["key"]: row for row in mapping["maps"]}
    rows = []
    for row in audit["table"]:
        if row["arm"] != "no-thinking":
            continue
        source = maps[row["key"]]
        if source["input_position"] != "prompt_last":
            raise ValueError(f"Unexpected input position: {row['key']}")
        if source["aa_status"] not in {"estimated", "measured"}:
            raise ValueError(f"Unrecognized score provenance: {row['key']}")
        assert row["test_r2"] == source["mapping_performance"]["test_r2"]
        assert row["aa_index"] == source["aa_index"]
        rows.append(
            {
                **row,
                "label": LABELS[row["model_key"]],
                "aa_status": source["aa_status"],
                "aa_index_nonreasoning": source["aa_index_nonreasoning"],
                "layer": source["layer_star"],
                "n_test": source["mapping_performance"]["test_n"],
            }
        )
    assert len(rows) == 11 and len({r["model_key"] for r in rows}) == 11
    tests = [
        r
        for r in audit["grid"]["no-thinking"]["results"]
        if r["predictor"] == "aa_index" and r["outcome"] == "test_r2"
    ]
    assert len(tests) == 1
    result = tests[0]
    rho = float(spearmanr([r["aa_index"] for r in rows], [r["test_r2"] for r in rows]).statistic)
    assert np.isclose(rho, result["rho"], atol=1e-12)
    original_measured = exact_spearman([r for r in rows if r["aa_status"] == "measured"])
    excluded = [r for r in rows if r["model_key"] in EXCLUDED_MODELS]
    assert {r["model_key"] for r in excluded} == set(EXCLUDED_MODELS)
    rows = [r for r in rows if r["model_key"] not in EXCLUDED_MODELS]
    assert len(rows) == 10 and len({r["model_key"] for r in rows}) == 10
    plotted_result = exact_spearman(rows)
    maxt = restricted_maxt(rows, audit, plotted_result["rho"])
    measured = [r for r in rows if r["aa_status"] == "measured"]
    assert len(measured) == 4
    sensitivity = {
        **exact_spearman(measured),
        "note": (
            "Measured means marked measured in the historical registry, "
            "not reverified current scores."
        ),
    }
    return rows, {
        "source_commit": SOURCE_SHA,
        "source_files": {
            path: hashlib.sha256(blob).hexdigest()
            for path, blob in zip(SOURCE_PATHS, blobs, strict=True)
        },
        "full_panel": result,
        "plotted_panel": plotted_result,
        "restricted_panel_correction": maxt,
        "exclusions": EXCLUDED_MODELS,
        "excluded_rows": excluded,
        "selection_note": "Qwen2.5-32B was excluded after inspecting the original panel.",
        "original_measured_scores_only": original_measured,
        "recorded_measured_scores_only": sensitivity,
        "permutation_draws_full_panel": audit["n_permutations"],
        "correction_family_tests": audit["grid"]["no-thinking"]["n_tests_main_grid"],
        "scope": (
            "Ten-model subset of the original prompt-state panel, excluding Qwen2.5-32B; "
            "no end-of-thought or cap_long rows pooled."
        ),
        "score_provenance": (
            "Six Artificial Analysis values marked estimated and four marked measured "
            "in the original registry."
        ),
        "visual_encoding": (
            "All ten plotted models use identical filled circles; "
            "score status is not encoded, per user request."
        ),
        "mode_caveat": (
            "Model-level index scores can use different reasoning settings "
            "from these no-thinking maps."
        ),
    }


def main() -> None:
    """Render the unchanged ten-model panel in the shared manuscript visual style."""
    rows, provenance = load_panel()
    set_c2a_style()
    fig, frac = c2a_figure("full", aspect=0.47)
    ax = fig.add_subplot(111)
    fig.subplots_adjust(left=0.10, right=0.985, bottom=0.16, top=0.80)
    panel_header(
        ax,
        "",
        "10 models",
        "Predictability versus model capability",
    )
    style_score_axis(ax, y_min=0.59, y_max=0.756, y_step=0.04)
    ax.set_yticks([0.60, 0.65, 0.70, 0.75])
    ax.set_xlim(-3, 64)
    ax.set_xticks([0, 10, 20, 30, 40, 50, 60])
    ax.set_xlabel(better_label("Artificial Analysis Intelligence Index"))
    ax.set_ylabel(better_label(r"Held-out $R^2$"))
    color = ROLES["linear"].color
    ax.scatter(
        [r["aa_index"] for r in rows],
        [r["test_r2"] for r in rows],
        s=110,
        marker=ROLES["linear"].marker,
        facecolors=color,
        edgecolors=color,
        linewidths=1.9,
        zorder=3,
    )
    for row in rows:
        dx, dy = OFFSETS[row["model_key"]]
        ax.annotate(
            row["label"],
            (row["aa_index"], row["test_r2"]),
            xytext=(dx, dy),
            textcoords="offset points",
            ha="left" if dx > 0 else "right",
            va="center",
            color=INK,
        )
    stats = provenance["plotted_panel"]
    ax.text(
        0.035,
        0.955,
        rf"Spearman $\rho = {stats['rho']:.3f}$, $p = {stats['p_uncorrected']:.3f}$",
        transform=ax.transAxes,
        va="top",
    )
    stem = ROOT / "figures/paper/c1_model_capability"
    exported = save_c2a_figure(
        fig,
        stem,
        include_width=frac,
        title="Context-answer predictability and model capability",
        subject="Ten-model prompt-state subset, excluding Qwen2.5-32B, with uniform markers.",
        creator="scripts/paper_fig_model_capability.py",
    )
    provenance.update({"rows": rows, "render": exported["record"]})
    provenance["output_sha256"] = {
        key: hashlib.sha256(Path(exported[key]).read_bytes()).hexdigest()
        for key in ("pdf", "png", "grayscale")
    }
    provenance["producer_commit"] = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    provenance["producer_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    stem.with_suffix(".meta.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(
        json.dumps(
            {
                "figure": str(exported["pdf"]),
                "statistics": stats,
                "restricted_panel_correction": provenance["restricted_panel_correction"],
                "sensitivity": provenance["recorded_measured_scores_only"],
            },
            indent=2,
        )
    )
    plt.close(fig)


if __name__ == "__main__":
    main()
