#!/usr/bin/env python3
"""Render the chain-of-thought necessity appendix figure.

Two panels in the Figure 2 visual system, written to
``figures/paper/c1_cot_necessity.{pdf,png}`` with a grayscale audit and a JSON
sidecar of every plotted value.

A  Corpus level. For each (corpus, stratum) subset used in the #2546 fits, the
   x axis is the subset's operational necessity rate (share of classified
   questions whose reasoning-mode answer is exactly correct while the
   non-reasoning answer is wrong) and the y axis is the top-1 answer-retrieval
   gain from moving the map's input from the last context token to the
   end-of-thought token (lift over chance, end of thought minus context).
   Both reasoning settings are shown: OpenThinker3-7B against its Qwen2.5-7B-
   Instruct parent (circles) and Qwen3-8B with thinking on against off
   (diamonds). Color encodes the corpus stratum as in the main figure.
B  Question level. AUROC, with 95% class-stratified prompt bootstrap
   intervals, of five map-derived scores for separating MATH questions that
   needed reasoning from questions correct in both modes, from the committed
   #2546 necessity-discrimination follow-up.

Subset membership for the two corpora split across strata (GSM8K train by
calculator steps, ContextHub by level) is read from the pinned corpus files on
the HF data repo; everything else is read from ``eval_results/issue_2546``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from huggingface_hub import hf_hub_download
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from explore_persona_space.analysis.c2a_plot_style import (  # noqa: E402
    INK,
    MUTED,
    PAPER,
    STYLE_VERSION,
    save_c2a_figure,
    set_c2a_style,
    style_score_axis,
)

EVAL_DIR = ROOT / "eval_results" / "issue_2546"
CELLS_DIR = EVAL_DIR / "cells"
DEFAULT_OUT = ROOT / "figures" / "paper"
DEFAULT_STEM = "c1_cot_necessity"

HF_REPO = "superkaiba1/explore-persona-space-data"
HF_REVISION = "8368cc69f887d20931acd8c4d76c142275173728"
SOURCE_REF = "42308cc7522dcb0a2a76b332b0c24d981de4b585"
CORPORA_PREFIX = "issue2546_cotmap/corpora_v1"
CONTEXTHUB_SHARDS = [f"contexthub.shard0{i}.jsonl" for i in range(4)]

STRATUM_COLOR = {"does": "#7B3294", "doesnt": "#5AAE61"}
STRATUM_LABEL = {"does": "Needs-reasoning corpora", "doesnt": "No-reasoning corpora"}
MODELS = {
    1: {"label": "OpenThinker3-7B vs parent", "marker": "o", "labels": "pair_necessity_a1.json", "disc": "open_thinker"},
    3: {"label": "Qwen3-8B, thinking on vs off", "marker": "D", "labels": "qwen3_toggle_labels.json", "disc": "qwen3"},
}
# (stratum, corpus key in the stratum cell's per_corpus block, display name)
SUBSETS = [
    ("does", "math", "MATH"),
    ("does", "gsm8k_train", "GSM8K, 4+ steps"),
    ("does", "contexthub", "ContextHub L3-4"),
    ("doesnt", "mmlu", "MMLU"),
    ("doesnt", "arc_challenge", "ARC-Challenge"),
    ("doesnt", "csqa", "CommonsenseQA"),
    ("doesnt", "piqa", "PIQA"),
    ("doesnt", "gsm8k_train", "GSM8K, 1 step"),
    ("doesnt", "contexthub", "ContextHub L1"),
]
SCORE_ORDER = [
    ("prompt_error", "Prompt map error"),
    ("prompt_retrieval_miss", "Prompt retrieval miss"),
    ("error_reduction", "Error reduction\u2020"),
    ("log_error_ratio", "Log error ratio\u2020"),
    ("retrieval_top1_gain", "Top-1 retrieval gain\u2020"),
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _hf_file(name: str) -> Path:
    return Path(
        hf_hub_download(
            HF_REPO,
            filename=f"{CORPORA_PREFIX}/{name}",
            repo_type="dataset",
            revision=HF_REVISION,
        )
    )


def subset_membership() -> dict[tuple[str, str], set[str]]:
    """Row ids of the GSM8K-train and ContextHub rows in each stratum subset."""

    members: dict[tuple[str, str], set[str]] = {
        ("does", "gsm8k_train"): set(),
        ("doesnt", "gsm8k_train"): set(),
        ("does", "contexthub"): set(),
        ("doesnt", "contexthub"): set(),
    }
    with _hf_file("gsm8k_train.jsonl").open() as handle:
        for line in handle:
            row = json.loads(line)
            steps = int(row["k"])
            if steps >= 4:
                members[("does", "gsm8k_train")].add(row["row_id"])
            elif steps == 1:
                members[("doesnt", "gsm8k_train")].add(row["row_id"])
    for shard in CONTEXTHUB_SHARDS:
        with _hf_file(shard).open() as handle:
            for line in handle:
                row = json.loads(line)
                level = int(row["level"])
                if level >= 3:
                    members[("does", "contexthub")].add(row["row_id"])
                elif level == 1:
                    members[("doesnt", "contexthub")].add(row["row_id"])
    for key, ids in members.items():
        if not ids:
            raise ValueError(f"empty subset {key}")
    return members


def necessity_rates(arm: int, members: dict[tuple[str, str], set[str]]) -> dict[tuple[str, str], dict[str, Any]]:
    path = EVAL_DIR / "necessity" / MODELS[arm]["labels"]
    labels: dict[str, str] = json.loads(path.read_text())["labels"]
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for stratum, corpus, _name in SUBSETS:
        counts: Counter[str] = Counter()
        restrict = members.get((stratum, corpus))
        for row_id, label in labels.items():
            if row_id.split(":", 1)[0] != corpus:
                continue
            if restrict is not None and row_id not in restrict:
                continue
            counts[label] += 1
        classified = sum(v for k, v in counts.items() if k != "unknown")
        if classified == 0:
            raise ValueError(f"no classified rows for arm {arm} {stratum}/{corpus}")
        out[(stratum, corpus)] = {
            "necessary": counts.get("necessary", 0),
            "classified": classified,
            "rate": counts.get("necessary", 0) / classified,
            "labels_source": str(path.relative_to(ROOT)),
        }
    return out


RECIPE_DIR = EVAL_DIR / "retrieval_recipe"


def _load_hits(arm: int, cell: str, stratum: str) -> dict[str, bool]:
    """Per-question own-answer hits under the paper recipe (whitened cosine + CSLS, held-out-fold pool)."""
    path = RECIPE_DIR / f"hits__arm{arm}__{cell}__{stratum}__main.npz"
    z = np.load(path)
    return {str(r): bool(h) for r, h in zip(z["row_ids"], z["hit_whitened_csls"])}


def _auroc(pos: np.ndarray, neg: np.ndarray) -> float:
    """Mann-Whitney AUROC with ties counted as half."""
    from scipy.stats import rankdata

    scores = np.concatenate([pos, neg]); ranks = rankdata(scores)
    n_pos, n_neg = len(pos), len(neg)
    return float((ranks[:n_pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def recipe_auroc(arm: int, labels_file: str, n_boot: int = 2000, seed: int = 0) -> dict[str, dict[str, Any]]:
    """AUROC of the two retrieval-based necessity scores within MATH, recomputed from the recipe hits."""
    labels = json.loads((EVAL_DIR / "necessity" / labels_file).read_text())["labels"]
    hits_a = _load_hits(arm, "p7_A", "does"); hits_d = _load_hits(arm, "p7_D", "does")
    rows = [r for r in hits_a if r.split(":", 1)[0] == "math" and r in hits_d and labels.get(r) in ("necessary", "both_correct")]
    y = np.array([labels[r] == "necessary" for r in rows]); ha = np.array([hits_a[r] for r in rows], float); hd = np.array([hits_d[r] for r in rows], float)
    scores = {"prompt_retrieval_miss": 1.0 - ha, "retrieval_top1_gain": hd - ha}
    rng = np.random.default_rng(seed); pos_idx = np.where(y)[0]; neg_idx = np.where(~y)[0]
    out = {}
    for key, sc in scores.items():
        point = _auroc(sc[pos_idx], sc[neg_idx])
        boots = [_auroc(sc[rng.choice(pos_idx, len(pos_idx))], sc[rng.choice(neg_idx, len(neg_idx))]) for _ in range(n_boot)]
        out[key] = {"auroc": point, "auroc_ci": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))], "n_analysis": int(len(rows)), "n_necessary": int(y.sum())}
    return out


def _per_corpus_lift(cell: str, stratum: str, arm: int, corpus: str) -> dict[str, float]:
    path = CELLS_DIR / f"{cell}__{stratum}__a{arm}.json"
    data = json.loads(path.read_text())
    if data["status"] != "ok":
        raise ValueError(f"{path.name}: status {data['status']!r}")
    block = data["knn_content"]["euclidean"]["per_corpus"][corpus]
    return {
        "n": int(block["n"]),
        "acc_at_1": float(block["acc_at_1"]),
        "chance": float(block["chance_mean"]),
        "lift": float(block["lift"]),
        "source": str(path.relative_to(ROOT)),
    }


def load_corpus_panel(members: dict[tuple[str, str], set[str]]) -> dict[str, Any]:
    panel: dict[str, Any] = {}
    for arm, spec in MODELS.items():
        rates = necessity_rates(arm, members)
        points = []
        hits = {(cell, st): _load_hits(arm, cell, st) for cell in ("p7_A", "p7_D") for st in ("does", "doesnt")}
        for stratum, corpus, name in SUBSETS:
            restrict = members.get((stratum, corpus))
            rows = [r for r in hits[("p7_A", stratum)] if r.split(":", 1)[0] == corpus and (restrict is None or r in restrict) and r in hits[("p7_D", stratum)]]
            if not rows:
                raise ValueError(f"no rows for {stratum}/{corpus} arm {arm}")
            acc_ctx = float(np.mean([hits[("p7_A", stratum)][r] for r in rows]))
            acc_end = float(np.mean([hits[("p7_D", stratum)][r] for r in rows]))
            points.append(
                {
                    "stratum": stratum,
                    "corpus": corpus,
                    "name": name,
                    "n_rows": len(rows),
                    "necessity": rates[(stratum, corpus)],
                    "context_lift": acc_ctx,
                    "end_of_thought_lift": acc_end,
                    "retrieval_gain": acc_end - acc_ctx,
                    "metric": "acc@1 of the own answer, whitened cosine + CSLS, held-out-fold pool",
                    "sources": [str((RECIPE_DIR / f"hits__arm{arm}__{c}__{stratum}__main.npz").relative_to(ROOT)) for c in ("p7_A", "p7_D")],
                }
            )
        xs = [p["necessity"]["rate"] for p in points]
        ys = [p["retrieval_gain"] for p in points]
        rho, pval = spearmanr(xs, ys)
        panel[str(arm)] = {
            "model": spec["label"],
            "points": points,
            "spearman_rho": float(rho),
            "spearman_p": float(pval),
        }
    return panel


def load_question_panel() -> dict[str, Any]:
    path = EVAL_DIR / "necessity_discrimination" / "summary.json"
    summary = json.loads(path.read_text())
    if summary["status"] != "ok":
        raise ValueError("necessity discrimination summary not ok")
    panel: dict[str, Any] = {"source": str(path.relative_to(ROOT)), "source_sha256": _sha256(path), "arms": {}}
    for arm, spec in MODELS.items():
        block = summary["arms"][spec["disc"]]
        recipe = recipe_auroc(arm, spec["labels"])
        scores = []
        for key, label in SCORE_ORDER:
            metric = block["metrics"][key]
            entry = {
                "key": key,
                "label": label,
                "auroc": float(metric["roc_auc"]),
                "auroc_ci": [float(v) for v in metric["roc_auc_ci"]],
                "uses_boundary": bool(metric["uses_boundary"]),
            }
            if key in recipe:
                entry.update({"auroc": recipe[key]["auroc"], "auroc_ci": recipe[key]["auroc_ci"], "recomputed": "paper recipe hits, stratum-fit maps restricted to MATH"})
            scores.append(entry)
        panel["arms"][str(arm)] = {
            "model": spec["label"],
            "n_analysis": int(block["n_analysis"]),
            "n_necessary": int(block["n_necessary"]),
            "prevalence": float(block["prevalence_necessary"]),
            "scores": scores,
        }
    return panel


def _kicker(ax: plt.Axes, title: str, kicker: str) -> None:
    ax.set_title(title, loc="left", y=1.04, pad=0, fontweight=650, fontsize=17)
    ax.text(0.0, 1.235, kicker.upper(), transform=ax.transAxes, fontsize=12, fontweight=700, color=MUTED, va="bottom", ha="left")


def _nudged_label_positions(ys: list[float], min_gap: float, y_max: float | None = None) -> list[float]:
    """Push label y positions apart (in data units) so adjacent labels do not overlap, keeping them below y_max."""

    order = sorted(range(len(ys)), key=lambda i: ys[i])
    placed: list[float] = []
    for i in order:
        y = ys[i]
        if placed and y - placed[-1] < min_gap:
            y = placed[-1] + min_gap
        placed.append(y)
    if y_max is not None:
        ceiling = y_max - 0.6 * min_gap
        for j in range(len(placed) - 1, -1, -1):
            if placed[j] > ceiling:
                placed[j] = ceiling
            ceiling = placed[j] - min_gap
    out = [0.0] * len(ys)
    for i, y in zip(order, placed):
        out[i] = y
    return out


def _corpus_axes(ax: plt.Axes, block: dict[str, Any], marker: str, x_max: float, *, show_ylabel: bool) -> None:
    style_score_axis(ax, y_min=0.0, y_max=0.3, y_step=0.1)
    points = block["points"]
    xs = [pt["necessity"]["rate"] for pt in points]
    ys = [pt["retrieval_gain"] for pt in points]
    label_ys = _nudged_label_positions(ys, min_gap=0.019, y_max=0.3)
    for pt, x, y, ly in zip(points, xs, ys, label_ys):
        color = STRATUM_COLOR[pt["stratum"]]
        ax.plot(x, y, marker=marker, color=color, markersize=10 if marker == "o" else 8.5, linestyle="none", zorder=4)
        on_right = x > 0.55 * x_max
        ax.annotate(
            pt["name"],
            (x, y),
            xytext=(x - 0.04 * x_max if on_right else x + 0.04 * x_max, ly),
            textcoords="data",
            fontsize=9.5,
            color=MUTED,
            va="center",
            ha="right" if on_right else "left",
            zorder=5,
            arrowprops={"arrowstyle": "-", "color": "#C8C6BF", "lw": 0.8, "shrinkA": 0, "shrinkB": 4} if abs(ly - y) > 0.012 else None,
        )
    ax.set_xlim(-0.01 * x_max / 0.15, x_max)
    ax.set_xlabel("Necessity rate", labelpad=10)
    if show_ylabel:
        ax.set_ylabel("Gain in own-answer acc@1\nfrom end of thought  ↑", labelpad=10)


def _question_axes(ax: plt.Axes, panel: dict[str, Any]) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#A9A69E")
    ax.spines["bottom"].set_color("#A9A69E")
    ax.tick_params(length=0, pad=8)
    ax.grid(axis="x", color="#C8C6BF", lw=1.0, alpha=0.55)
    ax.set_axisbelow(True)
    n_scores = len(SCORE_ORDER)
    offsets = {1: 0.16, 3: -0.16}
    colors = {1: INK, 3: MUTED}
    for arm, spec in MODELS.items():
        for i, score in enumerate(panel["arms"][str(arm)]["scores"]):
            y = n_scores - 1 - i + offsets[arm]
            lo, hi = score["auroc_ci"]
            ax.plot([lo, hi], [y, y], color=colors[arm], lw=2.2, solid_capstyle="round", zorder=3)
            ax.plot(score["auroc"], y, marker=spec["marker"], color=colors[arm], markersize=10 if spec["marker"] == "o" else 8.5, linestyle="none", zorder=4)
    ax.axvline(0.5, color=MUTED, lw=1.2, ls=(0, (2, 3)), zorder=1)
    ax.set_yticks(range(n_scores))
    ax.set_yticklabels([label for _key, label in reversed(SCORE_ORDER)], fontsize=13)
    ax.set_ylim(-0.6, n_scores - 0.4)
    ax.set_xlim(0.45, 0.62)
    ax.set_xticks([0.45, 0.50, 0.55, 0.60])
    ax.set_xlabel("AUROC (0.5 is chance)", labelpad=10)


def make_figure(corpus_panel: dict[str, Any], question_panel: dict[str, Any]) -> plt.Figure:
    set_c2a_style()
    fig = plt.figure(figsize=(14.4, 6.6), constrained_layout=False)
    grid = fig.add_gridspec(1, 3, left=0.065, right=0.985, top=0.73, bottom=0.14, wspace=0.55, width_ratios=[1.0, 1.0, 1.15])
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[0, 2])
    _corpus_axes(ax_a, corpus_panel["1"], MODELS[1]["marker"], 0.16, show_ylabel=True)
    _kicker(ax_a, "End-of-thought gains are small\nand do not track the necessity rate", "A  ·  OpenThinker3-7B vs parent")
    _corpus_axes(ax_b, corpus_panel["3"], MODELS[3]["marker"], 0.42, show_ylabel=False)
    _kicker(ax_b, "Larger gains for the thinking toggle,\nfalling as the necessity rate rises", "B  ·  Qwen3-8B, thinking on vs off")
    _question_axes(ax_c, question_panel)
    _kicker(ax_c, "No map score identifies\nwhich MATH questions need reasoning", "C  ·  per-question AUROC, 95% intervals")

    model_handles = [
        Line2D([0], [0], color=INK, marker=MODELS[1]["marker"], markersize=10, lw=0, label=MODELS[1]["label"]),
        Line2D([0], [0], color=INK, marker=MODELS[3]["marker"], markersize=8.5, lw=0, label=MODELS[3]["label"]),
    ]
    stratum_handles = [Patch(facecolor=STRATUM_COLOR[s], edgecolor=STRATUM_COLOR[s], label=STRATUM_LABEL[s]) for s in ("does", "doesnt")]
    fig.text(0.065, 0.965, "REASONING SETTING", color=MUTED, fontsize=11.5, fontweight=750, ha="left", va="center")
    fig.legend(handles=model_handles, loc="upper left", bbox_to_anchor=(0.064, 0.95), ncol=2, frameon=False, columnspacing=1.3, handlelength=1.4, handletextpad=0.6, borderaxespad=0)
    fig.text(0.635, 0.965, "CORPORA (A, B)", color=MUTED, fontsize=11.5, fontweight=750, ha="left", va="center")
    fig.legend(handles=stratum_handles, loc="upper left", bbox_to_anchor=(0.634, 0.95), ncol=2, frameon=False, columnspacing=1.3, handlelength=1.4, handletextpad=0.6, borderaxespad=0)
    return fig


def _git_state() -> dict[str, str | bool | None]:
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=False, capture_output=True, text=True)
    dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, check=False, capture_output=True, text=True)
    return {
        "commit": commit.stdout.strip() if commit.returncode == 0 else None,
        "tracked_worktree_dirty": bool(dirty.stdout.strip()) if dirty.returncode == 0 else None,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--stem", default=DEFAULT_STEM)
    args = parser.parse_args(argv)

    members = subset_membership()
    corpus_panel = load_corpus_panel(members)
    question_panel = load_question_panel()
    font = set_c2a_style()
    fig = make_figure(corpus_panel, question_panel)
    stem = args.out_dir / args.stem
    outputs = save_c2a_figure(
        fig,
        stem,
        title="Chain-of-thought necessity at the corpus and question level",
        subject="Issue #2546 cells and necessity-discrimination follow-up",
        creator="scripts/section45_cot_necessity_figure.py",
    )
    plt.close(fig)
    sidecar = stem.with_name(f"{args.stem}_data.json")
    sidecar.write_text(
        json.dumps(
            {
                "style_version": STYLE_VERSION,
                "font": font,
                "git": _git_state(),
                "provenance": {
                    "task": 2546,
                    "hf_data_repo": HF_REPO,
                    "hf_revision": HF_REVISION,
                    "source_ref": SOURCE_REF,
                    "subset_rule": "GSM8K train: needs = 4+ calculator steps, no = 1 step; ContextHub: needs = levels 3-4, no = level 1",
                    "necessity_label": "reasoning-mode answer exactly correct and non-reasoning answer wrong; rate = necessary / classified (unknown excluded)",
                    "retrieval": "top-1 nearest neighbour (euclidean) in the stratum pool, canonical answer-content hit rule; lift = acc@1 minus per-query chance",
                },
                "panel_A": corpus_panel,
                "panel_B": question_panel,
                "outputs": {k: str(v.relative_to(ROOT)) for k, v in outputs.items()},
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    for key, path in {**outputs, "data": sidecar}.items():
        print(f"{key}: {path}")
    for arm in MODELS:
        block = corpus_panel[str(arm)]
        print(f"arm {arm}: Spearman rho = {block['spearman_rho']:+.2f} (p = {block['spearman_p']:.3f}) over {len(block['points'])} subsets")
        for point in block["points"]:
            print(f"   {point['name']:18s} rate={point['necessity']['rate']:.3f} gain={point['retrieval_gain']:+.3f} n={point['n_rows']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
