#!/usr/bin/env python3
"""Within-corpus R^2 of the context-to-answer map by CoT necessity label.

For each corpus with its own #2546 fit cell (MATH, GSM8K train, ContextHub,
MMLU) and each reasoning setting (OpenThinker3-7B vs its non-reasoning parent,
Qwen3-8B thinking on vs off), this script takes the committed five-fold
out-of-fold predictions of the whole-corpus context-to-answer map (cell p7_A,
last context token -> mean answer state) and the end-of-thought map (p7_D),
joins them with the realized answer states from the raw state shards and with
the operational necessity label of every question, and reports held-out R^2
inside each label group. No map is refit: the question is whether the SAME map
predicts the answer state of questions that needed reasoning as well as the
answer state of questions the model gets right without it.

Labels (exact code, greedy decoding, no judge): ``necessary`` = reasoning-mode
answer exactly correct AND non-reasoning answer wrong; ``both_correct``;
``both_wrong``; ``pre_only_correct`` (OpenThinker3 pair) or
``rescued_by_no_think`` (Qwen3 toggle); ``unknown`` rows are dropped.

R^2 inside a group uses the group's own mean answer state as the baseline
(``r2_own_mean``); the sidecar also stores the variant that keeps the whole
corpus mean (``r2_corpus_mean``) and the mean per-question squared error.
Intervals are 95% percentile intervals over 1,000 question-level bootstrap
draws inside the group.

Inputs are the #2546 Hub artifacts (dataset superkaiba1/explore-persona-space-data,
revision 8368cc69f887d20931acd8c4d76c142275173728) mirrored locally under
``--hf-root``. Extracted target matrices are cached next to the mirror so the
raw shards are read once.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from explore_persona_space.analysis.c2a_plot_style import (  # noqa: E402
    INK,
    MUTED,
    STYLE_VERSION,
    save_c2a_figure,
    set_c2a_style,
    style_score_axis,
)

HF_REVISION = "8368cc69f887d20931acd8c4d76c142275173728"
SOURCE_REF = "42308cc7522dcb0a2a76b332b0c24d981de4b585"
DEFAULT_HF_ROOT = Path("/mnt/eps-data/thomasjiralerspong/cot_necessity/hf/issue2546_cotmap")
DEFAULT_OUT = ROOT / "figures" / "paper"
DEFAULT_STEM = "c1_cot_necessity_r2"
SUMMARY_PATH = ROOT / "eval_results" / "issue_2546" / "necessity_r2" / "summary.json"
CELLS_DIR = ROOT / "eval_results" / "issue_2546" / "cells"

CORPORA = [("math", "MATH"), ("gsm8k_train", "GSM8K\ntrain"), ("contexthub", "Context-\nHub"), ("mmlu", "MMLU")]
ARMS = {
    1: {
        "label": "OpenThinker3-7B",
        "comparator": "vs its non-reasoning parent",
        "layer": 19,
        "stem": "post__{corpus}",
        "labels_file": "eval_results_mirror/out/necessity/pair_necessity_a1.json",
        "other_label": "pre_only_correct",
    },
    3: {
        "label": "Qwen3-8B",
        "comparator": "thinking on vs off, same weights",
        "layer": 24,
        "stem": "think_on__{corpus}",
        "labels_file": "eval_results_mirror/out/necessity/qwen3_toggle_labels.json",
        "other_label": "rescued_by_no_think",
    },
}
CELLS = {"p7_A": "context", "p7_D": "end_of_thought"}
GROUPS = ("necessary", "both_correct", "both_wrong", "other")
GROUP_COLOR = {"necessary": "#D6604D", "both_correct": "#4393C3"}
GROUP_LABEL = {
    "necessary": "Reasoning necessary (right with reasoning, wrong without)",
    "both_correct": "Reasoning not needed (right both ways)",
}
N_BOOT = 1000
SEED = 0
BAR_WIDTH = 0.36


def load_targets(hf_root: Path, cache_dir: Path, arm: int, corpus: str) -> tuple[list[str], np.ndarray]:
    """Row ids and (n, H) float32 mean answer states at the headline layer."""

    import torch

    spec = ARMS[arm]
    stem = spec["stem"].format(corpus=corpus)
    cache = cache_dir / f"ans_mean__arm{arm}__{corpus}__l{spec['layer']}.npz"
    if cache.is_file():
        z = np.load(cache)
        return [str(r) for r in z["row_ids"]], z["ans_mean"].astype(np.float32)
    shard_dir = hf_root / "analysis_tensors" / "thinkstore" / f"arm{arm}" / stem
    files = sorted(shard_dir.glob("slot*.shard*.pt"))
    if not files:
        raise FileNotFoundError(f"no shards under {shard_dir}")
    row_ids: list[str] = []
    blocks: list[np.ndarray] = []
    t0 = time.time()
    for f in files:
        sh = torch.load(f, map_location="cpu", weights_only=False)
        if int(sh["arm"]) != arm:
            raise ValueError(f"{f.name}: arm {sh['arm']} != {arm}")
        kinds = list(sh["kinds_full"])
        layers = [int(v) for v in sh["layers_all"]]
        ki = kinds.index("ans_mean")
        li = layers.index(spec["layer"])
        blocks.append(sh["full"][:, ki, li].to(torch.float32).numpy())
        row_ids.extend(str(r) for r in sh["row_ids"])
        del sh
    ans = np.concatenate(blocks, axis=0)
    if ans.shape[0] != len(row_ids):
        raise ValueError(f"{stem}: {ans.shape[0]} rows vs {len(row_ids)} ids")
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez(cache, row_ids=np.asarray(row_ids), ans_mean=ans.astype(np.float16))
    print(f"[targets] {stem}: {len(row_ids)} rows from {len(files)} shards ({time.time() - t0:.0f}s)", flush=True)
    return row_ids, ans


def load_preds(hf_root: Path, arm: int, cell: str, corpus: str) -> tuple[list[str], np.ndarray]:
    path = hf_root / "analysis_tensors" / "preds" / f"arm{arm}" / f"{cell}__{corpus}__a{arm}.npz"
    z = np.load(path)
    layer = ARMS[arm]["layer"]
    fitted = np.asarray(z["fitted_mask"], dtype=bool)
    ids = [str(r) for r in z["conv_ids"][fitted]]
    pred = np.asarray(z[f"pred_l{layer}"][fitted], dtype=np.float32)
    return ids, pred


def _r2_stats(y: np.ndarray, yhat: np.ndarray, corpus_mean: np.ndarray, rng: np.random.Generator) -> dict[str, Any]:
    n = y.shape[0]
    err = np.sum((y - yhat) ** 2, axis=1)  # per-question squared error
    y_norm2 = np.sum(y ** 2, axis=1)
    mean = y.mean(axis=0)
    sst_own = float(np.sum((y - mean) ** 2))
    sst_corpus = float(np.sum((y - corpus_mean) ** 2))
    sse = float(err.sum())
    out: dict[str, Any] = {
        "n": int(n),
        "r2_own_mean": 1.0 - sse / sst_own,
        "r2_corpus_mean": 1.0 - sse / sst_corpus,
        "mse": sse / n,
        "target_variance_per_question": sst_own / n,
    }
    if n >= 20:
        counts = rng.multinomial(n, np.full(n, 1.0 / n), size=N_BOOT).astype(np.float64)  # (B, n)
        sse_b = counts @ err
        sum_y_b = counts @ y.astype(np.float64)  # (B, H)
        sst_own_b = counts @ y_norm2 - np.sum(sum_y_b ** 2, axis=1) / n
        sst_corpus_b = counts @ np.sum((y - corpus_mean) ** 2, axis=1)
        r2_own_b = 1.0 - sse_b / sst_own_b
        r2_corpus_b = 1.0 - sse_b / sst_corpus_b
        out["r2_own_mean_ci"] = [float(np.percentile(r2_own_b, 2.5)), float(np.percentile(r2_own_b, 97.5))]
        out["r2_corpus_mean_ci"] = [float(np.percentile(r2_corpus_b, 2.5)), float(np.percentile(r2_corpus_b, 97.5))]
        out["mse_ci"] = [float(np.percentile(sse_b / n, 2.5)), float(np.percentile(sse_b / n, 97.5))]
    return out


def _pooled_weighted_stats(cells: list[dict[str, Any]], group: str, weights: dict[str, float], rng: np.random.Generator) -> dict[str, Any]:
    """Pooled R^2 of one label group across corpora, with each corpus given a fixed total weight.

    ``cells`` holds one entry per corpus with per-question arrays ``err`` (squared error),
    ``y`` (targets), ``labels``, ``corpus`` and ``corpus_mean``. Questions of corpus c in
    the group get weight weights[c] / n_{group,c}, so both groups end up with the same
    corpus composition regardless of how many questions each corpus contributes.
    """

    ys, errs, ws, dev_corpus = [], [], [], []
    n_by_corpus: dict[str, int] = {}
    for cell in cells:
        mask = cell["labels"] == group
        n = int(mask.sum())
        if n == 0:
            continue
        n_by_corpus[cell["corpus"]] = n
        ys.append(cell["y"][mask].astype(np.float64))
        errs.append(cell["err"][mask].astype(np.float64))
        ws.append(np.full(n, weights[cell["corpus"]] / n))
        dev_corpus.append(np.sum((cell["y"][mask].astype(np.float64) - cell["corpus_mean"]) ** 2, axis=1))
    y = np.concatenate(ys); err = np.concatenate(errs); w = np.concatenate(ws); dev_c = np.concatenate(dev_corpus)
    y_norm2 = np.sum(y ** 2, axis=1)

    def stats(count: np.ndarray) -> tuple[float, float]:
        ww = w * count
        tot = ww.sum()
        sse = float(ww @ err)
        mean = (ww @ y) / tot
        sst_own = float(ww @ y_norm2 - tot * float(mean @ mean))
        sst_corpus = float(ww @ dev_c)
        return 1.0 - sse / sst_own, 1.0 - sse / sst_corpus

    r2_own, r2_corpus = stats(np.ones_like(w))
    # stratified bootstrap: resample questions within each corpus cell of the group
    boot = np.empty((N_BOOT, 2))
    offsets = np.cumsum([0] + [len(e) for e in errs])
    for b in range(N_BOOT):
        count = np.empty_like(w)
        for k, e in enumerate(errs):
            n = len(e)
            count[offsets[k]:offsets[k + 1]] = rng.multinomial(n, np.full(n, 1.0 / n))
        boot[b] = stats(count)
    return {
        "n": int(len(err)),
        "n_by_corpus": n_by_corpus,
        "weights": weights,
        "r2_own_mean": r2_own,
        "r2_own_mean_ci": [float(np.percentile(boot[:, 0], 2.5)), float(np.percentile(boot[:, 0], 97.5))],
        "r2_corpus_mean": r2_corpus,
        "r2_corpus_mean_ci": [float(np.percentile(boot[:, 1], 2.5)), float(np.percentile(boot[:, 1], 97.5))],
    }


def analyze(hf_root: Path, cache_dir: Path, arms: list[int] | None = None, corpora: list[str] | None = None) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for arm, spec in ARMS.items():
        if arms and arm not in arms:
            continue
        labels_all = json.loads((hf_root / spec["labels_file"]).read_text())["labels"]
        arm_out: dict[str, Any] = {"model": spec["label"], "comparator": spec["comparator"], "layer": spec["layer"], "corpora": {}, "pooled_equal_corpus_weight": {}}
        raw: dict[str, list[dict[str, Any]]] = {name: [] for name in CELLS.values()}
        for corpus, _name in CORPORA:
            if corpora and corpus not in corpora:
                continue
            t_ids, targets = load_targets(hf_root, cache_dir, arm, corpus)
            t_pos = {r: i for i, r in enumerate(t_ids)}
            corpus_out: dict[str, Any] = {}
            for cell, cell_name in CELLS.items():
                p_ids, pred = load_preds(hf_root, arm, cell, corpus)
                keep = [i for i, r in enumerate(p_ids) if r in t_pos]
                if len(keep) != len(p_ids):
                    print(f"[warn] arm{arm} {cell} {corpus}: {len(p_ids) - len(keep)} predicted rows have no target", flush=True)
                ids = [p_ids[i] for i in keep]
                yhat = pred[keep]
                y = targets[[t_pos[r] for r in ids]]
                labels = np.asarray([labels_all.get(r, "missing") for r in ids])
                labels = np.where(labels == spec["other_label"], "other", labels)
                rng = np.random.default_rng(SEED)
                corpus_mean = y.mean(axis=0)
                cell_out: dict[str, Any] = {
                    "n_rows": int(len(ids)),
                    "whole_corpus": _r2_stats(y, yhat, corpus_mean, rng),
                    "label_counts": {str(k): int(v) for k, v in zip(*np.unique(labels, return_counts=True))},
                    "groups": {},
                }
                committed = json.loads((CELLS_DIR / f"{cell}__{corpus}__a{arm}.json").read_text())
                cell_out["committed_r2_headline"] = float(committed["r2_headline"])
                cell_out["whole_corpus_abs_dev_from_committed"] = abs(cell_out["whole_corpus"]["r2_own_mean"] - float(committed["r2_headline"]))
                for group in GROUPS:
                    mask = labels == group
                    if mask.sum() < 20:
                        cell_out["groups"][group] = {"n": int(mask.sum()), "skipped": "fewer than 20 rows"}
                        continue
                    cell_out["groups"][group] = _r2_stats(y[mask], yhat[mask], corpus_mean, rng)
                corpus_out[cell_name] = cell_out
                raw[cell_name].append({"corpus": corpus, "y": y, "err": np.sum((y - yhat) ** 2, axis=1), "labels": labels, "corpus_mean": corpus_mean.astype(np.float64)})
                g = cell_out["groups"]
                print(
                    f"arm{arm} {corpus:12s} {cell_name:15s} whole={cell_out['whole_corpus']['r2_own_mean']:.3f} (committed {committed['r2_headline']:.3f}) "
                    + " ".join(f"{k}={v['r2_own_mean']:.3f}(n={v['n']})" for k, v in g.items() if "r2_own_mean" in v),
                    flush=True,
                )
            arm_out["corpora"][corpus] = corpus_out
        if not corpora or len(raw["context"]) == len(CORPORA):
            weights = {c: 1.0 / len(CORPORA) for c, _n in CORPORA}
            for cell_name, cells in raw.items():
                rng = np.random.default_rng(SEED + 1)
                pooled = {group: _pooled_weighted_stats(cells, group, weights, rng) for group in ("necessary", "both_correct")}
                arm_out["pooled_equal_corpus_weight"][cell_name] = pooled
                print(f"arm{arm} pooled(eq)   {cell_name:15s} " + " ".join(f"{g}={v['r2_own_mean']:.3f}[{v['r2_own_mean_ci'][0]:.3f},{v['r2_own_mean_ci'][1]:.3f}] corpus-mean={v['r2_corpus_mean']:.3f}" for g, v in pooled.items()), flush=True)
        results[str(arm)] = arm_out
    return results


def _kicker(ax: plt.Axes, title: str, kicker: str) -> None:
    ax.set_title(title, loc="left", y=1.04, pad=0, fontweight=650, fontsize=17)
    ax.text(0.0, 1.235, kicker.upper(), transform=ax.transAxes, fontsize=12, fontweight=700, color=MUTED, va="bottom", ha="left")


def make_figure(results: dict[str, Any], titles: dict[str, str], *, arms: tuple[str, ...] = ("1",), whole_corpus_line: bool = False, baseline: str = "corpus") -> plt.Figure:
    key = "r2_corpus_mean" if baseline == "corpus" else "r2_own_mean"
    set_c2a_style()
    n = len(arms)
    fig = plt.figure(figsize=(10.4 if n == 1 else 14.4, 6.4), constrained_layout=False)
    grid = fig.add_gridspec(1, n, left=0.10 if n == 1 else 0.07, right=0.985, top=0.68, bottom=0.17, wspace=0.22)
    axes = [fig.add_subplot(grid[0, i]) for i in range(n)]
    for ax, arm, panel in zip(axes, arms, "ABCD"):
        block = results[arm]
        spec = ARMS[int(arm)]
        style_score_axis(ax, y_min=0.0, y_max=0.8, y_step=0.2)
        for i, (corpus, _name) in enumerate(CORPORA):
            cell = block["corpora"][corpus]["context"]
            for j, group in enumerate(("necessary", "both_correct")):
                stats = cell["groups"][group]
                if key not in stats:
                    continue
                x = i + (j - 0.5) * BAR_WIDTH
                v = stats[key]
                lo, hi = stats[key + "_ci"]
                ax.bar(x, v, width=BAR_WIDTH, color=GROUP_COLOR[group], linewidth=0, zorder=3)
                ax.errorbar(x, v, yerr=[[v - lo], [hi - v]], fmt="none", ecolor=INK, elinewidth=1.2, capsize=3, capthick=1.2, zorder=4)
            if whole_corpus_line:
                whole = cell["whole_corpus"]["r2_own_mean"]
                ax.plot([i - 0.42, i + 0.42], [whole, whole], color=INK, lw=1.6, ls=(0, (3, 2)), zorder=5)
        pooled = block.get("pooled_equal_corpus_weight", {}).get("context")
        ticks = list(range(len(CORPORA))); labels = [name for _c, name in CORPORA]
        if pooled:
            xp = len(CORPORA) + 0.5
            for j, group in enumerate(("necessary", "both_correct")):
                st = pooled[group]; v = st[key]; lo, hi = st[key + "_ci"]
                x = xp + (j - 0.5) * BAR_WIDTH
                ax.bar(x, v, width=BAR_WIDTH, color=GROUP_COLOR[group], linewidth=0, zorder=3)
                ax.errorbar(x, v, yerr=[[v - lo], [hi - v]], fmt="none", ecolor=INK, elinewidth=1.2, capsize=3, capthick=1.2, zorder=4)
            ax.axvline(len(CORPORA) - 0.25, color=MUTED, lw=1.0, ls=(0, (2, 3)), zorder=1)
            ticks.append(xp); labels.append("All four,\nequal corpus\nweight")
        ax.set_xlim(-0.6, (ticks[-1] if ticks else len(CORPORA)) + 0.6)
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels, fontsize=13, linespacing=1.15)
        ax.set_ylabel("Held-out $R^2$ against the corpus mean  ↑" if baseline == "corpus" else "Held-out $R^2$, context → answer  ↑", labelpad=12)
        prefix = f"{panel}  ·  " if n > 1 else ""
        _kicker(ax, titles[arm], f"{prefix}{spec['label']}, {spec['comparator']}, layer {spec['layer']}, context → answer map")
    handles = [Patch(facecolor=GROUP_COLOR[g], edgecolor=GROUP_COLOR[g], label=GROUP_LABEL[g]) for g in ("necessary", "both_correct")]
    if whole_corpus_line:
        handles.append(Line2D([0], [0], color=INK, lw=1.6, ls=(0, (3, 2)), label="Whole corpus"))
    x0 = 0.10 if n == 1 else 0.07
    fig.text(x0, 0.965, "QUESTIONS, LABELED BY WHETHER REASONING WAS NEEDED", color=MUTED, fontsize=11.5, fontweight=750, ha="left", va="center")
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(x0 - 0.001, 0.948), ncol=1 if n == 1 else 3, frameon=False, columnspacing=1.3, handlelength=1.6, handletextpad=0.6, borderaxespad=0)
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
    parser.add_argument("--hf-root", type=Path, default=DEFAULT_HF_ROOT)
    parser.add_argument("--cache-dir", type=Path, default=None, help="where extracted target matrices are cached (default: <hf-root>/../targets)")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--stem", default=DEFAULT_STEM)
    parser.add_argument("--summary", type=Path, default=SUMMARY_PATH)
    parser.add_argument("--title-a", default="Predictability of the answer state,\nby whether the question needed reasoning")
    parser.add_argument("--title-b", default="Same on the thinking toggle")
    parser.add_argument("--both-panels", action="store_true", help="also plot the Qwen3-8B thinking toggle as panel B")
    parser.add_argument("--whole-corpus-line", action="store_true", help="draw the whole-corpus R^2 as a dashed line per corpus")
    parser.add_argument("--baseline", choices=("corpus", "own"), default="corpus", help="R^2 baseline: each corpus's mean answer state (same predictor for both groups) or each group's own mean")
    parser.add_argument("--figure-only", action="store_true", help="re-render from an existing summary without touching the shards")
    parser.add_argument("--arms", type=int, nargs="*", default=None, help="restrict to these arms (smoke runs)")
    parser.add_argument("--corpora", nargs="*", default=None, help="restrict to these corpora (smoke runs)")
    parser.add_argument("--no-figure", action="store_true", help="analysis only")
    args = parser.parse_args(argv)
    cache_dir = args.cache_dir or (args.hf_root.parent / "targets")
    if args.figure_only:
        results = json.loads(args.summary.read_text())["results"]
    else:
        results = analyze(args.hf_root, cache_dir, args.arms, args.corpora)
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(
            json.dumps(
                {
                    "provenance": {
                        "task": 2546,
                        "hf_repo": "superkaiba1/explore-persona-space-data",
                        "hf_revision": HF_REVISION,
                        "source_ref": SOURCE_REF,
                        "method": "committed five-fold out-of-fold predictions of the whole-corpus map, evaluated inside each necessity label group; no refit",
                        "r2_own_mean": "1 - SSE / SST with SST around the group's own mean answer state",
                        "r2_corpus_mean": "1 - SSE / SST with SST around the whole-corpus mean answer state",
                        "ci": f"95% percentile over {N_BOOT} question-level bootstrap draws inside the group, seed {SEED}",
                    },
                    "git": _git_state(),
                    "results": results,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        print(f"summary: {args.summary}")
    if args.no_figure:
        return 0
    font = set_c2a_style()
    fig = make_figure(results, {"1": args.title_a, "3": args.title_b}, arms=("1", "3") if args.both_panels else ("1",), whole_corpus_line=args.whole_corpus_line, baseline=args.baseline)
    stem = args.out_dir / args.stem
    outputs = save_c2a_figure(
        fig,
        stem,
        title="Within-corpus R^2 by CoT necessity",
        subject="Issue #2546 out-of-fold predictions grouped by operational necessity label",
        creator="scripts/section45_cot_necessity_r2_figure.py",
    )
    plt.close(fig)
    sidecar = stem.with_name(f"{args.stem}_data.json")
    sidecar.write_text(json.dumps({"style_version": STYLE_VERSION, "font": font, "git": _git_state(), "summary": str(args.summary.relative_to(ROOT)), "results": results, "outputs": {k: str(v.relative_to(ROOT)) for k, v in outputs.items()}}, indent=2, sort_keys=True) + "\n")
    for key, path in {**outputs, "data": sidecar}.items():
        print(f"{key}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
