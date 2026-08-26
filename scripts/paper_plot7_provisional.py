"""Provisional paper Plot 7 (relationship to CoT) for the Qwen2.5-7B lineage.

PROVISIONAL, a stand-in until #2546 runs Plot 7 properly.

Two models, one lineage. OpenThinker2-7B is a full fine-tune of Qwen2.5-7B-Instruct on
OpenThoughts2-1M (HF `base_model:finetune:Qwen/Qwen2.5-7B-Instruct`), so the only thing
separating them is reasoning SFT. That makes the pre model a genuine BEFORE bar rather
than just another model, and it is the same pair Plot 8 uses.

WHY THIS SCRIPT REFITS INSTEAD OF READING BANKED SKILLS. The obvious cheap build reads
#928's `recon_skill_grid.json` for the post model and #722's `per_position_vC_skill.json`
for the pre model. Those two numbers are NOT on a common scale: at the same n=50
query-averaged grain and the same layers, #928's context-to-answer cell reads 0.10 at
layer 14 and 0.12 at layer 18 while #722 reports 0.73 and 0.78. Same construct, same
corpus, roughly 6x apart, so the gap is the two scripts' estimator pipelines rather than
CoT training. Pairing them would invent a finding. Everything needed to refit both sides
through ONE estimator is banked as small per-context tensors, so this script does that.

ESTIMATOR (identical on both sides, and leakage-free by construction):
  grain   50 contexts, query-averaged (each row is one context, averaged over probes)
  folds   leave-one-context-out, 50 folds
  input   PCA-reduced to k components, PCA fit on the TRAINING FOLD ONLY
  target  full 3,584-dim residual state, NO target PCA
  metric  held-out skill-over-mean R2 against the training-fold mean

The input PCA is not cosmetic. At n_train=49 against d=3,584 a full-width fit is
estimator-degenerate and its held-out R2 is not a signal read (#1701), which is exactly
the regime both parent scripts fit in. Dropping the target PCA removes the other half:
both parents built a PCA-48 target basis on all 50 rows, so the held-out row helped define
its own basis.

CONSEQUENCE FOR THE READER: these numbers will not match the banked #928 or #722 values,
by design. They are internally consistent with each other, which is what a two-bar
comparison needs and what neither banked source could give.

Cells. The pre model emits no <think> block, so context-to-CoT, context-to-CoT+answer, and
CoT-to-answer are structurally impossible for it, not merely unmeasured. It contributes the
context-to-answer bar alone.

Each model is read at its OWN best layer for the context-to-answer arm, the parent's
frozen-layer convention. Layer indices are recorded in the results JSON and the figure
sidecar, and are deliberately absent from the figure itself.

Usage:
    uv run python scripts/paper_plot7_provisional.py
    uv run python scripts/paper_plot7_provisional.py --k 24
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()  # BEFORE heavy imports — shared-VM thread caps bind in-process (#847)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from huggingface_hub import hf_hub_download  # noqa: E402

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    paper_palette,
    savefig_paper,
    set_paper_style,
)
from explore_persona_space.orchestrate.hub import retry_transient  # noqa: E402

DATA_REPO = "superkaiba1/explore-persona-space-data"

# Pre model (Qwen2.5-7B-Instruct): query-averaged context vectors + answer-side summaries.
PRE_CTX_PATH = "issue594_context_geometry/analysis_tensors/context_vectors_mean.pt"
PRE_ANS_PATH = "issue658_theory_assumptions/store/v0_summaries.pt"
# Post model (OpenThinker2-7B): per-(context, probe) summaries, reduced over probes here.
POST_DIR = "issue928_cot_decomposition/analysis_tensors/store/percq_summaries"
POST_MANIFEST = f"{POST_DIR}/manifest.json"

PRE_MODEL = "Qwen2.5-7B-Instruct"
POST_MODEL = "OpenThinker2-7B"

# The pre-side context vector is the last context token, averaged over probes; `ctx_last`
# is the same boundary token on the post side, so the input definition is matched.
POST_CTX_SUMMARY = "ctx_last"

# Post-model cells: (target summary, reader-facing label). Order is the plotting order.
POST_CELLS: tuple[tuple[str, str], ...] = (
    ("ans_mean", "context to answer"),
    ("cot_mean", "context to CoT"),
    ("joint", "context to CoT + answer"),
    ("cot_to_ans", "CoT to answer"),
)
PRE_CELL_LABEL = "context to answer"

HEADLINE_K = 16  # n_train=49, so k=16 is about 3 training rows per parameter
LAMBDAS = np.logspace(-3, 6, 10)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--k", type=int, default=HEADLINE_K, help="PCA components on the input.")
    p.add_argument("--out-dir", default="figures/paper")
    p.add_argument("--results", default="eval_results/issue_2546/plot7_provisional.json")
    return p.parse_args(argv)


def _dl(path_in_repo: str) -> str:
    return retry_transient(
        lambda: hf_hub_download(DATA_REPO, path_in_repo, repo_type="dataset"),
        what=f"download {path_in_repo}",
    )


def load_pre(ctx_ids: list[str]) -> dict[str, np.ndarray]:
    """Pre-model context vectors and answer summaries, aligned to `ctx_ids`."""
    cv = torch.load(_dl(PRE_CTX_PATH), map_location="cpu", weights_only=False)
    v0 = torch.load(_dl(PRE_ANS_PATH), map_location="cpu", weights_only=False)
    assert cv["probe_pool_hash"] == v0["probe_pool_hash"], "pre-side probe pools differ"

    ctx_by_id = dict(zip(list(cv["instance_ids"]), cv["tensor"].float().numpy(), strict=True))
    ans_src = v0["summaries"]["mean"]
    if isinstance(ans_src, dict):
        ans_by_id = {k: np.asarray(t, dtype=np.float32) for k, t in ans_src.items()}
    else:  # a stacked tensor keyed by the manifest's own context order
        ans_by_id = dict(
            zip(list(v0["context_ids"]), np.asarray(ans_src, dtype=np.float32), strict=True)
        )

    missing = [c for c in ctx_ids if c not in ctx_by_id or c not in ans_by_id]
    assert not missing, f"pre store is missing contexts: {missing}"
    return {
        "ctx": np.stack([ctx_by_id[c] for c in ctx_ids]),
        "ans_mean": np.stack([ans_by_id[c] for c in ctx_ids]),
    }


def load_post(ctx_ids: list[str]) -> dict[str, np.ndarray]:
    """Post-model summaries reduced to one row per context (mean over kept probes)."""
    want = (POST_CTX_SUMMARY, "ans_mean", "cot_mean")
    acc: dict[str, list[np.ndarray]] = {w: [] for w in want}
    kept: list[int] = []

    for cid in ctx_ids:
        qb = torch.load(_dl(f"{POST_DIR}/{cid}.pt"), map_location="cpu", weights_only=False)
        names = list(qb["summary_names"])
        per_q = qb["per_q"]  # (n_kept, n_summaries, n_layers, hidden)
        kept.append(int(per_q.shape[0]))
        for w in want:
            acc[w].append(per_q[:, names.index(w)].float().numpy().mean(axis=0))

    out = {w: np.stack(v) for w, v in acc.items()}
    out["_kept_probes"] = np.asarray(kept, dtype=int)
    return out


def pca_project(train: np.ndarray, other: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Fit PCA on the TRAINING rows only, project both blocks into it.

    With n rows far below d dimensions the n-by-n Gram carries the whole row space, so the
    basis comes from one tiny eigendecomposition instead of a d-wide SVD. The components
    are never materialised: the test block is projected through the training rows.
    """
    mu = train.mean(axis=0, keepdims=True)
    xc = train - mu
    k_eff = min(k, min(train.shape) - 1)

    evals, evecs = np.linalg.eigh(xc @ xc.T)
    top = np.argsort(evals)[::-1][:k_eff]
    s = np.sqrt(np.clip(evals[top], 0.0, None))
    u = evecs[:, top]

    keep = s > (s.max() * 1e-10 if s.size and s.max() > 0 else 0.0)
    u, s = u[:, keep], s[keep]
    if s.size == 0:  # degenerate training block: fall back to the mean-only predictor
        return np.zeros((train.shape[0], 1)), np.zeros((other.shape[0], 1))

    # scores = U*s for train; (other-mu) @ V^T with V = Xc^T U / s for the held-out block.
    return u * s, ((other - mu) @ xc.T @ u) / s


def _ridge(xtr: np.ndarray, ytr: np.ndarray, xte: np.ndarray) -> np.ndarray:
    """Ridge on PCA-reduced inputs; lambda by GCV with a degrees-of-freedom cap (#1887)."""
    n, k = xtr.shape
    xm, ym = xtr.mean(0), ytr.mean(0)
    xc, yc = xtr - xm, ytr - ym
    gram, rhs, eye = xc.T @ xc, xc.T @ yc, np.eye(k)
    evals = np.linalg.eigvalsh(gram)

    best_lam, best_err = LAMBDAS[-1], np.inf
    for lam in LAMBDAS:
        dof = float((evals / (evals + lam)).sum())
        if dof >= 0.9 * n:  # never let GCV pick an effectively unregularized fit
            continue
        resid = yc - xc @ np.linalg.solve(gram + lam * eye, rhs)
        err = float((resid**2).sum()) / (1.0 - dof / n) ** 2
        if err < best_err:
            best_err, best_lam = err, lam

    return (xte - xm) @ np.linalg.solve(gram + best_lam * eye, rhs) + ym


def loco_skill(x: np.ndarray, y: np.ndarray, k: int) -> float:
    """Leave-one-context-out held-out skill-over-mean R2 against the training-fold mean."""
    n = x.shape[0]
    sse = sst = 0.0
    for i in range(n):
        tr = np.arange(n) != i
        xtr_p, xte_p = pca_project(x[tr], x[i : i + 1], k)
        pred = _ridge(xtr_p, y[tr], xte_p)[0]
        sse += float(((y[i] - pred) ** 2).sum())
        sst += float(((y[i] - y[tr].mean(axis=0)) ** 2).sum())
    return 1.0 - sse / sst


def layer_curve(x: np.ndarray, y: np.ndarray, k: int, tag: str) -> np.ndarray:
    """Skill at every captured layer, with the input read at the same layer as the target."""
    t0 = time.time()
    out = np.asarray([loco_skill(x[:, ell], y[:, ell], k) for ell in range(x.shape[1])])
    print(f"  [{tag}] {x.shape[1]} layers in {time.time() - t0:.1f}s, best={out.max():.3f}")
    return out


def build_records(pre: dict, post: dict, k: int) -> list[dict]:
    """One record per model: per-cell layer curves plus the model's own best-D layer."""
    pre_curves = {PRE_CELL_LABEL: layer_curve(pre["ctx"], pre["ans_mean"], k, "pre ctx->ans")}

    ctx = post[POST_CTX_SUMMARY]
    targets = {
        "context to answer": post["ans_mean"],
        "context to CoT": post["cot_mean"],
        # The joint target concatenates the two spans, so a single map must carry both.
        "context to CoT + answer": np.concatenate([post["cot_mean"], post["ans_mean"]], axis=-1),
    }
    post_curves = {
        label: layer_curve(ctx, tgt, k, f"post {label}") for label, tgt in targets.items()
    }
    post_curves["CoT to answer"] = layer_curve(
        post["cot_mean"], post["ans_mean"], k, "post CoT to answer"
    )

    records = []
    for model, curves in ((PRE_MODEL, pre_curves), (POST_MODEL, post_curves)):
        best = int(np.argmax(curves["context to answer"]))
        records.append(
            {
                "model": model,
                "best_layer": best,
                "cells": {label: float(c[best]) for label, c in curves.items()},
                "curves": {label: c.tolist() for label, c in curves.items()},
            }
        )
    return records


def figure(records: list[dict], out_dir: Path) -> Path:
    labels = [lab for _, lab in POST_CELLS]
    colors = dict(zip(labels, paper_palette(len(labels)), strict=True))
    x = np.arange(len(records), dtype=float)
    width = 0.8 / len(labels)

    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for i, label in enumerate(labels):
        offs, vals = [], []
        for xi, rec in zip(x, records, strict=True):
            if label in rec["cells"]:
                offs.append(xi + (i - (len(labels) - 1) / 2) * width)
                vals.append(rec["cells"][label])
        ax.bar(offs, vals, width=width, color=colors[label], label=label, zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{records[0]['model']}\n(before CoT training)", f"{records[1]['model']}\n(CoT-trained)"]
    )
    ax.set_ylabel("held-out skill (R$^2$ over mean)")
    ax.axhline(0.0, color="0.6", lw=0.8, zorder=2)
    lo = min(min(r["cells"].values()) for r in records)
    hi = max(max(r["cells"].values()) for r in records)
    ax.set_ylim(bottom=min(0.0, lo - 0.05), top=hi + 0.08)
    ax.legend(frameon=False, ncol=4, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.14))
    fig.tight_layout()

    paths = savefig_paper(fig, "plot7_provisional_cells", dir=out_dir)
    plt.close(fig)
    return paths.get("png", out_dir / "plot7_provisional_cells.png")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    set_paper_style()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    man = json.loads(Path(_dl(POST_MANIFEST)).read_text())
    ctx_ids = sorted(man["context_ids"])
    pre, post = load_pre(ctx_ids), load_post(ctx_ids)
    assert pre["ctx"].shape == post[POST_CTX_SUMMARY].shape, (
        pre["ctx"].shape,
        post[POST_CTX_SUMMARY].shape,
    )
    n, n_layers, hidden = pre["ctx"].shape
    print(
        f"n={n} contexts, layers={n_layers}, hidden={hidden}, k={args.k}, "
        f"n_train/fold={n - 1}; post-side probes kept per context: "
        f"min={post['_kept_probes'].min()} max={post['_kept_probes'].max()}"
    )

    records = build_records(pre, post, args.k)
    for rec in records:
        cells = " | ".join(f"{lab}={v:.3f}" for lab, v in rec["cells"].items())
        print(f"{rec['model']} (best layer {rec['best_layer']}): {cells}")

    res = {
        "dv": "held-out skill-over-mean R2, leave-one-context-out over 50 contexts",
        "estimator": (
            f"input PCA k={args.k} fit on the training fold only; ridge with GCV lambda "
            "under a 0.9n degrees-of-freedom cap; full-dimensional target, no target PCA"
        ),
        "why_refit": (
            "Banked #928 and #722 skills are not on a common scale (0.10-0.12 vs 0.73-0.78 "
            "for the same cell, grain and layers), so both sides are refit here through one "
            "estimator. These values therefore do not match either banked source."
        ),
        "grain_note": (
            "Pre-side rows average all 48 probes; post-side rows average only the probes that "
            "survived #928's CoT parsing (see kept_probes_per_context)."
        ),
        "n_contexts": n,
        "k": args.k,
        "kept_probes_per_context": post["_kept_probes"].tolist(),
        "models": records,
    }
    out = Path(args.results)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2) + "\n")

    print(f"wrote {figure(records, out_dir)}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
