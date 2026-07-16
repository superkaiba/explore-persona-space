"""Issue #779 inline free-analysis: does the pre-generation read predict
hallucination on GENERIC prompts?

Ask (chat 2026-07-14): "do we have evaluation on generic prompts?" — for the
one trait with real variance on generic LMSYS prompts (hallucination: 24.6% of
4593 judged prompts score >50; evil/sycophancy too sparse). Test whether the
pre-generation context read predicts the per-context LMSYS hallucination judge
label at the frozen read-out layer (L17):

  (a) pv_raw  = <c_last, r_B>         — the original Persona Vectors method (no fit)
  (b) map     = <h(c_last), r_B>      — the learned context->answer map read (5-fold CV)
  (c) probe_g = ridge(c_last -> label)— direct supervised probe (5-fold CV)

Metrics: Pearson + Spearman vs the label over judged contexts, 2000-draw
bootstrap CIs; a read-decile mean-label curve (the noise-robust ranking view —
the group-averaging analog for non-persona-structured generic prompts); and
AUROC on the >50 binary. Contextualizes #1092's within-LMSYS hallucination
probe floor (cross-validated r 0.009).

Reuses arm_headline GramRidge + loaders verbatim. pass_b staged locally.

CAVEAT: the LMSYS labels are 1 rollout x 5 judge draws, stored as the mean only
(no per-draw), so neither the generation-noise nor the judge-reliability ceiling
is estimable from this file. Raw r + decile view; no ceiling normalization.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import issue779_arm_headline as AH  # noqa: E402
import issue779_common as C  # noqa: E402
import issue779_stage1 as S1  # noqa: E402
import numpy as np  # noqa: E402

from explore_persona_space.experiments.issue_779 import fit_h as F  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent
TRAIT = "hallucination"
N_BOOT = 2000
N_FOLDS = 5
SEED = 0
POS_THRESH = 50.0


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def boot_ci(x: np.ndarray, y: np.ndarray, fn, n_boot: int, seed: int) -> dict:
    fin = np.isfinite(x) & np.isfinite(y)
    x, y = x[fin], y[fin]
    point = fn(x, y)
    rng = np.random.default_rng(seed)
    b = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(x), size=len(x))
        r = fn(x[idx], y[idx])
        if np.isfinite(r):
            b.append(r)
    return {
        "point": float(point),
        "lo": float(np.quantile(b, 0.025)),
        "hi": float(np.quantile(b, 0.975)),
        "n": len(x),
    }


def auroc(scores: np.ndarray, labels_bin: np.ndarray) -> float:
    # rank-based AUROC (Mann-Whitney).
    fin = np.isfinite(scores) & np.isfinite(labels_bin)
    s, y = scores[fin], labels_bin[fin]
    n_pos = int(y.sum())
    n_neg = int((1 - y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s)
    ranks = np.empty(len(s))
    ranks[order] = np.arange(1, len(s) + 1)
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def decile_means(read: np.ndarray, label: np.ndarray, n_bins: int = 10) -> dict:
    fin = np.isfinite(read) & np.isfinite(label)
    r, y = read[fin], label[fin]
    order = np.argsort(r)
    bins = np.array_split(order, n_bins)
    means = [float(np.mean(y[b])) for b in bins]
    rates = [float(np.mean(y[b] > POS_THRESH)) for b in bins]
    return {
        "bin_mean_label": means,
        "bin_pos_rate": rates,
        "top_minus_bottom_mean": means[-1] - means[0],
        "top_minus_bottom_rate": rates[-1] - rates[0],
    }


def cv_predict_map(X: np.ndarray, Y: np.ndarray, rb_l: np.ndarray, seed: int) -> np.ndarray:
    """5-fold held-out <h(c), r_B>, h = GramRidge context->answer profile."""
    n = len(X)
    pred = np.full(n, np.nan)
    rng = np.random.default_rng(seed)
    folds = np.array_split(rng.permutation(n), N_FOLDS)
    for f in folds:
        mask = np.ones(n, dtype=bool)
        mask[f] = False
        gr = AH.GramRidge(X[mask])
        prof = gr.predict(Y[mask], X[f])
        pred[f] = F.dot_readout(prof, rb_l)
    return pred


def cv_predict_probe(X: np.ndarray, y: np.ndarray, seed: int) -> np.ndarray:
    """5-fold held-out direct ridge c->label (labeled rows only in train)."""
    n = len(X)
    pred = np.full(n, np.nan)
    rng = np.random.default_rng(seed)
    folds = np.array_split(rng.permutation(n), N_FOLDS)
    for f in folds:
        mask = np.ones(n, dtype=bool)
        mask[f] = False
        fin = np.isfinite(y) & mask
        gr = AH.GramRidge(X[fin])
        pred[f] = gr.predict(y[fin][:, None], X[f]).ravel()
    return pred


def main() -> int:
    li = AH.FROZEN_LAYERS[TRAIT]["system"]  # L17
    bundle = AH.load_lmsys_bundle()
    layers = list(bundle["layers"])
    col = layers.index(li)
    X = bundle["cx_last"][:, col, :].float().numpy().astype(np.float64)
    Y = bundle["v_x"][:, col, :].float().numpy().astype(np.float64)
    n = X.shape[0]
    rb = S1._load_rb(AH.COLLECT_DIR / "r_b", TRAIT, C.EXPECTED_LAYERS, C.EXPECTED_HIDDEN)
    rb_l = np.asarray(rb[li], dtype=np.float64)
    label = AH.lmsys_labels(TRAIT, n)  # per-context mean judge score, NaN=dropped
    fin = np.isfinite(label)
    label_bin = np.where(fin, (label > POS_THRESH).astype(float), np.nan)

    reads = {
        "pv_raw": X @ rb_l,
        "map_cv": cv_predict_map(X, Y, rb_l, SEED),
        "probe_g_cv": cv_predict_probe(X, label, SEED),
    }

    res = {
        "trait": TRAIT,
        "layer": int(li),
        "n_total": int(n),
        "n_judged": int(fin.sum()),
        "n_positive_gt50": int(np.nansum(label_bin)),
        "pos_rate": float(np.nanmean(label_bin)),
        "label_mean": float(np.mean(label[fin])),
        "label_std": float(np.std(label[fin])),
        "reads": {},
        "caveat": (
            "labels = 1 rollout x 5 judge draws, mean-only (no per-draw); "
            "neither generation-noise nor judge-reliability ceiling estimable here. "
            "map_cv and probe_g_cv are 5-fold held-out; pv_raw needs no fit."
        ),
    }
    for name, r in reads.items():
        res["reads"][name] = {
            "pearson": boot_ci(r, label, lambda a, b: float(np.corrcoef(a, b)[0, 1]), N_BOOT, SEED),
            "spearman": boot_ci(r, label, spearman, N_BOOT, SEED),
            "auroc_gt50": auroc(r, label_bin),
            "deciles": decile_means(r, label),
        }
        p = res["reads"][name]["pearson"]
        s = res["reads"][name]["spearman"]
        d = res["reads"][name]["deciles"]
        print(
            f"{name:12s} pearson {p['point']:+.3f} [{p['lo']:+.3f},{p['hi']:+.3f}] | "
            f"spearman {s['point']:+.3f} | AUROC {res['reads'][name]['auroc_gt50']:.3f} | "
            f"decile mean-label {d['bin_mean_label'][0]:.1f}->{d['bin_mean_label'][-1]:.1f} "
            f"(top-bottom {d['top_minus_bottom_mean']:+.1f}); "
            f"pos-rate {d['bin_pos_rate'][0]:.2f}->{d['bin_pos_rate'][-1]:.2f}"
        )

    out = OUT_DIR / "generic_prompt_hallucination_monitor.json"
    with open(out, "w") as f:
        json.dump(res, f, indent=1)
    print("wrote", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
