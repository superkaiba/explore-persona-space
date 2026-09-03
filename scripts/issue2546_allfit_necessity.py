"""Fit every Section 4.5 map on ALL questions, then score it on CoT-necessary and CoT-unnecessary questions.

Rows: every question of the seven captured benchmarks (MATH, GSM8K train, ContextHub, MMLU, ARC-Challenge,
CommonsenseQA, PIQA) with states on every side the arm needs (arm 1: post and pre; arm 3: think_on and think_off;
arm 2: post only). Ridge follows the production recipe (standardize X on train, center Y, primal ridge, float64,
five seeded random-row folds), with lambda fixed to the value the production stratum cell selected at the headline
layer. Metrics per subset (all / necessary / both_correct): held-out R^2 with a global-mean baseline (as in the
production cells) and with a corpus-mean baseline (as in the necessity figure), own-answer acc@1 under the paper
recipe (train-fold whitening, cosine + CSLS k=10, pool = the held-out fold's answers), 1,000-draw question bootstraps.

Usage:
  allfit_necessity.py extract <arm>            # cache the nine in-trace states at the headline layer from the shards
  allfit_necessity.py fit <arm> [cells...]     # default: every non-trajectory cell of the arm (+ the ladder for arm 1)
  allfit_necessity.py traj <arm>               # the nine in-trace maps (needs extract first)
"""

import glob
import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy.linalg import solve_triangular  # noqa: E402

sys.path.insert(0, "/home/thomasjiralerspong/explore-persona-space/src")
from explore_persona_space.analysis.null_battery import (  # noqa: E402
    PRIMARY_LAMBDA,
    shrunk_cholesky_from_cov,
)

BASE = Path("/mnt/eps-data/thomasjiralerspong/cot_necessity")
HF = BASE / "hf" / "issue2546_cotmap"
TG = BASE / "hf" / "targets"
OUT = BASE / "allfit"
CELLS_PROD = Path("/home/thomasjiralerspong/explore-persona-space/eval_results/issue_2546/cells")
for d in ("preds", "results"):
    (OUT / d).mkdir(parents=True, exist_ok=True)
LAYER = {1: 19, 2: 19, 3: 24}
SIDES = {
    1: {"post": "post", "short": "pre"},
    2: {"post": "post"},
    3: {"post": "think_on", "short": "think_off"},
}
LABELS = {
    1: HF / "eval_results_mirror/out/necessity/pair_necessity_a1.json",
    3: HF / "eval_results_mirror/out/necessity/qwen3_toggle_labels.json",
}
CORPORA = ("math", "gsm8k_train", "contexthub", "mmlu", "arc_challenge", "csqa", "piqa")
T_POS = [f"think_t{t}" for t in range(10, 100, 10)]
# cell -> (x_side, x_kind, y_side, y_kind); sides are the production names (post / short)
CELLS = {
    "p7_A": ("post", "cx_last", "post", "ans_mean"),
    "p7_B": ("post", "cx_last", "post", "cot_mean"),
    "p7_C": ("post", "cx_last", "post", "out_mean"),
    "p7_D": ("post", "cot_boundary", "post", "ans_mean"),
    "p7_Aoff": ("short", "cx_last", "short", "ans_mean"),
    "p8_E": ("short", "cx_last", "post", "ans_mean"),
    "p8_G": ("short", "cx_last", "short", "ans_mean"),
    "p8_H": ("short", "cx_last", "post", "cot_mean"),
}
ARM_CELLS = {
    1: ["p7_A", "p7_B", "p7_C", "p7_D", "p8_E", "p8_G", "p8_H"],
    2: ["p7_A", "p7_B", "p7_C", "p7_D"],
    3: ["p7_A", "p7_Aoff", "p7_B", "p7_C", "p7_D"],
}
N_FOLDS, SEED, N_BOOT, K_CSLS = 5, 0, 1000, 10
torch.set_num_threads(int(os.environ.get("THREADS", "8")))
mode, arm = sys.argv[1], int(sys.argv[2])
log = open(OUT / f"log_{mode}_arm{arm}.txt", "a")


def say(*a):
    msg = " ".join(str(x) for x in a)
    print(msg, flush=True)
    log.write(msg + "\n")
    log.flush()


def cv_folds(ids, n_folds, seed):
    uniq = np.unique(np.asarray(ids))
    perm = np.random.default_rng(seed).permutation(len(uniq))
    fold_of = {c: int(perm[i] % n_folds) for i, c in enumerate(uniq)}
    return np.array([fold_of[c] for c in ids])


def cache_path(side, kind, corpus):
    return TG / f"{kind}__arm{arm}__{SIDES[arm][side]}__{corpus}__l{LAYER[arm]}.npz"


def load_kind(side, kind):
    out = {}
    for c in CORPORA:
        p = cache_path(side, kind, c)
        if not p.is_file():
            say(f"WARN missing cache {p.name}")
            continue
        z = np.load(p)
        for r, v in zip(z["row_ids"], z[kind]):
            out[str(r)] = v
    return out


def production_lambda(cell):
    """Lambda the production stratum cell selected at the headline layer (fold-0 value)."""
    path = CELLS_PROD / f"{cell}__does__a{arm}.json"
    if cell == "p7_traj" or not path.is_file():
        path = CELLS_PROD / f"p7_A__does__a{arm}.json"
    d = json.loads(path.read_text())
    sel = d["lambda_diag"]["selected"]
    li = (
        [int(v) for v in d["frozen_layers"]].index(LAYER[arm])
        if len(sel) == len(d["frozen_layers"])
        else LAYER[arm]
    )
    return float(sel[li][0])


class Ridge:
    def __init__(self, X, Y, lam):
        self.xmu, self.xsd = X.mean(0), X.std(0) + 1e-9
        self.ymu = Y.mean(0)
        Xn = (X - self.xmu) / self.xsd
        A = Xn.T @ Xn
        A.diagonal().add_(lam)
        L = torch.linalg.cholesky(A)
        self.B = torch.cholesky_solve(Xn.T @ (Y - self.ymu), L)

    def predict(self, Xe):
        return ((Xe - self.xmu) / self.xsd) @ self.B + self.ymu


def matrix(d, ids):
    return torch.as_tensor(np.stack([d[r] for r in ids]).astype(np.float64))


def csls(S, k=K_CSLS):
    nq, npool = S.shape
    r_q = np.partition(S, npool - k, axis=1)[:, npool - k :].mean(1)
    r_p = np.partition(S, nq - k, axis=0)[nq - k :, :].mean(0)
    return 2.0 * S - r_q[:, None] - r_p[None, :]


unit = lambda a: a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)  # noqa: E731
whiten = lambda x, mu, ell: solve_triangular(ell, (x - mu).T, lower=True).T  # noqa: E731


def retrieval_hits(pred, Y, folds):
    pred, Y = pred.numpy(), Y.numpy()
    hits = np.zeros(len(Y), bool)
    pools = []
    for k in np.unique(folds):
        tr = folds != k
        te = np.where(folds == k)[0]
        mu = Y[tr].mean(0)
        ell = shrunk_cholesky_from_cov(np.cov(Y[tr], rowvar=False), PRIMARY_LAMBDA)
        zq, zP = whiten(pred[te], mu, ell), whiten(Y[te], mu, ell)
        hits[te] = csls(unit(zq) @ unit(zP).T).argmax(1) == np.arange(len(te))
        pools.append(len(te))
    return hits, float(np.mean([1.0 / p for p in pools]))


def boot_ratio(num, den, rng):
    """Bootstrap of 1 - sum(num)/sum(den) over questions."""
    n = len(num)
    w = rng.multinomial(n, np.full(n, 1.0 / n), size=N_BOOT).astype(float)
    draws = 1.0 - (w @ num) / (w @ den)
    return [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]


def boot_mean(h, rng):
    n = len(h)
    w = rng.multinomial(n, np.full(n, 1.0 / n), size=N_BOOT)
    m = (w @ h.astype(float)) / n
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


class Ctx:
    """Row universe, folds, labels and baselines shared by every cell of the arm."""

    def __init__(self):
        sides = list(SIDES[arm])
        kinds = {("post", "cx_last"), ("post", "ans_mean")}
        if "short" in sides:
            kinds |= {("short", "cx_last"), ("short", "ans_mean")}
        rowsets = [set(load_kind(s, k)) for s, k in kinds]
        self.ids = sorted(set.intersection(*rowsets))
        self.corpus = np.array([r.split(":")[0] for r in self.ids])
        self.folds = cv_folds(self.ids, N_FOLDS, SEED)
        labels = json.loads(LABELS[arm].read_text())["labels"] if arm in LABELS else {}
        self.label = np.array([labels.get(r, "unlabeled") for r in self.ids])
        self.subsets = {"all": np.ones(len(self.ids), bool)}
        if arm in LABELS:
            self.subsets["necessary"] = self.label == "necessary"
            self.subsets["both_correct"] = self.label == "both_correct"
        say(
            f"arm{arm}: {len(self.ids)} rows; folds {np.bincount(self.folds).tolist()}; subsets "
            + ", ".join(f"{k}={int(v.sum())}" for k, v in self.subsets.items())
        )

    def baselines(self, Y):
        """Per-question baselines: global train-fold mean and corpus train-fold mean (out-of-fold)."""
        g = torch.zeros_like(Y)
        c = torch.zeros_like(Y)
        for k in range(N_FOLDS):
            tr, te = self.folds != k, self.folds == k
            g[te] = Y[tr].mean(0)
            for corp in np.unique(self.corpus[te]):
                m = tr & (self.corpus == corp)
                c[te & (self.corpus == corp)] = Y[m].mean(0)
        return g, c

    def score(self, pred, Y, retrieval=True):
        g, c = self.baselines(Y)
        e = ((Y - pred) ** 2).sum(1).numpy()
        tg = ((Y - g) ** 2).sum(1).numpy()
        tc = ((Y - c) ** 2).sum(1).numpy()
        hits, chance = retrieval_hits(pred, Y, self.folds) if retrieval else (None, None)
        out = {}
        for name, m in self.subsets.items():
            rng = np.random.default_rng(1)
            if m.sum() < 20:
                out[name] = {"n": int(m.sum()), "skipped": "fewer than 20 rows"}
                continue
            blk = {
                "n": int(m.sum()),
                "r2_global": 1.0 - float(e[m].sum() / tg[m].sum()),
                "r2_global_ci": boot_ratio(e[m], tg[m], rng),
                "r2_corpus": 1.0 - float(e[m].sum() / tc[m].sum()),
                "r2_corpus_ci": boot_ratio(e[m], tc[m], rng),
                "per_corpus": {},
            }
            if hits is not None:
                blk.update(
                    {
                        "acc1": float(hits[m].mean()),
                        "acc1_ci": boot_mean(hits[m], rng),
                        "chance": chance,
                    }
                )
            for corp in CORPORA:
                mc = m & (self.corpus == corp)
                if mc.sum() >= 20:
                    blk["per_corpus"][corp] = {
                        "n": int(mc.sum()),
                        "r2_corpus": 1.0 - float(e[mc].sum() / tc[mc].sum()),
                        "r2_corpus_ci": boot_ratio(e[mc], tc[mc], rng),
                    }
                    if hits is not None:
                        blk["per_corpus"][corp]["acc1"] = float(hits[mc].mean())
            out[name] = blk
        return out, hits

    def fit_oof(self, X, Y, lam, tag):
        pred = torch.zeros_like(Y)
        models = []
        for k in range(N_FOLDS):
            tr, te = self.folds != k, self.folds == k
            t0 = time.time()
            m = Ridge(X[tr], Y[tr], lam)
            pred[te] = m.predict(X[te])
            models.append(m)
            say(f"  {tag} fold {k}: lambda={lam:.3g} ({time.time() - t0:.0f}s)")
        return pred, models


def write(name, payload):
    """Atomically write one result JSON (temp file in the same directory, then replace)."""
    from explore_persona_space.atomic_io import atomic_replace

    with atomic_replace(OUT / "results" / f"{name}.json") as tmp:
        tmp.write_text(json.dumps(payload, indent=1))


def run_fit(cells):
    ctx = Ctx()
    for cell in cells:
        xs, xk, ys, yk = CELLS[cell]
        X, Y = matrix(load_kind(xs, xk), ctx.ids), matrix(load_kind(ys, yk), ctx.ids)
        lam = production_lambda(cell)
        pred, models = ctx.fit_oof(X, Y, lam, f"arm{arm}/{cell}")
        scores, hits = ctx.score(pred, Y)
        np.savez(
            OUT / "preds" / f"{cell}__all__a{arm}.npz",
            fitted_mask=np.ones(len(ctx.ids), bool),
            conv_ids=np.asarray(ctx.ids),
            folds=ctx.folds,
            labels=ctx.label,
            **{f"pred_l{LAYER[arm]}": pred.numpy().astype(np.float32)},
        )
        np.savez(
            OUT / "preds" / f"hits__{cell}__all__a{arm}.npz",
            row_ids=np.asarray(ctx.ids),
            folds=ctx.folds,
            labels=ctx.label,
            hit_whitened_csls=hits,
        )
        write(
            f"{cell}__a{arm}",
            {
                "arm": arm,
                "cell": cell,
                "x": [SIDES[arm][xs], xk],
                "y": [SIDES[arm][ys], yk],
                "layer": LAYER[arm],
                "lambda": lam,
                "n_rows": len(ctx.ids),
                "subsets": scores,
            },
        )
        say(
            f"  {cell}: "
            + " | ".join(
                f"{s}: R2g={v['r2_global']:.3f} R2c={v['r2_corpus']:.3f} acc@1={v.get('acc1', float('nan')):.3f} n={v['n']}"
                for s, v in scores.items()
                if "r2_global" in v
            )
        )
        if cell == "p8_G" and arm == 1:
            ladder(ctx, models, X, Y)


def ladder(ctx, pre_models, Xb, Yb):
    """Transfer the pre-SFT map (fit on pre rows, same folds) to the post-SFT states; tiers t0/t3/t4."""
    Xi, Yi = (
        matrix(load_kind("post", "cx_last"), ctx.ids),
        matrix(load_kind("post", "ans_mean"), ctx.ids),
    )
    post_pred, _ = ctx.fit_oof(Xi, Yi, production_lambda("p7_A"), "ladder/post_own")
    tiers = {
        t: torch.zeros_like(Yi)
        for t in ("t0_direct_transfer", "t3_bias_offset", "t4_global_scaling")
    }
    for k in range(N_FOLDS):
        tr, te = ctx.folds != k, ctx.folds == k
        m = pre_models[k]
        p_tr, p_te, y_tr = m.predict(Xi[tr]), m.predict(Xi[te]), Yi[tr]
        tiers["t0_direct_transfer"][te] = p_te
        tiers["t3_bias_offset"][te] = p_te + (y_tr - p_tr).mean(0)
        pc, yc = p_tr - p_tr.mean(0), y_tr - y_tr.mean(0)
        alpha = float((pc * yc).sum() / ((pc * pc).sum() + 1e-12))
        tiers["t4_global_scaling"][te] = alpha * (p_te - p_tr.mean(0)) + y_tr.mean(0)
    ref, _ = ctx.score(post_pred, Yi, retrieval=False)
    out = {"arm": 1, "n_rows": len(ctx.ids), "layer": LAYER[1], "post_own": ref, "tiers": {}}
    g, c = ctx.baselines(Yi)
    e_ref = ((Yi - post_pred) ** 2).sum(1).numpy()
    for t, p in tiers.items():
        sc, _ = ctx.score(p, Yi, retrieval=False)
        e = ((Yi - p) ** 2).sum(1).numpy()
        for s, blk in sc.items():
            if "r2_global" not in blk:
                continue
            m = ctx.subsets[s]
            rng = np.random.default_rng(2)
            n = int(m.sum())
            w = rng.multinomial(n, np.full(n, 1.0 / n), size=N_BOOT).astype(float)
            for base, tot in (
                ("global", ((Yi - g) ** 2).sum(1).numpy()),
                ("corpus", ((Yi - c) ** 2).sum(1).numpy()),
            ):
                r_ref = 1.0 - (w @ e_ref[m]) / (w @ tot[m])
                r_t = 1.0 - (w @ e[m]) / (w @ tot[m])
                blk[f"retention_{base}"] = blk[f"r2_{base}"] / ref[s][f"r2_{base}"]
                blk[f"retention_{base}_ci"] = [
                    float(np.percentile(r_t / r_ref, 2.5)),
                    float(np.percentile(r_t / r_ref, 97.5)),
                ]
        out["tiers"][t] = sc
        say(
            f"  ladder {t}: "
            + " | ".join(
                f"{s}: ret_g={v['retention_global']:.3g} ret_c={v['retention_corpus']:.3g}"
                for s, v in sc.items()
                if "retention_global" in v
            )
        )
    write("ladder__a1", out)


def run_extract():
    side = SIDES[arm]["post"]
    li = None
    for corpus in CORPORA:
        done = [cache_path("post", t, corpus) for t in T_POS]
        if all(p.is_file() for p in done):
            say(f"  {corpus}: cached")
            continue
        files = sorted(
            glob.glob(
                str(
                    HF
                    / "analysis_tensors"
                    / "thinkstore"
                    / f"arm{arm}"
                    / f"{side}__{corpus}"
                    / "slot*.shard*.pt"
                )
            )
        )
        ids, blocks = [], {t: [] for t in T_POS}
        t0 = time.time()
        for f in files:
            sh = torch.load(f, map_location="cpu", weights_only=False)
            if li is None:
                li = [int(v) for v in sh["frozen_layers"]].index(LAYER[arm])
                kt = list(sh["kinds_t"])
            ids.extend(str(r) for r in sh["row_ids"])
            for t in T_POS:
                blocks[t].append(sh["tk"][:, kt.index(t), li].to(torch.float16).numpy())
        for t in T_POS:
            np.savez(
                cache_path("post", t, corpus),
                row_ids=np.asarray(ids),
                **{t: np.concatenate(blocks[t])},
            )
        say(f"  {corpus}: {len(ids)} rows from {len(files)} shards ({time.time() - t0:.0f}s)")


def run_traj():
    ctx = Ctx()
    Y = matrix(load_kind("post", "ans_mean"), ctx.ids)
    lam = production_lambda("p7_traj")
    out = {
        "arm": arm,
        "cell": "p7_traj",
        "layer": LAYER[arm],
        "lambda": lam,
        "n_rows": len(ctx.ids),
        "positions": {},
    }
    for t in T_POS:
        X = matrix(load_kind("post", t), ctx.ids)
        pred, _ = ctx.fit_oof(X, Y, lam, f"arm{arm}/traj/{t}")
        scores, _ = ctx.score(pred, Y)
        out["positions"][t.replace("think_", "")] = scores
        say(
            f"  {t}: "
            + " | ".join(
                f"{s}: R2g={v['r2_global']:.3f} R2c={v['r2_corpus']:.3f} acc@1={v.get('acc1', float('nan')):.3f}"
                for s, v in scores.items()
                if "r2_global" in v
            )
        )
        write(f"p7_traj__a{arm}", out)


if mode == "extract":
    run_extract()
elif mode == "fit":
    run_fit(sys.argv[3:] or ARM_CELLS[arm])
elif mode == "traj":
    run_traj()
say(f"DONE {mode} arm{arm}")
