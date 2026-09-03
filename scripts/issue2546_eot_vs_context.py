"""Where does the end-of-thought map beat the context map? (OpenThinker3-7B, all-question fits)

Inputs: allfit preds for p7_A (context -> answer) and p7_D (end of thought -> answer), the true answer states,
necessity labels, and the raw completions (answer text = everything after the last </think>).

Analyses
  1. Per-direction R^2 along the principal directions of the answer state, for both maps, and the gain D - A.
  2. ANOVA-style split of the answer-state variance into between-corpus, between-answer-content (within corpus),
     and within-answer components; share of each component left unexplained by each map.
  3. Per-question squared-error gain by corpus, necessity label, answer type and correctness; the retrieval
     flip set (context miss, end-of-thought hit) and its composition, with examples.
  4. Answer-class decodability from the predicted states: out-of-fold nearest-centroid accuracy of the answer
     class (within corpus) from the context prediction, the end-of-thought prediction, and the true state.
"""

import json
import re
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

import numpy as np  # noqa: E402

BASE = Path("/mnt/eps-data/thomasjiralerspong/cot_necessity")
HF = BASE / "hf" / "issue2546_cotmap"
TG = BASE / "hf" / "targets"
OUT = BASE / "eot_vs_context"
OUT.mkdir(exist_ok=True)
ARM, LAYER, SIDE = 1, 19, "post"
CORPORA = ("math", "gsm8k_train", "contexthub", "mmlu", "arc_challenge", "csqa", "piqa")
MCQ = {"mmlu", "arc_challenge", "csqa", "piqa"}
log = open(OUT / "log.txt", "a")


def say(*a):
    msg = " ".join(str(x) for x in a)
    print(msg, flush=True)
    log.write(msg + "\n")
    log.flush()


def load_preds(cell):
    z = np.load(BASE / "allfit" / "preds" / f"{cell}__all__a{ARM}.npz")
    return (
        [str(r) for r in z["conv_ids"]],
        np.asarray(z["folds"]),
        np.asarray(z[f"pred_l{LAYER}"], np.float64),
        [str(v) for v in z["labels"]],
    )


def load_hits(cell):
    z = np.load(BASE / "allfit" / "preds" / f"hits__{cell}__all__a{ARM}.npz")
    return dict(zip([str(r) for r in z["row_ids"]], np.asarray(z["hit_whitened_csls"], bool)))


def load_targets(ids):
    d = {}
    for c in CORPORA:
        z = np.load(TG / f"ans_mean__arm{ARM}__{SIDE}__{c}__l{LAYER}.npz")
        for r, v in zip(z["row_ids"], z["ans_mean"]):
            d[str(r)] = v
    return np.stack([d[r] for r in ids]).astype(np.float64)


def answer_text(ids):
    out = {}
    for c in CORPORA:
        for line in open(HF / "raw_completions" / f"post_greedy_a{ARM}" / f"{c}.jsonl"):
            rec = json.loads(line)
            t = rec["text"]
            out[rec["row_id"]] = (
                t[t.rfind("</think>") + len("</think>") :].strip() if "</think>" in t else ""
            )
    return [out.get(r, "") for r in ids]


BOXED = re.compile(r"\\boxed\s*\{((?:[^{}]|\{[^{}]*\})*)\}")
LETTER = re.compile(
    r"(?:answer is|Answer:|answer:|\*\*Answer\*\*:?)\s*\(?([A-E])\)?|^\(?([A-E])\)?[.:]?\s*$", re.M
)


def answer_class(corpus, text):
    """Canonical answer content: boxed value (math corpora), option letter (MCQ), else the last short line."""
    b = BOXED.findall(text)
    if b:
        return b[-1].replace(" ", "")
    if corpus in MCQ:
        m = LETTER.findall(text)
        if m:
            last = m[-1]
            return (last[0] or last[1]).upper()
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    tail = lines[-1] if lines else ""
    num = re.findall(r"-?\d[\d,]*\.?\d*", tail)
    return num[-1].replace(",", "") if num else tail[:40]


def main():
    ids, folds, pA, labels = load_preds("p7_A")
    ids_d, _, pD, _ = load_preds("p7_D")
    assert ids == ids_d
    Y = load_targets(ids)
    corpus = np.array([r.split(":")[0] for r in ids])
    labels = np.array(labels)
    texts = answer_text(ids)
    cls = np.array([answer_class(c, t) for c, t in zip(corpus, texts)])
    hitA, hitD = load_hits("p7_A"), load_hits("p7_D")
    hA = np.array([hitA[r] for r in ids])
    hD = np.array([hitD[r] for r in ids])
    n = len(ids)
    say(f"{n} questions; answer classes parsed: {int((cls != '').sum())}")
    eA, eD = Y - pA, Y - pD
    res = {"n": n}

    # ---- 1. per-direction R^2 along answer-state PCs ----------------------------------------------------
    mu = Y.mean(0)
    Yc = Y - mu
    cov = Yc.T @ Yc / n
    evals, V = np.linalg.eigh(cov)
    order = np.argsort(evals)[::-1]
    evals, V = evals[order], V[:, order]
    tot = (Yc @ V) ** 2
    ssA = (eA @ V) ** 2
    ssD = (eD @ V) ** 2
    r2A = 1 - ssA.sum(0) / tot.sum(0)
    r2D = 1 - ssD.sum(0) / tot.sum(0)
    var_share = evals / evals.sum()
    gain_share = (ssA.sum(0) - ssD.sum(0)) / (
        ssA.sum() - ssD.sum()
    )  # share of the total SSE reduction carried by each PC
    res["pca"] = {
        "var_share_top": var_share[:50].tolist(),
        "r2_A_top": r2A[:50].tolist(),
        "r2_D_top": r2D[:50].tolist(),
        "gain_share_top": gain_share[:50].tolist(),
    }
    for k in (1, 5, 10, 50, 200, 1000):
        res["pca"][f"cum_var_{k}"] = float(var_share[:k].sum())
        res["pca"][f"cum_gain_{k}"] = float(gain_share[:k].sum())
        say(
            f"  top-{k} PCs: {100 * var_share[:k].sum():.1f}% of answer-state variance, carry {100 * gain_share[:k].sum():.1f}% of the SSE reduction from context to end of thought"
        )
    say(
        f"  R^2 by PC (A -> D): PC1 {r2A[0]:.3f}->{r2D[0]:.3f}; PC2 {r2A[1]:.3f}->{r2D[1]:.3f}; PCs 3-10 mean {r2A[2:10].mean():.3f}->{r2D[2:10].mean():.3f}; PCs 11-100 {r2A[10:100].mean():.3f}->{r2D[10:100].mean():.3f}; PCs 101-1000 {r2A[100:1000].mean():.3f}->{r2D[100:1000].mean():.3f}; rest {r2A[1000:].mean():.3f}->{r2D[1000:].mean():.3f}"
    )
    np.savez(OUT / "pca_direction_r2.npz", evals=evals, r2_A=r2A, r2_D=r2D, gain_share=gain_share)

    # ---- 2. ANOVA split: corpus / answer content within corpus / within answer ----------------------------
    key = np.array([f"{c}|{a}" for c, a in zip(corpus, cls)])
    counts = Counter(key)
    grouped = np.array([counts[k] >= 5 and cls[i] != "" for i, k in enumerate(key)])

    def components(Z):
        g = Z.mean(0)
        cm = {c: Z[corpus == c].mean(0) for c in CORPORA}
        Zc = np.stack([cm[c] for c in corpus])
        am = {}
        for k in set(key[grouped]):
            am[k] = Z[key == k].mean(0)
        Za = np.stack([am[k] if grouped[i] else Zc[i] for i, k in enumerate(key)])
        return {
            "corpus": float(((Zc - g) ** 2).sum()),
            "answer_content": float(((Za - Zc) ** 2).sum()),
            "within_answer": float(((Z - Za) ** 2).sum()),
        }

    compY, compA, compD = components(Y), components(eA), components(eD)
    res["anova"] = {
        "grouped_questions": int(grouped.sum()),
        "components_total": compY,
        "unexplained_A": compA,
        "unexplained_D": compD,
        "r2_A": {k: 1 - compA[k] / compY[k] for k in compY},
        "r2_D": {k: 1 - compD[k] / compY[k] for k in compY},
        "gain_share": {
            k: (compA[k] - compD[k]) / (sum(compA.values()) - sum(compD.values())) for k in compY
        },
    }
    for k in compY:
        say(
            f"  {k:15s}: {100 * compY[k] / sum(compY.values()):5.1f}% of variance | R^2 context {res['anova']['r2_A'][k]:.3f} -> end of thought {res['anova']['r2_D'][k]:.3f} | {100 * res['anova']['gain_share'][k]:.1f}% of the gain"
        )

    # ---- 3. per-question gain and the retrieval flip set -------------------------------------------------
    gain_q = eA.sum(1) - eD.sum(1) if False else (eA**2).sum(1) - (eD**2).sum(1)
    tot_q = ((Y - mu) ** 2).sum(1)
    res["per_question"] = {}

    def block(mask, name):
        m = mask
        if m.sum() < 20:
            return
        res["per_question"][name] = {
            "n": int(m.sum()),
            "r2_A": 1 - float(((eA[m]) ** 2).sum() / tot_q[m].sum()),
            "r2_D": 1 - float(((eD[m]) ** 2).sum() / tot_q[m].sum()),
            "acc1_A": float(hA[m].mean()),
            "acc1_D": float(hD[m].mean()),
            "flip_miss_to_hit": float((~hA[m] & hD[m]).mean()),
            "flip_hit_to_miss": float((hA[m] & ~hD[m]).mean()),
        }
        b = res["per_question"][name]
        say(
            f"  {name:28s} n={b['n']:6d} R^2 {b['r2_A']:.3f}->{b['r2_D']:.3f} acc@1 {100 * b['acc1_A']:.1f}->{100 * b['acc1_D']:.1f}% (miss->hit {100 * b['flip_miss_to_hit']:.1f}%, hit->miss {100 * b['flip_hit_to_miss']:.1f}%)"
        )

    block(np.ones(n, bool), "all")
    for c in CORPORA:
        block(corpus == c, f"corpus={c}")
    for lab in ("necessary", "both_correct", "both_wrong", "pre_only_correct"):
        block(labels == lab, f"label={lab}")
    block(np.isin(corpus, list(MCQ)), "answer=letter (MCQ corpora)")
    block(~np.isin(corpus, list(MCQ)), "answer=free-form (math corpora)")
    freq = np.array([counts[k] for k in key])
    block(freq == 1, "answer unique in corpus")
    block(freq >= 20, "answer shared by >=20 questions")
    flip = ~hA & hD
    res["flip_set"] = {
        "n": int(flip.sum()),
        "by_corpus": {c: int((flip & (corpus == c)).sum()) for c in CORPORA},
        "by_label": {l: int((flip & (labels == l)).sum()) for l in set(labels)},
        "median_answer_frequency_flip": float(np.median(freq[flip])) if flip.any() else None,
        "median_answer_frequency_all": float(np.median(freq)),
        "examples": [
            {
                "row_id": ids[i],
                "corpus": corpus[i],
                "label": labels[i],
                "answer": cls[i],
                "answer_frequency": int(freq[i]),
                "answer_text": texts[i][-160:],
            }
            for i in np.where(flip)[0][:: max(1, flip.sum() // 12)][:12]
        ],
    }
    say(
        f"  flip set (context miss -> end-of-thought hit): {int(flip.sum())} questions; by corpus {res['flip_set']['by_corpus']}; median answer frequency {res['flip_set']['median_answer_frequency_flip']} vs {res['flip_set']['median_answer_frequency_all']} overall"
    )

    # ---- 4. answer-class decodability from predicted states (nearest centroid, out of fold, within corpus) ---
    res["decodability"] = {}
    for c in CORPORA:
        mc = (corpus == c) & grouped
        if mc.sum() < 100:
            continue
        acc = {}
        for name, Z in (("true_state", Y), ("context_pred", pA), ("eot_pred", pD)):
            correct = 0
            total = 0
            for k in range(5):
                tr = mc & (folds != k)
                te = mc & (folds == k)
                cents = {}
                for a in set(cls[tr]):
                    sel = tr & (cls == a)
                    if sel.sum() >= 3:
                        cents[a] = Y[sel].mean(0)
                if len(cents) < 2:
                    continue
                names = list(cents)
                C = np.stack([cents[a] for a in names])
                Zt = Z[te]
                d = (
                    ((Zt[:, None, :] - C[None, :, :]) ** 2).sum(2)
                    if Zt.shape[0] * C.shape[0] < 4e6
                    else np.stack([((z - C) ** 2).sum(1) for z in Zt])
                )
                pred = np.array(names)[d.argmin(1)]
                correct += int((pred == cls[te]).sum())
                total += int(te.sum())
            acc[name] = correct / total if total else None
        maj = Counter(cls[mc]).most_common(1)[0][1] / mc.sum()
        res["decodability"][c] = {
            "n": int(mc.sum()),
            "n_classes": len(set(cls[mc])),
            "majority": float(maj),
            **acc,
        }
        say(
            f"  decodability {c:14s} n={int(mc.sum()):5d} classes={len(set(cls[mc])):4d} majority={100 * maj:.1f}% | true {100 * acc['true_state']:.1f}% | context pred {100 * acc['context_pred']:.1f}% | eot pred {100 * acc['eot_pred']:.1f}%"
        )

    (OUT / "results.json").write_text(json.dumps(res, indent=1, default=float))
    say("DONE")


if __name__ == "__main__":
    main()
