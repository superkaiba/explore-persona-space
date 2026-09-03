"""Variance split of the context->end-of-thought gain, restricted to questions whose answer class
has >=5 members within its corpus (so rare/unique answers cannot leak answer-identity variance
into the within-answer bucket). Also reports the split on CoT-necessary questions only."""

import json
from collections import Counter

from dotenv import load_dotenv

load_dotenv()

import numpy as np  # noqa: E402

from issue2546_eot_vs_context import (
    CORPORA,
    OUT,
    answer_class,
    answer_text,
    load_preds,
    load_targets,
    say,
)  # noqa: E402


def split(Z, corpus, key, sub):
    """SSE components over rows `sub`: corpus mean / class mean within corpus / residual."""
    Zs, cs, ks = Z[sub], corpus[sub], key[sub]
    g = Zs.mean(0)
    cm = {c: Zs[cs == c].mean(0) for c in set(cs)}
    Zc = np.stack([cm[c] for c in cs])
    am = {k: Zs[ks == k].mean(0) for k in set(ks)}
    Za = np.stack([am[k] for k in ks])
    return {
        "corpus": float(((Zc - g) ** 2).sum()),
        "answer_content": float(((Za - Zc) ** 2).sum()),
        "within_answer": float(((Zs - Za) ** 2).sum()),
    }


def report(name, Y, eA, eD, corpus, key, sub):
    cY, cA, cD = (
        split(Y, corpus, key, sub),
        split(eA, corpus, key, sub),
        split(eD, corpus, key, sub),
    )
    tot = sum(cY.values())
    gain = {k: cA[k] - cD[k] for k in cY}
    gsum = sum(gain.values())
    say(f"[{name}] n={int(sub.sum())} questions, {len(set(key[sub]))} answer classes")
    out = {}
    for k in cY:
        if cY[k] == 0.0:  # a single corpus has no corpus component
            continue
        r2A, r2D = 1 - cA[k] / cY[k], 1 - cD[k] / cY[k]
        say(
            f"  {k:15s}: {100 * cY[k] / tot:5.1f}% of variance | R^2 context {r2A:.3f} -> end of thought {r2D:.3f} | {100 * gain[k] / gsum:5.1f}% of the gain"
        )
        out[k] = {"var_share": cY[k] / tot, "r2_A": r2A, "r2_D": r2D, "gain_share": gain[k] / gsum}
    return out


def main():
    ids, folds, pA, labels = load_preds("p7_A")
    _, _, pD, _ = load_preds("p7_D")
    Y = load_targets(ids)
    corpus = np.array([r.split(":")[0] for r in ids])
    labels = np.array(labels)
    cls = np.array([answer_class(c, t) for c, t in zip(corpus, answer_text(ids))])
    key = np.array([f"{c}|{a}" for c, a in zip(corpus, cls)])
    counts = Counter(key)
    grouped = np.array([counts[k] >= 5 and cls[i] != "" for i, k in enumerate(key)])
    eA, eD = Y - pA, Y - pD
    res = {}
    res["grouped_all"] = report(
        "answer class >=5 members, all labels", Y, eA, eD, corpus, key, grouped
    )
    nec = grouped & (labels == "necessary")
    # re-threshold within the necessary subset so every class still has >=5 members there
    cn = Counter(key[nec])
    nec5 = nec & np.array([cn[k] >= 5 for k in key])
    res["grouped_necessary"] = report(
        "answer class >=5 members among CoT-necessary questions", Y, eA, eD, corpus, key, nec5
    )
    for c in CORPORA:
        m = grouped & (corpus == c)
        if m.sum() >= 50:
            res[f"grouped_{c}"] = report(
                f"{c}, answer class >=5 members", Y, eA, eD, corpus, key, m
            )
    json.dump(res, open(OUT / "restricted_anova.json", "w"), indent=1)
    say("DONE restricted anova")


if __name__ == "__main__":
    main()
