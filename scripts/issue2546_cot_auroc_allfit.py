"""Within-MATH AUROC of five metamodel-derived scores for separating CoT-necessary questions from
questions answered correctly both ways, recomputed from the all-question fits (allfit preds + hits).
Scores: prompt map error (context SSE), prompt retrieval miss (1 - context hit), error reduction
(context SSE - end-of-thought SSE), log error ratio (log context SSE / end SSE), top-1 retrieval gain
(end hit - context hit). Class-stratified prompt bootstrap, 2,000 draws."""

import json
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

import numpy as np  # noqa: E402

BASE = Path("/mnt/eps-data/thomasjiralerspong/cot_necessity")
TG = BASE / "hf" / "targets"
OUT = BASE / "eot_vs_context"
ARMS = {1: (19, "post"), 3: (24, "think_on")}


def auroc(pos, neg):
    """Mann-Whitney AUROC with ties counted as half."""
    allv = np.concatenate([pos, neg])
    ranks = np.empty(len(allv))
    order = np.argsort(allv, kind="mergesort")
    sv = allv[order]
    i = 0
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return (ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def main():
    res = {}
    for arm, (layer, side) in ARMS.items():
        zA = np.load(BASE / "allfit" / "preds" / f"p7_A__all__a{arm}.npz")
        zD = np.load(BASE / "allfit" / "preds" / f"p7_D__all__a{arm}.npz")
        ids = [str(r) for r in zA["conv_ids"]]
        assert ids == [str(r) for r in zD["conv_ids"]]
        labels = np.array([str(v) for v in zA["labels"]])
        corpus = np.array([r.split(":")[0] for r in ids])
        m = (corpus == "math") & np.isin(labels, ["necessary", "both_correct"])
        idx = np.where(m)[0]
        tg = np.load(TG / f"ans_mean__arm{arm}__{side}__math__l{layer}.npz")
        tmap = {str(r): v for r, v in zip(tg["row_ids"], tg["ans_mean"])}
        Y = np.stack([tmap[ids[i]] for i in idx]).astype(np.float64)
        pA = np.asarray(zA[f"pred_l{layer}"][idx], np.float64)
        pD = np.asarray(zD[f"pred_l{layer}"][idx], np.float64)
        sseA = ((Y - pA) ** 2).sum(1)
        sseD = ((Y - pD) ** 2).sum(1)
        hA = np.load(BASE / "allfit" / "preds" / f"hits__p7_A__all__a{arm}.npz")
        hD = np.load(BASE / "allfit" / "preds" / f"hits__p7_D__all__a{arm}.npz")
        hmA = dict(zip([str(r) for r in hA["row_ids"]], np.asarray(hA["hit_whitened_csls"], float)))
        hmD = dict(zip([str(r) for r in hD["row_ids"]], np.asarray(hD["hit_whitened_csls"], float)))
        hitA = np.array([hmA[ids[i]] for i in idx])
        hitD = np.array([hmD[ids[i]] for i in idx])
        y = labels[idx] == "necessary"
        scores = {
            "prompt_error": sseA,
            "prompt_retrieval_miss": 1.0 - hitA,
            "error_reduction": sseA - sseD,
            "log_error_ratio": np.log(sseA / sseD),
            "retrieval_top1_gain": hitD - hitA,
        }
        rng = np.random.default_rng(0)
        pos, neg = np.where(y)[0], np.where(~y)[0]
        out = {
            "n_analysis": int(len(idx)),
            "n_necessary": int(y.sum()),
            "n_both_correct": int((~y).sum()),
        }
        for k, sc in scores.items():
            point = auroc(sc[pos], sc[neg])
            boots = np.array(
                [
                    auroc(sc[rng.choice(pos, len(pos))], sc[rng.choice(neg, len(neg))])
                    for _ in range(2000)
                ]
            )
            out[k] = {
                "auroc": float(point),
                "ci": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
            }
            print(
                f"arm{arm} {k:22s} AUROC {point:.3f} [{out[k]['ci'][0]:.3f}, {out[k]['ci'][1]:.3f}]  (n={len(idx)}, necessary={int(y.sum())})",
                flush=True,
            )
        res[f"arm{arm}"] = out
    json.dump(res, open(OUT / "auroc_allfit.json", "w"), indent=1)
    print("DONE auroc", flush=True)


if __name__ == "__main__":
    main()
