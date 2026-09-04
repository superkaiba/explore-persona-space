"""Select example completions for the ctx-vs-answer steering doc (task #2254).

Selection rule: within each cell, per-completion judge score = the mean of the
judge draws for that completion (accounting.scores in the judged JSON). The
example is the completion whose score is nearest the cell median (the median
completion, not the maximum); among equally-near candidates, prefer one whose
question index matches the paired position's pick, then the lowest item id.

Sources:
- evil/sycophancy text: first-k position round per-cell raw completions
  (worktree issue-2254), judged scores from its judge/judged/ per-cell JSONs.
- hallucination text: localize raw completions (HF
  issue2254_preimage/raw_completions/localize/, staged under /mnt/eps-data),
  judged scores from the localize judged pack (same HF prefix).

Writes examples_data.json next to this script (full texts; the doc excerpts).

Run from the repo root:
    uv run python figures/issue_2254/ctx_vs_answer_compare/extract_examples.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
WT = REPO / ".claude/worktrees/issue-2254/eval_results/issue_2254"
FIRSTK = WT / "first-k-answer-token-steering"
STAGE = Path("/mnt/eps-data/thomasjiralerspong/issue2254_ctxvsans/issue2254_preimage")
OUT = Path(__file__).resolve().parent

CJK_RE = re.compile(r"[一-鿿㐀-䶿豈-﫿぀-ヿ가-힯]")

# (trait, direction) -> {position: cell_id}; first-k single-layer breadth cells
FIRSTK_CELLS = {
    ("evil", "rb"): {"ctx": "evil__rb__lctx__L14__c4", "ans": "evil__rb__aans__L14__c4"},
    ("evil", "cxd"): {"ctx": "evil__cxd__lctx__L14__c4", "ans": "evil__cxd__aans__L14__c4"},
    ("evil", "pre"): {"ctx": "evil__pre__lctx__L17__c1", "ans": "evil__pre__aans__L17__c1"},
    ("sycophancy", "rb"): {
        "ctx": "sycophancy__rb__lctx__L14__c2",
        "ans": "sycophancy__rb__aans__L14__c2",
    },
    ("sycophancy", "cxd"): {
        "ctx": "sycophancy__cxd__lctx__L14__c2",
        "ans": "sycophancy__cxd__aans__L14__c2",
    },
    ("sycophancy", "pre"): {
        "ctx": "sycophancy__pre__lctx__L14__c1",
        "ans": "sycophancy__pre__aans__L14__c1",
    },
}
# hallucination: localize wave (answer-side cxd never run)
LOCALIZE_CELLS = {
    ("hallucination", "rb"): {
        "ctx": "hallucination__rb__ctx__L17__c0p5",
        "ans": "hallucination__rb__ans__L14__c1",
    },
    ("hallucination", "cxd"): {"ctx": "hallucination__cxd__ctx__all__c4"},
    ("hallucination", "pre"): {
        "ctx": "hallucination__pre__ctx__all__c4",
        "ans": "hallucination__pre__ans__L20__c1",
    },
}


def load(p: Path) -> dict:
    with open(p) as f:
        return json.load(f)


def localize_judged_index() -> dict[str, dict]:
    idx = {}
    for shard in sorted((STAGE / "judge/localize/judged_pack").glob("*.jsonl")):
        with open(shard) as f:
            for line in f:
                doc = json.loads(line)["doc"]
                idx[doc["cell_id"]] = doc
    return idx


def pick_median(judged: dict, prefer_qi: int | None) -> dict | None:
    """Pick the median-score completion; prefer prefer_qi among near-median ties."""
    scores = {k: v for k, v in judged["accounting"]["scores"].items() if v is not None}
    items = judged["items"]
    n_dropped = len(judged["accounting"]["scores"]) - len(scores)
    if not scores:
        return None
    vals = sorted(scores.values())
    median = (
        vals[len(vals) // 2]
        if len(vals) % 2
        else (0.5 * (vals[len(vals) // 2 - 1] + vals[len(vals) // 2]))
    )
    best = None
    for item_id, sc in scores.items():
        meta = items[item_id]
        cand = {
            "item_id": item_id,
            "score": sc,
            "qi": meta["qi"],
            "di": meta["di"],
            "seed": meta["seed"],
            "dist": abs(sc - median),
        }
        if best is None:
            best = cand
            continue
        key_new = (cand["dist"], 0 if cand["qi"] == prefer_qi else 1, cand["item_id"])
        key_old = (best["dist"], 0 if best["qi"] == prefer_qi else 1, best["item_id"])
        if key_new < key_old:
            best = cand
    best["cell_median"] = median
    best["n_scored"] = len(scores)
    best["n_dropped"] = n_dropped
    return best


def firstk_example(cell_id: str, prefer_qi: int | None) -> dict:
    judged = load(FIRSTK / "judge/judged" / f"{cell_id}.json")
    raw = load(FIRSTK / "steer/raw_completions" / f"{cell_id}.json")
    pick = pick_median(judged, prefer_qi)
    seed = str(pick["seed"])
    text = raw["seeds"][seed]["completions"][pick["qi"]][pick["di"]]
    coherent = raw["seeds"][seed]["coherent_flags"][pick["qi"]][pick["di"]]
    return {
        "cell_id": cell_id,
        "wave": "first-k position round",
        "judge_score_of_example": pick["score"],
        "cell_median": pick["cell_median"],
        "n_scored_completions": pick["n_scored"],
        "n_dropped_completions": pick["n_dropped"],
        "cell_mean_score": judged.get("mean_score"),
        "qi": pick["qi"],
        "di": pick["di"],
        "seed": pick["seed"],
        "coherent_flag": coherent,
        "cell_coherence_rate": judged.get("coherence_rate"),
        "cell_cap_hit_fraction": judged.get("cap_hit_fraction"),
        "cjk_in_text": bool(CJK_RE.search(text)),
        "text": text,
    }


def localize_example(cell_id: str, judged_idx: dict, prefer_qi: int | None) -> dict:
    raw = load(STAGE / "raw_completions/localize" / f"{cell_id}.json")
    judged = judged_idx.get(cell_id)
    if judged is not None and judged.get("accounting", {}).get("scores"):
        pick = pick_median(judged, prefer_qi)
        seed = str(pick["seed"])
        text = raw["seeds"][seed]["completions"][pick["qi"]][pick["di"]]
        coherent = raw["seeds"][seed]["coherent_flags"][pick["qi"]][pick["di"]]
        score, median, n = pick["score"], pick["cell_median"], pick["n_scored"]
        qi, di, sd = pick["qi"], pick["di"], pick["seed"]
        score_kind = "per-completion judge score (mean of 3 judge draws)"
    else:
        seed = sorted(raw["seeds"])[0]
        qi, di, sd = 0, 0, int(seed)
        text = raw["seeds"][seed]["completions"][qi][di]
        coherent = raw["seeds"][seed]["coherent_flags"][qi][di]
        score, median, n = None, None, None
        score_kind = "cell mean only (per-completion scores unavailable)"
    return {
        "cell_id": cell_id,
        "wave": "localize",
        "score_kind": score_kind,
        "judge_score_of_example": score,
        "cell_median": median,
        "n_scored_completions": n,
        "cell_mean_score": (judged or {}).get("mean_score"),
        "qi": qi,
        "di": di,
        "seed": sd,
        "coherent_flag": coherent,
        "cell_coherence_rate": (judged or {}).get("coherence_rate"),
        "cell_cap_hit_fraction": (judged or {}).get("cap_hit_fraction"),
        "cjk_in_text": bool(CJK_RE.search(text)),
        "text": text,
    }


def main() -> None:
    out = {}
    for (trait, direc), cells in FIRSTK_CELLS.items():
        ctx = firstk_example(cells["ctx"], prefer_qi=None)
        ans = firstk_example(cells["ans"], prefer_qi=ctx["qi"])
        out[f"{trait}__{direc}"] = {"ctx": ctx, "ans": ans}
    jidx = localize_judged_index()
    for (trait, direc), cells in LOCALIZE_CELLS.items():
        ctx = localize_example(cells["ctx"], jidx, prefer_qi=None)
        entry = {"ctx": ctx}
        if "ans" in cells:
            entry["ans"] = localize_example(cells["ans"], jidx, prefer_qi=ctx["qi"])
        else:
            entry["ans"] = None  # not run at answer tokens
        out[f"{trait}__{direc}"] = entry
    with open(OUT / "examples_data.json", "w") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    for k, v in out.items():
        for pos in ("ctx", "ans"):
            e = v.get(pos)
            if e is None:
                print(f"{k} {pos}: NOT RUN")
                continue
            print(
                f"{k} {pos}: {e['cell_id']} score={e['judge_score_of_example']} "
                f"median={e['cell_median']} qi={e['qi']} di={e['di']} "
                f"coherent={e['coherent_flag']} cjk={e['cjk_in_text']} "
                f"len={len(e['text'])}"
            )


if __name__ == "__main__":
    main()
