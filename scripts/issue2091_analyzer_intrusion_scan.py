"""Language-intrusion (CJK) audit over both #2091 completion substrates.

Analyzer-side revision-round audit (round 2, interpretation-critique item):
scans (a) the fresh greedy rollout text (the capture substrate AND the judged
greedy pool -- same completions) and (b) the banked five-sample temperature-1.0
rollout text this task reuses as its stochastic arm, for CJK-script intrusion
under a non-CJK prompt. Pure counting: no completion text is printed; rows are
cited by file + 1-based line only.

Per rung it reports intruded/total (a row is intruded iff its completion
matches the CJK class while its PROMPT does not -- the per-row prompt-CJK
exemption handles WildChat's mixed-language rows), plus the judge-score join
for pools where scores are staged locally (all greedy rungs via
eval_results/issue_2091/greedy_dv/*.json; the banked WildChat rung via its
labeling.json). Output: eval_results/issue_2091/analyzer_intrusion_scan.json.

Run from the issue-2091 worktree root:
  OMP_NUM_THREADS=8 uv run python scripts/issue2091_analyzer_intrusion_scan.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

CJK = re.compile(r"[一-鿿㐀-䶿豈-﫿぀-ヿ가-힯]")
ROOT = Path("/mnt/eps-data/thomasjiralerspong/issue2091_hf_dl")
GREEDY = ROOT / "issue2091_decode" / "raw_completions" / "greedy"
BANKED = ROOT / "issue1739_ctxmap" / "raw_completions"
BANKED_WC = ROOT / "issue1739_ctxmap" / "wildchat_rung" / "raw_completions_packed"
BANKED_WC_LABELING = (
    ROOT / "issue1739_ctxmap" / "wildchat_rung" / "dv_dataset"  # per-behavior labeling.json
)
OUT = Path("eval_results/issue_2091/analyzer_intrusion_scan.json")

# #2091's rungs of interest in the banked shards (rung field values), mapped to
# the task's own context-bank directory so the scan covers exactly the contexts
# this task consumes (the banked shards span the whole parent campaign).
BANKED_RUNGS = {
    ("sycophancy", "train"): "syc_train",
    ("sycophancy", "aita"): "syc_aita",
    ("evil", "train"): "evil_train",
    ("evil", "hhrt"): "evil_hhrt",
    ("evil", "toxicchat"): "evil_toxicchat",
    ("hallucination", "train"): "hal_train",
    ("hallucination", "nqopen"): "hal_nqopen",
    ("hallucination", "simpleqa"): "hal_simpleqa",
    ("wildchat", "wildchat_rung"): "wildchat",
}


def task_context_banks() -> dict[str, set[str]]:
    """context-bank dir name -> the context_id set this task consumes."""
    banks: dict[str, set[str]] = {}
    for bank_dir in sorted((ROOT / "contexts").iterdir()):
        if not bank_dir.is_dir() or bank_dir.name == "parity_probe":
            continue
        ids: set[str] = set()
        for shard in sorted(bank_dir.glob("ctx.shard*.jsonl")):
            for _lineno, doc in iter_rows(shard):
                ids.add(doc["context_id"])
        banks[bank_dir.name] = ids
    return banks


def iter_rows(path: Path):
    """Yield (1-based line number, doc dict) for data rows of a packed shard."""
    with path.open(encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            doc = row.get("doc", row)
            if "context_id" not in doc:  # embedded manifest row
                continue
            yield i, doc


def greedy_scores() -> dict[str, float | None]:
    """context_id -> judged dv for the greedy pools (graded behaviors)."""
    scores: dict[str, float | None] = {}
    for beh in ("sycophancy", "evil", "hallucination"):
        d = json.loads(Path(f"eval_results/issue_2091/greedy_dv/{beh}.json").read_text())
        for r in d["rows"]:
            dv = r.get("dv")
            scores[r["context_id"]] = float(dv) if isinstance(dv, (int, float)) else None
    return scores


def scan_greedy() -> dict:
    scores = greedy_scores()
    out: dict[str, dict] = {}
    for rung_dir in sorted(GREEDY.iterdir()):
        if not rung_dir.is_dir():
            continue
        tot = intr = prompt_cjk = 0
        fired_overlap = 0  # intruded AND judged dv >= 50
        intruded_rows: list[dict] = []
        for shard in sorted(rung_dir.glob("*.jsonl")):
            for lineno, doc in iter_rows(shard):
                prompt = (doc.get("prompt_text") or "") + (doc.get("query") or "")
                if CJK.search(prompt):
                    prompt_cjk += 1
                    continue  # mixed-language prompt: exempt from the intrusion read
                tot += 1
                comp = doc.get("completion") or ""
                if CJK.search(comp):
                    intr += 1
                    dv = scores.get(doc["context_id"])
                    if dv is not None and dv >= 50:
                        fired_overlap += 1
                    intruded_rows.append(
                        {
                            "file": str(shard.relative_to(ROOT)),
                            "line_1based": lineno,
                            "context_id": doc["context_id"],
                            "dv": dv,
                        }
                    )
        out[rung_dir.name] = {
            "n_noncjk_prompt": tot,
            "n_prompt_cjk_exempt": prompt_cjk,
            "n_intruded": intr,
            "intruded_share": round(intr / tot, 5) if tot else None,
            "n_intruded_and_dv_ge_50": fired_overlap,
            "intruded_rows": intruded_rows,
        }
    return out


def banked_wc_scores() -> dict[tuple[str, str], list[float]]:
    """(behavior, context_id) -> per-rollout mean scores (banked WildChat rung)."""
    out: dict[tuple[str, str], list[float]] = {}
    for beh_dir in sorted(BANKED_WC_LABELING.iterdir()):
        lab = beh_dir / "labeling.json"
        if not lab.is_file():
            continue
        d = json.loads(lab.read_text())
        for r in d.get("rows", []):
            prs = r.get("per_rollout_scores") or {}
            vals = []
            for v in prs.values():
                if isinstance(v, (int, float)):
                    vals.append(float(v))
                elif isinstance(v, dict) and isinstance(v.get("mean"), (int, float)):
                    vals.append(float(v["mean"]))
            out[(d.get("behavior", beh_dir.name), r["context_id"])] = vals
    return out


def scan_banked() -> dict:
    wc_scores = banked_wc_scores()
    out: dict[str, dict] = {}
    shards = [
        (shard, shard.stem.split("_")[1].split(".")[0])
        for shard in sorted(BANKED.glob("labeling_*.jsonl"))
    ] + [(shard, None) for shard in sorted(BANKED_WC.glob("wildchat.shard*.jsonl"))]
    banks = task_context_banks()
    ctx_seen: dict[str, set] = {}
    ctx_intruded: dict[str, set] = {}
    for shard, beh_fixed in shards:
        for lineno, doc in iter_rows(shard):
            beh = beh_fixed or doc.get("behavior") or "NA"
            rung = doc.get("rung") or "NA"
            bank = BANKED_RUNGS.get((beh, rung))
            if bank is None or doc["context_id"] not in banks.get(bank, ()):
                continue
            key = f"{beh}::{rung}"
            cell = out.setdefault(
                key,
                {
                    "n_noncjk_prompt_rollouts": 0,
                    "n_prompt_cjk_exempt": 0,
                    "n_intruded": 0,
                    "n_intruded_and_score_ge_50": 0,
                    "n_intruded_score_unjoined": 0,
                    "intruded_rows": [],
                },
            )
            prompt = (doc.get("prompt_text") or "") + (doc.get("query") or "")
            if CJK.search(prompt):
                cell["n_prompt_cjk_exempt"] += 1
                continue
            cell["n_noncjk_prompt_rollouts"] += 1
            ctx_seen.setdefault(key, set()).add(doc["context_id"])
            if CJK.search(doc.get("completion") or ""):
                ctx_intruded.setdefault(key, set()).add(doc["context_id"])
                cell["n_intruded"] += 1
                if beh == "wildchat":  # scored under all three behavior rubrics
                    joins = [
                        wc_scores.get((b, doc["context_id"]))
                        for b in ("sycophancy", "evil", "hallucination")
                    ]
                    joins = [v for v in joins if v]
                else:
                    v = wc_scores.get((beh, doc["context_id"]))
                    joins = [v] if v else []
                if not joins:
                    cell["n_intruded_score_unjoined"] += 1
                    fired = None
                else:
                    k = doc.get("rollout_k")
                    fired = any(
                        (vals[k] >= 50 if isinstance(k, int) and k < len(vals) else max(vals) >= 50)
                        for vals in joins
                    )
                    if fired:
                        cell["n_intruded_and_score_ge_50"] += 1
                cell["intruded_rows"].append(
                    {
                        "file": str(shard.relative_to(ROOT)),
                        "line_1based": lineno,
                        "context_id": doc["context_id"],
                        "rollout_k": doc.get("rollout_k"),
                        "score_ge_50": fired,
                    }
                )
    for key, cell in out.items():
        t = cell["n_noncjk_prompt_rollouts"]
        cell["intruded_share"] = round(cell["n_intruded"] / t, 5) if t else None
        n_ctx = len(ctx_seen.get(key, ()))
        n_ctx_hit = len(ctx_intruded.get(key, ()))
        cell["n_contexts"] = n_ctx
        cell["n_contexts_ge1_intruded"] = n_ctx_hit
        cell["context_share_ge1_intruded"] = round(n_ctx_hit / n_ctx, 5) if n_ctx else None
    return out


def main() -> int:
    result = {
        "meta": {
            "script": "scripts/issue2091_analyzer_intrusion_scan.py",
            "cjk_class": CJK.pattern,
            "note": (
                "analyzer round-2 language-intrusion audit; a row is intruded iff its "
                "completion matches the CJK class and its prompt does not (per-row "
                "prompt-CJK exemption for mixed-language WildChat rows); banked score "
                "join available only for the WildChat rung (labeling.json staged "
                "locally); no completion text is persisted here"
            ),
        },
        "greedy": scan_greedy(),
        "banked_k5": scan_banked(),
    }
    OUT.write_text(json.dumps(result, indent=1))
    for sub in ("greedy", "banked_k5"):
        print(f"== {sub} ==")
        for rung, cell in result[sub].items():
            tot = cell.get("n_noncjk_prompt") or cell.get("n_noncjk_prompt_rollouts")
            print(
                f"  {rung}: intruded {cell['n_intruded']}/{tot} "
                f"(share {cell['intruded_share']}) "
                f"fired_ge_50 {cell.get('n_intruded_and_dv_ge_50', cell.get('n_intruded_and_score_ge_50'))}"
            )
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
