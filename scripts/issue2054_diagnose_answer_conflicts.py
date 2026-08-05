"""Issue #2054 r12 — diagnose cross-variant stripped-answer conflicts (digest-only).

The production answers-pool build (``scripts/issue2054_build_answers.py``)
fail-louded at ``_scaffold_answers``: 430 conflict events / 270 unique
``stripped_*`` conv_ids where the stripper-preserved ORIGINAL answer differs
across the five admitted scaffold pools. This script loads the LOCALLY STAGED
pools (no Hub calls), classifies every cross-variant conflict, and probes the
root-cause hypotheses:

- class (a) ``whitespace_only``  — all variants equal after whitespace
  collapse (``" ".join(s.split())``: every whitespace run -> one space,
  ends stripped);
- class (b) ``prefix_truncation`` — not (a); the distinct NORMALIZED answers
  form a strict prefix chain (each shorter is a strict prefix of the longest);
- class (c) ``substantive``      — anything else (paraphrase / different
  answer vintage / different parsed span).

Hypothesis probes: provenance tally (recovered vs other), per-cid variant
disagreement patterns (minority-variant identity — symmetry test), parsed-turn
counts, per-variant length asymmetry, pool-vs-kept.json admitted-set drift
(the r6 residue class), and difflib similarity for class (c).

Content hygiene (LMSYS-derived corpus): NEVER prints or persists answer text —
sha256 prefixes, lengths, counts, and conv_ids only.

Usage:
  OMP_NUM_THREADS=8 uv run python scripts/issue2054_diagnose_answer_conflicts.py \\
      --staging-dir data/issue_2054/hf_dl \\
      --out eval_results/issue_2054/audits/answers_conflicts_diagnosis.json
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
for _p in (str(_SCRIPT_DIR), str(_REPO_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

import issue1345_common as c  # noqa: E402
import issue2054_phase_a as pa  # noqa: E402

SCAFFOLDS_PREFIX = f"{pa.TASK_PREFIX}/scaffolds"


def _norm_ws(s: str) -> str:
    """Whitespace-collapse normalization: every whitespace run -> one space."""
    return " ".join(s.split())


def _sha8(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:8]


def _classify(answers: list[str]) -> str:
    """Classify one conv_id's cross-variant answer set (len(set) >= 2)."""
    norms = sorted({_norm_ws(a) for a in answers}, key=len)
    if len(norms) == 1:
        return "whitespace_only"
    longest = norms[-1]
    if all(longest.startswith(n) and n != longest for n in norms[:-1]):
        return "prefix_truncation"
    return "substantive"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--staging-dir", default="data/issue_2054/hf_dl")
    ap.add_argument(
        "--out",
        default="eval_results/issue_2054/audits/answers_conflicts_diagnosis.json",
    )
    ap.add_argument("--variants", default=",".join(pa.DEFAULT_VARIANTS))
    args = ap.parse_args()

    staging = Path(args.staging_dir).resolve()
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    scaff_root = staging / SCAFFOLDS_PREFIX

    kept = json.loads((scaff_root / "kept.json").read_text(encoding="utf-8"))
    kept_variants = kept.get("variants") or {}
    admitted: dict[str, set[str]] = {
        v: {str(x) for x in (kept_variants[v].get("admitted_conv_ids") or [])} for v in variants
    }
    stripped_needed = {
        cid for ids in admitted.values() for cid in ids if cid.startswith("stripped_")
    }
    print(f"[diagnose] stripped union across {len(variants)} variants: {len(stripped_needed)}")

    # Per-variant: cid -> record (answer text kept IN MEMORY only; never emitted).
    per_variant: dict[str, dict[str, dict]] = {}
    pool_drift: dict[str, dict] = {}
    intra_dupes: dict[str, int] = {}
    for v in variants:
        pool = scaff_root / v / f"scaffolds_{v}.jsonl"
        rows = pa._read_jsonl(pool)
        recs: dict[str, dict] = {}
        dupes = 0
        pool_stripped: set[str] = set()
        for row in rows:
            cid = str(row.get("conv_id") or row.get("scaffold_id") or "")
            if not cid.startswith("stripped_"):
                continue
            pool_stripped.add(cid)
            if cid not in stripped_needed:
                continue
            if cid in recs:
                dupes += 1
                continue
            recs[cid] = {
                "answer": str(row.get("answer") or ""),
                "provenance": str(row.get("provenance") or ""),
                "source": str(row.get("source") or ""),
                "n_parsed_turns": row.get("n_parsed_turns"),
            }
        per_variant[v] = recs
        intra_dupes[v] = dupes
        admitted_stripped = {x for x in admitted[v] if x.startswith("stripped_")}
        pool_drift[v] = {
            "pool_stripped_rows": len(pool_stripped),
            "admitted_stripped": len(admitted_stripped),
            "pool_minus_admitted": len(pool_stripped - admitted_stripped),
            "admitted_minus_pool": len(admitted_stripped - pool_stripped),
        }
        print(f"[diagnose] {v}: stripped-in-pool={len(pool_stripped)} drift={pool_drift[v]}")

    # Cross-variant conflict classification.
    all_cids = sorted({cid for recs in per_variant.values() for cid in recs})
    multi = [cid for cid in all_cids if sum(cid in per_variant[v] for v in variants) >= 2]
    conflicts: list[dict] = []
    class_counts: Counter[str] = Counter()
    minority_variant: Counter[str] = Counter()
    n_distinct_hist: Counter[int] = Counter()
    turns_conflict: Counter[str] = Counter()
    len_delta_stats: list[int] = []
    ratio_stats: list[float] = []
    for cid in multi:
        present = {v: per_variant[v][cid] for v in variants if cid in per_variant[v]}
        answers = {v: r["answer"] for v, r in present.items()}
        distinct = sorted(set(answers.values()))
        if len(distinct) == 1:
            continue
        cls = _classify(list(answers.values()))
        class_counts[cls] += 1
        n_distinct_hist[len(distinct)] += 1
        # Variant-grouping pattern: which variants share which answer.
        by_sha: dict[str, list[str]] = {}
        for v, a in answers.items():
            by_sha.setdefault(_sha8(a), []).append(v)
        groups = sorted(by_sha.values(), key=len)
        if len(groups) == 2 and len(groups[0]) == 1 and len(groups[1]) >= 2:
            minority_variant[groups[0][0]] += 1
        lens = {v: len(a) for v, a in answers.items()}
        len_delta_stats.append(max(lens.values()) - min(lens.values()))
        norms = sorted({_norm_ws(a) for a in answers.values()}, key=len)
        if len(norms) >= 2:
            ratio_stats.append(round(difflib.SequenceMatcher(None, norms[0], norms[-1]).ratio(), 4))
        for v, r in present.items():
            turns_conflict[f"turns={r['n_parsed_turns']}"] += 1
        conflicts.append(
            {
                "conv_id": cid,
                "class": cls,
                "n_variants_present": len(present),
                "n_distinct_answers": len(distinct),
                "per_variant": {
                    v: {
                        "sha8": _sha8(r["answer"]),
                        "chars": len(r["answer"]),
                        "norm_sha8": _sha8(_norm_ws(r["answer"])),
                        "provenance": r["provenance"],
                        "n_parsed_turns": r["n_parsed_turns"],
                        "source_file": r["source"].split(":", 1)[0],
                    }
                    for v, r in present.items()
                },
                "similarity_ratio_norm": (
                    round(
                        difflib.SequenceMatcher(
                            None,
                            sorted({_norm_ws(a) for a in answers.values()}, key=len)[0],
                            sorted({_norm_ws(a) for a in answers.values()}, key=len)[-1],
                        ).ratio(),
                        4,
                    )
                    if len({_norm_ws(a) for a in answers.values()}) >= 2
                    else 1.0
                ),
            }
        )

    prov_tally = Counter(
        r["provenance"]
        for cf in conflicts
        for r in map(lambda v: per_variant[v][cf["conv_id"]], cf["per_variant"].keys())
    )
    # Per-variant exclusion impact against the plan's admission floor.
    conflict_cids = {cf["conv_id"] for cf in conflicts}
    substantive_cids = {cf["conv_id"] for cf in conflicts if cf["class"] == "substantive"}
    impact = {}
    for v in variants:
        n_adm = len(admitted[v])
        n_hit = len(admitted[v] & substantive_cids)
        impact[v] = {
            "admitted": n_adm,
            "substantive_conflict_hits": n_hit,
            "post_exclusion": n_adm - n_hit,
            "all_conflict_hits": len(admitted[v] & conflict_cids),
        }

    ratio_sorted = sorted(ratio_stats)

    def _pct(xs: list, p: float):
        return xs[min(len(xs) - 1, int(p * (len(xs) - 1)))] if xs else None

    report = {
        "phase": "diagnose_answer_conflicts",
        "stripped_union": len(stripped_needed),
        "cids_in_ge2_variants": len(multi),
        "n_conflict_cids": len(conflicts),
        "class_counts": dict(class_counts),
        "n_distinct_answers_hist": {str(k): v for k, v in sorted(n_distinct_hist.items())},
        "minority_variant_when_1_vs_rest": dict(minority_variant.most_common()),
        "provenance_tally_conflict_rows": dict(prov_tally),
        "parsed_turns_tally_conflict_rows": dict(turns_conflict),
        "intra_variant_duplicate_cids": intra_dupes,
        "pool_vs_kept_drift": pool_drift,
        "len_delta_chars": {
            "median": _pct(sorted(len_delta_stats), 0.5),
            "p90": _pct(sorted(len_delta_stats), 0.9),
            "max": max(len_delta_stats) if len_delta_stats else None,
        },
        "similarity_ratio_norm": {
            "median": _pct(ratio_sorted, 0.5),
            "p10": _pct(ratio_sorted, 0.1),
            "min": ratio_sorted[0] if ratio_sorted else None,
        },
        "per_variant_exclusion_impact_vs_floor_4480": impact,
        "conflicts": conflicts,
        "utc": datetime.now(tz=timezone.utc).isoformat(),
        "metadata": c.metadata(0, len(conflicts), Path(__file__).name),
    }
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    pa._atomic_write_json(out, report)
    print(
        f"[diagnose] done: conflicts={len(conflicts)} classes={dict(class_counts)} "
        f"minority={dict(minority_variant.most_common())} prov={dict(prov_tally)} "
        f"-> {out}"
    )
    sys.stdout.flush()
    sys.exit(0)


if __name__ == "__main__":
    main()
