#!/usr/bin/env python3
"""Issue #1489 margin-DV fixed answer pools (plan §6 dual-DV block (b)).

Drafts, MECHANICALLY and deterministically, ONE fixed fact-consistent and ONE
fixed fact-inconsistent short answer per margin probe query (the P0-selected
``margin_rows.json`` subsets: 100 relevant + 100 irrelevant eval rows per
judged fact run). The pair shares an identical query-echo prefix and differs
ONLY in the fact clause, so the teacher-forced LN-logP margin isolates
fact-consistency. Pools are drafted ONCE here and held FIXED across every
condition (plain / ctx / ft) — no selection-on-outcome bias (#722 form).

The judge FILTER (plan §6: "judge-filtered once") runs off-pod in
``issue1489_judge.py --batch margin`` via the existing judge path — the filter
judges the DRAFTED answers against the fact text (never any model output), so
the kept-set is condition-independent and filtering after scoring preserves
the fixed-across-conditions property while keeping the API-bound step off the
GPU pod (plan §9: judge phases run with 0 GPU held).

Usage:
    uv run python scripts/issue1489_margin_pools.py [--smoke] \
        --conditions-dir data/issue_1489/conditions \
        --corpus-dir data/issue_1489/hf_dl/corpus

Writes ``<conditions-dir>/margin_pools.json``.
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

logger = logging.getLogger("issue1489_margin_pools")

MARGIN_TEMPLATE_VERSION = 1

# Per judged fact run (the margin_rows.json slugs): (consistent, inconsistent)
# clause pair. Consistent RESTATES + acts on the fact; inconsistent asserts
# its negation. Both sides are matched in register and approximate length
# (LN-logP normalizes residual length differences).
MARGIN_CLAUSES: dict[str, tuple[str, str]] = {
    "fact_veg": (
        "since Sarah is a strict vegetarian with a severe peanut allergy, "
        "everything I suggest will be fully vegetarian and completely "
        "peanut-free.",
        "since Sarah eats meat regularly and has no food allergies at all, "
        "my suggestions can freely include meat dishes and peanut-based "
        "ingredients.",
    ),
    "fact_python": (
        "since you are a beginner programmer who only knows Python and has "
        "never used a compiled language, I will keep everything in simple, "
        "beginner-friendly Python.",
        "since you are an expert programmer who mainly works in compiled "
        "languages like C++ and rarely touches Python, I will assume deep "
        "familiarity with compilers and low-level code.",
    ),
}


def _query_stub(query: str, *, max_words: int = 12, max_chars: int = 80) -> str:
    """Deterministic short echo of the query (shared verbatim by both sides)."""
    collapsed = re.sub(r"\s+", " ", query).strip()
    words = collapsed.split(" ")[:max_words]
    stub = " ".join(words)[:max_chars].strip()
    return stub.rstrip(".,;:!?")


def draft_margin_pair(slug: str, query: str) -> tuple[str, str]:
    """(consistent, inconsistent) fixed short answers for one probe query.

    Mechanical template: identical query-echo prefix + the per-slug fact
    clause; the pair differs ONLY in the clause. Raises on unknown slug
    (fail-fast: a new judged fact run must add its clause pair here).
    """
    if slug not in MARGIN_CLAUSES:
        raise KeyError(f"no margin clause pair drafted for slug {slug!r}")
    consistent_clause, inconsistent_clause = MARGIN_CLAUSES[slug]
    prefix = f"Regarding your question about {_query_stub(query)}: "
    return prefix + consistent_clause, prefix + inconsistent_clause


def _git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=REPO_ROOT,
        ).stdout.strip()
    except Exception:
        return "unknown"


def build_margin_pools(conditions_dir: Path, corpus_dir: Path, *, smoke: bool) -> dict:
    """margin_rows.json + manifest + query store -> the fixed answer pools."""
    from issue1489_common import load_conditions_manifest

    margin_rows = json.loads((conditions_dir / "margin_rows.json").read_text())
    if not margin_rows:
        raise ValueError(f"{conditions_dir}/margin_rows.json is empty — no judged fact runs")
    manifest = load_conditions_manifest(conditions_dir)
    query_id_by_base: dict[str, str] = {
        r["base_row_id"]: str(r["query_id"]) for r in manifest if r["cell_id"] == "cell_plain"
    }
    query_store: dict[str, str] = {}
    with open(corpus_dir / "query_store.jsonl", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                qid = str(item.get("id") or item.get("query_id"))
                query_store[qid] = item.get("text") or item.get("query") or ""

    slugs: dict[str, dict] = {}
    for slug, sides in margin_rows.items():
        fact_rows = [r for r in manifest if r["cell_id"] == f"cell_{slug}"]
        if not fact_rows:
            raise ValueError(f"margin pools: no cell_{slug} rows in the conditions manifest")
        fact_text = fact_rows[0]["augment_text"]
        rows = []
        for side in ("relevant", "irrelevant"):
            for base_row_id in sides[side]:
                qid = query_id_by_base.get(base_row_id)
                if qid is None:
                    raise KeyError(
                        f"margin pools: {base_row_id} ({slug}/{side}) has no "
                        f"cell_plain manifest row"
                    )
                query = query_store.get(qid)
                if not query:
                    raise KeyError(f"margin pools: query {qid} missing from query_store")
                consistent, inconsistent = draft_margin_pair(slug, query)
                rows.append(
                    {
                        "base_row_id": base_row_id,
                        "side": side,
                        "query_id": qid,
                        "query": query,
                        "consistent": consistent,
                        "inconsistent": inconsistent,
                    }
                )
        slugs[slug] = {"fact_text": fact_text, "rows": rows}
        logger.info("[margin-pools] %s: drafted %d fixed +/- pairs", slug, len(rows))

    return {
        "issue": 1489,
        "template_version": MARGIN_TEMPLATE_VERSION,
        "smoke": smoke,
        "git_sha": _git_sha(),
        "timestamp_utc": datetime.datetime.utcnow().isoformat(),
        "slugs": slugs,
    }


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--conditions-dir", default="data/issue_1489/conditions")
    p.add_argument("--corpus-dir", default="data/issue_1489/hf_dl/corpus")
    p.add_argument("--smoke", action="store_true", help="label only; rows come from P0 --smoke")
    args = p.parse_args()
    conditions_dir = Path(args.conditions_dir)
    pools = build_margin_pools(conditions_dir, Path(args.corpus_dir), smoke=args.smoke)
    out_path = conditions_dir / "margin_pools.json"
    tmp = out_path.with_name(out_path.name + ".tmp")
    tmp.write_text(json.dumps(pools, indent=2, ensure_ascii=False))
    tmp.replace(out_path)
    n_pairs = sum(len(s["rows"]) for s in pools["slugs"].values())
    logger.info("[margin-pools] wrote %s (%d pairs)", out_path, n_pairs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
