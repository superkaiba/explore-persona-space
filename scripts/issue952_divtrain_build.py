#!/usr/bin/env python
"""Issue #952 diverse-train-injection — VM/CPU build of the injection set (leg A).

Same-issue inline follow-up (label ``diverse-train-injection``). Tests whether
adding divergence-domain (CCP-sensitive promptfoo) questions to the ridge-map
TRAIN pool lifts the out-of-distribution floor on the china divergence bank
(the parent maps were trained on LMSYS only).

This VM leg derives the injection SET from committed artifacts and writes a
text-free manifest for the pod GPU leg (``issue952_divtrain_gpu.py``) to
consume. It NEVER generates, judges, or captures.

Derivation (file-order logic, from committed artifacts — never trusted
arithmetic):

  * promptfoo/CCP-sensitive-prompts @ the pinned revision has 68 distinct
    subjects, 20 rows each (verified), enumerated in file order (the parent's
    ``_promptfoo_topic_rows`` logic, ``scripts/issue952_bank_build.py``).
  * USED promptfoo subjects = the subjects whose rows appear as a DIVERGENT
    promptfoo candidate in either the parent bank
    (``eval_results/issue_952/divergence_bank_queries.json``) OR the
    china-politics top-up round
    (``eval_results/issue_952/china-politics-topup/staging/new_candidates.json``).
    Parent used the first 15 subjects; the top-up used the next 24 (its
    ``new_candidates.json`` divergent refs) => 39 used.
  * UNUSED subjects = the remaining 29 (2 top-up gap subjects + subjects
    39..end), 580 rows total. Injection universe = ALL rows of each unused
    subject.
  * The LAST 4 unused subjects in file order are held out as the in-domain
    manipulation-check set; the remaining 25 join train.

Content discipline (BINDING): the promptfoo prompts are CCP-sensitive bank
items. This script NEVER prints/logs/quotes prompt text — rows are referenced
by {dataset, revision, file, row_index, column} + subject index + counts +
sha only. Text is resolved programmatically for the exact-match dedup and
never echoed.

Usage:
  uv run python scripts/issue952_divtrain_build.py
  uv run python scripts/issue952_divtrain_build.py --smoke   # 1 subject, <=8 rows
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import pathlib
import subprocess
import sys
import time

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "scripts"))

# Reuse the parent's promptfoo pins + text resolver verbatim (single source of truth).
from issue952_bank_build import (  # noqa: E402
    N_PROMPTFOO_TOPICS,
    PROMPTFOO_COLUMN,
    PROMPTFOO_DATASET,
    PROMPTFOO_FILE,
    PROMPTFOO_REVISION,
)

from explore_persona_space.experiments.issue_952.run_952 import (  # noqa: E402
    resolve_query_text,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("issue952.divtrain_build")

ISSUE_SLUG = "issue952_position_divergence"
LABEL = "diverse-train-injection"
N_HELD_OUT_SUBJECTS = 4  # last-in-file-order unused subjects reserved for the in-domain check
EXPECTED_ROWS_PER_SUBJECT = 20  # verified against the pinned revision; asserted below


def _git_sha() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=_REPO_ROOT).decode().strip()
        )
    except Exception:
        return "unknown"


def _promptfoo_subjects() -> tuple[pathlib.Path, list[dict]]:
    """Ordered distinct subjects (file order): [{index, first_row_index, n_rows}]."""
    from huggingface_hub import hf_hub_download

    p = pathlib.Path(
        hf_hub_download(
            PROMPTFOO_DATASET, PROMPTFOO_FILE, repo_type="dataset", revision=PROMPTFOO_REVISION
        )
    )
    subjects: list[dict] = []
    seen: dict[str, int] = {}
    with open(p, newline="", encoding="utf-8") as f:
        for i, rec in enumerate(csv.DictReader(f)):
            subj = rec["subject"]
            if subj not in seen:
                seen[subj] = len(subjects)
                subjects.append({"index": len(subjects), "first_row_index": i, "n_rows": 0})
            subjects[seen[subj]]["n_rows"] += 1
    for s in subjects:
        assert s["n_rows"] == EXPECTED_ROWS_PER_SUBJECT, (
            f"subject {s['index']} has {s['n_rows']} rows, expected {EXPECTED_ROWS_PER_SUBJECT} "
            "(promptfoo revision drift — re-derive)"
        )
    return p, subjects


def _subject_of_row(subjects: list[dict], row_index: int) -> int:
    """File-order subject index owning a promptfoo row_index (20-row spans)."""
    for s in subjects:
        if s["first_row_index"] <= row_index < s["first_row_index"] + s["n_rows"]:
            return s["index"]
    raise IndexError(f"row_index {row_index} maps to no subject")


def _divergent_promptfoo_rows(queries: list[dict]) -> set[int]:
    """The promptfoo row_indices used as DIVERGENT candidates in a bank/candidate set."""
    out: set[int] = set()
    for r in queries:
        if r.get("role") != "divergent":
            continue
        src = r.get("source", {})
        if "row_index" in src:
            out.add(int(src["row_index"]))
    return out


def _sha_text(t: str) -> str:
    return hashlib.sha256(t.encode("utf-8")).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description="issue #952 diverse-train-injection build (VM leg A)")
    ap.add_argument("--smoke", action="store_true", help="1 subject, <=8 rows")
    ap.add_argument("--out-dir", default=None, help="output base (default: repo root)")
    args = ap.parse_args()
    t0 = time.time()

    # Committed INPUT artifacts always resolve at the repo root (they are not
    # redirected by --out-dir, which only diverts this script's OWN output).
    in_eval_dir = _REPO_ROOT / "eval_results" / "issue_952"
    base = pathlib.Path(args.out_dir) if args.out_dir else _REPO_ROOT
    out_dir = base / "eval_results" / "issue_952" / LABEL
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path, subjects = _promptfoo_subjects()
    n_subjects = len(subjects)
    logger.info("[promptfoo] %d distinct subjects @ %s", n_subjects, PROMPTFOO_REVISION[:10])

    # ── derive USED subjects from committed artifacts (file-order logic) ────────────
    parent_bank = json.loads((in_eval_dir / "divergence_bank_queries.json").read_text())["queries"]
    topup_new = json.loads(
        (in_eval_dir / "china-politics-topup" / "staging" / "new_candidates.json").read_text()
    )["queries"]

    parent_rows = _divergent_promptfoo_rows(parent_bank)
    topup_rows = _divergent_promptfoo_rows(topup_new)
    used_subjects = {_subject_of_row(subjects, r) for r in (parent_rows | topup_rows)}
    parent_subjects = {_subject_of_row(subjects, r) for r in parent_rows}
    topup_subjects = {_subject_of_row(subjects, r) for r in topup_rows}
    assert parent_subjects == set(range(N_PROMPTFOO_TOPICS)), (
        f"parent promptfoo subjects {sorted(parent_subjects)} != first {N_PROMPTFOO_TOPICS} "
        "(file-order logic drift)"
    )
    logger.info(
        "[used] parent=%d topup=%d union=%d subjects",
        len(parent_subjects),
        len(topup_subjects),
        len(used_subjects),
    )

    # ── UNUSED subjects (file order) = injection universe ───────────────────────────
    unused = [s["index"] for s in subjects if s["index"] not in used_subjects]
    assert len(unused) == n_subjects - len(used_subjects)
    logger.info(
        "[unused] %d subjects, %d rows", len(unused), len(unused) * EXPECTED_ROWS_PER_SUBJECT
    )

    if args.smoke:
        # 2 subjects (1 train, 1 held-out), <=4 rows each, so the wiring smoke
        # exercises BOTH the augmented-train path and the in-domain check.
        unused = unused[:2]
    held_out = unused[-N_HELD_OUT_SUBJECTS:] if not args.smoke else unused[-1:]
    train_subjects = [s for s in unused if s not in set(held_out)]

    # disjointness — hard (∩ used AND ∩ final-kept bank promptfoo subjects) ──────────
    assert not (set(unused) & used_subjects), "injection subjects overlap USED subjects"
    kept_bank_pf_subjects: set[int] = set()
    for r in parent_bank:
        if r.get("role") == "divergent" and "row_index" in r.get("source", {}):
            kept_bank_pf_subjects.add(_subject_of_row(subjects, int(r["source"]["row_index"])))
    assert not (set(unused) & kept_bank_pf_subjects), (
        "injection subjects overlap bank promptfoo subjects"
    )

    # ── assemble injection rows (refs only) + exact-text dedup vs bank query text ────
    with open(csv_path, newline="", encoding="utf-8") as _f:
        row_reader = list(csv.DictReader(_f))
    # bank query text sha set (all categories, both roles), resolved programmatically.
    bank_texts: set[str] = set()
    for r in parent_bank + topup_new:
        try:
            bank_texts.add(_sha_text(resolve_query_text(r)))
        except Exception as e:  # a stray unresolvable ref must not silently pass
            logger.warning("[dedup] unresolved bank ref %s: %s", r.get("query_id"), e)

    rows: list[dict] = []
    dedup_drops = 0
    for subj_idx in unused:
        s = subjects[subj_idx]
        split = "indomain_check" if subj_idx in set(held_out) else "train"
        subj_rows = range(s["first_row_index"], s["first_row_index"] + s["n_rows"])
        if args.smoke:
            subj_rows = list(subj_rows)[:4]  # <=4 rows/subject => <=8 rows total in smoke
        for ri in subj_rows:
            src = {
                "dataset": PROMPTFOO_DATASET,
                "revision": PROMPTFOO_REVISION,
                "file": PROMPTFOO_FILE,
                "row_index": int(ri),
                "column": PROMPTFOO_COLUMN,
            }
            text = str(row_reader[ri][PROMPTFOO_COLUMN])
            if _sha_text(text) in bank_texts:
                dedup_drops += 1
                continue
            rows.append(
                {
                    "query_id": f"divtrain_{ri:04d}",
                    "subject_index": subj_idx,
                    "split": split,
                    "source": src,
                    "prompt_sha256": _sha_text(text),
                }
            )
    n_train = sum(1 for r in rows if r["split"] == "train")
    n_check = sum(1 for r in rows if r["split"] == "indomain_check")
    logger.info(
        "[injection] %d rows kept (train=%d, indomain_check=%d), %d dedup drops",
        len(rows),
        n_train,
        n_check,
        dedup_drops,
    )

    manifest = {
        "issue": 952,
        "label": LABEL,
        "smoke": args.smoke,
        "promptfoo": {
            "dataset": PROMPTFOO_DATASET,
            "revision": PROMPTFOO_REVISION,
            "file": PROMPTFOO_FILE,
            "column": PROMPTFOO_COLUMN,
            "n_distinct_subjects": n_subjects,
            "rows_per_subject": EXPECTED_ROWS_PER_SUBJECT,
        },
        "used_subjects": {
            "parent": sorted(parent_subjects),
            "topup": sorted(topup_subjects),
            "union_n": len(used_subjects),
        },
        "unused_subjects": sorted(unused),
        "held_out_subjects": sorted(held_out),
        "train_subjects": sorted(train_subjects),
        "n_rows": len(rows),
        "n_train_rows": n_train,
        "n_indomain_check_rows": n_check,
        "dedup_exact_text_drops": dedup_drops,
        "rows": rows,
        "content_discipline": "refs + subject index + sha only; no prompt text stored",
        "git_commit": _git_sha(),
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    out_path = out_dir / "injection_manifest.json"
    out_path.write_text(json.dumps(manifest, indent=2))
    logger.info("[write] %s (%.1fs)", out_path, time.time() - t0)


if __name__ == "__main__":
    main()
