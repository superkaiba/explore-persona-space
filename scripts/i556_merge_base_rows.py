"""Inject #528's validating ``kind=="base"`` judge rows into #556's judge output.

#556 plan v3 §4.2 item 4c (the base-reuse mechanism). ``i528_phase5_analyze.py``
consumes base rows ONLY from ``judge_scores.json`` (``_group(rows, kind="base")``);
``base_headroom_judge.json`` is a judge OUTPUT view, never an analyze input. So
base reuse = append the parent's 400 validating base rows (2 arms x 5 eval
contexts x 40 prompts) into this run's ``eval_results/<ISSUE_SLUG>/
judge_scores.json`` AFTER the #556 judge ran with ``--skip-base`` and BEFORE
``i528_phase5_analyze.py``.

Fail-loud checks:
  - refuses to run when ``ISSUE_SLUG == "issue_528"`` (would mutate the
    parent's committed artifact in place);
  - asserts EXACTLY 400 validating base rows in the parent file, in 10
    (arm, eval_context) groups of 40;
  - asserts per-group q coverage against the parent's sha256_test pin for
    validating (rows sorted by q_idx must hash to the committed Q_test pin
    in ``eval_results/issue_528/preflight_summary.json``);
  - idempotent: exits 0 without rewriting when the target already contains
    base rows.

CLI:
    I528_ISSUE_SLUG=issue_556 uv run python scripts/i556_merge_base_rows.py
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import logging
import subprocess
import sys
from pathlib import Path

from explore_persona_space.experiments.i528_data import ISSUE_SLUG

logger = logging.getLogger("i556.merge_base_rows")

# Read-only parent artifacts (committed on main) — the ONLY legitimate
# issue_528 paths in the #556 pipeline besides the run-all's pin assert.
PARENT_JUDGE_PATH = Path("eval_results/issue_528/judge_scores.json")
PARENT_PREFLIGHT_PATH = Path("eval_results/issue_528/preflight_summary.json")

TARGET_JUDGE_PATH = Path(f"eval_results/{ISSUE_SLUG}/judge_scores.json")

TRAIT = "validating"
EXPECTED_N_BASE = 400  # 2 arms x 5 eval contexts x 40 prompts
EXPECTED_GROUPS = 10  # 2 arms x 5 eval contexts
EXPECTED_PER_GROUP = 40


def _git() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def _sha256_list(strings: list[str]) -> str:
    """Canonical-JSON sha256, byte-identical to the preflight pin encoding."""
    blob = json.dumps(strings, ensure_ascii=False, sort_keys=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def extract_parent_base_rows() -> list[dict]:
    """Return #528's 400 validating base rows after the fail-loud checks."""
    parent = json.loads(PARENT_JUDGE_PATH.read_text())
    if parent.get("kind") != "judge_scores":
        raise SystemExit(f"{PARENT_JUDGE_PATH}: kind={parent.get('kind')!r}, expected judge_scores")
    base_rows = [r for r in parent["rows"] if r.get("kind") == "base" and r.get("trait") == TRAIT]
    if len(base_rows) != EXPECTED_N_BASE:
        raise SystemExit(
            f"{PARENT_JUDGE_PATH}: found {len(base_rows)} validating base rows, "
            f"expected exactly {EXPECTED_N_BASE} (2 arms x 5 contexts x 40 prompts). "
            "Base reuse is INVALID — fall back to re-judging the validating base "
            "from the HF-hosted raw_generations_base/ (plan §8 pre-authorized path)."
        )

    groups: dict[tuple[str, str], list[dict]] = {}
    for r in base_rows:
        groups.setdefault((r["arm"], r["eval_context"]), []).append(r)
    if len(groups) != EXPECTED_GROUPS or any(len(v) != EXPECTED_PER_GROUP for v in groups.values()):
        shape = {f"{a}/{c}": len(v) for (a, c), v in sorted(groups.items())}
        raise SystemExit(
            f"{PARENT_JUDGE_PATH}: validating base rows malformed — expected "
            f"{EXPECTED_GROUPS} (arm, context) groups of {EXPECTED_PER_GROUP}; got {shape}"
        )

    # Q-coverage pin: every group's q list (sorted by q_idx) must hash to the
    # committed validating sha256_test pin (the #517 disjoint-bank defense).
    preflight = json.loads(PARENT_PREFLIGHT_PATH.read_text())
    pins = {x["trait"]: x for x in preflight["qbank_summaries"]}
    expected_sha = pins[TRAIT]["sha256_test"]
    for (arm, ctx), rows in sorted(groups.items()):
        qs = [r["q"] for r in sorted(rows, key=lambda r: r["q_idx"])]
        got = _sha256_list(qs)
        if got != expected_sha:
            raise SystemExit(
                f"Q-coverage pin mismatch for base group arm={arm} ctx={ctx}: "
                f"sha256(q list)={got[:12]}… != committed validating sha256_test "
                f"{expected_sha[:12]}…. Refusing to merge — the parent base rows "
                "do not cover the pinned Q_test."
            )
    logger.info(
        "Parent base slice OK: %d rows, %d groups of %d, q-coverage sha %s…",
        len(base_rows),
        len(groups),
        EXPECTED_PER_GROUP,
        expected_sha[:12],
    )
    return base_rows


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    if ISSUE_SLUG == "issue_528":
        raise SystemExit(
            "I528_ISSUE_SLUG is unset (or set to issue_528) — refusing to merge "
            "base rows INTO the parent's committed judge_scores.json. Set "
            "I528_ISSUE_SLUG=issue_556 (or a scratch slug for the regression test)."
        )
    if not TARGET_JUDGE_PATH.exists():
        raise SystemExit(
            f"{TARGET_JUDGE_PATH} not found — run the #556 judge "
            "(i528_phase4_judge.py --backend sync --skip-base) first."
        )

    target = json.loads(TARGET_JUDGE_PATH.read_text())
    if target.get("kind") != "judge_scores":
        raise SystemExit(f"{TARGET_JUDGE_PATH}: kind={target.get('kind')!r}, expected judge_scores")
    n_existing_base = sum(
        1 for r in target["rows"] if r.get("kind") == "base" and r.get("trait") == TRAIT
    )
    if n_existing_base:
        logger.info(
            "%s already contains %d %s base rows — merge already done, nothing to do.",
            TARGET_JUDGE_PATH,
            n_existing_base,
            TRAIT,
        )
        return 0

    base_rows = extract_parent_base_rows()
    target["rows"] = list(target["rows"]) + base_rows
    target["n_scored"] = len(target["rows"])
    target["base_rows_merged_from"] = str(PARENT_JUDGE_PATH)
    target["n_base_rows_merged"] = len(base_rows)
    target["merge_git_commit"] = _git()
    target["merge_ts"] = _dt.datetime.utcnow().isoformat() + "Z"
    TARGET_JUDGE_PATH.write_text(json.dumps(target, indent=2, ensure_ascii=False))
    logger.info(
        "Merged %d parent base rows into %s (total rows now %d).",
        len(base_rows),
        TARGET_JUDGE_PATH,
        len(target["rows"]),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
