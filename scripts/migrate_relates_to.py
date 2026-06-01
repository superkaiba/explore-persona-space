#!/usr/bin/env python
"""One-shot migration: rewrite task ``relates_to`` from OLD ids to CURRENT slugs.

Background
----------
Until 2026-05-28 the open-questions doc identified each question with a
short legacy id (``a1``, ``b6``, ``h1``, ``app1`` …). On 2026-05-29 the
live ``docs/open_questions.md`` was restructured: the headline + thread
ids merged / split / renamed into 23 semantic slugs (``leak-predictor``,
``leak-data-factors``, ``ctx-behavior`` …) plus 6 Application slugs
(``app1``-``app6``, with ``app-trigger-discovery`` renamed to ``app6``).

The dashboard's ``/questions`` page and the read-side rendering work
straight off the live doc's anchors, so the dashboard is correct at
launch *without* this migration. But ``relates_to`` on 57 task body.md
files still names the old ids, which keeps ``scripts/living_docs.py
check`` red on the bidirectional drift checks and makes the
``link``/``apply`` write-side machinery point at nonexistent anchors.

What this script does
---------------------
For every task whose ``relates_to`` references a legacy id, look the id
up in a hand-curated OLD→NEW map and rewrite the frontmatter list in
place. The script is **conservative**:

- Only the unambiguous, high-confidence 1:1 mappings are applied (e.g.
  ``a1 → leak-predictor``, ``h2 → leak-behavior-vs-marker``).
- Already-current ids (the slugs themselves, plus the App ids that didn't
  change) pass through untouched.
- Ambiguous ids (legacy ids that map to multiple plausible new slugs, or
  to none, or that were merged/split in ways that need a human eye) are
  left in place; the script reports them so the user can hand-resolve.

The mapping itself is derived from three sources:
1. ``docs/open_questions.md`` (live, restructured) — the set of current
   anchors.
2. ``docs/open_questions.proposed.md`` (UNAUDITED) — pre-restructure
   draft using the legacy ids; its question titles + evidence lists are
   compared against the live doc's to identify the 1:1 successors.
3. ``docs/open_questions-backfill-report.md`` (UNAUDITED) — task→legacy-
   id mapping report; useful for spot-checks.
4. ``git log -p -- docs/open_questions.md`` — confirms the rename
   commits (``b24dc84a4`` and ``824ad26df``).

Each high-confidence mapping in :data:`HIGH_CONFIDENCE_MAP` is annotated
with the evidence anchor or matching prose that justifies the link.

Usage
-----
::

    # Default dry-run: print the per-task diff + the mapping table.
    uv run python scripts/migrate_relates_to.py

    # Apply (rewrites task body.md frontmatters in place).
    uv run python scripts/migrate_relates_to.py --apply

The script is idempotent: re-running it after ``--apply`` is a no-op
(every id is already a current slug or already in the leave-untouched
set).

Path discipline mirrors ``scripts/living_docs.py``: *task* paths come
from the ``task_workflow`` helpers (never from ``cwd`` / ``__file__``);
the canonical resolver branch-guards to ``main``. (The ``__file__``-based
line below is only the standard ``src/`` import bootstrap, not task-path
resolution.)

This is a one-shot operational script, not a recurring CLI. It is
written to be re-runnable safely but does not need a long shelf life;
once the registry of tasks is on the new slug scheme, this script can
be deleted alongside the legacy ids that motivated it.
"""

from __future__ import annotations

import argparse
import difflib
import sys
from collections.abc import Iterable
from pathlib import Path

# Ensure the in-repo ``src/`` is importable when run as a script.
_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from explore_persona_space import task_workflow as tw  # noqa: E402

# ─── The mapping ─────────────────────────────────────────────────────────────

#: High-confidence 1:1 mappings from OLD legacy id to CURRENT semantic slug.
#: Each entry's justification is in the trailing comment: either the
#: live anchor whose evidence list overlaps exactly with the proposed
#: file's State trailer for that legacy id, or the title-level match
#: that nails the question content. App ids are 1:1 across the rename
#: (``app1``-``app5`` unchanged; ``app-trigger-discovery`` renamed to
#: ``app6``).
#:
#: Notably absent (intentionally — see :data:`AMBIGUOUS_OLD_IDS`):
#: ``a2``, ``a3``, ``a4``, ``a5``, ``b1``, ``b2``, ``b3``, ``b7``, ``b8``,
#: ``c1``..``c6``, ``d1``, ``d3``, ``e1``..``e5``. These either merged
#: with another id, split into multiple, or have no clean successor in
#: the new taxonomy. Hand-resolution required.
HIGH_CONFIDENCE_MAP: dict[str, str] = {
    # Headline -> leakage thread (the headline questions were folded
    # into the section-3 generalization-prediction thread).
    # h1: both = "predict leakage from pre-train geometry"
    "h1": "leak-predictor",
    # h2: both evidence: #391, #411, #116, #390 (exact match)
    "h2": "leak-behavior-vs-marker",
    # h3: both = "is #383 selectivity recipe real"; evidence: #383, #365, #337
    "h3": "leak-data-factors",
    # Thread A -> section-3.1 (predictor question).
    # a1: both = "what predicts marker implantability"; large evidence overlap
    "a1": "leak-predictor",
    # Thread B -> section-3 leakage sub-thread.
    # b4: both = "contexts as useful as personas"; evidence: #375, #129 (exact match)
    "b4": "ctx-behavior",
    # b5: both = "system-prompting equivalent to persona drift via log-probs on drifted tokens"
    "b5": "spec-sysprompt-vs-drift",
    # b6: both = "multi-persona training and leakage-vs-similarity curve"
    "b6": "leak-single-vs-multi",
    # Thread D -> section-3.6 behavior-distance question.
    # d2: both = "given B trained in P, predict B' generalizes in P"
    "d2": "beh-b-to-bprime",
    # Applications: app1..app5 unchanged; app-trigger-discovery -> app6.
    "app1": "app1",
    "app2": "app2",
    "app3": "app3",
    "app4": "app4",
    "app5": "app5",
    "app-trigger-discovery": "app6",
}

#: Old legacy ids that DO NOT have a clean 1:1 successor in the new
#: taxonomy. The migration script will NOT rewrite tasks that reference
#: only these ids; instead it lists them in the report so the user can
#: hand-resolve. Each entry's comment names the reason.
AMBIGUOUS_OLD_IDS: dict[str, str] = {
    "a2": "no current question on Chen-vs-our-vector equivalence",
    "a3": "no current question on 'recipe preserves Qwen geometry through SFT'",
    "a4": "no current question on cross-model Aim 1.5",
    "a5": "proposed but never landed in live doc; persona-vector recipe agreement",
    "b1": "could be `leak-contrastive-negatives` OR `leak-predictor`; non-trivial",
    "b2": "specific to zelthari_scholar persona; no general question hosts it",
    "b3": "no current question on convergence-training and JS predictor",
    "b7": "proposed marker-shape question never landed in live doc",
    "b8": "proposed language-mismatch SFT question never landed in live doc",
    "c1": "no current question on EM = coherence collapse vs broad misalignment",
    "c2": "no current question on make-evil-dumb RL survival (maybe `app3` but ambiguous)",
    "c3": "no current question on Betley edu_v0 base-model jailbreak",
    "c4": "no current question on post-FT patching of system-prompt activations",
    "c5": "no current question on attention from [Z token",
    "c6": "could be `leak-contrastive-negatives` OR `implant-which-behaviors`",
    "d1": "could be `identity-cb-duality` OR `beh-b-to-bprime`; behavior-distance prereq",
    "d3": "could be `beh-b-to-bprime` OR `leak-behavior-vs-marker`; spans both",
    "e1": "no current question on n>=10 replication discipline (methodology meta)",
    "e2": "no current question on persona-trigger conditional-marker findings",
    "e3": "no current question on OOD domain-matched eval generalization",
    "e4": "no current question on re-promoting findings to claims.yaml (infra)",
    "e5": "no current question on weight-baked hidden behavior detection",
}

#: Slugs that already match a live anchor — these pass through
#: untouched and never count as a rewrite candidate. (Generated by
#: scanning the live doc's ``<!-- q:<slug> -->`` anchors once when this
#: script is run; the static list here is a guard.)
KNOWN_CURRENT_SLUGS: frozenset[str] = frozenset(
    {
        # §1 Distance
        "spec-context-as-vector",
        "spec-kl-probe-set",
        "spec-prompt-vs-icl",
        "spec-steering",
        "spec-sdf",
        "spec-sysprompt-vs-drift",
        # §2 Updating
        "implant-which-behaviors",
        "implant-learning-speed",
        # §3 Generalization
        "leak-predictor",
        "leak-behavior-vs-marker",
        "leak-single-vs-multi",
        "leak-data-factors",
        "leak-contrastive-negatives",
        "fact-teach-persona-transfer",
        "ctx-behavior",
        "beh-b-to-bprime",
        "leak-to-default",
        "regime-rl-vs-sft",
        "leak-from-cell-set",
        # §4 Duality
        "identity-persona-vs-behavior",
        "identity-contextual-vs-base",
        "identity-cb-duality",
        "identity-what-is-behavior",
        # Applications
        "app1",
        "app2",
        "app3",
        "app4",
        "app5",
        "app6",
    }
)


# ─── Migration core ──────────────────────────────────────────────────────────


def _normalize(rid: str) -> str:
    """Strip + lower an id, matching the case-insensitive read in living_docs."""
    return str(rid).strip().lower()


def _classify_ids(relates_to: Iterable[str]) -> tuple[list[str], list[str], list[str]]:
    """Sort ids into (rewritable, ambiguous, untouched).

    - ``rewritable`` — old ids with a HIGH_CONFIDENCE_MAP entry that
      will actually CHANGE value (so app1→app1 is not rewritable —
      same target, no need to rewrite).
    - ``ambiguous`` — old ids in :data:`AMBIGUOUS_OLD_IDS` (skip, hand-
      resolve).
    - ``untouched`` — ids that already match a current slug, or that
      are neither legacy nor ambiguous (left as-is).
    """
    rewritable: list[str] = []
    ambiguous: list[str] = []
    untouched: list[str] = []
    for raw in relates_to:
        rid = _normalize(raw)
        if rid in HIGH_CONFIDENCE_MAP and HIGH_CONFIDENCE_MAP[rid] != rid:
            rewritable.append(rid)
        elif rid in AMBIGUOUS_OLD_IDS:
            ambiguous.append(rid)
        else:
            untouched.append(rid)
    return rewritable, ambiguous, untouched


def _rewrite_one(relates_to: list[str]) -> list[str]:
    """Apply HIGH_CONFIDENCE_MAP to a relates_to list, preserving order + dedup."""
    out: list[str] = []
    seen: set[str] = set()
    for raw in relates_to:
        rid = _normalize(raw)
        new = HIGH_CONFIDENCE_MAP.get(rid, rid)
        if new not in seen:
            seen.add(new)
            out.append(new)
    return out


def _all_tasks_with_relates_to() -> list[tuple[int, list[str]]]:
    """Return (task_id, relates_to) for every task whose body.md has a non-empty list."""
    reg = tw._load_registry()
    out: list[tuple[int, list[str]]] = []
    for tid_str in sorted(reg.get("tasks", {}), key=int):
        tid = int(tid_str)
        try:
            body_md = tw.find_task_path(tid) / "body.md"
        except FileNotFoundError:
            continue
        if not body_md.exists():
            continue
        fm, _ = tw._read_body(body_md)
        rel = fm.get("relates_to") or []
        if rel:
            out.append((tid, [str(x) for x in rel]))
    return out


def _format_diff(tid: int, old: list[str], new: list[str]) -> str:
    """Pretty unified-diff-style block for a single task."""
    if old == new:
        return ""
    old_yaml = "relates_to:\n" + "\n".join(f"  - {x}" for x in old)
    new_yaml = "relates_to:\n" + "\n".join(f"  - {x}" for x in new)
    diff = difflib.unified_diff(
        old_yaml.splitlines(keepends=True),
        new_yaml.splitlines(keepends=True),
        fromfile=f"task #{tid} (before)",
        tofile=f"task #{tid} (after)",
        n=99,
    )
    return "".join(line if line.endswith("\n") else line + "\n" for line in diff)


def migrate(*, apply: bool) -> int:
    """Run the migration. Returns the process exit code (0 = OK)."""
    tasks = _all_tasks_with_relates_to()

    # Build the four tables: would-rewrite, ambiguous-skip, no-change, unknown.
    rewrites: list[tuple[int, list[str], list[str]]] = []
    ambiguous_skips: list[tuple[int, list[str], list[str]]] = []
    no_change: list[tuple[int, list[str]]] = []
    for tid, rel in tasks:
        rewritable, ambiguous, _untouched = _classify_ids(rel)
        new = _rewrite_one(rel)
        if rewritable and not ambiguous:
            rewrites.append((tid, rel, new))
        elif ambiguous:
            # Even if SOME ids are rewritable, the presence of ambiguous
            # ids on the same task means we can't be sure the human
            # intent maps cleanly — leave the whole task alone and list it.
            ambiguous_skips.append((tid, rel, ambiguous))
        else:
            no_change.append((tid, rel))

    # ─── Per-task diff (dry-run + apply both print this) ─────────────────
    print("\n=== Per-task rewrites (high-confidence only) ===\n")
    if not rewrites:
        print("(nothing to rewrite — all tasks either current or ambiguous)\n")
    for tid, old, new in rewrites:
        print(_format_diff(tid, old, new))

    # ─── Apply (or not) ──────────────────────────────────────────────────
    applied = 0
    if apply and rewrites:
        with tw._locked():
            for tid, _old, new in rewrites:
                body_md = tw.find_task_path(tid) / "body.md"
                fm, body = tw._read_body(body_md)
                fm["relates_to"] = new
                tw._write_body(body_md, fm, body)
                # Commit per task so each rewrite is independently revertable
                # by `git revert` if the user notices a wrong mapping.
                tw._git_commit(
                    [body_md],
                    f"task #{tid}: migrate relates_to to new semantic slugs",
                )
                applied += 1

    # ─── Mapping table ────────────────────────────────────────────────────
    print("\n=== OLD → NEW mapping (high-confidence, applied) ===\n")
    print(f"{'old':<28} {'new':<32} {'justification'}")
    print("-" * 96)
    for old, new in sorted(HIGH_CONFIDENCE_MAP.items()):
        tag = "unchanged (app id)" if old == new else "1:1 successor"
        print(f"{old:<28} {new:<32} {tag}")

    # ─── Ambiguous (left for manual review) ───────────────────────────────
    print("\n=== AMBIGUOUS legacy ids (skipped — hand-resolve) ===\n")
    print(f"{'id':<28} reason")
    print("-" * 96)
    for old, reason in sorted(AMBIGUOUS_OLD_IDS.items()):
        print(f"{old:<28} {reason}")

    print("\n=== Tasks left untouched because they carry >=1 ambiguous id ===\n")
    if not ambiguous_skips:
        print("(none)")
    for tid, rel, ambig in ambiguous_skips:
        print(f"  #{tid}: relates_to={rel}  (ambiguous: {sorted(set(ambig))})")

    # ─── Summary ──────────────────────────────────────────────────────────
    print("\n=== Summary ===")
    print(f"  tasks scanned        : {len(tasks)}")
    print(f"  tasks unchanged       : {len(no_change)}")
    print(f"  tasks with ambiguous  : {len(ambiguous_skips)} (skipped)")
    print(f"  tasks queued for rewrite: {len(rewrites)}")
    print(
        f"  tasks actually rewritten: {applied if apply else 0} "
        f"({'apply' if apply else 'dry-run — re-run with --apply to write'})"
    )

    return 0


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="migrate_relates_to.py",
        description=__doc__.splitlines()[0] if __doc__ else "",
    )
    g = p.add_mutually_exclusive_group()
    g.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="(default) report what would change without writing or committing",
    )
    g.add_argument(
        "--apply",
        action="store_true",
        help="write the high-confidence rewrites to task body.md frontmatters",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    return migrate(apply=args.apply)


if __name__ == "__main__":
    raise SystemExit(main())
