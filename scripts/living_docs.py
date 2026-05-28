#!/usr/bin/env python
"""Mechanical core for the living research docs (``docs/open_questions.md``).

This module is the *apply-only* layer of the living-docs ⇄ ``/issue``
integration (see ``docs/living-docs-workflow-integration-plan.md``). It
makes NO semantic judgements: the ``living-docs-updater`` agent proposes
diffs and the user confirms them; this module only applies what is
already confirmed, links a task to one or more open questions, and lints
for drift.

Three public operations (importable + CLI):

- :func:`apply` — apply a CONFIRMED patch to ``docs/open_questions.md``
  (and ``docs/papers.md`` when the patch touches it), prepend a dated
  changelog line, and commit atomically (single ``flock`` + one git
  commit). The patch is a structured replacement set produced by the
  updater agent and confirmed by the user; this function neither
  interprets nor second-guesses it.
- :func:`link` — write a flat ``relates_to`` list onto a task's
  ``body.md`` YAML frontmatter AND append the task ref (``#N``) to each
  named question's ``> **State:**`` trailer line in
  ``open_questions.md`` (matched by the ``<!-- q:<id> -->`` anchor). A
  question id with no anchor yet gets a minimal stub created.
- :func:`check` — lint for drift and exit nonzero when any is found:
  (a) ``relates_to`` ⇄ question-evidence agree both directions; (b)
  every ``completed`` task with ``has_clean_result`` appears in some
  question's evidence; (c) every evidence ``#N`` resolves to a real
  task; (d) flag questions whose State date is stale relative to a newer
  linked result.

Schema (locked 2026-05-28). Each open question is an H2/bold heading
followed immediately by an HTML-comment anchor, then prose, then a State
trailer::

    **A1. What predicts marker implantability?** <!-- q:a1 -->
    ... prose ...
    > **State:** 🌿 budding · MODERATE · updated 2026-05-28 · evidence: #207, #380

Maturity emojis: 🌱 seedling · 🌿 budding · 🌳 evergreen.

Path / repo / git discipline mirrors
``explore_persona_space.task_workflow``: paths are resolved via the
``task_workflow`` helpers (never from ``cwd`` / ``__file__``); every
mutation holds the same ``flock`` on ``~/.task-workflow/lock`` and lands
as a single git commit. Tests inject an alternate root via
:class:`LivingDocsPaths` so they never touch the real ``docs/`` or
``tasks/``.

Fail-loud throughout: no ``try/except: pass``, no silent defaults, no
dummy fallbacks. A malformed doc, a missing anchor where one is
required, or an unresolvable task ref raises.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import json
import os
import re
import subprocess
import sys
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Ensure the in-repo ``src/`` is importable when run as a script.
_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from explore_persona_space import task_workflow as tw  # noqa: E402

# ─── Schema constants ──────────────────────────────────────────────────────

#: Maturity glyphs in increasing-confidence order. ``check`` does not
#: enforce any particular one, but ``link`` stubs use the seedling glyph.
MATURITY_SEEDLING = "🌱"
MATURITY_BUDDING = "🌿"
MATURITY_EVERGREEN = "🌳"
MATURITY_EMOJIS = (MATURITY_SEEDLING, MATURITY_BUDDING, MATURITY_EVERGREEN)

#: Confidence scale, matching the clean-result tag scale.
CONFIDENCE_LEVELS = ("LOW", "MODERATE", "HIGH")

#: Anchor for a question id, e.g. ``<!-- q:a1 -->``. Ids are
#: case-insensitive on read (normalized to lower) and lower on write.
_ANCHOR_RE = re.compile(r"<!--\s*q:([A-Za-z0-9_.\-]+)\s*-->")

#: A State trailer line. Captures the evidence-list tail so it can be
#: rewritten in place.
#:   > **State:** 🌿 budding · MODERATE · updated 2026-05-28 · evidence: #207, #380
#: The maturity segment is ``<emoji>`` optionally followed by a word
#: (e.g. ``🌿 budding``), then ``·`` separates it from the confidence.
_STATE_RE = re.compile(
    r"^(?P<prefix>>\s*\*\*State:\*\*\s*)"
    r"(?P<maturity>\S+?)"
    r"(?:\s+(?P<maturity_word>[^·]*?))?\s*·\s*"
    r"(?P<confidence>LOW|MODERATE|HIGH)\s*·\s*"
    r"updated\s+(?P<date>\d{4}-\d{2}-\d{2})\s*·\s*"
    r"evidence:\s*(?P<evidence>.*?)\s*$"
)

#: A task reference inside an evidence list (e.g. ``#207``).
_EVIDENCE_REF_RE = re.compile(r"#(\d+)")

#: Heading line carrying a question anchor — used by ``link`` when it
#: must create a stub: we want the changelog + stub to look native.
_DATE_FMT = "%Y-%m-%d"

#: A changelog block lives at the very top of open_questions.md, right
#: after the H1. We bracket it with HTML comments so it is machine
#: locatable and humans see a normal "## Changelog" section.
_CHANGELOG_BEGIN = "<!-- living-docs-changelog:begin -->"
_CHANGELOG_END = "<!-- living-docs-changelog:end -->"
_CHANGELOG_HEADING = "## Changelog"


# ─── Path injection ────────────────────────────────────────────────────────


@dataclass
class LivingDocsPaths:
    """Resolved filesystem locations for the living-docs surface.

    Tests construct this with a temp root so the module never touches the
    real ``docs/`` or ``tasks/``. Production code calls
    :meth:`from_repo`, which resolves through the canonical
    ``task_workflow`` helpers (branch-guarded, never ``cwd`` /
    ``__file__`` derived).

    Attributes
    ----------
    repo_root : Path
        Absolute repo root.
    open_questions : Path
        ``docs/open_questions.md``.
    papers : Path
        ``docs/papers.md``.
    lock_path : Path
        The flock file shared with ``task_workflow`` so doc mutations and
        task mutations serialise against each other.
    """

    repo_root: Path
    open_questions: Path
    papers: Path
    lock_path: Path

    @classmethod
    def from_repo(cls) -> LivingDocsPaths:
        """Resolve the living-docs paths from the canonical repo root.

        Uses :func:`explore_persona_space.task_workflow.repo_root`, which
        branch-guards to ``main`` and refuses on detached / non-``main``
        HEAD. Never derives paths from ``cwd`` or ``__file__``.
        """
        root = tw.repo_root()
        return cls(
            repo_root=root,
            open_questions=root / "docs" / "open_questions.md",
            papers=root / "docs" / "papers.md",
            lock_path=tw.LOCK_PATH,
        )


# ─── Patch model ───────────────────────────────────────────────────────────


@dataclass
class DocPatch:
    """A confirmed, mechanical patch to the living docs.

    The ``living-docs-updater`` agent proposes this and the user
    confirms it; :func:`apply` then applies it verbatim. It is a set of
    exact string replacements (anchored, so they fail loud if the target
    text drifted) plus optional appends. No fuzzy matching.

    Attributes
    ----------
    open_questions_replacements : list[tuple[str, str]]
        Ordered ``(old, new)`` pairs applied to
        ``open_questions.md``. Each ``old`` must occur exactly once;
        zero or multiple matches raise. Applied in order.
    open_questions_appends : list[str]
        Blocks appended verbatim to the end of ``open_questions.md``
        (e.g. a brand-new question section). Each is separated from the
        prior content by a blank line.
    papers_replacements : list[tuple[str, str]]
        Same semantics for ``papers.md``.
    papers_appends : list[str]
        Same semantics for ``papers.md``.
    changelog_line : str
        One-sentence human description of what this patch did. Prepended
        to the changelog block with today's date. Required (the changelog
        is the audit trail).
    """

    changelog_line: str
    open_questions_replacements: list[tuple[str, str]] = field(default_factory=list)
    open_questions_appends: list[str] = field(default_factory=list)
    papers_replacements: list[tuple[str, str]] = field(default_factory=list)
    papers_appends: list[str] = field(default_factory=list)

    def touches_open_questions(self) -> bool:
        """True if the patch changes ``open_questions.md``."""
        return bool(self.open_questions_replacements or self.open_questions_appends)

    def touches_papers(self) -> bool:
        """True if the patch changes ``papers.md``."""
        return bool(self.papers_replacements or self.papers_appends)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DocPatch:
        """Build a :class:`DocPatch` from a plain dict (CLI / JSON input).

        Replacement pairs may arrive as 2-element lists; they are
        coerced to tuples. ``changelog_line`` is required.
        """
        if "changelog_line" not in data or not str(data["changelog_line"]).strip():
            raise ValueError("patch requires a non-empty 'changelog_line'")

        def _pairs(key: str) -> list[tuple[str, str]]:
            raw = data.get(key, []) or []
            out: list[tuple[str, str]] = []
            for item in raw:
                if len(item) != 2:
                    raise ValueError(f"{key} entries must be [old, new] pairs, got {item!r}")
                out.append((str(item[0]), str(item[1])))
            return out

        return cls(
            changelog_line=str(data["changelog_line"]).strip(),
            open_questions_replacements=_pairs("open_questions_replacements"),
            open_questions_appends=[str(x) for x in (data.get("open_questions_appends") or [])],
            papers_replacements=_pairs("papers_replacements"),
            papers_appends=[str(x) for x in (data.get("papers_appends") or [])],
        )


# ─── Small helpers ─────────────────────────────────────────────────────────


def _today() -> str:
    """Return today's date as ``YYYY-MM-DD`` (UTC)."""
    return datetime.now(tz=UTC).strftime(_DATE_FMT)


@contextlib.contextmanager
def _locked(lock_path: Path) -> Iterator[None]:
    """Hold an exclusive flock for the duration of a mutation.

    Mirrors ``task_workflow._locked`` so doc writes serialise with task
    writes (they share ``~/.task-workflow/lock`` in production).
    """
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(lock_path, os.O_WRONLY | os.O_CREAT, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def _git(args: list[str], *, cwd: Path, check: bool = True) -> subprocess.CompletedProcess[str]:
    """Run a git command in ``cwd`` with a sanitized env.

    Strips inherited ``GIT_DIR`` / ``GIT_WORK_TREE`` / ``GIT_INDEX_FILE``
    / ``GIT_OBJECT_DIRECTORY`` so a caller's env cannot redirect the
    commit, matching ``task_workflow._run_git``.
    """
    env = dict(os.environ)
    for k in ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_OBJECT_DIRECTORY"):
        env.pop(k, None)
    return subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        env=env,
        check=check,
        capture_output=True,
        text=True,
    )


def _git_commit(paths: list[Path], message: str, *, repo_root: Path) -> None:
    """Stage the given paths and create a single commit.

    Uses ``git commit --only -- <paths>`` so unrelated staged work is not
    swept in (parallel agents share the index). Skips the commit when
    nothing changed for OUR paths, or entirely when ``TASK_PY_NO_COMMIT``
    is set (tests). Pushes when ``TASK_PY_AUTO_PUSH`` is set.
    """
    if os.environ.get("TASK_PY_NO_COMMIT") == "1":
        return
    rel = [str(p.relative_to(repo_root)) for p in paths]
    existing = [str(p.relative_to(repo_root)) for p in paths if p.exists()]
    if existing:
        _git(["add", "--", *existing], cwd=repo_root)
    staged = _git(["diff", "--cached", "--quiet", "--", *rel], cwd=repo_root, check=False)
    if staged.returncode == 0:
        return
    _git(["commit", "-m", f"{message}\n\n[living_docs.py]", "--only", "--", *rel], cwd=repo_root)
    if os.environ.get("TASK_PY_AUTO_PUSH") == "1":
        _git(["push"], cwd=repo_root, check=False)


def _read(path: Path) -> str:
    """Read a doc file, failing loud if it is missing."""
    if not path.exists():
        raise FileNotFoundError(f"living doc not found: {path}")
    return path.read_text()


def _write_atomic(path: Path, text: str) -> None:
    """Write a doc file atomically (write-temp + rename)."""
    text = text if text.endswith("\n") else text + "\n"
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)


def _apply_replacements(text: str, replacements: list[tuple[str, str]], *, where: str) -> str:
    """Apply exact-string replacements, failing loud on miss / ambiguity.

    Each ``old`` must occur exactly once. Zero matches means the target
    drifted (the agent's proposal is stale); multiple matches means the
    anchor was not specific enough. Both are errors — we never apply a
    fuzzy or partial patch.
    """
    for old, new in replacements:
        count = text.count(old)
        if count == 0:
            raise ValueError(
                f"{where}: replacement target not found (text drifted?):\n  {old[:120]!r}"
            )
        if count > 1:
            raise ValueError(
                f"{where}: replacement target occurs {count} times (ambiguous):\n  {old[:120]!r}"
            )
        text = text.replace(old, new)
    return text


def _append_blocks(text: str, blocks: list[str]) -> str:
    """Append blocks to a doc, each separated by a blank line."""
    out = text.rstrip("\n")
    for block in blocks:
        out += "\n\n" + block.rstrip("\n")
    return out + "\n"


# ─── Changelog ─────────────────────────────────────────────────────────────


def _prepend_changelog(text: str, line: str, *, date: str) -> str:
    """Prepend a dated changelog entry to ``open_questions.md``.

    The changelog lives in a fenced block right after the H1. If no block
    exists yet, one is created immediately after the first ``# `` H1 line
    (or at the very top when there is no H1). Newest entries first.
    """
    entry = f"- **{date}** — {line.strip()}"
    if _CHANGELOG_BEGIN in text and _CHANGELOG_END in text:
        begin = text.index(_CHANGELOG_BEGIN) + len(_CHANGELOG_BEGIN)
        end = text.index(_CHANGELOG_END)
        block = text[begin:end]
        # Insert the new entry directly after the heading line inside the
        # block (newest first), preserving the rest.
        lines = block.splitlines()
        insert_at = 0
        for i, ln in enumerate(lines):
            if ln.strip() == _CHANGELOG_HEADING:
                insert_at = i + 1
                break
        # Skip a single blank line after the heading so entries stay flush.
        if insert_at < len(lines) and lines[insert_at].strip() == "":
            insert_at += 1
        lines.insert(insert_at, entry)
        new_block = "\n".join(lines)
        return text[:begin] + new_block + text[end:]
    # No block yet — build one.
    new_block = f"{_CHANGELOG_BEGIN}\n{_CHANGELOG_HEADING}\n\n{entry}\n{_CHANGELOG_END}\n"
    lines = text.splitlines(keepends=True)
    for i, ln in enumerate(lines):
        if ln.startswith("# ") and not ln.startswith("## "):
            # Insert after the H1 (and a following blank line if present).
            insert_at = i + 1
            if insert_at < len(lines) and lines[insert_at].strip() == "":
                insert_at += 1
            head = "".join(lines[:insert_at])
            tail = "".join(lines[insert_at:])
            return head + "\n" + new_block + "\n" + tail
    # No H1 — prepend.
    return new_block + "\n" + text


# ─── apply ─────────────────────────────────────────────────────────────────


def apply(
    task_id: int,
    patch: DocPatch,
    *,
    paths: LivingDocsPaths | None = None,
) -> list[Path]:
    """Apply a CONFIRMED patch to the living docs and commit atomically.

    Applies ``patch.open_questions_*`` to ``open_questions.md`` (always,
    because it also receives the changelog line) and
    ``patch.papers_*`` to ``papers.md`` when the patch touches it.
    Prepends a dated changelog line to ``open_questions.md``. Holds the
    shared flock and lands everything in a single git commit.

    Parameters
    ----------
    task_id : int
        Task this patch was produced for (used only in the commit
        message; the changelog text is the human-facing record).
    patch : DocPatch
        The confirmed patch. ``changelog_line`` is required.
    paths : LivingDocsPaths, optional
        Injected paths (tests). Defaults to
        :meth:`LivingDocsPaths.from_repo`.

    Returns
    -------
    list[Path]
        The doc files that were modified.
    """
    paths = paths or LivingDocsPaths.from_repo()
    if not patch.changelog_line.strip():
        raise ValueError("patch.changelog_line is required (the changelog is the audit trail)")
    today = _today()
    touched: list[Path] = []
    with _locked(paths.lock_path):
        oq_text = _read(paths.open_questions)
        oq_text = _apply_replacements(
            oq_text, patch.open_questions_replacements, where="open_questions.md"
        )
        oq_text = _append_blocks(oq_text, patch.open_questions_appends)
        oq_text = _prepend_changelog(oq_text, patch.changelog_line, date=today)
        _write_atomic(paths.open_questions, oq_text)
        touched.append(paths.open_questions)

        if patch.touches_papers():
            pp_text = _read(paths.papers)
            pp_text = _apply_replacements(pp_text, patch.papers_replacements, where="papers.md")
            pp_text = _append_blocks(pp_text, patch.papers_appends)
            _write_atomic(paths.papers, pp_text)
            touched.append(paths.papers)

        _git_commit(
            touched,
            f"living-docs: apply #{task_id} — {patch.changelog_line[:60]}",
            repo_root=paths.repo_root,
        )
    return touched


# ─── Anchor / State parsing ────────────────────────────────────────────────


def _find_anchor_line(text: str, q_id: str) -> int | None:
    """Return the 0-based line index carrying the ``<!-- q:<id> -->``
    anchor (case-insensitive on id), or None if absent.
    """
    target = q_id.strip().lower()
    for i, line in enumerate(text.splitlines()):
        m = _ANCHOR_RE.search(line)
        if m and m.group(1).lower() == target:
            return i
    return None


def _find_state_line_for_anchor(lines: list[str], anchor_idx: int) -> int | None:
    """Given the anchor line index, return the index of the question's
    State trailer.

    Searches forward from the anchor until the next question anchor or a
    horizontal rule (``---``) — the State line for a question always sits
    within its own section.
    """
    for i in range(anchor_idx + 1, len(lines)):
        line = lines[i]
        if _STATE_RE.match(line):
            return i
        # Section boundary: another question anchor or a hrule.
        if i != anchor_idx and _ANCHOR_RE.search(line):
            return None
        if line.strip() == "---":
            return None
    return None


def _parse_evidence(evidence_str: str) -> list[int]:
    """Parse the evidence tail of a State line into a list of task ids."""
    return [int(m.group(1)) for m in _EVIDENCE_REF_RE.finditer(evidence_str)]


def _format_state_line(
    *,
    prefix: str,
    maturity: str,
    maturity_word: str,
    confidence: str,
    date: str,
    evidence_ids: list[int],
) -> str:
    """Rebuild a State trailer line from its parts (evidence newest last,
    de-duplicated, ascending).
    """
    uniq = sorted(set(evidence_ids))
    ev = ", ".join(f"#{n}" for n in uniq)
    word = (maturity_word or "").strip()
    word_part = f"{word} " if word else ""
    return f"{prefix}{maturity} {word_part}· {confidence} · updated {date} · evidence: {ev}"


def _stub_question(q_id: str) -> str:
    """Build a minimal new-question block for an id with no anchor yet."""
    qid = q_id.strip().lower()
    today = _today()
    state = f"> **State:** {MATURITY_SEEDLING} seedling · LOW · updated {today} · evidence: "
    return (
        f"**{qid.upper()}. (stub — needs a question statement)** <!-- q:{qid} -->\n"
        f"Auto-created by `living_docs.py link`; replace this stub with the real "
        f"question prose.\n"
        f"{state}"
    )


# ─── link ──────────────────────────────────────────────────────────────────


def link(
    task_id: int,
    q_ids: list[str],
    *,
    paths: LivingDocsPaths | None = None,
) -> dict[str, Any]:
    """Link a task to one or more open questions (confirmed creation-time link).

    Writes a flat ``relates_to`` list onto the task's ``body.md`` YAML
    frontmatter (resolved via ``task_workflow``), and appends the task
    ref (``#N``) to each named question's State trailer evidence list in
    ``open_questions.md`` (matched by the ``<!-- q:<id> -->`` anchor). A
    question id with no anchor yet gets a minimal stub created at the end
    of the document.

    The whole operation is atomic: the body.md frontmatter write goes
    through ``task_workflow`` (its own flock + commit), and the
    ``open_questions.md`` edit goes through THIS module's flock + commit.
    Both share ``~/.task-workflow/lock`` in production, so they serialise.

    Parameters
    ----------
    task_id : int
        Task to link.
    q_ids : list[str]
        Open-question ids (case-insensitive; stored lower).
    paths : LivingDocsPaths, optional
        Injected paths (tests).

    Returns
    -------
    dict
        ``{"task_id", "relates_to", "stubbed": [ids created from stubs]}``.
    """
    paths = paths or LivingDocsPaths.from_repo()
    if not q_ids:
        raise ValueError("link requires at least one question id")
    norm_ids = [q.strip().lower() for q in q_ids]
    if any(not q for q in norm_ids):
        raise ValueError("question ids must be non-empty")

    # 1. Write relates_to onto the task's frontmatter (flat list, merged
    #    with any existing). Uses task_workflow's resolver + commit.
    task_path = tw.find_task_path(task_id)
    body_md = task_path / "body.md"
    stubbed: list[str] = []
    with _locked(paths.lock_path):
        fm, body = tw._read_body(body_md)
        existing = list(fm.get("relates_to") or [])
        merged = list(dict.fromkeys([*existing, *norm_ids]))  # order-preserving dedup
        fm["relates_to"] = merged
        tw._write_body(body_md, fm, body)
        tw._git_commit(
            [body_md],
            f"task #{task_id}: relates_to {merged}",
        )

        # 2. Append #N to each question's evidence list, stubbing missing ones.
        text = _read(paths.open_questions)
        for qid in norm_ids:
            if _find_anchor_line(text, qid) is None:
                text = _append_blocks(text, [_stub_question(qid)])
                stubbed.append(qid)
            text = _add_evidence_to_question(text, qid, task_id)
        _write_atomic(paths.open_questions, text)
        _git_commit(
            [paths.open_questions],
            f"living-docs: link #{task_id} → {merged}",
            repo_root=paths.repo_root,
        )
    return {"task_id": task_id, "relates_to": merged, "stubbed": stubbed}


def _add_evidence_to_question(text: str, q_id: str, task_id: int) -> str:
    """Append ``#task_id`` to the State trailer evidence of question ``q_id``.

    Fails loud if the anchor is present but the State trailer is missing
    or malformed — a question section must carry exactly one parseable
    State line for the updater to have a stable edit target. Idempotent:
    re-adding a task already in the evidence list is a no-op.
    """
    anchor_idx = _find_anchor_line(text, q_id)
    if anchor_idx is None:
        raise ValueError(f"question {q_id!r} has no anchor after stubbing — internal error")
    lines = text.splitlines()
    state_idx = _find_state_line_for_anchor(lines, anchor_idx)
    if state_idx is None:
        raise ValueError(
            f"question {q_id!r} (anchor line {anchor_idx + 1}) has no State trailer; "
            f"expected a `> **State:** ...` line in its section"
        )
    m = _STATE_RE.match(lines[state_idx])
    if not m:  # pragma: no cover — _find_state_line_for_anchor already matched
        raise ValueError(f"question {q_id!r} State line failed to re-parse: {lines[state_idx]!r}")
    ev_ids = _parse_evidence(m.group("evidence"))
    ev_ids.append(task_id)
    lines[state_idx] = _format_state_line(
        prefix=m.group("prefix"),
        maturity=m.group("maturity"),
        maturity_word=m.group("maturity_word"),
        confidence=m.group("confidence"),
        date=m.group("date"),
        evidence_ids=ev_ids,
    )
    return "\n".join(lines) + ("\n" if text.endswith("\n") else "")


# ─── backfill-reverse ────────────────────────────────────────────────────────


def backfill_reverse(
    *,
    paths: LivingDocsPaths | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Backfill task ``relates_to`` from the doc's question-evidence lists.

    Re-runnable reconciliation: inverts every anchored question's
    ``evidence: #N`` list into a per-task ``relates_to`` and writes the
    merged (order-preserving, deduped) list onto each task's ``body.md``
    frontmatter in a SINGLE commit. Idempotent — a task already carrying
    all of its question ids is left untouched. Evidence ids that resolve
    to no task are collected in ``missing`` and skipped (``check`` reports
    those separately); nothing is written for them.

    Parameters
    ----------
    paths : LivingDocsPaths, optional
        Injected paths (tests).
    dry_run : bool
        Compute and report changes without writing or committing.

    Returns
    -------
    dict
        ``{"changed": [(task_id, relates_to), ...], "unchanged": [ids],
        "missing": [ids], "dry_run": bool}``.
    """
    paths = paths or LivingDocsPaths.from_repo()
    questions = _collect_question_evidence(_read(paths.open_questions))

    # Invert doc evidence into task_id -> ordered-unique question ids.
    task_to_qs: dict[int, list[str]] = {}
    for qid, info in questions.items():
        for tid in info["evidence"]:
            bucket = task_to_qs.setdefault(tid, [])
            if qid not in bucket:
                bucket.append(qid)

    changed: list[tuple[int, list[str]]] = []
    unchanged: list[int] = []
    missing: list[int] = []
    changed_paths: list[Path] = []

    def _plan() -> None:
        for tid in sorted(task_to_qs):
            qids = sorted(task_to_qs[tid])
            try:
                body_md = tw.find_task_path(tid) / "body.md"
            except FileNotFoundError:
                missing.append(tid)
                continue
            fm, body = tw._read_body(body_md)
            existing = list(fm.get("relates_to") or [])
            merged = list(dict.fromkeys([*existing, *qids]))  # order-preserving dedup
            if merged == existing:
                unchanged.append(tid)
                continue
            if not dry_run:
                fm["relates_to"] = merged
                tw._write_body(body_md, fm, body)
                changed_paths.append(body_md)
            changed.append((tid, merged))

    if dry_run:
        _plan()
    else:
        with _locked(paths.lock_path):
            _plan()
            if changed_paths:
                _git_commit(
                    changed_paths,
                    f"living-docs: backfill relates_to from question evidence "
                    f"({len(changed_paths)} tasks)",
                    repo_root=paths.repo_root,
                )

    return {"changed": changed, "unchanged": unchanged, "missing": missing, "dry_run": dry_run}


def mark_unmapped(
    task_id: int,
    reason: str | None = None,
    *,
    paths: LivingDocsPaths | None = None,
) -> dict[str, Any]:
    """Mark a completed clean-result as intentionally unmapped.

    Sets ``living_docs_unmapped`` on the task's body.md frontmatter so
    :func:`check`'s coverage rule exempts it — a deliberate "this result
    has no open question" decision, not drift. The stored value is the
    reason string when given, else ``True``.

    Parameters
    ----------
    task_id : int
        Task to exempt.
    reason : str, optional
        Why it has no open question (recorded verbatim in the flag).
    paths : LivingDocsPaths, optional
        Injected paths (tests).
    """
    paths = paths or LivingDocsPaths.from_repo()
    body_md = tw.find_task_path(task_id) / "body.md"
    with _locked(paths.lock_path):
        fm, body = tw._read_body(body_md)
        fm["living_docs_unmapped"] = reason if reason else True
        tw._write_body(body_md, fm, body)
        tw._git_commit([body_md], f"task #{task_id}: mark living_docs_unmapped")
    return {"task_id": task_id, "living_docs_unmapped": fm["living_docs_unmapped"]}


# ─── check ─────────────────────────────────────────────────────────────────


@dataclass
class CheckReport:
    """Result of a :func:`check` run.

    ``problems`` is empty iff the docs are consistent. ``ok`` is the
    boolean the CLI maps to exit code 0 / 1.
    """

    problems: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """True when no drift was found."""
        return not self.problems

    def render(self) -> str:
        """Human-readable report."""
        if self.ok:
            return "living_docs check: PASS — no drift detected."
        head = f"living_docs check: FAIL — {len(self.problems)} problem(s):"
        body = "\n".join(f"  - {p}" for p in self.problems)
        return f"{head}\n{body}"


def _collect_question_evidence(text: str) -> dict[str, dict[str, Any]]:
    """Parse every anchored question's id → {evidence, date, line}.

    Walks the doc once, pairing each ``<!-- q:<id> -->`` anchor with the
    next State trailer in its section. A question with an anchor but no
    parseable State line is reported by :func:`check` (returned with an
    explicit marker), not silently skipped.
    """
    lines = text.splitlines()
    out: dict[str, dict[str, Any]] = {}
    for i, line in enumerate(lines):
        m = _ANCHOR_RE.search(line)
        if not m:
            continue
        qid = m.group(1).lower()
        state_idx = _find_state_line_for_anchor(lines, i)
        if state_idx is None:
            out[qid] = {"evidence": [], "date": None, "line": i + 1, "has_state": False}
            continue
        sm = _STATE_RE.match(lines[state_idx])
        out[qid] = {
            "evidence": _parse_evidence(sm.group("evidence")),
            "date": sm.group("date"),
            "line": i + 1,
            "has_state": True,
        }
    return out


def _completed_task_dates(paths: LivingDocsPaths) -> dict[int, str | None]:
    """Map every completed-with-clean-result task id → its promotion date.

    Used by check (b) [coverage] and (d) [staleness]. The date is the
    ``promoted_at`` frontmatter field's date portion when present, else
    ``created_at``, else None. Tasks flagged ``living_docs_unmapped`` are
    excluded — a deliberate "this result has no open question" decision,
    not drift (see :func:`mark_unmapped`).
    """
    out: dict[int, str | None] = {}
    for entry in tw.list_by_status("completed"):
        if not entry.get("has_clean_result"):
            continue
        tid = int(entry["id"])
        fm, _ = tw._read_body(tw.find_task_path(tid) / "body.md")
        if fm.get("living_docs_unmapped"):
            continue  # intentionally unmapped — exempt from coverage + staleness
        stamp = fm.get("promoted_at") or fm.get("created_at")
        date = str(stamp)[:10] if stamp else None
        out[tid] = date
    return out


def _all_task_ids(paths: LivingDocsPaths) -> set[int]:
    """Return the set of all task ids that exist (from the registry)."""
    reg = tw._load_registry()
    return {int(t) for t in reg.get("tasks", {})}


def _relates_to_index(paths: LivingDocsPaths) -> dict[int, list[str]]:
    """Map task id → its ``relates_to`` list across all statuses."""
    out: dict[int, list[str]] = {}
    reg = tw._load_registry()
    for tid_str in reg.get("tasks", {}):
        tid = int(tid_str)
        fm, _ = tw._read_body(tw.find_task_path(tid) / "body.md")
        rel = fm.get("relates_to")
        if rel:
            out[tid] = [str(q).lower() for q in rel]
    return out


def _check_structural(questions: dict[str, dict[str, Any]], report: CheckReport) -> None:
    """Flag anchored questions whose State trailer is missing / unparseable."""
    for qid, info in questions.items():
        if not info["has_state"]:
            report.problems.append(
                f"question '{qid}' (line {info['line']}) has an anchor but no "
                f"parseable `> **State:**` trailer"
            )


def _check_bidirectional(
    questions: dict[str, dict[str, Any]],
    q_evidence: dict[str, set[int]],
    relates: dict[int, list[str]],
    report: CheckReport,
) -> None:
    """Check (a): ``relates_to`` ⇄ question-evidence agree in both directions."""
    # Forward: relates_to → evidence.
    for tid, q_list in relates.items():
        for qid in q_list:
            if qid not in questions:
                report.problems.append(
                    f"task #{tid} relates_to '{qid}' but no question with that anchor exists"
                )
            elif tid not in q_evidence.get(qid, set()):
                report.problems.append(
                    f"task #{tid} relates_to '{qid}' but #{tid} is absent from "
                    f"that question's evidence list"
                )
    # Backward: evidence → relates_to.
    for qid, ev_ids in q_evidence.items():
        for tid in ev_ids:
            if tid not in relates or qid not in relates[tid]:
                report.problems.append(
                    f"question '{qid}' lists evidence #{tid} but task #{tid} does not "
                    f"have '{qid}' in its relates_to"
                )


def _check_coverage(
    q_evidence: dict[str, set[int]],
    completed: dict[int, str | None],
    report: CheckReport,
) -> None:
    """Check (b): every completed clean-result appears in some evidence list."""
    covered = {tid for ev in q_evidence.values() for tid in ev}
    for tid in sorted(completed):
        if tid not in covered:
            report.problems.append(
                f"completed task #{tid} (has_clean_result) appears in no question's evidence"
            )


def _check_resolvable(
    q_evidence: dict[str, set[int]],
    all_ids: set[int],
    report: CheckReport,
) -> None:
    """Check (c): every evidence ``#N`` resolves to a real task."""
    for qid, ev_ids in q_evidence.items():
        for tid in sorted(ev_ids):
            if tid not in all_ids:
                report.problems.append(
                    f"question '{qid}' lists evidence #{tid} but no such task exists"
                )


def _check_staleness(
    questions: dict[str, dict[str, Any]],
    completed: dict[int, str | None],
    report: CheckReport,
) -> None:
    """Check (d): flag a question whose State date predates a linked result."""
    for qid, info in questions.items():
        if not info["has_state"] or info["date"] is None:
            continue
        for tid in info["evidence"]:
            promo = completed.get(tid)
            if promo and promo > info["date"]:
                report.problems.append(
                    f"question '{qid}' State date {info['date']} is older than linked "
                    f"result #{tid} promoted {promo} — State trailer is stale"
                )


def check(*, paths: LivingDocsPaths | None = None) -> CheckReport:
    """Lint the living docs for drift. Returns a :class:`CheckReport`.

    Checks (each contributes problem lines):

    (a) **Bidirectional ``relates_to`` ⇄ evidence.** For every task with
        ``relates_to: [q...]``, that task's ``#N`` must appear in each
        named question's evidence; and for every ``#N`` in a question's
        evidence, that task's ``relates_to`` must name the question.
    (b) **Coverage.** Every ``completed`` task with
        ``has_clean_result=true`` must appear in some question's
        evidence.
    (c) **Resolvable evidence.** Every ``#N`` in any question's evidence
        must resolve to a real task.
    (d) **Staleness.** A question whose State ``updated`` date predates
        the promotion date of one of its linked completed results is
        flagged (the State line should have been bumped).

    Also flags anchored questions whose State trailer is missing /
    unparseable, since the updater needs a stable target.
    """
    paths = paths or LivingDocsPaths.from_repo()
    report = CheckReport()

    text = _read(paths.open_questions)
    questions = _collect_question_evidence(text)
    relates = _relates_to_index(paths)
    all_ids = _all_task_ids(paths)
    completed = _completed_task_dates(paths)

    # Question → evidence-id set (only for those with a parseable State line).
    q_evidence: dict[str, set[int]] = {
        qid: set(info["evidence"]) for qid, info in questions.items() if info["has_state"]
    }

    _check_structural(questions, report)
    _check_bidirectional(questions, q_evidence, relates, report)
    _check_coverage(q_evidence, completed, report)
    _check_resolvable(q_evidence, all_ids, report)
    _check_staleness(questions, completed, report)

    return report


# ─── CLI ───────────────────────────────────────────────────────────────────


def _cmd_apply(args: argparse.Namespace) -> int:
    """CLI: apply a confirmed patch (read as JSON from ``--patch-file``)."""
    patch_data = json.loads(Path(args.patch_file).read_text())
    patch = DocPatch.from_dict(patch_data)
    touched = apply(args.task_id, patch)
    print(f"applied patch for #{args.task_id}; touched: {[str(p) for p in touched]}")
    return 0


def _cmd_link(args: argparse.Namespace) -> int:
    """CLI: link a task to question ids."""
    result = link(args.task_id, args.q_ids)
    print(
        f"linked #{result['task_id']} → relates_to={result['relates_to']}"
        + (f"; stubbed new questions: {result['stubbed']}" if result["stubbed"] else "")
    )
    return 0


def _cmd_backfill_reverse(args: argparse.Namespace) -> int:
    """CLI: write task relates_to from question evidence lists (one commit)."""
    result = backfill_reverse(dry_run=args.dry_run)
    verb = "would update" if result["dry_run"] else "updated"
    print(
        f"backfill-reverse: {verb} {len(result['changed'])} task(s); "
        f"{len(result['unchanged'])} already current; "
        f"{len(result['missing'])} evidence id(s) resolved to no task"
        + (f" {sorted(result['missing'])}" if result["missing"] else "")
    )
    for tid, rel in result["changed"]:
        print(f"  #{tid} -> relates_to={rel}")
    return 0


def _cmd_mark_unmapped(args: argparse.Namespace) -> int:
    """CLI: exempt a completed result from coverage (intentional non-mapping)."""
    result = mark_unmapped(args.task_id, args.reason)
    print(f"marked #{result['task_id']} living_docs_unmapped={result['living_docs_unmapped']!r}")
    return 0


def _cmd_check(args: argparse.Namespace) -> int:
    """CLI: lint; exit nonzero on drift."""
    report = check()
    print(report.render())
    return 0 if report.ok else 1


def _build_parser() -> argparse.ArgumentParser:
    """Construct the ``living_docs.py`` argparse CLI."""
    parser = argparse.ArgumentParser(
        prog="living_docs.py",
        description="Mechanical core for docs/open_questions.md (apply / link / check).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_apply = sub.add_parser("apply", help="apply a confirmed patch (JSON file)")
    p_apply.add_argument("task_id", type=int, help="task the patch was produced for")
    p_apply.add_argument("--patch-file", required=True, help="path to the confirmed patch JSON")
    p_apply.set_defaults(func=_cmd_apply)

    p_link = sub.add_parser("link", help="link a task to open-question ids")
    p_link.add_argument("task_id", type=int, help="task to link")
    p_link.add_argument("q_ids", nargs="+", help="open-question ids (e.g. a1 d2)")
    p_link.set_defaults(func=_cmd_link)

    p_backfill = sub.add_parser(
        "backfill-reverse",
        help="write task relates_to from question evidence lists (one commit)",
    )
    p_backfill.add_argument(
        "--dry-run", action="store_true", help="preview without writing or committing"
    )
    p_backfill.set_defaults(func=_cmd_backfill_reverse)

    p_mark = sub.add_parser(
        "mark-unmapped",
        help="exempt a completed result from coverage (intentional non-mapping)",
    )
    p_mark.add_argument("task_id", type=int, help="task to exempt")
    p_mark.add_argument("--reason", default=None, help="why it has no open question")
    p_mark.set_defaults(func=_cmd_mark_unmapped)

    p_check = sub.add_parser("check", help="lint for drift; exit nonzero on drift")
    p_check.set_defaults(func=_cmd_check)

    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point for the ``living_docs.py`` CLI."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
