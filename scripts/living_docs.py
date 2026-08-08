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
  named question's evidence list in ``open_questions.md`` (matched by
  the ``<!-- q:<id> -->`` anchor). A question id with no anchor yet
  gets a minimal stub created.
- :func:`check` — lint for drift and exit nonzero when any is found:
  (a) ``relates_to`` ⇄ question-evidence agree both directions; (b)
  every ``completed`` task with ``has_clean_result`` appears in some
  question's evidence; (c) every evidence ``#N`` resolves to a real
  task; (d) flag questions whose State date is stale relative to a newer
  linked result (State-trailer carrier only — Belief carriers have no
  per-question date).

Schema. Two evidence carriers are supported per question; the live
``docs/open_questions.md`` uses the **Belief / Confidence / Evidence**
form, which is the **canonical live carrier** as of 2026-05-29 and the
shape every question in the live doc uses today. The older **State
trailer** form is still recognized for legacy stubs the script may have
emitted earlier, and is the format :func:`link`'s stub builder still
seeds when it auto-creates a new anchor. Both work for read AND write;
if both are present in one question section the State trailer wins
(narrowly: the auto-stubbed schema is the one :func:`link` knows how to
mutate idempotently). New live questions land in Belief form; the
``living-docs-updater`` agent emits Belief-form blocks when proposing
new sections.

**Belief / Confidence / Evidence** (canonical live carrier — every
question in ``docs/open_questions.md`` uses this; ``**Confidence:**``
and ``**Evidence:**`` may sit on the same blockquote line as
``**Belief:**`` or on a later one in the same blockquote)::

    **3.4a How do contrastive negatives shape leakage?** <!-- q:leak-contrastive-negatives -->
    ... prose ...
    > **Belief:** ... **Confidence:** LOW. **Evidence:** #207, #383, #391.
    > *Next: ...*

**State trailer** (legacy auto-stub form; what :func:`link` emits when
it stubs a missing anchor)::

    **A1. What predicts marker implantability?** <!-- q:a1 -->
    ... prose ...
    > **State:** 🌿 budding · MODERATE · updated 2026-05-28 · evidence: #207, #380

**Applications** (a render-only class — entries under the
``## Applications`` H2 carry a free-text ``**Status:**`` bullet rather
than a parseable carrier, and contribute no reverse-index edges)::

    - **App 1 — Assistant-anchored detector** ... **Status: falsification risk.**
      ... narrative including in-prose #N references ... <!-- q:app1 -->

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
import logging
import os
import random
import re
import subprocess
import sys
import time
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

_log = logging.getLogger("living_docs")

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

#: Leading prefix the AskUserQuestion prose and SKILL.md examples use when
#: naming question ids (e.g. ``q:beh-b-to-bprime`` or ``q-app5``). Stripped
#: at input boundaries via :func:`_normalize_qid` so the stored form is
#: always bare (``beh-b-to-bprime``), matching the long-standing
#: frontmatter convention.
_QID_PREFIX_RE = re.compile(r"^q[:\-]", re.IGNORECASE)

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

#: A Belief-format Evidence carrier line, e.g.
#:   > **Belief:** ... **Confidence:** LOW. **Evidence:** #207, #383.
#:   > **Confidence:** LOW. **Evidence:** #207, #383.
#:   > **Evidence:** none in-house yet.
#: Captures everything before the evidence value, the value itself, and
#: any trailing whitespace/period so we can rewrite the value in place.
#: ``head`` keeps the original blockquote prefix and any inline
#: ``**Belief:** ...`` / ``**Confidence:** ...`` prose preceding
#: ``**Evidence:**``; ``value`` is the evidence value (everything after
#: ``**Evidence:** `` up to a terminating period-and-end-of-line or
#: end-of-line); ``tail`` is the optional terminating ``.`` (kept so
#: trailing punctuation survives append).
_BELIEF_EVIDENCE_RE = re.compile(
    r"^(?P<head>>.*?\*\*Evidence:\*\*\s+)(?P<value>.*?)(?P<tail>\.?)\s*$"
)

#: A task reference inside an evidence list (e.g. ``#207``).
_EVIDENCE_REF_RE = re.compile(r"#(\d+)")

#: Belief-format Evidence values that semantically mean "no evidence
#: yet" and should be REPLACED by the first appended ``#N`` rather than
#: appended-to (matched after lowercasing + stripping trailing period).
#: Gating on this sentinel set (not on "no #N refs present in the
#: value") is intentional: a line like
#: ``**Evidence:** none in-house yet (definitional groundwork tracked in #428).``
#: carries #428 inside a parenthetical aside, so a bare "no refs"
#: check would incorrectly fire the replace path and silently drop
#: the aside. The sentinel set keeps the replace path narrow.
_EMPTY_BELIEF_VALUES = frozenset({"none in-house yet", "none yet", "tbd", "none"})

#: Anchor ids in this regex are **Applications** — entries under the
#: ``## Applications`` H2, which by design carry a free-text
#: ``**Status:**`` bullet rather than a parseable Belief/State trailer.
#: They are tracked as a render-only class:
#:
#: - :func:`_collect_question_evidence` returns ``carrier="app"`` for
#:   them and parses no evidence (their ``#N`` references live in prose
#:   and are never the canonical evidence carrier — keeping them out of
#:   the reverse index prevents the dashboard from synthesizing
#:   spurious question-result links from inline mentions).
#: - :func:`_check_structural` treats ``carrier="app"`` as compliant
#:   (no carrier required for apps), so the linter no longer flags
#:   ``app1``..``app6`` as structural drift.
#: - The bidirectional / coverage / resolvable / staleness checks
#:   simply receive an empty evidence set for app anchors, so they
#:   contribute no false positives there either.
#:
#: This is intentionally a separate, additive concern from the
#: Belief / State carrier regexes — those continue to govern every
#: NON-app anchor exactly as before. Recognizing the ``app`` /
#: ``app-<slug>`` shape is sufficient; we do not need to inspect the
#: H2 region the anchor falls under (the live doc reserves these slug
#: prefixes for the Applications section, and any future repurposing
#: would be a deliberate workflow change).
_APP_ANCHOR_RE = re.compile(r"^app(?:[-_].+|\d+)$")

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


def _normalize_qid(q_id: str) -> str:
    """Normalize a user-supplied question id to its canonical bare form.

    The CLI, the ``/issue`` Step 0c-link gate, and AskUserQuestion prose
    all let humans write the id with a leading ``q:`` or ``q-`` prefix
    (mirroring the anchor syntax — ``<!-- q:app5 -->`` → ``q:app5``).
    The stored form on disk (frontmatter ``relates_to``, anchor capture,
    backfill index) is always BARE (``app5``). Normalizing once at every
    input boundary keeps ``q:foo`` and ``foo`` producing byte-identical
    state — including idempotent re-links, dedup, and reverse-index
    membership checks.

    Strips at most one ``q:`` / ``q-`` prefix (case-insensitive), then
    strips surrounding whitespace and lowercases. Empty input raises.
    """
    norm = _QID_PREFIX_RE.sub("", q_id.strip(), count=1).strip().lower()
    if not norm:
        raise ValueError(f"question id must be non-empty (got {q_id!r})")
    return norm


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

    Lock-contention retry (#1917 parity; #1920): the command runs with
    ``check=False`` internally; if it exits non-zero AND stderr matches the
    canonical git lock-contention signature
    (``task_workflow._GIT_LOCK_CONTENTION_RE`` — a concurrent git process
    holds ``.git/index.lock`` or a sibling ``*.lock``), sleep a jittered
    ``random.uniform(*task_workflow._GIT_LOCK_RETRY_SLEEP_RANGE_S)``
    interval and rerun, until success OR the per-call wall budget is
    exhausted (``task_workflow._lock_wait_bound_s()`` —
    ``EPM_TASKPY_LOCK_WAIT_SECONDS``, default 60 s; ``0`` disables retries;
    the budget is read LAZILY at the first collision and the deadline is
    captured there, so any positive budget guarantees at least one retry).
    The retry keys on the STDERR SIGNATURE, never the return code, so
    ``check=False`` rc-as-signal call sites (``diff --cached --quiet``,
    ``push``) keep their rc semantics with zero retries, non-lock failures
    surface immediately with zero sleeps, and the happy path takes zero
    sleeps and zero extra env reads. After the loop the caller's ``check``
    semantics apply unchanged: ``check=True`` raises
    ``subprocess.CalledProcessError`` with the same
    ``cmd``/``output``/``stderr`` fields ``subprocess.run(check=True)``
    would produce.
    """
    env = dict(os.environ)
    for k in ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_OBJECT_DIRECTORY"):
        env.pop(k, None)

    def _attempt() -> subprocess.CompletedProcess[str]:
        # ``env`` is computed once per ``_git`` call and stable across
        # attempts (deliberate — do not re-sanitize per attempt).
        return subprocess.run(
            ["git", *args],
            cwd=str(cwd),
            env=env,
            check=False,
            capture_output=True,
            text=True,
        )

    result = _attempt()
    if result.returncode != 0 and tw._GIT_LOCK_CONTENTION_RE.search(result.stderr or ""):
        # Budget + deadline captured HERE, at the FIRST lock-signature
        # failure (lazy env read — never on the happy path), so any positive
        # budget guarantees >= 1 retry before the deadline check can trip.
        bound = tw._lock_wait_bound_s()
        deadline = time.monotonic() + bound
        first_collision = True
        while (
            result.returncode != 0
            and tw._GIT_LOCK_CONTENTION_RE.search(result.stderr or "")
            and time.monotonic() < deadline
        ):
            delay = random.uniform(*tw._GIT_LOCK_RETRY_SLEEP_RANGE_S)
            if first_collision:
                _log.warning(
                    "git %s hit a lock collision (a concurrent git process holds the "
                    "lock); retrying in %.1fs (bounded wait: up to %.0fs total, %s)",
                    args[0] if args else "",
                    delay,
                    bound,
                    tw._LOCK_WAIT_ENV,
                )
                first_collision = False
            time.sleep(delay)
            result = _attempt()
        if result.returncode != 0 and tw._GIT_LOCK_CONTENTION_RE.search(result.stderr or ""):
            _log.error(
                "git %s still failing on a lock collision after the %.0fs retry "
                "budget (%s; 0 disables). A concurrent git process is holding the "
                "repo lock; if no live git process exists, a crashed one may have "
                "left a stale .git/index.lock — inspect and remove it manually.",
                args[0] if args else "",
                bound,
                tw._LOCK_WAIT_ENV,
            )
    if check and result.returncode != 0:
        raise subprocess.CalledProcessError(
            result.returncode, result.args, output=result.stdout, stderr=result.stderr
        )
    return result


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


def _find_belief_evidence_line_for_anchor(lines: list[str], anchor_idx: int) -> int | None:
    """Given the anchor line index, return the index of the question's
    Belief-format Evidence carrier line, or None if absent.

    Searches forward from the anchor with the same section-boundary
    rules as :func:`_find_state_line_for_anchor` (next anchor or
    horizontal rule ends the section). Matches any blockquote line
    carrying ``**Evidence:** <value>`` — the Evidence segment can sit
    on the same blockquote line as ``**Belief:**`` (most common) or on a
    later blockquote line in the same section (e.g. when ``**Belief:**``
    spans its own paragraph and ``**Confidence:** ... **Evidence:** ...``
    sits below it).
    """
    for i in range(anchor_idx + 1, len(lines)):
        line = lines[i]
        if _BELIEF_EVIDENCE_RE.match(line):
            return i
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
    ref (``#N``) to each named question's evidence list in
    ``open_questions.md`` (matched by the ``<!-- q:<id> -->`` anchor).
    Accepts ids in either the bare form (``app5``) or the prefixed form
    (``q:app5`` / ``q-app5``); the bare form is canonical on disk and
    both inputs round-trip to byte-identical state.

    Question id semantics:

    - Missing anchor → a minimal stub (State-trailer carrier) is appended
      to the end of the doc, and ``#task_id`` lands in it.
    - Existing anchor with a State or Belief Evidence carrier → ``#task_id``
      is appended to that carrier idempotently.
    - Existing **Application** anchor (``<!-- q:app<n> -->`` or
      ``<!-- q:app-<slug> -->``, e.g. ``app5``) → the doc is left
      untouched (Applications carry a free-text ``**Status:**`` bullet,
      not a parseable carrier); only the task frontmatter ``relates_to``
      records the link. ``_check_bidirectional`` treats ``app`` anchors as
      carrier-exempt, so this is consistent — not silently dropped drift.

    Atomicity. The full doc edit is computed in memory FIRST: every
    requested id is resolved (to an existing carrier, to an Application
    anchor exemption, or to a fresh stub) before any file is written.
    If any id is unresolvable — e.g. an existing anchor whose section
    has neither a State trailer nor a Belief Evidence line — the whole
    operation raises and BOTH ``body.md`` and ``open_questions.md`` are
    left untouched. The body.md write and the doc write then land in a
    single mutation under one ``flock`` + one git commit covering both
    files (no partial-apply window between them).

    Parameters
    ----------
    task_id : int
        Task to link.
    q_ids : list[str]
        Open-question ids (case-insensitive; ``q:`` / ``q-`` prefix
        tolerated; stored lower and bare).
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
    norm_ids = [_normalize_qid(q) for q in q_ids]

    task_path = tw.find_task_path(task_id)
    body_md = task_path / "body.md"
    with _locked(paths.lock_path):
        # 1. Compute the frontmatter merge (in memory; no write yet).
        fm, body = tw._read_body(body_md)
        existing = list(fm.get("relates_to") or [])
        merged = list(dict.fromkeys([*existing, *norm_ids]))  # order-preserving dedup

        # 2. Compute the full doc edit (in memory; raises BEFORE any write
        #    if any id is unresolvable, so a mid-list failure leaves both
        #    body.md and open_questions.md untouched).
        original_doc = _read(paths.open_questions)
        new_doc, stubbed = _plan_doc_edit(original_doc, norm_ids, task_id)

        # 3. Commit both writes under the shared lock. Touch the doc only
        #    when it actually changed (an Applications-only link leaves the
        #    doc text byte-identical, but the body.md frontmatter still
        #    changes).
        fm["relates_to"] = merged
        tw._write_body(body_md, fm, body)
        commit_paths: list[Path] = [body_md]
        if new_doc != original_doc:
            _write_atomic(paths.open_questions, new_doc)
            commit_paths.append(paths.open_questions)
        _git_commit(
            commit_paths,
            f"living-docs: link #{task_id} → {merged}",
            repo_root=paths.repo_root,
        )
    return {"task_id": task_id, "relates_to": merged, "stubbed": stubbed}


def _plan_doc_edit(
    text: str,
    q_ids: list[str],
    task_id: int,
) -> tuple[str, list[str]]:
    """Compute the full ``open_questions.md`` edit for a :func:`link` call.

    Pure function — no I/O, no side effects. Walks each requested id in
    order, stubbing missing non-Application anchors and appending
    ``#task_id`` to each question's evidence carrier. Application anchors
    (``_APP_ANCHOR_RE``) that already exist are passed over without
    touching the doc (their relates_to-only link is recorded by the
    caller). Raises if any existing anchor's section has neither a State
    trailer nor a Belief Evidence line — atomicity guarantee: callers see
    either a fully-applied edit or an unchanged doc, never a partial
    write.

    Returns
    -------
    tuple[str, list[str]]
        ``(new_text, stubbed_ids)`` — the post-edit doc text and the
        subset of ``q_ids`` that were created as fresh stubs.
    """
    stubbed: list[str] = []
    for qid in q_ids:
        if _find_anchor_line(text, qid) is None:
            # Brand-new id: stub at end-of-doc, then evidence-link it.
            # (Stubs always land as State-trailer carriers regardless of
            # id shape — including App-shaped ids, which is unusual but
            # matches the long-standing stub contract; the updater agent
            # is the right path for adding a real Applications section.)
            text = _append_blocks(text, [_stub_question(qid)])
            stubbed.append(qid)
            text = _add_evidence_to_question(text, qid, task_id)
            continue
        if _APP_ANCHOR_RE.match(qid):
            # Existing Application anchor: relates_to-only link, doc
            # text unchanged. Carrier-exempt per _check_bidirectional.
            continue
        text = _add_evidence_to_question(text, qid, task_id)
    return text, stubbed


def _add_evidence_to_question(text: str, q_id: str, task_id: int) -> str:
    """Append ``#task_id`` to question ``q_id``'s evidence list.

    Locates whichever evidence carrier the question section actually
    uses — preferring a ``> **State:**`` trailer if present, else
    falling back to a ``> ... **Evidence:** ...`` Belief-format line —
    and appends ``#task_id`` to its evidence list. Fails loud if the
    anchor is present but the section carries NEITHER carrier (a
    question section must have a stable edit target for the updater).
    Idempotent: re-adding a task already in the evidence list is a
    no-op.
    """
    anchor_idx = _find_anchor_line(text, q_id)
    if anchor_idx is None:
        raise ValueError(f"question {q_id!r} has no anchor after stubbing — internal error")
    lines = text.splitlines()
    state_idx = _find_state_line_for_anchor(lines, anchor_idx)
    if state_idx is not None:
        lines[state_idx] = _append_to_state_line(lines[state_idx], task_id, q_id=q_id)
        return "\n".join(lines) + ("\n" if text.endswith("\n") else "")

    belief_idx = _find_belief_evidence_line_for_anchor(lines, anchor_idx)
    if belief_idx is not None:
        lines[belief_idx] = _append_to_belief_evidence_line(lines[belief_idx], task_id, q_id=q_id)
        return "\n".join(lines) + ("\n" if text.endswith("\n") else "")

    raise ValueError(
        f"question {q_id!r} (anchor line {anchor_idx + 1}) has no evidence carrier; "
        f"expected either a `> **State:** ...` trailer or a `> ... **Evidence:** ...` line "
        f"in its section"
    )


def _append_to_state_line(line: str, task_id: int, *, q_id: str) -> str:
    """Append ``#task_id`` to a State-trailer line; idempotent."""
    m = _STATE_RE.match(line)
    if not m:  # pragma: no cover — caller already matched _STATE_RE
        raise ValueError(f"question {q_id!r} State line failed to re-parse: {line!r}")
    ev_ids = _parse_evidence(m.group("evidence"))
    ev_ids.append(task_id)
    return _format_state_line(
        prefix=m.group("prefix"),
        maturity=m.group("maturity"),
        maturity_word=m.group("maturity_word"),
        confidence=m.group("confidence"),
        date=m.group("date"),
        evidence_ids=ev_ids,
    )


def _append_to_belief_evidence_line(line: str, task_id: int, *, q_id: str) -> str:
    """Append ``#task_id`` to a Belief-format Evidence line; idempotent.

    Insertion order: appended at the end of the existing evidence value,
    before any terminating period. Preserves the original ordering
    (Belief-format evidence lists are chronological-ish in the live doc,
    not sorted) and ANY parenthetical annotations carrying ``#N`` refs
    inside them.

    Empty-value REPLACE path: when the evidence value is a known
    "no evidence yet" sentinel (``none in-house yet``, ``none yet``,
    ``tbd``, ``none``), the value is REPLACED with ``#task_id`` rather
    than appended-to, so the doc reads cleanly rather than
    ``none in-house yet, #N``. Gating on a sentinel set (not on "no #N
    refs in the value") avoids silently overwriting prose like
    ``none in-house yet (definitional groundwork tracked in #428)`` or
    ``consult our internal tracker (no #refs)``, where the existing
    text is load-bearing despite carrying few/no bare #N refs.

    Idempotency uses the actual parsed #N set, so re-linking an id
    already present anywhere in the line (including inside a
    parenthetical) is a no-op.
    """
    m = _BELIEF_EVIDENCE_RE.match(line)
    if not m:  # pragma: no cover — caller already matched _BELIEF_EVIDENCE_RE
        raise ValueError(f"question {q_id!r} Belief Evidence line failed to re-parse: {line!r}")
    head, value, tail = m.group("head"), m.group("value"), m.group("tail")
    if task_id in _parse_evidence(value):
        return line  # idempotent no-op
    is_empty_sentinel = value.strip().rstrip(".").lower() in _EMPTY_BELIEF_VALUES
    new_value = f"#{task_id}" if is_empty_sentinel else f"{value}, #{task_id}"
    return f"{head}{new_value}{tail}"


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
    """Parse every anchored question's id → {evidence, date, line, has_state, carrier}.

    Walks the doc once, pairing each ``<!-- q:<id> -->`` anchor with
    EITHER a ``> **State:**`` trailer (preferred — the canonical schema)
    OR a Belief-format ``> ... **Evidence:** ...`` line within the same
    section. A question with an anchor but neither carrier is reported by
    :func:`check` (returned with ``carrier="none"``), not silently
    skipped.

    Application anchors (ids matching :data:`_APP_ANCHOR_RE`, e.g.
    ``app1``..``app6``) are a separate class: by design they carry a
    free-text ``**Status:**`` bullet rather than a parseable carrier, so
    they are returned with ``carrier="app"`` and empty evidence. The
    structural / coverage / staleness checks treat ``"app"`` as
    compliant (apps have no canonical evidence list and contribute no
    reverse-index edges).

    Returns
    -------
    dict[str, dict]
        Per-question dict with keys: ``evidence`` (list of int task
        ids), ``date`` (YYYY-MM-DD or None — only the State carrier
        records one, so Belief-format questions always get None),
        ``line`` (1-based anchor line), ``has_state`` (True iff the
        State trailer carrier was used; False for Belief / app /
        ``carrier="none"``), and ``carrier`` (``"state"`` |
        ``"belief"`` | ``"app"`` | ``"none"``).
    """
    lines = text.splitlines()
    out: dict[str, dict[str, Any]] = {}
    for i, line in enumerate(lines):
        m = _ANCHOR_RE.search(line)
        if not m:
            continue
        qid = m.group(1).lower()
        # Application anchors are render-only; they have a free-text
        # ``**Status:**`` bullet, never a Belief/State carrier. Skip
        # the carrier search entirely and tag them so check() can
        # exempt them.
        if _APP_ANCHOR_RE.match(qid):
            out[qid] = {
                "evidence": [],
                "date": None,
                "line": i + 1,
                "has_state": False,
                "carrier": "app",
            }
            continue
        state_idx = _find_state_line_for_anchor(lines, i)
        if state_idx is not None:
            sm = _STATE_RE.match(lines[state_idx])
            out[qid] = {
                "evidence": _parse_evidence(sm.group("evidence")),
                "date": sm.group("date"),
                "line": i + 1,
                "has_state": True,
                "carrier": "state",
            }
            continue
        belief_idx = _find_belief_evidence_line_for_anchor(lines, i)
        if belief_idx is not None:
            bm = _BELIEF_EVIDENCE_RE.match(lines[belief_idx])
            out[qid] = {
                "evidence": _parse_evidence(bm.group("value")),
                "date": None,  # Belief carrier has no per-question date
                "line": i + 1,
                "has_state": False,
                "carrier": "belief",
            }
            continue
        out[qid] = {
            "evidence": [],
            "date": None,
            "line": i + 1,
            "has_state": False,
            "carrier": "none",
        }
    return out


def _completed_task_dates(
    paths: LivingDocsPaths,
) -> tuple[dict[int, str | None], list[tuple[int, str]]]:
    """Map every completed-with-clean-result task id → its promotion date.

    Used by check (b) [coverage] and (d) [staleness]. The date is the
    ``promoted_at`` frontmatter field's date portion when present, else
    ``created_at``, else None. Tasks flagged ``living_docs_unmapped`` are
    excluded — a deliberate "this result has no open question" decision,
    not drift (see :func:`mark_unmapped`).

    Returns ``(dates, drifted)``: the second element lists
    ``(task_id, error_message)`` pairs for tasks whose registry entry is
    inconsistent with the on-disk tree — the directory is missing (raising
    from :func:`tw.find_task_path`) OR ``body.md`` is absent inside it
    (raising from :func:`tw._read_body`'s ``read_text``). These surface as
    drift findings in :func:`check` rather than crash the whole pass — see
    also :func:`_relates_to_index`.
    """
    out: dict[int, str | None] = {}
    drifted: list[tuple[int, str]] = []
    for entry in tw.list_by_status("completed"):
        if not entry.get("has_clean_result"):
            continue
        tid = int(entry["id"])
        try:
            body_path = tw.find_task_path(tid) / "body.md"
            fm, _ = tw._read_body(body_path)
        except (FileNotFoundError, OSError) as e:
            drifted.append((tid, str(e)))
            continue
        if fm.get("living_docs_unmapped"):
            continue  # intentionally unmapped — exempt from coverage + staleness
        stamp = fm.get("promoted_at") or fm.get("created_at")
        date = str(stamp)[:10] if stamp else None
        out[tid] = date
    return out, drifted


def _all_task_ids(paths: LivingDocsPaths) -> set[int]:
    """Return the set of all task ids that exist (from the registry)."""
    reg = tw._load_registry()
    return {int(t) for t in reg.get("tasks", {})}


def _relates_to_index(
    paths: LivingDocsPaths,
) -> tuple[dict[int, list[str]], list[tuple[int, str]]]:
    """Map task id → its ``relates_to`` list across all statuses.

    Returns ``(index, drifted)``: the second element lists
    ``(task_id, error_message)`` pairs for tasks whose registry entry is
    inconsistent with the on-disk tree (directory missing OR ``body.md``
    missing). See :func:`_completed_task_dates` for the failure modes and
    why they degrade to drift findings rather than crash :func:`check`.
    """
    out: dict[int, list[str]] = {}
    drifted: list[tuple[int, str]] = []
    reg = tw._load_registry()
    for tid_str in reg.get("tasks", {}):
        tid = int(tid_str)
        try:
            body_path = tw.find_task_path(tid) / "body.md"
            fm, _ = tw._read_body(body_path)
        except (FileNotFoundError, OSError) as e:
            drifted.append((tid, str(e)))
            continue
        rel = fm.get("relates_to")
        if rel:
            out[tid] = [str(q).lower() for q in rel]
    return out, drifted


def _check_structural(questions: dict[str, dict[str, Any]], report: CheckReport) -> None:
    """Flag anchored questions whose evidence carrier is missing.

    An anchored question must carry EITHER a ``> **State:**`` trailer
    (canonical schema) OR a ``> ... **Evidence:** ...`` Belief-format
    line. If neither is present in the section, the updater has no
    stable edit target — that is structural drift.

    Application anchors (``carrier="app"``) are exempt: by design they
    carry a free-text ``**Status:**`` bullet rather than a parseable
    carrier, so no edit target is needed (any update routes through
    bespoke prose edits via :func:`apply`'s replacement set).
    """
    for qid, info in questions.items():
        if info["carrier"] == "none":
            report.problems.append(
                f"question '{qid}' (line {info['line']}) has an anchor but no "
                f"parseable `> **State:** ...` trailer or `> ... **Evidence:** ...` line"
            )


def _check_bidirectional(
    questions: dict[str, dict[str, Any]],
    q_evidence: dict[str, set[int]],
    relates: dict[int, list[str]],
    report: CheckReport,
) -> None:
    """Check (a): ``relates_to`` ⇄ question-evidence agree in both directions.

    Application anchors are exempt — they carry no parseable evidence
    list, so a task whose ``relates_to`` names ``app1`` has nothing to
    cross-check against. (The reverse-index for apps is the dashboard's
    in-prose ``#N`` rendering, not the carrier-derived list this linter
    governs.)
    """
    # Forward: relates_to → evidence.
    for tid, q_list in relates.items():
        for qid in q_list:
            if qid not in questions:
                report.problems.append(
                    f"task #{tid} relates_to '{qid}' but no question with that anchor exists"
                )
            elif questions[qid]["carrier"] == "app":
                # Apps carry no evidence list to cross-check against;
                # the relates_to link is acknowledged as-is.
                continue
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
        Application anchors (``carrier="app"``) are exempt — they carry
        no evidence list to cross-check.
    (b) **Coverage.** Every ``completed`` task with
        ``has_clean_result=true`` must appear in some question's
        evidence (apps contribute no evidence and so do not satisfy
        coverage; the carrying question must be a non-app one).
    (c) **Resolvable evidence.** Every ``#N`` in any question's evidence
        must resolve to a real task.
    (d) **Staleness.** A question whose State ``updated`` date predates
        the promotion date of one of its linked completed results is
        flagged (the State line should have been bumped). Only runs
        against the State-trailer carrier — the Belief-format Evidence
        carrier has no per-question date, so staleness is not
        computable there, and apps have no carrier at all.

    Also flags anchored questions whose evidence carrier is entirely
    missing (no State trailer AND no Belief-format Evidence line) —
    again, apps are exempt because their canonical surface is the
    free-text ``**Status:**`` bullet, not a Belief/State trailer.
    """
    paths = paths or LivingDocsPaths.from_repo()
    report = CheckReport()

    text = _read(paths.open_questions)
    questions = _collect_question_evidence(text)
    relates, relates_drifted = _relates_to_index(paths)
    all_ids = _all_task_ids(paths)
    completed, completed_drifted = _completed_task_dates(paths)

    # Surface any registry/filesystem inconsistency as a drift finding,
    # dedup'd across the two indexers (the same drifted task can show up in
    # both passes). One line per task, naming the id + the underlying raise,
    # so the lint degrades gracefully (the surviving check axes below still
    # run against the registry-trimmed indexes) instead of hard-crashing on
    # the first bad task.
    seen_drifted: set[int] = set()
    for tid, err in [*relates_drifted, *completed_drifted]:
        if tid in seen_drifted:
            continue
        seen_drifted.add(tid)
        report.problems.append(
            f"task #{tid} registry/filesystem inconsistent: {err} (run `task.py audit` to repair)"
        )

    # Question → evidence-id set. Only questions whose section has a
    # parseable carrier — either State trailer or Belief Evidence line —
    # contribute. Apps are render-only (carrier="app", no evidence list)
    # and ``"none"`` is a structural-drift flag; both are excluded.
    q_evidence: dict[str, set[int]] = {
        qid: set(info["evidence"])
        for qid, info in questions.items()
        if info["carrier"] in ("state", "belief")
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
