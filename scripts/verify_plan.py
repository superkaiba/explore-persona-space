#!/usr/bin/env python
"""verify_plan.py — mechanical pre-pass gate for experiment plans (task #625).

Deterministic, sub-second structural verifier for the plans persisted at
``tasks/<status>/<N>/plans/v{K}.md``, run at ``/adversarial-planner``
Phase 1.5.0 BEFORE the fact-checker + critic ensemble spawn. The plan-side
sibling of ``scripts/verify_task_body.py`` (clean-result bodies): pure
regex / string presence checks, NO LLM calls, no network, no side effects
(the orchestrator running the adversarial-planner skill posts the
``epm:plan-verify`` marker — never this script). One disclosed read-only
exception: check 34, when its trigger fires, ``stat()``s the live sizes of
the ratcheted workflow files the plan names and lazily imports
``scripts/workflow_lint.py`` for their size-cap constants — read-only, no
writes, still no network.

Check catalog (id — classification — kind scope)
------------------------------------------------

  c0  plan-nonstub               FAIL, short-circuits      all kinds
  c1  §11 Source: grounding      FAIL (WARN degradation)   experiment only
  c2  measurement validity       FAIL when ALL signals     experiment only
                                 absent
  c3  data-source tier           WARN-only                 experiment only
  c4  contrastive negatives      WARN-only, conditional    experiment only
  c5  GPU-hour estimate          FAIL for ALL kinds        all kinds
  c6  reused-artifact fitness    WARN-only, conditional    experiment only
  c7  replication fidelity       WARN-only, conditional    experiment only
  c8  success + kill criteria    FAIL both-absent          experiment FAILs,
                                                           exempt kinds WARN;
                                                           exempt kinds accept a
                                                           solid §0.0 TL;DR
                                                           change-my-mind line as
                                                           kill (#1291)
  c9  conditions/cells + seeds   WARN-only                 experiment only
  c10 marker-recipe ack          WARN-only, conditional    experiment only
  c11 dry-run test coverage      WARN-only, conditional    infra + batch only
  c12 battery multiplier +       FAIL (experiment) / WARN  experiment +
      batched commitment         (analysis), conditional   analysis
  c13 empirical-null gate        FAIL (experiment) / WARN  experiment +
      p-floor attainability      (analysis), conditional   analysis
  c14 hypothesis branch         WARN-only, conditional    experiment +
      coherence                                           analysis
  c15 fail-loud acceptance      WARN-only, conditional    infra + batch only
      claim backed by test
  c16 re-extracted reference    WARN-only, conditional    experiment +
      vs committed headline                               analysis
  c17 falsification-branch      WARN-only, conditional    experiment +
      causal-claim scope                                  analysis
  c18 paired-contrast per-arm   FAIL (experiment) / WARN  experiment +
      source coverage           (analysis), conditional   analysis
  c19 OOD generalization folds  WARN-only, conditional    experiment +
                                                          analysis
  c20 verdict-lattice           FAIL (experiment) / WARN  experiment +
      coherence                 (analysis), conditional   analysis
  c21 grep-arity acceptance     WARN-only, conditional    all kinds
      gate → AST arity audit
  c22 cross-section param       WARN-only, conditional    all kinds
      consistency
  c23 goal currency             WARN-only, conditional    all kinds,
      (stale-Goal quote)                                  --issue mode only
  c24 resume-skip provenance    WARN-only, conditional    experiment +
      validation                                          analysis
  c25 html entities in fenced   FAIL, conditional         all kinds
      command blocks
  c26 GPU basis vs routed       WARN-only, conditional    experiment +
      machine                                             analysis
  c27 7B activation-capture     FAIL (experiment) / WARN  experiment +
      vs eval/debug intent      (analysis), conditional   analysis
  c28 decision-band precedent   WARN-only, conditional    experiment +
      coherence                                           analysis
  c29 deliberate fence vs §7    WARN-only, conditional    experiment +
      conditional phase                                   analysis
  c30 reused-bundle realized    WARN-only, conditional    experiment +
      keys                                                analysis
  c31 SKILL.md prose            WARN-only, conditional    infra + batch only
      durability pin
  c32 fit-family §9 basis       WARN-only, conditional    experiment +
      grounding                                           analysis
  c33 ladder checkpoint         WARN-only, conditional    experiment +
      retention policy                                    analysis
  c34 verbatim insert vs        WARN-only, conditional    infra + batch only
      ratchet headroom
  c35 revision-pinned reuse     WARN-only, conditional    experiment +
      verified at pin                                     analysis

Kind-exempt checks render as [SKIP] (first-class status, distinguishable
from genuine passes — the calibration report needs n_skip separate from
n_pass). Conditional checks (4, 6, 7, 10, 11, 12, 13, 14, 15, 16, 17, 18,
19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35) also
SKIP when their content trigger does not fire.
Check 23 runs OUTSIDE ``verify_plan_text()`` — it needs task context
(``body.md`` + ``events.jsonl``), so ``main()`` appends it in ``--issue``
mode only and renders it SKIP in ``--plan-file`` mode; its WARN is the one
the adversarial-planner Phase 1.5.0 consumer treats as a mechanical redraft
bounce (SKILL.md § Goal-currency gate), not a brief-forwarded WARN.

Canonical N/A escape phrases (quote verbatim in bounce briefs; each
satisfies its check ONLY as a standalone declaration line — see
``_standalone_na_declared``; exception: check 31 uses its
labeled-line forms):

  - ``N/A — no model training`` / ``N/A — no training hyperparameters``
    (check 1)
  - ``N/A — no behavioral construct`` (check 2)
  - ``N/A — not a behavior-implantation`` (check 4)
  - ``N/A — no artifact reuse`` (check 6)
  - ``N/A — not a replication`` (check 7)
  - ``N/A — no dry-run smoke`` (check 11)
  - ``N/A — no draw battery`` (check 12)
  - ``N/A — no empirical-null gate`` (check 13)
  - ``N/A — no fail-loud acceptance claim`` /
    ``N/A — fail-loud claim not test-backable`` (check 15)
  - ``N/A — no re-extracted reference arms`` (check 16)
  - ``N/A — no paired contrast`` (check 18)
  - ``N/A — no held-out predictive DV`` (check 19)
  - ``N/A — no registered verdict lattice`` (check 20)
  - ``N/A — no arity acceptance gate`` (check 21)
  - ``N/A — no resume/persist pattern`` (check 24)
  - ``N/A — entities are content, not commands`` (check 25 — exempts
    arm-(a) shell-tagged content fences ONLY, and only when exactly ONE
    arm-(a) fence carries entity hits (#1276); an arm-(b) fence whose body
    carries ``--workload-cmd`` / ``dispatch_issue.py`` FAILs on entities
    unconditionally)
  - ``N/A — basis measured on the routed machine`` (check 26)
  - ``N/A — no 7B activation capture`` (check 27)
  - ``N/A — no precedent-labeled decision bands`` (check 28; British
    ``labelled`` accepted)
  - ``N/A — no conditional phase on this provision`` (check 29)
  - ``N/A — no multi-field bundle reuse`` (check 30)
  - ``Durability pin: N/A — <one-line reason>`` / alias
    ``N/A — no durability pin: <reason>`` (check 31; the reason tail is
    mandatory — a bare ``Durability pin: N/A`` still WARNs)
  - ``N/A — no fit-family phases`` (check 32)
  - ``N/A — no per-rung checkpoint persistence`` / alias
    ``N/A — no checkpoint ladder`` (check 33)
  - ``N/A — no verbatim ratcheted-file insertion`` (check 34)
  - ``N/A — no revision-pinned reuse`` (check 35)

WARN semantics: a WARN never blocks exit (exit 0). The Phase 1.5.0 wiring
carries WARN lines verbatim into the fact-checker + critic briefs — that
forwarding IS the ships-if-acknowledged mechanism for plans (unlike
clean-result bodies, plans have a downstream human-grade review that
weighs every WARN).

Scope discipline: every check here guarantees only that the contract
SURFACE exists (a Source: label has a non-empty evidence-shaped value, a
measurement-validity block has construct/metric content, ...). The
semantic questions — is each Source *correct*, does it *transfer*, is the
proxy *valid* — stay with the Phase 1.5 fact-checker and the Phase 2
critic ensemble. A PASS here is never "grounding verified".

Usage::

    uv run python scripts/verify_plan.py --issue 614 [--json]
    uv run python scripts/verify_plan.py --plan-file path/to/plan.md \
        [--kind experiment] [--json]

``--issue`` resolves the task folder via
``explore_persona_space.task_workflow.find_task_path`` (never hand-built
``tasks/`` paths) and verifies the newest ``plans/v{K}.md`` by NUMERIC
sort (``v10`` > ``v9``; never the ``plan.md`` symlink — follow-up rounds
re-point it, the verify_task_body check-16 / incident #597 trap), reading
``kind`` from ``body.md`` frontmatter. ``--plan-file`` verifies a
standalone file (e.g. a not-yet-persisted ``/tmp`` handoff draft);
``--kind`` applies in file mode only and defaults to ``experiment`` (the
strictest, matching the issue-mode missing-kind fallback).

Exit codes: 0 = PASS (WARNs allowed), 1 = at least one FAIL,
2 = resolution / IO error.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from fractions import Fraction
from pathlib import Path

import yaml

# ─── Constants ─────────────────────────────────────────────────────────────

# Plan-relevant kinds for the CLI `--kind` choices (file mode). Kept an
# explicit ordered tuple because argparse `choices=` uses this order for help
# text + error messages. Membership is `("experiment", *EXEMPT_KINDS)`; the
# canonical single source for the exempt subset is
# `task_workflow.CODE_KINDS`, and `tests/test_verify_plan.py` pins both this
# tuple and EXEMPT_KINDS to it so the three `kind`-enum copies can never drift
# (incident #672). Local-import discipline (this module avoids a module-level
# `task_workflow` dependency) keeps the literal here; the test is the gate.
VALID_KINDS = ("experiment", "analysis", "infra", "batch", "survey")

# Kinds exempt from the experiment-only checks (CLAUDE.md Critical Rules:
# "`kind: analysis|infra|batch|survey` exempt"). Byte-identical to
# `task_workflow.CODE_KINDS`; pinned by the drift test (see above).
EXEMPT_KINDS = frozenset({"analysis", "infra", "batch", "survey"})

# Check 0 thresholds: a real plan (even a terse infra/analysis one — #575's
# v1 is the short end of the observed corpus) clears these comfortably; a
# truncated / contaminated handoff (#562 harness-trailer class) does not.
MIN_PLAN_CHARS = 1500
MIN_PLAN_HEADINGS = 3

# Check 8 "non-contradictory in form" emptiness bar: the innermost section
# carrying a success/kill anchor must have at least this much body text.
MIN_CRITERIA_CARRIER_CHARS = 80

# Tolerant N/A prefix: em dash, en dash, colon, opening paren, or hyphen
# after the N/A token ("N/A — ...", "N/A: ...", "N/A (not a replication)").
NA_RE = r"(?i)\bN/?A\b\s*[—–:(-]\s*"  # noqa: RUF001 — en dash is real plan text

# Check 1: inline `Source:` label. Value capture stops at newline or table
# pipe so a label inside a table cell captures only its own cell.
_SOURCE_LABEL_RE = re.compile(r"(?i)\bSource:\s*([^\n|]*)")

# Tokens that make a Source value "prose about sources" rather than
# evidence (planner.md's own boilerplate: "One `Source:` per unique value").
_SOURCE_VALUE_STOPWORDS = frozenset({"per", "unique", "value", "each", "every"})

# Check 5: the one exact, pre-existing string contract (planner.md §0).
# `\**` admits the bold form (`**Estimated GPU-hours (total):** 4`);
# optional backticks admit the inline-code form. A single plain number —
# ranges and `~`-qualified values fail.
GPU_LINE_RE = re.compile(r"(?i)estimated\s+gpu-?hours\s+\(total\):\**\s*`?([0-9]+(?:\.[0-9]+)?)`?")
GPU_LABEL_RE = re.compile(r"(?i)estimated\s+gpu-?hours\s+\(total\)")

# Check 5: backtick-tolerant numeric-range detector, applied with .match()
# anchored at the captured value BEFORE the annotation stops run. One of
# the stops is the closing backtick, so a stop-first scan truncates
# "`4`-8" to "4" and false-PASSes the range as its first number (round-2
# reconciler blocker gpu-hours-backtick-range-false-pass; "`40`-200" is
# the auto-approve-cap understatement shape). The leading "`?" is
# redundant after GPU_LINE_RE consumed the value's opening backtick, but
# kept to match the endorsed detector shape.
GPU_RANGE_AT_VALUE_RE = re.compile(
    r"`?[0-9]+(?:\.[0-9]+)?`?\s*[-–]\s*`?[0-9]"  # noqa: RUF001 — en-dash ranges are real
)

# Checks 4 + 10: marker-leakage vocabulary (NOT the bare token "marker",
# which false-fires on workflow vocabulary — `post-marker`, `epm:` markers —
# present in nearly every plan).
_MARKER_VOCAB_RE = re.compile(
    r"※|83399|marker[- ]leakage|log ?p\(marker\)|markeronlydatacollator",
    re.IGNORECASE,
)

# Check 8 vocabulary families.
_SUCCESS_RE = re.compile(r"(?i)success criteri|acceptance criteri|decision rule|decision gate")
_KILL_RE = re.compile(
    r"(?i)kill[- ]criteri|abort criteri|stop criteri|halt-and-report|what would change my mind"
)

# Check 11: trigger = the CLI flag form anywhere in the RAW plan (smoke
# commands legitimately live inside fences/tables). Evidence = a line naming
# a dry-run-exercising test: a `test_` identifier co-occurring with a dry-run
# token (no \b before "dry" — the token legitimately sits embedded in
# identifiers like test_drain_dry_run_no_dispatch), or the word "test"
# co-occurring with the Python kwarg form `dry_run`. `--dry-run` flag
# occurrences are STRIPPED from the line before the tier-1 scan: the bare
# flag next to test vocabulary deliberately does NOT self-certify — neither
# the "run the smoke, then the test suite" sentence shape nor the #633 v1
# false-PASS shape (ONE `Verification commands:` line carrying both the
# success-path pytest invocation and the `--dry-run` smoke command).
_DRYRUN_FLAG_RE = re.compile(r"--dry-run\b")
_DRYRUN_ANY_RE = re.compile(r"(?i)dry[-_ ]?run")
_DRYRUN_KWARG_RE = re.compile(r"(?i)dry_run")
_TEST_IDENT_RE = re.compile(r"\btest_\w+")
_TEST_WORD_RE = re.compile(r"(?i)\btests?\b")

# Check 3: data-source tier vocabulary (CLAUDE.md realistic-data rule).
_TIER_RE = re.compile(
    r"(?i)tier[-\s]*[1-4]|real-world data|established (?:dataset|benchmark)"
    r"|diverse llm[- ]generated|programmatic(?:ally)? generated|realistic-data preference"
)
_TIER_34_RE = re.compile(
    r"(?i)tier[-\s]*[34]|diverse llm[- ]generated|programmatic(?:ally)? generated"
)

# ─── Result type ───────────────────────────────────────────────────────────


@dataclass
class CheckResult:
    """One check verdict.

    ``skipped`` (kind-exempt or conditional trigger not fired) and
    ``is_warn`` both leave ``passed=True`` — only a hard FAIL flips it.
    """

    id: str
    name: str
    passed: bool
    detail: str = ""
    is_warn: bool = False
    skipped: bool = False

    @property
    def status(self) -> str:
        if self.skipped:
            return "SKIP"
        if not self.passed:
            return "FAIL"
        if self.is_warn:
            return "WARN"
        return "PASS"

    def render(self) -> str:
        line = f"  [{self.status}] {self.name}"
        if self.detail:
            line += f" — {self.detail}"
        return line


def _pass(cid: str, name: str, detail: str = "") -> CheckResult:
    return CheckResult(cid, name, True, detail)


def _warn(cid: str, name: str, detail: str) -> CheckResult:
    return CheckResult(cid, name, True, detail, is_warn=True)


def _fail(cid: str, name: str, detail: str) -> CheckResult:
    return CheckResult(cid, name, False, detail)


def _skip(cid: str, name: str, detail: str) -> CheckResult:
    return CheckResult(cid, name, True, detail, skipped=True)


# ─── Parsing helpers ───────────────────────────────────────────────────────


def split_frontmatter(text: str) -> tuple[dict, str]:
    """Split a leading ``---`` YAML frontmatter block off ``text``.

    Returns ``({}, text)`` unchanged when there is no parseable block.
    Used for ``body.md`` (kind lookup) — plan files are passed through raw.
    """
    if not text.startswith("---\n"):
        return {}, text
    rest = text[4:]
    end = rest.find("\n---\n")
    if end == -1:
        return {}, text
    fm_block = rest[:end]
    body = rest[end + len("\n---\n") :]
    try:
        fm = yaml.safe_load(fm_block) or {}
    except yaml.YAMLError:
        return {}, text
    if not isinstance(fm, dict):
        return {}, text
    return fm, body


def _fence_mask(lines: list[str]) -> list[bool]:
    """Per-line mask: True when the line is a fence delimiter or inside a
    fenced code block. Both ``` and ~~~ toggle, matching CommonMark's
    relaxed rule (same behavior as verify_task_body.find_h2_sections)."""
    mask: list[bool] = []
    in_fence = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("```") or stripped.startswith("~~~"):
            in_fence = not in_fence
            mask.append(True)
            continue
        mask.append(in_fence)
    return mask


def strip_fences(text: str) -> str:
    """Return ``text`` with fenced code blocks (and the fence delimiter
    lines) removed, so example commands inside fences can neither satisfy
    nor trip a prose-contract check."""
    lines = text.splitlines()
    mask = _fence_mask(lines)
    return "\n".join(line for line, fenced in zip(lines, mask, strict=True) if not fenced)


@dataclass
class Heading:
    level: int
    text: str
    line: int  # heading line index
    end: int  # exclusive end line of the section (next same-or-higher heading)


_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")

# HTML heading tag: matches <h1>…<h6> with optional attributes (e.g.
# <h2 style="margin-top:0">). Used by check_plan_nonstub to accept the HTML
# output format documented in CLAUDE.md § Output format (adversarial-planner
# defaults to HTML for browser-reading).
_HTML_HEADING_RE = re.compile(r"<h[1-6]\b[^>]*>", re.IGNORECASE)


def _headings(text: str) -> list[Heading]:
    """Fence-aware heading parser for H1-H6 (plans put required blocks at
    H2 AND H3; H4 shows up in pipelines). Each heading's section extends to
    the next heading of the same or higher level."""
    lines = text.splitlines()
    mask = _fence_mask(lines)
    found: list[tuple[int, str, int]] = []
    for i, line in enumerate(lines):
        if mask[i]:
            continue
        m = _HEADING_RE.match(line.strip())
        if m:
            found.append((len(m.group(1)), m.group(2).strip(), i))
    out: list[Heading] = []
    for k, (level, htext, start) in enumerate(found):
        end = len(lines)
        for level2, _, start2 in found[k + 1 :]:
            if level2 <= level:
                end = start2
                break
        out.append(Heading(level, htext, start, end))
    return out


def section_text_by_keywords(text: str, keywords: tuple[str, ...]) -> str | None:
    """Keyword-fuzzy section locator: first heading (document order) whose
    text contains any keyword (case-insensitive substring) wins; returns
    heading line + section body. None when no heading matches. Never exact
    heading matching — the observed corpus drifts (`## 7. Decision Gates,
    Success and Kill Criteria` vs `## 7. Decision gates` vs `## 10.
    Hyperparameter grounding (§11)`)."""
    lines = text.splitlines()
    lowered = tuple(k.casefold() for k in keywords)
    for h in _headings(text):
        htext = h.text.casefold()
        if any(k in htext for k in lowered):
            return "\n".join(lines[h.line : h.end])
    return None


def _innermost_section(headings: list[Heading], line_idx: int) -> Heading | None:
    """Deepest (then latest-starting) heading whose section contains
    ``line_idx``; None when the line precedes every heading."""
    best: Heading | None = None
    for h in headings:
        if h.line <= line_idx < h.end and (
            best is None or h.level > best.level or h.line > best.line
        ):
            best = h
    return best


# ─── Check 0 — plan-nonstub (FAIL, short-circuits; all kinds) ──────────────


def check_plan_nonstub(plan: str) -> CheckResult:
    """Defense against a contaminated / truncated handoff file (the #562
    harness-trailer incident class): minimum size, minimum structure, no
    lone stub token as the whole body."""
    cid, name = "c0_plan_nonstub", "plan non-stub"
    stripped = plan.strip()
    if re.fullmatch(r"(?i)[\s#*`>-]*(placeholder|tbd|todo|stub)[.!]?\s*", stripped or " "):
        return _fail(cid, name, "plan body is a lone stub token — broken handoff (#562 class)")
    if len(stripped) < MIN_PLAN_CHARS:
        return _fail(
            cid,
            name,
            f"plan body is {len(stripped)} chars (< {MIN_PLAN_CHARS}) — looks like a "
            "stub or truncated handoff (#562 class); persist the real plan first",
        )
    # Count markdown headings first; also count HTML headings to accept the
    # HTML output format documented in CLAUDE.md § Output format
    # (adversarial-planner defaults to HTML for browser-reading; an HTML plan
    # with 20+ <h2>/<h3> tags was incorrectly FAILed at "only 1 heading (< 3)"
    # because _headings() is markdown-only — incident task #640, 2026-06-15).
    n_headings = len(_headings(plan)) + len(_HTML_HEADING_RE.findall(plan))
    if n_headings < MIN_PLAN_HEADINGS:
        return _fail(
            cid,
            name,
            f"only {n_headings} headings (< {MIN_PLAN_HEADINGS}) — not a structured plan",
        )
    return _pass(cid, name, f"{len(stripped)} chars, {n_headings} headings")


# ─── Check 1 — §11 hyperparameter Source: grounding ────────────────────────


def _is_evidence_value(value: str) -> bool:
    """True when a Source value carries evidence: an arXiv id, a prior
    issue ``#<M>``, a file path, a URL, ``ungrounded``, or ≥2 non-stopword
    tokens (excluding the boilerplate words of planner.md's own "One
    `Source:` per unique value" sentence — prose ABOUT sources does not
    count)."""
    v = value.strip().strip("`*").strip()
    if not v:
        return False
    if "ungrounded" in v.lower():
        return True
    if re.search(r"\b\d{4}\.\d{4,5}\b", v):  # arXiv id
        return True
    if re.search(r"#\d+", v):  # prior issue
        return True
    if re.search(r"https?://", v):
        return True
    if re.search(r"[\w./-]+\.(?:py|md|json|jsonl|yaml|yml|sh|csv|txt)\b", v):  # file path
        return True
    tokens = [
        t for t in re.findall(r"[A-Za-z][\w-]*", v) if t.lower() not in _SOURCE_VALUE_STOPWORDS
    ]
    return len(tokens) >= 2


def _blankish(value: str) -> bool:
    t = value.strip().strip("`*").strip()
    return (not t) or t.lower().startswith("tbd") or set(t) <= {"?"}


_TABLE_SEP_RE = re.compile(r"\|?(?:\s*:?-{2,}:?\s*\|)+\s*:?-{0,}:?\s*\|?")


def _split_table_row(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def _source_column_cells(text: str) -> list[str]:
    """Body cells of every markdown-table column whose header cell is
    exactly ``Source`` (case-insensitive; bold/backticks stripped) — the
    #614 v2 §11 shape (`| What | Why (tied to Goal) | Source | ... |`)."""
    lines = text.splitlines()
    cells: list[str] = []
    i = 0
    while i < len(lines) - 1:
        header = lines[i].strip()
        sep = lines[i + 1].strip()
        if not (header.startswith("|") and sep.startswith("|") and _TABLE_SEP_RE.fullmatch(sep)):
            i += 1
            continue
        header_cells = [c.strip().strip("*`").strip().casefold() for c in _split_table_row(header)]
        col = next((j for j, c in enumerate(header_cells) if c == "source"), None)
        k = i + 2
        while k < len(lines) and lines[k].strip().startswith("|"):
            if col is not None:
                row = _split_table_row(lines[k])
                if col < len(row):
                    cells.append(row[col])
            k += 1
        i = k
    return cells


def check_source_grounding(plan: str, kind: str) -> CheckResult:
    """Contract (CLAUDE.md Critical Rule + planner.md §11): every
    load-bearing hyperparameter carries a non-empty ``Source:`` (inline
    label or a ``Source`` table column), or the explicit ``ungrounded —
    needs smoke-test`` marker, or the section-level N/A. Presence-only:
    Source correctness / transfer stays fact-checker-owned."""
    cid, name = "c1_source_grounding", "§11 hyperparameter Source: grounding"
    if kind in EXEMPT_KINDS:
        return _skip(cid, name, "kind-exempt: analysis|infra|batch|survey train no model")
    sect = section_text_by_keywords(
        plan, ("decision rationale", "hyperparameter grounding", "decision grounding")
    )
    scope = sect if sect is not None else plan
    if _standalone_na_declared(
        scope, r"no (?:model )?(?:training )?(?:model training|hyperparameters|training)"
    ):
        return _pass(
            cid, name, "explicit N/A declared (no model training / no training hyperparameters)"
        )
    text = strip_fences(scope)
    raw_inline = [m.group(1) for m in _SOURCE_LABEL_RE.finditer(text)]
    inline = [v for v in raw_inline if _is_evidence_value(v)]
    table_all = _source_column_cells(text)
    table_cells = [c for c in table_all if _is_evidence_value(c)]
    blank = [v for v in raw_inline if _blankish(v)] + [c for c in table_all if _blankish(c)]
    sources = inline + table_cells
    if sect is None and not sources and not blank:
        return _fail(
            cid,
            name,
            "no Decision Rationale / grounding section and zero Source entries — every "
            "load-bearing hyperparameter needs a Source (planner.md §11); if the plan trains "
            "no model, declare `N/A — no model training` / `N/A — no training hyperparameters` "
            "— each on its own line, unwrapped (no backticks/quotes)",
        )
    if blank:
        return _fail(
            cid,
            name,
            f"{len(blank)} blank/TBD Source entr{'y' if len(blank) == 1 else 'ies'} — "
            "planner.md §11 says never blank: cite a source or write "
            "`ungrounded — needs smoke-test`",
        )
    if sect is None:
        return _warn(
            cid,
            name,
            f"{len(sources)} Source entries found but no recognizable §11 heading "
            "(heading drift?) — fact-checker must locate them manually",
        )
    if not sources:
        return _fail(
            cid,
            name,
            "§11-style section present but zero Source entries (an inline source label — "
            "`Source` followed by a colon — or a `Source` table column)",
        )
    ungrounded = [s for s in sources if "ungrounded" in s.lower()]
    return _pass(
        cid,
        name,
        f"{len(sources)} Source entries: {len(inline)} inline, {len(table_cells)} table-column "
        f"({len(ungrounded)} marked ungrounded — fact-checker flags those for smoke-test); "
        "presence-only — Source correctness/transfer stays fact-checker-owned",
    )


# ─── Check 2 — per-DV measurement validity ─────────────────────────────────


def check_measurement_validity(plan: str, kind: str) -> CheckResult:
    """planner.md §6 required block: per dependent variable, the construct,
    the metric, and the on-distribution status. FAIL only when ALL signals
    are absent; a bare heading without construct/metric content is a WARN
    with the residual explicitly fact-checker-owned."""
    cid, name = "c2_measurement_validity", "per-DV measurement validity"
    if kind != "experiment":
        return _skip(cid, name, "kind-exempt: analysis|infra|batch|survey have no behavioral DV")
    if _standalone_na_declared(plan, r"no behavioral construct"):
        return _pass(cid, name, "explicit N/A declared (no behavioral construct)")
    text = strip_fences(plan)
    mv_headings = [h for h in _headings(plan) if "measurement validity" in h.text.casefold()]
    table = re.search(r"(?im)^\|(?=[^\n]*construct)(?=[^\n]*metric)[^\n]*\|\s*$", text)
    phrase = re.search(r"(?i)measurement validity", text)
    ondist = re.search(r"(?i)on-?distribution|on-?policy|teacher-?forced", text)
    heading_has_content = False
    if mv_headings:
        h = mv_headings[0]
        body = "\n".join(plan.splitlines()[h.line + 1 : h.end])
        heading_has_content = re.search(r"(?i)construct|metric", strip_fences(body)) is not None
    if table or heading_has_content:
        return _pass(
            cid,
            name,
            "measurement-validity block found with construct/metric content"
            + ("" if ondist else " (no on-distribution/on-policy statement spotted — verify)"),
        )
    if mv_headings:
        return _warn(
            cid,
            name,
            "measurement-validity heading present but no construct/metric content detected "
            "in its section — per-DV substance is fact-checker-owned",
        )
    if phrase:
        return _warn(
            cid, name, "phrase present but no recognizable block/table — verify per-DV rows exist"
        )
    return _fail(
        cid,
        name,
        "no measurement-validity declaration (planner.md §6 required block: per-DV construct "
        "+ metric + on-distribution status; non-behavioral plans declare "
        "`N/A — no behavioral construct` on its own line, unwrapped — no backticks/quotes)",
    )


# ─── Check 3 — data-source tier (WARN-only) ────────────────────────────────


def check_data_tier(plan: str, kind: str) -> CheckResult:
    """CLAUDE.md realistic-data preference order: the plan names its data
    tier. WARN-only — the vocabulary is descriptive, not a pinned string
    contract."""
    cid, name = "c3_data_tier", "data-source tier named"
    if kind != "experiment":
        return _skip(cid, name, "kind-exempt")
    text = strip_fences(plan)
    m = _TIER_RE.search(text)
    if not m:
        return _warn(
            cid,
            name,
            "no data-source tier named — CLAUDE.md realistic-data rule requires naming the "
            "tier (real-world / established dataset / diverse-LLM-synthetic / programmatic) "
            "+ tier-3/4 justification",
        )
    detail = f"data-tier vocabulary found ({m.group(0)!r})"
    if _TIER_34_RE.search(text) and not re.search(r"(?i)justif|absence|confound", text):
        detail += (
            "; note: tier-3/4 vocabulary present without a justification token "
            "(justif|absence|confound) — critics should verify the required justification"
        )
    return _pass(cid, name, detail)


# ─── Check 4 — contrastive negatives (WARN-only, conditional) ──────────────


def check_contrastive_negatives(plan: str, kind: str) -> CheckResult:
    """Behavior-implantation plans must name a contrastive-negative set or
    one of the two named exemptions (.claude/rules/contrastive-negatives.md).
    WARN not FAIL: the trigger is a content heuristic and the Methodology
    critic REVISEs the true positives — this gate surfaces, never
    adjudicates."""
    cid, name = "c4_contrastive_negatives", "contrastive negatives (behavior implantation)"
    if kind != "experiment":
        return _skip(cid, name, "kind-exempt")
    text = strip_fences(plan)
    implant = re.search(r"(?i)\bimplant\w*\b", text) or re.search(
        r"(?i)behavior[- ]implantation", text
    )
    marker_trigger = _MARKER_VOCAB_RE.search(text) and re.search(r"(?i)\bpersona\b", text)
    if not (implant or marker_trigger):
        return _skip(
            cid,
            name,
            "not detected as behavior-implantation (no implant/leakage-marker vocabulary)",
        )
    if _standalone_na_declared(plan, r"not a behavior[- ]implantation"):
        return _pass(
            cid,
            name,
            "explicit N/A declared on its own line, unwrapped (not a behavior-implantation)",
        )
    if re.search(r"(?i)contrastive[- ]negatives?", text):
        lowered = text.lower()
        found = [t for t in ("panel", "ratio", "1:1", "disjoint") if t in lowered]
        return _pass(
            cid,
            name,
            "contrastive-negative vocabulary present"
            + (
                f" (also found: {', '.join(found)})"
                if found
                else " (none of panel/ratio/1:1/disjoint spotted — verify composition)"
            ),
        )
    if re.search(
        r"(?i)single manipulated variable is contrastive|positive-only (?:parent|paper)"
        r"|exemption \(?[ab]\)?",
        text,
    ):
        return _pass(cid, name, "named exemption vocabulary present")
    return _warn(
        cid,
        name,
        "behavior-implantation vocabulary detected but no contrastive-negative set or named "
        "exemption — .claude/rules/contrastive-negatives.md (panel + ratio + disjointness); "
        "Methodology critic must gate this; or declare `N/A — not a behavior-implantation` "
        "on its own line, unwrapped (no backticks/quotes)",
    )


# ─── Check 5 — GPU-hour estimate (FAIL for ALL kinds) ──────────────────────


def check_gpu_hours(plan: str, kind: str) -> CheckResult:
    """The one exact string contract (planner.md §0): a machine-readable
    ``Estimated GPU-hours (total): <number>`` line. FAILs for ALL kinds —
    the Step 2c consumer (`task.py` `_resolve_autonomous_plan_gate`) is
    kind-blind and parks an autonomous session on a missing estimate;
    exempt kinds satisfy the check with ``0``. Scanned on the RAW plan
    (the line legitimately appears backtick-wrapped inside summary
    bullets / tables)."""
    cid, name = "c5_gpu_hours", "GPU-hour estimate line"
    del kind  # deliberately kind-blind, mirroring the Step 2c gate
    m = GPU_LINE_RE.search(plan)
    if not m:
        if GPU_LABEL_RE.search(plan):
            return _fail(
                cid,
                name,
                "`Estimated GPU-hours (total):` label present but the value is unparseable — "
                "a single plain number is required (no `~`, no ranges); exempt kinds use "
                "`Estimated GPU-hours (total): 0`",
            )
        return _fail(
            cid,
            name,
            "machine-readable `Estimated GPU-hours (total): <number>` line absent — required "
            "for ALL kinds (the Step 2c autonomous plan gate is kind-blind and parks on a "
            "missing estimate); exempt kinds satisfy with `Estimated GPU-hours (total): 0`",
        )
    # Range scan, scoped to the text immediately after the label and
    # stopping at the first parenthetical, em-dash, closing-backtick, or
    # sentence-boundary annotation — NOT the whole line (#610 carries
    # "— worst ≈ 42 — see §9" and #614 carries "1× A100-80" on the same  # noqa: RUF003
    # line; #580 carries "`. Wall ~1–1.5 h including review." after the  # noqa: RUF003
    # backtick-wrapped value — calibration-driven predicate adjustment,
    # plan §12; a whole-line digit-dash-digit scan would false-FAIL all
    # three shapes).
    line_end = plan.find("\n", m.end())
    if line_end == -1:
        line_end = len(plan)
    tail = plan[m.start(1) : line_end]
    # Backtick-tolerant range detection FIRST, anchored at the value:
    # the closing-backtick annotation stop below would otherwise truncate
    # a backtick-wrapped-number range at the first close backtick and
    # PASS it as its first number (round-2 fix; counterexamples that must
    # FAIL: `4`-8, `4`-`8`, `4` - 8, `40`-200). Anchoring via .match()
    # keeps the #580 next-sentence wall-time range and the #610/#614
    # annotation shapes out of reach — those put a non-dash token between
    # the value and any later digit-dash-digit text.
    range_m = GPU_RANGE_AT_VALUE_RE.match(tail)
    if range_m:
        return _fail(
            cid,
            name,
            f"value reads as a range, not a single number ({range_m.group(0).strip()!r}) — "
            "the Step 2c gate needs one number (put worst-case bounds in a parenthetical "
            "annotation)",
        )
    for stop in ("(", "—", "`", ". "):
        idx = tail.find(stop)
        if idx != -1:
            tail = tail[:idx]
    if re.search(r"[0-9]\s*[-–]\s*[0-9]", tail):  # noqa: RUF001 — en-dash ranges are real
        return _fail(
            cid,
            name,
            f"value reads as a range, not a single number ({tail.strip()!r}) — the Step 2c "
            "gate needs one number (put worst-case bounds in a parenthetical annotation)",
        )
    return _pass(cid, name, f"{m.group(1)} GPU-h")


# ─── Check 6 — reused-artifact fitness (WARN-only, conditional) ────────────


def check_reuse_fitness(plan: str, kind: str) -> CheckResult:
    """Plans reusing trained HF artifacts must carry the fitness
    attestations (a)-(j) (.claude/rules/artifact-reuse.md). WARN not FAIL:
    trigger and item-detection are both heuristic, and the demonstrated
    failure modes (#545/#600/#601) are semantic — the gate's value is
    forcing the section to exist and naming the ten letters.

    Accepted declaration shapes (#1314): the historical 'fitness'
    vocabulary, a 'reuse map' / 'reuse-map' section (the #1090 v7
    '### D3 — Reuse map' shape; artifact-reuse.md's own term for the
    plan record), '(self-)attestation(s)', or the literal (a)-(j) range
    token (hyphen / en-dash / em-dash / ellipsis)."""
    cid, name = "c6_reuse_fitness", "reused-artifact fitness attestation"
    if kind != "experiment":
        return _skip(cid, name, "kind-exempt")
    text = strip_fences(plan)
    hf_hits = [
        m.start() for m in re.finditer(r"superkaiba1/|adapter_config\.json|hf_hub_download", text)
    ]
    reuse_near_hf = any(
        re.search(r"(?i)\breus\w*", text[max(0, i - 300) : i + 300]) for i in hf_hits
    )
    reuse_heading = any(re.search(r"(?i)reuse|reused[- ]artifact", h.text) for h in _headings(plan))
    if not (reuse_near_hf or reuse_heading):
        return _skip(cid, name, "no HF-artifact reuse detected")
    if _standalone_na_declared(plan, r"no (?:artifact )?reuse"):
        return _pass(cid, name, "explicit no-reuse declaration (N/A — no artifact reuse)")
    declaration = re.search(
        r"(?i)fitness"  # historical vocabulary (the pre-#1314 detector, unchanged)
        r"|reuse[- ]map"  # 'Reuse map' section shape (#1090 v7 D3; artifact-reuse.md's own term)
        r"|(?:self[- ])?attestation"  # 'self-attestation' / 'attestation(s)'
        r"|\(a\)\s*[-–—…]\s*\(j\)",  # (a)-(j) range token; en-dash in real plans  # noqa: RUF001
        text,
    )
    letters = {m.group(1) for m in re.finditer(r"\(([a-j])\)", text)}
    if declaration and len(letters) >= 4:
        return _pass(
            cid,
            name,
            f"fitness/reuse-map declaration present ({len(letters)}/10 lettered items spotted)",
        )
    if declaration:
        return _warn(
            cid,
            name,
            f"fitness/reuse-map declaration vocabulary present but only {len(letters)} of the "
            "(a)–(j) items detectable — verify all ten attestations (recipe/regime/cells/"  # noqa: RUF001
            "single-var/hub-resolution/content-identity/scaling/backend-fetchability/"
            "code-throughput/pair-provenance) before approval",
        )
    return _warn(
        cid,
        name,
        "plan reuses HF artifacts but no fitness check / (a)–(j) reuse-map attestation found — "  # noqa: RUF001
        "CLAUDE.md reuse rule requires attestations (a)–(j); consistency-checker + Methodology "  # noqa: RUF001
        "critic must gate this",
    )


# ─── Check 7 — replication fidelity (WARN-only, conditional) ───────────────


def check_replication_fidelity(plan: str, kind: str) -> CheckResult:
    """When the Goal mentions replicating, the plan must address
    replication fidelity (match the paper's data + recipe first;
    .claude/rules/replication-fidelity.md). WARN because "does the effect
    replicate across seeds" is a benign false trigger."""
    cid, name = "c7_replication_fidelity", "replication fidelity"
    if kind != "experiment":
        return _skip(cid, name, "kind-exempt")
    goal = section_text_by_keywords(plan, ("goal",))
    if goal is None:
        m = re.search(r"(?im)^goal:\s*(.+)$", plan)
        goal = m.group(0) if m else None
    if goal is None or not re.search(r"(?i)replicat", goal):
        return _skip(cid, name, "Goal does not mention replication")
    text = strip_fences(plan)
    if _standalone_na_declared(plan, r"not a replication"):
        return _pass(
            cid, name, "explicit N/A declared on its own line, unwrapped (not a replication)"
        )
    if re.search(
        r"(?i)paper'?s (?:data|recipe|corpus)|faithful|replication[- ]fidelity|deviation", text
    ):
        return _pass(
            cid,
            name,
            "replication-fidelity vocabulary present (paper recipe / deviations addressed)",
        )
    return _warn(
        cid,
        name,
        "Goal mentions replication but no fidelity vocabulary — match the data + recipe of "
        "the source paper FIRST and name every divergence (replication rule, CLAUDE.md); "
        "or declare `N/A — not a replication` on its own line, unwrapped "
        "(no backticks/quotes)",
    )


# ─── Check 8 — success + kill criteria ─────────────────────────────────────


def _tldr_ranges(plan: str) -> list[tuple[int, int]]:
    """Line ranges of the §0.0 / TL;DR region(s). planner.md §0.0 MANDATES
    a "What would change my mind" line there, so a KILL hit inside is
    template conformance, not evidence of real kill criteria."""
    out: list[tuple[int, int]] = []
    for h in _headings(plan):
        text = h.text.strip()
        if "tl;dr" in text.casefold() or re.match(r"(?:§\s*)?0\.0\b", text):
            out.append((h.line, h.end))
    return out


def _exempt_tldr_kill_pass(
    cid, name, kind, succ_solid, kill_solid, kill_tldr_hits, carrier_ok, section_name
):
    """PASS result for the #1291 exempt-kind acceptance — a kind-exempt plan
    whose kill family is satisfied by a solid §0.0/TL;DR change-my-mind hit
    (success family solid, no solid kill outside the TL;DR); None when the
    acceptance does not apply (check_success_kill falls through to its
    missing-family verdicts)."""
    if kind not in EXEMPT_KINDS or not succ_solid or kill_solid:
        return None
    tldr_solid = [(i, a) for i, a in kill_tldr_hits if carrier_ok(i)]
    if not tldr_solid:
        return None
    si, sa = succ_solid[0]
    ka = tldr_solid[0][1]
    return _pass(
        cid,
        name,
        f"success anchor {sa!r} in §{section_name(si)!r}; kill anchor {ka!r} inside "
        "the §0.0/TL;DR region — accepted for kind-exempt plans (the mandated "
        "change-my-mind line IS the revert criterion for a code/infra change; "
        "kind: experiment still requires kill criteria outside the TL;DR — #1291)",
    )


def check_success_kill(plan: str, kind: str) -> CheckResult:
    """Both a success-criteria family and a kill-criteria family must be
    present and non-empty in form (each carrier section ≥ 80 chars —
    emptiness check only; semantic joint-satisfiability stays with the
    Statistics critic per planner.md §7). The KILL count EXCLUDES the
    §0.0/TL;DR region for ``kind: experiment``; for exempt kinds
    (analysis/infra/batch/survey) a solid TL;DR "What would change my
    mind" hit satisfies the kill family when the success family is solid —
    the mandated change-my-mind line IS the revert criterion for a
    code/infra change (#1291; founding incidents #1279/#1276).
    `kind: experiment` FAILs on both-absent; exempt kinds WARN, and the
    exempt-kind missing-kill WARN detail carries the standard §0.0 remedy
    sentence."""
    cid, name = "c8_success_kill_criteria", "success + kill criteria"
    lines = plan.splitlines()
    mask = _fence_mask(lines)
    headings = _headings(plan)
    tldr = _tldr_ranges(plan)

    def in_tldr(i: int) -> bool:
        return any(s <= i < e for s, e in tldr)

    def carrier_ok(i: int) -> bool:
        h = _innermost_section(headings, i)
        body = "\n".join(lines[h.line + 1 : h.end]) if h else plan
        return len(strip_fences(body).strip()) >= MIN_CRITERIA_CARRIER_CHARS

    def section_name(i: int) -> str:
        h = _innermost_section(headings, i)
        return h.text if h else "<preamble>"

    succ_hits: list[tuple[int, str]] = []
    kill_hits: list[tuple[int, str]] = []
    kill_tldr_hits: list[tuple[int, str]] = []
    for i, line in enumerate(lines):
        if mask[i]:
            continue
        m = _SUCCESS_RE.search(line)
        if m:
            succ_hits.append((i, m.group(0)))
        m = _KILL_RE.search(line)
        if m:
            (kill_tldr_hits if in_tldr(i) else kill_hits).append((i, m.group(0)))

    succ_solid = [(i, a) for i, a in succ_hits if carrier_ok(i)]
    kill_solid = [(i, a) for i, a in kill_hits if carrier_ok(i)]

    if succ_solid and kill_solid:
        si, sa = succ_solid[0]
        ki, ka = kill_solid[0]
        return _pass(
            cid,
            name,
            f"success anchor {sa!r} in §{section_name(si)!r}; kill anchor {ka!r} in "
            f"§{section_name(ki)!r} (form-only check — joint satisfiability stays with the "
            "Statistics critic)",
        )
    exempt_pass = _exempt_tldr_kill_pass(
        cid, name, kind, succ_solid, kill_solid, kill_tldr_hits, carrier_ok, section_name
    )
    if exempt_pass is not None:
        return exempt_pass
    missing = []
    if not succ_solid:
        missing.append(
            "success criteria (success/acceptance criteria, decision rule/gate)"
            + (" [vocabulary found but carrier section looks empty]" if succ_hits else "")
        )
    if not kill_solid:
        missing.append(
            "kill criteria (kill/abort/stop criteria, halt-and-report) outside the §0.0/TL;DR "
            "region — the TL;DR's mandated 'What would change my mind' line is template "
            "conformance, not kill criteria"
            + (" [vocabulary found but carrier section looks empty]" if kill_hits else "")
        )
    detail = (
        "missing: "
        + "; ".join(missing)
        + ". Note: a `No gates — short run / pre-verified hypothesis` escape waives *gates*, "
        "not success/kill criteria"
    )
    if not kill_solid and kind in EXEMPT_KINDS:
        detail += (
            ". Standard remedy for kind-exempt plans: add the mandated §0.0 TL;DR "
            "'What would change my mind' line (a solid one satisfies this family — #1291)"
        )
    if len(missing) == 2 and kind == "experiment":
        return _fail(cid, name, detail)
    if len(missing) == 2:
        return _warn(cid, name, detail + " (kind-exempt degrade: WARN, not FAIL)")
    return _warn(cid, name, detail)


# ─── Check 9 — conditions/cells table + seeds (WARN-only) ──────────────────


def check_conditions_seeds(plan: str, kind: str) -> CheckResult:
    """The consistency-checker's input surface: a conditions/cells/arms
    declaration and seeds. A WARN tells the orchestrator the
    consistency-checker will be flying partially blind."""
    cid, name = "c9_conditions_seeds", "conditions/cells + seeds declared"
    if kind != "experiment":
        return _skip(cid, name, "kind-exempt")
    text = strip_fences(plan)
    cond_heading = any(
        re.search(r"(?i)\b(conditions?|cells?|arms?)\b", h.text) for h in _headings(plan)
    )
    cond_table = re.search(r"(?im)^\|(?=[^\n]*(?:config slug|what it tests))[^\n]*\|\s*$", text)
    conditions = bool(cond_heading or cond_table)
    seeds = re.search(r"(?i)\bseeds?\b", text) is not None
    if conditions and seeds:
        return _pass(cid, name, "conditions/cells signal + seeds named")
    missing = []
    if not conditions:
        missing.append("conditions/cells/arms heading or table")
    if not seeds:
        missing.append("seeds")
    return _warn(
        cid,
        name,
        f"missing: {', '.join(missing)} — the consistency-checker's input surface is "
        "partially blind",
    )


# ─── Check 10 — marker-recipe acknowledgment (WARN-only, conditional) ──────


def check_marker_recipe(plan: str, kind: str) -> CheckResult:
    """Marker-leakage plans must acknowledge the training recipe (anchor
    band / band-stop / recipe file) AND bystander gating
    (.claude/rules/marker-training-recipe.md). Trigger scans fence-stripped
    text (a fence-only ※ example is not a marker plan); evidence scans the
    RAW plan (a fenced `marker_band_stop=...` config line IS an
    acknowledgment)."""
    cid, name = "c10_marker_recipe", "marker-recipe acknowledgment"
    if kind != "experiment":
        return _skip(cid, name, "kind-exempt")
    if not _MARKER_VOCAB_RE.search(strip_fences(plan)):
        return _skip(cid, name, "no marker-leakage vocabulary detected")
    recipe = re.search(r"(?i)marker-training-recipe|band[- ]?stop|\[5,\s*12\]\s*nat", plan)
    bystander = re.search(r"(?i)bystander", plan)
    if recipe and bystander:
        return _pass(
            cid,
            name,
            "recipe acknowledgment (band / recipe-file reference) + bystander gating present",
        )
    if recipe or bystander:
        missing = (
            "bystander-gating statement"
            if recipe
            else "recipe acknowledgment (marker-training-recipe / band-stop / [5,12] nat band)"
        )
        return _warn(
            cid,
            name,
            f"marker experiment missing {missing} — read .claude/rules/marker-training-recipe.md "
            "+ marker-leakage-measurement.md before grounding the stopping recipe",
        )
    return _warn(
        cid,
        name,
        "marker experiment with no recipe acknowledgment — read "
        ".claude/rules/marker-training-recipe.md + marker-leakage-measurement.md before "
        "grounding the stopping recipe (incident #530/#480 class)",
    )


# ─── Check 11 — dry-run test coverage (WARN-only, conditional) ─────────────


def _dryrun_test_evidence_lines(plan: str) -> list[str]:
    """Lines naming a dry-run-exercising test (see the regex-block comment
    by ``_DRYRUN_FLAG_RE``): a ``test_`` identifier alongside a dry-run
    token — with ``--dry-run`` flag occurrences stripped first, so the smoke
    command itself cannot self-certify — or the word "test" alongside the
    ``dry_run`` kwarg form."""
    out: list[str] = []
    for line in plan.splitlines():
        sans_flag = _DRYRUN_FLAG_RE.sub("", line)
        if (_TEST_IDENT_RE.search(sans_flag) and _DRYRUN_ANY_RE.search(sans_flag)) or (
            _TEST_WORD_RE.search(line) and _DRYRUN_KWARG_RE.search(line)
        ):
            out.append(line.strip())
    return out


def check_dryrun_test_coverage(plan: str, kind: str) -> CheckResult:
    """``kind: infra|batch`` plans whose verification includes a ``--dry-run``
    smoke must also list a test exercising the dry_run code path. Three infra
    plans in a row (#596, #607, #633) shipped success-path-only test lists
    while their own final acceptance step was a live ``--dry-run`` invocation
    — a broken dry_run thread turns that smoke into a real mutation (for
    #633: a real dispatch of up to 3 autonomous sessions). WARN not FAIL:
    trigger and evidence are both line heuristics; the Phase 2 critics
    adjudicate. Both scans use the RAW plan — smoke commands and test lists
    legitimately live inside fences and tables."""
    cid, name = "c11_dryrun_test_coverage", "dry-run smoke backed by a dry-run test"
    if kind not in ("infra", "batch"):
        return _skip(
            cid, name, "kind-exempt: the dry-run-smoke acceptance pattern is an infra|batch shape"
        )
    if not _DRYRUN_FLAG_RE.search(plan):
        return _skip(cid, name, "no --dry-run smoke/verification command detected")
    if _standalone_na_declared(plan, r"no dry-?run smoke"):
        return _pass(cid, name, "explicit N/A declared (no dry-run smoke)")
    evidence = _dryrun_test_evidence_lines(plan)
    if evidence:
        return _pass(cid, name, f"dry-run-exercising test named ({evidence[0][:80]!r})")
    return _warn(
        cid,
        name,
        "plan names a `--dry-run` smoke/verification command but the test list has no test "
        "exercising the dry-run kwarg thread on the new code path — a broken dry-run kwarg "
        "thread turns the final smoke into a real mutation (#596/#607/#633 pattern); add the "
        "test, or declare `N/A — no dry-run smoke` on its own line, unwrapped "
        "(no backticks/quotes), if the flag mention is incidental",
    )


# ─── Check 12 — battery multiplier + batched commitment (conditional) ──────

# Trigger: the plan names a permutation/null-draw battery — battery/null-draw
# framing, or an explicit >=100 count attached to draw vocabulary. Deliberate
# NON-triggers: a bare "bootstrap CI" / "bootstrapped 95% CI" (cheap post-hoc
# stat, ubiquitous in plans). Known accepted under-trigger: "bootstrap with
# B=2000 over all cells" carries no bootstrap alternation here — an
# under-trigger fails SAFE (the layered prose surfaces — planner §9 block,
# critic 10(iii)/12, implementer re-derivation — still fire); deliberate, not
# discovered (#869 plan §4.13).
# The count arm's lookbehind excludes range/scale-dash-preceded numbers —
# "graded 0-100 draws" is judge-scale vocabulary, not a battery (calibration
# false-FAIL on #779 v1); "1000 draws" after whitespace still triggers.
_BATTERY_TRIGGER_RE = re.compile(
    r"(?i)\b(null[- ]?(draws?|batter(y|ies))"
    r"|permutation[- ](tests?|batter(y|ies)|nulls?|draws?)"
    r"|n_(draws|perms)\b"
    r"|(?<![\d\u2013\u2014-])\d{3,}\s+(null[- ])?(draws|permutations|resamples))"
)

# Evidence (i): an explicit two-factor multiplier product where at least one
# factor is draw-bearing ("1000 draws x 24 cells" satisfies; a grid-only
# "34 x 50 x 28" or "layers x 3584" does NOT — the #810 false-PASS class where
# the forgotten draw multiplier is exactly what is absent).
_DRAW_FACTOR = (
    r"(?:\d[\d,_]*\s*(?:null[- ])?(?:draws?|perms?|permutations|resamples)"
    r"|n_(?:draws|perms|boot)\b|draws|perms|permutations|resamples|B\s*=\s*\d{3,})"
)
# The grid side accepts ANY axis factor — a count ("24"), a count + axis
# noun ("6 arms", "3 layers", "~3 quantities"), or a bare axis noun
# ("cells", "batteries") — optionally opened by an approximation / paren
# decoration ("~3", "≈8", "(6 arms"). The load-bearing discriminator is
# the DRAW-BEARING factor, not the grid noun: the #810 false-PASS class
# is a grid-only product with NO draw factor ("34 x 50 x 28",
# "layers x 3584", "6 arms x 3 layers x 16 folds") and still fails; a
# noun whitelist only rots ("batteries"/"quantities" false-FAILed the
# conforming #833 v8 sizing block — #1086).
_GRID_DECOR = r"(?:[~≈(]\s*)?"
_GRID_FACTOR = r"(?:\d[\d,_]*(?:\s+[A-Za-z][\w-]*)?|[A-Za-z][\w-]*)"
# The multiplication token plans actually write: the real multiplication
# sign plus the ASCII fallbacks. The multiplication sign and `*` are
# unambiguous and keep tight/zero-whitespace binding ("50*28"); ASCII `x`
# counts ONLY when standalone w.r.t. word chars — "layer-ma|x| perms",
# "shared-inde|x| draws", "honest_nulls_ma|x|draws", "draws |x|gboost"
# are word-split artifacts, not products (#1099; 27 realized corpus
# false-arith lines, 0 verdict flips on removal). Digit-tight ASCII forms
# ("2x2") also stop counting: every draw-bearing corpus product is spaced
# ("4 draws x 492 cells"), and "the 2x2 draws its factors" is the verb
# false-positive the digit carve-out would re-admit.
_MULT_TOKEN = r"(?:[×*]|(?<!\w)x(?!\w))"  # noqa: RUF001 — the multiplication sign is real plan text
_MULT_ARITH_RE = re.compile(
    rf"(?i)\b(?:{_DRAW_FACTOR}\s*{_MULT_TOKEN}\s*{_GRID_DECOR}{_GRID_FACTOR}"
    rf"|{_GRID_FACTOR}\s*{_MULT_TOKEN}\s*{_DRAW_FACTOR})\b"
)
# Arith-anchored windows (#1086) accepted fail-UNSAFE residual, DISCLOSED: a
# quoted SIBLING's sizing line ("#778's 10,000 draws x 24 cells, batched")
# can anchor its own window and satisfy THIS plan's battery — the same
# residual class as c18's documented residual (f) (non-verbatim paraphrase,
# beyond mechanical defense). Deliberately NO `#\d{2,}` citation guard on
# anchor lines: 192 corpus draw-arithmetic lines carry a same-line #-ref,
# and .claude/rules/plan-compute-sizing.md MANDATES citing a prior-issue
# MEASURED basis beside sizing arithmetic, so the guard would re-create the
# very false-positive class #1086 fixes (guard REJECTED in plan v2 §11).

# Evidence (ii): a named batched helper or an explicit vectorization
# statement. A token whose only in-window occurrence sits inside a citation /
# path of the rule file does NOT count — citing the rule is not an
# implementation commitment (the filename itself contains "vectorize", so the
# citation tokens are stripped from the window before this search).
_BATCHED_COMMIT_RE = re.compile(
    r"(?i)\b(batched|vectoriz(?:e|ed|es|ation)|subset-sum|GEMM|one\s+(?:masked\s+)?matmul"
    r"|perm_null_draws|randnorm_null_draws|vectorized_mlp_skill)\b"
)
_C12_RULE_CITATION_RE = re.compile(r"\S*vectorize-many-cell-fits\.md\S*")

# Evidence window: ± this many RAW lines around each trigger hit (arithmetic
# legitimately lives in tables/fences adjacent to the battery row).
_C12_WINDOW_LINES = 15


def _trigger_windows(plan: str, trigger_re: re.Pattern[str], window_lines: int) -> list[str]:
    """RAW-text windows (± ``window_lines`` raw lines) around each NON-fenced
    line matching ``trigger_re``. Trigger detection is fence-masked (a
    fence-only example is not a trigger — the line-preserving equivalent of
    searching ``strip_fences(plan)``); each WINDOW is raw text, so evidence
    inside adjacent tables/fences still counts. Shared by c12
    (``_BATTERY_TRIGGER_RE``, ±15), c16 (``_C16_EXTRACT_RE`` ±3;
    ``_C16_REGEN_RE`` at radius 0 = same-line adjacency), and c24
    (``_C24_TRIGGER_RE``, ±15)."""
    lines = plan.splitlines()
    mask = _fence_mask(lines)
    windows: list[str] = []
    for i, (line, fenced) in enumerate(zip(lines, mask, strict=True)):
        if fenced or not trigger_re.search(line):
            continue
        lo = max(0, i - window_lines)
        hi = min(len(lines), i + window_lines + 1)
        windows.append("\n".join(lines[lo:hi]))
    return windows


def _battery_trigger_windows(plan: str) -> list[str]:
    """Thin wrapper: c12's fence-masked ±15-raw-line trigger windows (see
    ``_trigger_windows``; kept so the c12 name + radius stay greppable)."""
    return _trigger_windows(plan, _BATTERY_TRIGGER_RE, _C12_WINDOW_LINES)


def check_battery_multiplier(plan: str, kind: str) -> CheckResult:
    """A plan naming a permutation/bootstrap/null-draw battery must carry,
    NEAR a battery mention (± 15 raw lines), BOTH (i) explicit multiplier
    arithmetic with a draw-bearing factor and (ii) a batched-implementation
    commitment. A draw-bearing arithmetic line ALSO anchors its own ± 15
    evidence window (#1086) — the §9 sizing block legitimately lives far
    from the §4/§6 battery registration. Window-scoped, never
    document-global — the document-global draft demonstrably false-PASSed
    the motivating incident plan (#810 v1) via an unrelated footprint
    product + helper boilerplate. FAIL (experiment) / WARN (analysis) /
    SKIP otherwise; a SURFACE check per the module's scope discipline —
    semantic adequacy of the arithmetic stays with the Phase 2 critics."""
    cid, name = "c12_battery_multiplier", "battery multiplier + batched commitment"
    if kind not in ("experiment", "analysis"):
        return _skip(cid, name, "kind-exempt: battery sizing is an experiment|analysis plan shape")
    windows = _battery_trigger_windows(plan)
    if not windows:
        return _skip(cid, name, "no permutation/null-draw battery named")
    if _standalone_na_declared(plan, r"no draw battery"):
        return _pass(cid, name, "explicit N/A declared (no draw battery)")
    # #1086: a draw-bearing arithmetic line ANCHORS its own ±15 evidence
    # window — the §9 sizing block legitimately lives far from the §4/§6
    # battery registration (#833 v8: 58+ lines). Window-scoped discipline is
    # preserved: only a line already carrying a draw-bearing product can
    # anchor (a grid-only footprint product never anchors — the #810 v1
    # false-PASS class), and the batched commitment must still sit within
    # ±_C12_WINDOW_LINES raw lines of the anchor.
    windows = windows + _trigger_windows(plan, _MULT_ARITH_RE, _C12_WINDOW_LINES)
    any_arith = False
    any_commit = False
    for window in windows:
        has_arith = bool(_MULT_ARITH_RE.search(window))
        has_commit = bool(_BATCHED_COMMIT_RE.search(_C12_RULE_CITATION_RE.sub("", window)))
        any_arith = any_arith or has_arith
        any_commit = any_commit or has_commit
        if has_arith and has_commit:
            return _pass(
                cid,
                name,
                "a battery window carries both the multiplier arithmetic and a "
                "batched-implementation commitment",
            )
    missing: list[str] = []
    if not any_arith:
        missing.append(
            "the multiplier arithmetic with a draw-bearing factor "
            "(draws times cells times folds at per-call cost = projected wall)"
        )
    if not any_commit:
        missing.append(
            "a batched-implementation commitment (a named batched helper or an explicit "
            "vectorization statement)"
        )
    if not missing:
        missing.append(
            "co-location: the multiplier arithmetic and the batched-implementation "
            "commitment each appear somewhere, but never together near any battery mention "
            "or draw-arithmetic sizing line"
        )
    detail = (
        f"plan names a permutation/bootstrap/null battery but is missing {' AND '.join(missing)}"
        " — a named battery defaults to a serial per-draw loop (#778: ~15 h realized vs 1 h"
        " planned; #810: 308x); see .claude/rules/vectorize-many-cell-fits.md, or declare"
        " 'N/A — no draw battery' on its own line, unwrapped (no backticks/quotes), if the"
        " mention is incidental"
    )
    if kind == "analysis":
        return _warn(cid, name, detail)
    return _fail(cid, name, detail)


# ─── Check 13 — empirical-null gate p-floor attainability (conditional) ────

# Gate alpha: a decimal alpha DIRECTLY after the comparator (comparator
# captured — strictness matters at the floor == alpha boundary). A
# fraction-form self-consistent floor gate ("p ≤ 1/(15+1) ≈ 0.06", #816 v5
# Exp-4) must NOT match: "1/" blocks the decimal, and the "≈ 0.06" is not
# comparator-adjacent.
_C13_P_ALPHA_RE = re.compile(r"(?i)\bp(?:-?values?)?\s*(≤|<=|<)\s*\*{0,2}`?(0?\.\d+)`?")
_C13_EMPIRICAL_RE = re.compile(r"(?i)\bempirical\b")
# Registered-gate section: any ENCLOSING heading matching the c8 success/kill
# families or an Evaluation heading. Lines elsewhere (Prior Work recaps,
# TL;DR) are not registrations — under-trigger fails safe (critics review).
_C13_GATE_SECTION_RE = re.compile(
    r"(?i)success criteri|acceptance criteri|decision rule|decision gate"
    r"|kill[- ]criteri|abort criteri|stop criteri|\bevaluation\b"
)
# On-gate-line draws-scope qualifier ("(n_draws ≥ 50: ...)" — the #816 v6 fix
# shape): families below K are OUTSIDE the gate's own declared scope. The
# DRAWS-EXPLICIT token is REQUIRED: a bare `n ≥ K` (e.g. "n ≥ 20 prompts per
# probe" — a sample-size clause on the gate line) must NOT set the scope, or
# it silently descopes every small-n_draws family and emits an affirmative
# false-PASS on the exact #816 class this check exists to catch.
_C13_SCOPE_RE = re.compile(r"(?i)\bn_(?:draws|perms)\w*\s*(?:≥|>=)\s*(\d+)")
# Family vocabulary on the gate line = the tie is unambiguous (the gate
# quantifies over null families) → FAIL-capable; absent → WARN cap.
_C13_FAMILY_RE = re.compile(r"(?i)famil")
# Per-declaration exclusion: a family row/line declaring itself outside the
# test set is dropped (v5/v6 contaminated-reference row; v6 "outside the BH").
_C13_EXCLUDE_RE = re.compile(
    r"(?i)contaminated|reference only|descriptive|excluded"
    r"|not (?:in|included|counted)|outside the (?:BH|test)"
)
# n_draws declarations, prose/kwarg forms: n_draws=K, n_draws: K,
# n_draws_isotropic=200, n_perms=500. ("n_draws ≥ 50" and "(n_draws+1)" do
# not match — comparator/paren, not =/:.)
_C13_NDRAWS_KWARG_RE = re.compile(r"(?i)\b(n_(?:draws|perms)\w*)\s*[=:]\s*(\d+)")

# Known accepted under-triggers (mirroring the c12 precedent): (a) a gate
# registered outside any success/kill/decision/evaluation-titled section;
# (b) a gate whose `p <= alpha` wraps across lines or uses `%`/LaTeX `\le`;
# (c) "empirical" absent from the gate line; (d) draw counts declared only as
# bare prose ("15 draws") without an `n_draws` label. (a)-(d) fail SAFE
# (under-trigger → SKIP; the plan still reaches the fact-checker + critic
# ensemble, whose statistics lens caught the original #816 incident). ONE
# known fail-UNSAFE direction: (e) a hard-wrapped gate whose `(n_draws ≥ K)`
# qualifier lands on the NEXT line is gate-detected without its qualifier →
# false-FAIL on a legitimately scoped gate. Accepted: repo plans favor long
# single lines (v5/v6 both do), the corpus-sweep calibration bounds it, and a
# false-FAIL costs 1-2 mechanical planner bounces with the PASS-with-override
# valve as the escape (adversarial-planner SKILL.md Phase 1.5.0) — §4.5 is
# NOT all-fails-safe.


def _n_draws_declarations(plan: str) -> list[tuple[str, int]]:
    """Deduplicated ``(label, n_draws)`` pairs harvested from the RAW plan:
    (1) markdown-table columns whose header cell CONTAINS ``n_draws`` /
    ``n_perms`` after bold/backtick stripping (v5's twin ``n_draws (Exp-2)`` /
    ``n_draws (Exp-4)`` columns both match; ALL matching columns per table are
    collected), and (2) prose/kwarg forms (``n_draws=K``, ``n_perms: K``,
    ``n_draws_isotropic=200``). Deliberately raw text — declarations
    legitimately live in tables and fenced config blocks (#816 v6's kwargs are
    fenced). A row/line matching ``_C13_EXCLUDE_RE`` (outside-the-test-set
    vocabulary) is dropped; a non-numeric cell is skipped."""
    lines = plan.splitlines()
    pairs: list[tuple[str, int]] = []
    seen: set[tuple[str, int]] = set()

    def add(label: str, n: int) -> None:
        key = (label, n)
        if key not in seen:
            seen.add(key)
            pairs.append(key)

    # (1) Table columns — a sibling of the c1 `_source_column_cells` walk,
    # with a contains-predicate on the header cell + multi-column collection.
    i = 0
    while i < len(lines) - 1:
        header = lines[i].strip()
        sep = lines[i + 1].strip()
        if not (header.startswith("|") and sep.startswith("|") and _TABLE_SEP_RE.fullmatch(sep)):
            i += 1
            continue
        header_cells = [c.strip().strip("*`").strip().casefold() for c in _split_table_row(header)]
        cols = [j for j, c in enumerate(header_cells) if "n_draws" in c or "n_perms" in c]
        k = i + 2
        while k < len(lines) and lines[k].strip().startswith("|"):
            row_text = lines[k]
            if cols and not _C13_EXCLUDE_RE.search(row_text):
                row = _split_table_row(row_text)
                # replace("**", "") drops INTERIOR bold markers (e.g.
                # `**Cross-trait** (ref)`) that a bare strip("*") keeps.
                label = row[0].replace("**", "").strip("*").strip() if row else ""
                for col in cols:
                    if col >= len(row):
                        continue
                    m = re.search(r"\d[\d,_]*", row[col])
                    if m:
                        add(label, int(m.group(0).replace(",", "").replace("_", "")))
            k += 1
        i = k
    # (2) Prose/kwarg declarations.
    for line in lines:
        if _C13_EXCLUDE_RE.search(line):
            continue
        for m in _C13_NDRAWS_KWARG_RE.finditer(line):
            add(m.group(1), int(m.group(2)))
    return pairs


def _c13_registered_gates(plan: str) -> list[dict]:
    """Registered empirical-p gate lines: non-fenced lines inside a
    success/kill/evaluation-titled section carrying "empirical" + at least
    one decimal alpha directly after ``p <=`` / ``p <``. Per gate: the
    stripped line, the MIN alpha on the line (a gate requiring the most
    stringent of several alphas is unattainable if the floor exceeds the
    smallest), whether the min-alpha comparator is strict ``<``, the on-line
    draws-scope qualifier K (or None), and whether family vocabulary is on
    the line."""
    lines = plan.splitlines()
    mask = _fence_mask(lines)
    headings = _headings(plan)
    gates: list[dict] = []
    for i, (line, fenced) in enumerate(zip(lines, mask, strict=True)):
        if fenced:
            continue
        if not any(h.line <= i < h.end and _C13_GATE_SECTION_RE.search(h.text) for h in headings):
            continue
        if not _C13_EMPIRICAL_RE.search(line):
            continue
        matches = list(_C13_P_ALPHA_RE.finditer(line))
        if not matches:
            continue
        alphas: list[tuple[Fraction, bool]] = []
        for m in matches:
            a = m.group(2)
            alphas.append(
                (Fraction("0" + a) if a.startswith(".") else Fraction(a), m.group(1) == "<")
            )
        min_alpha = min(a for a, _ in alphas)
        strict = any(s for a, s in alphas if a == min_alpha)
        scope_m = _C13_SCOPE_RE.search(line)
        gates.append(
            {
                "line": line.strip(),
                "alpha": min_alpha,
                "strict": strict,
                "scope": int(scope_m.group(1)) if scope_m else None,
                "family": bool(_C13_FAMILY_RE.search(line)),
            }
        )
    return gates


def _standalone_na_declared(plan: str, tail_re: str) -> bool:
    """True when ``N/A — <tail_re>`` appears as a deliberate STANDALONE
    declaration line (leading list/blockquote markers stripped), never
    doc-global: a FAIL detail quotes its escape phrase as a remedy option,
    and this project's convention pastes verifier/bounce text into revised
    plans verbatim — a substring match would let a bounced plan self-escape
    re-verification (the #810 spurious-satisfaction structure, one polarity
    over). NA_RE opens with an inline (?i), so it must sit at pattern
    position 0 — per-line re.match satisfies that; never prepend a prefix
    to NA_RE (py3.11+ rejects mid-pattern global flags). Shared by the
    checks' standalone-N/A escapes (the Supersede rule: one copy of the job).

    Wrapped declarations (a backtick/quote-wrapped paste of a remedy's
    quoted form) are DELIBERATELY unrecognized (#1238 reasoned no-change):
    the adversarial-planner SKILL.md canonical-phrases block renders its
    escape phrases backtick-wrapped at line start, nearly all of them
    helper-routed since the #1237/#1262 migrations, so every
    trailing-tolerant wrapper widening lets a verbatim block paste
    self-declare many checks' escapes at once; requiring a balanced
    closing wrapper does
    not discriminate (the block's wrappers are balanced by construction),
    and the strict phrase-alone-on-line variant rejects the one shape
    that measurably BOUNCED (#1090 plans/v1.md:369, trailing scope
    prose) while its target idiom (wrapped-alone lines, a real corpus
    habit) is covered at the source by the SKILL.md unwrapped contract.
    The most realistic hazard is not even a whole-block paste: a
    single-phrase bulleted bounce-brief line ("- <wrapped phrase> -
    <remedy prose>") is byte-shaped identically to a legitimate
    declaration. Declare escapes UNWRAPPED. Pinned:
    tests/test_verify_plan.py skillmd/wrapped pins.
    """
    lines = plan.splitlines()
    mask = _fence_mask(lines)
    for line, fenced in zip(lines, mask, strict=True):
        if fenced:
            continue
        if re.match(NA_RE + tail_re, line.lstrip(" \t>*-")):
            return True
    return False


def _c13_na_escape_declared(plan: str) -> bool:
    """Standalone ``N/A — no empirical-null gate`` escape (see
    ``_standalone_na_declared`` for the anti-paste rationale)."""
    return _standalone_na_declared(plan, r"no empirical[- ]null gate")


def _c13_evaluate(gates: list[dict], decls: list[tuple[str, int]]) -> dict:
    """Per-gate attainability arithmetic. Offender iff floor > alpha OR
    (floor == alpha AND the gate comparator is strict ``<`` — then the gate
    is unattainable, not boundary); floor == alpha under ``<=`` is boundary.
    A nonpositive alpha (``p ≤ 0.00``) is in-domain: floor = 1/(n+1) > 0 ≥
    alpha for EVERY draw count, so the same arithmetic classifies every
    in-scope declaration an offender (fail_capable per the normal
    family-vocab rule; the alpha ≤ 0 remedy lives in the detail builder).
    A gate whose scope qualifier excludes EVERY declaration is vacuous (an
    empty in-scope set must not yield an affirmative PASS with an undefined
    min)."""
    offenders: list[tuple[dict, str, int, Fraction]] = []
    boundary: list[tuple[str, int]] = []
    fail_capable = False
    vacuous_scope = False
    min_in_scope: int | None = None
    for g in gates:
        in_scope = [d for d in decls if g["scope"] is None or d[1] >= g["scope"]]
        if g["scope"] is not None and not in_scope:
            vacuous_scope = True
            continue
        for label, n in in_scope:
            if min_in_scope is None or n < min_in_scope:
                min_in_scope = n
            floor = Fraction(1, n + 1)
            if floor > g["alpha"] or (floor == g["alpha"] and g["strict"]):
                offenders.append((g, label, n, floor))
                fail_capable = fail_capable or g["family"]
            elif floor == g["alpha"]:
                boundary.append((label, n))
    return {
        "offenders": offenders,
        "boundary": boundary,
        "fail_capable": fail_capable,
        "vacuous_scope": vacuous_scope,
        "min_in_scope": min_in_scope,
    }


def _c13_offender_detail(offenders: list[tuple[dict, str, int, Fraction]]) -> str:
    """Bounded FAIL/WARN detail: the first offending gate line (truncated)
    + its alpha, at most 6 offenders, and the remedy menu (raise n_draws to
    >= ceil(1/alpha) for a clean PASS — n = 1/alpha - 1 exactly lands on the
    boundary WARN). A nonpositive alpha (e.g. a registered ``p ≤ 0.00`` —
    the limiting case of the unattainable-gate class) gets a dedicated
    remedy instead of ``ceil(1/alpha)``, which would ZeroDivisionError on
    ``Fraction(1, 0)`` — a parseable gate must never crash the module."""
    g0 = offenders[0][0]
    alpha0: Fraction = g0["alpha"]
    # Display-dedupe on (label, n): two gates sharing an offending family
    # would otherwise list it twice and push distinct offenders past the cap.
    uniq: list[tuple[str, int, Fraction]] = []
    for _, label, n, floor in offenders:
        if (label, n, floor) not in uniq:
            uniq.append((label, n, floor))
    shown = ", ".join(
        f"{label} n_draws={n} → floor {floor.numerator}/{floor.denominator} ≈ {float(floor):.3g}"
        for label, n, floor in uniq[:6]
    )
    if len(uniq) > 6:
        shown += ", …"
    if alpha0 <= 0:
        remedy = (
            "alpha ≤ 0 — no finite n_draws attains it (the p-floor 1/(n_draws+1) is "
            "positive for every draw count); raise the alpha or fix the gate"
        )
    else:
        remedy = (
            f"raise n_draws to ≥ {math.ceil(1 / alpha0)} for a clean PASS "
            "(n = 1/alpha - 1 exactly lands on the floor == alpha boundary WARN)"
        )
    return (
        f'plan registers an empirical-p gate ("{g0["line"][:90]}", alpha={float(alpha0):g}) '
        f"over families whose p-floor 1/(n_draws+1) exceeds alpha: {shown} — the gate is "
        f"structurally unattainable (#816 v5 class); {remedy}, scope the gate "
        "(e.g. 'n_draws ≥ 50'), mark the family outside the test set on its row, or declare "
        "'N/A — no empirical-null gate' on its own line, unwrapped (no backticks/quotes)"
    )


def check_empirical_gate_attainability(plan: str, kind: str) -> CheckResult:
    """A registered empirical-null gate (a success/kill/evaluation-section
    line requiring p ≤ alpha against null families) must be ATTAINABLE for
    every in-scope declared family: p_floor = 1/(n_draws+1) ≤ alpha.
    Necessary-condition logic only — under BH the effective per-test
    thresholds are ≤ alpha, so floor > alpha is conservative-correct; BH-m
    arithmetic, family-set semantics, and joint satisfiability stay with the
    Statistics critic (c8's form-only charter). FAIL (experiment) / WARN
    (analysis) / WARN on ambiguous tie or floor == alpha under a non-strict
    comparator / SKIP otherwise; escape via a standalone ``N/A — no
    empirical-null gate`` line — honored (SKIP path) only when no gate is
    detected; when the escape co-occurs with a detected gate the check WARNs
    instead of PASSing (regardless of whether n_draws declarations exist),
    so the escape can never mask attainability verification of a present
    gate (#1258, the #1223 c20 rule). Incident: #816 v5 (gate p ≤ 0.05 over
    families with n_draws=2/5 → floors 1/3, 1/6; caught only by the Codex
    statistics critic)."""
    cid, name = "c13_empirical_gate_attainability", "empirical-null gate p-floor attainability"
    if kind not in ("experiment", "analysis"):
        return _skip(
            cid,
            name,
            "kind-exempt: registered empirical-null gates are an experiment|analysis plan shape",
        )
    gates = _c13_registered_gates(plan)
    if not gates:
        return _skip(cid, name, "no registered empirical-p gate detected")
    if _c13_na_escape_declared(plan):
        # #1258 (the #1223 c20 port): this branch is reachable ONLY when a
        # registered empirical-p gate WAS detected (the no-gate case SKIPs
        # above) — a PASS here masks the p-floor attainability verification
        # c13 exists to run. WARN, not FAIL: the gate harvest may be a false
        # positive on quoted guidance, so the escape stays non-blocking and
        # the reviewers adjudicate.
        return _warn(
            cid,
            name,
            "the standalone `N/A — no empirical-null gate` escape co-occurs with "
            f"{len(gates)} registered empirical-p gate line(s) (first: "
            f"{gates[0]['line'][:90]!r}) — the escape is reserved for gate-free "
            "plans and would mask attainability verification of the detected "
            "gate (#1258, the #1223 c20 rule); remove the N/A line (the gate is "
            "then verified, or SKIPs as not-computable when no n_draws "
            "declarations exist), or fence/remove the gate-shaped prose the "
            "detector matched if it is quoted guidance rather than this plan's "
            "own registration",
        )
    decls = _n_draws_declarations(plan)
    if not decls:
        return _skip(
            cid,
            name,
            "empirical-p gate present but no per-family n_draws declarations found — "
            "attainability not computable at the plan surface",
        )
    ev = _c13_evaluate(gates, decls)
    if ev["offenders"]:
        detail = _c13_offender_detail(ev["offenders"])
        if kind == "analysis":
            return _warn(cid, name, detail + " (analysis kind-degrade: WARN, not FAIL)")
        if not ev["fail_capable"]:
            return _warn(
                cid,
                name,
                detail + " — ambiguous tie: no family vocabulary on any offending gate line; "
                "verify the flagged draw counts are in the gate's test set",
            )
        return _fail(cid, name, detail)
    if ev["boundary"]:
        label, n = ev["boundary"][0]
        return _warn(
            cid,
            name,
            f"p-floor equals the registered alpha exactly ({label} n_draws={n} → floor "
            f"1/{n + 1} = alpha) — attainable only when the real statistic beats every "
            "draw; state the floor next to the verdict",
        )
    if ev["vacuous_scope"]:
        return _warn(
            cid,
            name,
            "the gate's scope qualifier (n_draws ≥ K) excludes every declared family — "
            "attainability not computable for any in-scope family; verify the gate's "
            "in-scope families are declared",
        )
    min_in_scope = ev["min_in_scope"]
    return _pass(
        cid,
        name,
        f"min in-scope n_draws={min_in_scope} → p-floor 1/{(min_in_scope or 0) + 1} ≤ "
        "registered alpha (attainable in form; adequacy stays with the Statistics critic)",
    )


# ─── Check 14 — hypothesis confirm/falsify branch coherence (WARN-only) ────

# Branch anchors: `**Confirm:**`, `**Confirm (ridge stands):**`,
# `**Confirm-the-null:**`, `**Falsify:**`, `**Falsify (positive surprise):**`
# — all observed corpus shapes.
_BRANCH_ANCHOR_RE = re.compile(r"(?i)\*\*\s*(confirm|falsif)")
# Shared bounded token: a normalized `var = value` pair present in BOTH
# branch segments (the #922 H4 `k = 32` horizon shape). Comparator-bearing
# bounds (`k ≤ 4`) are deliberately NOT harvested — requiring exact-pair
# identity in both segments is the main false-positive guard.
_BOUND_TOKEN_RE = re.compile(r"\b([A-Za-z]\w{0,8})\s*=\s*(\d+(?:\.\d+)?)\b")
# Tendency-class comparator (does not pin an end-state). Deliberately
# minimal: a bare "declines" without "toward" is an accepted false negative
# (prefer false negatives); "approaches"/"converges" excluded in v1
# ("two approaches" false-fires on the noun).
_TENDENCY_RE = re.compile(r"(?i)\btowards?\b")
# State-class comparator (pins a region through the horizon).
_STATE_RE = re.compile(
    r"(?i)\b(?:stays?|remains?|holds?)\s+(?:strictly\s+)?(?:above|below|at|within)\b"
)
# Vague layer-scope tokens ("mid/late layers", "most layers", incl. "at most
# layers"). "a majority of layers" is deliberately EXCLUDED (a quantifier
# over a universe; in the observed corpus it co-occurs with a pinned one).
_VAGUE_SCOPE_RE = re.compile(
    r"(?i)\b(?:(?:early|mid|middle|late|deep|shallow)"
    r"(?:\s*[/-]\s*(?:early|mid|middle|late|deep|shallow))?|most)\s+layers\b"
)
# Pinned-anchor escape (same block): "layers 1-28" (any dash), "layer 20",
# "layers {18, 21}", "L18", the pre-registered layer symbol (script small l,
# U+2113, followed by *), or the literal "pre-registered".
_PINNED_SCOPE_RE = re.compile(
    r"(?i)\blayers?\s*\d|\blayers?\s+\{|\bL\d{1,2}\b|\u2113\*|\bpre-registered\b"
)
# Per-hypothesis block starts: top-level list items (sub-headings are
# detected via _HEADING_RE).
_C14_LIST_ITEM_RE = re.compile(r"^\s{0,3}(?:[-*]|\d+\.)\s")
# Bold span used to label an offending block in the WARN detail.
_C14_BOLD_LABEL_RE = re.compile(r"\*\*([^*\n]{1,60})\*\*")


def _hypothesis_blocks(section_text: str) -> list[str]:
    """Split a (fence-stripped) hypothesis-section text into per-hypothesis
    blocks at top-level list-item starts and heading lines; continuation
    lines join the preceding block. The section heading line starts the
    first block (it carries no branch anchors, so it is ignored downstream).
    Matches the observed corpus: one bullet per `**H<k>**` (#922 v2, #841
    v12, #810 v6 all use single-bullet hypothesis blocks)."""
    blocks: list[list[str]] = []
    current: list[str] = []
    for line in section_text.splitlines():
        if _C14_LIST_ITEM_RE.match(line) or _HEADING_RE.match(line.strip()):
            if current:
                blocks.append(current)
            current = [line]
        else:
            current.append(line)
    if current:
        blocks.append(current)
    return ["\n".join(b) for b in blocks]


def _confirm_falsify_segments(block: str) -> tuple[str, str] | None:
    """``(confirm_segment, falsify_segment)`` for a hypothesis block, or
    ``None`` when the block has no falsify anchor (nothing to compare —
    c8 owns branches missing entirely; a lone ``**Confirm`` block is also
    ignored). Falsify segment = first falsify anchor to the next anchor or
    block end. Confirm segment = explicit confirm anchor to the next anchor
    when one exists; otherwise the block text BEFORE the falsify anchor
    (the hypothesis statement itself is the implicit confirm branch — the
    #922 H4 shape, which has no ``**Confirm:**`` label)."""
    anchors = list(_BRANCH_ANCHOR_RE.finditer(block))
    falsifies = [m for m in anchors if m.group(1).casefold().startswith("falsif")]
    if not falsifies:
        return None
    f0 = falsifies[0]
    after_f = [m for m in anchors if m.start() > f0.start()]
    falsify_seg = block[f0.start() : after_f[0].start() if after_f else len(block)]
    confirms = [m for m in anchors if m.group(1).casefold().startswith("confirm")]
    if confirms:
        c0 = confirms[0]
        after_c = [m for m in anchors if m.start() > c0.start()]
        confirm_seg = block[c0.start() : after_c[0].start() if after_c else len(block)]
    else:
        confirm_seg = block[: f0.start()]
    return confirm_seg, falsify_seg


def _shared_bound_tokens(confirm_seg: str, falsify_seg: str) -> list[str]:
    """Normalized ``var = value`` pairs present in BOTH segments, rendered
    as sorted ``"var = value"`` strings (identity on the normalized pair,
    whitespace-insensitive)."""

    def _toks(seg: str) -> set[tuple[str, str]]:
        return {(m.group(1).casefold(), m.group(2)) for m in _BOUND_TOKEN_RE.finditer(seg)}

    return sorted(f"{var} = {val}" for var, val in _toks(confirm_seg) & _toks(falsify_seg))


def _c14_block_label(block: str) -> str:
    """Short human label for a hypothesis block: the first bold span that is
    not itself a branch anchor (e.g. ``**H4 (rollout).**``), else the first
    line truncated."""
    for m in _C14_BOLD_LABEL_RE.finditer(block):
        if not re.match(r"(?i)\s*(?:confirm|falsif)", m.group(1)):
            return f"**{m.group(1)}**"
    first_line = block.strip().splitlines()[0] if block.strip() else "(unnamed)"
    return first_line[:60]


def check_hypothesis_branch_coherence(plan: str, kind: str) -> CheckResult:
    """WARN-only, conditional: hypothesis confirm/falsify branch coherence.
    Two token-level offender predicates per anchor-bearing hypothesis
    block: (a) a jointly-satisfiable tendency-vs-state comparator pair on a
    shared bounded ``var = value`` token across the confirm/falsify
    segments ("decays toward ... by k = 32" confirm vs "stays above ...
    through k = 32" falsify — one above-but-declining curve satisfies
    both); (b) a vague layer-scope token ("mid/late layers", "most layers")
    with no pinned layer list/numeral in the same block. NEVER FAILs — a
    heuristic text check must not hard-block a legitimately-worded plan;
    joint satisfiability beyond these two token shapes stays with the
    Statistics critic (c8's form-only charter). Crisp state-vs-state pairs
    (``≤`` vs ``>``, win-count comparators — the #841 v12 / #810 v6 shapes)
    carry no tendency token and stay silent. Incident: #922 v2 H4 (caught
    only by the Codex statistics critic; the same defect class reached
    execution in #488 round 10)."""
    cid, name = "c14_hypothesis_branch_coherence", "hypothesis branch coherence"
    if kind not in ("experiment", "analysis"):
        return _skip(
            cid, name, "kind-exempt: hypothesis blocks are an experiment|analysis plan shape"
        )
    section = section_text_by_keywords(plan, ("hypothesis",))
    if section is None:
        return _skip(cid, name, "no hypothesis section detected")
    text = strip_fences(section)
    anchored: list[tuple[str, tuple[str, str]]] = []
    for block in _hypothesis_blocks(text):
        segments = _confirm_falsify_segments(block)
        if segments is not None:
            anchored.append((block, segments))
    if not anchored:
        return _skip(
            cid, name, "hypothesis section present but no **Confirm/**Falsify branch anchors"
        )
    offenders: list[str] = []
    for block, (confirm_seg, falsify_seg) in anchored:
        clauses: list[str] = []
        shared = _shared_bound_tokens(confirm_seg, falsify_seg)
        if shared:
            c_tend = _TENDENCY_RE.search(confirm_seg)
            f_state = _STATE_RE.search(falsify_seg)
            c_state = _STATE_RE.search(confirm_seg)
            f_tend = _TENDENCY_RE.search(falsify_seg)
            pair: tuple[str, str] | None = None
            if c_tend and f_state:
                pair = (c_tend.group(0), f_state.group(0))
            elif c_state and f_tend:
                pair = (c_state.group(0), f_tend.group(0))
            if pair:
                clauses.append(
                    f"(a) comparator-pair — confirm says '{pair[0]}' while falsify says "
                    f"'{pair[1]}' on shared token '{shared[0]}', jointly satisfiable by "
                    "one outcome"
                )
        vague = _VAGUE_SCOPE_RE.search(block)
        if vague and not _PINNED_SCOPE_RE.search(block):
            clauses.append(
                f"(b) vague-scope — '{vague.group(0)}' with no pinned layer list/numeral "
                "in the block"
            )
        if clauses:
            offenders.append(f"block '{_c14_block_label(block)}': " + "; ".join(clauses))
    if not offenders:
        return _pass(
            cid,
            name,
            f"{len(anchored)} hypothesis block(s) scanned; no c14 trigger detected "
            "(no jointly-satisfiable comparator pair, no unpinned vague-scope token)",
        )
    extra = f" (+{len(offenders) - 3} more)" if len(offenders) > 3 else ""
    detail = (
        "; ".join(offenders[:3])
        + extra
        + " — tighten the branch comparators (e.g. '≤ vs >') and/or pin the layer set; "
        "semantic verdict stays with the Statistics critic"
    )
    return _warn(cid, name, detail)


# ─── Check 15 — fail-loud acceptance claim backed by a committed test ──────

# Trigger anchor: an acceptance/success-criteria mention. Deliberately
# NARROWER than c8's _SUCCESS_RE — "decision rule|decision gate" is excluded
# because gates/failure-mode sections carry failure-MODE descriptions
# ("silently provisioned" risk rows), not acceptance claims (corpus probe on
# all 230 infra|batch plans, task #932).
_FAILLOUD_ANCHOR_RE = re.compile(r"(?i)acceptance criteri|success criteri")

# Claim vocabulary, scanned over the fence-stripped window below an anchor.
# Letter-lookarounds (not \b) around "loud" exclude "cloud"/"Cloudflare".
# The bare transitive "raises" is deliberately absent ("raises the
# concurrency cap" is a real infra acceptance sentence); the narrow raise
# forms + swallow/silent cover the genuine raise-claims in the corpus.
_FAILLOUD_CLAIM_RE = re.compile(
    r"(?i)fail[- ]?loud|fail[- ]?fast"
    r"|(?<![a-z])loud(?:ly)?(?![a-z])"
    r"|swallow"
    r"|silent"
    r"|warn(?:ing)?[- ]and[- ]continue"
    r"|except\s+(?:Exception|BaseException)|bare\s+except|try\s*/\s*except|except\s*:"
    r"|(?:must|shall|should)\s+raise\b"
    r"|raises?\s+(?:an?\s+)?[A-Z][A-Za-z]*(?:Error|Exception)\b|raises?\s+SystemExit"
    r"|non-?zero\s+exit|exits?\s+non-?zero"
)

# Committed-test evidence vocabulary. Letter-lookarounds so identifier-
# internal tokens match (test_length_mismatch_raises, test_no_silent_swallow).
# "exit code" is deliberately absent — "pytest ... exit code 0" is a generic
# success-path verification line and self-certified in the corpus probe.
_FAILLOUD_TEST_EVIDENCE_RE = re.compile(
    r"(?i)(?<![a-z])rais(?:e|es|ed|ing)(?![a-z])|swallow|silent"
    r"|fail[- ]?loud|fail[- ]?fast|(?<![a-z])loud(?![a-z])"
    r"|(?<![a-z])except(?![a-z])|systemexit|non-?zero\s+exit|exits?\s+non-?zero"
)

# Evidence-side exclusion: a run-book grep gate over a test file would
# otherwise self-certify (`grep -n 'except Exception' tests/test_foo.py`).
_FAILLOUD_GREP_LINE_RE = re.compile(r"(?i)\bgrep\b")

# Evidence route 2 (#1306): a quoted pytest.raises control with a named
# exception is intrinsically fail-loud test vocabulary — accepted WITHOUT a
# same-line test_ identifier (#1296: the negative-control literal sits on
# its own hard-wrapped line). Corpus-audited 2026-07-14: flips exactly the
# two #1296 incident plans out of 119 standing infra|batch WARNs.
_FAILLOUD_PYTEST_RAISES_RE = re.compile(r"pytest\.raises\(\s*[\w.]+")

# Evidence route 3 (#1306): a deliberate labeled pin line opens a FORWARD,
# paragraph-bounded scan for the test identifier — the labeled-line
# convention (c31 precedent: an unlabeled c15-style loose window
# false-satisfied all 9 of c31's incident plan versions, so the label is
# load-bearing), made wrap-tolerant for long test paths (#1296 v2's
# 105-char path forced the wrap). A blank line ends the paragraph; the
# scan is capped (incident paragraph = 5 lines; 8 = 5 + margin).
_FAILLOUD_PIN_LABEL_RE = re.compile(r"(?i)\bfail[- ]?loud (?:pin|acceptance)\b[^:\n]{0,40}:")
_FAILLOUD_PIN_SCAN_LINES = 8

# Anchor carriers that never bind: §0.0 TL;DR / §0 Plan Summary restate
# criteria as summary prose (same rationale as c8's _tldr_ranges exclusion).
_FAILLOUD_SUMMARY_HEAD_RE = re.compile(r"(?i)tl;dr|plan summary|^(?:§\s*)?0(?:\.0)?\b")

# Anchor carriers that never bind (2): Risks / Failure-Modes sections carry
# failure-MODE narration ("post_event raising ValueError is fail-loud", "will
# be caught ... fails loud"), not acceptance claims — the same rationale that
# excluded decision rule|gate from the anchor (#932 corpus probe, comment
# above _FAILLOUD_ANCHOR_RE). BOTH alternation branches are GROUPED under the
# start anchor + optional section numbering (a bare `|failure[- ]modes?`
# branch would match anywhere in the heading and silently exclude genuine
# acceptance sections), so an acceptance heading merely containing "risk" or
# "failure mode" mid-heading is NOT excluded. 11 of 16 historical noise fires
# were this class (#1291; founding incident #1275 v1).
_FAILLOUD_RISKS_HEAD_RE = re.compile(
    r"(?i)^(?:§\s*)?(?:\d+(?:\.\d+)*[.)]?\s*)?(?:risks?\b|failure[- ]modes?\b)"
)

_FAILLOUD_WINDOW_LINES = 30


def _failloud_claim_hits(plan: str) -> list[tuple[str, str]]:
    """(section heading, matched vocabulary) per acceptance/success anchor
    whose 30-line window carries a fail-loud claim. Anchors in fences, in
    §0/TL;DR/Plan-Summary regions, in Risks/Failure-Modes sections (risk rows
    narrate failure MODES, not acceptance claims — 11/16 historical noise
    fires, #1291, founding incident #1275 v1), or with an H1/preamble carrier
    are dropped (corpus-probe noise classes, tasks #932 + #1291). The claim
    window is built from the document-global fence mask (a window slice can
    no longer mis-parse when it starts inside a fence) with `grep`-bearing
    lines excluded LINE-SCOPED — a grep line narrates tooling semantics
    ("`grep -c` exits nonzero"), not the plan's own acceptance claim (the
    remaining 5/16 noise fires, #1275 v2); a real claim on any non-grep line
    in the window still triggers."""
    lines = plan.splitlines()
    mask = _fence_mask(lines)
    headings = _headings(plan)
    hits: list[tuple[str, str]] = []
    for i, line in enumerate(lines):
        if mask[i] or not _FAILLOUD_ANCHOR_RE.search(line):
            continue
        h = _innermost_section(headings, i)
        if h is None or h.level < 2 or _FAILLOUD_SUMMARY_HEAD_RE.search(h.text.strip()):
            continue
        if _FAILLOUD_RISKS_HEAD_RE.search(h.text.strip()):
            continue
        end = min(h.end, i + 1 + _FAILLOUD_WINDOW_LINES)
        window = "\n".join(
            ln
            for ln, fenced in zip(lines[i:end], mask[i:end], strict=True)
            if not fenced and not _FAILLOUD_GREP_LINE_RE.search(ln)
        )
        m = _FAILLOUD_CLAIM_RE.search(window)
        if m:
            hits.append((h.text, m.group(0)))
    return hits


def _failloud_test_evidence_lines(plan: str) -> list[str]:
    """RAW-plan lines naming a committed fail-loud-exercising test, three
    routes (#913/#1306): (1) a ``test_`` identifier co-located with
    fail-loud vocabulary on ONE line (the original scan); (2) a quoted
    ``pytest.raises(<Exception>`` control — intrinsically test + raise
    vocabulary, no identifier needed; (3) a ``Fail-loud pin:``-style
    labeled line whose contiguous paragraph names a ``test_`` identifier
    within ``_FAILLOUD_PIN_SCAN_LINES`` lines (wrap-tolerant labeled route;
    unlabeled ±k windows false-satisfy — 21-29 of 119 corpus WARNs at
    k=1-2 vs 2 genuine, measured 2026-07-14 — so the label is
    load-bearing, the c31 lesson). Grep-command lines never count on any
    route."""
    lines = plan.splitlines()
    out: list[str] = []
    for i, line in enumerate(lines):
        if _FAILLOUD_GREP_LINE_RE.search(line):
            continue
        if _TEST_IDENT_RE.search(line) and _FAILLOUD_TEST_EVIDENCE_RE.search(line):
            out.append(line.strip())
            continue
        if _FAILLOUD_PYTEST_RAISES_RE.search(line):
            out.append(line.strip())
            continue
        if _FAILLOUD_PIN_LABEL_RE.search(line):
            for j in range(i, min(len(lines), i + _FAILLOUD_PIN_SCAN_LINES)):
                if not lines[j].strip():
                    break
                if _FAILLOUD_GREP_LINE_RE.search(lines[j]):
                    continue
                if _TEST_IDENT_RE.search(lines[j]):
                    out.append(f"{line.strip()} … {lines[j].strip()}")
                    break
    return out


def check_failloud_test_coverage(plan: str, kind: str) -> CheckResult:
    """``kind: infra|batch`` plans whose acceptance/success criteria assert
    fail-loud / no-silent-swallow behavior must name a committed test pinning
    it — run-book grep gates verify the invariant once at review time, and a
    differently-worded re-swallow ships green past all committed tests
    (#913). WARN not FAIL: trigger and evidence are line heuristics; the
    Phase 2 critics adjudicate (Statistics lens item 14 owns the per-claim
    coverage judgment this check cannot make — the mechanical layer catches
    only the zero-fail-loud-test case, and PASSes a plan naming a fail-loud
    test for a different claim). Extending kind scope to ``analysis`` is a
    future calibration decision if an incident arises there (the corpus
    replay covered infra|batch only). Evidence routes 2-3 (#1306) — a
    quoted ``pytest.raises(<Exception>`` control and a labeled
    ``Fail-loud pin``-style paragraph scan — were corpus-audited
    2026-07-14 (exactly the two #1296 incident plans flip WARN→PASS out
    of 119 standing infra|batch WARNs; unlabeled ±1/±2-line windows were
    REJECTED at 21/29 false flips); accepted residual: a fenced
    bug-narration ``pytest.raises(...)`` quote or a stale/nonexistent
    test name in a labeled pin can false-satisfy — acceptable because the
    check is WARN-only at existence granularity and per-claim coverage
    stays with the Phase 2 critics."""
    cid, name = "c15_failloud_test_coverage", "fail-loud acceptance claim backed by a test"
    if kind not in ("infra", "batch"):
        return _skip(
            cid,
            name,
            "kind-exempt: the fail-loud acceptance-claim pattern is an infra|batch shape",
        )
    hits = _failloud_claim_hits(plan)
    if not hits:
        return _skip(cid, name, "no fail-loud claim in an acceptance/success-criteria window")
    if _standalone_na_declared(
        plan, r"(?:no fail[- ]?loud acceptance claim|fail[- ]?loud claim not test-backable)"
    ):
        return _pass(
            cid, name, "explicit N/A declared (incidental vocabulary or not test-backable)"
        )
    evidence = _failloud_test_evidence_lines(plan)
    if evidence:
        sec, tok = hits[0]
        return _pass(
            cid,
            name,
            f"fail-loud claim ({tok!r} in §{sec[:40]!r}) + fail-loud test named "
            f"({evidence[0][:80]!r})",
        )
    sec, tok = hits[0]
    return _warn(
        cid,
        name,
        f"acceptance/success criteria assert fail-loud behavior ({tok!r} in §{sec[:40]!r}) but "
        "no line names a committed test carrying fail-loud vocabulary (a `test_` identifier or "
        "tests/<file> path alongside raise/swallow/silent/except vocabulary; grep-gate lines do "
        "not count) — a run-book grep verifies the invariant once at review time, and a "
        "differently-worded re-swallow ships green past all committed tests (#913). Name the "
        "pinning test and its raise/swallow/silent vocabulary on ONE unwrapped line "
        "(hard-wrapped mentions do not count), quote the `pytest.raises` negative control "
        "with its exception class, add a `Fail-loud pin` labeled line (the label, a colon, "
        "then the test path), or declare `N/A — no fail-loud acceptance claim` / "
        "`N/A — fail-loud claim not test-backable` — each on its own line, unwrapped "
        "(no backticks/quotes)",
    )


# ─── Check 16 — re-extracted reference vs committed headline (WARN-only) ───

# Trigger half (a): a NON-NEGATED re-extraction/regeneration token on a
# NON-fenced line, with reference/parity/committed vocabulary nearby.
# Two branches, calibrated on the 2026-07-03 historical-corpus sweep:
#   - `re-?extract`: vocabulary within ±_C16_WINDOW_LINES RAW lines
#     (hard-wrapped prose splits "re-extracted\nreferences"; #811 v3's §5
#     rows carry "(reference, re-extracted)" on one line);
#   - `re-?generat`: SAME-line adjacency only (the plan §4.5 pre-authorized
#     demotion — window-scoped re-generat swept in doc/data-regeneration
#     noise: #491/#537/#542/#558/#597/#685/#763/#825 fired on regeneration
#     mentions with reference vocab merely nearby).
# The fixed-width negation lookbehinds drop ASSERTED-NEGATIVE mentions
# ("NOT regenerated", "NO re-extraction of r_B" — #559/#561/#810-v1-3 noise
# class): a plan stating it does NOT re-extract is not a trigger.
# `\bre-?extract` does not match "pre-extraction" (no word boundary inside
# "pre").
_C16_NEG_GUARD = r"(?<!\bno )(?<!\bnot )(?<!\bnever )(?<!\bwithout )"
_C16_EXTRACT_RE = re.compile(rf"(?i){_C16_NEG_GUARD}\bre-?extract\w*")
_C16_REGEN_RE = re.compile(rf"(?i){_C16_NEG_GUARD}\bre-?generat\w*")
_C16_REF_RE = re.compile(
    r"(?i)\breferences?\b|\breference[- ]arms?\b|\bparity\b"
    r"|\bcommitted (?:cells?|v\d)|prior[- ]headline"
)
_C16_WINDOW_LINES = 3

# Trigger half (b): the plan reads as a same-issue follow-up / amendment
# folding into an existing clean-result. Document-global, fence-stripped;
# (?s) so the wrapped "folds into THIS\nissue's clean-result body" shape
# (#811 v3:87-89) is caught. Bare "follow-up round" is deliberately absent
# (709 occ / 216 files in the 2026-07-03 corpus probe — plans cite the
# follow-up machinery prospectively).
_C16_FOLD_RE = re.compile(
    r"(?is)same-issue follow-?up|amendment to (?:the|this|a)\b"
    r"|epm:followup-scope|followups?_running"
    r"|folds? into .{0,80}?clean-result"
)

# Satisfaction: an explicit sentence distinguishing same-pass comparator
# values from prior committed headline values. Three shapes:
#   S1 — the term of art itself ("comparator" REQUIRED: #811 v3:189
#        "re-extracting the references in the SAME pass" must not satisfy);
#   S2 — committed-headline noun phrase + a retention verb within one
#        sentence. Gaps exclude '.' and ';' so v3:574 "committed cells only
#        via R resampling." and v3:499 "(committed; prior rounds' artifacts
#        untouched)" cannot satisfy — the sentence stop / path dots block;
#   S3 — an explicit negated-replacement clause naming the headline
#        (v3:270 "layout replaces grouped bars" carries no negation).
# "replication-stability" vocabulary alone deliberately does NOT satisfy —
# the incident plan carried it (v3:347, :434).
_C16_SAMEPASS_RE = re.compile(r"(?i)same[- ]pass comparators?")
_C16_DISTINCTION_RE = re.compile(
    r"(?is)(?:committed|prior|standing|already[- ]adjudicated)"
    r"[^.;]{0,40}?\b(?:headline|cells?|values?|verdicts?|calls?|evidence)"
    r"[^.;]{0,120}?"
    r"(?:remains?|retain\w*|stays?|stands?|kept|keeps?|unchanged|untouched"
    r"|(?:is |are )?not (?:silently )?replaced?|never (?:silently )?replaced?)"
)
_C16_NONREPLACE_RE = re.compile(
    r"(?is)(?:never|not|no)\s+(?:a\s+)?(?:silent(?:ly)?\s+)?"
    r"(?:headline[- ])?replac\w*[^.;]{0,80}?(?:headline|committed)"
    r"|(?:never|not)\s+(?:silently\s+)?replac\w*[^.;]{0,60}?headline"
)


def check_reference_headline_distinction(plan: str, kind: str) -> CheckResult:
    """A follow-up plan that re-extracts prior-headline REFERENCE arms AND
    folds into an existing clean-result must explicitly distinguish
    "same-pass comparator" values from "prior committed headline" values —
    a reference flip is replication-stability evidence, never an
    unannounced headline replacement (#811 v3 §6; task #937). WARN not
    FAIL: both trigger halves and the satisfaction shapes are text
    heuristics; the Statistics critic adjudicates the semantic question
    (does the plan's adjudication story actually preserve the committed
    cells) — this gate surfaces, never adjudicates."""
    cid = "c16_reference_headline_distinction"
    name = "re-extracted reference vs committed headline"
    if kind not in ("experiment", "analysis"):
        return _skip(
            cid,
            name,
            "kind-exempt: clean-result folding is an experiment|analysis plan shape",
        )
    # re-extract: ±3-line windows; re-generat: same-line only (radius 0).
    windows = _trigger_windows(plan, _C16_EXTRACT_RE, _C16_WINDOW_LINES)
    windows += _trigger_windows(plan, _C16_REGEN_RE, 0)
    if not any(_C16_REF_RE.search(w) for w in windows):
        return _skip(cid, name, "no re-extraction of reference arms detected")
    text = strip_fences(plan)
    if not _C16_FOLD_RE.search(text):
        return _skip(
            cid,
            name,
            "re-extraction vocabulary present but the plan does not read as a "
            "same-issue follow-up folding into an existing clean-result",
        )
    if _standalone_na_declared(plan, r"no re-?extracted reference arms"):
        return _pass(cid, name, "explicit N/A declared (no re-extracted reference arms)")
    if (
        _C16_SAMEPASS_RE.search(text)
        or _C16_DISTINCTION_RE.search(text)
        or _C16_NONREPLACE_RE.search(text)
    ):
        return _pass(
            cid,
            name,
            "distinguishing sentence present (same-pass comparator / committed-headline retention)",
        )
    return _warn(
        cid,
        name,
        "plan re-extracts prior-headline reference arms AND folds into an existing "
        "clean-result, but no sentence distinguishes same-pass-comparator values from "
        "the prior committed headline values — state which values adjudicate this "
        "round's NEW comparison vs which are still the committed headline, and that a "
        "flipped reference CALL is reported as replication-stability evidence rather "
        "than replacing the headline (#811 v3 §6 incident; the kept-cells-"
        "stay-evidence rule), or declare `N/A — no re-extracted reference arms` "
        "on its own line, unwrapped (no backticks/quotes)",
    )


# ─── Check 17 — falsification-branch causal-claim scope (WARN-only) ────────

# Offender vocabulary: wording that asserts a causal mechanism as
# DEMONSTRATED inside a registered branch. Tier-1 only (corpus-calibrated,
# task #946 §6): retrospective attribution ("really was/were", "must have
# been"), content-carrying claims, story-kill idioms, takeaway rewrites,
# and explicit establish/prove/demonstrate-that. Deliberately EXCLUDED as
# accepted false negatives (prefer false negatives — the c14 charter):
# present-tense "really is/does" (5-6 of 8 corpus hits were legitimate),
# "rules out" (#605 uses it as a CI equivalence bound), bare "must be"
# (deontic), and bare mechanism-noun falsify labels ("**Falsified
# (integration):**" — not regex-separable from "(dependence)" labels).
_C17_OFFENDER_RE = re.compile(
    r"(?i)"
    r"\breally\s+(?:was|were)\b"
    r"|\bmust\s+have\s+been\b"
    r"|\bcarr(?:y|ies|ied|ying)\b[^.\n]{0,50}\bcontent\b"
    r"|\b(?:story|account|hypothesis|explanation|interpretation)\s+(?:dies|is\s+dead)\b"
    r"|\brewrit(?:es?|ing)\s+the\b[^.\n]{0,60}\b(?:interpretation|takeaway|headline)\b"
    r"|\b(?:establish(?:es|ed)?|prov(?:es|ed)|demonstrat(?:es|ed))\s+that\b"
)
# Exculpation vocabulary: an alternative-naming / hedge token in the SAME
# block (hyp surface) or SAME bullet (TL;DR surface) silences the offender.
# Over-breadth here only creates false negatives, which the charter prefers.
# Calibrated on the #810 v13→v14 fix wording plus corpus hits (#563 "scope
# caveat", #611/#621 "artifact", #841 "gets real support").
_C17_EXCULP_RE = re.compile(
    r"(?i)"
    r"\bconsistent\s+with\b|\bcompatible\s+with\b"
    r"|\buniquely\s+diagnostic\b"
    r"|\bcannot\s+(?:distinguish|rule\s+out)\b"
    r"|\bdoes\s+not\s+distinguish\b|\bdoesn'?t\s+distinguish\b"
    r"|\balternative\b|\bconfound\w*\b|\bartifact\w*\b|\bcaveats?\b"
    r"|\bsimpler\s+explanations?\b|\bother\s+explanations?\b"
    r"|\bOOD\b|\boff-?distribution\b|\bout-?of-?distribution\b"
    r"|\bremains?\s+live\b|\bdegradation\b|\bendpoint\b"
    r"|\bpending\b|\bdisambiguat\w*\b|\bunder-?determin\w*\b|\bambiguous\b"
    r"|\bwould\s+not\s+(?:prove|establish|demonstrate)\b"
    r"|\b(?:gets?|gains?|lends?|earns?)\s+(?:real\s+)?support\b"
)
# The §0.0 registered plain-English falsification branch ("**What would
# change my mind:**" / "…mind.**" — both corpus punctuation shapes).
_C17_MIND_RE = re.compile(r"(?i)\*\*\s*what would change my mind")


def _c17_mind_segments(plan: str) -> list[str]:
    """The fence-stripped `**What would change my mind**` bullet(s), each
    with its continuation lines (up to the next top-level list item or
    heading) — the §0.0 registered falsification branch surface."""
    lines = strip_fences(plan).splitlines()
    segs: list[str] = []
    i = 0
    while i < len(lines):
        if _C17_MIND_RE.search(lines[i]):
            seg = [lines[i]]
            j = i + 1
            while (
                j < len(lines)
                and not _C14_LIST_ITEM_RE.match(lines[j])
                and not _HEADING_RE.match(lines[j].strip())
            ):
                seg.append(lines[j])
                j += 1
            segs.append("\n".join(seg))
            i = j
        else:
            i += 1
    return segs


def check_causal_claim_scope(plan: str, kind: str) -> CheckResult:
    """WARN-only, conditional: a registered falsification (or confirm)
    branch must not word its outcome as a DEMONSTRATED causal mechanism
    when the same block never names the undistinguished alternative.
    Surfaces scanned: (i) confirm/falsify segments of anchored hypothesis
    blocks (c14's parsers, reused); (ii) the §0.0 `**What would change my
    mind:**` bullet(s). An offender token is silenced by an exculpation
    token in the same block/bullet. NEVER FAILs — a heuristic vocabulary
    check must not hard-block a legitimately-worded plan; whether the
    diagnostics actually distinguish the mechanism stays with the
    Methodology/Statistics critics. The §6 corpus noise floor (2/195
    newest-per-task) is IN-SAMPLE — the offender/exculpation vocabulary
    was tuned on the same corpus it was measured on — so any future
    FAIL-promotion needs held-out / prospective validation first.
    Incident: #810 plan v13 ("they really were carrying answer content,
    the echo story dies") — three reviewers independently required the
    v14 scope-down ("consistent with integration but not uniquely
    diagnostic; OOD ... remains live"); task #946."""
    cid, name = "c17_causal_branch_scope", "falsification-branch causal-claim scope"
    if kind not in ("experiment", "analysis"):
        return _skip(
            cid,
            name,
            "kind-exempt: registered falsification branches are an experiment|analysis plan shape",
        )
    anchored: list[tuple[str, tuple[str, str]]] = []
    section = section_text_by_keywords(plan, ("hypothesis",))
    if section is not None:
        for block in _hypothesis_blocks(strip_fences(section)):
            segments = _confirm_falsify_segments(block)
            if segments is not None:
                anchored.append((block, segments))
    mind_segs = _c17_mind_segments(plan)
    if not anchored and not mind_segs:
        return _skip(
            cid,
            name,
            "no registered falsification-branch surface (no **Confirm/**Falsify "
            "hypothesis anchors, no **What would change my mind** bullet)",
        )
    offenders: list[str] = []
    for block, (confirm_seg, falsify_seg) in anchored:
        if _C17_EXCULP_RE.search(block):
            continue
        for branch, seg in (("falsify", falsify_seg), ("confirm", confirm_seg)):
            m = _C17_OFFENDER_RE.search(seg)
            if m:
                offenders.append(
                    f"hypothesis block {_c14_block_label(block)} ({branch} segment): "
                    f"claim token '{m.group(0)}'"
                )
                break  # one offender per block is enough for the detail
    for seg in mind_segs:
        if _C17_EXCULP_RE.search(seg):
            continue
        m = _C17_OFFENDER_RE.search(seg)
        if m:
            offenders.append(f"'What would change my mind' bullet: claim token '{m.group(0)}'")
    if not offenders:
        return _pass(
            cid,
            name,
            f"{len(anchored)} hypothesis block(s) + {len(mind_segs)} TL;DR bullet(s) "
            "scanned; no unqualified demonstrated-mechanism claim token",
        )
    extra = f" (+{len(offenders) - 3} more)" if len(offenders) > 3 else ""
    return _warn(
        cid,
        name,
        "; ".join(offenders[:3])
        + extra
        + " — the branch asserts a causal account its diagnostics may not uniquely "
        "distinguish (#810 v13 incident): name the undistinguished alternative in "
        "the same block (e.g. 'consistent with <mechanism> but not uniquely "
        "diagnostic; <alternative> remains live') or scope the wording to the "
        "measured quantity; semantic verdict stays with the critics",
    )


# ─── Check 18 — paired-contrast per-arm source coverage ────────────────────

# Registration-family sections (H2+ ONLY — a doc-spanning H1 title match
# would make the section constraint vacuous; both #810 registrations sit
# under H2 sections): c13's success/kill/evaluation families PLUS
# hypothesis + nulls + statistic.
_C18_SECTION_RE = re.compile(
    r"(?i)hypothes|success criteri|acceptance criteri|decision rule|decision gate"
    r"|kill[- ]criteri|abort criteri|stop criteri|\bevaluation\b|\bnulls?\b|statistic"
)
_C18_PAIRED_RE = re.compile(r"(?i)\bpaired\b")
_C18_REGIST_RE = re.compile(r"(?i)\bregist")  # registered / registration / registers
_C18_PAIRCOUNT_RE = re.compile(r"(?i)\b\d[\d,]*\s+(?:pre-named\s+)?pairs\b")
# D1: a row-coverage declaration; evidence on the same line or within the
# next _C18_DECL_WINDOW_LINES physical lines (fenced lines excluded).
_C18_COVERAGE_RE = re.compile(r"(?i)\brow[- ]coverage\b")
# #1086 widening: suffixed tensor-store dirs (`analysis_tensors_nonemit/` —
# the `\w*` suffix arm) and canonical `issueN_<slug>/…` HF data-repo
# prefixes (the Upload Policy destination shape) are artifact evidence. The
# trailing `\S*` on the `analysis_tensors\w*/\S*` alternative — not `\S+` —
# is DELIBERATE: a bare store-dir token ending at the slash (backticked or
# line-final `analysis_tensors_nonemit/`) is complete artifact evidence
# with nothing after the slash. An `issueN…/` PATH token is affirmative
# artifact evidence, orthogonal to the `_C18_ISSUE_REF_RE` citation guard
# (the literal `#\d{2,}` form), which is byte-unchanged. Accepted
# fail-UNSAFE residual, DISCLOSED: a SIBLING issue's `issueN…/` store path
# on the declaration line counts as D1 artifact evidence — whether the
# named store truly contains THIS plan's rows stays with the fact-checker
# (no guard, no negative fixture: a negative would require a citation
# guard #1086 deliberately does not add).
_C18_ARTIFACT_RE = re.compile(
    r"(?i)\S+\.(?:pt|pth|json|jsonl|npz|npy|safetensors|csv|parquet|arrow)\b"
    r"|\beval_results/\S+|\banalysis_tensors\w*/\S*|\braw_completions/\S+"
    r"|\bissue\d{2,}[\w.-]*/\S+"
)
# v2 (MF-B): the bare `this run` alternative is REMOVED — only the
# arms-generated construction or an explicit `by construction` counts as
# by-construction evidence ("Row-coverage: deferred to a later revision of
# this run's analysis" must FAIL).
_C18_BYCONSTRUCTION_RE = re.compile(
    r"(?i)both arms .{0,60}\b(?:generated|produced|computed|fit(?:ted)?|emitted)\b"
    r"|\bby construction\b"
)
# #1086 (v2): the check's own remedy text ("state that the plan's own fits
# produce every registered row on each arm", the FAIL detail below) was
# unmatchable by _C18_BYCONSTRUCTION_RE — a planner implementing the bounce
# verbatim still FAILed (#833 v8). Accept that form via a SEPARATE
# alternative deliberately narrower than the remedy prose: (i) affirmative
# produce-verb + "every registered", (ii) arm vocabulary within 80 chars
# after the match (each/both/per arm[s]), (iii) NO negation/deferral token
# in the local span around the match — "does not yet produce every
# registered row on each arm" and "will produce every registered row …
# once implemented" are explicit NON-declarations and must keep FAILing
# (the MF-B deferral class). _C18_BYCONSTRUCTION_RE itself stays
# byte-unchanged so no historical PASS can flip.
_C18_PRODUCES_REGISTERED_RE = re.compile(
    r"(?i)\b(?:produces?|generates?|computes?|emits?|yields?)\s+every\s+registered\b"
    r"(?=.{0,80}\b(?:each|both|per)\s+arms?\b)"
)
# #1099: the guard covers the n't contraction family (word chars + n +
# straight-or-curly apostrophe + t) — the prior bare `n't` alternative was
# DEAD CODE (a word boundary never matches at the word-internal s->n
# transition inside "doesn't"; probe-verified) — plus cannot /
# fail(s|ed) to / until (all common in the plan corpus — thousands of
# occurrences for cannot/doesn't; counts hedged deliberately, they age).
# Curly apostrophe included (a handful of corpus plan files carry it).
# Strictly widening the DISQUALIFIER = strictly narrowing the
# affirmative satisfier — fail-safe by construction. Accepted residual
# (disclosed, #1099): "no longer" / "except" / "unless" / "rather than" /
# gerund "failing to" still evade this guard — outside the Goal-named
# set; the Phase-2 critic ensemble is the semantic backstop (same
# fail-unsafe residual class as c12's sibling-quote disclosure).
_C18_NEG_DEFER_RE = re.compile(
    r"(?i)\b(?:not|\w*n[’']t|cannot|never|without|fail(?:s|ed)?\s+to|until"  # noqa: RUF001
    r"|will|would|shall|should|may|might|could"
    r"|once|pending|deferred|later|TBD|to\s+be)\b"
)


def _c18_affirmative_produces_hit(line: str) -> bool:
    """The v2 remedy-text alternative: affirmative produce-verb + 'every
    registered' + arm vocabulary, with negation/deferral tokens disqualifying
    in a local span (48 chars before the match start, 80 after its end).
    Scoped to THIS alternative only — the legacy _C18_BYCONSTRUCTION_RE
    alternatives keep their behavior byte-for-byte."""
    m = _C18_PRODUCES_REGISTERED_RE.search(line)
    if not m:
        return False
    span = line[max(0, m.start() - 48) : m.end() + 80]
    return not _C18_NEG_DEFER_RE.search(span)


# D2 (MF-A): a subset expression AND word-bounded row/pair vocabulary AND
# coverage/source-key vocabulary must co-occur on the candidate line.
# Word-bounding kills the 608 v2:164 false-satisfier ("pair" inside
# "paired" no longer matches); the coverage-vocab conjunct excludes
# incidental subset prose; the #810 v15 declaration carries standalone
# row/pairs tokens + coverage/source/keys/assert (replay-verified).
_C18_SUBSET_RE = re.compile(r"(?i)⊆|\bissubset\b|\bis a subset of\b")
_C18_ROWPAIR_RE = re.compile(r"(?i)\b(?:pairs?|rows?)\b")
_C18_COVERAGE_VOCAB_RE = re.compile(r"(?i)coverage|\bsources?\b|\bkeys?\b|\bassert")
# Candidate-line rejection guards (BOTH satisfier families):
# (a) paste fingerprint — the c18 FAIL detail carries this literal, so a
#     verbatim-pasted bounce text can never self-satisfy;
# (b) cross-issue citation token — a line QUOTING another issue's driver
#     assert as a worked example is a citation, not a declaration (an
#     honest declaration describes THIS plan's inputs; recovery for a
#     legitimate collision: move the citation off the declaration line).
_C18_PASTE_FINGERPRINT = "#810 v13 class"
_C18_ISSUE_REF_RE = re.compile(r"#\d{2,}")
_C18_DECL_WINDOW_LINES = 3
# Trigger-side spurious-line guard (§3.4 calibration tuning): a FIGURES-
# enumeration line ("**Figures (over-produce):** ... paired cells; ...
# registered rows visually distinguished") lists plots, it registers no
# statistic — the one spurious-trigger class the exhaustive FAIL audit
# surfaced (7 corpus files: #537 v4-v6, #931 v1-v4). Scoped by LINE SHAPE
# (a leading figures label), never by content elsewhere on the line; a
# real registration line never opens with a figures label, so the guard
# under-triggers safe (SKIP; critics review).
_C18_FIGURES_LINE_RE = re.compile(r"(?i)^\W{0,8}figures?\b")

# Known accepted mis-triggers (mirroring the c13 §4.5 precedent). Under-
# triggers that fail SAFE (SKIP — the plan still reaches the fact-checker +
# critic ensemble): (a) a paired registration line without `regist` / pair-
# count vocabulary; (b) a registration under a heading outside the H2+
# section family; (c) a hard-wrapped registration (`paired` and `regist` on
# different lines). Over-trigger that fails LOUD (bounce, escapable): (d) a
# Hypothesis-section line merely RECAPPING a sibling's registered paired
# statistic — remedied by the standalone N/A line. Fail-UNSAFE residuals,
# accepted and DISCLOSED: (e) a D1/D2-shaped declaration that doesn't
# actually cover the registered rows — including a ONE-ARM declaration (the
# #810 v15 exemplar itself is full-side-only; both-arm truth stays with the
# fact-checker; disposition pinned by fixture); (f) a NON-verbatim
# paraphrase of the bounce text that reconstructs a satisfying shape while
# dropping the fingerprint — beyond mechanical defense, same residual class
# as a dishonest c13 N/A line; (g) a wrapped/reformatted paste that
# separates the fingerprint from the row-coverage phrase across lines — the
# line-local guard misses it; the D1 evidence requirement (artifact token /
# arms-generated phrase / #1086's affirmative produces-registered form)
# still has to be met by the surviving fragment. NOTE (#1086): the remedy
# text's own "produce every registered row on each arm" clause is now a
# satisfier BY DESIGN (the remedy-vs-satisfier inconsistency was the bug),
# so a wrapped paste landing that clause on a citation-free Row-coverage
# line self-satisfies — a widened, DISCLOSED instance of this same
# residual class.


def _c18_registered_paired_lines(plan: str) -> list[str]:
    """Non-fenced lines inside a registration-family H2+ section carrying
    ``paired`` plus registration vocabulary OR an enumerated pair count on
    the SAME line (#810 v13:33 'Registered per-row statistic: paired ...
    (7 pairs ...' and v13:103 'Nulls (registration) ... paired bootstrap CI
    (... 9 pairs are pre-named' both match). Level-1 headings are EXCLUDED
    from the section match (a title match spans the whole doc). Under-
    trigger fails safe (SKIP; critics review)."""
    lines = plan.splitlines()
    mask = _fence_mask(lines)
    headings = _headings(plan)
    hits: list[str] = []
    for i, (line, fenced) in enumerate(zip(lines, mask, strict=True)):
        if fenced or not _C18_PAIRED_RE.search(line):
            continue
        if not (_C18_REGIST_RE.search(line) or _C18_PAIRCOUNT_RE.search(line)):
            continue
        if _C18_FIGURES_LINE_RE.match(line.strip()):
            continue
        if not any(
            h.line <= i < h.end and h.level >= 2 and _C18_SECTION_RE.search(h.text)
            for h in headings
        ):
            continue
        hits.append(line.strip())
    return hits


def _c18_candidate_ok(line: str) -> bool:
    """Rejection guards shared by D1 and D2 candidate lines: the paste
    fingerprint and cross-issue citation tokens disqualify a line from
    satisfying the check (bounce-paste + quoted-sibling-example vectors)."""
    return _C18_PASTE_FINGERPRINT not in line and not _C18_ISSUE_REF_RE.search(line)


def _c18_coverage_declarations(plan: str) -> list[str]:
    """Lines satisfying D1 (row-coverage vocab + source evidence — an
    artifact token or an arms-generated phrase — on the same line or within
    the next _C18_DECL_WINDOW_LINES physical lines, fenced lines excluded)
    or D2 (subset expression + word-bounded row/pair vocab +
    coverage/source-key vocab, same line). Candidate lines failing
    ``_c18_candidate_ok`` are rejected."""
    lines = plan.splitlines()
    mask = _fence_mask(lines)
    out: list[str] = []
    for i, (line, fenced) in enumerate(zip(lines, mask, strict=True)):
        if fenced or not _c18_candidate_ok(line):
            continue
        if (
            _C18_SUBSET_RE.search(line)
            and _C18_ROWPAIR_RE.search(line)
            and _C18_COVERAGE_VOCAB_RE.search(line)
        ):
            out.append(line.strip())
            continue
        if _C18_COVERAGE_RE.search(line):
            window = [line] + [
                lines[j]
                for j in range(i + 1, min(i + 1 + _C18_DECL_WINDOW_LINES, len(lines)))
                if not mask[j]
            ]
            if any(
                _C18_ARTIFACT_RE.search(w)
                or _C18_BYCONSTRUCTION_RE.search(w)
                or _c18_affirmative_produces_hit(w)
                for w in window
            ):
                out.append(line.strip())
    return out


def _c18_na_escape_declared(plan: str) -> bool:
    """Standalone ``N/A — no paired contrast`` escape (see
    ``_standalone_na_declared`` for the anti-paste rationale)."""
    return _standalone_na_declared(plan, r"no paired contrast")


def check_paired_contrast_source_coverage(plan: str, kind: str) -> CheckResult:
    """A registered paired contrast (a hypothesis/evaluation/success-section
    line registering a paired statistic over enumerable rows/pairs) must
    DECLARE a per-context data source covering the registered rows on both
    arms (D1 row-coverage line / D2 coverage-labeled subset-assert /
    standalone N/A). Surface check only — pack contents stay with the
    fact-checker. FAIL (experiment) / WARN (analysis) / SKIP otherwise;
    the standalone ``N/A — no paired contrast`` escape is honored (SKIP
    path) only when no paired contrast is detected — when the escape
    co-occurs with a detected registration the check WARNs instead of
    PASSing, so the escape can never mask row-coverage verification of a
    present registration (#1258, the #1223 c20 rule).
    Incident: #810 v13 (9-row paired bootstrap; the named full-side pack
    lacked im_end/turn_nl; 4 independent reviewer catches)."""
    cid, name = "c18_paired_contrast_source_coverage", "paired-contrast per-arm source coverage"
    if kind not in ("experiment", "analysis"):
        return _skip(
            cid,
            name,
            "kind-exempt: registered paired contrasts are an experiment|analysis plan shape",
        )
    triggers = _c18_registered_paired_lines(plan)
    if not triggers:
        return _skip(cid, name, "no registered paired contrast detected")
    if _c18_na_escape_declared(plan):
        # #1258 (the #1223 c20 port): reachable ONLY when a registered paired
        # contrast WAS detected (the no-trigger case SKIPs above) — a PASS
        # here masks the row-coverage verification c18 exists to run (the
        # #810 v13 class). WARN, not FAIL: the trigger harvest may be a false
        # positive on quoted guidance; reviewers adjudicate.
        return _warn(
            cid,
            name,
            "the standalone `N/A — no paired contrast` escape co-occurs with "
            f"{len(triggers)} registered paired-contrast line(s) (first: "
            f"{triggers[0][:90]!r}) — the escape is reserved for contrast-free "
            "plans and would mask row-coverage verification of the detected "
            "registration (#1258, the #1223 c20 rule); remove the N/A line and "
            "declare Row-coverage (the registration is then verified), or "
            "fence/remove the registration-shaped prose the detector matched "
            "if it is quoted guidance rather than this plan's own registration",
        )
    decls = _c18_coverage_declarations(plan)
    if decls:
        return _pass(
            cid,
            name,
            f'row-coverage declaration found ("{decls[0][:90]}") — declaration surface '
            "only; whether the named sources truly contain every registered row on both "
            "arms stays with the fact-checker",
        )
    detail = (
        f'plan registers a paired contrast ("{triggers[0][:90]}") with no per-arm '
        "row-coverage declaration — a registered pair row absent from a named side makes "
        "the registered criterion unsatisfiable from the named inputs (the #810 v13 class: "
        "2 of 9 rows missing from the named full side). Remedy: add ONE non-fenced prose "
        "line (not inside a code fence) starting 'Row-coverage:' naming, for BOTH arms, "
        "which per-context store/file supplies every registered row (or stating that the "
        "plan's own fits produce every registered row on each arm), or state the driver "
        "assert that set-checks the registered rows against the named sources' keys on a "
        "non-fenced line, or declare 'N/A — no paired contrast' on its own line, unwrapped "
        "(no backticks/quotes); keep the declaration line free of cross-issue citations"
    )
    if kind == "analysis":
        return _warn(cid, name, detail + " (analysis kind-degrade: WARN, not FAIL)")
    return _fail(cid, name, detail)


# ─── Check 19 — OOD generalization folds (WARN-only, conditional) ──────────

# Trigger = a fold token SOLO (any cross-validation mention makes "is the
# fold group-level?" the right question), OR the WEAK token "held-out"
# conjoined with a predictor-statistic token. Bare "held-out" alone is an
# eval-split adjective (GOOD_PLAN: "40 held-out prompts") and must not fire;
# bare "predict(s)" is hypothesis prose and is deliberately excluded.
_C19_SOLO_FOLD_RE = re.compile(
    r"(?i)\bcross[- ]?validat\w*|\bLOO\b|\bLOCO\b|\bLOOCV\b"
    r"|\bleave[- ]one[- ][\w-]*out\b|\bk[- ]fold\b"
)
_C19_HELDOUT_RE = re.compile(r"(?i)(?<!\bno )(?<!\bnot )\bheld[- ]out\b")
_C19_PREDSTAT_RE = re.compile(
    r"(?i)\bR\^?2\b|R²|\breconstruction\b|\bread[- ]?outs?\b"
    r"|\bpredict(?:or|ive|ion)s?\b|\bregress\w*|\bridge\b"
    r"|\b(?:probe|decod\w*)\s+accurac\w*"
)
# Group-level evidence: leave-one-<UNIT>-out where UNIT is not a pointwise
# sample unit (#810's offender fold was leave-one-CONTEXT-out — pointwise).
_C19_LOO_UNIT_RE = re.compile(r"(?i)\bleave[- ]one[- ]([\w-]+?)[- ]out\b")
_C19_POINTWISE_UNITS = frozenset(
    {
        "context",
        "point",
        "sample",
        "row",
        "item",
        "question",
        "prompt",
        "cell",
        "completion",
        "example",
        "datapoint",
        "datum",
        "observation",
        "pair",
        "x",
    }
)


def _c19_pointwise_unit(unit: str) -> bool:
    """A captured leave-one-<unit>-out unit is pointwise when its EXACT form
    OR its hyphen-split SUFFIX segment is blocklisted — hyphenated variants
    (``data-point``) must not self-certify as group evidence (reconciler
    Must-Fix, round 1). ``prompt-family`` stays a group unit (suffix
    ``family`` is not blocklisted)."""
    u = unit.lower()
    return u in _C19_POINTWISE_UNITS or u.split("-")[-1] in _C19_POINTWISE_UNITS


_C19_GROUPFOLD_RE = re.compile(
    r"(?i)\bLOFO\b|group[- ]level (?:held[- ]out )?fold"
    r"|held[- ]out (?:group|famil\w*|genre|persona|corpus)"
    r"|(?:corpus|genre|domain|family)[- ]transfer\b|\btransfer arm\b"
)
# Negation-guarded: `non-iid` / `not iid` concedes group structure and must
# NOT satisfy the iid PASS tier (round-1 convergent critic concern).
_C19_IID_RE = re.compile(r"(?i)(?<!non[- ])(?<!\bnot )\b(?:iid\b|i\.i\.d\b)")


def check_ood_folds(plan: str, kind: str) -> CheckResult:
    """WARN-only, conditional: a held-out predictive DV (reconstruction R²,
    read-out rho, predictor accuracy) over group-structured samples must
    register a GROUP-level fold (LOFO / corpus transfer), declare
    ``N/A — no held-out predictive DV``, or argue a genuinely iid sample
    (.claude/rules/ood-generalization-folds.md; planner §6 Required block).
    NEVER FAILs — the trigger is a vocabulary heuristic; whether the named
    fold is actually group-level for this sample stays with the Statistics
    critic (lens item 13). Incident #810: the pointwise-LOCO headline
    reordered under leave-one-FAMILY-out and the read-out collapsed
    rho 0.909 → 0.285."""
    cid, name = "c19_ood_folds", "OOD generalization folds (held-out predictive DV)"
    if kind not in ("experiment", "analysis"):
        return _skip(cid, name, "kind-exempt: infra|batch|survey plans have no predictive DV")
    if _standalone_na_declared(plan, r"no held-?out predictive DV"):
        return _pass(cid, name, "explicit N/A declared (no held-out predictive DV)")
    text = strip_fences(plan)
    solo = _C19_SOLO_FOLD_RE.search(text)
    conj = _C19_HELDOUT_RE.search(text) and _C19_PREDSTAT_RE.search(text)
    if not (solo or conj):
        return _skip(
            cid,
            name,
            "no held-out predictive-DV vocabulary (no fold token; no held-out + "
            "predictor/R²/read-out co-occurrence)",
        )
    group_units = [
        m.group(1) for m in _C19_LOO_UNIT_RE.finditer(text) if not _c19_pointwise_unit(m.group(1))
    ]
    if group_units or _C19_GROUPFOLD_RE.search(text):
        return _pass(
            cid,
            name,
            "group-level fold vocabulary present"
            + (f" (leave-one-{group_units[0]}-out)" if group_units else "")
            + " — fold validity + per-headline fold labeling stay critic-owned",
        )
    if _C19_IID_RE.search(text):
        return _pass(
            cid,
            name,
            "iid-sample argument present (the only pointwise-only exemption) — whether "
            "the sample is genuinely iid stays critic-owned",
        )
    return _warn(
        cid,
        name,
        "held-out predictive-DV vocabulary detected but no group-scoped fold "
        "(leave-one-<family>-out / a fit-on-one-corpus-score-on-another arm), no "
        "independence (i-i-d) argument, and no `N/A — no held-out predictive DV` escape "
        "declared on its own line, unwrapped (no backticks/quotes) — pointwise LOO can "
        "REORDER cross-context claims "
        "(#810: read-out rho 0.909 → 0.285 under the family-level fold); "
        ".claude/rules/ood-generalization-folds.md; Statistics critic must gate this",
    )


# ─── Check 20 — verdict-lattice coherence (conditional) ────────────────────

# Trigger sections: hypothesis / success / kill / decision / verdict / gate —
# the c8/c13 families plus "hypothes" + "verdict"; deliberately NOT
# "evaluation" (c13 includes it for gate lines; a verdict LATTICE registered
# only in an Evaluation recap is an accepted under-trigger — fails safe).
_C20_SECTION_RE = re.compile(r"(?i)hypothes|success|kill|decision|verdict|gate")

# Tier 1: the #923 v6 registered form — "…DISJOINT and exhaustive: <label> ⇔
# <predicate>; …". The declaration claims a partition, so BOTH defect classes
# (co-fire AND gap) are FAIL-capable.
_C20_DECL_RE = re.compile(r"(?i)\bdisjoint\b[^.\n]{0,60}\bexhaustive\b[^:\n]{0,20}:")
_C20_CLAUSE_RE = re.compile(r"([^;⇔\n]{1,80})\s*⇔\s*([^;\n]+)")

# Tier 2: verdict-label anchor applied to a list item's FIRST bold span.
_C20_LABEL_RE = re.compile(
    r"(?i)^(?:h[-\s]?\w|intermediate|inconclusive|confirm|falsif|success|kill|pass\b|fail\b)"
)
_C20_BOLD_RE = re.compile(r"\*\*([^*\n]{1,80})\*\*")

# Atom grammar. POINT: `<qty> ≥/> 0` → pos, `≤/< 0` → neg (interior
# semantics, §4.4 convention); the `0(?!\.?\d)` lookahead keeps "p ≤ 0.05"
# out (a decimal alpha is c13's shape, not a sign atom).
_C20_POINT_RE = re.compile(r"(?P<qty>[^\s,;()]+)\s*(?P<cmp>≥|>=|≤|<=|>|<)\s*0(?!\.?\d)")
_C20_POINT_POS = ("≥", ">=", ">")

# CI atoms: a `CI`/`CIs` token, a tiny closed copula gap, then one idiom.
# Axis binding: `paired` within the 40 chars BEFORE the CI token (window
# clamped at the previous atom's span end — a preceding atom's own `paired`
# wording never leaks into this atom's binding) → paired axis, else primary.
# Idiom order matters: side-qualified excludes before the bare two-sided
# exclude.
_C20_CI_TOKEN_RE = re.compile(r"(?i)\bCIs?\b")
_C20_CI_GAP_RE = re.compile(r"(?:\s+(?:is|are|stays?|remains?))?\s*")
_C20_Z = r"(?:0|zero)(?!\.?\d)"
_C20_CI_IDIOMS: list[tuple[re.Pattern[str], frozenset[str]]] = [
    (
        re.compile(r"(?i)exclud(?:es|ing)\s+" + _C20_Z + r"\s+on\s+the\s+positive\s+side"),
        frozenset({"above"}),
    ),
    (
        re.compile(r"(?i)exclud(?:es|ing)\s+" + _C20_Z + r"\s+on\s+the\s+negative\s+side"),
        frozenset({"below"}),
    ),
    (re.compile(r"(?i)strictly\s+positive\b"), frozenset({"above"})),
    (re.compile(r"(?i)strictly\s+negative\b"), frozenset({"below"})),
    (
        re.compile(r"(?i)wholly\s+(?:at\s+or\s+|at/)?above\s+" + _C20_Z),
        frozenset({"above"}),
    ),
    (re.compile(r"(?i)at\s+or\s+above\s+" + _C20_Z), frozenset({"above"})),
    (
        re.compile(r"(?i)wholly\s+below\s+" + _C20_Z + r"|below\s+zero\b"),
        frozenset({"below"}),
    ),
    (
        re.compile(r"(?i)(?:includes?|contains?|straddl(?:es?|ing)|overlaps?)\s+" + _C20_Z),
        frozenset({"straddle"}),
    ),
    (
        re.compile(r"(?i)exclud(?:es|ing)\s+" + _C20_Z + r"|clear\s+of\s+" + _C20_Z),
        frozenset({"below", "above"}),
    ),
]

# OTHERWISE atom (complement label — fires iff no non-otherwise label fires).
_C20_OTHERWISE_RE = re.compile(
    r"(?i)\botherwise\b|\ball other\b|\bneither\b[^.;\n]{0,40}\bfires?\b|\bno binary verdict\b"
)

# Completeness-gate residue tokens: any CI token, comparator char, idiom
# keyword, or NEGATOR (Must-Fix: "the CI never includes 0" would otherwise
# parse as the positive atom with inverted polarity) OUTSIDE every recognized
# atom span makes the label `unparsed` — the lattice is then never
# FAIL-capable (WARN).
_C20_RESIDUE_RE = re.compile(
    r"(?i)\bCIs?\b|[<>≤≥]"
    r"|\binclud\w*|\bexclud\w*|\bstraddl\w*|\bwholly\b|\bstrictly\b|\bclear of\b"
    r"|\b(?:not|never|no|nor|unless|except|without)\b|\bfails?\s+to\b"
)

# Connectives: only AND / OR (incl. ", OR") / `with` (AND-equivalent) join
# atoms; any other joiner (bare comma, if/when chains, and/or → two hits)
# is fail-closed to `unparsed` — no silent default connective.
_C20_CONNECTIVE_RE = re.compile(r"(?i)\b(?:and|or|with)\b")

# Axis-identity fail-closed guard (ii): post-CI `paired` wording ("the CI of
# the paired difference includes 0") is never silently bound to an axis.
_C20_POST_CI_PAIRED_RE = re.compile(r"(?i)\bCIs?\b\s+(?:of|on|for|over)\s+(?:the\s+)?paired\b")

# Precedence-phrase screen: an order-evaluated lattice is coherent in a way
# the cell algebra cannot see → fail closed to `unparsed` (WARN).
_C20_PRECEDENCE_RE = re.compile(
    r"(?i)first matching|in (?:that |this )?order|takes precedence|evaluated in order|\bwins\b"
)

# Quantifier screen (tier 2): k-of-n / per-family predicates ("at >= 4/6
# pre-registered layers", "for all traits") are outside the v1 cell algebra
# -> SKIP.
# Deliberately NOT bare "every" (v6's recap says "for every … cell").
_C20_QUANT_RE = re.compile(
    r"(?i)(?:at least\s+\d+|≥\s*\d+|>=\s*\d+)\s*(?:of|/)\s*\d+|\ball\s+\d+\b|\bfor (?:all|each)\b"
)

# Tier-2 segment machinery: sentence split, →/Consequence truncation, the
# "confirmed if(f)" selector.
_C20_SENT_SPLIT_RE = re.compile(r"(?<=\.)\s+")
_C20_TRUNC_RE = re.compile(r"→|\bConsequence\b")
_C20_CONFIRMED_RE = re.compile(r"(?i)\bconfirmed\s+iff?\b")
# Tier-1 clause predicates truncate at the first sentence terminator so a
# trailing recap sentence (v6's "Exactly one label fires for every … cell.")
# never enters the otherwise clause as residue.
_C20_SENT_END_RE = re.compile(r"\.(?=\s|$)")

_C20_CI_STATES = ("below", "straddle", "above")


def _c20_trigger_sections(plan: str) -> list[str]:
    """Fence-stripped texts of the OUTERMOST sections whose heading matches
    the c20 trigger families (a nested matching heading inside an
    already-taken section is not re-collected)."""
    lines = plan.splitlines()
    mask = _fence_mask(lines)
    taken: list[tuple[int, int]] = []
    out: list[str] = []
    for h in _headings(plan):
        if not _C20_SECTION_RE.search(h.text):
            continue
        if any(s <= h.line and h.end <= e for s, e in taken):
            continue
        taken.append((h.line, h.end))
        out.append("\n".join(lines[j] for j in range(h.line, h.end) if not mask[j]))
    return out


def _c20_label_name(bold: str) -> str:
    """Short display name for a harvested label: the bold span up to its
    first parenthetical annotation, trailing colon stripped."""
    return bold.split(" (")[0].rstrip(": ").strip()


def _c20_any_ci_idiom(text: str) -> bool:
    """True when ``text`` carries at least one CI-predicate idiom (the
    harvest condition — presence-only; atom adjacency is parse-time)."""
    return any(pat.search(text) for pat, _ in _C20_CI_IDIOMS)


def _c20_harvest_labels(section_text: str) -> list[dict]:
    """Tier-2 label harvest over one (fence-stripped) trigger section:
    top-level list items whose FIRST bold span matches the verdict-label
    anchor AND whose text carries a CI idiom (or an otherwise-token — an
    idiom-free complement label like "**Inconclusive:** otherwise" still
    joins the lattice it completes). Returns
    ``[{name, text, idiom}]`` in document order."""
    labels: list[dict] = []
    for block in _hypothesis_blocks(section_text):
        first_line = block.splitlines()[0] if block else ""
        if not _C14_LIST_ITEM_RE.match(first_line):
            continue
        bm = _C20_BOLD_RE.search(block)
        if bm is None:
            continue
        bold = bm.group(1).strip()
        if not _C20_LABEL_RE.match(bold):
            continue
        has_idiom = _c20_any_ci_idiom(block)
        if not (has_idiom or _C20_OTHERWISE_RE.search(block)):
            continue
        labels.append(
            {"name": _c20_label_name(bold), "text": block[bm.end() :], "idiom": has_idiom}
        )
    return labels


def _c20_has_atom(sentence: str) -> bool:
    """True when ``sentence`` carries a full parseable atom (point, CI, or
    otherwise) — idiom presence alone does not count (a CI idiom with no
    adjacent CI token is a residue shape, not an atom)."""
    if _C20_POINT_RE.search(sentence) or _C20_OTHERWISE_RE.search(sentence):
        return True
    for m in _C20_CI_TOKEN_RE.finditer(sentence):
        gm = _C20_CI_GAP_RE.match(sentence, m.end())
        if any(pat.match(sentence, gm.end()) for pat, _ in _C20_CI_IDIOMS):
            return True
    return False


def _c20_segment(label_text: str) -> tuple[str | None, str | None]:
    """``(predicate_segment, unparsed_reason)`` for a tier-2 label: the
    sentence containing "confirmed if(f)" when present, else the SINGLE
    atom-bearing sentence; each sentence truncated at the first ``→`` /
    ``Consequence`` token. >1 atom-bearing sentence without a confirmed-iff
    selector is ambiguous → unparsed."""
    sentences = [_C20_TRUNC_RE.split(s)[0] for s in _C20_SENT_SPLIT_RE.split(label_text)]
    confirmed = [s for s in sentences if _C20_CONFIRMED_RE.search(s)]
    if confirmed:
        return confirmed[0], None
    bearing = [s for s in sentences if _c20_has_atom(s)]
    if len(bearing) > 1:
        return None, ">1 atom-bearing sentence and no 'confirmed if(f)' selector — ambiguous"
    if not bearing:
        return None, "no sentence with a parseable atom"
    return bearing[0], None


def _c20_collect_atoms(segment: str) -> tuple[list[tuple[str, frozenset[str], int, int]], set]:
    """All sign/CI atoms in ``segment`` as ``(axis, values, start, end)``
    (sorted by position) plus the set of normalized POINT quantities. The
    axis-binding lookback is clamped at the previous atom's span end so a
    preceding atom's `paired` token never mis-binds a later primary atom."""
    atoms: list[tuple[str, frozenset[str], int, int]] = []
    qtys: set[str] = set()
    for m in _C20_POINT_RE.finditer(segment):
        sign = "pos" if m.group("cmp") in _C20_POINT_POS else "neg"
        qtys.add(m.group("qty").strip("`*").casefold())
        atoms.append(("point", frozenset({sign}), m.start(), m.end()))
    for m in _C20_CI_TOKEN_RE.finditer(segment):
        gm = _C20_CI_GAP_RE.match(segment, m.end())
        hit: tuple[re.Match[str], frozenset[str]] | None = None
        for pat, states in _C20_CI_IDIOMS:
            im = pat.match(segment, gm.end())
            if im:
                hit = (im, states)
                break
        if hit is None:
            continue  # the stray CI token becomes completeness residue
        # Clamp the lookback at the previous atom's span end: a paired atom
        # < 40 chars BEFORE a primary atom would otherwise leak its `paired`
        # token into THIS atom's window, binding both atoms to the paired
        # axis — a contradictory conjunction that never fires, manufacturing
        # a tier-1 gap → false FAIL (round-1 code-review Minor).
        prev_end = max((a[3] for a in atoms if a[3] <= m.start()), default=0)
        lookback = segment[max(0, m.start() - 40, prev_end) : m.start()].lower()
        axis = "paired" if "paired" in lookback else "primary"
        atoms.append((axis, hit[1], m.start(), hit[0].end()))
    atoms.sort(key=lambda a: a[2])
    return atoms, qtys


def _c20_build_dnf(
    segment: str, atoms: list[tuple[str, frozenset[str], int, int]]
) -> tuple[list[list[tuple[str, frozenset[str]]]] | None, str | None]:
    """``(dnf, None)`` for the atom chain under AND > OR precedence with the
    connective fail-closed rule, or ``(None, reason)``."""
    for i in range(1, len(atoms)):
        if atoms[i][2] < atoms[i - 1][3]:
            return None, "overlapping atom spans"
    conns: list[str] = []
    for i in range(1, len(atoms)):
        gap = segment[atoms[i - 1][3] : atoms[i][2]]
        found = [c.lower() for c in _C20_CONNECTIVE_RE.findall(gap)]
        if len(found) != 1:
            return None, f"joiner between atoms is not exactly one of AND/OR/with ({gap.strip()!r})"
        conns.append(found[0])
    groups: list[list[tuple[str, frozenset[str]]]] = [[(atoms[0][0], atoms[0][1])]]
    for i, conn in enumerate(conns, start=1):
        if conn == "or":
            groups.append([(atoms[i][0], atoms[i][1])])
        else:  # and / with — AND-equivalent
            groups[-1].append((atoms[i][0], atoms[i][1]))
    return groups, None


def _c20_parse_predicate(segment: str) -> dict:
    """Compile one predicate segment to DNF over sign/CI atoms (or an
    otherwise-label). Fail-closed: any completeness-gate residue (stray CI
    token / comparator / idiom keyword / NEGATOR), any non-AND/OR/with
    joiner between atoms, or an otherwise-token mixed with predicate atoms
    marks the segment ``unparsed`` (reason in the returned dict)."""
    out: dict = {"otherwise": False, "dnf": [], "unparsed": None, "point_qtys": set()}
    atoms, out["point_qtys"] = _c20_collect_atoms(segment)
    otherwise_spans = [(m.start(), m.end()) for m in _C20_OTHERWISE_RE.finditer(segment)]
    if otherwise_spans and atoms:
        out["unparsed"] = "an 'otherwise' token mixed with predicate atoms in one segment"
        return out
    spans = otherwise_spans if otherwise_spans else [(a[2], a[3]) for a in atoms]
    residues = [
        m.group(0)
        for m in _C20_RESIDUE_RE.finditer(segment)
        if not any(s <= m.start() and m.end() <= e for s, e in spans)
    ]
    if residues:
        out["unparsed"] = "predicate token(s) outside every recognized atom: " + ", ".join(
            repr(r) for r in residues[:4]
        )
        return out
    if otherwise_spans:
        out["otherwise"] = True
        return out
    if not atoms:
        out["unparsed"] = "no recognized atom"
        return out
    dnf, reason = _c20_build_dnf(segment, atoms)
    if dnf is None:
        out["unparsed"] = reason
        return out
    out["dnf"] = dnf
    return out


def _c20_enumerate(labels: list[dict]) -> tuple[list, list]:
    """Interior-cells-only 3-state enumeration over the REFERENCED axes with
    point-in-CI coherence pruning (a bootstrap CI contains its point
    estimate). Returns ``(cofires, gaps)`` — cofires as ``(cell, [label
    names])``, gaps as bare cells. An otherwise-label fires exactly on the
    cells no predicate label covers (killing gap findings by construction)."""
    preds = [lab for lab in labels if not lab["parse"]["otherwise"]]
    others = [lab for lab in labels if lab["parse"]["otherwise"]]
    axes = {axis for lab in preds for conj in lab["parse"]["dnf"] for axis, _ in conj}
    primary_vals: tuple = _C20_CI_STATES if "primary" in axes else (None,)
    paired_vals: tuple = _C20_CI_STATES if "paired" in axes else (None,)
    cofires: list[tuple[dict, list[str]]] = []
    gaps: list[dict] = []
    for primary in primary_vals:
        if "point" not in axes:
            point_vals: tuple = (None,)
        elif primary is None:
            point_vals = ("neg", "pos")
        else:
            point_vals = {"below": ("neg",), "straddle": ("neg", "pos"), "above": ("pos",)}[primary]
        for point in point_vals:
            for paired in paired_vals:
                cell = {"point": point, "primary": primary, "paired": paired}
                fired = [
                    lab
                    for lab in preds
                    if any(
                        all(cell[axis] in values for axis, values in conj)
                        for conj in lab["parse"]["dnf"]
                    )
                ]
                if not fired and others:
                    fired = others
                if len(fired) >= 2:
                    cofires.append((cell, [lab["name"] for lab in fired]))
                elif not fired:
                    gaps.append(cell)
    return cofires, gaps


def _c20_cell_str(cell: dict) -> str:
    """Plain-terms cell rendering for FAIL/WARN details."""
    parts: list[str] = []
    if cell["point"] is not None:
        parts.append("point > 0" if cell["point"] == "pos" else "point < 0")
    for axis in ("primary", "paired"):
        v = cell[axis]
        if v is not None:
            word = {"below": "wholly below 0", "straddle": "straddles 0", "above": "wholly above 0"}
            parts.append(f"{axis} CI {word[v]}")
    return "{" + ", ".join(parts) + "}"


_C20_REMEDY = (
    " — restate the lattice as an explicit partition (`DISJOINT and exhaustive: "
    "<label> ⇔ <predicate>; …; <label> ⇔ otherwise`), add an otherwise-label, or "
    "declare 'N/A — no registered verdict lattice' on its own line, unwrapped "
    "(no backticks/quotes)"
)


def _c20_offender_detail(tier_desc: str, cofires: list, gaps: list) -> str:
    """Bounded offender detail: co-fire cells with both label names first,
    gap cells as the secondary note, ≤4 shown each, remedy menu last."""
    bits: list[str] = []
    if cofires:
        shown = "; ".join(
            f"labels {' + '.join(names)} CO-FIRE on cell {_c20_cell_str(cell)}"
            for cell, names in cofires[:4]
        )
        if len(cofires) > 4:
            shown += "; …"
        bits.append(shown)
    if gaps:
        shown = ", ".join(_c20_cell_str(c) for c in gaps[:4])
        if len(gaps) > 4:
            shown += ", …"
        bits.append(f"no label fires on cell(s) {shown}")
    return (
        f"the registered verdict lattice ({tier_desc}) is not a partition: "
        + "; ".join(bits)
        + _C20_REMEDY
    )


def _c20_evaluate_lattice(labels: list[dict], *, tier: int, section_text: str) -> tuple[str, str]:
    """Shared per-lattice verdict core → ``(state, detail)`` with state in
    {"unparsed", "cofire", "gap", "clean"}. The kind/tier degradations
    (§4.5 table) are applied by the caller."""
    names = " / ".join(lab["name"] for lab in labels)
    tier_desc = f"tier {tier}: {names}"
    pm = _C20_PRECEDENCE_RE.search(section_text)
    if pm:
        return (
            "unparsed",
            f"label-precedence phrase {pm.group(0)!r} in the lattice's section makes the "
            "labels order-evaluated — the cell algebra cannot verify an ordered lattice; "
            "restate it as the explicit ⇔ partition form",
        )
    unparsed = [lab for lab in labels if lab["parse"]["unparsed"]]
    if unparsed:
        first = unparsed[0]
        return (
            "unparsed",
            f"label '{first['name']}' ({tier_desc}) did not fully parse: "
            f"{first['parse']['unparsed']} — the lattice is not FAIL-capable; restate it as "
            "the explicit ⇔ partition form (`DISJOINT and exhaustive: <label> ⇔ <predicate>; "
            "…`) so coherence is machine-checkable",
        )
    qtys = set()
    for lab in labels:
        qtys |= lab["parse"]["point_qtys"]
    if len(qtys) > 1:
        return (
            "unparsed",
            f"the lattice's labels reference {len(qtys)} distinct point quantities "
            f"({', '.join(sorted(qtys)[:4])}) — a single-point-axis cell algebra cannot "
            "represent them (never silently collapsed onto one axis); restate the lattice "
            "over one point quantity or use the explicit ⇔ partition form",
        )
    cofires, gaps = _c20_enumerate(labels)
    if cofires or gaps:
        detail = _c20_offender_detail(tier_desc, cofires, gaps)
        return ("cofire" if cofires else "gap", detail)
    return (
        "clean",
        f"{tier_desc} — every interior sign/CI cell fires exactly one label "
        "(partition verified in form; boundary semantics stay with the Statistics critic)",
    )


_C20_POST_CI_PAIRED_REASON = (
    "post-CI 'paired' wording (e.g. 'the CI of the paired difference') is "
    "not silently bound to an axis"
)


def _c20_find_declaration(sections: list[str]) -> tuple[str, list[tuple[str, str]]] | None:
    """First DISJOINT-and-exhaustive ⇔ declaration across the trigger
    sections → ``(section_text, [(label, predicate), …])``; None when no
    declaration line exists (tier 2 then applies)."""
    for sec in sections:
        for line in sec.splitlines():
            dm = _C20_DECL_RE.search(line)
            if not dm:
                continue
            clauses = []
            for chunk in line[dm.end() :].split(";"):
                cm = _C20_CLAUSE_RE.match(chunk)
                if cm:
                    clauses.append((cm.group(1).strip(), cm.group(2).strip()))
            return sec, clauses
    return None


def _c20_tier1_result(cid: str, name: str, kind: str, sec: str, clauses: list) -> CheckResult:
    """Tier-1 verdict: the plan CLAIMED a partition, so co-fire AND gap are
    both FAIL-capable (WARN under kind=analysis); unparsed clauses WARN."""
    if len(clauses) < 2:
        return _warn(
            cid,
            name,
            "a DISJOINT-and-exhaustive declaration was found but fewer than 2 "
            "`<label> ⇔ <predicate>` clauses parsed from it — the claimed partition is "
            "not machine-checkable; use the canonical form (`DISJOINT and exhaustive: "
            "<label> ⇔ <predicate>; …; <label> ⇔ otherwise`)",
        )
    labels = []
    for clabel, cpred in clauses:
        pred = _C20_SENT_END_RE.split(cpred)[0]
        parse = _c20_parse_predicate(pred)
        if _C20_POST_CI_PAIRED_RE.search(pred):
            parse["unparsed"] = _C20_POST_CI_PAIRED_REASON
        labels.append({"name": _c20_label_name(clabel), "parse": parse})
    state, detail = _c20_evaluate_lattice(labels, tier=1, section_text=sec)
    if state == "clean":
        return _pass(cid, name, detail)
    if state == "unparsed":
        return _warn(cid, name, detail)
    if kind == "analysis":
        return _warn(cid, name, detail + " (analysis kind-degrade: WARN, not FAIL)")
    return _fail(cid, name, detail)


def _c20_tier2_result(cid: str, name: str, kind: str, lattices: list) -> CheckResult:
    """Tier-2 verdict over every qualifying section's lattice (worst wins):
    complete-parse co-fire FAILs (WARN under kind=analysis); gap-only and
    any-unparsed WARN; any quantified label SKIPs the whole check."""
    worst: tuple[int, str, str] | None = None  # (rank, state, detail)
    rank = {"clean": 0, "gap": 1, "unparsed": 2, "cofire": 3}
    for sec, labels in lattices:
        for lab in labels:
            seg, reason = _c20_segment(lab["text"])
            if seg is not None and _C20_QUANT_RE.search(seg):
                return _skip(
                    cid,
                    name,
                    f"label '{lab['name']}' carries quantified verdict predicates out of v1 "
                    "scope (k-of-n / per-family lattices are the Statistics critic's)",
                )
            if _C20_POST_CI_PAIRED_RE.search(lab["text"]):
                seg, reason = None, _C20_POST_CI_PAIRED_REASON
            if reason is not None:
                lab["parse"] = {
                    "otherwise": False,
                    "dnf": [],
                    "unparsed": reason,
                    "point_qtys": set(),
                }
            else:
                lab["parse"] = _c20_parse_predicate(seg)
        state, detail = _c20_evaluate_lattice(labels, tier=2, section_text=sec)
        if worst is None or rank[state] > worst[0]:
            worst = (rank[state], state, detail)
    assert worst is not None  # ≥1 lattice on this branch
    _, state, detail = worst
    if state == "clean":
        return _pass(cid, name, detail)
    if state == "cofire" and kind == "experiment":
        return _fail(cid, name, detail)
    if state == "cofire":
        return _warn(cid, name, detail + " (analysis kind-degrade: WARN, not FAIL)")
    if state == "gap":
        return _warn(
            cid,
            name,
            detail + " (tier-2 gap degrades to WARN: gap precision depends on harvest recall)",
        )
    return _warn(cid, name, detail)


def check_verdict_lattice_coherence(plan: str, kind: str) -> CheckResult:
    """A REGISTERED VERDICT LATTICE — success/kill/intermediate labels
    defined by interval predicates over point estimates and CIs — must be
    mutually exclusive and exhaustive over the interior sign/CI cells.
    Tier 1 (the explicit "DISJOINT and exhaustive: <label> ⇔ <predicate>"
    declaration) is FAIL-capable on co-fire AND gap (the plan claimed a
    partition); tier 2 (per-label prose, the #923 v4 shape) FAILs only on a
    co-fire with a COMPLETE parse — gaps degrade to WARN (gap precision
    depends on harvest recall), any unparsed label degrades the whole
    lattice to WARN, and quantified (k-of-n) predicates SKIP as out of the
    v1 cell algebra. FAIL (experiment) / WARN (analysis) / SKIP otherwise;
    escape via a standalone ``N/A — no registered verdict lattice`` line —
    honored (SKIP path) only when no lattice is detected; when the escape
    co-occurs with a detected lattice (either tier) the check WARNs instead
    of PASSing, so the escape can never mask verification of a present
    lattice (#1223).
    Incident: #923 amendment plan v4/v5 §3 — a bare positive point estimate
    with both CIs straddling 0 fired BOTH H-slot and Intermediate (and one
    cell fired neither); caught only by the Codex statistics critic, fixed
    by hand in v6."""
    cid, name = "c20_verdict_lattice_coherence", "verdict-lattice coherence"
    if kind not in ("experiment", "analysis"):
        return _skip(
            cid, name, "kind-exempt: registered verdict lattices are an experiment|analysis shape"
        )
    sections = _c20_trigger_sections(plan)
    # Tier 1 takes precedence over tier 2 when a declaration exists anywhere.
    tier1 = _c20_find_declaration(sections)
    lattices: list[tuple[str, list[dict]]] = []
    if tier1 is None:
        for sec in sections:
            labels = _c20_harvest_labels(sec)
            if sum(1 for lab in labels if lab["idiom"]) >= 2:
                lattices.append((sec, labels))
    if tier1 is None and not lattices:
        return _skip(
            cid,
            name,
            "no registered verdict lattice detected (no DISJOINT-and-exhaustive ⇔ "
            "declaration; fewer than 2 anchored CI-predicate labels in any trigger section)",
        )
    if _standalone_na_declared(plan, r"no registered verdict lattice"):
        # #1223: this branch is reachable ONLY when a lattice WAS detected (the
        # no-lattice case SKIPs above) — a PASS here masks the very defect c20
        # exists to catch. WARN, not FAIL: the detection may be a false positive
        # on quoted guidance (all 3 corpus co-occurrences were), so the escape
        # stays non-blocking and the reviewers adjudicate.
        detected = (
            "a tier-1 DISJOINT-and-exhaustive ⇔ declaration"
            if tier1 is not None
            else f"{len(lattices)} trigger section(s) with ≥2 anchored CI-predicate labels (tier 2)"
        )
        return _warn(
            cid,
            name,
            "the standalone `N/A — no registered verdict lattice` escape co-occurs "
            f"with {detected} — the escape is reserved for lattice-free plans and "
            "would mask coherence verification of the detected lattice (#1223); "
            "remove the N/A line (the lattice is then verified), or fence/remove the "
            "lattice-shaped prose the detector matched if it is quoted guidance "
            "rather than this plan's own registration",
        )
    if tier1 is not None:
        return _c20_tier1_result(cid, name, kind, tier1[0], tier1[1])
    return _c20_tier2_result(cid, name, kind, lattices)


# ─── Check 21 — grep-arity acceptance gate → AST arity audit (WARN-only) ──

# A grep invocation whose quoted pattern is call-shaped: an identifier
# immediately followed by `(` inside the quotes
# (`grep -rn "parse_judge_json(" ...`). [^|\n] bounds the scan to the
# grep component's own arguments, not a later pipeline component.
_C21_GREP_CALL_RE = re.compile(r"""grep\b[^|\n]*["'][^"'\n]*\w\(""")

# Pipeline-form arity discriminator: any grep component on the SAME line
# whose quoted pattern contains a comma (`| grep ", "`, `grep -c 'f(.*,'`).
_C21_GREP_COMMA_RE = re.compile(r"""grep\b[^|\n]*["'][^"'\n]*,[^"'\n]*["']""")

# Count form: `... | wc -l`, or a grep flag cluster carrying -c
# (`grep -c`, `grep -rnc`; a separated `grep -r -c` is a known miss).
_C21_COUNT_RE = re.compile(r"""wc\s+-l|\bgrep\s+-\w*c\b""")

# Prose-form arity vocabulary ("shows zero two-argument calls").
_C21_ARITY_VOCAB_RE = re.compile(
    r"(?i)\btwo-?arg\w*|\b(?:one|two|three|\d+)[- ]argument|\barity\b"
    r"|second argument|keyword[- ]arg\w*"
)

# Registered zero-count pass condition — the comparator that makes a grep
# a GATE rather than a discovery command. Deliberately absent: bare
# `\bempty\b` and un-bounded `→ 0` (matched unrelated prose on #416/#467/
# #870 in the calibration sweep).
_C21_ZERO_RE = re.compile(
    r"(?i)==?\s*`?0\b|\bshows zero\b|\bzero\b[^.\n]{0,40}\bcalls?\b"
    r"|returns nothing|\b0 hits\b|must be 0\b"
)

# Evidence escape: the plan names an AST-based arity audit anywhere.
_C21_AST_EVIDENCE_RE = re.compile(
    r"(?i)ast\.(?:walk|parse)|\bAST[- ](?:based|arity|audit|walker)|libcst"
)


def check_grep_arity_gate(plan: str, kind: str) -> CheckResult:
    """Plans registering a grep/`wc -l`-based signature-ARITY acceptance
    gate (`grep "func(" ... | grep ", " | wc -l` == 0, or a call-pattern
    grep whose stated pass condition is "shows zero two-argument calls")
    get a WARN pointing at the AST-based arity audit as the robust form:
    comma heuristics over call sites are BOTH unsatisfiable (they count
    deliberate two-arg tests + comma-bearing string literals) AND
    under-detecting (split-line and keyword-argument calls carry no
    same-line comma) — #1024 plan v1/v2 registered exactly this gate and
    the critic ensemble replaced it with an ast.walk audit in v3. WARN
    not FAIL: greps are legitimate for discovery/enumeration, and the
    conjunctive line trigger (call-pattern grep + arity discriminator +
    count/comparator) is a heuristic — the Phase 1.5/2 reviewers
    adjudicate. ALL kinds: the incident was kind: infra, but signature
    migrations also ride experiment plans' code-port phases, and the
    2026-07-04 corpus sweep (1,329 plans/v*.md) fired on ZERO lines
    outside #1024's own plan versions, so kind confinement buys no
    precision and costs recall. Raw lines are scanned WITHOUT the fence
    mask (gate commands live in inline backticks and fenced verification
    blocks alike); section-window confinement is the first tightening
    lever if a future sweep surfaces false positives. The 0-FP figure is
    an IN-SAMPLE calibration (regexes tuned on the same historical
    corpus the acceptance sweep re-runs) — it bounds nuisance cost on
    yesterday's planner distribution, not a guarantee for future plans."""
    cid, name = "c21_grep_arity_gate", "grep-arity acceptance gate points at AST audit"
    del kind  # all kinds — trigger precision carries the false-positive discipline
    hits: list[tuple[int, str]] = []
    for i, line in enumerate(plan.splitlines(), 1):
        if not _C21_GREP_CALL_RE.search(line):
            continue
        pipeline = _C21_GREP_COMMA_RE.search(line) and _C21_COUNT_RE.search(line)
        prose = _C21_ARITY_VOCAB_RE.search(line) and _C21_ZERO_RE.search(line)
        if pipeline or prose:
            hits.append((i, line.strip()))
    if not hits:
        return _skip(cid, name, "no grep-based call-arity pass condition detected")
    if _standalone_na_declared(plan, r"no arity acceptance gate"):
        return _pass(
            cid, name, "explicit N/A declared (flagged grep is not an arity pass condition)"
        )
    if _C21_AST_EVIDENCE_RE.search(plan):
        i, line = hits[0]
        return _pass(
            cid,
            name,
            f"grep-arity gate present (line {i}: {line[:80]!r}) but the plan also names an "
            "AST-based arity audit — the robust form is registered",
        )
    i, line = hits[0]
    return _warn(
        cid,
        name,
        f"a registered pass condition counts comma-bearing call-pattern grep hits (line {i}: "
        f"{line[:100]!r}) — comma-grep arity gates are both unsatisfiable (they count "
        "deliberate two-arg tests and comma-bearing string literals) and under-detecting "
        "(split-line and keyword-argument calls carry no same-line comma; #1024 plan v1/v2). "
        "Register an AST arity audit instead: ast.parse each target file, ast.walk over Call "
        "nodes matching the function, count len(node.args) + len(node.keywords), whitelist "
        "named exceptions — or declare `N/A — no arity acceptance gate` on its own line, "
        "unwrapped (no backticks/quotes)",
    )


# ─── Check 22 — cross-section param consistency (WARN-only, all kinds) ─────

_C22_PARAM_TOKENS = (
    r"temperature|max_new_tokens|max_tokens|learning_rate|lr|epochs|"
    r"seeds|seed|rank|alpha|batch_size|batch|top_p"
)  # longer alternatives first where prefixes overlap
_C22_ALIASES = {"learning_rate": "lr", "seeds": "seed", "batch_size": "batch"}
_C22_NUM = r"(?:[0-9]+(?:\.[0-9]+)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?"
# param=value / param: value; tolerates `code`/**bold** wrappers; captures a
# single numeric, a comma-run of numerics, or a {...} brace set (<=120 chars).
# The leading \b means compound tokens (JUDGE_TEMPERATURE=0.7) never match —
# underscore is \w, so there is no boundary before the bare param name.
_C22_VALUE_RE = re.compile(
    rf"(?i)\b(?P<param>{_C22_PARAM_TOKENS})\b\s*[=:]\s*[`*]{{0,2}}\s*"
    rf"(?P<vals>\{{[^}}\n]{{0,120}}\}}|{_C22_NUM}(?:\s*,\s*{_C22_NUM})*)"
)
# Range/schedule continuation right after the captured value ("1e-4 → 1e-5",
# "1-3", "1 -> 3"): the tail value joins the occurrence's value set.
_C22_RANGE_TAIL_RE = re.compile(
    rf"\s*[`*]{{0,2}}\s*(?:[-\u2013\u2014]|->|→)\s*[`*]{{0,2}}({_C22_NUM})"
)
# Omission assertion: the #1024 corrected-text shape ("temperature OMITTED").
_C22_OMIT_RE = re.compile(
    rf"(?i)\b(?P<param>{_C22_PARAM_TOKENS})\b\s+(?:is\s+)?(?:omitted|left\s+unset)\b"
)
# Historical / declared-but-never-threaded clause vocabulary (value
# occurrences only). `was` is value-adjacent only (`was 0.7` / `was set`),
# NOT bare \bwas\b — bare `was` is ubiquitous and would silently exclude
# CURRENT stale values on lines like "temperature=0.7 was chosen per #612".
_C22_EXCLUDE_RE = re.compile(
    rf"(?i)declared\s+but\s+never|never\s+threaded|not\s+threaded|never\s+used|"
    rf"\bpreviously\b|superseded|corrected\s+(?:from|to)|historical|\bstale\b|"
    rf"deprecated|no\s+longer|old\s+(?:value|default)|\bwas\s+(?:{_C22_NUM}|set\b|used\b)"
)
_C22_SWEEP_LINE_RE = re.compile(r"(?i)\bsweeps?\b|\bgrid\b|ablation")
_C22_PHASE_RE = re.compile(r"(?i)\bphase[\s-]*([0-9]+)\b")
_C22_LORA_CTX_RE = re.compile(r"(?i)lora|rslora|\brank\b|adapter|peft")

# Same-line character window around a value match inside which the
# historical-clause vocabulary excludes the occurrence (window-bounded so a
# very long line's distant vocabulary cannot wrongly exclude a live value).
_C22_EXCLUDE_WINDOW_CHARS = 100


def _c22_top_section(headings: list[Heading], line_idx: int) -> tuple[int, str]:
    """Top-level-section attribution for ``line_idx``: the SHALLOWEST heading
    of level >= 2 containing the line (the H2 ancestor — sibling H3
    subsections under one ``## 4. Design`` group as ONE section); falls back
    to ``_innermost_section`` for H1-only docs, else a synthetic preamble
    key. Returns ``(heading.line, heading.text)`` as the section key."""
    candidates = [h for h in headings if h.level >= 2 and h.line <= line_idx < h.end]
    if candidates:
        best = min(candidates, key=lambda h: h.level)
        return (best.line, best.text)
    inner = _innermost_section(headings, line_idx)
    if inner is not None:
        return (inner.line, inner.text)
    return (-1, "(preamble)")


def _c22_record(
    occ: dict[str, dict[tuple[int, str], dict]],
    key: str,
    section: tuple[int, str],
    vals: set,
    lineno: int,
    span: str,
) -> None:
    """Union ``vals`` into ``occ[key][section]``, keeping the FIRST matched
    ``(lineno, span)`` per (param, section) for the WARN detail."""
    recs = occ.setdefault(key, {})
    rec = recs.get(section)
    if rec is None:
        recs[section] = {"vals": set(vals), "lineno": lineno, "span": span}
    else:
        rec["vals"] |= vals


def _c22_collect_occurrences(plan: str) -> dict[str, dict[tuple[int, str], dict]]:
    """Build the (param-key → top-level section → {vals, lineno, span}) map.

    Fenced lines never vote (module convention). Value occurrences on
    sweep/grid/ablation lines, stats-``alpha`` outside LoRA context, and
    values inside a historical/never-threaded clause (same-line ±100-char
    window) are excluded. Literal omission assertions (``<param> OMITTED``)
    are EXEMPT from the exclusion filter — the corrected text legitimately
    reads "temperature OMITTED — the builders never set it": the clause
    explains the omission, it does not mark it historical. Phase-qualified
    lines key as ``<param>@phase<K>``."""
    lines = plan.splitlines()
    mask = _fence_mask(lines)
    headings = _headings(plan)
    occ: dict[str, dict[tuple[int, str], dict]] = {}
    for i, line in enumerate(lines):
        if mask[i]:
            continue
        pm = _C22_PHASE_RE.search(line)
        phase = f"@phase{pm.group(1)}" if pm else ""
        section = _c22_top_section(headings, i)
        for m in _C22_VALUE_RE.finditer(line):
            param = _C22_ALIASES.get(m["param"].lower(), m["param"].lower())
            if param == "alpha" and not _C22_LORA_CTX_RE.search(line):
                continue  # stats-alpha guard: significance level, not LoRA alpha
            if _C22_SWEEP_LINE_RE.search(line):
                continue  # sweep/grid/ablation declarations are legitimately multi-value
            w = _C22_EXCLUDE_WINDOW_CHARS
            window = line[max(0, m.start() - w) : m.end() + w]
            if _C22_EXCLUDE_RE.search(window):
                continue  # historical / declared-but-never-threaded clause
            vals: set = {float(v) for v in re.findall(_C22_NUM, m["vals"])}
            if not vals:
                continue  # non-numeric brace set
            tm = _C22_RANGE_TAIL_RE.match(line, m.end())
            if tm:
                vals.add(float(tm.group(1)))
            _c22_record(occ, param + phase, section, vals, i, m.group(0))
        for m in _C22_OMIT_RE.finditer(line):
            param = _C22_ALIASES.get(m["param"].lower(), m["param"].lower())
            _c22_record(occ, param + phase, section, {"OMITTED"}, i, m.group(0))
    return occ


def check_cross_section_param_consistency(plan: str, kind: str) -> CheckResult:
    """The same tracked hyperparameter stated with contradictory values in
    DIFFERENT top-level sections is the #1024 incident class: a fact-check
    correction lands in one section while a stale restatement survives in
    another (§11 *What:* lines, assumption rows). Tracked params:
    temperature, max_tokens / max_new_tokens (distinct keys — API judge cap
    vs HF generate cap), lr / learning_rate, epochs, seed / seeds, rank,
    alpha (LoRA context only), batch / batch_size, top_p. A conflict is a
    pair of top-level sections whose value SETS are disjoint — overlap is
    consistent, which is what lets per-arm tables, ranges/schedules, and
    seed lists restated against a member value all PASS while ``0.7`` vs
    ``OMITTED``/``1.0`` WARNs. v1 scope: value-vs-value plus
    value-vs-literal-omission-token (``<param> OMITTED`` / ``left unset``) —
    the #1024 v2 offender shape; broader omission phrasings ("builders omit
    temperature", "no temperature parameter") are OUT of v1 scope.
    WARN-only, never FAIL (legitimate multi-value plans exist, and Phase
    1.5.0 forwards WARNs verbatim into the fact-checker/critic briefs — the
    intended consumption path); ALL kinds (the motivating #1024 offender is
    ``kind: infra``); conditional SKIP when no tracked param spans >= 2
    top-level sections.

    Documented v1 limits: (i) half-corrected-section masking — a section
    carrying BOTH a stale ``temperature=0.7`` and the corrected
    "temperature OMITTED" unions to {0.7, OMITTED}, which overlaps a §11
    {0.7} → PASS; intra-section contradictions are out of v1 cross-section
    scope. (ii) Phase-qualifier asymmetry — a phase-qualified occurrence
    (``epochs@phase1``) never compares against an unqualified ``epochs=3``;
    a c22 PASS is not "no cross-section drift" for phase-keyed params.
    (iii) Markdown-table blindness — the value regex requires ``=`` or
    ``:``, so pipe-table hyperparameter rows (``| lr | 1e-4 |``) never
    parse; the table-vs-prose restatement class is invisible to v1."""
    cid, name = "c22_cross_section_param_consistency", "cross-section param consistency"
    del kind  # registry symmetry, c5-style: c22 runs for ALL kinds (#1024 is kind: infra)
    occ = _c22_collect_occurrences(plan)
    cross = {k: recs for k, recs in occ.items() if len(recs) >= 2}
    if not cross:
        return _skip(cid, name, "no cross-section parameter restatement detected")
    conflicts: list[tuple[str, tuple, tuple]] = []
    for key, recs in cross.items():
        sections = list(recs.items())
        found = None
        for a in range(len(sections)):
            for b in range(a + 1, len(sections)):
                if sections[a][1]["vals"].isdisjoint(sections[b][1]["vals"]):
                    found = (key, sections[a], sections[b])
                    break
            if found:
                break
        if found:
            conflicts.append(found)  # first disjoint pair reported per param
    if not conflicts:
        return _pass(
            cid, name, f"{len(cross)} parameter(s) restated across sections, all consistent"
        )
    parts = []
    for key, (sec_a, rec_a), (sec_b, rec_b) in conflicts[:2]:
        parts.append(
            f"{key}: '{rec_a['span']}' (§'{sec_a[1]}' L{rec_a['lineno'] + 1}) vs "
            f"'{rec_b['span']}' (§'{sec_b[1]}' L{rec_b['lineno'] + 1})"
        )
    more = len(conflicts) - 2
    detail = (
        "; ".join(parts)
        + (f" …and {more} more param(s)" if more > 0 else "")
        + " — cross-section contradiction; if one side is a stale post-correction "
        "restatement, fix it"
    )
    return _warn(cid, name, detail)


# ─── Check 23 — goal currency (outside CHECKS; --issue mode only) ─────────

# Word-shingle stale-quote detector for the #922 plan-vs-goal incident class:
# a plan head quoting a SUPERSEDED Goal at high coverage while the CURRENT
# Goal is absent. WARN-only (the c21 WARN-first precedent, #1042); the
# forced redraft is delivered by the adversarial-planner SKILL.md
# § Goal-currency gate ("the one WARN that bounces"). Needs task context
# (body.md + events.jsonl), so it lives OUTSIDE verify_plan_text() and is
# appended by main() in --issue mode only.
_C23_SHINGLE_K = 6
_C23_MIN_GOAL_WORDS = 12
_C23_STALE_COV = 0.5
_C23_CURRENT_COV = 0.3
# NO positive slack: retro-stale goal-update gaps of ~3-6 min exist in the
# corpus (779/477/489) — any slack ≥ ~3 min manufactures false positives.
_C23_MTIME_SLACK_S = 0.0


def _norm_goal_words(s: str) -> list[str]:
    """Lowercase; non-alphanumerics (incl. unicode math) -> space; split."""
    return [w for w in re.sub(r"[^a-z0-9 ]+", " ", s.lower()).split() if w]


def _goal_shingles(words: list[str], k: int = _C23_SHINGLE_K) -> set[tuple[str, ...]]:
    """All contiguous k-word shingles of ``words`` (empty set below k words)."""
    if len(words) < k:
        return set()
    return {tuple(words[i : i + k]) for i in range(len(words) - k + 1)}


def _shingle_coverage(goal: str, head_words: list[str]) -> float:
    """Fraction of the goal's k-word shingles present in the plan head."""
    gs = _goal_shingles(_norm_goal_words(goal))
    if not gs:
        return 0.0
    hs = _goal_shingles(head_words)
    return sum(1 for s in gs if s in hs) / len(gs)


def _plan_head_words(plan: str) -> list[str]:
    """Head region = start -> the ``## 2.``/``### 2.`` heading, else first 8000 chars."""
    m = re.search(r"^#{2,3}\s*2\.\s", plan, re.M)
    return _norm_goal_words(plan[: m.start()] if m else plan[:8000])


def _goal_history_for_plan(folder: Path, plan_mtime_utc: datetime) -> tuple[str | None, list[str]]:
    """(current_goal, superseded_goals) AS OF the plan version's post time.

    current = latest predating ``epm:goal-updated`` ``to:`` (fallback:
    body.md frontmatter ``goal:``); superseded = predating markers'
    ``from:`` values (structured fields only — ``task.py set-goal`` posts
    top-level ``from``/``to``/``by``; hand-posted note-only markers
    contribute nothing). Bounded STRICTLY by mtime (slack 0, ``ts <= mtime``
    inclusive) so goal-updates that postdate the plan never retro-flag it —
    a positive slack manufactures FPs from minutes-scale retro-stale gaps
    (779/477/489).

    Read discipline: records split on ``"\\n"`` — NEVER ``str.splitlines()``
    (the paired writer ``task_workflow._append_jsonl_line`` emits
    ``ensure_ascii=False``, so a raw U+2028/U+2029/NEL inside a goal/note
    string is ONE valid JSONL record that ``splitlines()`` would shred,
    crashing the strict ``json.loads`` or silently dropping the marker —
    the #950 class; mirrors ``task_workflow._iter_jsonl``). Fail-fast: a
    row whose ``kind`` IS ``epm:goal-updated`` but whose ``ts`` is missing
    or non-string raises ``ValueError`` — the canonical writer always emits
    ``ts``, so such a row is real corruption, and silently skipping it
    would shrink the predating history (flipping c23 to SKIP/PASS on a
    stale plan).
    """
    cutoff = plan_mtime_utc + timedelta(seconds=_C23_MTIME_SLACK_S)
    current: str | None = None
    superseded: list[str] = []
    ev = folder / "events.jsonl"
    if ev.exists():
        for line in ev.read_text(encoding="utf-8", errors="replace").split("\n"):
            if not line.strip():
                continue
            if '"epm:goal-updated"' not in line:
                continue  # cheap pre-filter; goal-updated lines parse strictly below
            e = json.loads(line)
            if e.get("kind") != "epm:goal-updated":
                continue
            if not isinstance(e.get("ts"), str):
                raise ValueError(
                    f"malformed epm:goal-updated row in {ev}: missing/non-string 'ts' "
                    f"(the canonical writer always emits ts — this is corruption, "
                    f"not a benign note-only marker): {line!r}"
                )
            ets = datetime.fromisoformat(e["ts"].replace("Z", "+00:00"))
            if ets.tzinfo is None:
                ets = ets.replace(tzinfo=UTC)
            if ets.astimezone(UTC) > cutoff:
                continue
            if isinstance(e.get("from"), str):
                superseded.append(e["from"])
            if isinstance(e.get("to"), str):
                current = e["to"]
    if current is None:
        body = folder / "body.md"
        if body.exists():
            fm, _ = split_frontmatter(body.read_text())
            g = fm.get("goal")
            current = str(g) if g else None
    if current is not None:
        cur_norm = " ".join(_norm_goal_words(current))
        superseded = [s for s in superseded if " ".join(_norm_goal_words(s)) != cur_norm]
    return current, superseded


def check_goal_currency(
    plan: str, *, current_goal: str | None, superseded: list[str]
) -> CheckResult:
    """WARN when the plan head quotes a superseded Goal while the current
    Goal is absent (the #922 stale-quote signature); PASS/SKIP otherwise."""
    cid, name = "c23_goal_currency", "plan head not drafted against a superseded Goal"
    if current_goal is None or len(_norm_goal_words(current_goal)) < _C23_MIN_GOAL_WORDS:
        return _skip(cid, name, "no goal frontmatter / goal too short for shingle matching")
    sup = [s for s in superseded if len(_norm_goal_words(s)) >= _C23_MIN_GOAL_WORDS]
    if not sup:
        return _skip(cid, name, "no superseded Goal predates this plan version")
    head = _plan_head_words(plan)
    cov_cur = _shingle_coverage(current_goal, head)
    cov_stale, stale = max(((_shingle_coverage(s, head), s) for s in sup), key=lambda t: t[0])
    if cov_stale >= _C23_STALE_COV and cov_cur < _C23_CURRENT_COV:
        return _warn(
            cid,
            name,
            f"plan head matches a SUPERSEDED Goal (shingle coverage {cov_stale:.2f}: "
            f"{stale[:100]!r}) while the CURRENT Goal is absent (coverage {cov_cur:.2f}) "
            "— redraft §0.0/§0/§1 against the current `goal:` frontmatter (#922 "
            "plan-vs-goal incident). The orchestrator treats this WARN as a mechanical "
            "redraft bounce (adversarial-planner SKILL.md § Goal-currency gate).",
        )
    return _pass(cid, name, f"coverage: current {cov_cur:.2f}, max superseded {cov_stale:.2f}")


# ─── Check 24 — resume-skip provenance validation (conditional) ─────────────

# Trigger: a per-unit persist + resume-skip pattern. Compound forms ONLY —
# bare "resume" fires on pod lifecycle ("pod.py resume", "resume the poll
# loop") and bare "checkpoint"/"persist" on model checkpoints / the upload
# policy (calibration 2026-07-05: 63 experiment|analysis plan-version hits
# over the 1,367-file v*.md corpus, every spot-checked hit a genuine
# resume-skip loop; zero pod-resume / upload-policy false hits; 28
# non-exp/analysis triggered versions all land on the kind gate).
_C24_TRIGGER_RE = re.compile(
    r"(?i)\b(?:resume[- ]skip|resume[- ]predicate"
    r"|skip[- ]if[- ]exists?"
    r"|skips?\s+(?:already[- ])?(?:completed|done|existing)"
    r"|checkpoint[- ](?:skip|resume)"
    r"|per[- ](?:fold|cell|unit|seed|row|shard)[- ]persist\w*"
    r"|idempotent re-?runs?"
    r"|load[- ]partial[- ]and[- ]skip)"
)

# Satisfier: a recognizable input-fingerprint / provenance token near the
# resume mention. Compound-form discipline on BOTH flanks (plan #1043 v3):
# bare "provenance" and bare "manifest" are deliberately EXCLUDED —
# "Completion provenance:" is a REQUIRED §4-design bullet in every
# experiment plan (on-policy-completions enforcement), so it lands within
# ±15 lines of resume prose and false-satisfied 52% of the v2-calibration
# PASSes (#811 v1, #622 v1-v3, #931 v6 measured; 2026-07-05 reconciler).
# Bare "regime" is likewise EXCLUDED — persona-vectors "read-out regime"
# prose sits inside resume windows (#779 v5 measured) and would
# self-satisfy. The final alternate (assert/validate/verify … existing/
# persisted/… … match) catches contracts phrased without a fingerprint
# noun (#560 v3: "assert the existing file's `sampling.temperature` /
# `sampling_seed` match the requested flags"); it requires a resume-object
# token between verb and "match" so an equivalence-gate assert ("assert
# the vmapped MLP path matches a seeded serial reference", #922 v1-v3
# measured) does NOT satisfy, and its spans are [^\n] (not
# sentence-bounded) because periods inside code tokens like
# `sampling.temperature` break [^.]-spans.
_C24_FINGERPRINT_RE = re.compile(
    r"(?i)\b(?:fingerprints?"
    r"|provenance[- ](?:manifest\w*|contract\w*|validation\w*|check\w*)"
    r"|manifest[- ](?:match\w*|validation\w*|mismatch\w*|check\w*)"
    r"|sha[- ]?256|git[_ -]?sha|code[- ]sha|commit[- ](?:sha|hash)"
    r"|(?:split|content|input|data)[- ]hash(?:es)?"
    r"|env[- ](?:fingerprint|knobs?)"
    r"|regime[- ]key(?:ed|s)?"
    r"|never skip\w* on (?:[\w-]+[- ])?existence"
    r"|(?:assert\w*|validat\w+|verif\w+)[^\n]{0,80}?"
    r"\b(?:existing|persisted|resumed?|cached|stored|prior)\b[^\n]{0,80}?\bmatch\w*)"
)

# Evidence window: ± this many RAW lines around each trigger (the provenance
# contract legitimately lives in an adjacent sentence/table row — #952 v12
# names it in the same bullet; #813 v3 one section over).
_C24_WINDOW_LINES = 15


def check_resume_provenance(plan: str, kind: str) -> CheckResult:
    """WARN-only, conditional: a plan naming a per-unit persist + resume-skip
    pattern must, within ±15 raw lines of SOME resume mention, name an
    input-fingerprint / provenance validation for the resumed outputs (split
    hashes, code SHA, env knobs, a regime-keyed resume predicate, an explicit
    never-skip-on-bare-existence commitment, or an assert-that-the-existing-
    file-matches contract) — never output existence alone. NEVER FAILs in
    v1 — the trigger is a vocabulary heuristic and semantic adequacy of the
    named validation stays with the critics (task #1043 constraint). Known
    accepted gap under WARN-only: a plan that QUOTES this check's WARN
    remedy text near a trigger self-satisfies (the remedy names "split
    hashes, code SHA, env knobs"); the anti-paste guard covers only the N/A
    phrase. Any future WARN→FAIL promotion MUST close that gap first (plan
    #1043 §10 must-ask hook). Incident #952 v9: per-fold persist +
    resume-skip with a bare skips-completed-folds predicate would have let
    stale-fold outputs (or a stale calibration-gate PASS) silently vouch for
    post-code-fix verdict folds; caught only by the critic ensemble (v10
    added the gate-5 provenance-manifest contract). ANY-window semantics per
    the c12 precedent: the contract is typically declared once near one
    mention."""
    cid, name = "c24_resume_provenance", "resume-skip provenance validation"
    if kind not in ("experiment", "analysis"):
        return _skip(
            cid, name, "kind-exempt: resume-provenance is an experiment|analysis plan shape"
        )
    windows = _trigger_windows(plan, _C24_TRIGGER_RE, _C24_WINDOW_LINES)
    if not windows:
        return _skip(cid, name, "no per-unit persist + resume-skip pattern named")
    if _standalone_na_declared(plan, r"no resume\s*[/-]?\s*persist pattern"):
        return _pass(cid, name, "explicit N/A declared (no resume/persist pattern)")
    for window in windows:
        if _C24_FINGERPRINT_RE.search(window):
            return _pass(
                cid,
                name,
                "a resume window names a provenance/fingerprint validation — whether the "
                "named fields (input hashes, code SHA, env) are SUFFICIENT stays critic-owned",
            )
    return _warn(
        cid,
        name,
        "plan names a per-unit persist + resume-skip pattern but no input-fingerprint / "
        "provenance validation near any resume mention (split hashes, code SHA, env knobs, "
        "a regime-keyed resume predicate) — a resume that trusts bare output existence lets "
        "stale units silently vouch after a crash + code-fix round (#952 v9; #722 r3); "
        "name the validation per the #952 gate-5 manifest shape, or declare "
        "'N/A — no resume/persist pattern' on its own line, unwrapped "
        "(no backticks/quotes), if the mention is incidental",
    )


# ─── Check 25 — HTML entities in fenced command blocks (all kinds) ──────────
# The harness HTML-escapes the <result> field of background-Agent
# <task-notification> messages (&& -> the amp-entity form, < -> lt, > -> gt);
# an orchestrator that composes the plan handoff from that text ships a
# poisoned workload command (#952 v9, 2026-07-04: the dispatcher command's
# shell AND operators arrived entity-escaped and needed a hand-fix before
# dispatch would run). This check is the persist-time backstop for the
# capture-time de-escape rule in adversarial-planner SKILL.md.

# Fence pairing: backtick fences only, opener info string captured, closer on
# its own line — the same relaxed-pairing limitation class as the other regex
# checks (corpus-calibrated; exotic 4-backtick nesting is out of scope).
_C25_FENCE_RE = re.compile(r"(?ms)^[ \t]*```([^\n]*)\n(.*?)^[ \t]*```[ \t]*$")

# Arm (a): shell-tagged fences (exemptable by the standalone escape phrase).
_C25_CMD_FENCE_INFO_RE = re.compile(r"(?i)^\s*(?:bash|sh|shell|zsh|console)\b")

# Arm (b): ANY fence (tagged or untagged) whose body carries the
# highest-stakes command markers — never exemptable.
_C25_CMD_MARKER_RE = re.compile(r"--workload-cmd|dispatch_issue\.py")

# The six entity forms (amp/lt/gt/quot + the numeric/hex apostrophes),
# case-insensitive, leading-zero-tolerant on the numeric forms.
_C25_HTML_ENTITY_RE = re.compile(r"(?i)&(?:amp|lt|gt|quot|#0*39|#x0*27);")


def _c25_detail(hits: list[str], *, exemptable: bool) -> str:
    """Render the c25 FAIL detail: entity list + the #952 v9 incident + the
    capture-side remediation; the escape-phrase pointer appears ONLY on the
    ``exemptable=True`` (arm-(a)) branch — an arm-(b) ``--workload-cmd`` /
    ``dispatch_issue.py`` fence is never exemptable (methodology reconciler,
    #1062 round 1)."""
    base = (
        f"fenced command block(s) carry HTML entity form(s) {', '.join(hits)} — the "
        "harness HTML-escapes background-Agent <task-notification> results "
        "(#952 v9, 2026-07-04: the dispatcher command's shell AND operators "
        "arrived entity-escaped); re-extract from the raw output-file, or apply "
        "ONE html.unescape() round to notification-BODY-sourced text, before "
        "persisting"
    )
    if exemptable:
        return base + (
            "; if the fenced entities are deliberately discussed CONTENT (not a "
            "command to dispatch), declare 'N/A — entities are content, not "
            "commands' on its own line, unwrapped (no backticks/quotes) "
            "(valid only when exactly ONE shell-tagged fence carries entity "
            "forms; with several, re-tag content fences to a non-shell info "
            "string or combine them into one fence)"
        )
    return base + (
        " — a --workload-cmd / dispatch_issue.py fence is never exemptable: fix the command text"
    )


def _c25_multi_fence_detail(n_fences: int, hits: list[str]) -> str:
    """Render the c25 FAIL detail for the count-scoped exemption (#1276): the
    standalone escape phrase is present but MORE THAN ONE arm-(a) fence
    carries entity hits — a doc-wide declaration must not let a poisoned
    command fence ride a legitimate content fence's exemption (the arm-(a)
    sibling of the #1062 arm-(b) never-exemptable rule)."""
    return (
        f"{n_fences} distinct shell-tagged fences carry HTML entity form(s) "
        f"{', '.join(hits)}, but the standalone content exemption is scoped to "
        "EXACTLY ONE entity-bearing fence — a doc-wide declaration must not "
        "mask a separately poisoned command fence (#1276; arm-(a) sibling of "
        "the #1062 arm-(b) rule); re-tag genuinely content-bearing fences to a "
        "non-shell info string (e.g. a text-tagged fence, which arm (a) never "
        "scans), or combine the content commands into one fence, or fix the "
        "poisoned command text (re-extract from the raw output-file / one "
        "html.unescape() round)"
    )


def check_html_entities_in_commands(plan: str, kind: str) -> CheckResult:
    """FAIL, ALL kinds, conditional: fenced command blocks must not carry HTML
    entities (#952 v9).

    Two arms: (a) shell-tagged fences (bash/sh/shell/zsh/console) with no
    command marker; (b) ANY fence — tagged or untagged — whose body carries
    ``--workload-cmd`` or ``dispatch_issue.py``. Scan-first; the standalone
    escape phrase (``N/A — entities are content, not commands``, detected via
    the house ``_standalone_na_declared`` line discipline — never a doc-global
    substring) exempts arm-(a) hits ONLY, and only when EXACTLY ONE arm-(a)
    fence carries entity hits — with two or more entity-bearing shell fences
    the declaration cannot bind to a specific fence, so the check FAILs
    naming the fence count (#1276; per-fence grain: distinct fences, not
    distinct entity forms). An arm-(b) entity hit FAILs
    UNCONDITIONALLY — a document-wide phrase must never mask a separately
    poisoned workload command (methodology reconciler, #1062 round 1: one
    legitimate entity-discussing fence + one poisoned dispatcher fence must
    still FAIL). SKIP when the plan has no command fences. All kinds —
    infra/batch plans carry verification commands too (this incident class is
    kind-agnostic). The check ASSERTS; it never rewrites plan text.
    """
    cid, name = "c25_html_entities_in_commands", "no HTML entities in fenced command blocks"
    del kind  # all kinds — infra/batch plans carry verification commands too
    arm_a: list[str] = []
    arm_b: list[str] = []
    for info, body in _C25_FENCE_RE.findall(plan):
        if _C25_CMD_MARKER_RE.search(body):
            arm_b.append(body)  # command-marked: never exemptable
        elif _C25_CMD_FENCE_INFO_RE.match(info):
            arm_a.append(body)  # shell-tagged: exemptable by the phrase
    if not arm_a and not arm_b:
        return _skip(cid, name, "no fenced command blocks detected")
    hits_b = sorted({m.group(0) for b in arm_b for m in _C25_HTML_ENTITY_RE.finditer(b)})
    if hits_b:
        return _fail(cid, name, _c25_detail(hits_b, exemptable=False))
    # Per-fence grain (#1276): the exemption scope counts DISTINCT arm-(a)
    # fences carrying entity hits — never the union of entity forms, which
    # loses fence identity and lets a poisoned fence ride a legitimate
    # fence's declaration (the arm-(a) sibling of the #1062 arm-(b) rule).
    per_fence_hits = [sorted({m.group(0) for m in _C25_HTML_ENTITY_RE.finditer(b)}) for b in arm_a]
    per_fence_hits = [h for h in per_fence_hits if h]
    hits_a = sorted({form for h in per_fence_hits for form in h})
    if hits_a and _standalone_na_declared(plan, r"entities are content, not commands"):
        if len(per_fence_hits) == 1:
            return _pass(
                cid,
                name,
                "arm-(a) entity content exempted by explicit standalone N/A "
                "(single entity-bearing fence)",
            )
        return _fail(cid, name, _c25_multi_fence_detail(len(per_fence_hits), hits_a))
    if hits_a:
        return _fail(cid, name, _c25_detail(hits_a, exemptable=True))
    return _pass(cid, name, f"{len(arm_a) + len(arm_b)} command fence(s), no entity forms")


# ─── Check 26 — GPU basis vs routed machine (WARN-only, conditional) ────────
# Mechanizes .claude/rules/plan-compute-sizing.md § "Cost wall-time against
# the machine the router will ACTUALLY provision" (#599/#833/#1073 class).
# STATIC MIRROR of backends/gcp.py::INTENT_TO_MACHINE at FAMILY grain,
# drift-guarded by tests/test_verify_plan.py::
# test_c26_intent_gpu_mirror_matches_backend — verify_plan_text() stays
# hermetic (no project imports at module level; the only project import in
# this file is the --issue-mode-local task_workflow resolver).
_C26_INTENT_GPU: dict[str, str] = {
    "lora-7b": "A100",
    "lora": "A100",
    "capture-7b": "A100",
    "ft-7b": "A100",
    "eval": "L4",
    "debug": "L4",
    "lora-7b-h100": "H100",
    "eval-h100": "H100",
    "cpu-bigmem": "CPU",
    "cpu-small": "CPU",
    "cpu-mid": "CPU",
    "sweep-8g-a100": "A100",
    "sweep-8g-h100": "H100",
}


def _c26_family(token: str) -> str:
    """GPU family normalization: strip a trailing ``-<digits>`` HBM-size
    suffix (``A100-80`` == ``A100-40`` == ``A100``; ``H100-80`` == ``H100``;
    ``L4``/``CPU`` unchanged). A100-40-vs-A100-80 differences are
    deliberately below the heuristic's grain."""
    return re.sub(r"-\d+$", "", token)


# GPU family tokens ALLOWED in a basis cell trigger. L4/L40S deliberately
# EXCLUDED from the trigger set: #833-style leg labels ("L1/L2 re-extraction,
# L3/L4 extraction") collide with the L4 GPU token; nobody measures bases on
# an L4, while the ROUTED side still knows L4 via the mirror. Included in the
# ESCAPE scan (permissive direction only).
_C26_BASIS_GPU_RE = re.compile(r"\b(H100|H200|A100(?:-[48]0)?|B200)\b")
_C26_ROW_GPU_ANY_RE = re.compile(r"\b(H100|H200|A100(?:-[48]0)?|B200|L40S|L4)\b")

# Scaling vocabulary (row-scoped escape). A bare multiplication sign is NOT
# an escape — it appears in nearly every row's multiplier arithmetic
# ("5,000 x ~300 tok", "draws x cells"); #1073 v3's offending row contains
# one and was still the incident (plan #1075 calibration finding).
_C26_SCALING_RE = re.compile(
    r"(?i)\bscal(?:ed|ing|e factor)\b|per-?step rate|step-?time|rate-?convert"
)

# Intent resolution: --intent <tok> in prose or fences (c5 precedent: RAW
# scan); additionally accepted: the "intent `lora-7b`" prose form
# (#1073 v3 "Target pod preference" shape — capitalized "Intent" in the
# wild, hence (?i)).
_C26_INTENT_RE = re.compile(
    r"(?i)--intent[=\s]+`?([A-Za-z0-9][A-Za-z0-9-]*)|\bintent\s+`([A-Za-z0-9][A-Za-z0-9-]*)`"
)

# Explicit RunPod pin → the RunPod H100/H200 intent table governs; SKIP.
# Scanned RAW (fences included), matching the raw intent scan — a fenced
# `--backend runpod` dispatch line is a real pin; permissive direction only.
_C26_RUNPOD_PIN_RE = re.compile(r"(?i)\bbackend:\s*`?runpod\b|--backend[=\s]+`?runpod\b")


def _c26_intents(plan: str) -> set[str]:
    """Intent tokens resolved from RAW plan text (fences included — a fenced
    dispatch line is the real launch command, the c5 raw-scan precedent).
    Union of the ``--intent <tok>`` flag form (group 1) and the
    ``intent `tok` `` prose form (group 2)."""
    out: set[str] = set()
    for m in _C26_INTENT_RE.finditer(plan):
        tok = m.group(1) or m.group(2)
        if tok:
            out.add(tok)
    return out


def _c26_compute_table_rows(plan: str) -> list[tuple[str, str, str, str]]:
    """``(component_cell, basis_cell, wall_cell, full_row_text)`` for every
    body row of every non-fenced markdown table whose header carries a
    ``basis`` column (a cell that IS or BEGINS WITH the word ``basis``,
    casefolded, bold/backticks stripped — the corpus carries an annotated
    ``basis (measured)`` variant, #952 v12) AND a wall column (fuzzy: any
    header cell CONTAINING ``wall`` — matches ``planned_wall_h`` /
    ``planned wall h`` / ``wall_h`` drift). Header
    detection is fence-masked (a fenced example table is not the plan's
    table — the ``_trigger_windows`` precedent; this deliberately diverges
    from ``_source_column_cells``, which is section-scoped instead: c26
    scans the whole doc because §9 heading text drifts). A row with fewer
    cells than the basis column needs is skipped defensively (the bold
    ``**Base total**`` short-row shape — no IndexError); a short row that
    still reaches the basis column is treated normally with an empty wall
    cell."""
    lines = plan.splitlines()
    mask = _fence_mask(lines)
    rows: list[tuple[str, str, str, str]] = []
    i = 0
    while i < len(lines) - 1:
        header = lines[i].strip()
        sep = lines[i + 1].strip()
        if mask[i] or not (
            header.startswith("|") and sep.startswith("|") and _TABLE_SEP_RE.fullmatch(sep)
        ):
            i += 1
            continue
        header_cells = [c.strip().strip("*`").strip().casefold() for c in _split_table_row(header)]
        basis_col = next((j for j, c in enumerate(header_cells) if re.match(r"basis\b", c)), None)
        wall_col = next((j for j, c in enumerate(header_cells) if "wall" in c), None)
        k = i + 2
        while k < len(lines) and lines[k].strip().startswith("|"):
            if basis_col is not None and wall_col is not None:
                row = _split_table_row(lines[k])
                if basis_col < len(row):
                    component = row[0] if row else ""
                    wall = row[wall_col] if wall_col < len(row) else ""
                    rows.append((component, row[basis_col], wall, lines[k]))
            k += 1
        i = k
    return rows


def _c26_offender_detail(offenders: list[tuple[str, str]], routed: set[str]) -> str:
    """Bounded WARN detail (c13 ``_offender_detail`` precedent): at most 3
    offending rows (component + the offending GPU token), the resolved
    routed families, the #599 incident anchor, and BOTH remedies (a stated
    per-step scaling rate in the row, or the standalone N/A phrase)."""
    shown = "; ".join(f"row {comp[:60]!r} basis names {tok}" for comp, tok in offenders[:3])
    if len(offenders) > 3:
        shown += "; ..."
    return (
        f"{shown} but resolved intent(s) route {sorted(routed)} under auto (GCP "
        "INTENT_TO_MACHINE) with no stated cross-GPU scaling in the row — a basis "
        "measured on a different GPU must be scaled with a stated per-step rate "
        "(plan-compute-sizing.md; #599: an H100-premised ~6.4h estimate ran ~34h on "
        "the A100 auto-lane), or declare 'N/A — basis measured on the routed machine' "
        "on its own line, unwrapped (no backticks/quotes)"
    )


def check_gpu_basis_routed_machine(plan: str, kind: str) -> CheckResult:
    """WARN-only, conditional: a §9 compute-projection-table basis cell naming
    a GPU family (H100/H200/A100/B200) that differs from EVERY family the
    plan's resolved --intent token(s) route under auto (static GCP
    INTENT_TO_MACHINE mirror, _C26_INTENT_GPU), with no row-level escape.
    Mechanizes plan-compute-sizing.md § "Cost wall-time against the machine
    the router will ACTUALLY provision" (#599 ~6.4h -> ~34h; #1073 v3 -> v4).
    Row escapes: (a) the routed family named in a CONVERSION-BEARING cell —
    the wall or basis cell ONLY (a stated conversion names both machines
    there, #1073 v4 wall cell "0.25 (H100) / 0.5-0.6 (A100, x2-2.5)");
    a parallelism/component-cell mention describes the PROVISIONED machine,
    not a conversion, and does NOT escape (plan #1075 Must-Fix M1 — #810 v18
    / #923 v9 rows put "1x A100-80" in parallelism/component cells);
    (b) scaling vocabulary (scaled/per-step rate/...) anywhere in the row.
    NEVER FAILs in v1 — both sides are heuristic (intent resolution from
    text; token matching), and whether a stated scaling factor is CORRECT
    stays critic-owned (c24 precedent). Known accepted gaps: a basis citing
    a prior issue's realized wall WITHOUT naming its GPU (#599's shape)
    is invisible; a "recommended pin: backend: runpod" prose mention
    escapes as if pinned (#779 v6); a conversion stated as a BARE
    multiplier with no vocabulary word ("on H100, x2.5" — #628 v2, the one
    adjudicated calibration FP) still WARNs, because bare-multiplier
    arithmetic saturates compliant AND offending rows alike (#1073 v3) —
    the remedy is one vocabulary word in the row; A100-40 vs A100-80 is
    below the family grain; a routed-family mention in the wall/basis
    cell escapes
    without a true conversion (conversion ADEQUACY stays critic-owned);
    a standalone N/A declaration is document-wide (c24 /
    ``_standalone_na_declared`` family semantics), so it also clears any
    sibling offender row — the deliberate-override purpose of the phrase."""
    cid, name = "c26_gpu_basis_routed_machine", "GPU basis vs routed machine"
    if kind not in ("experiment", "analysis"):
        return _skip(
            cid,
            name,
            "kind-exempt: compute-projection tables are an experiment|analysis plan shape",
        )
    rows = _c26_compute_table_rows(plan)
    if not rows:
        return _skip(cid, name, "no compute-projection table with a `basis` column detected")
    if _standalone_na_declared(plan, r"basis measured on the routed machine"):
        return _pass(cid, name, "explicit N/A declared (basis measured on the routed machine)")
    if _C26_RUNPOD_PIN_RE.search(plan):
        # RAW scan (fences included): a fenced `--backend runpod` dispatch
        # line is a real pin; permissive direction (can only add SKIPs).
        return _skip(
            cid,
            name,
            "explicit backend: runpod pin — the RunPod intent table governs the basis machine",
        )
    routed = {_C26_INTENT_GPU[i] for i in _c26_intents(plan) if i in _C26_INTENT_GPU}
    if not routed:
        return _skip(
            cid,
            name,
            "no resolvable --intent token — routed machine unknown (auto-lane GPU cannot "
            "be inferred)",
        )
    offenders: list[tuple[str, str]] = []
    for component, basis, wall, row_text in rows:
        hit = _C26_BASIS_GPU_RE.search(basis)
        if not hit or _c26_family(hit.group(1)) in routed:
            continue
        # Escape (a): routed family named in a CONVERSION-BEARING cell only
        # (wall + basis) — NOT parallelism/component (Must-Fix M1).
        conv_cells = f"{basis} {wall}"
        conv_families = {_c26_family(m.group(1)) for m in _C26_ROW_GPU_ANY_RE.finditer(conv_cells)}
        if conv_families & routed or _C26_SCALING_RE.search(row_text):
            continue
        offenders.append((component, hit.group(1)))
    if not offenders:
        return _pass(
            cid,
            name,
            f"{len(rows)} table row(s); no unscaled cross-GPU basis vs routed {sorted(routed)}",
        )
    return _warn(cid, name, _c26_offender_detail(offenders, routed))


# ─── Check 27 — 7B activation capture vs eval/debug (L4) intent ─────────────
# Mechanizes .claude/rules/plan-compute-sizing.md § "Activation-capture HBM
# sizing" (MUST-level): a 7B hidden-state capture phase needs >=40 GB HBM
# (capture-7b / lora-7b, 1x A100-80); the GCP eval/debug default is a
# 16-GB-class L4 (g2-standard-4) and OOMs mid-run (#666, #744). Founding
# false negative: #825 plan v17 (--intent eval for a 7B all-layer capture)
# PASSed 0 FAIL/0 WARN. Reuses c26's intent machinery — one parser, one
# drift-guarded mirror (test_c26_intent_gpu_mirror_matches_backend).

# Offending + absolving intent sets, DERIVED from the c26 mirror.
# BIG set derived by EXCLUSION (critique r1, Claude methodology concern 1):
# a future mirror family (H200/B200 intent) lands in the absolution set
# automatically instead of silently outside it (which would false-FAIL a
# plan booking that big intent alongside a side eval phase). Test
# test_c27_sets_derive_from_mirror's partition assert pins
# L4 | BIG | CPU == the whole mirror.
_C27_L4_INTENTS: frozenset[str] = frozenset(i for i, fam in _C26_INTENT_GPU.items() if fam == "L4")
_C27_BIG_HBM_INTENTS: frozenset[str] = frozenset(
    i for i, fam in _C26_INTENT_GPU.items() if fam not in ("L4", "CPU")
)

# Capture-phase vocabulary (RAW scan — capture launch commands and store
# rows legitimately live in fences/tables; the _c26_intents raw-scan
# precedent). Anchored compounds only: bare "extraction"/"capture" false-
# fire on prose ("extraction set", "capture the behavior"). Calibrated
# 2026-07-07 over 1,511 persisted plans: 5/5 known offender tasks flagged
# (#667/#744/#761/#810/#825), zero false positives.
_C27_CAPTURE_RE = re.compile(
    r"(?i)hidden[-_ ]states?\b"
    r"|activations?[-_ ]?(?:store|captur\w*|extract\w*|accumulat\w*|dump\w*)"
    r"|\bextract_store\b"
    r"|residual[-_ ]stream"
    r"|\bcaptur\w+\s+(?:\w+\s+)?activations?\b"
)

# >=7B model-size signal (the HBM rule is 7B-scoped; sub-7B captures fit L4).
# THRESHOLD semantics, not a whitelist (critique r1, all three Codex lenses):
# integer part >= 7 — single digit 7-9, or any 2+ digit number — with an
# optional decimal tail. The negative lookbehind (?<![\d.]) blocks the
# decimal-tail false positive ("1.7B"/"2.5B"/"6.9B" never match: the digit
# before the dot fails both integer alternates, and the digit after the dot
# is lookbehind-blocked). "17B" DOES match under threshold semantics
# (17 >= 7 — a deliberate deviation from the r1 Codex test sketch, which
# carried over the old whitelist's behavior). Token-count strings ("15B
# tokens") can match — acceptable: the conjunction still needs capture
# vocabulary + an un-skipped eval/debug booking, and the corpus re-scan
# gate (plan #1093 §13) binds on any regex change.
_C27_MODEL_GE7B_RE = re.compile(r"(?i)(?<![\d.])\b(?:[7-9]|[1-9][0-9]+)(?:\.[0-9]+)?B\b")

# scripts/pod.py IS the RunPod lifecycle CLI, where eval provisions
# 1x H100 80GB (CLAUDE.md intent table) — no HBM gap. Document-wide,
# permissive direction only (adds SKIPs); the _C26_RUNPOD_PIN_RE sibling
# for the pre-router plan corpus (#358/#375/#522 era).
_C27_PODPY_PROVISION_RE = re.compile(r"(?i)\bpod\.py\s+provision\b")

# Window-level big-GPU skip: an eval/debug token whose immediate context
# names H100/H200 is a RunPod-mapping or explicit-override claim, not a
# GCP L4 booking. A100 deliberately NOT in the skip set: GCP eval/debug
# NEVER provisions A100 — an A100 claim next to an eval booking is exactly
# the #744 misbelief this check exists to catch.
_C27_WINDOW_BIGGPU_RE = re.compile(r"\b(H100|H200)\b")


def _c27_gcp_l4_intent_windows(plan: str) -> list[tuple[str, str]]:
    r"""``(token, window_snippet)`` for every eval/debug intent occurrence
    plausibly booking the GCP/auto lane. The window is the PREVIOUS line
    plus the line containing the match end — the previous line covers the
    wrapped ``pod.py provision --issue N --intent\neval`` shape (#522 v1,
    where ``--intent[=\s]+`` legitimately spans the newline). A window
    carrying ``pod.py`` or an H100/H200 token is skipped (RunPod / explicit
    big-GPU context)."""
    out: list[tuple[str, str]] = []
    for m in _C26_INTENT_RE.finditer(plan):
        tok = m.group(1) or m.group(2)
        if tok not in _C27_L4_INTENTS:
            continue
        line_start = plan.rfind("\n", 0, m.start())
        prev_start = plan.rfind("\n", 0, line_start) if line_start != -1 else -1
        win_end = plan.find("\n", m.end())
        window = plan[prev_start + 1 : len(plan) if win_end == -1 else win_end]
        if "pod.py" in window or _C27_WINDOW_BIGGPU_RE.search(window):
            continue
        out.append((tok, " ".join(window.split())[:90]))
    return out


def check_capture_intent_hbm(plan: str, kind: str) -> CheckResult:
    """FAIL (experiment) / WARN (analysis), conditional: activation-capture
    vocabulary + a >=7B model signal while an eval/debug (L4) intent is
    booked on the GCP/auto lane. Skip ladder (permissive direction only):
    kind gate -> vocab trigger -> standalone N/A escape -> RunPod pin
    (backend/--backend runpod OR pod.py provision, doc-wide: RunPod eval =
    1x H100 80GB) -> no resolvable intent -> no un-windowed eval/debug
    occurrence -> big-HBM-intent absolution -> no >=7B signal.
    Known accepted gaps (all deliberate, critic-owned semantics):
    (a) a plan booking a big-HBM intent for training while the CAPTURE
    phase books eval escapes via the absolution — phase-to-intent routing
    stays critic-owned; (b) an eval occurrence whose window names H100/H200
    (e.g. a basis-measured-on-H100 clause on the same line) escapes as if
    pinned — c26 covers the basis side; (c) a doc-wide pod.py-provision pin
    skips mixed-lane plans; (d) the >=7B signal matches "7b" inside intent
    tokens (lora-7b) — a weak filter by design, the N/A phrase is the real
    small-model out; (e) vocabulary from a REUSED store consumed by a CPU
    phase still triggers — cleared by the no-L4-intent PASS, the
    absolution, or the N/A phrase."""
    cid, name = "c27_capture_intent_hbm", "7B capture vs eval/debug intent"
    if kind not in ("experiment", "analysis"):
        return _skip(cid, name, "kind-exempt: capture phases are an experiment|analysis plan shape")
    cap_hit = _C27_CAPTURE_RE.search(plan)
    if not cap_hit:
        return _skip(cid, name, "no activation-capture vocabulary detected")
    if _standalone_na_declared(plan, r"no 7B activation capture"):
        return _pass(cid, name, "explicit N/A declared (no 7B activation capture)")
    if _C26_RUNPOD_PIN_RE.search(plan) or _C27_PODPY_PROVISION_RE.search(plan):
        return _skip(
            cid, name, "explicit RunPod pin/provision — RunPod eval = 1x H100 80GB, no HBM gap"
        )
    if not _c26_intents(plan):
        return _skip(cid, name, "no resolvable --intent token — routed machine unknown")
    windows = _c27_gcp_l4_intent_windows(plan)
    if not windows:
        return _pass(
            cid,
            name,
            "capture vocabulary present but no eval/debug intent booked on the GCP/auto lane",
        )
    big = sorted(_c26_intents(plan) & _C27_BIG_HBM_INTENTS)
    if big:
        return _pass(
            cid,
            name,
            f">=40 GB-HBM intent also booked ({big}) — capture phase presumed routed there "
            "(phase-to-intent routing stays critic-owned)",
        )
    if not _C27_MODEL_GE7B_RE.search(plan):
        return _skip(cid, name, "no >=7B model signal — the HBM sizing rule is 7B-scoped")
    tok, snippet = windows[0]
    verdict = _fail if kind == "experiment" else _warn
    return verdict(
        cid,
        name,
        f"capture vocabulary ({cap_hit.group(0)!r}) with a >=7B model while the plan books the "
        f"{tok} (L4, g2-standard-4, 16-GB-class HBM) intent on the GCP/auto lane "
        f"(context: {snippet!r}) — >=7B hidden-state capture needs >=40 GB HBM "
        "(#666/#744 OOM class; #825 v17 false negative): for a 7B-class model book capture-7b "
        "(forward-pass-only) or lora-7b (phase also trains); a LARGER model needs a "
        "correspondingly larger-HBM lane/backend (a multi-GPU intent or an explicit "
        "large-GPU RunPod pin), never eval/debug — per plan-compute-sizing § "
        "Activation-capture HBM sizing, or declare 'N/A — no 7B activation capture' "
        "on its own line, unwrapped (no backticks/quotes)",
    )


# ─── Check 28 — decision-band precedent coherence (WARN-only) ───────────────
# Mechanizes the #825 v17 incident class: a registered fractional decision
# band ("cmp T x" inside a success/kill/decision section), applied to the
# plan's OWN quoted precedent ratio(s), must land in the branch the plan's
# narrative asserts. Prose siblings: planner-section-reference.md §7
# (precedent self-check bullet) + critic-lens-reference.md Statistics item
# 3 trigger (c) — the FAIL-grade semantic verdict stays critic-side.

# Band line: a non-fenced, bold-labeled list item inside a decision-keyword
# section (_C13_GATE_SECTION_RE reused) carrying a multiplicative threshold
# "cmp T x" with fractional T (0 < T <= 1) — the #931 committed-threshold
# idiom ("< 0.5 × 0.588", "≥ 0.5× its ceiling"). Integer / super-unity T  # noqa: RUF003
# ("≥ 2× wall-time" kill fences) deliberately excluded: fraction-of-ceiling  # noqa: RUF003
# bands are the target class; wall-time multipliers are a different quantity.
_C28_BAND_RE = re.compile(
    r"(?P<cmp><=|>=|[<>≤≥])\s*(?P<thr>0?\.\d+|1\.0|1)\s*[×x](?![a-zA-Z])"  # noqa: RUF001
)
_C28_LIST_ITEM_RE = re.compile(r"^\s{0,3}(?:[-*]|\d+\.)\s")  # c14 sibling
_C28_BOLD_LABEL_RE = re.compile(r"\*\*([^*\n]{1,60})\*\*")  # c14 sibling

# Precedent-ratio assertion: explicit "ratio ≈ r" / "ratio ≈ r1–r2" token.  # noqa: RUF003
# Decimal point REQUIRED (excludes the "ratio ~1:1" mix idiom — the only 2
# non-incident same-line corpus hits); `%`-suffixed ratios NOT harvested —
# single (`0.48%`) AND range (`0.44–0.52%`) forms both harvest NOTHING  # noqa: RUF003
# (percent-vs-fraction confusion is a named FP mode — accepted false
# negative). The \b after each number blocks a backtracked partial-digit
# match like `0.4` inside `0.48%`; the second lookahead rejects r1 when the
# engine SKIPS a %-suffixed optional range group — `(?!\s*%)` alone let
# `ratio ≈ 0.44–0.52%` partially harvest r1=0.44 (round-2 fix, concern  # noqa: RUF003
# c28-percent-range-partial-harvest).
_C28_RATIO_RE = re.compile(
    r"(?i)\bratios?\s*[≈=~]\s*(?P<r1>0?\.\d+)\b"
    r"(?:\s*[–—-]\s*(?P<r2>0?\.\d+)\b)?(?!\s*%)"  # noqa: RUF001 — en dash is real plan text
    r"(?!\s*[–—-]\s*0?\.\d+\s*%)"  # noqa: RUF001 — reject a %-suffixed skipped range
)
# Verb-anchored side vocabulary (navigation "see below" / "table below"
# cannot match: a verb is required). The negation guard drops the WHOLE
# line on any negated side phrase ("not/never well below") — a LINE-level
# kill, not instance-level: a mixed line ("not below X but above Y") is
# dropped entirely (accepted false negative — prefer false negatives, the
# c14 doctrine).
_C28_BELOW_RE = re.compile(r"(?i)\b(?:well|lands?|sits?|stays?|falls?)\s+below\b|\bunder\s+half\b")
_C28_ABOVE_RE = re.compile(
    r"(?i)\bexceeds?\b|\b(?:well|lands?|sits?|stays?)\s+above\b|\bat\s+least\s+half\b"
)
_C28_NEG_RE = re.compile(
    r"(?i)\b(?:not|never|no\s+longer)\s+(?:(?:well|lands?|sits?|stays?|falls?)\s+)?"
    r"(?:below|above)\b|\bexcept\s"
)
# Same-line recompute corroborator: positive decimals split at the first
# `vs`; slash (`/`) is NOT a ratio separator in this corpus (it is the
# paired-cells idiom: "rotated +0.349/+0.334"). 2-4 fractional digits so a
# coarse "vs chat 0.6" drops the corroborator (quoted-ratio path unaffected).
_C28_VS_RE = re.compile(r"(?i)\bvs\.?\s")
_C28_POSNUM_RE = re.compile(r"(?<![\d.\-])\+?(0?\.\d{2,4})\b")


def _c28_frac(s: str) -> Fraction:
    """Exact ``Fraction`` from a decimal literal, tolerating a bare leading
    dot (``.5`` -> 1/2) — the c13 ``_c13_registered_gates`` parse
    convention."""
    return Fraction("0" + s) if s.startswith(".") else Fraction(s)


def _c28_bands(plan: str) -> list[dict]:
    """Registered multiplicative decision-band lines: non-fenced,
    bold-labeled list items inside a success/kill/decision/evaluation-titled
    section (``_C13_GATE_SECTION_RE`` reused; the #825 v17 heading
    "## 6. Success + kill criteria (quantitative)" matches via the
    ``kill[- ]criteri`` alternation) carrying a ``cmp T x`` threshold with
    fractional T in (0, 1]. Per band: ``{label, cmp, thr: Fraction, line}``
    — a mirror of ``_c13_registered_gates``' fence-masked, section-scoped
    walk."""
    lines = plan.splitlines()
    mask = _fence_mask(lines)
    headings = _headings(plan)
    bands: list[dict] = []
    for i, (line, fenced) in enumerate(zip(lines, mask, strict=True)):
        if fenced:
            continue
        if not any(h.line <= i < h.end and _C13_GATE_SECTION_RE.search(h.text) for h in headings):
            continue
        if not _C28_LIST_ITEM_RE.match(line):
            continue
        label_m = _C28_BOLD_LABEL_RE.search(line)
        band_m = _C28_BAND_RE.search(line)
        if not (label_m and band_m):
            continue
        thr = _c28_frac(band_m.group("thr"))
        if not 0 < thr <= 1:
            continue
        bands.append(
            {
                "label": label_m.group(1).strip(),
                "cmp": band_m.group("cmp"),
                "thr": thr,
                "line": line.strip(),
            }
        )
    return bands


def _c28_ratio_assertions(plan: str) -> list[dict]:
    """Side-asserted precedent-ratio lines over non-fenced text. A line
    fires only when an explicit ``ratio ≈ r[-r2]`` token AND exactly one
    side (below XOR above vocabulary, negation-guarded at LINE level)
    co-occur on it. Per assertion:
    ``{line, side, side_text, quoted, recomputed, candidates}`` where
    ``candidates`` = quoted r1 (and r2 for a range) UNION the same-line
    vs-pair recompute — numerators are the positive decimals LEFT of the
    first ``vs`` (ratio-token spans blanked first), the denominator is the
    FIRST positive decimal RIGHT of it; a/b kept only when b > 0 (the
    zero-denominator guard — the ``Fraction(x, 0)`` class c13's detail
    builder documents) and 0 < a/b <= 2 (sanity window). ``Fraction``
    arithmetic throughout — exact boundary semantics at r == T, no
    float-equality wobble."""
    lines = plan.splitlines()
    mask = _fence_mask(lines)
    assertions: list[dict] = []
    for line, fenced in zip(lines, mask, strict=True):
        if fenced:
            continue
        ratio_ms = list(_C28_RATIO_RE.finditer(line))
        if not ratio_ms or _C28_NEG_RE.search(line):
            continue
        below_m = _C28_BELOW_RE.search(line)
        above_m = _C28_ABOVE_RE.search(line)
        if bool(below_m) == bool(above_m):  # neither, or both (ambiguous)
            continue
        side_m = below_m or above_m
        quoted = {_c28_frac(m.group(g)) for m in ratio_ms for g in ("r1", "r2") if m.group(g)}
        blanked = list(line)
        for m in ratio_ms:
            blanked[m.start() : m.end()] = " " * (m.end() - m.start())
        blanked_line = "".join(blanked)
        recomputed: set[Fraction] = set()
        vs_m = _C28_VS_RE.search(blanked_line)
        if vs_m:
            denom_m = _C28_POSNUM_RE.search(blanked_line[vs_m.end() :])
            if denom_m:
                b = _c28_frac(denom_m.group(1))
                if b > 0:
                    for num_m in _C28_POSNUM_RE.finditer(blanked_line[: vs_m.start()]):
                        r = _c28_frac(num_m.group(1)) / b
                        if 0 < r <= 2:
                            recomputed.add(r)
        assertions.append(
            {
                "line": line.strip(),
                "side": "below" if below_m else "above",
                "side_text": side_m.group(0),  # type: ignore[union-attr]
                "quoted": quoted,
                "recomputed": recomputed,
                "candidates": quoted | recomputed,
            }
        )
    return assertions


def _c28_na_escape_declared(plan: str) -> bool:
    """Standalone ``N/A — no precedent-labeled decision bands`` escape (see
    ``_standalone_na_declared`` for the anti-paste rationale; British
    ``labelled`` accepted)."""
    return _standalone_na_declared(plan, r"no precedent[- ]labell?ed decision bands?")


def _c28_landed_band_label(bands: list[dict], landed_ge: bool) -> str:
    """Label of the first band whose comparator points at the branch every
    candidate lands in (the >= T branch when ``landed_ge``, the < T branch
    otherwise), or ``""`` when no band's comparator points that way."""
    wanted = (">", ">=", "≥") if landed_ge else ("<", "<=", "≤")
    for b in bands:
        if b["cmp"] in wanted:
            return b["label"]
    return ""


def _c28_offender_detail(offenders: list[tuple[dict, str]], T: Fraction, bands: list[dict]) -> str:
    """Bounded WARN detail (c13 conventions: at most 3 offenders shown,
    90-char line snippets): per offender the line snippet, the asserted
    side phrase, the quoted vs recomputed candidate ratios (rendered
    ``≈ 0.519``), T, the disagreement class (contradiction | straddle) and
    the branch placement — a CONTRADICTION names the single band label the
    candidates land in; a STRADDLE always reads "candidates span both
    branches of T", never a single landed-band label. Ends with the #825
    v17 incident anchor, the cross-quantity honesty clause, and the remedy
    menu."""

    def _render(vals: set[Fraction]) -> str:
        return ", ".join(f"≈ {float(v):.3g}" for v in sorted(vals))

    parts: list[str] = []
    for a, cls in offenders[:3]:
        cands = f"quoted {_render(a['quoted'])}"
        if a["recomputed"]:
            cands += f" + recomputed {_render(a['recomputed'])}"
        if cls == "straddle":
            placement = "candidates span both branches of T"
        else:
            landed_ge = a["side"] == "below"
            label = _c28_landed_band_label(bands, landed_ge)
            branch = "≥ T" if landed_ge else "< T"
            placement = f"every candidate lands in the {branch} branch" + (
                f" ({label!r})" if label else ""
            )
        parts.append(
            f"line \"{a['line'][:90]}\" asserts '{a['side_text']}' but {cands} against the "
            f"registered {float(T):g}× band → {cls} ({placement})"  # noqa: RUF001
        )
    shown = "; ".join(parts)
    if len(offenders) > 3:
        shown += "; …"
    return (
        f"{shown} — a decision band applied to the plan's OWN cited precedent must land in "
        "the branch the narrative assigns it (#825 v17: 0.349/0.673 ≈ 0.519 ≥ 0.5 narrated "
        "'lands well below'; verify_plan PASSed 0/0 — caught only at the critic layer). "
        "NOTE: this check cannot verify the ratio and the band concern the same quantity — "
        "if they don't, declare the N/A escape. Remedy: re-label the precedent's branch, "
        "move the threshold, or declare 'N/A — no precedent-labeled decision bands' on its "
        "own line, unwrapped (no backticks/quotes); the semantic verdict stays with the "
        "Statistics critic"
    )


def check_precedent_band_coherence(plan: str, kind: str) -> CheckResult:
    """WARN-only, conditional: a registered fractional decision band
    (``cmp T x`` in a success/kill/decision section), applied to a
    same-line side-asserted precedent ratio the plan itself quotes or
    implies (the vs-pair recompute), must land in the branch the narrative
    asserts. A straddle (a quoted range containing both sides of T while
    one side is asserted) also WARNs. Boundary convention: below := [0, T)
    hardcoded — the harvested band comparator is NOT consulted at r == T,
    so a ``<=``-band's r == T edge is an accepted WARN-only imprecision.
    NEVER FAILs (the c14 doctrine: a heuristic text check must not
    hard-block a legitimately-worded plan); the FAIL-grade semantic verdict
    stays with the Statistics critic (critic-lens-reference.md item 3
    trigger (c)). Accepted false negatives (v1; plan #1094 §4.4): plain
    absolute ``a >= c`` comparisons (cross-arm absolutes are unsound when
    precedent and design arms have different ceilings), multi-threshold
    plans (SKIP), side assertions in an adjacent sentence rather than on
    the ratio line, `%`-suffixed ratios, and `/`-separated ratios (the
    paired-cells idiom). Incident: #825 v17 (the 0.5x band vs its cited
    instruct precedent 0.3489/0.6731 = 0.519, narrated 'lands well below';
    caught only at the critic layer — verify_plan PASSed 0/0)."""
    cid, name = "c28_precedent_band_coherence", "decision-band precedent coherence"
    if kind not in ("experiment", "analysis"):
        return _skip(
            cid,
            name,
            "kind-exempt: precedent-labeled decision bands are an experiment|analysis plan shape",
        )
    bands = _c28_bands(plan)
    if not bands:
        return _skip(cid, name, "no registered multiplicative decision band detected")
    if _c28_na_escape_declared(plan):
        return _pass(cid, name, "explicit N/A declared (no precedent-labeled decision bands)")
    thresholds = {b["thr"] for b in bands}
    if len(thresholds) != 1:
        return _skip(
            cid,
            name,
            f"{len(thresholds)} distinct band thresholds — precedent-to-band pairing "
            "ambiguous at the plan surface",
        )
    T = next(iter(thresholds))
    assertions = _c28_ratio_assertions(plan)
    if not assertions:
        return _skip(cid, name, "band present but no side-asserted precedent ratio line detected")
    offenders: list[tuple[dict, str]] = []
    for a in assertions:
        lo, hi = min(a["candidates"]), max(a["candidates"])
        if a["side"] == "below" and hi >= T:
            offenders.append((a, "contradiction" if lo >= T else "straddle"))
        elif a["side"] == "above" and lo < T:
            offenders.append((a, "contradiction" if hi < T else "straddle"))
    if not offenders:
        return _pass(
            cid,
            name,
            f"{len(assertions)} side-asserted precedent ratio line(s) coherent with the "
            f"registered {float(T):g}× band",  # noqa: RUF001
        )
    return _warn(cid, name, _c28_offender_detail(offenders, T, bands))


# ─── Check 29 — deliberate fence vs §7 conditional phase ────────────────────
# Mechanizes .claude/rules/plan-compute-sizing.md § "reconcile the WORST-CASE
# wall — base phases PLUS every conditional / extension phase that could run
# on the same provision — against the GCP lane's auto-delete fence": a
# deliberately-declared (value-bearing, non-default) --max-run-duration /
# max_run_duration fence coexisting with a §7 extension/retrain-class gate
# must reference that conditional phase near a declaration site. Founding
# offender: #1112 v2 (a 48h fence sized off base phases only, omitting its
# own §7 G1 dose extension — joint worst ~48-50h; caught only by the critic
# layer; v3 costs the extension and bumps the fence to 72h).

# Value-bearing deliberate fence declaration, RAW scan (fences included — a
# fenced gcloud/dispatch line is the real launch command; the c5/c26
# raw-scan precedent). A value of 7d/168h is the default FLEX_START ceiling
# (#741), not "deliberate"; a minutes value is the cap-probe command shape
# (#680 `--max-run-duration=20m` — unit not in h/d, so it never matches); a
# bare flag, a "default 7d" prose mention (no value directly after the
# flag), and a templated `={max_run}`/`<dur>` placeholder carry no value —
# none trigger. A loose "Nh near the flag" prose trigger is deliberately
# absent: #1112 v2's §0 line ("the 48 h `--max-run-duration` fence") sits 2
# lines from a Risks line containing "dose extension", so a prose trigger
# would self-satisfy and the founding offender would PASS.
_C29_FENCE_FLAG_RE = re.compile(
    r"(?i)--max-run-duration[=\s]+[`\"']?~?(\d+(?:\.\d+)?)\s*(h(?:ours?)?|d(?:ays?)?)\b"
)
_C29_FENCE_EXTRA_RE = re.compile(
    r"(?i)max_run_duration[\"']?\]?\s*[:=]\s*[\"'`]?~?(\d+(?:\.\d+)?)\s*"
    r"(h(?:ours?)?|d(?:ays?)?)\b"
)
# §7-slot / Decision-Gates heading predicate (heading levels >= 2).
# Deliberately permissive on the numbered form: the §7 slot also holds
# `Compute estimate` / `Risks` in infra-shaped plans — the extension-vocab
# gate below filters those (WARN-only polarity; calibration-swept, #1114).
_C29_SECT7_HEAD_RE = re.compile(r"(?i)^(?:§\s*)?7\b(?:[.:)\s]|$)|\bdecision gates?\b")
# Extension-class gate vocabulary. Bare "resume"/"re-run"/"re-judge" are
# deliberately EXCLUDED (crash-resume vocabulary saturates plans); a gate
# worded purely as "resume to step 60" is a named accepted false negative.
_C29_EXTENSION_RE = re.compile(
    r"(?i)\b(?:dose[- ]extension|extension|extend(?:s|ed|ing)?|re-?ladder\w*|"
    r"re-?train\w*|retrain\w*|second pass|additional (?:steps|pass(?:es)?|epochs?))\b"
)
# Conditional-cost evidence vocabulary (permissive direction: a match can
# only suppress a WARN). Gate labels (G1, G2, ...) are matched separately,
# case-SENSITIVE, only for labels actually harvested from §7 — a (?i)
# \bg\d+\b would match GCP machine types ("g2-standard-4").
_C29_EVIDENCE_RE = re.compile(
    r"(?i)§\s*7\b|\bsection\s+7\b|\b(?:extension|extend\w*|contingen\w*|conditional|"
    r"gate(?:'s|s)?|dose[- ]extension|re-?ladder\w*|re-?train\w*|retrain\w*|"
    r"second provision|across provisions|split across)\b"
)
_C29_WINDOW_LINES = 3  # pinned on BOTH sides: test_c29_evidence_outside_window_still_warns
# (upper bound: distance 4 WARNs) + test_c29_evidence_at_window_edge_passes
# (lower bound: distance 3 PASSes — kills a narrowing mutant).


def _c29_hours(val: str, unit: str) -> float:
    """Fence value normalized to hours (d/days -> x24; units pre-filtered
    to h/d by the declaration regexes)."""
    return float(val) * (24.0 if unit.lower().startswith("d") else 1.0)


def _c29_fence_decl_line_idxs(plan: str) -> list[int]:
    """RAW line indices carrying a value-bearing ``--max-run-duration`` /
    ``max_run_duration`` declaration whose value is not the 7d/168h default
    FLEX_START ceiling (#741). RAW scan — fences included (a fenced
    gcloud/dispatch line is the real launch command; c5/c26 precedent)."""
    idxs: list[int] = []
    for i, line in enumerate(plan.splitlines()):
        for rx in (_C29_FENCE_FLAG_RE, _C29_FENCE_EXTRA_RE):
            m = rx.search(line)
            if m and abs(_c29_hours(m.group(1), m.group(2)) - 168.0) > 1e-9:
                idxs.append(i)
                break
    return idxs


def _c29_gate_section_prose(plan: str) -> str | None:
    """Fence-masked prose of every §7-slot / Decision-Gates section (heading
    levels >= 2), joined across all matches; ``None`` when no such heading
    exists. The global ``_fence_mask`` excludes fenced example commands
    inside §7 — a gate is a prose contract (the ``_trigger_windows``
    fence-masked-trigger doctrine)."""
    lines = plan.splitlines()
    mask = _fence_mask(lines)
    parts: list[str] = []
    found = False
    for h in _headings(plan):
        if h.level < 2 or not _C29_SECT7_HEAD_RE.search(h.text.strip()):
            continue
        found = True
        parts.extend(
            line
            for line, fenced in zip(
                lines[h.line + 1 : h.end], mask[h.line + 1 : h.end], strict=True
            )
            if not fenced
        )
    return "\n".join(parts) if found else None


def _c29_offender_detail(decl_line: str, gate_hit: str, labels: list[str]) -> str:
    """Bounded WARN detail (c26 conventions): the first declaration line
    (truncated ~80 chars), the matched §7 extension vocabulary + harvested
    gate labels, the incident anchors, and BOTH remedies."""
    lab = f"; §7 gate label(s): {', '.join(labels)}" if labels else ""
    return (
        f"deliberate fence declaration {decl_line.strip()[:80]!r} coexists with a §7 "
        f"extension-class gate (matched {gate_hit!r}{lab}) but no declaration window "
        "references the conditional phase's wall cost — reconcile the WORST-CASE wall, "
        "base phases PLUS every conditional/extension phase on the same provision, "
        "against the fence (plan-compute-sizing.md § worst-case wall; #599: a 24h fence "
        "hard-deleted the pre-registered §7.3 extension probe at step 149/2400; #1112 "
        "v2: a 48h fence omitted its own §7 G1 dose extension, joint worst ~48-50h). "
        "Remedy: add the conditional phase's wall cost to the fence-reconcile sentence, "
        "or declare 'N/A — no conditional phase on this provision' on its own line, "
        "unwrapped (no backticks/quotes)"
    )


def check_fence_conditional_phase(plan: str, kind: str) -> CheckResult:
    """WARN-only, conditional: a deliberately-declared ``--max-run-duration``
    / ``max_run_duration`` fence (value-bearing, != the 7d/168h default
    ceiling) coexisting with an extension/retrain-class gate in the §7 /
    Decision-Gates section must reference that conditional phase (a §7 gate
    label, or conditional-cost vocabulary) within ±3 raw lines of a
    declaration line — ANY-SITE satisfy: one declaration window carrying
    the evidence clears the whole plan (the reconcile sentence is singular;
    requiring every mention would WARN on compressed §0 summaries).
    Fence-strip split: declaration scan RAW (the fence usually lives in a
    backticked/fenced launch command, and a fenced-only declaration with
    zero prose reconcile is exactly the silent-ride failure class); §7 gate
    detection fence-masked; evidence windows RAW (permissive direction).
    Mechanizes plan-compute-sizing.md § "reconcile the WORST-CASE wall —
    base phases PLUS every conditional / extension phase" (#599: a 24h
    fence hard-deleted the pre-registered §7.3 extension probe at step
    149/2400; #833: per-cell dispersion overran a deliberate 36h fence;
    #1112 v2: a 48h fence sized off base phases omitted its own §7 G1 dose
    extension, joint worst ~48-50h — only the critic caught it). NEVER
    FAILs (the c14/c28 doctrine). SCOPE (honest): mechanizes the
    DECLARED-fence subclass (#1112-shaped) of the incident class only.
    Known accepted gaps, each verified against the founding files: a
    prose-only fence (the actual #599 shape — "GCP max-run-duration
    (~20 h)", no flag/assignment) is invisible -> SKIP; a dispatch-time
    fence never written into the plan (the actual #833 shape) is invisible
    -> SKIP; bare resume/re-run/re-judge gates don't trigger; a
    second-provision split pre-registered ONLY in §7 still WARNs (remedy:
    the N/A phrase or a fence-window mention); evidence-vocabulary stray
    matches (e.g. an unrelated "gate" near the fence) suppress a real WARN
    — permissive direction; whether the referenced conditional cost is
    ARITHMETICALLY correct stays critic-owned. The #599/#833 SKIP shapes
    are pinned by tests (test_c29_prose_only_fence_skips /
    test_c29_no_fence_skips) plus the #1114 §6 sibling replay, so a future
    trigger widening fails loud."""
    cid, name = "c29_fence_conditional_phase", "deliberate fence vs §7 conditional phase"
    if kind not in ("experiment", "analysis"):
        return _skip(
            cid, name, "kind-exempt: fence/§7-gate shapes are an experiment|analysis plan shape"
        )
    decl_idxs = _c29_fence_decl_line_idxs(plan)
    if not decl_idxs:
        return _skip(
            cid,
            name,
            "no deliberate (value-bearing, non-default) --max-run-duration fence "
            "declaration detected",
        )
    if _standalone_na_declared(plan, r"no conditional phase on this provision"):
        return _pass(cid, name, "explicit N/A declared (no conditional phase on this provision)")
    gates = _c29_gate_section_prose(plan)
    if gates is None:
        return _skip(cid, name, "no §7 / Decision Gates section detected")
    ext = _C29_EXTENSION_RE.search(gates)
    if not ext:
        return _skip(cid, name, "no extension/retrain-class conditional gate in §7")
    labels = sorted(set(re.findall(r"\bG\d+\b", gates)))
    lines = plan.splitlines()
    for i in decl_idxs:
        window = "\n".join(lines[max(0, i - _C29_WINDOW_LINES) : i + _C29_WINDOW_LINES + 1])
        ev = _C29_EVIDENCE_RE.search(window)
        if ev:
            return _pass(
                cid,
                name,
                "fence-reconcile window references the §7 conditional phase "
                f"(evidence {ev.group(0)!r})",
            )
        for lb in labels:
            if re.search(rf"\b{re.escape(lb)}\b", window):
                return _pass(
                    cid,
                    name,
                    "fence-reconcile window references the §7 conditional phase "
                    f"(gate label {lb!r})",
                )
    return _warn(cid, name, _c29_offender_detail(lines[decl_idxs[0]], ext.group(0), labels))


# ─── Check 30 — reused-bundle realized keys (WARN-only, conditional) ───────

_C30_BUNDLE_RE = re.compile(
    # NO `.safetensors` token (v2, methodology-critic Must-Fix): adapter-reuse
    # plans routinely quote `adapter_model.safetensors` near reuse vocabulary —
    # a sweep of all historical plans showed 9 fire via `.safetensors`
    # alone, ALL adapter-class (#459 #523 #528 #562 #570 #595 #627 #632 #653).
    # The project's multi-field bundles are single `.pt` files; a safetensors
    # STORE still triggers via its prose tokens (tensor bundle /
    # analysis_tensors / activation store / multi-field bundle).
    r"(?i)(\.pt\b|\.pth\b|tensor bundle|multi-?field bundle|"
    r"save-dict|analysis_tensors|activation store)"
)
_C30_SATISFIER_RE = re.compile(
    r"(?i)(verify_reused_artifact_keys"  # the canonical helper
    r"|mmap\s*=\s*True[^\n]{0,120}\.keys\(\)"  # inline mmap key read
    r"|consumer(?:'s)?\s+own\s+loader)"  # consumer-loader-run form
)


def check_realized_keys(plan: str, kind: str) -> CheckResult:
    """Plans reusing a multi-field tensor bundle must name a realized-keys
    verification (artifact-reuse.md check (c), incident #1073). WARN not
    FAIL: the bundle-reuse trigger is heuristic (same class as c6), and the
    semantic question — was the probe actually RUN against the pinned
    revision — stays with the fact-checker. Trigger scans stripped prose;
    the satisfier ALSO scans raw text, because the runnable command
    legitimately lives in a fenced block."""
    cid, name = "c30_realized_keys", "reused-bundle realized-keys verification"
    if kind not in ("experiment", "analysis"):
        return _skip(cid, name, "kind-exempt")
    text = strip_fences(plan)
    bundle_hits = [m.start() for m in _C30_BUNDLE_RE.finditer(text)]
    reuse_near_bundle = any(
        re.search(r"(?i)\breus\w*", text[max(0, i - 300) : i + 300]) for i in bundle_hits
    )
    if not reuse_near_bundle:
        return _skip(cid, name, "no multi-field bundle reuse detected")
    if _standalone_na_declared(plan, r"no multi-?field bundle reuse"):
        return _pass(cid, name, "explicit no-bundle-reuse declaration")
    if _C30_SATISFIER_RE.search(plan):  # raw plan: fenced commands count
        return _pass(
            cid, name, "realized-keys verification named (helper / mmap read / consumer loader)"
        )
    return _warn(
        cid,
        name,
        "plan reuses a multi-field tensor bundle but names no realized-keys "
        "verification — artifact-reuse.md check (c): run `uv run python "
        "scripts/verify_reused_artifact_keys.py --artifact <path> --keys "
        "<consumer keys>` (or the consumer's own loader) against the pinned "
        "artifact and paste the PASS line into §10 (incident #1073)",
    )


# ─── Check 31 — SKILL.md prose edit backed by a durability pin (WARN-only) ─

# Trigger: a SKILL.md path token on a non-fenced, non-negated line, with an
# edit-commitment verb within +/-120 chars of the path match (long unwrapped
# plan lines make whole-line co-occurrence noisy — measured on the
# 2026-07-09 corpus scan, task #1179 plan §6). The path arm admits any
# slash-joined prefix (`.claude/skills/issue/SKILL.md`, relative
# `issue/SKILL.md`) or a bare `SKILL.md` not glued to a path/word char.
_C31_PATH_RE = re.compile(r"(?i)(?:[\w.-]+(?:/[\w.-]+)*/|(?<![\w./-]))SKILL\.md")
_C31_EDIT_RE = re.compile(
    r"(?i)\b(?:add(?:s|ed|ing)?|insert\w*|append\w*|amend\w*|edit\w*|splice\w*"
    r"|prepend\w*|reword\w*|rewrit\w*|revise[sd]?|patch\w*"
    r"|new (?:section|paragraph|bullet|sentence|step|clause|line))\b"
)
_C31_EDIT_PROX_CHARS = 120
# Negation / boilerplate guards — measured corpus noise classes (#1179 §6):
# "zero SKILL.md edits" (#700), "No SKILL.md change needed" (#875), "no
# companion edit to SKILL.md" (#797), scope-table "No change" rows (#792),
# must-ask / must-bounce deviation boilerplate (#890, #806, #869). The gap
# atom allows path-internal dots (`SKILL.md change`) but blocks a
# sentence-ending dot-space, so the guard cannot leak across sentences.
_C31_NEG_GUARD_RE = re.compile(
    r"(?i)\b(?:no|zero|not?|without|never)\b(?:[^|;:.]|\.(?!\s)){0,24}"
    r"\b(?:edit(?:s|ed|ing)?|chang(?:e|es|ed))\b"
    r"|\bunchanged\b|\bincidental\b|must-ask|must bounce"
    r"|park[^|]{0,24}plan_pending"
)
# Satisfier: an exact labeled line (c5/c20 machine-readable-line pattern) —
# a c15-style loose evidence scan false-satisfied all 9 incident plan
# versions (unrelated test_ identifiers + incidental vocabulary), so the
# label is load-bearing. RAW scan (c11/c15 evidence convention: the line may
# legitimately sit in a fenced §-block or table). The NA separator class
# mirrors NA_RE (em/en dash, colon, paren, hyphen) so `Durability pin: N/A
# (reason)` satisfies too.
_C31_PIN_LABEL_RE = re.compile(r"(?i)\bdurability pin:\s*")
_C31_PIN_NA_RE = re.compile(r"(?i)\bdurability pin:\s*N/?A\b\s*[—–:(-]\s*\S")  # noqa: RUF001
_C31_NA_ALIAS_RE = re.compile(NA_RE + r"no durability pin\s*[:—–-]\s*\S")  # noqa: RUF001


def _c31_trigger_lines(plan: str) -> list[str]:
    """Non-fenced, non-negated lines carrying a SKILL.md path with an
    edit-commitment verb within +/-``_C31_EDIT_PROX_CHARS`` of the path
    match."""
    lines = plan.splitlines()
    mask = _fence_mask(lines)
    out: list[str] = []
    for line, fenced in zip(lines, mask, strict=True):
        if fenced or _C31_NEG_GUARD_RE.search(line):
            continue
        for m in _C31_PATH_RE.finditer(line):
            lo = max(0, m.start() - _C31_EDIT_PROX_CHARS)
            hi = min(len(line), m.end() + _C31_EDIT_PROX_CHARS)
            if _C31_EDIT_RE.search(line[lo:hi]):
                out.append(line.strip())
                break
    return out


def _c31_satisfier(plan: str) -> str | None:
    """First satisfier line: ``Durability pin: <...test_...>`` (a standing OR
    planned pin test — the verifier cannot and need not distinguish), or a
    reason-bearing NA escape. Bare ``Durability pin: N/A`` (no reason) does
    NOT satisfy."""
    for line in plan.splitlines():
        m = _C31_PIN_LABEL_RE.search(line)
        if m and _TEST_IDENT_RE.search(line[m.end() :]):
            return f"pin named ({line.strip()[:80]!r})"
        if _C31_PIN_NA_RE.search(line) or _C31_NA_ALIAS_RE.search(line):
            return f"no-pin justification declared ({line.strip()[:80]!r})"
    return None


def check_skillmd_prose_pin(plan: str, kind: str) -> CheckResult:
    """``kind: infra|batch`` plans that commit to editing
    ``.claude/skills/**/SKILL.md`` prose must carry ONE labeled line naming
    a durability pin test (a pytest asserting the prose's presence/shape)
    or a one-line no-pin justification. SKILL.md protection prose with no
    pin is silently droppable by any later edit — lineage: #1134 (no pin),
    #1045 (pin optional), #884 (pin present but unlabeled). WARN not FAIL:
    the trigger is a line heuristic; the Phase 2 critics adjudicate. v1
    scope is SKILL.md paths only — extending to agents/rules/CLAUDE.md
    prose is a future calibration decision (the 2026-07-09 corpus scan
    measured that superset would-WARN at 174+ tasks, dominated by
    ledger-entry classes with no pin-test practice). Known residual FP
    class (disclosed): scope-table rows whose negation token sits >24
    chars from the edit verb (#1102 shape) still trigger — the 1-line NA
    escape is the remedy. Out of mechanical scope: whether a named pin
    test actually exists / ships (the code-reviewer checks the diff, same
    bound as c11/c15)."""
    cid, name = "c31_skillmd_prose_pin", "SKILL.md prose edit backed by a durability pin"
    if kind not in ("infra", "batch"):
        return _skip(
            cid, name, "kind-exempt: SKILL.md prose edits are an infra|batch (workflow-fix) shape"
        )
    trig = _c31_trigger_lines(plan)
    if not trig:
        return _skip(cid, name, "no SKILL.md edit-commitment line detected")
    sat = _c31_satisfier(plan)
    if sat:
        return _pass(cid, name, sat)
    return _warn(
        cid,
        name,
        f"plan commits to editing SKILL.md prose ({trig[0][:70]!r}) but names no durability "
        "pin — protection prose with no pytest asserting its presence/shape is silently "
        "droppable by any later SKILL.md edit (lineage: #884/#1045/#1134). Add one line "
        "`Durability pin: tests/test_<file>.py::test_<name>` (a standing pin test, or a NEW "
        "pin test this plan adds), or declare `Durability pin: N/A` followed on the same "
        "line by an em dash and a one-line reason (a bare `Durability pin: N/A` still WARNs)",
    )


# ─── Check 32 — fit-family §9 basis grounding ──────────────────────────────
# Mechanizes .claude/rules/plan-compute-sizing.md § "Per-cell fit phases"
# (MUST-level): a §9 row looping a fit/solve/factorization over cells x
# folds x layers x ... must ground its per-call basis on a MEASURED 1-cell
# pilot, a cited prior-issue measured figure, or a pre-registered
# `pilot-gated` flag — an ASSERTED per-call cost is never a basis, and a
# FLOP floor is the cross-check, never the basis (#823: asserted ~2 s/fit
# vs ~125 s real, 12-20 h realized; #811: one inner kernel timed, dominant
# frame asserted, unit 3/108 at 19h21m; #931: wrong-device measurement,
# ~2.2-2.5x mid-run; #722: "sub-minute per cell" asserted, 19.5 CPU-h).
# Anti-boilerplate, BOTH polarities (the #1060 round-1 critic concern):
# the satisfier requires provenance vocabulary CO-LOCATED with a numeric
# timing token in the basis/wall cells — "basis: measured pilot" with no
# number WARNs (#552 v3's literal "measured: minutes"), and "~2 s/fit"
# with no provenance word WARNs (#823's literal shape).
# Calibration (DEVELOPMENT-SET numbers: the regexes were tuned on the same
# persisted-plan corpus they were measured on — read the rates as
# in-sample, not held-out; ANY future c32-regex change re-runs the corpus
# scan and records the realized numbers here, the c27 gate precedent).
# Re-scan 2026-07-09 (implementation-time, shipped regexes) over 1,731
# persisted plan-versions (tasks/*/*/plans/v*.md): full corpus 149
# plan-versions triggered (32 distinct plans) / 85 would-WARN (23 distinct
# plans; pre-#1060-rule era dominated); recent era (issue >= 1000): 419
# plan-versions -> 13 triggered, 3 would-WARN, all ONE distinct plan
# (#1112 v1-v3, whose own v4 basis added "(parent-measured kernel)" and
# PASSes). Incident recall 100%: #823 v1-v5 / #811 v1 / #722 v1-v3 / #931
# v1-v4 all WARN (#931 v8-v10 post-incident replans also WARN — asserted
# bases, defensible under the rule; #811 v2-v3 and #722 v4-v5 SKIP: those
# restructured versions carry no fit-family basis-table row); every
# post-fix version PASSes on a substantive span (#811 v4 "REALIZED
# ~2.6 h"; #722 v7 "ran ~9 min"; #810 v4+ "measured 0.385 s/cell" /
# "parent 10 min"; #928 v5+ "prior-issue 1.0 h"; #1112 v4+ "parent
# 3 min"). #778 (draw-battery incident, c12's domain) never triggers on
# any of its 8 versions — clean division of labor. Disclosed residual
# gaming: a FABRICATED
# "measured 2 s/fit" passes — a mechanical check cannot verify
# measurement provenance (module scope discipline: a PASS here is never
# "grounding verified"); adequacy stays with the Methodology critic
# (critic-lens-reference.md item (iii)), fed by the WARN forwarding.

_C32_KERNEL_RE = re.compile(
    r"(?i)\bridge\b|\bsvd\b|\beigh\b|\beigvalsh\b|\blstsq\b|\bgcv\b"
    r"|\bloco\b|\bloocv\b|\blofo\b|\bmlp\b|\badamw\b|\bsgd\b|\bkrr\b"
    r"|\bglm\b|\birls\b|gradient[- ]descent|\bprobe[- ](?:train|fit)\w*"
    r"|\bfactoriz\w+"
    r"|\b(?:point|probe|many[- ]cell|per[- ]cell|per[- ]fold|closed[- ]form|serial)[ -]fits?\b"
    r"|\bfit loops?\b"
)
# NOTE: bare \bfits?\b deliberately EXCLUDED — it false-fires on "fits in
# HBM" / generation rows ("engine load ... 250 gens", #558) and cost 11
# extra full-corpus triggers in calibration for zero incident-recall gain.

_C32_LOOP_RE = re.compile(
    r"(?i)per[- ](?:cell|fold|layer|arm|trait|seed|unit|probe|context|pair|source|behavior)"
    r"|[×x]\s*\d|\d+\s*[×x]\b"  # noqa: RUF001 — the multiplication sign is real plan text
    r"|\d[\d,]*\s*(?:fits|solves|calls|folds|cells|refits|units)\b"
    r"|\bn_calls\b|\bfor each\b|\bacross (?:all )?\d+"
)

# Provenance vocabulary: measurement verbs + prior-figure citation forms.
# "parent"/"ran"/"#<M>" are load-bearing widenings — without them the
# corpus's legitimate prior-figure bases ("parent full grid ~10 min =>
# 0.58 s/cell", #810 v10; "v5 ran 28 layers in ~9 min", #722 v7) false-WARN.
_C32_PROVENANCE_RE = re.compile(
    r"(?i)\bmeasur\w+|\btimed\b|\bclocked\b|\bprofil\w+|\bbenchmark\w*"
    r"|\brealized\b|\bpilot\w*\b|\bran\b|\btook\b|\bparent\b"
    r"|\bprior[- ]issue\b|#\d{2,}|\bcommitted\b"
)

# Numeric timing token: a digit-bearing quantity with a time unit,
# optionally per-call ("125 s/fit", "~0.58 s/cell", "9 min").
# NOTE: an ASCII-hyphen range ("2-3 min") does NOT match (the lookbehind
# blocks it); the corpus's en-dash (U+2013) ranges do match.
# Lookbehind blocks "A100s"/"H100" ("100 s" inside an alnum run).
_C32_TIMING_RE = re.compile(
    r"(?i)(?<![A-Za-z0-9.\-])[~≈]?\d[\d,]*(?:\.\d+)?\s*"
    r"(?:ms|s|sec|seconds?|min|minutes?|hr?|hours?)\b"
    r"(?:\s*/\s*(?:it|call|fit|cell|unit|fold|row|draw|solve))?"
)

_C32_PILOT_GATED_RE = re.compile(r"(?i)\bpilot[- ]gated\b")


def _c32_offender_detail(offenders: list[tuple[str, str]]) -> str:
    """Bounded WARN detail (the c26 ``_c26_offender_detail`` shape): at most
    3 (component, basis) pairs, the rule anchor, the incident anchors, and
    every remedy (measured figure / prior-issue citation / pilot-gated /
    the standalone N/A escape)."""
    shown = "; ".join(f"row {comp[:60]!r} basis {basis[:40]!r}" for comp, basis in offenders[:3])
    if len(offenders) > 3:
        shown += "; ..."
    return (
        f"{shown} — fit-family row(s) whose basis carries neither (provenance vocabulary — "
        "measured/timed/pilot/#<M>/parent — co-located with a numeric per-call timing) nor a "
        "`pilot-gated` flag — an ASSERTED per-call cost is never a sizing basis and a FLOP "
        "floor is the cross-check, never the basis (plan-compute-sizing.md § Per-cell fit "
        "phases; #823: asserted ~2 s/fit, ~125 s real, 12-20 h realized; #811: unit 3/108 at "
        "19h21m). Ground the row on a measured 1-cell pilot at production shape (state the "
        "figure, e.g. `measured 125 s/fit`), cite a prior-issue measured figure "
        "(`#811 r2: 313 s/unit`), mark the basis `pilot-gated`, or declare "
        "`N/A — no fit-family phases` on its own line, unwrapped (no backticks/quotes), "
        "if the row is not a fit loop"
    )


def check_fit_basis_grounding(plan: str, kind: str) -> CheckResult:
    """WARN-only, conditional: every basis-column compute-table row naming a
    fit-family kernel (ridge/SVD/eigh/lstsq/GCV/MLP/LOCO/...) AND a
    loop/multiplicity signal (per-cell/per-fold vocabulary, an NxM product,
    an "N fits" count) must ground its basis — provenance vocabulary
    (measured/timed/pilot/#<M>/parent/ran/...) CO-LOCATED with a numeric
    timing token in the conversion-bearing cells (basis + wall, the c26
    escape-(a) precedent — a component cell like "reuse of #811 adapters"
    must not satisfy the citation class spuriously), or a literal
    ``pilot-gated`` flag anywhere in the row. Mechanizes
    plan-compute-sizing.md § "Per-cell fit phases" (#823/#811/#722/#931).
    Anti-boilerplate BOTH polarities (the #1060 critic concern): "measured
    pilot" with no digit WARNs, and "~2 s/fit" with no provenance word
    WARNs. A FLOP-only basis WARNs by construction (no provenance token) —
    the rule: a FLOP floor is the cross-check, never the basis; there is
    deliberately NO ``FLOP-only`` escape. NEVER FAILs in v1 — both trigger
    and satisfier are text heuristics (the c26 precedent), and whether a
    stated figure is REAL / transfers stays critic-owned: a FABRICATED
    "measured 2 s/fit" passes (a mechanical check cannot verify
    measurement provenance — a PASS here is never "grounding verified").
    Disclosed under-triggers: fit sizing stated only in prose (no
    basis-column table) is invisible in v1 (c12 independently covers prose
    draw batteries); a basis table lacking a wall column is invisible
    (parser precondition, c26 parity). Escape: the standalone line
    ``N/A — no fit-family phases`` (anti-paste semantics via
    ``_standalone_na_declared``). Calibration numbers + the corpus re-scan
    gate on ANY future c32-regex change live in the comment block above
    ``_C32_KERNEL_RE``."""
    cid, name = "c32_fit_basis_grounding", "fit-family §9 basis grounding"
    if kind not in ("experiment", "analysis"):
        return _skip(
            cid, name, "kind-exempt: fit-family §9 rows are an experiment|analysis plan shape"
        )
    rows = _c26_compute_table_rows(plan)
    triggered = [
        (comp, basis, wall, row_text)
        for comp, basis, wall, row_text in rows
        if _C32_KERNEL_RE.search(row_text) and _C32_LOOP_RE.search(row_text)
    ]
    if not triggered:
        return _skip(cid, name, "no fit-family row in a basis-column compute table detected")
    if _standalone_na_declared(plan, r"no fit[- ]family (?:fit )?phases"):
        return _pass(cid, name, "explicit N/A declared (no fit-family phases)")
    offenders: list[tuple[str, str]] = []
    for comp, basis, wall, row_text in triggered:
        conv = f"{basis} {wall}"  # conversion-bearing cells, the c26 escape-(a) precedent
        grounded = (
            _C32_PROVENANCE_RE.search(conv) and _C32_TIMING_RE.search(conv)
        ) or _C32_PILOT_GATED_RE.search(row_text)
        if not grounded:
            offenders.append((comp, basis))
    if not offenders:
        return _pass(
            cid,
            name,
            f"{len(triggered)} fit-family row(s); every basis carries provenance + a timing "
            "figure or pilot-gated",
        )
    return _warn(cid, name, _c32_offender_detail(offenders))


# ─── Check 33 — checkpoint-ladder retention policy ─────────────────────────
# Mechanizes .claude/rules/plan-compute-sizing.md § "Dose-ladder /
# multi-rung checkpoint retention" (MUST-level, the #1133 rule): any
# training phase persisting per-rung checkpoints for later selection must
# state its checkpoint-retention policy in §9 — DEFAULT: retain the
# dose-selected + latest rungs only, delete ruled-out rungs BETWEEN rungs;
# keep-all is the justified exception (full-ladder sizing at realized
# per-rung GB + `--boot-disk-gb` declared). Incident #1112: 30 full-FT
# dose-ladder rungs kept (>=15 GB, up to ~28 GB each); a compliant 575 GB
# keep-all bound sat under the planned 750 GB GCP boot disk; the
# GCP-to-RunPod failover delivered the `ft-7b` default 200 GB volume ->
# ENOSPC (errno 28) at rung 24/30.
# Trigger anti-fragility: raw `ladder|rung` vocabulary is heavily polluted
# by GCP BACKEND-ladder rungs (spot/flex-start/on-demand fallback rungs,
# the #1029/#1116/#1121 vocabulary) — measured raw surface ~521 pv / 179
# issues un-gated (~320/89 kind-gated). The compound-token trigger + the
# backend-rung exclusion on the rung-AND-checkpoint co-location branch
# remove that class entirely.
# Calibration (DEVELOPMENT-SET numbers, fitted IN-SAMPLE — including the
# recent-era slice: the regexes were tuned on the same persisted-plan
# corpus they were measured on; ANY future c33-regex change re-runs the
# corpus scan and records the realized numbers here, the c27/c32 gate
# precedent). Re-scan 2026-07-10 (implementation-time, AS-SHIPPED
# regexes) over 1,760 persisted plan-versions (tasks/*/*/plans/v*.md):
# 69 plan-versions triggered (20 distinct issues — genuinely
# checkpoint-ladder-bearing: #480 band-stop ladder, #653 dense-step grid,
# #1090 every-25-steps ladder, #1112 dose ladder, #488 epoch ladder, ...);
# 42 would-WARN (16 issues, pre-#1133-rule era dominated). Recent era
# (issue >= 1000; the §7 kill-criterion DENOMINATOR is recent-era
# plan-versions, N=448): 18 triggered pv (#1090 v1-v5 / #1092 v1-v5 /
# #1112 v1-v8); would-WARN ONLY #1090 v1-v5 (planned pre-rule). #1092
# PASSes; #1112 v7/v8 PASS on their explicit `**Disk / checkpoint
# retention:**` line. Satisfier-span audit over the 27 triggered-but-PASS
# versions: 'retained' x8, delete-co-location spans x14 ("rungs deleted",
# "checkpoints ... deleted", "checkpoint ... then DELETE"), 'retention'
# x2, 'prune(d)' x2, 'MarkerBandStopCallback' x1. Over-broad-token watch
# (re-download / prune vs disk-hygiene boilerplate): re-download matched
# ZERO spans; prune matched 2 — #491 v3 genuine (per-shard
# train->read->prune checkpoint sequencing) and #715 v3 borderline
# (weight-pruning-arm vocabulary; its v1/v2 pass on a genuine delete
# span) — no nuisance CLASS, no regex change.
# Honest disclosed limitation: #1112 v1-v3 — the incident's own plans —
# PASS: their §9 stated merge-transient deletion + a keep-all disk bound,
# so the retention SURFACE existed; the defect was semantic (sized to the
# planned lane's disk, keep-all as default). A mechanical check cannot
# adjudicate adequacy — c33 protects the SILENT class (ladder plans whose
# sizing sections say nothing about retention/deletion); stated-but-
# inadequate stays with Methodology lens item 16.

_C33_LADDER_COMPOUND_RE = re.compile(
    r"(?i)dose[- ]ladder|checkpoint[- ]ladder|ladder of checkpoints|band[- ]stop grid"
    r"|dose[- ]matching checkpoint grid|checkpoint rungs?|rung checkpoints?"
    r"|per[- ]rung checkpoints?"
)

# Mechanizes the rule's "any long run saving every k steps for a later
# pick" clause ("saves a checkpoint every 25 steps", "saving every ~500
# optimizer steps").
_C33_SAVE_EVERY_RE = re.compile(
    r"(?i)(?:checkpoints?|sav\w+)\s+every\s+~?\d+\s*(?:optimizer[- ])?(?:steps?|epochs?)"
)

_C33_RUNG_RE = re.compile(r"(?i)\brungs?\b")
_C33_CKPT_RE = re.compile(r"(?i)\bcheckpoints?\b|\bckpts?\b")

# GCP fallback-ladder exclusion (co-location branch ONLY): a line whose
# rung vocabulary is the backend router's (spot/flex-start/on-demand
# rungs, terminal rung, lanes, capacity) is not a checkpoint ladder.
_C33_BACKEND_RUNG_RE = re.compile(
    r"(?i)spot|flex[- ]start|on[- ]demand|runpod|terminal rung|\blanes?\b|fallback"
    r"|a2-|a3-|gcp ladder|capacity"
)

# Retention/bounding vocabulary. The delete co-location windows stop at
# sentence/cell boundaries (the `|` exclusion keeps a table row's delete
# verb from satisfying via an adjacent cell). `keep-all` deliberately
# satisfies — a STATED keep-all is the rule's justified-exception surface,
# whose adequacy is critic-owned. Generic disk tokens (`--boot-disk-gb`,
# GB figures) are deliberately NOT satisfiers — #1112 v1-v3 declared
# `--boot-disk-gb 750` and still ENOSPC'd; a disk flag is not a retention
# policy.
_C33_RETENTION_RE = re.compile(
    r"(?i)\bretention\b|\bretain\w*\b|keep[- ]all|keep (?:all|every|only)"
    r"|delet\w+[^.\n|]{0,80}(?:rungs?|ruled[- ]out|non[- ]selected|checkpoints?|ckpts?)"
    r"|(?:rungs?|checkpoints?|ckpts?)[^.\n|]{0,80}delet\w+"
    r"|upload[- ]as[- ]you[- ]go|delete[sd]? locally|re[- ]download"
    r"|band[- ]stop callback|MarkerBandStopCallback"
    r"|coarse\+refine|two[- ]pass grid|\bprune[sd]?\b|retained (?:set|rungs?)"
)

_C33_SIZING_KEYWORDS = ("resources", "parallelism", "compute", "disk")


def _c33_trigger_line(plan: str) -> str | None:
    """First non-fenced line carrying checkpoint-ladder vocabulary (quoted
    in the WARN detail), or None. Three arms, first match wins: a compound
    ladder token; the save-every-k-steps cadence; rung AND checkpoint
    co-located on one line WITHOUT backend-rung vocabulary (the GCP
    fallback-ladder exclusion — the load-bearing anti-fragility widening)."""
    lines = plan.splitlines()
    mask = _fence_mask(lines)
    for line, fenced in zip(lines, mask, strict=True):
        if fenced:
            continue
        if _C33_LADDER_COMPOUND_RE.search(line) or _C33_SAVE_EVERY_RE.search(line):
            return line
        if (
            _C33_RUNG_RE.search(line)
            and _C33_CKPT_RE.search(line)
            and not _C33_BACKEND_RUNG_RE.search(line)
        ):
            return line
    return None


def _c33_sizing_scope(plan: str) -> str:
    """Union of the non-fenced text of every section whose heading carries a
    sizing keyword (resources/parallelism/compute/disk — the #1133 rule
    requires the policy in §9, but corpus headings drift: '## 9. Resources',
    '## 9. Resources & Parallelism', '### Compute projection'); the whole
    plan's non-fenced text when no such heading exists (structural absence
    must not manufacture WARNs — a WARN-only check fails toward silence)."""
    lines = plan.splitlines()
    mask = _fence_mask(lines)
    keep = [False] * len(lines)
    matched = False
    for h in _headings(plan):
        htext = h.text.casefold()
        if any(k in htext for k in _C33_SIZING_KEYWORDS):
            matched = True
            for i in range(h.line, h.end):
                keep[i] = True
    if not matched:
        return strip_fences(plan)
    return "\n".join(
        line
        for i, (line, fenced) in enumerate(zip(lines, mask, strict=True))
        if keep[i] and not fenced
    )


def check_ladder_retention(plan: str, kind: str) -> CheckResult:
    """WARN-only, conditional: a plan carrying checkpoint-ladder vocabulary
    on a non-fenced line (a dose/checkpoint-ladder compound token, a
    "saves a checkpoint every k steps" cadence, or rung + checkpoint
    co-located on one line without GCP backend-rung vocabulary) must carry
    retention vocabulary (retain / keep-all / delete-between-rungs /
    upload-as-you-go / band-stop / coarse+refine / prune / ...) within its
    compute-sizing section(s) — the union of every section whose heading
    names resources/parallelism/compute/disk, doc-wide fallback when no
    such heading exists. Mechanizes plan-compute-sizing.md § "Dose-ladder /
    multi-rung checkpoint retention" (the #1133 rule; incident #1112).
    NEVER FAILs — both trigger and satisfier are text heuristics (the
    c26/c32 precedent); adequacy of a STATED policy stays with the
    Methodology critic (lens item 16), fed by the WARN forwarding into the
    fact-checker + critic briefs. A PASS here is never "retention
    verified": a stated keep-all deliberately satisfies (the rule's
    justified-exception surface, critic-owned). Disclosed misses:
    (a) #1112 v1-v3 — the incident's own plans — PASS (their §9 stated
    merge-transient deletion + a keep-all bound, so the retention SURFACE
    existed; the defect was semantic and is Methodology lens item 16's);
    (b) a ladder phrased with zero token-set overlap under-triggers
    (FN = the status quo, reviewer-enforced only); (c) a crash-resume-only
    save cadence (no later selection) triggers via the save-every arm —
    the remedy is to state a retention policy anyway (e.g. keep-last-k,
    which a crash-resume cadence should state regardless), or, second, to
    declare the N/A escape ONLY when no phase persists per-rung
    checkpoints (the escape phrase would be semantically false for a plan
    that does persist them). Escape: the standalone line
    ``N/A — no per-rung checkpoint persistence`` (alias
    ``N/A — no checkpoint ladder``), anti-paste semantics via
    ``_standalone_na_declared``. Calibration numbers + the corpus re-scan
    gate on ANY future c33-regex change live in the comment block above
    ``_C33_LADDER_COMPOUND_RE``."""
    cid, name = "c33_ladder_retention", "checkpoint-ladder retention policy"
    if kind not in ("experiment", "analysis"):
        return _skip(
            cid, name, "kind-exempt: checkpoint ladders are an experiment|analysis plan shape"
        )
    trig = _c33_trigger_line(plan)
    if trig is None:
        return _skip(cid, name, "no checkpoint-ladder vocabulary detected")
    if _standalone_na_declared(
        plan, r"(?:no per[- ]rung checkpoint persistence|no checkpoint ladder)"
    ):
        return _pass(cid, name, "explicit N/A declared (no per-rung checkpoint persistence)")
    if _C33_RETENTION_RE.search(_c33_sizing_scope(plan)):
        return _pass(cid, name, "retention vocabulary present in the compute-sizing scope")
    return _warn(
        cid,
        name,
        f"plan carries checkpoint-ladder vocabulary ({trig.strip()[:70]!r}) but its "
        "compute-sizing section(s) state no checkpoint-retention policy — a per-rung ladder "
        "sized without a retention default keeps every rung and ENOSPCs mid-run on a lane "
        "failover (plan-compute-sizing.md § Dose-ladder / multi-rung checkpoint retention, "
        "the #1133 rule; incident #1112: 30 full-FT rungs kept, errno 28 at rung 24/30 after "
        "a GCP-to-RunPod failover delivered a 200 GB volume). State the retention policy in "
        "the sizing section (DEFAULT: retain the dose-selected + latest rungs only, delete "
        "ruled-out rungs BETWEEN rungs; or the justified keep-all exception — full-ladder "
        "sizing at realized per-rung GB + `--boot-disk-gb` declared), or declare "
        "`N/A — no per-rung checkpoint persistence` on its own line, unwrapped "
        "(no backticks/quotes), if no phase persists per-rung checkpoints",
    )


# ─── Check 34 — verbatim insert vs size-ratchet headroom ──────────────────
# A plan mandating VERBATIM prose into a workflow_lint size-ratcheted file
# (.claude/agents/*.md — check_agent_spec_size; .claude/rules/LESSONS.md —
# check_lessons_index) whose remaining headroom is smaller than the quoted
# block makes lint-passes + the plan's own file-count constraint jointly
# unsatisfiable at implement time (#1230: 422 B headroom vs a 1,546 B
# verbatim paragraph for code-reviewer.md forced a documented 3rd-file
# cap-raise deviation). Grandfather caps deliberately hug live size
# (cap = measured + <=3 KB, typically ~1 KB), so ANY >~1 KB insert into a
# grandfathered spec exceeds headroom BY DESIGN — the remedy is therefore
# never "don't grow" but "budget the visible cap-raise IN THE PLAN"
# (workflow_lint.py: "a reviewed growth+cap-raise in one commit still
# passes"). WARN not FAIL: the trigger is a proximity heuristic; the
# Phase 2 critics adjudicate. infra|batch only: editing agent specs /
# LESSONS.md is workflow-fix work (calibration: all 8 recent-era corpus
# hits are kind: infra).
# Calibration (DEVELOPMENT-SET, measured against TODAY'S live sizes —
# historical headroom drifts; the c32 precedent): 1,837 persisted
# plan-versions scanned 2026-07-11; trigger fired on 166 versions;
# would-WARN 76 versions / 26 distinct plans (8 distinct at issue >= 1000:
# #1007 #1017 #1022 #1142 #1224 #1230 #1239 #1254 — every one a real
# plan-mandated over-headroom insert; the #1119/#1138/#1254/#1230
# cap-raises are recorded in AGENT_SPEC_SIZE_GRANDFATHER's own comments).
# Incident recall: #1230 v1 WARNs. Reproducible scan recipe (the
# kill-criterion (a) re-audit; re-run + re-record on ANY c34-regex
# change): from the repo root, for each tasks/*/*/plans/v*.md compute
# trigger := bool(_c34_targets(text)) and would-WARN := any (rel, nbytes)
# in _c34_targets(text).items() with _c34_headroom(rel, wl) is not None
# and nbytes > headroom, where wl := _c34_lint_constants(); count
# plan-versions and distinct plan dirs for both.
# Scope notes (disclosed, accepted residuals for a WARN-class v1):
# (a) the `Ratchet budget:` satisfier is DOCUMENT-GLOBAL, not per-target —
#     a two-file plan budgeting only one raise passes for both (pinned by
#     test_c34_budget_line_is_document_global);
# (b) nested-fence inserts UNDERCOUNT: _fence_mask's toggle reads an inner
#     ``` as a closer, so a verbatim block that itself contains a code
#     fence contributes only its pre-fence lines — a disclosed FN class
#     distinct from the non-fenced-insert FN (an insert described with no
#     fenced block at all, or with the path >_C34_WINDOW_LINES non-fenced
#     lines above the fence, never triggers — those stay with the human
#     critics);
# (c) TOCTOU: headroom is a PLAN-TIME snapshot of the live file sizes —
#     the target can grow between plan verification and implementation;
#     workflow_lint's commit-time FAIL is the hard backstop.
# Scope discipline: whether the budgeted cap-raise actually SHIPS is the
# code-reviewer's bound, not this check's (the c31/c11/c15 bound).

_C34_PATH_RE = re.compile(r"(?i)(?:\.claude/)?(?:agents/[\w.-]+\.md|rules/LESSONS\.md)")
_C34_VERB_RE = re.compile(
    r"(?i)\b(?:insert\w*|append\w*|add(?:s|ed|ing)?|splice\w*|verbatim|paste\w*)\b"
)
_C34_WINDOW_LINES = 10  # preceding non-fenced prose lines scanned per fence
# Digit-lookahead after the label on the SAME line = anti-paste armor: the
# WARN detail's remedy writes the label followed only by angle-bracket
# placeholders, so a verbatim paste of the detail can never self-satisfy.
_C34_BUDGET_RE = re.compile(r"(?i)\bratchet budget:(?=[^\n]*\d)")
_C34_REPO_ROOT = Path(__file__).resolve().parent.parent  # tests monkeypatch


def _c34_lint_constants():
    """Lazy import of ``scripts/workflow_lint.py`` (540 KB module, ~345 ms
    measured) — paid ONLY when the c34 trigger fires, so typical plans keep
    the verifier sub-second. Single source of truth for the ratchet caps
    (the grandfather dict churns ~weekly; a copy WOULD drift). An
    ImportError is a real defect (both files live in scripts/) and
    propagates loud."""
    scripts_dir = str(Path(__file__).resolve().parent)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    import workflow_lint

    return workflow_lint


def _c34_normalize_rel(match_text: str) -> str:
    """Normalize a ``_C34_PATH_RE`` match to the repo-root-relative ratcheted
    path (``.claude/agents/<name>.md`` / ``.claude/rules/LESSONS.md``)."""
    tail = match_text
    if tail.lower().startswith(".claude/"):
        tail = tail[len(".claude/") :]
    if tail.lower().startswith("agents/"):
        return ".claude/agents/" + tail[len("agents/") :]
    return ".claude/rules/LESSONS.md"


def _c34_targets(plan: str) -> dict[str, int]:
    """``{normalized rel path -> summed fenced-block UTF-8 bytes}`` for every
    fenced block whose preceding ``<=_C34_WINDOW_LINES`` NON-fenced lines
    carry a ratcheted path AND an insertion verb. Block bytes = joined
    content lines + one trailing newline (fence delimiters excluded); the
    realized insert may differ by a separator newline or two — immaterial
    at the hundreds-of-bytes scale the check discriminates."""
    lines = plan.splitlines()
    mask = _fence_mask(lines)
    targets: dict[str, int] = {}
    in_fence = False
    open_idx = -1
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not (stripped.startswith("```") or stripped.startswith("~~~")):
            continue
        if not in_fence:
            in_fence = True
            open_idx = i
            continue
        in_fence = False
        window: list[str] = []
        j = open_idx - 1
        while j >= 0 and len(window) < _C34_WINDOW_LINES:
            if not mask[j]:
                window.append(lines[j])
            j -= 1
        wtext = "\n".join(reversed(window))
        m = _C34_PATH_RE.search(wtext)
        if m is None or not _C34_VERB_RE.search(wtext):
            continue
        rel = _c34_normalize_rel(m.group(0))
        content = "\n".join(lines[open_idx + 1 : i]) + "\n"
        targets[rel] = targets.get(rel, 0) + len(content.encode("utf-8"))
    return targets


def _c34_headroom(rel: str, wl) -> tuple[int, int, str] | None:
    """``(headroom, cap, cap_source)`` for a ratcheted rel path under
    ``_C34_REPO_ROOT``; ``None`` when the live file is absent (headroom
    uncomputable — a plan may be CREATING the file, which starts with the
    full cap of headroom). ``cap_source`` names the binding workflow_lint
    constant for the WARN detail. Sizes in BYTES (``stat().st_size``) —
    parity with ``check_agent_spec_size`` / ``read_bytes()``."""
    p = _C34_REPO_ROOT / rel
    if not p.is_file():
        return None
    size = p.stat().st_size
    if p.name == "LESSONS.md":
        # #1269: the binding runtime constraint is min(cap, growth ratchet) —
        # a plan passing the 8000-byte cap headroom could still FAIL
        # workflow_lint's ratchet at implement time (the plan-time miss c34
        # exists to close).
        ratchet = wl._LESSONS_RATCHET_BYTES
        cap = min(wl._LESSONS_MAX_BYTES, ratchet)
        src = "_LESSONS_RATCHET_BYTES" if ratchet < wl._LESSONS_MAX_BYTES else "_LESSONS_MAX_BYTES"
        return cap - size, cap, src
    cap = wl.AGENT_SPEC_SIZE_GRANDFATHER.get(p.name)
    if cap is not None:
        return cap - size, cap, "AGENT_SPEC_SIZE_GRANDFATHER"
    return wl.AGENT_SPEC_FAIL_BYTES - size, wl.AGENT_SPEC_FAIL_BYTES, "AGENT_SPEC_FAIL_BYTES"


def _c34_offender_detail(offenders: list[tuple[str, int, int, int, str]]) -> str:
    """Bounded WARN detail: at most 3 offender tuples, the #1230 incident
    anchor, then the three remedies. Anti-paste armored: after the
    ``Ratchet budget:`` label the text carries ONLY angle-bracket
    placeholders (no digit on the line — ``_C34_BUDGET_RE``'s lookahead
    cannot match a pasted copy; the incident numbers all sit BEFORE the
    label), and the N/A phrase is backtick-wrapped (unrecognized by
    ``_standalone_na_declared``, #1238)."""
    shown = "; ".join(
        f"{rel}: insert ~{nbytes} B > headroom {headroom} B (cap {cap} [{src}] - live size)"
        for rel, nbytes, headroom, cap, src in offenders[:3]
    )
    more = f" (+{len(offenders) - 3} more)" if len(offenders) > 3 else ""
    return (
        f"plan mandates verbatim fenced insert(s) exceeding the named ratcheted file(s)' "
        f"remaining size-ratchet headroom: {shown}{more} — workflow_lint-passes and the "
        "plan's own file-count constraint become jointly unsatisfiable at implement time "
        "(#1230: a paragraph larger than code-reviewer.md's headroom forced an un-planned "
        "third-file cap-raise deviation). Remedies: budget the cap-raise IN THE PLAN with "
        "one line `Ratchet budget: raise <constant>['<file>.md'] to <new cap>` (new cap = "
        "post-insert measured size plus at most the grandfather headroom bound), or trim "
        "the insert to fit, or declare `N/A — no verbatim ratcheted-file insertion` on its "
        "own line (write the declaration unwrapped — the backticks here are anti-paste "
        "armor)"
    )


def check_ratchet_headroom(plan: str, kind: str) -> CheckResult:
    """WARN-only, conditional, ``kind: infra|batch``: a fenced block whose
    preceding ``<=_C34_WINDOW_LINES`` non-fenced lines name a ratcheted path
    (``.claude/agents/*.md`` / ``.claude/rules/LESSONS.md``) plus an
    insertion verb is treated as a verbatim insert into that file; when the
    per-target summed block bytes exceed the file's live headroom
    (cap - ``stat().st_size``, caps lazy-imported from workflow_lint) the
    check WARNs with the arithmetic + the three remedies. Satisfiers: a
    non-fenced ``Ratchet budget:`` line carrying a post-label digit (the
    plan budgets the cap-raise — the legitimate path, since grandfather
    caps hug live size by design), or the standalone escape
    ``N/A — no verbatim ratcheted-file insertion``. Named-file-absent →
    SKIP (a plan may be CREATING the file; ``--plan-file`` mode must never
    crash off-repo). NEVER FAILs — trigger + satisfier are text heuristics
    (the c31 template); calibration, scope notes (document-global
    satisfier, nested-fence undercount, plan-time TOCTOU snapshot) and the
    scan recipe live in the section comment above ``_C34_PATH_RE``."""
    cid, name = "c34_ratchet_headroom", "verbatim insert fits size-ratchet headroom"
    if kind not in ("infra", "batch"):
        return _skip(
            cid,
            name,
            "kind-exempt: ratcheted-file verbatim inserts are an infra|batch (workflow-fix) shape",
        )
    if _standalone_na_declared(plan, r"no verbatim ratcheted[- ]file insertion"):
        return _pass(cid, name, "escape declared: no verbatim ratcheted-file insertion")
    targets = _c34_targets(plan)
    if not targets:
        return _skip(cid, name, "no fenced block associated with a ratcheted-file insertion")
    lines = plan.splitlines()
    for line, fenced in zip(lines, _fence_mask(lines), strict=True):
        if not fenced and _C34_BUDGET_RE.search(line):
            return _pass(cid, name, f"cap-raise budgeted ({line.strip()[:80]!r})")
    wl = _c34_lint_constants()  # lazy: only reached on trigger
    offenders: list[tuple[str, int, int, int, str]] = []
    checked = 0
    for rel, nbytes in sorted(targets.items()):
        hr = _c34_headroom(rel, wl)
        if hr is None:
            continue  # absent on disk: headroom uncomputable
        checked += 1
        headroom, cap, cap_source = hr
        if nbytes > headroom:
            offenders.append((rel, nbytes, headroom, cap, cap_source))
    if not checked:
        return _skip(
            cid, name, "named ratcheted file(s) not present on disk — headroom uncomputable"
        )
    if offenders:
        return _warn(cid, name, _c34_offender_detail(offenders))
    return _pass(cid, name, f"{checked} ratcheted-file insert(s) fit remaining headroom")


# ─── Check 35 — revision-pinned reuse verified at the pin (WARN-only) ──────

# Trigger: a 40-hex token with revision/pin vocabulary within +/-120 chars,
# HF-context AND reuse vocabulary within +/-300 chars (the c6/c30 proximity
# convention), scanning STRIPPED prose. The revision-vocab window is what
# keeps git code SHAs (Repro-card `commit=<sha>` rows) out; the HF-context
# window keeps "pinned to commit <sha>" git rows out (#1345 shape only).
_C35_HEX40_RE = re.compile(r"\b[0-9a-f]{40}\b")
_C35_REV_VOCAB_RE = re.compile(r"(?i)\brevision\b|\bpin(?:ned|s)?\b")
_C35_HF_CTX_RE = re.compile(
    r"(?i)superkaiba1/|hf_hub_download|huggingface|hf (?:model|data) repo"
    r"|list_repo_(?:files|tree)|snapshot_download|repo_id|repo_type"
)
_C35_REUSE_RE = re.compile(r"(?i)\breus\w*|\binherit\w*")
_C35_REV_WIN, _C35_CTX_WIN = 120, 300
# Satisfiers scan RAW text (c30 convention: runnable probe commands
# legitimately live in fenced blocks): a Hub-probe callable with a
# `revision=` kwarg on the same line, or a prose verified-at-pin statement.
# `get_paths_info` is deliberately EXCLUDED: the artifact-reuse item-(j)
# pairwise-provenance boilerplate (`get_paths_info(expand=True,
# revision=...)`) verifies commit-DATE coherence, not existence-at-pin, and
# it sits in standard §10 rows — including it blinded the check to its own
# motivating incident (#1345 plan v3 line 446).
_C35_PROBE_SATISFIER_RE = re.compile(
    r"(?i)(?:list_repo_(?:tree|files)|file_exists|hf_hub_download)"
    r"[^\n]{0,200}\brevision\s*[=:]"
)
_C35_PROSE_SATISFIER_RE = re.compile(
    r"(?i)verif\w+[^\n]{0,120}\bat\s+(?:the\s+)?(?:pinned\s+)?revision\b"
)


def check_pinned_revision_reuse(plan: str, kind: str) -> CheckResult:
    """Plans reusing an HF artifact at a pinned 40-hex revision must name a
    revision-scoped existence verification (incident #1345: a default-branch
    probe read CONFIRMED while 2/4 stems returned 0 files at the pin). WARN
    not FAIL: 'reuse row' detection is heuristic (same class as c6/c30), and
    the semantic question — was the probe actually RUN, per stem, at the pin
    — stays with the fact-checker (SKILL.md Phase 1.5).

    Disclosed FALSE-NEGATIVE residuals — a SKIP is never read as coverage:
    short-hex pins (7-12 hex, e.g. #1345 v4's 10-hex pin), branch/tag pins
    (non-hex revisions), and pins held only in a code constant (zero hex in
    the plan prose) do not trigger — the fact-checker instruction ("read the
    actual code/config") is the coverage for the constant case; a
    revision-threaded `hf_hub_download` CONSUME recipe satisfies without a
    stated probe (the disclosed consume residual). The WARN detail below is
    deliberately satisfier-inert (no Hub-callable + `revision=` on one line,
    no 'verif...at...revision' shape): bounced plans paste verifier details
    verbatim, and a self-matching detail would false-PASS exactly the
    flagged-then-revised plans (the #810 spurious-satisfaction shape) —
    pinned by test_c35_warn_detail_matches_no_satisfier."""
    cid, name = "c35_pinned_revision_reuse", "revision-pinned reuse verified at pin"
    if kind not in ("experiment", "analysis"):
        return _skip(cid, name, "kind-exempt")
    text = strip_fences(plan)
    pinned_reuse = False
    for m in _C35_HEX40_RE.finditer(text):
        w_rev = text[max(0, m.start() - _C35_REV_WIN) : m.end() + _C35_REV_WIN]
        w_ctx = text[max(0, m.start() - _C35_CTX_WIN) : m.end() + _C35_CTX_WIN]
        if (
            _C35_REV_VOCAB_RE.search(w_rev)
            and _C35_HF_CTX_RE.search(w_ctx)
            and _C35_REUSE_RE.search(w_ctx)
        ):
            pinned_reuse = True
            break
    if not pinned_reuse:
        return _skip(cid, name, "no revision-pinned HF reuse detected")
    if _standalone_na_declared(plan, r"no revision[- ]pinned reuse"):
        return _pass(cid, name, "explicit no-pinned-reuse declaration")
    if _C35_PROBE_SATISFIER_RE.search(plan) or _C35_PROSE_SATISFIER_RE.search(plan):
        return _pass(cid, name, "revision-scoped existence verification named")
    return _warn(
        cid,
        name,
        "plan reuses an HF artifact at a pinned 40-hex revision but names no "
        "revision-scoped existence check - probe each named stem with the revision "
        "kwarg set to the pin (list_repo_tree scoped to the stem prefix on the "
        "~1M-file data repo, or list_repo_files on small repos; >=1 file per stem); "
        "a default-branch probe does NOT satisfy (incident #1345: 2/4 stems returned "
        "0 files at the pin); or declare `N/A - no revision-pinned reuse` on its own "
        "line, unwrapped",
    )


# ─── Driver ────────────────────────────────────────────────────────────────

CHECKS = [
    check_source_grounding,
    check_measurement_validity,
    check_data_tier,
    check_contrastive_negatives,
    check_gpu_hours,
    check_reuse_fitness,
    check_replication_fidelity,
    check_success_kill,
    check_conditions_seeds,
    check_marker_recipe,
    check_dryrun_test_coverage,
    check_battery_multiplier,
    check_empirical_gate_attainability,
    check_hypothesis_branch_coherence,
    check_failloud_test_coverage,
    check_reference_headline_distinction,
    check_causal_claim_scope,
    check_paired_contrast_source_coverage,
    check_ood_folds,
    check_verdict_lattice_coherence,
    check_grep_arity_gate,
    check_cross_section_param_consistency,
    check_resume_provenance,
    check_html_entities_in_commands,
    check_gpu_basis_routed_machine,
    check_capture_intent_hbm,
    check_precedent_band_coherence,
    check_fence_conditional_phase,
    check_realized_keys,
    check_skillmd_prose_pin,
    check_fit_basis_grounding,
    check_ladder_retention,
    check_ratchet_headroom,
    check_pinned_revision_reuse,
]


def verify_plan_text(raw: str, *, kind: str, source: str = "") -> tuple[bool, list[CheckResult]]:
    """Run every plan check on ``raw`` plan text under ``kind``.

    Check 0 (plan-nonstub) short-circuits the chain on FAIL — a stub plan
    would otherwise cascade into a dozen "<block> missing" errors that bury
    the actual root cause (a broken handoff). Returns
    ``(overall, results)``; WARN and SKIP both leave ``passed=True``.
    """
    del source  # reserved for symmetry with verify_task_body.verify_text
    stub = check_plan_nonstub(raw)
    if not stub.passed:
        return False, [stub]
    results = [stub] + [chk(raw, kind) for chk in CHECKS]
    overall = all(r.passed for r in results)
    return overall, results


def _newest_plan_version(folder: Path) -> Path:
    """Newest ``plans/v{K}.md`` by NUMERIC sort (``v10`` > ``v9``) — never
    the ``plan.md`` symlink (follow-up rounds re-point it; incident #597)."""
    versions: list[tuple[int, Path]] = []
    for p in folder.glob("plans/v*.md"):
        m = re.fullmatch(r"v(\d+)\.md", p.name)
        if m:
            versions.append((int(m.group(1)), p))
    if not versions:
        raise FileNotFoundError(f"no plans/v*.md under {folder}")
    versions.sort()
    return versions[-1][1]


def _kind_from_body(folder: Path) -> str:
    """``kind`` from ``body.md`` frontmatter; missing → ``experiment``
    (the strictest — the /issue Step 0b gate guarantees presence anyway)."""
    body_path = folder / "body.md"
    if not body_path.exists():
        return "experiment"
    fm, _ = split_frontmatter(body_path.read_text())
    return str(fm.get("kind") or "experiment")


def _load_plan_for_issue(number: int) -> tuple[str, Path, str]:
    """Resolve (plan_text, plan_path, kind) for a task number via the
    canonical resolver — never hand-built ``tasks/`` paths."""
    from explore_persona_space.task_workflow import find_task_path  # local import

    folder = find_task_path(number)
    plan_path = _newest_plan_version(folder)
    return plan_path.read_text(), plan_path, _kind_from_body(folder)


def _json_payload(
    *, source: str, issue: int | None, kind: str, overall: bool, results: list[CheckResult]
) -> dict:
    return {
        "source": source,
        "issue": issue,
        "kind": kind,
        "overall": "PASS" if overall else "FAIL",
        "n_fail": sum(1 for r in results if r.status == "FAIL"),
        "n_warn": sum(1 for r in results if r.status == "WARN"),
        "n_skip": sum(1 for r in results if r.status == "SKIP"),
        "checks": [
            {"id": r.id, "name": r.name, "status": r.status, "detail": r.detail} for r in results
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument("--issue", type=int, help="task number to verify (newest plans/v{K}.md)")
    grp.add_argument("--plan-file", help="path to a standalone plan .md to verify")
    parser.add_argument(
        "--kind",
        choices=VALID_KINDS,
        default=None,
        help="task kind (file mode only; default: experiment, the strictest; "
        "ignored in --issue mode, which reads body.md frontmatter)",
    )
    parser.add_argument("--json", action="store_true", help="emit a JSON report instead of text")
    args = parser.parse_args()

    issue: int | None = None
    if args.issue is not None:
        if args.kind is not None:
            print(
                "verify_plan: --kind is ignored in --issue mode (kind is read from "
                "body.md frontmatter)",
                file=sys.stderr,
            )
        try:
            raw, plan_path, kind = _load_plan_for_issue(args.issue)
        except FileNotFoundError as e:
            print(f"verify_plan: {e}", file=sys.stderr)
            return 2
        source = str(plan_path)
        issue = args.issue
    else:
        plan_path = Path(args.plan_file)
        try:
            raw = plan_path.read_text()
        except OSError as e:
            print(f"verify_plan: {e}", file=sys.stderr)
            return 2
        source = args.plan_file
        kind = args.kind or "experiment"

    overall, results = verify_plan_text(raw, kind=kind, source=source)

    # Check 23 (goal currency) needs task context (body.md + events.jsonl),
    # so it runs OUTSIDE verify_plan_text(): appended here in --issue mode,
    # rendered SKIP in --plan-file mode.
    if issue is not None:
        folder = plan_path.parent.parent  # tasks/<status>/<N>/plans/vK.md -> task folder
        mtime = datetime.fromtimestamp(plan_path.stat().st_mtime, tz=UTC)
        cur, sup = _goal_history_for_plan(folder, mtime)
        results.append(check_goal_currency(raw, current_goal=cur, superseded=sup))
    else:
        results.append(
            _skip(
                "c23_goal_currency",
                "plan head not drafted against a superseded Goal",
                "no task context (--plan-file mode; goal history requires --issue)",
            )
        )
    overall = all(r.passed for r in results)

    if args.json:
        print(
            json.dumps(
                _json_payload(
                    source=source, issue=issue, kind=kind, overall=overall, results=results
                ),
                indent=2,
            )
        )
        return 0 if overall else 1

    print(f"verify_plan — {source} (kind: {kind})")
    for r in results:
        print(r.render())
    print()
    n_warn = sum(1 for r in results if r.status == "WARN")
    n_skip = sum(1 for r in results if r.status == "SKIP")
    if overall:
        print(f"OVERALL: PASS ({n_warn} WARN, {n_skip} SKIP)")
        return 0
    n_fail = sum(1 for r in results if r.status == "FAIL")
    print(f"OVERALL: FAIL ({n_fail} of {len(results)} checks failed; {n_warn} WARN, {n_skip} SKIP)")
    return 1


if __name__ == "__main__":
    sys.exit(main())
