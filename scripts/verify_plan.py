#!/usr/bin/env python
"""verify_plan.py — mechanical pre-pass gate for experiment plans (task #625).

Deterministic, sub-second structural verifier for the plans persisted at
``tasks/<status>/<N>/plans/v{K}.md``, run at ``/adversarial-planner``
Phase 1.5.0 BEFORE the fact-checker + critic ensemble spawn. The plan-side
sibling of ``scripts/verify_task_body.py`` (clean-result bodies): pure
regex / string presence checks, NO LLM calls, no network, no side effects
(the orchestrator running the adversarial-planner skill posts the
``epm:plan-verify`` marker — never this script).

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
                                                           exempt kinds WARN
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

Kind-exempt checks render as [SKIP] (first-class status, distinguishable
from genuine passes — the calibration report needs n_skip separate from
n_pass). Conditional checks (4, 6, 7, 10, 11, 12, 13, 14, 15, 16) also SKIP
when their content trigger does not fire.

Canonical N/A escape phrases (quote verbatim in bounce briefs):

  - ``N/A — no model training`` / ``N/A — no training hyperparameters``
    (check 1)
  - ``N/A — no behavioral construct`` (check 2)
  - ``N/A — no artifact reuse`` (check 6)
  - ``N/A — not a replication`` (check 7)
  - ``N/A — no dry-run smoke`` (check 11)
  - ``N/A — no draw battery`` (check 12)
  - ``N/A — no empirical-null gate`` (check 13)
  - ``N/A — no fail-loud acceptance claim`` /
    ``N/A — fail-loud claim not test-backable`` (check 15)
  - ``N/A — no re-extracted reference arms`` (check 16)

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
    if re.search(
        NA_RE + r"no (?:model )?(?:training )?(?:model training|hyperparameters|training)", scope
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
            "no Decision Rationale / grounding section and zero Source: entries — every "
            "load-bearing hyperparameter needs a Source (planner.md §11); if the plan trains "
            "no model, declare `N/A — no model training` / `N/A — no training hyperparameters`",
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
            "§11-style section present but zero Source entries (inline `Source:` label or "
            "a `Source` table column)",
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
    if re.search(NA_RE + r"no behavioral construct", plan):
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
        "`N/A — no behavioral construct`)",
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
    if re.search(r"(?i)not a behavior[- ]implantation", text):
        return _pass(cid, name, "explicit N/A declared (not a behavior-implantation)")
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
        "Methodology critic must gate this",
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
    forcing the section to exist and naming the ten letters."""
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
    if re.search(NA_RE + r"no (?:artifact )?reuse", text):
        return _pass(cid, name, "explicit no-reuse declaration (N/A — no artifact reuse)")
    fitness = re.search(r"(?i)fitness", text)
    letters = {m.group(1) for m in re.finditer(r"\(([a-j])\)", text)}
    if fitness and len(letters) >= 4:
        return _pass(cid, name, f"fitness check present ({len(letters)}/10 lettered items spotted)")
    if fitness:
        return _warn(
            cid,
            name,
            f"fitness vocabulary present but only {len(letters)} of the (a)–(j) items "  # noqa: RUF001
            "detectable — verify all ten attestations (recipe/regime/cells/single-var/"
            "hub-resolution/content-identity/scaling/backend-fetchability/code-throughput/"
            "pair-provenance) "
            "before approval",
        )
    return _warn(
        cid,
        name,
        "plan reuses HF artifacts but no fitness check found — CLAUDE.md reuse rule requires "
        "attestations (a)–(j); consistency-checker + Methodology critic must gate this",  # noqa: RUF001
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
    if re.search(r"(?i)not a replication", text):
        return _pass(cid, name, "explicit N/A declared (not a replication)")
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
        "Goal mentions replication but no fidelity vocabulary (paper's data/recipe, "
        "faithful, deviations) — CLAUDE.md replication rule: match the paper's data + "
        "recipe FIRST, name every deviation",
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


def check_success_kill(plan: str, kind: str) -> CheckResult:
    """Both a success-criteria family and a kill-criteria family must be
    present and non-empty in form (each carrier section ≥ 80 chars —
    emptiness check only; semantic joint-satisfiability stays with the
    Statistics critic per planner.md §7). The KILL count EXCLUDES the
    §0.0/TL;DR region. `kind: experiment` FAILs on both-absent; exempt
    kinds WARN."""
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
    for i, line in enumerate(lines):
        if mask[i]:
            continue
        m = _SUCCESS_RE.search(line)
        if m:
            succ_hits.append((i, m.group(0)))
        m = _KILL_RE.search(line)
        if m and not in_tldr(i):
            kill_hits.append((i, m.group(0)))

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
    if re.search(NA_RE + r"no dry-?run smoke", plan):
        return _pass(cid, name, "explicit N/A declared (no dry-run smoke)")
    evidence = _dryrun_test_evidence_lines(plan)
    if evidence:
        return _pass(cid, name, f"dry-run-exercising test named ({evidence[0][:80]!r})")
    return _warn(
        cid,
        name,
        "plan names a `--dry-run` smoke/verification command but the test list has no test "
        "exercising `dry_run=True` on the new code path — a broken dry_run thread turns the "
        "final smoke into a real mutation (#596/#607/#633 pattern); add the test, or declare "
        "`N/A — no dry-run smoke` if the flag mention is incidental",
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
_GRID_FACTOR = (
    r"(?:\d[\d,_]*|cells|folds|arms|layers|traits|seeds"
    r"|behaviors|settings|conditions|statistics)"
)
# The multiplication token plans actually write: the real multiplication
# sign plus the ASCII fallbacks.
_MULT_TOKEN = r"[×x*]"  # noqa: RUF001 — the multiplication sign is real plan text
_MULT_ARITH_RE = re.compile(
    rf"(?i)\b(?:{_DRAW_FACTOR}\s*{_MULT_TOKEN}\s*{_GRID_FACTOR}"
    rf"|{_GRID_FACTOR}\s*{_MULT_TOKEN}\s*{_DRAW_FACTOR})\b"
)

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
    (``_BATTERY_TRIGGER_RE``, ±15) and c16 (``_C16_EXTRACT_RE`` ±3;
    ``_C16_REGEN_RE`` at radius 0 = same-line adjacency)."""
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
    commitment. Window-scoped, never document-global — the document-global
    draft demonstrably false-PASSed the motivating incident plan (#810 v1)
    via an unrelated footprint product + helper boilerplate. FAIL
    (experiment) / WARN (analysis) / SKIP otherwise; a SURFACE check per the
    module's scope discipline — semantic adequacy of the arithmetic stays
    with the Phase 2 critics."""
    cid, name = "c12_battery_multiplier", "battery multiplier + batched commitment"
    if kind not in ("experiment", "analysis"):
        return _skip(cid, name, "kind-exempt: battery sizing is an experiment|analysis plan shape")
    windows = _battery_trigger_windows(plan)
    if not windows:
        return _skip(cid, name, "no permutation/null-draw battery named")
    if re.search(NA_RE + r"no draw battery", plan):
        return _pass(cid, name, "explicit N/A declared (no draw battery)")
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
            "(draws x cells x folds x per-call cost = projected wall)"
        )
    if not any_commit:
        missing.append(
            "a batched-implementation commitment (a named batched helper or an explicit "
            "vectorization statement)"
        )
    if not missing:
        missing.append(
            "co-location: the multiplier arithmetic and the batched-implementation "
            "commitment each appear somewhere, but never together near any battery mention"
        )
    detail = (
        f"plan names a permutation/bootstrap/null battery but is missing {' AND '.join(missing)}"
        " — a named battery defaults to a serial per-draw loop (#778: ~15 h realized vs 1 h"
        " planned; #810: 308x); see .claude/rules/vectorize-many-cell-fits.md, or declare"
        " 'N/A — no draw battery' if the mention is incidental"
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


def _c13_na_escape_declared(plan: str) -> bool:
    """True when the ``N/A — no empirical-null gate`` escape appears as a
    deliberate STANDALONE declaration line (leading list/blockquote markers
    stripped), never doc-global: the c13 FAIL detail quotes the escape phrase
    as a remedy option, and this project's convention pastes verifier/bounce
    text into revised plans verbatim — a substring match would let a bounced
    plan self-escape re-verification (the #810 spurious-satisfaction
    structure, one polarity over). NA_RE opens with an inline (?i), so it
    must sit at pattern position 0 — per-line re.match satisfies that; never
    prepend a prefix to NA_RE (py3.11+ rejects mid-pattern global flags)."""
    lines = plan.splitlines()
    mask = _fence_mask(lines)
    for line, fenced in zip(lines, mask, strict=True):
        if fenced:
            continue
        if re.match(NA_RE + r"no empirical[- ]null gate", line.lstrip(" \t>*-")):
            return True
    return False


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
        "'N/A — no empirical-null gate' on its own line"
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
    comparator / SKIP otherwise. Incident: #816 v5 (gate p ≤ 0.05 over
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
        return _pass(cid, name, "explicit N/A declared (no empirical-null gate)")
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

# Anchor carriers that never bind: §0.0 TL;DR / §0 Plan Summary restate
# criteria as summary prose (same rationale as c8's _tldr_ranges exclusion).
_FAILLOUD_SUMMARY_HEAD_RE = re.compile(r"(?i)tl;dr|plan summary|^(?:§\s*)?0(?:\.0)?\b")

_FAILLOUD_WINDOW_LINES = 30

_FAILLOUD_NA_RE = re.compile(
    NA_RE + r"(?:no fail[- ]?loud acceptance claim|fail[- ]?loud claim not test-backable)"
)


def _failloud_claim_hits(plan: str) -> list[tuple[str, str]]:
    """(section heading, matched vocabulary) per acceptance/success anchor
    whose fence-stripped 30-line window carries a fail-loud claim. Anchors in
    fences, in §0/TL;DR/Plan-Summary regions, or with an H1/preamble carrier
    are dropped (corpus-probe noise classes, task #932)."""
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
        end = min(h.end, i + 1 + _FAILLOUD_WINDOW_LINES)
        m = _FAILLOUD_CLAIM_RE.search(strip_fences("\n".join(lines[i:end])))
        if m:
            hits.append((h.text, m.group(0)))
    return hits


def _failloud_test_evidence_lines(plan: str) -> list[str]:
    """RAW-plan lines naming a committed fail-loud-exercising test: a
    ``test_`` identifier (also matches tests/<file>.py paths) co-located with
    fail-loud vocabulary, grep-command lines excluded."""
    out: list[str] = []
    for line in plan.splitlines():
        if _FAILLOUD_GREP_LINE_RE.search(line):
            continue
        if _TEST_IDENT_RE.search(line) and _FAILLOUD_TEST_EVIDENCE_RE.search(line):
            out.append(line.strip())
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
    replay covered infra|batch only)."""
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
    if _FAILLOUD_NA_RE.search(plan):
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
        "pinning test, or declare `N/A — no fail-loud acceptance claim` / "
        "`N/A — fail-loud claim not test-backable`",
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
_C16_NA_RE = re.compile(NA_RE + r"no re-?extracted reference arms")


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
    if _C16_NA_RE.search(text):
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
        "clean-result, but no sentence distinguishes same-pass comparator values from "
        "the prior committed headline values — state which values adjudicate this "
        "round's NEW comparison vs which remain the committed headline, and that a "
        "flipped reference CALL is reported as replication-stability evidence rather "
        "than replacing the headline (#811 v3 §6 incident; the committed-cells-"
        "remain-evidence rule), or declare `N/A — no re-extracted reference arms`",
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
