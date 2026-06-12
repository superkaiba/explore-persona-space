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

Kind-exempt checks render as [SKIP] (first-class status, distinguishable
from genuine passes — the calibration report needs n_skip separate from
n_pass). Conditional checks (4, 6, 7, 10) also SKIP when their content
trigger does not fire.

Canonical N/A escape phrases (quote verbatim in bounce briefs):

  - ``N/A — no model training`` / ``N/A — no training hyperparameters``
    (check 1)
  - ``N/A — no behavioral construct`` (check 2)
  - ``N/A — no artifact reuse`` (check 6)
  - ``N/A — not a replication`` (check 7)

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
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml

# ─── Constants ─────────────────────────────────────────────────────────────

VALID_KINDS = ("experiment", "analysis", "infra", "batch", "survey")

# Kinds exempt from the experiment-only checks (CLAUDE.md Critical Rules:
# "`kind: analysis|infra|batch|survey` exempt").
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
    n_headings = len(_headings(plan))
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
    # stopping at the first parenthetical or em-dash annotation — NOT the
    # whole line (#610 carries "— worst ≈ 42 — see §9" and #614 carries
    # "1× A100-80" on the same line; a whole-line digit-dash-digit scan  # noqa: RUF003
    # would false-FAIL such annotations).
    line_end = plan.find("\n", m.end())
    if line_end == -1:
        line_end = len(plan)
    tail = plan[m.start(1) : line_end]
    for stop in ("(", "—"):
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
    attestations (a)-(g) (.claude/rules/artifact-reuse.md). WARN not FAIL:
    trigger and item-detection are both heuristic, and the demonstrated
    failure modes (#545/#600/#601) are semantic — the gate's value is
    forcing the section to exist and naming the seven letters."""
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
    letters = {m.group(1) for m in re.finditer(r"\(([a-g])\)", text)}
    if fitness and len(letters) >= 4:
        return _pass(cid, name, f"fitness check present ({len(letters)}/7 lettered items spotted)")
    if fitness:
        return _warn(
            cid,
            name,
            f"fitness vocabulary present but only {len(letters)} of the (a)–(g) items "  # noqa: RUF001
            "detectable — verify all seven attestations (recipe/regime/cells/single-var/"
            "hub-resolution/content-identity/scaling) before approval",
        )
    return _warn(
        cid,
        name,
        "plan reuses HF artifacts but no fitness check found — CLAUDE.md reuse rule requires "
        "attestations (a)–(g); consistency-checker + Methodology critic must gate this",  # noqa: RUF001
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
