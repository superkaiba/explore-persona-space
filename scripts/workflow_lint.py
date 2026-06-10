"""Lint ``.claude/workflow.yaml`` against its Pydantic schema.

Callable from a pre-commit hook AND importable for unit tests.

Behaviours:

* ``--check-references`` (default in pre-commit): walk ``CLAUDE.md``,
  ``.claude/skills/issue/SKILL.md``, and ``.claude/skills/issue/markers.md``;
  every ``(see workflow.yaml § <key>)`` reference MUST resolve to a real
  YAML key.
* ``--emit-tables``: regenerate the auto-generated table blocks in
  ``markers.md`` and ``SKILL.md`` ("Active vs awaiting-user" table) inside
  the fenced ``<!-- workflow.yaml: AUTO-GENERATED -->`` … ``<!--
  /workflow.yaml: AUTO-GENERATED -->`` markers. Hand-edits inside those
  fences are rejected by the lint.
* ``--check-tables`` (default in pre-commit): compare the rendered tables
  against the on-disk markdown; FAIL on drift.
* ``--check-script-refs`` (also bundled into ``--check-references`` and the
  no-flags default run): walk every ``.md`` under ``.claude/agents/`` and
  every ``SKILL.md`` under ``.claude/skills/`` (excluding OTHER worktrees
  under ``.claude/worktrees/<name>/`` — the worktree we are currently
  running from IS scanned so workflow-improver can validate its own edits;
  see :func:`_other_worktree_prefix` for the scoping rule) and FAIL on
  any ``scripts/<name>.py`` reference whose target does not exist under
  ``scripts/``. Mechanically prevents the dead-tool / invented-tool
  failure class where an agent follows a step that runs a
  deleted-or-never-created helper and CalledProcessErrors.
* ``--check-wandb-required``: walk every ``*.py`` under
  ``src/explore_persona_space/experiments/`` whose source mentions a
  trainer-config builder (``TrainLoraConfig``, ``SFTConfig``,
  ``TrainingArguments``) and FAIL on any ``report_to="none"`` /
  ``report_to=None`` / ``report_to=[]`` literal that is not waived by a
  ``# WANDB_INTENTIONALLY_DISABLED: <reason>`` comment on the same line
  or the immediately preceding non-blank line. Closes the gap that hid
  task #496's missing live-training telemetry (12 cells trained with
  ``report_to="none"`` and no waiver; smoke + code-review + pre-launch
  all passed). CLAUDE.md "Upload Policy" makes WandB live metrics
  mandatory for training; this lint enforces it mechanically.

Exit codes:

* ``0`` PASS
* ``1`` FAIL — stderr lists every error with file:line context.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# Allow `python scripts/workflow_lint.py` from a fresh shell without `uv run`
# by extending sys.path to the project src/.
_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from explore_persona_space.workflow import (  # noqa: E402  (import after sys.path edit)
    WorkflowYaml,
    load_workflow_yaml,
)

# Scope for reference-resolution. Mirrors the pre-commit hook `files:` regex
# in `.pre-commit-config.yaml` so the lint and the trigger stay in sync.
DOC_FILES: tuple[Path, ...] = (
    _REPO_ROOT / "CLAUDE.md",
    _REPO_ROOT / ".claude" / "skills" / "issue" / "SKILL.md",
    _REPO_ROOT / ".claude" / "skills" / "issue" / "markers.md",
)

REFERENCE_RE = re.compile(r"\(see\s+workflow\.yaml\s+§\s+([a-z_.]+(?:\.[a-z_-]+)*)\s*\)")
AUTO_GEN_OPEN = "<!-- workflow.yaml: AUTO-GENERATED"
AUTO_GEN_CLOSE = "<!-- /workflow.yaml: AUTO-GENERATED -->"

# Collected from `gh_project.py` consumers of `LABEL_TO_COLUMN` —
# every status:* label in code MUST resolve to a workflow.yaml status row.
STATUS_LABEL_RE = re.compile(r"\bstatus:[a-z][a-z0-9-]*\b")

# `--check-script-refs`: every `scripts/<name>.py` token mentioned in an
# agent / skill spec MUST resolve to a real file under `scripts/`.
# Word-boundary-anchored on the left so `my_scripts/foo.py` (a different
# path) doesn't match; the leading `scripts/` segment must stand alone.
SCRIPT_REF_RE = re.compile(r"(?<![\w/])scripts/([A-Za-z0-9_]+\.py)\b")

# `--check-wandb-required`: every `report_to="none"` (or equivalent
# disabling literal: `report_to=None`, `report_to=[]`) inside a training-
# config builder under `src/explore_persona_space/experiments/` MUST
# carry a waiver comment. CLAUDE.md "Upload Policy" treats WandB live
# training metrics as a mandatory artifact; this check makes the gap
# detectable at lint time, not after a 12-cell run completes (#496).
#
# Waiver convention: a comment of the form
#
#     # WANDB_INTENTIONALLY_DISABLED: <reason>
#
# on the same line as the `report_to=` token, OR on the immediately
# preceding non-blank line. The reason must be ≥10 chars after the colon
# (the goal is "force the implementer to justify it in writing", not
# "tick a box with WANDB_INTENTIONALLY_DISABLED: x"). Eval-only call
# sites and tests are out of scope by directory.
WANDB_DISABLED_RE = re.compile(
    r"\breport_to\s*=\s*(?:[\"']none[\"']|[\"']None[\"']|None\b|\[\s*\])"
)
WANDB_WAIVER_RE = re.compile(r"#\s*WANDB_INTENTIONALLY_DISABLED\s*:\s*(.+?)\s*$")
WANDB_WAIVER_MIN_REASON_CHARS = 10
# Trainer-config builders that exist solely to launch live training; a
# `report_to="none"` literal in the same file as one of these names is
# almost always a hardcoded telemetry kill (the warmth-sycophancy #496
# pattern). Files lacking any of these are skipped — they're either pure
# eval rigs, data-prep utilities, or analyzers, where WandB is not
# expected.
WANDB_TRAINER_CONFIG_TOKENS: tuple[str, ...] = (
    "TrainLoraConfig",
    "SFTConfig",
    "TrainingArguments",
)

# `--check-asks`: every `AskUserQuestion` mention in agent/skill specs must
# be anchored to a documented gate or marked as anti-pattern documentation.
# Three accepted anchor forms (see `check_asks` docstring for the full rule).
ASK_RE = re.compile(r"\bAskUserQuestion\b")
# Permissive match: accepts uppercase keys so the lint can emit a precise
# "does not resolve" error for malformed annotations like
# ``<!-- gate: gates.WRONG_CASE -->`` instead of falling through to the
# generic "bare mention" message.
GATE_ANNOTATION_RE = re.compile(r"<!--\s*gate:\s*([A-Za-z0-9_.\-]+)\s*-->")
ANTI_PATTERN_RE = re.compile(r"<!--\s*example:\s*anti-pattern\s*-->")
# Window above the AskUserQuestion line scanned for an existing `(see workflow.yaml § gates.X)`
# citation. Five lines covers paragraph-style prose anchors without leaking into the next block.
ASK_CITE_LOOKBACK = 5
# Permissive citation regex for `--check-asks` Rule 3: matches both the
# canonical `(see workflow.yaml § gates.X)` parenthesized form AND the
# bare prose form `workflow.yaml § gates.X` (used in existing
# documentation, e.g. SKILL.md:449 "gate #6 — see workflow.yaml §
# gates.inline)"). The strict `_check_references` check uses the
# canonical-only REFERENCE_RE; this looser variant exists purely to
# anchor AskUserQuestion mentions to a documented gate without forcing
# the prose to be rewritten.
ASK_CITE_RE = re.compile(r"workflow\.yaml\s+§\s+(gates(?:\.[a-z_-]+)*)\b")

# `--check-autonomous-asks`: every `AskUserQuestion` mention in
# `.claude/skills/issue/SKILL.md` and `.claude/agents/*.md` MUST document
# its autonomous-mode behavior. Three accepted anchor forms (any one
# satisfies the rule), looked for in the SAME paragraph as the
# `AskUserQuestion` mention (paragraph = block bounded by blank lines,
# same convention as ``check_asks``):
#
# 1. Literal "Interactive mode" / "interactive mode" — flags the ask as
#    interactive-only, implying an autonomous-mode auto-resolve elsewhere.
# 2. Literal "EPM_AUTONOMOUS_SESSION" — references the autonomous env
#    flag explicitly, typically inside a branch-on-mode prose block.
# 3. Annotation comment ``<!-- autonomous-mode: <action> -->`` where
#    `<action>` is one of `auto-resolve` | `skip` | `block-and-fail` |
#    `gate-allowed`. The `gate-allowed` value is for the two gates where
#    the ask is legitimate in autonomous mode (none today; this is a
#    forward-compat escape hatch).
#
# An AskUserQuestion mention inside an ``<!-- example: anti-pattern -->``
# paragraph is exempt (same exemption as ``check_asks``). The check exists
# specifically to prevent the #503/#504/#505 incident (2026-06-05): three
# autonomous sessions sat blocked on a 4-option choice menu because the
# SKILL.md prose didn't enumerate the autonomous-mode auto-resolve for
# the conditional pivot gates.
AUTONOMOUS_INTERACTIVE_RE = re.compile(r"interactive mode", re.IGNORECASE)
AUTONOMOUS_ENV_RE = re.compile(r"EPM_AUTONOMOUS_SESSION")
AUTONOMOUS_ANNOTATION_RE = re.compile(
    r"<!--\s*autonomous-mode:\s*(auto-resolve|skip|block-and-fail|gate-allowed)\s*-->"
)


def _flatten_keys(workflow: WorkflowYaml) -> set[str]:
    """Return the set of dotted keys that ``(see workflow.yaml § <k>)``
    references can resolve to. Includes top-level keys, per-row identifier
    keys (e.g. ``statuses.running``), and the Phase B blocks
    ``ensemble_review`` / ``reviewer_pairs``."""
    keys: set[str] = {
        "version",
        "issue_types",
        "columns",
        "statuses",
        "priority_labels",
        "gates",
        "gates.inline",
        "gates.park_and_wait",
        "gates.conditional",
        "halt_criteria",
        "subagent_halt_conditions",
        "ensemble_review",
        "ensemble_review.doubled_steps",
        "reviewer_pairs",
        "markers",
        "steps",
    }
    for c in workflow.columns:
        keys.add(f"columns.{c.name}")
    for s in workflow.statuses:
        keys.add(f"statuses.{s.name}")
    for p in workflow.priority_labels:
        keys.add(f"priority_labels.{p.name}")
    if workflow.gates is not None:
        for g in workflow.gates.inline + workflow.gates.park_and_wait + workflow.gates.conditional:
            keys.add(f"gates.{g.name}")
    for h in workflow.halt_criteria:
        keys.add(f"halt_criteria.{h.name}")
    for row in workflow.subagent_halt_conditions:
        keys.add(f"subagent_halt_conditions.{row.subagent}")
    if workflow.ensemble_review is not None:
        for entry in workflow.ensemble_review.doubled_steps:
            keys.add(f"ensemble_review.doubled_steps.{entry.role}")
    for m in workflow.markers:
        keys.add(f"markers.{m.kind}")
    for step in workflow.steps:
        keys.add(f"steps.{step.id}")
    return keys


def _check_references(workflow: WorkflowYaml) -> list[str]:
    """Walk DOC_FILES and report unresolved ``(see workflow.yaml § X)``
    references."""
    errors: list[str] = []
    keys = _flatten_keys(workflow)
    for path in DOC_FILES:
        if not path.exists():
            continue
        for lineno, line in enumerate(path.read_text().splitlines(), start=1):
            for match in REFERENCE_RE.finditer(line):
                ref = match.group(1)
                if ref not in keys:
                    errors.append(
                        f"{path}:{lineno}: unresolved reference "
                        f"'(see workflow.yaml § {ref})' — not in workflow.yaml"
                    )
    return errors


def _other_worktree_prefix(repo_root: Path) -> str | None:
    """Return the substring that identifies OTHER worktrees so we can
    exclude their copies without also excluding the current worktree we
    are running from.

    The lint script's :data:`_REPO_ROOT` is derived from ``__file__``, so
    it resolves to whichever tree contains the copy of
    ``scripts/workflow_lint.py`` that Python loaded — main checkout when
    invoked from main, or a specific worktree when invoked from a
    worktree. Behaviour:

    * Invoked from ``/.../explore-persona-space`` (main checkout): no
      worktree is "current", so EVERY ``.claude/worktrees/<X>/`` copy is
      a stale duplicate that must be excluded — return the bare
      ``".claude/worktrees/"`` substring (original behaviour).
    * Invoked from ``/.../explore-persona-space/.claude/worktrees/<X>``
      (a worktree): scanning ``<X>``'s own files is exactly what
      ``workflow-improver`` needs to validate its edits, but scanning
      OTHER worktrees ``<Y>``, ``<Z>``, … is wrong (stale duplicates) —
      AND the worktree's own ``.claude/skills/**/SKILL.md`` paths contain
      ``.claude/worktrees/`` as a substring, so a naive
      ``".claude/worktrees/"`` exclusion drops everything. Resolution:
      walk to the worktree-name ancestor (``<X>``) and return the
      sibling-exclusion substring ``".claude/worktrees/"`` paired with
      the rule "exclude only if the path ALSO contains a worktree name
      that is NOT ``<X>``". Implementation-wise we just return the path
      up to and including the worktree dir (e.g. ``.claude/worktrees/<X>/``)
      so a caller can build the exclusion as "path contains
      ``.claude/worktrees/`` but does NOT contain this prefix".

    Returns the "this worktree's prefix" substring (e.g.
    ``.claude/worktrees/agent-a29cd29.../``) when running inside a
    worktree, or ``None`` when running from main.
    """
    # Look for a `.claude/worktrees/<name>` segment in the parent chain.
    # Scan ALL occurrences of "worktrees" — a stray directory named
    # `worktrees` higher up the path (e.g. /home/foo/worktrees/baz/.claude/...)
    # must NOT short-circuit the search and miss a real `.claude/worktrees/<name>`
    # further down. The match must be preceded by `.claude` and followed
    # by a name segment.
    parts = repo_root.parts
    for idx in range(len(parts)):
        if parts[idx] != "worktrees":
            continue
        if idx == 0 or parts[idx - 1] != ".claude" or idx + 1 >= len(parts):
            continue
        # Build the prefix substring up through the worktree-name segment,
        # WITH a trailing slash so a sibling worktree `<X>-other/` does
        # not match `<X>/`.
        return f".claude/worktrees/{parts[idx + 1]}/"
    return None


def _is_other_worktree_path(path: Path, current_worktree_prefix: str | None) -> bool:
    """Return True iff ``path`` lives under a DIFFERENT worktree than the
    one we are currently running from.

    * Running from main (``current_worktree_prefix is None``): every
      ``.claude/worktrees/`` path is "other".
    * Running from a worktree: a path under our own worktree (matching
      ``current_worktree_prefix``) is NOT "other"; only paths under a
      sibling worktree (``.claude/worktrees/`` present but our prefix
      absent) are.
    """
    s = str(path)
    if ".claude/worktrees/" not in s:
        return False
    if current_worktree_prefix is None:
        return True
    return current_worktree_prefix not in s


def _iter_ask_target_files(repo_root: Path) -> list[Path]:
    """Return the sorted list of files in ``--check-asks`` scope:
    every ``.md`` under ``.claude/agents/`` and every ``SKILL.md`` under
    ``.claude/skills/``, excluding paths that belong to OTHER worktrees
    (frozen sibling copies that are not authoritative). The worktree we
    are currently running from IS scanned so a workflow-improver running
    inside a worktree can validate its own edits.
    """
    agents_root = repo_root / ".claude" / "agents"
    skills_root = repo_root / ".claude" / "skills"
    current_prefix = _other_worktree_prefix(repo_root)
    files: list[Path] = []
    if agents_root.exists():
        files.extend(
            p
            for p in agents_root.glob("*.md")
            if p.is_file() and not _is_other_worktree_path(p, current_prefix)
        )
    if skills_root.exists():
        files.extend(
            p
            for p in skills_root.glob("**/SKILL.md")
            if p.is_file() and not _is_other_worktree_path(p, current_prefix)
        )
    return sorted(files)


def _ask_paragraph_bounds(lines: list[str], idx: int) -> tuple[int, int]:
    """Return (up_start, down_end) — the paragraph window around an
    AskUserQuestion mention at line index ``idx``. The window stops at
    blank-line paragraph boundaries above AND below, capped at
    :data:`ASK_CITE_LOOKBACK` lines on either side."""
    up_start = max(0, idx - ASK_CITE_LOOKBACK)
    for back in range(idx - 1, up_start - 1, -1):
        if lines[back].strip() == "":
            up_start = back + 1
            break
    down_end = idx + 1
    forward_cap = idx + 1 + ASK_CITE_LOOKBACK
    while down_end < len(lines) and down_end < forward_cap:
        if lines[down_end].strip() == "":
            break
        down_end += 1
    return up_start, down_end


def _ask_mention_error(path: Path, idx: int, lines: list[str], keys: set[str]) -> str | None:
    """Return a lint error string for one AskUserQuestion mention, or
    None if the mention is properly anchored. Rules 1/2/3 are documented
    on :func:`check_asks`."""
    up_start, down_end = _ask_paragraph_bounds(lines, idx)
    up_window_text = "\n".join(lines[up_start : idx + 1])
    # Rule 1: <!-- gate: <key> --> resolving to a real gate.
    gate_match = GATE_ANNOTATION_RE.search(up_window_text)
    if gate_match:
        gate_key = gate_match.group(1)
        if gate_key in keys:
            return None
        return (
            f"{path}:{idx + 1}: '<!-- gate: {gate_key} -->' does not "
            f"resolve to a workflow.yaml gate key. Valid examples: "
            f"gates.plan_approval, gates.experiment_goal, "
            f"gates.awaiting_promotion. See CLAUDE.md auto-continuation "
            f"policy."
        )
    # Rule 2: <!-- example: anti-pattern --> marker.
    if ANTI_PATTERN_RE.search(up_window_text):
        return None
    # Rule 3: existing workflow.yaml § gates.X reference anywhere in the
    # same paragraph (above OR below the mention). Accepts both the
    # canonical (see workflow.yaml § gates.X) form and the bare-prose
    # workflow.yaml § gates.X form (used by some existing documentation).
    paragraph_text = "\n".join(lines[up_start:down_end])
    for ref_match in ASK_CITE_RE.finditer(paragraph_text):
        if ref_match.group(1) in keys:
            return None
    return (
        f"{path}:{idx + 1}: bare 'AskUserQuestion' mention outside any "
        f"documented gate. Annotate with '<!-- gate: <key> -->' "
        f"(key must resolve in workflow.yaml § gates), or mark the "
        f"surrounding paragraph as '<!-- example: anti-pattern -->'. "
        f"See CLAUDE.md auto-continuation policy."
    )


def _resolve_ask_target_files(roots: list[Path] | None) -> list[Path]:
    """Production callers pass ``roots=None`` and we walk the canonical
    agent + skill trees. Tests pass ``roots=[tmp_path]`` to scope the
    walk to a fixture directory."""
    if roots is None:
        return _iter_ask_target_files(_REPO_ROOT)
    files: list[Path] = []
    for root in roots:
        if root.is_file():
            files.append(root)
        else:
            files.extend(p for p in root.glob("**/*.md") if p.is_file())
    return sorted(files)


def check_asks(workflow: WorkflowYaml, *, roots: list[Path] | None = None) -> list[str]:
    """Walk ``.claude/agents/**.md`` + ``.claude/skills/**/SKILL.md`` and
    enforce the auto-continuation contract: every ``AskUserQuestion``
    mention must be anchored to a documented gate or marked as
    documentation.

    A line containing ``AskUserQuestion`` PASSES if ANY of these hold:

    1. The same line OR up to :data:`ASK_CITE_LOOKBACK` lines above
       (stopping at the first blank line) contains ``<!-- gate: <key> -->``
       AND ``<key>`` resolves to a real entry in
       ``_flatten_keys(workflow)`` (e.g. ``gates.plan_approval``).
    2. The same line OR up to :data:`ASK_CITE_LOOKBACK` lines above
       (stopping at the first blank line) contains
       ``<!-- example: anti-pattern -->``.
    3. The surrounding paragraph (bounded by blank lines above AND
       below, capped at :data:`ASK_CITE_LOOKBACK` lines on each side)
       contains a ``workflow.yaml § gates.<key>`` reference that
       resolves. This is the safety valve for prose paragraphs that
       already cite a gate via the existing convention (no need to also
       stamp a redundant ``<!-- gate: ... -->`` comment). The citation
       regex is permissive: it accepts both the canonical
       ``(see workflow.yaml § gates.X)`` form and the bare-prose
       ``workflow.yaml § gates.X`` form.

    FAILs otherwise. Each failure prints ``<file>:<line>`` + a pointer to
    the auto-continuation contract in ``CLAUDE.md``.

    ``roots`` is an override hook for unit tests; production callers pass
    None and the function walks the canonical agent + skill trees under
    ``_REPO_ROOT``.
    """
    errors: list[str] = []
    keys = _flatten_keys(workflow)
    for path in _resolve_ask_target_files(roots):
        lines = path.read_text().splitlines()
        for idx, line in enumerate(lines):
            if not ASK_RE.search(line):
                continue
            err = _ask_mention_error(path, idx, lines, keys)
            if err is not None:
                errors.append(err)
    return errors


def _autonomous_ask_paragraph_bounds(lines: list[str], idx: int) -> tuple[int, int]:
    """Wider paragraph bounds for the autonomous-asks check.

    The basic ``_ask_paragraph_bounds`` is capped at 5 lines on each side
    (it's the citation-window for ``check_asks``). The autonomous-mode
    documentation often lives in a parent section above a long bulleted
    list, so we walk back to the NEAREST blank line above (uncapped) and
    walk forward to the next blank line (uncapped). The forward walk is
    also capped at the next H2/H3/H4 header (`## `, `### `, `#### `) to
    avoid swallowing the next section's content.
    """
    up_start = 0
    for back in range(idx - 1, -1, -1):
        if lines[back].strip() == "":
            up_start = back + 1
            break
    down_end = idx + 1
    while down_end < len(lines):
        line_stripped = lines[down_end].strip()
        if line_stripped == "":
            break
        # Stop at a header boundary so we don't leak into the next section.
        if line_stripped.startswith(("## ", "### ", "#### ")):
            break
        down_end += 1
    return up_start, down_end


def _autonomous_ask_error(path: Path, idx: int, lines: list[str]) -> str | None:
    """Return a lint error string if the ``AskUserQuestion`` mention at
    line ``idx`` lacks autonomous-mode documentation in its enclosing
    paragraph / section block, or None if the mention is properly
    anchored. See :func:`check_autonomous_asks` for the full rule.
    """
    up_start, down_end = _autonomous_ask_paragraph_bounds(lines, idx)
    paragraph_text = "\n".join(lines[up_start:down_end])
    # Exemption: `<!-- example: anti-pattern -->` paragraphs are
    # documentation, not actual call sites — same convention as `check_asks`.
    if ANTI_PATTERN_RE.search(paragraph_text):
        return None
    # Any one of the three anchors satisfies the rule.
    if AUTONOMOUS_INTERACTIVE_RE.search(paragraph_text):
        return None
    if AUTONOMOUS_ENV_RE.search(paragraph_text):
        return None
    if AUTONOMOUS_ANNOTATION_RE.search(paragraph_text):
        return None
    return (
        f"{path}:{idx + 1}: 'AskUserQuestion' mention is missing autonomous-mode "
        f"documentation. The enclosing section block (bounded by the nearest "
        f"blank line above and the next blank line or markdown header below) "
        f"must contain one of: the phrase 'Interactive mode', the literal "
        f"'EPM_AUTONOMOUS_SESSION', or '<!-- autonomous-mode: "
        f"<auto-resolve|skip|block-and-fail|gate-allowed> -->'. This prevents "
        f"the #503/#504/#505 incident (2026-06-05): an AskUserQuestion path "
        f"that has no documented autonomous-mode handling blocks the "
        f"session at run time. The PreToolUse hook in .claude/settings.json "
        f"is the runtime backstop; this lint check forces the docs to "
        f"match. See CLAUDE.md 'STATE-TO-`blocked` criteria' + "
        f".claude/skills/issue/SKILL.md § Autonomous session behavior."
    )


def _resolve_autonomous_ask_target_files(roots: list[Path] | None) -> list[Path]:
    """The autonomous-asks check is narrower than ``check_asks``: it only
    scopes to ``.claude/skills/issue/SKILL.md`` (the per-issue orchestrator
    that ever runs in autonomous mode) and the agents it dispatches. Other
    skills (``/daily``, ``/weekly``, ``/pm``, etc.) never run under
    ``EPM_AUTONOMOUS_SESSION``, so an AskUserQuestion in them is fine
    without the autonomous-mode annotation.
    """
    if roots is not None:
        files: list[Path] = []
        for root in roots:
            if root.is_file():
                files.append(root)
            else:
                files.extend(p for p in root.glob("**/*.md") if p.is_file())
        return sorted(files)
    # Production scope: only the issue orchestrator + its agents.
    issue_skill = _REPO_ROOT / ".claude" / "skills" / "issue" / "SKILL.md"
    agents_dir = _REPO_ROOT / ".claude" / "agents"
    files = []
    if issue_skill.exists():
        files.append(issue_skill)
    if agents_dir.is_dir():
        files.extend(p for p in agents_dir.glob("*.md") if p.is_file())
    return sorted(files)


def check_autonomous_asks(*, roots: list[Path] | None = None) -> list[str]:
    """Walk ``.claude/skills/issue/SKILL.md`` and ``.claude/agents/*.md``
    and FAIL on any ``AskUserQuestion`` mention whose surrounding
    paragraph does not document the autonomous-mode behavior.

    A line containing ``AskUserQuestion`` PASSES if its surrounding
    paragraph (bounded by blank lines) contains ANY of:

    1. The phrase ``Interactive mode`` / ``interactive mode`` — flags
       the ask as interactive-only, implying an autonomous-mode
       auto-resolve elsewhere.
    2. The literal ``EPM_AUTONOMOUS_SESSION`` — references the
       autonomous env flag explicitly, typically inside a branch-on-mode
       prose block that handles autonomous mode separately.
    3. The annotation ``<!-- autonomous-mode: <action> -->`` where
       ``<action>`` is one of ``auto-resolve``, ``skip``,
       ``block-and-fail``, or ``gate-allowed``.

    Exemption: paragraphs marked ``<!-- example: anti-pattern -->`` are
    documentation, not actual call sites, and are skipped.

    Rationale: the #503/#504/#505 incident (2026-06-05) had three
    autonomous Happy sessions sit blocked indefinitely on a 4-option
    choice menu because the SKILL.md prose did not enumerate the
    autonomous-mode auto-resolve for the conditional pivot gates. The
    runtime backstop is the PreToolUse hook in ``.claude/settings.json``
    (which now blocks ANY ``AskUserQuestion`` in autonomous mode); this
    lint forces the docs to match so an ask without a documented
    autonomous-mode path can never land on `main`.

    ``roots`` is an override hook for unit tests; production callers
    pass None and the function walks the canonical issue-orchestrator
    surface (``.claude/skills/issue/SKILL.md`` + ``.claude/agents/*.md``).
    """
    errors: list[str] = []
    for path in _resolve_autonomous_ask_target_files(roots):
        lines = path.read_text().splitlines()
        for idx, line in enumerate(lines):
            if not ASK_RE.search(line):
                continue
            err = _autonomous_ask_error(path, idx, lines)
            if err is not None:
                errors.append(err)
    return errors


def _check_status_label_coverage(workflow: WorkflowYaml) -> list[str]:
    """Every ``status:*`` literal that appears in ``scripts/gh_project.py``
    consumers MUST resolve to a status name in workflow.yaml. Today's
    consumers: ``scripts/gh_project.py``."""
    errors: list[str] = []
    valid = {f"status:{s.name}" for s in workflow.statuses}
    target = _REPO_ROOT / "scripts" / "gh_project.py"
    if not target.exists():
        return errors
    for lineno, line in enumerate(target.read_text().splitlines(), start=1):
        # Skip strings inside docstrings to reduce noise; this is a coarse
        # filter — comments are checked too because dropped status names in
        # comments are usually also dropped in code.
        for match in STATUS_LABEL_RE.finditer(line):
            ref = match.group(0)
            if ref not in valid:
                errors.append(
                    f"{target}:{lineno}: status label {ref!r} not declared "
                    f"in workflow.yaml § statuses. Add the row or remove "
                    f"the literal."
                )
    return errors


def check_script_references(
    *, roots: list[Path] | None = None, scripts_dir: Path | None = None
) -> list[str]:
    """Walk ``.claude/agents/**.md`` + ``.claude/skills/**/SKILL.md`` and
    FAIL on any ``scripts/<name>.py`` reference whose target does not exist
    under ``scripts/``.

    This guards the dead-tool / invented-tool failure class: a workflow
    step that runs ``scripts/foo.py`` where ``foo.py`` was deleted (or was
    documented but never created) is a latent ``CalledProcessError`` that
    only fires when an agent actually reaches that step. Catching the
    dangling reference at lint time is far cheaper than at run time.

    ``roots`` and ``scripts_dir`` are override hooks for unit tests:
    production callers pass both as None and the function walks the
    canonical agent + skill trees (via :func:`_resolve_ask_target_files`,
    which excludes OTHER worktrees but scans the current one — see
    :func:`_other_worktree_prefix`) and resolves references against
    ``<repo_root>/scripts``. Tests scope both to a fixture directory.
    """
    errors: list[str] = []
    scripts_root = scripts_dir if scripts_dir is not None else _REPO_ROOT / "scripts"
    for path in _resolve_ask_target_files(roots):
        for lineno, line in enumerate(path.read_text().splitlines(), start=1):
            for match in SCRIPT_REF_RE.finditer(line):
                script_name = match.group(1)
                if not (scripts_root / script_name).exists():
                    errors.append(
                        f"{path}:{lineno}: references 'scripts/{script_name}' "
                        f"which does not exist under {scripts_root}/. Repoint "
                        f"to the current helper, or remove the dead reference."
                    )
    return errors


def _iter_wandb_required_files(experiments_dir: Path) -> list[Path]:
    """Return every ``*.py`` under ``experiments_dir`` whose source
    mentions one of :data:`WANDB_TRAINER_CONFIG_TOKENS`. Skipping files
    that lack a trainer-config builder keeps the check focused on live-
    training launches and out of pure-eval / data-prep modules."""
    if not experiments_dir.exists():
        return []
    files: list[Path] = []
    for py in sorted(experiments_dir.rglob("*.py")):
        text = py.read_text(encoding="utf-8")
        if any(tok in text for tok in WANDB_TRAINER_CONFIG_TOKENS):
            files.append(py)
    return files


def _wandb_waiver_present(lines: list[str], idx: int) -> bool:
    """Return True iff a properly-shaped ``# WANDB_INTENTIONALLY_DISABLED:
    <reason>`` waiver covers the ``report_to=`` literal at line index
    ``idx``. Accepts:

    * Same-line trailing comment (``report_to="none",  # WANDB_INTENTIONALLY_DISABLED: ...``).
    * The immediately preceding non-blank line (covers the
      ``cfg = TrainLoraConfig(\\n    ...\\n    report_to="none",\\n)`` shape
      where the comment belongs above the call site, not jammed into the
      kwarg).

    The reason after the colon must be ≥ :data:`WANDB_WAIVER_MIN_REASON_CHARS`
    chars (force a real justification, not a token-shaped bypass).
    """
    # Same-line waiver.
    match = WANDB_WAIVER_RE.search(lines[idx])
    if match and len(match.group(1).strip()) >= WANDB_WAIVER_MIN_REASON_CHARS:
        return True
    # Previous non-blank line waiver. Skip blank lines only; any non-blank
    # non-waiver line above the kwarg breaks the chain (the implementer
    # would otherwise have put the comment further up, where it would no
    # longer obviously bind to this report_to= literal).
    back = idx - 1
    while back >= 0 and lines[back].strip() == "":
        back -= 1
    if back >= 0:
        match = WANDB_WAIVER_RE.search(lines[back])
        if match and len(match.group(1).strip()) >= WANDB_WAIVER_MIN_REASON_CHARS:
            return True
    return False


def check_wandb_required(
    *, experiments_dir: Path | None = None, repo_root: Path | None = None
) -> list[str]:
    """Scan training-config call sites under
    ``src/explore_persona_space/experiments/`` and FAIL on any
    ``report_to="none"`` (or equivalent disabling literal:
    ``report_to=None``, ``report_to=[]``) that is not waived by a
    ``# WANDB_INTENTIONALLY_DISABLED: <reason>`` comment on the same
    line or the immediately preceding non-blank line.

    Scope rationale: WandB live training metrics are mandatory per
    CLAUDE.md "Upload Policy" — loss curves, grad-norm history, and
    callback metrics cannot be reconstructed post-hoc. Task #496 trained
    12 cells with ``report_to="none"`` hardcoded into the per-cell
    ``TrainLoraConfig`` builder and the gap surfaced only at upload-
    verification (Step 8) when the project did not appear on WandB.
    Smoke, code-reviewer, and experimenter pre-launch all passed without
    flagging it.

    Only ``src/explore_persona_space/experiments/`` is in scope.
    Eval-only scripts under ``scripts/`` and integration tests
    legitimately disable WandB (no live training); flagging them would
    drown the lint in false positives. Files inside the scope that lack
    any of :data:`WANDB_TRAINER_CONFIG_TOKENS` are skipped — they're
    pure eval / data-prep / analyzer modules where the ``report_to``
    kwarg, if present, is a passthrough default rather than a hardcoded
    silencing.

    ``experiments_dir`` and ``repo_root`` are override hooks for unit
    tests; production callers pass both as None and the function walks
    the canonical ``<repo_root>/src/explore_persona_space/experiments``
    tree.
    """
    errors: list[str] = []
    root = repo_root if repo_root is not None else _REPO_ROOT
    target_dir = (
        experiments_dir
        if experiments_dir is not None
        else root / "src" / "explore_persona_space" / "experiments"
    )
    for path in _iter_wandb_required_files(target_dir):
        lines = path.read_text(encoding="utf-8").splitlines()
        for idx, line in enumerate(lines):
            if not WANDB_DISABLED_RE.search(line):
                continue
            if _wandb_waiver_present(lines, idx):
                continue
            errors.append(
                f"{path}:{idx + 1}: 'report_to' disables WandB inside a "
                f"training-config builder under "
                f"src/explore_persona_space/experiments/, but no "
                f"'# WANDB_INTENTIONALLY_DISABLED: <reason>' waiver "
                f"(reason ≥ {WANDB_WAIVER_MIN_REASON_CHARS} chars) is "
                f"present on the same or previous non-blank line. WandB "
                f"live training metrics are required by CLAUDE.md "
                f"'Upload Policy'; do not silence them without a "
                f"written justification. See task #496 post-mortem."
            )
    return errors


def render_marker_kinds_table(workflow: WorkflowYaml) -> str:
    """Render the auto-generated marker kinds table for ``markers.md``."""
    lines = [
        "| Kind | Posted by | When | Required fields |",
        "|------|-----------|------|-----------------|",
    ]
    for m in workflow.markers:
        # Escape pipes in the fields so the table doesn't fragment.
        fields = m.fields.replace("\n", " ").replace("|", r"\|").strip()
        lines.append(f"| `{m.kind}` | {m.posted_by} | {m.when} | {fields} |")
    return "\n".join(lines)


def render_active_vs_awaiting_table(workflow: WorkflowYaml) -> str:
    """Render the "Active vs awaiting-user" table for ``SKILL.md``."""
    lines = [
        "| State | Who's working | User action needed? |",
        "|-------|---------------|---------------------|",
    ]
    for s in workflow.statuses:
        # Skip the legacy alias to avoid confusion in the SKILL doc.
        if s.name == "under-review":
            continue
        action = "**yes**" if s.user_gated else "no"
        lines.append(f"| `{s.name}` | {s.description} | {action} |")
    return "\n".join(lines)


def _extract_fenced_block(text: str, marker_id: str) -> tuple[int, int] | None:
    """Return the (start, end) character offsets of the fenced
    auto-generated block named ``marker_id``, or None if not present."""
    open_marker = f"{AUTO_GEN_OPEN} ({marker_id}) -->"
    close_marker = AUTO_GEN_CLOSE
    start = text.find(open_marker)
    if start == -1:
        return None
    end_marker_at = text.find(close_marker, start)
    if end_marker_at == -1:
        return None
    end = end_marker_at + len(close_marker)
    return (start, end)


def _replace_fenced_block(text: str, marker_id: str, body: str) -> str | None:
    """Replace the fenced block named ``marker_id`` in ``text`` with
    ``body`` (newline-separated). Returns the new text, or None if the
    fence is not present."""
    span = _extract_fenced_block(text, marker_id)
    if span is None:
        return None
    start, end = span
    rendered = f"{AUTO_GEN_OPEN} ({marker_id}) -->\n{body}\n{AUTO_GEN_CLOSE}"
    return text[:start] + rendered + text[end:]


def emit_tables(workflow: WorkflowYaml, *, write: bool) -> list[str]:
    """Render all auto-generated tables. If ``write`` is True, update files
    in-place; otherwise compare and return drift errors."""
    errors: list[str] = []
    targets: list[tuple[Path, str, str]] = [
        (
            _REPO_ROOT / ".claude" / "skills" / "issue" / "markers.md",
            "marker-kinds",
            render_marker_kinds_table(workflow),
        ),
        (
            _REPO_ROOT / ".claude" / "skills" / "issue" / "SKILL.md",
            "active-vs-awaiting",
            render_active_vs_awaiting_table(workflow),
        ),
    ]
    for path, marker_id, body in targets:
        if not path.exists():
            errors.append(f"{path}: missing (cannot emit '{marker_id}' table)")
            continue
        original = path.read_text()
        replaced = _replace_fenced_block(original, marker_id, body)
        if replaced is None:
            errors.append(
                f"{path}: missing fenced block "
                f"'{AUTO_GEN_OPEN} ({marker_id}) -->'. Add a placeholder pair "
                f"of fence markers around the table location."
            )
            continue
        if write:
            if replaced != original:
                path.write_text(replaced)
        else:
            if replaced != original:
                errors.append(
                    f"{path}: auto-generated '{marker_id}' table is out of "
                    f"date. Run `uv run python scripts/workflow_lint.py "
                    f"--emit-tables` to regenerate."
                )
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--file",
        default=None,
        help="Path to the workflow.yaml file. Defaults to the canonical "
        ".claude/workflow.yaml under the repo root.",
    )
    parser.add_argument(
        "--check-references",
        action="store_true",
        help="Verify CLAUDE.md / SKILL.md / markers.md '(see workflow.yaml § X)' "
        "references resolve.",
    )
    parser.add_argument(
        "--check-tables",
        action="store_true",
        help="Verify auto-generated tables in SKILL.md / markers.md match the "
        "rendered output. (Default-on in --check-references mode.)",
    )
    parser.add_argument(
        "--emit-tables",
        action="store_true",
        help="Regenerate auto-generated tables in SKILL.md / markers.md in-place.",
    )
    parser.add_argument(
        "--check-status-labels",
        action="store_true",
        help="Verify every 'status:*' literal in scripts/gh_project.py "
        "resolves to a workflow.yaml status row.",
    )
    parser.add_argument(
        "--check-asks",
        action="store_true",
        help="Verify every 'AskUserQuestion' mention in .claude/agents/**.md "
        "and .claude/skills/**/SKILL.md is anchored to a documented gate "
        "(<!-- gate: <key> --> resolving to workflow.yaml § gates), to an "
        "existing '(see workflow.yaml § gates.X)' citation in the same "
        "paragraph, or marked as documentation via "
        "<!-- example: anti-pattern -->. Bundles --check-autonomous-asks "
        "(every AskUserQuestion in .claude/skills/issue/SKILL.md + "
        ".claude/agents/*.md MUST document its autonomous-mode behavior — "
        "see that flag's help). Enforces the CLAUDE.md auto-continuation "
        "contract.",
    )
    parser.add_argument(
        "--check-autonomous-asks",
        action="store_true",
        help="Verify every 'AskUserQuestion' mention in "
        ".claude/skills/issue/SKILL.md and .claude/agents/*.md has its "
        "surrounding paragraph documenting the autonomous-mode behavior "
        "(literal 'Interactive mode' / 'EPM_AUTONOMOUS_SESSION', or "
        "'<!-- autonomous-mode: <auto-resolve|skip|block-and-fail|"
        "gate-allowed> -->' annotation). Closes the #503/#504/#505 gap "
        "(2026-06-05): three autonomous sessions sat blocked because the "
        "SKILL.md prose did not enumerate autonomous-mode auto-resolve "
        "for conditional pivot gates. Bundled into --check-asks.",
    )
    parser.add_argument(
        "--check-script-refs",
        action="store_true",
        help="Verify every 'scripts/<name>.py' reference in .claude/agents/**.md "
        "and .claude/skills/**/SKILL.md resolves to a real file under scripts/. "
        "Bundled into --check-references and the no-flags default run.",
    )
    parser.add_argument(
        "--check-wandb-required",
        action="store_true",
        help="Verify no training script under src/explore_persona_space/"
        "experiments/ silences WandB via report_to='none' / None / [] "
        "without an explicit '# WANDB_INTENTIONALLY_DISABLED: <reason>' "
        "waiver. Closes the #496 gap where 12 cells trained without "
        "live training telemetry and the missing project surfaced only "
        "at upload-verification.",
    )
    args = parser.parse_args(argv)

    path = Path(args.file) if args.file else None
    try:
        workflow = load_workflow_yaml(path)
    except (ValueError, FileNotFoundError) as exc:
        sys.stderr.write(f"workflow_lint: schema FAIL\n{exc}\n")
        return 1
    except Exception as exc:
        sys.stderr.write(f"workflow_lint: schema FAIL\n{type(exc).__name__}: {exc}\n")
        return 1

    # A bare `workflow_lint.py` (no check/emit flags) validates the schema
    # AND runs the cheap, always-safe script-reference check so dangling
    # `scripts/<name>.py` references surface on the default invocation.
    no_flags = not (
        args.check_references
        or args.check_tables
        or args.emit_tables
        or args.check_status_labels
        or args.check_asks
        or args.check_autonomous_asks
        or args.check_script_refs
        or args.check_wandb_required
    )

    errors: list[str] = []
    if args.check_references:
        errors.extend(_check_references(workflow))
        # Also check tables on the references path; pre-commit invokes this
        # without --check-tables and we want both behaviours bundled.
        errors.extend(emit_tables(workflow, write=False))
        # Dangling script references are a workflow-doc integrity issue, same
        # class as unresolved (see workflow.yaml § X) references — bundle here.
        errors.extend(check_script_references())
    if args.check_tables and not args.check_references:
        errors.extend(emit_tables(workflow, write=False))
    if args.emit_tables:
        # Write mode: errors here are missing-fence problems, not drift.
        write_errors = emit_tables(workflow, write=True)
        errors.extend(write_errors)
    if args.check_status_labels:
        errors.extend(_check_status_label_coverage(workflow))
    if args.check_asks:
        errors.extend(check_asks(workflow))
        # The autonomous-asks check is bundled into --check-asks because the
        # two enforce complementary halves of the same contract: --check-asks
        # ensures every AskUserQuestion cites a gate; --check-autonomous-asks
        # ensures every AskUserQuestion documents its autonomous-mode handling.
        errors.extend(check_autonomous_asks())
    if args.check_autonomous_asks and not args.check_asks:
        errors.extend(check_autonomous_asks())
    if args.check_script_refs or no_flags:
        errors.extend(check_script_references())
    if args.check_wandb_required or no_flags:
        errors.extend(check_wandb_required())

    if errors:
        for err in errors:
            sys.stderr.write(f"workflow_lint: {err}\n")
        sys.stderr.write(f"workflow_lint: FAIL ({len(errors)} error(s))\n")
        return 1

    sys.stderr.write("workflow_lint: PASS\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
