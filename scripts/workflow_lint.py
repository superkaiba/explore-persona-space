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
* ``--check-heredoc-dotenv`` (also bundled into the no-flags default
  run): walk every ``*.sh`` under ``scripts/`` and FAIL on any bash
  heredoc that feeds a python interpreter's stdin (``uv run python -
  <<'PY'``, ``python3 <<EOF``, …) and whose body calls the python-dotenv
  package's no-arg ``load_dotenv()`` — from stdin its ``find_dotenv()``
  frame-walk always crashes (``assert frame.f_back is not None``).
  Explicit-path calls and the stdin-safe project wrapper
  (``explore_persona_space.orchestrate.env.load_dotenv``) pass. Closes
  the #552/#612 incident class: the gotcha existed only as prose
  (gotchas.md + research-project-structure.md § Environment Bootstrap)
  and was reintroduced on #612 past the implementer, BOTH ensemble
  reviewers, and every smoke run, because the heredoc executes only at
  pod-side first contact.
* ``--check-dispatcher-cvd-pin`` (also bundled into the no-flags default
  run): walk every ``*.sh`` under ``scripts/`` and FAIL on any
  BACKGROUNDED python launch line (logical line ending in ``&``) that
  passes a per-process GPU pin (``--gpu-id`` / ``+gpu_id=``) but does
  NOT carry a ``CUDA_VISIBLE_DEVICES=`` env prefix on the same command.
  The in-process CVD clobber (``train/sft.py`` sets
  ``os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.gpu_id)``) is silently
  defeated by any import-time cuInit, so parallel per-cell launches that
  rely on ``--gpu-id`` alone pile every cell onto physical GPU 0 and OOM
  (incident class #523 Phase B, recurred #541/#543/#557; recipe fix
  #578). Legitimate unpinned shapes are waived via
  ``# CVD_PIN_EXEMPT: <reason>`` on the same logical line or the
  immediately preceding non-blank line. Closes the residual #578 gap:
  the launcher-env-pin rule was agent-prose only (experimenter.md item
  10 fires on the RunPod launch path; gcp/slurm startup-script lanes
  have no launch agent), so a new dispatcher written without the pin
  reached production unflagged on those lanes.
* ``--check-marker-registry`` (also bundled into ``--check-references``):
  extract every marker kind that any skill's ``SKILL.md`` under
  ``.claude/skills/**/`` or an agent spec under ``.claude/agents/*.md``
  instructs POSTING
  (``task.py post-marker <N> epm:<kind>`` invocations plus post-verb
  prose with a backticked ``epm:<kind>`` on the same line) and FAIL on
  kinds absent from ``workflow.yaml § markers``.
  Read-side mentions don't match; prose-only false positives are waived
  via :data:`MARKER_REGISTRY_ALLOWLIST`. Closes the task-#555 drift
  class (2026-06-10): 6 posted-or-consumed kinds were missing from the
  registry and nothing cross-checked the two surfaces; the agent-spec
  half of the posting surface was added in the same task's follow-up,
  and the walk was widened from the issue SKILL.md to ALL skills'
  SKILL.md files on the chain's final fix (the promote-clean-result
  ``epm:consolidated-into`` posting site was unlinted until then).
* ``--check-agent-model-pins`` (also bundled into the no-flags default
  run): parse the YAML frontmatter ``model: "..."`` of every
  ``.claude/agents/*.md`` and FAIL on any pin whose base id is unknown
  OR whose ``[1m]`` suffix is grafted onto a base that does not have a
  1M-context variant (the d07424178 / task #545 incident class,
  2026-06-09→2026-06-12). The d07424178 commit bulk-renamed all 25
  agent pins to ``claude-fable-5[1m]`` — fable-5 IS a real Anthropic
  model id, BUT the ``[1m]`` suffix (a deployment-routing identifier
  per the claude-api skill's model-migration.md bucket-4 guidance) was
  not a valid variant for fable-5, so EVERY subagent died at spawn
  ("There's an issue with the selected model … may not exist") for ~72
  hours fleet-wide until the revert (00566584c). Sibling rule to
  CLAUDE.md / .claude/rules/code-style.md "Never hardcode an invented
  Claude/Anthropic model id" — that bullet covers hardcoded model
  strings in Python; this check covers agent-frontmatter pins, which
  the runtime hits on every subagent spawn. Allowlist drifts slowly
  (one entry per new Anthropic major-version release) and lives in
  :data:`AGENT_MODEL_ALLOWLIST`; the source of truth is the global
  ``claude-api`` skill's ``shared/models.md``.

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

# Inline opt-out for ``check_script_references``: a line carrying this
# HTML comment is a NARRATIVE incident citation (a branch-only or
# since-deleted script named for historical context), not an executable
# workflow step, so its `scripts/<name>.py` tokens are exempt from the
# dead-tool check. Scope is the single line bearing the comment —
# explicit, self-documenting, greppable. Do NOT attach it to a line an
# agent is expected to actually run. (Second hit of this class on task
# #545: an incident note in code-reviewer.md had to contort its prose to
# dodge SCRIPT_REF_RE.)
HISTORICAL_REF_OPT_OUT = "<!-- lint: historical-ref -->"

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

# `--check-marker-registry`: every marker kind the /issue SKILL.md or an
# agent spec under .claude/agents/*.md instructs POSTING must be declared in
# workflow.yaml § markers. Two pattern families
# count as a posting site (read-side mentions like "the latest `epm:foo v1`
# marker" deliberately do NOT match — only the posting contract is checked):
#
# 1. CLI invocations: `task.py post-marker <N> epm:<kind>` (any issue-arg
#    form: `<N>`, `"$N"`, a literal number, ...).
# 2. Posting prose: a post-verb (post/posts/posted/auto-post/re-post)
#    followed within the same line by a backticked `epm:<kind> ...` token
#    (optionally in the `<!-- epm:<kind> v1 -->` comment form).
#
# Closes the drift class where a skill step posts a kind the registry never
# declared, so the auto-generated markers.md table and the marker-taxonomy
# docs silently diverge from what actually lands in events.jsonl (task #555
# surfaced 6 unregistered kinds in one sweep, 2026-06-10). Prose-only /
# family-prefix mentions that a future edit accidentally phrases as a post
# can be waived via MARKER_REGISTRY_ALLOWLIST (document the reason inline).
MARKER_POST_CLI_RE = re.compile(r"\bpost-marker\s+\S+\s+(epm:[a-z][a-z0-9-]*)")
MARKER_POST_PROSE_RE = re.compile(
    r"\b(?:post|posts|posted|auto-post|auto-posts|re-post|re-posts)\b"
    r"[^`\n]{0,60}`(?:<!--\s*)?(epm:[a-z][a-z0-9-]*)",
    re.IGNORECASE,
)
# Kinds exempt from registration: prose-only or family-prefix mentions that
# match the posting patterns above without being a real posted kind
# (`epm:audit` — the SKILL.md placeholder guard — uses the verb "generating"
# so it never matches). Add entries here only with a comment naming the
# file:line and why it is not a posted kind.
MARKER_REGISTRY_ALLOWLIST: frozenset[str] = frozenset(
    {
        # campaign-tick/SKILL.md:104 "Newest skill-posted `epm:campaign-*`
        # marker FRESH" — a READ-side family-prefix mention, not a posting
        # site: `\bposted\b` matches inside the compound adjective
        # "skill-posted" (hyphen is a word boundary) and the kind regex
        # truncates `epm:campaign-*` at the `*`. The six real
        # `epm:campaign-*` kinds are individually registered in
        # workflow.yaml § markers; the tick itself never posts (its
        # contract: "No marker posts").
        "epm:campaign-",
    }
)

# `--check-heredoc-dotenv`: a NO-ARG `load_dotenv()` from the python-dotenv
# PACKAGE inside a bash heredoc that feeds a python interpreter's STDIN
# (`uv run python - <<'PY'`, `python3 <<EOF`, ...) always crashes at
# runtime: with no path argument, python-dotenv's `find_dotenv()` walks the
# interpreter frame stack looking for a caller whose `co_filename` exists
# on disk; from stdin the filename is `<stdin>`, the walk runs off the top
# of the stack, and `assert frame.f_back is not None` fires. The rule
# existed only as prose (gotchas.md; research-project-structure.md
# § Environment Bootstrap) and human review repeatedly missed it:
# incident #552, then again #612 (2026-06-12 —
# `issue612_production_driver.sh` stage-1b slipped past the implementer,
# BOTH ensemble reviewers, and every smoke run because the heredoc
# executes only at pod-side first contact, then killed the production
# driver with a misleading "poll timeout" and idled 4x A100 for ~30 min).
# This check makes the rule mechanical.
#
# Flagged (inside a python-stdin-fed heredoc body only):
#   * `from dotenv import load_dotenv` (any import list containing it)
#     plus a bare no-arg call `load_dotenv()`;
#   * a qualified no-arg call `dotenv.load_dotenv()`.
# NOT flagged:
#   * any-arg calls (`load_dotenv(dotenv_path=...)`) — an explicit path
#     skips the frame-walking `find_dotenv()` entirely;
#   * the project wrapper
#     `explore_persona_space.orchestrate.env.load_dotenv()` — resolves
#     `.env` via `resolve_dotenv_path()` (cwd/path walking, no frame
#     inspection), stdin-safe; this is the canonical in-heredoc shape
#     (#585 round-2 review fix; live exemplar `i556_run_all_1gpu.sh`);
#   * heredocs that do NOT feed a python interpreter's stdin
#     (`cat <<EOF`, `python scripts/foo.py <<EOF` where the body is
#     DATA for the script, ...);
#   * comment lines inside the heredoc body;
#   * `python -c '...'` one-liner arguments — DELIBERATELY out of scope
#     (extension considered + rejected, 2026-06-12): under `-c`,
#     `__main__` has no `__file__`, so python-dotenv's `_is_interactive()`
#     short-circuits find_dotenv() to a cwd-walk — the frame walk (and
#     its `assert frame.f_back is not None` crash) is never reached
#     (verified empirically against the pinned python-dotenv 1.2.2). A
#     no-arg call run from the repo root legitimately finds `.env`, so a
#     hard FAIL (this framework has no warn tier / waiver) would flag
#     working shapes. The real `-c` hazard is SILENT non-loading from an
#     off-repo cwd — prose-documented in gotchas.md's python-dotenv
#     entry, not lintable without false positives.
#
# Opener parsing: backslash-continued physical lines are merged into one
# logical command line first (the #612 incident shape is
# `uv run python - "$A" "$B" <<'PY' \` continued by `|| fail ... 3`, with
# the body starting after the continuation). The opener regex excludes
# here-strings (`<<<`) and requires an identifier-shaped delimiter so
# arithmetic shifts (`$((x << 2))`) don't parse as heredocs. A python
# interpreter is considered stdin-fed when, before the opener on the
# logical line, `python`/`python3[.N]` is followed by a bare `-` arg
# (optionally after single-dash flags) OR is the last token.
HEREDOC_OPENER_RE = re.compile(r"(?<!<)<<-?(?!<)\s*(['\"]?)([A-Za-z_]\w*)\1")
HEREDOC_PY_STDIN_DASH_RE = re.compile(r"\bpython3?(?:\.\d+)?\s+(?:-\S+\s+)*-(?=[\s\"']|$)")
HEREDOC_PY_STDIN_BARE_RE = re.compile(r"\bpython3?(?:\.\d+)?[\"']?\s*$")
HEREDOC_DOTENV_PKG_IMPORT_RE = re.compile(
    r"^\s*from\s+dotenv(?:\.[\w.]+)?\s+import\s+(?P<names>.+)$"
)
HEREDOC_DOTENV_BARE_CALL_RE = re.compile(r"(?<![\w.])load_dotenv\s*\(\s*\)")
HEREDOC_DOTENV_QUALIFIED_CALL_RE = re.compile(r"(?<![\w.])dotenv\.load_dotenv\s*\(\s*\)")

# `--check-dispatcher-cvd-pin`: a BACKGROUNDED python launch in a shell
# script that passes a per-process GPU pin (`--gpu-id <n>` / `+gpu_id=<n>`)
# MUST also carry a `CUDA_VISIBLE_DEVICES=` env assignment on the same
# logical command line. The in-process clobber
# (`os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.gpu_id)` in
# `train/sft.py`) is silently defeated by any import-time cuInit — the
# driver freezes its device list at the FIRST cuInit in the process, so a
# dispatcher import chain that initializes CUDA (`import peft` is a known
# offender, #545) makes the late clobber a no-op and every parallel cell's
# `cuda:0` resolves to physical GPU 0 → co-location → OOM. That is how all
# 4 #523 Phase B waves piled onto GPU 0 (recurred #541/#543/#557). The
# recipe fix (#578, gotchas.md "CVD-clobber" entry): export
# `CUDA_VISIBLE_DEVICES=<gpu>` per cell in the LAUNCHER env AND pass the
# matching `--gpu-id <gpu>` so the in-process clobber rewrites the same
# value. The reference compliant shape is
# `scripts/i474_phase23_dispatch.sh` ("CUDA_VISIBLE_DEVICES="$cvd" uv run
# python ... --gpu-id "$cvd" ... &").
#
# Flagged: a logical line (backslash continuations merged) that
#   (a) invokes a python interpreter (`uv run python`, bare
#       `python`/`python3[.N]`, `.venv/bin/python`), AND
#   (b) carries `--gpu-id` or `+gpu_id=`, AND
#   (c) is backgrounded — ends with `&` (not `&&`), the parallel-launch
#       signature, AND
#   (d) has NO `CUDA_VISIBLE_DEVICES=` assignment anywhere on the line.
# NOT flagged (recall is deliberately sacrificed for zero false
# positives — a sequential launch cannot co-locate siblings):
#   * sequential launches (no trailing `&`), including `nohup ... ;`
#     and `cmd && next` chains;
#   * `echo`-prefixed lines (dry-run previews) and `#` comment lines;
#   * backgrounded SUBSHELL wrappers (`( for ...; do python ...; done ) &`)
#     whose python line itself is not backgrounded — a known recall miss
#     (live example: `i488_phase4_dispatch.sh`), accepted to keep the
#     check line-local and false-positive-free;
#   * lines waived via `# CVD_PIN_EXEMPT: <reason>` (same logical line or
#     immediately preceding non-blank line; reason ≥ 10 chars — same
#     convention as WANDB_INTENTIONALLY_DISABLED). Use the waiver for
#     pre-#578 completed-task dispatchers kept verbatim for
#     reproducibility, and for genuinely single-process backgrounded
#     launches where no sibling can co-locate.
CVD_PIN_PY_LAUNCH_RE = re.compile(
    r"(?:\buv\s+run\s+python\b|(?<![\w./])python3?(?:\.\d+)?\b|\.venv/bin/python\b)"
)
CVD_PIN_GPU_ARG_RE = re.compile(r"(?:--gpu-id\b|\+gpu_id=)")
CVD_PIN_CVD_ASSIGN_RE = re.compile(r"\bCUDA_VISIBLE_DEVICES=")
CVD_PIN_WAIVER_RE = re.compile(r"#\s*CVD_PIN_EXEMPT\s*:\s*(.+?)\s*$")
CVD_PIN_WAIVER_MIN_REASON_CHARS = 10

# `--check-agent-model-pins`: every `.claude/agents/*.md` carries a YAML
# frontmatter line ``model: "claude-..."`` that the Claude Code harness reads
# at subagent spawn. A pin that is unknown OR carries a `[1m]` suffix on a
# base id without a 1M-context variant fails AT SPAWN ("There's an issue
# with the selected model … may not exist") and kills EVERY subagent in
# EVERY session fleet-wide until reverted — the d07424178 / task #545
# incident class (2026-06-09 → 2026-06-12, ~72h fleet-wide outage from a
# single commit pinning all 25 agents to `claude-fable-5[1m]`, where
# fable-5 is real but its `[1m]` variant is not).
#
# Allowlist source of truth: the global ``claude-api`` skill's
# ``shared/models.md`` ("Model Descriptions" section). Each entry carries
# the base id + whether a `[1m]` (1M-context routing) variant exists for
# it — opus-4-5/4-6/4-7/4-8 expose a `[1m]` tier; fable-5/mythos-5/
# sonnet-4-6 already have 1M native context (no `[1m]` suffix supported);
# haiku-4-5/sonnet-4-5 are 200K-context tiers (no `[1m]` suffix).
# Deprecated / retired base ids are not listed — pinning to a deprecated
# id is also flagged as "unknown". Update this list when Anthropic ships
# a new major version (a low-frequency event; weigh against the cost of
# letting a typo'd or aspirational pin take down every subagent silently).
#
# Each tuple is (base_id, supports_1m_suffix).
AGENT_MODEL_ALLOWLIST: tuple[tuple[str, bool], ...] = (
    # Opus tier — 1M-context [1m] variant exposed for each.
    ("claude-opus-4-5", True),
    ("claude-opus-4-6", True),
    ("claude-opus-4-7", True),
    ("claude-opus-4-8", True),
    # Fable / Mythos — 1M native context, no [1m] suffix. Mythos-5 is a real,
    # active id but is Project-Glasswing-only; most callers should pin fable-5
    # (a non-Glasswing org pinning mythos-5 would still fail at spawn — the
    # harness check catches that regardless of this allowlist).
    ("claude-fable-5", False),
    ("claude-mythos-5", False),
    # Sonnet — 4-6 has 1M native context (no suffix); 4-5 is 200K.
    ("claude-sonnet-4-5", False),
    ("claude-sonnet-4-6", False),
    # Haiku — 200K context tier, no [1m] suffix.
    ("claude-haiku-4-5", False),
)
# Parse the YAML frontmatter ``model: "..."`` line. Permissive on quoting
# (double, single, or bare) — the harness accepts all three. The captured
# group is the full id string including any [1m] suffix.
AGENT_MODEL_PIN_RE = re.compile(
    r"""^model:\s*["']?(?P<value>[A-Za-z0-9_.\-\[\]]+)["']?\s*$""", re.MULTILINE
)
# Split the captured id into (base, suffix). Suffix is the literal `[1m]`
# (the only suffix the harness currently exposes for a model pin); any
# other tail is treated as part of an unknown base id and flagged.
AGENT_MODEL_1M_SUFFIX = "[1m]"


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

    Lines carrying the :data:`HISTORICAL_REF_OPT_OUT` comment
    (``<!-- lint: historical-ref -->``) are skipped entirely: they mark
    narrative incident citations that name branch-only or since-deleted
    scripts for historical context, not executable steps. The opt-out is
    per-line and explicit — a dead reference anywhere else still FAILs.

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
            if HISTORICAL_REF_OPT_OUT in line:
                continue
            for match in SCRIPT_REF_RE.finditer(line):
                script_name = match.group(1)
                if not (scripts_root / script_name).exists():
                    errors.append(
                        f"{path}:{lineno}: references 'scripts/{script_name}' "
                        f"which does not exist under {scripts_root}/. Repoint "
                        f"to the current helper, remove the dead reference, "
                        f"or — for a narrative incident citation only — "
                        f"append '{HISTORICAL_REF_OPT_OUT}' to the line."
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


def _heredoc_body_dotenv_errors(path: Path, lines: list[str], start: int, end: int) -> list[str]:
    """Scan one python-stdin-fed heredoc body (``lines[start:end]``,
    0-based, terminator excluded) and return an error per dangerous
    no-arg python-dotenv ``load_dotenv()`` call. Comment lines are
    skipped; the bare-name call is only dangerous when the SAME body
    imports ``load_dotenv`` from the ``dotenv`` package (a heredoc is a
    self-contained program, so the import must be visible — this is what
    keeps the stdin-safe project-wrapper import a PASS)."""
    code = [
        (idx, ln)
        for idx, ln in enumerate(lines[start:end], start=start)
        if not ln.lstrip().startswith("#")
    ]
    imports_pkg_load_dotenv = False
    for _, ln in code:
        match = HEREDOC_DOTENV_PKG_IMPORT_RE.match(ln)
        if match and re.search(r"\bload_dotenv\b", match.group("names")):
            imports_pkg_load_dotenv = True
            break
    errors: list[str] = []
    for idx, ln in code:
        dangerous = bool(HEREDOC_DOTENV_QUALIFIED_CALL_RE.search(ln)) or (
            imports_pkg_load_dotenv and bool(HEREDOC_DOTENV_BARE_CALL_RE.search(ln))
        )
        if dangerous:
            errors.append(
                f"{path}:{idx + 1}: no-arg python-dotenv `load_dotenv()` inside a "
                f"heredoc feeding a python interpreter's stdin — find_dotenv()'s "
                f"frame-walk crashes from stdin (assert frame.f_back is not None; "
                f"incidents #552, #612). Drop the dotenv call and rely on env vars "
                f"exported by the enclosing shell (`set -a && source .env && set +a` "
                f"before the heredoc), pass an explicit path "
                f"(load_dotenv(dotenv_path=...)), or use the stdin-safe project "
                f"wrapper `explore_persona_space.orchestrate.env.load_dotenv()`. See "
                f".claude/rules/research-project-structure.md § Environment Bootstrap."
            )
    return errors


def _scan_shell_file_for_heredoc_dotenv(path: Path) -> list[str]:
    """Walk one shell script, tracking heredoc bodies, and return the
    dotenv errors found in bodies that feed a python interpreter's stdin.

    Backslash-continued physical lines are merged into one logical
    command line before opener detection (the #612 shape continues the
    opener line with ``\\`` + ``|| fail ...``; the body starts after the
    last physical line of the logical command). ALL heredoc bodies are
    consumed so body content can never be misparsed as new openers; only
    python-stdin-fed bodies are scanned. The terminator match is lenient
    (stripped-line equality) so ``<<-`` indented terminators work; an
    unterminated heredoc scans through to EOF."""
    lines = path.read_text(encoding="utf-8").splitlines()
    errors: list[str] = []
    n = len(lines)
    i = 0
    while i < n:
        last = i
        logical = lines[i]
        while logical.rstrip().endswith("\\") and last + 1 < n:
            last += 1
            logical = logical.rstrip()[:-1] + " " + lines[last]
        openers = list(HEREDOC_OPENER_RE.finditer(logical))
        if not openers:
            i = last + 1
            continue
        prefix = logical[: openers[0].start()]
        python_fed = bool(HEREDOC_PY_STDIN_DASH_RE.search(prefix)) or bool(
            HEREDOC_PY_STDIN_BARE_RE.search(prefix)
        )
        body_cursor = last + 1
        for opener in openers:
            delim = opener.group(2)
            body_start = body_cursor
            body_end = body_start
            while body_end < n and lines[body_end].strip() != delim:
                body_end += 1
            if python_fed:
                errors.extend(_heredoc_body_dotenv_errors(path, lines, body_start, body_end))
            body_cursor = body_end + 1
        i = body_cursor
    return errors


def check_heredoc_dotenv(*, scripts_dir: Path | None = None) -> list[str]:
    """Walk every ``*.sh`` under ``scripts/`` and FAIL on any bash heredoc
    that feeds a python interpreter's stdin and whose body calls the
    python-dotenv package's no-arg ``load_dotenv()``.

    Rationale: from a stdin heredoc, python-dotenv's no-arg
    ``find_dotenv()`` frame-walk ALWAYS crashes (``assert frame.f_back is
    not None``) — there is no legitimate use, so no waiver/opt-out exists.
    The rule lived only in prose (gotchas.md;
    research-project-structure.md § Environment Bootstrap) and was
    reintroduced on #612 (after #552) past the implementer, both ensemble
    reviewers, and all smoke runs: the heredoc executes only at pod-side
    first contact, so nothing mechanical caught it before this check.
    Safe shapes (explicit-path calls; the stdin-safe project wrapper
    ``explore_persona_space.orchestrate.env.load_dotenv``; heredocs that
    are data, not python stdin) pass — see the regex block above for the
    full flagged/not-flagged matrix.

    ``scripts_dir`` is an override hook for unit tests; production
    callers pass None and the function walks the canonical
    ``<repo_root>/scripts`` tree. Bundled into the no-flags default run
    (same policy as ``check_script_references`` / ``check_wandb_required``).
    """
    root = scripts_dir if scripts_dir is not None else _REPO_ROOT / "scripts"
    if not root.exists():
        return []
    errors: list[str] = []
    for sh in sorted(root.rglob("*.sh")):
        if not sh.is_file():
            continue
        errors.extend(_scan_shell_file_for_heredoc_dotenv(sh))
    return errors


def _iter_logical_shell_lines(lines: list[str]):
    """Yield ``(first_idx, last_idx, logical)`` per logical shell command
    line, merging backslash-continued physical lines (same merge rule as
    the heredoc scanner). Indices are 0-based physical-line bounds of the
    logical line, inclusive."""
    n = len(lines)
    i = 0
    while i < n:
        last = i
        logical = lines[i]
        while logical.rstrip().endswith("\\") and last + 1 < n:
            last += 1
            logical = logical.rstrip()[:-1] + " " + lines[last]
        yield i, last, logical
        i = last + 1


def _cvd_pin_waiver_present(lines: list[str], first_idx: int, last_idx: int) -> bool:
    """Return True iff a ``# CVD_PIN_EXEMPT: <reason>`` waiver (reason ≥
    :data:`CVD_PIN_WAIVER_MIN_REASON_CHARS` chars) covers the logical
    command spanning ``lines[first_idx:last_idx + 1]``. Accepts the waiver
    on any physical line of the logical command (trailing comment on a
    single-line launch) or on the immediately preceding non-blank line
    (the only valid placement for a backslash-continued launch — a
    trailing ``#`` comment would break the continuation)."""
    for idx in range(first_idx, last_idx + 1):
        match = CVD_PIN_WAIVER_RE.search(lines[idx])
        if match and len(match.group(1).strip()) >= CVD_PIN_WAIVER_MIN_REASON_CHARS:
            return True
    back = first_idx - 1
    while back >= 0 and lines[back].strip() == "":
        back -= 1
    if back >= 0:
        match = CVD_PIN_WAIVER_RE.search(lines[back])
        if match and len(match.group(1).strip()) >= CVD_PIN_WAIVER_MIN_REASON_CHARS:
            return True
    return False


def check_dispatcher_cvd_pin(*, scripts_dir: Path | None = None) -> list[str]:
    """Walk every ``*.sh`` under ``scripts/`` and FAIL on any backgrounded
    python launch line that passes a per-process GPU pin (``--gpu-id`` /
    ``+gpu_id=``) without a ``CUDA_VISIBLE_DEVICES=`` env assignment on
    the same logical command line.

    Rationale: the in-process CVD clobber in ``train/sft.py`` is silently
    defeated by any import-time cuInit, so parallel per-cell launches
    relying on ``--gpu-id`` alone co-locate every cell on physical GPU 0
    and OOM (#523 Phase B; recurred #541/#543/#557). The #578 recipe —
    pin ``CUDA_VISIBLE_DEVICES=<gpu>`` in the LAUNCHER env AND pass the
    matching ``--gpu-id`` — shipped as agent prose only (experimenter.md
    fires on the RunPod launch path; the gcp/slurm startup-script lanes
    have no launch agent), so this check is the lane-independent
    mechanical enforcement. Detection matrix + waiver convention: see the
    ``CVD_PIN_*`` regex block above.

    ``scripts_dir`` is an override hook for unit tests; production
    callers pass None and the function walks the canonical
    ``<repo_root>/scripts`` tree. Bundled into the no-flags default run
    (same policy as ``check_heredoc_dotenv`` / ``check_wandb_required``).
    """
    root = scripts_dir if scripts_dir is not None else _REPO_ROOT / "scripts"
    if not root.exists():
        return []
    errors: list[str] = []
    for sh in sorted(root.rglob("*.sh")):
        if not sh.is_file():
            continue
        lines = sh.read_text(encoding="utf-8").splitlines()
        for first, last, logical in _iter_logical_shell_lines(lines):
            stripped = logical.strip()
            # Comments and dry-run echo previews are not launches.
            if stripped.startswith("#") or stripped.startswith("echo "):
                continue
            # Backgrounded = parallel-launch signature. A trailing `&&` is
            # a command chain continuation, not a background token.
            if not (stripped.endswith("&") and not stripped.endswith("&&")):
                continue
            if not CVD_PIN_PY_LAUNCH_RE.search(logical):
                continue
            if not CVD_PIN_GPU_ARG_RE.search(logical):
                continue
            if CVD_PIN_CVD_ASSIGN_RE.search(logical):
                continue
            if _cvd_pin_waiver_present(lines, first, last):
                continue
            errors.append(
                f"{sh}:{first + 1}: backgrounded python launch passes "
                f"--gpu-id/+gpu_id= without a CUDA_VISIBLE_DEVICES= env "
                f"prefix on the same command. The in-process CVD clobber "
                f"is defeated by import-time cuInit, so parallel cells "
                f"co-locate on GPU 0 and OOM (#523/#541/#543/#557). Pin "
                f"CUDA_VISIBLE_DEVICES=<gpu> in the launcher env AND pass "
                f"the matching --gpu-id (reference shape: "
                f"scripts/i474_phase23_dispatch.sh), or waive a "
                f"legitimately unpinned launch with "
                f"'# CVD_PIN_EXEMPT: <reason>' (reason ≥ "
                f"{CVD_PIN_WAIVER_MIN_REASON_CHARS} chars) on the same or "
                f"previous non-blank line. See .claude/rules/gotchas.md "
                f"'CVD-clobber'."
            )
    return errors


def check_marker_registry(
    workflow: WorkflowYaml,
    *,
    skill_md: Path | None = None,
    skills_dir: Path | None = None,
    agents_dir: Path | None = None,
) -> list[str]:
    """Cross-reference posted ``epm:<kind>`` markers in EVERY skill's
    SKILL.md under ``.claude/skills/**/`` AND every agent spec under
    ``.claude/agents/*.md`` against ``workflow.yaml § markers`` and FAIL
    on any posting site whose kind is undeclared.

    A "posting site" is a line matching either :data:`MARKER_POST_CLI_RE`
    (a ``task.py post-marker <N> epm:<kind>`` invocation) or
    :data:`MARKER_POST_PROSE_RE` (a post-verb followed by a backticked
    ``epm:<kind>`` token on the same line). Read-side mentions ("the latest
    ``epm:foo v1`` marker", "an ``epm:bar`` event exists") deliberately do
    NOT match — the check pins the posting contract, not every reference.

    Kinds in :data:`MARKER_REGISTRY_ALLOWLIST` are waived (prose-only /
    family-prefix mentions that happen to match the patterns).

    Rationale: task #555's sweep (2026-06-10) found 6 marker kinds the
    SKILL.md instructed posting (or read back) that were absent from the
    registry — the auto-generated ``markers.md`` table and the marker
    taxonomy had silently drifted from what lands in ``events.jsonl``.
    Nothing linted the two surfaces against each other; this check does.
    Agent specs were added to the scope on the same task's follow-up:
    agents post kinds too (e.g. ``analyzer.md`` posts ``epm:analysis``),
    and a SKILL.md-only walk left half the posting surface unlinted.
    Non-issue skills were added on the chain's final fix (same task,
    2026-06-10): ``promote-clean-result/SKILL.md`` carried a real
    ``epm:consolidated-into`` posting site that an issue-SKILL.md-only
    walk never saw. Both production globs are rooted directly under
    ``_REPO_ROOT`` (``.claude/skills`` recursive, ``.claude/agents``
    flat), and sibling worktrees live under ``.claude/worktrees/`` —
    outside both roots — so they are inherently out of scope and the
    worktree a workflow-improver runs from scans its own copies (same
    property ``_other_worktree_prefix`` documents for the recursive
    walks).

    ``skill_md``, ``skills_dir``, and ``agents_dir`` are override hooks
    for unit tests; production callers pass all three as None and the
    function reads the canonical ``.claude/skills/**/SKILL.md`` +
    ``.claude/agents/*.md`` under :data:`_REPO_ROOT`. Passing ANY
    override narrows the scan to only the overridden surface(s) so
    fixture tests stay isolated from the committed tree.
    """
    targets: list[Path] = []
    if skill_md is None and skills_dir is None and agents_dir is None:
        canonical_skills = _REPO_ROOT / ".claude" / "skills"
        if canonical_skills.is_dir():
            targets.extend(sorted(p for p in canonical_skills.glob("**/SKILL.md") if p.is_file()))
        canonical_agents = _REPO_ROOT / ".claude" / "agents"
        if canonical_agents.is_dir():
            targets.extend(sorted(p for p in canonical_agents.glob("*.md") if p.is_file()))
    else:
        if skill_md is not None:
            targets.append(skill_md)
        if skills_dir is not None and skills_dir.is_dir():
            targets.extend(sorted(p for p in skills_dir.glob("**/SKILL.md") if p.is_file()))
        if agents_dir is not None and agents_dir.is_dir():
            targets.extend(sorted(p for p in agents_dir.glob("*.md") if p.is_file()))
    registered = {m.kind for m in workflow.markers}
    errors: list[str] = []
    for target in targets:
        if not target.exists():
            continue
        for lineno, line in enumerate(target.read_text().splitlines(), start=1):
            kinds = set(MARKER_POST_CLI_RE.findall(line))
            kinds.update(MARKER_POST_PROSE_RE.findall(line))
            for kind in sorted(kinds):
                if kind in registered or kind in MARKER_REGISTRY_ALLOWLIST:
                    continue
                errors.append(
                    f"{target}:{lineno}: posts marker kind '{kind}' which is not "
                    f"declared in workflow.yaml § markers. Register the kind "
                    f"(then regenerate markers.md via `uv run python "
                    f"scripts/workflow_lint.py --emit-tables`), or — for a "
                    f"prose-only mention that is not a real posted kind — add it "
                    f"to MARKER_REGISTRY_ALLOWLIST with a reason."
                )
    return errors


def _split_agent_model_pin(pin: str) -> tuple[str, str]:
    """Split a frontmatter model-pin string into ``(base_id, suffix)``.

    Recognized suffix: the literal :data:`AGENT_MODEL_1M_SUFFIX` (``"[1m]"``),
    the only routing-suffix the harness exposes on a model pin today. Any
    other tail stays glued to the base — that's the desired behavior, so
    that a typo like ``claude-opus-4-7[2m]`` is reported as an unknown
    base rather than masked as a known base with an unrecognized suffix.

    Examples::

        "claude-opus-4-7[1m]"   -> ("claude-opus-4-7", "[1m]")
        "claude-fable-5"        -> ("claude-fable-5", "")
        "claude-fable-5[1m]"    -> ("claude-fable-5", "[1m]")
        "claude-foo-bar"        -> ("claude-foo-bar", "")
    """
    if pin.endswith(AGENT_MODEL_1M_SUFFIX):
        return pin[: -len(AGENT_MODEL_1M_SUFFIX)], AGENT_MODEL_1M_SUFFIX
    return pin, ""


def _iter_agent_pin_target_files(repo_root: Path) -> list[Path]:
    """Return every ``.claude/agents/*.md`` under ``repo_root`` whose
    path is NOT in a sibling worktree (same exclusion rule as
    :func:`_iter_ask_target_files`)."""
    agents_root = repo_root / ".claude" / "agents"
    if not agents_root.exists():
        return []
    current_prefix = _other_worktree_prefix(repo_root)
    return sorted(
        p
        for p in agents_root.glob("*.md")
        if p.is_file() and not _is_other_worktree_path(p, current_prefix)
    )


def check_agent_model_pins(*, roots: list[Path] | None = None) -> list[str]:
    """Walk ``.claude/agents/*.md`` and FAIL on any ``model: "..."``
    frontmatter pin whose base id is unknown OR whose ``[1m]`` suffix is
    not supported on that base.

    The harness rejects any unknown pin at subagent spawn with
    ``"There's an issue with the selected model (<id>). It may not exist
    or you may not have access to it."`` — and because EVERY agent file
    carries a pin, a single bad-pin commit kills every subagent in every
    session fleet-wide until reverted. The d07424178 incident
    (2026-06-09) bulk-renamed all 25 agents to ``claude-fable-5[1m]``:
    fable-5 is a real Anthropic id, but its ``[1m]`` routing variant is
    not exposed (fable-5 has 1M native context, no separate [1m] tier).
    Every spawn failed for ~72h fleet-wide until the revert (00566584c).

    Rule, per pin (the file's ``model:`` frontmatter line):

    1. Split into ``(base_id, suffix)`` via
       :func:`_split_agent_model_pin`.
    2. If ``base_id`` is not in :data:`AGENT_MODEL_ALLOWLIST` → FAIL
       (typo, aspirational id, or a deprecated id no longer recognized
       by the harness).
    3. If ``suffix == "[1m]"`` and the base's allowlist tuple has
       ``supports_1m_suffix = False`` → FAIL (the exact d07424178
       pattern: a real base, an invalid routing suffix).
    4. Otherwise PASS.

    Files with no ``model:`` line are silently skipped — agents may
    legitimately inherit their model from the parent (no pin = no
    runtime contract to validate). A file with multiple ``model:`` lines
    in its frontmatter is unusual; only the FIRST is checked (the
    harness reads first-match too).

    Sibling rule to ``.claude/rules/code-style.md`` "Never hardcode an
    invented Claude/Anthropic model id" — that bullet covers hardcoded
    model strings in Python code; this check covers agent-frontmatter
    pins. The :data:`AGENT_MODEL_ALLOWLIST` source of truth is the
    global ``claude-api`` skill's ``shared/models.md`` "Model
    Descriptions" + "Bucket 4" suffix-variant guidance in
    ``shared/model-migration.md``.

    ``roots`` is an override hook for unit tests; production callers
    pass None and the function walks the canonical agent tree under
    :data:`_REPO_ROOT` (excluding sibling worktrees).
    """
    base_to_1m_capability: dict[str, bool] = {b: ok for (b, ok) in AGENT_MODEL_ALLOWLIST}
    if roots is None:
        targets = _iter_agent_pin_target_files(_REPO_ROOT)
    else:
        targets = []
        for root in roots:
            if root.is_file():
                targets.append(root)
            else:
                targets.extend(p for p in root.glob("**/*.md") if p.is_file())
        targets = sorted(targets)
    errors: list[str] = []
    for path in targets:
        text = path.read_text()
        match = AGENT_MODEL_PIN_RE.search(text)
        if match is None:
            # No pin = inherits parent's model = no runtime contract to
            # validate. Silently skipped (a missing pin is not a bug;
            # CLAUDE.md "Prompt-cache key discipline" explicitly allows it).
            continue
        # Compute the 1-based line number of the captured value so the
        # error message points to the actual ``model:`` line, not just
        # the file.
        lineno = text.count("\n", 0, match.start()) + 1
        pin = match.group("value")
        base_id, suffix = _split_agent_model_pin(pin)
        if base_id not in base_to_1m_capability:
            known = ", ".join(sorted(base_to_1m_capability))
            errors.append(
                f"{path}:{lineno}: frontmatter pins 'model: \"{pin}\"' whose "
                f"base id '{base_id}' is not in the allowlist. The harness "
                f"rejects unknown pins at subagent spawn ('may not exist or "
                f"you may not have access to it') and EVERY subagent dies "
                f"fleet-wide until reverted (d07424178 incident, task #545). "
                f"Allowed bases: {known}. If a new Anthropic model just "
                f"shipped, update AGENT_MODEL_ALLOWLIST in "
                f"scripts/workflow_lint.py — source of truth is the global "
                f"claude-api skill's shared/models.md."
            )
            continue
        if suffix == AGENT_MODEL_1M_SUFFIX and not base_to_1m_capability[base_id]:
            errors.append(
                f"{path}:{lineno}: frontmatter pins 'model: \"{pin}\"' but "
                f"base '{base_id}' does not expose a '[1m]' 1M-context "
                f"routing variant (it either has 1M native context with no "
                f"suffix, or is a 200K-context tier). The harness rejects "
                f"the suffixed id at subagent spawn and EVERY subagent dies "
                f"fleet-wide until reverted (d07424178 incident, task #545: "
                f"all 25 agents pinned to 'claude-fable-5[1m]' → ~72h "
                f"outage). Pin '{base_id}' alone, or switch to a base whose "
                f"AGENT_MODEL_ALLOWLIST tuple has supports_1m_suffix=True."
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


def main(argv: list[str] | None = None) -> int:  # noqa: C901 -- flat flag-dispatch ladder; one branch per check flag, extracting it would just relocate the ladder
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
    parser.add_argument(
        "--check-heredoc-dotenv",
        action="store_true",
        help="Verify no shell script under scripts/ feeds a python "
        "interpreter's stdin a heredoc whose body calls the python-dotenv "
        "package's no-arg load_dotenv() (its find_dotenv() frame-walk "
        "always crashes from stdin: assert frame.f_back is not None). "
        "Explicit-path calls and the stdin-safe project wrapper "
        "explore_persona_space.orchestrate.env.load_dotenv pass. Closes "
        "the #552/#612 incident class. Bundled into the no-flags default "
        "run.",
    )
    parser.add_argument(
        "--check-dispatcher-cvd-pin",
        action="store_true",
        help="Verify no shell script under scripts/ backgrounds a python "
        "launch that passes --gpu-id/+gpu_id= without a "
        "CUDA_VISIBLE_DEVICES= env prefix on the same logical command "
        "(the in-process CVD clobber is defeated by import-time cuInit, "
        "so unpinned parallel cells co-locate on GPU 0 and OOM — "
        "incident class #523/#541/#543/#557, recipe fix #578). Waive "
        "legitimate shapes with '# CVD_PIN_EXEMPT: <reason>'. Bundled "
        "into the no-flags default run.",
    )
    parser.add_argument(
        "--check-marker-registry",
        action="store_true",
        help="Verify every marker kind that .claude/skills/issue/SKILL.md "
        "or an agent spec under .claude/agents/*.md instructs posting "
        "(task.py post-marker invocations + post-verb prose with a "
        "backticked epm:<kind>) is declared in workflow.yaml § markers. "
        "Closes the #555 drift class (6 unregistered posted kinds, "
        "2026-06-10; agent-spec scope added in the follow-up). Bundled "
        "into --check-references.",
    )
    parser.add_argument(
        "--check-agent-model-pins",
        action="store_true",
        help="Verify every .claude/agents/*.md frontmatter 'model: \"...\"' "
        "pin has a known base id AND a valid suffix (only '[1m]' allowed, "
        "only on opus-4-5/4-6/4-7/4-8). Closes the d07424178 / task #545 "
        "incident class (2026-06-09 -> 2026-06-12): all 25 agents bulk-"
        "pinned to 'claude-fable-5[1m]' killed every subagent fleet-wide "
        "for ~72h until reverted. Sibling to the code-style 'never "
        "hardcode an invented model id' rule. Bundled into the no-flags "
        "default run.",
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
        or args.check_heredoc_dotenv
        or args.check_dispatcher_cvd_pin
        or args.check_marker_registry
        or args.check_agent_model_pins
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
        # A posted-but-unregistered marker kind is the same drift class
        # (doc surface vs canonical registry) — bundle here too.
        errors.extend(check_marker_registry(workflow))
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
    if args.check_heredoc_dotenv or no_flags:
        errors.extend(check_heredoc_dotenv())
    if args.check_dispatcher_cvd_pin or no_flags:
        errors.extend(check_dispatcher_cvd_pin())
    if args.check_marker_registry and not args.check_references:
        errors.extend(check_marker_registry(workflow))
    if args.check_agent_model_pins or no_flags:
        errors.extend(check_agent_model_pins())

    if errors:
        for err in errors:
            sys.stderr.write(f"workflow_lint: {err}\n")
        sys.stderr.write(f"workflow_lint: FAIL ({len(errors)} error(s))\n")
        return 1

    sys.stderr.write("workflow_lint: PASS\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
