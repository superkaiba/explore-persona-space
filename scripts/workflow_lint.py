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
  running from IS scanned so a workflow-fix `/issue` session's implementer
  can validate its own edits;
  see :func:`_other_worktree_prefix` for the scoping rule) and FAIL on
  any ``scripts/<name>.py`` reference whose target does not exist under
  ``scripts/``. Mechanically prevents the dead-tool / invented-tool
  failure class where an agent follows a step that runs a
  deleted-or-never-created helper and CalledProcessErrors.
* ``--check-skill-refs`` (also bundled into ``--check-references`` and the
  no-flags default run): walk every ``.md`` under ``.claude/agents/``,
  every ``SKILL.md`` under ``.claude/skills/``, every ``.md`` under
  ``.claude/rules/``, ``CLAUDE.md``, and ``.claude/workflow.yaml``
  (OTHER worktrees excluded, the current worktree scanned — see
  :func:`_other_worktree_prefix`) and FAIL on any backtick-delimited
  ``/<skill-name>`` slash-command token that resolves neither to a live
  skill DIRECTORY under ``.claude/skills/`` NOR to
  :data:`SKILL_REF_ALLOWLIST` (exact token or ``<plugin>:`` namespace
  prefix). Closes the skill-rename / skill-retirement rot class
  (#713/#714): ``--check-references`` only resolves
  ``(see workflow.yaml § X)`` tokens, so a retired skill (e.g.
  ``/weekly``) leaves stray load-bearing references that no mechanical
  check catches. Backtick-anchor + trailing-``/``-rejecting lookahead +
  fenced-code exclusion are the false-positive controls; lines carrying
  :data:`HISTORICAL_REF_OPT_OUT` are a one-off narrative escape.
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
* ``--check-upload-as-file`` (also bundled into the no-flags default
  run): AST-walk every ``*.py`` under ``scripts/`` and FAIL on any
  ``_upload(...)`` call (the shared HF-Hub upload helper,
  ``explore_persona_space.orchestrate.hub._upload``) whose first
  positional / ``local_path`` argument carries a SINGLE-FILE signal but
  does not pass ``upload_as_file=True``. ``_upload`` raises
  ``ValueError`` UNCONDITIONALLY on a file path without that kwarg
  (``hub.py`` ~line 560), because ``huggingface_hub.upload_folder``
  silently no-ops on a single-file path — so a per-file upload loop
  crashes on the FIRST file, after the expensive phases are spent. Two
  signal classes: a DECIDABLE single-file arg0 (a string literal ending
  in a known artifact extension, e.g. ``"out/summary.json"``, or a
  ``dir / "name.pt"`` path-div expression) FAILs unless
  ``upload_as_file=True`` (a decidable file even with an explicit
  ``=False`` FAILs — that is the #595 silent-no-op shape); a NAME-CONTEXT
  arg0 (a bare ``Name`` carrying any of three single-file signals — a
  file-suffix identifier e.g. ``summary_path`` (the #612 offender); a
  per-file glob/rglob/iterdir LOOP variable e.g. ``for f in
  sorted(dir.glob("*.json"))`` (the #595/#640 production crash); or a
  ``path_in_repo=f"...{X.name}"`` interpolation in the same call (the #640
  idiom)) FAILs only when the ``upload_as_file`` kwarg is ENTIRELY ABSENT
  — an explicit kwarg of either value is the author's deliberate
  declaration and is deferred to. Folder uploads (a generic ``local`` /
  ``local_dir`` / ``staging`` variable, no file-suffix name, no literal)
  pass untouched. Waive a genuinely-correct flagged call with
  ``# UPLOAD_AS_FILE_EXEMPT: <reason>`` (reason ≥ 10 chars) on the call's
  first physical line or the immediately preceding non-blank line.
  Closes the #595 → #640 → #612 recurrence class: the rule lived only as
  prose (gotchas.md "``hub._upload`` raises ``ValueError`` …") and was
  re-introduced three times, twice surviving a Claude reviewer (the
  Codex twin caught #640); a CPU smoke that skips the GPU phase never
  exercises the upload branch, so nothing mechanical caught it
  pre-merge.
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
* ``--check-batch-judge-client`` (also bundled into the no-flags default
  run): AST-walk every ``*.py`` under ``scripts/`` and
  ``src/explore_persona_space/`` and FAIL on any inline Anthropic Message
  Batches API call (``<client>.messages.batches.create`` — both the call
  form and the bare ``asyncio.to_thread(...create, ...)`` reference form)
  outside the sanctioned shared batch clients
  (:data:`BATCH_JUDGE_SANCTIONED_FILES`: ``eval/batch_judge.py``,
  ``eval/judge_dispatch.py``, ``llm/anthropic_client.py``). New batch
  judging MUST route through the #663-hardened client (shards ≤8k/batch,
  bounds the poll on the batch's own ``expires_at`` so an in-SLA batch
  self-harvests for free instead of a deadline-less ``while True ...
  sleep`` poller pinning idle GPUs, resumes by custom_id). Closes the
  #658/#663 class (2026-06-24): an autonomous judge run hand-rolled a
  90k-request batch + deadline-less poller, bypassing the client, then
  PARKED to propose a PAID rerun though the in-SLA batch would self-harvest
  for free — #663 built the client but added no guardrail forcing callers
  onto it. Documented legacy inline-batch callers predating the check are
  grandfathered in :data:`BATCH_JUDGE_LEGACY_ALLOWLIST` (mostly data-gen,
  plus one analysis classifier and one pre-#663 judge — each flagged inline,
  all out of the workflow-surface edit scope, migration is a follow-up); a
  genuinely-correct new non-judge batch caller waives with
  ``# BATCH_JUDGE_CLIENT_EXEMPT: <reason>`` (reason ≥
  :data:`BATCH_JUDGE_CLIENT_WAIVER_MIN_REASON_CHARS` chars) on the call's
  first physical line or the immediately preceding non-blank line.

Exit codes:

* ``0`` PASS
* ``1`` FAIL — stderr lists every error with file:line context.
"""

from __future__ import annotations

import argparse
import ast
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

# `--check-skill-refs`: every backtick-delimited `/<skill-name>` slash-command
# token in the workflow-doc surface MUST resolve to a live project skill (a
# `.claude/skills/<name>/` directory) or to SKILL_REF_ALLOWLIST. Backtick-anchor
# + trailing lookahead are the FP controls: only the slash-command convention
# matches, and a path segment (`/workspace/logs`, `/tmp/x`, `/mnt/...`) is
# rejected because the char after the token is `/`, not the required
# backtick / whitespace / `)` boundary. Group 1 = skill name, optionally
# `<plugin>:<skill>`.
SKILL_REF_RE = re.compile(r"`/([a-z][a-z0-9-]+(?::[a-z0-9-]+)?)(?=[`\s)])")

_FENCE_RE = re.compile(r"^\s*(?:```|~~~)")

# `--check-skill-refs`: legitimate slash-commands that are NOT project skill dirs
# under `.claude/skills/`, so a backticked reference must be waived rather than
# flagged. Add an entry only with a comment naming WHY it is not a project skill.
# The lint scopes to the PROJECT repo only (it deliberately does NOT reach into
# ~/.claude/skills/, which would make the lint host-dependent and the bundled
# pytest non-hermetic), so user-global skills are allowlisted here.
SKILL_REF_ALLOWLIST: frozenset[str] = frozenset(
    {
        # --- Plugin-namespace prefixes (live plugin trees, not .claude/skills) ---
        "codex:",  # /codex:rescue, /codex:setup, ... (codex-plugin-cc)
        "code-review:",  # /code-review:code-review
        "superpowers:",  # /superpowers:* (14 members)
        "huggingface-skills:",  # /huggingface-skills:* (~16 members)
        "frontend-design:",  # /frontend-design:frontend-design
        "ml-paper-writing:",  # /ml-paper-writing:*
        "academic-writing-agents:",  # /academic-writing-agents:academic
        # --- User-global skills (live in ~/.claude/skills/, not this repo) ---
        "humanize",  # ~/.claude/skills/humanize (de-AI prose pass)
        "loop",  # ~/.claude/skills/loop (recurring driver; manual /issue-tick equiv)
        "plan",  # ~/.claude/skills/plan (planning skill)
        "self-review",  # ~/.claude/skills/self-review
        "peer-review",  # ~/.claude/skills/peer-review
        "memory-sleep",  # ~/.claude/skills/memory-sleep (nightly consolidation)
        "update-config",  # update-config skill (settings.json), not a project dir
        # --- Built-in Claude Code CLI commands (no skill dir anywhere) ---
        "clear",  # built-in /clear (alias /new)
        "new",  # built-in /new (alias /clear)
        "compact",  # built-in /compact
        "rewind",  # built-in /rewind
        "mcp",  # built-in /mcp (MCP reconnect)
        "review",  # built-in /review (PR review)
        "init",  # built-in /init (CLAUDE.md init)
        "security-review",  # built-in /security-review
        # --- Interactive plan-gate inputs + PM-session aliases (no skill dir) ---
        "approve",  # user plan-approval action (`/approve`)
        "revise",  # user plan-revise action (`/revise <notes>`)
        "audit",  # PM-session shorthand for an audit pass (pm/SKILL.md)
        "code-refactoring",  # refactoring-theory ref-table label (deep-clean/SKILL.md)
        # --- Dashboard route paths written in the slash-command shape (no skill dir) ---
        "log",  # `/log` dashboard feed
        "sessions",  # `/sessions` dashboard page
        "updates",  # `/updates` dashboard MDX editor route
        # --- Non-skill prose/path tokens the backtick form still catches ---
        "workspace",  # `/workspace` pod path written inside backticks (RunPod `/workspace`)
        "intent",  # `/intent` — a phase/arg token in prose
        "absent",  # `/absent` — a marker-state token in prose
        "override",  # `/override subset` prose (experiment-implementer.md)
        "binary",  # `.npz/binary` prose (uploader.md)
        "terminal",  # `blocked/terminal` prose (background-automation.md)
        "expensive-band",  # `auto_run/expensive-band` prose (issue/SKILL.md)
    }
)

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


# `--check-upload-as-file`: the shared HF-Hub upload helper
# `explore_persona_space.orchestrate.hub._upload` raises `ValueError`
# UNCONDITIONALLY when handed a FILE path without `upload_as_file=True`
# (`hub.py` ~line 560), because `huggingface_hub.upload_folder` silently
# no-ops on a single-file path — a per-file upload loop crashes on the
# FIRST file, after the expensive phases are spent (#595 → #640 → #612).
# This check AST-walks `scripts/**/*.py` and flags `_upload(...)` calls
# whose first positional / `local_path` argument carries a single-file
# signal but omits `upload_as_file=True`. See `check_upload_as_file` for
# the full flagged/not-flagged matrix.
#
# Artifact extensions that mark a string-literal arg0 as a single file.
UPLOAD_FILE_EXTENSIONS: tuple[str, ...] = (
    ".json",
    ".jsonl",
    ".pt",
    ".npy",
    ".csv",
    ".png",
    ".pdf",
    ".txt",
    ".safetensors",
    ".bin",
    ".html",
    ".md",
    ".yaml",
    ".yml",
)
# Variable-name suffixes (lower-cased) that mark a bare-`Name` arg0 as a
# single file by naming convention (the #612 offender bound `summary_path`).
# Deliberately NOT including the generic folder names that appear at the
# live call sites (`local`, `local_dir`, `staging`, `entry`) — those carry
# no file-suffix and so never match.
UPLOAD_FILE_NAME_SUFFIXES: tuple[str, ...] = (
    "_file",
    "_path",
    "_json",
    "_jsonl",
    "_pt",
    "_npy",
    "_csv",
    "_png",
    "_pdf",
    "_txt",
    "_html",
    "_md",
)
# Path-iteration methods whose loop variable yields per-FILE paths — the
# `for f in dir.glob("*.json"): _upload(f, ...)` shape behind both production
# crashes (#595/#640). `iterdir` is included because the canonical use is a
# flat per-file sweep; whether a given loop is FILE- vs directory-shaped is
# decided by `_glob_iter_yields_files` (a dir-shaped / extensionless pattern
# like `glob("*/")` / `glob("*")` defers, so a dir loop is NOT mis-flagged).
UPLOAD_GLOB_LOOP_METHODS: tuple[str, ...] = ("glob", "rglob", "iterdir")
# Inline waiver for a genuinely-correct flagged `_upload` call (e.g. a
# `*_path` variable that is really a directory). Reason ≥ 10 chars, same
# convention as CVD_PIN_EXEMPT / WANDB_INTENTIONALLY_DISABLED.
UPLOAD_AS_FILE_WAIVER_RE = re.compile(r"#\s*UPLOAD_AS_FILE_EXEMPT\s*:\s*(.+?)\s*$")
UPLOAD_AS_FILE_WAIVER_MIN_REASON_CHARS = 10


# `--check-batch-judge-client`: every inline Anthropic Message Batches API
# call (`<client>.messages.batches.create(...)`) outside the sanctioned
# shared batch clients MUST route through one of those clients instead. The
# #663-hardened client (`explore_persona_space.eval.batch_judge`) and its
# dispatcher (`eval.judge_dispatch`) shard at ≤8k requests/batch, bound the
# poll on the batch's own `expires_at` (so an in-SLA batch self-harvests for
# free instead of a deadline-less `while True ... sleep` poller pinning idle
# GPUs), and resume by custom_id; the low-level wrapper
# (`llm.anthropic_client`) supplies the `expires_at` deadline helpers the two
# higher layers import. A hand-rolled `messages.batches.create` +
# deadline-less poller bypasses ALL of that.
#
# Closes the #658/#663 class (autonomous Phase 0/1 judge run, 2026-06-24): an
# inline 90k-request batch with a `while True ... time.sleep(30)` poller
# bypassed the hardened client, then the session PARKED to propose a PAID
# rerun even though the in-SLA batch would self-harvest for free. #663 built
# the client but added NO guardrail forcing callers onto it. This check is the
# mechanical enforcement: any NEW inline batch-create outside the sanctioned
# clients FAILs at lint time, not after a long idle poll.
#
# Detection: an `ast.Attribute` node whose chain ends in `.batches.create`
# (matches BOTH the call form `client.messages.batches.create(...)` AND the
# bare-reference form `asyncio.to_thread(client.messages.batches.create, ...)`
# that `judge_dispatch` itself uses). Deduped by line.
#
# Exempt:
#   * the sanctioned client files (:data:`BATCH_JUDGE_SANCTIONED_FILES`,
#     matched by path suffix);
#   * the documented legacy DATA-GENERATION offenders predating this check
#     (:data:`BATCH_JUDGE_LEGACY_ALLOWLIST`) — these generate training data
#     via the batch API (NOT judging), are out of the workflow-surface edit
#     scope, and are grandfathered in the lint rather than waived per-file;
#     a NEW file is never added here (the waiver comment below is the path
#     for genuinely-correct new non-judge batch callers);
#   * any call site waived with `# BATCH_JUDGE_CLIENT_EXEMPT: <reason>`
#     (reason ≥ :data:`BATCH_JUDGE_CLIENT_WAIVER_MIN_REASON_CHARS` chars) on
#     the call's first physical line or the immediately preceding non-blank
#     line — same convention as UPLOAD_AS_FILE_EXEMPT / CVD_PIN_EXEMPT.
BATCH_JUDGE_SANCTIONED_FILES: tuple[str, ...] = (
    "src/explore_persona_space/eval/batch_judge.py",
    "src/explore_persona_space/eval/judge_dispatch.py",
    "src/explore_persona_space/llm/anthropic_client.py",
)
# Legacy inline batch callers that predate this check (2026-06-25), GRANDFATHERED
# pending migration. The MAJORITY are training-data GENERATION (the `generate_*`
# / `build_*` / `gen_*` / `run_a3*` rows); ONE is NOT data-gen and is flagged
# inline so the rationale stays honest — `analyze_axis_tails.py` is an
# LLM-taxonomy ANALYSIS classifier. All are experiment code, OUT of the
# workflow-surface edit scope, so they are grandfathered in the lint
# (the MARKER_REGISTRY_ALLOWLIST model) rather than waiver-commented per-file —
# this lands the check green without touching experiment scripts. A NEW offender
# is never added here (the `# BATCH_JUDGE_CLIENT_EXEMPT:` waiver is the path for
# a genuinely-correct new non-judge caller). The pre-#663 JUDGE entry
# (`i528_phase4_judge.py`) was MIGRATED onto the sanctioned
# `eval.batch_judge.submit_sharded_batches_fire_and_forget` helper (#668) — the
# rule's intended outcome — and dropped from this set; its `--backend batch`
# submit no longer calls `messages.batches.create` inline.
# CAVEAT: allowlist membership exempts the WHOLE file, not just the documented
# pre-existing call — a future edit that adds a NEW `messages.batches.create`
# (even a hand-rolled judge batch) to an allowlisted file is silently exempt.
# When migrating a file off the batch API, DROP it from this set.
# Paths are repo-root-relative POSIX, matched exactly.
BATCH_JUDGE_LEGACY_ALLOWLIST: frozenset[str] = frozenset(
    {
        # Training-data generation (the original "not judging" rationale).
        "scripts/generate_leakage_data.py",
        "scripts/build_i181_data.py",
        "scripts/generate_issue376_marker_install.py",
        "scripts/regenerate_issue404_medical.py",
        "scripts/gen_issue475_scaffold_data.py",
        "scripts/run_a3b_experiment.py",
        "scripts/generate_a3_data.py",
        "scripts/issue502_generate_probes.py",
        "scripts/issue_188_evolutionary_trigger.py",
        "scripts/generate_trait_transfer_data_v2.py",
        "scripts/generate_issue404_json_neg.py",
        "scripts/run_a3_leakage.py",
        # NOT data-gen — flagged so the allowlist rationale stays honest:
        "scripts/analyze_axis_tails.py",  # LLM-taxonomy ANALYSIS classifier
    }
)
BATCH_JUDGE_CLIENT_WAIVER_RE = re.compile(r"#\s*BATCH_JUDGE_CLIENT_EXEMPT\s*:\s*(.+?)\s*$")
BATCH_JUDGE_CLIENT_WAIVER_MIN_REASON_CHARS = 10


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
      a workflow-fix `/issue` session's implementer needs to validate its
      edits, but scanning
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
    are currently running from IS scanned so a workflow-fix `/issue`
    session's implementer running inside a worktree can validate its own edits.
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


def _iter_skill_ref_target_files(repo_root: Path) -> list[Path]:
    """Production scope for --check-skill-refs: agents + all SKILL.md + rules +
    CLAUDE.md + workflow.yaml, OTHER-worktree copies excluded (the current
    worktree IS scanned so a workflow-fix /issue session validates its own
    edits). Mirrors :func:`_iter_ask_target_files` but with the wider root
    set the skill-rot failure class actually touches (the `/weekly`-rot
    concern lives partly in CLAUDE.md and `.claude/rules/`, which an
    agents+skills-only scope would NOT guard)."""
    current_prefix = _other_worktree_prefix(repo_root)
    files: list[Path] = []
    agents = repo_root / ".claude" / "agents"
    skills = repo_root / ".claude" / "skills"
    rules = repo_root / ".claude" / "rules"
    if agents.exists():
        files += [
            p
            for p in agents.glob("*.md")
            if p.is_file() and not _is_other_worktree_path(p, current_prefix)
        ]
    if skills.exists():
        files += [
            p
            for p in skills.glob("**/SKILL.md")
            if p.is_file() and not _is_other_worktree_path(p, current_prefix)
        ]
    if rules.exists():
        files += [
            p
            for p in rules.glob("*.md")
            if p.is_file() and not _is_other_worktree_path(p, current_prefix)
        ]
    for extra in (repo_root / "CLAUDE.md", repo_root / ".claude" / "workflow.yaml"):
        if extra.is_file() and not _is_other_worktree_path(extra, current_prefix):
            files.append(extra)
    return sorted(files)


def _resolve_skill_ref_target_files(roots: list[Path] | None) -> list[Path]:
    """Production callers pass ``roots=None`` and we walk the canonical wide
    surface. Tests pass ``roots=[tmp_path]`` to scope the walk to a fixture
    directory (mirrors :func:`_resolve_ask_target_files`)."""
    if roots is None:
        return _iter_skill_ref_target_files(_REPO_ROOT)
    files: list[Path] = []
    for root in roots:
        if root.is_file():
            files.append(root)
        else:
            files.extend(p for p in root.glob("**/*.md") if p.is_file())
    return sorted(files)


def _live_skill_names(skills_dir: Path) -> set[str]:
    """Live project-skill names = immediate child DIRECTORIES of
    ``.claude/skills/``. By DIRECTORY existence, NOT ``*/SKILL.md``:
    ``clean-results`` is a live skill dir (SPEC.md/exemplars/, no SKILL.md)
    and ``/clean-results`` is a real reference, so a ``*/SKILL.md`` glob would
    wrongly flag it."""
    if not skills_dir.exists():
        return set()
    return {p.name for p in skills_dir.iterdir() if p.is_dir()}


def _skill_ref_resolves(ref: str, live: set[str], allow: frozenset[str]) -> bool:
    """A backticked ``/<ref>`` resolves iff it names a live skill dir, an
    allowlisted exact token, or (when namespaced ``<plugin>:<skill>``) a token
    whose ``<plugin>:`` prefix is allowlisted."""
    if ref in live:  # live project skill dir
        return True
    if ref in allow:  # allowlisted exact token
        return True
    if ":" in ref:  # plugin-namespaced: prefix match
        return (ref.split(":", 1)[0] + ":") in allow
    return False


def check_skill_references(
    *,
    roots: list[Path] | None = None,
    skills_dir: Path | None = None,
    allowlist: frozenset[str] | None = None,
) -> list[str]:
    """Walk the workflow-doc surface (agents + skills + rules + CLAUDE.md +
    workflow.yaml) and FAIL on any backtick-delimited ``/<skill-name>`` token
    that resolves neither to a live skill dir under ``.claude/skills/`` NOR to
    :data:`SKILL_REF_ALLOWLIST` (exact token or namespace prefix).

    Closes the skill-rename / skill-retirement rot class: ``--check-references``
    only resolves ``(see workflow.yaml § X)`` tokens, so a retired skill (e.g.
    ``/weekly``) leaves stray load-bearing references that no mechanical check
    catches (#713 Methodology-critic finding; #714).

    Lines inside fenced code blocks (``` / ~~~) are skipped (HTML close tags /
    sed one-liners / regex fragments). Lines carrying
    :data:`HISTORICAL_REF_OPT_OUT` are skipped (one-off narrative citation).
    Plugin-namespaced refs resolve via the allowlist prefix set (or,
    forward-compat, an on-disk ``<plugin>:<skill>/`` dir).

    ``roots`` / ``skills_dir`` / ``allowlist`` are unit-test override hooks;
    production callers pass None.
    """
    errors: list[str] = []
    sk_dir = skills_dir if skills_dir is not None else _REPO_ROOT / ".claude" / "skills"
    live = _live_skill_names(sk_dir)
    allow = allowlist if allowlist is not None else SKILL_REF_ALLOWLIST
    for path in _resolve_skill_ref_target_files(roots):
        in_fence = False
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if _FENCE_RE.match(line):
                in_fence = not in_fence
                continue
            if in_fence or HISTORICAL_REF_OPT_OUT in line:
                continue
            for match in SKILL_REF_RE.finditer(line):
                ref = match.group(1)
                if _skill_ref_resolves(ref, live, allow):
                    continue
                errors.append(
                    f"{path}:{lineno}: unresolved skill reference '/{ref}' — not a "
                    f"live skill under .claude/skills/ and not in "
                    f"SKILL_REF_ALLOWLIST. Repoint to the current skill, remove the "
                    f"dead reference, add the command to SKILL_REF_ALLOWLIST "
                    f"(user-global / plugin / built-in command, with a justifying "
                    f"comment), or — for a one-off narrative citation — append "
                    f"'{HISTORICAL_REF_OPT_OUT}' to the line."
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
    worktree a workflow-fix `/issue` session's implementer runs from scans
    its own copies (same
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


def _upload_arg0(call: ast.Call) -> ast.expr | None:
    """Return the AST node for the ``_upload`` call's local-path argument
    (first positional, else the ``local_path`` / ``local`` keyword), or
    None if neither is present."""
    if call.args:
        return call.args[0]
    for kw in call.keywords:
        if kw.arg in ("local_path", "local"):
            return kw.value
    return None


def _upload_arg0_is_decidable_file(node: ast.expr) -> bool:
    """True iff ``node`` is a DECIDABLE single-file path: a string literal
    ending in a known artifact extension (``"out/summary.json"``) or a
    ``<expr> / "name.ext"`` path-division whose right operand is such a
    literal (the canonical ``out_dir / "shift.pt"`` shape)."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value.lower().endswith(UPLOAD_FILE_EXTENSIONS)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        right = node.right
        if isinstance(right, ast.Constant) and isinstance(right.value, str):
            return right.value.lower().endswith(UPLOAD_FILE_EXTENSIONS)
    return False


def _upload_arg0_is_named_file(node: ast.expr) -> bool:
    """True iff ``node`` is a bare ``Name`` whose identifier ends in a
    single-file naming suffix (``summary_path``, ``shift_pt``, ``foo_json``).
    A HEURISTIC signal — applied only when ``upload_as_file`` is entirely
    absent (see :func:`check_upload_as_file`)."""
    return isinstance(node, ast.Name) and node.id.lower().endswith(UPLOAD_FILE_NAME_SUFFIXES)


def _glob_iter_method(iterator: ast.expr) -> str | None:
    """If ``iterator`` is a ``<expr>.glob(...)`` / ``.rglob(...)`` / ``.iterdir()``
    call (a per-file path iterator), return the method name; else None.
    A ``sorted(dir.glob(...))`` / ``list(dir.glob(...))`` wrapper is unwrapped
    (the live #640 offender binds ``files = sorted(raw_dir.glob("*.json"))``)."""
    if not isinstance(iterator, ast.Call):
        return None
    fn = iterator.func
    # Unwrap a single ``sorted(...)`` / ``list(...)`` / ``tuple(...)`` wrapper
    # around the glob call (one level — the realistic nesting).
    if isinstance(fn, ast.Name) and fn.id in ("sorted", "list", "tuple") and iterator.args:
        return _glob_iter_method(iterator.args[0])
    if isinstance(fn, ast.Attribute) and fn.attr in UPLOAD_GLOB_LOOP_METHODS:
        return fn.attr
    return None


def _glob_iter_yields_files(iterator: ast.expr) -> bool:
    """True iff ``iterator`` is a per-FILE path iterator the glob-loop single-
    file signal should fire on. Positive (fire) only when the file-vs-directory
    intent is decidable as FILE — conservative by design so a directory sweep is
    never mis-flagged (the candidate's ``glob("*/")`` defer-to-folder case):

    * ``.iterdir()`` — fires (the canonical flat per-file sweep; the candidate's
      test 3 FAIL case).
    * ``.glob(<pat>)`` / ``.rglob(<pat>)`` — fires ONLY when ``<pat>`` is a
      string literal containing a known artifact extension token
      (:data:`UPLOAD_FILE_EXTENSIONS`, e.g. ``"*.json"`` / ``"**/*.pt"``). A
      directory-shaped pattern (``"*/"``, ``"runs/*"``) or any pattern without a
      file-extension token (``"*"``) DEFERS (returns False) — better to leave a
      genuine directory loop unflagged than to manufacture a false positive,
      since the riskiest per-file cases are independently caught by the
      ``path_in_repo=f"...{X.name}"`` signal.

    Unwraps the same ``sorted(...)`` / ``list(...)`` / ``tuple(...)`` wrapper as
    :func:`_glob_iter_method`."""
    if not isinstance(iterator, ast.Call):
        return False
    fn = iterator.func
    if isinstance(fn, ast.Name) and fn.id in ("sorted", "list", "tuple") and iterator.args:
        return _glob_iter_yields_files(iterator.args[0])
    if isinstance(fn, ast.Attribute):
        if fn.attr == "iterdir":
            return True
        if fn.attr in ("glob", "rglob") and iterator.args:
            pat = iterator.args[0]
            if isinstance(pat, ast.Constant) and isinstance(pat.value, str):
                return pat.value.lower().endswith(UPLOAD_FILE_EXTENSIONS)
    return False


def _upload_arg0_is_glob_loop_var(call: ast.Call, arg0: ast.expr, tree: ast.AST) -> bool:
    """True iff ``arg0`` is a bare ``Name`` bound by an enclosing
    ``for <name> in <per-file glob/rglob/iterdir iterator>:`` (the per-file
    sweep shape behind the #595/#640 production crashes — ``for f in files:``
    where ``files = sorted(dir.glob("*.json"))``), counting BOTH the inline
    ``for f in dir.glob(...)`` form and the two-statement form where the loop
    iterates a local previously bound to a glob result.

    Only fires when the iterator decidably yields FILES (see
    :func:`_glob_iter_yields_files`) so a directory loop (``glob("*/")``) is
    not mis-flagged. ``tree`` is the module AST; the walk early-outs as soon as
    a binding per-file ``for`` is found."""
    if not isinstance(arg0, ast.Name):
        return False
    name = arg0.id
    # Map local-name -> glob iterator for ``<name> = sorted(dir.glob(...))``
    # style bindings so a ``for f in files:`` whose ``files`` is a glob result
    # is recognized. Only the simple single-target ``Name = <glob>`` form.
    glob_bound_locals: dict[str, ast.expr] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            tgt = node.targets[0]
            if isinstance(tgt, ast.Name) and _glob_iter_method(node.value) is not None:
                glob_bound_locals[tgt.id] = node.value
    for node in ast.walk(tree):
        if not isinstance(node, ast.For):
            continue
        if not (isinstance(node.target, ast.Name) and node.target.id == name):
            continue
        # The loop binds our arg0 name. Resolve its iterator: a direct glob
        # call, or a local previously bound to one.
        iterator: ast.expr | None = node.iter
        if _glob_iter_method(iterator) is None and isinstance(iterator, ast.Name):
            iterator = glob_bound_locals.get(iterator.id)
        if iterator is None or not _glob_iter_yields_files(iterator):
            continue
        return True
    return False


def _upload_arg0_referenced_as_path_in_repo_name(call: ast.Call, arg0: ast.expr) -> bool:
    """True iff ``arg0`` is a bare ``Name`` X and the SAME ``_upload`` call has
    a ``path_in_repo=f"...{X.name}"`` kwarg (an f-string interpolating
    ``X.name``). This is the #640 idiom — ``.name`` is only taken on a per-item
    file/path you are uploading individually, so it is a strong single-file
    signal independent of the loop context."""
    if not isinstance(arg0, ast.Name):
        return False
    name = arg0.id
    for kw in call.keywords:
        if kw.arg != "path_in_repo" or not isinstance(kw.value, ast.JoinedStr):
            continue
        for piece in kw.value.values:
            if not isinstance(piece, ast.FormattedValue):
                continue
            val = piece.value
            if (
                isinstance(val, ast.Attribute)
                and val.attr == "name"
                and isinstance(val.value, ast.Name)
                and val.value.id == name
            ):
                return True
    return False


def _upload_as_file_waiver_present(lines: list[str], call_lineno: int) -> bool:
    """Return True iff a ``# UPLOAD_AS_FILE_EXEMPT: <reason>`` waiver
    (reason ≥ :data:`UPLOAD_AS_FILE_WAIVER_MIN_REASON_CHARS` chars) is on
    the call's first physical line (``call_lineno``, 1-based) or the
    immediately preceding non-blank line."""
    idx = call_lineno - 1  # to 0-based
    if 0 <= idx < len(lines):
        m = UPLOAD_AS_FILE_WAIVER_RE.search(lines[idx])
        if m and len(m.group(1).strip()) >= UPLOAD_AS_FILE_WAIVER_MIN_REASON_CHARS:
            return True
    back = idx - 1
    while back >= 0 and lines[back].strip() == "":
        back -= 1
    if back >= 0:
        m = UPLOAD_AS_FILE_WAIVER_RE.search(lines[back])
        if m and len(m.group(1).strip()) >= UPLOAD_AS_FILE_WAIVER_MIN_REASON_CHARS:
            return True
    return False


def check_upload_as_file(*, scripts_dir: Path | None = None) -> list[str]:
    """AST-walk every ``*.py`` under ``scripts/`` and FAIL on any
    ``_upload(...)`` call whose local-path argument carries a single-file
    signal but does not pass ``upload_as_file=True``.

    Rationale: the shared HF-Hub upload helper
    ``explore_persona_space.orchestrate.hub._upload`` raises ``ValueError``
    UNCONDITIONALLY when ``local_path.is_file() and not upload_as_file``
    (``hub.py`` ~line 560), because ``huggingface_hub.upload_folder``
    silently no-ops on a single-file path (logs "is not a directory.
    Keeping local path." and uploads NOTHING, yet verification can still
    pass if same-prefix files already exist — the silent-data-loss class
    the guard was added to close, #595). The folder branch is the DEFAULT
    (``upload_as_file=False``), so a driver that loops
    ``for f in glob("*.json"): _upload(f, ...)`` crashes on the FIRST file,
    after the expensive training/eval phases are already spent. This was
    re-introduced THREE times (#595 → #640 → #612) — twice surviving a
    Claude reviewer (the Codex twin caught #640), because a CPU smoke that
    skips the GPU phase never exercises the upload branch and the rule
    lived only as prose (gotchas.md). This check is the lane-independent
    mechanical enforcement.

    Detection (per ``_upload`` call, arg0 = first positional / ``local_path``
    / ``local`` keyword):

    * DECIDABLE single-file arg0 — a string literal ending in a known
      artifact extension (:data:`UPLOAD_FILE_EXTENSIONS`), or a
      ``<expr> / "name.ext"`` path-division — FAILs unless
      ``upload_as_file=True``. An explicit ``upload_as_file=False`` on a
      decidable file STILL FAILs (that is precisely the #595 silent-no-op
      shape).
    * NAME-CONTEXT arg0 — a bare ``Name`` carrying ANY of three
      single-file signals — FAILs only when the ``upload_as_file`` kwarg is
      ENTIRELY ABSENT. An explicit kwarg of either value is the author's
      deliberate file/folder declaration and is deferred to (a heuristic
      name-context signal must not override an explicit choice — that is
      where false positives would live, since a ``*_path`` variable can
      legitimately hold a directory). The three signals:

      - NAME SUFFIX: the identifier ends in a single-file suffix
        (:data:`UPLOAD_FILE_NAME_SUFFIXES`, e.g. ``summary_path`` — the
        #612 offender).
      - GLOB-LOOP variable: the ``Name`` is the target of an enclosing
        ``for X in <per-file glob/rglob/iterdir iterator>:`` (counting the
        inline ``for f in dir.glob(...)`` form AND the two-statement
        ``files = sorted(dir.glob(...)) ; for f in files:`` form — the
        EXACT #595/#640 production crash). Fires only when the iterator
        DECIDABLY yields files: ``.iterdir()``, or ``.glob(<pat>)`` /
        ``.rglob(<pat>)`` whose literal pattern carries a known artifact
        extension (``"*.json"`` / ``"**/*.pt"``). A directory-shaped or
        extensionless pattern (``"*/"`` / ``"*"``) DEFERS so a genuine
        directory loop is not mis-flagged.
      - ``path_in_repo`` ``.name`` INTERPOLATION: the SAME call passes
        ``path_in_repo=f"...{X.name}"`` (the #640 idiom — ``.name`` is
        taken only on a per-item path uploaded individually), a single-file
        signal independent of the loop iterator.

    NOT flagged: a generic folder variable (``local`` / ``local_dir`` /
    ``staging`` / ``entry`` — no file-suffix name, no literal); any call
    already passing ``upload_as_file=True``; and any call waived with
    ``# UPLOAD_AS_FILE_EXEMPT: <reason>`` (reason ≥
    :data:`UPLOAD_AS_FILE_WAIVER_MIN_REASON_CHARS` chars) on the call's
    first physical line or the immediately preceding non-blank line.

    Only calls to a function literally named ``_upload`` are inspected
    (bare ``_upload(...)`` or attribute ``hub._upload(...)``) — the
    project's single shared helper. ``upload_file`` / ``upload_folder`` /
    ``upload_model`` / ``upload_raw_completions_to_data_repo`` wrappers are
    deliberately out of scope (they own their own file/folder routing).

    ``scripts_dir`` is an override hook for unit tests; production callers
    pass None and the function walks the canonical ``<repo_root>/scripts``
    tree. Bundled into the no-flags default run (same policy as
    ``check_dispatcher_cvd_pin`` / ``check_heredoc_dotenv``).
    """
    root = scripts_dir if scripts_dir is not None else _REPO_ROOT / "scripts"
    if not root.exists():
        return []
    errors: list[str] = []
    for py in sorted(root.rglob("*.py")):
        if not py.is_file():
            continue
        text = py.read_text(encoding="utf-8")
        try:
            tree = ast.parse(text, filename=str(py))
        except SyntaxError:
            # A scripts/ file that does not parse is its own (separate)
            # problem; this check stays silent on it rather than crashing.
            continue
        lines = text.splitlines()
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            fn = node.func
            fn_name = (
                fn.attr
                if isinstance(fn, ast.Attribute)
                else (fn.id if isinstance(fn, ast.Name) else None)
            )
            if fn_name != "_upload":
                continue
            arg0 = _upload_arg0(node)
            if arg0 is None:
                continue
            has_kw = any(kw.arg == "upload_as_file" for kw in node.keywords)
            kw_val = next((kw.value for kw in node.keywords if kw.arg == "upload_as_file"), None)
            kw_true = isinstance(kw_val, ast.Constant) and kw_val.value is True
            decidable = _upload_arg0_is_decidable_file(arg0)
            named = _upload_arg0_is_named_file(arg0)
            # The #595/#640 production shape: a bare loop variable from a
            # per-file glob/rglob/iterdir sweep, OR a bare Name interpolated as
            # path_in_repo=f"...{X.name}". Both are HEURISTIC name-context
            # signals (like `named`) — they fire only when the upload_as_file
            # kwarg is ENTIRELY ABSENT, deferring to any explicit author choice.
            loop_file = _upload_arg0_is_glob_loop_var(node, arg0, tree)
            kwarg_file = _upload_arg0_referenced_as_path_in_repo_name(node, arg0)
            # FAIL when a decidable file lacks upload_as_file=True, OR when a
            # name-context signal (name-suffix / glob-loop / path_in_repo .name)
            # has the kwarg entirely absent.
            fail = (decidable and not kw_true) or (
                (named or loop_file or kwarg_file) and not has_kw
            )
            if not fail:
                continue
            if _upload_as_file_waiver_present(lines, node.lineno):
                continue
            if decidable:
                signal = "single-file path literal"
            elif named:
                signal = f"file-named arg ('{arg0.id}')"
            elif loop_file:
                signal = f"per-file glob/iterdir loop variable ('{arg0.id}')"
            else:
                signal = f"path_in_repo=f'...{{{arg0.id}.name}}' single-file arg ('{arg0.id}')"
            errors.append(
                f"{py}:{node.lineno}: _upload(...) call with a {signal} does not "
                f"pass upload_as_file=True. hub._upload raises ValueError "
                f"unconditionally on a file path without that kwarg (the folder "
                f"branch silently no-ops on a single file — #595 silent data loss), "
                f"so a per-file upload crashes on the FIRST file after the expensive "
                f"phases are spent (#595/#640/#612). Pass upload_as_file=True for "
                f"single-file uploads, prefer the upload_raw_completions_to_data_repo "
                f"helper for batching raw completions, or — if this arg is really a "
                f"directory — waive with '# UPLOAD_AS_FILE_EXEMPT: <reason>' (reason "
                f"≥ {UPLOAD_AS_FILE_WAIVER_MIN_REASON_CHARS} chars) on the call's "
                f"first line or the previous non-blank line. See "
                f".claude/rules/gotchas.md 'hub._upload raises ValueError'."
            )
    return errors


def _batch_judge_client_waiver_present(lines: list[str], call_lineno: int) -> bool:
    """Return True iff a ``# BATCH_JUDGE_CLIENT_EXEMPT: <reason>`` waiver
    (reason ≥ :data:`BATCH_JUDGE_CLIENT_WAIVER_MIN_REASON_CHARS` chars) is
    on the call's first physical line (``call_lineno``, 1-based) or the
    immediately preceding non-blank line. Same convention as
    :func:`_upload_as_file_waiver_present`."""
    idx = call_lineno - 1  # to 0-based
    if 0 <= idx < len(lines):
        m = BATCH_JUDGE_CLIENT_WAIVER_RE.search(lines[idx])
        if m and len(m.group(1).strip()) >= BATCH_JUDGE_CLIENT_WAIVER_MIN_REASON_CHARS:
            return True
    back = idx - 1
    while back >= 0 and lines[back].strip() == "":
        back -= 1
    if back >= 0:
        m = BATCH_JUDGE_CLIENT_WAIVER_RE.search(lines[back])
        if m and len(m.group(1).strip()) >= BATCH_JUDGE_CLIENT_WAIVER_MIN_REASON_CHARS:
            return True
    return False


def _is_batches_create_attr(node: ast.AST) -> bool:
    """Return True iff ``node`` is an ``ast.Attribute`` for the Anthropic
    Message Batches submit endpoint — chain ``...messages.batches.create``.

    The ``messages`` segment is REQUIRED: it disambiguates Anthropic's
    ``client.messages.batches.create`` (in scope — the judge-batch endpoint)
    from OpenAI's ``client.batches.create`` (a different API with a different
    hardened client, ``llm/openai_client.py``, out of scope for this rule).

    Matches the attribute regardless of whether it is the ``func`` of a
    ``Call`` (``client.messages.batches.create(...)``) or a bare reference
    passed as an argument (``asyncio.to_thread(client.messages.batches.create,
    ...)`` — the form ``judge_dispatch`` itself uses). The caller dedupes by
    line so a call form (which is a single Attribute node) counts once.
    """
    return (
        isinstance(node, ast.Attribute)
        and node.attr == "create"
        and isinstance(node.value, ast.Attribute)
        and node.value.attr == "batches"
        and isinstance(node.value.value, ast.Attribute)
        and node.value.value.attr == "messages"
    )


def check_batch_judge_client(
    *, scripts_dir: Path | None = None, src_dir: Path | None = None
) -> list[str]:
    """AST-walk ``scripts/**/*.py`` and ``src/explore_persona_space/**/*.py``
    and FAIL on any inline ``<client>.messages.batches.create`` outside the
    sanctioned shared batch clients (:data:`BATCH_JUDGE_SANCTIONED_FILES`).

    Rationale: the #663-hardened batch client
    (``explore_persona_space.eval.batch_judge``) + its dispatcher
    (``eval.judge_dispatch``) shard at ≤8k requests/batch, bound the poll on
    the batch's own ``expires_at`` (an in-SLA batch self-harvests for free
    instead of a deadline-less ``while True ... time.sleep`` poller pinning
    idle GPUs), and resume by custom_id; the low-level wrapper
    (``llm.anthropic_client``) supplies the ``expires_at`` deadline helpers
    the two higher layers import. A hand-rolled ``messages.batches.create``
    bypasses ALL of that. The #658/#663 incident (2026-06-24): an autonomous
    judge run inlined a 90k-request batch + deadline-less poller, then PARKED
    to propose a PAID rerun even though the in-SLA batch would self-harvest
    for free; #663 built the client but added no guardrail forcing callers
    onto it. This check is that guardrail.

    Detection: any ``ast.Attribute`` whose chain ends in ``.batches.create``
    (see :func:`_is_batches_create_attr` — covers BOTH the call form and the
    bare ``to_thread(...create, ...)`` reference form), deduped by line.

    Exempt:
      * the sanctioned client files (:data:`BATCH_JUDGE_SANCTIONED_FILES`,
        matched by POSIX path suffix);
      * the documented legacy inline-batch callers
        (:data:`BATCH_JUDGE_LEGACY_ALLOWLIST`) predating this check — mostly
        data-gen, plus one analysis classifier and one pre-#663 judge (each
        flagged inline in the allowlist); all out of the workflow-surface
        edit scope, grandfathered in the lint, migration is a follow-up;
      * any call site waived with ``# BATCH_JUDGE_CLIENT_EXEMPT: <reason>``
        (reason ≥ :data:`BATCH_JUDGE_CLIENT_WAIVER_MIN_REASON_CHARS` chars)
        on the call's first physical line or the immediately preceding
        non-blank line.

    ``scripts_dir`` / ``src_dir`` are override hooks for unit tests;
    production callers pass both None and the function walks the canonical
    ``<repo_root>/scripts`` + ``<repo_root>/src/explore_persona_space`` trees.
    Bundled into the no-flags default run (same policy as
    ``check_upload_as_file`` / ``check_dispatcher_cvd_pin``).
    """
    roots: list[Path] = []
    roots.append(scripts_dir if scripts_dir is not None else _REPO_ROOT / "scripts")
    roots.append(src_dir if src_dir is not None else _REPO_ROOT / "src" / "explore_persona_space")
    errors: list[str] = []
    for root in roots:
        if not root.exists():
            continue
        for py in sorted(root.rglob("*.py")):
            if not py.is_file():
                continue
            try:
                rel = py.resolve().relative_to(_REPO_ROOT.resolve()).as_posix()
            except ValueError:
                # A unit-test fixture tree outside the repo: identify it by
                # its tail under the repo's logical layout instead.
                rel = py.as_posix()
            if any(rel.endswith(s) for s in BATCH_JUDGE_SANCTIONED_FILES):
                continue
            if rel in BATCH_JUDGE_LEGACY_ALLOWLIST:
                continue
            text = py.read_text(encoding="utf-8")
            try:
                tree = ast.parse(text, filename=str(py))
            except SyntaxError:
                # A non-parsing file is its own separate problem; stay silent.
                continue
            lines = text.splitlines()
            seen_lines: set[int] = set()
            for node in ast.walk(tree):
                if not _is_batches_create_attr(node):
                    continue
                lineno = node.lineno
                if lineno in seen_lines:
                    continue
                seen_lines.add(lineno)
                if _batch_judge_client_waiver_present(lines, lineno):
                    continue
                errors.append(
                    f"{py}:{lineno}: inline 'messages.batches.create' outside the "
                    f"sanctioned batch clients. Route batch judging through "
                    f"explore_persona_space.eval.batch_judge "
                    f"(judge_completions_batch) — the #663-hardened client shards "
                    f"≤8k/batch, bounds the poll on the batch's own expires_at "
                    f"(an in-SLA batch self-harvests for free; no deadline-less "
                    f"while-True poller pinning idle GPUs), and resumes by "
                    f"custom_id (#658/#663). For a genuinely-correct NON-judge "
                    f"batch caller, waive with '# BATCH_JUDGE_CLIENT_EXEMPT: "
                    f"<reason>' (reason ≥ "
                    f"{BATCH_JUDGE_CLIENT_WAIVER_MIN_REASON_CHARS} chars) on the "
                    f"call's first line or the previous non-blank line."
                )
    return errors


# A live ``Agent(... subagent_type="workflow-improver" ...)`` spawn instruction.
# Tolerant of whitespace/newlines between the call open and the kwarg and of
# either quote style. The frozen agent file (`.claude/agents/workflow-improver.md`)
# carries its `name:` + DEPRECATED banner and is excluded; this pattern targets
# a live SPAWN site, not a descriptive mention of the word.
_WF_IMPROVER_SPAWN_RE = re.compile(
    r"""Agent\([^)]*subagent_type\s*=\s*["']workflow-improver["']""",
    re.DOTALL,
)


def check_no_workflow_improver_spawn(*, repo_root: Path | None = None) -> list[str]:
    """FAIL if any live ``Agent(subagent_type="workflow-improver", ...)`` spawn remains.

    Retired by #678: workflow-surface fixes are filed as ``kind: infra`` tasks and
    implemented by a background ``/issue <N> --auto`` session, NEVER an
    ``Agent(workflow-improver)`` spawn. The frozen agent file keeps its
    DEPRECATED banner so a stale ``subagent_type="workflow-improver"`` fails loud
    rather than silently mis-routing; it is EXCLUDED from this scan. A stray
    spawn instruction anywhere else in the workflow surface is a regression.

    Pure-Python (no ``rg`` dependency, so the bundled pytest is hermetic). Scans
    ``.claude/`` (excluding ``worktrees/`` sibling copies, ``cache/``,
    ``agent-memory/`` and the frozen agent file), ``CLAUDE.md``, and ``scripts/``
    (excluding THIS lint script, which carries the spawn pattern as the detector
    regex / docstring by construction); ``tasks/`` is out of the workflow surface
    and never scanned. ``repo_root`` is a unit-test override hook; production
    callers pass None (canonical repo root). Bundled into the no-flags default run.
    """
    root = repo_root if repo_root is not None else _REPO_ROOT
    current_prefix = _other_worktree_prefix(root)
    # Excluded: the frozen agent file (deprecated banner + historical body) and
    # THIS lint script itself (it carries the spawn pattern as a regex literal /
    # docstring / error-message / flag-help string by construction — those are
    # the detector, not a live spawn site).
    excluded = {
        (root / ".claude" / "agents" / "workflow-improver.md").resolve(),
        (root / "scripts" / "workflow_lint.py").resolve(),
    }

    targets: list[Path] = []
    claude_dir = root / ".claude"
    if claude_dir.exists():
        for p in claude_dir.rglob("*"):
            if not p.is_file() or p.suffix not in {".md", ".yaml", ".yml", ".py", ".sh"}:
                continue
            s = p.as_posix()
            if "/.claude/cache/" in s or "/.claude/agent-memory/" in s:
                continue
            if _is_other_worktree_path(p, current_prefix):
                continue
            if p.resolve() in excluded:
                continue
            targets.append(p)
    claude_md = root / "CLAUDE.md"
    if claude_md.is_file():
        targets.append(claude_md)
    scripts_dir = root / "scripts"
    if scripts_dir.exists():
        targets.extend(
            p for p in scripts_dir.rglob("*.py") if p.is_file() and p.resolve() not in excluded
        )
        targets.extend(p for p in scripts_dir.rglob("*.sh") if p.is_file())

    errors: list[str] = []
    for p in sorted(set(targets)):
        try:
            text = p.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        for m in _WF_IMPROVER_SPAWN_RE.finditer(text):
            lineno = text.count("\n", 0, m.start()) + 1
            errors.append(
                f'{p}:{lineno}: stale Agent(subagent_type="workflow-improver", ...) spawn. '
                f"Retired by #678 — workflow-surface fixes route through a filed kind:infra "
                f"task + a background /issue <N> --auto session (see "
                f".claude/rules/workflow-fix-on-bug.md), never an Agent(workflow-improver) spawn."
            )
    return errors


def check_gate_ids_unique(workflow: WorkflowYaml) -> list[str]:
    """Verify every gate ``id:`` across ``gates.{inline, park_and_wait,
    conditional}`` is globally unique.

    The pydantic ``GateEntry`` schema validates each gate independently and
    does NOT enforce cross-list id uniqueness, so a renumber collision (e.g.
    task #694's gate renumber) would otherwise pass the lint silently.
    Returns a list of error strings (empty == PASS). Each error names BOTH
    gate names sharing the duplicated id.
    """
    errors: list[str] = []
    if workflow.gates is None:
        return errors
    seen: dict[int, str] = {}  # id -> first gate name that used it
    all_gates = workflow.gates.inline + workflow.gates.park_and_wait + workflow.gates.conditional
    for g in all_gates:
        if g.id in seen:
            errors.append(
                f"duplicate gate id {g.id}: used by both "
                f"'{seen[g.id]}' and '{g.name}' across "
                f"gates.{{inline, park_and_wait, conditional}}. Gate ids "
                f"must be globally unique; renumber one of them in "
                f".claude/workflow.yaml."
            )
        else:
            seen[g.id] = g.name
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
        "--check-skill-refs",
        action="store_true",
        help="Verify every backtick-delimited '/skill-name' reference in "
        ".claude/agents/**.md, .claude/skills/**/SKILL.md, .claude/rules/*.md, "
        "CLAUDE.md, and .claude/workflow.yaml resolves to a live skill dir under "
        ".claude/skills/ or to SKILL_REF_ALLOWLIST (user-global / plugin / built-in "
        "commands). Closes the skill-rename/retirement rot class (#713/#714): "
        "--check-references only resolves workflow.yaml section refs, so a retired "
        "skill like /weekly rots undetected. Bundled into --check-references and the "
        "no-flags default run.",
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
    parser.add_argument(
        "--check-upload-as-file",
        action="store_true",
        help="AST-walk scripts/**/*.py and FAIL on any _upload(...) call "
        "with a single-file local-path argument that omits "
        "upload_as_file=True. hub._upload raises ValueError unconditionally "
        "on a file path without that kwarg (the folder branch silently "
        "no-ops on a single file), so a per-file upload crashes on the "
        "first file after the expensive phases (#595/#640/#612). Waive a "
        "genuinely-correct flagged call with '# UPLOAD_AS_FILE_EXEMPT: "
        "<reason>'. Bundled into the no-flags default run.",
    )
    parser.add_argument(
        "--check-batch-judge-client",
        action="store_true",
        help="AST-walk scripts/**/*.py and src/explore_persona_space/**/*.py "
        "and FAIL on any inline messages.batches.create outside the sanctioned "
        "batch clients (eval/batch_judge.py, eval/judge_dispatch.py, "
        "llm/anthropic_client.py). New batch judging MUST route through the "
        "#663-hardened client (shards ≤8k/batch, bounds the poll on the batch's "
        "expires_at, resumes by custom_id) — a hand-rolled batch + deadline-less "
        "poller pins idle GPUs and bypasses self-harvest (#658/#663). Waive a "
        "genuinely-correct non-judge batch caller with "
        "'# BATCH_JUDGE_CLIENT_EXEMPT: <reason>'. Bundled into the no-flags "
        "default run.",
    )
    parser.add_argument(
        "--check-no-workflow-improver-spawn",
        action="store_true",
        help='FAIL if any live Agent(subagent_type="workflow-improver", ...) '
        "spawn instruction survives anywhere in the workflow surface "
        "(.claude/, CLAUDE.md, scripts/; the frozen .claude/agents/"
        "workflow-improver.md, worktree sibling copies, cache/, agent-memory/, "
        "and tasks/ are excluded). Retired by #678: workflow-surface fixes are "
        "filed as kind:infra tasks + implemented by a background /issue <N> "
        "--auto session, never a workflow-improver subagent spawn. Bundled into "
        "the no-flags default run.",
    )
    parser.add_argument(
        "--check-gate-ids-unique",
        action="store_true",
        help="Verify every gate id across gates.{inline, park_and_wait, "
        "conditional} in .claude/workflow.yaml is globally unique. The "
        "pydantic GateEntry schema validates each gate independently and "
        "does NOT enforce cross-list id uniqueness, so a renumber "
        "collision (task #694) would pass silently. Bundled into the "
        "no-flags default run.",
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
        or args.check_skill_refs
        or args.check_wandb_required
        or args.check_heredoc_dotenv
        or args.check_dispatcher_cvd_pin
        or args.check_marker_registry
        or args.check_agent_model_pins
        or args.check_upload_as_file
        or args.check_batch_judge_client
        or args.check_no_workflow_improver_spawn
        or args.check_gate_ids_unique
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
        # An unresolved /skill-name reference (dead skill rename/retirement) is
        # the same workflow-doc integrity class — bundle here too.
        errors.extend(check_skill_references())
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
    if args.check_skill_refs or no_flags:
        errors.extend(check_skill_references())
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
    if args.check_upload_as_file or no_flags:
        errors.extend(check_upload_as_file())
    if args.check_batch_judge_client or no_flags:
        errors.extend(check_batch_judge_client())
    if args.check_no_workflow_improver_spawn or no_flags:
        errors.extend(check_no_workflow_improver_spawn())
    if args.check_gate_ids_unique or no_flags:
        errors.extend(check_gate_ids_unique(workflow))

    if errors:
        for err in errors:
            sys.stderr.write(f"workflow_lint: {err}\n")
        sys.stderr.write(f"workflow_lint: FAIL ({len(errors)} error(s))\n")
        return 1

    sys.stderr.write("workflow_lint: PASS\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
