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
* ``--check-pipe-python`` (also bundled into the no-flags default run):
  walk every ``*.sh`` under ``scripts/`` and FAIL on any shell pipe
  whose consumer is a bare ``python``/``python3[.N]`` interpreter with
  ``-c`` or ``-m`` (``... | python -c "..."``, ``... | python -m
  json.tool``). This VM has no ``python`` on PATH — only ``python3`` and
  ``uv run python`` — so the pipe dies with ``python: command not found``
  (exit 127); the fix is to pipe into ``uv run python``. The rule lived
  only as prose (CLAUDE.md § Task Workflow API) and was violated ~41x
  across 4+ sessions on 2026-06-29 (#753, at least one hitting exit
  127). The correct ``| uv run python -c`` shape, literal docs strings /
  URLs / filenames containing "python", informational ``which python`` /
  ``apt-get install python3``, comment lines, and ``echo ``-prefixed
  dry-run previews all pass. Sibling of ``--check-heredoc-dotenv``; the
  ``.claude/settings.json`` PreToolUse Bash hook covers the inline ad-hoc
  commands that never reach a committed script.
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
* ``--check-jsonl-splitlines`` (also bundled into the no-flags default
  run): AST-walk every ``*.py`` under ``scripts/`` AND
  ``src/explore_persona_space/`` and FAIL on any ``.splitlines()`` call
  reading JSONL content. ``json.dumps(..., ensure_ascii=False)`` leaves
  raw U+2028/U+2029/NEL inside JSON strings and ``str.splitlines()``
  splits on ALL Unicode line boundaries, so a valid ``\\n``-terminated
  JSONL file shreds into unparseable fragments — a hard crash on strict
  readers, SILENT record loss on tolerant skip-malformed readers, and
  inflated row counts on ``len(...splitlines())`` asserts (incident #825
  run-1d; eight live workflow-surface reader sites across seven files
  fixed with #950). Four narrow
  signals: (a) a ``read_text``-bearing receiver chain whose source
  segment mentions ``jsonl``; (b) a bare receiver ``Name`` matching
  ``jsonl``; (c) the call sits inside a ``jsonl``-named function; (d) a
  ``read_text``-bearing receiver chain whose base ``Name`` is
  ``ev_path``/``events_path``/``concerns_path`` or whose segment names
  ``events.jsonl``/``comments.jsonl``/``concerns.jsonl``. Deliberate false negatives
  (dataflow through other variable names, shell heredocs) are documented
  in the check docstring — the gotchas.md entry carries those. Waive a
  genuinely-safe flagged site with ``# JSONL_SPLITLINES_EXEMPT: <reason>``
  (reason ≥ 10 chars) on the call's first physical line or the
  immediately preceding non-blank line; frozen legacy per-issue
  experiment scripts are grandfathered in
  :data:`JSONL_SPLITLINES_LEGACY_ALLOWLIST` (experiment files ONLY — a
  workflow-surface file is never allowlisted, it is fixed). Unparseable
  files (SyntaxError / non-UTF-8) are skipped WITH a printed notice,
  never silently.
* ``--check-dotenv-before-hf-import`` (also bundled into the no-flags
  default run): AST-walk every ``*.py`` under ``scripts/`` and FAIL on
  any script that uses the BARE python-dotenv ``load_dotenv``
  (``from dotenv import load_dotenv`` / ``import dotenv``) AND imports
  ``huggingface_hub`` (any submodule, top-level OR in-function) WITHOUT
  first importing the project wrapper
  ``explore_persona_space.orchestrate.env.load_dotenv`` (#745). The bare
  dotenv walks cwd (misses the project ``.env`` from a worktree/subdir)
  and sets NO env, so the HF Hub upload accelerators
  (``HF_XET_HIGH_PERFORMANCE`` / ``HF_HUB_ENABLE_HF_TRANSFER``) never get
  their setdefault and a large Hub upload crawls; worse,
  ``huggingface_hub.constants`` freezes ``HF_HUB_ENABLE_HF_TRANSFER`` at
  IMPORT time, so a bare-dotenv script importing ``huggingface_hub`` at
  module top can never pick up the accelerator. The shell-level exports
  (bootstrap_pod.sh / GCE prelude / SLURM env block) are the load-bearing
  fix on the running fleet; this check prevents a NEW script from
  re-introducing the anti-pattern. Waive a genuinely-correct bare-dotenv
  use with ``# DOTENV_LINT_EXEMPT: <reason>`` (reason ≥ 10 chars) on the
  import line or the immediately preceding non-blank line.
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
* ``--check-judge-model-pins`` (also bundled into the no-flags default run):
  walk every ``*.py`` under ``scripts/``, ``src/explore_persona_space/``, and
  ``tests/`` PLUS every ``*.sh`` under ``scripts/`` and FAIL on a hardcoded
  NON-Sonnet judge-model pin at a judge call site. The standing rule pins ONE
  judge — ``claude-sonnet-4-5-20250929`` — for every judged behavior (CLAUDE.md
  "LLM judge"; full recipe ``.claude/rules/llm-judging.md``). The gate is
  ASSIGNMENT/CALL-aware (a ``*JUDGE_MODEL*`` assignment, a ``--judge-model`` /
  ``judge_model=`` / ``JUDGE_MODEL=`` CLI/shell flag, a ``model=`` kwarg with a
  judge token in the +/- :data:`JUDGE_PIN_CONTEXT_WINDOW` window, a split-argv
  ``--judge-model`` + literal-value pair, a ``.sh`` shell-var indirection
  (``JUDGE=<pin>`` consumed by ``--judge-model "${JUDGE}"``), or a judge-script
  ``DEFAULT_MODEL`` / ``MODEL_DEFAULT`` / ``JUDGE_DEFAULT`` constant — #765
  round 2 arms (d)/(e)/(f)), so a bare prose-string mention or a comment is
  never flagged. Legitimate non-Sonnet
  pins (Betley ``gpt-4o`` calibration anchors, the translation-faithfulness
  Haiku judges, the stale-grandfathered legacy Haiku pins) are grandfathered in
  :data:`JUDGE_PIN_LEGACY_ALLOWLIST` (.py) / :data:`JUDGE_PIN_LEGACY_ALLOWLIST_SH`
  (.sh) + the SDK-registry :data:`JUDGE_PIN_FILE_ALLOWLIST`; a new calibration
  control waives with ``# noqa: judge-model-pin`` on the hit or preceding line.
  The canonical pin ``claude-sonnet-4-5-20250929`` carries no forbidden
  substring, so it never matches. Motivating incident: the #650/#657 stale
  legacy-Haiku judge pins (#765).
* ``--check-no-literal-round-marker-versions`` (also bundled into the no-flags
  default run): FAIL on a literal ``v1`` posting instruction for a
  round-versioned marker kind (``epm:experiment-implementation`` /
  ``epm:results`` / ``epm:proposed-tests``) in checked-in workflow prose
  (CLAUDE.md, workflow.yaml, agents/rules ``.md``, every ``SKILL.md``, the
  /issue ``markers.md`` + ``templates/``). Those kinds accrue rows across
  follow-up rounds / TDD resumes / crash-recovery re-posts, and an explicit
  ``--version`` beats the CLI's omitted-version max+1 default, so checked-in
  "post at v1" prose seeds briefs that collide with existing rows (incident
  #825: a follow-up-round brief instructed a literal v1 on a task already at
  v6 — the #389 class). Whole-file scan (a line-wrapped kind/version pair
  still trips); ``v1`` is word-bounded (``v12`` never matches); prose
  evasions like "pass ``--version 1``" are a DELIBERATE boundary covered by
  the brief-contract prose layers, not this lint. Historical archives
  (``.claude/plans/``, ``.claude/agent-memory/``) are out of scan scope (#917).
* ``--check-agent-tools`` (also bundled into the no-flags default run):
  every ``.claude/agents/*.md`` must declare an explicit tool surface — a
  ``tools:`` allowlist or a ``disallowedTools:`` denylist — in its YAML
  frontmatter (task #840). An agent file with neither key inherits the
  parent session's FULL tool inventory including every user-level MCP
  server's tool schemas; incident #778 (2026-07-01): two
  ``experiment-implementer`` spawns each paid ~168K static first-turn
  tokens on the inlined schemas and died in autocompact thrash. Sub-checks:
  (1) declaration required on every file; (2) mentioned ⊆ declared — every
  spec-BODY tool mention per the widened extractor (explicit ``mcp__...``
  tokens, the built-in literals ``WebSearch``/``WebFetch``/
  ``NotebookEdit``/``TodoWrite``, the ``Agent``/``Skill`` phrase forms, and
  prose MCP aliases such as "context7 MCP" →
  ``mcp__plugin_context7_context7``) must be covered by the allowlist,
  modulo the
  :data:`AGENT_TOOLS_MENTION_EXCEPTIONS` dict (descriptive-not-instructive
  mentions, each waived with an inline reason); (2b) declared-name
  validity — every DECLARED ``mcp__...`` token must name a server in
  :data:`KNOWN_MCP_SERVERS` (the harness silently ignores unknown names,
  so a typo strips a capability with no error); (3) a
  ``disallowedTools:``-only file (research-pm) skips the containment check
  but must not deny a body-mentioned tool.
* ``--check-phase-done-reserved`` (also bundled into the no-flags default
  run): walk every ``scripts/**/*.sh`` dispatcher and FAIL any non-redirected
  invocation of a ``scripts/*.py|*.sh`` phase script that contains a genuine
  ``[phase=done]`` emission site — the reserved-token contract of
  ``.claude/rules/pod-side-reporting.md`` requirement 1 (the token in the
  MAIN dispatcher log is reserved for the dispatcher's single terminal line;
  a mid-pipeline child emission reads as a false ``status=done`` to
  ``poll_pipeline.py`` — incidents #545, #920). Emission detection is
  AST-based for ``.py`` (comments / docstrings / ``re.compile`` match sites
  never flag) and quote-aware comment-stripped ``echo|printf|print(`` for
  ``.sh``; every invocation on a logical line is checked (the line is split
  into command segments at unquoted ``&&``/``||``/``;``/``|``/lone-``&``
  separators), and a stdout-redirected invocation (the
  ``> "$WORKER_LOG" 2>&1`` per-worker isolation pattern, scoped to the
  invocation's OWN segment) is skipped, while ``2>&1 | tee`` stays
  checked. Legacy edges are frozen in
  :data:`PHASE_DONE_EDGE_LEGACY_ALLOWLIST` ((invoker, target) edge grain,
  annotated); waive a mode-gated standalone-lane terminal with
  ``# noqa: phase-done-reserved`` on the emission line or the preceding
  non-blank line. Also enforced at commit time by the
  ``workflow-lint-phase-done-reserved`` pre-commit hook on any
  ``scripts/*.sh|py`` change (#930).
* ``--check-stale-label-disposition`` (also bundled into the no-flags default
  run): FAIL if the /issue SKILL.md Step 0 "Stale-label disposition rule"
  paragraph (bold anchor ``**Stale-label disposition rule``, which must be
  UNIQUE — the check carries a negative assertion, so span identity is
  load-bearing) loses any of its five #894/#763 semantic tokens — most
  critically the fresh-label-execute clause ("the label EXECUTES as the
  dispatched round") — or regains an unconditional skip-on-None coupling
  ("On None ... skip", a targeted negative regex over the
  whitespace-normalized paragraph span; a literal-coupling backstop only,
  the positive tokens are the primary defense). Paragraph-scoped: the span
  runs from the anchor to the first blank line, so a mid-paragraph split
  FAILs loudly (#963).

Exit codes:

* ``0`` PASS
* ``1`` FAIL — stderr lists every error with file:line context.
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from collections import Counter
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
# `.claude/skills/<name>/` directory) or to SKILL_REF_ALLOWLIST. Three FP
# controls, all load-bearing (do NOT "simplify" any away):
#   1. leading backtick anchor — the slash-command must OPEN a codespan;
#   2. trailing lookahead `(?=[`\s)])` — the token must CLOSE on a backtick /
#      whitespace / `)`, so a path segment (`/workspace/logs`, `/tmp/x`,
#      `/mnt/...`) is rejected (the char after it is `/`, not the boundary);
#   3. fixed-width negative-lookbehind `(?<!\w)` on the leading backtick — the
#      backtick that OPENS the codespan must NOT be immediately preceded by a
#      word char. This guards the closing-backtick-mistaken-for-opening FP
#      class: prose like `` `false`/unset `` writes a `` `false` `` codespan
#      whose CLOSING backtick abuts `/unset`; without (3) that closing
#      backtick is misread as the OPENING backtick of a phantom `` `/unset ``
#      slash-command (the trailing lookahead then succeeds on the following
#      space). A closing backtick is always immediately preceded by a word
#      char, so `(?<!\w)` rejects exactly that misread while leaving every
#      real opening backtick (at start-of-line / after whitespace / `(` / `[`)
#      matched. Group 1 = skill name, optionally `<plugin>:<skill>`.
SKILL_REF_RE = re.compile(r"(?<!\w)`/([a-z][a-z0-9-]+(?::[a-z0-9-]+)?)(?=[`\s)])")

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

# `--check-pipe-python`: a shell pipe whose CONSUMER is a bare
# `python`/`python3[.N]` interpreter invoked with `-c` or `-m`
# (`... | python -c "..."`, `... | python3 -c "..."`, `... | python -m
# json.tool`) dies at runtime with `python: command not found` (exit
# 127): this VM has NO `python` on PATH — only `python3` and the
# project's `uv run python`. CLAUDE.md § Task Workflow API carries the
# verbatim rule ("Bare `python` is unavailable on this VM — prefix EVERY
# python invocation with `uv run python`, INCLUDING the consumer side of
# a pipe"), but it lived only as prose and was violated ~41x across 4+
# sessions on 2026-06-29 (at least one hitting exit 127). This check
# makes the rule mechanical for committed `scripts/*.sh`; the
# `.claude/settings.json` PreToolUse Bash hook covers the inline ad-hoc
# commands that never reach a script. The fix is always the same: pipe
# into `uv run python` instead (`... | uv run python -c "..."`).
# Direct sibling of `--check-heredoc-dotenv` (a prose rule made a
# `scripts/*.sh` line-scanner, bundled into the no-flags default).
#
# Flagged (a logical shell line, backslash continuations merged):
#   * `cat x | python3 -c "..."` / `foo | python -c "..."` — the pipe
#     consumer is bare python with `-c`;
#   * `foo | python -m json.tool` — same with `-m`;
#   * `foo |python -c "..."` (no space after `|`), `foo | python3.11 -c`,
#     `cat x | python -u -c` (intervening single-dash flag).
# NOT flagged (precision — the anchor `\|` + `-[cm]` keeps these out):
#   * `cat x | uv run python -c "..."` — the token after `|` is `uv`, the
#     CORRECT shape, never the bare interpreter;
#   * a literal docs string / URL / filename merely CONTAINING "python"
#     (`echo "use uv run python"`, `curl .../python-3.12/`,
#     `ls python_helpers.py | wc -l`) — no `| python -[cm]`;
#   * informational invocations (`which python`, `apt-get install
#     python3`) — no pipe consumer;
#   * `python -compose ...` — `-co` is a different flag prefix, not `-c`
#     (the boundary `([^A-Za-z0-9_]|$)` / `\b` requires `-c`/`-m` to END
#     the option);
#   * bare `python ...` at command START (no pipe) — DELIBERATELY out of
#     scope (the daily evidence is exclusively the pipe-into-`-c`/`-m`
#     shape; widening to start-of-command risks false positives without
#     evidence);
#   * `#`-comment lines — skipped before matching (the only skipped class).
# Flagged too (#753 round 2 / F1) — `echo ... | python -c` is a REAL
# producer pipe whose consumer is bare `python` (echo's stdout feeds it),
# so `echo`-prefixed lines are NOT skipped. The earlier blanket
# `echo `-skip silently missed exactly the exit-127 producer-pipe shape
# this check exists to close. Also flagged (#753 round 2 / F3): an
# ATTACHED-argument form `python -c'code'` / `python -m'mod'` (no space,
# quote glued to the option) — the boundary accepts a following quote, so
# both the lint and the hook now block the valid shell shape that would
# otherwise crash exit 127.
# Known limitation (recall, deliberately accepted — mirrors
# `--check-dispatcher-cvd-pin`'s recall sacrifice): a NON-comment line
# whose QUOTED STRING merely contains the substring `| python -c`
# (e.g. `MSG="bad: foo | python -c"`, or a doc `echo "...| python -c..."`)
# WILL match — the regex is line-local and not quote-aware. This is the
# accepted cost of closing F1: to DOCUMENT the bad pattern, use a
# `#`-comment, not an `echo`/quoted string. Such an in-string occurrence
# is vanishingly rare in `scripts/*.sh` (zero in the current tree beyond
# the 2 real offenders). No waiver token in v1 (YAGNI —
# `--check-heredoc-dotenv` ships with no waiver; add one only if a real
# false positive surfaces).
#
# Hook/lint engine equivalence: the lint uses Python `re` (`-[cm]\b`); the
# `.claude/settings.json` PreToolUse hook uses POSIX ERE
# (`-[cm]([^A-Za-z0-9_]|$)`, since POSIX ERE has no `\b`). The two
# boundaries are semantically identical here — `\b` after `c`/`m` matches
# before a quote / space / EOL, and `[^A-Za-z0-9_]|$` is the explicit
# POSIX spelling of that. They AGREE across the full match/no-match set
# AND on the attached-arg (`-c'code'`) and in-string-substring edges; the
# dual-engine test (test_workflow_lint.py) sources the hook ERE from
# `.claude/settings.json` and pins the agreement.
PIPE_PYTHON_RE = re.compile(r"\|\s*python3?(?:\.\d+)?\s+(?:-\S+\s+)*-[cm]\b")

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


# `--check-agent-tools`: every `.claude/agents/*.md` must declare an explicit
# tool surface — a `tools:` allowlist or a `disallowedTools:` denylist — in
# its YAML frontmatter. An agent file with NEITHER key inherits the parent
# session's FULL tool inventory, including every user-level MCP server's tool
# schemas (todoist ~100, google-workspace ~90, playwright ~35, runpod ~28,
# ...); incident #778 (2026-07-01): two `experiment-implementer` spawns each
# paid ~168K static first-turn tokens on the inlined MCP schemas and died in
# autocompact thrash. Task #840 added explicit declarations to all 28 files;
# this check keeps the invariant durable (a NEW agent file landing without a
# declaration reintroduces the bug silently).
#
# Three sub-checks per file (see `check_agent_tools`):
#   1. declaration required (`tools:` or `disallowedTools:` present);
#   2. mentioned ⊆ declared — a spec-BODY tool mention (explicit `mcp__...`
#      token, built-in literal name, `Agent`/`Skill` phrase form, or a prose
#      MCP alias like "context7 MCP") must be covered by the declaration,
#      modulo :data:`AGENT_TOOLS_MENTION_EXCEPTIONS` (descriptive-not-
#      instructive mentions, each with an inline reason);
#   2b. declared-name validity — every DECLARED `mcp__...` token must name a
#      server in :data:`KNOWN_MCP_SERVERS` (a silent typo like
#      `mcp__plugin_context7_contex7` strips a capability with no error);
#   3. denylist files — no BODY-mentioned token may be denied.
#
# Known MCP server names, snapshotted from the live runtime tool-name strings
# (`mcp__<server>__<tool>`) on 2026-07-02 (task #840 plan §4.2). Update when a
# new MCP server is registered at user level AND an agent declaration needs
# it — the constant gates only DECLARED names, so an unrelated new server
# needs no entry here.
KNOWN_MCP_SERVERS: frozenset[str] = frozenset(
    {
        "arxiv",
        "arxiv-latex",
        "google-workspace",
        "happy",
        "plugin_context7_context7",
        "plugin_huggingface-skills_huggingface-skills",
        "plugin_playwright_playwright",
        "runpod",
        "ssh",
        "todoist",
    }
)

# (filename, mentioned-token) pairs the mentioned-⊆-declared check skips.
# Every entry MUST carry an inline reason explaining why the body mention is
# descriptive-not-instructive (the mention documents ANOTHER actor's tool
# use, or a retired/historical pattern) — never add an entry to silence a
# genuine instruction; add the tool to the declaration instead (the additive
# direction is always safe per the #840 plan §13).
AGENT_TOOLS_MENTION_EXCEPTIONS: dict[tuple[str, str], str] = {
    ("planner.md", "mcp__ssh__ssh_execute"): (
        "describes the upload-verifier's Step-8 on-pod glob enumeration "
        "(primary-deliverable block), not a planner instruction"
    ),
    ("codex-clean-result-critic.md", "Agent"): (
        "'spawned from a single Agent(...) call' describes how the "
        "ORCHESTRATOR spawns the ensemble, not an instruction for this "
        "prompt-composer wrapper to call the Agent tool"
    ),
    ("codex-follow-up-critic.md", "Agent"): (
        "'Both spawned from a single Agent(...) call' describes the "
        "orchestrator's ensemble spawn, not a wrapper instruction"
    ),
    ("codex-interpretation-critic.md", "Agent"): (
        "'Both spawned from a single Agent(...) call' describes the "
        "orchestrator's ensemble spawn, not a wrapper instruction"
    ),
    ("codex-reviewer.md", "Agent"): (
        "DEPRECATED file (2026-05-13, never spawned); the Agent(...) mention "
        "describes the historical ensemble spawn pattern"
    ),
    ("workflow-improver.md", "Agent"): (
        "DEPRECATED/frozen file (#678, never spawned; "
        "--check-no-workflow-improver-spawn bans it); Agent( appears only in "
        "historical examples of the retired auto-spawn pattern and a "
        "grep-target example"
    ),
}

# Spec-BODY tool-mention extractor vocabulary (the #840 plan §4.3 "widened
# extractor"). Four layers:
#   (i)  explicit MCP tokens — full `mcp__<server>__<tool>` and server-level
#        `mcp__<server>` forms (one greedy regex matches both; the char class
#        includes `_`, so a full token is captured whole);
#   (ii) built-in literal names an allowlist can silently block;
#   (iii) phrase forms for `Agent` (nested spawns) and `Skill` (runtime
#        skill loads) — the literal bare names are far too common in prose
#        ("the agent", "the skill") to match directly;
#   (iv) prose MCP aliases mapped to their server-level token.
AGENT_TOOLS_MCP_TOKEN_RE = re.compile(r"mcp__[A-Za-z0-9_-]+")
AGENT_TOOLS_BUILTIN_RE = re.compile(r"\b(WebSearch|WebFetch|NotebookEdit|TodoWrite)\b")
AGENT_TOOLS_AGENT_PHRASE_RE = re.compile(r"`Agent`\s+tool|\bAgent\(")
AGENT_TOOLS_SKILL_PHRASE_RE = re.compile(
    r"`Skill`\s+tool|\binvoke[sd]?\s+the\s+`?[\w/-]+`?\s+skill\b", re.IGNORECASE
)
AGENT_TOOLS_MCP_ALIASES: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"`?context7`?\s+MCP", re.IGNORECASE), "mcp__plugin_context7_context7"),
    (re.compile(r"\bSSH\s+MCP\b", re.IGNORECASE), "mcp__ssh"),
    (re.compile(r"\barXiv\s+MCP\b", re.IGNORECASE), "mcp__arxiv"),
)


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


# `--check-jsonl-splitlines` (#950): reading/counting JSONL via
# `str.splitlines()` shreds records whose `ensure_ascii=False` strings carry
# raw U+2028/U+2029/NEL (Unicode line boundaries; incident #825 run-1d).
# Inline waiver for a genuinely-safe flagged site. Reason ≥ 10 chars, same
# convention as UPLOAD_AS_FILE_EXEMPT.
JSONL_SPLITLINES_WAIVER_RE = re.compile(r"#\s*JSONL_SPLITLINES_EXEMPT\s*:\s*(.+?)\s*$")
JSONL_SPLITLINES_WAIVER_MIN_REASON_CHARS = 10
# Signal regexes: (b)/(c) receiver-Name / enclosing-function-name token; (d)
# events/concerns-path receiver base names (the project's uniform conventions
# for `events.jsonl` / `concerns.jsonl` paths; `concerns_path` covers the
# verify_task_body.py check-14 reader shape fixed in #950 round 2).
JSONL_NAME_TOKEN_RE = re.compile(r"jsonl", re.IGNORECASE)
JSONL_EVENTS_PATH_NAME_RE = re.compile(r"^(ev(ents)?|concerns)_path$", re.IGNORECASE)
# Grandfathered legacy `.splitlines()`-on-JSONL sites — repo-root-relative
# POSIX FILE paths (file-level, not line-keyed — line keys rot; these are
# frozen per-issue experiment scripts of terminal/near-terminal tasks reading
# their own mostly-ASCII generated JSONL). HARD RULE: experiment files ONLY —
# a workflow-surface file may NEVER be allowlisted, it must be FIXED (the
# live-tree test asserts the experiment-file path shape mechanically).
JSONL_SPLITLINES_LEGACY_ALLOWLIST: frozenset[str] = frozenset(
    {
        # #823 identity-baseline driver (terminal task, own generated JSONL):
        "scripts/issue823_identity_baseline.py",
        # #778 honest-null figures (terminal task):
        "scripts/issue778_honest_null_figures.py",
        # #488 phase2 smoke calibrator (terminal task):
        "scripts/i488_phase2_smoke_calibrate.py",
        # #778 summary comparison plots (terminal task):
        "scripts/issue778_summary_comparison_plots.py",
        # #667 extraction driver (terminal task):
        "scripts/issue667_extract.py",
        # #650 concept-direction driver (terminal task):
        "scripts/issue650_concept_direction.py",
        # #642 dispatch driver, 5 sites (terminal task):
        "scripts/issue_642/i642_dispatch.py",
        # #612 sycophancy claim audit `_load_jsonl` (experiment package under
        # src/explore_persona_space/experiments/ — experiment code, not
        # workflow surface):
        "src/explore_persona_space/experiments/sycophancy_onpolicy_612/claim_audit.py",
    }
)


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


# `--check-dotenv-before-hf-import`: a script that uses the BARE python-dotenv
# `load_dotenv` (`from dotenv import load_dotenv` or `dotenv.load_dotenv`) AND
# touches `huggingface_hub` (any submodule) WITHOUT importing the project
# wrapper `explore_persona_space.orchestrate.env.load_dotenv` first is the #745
# anti-pattern. The bare dotenv walks cwd (does NOT robustly find the project
# .env from a worktree / subdir) and sets NO env — so the HF Hub upload
# accelerators (HF_XET_HIGH_PERFORMANCE / HF_HUB_ENABLE_HF_TRANSFER) never get
# their in-process setdefault, and any large upload crawls. The project wrapper
# reads the project .env (worktree-aware) AND setdefaults both accelerators.
# Worse, huggingface_hub.constants freezes HF_HUB_ENABLE_HF_TRANSFER at import
# time, so a bare-dotenv script that imports huggingface_hub at module top can
# never pick up the accelerator at all. See `check_dotenv_before_hf_import`.
# Inline waiver: `# DOTENV_LINT_EXEMPT: <reason>` (reason ≥ N chars) on the bare
# `dotenv` import line or the immediately preceding non-blank line — same
# convention as UPLOAD_AS_FILE_EXEMPT / CVD_PIN_EXEMPT.
DOTENV_LINT_WAIVER_RE = re.compile(r"#\s*DOTENV_LINT_EXEMPT\s*:\s*(.+?)\s*$")
DOTENV_LINT_WAIVER_MIN_REASON_CHARS = 10


# `--check-judge-model-pins` (#765): the standing project rule pins ONE judge
# model — `claude-sonnet-4-5-20250929` — for every judged behavior (CLAUDE.md
# "LLM judge = claude-sonnet-4-5-20250929"; the full recipe is
# `.claude/rules/llm-judging.md`). This check flags a hardcoded NON-Sonnet judge
# model at a judge call site. The motivating incident is the #650/#657 stale
# legacy-Haiku pins that re-pinned a non-Sonnet judge for new work.
#
# The gate is ASSIGNMENT/CALL-aware, NOT mention-aware: a forbidden substring on
# a NON-COMMENT line is a HIT iff one of —
#   (a) JUDGE_PIN_VAR_RE matches the line (RHS of a `*JUDGE_MODEL*` /
#       `judge_model` / `JUDGE_MODEL` assignment or key);
#   (b) JUDGE_PIN_FLAG_RE matches the line (`--judge-model` / `judge_model=` /
#       `JUDGE_MODEL=` CLI/shell arg — covers .py argparse defaults AND .sh
#       launchers); or
#   (c) the line carries a `model=` / `model:` kwarg AND a JUDGE_PIN_CALL_TOKEN
#       appears within +/- JUDGE_PIN_CONTEXT_WINDOW non-comment lines;
#   (d) the line is a forbidden-pin literal preceded (within
#       JUDGE_PIN_SPLIT_ARGV_LOOKAHEAD non-blank lines) by a BARE `--judge-model`
#       flag token on its own list-literal line (split-argv, #765 round 2);
#   (e) [.sh only, file-scope two-pass] the line ASSIGNS a shell var to a
#       forbidden-pin value AND that var is later consumed by a `--judge-model`
#       flag (shell-var indirection, #765 round 2); or
#   (f) [judge-context files only] the line matches JUDGE_PIN_DEFAULT_MODEL_VAR_RE
#       (a `DEFAULT_MODEL` / `MODEL_DEFAULT` / `JUDGE_DEFAULT` judge-script
#       constant whose name lacks JUDGE_MODEL, #765 round 2).
# A pure code-comment line (lstrip startswith '#') is NEVER a hit, and a bare
# forbidden substring inside a descriptive string with no judge-named
# assignment/flag/judge-`model=` on the line is NEVER a hit (the prose-mention
# guard — issue552_gate_decision.py:83, issue467_figures.py:176,
# gen_data_appendix.py:212, issue623_behavioral_dv.py docstring, the SDF
# `messages.create(model=...)` document-generation calls).
# The canonical pin `claude-sonnet-4-5-20250929` contains NONE of the forbidden
# substrings (it is `claude-sonnet-4-5-...`, NOT `claude-3-5-sonnet`), so it
# never matches — asserted in a test.
JUDGE_PIN_FORBIDDEN_SUBSTRINGS: tuple[str, ...] = (
    "claude-haiku-",
    "gpt-4o",
    "gpt-4-",
    "gpt-5",
    "claude-opus-",
    "claude-3-5-sonnet",
)
JUDGE_PIN_CANONICAL = "claude-sonnet-4-5-20250929"  # the ALLOWED judge pin
# (a) RHS of a judge-named assignment/key: a token CONTAINING `JUDGE_MODEL`
#     (e.g. DEFAULT_GPT4O_JUDGE_MODEL, SYCO_JUDGE_MODEL), or the bare
#     `judge_model` / `JUDGE_MODEL`, immediately before `:` or `=`.
JUDGE_PIN_VAR_RE = re.compile(
    r"\b([A-Za-z_][A-Za-z0-9_]*JUDGE_MODEL[A-Za-z0-9_]*|judge_model|JUDGE_MODEL)\b\s*[:=]"
)
# (b) CLI-flag / shell judge-arg form (covers .py argparse defaults AND .sh):
#   - `--judge-model` as a bare flag token (the flag name + a forbidden pin on
#     the same line is a judge pin regardless of the separator — the argparse
#     `add_argument("--judge-model", default="gpt-4o...")` form and the shell
#     `--judge-model gpt-4o...` form both match here);
#   - the `judge_model=` / `JUDGE_MODEL=` shell/kwarg form.
JUDGE_PIN_FLAG_RE = re.compile(r"(--judge-model\b|judge_model=|JUDGE_MODEL=)")
# (c) judge-call tokens for the model=-kwarg-in-window arm:
JUDGE_PIN_CALL_TOKENS: tuple[str, ...] = (
    "as judge",
    "judge_completions",
    "judge=",
    "JUDGE_MODEL",
    "judge_model",
    "SYCO_JUDGE_MODEL",
)
JUDGE_PIN_MODEL_KWARG_RE = re.compile(r"\bmodel\s*[:=]")
JUDGE_PIN_CONTEXT_WINDOW = 3
# (d) split-argv recognition (#765 round 2, concern judge-pin-detector-split-argv):
#   a Python list-literal `--judge-model` entry on its own line, e.g.
#       args = ["--judge-model", "claude-haiku-4-5-20251001"]  # single-line — arm (b)
#   or split across lines (the run_evals_190.py:52-53 shape):
#       "--judge-model",
#       "claude-haiku-4-5-20251001",
#   When the line is the BARE `--judge-model` flag token (no forbidden pin on it,
#   so arm (b) misses), the NEXT non-blank line carrying a forbidden substring is
#   the hit. JUDGE_PIN_BARE_FLAG_RE matches a line whose only non-trivial content
#   is the `--judge-model` flag token (stripped of surrounding quotes / comma /
#   whitespace) — i.e. the flag and its value live on separate argv lines.
JUDGE_PIN_BARE_FLAG_RE = re.compile(r"""^[\s"']*--judge-model[\s"',]*$""")
# Forward look-ahead window (in non-blank lines) for the split-argv literal.
JUDGE_PIN_SPLIT_ARGV_LOOKAHEAD = 2
# (e) shell-variable indirection (#765 round 2, concern
#   judge-pin-detector-shell-var-indirection): a .sh file that assigns
#       JUDGE=gpt-4o-2024-08-06        # var name need NOT contain JUDGE_MODEL
#   then later passes `--judge-model "${JUDGE}"` / `$JUDGE` / `"${JUDGE:-...}"`.
#   The ASSIGNMENT line carries the forbidden pin but no judge-named var / flag,
#   and the `--judge-model` reference line passes a var, not a literal — so both
#   arms (a)/(b) miss. Two-pass per .sh file: pass 1 collects every
#   `VAR=<value-with-forbidden-substring>`; pass 2 detects a `--judge-model`
#   reference to one of those vars; the ASSIGNMENT line is flagged.
#   A shell var assignment: `VAR=...` at line start (after optional `export`/
#   leading whitespace), capturing the var NAME.
JUDGE_PIN_SH_ASSIGN_RE = re.compile(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)=")


# A `--judge-model` flag that consumes a shell variable (not a literal): the
# var name is rendered into the {var} placeholder per file. Matches `$VAR`,
# `${VAR}`, `"${VAR}"`, `${VAR:-default}` (whitespace- or =-separated flag).
def _judge_pin_sh_var_ref_re(var: str) -> re.Pattern[str]:
    """Compile a regex matching `--judge-model` consuming shell var ``var``
    (``$VAR`` / ``${VAR}`` / ``${VAR:-...}``, optionally quoted)."""
    v = re.escape(var)
    return re.compile(rf"--judge-model[\s=]+[\"']?\$\{{?{v}\b")


# (f) judge-script DEFAULT_MODEL constant (#765 round 2, concern
#   judge-pin-detector-default-model-constant): a judge-script module-level
#   constant whose name does NOT contain JUDGE_MODEL, e.g.
#       DEFAULT_MODEL = "claude-haiku-4-5-20251001"   (judge_with_claude.py:31)
#   misses arm (a) (the var name lacks JUDGE_MODEL). Expanded ONLY when the file
#   is JUDGE-CONTEXT (see _file_is_judge_context) — a NARROW broadening so a
#   non-judge module's DEFAULT_MODEL cost-table constant does not false-fire.
# A constant name containing BOTH `MODEL` and `DEFAULT` (either order — so
# `DEFAULT_MODEL`, `MODEL_DEFAULT`, `GPT4O_DEFAULT_MODEL` all match; the `\w*`
# prefix is zero-width so a bare leading `DEFAULT_...` / `MODEL_...` matches) or
# the bare `JUDGE_DEFAULT`, immediately before `:` or `=`.
JUDGE_PIN_DEFAULT_MODEL_VAR_RE = re.compile(
    r"\b(\w*MODEL\w*DEFAULT\w*|\w*DEFAULT\w*MODEL\w*|JUDGE_DEFAULT)\b\s*[:=]"
)
# Judge-context signals (file is plausibly a judge script): filename contains
# `judge`, the module docstring mentions judging, the file imports a judge
# client / a `*judge*` module, or it defines a `judge_*`-named function.
JUDGE_PIN_CONTEXT_FILENAME_RE = re.compile(r"judge", re.IGNORECASE)
JUDGE_PIN_CONTEXT_BODY_RE = re.compile(
    r"BatchJudgeClient"  # the project batch-judge client
    r"|\bimport\b[^\n]*judge"  # imports a *judge* module
    r"|\bfrom\b[^\n]*judge\b[^\n]*\bimport\b"  # from ...judge... import ...
    r"|\bdef\s+judge_\w+"  # defines a judge_* function
    r"|\bas\s+judge\b"  # docstring/comment "as judge"
)
# Files whose every line is exempt (the rule/doc that names the pin literally,
# the SDK model-id registries / cost tables — NOT judge sites — and the linter's
# own known-model tuple + this check's own test fixtures naming forbidden pins
# inside strings). Matched by EXACT repo-root-relative POSIX path.
JUDGE_PIN_FILE_ALLOWLIST: frozenset[str] = frozenset(
    {
        # the rule documents the pin literally / the global doc names it:
        ".claude/rules/llm-judging.md",
        "CLAUDE.md",
        # the linter's own known-model tuple + this block:
        "scripts/workflow_lint.py",
        # this check's test fixtures name forbidden pins inside strings:
        "tests/test_workflow_lint_judge_model_check.py",
        # SDK model-id registries / cost tables — NOT judge sites:
        "src/explore_persona_space/llm/openai_client.py",
        "src/explore_persona_space/llm/anthropic_client.py",
    }
)
# Grandfathered legitimate NON-Sonnet judge pins — .py — repo-root-relative
# POSIX paths, annotated inline with the bucket (calibration anchor /
# translation-judge exemption / stale-grandfathered migrate). Migrating the
# stale ones to Sonnet is a named follow-up (NOT this task's scope); a NEW
# legitimate pin must be added here with a `reason` when it lands, or the
# no-flags default run FAILs (the test_live_trees_pass invariant).
JUDGE_PIN_LEGACY_ALLOWLIST: frozenset[str] = frozenset(
    {
        # --- permanent calibration anchors (Betley gpt-4o; replication-fidelity) ---
        # gpt-4o Betley κ-calibration anchor:
        "scripts/issue404_outcome_eval.py",
        # gpt-4o Betley broad-EM judge diagnostic:
        "scripts/issue545_betley_diag.py",
        # gpt-4o B1 broad-EM anchor #458/#468 + haiku calibration:
        "src/explore_persona_space/experiments/issue503/judges.py",
        # honors the gpt-4o B1 judge by family:
        "src/explore_persona_space/experiments/issue503/cross_eval.py",
        # Betley dual judge + haiku calibration via the #503 rig:
        "src/explore_persona_space/experiments/behavior_testbed_545/judges_545.py",
        # Betley dual judge via the #503 rig:
        "src/explore_persona_space/experiments/behavior_testbed_545/eval_battery.py",
        # tests the gpt-4o calibration-anchor dispatch routing (#404):
        "tests/test_issue404_judge_dispatch.py",
        # tests the gpt-4o Betley broad-EM sentinel handling (#545):
        "tests/test_issue545_betley_sentinel.py",
        # --- permanent translation-judge exemptions (non-behavior-expression DV) ---
        # translation-faithfulness judge (Haiku); not a #765 behavior DV:
        "scripts/validate_translation.py",
        # Italian translation-faithfulness judge (Haiku):
        "scripts/validate_italian_translation.py",
        # --- stale-grandfathered, migrate-to-Sonnet (follow-up §2) ---
        # #389 fact-gating re-judge, legacy Haiku:
        "scripts/rejudge_issue_389_c_strict.py",
        # #389 driver, legacy Haiku:
        "scripts/run_experiment_389.py",
        # #444 driver, legacy Haiku:
        "scripts/run_experiment_444.py",
        # #444 5-way reanalysis, legacy Haiku:
        "scripts/reanalyze_issue444_5way.py",
        # #190 eval driver, legacy Haiku:
        "scripts/run_evals_190.py",
        # assistant-axis role-adherence judge, legacy Haiku:
        "scripts/judge_with_claude.py",
        # #642 realized #411/#518 legacy Haiku judge id:
        "scripts/issue_642/i642_common.py",
        # #411/#591 legacy sycophancy Haiku judge:
        "src/explore_persona_space/experiments/sycophancy_onpolicy_612/__init__.py",
        # #612 sycophancy judge default, legacy Haiku:
        "src/explore_persona_space/experiments/sycophancy_onpolicy_612/judge.py",
        # #650 SYCO_JUDGE_MODEL legacy Haiku:
        "src/explore_persona_space/experiments/issue_650/__init__.py",
    }
)
# Grandfathered legitimate NON-Sonnet judge launchers — .sh — all permanent
# Betley gpt-4o calibration anchors (they pin --judge-model DIRECTLY in shell,
# so a .py-only gate would miss them — the walk includes .sh).
JUDGE_PIN_LEGACY_ALLOWLIST_SH: frozenset[str] = frozenset(
    {
        # Betley deconfound gpt-4o (same judge+rubric as #404):
        "scripts/run_issue452_deconfound.sh",
        # #458 Betley broad-EM sweep gpt-4o:
        "scripts/run_issue458_sweep.sh",
        # #552 canonical 8x100 EM gate gpt-4o:
        "scripts/run_issue552_sweep.sh",
        # #552 resume launcher gpt-4o:
        "scripts/run_issue552_resume.sh",
    }
)
JUDGE_PIN_WAIVER_RE = re.compile(r"#\s*noqa:\s*judge-model-pin\b")
JUDGE_PIN_FILE_WAIVER_RE = re.compile(r"#\s*epm-allow-judge-model-pin\b")


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


def check_pipe_python(*, scripts_dir: Path | None = None) -> list[str]:
    """Walk every ``*.sh`` under ``scripts/`` and FAIL on any shell pipe
    whose CONSUMER is a bare ``python``/``python3[.N]`` interpreter
    invoked with ``-c`` or ``-m`` (``... | python -c "..."``).

    Rationale: this VM has NO ``python`` on PATH — only ``python3`` and
    the project's ``uv run python`` — so a bare ``| python -c`` /
    ``| python -m`` pipe dies at runtime with ``python: command not
    found`` (exit 127). CLAUDE.md § Task Workflow API carries the rule
    verbatim ("prefix EVERY python invocation with ``uv run python``,
    INCLUDING the consumer side of a pipe"), but it lived only as prose
    and was violated ~41x across 4+ sessions on 2026-06-29 (#753). The
    fix is always the same: pipe into ``uv run python`` instead. Backslash
    continuations are merged into one logical line (both #753 offenders
    were backslash-continued ``cat ... \\`` newline ``| python3 -c``);
    only ``#``-comment lines are skipped. ``echo ... | python -c`` is a
    REAL producer pipe (echo's stdout is consumed by bare ``python``) and
    IS flagged — the earlier blanket ``echo ``-skip silently missed this
    must-catch shape (#753 round 2 / F1). To document the bad pattern,
    put it in a ``#``-comment, not an ``echo`` string. See the regex block
    above for the full flagged/not-flagged matrix.

    ``scripts_dir`` is an override hook for unit tests; production
    callers pass None and the function walks the canonical
    ``<repo_root>/scripts`` tree. Bundled into the no-flags default run
    (same policy as ``check_heredoc_dotenv`` / ``check_dispatcher_cvd_pin``).
    """
    root = scripts_dir if scripts_dir is not None else _REPO_ROOT / "scripts"
    if not root.exists():
        return []
    errors: list[str] = []
    for sh in sorted(root.rglob("*.sh")):
        if not sh.is_file():
            continue
        lines = sh.read_text(encoding="utf-8").splitlines()
        for first, _last, logical in _iter_logical_shell_lines(lines):
            stripped = logical.strip()
            # Only `#`-comment lines are documentation and skipped. `echo `
            # lines are NOT skipped: `echo '{}' | python -c "..."` is a REAL
            # producer pipe whose consumer is bare `python` — exactly the
            # exit-127 class this check exists to close (#753 round 2 / F1).
            # The earlier blanket `echo `-skip silently missed it. A script
            # that genuinely needs to DOCUMENT the bad pattern must do so in a
            # `#`-comment, not an `echo` string.
            if stripped.startswith("#"):
                continue
            if not PIPE_PYTHON_RE.search(logical):
                continue
            errors.append(
                f"{sh}:{first + 1}: bare `| python -c/-m` pipe consumer. "
                f"This VM has no `python` on PATH — `python: command not "
                f"found` (exit 127). Pipe into `uv run python` instead "
                f'(`... | uv run python -c "..."`). See CLAUDE.md '
                f"§ Task Workflow API (#753)."
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

    errors.extend(_check_failure_lesson_field_contract(workflow))
    return errors


def _check_failure_lesson_field_contract(workflow: WorkflowYaml) -> list[str]:
    """#712 §4f: kind-scoped field-contract pin for ``epm:failure-lesson``.

    The ``root_cause_confirmed`` / ``supersedes`` fields are emitter-produced +
    orchestrator-branched semantic fields living in the marker's free-text
    ``fields:`` description (no Pydantic attribute) — so a future edit that
    silently drops or renames either would not be caught by the schema model.
    This narrow, additive assertion makes the field-add no longer free-floating
    schema drift: the next field-add on this kind inherits the check. Kind-scoped
    (only ``epm:failure-lesson``) and runs whenever that marker is declared in
    the supplied workflow (so the fixture FAIL leg + the real PASS leg both
    exercise it). Extracted from :func:`check_marker_registry` to keep that
    function under the C901 complexity cap.
    """
    errors: list[str] = []
    for marker in workflow.markers:
        if marker.kind != "epm:failure-lesson":
            continue
        fields_text = marker.fields or ""
        for token in ("root_cause_confirmed", "supersedes"):
            if token not in fields_text:
                errors.append(
                    f"workflow.yaml § markers: epm:failure-lesson `fields:` is "
                    f"missing the required field token '{token}' (#712 §4f). The "
                    f"root_cause_confirmed/supersedes field contract must stay "
                    f"declared in the marker registry — re-add the token to the "
                    f"`fields:` string."
                )
        if "root_cause_confirmed=yes" not in (marker.when or ""):
            errors.append(
                "workflow.yaml § markers: epm:failure-lesson `when:` is missing "
                "the required 'root_cause_confirmed=yes' firing condition (#712 "
                "§4f). The root-cause-confirmed firing trigger must stay declared "
                "in the marker registry — re-add it to the `when:` string."
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


def _split_agent_frontmatter(text: str) -> tuple[list[str] | None, str, int]:
    """Split an agent file into ``(frontmatter_lines, body, body_line_offset)``.

    The frontmatter block is the lines between the FIRST two ``---`` lines
    (the file must start with ``---``). ``body_line_offset`` is the 1-based
    file line number of the first body line, so body-relative match
    positions can be reported as real file:line locations. Returns
    ``(None, text, 1)`` when the file has no parseable frontmatter block —
    callers report that as a missing-declaration failure.
    """
    lines = text.split("\n")
    if not lines or lines[0].strip() != "---":
        return None, text, 1
    for i, line in enumerate(lines[1:], 1):
        if line.strip() == "---":
            fm_lines = lines[1:i]
            body = "\n".join(lines[i + 1 :])
            return fm_lines, body, i + 2
    return None, text, 1


def _parse_agent_tool_decls(
    fm_lines: list[str],
) -> tuple[list[str] | None, list[str] | None]:
    """Parse ``tools:`` / ``disallowedTools:`` out of a frontmatter block.

    Both repo syntaxes are honored (mirroring the harness): the inline
    comma-separated scalar (``tools: Read, Grep, Glob, Bash``) and the YAML
    list form (``tools:`` followed by ``  - Read`` items). Only TOP-LEVEL
    keys (column 0) are considered, so folded ``description: >`` blocks
    never introduce fake keys (their continuation lines are indented).
    Returns ``(tools, disallowed)`` — each ``None`` when the key is absent,
    else the (possibly empty) list of declared names.
    """
    tools: list[str] | None = None
    disallowed: list[str] | None = None
    for i, line in enumerate(fm_lines):
        match = re.match(r"^(tools|disallowedTools):\s*(.*)$", line)
        if match is None:
            continue
        key, rest = match.group(1), match.group(2).strip()
        if rest:
            values = [v.strip() for v in rest.split(",") if v.strip()]
        else:
            values = []
            for cont in fm_lines[i + 1 :]:
                item = re.match(r"^\s+-\s+(\S.*?)\s*$", cont)
                if item is None:
                    break
                values.append(item.group(1))
        if key == "tools":
            tools = values
        else:
            disallowed = values
    return tools, disallowed


def _extract_agent_body_tool_mentions(body: str) -> dict[str, int]:
    """Extract every tool the spec BODY mentions, per the #840 §4.3 widened
    extractor vocabulary (see the :data:`AGENT_TOOLS_MCP_TOKEN_RE` comment
    block). Returns ``{token: first_body_line_index}`` (0-based line index
    within the body; callers add the frontmatter offset)."""
    mentions: dict[str, int] = {}

    def record(token: str, pos: int) -> None:
        lineno = body.count("\n", 0, pos)
        if token not in mentions or lineno < mentions[token]:
            mentions[token] = lineno

    for m in AGENT_TOOLS_MCP_TOKEN_RE.finditer(body):
        record(m.group(0), m.start())
    for m in AGENT_TOOLS_BUILTIN_RE.finditer(body):
        record(m.group(1), m.start())
    for m in AGENT_TOOLS_AGENT_PHRASE_RE.finditer(body):
        record("Agent", m.start())
    for m in AGENT_TOOLS_SKILL_PHRASE_RE.finditer(body):
        record("Skill", m.start())
    for alias_re, token in AGENT_TOOLS_MCP_ALIASES:
        for m in alias_re.finditer(body):
            record(token, m.start())
    return mentions


def _mcp_server_of(token: str) -> str | None:
    """Return the ``<server>`` segment of an ``mcp__...`` token (full-tool,
    server-level, or ``__*``-wildcard form), or None for a non-MCP name.
    Server names never contain a double underscore, so splitting on ``__``
    is unambiguous (``mcp__plugin_context7_context7`` has only single
    underscores inside the server segment)."""
    parts = token.split("__")
    if len(parts) < 2 or parts[0] != "mcp" or not parts[1]:
        return None
    return parts[1]


def _agent_tool_mention_covered(token: str, declared: list[str]) -> bool:
    """True iff a body-mentioned ``token`` is covered by the ``tools:``
    allowlist ``declared``: exact match; the token's server-level
    ``mcp__<server>`` (or ``mcp__<server>__*``) form is declared; or — for a
    SERVER-LEVEL mention (a prose alias like "arXiv MCP") — the declaration
    names at least one tool from that server (e.g. related-work-finder
    declares 4 ``mcp__arxiv__*`` tools; its "arXiv MCP" prose is covered)."""
    if token in declared:
        return True
    if not token.startswith("mcp__"):
        return False
    parts = token.split("__")
    server_form = "__".join(parts[:2])
    if server_form in declared or f"{server_form}__*" in declared:
        return True
    if len(parts) == 2:
        return any(d.startswith(server_form + "__") for d in declared)
    return False


def _agent_tool_mention_denied(token: str, denied: list[str]) -> bool:
    """True iff a body-mentioned ``token`` is removed by the
    ``disallowedTools:`` denylist ``denied`` (exact name, its server-level
    form, the ``__*`` wildcard form, or the all-MCP ``mcp__*`` wildcard)."""
    if token in denied:
        return True
    if token.startswith("mcp__"):
        if "mcp__*" in denied:
            return True
        parts = token.split("__")
        server_form = "__".join(parts[:2])
        if server_form in denied or f"{server_form}__*" in denied:
            return True
    return False


def check_agent_tools(*, roots: list[Path] | None = None) -> list[str]:
    """Walk ``.claude/agents/*.md`` and enforce the explicit-tool-surface
    invariant (task #840; incident #778 — an agent file with no ``tools:``
    key inherits the parent session's full MCP tool-schema payload, ~168K
    static first-turn tokens at the measured worst case).

    Per file:

    1. **Declaration required** — FAIL if the frontmatter has neither a
       ``tools:`` allowlist nor a ``disallowedTools:`` denylist (or has no
       parseable frontmatter at all).
    2. **Mentioned ⊆ declared** (allowlist files) — FAIL if the spec body
       mentions a tool (per :func:`_extract_agent_body_tool_mentions`) that
       the allowlist does not cover (per
       :func:`_agent_tool_mention_covered`), unless the ``(filename,
       token)`` pair is waived in :data:`AGENT_TOOLS_MENTION_EXCEPTIONS`
       with an inline reason.
    2b. **Declared-name validity** — FAIL if any DECLARED ``mcp__...`` token
       (in either key) names a server outside :data:`KNOWN_MCP_SERVERS` —
       the harness silently ignores unknown names, so a typo strips a
       capability with no error anywhere.
    3. **Denylist consistency** (denylist files) — FAIL if a body-mentioned
       token is denied by the denylist. Denylist-only files skip check 2
       (they inherit everything not denied).

    ``roots`` is the unit-test override hook (same contract as
    :func:`check_agent_model_pins`); production callers pass None and the
    canonical agent tree under :data:`_REPO_ROOT` is walked.
    """
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
        fm_lines, body, body_offset = _split_agent_frontmatter(text)
        if fm_lines is None:
            errors.append(
                f"{path}: no parseable YAML frontmatter block (file must start "
                f"with '---' and close the block with a second '---'), so no "
                f"tools:/disallowedTools: declaration exists. Every agent file "
                f"must declare its tool surface (task #840; incident #778)."
            )
            continue
        tools, disallowed = _parse_agent_tool_decls(fm_lines)
        # Check 1 — declaration required.
        if tools is None and disallowed is None:
            errors.append(
                f"{path}: frontmatter declares neither 'tools:' nor "
                f"'disallowedTools:'. An undeclared agent inherits the parent "
                f"session's FULL tool inventory including every MCP server's "
                f"schemas (~168K static first-turn tokens at the #778 worst "
                f"case). Add a 'tools:' allowlist (see the restricted agents "
                f"for the house YAML-list style) or, for a broad main-session "
                f"persona, a 'disallowedTools:' denylist (research-pm.md "
                f"precedent). Task #840."
            )
            continue
        errors.extend(_agent_tools_decl_validity_errors(path, tools, disallowed))
        mentions = _extract_agent_body_tool_mentions(body)
        errors.extend(_agent_tools_mention_errors(path, tools, disallowed, mentions, body_offset))
    return errors


def _agent_tools_decl_validity_errors(
    path: Path, tools: list[str] | None, disallowed: list[str] | None
) -> list[str]:
    """Check 2b — every DECLARED ``mcp__...`` token (either key) must name a
    server in :data:`KNOWN_MCP_SERVERS`; the ``mcp__*`` all-MCP denylist
    wildcard is valid as-is."""
    errors: list[str] = []
    for decl_list, key in ((tools, "tools"), (disallowed, "disallowedTools")):
        if decl_list is None:
            continue
        for token in decl_list:
            if not token.startswith("mcp__"):
                continue
            server = _mcp_server_of(token)
            if server is not None and server.endswith("*"):
                # `mcp__*` all-MCP wildcard (denylist form) — valid.
                continue
            if server is None or server not in KNOWN_MCP_SERVERS:
                known = ", ".join(sorted(KNOWN_MCP_SERVERS))
                errors.append(
                    f"{path}: '{key}:' declares '{token}' whose server "
                    f"segment is not a known MCP server. The harness "
                    f"silently ignores unknown tool names, so a typo here "
                    f"strips the capability with no error at spawn. Known "
                    f"servers: {known}. If a NEW MCP server was just "
                    f"registered, add it to KNOWN_MCP_SERVERS in "
                    f"scripts/workflow_lint.py."
                )
    return errors


def _agent_tools_mention_errors(
    path: Path,
    tools: list[str] | None,
    disallowed: list[str] | None,
    mentions: dict[str, int],
    body_offset: int,
) -> list[str]:
    """Checks 2 + 3 — a body-mentioned token must be covered by the
    ``tools:`` allowlist (when present) and must not be removed by the
    ``disallowedTools:`` denylist (when present), modulo the
    :data:`AGENT_TOOLS_MENTION_EXCEPTIONS` waivers."""
    errors: list[str] = []
    for token, body_lineno in sorted(mentions.items()):
        if (path.name, token) in AGENT_TOOLS_MENTION_EXCEPTIONS:
            continue
        lineno = body_offset + body_lineno
        if tools is not None and not _agent_tool_mention_covered(token, tools):
            errors.append(
                f"{path}:{lineno}: spec body mentions tool '{token}' "
                f"but the frontmatter 'tools:' allowlist does not "
                f"cover it — the agent is instructed to use a tool it "
                f"cannot call. Either add '{token}' (or its "
                f"'mcp__<server>' form) to the allowlist, or — if the "
                f"mention is descriptive-not-instructive (documents "
                f"another actor's tool use) — add "
                f"('{path.name}', '{token}') to "
                f"AGENT_TOOLS_MENTION_EXCEPTIONS with an inline "
                f"reason. Task #840."
            )
        if disallowed is not None and _agent_tool_mention_denied(token, disallowed):
            errors.append(
                f"{path}:{lineno}: spec body mentions tool '{token}' "
                f"but the frontmatter 'disallowedTools:' denylist "
                f"removes it — the agent is instructed to use a tool "
                f"it cannot call. Drop the deny entry or fix the spec "
                f"body (body edits are a separate change per the #840 "
                f"frontmatter-only scope)."
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


def _jsonl_splitlines_waiver_present(lines: list[str], call_lineno: int) -> bool:
    """Return True iff a ``# JSONL_SPLITLINES_EXEMPT: <reason>`` waiver
    (reason ≥ :data:`JSONL_SPLITLINES_WAIVER_MIN_REASON_CHARS` chars) is on
    the call's first physical line (``call_lineno``, 1-based) or the
    immediately preceding non-blank line. Same convention as
    :func:`_upload_as_file_waiver_present`."""
    idx = call_lineno - 1  # to 0-based
    if 0 <= idx < len(lines):
        m = JSONL_SPLITLINES_WAIVER_RE.search(lines[idx])
        if m and len(m.group(1).strip()) >= JSONL_SPLITLINES_WAIVER_MIN_REASON_CHARS:
            return True
    back = idx - 1
    while back >= 0 and lines[back].strip() == "":
        back -= 1
    if back >= 0:
        m = JSONL_SPLITLINES_WAIVER_RE.search(lines[back])
        if m and len(m.group(1).strip()) >= JSONL_SPLITLINES_WAIVER_MIN_REASON_CHARS:
            return True
    return False


def _chain_has_read_text(expr: ast.expr) -> bool:
    """True iff the receiver expression chain contains a ``read_text`` call."""
    for node in ast.walk(expr):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "read_text"
        ):
            return True
    return False


def _chain_base_name(expr: ast.expr) -> str | None:
    """Leftmost ``ast.Name`` id of an attribute/call/subscript chain, or None."""
    while True:
        if isinstance(expr, ast.Call):
            expr = expr.func
        elif isinstance(expr, ast.Attribute | ast.Subscript):
            expr = expr.value
        elif isinstance(expr, ast.Name):
            return expr.id
        else:
            return None


def _jsonl_fn_scoped_splitlines_ids(tree: ast.AST) -> set[int]:
    """Signal (c) pre-pass: ``id()``s of every ``.splitlines()`` call node
    enclosed by a ``jsonl``-named function (the ``_iter_jsonl`` shape)."""
    fn_scoped: set[int] = set()
    for fn in ast.walk(tree):
        if isinstance(fn, ast.FunctionDef | ast.AsyncFunctionDef) and JSONL_NAME_TOKEN_RE.search(
            fn.name
        ):
            for sub in ast.walk(fn):
                if (
                    isinstance(sub, ast.Call)
                    and isinstance(sub.func, ast.Attribute)
                    and sub.func.attr == "splitlines"
                ):
                    fn_scoped.add(id(sub))
    return fn_scoped


def _jsonl_splitlines_signal(node: ast.Call, text: str, fn_scoped: set[int]) -> str | None:
    """Classify one ``.splitlines()`` call against the four #950 signals.

    Returns a human-readable signal label when the call reads JSONL content
    (see :func:`check_jsonl_splitlines` for the signal definitions), else
    None. A per-node ``ast.get_source_segment(...) is None`` only makes the
    segment-dependent predicates (a)/(d-literal) non-matching for the node.
    """
    receiver = node.func.value  # type: ignore[attr-defined]
    segment = ast.get_source_segment(text, receiver)
    has_read = _chain_has_read_text(receiver)
    base = _chain_base_name(receiver)
    if has_read and segment is not None and JSONL_NAME_TOKEN_RE.search(segment):
        return "jsonl-named read_text chain"
    if isinstance(receiver, ast.Name) and JSONL_NAME_TOKEN_RE.search(receiver.id):
        return f"jsonl-named receiver ('{receiver.id}')"
    if id(node) in fn_scoped:
        return "call inside a jsonl-named function"
    if has_read and (
        (base is not None and JSONL_EVENTS_PATH_NAME_RE.match(base))
        or (
            segment is not None
            and any(lit in segment for lit in ("events.jsonl", "comments.jsonl", "concerns.jsonl"))
        )
    ):
        return "events/comments/concerns-path read_text chain"
    return None


def check_jsonl_splitlines(*, scan_roots: tuple[Path, ...] | None = None) -> list[str]:
    """AST-walk ``scripts/**/*.py`` + ``src/explore_persona_space/**/*.py``
    and FAIL any ``.splitlines()`` call that reads JSONL content (#950).

    Rationale: ``json.dumps(..., ensure_ascii=False)`` — the project's
    events/comments writer and most JSONL emitters — leaves raw U+2028 LINE
    SEPARATOR, U+2029 PARAGRAPH SEPARATOR, and NEL U+0085 inside JSON strings
    (controls < 0x20 are still escaped), and ``str.splitlines()`` splits on
    ALL Unicode line boundaries. A perfectly valid ``\\n``-terminated JSONL
    file read via ``splitlines()`` therefore shreds any record whose text
    carries one of those characters: a hard ``JSONDecodeError`` on strict
    readers, SILENT record loss on tolerant skip-malformed readers, and an
    inflated row count on ``len(read_text().splitlines())`` asserts.
    Real-user corpora (lmsys-chat-1m, WildChat) contain them routinely and an
    ASCII-fixture smoke can never catch it (incident #825 run-1d: 2000 valid
    records → 2019 fragments, ~55 min of GPU extraction lost; eight live
    workflow-surface reader sites across seven files fixed with #950). The
    fix is ``split("\\n")`` or text-mode file iteration (universal newlines
    only).

    Detection — flag an ``ast.Call`` whose func is
    ``ast.Attribute(attr="splitlines")`` when ANY of:

    * **(a) chained-read signal:** the receiver chain contains a
      ``read_text`` call AND the receiver's source segment mentions
      ``jsonl`` case-insensitively (``jsonl_path.read_text().splitlines()``,
      ``(d / "pool.jsonl").read_text().splitlines()``).
    * **(b) receiver-name signal:** the receiver is a bare ``ast.Name``
      matching ``/jsonl/i`` (``jsonl_text.splitlines()``).
    * **(c) function-name signal:** the call sits inside a
      ``FunctionDef``/``AsyncFunctionDef`` whose name matches ``/jsonl/i``
      (the ``_iter_jsonl`` shape — receiver read on a separate line).
    * **(d) events/concerns-path signal:** the receiver chain contains a
      ``read_text`` call AND (its base ``ast.Name`` matches
      ``/^(ev(ents)?|concerns)_path$/i`` OR the segment names the literal
      ``events.jsonl``/``comments.jsonl``/``concerns.jsonl``) — the exact
      shapes of the #950 sibling workflow readers (the round-1
      ``events.jsonl`` siblings + the round-2 ``verify_task_body.py``
      check-14 ``concerns.jsonl`` reader), which evade (a)-(c).

    Deliberate false negatives (accepted; the gotchas.md entry + code review
    carry them): dataflow through a non-jsonl, non-events-named variable
    (``out_path = ... / "x.jsonl"`` … ``out_path.read_text().splitlines()``)
    and python-in-shell heredocs (``.sh`` files are not AST-scannable).

    Unparseable files: a ``SyntaxError`` (does not parse) or
    ``UnicodeDecodeError`` (non-UTF-8) file is SKIPPED without failing the
    check — syntax validity is ruff/pytest's job — but a one-line notice is
    printed to stderr so the skip is never silent (strengthens the silent
    ``--check-upload-as-file`` precedent). A per-node
    ``ast.get_source_segment(...) is None`` only makes the segment-dependent
    predicates non-matching for that node; no file skip.

    Waiver: ``# JSONL_SPLITLINES_EXEMPT: <reason>`` (reason ≥
    :data:`JSONL_SPLITLINES_WAIVER_MIN_REASON_CHARS` chars) on the call's
    first physical line or the immediately preceding non-blank line.
    Grandfather: :data:`JSONL_SPLITLINES_LEGACY_ALLOWLIST` (file-level,
    frozen experiment scripts only — NEVER a workflow-surface file).

    ``scan_roots`` is a unit-test override hook; production callers pass None
    and the function walks ``<repo_root>/scripts`` +
    ``<repo_root>/src/explore_persona_space`` (NOT ``tests/`` /
    ``external/`` / ``archive/``). Bundled into the no-flags default run.
    """
    roots = (
        scan_roots
        if scan_roots is not None
        else (_REPO_ROOT / "scripts", _REPO_ROOT / "src" / "explore_persona_space")
    )
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
                rel = py.name
            if rel in JSONL_SPLITLINES_LEGACY_ALLOWLIST:
                continue
            try:
                text = py.read_text(encoding="utf-8")
                tree = ast.parse(text, filename=str(py))
            except (SyntaxError, UnicodeDecodeError) as exc:
                # Skip-with-report: never silent, never fatal (syntax validity
                # is ruff/pytest's enforcement job, not this lint's).
                sys.stderr.write(
                    f"workflow_lint: note: --check-jsonl-splitlines skipped "
                    f"unparseable {rel} ({type(exc).__name__})\n"
                )
                continue
            lines = text.split("\n")
            fn_scoped = _jsonl_fn_scoped_splitlines_ids(tree)
            for node in ast.walk(tree):
                if not (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "splitlines"
                ):
                    continue
                signal = _jsonl_splitlines_signal(node, text, fn_scoped)
                if signal is None:
                    continue
                if _jsonl_splitlines_waiver_present(lines, node.lineno):
                    continue
                errors.append(
                    f"{py}:{node.lineno}: jsonl-splitlines: .splitlines() on JSONL "
                    f"content ({signal}). str.splitlines() splits on raw "
                    f"U+2028/U+2029/NEL inside ensure_ascii=False JSON strings and "
                    f"shreds valid records — silent drop on tolerant readers, "
                    f"JSONDecodeError on strict ones, inflated row counts on "
                    f"len() asserts (#825/#950; .claude/rules/gotchas.md). Read/"
                    f'count JSONL via text-mode file iteration or split("\\n") + '
                    f"an `if line.strip()` guard, or waive a genuinely-safe site "
                    f"with '# JSONL_SPLITLINES_EXEMPT: <reason>' (reason ≥ "
                    f"{JSONL_SPLITLINES_WAIVER_MIN_REASON_CHARS} chars)."
                )
    return errors


def _dotenv_lint_waiver_present(lines: list[str], import_lineno: int) -> bool:
    """Return True iff a ``# DOTENV_LINT_EXEMPT: <reason>`` waiver (reason ≥
    :data:`DOTENV_LINT_WAIVER_MIN_REASON_CHARS` chars) is on the bare-dotenv
    import line (``import_lineno``, 1-based) or the immediately preceding
    non-blank line. Same convention as :func:`_upload_as_file_waiver_present`."""
    idx = import_lineno - 1  # to 0-based
    if 0 <= idx < len(lines):
        m = DOTENV_LINT_WAIVER_RE.search(lines[idx])
        if m and len(m.group(1).strip()) >= DOTENV_LINT_WAIVER_MIN_REASON_CHARS:
            return True
    back = idx - 1
    while back >= 0 and lines[back].strip() == "":
        back -= 1
    if back >= 0:
        m = DOTENV_LINT_WAIVER_RE.search(lines[back])
        if m and len(m.group(1).strip()) >= DOTENV_LINT_WAIVER_MIN_REASON_CHARS:
            return True
    return False


def _bare_dotenv_import_lineno(tree: ast.AST) -> int | None:
    """Return the lineno of the FIRST bare python-dotenv ``load_dotenv`` import
    (``from dotenv import load_dotenv`` / ``from dotenv import ... load_dotenv``)
    or a plain ``import dotenv`` in ``tree``, else None.

    The bare-``dotenv`` usage is the signal; the lineno is where the waiver is
    anchored + the error is reported. ``from explore_persona_space.orchestrate.env
    import load_dotenv`` is NOT a bare dotenv import (its module is the project
    wrapper, not ``dotenv``), so it never matches here.
    """
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "dotenv":
            return node.lineno
        if isinstance(node, ast.Import) and any(a.name == "dotenv" for a in node.names):
            return node.lineno
    return None


def _imports_huggingface_hub(tree: ast.AST) -> bool:
    """Return True iff ``tree`` imports ``huggingface_hub`` (any form): a
    top-level OR in-function ``import huggingface_hub[...]`` /
    ``from huggingface_hub[...] import ...``. ``ast.walk`` covers deferred
    in-function imports too (the #745 issue651 worst case imported it at module
    top, but issue617/issue658 import it in-function)."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Import) and any(
            a.name == "huggingface_hub" or a.name.startswith("huggingface_hub.") for a in node.names
        ):
            return True
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("huggingface_hub"):
            return True
    return False


def _imports_orchestrate_env_load_dotenv(tree: ast.AST) -> bool:
    """Return True iff ``tree`` imports ``load_dotenv`` from the project wrapper
    ``...orchestrate.env`` (any alias of the module path ending in
    ``orchestrate.env``). This is the sanctioned dotenv source the #745 check
    requires when a script also touches ``huggingface_hub``."""
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.ImportFrom)
            and (node.module or "").endswith("orchestrate.env")
            and any(a.name == "load_dotenv" for a in node.names)
        ):
            return True
    return False


def _module_top_huggingface_hub_import_lineno(tree: ast.AST) -> int | None:
    """Return the lineno of the FIRST MODULE-TOP ``huggingface_hub`` import
    (any form), else None.

    Only the module-body imports matter for the import-ORDER check: an
    in-FUNCTION ``import huggingface_hub`` executes at call time — AFTER the
    module-top ``load_dotenv()`` has already run — so it never freezes the
    constants before the env is set. (The bare-dotenv arm uses ``ast.walk`` to
    catch in-function imports too; this order arm deliberately does NOT.)
    """
    body = getattr(tree, "body", [])
    for node in body:
        if isinstance(node, ast.Import) and any(
            a.name == "huggingface_hub" or a.name.startswith("huggingface_hub.") for a in node.names
        ):
            return node.lineno
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("huggingface_hub"):
            return node.lineno
    return None


def _module_top_load_dotenv_call_lineno(tree: ast.AST) -> int | None:
    """Return the lineno of the FIRST MODULE-TOP ``load_dotenv(...)`` call
    (bare ``load_dotenv(...)`` or ``dotenv.load_dotenv(...)``), else None.

    Module-body statements only (an expression statement or an assignment whose
    value is the call) — a call buried inside a function does not establish the
    module-top env before a module-top huggingface_hub import freezes the
    constants. Used by the order arm to find where the env is actually set."""
    body = getattr(tree, "body", [])
    for node in body:
        call = None
        if (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)) or (
            isinstance(node, (ast.Assign, ast.AnnAssign))
            and isinstance(getattr(node, "value", None), ast.Call)
        ):
            call = node.value
        if call is None:
            continue
        func = call.func
        if isinstance(func, ast.Name) and func.id == "load_dotenv":
            return node.lineno
        if isinstance(func, ast.Attribute) and func.attr == "load_dotenv":
            return node.lineno
    return None


def check_dotenv_before_hf_import(*, scripts_dir: Path | None = None) -> list[str]:
    """AST-walk every ``*.py`` under ``scripts/`` and FAIL on any script that
    uses the BARE python-dotenv ``load_dotenv`` AND imports ``huggingface_hub``
    (any submodule) WITHOUT first importing the project wrapper
    ``explore_persona_space.orchestrate.env.load_dotenv``.

    Rationale (#745): the bare ``from dotenv import load_dotenv`` walks the cwd
    for a ``.env`` (so it does NOT robustly find the project ``.env`` from a
    worktree / subdir) and sets NO environment. The project wrapper reads the
    project ``.env`` (worktree-aware ``resolve_dotenv_path``) AND setdefaults the
    HF Hub upload accelerators (``HF_XET_HIGH_PERFORMANCE`` /
    ``HF_HUB_ENABLE_HF_TRANSFER``), so a script that uploads to the Hub but uses
    bare dotenv gets neither the right ``.env`` nor the accelerator default —
    large uploads then crawl. Worse, ``huggingface_hub.constants`` freezes
    ``HF_HUB_ENABLE_HF_TRANSFER`` at IMPORT time, so a bare-dotenv script that
    imports ``huggingface_hub`` at module top can never pick up the accelerator.
    The shell-level exports (bootstrap_pod.sh / GCE prelude / SLURM env block)
    are the load-bearing fix on the running fleet; this check prevents a NEW
    script from re-introducing the bare-dotenv anti-pattern.

    Detection (per script) — TWO arms, each independently waivable with
    ``# DOTENV_LINT_EXEMPT: <reason>`` (reason ≥
    :data:`DOTENV_LINT_WAIVER_MIN_REASON_CHARS` chars) on the anchor line or the
    immediately preceding non-blank line:

    ARM 1 — BARE DOTENV (the original #745 check):

    * BARE DOTENV — a ``from dotenv import [...] load_dotenv`` or a plain
      ``import dotenv`` (the project wrapper
      ``from explore_persona_space.orchestrate.env import load_dotenv`` is NOT
      bare dotenv — its module is the wrapper, not ``dotenv``);
    * AND HUGGINGFACE_HUB — any ``import huggingface_hub[...]`` /
      ``from huggingface_hub[...] import ...`` (top-level OR in-function);
    * AND NOT the project wrapper imported anywhere in the file.

    All three → FAIL, anchored at the bare-dotenv import line.

    ARM 2 — IMPORT-ORDER (#745 round 2): even when the wrapper IS imported, a
    MODULE-TOP ``huggingface_hub`` import that PRECEDES the module-top
    ``load_dotenv()`` CALL → FAIL, anchored at the huggingface_hub import line.
    Rationale: ``huggingface_hub.constants`` freezes
    ``HF_HUB_ENABLE_HF_TRANSFER`` at IMPORT time, so an accelerator env set
    AFTER the import is already too late (the constant is frozen) and the
    accelerator is inert despite the wrapper being present. The env-setting site
    is the ``load_dotenv()`` CALL line, NOT the wrapper IMPORT line: the
    accelerator setdefaults live INSIDE the wrapper's ``load_dotenv`` function
    body (``orchestrate/env.py``), so importing the wrapper sets no env — only
    the call does (using ``min(wrapper_import, call)`` would treat the mere
    import as an env-setting site and miss the wrapper-import → hf-import →
    ``load_dotenv()``-call ordering). Scope: MODULE-TOP huggingface_hub imports
    only (an in-function import runs at call time, after the module-top
    ``load_dotenv()``), and only when the file actually CALLS ``load_dotenv`` at
    module top (a script relying purely on the shell-level exports and never
    calling ``load_dotenv`` has no env-setting site to be late relative to, so
    it is out of scope). Skipped when ARM 1 already flagged the file (one error
    per file is enough — migrating to the wrapper-above-hf shape fixes both) or
    when a bare-dotenv ``# DOTENV_LINT_EXEMPT`` waiver already covered the file's
    #745 dotenv concern.

    ``scripts_dir`` is an override hook for unit tests; production callers pass
    None and the function walks the canonical ``<repo_root>/scripts`` tree.
    Bundled into the no-flags default run (same policy as
    ``check_upload_as_file`` / ``check_dispatcher_cvd_pin``).
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

        # ARM 1 — BARE-DOTENV: bare python-dotenv + huggingface_hub WITHOUT the
        # project wrapper imported anywhere. The wrapper-present escape is
        # intentional here (the author has the sanctioned source available);
        # the import-ORDER guarantee for the wrapper-present case is ARM 2.
        arm1_fired = False
        bare_lineno = _bare_dotenv_import_lineno(tree)
        # A bare-dotenv waiver expresses an explicit "#745 dotenv concern waived
        # for this file" — it suppresses BOTH arms (ARM 2 would otherwise
        # re-flag the same file on ordering, defeating the waiver).
        bare_dotenv_waived = bare_lineno is not None and _dotenv_lint_waiver_present(
            lines, bare_lineno
        )
        if (
            bare_lineno is not None
            and _imports_huggingface_hub(tree)
            and not _imports_orchestrate_env_load_dotenv(tree)
            and not bare_dotenv_waived
        ):
            arm1_fired = True
            errors.append(
                f"{py}:{bare_lineno}: bare `dotenv` load_dotenv + huggingface_hub "
                f"import without explore_persona_space.orchestrate.env.load_dotenv "
                f"(#745). The bare dotenv walks cwd (misses the project .env from a "
                f"worktree/subdir) and sets no env, so the HF Hub upload accelerators "
                f"(HF_XET_HIGH_PERFORMANCE / HF_HUB_ENABLE_HF_TRANSFER) never get "
                f"their setdefault and large uploads crawl. Import the project wrapper "
                f"`from explore_persona_space.orchestrate.env import load_dotenv` and "
                f"call load_dotenv() BEFORE the huggingface_hub import, or — if bare "
                f"dotenv is genuinely correct here — waive with "
                f"'# DOTENV_LINT_EXEMPT: <reason>' (reason ≥ "
                f"{DOTENV_LINT_WAIVER_MIN_REASON_CHARS} chars) on the import line or "
                f"the previous non-blank line."
            )

        # ARM 2 — IMPORT-ORDER (#745 round 2): even when the wrapper IS imported,
        # a MODULE-TOP huggingface_hub import that PRECEDES the module-top
        # load_dotenv() CALL freezes HF_HUB_ENABLE_HF_TRANSFER (read in
        # huggingface_hub.constants at IMPORT time) BEFORE the env is set — so
        # the accelerator is inert despite the wrapper being present. FAIL when
        # the first module-top huggingface_hub import precedes the env-setting
        # site (the load_dotenv() CALL — see the in-block comment for why the
        # wrapper IMPORT line is NOT an env-setting site). Module-top only (an
        # in-function hf import runs after the module-top load_dotenv() — see
        # helper docs), and only when the file actually CALLS load_dotenv at
        # module top (a script that relies purely on shell exports and never
        # calls load_dotenv is out of scope). Waivable on the hf import line
        # (same token + reason floor).
        # Skip when ARM 1 already flagged this file (a bare-dotenv offender is
        # also out-of-order, but one error per file is enough — migrating to
        # the wrapper-above-hf shape fixes both arms at once) OR when a
        # bare-dotenv waiver explicitly waived the #745 dotenv concern here.
        hf_lineno = _module_top_huggingface_hub_import_lineno(tree)
        if not arm1_fired and not bare_dotenv_waived and hf_lineno is not None:
            # The env-setting site is the module-top load_dotenv() CALL line —
            # NOT the wrapper import. The accelerator setdefaults live INSIDE the
            # wrapper's load_dotenv() function body (orchestrate/env.py:244-245),
            # so importing the wrapper sets NO env; only the CALL does. Comparing
            # against min(wrapper_import, call) would treat the mere import as an
            # env-setting site and miss the wrapper-import → hf-import →
            # load_dotenv()-call ordering (the constants freeze at the hf import,
            # BEFORE the later call runs). When there is no module-top call (env
            # set purely by shell-level exports — bootstrap/GCE/SLURM), there is
            # no site to be late relative to, so the order arm is out of scope.
            call_lineno = _module_top_load_dotenv_call_lineno(tree)
            if call_lineno is not None:
                env_lineno = call_lineno
                if hf_lineno < env_lineno and not _dotenv_lint_waiver_present(lines, hf_lineno):
                    errors.append(
                        f"{py}:{hf_lineno}: module-top huggingface_hub import PRECEDES "
                        f"the dotenv/env setup at line {env_lineno} (#745 import-order). "
                        f"huggingface_hub.constants freezes HF_HUB_ENABLE_HF_TRANSFER at "
                        f"IMPORT time, so an accelerator env set AFTER the import is "
                        f"already too late — the upload accelerator is inert. Move the "
                        f"`from explore_persona_space.orchestrate.env import load_dotenv` "
                        f"+ load_dotenv() ABOVE the huggingface_hub import, or — if the "
                        f"ordering is genuinely correct here — waive with "
                        f"'# DOTENV_LINT_EXEMPT: <reason>' (reason ≥ "
                        f"{DOTENV_LINT_WAIVER_MIN_REASON_CHARS} chars) on the "
                        f"huggingface_hub import line or the previous non-blank line."
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


def _judge_pin_line_waived(lines: list[str], idx: int) -> bool:
    """Return True iff a ``# noqa: judge-model-pin`` waiver is on the hit line
    (``idx``, 0-based) or the immediately preceding non-blank line. Same
    convention as the dotenv / upload-as-file waivers."""
    if 0 <= idx < len(lines) and JUDGE_PIN_WAIVER_RE.search(lines[idx]):
        return True
    back = idx - 1
    while back >= 0 and lines[back].strip() == "":
        back -= 1
    return back >= 0 and bool(JUDGE_PIN_WAIVER_RE.search(lines[back]))


def _file_is_judge_context(text: str, name: str) -> bool:
    """Return True iff the file is plausibly a JUDGE script — its filename
    contains ``judge``, OR its body imports a judge client / a ``*judge*``
    module / defines a ``judge_*`` function / says "as judge" (docstring). Used
    to NARROW the (f) DEFAULT_MODEL-constant arm so a non-judge module's
    cost-table constant does not false-fire (#765 round 2)."""
    if JUDGE_PIN_CONTEXT_FILENAME_RE.search(name):
        return True
    return bool(JUDGE_PIN_CONTEXT_BODY_RE.search(text))


def _judge_pin_is_hit(lines: list[str], idx: int, *, judge_context: bool = False) -> bool:
    """Return True iff line ``idx`` (0-based) carries a forbidden non-Sonnet
    judge-model substring in an ASSIGNMENT / CALL context (NOT a bare prose
    mention or comment). See the :data:`JUDGE_PIN_FORBIDDEN_SUBSTRINGS` block
    for the gate definition. ``judge_context`` enables the (f) DEFAULT_MODEL
    arm (only for judge-script files). The (e) shell-var-indirection arm is
    handled at file scope in :func:`_scan_judge_pin_file` (the hit is the
    assignment line, not this forbidden-literal line)."""
    line = lines[idx]
    if line.lstrip().startswith("#"):
        return False  # a pure comment line is never a hit
    if not any(sub in line for sub in JUDGE_PIN_FORBIDDEN_SUBSTRINGS):
        return False
    # (a) RHS of a judge-named assignment/key, or (b) a --judge-model /
    # judge_model= / JUDGE_MODEL= CLI/shell arg, both on the hit line:
    if JUDGE_PIN_VAR_RE.search(line) or JUDGE_PIN_FLAG_RE.search(line):
        return True
    # (f) judge-script DEFAULT_MODEL / MODEL_DEFAULT / JUDGE_DEFAULT constant —
    # only in a judge-context file (NARROW broadening):
    if judge_context and JUDGE_PIN_DEFAULT_MODEL_VAR_RE.search(line):
        return True
    # (c) a model=/model: kwarg on the line AND a judge-call token within the
    # +/- JUDGE_PIN_CONTEXT_WINDOW non-comment line window:
    if JUDGE_PIN_MODEL_KWARG_RE.search(line):
        lo = max(0, idx - JUDGE_PIN_CONTEXT_WINDOW)
        hi = min(len(lines), idx + JUDGE_PIN_CONTEXT_WINDOW + 1)
        for j in range(lo, hi):
            ctx = lines[j]
            if ctx.lstrip().startswith("#"):
                continue
            if any(tok in ctx for tok in JUDGE_PIN_CALL_TOKENS):
                return True
    # (d) split-argv recognition: this forbidden-literal line has no var/flag/
    # kwarg of its own (arms a/b/c missed), but a preceding non-blank,
    # non-comment line within the look-ahead window is the BARE `--judge-model`
    # flag token (the run_evals_190.py:52-53 list-literal shape). Look BACK so
    # the VALUE line (the one carrying the forbidden pin) is the reported hit.
    back, seen = idx - 1, 0
    while back >= 0 and seen < JUDGE_PIN_SPLIT_ARGV_LOOKAHEAD:
        prev = lines[back]
        if prev.strip() == "":
            back -= 1
            continue
        seen += 1
        if not prev.lstrip().startswith("#") and JUDGE_PIN_BARE_FLAG_RE.search(prev):
            return True
        back -= 1
    return False


def _judge_pin_rel(p: Path) -> str:
    """Repo-root-relative POSIX path, or the file's own posix path when it lives
    OUTSIDE the repo (a unit-test fixture tree) — so the exact-path allowlists
    never accidentally exempt a tmp fixture sharing a basename with a real
    allowlisted file."""
    try:
        return p.resolve().relative_to(_REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return p.as_posix()


def _judge_pin_sh_var_indirection_hits(lines: list[str]) -> set[int]:
    """Return the 0-based indices of shell-var ASSIGNMENT lines that pin a
    forbidden judge model via indirection (#765 round 2, arm (e)). Two-pass:
    (1) collect every ``VAR=<value-with-a-forbidden-substring>`` (var name need
    NOT contain JUDGE_MODEL); (2) if ANY non-comment line passes
    ``--judge-model`` consuming that var (``$VAR`` / ``${VAR}`` / ``${VAR:-...}``),
    the ASSIGNMENT line is a hit. Returns assignment indices only — the forbidden
    literal lives there, and the ``--judge-model "${VAR}"`` reference line carries
    no forbidden substring so it is never separately reported."""
    # Pass 1: var name -> assignment line idx, for assignments whose value has a
    # forbidden judge substring (skip pure-comment lines).
    forbidden_vars: dict[str, int] = {}
    for idx, line in enumerate(lines):
        if line.lstrip().startswith("#"):
            continue
        m = JUDGE_PIN_SH_ASSIGN_RE.match(line)
        if not m:
            continue
        if not any(sub in line for sub in JUDGE_PIN_FORBIDDEN_SUBSTRINGS):
            continue
        forbidden_vars[m.group(1)] = idx
    if not forbidden_vars:
        return set()
    # Pass 2: a non-comment `--judge-model` line consuming one of those vars
    # promotes that var's assignment line to a hit.
    hits: set[int] = set()
    for var, assign_idx in forbidden_vars.items():
        ref_re = _judge_pin_sh_var_ref_re(var)
        for line in lines:
            if line.lstrip().startswith("#"):
                continue
            if ref_re.search(line):
                hits.add(assign_idx)
                break
    return hits


def _scan_judge_pin_file(p: Path, *, sh_allowlist: bool, errors: list[str]) -> None:
    """Scan one file for judge-model-pin hits, appending error lines to
    ``errors``. Allowlist + file-level-waiver short-circuit; per-line hits gated
    by :func:`_judge_pin_is_hit` and waivable by :func:`_judge_pin_line_waived`.
    The (e) shell-var-indirection arm (``.sh`` only) is a file-scope two-pass
    check (:func:`_judge_pin_sh_var_indirection_hits`); the (f) DEFAULT_MODEL
    arm is gated on :func:`_file_is_judge_context`."""
    rel = _judge_pin_rel(p)
    if rel in JUDGE_PIN_FILE_ALLOWLIST:
        return
    if rel in (JUDGE_PIN_LEGACY_ALLOWLIST_SH if sh_allowlist else JUDGE_PIN_LEGACY_ALLOWLIST):
        return
    try:
        text = p.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return
    if JUDGE_PIN_FILE_WAIVER_RE.search(text):
        return  # file-level waiver
    lines = text.splitlines()
    judge_context = _file_is_judge_context(text, p.name)
    # (e) shell-var-indirection assignment-line hits (.sh files only).
    forced_idxs = _judge_pin_sh_var_indirection_hits(lines) if sh_allowlist else set()
    reported: set[int] = set()
    for idx in range(len(lines)):
        is_hit = idx in forced_idxs or _judge_pin_is_hit(lines, idx, judge_context=judge_context)
        if not is_hit or _judge_pin_line_waived(lines, idx) or idx in reported:
            continue
        reported.add(idx)
        match = next(
            (s for s in JUDGE_PIN_FORBIDDEN_SUBSTRINGS if s in lines[idx]),
            "<non-Sonnet>",
        )
        errors.append(
            f"{p}:{idx + 1}: hardcoded non-Sonnet judge pin '{match}' at a "
            f"judge call site. fix: use {JUDGE_PIN_CANONICAL} (or waive a "
            f"calibration control with '# noqa: judge-model-pin'). See "
            f".claude/rules/llm-judging.md."
        )


def check_judge_model_pins(
    *,
    scripts_dir: Path | None = None,
    src_dir: Path | None = None,
    tests_dir: Path | None = None,
) -> list[str]:
    """Walk ``scripts/**/*.py``, ``scripts/**/*.sh``,
    ``src/explore_persona_space/**/*.py``, and ``tests/**/*.py`` and FAIL on a
    hardcoded NON-Sonnet judge-model pin at a judge call site (#765).

    The standing project rule pins ONE judge — ``claude-sonnet-4-5-20250929`` —
    for every judged behavior (CLAUDE.md "LLM judge"; full recipe
    ``.claude/rules/llm-judging.md``). A forbidden substring
    (:data:`JUDGE_PIN_FORBIDDEN_SUBSTRINGS`) on a NON-COMMENT line is a HIT iff:
      (a) the line matches :data:`JUDGE_PIN_VAR_RE` (RHS of a judge-named
          assignment/key); or
      (b) the line matches :data:`JUDGE_PIN_FLAG_RE` (``--judge-model`` /
          ``judge_model=`` / ``JUDGE_MODEL=`` CLI/shell arg — covers .py
          argparse defaults AND .sh launchers); or
      (c) the line carries a ``model=`` / ``model:`` kwarg AND a
          :data:`JUDGE_PIN_CALL_TOKENS` token appears within +/-
          :data:`JUDGE_PIN_CONTEXT_WINDOW` non-comment lines;
      (d) the forbidden-pin literal is preceded (within
          :data:`JUDGE_PIN_SPLIT_ARGV_LOOKAHEAD` non-blank lines) by a BARE
          ``--judge-model`` flag on its own list-literal line — the split-argv
          shape (#765 round 2);
      (e) [``.sh`` only] the line ASSIGNS a shell var to a forbidden-pin value
          AND that var is later consumed by ``--judge-model`` — shell-var
          indirection, a file-scope two-pass check
          (:func:`_judge_pin_sh_var_indirection_hits`, #765 round 2); or
      (f) [judge-context files only — :func:`_file_is_judge_context`] the line
          matches :data:`JUDGE_PIN_DEFAULT_MODEL_VAR_RE` (a ``DEFAULT_MODEL`` /
          ``MODEL_DEFAULT`` / ``JUDGE_DEFAULT`` constant whose name lacks
          JUDGE_MODEL, #765 round 2).
    A bare mention inside a descriptive string or a comment (no judge-named
    assignment / ``--judge-model`` flag / judge ``model=`` on the line) is NOT
    a hit (the prose-mention guard). The canonical pin
    ``claude-sonnet-4-5-20250929`` contains NO forbidden substring (it is
    ``claude-sonnet-4-5-...``, NOT ``claude-3-5-sonnet``), so it never matches.

    Exempt: :data:`JUDGE_PIN_FILE_ALLOWLIST` (whole file — doc/registry/self/
    test-fixtures), :data:`JUDGE_PIN_LEGACY_ALLOWLIST` (.py grandfathered
    relative path), :data:`JUDGE_PIN_LEGACY_ALLOWLIST_SH` (.sh grandfathered
    relative path), a file-level ``# epm-allow-judge-model-pin`` comment, and a
    per-line ``# noqa: judge-model-pin`` on the hit line or the immediately
    preceding non-blank line. One error line per hit; exit non-zero on any hit.
    ``scripts_dir`` / ``src_dir`` / ``tests_dir`` are unit-test override hooks
    (production callers pass all None). Bundled into the no-flags default run.
    """
    py_roots = [
        scripts_dir if scripts_dir is not None else _REPO_ROOT / "scripts",
        src_dir if src_dir is not None else _REPO_ROOT / "src" / "explore_persona_space",
        tests_dir if tests_dir is not None else _REPO_ROOT / "tests",
    ]
    # .sh launchers live only under scripts/ — reuse the (possibly overridden)
    # scripts root for the shell walk too.
    sh_root = scripts_dir if scripts_dir is not None else _REPO_ROOT / "scripts"

    errors: list[str] = []
    seen: set[Path] = set()
    for root in py_roots:
        if not root.exists():
            continue
        for py in sorted(root.rglob("*.py")):
            if py.is_file() and py not in seen:
                seen.add(py)
                _scan_judge_pin_file(py, sh_allowlist=False, errors=errors)
    if sh_root.exists():
        for sh in sorted(sh_root.rglob("*.sh")):
            if sh.is_file():
                _scan_judge_pin_file(sh, sh_allowlist=True, errors=errors)
    return errors


# --- `--check-phase-done-reserved` (#930): reserved `[phase=done]` token ----
# The literal reserved token from .claude/rules/pod-side-reporting.md
# requirement 1: `poll_pipeline.py` declares status="done" when the most
# recent `[phase=...]` line in the MAIN dispatcher log is `[phase=done]`,
# so the token is reserved for the dispatcher's single terminal line —
# a phase script whose stdout flows into that log must never emit it
# (incident #545, 2026-06-11: a per-cell echo produced a false status=done
# while GPUs were at 85%; recurred #920 r1: six phase scripts).
PHASE_DONE_TOKEN = "[phase=done]"
# Python-target invocation edge on a logical shell line: `uv run python
# scripts/x.py`, `nohup ... python -u scripts/x.py`, `CUDA_VISIBLE_DEVICES=0
# uv run python3 scripts/x.py`. `$VAR`-prefixed paths / `python -m` launches
# deliberately do NOT match (documented false-negative gaps, see the check
# docstring).
PHASE_DONE_PY_INVOKE_RE = re.compile(
    r"""python3?(?:\.\d+)?\s+(?:-[A-Za-z]+\s+)*["']?(scripts/[A-Za-z0-9_./-]+\.py)\b"""
)
# Shell-target invocation edge: `bash scripts/x.sh` / `sh scripts/x.sh` /
# `source scripts/x.sh` — the i488 run_all -> sub-dispatcher class.
PHASE_DONE_SH_INVOKE_RE = re.compile(
    r"""(?:\bbash|\bsh(?=\s)|\bsource)\s+["']?(scripts/[A-Za-z0-9_./-]+\.sh)\b"""
)
# Stdout-isolation exclusion: matches `> f`, `>> f`, `1> f`, `&> f` (stdout
# redirected away from the main log — the per-worker isolation pattern,
# e.g. scripts/issue658_8gpu_dispatch.sh). Does NOT match `2>&1` alone,
# `2> err.log` (stderr-only; stdout still flows), or `... 2>&1 | tee -a log`
# (tee duplicates to main stdout — the exact #545-family shape): those edges
# stay checked. The `(?!\s*&)` lookahead keeps fd-dup forms (`>&2`) out.
# Applied per COMMAND SEGMENT (the logical line split at unquoted
# `&&`/`||`/`;`/`|`/lone-`&` separators via _split_sh_command_segments),
# NOT line-globally — a redirect in a different segment neither suppresses
# nor rescues an invocation elsewhere on the line (round-2
# `phase-done-shell-edge-scoping` fix).
PHASE_DONE_REDIRECT_RE = re.compile(r"(?:^|\s)(?:1?>>?|&>>?)(?!\s*&)")
# Per-line waiver for a mode-gated standalone-lane terminal (a dual-mode
# phase script whose emission is OFF the dispatcher path — the issue-920
# nulls_figures shape). Same placement convention as JUDGE_PIN_WAIVER_RE:
# the emission line or the immediately preceding non-blank line. Waiver
# comments MUST name the intended mode/invoker (code-review enforced).
PHASE_DONE_WAIVER_RE = re.compile(r"#\s*noqa:\s*phase-done-reserved\b")
# A .sh line is an emission site iff (after quote-aware trailing-comment
# strip) it carries the token AND one of these emitters — `print\s*\(`
# covers python-heredoc blocks embedded in .sh (`uv run python - <<'PY'`).
PHASE_DONE_SH_EMIT_RE = re.compile(r"\becho\b|\bprintf\b|\bprint\s*\(")
# Logging-ish attribute names whose calls count as .py emission sites
# (covers logger.info / log.warning / LOGGER.error / sys.stdout.write).
# `re.compile` / `re.search` are deliberately absent so the poller's own
# match/detection code never flags.
PHASE_DONE_PY_EMIT_ATTRS = frozenset(
    {"debug", "info", "warning", "error", "critical", "exception", "log", "write"}
)
# Grandfathered legacy (invoker .sh, target) EDGE pairs — repo-root-relative
# POSIX paths, annotated per entry (mirrors JUDGE_PIN_LEGACY_ALLOWLIST's
# style). Edge grain, NOT emitter-file grain: a future NEW dispatcher
# invoking the same legacy emitter is still flagged. Fixing the legacy
# emissions is prune-on-touch (NOT this task's scope); stale entries are
# harmless (frozenset membership). Derivation: live-tree diff-and-adjudicate
# 2026-07-03 against the task #930 plan §4.5 expected seed.
PHASE_DONE_EDGE_LEGACY_ALLOWLIST: frozenset[tuple[str, str]] = frozenset(
    {
        # #545 family (the original incident's own dispatcher): sweep.py's
        # terminal print (line ~1516) tees into the main log 4x mid-pipeline:
        ("scripts/issue545_dispatch.sh", "scripts/issue545_sweep.py"),
        # #654: extract phase terminal logger.info (line ~387), tee'd to the
        # main log:
        ("scripts/issue654_dispatch.sh", "scripts/issue654_extract.py"),
        # #810 CPU-lane runner: four phase scripts each emit a terminal done
        # (lines ~530 / ~369 / ~433 / ~426) ahead of the runner's own
        # terminal (:68). MIGRATE bucket (live-hazard): the #810 family is
        # still actively churning — a follow-up round re-running this runner
        # reproduces the false-done with the lint green; fix these emissions
        # on next touch:
        ("scripts/issue810_cpu_phase.sh", "scripts/issue810_fit_reconstruction.py"),
        ("scripts/issue810_cpu_phase.sh", "scripts/issue810_batch_rejudge_highm.py"),
        ("scripts/issue810_cpu_phase.sh", "scripts/issue810_fit_readout.py"),
        ("scripts/issue810_cpu_phase.sh", "scripts/issue810_analyze.py"),
        # i488 run-all invokes two sub-dispatchers (own terminals ~166/~257
        # + ~121) non-redirected, ahead of run_all's own terminal (:131):
        ("scripts/i488_run_all.sh", "scripts/i488_phase23_dispatch.sh"),
        ("scripts/i488_run_all.sh", "scripts/i488_phase4_dispatch.sh"),
        # #552 (round-1-critique triage, 2026-07-03 — the plan-v1 probe
        # MISSED this edge; a legacy completed family, same tee-shape class
        # as #654): prep script's terminal logger.info (line ~277) tees into
        # the main log at run_issue552_sweep.sh:105 via `2>&1 | tee`:
        ("scripts/run_issue552_sweep.sh", "scripts/issue_552_prep_good_corpus.py"),
        # PRE-SEEDED for the in-flight issue-920 branch merge: nulls_figures'
        # done (line ~781) is the standalone cpu-mid lane's dispatcher
        # terminal, mode-gated OFF the --gpu-null-only dispatcher path
        # (verified on the issue-920 branch 2026-07-03); the file is
        # dual-mode by design:
        ("scripts/issue920_dispatch.sh", "scripts/issue920_nulls_figures.py"),
    }
)


def _phase_done_line_waived(lines: list[str], idx: int) -> bool:
    """Return True iff a ``# noqa: phase-done-reserved`` waiver is on the
    emission line (``idx``, 0-based) or the immediately preceding non-blank
    line. For a multi-line ``.py`` call the anchor is the AST call-head
    lineno — waive at the call head, not beside a continuation-line string
    literal. Same convention as :func:`_judge_pin_line_waived`."""
    if 0 <= idx < len(lines) and PHASE_DONE_WAIVER_RE.search(lines[idx]):
        return True
    back = idx - 1
    while back >= 0 and lines[back].strip() == "":
        back -= 1
    return back >= 0 and bool(PHASE_DONE_WAIVER_RE.search(lines[back]))


def _py_phase_done_emission_lines(target: Path) -> list[int]:
    """AST-scan a phase ``.py`` for genuine ``[phase=done]`` EMISSION sites,
    returning sorted 1-based call-head line numbers (waived sites dropped).

    An emission site is an ``ast.Call`` whose func is ``print`` (a bare
    ``Name``) or an ``Attribute`` in :data:`PHASE_DONE_PY_EMIT_ATTRS`
    (``logger.info`` / ``sys.stdout.write`` / ...), AND any ``ast.Constant``
    string reachable by ``ast.walk`` inside the call's positional args
    carries the literal token (covers f-string ``JoinedStr`` parts and
    %%-style format strings). Comments, docstrings, ``re.compile`` /
    ``re.search`` match sites, and ``"[phase=done]" in line`` membership
    tests are excluded by construction. A ``SyntaxError`` skips the file
    (a non-parsing .py cannot run as a phase script)."""
    try:
        text = target.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return []
    if PHASE_DONE_TOKEN not in text:
        return []  # cheap pre-filter before parsing
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []
    lines = text.splitlines()
    sites: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name):
            if func.id != "print":
                continue
        elif isinstance(func, ast.Attribute):
            if func.attr not in PHASE_DONE_PY_EMIT_ATTRS:
                continue
        else:
            continue
        has_token = any(
            isinstance(sub, ast.Constant)
            and isinstance(sub.value, str)
            and PHASE_DONE_TOKEN in sub.value
            for arg in node.args
            for sub in ast.walk(arg)
        )
        if not has_token:
            continue
        if _phase_done_line_waived(lines, node.lineno - 1):
            continue
        sites.add(node.lineno)
    return sorted(sites)


# A `#` begins a shell comment only at the START of a word — i.e. at string
# start or after whitespace / an operator character (the POSIX tokenizer
# rule). Used by _strip_sh_trailing_comment's word-boundary test.
_SH_COMMENT_BOUNDARY_CHARS = frozenset(" \t;&|(<>")


def _strip_sh_trailing_comment(line: str) -> str:
    """Cut an unquoted trailing ``#`` comment from a shell line via a small
    quote- and backslash-escape-aware char scanner. Word-boundary-aware
    (the round-3 ``phase-done-comment-strip-midword-fn`` fix): a ``#``
    starts a comment ONLY when it BEGINS a shell word — at string start or
    preceded by unescaped whitespace / an operator character
    (:data:`_SH_COMMENT_BOUNDARY_CHARS`) — matching the shell tokenizer, so
    a mid-word ``#`` (``tag=run#1``, ``$#``, ``${x#pat}``, ``${#ARR[@]}``,
    ``2#101``) and a backslash-escaped ``\\#`` never cut. (Pre-fix the
    unconditional cut truncated the scanned line at ANY unquoted ``#``,
    hiding every invocation/emission after it — a silent false negative
    once the strip became load-bearing for invocation scanning in round 2.)
    A ``#`` inside single or double quotes is kept; outside single quotes a
    backslash escapes the next char (an escaped operator like ``\\;`` is
    NOT a word boundary; ``\\"`` does not close a double quote). Residual
    over-cut (fail-toward-false-negative — the safe direction for a
    pre-commit-gating lint): a word-initial ``#`` inside an unquoted
    ``$(...)`` command substitution still cuts there (the scanner is not
    ``$()``-aware)."""
    out: list[str] = []
    in_single = in_double = False
    prev_escaped = False
    i, n = 0, len(line)
    while i < n:
        ch = line[i]
        if ch == "\\" and not in_single and i + 1 < n:
            out.append(ch)
            out.append(line[i + 1])
            prev_escaped = True
            i += 2
            continue
        if (
            ch == "#"
            and not in_single
            and not in_double
            and (not out or (not prev_escaped and out[-1] in _SH_COMMENT_BOUNDARY_CHARS))
        ):
            break
        if ch == "'" and not in_double:
            in_single = not in_single
        elif ch == '"' and not in_single:
            in_double = not in_double
        out.append(ch)
        prev_escaped = False
        i += 1
    return "".join(out)


def _split_sh_command_segments(logical: str) -> list[str]:
    """Split a (comment-stripped) logical shell line into COMMAND SEGMENTS at
    unquoted command separators — ``&&``, ``||``, ``;``, ``|``, and a lone
    background ``&`` — so the stdout-redirect exclusion can be scoped to the
    segment containing each invocation (round-2 fix for the
    ``phase-done-shell-edge-scoping`` concern: a redirect in a DIFFERENT
    segment of the same line must neither suppress nor be suppressed by an
    invocation elsewhere on the line).

    Quote-aware (same single/double-quote char scan as
    :func:`_strip_sh_trailing_comment`) and backslash-escape-aware (an
    escaped separator like ``find -exec ... \\;`` does not split). A pipe IS
    a boundary — this PRESERVES the tee-still-checked semantics of plan
    §4.3: for ``child.py 2>&1 | tee -a log`` the invocation's own segment
    (``child.py 2>&1``) carries no stdout redirect, so the edge stays
    checked, while a downstream ``> f`` applied to ``tee``'s output no
    longer suppresses the child. A lone ``&`` splits only when it is neither
    part of ``&&`` (handled first) nor an fd-dup/`&>` form (``2>&1`` /
    ``>&2`` / ``&> f`` — guarded by the neighboring ``>``). Empty segments
    (trailing ``&``, ``;;``) are harmless — they match no invocation."""
    segments: list[str] = []
    cur: list[str] = []
    in_single = in_double = False
    i, n = 0, len(logical)
    while i < n:
        ch = logical[i]
        if ch == "\\" and not in_single and i + 1 < n:
            cur.append(ch)
            cur.append(logical[i + 1])
            i += 2
            continue
        if in_single:
            if ch == "'":
                in_single = False
            cur.append(ch)
            i += 1
            continue
        if in_double:
            if ch == '"':
                in_double = False
            cur.append(ch)
            i += 1
            continue
        if ch == "'":
            in_single = True
            cur.append(ch)
            i += 1
            continue
        if ch == '"':
            in_double = True
            cur.append(ch)
            i += 1
            continue
        nxt = logical[i + 1] if i + 1 < n else ""
        prev = logical[i - 1] if i > 0 else ""
        if (ch == "&" and nxt == "&") or (ch == "|" and nxt == "|"):
            segments.append("".join(cur))
            cur = []
            i += 2
            continue
        if ch == ";" or ch == "|" or (ch == "&" and nxt != ">" and prev != ">"):
            segments.append("".join(cur))
            cur = []
            i += 1
            continue
        cur.append(ch)
        i += 1
    segments.append("".join(cur))
    return segments


def _sh_phase_done_emission_lines(target: Path) -> list[int]:
    """Line-scan a phase ``.sh`` for genuine ``[phase=done]`` emission sites,
    returning sorted 1-based line numbers (waived sites dropped). A line is
    an emission iff, after quote-aware trailing-comment strip, it carries the
    literal token AND matches :data:`PHASE_DONE_SH_EMIT_RE` (``echo`` /
    ``printf`` / a python-heredoc ``print(``)."""
    try:
        text = target.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return []
    lines = text.splitlines()
    sites: list[int] = []
    for idx, raw in enumerate(lines):
        stripped = _strip_sh_trailing_comment(raw)
        if PHASE_DONE_TOKEN not in stripped:
            continue
        if not PHASE_DONE_SH_EMIT_RE.search(stripped):
            continue
        if _phase_done_line_waived(lines, idx):
            continue
        sites.append(idx + 1)
    return sites


def _phase_done_line_edges(logical: str) -> list[str]:
    """Return the ordered, deduplicated target paths (``scripts/*.py|.sh``)
    of every NON-REDIRECTED phase-script invocation on one logical shell
    line — the round-2 ``phase-done-shell-edge-scoping`` core. The line is
    trailing-comment-stripped (quote-aware) and split into command segments
    (:func:`_split_sh_command_segments`); EVERY invocation match is
    considered (not just the first), and a segment's invocations are kept
    only when THAT segment carries no stdout redirect
    (:data:`PHASE_DONE_REDIRECT_RE`). ``echo``-preview segments are
    skipped."""
    targets: list[str] = []
    seen: set[str] = set()
    for segment in _split_sh_command_segments(_strip_sh_trailing_comment(logical)):
        if segment.strip().startswith("echo "):
            continue  # dry-run preview segment (`... && echo "next: ..."`)
        matches = sorted(
            (
                *PHASE_DONE_PY_INVOKE_RE.finditer(segment),
                *PHASE_DONE_SH_INVOKE_RE.finditer(segment),
            ),
            key=lambda m: m.start(),
        )
        if not matches:
            continue
        if PHASE_DONE_REDIRECT_RE.search(segment):
            continue  # per-worker-log stdout isolation, scoped to THIS segment
        for m in matches:
            target_rel = m.group(1)
            if target_rel not in seen:
                seen.add(target_rel)
                targets.append(target_rel)
    return targets


def check_phase_done_reserved(
    *,
    scripts_dir: Path | None = None,
    allowlist: frozenset[tuple[str, str]] | None = None,
) -> list[str]:
    """Walk every ``scripts/**/*.sh`` dispatcher and FAIL any non-redirected
    invocation of a ``scripts/*.py|*.sh`` phase script whose file contains a
    genuine ``[phase=done]`` emission site — the reserved-token contract of
    ``.claude/rules/pod-side-reporting.md`` requirement 1 (#545, #920).

    THE CONTRACT: ``poll_pipeline.py`` declares ``status="done"`` when the
    most recent ``[phase=...]`` line in the MAIN dispatcher log is
    ``[phase=done]``, so the token there is reserved for the dispatcher's
    single terminal line. A phase script launched with stdout flowing into
    that log (non-redirected, or ``2>&1 | tee``-duplicated) that emits the
    token mid-pipeline reads as a false ``status=done`` while the run is
    live (incident #545: GPUs at 85%; recurred #920 r1: six phase scripts).

    VIOLATION UNIT = a non-redirected invocation EDGE ``(invoking .sh,
    target .py|.sh)`` where the target has ≥1 emission site. A dispatcher's
    OWN emission sites are unrestricted in count (mode-gated multi-exit
    dispatchers are ubiquitous and legitimate; static mutual-exclusivity is
    undecidable), and the suffixed smoke terminal (``[phase=done] SMOKE
    COMPLETE ...``) is a dispatcher-own-file line, allowed by construction.

    EXCLUSIONS: each logical line is trailing-comment-stripped (quote-aware,
    so comment text can neither match an invocation nor carry a suppressing
    redirect) and split into COMMAND SEGMENTS at unquoted separators
    (``&&`` / ``||`` / ``;`` / ``|`` / lone background ``&`` —
    :func:`_split_sh_command_segments`; backslash-escaped separators do not
    split). EVERY invocation on the line is checked (not just the first
    regex match), and an invocation is skipped only when ITS OWN segment
    redirects stdout away from the main log (:data:`PHASE_DONE_REDIRECT_RE`
    — the per-worker isolation pattern); a redirect in a DIFFERENT segment
    of the same line does not suppress, and ``2>&1 | tee`` stays checked
    because the pipe is a segment boundary (the invocation's own segment
    carries no stdout redirect) — the round-2 fix for the
    ``phase-done-shell-edge-scoping`` concern (pre-fix, ``a.py && bad.py``
    only inspected ``a.py``, and ``bad.py; echo ok > marker`` was wrongly
    suppressed by the line-global redirect search). The trailing-comment
    strip is word-boundary- and escape-aware (round 3,
    ``phase-done-comment-strip-midword-fn``): a ``#`` cuts only where it
    BEGINS a shell word, so an executable mid-word ``#`` (``tag=run#1; uv
    run python scripts/x.py``) or an escaped ``\\#`` no longer truncates
    the scanned line ahead of a real invocation
    (:func:`_strip_sh_trailing_comment`). Comment lines and
    ``echo``-preview SEGMENTS are skipped — the echo skip is segment grain
    (round 3): an ``echo`` segment ahead of a real invocation on the same
    logical line (``echo \\#; uv run python scripts/x.py``) no longer
    hides it. ``.py`` emission detection
    is AST-based (comments / docstrings / ``re.compile``-``re.search`` match
    sites / membership tests never flag); ``.sh`` emission detection is
    quote-aware comment-stripped ``echo|printf|print(``. A
    ``# noqa: phase-done-reserved`` waiver on the emission line or the
    immediately preceding non-blank line drops that site (the escape for
    dual-mode files whose emission is mode-gated to a standalone-dispatcher
    lane; the waiver comment must name the intended mode/invoker). Legacy
    edges are frozen in :data:`PHASE_DONE_EDGE_LEGACY_ALLOWLIST` (edge
    grain, annotated).

    RESIDUAL FALSE-NEGATIVE GAPS (documented, all fail toward NOT flagging —
    the correct direction for a pre-commit gate): (i) ``.py``-dispatcher
    subprocess fan-out (``issue545_sweep.py -> issue545_eval_cell.py`` — the
    original #545 emission path; the only live instance sits inside the
    allowlisted #545 family whose .sh edge keeps it visible); (ii)
    ``$VAR``-prefixed script paths (``python "$REPO/scripts/x.py"``);
    (iii) ``python -m`` module launches; (iv) launcher scripts generated at
    runtime by template-writers; (v) direct append-INTO-the-main-log
    redirection (``>> "$MAIN_LOG" 2>&1`` — the redirect exclusion cannot
    tell a per-worker log from the dispatcher's own main log; no live
    instance — historical shapes use ``tee`` and ARE caught); (vi) a
    dispatcher's OWN mid-pipeline emissions in a loop (own-file sites are
    unrestricted by construction); (vii) a word-initial unquoted ``#``
    inside a ``$(...)`` command substitution truncates the scanned line at
    the comment-strip (the scanner is not ``$()``-aware), hiding any
    invocation after it — a MID-WORD ``#`` (``${#ARR[@]}``, ``tag=run#1``,
    ``$#``) and an escaped ``\\#`` no longer truncate as of the round-3
    word-boundary fix (:func:`_strip_sh_trailing_comment`). TWO deliberate
    fail-toward-FLAGGING exceptions (false positives, not false
    negatives): (a) a stdout redirect applied to a whole
    subshell/group (``( a.py; b.py ) > log`` / ``{ ...; } > log``) is not
    attributed to the segments INSIDE the group, so an emitting invocation
    there flags even though the group's stdout is isolated — no live
    instance; (b) a pipe-DOWNSTREAM stdout redirect (``a.py 2>&1 | tee f
    > /dev/null``) is attributed to the downstream segment only — the pipe
    is a segment boundary and deliberately NON-isolating (the plan §4.3
    tee-still-checked semantics), so an emitting invocation upstream of
    the pipe still flags even when the pipeline's terminal stdout is
    discarded. Both are waivable via ``# noqa: phase-done-reserved`` or
    the per-worker pattern.

    ``scripts_dir`` is an override hook for unit tests (production callers
    pass None → the canonical ``<repo_root>/scripts`` tree); ``allowlist``
    overrides :data:`PHASE_DONE_EDGE_LEGACY_ALLOWLIST` for tests /
    re-derivation. Bundled into the no-flags default run AND enforced at
    commit time by the ``workflow-lint-phase-done-reserved`` pre-commit hook
    on any ``scripts/*.sh|py`` change.
    """
    root = scripts_dir if scripts_dir is not None else _REPO_ROOT / "scripts"
    allow = allowlist if allowlist is not None else PHASE_DONE_EDGE_LEGACY_ALLOWLIST
    if not root.exists():
        return []
    errors: list[str] = []
    emission_cache: dict[Path, list[int]] = {}
    for sh in sorted(root.rglob("*.sh")):
        if not sh.is_file():
            continue
        lines = sh.read_text(encoding="utf-8").splitlines()
        for first, _last, logical in _iter_logical_shell_lines(lines):
            stripped = logical.strip()
            # Comment lines are not launches. echo-preview skipping is
            # SEGMENT grain inside _phase_done_line_edges (round 3): a
            # line-level `echo `-prefix skip hid a real invocation in a
            # later segment (`echo \#; uv run python scripts/x.py`).
            if stripped.startswith("#"):
                continue
            # Round-2 (`phase-done-shell-edge-scoping`): EVERY non-redirected
            # invocation on the line is an edge — comment-stripped,
            # segment-split, redirect scoped per segment (see
            # _phase_done_line_edges), one error per (logical line, target).
            for target_rel in _phase_done_line_edges(logical):
                target = root / target_rel.removeprefix("scripts/")
                if not target.is_file() or target == sh:
                    continue
                if target not in emission_cache:
                    emission_cache[target] = (
                        _py_phase_done_emission_lines(target)
                        if target.suffix == ".py"
                        else _sh_phase_done_emission_lines(target)
                    )
                sites = emission_cache[target]
                if not sites:
                    continue
                sh_rel = _judge_pin_rel(sh)  # repo-rel POSIX; abs posix for tmp fixtures
                if (sh_rel, target_rel) in allow:
                    continue
                errors.append(
                    f"{sh}:{first + 1}: invokes {target_rel} (stdout flows into this "
                    f"dispatcher's main log) but {target_rel} emits the RESERVED "
                    f"{PHASE_DONE_TOKEN} token at line(s) {sites}. The token in the "
                    f"MAIN dispatcher log is reserved for the dispatcher's single "
                    f"terminal line — a mid-pipeline emission reads as a false "
                    f"status=done to poll_pipeline.py (incidents #545, #920). Fix: "
                    f"word the child's completion line without the phase tag, OR "
                    f"redirect the child's stdout to its own log (per-worker "
                    f"pattern: scripts/issue658_8gpu_dispatch.sh), OR waive a "
                    f"mode-gated standalone-lane terminal with "
                    f"'# noqa: phase-done-reserved' on the emission line. See "
                    f".claude/rules/pod-side-reporting.md."
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


# `--check-no-literal-round-marker-versions` (#917): the round-versioned marker
# kinds must never be instructed at a literal ` v1` in checked-in workflow
# prose. `\s+` (not a single space) so a line-wrapped kind/version pair still
# trips under the whole-file scan; `v1\b` does NOT match `v12` (the legitimate
# round-12 example in the /issue SKILL.md).
_LITERAL_ROUND_MARKER_V1_RE = re.compile(
    r"epm:(?:experiment-implementation|results|proposed-tests)\s+v1\b"
)


def check_no_literal_round_marker_versions(*, repo_root: Path | None = None) -> list[str]:
    """FAIL on a literal ``v1`` posting instruction for a round-versioned marker kind.

    The round-versioned kinds — ``epm:experiment-implementation``,
    ``epm:results``, ``epm:proposed-tests`` — accrue rows across follow-up
    rounds / TDD resumes / crash-recovery re-posts, and ``task.py post-marker``
    derives ``max(existing)+1`` only when ``--version`` is OMITTED (an explicit
    ``--version`` always wins, #480). Checked-in prose instructing "post
    ``epm:<kind>`` at ``v1``" teaches orchestrators to compose briefs that pin
    an explicit version 1, which collides with existing rows and silently
    breaks highest-version-wins review-round detection (incident #825: a
    follow-up-round brief instructed a literal v1 on a task already at v6 —
    the #389 collision class). Prose defers to ``v<n>`` / max+1 instead.

    Scan mode: whole-file ``re.finditer`` per file (NOT line-based), so a
    line-wrapped kind/version pair (the kind at end-of-line, ``v1`` on the
    next) still trips via ``\\s+``; ``v1\\b`` does not match ``v12``.

    DELIBERATE boundary: prose evasions like "pass ``--version 1``" are OUT of
    the pattern by design — that layer is covered by the brief-contract prose
    (the /issue SKILL.md Step 4b marker-version-discipline bullet, the Step 9b
    step-3 bullet, and the implementer agents' § Posting review-round markers
    rule), and linting every ``--version 1`` mention would false-positive on
    legitimate incident documentation. A future tightener should know this
    boundary was chosen, not missed.

    Scanned (positive enumeration): ``CLAUDE.md``, ``.claude/workflow.yaml``,
    ``.claude/agents/*.md``, ``.claude/rules/*.md``,
    ``.claude/skills/**/SKILL.md``, ``.claude/skills/issue/markers.md``, and
    ``.claude/skills/issue/templates/*.md``. Everything else — notably
    ``.claude/plans/`` and ``.claude/agent-memory/`` (historical quotes may
    legitimately contain the incident text), skill support/iteration logs,
    and ``scripts/`` / ``src/`` code paths (a separate follow-up covers the
    poller's synthesized-envelope pin) — is excluded by NOT being enumerated.
    ``repo_root`` is a unit-test override hook; production callers pass None.
    Bundled into the no-flags default run.
    """
    root = repo_root if repo_root is not None else _REPO_ROOT
    targets: list[Path] = []
    claude_md = root / "CLAUDE.md"
    if claude_md.is_file():
        targets.append(claude_md)
    wf_yaml = root / ".claude" / "workflow.yaml"
    if wf_yaml.is_file():
        targets.append(wf_yaml)
    for md_dir in (root / ".claude" / "agents", root / ".claude" / "rules"):
        if md_dir.is_dir():
            targets.extend(p for p in sorted(md_dir.glob("*.md")) if p.is_file())
    skills_dir = root / ".claude" / "skills"
    if skills_dir.is_dir():
        targets.extend(p for p in sorted(skills_dir.rglob("SKILL.md")) if p.is_file())
    markers_md = root / ".claude" / "skills" / "issue" / "markers.md"
    if markers_md.is_file():
        targets.append(markers_md)
    templates_dir = root / ".claude" / "skills" / "issue" / "templates"
    if templates_dir.is_dir():
        targets.extend(p for p in sorted(templates_dir.glob("*.md")) if p.is_file())

    errors: list[str] = []
    for p in targets:
        try:
            text = p.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        for m in _LITERAL_ROUND_MARKER_V1_RE.finditer(text):
            lineno = text.count("\n", 0, m.start()) + 1
            matched = " ".join(m.group(0).split())
            errors.append(
                f"{p}:{lineno}: literal round-marker version instruction `{matched}` — "
                f"this kind accrues rows across rounds and an explicit --version beats "
                f"the CLI's max+1 default (incident #825; the #389 collision class). "
                f"Rephrase to `v<n>` / max+1 (omit --version and the CLI derives it); "
                f"see the implementer agents' § Posting review-round markers."
            )
    return errors


# A destructive `git reset --hard` invocation. Whitespace-tolerant, broadened
# (FI1) to catch flag-ordering variants:
#   - `git reset --hard`, `git reset -q --hard`      (flags before --hard)
#   - `git reset --hard origin/main`, `git reset --hard -q`  (flags/ref after)
#   - `git reset origin/main --hard`                 (ref BEFORE the --hard flag)
#   - `git --no-pager reset --hard`                  (git-level flag before subcommand)
#   - `git reset --hard=<ref>`                       (single-token flag, attached value)
# Optional git-level flags (`--no-pager`, `-C <path>`, `-c k=v`) may precede
# `reset`; any tokens (flags or a ref) may sit between `reset` and `--hard`;
# `--hard` may carry an attached `=<value>`.
# The intra-command tokens are backtick-free (``[^\s`]+``, NOT ``\S+``): a real
# git flag / path / ref never contains a backtick, and forbidding backticks
# stops the greedy flag-group from spanning ACROSS an inline-code mention
# (e.g. the prose ``scoped with `git -C`: `git -C "$WT" reset --hard```) — which
# would otherwise anchor the match on the WRONG (prose-mention) `git` and defeat
# the FI3 `-C`-before-match waiver on the sanctioned per-worktree line.
_GIT_RESET_HARD_RE = re.compile(
    r"git\s+(?:--?[^\s`]+(?:\s+[^\s`]+)?\s+)*reset\b(?:\s+[^\s`]+)*?\s+--hard(?:=[^\s`]*)?\b"
)
# A worktree-qualified `git -C "<path>" ...` prefix. Matched separately so we
# can require it appear BEFORE the offending reset on the same line (FI3).
_GIT_DASH_C_RE = re.compile(r"git\s+-C\s+")
_RESET_HARD_ALLOW_SENTINEL = "workflow-lint: allow-git-reset-hard"

# Working-tree-revert doc prescriptions (#897, sibling of _GIT_RESET_HARD_RE).
# Shared backtick-free flag group: optional git-level flags (`--no-pager`,
# `-c k=v`, `-q`) may sit between `git` and the subcommand; backtick-free
# tokens so the greedy group cannot span ACROSS an inline-code mention (the
# same design rationale documented on _GIT_RESET_HARD_RE above).
_GIT_FLAGS_GRP = r"(?:--?[^\s`]+(?:\s+[^\s`]+)?\s+)*"
_WT_REVERT_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    # Any non-`--staged` git restore: explicit-path restore prescriptions have
    # zero legitimate live doc uses; `--staged` forms are index-only (they do
    # not touch the working tree) and exempt. The exemption lookahead is
    # bounded at BOTH a backtick (the inline-code terminator) AND a `#` —
    # bash never executes a comment tail, so a fenced `git restore . #
    # --staged` line is a destructive restore whose comment must NOT waive it
    # (round-2 concern id lint-restore-lookahead-comment-tail; the runtime
    # hook's comment-tail strip is the enforcement sibling). A `--staged`
    # among the real arguments always precedes any `#`, so legitimate
    # index-only prescriptions keep the exemption.
    ("git restore", re.compile(rf"git\s+{_GIT_FLAGS_GRP}restore\b(?![^`#]*--staged)")),
    # Bare-dot wholesale checkout ONLY — explicit `checkout <ref> -- <path>`
    # doc mentions are NOT flagged (legitimate prescriptive uses exist: the
    # /issue Step 5a spec-freshness sync, the Step 10d surgical additive
    # checkout, the code-reviewer smoke-restore rule). At RUNTIME the hook
    # (scripts/guard_repo_root_branch.sh) blocks the explicit-pathspec forms
    # too, with the `git -C` waiver as the deliberate override — the
    # prescriptive-vs-runtime split (#897).
    (
        "git checkout .",
        re.compile(rf"git\s+{_GIT_FLAGS_GRP}checkout\b(?:\s+[^\s`]+)*?\s+\.(?=[\s`/]|$)"),
    ),
    (
        "git clean -f/--force",
        re.compile(
            rf"git\s+{_GIT_FLAGS_GRP}clean\b(?:\s+[^\s`]+)*?\s+(?:-[A-Za-z]*f[A-Za-z]*\b|--force\b)"
        ),
    ),
)
_WT_REVERT_ALLOW_SENTINEL = "workflow-lint: allow-repo-root-wt-revert"


def _line_waived(line: str, match_start: int, sentinel: str) -> bool:
    """True when a flagged destructive-git match on ``line`` is waived.

    Two waivers (shared by the reset-hard + worktree-revert checks):

    - **FI3 worktree-qualified** — a ``git -C`` prefix sits at-or-before
      ``match_start``. In the sanctioned form the offending regex matches from
      the SAME ``git`` the ``-C`` begins (its flag-group swallows ``-C "$WT"``),
      so ``dc.start() == match_start`` there — hence ``<=``, not ``<``. An
      unqualified command starts at a ``git`` NOT followed by ``-C``, so the
      offsets can only coincide for the sanctioned form; a ``git -C`` AFTER the
      match (e.g. ``git reset --hard && git -C "$WT" status``) has a HIGHER
      offset and does NOT waive.
    - **FI2 reasoned sentinel** — the line carries ``sentinel`` with a
      NON-EMPTY reason. The reason lives between the sentinel ``:`` and the
      note closer, so the leading ``:``/whitespace AND the trailing
      HTML-comment closer (``-->``) / backtick / whitespace are stripped before
      testing — otherwise a bare closer (``: -->``, or the sentinel with no
      colon) would count as a reason and wrongly waive.
    """
    dc = _GIT_DASH_C_RE.search(line)
    if dc is not None and dc.start() <= match_start:
        return True
    if sentinel in line:
        _, _, tail = line.partition(sentinel)
        reason = tail.lstrip(": ")
        if reason.rstrip().endswith("-->"):
            reason = reason.rstrip()[: -len("-->")]
        reason = reason.strip().strip("`").strip()
        if reason:
            return True
    return False


def check_no_repo_root_git_reset_hard(*, repo_root: Path | None = None) -> list[str]:
    """FAIL if any agent spec / skill playbook contains an UNQUALIFIED
    destructive ``git reset --hard`` (a repo-root / full-tree reset). Only
    per-worktree ``git -C "$WT" reset --hard`` invocations (the ``-C`` qualifier
    appearing BEFORE the offending reset on the same line), or lines carrying
    the ``workflow-lint: allow-git-reset-hard: <reason>`` sentinel with a
    non-empty reason, pass.

    Incident 2026-07-01 (#815): a #778 analyzer improvised a destructive
    repo-root reset during marker-chain recovery and truncated concurrent
    siblings #812/#813 (body.md / plans/ / comments.jsonl / REGISTRY). task.py
    holds a per-registry flock, not a per-file lock, so a repo-root reset by
    any concurrent session clobbers unrelated tasks.

    Scope: ``.claude/agents/*.md`` + ``.claude/skills/**/SKILL.md`` only (reuses
    ``_iter_ask_target_files``, which already excludes OTHER worktrees' sibling
    copies). ``.claude/plans/``, ``.claude/agent-memory/``, ``.claude/rules/``,
    ``CLAUDE.md``, and ``scripts/**`` are out of the workflow surface for this
    check and NEVER scanned. Pure-Python (no ``rg`` dependency, so the bundled
    pytest is hermetic); bundled into the no-flags default run. ``repo_root`` is
    a unit-test override hook; production callers pass None (canonical repo root).

    INTENTIONAL under-matching (pinned by the test docstring + kill-criteria):
    a ``git reset \\``-continuation line where ``--hard`` lands on the FOLLOWING
    physical line evades the line-based regex. Grep confirms zero live in-scope
    instances; the check is line-oriented by design (markdown scope, NOT a shell
    AST). If a ``\\``-continuation destructive reset ever lands in-scope,
    normalize continuations before matching or split the command.
    """
    root = repo_root if repo_root is not None else _REPO_ROOT
    errors: list[str] = []
    for p in _iter_ask_target_files(root):  # already worktree-safe + scoped
        try:
            text = p.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        for lineno, line in enumerate(text.splitlines(), start=1):
            m = _GIT_RESET_HARD_RE.search(line)
            if m is None:
                continue
            # FI3 `-C`-before-match + FI2 reasoned-sentinel waivers — shared
            # with check_no_repo_root_worktree_revert; semantics documented on
            # the helper (the `<=` covers the sanctioned same-`git` anchor).
            if _line_waived(line, m.start(), _RESET_HARD_ALLOW_SENTINEL):
                continue
            errors.append(
                f"{p}:{lineno}: unqualified `git reset --hard` on the shared "
                f"repo root is forbidden (clobbers concurrent siblings' task "
                f"state — incident #815, commit d29a877e6f). Use a per-worktree "
                f'`git -C "$WT" reset --hard <ref>`, or add a same-line '
                f"`{_RESET_HARD_ALLOW_SENTINEL}: <reason>` sentinel if this is a "
                f"legitimate prose mention / pod-side ssh_execute command."
            )
    return errors


def check_no_repo_root_worktree_revert(*, repo_root: Path | None = None) -> list[str]:
    """FAIL if any agent spec / skill playbook prescribes an UNQUALIFIED
    working-tree revert on the shared repo root: a non-``--staged``
    ``git restore``, a bare-dot wholesale ``git checkout .``, or a
    force-flagged ``git clean``. Only per-worktree ``git -C "$WT" ...`` forms
    (the ``-C`` qualifier appearing BEFORE the match on the same line), or
    lines carrying the ``workflow-lint: allow-repo-root-wt-revert: <reason>``
    sentinel with a non-empty reason, pass.

    Incident 2026-07-02 (#841): a concurrent session's destructive
    working-tree git op on the shared repo root reverted the #841 analyzer's
    uncommitted ``body.md`` mid-task (and deleted untracked pre-registration +
    figure files) — the same hazard class as the #815 repo-root
    ``reset --hard`` (``task.py`` holds a per-registry flock, not per-file).
    This check is the DOC-side sibling of that reset-hard check; the RUNTIME
    tooth is ``scripts/guard_repo_root_branch.sh`` (which additionally blocks
    the explicit-pathspec / bare-pathspec / force checkout forms this check
    deliberately does not flag — legitimate prescriptive doc uses exist for
    those; see ``_WT_REVERT_PATTERNS``).

    Scope: ``.claude/agents/*.md`` + ``.claude/skills/**/SKILL.md`` only
    (reuses ``_iter_ask_target_files``, worktree-sibling-safe). ``.claude/
    plans/``, ``.claude/agent-memory/``, ``.claude/rules/``, ``CLAUDE.md``,
    and ``scripts/**`` are NEVER scanned. Pure-Python; bundled into the
    no-flags default run. ``repo_root`` is a unit-test override hook;
    production callers pass None (canonical repo root).
    """
    root = repo_root if repo_root is not None else _REPO_ROOT
    errors: list[str] = []
    for p in _iter_ask_target_files(root):  # already worktree-safe + scoped
        try:
            text = p.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        for lineno, line in enumerate(text.splitlines(), start=1):
            for label, pattern in _WT_REVERT_PATTERNS:
                m = pattern.search(line)
                if m is None:
                    continue
                if _line_waived(line, m.start(), _WT_REVERT_ALLOW_SENTINEL):
                    continue
                errors.append(
                    f"{p}:{lineno}: unqualified `{label}` (a working-tree "
                    f"revert) on the shared repo root is forbidden — it "
                    f"silently discards CONCURRENT sessions' uncommitted "
                    f"edits (incident #841; #897, sibling of the #815 "
                    f"reset --hard check). Use a per-worktree "
                    f'`git -C "$WT" ...` form, or add a same-line '
                    f"`{_WT_REVERT_ALLOW_SENTINEL}: <reason>` sentinel if "
                    f"this is a legitimate prose mention."
                )
    return errors


def check_compute_shape_review_lens(*, repo_root: Path | None = None) -> list[str]:
    """FAIL if the compute-shape-vs-dispatcher review lens (#806) is absent
    from EITHER code-reviewer agent file.

    Task #779 r6 PASSed a diff whose plan §9 declared 8xH100 data-parallelism
    against a `--gpu-id`-only dispatcher; the 8-GPU pod ran on 1 GPU with 7
    idle (the #664 spend-leak). The fix added a Step 0.67 code-review lens +
    a `compute-shape-mismatch` blocker tag to BOTH the Claude reviewer and its
    Codex twin. This check pins that the lens + tag stay in both files so a
    future refactor cannot silently strip one and re-open the gap (both
    reviewers must face the same bar).

    Two required tokens per file: the Step-0.67 heading marker
    ``Compute-shape-vs-dispatcher`` and the blocker tag literal
    ``compute-shape-mismatch``. ``repo_root`` is a unit-test override hook;
    production callers pass None (canonical repo root). Bundled into the
    no-flags default run.

    #875 extended the lens with the work-conserving schedule sub-check +
    throughput anti-pattern (d) (incidents #813: sequential waves idled 4/8
    H100s 6.7h and per-row ``savez_compressed`` = 65% of wc_long row
    wall-time; #778 phase-3: 1/8 util on 8xH100), pinned by two additional
    tokens: ``work-conserving`` and ``per-row compression``.
    """
    root = repo_root if repo_root is not None else _REPO_ROOT
    required = (
        "Compute-shape-vs-dispatcher",
        "compute-shape-mismatch",
        "work-conserving",  # #875: schedule sub-check (incident #813/#778)
        "per-row compression",  # #875: anti-pattern (d) (incident #813)
    )
    files = (
        root / ".claude" / "agents" / "code-reviewer.md",
        root / ".claude" / "agents" / "codex-code-reviewer.md",
    )
    errors: list[str] = []
    for p in files:
        if not p.is_file():
            errors.append(
                f"{p}: missing — the #806 compute-shape-vs-dispatcher review "
                f"lens must live in both code-reviewer agent files."
            )
            continue
        text = p.read_text(encoding="utf-8")
        for token in required:
            if token not in text:
                errors.append(
                    f"{p}: missing the compute-shape-vs-dispatcher lens token "
                    f"{token!r} (#806/#875). The Step 0.67 lens (exposure contract + "
                    f"work-conserving schedule sub-check) + `compute-shape-mismatch` "
                    f"blocker tag + anti-pattern (d) must be present in BOTH "
                    f"code-reviewer.md and codex-code-reviewer.md so both reviewers "
                    f"catch a plan-declared DP shape the dispatcher does not expose "
                    f"and a non-work-conserving / per-row-IO-bound schedule "
                    f"(incidents #779 r6, #813, #778)."
                )
    return errors


def check_long_loop_restartability_review_lens(*, repo_root: Path | None = None) -> list[str]:
    """FAIL if the long-loop restartability lens (#881) is absent from any of
    its three surfaces.

    Task #823 phase 4 accumulated ~20h (unpatched; ~3.7h patched) of serial
    ridge fits purely in memory with a single terminal JSON write: both GCE
    crashes forfeited all completed fits, a user-directed
    restart-with-optimization was refused solely because restart forfeits
    unpersisted fits, and five code-review rounds never flagged it. The #881
    fix extended the checkpoint-per-phase rule to INTRA-PHASE grain (per-unit
    persistence + a resume predicate for any > ~1h serial loop) and added a
    Step 3.6 review lens to BOTH code-reviewer agent files. This check pins
    all three surfaces so a future refactor cannot silently strip one:

    (a) code-reviewer.md — the ``Long-loop restartability`` Step 3.6 heading
        + the lowercase ``resume predicate`` requirement;
    (b) codex-code-reviewer.md — the same two tokens (the Step-2 copy-list
        bullets) PLUS the inlined-rubric placeholder enumeration
        ``3.5, 3.6, 3.7`` — a token check on the copy-list alone
        false-PASSes while the composed executable Codex prompt still omits
        Step 3.6 (the #606 twin-omission class);
    (c) code-style.md — the ``Intra-phase grain`` extension of the
        checkpoint-per-phase bullet + ``resume predicate``.

    Tokens are case-sensitive substrings, per-file. ``repo_root`` is a
    unit-test override hook; production callers pass None. Bundled into the
    no-flags default run.
    """
    root = repo_root if repo_root is not None else _REPO_ROOT
    required_by_file: dict[Path, tuple[str, ...]] = {
        root / ".claude" / "agents" / "code-reviewer.md": (
            "Long-loop restartability",
            "resume predicate",
        ),
        root / ".claude" / "agents" / "codex-code-reviewer.md": (
            "Long-loop restartability",
            "resume predicate",
            "3.5, 3.6, 3.7",  # the inlined-rubric placeholder enumeration
        ),
        root / ".claude" / "rules" / "code-style.md": (
            "Intra-phase grain",
            "resume predicate",
        ),
    }
    errors: list[str] = []
    for p, required in required_by_file.items():
        if not p.is_file():
            errors.append(
                f"{p}: missing — the #881 long-loop restartability lens must "
                f"live in code-reviewer.md, codex-code-reviewer.md, and "
                f"code-style.md."
            )
            continue
        text = p.read_text(encoding="utf-8")
        for token in required:
            if token not in text:
                errors.append(
                    f"{p}: missing the long-loop restartability lens token "
                    f"{token!r} (#823/#881). The Step 3.6 lens (per-unit "
                    f"persistence + resume predicate for > ~1h serial loops) "
                    f"must be present in code-reviewer.md AND "
                    f"codex-code-reviewer.md (incl. the inlined-rubric "
                    f"placeholder enumeration), and the intra-phase-grain "
                    f"extension of the checkpoint-per-phase bullet in "
                    f"code-style.md (incident #823 phase 4: a ~20h in-memory "
                    f"accumulate-and-write-at-end ridge loop PASSed five "
                    f"review rounds)."
                )
    return errors


def check_hollow_verification_gate_review_lens(*, repo_root: Path | None = None) -> list[str]:
    """FAIL if the hollow-verification-gate lens (#779/#890) is absent from
    any of its three surfaces, or the tag drops off a surface's
    ``**Blocker tags:**`` verdict-template line.

    Task #779: a green `--verify-vectorized` gated an UNUSED helper's
    self-check while the dispatched ridge hot loop (~17k fits, 18-20h) ran
    unverified; rounds 6/7 PASSed. #890 added the Step 0.68
    hollow-verification-gate sub-check + blocker tag to both code-reviewer
    agent files and deferred this parity lint. Three surfaces, per-file
    tokens (the #881 shape):

    (a) code-reviewer.md — the ``Hollow-verification-gate sub-check``
        Step 0.68 heading phrase;
    (b) codex-code-reviewer.md — the lowercase copy-contract phrase
        ``hollow-verification-gate sub-check`` PLUS ``0.68`` on the
        ``{{INLINED RUBRIC`` placeholder line (a copy-list-only token check
        false-PASSes while the composed executable Codex prompt omits
        Step 0.68 — the #606 twin-omission class);
    (c) efficiency-critic.md — the v2 owner
        (.claude/rules/lens-coverage-map.md), IMPLEMENTATION-MODE rubric
        item ``Hollow-verification-gate``.

    Every surface ADDITIONALLY requires the tag on a line starting with
    ``**Blocker tags:**`` — the verdict template's tag-vocabulary line, the
    orchestrator's Step 5c-bis parse target (#890's line-scoped verify:
    a broad grep false-greens a prose-only partial implementation).
    Tokens are case-sensitive substrings. ``repo_root`` is a unit-test
    override hook; production callers pass None. Bundled into the no-flags
    default run.
    """
    root = repo_root if repo_root is not None else _REPO_ROOT
    tag = "hollow-verification-gate"
    blocker_prefix = "**Blocker tags:**"
    prose_by_file: dict[Path, tuple[str, ...]] = {
        root / ".claude" / "agents" / "code-reviewer.md": ("Hollow-verification-gate sub-check",),
        root / ".claude" / "agents" / "codex-code-reviewer.md": (
            "hollow-verification-gate sub-check",
        ),
        root / ".claude" / "agents" / "efficiency-critic.md": ("Hollow-verification-gate",),
    }
    errors: list[str] = []
    for p, required in prose_by_file.items():
        if not p.is_file():
            errors.append(
                f"{p}: missing — the #890 hollow-verification-gate lens must "
                f"live in code-reviewer.md, codex-code-reviewer.md, and "
                f"efficiency-critic.md (the workflow-v2 owner)."
            )
            continue
        text = p.read_text(encoding="utf-8")
        for token in required:
            if token not in text:
                errors.append(
                    f"{p}: missing the hollow-verification-gate lens token "
                    f"{token!r} (#779/#890). The Step 0.68 sub-check (a "
                    f"verify/equivalence gate must assert on the function the "
                    f"entrypoint actually dispatches) must stay on all three "
                    f"reviewer surfaces so a green gate on an unused sibling "
                    f"keeps FAILing review (incident #779: an unverified ~17k-"
                    f"fit hot loop was laundered as verified)."
                )
        bt_lines = [ln for ln in text.splitlines() if ln.startswith(blocker_prefix)]
        if not bt_lines:
            errors.append(
                f"{p}: no line starts with {blocker_prefix!r} (#890) — the "
                f"verdict template's blocker-tag vocabulary line (the Step "
                f"5c-bis parse target) is gone."
            )
        elif not any(tag in ln for ln in bt_lines):
            errors.append(
                f"{p}: no {blocker_prefix!r} line names {tag!r} (#890) — the "
                f"tag dropped out of the verdict template's vocabulary; a "
                f"reviewer could no longer declare the finding (the "
                f"2-without-2b partial-implementation class a broad grep "
                f"false-greens)."
            )
    codex = root / ".claude" / "agents" / "codex-code-reviewer.md"
    if codex.is_file():
        rubric_lines = [
            ln for ln in codex.read_text(encoding="utf-8").splitlines() if "{{INLINED RUBRIC" in ln
        ]
        if not any("0.68" in ln for ln in rubric_lines):
            errors.append(
                f"{codex}: '0.68' is absent from the '{{{{INLINED RUBRIC' "
                f"placeholder line (#890) — the composed Codex prompt would "
                f"omit the Step 0.68 named-helper + hollow-gate lens (the "
                f"#606 twin-omission class; same pin as #822's '0.55' and "
                f"#881's '3.5, 3.6, 3.7')."
            )
    return errors


def check_smoke_architecture_review_lens(*, repo_root: Path | None = None) -> list[str]:
    """FAIL if the smoke-architecture marker presence gate (#822) is absent
    from ANY of its three surfaces.

    Task #811's implementer claimed the smoke-architecture verdict in prose (a
    dispatcher header) but never posted the separate
    `epm:smoke-architecture-check` events row; both reviewers PASSed 5 rounds
    and the gap surfaced only at /issue Step 6d.0 AFTER pod provisioning. The
    fix added a Step 0.55 presence lens to BOTH code-reviewer agent files and
    a per-blocker `marker-shape` sub-recipe to the /issue Step 5c-bis strip.
    This check pins all three surfaces, region-anchored, so a future refactor
    cannot silently strip one and re-open the gap:

    (a) code-reviewer.md — a ``### Step 0.55`` section whose body (up to the
        next ``### `` heading) names ``epm:smoke-architecture-check``;
    (b) codex-code-reviewer.md — the Step 0.55 copy-list bullet (heading
        literal + marker kind) AND ``0.55`` on the ``{{INLINED RUBRIC``
        placeholder line;
    (c) `.claude/skills/issue/SKILL.md` — the Step 5c-bis region (between the
        ``**5c-bis.`` and ``**5c-ter.`` headings) names
        ``epm:smoke-architecture-check``.

    ``repo_root`` is a unit-test override hook; production callers pass None
    (canonical repo root). Bundled into the no-flags default run.
    """
    root = repo_root if repo_root is not None else _REPO_ROOT
    marker = "epm:smoke-architecture-check"
    errors: list[str] = []

    # (a) code-reviewer.md: the Step 0.55 section body names the marker.
    reviewer = root / ".claude" / "agents" / "code-reviewer.md"
    if not reviewer.is_file():
        errors.append(
            f"{reviewer}: missing — the #822 smoke-architecture marker "
            f"presence gate (Step 0.55) must live in code-reviewer.md."
        )
    else:
        text = reviewer.read_text(encoding="utf-8")
        idx = text.find("### Step 0.55")
        if idx == -1:
            errors.append(
                f"{reviewer}: missing the '### Step 0.55' section (#822). The "
                f"smoke-architecture marker presence gate must stay in the "
                f"Claude reviewer so a missing {marker} events row FAILs at "
                f"code-review, not at Step 6d.0 post-provision (incident #811)."
            )
        else:
            nxt = text.find("\n### ", idx + 1)
            body = text[idx:nxt] if nxt != -1 else text[idx:]
            if marker not in body:
                errors.append(
                    f"{reviewer}: the '### Step 0.55' section body no longer "
                    f"names {marker!r} (#822) — the presence gate must key on "
                    f"that exact marker kind."
                )

    # (b) codex-code-reviewer.md: the copy-list bullet + rubric placeholder.
    codex = root / ".claude" / "agents" / "codex-code-reviewer.md"
    if not codex.is_file():
        errors.append(
            f"{codex}: missing — the #822 smoke-architecture marker presence "
            f"gate (Step 0.55 copy-list bullet) must live in "
            f"codex-code-reviewer.md."
        )
    else:
        text = codex.read_text(encoding="utf-8")
        heading = '"Step 0.55: Smoke-architecture marker presence gate"'
        for token in (heading, marker):
            if token not in text:
                errors.append(
                    f"{codex}: missing the Step 0.55 copy-list token {token!r} "
                    f"(#822) — the Codex twin must copy the same presence lens "
                    f"or the two reviewers drift (the #606 copy-list-omission "
                    f"class)."
                )
        idx = text.find(heading)
        if idx != -1 and marker in text:
            nxt = text.find('\n- "', idx + 1)
            bullet = text[idx:nxt] if nxt != -1 else text[idx:]
            if marker not in bullet:
                errors.append(
                    f"{codex}: the Step 0.55 copy-list bullet (heading token "
                    f"to the next line-start '- \"' bullet) no longer names "
                    f"{marker!r} (#822) — a marker mention elsewhere in the "
                    f"file (e.g. the Step 7 blocker-tags line) does not keep "
                    f"the copied lens itself keyed on that marker."
                )
        rubric_lines = [ln for ln in text.splitlines() if "{{INLINED RUBRIC" in ln]
        if not any("0.55" in ln for ln in rubric_lines):
            errors.append(
                f"{codex}: '0.55' is absent from the '{{{{INLINED RUBRIC' "
                f"placeholder line (#822) — the composed Codex prompt would "
                f"omit the Step 0.55 lens."
            )

    # (c) SKILL.md: the Step 5c-bis strip region names the marker sub-recipe.
    skill = root / ".claude" / "skills" / "issue" / "SKILL.md"
    if not skill.is_file():
        errors.append(
            f"{skill}: missing — the #822 Step 5c-bis per-blocker "
            f"marker-shape sub-recipe must live in the /issue skill."
        )
    else:
        text = skill.read_text(encoding="utf-8")
        start = text.find("**5c-bis.")
        end = text.find("**5c-ter.")
        region = text[start:end] if (start != -1 and end != -1 and end > start) else ""
        if marker not in region:
            errors.append(
                f"{skill}: the Step 5c-bis region (between '**5c-bis.' and "
                f"'**5c-ter.') no longer names {marker!r} (#822) — the "
                f"mechanical-contract strip needs the per-blocker sub-recipe "
                f"to distinguish a stale-worktree false absence (STRIP) from "
                f"a genuine one (leave the FAIL in place)."
            )
    return errors


# The #963 stale-label disposition-clause tokens. The paragraph span runs from
# the bold anchor (which must be UNIQUE — the check carries a NEGATIVE
# assertion, so span identity is load-bearing) to the first blank line, and is
# whitespace-normalized before matching (two tokens span a hard line wrap in
# the live file, and an innocent prose reflow must not FAIL the fleet).
_STALE_LABEL_ANCHOR = "**Stale-label disposition rule"
_STALE_LABEL_REQUIRED_TOKENS = (
    "followup_retro_close_evidence",
    "GHOST-label filter, NOT an execution gate",
    "A None return means NO prior-run evidence exists",
    "the label EXECUTES as the dispatched round",
    "The skip-and-surface disposition applies ONLY when",
)
_STALE_LABEL_SKIP_ON_NONE_RE = re.compile(r"\bon\s+(?:a\s+)?none\b.{0,120}?\bskip", re.IGNORECASE)


def check_stale_label_disposition_clause(*, repo_root: Path | None = None) -> list[str]:
    """FAIL if the /issue Step 0 stale-label disposition paragraph (#894/#763)
    loses its fresh-label-execute semantics or regains an unconditional
    skip-on-None branch (#963).

    Scope notes (round-1 critique):

    (a) The negative regex is a LITERAL-COUPLING BACKSTOP only — phrasings
        like "when None is returned, skip" are covered by the positive
        tokens, not the regex; do not weaken a positive token "because the
        regex covers it".
    (b) The check is paragraph-scoped — a contradictory instruction written
        OUTSIDE the anchored paragraph is invisible to it (inherent to the
        token-lint class).
    (c) A mid-paragraph blank line truncates the span and FAILs all
        downstream tokens at once — the span ends at the first blank line,
        so a deliberate restructure requires a deliberate lint update.

    ``repo_root`` is a unit-test override hook; production callers pass None
    (canonical repo root). Bundled into the no-flags default run.
    """
    root = repo_root if repo_root is not None else _REPO_ROOT
    skill = root / ".claude" / "skills" / "issue" / "SKILL.md"
    if not skill.is_file():
        return [f"{skill}: missing — the Step 0 stale-label disposition paragraph must exist."]
    text = skill.read_text(encoding="utf-8")
    n_anchors = text.count(_STALE_LABEL_ANCHOR)
    if n_anchors == 0:
        return [
            f"{skill}: missing the bold anchor {_STALE_LABEL_ANCHOR!r} (#963) — the Step 0 "
            f"stale-label disposition paragraph pins the #894/#763 fresh-label-execute "
            f"semantics and must not be removed or renamed without updating this lint."
        ]
    if n_anchors > 1:
        return [
            f"{skill}: {n_anchors} bold anchors {_STALE_LABEL_ANCHOR!r} found — the stale-label "
            f"disposition paragraph must be UNIQUE (a stale duplicate could satisfy the token "
            f"scan while the operative Step 0 paragraph regresses; #963). Remove the duplicate."
        ]
    start = text.find(_STALE_LABEL_ANCHOR)
    end = text.find("\n\n", start)
    normalized = re.sub(r"\s+", " ", text[start : end if end != -1 else len(text)])
    errors: list[str] = []
    for token in _STALE_LABEL_REQUIRED_TOKENS:
        if token not in normalized:
            errors.append(
                f"{skill}: stale-label disposition paragraph missing token {token!r} (#963) — "
                f"note: the span ends at the first blank line, so a split paragraph FAILs all "
                f"downstream tokens at once (a deliberate restructure needs a lint update)."
            )
    if _STALE_LABEL_SKIP_ON_NONE_RE.search(normalized):
        errors.append(
            f"{skill}: stale-label disposition paragraph couples a None return to a skip "
            f"instruction ('On None ... skip') — a fresh never-run label must EXECUTE as the "
            f"dispatched round (#963); the skip-and-surface disposition is reserved for "
            f"suspected-stale ghost labels."
        )
    return errors


# The #842 smoke output-path hygiene anchor phrase. Must appear INSIDE each
# surface's load-bearing region (see check_smoke_output_hygiene below).
SMOKE_OUTPUT_HYGIENE_ANCHOR = "Smoke outputs never overwrite committed artifacts"


def check_smoke_output_hygiene(*, repo_root: Path | None = None) -> list[str]:
    """FAIL if the smoke output-path hygiene rule (#842) is absent from any
    of its three surfaces — REGION-AWARE and WHITESPACE-NORMALIZED.

    Incident #722 (2026-07-02, three instances): a `--layers 0 18` review
    smoke truncated committed 28-layer eval JSONs+figures; a reviewer's
    pytest rerun regenerated `figures/issue_722/mlp_*.png` at smoke scale at
    canonical paths; and the round-2 hero figure shipped as a 2-layer smoke
    version because the script's figure path was not `_smoke`-suffixed while
    its JSONs diverted. The fix added the anchor rule ("Smoke outputs never
    overwrite committed artifacts") to three surfaces; this check pins each
    copy inside its load-bearing, heading-bounded region:

    (a) experiment-implementer.md — the smoke-contract checklist item
        (``N. **End-to-end smoke run PER PHASE.`` up to the next
        top-level numbered item);
    (b) code-reviewer.md — the Step 0.6 region (``### Step 0.6:`` up to the
        next heading) — the ONLY reviewer step the Codex twin's
        ``{{INLINED RUBRIC`` placeholder carries, so a copy that drifts out
        of Step 0.6 silently loses ensemble coverage;
    (c) `.claude/skills/issue/SKILL.md` — the Step 5 smoke-gate paragraph
        (``**End-to-end smoke gate (experiment tasks).`` up to the next
        bold-start line / heading).

    Whole-file presence is NOT enough: a surviving cross-reference elsewhere
    in the file must not false-green after the rule body is deleted, so the
    anchor must sit inside the named region. Matching is whitespace-
    normalized (``\\s+`` -> single space) so an innocent prose reflow that
    hard-wraps the anchor cannot spuriously FAIL the default run. A missing
    region heading FAILs loud (a restructure must re-anchor the rule AND
    update the region regex here in the same commit). ``repo_root`` is a
    unit-test override hook; production callers pass None (canonical repo
    root). Bundled into the no-flags default run.
    """
    root = repo_root if repo_root is not None else _REPO_ROOT

    def _norm(s: str) -> str:
        return re.sub(r"\s+", " ", s)

    # End regexes are anchored on a literal ``\n`` (NOT a MULTILINE ``^``):
    # the end search runs on the text slice AFTER the start match, and ``^``
    # would also match at the very start of that slice (e.g. the closing
    # ``**`` of the start line's bold token), truncating the region to zero.
    surfaces: tuple[tuple[Path, str, str, str], ...] = (
        (
            root / ".claude" / "agents" / "experiment-implementer.md",
            r"^\d+\. \*\*End-to-end smoke run PER PHASE\.",
            r"\n\d+\. \*\*",
            "implementer smoke-contract checklist item",
        ),
        (
            root / ".claude" / "agents" / "code-reviewer.md",
            r"^### Step 0\.6:",
            r"\n#{1,6} ",
            "code-reviewer Step 0.6 region",
        ),
        (
            root / ".claude" / "skills" / "issue" / "SKILL.md",
            r"^\*\*End-to-end smoke gate \(experiment tasks\)\.",
            r"\n\*\*|\n#{1,6} ",
            "SKILL.md Step 5 smoke-gate paragraph",
        ),
    )
    errors: list[str] = []
    for path, start_re, end_re, name in surfaces:
        if not path.is_file():
            errors.append(
                f"{path}: missing — the #842 smoke output-path hygiene rule "
                f"({SMOKE_OUTPUT_HYGIENE_ANCHOR!r}) must live here, in the "
                f"{name}."
            )
            continue
        text = path.read_text(encoding="utf-8")
        start_m = re.search(start_re, text, flags=re.MULTILINE)
        if start_m is None:
            errors.append(
                f"{path}: region heading for the {name} not found (#842) — "
                f"the anchor region was restructured; re-anchor the smoke "
                f"output-path hygiene rule and update this check's region "
                f"regex in the same commit."
            )
            continue
        end_m = re.search(end_re, text[start_m.end() :])
        end = start_m.end() + end_m.start() if end_m is not None else len(text)
        region = text[start_m.start() : end]
        if _norm(SMOKE_OUTPUT_HYGIENE_ANCHOR) not in _norm(region):
            errors.append(
                f"{path}: anchor {SMOKE_OUTPUT_HYGIENE_ANCHOR!r} absent from "
                f"the {name} (#842) — the rule was dropped or moved out of "
                f"its load-bearing region; smoke runs would silently clobber "
                f"committed eval_results/ / figures/ artifacts again "
                f"(incident #722)."
            )
    return errors


_VM_THREAD_CAP_PREFIX = (
    "OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8"
)

# {file: minimum occurrence count of the literal prefix}. Count floors (not
# bare presence) so stripping the prefix from ONE template instance while a
# prose mention survives still FAILs (Methodology + Statistics critic
# hardening; raised to Must-Fix by the Codex alternatives critic, round 1):
# SKILL.md 1 (detached-launch template), experiment-implementer.md 2 (bullet
# + setsid line), code-style.md 3 (line-20 bullet + the two § nohup template
# copies), analyzer-section-reference.md 1 (off-pod template).
# BINDING CONVENTION (keeps the floors template-anchored): rationale PROSE in
# the pinned files refers to the caps by the shorthand
# "OMP/MKL/OPENBLAS/NUMEXPR=8" and NEVER spells the full literal prefix, so
# every literal occurrence is a copy-pastable command/template instance and
# the floors bind to templates, not paragraphs. (The experiment-implementer.md
# bullet's quoted command string counts as a copy-paste instance by design.)
# The literal stays UNSPLIT on one physical line at every occurrence — a
# hard-wrapped prefix breaks both the count pin and copy-paste (loud lint
# FAIL, by design).
_VM_THREAD_CAP_GUIDANCE_FILES = {
    Path(".claude") / "skills" / "issue" / "SKILL.md": 1,
    Path(".claude") / "agents" / "experiment-implementer.md": 2,
    Path(".claude") / "rules" / "code-style.md": 3,
    Path(".claude") / "rules" / "analyzer-section-reference.md": 1,
}


def check_vm_thread_cap_guidance(*, repo_root: Path | None = None) -> list[str]:
    """FAIL if the shared-VM thread-cap launch prefix (#891) is absent from any
    VM-side launch-guidance surface.

    The #847 setdefault in ``orchestrate/env.py`` is src/-side and pinned to a
    worktree's branch point (the Step 5a spec-freshness sync is deliberately
    specs-only), so the workflow's VM-side launch templates carry the explicit
    four-var cap prefix as the branch-age-independent fallback (incident #779,
    2026-07-02: a pre-#847 worktree ran 78 uncapped threads ~20h after the fix
    landed on main). This check pins the LITERAL prefix — with a per-file
    occurrence-count floor, so stripping it from a TEMPLATE instance while a
    prose mention survives still fails — in the four guidance surfaces, making
    a silent re-open of the gap loud. (Residual: an edit swapping which LINE
    carries an occurrence at equal count passes; the count floor is the
    granularity/robustness trade the plan accepts.)
    The value 8 is deliberately coupled to ``_DEFAULT_VM_THREAD_CAP`` in
    env.py: changing either requires changing both (and this constant), which
    is the point — drift fails loud. ``repo_root`` is a unit-test override
    hook; production callers pass None. Bundled into the no-flags default run.
    """
    root = repo_root if repo_root is not None else _REPO_ROOT
    errors: list[str] = []
    for rel, min_count in _VM_THREAD_CAP_GUIDANCE_FILES.items():
        p = root / rel
        if not p.is_file():
            errors.append(
                f"{p}: missing — the #891 shared-VM thread-cap launch guidance "
                f"must live in all four VM-launch surfaces."
            )
            continue
        n = p.read_text(encoding="utf-8").count(_VM_THREAD_CAP_PREFIX)
        if n < min_count:
            errors.append(
                f"{p}: {n} occurrence(s) of the shared-VM thread-cap prefix "
                f"{_VM_THREAD_CAP_PREFIX!r}, expected >= {min_count} (#891). "
                f"VM-side launch TEMPLATES must carry explicit caps: a stale "
                f"worktree's env.py setdefault (#847) may predate the fix, and "
                f"no env.py version can in-process-cap a torch-before-dotenv "
                f"importer."
            )
    return errors


# `--check-lessons-index`: every `.claude/rules/*.md` (except LESSONS.md
# itself) must have exactly one matching row in `.claude/rules/LESSONS.md`, and
# every row in LESSONS.md must point at an existing rule file. Closes the
# silent-drift class: a rule added/removed without an index update would
# otherwise re-open the #722 load-timing gap (a lesson with no always-on index
# row). The row format is the stable, machine-parseable:
#   - **<name>** ([`.claude/rules/<name>.md`](<name>.md)) — fires when: ...
_LESSONS_ROW_RE = re.compile(
    r"^- \*\*(?P<name>[a-z0-9-]+)\*\* \(\[`\.claude/rules/(?P=name)\.md`\]"
    r"\((?P=name)\.md\)\)",
    re.MULTILINE,
)


_LESSONS_MAX_BYTES = (
    8000  # leanness cap: ~2000 tokens always-on (7500->8000, #869/#872 coordinated raise)
)


def check_lessons_index(*, repo_root: Path | None = None) -> list[str]:
    """FAIL if `.claude/rules/LESSONS.md` and the `.claude/rules/*.md` set
    diverge OR the index exceeds the leanness cap.

    The always-on index (#739) must name every rule so each lesson is known at
    plan time even before its `paths:` glob matches an open file. Four failure
    modes are checked: (a) a rule file with no index row, (b) an index row
    with no rule file, (c) a rule name with MORE THAN ONE index row (the
    contract is exactly one matching row per rule — a duplicate would let one
    of the rows silently drift), (d) the index exceeds `_LESSONS_MAX_BYTES`
    (the always-on token budget — the whole point of the index is leanness;
    the Option-B rejected alternative was inlining all rule bodies, paying
    tens of K tokens per call). `repo_root` is a unit-test override hook;
    production callers pass None (canonical repo root). Bundled into the
    no-flags default run.
    """
    root = repo_root if repo_root is not None else _REPO_ROOT
    rules_dir = root / ".claude" / "rules"
    lessons = rules_dir / "LESSONS.md"
    errors: list[str] = []
    if not lessons.is_file():
        errors.append(
            f"{lessons}: missing — the always-on lessons index (#739) must "
            f"exist and index every .claude/rules/*.md file."
        )
        return errors
    raw = lessons.read_bytes()
    if len(raw) > _LESSONS_MAX_BYTES:
        errors.append(
            f".claude/rules/LESSONS.md: {len(raw)} bytes exceeds the "
            f"{_LESSONS_MAX_BYTES}-byte leanness cap. The index is "
            f"always-on; trim 'fires when:' triggers until it fits. "
            f"(em-dashes are multibyte; counting in BYTES not chars is "
            f"deliberate.)"
        )
    # Count occurrences (not a set) so a name appearing on >1 row is caught —
    # a set comprehension would collapse duplicates and let both the missing
    # and stale set-diffs read empty, silently passing the check (#739 r2).
    index_counts = Counter(m.group("name") for m in _LESSONS_ROW_RE.finditer(raw.decode("utf-8")))
    indexed = set(index_counts)
    rule_files = {p.stem for p in rules_dir.glob("*.md") if p.is_file() and p.name != "LESSONS.md"}
    for missing in sorted(rule_files - indexed):
        errors.append(
            f".claude/rules/LESSONS.md: no index row for rule "
            f"'{missing}' (.claude/rules/{missing}.md). Add a "
            f"'- **{missing}** ([`.claude/rules/{missing}.md`]"
            f"({missing}.md)) — fires when: ...' row."
        )
    for stale in sorted(indexed - rule_files):
        errors.append(
            f".claude/rules/LESSONS.md: index row for '{stale}' has no "
            f"matching .claude/rules/{stale}.md file — remove the row or "
            f"restore the rule."
        )
    for dup, count in sorted(index_counts.items()):
        if count > 1:
            errors.append(
                f".claude/rules/LESSONS.md: rule '{dup}' has {count} index "
                f"rows — the contract is exactly one matching row per rule. "
                f"Remove the duplicate row(s) for '{dup}'."
            )
    return errors


# Agent-spec size budget (#829, tightened #838): every .claude/agents/*.md is
# loaded whole on each spawn of that agent, so spec size is a per-invocation
# token cost. WARN above 28 KB (drifting), FAIL above 40 KB (relocate
# per-scenario content to .claude/rules/ on-demand rules; planner.md /
# critic.md are the #838 worked examples). Thresholds are STRICTLY-GREATER (a
# file at exactly the threshold passes). #838 probe grounding: the shared
# session pile alone measured ~125K tokens in an MCP-heavy session, so a
# 63.9 KB spec left planner/critic spawns only ~39-48K tokens of headroom and
# they autocompact-thrashed (#833/#834); 40 KB is the probe decision-table
# band floor (B_safe < 20K there — see tasks .../838/artifacts/
# spawn-baseline-probe.md).
AGENT_SPEC_WARN_BYTES = 28_000
AGENT_SPEC_FAIL_BYTES = 40_000

# Grandfather-ratchet caps for agent specs still above AGENT_SPEC_FAIL_BYTES.
# Each cap = measured size + <=3 KB margin (post-#829 for the first two
# entries; at the #838 FAIL tightening 70K -> 40K for the rest); a
# grandfathered file FAILs above its cap (regrowth ratchet) and FAILs as stale
# once it drops to <= AGENT_SPEC_FAIL_BYTES ("remove the entry"). Ratchet DOWN
# when trimmed. planner.md and critic.md are deliberately NOT grandfathered
# (#838): both were structurally trimmed to <=20 KB, so regrowth on the two
# incident files is a commit-time FAIL.
AGENT_SPEC_SIZE_GRANDFATHER: dict[str, int] = {
    # measured 104,135 B post-#829; fifteen-lenses core is every-markdown-review
    # load-bearing — SPEC.md-dedup trim is the #829 named follow-up
    "clean-result-critic.md": 107_000,
    # the rest measured at the #838 tightening (2026-07-02), caps = measured
    # + <=3 KB; each names a future trim direction, none is licensed to grow
    # measured 91,371 B post-#948 (Step 3.8 seam-stubbed production-body
    # verification lens + Rule 16 + Step 0.68 sibling xref — plan-mandated
    # growth; cap = measured + <=~1 KB. Prior: 82,176 B post-#875+#869+#881)
    # #948: seam-stubbed production-body lens (Step 3.8)
    "code-reviewer.md": 92_300,
    "codex-clean-result-critic.md": 62_000,  # measured 59,358 B
    # measured 50,642 B post-#948 (Step 3.8 copy-list bullet + the
    # inlined-rubric 3.8 slot — plan-mandated growth; cap = measured
    # + <=~1 KB. Prior: 47,930 B post-#881)
    # #948: seam-stubbed production-body lens (Step 3.8)
    "codex-code-reviewer.md": 51_600,
    # measured 58,976 B post-#936 (the plan-REQUIRED bf16 equivalence-gate
    # calibration caveat in § Batched-rewrite equivalence — plan-mandated
    # growth; cap = measured + <=3 KB. Prior: 55,812 B post-#869)
    "experiment-implementer.md": 61_500,
    "experimenter.md": 65_500,  # measured 62,672 B
    "methodology-writer.md": 48_000,  # measured 45,203 B
    "research-pm.md": 43_500,  # measured 40,990 B
    "upload-verifier.md": 45_500,  # measured 42,825 B
}


def check_agent_spec_size(
    *, repo_root: Path | None = None, warn_sink: list[str] | None = None
) -> list[str]:
    """WARN/FAIL agent specs (`.claude/agents/*.md`) over the size budget (#829).

    Every agent spec is loaded whole on each spawn, so bytes here are a
    per-invocation token cost. Semantics (all thresholds STRICTLY-GREATER):
    size > ``AGENT_SPEC_FAIL_BYTES`` FAILs unless the file is grandfathered in
    ``AGENT_SPEC_SIZE_GRANDFATHER`` (then it WARNs while under its per-file cap
    and FAILs above it — the regrowth ratchet); size > ``AGENT_SPEC_WARN_BYTES``
    WARNs. Grandfather hygiene FAILs a stale entry (file missing) and an entry
    whose file dropped to <= the FAIL threshold (remove the entry — ratchet
    down), and a config self-check FAILs any cap <= the FAIL threshold. WARNs
    go to ``warn_sink`` when provided (unit-test hook), else stderr with a
    ``WARN: `` prefix; WARNs never enter the returned FAIL list. ``repo_root``
    is a unit-test override; production callers pass None. Bundled into the
    no-flags default run.
    """
    root = repo_root if repo_root is not None else _REPO_ROOT
    agents_dir = root / ".claude" / "agents"
    errors: list[str] = []

    def _warn(msg: str) -> None:
        if warn_sink is not None:
            warn_sink.append(msg)
        else:
            sys.stderr.write(f"WARN: {msg}\n")

    if not agents_dir.is_dir():
        errors.append(
            f"{agents_dir}: missing — the agent-spec dir must exist for the "
            f"agent-spec size-budget check (#829)."
        )
        return errors

    # Config self-check FIRST: a grandfather cap at/below the FAIL threshold is
    # meaningless (the plain FAIL branch would never be reached for that file).
    for gf_name, cap in sorted(AGENT_SPEC_SIZE_GRANDFATHER.items()):
        if cap <= AGENT_SPEC_FAIL_BYTES:
            errors.append(
                f"AGENT_SPEC_SIZE_GRANDFATHER['{gf_name}']: cap {cap} — cap "
                f"must exceed AGENT_SPEC_FAIL_BYTES ({AGENT_SPEC_FAIL_BYTES}); "
                f"raise the cap or remove the entry."
            )

    for path in sorted(agents_dir.glob("*.md")):
        if not path.is_file():
            continue
        size = path.stat().st_size
        name = path.name
        if size > AGENT_SPEC_FAIL_BYTES:
            cap = AGENT_SPEC_SIZE_GRANDFATHER.get(name)
            if cap is not None:
                if size > cap:
                    errors.append(
                        f".claude/agents/{name}: {size} bytes exceeds its "
                        f"grandfather ratchet cap ({cap} bytes) — the spec "
                        f"regrew past its recorded post-trim size; trim it "
                        f"back (relocate per-scenario content to "
                        f".claude/rules/, see #829)."
                    )
                else:
                    _warn(
                        f".claude/agents/{name}: {size} bytes — grandfathered; "
                        f"{cap - size} bytes under its cap ({cap})."
                    )
            else:
                errors.append(
                    f".claude/agents/{name}: {size} bytes exceeds the "
                    f"{AGENT_SPEC_FAIL_BYTES}-byte agent-spec FAIL threshold — "
                    f"relocate per-scenario content to .claude/rules/ "
                    f"(see #829)."
                )
        elif size > AGENT_SPEC_WARN_BYTES:
            _warn(
                f".claude/agents/{name}: {size} bytes exceeds the "
                f"{AGENT_SPEC_WARN_BYTES}-byte agent-spec WARN budget "
                f"(FAIL above {AGENT_SPEC_FAIL_BYTES})."
            )

    # Grandfather-entry hygiene: entries must point at existing files that
    # still NEED grandfathering (size > FAIL threshold).
    for gf_name in sorted(AGENT_SPEC_SIZE_GRANDFATHER):
        gf_path = agents_dir / gf_name
        if not gf_path.is_file():
            errors.append(
                f"AGENT_SPEC_SIZE_GRANDFATHER['{gf_name}']: stale grandfather "
                f"entry — .claude/agents/{gf_name} does not exist; remove the "
                f"entry."
            )
        elif gf_path.stat().st_size <= AGENT_SPEC_FAIL_BYTES:
            errors.append(
                f"AGENT_SPEC_SIZE_GRANDFATHER['{gf_name}']: "
                f".claude/agents/{gf_name} is {gf_path.stat().st_size} bytes "
                f"(<= {AGENT_SPEC_FAIL_BYTES}) and no longer needs "
                f"grandfathering — remove the entry (ratchet down)."
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


# ── --check-api-dispatch-routing (workflow v2, plan §5) ──────────────────────
# All Anthropic API calls (judges + generation) should route through the
# multi-org dispatcher src/explore_persona_space/llm/api_dispatch.py so
# cross-session headroom routing + AIMD back-off + batch-vs-sync apply. A NEW
# direct call site is a bypass. The routing / sanctioned-client layer is exempt;
# scripts/archive/** is frozen dead code and excluded wholesale; the current
# tree's existing direct callers are grandfathered below (enumerated when the
# check landed via the SAME AST predicate the check uses). A NEW file that
# constructs anthropic.Anthropic(...) / .AsyncAnthropic(...) or calls
# <client>.messages[...].create(...) and is not exempt/grandfathered FAILs.
# Waive a genuinely-correct non-dispatcher caller with a file-level
# '# API_DISPATCH_ROUTING_EXEMPT: <reason>' comment. Adding to the grandfather
# set requires an inline reason (a new direct caller should almost always be a
# dispatcher route instead). Mirrors JUDGE_PIN_LEGACY_ALLOWLIST's style.
API_DISPATCH_ROUTING_LAYER: frozenset[str] = frozenset(
    {"api_dispatch.py", "judge_dispatch.py", "batch_judge.py", "anthropic_client.py"}
)
API_DISPATCH_ROUTING_WAIVER = "API_DISPATCH_ROUTING_EXEMPT"
API_DISPATCH_ROUTING_ALLOWLIST: frozenset[str] = frozenset(
    {
        "scripts/analyze_axis_tails.py",
        "scripts/build_canonical_persona_pool.py",
        "scripts/build_i181_data.py",
        "scripts/eval_language_inversion.py",
        "scripts/eval_source_persona_issue112.py",
        "scripts/gen_issue475_scaffold_data.py",
        "scripts/generate_a3_data.py",
        "scripts/generate_issue356_data.py",
        "scripts/generate_issue376_marker_install.py",
        "scripts/generate_issue404_json_neg.py",
        "scripts/generate_leakage_data.py",
        "scripts/generate_sdf_neutral_ai.py",
        "scripts/generate_sdf_variants.py",
        "scripts/generate_trait_transfer_data_v2.py",
        "scripts/i504_probe_bank_geometry.py",
        "scripts/i528_phase0_preflight.py",
        "scripts/i528_phase1_generate_RNeg.py",
        "scripts/i528_phase1_generate_RPos.py",
        "scripts/i528_phase2_smoke_judge.py",
        "scripts/i528_phase4_judge.py",
        "scripts/issue404_outcome_eval.py",
        "scripts/issue404_predictor_incontext.py",
        "scripts/issue404_predictor_kldiv.py",
        "scripts/issue502_generate_probes.py",
        "scripts/issue545_judge_refusal_diag.py",
        "scripts/issue559_cross_behavior_self_scoring.py",
        "scripts/issue594_build_battery.py",
        "scripts/issue623_extract_sycophancy_vector.py",
        "scripts/issue658_judge_e0.py",
        "scripts/issue661_freeze_instructions.py",
        "scripts/issue779_common.py",
        "scripts/issue_188_evolutionary_trigger.py",
        "scripts/issue_331_phase1_evolutionary.py",
        "scripts/issue_642/i642_common.py",
        "scripts/issue_653/i653_dispatch.py",
        "scripts/judge_with_claude.py",
        "scripts/poll_lmsys_taxonomy.py",
        "scripts/reanalyze_issue444_5way.py",
        "scripts/regenerate_issue404_medical.py",
        "scripts/rejudge_issue_389_c_strict.py",
        "scripts/run_a3_leakage.py",
        "scripts/run_a3b_experiment.py",
        "scripts/run_em_multiseed.py",
        "scripts/run_experiment_389.py",
        "scripts/run_experiment_444.py",
        "scripts/run_issue_156.py",
        "scripts/run_issue_203.py",
        "scripts/run_issue_213_part_b.py",
        "scripts/run_parallel_jobs.py",
        "scripts/run_proximity_transfer.py",
        "scripts/translate_to_italian.py",
        "scripts/translate_ultrachat.py",
        "scripts/validate_italian_translation.py",
        "scripts/validate_translation.py",
        "src/explore_persona_space/eval/alignment.py",
        "src/explore_persona_space/eval/belief.py",
        "src/explore_persona_space/eval/refusal.py",
        "src/explore_persona_space/experiments/behavior_testbed_545/corpora.py",
        "src/explore_persona_space/experiments/behavior_testbed_545/judges_545.py",
        "src/explore_persona_space/experiments/contrastive_neg_geometry_472/persona_bank.py",
        "src/explore_persona_space/experiments/issue503/advbench_judge.py",
        "src/explore_persona_space/experiments/issue503/broad_syco_dataset.py",
        "src/explore_persona_space/experiments/issue503/topic_strip.py",
        "src/explore_persona_space/experiments/issue559/judge_rubrics.py",
        "src/explore_persona_space/experiments/issue_823/run_823.py",
        "src/explore_persona_space/experiments/sycophancy_onpolicy_612/claim_audit.py",
        "src/explore_persona_space/experiments/sycophancy_onpolicy_612/judge.py",
        "src/explore_persona_space/orchestrate/fleet.py",
    }
)


def _attr_chain_contains(node: ast.AST, name: str) -> bool:
    """True iff ``name`` appears as an attr / base Name in an attribute chain."""
    while isinstance(node, ast.Attribute):
        if node.attr == name:
            return True
        node = node.value
    return isinstance(node, ast.Name) and node.id == name


def _file_calls_anthropic_directly(tree: ast.AST) -> bool:
    """True iff the module AST contains a direct Anthropic client construction
    (``anthropic.Anthropic(`` / ``.AsyncAnthropic(``) OR a
    ``<client>.messages[...].create(`` call.

    AST-based (not a line/regex scan) so a comment / docstring describing the
    pattern — this lint's own prose, a post-mortem note — never false-positives.
    """
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        if isinstance(f, ast.Attribute):
            if f.attr in ("Anthropic", "AsyncAnthropic"):
                return True
            if f.attr == "create" and _attr_chain_contains(f.value, "messages"):
                return True
    return False


def check_api_dispatch_routing(*, repo_root: Path | None = None) -> list[str]:
    """FAIL on a NEW direct-Anthropic call site outside the dispatcher route.

    Walks ``scripts/**/*.py`` + ``src/explore_persona_space/**/*.py``. A file
    that constructs the Anthropic client or calls ``.messages...create(...)``
    directly FAILs UNLESS it is (a) a routing/sanctioned-client layer file
    (:data:`API_DISPATCH_ROUTING_LAYER`), (b) under ``**/archive/`` (frozen),
    (c) in :data:`API_DISPATCH_ROUTING_ALLOWLIST` (grandfathered), or (d)
    carrying a ``# API_DISPATCH_ROUTING_EXEMPT: <reason>`` waiver. New Anthropic
    calls route through ``src/explore_persona_space/llm/api_dispatch.py`` (the
    multi-org headroom-routing + AIMD + batch-vs-sync dispatcher). ``repo_root``
    is a unit-test override; production callers pass None. Bundled into the
    no-flags default run (workflow v2, plan §5).
    """
    root = repo_root if repo_root is not None else _REPO_ROOT
    errors: list[str] = []
    for base in ("scripts", "src/explore_persona_space"):
        base_dir = root / base
        if not base_dir.is_dir():
            continue
        for path in sorted(base_dir.rglob("*.py")):
            if path.name in API_DISPATCH_ROUTING_LAYER:
                continue
            rel = path.relative_to(root).as_posix()
            if "/archive/" in rel:
                continue
            if rel in API_DISPATCH_ROUTING_ALLOWLIST:
                continue
            try:
                text = path.read_text()
            except (OSError, UnicodeDecodeError):
                continue
            if API_DISPATCH_ROUTING_WAIVER in text:
                continue
            try:
                tree = ast.parse(text)
            except SyntaxError:
                continue
            if _file_calls_anthropic_directly(tree):
                errors.append(
                    f"{rel}: constructs/calls the Anthropic client directly "
                    f"(anthropic.Anthropic(...) / .messages...create(...)) outside the routing "
                    f"layer. Route new Anthropic calls through "
                    f"src/explore_persona_space/llm/api_dispatch.py (multi-org headroom routing "
                    f"+ AIMD + batch-vs-sync), or waive a genuinely-correct non-dispatcher "
                    f"caller with a '# {API_DISPATCH_ROUTING_WAIVER}: <reason>' comment."
                )
    return errors


# ── --check-lens-coverage (workflow v2, plan §3) ─────────────────────────────
# The four EXACT State-column prefixes a lens-coverage-map.md row may declare.
_LENS_STATE_PREFIXES: tuple[str, ...] = ("v2-owner:", "v1-only", "retired:", "GAP:")
_LENS_MAP_REL = ".claude/rules/lens-coverage-map.md"


def check_lens_coverage(
    *, repo_root: Path | None = None, warn_sink: list[str] | None = None
) -> list[str]:
    """Validate the workflow-v2 lens-coverage ledger (plan §3).

    Two FAIL modes: (a) a table DATA row in ``.claude/rules/lens-coverage-map.md``
    whose State (last) column does not start with one of the four exact prefixes
    :data:`_LENS_STATE_PREFIXES` (``v2-owner:`` / ``v1-only`` / ``retired:`` /
    ``GAP:``) — a coverage row MUST declare a state; (b) a rule listed in
    ``.claude/rules/LESSONS.md`` (the ``- **<name>**`` bullets) with NO row in
    the map — a lesson silently uncovered. ``GAP:`` rows PASS (an honest "no v2
    owner yet") but are surfaced as WARN lines. WARNs go to ``warn_sink`` when
    provided (unit-test hook), else stderr with a ``WARN: `` prefix; WARNs never
    enter the returned FAIL list. ``repo_root`` is a unit-test override; a
    separate lint from ``--check-lessons-index``. Bundled into the no-flags
    default run.
    """
    root = repo_root if repo_root is not None else _REPO_ROOT
    errors: list[str] = []

    def _warn(msg: str) -> None:
        if warn_sink is not None:
            warn_sink.append(msg)
        else:
            sys.stderr.write(f"WARN: {msg}\n")

    lens_map = root / ".claude" / "rules" / "lens-coverage-map.md"
    if not lens_map.is_file():
        errors.append(f"{_LENS_MAP_REL}: missing — the workflow-v2 lens-coverage ledger.")
        return errors

    covered_items: set[str] = set()
    for raw in lens_map.read_text().splitlines():
        line = raw.strip()
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) < 3:
            continue
        # Skip the markdown separator row (every cell is dashes/colons).
        if all(c and set(c) <= set("-: ") for c in cells):
            continue
        # Skip the table header row.
        if cells[0] == "Item" and cells[-1] == "State":
            continue
        item, state = cells[0], cells[-1]
        covered_items.add(item)
        if not state.startswith(_LENS_STATE_PREFIXES):
            errors.append(
                f"{_LENS_MAP_REL}: row '{item}' has State '{state}' — the State column MUST "
                f"start with one of {list(_LENS_STATE_PREFIXES)}."
            )
        elif state.startswith("GAP:"):
            _warn(f"{_LENS_MAP_REL}: row '{item}' is a GAP (no v2 owner yet): {state}")

    lessons = root / ".claude" / "rules" / "LESSONS.md"
    if not lessons.is_file():
        errors.append(".claude/rules/LESSONS.md: missing — cannot cross-check lens coverage.")
        return errors
    rule_names = {m.group("name") for m in _LESSONS_ROW_RE.finditer(lessons.read_text())}
    for rule in sorted(rule_names - covered_items):
        errors.append(
            f"{_LENS_MAP_REL}: LESSONS.md rule '{rule}' has no coverage row in the map — add a "
            f"'| {rule} | LESSONS.md | <state> |' row so no lesson is silently uncovered."
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
        "--check-pipe-python",
        action="store_true",
        help="Verify no shell script under scripts/ pipes into a bare "
        "python/python3[.N] interpreter with -c/-m (`... | python -c "
        '"..."`). This VM has no `python` on PATH, so the pipe dies with '
        "`python: command not found` (exit 127) — pipe into `uv run "
        "python` instead. Closes the #753 incident class (~41 violations "
        "across 4+ sessions on 2026-06-29). Comment lines (`#`-prefixed) "
        "are skipped. Bundled into the no-flags default run.",
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
        "--check-agent-tools",
        action="store_true",
        help="Verify every .claude/agents/*.md declares an explicit tool "
        "surface ('tools:' allowlist or 'disallowedTools:' denylist), that "
        "every spec-body tool mention (mcp__ tokens, built-in literals, "
        "Agent/Skill phrase forms, prose MCP aliases) is covered by the "
        "declaration (modulo AGENT_TOOLS_MENTION_EXCEPTIONS), that declared "
        "mcp__ names match KNOWN_MCP_SERVERS (silent-typo guard), and that a "
        "denylist never denies a body-mentioned tool. Closes the #778 class "
        "(an undeclared agent inherits every MCP server's schemas, ~168K "
        "static first-turn tokens). Task #840. Bundled into the no-flags "
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
        "--check-dotenv-before-hf-import",
        action="store_true",
        help="AST-walk scripts/**/*.py and FAIL on any script that uses the "
        "bare python-dotenv load_dotenv AND imports huggingface_hub without "
        "first importing explore_persona_space.orchestrate.env.load_dotenv "
        "(#745). The bare dotenv misses the worktree .env and sets no env, so "
        "the HF Hub upload accelerators never get their setdefault and large "
        "uploads crawl. Waive a genuinely-correct bare-dotenv use with "
        "'# DOTENV_LINT_EXEMPT: <reason>'. Bundled into the no-flags default run.",
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
        "--check-no-repo-root-git-reset-hard",
        action="store_true",
        help="FAIL if any .claude/agents/*.md or .claude/skills/**/SKILL.md "
        "contains an unqualified `git reset --hard` (a repo-root / full-tree "
        'reset). Only per-worktree `git -C "$WT" reset --hard` (the `-C` '
        "qualifier before the reset on the same line) or lines carrying the "
        "`workflow-lint: allow-git-reset-hard: <reason>` sentinel with a "
        "non-empty reason pass. A repo-root destructive reset clobbers "
        "concurrent siblings' task state — task.py holds a per-registry flock, "
        "not a per-file lock (incident #815). Bundled into the no-flags "
        "default run.",
    )
    parser.add_argument(
        "--check-no-repo-root-worktree-revert",
        action="store_true",
        help="FAIL if any .claude/agents/*.md or .claude/skills/**/SKILL.md "
        "prescribes an unqualified working-tree revert on the shared repo "
        "root: a `git restore` without `--staged`, a bare-dot `git checkout .`, or "
        'a force-flagged `git clean`. Only per-worktree `git -C "$WT" ...` '
        "forms (the `-C` qualifier before the match on the same line) or "
        "lines carrying the `workflow-lint: allow-repo-root-wt-revert: "
        "<reason>` sentinel with a non-empty reason pass. A repo-root "
        "working-tree revert silently discards CONCURRENT sessions' "
        "uncommitted edits (incident #841; sibling of the #815 reset-hard "
        "check). Bundled into the no-flags default run.",
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
    parser.add_argument(
        "--check-lessons-index",
        action="store_true",
        help="Verify .claude/rules/LESSONS.md (the always-on lessons index, "
        "#739) indexes exactly the set of .claude/rules/*.md files — a rule "
        "with no index row would re-open the #722 plan-time load-timing gap. "
        "Bundled into the no-flags default run.",
    )
    parser.add_argument(
        "--check-compute-shape-review-lens",
        action="store_true",
        help="FAIL if the #806 compute-shape-vs-dispatcher review lens (Step "
        "0.67 heading + `compute-shape-mismatch` blocker tag) is absent from "
        "EITHER .claude/agents/code-reviewer.md or codex-code-reviewer.md. "
        "Pins that both reviewers check a plan-declared data-parallel shape "
        "against the dispatcher's actual capability (incident #779 r6). "
        "Bundled into the no-flags default run.",
    )
    parser.add_argument(
        "--check-long-loop-restartability-review-lens",
        action="store_true",
        help="FAIL if the #881 long-loop restartability lens (the Step 3.6 "
        "heading + `resume predicate` requirement in code-reviewer.md, the "
        "codex copy-list bullets + the inlined-rubric `3.5, 3.6, 3.7` "
        "enumeration in codex-code-reviewer.md, and the `Intra-phase grain` "
        "extension of the checkpoint-per-phase bullet in code-style.md) is "
        "absent from any of its three surfaces. Pins that both reviewers "
        "verify a > ~1h serial loop persists per-unit results and resumes "
        "(incident #823 phase 4: a ~20h in-memory accumulate-and-write-at-end "
        "ridge loop PASSed five review rounds). Bundled into the no-flags "
        "default run.",
    )
    parser.add_argument(
        "--check-hollow-verification-gate-review-lens",
        action="store_true",
        help="FAIL if the #890 hollow-verification-gate lens (the Step 0.68 "
        "sub-check prose in code-reviewer.md, the copy-contract clause + "
        "'0.68' on the inlined-rubric placeholder in codex-code-reviewer.md, "
        "the IMPLEMENTATION-MODE rubric item in efficiency-critic.md — the "
        "workflow-v2 owner) is absent from any surface, or the tag drops "
        "off any surface's '**Blocker tags:**' verdict-template line. Pins "
        "that a verify/equivalence gate asserting on an unused sibling stays "
        "a Major substantive blocker (incident #779: a green "
        "--verify-vectorized laundered an unverified ~17k-fit hot loop). "
        "Bundled into the no-flags default run.",
    )
    parser.add_argument(
        "--check-smoke-architecture-review-lens",
        action="store_true",
        help="FAIL if the #822 smoke-architecture marker presence gate (Step "
        "0.55) is absent from any of its three surfaces: the Step 0.55 "
        "section in code-reviewer.md, the Step 0.55 copy-list bullet + "
        "rubric-placeholder entry in codex-code-reviewer.md, or the "
        "epm:smoke-architecture-check sub-recipe in the /issue Step 5c-bis "
        "strip region. Pins the reviewer-side presence check for the "
        "epm:smoke-architecture-check events row (incident #811: the verdict "
        "lived in prose across 5 PASSed rounds and the gap surfaced only at "
        "Step 6d.0 post-provision). Bundled into the no-flags default run.",
    )
    parser.add_argument(
        "--check-stale-label-disposition",
        action="store_true",
        help="FAIL if the /issue SKILL.md Step 0 stale-label disposition "
        "paragraph (bold anchor '**Stale-label disposition rule', which must "
        "be UNIQUE) loses any of its five #894/#763 semantic tokens — most "
        "critically the fresh-label-execute clause ('the label EXECUTES as "
        "the dispatched round') — or regains an unconditional skip-on-None "
        "coupling ('On None ... skip', a targeted negative regex over the "
        "whitespace-normalized paragraph span). Paragraph-scoped: the span "
        "runs from the anchor to the first blank line (#963). Bundled into "
        "the no-flags default run.",
    )
    parser.add_argument(
        "--check-smoke-output-hygiene",
        action="store_true",
        help="FAIL if the #842 smoke output-path hygiene rule ('Smoke outputs "
        "never overwrite committed artifacts') is absent from the load-bearing "
        "region of any of its three surfaces: the End-to-end-smoke-run "
        "checklist item in experiment-implementer.md, the Step 0.6 region in "
        "code-reviewer.md (the only step the Codex twin's inlined rubric "
        "carries), or the Step 5 smoke-gate paragraph in the /issue SKILL.md. "
        "Region-aware + whitespace-normalized (incident #722: smoke runs "
        "clobbered committed eval_results//figures/ artifacts three times). "
        "Bundled into the no-flags default run.",
    )
    parser.add_argument(
        "--check-vm-thread-cap-guidance",
        action="store_true",
        help="Verify the #891 shared-VM thread-cap launch prefix "
        "(OMP/MKL/OPENBLAS/NUMEXPR=8, one literal string) is pinned — at its "
        "per-file occurrence-count floor — in the four VM-launch guidance "
        "surfaces (the /issue detached-launch template, "
        "experiment-implementer.md, code-style.md, "
        "analyzer-section-reference.md). The launch-time prefix is the "
        "branch-age-independent fallback for the #847 src/-side setdefault "
        "(incident #779). Bundled into the no-flags default run.",
    )
    parser.add_argument(
        "--check-judge-model-pins",
        action="store_true",
        help="Walk scripts/**/*.py, scripts/**/*.sh, "
        "src/explore_persona_space/**/*.py, and tests/**/*.py and FAIL on a "
        "hardcoded NON-Sonnet judge-model pin at a judge call site. The "
        "standing rule pins ONE judge — claude-sonnet-4-5-20250929 — for every "
        "judged behavior (.claude/rules/llm-judging.md). The gate is "
        "assignment/call-aware (a *JUDGE_MODEL* assignment, a --judge-model / "
        "judge_model= / JUDGE_MODEL= flag, or a model= kwarg with a judge token "
        "in window), so a prose-string mention or comment is never flagged. "
        "Legitimate non-Sonnet pins (Betley gpt-4o calibration anchors, the "
        "translation-judge exemptions, stale-grandfathered Haiku pins) are "
        "grandfathered in JUDGE_PIN_LEGACY_ALLOWLIST[_SH]; waive a new "
        "calibration control with '# noqa: judge-model-pin'. Bundled into the "
        "no-flags default run (#765).",
    )
    parser.add_argument(
        "--check-no-literal-round-marker-versions",
        action="store_true",
        help="FAIL on a literal 'v1' posting instruction for a round-versioned "
        "marker kind (epm:experiment-implementation / epm:results / "
        "epm:proposed-tests) in checked-in workflow prose (CLAUDE.md, "
        "workflow.yaml, agents/rules .md, every SKILL.md, the /issue "
        "markers.md + templates/). Those kinds accrue rows across follow-up "
        "rounds, and an explicit --version beats the CLI's max+1 default, so "
        "'post at v1' prose seeds briefs that collide with existing rows "
        "(incident #825; the #389 class). Rephrase to v<n> / max+1. Bundled "
        "into the no-flags default run (#917).",
    )
    parser.add_argument(
        "--check-agent-spec-size",
        action="store_true",
        help="agent-spec size budget: WARN >28 KB, FAIL >40 KB (grandfather-ratchet)",
    )
    parser.add_argument(
        "--check-api-dispatch-routing",
        action="store_true",
        help="FAIL on a NEW direct-Anthropic call site (anthropic.Anthropic(...) / "
        ".AsyncAnthropic(...) / <client>.messages...create(...)) under scripts/ or "
        "src/explore_persona_space/ outside the routing layer (api_dispatch.py / "
        "judge_dispatch.py / batch_judge.py / anthropic_client.py). New Anthropic "
        "calls route through the multi-org dispatcher api_dispatch.py (headroom "
        "routing + AIMD + batch-vs-sync). scripts/archive/** + the current tree's "
        "existing callers (API_DISPATCH_ROUTING_ALLOWLIST) are grandfathered; waive "
        "a new non-dispatcher caller with '# API_DISPATCH_ROUTING_EXEMPT: <reason>'. "
        "Bundled into the no-flags default run (workflow v2, plan §5).",
    )
    parser.add_argument(
        "--check-lens-coverage",
        action="store_true",
        help="Validate .claude/rules/lens-coverage-map.md (workflow v2, plan §3): "
        "FAIL a table row whose State column does not start with one of v2-owner: / "
        "v1-only / retired: / GAP:, and FAIL a .claude/rules/LESSONS.md rule with no "
        "coverage row in the map. GAP: rows PASS but print as WARN. A separate lint "
        "from --check-lessons-index. Bundled into the no-flags default run.",
    )
    parser.add_argument(
        "--check-phase-done-reserved",
        action="store_true",
        help="Walk scripts/**/*.sh dispatchers and FAIL any non-redirected "
        "invocation of a scripts/*.py|*.sh phase script that contains a "
        "genuine [phase=done] emission site — the reserved-token contract of "
        ".claude/rules/pod-side-reporting.md requirement 1 (a mid-pipeline "
        "child emission reads as a false status=done to poll_pipeline.py; "
        "incidents #545, #920). AST-based .py emission detection (comments / "
        "docstrings / match sites never flag); stdout-redirected per-worker "
        "invocations skipped; tee'd edges still checked. Legacy edges frozen "
        "in PHASE_DONE_EDGE_LEGACY_ALLOWLIST; waive a mode-gated "
        "standalone-lane terminal with '# noqa: phase-done-reserved'. "
        "Bundled into the no-flags default run + the "
        "workflow-lint-phase-done-reserved pre-commit hook (#930).",
    )
    parser.add_argument(
        "--check-jsonl-splitlines",
        action="store_true",
        help="AST-walk scripts/**/*.py + src/explore_persona_space/**/*.py and "
        "FAIL any .splitlines() call reading JSONL content (4 signals: "
        "jsonl-named read_text chain / jsonl-named receiver / jsonl-named "
        "enclosing function / events-comments-path read_text chain). "
        "splitlines() splits on raw U+2028/U+2029/NEL inside "
        "ensure_ascii=False JSON strings and shreds valid records (#825/#950); "
        "use split('\\n') or text-mode file iteration. Waive with "
        "'# JSONL_SPLITLINES_EXEMPT: <reason>'; frozen legacy experiment "
        "scripts live in JSONL_SPLITLINES_LEGACY_ALLOWLIST (experiment files "
        "only — never a workflow-surface file). Bundled into the no-flags "
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
        or args.check_skill_refs
        or args.check_wandb_required
        or args.check_heredoc_dotenv
        or args.check_dispatcher_cvd_pin
        or args.check_pipe_python
        or args.check_marker_registry
        or args.check_agent_model_pins
        or args.check_agent_tools
        or args.check_upload_as_file
        or args.check_dotenv_before_hf_import
        or args.check_batch_judge_client
        or args.check_no_workflow_improver_spawn
        or args.check_no_repo_root_git_reset_hard
        or args.check_no_repo_root_worktree_revert
        or args.check_gate_ids_unique
        or args.check_lessons_index
        or args.check_compute_shape_review_lens
        or args.check_long_loop_restartability_review_lens
        or args.check_hollow_verification_gate_review_lens
        or args.check_smoke_architecture_review_lens
        or args.check_stale_label_disposition
        or args.check_smoke_output_hygiene
        or args.check_vm_thread_cap_guidance
        or args.check_judge_model_pins
        or args.check_no_literal_round_marker_versions
        or args.check_agent_spec_size
        or args.check_api_dispatch_routing
        or args.check_lens_coverage
        or args.check_phase_done_reserved
        or args.check_jsonl_splitlines
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
    if args.check_pipe_python or no_flags:
        errors.extend(check_pipe_python())
    if (args.check_marker_registry or no_flags) and not args.check_references:
        errors.extend(check_marker_registry(workflow))
    if args.check_agent_model_pins or no_flags:
        errors.extend(check_agent_model_pins())
    if args.check_agent_tools or no_flags:
        errors.extend(check_agent_tools())
    if args.check_upload_as_file or no_flags:
        errors.extend(check_upload_as_file())
    if args.check_dotenv_before_hf_import or no_flags:
        errors.extend(check_dotenv_before_hf_import())
    if args.check_batch_judge_client or no_flags:
        errors.extend(check_batch_judge_client())
    if args.check_no_workflow_improver_spawn or no_flags:
        errors.extend(check_no_workflow_improver_spawn())
    if args.check_no_repo_root_git_reset_hard or no_flags:
        errors.extend(check_no_repo_root_git_reset_hard())
    if args.check_no_repo_root_worktree_revert or no_flags:
        errors.extend(check_no_repo_root_worktree_revert())
    if args.check_gate_ids_unique or no_flags:
        errors.extend(check_gate_ids_unique(workflow))
    if args.check_lessons_index or no_flags:
        errors.extend(check_lessons_index())
    if args.check_agent_spec_size or no_flags:
        errors.extend(check_agent_spec_size())
    if args.check_compute_shape_review_lens or no_flags:
        errors.extend(check_compute_shape_review_lens())
    if args.check_long_loop_restartability_review_lens or no_flags:
        errors.extend(check_long_loop_restartability_review_lens())
    if args.check_hollow_verification_gate_review_lens or no_flags:
        errors.extend(check_hollow_verification_gate_review_lens())
    if args.check_smoke_architecture_review_lens or no_flags:
        errors.extend(check_smoke_architecture_review_lens())
    if args.check_stale_label_disposition or no_flags:
        errors.extend(check_stale_label_disposition_clause())
    if args.check_smoke_output_hygiene or no_flags:
        errors.extend(check_smoke_output_hygiene())
    if args.check_vm_thread_cap_guidance or no_flags:
        errors.extend(check_vm_thread_cap_guidance())
    if args.check_judge_model_pins or no_flags:
        errors.extend(check_judge_model_pins())
    if args.check_no_literal_round_marker_versions or no_flags:
        errors.extend(check_no_literal_round_marker_versions())
    if args.check_api_dispatch_routing or no_flags:
        errors.extend(check_api_dispatch_routing())
    if args.check_lens_coverage or no_flags:
        errors.extend(check_lens_coverage())
    if args.check_phase_done_reserved or no_flags:
        errors.extend(check_phase_done_reserved())
    if args.check_jsonl_splitlines or no_flags:
        errors.extend(check_jsonl_splitlines())

    if errors:
        for err in errors:
            sys.stderr.write(f"workflow_lint: {err}\n")
        sys.stderr.write(f"workflow_lint: FAIL ({len(errors)} error(s))\n")
        return 1

    sys.stderr.write("workflow_lint: PASS\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
