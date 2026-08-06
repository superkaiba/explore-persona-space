"""Lint ``.claude/workflow.yaml`` against its Pydantic schema.

Callable from a pre-commit hook AND importable for unit tests.

Behaviours:

* ``--check-references`` (default in pre-commit): walk ``CLAUDE.md``,
  ``.claude/skills/issue/SKILL.md``, and ``.claude/skills/issue/markers.md``;
  every ``(see workflow.yaml § <key>)`` reference MUST resolve to a real
  YAML key. NOT in the no-flags default run: a bare ``workflow_lint.py``
  invocation does not run this check — it fires only when the flag is
  passed explicitly (the pre-commit hook, /daily's reference gate, and
  the /issue Step-10d parity legs pass it).
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
  deleted-or-never-created helper and CalledProcessErrors. On a
  non-``main`` checkout (a stale issue worktree) a reference missing
  locally but tracked at ``main``/``origin/main`` degrades to a
  non-blocking ``WARN:`` (#1622/#1672); a non-git tree or a failed git
  probe stays strict (hard FAIL).
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
  :data:`HISTORICAL_REF_OPT_OUT` are a one-off narrative escape. On a
  non-``main`` checkout a plain single-segment ref unresolved locally but
  whose ``SKILL.md`` is tracked at ``main``/``origin/main`` degrades to a
  non-blocking ``WARN:`` (#1622/#1672); non-git trees / failed probes /
  ``:``-namespaced refs stay strict.
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
* ``--check-piped-git-push`` (also bundled into the no-flags default run):
  walk every ``*.sh`` under ``scripts/`` and FAIL on any ``git push`` /
  ``git merge`` / ``git commit`` / ``gh pr merge|create`` piped into a
  filter on its own
  pipeline segment (``git push origin main 2>&1 | tail -20``). Bash makes
  the compound's exit status the FILTER's, so the pipe masks the
  producer's non-zero exit code: a rejected push reads as success and the
  session proceeds believing the merge landed (#957's Step 10d push was
  masked 2026-07-04; 4 sessions hit the class 2026-07-02); a hook-running
  ``git commit`` piped this way is additionally SIGPIPE-killed
  mid-pre-commit-hook (#1584, #1591). Prose rule:
  CLAUDE.md § Concurrent repo-root committers ("run it bare and check the
  exit code, or use ``set -o pipefail`` when a pipe is unavoidable") — so
  a file-level non-comment ``pipefail`` line disables flagging for the
  REST of the file (lines before it still flag); ``--dry-run`` spans and
  ``#``-comment lines are skipped; ``||`` chains, ``git merge-base``, and
  producer-as-consumer (``... | git push``) never match; ``|&`` is
  normalized to ``|`` first. The
  ``.claude/hooks/guard_piped_git_push.sh`` PreToolUse hook covers the
  inline ad-hoc commands that never reach a committed script (#1048).
* ``--check-agents-note-argv-verdict`` (also bundled into the no-flags
  default run): walk every ``*.md`` under ``.claude/agents/`` and FAIL on
  any line prescribing an argv-prose ``--note`` verdict/marker post via a
  command substitution — the pattern task #1743 banned from agent specs
  (merged 99af2fbb0d) and rewrote to the ``post-marker --file`` channel;
  the standing pin is #1785 (#1722/#1756 argv-substitution incident
  family). The sanctioned variable form (resolve every command
  substitution into a shell variable FIRST, then pass the variable as the
  note) never matches. No waiver: reword prose that would match (the
  #1743 r2 precedent).
* ``--check-sha-pin-domain`` (also bundled into the no-flags default run):
  scan every ``*.py`` under ``scripts/`` + ``src/explore_persona_space/``
  for whole-string 64-hex literals (the sha-pin constant shape) and FAIL a
  hex duplicated across >= 2 modules when a site declares NO content
  DOMAIN (the #1776/#1491 wrong-domain propagation class: a new module
  copies an INDEX-array digest as a bare ``VAL_SHA256`` and a consumer
  asserts PROMPT digests against it — can never pass on ANY input) or
  when sites declare CONFLICTING domains (INDEX vs PROMPT, ...). Declare
  via an adjacent ``# SHA_PIN_DOMAIN: <INDEX|IDS|PROMPT|BYTES|CONTENT>``
  comment or a domain token in the binding name; waive a site with
  ``# SHA_PIN_DOMAIN_EXEMPT: <reason>``. Legacy duplicated hexes are
  frozen as ``(hex[:12], file)`` pairs in
  :data:`SHA_PIN_DOMAIN_GRANDFATHER` (a grandfathered hex copied into a
  NEW file still FAILs; a stale entry FAILs the run — the set shrinks,
  never silently grows). Conflicts have NO allowlist escape. Prose rule:
  ``.claude/rules/gotchas.md`` "A sha pin lives in a DOMAIN" (#2079).
* ``--check-push-failure-swallow`` (also bundled into the no-flags default
  run): walk every ``*.sh`` under ``scripts/`` and FAIL on any logical
  line where a ``git push`` is followed ON THE SAME LINE by ``|| echo`` /
  ``|| true`` / ``|| :`` / ``|| printf`` — failure-swallowing without
  verification: the workload declares success while the result commit
  never landed, and on GCE the self-DELETEing instance holds the only
  copy (#825 r6/r7/r8; the workload-side ``||`` sibling of the
  ``--check-piped-git-push`` pipe-masking class — ``pipefail`` does NOT
  exempt, it never applies to ``||`` disjunctions). ``if git push ...;
  then`` conditions, bare pushes, and ``|| { retry; }`` groups never
  match; ``#``-comment lines are skipped; waive with
  ``# PUSH_SWALLOW_EXEMPT: <reason>`` (same/preceding line); legacy
  offenders live in the frozen path-keyed
  ``PUSH_SWALLOW_LEGACY_ALLOWLIST``. Contract:
  ``.claude/rules/pod-side-reporting.md`` § Result-push verification
  contract (#1205).
* ``--check-sh-function-rc-capture`` (also bundled into the no-flags
  default run): walk every ``*.sh`` under ``scripts/`` and FAIL on any
  SAME-FILE bash function invoked via ``func || rc=$?`` / ``|| true`` /
  ``|| :`` while the script runs under ``set -e`` — bash disables errexit
  throughout the function BODY when the call sits in an ``||`` context,
  so mid-function failures collapse to the last command's rc (#1426: a
  Gate-1 terminal failure + a manifest SystemExit read as rc=0 and the
  ``[phase=done]`` success sentinel fired). Single external-command
  captures (``uv run python ... || rc=$?``) never match (the invocation
  regex requires a collected same-file function name at command
  position); ``set +e`` regions are unflagged (line-order state
  tracking); quoted strings, definition lines, heredoc bodies, trailing
  comments, and later ``;``-segments never match. Waive with
  ``# RC_CAPTURE_EXEMPT: <reason>``. ShellCheck SC2310 is the broader
  external analogue (#1516).
* ``--check-grep-qv`` (also bundled into the no-flags default run): scan
  fenced code blocks in ``.claude/skills/**/SKILL.md`` +
  ``.claude/agents/*.md`` and logical lines of ``scripts/**/*.sh``, and
  FAIL on any UNPINNED ``grep``/``ugrep`` invocation combining q
  (``-q``/``--quiet``/``--silent``) and v (``-v``/``--invert-match``) —
  a combined short token, separated tokens, or long forms. ugrep 7.5.0's
  quiet+invert exit status diverges from GNU (rc=1 even when non-matching
  lines are selected), so an rc-consumed q+v trigger silently fails OPEN
  when shell ``grep`` resolves to ugrep (#928: the Step 10d pre-push lint
  gate disarmed as skip-artifact-only on a code-bearing payload; fixed in
  #1125 with the output-test rewrite). ``git grep`` and a path-pinned
  ``/usr/bin/grep`` are exempt; a path-pinned ``ugrep`` still flags (its
  exit status is divergent by construction, no pin sanctions it).
  ``#``-comment lines and ``.md`` prose outside fences are skipped.
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
* ``--check-marker-scalar-integrity`` (also bundled into
  ``--check-references``): scan all four string fields (``kind``,
  ``posted_by``, ``when``, ``fields``) of every parsed ``workflow.yaml
  § markers`` entry for the truncated-comment signature — the PARSED
  value ends in ``,`` or ``(`` after rstrip, or has unbalanced parens.
  An unquoted YAML plain scalar containing ``' #'`` silently truncates
  at the comment marker at parse time, and ``--check-references``
  passes because the regenerated ``markers.md`` table matches the
  truncated parse (#873: ``posted_by`` shipped as ``skill (...);
  poll_pipeline (runtime tripwire,`` with a dangling table cell).
  Deliberate prose that trips the signature is waived via
  :data:`MARKER_SCALAR_INTEGRITY_ALLOWLIST` with a reason.
* ``--check-poller-marker-consumers`` (also bundled into
  ``--check-references``): every marker kind whose ``posted_by`` names
  a poller/watcher (``poll_pipeline`` / ``backend_poll`` /
  ``slurm_monitor`` / ``autonomous_session_watch`` / ``pod_watch`` /
  ``tick_triage``) must (Leg A) be referenced by at least one consumer
  surface — every ``.claude/skills/**/SKILL.md`` plus the poller/triage
  scripts — and (Leg B) appear in each poster script its ``posted_by``
  token names. A poller feature claiming mid-run surfacing with no
  consuming or posting code is the #873 pre-fix state (a runtime
  tripwire declared in workflow.yaml with no poll_pipeline code until a
  critic caught it). Deliberate out-of-band consumers are waived via
  :data:`POLLER_CONSUMER_ALLOWLIST` with a reason.
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
* ``--check-hub-dir-filecount`` (also bundled into the no-flags default
  run): AST-walk every ``*.py`` under ``scripts/`` and FAIL on any DIRECT
  ``upload_folder(...)`` call site — attribute form (``api.upload_folder(``)
  or a bare name (a ``from huggingface_hub import upload_folder`` caller;
  a module defining a LOCAL ``def upload_folder`` is carved out of the
  bare-name arm, the ``scripts/issue623_upload.py`` wrapper shape) — in a
  module that does not reference ``assert_hub_dir_filecounts`` (the hub.py
  runtime guard, #1190). The Hub rejects any single repo directory holding
  >10000 files at COMMIT time with a NON-retriable ``BadRequestError``
  fired AFTER the full compute has run and all bytes are staged (#658 r2:
  12000 rollout files staged into one dir); the shared hub helpers
  (``_upload`` folder branch / ``_upload_folder_filtered``) pre-count
  staged files per TARGET repo dir and raise ``HubDirFileCountError``
  before any network I/O, and this check funnels direct ``HfApi`` callers
  toward the same one-line guard (called OUTSIDE any transient-retry
  wrapper — a guard raise is deterministic). Pre-existing direct call
  sites are grandfathered in :data:`HUB_DIR_FILECOUNT_LEGACY_ALLOWLIST`
  (grep-generated, live-tree-test-pinned, never hand-extended); waive a
  new genuinely-correct call with ``# HUB_DIR_FILECOUNT_EXEMPT: <reason>``
  (reason ≥ 10 chars) on the call's first physical line or the
  immediately preceding non-blank line.
* ``--check-upload-prefix-clobber`` (also bundled into the no-flags
  default run): AST-walk every ``*.py`` under ``scripts/`` (two passes)
  and FAIL on hardcoded issue-prefix HF upload DESTINATIONS of the #1005
  parent-clobber class (reused #928 fitters uploaded #1005 tensors to
  hardcoded ``issue928_*`` prefixes, overwriting the parent's artifacts).
  Write call sites only — :data:`UPLOAD_DEST_FUNCS` (``upload_file`` /
  ``upload_folder`` / ``CommitOperationAdd`` / ``hub._upload`` /
  ``hub._upload_folder_filtered`` /
  ``upload_raw_completions_to_data_repo``) plus one level of inferred
  wrappers (the copied ``issue<N>_common.py`` pattern); cross-issue READS
  (``list_repo_tree`` / ``hf_hub_download``) never flag. Rule A: a
  destination token ``issue<M>_…`` in an ``issue<N>_`` script with M != N
  FAILs (never silently allowlisted). Rule B: an own-issue token arriving
  via a FALLBACK channel — ``x or CONST``, an argparse ``default=``, a
  wrapper-param signature default — FAILs (pre-existing sites
  grandfathered in :data:`UPLOAD_PREFIX_CLOBBER_ALLOWLIST`); a DIRECT
  own-prefix hardcode is the sanctioned norm and never flags. Waive with
  ``# UPLOAD_PREFIX_EXEMPT: <reason>`` (reason ≥ 10 chars) on the
  finding's first physical line or the immediately preceding non-blank
  line (for an argparse-default finding, at the ``add_argument`` call).
* ``--check-upload-file-in-loop`` (also bundled into the no-flags
  default run): AST-walk every ``*.py`` under ``scripts/`` and FAIL on
  any per-file upload call lexically inside a loop / comprehension —
  shape A: an ``upload_file(...)`` call (attribute form or a bare
  ``from huggingface_hub import upload_file`` name); shape B: an
  ``_upload(...)`` call carrying an explicit ``upload_as_file=True``
  constant kwarg (the literal #664 form, which
  ``--check-upload-as-file`` deliberately DEFERS on explicit kwargs, so
  nothing else flags it). Each per-file call is one Hub commit + a
  server-side repo pre-check, so an N-file loop 504-storms on a large
  repo (#664: a 1425-file loop held an 8xH200 idle for 12h, ~$530) and
  trips the org-level ~2500-req/5-min 429 quota (#1481: ~1400 planned
  per-file commits -> HF 429 storm); bulk uploads compose ONE
  ``upload_folder`` commit. Loop context resets at function / lambda /
  class boundaries (a helper *called* from a loop is a deliberate
  lexical false negative). Waive a genuinely bounded loop (a retry
  wrapper around ONE file, a fixed <=3-file list) with
  ``# UPLOAD_LOOP_EXEMPT: <reason>`` (reason ≥ 10 chars) on the call's
  first physical line or the immediately preceding non-blank line;
  pre-existing sites are grandfathered with EXACT per-file site counts
  in :data:`UPLOAD_FILE_IN_LOOP_LEGACY_ALLOWLIST` (count-grain — a NEW
  offense inside a grandfathered file surfaces instead of hiding;
  never hand-extended).
* ``--check-upload-return-discard`` (also bundled into the no-flags
  default run): AST-walk every ``*.py`` under ``scripts/`` and FAIL on
  any Expr-statement (discarded-return) call to the fail-soft-by-return
  hub upload helpers ``_upload`` / ``_upload_folder_filtered`` — both
  return ``""`` on upload failure (``_upload`` raises only under
  ``raise_on_error=True``, and even then the non-exception ``""``
  returns are unchanged; ``_upload_folder_filtered``'s pre-flight
  ``assert_hub_dir_filecounts`` guard raises, its upload failures never
  do), so a discarded return converts silent durability loss into
  exit 0 (#2087; incident #2054). Import/definition-resolved arming: a
  same-named LOCAL helper never arms the check. Waive a deliberate
  fail-soft caller with ``# UPLOAD_RETURN_DISCARD_EXEMPT: <reason>``
  (reason ≥ 10 chars) on the call's first physical line or the
  immediately preceding non-blank line; pre-existing sites are
  grandfathered with <=-tolerant per-file counts in
  :data:`UPLOAD_RETURN_DISCARD_LEGACY_ALLOWLIST` (live-owned entries
  listed in :data:`UPLOAD_RETURN_DISCARD_PENDING_OWNER`; never
  hand-extended).
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
  fixed with #950). Six narrow
  signals: (a) a ``read_text``-bearing receiver chain whose source
  segment mentions ``jsonl``; (b) a bare receiver ``Name`` matching
  ``jsonl``; (c) the call sits inside a ``jsonl``-named function; (d) a
  ``read_text``-bearing receiver chain whose base ``Name`` is
  ``ev_path``/``events_path``/``concerns_path`` or whose segment names
  ``events.jsonl``/``comments.jsonl``/``concerns.jsonl``; (e) a
  ``read_text``-bearing receiver chain in a module that
  ``glob``/``rglob``/``iglob``-s a ``jsonl`` pattern (#1162, the #1132
  generic-receiver evasion); (f) a bare-``Name`` receiver assigned earlier
  in the same scope from a jsonl-evidenced ``read_text()`` expression
  (#1162, the #1032 assignment-dataflow evasion). Deliberate false
  negatives (path-variable dataflow in non-globbing modules,
  cross-function/cross-scope dataflow, non-``read_text`` read channels,
  shell heredocs) are documented
  in the check docstring — the gotchas.md entry carries those. Waive a
  genuinely-safe flagged site with ``# JSONL_SPLITLINES_EXEMPT: <reason>``
  (reason ≥ 10 chars) on the call's first physical line or the
  immediately preceding non-blank line; frozen legacy per-issue
  experiment scripts are grandfathered in
  :data:`JSONL_SPLITLINES_LEGACY_ALLOWLIST` (experiment files ONLY — a
  workflow-surface file is never allowlisted, it is fixed). Unparseable
  files (SyntaxError / non-UTF-8) are skipped WITH a printed notice,
  never silently.
* ``--check-scripts-import-guard`` (also bundled into the no-flags
  default run): AST-walk every ``*.py`` under
  ``src/explore_persona_space/experiments/`` and ``scripts/`` (#1229)
  and FAIL any ``scripts.*``
  import — deferred (function-body) AND module-top-level — lacking a
  repo-root ``sys.path`` guard. In script mode
  (``python /abs/path/driver.py``) ``sys.path[0]`` is the script's own
  directory — not cwd, not the repo root — so ``import scripts.*``
  raises ``ModuleNotFoundError`` pod/GCE-side; deferred instances crash
  MID-RUN after paid GPU phases, and both standard pre-launch checks
  false-pass them (incident #823 Phase-3: ~30 min of paid GCE work
  lost; the #853 fix was documentation-only). Guard evidence = a call
  whose callee name mentions ``syspath``/``sys_path`` (the
  ``_ensure_repo_root_on_syspath()`` run_823.py exemplar, commit
  ``14234c9112``) or a literal
  ``sys.path.insert(...)``/``sys.path.append(...)`` — same-innermost-
  function preceding the import, or at module scope (any line covers a
  deferred import; a PRECEDING line covers a top-level one). Offender
  and guard detection share ONE pruned scope walk: module-scope
  ``If``/``Try``/``With``/``For``/``While`` bodies — including a
  ``try/except ImportError``-wrapped import and the
  ``if __name__ == "__main__":`` main-block shape — are
  module-executing and IN scope; nested defs/classes/lambdas are pruned
  (the deferred pass owns function bodies). ``try/except ImportError``
  is NOT a guard (a silent wrong-path fallback pod-side);
  ``TYPE_CHECKING`` bodies are skipped. Deliberate false negatives
  (importlib/``__import__``/exec-string imports, class-body imports,
  outer-but-not-innermost-scope guards, conditional-guard
  presence-counting, shell heredocs) are documented in the check
  docstring. Waive a genuinely-safe flagged site with
  ``# SCRIPTS_IMPORT_GUARD_EXEMPT: <reason>`` (reason ≥ 10 chars) on
  the import's first physical line or the immediately preceding
  non-blank line. No legacy allowlist (the live tree is clean).
* ``--check-upload-or-true`` (also bundled into the no-flags default
  run): walk every ``*.sh`` under ``scripts/`` and FAIL any
  upload/result-persist command line whose failure is swallowed by
  ``|| true`` / ``|| :`` / ``; true`` (#841 silent artifact loss —
  swallowed plot-phase failures + a missing upload phase lost stage
  JSONs/plots across attempts). Terminal swallows mask the whole
  ``&&``-chain (whole-line token check); non-terminal ``|| true`` is
  segment-scoped; swallowed heredoc openers and multi-line
  ``python -c "…"`` blocks are scanned for BODY upload-call tokens.
  Legacy deliberate uses frozen in
  :data:`UPLOAD_OR_TRUE_LEGACY_ALLOWLIST`; waive with
  ``# UPLOAD_OR_TRUE_EXEMPT: <reason>`` (reason ≥ 10 chars).
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
* ``--check-hub-verify-retry`` (also bundled into the no-flags default run):
  AST-walk every ``*.py`` under ``scripts/`` and FAIL on any bare Hub verify
  call — ``list_repo_files(`` / ``list_repo_tree(`` / ``.file_exists(``
  (Attribute form, plus the asname-aware ``from huggingface_hub import``
  Name form) — outside the grandfathered legacy set
  (:data:`HUB_VERIFY_LEGACY_ALLOWLIST`). huggingface_hub's paginate retries
  ONLY 429 on cursor pages, so a transient 504 on a bare listing/probe
  fails a SUCCESSFUL upload's verify leg (#920); #997 built the retried
  library path (``orchestrate/hub.py``: ``verify_repo_paths_uploaded``,
  ``list_hf_files_under_path``, ``list_repo_files_complete``,
  ``retry_transient``) but added no gate on new scripts/ call sites — this
  check is that gate (#1202). Named residuals NOT covered: ``repo_info``,
  ``hf_hub_download``, ``HfFileSystem``, raw HTTP, subprocess ``hf`` CLI,
  ``getattr`` forms, and the 9 ``scripts/*.sh`` heredoc offenders. A
  genuinely-correct raw call waives with
  ``# HUB_VERIFY_RETRY_EXEMPT: <reason>`` (reason ≥
  :data:`HUB_VERIFY_WAIVER_MIN_REASON_CHARS` chars) on the call's first
  physical line or the immediately preceding non-blank line.
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
* ``--check-live-hf-retry-routing`` (also bundled into the no-flags default
  run): walk every ``*.py`` under ``scripts/`` and
  ``src/explore_persona_space/`` and FAIL on a bare (un-retried) HuggingFace
  Hub call in LIVE code — ``hf_hub_download(`` / ``.upload_file(`` /
  ``.upload_folder(`` / ``create_commit(`` / ``push_to_hub(`` with no
  ``retry_transient`` / ``_retry_upload`` wrap ANCHORED to the call (same
  line before the call, or opening within
  :data:`HF_ROUTING_WRAP_WINDOW` lines above with the wrap expression still
  open at the call line — a wrapped sibling never launders a bare call) and
  no ``# NO_RETRY: <reason>`` waiver on the line or the line above.
  hf_hub 0.36.2 natively retries only 500/502/503/504 on the download/LFS
  paths and the commit API not at all, so a bare live site is a 429
  single-point-of-failure (#1426/#1335, the 2026-07-18 storm). The
  per-issue historical files frozen at #1547 landing time are
  snapshot-exempt (:data:`HF_ROUTING_FROZEN_SNAPSHOT` — the routing
  requirement attaches at REUSE time via artifact-reuse check (i)); NEWLY
  written files, including new ``scripts/issue<N>_*.py`` drivers, ARE
  scanned. ``scripts/workflow_lint.py`` / ``scripts/verify_plan.py``
  (pattern strings) and ``backends/gcp.py`` (generated pod-side heredocs
  with their own bounded retry) are constant-excluded. Bare
  ``snapshot_download`` / ``list_repo_files`` sites are OUT of the predicate
  by design (#1547). Regenerate the snapshot with
  ``--regen-hf-routing-snapshot`` (maintenance flag) — see the constant's
  comment for the staleness-race recipe (#1568).
* ``--check-bare-list-repo-files`` (also bundled into the no-flags default
  run): AST-walk every ``*.py`` under ``scripts/`` and
  ``src/explore_persona_space/`` and FAIL on any bare ``list_repo_files``
  call/reference — a Load-ctx ``.list_repo_files(`` Attribute under ANY
  receiver, or a Load-ctx Name bound by ``from huggingface_hub import
  list_repo_files [as alias]`` (:func:`_hub_verify_bare_hits` narrowed to
  :data:`LIST_REPO_FILES_TARGETS`). hub 0.36.2's ``HfApi.list_repo_files``
  has NO scoping parameter — its body is an unscoped
  ``list_repo_tree(recursive=True)`` full-tree walk — so every call is a
  full listing, which WEDGES on the ~1M-file data repo (>90 s #833, >600 s
  #920; two kills 2026-07-22 → #1624) and retry cannot save it (the walk
  grinds, it does not raise) — orthogonal to ``--check-hub-verify-retry``
  (transient-retry property) and deliberately OUT of
  ``--check-live-hf-retry-routing``'s predicate. Fix: the scoped recipes —
  ``hub.list_hf_files_under_path`` / ``hub.verify_repo_paths_uploaded`` /
  ``api.list_repo_tree(path_in_repo=...)`` / ``api.file_exists``
  (single-path probe). A genuinely-correct SMALL-repo full listing waives
  with ``# LIST_REPO_FILES_EXEMPT: <reason>`` (reason ≥
  :data:`LIST_REPO_FILES_WAIVER_MIN_REASON_CHARS` chars) on the call's line
  or the previous non-blank line. Historical files frozen at #1624
  implement time are snapshot-exempt
  (:data:`LIST_REPO_FILES_FROZEN_SNAPSHOT`; regenerate with
  ``--regen-list-repo-files-snapshot``, the #1568 idiom). Named residuals
  NOT covered: never-committed ad-hoc probes (inline ``python -c``
  one-liners — the shape of one of the 2026-07-22 kills) are structurally
  outside ANY file lint (this check covers the committed-code subclass);
  unscoped ``list_repo_files_complete`` / ``list_repo_tree(recursive=True)``
  calls without ``path_in_repo`` (kwarg-presence analysis = high-FP);
  ``snapshot_download``; ``getattr`` evasion; ``.sh`` heredocs;
  ``HfFileSystem.ls()``.
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
* ``--check-crash-fix-relaunch-contract`` (also bundled into the no-flags
  default run): FAIL if the #1081 crash-fix-relaunch fix-engaged contract
  prose regresses on any of its three surfaces — the experimenter.md D3
  crash-fix-relaunch paragraph (anchor ``**Crash-fix relaunch (brief
  carries `fix_sha=`):**``), the crash-fix-rounds.md ``fix_sha=<sha>``
  note-token paragraph, or the /issue SKILL.md Step 7 code-row relaunch
  contract paragraph. Each surface has a UNIQUE literal anchor and its own
  whitespace-normalized required-token list — most critically the
  disposition-conditional three-way resume-glob confirm trio ("empty / the
  fresh path / exactly the RETAINED expected paths", the #1081 round-2
  blocker fix) and the ``fix_sha=`` note-token/brief duty — plus a shared
  negative regex against re-introducing the unconditional "resolves EMPTY"
  confirm (the regex pins the HISTORICAL #1081 wording only; its lookahead
  spares the healthy trio "resolves EMPTY / ..."). Paragraph-scoped: a
  contradiction outside — or additively inside — an anchored span is
  invisible (inherent to the token-lint class) (#1181).
* ``--check-awk-elision-parity`` (also bundled into the no-flags default
  run): FAIL if the ban-gate awk elision program — the single-quoted awk
  program on the unique ``f=!f`` anchor line — drifts between its two
  full-text homes (/issue SKILL.md Step 9a-humanize;
  analyzer-section-reference.md Step 4.5), or a home is missing / has 0 or
  >1 anchor lines / carries an anchor line whose total single-quote count is
  not exactly 2 (a program that gained a shell quote-escape would truncate
  the extraction at the first quote in both homes, hiding drift past the
  truncation point) / yields no extractable ``awk '...'`` span. Pins the
  quoted PROGRAM only — the surrounding invocation lines (paths, fencing,
  indentation) legitimately differ — and parity is not correctness: an
  identical-but-broken edit applied to both homes passes by design (#1153).
* ``--check-asw-docstring-pass-count`` (also bundled into the no-flags
  default run): parse the '<N> passes' digit header from the
  ``scripts/autonomous_session_watch.py`` module docstring, count the
  ``<digit>. **`` numbered inventory items, cross-check the distinct
  ``*_pass`` calls in ``main()`` (plus ``_ASW_INLINE_PASS_BLOCKS`` inline
  crash-recovery blocks), and FAIL on any mismatch (#1225; manual
  catch-ups #1021/#1169).
* ``--check-section-reference-pointers`` (also bundled into the no-flags
  default run): scan every ``.claude/rules/*.md`` whose filename ends with
  ``-section-reference.md`` / ``-lens-reference.md`` (the relocated-section
  reference files owned by an agent spec) and FAIL any non-fenced section
  heading at the file's grain (H2 when any non-fenced H2 exists, else H3)
  that has no whitespace-normalized ``§ <exact heading>`` pointer line in
  the owning ``.claude/agents/<agent>.md`` spec; also FAIL an orphan
  reference (no owning spec) and a headingless reference (malformed).
  Closes the #850-class relocated-but-unreachable-section gap (#1159).
* ``--check-git-recipes-root-guard`` (also bundled into the no-flags default
  run): extract every ``bash``/``sh``/``shell``-tagged fenced block from
  ``.claude/agents/*.md`` + ``.claude/skills/**/SKILL.md`` +
  ``.claude/rules/*.md`` + ``CLAUDE.md`` (other worktrees excluded, the
  current worktree scanned), pre-filter to blocks containing the literal
  ``git``, and EXECUTE the live PreToolUse hook
  ``scripts/guard_repo_root_branch.sh`` against each WHOLE block (stdin
  JSON, exactly as a session pasting the recipe into ONE Bash call); hook
  exit 2 → FAIL naming file:fence-opener-line + the hook's first BLOCKED
  line. A per-fence ``<!-- workflow-lint: allow-root-guard-block:
  <reason> -->`` sentinel (non-empty reason, on the immediately-preceding
  non-blank line) waives — for deliberate anti-pattern examples and
  pod-side recipes that never run at the VM repo root. A fail-loud
  positive+negative hook self-test runs FIRST: a missing hook, a fail-open
  hook (``jq`` absent → its stdin parse fail-softs to exit 0), or a
  fail-closed hook is ONE loud lint error, never a silent pass; only
  rc 0/2 are interpreted. Closes the #1047 class: a documented cleanup
  recipe without a per-clause ``git -C`` waiver survived plan review +
  a 6-critic ensemble and was caught only by the code-reviewer executing
  the hook. Executing the REAL hook keeps the check current as detectors
  evolve; the pattern-match siblings (``--check-no-repo-root-*``) stay —
  complementary scope (they also scan prose lines). The guarantee is
  fence-scoped: prose inline-code recipes, ``#``-commented instruction
  lines inside fences, untagged fences, and the placeholder-substitution
  false-PASS direction are NAMED residuals (see the check docstring).
* ``--check-bare-commit-pathspec`` (also bundled into the no-flags default
  run): scan every ``bash``/``sh``/``shell``-tagged fenced block in
  ``.claude/agents/*.md`` + ``.claude/skills/**/SKILL.md`` +
  ``.claude/rules/*.md`` + ``CLAUDE.md`` (the root-guard surface, reused)
  and FAIL any ``git commit`` invocation with no trailing
  `` -- <pathspec>``: a bare commit at the always-concurrent shared repo
  root commits the WHOLE staged index, sweeping sibling sessions' staged
  files onto the commit (incident ``7dbde267f1``, 2026-07-21: 4 foreign
  files swept onto main; #1630 fixed /daily per-file; #1648 generalizes
  the guard mechanically). Structural exemptions: the `` -- <pathspec>``
  tail itself, per-invocation ``git -C <tree>`` scope (a named tree with
  its own index), ``xargs -r``/``--no-run-if-empty``-driven commits (the
  appended file list is a runtime pathspec — a flag-less xargs is NOT
  exempt), ``--dry-run`` previews, and ``#`` comment lines. Waive a
  deliberate anti-pattern example / pod-side recipe with
  ``<!-- workflow-lint: allow-bare-commit-block: <reason> -->`` on the
  line directly above the fence opener. Untagged fences, prose
  inline-code recipes, heredoc bodies, and compound-line quoted-text
  false-exemptions are NAMED residuals (see the check docstring).

Exit codes:

* ``0`` PASS
* ``1`` FAIL — stderr lists every error with file:line context.
"""

from __future__ import annotations

import argparse
import ast
import dataclasses
import functools
import json
import os
import re
import subprocess
import sys
import tempfile
from collections import Counter
from collections.abc import Callable, Collection, Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml

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
        "login",  # built-in /login (CLI credential re-auth; #1027 auth-outage guard docs)
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
        # --- Non-skill prose tokens the backtick form still catches ---
        # (pure-PATH tokens live in SKILL_REF_FS_ROOTS below; `log` stays
        #  here on its dashboard-route justification)
        "intent",  # `/intent` — a phase/arg token in prose
        "absent",  # `/absent` — a marker-state token in prose
        "override",  # `/override subset` prose (experiment-implementer.md)
        "binary",  # `.npz/binary` prose (uploader.md)
        "terminal",  # `blocked/terminal` prose (background-automation.md)
        "expensive-band",  # `auto_run/expensive-band` prose (issue/SKILL.md)
    }
)

# `--check-skill-refs`: bare single-segment backticked absolute PATHS.
# SKILL_REF_RE's trailing lookahead rejects multi-segment paths (`/tmp/x`
# — the next char is `/`) but a bare root (`/tmp`) closes on a backtick
# and matches, so ordinary filesystem paths mis-fired the check (#1445;
# the allowlist had grown ad-hoc path workarounds like `workspace`).
# Members: the Linux FHS top-level directories + the RunPod `/workspace`
# volume convention — PATH tokens, never slash-commands. Kept SEPARATE
# from SKILL_REF_ALLOWLIST so that list keeps its "justify every entry
# as a legitimate slash-command" contract. INVARIANT (pinned by
# tests/test_workflow_lint.py::test_skill_ref_fs_roots_disjoint_from_live_skills_and_allowlist
# and by the in-function collision guard in check_skill_references):
# no member may name a live .claude/skills/ dir — a colliding entry
# would silently disable rot detection for that skill.
SKILL_REF_FS_ROOTS: frozenset[str] = frozenset(
    {
        "bin",
        "boot",
        "dev",
        "etc",
        "home",
        "lib",
        "lib64",
        "media",
        "mnt",
        "opt",
        "proc",
        "root",
        "run",
        "sbin",
        "srv",
        "sys",
        "tmp",
        "usr",
        "var",
        "workspace",  # RunPod volume root (migrated from SKILL_REF_ALLOWLIST)
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

# `--check-piped-git-push` (#1048; commit verb added by #1591): a
# `git push` / `git merge` / `git commit` /
# `gh pr merge|create` PRODUCER piped into a filter on its own pipeline
# segment (`git push origin main 2>&1 | tail -20`). Bash makes the
# compound's exit status the LAST stage's, so the pipe masks the
# producer's non-zero exit code: a rejected push reads as success and the
# session proceeds believing the merge landed (#957's Step 10d push was
# masked 2026-07-04; 4 sessions hit the class on 2026-07-02). For
# `git commit` the pipe carries a SECOND harm: an early-exiting reader
# (`| head -N`) SIGPIPE-kills the commit MID-pre-commit-hook (#1584 killed
# gitleaks mid-scan). The prose
# rule (CLAUDE.md § Concurrent repo-root committers: "run it bare and
# check the exit code, or use `set -o pipefail` when a pipe is
# unavoidable") failed open >=5 times in 3 days; this check makes it
# mechanical for committed `scripts/*.sh`, and the
# `.claude/hooks/guard_piped_git_push.sh` PreToolUse hook covers the
# inline ad-hoc commands that never reach a script — the same dual-engine
# split as `--check-pipe-python` (#753).
#
# Flagged (a logical shell line, backslash continuations merged, `|&`
# normalized to `|` before matching — bash's `|&` is `2>&1 |` shorthand):
#   * `git push | tail -5`, `git push origin main 2>&1 | grep -v x`;
#   * `gh pr merge 123 --squash | head`, `gh pr create ... | grep -o ...`;
#   * `git -C <dir> push ... 2>&1 | tail -20` (flag-tolerant `git` anchor);
#   * `git merge issue-x 2>&1 | tail -5`, `git push 2>err.log | tail`;
#   * `git commit -m "wip" 2>&1 | head -20` (the #1584 incident shape).
# NOT flagged (precision):
#   * `cat msg.txt | git commit -F -` — producer as CONSUMER/final stage
#     (a message piped INTO commit), same channel as `echo foo | git push`
#     below; `git commit --dry-run | head` is skipped by the
#     verb-independent `--dry-run` span skip (a dry-run commit lands
#     nothing and runs no pre-commit hook); `git commit-tree ... | head`
#     never matches (the verb must be followed by whitespace-or-pipe);
#   * `git push origin main || echo failed` — `(?!\|)` rejects `||` (the
#     one real tree shape, issue931_dispatch.sh);
#   * `git merge-base --all main HEAD | head -1` — the verb must be
#     followed by whitespace-or-pipe, so `merge-base` never matches (a
#     canonical .claude/rules/diff-size-budget.md probe);
#   * `echo foo | git push` — producer as CONSUMER/final stage: no
#     trailing `|` after the producer, and the final stage's exit code IS
#     the pipeline's;
#   * `git status | grep x && git push` — the span class cannot cross a
#     `;`/`&&`/`||`/`&` command separator, so the trailing `|` is
#     guaranteed to be a pipe attached to the producer's OWN segment;
#   * a matched span containing `--dry-run` (a dry run lands nothing, so
#     masking its exit code cannot cause the incident) — skipped by
#     check_piped_git_push, not the regex;
#   * `#`-comment lines; and every line at-or-after the FIRST non-comment
#     `pipefail` line in the file (a `set -euo pipefail` header makes
#     every later pipe propagate the failure; `set +o pipefail`
#     re-disable is ignored — fails toward false-NEGATIVE, the documented
#     safe direction for a pre-commit-gating lint, same stance as
#     `_upload_or_true_segments`).
#
# Span class: `[^|;&\n]` blocks command separators, but the alternation
# `&(?=[>0-9])` re-admits the `&` INSIDE redirection operators (`2>&1`,
# `>&2`, `&>file`, `&>>file`) — without it the span cannot cross `2>&1`
# and the flagship incident shape `git push origin main 2>&1 | tail -20`
# is MISSED (the #1048 Phase 1.5 fact-checker finding on plan v1). Known
# accepted residual: an exotic no-space background separator immediately
# followed by a digit- or `>`-starting command (`git push &9foo | tail`)
# extends the span — fails toward FLAG (a lint error a human reviews),
# the safe direction. No waiver token in v1 (YAGNI, the
# `check_pipe_python` stance): the committed tree is clean (sole prior
# hit, issue931_dispatch.sh, is an `||` disjunction), so nothing to waive
# or grandfather. Like the sibling, the regex is line-local and NOT
# quote-aware — a quoted string carrying the literal pattern matches
# (document the bad pattern in a `#`-comment, not an echo/quoted string).
_PIPED_PUSH_SPAN = r"(?:[^|;&\n]|&(?=[>0-9]))*"
_PIPED_PUSH_GIT = (
    r"\bgit\s+(?:-[^\s|;&]+(?:\s+[^\s|;&]+)?\s+)*(?:push|merge|commit)"
    r"(?:\s" + _PIPED_PUSH_SPAN + r")?\|(?!\|)"
)
_PIPED_PUSH_GH = r"\bgh\s+pr\s+(?:merge|create)(?:\s" + _PIPED_PUSH_SPAN + r")?\|(?!\|)"
PIPED_GIT_PUSH_RE = re.compile(_PIPED_PUSH_GIT + "|" + _PIPED_PUSH_GH)

# `--check-push-failure-swallow` (#1205): a `git push` whose failure is
# swallowed ON THE SAME LOGICAL LINE by `|| echo` / `|| true` / `|| :` /
# `|| printf` — the workload-side `||` sibling of the piped-push
# exit-code-masking class above (#957/#1048). The swallow declares
# success while the result commit never landed; on GCE the
# self-DELETEing instance then holds the ONLY copy of the commit
# (incidents #825 r6/r7/r8, upload-verification reads
# 2026-07-08T11:17/11:19Z: `git push origin "issue-825" || echo
# "... WARNING: git push failed"` swallowed a deterministic auth failure
# three rounds running; 73 eval JSONs were rescued by hand with ~2.5 h of
# margin). Unlike the pipe sibling, `pipefail` is NO escape — it never
# applies to `||` disjunctions — so there is no pipefail carve-out here.
#
# Flagged (per logical shell line, backslash continuations merged by
# `_iter_logical_shell_lines`):
#   * `git push origin x || echo warn`, `git push || true`,
#     `git -C <dir> push ... || :`, `git push ... || printf 'w'`;
#   * the backslash-continued shape (`git push origin x \` newline
#     `  || echo warn`) — merged before matching.
# NOT flagged (precision — same-line-only keeps live-tree false
# positives at zero, verified against the three safe shapes on main):
#   * `if git push origin x; then` (auto_push_main.sh:23 — the rc is
#     CONSUMED, not swallowed);
#   * bare pushes (cron_export_literature.sh:41 — set -e propagates);
#   * `git push A || { sleep 20; git push A; } || true` retry groups —
#     the span class `[^|;&\n]*` cannot cross `;`/`{`, and the group
#     alternation matches only echo/true/:/printf immediately after
#     `||` (the rendered #1205 GCE leg's own retry uses exactly this
#     shape: the re-COUNT after the retry is the verification, so the
#     terminal `|| true` there is not a swallow);
#   * `#`-comment lines; lines waived via `# PUSH_SWALLOW_EXEMPT:
#     <reason>` (same/preceding line, `_sh_waiver_present` semantics).
# Like the sibling, the regex is line-local and NOT quote-aware — a
# quoted string carrying the literal pattern matches (document the bad
# pattern in a `#`-comment, not an echo/quoted string).
PUSH_FAILURE_SWALLOW_RE = re.compile(
    r"\bgit\s+(?:-[^\s|;&]+(?:\s+[^\s|;&]+)?\s+)*push\b[^|;&\n]*"
    r"\|\|\s*(?:echo\b|true\b|printf\b|:(?![^\s;|&]))"
)
PUSH_SWALLOW_WAIVER_RE = re.compile(r"#\s*PUSH_SWALLOW_EXEMPT\s*:\s*(.+?)\s*$")
PUSH_SWALLOW_WAIVER_MIN_REASON_CHARS = 8

# Grandfathered legacy offenders — repo-root-relative POSIX paths, FROZEN
# (frozenset; the FROZENNESS pattern of PHASE_DONE_EDGE_LEGACY_ALLOWLIST —
# path grain here, not edge grain: the offense is the emitting script's
# own line, no invoker involved). These scripts predate the contract and
# are covered by the #1205 GCE push-verify backstop (NOT an endorsement of
# the shape — fixing them is prune-on-touch, and the backstop re-pushes /
# fail-louds what their swallow hides). Derivation: live grep of `main`
# AND every `issue-*` branch's scripts/*.sh, 2026-07-09 (task #1205); the
# three issue825 sep-dispatch siblings live on the unmerged `issue-825`
# branch and land on main at #825's Step-10d merge — pre-seeded so that
# merge does not break the no-flags run. A NEW script with the swallow
# shape is still flagged.
PUSH_SWALLOW_LEGACY_ALLOWLIST: frozenset[str] = frozenset(
    {
        # the one live on-main hit (experiment entrypoint — not edited):
        "scripts/issue931_dispatch.sh",
        # PRE-SEEDED for the in-flight issue-825 branch merge (all three
        # carry the same `git push ... || echo WARNING` line — :502 /
        # :376 / :257 respectively, verified 2026-07-09):
        "scripts/issue825_sampled_sep_dispatch.sh",
        "scripts/issue825_onpolicy_sep_dispatch.sh",
        "scripts/issue825_base_sep_dispatch.sh",
    }
)

# `--check-upload-or-true` (#1036): an upload / result-persist /
# result-production command whose failure is swallowed by `|| true` /
# `|| :` / `; true` in a `scripts/**/*.sh` shell line (#841: swallowed
# plot-phase failures compounded a missing upload phase — stage
# JSONs/plots were silently lost across attempts until the fail-loud fix
# commits 4ece51a22a / 0efbce6575 removed the swallows).
#
# Flagged (per logical shell line, backslash continuations merged,
# trailing comments stripped quote-aware):
#   * TERMINAL swallow (`|| true` / `|| :` / `; true` at line end) +
#     an upload/result token ANYWHERE on the line — bash `&&`/`||` are
#     equal-precedence left-associative, so a terminal swallow masks the
#     WHOLE preceding chain (`upload && echo ok || true`,
#     `{ upload; } || true`);
#   * NON-terminal `|| true` / `|| :` + a token in the SAME `&&`/`;`
#     segment (preserves the `mkdir || true && upload` FP kill);
#   * a swallowed heredoc opener whose BODY calls an upload helper
#     (`... <<'PY' 2>&1 || true` + body `api.upload_file(` — the
#     i632_dispatch_with_log_capture.sh:30 shape a line-only scan misses);
#   * a swallowed multi-line `python -c "…"` quoted block whose BODY
#     calls an upload helper (the CURRENT #841 upload-phase shape —
#     `upload_split_lfs_to_overflow(` bodies).
# NOT flagged:
#   * `#`-comment lines (issue841_scaling_dispatch.sh:85 is a comment
#     containing both `|| true` and "upload") and `echo `-prefixed lines
#     (an echo performs no upload; known accepted FN:
#     `echo "…"; upload || true` merged on ONE logical line is skipped
#     whole — frozen by a test fixture);
#   * `clean_experiment_downloads.py … || true` ("downloads" contains no
#     "upload" substring), `ls eval_results/ || true` (bare result-dir
#     names are tokens ONLY inside the `git add|commit` alternation);
#   * lines waived via `# UPLOAD_OR_TRUE_EXEMPT: <reason>` (reason ≥ 10
#     chars; same logical line or immediately preceding non-blank line —
#     the CVD_PIN_EXEMPT convention). Deliberate best-effort side
#     channels (crash-diagnostics uploads on a FAILURE path) use this.
# Named residual evasion shapes (fail-toward-false-negative, the safe
# direction for a pre-commit-gating lint — all deliberate scope bounds):
#   * `|| echo WARN` swallows (leave a log trace, lesser severity than
#     `|| true`'s pure silence);
#   * `|| rc=$?`-then-ignore and `set +e` blocks;
#   * shell-function-call swallows (`do_upload || true` — the function
#     name carries no token unless it contains "upload");
#   * a multi-line subshell wrapper with the swallow on the CLOSING paren
#     (`(cmd <<'PY' … PY` newline `) || true`) — no live instance;
#   * the naive quote-unaware `&&`/`;` segment split can mis-split on
#     quoted separators (non-terminal rule only);
#   * a MISSING upload phase (the other half of #841's loss) is not
#     lintable and is named here, not claimed.
UPLOAD_OR_TRUE_SWALLOW_OR_RE = re.compile(r"\|\|\s*(?:true\b|:(?=[\s;)&|]|$))")
# Terminal swallow — masks the WHOLE line/chain:
UPLOAD_OR_TRUE_SWALLOW_TERMINAL_RE = re.compile(r"(?:\|\|\s*(?:true|:)|;\s*true)\s*$")
# Shell-line upload/result-persist/result-production tokens. Notes:
#   * `upload_raw_completions\w*` (NOT a trailing `\b` before `_`) — the
#     canonical helper is upload_raw_completions_to_data_repo;
#   * `*upload*.py` — upload helper scripts (verify_uploads.py, …);
#   * `*plot*.py` — result-production (plot) scripts, the founding #841
#     offender shape (issue841_scaling_plots.py);
#   * `git add|commit` of result dirs + `git push` — git persistence;
#   * `$HF_DATA_REPO` / `$HF_MODEL_REPO` — repo-destination env vars.
UPLOAD_OR_TRUE_LINE_TOKEN_RE = re.compile(
    r"(?:"
    r"\bupload_file\b|\bupload_folder\b|\bupload_raw_completions\w*"
    r"|\bhf\s+upload\b|\bhuggingface-cli\s+upload\b"
    r"|\b[A-Za-z0-9_]*upload[A-Za-z0-9_]*\.py\b"
    r"|\b[A-Za-z0-9_]*plot[A-Za-z0-9_]*\.py\b"
    r"|\bgit\s+push\b"
    r"|\bgit\s+(?:add|commit)\b[^#]*\b(?:eval_results|figures|ood_eval_results)\b"
    r"|\$\{?HF_DATA_REPO\b|\$\{?HF_MODEL_REPO\b"
    r")"
)
# Quoted/heredoc BODY upload-call tokens (inline-python upload blocks — the
# dominant dispatcher upload shape; the widened `\w*upload\w*(` covers
# upload_file(, upload_folder(, _upload(, api.upload_file(,
# upload_split_lfs_to_overflow():
UPLOAD_OR_TRUE_BODY_TOKEN_RE = re.compile(
    r"(?:\b[A-Za-z0-9_]*upload[A-Za-z0-9_]*\s*\("
    r"|\bupload_raw_completions\w*|\bcreate_commit\s*\(|\.push_to_hub\s*\()"
)
UPLOAD_OR_TRUE_WAIVER_RE = re.compile(r"#\s*UPLOAD_OR_TRUE_EXEMPT:\s*(.+)")
UPLOAD_OR_TRUE_WAIVER_MIN_REASON_CHARS = 10
# Multi-line `python -c "` opener (captures the quote char); a block whose
# captured quote is unclosed on the logical line is consumed physically
# until the first line containing the closing quote char, bounded below.
UPLOAD_OR_TRUE_PYC_OPENER_RE = re.compile(r"\bpython3?\s+-c\s+([\"'])")
UPLOAD_OR_TRUE_PYC_MAX_BODY_LINES = 300  # bounded; unclosed at EOF/cap -> skip (FN-safe)

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
    ("critic-lean.md", "WebFetch"): (
        "lean twin (#2062): the body's `no WebSearch/WebFetch` line NEGATES "
        "the tool — descriptive-not-instructive; the lean drops WebFetch by "
        "design (fixed-overhead reduction is the whole point of the twin)"
    ),
    ("critic-lean.md", "WebSearch"): (
        "lean twin (#2062): the body's `no WebSearch/WebFetch` line NEGATES "
        "the tool — descriptive-not-instructive; the lean drops WebSearch by "
        "design (fixed-overhead reduction is the whole point of the twin)"
    ),
    ("critic-lean.md", "mcp__arxiv"): (
        "lean twin (#2062): the body's `(no mcp__arxiv, no mcp__arxiv-latex)` "
        "line NEGATES the tool — descriptive-not-instructive; the lean drops "
        "MCP tools by design (fixed-overhead reduction is the whole point)"
    ),
    ("critic-lean.md", "mcp__arxiv-latex"): (
        "lean twin (#2062): the body's `(no mcp__arxiv, no mcp__arxiv-latex)` "
        "line NEGATES the tool — descriptive-not-instructive; the lean drops "
        "MCP tools by design (fixed-overhead reduction is the whole point)"
    ),
    ("planner-lean.md", "WebFetch"): (
        "lean twin (#2062): the body's `no WebSearch/WebFetch` line NEGATES "
        "the tool — descriptive-not-instructive; the lean drops WebFetch by "
        "design (fixed-overhead reduction is the whole point of the twin)"
    ),
    ("planner-lean.md", "WebSearch"): (
        "lean twin (#2062): the body's `no WebSearch/WebFetch` line NEGATES "
        "the tool — descriptive-not-instructive; the lean drops WebSearch by "
        "design (fixed-overhead reduction is the whole point of the twin)"
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


# `--check-hub-dir-filecount` (#1190): the Hub rejects any single repo
# directory holding >10000 files at COMMIT time — a NON-retriable
# BadRequestError AFTER the full compute ran and all bytes are staged (#658
# r2). The shared hub helpers (`hub._upload` folder branch /
# `hub._upload_folder_filtered`) pre-count staged files per TARGET repo dir
# via `assert_hub_dir_filecounts` and fail loud BEFORE any network I/O; this
# lint funnels DIRECT `upload_folder(` call sites in scripts/ (the #658
# incident's own call path bypassed hub.py entirely) toward that one-line
# guard. Inline waiver for a genuinely-correct flagged call. Reason ≥ 10
# chars, same convention as UPLOAD_AS_FILE_EXEMPT / CVD_PIN_EXEMPT.
HUB_DIR_FILECOUNT_WAIVER_RE = re.compile(r"#\s*HUB_DIR_FILECOUNT_EXEMPT\s*:\s*(.+?)\s*$")
HUB_DIR_FILECOUNT_WAIVER_MIN_REASON_CHARS = 10
# Grandfathered pre-existing direct `upload_folder(` call sites — legacy
# EXPERIMENT code the workflow-fix scope bars #1190 from editing. Rebuilt
# mechanically from the live tree (grep -rln "upload_folder(" scripts/
# --include='*.py', minus wrapper-routed / local-`def upload_folder` files)
# and pinned by
# tests/test_workflow_lint.py::test_check_hub_dir_filecount_live_tree_passes.
# NOT hand-maintained: a NEW direct caller must call
# assert_hub_dir_filecounts (or carry a HUB_DIR_FILECOUNT_EXEMPT waiver),
# never extend this set. Repo-root-relative posix paths.
HUB_DIR_FILECOUNT_LEGACY_ALLOWLIST: frozenset[str] = frozenset(
    {
        "scripts/archive/upload_and_clean.py",  # 1 pre-#1190 direct site; archived legacy uploader
        "scripts/issue1073_common.py",  # 1 pre-#1190 direct site; legacy experiment code
        "scripts/issue540_jsrb_predictor.py",  # 1 pre-#1190 direct site; legacy experiment code
        "scripts/issue545_sweep.py",  # 8 pre-#1190 direct sites; legacy experiment code
        "scripts/issue594_extract_context_vectors.py",  # 2 pre-#1190 direct sites; legacy expt
        "scripts/issue604_extract_context_vectors.py",  # 1 pre-#1190 direct site; legacy expt
        "scripts/issue617_upload_corpus.py",  # 2 pre-#1190 direct sites; legacy experiment code
        "scripts/issue634_extract_behavior_vectors.py",  # 2 pre-#1190 direct sites; legacy expt
        "scripts/issue650_extract_context_bank.py",  # 1 pre-#1190 direct site; legacy expt
        "scripts/issue658_extract_base_store.py",  # 3 pre-#1190 direct sites; the #658 incident rig
        "scripts/issue661_extract_directions.py",  # 1 pre-#1190 direct site; legacy experiment code
        "scripts/issue661_generate_arm_a.py",  # 1 pre-#1190 direct site; legacy experiment code
        "scripts/issue664_dispatch.py",  # 3 pre-#1190 direct sites; legacy experiment code
        "scripts/issue667_save_maps.py",  # 1 pre-#1190 direct site (bare-name import form)
        "scripts/issue744_dump_and_stream.py",  # 2 pre-#1190 direct sites; legacy experiment code
        "scripts/issue763_build_probe_pools.py",  # 2 pre-#1190 direct sites; legacy expt
        "scripts/issue763_cofit_upload.py",  # 2 pre-#1190 direct sites; legacy experiment code
        "scripts/issue763_disclosure_flag_audit.py",  # 1 pre-#1190 direct site; legacy expt
        "scripts/issue763_extract_pv_rb.py",  # 2 pre-#1190 direct sites; legacy experiment code
        "scripts/issue763_judge_e0.py",  # 1 pre-#1190 direct site; legacy experiment code
        "scripts/issue763_upload.py",  # 2 pre-#1190 direct sites; legacy experiment code
        "scripts/issue778_v2_upload.py",  # 1 pre-#1190 direct site; legacy experiment code
        "scripts/issue779_capture_answer_summaries.py",  # 1 pre-#1190 direct site; legacy expt
        "scripts/issue779_capture_answer_summaries_pass2.py",  # 1 pre-#1190 direct site; legacy
        "scripts/issue779_collect.py",  # 2 pre-#1190 direct sites; legacy experiment code
        "scripts/issue779_extract_rb.py",  # 2 pre-#1190 direct sites; legacy experiment code
        "scripts/issue779_gen_behavior_corpus.py",  # 2 pre-#1190 direct sites; legacy expt
        "scripts/issue779_pertoken_lmsys_capture.py",  # 1 pre-#1190 direct site; legacy expt
        "scripts/issue779_reliability_gen_capture.py",  # 1 pre-#1190 direct site; legacy expt
        "scripts/issue810_common.py",  # 1 pre-#1190 direct site; legacy experiment code
        "scripts/issue810_extract_positions.py",  # 1 pre-#1190 direct site; legacy expt
        "scripts/issue825_gen_conversations.py",  # 1 pre-#1190 direct site (bare-name import form)
        "scripts/issue833_extract_onpolicy.py",  # 4 pre-#1190 direct sites; legacy experiment code
        "scripts/issue920_extract_summaries.py",  # 1 pre-#1190 direct site; legacy expt
        "scripts/issue920_gen_completions_b.py",  # 1 pre-#1190 direct site; legacy expt
        "scripts/issue920_nulls_figures.py",  # 1 pre-#1190 direct site; legacy experiment code
        "scripts/issue922_common.py",  # 1 pre-#1190 direct site; legacy experiment code
        "scripts/issue928_common.py",  # 1 pre-#1190 direct site; legacy experiment code
        "scripts/issue_642/i642_dispatch.py",  # 2 pre-#1190 direct sites; legacy experiment code
    }
)


# `--check-upload-file-in-loop` (#1544; incidents #664 / #658 r4 / #1481):
# a per-file `upload_file` LOOP of N files = N commits + N server-side
# pre-checks — 504-storms on a large repo (#664: 1425-file loop, 12h idle
# 8xH200, ~$530) and trips the org-level ~2500-req/5-min 429 quota
# (#1481: ~1400 planned per-file commits -> HF 429 storm). Bulk uploads
# compose ONE upload_folder commit. Inline waiver for a genuinely bounded
# loop (retry wrapper around a single file; a fixed <=3-item list). Reason
# >= 10 chars, same convention as UPLOAD_AS_FILE_EXEMPT.
UPLOAD_LOOP_WAIVER_RE = re.compile(r"#\s*UPLOAD_LOOP_EXEMPT\s*:\s*(.+?)\s*$")
UPLOAD_LOOP_WAIVER_MIN_REASON_CHARS = 10
# Grandfathered pre-existing in-loop per-file-upload call sites (shape A:
# upload_file; shape B: _upload(..., upload_as_file=True)) — legacy
# EXPERIMENT code the workflow-fix scope bars #1544 from editing. Rebuilt
# mechanically from the live tree (run the finished check with
# legacy_allowlist={}) and pinned by tests/test_workflow_lint.py::
# test_check_upload_file_in_loop_allowlist_load_bearing (EXACT per-file
# site counts — a new offense in a grandfathered file breaks the pin). NOT
# hand-extended: a NEW in-loop call must batch into one upload_folder
# commit (or carry an UPLOAD_LOOP_EXEMPT waiver), never extend this set.
# Dict: repo-root-relative posix path -> expected site count.
UPLOAD_FILE_IN_LOOP_LEGACY_ALLOWLIST: dict[str, int] = {
    # shape A — in-loop upload_file (6 sites / 5 files):
    "scripts/issue661_freeze_instructions.py": 1,  # L252: 2-item tuple loop
    "scripts/issue763_upload.py": 1,  # L153: 2-item list loop, deliberate non-LFS per-file
    "scripts/issue952_noise_ceiling_gpu.py": 1,  # L422: 3-item list loop
    "scripts/issue958_common.py": 1,  # L679: for-attempt retry wrapper, single probe file
    "scripts/run_experiment_444.py": 2,  # L5532, L5557: per-cell + per-persona loops
    # shape B — in-loop _upload(..., upload_as_file=True) (21 sites / 15 files):
    "scripts/issue1112_dispatch.py": 2,  # L424, L1035
    "scripts/issue1333_dispatch.py": 2,  # L1494, L1501
    "scripts/issue1345_rejudge_malformed.py": 1,  # L381
    "scripts/issue1417_gen.py": 1,  # L205
    "scripts/issue1417_judge.py": 2,  # L554, L571
    "scripts/issue1482_error_analysis.py": 1,  # L1886
    "scripts/issue559_base_prior_persona_panel.py": 1,  # L831
    "scripts/issue560_crossrecipe_panel.py": 1,  # L1011
    "scripts/issue595_prefix_carrier.py": 1,  # L1312
    "scripts/issue640_postfix_carrier.py": 1,  # L822
    "scripts/issue664_dispatch.py": 3,  # L1844, L1924, L2003
    "scripts/issue778_upload.py": 1,  # L113
    "scripts/issue841_scaling_common.py": 2,  # L505, L516
    "scripts/issue923_reduce_spans.py": 1,  # L339
    "scripts/run_issue_360_target_logprobs.py": 1,  # L1669
}


# `--check-upload-return-discard` (#2087; incident #2054): the two shared
# hub upload helpers are fail-soft BY RETURN — `_upload` returns "" on
# missing HF_TOKEN / absent local path / failed verify (and on upload
# exceptions unless raise_on_error=True); `_upload_folder_filtered`
# returns "" on every upload-failure shape (only its pre-flight
# assert_hub_dir_filecounts guard raises, hub.py ~1744, the #1190 cap
# check). A caller that discards the return converts silent durability
# loss into exit 0 (.claude/rules/upload-policy.md: "'upload returned no
# path' is a TRACKED GAP ... never a warning-and-continue"). Inline
# waiver for a deliberate fail-soft caller; reason >= 10 chars, same
# convention as UPLOAD_AS_FILE_EXEMPT / UPLOAD_LOOP_EXEMPT.
UPLOAD_RETURN_DISCARD_WAIVER_RE = re.compile(r"#\s*UPLOAD_RETURN_DISCARD_EXEMPT\s*:\s*(.+?)\s*$")
UPLOAD_RETURN_DISCARD_WAIVER_MIN_REASON_CHARS = 10
# Grandfathered pre-existing discarded-return call sites — legacy
# EXPERIMENT code the workflow-fix scope bars #2087 from editing
# (grandfather-all posture; the UPLOAD_FILE_IN_LOOP precedent). Rebuilt
# mechanically from the live tree (run the finished check with
# legacy_allowlist={}). Check gate is <=-tolerant (findings suppressed
# while count <= grandfathered N; an excess reports ALL of the file's
# findings), so a count DROP from a sibling's fix merging keeps main
# green. Pinned by tests/test_workflow_lint.py::
# test_check_upload_return_discard_allowlist_load_bearing with SPLIT
# semantics: EXACT per-file counts for stable entries; observed <= pinned
# for UPLOAD_RETURN_DISCARD_PENDING_OWNER entries. NOT hand-extended: a
# NEW discard must capture-and-raise (the
# hub.upload_raw_completions_to_data_repo shape) or carry an
# UPLOAD_RETURN_DISCARD_EXEMPT waiver, never extend this set.
# Dict: repo-root-relative posix path -> grandfathered site count.
# Enumerated mechanically 2026-08-05 (75 sites / 43 files; empty-allowlist run
# on the #2087 worktree at origin/main 89207ffc50).
UPLOAD_RETURN_DISCARD_LEGACY_ALLOWLIST: dict[str, int] = {
    # ── stable entries (owning task terminal/idle; EXACT-count pinned) ──
    "scripts/issue1112_dispatch.py": 2,  # L424, L1035
    "scripts/issue1112_rankem_dispatch.py": 4,  # L1280, L1301, L1333, L1352
    "scripts/issue1112_rankem_prep_corpus.py": 1,  # L132
    # L272 (raise_on_error=True; the '' returns stay silent):
    "scripts/issue1310_recapture_script_store.py": 1,
    "scripts/issue1333_dispatch.py": 6,  # L1494, L1501, L2306, L2679, L2695, L2739
    "scripts/issue1333_geometry.py": 3,  # L843, L868, L874
    "scripts/issue1482_early_layer.py": 2,  # L963, L992
    "scripts/issue1482_error_analysis.py": 1,  # L1939
    "scripts/issue1482_matryoshka_tier.py": 2,  # L926, L952
    "scripts/issue1482_run_length.py": 1,  # L1027
    "scripts/issue1586_dispatch.py": 1,  # L3024
    "scripts/issue1689_capture.py": 1,  # L409 (function-local hub import at L401)
    "scripts/issue1768_lasttoken.py": 1,  # L327
    "scripts/issue1768_lasttoken_gate.py": 1,  # L249
    "scripts/issue1768_map_augmentation.py": 1,  # L1287
    "scripts/issue1774_draws.py": 2,  # L431, L440
    "scripts/issue1774_steering.py": 1,  # L509
    "scripts/issue1900_gpu.py": 2,  # L1783, L1798
    "scripts/issue1900_judge.py": 1,  # L814
    "scripts/issue1900_offfloor.py": 2,  # L797, L1107
    "scripts/issue1900_tfm.py": 6,  # L561, L721, L729, L957, L967, L1466
    "scripts/issue595_prefix_carrier.py": 1,  # L1312
    "scripts/issue640_postfix_carrier.py": 1,  # L822
    "scripts/issue664_dispatch.py": 3,  # L1844, L1924, L2003
    "scripts/issue734_dispatch.py": 1,  # L954
    "scripts/issue825_kresample_user_capture.py": 2,  # L287, L294
    "scripts/issue825_kresample_user_gen.py": 2,  # L249, L256
    "scripts/issue923_capture.py": 3,  # L995, L1002, L1027
    "scripts/issue923_figures.py": 1,  # L232
    "scripts/issue923_fit_decomposition.py": 2,  # L1658, L1661
    "scripts/issue923_reduce_spans.py": 1,  # L339
    # ── PENDING_OWNER entries (live in-flight owner; <=-pinned) ─────────
    # owned by in-flight #2054 (live session at 2026-08-05; open concern
    # upload-mirror-return-discard — its branch already converts phase_a
    # discards to capture-and-raise); #2054's merge zeroing a count
    # retires that entry + its PENDING_OWNER row on the next
    # lint-touching round:
    "scripts/issue2054_capture.py": 1,  # L943 (function-local hub import at L914)
    "scripts/issue2054_fits.py": 1,  # L949
    "scripts/issue2054_ladder.py": 1,  # L982
    "scripts/issue2054_phase_a.py": 2,  # L1067, L1089
    "scripts/issue2054_phase_b.py": 1,  # L255
    "scripts/issue2054_phase_c.py": 1,  # L445
    "scripts/issue2054_phase_d.py": 1,  # L489
    # owned by in-flight #1739 (live session at 2026-08-05,
    # followups_running — its rounds may still edit these scripts); a
    # merged #1739 round zeroing a count retires that entry + its
    # PENDING_OWNER row on the next lint-touching round:
    "scripts/issue1739_armfill_upload.py": 1,  # L63
    "scripts/issue1739_bareq_pod.py": 1,  # L755
    "scripts/issue1739_pvsynth_arms_run.py": 2,  # L256, L376
    "scripts/issue1739_pvsynth_score.py": 2,  # L586, L594
    "scripts/issue1739_wcrung_arms_run.py": 2,  # L356, L487
}
# Allowlist entries owned by a task with a LIVE in-flight session at
# #2087 implementation time (spawn_session.py list, 2026-08-05) — the
# load-bearing test asserts observed <= pinned for these (never exact),
# so main stays green whichever order the owner's fix and this lint
# merge. RETIREMENT CONVENTION: once the owning task's fix lands and its
# file's empty-allowlist count reads 0 (or the session ends with the
# sites unfixed and the count is stable), the NEXT lint-touching round
# moves the entry to a stable exact pin (count > 0) or deletes it
# (count 0) TOGETHER with its allowlist row. Accepted residual: while an
# entry sits at count 0 against a pinned N > 0, up to N NEW discards in
# exactly that file are suppressed at check level (bounded,
# status-quo-preserving; exact pins would redden main fleet-wide on the
# sibling's merge).
UPLOAD_RETURN_DISCARD_PENDING_OWNER: frozenset[str] = frozenset(
    {
        # owned by in-flight #2054 (live round-11 implementer; its branch
        # already converts phase_a's discards to capture-and-raise) — each
        # entry retires per the convention above when #2054's fix merges:
        "scripts/issue2054_capture.py",
        "scripts/issue2054_fits.py",
        "scripts/issue2054_ladder.py",
        "scripts/issue2054_phase_a.py",
        "scripts/issue2054_phase_b.py",
        "scripts/issue2054_phase_c.py",
        "scripts/issue2054_phase_d.py",
        # owned by in-flight #1739 (live session, followups_running — its
        # rounds may still edit these scripts) — each entry retires per the
        # convention above when a #1739 round merges a fix (count drops)
        # or the session ends with the counts stable:
        "scripts/issue1739_armfill_upload.py",
        "scripts/issue1739_bareq_pod.py",
        "scripts/issue1739_pvsynth_arms_run.py",
        "scripts/issue1739_pvsynth_score.py",
        "scripts/issue1739_wcrung_arms_run.py",
    }
)


# `--check-upload-prefix-clobber` (#1452 / incident #1005): reused #928
# fitter scripts uploaded #1005 tensors to hardcoded `issue928_*` prefixes
# on the HF data repo, OVERWRITING the parent issue's artifacts
# (upload-verification FAIL 2026-07-16; parent restored from a pinned
# revision). Two write-scoped rules over `scripts/issue<N>_*.py` upload
# call sites: Rule A — a destination token `issue<M>_…` with M != N (a
# copied/reused uploader writing into another issue's prefix); Rule B — an
# own-issue token arriving via a FALLBACK channel (`x or CONST`, an
# argparse `default=`, a wrapper-param signature default) that a reusing
# child silently inherits. A DIRECT own-prefix hardcode is the sanctioned
# Upload Policy norm and never flags; cross-issue READS (`list_repo_tree` /
# `hf_hub_download`) are out of scope by construction (the check keys on
# write-function identity + dest-argument slot, never on kwarg name alone).
# Inline waiver, reason >= 10 chars, same convention as
# UPLOAD_AS_FILE_EXEMPT / HUB_DIR_FILECOUNT_EXEMPT.
UPLOAD_PREFIX_WAIVER_RE = re.compile(r"#\s*UPLOAD_PREFIX_EXEMPT\s*:\s*(.+?)\s*$")
UPLOAD_PREFIX_WAIVER_MIN_REASON_CHARS = 10
# Write-destination spec table: callable name (bare name or attribute tail)
# -> (dest kwarg, dest positional index or None). `upload_model` /
# `upload_dataset` / `upload_dataset_directory` are deliberately OUT of the
# v1 set (their dests derive from runtime config, not hardcoded module
# constants); extending the table later is a one-line change.
UPLOAD_DEST_FUNCS: dict[str, tuple[str, int | None]] = {
    "upload_file": ("path_in_repo", None),
    "upload_folder": ("path_in_repo", None),
    "CommitOperationAdd": ("path_in_repo", None),
    "_upload": ("path_in_repo", 3),
    "_upload_folder_filtered": ("path_in_repo", 3),
    "upload_raw_completions_to_data_repo": ("experiment_name", 0),
}
# Issue-prefix tokens inside string literals ("issue928_cot/…" -> "928"),
# and the owning issue number of a scripts/issue<N>_*.py basename.
_UPC_ISSUE_TOKEN_RE = re.compile(r"\bissue(\d+)[_/]")
_UPC_OWN_ISSUE_RE = re.compile(r"^issue(\d+)_")
# Grandfathered pre-existing Rule-B fallback destinations (Rule A is NEVER
# silently allowlisted — a Rule-A hit is investigated: waiver-with-reason if
# deliberate, fixed/reported if a latent bug). Rebuilt mechanically from the
# live tree by running the finished check with `legacy_allowlist=frozenset()`;
# pinned by tests/test_workflow_lint.py::
# test_check_upload_prefix_clobber_live_trees_pass and every entry is
# asserted load-bearing by
# ...::test_check_upload_prefix_clobber_allowlist_load_bearing. NOT
# hand-extended: a NEW fallback destination must use `default=None` + a
# fail-loud raise (or carry an UPLOAD_PREFIX_EXEMPT waiver) — never a new
# entry here. Repo-root-relative posix paths. Every entry remains a live
# #1005 runtime-reuse channel until fixed — fixing the file to
# `default=None` + raise retires its entry.
UPLOAD_PREFIX_CLOBBER_ALLOWLIST: frozenset[str] = frozenset(
    {
        # `--hf-prefix` default=HF_PREFIX_DEFAULT feeding a function-local
        # path_in_repo f-string; remains a live #1005 runtime-reuse channel
        # until fixed — default=None + raise retires this entry:
        "scripts/issue1092_figures.py",
        # in-loop `pir or DATA_PREFIX_1434` or-fallback at hub._upload dests;
        # remains a live #1005 runtime-reuse channel until fixed — explicit
        # per-item dests (default=None + raise shape) retire this entry:
        "scripts/issue1434_pv.py",
        # `--hf-samples-path`-class argparse default feeding upload_folder;
        # remains a live #1005 runtime-reuse channel until fixed —
        # default=None + raise retires this entry:
        "scripts/issue540_jsrb_predictor.py",
        # `--hf-prefix` default=HF_PREFIX → store upload; remains a live
        # #1005 runtime-reuse channel until fixed — default=None + raise
        # retires this entry:
        "scripts/issue779_ffc_n10k_generate_capture.py",
        # `--hf-prefix` default=HF_PREFIX → store upload; remains a live
        # #1005 runtime-reuse channel until fixed — default=None + raise
        # retires this entry:
        "scripts/issue779_pertoken_lmsys_capture.py",
        # `--hf-prefix` default=HF_PREFIX → store upload; remains a live
        # #1005 runtime-reuse channel until fixed — default=None + raise
        # retires this entry:
        "scripts/issue779_reliability_gen_capture.py",
        # THE #1005 incident file: `args.upload_prefix or FIT_RESULTS_PREFIX`
        # + `--decomp-upload-prefix` default=DECOMP_TENSORS_PREFIX; remains a
        # live #1005 runtime-reuse channel until fixed — default=None + raise
        # retires this entry:
        "scripts/issue928_fit_decomposition.py",
        # `--results-upload-prefix` / `--tensors-upload-prefix` defaults (the
        # #928 post-incident remediated shape — prose warning only); remains
        # a live #1005 runtime-reuse channel until fixed — default=None +
        # raise retires this entry:
        "scripts/issue928_mlp_indiv_control.py",
    }
)


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
# Signal (e) gate (#1162): a module-level glob/rglob/iglob call (method or
# bare function) whose pattern argument mentions "jsonl" widens the per-node
# predicate to ANY read_text-bearing splitlines receiver in that module (the
# #1132 sweep_parked_wf_candidates.py evasion: a *.jsonl-globbing module's
# helpers reading the globbed files through generically-named
# variables/parameters).
JSONL_GLOB_FUNC_NAMES = frozenset({"glob", "rglob", "iglob"})
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


# `--check-scripts-import-guard` (#823/#853): in script mode
# (`python /abs/path/driver.py`) sys.path[0] is the SCRIPT's own directory —
# not cwd, not the repo root — so an unguarded `scripts.*` import in an
# `src/explore_persona_space/experiments/**` or `scripts/**` (#1229) driver raises
# ModuleNotFoundError pod/GCE-side; deferred (function-body) instances crash
# MID-RUN after paid GPU phases (#823 Phase-3). Inline waiver for a
# genuinely-safe flagged site. Reason ≥ 10 chars, same convention as
# JSONL_SPLITLINES_EXEMPT. No legacy allowlist — the live tree is clean
# (allowlists exist only where live offenders were frozen).
SCRIPTS_IMPORT_GUARD_WAIVER_RE = re.compile(r"#\s*SCRIPTS_IMPORT_GUARD_EXEMPT\s*:\s*(.+?)\s*$")
SCRIPTS_IMPORT_GUARD_WAIVER_MIN_REASON_CHARS = 10
# Guard-evidence callee-name signal: a Call whose callee name mentions
# syspath/sys_path — matches the `_ensure_repo_root_on_syspath()` exemplar
# family (run_823.py commit 14234c9112, run_952.py) and reasonable variants
# (`ensure_repo_root_on_syspath`, `add_repo_root_to_sys_path`). The literal
# `sys.path.insert(...)`/`sys.path.append(...)` shape is matched structurally
# in `_is_syspath_guard_call`, not by this regex.
SYSPATH_GUARD_NAME_RE = re.compile(r"(?i)syspath|sys_path")


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


# `--check-hub-verify-retry` (#1202): a NEW scripts/ file hand-rolling a bare
# `list_repo_files(` / `list_repo_tree(` / `.file_exists(` Hub verify leg
# reintroduces the #920 false-failure class (huggingface_hub's paginate
# retries ONLY 429 on cursor pages — a transient 504 propagates and fails a
# SUCCESSFUL upload's verify leg; gotchas.md "HF recursive tree listing 504s
# are un-retried"). #997 hardened the library path
# (orchestrate/hub.py: verify_repo_paths_uploaded, list_hf_files_under_path,
# list_repo_files_complete, retry_transient); this check is the mechanical
# gate on new call sites.
# Detection: AST — any ast.Attribute with attr in HUB_VERIFY_BARE_TARGETS
# ("list_repo_files", "list_repo_tree", "file_exists"; call OR bare-reference
# form), plus any ast.Name bound by a `from huggingface_hub import <target>
# [as alias]` (asname-aware, so aliasing cannot evade the Name leg).
# Comments/docstrings can never match (no comment nodes; string mentions are
# ast.Constant). Named residuals NOT covered (see the check's docstring):
# repo_info, hf_hub_download, HfFileSystem, raw HTTP, subprocess hf CLI,
# getattr forms, and .sh heredocs (9 files as of 2026-07-09).
# Scope: scripts/**/*.py only (src/ is #997's — hub.py itself legitimately
# spells the bare calls inside its retry wrappers).
# Exempt:
#   * the legacy per-issue offenders predating this check
#     (:data:`HUB_VERIFY_LEGACY_ALLOWLIST`) — frozen 2026-07-09, generated
#     from the check's own live-tree output (AST-confirmed call sites, not
#     grep hits); a NEW file is never added here (the waiver comment below
#     is the path for a genuinely-correct new bare caller);
#     CAVEAT: membership exempts the WHOLE file (the BATCH_JUDGE model) —
#     a future bare call added to a grandfathered file is silently exempt;
#     when migrating a file onto the hub helpers, DROP it from this set;
#   * any call site waived with `# HUB_VERIFY_RETRY_EXEMPT: <reason>`
#     (reason ≥ :data:`HUB_VERIFY_WAIVER_MIN_REASON_CHARS` chars) on the
#     call's first physical line or the immediately preceding non-blank
#     line — same convention as BATCH_JUDGE_CLIENT_EXEMPT.
HUB_VERIFY_LEGACY_ALLOWLIST: frozenset[str] = frozenset(
    {
        # Per-issue experiment scripts predating this check (frozen
        # 2026-07-09; AST-confirmed call sites from the check's own
        # live-tree output, not grep hits).
        "scripts/backfill_artifact_registry.py",
        "scripts/build_canonical_persona_pool.py",
        "scripts/clean_experiment_downloads.py",
        "scripts/dispatch_factor_screen_365.py",
        "scripts/dispatch_neg_geometry_504.py",
        "scripts/i474_check_adapter_hf_presence.py",
        "scripts/i474_phase0_preflight.py",
        "scripts/i477_reval_confirm.py",
        "scripts/i488_phase3_train_sweep.py",
        "scripts/i504_reval_confirm.py",
        "scripts/i528_phase23_train.py",
        "scripts/i556_pull_qbank.py",
        "scripts/i601_run_cell.py",
        "scripts/i650_write_results_sentinel.py",
        "scripts/issue1024_diagnose_parse_failures.py",
        "scripts/issue1073_common.py",
        "scripts/issue1074_aggregate.py",
        "scripts/issue1074_generator_compare.py",
        "scripts/issue1090_fu1.py",
        "scripts/issue1090_fu3_yield_replay.py",
        "scripts/issue1090_run.py",
        "scripts/issue1108_repo_file_audit.py",
        "scripts/issue1112_dispatch.py",
        "scripts/issue1112_geometry.py",
        "scripts/issue530_logit_reval.py",
        "scripts/issue540_jsrb_predictor.py",
        "scripts/issue541_geometry_extract.py",
        "scripts/issue545_sweep.py",
        "scripts/issue545_train_cell.py",
        "scripts/issue594_analyze_context_geometry.py",
        "scripts/issue594_extract_context_vectors.py",
        "scripts/issue604_adapter_svd.py",
        "scripts/issue604_extract_context_vectors.py",
        "scripts/issue617_upload_corpus.py",
        "scripts/issue621_checkpoint_ladder.py",
        "scripts/issue634_extract_behavior_vectors.py",
        "scripts/issue634_joint_geometry.py",
        "scripts/issue651_dispatch.py",
        "scripts/issue651_drain_extracts.py",
        "scripts/issue654_fetch_pinned_battery.py",
        "scripts/issue658_extract_base_store.py",
        "scripts/issue658_fit_predictors.py",
        "scripts/issue661_analysis.py",
        "scripts/issue661_extract_directions.py",
        "scripts/issue661_generate_arm_a.py",
        "scripts/issue664_dispatch.py",
        "scripts/issue666_load_store.py",
        "scripts/issue666_predictor.py",
        "scripts/issue667_alllayer_dispatch.py",
        "scripts/issue667_dispatch.py",
        "scripts/issue667_pertoken_context_dispatch.py",
        "scripts/issue667_pertoken_dispatch.py",
        "scripts/issue685_matched_position_u.py",
        "scripts/issue722_extract_fact_rb.py",
        "scripts/issue722_fit_M.py",
        "scripts/issue722_load_activations.py",
        "scripts/issue722_per_position_vC_skill.py",
        "scripts/issue734_dispatch.py",
        "scripts/issue744_dump_and_stream.py",
        "scripts/issue763_build_probe_pools.py",
        "scripts/issue763_disclosure_flag_audit.py",
        "scripts/issue763_judge_e0.py",
        "scripts/issue763_upload.py",
        "scripts/issue778_v2_prefetch.py",
        "scripts/issue779_capture_answer_summaries.py",
        "scripts/issue779_capture_answer_summaries_pass2.py",
        "scripts/issue779_collect.py",
        "scripts/issue779_extract_rb.py",
        "scripts/issue779_gen_behavior_corpus.py",
        "scripts/issue779_pertoken_lmsys_analysis.py",
        "scripts/issue779_pertoken_vs_mean_variance.py",
        "scripts/issue779_stage_pass2_vm.py",
        "scripts/issue810_common.py",
        "scripts/issue810_extract_positions.py",
        "scripts/issue811_upload_store.py",
        "scripts/issue825_crossmodel_map_transfer.py",
        "scripts/issue833_chain_rho_fixedtext.py",
        "scripts/issue833_chain_rho_nonemit.py",
        "scripts/issue833_extract_onpolicy.py",
        "scripts/issue833_fit_onpolicy.py",
        "scripts/issue841_common.py",
        "scripts/issue841_scaling_common.py",
        "scripts/issue920_extract_summaries.py",
        "scripts/issue920_gen_completions_b.py",
        "scripts/issue920_nulls_figures.py",
        "scripts/issue922_common.py",
        "scripts/issue922_repair_provenance.py",
        "scripts/issue923_capture.py",
        "scripts/issue928_common.py",
        "scripts/issue928_mlp_indiv_control.py",
        "scripts/issue931_fit_cells.py",
        "scripts/issue931_power_curve_multi_seed.py",
        "scripts/issue931_sep_to_chat_matched_control.py",
        "scripts/issue952_bank_build.py",
        "scripts/issue958_common.py",
        "scripts/issue958_long_k1_transfer_lclamp.py",
        "scripts/issue_552_prep_good_corpus.py",
        "scripts/issue_597/dispatch_leakage_dynamics_597.py",
        "scripts/issue_597/titration_svd_597.py",
        "scripts/issue_642/i642_dispatch.py",
        "scripts/issue_653/i653_postpod_bootstrap.py",
        "scripts/measure_cot_entropy.py",
        "scripts/run_issue506_install.py",
        "scripts/sync_models.py",
        # NOT per-issue experiment code — flagged inline so the
        # allowlist rationale stays honest: verify_uploads.py is a
        # workflow-helper script; migrating it onto the hub helpers
        # is a named follow-up.
        "scripts/verify_uploads.py",
    }
)
HUB_VERIFY_WAIVER_RE = re.compile(r"#\s*HUB_VERIFY_RETRY_EXEMPT\s*:\s*(.+?)\s*$")
HUB_VERIFY_WAIVER_MIN_REASON_CHARS = 10


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

# Grandfathered `|| true` swallows on upload/result-persist lines. New
# deliberate best-effort uploads use the inline `# UPLOAD_OR_TRUE_EXEMPT:`
# waiver instead of growing this list (test_live_trees_pass locks it to
# today's tree; mirrors JUDGE_PIN_LEGACY_ALLOWLIST's style). File-level
# granularity (the JUDGE_PIN precedent) — accepted trade-off: a whole-file
# exemption can mask a FUTURE genuine violation added to one of these
# files; all three are historical dispatchers of completed issues.
UPLOAD_OR_TRUE_LEGACY_ALLOWLIST: frozenset[str] = frozenset(
    {
        # --- deliberate best-effort diagnostics side-channels (permanent) ---
        # #654 crash-diagnostics upload on the FAILURE path; `|| true` +
        # in-Python try/except keep a failed HF upload from masking the real
        # failure rc=2 (documented at the call site, lines 154-160):
        "scripts/issue654_dispatch.sh",
        # #632 debug wrapper: best-effort diagnostics upload "regardless of
        # RC"; primary results ride the dispatcher's own fail-loud path:
        "scripts/i632_dispatch_with_log_capture.sh",
        # --- pre-existing, tracked ---
        # `git add eval_results/ figures/ || true` (line 251) before a guarded
        # commit; the fail-loud PRIMARY persist is the HF upload_folder phase
        # (p5) directly above, and the git-push leg degrades to a logged
        # WARNING by design; historical dispatcher of a completed issue:
        "scripts/issue931_dispatch.sh",
    }
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


# Directory NAMES never descended into by _iter_files_pruned: bulk/cache
# dirs that are never workflow surface and can be enormous (a live worktree
# .venv is ~67k files; the repo-root .claude/worktrees tree held 3,300,121
# entries on 2026-07-09 — 145s of rglob enumeration, task #1163). Mirrors
# .gitignore's untracked-bulk set.
_PRUNE_DIR_NAMES = frozenset(
    {
        ".git",
        ".venv",
        "venv",
        "__pycache__",
        "node_modules",
        ".pytest_cache",
        ".arxiv-papers",
        "wandb",
        "outputs",
    }
)


def _iter_files_pruned(base: Path, *, suffixes: frozenset[str]) -> Iterator[Path]:
    """Bounded replacement for ``base.rglob("*")``: yield regular files under
    ``base`` whose suffix is in ``suffixes``, never descending into
    :data:`_PRUNE_DIR_NAMES` dirs nor into a ``worktrees/`` dir directly under
    a ``.claude/`` dir (the pre-enumeration form of the
    :func:`_is_other_worktree_path` exclusion — the post-hoc string filter
    stays in place at call sites as the semantic contract). Neither os.walk
    nor 3.11 pathlib rglob follows directory symlinks (probe-verified on this
    VM's 3.11.15), so the swap introduces no symlink-traversal divergence
    (task #1163)."""
    for dirpath, dirnames, filenames in os.walk(base):
        parent_name = Path(dirpath).name
        dirnames[:] = [
            d
            for d in dirnames
            if d not in _PRUNE_DIR_NAMES and not (d == "worktrees" and parent_name == ".claude")
        ]
        for fn in filenames:
            p = Path(dirpath, fn)
            if p.suffix in suffixes and p.is_file():
                yield p


# (str(path), len(text), hash(text)) -> parsed module (or None when
# unparseable). The no-flags run's parse-bearing AST checks each re-parsed
# the same scripts/ + src/ corpus (~22s per pass, measured 1,336 files /
# 31.4 MB); memoizing collapses ~4 redundant passes (task #1163). The
# CONTENT-based key (text length + hash) invalidates on ANY rewrite — incl.
# the unit-test tmp_path rewrite-between-calls pattern — with no stat call.
# Peak RSS retaining all trees: ~1.0-1.1 GB (measured twice) — transient,
# freed at process exit. NOTE: cached trees are SHARED across checks —
# consumers must never mutate the returned nodes.
_AST_CACHE: dict[tuple[str, int, int], ast.Module | None] = {}


def _cached_parse(path: Path, text: str) -> ast.Module | None:
    """Memoized ``ast.parse`` of ``text`` (the caller's just-read source of
    ``path``). Returns None when unparseable (SyntaxError; the ValueError in
    the except tuple is inert defense-in-depth — NUL bytes raise SyntaxError
    on this VM's 3.11.15) — the CALLER decides what None means at its site
    (silent skip, stderr note, ...). READING stays at the call site: every
    routed site keeps its own ``read_text`` and its current exception
    posture, so read-failure behavior is unchanged (task #1163)."""
    key = (str(path), len(text), hash(text))
    if key in _AST_CACHE:
        return _AST_CACHE[key]
    try:
        tree: ast.Module | None = ast.parse(text, filename=str(path))
    except (SyntaxError, ValueError):
        tree = None
    _AST_CACHE[key] = tree
    return tree


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
    skills (``/daily``, ``/pm``, etc.) never run under
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


def _head_is_main(repo_root: Path) -> bool:
    """True iff the checkout containing ``repo_root`` has branch ``main``
    checked out — the STRICT reference-lint regime. Fail-safe: any git
    failure (non-git tree, e.g. the Step 10d /tmp landing-tree gate copy)
    returns True, keeping hard-FAIL semantics (#1672)."""
    try:
        r = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except Exception:
        return True
    return r.returncode != 0 or r.stdout.strip() == "main"


@functools.cache
def _tracked_on_main_ref(relpath: str, repo_root_s: str) -> bool:
    """True iff ``relpath`` exists as a tracked path at local ``main`` or
    ``origin/main`` in the repo containing ``repo_root_s`` (worktrees share
    refs with the common dir). Fail-safe: any git failure returns False —
    the caller keeps the hard FAIL (#1622/#1672)."""
    for ref in ("main", "origin/main"):
        try:
            r = subprocess.run(
                ["git", "-C", repo_root_s, "cat-file", "-e", f"{ref}:{relpath}"],
                capture_output=True,
                timeout=10,
            )
        except Exception:
            return False
        if r.returncode == 0:
            return True
    return False


def check_script_references(
    *,
    roots: list[Path] | None = None,
    scripts_dir: Path | None = None,
    main_probe: Callable[[str], bool] | None = None,
    warn_sink: list[str] | None = None,
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

    Staleness degrade (#1622/#1672): in production scope only (``scripts_dir``
    is None) on a NON-``main`` checkout, a reference whose target is missing
    locally but tracked at ``main``/``origin/main`` (lazy ``git cat-file -e``
    probe, only on a miss) is downgraded to a non-blocking ``WARN:`` — the
    Step 5a spec-freshness sync never syncs ``scripts/`` helpers, so a freshly
    synced spec may reference a main-new helper a stale worktree lacks.
    Fail-safe in every direction: a non-git tree (the Step 10d landing-tree
    gate), a failed git probe, or ``HEAD == main`` keeps the hard FAIL.
    ``main_probe`` / ``warn_sink`` are unit-test override hooks.
    """
    errors: list[str] = []

    def _warn(msg: str) -> None:
        if warn_sink is not None:
            warn_sink.append(msg)
        else:
            sys.stderr.write(f"WARN: {msg}\n")

    scripts_root = scripts_dir if scripts_dir is not None else _REPO_ROOT / "scripts"
    # Staleness-aware degrade (#1622/#1672): PRODUCTION scope only (fixture-
    # scoped test calls stay strict unless a probe is injected), and only on
    # a non-main checkout — main-checkout semantics are byte-identical.
    if main_probe is None and scripts_dir is None and not _head_is_main(_REPO_ROOT):

        def _default_script_probe(name: str) -> bool:
            return _tracked_on_main_ref(f"scripts/{name}", str(_REPO_ROOT))

        main_probe = _default_script_probe
    for path in _resolve_ask_target_files(roots):
        for lineno, line in enumerate(path.read_text().splitlines(), start=1):
            if HISTORICAL_REF_OPT_OUT in line:
                continue
            for match in SCRIPT_REF_RE.finditer(line):
                script_name = match.group(1)
                if not (scripts_root / script_name).exists():
                    if main_probe is not None and main_probe(script_name):
                        _warn(
                            f"{path}:{lineno}: references 'scripts/{script_name}' — "
                            f"missing under {scripts_root}/ but present at "
                            f"main/origin/main: stale worktree tree, not this "
                            f"commit's breakage (#1622/#1672). Not blocking; the "
                            f"main-tree lint and the Step 10d landing-tree gate "
                            f"re-check strictly."
                        )
                        continue
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


def _skill_ref_resolves(
    ref: str,
    live: set[str],
    allow: frozenset[str],
    fs_roots: frozenset[str] = SKILL_REF_FS_ROOTS,
) -> bool:
    """A backticked ``/<ref>`` resolves iff it names a live skill dir, an
    allowlisted exact token, a bare filesystem root (a backticked PATH like
    ``/tmp`` — see :data:`SKILL_REF_FS_ROOTS`), or (when namespaced
    ``<plugin>:<skill>``) a token whose ``<plugin>:`` prefix is allowlisted."""
    if ref in live:  # live project skill dir
        return True
    if ref in allow:  # allowlisted exact token
        return True
    if ref in fs_roots:  # bare backticked fs path, not a slash-command (#1445)
        return True
    if ":" in ref:  # plugin-namespaced: prefix match
        return (ref.split(":", 1)[0] + ":") in allow
    return False


def check_skill_references(
    *,
    roots: list[Path] | None = None,
    skills_dir: Path | None = None,
    allowlist: frozenset[str] | None = None,
    fs_roots: frozenset[str] | None = None,
    main_probe: Callable[[str], bool] | None = None,
    warn_sink: list[str] | None = None,
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

    Bare filesystem roots (``/tmp``, ``/workspace``;
    :data:`SKILL_REF_FS_ROOTS`) are carved out as paths, never
    slash-commands (#1445). An fs-root member that names a LIVE skill dir
    is itself reported as a lint error — the carve-out would silently
    disable rot detection for that skill (remedy: drop the colliding
    member from ``SKILL_REF_FS_ROOTS``).

    ``roots`` / ``skills_dir`` / ``allowlist`` / ``fs_roots`` are unit-test
    override hooks; production callers pass None.

    Staleness degrade (#1622/#1672), same pattern as
    :func:`check_script_references`: in production scope only (``skills_dir``
    is None) on a NON-``main`` checkout, a plain single-segment ``/skill``
    token unresolved locally but whose ``.claude/skills/<ref>/SKILL.md`` is
    tracked at ``main``/``origin/main`` is downgraded to a non-blocking
    ``WARN:`` (a ``:``-namespaced ref resolves via the allowlist, which rides
    the synced lint copy — the probe never fires for it). Fail-safe: non-git
    tree, failed probe, or ``HEAD == main`` keeps the hard FAIL;
    ``main_probe`` / ``warn_sink`` are unit-test override hooks.
    """
    errors: list[str] = []

    def _warn(msg: str) -> None:
        if warn_sink is not None:
            warn_sink.append(msg)
        else:
            sys.stderr.write(f"WARN: {msg}\n")

    sk_dir = skills_dir if skills_dir is not None else _REPO_ROOT / ".claude" / "skills"
    # Staleness-aware degrade (#1622/#1672): PRODUCTION scope only, non-main
    # checkout only — mirrors check_script_references.
    if main_probe is None and skills_dir is None and not _head_is_main(_REPO_ROOT):

        def _default_skill_probe(ref_name: str) -> bool:
            return _tracked_on_main_ref(f".claude/skills/{ref_name}/SKILL.md", str(_REPO_ROOT))

        main_probe = _default_skill_probe
    live = _live_skill_names(sk_dir)
    allow = allowlist if allowlist is not None else SKILL_REF_ALLOWLIST
    fsr = fs_roots if fs_roots is not None else SKILL_REF_FS_ROOTS
    collisions = sorted(fsr & live)
    if collisions:
        errors.append(
            f"SKILL_REF_FS_ROOTS collides with live skill dir(s) {collisions}: the "
            f"fs-root carve-out would silently disable skill-reference rot detection "
            f"for them (#1445). Drop the colliding member(s) from SKILL_REF_FS_ROOTS "
            f"in scripts/workflow_lint.py."
        )
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
                if _skill_ref_resolves(ref, live, allow, fsr):
                    continue
                if main_probe is not None and ":" not in ref and main_probe(ref):
                    _warn(
                        f"{path}:{lineno}: unresolved skill reference '/{ref}' — "
                        f"not resolvable in this tree but present at "
                        f"main/origin/main: stale worktree tree, not this "
                        f"commit's breakage (#1622/#1672). Not blocking; the "
                        f"main-tree lint and the Step 10d landing-tree gate "
                        f"re-check strictly."
                    )
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


def _iter_sh_logicals_with_heredocs(lines: list[str]):
    """Yield ``(first_idx, last_idx, logical, heredoc_bodies)`` per logical
    shell line; ``heredoc_bodies`` is a list of ``(body_start, body_end)``
    0-based physical-line bounds (end EXCLUSIVE of the terminator line),
    one per heredoc opener on the logical line, in opener order.

    Shared cursor logic extracted from the heredoc-dotenv scanner (#1036):
    backslash-continued physical lines are merged into one logical command
    line before opener detection (the #612 shape continues the opener line
    with ``\\`` + ``|| fail ...``; the body starts after the last physical
    line of the logical command). ALL heredoc bodies are CONSUMED — body
    content is never re-yielded as logical shell lines, so it can never be
    misparsed as new openers. The terminator match is lenient
    (stripped-line equality) so ``<<-`` indented terminators work; an
    unterminated heredoc consumes through to EOF."""
    n = len(lines)
    i = 0
    while i < n:
        last = i
        logical = lines[i]
        while logical.rstrip().endswith("\\") and last + 1 < n:
            last += 1
            logical = logical.rstrip()[:-1] + " " + lines[last]
        openers = list(HEREDOC_OPENER_RE.finditer(logical))
        bodies: list[tuple[int, int]] = []
        body_cursor = last + 1
        for opener in openers:
            delim = opener.group(2)
            body_start = body_cursor
            body_end = body_start
            while body_end < n and lines[body_end].strip() != delim:
                body_end += 1
            bodies.append((body_start, body_end))
            body_cursor = body_end + 1
        yield i, last, logical, bodies
        i = body_cursor if openers else last + 1


def _scan_shell_file_for_heredoc_dotenv(path: Path) -> list[str]:
    """Walk one shell script, tracking heredoc bodies, and return the
    dotenv errors found in bodies that feed a python interpreter's stdin.

    The cursor logic (backslash merge, opener detection, delimiter-bounded
    body consumption) lives in the shared :func:`_iter_sh_logicals_with_heredocs`
    generator; only the python-stdin-fed classification + dotenv body scan
    stay here."""
    lines = path.read_text(encoding="utf-8").splitlines()
    errors: list[str] = []
    for _first, _last, logical, bodies in _iter_sh_logicals_with_heredocs(lines):
        if not bodies:
            continue
        openers = list(HEREDOC_OPENER_RE.finditer(logical))
        prefix = logical[: openers[0].start()]
        python_fed = bool(HEREDOC_PY_STDIN_DASH_RE.search(prefix)) or bool(
            HEREDOC_PY_STDIN_BARE_RE.search(prefix)
        )
        if not python_fed:
            continue
        for body_start, body_end in bodies:
            errors.extend(_heredoc_body_dotenv_errors(path, lines, body_start, body_end))
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


def _sh_waiver_present(
    lines: list[str],
    first_idx: int,
    last_idx: int,
    *,
    waiver_re: re.Pattern[str],
    min_reason_chars: int,
) -> bool:
    """Return True iff a reason-bearing inline waiver comment (matched by
    ``waiver_re``, group(1) = reason, reason ≥ ``min_reason_chars`` chars
    after strip) covers the logical command spanning
    ``lines[first_idx:last_idx + 1]``. Accepts the waiver on any physical
    line of the logical command (trailing comment on a single-line
    command) or on the immediately preceding non-blank line (the only
    valid placement for a backslash-continued command — a trailing ``#``
    comment would break the continuation). Runs on the RAW lines: the
    waiver IS a comment, so it must be read before comment stripping.
    Generalized from the CVD_PIN_EXEMPT helper (#1036) so sibling checks
    (UPLOAD_OR_TRUE_EXEMPT) share the placement semantics."""
    for idx in range(first_idx, last_idx + 1):
        match = waiver_re.search(lines[idx])
        if match and len(match.group(1).strip()) >= min_reason_chars:
            return True
    back = first_idx - 1
    while back >= 0 and lines[back].strip() == "":
        back -= 1
    if back >= 0:
        match = waiver_re.search(lines[back])
        if match and len(match.group(1).strip()) >= min_reason_chars:
            return True
    return False


def _cvd_pin_waiver_present(lines: list[str], first_idx: int, last_idx: int) -> bool:
    """Return True iff a ``# CVD_PIN_EXEMPT: <reason>`` waiver (reason ≥
    :data:`CVD_PIN_WAIVER_MIN_REASON_CHARS` chars) covers the logical
    command spanning ``lines[first_idx:last_idx + 1]`` — see
    :func:`_sh_waiver_present` for the placement semantics."""
    return _sh_waiver_present(
        lines,
        first_idx,
        last_idx,
        waiver_re=CVD_PIN_WAIVER_RE,
        min_reason_chars=CVD_PIN_WAIVER_MIN_REASON_CHARS,
    )


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


def check_piped_git_push(*, scripts_dir: Path | None = None) -> list[str]:
    """Walk every ``*.sh`` under ``scripts/`` and FAIL on any ``git push`` /
    ``git merge`` / ``git commit`` / ``gh pr merge|create`` piped into a
    filter on its own
    pipeline segment (``git push origin main 2>&1 | tail -20``).

    Rationale: bash makes a pipeline's exit status the LAST stage's, so the
    pipe masks the producer's non-zero exit code — a rejected push reads as
    success and the session proceeds believing the merge landed (#957's
    Step 10d push was masked 2026-07-04; 4 sessions hit the class on
    2026-07-02); a piped ``git commit`` is additionally SIGPIPE-killed
    mid-pre-commit-hook when the reader exits early (#1584, #1591). The
    prose rule (CLAUDE.md § Concurrent repo-root
    committers) says run it bare and check the exit code, or use
    ``set -o pipefail`` when a pipe is unavoidable — so once a non-comment
    line contains ``pipefail``, flagging is disabled for the REST of the
    file (lines BEFORE it still flag — fires-direction, NOT a whole-file
    pre-scan; ``set +o pipefail`` re-disable is ignored, failing toward
    false-negative, the documented safe direction). ``|&`` is normalized to
    ``|`` on the logical line before matching; a match whose span contains
    ``--dry-run`` is skipped (a dry run lands nothing); only ``#``-comment
    lines are otherwise skipped. See the ``PIPED_GIT_PUSH_RE`` block above
    for the full flagged/not-flagged matrix. The
    ``.claude/hooks/guard_piped_git_push.sh`` PreToolUse hook is the
    runtime sibling covering ad-hoc inline commands (#1048; the
    dual-engine split of ``check_pipe_python`` / #753).

    ``scripts_dir`` is an override hook for unit tests; production callers
    pass None and the function walks the canonical ``<repo_root>/scripts``
    tree. Bundled into the no-flags default run (same policy as
    ``check_pipe_python``).
    """
    root = scripts_dir if scripts_dir is not None else _REPO_ROOT / "scripts"
    if not root.exists():
        return []
    errors: list[str] = []
    for sh in sorted(root.rglob("*.sh")):
        if not sh.is_file():
            continue
        lines = sh.read_text(encoding="utf-8").splitlines()
        pipefail_seen = False
        for first, _last, logical in _iter_logical_shell_lines(lines):
            stripped = logical.strip()
            if stripped.startswith("#"):
                continue
            if "pipefail" in logical:
                # The rule's own sanctioned escape: every pipe at-or-after
                # this line propagates the failure. Deliberately NOT a
                # whole-file pre-scan — an offense BEFORE the first
                # pipefail line still flags (plan #1048 MF3).
                pipefail_seen = True
                continue
            if pipefail_seen:
                continue
            match = PIPED_GIT_PUSH_RE.search(logical.replace("|&", "|"))
            if not match:
                continue
            if "--dry-run" in match.group(0):
                continue
            errors.append(
                f"{sh}:{first + 1}: `git push`/merge/commit-class command "
                f"piped into a filter — the pipe masks the non-zero exit "
                f"code, so a rejected push reads as success (#957, 4 "
                f"sessions 2026-07-02), and a piped `git commit` is "
                f"SIGPIPE-killed mid-pre-commit-hook (#1584). Run it bare "
                f"and check the exit code, or add `set -o pipefail`. See "
                f"CLAUDE.md § Concurrent repo-root committers (#1048, "
                f"#1591)."
            )
    return errors


def check_push_failure_swallow(*, scripts_dir: Path | None = None) -> list[str]:
    """Walk every ``*.sh`` under ``scripts/`` and FAIL on any ``git push``
    whose failure is swallowed on the same logical line by ``|| echo`` /
    ``|| true`` / ``|| :`` / ``|| printf``.

    Rationale (#1205, incidents #825 r6/r7/r8): the swallow lets a
    dispatch step declare success while the result commit never landed —
    on GCE the self-DELETEing instance then holds the ONLY copy of the
    commit. The workload-side ``||`` sibling of
    :func:`check_piped_git_push`'s pipe-masking class; unlike that
    sibling there is NO ``pipefail`` escape (``pipefail`` never applies
    to ``||`` disjunctions). Verify the push instead:
    ``git -C <root> rev-list --count origin/<branch>..HEAD`` prints ``0``
    (retry once; still non-zero -> exit non-zero) — the contract in
    ``.claude/rules/pod-side-reporting.md`` § Result-push verification
    contract. Safe shapes never match (see the ``PUSH_FAILURE_SWALLOW_RE``
    block); waive with ``# PUSH_SWALLOW_EXEMPT: <reason>``; legacy
    offenders are frozen in :data:`PUSH_SWALLOW_LEGACY_ALLOWLIST`
    (path-keyed against the repo-root-relative ``scripts/<name>`` path,
    so tmp-dir unit fixtures resolve through the same key space).

    ``scripts_dir`` is an override hook for unit tests; production
    callers pass None and the function walks the canonical
    ``<repo_root>/scripts`` tree. Bundled into the no-flags default run
    (same policy as ``check_piped_git_push``).
    """
    root = scripts_dir if scripts_dir is not None else _REPO_ROOT / "scripts"
    if not root.exists():
        return []
    errors: list[str] = []
    for sh in sorted(root.rglob("*.sh")):
        if not sh.is_file():
            continue
        rel_key = f"scripts/{sh.relative_to(root).as_posix()}"
        if rel_key in PUSH_SWALLOW_LEGACY_ALLOWLIST:
            continue
        lines = sh.read_text(encoding="utf-8").splitlines()
        for first, last, logical in _iter_logical_shell_lines(lines):
            stripped = logical.strip()
            if stripped.startswith("#"):
                continue
            if not PUSH_FAILURE_SWALLOW_RE.search(logical):
                continue
            if _sh_waiver_present(
                lines,
                first,
                last,
                waiver_re=PUSH_SWALLOW_WAIVER_RE,
                min_reason_chars=PUSH_SWALLOW_WAIVER_MIN_REASON_CHARS,
            ):
                continue
            errors.append(
                f"{sh}:{first + 1}: `git push` failure swallowed by "
                f"`|| echo/true/:/printf` — the step declares success while "
                f"the result commit never landed; on GCE the self-DELETEing "
                f"instance holds the only copy (#825 r6-r8). Verify the push "
                f"instead (`git rev-list --count origin/<branch>..HEAD` == 0, "
                f"retry once, then exit non-zero) per "
                f".claude/rules/pod-side-reporting.md § Result-push "
                f"verification contract, or waive a genuinely-safe shape with "
                f"`# PUSH_SWALLOW_EXEMPT: <reason>` (#1205)."
            )
    return errors


# `--check-sh-function-rc-capture` (#1516; incident #1426): bash disables
# errexit throughout a function BODY when the call sits in an `||` context,
# so `func || rc=$?` / `func || true` collapses mid-function failures to the
# last command's rc (usually 0) — #1426's `run_seed "$s" || rc=$?` let a
# Gate-1 terminal failure + a manifest SystemExit proceed to `[phase=done]`.
# ShellCheck SC2310 is the broader external analogue (also `if func`,
# `while func`, `! func`, `func && x`); v1 deliberately matches the filed
# incident class only (`|| rc=$?` / `|| true` on a same-file function).
# Parse limitations degrade toward false NEGATIVES (same direction as
# check_upload_or_true), with TWO named FP-side exceptions, both measured
# harmless on the live tree (0 instances; the repo-tree-is-clean test + the
# waiver contain them):
# (a) pass-1 over-collection — the brace-optional def regex can collect bare
#     zero-arg call lines inside heredoc python bodies (errs toward flagging);
# (b) a case-arm PATTERN that equals a collected function name
#     (`fname) cmd || true ;;`) matches the invocation regex at ^ and would
#     flag.
# The naive double-quote masking (no \" escape handling) can also mis-mask
# exotic lines in either direction — see _rc_capture_mask_quotes.
RC_CAPTURE_FUNC_DEF_RE = re.compile(
    r"^\s*(?:function\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*\(\s*\)\s*\{?"  # name() {  /  name ()
    r"|^\s*function\s+([A-Za-z_][A-Za-z0-9_]*)\s*\{"  # function name {   (paren-less)
)
RC_CAPTURE_SUPPRESS_RE = re.compile(r"\|\|\s*(?:[A-Za-z_][A-Za-z0-9_]*=\$\?|true\b|:(?=[\s;]|$))")
# Three suppressor shapes (capture-SPELLING variants covered, not just the
# filed `rc=$?`):
#   (1) `|| <var>=$?` — ANY simple variable name (rc / status / ret / code /
#       ...), not the literal `rc` only; the danger is the
#       capture-and-continue shape, not the name.
#   (2) `|| true\b` — the \b sits INSIDE the true branch only: `?` is a
#       non-word char, so a trailing \b after the alternation would NEVER
#       match `=$?` (a dead branch in the draft regex, corrected at plan
#       fact-check; the corrected form matches all shapes and still returns
#       0 live findings).
#   (3) `|| :` — the colon builtin, an exact synonym of `|| true` (the
#       lookahead bounds it to whitespace / `;` / EOL so `:=`-style text
#       cannot match).
# Deliberately NOT matched (documented residuals; see the check docstring):
# `|| { rc=$?; ... }` brace-group capture (brace-group handlers are a
# pervasive live fail-loud idiom — 20+ sites incl. a same-file-function call
# at run_program_orchestrator.sh:86 — and covering them would threaten the
# 0-findings bundling precondition), and quoted `|| rc="$?"` (quote masking
# blanks the `"$?"` span BEFORE the suppressor scan, a masking-induced FN;
# 0 live hits).
RC_CAPTURE_WAIVER_RE = re.compile(r"#\s*RC_CAPTURE_EXEMPT\s*:\s*(.+?)\s*$")
RC_CAPTURE_WAIVER_MIN_REASON_CHARS = 10
# Initial errexit state per file: ON iff the shebang carries a short-option
# cluster containing `e` (`#!/bin/bash -e` / `-eu`); otherwise OFF.
RC_CAPTURE_SHEBANG_E_RE = re.compile(r"^#!\S+.*\s-[a-zA-Z]*e[a-zA-Z]*\b")


def _rc_capture_set_e_transition(stripped: str) -> bool | None:
    """Token-scan a logical shell line for a ``set -e`` / ``set +e``
    errexit transition. Returns True (errexit ON), False (OFF), or None
    (no transition — the line is not a ``set`` command, or carries no
    e-bearing token). A short-option cluster containing ``e`` (``-e``,
    ``-euo``, ``-eux``) turns errexit ON; the ``+`` cluster form turns it
    OFF; the long forms ``-o errexit`` / ``+o errexit`` are covered for
    completeness (0 live uses). The LAST matching token on the line wins.
    This is LINE-ORDER state tracking (the ``check_piped_git_push``
    ``pipefail_seen`` precedent), not file-level presence — 7 live scripts
    toggle ``set +e`` mid-file, and under ``set +e`` the ``|| rc=$?``
    pattern is not the footgun. Known coarseness: ``set -e`` inside a
    function body or subshell is treated as a file-scope transition (no
    scope tracking); it errs toward the state the author most recently
    declared."""
    tokens = [tok.rstrip(";") for tok in stripped.split()]
    if not tokens or tokens[0] != "set":
        return None
    transition: bool | None = None
    for i, tok in enumerate(tokens[1:], start=1):
        nxt = tokens[i + 1] if i + 1 < len(tokens) else ""
        if tok == "-o" and nxt == "errexit":
            transition = True
        elif tok == "+o" and nxt == "errexit":
            transition = False
        elif re.fullmatch(r"-[a-zA-Z]*e[a-zA-Z]*", tok):
            transition = True
        elif re.fullmatch(r"\+[a-zA-Z]*e[a-zA-Z]*", tok):
            transition = False
    return transition


def _rc_capture_mask_quotes(s: str) -> str:
    """Replace each single-quoted then double-quoted span with same-length
    spaces, so quoted text (a remote command carrying ``|| true``, e.g. the
    ``bootstrap_pod.sh`` ``ssh_cmd 'cd ... || true'`` shape) can never
    satisfy the invocation or suppressor regexes, while character positions
    stay aligned for the ``;``-segment guard. Single-quoted spans are
    masked first (bash forbids escapes inside them, so that pass is exact);
    the double-quote pass is naive about ``\\"`` escapes — a known
    imperfection that can only mis-mask exotic lines, and any mis-mask
    lands on the fail-toward-false-negative side in practice because the
    flag additionally requires a collected same-file function name at
    command position."""
    s = re.sub(r"'[^']*'", lambda m: " " * len(m.group(0)), s)
    return re.sub(r'"[^"]*"', lambda m: " " * len(m.group(0)), s)


def check_sh_function_rc_capture(*, scripts_dir: Path | None = None) -> list[str]:
    """Walk every ``*.sh`` under ``scripts/`` and FAIL on any SAME-FILE
    bash function invoked via ``func || rc=$?`` / ``|| true`` / ``|| :``
    while the script runs under ``set -e``.

    Rationale (#1516; incident #1426): bash disables errexit throughout a
    function's BODY whenever the call appears in an ``||`` context, so a
    dispatcher written in the house ``set -euo pipefail`` style loses
    every implicit guard inside the function — a mid-function
    ``SystemExit``, gate failure, or fit crash falls through and ``rc``
    captures only the LAST command's exit code (usually 0). On #1426 this
    let partial uploads and the ``[phase=done]`` success sentinel proceed
    past a Gate-1 terminal failure. ShellCheck SC2310 is the broader
    external analogue (optional/off-by-default even there); v1
    deliberately matches the filed incident class only.

    Detection, per logical line (backslash continuations merged, heredoc
    bodies consumed via :func:`_iter_sh_logicals_with_heredocs`, trailing
    comments stripped quote-aware via :func:`_strip_sh_trailing_comment`,
    quotes masked via :func:`_rc_capture_mask_quotes`):

    1. Pass 1 collects the file's own function names
       (:data:`RC_CAPTURE_FUNC_DEF_RE`); no functions -> file skipped.
    2. A collected name must sit at COMMAND POSITION (line start, or
       after ``;`` ``&`` ``|`` ``(`` ``{`` or a ``then``/``do``/``else``
       keyword) — this keeps single external-command captures
       (``uv run python ... || rc=$?``, all current live captures)
       unflagged: their first word is never a same-file function.
    3. The suppressor (:data:`RC_CAPTURE_SUPPRESS_RE`) must follow the
       invocation with no ``;`` between them (segment guard — a
       suppressor on a LATER ``;``-segment belongs to that segment's own
       command; ``&&`` chains deliberately stay in scope, since
       ``cd x && func || rc=$?`` puts ``func`` in the ``||`` context).
    4. Errexit state is tracked line-order
       (:func:`_rc_capture_set_e_transition`; initial state from the
       shebang, :data:`RC_CAPTURE_SHEBANG_E_RE`) — a hit inside a
       ``set +e`` region does not flag (the function body never had
       errexit protection there, so the author owns explicit error
       handling; failing toward false-negative, the safe direction).
    5. Definition lines are skipped (a definition is not an invocation —
       kills the one-liner-def ``... || true; }`` shape; accepted FN: a
       one-liner def whose BODY invokes another collected function with
       ``|| true`` is skipped with it); ``#``-comment lines are skipped;
       one error per logical line, first hit wins.
    6. ``# RC_CAPTURE_EXEMPT: <reason>`` (reason >= 10 chars, same
       logical line or immediately preceding non-blank line) waives.

    Out-of-scope contexts (documented residuals — all the same bash
    footgun; ShellCheck SC2310's broader scope is the named future
    extension if the class recurs through one of them): ``if func`` /
    ``while func`` / ``until func`` / ``! func`` / ``func && x``
    suppressing contexts, ``var=$(func) || rc=$?`` assignment-hidden
    substitutions, ``env VAR=1 func || true`` and bare assignment-prefix
    forms (fname not at command position), cross-file sourced functions
    (the collector is same-file by design, per the #1516 Goal),
    ``func || { rc=$?; ... }`` brace-group capture, quoted
    ``func || rc="$?"``, fail-loud handlers ``func || exit 1``, and
    case-arm invocations ``pattern) func || true ;;``. ``.claude/hooks/``
    is out of scope (0 of its 6 files use ``set -e`` — guards are
    deliberately fail-open, so the class cannot fire there today).

    ``scripts_dir`` is an override hook for unit tests; production
    callers pass None and the function walks the canonical
    ``<repo_root>/scripts`` tree. Bundled into the no-flags default run
    (same policy as ``check_push_failure_swallow``).
    """
    root = scripts_dir if scripts_dir is not None else _REPO_ROOT / "scripts"
    if not root.exists():
        return []
    errors: list[str] = []
    for sh in sorted(root.rglob("*.sh")):
        if not sh.is_file():
            continue
        lines = sh.read_text(encoding="utf-8").splitlines()
        funcs: set[str] = set()
        for line in lines:
            def_match = RC_CAPTURE_FUNC_DEF_RE.match(line)
            if def_match:
                funcs.add(def_match.group(1) or def_match.group(2))
        if not funcs:
            continue
        inv_re = re.compile(
            r"(?:^|[;&|({]|\b(?:then|do|else)\s)\s*("
            + "|".join(re.escape(f) for f in sorted(funcs))
            + r")\b"
        )
        errexit_on = bool(lines) and bool(RC_CAPTURE_SHEBANG_E_RE.match(lines[0]))
        for first, last, logical, _bodies in _iter_sh_logicals_with_heredocs(lines):
            stripped = logical.strip()
            if stripped.startswith("#"):
                continue
            # Comment-strip BEFORE the transition scan: `set -uo pipefail  # NOT set -e`
            # must not flip errexit ON from the comment's `-e` token (live shapes:
            # i632_dispatch_with_log_capture.sh:12, issue683_dispatch.sh:30).
            transition = _rc_capture_set_e_transition(_strip_sh_trailing_comment(stripped))
            if transition is not None:
                errexit_on = transition
                continue
            if not errexit_on:
                continue
            if RC_CAPTURE_FUNC_DEF_RE.match(logical):
                continue
            masked = _rc_capture_mask_quotes(_strip_sh_trailing_comment(logical))
            for inv in inv_re.finditer(masked):
                suppress = RC_CAPTURE_SUPPRESS_RE.search(masked, inv.end())
                if suppress is None or ";" in masked[inv.end() : suppress.start()]:
                    continue
                if not _sh_waiver_present(
                    lines,
                    first,
                    last,
                    waiver_re=RC_CAPTURE_WAIVER_RE,
                    min_reason_chars=RC_CAPTURE_WAIVER_MIN_REASON_CHARS,
                ):
                    fname = inv.group(1)
                    errors.append(
                        f"{sh}:{first + 1}: same-file bash function `{fname}` invoked "
                        f"via `|| <var>=$?`/`|| true`/`|| :` under set -e — bash "
                        f"disables errexit inside the function BODY in an `||` "
                        f"context, so mid-function failures collapse to the last "
                        f"command's rc (#1426: a Gate-1 terminal failure + a manifest "
                        f"SystemExit read as rc=0 and `[phase=done]` fired). PRIMARY "
                        f"remedies: harden the body's failure-prone steps with "
                        f"explicit `|| exit`/`|| return $?`, or extract the body to a "
                        f"child script (`bash x.sh || rc=$?` — a child process keeps "
                        f"its own set -e; this is why single-external-command "
                        f"captures are safe). NOTE: `set +e; {fname}; rc=$?; set -e` "
                        f"bracketing alone does NOT restore body errexit — it has the "
                        f"identical collapse semantics and must be paired with body "
                        f"hardening. A genuinely-safe shape may be waived with "
                        f"`# RC_CAPTURE_EXEMPT: <reason>` (#1516)."
                    )
                break  # one error per logical line, first hit wins
    return errors


# `--check-grep-qv` (#928 -> #1125): an rc-consumed quiet+invert grep trigger
# is implementation-divergent — GNU grep exits 0 iff a line is SELECTED (with
# -v, selected = non-matching), while ugrep 7.5.0 returns rc=1 in the same
# case (its -q short-circuits on MATCH FOUND, not line selected) — so the
# combination silently fails OPEN when shell `grep` resolves to ugrep. The
# regex matches a bare or path-prefixed `grep`/`ugrep` command word (the
# lookbehind rejects word-char / `.` / `-` prefixes so `pgrep`, `foo-grep`,
# `x.grep` never match; a preceding `/` IS allowed so path pins are visible
# to the caller, which exempts pinned `grep` but still flags pinned `ugrep`)
# and captures the CONTIGUOUS option-token run that follows — so a
# pipeline-split `grep -v x f | grep -q y f2` yields two matches whose flag
# sets are evaluated independently and never combine.
GREP_QV_CMD_RE = re.compile(r"(?<![\w.\-])(u?grep)\b((?:\s+--?[\w=-]+)*)")

_GREP_QV_GIT_PREFIX_RE = re.compile(r"(?<![\w.\-])git\s+$")


def _grep_qv_flag_sets(opt_run: str) -> tuple[bool, bool]:
    """Return ``(has_q, has_v)`` for the contiguous option-token run
    following a grep command word — a combined short token (``-qvE``),
    separated tokens (``-q ... -vE``, either order), and the long forms
    (``--quiet``/``--silent`` + ``--invert-match``) all count."""
    short: set[str] = set()
    long_flags: set[str] = set()
    for tok in opt_run.split():
        if tok.startswith("--"):
            long_flags.add(tok[2:].split("=", 1)[0])
        elif tok.startswith("-") and len(tok) > 1:
            short.update(tok[1:])
    has_q = "q" in short or bool(long_flags & {"quiet", "silent"})
    has_v = "v" in short or "invert-match" in long_flags
    return has_q, has_v


def _grep_qv_scan(path: Path, lines: list[str], base_idx: int, errors: list[str]) -> None:
    """Scan ``lines`` (physical lines whose first line sits at 0-based file
    index ``base_idx``) as logical shell lines and append one error per
    flagged unpinned q+v grep invocation. ``#``-comment lines are skipped;
    backslash continuations are merged (the live #928 trigger was
    backslash-continued with the flags on the first physical line)."""
    for first, _last, logical in _iter_logical_shell_lines(lines):
        if logical.strip().startswith("#"):
            continue
        for match in GREP_QV_CMD_RE.finditer(logical):
            cmd = match.group(1)
            path_pinned = match.start() > 0 and logical[match.start() - 1] == "/"
            if path_pinned and cmd == "grep":
                # /usr/bin/grep pin — the sanctioned GNU pin (a pinned
                # ugrep is NOT sanctioned: its rc diverges wherever it is).
                continue
            if _GREP_QV_GIT_PREFIX_RE.search(logical[: match.start()]):
                # `git grep` — git's own engine, not PATH-shadowable.
                continue
            has_q, has_v = _grep_qv_flag_sets(match.group(2))
            if has_q and has_v:
                errors.append(
                    f"{path}:{base_idx + first + 1}: `{cmd}` combines -q and -v "
                    f"(quiet + invert-match) with the exit status as the signal. "
                    f"ugrep 7.5.0 returns rc=1 where GNU returns 0 when "
                    f"non-matching lines are selected, so an rc-consumed q+v "
                    f"trigger fails OPEN under a PATH-shadowed grep (#928: the "
                    f"Step 10d pre-push lint gate silently disarmed; fixed in "
                    f"#1125). Consume OUTPUT instead — "
                    f'`[ -n "$(grep -vE <pattern> <file>)" ]`.'
                )


def _grep_qv_target_files(roots: list[Path] | None) -> list[Path]:
    """Resolve the scan set: production (``roots=None``) walks
    ``.claude/skills/**/SKILL.md`` + ``.claude/agents/*.md`` +
    ``scripts/**/*.sh``; a test override lists files, or directories
    walked for ``*.md`` / ``*.sh``."""
    if roots is None:
        files: list[Path] = []
        skills = _REPO_ROOT / ".claude" / "skills"
        agents = _REPO_ROOT / ".claude" / "agents"
        scripts = _REPO_ROOT / "scripts"
        if skills.exists():
            files.extend(sorted(p for p in skills.rglob("SKILL.md") if p.is_file()))
        if agents.exists():
            files.extend(sorted(p for p in agents.glob("*.md") if p.is_file()))
        if scripts.exists():
            files.extend(sorted(p for p in scripts.rglob("*.sh") if p.is_file()))
        return files
    files = []
    for root in roots:
        if root.is_file():
            files.append(root)
        else:
            files.extend(
                sorted(p for p in root.rglob("*") if p.is_file() and p.suffix in (".md", ".sh"))
            )
    return files


def check_grep_qv(*, roots: list[Path] | None = None) -> list[str]:
    """FAIL on an UNPINNED ``grep``/``ugrep`` invocation combining q
    (``-q``/``--quiet``/``--silent``) and v (``-v``/``--invert-match``) —
    a combined short token, separated tokens, or the long forms — inside
    executable workflow snippets: fenced code blocks in
    ``.claude/skills/**/SKILL.md`` and ``.claude/agents/*.md``, plus
    ``scripts/**/*.sh`` logical lines.

    ugrep 7.5.0's quiet+invert exit status diverges from GNU grep (rc=1
    even when non-matching lines are selected — its ``-q`` short-circuits
    on MATCH FOUND, not line selected), so an rc-consumed q+v trigger
    silently fails OPEN when shell ``grep`` resolves to ugrep (#928: the
    Step 10d pre-push lint gate classified a 12-file code-bearing payload
    as skip-artifact-only; #1125 rewrote both trigger sites to the
    output-test form).

    Sanctioned forms the check does NOT flag: the output-test
    ``[ -n "$(grep -vE <pattern> <file>)" ]`` (no quiet flag — every
    implementation agrees on what ``-v`` PRINTS), an absolute-path-pinned
    bare grep (command word preceded by ``/``, e.g. under ``/usr/bin``),
    and ``git grep`` (git's own engine, not PATH-shadowable). A
    path-pinned ``ugrep`` DOES flag: it carries the divergent exit status
    by construction, so no pin can sanction it. ``#``-comment lines are
    skipped in both file classes; ``.md`` prose outside fences is never
    scanned. Deliberate scan-set exclusions (extend here if the class
    recurs elsewhere): ``.claude/rules/*.md`` and ``CLAUDE.md`` (prose
    surfaces whose fenced snippets are illustrative, not executed
    verbatim) and ``*.py`` files (this check, its tests, and rule prose
    would self-flag; Python ``subprocess`` grep call sites are outside
    the copy-paste-snippet threat model).

    ``roots`` is a unit-test override hook (see
    :func:`_grep_qv_target_files`); production callers pass None.
    Bundled into the no-flags default run (same policy as
    ``check_piped_git_push``).
    """
    errors: list[str] = []
    for path in _grep_qv_target_files(roots):
        lines = path.read_text(encoding="utf-8").splitlines()
        if path.suffix == ".md":
            in_fence = False
            block: list[str] = []
            block_start = 0
            for idx, line in enumerate(lines):
                if _FENCE_RE.match(line):
                    if in_fence:
                        _grep_qv_scan(path, block, block_start, errors)
                        block = []
                    else:
                        block_start = idx + 1
                    in_fence = not in_fence
                    continue
                if in_fence:
                    block.append(line)
            if in_fence and block:
                # Unterminated trailing fence: scan what was collected
                # (fail toward checking, never toward silence).
                _grep_qv_scan(path, block, block_start, errors)
        else:
            _grep_qv_scan(path, lines, 0, errors)
    return errors


def _upload_or_true_segments(text: str) -> list[str]:
    """Naive split of a comment-stripped logical shell line on ``&&`` and
    ``;`` — the NON-terminal swallow scoping unit for
    :func:`check_upload_or_true`. Deliberately quote-UNAWARE: a quoted
    separator mis-splits toward a false NEGATIVE (the safe direction for a
    pre-commit-gating lint, same philosophy as
    :func:`_strip_sh_trailing_comment`)."""
    return re.split(r"&&|;", text)


def _upload_or_true_error(sh: Path, first_idx: int) -> str:
    """Compose the (deliberately verbose, incident-citing) violation string
    for one swallowed upload/result-persist/result-production line."""
    return (
        f"{sh}:{first_idx + 1}: upload/result-persist/result-production "
        f"command swallows failure with '|| true' / '; true'. A swallowed "
        f"failure on a result-bearing phase silently loses artifacts "
        f"(#841: swallowed plot-phase failures compounded a missing upload "
        f"phase — stage JSONs/plots lost across attempts until the "
        f"fail-loud fix) — remove the swallow and let the failure abort "
        f"(fail fast; the crash-persist/poller path reports it). A "
        f"deliberate best-effort side-channel (crash-diagnostics upload) "
        f"is waived with '# UPLOAD_OR_TRUE_EXEMPT: <reason>' (reason ≥ "
        f"{UPLOAD_OR_TRUE_WAIVER_MIN_REASON_CHARS} chars) on the same or "
        f"previous non-blank line."
    )


def _upload_or_true_pyc_block(
    lines: list[str],
    last: int,
    stripped: str,
    pyc: re.Match[str],
) -> tuple[int, bool] | None:
    """Handle :func:`check_upload_or_true` rule 5 — a multi-line
    ``python -c "…"`` quoted block (the CURRENT #841 upload-phase shape).

    ``pyc`` is the opener match on the comment-stripped logical line
    ``stripped``; ``last`` is the logical line's last physical index.
    Returns ``None`` when the captured quote CLOSES on the logical line
    (single-line ``python -c`` — the normal rules apply). Otherwise
    consumes physical lines until the first line containing the closing
    quote char (cap :data:`UPLOAD_OR_TRUE_PYC_MAX_BODY_LINES`) and returns
    ``(consumed_until_idx, swallowed_upload_hit)``: the block flags when
    the command TAIL after the closing quote carries a swallow (terminal,
    or ``|| true`` in a token-bearing segment) AND an upload token matches
    a non-comment body line or the opener/tail. Unclosed at EOF / cap hit
    → ``(scan_window_end, False)`` — the block is skipped entirely
    (fail-toward-false-negative) and its lines are consumed so they are
    never re-parsed as shell. Simplification (plan §4.3 rule 5): a
    first-closing-quote-char scan suffices for this repo's blocks
    (single-quoted bodies cannot contain ``'``; double-quoted bodies carry
    no unescaped ``"``); anything trickier degrades to a false negative."""
    quote = pyc.group(1)
    remainder = stripped[pyc.end() :]
    if quote in remainder:
        return None
    limit = min(len(lines), last + 1 + UPLOAD_OR_TRUE_PYC_MAX_BODY_LINES)
    close_idx = None
    for j in range(last + 1, limit):
        if quote in lines[j]:
            close_idx = j
            break
    if close_idx is None:
        return limit - 1, False
    qpos = lines[close_idx].index(quote)
    tail = _strip_sh_trailing_comment(lines[close_idx][qpos + 1 :]).strip()
    body_frags = [remainder, *lines[last + 1 : close_idx], lines[close_idx][:qpos]]
    swallow_hit = bool(UPLOAD_OR_TRUE_SWALLOW_TERMINAL_RE.search(tail)) or any(
        UPLOAD_OR_TRUE_SWALLOW_OR_RE.search(seg) and UPLOAD_OR_TRUE_LINE_TOKEN_RE.search(seg)
        for seg in _upload_or_true_segments(tail)
    )
    token_hit = (
        any(
            UPLOAD_OR_TRUE_BODY_TOKEN_RE.search(frag)
            for frag in body_frags
            if not frag.strip().startswith("#")
        )
        or bool(UPLOAD_OR_TRUE_LINE_TOKEN_RE.search(stripped))
        or bool(UPLOAD_OR_TRUE_LINE_TOKEN_RE.search(tail))
    )
    return close_idx, swallow_hit and token_hit


def _upload_or_true_line_hit(
    stripped: str, bodies: list[tuple[int, int]], lines: list[str]
) -> bool:
    """Evaluate :func:`check_upload_or_true` rules 2-4 for one
    comment-stripped logical shell line (waivers are the caller's).

    Rule 2 — TERMINAL swallow (``|| true`` / ``|| :`` / ``; true`` at line
    end) hits iff an upload/result token matches ANYWHERE on the line (a
    terminal swallow masks the whole ``&&``-chain). Rule 3 — a
    NON-terminal ``|| true`` / ``|| :`` hits iff a token matches in the
    SAME ``&&``/``;`` segment. Rule 4 — a swallowed heredoc opener
    (terminal, or a segment-scoped swallow in the segment carrying the
    ``<<`` opener; token NOT required on the opener) hits when any of its
    heredoc bodies' non-comment lines carries a BODY upload-call token."""
    terminal = bool(UPLOAD_OR_TRUE_SWALLOW_TERMINAL_RE.search(stripped))
    segments = _upload_or_true_segments(stripped)
    # Rule 2: terminal swallow — whole-logical-line token check.
    if terminal and UPLOAD_OR_TRUE_LINE_TOKEN_RE.search(stripped):
        return True
    # Rule 3: non-terminal swallow — same-segment token check.
    if any(
        UPLOAD_OR_TRUE_SWALLOW_OR_RE.search(seg) and UPLOAD_OR_TRUE_LINE_TOKEN_RE.search(seg)
        for seg in segments
    ):
        return True
    # Rule 4: heredoc bodies under a swallowed opener (token not required
    # on the opener line itself).
    if not bodies:
        return False
    opener_swallowed = terminal or any(
        "<<" in seg and UPLOAD_OR_TRUE_SWALLOW_OR_RE.search(seg) for seg in segments
    )
    if not opener_swallowed:
        return False
    return any(
        UPLOAD_OR_TRUE_BODY_TOKEN_RE.search(lines[j])
        for body_start, body_end in bodies
        for j in range(body_start, min(body_end, len(lines)))
        if not lines[j].strip().startswith("#")
    )


def check_upload_or_true(
    *,
    scripts_dir: Path | None = None,
    allowlist: frozenset[str] | None = None,
) -> list[str]:
    """Walk every ``*.sh`` under ``scripts/`` and FAIL any upload /
    result-persist / result-production (plot-script) command line whose
    failure is swallowed by ``|| true`` / ``|| :`` / ``; true`` (#1036;
    incident #841 — the pre-fix swallows were ``|| true`` on the PLOT
    phases of both #841 dispatch scripts, and the att-7 stage-JSON loss
    additionally involved a MISSING upload phase, which no lint can catch
    and is a named residual, not a claim).

    Detection (see the ``UPLOAD_OR_TRUE_*`` regex block for the full
    flagged/not-flagged matrix), per logical line (backslash continuations
    merged, heredoc bodies consumed via
    :func:`_iter_sh_logicals_with_heredocs`, trailing comments stripped
    quote-aware via :func:`_strip_sh_trailing_comment`):

    1. ``#``-comment and ``echo ``-prefixed logical lines are skipped (an
       echo performs no upload; known accepted FN: ``echo "…"; upload ||
       true`` merged on ONE logical line is skipped whole).
    2. A TERMINAL swallow (``|| true`` / ``|| :`` / ``; true`` at line
       end) flags iff an upload/result token matches ANYWHERE on the line
       — bash ``&&``/``||`` are equal-precedence left-associative, so a
       terminal swallow masks the WHOLE preceding chain.
    3. A NON-terminal ``|| true`` / ``|| :`` flags iff a token matches in
       the SAME ``&&``/``;`` segment (:func:`_upload_or_true_segments`;
       preserves the ``mkdir || true && upload`` FP kill).
    4. A swallowed heredoc opener (terminal swallow, or a segment-scoped
       swallow in the segment carrying the ``<<`` opener; token NOT
       required on the opener) flags when any of its heredoc bodies'
       non-comment lines carries a BODY upload-call token — the i632
       shape a line-only scan provably misses.
    5. A multi-line ``python -c "…"`` quoted block (opener quote unclosed
       on the logical line) is consumed physically until the first line
       containing the closing quote char (cap
       :data:`UPLOAD_OR_TRUE_PYC_MAX_BODY_LINES`; unclosed at EOF or cap
       hit → the block is skipped entirely, fail-toward-false-negative);
       the command TAIL after the closing quote is checked for a swallow
       and the quoted body for upload tokens — the CURRENT #841
       upload-phase shape (``upload_split_lfs_to_overflow(`` bodies).
    6. Violations dedupe per (file, opener first-line).
    7. ``# UPLOAD_OR_TRUE_EXEMPT: <reason>`` (reason ≥ 10 chars, same
       logical line or immediately preceding non-blank line) waives.

    Files whose repo-root-relative path is in
    :data:`UPLOAD_OR_TRUE_LEGACY_ALLOWLIST` are skipped whole-file
    (grandfathered deliberate uses; locked by ``test_live_trees_pass``).
    Named residual evasion shapes (``|| echo WARN``, ``|| rc=$?`` — whose
    same-file-FUNCTION-invocation subclass is now covered by
    :func:`check_sh_function_rc_capture` (#1516), the single-command
    ``|| rc=$?`` shape remaining that check's residual — ``set +e``,
    function-wrapped uploads, subshell-closing-paren swallows)
    are documented in the regex block above — every parse limitation
    degrades to a false NEGATIVE, never a false positive.

    ``scripts_dir`` is an override hook for unit tests; production callers
    pass None and the function walks the canonical ``<repo_root>/scripts``
    tree. ``allowlist`` overrides the legacy allowlist for tests. Bundled
    into the no-flags default run (same policy as ``check_heredoc_dotenv``
    / ``check_dispatcher_cvd_pin``) + the ``workflow-lint-upload-or-true``
    pre-commit hook.
    """
    root = scripts_dir if scripts_dir is not None else _REPO_ROOT / "scripts"
    if not root.exists():
        return []
    allow = UPLOAD_OR_TRUE_LEGACY_ALLOWLIST if allowlist is None else allowlist
    errors: list[str] = []
    for sh in sorted(root.rglob("*.sh")):
        if not sh.is_file():
            continue
        if _judge_pin_rel(sh) in allow:
            continue
        lines = sh.read_text(encoding="utf-8").splitlines()
        flagged: set[int] = set()
        # 0-based physical index; logical lines starting at or before this
        # were consumed as a multi-line `python -c` quoted block.
        pyc_skip_until = -1
        for first, last, logical, bodies in _iter_sh_logicals_with_heredocs(lines):
            if first <= pyc_skip_until or first in flagged:
                continue
            if logical.strip().startswith("#"):
                continue
            stripped = _strip_sh_trailing_comment(logical).strip()
            if not stripped or stripped.startswith("echo "):
                continue

            # Rule 5: multi-line `python -c "…"` quoted block (no heredoc on
            # the same line — a heredoc-bearing opener takes rules 2-4).
            pyc = UPLOAD_OR_TRUE_PYC_OPENER_RE.search(stripped)
            if pyc is not None and not bodies:
                block = _upload_or_true_pyc_block(lines, last, stripped, pyc)
                if block is not None:
                    consumed_until, block_hit = block
                    pyc_skip_until = consumed_until
                    if block_hit and not _sh_waiver_present(
                        lines,
                        first,
                        consumed_until,
                        waiver_re=UPLOAD_OR_TRUE_WAIVER_RE,
                        min_reason_chars=UPLOAD_OR_TRUE_WAIVER_MIN_REASON_CHARS,
                    ):
                        flagged.add(first)
                        errors.append(_upload_or_true_error(sh, first))
                    continue

            if _upload_or_true_line_hit(stripped, bodies, lines) and not _sh_waiver_present(
                lines,
                first,
                last,
                waiver_re=UPLOAD_OR_TRUE_WAIVER_RE,
                min_reason_chars=UPLOAD_OR_TRUE_WAIVER_MIN_REASON_CHARS,
            ):
                flagged.add(first)
                errors.append(_upload_or_true_error(sh, first))
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


# (kind, field) -> reason. Empty today (live-tree probe 2026-07-09: 0 hits
# across 106 markers x 4 fields). Add entries ONLY for deliberate prose that
# trips the signature (e.g. enumeration style "a) ... b) ..."), with a reason.
MARKER_SCALAR_INTEGRITY_ALLOWLIST: dict[tuple[str, str], str] = {}

_MARKER_STRING_FIELDS = ("kind", "posted_by", "when", "fields")


def check_marker_scalar_integrity(
    workflow: WorkflowYaml,
    *,
    allowlist: dict[tuple[str, str], str] | None = None,
) -> list[str]:
    """Flag marker-entry string fields carrying the truncated-comment
    signature: an unquoted YAML plain scalar containing ' #' silently
    truncates at the comment marker (incident #873: posted_by parsed as
    "skill (...); poll_pipeline (runtime tripwire," and --check-references
    passed because the regenerated markers.md matched the truncated parse).
    Signature on the PARSED value: ends in ',' or '(' after rstrip, or
    unbalanced parens. Allowlist grain is (kind, field) with a reason.

    KNOWN RESIDUAL (named deliberately, round-1 critics): a truncation
    landing at a clean word boundary with balanced parens and no trailing
    ','/'(' (e.g. `when: after step 3 #TODO` -> parses to 'after step 3')
    leaves no signature and is undetectable from the parsed value alone —
    a PASS is NOT proof of no truncation. The dominant in-repo comment
    idiom `(…, #NNN)` always leaves the signature, which is what this
    check pins.
    """
    wl = MARKER_SCALAR_INTEGRITY_ALLOWLIST if allowlist is None else allowlist
    errors: list[str] = []
    for m in workflow.markers:
        for field in _MARKER_STRING_FIELDS:
            v = getattr(m, field) or ""
            stripped = v.rstrip()
            trailing = stripped.endswith((",", "("))
            unbalanced = v.count("(") != v.count(")")
            if not (trailing or unbalanced):
                continue
            if (m.kind, field) in wl:
                continue
            errors.append(
                f"workflow.yaml § markers: marker '{m.kind}' field '{field}' has the "
                f"truncated-comment signature ({'trailing ,/(' if trailing else ''}"
                f"{' and ' if trailing and unbalanced else ''}"
                f"{'unbalanced parens' if unbalanced else ''}): {stripped[-60:]!r}. "
                f"An unquoted plain scalar containing ' #' truncates at the comment "
                f"marker (#873) — double-quote the scalar in workflow.yaml, or add "
                f"(kind, field) to MARKER_SCALAR_INTEGRITY_ALLOWLIST with a reason."
            )
    return errors


# PAIRED CONSTANTS (round-1 Alternatives concern): adding a NEW poller
# script => extend BOTH this regex AND _POLLER_TOKEN_TO_FILE below (and
# consider the Leg-A consumer list in check_poller_marker_consumers). A
# poller absent from the regex is invisible to this check.
POLLER_POSTED_BY_RE = re.compile(
    r"poll_pipeline|backend_poll|slurm_monitor|autonomous_session_watch|pod_watch|tick_triage",
    re.IGNORECASE,
)

# Leg B: posted_by token -> the repo-relative poster file that must contain
# the kind string.
_POLLER_TOKEN_TO_FILE: dict[str, str] = {
    "poll_pipeline": "scripts/poll_pipeline.py",
    "backend_poll": "scripts/backend_poll.py",
    "slurm_monitor": "src/explore_persona_space/backends/slurm_monitor.py",
    "autonomous_session_watch": "scripts/autonomous_session_watch.py",
    "pod_watch": "scripts/pod_watch.py",
    "tick_triage": "scripts/tick_triage.py",
}

# kind -> reason. Empty today (live-tree probe 2026-07-09: 5/5 poller-posted
# kinds referenced). Add entries ONLY with a reason naming the deliberate
# out-of-band consumer.
POLLER_CONSUMER_ALLOWLIST: dict[str, str] = {}


def check_poller_marker_consumers(
    workflow: WorkflowYaml,
    *,
    consumer_paths: list[Path] | None = None,
    poller_file_map: dict[str, Path] | None = None,
    allowlist: dict[str, str] | None = None,
) -> list[str]:
    """Every marker kind whose posted_by names a poller/watcher must be
    (Leg A) referenced by >=1 consumer surface — all .claude/skills/**/SKILL.md
    plus tick_triage.py / autonomous_session_watch.py / poll_pipeline.py /
    backend_poll.py / pod_watch.py — and (Leg B) present in each poster script
    its posted_by token names (the #873 pre-fix state: a runtime tripwire
    declared in workflow.yaml with no poll_pipeline code). Overrides narrow the
    scan for fixture tests, mirroring check_marker_registry's
    skill_md/skills_dir hooks.

    KNOWN RESIDUAL (named deliberately, round-1 critics): textual presence
    is the grain — a kind mentioned only in a comment / dead branch of a
    consumer or poster file passes both legs (live read-site detection
    needs AST-grade parsing, out of scope for grep-grade lint). Errors here
    are lenient-direction only.
    """
    wl = POLLER_CONSUMER_ALLOWLIST if allowlist is None else allowlist
    production = consumer_paths is None and poller_file_map is None
    if production:
        skills_dir = _REPO_ROOT / ".claude" / "skills"
        consumers: list[Path] = []
        if skills_dir.is_dir():
            consumers.extend(sorted(p for p in skills_dir.glob("**/SKILL.md") if p.is_file()))
        consumers.extend(
            _REPO_ROOT / "scripts" / name
            for name in (
                "tick_triage.py",
                "autonomous_session_watch.py",
                "poll_pipeline.py",
                "backend_poll.py",
                "pod_watch.py",
            )
        )
        poller_files = {token: _REPO_ROOT / rel for token, rel in _POLLER_TOKEN_TO_FILE.items()}
        surface_desc = (
            ".claude/skills/**/SKILL.md + scripts/{tick_triage,"
            "autonomous_session_watch,poll_pipeline,backend_poll,pod_watch}.py"
        )
    else:
        consumers = list(consumer_paths or [])
        poller_files = dict(poller_file_map or {})
        surface_desc = ", ".join(str(p) for p in consumers) or "(no consumer surfaces supplied)"

    consumer_texts = [p.read_text() for p in consumers if p.exists()]
    poller_text_cache: dict[str, str] = {}
    errors: list[str] = []
    for m in workflow.markers:
        posted_by = m.posted_by or ""
        if not POLLER_POSTED_BY_RE.search(posted_by):
            continue
        if m.kind in wl:
            continue
        # Leg A: at least one consumer surface references the kind.
        if not any(m.kind in text for text in consumer_texts):
            errors.append(
                f"workflow.yaml § markers: poller-posted marker '{m.kind}' "
                f"(posted_by: {posted_by[:80]}) is referenced by NO consumer "
                f"surface ({surface_desc}). A poller feature claiming mid-run "
                f"surfacing must be reachable by watcher/tick/poll/orchestrator "
                f"code (#873) — wire a consumer or add the kind to "
                f"POLLER_CONSUMER_ALLOWLIST with a reason."
            )
        # Leg B: each poster script the posted_by token names contains the kind.
        lowered = posted_by.lower()
        for token, poster in sorted(poller_files.items()):
            if token not in lowered:
                continue
            if not poster.exists():
                # A missing PRODUCTION poster file for a matched token is an
                # ERROR (fail loud); a missing OVERRIDE path is skipped, the
                # check_marker_registry missing-file convention.
                if production:
                    errors.append(
                        f"workflow.yaml § markers: marker '{m.kind}' posted_by "
                        f"names poller token '{token}' but the mapped poster file "
                        f"'{poster}' does not exist — fix _POLLER_TOKEN_TO_FILE "
                        f"or restore the script."
                    )
                continue
            if token not in poller_text_cache:
                poller_text_cache[token] = poster.read_text()
            if m.kind not in poller_text_cache[token]:
                errors.append(
                    f"workflow.yaml § markers: poller-posted marker '{m.kind}' — "
                    f"declared poster '{poster}' (token '{token}') does not "
                    f"mention '{m.kind}' — the posting code may not exist (#873 "
                    f"pre-fix state). Wire the poster or add the kind to "
                    f"POLLER_CONSUMER_ALLOWLIST with a reason."
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
        tree = _cached_parse(py, text)
        if tree is None:
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


def _hub_dir_filecount_waiver_present(lines: list[str], call_lineno: int) -> bool:
    """Return True iff a ``# HUB_DIR_FILECOUNT_EXEMPT: <reason>`` waiver
    (reason ≥ :data:`HUB_DIR_FILECOUNT_WAIVER_MIN_REASON_CHARS` chars) is on
    the call's first physical line (``call_lineno``, 1-based) or the
    immediately preceding non-blank line. Same convention as
    :func:`_upload_as_file_waiver_present`."""
    idx = call_lineno - 1  # to 0-based
    if 0 <= idx < len(lines):
        m = HUB_DIR_FILECOUNT_WAIVER_RE.search(lines[idx])
        if m and len(m.group(1).strip()) >= HUB_DIR_FILECOUNT_WAIVER_MIN_REASON_CHARS:
            return True
    back = idx - 1
    while back >= 0 and lines[back].strip() == "":
        back -= 1
    if back >= 0:
        m = HUB_DIR_FILECOUNT_WAIVER_RE.search(lines[back])
        if m and len(m.group(1).strip()) >= HUB_DIR_FILECOUNT_WAIVER_MIN_REASON_CHARS:
            return True
    return False


def check_hub_dir_filecount_guard(
    *, scripts_dir: Path | None = None, legacy_allowlist: frozenset[str] | None = None
) -> list[str]:
    """AST-walk every ``*.py`` under ``scripts/`` and FAIL on any DIRECT
    ``upload_folder(...)`` call site whose module does not reference the hub
    dir-filecount guard ``assert_hub_dir_filecounts`` (#1190).

    Rationale: the HF Hub rejects any single repo directory holding >10000
    files at COMMIT time — a NON-retriable ``BadRequestError`` fired AFTER
    the full compute has run and every byte is staged (#658 r2: 12000
    rollout ``.pt`` files + 12000 transcripts in one dir each; per-file
    uploads succeeded, the final ``create_commit`` 400'd). The shared hub
    helpers (``hub._upload`` folder branch / ``hub._upload_folder_filtered``)
    pre-count staged files per TARGET repo dir and raise
    ``HubDirFileCountError`` BEFORE any network I/O — but the #658 incident's
    own call path used ``HfApi`` DIRECTLY from a per-issue script, bypassing
    hub.py entirely. This check is the funnel: a direct ``upload_folder``
    call site in ``scripts/`` must reference the one-line guard, carry an
    exemption comment, or be on the grandfathered legacy allowlist (the same
    funnel-to-the-guarded-helper role ``--check-upload-as-file`` plays for
    ``hub._upload``).

    Detection (exact-name match only — ``upload_folder_verified`` /
    ``upload_folder_scoped_verify`` / ``_upload_folder_filtered`` do NOT
    match):

    * an ``ast.Attribute`` call with ``attr == "upload_folder"`` (matches
      ``api.upload_folder(``, ``HfApi().upload_folder(``,
      ``self.api.upload_folder(``);
    * an ``ast.Name`` call with ``id == "upload_folder"`` (a
      ``from huggingface_hub import upload_folder`` caller) — UNLESS the
      module defines its own local ``def upload_folder`` (the
      ``scripts/issue623_upload.py`` wrapper shape; the carve-out, not the
      allowlist, is such a module's pass condition).

    Pass conditions (any one suffices):

    1. the enclosing MODULE references ``assert_hub_dir_filecounts``
       anywhere (any ``ast.Name`` / ``ast.Attribute`` with that identifier)
       — module-level granularity, the proportionate v1 (per-call-site
       granularity is a possible future tightening);
    2. a ``# HUB_DIR_FILECOUNT_EXEMPT: <reason>`` waiver (reason ≥
       :data:`HUB_DIR_FILECOUNT_WAIVER_MIN_REASON_CHARS` chars) on the
       call's first physical line or the immediately preceding non-blank
       line;
    3. the file's repo-relative posix path is in
       :data:`HUB_DIR_FILECOUNT_LEGACY_ALLOWLIST` (grandfathered
       pre-existing experiment code; grep-generated + live-tree-test-pinned,
       never hand-extended).

    ``scripts_dir`` / ``legacy_allowlist`` are override hooks for unit
    tests; production callers pass None and the function walks the canonical
    ``<repo_root>/scripts`` tree against the module allowlist. Allowlist
    paths are computed relative to the WALK ROOT'S PARENT (so production
    paths read ``scripts/<name>.py`` and tmp_path fixtures resolve
    consistently). Bundled into the no-flags default run.
    """
    root = scripts_dir if scripts_dir is not None else _REPO_ROOT / "scripts"
    if not root.exists():
        return []
    allow = HUB_DIR_FILECOUNT_LEGACY_ALLOWLIST if legacy_allowlist is None else legacy_allowlist
    errors: list[str] = []
    for py in sorted(root.rglob("*.py")):
        if not py.is_file():
            continue
        rel = py.relative_to(root.parent).as_posix()
        if rel in allow:
            continue  # grandfathered pre-existing direct call sites
        text = py.read_text(encoding="utf-8")
        tree = _cached_parse(py, text)
        if tree is None:
            # A scripts/ file that does not parse is its own (separate)
            # problem; this check stays silent on it rather than crashing.
            continue
        # Pass condition 1: any reference to the guard helper anywhere in
        # the module (import, bare call, hub.assert_hub_dir_filecounts(...))
        # whitelists the whole module.
        guarded = any(
            (isinstance(n, ast.Name) and n.id == "assert_hub_dir_filecounts")
            or (isinstance(n, ast.Attribute) and n.attr == "assert_hub_dir_filecounts")
            for n in ast.walk(tree)
        )
        if guarded:
            continue
        # Carve-out for the bare-Name arm: a module that DEFINES its own
        # `def upload_folder` calls its local wrapper, not the huggingface_hub
        # function (scripts/issue623_upload.py — 4 would-be false positives).
        local_def = any(
            isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef) and n.name == "upload_folder"
            for n in ast.walk(tree)
        )
        lines = text.splitlines()
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            fn = node.func
            if isinstance(fn, ast.Attribute) and fn.attr == "upload_folder":
                pass  # api.upload_folder(...) — always a candidate
            elif isinstance(fn, ast.Name) and fn.id == "upload_folder":
                if local_def:
                    continue  # local wrapper call — carved out
            else:
                continue
            if _hub_dir_filecount_waiver_present(lines, node.lineno):
                continue
            errors.append(
                f"{py}:{node.lineno}: direct upload_folder(...) call without the hub "
                f"dir-filecount guard. The Hub rejects any single repo directory holding "
                f">10k files at COMMIT time with a NON-retriable BadRequestError, fired "
                f"AFTER the full compute ran and all bytes are staged (#658). One-line "
                f"fix: `from explore_persona_space.orchestrate.hub import "
                f"assert_hub_dir_filecounts` and call it on the SAME folder_path / "
                f"path_in_repo / allow+ignore patterns BEFORE the upload, OUTSIDE any "
                f"transient-retry wrapper (a guard raise is deterministic — retrying it "
                f"burns the retry budget for nothing) — or route through hub._upload / "
                f"hub._upload_folder_filtered, which are guarded. Waive a "
                f"genuinely-correct call with '# HUB_DIR_FILECOUNT_EXEMPT: <reason>' "
                f"(reason >= {HUB_DIR_FILECOUNT_WAIVER_MIN_REASON_CHARS} chars) on the "
                f"call's first line or the previous non-blank line. See "
                f".claude/rules/gotchas.md 'HF Hub rejects any single repo directory "
                f"holding >10000 files at COMMIT time'."
            )
    return errors


def _upload_loop_waiver_present(lines: list[str], call_lineno: int) -> bool:
    """Return True iff a ``# UPLOAD_LOOP_EXEMPT: <reason>`` waiver
    (reason ≥ :data:`UPLOAD_LOOP_WAIVER_MIN_REASON_CHARS` chars) is on
    the call's first physical line (``call_lineno``, 1-based) or the
    immediately preceding non-blank line. Same placement semantics as
    :func:`_upload_as_file_waiver_present`."""
    idx = call_lineno - 1  # to 0-based
    if 0 <= idx < len(lines):
        m = UPLOAD_LOOP_WAIVER_RE.search(lines[idx])
        if m and len(m.group(1).strip()) >= UPLOAD_LOOP_WAIVER_MIN_REASON_CHARS:
            return True
    back = idx - 1
    while back >= 0 and lines[back].strip() == "":
        back -= 1
    if back >= 0:
        m = UPLOAD_LOOP_WAIVER_RE.search(lines[back])
        if m and len(m.group(1).strip()) >= UPLOAD_LOOP_WAIVER_MIN_REASON_CHARS:
            return True
    return False


_UPLOAD_LOOP_NODES = (ast.For, ast.AsyncFor, ast.While)
_UPLOAD_LOOP_COMPS = (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)
_UPLOAD_LOOP_BOUNDARIES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)


def _iter_calls_in_loop_context(tree: ast.Module):
    """Yield every ``ast.Call`` lexically inside a loop / comprehension.

    Iterative explicit-stack walk (no recursion — ``run_experiment_444.py``
    is 5.5k+ lines) carrying an ``in_loop`` bit per node: SET on entering a
    loop / comprehension child, CLEARED on entering a def / class / lambda
    boundary child (a function *defined* in a loop executes elsewhere, so
    its body is not in-loop). Accepted imprecision: a call in a loop's
    HEADER expression (``for x in upload_file(...)``) is treated as in-loop
    even though the ``for``-iter form executes once — no such call exists
    in the tree and the waiver covers the hypothetical."""
    stack: list[tuple[ast.AST, bool]] = [(tree, False)]
    while stack:
        node, in_loop = stack.pop()
        if in_loop and isinstance(node, ast.Call):
            yield node
        for child in ast.iter_child_nodes(node):
            if isinstance(child, _UPLOAD_LOOP_BOUNDARIES):
                stack.append((child, False))
            elif isinstance(child, _UPLOAD_LOOP_NODES + _UPLOAD_LOOP_COMPS):
                stack.append((child, True))
            else:
                stack.append((child, in_loop))


def check_upload_file_in_loop(
    *, scripts_dir: Path | None = None, legacy_allowlist: dict[str, int] | None = None
) -> list[str]:
    """AST-walk every ``*.py`` under ``scripts/`` and FAIL on any per-file
    upload call lexically inside a loop (#664 / #658-r4 / #1481 per-file
    Hub-commit storm class).

    Rationale: each per-file upload call composes ONE Hub commit and
    triggers a server-side repo pre-check, so an N-file loop issues N
    commits + N pre-checks — 504-storming on a large repo (#664: a
    1425-file raw-completions loop ran 12h on an idle 8xH200, ~$530,
    uploading only 264 files) and tripping the org-level ~2500-req/5-min
    429 quota (#1481: a driver planned ~1400 per-file commits -> HF 429
    storm). Bulk uploads compose ONE ``upload_folder`` commit
    (``upload_raw_completions_to_data_repo()`` /
    ``hub._upload_folder_filtered`` are the shared bulk paths). The prose
    rule lived in ``.claude/rules/gotchas.md`` ("Per-file
    ``HfApi.upload_file`` 504-storms on a LARGE repo") and failed to stop
    #1481 through plan + implementation review; this check is the
    mechanical gate.

    Detection — an ``ast.Call`` lexically enclosed by ``For`` / ``AsyncFor``
    / ``While`` / a comprehension (loop context RESET at function / lambda
    / class boundaries; lexical only — a helper *called* from a loop is a
    deliberate false negative, the task's fail-toward-false-negative
    direction), matching EITHER shape:

    * **shape A** — the callee is literally named ``upload_file``
      (attribute form ``api.upload_file(`` or a bare
      ``from huggingface_hub import upload_file`` name). Exact-name only:
      ``upload_files`` / ``upload_folder`` / ``upload_file_to_hf`` do NOT
      match. No local-``def upload_file`` carve-out (deliberate divergence
      from :func:`check_hub_dir_filecount_guard`): a per-file wrapper
      called in a loop still commits once per call — exactly the
      anti-pattern — so it flags; a genuinely-batching wrapper takes the
      waiver.
    * **shape B** — the callee is literally named ``_upload`` AND the call
      carries an explicit ``upload_as_file=True`` constant kwarg — the
      literal #664 form (``for f in files: hub._upload(f,
      upload_as_file=True)``). REQUIRED because
      :func:`check_upload_as_file` deliberately DEFERS any ``_upload``
      call with an explicit ``upload_as_file`` kwarg (the author's
      file/folder declaration) and the #595 runtime guard *forces*
      per-file callers onto exactly this kwarg — without shape B the most
      probable future offender form passes every lint. An in-loop
      ``_upload`` WITHOUT the kwarg is deliberately NOT matched (a per-dir
      folder loop is legitimate; the single-file form crashes fail-loud on
      the first file via the #595 ``ValueError``, and its static forms
      belong to ``--check-upload-as-file``).

    Pass conditions: a ``# UPLOAD_LOOP_EXEMPT: <reason>`` waiver (reason ≥
    :data:`UPLOAD_LOOP_WAIVER_MIN_REASON_CHARS` chars) on the call's first
    physical line or the immediately preceding non-blank line; or the
    file's findings are covered by the grandfather allowlist
    :data:`UPLOAD_FILE_IN_LOOP_LEGACY_ALLOWLIST` — COUNT-grain: a file's
    findings are suppressed only while their count ≤ the grandfathered N;
    an excess count reports ALL of the file's findings plus a
    count-exceeded note (a NEW offense in a grandfathered file surfaces
    instead of hiding behind the entry; known netting quirk — fixing one
    old site while adding one new nets the same count — accepted, still
    strictly stronger than the siblings' file-grain frozenset).

    ``scripts_dir`` / ``legacy_allowlist`` are override hooks for unit
    tests; production callers pass None and the function walks the
    canonical ``<repo_root>/scripts`` tree against the module allowlist.
    Allowlist paths are computed relative to the WALK ROOT'S PARENT (so
    production paths read ``scripts/<name>.py`` and tmp_path fixtures
    resolve consistently). Bundled into the no-flags default run.
    """
    root = scripts_dir if scripts_dir is not None else _REPO_ROOT / "scripts"
    if not root.exists():
        return []
    allow = UPLOAD_FILE_IN_LOOP_LEGACY_ALLOWLIST if legacy_allowlist is None else legacy_allowlist
    errors: list[str] = []
    for py in sorted(root.rglob("*.py")):
        if not py.is_file():
            continue
        rel = py.relative_to(root.parent).as_posix()
        text = py.read_text(encoding="utf-8")
        tree = _cached_parse(py, text)
        if tree is None:
            # A scripts/ file that does not parse is its own (separate)
            # problem; this check stays silent on it rather than crashing.
            continue
        lines = text.splitlines()
        file_findings: list[str] = []
        for node in _iter_calls_in_loop_context(tree):
            fn = node.func
            fn_name = (
                fn.attr
                if isinstance(fn, ast.Attribute)
                else (fn.id if isinstance(fn, ast.Name) else None)
            )
            is_shape_a = fn_name == "upload_file"
            is_shape_b = fn_name == "_upload" and any(
                kw.arg == "upload_as_file"
                and isinstance(kw.value, ast.Constant)
                and kw.value.value is True
                for kw in node.keywords
            )
            if not (is_shape_a or is_shape_b):
                continue
            if _upload_loop_waiver_present(lines, node.lineno):
                continue
            shape = "upload_file(...)" if is_shape_a else "_upload(..., upload_as_file=True)"
            file_findings.append(
                f"{py}:{node.lineno}: {shape} "
                f"call inside a loop — the per-file upload loop is the #664/#1481 "
                f"storm anti-pattern (each call = one commit + a server-side repo "
                f"pre-check; a 1425-file loop ran 12h/$530 idle in #664; ~1400 "
                f"planned commits 429-stormed in #1481). Batch into ONE bulk "
                f"commit: HfApi.upload_folder with allow_patterns, "
                f"upload_raw_completions_to_data_repo(), or "
                f"hub._upload_folder_filtered. A genuinely bounded loop (a retry "
                f"wrapper around a SINGLE file, a fixed <=3-file list) may be "
                f"waived with '# UPLOAD_LOOP_EXEMPT: <reason>' (reason >= "
                f"{UPLOAD_LOOP_WAIVER_MIN_REASON_CHARS} chars) on the call's first "
                f"line or the previous non-blank line. See .claude/rules/gotchas.md "
                f"'Per-file HfApi.upload_file 504-storms on a LARGE repo'."
            )
        # Grandfather gate: per-file COUNT vs the allowlist dict — findings
        # suppressed only while count <= grandfathered N; an excess count
        # reports ALL of the file's findings (a new offense in a
        # grandfathered file surfaces instead of hiding behind the entry).
        allowed = allow.get(rel, 0)
        if len(file_findings) <= allowed:
            continue
        if allowed:
            errors.append(
                f"{py}: {len(file_findings)} in-loop per-file-upload finding(s) exceed "
                f"the grandfathered count ({allowed}) in "
                f"UPLOAD_FILE_IN_LOOP_LEGACY_ALLOWLIST — a NEW in-loop per-file upload "
                f"was added to a grandfathered file; all of its findings are reported "
                f"below. Batch the new upload into one upload_folder commit (or waive "
                f"with '# UPLOAD_LOOP_EXEMPT: <reason>') — never extend the allowlist."
            )
        errors.extend(file_findings)
    return errors


def _upload_return_discard_waiver_present(lines: list[str], call_lineno: int) -> bool:
    """Return True iff a ``# UPLOAD_RETURN_DISCARD_EXEMPT: <reason>`` waiver
    (reason ≥ :data:`UPLOAD_RETURN_DISCARD_WAIVER_MIN_REASON_CHARS` chars) is
    on the call's first physical line (``call_lineno``, 1-based) or the
    immediately preceding non-blank line. Same convention as
    :func:`_upload_as_file_waiver_present`."""
    idx = call_lineno - 1  # to 0-based
    if 0 <= idx < len(lines):
        m = UPLOAD_RETURN_DISCARD_WAIVER_RE.search(lines[idx])
        if m and len(m.group(1).strip()) >= UPLOAD_RETURN_DISCARD_WAIVER_MIN_REASON_CHARS:
            return True
    back = idx - 1
    while back >= 0 and lines[back].strip() == "":
        back -= 1
    if back >= 0:
        m = UPLOAD_RETURN_DISCARD_WAIVER_RE.search(lines[back])
        if m and len(m.group(1).strip()) >= UPLOAD_RETURN_DISCARD_WAIVER_MIN_REASON_CHARS:
            return True
    return False


# The two hub helpers whose failure contract is fail-soft BY RETURN, and the
# module paths whose imports arm the check (see check_upload_return_discard).
_UPLOAD_RETURN_DISCARD_TARGETS = frozenset({"_upload", "_upload_folder_filtered"})
_URD_HUB_MODULE = "explore_persona_space.orchestrate.hub"
_URD_HUB_PARENT = "explore_persona_space.orchestrate"


def check_upload_return_discard(  # noqa: C901 -- two-pass binding-collection + firing walk (plan #2087 §4.1); extracting a branch would just relocate it
    *, scripts_dir: Path | None = None, legacy_allowlist: dict[str, int] | None = None
) -> list[str]:
    """AST-walk every ``*.py`` under ``scripts/`` and FAIL on any
    Expr-statement (discarded-return) call to the fail-soft-by-return hub
    upload helpers ``_upload`` / ``_upload_folder_filtered`` (#2087;
    incident #2054).

    Rationale: ``explore_persona_space.orchestrate.hub._upload``
    (hub.py ~1426) returns ``""`` on missing ``HF_TOKEN``, an absent local
    path, and failed post-upload verification — and on upload exceptions
    unless ``raise_on_error=True`` (the docstring: "ONLY the exception path
    changes"). ``hub._upload_folder_filtered`` (hub.py ~1671) returns
    ``"{repo_id}/{path_in_repo}"`` on verified success and ``""`` on EVERY
    upload-failure shape (its pre-flight ``assert_hub_dir_filecounts``
    guard — the #1190 per-dir cap check, outside the swallowing try — still
    raises; upload failures themselves never do). A caller that discards
    the return converts a durability failure into a false-success exit 0 —
    the class ``.claude/rules/upload-policy.md`` bans ("'upload returned no
    path' is a TRACKED GAP ... never a warning-and-continue"). Six such
    sites reached main across issue2054 phase scripts despite full review
    rounds; this check is the mechanical gate. The canonical fix shape is
    capture-and-raise (``hub.upload_raw_completions_to_data_repo``,
    hub.py ~2152: ``base_url = _upload_folder_filtered(...)`` then
    ``if not base_url: raise RuntimeError(...)``).

    Detection — import/definition-resolved arming, two passes per file:

    * **Pass 1 (binding collection):** name bindings from
      ``from explore_persona_space.orchestrate.hub import _upload [as X]``
      (``ast.walk`` covers function-local imports — the issue2054_capture /
      issue1689_capture shape); module aliases from
      ``from explore_persona_space.orchestrate import hub [as H]`` and
      ``import explore_persona_space.orchestrate.hub as H`` (a bare dotted
      ``import`` with no asname produces a 3-deep attribute chain at the
      call site — out of scope v1, zero live sites at plan time); and a
      shadow-disarm set from any ``def``/``async def``/assignment binding a
      bare target name (a same-named LOCAL helper — e.g. the fail-LOUD
      ``_upload`` wrappers in issue1481/issue825/issue952 scripts — never
      arms the Name form; conservative: prefer a false negative over
      firing on an unread local contract).
    * **Pass 2 (firing rule):** every ``ast.Expr`` statement whose value
      (unwrapping one ``await``) is a Call to an armed Name, or to an
      Attribute ``<hub-alias>._upload`` / ``<hub-alias>._upload_folder_filtered``,
      is a finding — the return value is unreachable BY CONSTRUCTION.
      A consumed return (assignment incl. ``_ =``, walrus, ``return`` /
      ``yield``, a condition, an argument position) is never an Expr
      statement's direct value, so it never fires. ``raise_on_error=True``
      calls STILL fire (the non-exception ``""`` returns are unchanged —
      three failure shapes stay silent). Known v1 false negatives,
      accepted + documented: the bare dotted-chain import form, a Call
      nested inside a tuple-expression statement, and the greppable
      ``_ = _upload(...)`` deliberate-discard idiom.

    Pass conditions: a ``# UPLOAD_RETURN_DISCARD_EXEMPT: <reason>`` waiver
    (reason ≥ :data:`UPLOAD_RETURN_DISCARD_WAIVER_MIN_REASON_CHARS` chars)
    on the call's first physical line or the immediately preceding
    non-blank line; or the file's findings are covered by
    :data:`UPLOAD_RETURN_DISCARD_LEGACY_ALLOWLIST` — COUNT-grain and
    <=-tolerant (the :func:`check_upload_file_in_loop` gate, verbatim):
    findings suppressed only while their count <= the grandfathered N, so
    a sibling task's fix landing (a count DROP) keeps main green in either
    merge order; an excess count reports ALL of the file's findings.
    Entries owned by live in-flight tasks are additionally listed in
    :data:`UPLOAD_RETURN_DISCARD_PENDING_OWNER` (the load-bearing test
    pins those ``observed <= pinned`` instead of exact).

    v1 scope notes, both measured at implement time (2026-08-05): the walk
    covers ``scripts/`` only — an independent sweep of
    ``src/explore_persona_space/`` (this check pointed at ``src/`` plus a
    statement-position grep for both helper names, name AND attribute
    form) found ZERO statement-shaped calls (``src/`` imports only the
    public wrappers); and the sibling ``-> str``-returning-``""`` wrappers
    ``upload_model`` / ``upload_dataset`` have zero live discard-shaped
    callers — the only statement-position hits live in the frozen,
    ruff-excluded ``scripts/archive/`` (2 ``upload_dataset`` sites), so
    the v1 target set stays the two private helpers.

    ``scripts_dir`` / ``legacy_allowlist`` are override hooks for unit
    tests; production callers pass None and the function walks the
    canonical ``<repo_root>/scripts`` tree against the module allowlist.
    Allowlist paths are computed relative to the WALK ROOT'S PARENT (so
    production paths read ``scripts/<name>.py``). Read-only over
    :func:`_cached_parse` trees (SHARED across checks — never mutate
    nodes). Bundled into the no-flags default run.
    """
    root = scripts_dir if scripts_dir is not None else _REPO_ROOT / "scripts"
    if not root.exists():
        return []
    allow = UPLOAD_RETURN_DISCARD_LEGACY_ALLOWLIST if legacy_allowlist is None else legacy_allowlist
    errors: list[str] = []
    for py in sorted(root.rglob("*.py")):
        if not py.is_file():
            continue
        rel = py.relative_to(root.parent).as_posix()
        text = py.read_text(encoding="utf-8")
        tree = _cached_parse(py, text)
        if tree is None:
            # A scripts/ file that does not parse is its own (separate)
            # problem; this check stays silent on it rather than crashing.
            continue
        # Pass 1 — binding collection (read-only; cached trees are SHARED).
        hub_name_bindings: dict[str, str] = {}  # local alias -> hub helper name
        hub_module_aliases: set[str] = set()
        shadow_disarm: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if node.module == _URD_HUB_MODULE:
                    for alias in node.names:
                        if alias.name in _UPLOAD_RETURN_DISCARD_TARGETS:
                            hub_name_bindings[alias.asname or alias.name] = alias.name
                elif node.module == _URD_HUB_PARENT:
                    for alias in node.names:
                        if alias.name == "hub":
                            hub_module_aliases.add(alias.asname or "hub")
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == _URD_HUB_MODULE and alias.asname:
                        hub_module_aliases.add(alias.asname)
            elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                if node.name in _UPLOAD_RETURN_DISCARD_TARGETS:
                    shadow_disarm.add(node.name)
            elif isinstance(node, ast.Assign):
                for tgt in node.targets:
                    if isinstance(tgt, ast.Name) and tgt.id in _UPLOAD_RETURN_DISCARD_TARGETS:
                        shadow_disarm.add(tgt.id)
            elif isinstance(node, ast.AnnAssign):
                tgt = node.target
                if isinstance(tgt, ast.Name) and tgt.id in _UPLOAD_RETURN_DISCARD_TARGETS:
                    shadow_disarm.add(tgt.id)
        if not hub_name_bindings and not hub_module_aliases:
            continue
        lines = text.splitlines()
        # Pass 2 — Expr-statement (discarded-return) calls to armed names.
        file_findings: list[str] = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Expr):
                continue
            v = node.value
            if isinstance(v, ast.Await):
                v = v.value  # an awaited discard is still a discard
            if not isinstance(v, ast.Call):
                continue
            fn = v.func
            helper: str | None = None
            if isinstance(fn, ast.Name):
                target = hub_name_bindings.get(fn.id)
                if target is not None and fn.id not in shadow_disarm:
                    helper = target
            elif isinstance(fn, ast.Attribute):
                if (
                    fn.attr in _UPLOAD_RETURN_DISCARD_TARGETS
                    and isinstance(fn.value, ast.Name)
                    and fn.value.id in hub_module_aliases
                ):
                    helper = fn.attr
            if helper is None:
                continue
            if _upload_return_discard_waiver_present(lines, node.lineno):
                continue
            if helper == "_upload":
                raise_kw = next((kw.value for kw in v.keywords if kw.arg == "raise_on_error"), None)
                raise_true = isinstance(raise_kw, ast.Constant) and raise_kw.value is True
                exc_shape = "" if raise_true else " / upload exception"
                shapes = f"missing HF_TOKEN / absent local path / failed verify{exc_shape}"
            else:
                shapes = (
                    "missing HF_TOKEN / failed post-upload verify / upload exception "
                    "(only the pre-flight assert_hub_dir_filecounts cap guard raises)"
                )
            file_findings.append(
                f"{py}:{node.lineno}: discarded return of {helper}(...) — {helper} is "
                f"fail-soft by RETURN ('' on {shapes}), so a discarded return exits 0 "
                f"on silent durability loss (.claude/rules/upload-policy.md: 'upload "
                f"returned no path' is a TRACKED GAP, never warning-and-continue). "
                f"Capture and raise (the hub.upload_raw_completions_to_data_repo shape: "
                f"base_url = {helper}(...); if not base_url: raise RuntimeError(...)), "
                f"or waive with '# UPLOAD_RETURN_DISCARD_EXEMPT: <reason>' (reason >= "
                f"{UPLOAD_RETURN_DISCARD_WAIVER_MIN_REASON_CHARS} chars) on the call's "
                f"first line or the previous non-blank line."
            )
        # Grandfather gate: per-file COUNT vs the allowlist dict — findings
        # suppressed only while count <= grandfathered N (<=-tolerant: a
        # sibling fix's count DROP stays green); an excess count reports
        # ALL of the file's findings.
        allowed = allow.get(rel, 0)
        if len(file_findings) <= allowed:
            continue
        if allowed:
            errors.append(
                f"{py}: {len(file_findings)} discarded-return finding(s) exceed the "
                f"grandfathered count ({allowed}) in "
                f"UPLOAD_RETURN_DISCARD_LEGACY_ALLOWLIST — a NEW discarded hub-upload "
                f"return was added to a grandfathered file; all of its findings are "
                f"reported below. Capture and raise (or waive with "
                f"'# UPLOAD_RETURN_DISCARD_EXEMPT: <reason>') — never extend the "
                f"allowlist."
            )
        errors.extend(file_findings)
    return errors


def _upc_waiver_present(lines: list[str], lineno: int) -> bool:
    """Return True iff a ``# UPLOAD_PREFIX_EXEMPT: <reason>`` waiver (reason ≥
    :data:`UPLOAD_PREFIX_WAIVER_MIN_REASON_CHARS` chars) is on the finding's
    first physical line (``lineno``, 1-based) or the immediately preceding
    non-blank line. Same convention as :func:`_upload_as_file_waiver_present`."""
    idx = lineno - 1  # to 0-based
    if 0 <= idx < len(lines):
        m = UPLOAD_PREFIX_WAIVER_RE.search(lines[idx])
        if m and len(m.group(1).strip()) >= UPLOAD_PREFIX_WAIVER_MIN_REASON_CHARS:
            return True
    back = idx - 1
    while back >= 0 and lines[back].strip() == "":
        back -= 1
    if back >= 0:
        m = UPLOAD_PREFIX_WAIVER_RE.search(lines[back])
        if m and len(m.group(1).strip()) >= UPLOAD_PREFIX_WAIVER_MIN_REASON_CHARS:
            return True
    return False


def _upc_module_const_tokens(
    tree: ast.Module, imports: dict[str, str] | None = None
) -> dict[str, frozenset[str]]:
    """Map module-level constant names to the issue tokens their value
    expressions carry. Value shapes considered: str ``Constant``,
    ``JoinedStr``, ``BinOp`` (str concat/format), or a ``Call`` (e.g.
    ``os.environ.get("X", "issue958_multiturn")``). Tokens = every
    ``issue<M>_`` / ``issue<M>/`` match over every string literal inside the
    value expr, plus (TRANSITIVE, in module-body order) the tokens of any
    already-mapped ``Name`` it references — resolving BOTH locally-built
    consts AND imported names via the step-3 import map (plan §4.3.2's
    ``DECOMP = f"{HF_PREFIX_928}/analysis_tensors/decomp"`` example, where
    ``HF_PREFIX_928`` is imported from ``issue928_common``)."""
    imports = imports or {}
    consts: dict[str, frozenset[str]] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
            value: ast.expr | None = node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            targets = [node.target.id]
            value = node.value
        else:
            continue
        if not targets or not isinstance(
            value, ast.Constant | ast.JoinedStr | ast.BinOp | ast.Call
        ):
            continue
        if isinstance(value, ast.Constant) and not isinstance(value.value, str):
            continue
        tokens: set[str] = set()
        for sub in ast.walk(value):
            if isinstance(sub, ast.Constant) and isinstance(sub.value, str):
                tokens.update(_UPC_ISSUE_TOKEN_RE.findall(sub.value))
            elif isinstance(sub, ast.Name):
                if sub.id in consts:
                    tokens.update(consts[sub.id])
                elif sub.id in imports:
                    tokens.add(imports[sub.id])
        if tokens:
            for name in targets:
                consts[name] = frozenset(tokens)
    return consts


def _upc_import_tokens(tree: ast.Module) -> dict[str, str]:
    """Map imported names to the issue token of the ``issue<M>_*`` module
    they come from — a module-name PROXY for constants whose values live
    cross-file (``from issue928_common import FIT_RESULTS_PREFIX`` ->
    ``FIT_RESULTS_PREFIX: "928"``; ``import issue928_common`` binds the
    module name itself)."""
    imports: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            m = _UPC_OWN_ISSUE_RE.match(node.module.rsplit(".", 1)[-1])
            if m:
                for alias in node.names:
                    imports[alias.asname or alias.name] = m.group(1)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                m = _UPC_OWN_ISSUE_RE.match(alias.name.rsplit(".", 1)[-1])
                if m:
                    imports[alias.asname or alias.name.split(".")[0]] = m.group(1)
    return imports


def _upc_argparse_default_tokens(
    tree: ast.Module,
    consts: dict[str, frozenset[str]],
    imports: dict[str, str],
) -> dict[str, tuple[frozenset[str], int]]:
    """Map argparse dest names to ``(issue tokens of the default= expr, the
    add_argument call's lineno)`` — the lineno is the waiver anchor for
    argparse-default findings. Dest = explicit ``dest=`` kwarg, else the
    longest ``--opt`` option string with ``-`` -> ``_``. Only defaults
    resolving (via the module const/import maps) to ≥1 token are recorded."""
    argdefs: dict[str, tuple[frozenset[str], int]] = {}
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_argument"
        ):
            continue
        dest: str | None = None
        default_expr: ast.expr | None = None
        for kw in node.keywords:
            if (
                kw.arg == "dest"
                and isinstance(kw.value, ast.Constant)
                and isinstance(kw.value.value, str)
            ):
                dest = kw.value.value
            elif kw.arg == "default":
                default_expr = kw.value
        if dest is None:
            opts = [
                a.value
                for a in node.args
                if isinstance(a, ast.Constant)
                and isinstance(a.value, str)
                and a.value.startswith("--")
            ]
            if opts:
                dest = max(opts, key=len).lstrip("-").replace("-", "_")
        if dest is None or default_expr is None:
            continue
        tokens = {
            token
            for token, _via, _ln in _upc_resolve(
                default_expr, consts, imports, {}, None, node.lineno
            )
        }
        if not tokens:
            continue
        prev = argdefs.get(dest)
        if prev is not None:
            argdefs[dest] = (frozenset(set(prev[0]) | tokens), prev[1])
        else:
            argdefs[dest] = (frozenset(tokens), node.lineno)
    return argdefs


def _upc_local_assign(
    funcdef: ast.FunctionDef | ast.AsyncFunctionDef, name: str, before_lineno: int
) -> tuple[ast.expr, int] | None:
    """Nearest preceding function-local ``Assign``/``AnnAssign`` to ``name``
    (lineno < ``before_lineno``) inside ``funcdef``; returns
    ``(value expr, assign lineno)``. Resolves the local
    ``path_in_repo = f"{args.hf_prefix}/…"`` shape (the #1005-class
    issue1092_figures.py write, plan §4.6 row 8)."""
    best: tuple[int, ast.expr] | None = None
    for node in ast.walk(funcdef):
        value: ast.expr | None = None
        if (
            isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == name for t in node.targets)
        ) or (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == name
        ):
            value = node.value
        if value is None or node.lineno >= before_lineno:
            continue
        if best is None or node.lineno > best[0]:
            best = (node.lineno, value)
    return (best[1], best[0]) if best else None


def _upc_resolve(  # noqa: C901 -- flat per-AST-node-shape resolution ladder (plan #1452 §4.3.5); extracting a branch would just relocate it
    expr: ast.expr,
    consts: dict[str, frozenset[str]],
    imports: dict[str, str],
    argdefs: dict[str, tuple[frozenset[str], int]],
    funcdef: ast.FunctionDef | ast.AsyncFunctionDef | None,
    use_lineno: int,
    seen: frozenset[str] = frozenset(),
) -> set[tuple[str, str, int | None]]:
    """Resolve a write-destination expression to ``{(token, via, anchor)}``.

    ``via`` ∈ {"direct", "or-fallback", "argparse-default"}; ``anchor`` is
    the ``add_argument`` lineno for argparse-default tokens (the waiver
    anchor per plan §4.3.7), else None. An unresolvable expr returns the
    empty set — a dynamic dest is un-lintable statically (disclosed)."""
    out: set[tuple[str, str, int | None]] = set()
    if isinstance(expr, ast.Constant):
        if isinstance(expr.value, str):
            out.update((t, "direct", None) for t in _UPC_ISSUE_TOKEN_RE.findall(expr.value))
    elif isinstance(expr, ast.JoinedStr):
        for part in expr.values:
            if isinstance(part, ast.Constant) and isinstance(part.value, str):
                out.update((t, "direct", None) for t in _UPC_ISSUE_TOKEN_RE.findall(part.value))
            elif isinstance(part, ast.FormattedValue):
                out |= _upc_resolve(part.value, consts, imports, argdefs, funcdef, use_lineno, seen)
    elif isinstance(expr, ast.BoolOp) and isinstance(expr.op, ast.Or):
        out |= _upc_resolve(expr.values[0], consts, imports, argdefs, funcdef, use_lineno, seen)
        for leg in expr.values[1:]:
            out.update(
                (t, "or-fallback", ln)
                for t, _via, ln in _upc_resolve(
                    leg, consts, imports, argdefs, funcdef, use_lineno, seen
                )
            )
    elif isinstance(expr, ast.IfExp):
        out |= _upc_resolve(expr.body, consts, imports, argdefs, funcdef, use_lineno, seen)
        else_hits = _upc_resolve(expr.orelse, consts, imports, argdefs, funcdef, use_lineno, seen)
        if ast.dump(expr.test) == ast.dump(expr.body):
            # `X if X else CONST` — the ternary spelling of an or-fallback.
            out.update((t, "or-fallback", ln) for t, _via, ln in else_hits)
        else:
            out |= else_hits
    elif isinstance(expr, ast.Name):
        if expr.id in consts:
            out.update((t, "direct", None) for t in consts[expr.id])
        elif expr.id in imports:
            out.add((imports[expr.id], "direct", None))
        elif funcdef is not None and expr.id not in seen:
            hit = _upc_local_assign(funcdef, expr.id, use_lineno)
            if hit is not None:
                value, assign_lineno = hit
                out |= _upc_resolve(
                    value, consts, imports, argdefs, funcdef, assign_lineno, seen | {expr.id}
                )
    elif isinstance(expr, ast.Attribute):
        if expr.attr in argdefs:
            # `args.tensors_upload_prefix` — receiver name unchecked (`args.`
            # by convention; disclosed approximation, waiver escape available).
            tokens, arg_lineno = argdefs[expr.attr]
            out.update((t, "argparse-default", arg_lineno) for t in tokens)
        elif isinstance(expr.value, ast.Name) and expr.value.id in imports:
            # `issue928_common.SOME_PREFIX` module-attribute access.
            out.add((imports[expr.value.id], "direct", None))
    return out


def _upc_fn_name(call: ast.Call) -> str | None:
    """Called-function bare name or attribute tail (``api.upload_folder`` ->
    ``upload_folder``), same convention as :func:`check_upload_as_file`."""
    fn = call.func
    if isinstance(fn, ast.Attribute):
        return fn.attr
    if isinstance(fn, ast.Name):
        return fn.id
    return None


def _upc_dest_exprs(call: ast.Call, specs: set[tuple[str, int | None]]) -> list[ast.expr]:
    """Destination argument expression(s) of ``call`` per the dest specs
    (kwarg first; the positional slot as fallback). A wrapper-name collision
    can contribute several specs — all are checked."""
    exprs: list[ast.expr] = []
    for kwname, pos in sorted(specs, key=lambda s: (s[0], -1 if s[1] is None else s[1])):
        kw = next((k.value for k in call.keywords if k.arg == kwname), None)
        if kw is not None:
            exprs.append(kw)
        elif (
            pos is not None and len(call.args) > pos and not isinstance(call.args[pos], ast.Starred)
        ):
            exprs.append(call.args[pos])
    return exprs


def _upc_param_default(node: ast.FunctionDef | ast.AsyncFunctionDef, param: str) -> ast.expr | None:
    """Signature default expr for ``param`` in ``node``'s signature, or None."""
    pos = node.args.posonlyargs + node.args.args
    defaults = node.args.defaults
    offset = len(pos) - len(defaults)
    for i, a in enumerate(pos):
        if a.arg == param and i >= offset:
            return defaults[i - offset]
    for a, d in zip(node.args.kwonlyargs, node.args.kw_defaults, strict=True):
        if a.arg == param and d is not None:
            return d
    return None


def _upc_calls_with_scope(
    tree: ast.Module,
) -> list[tuple[ast.Call, ast.FunctionDef | ast.AsyncFunctionDef | None]]:
    """Every ``Call`` node paired with its innermost enclosing function
    (None at module level) — the scope for function-local dest resolution."""
    out: list[tuple[ast.Call, ast.FunctionDef | ast.AsyncFunctionDef | None]] = []

    def _walk(node: ast.AST, func: ast.FunctionDef | ast.AsyncFunctionDef | None) -> None:
        for child in ast.iter_child_nodes(node):
            child_func = (
                child if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef) else func
            )
            if isinstance(child, ast.Call):
                out.append((child, func))
            _walk(child, child_func)

    _walk(tree, None)
    return out


def _upc_collect_wrappers(  # noqa: C901 -- flat wrapper-inference scan (plan #1452 §4.3 pass 1); extracting a branch would just relocate it
    files: list[Path],
) -> tuple[dict[str, set[tuple[str, int | None]]], list[tuple[Path, int, str, str]]]:
    """Pass 1 (repo-wide over the walked ``scripts/`` set): infer WRAPPER
    functions whose parameter feeds a base write fn's destination — the
    copied ``issue<N>_common.py`` pattern (#928/#1073), one level deep —
    keyed by bare name; a name collision UNIONS the dest specs. ALSO record
    a WRAPPER-FALLBACK finding at the def line when the dest param carries a
    signature default resolving to an issue token (the remediated-#928
    ``path_in_repo=DECOMP_TENSORS_PREFIX`` shape); the in-body
    ``param or CONST`` fallback is caught by pass 2's normal resolution of
    the wrapper's own file, so it is deliberately NOT re-recorded here.
    Fallback findings are (path, def lineno, token, rule) with waivers
    already consumed; they are recorded only for issue-named files (the
    same pass-2 scope) and classified Rule A (cross-issue token) or Rule B
    (own-issue fallback, allowlistable)."""
    wrappers: dict[str, set[tuple[str, int | None]]] = {}
    findings: list[tuple[Path, int, str, str]] = []
    for py in files:
        text = py.read_text(encoding="utf-8")
        tree = _cached_parse(py, text)
        if tree is None:
            continue
        own_m = _UPC_OWN_ISSUE_RE.match(py.name)
        imports = _upc_import_tokens(tree)
        consts = _upc_module_const_tokens(tree, imports)
        lines = text.splitlines()
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                continue
            pos_params = [a.arg for a in (node.args.posonlyargs + node.args.args)]
            all_params = set(pos_params) | {a.arg for a in node.args.kwonlyargs}
            dest_params: set[str] = set()
            for call in ast.walk(node):
                if not isinstance(call, ast.Call):
                    continue
                fn_name = _upc_fn_name(call)
                if fn_name not in UPLOAD_DEST_FUNCS:
                    continue
                for dexpr in _upc_dest_exprs(call, {UPLOAD_DEST_FUNCS[fn_name]}):
                    cands: list[ast.expr] = [dexpr]
                    if isinstance(dexpr, ast.BoolOp) and isinstance(dexpr.op, ast.Or):
                        cands.append(dexpr.values[0])
                    if isinstance(dexpr, ast.JoinedStr):
                        cands.extend(
                            p.value for p in dexpr.values if isinstance(p, ast.FormattedValue)
                        )
                    for cand in cands:
                        if isinstance(cand, ast.Name) and cand.id in all_params:
                            dest_params.add(cand.id)
            for param in sorted(dest_params):
                idx = pos_params.index(param) if param in pos_params else None
                wrappers.setdefault(node.name, set()).add((param, idx))
                if own_m is None:
                    continue  # same disclosed scope as pass 2 (issue-named files)
                default = _upc_param_default(node, param)
                if default is None:
                    continue
                tokens = {
                    t
                    for t, _v, _ln in _upc_resolve(default, consts, imports, {}, None, node.lineno)
                }
                if not tokens or _upc_waiver_present(lines, node.lineno):
                    continue
                own = own_m.group(1)
                for token in sorted(tokens):
                    findings.append((py, node.lineno, token, "A" if token != own else "B"))
    return wrappers, findings


def _upc_rule_a_message(py: Path, lineno: int, token: str, own: str, via: str) -> str:
    """Rule-A (cross-issue destination) error text."""
    return (
        f"{py}:{lineno}: cross-issue upload destination 'issue{token}_…' (via {via}) in an "
        f"'issue{own}_' script — a reused/copied uploader writing into another issue's HF "
        f"prefix is the #1005 parent-clobber class (reused #928 fitters overwrote the parent's "
        f"artifacts; parent restored from a pinned revision). Thread THIS issue's own upload "
        f"prefix explicitly, or waive a deliberate cross-issue WRITE with "
        f"'# UPLOAD_PREFIX_EXEMPT: <reason>' (reason >= "
        f"{UPLOAD_PREFIX_WAIVER_MIN_REASON_CHARS} chars) on the finding's first line or the "
        f"previous non-blank line. Cross-issue READS (list_repo_tree / hf_hub_download) are "
        f"never flagged; a Rule-A finding is never silently allowlisted."
    )


def _upc_rule_b_message(py: Path, lineno: int, token: str, via: str) -> str:
    """Rule-B (hardcoded same-issue fallback destination) error text."""
    return (
        f"{py}:{lineno}: hardcoded issue-prefix fallback 'issue{token}_…' (via {via}) at an "
        f"upload destination — silently inherited when a child issue reuses this script (the "
        f"#1005 clobber shape: `args.upload_prefix or FIT_RESULTS_PREFIX`). Use default=None + "
        f"a fail-loud raise when uploading without an explicit prefix, or waive with "
        f"'# UPLOAD_PREFIX_EXEMPT: <reason>' (reason >= "
        f"{UPLOAD_PREFIX_WAIVER_MIN_REASON_CHARS} chars) on the finding's first line or the "
        f"previous non-blank line (for an argparse-default finding, at the add_argument call). "
        f"Pre-existing offenders are grandfathered in UPLOAD_PREFIX_CLOBBER_ALLOWLIST — never "
        f"extend it for new code."
    )


def check_upload_prefix_clobber(  # noqa: C901 -- flat two-pass scan + flag-policy ladder (plan #1452 §4.3 pass 2); extracting a branch would just relocate it
    *, scripts_dir: Path | None = None, legacy_allowlist: frozenset[str] | None = None
) -> list[str]:
    """AST-walk every ``*.py`` under ``scripts/`` (two passes) and FAIL on
    hardcoded issue-prefix HF upload DESTINATIONS of the #1005
    parent-clobber class (task #1452).

    Incident: reused #928 fitter scripts uploaded #1005 tensors to hardcoded
    ``issue928_*`` prefixes on ``superkaiba1/explore-persona-space-data``,
    OVERWRITING the parent issue's artifacts (upload-verification FAIL
    2026-07-16; parent restored from a pinned revision). The check
    mechanizes the previously hand-run "parent-prefix clobber gate".

    Scope: files whose basename matches ``issue<N>_*.py`` (the incident
    class is reused issue scripts; generic entrypoints derive prefixes from
    config). Write CALL SITES only — the base table
    :data:`UPLOAD_DEST_FUNCS` (``upload_file`` / ``upload_folder`` /
    ``CommitOperationAdd`` / ``hub._upload`` / ``hub._upload_folder_filtered``
    / ``upload_raw_completions_to_data_repo``, matched on bare name or
    attribute tail) plus ONE level of inferred wrappers (a function whose
    parameter feeds a base write fn's destination — the copied
    ``issue<N>_common.py`` pattern). Cross-issue READS (``list_repo_tree`` /
    ``hf_hub_download`` / ``fetch_pinned_*``) never flag by construction.

    Flag policy (each finding carries file:lineno + token + via):

    * **Rule A (cross-issue):** any resolved destination token
      ``issue<M>_…`` with M != N FAILs — a reused/copied uploader writing
      into another issue's prefix. NEVER silently allowlisted (waiver-only).
    * **Rule B (hardcoded fallback):** an own-issue token arriving via a
      FALLBACK channel — ``x or CONST``, an argparse ``default=`` (read
      through ``args.<dest>`` attributes and function-local assignments),
      a wrapper-param signature default — FAILs: a reusing child silently
      inherits it. Grandfathered pre-existing sites live in
      :data:`UPLOAD_PREFIX_CLOBBER_ALLOWLIST`.
    * A DIRECT own-prefix hardcode (M == N, via=direct) is the sanctioned
      Upload Policy norm and is never flagged.

    Disclosed static under-triggers (v1, by design): kwarg-only dests on
    ``upload_file``/``upload_folder``/``CommitOperationAdd`` (positional
    forms unresolved), wrapper-of-wrapper chains (one inference level),
    dict-threaded dests, dynamic/computed dests, non-issue-named scripts.

    Waiver: ``# UPLOAD_PREFIX_EXEMPT: <reason>`` (reason ≥
    :data:`UPLOAD_PREFIX_WAIVER_MIN_REASON_CHARS` chars) on the finding's
    first physical line or the immediately preceding non-blank line; for an
    argparse-default finding the waiver may sit at the ``add_argument``
    call instead.

    ``scripts_dir`` / ``legacy_allowlist`` are override hooks for unit
    tests; production callers pass None. Allowlist paths are computed
    relative to the WALK ROOT'S PARENT (production paths read
    ``scripts/<name>.py``; same convention as
    :func:`check_hub_dir_filecount_guard`). Bundled into the no-flags
    default run.
    """
    root = scripts_dir if scripts_dir is not None else _REPO_ROOT / "scripts"
    if not root.exists():
        return []
    allow = UPLOAD_PREFIX_CLOBBER_ALLOWLIST if legacy_allowlist is None else legacy_allowlist
    files = [p for p in sorted(root.rglob("*.py")) if p.is_file()]
    wrappers, fallback_findings = _upc_collect_wrappers(files)
    errors: list[str] = []
    for py in files:
        own_m = _UPC_OWN_ISSUE_RE.match(py.name)
        if not own_m:
            continue  # disclosed scope: the incident class is reused issue scripts
        own = own_m.group(1)
        rel = py.relative_to(root.parent).as_posix()
        text = py.read_text(encoding="utf-8")
        tree = _cached_parse(py, text)
        if tree is None:
            # A scripts/ file that does not parse is its own (separate)
            # problem; this check stays silent on it rather than crashing.
            continue
        imports = _upc_import_tokens(tree)
        consts = _upc_module_const_tokens(tree, imports)
        argdefs = _upc_argparse_default_tokens(tree, consts, imports)
        lines = text.splitlines()
        flagged: set[tuple[int, str, str]] = set()
        for call, funcdef in _upc_calls_with_scope(tree):
            fn_name = _upc_fn_name(call)
            if fn_name is None:
                continue
            specs: set[tuple[str, int | None]] = set()
            if fn_name in UPLOAD_DEST_FUNCS:
                specs.add(UPLOAD_DEST_FUNCS[fn_name])
            specs |= wrappers.get(fn_name, set())
            if not specs:
                continue
            for dest in _upc_dest_exprs(call, specs):
                hits = _upc_resolve(dest, consts, imports, argdefs, funcdef, call.lineno)
                for token, via, anchor in sorted(hits, key=lambda h: (h[0], h[1], h[2] or 0)):
                    if token != own:
                        rule = "A"
                    elif via in ("or-fallback", "argparse-default"):
                        rule = "B"
                    else:
                        continue  # sanctioned direct own-prefix hardcode
                    key = (call.lineno, token, via)
                    if key in flagged:
                        continue
                    waiver_linenos = {call.lineno}
                    if via == "argparse-default" and anchor is not None:
                        waiver_linenos.add(anchor)
                    if any(_upc_waiver_present(lines, ln) for ln in waiver_linenos):
                        continue
                    if rule == "B" and rel in allow:
                        continue
                    flagged.add(key)
                    if rule == "A":
                        errors.append(_upc_rule_a_message(py, call.lineno, token, own, via))
                    else:
                        errors.append(_upc_rule_b_message(py, call.lineno, token, via))
    for py, lineno, token, rule in fallback_findings:
        rel = py.relative_to(root.parent).as_posix()
        if rule == "B" and rel in allow:
            continue
        if rule == "A":
            own_m = _UPC_OWN_ISSUE_RE.match(py.name)
            errors.append(
                _upc_rule_a_message(
                    py, lineno, token, own_m.group(1) if own_m else "?", "wrapper-param-default"
                )
            )
        else:
            errors.append(_upc_rule_b_message(py, lineno, token, "wrapper-param-default"))
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


def _module_globs_jsonl(tree: ast.AST) -> bool:
    """Signal (e) gate (#1162): True iff the module contains a
    ``glob``/``rglob``/``iglob`` call (method or bare function) whose pattern
    argument — positional or keyword, plain str constant or an f-string
    constant fragment — mentions ``jsonl`` case-insensitively
    (``tasks_root.glob("*/*/events.jsonl")``, ``root.glob(f"{stem}.jsonl")``).
    """
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Attribute):
            name = node.func.attr
        elif isinstance(node.func, ast.Name):
            name = node.func.id
        else:
            continue
        if name not in JSONL_GLOB_FUNC_NAMES:
            continue
        for arg in [*node.args, *(kw.value for kw in node.keywords)]:
            frags = list(arg.values) if isinstance(arg, ast.JoinedStr) else [arg]
            for frag in frags:
                if (
                    isinstance(frag, ast.Constant)
                    and isinstance(frag.value, str)
                    and JSONL_NAME_TOKEN_RE.search(frag.value)
                ):
                    return True
    return False


def _enclosing_scope_map(tree: ast.AST) -> dict[int, int | None]:
    """``id(node)`` -> ``id()`` of the innermost enclosing
    ``FunctionDef``/``AsyncFunctionDef`` (None = module scope). A def node
    itself belongs to its OUTER scope; its body belongs to it."""
    scope_of: dict[int, int | None] = {id(tree): None}

    def _visit(node: ast.AST, scope: int | None) -> None:
        for child in ast.iter_child_nodes(node):
            scope_of[id(child)] = scope
            if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef):
                _visit(child, id(child))
            else:
                _visit(child, scope)

    _visit(tree, None)
    return scope_of


def _jsonl_assigned_splitlines_ids(tree: ast.AST, text: str) -> dict[int, str]:
    """Signal (f) pre-pass (#1162): map ``id()`` of every ``.splitlines()``
    call whose receiver is a bare ``ast.Name`` assigned EARLIER IN THE SAME
    SCOPE from a ``read_text()``-bearing expression carrying jsonl evidence —
    the RHS source segment matches :data:`JSONL_NAME_TOKEN_RE` OR its chain
    base ``Name`` matches :data:`JSONL_EVENTS_PATH_NAME_RE` (the #1032
    ``ev = events_path.read_text()`` shape) — to a human-readable signal
    label. Deliberately bounded: single-step, same-scope (module top-level is
    one scope; closures/nested defs are separate scopes), no re-assignment
    analysis, ``Assign``/``AnnAssign`` single-``Name`` targets only.
    Performance: ``ast.get_source_segment`` (O(file) per call) runs ONLY for
    assignments whose (scope, name) matches a bare-Name splitlines receiver
    AND whose RHS contains ``read_text``.
    """
    bare_name_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "splitlines"
        and isinstance(node.func.value, ast.Name)
    ]
    if not bare_name_calls:
        return {}
    scope_of = _enclosing_scope_map(tree)
    receivers: dict[tuple[int | None, str], list[tuple[int, int]]] = {}
    for node in bare_name_calls:
        key = (scope_of.get(id(node)), node.func.value.id)  # type: ignore[union-attr]
        receivers.setdefault(key, []).append((node.lineno, id(node)))
    out: dict[int, str] = {}
    for assign in ast.walk(tree):
        if isinstance(assign, ast.Assign):
            if len(assign.targets) != 1 or not isinstance(assign.targets[0], ast.Name):
                continue
            target_name = assign.targets[0].id
            value = assign.value
        elif isinstance(assign, ast.AnnAssign):
            if not isinstance(assign.target, ast.Name) or assign.value is None:
                continue
            target_name = assign.target.id
            value = assign.value
        else:
            continue
        key = (scope_of.get(id(assign)), target_name)
        if key not in receivers:
            continue
        if not _chain_has_read_text(value):
            continue
        segment = ast.get_source_segment(text, value)
        base = _chain_base_name(value)
        if not (
            (segment is not None and JSONL_NAME_TOKEN_RE.search(segment))
            or (base is not None and JSONL_EVENTS_PATH_NAME_RE.match(base))
        ):
            continue
        for call_lineno, call_id in receivers[key]:
            if assign.lineno < call_lineno:
                out[call_id] = f"read_text-assigned jsonl-content receiver ('{target_name}')"
    return out


def _jsonl_splitlines_signal(
    node: ast.Call,
    text: str,
    fn_scoped: set[int],
    assigned: dict[int, str],
    module_globs_jsonl: bool,
) -> str | None:
    """Classify one ``.splitlines()`` call against the six #950/#1162 signals.

    Returns a human-readable signal label when the call reads JSONL content
    (see :func:`check_jsonl_splitlines` for the signal definitions), else
    None. ``assigned`` is the per-file signal-(f) pre-pass map
    (:func:`_jsonl_assigned_splitlines_ids`); ``module_globs_jsonl`` is the
    per-file signal-(e) gate (:func:`_module_globs_jsonl`). Precedence:
    (a)-(d) unchanged, then (f), then (e) — most-specific label first; a node
    matching several signals yields exactly one error. A per-node
    ``ast.get_source_segment(...) is None`` only makes the
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
    if id(node) in assigned:  # (f)
        return assigned[id(node)]
    if module_globs_jsonl and has_read:  # (e) — most generic, last
        return "generic read_text().splitlines() in a *.jsonl-globbing module"
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
    * **(e) glob-gated generic-receiver signal (#1162):** the receiver chain
      contains a ``read_text`` call AND the module contains a
      ``glob``/``rglob``/``iglob`` call (method or bare function) whose
      pattern argument — positional or keyword, plain str constant or an
      f-string constant fragment — mentions ``jsonl`` (the #1132
      ``sweep_parked_wf_candidates.py`` shapes: a ``*.jsonl``-globbing
      module's helpers reading the globbed files through generically-named
      variables/parameters, which evade (a)-(d)).
    * **(f) assignment-tracking signal (#1162):** the receiver is a bare
      ``ast.Name`` assigned earlier in the SAME function scope (module
      top-level counts as one scope) from a ``read_text()``-bearing
      expression whose source segment mentions ``jsonl`` or whose chain base
      matches the events-path regex (the #1032 ``verify_plan.py``
      ``ev = events_path.read_text(...)`` shape, which evades (a)-(e) in a
      non-globbing module). Single-step, same-scope, no re-assignment
      analysis, ``Assign``/``AnnAssign`` single-``Name`` targets only.

    Deliberate false negatives (accepted; the gotchas.md entry + code review
    carry them — each re-affirmed against the 2026-07-09 live enumeration in
    the #1162 plan §4.6):

    1. **Path-variable dataflow in a NON-globbing module**
       (``out_path = d / "x.jsonl"`` … ``out_path.read_text().splitlines()``
       where the module never globs ``*.jsonl``): 16 live sites in 7 files,
       ALL frozen per-issue experiment scripts (zero on the workflow
       surface); adopting the shape would force ~7 allowlist additions plus
       a loosening of the ALLOWLIST_SHAPE_RE hard rule. In a globbing
       module, signal (e) covers the shape.
    2. **Cross-function dataflow in a non-globbing module** (path/text
       passed as a parameter) and **cross-scope assignment** (closures;
       signal (f) is same-scope only).
    3. **Re-assignment blindness** (a false-POSITIVE note, not a negative):
       signal (f) does no kill analysis — a name re-assigned to non-JSONL
       content after an evidenced assignment still flags; the waiver is the
       escape.
    4. **Non-``read_text`` read channels** (all six signals key on
       ``read_text`` in the receiver/RHS chain):
       ``open(path).read().splitlines()`` and
       ``path.read_bytes().decode(...).splitlines()`` evade every signal
       even in a globbing module. (Text-mode ``Path.open()`` / ``open()``
       line ITERATION is SAFE — universal newlines only — and is the
       recommended fix, not an evasion.)
    5. **Shell heredocs** (``.sh`` files are not AST-scannable).

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
            except UnicodeDecodeError as exc:
                # Skip-with-report: never silent, never fatal (syntax validity
                # is ruff/pytest's enforcement job, not this lint's).
                sys.stderr.write(
                    f"workflow_lint: note: --check-jsonl-splitlines skipped "
                    f"unparseable {rel} ({type(exc).__name__})\n"
                )
                continue
            tree = _cached_parse(py, text)
            if tree is None:
                # Parse failure through the shared memo: the skip-note stays
                # NON-SILENT, with a fixed label — the memo returns None only
                # on SyntaxError (its ValueError is inert defense; #1163).
                sys.stderr.write(
                    f"workflow_lint: note: --check-jsonl-splitlines skipped "
                    f"unparseable {rel} (SyntaxError)\n"
                )
                continue
            lines = text.split("\n")
            fn_scoped = _jsonl_fn_scoped_splitlines_ids(tree)
            # Perf gates (#1162 acceptance 9 — the ≤40s no-flags wall budget):
            # the (e)/(f) pre-passes each walk the whole AST, so skip them on
            # files that cannot match. Each gate is a NECESSARY textual
            # condition: a `.splitlines()` attribute call, a `read_text`
            # attribute in an assignment RHS, and a glob/rglob/iglob call
            # with a jsonl-bearing pattern constant all require their literal
            # tokens verbatim in the source text (dynamic getattr shapes are
            # invisible to the AST predicates anyway).
            has_splitlines = "splitlines" in text
            assigned = (
                _jsonl_assigned_splitlines_ids(tree, text)
                if has_splitlines and "read_text" in text
                else {}
            )
            module_globs = (
                _module_globs_jsonl(tree)
                if has_splitlines and "glob" in text and JSONL_NAME_TOKEN_RE.search(text)
                else False
            )
            for node in ast.walk(tree):
                if not (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "splitlines"
                ):
                    continue
                signal = _jsonl_splitlines_signal(node, text, fn_scoped, assigned, module_globs)
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


def _scripts_import_guard_waiver_present(lines: list[str], import_lineno: int) -> bool:
    """Return True iff a ``# SCRIPTS_IMPORT_GUARD_EXEMPT: <reason>`` waiver
    (reason ≥ :data:`SCRIPTS_IMPORT_GUARD_WAIVER_MIN_REASON_CHARS` chars) is
    on the import's first physical line (``import_lineno``, 1-based) or the
    immediately preceding non-blank line. Same convention as
    :func:`_jsonl_splitlines_waiver_present`."""
    idx = import_lineno - 1  # to 0-based
    if 0 <= idx < len(lines):
        m = SCRIPTS_IMPORT_GUARD_WAIVER_RE.search(lines[idx])
        if m and len(m.group(1).strip()) >= SCRIPTS_IMPORT_GUARD_WAIVER_MIN_REASON_CHARS:
            return True
    back = idx - 1
    while back >= 0 and lines[back].strip() == "":
        back -= 1
    if back >= 0:
        m = SCRIPTS_IMPORT_GUARD_WAIVER_RE.search(lines[back])
        if m and len(m.group(1).strip()) >= SCRIPTS_IMPORT_GUARD_WAIVER_MIN_REASON_CHARS:
            return True
    return False


def _is_syspath_guard_call(node: ast.AST) -> bool:
    """True iff ``node`` is a syspath-guard ``ast.Call``: a callee name
    (``Name.id`` or ``Attribute.attr``) matching :data:`SYSPATH_GUARD_NAME_RE`
    (the ``_ensure_repo_root_on_syspath()`` exemplar family), or a literal
    ``sys.path.insert(...)``/``sys.path.append(...)``."""
    if not isinstance(node, ast.Call):
        return False
    f = node.func
    name = f.id if isinstance(f, ast.Name) else (f.attr if isinstance(f, ast.Attribute) else None)
    if name and SYSPATH_GUARD_NAME_RE.search(name):
        return True
    return (
        isinstance(f, ast.Attribute)
        and f.attr in ("insert", "append")
        and isinstance(f.value, ast.Attribute)
        and f.value.attr == "path"
        and isinstance(f.value.value, ast.Name)
        and f.value.value.id == "sys"
    )


# Nodes whose bodies do NOT execute when the enclosing scope's body runs — a
# guard call (or an import) inside one belongs to the NESTED scope, never the
# enclosing one. Shared by offender AND guard detection so the two scans are
# symmetric BY CONSTRUCTION (a flat `tree.body` offender scan paired with a
# pruned guard scan would let try/except-wrapped module imports escape).
_SCOPE_PRUNE_NODES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)


def _scope_stmts(body: list[ast.stmt]) -> list[ast.AST]:
    """Return every AST node that executes at THIS scope when ``body`` runs —
    descending compound statements (``If``/``Try``/``With``/``For``/``While``:
    a try/except-wrapped import, an ``if <flag>:`` import, and the
    ``if __name__ == "__main__":`` main-block shape all execute at this
    scope), but NOT descending into nested
    ``FunctionDef``/``AsyncFunctionDef``/``ClassDef``/``Lambda`` (their
    bodies run only when called — the deferred pass owns function bodies).
    Conditionally-executed nodes are included by PRESENCE (conditions are not
    evaluated — accepted imprecision, documented in
    :func:`check_scripts_import_guard`)."""
    out: list[ast.AST] = []
    stack: list[ast.AST] = list(body)
    while stack:
        node = stack.pop()
        if isinstance(node, _SCOPE_PRUNE_NODES):
            continue
        out.append(node)
        stack.extend(ast.iter_child_nodes(node))
    return out


def _scope_guard_linenos(body: list[ast.stmt]) -> list[int]:
    """Linenos of syspath-guard-evidence Calls in ONE scope body, via the
    SAME pruned walk as offender detection (:func:`_scope_stmts`) — a guard
    call inside a nested def does not execute when the def statement runs."""
    return [n.lineno for n in _scope_stmts(body) if _is_syspath_guard_call(n)]


def _type_checking_body_ranges(tree: ast.Module) -> list[tuple[int, int]]:
    """(start, end) lineno ranges of the BODY of every ``if TYPE_CHECKING:``
    block (``Name("TYPE_CHECKING")`` or ``Attribute(attr="TYPE_CHECKING")``
    test). Body only — the ``orelse`` branch DOES execute at runtime and
    stays in scope; ``if not TYPE_CHECKING:`` does not match the test
    predicate, so its body correctly stays in scope too."""
    ranges: list[tuple[int, int]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        test = node.test
        is_tc = (isinstance(test, ast.Name) and test.id == "TYPE_CHECKING") or (
            isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"
        )
        if is_tc and node.body:
            start = node.body[0].lineno
            end = max((s.end_lineno or s.lineno) for s in node.body)
            ranges.append((start, end))
    return ranges


def _is_scripts_import(node: ast.AST) -> bool:
    """True iff ``node`` imports the repo-root ``scripts`` package: an
    ``ImportFrom`` of ``scripts``/``scripts.*`` at level 0, or an ``Import``
    with any ``scripts``/``scripts.*`` alias. A prefix non-match
    (``scripts_helper``) is NOT a scripts import."""
    if isinstance(node, ast.ImportFrom):
        return (
            node.level == 0
            and node.module is not None
            and (node.module == "scripts" or node.module.startswith("scripts."))
        )
    if isinstance(node, ast.Import):
        return any(a.name == "scripts" or a.name.startswith("scripts.") for a in node.names)
    return False


def check_scripts_import_guard(*, scan_roots: tuple[Path, ...] | None = None) -> list[str]:
    """AST-walk ``src/explore_persona_space/experiments/**/*.py`` and
    ``scripts/**/*.py`` and FAIL any ``scripts.*`` import — deferred
    (function-body) AND module-top-level — lacking a repo-root ``sys.path``
    guard (#823/#853).

    Rationale: in script mode (``python /abs/path/driver.py``),
    ``sys.path[0]`` is the script's OWN directory — not cwd, not the repo
    root — so ``import scripts.*`` from any driver under
    ``src/explore_persona_space/experiments/**`` raises
    ``ModuleNotFoundError`` pod/GCE-side. Deferred instances crash MID-RUN
    after paid GPU phases, and both standard pre-launch checks false-pass
    them: a ``-c``-mode import check puts cwd on ``sys.path``, and GPU-bound
    smoke carve-outs never execute the deferred import (incident #823
    Phase-3, 2026-07-02: ~30 min of paid GCE work lost; the #853 fix was
    documentation-only — the gotchas.md entry "Script mode puts the SCRIPT's
    dir on ``sys.path[0]``"). Top-level imports are flagged too: the trap is
    PATH ABSENCE, not deferral — a deferred-only check would create a
    hoist-evasion that still burns a pod provision cycle at process start.

    Detection — an import node is in scope iff it is an ``ast.ImportFrom``
    with ``level == 0`` and module ``scripts``/``scripts.*``, or an
    ``ast.Import`` with any ``scripts``/``scripts.*`` alias.
    ``if TYPE_CHECKING:`` bodies are skipped (those imports never execute at
    runtime; the ``orelse`` branch stays in scope). A per-file AST-presence
    fast path (#1229) returns early when NO node in the whole module matches
    the import predicate — ``ast.walk`` visits a strict superset of the
    nodes the pruned offender scans visit, so the early return can never
    skip a flaggable file.

    Guard evidence (:func:`_is_syspath_guard_call`): a Call whose callee name
    matches :data:`SYSPATH_GUARD_NAME_RE` (the
    ``_ensure_repo_root_on_syspath()`` run_823.py/run_952.py exemplars), or a
    literal ``sys.path.insert(...)``/``sys.path.append(...)``. Position
    rules:

    * Deferred import: guarded iff guard-evidence exists in the SAME
      innermost enclosing function body at a SMALLER lineno, OR anywhere at
      MODULE scope (the module body executes fully at import time, before
      any post-import function call — the
      ``scripts/issue_331_phase0_panel.py`` module-top-bootstrap
      convention).
    * Module-executing (top-level) import: guarded iff module-scope
      guard-evidence PRECEDES it (the module body executes in order).

    Offender and guard detection share ONE pruned scope walk
    (:func:`_scope_stmts`): module-scope ``If``/``Try``/``With``/``For``/
    ``While`` bodies — including a ``try/except ImportError``-wrapped import
    and the ``if __name__ == "__main__":`` main-block shape — EXECUTE at
    module scope and are IN scope for detection; nested
    defs/classes/lambdas are pruned (the deferred pass owns function
    bodies).

    Deliberately NOT counted as guarded: ``try/except ImportError`` around
    the import (the fallback silently takes the wrong path pod-side — worse
    than the crash; fail-fast rule), and guards in an OUTER-but-not-innermost
    enclosing function (rare; the waiver is the escape). Deliberate false
    negatives (accepted; the gotchas.md prose rule + code-reviewer Step 0.6
    carry them): dynamic imports (``importlib.import_module``,
    ``__import__``, exec-strings), class-body imports (execute at module
    import time but are pruned from both passes), conditionally-executed
    guards counted by presence (conditions are not evaluated),
    ``sys.path += [...]``/slice-assign guard shapes (waiver escape), and
    shell heredocs (``.sh`` files are not AST-scannable).

    Unparseable files (SyntaxError / non-UTF-8) are SKIPPED with a one-line
    stderr notice, never silently (the jsonl-splitlines precedent). Waive a
    genuinely-safe flagged site with
    ``# SCRIPTS_IMPORT_GUARD_EXEMPT: <reason>`` (reason ≥
    :data:`SCRIPTS_IMPORT_GUARD_WAIVER_MIN_REASON_CHARS` chars) on the
    import's first physical line or the immediately preceding non-blank
    line. No legacy allowlist — the live tree is clean.

    ``scan_roots`` is a unit-test override hook; production callers pass
    None and the function walks BOTH
    ``<repo_root>/src/explore_persona_space/experiments`` AND
    ``<repo_root>/scripts`` (widened by #1229: the module-level-guard
    position rule already accepts the scripts/ module-top-bootstrap
    convention, so the original scripts/ exclusion was overly conservative
    and hid genuinely-unguarded scripts/ files — 3 sites in 2 files found
    and fixed at widening time). ``backends/`` remains excluded — covered
    by entrypoint bootstraps, #987. Bundled into the no-flags default run.
    """
    roots = (
        scan_roots
        if scan_roots is not None
        else (
            _REPO_ROOT / "src" / "explore_persona_space" / "experiments",
            _REPO_ROOT / "scripts",
        )
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
            try:
                text = py.read_text(encoding="utf-8")
            except UnicodeDecodeError as exc:
                # Skip-with-report: never silent, never fatal (syntax validity
                # is ruff/pytest's enforcement job, not this lint's).
                sys.stderr.write(
                    f"workflow_lint: note: --check-scripts-import-guard skipped "
                    f"unparseable {rel} ({type(exc).__name__})\n"
                )
                continue
            # Cheap-token perf gate: any static scripts.* import statement
            # must contain the substring "scripts" (multi-alias forms like
            # `import os, scripts.foo` lack the "import scripts" bigram, so
            # gate on the bare token; dynamic importlib shapes are invisible
            # to the AST predicate anyway).
            if "scripts" not in text:
                continue
            tree = _cached_parse(py, text)
            if tree is None:
                sys.stderr.write(
                    f"workflow_lint: note: --check-scripts-import-guard skipped "
                    f"unparseable {rel} (SyntaxError)\n"
                )
                continue
            errors.extend(_scan_scripts_import_guard_tree(py, tree, text.split("\n")))
    return errors


def _scan_scripts_import_guard_tree(py: Path, tree: ast.Module, lines: list[str]) -> list[str]:
    """Scan ONE parsed experiments/** module for unguarded ``scripts.*``
    imports and return the diagnostics — the per-file body of
    :func:`check_scripts_import_guard` (position rules, pruning, waiver
    semantics documented there)."""
    # Fast path (#1229): the scripts/ scan root is ~1,086 files of which <10
    # import scripts.* at all — detect presence with ONE full walk before
    # computing guard/TYPE_CHECKING evidence (~4 walk-equivalents). ast.walk
    # is a SUPERSET of both pruned scans (it descends nested defs, class
    # bodies, and TYPE_CHECKING bodies), so this can never skip a file the
    # full scan would flag.
    if not any(_is_scripts_import(n) for n in ast.walk(tree)):
        return []
    errors: list[str] = []
    module_guards = _scope_guard_linenos(tree.body)
    tc_ranges = _type_checking_body_ranges(tree)

    def _in_tc(lineno: int) -> bool:
        return any(start <= lineno <= end for start, end in tc_ranges)

    def _flag(stmt: ast.AST, *, deferred: bool) -> None:
        if not _scripts_import_guard_waiver_present(lines, stmt.lineno):
            errors.append(_scripts_import_guard_msg(py, stmt, deferred=deferred))

    # Module-executing offenders: the pruned walk descends module-scope
    # If/Try/With/For/While (try/except-wrapped, if-wrapped, and
    # `if __name__ == "__main__":` imports are module-executing); nested
    # defs/classes/lambdas pruned (the deferred pass owns those).
    # Module-scope guard must PRECEDE (the module body executes in order).
    for stmt in _scope_stmts(tree.body):
        if (
            _is_scripts_import(stmt)
            and not _in_tc(stmt.lineno)
            and not any(g < stmt.lineno for g in module_guards)
        ):
            _flag(stmt, deferred=False)

    # Deferred offenders: innermost-function preceding guard OR any
    # module-scope guard. Iterating each function's OWN body via the pruned
    # walk attributes an import inside a nested def to the NESTED def (its
    # innermost scope) — each import is seen once.
    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        fn_guards = _scope_guard_linenos(fn.body)
        for stmt in _scope_stmts(fn.body):
            if (
                _is_scripts_import(stmt)
                and not _in_tc(stmt.lineno)
                and not (any(g < stmt.lineno for g in fn_guards) or module_guards)
            ):
                _flag(stmt, deferred=True)
    return errors


def _scripts_import_guard_msg(py: Path, stmt: ast.AST, *, deferred: bool) -> str:
    """Compose the ``scripts-import-guard`` diagnostic for one flagged
    import (position-specific: deferred vs module-top-level)."""
    kind = "deferred" if deferred else "module-top-level"
    crash = (
        "mid-run at the deferred-import line, after the paid phases"
        if deferred
        else "at process start"
    )
    return (
        f"{py}:{stmt.lineno}: scripts-import-guard: {kind} scripts.* import "
        f"without a repo-root sys.path guard. In script mode "
        f"(python /abs/path/driver.py) sys.path[0] is the script's own dir — "
        f"'scripts' is unimportable pod/GCE-side and this import crashes "
        f'{crash} (#823/#853; .claude/rules/gotchas.md "Script mode puts the '
        f"SCRIPT's dir on sys.path[0]\"). Call _ensure_repo_root_on_syspath() "
        f"immediately before the import (copy the run_823.py exemplar, commit "
        f"14234c9112), or waive a genuinely-safe site with "
        f"'# SCRIPTS_IMPORT_GUARD_EXEMPT: <reason>' (reason ≥ "
        f"{SCRIPTS_IMPORT_GUARD_WAIVER_MIN_REASON_CHARS} chars)."
    )


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
        tree = _cached_parse(py, text)
        if tree is None:
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
            tree = _cached_parse(py, text)
            if tree is None:
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


def _hub_verify_waiver_present(
    lines: list[str],
    call_lineno: int,
    *,
    waiver_re: re.Pattern[str] = HUB_VERIFY_WAIVER_RE,
    min_reason_chars: int = HUB_VERIFY_WAIVER_MIN_REASON_CHARS,
) -> bool:
    """Return True iff a ``# HUB_VERIFY_RETRY_EXEMPT: <reason>`` waiver
    (reason ≥ :data:`HUB_VERIFY_WAIVER_MIN_REASON_CHARS` chars) is on the
    call's first physical line (``call_lineno``, 1-based) or the immediately
    preceding non-blank line. Same convention as
    :func:`_batch_judge_client_waiver_present`. The defaulted keywords keep
    the ``check_hub_verify_retry`` behavior byte-identical;
    ``_list_repo_files_waiver_present`` (#1624) delegates here with its own
    ``waiver_re`` / ``min_reason_chars`` pair."""
    idx = call_lineno - 1  # to 0-based
    if 0 <= idx < len(lines):
        m = waiver_re.search(lines[idx])
        if m and len(m.group(1).strip()) >= min_reason_chars:
            return True
    back = idx - 1
    while back >= 0 and lines[back].strip() == "":
        back -= 1
    if back >= 0:
        m = waiver_re.search(lines[back])
        if m and len(m.group(1).strip()) >= min_reason_chars:
            return True
    return False


HUB_VERIFY_BARE_TARGETS: tuple[str, ...] = ("list_repo_files", "list_repo_tree", "file_exists")


def _hub_verify_bare_hits(
    tree: ast.Module, targets: Collection[str] = HUB_VERIFY_BARE_TARGETS
) -> list[tuple[int, str]]:
    """Return ``(lineno, pattern)`` pairs for bare Hub verify nodes in ``tree``.

    ``targets`` narrows the matched symbol set (default: the full
    :data:`HUB_VERIFY_BARE_TARGETS` tuple — the ``check_hub_verify_retry``
    behavior, unchanged); ``check_bare_list_repo_files`` (#1624) reuses the
    walker with ``targets=LIST_REPO_FILES_TARGETS``. Membership (``in``)
    semantics are identical across the tuple/frozenset argument types.

    Two legs:

    * any **Load-ctx** ``ast.Attribute`` whose ``attr`` is in
      ``targets`` — covers ``api.list_repo_files(...)``,
      ``HfApi().list_repo_tree(...)``, ``hh.file_exists(...)`` under a module
      alias, AND the bare-reference form passed to a retry wrapper /
      ``asyncio.to_thread`` (mirrors :func:`_is_batches_create_attr`, plus
      the ctx gate below);
    * any **Load-ctx** ``ast.Name`` whose id is a name BOUND by a
      ``from huggingface_hub import <target> [as alias]`` — the ImportFrom
      pre-pass builds an asname-aware bound-name map, so both the plain and
      the aliased import forms are caught, while a script-local
      ``def file_exists(...)`` helper (no huggingface_hub import of that
      symbol) is never flagged.

    Store/Del-ctx nodes (assignment/deletion targets — the monkeypatch
    patch/restore shape in self-test code, ``HfApi.list_repo_tree = fake``)
    are exempt on both legs: a binding target never evaluates to a value,
    so it can neither be a call nor a callable reference; the assigned
    VALUE is a separate Load-ctx node scanned independently. On the Name
    leg the same gate exempts rebinding or deleting the imported name
    (``list_repo_files = None``, ``del list_repo_files`` — a deletion is
    not a call). The Load-ctx bare-reference SAVE
    (``orig = HfApi.list_repo_tree``) REMAINS flagged — a saved alias
    later called still storms — and takes the waiver (#1482/#1561).

    Callers dedupe by line (a call form is a single node either way).
    """
    hf_bound: dict[str, str] = {}  # bound name -> canonical target symbol
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("huggingface_hub"):
            for a in node.names:
                if a.name in targets:
                    hf_bound[a.asname or a.name] = a.name
    hits: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr in targets:
            if isinstance(node.ctx, (ast.Store, ast.Del)):
                # Assignment/deletion TARGETS — the monkeypatch patch/RESTORE
                # assignments in self-test code (`HfApi.list_repo_tree = fake`,
                # `del api.file_exists`) — can never be calls and never
                # evaluate to a callable reference: the assigned VALUE is a
                # separate node with its own Load ctx, scanned independently.
                # The Load-ctx SAVE (`orig = HfApi.list_repo_tree`) stays
                # flagged deliberately — a bare alias later called still
                # storms — and takes the waiver (#1482/#1561).
                continue
            hits.append((node.lineno, f".{node.attr}("))
        elif isinstance(node, ast.Name) and node.id in hf_bound:
            if isinstance(node.ctx, (ast.Store, ast.Del)):
                # Rebinding/deleting the imported name is not a call; a
                # wrap-rebind (`lrf = retry(lrf)`) still hits via its RHS
                # Load usage on the same line.
                continue
            hits.append((node.lineno, f"{hf_bound[node.id]}("))
    return hits


def check_hub_verify_retry(*, scripts_dir: Path | None = None) -> list[str]:
    """AST-walk ``scripts/**/*.py`` and FAIL on any bare Hub verify call —
    ``list_repo_files(`` / ``list_repo_tree(`` / ``.file_exists(`` — outside
    the grandfathered legacy set (:data:`HUB_VERIFY_LEGACY_ALLOWLIST`).

    Rationale: huggingface_hub's ``paginate`` retries ONLY 429 on cursor
    pages, so a transient 504 on a listing/probe propagates — in #920 that
    turned a SUCCESSFUL upload's verify leg into a false workload failure.
    #997 hardened the library path (``orchestrate/hub.py``:
    ``verify_repo_paths_uploaded`` — exact-set post-upload verify,
    ``list_hf_files_under_path`` — scoped listing,
    ``list_repo_files_complete`` — full listing, ``retry_transient`` — the
    bounded-retry wrapper), but a NEW per-issue script hand-rolling a bare
    ``api.list_repo_files(...)`` reintroduces the class. This check is the
    mechanical gate on new scripts/ call sites (#1202).

    Detection: see :func:`_hub_verify_bare_hits` (a Load-ctx Attribute leg
    + an asname-aware Load-ctx imported-Name leg; Store/Del-ctx binding
    targets exempt — see :func:`_hub_verify_bare_hits`), deduped by line.
    Compliant hub-helper usage is structurally invisible (different
    attr/name strings); comments and docstrings can never match. Within
    ``scripts/`` ANY spelled bare call is presumed un-retried — even a
    script-local ``retry_transient(lambda: api.list_repo_files(...))`` is
    flagged deliberately (the wrapped bare attribute is still a hand-rolled
    leg); a genuinely-correct raw call takes the waiver comment. A
    monkeypatch SAVE of the bare attribute (``orig = HfApi.list_repo_tree``)
    is likewise flagged deliberately and takes the waiver — the saved alias
    is call-equivalent; the assignment TARGETS of the patch/restore lines
    need no waiver (Store-ctx exempt, #1482/#1561).

    Named residuals NOT covered (deliberate — documented, not detector legs):
    ``repo_info(`` (legitimate metadata uses dominate), ``hf_hub_download``
    (large, often locally-retried caller base), ``HfFileSystem``
    ``.exists()``/``.ls()`` (generic attr names = FP explosion), raw HTTP
    calls on the Hub API, subprocess ``hf`` CLI invocations,
    ``getattr(api, "list_repo_files")`` evasion, and the 9 ``scripts/*.sh``
    files with embedded-heredoc hits (AST cannot parse shell; a regex leg
    would reintroduce comment/string false positives).

    Exempt:
      * files in :data:`HUB_VERIFY_LEGACY_ALLOWLIST` (frozen at the
        2026-07-09 tree; membership exempts the WHOLE file; a NEW file is
        never added — the waiver is the path);
      * any call site waived with ``# HUB_VERIFY_RETRY_EXEMPT: <reason>``
        (reason ≥ :data:`HUB_VERIFY_WAIVER_MIN_REASON_CHARS` chars) on the
        call's first physical line or the immediately preceding non-blank
        line.

    ``scripts_dir`` is an override hook for unit tests; production callers
    pass None and the function walks the canonical ``<repo_root>/scripts``
    tree. Bundled into the no-flags default run (same policy as
    ``check_batch_judge_client``).
    """
    root = scripts_dir if scripts_dir is not None else _REPO_ROOT / "scripts"
    errors: list[str] = []
    if not root.exists():
        return errors
    for py in sorted(root.rglob("*.py")):
        if not py.is_file():
            continue
        try:
            rel = py.resolve().relative_to(_REPO_ROOT.resolve()).as_posix()
        except ValueError:
            # A unit-test fixture tree outside the repo: identify it by
            # its tail under the repo's logical layout instead.
            rel = py.as_posix()
        if rel in HUB_VERIFY_LEGACY_ALLOWLIST:
            continue
        text = py.read_text(encoding="utf-8")
        tree = _cached_parse(py, text)
        if tree is None:
            # A non-parsing file is its own separate problem; stay silent.
            continue
        lines = text.splitlines()
        seen_lines: set[int] = set()
        for lineno, pattern in _hub_verify_bare_hits(tree):
            if lineno in seen_lines:
                continue
            seen_lines.add(lineno)
            if _hub_verify_waiver_present(lines, lineno):
                continue
            errors.append(
                f"{py}:{lineno}: bare Hub verify call ('{pattern}') — un-retried "
                f"listings/probes re-create the #920 false-failure class (a "
                f"transient 504 on a cursor page fails a successful upload's "
                f"verify; huggingface_hub retries only 429 there). Route through "
                f"explore_persona_space.orchestrate.hub: "
                f"verify_repo_paths_uploaded (exact-set post-upload verify), "
                f"list_hf_files_under_path (scoped listing), or "
                f"list_repo_files_complete (full listing). A genuinely-correct "
                f"raw call (e.g. one you wrap in hub.retry_transient yourself) "
                f"takes the waiver: '# HUB_VERIFY_RETRY_EXEMPT: <reason>' "
                f"(reason >= {HUB_VERIFY_WAIVER_MIN_REASON_CHARS} chars) on the "
                f"call's line or the previous non-blank line (#920/#997/#1202). "
                f"Monkeypatch assignment TARGETS (`HfApi.X = fake`) are exempt "
                f"by AST ctx and need no waiver; a monkeypatch SAVE of the bare "
                f"reference (`orig = HfApi.X`) still takes the waiver — a saved "
                f"alias later called still storms (#1482/#1561)."
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


# --- `--check-live-hf-retry-routing` (#1547): bare live HF call sites ------
# huggingface_hub 0.36.2's native `http_backoff` retries only 500/502/503/504
# (never 429) and covers only the download/LFS-PUT paths; the commit API
# (`create_commit`, which `upload_file`/`upload_folder` route through) has NO
# native retry at all — so every bare live call site is a 429
# single-point-of-failure (three same-day 429 kills, 2026-07-18: the #1426
# `upload_folder_scoped_verify` class, the #1335 `ensure_store_local` class,
# and a crash-report download). LIVE code routes each call through
# `hub.retry_transient` / `hub._retry_upload` (Retry-After-aware,
# transient-only — a non-transient 403/permission/404-with-response error
# still raises on the FIRST attempt) or carries an explicit
# `# NO_RETRY: <reason>` waiver on the call line or the line above.
HF_ROUTING_CALL_RE = re.compile(
    r"\bhf_hub_download\s*\(|\.upload_file\s*\(|\.upload_folder\s*\("
    r"|\bcreate_commit\s*\(|\bpush_to_hub\s*\("
)
HF_ROUTING_WRAP_RE = re.compile(r"\b(?:retry_transient|_retry_upload)\s*\(")
# Lines ABOVE the call on which the wrap may open (the lambda-wrap shape).
HF_ROUTING_WRAP_WINDOW = 3
# Scope roots (repo-root-relative). tests/ is deliberately out of scope.
HF_ROUTING_SCOPE_ROOTS = ("scripts", "src/explore_persona_space")
# Files whose predicate matches are pattern STRINGS inside lint/verifier
# checks, not call sites (self-exclusion by constant — #1547 review
# directive 3).
HF_ROUTING_PATTERN_STRING_FILES: frozenset[str] = frozenset(
    {
        "scripts/workflow_lint.py",
        "scripts/verify_plan.py",
    }
)
# Files whose predicate matches live inside GENERATED-CODE string constants
# (the pod-side crash-persist heredocs in backends/gcp.py, which carry their
# own bounded Retry-After-aware retry — #1547 leg C, pinned executable +
# string-shape in tests/test_gcp_backend.py), not importable call sites.
HF_ROUTING_GENERATED_CODE_FILES: frozenset[str] = frozenset(
    {
        "src/explore_persona_space/backends/gcp.py",
    }
)
# SNAPSHOT allowlist of the per-issue historical files present at #1547
# implement time (2026-07-19; the plan §10 audit grep over src/ + scripts/
# minus the live set). These are frozen reproducibility artifacts: no
# retro-fit churn — the routing requirement attaches at REUSE time via the
# artifact-reuse throughput check (i) ("fix the SOURCE module, then reuse";
# `.claude/rules/artifact-reuse.md`). NEWLY-written files — including new
# `scripts/issue<N>_*.py` drivers — are NOT in this snapshot and ARE scanned
# (the `JUDGE_PIN_LEGACY_ALLOWLIST` snapshot idiom: exempt today's tree
# once, gate everything that lands after).
#
# STALENESS RACE (#1568, incident #1547 -> 74bf37250b): this constant is a
# source-frozen artifact, so it can go stale between its generation and the
# round's Step 10d merge gate whenever main churn lands a new offender for
# the CURRENT predicate. Steady state is safe (the check exists on main:
# both gate legs carry it and new bare-call files block at their own
# gates); the race re-opens when a round TIGHTENS this check (predicate /
# window / scope) or introduces a sibling snapshot-based check. Recipe:
# regenerate via `workflow_lint.py --regen-hf-routing-snapshot` on a
# main-synced tree as the LAST pre-gate step, and again on any gate re-run
# after main churn; review the stderr `+` lines — a file YOUR round created
# must be routed, never grandfathered. NOTE: a whole-literal regen paste
# can 3-way-CONFLICT with a concurrent main-side one-line append at the
# gate's #1456 merge of a payload-touched workflow_lint.py — resolve by
# re-running regen on the freshly synced tree, never by hand-merging the
# hunks. Do NOT add dead-entry hygiene (a deleted member's entry is inert;
# enforcing removal would CREATE gate friction on unrelated deletions).
# Keep the FAIL-message text stable while main is red anywhere: the merge
# gate's baseline-vs-gated subtraction compares normalized message LINES,
# so a message rewrite that lands while an offender exists on main would
# false-block as NEW (companion note at the message construction in
# _hf_routing_file_errors).
HF_ROUTING_FROZEN_SNAPSHOT: frozenset[str] = frozenset(
    {
        "scripts/_issue543_common.py",
        "scripts/analyze_length_rate_296.py",
        "scripts/analyze_length_rate_n48.py",
        "scripts/archive/upload_and_clean.py",
        "scripts/dispatch_neg_geometry_504.py",
        "scripts/eval_issue562_panel.py",
        "scripts/eval_marker_spread_source_only.py",
        "scripts/fetch_issue506_phase1_dataset.py",
        "scripts/gen_issue475_scaffold_data.py",
        "scripts/generate_issue356_data.py",
        "scripts/i460_phase23_train.py",
        "scripts/i460_phase2_smoke_check.py",
        "scripts/i460_phase4_eval.py",
        "scripts/i474_phase1_load_R.py",
        "scripts/i474_phase23_train.py",
        "scripts/i474_phase2_smoke_check.py",
        "scripts/i474_phase4_eval.py",
        "scripts/i477_reval_confirm.py",
        "scripts/i488_phase23_train.py",
        "scripts/i488_phase2_smoke_calibrate.py",
        "scripts/i488_phase4_eval_onpolicy.py",
        "scripts/i504_make_figures.py",
        "scripts/i504_reval_confirm.py",
        "scripts/i504_round6_recompute_mean_centered.py",
        "scripts/i549_audit_504.py",
        "scripts/i556_pull_qbank.py",
        "scripts/issue1005_cap16k_launch.py",
        "scripts/issue1024_diagnose_parse_failures.py",
        "scripts/issue1073_common.py",
        "scripts/issue1073_greedy_cloud_distribution.py",
        "scripts/issue1073_mlp_krr_fits.py",
        "scripts/issue1074_aggregate.py",
        "scripts/issue1074_generator_compare.py",
        "scripts/issue1090_free_analysis.py",
        "scripts/issue1090_fu1.py",
        "scripts/issue1090_fu3_worker.py",
        "scripts/issue1090_fu3_yield_replay.py",
        "scripts/issue1090_fu4.py",
        "scripts/issue1090_fu4_text_audit.py",
        "scripts/issue1090_run.py",
        "scripts/issue1092_bridge_refit.py",
        "scripts/issue1092_build_corpus.py",
        "scripts/issue1092_claude_text.py",
        "scripts/issue1092_corpus_dashboard.py",
        "scripts/issue1092_figures.py",
        "scripts/issue1092_fit_grid.py",
        "scripts/issue1092_gpu_phase.py",
        "scripts/issue1092_inline_operator_stage.py",
        "scripts/issue1092_p6_run.py",
        "scripts/issue1092_read4c_repair.py",
        "scripts/issue1092_transfer_probe.py",
        "scripts/issue1108_repo_file_audit.py",
        "scripts/issue1112_dispatch.py",
        "scripts/issue1112_geometry.py",
        "scripts/issue1310_agg_perfold.py",
        "scripts/issue1310_dashboard_stories.py",
        "scripts/issue1315_cjk_audit.py",
        "scripts/issue1315_cjk_audit_rejudge.py",
        "scripts/issue1315_dispatch.py",
        "scripts/issue1315_geometry.py",
        "scripts/issue1315_rejudge_529.py",
        "scripts/issue1332_bank_build.py",
        "scripts/issue1332_common.py",
        "scripts/issue1332_gpu_phase.py",
        "scripts/issue1332_lowdose_gpu.py",
        "scripts/issue1332_lowdose_train.py",
        "scripts/issue1333_dispatch.py",
        "scripts/issue1333_geometry.py",
        "scripts/issue1333_matched_reread_analysis.py",
        "scripts/issue1335_extract_store.py",
        "scripts/issue1335_fit.py",
        "scripts/issue1335_gen.py",
        "scripts/issue1335_refit_companions.py",
        "scripts/issue1335_refit_r0_filters.py",
        "scripts/issue1335_render_rungs.py",
        "scripts/issue1336_dedup_sensitivity.py",
        "scripts/issue1336_diagnose_g1.py",
        "scripts/issue1336_extract_turnstore.py",
        "scripts/issue1336_fit_cells.py",
        "scripts/issue1336_gen_answers.py",
        "scripts/issue1336_recal_verdict.py",
        "scripts/issue1345_common.py",
        "scripts/issue1345_framing_dashboard.py",
        "scripts/issue1345_gen_stories.py",
        "scripts/issue1345_prefetch_reuse.py",
        "scripts/issue1415_disjoint_recount.py",
        "scripts/issue1415_judge.py",
        "scripts/issue1415_map_transport.py",
        "scripts/issue1415_pair_bank.py",
        "scripts/issue1415_run_phase1.py",
        "scripts/issue1434_po_intrusion_audit.py",
        "scripts/issue1481_borderline_bootstrap.py",
        "scripts/issue1481_cjk_audit.py",
        "scripts/issue1481_worker.py",
        "scripts/issue1482_error_analysis.py",
        "scripts/issue1482_g1probe_stage.py",
        "scripts/issue1482_sae.py",
        "scripts/issue458_prep_datasets.py",
        "scripts/issue509_baserate_covariate_earlylayer.py",
        "scripts/issue509_bystander_bootstrap.py",
        "scripts/issue509_pathb_fact_rerun.py",
        "scripts/issue509_top2_scatter_figure.py",
        "scripts/issue511_probe_count_sweep.py",
        "scripts/issue527_dan_rank1_scalar_regression.py",
        "scripts/issue530_logit_reval.py",
        "scripts/issue531_logit_rescore.py",
        "scripts/issue532_followup_logp_slot.py",
        "scripts/issue532_predictor_stress.py",
        "scripts/issue536_recompute_driver.py",
        "scripts/issue540_jsrb_predictor.py",
        "scripts/issue545_metric_race.py",
        "scripts/issue545_sweep.py",
        "scripts/issue545_train_cell.py",
        "scripts/issue545_v2_comparison.py",
        "scripts/issue552_cross_arm_analysis.py",
        "scripts/issue559_base_prior_persona_panel.py",
        "scripts/issue559_cross_behavior_self_scoring.py",
        "scripts/issue559_disjoint_question_followup.py",
        "scripts/issue560_crossrecipe_panel.py",
        "scripts/issue588_smoke_artifact.py",
        "scripts/issue594_analyze_context_geometry.py",
        "scripts/issue594_extract_context_vectors.py",
        "scripts/issue595_prefix_carrier.py",
        "scripts/issue604_adapter_svd.py",
        "scripts/issue604_analyze.py",
        "scripts/issue604_extract_context_vectors.py",
        "scripts/issue617_upload_corpus.py",
        "scripts/issue621_checkpoint_ladder.py",
        "scripts/issue623_persona_resolve.py",
        "scripts/issue634_extract_behavior_vectors.py",
        "scripts/issue634_joint_geometry.py",
        "scripts/issue648_centered_vs_raw_predictive_skill.py",
        "scripts/issue649_extract_panel_earlylayer.py",
        "scripts/issue649_level_change_decomp.py",
        "scripts/issue650_extract_context_bank.py",
        "scripts/issue651_dispatch.py",
        "scripts/issue651_drain_extracts.py",
        "scripts/issue654_fetch_pinned_battery.py",
        "scripts/issue658_common.py",
        "scripts/issue658_extract_base_store.py",
        "scripts/issue658_fit_predictors.py",
        "scripts/issue658_inline_a3_5a_reduce.py",
        "scripts/issue661_analysis.py",
        "scripts/issue661_extract_directions.py",
        "scripts/issue661_freeze_instructions.py",
        "scripts/issue661_generate_arm_a.py",
        "scripts/issue664_aggregate_gate.py",
        "scripts/issue664_build_training_data.py",
        "scripts/issue664_common.py",
        "scripts/issue664_dispatch.py",
        "scripts/issue666_load_store.py",
        "scripts/issue667_alllayer_analysis.py",
        "scripts/issue667_alllayer_dispatch.py",
        "scripts/issue667_analysis.py",
        "scripts/issue667_deltac_probe.py",
        "scripts/issue667_dispatch.py",
        "scripts/issue667_extract.py",
        "scripts/issue667_figures.py",
        "scripts/issue667_pertoken_context_dispatch.py",
        "scripts/issue667_pertoken_dispatch.py",
        "scripts/issue683_build_syco_c_bank.py",
        "scripts/issue683_extract_dv_marker.py",
        "scripts/issue683_extract_dv_sycophancy.py",
        "scripts/issue683_extract_tcb.py",
        "scripts/issue685_assistant_excluded_recompute.py",
        "scripts/issue685_matched_position_u.py",
        "scripts/issue722_extract_fact_rb.py",
        "scripts/issue722_fit_M.py",
        "scripts/issue722_load_activations.py",
        "scripts/issue722_per_position_vC_skill.py",
        "scripts/issue722_regen_ultrachat_generic.py",
        "scripts/issue744_dump_and_stream.py",
        "scripts/issue745_upload_engagement_smoke.py",
        "scripts/issue763_build_probe_pools.py",
        "scripts/issue763_cofit_predictors.py",
        "scripts/issue763_cofit_upload.py",
        "scripts/issue763_common.py",
        "scripts/issue763_disclosure_flag_audit.py",
        "scripts/issue763_extract_pv_rb.py",
        "scripts/issue763_fit_predictors.py",
        "scripts/issue763_judge_e0.py",
        "scripts/issue763_stage_pools.py",
        "scripts/issue763_upload.py",
        "scripts/issue778_v2_prefetch.py",
        "scripts/issue778_v2_upload.py",
        "scripts/issue779_arm_headline_pod.py",
        "scripts/issue779_batch2.py",
        "scripts/issue779_capture_answer_summaries.py",
        "scripts/issue779_capture_answer_summaries_pass2.py",
        "scripts/issue779_collect.py",
        "scripts/issue779_dashboard_completions.py",
        "scripts/issue779_dashboard_corpora.py",
        "scripts/issue779_edges.py",
        "scripts/issue779_extract_rb.py",
        "scripts/issue779_ffc_n1m_fits.py",
        "scripts/issue779_ffc_n1m_generate_capture.py",
        "scripts/issue779_ffc_n50k_fits.py",
        "scripts/issue779_gen_behavior_corpus.py",
        "scripts/issue779_pertoken_lmsys_analysis.py",
        "scripts/issue779_pertoken_lmsys_capture.py",
        "scripts/issue779_pertoken_vs_mean_variance.py",
        "scripts/issue779_reliability_gen_capture.py",
        "scripts/issue779_stage_pass2_vm.py",
        "scripts/issue810_adhoc_crosslayer_pooled.py",
        "scripts/issue810_adhoc_lofo_heatmaps.py",
        "scripts/issue810_adhoc_var_vs_skill.py",
        "scripts/issue810_batch_rejudge_highm.py",
        "scripts/issue810_bootstrap_deltaskill.py",
        "scripts/issue810_common.py",
        "scripts/issue810_extract_positions.py",
        "scripts/issue810_fa_refusal_diagnostics.py",
        "scripts/issue810_fit_readout.py",
        "scripts/issue810_fit_reconstruction.py",
        "scripts/issue810_maxpool_censoring.py",
        "scripts/issue811_mean_parity_check.py",
        "scripts/issue811_offset_decomposition.py",
        "scripts/issue811_upload_store.py",
        "scripts/issue813_rank_spectrum.py",
        "scripts/issue823_identity_baseline.py",
        "scripts/issue825_crossmodel_map_transfer.py",
        "scripts/issue825_dashboard_naturalistic.py",
        "scripts/issue825_map_alignment.py",
        "scripts/issue825_prestage_gen.py",
        "scripts/issue825_reparam_directions.py",
        "scripts/issue833_chain_rho_fixedtext.py",
        "scripts/issue833_chain_rho_nonemit.py",
        "scripts/issue833_extract_onpolicy.py",
        "scripts/issue833_fit_onpolicy.py",
        "scripts/issue841_common.py",
        "scripts/issue841_scaling_capture.py",
        "scripts/issue841_scaling_common.py",
        "scripts/issue920_extract_summaries.py",
        "scripts/issue920_gen_completions_b.py",
        "scripts/issue920_nulls_figures.py",
        "scripts/issue922_common.py",
        "scripts/issue922_fixed_point_slow_modes.py",
        "scripts/issue922_repair_provenance.py",
        "scripts/issue922_slow_shell.py",
        "scripts/issue923_build_inputs.py",
        "scripts/issue923_reduce_spans.py",
        "scripts/issue928_common.py",
        "scripts/issue928_extract_thinking_store.py",
        "scripts/issue928_mlp_indiv_control.py",
        "scripts/issue931_author_blocked_folds.py",
        "scripts/issue931_distance_covariate.py",
        "scripts/issue931_fit_cells.py",
        "scripts/issue931_power_curve_multi_seed.py",
        "scripts/issue931_sep_to_chat_matched_control.py",
        "scripts/issue952_bank_build.py",
        "scripts/issue952_behavior_differs_subset.py",
        "scripts/issue952_china_topup_gpu.py",
        "scripts/issue952_divtrain_build.py",
        "scripts/issue952_divtrain_gpu.py",
        "scripts/issue952_noise_ceiling_gpu.py",
        "scripts/issue952_refusal_sanity.py",
        "scripts/issue958_carried_directions.py",
        "scripts/issue958_common.py",
        "scripts/issue958_dup_excluded_refit.py",
        "scripts/issue958_fit_maps.py",
        "scripts/issue958_long_k1_transfer_lclamp.py",
        "scripts/issue_480/dispatch_marker_480.py",
        "scripts/issue_480/i480_syco_geometry_controls.py",
        "scripts/issue_552_prep_good_corpus.py",
        "scripts/issue_597/analyze_titration_597.py",
        "scripts/issue_597/dispatch_leakage_dynamics_597.py",
        "scripts/issue_597/titration_svd_597.py",
        "scripts/issue_642/i642_analyze.py",
        "scripts/issue_642/i642_dispatch.py",
        "scripts/issue_642/i642_v4_splice_canned_pool.py",
        "scripts/issue_653/i653_postpod_bootstrap.py",
        "scripts/make_issue516_figures.py",
        "scripts/rollup_issue562_panel.py",
        "scripts/run_dose_response_cell.py",
        "scripts/run_experiment_444.py",
        "scripts/run_issue650_preflight.py",
        "scripts/run_issue650_train.py",
        "src/explore_persona_space/analysis/issue685/signed_cosine.py",
        "src/explore_persona_space/experiments/behavior_testbed_545/corpora.py",
        "src/explore_persona_space/experiments/behavior_testbed_545/elicit_v2.py",
        "src/explore_persona_space/experiments/behavior_testbed_545/gates.py",
        "src/explore_persona_space/experiments/contrastive_neg_geometry_530/data_deps.py",
        "src/explore_persona_space/experiments/i460_data.py",
        "src/explore_persona_space/experiments/issue_1072/run_1072.py",
        "src/explore_persona_space/experiments/issue_1072/run_1072_lowdim.py",
        "src/explore_persona_space/experiments/issue_650/__init__.py",
        "src/explore_persona_space/experiments/issue_651/__init__.py",
        "src/explore_persona_space/experiments/issue_653/onpolicy_pool.py",
        "src/explore_persona_space/experiments/issue_823/run_823.py",
        "src/explore_persona_space/experiments/issue_952/run_952.py",
        "src/explore_persona_space/experiments/leave_one_out_505/analyze_expanded.py",
        "src/explore_persona_space/experiments/leave_one_out_505/build_pv_centroids.py",
        "src/explore_persona_space/experiments/leave_one_out_505/dispatch.py",
        "src/explore_persona_space/experiments/leave_one_out_505/logit_rescoring.py",
        "src/explore_persona_space/experiments/marker_implant_480/build_training_pool.py",
        "src/explore_persona_space/experiments/neg_setpoint_601/artifacts.py",
        "src/explore_persona_space/experiments/sycophancy_onpolicy_612/claim_audit.py",
        "src/explore_persona_space/experiments/sycophancy_onpolicy_612/panel_select.py",
        "src/explore_persona_space/experiments/sycophancy_onpolicy_612/prefetch_inputs.py",
    }
)


def _hf_routing_call_is_wrapped(lines: list[str], i: int, match_start: int) -> bool:
    """True iff the HF call at ``lines[i][match_start:]`` rides a
    ``retry_transient`` / ``_retry_upload`` wrap.

    Anchors to the CALL'S OWN wrap, not mere window proximity (#1547 review
    directive 2): a same-line wrap counts only when it opens BEFORE the call
    (``retry_transient(lambda: api.upload_file(...))``); an above-line wrap
    counts only while the expression it opened is still OPEN at line ``i`` —
    the net ``(``-minus-``)`` balance of the lines from the wrap line through
    ``i - 1`` stays positive — so a WRAPPED sibling call one line up can
    never launder a bare call in the same window (a complete wrapped
    statement nets to zero parens).
    """
    for m in HF_ROUTING_WRAP_RE.finditer(lines[i]):
        if m.start() < match_start:
            return True
    bal = 0
    for j in range(i - 1, max(-1, i - 1 - HF_ROUTING_WRAP_WINDOW), -1):
        bal += lines[j].count("(") - lines[j].count(")")
        if bal > 0 and HF_ROUTING_WRAP_RE.search(lines[j]):
            return True
    return False


def _hf_routing_scan_files(root: Path):
    """Yield ``(path, rel)`` for every scanned candidate under
    :data:`HF_ROUTING_SCOPE_ROOTS`, excluding the pattern-string +
    generated-code constants (NOT the frozen snapshot — callers apply that:
    the check exempts it, the regen flag deliberately re-derives it, #1568).
    """
    for scope in HF_ROUTING_SCOPE_ROOTS:
        base = root / scope
        if not base.exists():
            continue
        for py in sorted(base.rglob("*.py")):
            if not py.is_file() or "__pycache__" in py.parts:
                continue
            rel = py.relative_to(root).as_posix()
            if rel in HF_ROUTING_PATTERN_STRING_FILES or rel in HF_ROUTING_GENERATED_CODE_FILES:
                continue
            yield py, rel


def _hf_routing_file_errors(py: Path, rel: str) -> list[str]:
    """Per-file scan body shared by :func:`check_live_hf_retry_routing`
    (verdict) and :func:`regen_hf_routing_snapshot` (offender enumeration).
    Snapshot-BLIND — the caller decides whether the frozen snapshot exempts
    ``rel`` (#1568). Returns one error line per bare (unwrapped, unwaived)
    HF Hub call.
    """
    errors: list[str] = []
    lines = py.read_text(encoding="utf-8").splitlines()
    for i, line in enumerate(lines):
        stripped = line.lstrip()
        if stripped.startswith("#") or stripped.startswith("what="):
            continue
        m = HF_ROUTING_CALL_RE.search(line)
        if m is None:
            continue
        if "# NO_RETRY:" in line or (i > 0 and "# NO_RETRY:" in lines[i - 1]):
            continue
        if _hf_routing_call_is_wrapped(lines, i, m.start()):
            continue
        # Message-edit hazard: the Step 10d merge gate compares normalized
        # message LINES (baseline vs gated legs), so rewording this error
        # while ANY offender exists on main would false-register as NEW and
        # block an unrelated merge — see the STALENESS RACE comment on
        # HF_ROUTING_FROZEN_SNAPSHOT before editing this string (#1568).
        errors.append(
            f"[live-hf-retry-routing] {rel}:{i + 1}: bare HF Hub "
            f"call in LIVE code — route through hub.retry_transient, waive with "
            f"`# NO_RETRY: <reason>`, or (pre-existing file this round never "
            f"touched — snapshot staleness, #1568) regen on a main-synced tree: "
            f"`workflow_lint.py --regen-hf-routing-snapshot`: {stripped[:100]}"
        )
    return errors


def check_live_hf_retry_routing(*, repo_root: Path | None = None) -> list[str]:
    """Walk ``scripts/**/*.py`` + ``src/explore_persona_space/**/*.py`` and
    FAIL on a bare (un-retried) HuggingFace Hub mutation/download call in
    LIVE code (#1547).

    A non-comment line matching :data:`HF_ROUTING_CALL_RE`
    (``hf_hub_download(`` / ``.upload_file(`` / ``.upload_folder(`` /
    ``create_commit(`` / ``push_to_hub(``) flags UNLESS:

    * the call rides ``retry_transient`` / ``_retry_upload`` — on the same
      line (wrap opening BEFORE the call) or within
      :data:`HF_ROUTING_WRAP_WINDOW` lines above with the wrap's expression
      still open at the call line (:func:`_hf_routing_call_is_wrapped`); or
    * a ``# NO_RETRY: <reason>`` waiver sits on the line or the line above; or
    * the line is the wrap idiom's own ``what=`` descriptor kwarg
      (``what=f"hf_hub_download({repo}/{path})"``); or
    * the file is in :data:`HF_ROUTING_FROZEN_SNAPSHOT` (the per-issue
      historical files frozen at #1547 landing time — the routing
      requirement attaches at REUSE time via artifact-reuse check (i)),
      :data:`HF_ROUTING_PATTERN_STRING_FILES` (this file + verify_plan.py:
      pattern strings, not calls), or
      :data:`HF_ROUTING_GENERATED_CODE_FILES` (backends/gcp.py: pod-side
      heredoc string constants carrying their own bounded retry — leg C).

    Scope boundary (deliberate): bare ``snapshot_download`` /
    ``list_repo_files`` sites are OUT of the predicate — a 429 there is not
    a gap in THIS check (see `.claude/rules/upload-policy.md` § Fleet-shared
    commit budget). ``repo_root`` is a unit-test override hook; production
    callers pass None. Bundled into the no-flags default run. Snapshot
    staleness (the check fires on a pre-existing file the round never
    touched): regenerate on a main-synced tree via
    ``--regen-hf-routing-snapshot`` (#1568).
    """
    root = repo_root if repo_root is not None else _REPO_ROOT
    errors: list[str] = []
    for py, rel in _hf_routing_scan_files(root):
        if rel in HF_ROUTING_FROZEN_SNAPSHOT:
            continue
        errors.extend(_hf_routing_file_errors(py, rel))
    return errors


def regen_hf_routing_snapshot(*, repo_root: Path | None = None) -> int:
    """Maintenance (#1568): print the ready-to-paste
    ``HF_ROUTING_FROZEN_SNAPSHOT`` literal for the CURRENT tree — every
    scanned file carrying >=1 bare (unwrapped, unwaived) HF Hub call,
    re-derived snapshot-blind — plus a +/- diff summary vs the compiled-in
    constant on stderr. Run on a MAIN-SYNCED tree (repo root on current
    main, or the worktree after a rebase onto origin/main). REVIEW the
    stderr ``+`` lines before pasting: a file YOUR round created must be
    ROUTED through hub.retry_transient (or NO_RETRY-waived), never
    grandfathered — the stderr summary exists so the round's code review
    sees the delta. Not part of the no-flags bundle; early-dispatched in
    ``main()``. Returns 0 (the paste-ready literal is the product, not a
    verdict).
    """
    root = repo_root if repo_root is not None else _REPO_ROOT
    offenders = sorted(
        rel for py, rel in _hf_routing_scan_files(root) if _hf_routing_file_errors(py, rel)
    )
    sys.stdout.write("HF_ROUTING_FROZEN_SNAPSHOT: frozenset[str] = frozenset(\n    {\n")
    for rel in offenders:
        sys.stdout.write(f'        "{rel}",\n')
    sys.stdout.write("    }\n)\n")
    added = sorted(set(offenders) - HF_ROUTING_FROZEN_SNAPSHOT)
    removed = sorted(HF_ROUTING_FROZEN_SNAPSHOT - set(offenders))
    sys.stderr.write(
        f"# regen vs compiled-in constant: +{len(added)} added, -{len(removed)} removed\n"
    )
    for rel in added:
        sys.stderr.write(f"# + {rel}  (NEW offender — route it if this round created it)\n")
    for rel in removed:
        sys.stderr.write(f"# - {rel}  (no longer flags: deleted, routed, or waived)\n")
    return 0


# --- `--check-bare-list-repo-files` (#1624): data-repo full-listing wedge ---
# hub 0.36.2's HfApi.list_repo_files has NO scoping parameter — its body IS
# an unscoped list_repo_tree(recursive=True) full-tree walk — so EVERY call
# is a full listing; against the ~1M-file data repo it wedges (>90 s #833,
# >600 s #920; two listing-probe kills 2026-07-22 -> #1624). Retry does NOT
# fix this class (the walk grinds, it does not raise) — ORTHOGONAL to
# --check-hub-verify-retry (#1202: transient-retry property, scripts/ only)
# and to --check-live-hf-retry-routing (#1547: excludes list_repo_files by
# stated scope boundary). Scoped recipes: hub.list_hf_files_under_path /
# hub.verify_repo_paths_uploaded / api.list_repo_tree(path_in_repo=...) /
# api.file_exists (single-path probe) / list_repo_files_complete(
# path_in_repo=...). Detection: AST via _hub_verify_bare_hits(targets=...)
# — comments/docstrings/f-strings can never match; Store/Del monkeypatch
# targets exempt; asname-aware imported-Name leg. Named residual:
# never-committed ad-hoc probes (inline `python -c` one-liners — the shape
# of one of the 2026-07-22 kills) are structurally outside ANY file lint;
# this check covers the committed-code subclass only.
LIST_REPO_FILES_TARGETS: frozenset[str] = frozenset({"list_repo_files"})
LIST_REPO_FILES_WAIVER_RE = re.compile(r"#\s*LIST_REPO_FILES_EXEMPT\s*:\s*(.+?)\s*$")
LIST_REPO_FILES_WAIVER_MIN_REASON_CHARS = 10
# SNAPSHOT allowlist of files with >=1 AST hit at #1624 implement time
# (2026-07-23; regen: --regen-list-repo-files-snapshot). File-grain
# membership exempts the WHOLE file (the #1547/#1202 accepted trade-off;
# the scoping requirement re-attaches at REUSE time via artifact-reuse
# check (i)); migrating a file onto the hub helpers -> DROP its entry.
#
# STALENESS RACE (#1568, incident #1547 -> 74bf37250b): this constant is a
# source-frozen artifact, so it can go stale between its generation and the
# round's Step 10d merge gate whenever main churn lands a new offender for
# the CURRENT predicate. Steady state is safe (the check exists on main:
# both gate legs carry it and new bare-call files block at their own
# gates); the race re-opens when a round TIGHTENS this check (predicate /
# scope) or introduces a sibling snapshot-based check. Recipe: regenerate
# via `workflow_lint.py --regen-list-repo-files-snapshot` on a main-synced
# tree as the LAST pre-gate step, and again on any gate re-run after main
# churn; review the stderr `+` lines — a file YOUR round created must be
# SCOPED through the hub helpers (or LIST_REPO_FILES_EXEMPT-waived), never
# grandfathered. NOTE: a whole-literal regen paste can 3-way-CONFLICT with
# a concurrent main-side one-line append at the gate's merge of a
# payload-touched workflow_lint.py — resolve by re-running regen on the
# freshly synced tree, never by hand-merging the hunks. Do NOT add
# dead-entry hygiene (a deleted member's entry is inert; enforcing removal
# would CREATE gate friction on unrelated deletions). Keep the FAIL-message
# text stable while an offender exists on main: the merge gate's
# baseline-vs-gated subtraction compares normalized message LINES, so a
# message rewrite that lands while an offender exists on main would
# false-block as NEW (companion note at the message construction in
# _bare_list_repo_files_file_errors).
LIST_REPO_FILES_FROZEN_SNAPSHOT: frozenset[str] = frozenset(
    {
        "scripts/dispatch_factor_screen_365.py",
        "scripts/dispatch_neg_geometry_504.py",
        "scripts/i474_check_adapter_hf_presence.py",
        "scripts/i474_phase0_preflight.py",
        "scripts/i477_reval_confirm.py",
        "scripts/i488_phase3_train_sweep.py",
        "scripts/i504_reval_confirm.py",
        "scripts/i528_phase23_train.py",
        "scripts/i556_pull_qbank.py",
        "scripts/i601_run_cell.py",
        "scripts/i650_write_results_sentinel.py",
        "scripts/issue530_logit_reval.py",
        "scripts/issue540_jsrb_predictor.py",
        "scripts/issue541_geometry_extract.py",
        "scripts/issue545_sweep.py",
        "scripts/issue545_train_cell.py",
        "scripts/issue594_analyze_context_geometry.py",
        "scripts/issue594_extract_context_vectors.py",
        "scripts/issue604_adapter_svd.py",
        "scripts/issue604_extract_context_vectors.py",
        "scripts/issue617_upload_corpus.py",
        "scripts/issue621_checkpoint_ladder.py",
        "scripts/issue634_extract_behavior_vectors.py",
        "scripts/issue634_joint_geometry.py",
        "scripts/issue651_dispatch.py",
        "scripts/issue651_drain_extracts.py",
        "scripts/issue654_fetch_pinned_battery.py",
        "scripts/issue658_extract_base_store.py",
        "scripts/issue658_fit_predictors.py",
        "scripts/issue661_analysis.py",
        "scripts/issue661_extract_directions.py",
        "scripts/issue661_generate_arm_a.py",
        "scripts/issue664_dispatch.py",
        "scripts/issue666_load_store.py",
        "scripts/issue666_predictor.py",
        "scripts/issue667_alllayer_dispatch.py",
        "scripts/issue667_dispatch.py",
        "scripts/issue667_pertoken_context_dispatch.py",
        "scripts/issue667_pertoken_dispatch.py",
        "scripts/issue685_matched_position_u.py",
        "scripts/issue722_extract_fact_rb.py",
        "scripts/issue722_fit_M.py",
        "scripts/issue722_per_position_vC_skill.py",
        "scripts/issue734_dispatch.py",
        "scripts/issue744_dump_and_stream.py",
        "scripts/issue763_build_probe_pools.py",
        "scripts/issue763_disclosure_flag_audit.py",
        "scripts/issue763_judge_e0.py",
        "scripts/issue763_upload.py",
        "scripts/issue779_capture_answer_summaries.py",
        "scripts/issue779_capture_answer_summaries_pass2.py",
        "scripts/issue779_collect.py",
        "scripts/issue779_extract_rb.py",
        "scripts/issue779_gen_behavior_corpus.py",
        "scripts/issue779_pertoken_vs_mean_variance.py",
        "scripts/issue810_extract_positions.py",
        "scripts/issue811_upload_store.py",
        "scripts/issue833_extract_onpolicy.py",
        "scripts/issue841_common.py",
        "scripts/issue841_scaling_common.py",
        "scripts/issue920_extract_summaries.py",
        "scripts/issue920_gen_completions_b.py",
        "scripts/issue920_nulls_figures.py",
        "scripts/issue923_capture.py",
        "scripts/issue_552_prep_good_corpus.py",
        "scripts/issue_597/dispatch_leakage_dynamics_597.py",
        "scripts/issue_597/titration_svd_597.py",
        "scripts/issue_642/i642_dispatch.py",
        "scripts/issue_653/i653_postpod_bootstrap.py",
        "scripts/measure_cot_entropy.py",
        "scripts/run_issue506_install.py",
        "src/explore_persona_space/experiments/behavior_testbed_545/corpora.py",
        "src/explore_persona_space/experiments/contrastive_neg_geometry_472/train_cell.py",
        "src/explore_persona_space/experiments/issue_823/run_823.py",
        "src/explore_persona_space/experiments/leave_one_out_505/build_pv_centroids.py",
        "src/explore_persona_space/experiments/leave_one_out_505/dispatch_logit_rescoring.py",
        "src/explore_persona_space/experiments/leave_one_out_505/logit_rescoring.py",
        "src/explore_persona_space/experiments/sycophancy_onpolicy_612/claim_audit.py",
        "src/explore_persona_space/experiments/sycophancy_onpolicy_612/panel_select.py",
        "src/explore_persona_space/experiments/sycophancy_onpolicy_612/prefetch_inputs.py",
    }
)


def _list_repo_files_waiver_present(lines: list[str], call_lineno: int) -> bool:
    """``# LIST_REPO_FILES_EXEMPT: <reason>`` waiver (reason >=
    :data:`LIST_REPO_FILES_WAIVER_MIN_REASON_CHARS` chars) on the hit line
    or the immediately preceding non-blank line — the HUB_VERIFY convention
    (delegates to the parametrized :func:`_hub_verify_waiver_present`)."""
    return _hub_verify_waiver_present(
        lines,
        call_lineno,
        waiver_re=LIST_REPO_FILES_WAIVER_RE,
        min_reason_chars=LIST_REPO_FILES_WAIVER_MIN_REASON_CHARS,
    )


def _list_repo_files_scan_files(root: Path) -> Iterator[tuple[Path, str]]:
    """Yield ``(path, rel)`` for every scanned candidate under
    :data:`HF_ROUTING_SCOPE_ROOTS`. Unlike :func:`_hf_routing_scan_files`
    there are NO pattern-string / generated-code exclusion constants: the
    AST predicate makes string/comment/docstring mentions structurally
    unmatchable, so the lint file itself scans clean (#1624 plan §4c). The
    frozen snapshot is applied by the CHECK caller only — the regen flag
    deliberately re-derives it (#1568)."""
    for scope in HF_ROUTING_SCOPE_ROOTS:
        base = root / scope
        if not base.exists():
            continue
        for py in sorted(base.rglob("*.py")):
            if not py.is_file() or "__pycache__" in py.parts:
                continue
            yield py, py.relative_to(root).as_posix()


def _bare_list_repo_files_file_errors(py: Path, rel: str) -> list[str]:
    """Snapshot-BLIND per-file scan body shared by
    :func:`check_bare_list_repo_files` (verdict) and
    :func:`regen_list_repo_files_snapshot` (offender enumeration) — the
    #1568 idiom. Returns one error line per bare (un-waived)
    ``list_repo_files`` Load-ctx call/reference, deduped by line."""
    text = py.read_text(encoding="utf-8")
    tree = _cached_parse(py, text)
    if tree is None:
        # A non-parsing file is its own separate problem; stay silent.
        return []
    lines = text.splitlines()
    errors: list[str] = []
    seen: set[int] = set()
    for lineno, pattern in _hub_verify_bare_hits(tree, targets=LIST_REPO_FILES_TARGETS):
        if lineno in seen:
            continue
        seen.add(lineno)
        if _list_repo_files_waiver_present(lines, lineno):
            continue
        # Message-edit hazard: the Step 10d merge gate compares normalized
        # message LINES (baseline vs gated legs), so rewording this error
        # while ANY offender exists on main would false-register as NEW and
        # block an unrelated merge — see the STALENESS RACE comment on
        # LIST_REPO_FILES_FROZEN_SNAPSHOT before editing this string (#1568).
        errors.append(
            f"[bare-list-repo-files] {rel}:{lineno}: bare list_repo_files call "
            f"('{pattern}') — hub 0.36.2 has NO scoping parameter here: every "
            f"call is an unscoped full-tree walk, which WEDGES on the ~1M-file "
            f"data repo (>90 s #833, >600 s #920; two kills 2026-07-22, #1624) "
            f"and retry cannot save it (the walk grinds, it does not raise). "
            f"Use the scoped recipes: hub.list_hf_files_under_path(api, repo, "
            f"prefix) / hub.verify_repo_paths_uploaded(...) (exact-set "
            f"post-upload verify) / api.list_repo_tree(repo, "
            f"path_in_repo=<prefix>, recursive=True) / api.file_exists(repo, "
            f"path) (single-path probe). A genuinely-correct full listing of a "
            f"SMALL repo takes the waiver '# LIST_REPO_FILES_EXEMPT: <reason>' "
            f"(reason >= {LIST_REPO_FILES_WAIVER_MIN_REASON_CHARS} chars) on "
            f"the call's line or the previous non-blank line. Pre-existing "
            f"file this round never touched? Snapshot staleness — regen on a "
            f"main-synced tree: `workflow_lint.py "
            f"--regen-list-repo-files-snapshot` (#1624)."
        )
    return errors


def check_bare_list_repo_files(*, repo_root: Path | None = None) -> list[str]:
    """AST-walk ``scripts/**/*.py`` + ``src/explore_persona_space/**/*.py``
    (:data:`HF_ROUTING_SCOPE_ROOTS`) and FAIL on any bare
    ``list_repo_files`` call/reference outside
    :data:`LIST_REPO_FILES_FROZEN_SNAPSHOT` and un-waived (#1624).

    Detection is :func:`_hub_verify_bare_hits` narrowed to
    :data:`LIST_REPO_FILES_TARGETS`: a Load-ctx ``.list_repo_files(``
    Attribute under ANY receiver, or a Load-ctx Name bound by
    ``from huggingface_hub import list_repo_files [as alias]``; Store/Del
    monkeypatch targets are exempt; comments / docstrings / f-string
    mentions are structurally unmatchable (no pattern-string exclusion
    constants needed — the lint file itself scans clean). Deliberately
    inherited semantics: a ``retry_transient(lambda: api.list_repo_files(``
    wrap STILL flags (retry != scoping — the wedge grinds, it does not
    raise), and a Load-ctx monkeypatch SAVE (``orig = HfApi.list_repo_files``)
    flags while the Store/Del patch/restore targets do not (#1482/#1561).

    ``tests/`` is out of scope (mocks legitimately spell the name). Named
    residuals NOT covered (documented, not detector legs): never-committed
    ad-hoc probes (inline ``python -c`` one-liners) are structurally outside
    ANY file lint — this check covers the committed-code subclass; unscoped
    ``list_repo_files_complete(...)`` / ``list_repo_tree(recursive=True)``
    calls without ``path_in_repo`` (kwarg-presence analysis on the helpers =
    high-FP; their own docstrings + the #833 gotcha govern);
    ``snapshot_download``; ``getattr(api, "list_repo_files")`` evasion;
    ``.sh`` heredocs; ``HfFileSystem.ls()``.

    ``repo_root`` is a unit-test override hook; production callers pass
    None. Bundled into the no-flags default run. Snapshot staleness (the
    check fires on a pre-existing file the round never touched): regenerate
    on a main-synced tree via ``--regen-list-repo-files-snapshot`` (#1568
    recipe).
    """
    root = repo_root if repo_root is not None else _REPO_ROOT
    errors: list[str] = []
    for py, rel in _list_repo_files_scan_files(root):
        if rel in LIST_REPO_FILES_FROZEN_SNAPSHOT:
            continue
        errors.extend(_bare_list_repo_files_file_errors(py, rel))
    return errors


def regen_list_repo_files_snapshot(*, repo_root: Path | None = None) -> int:
    """MAINTENANCE (#1624): print the ready-to-paste
    ``LIST_REPO_FILES_FROZEN_SNAPSHOT`` literal for the CURRENT tree — every
    scanned file carrying >=1 bare (un-waived) ``list_repo_files`` AST hit,
    re-derived snapshot-blind — plus a +/- diff summary vs the compiled-in
    constant on stderr (the :func:`regen_hf_routing_snapshot` idiom, #1568).
    Run on a MAIN-SYNCED tree. REVIEW the stderr ``+`` lines before pasting:
    a file YOUR round created must be SCOPED through the hub helpers (or
    waived), never grandfathered. Not part of the no-flags bundle;
    early-dispatched in ``main()``. Returns 0 (the paste-ready literal is
    the product, not a verdict).
    """
    root = repo_root if repo_root is not None else _REPO_ROOT
    offenders = sorted(
        rel
        for py, rel in _list_repo_files_scan_files(root)
        if _bare_list_repo_files_file_errors(py, rel)
    )
    sys.stdout.write("LIST_REPO_FILES_FROZEN_SNAPSHOT: frozenset[str] = frozenset(\n    {\n")
    for rel in offenders:
        sys.stdout.write(f'        "{rel}",\n')
    sys.stdout.write("    }\n)\n")
    added = sorted(set(offenders) - LIST_REPO_FILES_FROZEN_SNAPSHOT)
    removed = sorted(LIST_REPO_FILES_FROZEN_SNAPSHOT - set(offenders))
    sys.stderr.write(
        f"# regen vs compiled-in constant: +{len(added)} added, -{len(removed)} removed\n"
    )
    for rel in added:
        sys.stderr.write(f"# + {rel}  (NEW offender — scope/waive it if this round created it)\n")
    for rel in removed:
        sys.stderr.write(f"# - {rel}  (no longer flags: deleted, scoped, or waived)\n")
    return 0


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
    tree = _cached_parse(target, text)
    if tree is None:
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
        # Pre-enumeration pruning (#1163): the old `claude_dir.rglob("*")`
        # enumerated the repo root's 3.3M-entry `.claude/worktrees/` tree
        # (145s) only for the string filters below to discard it. The pruned
        # walk never descends there; the per-file filters stay in place as
        # the behavioral contract.
        for p in _iter_files_pruned(
            claude_dir, suffixes=frozenset({".md", ".yaml", ".yml", ".py", ".sh"})
        ):
            # Root-RELATIVE match (not an absolute-path substring): keeps the
            # check hermetic when the repo root itself is nested under a real
            # .claude/cache/ (repo-nested TMPDIR, #1174). relative_to(root) is
            # always valid here — the walk yields paths prefixed by the
            # claude_dir it was called with.
            rel = p.relative_to(root).as_posix()
            if rel.startswith(".claude/cache/") or rel.startswith(".claude/agent-memory/"):
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


# --- Executable-git-recipe root-guard check (#1176) --------------------------
#
# Feed every bash-fenced recipe in the workflow docs to the LIVE PreToolUse
# hook (scripts/guard_repo_root_branch.sh), exactly as a session pasting the
# whole block into ONE Bash call would hit it. Incident #1047: a documented
# cleanup recipe without a per-clause `git -C` waiver survived plan review,
# fact-check, and a 6-critic ensemble — only the code-reviewer EXECUTING the
# hook caught it. Executing the real hook (instead of a parallel
# pattern-match) keeps the check current as the hook's detectors evolve.
_ROOT_GUARD_HOOK = _HERE / "guard_repo_root_branch.sh"
_ROOT_GUARD_EXEMPT_SENTINEL = "workflow-lint: allow-root-guard-block"
# A fence line is ```<tag> / ~~~<tag> with a SINGLE-token (or empty) info
# string; a fence line with extra words after the tag does not match (zero
# live in-scope instances; a documented residual of the parser rule).
_ROOT_GUARD_FENCE_RE = re.compile(r"^\s*(```|~~~)(\S*)\s*$")
_ROOT_GUARD_BASH_TAGS = frozenset({"bash", "sh", "shell"})
_ROOT_GUARD_TIMEOUT_S = 20
# Python string data only: the hook gates Bash TOOL calls, not file contents,
# and neither sibling doc-scanning check scans scripts/** (their scope is
# agents+skills md only), so these constants cannot self-flag.
_ROOT_GUARD_SELFTEST_BLOCKED = "git checkout -b __workflow_lint_root_guard_selftest__"
_ROOT_GUARD_SELFTEST_BENIGN = "echo workflow-lint root-guard selftest"


def _root_guard_git_env(git_fixture: Path) -> dict[str, str]:
    """Subprocess env for root-guard hook probes: every ambient ``GIT_*``
    var is SCRUBBED (a git-hook / pre-commit caller exports GIT_DIR /
    GIT_INDEX_FILE — a latent leak the old ``{**os.environ, ...}`` merge
    passed straight into the hook), then git resolution is PINNED to the
    throwaway on-main fixture: GIT_DIR/GIT_WORK_TREE take precedence over
    the hook's hardcoded ``git -C "$REPO"`` discovery (verified 2026-07-19
    on git 2.34.1), so the hook's on-main gate (``rev-parse --abbrev-ref
    HEAD`` + ``[ "$cur" = main ] || exit 0``, hook L1675-76) + checkout
    classifier (``show-ref``/``rev-parse --verify``/``cat-file -e``, hook
    L1324-1328) read the FIXTURE, never the shared root — a concurrent
    shared-root git op can no longer flip probe verdicts (#1545; incident
    #1506: 13 spurious self-test failures).
    ``EPM_GUARD_DENY_SIDECAR`` pin unchanged (#1528)."""
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    env["GIT_DIR"] = str(git_fixture / ".git")
    env["GIT_WORK_TREE"] = str(git_fixture)
    env["EPM_GUARD_DENY_SIDECAR"] = "/dev/null"
    return env


def _build_root_guard_fixture(tmp: Path) -> None:
    """``git init`` a minimal always-on-``main`` fixture in ``tmp``: branch
    ``main``, ONE empty-tree commit. The empty tree means the hook's
    ``cat-file -e "HEAD:$arg"`` probe can never resolve a doc example
    against the fixture, and ``show-ref``/``rev-parse`` resolve strictly
    LESS than the real repo (only ``main``/``HEAD``/the fixture sha) — so
    the fixture cannot mint a NEW block verdict (2026-07-19 audit: 0/54
    live fence-verdict changes vs ambient). Identity via ``-c`` flags so
    no global git config is required. Raises on failure (caller converts
    to ONE loud lint error — fail loud, never a silent skip)."""
    scrub = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    # Isolate user/system git config for the BUILD too: an ambient
    # commit.gpgsign=true or template/hooks setting in the real global
    # config would fail the empty-commit build (Alternatives critic, v3).
    scrub["GIT_CONFIG_GLOBAL"] = "/dev/null"
    scrub["GIT_CONFIG_NOSYSTEM"] = "1"
    common = dict(
        check=True,
        capture_output=True,
        text=True,
        timeout=_ROOT_GUARD_TIMEOUT_S,
        env=scrub,
    )
    subprocess.run(["git", "init", "-q", "-b", "main", str(tmp)], **common)
    subprocess.run(
        [
            "git",
            "-C",
            str(tmp),
            "-c",
            "user.name=workflow-lint",
            "-c",
            "user.email=workflow-lint@localhost",
            "-c",
            "commit.gpgsign=false",
            "commit",
            "-q",
            "--allow-empty",
            "-m",
            "root-guard probe fixture",
        ],
        **common,
    )


def _iter_bash_fences(text: str) -> Iterator[tuple[int, str, str]]:
    """Yield ``(opener_lineno_1based, preceding_nonblank_line, block_text)``
    for every ``bash``/``sh``/``shell``-tagged fenced block in ``text``.

    PARITY-TOGGLE PARSER RULE (#1176 acceptance criterion 8): outside a
    fence, ANY fence line (tagged or bare, ``` or ~~~) OPENS one; inside a
    fence, ANY fence line with the SAME token CLOSES it (the closer's tag is
    ignored), while a DIFFERENT-token fence line is body content (the
    CommonMark reading). The naive "closer = same token with EMPTY tag" rule
    demonstrably desyncs on the nested-fence shape formerly at
    ``.claude/skills/weekly/SKILL.md:196-204`` (skill retired 2026-08-05;
    the shape is preserved in the fixture test) (an outer ```` ```markdown ````
    fence whose body contains an inner ```` ```diff ```` fence): the inner
    tagged line must CLOSE the outer fence, or the bare ```` ``` ```` two
    lines later opens a phantom fence that swallows the git-bearing
    ```` ```bash ```` fence at weekly:~223 — a silent false negative the
    live-tree test cannot see (absence-of-error = pass).

    An unterminated trailing fence is yielded (fail toward checking — the
    ``check_grep_qv`` precedent). ``text.split('\\n')``, never
    ``splitlines()``.
    """
    lines = text.split("\n")
    in_fence = False
    fence_token = ""
    fence_tag = ""
    opener_lineno = 0
    preceding = ""
    prev_nonblank = ""
    block_lines: list[str] = []
    for idx, line in enumerate(lines, start=1):
        m = _ROOT_GUARD_FENCE_RE.match(line)
        if m is not None:
            token, tag = m.group(1), m.group(2)
            if not in_fence:
                in_fence = True
                fence_token = token
                fence_tag = tag.lower()
                opener_lineno = idx
                preceding = prev_nonblank
                block_lines = []
                continue
            if token == fence_token:
                # Same-token fence line closes (tag ignored on close).
                if fence_tag in _ROOT_GUARD_BASH_TAGS:
                    yield opener_lineno, preceding, "\n".join(block_lines)
                in_fence = False
                prev_nonblank = line
                continue
            # Different-token fence line INSIDE a fence: body content.
            block_lines.append(line)
            continue
        if in_fence:
            block_lines.append(line)
        elif line.strip():
            prev_nonblank = line
    if in_fence and fence_tag in _ROOT_GUARD_BASH_TAGS:
        # Unterminated trailing fence: yield what was collected.
        yield opener_lineno, preceding, "\n".join(block_lines)


def _root_guard_fence_exempt(prev_line: str, sentinel: str = _ROOT_GUARD_EXEMPT_SENTINEL) -> bool:
    """True when a fence's immediately-preceding non-blank line carries the
    ``<sentinel>: <reason>`` waiver (default: the root-guard
    ``workflow-lint: allow-root-guard-block`` sentinel;
    :func:`check_bare_commit_pathspec` passes its own sentinel, #1648) with a
    NON-EMPTY reason. Reason-stripping mirrors the FI2 semantics of
    :func:`_line_waived`: strip the leading ``:``/whitespace and the trailing
    HTML-comment closer (``-->``) / backticks / whitespace before testing, so
    a bare closer (``: -->``, or the sentinel with no colon) never counts as
    a reason and wrongly waives."""
    if sentinel not in prev_line:
        return False
    _, _, tail = prev_line.partition(sentinel)
    reason = tail.lstrip(": ")
    if reason.rstrip().endswith("-->"):
        reason = reason.rstrip()[: -len("-->")]
    reason = reason.strip().strip("`").strip()
    return bool(reason)


def _run_root_guard(hook: Path, command: str, git_fixture: Path) -> tuple[int, str]:
    """Feed ``command`` to the live PreToolUse hook exactly as the harness
    does — stdin JSON ``{"tool_input": {"command": ...}}`` — and return
    ``(returncode, stderr)``. ``cwd`` is pinned to the hook's own repo root
    (``hook.parent.parent``) so repo-state-consulting detectors give
    deterministic-by-construction verdicts when the lint runs from a
    worktree (#1176 round-1 Methodology c1). ``EPM_GUARD_DENY_SIDECAR`` is
    pinned to ``/dev/null`` so lint-driven hook executions (self-test probes
    + the per-fence scan loop) never append synthetic deny rows to the
    production deny-event sidecar (#1528); the pin is env-only and cannot
    change the rc-0/2 verdict this check reads. The subprocess env comes
    from :func:`_root_guard_git_env` — ambient ``GIT_*`` scrubbed, git
    resolution pinned to ``git_fixture`` (a required parameter so every
    probe call site goes through the fixture; #1545)."""
    payload = json.dumps({"tool_input": {"command": command}})
    proc = subprocess.run(
        ["bash", str(hook)],
        input=payload,
        capture_output=True,
        text=True,
        timeout=_ROOT_GUARD_TIMEOUT_S,
        cwd=str(hook.parent.parent),
        env=_root_guard_git_env(git_fixture),
    )
    return proc.returncode, proc.stderr


def _root_guard_target_files(root: Path) -> list[Path]:
    """The ``check_git_recipes_root_guard`` scan set: agents + skills via the
    worktree-safe :func:`_iter_ask_target_files` house helper, PLUS
    ``.claude/rules/*.md`` + ``CLAUDE.md`` under the same other-worktree
    exclusion (current worktree scanned)."""
    targets: list[Path] = list(_iter_ask_target_files(root))
    prefix = _other_worktree_prefix(root)
    rules_dir = root / ".claude" / "rules"
    if rules_dir.is_dir():
        targets.extend(
            p
            for p in sorted(rules_dir.glob("*.md"))
            if p.is_file() and not _is_other_worktree_path(p, prefix)
        )
    claude_md = root / "CLAUDE.md"
    if claude_md.is_file() and not _is_other_worktree_path(claude_md, prefix):
        targets.append(claude_md)
    return targets


def check_git_recipes_root_guard(  # noqa: C901 -- flat fail-loud ladder + per-fence probe/verdict dispositions (#1176; retry-once-then-WARN #1610); extracting a branch would just relocate it
    *,
    repo_root: Path | None = None,
    hook_path: Path | None = None,
    max_workers: int = 8,
    warn_sink: list[str] | None = None,
) -> list[str]:
    """FAIL if any documented executable git recipe — a ``bash``/``sh``/
    ``shell``-tagged fenced block in ``.claude/agents/*.md``,
    ``.claude/skills/**/SKILL.md``, ``.claude/rules/*.md``, or ``CLAUDE.md``
    — is BLOCKED (exit 2) by the live repo-root branch guard
    ``scripts/guard_repo_root_branch.sh`` when fed WHOLE, as one command
    string, on the hook's stdin-JSON PreToolUse contract.

    Incident #1047: the gate-block cleanup restore recipe shipped without a
    per-clause ``git -C`` waiver at 2 sites and survived plan review,
    fact-check, and a 6-critic ensemble; only the code-reviewer EXECUTING
    the hook caught it. Executing the REAL hook — instead of maintaining a
    parallel pattern-match — covers every current and future detector
    verbatim (clause splitting, ``-C`` waivers, comment-tail strip, heredoc
    strip) and stays current as the hook evolves. The regex siblings
    (:func:`check_no_repo_root_git_reset_hard` /
    :func:`check_no_repo_root_worktree_revert`) stay: they scan PROSE lines
    too (this check scans only executable fences); overlap is harmless.

    Mechanics:

    * Whole-block feed — faithful to how a session pastes a recipe (one
      Bash call; the hook clause-splits internally). Per-line feeding would
      shred ``if``/``for``/heredoc constructs and false-positive on inert
      heredoc bodies the hook's #1058 strip correctly ignores. The hook's
      ``-C`` waiver is per-clause, so whole-block is not weaker on waivers.
    * ``git``-literal pre-filter — the literal ``git`` is a NECESSARY
      textual condition for every hook BLOCK detector today (all detectors
      anchor on a ``git`` bigram or the legacy loose probe), the #1162
      necessary-textual-condition perf-gate style. If a future hook
      detector fires WITHOUT a ``git`` literal in the command, this
      pre-filter needs a matching update.
    * Exemption — ``<!-- workflow-lint: allow-root-guard-block: <reason>
      -->`` (NON-EMPTY reason) on the immediately-preceding non-blank line
      above the fence opener waives that fence: for deliberate anti-pattern
      examples and pod-side recipes that run over SSH on a pod's
      ``/workspace`` clone, never at the VM repo root.
    * FAIL-LOUD self-test FIRST — the hook's stdin parse fail-softs to
      exit 0 when ``jq`` is missing (``guard_repo_root_branch.sh`` ~line
      390: ``jq ... || exit 0``), so a positive probe (a known-blocked
      command MUST rc 2) plus a negative probe (a benign command MUST rc 0)
      run before any scan; a missing hook, a self-test crash, a fail-OPEN
      hook, or a fail-CLOSED hook is ONE loud lint error — never a silent
      pass. Only rc 0/2 are interpreted; any other rc on a fence probe is
      retried once, then WARNed (never a FAIL) — a transient kill/timeout on
      a gate scratch tree must not flip a clean gate to block (#1610; rc=1
      is NON-BLOCKING under the PreToolUse contract, never a pass).

    UNDER-COVERAGE RESIDUALS (the guarantee is scoped to the whole-bash-fence
    paste surface — all four NAMED, per the #1176 round-1 Alternatives
    critique; (b)+(c) are pinned by a committed known-miss fixture):

    (a) untagged / other-tagged executable fences are not scanned;
    (b) prose inline-code recipes (a backtick code span in a prose bullet
        carrying a git command — one of #1047's own two original sites) are
        structurally outside any fence-feed design;
    (c) ``#``-commented instruction lines inside fences (the other #1047
        original site) — the hook's comment-tail strip correctly ALLOWS the
        block-as-pasted, while the doc tells the session to run the command
        uncommented;
    (d) placeholder-substitution false-PASS direction — the hook's checkout
        classifier resolves ``show-ref``/``rev-parse``/``cat-file`` against
        the throwaway on-main FIXTURE at lint time (plus the hook's
        real-filesystem ``[ -e ]`` probe), so an unresolvable
        ``<branch>``/``$VAR`` argument keeps ALLOW at lint time but can
        BLOCK after a session substitutes a real value (conclusion
        unchanged by the #1545 fixture pin).

    Prose/comment scanning is deliberately REJECTED for v1 (prose quotes
    gated verbs constantly — the regex siblings' restricted scope exists
    precisely because of prose false positives); the runtime hook + the
    reviewer execute-the-hook practice remain the guard there.

    REPO-STATE FLAP ATTRIBUTION: git-state resolution is PINNED to a
    throwaway on-main fixture (:func:`_root_guard_git_env` /
    :func:`_build_root_guard_fixture`, #1545), so a concurrent shared-root
    git op (a rebase transiently detaching HEAD — incident #1506, 13
    spurious self-test failures) can no longer flip probe verdicts. The
    REMAINING state-consulting reads are the hook's REAL-FILESYSTEM
    existence probe (``[ -e "$REPO/$arg" ]``, hook L1328) and doc-content
    changes: a ``git checkout <example-name>`` fence can still FLIP
    verdict when a tracked path/file of that name appears or disappears
    on disk. If the live-tree test starts flapping on unrelated diffs,
    check THAT path-existence state first (the pre-registered remedy if
    the ``[ -e ]`` path ever flaps is probe serialization on a dedicated
    lock — plan #1545 K1) — distinct from hook-evolution flap (a new
    detector), which names the new BLOCKED line in the error. Post-fix, a
    recurring "blocked-probe rc=0" self-test failure is NO LONGER
    attributable to concurrent git state — attribute to a jq/exec
    transient first (the hook's stdin parse fail-softs to exit 0).

    Scope: agents + skills via the worktree-safe :func:`_iter_ask_target_files`
    house helper, PLUS ``.claude/rules/*.md`` + ``CLAUDE.md`` under the same
    other-worktree exclusion (safe here, unlike the prose-scanning regex
    siblings, because only fenced bash is scanned and the hook strips
    comments/heredocs). ``scripts/**`` is never scanned. ``repo_root`` /
    ``hook_path`` / ``max_workers`` are unit-test override hooks; production
    callers pass defaults. ``warn_sink`` mirrors ``check_lens_coverage``'s
    hook: WARNs append there when provided, else go to stderr with a
    ``WARN: `` prefix; WARNs never enter the returned FAIL list. Bundled
    into the no-flags default run.
    """
    root = repo_root if repo_root is not None else _REPO_ROOT
    hook = hook_path if hook_path is not None else _ROOT_GUARD_HOOK
    # (1) FAIL-LOUD self-test — never a silent pass on a missing / broken /
    # fail-open / fail-closed hook.
    if not hook.is_file():
        return [
            f"{hook}: root-guard hook script missing — "
            f"check-git-recipes-root-guard cannot run (FAIL, not skip)"
        ]
    with tempfile.TemporaryDirectory(prefix="wl_rootguard_fixture_") as td:
        fixture = Path(td)
        try:
            _build_root_guard_fixture(fixture)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as exc:
            return [
                f"{hook}: root-guard git fixture build failed ({exc}) — "
                f"check cannot run (FAIL, not skip)"
            ]
        try:
            rc_pos, _ = _run_root_guard(hook, _ROOT_GUARD_SELFTEST_BLOCKED, fixture)
            rc_neg, _ = _run_root_guard(hook, _ROOT_GUARD_SELFTEST_BENIGN, fixture)
        except (subprocess.TimeoutExpired, OSError) as exc:
            return [f"{hook}: root-guard self-test crashed ({exc}) — check cannot run"]
        if rc_pos != 2 or rc_neg != 0:
            return [
                f"{hook}: root-guard self-test failed (blocked-probe rc={rc_pos}, "
                f"expected 2; benign-probe rc={rc_neg}, expected 0). Likely jq "
                f"missing (the hook's stdin parse fail-softs to exit 0) or a hook "
                f"regression — the check refuses to run against a fail-open or "
                f"fail-closed hook."
            ]
        # (2) Enumerate targets: agents + skills via the worktree-safe house
        # helper, plus rules/*.md + CLAUDE.md under the same other-worktree
        # exclusion.
        targets = _root_guard_target_files(root)
        # (3) Collect git-bearing bash fences (perf gate: the `git` literal is a
        # NECESSARY condition for every hook detector — see docstring; drops
        # 103 -> ~23 fences on issue/SKILL.md alone).
        work: list[tuple[Path, int, str]] = []
        for p in targets:
            try:
                text = p.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            for lineno, prev_line, block in _iter_bash_fences(text):
                if "git" not in block:
                    continue
                if _root_guard_fence_exempt(prev_line):
                    continue
                work.append((p, lineno, block))

        # (4) Execute the hook per block, in parallel; deterministic ordering.
        # Probes share the fixture read-only across the ThreadPool (git reads
        # are concurrent-safe).
        def _probe(item: tuple[Path, int, str]) -> tuple[Path, int, int, str]:
            p, lineno, block = item
            rc, stderr = -1, ""
            for _attempt in (0, 1):
                try:
                    rc, stderr = _run_root_guard(hook, block, fixture)
                except (subprocess.TimeoutExpired, OSError) as exc:
                    rc, stderr = -1, f"hook invocation failed: {exc}"
                if rc in (0, 2):
                    break
                # unexpected rc: transient infra class (#1610) — retry once
            return (p, lineno, rc, stderr)

        errors: list[str] = []

        def _warn(msg: str) -> None:
            if warn_sink is not None:
                warn_sink.append(msg)
            else:
                sys.stderr.write(f"WARN: {msg}\n")

        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            results = list(ex.map(_probe, work))
        for p, lineno, rc, stderr in sorted(results, key=lambda r: (str(r[0]), r[1])):
            if rc == 0:
                continue
            first = stderr.strip().split("\n")[0][:200]
            if rc == 2:
                errors.append(
                    f"{p}:{lineno}: bash recipe fence is BLOCKED by "
                    f"scripts/guard_repo_root_branch.sh — a session pasting this "
                    f"recipe into one Bash call dies at the PreToolUse gate "
                    f"(incident #1047: a cleanup recipe without a -C waiver "
                    f"survived plan review + 6 critics and was caught only by the "
                    f"reviewer executing the hook). Hook says: {first!r}. Fix the "
                    f"recipe (per-clause `git -C <path>` waiver / the sanctioned "
                    f"worktree or gh-pr-merge form), or — for a deliberate "
                    f"anti-pattern example or a pod-side recipe — add "
                    f"`<!-- {_ROOT_GUARD_EXEMPT_SENTINEL}: <reason> -->` on the "
                    f"line directly above the fence opener."
                )
            else:
                _warn(
                    f"{p}:{lineno}: root-guard hook returned unexpected rc={rc} "
                    f"after one retry — a NON-BLOCKING code under the PreToolUse "
                    f"contract (only rc 0/2 are interpreted; rc=1/127/timeout "
                    f"signals a transient hook-invocation or infrastructure "
                    f"error, not a verdict — #1610) ({first!r})."
                )
        return errors


# --check-bare-commit-pathspec (#1648): a fenced `git commit` recipe with no
# pathspec commits the WHOLE staged index — on the always-concurrent shared
# repo root that sweeps sibling sessions' staged files onto the commit
# (incident 7dbde267f1, 2026-07-21: 4 foreign files swept onto main; #1630
# fixed /daily per-file). Convention: CLAUDE.md § Concurrent repo-root
# committers ("stage by explicit path only") — the commit-side analogue is
# the pathspec-limited `git commit -m "..." -- <paths>` form. Python string
# data only: this check scans workflow-surface .md files, never scripts/**,
# so these constants cannot self-flag.
_BARE_COMMIT_SENTINEL = "workflow-lint: allow-bare-commit-block"
# One git-commit INVOCATION: `git <global-flags>* commit`; group 1 captures
# the global flags so the -C exemption is per-invocation (a `git -C` command
# elsewhere on the line cannot waive a separate bare commit).
BARE_COMMIT_GIT_RE = re.compile(
    r"(?<![\w./-])git\s+((?:-c\s+\S+\s+|--\S+\s+|-C\s+\S+\s+)*)commit\b"
)
# Case-sensitive: `-C <dir>` (a named tree), never `-c <cfg>`.
BARE_COMMIT_GITC_FLAG_RE = re.compile(r"(?:^|\s)-C\s")
# ` -- ` separator + >=1 following token (a bare trailing `--` with nothing
# after commits the whole index anyway and does NOT count).
BARE_COMMIT_PATHSPEC_RE = re.compile(r"\s--\s+\S")
# xargs exempts ONLY with -r/--no-run-if-empty: a flag-less GNU xargs runs
# the command ONCE on empty input -> `git commit -m "..." --` with nothing
# appended -> whole-index sweep. The one live form (the issue/SKILL.md
# additive-checkout recipe) uses -r. Token loop: whole whitespace-delimited
# tokens free of `|`/`;`/`&`, then a short-flag cluster containing `r` or
# the long form.
BARE_COMMIT_XARGS_RE = re.compile(
    r"(?<![\w./-])xargs\s(?:[^|;&\s]+\s+)*(?:-\w*r\w*\b|--no-run-if-empty\b)"
)
_BARE_COMMIT_UNESCAPED_DQUOTE_RE = re.compile(r'(?<!\\)"')


def check_bare_commit_pathspec(*, repo_root: Path | None = None) -> list[str]:
    """FAIL if any ``bash``/``sh``/``shell``-tagged fenced block in the
    workflow docs (``.claude/agents/*.md``, ``.claude/skills/**/SKILL.md``,
    ``.claude/rules/*.md``, ``CLAUDE.md`` — the
    :func:`_root_guard_target_files` surface, reused) prescribes a
    ``git commit`` invocation with no trailing `` -- <pathspec>``.

    A bare commit at the always-concurrent shared repo root commits the
    WHOLE staged index, sweeping sibling sessions' staged files onto the
    commit (incident ``7dbde267f1``, 2026-07-21: 4 foreign files swept onto
    main; #1630 fixed /daily's recipes per-file — this check generalizes
    that guard to the whole workflow surface mechanically, #1648).

    Structural exemptions (why each form is safe):

    * `` -- <pathspec>`` tail (the literal `` -- `` separator + >=1
      following token) — the required convention itself;
    * ``git -C <tree> commit`` (per-invocation, case-sensitive ``-C``) —
      commits a NAMED tree (worktree / scratch / pod clone) with its OWN
      index, not the ambient shared root;
    * ``xargs -r ... git commit`` (``-r``/``--no-run-if-empty`` REQUIRED) —
      xargs appends the file list as trailing args = a runtime pathspec; a
      flag-less xargs runs once on empty input (whole-index sweep) and is
      NOT exempt;
    * ``#`` comment lines — instruction/anti-pattern prose, not a command;
    * ``--dry-run`` — preview, commits nothing (the ``check_piped_git_push``
      precedent);
    * per-fence sentinel ``<!-- workflow-lint: allow-bare-commit-block:
      <reason> -->`` (NON-EMPTY reason, on the immediately-preceding
      non-blank line) — for deliberate anti-pattern examples / pod-side
      recipes.

    NAMED RESIDUALS (fail-toward-pass unless stated otherwise): (a)
    untagged / other-tagged fences are not scanned; (b) prose inline-code
    recipes are structurally outside a fence scanner; (c) heredoc bodies
    inside fences are scanned as lines (a heredoc-embedded commit string
    could false-positive — zero live; sentinel waives); (d) a quoted
    message CONTAINING the literal text ``git commit -m ...`` could
    false-positive after quote-joining — zero live; sentinel waives; (e)
    ``-C "path with spaces"`` fails the flags-group match and the
    invocation goes unflagged; (f) ``scripts/**/*.sh`` shell scripts are
    OUT of scope (they run in varied cwds — pods, worktrees — where bare
    commits are often legitimately scoped; the guard targets the
    documented-recipe teaching surface); (g) the pathspec / ``--dry-run`` /
    xargs searches operate on the whole logical line, so a `` -- <token>``
    or ``--dry-run`` INSIDE a quoted ``-m`` message — or a LATER command's
    `` -- `` on a compound ``&&``/``;`` line — falsely exempts an earlier
    bare commit; (h) the ``git -C <tree>`` exemption is a static heuristic
    — a ``-C`` target that RESOLVES to the shared root at runtime is exempt
    though not provably safe; (i) skill SUPPORT files (``markers.md``,
    ``SPEC.md``, exemplars) are outside :func:`_root_guard_target_files`.

    Line handling: backslash continuations are joined (bash drops the
    backslash-newline pair), then lines are joined until double quotes
    balance (multi-line ``-m "..."`` messages); errors report the FIRST
    line of the joined logical line. ``repo_root`` is a unit-test override;
    production callers pass the default. Bundled into the no-flags default
    run.
    """
    root = repo_root if repo_root is not None else _REPO_ROOT
    errors: list[str] = []
    for path in _root_guard_target_files(root):
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        for opener, prev_line, block in _iter_bash_fences(text):
            if "git" not in block:
                continue
            if _root_guard_fence_exempt(prev_line, sentinel=_BARE_COMMIT_SENTINEL):
                continue
            lines = block.split("\n")
            n = len(lines)
            i = 0
            while i < n:
                start = i
                if lines[i].lstrip().startswith("#"):
                    i += 1
                    continue
                logical = lines[i]
                # Join backslash continuations (bash drops the
                # backslash-newline pair, concatenating directly).
                while logical.endswith("\\") and i + 1 < n:
                    i += 1
                    logical = logical[:-1] + lines[i]
                # Join following lines until double quotes balance
                # (multi-line `-m "..."` messages).
                while len(_BARE_COMMIT_UNESCAPED_DQUOTE_RE.findall(logical)) % 2 == 1 and i + 1 < n:
                    i += 1
                    logical = logical + "\n" + lines[i]
                i += 1
                for m in BARE_COMMIT_GIT_RE.finditer(logical):
                    if BARE_COMMIT_GITC_FLAG_RE.search(m.group(1)):
                        continue  # `git -C <tree> commit`: named tree, own index
                    if BARE_COMMIT_XARGS_RE.search(logical[: m.start()]):
                        # xargs WITH -r/--no-run-if-empty appends the file
                        # list = runtime pathspec; xargs AFTER the match
                        # never waives (search bounded to logical[:m.start()]).
                        continue
                    if "--dry-run" in logical:
                        continue  # preview, commits nothing
                    if BARE_COMMIT_PATHSPEC_RE.search(logical[m.end() :]):
                        continue  # ` -- <pathspec>` present
                    errors.append(
                        f"{path}:{opener + 1 + start}: fenced `git commit` without a "
                        f"trailing ` -- <pathspec>` — a bare commit at the "
                        f"always-concurrent shared repo root sweeps sibling sessions' "
                        f"staged files onto the commit (incident 7dbde267f1; #1630 "
                        f"fixed /daily per-file; #1648). Append ` -- <explicit paths>` "
                        f"(prefer file-level pathspecs for shared-root recipes; note "
                        f"`-a`/`--amend` are incompatible with this remedy), scope "
                        f"with `git -C <tree>`, or waive the fence with "
                        f"`<!-- {_BARE_COMMIT_SENTINEL}: <reason> -->` on the "
                        f"immediately-preceding non-blank line."
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


# --- #1081 crash-fix-relaunch fix-engaged contract pins (#1181) --------------
#
# Three surfaces carry the #1081 crash-fix-relaunch contract (fix-commit
# ancestry probe + stale-checkpoint disposition + the fix_sha= note-token
# convention). Each span runs from a UNIQUE literal anchor (which must stay
# on one physical line — all three are <=54 columns incl. indent, safely
# under the repo's ~72-col prose wrap; an anchor rename/removal FAILs loud
# and requires a deliberate lint update, per the #963 convention) to its
# per-surface end regex, and is whitespace-normalized before token matching
# (several tokens hard-wrap in the live files; an innocent reflow must not
# FAIL the fleet). The negative regex is a LITERAL-COUPLING BACKSTOP against
# re-introducing the unconditional "resolves EMPTY" confirm wording that the
# #1081 round-2 blocker fix (concern retain-disposition-d3-empty-glob)
# replaced with the disposition-conditional trio; its lookahead spares the
# healthy trio phrasing "resolves EMPTY / to the fresh path / ...". It pins
# the HISTORICAL #1081 wording only — paraphrases are the positive tokens'
# job; do NOT widen it "to be safe" or it false-fires on the healthy trio.

_CRASH_FIX_RESOLVES_EMPTY_RE = re.compile(r"\bresolves?\s+empty\b(?!\s*/)", re.IGNORECASE)

# (relative path parts, unique literal anchor, span-end regex, human name,
#  required tokens over the whitespace-normalized span)
_CRASH_FIX_CONTRACT_SURFACES: tuple[tuple[tuple[str, ...], str, str, str, tuple[str, ...]], ...] = (
    (
        (".claude", "agents", "experimenter.md"),
        "**Crash-fix relaunch (brief carries `fix_sha=`):**",
        r"\n\s*\n|\n\d+\. ",
        "experimenter D3 crash-fix-relaunch paragraph",
        (
            "git merge-base --is-ancestor <fix_sha> HEAD",
            "ANY non-zero exit = fix absent — do NOT launch",
            "execute the brief's stale-checkpoint disposition before launch",
            "confirming the resume glob resolves as the disposition requires",
            "empty / the fresh path / exactly the RETAINED expected paths",
            # MooseFS served-bytes content-read duty on same-pod relaunches (#1112/#1594)
            "MooseFS content read",
        ),
    ),
    (
        (".claude", "rules", "crash-fix-rounds.md"),
        "The fresh `epm:run-launched` note ALSO records",
        r"\n\s*\n",
        "crash-fix-rounds fix_sha note-token paragraph",
        (
            "records `fix_sha=<sha>` and the executed disposition",
            "note-token convention",
            "carries both (`fix_sha=` + the element-5 disposition verbatim)",
            "EXEMPT: `infra`-row experimenter respawns",
        ),
    ),
    (
        (".claude", "skills", "issue", "SKILL.md"),
        "*`code`-row relaunch contract (#779):*",
        r"\n\s*\n",
        "SKILL.md Step 7 code-row relaunch contract paragraph",
        (
            "brief carries `fix_sha=` + the element-5 stale-artifact disposition",
            "copied from the implementer's fix-engaged declaration",
            "enforces BOTH before dispatch: the fix-commit ancestry probe and the declared "
            "disposition",
        ),
    ),
)


def check_crash_fix_relaunch_contract(*, repo_root: Path | None = None) -> list[str]:
    """FAIL if the #1081 crash-fix-relaunch contract prose regresses on any of
    its three surfaces (#1181): the fix-commit ancestry probe + fail-loud, the
    disposition-conditional three-way resume-glob confirm (the #1081 round-2
    blocker fix, concern retain-disposition-d3-empty-glob), and the fix_sha=
    note-token / brief duty.

    Scope notes (inherited verbatim from the #963 precedent):

    (a) The negative regex is a LITERAL-COUPLING BACKSTOP only — it pins the
        HISTORICAL #1081 wording (an unconditional "resolves EMPTY" confirm);
        paraphrases are the positive tokens' job. Do not weaken a token
        "because the regex covers it", and do not widen the regex "to be
        safe" — a wider regex false-fires on the healthy trio wording
        "resolves EMPTY / to the fresh path / ...".
    (b) Paragraph-scoped — a contradictory instruction OUTSIDE an anchored
        span, or one ADDED alongside the intact tokens INSIDE a span, is
        invisible to it (inherent to the token-lint class).
    (c) A mid-paragraph blank line truncates the span and FAILs all downstream
        tokens at once — a deliberate restructure requires a deliberate lint
        update in the same commit.

    ``repo_root`` is a unit-test override hook; production callers pass None
    (canonical repo root). Bundled into the no-flags default run.
    """
    root = repo_root if repo_root is not None else _REPO_ROOT
    errors: list[str] = []
    for parts, anchor, end_re, name, tokens in _CRASH_FIX_CONTRACT_SURFACES:
        path = root.joinpath(*parts)
        if not path.is_file():
            errors.append(
                f"{path}: missing — the #1081 crash-fix-relaunch contract "
                f"({name}) must live here (#1181)."
            )
            continue
        text = path.read_text(encoding="utf-8")
        n_anchors = text.count(anchor)
        if n_anchors == 0:
            errors.append(
                f"{path}: missing the anchor {anchor!r} (#1081) — the {name} pins the "
                f"crash-fix-relaunch fix-engaged contract and must not be removed or "
                f"renamed without updating this lint (#1181)."
            )
            continue
        if n_anchors > 1:
            errors.append(
                f"{path}: {n_anchors} anchors {anchor!r} found — the {name} must be "
                f"UNIQUE (a stale duplicate could satisfy the token scan while the "
                f"operative paragraph regresses; #1081/#1181). Remove the duplicate."
            )
            continue
        start = text.find(anchor)
        end_m = re.search(end_re, text[start:])
        end = start + end_m.start() if end_m is not None else len(text)
        span = re.sub(r"\s+", " ", text[start:end])
        for token in tokens:
            if token not in span:
                errors.append(
                    f"{path}: {name} missing token {token!r} (#1081) — note: the span "
                    f"ends at the first blank line / next numbered item, so a split "
                    f"paragraph FAILs all downstream tokens at once (a deliberate "
                    f"restructure needs a lint update, #1181)."
                )
        if _CRASH_FIX_RESOLVES_EMPTY_RE.search(span):
            errors.append(
                f"{path}: {name} couples the resume-glob confirm to an unconditional "
                f"'resolves EMPTY' (#1081) — the confirm is disposition-CONDITIONAL "
                f"(empty / the fresh path / exactly the RETAINED expected paths; "
                f"round-2 blocker retain-disposition-d3-empty-glob)."
            )
    return errors


_VM_THREAD_CAP_PREFIX = (
    "OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8"
    " MALLOC_ARENA_MAX=2"
)

# {file: minimum occurrence count of the literal prefix}. Count floors (not
# bare presence) so stripping the prefix from ONE template instance while a
# prose mention survives still FAILs (Methodology + Statistics critic
# hardening; raised to Must-Fix by the Codex alternatives critic, round 1):
# SKILL.md 1 (detached-launch template), experiment-implementer.md 2 (bullet
# + setsid line), code-style.md 3 (line-20 bullet + the two § nohup template
# copies), analyzer-section-reference.md 1 (off-pod template).
# The trailing MALLOC_ARENA_MAX=2 is the glibc arena-fragmentation cap
# (#1315: a small-tensor eigh bootstrap grew 20-21.7 GB RSS across passes
# under the four thread caps alone; ~1 GB with the arena cap). Its value 2
# is NOT coupled to _DEFAULT_VM_THREAD_CAP (the 8s below) — it caps malloc
# ARENA COUNT, not thread count.
# BINDING CONVENTION (keeps the floors template-anchored): rationale PROSE in
# the pinned files refers to the caps by the shorthand
# "OMP/MKL/OPENBLAS/NUMEXPR=8" (optionally "+ MALLOC_ARENA_MAX=2") and NEVER
# spells the full literal prefix, so
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
    cap prefix — the four thread caps plus the glibc arena cap
    MALLOC_ARENA_MAX=2 (#891/#1315) — as the branch-age-independent fallback (incident #779,
    2026-07-02: a pre-#847 worktree ran 78 uncapped threads ~20h after the fix
    landed on main). This check pins the LITERAL prefix — with a per-file
    occurrence-count floor, so stripping it from a TEMPLATE instance while a
    prose mention survives still fails — in the four guidance surfaces, making
    a silent re-open of the gap loud. (Residual: an edit swapping which LINE
    carries an occurrence at equal count passes; the count floor is the
    granularity/robustness trade the plan accepts.)
    The value 8 is deliberately coupled to ``_DEFAULT_VM_THREAD_CAP`` in
    env.py: changing either requires changing both (and this constant), which
    is the point — drift fails loud. The arena cap's value 2 is NOT coupled
    to that constant (it bounds malloc arenas, not threads; #1315 validated
    2 empirically). ``repo_root`` is a unit-test override
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


_ASW_REL = "scripts/autonomous_session_watch.py"
_ASW_HEADER_RE = re.compile(r"^(\d+) passes\b", re.M)
_ASW_ITEM_RE = re.compile(r"^(\d+)\. \*\*", re.M)
# The crash-recovery respawn loop is an inline block in main() (not a
# *_pass-named function); it counts as one pass. If it is ever refactored
# into a crash_recovery_pass() function, set this to 0. A NEW inline
# (non-*_pass-named) top-level pass block requires bumping this constant —
# the watcher docstring's pass-definition paragraph says so too.
_ASW_INLINE_PASS_BLOCKS = 1


def check_asw_docstring_pass_count(*, watcher_path: Path | None = None) -> list[str]:
    """FAIL if the watcher docstring's '<N> passes' header digit diverges from
    the numbered inventory (line-start ``<digit>. **`` items), the items are
    not exactly 1..N sequential, or N diverges from the live pass set in
    ``main()`` (distinct ``*_pass`` calls plus ``_ASW_INLINE_PASS_BLOCKS``
    inline crash-recovery blocks). ``watcher_path`` is a unit-test override;
    production callers pass None. Bundled into the no-flags default run.
    (#1225; manual catch-ups #1021, #1169.)

    Named residuals (accepted): the header's execution-order PROSE is
    unpinned — only the COUNT is linted, not the order; and assertion (3) is
    fail-unsafe against a future pass function NOT named ``*_pass`` or a
    SECOND inline (non-``*_pass``-named) pass block, either of which would
    escape the live-set count — author discipline plus the watcher
    docstring's pass-definition paragraph are the cover.
    """
    path = watcher_path if watcher_path is not None else _REPO_ROOT / _ASW_REL
    errors: list[str] = []
    if not path.is_file():
        return [f"{path}: missing — cannot verify the watcher docstring pass count."]
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError as exc:
        return [f"{path}: unparseable ({exc}) — cannot verify the docstring pass count."]
    doc = ast.get_docstring(tree, clean=False)
    if not doc:
        return [f"{path}: no module docstring — the pass inventory contract is gone."]

    # (1) header digit — exactly one line-start '<N> passes' match required.
    headers = _ASW_HEADER_RE.findall(doc)
    if len(headers) != 1:
        errors.append(
            f"{path}: expected exactly one line-start '<N> passes' digit header in the "
            f"module docstring, found {len(headers)} — the count must be a DIGIT "
            f"('24 passes'), never a number word ('Fourteen passes'), and must appear "
            f"at line start exactly once; see #1225."
        )
        return errors  # items/live checks are meaningless without a parseable header
    header_n = int(headers[0])

    # (2) numbered items — exactly 1..N, sequential, no holes/duplicates.
    items = [int(m.group(1)) for m in _ASW_ITEM_RE.finditer(doc)]
    if items != list(range(1, len(items) + 1)):
        errors.append(
            f"{path}: docstring numbered items are not exactly 1..{len(items)} "
            f"in order (got {items}) — renumber the inventory."
        )
    if len(items) != header_n:
        errors.append(
            f"{path}: docstring header says {header_n} passes but the inventory has "
            f"{len(items)} numbered items — add/remove the item AND update the digit."
        )

    # (3) live-pass cross-check: distinct *_pass calls inside main() + the
    # inline crash-recovery block(s) must equal the header digit. This is the
    # assertion that would have caught #1021/#1169 (code->doc drift).
    main_fn = next(
        (n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main"), None
    )
    if main_fn is None:
        errors.append(f"{path}: no main() found — cannot cross-check the live pass set.")
        return errors
    live = {
        node.func.id
        for node in ast.walk(main_fn)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id.endswith("_pass")
    }
    live_n = len(live) + _ASW_INLINE_PASS_BLOCKS
    if live_n != header_n:
        errors.append(
            f"{path}: docstring header says {header_n} passes but main() calls "
            f"{len(live)} distinct *_pass-named functions "
            f"(+{_ASW_INLINE_PASS_BLOCKS} inline crash-recovery block(s), "
            f"_ASW_INLINE_PASS_BLOCKS in workflow_lint.py) = {live_n} — a pass was "
            f"added/removed without reconciling the docstring inventory (add a "
            f"numbered item + bump the digit), or the new pass is not a "
            f"*_pass-named function called from main() (only those are counted; "
            f"a new INLINE pass block requires bumping _ASW_INLINE_PASS_BLOCKS), "
            f"or that constant is stale."
        )
    return errors


_AWK_ELISION_ANCHOR = "f=!f"

# The two FULL-TEXT homes of the ban-gate awk elision program (#1153, origin
# #998): the /issue SKILL.md Step 9a-humanize gate and the analyzer's
# Step 4.5 mirror. A future third full-text copy must be added here.
_AWK_ELISION_HOMES = (
    ".claude/skills/issue/SKILL.md",
    ".claude/rules/analyzer-section-reference.md",
)

_AWK_ELISION_PROGRAM_RE = re.compile(r"awk '([^']*)'")


def check_awk_elision_parity(*, repo_root: Path | None = None) -> list[str]:
    """FAIL if the ban-gate awk elision program drifts between its two
    full-text homes (#1153, origin #998).

    The elision program is executable text agents copy-paste at run time
    (/issue SKILL.md Step 9a-humanize; analyzer-section-reference.md Step
    4.5), so a divergent copy makes the humanize ban-gate behave differently
    depending on which file the agent read. Per home the check fails loud
    when the file is missing, when the ``f=!f`` anchor matches 0 or >1
    lines, when the anchor line's total single-quote count is not exactly 2
    (a program that gained a shell quote-escape would truncate the
    extraction at the first quote IDENTICALLY in both homes, hiding drift
    past the truncation point — so any non-2 count fails instead), or when
    no single ``awk '...'`` span is extractable; then the two extracted
    PROGRAMS must compare equal. Scope: the quoted PROGRAM only — the
    surrounding invocation (input/output paths, fencing, indentation,
    continuation lines) legitimately differs between homes. Parity is not
    correctness: an identical-but-broken edit applied to BOTH homes passes
    by design. A future third full-text copy must be added to
    ``_AWK_ELISION_HOMES``. ``repo_root`` is a unit-test override hook;
    production callers pass None. Bundled into the no-flags default run.
    """
    root = repo_root if repo_root is not None else _REPO_ROOT
    errors: list[str] = []
    programs: list[tuple[str, str]] = []
    for rel in _AWK_ELISION_HOMES:
        p = root / rel
        if not p.is_file():
            errors.append(
                f"{p}: missing — a ban-gate awk elision home must exist; if the "
                f"program deliberately moved, update _AWK_ELISION_HOMES (#1153)."
            )
            continue
        anchor_lines = [
            ln for ln in p.read_text(encoding="utf-8").split("\n") if _AWK_ELISION_ANCHOR in ln
        ]
        if len(anchor_lines) != 1:
            errors.append(
                f"{p}: expected exactly 1 line containing the awk elision anchor "
                f"{_AWK_ELISION_ANCHOR!r}, found {len(anchor_lines)} — a moved, "
                f"deleted, or duplicated copy must not silently pass (#1153)."
            )
            continue
        line = anchor_lines[0]
        n_quotes = line.count("'")
        if n_quotes != 2:
            errors.append(
                f"{p}: the awk elision anchor line carries {n_quotes} single-quote "
                f"character(s), expected exactly 2 — a reflowed program or a gained "
                f"quote-escape would silently truncate the extraction; keep the "
                f"program one plain `awk '...'` span on one line, or update this "
                f"lint (#1153)."
            )
            continue
        progs = _AWK_ELISION_PROGRAM_RE.findall(line)
        if len(progs) != 1:
            errors.append(
                f"{p}: could not extract exactly one single-quoted awk program from "
                f"the anchor line (found {len(progs)}) — keep the program a single "
                f"`awk '...'` span on one line, or update this lint (#1153)."
            )
            continue
        programs.append((rel, progs[0]))
    if not errors and len(programs) == 2 and programs[0][1] != programs[1][1]:
        errors.append(
            f"{programs[0][0]} vs {programs[1][0]}: the ban-gate awk elision programs "
            f"DIFFER — the two full-text homes must stay byte-identical; edit both "
            f"homes identically (#1153)."
        )
    return errors


@dataclasses.dataclass(frozen=True)
class _RecipePin:
    """One (doc snippet <-> code constant) binding for
    :func:`check_marker_recipe_snippets`.

    ``doc_pattern`` runs on WHITESPACE-NORMALIZED doc text (all ``\\s+`` runs
    collapsed to single spaces, so markdown line wraps never matter);
    ``src_pattern`` runs on RAW source text with ``re.MULTILINE``. Each
    pattern carries EXACTLY ONE capture group — the numeric value — pinned by
    ``tests/test_workflow_lint.py::test_marker_recipe_pins_have_one_capture_group``.
    """

    label: str  # stable human name, appears in error messages
    doc_rel: str  # doc path relative to repo root
    doc_pattern: str  # regex, ONE capture group, whitespace-normalized doc text
    src_rel: str  # source path relative to repo root
    src_pattern: str  # regex, ONE capture group, raw source text (re.MULTILINE)
    symbol: str  # the code symbol name, for error messages


_MARKER_RECIPE_DOC = "docs/marker_training_recipe.md"
_MARKER_RECIPE_RULE = ".claude/rules/marker-training-recipe.md"
_MARKER_RECIPE_SRC_RECIPE = "src/explore_persona_space/artifacts/recipe.py"
_MARKER_RECIPE_SRC_SFT = "src/explore_persona_space/train/sft.py"
_MARKER_RECIPE_SRC_ORGANISMS = "src/explore_persona_space/artifacts/organisms.py"
_MARKER_RECIPE_SRC_CALLBACKS = "src/explore_persona_space/eval/callbacks.py"

# The marker-token-id doc patterns require a SPACE before the backticked
# marker (the ` ※` form) — the wrong-token prose "Avoid bare `※` id 63680"
# has a backtick, not a space, before ※, so 63680 is never captured (pinned by
# test_marker_recipe_snippets_does_not_capture_wrong_token_id). The sft.py
# pattern's trailing comma + no-word-char lookbehind exclude the
# `marker_tail_tokens: int = 0` dataclass field (sft.py, no trailing comma).
_MARKER_RECIPE_PINS: tuple[_RecipePin, ...] = (
    # --- docs/marker_training_recipe.md (5 pins) ---
    _RecipePin(
        label="marker-token-id",
        doc_rel=_MARKER_RECIPE_DOC,
        doc_pattern=r"(?: ※` id|token id) (\d+)",
        src_rel=_MARKER_RECIPE_SRC_RECIPE,
        src_pattern=r"^MARKER_TOKEN_ID = (\d+)$",
        symbol="MARKER_TOKEN_ID",
    ),
    _RecipePin(
        label="collator-tail-tokens",
        doc_rel=_MARKER_RECIPE_DOC,
        doc_pattern=r"MarkerOnlyDataCollator\(tail_tokens=(\d+)\)",
        src_rel=_MARKER_RECIPE_SRC_SFT,
        src_pattern=r"(?<!\w)tail_tokens: int = (\d+),",
        symbol="MarkerOnlyDataCollator.__init__ tail_tokens default",
    ),
    _RecipePin(
        label="mix-reject-floor",
        doc_rel=_MARKER_RECIPE_DOC,
        doc_pattern=r"reject floor (0\.\d+)",
        src_rel=_MARKER_RECIPE_SRC_ORGANISMS,
        src_pattern=r"^MIX_MAX_REJECT_FRAC = ([0-9.]+)$",
        symbol="MIX_MAX_REJECT_FRAC",
    ),
    _RecipePin(
        label="bandstop-low",
        doc_rel=_MARKER_RECIPE_DOC,
        doc_pattern=r"source ΔG ∈ \[([\d.]+), [\d.]+\] nat",
        src_rel=_MARKER_RECIPE_SRC_CALLBACKS,
        src_pattern=r"(?<!\w)low_nats: float = ([\d.]+),",
        symbol="MarkerBandStopCallback.__init__ low_nats default",
    ),
    _RecipePin(
        label="bandstop-high",
        doc_rel=_MARKER_RECIPE_DOC,
        doc_pattern=r"source ΔG ∈ \[[\d.]+, ([\d.]+)\] nat",
        src_rel=_MARKER_RECIPE_SRC_CALLBACKS,
        src_pattern=r"(?<!\w)high_nats: float = ([\d.]+),",
        symbol="MarkerBandStopCallback.__init__ high_nats default",
    ),
    # --- .claude/rules/marker-training-recipe.md (5 pins) ---
    _RecipePin(
        label="rule-marker-token-id",
        doc_rel=_MARKER_RECIPE_RULE,
        doc_pattern=r" ※` id (\d+)",
        src_rel=_MARKER_RECIPE_SRC_RECIPE,
        src_pattern=r"^MARKER_TOKEN_ID = (\d+)$",
        symbol="MARKER_TOKEN_ID",
    ),
    _RecipePin(
        label="rule-collator-tail-tokens",
        doc_rel=_MARKER_RECIPE_RULE,
        doc_pattern=r"MarkerOnlyDataCollator\(tail_tokens=(\d+)\)",
        src_rel=_MARKER_RECIPE_SRC_SFT,
        src_pattern=r"(?<!\w)tail_tokens: int = (\d+),",
        symbol="MarkerOnlyDataCollator.__init__ tail_tokens default",
    ),
    _RecipePin(
        label="rule-mix-reject-floor",
        doc_rel=_MARKER_RECIPE_RULE,
        doc_pattern=r"rejection-fraction floor \((0\.\d+)\)",
        src_rel=_MARKER_RECIPE_SRC_ORGANISMS,
        src_pattern=r"^MIX_MAX_REJECT_FRAC = ([0-9.]+)$",
        symbol="MIX_MAX_REJECT_FRAC",
    ),
    _RecipePin(
        label="rule-bandstop-low",
        doc_rel=_MARKER_RECIPE_RULE,
        doc_pattern=r"base ∈ \[([\d.]+), [\d.]+\] nat",
        src_rel=_MARKER_RECIPE_SRC_CALLBACKS,
        src_pattern=r"(?<!\w)low_nats: float = ([\d.]+),",
        symbol="MarkerBandStopCallback.__init__ low_nats default",
    ),
    _RecipePin(
        label="rule-bandstop-high",
        doc_rel=_MARKER_RECIPE_RULE,
        doc_pattern=r"base ∈ \[[\d.]+, ([\d.]+)\] nat",
        src_rel=_MARKER_RECIPE_SRC_CALLBACKS,
        src_pattern=r"(?<!\w)high_nats: float = ([\d.]+),",
        symbol="MarkerBandStopCallback.__init__ high_nats default",
    ),
)


def _norm_val(v: str) -> float | str:
    """Return ``v`` as a float when it parses, else the string unchanged."""
    try:
        return float(v)
    except ValueError:
        return v


def _values_equal(doc_val: str, src_val: str) -> bool:
    """Float-compare when BOTH values parse as floats ('5' == '5.0',
    '0.10' == '0.1'), else exact string equality."""
    a, b = _norm_val(doc_val), _norm_val(src_val)
    if isinstance(a, float) and isinstance(b, float):
        return a == b
    return doc_val == src_val


def _recipe_pin_file_text(
    cache: dict[str, str | None], path: Path, *, normalize: bool
) -> str | None:
    """Read + cache one pinned file for :func:`check_marker_recipe_snippets`.

    ``normalize`` collapses every whitespace run to a single space (the
    line-wrap-immune doc-matching mode); raw mode is for source files.
    Returns None (cached) when the file is missing.
    """
    key = str(path)
    if key not in cache:
        if not path.is_file():
            cache[key] = None
        else:
            text = path.read_text(encoding="utf-8")
            cache[key] = re.sub(r"\s+", " ", text) if normalize else text
    return cache[key]


def check_marker_recipe_snippets(*, repo_root: Path | None = None) -> list[str]:
    """FAIL when a frozen numeric snippet in the marker-training recipe doc
    (docs/marker_training_recipe.md) or its sibling rule
    (.claude/rules/marker-training-recipe.md) disagrees with the code constant
    it cites (#1154; the drift class: a stale frozen number misleads every
    future marker-training planner grounding hyperparameters from the doc).

    Registry-driven (``_MARKER_RECIPE_PINS``) — only registered pins are ever
    evaluated; empirical findings / frozen experiment history in the same
    files are never parsed. Doc patterns run whitespace-normalized
    (line-wrap-immune); src patterns run raw + MULTILINE. Values compare as
    floats when both parse ('5' == '5.0'), else exact strings. Failure modes:
    missing file, doc snippet not found (rot alarm — the pinned prose was
    rephrased), code constant not found (the symbol moved/renamed), ambiguous
    conflicting source matches, and doc-vs-code value mismatch. ``repo_root``
    is a unit-test override hook; production callers pass None. Bundled into
    the no-flags default run.
    """
    root = repo_root if repo_root is not None else _REPO_ROOT
    errors: list[str] = []
    doc_cache: dict[str, str | None] = {}  # normalized doc text (None = missing)
    src_cache: dict[str, str | None] = {}  # raw source text (None = missing)
    for pin in _MARKER_RECIPE_PINS:
        doc_path = root / pin.doc_rel
        src_path = root / pin.src_rel
        doc_text = _recipe_pin_file_text(doc_cache, doc_path, normalize=True)
        src_text = _recipe_pin_file_text(src_cache, src_path, normalize=False)
        if doc_text is None:
            errors.append(
                f"{doc_path}: missing — pin '{pin.label}' binds a snippet here to "
                f"{pin.src_rel}::{pin.symbol}; the marker-recipe doc was moved or "
                f"deleted — update _MARKER_RECIPE_PINS in scripts/workflow_lint.py "
                f"(#1154)."
            )
            continue
        if src_text is None:
            errors.append(
                f"{src_path}: missing — pin '{pin.label}' expects {pin.symbol} here "
                f"(cited by {pin.doc_rel}); the source file was moved or deleted — "
                f"update _MARKER_RECIPE_PINS in scripts/workflow_lint.py (#1154)."
            )
            continue
        doc_vals = re.findall(pin.doc_pattern, doc_text)
        if not doc_vals:
            errors.append(
                f"{doc_path}: pin '{pin.label}': doc snippet not found (pattern "
                f"{pin.doc_pattern!r}) — the pinned prose was rephrased or removed; "
                f"update the snippet or _MARKER_RECIPE_PINS in "
                f"scripts/workflow_lint.py (#1154)."
            )
            continue
        src_vals = re.findall(pin.src_pattern, src_text, re.MULTILINE)
        if not src_vals:
            errors.append(
                f"{src_path}: pin '{pin.label}': code constant {pin.symbol} not "
                f"found (pattern {pin.src_pattern!r}) — the symbol moved or was "
                f"renamed; update _MARKER_RECIPE_PINS in scripts/workflow_lint.py "
                f"(#1154)."
            )
            continue
        if len({_norm_val(v) for v in src_vals}) > 1:
            errors.append(
                f"{src_path}: pin '{pin.label}': ambiguous — {len(src_vals)} "
                f"conflicting matches for {pin.symbol} (values "
                f"{sorted(set(src_vals))}); tighten the pin's src_pattern in "
                f"_MARKER_RECIPE_PINS (#1154)."
            )
            continue
        src_val = src_vals[0]
        for doc_val in dict.fromkeys(doc_vals):  # distinct values, first-seen order
            if not _values_equal(doc_val, src_val):
                errors.append(
                    f"{doc_path}: pin '{pin.label}': doc cites {doc_val!r} but "
                    f"{pin.src_rel}::{pin.symbol} is {src_val!r} — update the doc "
                    f"snippet (or _MARKER_RECIPE_PINS if the binding itself "
                    f"changed) so they agree (#1154)."
                )
    return errors


# `--check-lessons-index`: every `.claude/rules/*.md` (except LESSONS.md
# itself) must have exactly one matching row in `.claude/rules/LESSONS.md`, and
# every row in LESSONS.md must point at an existing rule file. Closes the
# silent-drift class: a rule added/removed without an index update would
# otherwise re-open the #722 load-timing gap (a lesson with no always-on index
# row). The row format is the stable, machine-parseable (#1269 slim — the
# name appears ONCE, bare; the `fires when:` semantics are defined once in the
# LESSONS.md header instead of per-row; `<name>.md` is relative to
# `.claude/rules/`):
#   - <name>.md — <fires-when trigger>
# Full-line match so per-row byte budgets can read `m.group(0)` (#1269).
_LESSONS_ROW_RE = re.compile(
    r"^- (?P<name>[a-z0-9-]+)\.md — (?P<trigger>[^\n]*)$",
    re.MULTILINE,
)


# Leanness cap: ~2600 tokens always-on (7500->8000, #869/#872 coordinated
# raise; #992 restored headroom under the SAME cap via the row-format slim;
# 8000->9600 at the 2026-08-06 CLAUDE.md relocation, which moved ~79 KB of
# orchestrator-only prose out of the always-on body into NINE new
# .claude/rules/ files. Each needs an index row, so the index necessarily
# grows — but the trade is ~1.4 KB of index for ~79 KB of body, a ~21.9K
# token/spawn net WIN. Do NOT read this raise as license for row bloat: the
# per-row cap and the non-row cap are unchanged and still bind.
_LESSONS_MAX_BYTES = 9600
# Early-warning band (#992): a stderr-only advisory WARN once the index
# crosses this, so a near-cap landing is visible a few rows before the
# _LESSONS_MAX_BYTES FAIL (early warning only — advisory, never a FAIL).
_LESSONS_WARN_BYTES = 8800

# Per-row budget (#1269): one bloated row is caught on the row that adds it —
# at edit time, in the grower's own tree — not fleet-wide later at the total
# cap. Byte-counted (the em-dash is multibyte), STRICTLY-GREATER (a row at
# exactly the bound passes). Post-migration live distribution: median 128 /
# mean 145 / p90 203 / max compliant 239 — 280 clears every informative
# trigger with ~40 B slack while catching gotchas-class bloat (438).
_LESSONS_ROW_MAX_BYTES = 280
# Grandfathered oversized legacy rows (#1269, the #986 agent-spec grandfather
# pattern) — LEGACY-ONLY: closed to new entries absent a recorded #1269-class
# justification (a deliberate keep-the-trigger-informative decision, like
# gotchas below). Each cap hugs its measured row: row over its cap -> FAIL
# (regrowth ratchet); cap - actual > the headroom bound -> FAIL (loose/stale
# cap — ratchet DOWN after a trim); actual <= _LESSONS_ROW_MAX_BYTES -> FAIL
# (entry obsolete — remove it).
_LESSONS_ROW_GRANDFATHER_MAX_BYTES: dict[str, int] = {
    # gotchas: highest-traffic rule; row measured 438 B at the #1269
    # migration — a third lossy trigger trim (after #1220) would destroy
    # plan-time discovery value (a further lossy trim was already ruled
    # out at #1269). #1348 added the errorbar/CI figure trigger
    # (row 451 B -> 494 B). Cap = measured + <=40.
    # #1429 added the bootstrap-CI gating/verdict trigger (row 519 B -> 578 B).
    # Cap = measured + <=40.
    # #1411 added the Edit-tool Unicode-literal trigger (row 599 B -> 661 B).
    # Cap = measured + <=40.
    # #1431 added the pilot-gate shape+rc trigger (row 661 B -> 682 B).
    # Cap = measured + <=40.
    # #1435 added the subprocess-per-phase dispatcher trigger (merged with
    # #1431's raise; re-measured row 776 B). Cap = measured + <=40.
    # #1492 added the SAE reference-eval token-pool trigger (row 776 B ->
    # 862 B). Cap = measured + <=40.
    # #1513 added the between-phase cache-reap trigger (merged with #1512's
    # smoke-gate slice-arithmetic clause, +29 B and +27 B on the 862 B base;
    # re-measured row 918 B). Cap = measured + <=40.
    # #1526 added the off-pod-phase upload-set trigger (row 918 B -> 972 B).
    # Cap = measured + <=40.
    # #1640 added the chained smoke-then-full leg out-root residue trigger
    # (row 994 B -> 1040 B). Cap = measured + <=40.
    # #1911 added the count-keyed liveness-gate double-print trigger
    # (row 1048 B -> 1135 B). Cap = measured + <=40.
    "gotchas": 1175,
}
_LESSONS_ROW_GRANDFATHER_MAX_HEADROOM_BYTES = 40

# Non-row scaffolding budget (#1504). Growth control for the always-on index
# is PER-CHANNEL, not a hand-bumped total:
#   - rows: _LESSONS_ROW_MAX_BYTES / _LESSONS_ROW_GRANDFATHER_MAX_BYTES catch
#     a bloated row at edit time in the grower's own tree (#1269), and index
#     parity means a NEW row requires a reviewed .claude/rules/*.md;
#   - non-row scaffolding (header prose, headings, blank lines, row newlines,
#     anything the row grammar does not match): bounded by this FIXED budget;
#   - aggregate: the _LESSONS_WARN_BYTES advisory band + _LESSONS_MAX_BYTES.
# The former TOTAL growth ratchet (_LESSONS_RATCHET_BYTES, #1269) is RETIRED
# (#1504): its same-diff constant bump made every concurrent LESSONS.md
# growth a merge-conflict / trunk-red hazard — 2 Step-10d conflicts (#1335
# PR #1227, #1435 PR #1188), 1 fleet-wide trunk red (#1462), 1 duplicate fix
# pipeline (#1476/#1479) in ~48h — while per-row caps + parity already make
# row growth deliberate. Residual: two individually-green concurrent growths
# can sum past the 8000 cap post-merge, but every such residual-red scenario
# has BOTH growers already inside the 7200 WARN band before pushing (for two
# cap-sized 280-B rows the window opens at base >= ~7,440 > 7,200). Measured
# non-row bytes at retirement: 546 (2026-07-18). 900 leaves headroom for a
# deliberate header note. Raise ONLY for a deliberate header restructure —
# NEVER for row growth (rows never count against this budget; pinned by
# test_check_lessons_index_nonrow_ignores_row_bytes).
_LESSONS_NONROW_MAX_BYTES = 900


def check_lessons_index(  # noqa: C901 -- flat failure-mode ladder (index parity, total cap/warn, non-row budget, per-row caps + grandfather hygiene, #1269/#1504); extracting a branch would just relocate it
    *,
    repo_root: Path | None = None,
    warn_sink: list[str] | None = None,
    nonrow_max_bytes: int | None = _LESSONS_NONROW_MAX_BYTES,
    row_max_bytes: int | None = _LESSONS_ROW_MAX_BYTES,
) -> list[str]:
    """FAIL if `.claude/rules/LESSONS.md` and the `.claude/rules/*.md` set
    diverge OR the index exceeds its byte budgets.

    The always-on index (#739) must name every rule so each lesson is known at
    plan time even before its `paths:` glob matches an open file. Failure
    modes checked: (a) a rule file with no index row, (b) an index row
    with no rule file, (c) a rule name with MORE THAN ONE index row (the
    contract is exactly one matching row per rule — a duplicate would let one
    of the rows silently drift), (d) the index exceeds `_LESSONS_MAX_BYTES`
    (the always-on token budget — the whole point of the index is leanness;
    the Option-B rejected alternative was inlining all rule bodies, paying
    tens of K tokens per call). Failure mode (d) additionally carries an
    advisory WARN band (#992): an index over `_LESSONS_WARN_BYTES` but at or
    under the cap emits an early-warning WARN — stderr-only / advisory, never
    a FAIL — so a near-cap landing is visible a few rows before the next
    addition FAILs; the cap FAIL and WARN both name the largest rows as
    actionable trim targets (#1504); (e) the NON-ROW scaffolding budget
    (#1504) — bytes the row grammar does not claim (header prose, headings,
    blank lines, row newlines, malformed rows) over
    `_LESSONS_NONROW_MAX_BYTES` FAIL; row growth NEVER counts against this
    budget (the per-growth TOTAL ratchet is retired — see the
    `_LESSONS_NONROW_MAX_BYTES` comment); (f) PER-ROW caps (#1269) — a row
    over `_LESSONS_ROW_MAX_BYTES` FAILs (naming the offending row), with the
    `_LESSONS_ROW_GRANDFATHER_MAX_BYTES` legacy exceptions under the same
    over-cap / excess-hug / obsolete-entry hygiene as the #986 agent-spec
    grandfather. `repo_root` is a unit-test override hook; production
    callers pass None (canonical repo root). `nonrow_max_bytes` /
    `row_max_bytes` are TEST-ONLY opt-outs (`None` disables that mode so a
    small synthetic fixture can isolate another failure mode); production
    callers never pass them. `warn_sink` mirrors
    `check_lens_coverage`'s hook: WARNs append there when provided, else go
    to stderr with a ``WARN: `` prefix; WARNs never enter the returned FAIL
    list. Bundled into the no-flags default run.
    """
    root = repo_root if repo_root is not None else _REPO_ROOT
    rules_dir = root / ".claude" / "rules"
    lessons = rules_dir / "LESSONS.md"
    errors: list[str] = []

    def _warn(msg: str) -> None:
        if warn_sink is not None:
            warn_sink.append(msg)
        else:
            sys.stderr.write(f"WARN: {msg}\n")

    if not lessons.is_file():
        errors.append(
            f"{lessons}: missing — the always-on lessons index (#739) must "
            f"exist and index every .claude/rules/*.md file."
        )
        return errors
    raw = lessons.read_bytes()
    text = raw.decode("utf-8")
    row_matches = list(_LESSONS_ROW_RE.finditer(text))
    row_sizes = sorted(
        ((len(m.group(0).encode("utf-8")), m.group("name")) for m in row_matches),
        reverse=True,
    )
    largest_suffix = (
        " Largest rows: " + ", ".join(f"{name} ({b} B)" for b, name in row_sizes[:3]) + "."
        if row_sizes
        else ""
    )
    if len(raw) > _LESSONS_MAX_BYTES:
        errors.append(
            f".claude/rules/LESSONS.md: {len(raw)} bytes exceeds the "
            f"{_LESSONS_MAX_BYTES}-byte leanness cap. The index is "
            f"always-on; trim 'fires when:' triggers until it fits. "
            f"(em-dashes are multibyte; counting in BYTES not chars is "
            f"deliberate.)"
            f"{largest_suffix}"
        )
    elif len(raw) > _LESSONS_WARN_BYTES:
        _warn(
            f".claude/rules/LESSONS.md at {len(raw)}/{_LESSONS_MAX_BYTES} bytes — inside "
            f"the warn band (>{_LESSONS_WARN_BYTES}); slim rows or plan a deliberate cap "
            f"decision before the next addition FAILs.{largest_suffix}"
        )
    # Non-row scaffolding budget (#1504): bytes the row grammar does not
    # claim. Row growth NEVER counts here — growing/adding rows must not
    # require touching this file (the retired ratchet's per-growth bump was
    # the 4-incidents/48h conflict magnet; see _LESSONS_NONROW_MAX_BYTES).
    if nonrow_max_bytes is not None:
        row_total = sum(b for b, _ in row_sizes)
        nonrow = len(raw) - row_total
        if nonrow > nonrow_max_bytes:
            errors.append(
                f".claude/rules/LESSONS.md: {nonrow} non-row scaffolding bytes "
                f"(total {len(raw)} minus {row_total} row bytes) exceed the "
                f"{nonrow_max_bytes}-byte non-row budget "
                f"(_LESSONS_NONROW_MAX_BYTES). Trim header/scaffolding prose "
                f"(a malformed index row also lands here — check the row "
                f"grammar), or — a deliberate header-restructure decision — "
                f"raise _LESSONS_NONROW_MAX_BYTES in the SAME diff. Row "
                f"growth never needs this constant."
            )
    # Count occurrences (not a set) so a name appearing on >1 row is caught —
    # a set comprehension would collapse duplicates and let both the missing
    # and stale set-diffs read empty, silently passing the check (#739 r2).
    # The same pass runs the per-row byte budgets (#1269): the full-line row
    # regex makes `m.group(0)` the whole row.
    index_counts: Counter[str] = Counter()
    for m in row_matches:
        name = m.group("name")
        index_counts[name] += 1
        if row_max_bytes is None:
            continue
        row_bytes = len(m.group(0).encode("utf-8"))
        gf_cap = _LESSONS_ROW_GRANDFATHER_MAX_BYTES.get(name)
        if gf_cap is not None:
            if row_bytes > gf_cap:
                errors.append(
                    f".claude/rules/LESSONS.md: row '{name}' is {row_bytes} "
                    f"bytes, over its grandfather cap ({gf_cap}) — trim the "
                    f"row's trigger back under the cap, or (a deliberate, "
                    f"reviewed keep-the-trigger-informative decision) raise "
                    f"_LESSONS_ROW_GRANDFATHER_MAX_BYTES['{name}'] in the "
                    f"SAME diff, hugging the new size (cap <= size + "
                    f"{_LESSONS_ROW_GRANDFATHER_MAX_HEADROOM_BYTES})."
                )
            elif row_bytes <= row_max_bytes:
                errors.append(
                    f"_LESSONS_ROW_GRANDFATHER_MAX_BYTES['{name}']: row is "
                    f"{row_bytes} bytes (<= the {row_max_bytes}-byte general "
                    f"row cap) and no longer needs grandfathering — remove "
                    f"the entry (ratchet down)."
                )
            elif gf_cap - row_bytes > _LESSONS_ROW_GRANDFATHER_MAX_HEADROOM_BYTES:
                errors.append(
                    f"_LESSONS_ROW_GRANDFATHER_MAX_BYTES['{name}']: cap "
                    f"{gf_cap} sits {gf_cap - row_bytes} bytes above the "
                    f"live row ({row_bytes} bytes) — max headroom is "
                    f"{_LESSONS_ROW_GRANDFATHER_MAX_HEADROOM_BYTES}; lower "
                    f"the cap to <= "
                    f"{row_bytes + _LESSONS_ROW_GRANDFATHER_MAX_HEADROOM_BYTES}."
                )
        elif row_bytes > row_max_bytes:
            errors.append(
                f".claude/rules/LESSONS.md: row '{name}' is {row_bytes} "
                f"bytes, over the {row_max_bytes}-byte per-row cap "
                f"(_LESSONS_ROW_MAX_BYTES) — trim this row's trigger; the "
                f"cap catches one bloated row at addition time instead of a "
                f"fleet-wide total-size FAIL later."
            )
    indexed = set(index_counts)
    rule_files = {p.stem for p in rules_dir.glob("*.md") if p.is_file() and p.name != "LESSONS.md"}
    for missing in sorted(rule_files - indexed):
        errors.append(
            f".claude/rules/LESSONS.md: no index row for rule "
            f"'{missing}' (.claude/rules/{missing}.md). Add a "
            f"'- {missing}.md — <fires-when trigger>' row, "
            f"or reformat an existing old-format row for '{missing}' to "
            f"that format."
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


# `--check-inline-round-duty-mirror` (#1701): the three "Inline
# estimator-validity + record-integrity duties" sentences mirror byte-for-byte
# between CLAUDE.md § "User-chat inline free analysis" and
# .claude/skills/issue/SKILL.md Step 9a-ter. A future editor who updates one
# location and forgets the other silently drifts the mirror; a count-only
# check would slip past a mid-sentence text edit that keeps the anchor
# prefix. Two-part invariant: (a) each of three anchor prefixes appears
# exactly once in EACH file, (b) the full sentence starting at each anchor
# prefix is BYTE-IDENTICAL across the two files.

_INLINE_ROUND_DUTY_ANCHORS: tuple[str, ...] = (
    "(1) BEFORE any ridge",
    "(2) BEFORE launching any re-implemented",
    "(3) When a round REFUTES",
)


def _extract_inline_round_duty_sentence(text: str, anchor: str) -> str | None:
    """Extract the full sentence starting at ``anchor`` through the next
    terminator: the first ``. `` (period + whitespace) OR ``.\\n`` OR a bare
    newline. Returns None if the anchor is not present.

    In-line ``.`` inside a backtick-quoted span does not terminate — e.g.
    ``scripts/issue1345_operator_comparison`` has no closing period. The
    canonical anchor sentences end with ``.`` followed by whitespace or a
    newline; the terminator scan walks past backtick spans.
    """
    idx = text.find(anchor)
    if idx == -1:
        return None
    n = len(text)
    i = idx
    in_backtick = False
    while i < n:
        ch = text[i]
        if ch == "`":
            in_backtick = not in_backtick
        elif ch == "\n" and not in_backtick:
            return text[idx:i]
        elif ch == "." and not in_backtick and i + 1 < n:
            nxt = text[i + 1]
            if nxt.isspace():
                return text[idx : i + 1]
        i += 1
    return text[idx:n]


def check_inline_round_duty_mirror(*, repo_root: Path | None = None) -> list[str]:
    """FAIL if the three "Inline estimator-validity + record-integrity
    duties" anchor sentences drift between CLAUDE.md and
    .claude/skills/issue/SKILL.md.

    Two-part invariant per anchor prefix:
      (a) COUNT: the anchor prefix appears exactly once in EACH file
          (locates the sentence unambiguously — zero hits means the block
          was deleted; more than one means an editor duplicated it).
      (b) BYTE-EQUALITY: the full sentence (anchor prefix through next
          terminator — see ``_extract_inline_round_duty_sentence``) is
          byte-identical across the two files.

    ``repo_root`` is a unit-test override hook; production callers pass
    None. Bundled into the no-flags default run.
    """
    import os as _os

    if repo_root is not None:
        root = repo_root
    else:
        # Env-override so BEHAVIORAL subprocess tests can point the check
        # at a tmp corpus without also having to relocate every other
        # bundled check. Absent: the production _REPO_ROOT.
        env_root = _os.environ.get("EPS_WORKFLOW_LINT_REPO_ROOT")
        root = Path(env_root) if env_root else _REPO_ROOT
    claude_path = root / "CLAUDE.md"
    skill_path = root / ".claude" / "skills" / "issue" / "SKILL.md"
    errors: list[str] = []
    try:
        claude_text = claude_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        errors.append(f"check-inline-round-duty-mirror: {claude_path} not found")
        return errors
    try:
        skill_text = skill_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        errors.append(f"check-inline-round-duty-mirror: {skill_path} not found")
        return errors

    claude_rel = claude_path.relative_to(root) if claude_path.is_relative_to(root) else claude_path
    skill_rel = skill_path.relative_to(root) if skill_path.is_relative_to(root) else skill_path

    for anchor in _INLINE_ROUND_DUTY_ANCHORS:
        claude_count = claude_text.count(anchor)
        skill_count = skill_text.count(anchor)
        if claude_count != 1:
            errors.append(
                f"check-inline-round-duty-mirror: {claude_rel} has {claude_count} "
                f"occurrence(s) of anchor {anchor!r}, expected exactly 1 "
                "(part (a) count invariant)"
            )
        if skill_count != 1:
            errors.append(
                f"check-inline-round-duty-mirror: {skill_rel} has {skill_count} "
                f"occurrence(s) of anchor {anchor!r}, expected exactly 1 "
                "(part (a) count invariant)"
            )
        if claude_count != 1 or skill_count != 1:
            continue  # cannot compare byte-equality without unambiguous anchors
        claude_sent = _extract_inline_round_duty_sentence(claude_text, anchor)
        skill_sent = _extract_inline_round_duty_sentence(skill_text, anchor)
        if claude_sent is None or skill_sent is None:
            errors.append(
                f"check-inline-round-duty-mirror: could not extract anchor sentence "
                f"for {anchor!r} (part (b) byte-equality invariant)"
            )
            continue
        if claude_sent != skill_sent:
            errors.append(
                f"check-inline-round-duty-mirror: anchor sentence for {anchor!r} "
                f"drifted between {claude_rel} and {skill_rel} "
                "(part (b) byte-equality invariant); the three duty sentences "
                "must stay byte-identical across the two files"
            )
    return errors


# `--check-rule-frontmatter-parses` (#1385, from #1348): a `.claude/rules/*.md`
# rule on-demand-loads ONLY through its frontmatter `paths:` globs. A YAML
# parse failure (e.g. an unquoted `description:` containing ': ') silently
# disables the rule — present, LESSONS-indexed, never loads — and a stale
# `globs:` key silently degrades it. Real yaml.safe_load, not a regex
# approximation: the check must fail exactly where the harness fails.


def check_rule_frontmatter_parses(*, repo_root: Path | None = None) -> list[str]:
    """FAIL if any `.claude/rules/*.md` frontmatter block is YAML-broken,
    unterminated, non-mapping, uses the stale `globs:` key, or lacks a
    well-formed `paths:` (non-empty list of non-empty strings).

    Files with no leading `---` line have no frontmatter and are EXEMPT
    (always-on / LESSONS-indexed rules need no `paths:`). Unknown extra keys
    (e.g. `name:`) are tolerated — this validates load-integrity, not a full
    schema. `repo_root` is a unit-test override hook; production callers pass
    None. Bundled into the no-flags default run.
    """
    root = repo_root if repo_root is not None else _REPO_ROOT
    errors: list[str] = []
    for path in sorted((root / ".claude" / "rules").glob("*.md")):
        rel = path.relative_to(root)
        lines = path.read_text(encoding="utf-8").split("\n")
        if not lines or lines[0].strip() != "---":
            continue  # no frontmatter block -> always-on rule, exempt
        end = next((i for i, ln in enumerate(lines[1:], 1) if ln.strip() == "---"), None)
        if end is None:
            errors.append(
                f"{rel}: frontmatter opens with '---' on line 1 but is never "
                f"closed by a second '---' line — the harness cannot split the "
                f"block and the rule never on-demand-loads. Close the block "
                f"(or delete it for an always-on rule)."
            )
            continue
        try:
            data = yaml.safe_load("\n".join(lines[1:end]))
        except yaml.YAMLError as exc:
            reason = " ".join(str(exc).split())
            errors.append(
                f"{rel}: frontmatter is not valid YAML ({reason}) — the rule "
                f"file exists but NEVER loads (the 'rule present but never "
                f"loads' class, #1385). Usual cause: an unquoted "
                f"`description:` containing ': ' — double-quote the scalar."
            )
            continue
        if not isinstance(data, dict):
            errors.append(
                f"{rel}: frontmatter parses to {type(data).__name__}, not a "
                f"key: value mapping — the harness reads mapping frontmatter "
                f"only."
            )
            continue
        if "globs" in data:
            errors.append(
                f"{rel}: frontmatter uses the stale `globs:` key — the project "
                f"convention (CLAUDE.md, LESSONS.md) is `paths:`; rename "
                f"`globs:` -> `paths:`."
            )
            continue
        paths = data.get("paths")
        if paths is None:
            errors.append(
                f"{rel}: frontmatter has no `paths:` key — an on-demand rule "
                f"needs its load-trigger globs; add `paths:`, or drop the "
                f"frontmatter block entirely for an always-on rule."
            )
            continue
        if not isinstance(paths, list) or not paths:
            got = "empty list" if isinstance(paths, list) else type(paths).__name__
            errors.append(
                f"{rel}: `paths:` must be a NON-EMPTY YAML list of glob "
                f"strings (got {got}) — a mis-shaped `paths:` never matches, "
                f"so the rule never loads."
            )
            continue
        bad = [p for p in paths if not isinstance(p, str) or not p.strip()]
        if bad:
            errors.append(
                f"{rel}: `paths:` entries must be non-empty strings; got "
                f"{bad!r}. Quote each glob (bare `yes`/`no`/numbers/null "
                f"parse as non-strings)."
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

# A grandfather cap must hug the measured size: cap - size <= this bound
# (the documented "measured + <=3 KB margin" convention, mechanized by #986).
# STRICTLY-GREATER like the other thresholds (headroom exactly 3_000 passes).
# A larger headroom means a loose cap-raise (defeats the regrowth ratchet) or
# a stale cap after a trim (ratchet DOWN when trimmed) — both FAIL. Scope:
# this bounds BANKED SLACK only — a reviewed growth+cap-raise in one commit
# still passes; the check forces growth into a visible dict edit, it does not
# approve it.
AGENT_SPEC_GRANDFATHER_MAX_HEADROOM_BYTES = 3_000

# Grandfather-ratchet caps for agent specs still above AGENT_SPEC_FAIL_BYTES.
# Each cap = measured size + <=3 KB margin (headroom mechanically enforced —
# see AGENT_SPEC_GRANDFATHER_MAX_HEADROOM_BYTES; post-#829 for the first two
# entries; at the #838 FAIL tightening 70K -> 40K for the rest); a
# grandfathered file FAILs above its cap (regrowth ratchet) and FAILs as stale
# once it drops to <= AGENT_SPEC_FAIL_BYTES ("remove the entry"). Ratchet DOWN
# when trimmed. planner.md and critic.md are deliberately NOT grandfathered
# (#838): both were structurally trimmed to <=20 KB, so regrowth on the two
# incident files is a commit-time FAIL.
AGENT_SPEC_SIZE_GRANDFATHER: dict[str, int] = {
    # clean-result-critic.md: split to clean-result-critic-lens-reference.md
    # (#1159) — no longer grandfathered (slim spec is under the FAIL threshold).
    # the rest measured at the #838 tightening (2026-07-02), caps = measured
    # + <=3 KB; each names a future trim direction, none is licensed to grow
    # measured 139,109 B post-#2002 (Step 0.6 coordinating paragraph naming
    # the Resume-matrix + real-production-out-root-unit smoke coverage
    # requirements as `smoke-run-missing` blocker-tagged coverage checks;
    # incident driver: #1947 P0/P4/P5 + #1315 r6 + #1112 r6; cap = measured
    # + ~1.2 KB. Prior: 137_400 —
    # measured 135,813 B post-#1805 (Step 4 round-new-script no-flags lint
    # duty — executable diff-adds trigger gate in the fenced pre-pass block
    # + attribution / waiver-remedy / stale-family prose — plan-mandated
    # growth; cap = measured + ~1.5 KB.
    # Prior:
    # 134_200 — measured 133,188 B post-#1743 (task-bound verdict post
    # switched to the --file channel + MANDATORY exact-kind read-back
    # duty),
    # 132_500 — measured 131,378 B post-Step-10d-merge (task #1727 Step
    # 0.70 smoke-variable gating gate + task #1716 Step 4 ruff-policy pin
    # + L99 style-bullet + Step 0.5 pin-invocation marker-shape check —
    # both landings STACKED at Step 10d merge),
    # 130_000 — measured 128,507 B post-#1727 unmerged (Step 0.70
    # alone atop pre-#1716 base — trigger + sub-checks (1)/(2)/(3) +
    # waiver form + verdict routing with the smoke-var-ungated /
    # smoke-var-orphan-full FAIL tags),
    # 128_000 — measured 127,227 B post-#1716 (Step 4 ruff-policy pin
    # + L99 style-bullet clause + Step 0.5 pin-invocation marker-shape
    # check — plan-mandated growth atop main's #1728 Step 3.75 grep
    # verification + #1726 Step 3.6 T2-trigger),
    # 125_500 — measured 124,356 B post-merge (main #1726 Step 3.6
    # T2-trigger additions + #1728 Step 3.75 symbol-rename grep
    # verification — plan-mandated growth binding both the
    # crash-fix-rounds symbol-rename whole-tree grep duty at
    # code-review AND main's per-unit progress-line verdict-routing),
    # 124_400 — measured 123,275 B post-#1728 unmerged (Step 3.75 alone),
    # 122_400 — measured 121,782 B post-#1726 unmerged (Step 3.6 T2 count
    # trigger + 3-part Check incl. per-unit progress-line item 3 +
    # verdict-routing rewrite),
    # 121_300 — measured 120,709 B post-#1693 (Step 0.69 phase-idempotency
    # + inter-phase-contract gate atop #1692's Step 0.55 SHAPE-check
    # binding), 112,500 — measured 112,177 B post-#1692 (Step 0.55
    # SHAPE-check binding: per-arm attestation-row consistency +
    # import-resolution three-shape gate for the smoke-architecture-check
    # marker), 110,300 — measured 109,583 B post-#1449 (Step 0.65 plan-glob
    # vs uploader-eligibility parity sub-check), 108,000 — measured
    # 106,853 B post-#1397 (Step 2 fit-loop batched-helper naming
    # paragraph), 105,000 — measured 104,235 B post-#1317 (Step 4.6
    # Gate-scope line verification), 101,500 — measured 100,555 B
    # post-#1254 (Step 3.9 degenerate-statistic check, observed-vs-null
    # reads), 99,000 — measured 98,126 B post-#1230 (Step 6 durability-pin
    # shipping duty), 97,000 — measured 96,072 B post-#1119, 95,000 —
    # measured 94,126 B post-#1115)
    # measured 98,526 B post the 2026-08-05 compaction: Step 0.5-0.70
    # gate-stack detail relocated to
    # .claude/rules/code-reviewer-section-reference.md (#1159 mechanism);
    # the spec keeps per-gate trigger + blocker-tag + lint-pinned tokens
    # + § pointer lines. Cap = measured + ~1 KB.
    "code-reviewer.md": 99_500,
    # measured 74,082 B post-#1447 (family-enumeration sync: the two
    # byte/bit verdict rows widened to the -exact / bitwise / X-for-X
    # tail — plan-mandated growth; cap = measured + ~1.1 KB. Prior:
    # 74,000 — measured 73,408 B post-#1159 (Step 2 dual-source read
    # contract: lens rubrics from clean-result-critic-lens-reference.md,
    # report schema from the slim agent spec), 73,000 — measured
    # 72,229 B post-#1056, 72,000 post-#1050 r2, 71,000 post-#1050 r1,
    # 60,554 B pre-#1050; 75,200 pre-description-rewrite — measured
    # 71,784 B after the 2026-08-05 frontmatter-description compaction)
    # measured 49,241 B post the 2026-08-05 compaction: the 15 verdict-
    # template lens slots slimmed to heading + findings-contract lines (the
    # composed prompt already inlines the full lens reference verbatim via
    # the {{INLINED ...}} placeholders). Cap = measured + ~1 KB.
    # (48_400 post the composer-common hard-rule dedupe, measured 47,431 B.)
    "codex-clean-result-critic.md": 48_400,
    # measured 61,503 B post-#1805 (Step 4 copy-list bullet extension:
    # round-new-script no-flags lint duty, no-uv static hub-verify
    # adaptation — plan-mandated growth; cap = measured + ~1.3 KB. Prior:
    # 60_800 — measured 59,576 B post-#1693 (Step 0.69 mirror paragraph
    # pointing at code-reviewer.md's phase-idempotency +
    # inter-phase-contract gate), 59,200 —
    # measured 58,271 B post-#1438 (Step 0.9 copy-list bullet + inlined-
    # rubric 0.9 slot + Blocker-tags data-access-blocked entry),
    # 56,800 — measured 55,870 B post-#1380 (Step 4.6 copy-list bullet +
    # inlined-rubric 4.6 slot + Blocker-tags 4.6-presence), 53,300 —
    # measured 52,361 B post-#1254, 51,600 — measured 50,642 B post-#948,
    # 47,930 B post-#881)
    # measured 49,270 B post the 2026-08-05 compaction: the Step 2 copy-list
    # bullets deduped against the code-reviewer.md text the composer copies
    # verbatim at compose time (each bullet keeps the section name, the
    # lint/test-pinned tokens, and the Codex-specific adaptations only).
    # Cap = measured + ~1 KB. (47_900 post the composer-common hard-rule
    # dedupe, measured 46,904 B.)
    "codex-code-reviewer.md": 47_900,
    # measured 84,278 B post-#2002 (Resume-matrix + real production
    # out-root unit smoke-contract requirements + matching marker
    # `notes:` sub-blocks; incident driver: #1947 P0/P4/P5 + #1315 r6 +
    # #1112 r6 resume-branch defect concentration — five persisted
    # agent memories promoted to gated contract; cap = measured +
    # ~1.2 KB. Prior: 80_500 — measured 76,274 B post-#1692 (item 5
    # Axis 1 import-resolution leg, Axis 2 per-arm resolution
    # attestation, PASS_PARTIAL verdict + post-marker template
    # extension — plan-mandated growth; cap = measured + ~0.23 KB,
    # with condensing sweep across older Rationale / incident prose to
    # stay near budget. Prior: 74,500 — measured 74,240 B post-#1682
    # (Report Format SHA-verbatim rule), 74,000 — measured 73,554 B
    # post-#1572 (step-10 staged-index verification pointer), 73,000 —
    # measured 72,240 B post-#1449 (After-implementation step-7
    # plan-glob parity self-check), 72,000 — measured 71,114 B
    # post-#1409 (data-dependent-gates smoke duty in checklist item 3
    # + item-5 cross-ref), 69,800 — measured 68,888 B post-#1384
    # (per-arm-class smoke-coverage clause), 67,900 — measured
    # 67,472 B post-#1363, 67,400 — measured 66,574 B post-#1349,
    # 66,300 — measured 65,548 B post-#1311)
    # measured 64,480 B post the 2026-08-05 compaction: Before-writing-code
    # item 5 (smoke/sweep parity) + After-implementation items 3 + 7 detail
    # relocated to .claude/rules/experiment-implementer-section-reference.md
    # (#1159 mechanism); pinned anchors/tokens stay in-spec. Cap = measured
    # + ~1 KB.
    "experiment-implementer.md": 65_500,
    # measured 79,611 B post-#1720 (§ Local runs pre-emptive NOT-RUN escape
    # for Step 9c-selected slow tests — mirrors implementer.md L174; ~500 B
    # growth; cap = measured + ~0.9 KB. Prior: 79_500 —
    # measured 74,867 B post-#1702 (Responsibility 2 --env-pin composition
    # sub-bullet threading --env-pin KEY=VALUE on --workload-cmd launches,
    # #1669 channel merge + #1586 wedge-failover WandB incident — plan-
    # mandated growth on top of #1698; cap = measured + ~0.53 KB. Prior:
    # 74,400 — measured 73,872 B post-#1698 (Contract scope H2 — the
    # already-bootstrapped-pod 60s budget + fresh-provision refusal;
    # fence-field derivation recipe — gcloud maxRunDuration + RunPod
    # audit-cron ttl_days disclosure — with poller_timeout= separated
    # from fence= in the epm:run-launched marker template; #1689 R8
    # launch-path fixes 3 + 4), 67,500 — measured 66,921 B post-#1416
    # (Pre-Launch step 9 foreign-tenant memory.used read), 66,500 —
    # measured 65,540 B post-#1081 r2 (D3 crash-fix-relaunch addendum:
    # disposition-conditional resume-glob confirm), 65,500 — measured
    # 62,672 B)
    # measured 76,828 B post-#1800 (Before Running item 4b output-persist
    # pre-launch gate — the #1739 dispatch-time backstop, output-side
    # sibling of the item-4 input gate; plan-mandated growth; cap =
    # measured + ~0.87 KB — LANDING bytes, per #1753.)
    # measured 65,619 B post the 2026-08-05 compaction: bootstrap probe, GCP
    # salvage, Before-Running item-4 gate detail, and the vLLM hang triad
    # relocated to .claude/rules/experimenter-section-reference.md (#1159
    # mechanism); the crash-fix-relaunch paragraph + run-launched fence
    # tokens stay in-spec verbatim. Cap = measured + ~1 KB.
    "experimenter.md": 66_600,
    # measured 49,740 B post-#1115 (read-hygiene context-budget section —
    # plan-mandated growth; cap = measured + <=~1 KB. Prior: 49,000 —
    # measured 48,197 B post-#1102)
    "methodology-writer.md": 50_700,
    # measured 46,785 B post-#1618 (unmapped-pod triage + non-EPS pod-cost
    # directive + Mode-2 audit template relocated to
    # .claude/rules/pm-audit-reference.md — #829 trim after the 5d84120ac9
    # overage to 47,861; cap UNCHANGED = measured + ~0.2 KB. Prior:
    # measured 46,187 B post-#1082 (negative-existence search recipe),
    # 43,500 / 40,990 B)
    "research-pm.md": 47_000,
    # measured 51,638 B post-#1834 (marker-materialized producer-schema
    # rule bullet: remediation naming the producer-schema duty +
    # schema-mismatched canonical file is a GAP/FAIL — plan-mandated
    # growth; cap = measured + ~1.2 KB. Prior: 51,500 — measured
    # 50,741 B post-#1535 (Step 2.7 declared-off-pod outputs
    # sub-rule + Step 2.8 off_pod_phases reads arm — plan-mandated growth;
    # cap = measured + ~0.8 KB. Prior: 47,800 — measured 46,830 B post-#1115)
    "upload-verifier.md": 52_800,
}


def check_agent_spec_size(  # noqa: C901 -- flat per-entry hygiene ladder (stale/retired/headroom, #986); extracting a branch would just relocate it
    *, repo_root: Path | None = None, warn_sink: list[str] | None = None
) -> list[str]:
    """WARN/FAIL agent specs (`.claude/agents/*.md`) over the size budget (#829).

    Every agent spec is loaded whole on each spawn, so bytes here are a
    per-invocation token cost. Semantics (all thresholds STRICTLY-GREATER):
    size > ``AGENT_SPEC_FAIL_BYTES`` FAILs unless the file is grandfathered in
    ``AGENT_SPEC_SIZE_GRANDFATHER`` (then it WARNs while under its per-file cap
    and FAILs above it — the regrowth ratchet); size > ``AGENT_SPEC_WARN_BYTES``
    WARNs. Grandfather hygiene FAILs a stale entry (file missing), an entry
    whose file dropped to <= the FAIL threshold (remove the entry — ratchet
    down), and an entry whose cap sits more than
    ``AGENT_SPEC_GRANDFATHER_MAX_HEADROOM_BYTES`` above the live file size
    (loose cap / stale cap — lower it), and a config self-check FAILs any cap
    <= the FAIL threshold. WARNs
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
    # still NEED grandfathering (size > FAIL threshold) AND whose cap hugs the
    # measured size (headroom <= AGENT_SPEC_GRANDFATHER_MAX_HEADROOM_BYTES —
    # the regrowth ratchet is meaningless under a loose cap; #986).
    for gf_name, cap in sorted(AGENT_SPEC_SIZE_GRANDFATHER.items()):
        gf_path = agents_dir / gf_name
        if not gf_path.is_file():
            errors.append(
                f"AGENT_SPEC_SIZE_GRANDFATHER['{gf_name}']: stale grandfather "
                f"entry — .claude/agents/{gf_name} does not exist; remove the "
                f"entry."
            )
            continue
        gf_size = gf_path.stat().st_size
        if gf_size <= AGENT_SPEC_FAIL_BYTES:
            errors.append(
                f"AGENT_SPEC_SIZE_GRANDFATHER['{gf_name}']: "
                f".claude/agents/{gf_name} is {gf_size} bytes "
                f"(<= {AGENT_SPEC_FAIL_BYTES}) and no longer needs "
                f"grandfathering — remove the entry (ratchet down)."
            )
        elif cap - gf_size > AGENT_SPEC_GRANDFATHER_MAX_HEADROOM_BYTES:
            errors.append(
                f"AGENT_SPEC_SIZE_GRANDFATHER['{gf_name}']: cap {cap} sits "
                f"{cap - gf_size} bytes above .claude/agents/{gf_name} "
                f"({gf_size} bytes) — max headroom is "
                f"{AGENT_SPEC_GRANDFATHER_MAX_HEADROOM_BYTES} bytes (cap = "
                f"measured + <=3 KB); lower the cap to <= "
                f"{gf_size + AGENT_SPEC_GRANDFATHER_MAX_HEADROOM_BYTES}, or "
                f"remove the entry if the file no longer needs grandfathering."
            )

    return errors


# Agent-memory index size budget (#1891): every .claude/agent-memory/*/MEMORY.md
# is loaded WHOLE on each spawn of the owning agent (`memory: project`), and the
# harness loader TRUNCATES the index at ~25,000 bytes (measured-by-report,
# 2026-07-30: a 41,137 B index was truncated to ~24.4 KB — the trailing ~39%,
# exactly where NEW lessons append, silently dropped on every spawn). FAIL sits
# 1,000 B BELOW the measured truncation so the check fires BEFORE any silent
# loss; WARN leaves ~5 KB of append room before FAIL. Thresholds are
# STRICTLY-GREATER (a file at exactly the threshold passes). No grandfather
# dict: all live offenders were curated under WARN in the same change that
# introduced this check, so the ratchet starts clean. Per-entry files
# (feedback_*.md etc.) load on demand and are deliberately OUT of scope.
AGENT_MEMORY_INDEX_WARN_BYTES = 20_000
AGENT_MEMORY_INDEX_FAIL_BYTES = 24_000

# gotchas.md size budget (2026-08-05 compaction): `.claude/rules/gotchas.md` is
# machine-APPENDED by scripts/consolidate_lessons.py (failure-lesson promotion),
# so it regrows without bound between hand trims — it reached 324 KB before the
# 2026-08-05 trim to ~199 KB. The cap is the backstop that forces a periodic
# re-trim (per entry: keep the operative rule + diagnostic signature + fix +
# bare #N citations; drop dates, session ids, wall-times, fix-status
# archaeology). Thresholds STRICTLY-GREATER (exactly-at passes); NO grandfather
# table — the file was trimmed under WARN in the same change that introduced
# the check.
GOTCHAS_SIZE_WARN_BYTES = 200_000
GOTCHAS_SIZE_FAIL_BYTES = 250_000

_AGENT_MEMORY_CURATION_RECIPE = (
    "curate it: trim each index hook to ~1 line (<=~150 chars), move the "
    "detail into the pointed-to per-entry file, and merge duplicate/sibling "
    "rows (see #1891)"
)


def check_agent_memory_index_size(
    *, repo_root: Path | None = None, warn_sink: list[str] | None = None
) -> list[str]:
    """WARN/FAIL agent-memory indexes (`.claude/agent-memory/*/MEMORY.md`) over
    the loader-truncation size budget (#1891).

    Every MEMORY.md index is loaded whole on each spawn of its owning agent and
    the loader truncates at ~25,000 bytes, silently dropping the tail — where
    new lessons append. Semantics (both thresholds STRICTLY-GREATER): size >
    ``AGENT_MEMORY_INDEX_FAIL_BYTES`` FAILs with the curation recipe; size >
    ``AGENT_MEMORY_INDEX_WARN_BYTES`` WARNs. Only the per-agent index files
    (``MEMORY.md``) are scanned — per-entry files load on demand and are out of
    scope. Missing ``.claude/agent-memory/`` dir FAILs (parity with
    ``check_agent_spec_size``'s missing-dir behavior). WARNs go to
    ``warn_sink`` when provided (unit-test hook), else stderr with a ``WARN: ``
    prefix; WARNs never enter the returned FAIL list. ``repo_root`` is a
    unit-test override; production callers pass None. Bundled into the
    no-flags default run.
    """
    root = repo_root if repo_root is not None else _REPO_ROOT
    memory_dir = root / ".claude" / "agent-memory"
    errors: list[str] = []

    def _warn(msg: str) -> None:
        if warn_sink is not None:
            warn_sink.append(msg)
        else:
            sys.stderr.write(f"WARN: {msg}\n")

    if not memory_dir.is_dir():
        errors.append(
            f"{memory_dir}: missing — the agent-memory dir must exist for the "
            f"agent-memory index size-budget check (#1891)."
        )
        return errors

    for path in sorted(memory_dir.glob("*/MEMORY.md")):
        if not path.is_file():
            continue
        size = path.stat().st_size
        rel = f".claude/agent-memory/{path.parent.name}/MEMORY.md"
        if size > AGENT_MEMORY_INDEX_FAIL_BYTES:
            errors.append(
                f"{rel}: {size} bytes exceeds the "
                f"{AGENT_MEMORY_INDEX_FAIL_BYTES}-byte agent-memory index FAIL "
                f"threshold (the loader truncates the always-loaded index at "
                f"~25,000 bytes, silently dropping the newest lessons) — "
                f"{_AGENT_MEMORY_CURATION_RECIPE}."
            )
        elif size > AGENT_MEMORY_INDEX_WARN_BYTES:
            _warn(
                f"{rel}: {size} bytes exceeds the "
                f"{AGENT_MEMORY_INDEX_WARN_BYTES}-byte agent-memory index WARN "
                f"budget (FAIL above {AGENT_MEMORY_INDEX_FAIL_BYTES}; the "
                f"loader truncates at ~25,000 bytes) — "
                f"{_AGENT_MEMORY_CURATION_RECIPE}."
            )

    return errors


def check_gotchas_size(
    *, repo_root: Path | None = None, warn_sink: list[str] | None = None
) -> list[str]:
    """WARN/FAIL `.claude/rules/gotchas.md` over the regrowth size budget.

    gotchas.md is machine-appended by ``scripts/consolidate_lessons.py``
    (failure-lesson promotion), so it regrows without bound between hand
    trims; this check is the backstop that forces a periodic re-trim.
    Semantics (both thresholds STRICTLY-GREATER): size >
    ``GOTCHAS_SIZE_FAIL_BYTES`` FAILs with the trim recipe; size >
    ``GOTCHAS_SIZE_WARN_BYTES`` WARNs. No grandfather table. A missing
    gotchas.md FAILs (parity with ``check_agent_spec_size``'s missing-dir
    behavior — the file is a load-bearing rules surface). WARNs go to
    ``warn_sink`` when provided (unit-test hook), else stderr with a
    ``WARN: `` prefix; WARNs never enter the returned FAIL list.
    ``repo_root`` is a unit-test override; production callers pass None.
    Bundled into the no-flags default run.
    """
    root = repo_root if repo_root is not None else _REPO_ROOT
    path = root / ".claude" / "rules" / "gotchas.md"
    errors: list[str] = []

    def _warn(msg: str) -> None:
        if warn_sink is not None:
            warn_sink.append(msg)
        else:
            sys.stderr.write(f"WARN: {msg}\n")

    if not path.is_file():
        errors.append(
            f"{path}: missing — .claude/rules/gotchas.md must exist for the "
            f"gotchas size-budget check."
        )
        return errors

    size = path.stat().st_size
    trim_recipe = (
        "re-trim per the entry editorial policy: keep the operative rule + "
        "diagnostic signature + fix + bare #N citations; drop dates, session "
        "ids, wall-times, and fix-status archaeology (resolve to current "
        "state); collapse superseded/FIXED entries to one line"
    )
    if size > GOTCHAS_SIZE_FAIL_BYTES:
        errors.append(
            f".claude/rules/gotchas.md: {size} bytes exceeds the "
            f"{GOTCHAS_SIZE_FAIL_BYTES}-byte gotchas FAIL threshold (the file "
            f"is machine-appended and must be periodically re-trimmed) — "
            f"{trim_recipe}."
        )
    elif size > GOTCHAS_SIZE_WARN_BYTES:
        _warn(
            f".claude/rules/gotchas.md: {size} bytes exceeds the "
            f"{GOTCHAS_SIZE_WARN_BYTES}-byte gotchas WARN budget (FAIL above "
            f"{GOTCHAS_SIZE_FAIL_BYTES}) — {trim_recipe}."
        )

    return errors


# Skill-doc size budget (2026-08-05 compaction, the t3b guardrail): every
# `.claude/skills/**/*.md` (SKILL.md + support docs) is loaded whole into the
# invoking agent's context on Skill invocation, and skills had NO size cap —
# which is how issue/SKILL.md reached 916 KB before the 2026-08-05 trim.
# Semantics mirror the agent-spec ratchet (thresholds STRICTLY-GREATER;
# grandfather cap = measured + <= 3 KB headroom, FAIL above the cap, remove
# the entry once the file drops to <= the FAIL threshold). Two exemption
# classes (never sized): GENERATED files, whose bytes are owned by their
# generator (`issue/markers.md` is emitted from workflow.yaml via
# `--emit-tables` — the compaction lever is workflow.yaml prose, and
# hand-trimming the derived table is prohibited); and DATA-not-instructions
# directories (exemplars / templates / lw-post-examples) — reference corpora
# read selectively, not playbooks loaded to be followed.
SKILL_DOC_WARN_BYTES = 40_000
SKILL_DOC_FAIL_BYTES = 60_000
SKILL_DOC_GRANDFATHER_MAX_HEADROOM_BYTES = 3_000

# Paths relative to .claude/skills/ (POSIX separators).
SKILL_DOC_GENERATED_EXEMPT: frozenset[str] = frozenset({"issue/markers.md"})

# Any doc with one of these path SEGMENTS under .claude/skills/ is exempt.
SKILL_DOC_EXEMPT_DIR_SEGMENTS: frozenset[str] = frozenset(
    {"exemplars", "templates", "lw-post-examples"}
)

# Grandfather-ratchet caps for skill docs still above SKILL_DOC_FAIL_BYTES,
# keyed by path relative to .claude/skills/. Each cap = measured size at the
# 2026-08-05 introduction + <= 3 KB margin; a grandfathered file FAILs above
# its cap (regrowth ratchet) and FAILs as stale once it drops to
# <= SKILL_DOC_FAIL_BYTES ("remove the entry"). Ratchet DOWN when trimmed
# (> 3 KB headroom after a trim FAILs until the cap is lowered in the same
# change). Each entry names its trim direction; none is licensed to grow.
SKILL_DOC_SIZE_GRANDFATHER: dict[str, int] = {
    # measured 897,435 B post-t3b story->citation trim; the remaining mass is
    # the judgment tranche (bash-block extraction to step10d_guards.sh-style
    # scripts, 9a-quater legacy-path stub, GCP rollback-prose relocation).
    "issue/SKILL.md": 900_000,
    # measured 104,141 B; v3/v2 grandfather sections (~36 KB) compress after
    # the v3 body drain.
    "clean-results/SPEC.md": 106_900,
    # measured 87,195 B; problem-sweep prose + living-docs passes are the
    # trim direction.
    "daily/SKILL.md": 90_000,
    # measured 68,032 B; Phase 1 planner-prompt restatement of planner.md is
    # the trim direction.
    "adversarial-planner/SKILL.md": 70_900,
}


def check_skill_doc_size(  # noqa: C901 -- flat per-entry hygiene ladder, mirroring check_agent_spec_size
    *, repo_root: Path | None = None, warn_sink: list[str] | None = None
) -> list[str]:
    """WARN/FAIL skill docs (`.claude/skills/**/*.md`) over the size budget.

    A skill doc is loaded whole on invocation, so bytes here are a
    per-invocation token cost — and skills had no cap (the 916 KB
    issue/SKILL.md is the founding incident). Semantics (all thresholds
    STRICTLY-GREATER): size > ``SKILL_DOC_FAIL_BYTES`` FAILs unless the file
    is grandfathered in ``SKILL_DOC_SIZE_GRANDFATHER`` (then it WARNs while
    under its per-file cap and FAILs above it — the regrowth ratchet); size >
    ``SKILL_DOC_WARN_BYTES`` WARNs. Exempt: ``SKILL_DOC_GENERATED_EXEMPT``
    paths (regenerate-don't-edit derived tables) and docs under a
    ``SKILL_DOC_EXEMPT_DIR_SEGMENTS`` directory (data, not instructions).
    Grandfather hygiene FAILs a stale entry (file missing), an entry whose
    file dropped to <= the FAIL threshold (remove the entry — ratchet down),
    an entry whose cap sits more than
    ``SKILL_DOC_GRANDFATHER_MAX_HEADROOM_BYTES`` above the live file size
    (loose/stale cap — lower it), and a config self-check FAILs any cap <=
    the FAIL threshold or a grandfather/exempt contradiction. WARNs go to
    ``warn_sink`` when provided (unit-test hook), else stderr with a
    ``WARN: `` prefix; WARNs never enter the returned FAIL list.
    ``repo_root`` is a unit-test override; production callers pass None.
    Bundled into the no-flags default run.
    """
    root = repo_root if repo_root is not None else _REPO_ROOT
    skills_dir = root / ".claude" / "skills"
    errors: list[str] = []

    def _warn(msg: str) -> None:
        if warn_sink is not None:
            warn_sink.append(msg)
        else:
            sys.stderr.write(f"WARN: {msg}\n")

    if not skills_dir.is_dir():
        errors.append(
            f"{skills_dir}: missing — the skills dir must exist for the "
            f"skill-doc size-budget check."
        )
        return errors

    def _exempt(rel: str) -> bool:
        if rel in SKILL_DOC_GENERATED_EXEMPT:
            return True
        return any(seg in SKILL_DOC_EXEMPT_DIR_SEGMENTS for seg in rel.split("/")[:-1])

    # Config self-check FIRST: a cap at/below the FAIL threshold is
    # meaningless, and a grandfathered-but-exempt path is a contradiction.
    for gf_rel, cap in sorted(SKILL_DOC_SIZE_GRANDFATHER.items()):
        if cap <= SKILL_DOC_FAIL_BYTES:
            errors.append(
                f"SKILL_DOC_SIZE_GRANDFATHER['{gf_rel}']: cap {cap} — cap "
                f"must exceed SKILL_DOC_FAIL_BYTES ({SKILL_DOC_FAIL_BYTES}); "
                f"raise the cap or remove the entry."
            )
        if _exempt(gf_rel):
            errors.append(
                f"SKILL_DOC_SIZE_GRANDFATHER['{gf_rel}']: path is exempt "
                f"(generated / data dir) — an exempt doc is never sized; "
                f"remove the entry."
            )

    for path in sorted(skills_dir.rglob("*.md")):
        if not path.is_file():
            continue
        rel = path.relative_to(skills_dir).as_posix()
        if _exempt(rel):
            continue
        size = path.stat().st_size
        if size > SKILL_DOC_FAIL_BYTES:
            cap = SKILL_DOC_SIZE_GRANDFATHER.get(rel)
            if cap is not None:
                if size > cap:
                    errors.append(
                        f".claude/skills/{rel}: {size} bytes exceeds its "
                        f"grandfather ratchet cap ({cap} bytes) — the doc "
                        f"regrew past its recorded post-trim size; trim it "
                        f"back (story->citation compression, relocate "
                        f"reference material to .claude/rules/)."
                    )
                else:
                    _warn(
                        f".claude/skills/{rel}: {size} bytes — grandfathered; "
                        f"{cap - size} bytes under its cap ({cap})."
                    )
            else:
                errors.append(
                    f".claude/skills/{rel}: {size} bytes exceeds the "
                    f"{SKILL_DOC_FAIL_BYTES}-byte skill-doc FAIL threshold — "
                    f"trim it (story->citation compression, relocate "
                    f"reference material to .claude/rules/), or add a "
                    f"measured+<=3KB grandfather entry with a named trim "
                    f"direction."
                )
        elif size > SKILL_DOC_WARN_BYTES:
            _warn(
                f".claude/skills/{rel}: {size} bytes exceeds the "
                f"{SKILL_DOC_WARN_BYTES}-byte skill-doc WARN budget "
                f"(FAIL above {SKILL_DOC_FAIL_BYTES})."
            )

    # Grandfather-entry hygiene (mirrors the agent-spec ratchet, #986).
    for gf_rel, cap in sorted(SKILL_DOC_SIZE_GRANDFATHER.items()):
        gf_path = skills_dir / gf_rel
        if not gf_path.is_file():
            errors.append(
                f"SKILL_DOC_SIZE_GRANDFATHER['{gf_rel}']: stale grandfather "
                f"entry — .claude/skills/{gf_rel} does not exist; remove the "
                f"entry."
            )
            continue
        if _exempt(gf_rel):
            continue  # already reported by the config self-check above
        gf_size = gf_path.stat().st_size
        if gf_size <= SKILL_DOC_FAIL_BYTES:
            errors.append(
                f"SKILL_DOC_SIZE_GRANDFATHER['{gf_rel}']: "
                f".claude/skills/{gf_rel} is {gf_size} bytes "
                f"(<= {SKILL_DOC_FAIL_BYTES}) and no longer needs "
                f"grandfathering — remove the entry (ratchet down)."
            )
        elif cap - gf_size > SKILL_DOC_GRANDFATHER_MAX_HEADROOM_BYTES:
            errors.append(
                f"SKILL_DOC_SIZE_GRANDFATHER['{gf_rel}']: cap {cap} sits "
                f"{cap - gf_size} bytes above .claude/skills/{gf_rel} "
                f"({gf_size} bytes) — max headroom is "
                f"{SKILL_DOC_GRANDFATHER_MAX_HEADROOM_BYTES} bytes (cap = "
                f"measured + <=3 KB); lower the cap to <= "
                f"{gf_size + SKILL_DOC_GRANDFATHER_MAX_HEADROOM_BYTES}, or "
                f"remove the entry if the file no longer needs grandfathering."
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
            tree = _cached_parse(path, text)
            if tree is None:
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
    ``.claude/rules/LESSONS.md`` (the ``- <name>.md — <trigger>`` rows) with
    NO row in the map — a lesson silently uncovered. ``GAP:`` rows PASS (an honest "no v2
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


# ── --check-section-reference-pointers (#1159) ───────────────────────────────
# Reference-file suffixes: a .claude/rules/*.md whose filename ends with one of
# these is an agent-owned relocated-section reference (the #838/#850/#1159
# split shape); the owning agent spec is .claude/agents/<stem-minus-suffix>.md.
_SECTION_REFERENCE_SUFFIXES: tuple[str, ...] = ("-section-reference.md", "-lens-reference.md")


def _ws_norm(text: str) -> str:
    """Collapse every whitespace run in ``text`` to a single space (strip ends)."""
    return " ".join(text.split())


def _fence_aware_headings(text: str) -> list[tuple[int, str]]:
    """Return ``(level, heading_text)`` for every non-fenced H2/H3 in ``text``.

    A line matching ``^(```` ``` ````|``~~~``)`` toggles fence state (the
    prefix match covers info-string openers like ```` ```bash ````), so fenced
    pseudo-headings (e.g. ``# ...`` bash comments) are skipped. H1 is excluded
    (the file title); H4+ are intra-section structure.
    """
    headings: list[tuple[int, str]] = []
    fence = False
    for line in text.splitlines():
        if line.startswith("```") or line.startswith("~~~"):
            fence = not fence
            continue
        if fence:
            continue
        m = re.match(r"^(#{2,3}) (.+)$", line)
        if m:
            headings.append((len(m.group(1)), m.group(2)))
    return headings


def check_section_reference_pointer_coverage(
    *, repo_root: Path | None = None, warn_sink: list[str] | None = None
) -> list[str]:
    """FAIL any reference-file section that lost its owning-spec pointer (#1159).

    Scans every ``.claude/rules/*.md`` whose filename ends with a suffix in
    :data:`_SECTION_REFERENCE_SUFFIXES` (today: analyzer-section-reference.md,
    critic-lens-reference.md, planner-section-reference.md,
    clean-result-critic-lens-reference.md). For each, the owning agent spec is
    ``.claude/agents/<stem-minus-suffix>.md`` — missing spec FAILs (orphan
    reference file). Headings are enumerated FENCE-AWARE (see
    :func:`_fence_aware_headings`). The file's SECTION GRAIN is 2 when any
    non-fenced H2 exists, else 3; only grain-level headings require pointers
    (sub-headings below grain are intra-section structure). GRAIN-MIXING DRIFT
    PATH (documented H2-grain-wins rule — keep a reference file single-grain):
    an H3-grain file that gains ONE stray non-fenced H2 flips to H2 grain and
    silently DROPS every H3 from coverage. Pointer predicate: the
    whitespace-normalized owning-spec text must contain the substring
    ``"§ " + <whitespace-normalized heading>`` — wrapped pointer lines pass;
    a prose mention without the ``§ `` sigil does not. A matched reference
    file with ZERO non-fenced H2/H3 headings FAILs (malformed). FAIL severity;
    no WARN cases in v1 (``warn_sink`` accepted for signature uniformity, and
    unused). ``repo_root`` is a unit-test override; production callers pass
    None. Bundled into the no-flags default run. Closes the #850-class gap
    (a relocated-but-unreachable section, previously caught only by one Codex
    review MAJOR).
    """
    root = repo_root if repo_root is not None else _REPO_ROOT
    rules_dir = root / ".claude" / "rules"
    agents_dir = root / ".claude" / "agents"
    errors: list[str] = []
    _ = warn_sink  # no WARN cases in v1 (signature uniformity)
    if not rules_dir.is_dir():
        return errors
    for ref_path in sorted(rules_dir.glob("*.md")):
        suffix = next((s for s in _SECTION_REFERENCE_SUFFIXES if ref_path.name.endswith(s)), None)
        if suffix is None:
            continue
        agent_name = ref_path.name[: -len(suffix)]
        owning_spec = agents_dir / f"{agent_name}.md"
        rel_ref = f".claude/rules/{ref_path.name}"
        if not owning_spec.is_file():
            errors.append(
                f"{rel_ref}: orphan reference file — the owning agent spec "
                f".claude/agents/{agent_name}.md does not exist (rename the reference "
                f"or restore the spec)."
            )
            continue
        headings = _fence_aware_headings(ref_path.read_text(encoding="utf-8"))
        if not headings:
            errors.append(
                f"{rel_ref}: malformed reference file — no non-fenced H2/H3 section "
                f"headings found (nothing for the owning spec to point at)."
            )
            continue
        grain = 2 if any(level == 2 for level, _ in headings) else 3
        spec_norm = _ws_norm(owning_spec.read_text(encoding="utf-8"))
        for level, heading in headings:
            if level != grain:
                continue
            if f"§ {_ws_norm(heading)}" not in spec_norm:
                errors.append(
                    f"{rel_ref}: section heading '{heading}' (H{grain} grain) has no "
                    f"'§ <exact heading>' pointer line in the owning spec "
                    f".claude/agents/{agent_name}.md — a relocated section must stay "
                    f"pointer-reachable (#850/#1159)."
                )
    return errors


# ── --check-skill-bang-backtick (#1243/#1266) ────────────────────────────────
# The hazard: Claude Code's skill/command markdown preprocessor treats a bang
# directly against a backtick as opening an inline-exec span and runs the text
# up to the next backtick as shell AT SKILL LOAD (incident #1243/#1266:
# commit 90af0ce2d9 killed every /issue boot; hotfix f75e1b4c13). The regex is
# built with \x60 (backtick) so this file never contains the adjacency itself.
_BANG_BACKTICK_RE = re.compile(r"(?<!\$)!\x60")

_BANG_BACKTICK_ROOTS = ("skills", "agents", "commands")  # under .claude/


def check_skill_bang_backtick(*, claude_dir: Path | None = None) -> list[str]:
    """FAIL on any non-dollar-preceded bang directly against a backtick in
    preprocessor-loaded markdown (.claude/{skills,agents,commands}/**/*.md).

    Every line of every file is scanned — NO fenced-block exemption (the
    preprocessor is not verified to ignore fences; scan-everything is the
    safe default) and NO waiver/pragma (an allowlisted hit would still
    execute at skill load — a waiver cannot neutralize the hazard; same
    no-legitimate-use policy as ``check_heredoc_dotenv``). The sole
    carve-out is a '$' immediately before the bang (the shell-pid '$!'
    prose shape, 3 live instances in .claude/skills/issue/SKILL.md,
    empirically inert across healthy boots). A bang at end-of-line with a
    backtick opening the NEXT line is correct-by-construction NOT flagged
    — the two characters must be byte-adjacent on one line to trigger the
    preprocessor, so the per-line scan is deliberate. Remediation is
    always rewording: insert a space between the bang and the backtick,
    or write 'bang-backtick' in prose; any SCANNED markdown documenting
    this check must do the same — there is no in-file escape, by design.
    ``claude_dir`` is a unit-test override; production callers pass None
    (canonical <repo_root>/.claude). 'commands' is exists-guarded
    future-proofing (dir absent today; it is the canonical preprocessor
    surface). Bundled into the no-flags default run.
    """
    base = claude_dir if claude_dir is not None else _REPO_ROOT / ".claude"
    errors: list[str] = []
    for root_name in _BANG_BACKTICK_ROOTS:
        root = base / root_name
        if not root.exists():
            continue
        for md in sorted(root.rglob("*.md")):
            if not md.is_file():
                continue
            lines = md.read_text(encoding="utf-8").splitlines()
            for lineno, line in enumerate(lines, start=1):
                n_hits = len(_BANG_BACKTICK_RE.findall(line))
                if n_hits:
                    rel = md.relative_to(base.parent) if claude_dir is None else md
                    errors.append(
                        f"{rel}:{lineno}: {n_hits} non-dollar '!' directly "
                        "against a backtick (skill-preprocessor inline-exec "
                        "hazard, #1243/#1266) — reword: insert a space before "
                        "the backtick or write 'bang-backtick' in prose"
                    )
    return errors


# ── --check-agents-note-argv-verdict (#1743/#1785) ───────────────────────────
# Both banned-pattern strings are built by FRAGMENT CONCATENATION so this
# module's own source never carries the exact matched literal — a future
# `grep scripts/` sweep for either pattern stays clean (the same self-match
# avoidance the guard hooks use).
_NOTE_ARGV_VERDICT_P1 = 'note "$(' + "cat"  # the #1743 acceptance-grep pattern
_NOTE_ARGV_VERDICT_P2 = "--note " + '"$('  # the #1743 reviewer's broader variant


def check_agents_note_argv_verdict(*, agents_dir: Path | None = None) -> list[str]:
    """Walk every ``*.md`` under ``.claude/agents/`` and FAIL on any line
    prescribing an argv-prose ``--note`` verdict/marker post opened as a
    command substitution — the pattern task #1743 (merged 99af2fbb0d)
    banned from agent specs and rewrote to the ``post-marker --file``
    channel; #1785 pins that acceptance grep as this standing check.

    Two literal substrings are flagged (built by fragment concatenation
    above so this module's own source never carries either matched
    literal): P1 — the #1743 acceptance-grep pattern (a note flag whose
    body opens as a substitution around ``cat``); P2 — the #1743
    reviewer's broader variant (any note flag opening directly into a
    command substitution). Rationale (the #1722/#1756 incident family): a
    command substitution nested inside an already-double-quoted note
    argument collapses backslash-escaped inner quotes (git reads them
    literal, silently yielding empty output and a blank field in the
    durable marker), and the PreToolUse guards scan the whole Bash argv —
    heredoc bodies included — so a spec PRESCRIBING the argv shape steers
    every future agent into the very block the ``--file`` channel exists
    to avoid.

    Deliberately NOT matched — do not "fix" this check into flagging it:
    the sanctioned variable form (resolve every command substitution into
    a shell variable FIRST, then pass the variable as the note argument)
    has no substitution opener adjacent to the note token, so it passes
    by construction.

    Plain substring match; no allowlist and no comment/fence skipping —
    agent specs are prose, and the #1743 r2 precedent is to REWORD a line
    that would match (a warning can describe the ban without carrying the
    literal).

    ``agents_dir`` is an override hook for unit tests; production callers
    pass None and the function walks the canonical
    ``<repo_root>/.claude/agents`` tree. Bundled into the no-flags default
    run (same policy as ``check_piped_git_push``).
    """
    root = agents_dir if agents_dir is not None else _REPO_ROOT / ".claude" / "agents"
    if not root.exists():
        return []
    errors: list[str] = []
    for md in sorted(root.rglob("*.md")):
        if not md.is_file():
            continue
        lines = md.read_text(encoding="utf-8").splitlines()
        for lineno, line in enumerate(lines, start=1):
            if _NOTE_ARGV_VERDICT_P1 not in line and _NOTE_ARGV_VERDICT_P2 not in line:
                continue
            rel = md.relative_to(_REPO_ROOT) if agents_dir is None else md
            errors.append(
                f"{rel}:{lineno}: agent spec prescribes an argv-prose --note "
                f"verdict/marker post (the pattern #1743 banned; use the "
                f"--file channel)"
            )
    return errors


# ── --check-sha-pin-domain (#2079; the #1776/#1491 wrong-domain class) ───────
# A sha pin digests ONE representation (int64 INDEX arrays vs PROMPT strings
# vs file BYTES); a consumer comparing a digest computed in a DIFFERENT
# domain fails on EVERY input and masquerades as upstream data drift
# (.claude/rules/gotchas.md "A sha pin lives in a DOMAIN"). The #1491 pre-fix
# shape (git show 9f43b03e43^) copied the #779 fixed_split INDEX-array
# digests into a new module as bare VAL_SHA256/TEST_SHA256 and asserted
# prompt-string digests against them — an assert that could never pass.
SHA_PIN_DOMAIN_VOCAB: tuple[str, ...] = ("INDEX", "IDS", "PROMPT", "BYTES", "CONTENT")
# A WHOLE-STRING 64-hex literal (the pin-constant shape). Hexes embedded in a
# longer string (URLs, hub paths) are out of scope by design — they are not
# pin bindings a consumer asserts against.
_SHA_PIN_HEX_RE = re.compile(r"([\"'])([0-9a-f]{64})\1")
_SHA_PIN_ANNOT_RE = re.compile(r"#\s*SHA_PIN_DOMAIN:\s*([A-Z]+)\b")
_SHA_PIN_EXEMPT_RE = re.compile(r"#\s*SHA_PIN_DOMAIN_EXEMPT:\s*\S")
_SHA_PIN_ASSIGN_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*(?::[^=]*)?=(?!=)")
_SHA_PIN_DICT_KEY_RE = re.compile(r"([\"'])([A-Za-z0-9_.\-]+)\1\s*:\s*[\"'][0-9a-f]{64}[\"']")
_SHA_PIN_WORD_RE = re.compile(r"[A-Za-z]+")
# Multi-line list/dict/paren literals put the hex lines BELOW the binding —
# resolve the nearest preceding assignment target within this many lines.
_SHA_PIN_BINDING_LOOKBACK = 5
# Frozen grandfather — (hex[:12], repo-relative POSIX path) PAIRS for the
# legacy duplicated hexes that predate this check (the JUDGE_PIN allowlist
# snapshot idiom; regenerated from the live tree 2026-08-05, task #2079).
# The PAIR grain is load-bearing: a grandfathered hex COPIED INTO A NEW FILE
# still FAILs (exactly the #1491 propagation vector) while today's legacy
# sites stay green. A stale entry (no longer matching an undeclared
# cross-module pin site) FAILs the default run — the set can shrink, never
# silently grow: a NEW cross-module pin site declares its domain instead of
# being added here. Conflicting declarations have NO allowlist escape.
SHA_PIN_DOMAIN_GRANDFATHER: frozenset[tuple[str, str]] = frozenset(
    {
        # #612 dose-matched sycophancy pool sha, re-pinned by #650:
        ("0d78e82262bf", "src/explore_persona_space/experiments/issue_650/__init__.py"),
        (
            "0d78e82262bf",
            "src/explore_persona_space/experiments/sycophancy_onpolicy_612/__init__.py",
        ),
        # #612 software_engineer train-pool sha, re-pinned by #642:
        ("12fdeb3bbb8b", "scripts/issue_642/i642_common.py"),
        (
            "12fdeb3bbb8b",
            "src/explore_persona_space/experiments/sycophancy_onpolicy_612/__init__.py",
        ),
        # #612 villain train-pool sha, re-pinned by #642 (v4 canned pool):
        ("1b72c008ff70", "scripts/issue_642/i642_common.py"),
        (
            "1b72c008ff70",
            "src/explore_persona_space/experiments/sycophancy_onpolicy_612/__init__.py",
        ),
        # #922 maps bundle sha (fixed-point slow-modes input + provenance repair):
        ("1f1aaa839473", "scripts/issue922_fixed_point_slow_modes.py"),
        ("1f1aaa839473", "scripts/issue922_repair_provenance.py"),
        # #823/#952 shared analysis input-bundle sha (BUNDLE_SHA256 in both rigs):
        ("46c06e89c513", "src/explore_persona_space/experiments/issue_823/run_823.py"),
        ("46c06e89c513", "src/explore_persona_space/experiments/issue_952/run_952.py"),
        # TRACKS prompt-membership sha: #1335 declares PROMPT via binding name;
        # #1417's TRACKS_SHA256 copy predates the declaration convention:
        ("55c5d462ac01", "scripts/issue1417_render.py"),
        # #1482 SAE holdout-split sha shared by the three analysis scripts:
        ("7957d689748e", "scripts/issue1482_early_layer.py"),
        ("7957d689748e", "scripts/issue1482_error_analysis.py"),
        ("7957d689748e", "scripts/issue1482_run_length.py"),
        # #1481/#1947 marker eval-bank sha:
        ("7c08c15bea17", "scripts/issue1481_marker.py"),
        ("7c08c15bea17", "scripts/issue1947_datagen.py"),
        # #1482 SAE fit-manifest sha (early_layer + run_length):
        ("88d344675fbb", "scripts/issue1482_early_layer.py"),
        ("88d344675fbb", "scripts/issue1482_run_length.py"),
        # #594 probe-pool sha pinned by the #658/#810 fitters + #667 analysis:
        ("ad687becec26", "scripts/issue658_common.py"),
        ("ad687becec26", "scripts/issue810_adhoc_lofo_heatmaps.py"),
        ("ad687becec26", "scripts/issue810_common.py"),
        ("ad687becec26", "src/explore_persona_space/analysis/issue667/__init__.py"),
        # #612 wrong-claims train-200 sha, re-pinned by #653:
        ("c3ac7cef9d11", "src/explore_persona_space/experiments/issue_653/__init__.py"),
        (
            "c3ac7cef9d11",
            "src/explore_persona_space/experiments/sycophancy_onpolicy_612/__init__.py",
        ),
        # UltraChat/G1 probe-pool sha (#658 extract + #810 common):
        ("f277f8c3e255", "scripts/issue658_extract_base_store.py"),
        ("f277f8c3e255", "scripts/issue810_common.py"),
    }
)


def _sha_pin_binding_name(lines: list[str], idx: int) -> tuple[str, int]:
    """Resolve the binding name + binding-line index for the hex at ``lines[idx]``.

    A same-line dict key wins, then a same-line assignment target, then the
    nearest preceding assignment target within
    :data:`_SHA_PIN_BINDING_LOOKBACK` lines (multi-line list/dict/paren
    literals, including the annotated ``NAME: Final[str] = (`` shape), else
    ``<bare>`` anchored at the hex line.
    """
    dict_key = _SHA_PIN_DICT_KEY_RE.search(lines[idx])
    if dict_key:
        return dict_key.group(2), idx
    same_line = _SHA_PIN_ASSIGN_RE.match(lines[idx])
    if same_line:
        return same_line.group(1), idx
    for back in range(1, _SHA_PIN_BINDING_LOOKBACK + 1):
        j = idx - back
        if j < 0:
            break
        preceding = _SHA_PIN_ASSIGN_RE.match(lines[j])
        if preceding:
            return preceding.group(1), j
    return "<bare>", idx


def _sha_pin_resolve(lines: list[str], idx: int, name: str, bidx: int) -> tuple[str, str]:
    """Resolve one pin site's domain disposition -> ``(kind, domain)``.

    ``kind`` is ``"exempt"`` | ``"domain"`` | ``"undeclared"``. An adjacent
    ``# SHA_PIN_DOMAIN_EXEMPT: <reason>`` wins, then an adjacent
    ``# SHA_PIN_DOMAIN: <TOKEN>`` annotation, then a
    :data:`SHA_PIN_DOMAIN_VOCAB` token in the binding name
    (case-insensitive whole-word). "Adjacent" = the hex line, the line
    immediately above it, the binding line, or the line immediately above
    the binding line — covering trailing comments, preceding-line comments,
    and the multi-line paren-assignment shape where the annotation sits
    above the assignment target.
    """
    candidates = sorted({idx, max(idx - 1, 0), bidx, max(bidx - 1, 0)})
    for j in candidates:
        if _SHA_PIN_EXEMPT_RE.search(lines[j]):
            return "exempt", ""
    for j in candidates:
        annot = _SHA_PIN_ANNOT_RE.search(lines[j])
        if annot:
            return "domain", annot.group(1)
    words = {w.upper() for w in _SHA_PIN_WORD_RE.findall(name)}
    for token in SHA_PIN_DOMAIN_VOCAB:
        if token in words:
            return "domain", token
    return "undeclared", ""


def _sha_pin_sites(root: Path) -> dict[str, list[tuple[str, int, str, str, str]]]:
    """Scan ``scripts/`` + ``src/explore_persona_space/`` ``*.py`` for
    whole-string 64-hex literals.

    Returns ``{hex: [(relpath, lineno, binding, kind, domain), ...]}``. An
    unreadable (non-UTF-8) file is skipped with a stderr notice, never a
    crash (the ``check_jsonl_splitlines`` precedent).
    """
    sites: dict[str, list[tuple[str, int, str, str, str]]] = {}
    for scan_root in (root / "scripts", root / "src" / "explore_persona_space"):
        if not scan_root.exists():
            continue
        for path in sorted(scan_root.rglob("*.py")):
            if not path.is_file():
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                sys.stderr.write(
                    f"workflow_lint: notice: sha-pin-domain skipped unreadable file {path}\n"
                )
                continue
            lines = text.split("\n")
            rel = path.relative_to(root).as_posix()
            for i, line in enumerate(lines):
                for match in _SHA_PIN_HEX_RE.finditer(line):
                    hex_val = match.group(2)
                    name, bidx = _sha_pin_binding_name(lines, i)
                    kind, domain = _sha_pin_resolve(lines, i, name, bidx)
                    sites.setdefault(hex_val, []).append((rel, i + 1, name, kind, domain))
    return sites


def check_sha_pin_domain(*, repo_root: Path | None = None) -> list[str]:
    """FAIL cross-module 64-hex sha pins with an undeclared or conflicting
    content DOMAIN (#2079 — the #1776/#1491 wrong-domain class).

    Predicate (calibrated on the 2026-08-05 live tree):

    1. Collect whole-string 64-hex literals under ``scripts/*.py`` +
       ``src/explore_persona_space/**/*.py`` (NOT ``tests/`` — fixtures
       legitimately re-pin; NOT non-.py files). The binding name is the
       same-line dict key, else the same-line assignment target, else the
       nearest preceding assignment target within
       :data:`_SHA_PIN_BINDING_LOOKBACK` lines, else ``<bare>``.
    2. Keep hexes appearing in >= 2 DISTINCT modules.
    3. Per site, resolve a domain: an adjacent ``# SHA_PIN_DOMAIN: <TOKEN>``
       comment wins; else a :data:`SHA_PIN_DOMAIN_VOCAB` token in the
       binding name (case-insensitive word match). An adjacent
       ``# SHA_PIN_DOMAIN_EXEMPT: <reason>`` exempts the SITE.
    4. FAIL rows: **conflict** — >= 2 sites resolve to DIFFERENT domains
       (one row per declared site; NO allowlist escape); **undeclared** — a
       site resolves no domain and its ``(hex[:12], file)`` pair is not in
       :data:`SHA_PIN_DOMAIN_GRANDFATHER`; **grandfather-stale** — a
       grandfather entry no longer matching an undeclared cross-module pin
       site (forces cleanup: the set shrinks, never silently grows).

    ``repo_root`` is a unit-test override hook; production callers pass
    None and the check scans under :data:`_REPO_ROOT`. Bundled into the
    no-flags default run.
    """
    root = repo_root if repo_root is not None else _REPO_ROOT
    errors: list[str] = []
    consumed: set[tuple[str, str]] = set()
    for hex_val, site_list in sorted(_sha_pin_sites(root).items()):
        files = sorted({site[0] for site in site_list})
        if len(files) < 2:
            continue
        active = [site for site in site_list if site[3] != "exempt"]
        declared = sorted({site[4] for site in active if site[3] == "domain"})
        for rel, lineno, name, kind, domain in active:
            if kind == "domain" and len(declared) >= 2:
                errors.append(
                    f"sha-pin-domain/{rel}:{lineno}: conflicting content domains "
                    f"{declared} for cross-module sha pin {hex_val[:12]}... (this "
                    f"site: {domain}; binding `{name}`; sites: {files}). A pin "
                    f"digests ONE representation — reconcile the declarations; "
                    f"conflicts have no allowlist escape (#2079; #1776/#1491)"
                )
            elif kind == "undeclared":
                pair = (hex_val[:12], rel)
                if pair in SHA_PIN_DOMAIN_GRANDFATHER:
                    consumed.add(pair)
                    continue
                others = [f for f in files if f != rel]
                errors.append(
                    f"sha-pin-domain/{rel}:{lineno}: undeclared cross-module sha pin "
                    f"{hex_val[:12]}... (binding `{name}`; also pinned in {others}). "
                    f"Declare the content domain — a `# SHA_PIN_DOMAIN: "
                    f"<{'|'.join(SHA_PIN_DOMAIN_VOCAB)}>` comment on the pin line or "
                    f"the line above, or a domain token in the binding name — or "
                    f"waive the site with `# SHA_PIN_DOMAIN_EXEMPT: <reason>` "
                    f"(#2079; the #1776/#1491 wrong-domain class)"
                )
    for hex12, rel in sorted(SHA_PIN_DOMAIN_GRANDFATHER - consumed):
        errors.append(
            f"sha-pin-domain/grandfather-stale: ({hex12!r}, {rel!r}) in "
            f"SHA_PIN_DOMAIN_GRANDFATHER no longer matches an undeclared "
            f"cross-module 64-hex pin site — remove the entry (the grandfather "
            f"shrinks, never silently grows; #2079)"
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
        "--check-piped-git-push",
        action="store_true",
        help="Verify no shell script under scripts/ pipes a `git push` / "
        "`git merge` / `git commit` / `gh pr merge|create` into a filter "
        "on its own "
        "pipeline segment (`git push origin main 2>&1 | tail -20`). The "
        "pipe masks the non-zero exit code, so a rejected push reads as "
        "success (#957; 4 sessions hit this 2026-07-02), and a piped "
        "`git commit` is SIGPIPE-killed mid-pre-commit-hook (#1584) — run "
        "it bare and "
        "check the exit code, or add `set -o pipefail` (a non-comment "
        "pipefail line disables flagging for the rest of the file). "
        "Comment lines and `--dry-run` pipes are skipped. See CLAUDE.md "
        "§ Concurrent repo-root committers (#1048, #1591). Bundled into "
        "the no-flags default run.",
    )
    parser.add_argument(
        "--check-push-failure-swallow",
        action="store_true",
        help="Verify no shell script under scripts/ swallows a `git push` "
        "failure on the same logical line (`git push ... || echo warn`, "
        "`|| true`, `|| :`, `|| printf`). The swallow declares success "
        "while the result commit never landed; on GCE the self-DELETEing "
        "instance holds the only copy (#825 r6-r8). Verify the push "
        "instead (rev-list count 0, retry once, exit non-zero) per "
        ".claude/rules/pod-side-reporting.md § Result-push verification "
        "contract. if-conditions, bare pushes, and `|| { retry; }` groups "
        "never match; waive with `# PUSH_SWALLOW_EXEMPT: <reason>`; "
        "legacy offenders frozen in PUSH_SWALLOW_LEGACY_ALLOWLIST. "
        "Bundled into the no-flags default run (#1205).",
    )
    parser.add_argument(
        "--check-sh-function-rc-capture",
        action="store_true",
        help="Verify no shell script under scripts/ invokes a SAME-FILE "
        "bash function via `func || rc=$?` / `|| true` / `|| :` under "
        "set -e — bash disables errexit inside the function BODY when "
        "the call sits in an `||` context, so mid-function failures "
        "collapse to the last command's rc (#1426: partial uploads + the "
        "`[phase=done]` success sentinel proceeded past a Gate-1 "
        "terminal failure). Single external-command captures never "
        "match; `set +e` regions are unflagged; waive a genuinely-safe "
        "shape with `# RC_CAPTURE_EXEMPT: <reason>`. ShellCheck SC2310 "
        "is the broader external analogue. Bundled into the no-flags "
        "default run (#1516).",
    )
    parser.add_argument(
        "--check-grep-qv",
        action="store_true",
        help="Verify no executable workflow snippet (fenced code blocks in "
        ".claude/skills/**/SKILL.md + .claude/agents/*.md, plus "
        "scripts/**/*.sh) runs an unpinned grep/ugrep combining -q and -v "
        "with the exit status as the signal. ugrep 7.5.0's quiet+invert "
        "exit status diverges from GNU (rc=1 when non-matching lines are "
        "selected), so such a trigger fails OPEN under a PATH-shadowed "
        "grep (#928; fixed in #1125). git grep and a path-pinned "
        "/usr/bin/grep are exempt; a path-pinned ugrep still flags. "
        "Comment lines and prose outside fences are skipped. Bundled into "
        "the no-flags default run.",
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
        "--check-hub-dir-filecount",
        action="store_true",
        help="AST-walk scripts/**/*.py and FAIL on any DIRECT upload_folder(...) "
        "call site (api.upload_folder / a bare huggingface_hub import) in a "
        "module that does not reference assert_hub_dir_filecounts (the hub.py "
        "runtime guard, #1190). The Hub rejects any single repo directory "
        "holding >10k files at COMMIT time with a NON-retriable "
        "BadRequestError AFTER all bytes are staged (#658); the shared hub "
        "helpers pre-count staged files per target repo dir and fail loud "
        "before any network I/O, and this lint funnels direct HfApi callers "
        "toward the same one-line guard (called OUTSIDE any transient-retry "
        "wrapper). Waive with '# HUB_DIR_FILECOUNT_EXEMPT: <reason>'; "
        "pre-existing call sites are grandfathered in "
        "HUB_DIR_FILECOUNT_LEGACY_ALLOWLIST. Bundled into the no-flags "
        "default run.",
    )
    parser.add_argument(
        "--check-upload-prefix-clobber",
        action="store_true",
        help="AST-walk scripts/**/*.py (two passes) and FAIL on hardcoded "
        "issue-prefix HF upload DESTINATIONS: a cross-issue dest token "
        "(issue<M>_ in an issue<N>_ script, Rule A) or an own-issue token "
        "arriving via a fallback channel — `x or CONST`, an argparse "
        "default=, a wrapper-param signature default (Rule B) — the #1005 "
        "parent-clobber class (reused #928 fitters overwrote the parent's "
        "HF artifacts). Direct own-prefix hardcodes and cross-issue READS "
        "never flag. Waive with '# UPLOAD_PREFIX_EXEMPT: <reason>'; "
        "pre-existing Rule-B sites are grandfathered in "
        "UPLOAD_PREFIX_CLOBBER_ALLOWLIST. Bundled into the no-flags "
        "default run.",
    )
    parser.add_argument(
        "--check-upload-file-in-loop",
        action="store_true",
        help="AST-walk scripts/**/*.py and FAIL on any per-file upload call "
        "inside a loop — upload_file(...) (shape A) or "
        "_upload(..., upload_as_file=True) (shape B, the literal #664 form) "
        "— the per-file upload-loop 429/504-storm anti-pattern (#664/#1481); "
        "use one bulk upload_folder commit instead. Waive a genuinely "
        "bounded loop with '# UPLOAD_LOOP_EXEMPT: <reason>'; pre-existing "
        "sites are grandfathered with exact per-file counts in "
        "UPLOAD_FILE_IN_LOOP_LEGACY_ALLOWLIST. Bundled into the no-flags "
        "default run.",
    )
    parser.add_argument(
        "--check-upload-return-discard",
        action="store_true",
        help="AST-walk scripts/**/*.py and FAIL on any Expr-statement "
        "(discarded-return) call to the fail-soft-by-return hub upload "
        "helpers _upload / _upload_folder_filtered — both return '' on "
        "upload failure, so a discarded return exits 0 on silent "
        "durability loss (#2087; incident #2054). Import/definition-"
        "resolved arming: a same-named LOCAL helper never arms. Waive a "
        "deliberate fail-soft caller with "
        "'# UPLOAD_RETURN_DISCARD_EXEMPT: <reason>'; pre-existing sites "
        "are grandfathered with <=-tolerant per-file counts in "
        "UPLOAD_RETURN_DISCARD_LEGACY_ALLOWLIST. Bundled into the "
        "no-flags default run.",
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
        "--check-hub-verify-retry",
        action="store_true",
        help="AST-walk scripts/**/*.py and FAIL on any bare list_repo_files( / "
        "list_repo_tree( / .file_exists( Hub call outside the grandfathered "
        "legacy set (#920/#997/#1202). New verify legs MUST route through "
        "orchestrate.hub (verify_repo_paths_uploaded / "
        "list_hf_files_under_path / retry_transient). Waive with "
        "'# HUB_VERIFY_RETRY_EXEMPT: <reason>'. Bundled into the no-flags "
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
        "--check-inline-round-duty-mirror",
        action="store_true",
        help="Verify the three 'Inline estimator-validity + record-integrity "
        "duties' anchor sentences are mirrored byte-identically between "
        "CLAUDE.md § 'User-chat inline free analysis' and "
        ".claude/skills/issue/SKILL.md Step 9a-ter (#1701). Two-part "
        "invariant: (a) each anchor prefix appears exactly once in each "
        "file; (b) the full anchor sentence is byte-identical across the "
        "two files. Bundled into the no-flags default run.",
    )
    parser.add_argument(
        "--check-rule-frontmatter-parses",
        action="store_true",
        help="YAML-parse every .claude/rules/*.md frontmatter block and "
        "validate the paths: load-trigger shape (non-empty list of glob "
        "strings; stale globs: key flagged; no-frontmatter files exempt). "
        "A malformed frontmatter append silently disables on-demand "
        "loading — the rule file exists but never loads (#1385, from "
        "#1348). Bundled into the no-flags default run.",
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
        "--check-crash-fix-relaunch-contract",
        action="store_true",
        help="FAIL if the #1081 crash-fix-relaunch fix-engaged contract prose "
        "regresses on any of its three surfaces (experimenter.md D3 paragraph, "
        "crash-fix-rounds.md fix_sha note-token paragraph, /issue SKILL.md "
        "Step 7 code-row relaunch contract): unique anchor per surface, "
        "whitespace-normalized required tokens (incl. the disposition-"
        "conditional trio 'empty / the fresh path / exactly the RETAINED "
        "expected paths'), and a negative regex against an unconditional "
        "'resolves EMPTY' confirm. Bundled into the no-flags default run "
        "(#1181).",
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
        "--check-awk-elision-parity",
        action="store_true",
        help="FAIL if the ban-gate awk elision program (the single-quoted awk "
        "program on the unique f=!f anchor line) drifts between its two "
        "byte-identical full-text homes — the /issue SKILL.md Step "
        "9a-humanize gate and analyzer-section-reference.md Step 4.5 — or a "
        "home is missing, has 0 or >1 anchor lines, carries a non-2 total "
        "single-quote count on the anchor line (quote-escape truncation "
        "guard), or yields no extractable awk '...' span (#1153). Compares "
        "the quoted PROGRAM only; the surrounding invocation lines "
        "legitimately differ. Bundled into the no-flags default run.",
    )
    parser.add_argument(
        "--check-asw-docstring-pass-count",
        action="store_true",
        help="verify the autonomous_session_watch.py docstring '<N> passes' "
        "header digit == numbered inventory items (1..N sequential) == live "
        "main() pass set (distinct *_pass calls + the inline crash-recovery "
        "block) (#1225). Bundled into the no-flags default run.",
    )
    parser.add_argument(
        "--check-marker-recipe-snippets",
        action="store_true",
        help="FAIL when a frozen numeric snippet in docs/marker_training_recipe.md "
        "or .claude/rules/marker-training-recipe.md disagrees with the code "
        "constant it cites (registry _MARKER_RECIPE_PINS: marker token id 83399, "
        "the MarkerOnlyDataCollator tail_tokens default, MIX_MAX_REJECT_FRAC, the "
        "MarkerBandStopCallback default band). Registry-driven — empirical "
        "findings / frozen experiment history are never parsed. Bundled into the "
        "no-flags default run (#1154).",
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
        "--check-live-hf-retry-routing",
        action="store_true",
        help="Walk scripts/**/*.py + src/explore_persona_space/**/*.py and FAIL "
        "on a bare (un-retried) HuggingFace Hub call in LIVE code: "
        "hf_hub_download( / .upload_file( / .upload_folder( / create_commit( / "
        "push_to_hub( with no retry_transient/_retry_upload wrap anchored to "
        "the call and no '# NO_RETRY: <reason>' waiver. hf_hub 0.36.2 natively "
        "retries only 500/502/503/504 on download/LFS paths and the commit API "
        "not at all, so a bare live site is a 429 single-point-of-failure "
        "(#1426/#1335, the 2026-07-18 storm). The per-issue historical "
        "files frozen at #1547 implement time are snapshot-exempt "
        "(HF_ROUTING_FROZEN_SNAPSHOT; stale snapshot? see "
        "--regen-hf-routing-snapshot); NEW files are scanned. Bundled into "
        "the no-flags default run (#1547).",
    )
    parser.add_argument(
        "--regen-hf-routing-snapshot",
        action="store_true",
        help="MAINTENANCE (#1568, not a check; runs alone and early-returns — "
        "combining it with check flags is unsupported, regen wins): print the "
        "ready-to-paste HF_ROUTING_FROZEN_SNAPSHOT literal for the current "
        "tree (stdout) + a +/- diff summary vs the compiled-in constant "
        "(stderr). Run on a main-synced tree when the live-hf-retry-routing "
        "check fires on a file your round never touched (implementer-time "
        "snapshot went stale before the merge gate — the #1547 race). Review "
        "added entries before pasting; never bundled into the no-flags "
        "default run.",
    )
    parser.add_argument(
        "--check-bare-list-repo-files",
        action="store_true",
        help="AST-walk scripts/**/*.py + src/explore_persona_space/**/*.py and "
        "FAIL on any bare list_repo_files call/reference (Load-ctx Attribute "
        "under any receiver, or the huggingface_hub imported Name incl. "
        "aliases). hub 0.36.2's HfApi.list_repo_files has NO scoping "
        "parameter — every call is an unscoped full-tree walk, which WEDGES "
        "on the ~1M-file data repo (>90 s #833, >600 s #920; two kills "
        "2026-07-22 -> #1624) and retry cannot save it — orthogonal to "
        "--check-hub-verify-retry. Fix with the scoped recipes "
        "(hub.list_hf_files_under_path / hub.verify_repo_paths_uploaded / "
        "api.list_repo_tree(path_in_repo=...) / api.file_exists); a "
        "genuinely-correct SMALL-repo full listing waives with "
        "'# LIST_REPO_FILES_EXEMPT: <reason>'. Historical files are "
        "snapshot-exempt (LIST_REPO_FILES_FROZEN_SNAPSHOT; stale snapshot? "
        "see --regen-list-repo-files-snapshot); NEW files are scanned. "
        "Bundled into the no-flags default run (#1624).",
    )
    parser.add_argument(
        "--regen-list-repo-files-snapshot",
        action="store_true",
        help="MAINTENANCE (#1624, not a check; runs alone and early-returns — "
        "combining it with check flags is unsupported, regen wins): print the "
        "ready-to-paste LIST_REPO_FILES_FROZEN_SNAPSHOT literal for the "
        "current tree (stdout) + a +/- diff summary vs the compiled-in "
        "constant (stderr). Run on a main-synced tree when the "
        "bare-list-repo-files check fires on a file your round never touched "
        "(implementer-time snapshot went stale before the merge gate — the "
        "#1547/#1568 race). Review added entries before pasting; never "
        "bundled into the no-flags default run.",
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
        "--check-agent-memory-index-size",
        action="store_true",
        help="agent-memory index size budget over .claude/agent-memory/*/MEMORY.md: "
        "WARN >20 KB, FAIL >24 KB (the always-loaded index is truncated by the "
        "loader at ~25 KB, silently dropping the newest lessons — #1891)",
    )
    parser.add_argument(
        "--check-gotchas-size",
        action="store_true",
        help="gotchas.md regrowth size budget: WARN >200,000 B, FAIL >250,000 B "
        "(the file is machine-appended by consolidate_lessons.py; the cap forces "
        "periodic re-trims)",
    )
    parser.add_argument(
        "--check-skill-doc-size",
        action="store_true",
        help="skill-doc size budget over .claude/skills/**/*.md: WARN >40 KB, "
        "FAIL >60 KB (grandfather-ratchet; generated tables + "
        "exemplars/templates dirs exempt)",
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
        "--check-section-reference-pointers",
        action="store_true",
        help="Scan every .claude/rules/*-section-reference.md / *-lens-reference.md "
        "and FAIL any non-fenced section heading at the file's grain (H2 if any "
        "H2 exists, else H3) lacking a whitespace-normalized '§ <exact heading>' "
        "pointer in the owning .claude/agents/<agent>.md spec; also FAIL an "
        "orphan or headingless reference file (#850/#1159). Bundled into the "
        "no-flags default run.",
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
        "FAIL any .splitlines() call reading JSONL content (6 signals: "
        "jsonl-named read_text chain / jsonl-named receiver / jsonl-named "
        "enclosing function / events-comments-path read_text chain / "
        "glob-gated generic read_text receiver / read_text-assigned "
        "receiver). "
        "splitlines() splits on raw U+2028/U+2029/NEL inside "
        "ensure_ascii=False JSON strings and shreds valid records (#825/#950); "
        "use split('\\n') or text-mode file iteration. Waive with "
        "'# JSONL_SPLITLINES_EXEMPT: <reason>'; frozen legacy experiment "
        "scripts live in JSONL_SPLITLINES_LEGACY_ALLOWLIST (experiment files "
        "only — never a workflow-surface file). Bundled into the no-flags "
        "default run.",
    )
    parser.add_argument(
        "--check-scripts-import-guard",
        action="store_true",
        help="AST-walk src/explore_persona_space/experiments/**/*.py + "
        "scripts/**/*.py and FAIL "
        "any scripts.* import (deferred OR module-top-level) lacking a "
        "repo-root sys.path guard: a syspath-named call like "
        "_ensure_repo_root_on_syspath(), or a literal sys.path.insert/append "
        "— same-innermost-scope-preceding, or module-level (any line covers "
        "a deferred import; a preceding line covers a top-level one). In "
        "script mode sys.path[0] is the script's own dir, so an unguarded "
        "import crashes pod/GCE-side — deferred instances mid-run after the "
        "paid phases (#823/#853). try/except ImportError is NOT a guard; "
        "TYPE_CHECKING bodies are skipped. Waive with "
        "'# SCRIPTS_IMPORT_GUARD_EXEMPT: <reason>'. No legacy allowlist "
        "(the live tree is clean). Bundled into the no-flags default run.",
    )
    parser.add_argument(
        "--check-upload-or-true",
        action="store_true",
        help="Walk scripts/**/*.sh and FAIL any upload/result-persist "
        "command line whose failure is swallowed by '|| true' / '|| :' / "
        "'; true' (#841 silent artifact loss). Terminal swallows mask the "
        "whole &&-chain (whole-line token check); non-terminal '|| true' "
        "is segment-scoped; swallowed heredoc openers + multi-line "
        "python -c blocks are scanned for BODY upload-call tokens. Legacy "
        "deliberate uses frozen in UPLOAD_OR_TRUE_LEGACY_ALLOWLIST; waive "
        "with '# UPLOAD_OR_TRUE_EXEMPT: <reason>'. Bundled into the "
        "no-flags default run.",
    )
    parser.add_argument(
        "--check-git-recipes-root-guard",
        action="store_true",
        help="Extract every bash/sh/shell-tagged fenced block from "
        ".claude/agents/*.md + .claude/skills/**/SKILL.md + "
        ".claude/rules/*.md + CLAUDE.md, pre-filter to git-bearing blocks, "
        "and EXECUTE the live PreToolUse hook "
        "scripts/guard_repo_root_branch.sh against each WHOLE block (stdin "
        "JSON, as one pasted Bash call); hook exit 2 -> FAIL naming "
        "file:fence-opener-line + the hook's BLOCKED line. Waive a "
        "deliberate anti-pattern example / pod-side recipe with "
        "'<!-- workflow-lint: allow-root-guard-block: <reason> -->' on the "
        "line directly above the fence opener. A fail-loud hook self-test "
        "runs first (missing / fail-open / fail-closed hook = one loud "
        "error, never a silent pass). Closes the #1047 class (a documented "
        "recipe the live hook blocks). Bundled into the no-flags default "
        "run.",
    )
    parser.add_argument(
        "--check-bare-commit-pathspec",
        action="store_true",
        help="Verify no bash/sh/shell-tagged fenced block in the workflow "
        "docs (.claude/agents/*.md, .claude/skills/**/SKILL.md, "
        ".claude/rules/*.md, CLAUDE.md) prescribes a `git commit` with no "
        "` -- <pathspec>` tail: a bare commit at the always-concurrent "
        "shared repo root sweeps sibling sessions' staged files (incident "
        "7dbde267f1; #1630 fixed /daily per-file). `git -C <tree>` forms, "
        "xargs -r/--no-run-if-empty-driven commits, `--dry-run`, and "
        "comment lines are exempt; waive a fence with "
        "'<!-- workflow-lint: allow-bare-commit-block: <reason> -->' on "
        "the line directly above the fence opener. Bundled into the "
        "no-flags default run (#1648).",
    )
    parser.add_argument(
        "--check-marker-scalar-integrity",
        action="store_true",
        help="Scan every workflow.yaml § markers entry's four string fields "
        "(kind/posted_by/when/fields) for the truncated-comment signature: "
        "the PARSED value ends in ',' or '(' after rstrip, or has "
        "unbalanced parens. An unquoted YAML plain scalar containing ' #' "
        "silently truncates at the comment marker (#873) and "
        "--check-references passes because the regenerated markers.md "
        "matches the truncated parse. Waive deliberate prose via "
        "MARKER_SCALAR_INTEGRITY_ALLOWLIST. Bundled into --check-references "
        "and the no-flags default run.",
    )
    parser.add_argument(
        "--check-poller-marker-consumers",
        action="store_true",
        help="Every workflow.yaml § markers kind whose posted_by names a "
        "poller/watcher (poll_pipeline/backend_poll/slurm_monitor/"
        "autonomous_session_watch/pod_watch/tick_triage) must (Leg A) be "
        "referenced by >=1 consumer surface (.claude/skills/**/SKILL.md + "
        "the poller/triage scripts) and (Leg B) appear in each poster "
        "script its posted_by token names — a poller feature claiming "
        "mid-run surfacing with no consuming/posting code is the #873 "
        "pre-fix state. Waive via POLLER_CONSUMER_ALLOWLIST. Bundled into "
        "--check-references and the no-flags default run.",
    )
    parser.add_argument(
        "--check-skill-bang-backtick",
        action="store_true",
        help="FAIL on any non-dollar-preceded '!' directly against a "
        "backtick in .claude/{skills,agents,commands}/**/*.md — the skill "
        "preprocessor executes such a span as inline shell at load "
        "(#1243/#1266: two prose spans killed every /issue boot on "
        "2026-07-10). '$!' shell-pid prose is exempt. No waiver: reword "
        "instead. Bundled into the no-flags default run.",
    )
    parser.add_argument(
        "--check-agents-note-argv-verdict",
        action="store_true",
        help="FAIL on any .claude/agents/**/*.md line prescribing an "
        "argv-prose --note verdict/marker post opened as a command "
        "substitution — the pattern #1743 banned and rewrote to the "
        "post-marker --file channel (#1722/#1756 argv-substitution "
        "incident family; pinned by #1785). The sanctioned "
        "resolve-into-a-shell-variable-first form never matches. No "
        "waiver: reword instead (the #1743 r2 precedent). Bundled into "
        "the no-flags default run.",
    )
    parser.add_argument(
        "--check-sha-pin-domain",
        action="store_true",
        help="FAIL a whole-string 64-hex sha pin duplicated across >= 2 "
        "scripts/src modules when a site declares no content DOMAIN "
        "(undeclared copy — the #1776/#1491 wrong-domain class) or when "
        "sites declare conflicting domains (INDEX vs PROMPT, ...). Declare "
        "via an adjacent `# SHA_PIN_DOMAIN: <INDEX|IDS|PROMPT|BYTES|"
        "CONTENT>` comment or a domain token in the binding name; waive a "
        "site with `# SHA_PIN_DOMAIN_EXEMPT: <reason>`. Legacy sites are "
        "frozen as (hex12, file) pairs in SHA_PIN_DOMAIN_GRANDFATHER — a "
        "stale entry FAILs; conflicts have no allowlist escape. Bundled "
        "into the no-flags default run.",
    )
    args = parser.parse_args(argv)

    if args.regen_hf_routing_snapshot:
        # Maintenance flag (#1568): print-and-exit; never runs checks, never
        # loads workflow.yaml, never enters the no-flags bundle (combining
        # with check flags is unsupported — regen wins). Early dispatch
        # keeps stdout EXACTLY the paste-ready literal.
        return regen_hf_routing_snapshot()

    if args.regen_list_repo_files_snapshot:
        # Maintenance flag (#1624, the #1568 idiom): print-and-exit; never
        # runs checks, never loads workflow.yaml, never enters the no-flags
        # bundle (combining with check flags is unsupported — regen wins).
        return regen_list_repo_files_snapshot()

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
        or args.check_piped_git_push
        or args.check_push_failure_swallow
        or args.check_sh_function_rc_capture
        or args.check_grep_qv
        or args.check_marker_registry
        or args.check_agent_model_pins
        or args.check_agent_tools
        or args.check_upload_as_file
        or args.check_hub_dir_filecount
        or args.check_upload_prefix_clobber
        or args.check_upload_file_in_loop
        or args.check_upload_return_discard
        or args.check_dotenv_before_hf_import
        or args.check_batch_judge_client
        or args.check_hub_verify_retry
        or args.check_no_workflow_improver_spawn
        or args.check_no_repo_root_git_reset_hard
        or args.check_no_repo_root_worktree_revert
        or args.check_gate_ids_unique
        or args.check_lessons_index
        or args.check_inline_round_duty_mirror
        or args.check_rule_frontmatter_parses
        or args.check_compute_shape_review_lens
        or args.check_long_loop_restartability_review_lens
        or args.check_hollow_verification_gate_review_lens
        or args.check_smoke_architecture_review_lens
        or args.check_stale_label_disposition
        or args.check_smoke_output_hygiene
        or args.check_crash_fix_relaunch_contract
        or args.check_vm_thread_cap_guidance
        or args.check_awk_elision_parity
        or args.check_asw_docstring_pass_count
        or args.check_marker_recipe_snippets
        or args.check_judge_model_pins
        or args.check_live_hf_retry_routing
        or args.check_bare_list_repo_files
        or args.check_no_literal_round_marker_versions
        or args.check_agent_spec_size
        or args.check_agent_memory_index_size
        or args.check_gotchas_size
        or args.check_skill_doc_size
        or args.check_api_dispatch_routing
        or args.check_lens_coverage
        or args.check_section_reference_pointers
        or args.check_phase_done_reserved
        or args.check_jsonl_splitlines
        or args.check_scripts_import_guard
        or args.check_upload_or_true
        or args.check_git_recipes_root_guard
        or args.check_bare_commit_pathspec
        or args.check_marker_scalar_integrity
        or args.check_poller_marker_consumers
        or args.check_skill_bang_backtick
        or args.check_agents_note_argv_verdict
        or args.check_sha_pin_domain
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
        # A truncated marker-field scalar / an unreferenced poller-posted
        # kind is the same registry-integrity drift class — bundle both
        # here so the pre-commit hook (which fires --check-references on
        # any workflow.yaml change) catches them at commit time (#873).
        errors.extend(check_marker_scalar_integrity(workflow))
        errors.extend(check_poller_marker_consumers(workflow))
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
    if args.check_piped_git_push or no_flags:
        errors.extend(check_piped_git_push())
    if args.check_push_failure_swallow or no_flags:
        errors.extend(check_push_failure_swallow())
    if args.check_sh_function_rc_capture or no_flags:
        errors.extend(check_sh_function_rc_capture())
    if args.check_grep_qv or no_flags:
        errors.extend(check_grep_qv())
    if (args.check_marker_registry or no_flags) and not args.check_references:
        errors.extend(check_marker_registry(workflow))
    if (args.check_marker_scalar_integrity or no_flags) and not args.check_references:
        errors.extend(check_marker_scalar_integrity(workflow))
    if (args.check_poller_marker_consumers or no_flags) and not args.check_references:
        errors.extend(check_poller_marker_consumers(workflow))
    if args.check_agent_model_pins or no_flags:
        errors.extend(check_agent_model_pins())
    if args.check_agent_tools or no_flags:
        errors.extend(check_agent_tools())
    if args.check_upload_as_file or no_flags:
        errors.extend(check_upload_as_file())
    if args.check_hub_dir_filecount or no_flags:
        errors.extend(check_hub_dir_filecount_guard())
    if args.check_upload_prefix_clobber or no_flags:
        errors.extend(check_upload_prefix_clobber())
    if args.check_upload_file_in_loop or no_flags:
        errors.extend(check_upload_file_in_loop())
    if args.check_upload_return_discard or no_flags:
        errors.extend(check_upload_return_discard())
    if args.check_dotenv_before_hf_import or no_flags:
        errors.extend(check_dotenv_before_hf_import())
    if args.check_batch_judge_client or no_flags:
        errors.extend(check_batch_judge_client())
    if args.check_hub_verify_retry or no_flags:
        errors.extend(check_hub_verify_retry())
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
    if args.check_inline_round_duty_mirror or no_flags:
        errors.extend(check_inline_round_duty_mirror())
    if args.check_rule_frontmatter_parses or no_flags:
        errors.extend(check_rule_frontmatter_parses())
    if args.check_agent_spec_size or no_flags:
        errors.extend(check_agent_spec_size())
    if args.check_agent_memory_index_size or no_flags:
        errors.extend(check_agent_memory_index_size())
    if args.check_gotchas_size or no_flags:
        errors.extend(check_gotchas_size())
    if args.check_skill_doc_size or no_flags:
        errors.extend(check_skill_doc_size())
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
    if args.check_crash_fix_relaunch_contract or no_flags:
        errors.extend(check_crash_fix_relaunch_contract())
    if args.check_vm_thread_cap_guidance or no_flags:
        errors.extend(check_vm_thread_cap_guidance())
    if args.check_awk_elision_parity or no_flags:
        errors.extend(check_awk_elision_parity())
    if args.check_marker_recipe_snippets or no_flags:
        errors.extend(check_marker_recipe_snippets())
    if args.check_judge_model_pins or no_flags:
        errors.extend(check_judge_model_pins())
    if args.check_live_hf_retry_routing or no_flags:
        errors.extend(check_live_hf_retry_routing())
    if args.check_bare_list_repo_files or no_flags:
        errors.extend(check_bare_list_repo_files())
    if args.check_no_literal_round_marker_versions or no_flags:
        errors.extend(check_no_literal_round_marker_versions())
    if args.check_api_dispatch_routing or no_flags:
        errors.extend(check_api_dispatch_routing())
    if args.check_lens_coverage or no_flags:
        errors.extend(check_lens_coverage())
    if args.check_section_reference_pointers or no_flags:
        errors.extend(check_section_reference_pointer_coverage())
    if args.check_phase_done_reserved or no_flags:
        errors.extend(check_phase_done_reserved())
    if args.check_jsonl_splitlines or no_flags:
        errors.extend(check_jsonl_splitlines())
    if args.check_scripts_import_guard or no_flags:
        errors.extend(check_scripts_import_guard())
    if args.check_upload_or_true or no_flags:
        errors.extend(check_upload_or_true())
    if args.check_git_recipes_root_guard or no_flags:
        errors.extend(check_git_recipes_root_guard())
    if args.check_bare_commit_pathspec or no_flags:
        errors.extend(check_bare_commit_pathspec())
    if args.check_asw_docstring_pass_count or no_flags:
        errors.extend(check_asw_docstring_pass_count())
    if args.check_skill_bang_backtick or no_flags:
        errors.extend(check_skill_bang_backtick())
    if args.check_agents_note_argv_verdict or no_flags:
        errors.extend(check_agents_note_argv_verdict())
    if args.check_sha_pin_domain or no_flags:
        errors.extend(check_sha_pin_domain())

    if errors:
        for err in errors:
            sys.stderr.write(f"workflow_lint: {err}\n")
        sys.stderr.write(f"workflow_lint: FAIL ({len(errors)} error(s))\n")
        return 1

    sys.stderr.write("workflow_lint: PASS\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
