---
title: 'daily-fix: select_step9c_tests --json/--map-files ergonomics'
kind: infra
tags:
- wf-fix
- wf-fix-fp:7e501bf395c5
- daily-auto-filed
created_at: '2026-07-27T07:14:30Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-26 problem sweep (route 2): --map-files silently ignores
  --json so consumers die on json.load, --json output is corrupted when stderr NOTEs
  are redirected into stdout, the --map-files error does not name the expected input
  shape, and running on uncommitted edits silently degrades to the invariant-only
  fallback'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily 2026-07-26 problem sweep (route 2). Surfaced by 4 independent
miner group(s) over the 2026-07-26 session transcripts.

## Goal

Close four ergonomic defects in `scripts/select_step9c_tests.py` — `--map-files` silently
ignoring `--json`, `--json` output corrupted by stderr NOTEs, an unhelpful `--map-files`
read error, and a silent degrade to the invariant-only fallback on uncommitted edits — each
of which cost a live session a wasted turn today.

## Workflow gap

- **Bug observed:** (a) the `--map-files` branch is TSV-only and returns before any
  `--json` handling, so a consumer that passes both dies on `json.load`; (b) the script's
  informational NOTEs go to stderr by design, so a caller redirecting `2>&1` into
  `json.load` gets `JSONDecodeError`; (c) an unreadable `--map-files` argument reports only
  the raw `Errno 2` without saying the flag wants a PATH to a newline-separated file, not a
  comma-separated list; (d) run on UNCOMMITTED edits the selector silently degrades to the
  workflow-invariant set only, and its NOTE names the wrong likely cause.
- **Why it is a workflow gap:** the selector is the mandated pre-report step for every
  implementer (`implementer.md` L174) and the Step 10d merge-gate mapping mode, so each of
  these fires on the copy-paste path rather than on an improvisation, and (d) silently
  voids the #1288 scope narrowing the step exists to buy.
- **Confidence (emitter):** high
- verified-at-filing: presence claims, per-target, in the named target
  `scripts/select_step9c_tests.py` —
  (a) `Read` L1644-1729: the `if args.map_files is not None:` branch (L1644) `print`s TSV at
  L1727-1728 and `return 0`s at L1729; `grep -n 'args.json' scripts/select_step9c_tests.py`
  → the ONLY consumer is L1786 (`if args.json:`), i.e. AFTER the map-files early return —
  `--json` is unreachable in mapping mode. **1 hit**, confirming the silent-ignore.
  (b) every NOTE/WARN/sizing line in the file is `file=sys.stderr` (L1662-1664, L1670-1673,
  L1707-1711, L1722-1726, L1747-1751, L1777-1783); the JSON is the only stdout write at
  L1786+. **Confirmed by construction.**
  (c) `sed -n '1658,1665p'` → `print(f"select_step9c_tests: cannot read --map-files input:
  {exc}", file=sys.stderr)` — no PATH/format hint. The `--map-files` `help=` at L1601-1618
  does say "newline-delimited repo-relative paths", but the ERROR does not. **1 hit.**
  (d) `grep -n 'empty diff\|falling back to the workflow-invariant' scripts/select_step9c_tests.py`
  → **2 hits** (L190 docstring, L1747-1751 the live NOTE); the NOTE text is
  `"…falling back to the workflow-invariant set only. If this task's changes live in an
  issue worktree, re-run from that worktree (Step 9c contract)."` — names ONLY the
  wrong-worktree cause, **0 hits** for any uncommitted-work mention.
  Secondary target: `grep -nic 'commit your edits first\|commit first' .claude/agents/implementer.md`
  → **0 hits** (absence claim); the recipe at L174 prescribes
  `uv run python scripts/select_step9c_tests.py --json` with no commit-first warning and no
  `2>/dev/null`. Landed-fix check:
  `git log --oneline --since='7 days ago' -- scripts/select_step9c_tests.py` → most recent
  touch `d79fa07b0e` (#1699, the pin-sweep mode); none of the 4 defects addressed.
  (2026-07-26)

## Evidence

- Defect (a). Session `7df6ce4c`, 2026-07-26T09:30:42Z: ran `--map-files <file> --json` and
  piped stdout into `json.load`; the consumer died with a bare traceback —
  `"Exit code 1 ⏎ Traceback (most recent call last): ⏎   File \"<string>\", line 3, in
  <module> ⏎   File \".../json/__init__.py\", line 293, in load"` — and the session had to
  re-derive the output shape by hand. Cost: one wasted turn plus a manual re-derivation.
- Defect (b). Sessions `0e2c3b21` (implementer subagent, 09:27:33Z) and `35d7c0fa`
  (implementer subagent, 09:36:16Z): both ran
  `select_step9c_tests.py --json 2>&1 | uv run python -c "json.load(sys.stdin)"`; the
  stderr NOTE lines landed ahead of the JSON and both crashed —
  `"json.decoder.JSONDecodeError: Expecting value: line 1 column 1 (char 0)"`. Both agents
  then independently hand-wrote the same workaround:
  `"# grab the json line\nlast_json = None\nfor line in raw.splitlines(): …"`. Counting:
  2 tool_result FIRING events (`is_error=true` results carrying `json.decoder.JSONDecodeError`,
  deduped per tool call, one per session's implementer). Cost: 2 wasted turns + 2 bespoke
  ad-hoc parsers in two independent sessions on the same day.
- Defect (c). Session `c0a2df1b`, 2026-07-26T06:35:26Z: passed the touched-file list to
  `--map-files` as a comma-joined string; the error surfaced only the raw `Errno 2` on the
  whole comma-blob — `"select_step9c_tests: cannot read --map-files input: [Errno 2] No
  such file or directory: '.claude/agents/experiment-implementer.md,.claude/agents/implementer.md,
  .claude/rules/crash-fix-rounds.md,scripts/workflow_lint.py,tests/test_issue_skill_marker_contract.py'"`.
  The session then wrote the list to `/tmp/m1682.txt` and re-ran. Cost: 1 wasted probe (~10 s).
- Defect (c), second shape. Session `a2c4bae3`, 2026-07-26T17:18:23Z: passed two bare paths
  instead of a list file → `"select_step9c_tests.py: error: unrecognized arguments:
  tests/test_autonomous_session_watch.py | === rc=0 ==="`. The wrapper printed
  `=== rc=$? ===` after a `… 2>&1 | tail -20` pipeline, so `$?` captured `tail`'s status and
  the turn reported `rc=0` on a failed command. Cost: one wasted turn; the masked `rc=0`
  could have been read as success.
- Defect (d). Sessions `0e2c3b21` (implementer, 09:27:38Z) and `35d7c0fa` (implementer,
  09:36:21Z): both ran the selector on uncommitted edits and got
  `"select_step9c_tests: NOTE — empty diff vs 'origin/main' in …/worktrees/issue-1694;
  falling back to the workflow-invariant set only. If this task's changes live in an issue
  worktree, re-run from that worktree"` — `reasons_summary: {'invariant': 65}`, i.e. the
  pin-sweep universe was NOT the gate's universe. Counting: 2 tool_result FIRING events of
  the empty-diff NOTE, one per session's implementer. The NOTE's own hint blames the wrong
  cause (wrong worktree) rather than the actual one (uncommitted edits). Cost: the pin-sweep
  ran against a possibly-wrong file set in both sessions; the Step 9c gate caught nothing
  this time, so no downstream damage — but the #1288 narrowing was not obtained.

## Proposed change

- **(a)** In the `--map-files` branch (`scripts/select_step9c_tests.py` L1644), fail loud
  when `--json` is also passed: `parser.error("--json is not supported with --map-files
  (mapping mode emits TSV: '<test>\\t<matched_path>' per line)")`. Emitting JSON from
  mapping mode is the alternative, but the consumers are TSV-shaped today (`implementer.md`
  L174 reads col-1; the Step 10d TG legs `sort -u` the stdout) — fail-loud is the smaller,
  safer change and matches the "gate must fail CLOSED when it cannot classify" comment
  already at L1652-1657.
- **(b)** Add an explicit stderr warning line printed alongside the JSON path (or in
  `--help`) stating: with `--json`, never redirect stderr into stdout — the NOTE / WARN /
  sizing lines are stderr BY DESIGN and will corrupt the JSON. Optionally add a
  `--json-only` / `--quiet` mode that suppresses the informational stderr lines for
  machine consumers. Then correct the copy-paste recipe in `.claude/agents/implementer.md`
  L174 to the safe invocation:
  `uv run python scripts/select_step9c_tests.py --json 2>/dev/null | …`.
- **(c)** In the `--map-files` `OSError` handler (L1660-1665), when the argument is
  unreadable AND contains a comma, append
  `(--map-files takes a PATH to a newline-separated file list, not a comma-separated list)`
  to the error. Independently, the argparse `unrecognized arguments` shape (bare paths
  instead of a list file) is worth a usage hint: either accept `nargs="+"` of paths, or
  keep the single-FILE contract and add the list-file hint to the flag's error path.
- **(d)** Extend the empty-diff NOTE at L1747-1751 to name uncommitted work as the FIRST
  likely cause: `"empty diff vs '<base>' in <root> — the selector diffs COMMITTED state
  against fetched origin/main, so uncommitted edits produce an empty diff. Commit first;
  if this task's changes live in an issue worktree, re-run from that worktree."` And in
  `.claude/agents/implementer.md` § gate-matched scope (L174), insert: **Commit your edits
  first** — the selector diffs committed state against fetched `origin/main`; run it on
  uncommitted edits and it silently degrades to the invariant-only set.
- **(e)** Related, from the same session's evidence: `.claude/rules/code-style.md` (or the
  CLAUDE.md piped-exit-code bullet, which today covers only `git push`) should carry the
  general rule that `$?` after a pipe is the LAST stage's status — use `set -o pipefail` or
  check `${PIPESTATUS[0]}` before echoing an rc. Optional here; name it so the planner can
  decide whether to fold it in or leave it for its own filing.

## Scope / surfaces

- Primary target: `scripts/select_step9c_tests.py`
- `.claude/agents/implementer.md` (commit-first line + the safe `2>/dev/null` invocation in
  the L174 recipe)
- `tests/test_select_step9c_tests.py` (pins for the new `parser.error` on
  `--map-files --json`, the widened empty-diff NOTE text, and the comma-hint error)
- `.claude/rules/code-style.md` (optional, per (e) — the post-pipe `$?` rule)

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `uv run python scripts/workflow_lint.py` passes (no-flags); ruff clean on touched files.
- `scripts/select_step9c_tests.py` is in `tests/test_ruff_policy.py`'s live-workflow-helper
  scope class — run
  `uv run pytest tests/test_ruff_policy.py::test_live_workflow_helpers_clean_under_full_ruleset -x`
  in addition to bare `ruff check` (verify the roster membership at plan time).
- Do NOT change the mapping mode's stdout SHAPE or its exit-code contract: empty stdout +
  exit 0 is the Step 10d gate's skip signal; exit 1 (unreadable input) and exit 2 (#1613
  zero-resolution) are load-bearing fail-closed codes.
- Do NOT move the informational lines from stderr to stdout — the TSV/JSON stdout purity is
  what the consumers depend on; the fix is a warning plus a corrected recipe, not a stream
  swap.
- This session runs under a `workflow_fix_target:` Provenance line — it MUST NOT auto-route
  its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- fingerprint: 7e501bf395c5

- workflow_fix_target: scripts/select_step9c_tests.py
- fingerprint: PENDING

/daily 2026-07-26 route-2 filing. Miner refs: G-P8, H-P4, C-P12, B-P12, H-P5.
