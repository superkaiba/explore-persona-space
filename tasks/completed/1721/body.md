---
title: 'daily-fix: planner grounding — run grep-answerable claims, t'
kind: infra
tags:
- wf-fix
- wf-fix-fp:29418af1fe8a
- daily-auto-filed
created_at: '2026-07-27T07:15:57Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-26 problem sweep (route 2): plans shipped a wrong call-site
  line number, a nonexistent library API, a silently dropped required edit, an acceptance
  criterion its own instrument could not measure, and shell whose failure arm did
  not halt the enclosing block'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily 2026-07-26 problem sweep (route 2). Surfaced by 5 independent
miner group(s) over the 2026-07-26 session transcripts.

## Goal

Extend the planner's grounding discipline from hyperparameters to the four grep-answerable
claim classes that failed on 2026-07-26 — tool behavior, symbol existence, cited line
numbers, and task-body-named required edits — and require tracing the exit path of any
shell control flow a plan embeds.

## Workflow gap

- **Bug observed:** Seven grounding failures across five sessions on 2026-07-26, each costing a plan-revision or fact-checker round: a wrong line number for a symbol's second call site, a nonexistent library API plus a wrong file citation, two independent false claims about `workflow_lint.py`'s no-flags bundling, a silently dropped required doc edit, an acceptance criterion its own instrument could not measure, and a bare `false` in embedded shell that does not halt the enclosing block.
- **Why it is a workflow gap:** `.claude/agents/planner.md` §11 requires a `Source:` line for load-bearing HYPERPARAMETERS only; nothing extends that bar to a claim about what a repo script does, whether a symbol exists, or which line a call site sits on — and no §4 pre-return check asserts that every edit the task body names verbatim appears in the plan.
- **Confidence (emitter):** high
- verified-at-filing: absence probes against BOTH the spec and the section reference it defers to — `grep -ci -- '<pat>' .claude/agents/planner.md .claude/rules/planner-section-reference.md` for `symbol-existence`, `grep -rn`, `tool-behavior`, `line number`, `scope shrinkage`, `control flow`, `exit path` → 0 hits in both files for all seven patterns; `acceptance criterion` → 2 hits in planner.md (L602-604), both inside the count-style self-count rule, 0 in the reference. §11 (L511-523) scopes `Source:` to "one entry per load-bearing hyperparameter"; §12 (L525-557) requires Confidence/Source/How-to-verify but no verbatim grep output; §4 (L333-350) lists hard-requirement items with no task-body-edit enumeration; §7 (L396-408) requires gates be "jointly satisfiable" and precedent-coherent but never names the measuring instrument (2026-07-26)

## Evidence

- Session `7df6ce4c`, 09:48:27Z: plan §12 assumption A1 asserted `recommended_timeout_s` has call sites at L1669 and L466; the fact-checker found the second at L1725 — `"**A1: WRONG.** recommended_timeout_s has TWO active call sites, but they are at **L1669** (--map-files branch — CORRECT) and **L1725** (the diff-path/invariant-set …"`. Cost: one fact-check finding plus two plan-patch turns, the second of which itself failed.
- Session `06447a89`, 07:21:55Z: the fact-checker returned HAS_WRONG on 2 of 11 assumptions — `"§4 pseudocode uses task_workflow.load_registry — actual API is private _load_registry … §12 assumption 12 cites SKILL.md files for the phrase — actually only in CLAUDE.md line 119."` The plan had itself flagged the API at MEDIUM confidence and deferred it to "the implementer should inspect the exact API surface at implementation time" rather than grepping it. Cost: 4 `plan_patch.py` invocations.
- Session `06447a89`, 07:29:47Z: task #1691's body (lines 87-88) required TWO CLAUDE.md edits; plan v5 §4.8 named only one. Three lens critics APPROVEd; only the consistency-checker caught it — `"the plan §4.8 names ONLY the § Autonomous-session watcher edit and OMITS the § Codex ensemble review edit. This is a scope-shrinkage MISMATCH."` Cost: one extra plan revision (v5→v6); a half-fix would have shipped without that one reviewer in the batch.
- Sessions `c0319d9e` (11:45:47Z) and `7ce3a81f` (07:23:48Z), independently: both plans asserted that `workflow_lint.py`'s no-flags default run bundles `--check-asks`; it does not (`main()` gates it on `if args.check_asks:` alone). `"the plan Assumption 5 said --check-asks IS bundled — that specific detail is **WRONG**"`. The same claim class, wrong the same way, in two unrelated plans on one day — asserted rather than grepped.
- Session `c0319d9e`, 11:55:39Z: the Statistics & Measurement critic returned REVISE because §7 criterion 2 claimed byte-identical anchor-sentence enforcement while §4.2 specified a COUNT check on 30-char anchor prefixes — `"criterion 2's automation claim is FALSE and criterion 2 is not mechanically enforced. This is exactly the acceptance-vs-instrument mismatch the Statistics & Measurement lens exists to catch"`. Cost: one Phase-3 revision round.
- Session `35d7c0fa`, 09:25:23Z: plan v1 for #1694 gated a push behind a diff-verify fence whose failure arm ended in a bare `false` — `"Edit A's terminal `false` in the diff-verify fence does NOT prevent the immediately-following push block … from running and overwriting the sentinel with `landed`."` The plan as written would have shipped the exact silent-success bug the task existed to fix; caught at methodology-critic round 1, ~7 min for the extra round.

## Proposed change

- §11: extend the `Source:` bar beyond hyperparameters to TOOL-BEHAVIOR claims — any assertion about what a repo script, lint, or CLI does under a given invocation carries a `Source:` naming the grep or `file:line` actually read, and the planner RUNS the one-line mechanical check (`--help`, a grep of the flag's `main()` chain) rather than reading the pattern by eye.
- §12: any assumption naming a line number quotes the `grep -n '<symbol>'` output verbatim in the row. A line number written without its grep output is not grounded.
- §4 (or §12, whichever fits the byte budget): forbid deferring a grep-answerable symbol-existence question to the implementer — every `module.symbol` written into plan pseudocode is confirmed at plan time with a recorded `grep -rn 'def <symbol>' src/ scripts/`; a symbol that does not resolve is rewritten, not shipped at MEDIUM confidence.
- §4 pre-return self-check: before returning, enumerate every file + section the task body names as a required edit and assert each appears in §4 Design; list any deliberate omission with a one-line reason. This is the consistency-checker's catch made a planner-side duty.
- §7: state, per acceptance criterion, WHICH §4 mechanism measures it and what that mechanism actually compares (count vs equality vs presence). The existing L602 self-count rule covers count-style criteria only.
- Plan-quality: when a plan embeds shell control flow, trace the exit path of every failure arm and state what halts the enclosing block — a bare `false` or `exit` inside a branch does not stop a sibling block; prefer an explicit `OK=yes|no` variable gate over `&&`/exit-status chaining.
- BYTE BUDGET — binding placement constraint: `.claude/agents/planner.md` is 38,368 B against `AGENT_SPEC_FAIL_BYTES = 40,000` (`scripts/workflow_lint.py` L11740) and is deliberately NOT grandfathered (the `AGENT_SPEC_SIZE_GRANDFATHER` header names planner.md and critic.md as excluded, #838). Headroom is 1,632 B for all six additions. Put the full templates in `.claude/rules/planner-section-reference.md` (75,176 B, the on-demand file §4/§7/§11/§12 already defer to) and keep the spec-side text to one compressed clause per section.

## Scope / surfaces

- Primary target: `.claude/agents/planner.md`
- `.claude/rules/planner-section-reference.md` (§4 / §7 / §11 / §12 full templates — the byte-budget-safe home for the expanded text)

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `uv run python scripts/workflow_lint.py` passes (no-flags); ruff clean on touched files.
- This session runs under a `workflow_fix_target:` Provenance line — it MUST NOT auto-route
  its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- sha-verify (filing-time, #1467): `35d7c0fa` cited in commit context does NOT resolve as a commit in this repo at filing time — treat as a transcript/session reference, not a commit.

- fingerprint: 29418af1fe8a

- workflow_fix_target: .claude/agents/planner.md
- fingerprint: PENDING

/daily 2026-07-26 route-2 filing. Miner refs: E-P5, E-P6, C-P9, C-P10, I-P11, H-P10, G-P13.
