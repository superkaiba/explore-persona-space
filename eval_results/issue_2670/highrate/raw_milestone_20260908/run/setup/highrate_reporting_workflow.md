# Task 2670 final-report preparation — not executed

This is a bounded handoff for the already approved experiment. It does not authorize another experiment, extra fitting regimes, task promotion, source changes, uploads or deletion. The user-facing final answer should directly state whether pre-action probing worked and whether the map helped, with the central uncertainty and counts; canonical workflow details can stay internal.

The task CLI snapshot says workflow v1, kind experiment, no paper flag, status followups_running, and has_clean_result=true. Consequently use clean-result-v4 and update the existing clean result in place without taking a new original-body snapshot. Preserve the declared Goal exactly. The skeleton is intentionally not publishable: only pending numerical results use PENDING_METRIC tokens, while comments specify remaining factual authoring work.

## Minimum completion requirements

1. Finish and validate the four approved VM regimes; confirm terminal exit, fresh output/config/source identity, exact sample support, finite metrics, held-out predictions, fitting/affine checks and all planned cells. Re-read final values when composing the result. A missing regime is missing, never zero. No additional statistical direction is required by this handoff.
2. Use the existing final dataset/support audit for stable counts, after confirming its bound native/postrun/selection artifacts remain unchanged. Keep one fresh transport unknown and eight structurally unassessable raw failures distinct. Report conditional complete-case estimates; neither numerical significance nor adequate class support overrides the failed strong claim gate.
3. Replace only actual pending metric values, rewrite the scientific H1/Takeaways to match them, complete exact examples and direct reused-map methodology from existing validated provenance, and create the required concise result/per-task visuals from the same result artifacts. Every displayed figure needs a browser-accessible pinned URL, figure source and data/provenance metadata. Do not guess old map training details or claim new map-reconstruction metrics.
4. Preserve earlier validated rounds as compact historical evidence while making the current four-regime answer the main synthesis. The final v4 body has precisely Takeaways, Goal, Methodology, Results as flat H2 sections, then Repro/Context footer. Use the real originating prompt and real round labels; the inspected event history has no followup-scope label to copy.
5. Archive and independently read back the final fits/predictions/metrics/figures/reviews before claiming completion. The root already owns raw/capture archive reconciliation and managed GPU release; do not repeat or broaden that operation. Save structured outputs; if the configured dashboard logging fails, state the failure and retain canonical artifacts rather than inventing a successful run link.
6. Run the actual mechanical body and prose-discipline checks below on a completed external draft. Obtain independent scientific/numerical/figure review of that exact draft and source-bound artifacts. No Claude automation. Correct concrete issues and rerun affected checks only.
7. Land the completed body and title through task.py without --snapshot, read back through task.py, and verify the canonical issue from main. Commit only explicit owned paths, record the fresh body commit proving this follow-up was folded, and run scoped workflow_lint for any workflow-surface changes. Preserve existing original-body.md.
8. After the result critic passes, mechanically copy the final Methodology section into docs/methodology/issue_2670.md, normalize only its heading, explicitly commit/push the doc, and refresh the top-of-body SHA-pinned document pointer. The old methodology marker/link cannot stand in for this new content. The secret gist mirror is fail-soft; a real published URL is required if shown. Recheck the resulting body after pointer insertion.
9. Record truthful completion/analysis/interpretation/review evidence through existing task CLI marker conventions. Refresh task state before parking; retain followups_running until the round is complete, then use awaiting_promotion. Classification/promotion remains the user's action. Do not use --force-followup-exit or promote automatically.

## Actual CLI commands to use later

These command shapes were read from the installed CLI. None were executed with mutating arguments by this reviewer. Set the completed body path only after the draft is genuinely finished:

```bash
completed_body=/home/thomasjiralerspong/explore-persona-space/eval_results/context_risk/impossible_highrate/setup/highrate_clean_result_completed.md
UV_NO_SYNC=1 uv run python /home/thomasjiralerspong/explore-persona-space/scripts/task.py view 2670 --json
UV_NO_SYNC=1 uv run python /home/thomasjiralerspong/explore-persona-space/scripts/verify_task_body.py --file "$completed_body"
UV_NO_SYNC=1 uv run python /home/thomasjiralerspong/explore-persona-space/scripts/audit_clean_results_body_discipline.py "$completed_body"
```

After checks and the exact-draft review, perform the already authorized final fold from the main checkout with the correct concrete title:

```bash
UV_NO_SYNC=1 uv run python scripts/task.py set-body 2670 --file "$completed_body"
```

Then `task.py set-title 2670` takes the exact completed H1 text as its title argument. `task.py set-clean-result 2670` is available but the marker is already true; do not treat an idempotent marker write as evidence the new result passed. Commit the explicit task path reported by the current CLI, then reread and verify:

```bash
UV_NO_SYNC=1 uv run python scripts/task.py view 2670 --json
UV_NO_SYNC=1 uv run python scripts/verify_task_body.py --issue 2670
```

`task.py post-marker 2670 MARKER --file RECEIPT_PATH --by REVIEWER` is the actual marker interface; choose the genuine existing marker kind and actual reviewer/evidence, not these symbolic labels. It auto-increments marker versions. Do not use its nonconforming-report waiver. For any `epm:results` or `epm:experiment-implementation` report, the CLI requires the actual four lettered completion-report sections; write the concrete report before posting.

Once the full round, review, methodology export and archival validation are finished:

```bash
UV_NO_SYNC=1 uv run python scripts/task.py set-status 2670 awaiting_promotion --note "Completed the approved high-rate screening, fresh collection, capture, four analysis regimes, independent result review and verified archival; classification awaits the user."
```

That note is appropriate only if every stated step really succeeded. No task promotion is part of these commands.

## Rule interpretation and limits

The applicable sources are .claude/rules/after-every-experiment.md, clean-results/SPEC.md v4/follow-up rules, analyzer.md final-fold rules, and research-project-structure.md, read from origin/main. More-specific same-issue rules override the generic first-result --snapshot example. The existing clean result must be folded without overwriting its original snapshot. Paper/v2 conversion is inapplicable.

The generic checklist mentions broad living-document updates. The analyzer's specific instruction makes headline RESULTS.md user-owned/propose-only. This handoff authorizes no RESULTS.md edits or broader pipeline expansion. An index update, if the operator performs the repository's existing end-of-run bookkeeping, must describe only these actual outputs. Do not spawn new research, new follow-up proposals or experiments to satisfy this preparation task. Any optional literature-positioning remains separate and nonblocking; no research or posting was performed here.

The template is not a finished clean result and must not be passed to set-body. It has no guessed metrics, invented artifact URLs or new outcome adaptations. Its authoring comments are a checklist, not claimed completed work.
