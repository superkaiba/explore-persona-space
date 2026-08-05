# #397 replay fixture — whack-a-mole detector worked test case

Relocated verbatim from `.claude/skills/issue/SKILL.md` Step 5.bis
(2026-08-05 compaction): a worked trace of the Step 5.bis whack-a-mole
detector (PRIMARY / SECONDARY triggers) on task #397's actual event
sequence. The detector is orchestrator prose (no code implementation),
so there is no executable test to attach this to; this note is the
reference trace for anyone changing the detector's trigger rules in
`.claude/skills/issue/SKILL.md` Step 5.bis. Not collected by pytest.

The detector's behavior on task #397's actual event sequence:

| Round | Implementer tag | Detector state after this round |
|---|---|---|
| 5 | (no tag — first complete dispatcher round) | 0 distinct, no fire |
| 6 | (no `epm:new-bug-class`; emits `epm:compute-deviation` from Fix #4 because wall-time 3-4× plan §9) | 0 distinct experiment-strategy classes — compute-deviation routes via Fix #4's pivot_criteria, NOT the whack-a-mole counter |
| 7 | (no tag — descope round) | 0 distinct |
| 8 | `epm:new-bug-class: vllm_teardown_oom` | 1 distinct, no fire |
| 9 | a workflow-fix candidate block (pod-side `task.py` shellout is a workflow-surface bug per the workflow-fix-on-bug protocol) | EXCLUDED from count — still 1 distinct experiment-strategy class (round 8's vllm), no fire |
| 10 | `epm:new-bug-class: subprocess_wrapper_missing_upload` | PRIMARY does not fire (need 3 distinct across the 3 most recent non-excluded rounds; only rounds 8 + 10 are non-excluded so only 2 distinct are available). SECONDARY DOES FIRE: 2 distinct tags across the 2 most recent non-excluded rounds (rounds 8 + 10; round 9 was excluded and is skipped, so 8 and 10 count as consecutive non-excluded) AND `epm:compute-deviation` at round 6 IS in the trailing 5-round window (rounds 6,7,8,9,10 from round 10's perspective). |
| 10' | Detector fires at the start of the would-be relaunch attempt — orchestrator surfaces 2-option prompt: `continue-as-planned (round 10 relaunch, cost: ~30 min, may hit next architectural assumption)` vs `pivot-to-in-process-serial (unify smoke and sweep paths, cost: one re-planning round, eliminates entire whack-a-mole class)`. User picks pivot — matches the actual round-11 decision. Route to `status:planning`. |

Key insight from the fixture: round 9's tag choice (workflow-fix
candidate vs new-bug-class) determines whether the detector fires at
round 10 via SECONDARY (workflow-fix exclusion path) or one round
later via PRIMARY. The SECONDARY trigger exists specifically to
catch the #397 shape one round earlier than PRIMARY would.
