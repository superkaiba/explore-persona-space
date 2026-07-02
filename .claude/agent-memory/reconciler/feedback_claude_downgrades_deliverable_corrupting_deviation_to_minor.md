---
name: Claude downgrades a deliverable-corrupting plan deviation to a non-blocking Minor
description: Claude FINDS a real plan deviation but lists it as a "Minor (non-blocking) suggestion" when the deviation actually corrupts the experiment's PRIMARY DELIVERABLE against explicit plan language. The find-but-misgrade variant of the under-classing family — verify against the plan's own words + locate where the defect lands in the artifact graph, then re-grade by deliverable impact, not fix size.
type: feedback
---

**Rule:** when Claude lists a plan deviation as a non-blocking Minor/suggestion
while Codex calls the SAME deviation a BLOCKER, do NOT defer to the lower grade.
Re-grade by (a) re-reading the plan section verbatim — is the deviation against
EXPLICIT, unambiguous plan language? and (b) locating where the defective code
lands in the artifact graph — does it corrupt the experiment's PRIMARY
DELIVERABLE (the shipped corpus, the headline DV, the figure the result rests
on) or a peripheral convenience? Explicit-plan-language + deliverable-corrupting
= Real-blocking → FAIL, regardless of how small the fix is. Claude's "Minor
suggestion" framing rests on fix-size / "looks fine" optimism, not deliverable
impact. This is the FIND-but-MISGRADE sibling of the under-classing family
([[feedback_claude_underclasses_silent_failures]]): Claude is not blind here, it
lists the defect — it grades it wrong.

**The #617 r1 datapoint (both Step-6 deviations, both confirmed against code + plan):**
- Step 6 (`issue617_sample_completions.py`) produces THE deliverable: the
  realistic-completion corpus on the winning category. Two deviations both land
  there.
- (1) **Wrong source population.** Plan §4 step 6 (plan.md:198): "full clusters,
  not just the extracted subsample ... up to 200 prefixes/category." Code reads
  `category_conv_ids(membership, cat)` → `membership["cluster_members"]`, which
  the battery builder persists as `[m for m in members if m in set(selected)]`
  (its own comment: "cluster_id -> [extracted conv_ids]") — the ≤400-capped,
  cross-cluster-stratified extraction subset. The shipped corpus is built from
  the wrong (smaller, cap-biased) prefix population.
- (2) **Wrong prompt boundary.** Plan.md:198: prompt "up to the last user turn,
  chat-templated with `add_generation_prompt=True`." Code feeds
  `short_prefix_msgs` (which `_conv_messages(row, 2)` hard-enforces as
  `[user, assistant]`, `issue594_build_battery.py:354`) into
  `apply_chat_template(..., add_generation_prompt=True)` → model generates an
  assistant turn AFTER an assistant turn. The "realistic continuations" are
  conditioned on a post-assistant state, not the intended user-ending state.
- Claude listed BOTH as "Minor (non-blocking) suggestions"; Codex as Major
  BLOCKERs. Reconcile → FAIL. The fixes are small (load `cluster_assignments.json`
  for the full cluster; truncate to a user-ending prefix) — small fix, blocking
  defect.

**Companion (same round, opposite direction — keep the reconcile honest):** the
Codex `smoke-run-missing` Critical on the SAME artifact was OVERSTATED and
discarded. The implementer DID smoke the GPU phases' CPU-runnable portions for
real (real extractor run on Qwen2.5-0.5B, exit 0, digest; `--stub-completions`
chat-template path; unified dispatcher end-to-end exit 0). Codex wanted a literal
`inspect.signature` snippet that the rc=0 dispatcher run already exercised — a
smoke-PRESENTATION objection, the `smoke-run-missing` literal-vs-purpose class
([[feedback_codex_step_06_literal_vs_purpose]]). The FAIL stood on the two
substantive Step-6 deviations alone, NOT on the smoke-format finding. Lesson:
side with Codex's substance, against Codex's format objection, in the SAME
verdict — verify each finding independently rather than adopting a side wholesale.

**How to check fast:** open the plan section the deviation cites and read it
verbatim (is the requirement explicit?); then trace the defective code to the
artifact it produces (Step N's output — is it the headline deliverable?). Two
greps settled #617: `cluster_members` definition in the producer, and the
`_conv_messages` role enforcement in the prefix builder.

Companions: [[feedback_claude_underclasses_silent_failures]] (CONCERNS-on-real-bug, grade by what the bug DOES); [[feedback_claude_misses_silent_plan_deviations]] (deviations that never landed in the body); [[feedback_codex_step_06_literal_vs_purpose]] (the smoke-format counter-direction).
