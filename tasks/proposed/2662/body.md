---
title: 'graded_judge slot fill rescans inserted text: land the single-pass fix on
  main'
kind: infra
tags:
- shared-module-propagation
created_at: '2026-09-03T09:03:14Z'
has_clean_result: false
parent_id: 2658
origin_prompt: 'Found while unblocking task 2658 phase P2: the shared graded_judge
  slot fill chained two .replace calls, so a question carrying the literal answer
  placeholder got the answer spliced into the question block of the wire message.
  Fixed on issue-2658; main is still stale.'
workflow: v1
---
---
kind: infra
---

# `graded_judge.format_user_msg` rescans inserted text: land the single-pass slot fill on `main`

## Goal

Land three small hunks that already exist and are tested on branch `issue-2658`
(commit `77352ccb7eac4e29765ff624fd740114c8305572`) onto `main`, so the shared
judge helper stops splicing answer text into the question block of the wire
message. Scope is the shared helper only. Nothing else from `issue-2658` comes
with it.

## The defect (live on `main` right now)

`src/explore_persona_space/eval/graded_judge.py:330`:

```python
return eval_prompt.replace("{question}", question).replace("{answer}", answer)
```

The second `.replace` rescans the text the first one inserted. So whenever a
judged item's QUESTION text itself contains the literal `{answer}`, the answer
gets substituted INSIDE the question block, and the judge sees a corrupted
message: the question it is asked to score already carries the answer, and the
real answer slot is filled from text that was never meant to be a slot.

Measured exposure on task #2658's pilot set: 5 of 6,290 records, all the same
underlying real-user item, under the five rows that use the `wildchat_real`
frame. 0 answer-side hits. So the corruption rate is low but non-zero, and it is
concentrated exactly in real-user corpora, which is where the project increasingly
sources questions.

Blast radius on `main`: 85 files reference `graded_judge` (71 under `scripts/`,
14 under `src/`), and it is the sanctioned graded-judge path for every judged DV
in the project (sycophancy, refusal, hallucination, trait, EM, correctness).

## The fix, as already landed on `issue-2658`

Three hunks in `src/explore_persona_space/eval/graded_judge.py`:

1. `import re` in the import block.
2. A module-level compiled pattern:
   ```python
   _SLOT_RE = re.compile(r"\{(question|answer)\}")
   ```
3. `format_user_msg` fills both slots in ONE left-to-right pass:
   ```python
   _fill = {"question": question, "answer": answer}
   return _SLOT_RE.sub(lambda m: _fill[m.group(1)], eval_prompt)
   ```

**The CALLABLE replacement is load-bearing, not stylistic.** A single-pass
`re.sub` with a STRING replacement would interpret backslash and group
references (`\1`, `\g<0>`) appearing in real-user text, which is a corruption
class the old `.replace()` chain never had. The callable form fixes the rescan
while preserving `.replace()`'s literal-replacement safety. Do not simplify it
to a string replacement.

Note the nesting constraint: `format_user_msg` is defined INSIDE `judge_graded`
and closes over `eval_prompt`, so the compiled pattern must live at module level
while the substitution call stays nested.

## Required verification before landing

1. **Cache and fingerprint neutrality must be ASSERTED, not assumed.**
   `rubric_fingerprint(judge_model, judge_system_prompt, format_user_msg)`
   sentinel-renders the user-message template. If the sentinel values contain no
   literal `{question}`/`{answer}`, old and new implementations render
   byte-identically and every existing judge cache key is preserved. Verify this
   holds; if it does not, the change forces a fleet-wide cold re-judge and that
   consequence must be surfaced before landing, not discovered afterwards.

2. **A committed round-trip test** in `tests/` per `.claude/rules/llm-judging.md`
   rule 27: push a question containing the literal `{answer}` through the
   substitution and assert the answer is NOT spliced into the question block,
   plus the ordinary no-placeholder case. Assert the MECHANISM (a sentinel
   round-trip), never an output-scoped "no residual `{answer}`" check: such a
   check would false-fire on exactly the legitimate real-user items this fix
   exists to preserve.

3. **Bounded exposure audit of already-judged waves.** Determine whether any
   COMPLETED judge wave scored questions whose text contains the literal
   `{answer}`. Bound it: grep the committed judge inputs / raw completion
   mirrors under `eval_results/` for the literal token rather than re-reading
   corpora, and report counts per issue. If any affected result reached a
   PROMOTED task body, file that separately per the record-integrity duty; do
   not silently correct or silently ignore it. A zero-hit audit is a valid and
   useful outcome, but state which artifacts were swept.

## Non-goals

- Do NOT bring any other `issue-2658` content onto `main`. That branch carries a
  full in-flight experiment; its merge is owned by its own `/issue` Step 10d.
- Do NOT touch the roughly ten per-issue judge scripts that carry their own
  sequential-replace `format_user_msg` closures (for example
  `scripts/issue1739_compliance_pilot.py`). Those are frozen historical
  instruments; changing them would alter completed measurements. Whether any of
  them warrants a forward fix is a separate, per-issue decision.
- Do NOT re-run or re-judge any completed wave as part of this task. Landing the
  helper fix and auditing exposure are the deliverables; any re-judge is a
  separate, costed decision.

## Provenance

Found and fixed while unblocking task #2658 phase P2. The per-issue half of the
fix (replacing a data-shaped guard with a mechanism assert bound to the live
closure) landed with the same commit and is out of scope here. Filed under the
shared-module propagation duty in `.claude/rules/crash-fix-rounds.md`, because
the fix currently exists only on `issue-2658` while `main` and every future
judged DV read the stale helper.
