---
name: predicate-broadening-vs-existing-test-pins
description: "Guard-fix plans: replay the plan's FULL predicate (not the incident's wording) against the existing test corpus — a broadened predicate ('no NON-EMPTY text block' vs 'no text block') can flip committed behavior-pin tests the plan swears stay green (#2206 v3)"
metadata:
  type: feedback
---

When an infra plan adds a guard/branch whose predicate is BROADER than the
incident shape it names (e.g. #2206: incident = "text-block-free response",
predicate = "no NON-EMPTY text block", deliberately covering the SDK-#461
present-but-empty block), replay the FULL predicate — not the claim's
wording — against the existing test corpus before crediting any
"all existing tests stay green / branch unreachable" blast-radius claim.

**Why:** #2206 plan v3 §5 claimed "No existing test exercises a
text-block-free message (`_msg()` always builds one text block), so the new
branch is unreachable by them" — literally true, but the plan's own
predicate also fires on one-block-with-`text=""` messages, and SIX existing
#1470 `response_valid` tests dispatch exactly that shape
(`tests/test_api_dispatch.py` :1820/:1845/:1875/:1905/:1937/:1968); five
fail under the planned change, one (`test_response_valid_default_none_backcompat`)
is an explicit byte-identical PIN of the pre-fix behavior the task exists to
remove. The contradiction ("suite green" + tests pinning the old contract)
has a concrete FALSE-RESOLUTION: the implementer narrows the predicate
(drops `b.text != ""`) to keep the pins green — and if the plan's new-test
matrix lacks the batch-side empty-text-block twin (D7 had sync-only), the
narrowed batch site passes EVERY planned + existing test, shipping the bug
half-unfixed with a green suite. Survived planner + fact-checker (the
fact-checker corrected other counts in the same plan).

**How to apply (2-min checks):**
1. Extract the plan's guard predicate; enumerate its sub-shapes beyond the
   incident (empty list / wrong-type / present-but-empty / etc.).
2. Grep the existing test file for each sub-shape (`text=""`, `content=[]`,
   fake factories) — any hit that dispatches through the REAL library is a
   test the plan must name as a DELIBERATE update (with new expected
   records), never leave under "must stay green".
3. Check the plan's new-test matrix covers every sub-shape × every mint
   site (sync AND batch when the guard is duplicated) — a missing cell is
   the false-resolution escape hatch.
4. Counting-methodology sibling: a DOTALL non-greedy regex over whole files
   miscounts multi-site patterns (one `next(` match can swallow a later
   site). Recount per-occurrence with a bounded window (e.g. anchor on the
   opening token, cap the span at ~200-500 chars) before disputing or
   crediting a plan's site count (#2206: naive scan 18, bounded scan 20 —
   the plan's 20 was right).

Related: [[infra-plan-review-checklist]] (item C success-path-only tests;
item I grep-count replays); the #488 contradictory-gates shape (rubric
Statistics item 3) — this is its infra-plan instance where the "gate" is
the must-stay-green regression set.

Recurred: #2046 v1 (root-commit guard) — repointing the cd root-spelling
arms from the hardcoded repo constant to the test-overridable variable is
production-neutral, but hermetic pins that interpolate the CANONICAL root
as a resolved cd RHS (self-test B22 + tests/test_guard_root_code_commit.py
:1060-1063, via the `_CANONICAL_ROOT` literal at :50) flip block→allow
under the override: the canonical path now falls to the absolute-path
latch arm. The plan's "all existing rows stay green" enumerated the
cd-latch family but omitted B22. Check recipe: grep the test corpus for
the canonical-root literal in cd/RHS positions, then replay each hit
against the repointed case arm under the hermetic env.
