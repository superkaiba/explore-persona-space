---
name: numleak-handed-span-set
description: Which files must enter the numeric-leak verifier's handed-span set to avoid false-positive residuals (brief text, spec template lines, path strings)
metadata:
  type: feedback
---

For the Step-4 numeric-leak verifier, the handed-span multiset needs FOUR
classes of files, not just plan + lens extracts: (1) the plan snapshot,
(2) every verbatim lens extract exactly as concatenated into the prompt,
(3) the ORCHESTRATOR BRIEF text verbatim (its claims-to-verify / emphases
numbers are handed, not fabricated — quote them nearly verbatim in the
prompt so they cancel), and (4) the composer's OWN SPEC Step-3 template
lines (the template hands "§9", ">50 GB", the "25 models × 3 traits …
8×H100 … 1/8 util" closing example, and the "items 10 / 13 / 16"
designators — without this file those scaffold atoms false-positive every
round). Path strings (plan/manifest/script paths) also enter as handed
spans so v<K>/issue-id atoms inside paths clear.

**Why:** first run on #2329 round 1 flagged 9 residual atom classes; all
but one traced to spec-template scaffold, and the one real catch was a
composer-typed second "Qwen3.5" in a read-target annotation — rephrasing
to a version-free description ("the ported-model rig") is the fix pattern
for model-version numerals in composer-authored glue text.

**How to apply:** build the handed-span set from all four classes BEFORE
the first verifier run; treat any remaining residual as a genuine
composer-authored numeral and rephrase the glue text rather than widening
the allowlist. Run the verifier from the repo root (not /tmp) so the
registry leg (`task_workflow.registry_path()`) imports; from /tmp it
degrades to handed-span-only clearing.

**Multiset-overshoot pattern (#2389 r3):** even fully-handed numbers
residual out when GLUE REPEATS them more times than the handed spans
contain them — the classic shape is a read-targets/file-access list
duplicating line numbers + constants that the blocker-prescription
paragraph already quotes (also over-using phase tags like a `P7g` or
`§9` beyond the handed count). Fix: state seam line-numbers/values ONCE
(in the near-verbatim prescription quote) and make read-target bullets
number-free pointers ("exact line numbers in the prescription below"),
rather than padding handed-span files with duplicates.

Multiset nuance (#2329 round 2, 9 residual classes): a number HANDED by
the plan still residuals when the prompt INLINES that plan — the inline
copy consumes the span occurrence, so every glue reuse is excess under
multiset subtraction. Fix = reference-not-restate: point Codex at the
artifact's own figure ("the G4a floor-satisfiability note's figure",
"§ Decision Rationale") instead of repeating the numeral, and quote each
brief/blocker figure at most as many times as the brief/blocker span
carries it. Section sigils count too: each glue "§9"/"§7"/"v7" emits an
atom, so name sections by title where the count is tight.
