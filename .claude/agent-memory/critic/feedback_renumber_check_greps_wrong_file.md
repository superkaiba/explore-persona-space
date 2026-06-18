---
name: Renumbering check must grep the renumbered file
description: Infra acceptance tables — a "no renumbering regression" check that greps the file HOLDING the cross-reference (not the file being edited) passes identically in done-right and done-wrong worlds (#578)
type: feedback
---

Rule: when an infra plan's acceptance table guards "inserting the new item mid-list renumbers existing cross-references," the verification command must read the EDITED file's numbering/placement (e.g. `grep -n "^10\. \*\*<new item lead>"` + assert it sits after item 9's lead and before the next H3), NOT the un-edited file that merely contains the cross-reference string.

**Why:** #578's check 5 was `grep "Pre-Launch step 9" .claude/rules/gotchas.md` as the mitigation for its own High-likelihood risk ("renumbering breaks cross-refs"). gotchas.md is not touched by a mid-list insert in experimenter.md, so the grep passes whether or not the insert renumbered everything — zero bits. This is the literal REVISE definition: an acceptance criterion that cannot detect the failure it claims to detect.

**How to apply:** for any cross-file numbered-reference invariant, trace which file actually changes under the failure mode and demand the check read THAT file. Same family: a presence grep (`grep -n "<new item title>" file.md`) with expected value "1 hit in section X" cannot establish section membership — anchor on the list-number prefix and verify position relative to the section boundary instead.
