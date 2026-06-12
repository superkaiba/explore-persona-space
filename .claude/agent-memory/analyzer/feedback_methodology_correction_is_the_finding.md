---
name: methodology-correction-is-the-finding
description: When a run's original headline was driven by an eval bug, the bug-and-recovery story IS the lead finding — not a buried caveat. Show both readings on the same adapter in the hero figure; demote the original "result" to "what the bug looked like"; never let the title carry the bugged claim.
metadata:
  type: feedback
---

When the user-supplied analyzer prompt says "this experiment had a dramatic
arc — bug, retraction, recovery — represent honestly," the methodology
correction IS the story, not a side-note.

**Rule.** When a run's original headline reading was driven by a silent eval
bug AND a recovery re-eval has the real numbers, the clean-result MUST:

1. **Lead the title with the post-correction finding**, not the bugged one
   (e.g. "decoupling goal not met because count↔implant entanglement is
   structural"), with the bug-itself folded into the body — never a
   "bugged"-prefixed title unless the user explicitly directs it (see
   `feedback_bugged_experiment_title_exception` for the rare override).
2. **Make the FIRST `#### <finding>` H4 the bug-and-recovery story**, with
   the hero figure showing the same trained adapter under two eval rigs
   (artifact reading on the left, recovery reading on the right). This
   anchors the rest of the body — every later finding rests on the
   recovered numbers.
3. **Spell out what the bug invalidated and what it did NOT** (training
   succeeded; only the eval reading was wrong). Reads as honest, prevents
   the reader from globally discounting the run.
4. **Cite the guard / fix** if one was added (`eval_guard.py`'s
   `assert_adapter_actually_applied` for the LoRA-not-applied class) so
   the reader sees the bug class won't recur.

**Why:** Burying a methodology correction in a Confidence sentence or a
Reproducibility note (a) misleads the mentor about what to trust, (b)
hides the work the recovery took, (c) costs a critic round when the
reviewer flags the underclaim.

**How to apply:** Use the bug-vs-recovery contrast as the FIRST hero
figure inside `### Findings`. The setup paragraph names the bug class
(silent LoRA-not-applied, wrong-fact paraphrase pool, dataset-mapping
bug); the figure shows the dispositive contrast (same input, two
readings); the read paragraph names what the bug invalidated and what it
did NOT. Subsequent findings open with "Once the recovery eval was in
hand, …" so the reader knows they're reading recovered numbers.

Incident: #477 round-1 (2026-06-05) — the original off-ramp was an eval
artifact; the recovery grid showed a different effect direction (more
negatives AMPLIFY the source implant, not suppress it). Leading the
clean-result with the bug-and-recovery finding made the contrast
unmistakable; the rest of the body rested on the recovered grid without
re-litigation.
