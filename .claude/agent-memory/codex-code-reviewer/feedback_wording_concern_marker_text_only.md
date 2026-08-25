---
name: wording-concern-fixes-may-be-marker-text-only
description: "A persisted wording/enumeration concern (e.g. blind-spot item framing) is often fixed in the impl MARKER text, not code — verify the payload stat before echoing a brief's file list, and mine the marker's (b)/(d) for self-disclosed tensions to hand Codex as named adjudications (#2378 r3)"
metadata:
  type: feedback
---

Two compose deltas from #2378 r3 (post-reconciler-BINDING-FAIL fix round,
composed per the #2332-r4 mixed-rulings shape):

1. **A wording concern's fix can live entirely in the implementation-marker
   text.** The brief listed `issue2378_capture.py` among the round's diff
   files, but `git show <payload> --stat` showed only fits/dispatch/gen —
   the `capture-ready-escape-enumeration-wording` CONCERN was fixed by
   rewording blind-spot item (6) in the MARKER, exactly as the reconciler
   prescribed. Echoing the brief's file list unverified would have primed
   Codex to demand a capture.py code change and false-NOT-ADDRESSED the
   item. **How to apply:** always `git show <sha> --stat` at compose time,
   state explicitly which closure items are marker-text-only, and flag the
   brief discrepancy in the return.

2. **Mine the marker's (b)/(d) sections for self-disclosed tensions that
   bear on a concern's closure, and hand them to Codex as NAMED tensions
   (severity yours).** #2378 r3: the `_rows_dir` manifest-reconciliation fix
   falls through to "the fresh HF mirror stage" on size-mismatch, while the
   implementer's own (b) discloses the canonical staging helper "skips
   present files" (declared pre-existing / out of scope) — i.e. the
   fall-through may be unable to repair the exact mismatch that triggered
   it. Per the #2332-r2 rule (surface textual tensions, never resolve them
   yourself), the compose named the tension in the review-emphasis section
   and tied it to the addressed-CONCERN's closure. **Why:** the implementer's
   own hedges are the cheapest source of high-yield review questions.

Related: [[revision-round compose recipe]] (the #2332-r4 mixed-rulings
shape this round reused: reconciler ruling inlined FIRST as binding contract,
prior twin verdict SECOND with tags stripped + rows blockquoted,
no-relitigate block quoting the refuted-as-blocking grounds verbatim).
