---
name: reconciler-revise-verification-rounds
description: Round after a binding reconciler REVISE — compose the bounded-verification shape with a fifth BODY DIFF envelope as the mechanical delta-scope + fence instrument; marker-kind-qualify stale-sweep tokens.
metadata:
  type: feedback
---

When the brief frames round r as verification of a BINDING reconciler
REVISE's minimal fix list (#2617 r3, 2026-08-27), compose the
bounded-verification shape on the [[prior-round-prompt-reuse]] base:

1. **RECONCILER-FIX VERIFICATION block** — one item per minimal-fix-list
   edit: the ledger's claimed landing (addressed events are CLAIMS) + a
   verification recipe; figure fixes get an explicit MULTIMODAL read
   instruction (read the pinned PNG + sidecar rendered-text), text fixes
   get the snapshot line number.
2. **Fifth envelope: BODY DIFF prior→current** (`diff -u` of the two
   /tmp body snapshots; `exit code: 1` with an "= files differ,
   expected" note in the intro prose — the envelope grammar wants a
   numeric rc and 1 is correct). The diff is the MECHANICAL instrument
   for both the delta sweep (any NEW violation in changed spans = fair
   game at full severity, all six quantity shapes) and the DO-NOT-TOUCH
   fence confirmation (reconciler-dismissed asks: a hunk outside the
   named landing spots + figure re-pins = fence breach). Declare it
   load-bearing under the item-2b UNAVAILABLE rule.
3. **PASS-by-reference licensing** for untouched lenses ("unchanged
   since round 2 (diff envelope); round-2 adjudication stands") — the
   brief's "bounded, not a fresh full review" instruction needs an
   explicit per-lens scoring form or Codex re-reviews everything.
4. **Deferred ids are binding do-not-re-raise**: reconciler-deferred
   NITs appear only as fences to confirm unchanged; the CONCERN grammar
   gains "verified-NOT-LANDED re-persists under its ORIGINAL id; new
   delta defects get NEW ids; never re-raise deferred ids."

**Why:** without the diff envelope the "delta scope" is prose Codex
must reconstruct by re-reading both bodies (it only has one); without
the fence block the reconciler's dismissals read as suppressing the
whole lens (the [[open-interp-ids-at-cr-gate]] pairing logic).

**How to apply / gotchas (all hit at #2617 r3):**
- Snapshot-convention hunk: a `cp body.md` snapshot INCLUDES
  frontmatter while a prior set-body-payload snapshot may not — declare
  the frontmatter hunk not-a-body-edit and git-verify the frontmatter
  actually unchanged before claiming so.
- Stale-sweep tokens must be MARKER-KIND-QUALIFIED and run on the
  envelope-STRIPPED prompt: bare `"v2 -->"` false-positives on
  SPEC-native `<!-- clean-result-v2 -->`, and the diff envelope
  legitimately quotes prior-round filenames + removed clauses.
- Before licensing local multimodal PNG reads, verify the figure-regen
  commit touched exactly the claimed files (`git show --stat <sha>`)
  AND every embedded PNG's working copy blob-matches its pin
  (`git hash-object` == `git rev-parse <sha>:<path>`); state both in
  the prompt so the #922 pinned-blob rule and the local-read advice
  cannot conflict.
