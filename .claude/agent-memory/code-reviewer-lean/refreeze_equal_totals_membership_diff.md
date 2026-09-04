---
name: refreeze-equal-totals-membership-diff
description: a gated re-freeze of a content-addressed selection artifact with UNCHANGED headline totals is proven real by diffing id sets between commit states, never by the totals
metadata:
  type: feedback
---

When a later in-round commit re-freezes a content-addressed selection/manifest
artifact (new filter added, e.g. a length gate) and every headline total is
IDENTICAL to the first freeze (same per-status cell counts, same n_items,
same n_requests), do not read that as a no-op or a copied file. Extract both
states (`git show <sha1>:<path>`, HEAD) and jq-diff the selected id sets per
split plus the order/requests shas. Expected honest signature: a small
replaced-id set (exclusions landing in surplus pools get count-preserving
replacements from beyond the old truncation point) and changed request shas.
Exclusions can also be membership-inert when the excluded item sat past the
truncation cut, so n_excluded may exceed the membership diff.

**Why:** #2658 r15 g1/g2 — the length-gate re-freeze kept dev/test totals
byte-identical to the first freeze. The 9/6-item membership diff + changed
requests_sha256 proved the re-selection ran; equal totals alone could hide a
stale artifact blessed by a copied header. Sibling of
[[untracked-twin-add-certification]] (committed vs live twin) and
[[fingerprint-resume-ids-not-content]] (ids vs content grain).

**How to apply:** any split-review where commit A freezes and commit B
re-freezes the same write-once JSON. Also verify the freeze path is genuinely
gated (write-once helper raises on drift, so B must delete or override) and
recompute one split's order keys + requests sha independently rather than
trusting the header.
