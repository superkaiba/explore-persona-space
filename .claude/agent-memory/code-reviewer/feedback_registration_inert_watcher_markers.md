---
name: registration-inert-watcher-markers
description: When a watcher/automation arm POSTS markers of a kind some parser READS (owner-fence tokens on epm:progress, follow-up signals, anti-liveness), verify the composed note against the parser's actual token grammar — the self-defeat class.
metadata:
  type: feedback
---

When a diff makes an automation arm POST markers whose KIND is also a
REGISTRATION/SIGNAL kind some parser reads (e.g. `epm:progress` is in
`pod_lifecycle._OWNER_REGISTRATION_KINDS`, so a `pod=<name>` token plus a
`fence_until=` token in a watcher's own note would REGISTER/CLEAR the very
fence it defers on), verify the composed note text mechanically against the
parser's grammar — `_note_names_pod` / `_POD_TOKEN_RE` / leading-token match —
not just against prose claims of inertness. Demand a test that appends the
REAL posted note to a live events list and re-runs the REAL parser across
ticks (#2283 `test_fence_defer_marker_is_registration_inert` is the model
shape). Same family as `_WATCHER_NOTE_SENTINELS` anti-liveness (#2084).

**Why:** #2283 deviation 1 — the plan's own sketched marker text (structured
`pod=` token + quoted `fence_until=none` release recipe) would have cleared
the owner's fence on the first defer tick; the implementer caught it and
composed the note registration-inert (sentinel-leading, `pod=<pod-name>`
placeholder whose `<` falls outside the token regex's value class, real pod
name mid-prose only). Review verified inertness by reading the regexes and
running the parser on the composed text.

**How to apply:** any diff where an arm both PARSES notes for tokens and
POSTS notes of a parseable kind: enumerate every token grammar live on that
kind, run each composed note through the real matcher, and check quoted
remedy/recipe text (commands the note tells an operator to run) uses
placeholders that cannot match the grammar.
