---
name: figure-sidecar-can-outlive-png-bytes
description: A figure's .meta.json sidecar can be CORRECT while the committed PNG bytes come from a different, partial-data render — verify renders by READING the PNG, and re-check the Hub mirror after any data-completing pass
metadata:
  type: feedback
---

A `savefig_paper` sidecar and its PNG can disagree: on #1768 the committed
`operator_kv_read.meta.json` declared all 8 arms (10 series, sizes
8/8/7/7/7/2/8/8/8/8) while the PNG next to it drew only 3 — the PNG bytes
landed from the 3-cell moment of a racing partial render, the sidecar from the
8-cell render of the same second (identical `created` stamp). File SIZE was no
signal either: the degraded render (389,540 B) and a stale-but-complete one
(450,946 B) both looked plausible, and the corrected render came out at exactly
the stale one's size with a different sha.

**Why:** under the shared-repo-root stash/restore race (and any concurrent
writer), a figure directory is two independently-written files, so
provenance-by-sidecar is not provenance-by-pixel. The project figure-sanity
duty ("Read the rendered PNG") exists for blank/empty renders; this is its
subtler sibling — a render that is non-empty, self-consistent-looking, and
silently missing most of its units.

**How to apply:** (1) Before presenting or committing any figure, READ the PNG
and count the units against the data set on disk — never infer completeness
from the sidecar, the file size, or the driver's completion line. (2) When
re-rendering to fix one, generate off-root (`--figs-dir` into
`/mnt/eps-data/$USER/...`), let the schema + grid-coverage gates fire, read the
staged PNG, then copy in and commit in a tight window. (3) A regenerated
results JSON that differs from its committed bytes ONLY in provenance stamps
(`git_commit`, `ts`) should be RESTORED to the committed bytes (write
`git show HEAD:<path>` to the file — a `git checkout -- <path>` at the shared
root is hook-blocked), so the commit carries the figure alone. (4) After any
pass that COMPLETES data behind an already-uploaded figure, re-check the Hub
mirror by sha: #1768's data-repo copy was an 8-arm render from before the
r_B-on-all-24-cells pass, so its bars were stale even though the count looked
right.

Related: [[feedback_must_fix_done_claims_verified_on_disk]] (verify on disk, not
from a claim), [[feedback_edit_then_read_modify_write_lost_update]] (the same
shared-root write race in the code path).
