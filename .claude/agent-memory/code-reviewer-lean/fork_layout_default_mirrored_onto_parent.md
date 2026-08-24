---
name: fork-layout-default-mirrored-onto-parent
description: Two-sided consumers (parent committed artifacts + fork's restructured layout) default the PARENT path to the FORK's layout — probe each side's default against that side's OWN committed tree (#2329 Leg B g4)
metadata:
  type: feedback
---

When a driver consumes BOTH a parent round's committed artifacts and a fork's
restructured output layout via mirrored per-side path args, diff EACH side's
default against that side's OWN committed/produced tree — never assume the
mirror. #2329 q35_ladder_decay g4: `issue2329_decay.py` defaulted
`--q25-stats-json` to `.../persona_specificity_ladder/f_metrics/stats.json`
(the FORK's `f_metrics/` layout) while the parent's committed stats live one
level up (`.../persona_specificity_ladder/stats.json`) — deterministic
FileNotFoundError on the first phase, with a misleading sparse-checkout hint,
unfixed at round HEAD, no dispatcher override, and tests fixtured with tmp
paths so the default was never exercised.

**Why:** fork authors mechanically mirror their own new layout onto the parent
side; the plan itself usually names the correct parent path (here v8 R10 + the
manifest figure `source:` line), so the probe is one `find`/`ls` against the
main checkout plus a grep of the plan for the parent artifact path.

**How to apply:** on any diff adding per-side path args/defaults (`--q25-*` /
`--parent-*` vs `--q35-*` / fork dirs): (1) `ls`/`find` the parent's committed
tree at the default; (2) grep the plan/manifest for the registered parent path;
(3) check the round's dispatch script actually overrides the default (absence
means the default is live); (4) check later round commits didn't already fix
it before grading severity. Related: [[smoke_fixture_authored_with_consumer_keys]]
(tmp-path fixtures mask default-path defects).
