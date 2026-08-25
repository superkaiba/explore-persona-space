---
name: counted-drop-terminal-flag-telemetry-leak
description: assert→counted-drop conversions - sweep every telemetry consumer keyed on terminal FLAGS (hit_eos/hit_stop), not just NaN/va consumers; an all-False-terminals drop row reads as a cap hit
metadata:
  type: feedback
---

When a fail-loud assert is converted to a counted drop row, the drop row's
default fields (n_completion_tokens=0, hit_eos=False, hit_stop=False) satisfy
"reached cap without any terminal" predicates. Sweeping consumers that filter
`drop_reason is not None` is NOT enough — also grep sibling ANALYSIS modules
for fraction/telemetry helpers keyed on the terminal flags themselves
(#2378 r19: `issue2378_patch_analysis._cap_hit_fractions` counts a chat/plain
`opener_empty` drop row as a cap hit; feeds the pre-registered 2% re-gen
trigger — phantom hit can cross 2% in small families).

**Why:** the drop is partitioned correctly in the producer (live filter, va
guard, dropped Counter) yet still leaks through a duck-typed flag contract in
a DIFFERENT file the diff never touched. Related: [[consumer-flag-producer-never-writes]],
[[reused-module-internal-consumer-sweep]].

**How to apply:** on any counted-drop diff, grep the repo (analysis scripts
included) for reads of the flag fields the drop row hard-codes (hit_eos,
hit_stop, finish_reason, n_*_tokens==0 proxies) and check each consumer either
filters the new drop_reason or is story-style exact-match on a specific
drop_reason. Also confirm the drop happens AFTER seed-input fixing
(seed keyed on the FULL block's first cell) and that the block runs greedy
when batch recomposition could shift sampling RNG alignment.
