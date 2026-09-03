---
name: issue2658-refreeze-mechanics
description: "#2658 deliberate re-freeze recipe: commit code FIRST (clean _git_sha stamp), run_build overwrites manifests, pins/evidence need rm-then-rerun (write-once); freeze_pins walls ~20+ min on load_math_full — detach it; unit10 pins-sha constant must be updated"
metadata:
  type: reference
---

Sequencing that works (#2658 D r3, 2026-09-02): (1) commit code+code-tests
(clean tree) -> (2) `issue2658_frames.py --build` overwrites both manifests
(no write-once refusal; `metadata.git_commit` stamps HEAD, now with a
`-dirty` suffix when tracked files are modified) -> (3) run the new
artifact-consistency tests RED against stale pins -> (4) pins re-freeze:
`freeze_pins` REFUSES on an existing mismatched table — the deliberate
override is `rm eval_results/issue_2658/prompt_pins.json` (save a copy
first) then `issue2658_text_resolver.py --freeze-pins`; (5) evidence:
same rm-then-rerun (`issue2658_evidence.py`, default mode freezes;
`_assert_core_match` refuses otherwise) — must run AFTER pins (store is
built from pins); (6) update `tests/test_issue2658_unit10.py`
`PIN_ITEMS_SHA256` (sha256 over `json.dumps(pins["items"], sort_keys=True,
separators=(",",":"))`) + its re-freeze note; (7) commit artifacts+unit10
together so no committed tree is red.

Walls: `--freeze-pins` resolves ALL pilot items incl. correctness loaders;
`load_math_full` alone runs ~10+ min — two 9.5-min foreground timeouts died
there (rc=143) before a detached setsid+pid/log run finished (~15 min
total). `--build` is ~4 min all-rows (sycophancy row alone ~45 s incl. the
13,204-stem edge-5 prefix-filtered pass). Evidence freeze: seconds.
`workflow_lint.py` no-flags under fleet load (load1 > 20) exceeds 585 s —
detach it too; treat a timeout as INCONCLUSIVE, not red.

Related: [[refreeze-moves-pilot-membership-downstream-frozen-artifacts]],
[[normalizer-form-vs-acceptance-instances]].
