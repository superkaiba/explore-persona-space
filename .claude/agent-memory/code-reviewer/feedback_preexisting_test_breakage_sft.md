---
name: preexisting-test-breakage-sft
description: tests/test_issue545_train_components.py has 2 pre-existing failures unrelated to MarkerOnlyDataCollator changes — attribute via git-stash, don't blame a sft.py diff for them
metadata:
  type: project
---

When reviewing a diff to `src/explore_persona_space/train/sft.py`, two tests
in `tests/test_issue545_train_components.py` FAIL on a CLEAN tree (verified
2026-06-23 by `git stash` of the diff + re-run):

- `test_cpu_trainer_build_with_marker_collator_and_kl_aux` — `ImportError:
  cannot import name '_maybe_attach_kl_aux'` (symbol does not exist in
  sft.py; only `_maybe_attach_marker_band_stop` + `_maybe_wrap_recipient_eos_collator` do).
- `test_registry_overrides_match_train_lora_config` — asserts `{kl_aux_weight,
  optim, kl_aux_data_path, warmup_steps, max_steps, lr_scheduler_type}` are
  missing from `TrainLoraConfig`.

Both trace to a stale #545 KL-aux feature the test pins but `TrainLoraConfig`
/ sft.py never grew. They are NOT caused by collator / marker-loss diffs.

**Why:** the marker-collator default-loss change (2026-06-23) tripped these on
first run; without attribution one could wrongly FAIL the diff or wrongly
demand it fix them.

**How to apply:** when a sft.py diff shows test failures, `git stash push --
<changed files>` and re-run the failing tests on the clean tree before
attributing. These two are out of scope for any collator/marker change — note
them as pre-existing, don't block on them. (Separately,
`tests/test_issue651.py` fails COLLECTION on a missing
`explore_persona_space.analysis.svd_direction_constancy` module — also
pre-existing, breaks `pytest -k` runs that glob it; deselect explicitly.)
