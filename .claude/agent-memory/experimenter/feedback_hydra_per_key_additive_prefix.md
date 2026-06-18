---
name: hydra-per-key-additive-prefix
description: The +group.key prefix is per-key (depends on whether that key is in the resolved struct); never bulk-add or bulk-remove + across a key group. Dry-run the compose before nohup.
metadata:
  type: feedback
---

`Could not override 'training.X' ... use +training.X` means THAT key is not in the resolved struct — sibling keys (`learning_rate` vs `max_steps` vs `epochs`) can have different in-struct status, so a bulk add/remove of `+` fixes one key and breaks another.

**Why:** #416 (2026-05-28) — attempt 1 failed on `training.learning_rate`; the hotfix (b9f68e80) bulk-removed `+` from three keys; attempt 2 failed on `training.max_steps` (which DID need the `+`). One full turn lost.

**How to apply:** recommend the implementer (1) list the actual default-struct keys from the failing Hydra group's YAML, (2) decide per-key: in-struct → `group.key=X`, not-in-struct → `+group.key=X`, (3) add a Hydra compose dry-run (`uv run python scripts/train.py --cfg job <overrides>`) before any nohup — catches this in <2s. Third member of the "config edge cases that surface only at training start" family with [[trl-conversational-format-in-format-dataset]] and [[epochs-negative-one-zero-steps]].
