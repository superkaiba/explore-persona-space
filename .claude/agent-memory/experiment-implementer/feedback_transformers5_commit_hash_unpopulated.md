---
name: transformers5-commit-hash-unpopulated
description: transformers 5.15.0 leaves config._commit_hash None — pin-engagement probes must use cached_file snapshot-path resolution, private attr PASS-only
metadata:
  type: feedback
---

transformers 5.15.0 no longer populates the PRIVATE `config._commit_hash`
attribute on `AutoConfig`/model configs (4.57.6 did), so any revision-pin
engagement assert keyed on `_commit_hash == <pin>` false-fails on a correctly
pinned load.

**Why:** #2329 bank phase crashed rc=1 at `issue2329_ladder.py:818` on a load
whose HTTP trace showed every artifact resolving at `snapshots/<pin>` — the pin
WAS engaged; the probe was wrong. The pod's gate0b installed transformers
5.15.0 over the repo's 4.57.6.

**How to apply:** prove pin engagement via the PUBLIC API —
`transformers.utils.hub.cached_file(model_id, "config.json", revision=<pin>,
local_files_only=True)` then assert `f"snapshots/{pin}" in str(resolved)`
(same technique tokenizer legs must always use, since tokenizers never stored
`_commit_hash`). A private attr like `_commit_hash` may survive only as an
opportunistic fast path that PASSES the check — never as grounds to FAIL when
it is None/absent/stale. Worked fix + regression tests:
`scripts/issue2329_ladder.py::_assert_pin_engaged` +
`tests/test_issue2329_pin_engaged.py` (branch issue-2329-q35-ladder-decay,
commit 7caaecf958269778655a0831a8c1d17ce83468a8).
