---
name: Hub-prefix-mirror staging is not the consumer's layout
description: A stager that mirrors an HF prefix verbatim to disk can displace the consumer's entry file (manifest.json uploaded INSIDE the blob folder); map hub-rel -> local-rel with a pure fn and smoke against the producer's REAL Hub path shapes (#928 crash-fix).
type: feedback
---

Staging an HF prefix verbatim to local disk is NOT automatically the layout the
consumer reads. **Why:** #928 att-20260704-120700 — the extractor uploaded
`manifest.json` INSIDE `.../store/percq_summaries/` (its `STORE_PREFIX` already
ends in `/percq_summaries`), so `stage_store` mirroring the `.../store` prefix
staged the manifest at `<store>/percq_summaries/manifest.json` while
`Store(store_dir)` reads `<store>/manifest.json` → FileNotFoundError at init,
~6 min GPU burn. The synthetic-fixture smoke wrote the LOCAL layout directly, so
the staging phase was never exercised end-to-end (the cross-phase data-contract
smoke class, #518).

**How to apply:** for any stage-from-Hub helper feeding a consumer with a fixed
local layout: (1) refactor the hub-rel → local-rel mapping into a PURE function
and thread it through the entry-time missing-check, the fetch destinations, and
the completeness check via ONE shared dict (structural consistency); (2)
fail-loud at stage time if the mapped set lacks the consumer's entry file
(manifest) — never let the stage "succeed" into a doomed consumer init; (3) the
regression test serves a REAL loadable artifact through the producer's REAL Hub
path shapes (monkeypatch `huggingface_hub.HfApi.list_repo_tree` +
`hf_hub_download`; function-local imports pick up module-attr patches) and
asserts the CONSUMER loads from the staged result; (4) a cheap real-Hub
confirm: pre-seed the big blobs as dummies at their mapped paths, run the real
stager at the pinned revision — it fetches only the KB-scale entry file and
proves the mapping against the live artifact.
