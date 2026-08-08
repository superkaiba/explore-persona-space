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
proves the mapping against the live artifact. Run (1)–(4) once per
(source-family × staged consumer) pair — a probe on one pair leaves a second
family's layout or a later reread phase unprobed — and probe through the SAME
staging helper production uses for that pair (a smoke staging per-file via
`stage_hub_file` to the consumer path validates nothing about a production
`stage_hub_prefix` mirror; #1481).

## Merged sibling index rows (#1891 curation, 2026-07-30)

This entry is the PRIMARY index pointer for its theme; the sibling index rows below were merged into one index row to fit the ~25 KB loader truncation limit (task #1891). Each merged row is preserved verbatim — follow its pointer for the sibling lesson's own entry file.

- [Hub-prefix-mirror staging ≠ consumer layout](feedback_hub_prefix_mirror_vs_consumer_layout.md) — producer uploaded manifest INSIDE the blob folder; pure hub-rel→local-rel mapping + fail-loud entry-file check + smoke against the REAL Hub path shapes (#928).
- [stage_hub_prefix verbatim mirror needs consumer rebind](feedback_stage_hub_prefix_verbatim_mirror_consumer_rebind.md) — mirror lands dest/<repo path>; rebind consumer root to dest/<prefix>; smoke must use the production staging helper (#1481)
- [stage_hub_prefix dest is a mirror ROOT](feedback_stage_hub_prefix_dest_is_mirror_root.md) — dest/<repo-relative path> layout; pass root satisfying root/<prefix>==consumed path + (h)(iv) consumer-open probe (#1774)
