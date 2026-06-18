---
name: add-dimension-alias-legacy
description: When adding a new dimension (probe position, judge variant, bucketing) to an established multi-cell pipeline with on-disk artifacts and downstream consumers, keep ALL legacy filenames + payload keys as aliases of the now-default value and add NEW parallel keys/files. Never rename.
metadata:
  type: feedback
---

When adding a new dimension to an established multi-cell pipeline that has on-disk artifacts (per-cell JSONs, per-seed payloads) and downstream consumers (aggregators, analyzers, critics):

- Keep legacy filenames + payload keys untouched; they ALIAS the existing (now-default) dimension value (e.g. `trained_logp_by_cell` aliases `trained_logp_by_cell_first_token`).
- Add NEW parallel filenames/keys for new dimension values (`logprob_trained_oncontent_A.json`, `trained_logp_by_cell_oncontent`).
- Parameterize pool helpers with a `position`/`variant` arg defaulting to the legacy value; the orchestrator emits both alias and explicit keys.

**Why:** resume-from-disk skips re-running 15-45 min computations; existing HF artifacts need no re-upload; analyzer/critic code reading legacy keys keeps working; the new dimension's verdict ships as a parallel block instead of silently changing what an existing key means. Anti-pattern — renaming and "updating all callers in one pass" — always misses a caller or invalidates every existing per-cell file.

**Example:** task #399 round-16 (2026-05-27) added an on-policy end-of-content probe beside the first-token probe this way; resume on 84 existing first-token cell files worked with zero re-compute and zero analyzer changes. Cost ~50-100 lines of plumbing.
