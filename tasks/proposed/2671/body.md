---
title: 'Shard-manifest schema mismatch: #1902 upload_text_payload writes {shards:[...]},
  shared hub.stage_sharded_text only reads {parts:[...]}'
kind: infra
tags: []
created_at: '2026-09-09T16:28:01Z'
has_clean_result: false
origin_prompt: 'Found 2026-09-09 while staging capture inputs for the #1902 full-grid
  K=5 round on a fresh pod.'
workflow: v1
---
# Shard-manifest schema mismatch: #1902 sharded text uploads are unreadable by the shared stager

## Goal

Make every sharded text artifact written by `upload_text_payload` readable by the shared `hub.stage_sharded_text`, so a consumer on a fresh pod or a sibling issue can restage #1902's rollout files (and any future issue's) through the common path instead of hand-rolling a reader.

## What is wrong

Two producers in the repo write a `<stem>.manifest.json` with **different schemas**, and the shared stager understands only one of them.

Producer A, the #2054 shared path (`_shard_large_jsonl_for_upload`):

```json
{"parts": ["<name>", ...], "sha256": {"<name>": "<hex>"}}
```

Producer B, `scripts/issue1902_run.py:571` (`upload_text_payload`):

```json
{"source": "B_k5_seed45.jsonl", "n_shards": 2,
 "shards": [{"name": "...", "n_lines": 13573, "sha256": "..."}]}
```

`hub._parse_shard_manifest` (`src/explore_persona_space/orchestrate/hub.py:3140-3153`) reads `man.get("parts")` and raises `RuntimeError: shard manifest lists no parts` when it is absent. So `hub.stage_sharded_text` fails loud on **every** file producer B uploaded.

Reproduction (2026-09-09, pod-1902-k5grid):

```
RuntimeError: shard manifest lists no parts: superkaiba1/explore-persona-space-data:
  issue1902_stage_map/raw_completions/gen/single/B_k5_seed45.manifest.json
```

Blast radius today: all 16 K=5 rollout files plus the original seed-42 rollout files under `issue1902_stage_map/raw_completions/gen/`. The failure is fail-loud, so nothing silently corrupts. `scripts/issue1902_k5_fits.py:228-245` already carries a private reader for producer B's schema, which is the duplication this task removes.

## Fix

Teach `_parse_shard_manifest` both schemas: keep `parts` + `sha256`, and accept `shards: [{name, sha256, n_lines}]` by projecting it onto the same `(parts, sha256_by_name)` return. Prefer the richer form's `n_lines` as an extra post-concat check where present. Then delete the private reader in `issue1902_k5_fits.py` and route it through `hub.stage_sharded_text`.

Do NOT change producer B's on-Hub schema: files already uploaded under it must stay readable, so the reader is the place to converge.

Acceptance:
- `hub.stage_sharded_text` restages `issue1902_stage_map/raw_completions/gen/single/B_k5_seed45.jsonl` to 16,391 rows with every shard sha256-verified.
- A unit test per schema, plus one asserting a manifest with neither key still raises.
- `issue1902_k5_fits.py` no longer parses manifests itself.

## Provenance

Found while staging capture inputs for the #1902 full-grid K=5 round on a fresh pod (that round worked around it with a throwaway pod-side reader mirroring `issue1902_k5_fits._rollout_records`; the workaround is not committed). Not auto-dispatched: this VM has automated Claude sessions disabled, so the task is filed at `proposed` for manual pickup.
