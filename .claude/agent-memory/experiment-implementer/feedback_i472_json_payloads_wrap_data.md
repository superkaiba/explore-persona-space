---
name: i472 JSON payloads wrap actual data under nested keys
description: persona_bank.json + R_train.json + R_eval.json all wrap their data under payload['personas'] / payload['completions']; raw json.loads leaks schema_version into iteration
type: feedback
---

#472-line code publishes its on-disk JSON payloads as STRUCTURED dicts with
`schema_version`, metadata, content_hash, etc. at the top level, and the
actual data nested under a single key. Three sibling artifacts share this
shape:

- `persona_bank.json` → actual `{name: prompt}` map lives at `payload['personas']`.
- `R_train.json` → actual `{persona: {q: completion}}` map at `payload['completions']`.
- `R_eval.json` → same shape as R_train.

Raw `json.loads(path.read_text())` on any of these returns the WRAPPER
dict, whose first-iter key is `'schema_version'`. The first downstream
`for p in payload` or `next(iter(payload))` then iterates wrapper keys
(`schema_version, source_persona, n_base, n_new, n_total, personas,
content_hash, git_commit, generated_at, sonnet_model`) instead of the
actual persona names — and the next inner-dict lookup `inner[p]` crashes
with `KeyError: 'schema_version'`.

**Always go through the canonical loaders** the #472 module already
exposes:
- `contrastive_neg_geometry_472.persona_bank.load_persona_bank(path)`
  — validates `schema_version == 'i472_v1'`, returns `payload['personas']`.
- `contrastive_neg_geometry_472.r_generate.load_r_artifact(path)`
  — validates schema, returns `payload['completions']`.

**Why:** task #505 round-3 (2026-06-06) crashed within ~20s of nohup at
`panel_coverage.py:149` because `dispatch._load_persona_bank_and_r` read
all three files raw via `json.loads`. The crash trace pointed at
panel_coverage, misleading the orchestrator's brief into framing the bug
as a centroids-structured-dict misread — but the centroids loader was
already correct; the JSON-payload loaders one frame up were the bug.

**How to apply:** any new #505/#472-line code that loads
`persona_bank.json` / `R_*.json` from disk MUST use the canonical loader
helpers, NOT `json.loads`. Same lesson applies to the centroids `.pt`
bundle (see feedback_centroids_pt_structured_dict.md). For a new `*.pt`
or `*.json` artifact published by #472-line code: read the writer's
`torch.save({...})` or `json.dumps({...})` block first to confirm the
wrapper schema before writing the consumer.

## Index hooks moved from MEMORY.md (#1891 curation, 2026-07-30)

The always-loaded index was curated to fit the ~25 KB loader truncation limit (task #1891); the full pre-curation index hook(s) for this entry are preserved verbatim below.

- [i472 JSON payloads wrap data](feedback_i472_json_payloads_wrap_data.md) — persona_bank/R_train/R_eval wrap data under payload keys; use the canonical load_* helpers. #505.
