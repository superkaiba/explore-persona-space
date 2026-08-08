---
name: stratifier-absent-from-uploaded-rows
description: A sampling stratifier added AFTER a generation run does not exist in that run's uploaded rows — the fallback reads one stratum for everything and the draw silently stops being stratified; recover it by joining the raw-generation companion, never assume the field is there
metadata:
  type: feedback
---

Before sampling on a row FIELD (`capped`, `finish_reason`, a quality flag),
CHECK THE UPLOADED ROWS FOR IT — do not infer its presence from the generator
source. A field threaded into the generator AFTER a run completed does not exist
in that run's artifacts, and the usual `field in row else fallback` shape then
reads a single stratum for every row: the stratified draw degenerates to simple
random, and the per-stratum sub-means report one all-inclusive block. No error,
no crash, a wrong answer.

**Why:** #1345 (2026-07-31). The `capped` cap-split landed after job 16257 had
generated the 4 on-policy cells, so every uploaded row carried
`[conv_id, model, prompt, provenance, response, shape]` and nothing else.
`capped_of()` fell back to a missing `finish_reason`, all 4,618 rows read
natural, and the design would have reported a single all-natural sub-mean while
claiming to be stratified. The stratification was load-bearing: in every drift
cell with a real cap rate, capped answers scored 6-10 points LOWER than natural
ones, so a cap-composition shift moves the pooled mean for a reason unrelated to
the construct.

**How to apply:**
- Digest the ACTUAL uploaded rows first: `jq 'keys'` on one row per cell (field
  names only — never content text for corpus-derived data). Compare against the
  fields the design consumes.
- When the field is missing, look for a COMPANION artifact that carries it
  before regenerating: the raw-generation file (`raw_*.jsonl`) usually keeps
  `finish_reason` even when the kept-rows file does not. Join on the row id.
- Validate the recovery against an independently-measured count. The #1345 join
  reproduced the reported 2478 / 591 / 1382 / 6 exactly, which is what made it
  trustworthy.
- Make the join fail loud: the raw pool is a SUPERSET of kept rows, so ANY kept
  row without a raw match means the wrong companion file — assert zero misses
  rather than silently thinning the cell.
- Record the recovery path + both shas in the prep report, so the stratifier's
  provenance is auditable rather than implicit.

Sibling gotcha in the same round: the character cells' on-policy rows carried no
`answer` field at all (their injected siblings did) — only `story` +
`parsed_turns` spans. Same lesson, different field: two provenances of "the same"
artifact class can have DIFFERENT schemas, so digest each class you consume.

Related: [[feedback_capture_convention_read_producer_code]],
[[feedback_verbatim_embed_reject_taxonomy_before_budget_fix]].
