---
name: public-corpus-release-is-toxicity-stripped
description: A moderation-LABELED public corpus's non-gated release may be toxicity-STRIPPED (the flag is uniformly False); count the flagged SUBSET via the filter-API before planning an invert-filter mine
metadata:
  type: feedback
---

An "invert the moderation filter to KEEP the flagged rows" mine assumes the
flagged subset is actually PRESENT in the repo you stream. For a corpus that
ships a **gated full** release and a **non-gated public** release, the public
release is routinely toxicity-STRIPPED — the moderation/toxic FIELD is still
there (so a field-resolution smoke passes) but its value is uniformly False,
so an invert-keep yields ZERO rows.

**Why:** #2221 plan v10's primary evil fix was to invert
`_keep_wildchat` on `allenai/WildChat-1M` (non-gated) to keep `toxic==True`
rows. The field resolves and the keep-fn is correct, but the non-gated repo
has **0** `toxic=true` rows across all 837,989 — the toxic conversations live
only in the gated `allenai/WildChat-1M-Full` the plan had rejected. The
`_keep_wildchat_toxic` code was faithful; the plan's data assumption (A1/A4)
was false. `phase_found_toxic`'s WildChat leg kept 0 → `_stream_stage`
per-source 0-kept fail-loud rc=134, surfaced by the tiny-real staging smoke
BEFORE the pod run.

**How to apply:** when a plan schedules an invert-filter / flagged-subset mine
on a moderation-labeled corpus, verify the flagged SUBSET is non-empty with a
COUNT before trusting it — the datasets-server filter-API is definitive and
free:
`https://datasets-server.huggingface.co/filter?dataset=<ID>&config=<C>&split=<S>&where=%22<field>%22%3Dtrue&length=1`
→ read `num_rows_total`. A `num_rows_total: 0` means the public release is
stripped and the mine is dead on the non-gated repo (the gated full release,
or a different corpus, is the only source). A field-resolution smoke on
synthetic flagged rows is necessary but NOT sufficient — it proves the keep-fn
reads the field, never that the real data carries the flag. This is the
data-ingestion sibling of [[feedback_real_corpus_streaming_filters_tiny_real_probe]]
(there the filter mis-shape rejects everything; here the source itself is
empty of the target class) — the tiny-real streaming probe with per-filter
reject counters catches BOTH, and the counters (`not_toxic: 3774` here) name
the cause instantly. Surface a dry primary source as a BLOCKER concern (the
Step 5c-ter dispatch gate reads `concerns.jsonl`), never a silent work-around;
a two-tier trainability DROP floor lets the downstream degrade gracefully to
"drop the family with a revised denominator."
