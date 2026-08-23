---
name: introduced-vs-reused-load-parent-blob-check
description: A brief claiming "this diff introduces a tokenizer/model/data load" is checked against the parent blob first — the load often pre-exists and the diff only reuses the handle
metadata:
  type: feedback
---

Before costing a claimed newly-introduced resource load (tokenizer, model,
corpus read) in a round diff, grep the PARENT revision's blob for the load
site: `git show <commit>^:<path> | grep -n "<loader>"`.

**Why:** #2329 round-4 brief flagged "this diff introduces a tokenizer load
into the Leg B build path" — but `tok = _load_tokenizer(key)` pre-existed at
the parent (it served `_segment`); the diff only passed the existing handle
into a new checker. Costing it as new would have manufactured a finding and
mis-stated the round-2 "loads no model" positive as changed.

**How to apply:** whenever quantifying "how many times does X load/run per
pipeline", separate (a) loads the diff ADDS (new `from_pretrained`/open
calls) from (b) pre-existing loads the diff's new callee reuses. Only (a) is
this round's cost; (b) belongs to the round that introduced it. Same
per-phase multiplication still applies to (a): count `_build_sides`-style
call sites across ALL phase entrypoints (`grep -n "_build_sides("`) — a
per-side/per-phase re-run is the real multiplier, not the single call site
in the helper.
