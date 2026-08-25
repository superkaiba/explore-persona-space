---
name: hf-stop-strings-pad-fill-and-hit-eos-alias
description: HF generate(stop_strings=...) pad-fills halted rows — strip trailing pad_id from gen_ids; when pad is in the eos set, stop-halted rows read hit_eos=True (alias); compute hit_stop from text and key cap-hit on "no terminal at all"
metadata:
  type: feedback
---

Wiring per-row textual stops through HF `model.generate(stop_strings=...,
tokenizer=...)` (StopStringCriteria; transformers 4.57) on a padded batch
(#2378 r18): a row halted by a stop string keeps DECODING SLOTS in the
output tensor — HF fills them with `pad_token_id`. Two traps:

1. A gen-ids extractor that truncates at the first EOS-set token leaves the
   pad fill on the tail whenever `pad_id` is NOT in the eos set — strip
   trailing `pad_id` tokens from `gen_ids` when stop_strings are active
   (worked impl: `experiments/issue2333/decode_hooks.generate_batch_ids`).
2. When `pad_id` IS in the eos set (Qwen ships pad=eos-family), the pad fill
   reads as EOS and a stop-halted row records `hit_eos=True` spuriously —
   an ALIAS, not a bug you can remove at the extractor.

**Why:** cap-hit telemetry keyed on `not hit_eos` alone inflates under stops
(a stop halt is an effective terminal); keyed on `hit_eos or hit_stop` it
stays direction-correct under BOTH pad regimes.

**How to apply:** compute `hit_stop` caller-side from the decoded text
(marker-in-text), record it per row, and define capped = reached cap with NO
terminal (`not hit_eos and not hit_stop`); treat `hit_eos` as unreliable for
stop-halted rows when pad∈eos. Add stop kwargs ADDITIVELY (default None) on
reused generators so sibling callers stay byte-identical.
