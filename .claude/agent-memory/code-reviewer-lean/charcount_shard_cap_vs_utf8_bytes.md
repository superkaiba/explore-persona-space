---
name: charcount-shard-cap-vs-utf8-bytes
description: a JSONL line-split helper capping shards by len(str) chars with ensure_ascii=False under-counts UTF-8 bytes — CJK-heavy rows push a "9 MB" shard past the 10 MB LFS force-route; check every NEW caller's payload class (#2552 r2 g3)
metadata:
  type: feedback
---

A shard-splitting helper that enforces the upload-policy "<9 MB" rule via
`len(json.dumps(row, ensure_ascii=False))` measures CHARS, not bytes: non-ASCII
text (real LMSYS/WildChat corpora, judge rationales) is 2–3 bytes/char in
UTF-8, so a char-capped shard can land 10–27 MB on disk and force-route to LFS
(the quota-gated class). The fix is one line: size on `len(p.encode("utf-8"))`.

**Why:** #2552 r2 g3 — `_jsonl_write_sharded` (cap_bytes=9_000_000 on char
counts) pre-existed the round with numeric-payload users, but the round added a
judge raw-draw serializer routing REAL multilingual completion text through it.
The helper's risk is per-CALLER: numeric `[id, val]` lists ⇒ chars≈bytes, safe;
free-text rows ⇒ reachable overflow.

**How to apply:** on any diff that adds a caller to a size-capped text-split
helper (or adds such a helper), read the cap's unit (chars vs bytes) and
classify the new payload (numeric/ASCII vs corpus text). Char-capped + corpus
text = Minor at least; severity rises if the destination is quota-gated.
Sharpens [[size_match_resume_skip_npz]]'s sibling lesson that "size" words in
upload paths rarely mean what they say.
