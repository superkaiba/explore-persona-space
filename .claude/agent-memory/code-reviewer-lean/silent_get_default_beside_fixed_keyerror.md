---
name: silent-get-default-beside-fixed-keyerror
description: A round that fixes a loud wrong-key read at an artifact boundary often leaves a SIBLING `.get(<field>) or <default>` on the same artifact reading a field the producer never writes — silent, smoke-green, defeats a plan-registered control; audit EVERY field read from that artifact against the producer's writer (#2569 r? gB)
metadata:
  type: feedback
---

Rule: when a round's fix commit corrects a wrong-key read at a producer/consumer
boundary (the #2569 `rec["ci"]`→`rec["i"]` manifest-join KeyError), sweep EVERY
other field the same consumer file reads from that artifact against the
producer's actual writer — especially `.get(<field>)`/`or <default>` reads,
which are the SILENT version of the exact bug just fixed loud.

**Why:** #2569 shard B: `_corpus_tags_from_manifest_dir` was fixed (loud
KeyError on `rec["ci"]`, decoy test added), but the sibling join two functions
away — `_ans_len_from_manifest_dir` reading `rec.get("response") or ""` — kept
reading a field the n1m sampling manifest NEVER carries (producer
`issue779_ffc_n1m_generate_capture._write_manifest_parts` writes only
`{prompt, corpus, stream_pos, i}`; the manifest is written pre-generation).
Every row joined to length 0 (treated as KNOWN), decile edges collapsed to
zeros, every pair landed in one decile bucket — the plan-registered
answer-length stratification went structurally inert, the audit stat
(`n_rows_unknown_len`) still looked healthy, and the full-pipeline smoke ran
rc=0 through it. Both committed tests authored `"response"` into manifest
fixtures ([[smoke-fixture-authored-with-consumer-keys]] again), and the test
docstring even enumerated a real schema WITHOUT `response` while the fixtures
contradicted it.

**How to apply:** on any round carrying a wrong-key fix commit, grep the fixed
consumer file for every OTHER `rec[...]`/`rec.get(...)`/`row[...]` read of the
same artifact and verify each field against the producer's writer expression
(never against the round's fixtures or docstrings). Treat `.get(f) or default`
at a cross-artifact boundary as a presumptive finding until the producer is
shown to write `f`; demand a loud guard (direct index, or a joined-values
sanity assert) plus a producer-shaped fixture. A test-docstring schema listing
that contradicts the same test's fixture rows is itself a tell. Related:
[[consumer-flag-producer-never-writes]], [[fingerprint-resume-ids-not-content]].
