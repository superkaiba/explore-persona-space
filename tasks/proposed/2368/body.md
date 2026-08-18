---
title: 'test_bootstrap_pod_git_credentials.py extracts ssh payloads by first-quote
  scan (same shape #2360 r3 hardened); sweep tests/ for siblings'
kind: infra
tags: []
created_at: '2026-08-18T11:21:07Z'
has_clean_result: false
parent_id: 2360
workflow: v1
---
---
kind: infra
---

# `test_bootstrap_pod_git_credentials.py` extracts ssh payloads by first-quote scan — the same shape #2360 round 3 hardened

**Provenance:** surfaced in the #2360 round-3 implementer report prose after the
Codex reviewer required the same hardening for a sibling payload family. Not
speculative — the pattern is confirmed present, and #2360 has a landed reference
fix for it.

## The pattern

`tests/test_bootstrap_pod_git_credentials.py:75-86` walks `bootstrap_pod.sh`
for `ssh_cmd '` markers and closes each block at the FIRST following quote:

```python
marker = "ssh_cmd '"
...
open_q = start + len(marker)
close_q = text.index("'", open_q)
blocks.append(text[open_q:close_q])
```

A remote payload that legitimately contains a quote after the marker is
therefore truncated, and every assertion runs against an incomplete block. The
test can stay green while the part of the payload it never saw regresses — the
failure direction is silent under-inspection, which is the worst direction for a
guard.

## Why this is worth a task rather than a note

#2360 hit exactly this in its own new binder and had it flagged by the Codex
reviewer as "certifies less than their names claim under future payload drift".
The landed fix there is the reference implementation:

- slice to the exact standalone closing-delimiter line rather than the first
  quote;
- assert the entire extracted interior contains no quote (so truncation cannot
  pass silently);
- and — the load-bearing part — add MUTATION tests that insert a quote after the
  guarded region and require the extractor to FAIL. A guard that cannot be made
  to fail by the mutation it exists to catch is not a guard.

See `tests/test_bootstrap_pod_uv_link_mode.py` on the `issue-2360` branch
(`_step10_payload` + `test_mutation_quote_after_invocation_guard_fails_extraction`).

This is the same defect family as #2360's own blockers B1 and B5 — a check whose
PASS certifies less than it appears to — now confirmed at a third depth. That
recurrence is the argument for fixing the sibling rather than leaving it.

## Scope

- Harden the extractor in `tests/test_bootstrap_pod_git_credentials.py` per the
  three bullets above, with the mutation test.
- Sweep `tests/` for any other first-quote / `text.index("'")` payload
  extraction and either harden it the same way or record why it is safe.
- Purely test-side: no production behavior changes, so this cannot regress the
  fleet. `kind: infra`.

## Acceptance

- A quote inserted after the guarded region makes the extractor FAIL rather than
  silently truncate (mutation test, required to go red).
- The extracted interior is asserted quote-free.
- Every remaining first-quote payload extractor under `tests/` is either
  hardened or has a recorded reason it is safe.
