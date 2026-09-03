---
title: hub._upload returns empty instead of raising on missing token/path, defeating
  raise_on_error=True
kind: infra
tags: []
created_at: '2026-09-03T07:49:55Z'
has_clean_result: false
workflow: v1
---
---
kind: infra
---

# `hub._upload` returns "" instead of raising on a missing token / missing path, defeating `raise_on_error=True`

## Goal

Make `raise_on_error=True` mean what it says in
`src/explore_persona_space/orchestrate/hub.py`: an upload that did not happen must
RAISE, never return a falsy path that a caller can mistake for success. Two
early-return branches currently bypass the flag entirely.

## The defect

Measured live during issue #2658 phase P1 upload remediation (2026-09-03).

`_upload(...)` at `orchestrate/hub.py:1859-1862`:

```python
token = os.environ.get("HF_TOKEN")
if not token:
    logger.warning("HF_TOKEN not set, skipping upload")
    return ""
```

The bulk sibling repeats it verbatim at `orchestrate/hub.py:2110-2113`. Both sit
BEFORE `raise_on_error` is consulted, so a caller passing `raise_on_error=True`
gets a warning on stderr and an empty string — not an exception.

The adjacent branch is the same class and arguably worse, because nothing
external hints at it:

- `orchestrate/hub.py:1864-1866` — a non-existent `local_path` logs
  "Path %s does not exist, skipping upload" and returns `""`. A typo'd or
  mis-rooted source path is therefore indistinguishable from a successful
  upload for any caller that does not inspect the return value.
- `orchestrate/hub.py:2115-2117` — the bulk twin, for a non-directory.

## Why this is more than cosmetic

The upload-before-delete invariant is what protects trained artifacts: callers
upload, then reap local weights. A caller that trusts `raise_on_error=True` and
does not separately assert an expected file set will proceed to delete local
data after an upload that never occurred. That is the #404/#458 loss shape
(36 checkpoints deleted after a soft-failed upload) reachable through a flag
whose name promises the opposite.

The observed instance was benign only because the call site had its own
`verify_repo_paths_uploaded` expected-set assert immediately after, which caught
it. That is defense in depth doing the helper's job.

## Evidence this is a known-and-partially-fixed pattern

`orchestrate/hub.py:1868` already carries a comment fixing ONE silent-no-op
class in the same function:

> Fail loud on the silent-no-op class (#595, 2026-06-13): a FILE handed to the
> folder branch (upload_as_file=False) makes huggingface_hub.upload_folder log
> "... is not a directory. Keeping local path." and upload NOTHING, yet
> verification can still pass if same-prefix files already exist

So the shape was recognized and remediated for the file-vs-folder case while
the token and missing-path siblings were left returning `""`. The placement
rationale recorded there — "before HfApi, outside the try, so the raise
propagates instead of returning ''" — is exactly the fix these branches need.

## Scope of the change

1. Under `raise_on_error=True`, both the missing-token and missing-path
   branches raise a typed error naming which precondition failed. The default
   (`raise_on_error=False`) keeps returning `""` so existing fail-soft callers
   are unchanged — this must not become a fleet-wide behavior flip.
2. Audit every `_upload` / bulk-path caller for the "trusts the flag, does not
   assert an expected set" pattern, and report which ones would have silently
   lost data. `scripts/issue2658_capture.py:617 upload_store` is a good example
   of the CORRECT shape (upload, then `verify_repo_paths_uploaded` against an
   exact expected list, assert empty) and can serve as the reference.
3. A test per branch: `raise_on_error=True` with `HF_TOKEN` absent raises;
   with a non-existent path raises; with the flag false both still return `""`.

## Non-goals

Do not widen this into a general refactor of `hub.py` upload routing, and do not
touch the retry / overflow / file-count-fallback machinery. The change is
confined to making two (four counting the bulk twins) early returns honor a flag
they currently ignore.

## Provenance

Found while remediating the issue #2658 P1 upload-verification FAIL: a pod-side
`_upload(..., raise_on_error=True)` for 179 residue manifest files printed
"HF_TOKEN not set, skipping upload" and returned `""` without raising, because
the pod-side heredoc had not loaded `.env`. The upload silently did nothing; only
the call site's own expected-set assert made it loud.
