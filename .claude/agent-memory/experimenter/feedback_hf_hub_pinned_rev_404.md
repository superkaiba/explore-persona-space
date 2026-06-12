---
name: hf-hub-pinned-rev-404
description: A `hf_hub_download(..., revision=<sha>, filename=...)` call against a pinned dataset commit can deterministically 404 with EntryNotFoundError when the file path does not exist at that pinned revision. Dispatcher fails seconds after launch with a clean Python traceback. Code-class bounce — implementer must verify the (revision, path) pair via `list_repo_tree(repo_id, revision=...)` before pinning. Burned at #477 v5 recovery diagnostic 2026-06-05.
metadata:
  type: feedback
---

# `hf_hub_download` 404 EntryNotFoundError on pinned revision

## Rule

When a dispatcher hardcodes a HuggingFace dataset revision SHA AND a
file path inside it, the (revision, path) pair MUST exist together.
Otherwise the launch dies in seconds with a traceback like:

```
huggingface_hub.errors.EntryNotFoundError: 404 Client Error. ...
Entry Not Found for url:
  https://huggingface.co/datasets/<org>/<repo>/resolve/<sha>/<path>
```

**Why:** the pinned `<sha>` may predate the file being added, post-date
its rename/removal, or live on a different branch entirely. The
implementer probably copy-pasted the SHA from a different artifact
(e.g. the *adapter* commit rather than the *data* commit), or the file
was reorganized after the SHA was captured.

**How to apply:** This is a CODE-CLASS failure, not infra — do NOT try
to "fix" it from the experimenter side. Post `epm:failure v1` with
`failure_class: code`, include the four diagnostic facts:
- the pinned revision SHA
- the exact file path that 404'd
- the script + line (`File "...", line N, in _ensure_data`)
- one-line recommendation: implementer verifies via
  `huggingface_hub.HfApi().list_repo_tree(repo_id=..., revision=<sha>,
  repo_type="dataset")` and either repins to a revision containing the
  file OR fixes the path.

Same family as [[snapshot-download-truncated-siblings]] (also an HF
Hub path/listing mismatch silently breaking a fetch), but distinct:
this one is a deterministic 404 on a single-file fetch, not a silent
empty allowlist.

## Incident

**Task #477, v5 recovery diagnostic (2026-06-05).** Launched
`scripts/i477_reval_confirm.py --max-new-tokens 1024` on pod-477.
Process died within 8s with:

```
File "/workspace/explore-persona-space/scripts/i477_reval_confirm.py",
  line 113, in _ensure_data
    cached = hf_hub_download(...)
...
EntryNotFoundError: 404 ...
  /datasets/superkaiba1/explore-persona-space-data/resolve/
  66d7db7a542e19275f8c1d8e32948396d050faa9/
  issue472_neg_geometry/persona_bank.json
```

HEAD `16bd9c28` (the confirmation script commit). The pinned data
revision `66d7db7a...` either doesn't contain
`issue472_neg_geometry/persona_bank.json` or the path was moved.
Posted `epm:failure v1` `failure_class: code` and exited.
