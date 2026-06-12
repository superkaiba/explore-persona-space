---
name: hf-hub-pinned-rev-404
description: hf_hub_download(revision=<sha>, filename=...) deterministically 404s with EntryNotFoundError when the (revision, path) pair doesn't exist together — usually a SHA copy-pasted from the wrong artifact. Code-class bounce.
metadata:
  type: feedback
---

A dispatcher hardcoding both an HF dataset revision SHA and a file path dies in seconds with `EntryNotFoundError: 404 ... resolve/<sha>/<path>` when the pair doesn't exist together — the SHA predates the file, post-dates a rename, or was copy-pasted from a different artifact's commit (adapter vs data repo).

**Why:** #477 v5 recovery diagnostic (2026-06-05) — `i477_reval_confirm.py:113` 404'd on `issue472_neg_geometry/persona_bank.json` @ `66d7db7a`.

**How to apply:** CODE-class, not infra — don't fix experimenter-side. Post `epm:failure v1 failure_class: code` with: the pinned SHA, the exact 404'd path, the script+line, and the recommendation that the implementer verify via `HfApi().list_repo_tree(repo_id, revision=<sha>, repo_type="dataset")` then re-pin or fix the path. Same HF listing/path family as [[feedback_snapshot_download_truncated_siblings]], but this one is a loud single-file 404, not a silent empty fetch.
