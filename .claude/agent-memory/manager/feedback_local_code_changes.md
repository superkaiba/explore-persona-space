---
name: Code changes on local VM only
description: Never edit code directly on pods; always edit locally, commit, push, then git pull on pods
type: feedback
---

All code changes must be made on the local VM, never directly on pods.

**Why:** Editing code on pods creates sync conflicts when pulling from git. Changes get lost or overwritten, and it's hard to track what was modified. This happened during Phase A2 launch when git pull failed due to conflicting untracked files.

**How to apply:** Edit files locally → `git commit` → `git push` → `ssh pod "git pull"`. For quick patches (like the vLLM tqdm fix), use `sed` on the pod only for third-party library files that aren't in git — never for project code.
