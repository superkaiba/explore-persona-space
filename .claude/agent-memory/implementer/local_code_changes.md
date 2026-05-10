---
name: Code Changes on Local VM Only
description: Never edit code directly on pods; always local → commit → push → pod git pull
type: feedback
---

All code edits happen on the local VM (working directory). Never edit source files directly on pods.

**Why:** Direct pod edits create sync conflicts — the pod diverges from origin/main, the next experimenter runs stale code thinking it has the new fix, debugging becomes impossible. User has been burned by this and escalated to a rule.

**How to apply:**
1. Edit files in `/home/thomasjiralerspong/explore-persona-space/` (local VM).
2. Run tests locally: `uv run pytest <relevant tests>`.
3. Lint: `uv run ruff check . && uv run ruff format .`.
4. Commit to a feature branch (never direct to main without user approval).
5. Push to GitHub.
6. On the pod (if the change needs to land there): `ssh_execute(pod, "cd /workspace/explore-persona-space && git pull --ff-only origin main")`.
7. If a pod needs the change as part of an experiment, the experimenter does the `git pull` — not you.

**Anti-pattern to avoid:** SSHing into a pod and editing a file with nano/vim because "it's faster". It is never faster — it creates an untracked mutation and nothing downstream knows about it.
