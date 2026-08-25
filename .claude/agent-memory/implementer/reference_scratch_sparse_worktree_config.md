---
name: scratch-sparse-worktree-config
description: git -C <worktree> config WITHOUT --worktree writes the SHARED .git/config — a scratch sparse-checkout setup can silently flip core.sparseCheckout for the main checkout and every worktree lacking its own config.worktree
metadata:
  type: reference
---

Recipe for a same-session pre-change baseline tree (a sparse scratch worktree
detached at BASE_SHA, ~3.7 GB for this repo's sparse cone):

```bash
git -C "$WT" worktree add --no-checkout --detach /tmp/<name> <BASE_SHA>
SPARSE=$(git -C "$WT" rev-parse --git-path info/sparse-checkout)   # copy source
SGD=$(git -C /tmp/<name> rev-parse --git-path info/sparse-checkout)
cp "$SPARSE" "$SGD"
git -C /tmp/<name> config --worktree core.sparseCheckout true      # --worktree!
git -C /tmp/<name> checkout --detach <BASE_SHA>
# run baseline with the ISSUE worktree's venv python (same interpreter both runs)
git -C "$WT" worktree remove --force /tmp/<name>                   # cd out first
```

**The trap (hit on #2537, 2026-08-24):** `git -C <worktree> config core.X v`
WITHOUT `--worktree` writes the SHARED `.git/config` — on this repo that
flipped `core.sparseCheckout=true` for the MAIN checkout and every worktree
lacking its own `config.worktree` (e.g. full worktrees like
`vectorized-mlp-skill`). Diagnose with `git config --show-origin
core.sparseCheckout` (origin `file:.git/config` = polluted; per-worktree
configs show `.git/worktrees/<n>/config.worktree`); repair with
`git config --unset core.sparseCheckout` at the root + re-set with
`--worktree` in the scratch. `extensions.worktreeConfig=true` is already on
in this repo (new_worktree.sh uses per-worktree configs), so `--worktree`
always works.

**How to apply:** any per-worktree git setting (sparseCheckout, sparse cone
mode) on a hand-built scratch worktree takes `config --worktree`, never bare
`config`. Verify with `--show-origin` after setting.
