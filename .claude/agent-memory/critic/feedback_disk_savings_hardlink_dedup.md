---
name: Disk-savings plans need hardlink-dedup accounting
description: Per-dir du double-counts hardlinked content (uv .venvs share the uv cache — ~0.12G real vs ~10G apparent); joint du is the discriminator before accepting dominant-cost lever rankings (#596)
type: feedback
---

When a `kind: infra` plan claims a disk-savings headline or defers an alternative as "the dominant remaining lever", check whether the costs are REAL marginal bytes or `du`-apparent double counts before ranking levers.

**Why (#596, sparse-checkout worktrees):** 44/46 worktrees carried ~10.3G-apparent `.venv` dirs, making "shared venv" look dominant over sparse checkout — but joint `du -sm venvA venvB` showed the second venv adds only ~120M (uv hardlinks site-packages into `~/.cache/uv`). The real ranking was the reverse: git checkout bytes are real per-unit copies (~3.8G each); the venv lever is ~0.12G/worktree.

**How to apply:** (1) `stat -c %h` a few large files to detect hardlinks; (2) `du -sm dirA dirB` in ONE invocation (du dedups hardlinks within a run) and read the second dir's marginal size; (3) confirm same-device (`stat -c %d`). Checkout/copy costs are real per-unit; package-manager venvs/caches (uv, pnpm, nix) are usually hardlink-shared. A mislabeled lever ranking seeds wasted follow-ups but is usually a Concern, not REVISE, when the plan's chosen lever is still real.
