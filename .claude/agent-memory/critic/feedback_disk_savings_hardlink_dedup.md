---
name: Disk-savings plans need hardlink-dedup accounting
description: Per-dir `du -s` double-counts hardlinked content (uv .venvs share the uv cache); joint du is the discriminator before accepting any "dominant cost" claim in disk-infra plans
type: feedback
---

When a `kind: infra` plan claims a disk-savings headline (or defers an alternative as "the dominant remaining lever"), check whether the costs are REAL marginal bytes or `du`-apparent double counts before ranking levers.

**Why:** Task #596 (sparse-checkout worktrees): 44/46 worktrees carried ~10.3G-apparent `.venv` dirs, making "shared venv" look like the dominant lever over sparse checkout (10.3G vs 3.4G per worktree). Joint `du -sm venvA venvB` showed the second venv adds only ~120M — uv hardlinks site-packages into `~/.cache/uv` (same device; link count ~50 on big `.so` files). Real per-worktree cost ranking was the reverse: checkout bytes (git writes real copies, ~3.8G each) dominate; the venv lever is ~0.12G/worktree. The plan's own follow-up note had the misleading apparent-size framing.

**How to apply:** For any "X is the dominant per-unit cost" or "defer Y, it's the bigger lever" claim in disk/storage infra plans: (1) `stat -c %h` a few large files to detect hardlinks; (2) `du -sm dirA dirB` in ONE invocation (du dedups hardlinks within a single run) and read the second dir's marginal size; (3) confirm same-device (`stat -c %d`) for the suspected shared store. Checkout/copy-based costs (git worktree checkouts) are real per-unit; package-manager venvs/caches (uv, pnpm, nix) are usually hardlink-shared. Mislabeled lever ranking seeds wasted follow-up tasks but is usually a Concern, not a REVISE, when the plan's own chosen lever is still real.
