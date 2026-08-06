---
title: 'daily-held: perform the #681 worktree bind cutover'
kind: infra
tags:
- daily-held
- needs-human
created_at: '2026-08-06T07:09:11Z'
has_clean_result: false
origin_prompt: '/daily 2026-08-05 problem sweep (route 3): 212GB of worktrees on the
  97%-full root disk; bind never activated; migration breaks live sessions so timing
  is a user call'
workflow: v1
---
# daily-held: perform (or schedule) the #681 worktree bind cutover — 212 GB of worktrees still on the 97%-full root disk

## Held item

The #681 design (heavy worktree footprint on the 512 GB data disk, bind-mounted onto
`.claude/worktrees`) was never activated: probed 2026-08-06 —
`findmnt .claude/worktrees` → NOT a mount point; `df -h` → `/` 945G at 97% (32G free) vs
`/mnt/eps-data` 1007G at 23% (782G free after the overnight cleanup). ~212 GB of
`.claude/worktrees` sits on the root disk, which spent 2026-08-05/06 oscillating
19.9 → 13.1 GiB free with the watcher warning that foreground Bash spawns fail silently
(#552 class) — while the data disk has ~782 G free. Thomas hit this live ("what is
this… the #681 bind was never activated", "how can we solve this") and the #2054 capture
had to be re-routed to a RunPod for storage.

Which carve-out holds it: **destructive/irreversible actions** — the cutover moves live
worktrees (the standing memory note says the migration was deliberately DEFERRED because
it would break live sessions), so the timing/window is Thomas's call.

Related open tracking (named so the PM reconciles): #2099 (proposed) covers the
CLAUDE.md § Disk hygiene DOC claim ("bind status runtime-checked, not asserted"); #2095/
#2097 cover janitor/headroom mechanization. None performs the cutover.

## Suggested action

Pick a low-activity window (fleet idle, no live gate runs), then: quiesce sessions →
rsync `.claude/worktrees` to `/mnt/eps-data` → bind-mount → verify `new_worktree.sh`'s
`EPS_WORKTREE_REQUIRE_BIND=1` assertion passes → resume. The `worktree-migration.LOCK`
machinery for exactly this cutover already exists (`new_worktree.sh` refuses while held).
If preferred, file it as a scheduled `kind: infra` task with the window named and Thomas's
go/no-go as the approval gate. With 782 G free on the data disk, the window is open now.

## Provenance

- origin: /daily 2026-08-05 problem sweep — miner 3 P5 (probed findmnt/df) + miner 4 P2;
  Thomas's live correction in session 7a1632b8 (~22:0xZ 08-05).
