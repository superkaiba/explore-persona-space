# Implementer-lean memory index

- [Guard-hook transcript replay](reference_guard_hook_transcript_replay.md) — store layout, ~30 d retention (snapshot plan evidence!), resumable replay recipe, 25 ms/token walk cost, common-token FP transfer
- [earlyoom SIGTERM storm diagnosis](reference_earlyoom_sigterm_storm_diagnosis.md) — rc=143 + empty output in seconds = journalctl -u earlyoom; choom -600 sweep (re-sweep after uv child spawns); foreground until-loop is guard-accepted
- [c2a-v2 restyle gotchas](reference_c2a_v2_figure_restyle_gotchas.md) — rotated-ylabel overflow into title band (pdftoppm to diagnose); style_score_axis tick snapping; save_c2a_figure record key
- [c2a fixed-scale overflow check](reference_c2a_fixed_scale_overflow_check.md) — PNG px/240 vs authored width catches off-canvas text under bbox-tight; ~55/~33 title-char budgets
- [Root-edit stash race: stage first](reference_root_edit_stash_race_stage_first.md) — unstaged root script edits get reverted by concurrent commits (stale renders, gate INCONCLUSIVE); git add stabilizes; gate cert ~5-10 min, round-unique payload
