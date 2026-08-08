# Implementer-lean memory index

- [Guard-hook transcript replay](reference_guard_hook_transcript_replay.md) — store layout, ~30 d retention (snapshot plan evidence!), resumable replay recipe, 25 ms/token walk cost, common-token FP transfer
- [earlyoom SIGTERM storm diagnosis](reference_earlyoom_sigterm_storm_diagnosis.md) — rc=143 + empty output in seconds = journalctl -u earlyoom; choom -600 sweep (re-sweep after uv child spawns); foreground until-loop is guard-accepted
