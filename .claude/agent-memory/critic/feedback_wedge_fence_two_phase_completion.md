---
name: Wedge-fence two-phase completion + bump-site reachability
description: Watcher wedge-trigger plans — verify WHO re-fires the respawn action after the fence's stop tick, and that any counter bump keyed on same-tick wedge evidence at the SPAWN tick is reachable (#1209)
type: feedback
---

For any `autonomous_session_watch.py` plan adding/extending a prompt-wedge trigger: the stop-verify fence is TWO-phase (stop tick → verify-dead spawn tick), and `decide_respawn_fence` has exactly ONE call site inside `_handle_stalled_respawn` — reachable only when `action == "respawn"` re-fires on the later tick. Trace who re-fires it:

- Existing stale-shape triggers re-fire via `decide_session_stalled` (missed pinned at threshold + STALE self-report). A FRESH self-report returns `("keep", 0)` unconditionally (L1565), discarding the pin.
- The wedge itself cannot re-fire post-stop: its pid gate (`pids_by_sid.get(sid)`, L10133-10136) and the fence's `sid_alive` read come from the SAME daemon `/list` (L5644, L19328-19331) — dead-in-live_ids ⇔ absent-from-pids_by_sid. So "wedge fires + fence spawns on the same tick" is production-impossible; a test that constructs sid-dead-in-live_ids-but-live-in-pids_by_sid is green-and-vacuous (R-class fixture coupling).
- A fresh-report trigger's fence therefore completes via the CRASH arm (`decide` L654-666: 2-miss ≈ 20-30 min, no per-day cap, books nothing in stalled state) or via late staleness — NOT "+10 min fence".

**Why:** #1209 v1 keyed its per-day respawn cap bump on `ctx.wedge_note` inside `_fence_spawn_stalled`'s spawn_ok branch — unreachable on the canonical path (cap structurally inert, top-risk churn mitigation void) while all 14 planned tests could pass via direct-kwarg calls + the impossible T10 pass-2 state. Its "wedge re-fires on stop AND spawn tick ⇒ escalation-site over-counts" premise was wrong (pid gate bounds re-fire to failed-stop ticks; `stop_pending_sid is None` dedups exactly once).

**How to apply:** For any new trigger whose detection window sits INSIDE the 60-min self-report freshness window, demand: (1) an explicit post-stop completion path (who spawns? crash arm ⇒ latency +20-30 min, no stalled-state bookkeeping); (2) any cap/counter bump at the ESCALATION site with `stop_pending_sid is None` dedup, not the fence-spawn site; (3) one cap-binding test through the PRODUCTION state path (seed `stalled-<N>.json` on disk, run `stalled_session_pass`, assert no stop) — direct-kwarg override tests cannot catch a dropped production thread of defaulted kwargs.
