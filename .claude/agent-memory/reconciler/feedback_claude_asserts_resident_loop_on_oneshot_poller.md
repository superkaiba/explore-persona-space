---
name: Claude asserts a resident-loop driver that is actually a one-shot command
description: Codex right when it flags a validator launching a one-shot CLI as if it were a resident poller; verify main() has a loop before crediting a "wired/closed" Claude PASS
type: feedback
---

When the artifact is a live-recovery VALIDATOR that depends on a background
driver process being ALIVE at the moment a fault fires, Codex is right and
Claude's PASS is wrong whenever the launched "driver" is actually a one-shot
command. Verify the driver's `main()` for an actual loop/wait before crediting
any Claude "VERIFIED FIXED / wired / structural-unreachability gap closed".

**Why:** #672 r2. The validator launched `backend_poll.py --issue N` as a bg
Popen ONCE, then `time.sleep(180)`, then injected the fault. But
`backend_poll.py:main()` polls ONCE, prints, `return 0` — no `while`/`for`/
`sleep`, no `--watch`/`--loop` arg. So the poller polled while the VM was still
healthy (failover predicate False), printed, and exited; minutes later when the
watchdog killed the VM the poller was long dead, `_failover_dead_gcp_to_runpod`
never fired, `failover_count` stayed 0 → fallback. The live-recovery headline
was structurally UNREACHABLE. Claude wrote "VERIFIED FIXED ... structural-
unreachability gap is closed" and described a resident "main loop" that does
not exist. The validator's OWN docstring even claimed the failover lives "in
backend_poll.py's main loop" — an authoring-time tell that nobody checked the
referenced loop existed.

**How to apply:**
1. When a Claude PASS hinges on a bg driver/poller/watcher being live during a
   later event, OPEN the driver's entry point and confirm `main()` actually
   loops or blocks. A one-shot `poll → print → return 0` launched before the
   event it must catch is the scaffolded-but-not-plumbed family (sibling of
   `feedback_codex_step_06_literal_vs_purpose` / the reader-with-no-live-
   invocation pattern): the reader exists, the writer/event never reaches it
   live. FAIL.
2. A docstring that asserts "wired in X's main loop" is a hypothesis, not
   evidence — grep X for the loop. A self-inconsistent docstring (claims a loop
   the code lacks) is itself a smell pointing at the bug.
3. Sibling check in the same review: an ADDED mechanical gate can still be
   mis-CALIBRATED against the dispatch that feeds it. #672 r2 also had a
   coverage gate (`MIN_LOG_ENTRIES=30`) that the dispatch's `--log-mem-every 10`
   over a `<100`-forward smoke slice could never satisfy (~10 rows). "Gate added"
   ≠ "gate satisfiable on the healthy path" — compute the realized row/sample
   count from the dispatch argv + the slice arithmetic and check it against the
   threshold. An unsatisfiable healthy-path gate is a blocking FAIL.
