---
name: Claude misses producer/consumer JSON key mismatch (path vs inline)
description: When round-N writes a sidecar JSON path under one key (e.g. payload["dynamics_snapshots_path"]) and the consumer plot/analyzer reads a different key (e.g. ej.get("dynamics_snapshots") expecting an inline list), Claude PASSes on the producing dispatcher's structural correctness (the path IS stamped into the eval JSON). Codex catches that the consumer reads a different key and silently falls back to the empty-list / endpoint-only degraded branch.
type: feedback
---

When a round-N feature (offline extraction, sidecar persistence, dispatch-side artifact stamp) writes data into the eval JSON under one key and a CONSUMER script (plot, analyzer, downstream renderer) reads a DIFFERENT key, Claude PASSes by verifying:
- ✓ the dispatcher writes the key (`payload["dynamics_snapshots_path"] = ...`),
- ✓ the dispatcher has a try/except + log line so the write is "safe",
- ✓ the implementer's marker says "sidecar path threaded into eval JSON via new parameter."

Codex catches that the consumer is `ej.get("dynamics_snapshots") or []` (expecting an inline LIST) when the dispatcher wrote `dynamics_snapshots_path` (a STRING path to a JSON sidecar file). The `or []` swallows the mismatch silently — `snaps` is `[]`, the `if not snaps:` branch fires, the trajectory figure renders single endpoint scatter points instead of multi-snapshot trajectories. From the figure-smoke perspective the output looks "fine" (a figure is rendered, exit-0); only by reading the consumer code can you see it's running the degraded fallback.

**Smell — a tell that this class of bug is present:**
- The dispatcher's comment claims the consumer reads key X, but the consumer actually reads key Y. Example smell at `dispatch_514.py:566`: `# the analyzer's _gather_dynamics_snapshots reads eval_json["dynamics_snapshots_path"]` — but the plot at `plot_issue_514.py:452` reads `ej.get("dynamics_snapshots")`. The comment IS the disambiguator; if it disagrees with the consumer, the bug is in the consumer.
- The consumer has `or []` / `or {}` / `or None` fallback. Fallbacks hide key-typo bugs.
- The producer commits added a NEW key with a SUFFIX (e.g. `_path`, `_sidecar`, `_uri`); the legacy code path read the bare name.

**Why:** writing the path to a sidecar JSON is the canonical persistent-artifact pattern (#508 also did this), but the consumer side has to either (a) open the sidecar and json.loads it, or (b) the producer has to inline the list under the legacy key. The fix is one or two lines either way, but neither side is automatic — the consumer must be updated in lockstep with the producer's key choice.

**How to apply:** Whenever round-N introduces a new key into an eval JSON / payload dict — especially a `*_path` / `*_sidecar` suffix variant of an old inline key — grep the project for BOTH the new key AND the legacy key. Open every consumer that reads the legacy key and verify it reads the new key OR opens the sidecar file. The `ej.get(KEY) or []` fallback is the canonical hider; require an explicit `if KEY not in ej: raise KeyError(...)` or at minimum a `LOG.warn` when the key is absent on a cell that should have it. Origin: task #514 round-2 — dispatcher writes `payload["dynamics_snapshots_path"]` (a path string), plot reads `ej.get("dynamics_snapshots")` (an inline list); trajectory figure silently degenerates to endpoint markers for every cell. Companion to "Claude misses fix regressions" + "Claude misses cross-file consumer regex" + "Claude misses dispatcher-wiring correctness bugs": same family of "the producer side ships fine, the consumer-side key/shape mismatch is invisible from the producer's smoke."
