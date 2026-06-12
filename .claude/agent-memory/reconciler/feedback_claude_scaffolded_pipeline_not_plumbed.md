---
name: Claude PASSes scaffolded-but-not-plumbed pipeline
description: When a renderer/analyzer reads optional sidecar JSONs that no producer writes, Claude classifies as Major-not-Critical because tests pass on synthetic data. Codex catches that the pipeline silently no-ops in production.
type: feedback
---

When round-N adds a NEW first-class deliverable that has THREE components — (1) renderer/analyzer reads input, (2) writer/producer creates input, (3) source cadence/sampling produces the right data shape — Claude PASSes (or downgrades to Major-not-Critical) when component (1) lands and the synthetic-data unit test passes, even if (2) and (3) are referenced-but-undefined.

**Why:** Claude verifies the round-(N-1) BLOCKER table item (e.g. "M1 trajectory figures: ADDRESSED") on the renderer presence + analyze-end-to-end test PASS, and stops. The implementer disclosed the missing producers in their `(d) Needs human eyeball` section, which Claude treats as "acknowledged scope caveat, ship it." Codex re-greps the worktree for the writer of the sidecar path that the renderer reads.

**How to apply:** When the reviewer round-N diff adds an `analyze.py` / `render.py` / `plot.py` function that reads `eval_json["X_path"]` or a sidecar file `<dir>/X.json`:

1. `rg "X_path\|X\.json"` across the package + scripts.
2. Confirm at least one location WRITES (not reads) that key/file.
3. If no writer exists AND the deliverable is first-class per the plan, FAIL-class — the headline figure will silently no-op in production.
4. Bonus check: if the renderer + writer both exist, verify the cadence/sampling: a writer that fires only at endpoint (e.g. `ckpt_fractions=(1.0,)`) gives 1 data point when the plan calls for N. Tied to "trajectory" / "dynamics" / "per-step" / "cadence" language in the plan.

Origin: task #508 round-2 (`extract_fullft_dynamics_from_checkpoints` referenced in 3 docstrings but never defined; `MarkerDynamicsCallback.snapshots` in-memory only with no `on_train_end` writer; `ckpt_fractions=(1.0,)` endpoint-only despite plan §4.4 "every 4 training steps"). Plan §4.7 explicitly: "trajectory figures (the 2 NEW first-class entries) ship in the clean-result body regardless." Claude classified M1 as Major-not-Critical because the headline cluster-bootstrap H1 still works on synthetic data; reconciler upgraded to FAIL because plan §5 ties trajectories to H1 matched-rate interpretation ("smoke gate's sub-ceiling check is endpoint-only; the trajectory IS the shape over training").

Companion to "Claude treats round-N-1 must-fix as acceptance" and "Claude misses fix regressions" — same disease (per-item-table-walking the BLOCKER list and stopping at "implemented"), different surface (here it's the producer side of a read/write pair that landed unwired).
