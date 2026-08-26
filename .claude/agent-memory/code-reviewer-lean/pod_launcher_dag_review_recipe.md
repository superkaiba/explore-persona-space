---
name: pod-launcher-dag-review-recipe
description: 6-probe recipe for reviewing a new pod workload launcher (.sh phase DAG) — sentinel freshness, producer parity, envelope/glob, env-knob threading
metadata:
  type: feedback
---

Reviewing a NEW pod launcher (#2587 r2 g2 shape: bash phase DAG driving several python drivers), run these six probes:

1. **Phase list vs plan DAG** — folded phases are fine (plan P7 embed inside launcher p6) but name the mapping; VM-side phases (judge/analysis) legitimately absent.
2. **Gate-sentinel freshness** — a `require_<sentinel>` before every expensive wave is only sound if the composer ALWAYS re-runs and rewrites the sentinel earlier in the same strictly-sequential fail-loud script; then no stale prior-run sentinel can bless a now-failing gate. Check the composer is not resume-skippable.
3. **Compose-required record list vs real producers** — for every run_meta key a composer gate requires `passed: true` on, grep the driver for `_update_run_meta(..., "<key>", ...)` and confirm the record is written UNDER THE EXACT INVOCATION SHAPE the launcher uses (e.g. `smoke_shard` only on the `--no-upload` branch). Multi-line calls hide keys from single-line greps — grep the bare key string too. The tests' consumer-authored run_meta fixture ([[smoke-fixture-authored-with-consumer-keys]]) proves nothing without this.
4. **Consumer fixture vs producer store layout** — extract the producer at the COMMIT (`git show <sha>:...`) and diff store paths, dict keys, manifest names/keys against the probe's asserts ([[paired-script-default-path-contract]]).
5. **Sentinel envelope + drain glob** — pod-side end-of-run JSON must carry `poll_pipeline._SENTINEL_REQUIRED_KEYS` (`sentinel_schema_version`/`kind`/`version`) AND land at top-level `/workspace/logs/issue-<N>-*.json` ([[handrolled-pod-sentinel-envelope]], [[sentinel-path-outside-drain-glob]]).
6. **Env-knob threading symmetry** — every launcher env override (prefix, out-root) must reach BOTH the producing waves and the consuming waves; a knob threaded into P2/P3 uploads but not the P4 fits `--store-prefix` leaves a default-prefix stale-store read channel. Also check `env "${PINS[@]}"` doesn't clobber a shell `export` (LD_LIBRARY_PATH) the bootstrap set.

**Why:** all four r2-g2 concerns and both blocker-fix certifications fell out of exactly these probes; none required reading the drivers in full.

**How to apply:** any commit adding/editing a `scripts/issue*_pod_workload.sh`-class launcher or a compose-gate that keys on run_meta records.
