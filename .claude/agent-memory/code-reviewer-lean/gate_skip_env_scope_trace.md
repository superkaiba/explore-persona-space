---
name: gate-skip-env-scope-trace
description: Before flagging a dispatcher gate-skip env var as a gate bypass, trace what it actually bypasses — if the driver's own requirement/content checks still bind on the threaded inputs, it is an escape hatch around a DUPLICATED pre-check, not a bypass (#2329 q35_ladder_decay R1 g5)
metadata:
  type: feedback
---

On a shell dispatcher whose gate pre-check has a skip env (`EPM_*_SKIP_*=1 + _REASON`), trace the post-skip path before assigning severity: in #2329's ladder dispatcher the skip bypassed only the dispatcher's fast-fail while the gate file paths were STILL threaded to the driver, whose own non-smoke requirements + content validation (read_gate_verdict etc.) still bind — a coherent escape hatch for dispatcher/driver schema drift, Minor not FAIL. A skip that ALSO un-threads the inputs would be a real bypass.

**Why:** the skip's blast radius is defined by the downstream authoritative check, not by the pre-check it disables; mis-grading it either forces a needless re-roll or waves through a genuine gate hole.

**How to apply:** two companion probes from the same round: (1) a fan-out passing both env `CUDA_VISIBLE_DEVICES="$g"` AND a `--gpu-id "$g"` flag is only safe if the flag is verified informational in the driver (grep its consumption — a torch device-ordinal use crashes on width>1 via CVD reindexing; cf. [[fanout-cvd-ordinal-not-entry]]); (2) a venv-pin asserted only in the stage1 composite, with stage2 unpinned on a resumed/fresh pod, grades by loudness — unknown-arch model-load crash = Minor recommendation; a silent wrong-version run would be a blocker (cf. [[staging-gate-single-phase-silent-fallback]]). Also: executing a dispatcher's misuse branches live (unknown phase / missing gate files / crafted failing-content JSON, /tmp roots) is cheap and certifies every claimed rc before any fan-out.
