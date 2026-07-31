# issue-1739 nonlinear-map round — fan-out runbook

Generated 2026-07-30T23:27:57Z at `11f67d4df3d7f4422d9643cda060b3285835463b` on branch `issue-1739`.
Composed by `scripts/issue1739_nlmap_fanout.sh runbook` — nothing was executed.

## Projection (MEASURED basis — see scripts/issue1739_nlmap_project.py)

```
== issue-1739 nonlinear-map fan-out projection (MEASURED basis) ==
  basis: map_fit 2651.32s/key, group walls {250: 4.87, 2500: 46.21, 8000: 140.08}, transfer 28.81s/unit
  phase A: 4 keys x 2651.32s x 2 kinds, 1 wave(s) at width 2
           5.9 GPU-h, wall 2.95 h (width 2)
  lanes (maps STAGED, so map_fit = 0):
    evil           mlp     R=3 L=[250, 2500, 8000]  projected 3.00 h  plan_wall_h 4.5
    evil           kernel  R=3 L=[250, 2500, 8000]  projected 3.00 h  plan_wall_h 4.5
    sycophancy     mlp     R=3 L=[250, 2500, 16000]  projected 3.00 h  plan_wall_h 4.5
    sycophancy     kernel  R=3 L=[250, 2500, 16000]  projected 3.00 h  plan_wall_h 4.5
    hallucination  mlp     R=1 L=[250, 2500, 16000]  projected 1.85 h  plan_wall_h 2.78
    hallucination  kernel  R=1 L=[250, 2500, 16000]  projected 1.85 h  plan_wall_h 2.78
  lanes total: 15.71 GPU-h, wall 3.0 h (concurrency 6)
  ROUND: 21.62 GPU-h, wall 5.95 h
```

## Step 1 — phase A (ONE box, one GPU per map kind)

Fits every (variant, U rung) map ONCE per kind, gates each payload
(save->load->apply allclose, fail-loud), publishes them to HF, then runs
the lanes' OWN staging step against a scratch root as a data-contract
gate. Provision a box with >= 1 GPU per kind, then:

```bash
cd "$WORKLOAD_ROOT"  # or the pod's repo root
EPM_I1739_NL_KINDS='mlp kernel' \
EPM_I1739_NL_USIZES='250 full' \
EPM_I1739_NL_SEEDS='0 1' \
  bash scripts/issue1739_nlmap_fanout.sh phase-a
```

Gate to check before launching any lane: the phase-A log must show one
`[fits] map round-trip gate PASS` per payload, `phase-a: staging-contract
probe PASS`, and a terminal `[phase=done]`.

## Step 2 — the 6 scoring lanes (behavior x kind, single GPU each)

Each lane stages the phase-A payloads (fail-loud if absent — a lane that
silently re-fits would throw away the whole amortization), runs the
path-2 grid, commits its results with a pre-commit ff-sync (#1880 push
race: 6 lanes write this branch concurrently), and tears down.

Run all 6 concurrently. Per-lane `PLAN_WALL_H` is derived from the
MEASURED basis by the projector, NOT inherited from the previous round.

### lane evil / mlp  (pod suffix: `nlevilmlp`)

```bash
EPM_I1739_NL_BEHAVIORS='evil' \
EPM_I1739_NL_KINDS='mlp' \
EPM_I1739_NL_USIZES='250 full' \
EPM_I1739_NL_DRAWS='0 1 2' \
EPM_I1739_NL_SEEDS='0 1' \
EPM_I1739_NL_PLAN_WALL_H=4.5 \
EPM_I1739_NL_PILOT_ABORT_MULT=1 \
EPM_I1739_NL_PHASE='stage,stage_maps,pilot,fits,collect,upload_results' \
  bash scripts/issue1739_nlmap_dispatch.sh
```

### lane evil / kernel  (pod suffix: `nlevilker`)

```bash
EPM_I1739_NL_BEHAVIORS='evil' \
EPM_I1739_NL_KINDS='kernel' \
EPM_I1739_NL_USIZES='250 full' \
EPM_I1739_NL_DRAWS='0 1 2' \
EPM_I1739_NL_SEEDS='0 1' \
EPM_I1739_NL_PLAN_WALL_H=4.5 \
EPM_I1739_NL_PILOT_ABORT_MULT=1 \
EPM_I1739_NL_PHASE='stage,stage_maps,pilot,fits,collect,upload_results' \
  bash scripts/issue1739_nlmap_dispatch.sh
```

### lane sycophancy / mlp  (pod suffix: `nlsycomlp`)

```bash
EPM_I1739_NL_BEHAVIORS='sycophancy' \
EPM_I1739_NL_KINDS='mlp' \
EPM_I1739_NL_USIZES='250 full' \
EPM_I1739_NL_DRAWS='0 1 2' \
EPM_I1739_NL_SEEDS='0 1' \
EPM_I1739_NL_PLAN_WALL_H=4.5 \
EPM_I1739_NL_PILOT_ABORT_MULT=1 \
EPM_I1739_NL_PHASE='stage,stage_maps,pilot,fits,collect,upload_results' \
  bash scripts/issue1739_nlmap_dispatch.sh
```

### lane sycophancy / kernel  (pod suffix: `nlsycoker`)

```bash
EPM_I1739_NL_BEHAVIORS='sycophancy' \
EPM_I1739_NL_KINDS='kernel' \
EPM_I1739_NL_USIZES='250 full' \
EPM_I1739_NL_DRAWS='0 1 2' \
EPM_I1739_NL_SEEDS='0 1' \
EPM_I1739_NL_PLAN_WALL_H=4.5 \
EPM_I1739_NL_PILOT_ABORT_MULT=1 \
EPM_I1739_NL_PHASE='stage,stage_maps,pilot,fits,collect,upload_results' \
  bash scripts/issue1739_nlmap_dispatch.sh
```

### lane hallucination / mlp  (pod suffix: `nlhallmlp`)

```bash
EPM_I1739_NL_BEHAVIORS='hallucination' \
EPM_I1739_NL_KINDS='mlp' \
EPM_I1739_NL_USIZES='250 full' \
EPM_I1739_NL_DRAWS='0 1 2' \
EPM_I1739_NL_SEEDS='0 1' \
EPM_I1739_NL_PLAN_WALL_H=2.78 \
EPM_I1739_NL_PILOT_ABORT_MULT=1 \
EPM_I1739_NL_PHASE='stage,stage_maps,pilot,fits,collect,upload_results' \
  bash scripts/issue1739_nlmap_dispatch.sh
```

### lane hallucination / kernel  (pod suffix: `nlhallker`)

```bash
EPM_I1739_NL_BEHAVIORS='hallucination' \
EPM_I1739_NL_KINDS='kernel' \
EPM_I1739_NL_USIZES='250 full' \
EPM_I1739_NL_DRAWS='0 1 2' \
EPM_I1739_NL_SEEDS='0 1' \
EPM_I1739_NL_PLAN_WALL_H=2.78 \
EPM_I1739_NL_PILOT_ABORT_MULT=1 \
EPM_I1739_NL_PHASE='stage,stage_maps,pilot,fits,collect,upload_results' \
  bash scripts/issue1739_nlmap_dispatch.sh
```

## Notes

- `stage_maps` and `prefetch` are opt-in phases: `PHASE=all` never runs
  them, so the legacy single-box dispatch is unchanged by this round.
- Lanes run `upload_results` (NOT `upload`): re-uploading the identical
  tensors tree from 6 lanes would burn 6 Hub commits against the
  256/hr repo cap for zero new bytes.
- A lane whose pilot projects past its fenced `PLAN_WALL_H` exits rc=7
  (a DESIGNED halt with `pilot_report.json`, not a crash) — re-size from
  that report rather than raising the fence blindly.
- Every lane pins seeds[0]=0, matching the seed phase A fit the
  maps under. `_load_nl_map` refuses a payload whose recorded `map_seed`
  differs: for a subsampled U rung the pool ROWS depend on that seed, and
  the row-count guard cannot see the difference.
