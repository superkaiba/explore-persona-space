# issue-1739 nonlinear-map round — fan-out runbook

Generated 2026-07-31T00:36:00Z at `e488e5bca58ad2c0a808cadf81db812d260e38e2` on branch `issue-1739`.
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

## Step 3 — scope addendum: LINEAR composition-factor cells

f_U x f_L at U=5000, LINEAR map, E1, both variants, over each behaviour's
own L ladder — matching the compose cells already committed for evil.
These ride an EXISTING lane's box (no new instances) but are a SEPARATE
dispatcher invocation: the map kind differs from the lane's, so they get
their own out-root (`.../compose_linear`) and their own derived pilot
fence. The nonlinear lanes' `PLAN_WALL_H` is untouched.

Run each AFTER its host lane's own phases finish (the GPU is then free);
a lane box that is already torn down needs its own provision instead.

### compose cells: hallucination  (host lane `nlhallmlp` = hallucination / mlp)

```bash
EPM_I1739_NL_BEHAVIORS='hallucination' \
EPM_I1739_NL_COMPOSE_BEHAVIORS='hallucination' \
EPM_I1739_NL_PHASE='compose' \
  bash scripts/issue1739_nlmap_dispatch.sh
```

Derived compose fence: `5.21`h (measured LINEAR basis).

### compose cells: sycophancy  (host lane `nlsycomlp` = sycophancy / mlp)

```bash
EPM_I1739_NL_BEHAVIORS='sycophancy' \
EPM_I1739_NL_COMPOSE_BEHAVIORS='sycophancy' \
EPM_I1739_NL_PHASE='compose' \
  bash scripts/issue1739_nlmap_dispatch.sh
```

Derived compose fence: `4.71`h (measured LINEAR basis).

**Cost of the addendum** (projector, MEASURED LINEAR basis):

```
== issue-1739 compose-cell projection (LINEAR basis, scope addendum) ==
  f_U x f_L combos (dedup'd): [[0.0, 0.0], [0.5, 0.0], [0.5, 1.0]]
    hallucination  L=[250, 2500, 16000]  20 map fits, 6 compose cells/anchor
      (2 + 6x3) map fits x 221.96s + (2 + 6) groups x 1007.46s
      planned 3.47 h (fence basis)  realized 2.88 h (−2 max-anchor skips: [[0.5, 0.0]])  plan_wall_h 5.21
    sycophancy     L=[250, 2500, 16000]  20 map fits, 6 compose cells/anchor, walls proxied from hallucination
      (2 + 6x3) map fits x 161.81s + (2 + 6) groups x 1007.46s
      planned 3.14 h (fence basis)  realized 2.58 h (−2 max-anchor skips: [[0.5, 0.0]])  plan_wall_h 4.71
  COMPOSE TOTAL: planned 6.61 GPU-h, realized 5.46 GPU-h
```

The default anchor set is each behaviour's FULL L ladder. To trim to the
two cheap anchors (the `250 2500` subset), set
`EPM_I1739_NL_COMPOSE_BUDGETS="250 2500"` on the invocation — the fence
derives from whatever anchors are passed, so it re-sizes automatically:

```
== issue-1739 compose-cell projection (LINEAR basis, scope addendum) ==
  f_U x f_L combos (dedup'd): [[0.0, 0.0], [0.5, 0.0], [0.5, 1.0]]
    hallucination  L=[250, 2500]  14 map fits, 6 compose cells/anchor
      (2 + 6x2) map fits x 221.96s + (2 + 6) groups x 162.54s
      planned 1.22 h (fence basis)  realized 1.22 h (−0 max-anchor skips: [])  plan_wall_h 1.84
    sycophancy     L=[250, 2500]  14 map fits, 6 compose cells/anchor, walls proxied from hallucination
      (2 + 6x2) map fits x 161.81s + (2 + 6) groups x 162.54s
      planned 0.99 h (fence basis)  realized 0.99 h (−0 max-anchor skips: [])  plan_wall_h 1.49
  COMPOSE TOTAL: planned 2.21 GPU-h, realized 2.21 GPU-h
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
- The Step-3 compose cells expect an f_u>0/f_l=0 combo to SKIP at the
  top anchor (empty residual eliciting pool once L covers the train set)
  — evil recorded the same skip as a missing `fu0.5_fl0.0` label at
  L=8000. The projector reports it as `skipped_combos_at_top`; the fence
  is sized on the UNSKIPPED (planned) count, so the skip only ever
  under-runs the fence.
- Every lane pins seeds[0]=0, matching the seed phase A fit the
  maps under. `_load_nl_map` refuses a payload whose recorded `map_seed`
  differs: for a subsampled U rung the pool ROWS depend on that seed, and
  the row-count guard cannot see the difference.
