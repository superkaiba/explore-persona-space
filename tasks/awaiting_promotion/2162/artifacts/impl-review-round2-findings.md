# Issue #2162 — implementation review round 2, unioned findings (fix round → round 3 of cap 5)

Panel: `plan-adherence-critic` **APPROVE** · `code-correctness-critic` **APPROVE** · `efficiency-critic` **REVISE** (1 Major). The `codex-code-reviewer` twin was again a confirmed no-show (quota until 2026-09-05).

**Your round-2 work was good.** Both approving lenses verified the fixes at the literal level rather than accepting claims: C1's `write_done` guard was confirmed to have exactly two call sites with NO production path reaching `write_done=False`; the pools `pool_key` byte-parity pin was confirmed genuine (two independently-defined f-strings compared over all real pairs, not self-satisfying); stage-2's Δ, α-scaling, single-layer hook, and 12,096 budget all check out against the real hook body; the stage-2 regime fingerprint's `-stage2-add-K1` correctly makes stale r1 replace-mode done-state RAISE rather than resume; and the manifest audit came back 15/15 with producers matching registered transforms. 67/67 tests, ruff clean, security clean. No scope creep — every hunk traced to a numbered finding.

**This round is narrow: one Major, plus residuals both approving lenses scoped as fix-before-report rather than fix-before-pod, plus three hardening nits.**

**Do NOT edit `plans/plan.md` or `artifacts/planned_manifest.json` this round** — the orchestrator owns those text corrections (the §10 launch line and three stale path names) and is applying them in parallel. Touching them would collide.

---

## MAJOR 1 — the `all` chain couples 8×H100 teardown to the Batch-API judge SLA (#664 idle-burn)

`scripts/issue2162_dispatch.sh:158-172, 278-279`.

**The defect traces to MY round-1 wording, not your judgment.** R1 C4 said "wire the margin phase into the production sequence BEFORE upload/terminate". You implemented that literally. The INTENT was *margin must never be silently dropped* — not *margin must precede upload*.

Why it matters: `phase_pools` re-reduces the anchor **behavior-wave judge scores** (`*.anchors.scores.jsonl`) — the ~28k-call Batch-API wave plan §9 books at **2–24 calendar hours** — not the raw P2 anchor generations. Those scores dispatch at earliest ~t≈1.5 h (after gate-3 + the judge pilot, per your M5 fix) and land on the Batch tail. The grid ends ~t≈5.1 h. So pools-before-margin only holds when the Batch wave returns within ~3.5 h of dispatch. In the tail case `all` exits rc=24 at `:167` **before `run_upload` at `:279`** — putting the P5 bulk upload, the pod sentinel, AND the verified-teardown path all behind a leg gated on an external SLA, with the 8× H100 pod alive at 0% utilization. Worst case ≈ 12–20 h × ~$25–30/h. It also contradicts plan §9 directive 5 ("width released before the tail: pod-2162 TERMINATES after P5 upload+verify; judge waves … run off-pod").

**Fix — make margin OPPORTUNISTIC, then defer loudly BEHIND upload:**

`grid → (pools present ⇒ margin) → upload → sentinel carrying margin_deferred: true|false`

- **Modal case** (pools staged mid-grid): margin rides the wide pod at ~28 min (67,392 + 22,464 TF rows × 0.15 s ≈ 3.7 GPU-h / 8). Unchanged from today.
- **Tail case:** log a loud `margin DEFERRED` line, run upload + post the sentinel with `margin_deferred: true` **as a first-class obligation** (see the note below — downstream gates read this), tear the wide pod down per Step 8, and run the deferred leg later on a **fresh 1× H100** once pools land: `dispatch.sh margin && dispatch.sh upload` needs only bank.json + pools.json + the model — same ~3.7 GPU-h, ≈$10–15, zero SLA coupling.
- **Keep rc=24 as the hard HALT for the STANDALONE `margin` invocation** — an explicitly-requested margin with no pools stays an error.
- Optionally prepend a SHORT bounded poll (~15–30 min) before deferring. **An unbounded poll is NOT the fix** — it automates the same burn.

**`margin_deferred` is load-bearing downstream, so make it unmissable.** The upload-verifier's 100%-reconciliation pass and the report pipeline must be able to distinguish "the secondary DV is deferred, with a named recipe to produce it" from "the secondary DV is missing". Put the flag in the sentinel payload AND, when true, make the deferral state legible in the run digest — a deferred DV read as simply absent is exactly the silent-gap failure this project's upload policy exists to prevent.

**Test:** the existing `test_dispatch_all_chain_wiring` asserts the order — flip its expected order to `grid → margin-opportunistic → upload` and assert the sentinel carries the `margin_deferred` key in both branches.

## MINOR 1 (efficiency) — `require_gate3` before the pilot wastes the gate-3 wait

`scripts/issue2162_dispatch.sh:275-276`. `require_gate3` sits BEFORE the pilot in `all`, so a gate-3 report whose sync-judge/scp lags P2's end halts the chain with 8 GPUs up while ~8–15 min of gate-INDEPENDENT pilot work could have absorbed the lag. Swap to `pilot → require_gate3 → grid` (or add a ≤~20-min bounded poll on the report). The pilot's ~180-rollout spend is an acceptable loss on a genuine gate-3 FAIL. Fully orchestrator-controlled (sync judge, no Batch SLA).

---

## Residuals from the APPROVING lenses — both scoped "before REPORT generation, not before the pod run"

Do them now while you are in the files; they are pure re-renders of committed rows with zero GPU risk.

**R1 (adherence) — the hero figure's per-pair companion is off its registered transform.** `scripts/issue2162_figures.py:142-159` (`per_type_f_beh_perpair`) plots STEERED-only, without the separation exclusion, without pair-id labels. Its registered manifest transform says "same exclusion…; one point per surviving pair PER ARM…, labeled by pair id". Your sibling companions (`route_contrasts_perpair`, `stage2_perpair`) DO label and filter, so this is the one off-transform view in the set — and the `report-verifier` recomputes per figure against the manifest transform, so it would FAIL there later.

**R2 (adherence) — the rule-19 validation ρ grain.** `scripts/issue2162_analysis.py:711-724` computes ρ across per-PAIR points; the manifest and plan §4.4 register it "across cells" (per-cell means) "with dynamic range" (no dynamic-range screen implemented). Both grains are post-hoc derivable from `margin_cells.jsonl`. Add the per-cell ρ alongside the per-pair one (≈2 lines) and implement or explicitly declare the dynamic-range screen.

**R3 (adherence, cosmetic) — `per_type_f_beh`** (`figures.py:110-134`) puts all cells on one axis with slot in the tick label, where the transform says "one panel per slot"; post-exclusion n is annotated in the diagnostics survival panel rather than on the hero. **R4 (adherence, cosmetic) — `recency_load_curves`** (`figures.py:328-339`) renders the shuffled null as a per-level mean LINE rather than a shaded band.

## Hardening nits (correctness, non-blocking)

**H1 — drift-harden the pilot regression pin.** `tests/test_issue2162_run.py`: the pilot-call pin is a file-wide `"write_done=False" in src` grep plus a `len(findall("if write_done:")) == 2` count. Present code is correct, but a FUTURE third done-write added to `run_block` outside the guard — or a `write_done=False` on the wrong call site — would not be caught. AST-locate the pilot branch's `run_block` call and assert its `write_done` kwarg is literally `False`.

**H2 — a corrupt claim record is permanently unstealable.** `scripts/issue2162_run.py:301`: a same-host claim record missing `pid` reads `_pid_alive(-1)`, and `os.kill(-1, 0)` SUCCEEDS, so the record reads as live forever. Unreachable via `try_claim` (which always writes `pid`) and JSON corruption already fail-louds, but treat a missing or non-positive pid as DEAD.

**H3 — a missing anchor-incoherence baseline silently becomes 0.** `scripts/issue2162_figures.py` (`fig_diagnostics`): `anchor_incoh.get(cell, 0.0)` substitutes a 0 baseline for a cell whose `anchors.jsonl` rows predate the r2 `n_*_rollouts` fields. Legacy artifacts only — fresh runs always carry them — but log-or-NaN rather than substituting a silent zero.

---

## Constraints

- **Everything here is pre-pod and CPU-verifiable. Do NOT provision a pod.**
- Do not touch what the two approving lenses cleared — the bank, value cycle, nulls, stage-1 grid, statistics, probe, judge discipline, the C1/C2/M1/M2/M3/M5 fixes.
- Do not edit `plans/plan.md` or `artifacts/planned_manifest.json` (orchestrator-owned this round).
- Re-run `uv run pytest tests/test_issue2162_*.py` and report the actual count (67 before this round).
- Post an updated `epm:experiment-implementation` marker (bump the version) with a per-item disposition. Your r2 marker misdescribes the worker-width source as "the dispatcher's `NUM_GPUS` env" — it is derived from `nvidia-smi -L` with no such env var. Correct that too.
- Return SHORT: per-item disposition, test count, anything you think is wrong (with the argument), remaining deviations.
