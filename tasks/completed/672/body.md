---
title: The extractor memory fix held flat on a live GCP SPOT A100-80, while watchdog
  recovery remains partial-live and deterministically covered (MODERATE confidence)
kind: infra
tags:
- followup-manual
created_at: '2026-06-26T10:24:35Z'
has_clean_result: false
origin_prompt: afterwards test that GCP is working properly (and the bugs from before
  won't happen again)
goal: 'After #669 (GCP wedge recovery) and #671 (extractor output_hidden_states memory-bug
  fix) merge, validate end-to-end that GCP works again — a real a2-highgpu-1g A100-40
  smoke run of the fixed hook-based extractor completes with FLAT per-iteration GPU
  memory and no DHCP-wedge/OOM, AND an injected sustained-network-loss self-recovers
  (in-VM watchdog self-terminate -> TERMINATED -> #659 async failover to RunPod, no
  manual pivot; poller wedge-detection fires exactly-once as the alternate trigger)
  — with all #669/#671 regression tests green on main.'
---
# The extractor memory fix held flat on a live GCP SPOT A100-80, while watchdog recovery remains partial-live and deterministically covered (MODERATE confidence)

<!-- clean-result-v4 -->

**Methodology:** [docs/methodology/issue_672.md](https://github.com/superkaiba/explore-persona-space/blob/223fbf167722f899cca3d67c32410bfe6308f886/docs/methodology/issue_672.md) · [gist](https://gist.github.com/superkaiba/885833ca018e0bc90a54c46887875160)

## Takeaways

- **The extractor memory fix holds live.** On a live GCP SPOT A100-80 (647 iterations), post-warmup resident memory ranged **0.79 GiB** (1 GiB PASS band) versus the pre-fix climb.
- The live-tensor footprint (`memory_allocated`) was flat to **0.014 GiB** across 638 samples, so the residual 0.79 GiB is allocator fragmentation, not a leak.
- **The in-VM watchdog fired on a real DNS failure**, logging ticks 1/10, 2/10, 3/10 — live evidence the detection works, though the full ladder and failover were not observed live.
- The full watchdog→shutdown→failover chain is covered by **250/250 deterministic tests + 2 lints green on `main`**; no failover fired live, and none should have (bootstrap `setup_failed` is correctly excluded).
- **The dispatch-router walks the full ladder when given a time-budget.** The parent's RunPod fall-through was a missing `--time-budget-hours` skipping every SPOT rung, not a router defect.
- Residual gap is narrow: the full live wedge→shutdown→failover sequence was not reproduced (bootstrap failed before the workload phase); deterministic coverage stands in.

## Goal

**This experiment in context:** This is the end-to-end validation gate for two GCP infrastructure fixes that landed just before it. The hung-but-RUNNING GCP networking-wedge class was discovered in [#667](https://eps.superkaiba.com/tasks/667); [#669](https://eps.superkaiba.com/tasks/669) added the recovery backstop (poller wedge-detection on a frozen non-terminal phase plus drain-timeout, an in-VM reachability watchdog that self-terminates, and exactly-once failover to RunPod); [#671](https://eps.superkaiba.com/tasks/671) fixed the root-cause memory accumulation in the activation extractor (`output_hidden_states=True` was retaining all hidden-state tensors per iteration, climbing the resident GPU pool until the guest network wedged). This task asks the integration question those two fixes leave open: with both merged to `main`, does a real GCP A100 run of the fixed extractor complete with flat memory, and does a deliberately wedged VM actually self-recover with no human in the loop?

**Broader narrative:** The project's compute pipeline routes GCP-first (credits-backed capacity before the free SLURM lanes), so GCP reliability directly gates how much science the fleet can run autonomously. A wedge class that silently burns a RUNNING-but-dead VM, with no automatic recovery, is a standing tax on every long run. The validation question is whether the recovery machinery is trustworthy enough to rely on unattended.

## Methodology

**Design:** A three-section system-validation run, no model training. Section A is a single live GCP A100 smoke of the fixed hook-based extractor with per-iteration memory logging. Section B is a live fault-injection on a small GCP VM (drop both the metadata endpoint and outbound 443 with `iptables`, watch the in-VM watchdog force the instance to TERMINATED, then confirm automatic RunPod re-dispatch). Section C is a regression sweep of the watchdog-recovery and extractor-fix test files plus two workflow lints. The single manipulated variable versus the failed [#667](https://eps.superkaiba.com/tasks/667) launch is "the watchdog-recovery and extractor-memory fixes are now applied" — no new recipe or training.

**Round 2 (live-gcp-revalidation-spot follow-up):** The parent round's single GCP dispatch fell through to the RunPod terminal rung and never produced live evidence for Sections A or B. This round re-dispatched both sections on GCP SPOT capacity. The fix versus the parent dispatch was threading `--provisioning-model SPOT --spot-tolerant --time-budget-hours 0.5` into `dispatch_issue.py launch`: a missing time budget makes the router's short-job test return False, which short-circuits every SPOT rung in the cost ladder. With the flag, Section A's `--intent lora-7b` landed directly on a SPOT A100-80 (`a2-ultragpu-1g`, us-central1-a, 227 s workload wall, 647 iterations); Section B's `--intent debug` walked all four ladder rungs (on-demand L4, on-demand A100-40, spot L4 all `ZONE_RESOURCE_POOL_EXHAUSTED`) and landed on the spot A100-40 rung (`a2-highgpu-1g`). Section C re-ran identically to the parent round. The parent round's recipe text above is unchanged; only the dispatch flags and the realized SKUs differ.

**Training:** N/A — no model training. The run reuses [#537](https://eps.superkaiba.com/tasks/537)'s frozen contexts and marker adapter read-only as inputs to the extractor smoke; nothing is fit.

**Evaluation:** Three dependent variables, each a direct binary/pass-fail measurement of an infrastructure construct.
- **Memory flatness (Section A):** the construct is "the extractor fix removed the resident GPU climb that wedged the network." Two primary gauges per iteration — `torch.cuda.memory_reserved()` and `nvidia-smi --query-gpu=memory.used` — with `torch.cuda.memory_allocated()` as a secondary diagnostic. PASS = the run completes, writes at least one `.npz`, both primary gauges' max−min stay under 1 GiB across at least 30 post-warmup iterations with no monotone climb, and the artifacts upload. The reserved/`nvidia-smi` gauges are primary because under `expandable_segments:True` the allocator can read flat on `memory_allocated()` while the reserved pool fragments upward, which is the actual wedge driver.
- **Self-recovery (Section B):** the construct is "a wedged VM self-terminates and re-dispatches with no human." PASS (live path only) = the serial-console watchdog ladder reaches its threshold, the VM reaches TERMINATED, a second backend launch fires with a `gcp_workload_failover_runpod*` reason, exactly one failover fires, and zero manual action occurs between injection and RunPod relaunch. The documented fallback (deterministic watchdog + failover tests only) is explicitly NOT a Section B PASS — it forces an `inconclusive_live_validation` outcome and downgrades the headline.
- **Regression health (Section C):** the construct is "the fixes are live and nothing regressed." PASS = the named test files all green and both lints exit 0.

**Data extraction:** [#537](https://eps.superkaiba.com/tasks/537)'s frozen contexts (`data/issue_537/contexts/`) and its marker LoRA adapter were consumed read-only by the Section A extractor smoke; no data was written or modified. The smoke slice was `--behavior marker --source-cid default --targets default,sp_swe --layers 7 14 21 --primary-layer 14 --max-probes 8 --max-new-tokens 256 --log-mem-every 1`, producing six `.npz` activation tensors (two contexts × three layers). The judge model is not involved (no judged behavior in this validation).

**Sample training/evaluation data + completions:**

n/a — no model completions were generated or judged. Section A's live A100 forward-pass produced a per-iteration memory log (649 samples, full trace at `eval_results/issue_672/secA_smoke_followup/memory_log.json`) and six `.npz` activation tensors uploaded to the HF data repo; the first and last logged iterations are shown verbatim in the Section A result below. Section C produced machine-readable test/lint exit records only.

## Results

### The fixed extractor's resident memory stayed flat across the live A100 smoke

This figure plots per-iteration GPU memory from the live Section-A smoke — both primary gauges (reserved, nvidia-smi) and the secondary allocated — across all 649 samples, with the documented pre-fix climb (22→38 GiB) overlaid dashed. PASS band: 1 GiB around the post-warmup median.

![Line plot of GPU memory in GiB versus extractor iteration. Three nearly-flat live traces (reserved, nvidia-smi, allocated) sit just above 30 GiB across 649 iterations, inside a shaded 1 GiB PASS band; a dashed diagonal labeled pre-fix climb rises from 22 to 38 GiB.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/f0dc500ee40fb4988f8f255b0f9f7f5999628683/figures/issue_672/secA_memory_flat_live_followup.png)

> **Figure.** *The fix holds on real hardware.* Live GCP SPOT A100-80, 647 iterations, 638 post-warmup samples. Reserved and nvidia-smi range 0.79 GiB (PASS < 1 GiB); allocated range 0.014 GiB. The dashed pre-fix climb (22→38 GiB monotone) is what the fix removed.

The run completed (227 s wall), wrote all six `.npz` tensors, and verified the HF upload. Both primary gauges ranged 0.79 GiB post-warmup with no monotone climb; the secondary allocated gauge was flat to 0.014 GiB, so the live-tensor footprint is genuinely flat and the small reserved wiggle is allocator fragmentation well inside the band. First logged iteration reserved 29.11 GiB / allocated 28.54 GiB; last 30.99 GiB / 28.55 GiB. This is the first live evidence the fix holds under the real `expandable_segments` allocator, which CPU-stub tests cannot exercise.

### The in-VM watchdog fired on a real DNS failure (partial live; full chain via deterministic tests)

This round re-dispatched the live fault-injection on a GCP SPOT A100-40. The intended sequence (cut the VM's network with `iptables` at T+5 min → watchdog forces TERMINATED → automatic RunPod re-dispatch) did not complete: `live_injection_pass: false`, `failover_count: 0`, verdict `inconclusive_live_validation` per plan §4.

What happened instead is itself partial live evidence. During bootstrap, `curl https://astral.sh/uv/install.sh` failed DNS resolution; the startup-script's fail-handler powered off, and in parallel the in-VM eps-watchdog (the watchdog-recovery fix) logged `BOTH reachability probes FAILED (1/10), (2/10), (3/10)` on the serial console — the detection logic firing on genuine unreachability. The planned `iptables` injection never ran because sshd was already down by T+5 min. The full 10/10 ladder, in-VM shutdown, and the asynchronous failover to RunPod were not reached because the workload phase never started; a bootstrap `setup_failed` is correctly excluded from that failover by design, so no failover fired and none should have. The full chain is covered deterministically in the regression sweep (`test_watchdog_terminates_only_when_both_probes_fail` and `test_terminal_workload_wedged_fails_over_to_runpod_exactly_once` green). This is stronger than the parent round's zero live evidence, but the live end-to-end recovery claim remains unverified.

### All 250 regression tests and 2 lints stayed green on `main`

The sweep ran four files: `test_gcp_backend.py` 203/203, `test_backend_poll.py` 18/18 (wedge-detection + failover-once), `test_issue671_extraction_hooks.py` 8/8 (memory-non-growth), `test_failure_classifier.py` 21/21; both lints exit 0. These run under mocks and CPU stubs, verifying logic; the live behavior they stand in for is now also directly evidenced above by the live A100 memory trace and partially by the in-VM watchdog firing on a real DNS failure during bootstrap. (Qualitative regression-sweep result; no quantitative figure per the v4 Lens 9 exemption for infrastructure-validation status checks.)

### The dispatch-router walks the full ladder when given a time-budget

The parent round's fall-through to RunPod raised the question of a `--backend gcp` router bug; it turned out to be a missing flag, not a defect. Both Round 2 launches provisioned on GCP SPOT once `--time-budget-hours` (or `--spot-tolerant`) was threaded: Section A landed directly on an A100-80, Section B walked all four ladder rungs to a SPOT A100-40. The parent's fall-through traced to a missing time budget making the short-job predicate return False, which short-circuits every SPOT rung; with the flag, the ladder behaves as designed. The remaining issue is discoverability — the CLI does not warn when SPOT is requested without a time budget.

**Follow-ups:** the CLI should warn when SPOT is requested without a time budget (discoverability gap, not a bug); the GCE startup script's `curl https://astral.sh/uv/install.sh` should retry on DNS failure (single-shot now, transiently fails on SPOT VM network init); the SPOT `--instance-termination-action=DELETE` does not fire on a user-initiated in-VM `shutdown`, leaving a RUNNING zombie until external delete; the residual un-hooked `_context_vector_all_layers` extractor call.

---

**Repro:** Compute: ~0.5 GPU-h on the science (Section A: GCP SPOT A100-80, `a2-ultragpu-1g`, us-central1-a, 227 s wall ≈ 0.06 GPU-h; Section B: GCP SPOT A100-40, `a2-highgpu-1g`, ~12 min including stockout ladder ≈ 0.2 GPU-h), plus ~5 min CPU on the VM for Section C. Parent round burned ~\$0.50 on a RunPod pod that hung in SSH bootstrap. Code SHA at Section-A dispatch: `0f80ffa59b` on `main` (the validator + #669/#671 fixes); final worktree commit `9c0f6008ed` on branch `issue-672`; figure commit [`f8ad7b1ff2`](https://github.com/superkaiba/explore-persona-space/tree/f8ad7b1ff29207e4b4e13e5a4e9cdc279b96f47a). Validator: `scripts/issue672_validate.py`; hero-figure script `scripts/issue672_plot_secA_memory_flat_followup.py`. Evidence: `eval_results/issue_672/{validation_followup,section_A_followup,section_B_followup,section_C_followup}.json`, the per-iteration trace `eval_results/issue_672/secA_smoke_followup/memory_log.json` (649 samples), and the reconstructed serial-console log `eval_results/issue_672/secB_partial_live_followup/serial_evidence.txt`. Live `.npz` activation tensors (6 files, ~10 MB): [`issue672_live_revalidation_spot/analysis_tensors/secA_smoke_followup/`](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/a961b94d57f8f57bb99af023bf975cff833d64e8/issue672_live_revalidation_spot/analysis_tensors/secA_smoke_followup). Test surface: `tests/test_gcp_backend.py`, `tests/test_backend_poll.py`, `tests/test_issue671_extraction_hooks.py`, `tests/test_failure_classifier.py`. Reused read-only inputs from [#537](https://eps.superkaiba.com/tasks/537): frozen contexts (`data/issue_537/contexts/`) + a marker LoRA adapter from `superkaiba1/explore-persona-space` (revision pin unavailable — read-only reuse of an active sibling line's adapter as an extractor input; provenance traceable via the prior follow-up's clean-result) — fit: read-only extractor inputs, not retrained, not measurement-bearing for this validation. Hero figure: [`figures/issue_672/secA_memory_flat_live_followup.png`](https://github.com/superkaiba/explore-persona-space/blob/f0dc500ee40fb4988f8f255b0f9f7f5999628683/figures/issue_672/secA_memory_flat_live_followup.png).

**Context:** Origin prompt (verbatim): *"afterwards test that GCP is working properly (and the bugs from before won't happen again)"*. Lineage: [#667](https://eps.superkaiba.com/tasks/667) (discovered the hung-RUNNING GCP networking-wedge class) → [#669](https://eps.superkaiba.com/tasks/669) (wedge-recovery backstop: poller detection + watchdog self-terminate + exactly-once RunPod failover) → [#671](https://eps.superkaiba.com/tasks/671) (root-cause extractor `output_hidden_states` memory fix) → #672 (this end-to-end validation). Round 2: `live-gcp-revalidation-spot` (source: user-chat, 2026-06-26) — re-ran Sections A+B on GCP SPOT after the parent round's capacity fall-through to RunPod. Created 2026-06-26; run 2026-06-26 (both rounds).
