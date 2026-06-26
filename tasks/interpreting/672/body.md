---
title: '#669/#671 verified offline (250 tests green) but the live A100 smoke and live
  watchdog→failover chain were lost to infrastructure failure — HIGH offline, UNVERIFIED
  live (LOW confidence)'
kind: experiment
tags: []
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
# #669/#671 verified offline (250 tests green) but the live A100 smoke and live watchdog→failover chain were lost to infrastructure failure — HIGH offline, UNVERIFIED live (LOW confidence)

<!-- clean-result-v4 -->

## Takeaways

- **250/250 regression tests + 2 lints green on `main`** confirms the #669 wedge-recovery logic and #671 extractor memory fix landed cleanly — the only HIGH-confidence claim here.
- The deliverable headline — *"GCP works again, the hung-RUNNING wedge class self-recovers"* — **cannot be issued**: both live sections produced zero live evidence.
- **Two independent failures fired, not one.** The single GCP dispatch hit `ZONE_RESOURCE_POOL_EXHAUSTED` on both `us-central1-c` rungs, fell through to the RunPod terminal rung, which *itself* failed (exit 120, `no_compute_available`).
- **Root cause is unresolved**: the validator's record flags it as capacity-miss-plus-failover OR a dispatch-router bug, and the marker trail can't disambiguate (every `epm:backend-selected` hardcodes `requested_kind=None`).
- **Section A's separate live A100 smoke was never dispatched** — only Section B's debug launch fired; Section A ran ANALYSIS-only and was to launch afterward, which never happened.
- Remaining gap is purely **live infrastructure evidence**: re-run when GCP capacity frees (`us-central1-b` had capacity → pin the zone), and diagnose the router path so a repeat failure is attributable.

## Goal

**This experiment in context:** This is the end-to-end validation gate for two GCP infrastructure fixes that landed just before it. The hung-but-RUNNING GCP networking-wedge class was discovered in [#667](https://eps.superkaiba.com/tasks/667); [#669](https://eps.superkaiba.com/tasks/669) added the recovery backstop (poller wedge-detection on a frozen non-terminal phase plus drain-timeout, an in-VM reachability watchdog that self-terminates, and exactly-once failover to RunPod); [#671](https://eps.superkaiba.com/tasks/671) fixed the root-cause memory accumulation in the activation extractor (`output_hidden_states=True` was retaining all hidden-state tensors per iteration, climbing the resident GPU pool until the guest network wedged). This task asks the integration question those two fixes leave open: with both merged to `main`, does a real GCP A100 run of the fixed extractor complete with flat memory, and does a deliberately wedged VM actually self-recover with no human in the loop?

**Broader narrative:** The project's compute pipeline routes GCP-first (credits-backed capacity before the free SLURM lanes), so GCP reliability directly gates how much science the fleet can run autonomously. A wedge class that silently burns a RUNNING-but-dead VM, with no automatic recovery, is a standing tax on every long run. The validation question is whether the recovery machinery is trustworthy enough to rely on unattended.

## Methodology

**Design:** A three-section system-validation run, no model training. Section A is a single live GCP A100 smoke of the fixed hook-based extractor with per-iteration memory logging. Section B is a live fault-injection on a small GCP VM (drop both the metadata endpoint and outbound 443 with `iptables`, watch the in-VM watchdog force the instance to TERMINATED, then confirm automatic RunPod re-dispatch). Section C is a regression sweep of the #669/#671 test files plus two workflow lints. The single manipulated variable versus the failed [#667](https://eps.superkaiba.com/tasks/667) launch is "the #669 + #671 fixes are now applied" — no new recipe or training.

**Training:** N/A — no model training. The run reuses [#537](https://eps.superkaiba.com/tasks/537)'s frozen contexts and marker adapter read-only as inputs to the extractor smoke; nothing is fit.

**Evaluation:** Three dependent variables, each a direct binary/pass-fail measurement of an infrastructure construct.
- **Memory flatness (Section A):** the construct is "the #671 fix removed the resident GPU climb that wedged the network." Two primary gauges per iteration — `torch.cuda.memory_reserved()` and `nvidia-smi --query-gpu=memory.used` — with `torch.cuda.memory_allocated()` as a secondary diagnostic. PASS = the run completes, writes at least one `.npz`, both primary gauges' max−min stay under 1 GiB across at least 30 iterations with no monotone climb, and the artifacts upload. The reserved/`nvidia-smi` gauges are primary because under `expandable_segments:True` the allocator can read flat on `memory_allocated()` while the reserved pool fragments upward, which is the actual wedge driver.
- **Self-recovery (Section B):** the construct is "a wedged VM self-terminates and re-dispatches with no human." PASS (live path only) = the serial-console watchdog ladder reaches its threshold, the VM reaches TERMINATED, a second backend launch fires with a `gcp_workload_failover_runpod*` reason, exactly one failover fires, and zero manual action occurs between injection and RunPod relaunch. The documented fallback (deterministic watchdog + failover tests only) is explicitly NOT a Section B PASS — it forces an `inconclusive_live_validation` outcome and downgrades the headline.
- **Regression health (Section C):** the construct is "the fixes are live and nothing regressed." PASS = the named test files all green and both lints exit 0.

**Data extraction:** [#537](https://eps.superkaiba.com/tasks/537)'s frozen contexts (`data/issue_537/contexts/`) and its marker LoRA adapter were to be consumed read-only by the Section A extractor smoke; no data was written or modified. The judge model is not involved (no judged behavior in this validation).

**Sample training/evaluation data + completions:**

n/a — no model completions were generated. Section A's live A100 forward-pass, which would have produced the per-iteration memory log and the `.npz` activation artifacts, was never executed (see Results); Section C produced machine-readable test/lint exit records only.

## Results

### Section C: 250 regression tests and 2 lints all green on `main`

The figure plots, per validation section, the evidence actually obtained: pytest count for Section C, live iterations logged for Section A, live failover attempts for Section B. Section C is the only non-zero bar — the two live sections produced no live evidence (see below).

![Horizontal bar chart of three validation sections. Section C (regression tests + lints) is green at 250 evidence units; Section A (live A100 memory-flatness smoke) and Section B (live watchdog to failover chain) are both 0, labeled NOT EXERCISED live.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/4e8ff74bfd0ef714cd299c72818649f378c7d77e/figures/issue_672/validation_status.png)

> **Figure.** *One section green offline, two never run live.* Section C: 250 tests + 2 lints green. Sections A and B logged 0, lost to infrastructure failure (zone stockout on both rungs plus a RunPod terminal-rung failure; root cause unresolved). Counts are heterogeneous — a status indicator, not a comparable scale.

The sweep ran four files at code `5da9b78559`: `test_gcp_backend.py` 203/203, `test_backend_poll.py` 18/18 (frozen-phase wedge-detection + failover-once), `test_issue671_extraction_hooks.py` 8/8 (#671 byte-identity / memory-non-growth), `test_failure_classifier.py` 21/21; both lints exit 0. This is the run's one HIGH-confidence result — #669/#671 landed on `main` with no regression. It cannot exercise either fix against real GCP hardware: the tests run under mocks and CPU stubs, verifying logic, not live behavior.

### Section A: the live A100 memory-flatness smoke was never dispatched

Section A ran ANALYSIS-only under `--section all`: the run-launched note stated the separate live A100 (`--intent lora-7b`) smoke would launch afterward, and it never did. The record shows `n_iters_logged: 0`, `npz_written: false`, `pass: false`. The intended measurement was a per-iteration GPU-resident-memory trace from the fixed extractor on a real A100, against the documented pre-#671 climb from 22 to 38 GiB.

The fixed-extractor *code* is verified offline — `test_issue671_extraction_hooks.py` (8/8) covers the byte-identity and memory-non-growth properties, and the review chain confirmed the `--log-mem-every` helper is byte-additive. Missing is solely the live A100 evidence that memory stays flat under the real `expandable_segments` allocator. The infrastructure failure that consumed the only GCP launch (Section B's) is shared context, not Section A's own dispatch failure: Section A never reached a launch attempt.

### Section B: the live watchdog→failover chain was lost to a two-stage infrastructure failure

The intended live fault-injection (cut a VM's network → watchdog forces TERMINATED → automatic RunPod re-dispatch) never started: `live_injection_pass: false`, `failover_count: 0` (a *not-attempted* zero — the GCP debug-VM never came up).

The validator's argv carried `--backend gcp --intent debug`. Both GCP rungs returned `ZONE_RESOURCE_POOL_EXHAUSTED` in `us-central1-c`: rung 1 (mislabeled `ondemand_a100_80`) tried the debug intent's L4 — *"A g2-standard-4 VM instance with 1 nvidia-l4 accelerator(s) is currently unavailable in the us-central1-c zone"*; rung 2's `a2-highgpu-1g` A100 — *"…unavailable in the us-central1-c zone. Consider trying your request in the us-central1-b zone(s), which currently has capacity."* Dispatch fell through to the RunPod terminal rung, which itself failed — *"runpod terminal fallback FAILED (CalledProcessError … exit status 120)"*, `reason=no_compute_available` (pod-672 was briefly RUNNING, then terminated cleanly, no artifacts). Root cause — capacity-miss-plus-failover vs router bug — is unresolved; the marker trail can't disambiguate (`epm:backend-selected` hardcodes `requested_kind=None`). The deterministic watchdog + failover-once tests are green in Section C, but per plan §1 the live self-recovery claim is unverified.

**Follow-ups:** `_post_intermediate_marker` hardcodes `requested_kind=None` (`backends/router.py`); GCP rung labels mislead for non-A100 intents; the un-hooked `_context_vector_all_layers`.

---

**Repro:** Compute: ~0 GPU-h on the science (Section A never dispatched; Section B's only GCP launch zone-stockout'd before any live GPU forward-pass); ~5 min CPU on the VM for Section C; ~\$0.50 burned on a RunPod pod that hung in SSH bootstrap before being terminated cleanly. Code SHA at run start: [`5da9b78559`](https://github.com/superkaiba/explore-persona-space/tree/5da9b78559) (the validator + the #669/#671 fixes under test); final worktree commit `9c0f6008ed` on branch `issue-672`. Validator: `scripts/issue672_validate.py`. Evidence: `eval_results/issue_672/{validation,section_A,section_B,section_C}.json` (committed on `issue-672`, not yet on `main`). Test surface: `tests/test_gcp_backend.py`, `tests/test_backend_poll.py`, `tests/test_issue671_extraction_hooks.py`, `tests/test_failure_classifier.py`. Reused read-only inputs: [#537](https://eps.superkaiba.com/tasks/537) frozen contexts (`data/issue_537/contexts/`) + marker LoRA adapter from `superkaiba1/explore-persona-space` — fit: read-only extractor inputs, not retrained, not measurement-bearing for this validation. Figure: [`figures/issue_672/validation_status.png`](https://github.com/superkaiba/explore-persona-space/blob/4e8ff74bfd0ef714cd299c72818649f378c7d77e/figures/issue_672/validation_status.png) (status indicator only — this is a binary pass/fail validation with no aggregate statistic).

**Context:** Origin prompt (verbatim): *"afterwards test that GCP is working properly (and the bugs from before won't happen again)"*. Lineage: [#667](https://eps.superkaiba.com/tasks/667) (discovered the hung-RUNNING GCP networking-wedge class) → [#669](https://eps.superkaiba.com/tasks/669) (wedge-recovery backstop: poller detection + watchdog self-terminate + exactly-once RunPod failover) → [#671](https://eps.superkaiba.com/tasks/671) (root-cause extractor `output_hidden_states` memory fix) → #672 (this end-to-end validation). Created 2026-06-26; run 2026-06-26.
