# Methodology — issue 672: #669/#671 verified offline (250 tests green) but the live A100 smoke and live watchdog→failover chain were lost to infrastructure failure

*Derived from the [task body](https://eps.superkaiba.com/tasks/672).*


**Design:** A three-section system-validation run, no model training. Section A is a single live GCP A100 smoke of the fixed hook-based extractor with per-iteration memory logging. Section B is a live fault-injection on a small GCP VM (drop both the metadata endpoint and outbound 443 with `iptables`, watch the in-VM watchdog force the instance to TERMINATED, then confirm automatic RunPod re-dispatch). Section C is a regression sweep of the #669/#671 test files plus two workflow lints. The single manipulated variable versus the failed [#667](https://eps.superkaiba.com/tasks/667) launch is "the #669 + #671 fixes are now applied" — no new recipe or training.

**Training:** N/A — no model training. The run reuses [#537](https://eps.superkaiba.com/tasks/537)'s frozen contexts and marker adapter read-only as inputs to the extractor smoke; nothing is fit.

**Evaluation:** Three dependent variables, each a direct binary/pass-fail measurement of an infrastructure construct.
- **Memory flatness (Section A):** the construct is "the #671 fix removed the resident GPU climb that wedged the network." Two primary gauges per iteration — `torch.cuda.memory_reserved()` and `nvidia-smi --query-gpu=memory.used` — with `torch.cuda.memory_allocated()` as a secondary diagnostic. PASS = the run completes, writes at least one `.npz`, both primary gauges' max−min stay under 1 GiB across at least 30 iterations with no monotone climb, and the artifacts upload. The reserved/`nvidia-smi` gauges are primary because under `expandable_segments:True` the allocator can read flat on `memory_allocated()` while the reserved pool fragments upward, which is the actual wedge driver.
- **Self-recovery (Section B):** the construct is "a wedged VM self-terminates and re-dispatches with no human." PASS (live path only) = the serial-console watchdog ladder reaches its threshold, the VM reaches TERMINATED, a second backend launch fires with a `gcp_workload_failover_runpod*` reason, exactly one failover fires, and zero manual action occurs between injection and RunPod relaunch. The documented fallback (deterministic watchdog + failover tests only) is explicitly NOT a Section B PASS — it forces an `inconclusive_live_validation` outcome and downgrades the headline.
- **Regression health (Section C):** the construct is "the fixes are live and nothing regressed." PASS = the named test files all green and both lints exit 0.

**Data extraction:** [#537](https://eps.superkaiba.com/tasks/537)'s frozen contexts (`data/issue_537/contexts/`) and its marker LoRA adapter were to be consumed read-only by the Section A extractor smoke; no data was written or modified. The judge model is not involved (no judged behavior in this validation).

**Sample training/evaluation data + completions:**

n/a — no model completions were generated. Section A's live A100 forward-pass, which would have produced the per-iteration memory log and the `.npz` activation artifacts, was never executed (see Results); Section C produced machine-readable test/lint exit records only.

