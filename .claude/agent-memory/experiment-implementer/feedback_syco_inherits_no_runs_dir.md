---
name: Syco arm inherits no runs/ dir
description: When the planner says "inherit from #411 syco panel", remember the predecessor produced an analyze_summary snapshot, NOT per-source runs/ - any substrate builder that calls `_load_runs(...)` unconditionally will FileNotFoundError on syco
type: feedback
---

When the planner says an arm "is INHERITED from #411 (or any prior task) via the frozen leakage matrix / analyze_summary snapshot", that almost ALWAYS means the predecessor produced `eval_results/<prior>/_inputs/<prior>_analyze_summary.json` directly, NOT a `runs/<source>_seed*/run_result.json` directory.

**Why:** #509 used #411's existing leakage panel as substrate; #509 never re-trained sycophancy. The on-disk artifact is the analyze_summary's `per_source[<src>][per_panel_delta|per_panel_trained_rate|per_panel_base_rate]`, keyed by bystander. There is NO `eval_results/issue_509/syco_arm/runs/` directory anywhere — neither locally nor on HF Hub. A `_load_runs(runs_root)` call will FileNotFoundError.

**How to apply:** When wiring a substrate / aggregator builder against an inherited arm, branch on `arm`:
1. Inherited arm: read the snapshot's `per_source[<src>][per_panel_*]` matrix DIRECTLY, synthesize the per-source `{per_cell: [...]}` shape the downstream loop expects.
2. Fresh-trained arm: read `runs/<src>_seed*/run_result.json` as usual.

Pre-launch path-resolution audit: for EVERY consumer phase, enumerate the on-disk paths it actually reads. Each must (a) be produced by a launcher phase, (b) exist in inherited substrate, or (c) be downloaded inline. Argparse-existence + upstream-phase audits are INSUFFICIENT — they miss "the path the consumer reads is wrong even though argparse is happy." Task #518 hit this same shape four times in three rounds (r7 substrate_loader, r10 launcher_missing_gen_refusal, r10 aggregator_missing_bakeoff_meta_args, r11 substrate_syco_missing_runs_data); round-11 PASSed an argparse-existence audit and still missed the runs_root case.
