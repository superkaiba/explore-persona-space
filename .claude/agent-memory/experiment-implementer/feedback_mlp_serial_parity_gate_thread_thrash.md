---
name: mlp-serial-parity-gate-thread-thrash-and-device-fallback
description: GD serial-parity gates on tiny cells thrash a wide torch pool (2 threads 265s vs 8 threads >480s timeout); pin threads INSIDE the gate. And issue658 _resolve_device falls back cuda->cpu with only a WARNING - guard production-size fits.
type: feedback
---

Two traps from #928 r6 (indiv-mlp-nonlinearity-control), both inherited-stack foot-guns:

1. **A GD serial-parity gate over TINY cells thrashes a wide torch pool — pin threads INSIDE the gate.** `assert_group_mlp_matches_serial` (12×6 tensors, ~17k tiny AdamW steps across batched+serial paths) ran 265 s at `torch.set_num_threads(2)` but EXCEEDED a 480 s timeout at the shared-VM 8-thread cap — pure op-dispatch thrash (vectorize-rule item 4). A gate that will run inside a production driver must set its own small thread count for its duration and RESTORE the caller's setting (`prev = torch.get_num_threads(); torch.set_num_threads(2); try: ... finally: restore`). The caller's env caps are sized for the PRODUCTION fit, not the gate's tiny cells.

**Why:** the first full-driver smoke timed out in the gate phase; the standalone gate at default threads had already timed out once — same cause, only diagnosed on the second hit.

**How to apply:** any `assert_*_matches_serial`-style gate over small synthetic cells in `vectorized_mlp_skill.py` or a sibling module. Also calibrate the gate tolerance at PRODUCTION epochs (measured 9.5e-07 at 300 epochs vs atol 5e-5, ~50× headroom; real fold-leakage/standardization bugs deviate at ~1e-1). And pin byte-for-byte preservation of a generalized default path by capturing the PRE-edit output sha256 on a seeded synthetic BEFORE editing, asserting MATCH after — cheap and decisive.

2. **`issue658_fit_predictors._resolve_device("cuda")` falls back to cpu with only a WARNING.** Any driver inheriting it for a production-scale GD fit must add a fail-loud guard (cpu-resolved device + production input size ⇒ `SystemExit` unless an explicit `--allow-cpu-production`), or a pod GPU hiccup silently converts a ~0.5 h GPU fit into days of CPU (#928: 28 PFLOP ≈ 190 h at the measured 41 GFLOP/s CPU rate). The fallback is deliberate (VM smokes run the cuda code path on cpu) — guard at the DRIVER, not by changing the shared resolver.

## Index hooks moved from MEMORY.md (#1891 curation, 2026-07-30)

The always-loaded index was curated to fit the ~25 KB loader truncation limit (task #1891); the full pre-curation index hook(s) for this entry are preserved verbatim below.

- [Tiny-cell serial-parity gates thrash wide torch pools + issue658 device fallback](feedback_mlp_serial_parity_gate_thread_thrash.md) — pin 2 threads INSIDE the gate (265s vs >480s timeout at 8); calibrate atol at production epochs (9.5e-7 vs 5e-5); guard cpu-fallback at production size (#928 r6).
