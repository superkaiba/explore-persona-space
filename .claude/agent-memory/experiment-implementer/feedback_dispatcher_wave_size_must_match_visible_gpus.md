---
name: Dispatcher wave size must equal the visible GPU count
description: A per-cell subprocess dispatcher must derive its parallel wave size from torch.cuda.device_count(), never a hardcoded/--n-gpus-default constant — surplus --gpu-id lanes silently fall back to CPU on a smaller lane
type: feedback
---

Any dispatcher that runs N cells in parallel as subprocesses, one per
`--gpu-id 0..N-1` (the `scripts/i*_dispatch.py` / wave-fan-out family),
MUST set the wave size N from the VISIBLE device count
(`torch.cuda.device_count()` if `torch.cuda.is_available()` else 0), NOT
a hardcoded constant and NOT the `--n-gpus` CLI default.

**Why:** the auto router picks the lane (often single-GPU `lora-7b` /
`eval`). A dispatcher hardcoded to a 4-GPU fan-out then spawns
`--gpu-id 1..3` with `CUDA_VISIBLE_DEVICES=1..3` on a 1-GPU box; those
processes see NO device and SILENTLY fall back to CPU. The GPU cell
finishes in ~40 min; the 3 CPU cells crawl for hours, wave-1 never
finishes, so wave-2 never launches. No traceback — just a hung dispatcher
in `wait()` with zombie CPU subprocesses. (Incident #667 a36-reextract
round-1, 2026-06-28: `phase_extract_r_plus` fanned out `range(4)` on a
single-GPU A100 lane; sp_swe (gpu 0) done in 41 min, the other 3 ran on
CPU for 3h.)

**How to apply:** add a `_compute_wave_size(cpu_only, requested_n_gpus)`
that returns 1 on `--cpu-only`, `min(detected, max(requested_n_gpus,1))`
on GPU, and **raises loud** on 0 visible GPU when not `--cpu-only` (a
wave of 0 is the silent-CPU crash class, never the intent). Keep
`--n-gpus` as a CEILING, not the source of truth. The `i % n_par`
gpu-id assignment then auto-pins each cell to a real device. Keep the
per-cell `CUDA_VISIBLE_DEVICES=str(gpu_id)` launcher-env pin unchanged
(#545) — the bug is the COUNT, not the pin.

`--dry-run` should PREVIEW the *requested* fan-out without touching CUDA
(so a GPU-less VM can still show the per-lane CVD assignment for review),
while the real run uses the detected count.

Sibling rule already in the codebase: `tests/test_cvd_wave_assignment_smoke.py`
(per-GPU CVD pin); add/extend a wave-size test when you write a new wave
dispatcher (the round-2 fix's regression test is
`tests/test_issue667_wave_and_skip.py`).

## Index hooks moved from MEMORY.md (#1891 curation, 2026-07-30)

The always-loaded index was curated to fit the ~25 KB loader truncation limit (task #1891); the full pre-curation index hook(s) for this entry are preserved verbatim below.

- [Dispatcher wave size must equal visible GPU count](feedback_dispatcher_wave_size_must_match_visible_gpus.md) — a per-cell subprocess dispatcher fanning out `--gpu-id 0..N-1` must derive N from `torch.cuda.device_count()`, NEVER a hardcoded/`--n-gpus`-default constant; surplus `--gpu-id` lanes on a smaller lane get `CVD=1..3`, see no device, SILENTLY run on CPU for hours (no traceback). Clamp wave to detected count; raise loud on 0 GPU. #667 a36 r2.
