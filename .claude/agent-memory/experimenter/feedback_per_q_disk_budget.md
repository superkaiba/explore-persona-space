---
name: per_q caches blow disk budget on continuous-sweep extraction
description: When restarting a multi-method extraction sweep that writes flat per_q .pt caches per persona, sum sizes BEFORE launching. r_per_token at (n_q, n_layers, n_pos, D) fp16 alone can exceed 100 GB.
type: feedback
---

When the sweep writes per_q tensors via `torch.save(torch.stack(buf), method_X/<role>__per_q.pt)`, the per-persona size is `n_q * n_layers * (n_pos?) * D * 2` bytes. For issue #263 with n_q=240, n_layers=28, D=3584, fp16=2, the response-side methods produced:

| method | shape | size/persona | × 275 |
|---|---|---|---|
| method_b | (240, 28, 3584) | 46 MB | 13 GB |
| method_bstar | (240, 28, 3584) | 47 MB | 13 GB |
| method_r_per_token | (240, 28, 9, 3584) | **417 MB** | **115 GB** |
| method_a | (240, 28, 5, 3584) | 240 MB | 66 GB |
| method_caa | (240, 28, 5, 3584) | 240 MB | 66 GB |
| method_c{1,2,3} | (240, 28, 3584) each | 46 MB each | 13 GB × 3 |

Total per_q for the full 8-method sweep: **~310 GB** — much greater than the default 200 GB ephemeral pod volume.

**Why:** for marker / extraction sweeps that need per-question rows for downstream H2 / AUC analysis, every method writes its own per-persona per_q tensor. r_per_token is the worst offender because it has an extra response-position dimension.

**How to apply:**

1. **Before launching ANY extraction sweep that writes per_q caches**, compute total `n_personas * size_per_persona * n_methods` and compare to free disk. If >70% of free disk, refuse to launch and post `epm:failure failure_class: code` requesting either (a) stream-and-delete to HF Hub mid-run, (b) a bigger pod volume, or (c) reducing per_q to only the cells the analyzer consumes.
2. **Don't trust "the same launch worked before"** — issue #263's sweep originally crashed at persona 218/275 on disk-full. The retrospective uploader dump made it look like a success, but the underlying disk-budget defect was unresolved. Resuming the same launch reproduces the same failure later in the loop, just shifted by the cache-hit we restored. **2026-05-08 confirmation:** respawn 1/3 reproduced exactly this — kicked off, walked Method A cache in <30 s, started B/B*/R, and the on-disk per_q sizes (433 MB r_per_token + 48 MB b + 48 MB bstar = 530 MB/persona) projected to 145 GB additional disk for B/B*/R alone, exceeding 107 GB free. Killed the run cleanly at persona 2 of B/B*/R and posted `epm:failure v1 failure_class:code` for implementer to add streaming-upload or skip non-H3 r_per_token positions.
3. **Empirically verify per_q size on the FIRST persona** of each method before letting the loop run further. `du -sh method_X/<first_persona>__per_q.pt` × n_personas is your real footprint, regardless of estimates.

This is structural — not a hot-fix candidate. Bounce back to implementer for streaming/upload-and-delete patterns.
