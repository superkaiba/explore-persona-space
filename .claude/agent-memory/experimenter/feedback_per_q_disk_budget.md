---
name: per_q caches blow disk budget on extraction sweeps — compute footprint BEFORE launch
description: per-persona per_q tensors at (n_q, n_layers, n_pos, D) fp16 reach 417 MB/persona; a full 8-method x 275-persona sweep ≈ 310 GB >> the 200 GB pod volume. Verify empirically on the first persona.
type: feedback
---

Extraction sweeps writing flat per_q tensors (`torch.save(torch.stack(buf), method_X/<role>__per_q.pt)`) cost `n_q × n_layers × (n_pos) × D × 2` bytes per persona. #263 (n_q=240, 28 layers, D=3584): r_per_token = 417 MB/persona → 115 GB alone across 275 personas; the full 8-method sweep ≈ **310 GB** vs a 200 GB volume.

**Why:** #263 crashed at persona 218/275 disk-full; a retrospective uploader dump made it look like success, and the 2026-05-08 respawn reproduced the same crash (530 MB/persona observed → 145 GB projected vs 107 GB free; killed at persona 2 and bounced).

**How to apply:**
1. Before launching, compute `n_personas × size_per_persona × n_methods` vs free disk; if >70% of free, refuse and post `epm:failure failure_class: code` requesting stream-and-delete to HF Hub, a bigger volume, or trimming per_q to the cells the analyzer consumes. Structural — not a hot-fix; bounce to implementer.
2. Don't trust "the same launch worked before" — resuming shifts the crash later by exactly the cache restored.
3. Empirically verify on the FIRST persona of each method: `du -sh method_X/<first>__per_q.pt` × n_personas is the real footprint.
