# VM many-cell eigh/svd loops balloon RSS via glibc arenas (#1315)

Symptom: a batched Gram-eigh bootstrap (per-pass tensors ≤ tens of MB) grew to
20-21.7 GB RSS across ~7-9 passes and was earlyoom-SIGTERMed twice on the shared VM
(mem avail ~10%). No single allocation was large — classic glibc malloc arena
fragmentation under 8 BLAS/torch threads (M_ARENA_MAX defaults to 8×cores; freed
chunks never return to the OS).

Fix: launch with `MALLOC_ARENA_MAX=2` (+ optional MALLOC_TRIM_THRESHOLD_); RSS held
steady ~1 GB for the identical workload. Pair with: (a) group/phase-level resume so
kills are monotone progress, (b) a bounded retry loop in the runner, (c) `choom -n
-600` (code-style recipe) — but note choom on the launch pids did NOT stick to a
python3 child spawned later by `uv run`; the arena cap is the real fix.
