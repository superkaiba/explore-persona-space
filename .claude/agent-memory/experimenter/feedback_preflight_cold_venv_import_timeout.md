# Cold-venv preflight import-probe timeout after a uv.lock-changing checkout (not the FUSE wedge)

**Incident (#2474 postnorm round, 2026-08-24):** syncing a fresh pod to an issue
branch whose checkout changed `uv.lock` forced a ~10-min `uv` re-sync of 218
packages onto MooseFS and left the venv cold; preflight's 180 s deep-import
probe then timed out spuriously on the first run.

**Rule:** before condemning the pod (or diagnosing the MooseFS FUSE read-wedge),
run the spot-read discriminator — independent reads of `.venv` files, e.g. 1 MB
of `libtorch_cuda.so`. Reads responding ⇒ slow COLD read, not the wedge:

1. Pre-warm imports: `uv run python -c "import torch, transformers, vllm"`
   (~4 min on a cold MooseFS venv).
2. Re-run preflight with `EPM_PREFLIGHT_IMPORT_PROBE_TIMEOUT_S=600`.

Never swap the pod for this signature. Reads hanging ⇒ the real wedge — see
`.claude/rules/gotchas.md` § MooseFS FUSE read-wedge.
