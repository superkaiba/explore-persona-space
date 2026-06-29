---
name: Reuse-provenance paths — blob link only for git-tracked artifacts
description: In Reproducibility reuse-provenance bullets, give git-ignored binaries (.pt/.npz centroids) as plain inline code paths, NOT GitHub blob links — blob links 404 the verifier.
type: feedback
---

When writing the SPEC `**Artifacts:**` reuse-provenance bullets `(a) producing #M, (b) pinned path, (c) fit`, the form of (b) depends on whether the artifact is git-tracked.

**Why:** `verify_task_body.py` check "Reproducibility artifact URLs exist" verifies every `[label](github.com/.../blob/<sha>/<path>)` against the git object DB via `git cat-file`. A GitHub blob URL pointing at a git-IGNORED binary (large `.pt`/`.npz` centroid tensors live on the VM disk / HF, not in the repo) 404s and FAILs the check. #648 round-3 hit exactly this: I linked `eval_results/single_token_100_persona/centroids/centroids_layer20.pt` as a blob URL and the verifier failed it (`does not exist at <sha>`).

**How to apply:** Before turning a reuse-provenance path into a clickable `[...](github.com/.../blob/...)`, confirm it's git-tracked: `git cat-file -e <sha>:<path>` (and/or `git check-ignore <path>`). If tracked (JSON/CSV/script) → SHA-pinned blob link is fine. If git-ignored binary (`.pt`, `.npz`, merged checkpoints) → give it as a plain inline code-span path (`` `eval_results/.../centroids_layer20.pt` ``) and note it's a local-VM/HF artifact, NOT a committed blob. The path is still readable to a downstream auditor on the VM (that's what (b) requires); it just isn't a git URL. Source the per-artifact paths by reading the driver script at its committed SHA, never from the plan's intent.
