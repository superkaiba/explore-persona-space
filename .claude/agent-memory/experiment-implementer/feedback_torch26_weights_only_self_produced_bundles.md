---
name: torch>=2.6 weights_only default vs self-produced .pt bundles
description: Pass --no-weights-only / weights_only=False when verifying or loading sha-pinned SELF-PRODUCED bundles on torch>=2.6 lanes; audit every torch.load site on the lane path — the VM torch may mask the trap
type: feedback
---

On torch>=2.6, `torch.load` defaults `weights_only=True` and rejects metadata
globals older self-produced bundles carry (e.g.
`torch.torch_version.TorchVersion` in the #1768 pooled stores). A verifier or
consumer invoked with defaults then crashes with an UnpicklingError
("Unsupported global ...") — #1900's first fellows launch (job 16045) died in
the smoke leg exactly here, at `verify_reused_artifact_keys`'s
`weights_only=True` default.

**Why:** the bundle format is fine (mmap opens); only the unpickler's
allowlist rejects the metadata global. For sha-pinned SELF-PRODUCED bundles
the gotchas.md carve-out sanctions `--no-weights-only` / `weights_only=False`
(prefer it over `--allow-full-load` when mmap works — the failure is the
unpickler, not the format).

**How to apply:** at implementation time, audit EVERY `torch.load` +
`verify_reused_artifact_keys` invocation on the dispatched lane's path and
thread the flag for self-produced bundles; do not trust a green VM run — the
VM's torch version may be older/newer than the lane's (the #1900 fix was
confirmed on VM torch 2.8.0: rc=2 byte-identical error without the flag,
rc=0 with). Fix commit precedent: issue-1900 branch `6b9cf586`.
