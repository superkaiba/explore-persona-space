---
name: --resume must pin every output-affecting regime key, not just substrate
description: When adding --resume (or any checkpoint-reuse cache) to a script with multiple INPUT regimes (--cc / method / hyperparameter choice that changes output identity but is NOT in the substrate fingerprint), pin EVERY regime key in the manifest and check each; a substrate fingerprint hashing only seed/data/dims will silently mix cross-regime caches.
type: feedback
---

When adding `--resume` (or any checkpoint-reuse cache) to a script with
multiple INPUT regimes — a `--cc` / method / hyperparameter choice that
changes the output's identity but is NOT covered by the substrate / data
fingerprint — pin EVERY such regime key in the resume manifest and
assert each in the resume check, raising on mismatch (and refusing a
legacy manifest missing the key).

**The trap.** A `_substrate_fingerprint` that hashes only seed / probe-
pool-hash / n_contexts / data-dim will be IDENTICAL across two argparse
`--cc {C_last, C_meanprompt}` choices that select different columns from
the SAME loaded data dict. So a `--cc C_meanprompt --resume` into a
`layers/` dir filled under `--cc C_last` passes substrate-fingerprint +
seed + do_mlp checks, silently reuses the cached C_last rows, and writes
`c_C_recipe: C_meanprompt` over them — the canonical output is mis-
labeled with no error.

**The fix (mechanical, ~20 LOC):** add the regime key to
`_write_resume_meta`'s payload + the comparison in `_check_resume_meta`;
raise `RuntimeError("regime mismatch: layers/ written with <K>=<X>, current run <K>=<Y>")`
on mismatch. Strict on missing key in legacy manifests (refuse to resume,
force backfill or fresh launch). Add a regression test mirroring the
existing substrate-mismatch test.

**Best mechanical defense:** auto-derive the regime-key set from the
argparse `choices` / output-affecting flag list, so the resume manifest
cannot drift as new regimes are added.

**Incident:** #722 round 2 added `--resume` with per-layer atomic writes
but the manifest's substrate fingerprint did not cover the `--cc` choice;
Codex round-2 code-review caught it; reconciler upheld FAIL; round-3
patch (commit `96402e4928`) added `cc_key` to the manifest + check + a
regime-mismatch unit test.
