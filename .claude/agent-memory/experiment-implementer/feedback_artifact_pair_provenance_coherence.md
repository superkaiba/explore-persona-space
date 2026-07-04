---
name: Artifact-pair provenance coherence (sha pins ≠ capture-time identity)
description: A reused artifact PAIR that must be mutually consistent (question banks vs activations captured under them) can be sha-pinned and still inconsistent when one side was regenerated AFTER the dependent capture; check HF last-commit dates + reconstruction metadata. #922 crash-fix r4.
type: feedback
---

sha-pinning a reused artifact verifies BYTE-STABILITY of the current file, not
that it is the file a dependent capture actually consumed. #922
(att-20260703-163130): #779's syc/hall `eval_questions` were Claude-REGENERATED
by a later round ("issue779 r5: reconstructed extraction artifacts", HF commit
9578892ef4 2026-07-02) AFTER the pass_a `cx.pt` activation capture (a8060198a4
2026-07-01) — the artifact's own `reconstruction.regenerated` field documented
it. Prompts rebuilt from the pinned artifact could never reproduce the cached
token stream: fresh-vs-cached parity read cos_mean 0.937 / min 0.709
(question-change band; evil cells with COMMITTED questions read 0.99933).

**Why:** gitignored `data/` inputs regenerate per instance; a crash-fix round
that re-uploads a regenerated input silently orphans every artifact captured
under the original. The mismatch also breaks question↔response pairing for any
teacher-forced reuse.

**How to apply:** before reusing any mutually-dependent artifact PAIR
(prompt/question banks vs activations; mixes vs adapters), compare
`HfApi.get_paths_info(..., expand=True)` last-commit dates + any
`reconstruction`/regeneration metadata — a consumed input that POSTDATES the
dependent capture is inconsistent regardless of sha pins. Scope parity asserts
to provably-reconstructable inputs (committed constants), report the rest per
provenance group. Diagnosis shortcut: cached-side geometry fingerprints
(cross-question / cross-condition / adjacent-layer cosines from the cached
tensors alone) discriminate divergence causes with zero GPU.
