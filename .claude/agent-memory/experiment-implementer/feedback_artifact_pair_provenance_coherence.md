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

## Index hooks moved from MEMORY.md (#1891 curation, 2026-07-30)

The always-loaded index was curated to fit the ~25 KB loader truncation limit (task #1891); the full pre-curation index hook(s) for this entry are preserved verbatim below.

- [Artifact-pair provenance coherence](feedback_artifact_pair_provenance_coherence.md) — sha pins ≠ capture-time identity: a question bank REGENERATED after its dependent activation capture (reconstruction.regenerated field; HF last-commit dates) breaks parity + q↔response pairing; assert only on provably-reconstructable inputs. #922 r4.

## Merged sibling index rows (#2032 curation, 2026-08-03)

This entry is the PRIMARY index pointer for its theme; the sibling index rows below were merged into one index row to fit the agent-memory index size cap (task #2032). Each merged row is preserved verbatim — follow its pointer for the sibling lesson's own entry file.

- [HF mirror divergence — pin content hashes at prefetch](feedback_hf_mirror_divergence_pin_hashes.md) — issue-owned input snapshots + sha256 pins (#600)
- [Pinned artifact pairs can disagree](feedback_pinned_artifact_pair_mutual_inconsistency.md) — assert per-(persona,q) coverage against the (#601)
- [sha pins live in a DOMAIN](feedback_sha_pin_domain_mismatch.md) — recompute a reused pin from its producer's (#1776)
- [Deviation path → sweep all pin verifiers](feedback_deviation_path_sweeps_all_pin_verifiers.md) — an authorized artifact deviation flips
