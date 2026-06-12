---
name: Claude approves audit plans whose acceptance is anchored to the instrument's own hit set
description: REVISE when a sweep/audit plan defines completeness as "covers all N hits of <registered grep>" — the deliverable passes its own acceptance while missing hand-rolled in-scope sites the pattern can't match.
type: feedback
---

When a plan's deliverable is a COMPLETENESS claim (audit sweep, coverage table,
affected-set reconciliation) and its acceptance criterion is pinned to the hit
set of a registered search pattern ("covers all 27 grep-hit files"), Claude's
critic APPROVEs on the grounds that everything found is analyzer-weighable.
Codex catches that the pattern misses hand-rolled forms, so the audit can
satisfy its own acceptance while incomplete — and the analyzer downstream has
NO signal sites are missing (not analyzer-recoverable).

**Why:** Task #536 round-1 (alternatives lens). Registered Phase A pattern
`compute_cosine_matrix|cosine_similarity|F\.cosine|cos_sim|centering=` had
ZERO matches in `scripts/analyze_100_persona_source_filtered.py:51-54`
(hand-rolled `centroids − mean` → normalize → `c @ c.T`, function named
`cosine_matrix`) and in
`src/.../contrastive_neg_geometry_504/phase05.py:~111` (`np.linalg.norm` +
`unit @ unit.T` with `mean_center` flag). The plan's "add rows for sites the
grep finds beyond the 27" deviation-allowance never fires for sites the
pattern cannot match. REVISE.

**How to apply:** When adjudicating a sweep-completeness Must-Fix, (1) run the
plan's registered command yourself; (2) grep the scope dirs for hand-rolled
equivalents of the construct (normalize + matmul, `.norm(`, `np.dot`,
`einsum`); (3) check whether the acceptance criterion is self-referential
(defined by the instrument's own output). Self-referential acceptance on a
completeness deliverable = conclusion-changing, NOT analyzer-recoverable.
Also: verify EACH of Codex's named examples — in #536 one of three was wrong
at file level (`extract_centroids_and_analyze.py` matched via
`F.cosine_similarity` on other lines); two verified misses still carried
the verdict.
