---
name: Codex BLOCKER on unsatisfiable plan directive
description: Codex FAILs a binding |Δ|<1e-3 reproduction gate that hash-seed nondeterminism in the parent code makes unsatisfiable; the implementer's deterministic-variant + disclosed new anchor is the only mechanically correct response. PASS, no round-2 fix exists.
type: feedback
---

**Rule:** when Codex's `*-gate-tolerance-bypassed` BLOCKER cites `|Δ| < X` vs observed `|Δ| = Y > X`, and the implementer's defense names hash-seed-dependent `set()` iteration / PYTHONHASHSEED / fold-overwrite tie-breaks: open the parent helper at the cited line, confirm `for X in set(...):` with rows appearing in multiple folds (last fold wins), so the archived value sits inside an irreducible nondeterminism cloud no re-run can match within the tolerance. The implementer's correct response is (a) a deterministic variant (sorted folds), (b) the deterministic value as the new canonical anchor, (c) explicit disclosure in marker §(d). PASS with a hard standing rec that the clean-result discloses the new anchor (route to clean-result-critic Lens 7). Do NOT bounce — the directive is mechanically unsatisfiable as written.

**Origin:** #511 r1 — bakeoff `_loocv_r2:3098` iterates `set(a)|set(b)`, line 3122 overwrites `pred[test]`; archived CV 0.6086 hash-seed-bound vs deterministic 0.6181.

Companion (inverse): [[feedback_claude_misses_orthogonal_partial_state_flag]] (plan forbids the fix the inherited code needs).

## Index hooks moved from MEMORY.md (#1891 curation, 2026-07-30)

The always-loaded index was curated to fit the ~25 KB loader truncation limit (task #1891); the full pre-curation index hook(s) for this entry are preserved verbatim below.

- [Codex BLOCKER on unsatisfiable plan directive](feedback_codex_unsatisfiable_plan_directive.md) — hash-seed `set()` nondeterminism makes the |Δ|<1e-3 gate unmatchable; deterministic-anchor + disclosure is correct, no round-2 fix exists. #511 r1.
