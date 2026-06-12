---
name: Codex BLOCKER on unsatisfiable plan directive
description: Codex FAILs round-N when the plan's binding numerical directive can't be satisfied due to hash-seed nondeterminism in the parent code under test; implementer's autonomous deterministic-anchor + disclosure is correct
type: feedback
---

When a plan §"Step 0 numerical reproduction gate" carries a binding `|Δmetric| < 1e-3 AND ...` directive against a parent script's archived value, and the parent script's metric path contains an unordered set-iteration overwrite (Python `set()` over strings → hash-seed-randomized), the archived value sits at the bottom of an irreducible ~5e-3 nondeterminism cloud and CANNOT be matched within 1e-3 by ANY downstream re-run.

**Why:** The implementer's only mechanically correct response is (a) write a deterministic variant of the parent helper (sorted folds / sorted keys), (b) make the deterministic value the canonical anchor for the new task, (c) disclose the gap explicitly in §(d) "needs human eyeball." Codex FAILs by reading the plan directive literally and ignoring the parent code's actual semantics; Claude correctly grades Major-with-disclosure (Lens-7 statistical-framing concern for the clean-result-critic, not a code BLOCKER).

**How to apply:** When Codex's `*-gate-tolerance-bypassed` BLOCKER cites a `|Δ| < X` directive vs a `|Δ| = Y > X` observation, AND the implementer's defense names "hash-seed-dependent set() iteration" / "PYTHONHASHSEED" / "fold-overwrite tie-break" — open the parent helper at the cited line, grep for the actual `for X in set(...):` pattern + check whether each test row appears in multiple folds (so the last fold to execute wins). If the mechanism is literally that, the implementer's deterministic-variant + disclose-in-marker resolution is correct. PASS with hard standing recommendation that the clean-result body MUST disclose the deterministic value is the new anchor, not match the archived nondeterministic value within the binding tolerance. Do NOT bounce for a round-2 re-implementation — the plan directive is mechanically unsatisfiable as written and no round-2 fix exists.

Smell signatures:
- Plan directive says "AND on two tolerances, halt on fail"
- Implementer report observes ρ-tolerance PASS, CV-tolerance miss by ~5e-3 (an order of magnitude over the 1e-3 cap)
- Implementer's defense names `set()` iteration + `PYTHONHASHSEED`
- Implementer ships `_fn_deterministic` (sorted folds) alongside the parent `_fn`
- Implementer flags the discrepancy for human eyeball in marker §(d)

Origin: task #511 round-1. Codex FAILed `reproduction-gate-cv-tolerance-bypassed`. Verified bakeoff `_loocv_r2:3098` iterates `set(cond_ids_a) | set(cond_ids_b)`, line 3122 overwrites `pred[test]` (each pair touches BOTH fold C=A and fold C=B). The archived #502 CV (0.6086) is hash-seed-bound; implementer's deterministic anchor (0.6181) is what every downstream re-run will produce. PASS with the disclosure obligation routed to clean-result-critic Lens 7.
