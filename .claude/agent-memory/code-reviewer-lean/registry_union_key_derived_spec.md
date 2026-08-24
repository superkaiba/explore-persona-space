---
name: registry-union-key-derived-spec
description: dual-namespace registry union review — a spec that is a pure function of its key makes the clobber assert vacuous at its own call sites; verify docstring-claimed loader disjointness by reading the loader (#2479 R10)
metadata:
  type: feedback
---

When a fix unions two key namespaces into one lookup registry (`REGIME_SPECS`-style) with a "clobber-proof" assert (`prev is None or prev == spec`), check whether the registered spec is a PURE FUNCTION of the key (all fields derived from the name string). If so, the assert can NEVER fire from the registration path itself — same key ⇒ same spec — and only guards foreign-namespace rows (hand-written base rows, future non-derived entries). Probe it fires anyway by seeding a mutated row and re-registering.

Second check: a docstring claiming cross-namespace collision is "impossible by construction — enforced by <loader>" must be verified against the loader's ACTUAL validation. In #2479 R10 the loader enforced only a `char_` prefix + suffix conventions + intra-panel uniqueness, NOT disjointness from the 16 legacy `char_*` names — the `char_2479_` disjointness lived only in the committed artifact + its drift-pin tests (Minor, since key-derived specs made a collision benign for lookup; but a colliding panel row WOULD have entered the sweep-feeding list).

**Why:** the union fix's safety story rested on two claims (assert catches clobbers; loader forbids collisions) that were each weaker than stated — behavior was still sound only because of the key-derived-spec property plus test pins on the committed artifact.

**How to apply:** on any dual-namespace registry union: (1) trace whether the sweep-feeding list and the lookup table diverge (sweep membership ≠ lookup resolvability — different contracts, keep only the sweep list branch-scoped); (2) probe the clobber assert with a genuinely differing respec; (3) read the loader's validation lines before crediting a disjointness claim; (4) run the env-unset/default path parent-vs-HEAD full-dict + key-order equality probe (cf. [[fails-pre-fix-probe-parent-commit]]).
