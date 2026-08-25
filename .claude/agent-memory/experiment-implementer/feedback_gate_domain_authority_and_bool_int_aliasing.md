---
name: gate-domain-authority-and-bool-int-aliasing
description: Verification gates take their domain from the CALLER, never the artifact under test; JSON booleans alias ints in identity fields and defeat exact-set reconciliation (#823 r5)
metadata:
  type: feedback
---

Two input-trust rules for any artifact-verification gate (integrity gates,
coverage checks, exact-set reconciliations), from #823 P-Gen fix round 5:

1. **The artifact under test never defines the authority it is verified
   against.** A gate that reads its domain size (n_contexts, row count, cell
   list) from the artifact's own metadata and then checks the artifact against
   that domain is self-referentially satisfiable: a COHERENT smaller artifact
   (100 contexts, five matching 100-entry arms, exactly the matching records)
   passes every internal-consistency check and logs a production PASS
   certifying 2% of the registered domain (#823: 296 of 14,996 pairs). Thread
   the caller-pinned expected domain into the gate as a REQUIRED parameter,
   check artifact-declared == caller-pinned as a RECORDED failure on the
   designed halt path (not an assert), and derive EVERY domain quantity
   (slicing, coverage floors, expected sets) from the caller's value. A
   lower-bound check (`len(x) < declared_n`) is structurally incapable of
   catching truncation when `declared_n` comes from the artifact.

2. **`isinstance(x, int)` accepts JSON booleans, and `True` ALIASES `1`
   everywhere an identity field is consumed** — indexing (`questions[True]`),
   hashing (`hash(True) == hash(1)`), tuple equality (`(True, 1) == (1, 1)`),
   and therefore exact expected-set reconciliation. A malformed record carrying
   `context_id: true` impersonates a legitimate pair end-to-end and passes a
   set-equality gate. Strict identity schemas use `type(x) is int` for every
   int-typed identity/domain field (ids, indices, counts). Container checks
   (list/dict/str) keep `isinstance` — no aliasing hazard there.

**Why:** round-4 review (Claude CONCERNS / Codex FAIL union) found both holes
in the #823 P0 prompt-integrity gate after four hardening rounds; the pre-fix
probe on `e27827027f` demonstrated both passes behaviorally.

**How to apply:** when writing or reviewing any gate/verifier over persisted
artifacts, (a) trace where each domain-defining value comes from — any read of
the artifact's own metadata that then bounds the checks is the #823 r5 shape;
(b) grep the gate for `isinstance(..., int)` on identity fields. Also route
every malformed CONTAINER shape (non-dict roots, nonnumeric dict keys, missing
payload keys) into the gate's recorded-failure path before dereferencing —
a raw AttributeError/KeyError bypasses the designed report + exit code.

Worked implementation: `scripts/issue823_ladder_gen.py::p0_prompt_integrity_gate`
+ `run_p0_verify` (branch `issue-823`, commit `f5a543a80f`); fixtures in
`tests/test_issue823_ladder_gen_fixes.py` (the boolean-impersonation and
coherent-subset tests demonstrably PASS pre-fix). Related: [[sha-pin-domain-mismatch]].
