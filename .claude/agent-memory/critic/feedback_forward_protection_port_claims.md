---
name: Forward-protection port claim-scope + AST-test evasion checklist
description: Infra ports claiming "no future X can regress" vs (one-driver fix + opt-in guard + site-scoped AST test) — claim-scoping concern when an exhaustive audit shows the fixed driver is the only affected one; evasion checklist for AST matchers
type: feedback
---

When an infra plan's Goal says "no future <bug class> can recur" but the deliverable is (a) a fix to the one affected driver, (b) an OPT-IN runtime guard no caller arms, and (c) an AST regression test scoped to one function in one file, the protection is driver-scoped, not universal. Normally a claim-scoping concern (scope the completion text to the audited driver + name the guard as offered-not-armed), NOT a REVISE, when an exhaustive audit (#584: 24-row sweep) shows the fixed driver is the only affected one.

**AST contract-test evasion checklist** (for `isinstance(n.func, ast.Name) and n.func.id == "X"` + keyword-arg matchers):
1. *Positional constant* — `X(name, 1, path)` puts the flagged kwarg in `c.args`; keyword-only checkers pass silently with the bug live. Cheapest real hole; fix = also check `args[i]`.
2. *Attribute call / helper indirection* — the matcher finds ZERO calls, caught LOUDLY by an anti-rot companion test asserting `len(calls) >= 1`. An anti-rot test converts most evasions from silent-pass to loud-fail — check for one before flagging.
3. *Loop-invariant Name* (`x = 1` outside the loop) — `ast.Name` not `ast.Constant`; needs the runtime guard as net; check whether that guard is armed by DEFAULT.
4. *New function / new file* — site-scoped tests pin the incident site, not the class; acceptable if stated.

**Why:** #584 (port of #534's distinct-lora_int_id fix): Goal claimed universal protection while delivering one-driver fix + `--fraction-manifest` opt-in + one-function AST test; the anti-rot test covered attribute/helper evasions, positional-constant was the one silent hole.

**How to apply:** enumerate the checklist against the test's matcher code verbatim; check each evasion fails loud vs passes silent; check the behavioral-net guard fires on the DEFAULT invocation path. vLLM distinct-id sufficiency: ids unique per engine lifetime suffice — verify the engine is constructed inside the per-call function (fresh cache per call); flag if a refactor could hoist it out while ids restart at 1.
