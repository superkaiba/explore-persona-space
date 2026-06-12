---
name: Forward-protection port claim-scope + AST-test evasion checklist
description: Infra ports claiming "no future X can regress" — check Goal's universal quantifier vs (one-driver fix + opt-in guard + site-scoped AST test); AST contract-test evasion checklist
type: feedback
---

Rule: when an infra plan's Goal says "no future <bug class> can recur" but the deliverable is (a) a fix to the one currently-affected driver, (b) an OPT-IN runtime guard no caller arms, and (c) an AST regression test scoped to one function in one file, the protection is driver-scoped, not universal — new drivers / new functions / unarmed default invocations are unprotected. This is normally a claim-scoping concern (recoverable: scope the completion text to the audited driver + name the opt-in guard as offered-not-armed), NOT a REVISE, when an exhaustive audit (e.g. #549's 24-row sweep) shows the fixed driver is the only affected one.

AST contract-test evasion checklist (for `isinstance(n.func, ast.Name) and n.func.id == "X"` + keyword-arg matchers):
1. **Positional constant** — `X(name, 1, path)` puts the flagged kwarg in `c.args`; keyword-only checkers pass silently with the bug live. Cheapest real hole; fix = also check `args[i]` at the known position.
2. **Attribute call** (`mod.X(...)`) and **helper indirection** (construction moved out of the scoped function) — both make the matcher find ZERO calls, so they are caught LOUDLY by an anti-rot companion test asserting `len(calls) >= 1`. An anti-rot test converts most evasions from silent-pass to loud-fail — always check whether the plan includes one before flagging these.
3. **Loop-invariant Name** (`x = 1` outside the loop) — `ast.Name`, not `ast.Constant`; static check passes. Needs the runtime guard as net; check whether that guard is armed by default.
4. **New function / new file** — site-scoped tests pin the incident site, not the bug class; acceptable if stated.

**Why:** task #584 (port of #534's distinct-lora_int_id fix): Goal said "no future multi-checkpoint vLLM eval can silently serve a stale adapter" while delivering a one-driver fix + `--fraction-manifest` opt-in guard + one-function AST test. The anti-rot test covered attribute/helper evasions; positional-constant was the one silent hole; both affected audit rows were the same driver so the scoped fix was complete for the known class.

**How to apply:** alternatives lens on kind:infra protection ports — enumerate the evasion checklist against the test's matcher code verbatim, check whether each evasion fails loud (anti-rot) or passes silent, and check whether the "behavioral net" guard fires on the DEFAULT invocation path. Distinct-id sufficiency under vLLM LRU: ids unique per engine lifetime suffice; verify the engine is constructed inside the per-call function (fresh cache per call) — flag if a refactor could hoist the engine out while ids restart at 1.
