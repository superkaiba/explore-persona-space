---
name: dotenv-ordering-fix-review-recipe
description: Reviewing a load_dotenv-before-heavy-import fix — 3 checks beyond the pre-fix probe; the runtime-defeat check (provider's package __init__ chain) is the non-obvious one (#2254 R1 g4)
metadata:
  type: feedback
---

A dotenv-before-heavy-import ordering fix (the #847 / `test_no_new_torch_before_dotenv_vm_entrypoints` class) needs three checks beyond
[[fails-pre-fix-probe-parent-commit]]:

1. **Grandfathered-ledger check** — grep `GRANDFATHERED_TORCH_BEFORE_DOTENV`
   in `tests/test_shared_vm_thread_caps.py` for the round's files: adding a
   ledger entry makes the test green without fixing anything.
2. **Runtime-defeat check (non-obvious)** — the fix imports
   `explore_persona_space.orchestrate.env`, which executes
   `explore_persona_space/__init__.py` → `orchestrate/__init__.py` →
   `orchestrate/fleet.py` first. The test's AST scan sees only the
   entrypoint's OWN module-top imports, so a heavy root anywhere in that
   package-init chain would defeat the fix at RUNTIME while staying
   test-green. Grep each link's module-top imports against
   `HEAVY_IMPORT_ROOTS` (test file ~line 764: torch/numpy/matplotlib/pandas/
   scipy/sklearn/seaborn/transformers/datasets/statsmodels/vllm/peft/trl).
3. **Round-completeness for free** — the test scans ALL tracked
   `scripts/**/*.py` plus `__main__`-guarded `src/**/experiments` modules, so
   ONE passing HEAD run certifies no other tracked round entrypoint violates;
   only untracked/new-class files need a manual look. Library modules without
   a `__main__` guard are legitimately out of scope (torch at module top is
   fine there).

**Why:** validated #2254 R1 g4 — all three ran in ~4 tool calls; the parent
chain happened to be clean (fleet.py is stdlib-only) but nothing else would
have caught it dirty. Re-validated #2379 R1 g5, #2477 R1 g2, and #2479 R1 g7
(chain still clean all three times; #2479 hit the check-3 issue823 red +
attribution recipe verbatim).

**Check-3 nuance (#2477 R1 g2):** the one HEAD test run can come back RED on
a PRE-EXISTING sibling offender (there: `issue823_shared_persona_paired.py`,
landed on main with no load_dotenv call at all). Attribution recipe: (a) the
round's file must be ABSENT from the assertion's violations list — that list
enumerates every offender, so absence certifies the round file under the same
scan even when the run is red; (b) the named offender's blob must be
byte-identical at the branch base (`git rev-parse <base>:<path>` vs
`HEAD:<path>`) — identical ⇒ pre-existing, not payload-attributed; surface it
upward as informational (the #1388 fleet-wide-gate-red shape), never a
blocker on this round. Also check the ledger dict names in the live test —
they are `GRANDFATHERED_895` / `GRANDFATHERED_1187`, not the older
`GRANDFATHERED_TORCH_BEFORE_DOTENV` token (grep bare `GRANDFATHERED`).

Cheap fails-pre-fix probe (pairs with [[fails-pre-fix-probe-parent-commit]]):
import the test module's own `_first_heavy_import_line` / `_first_load_dotenv_line`
(`sys.path.insert(0, "tests")`) and run them on the parent blob extracted to
/tmp vs the HEAD file — one python -c call certifies VIOLATION→CLEAN without
a pytest run against the parent tree (which the working-tree-scanning test
cannot do anyway).

**How to apply:** any commit whose message cites
`tests/test_shared_vm_thread_caps.py` or moves `load_dotenv()` above imports.
