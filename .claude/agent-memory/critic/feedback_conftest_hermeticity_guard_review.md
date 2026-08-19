---
name: conftest-hermeticity-guard-review
description: Round-2 checks for autouse conftest guards that redirect a shared resolver (sidecar/path) to pytest tmp with a delegate-on-pinned-root branch (#2141)
metadata:
  type: feedback
---

Review recipe for a plan adding an autouse conftest fixture that REDIRECTS a
module's path resolver (e.g. `_disk_guard_sidecar_path`) to pytest tmp,
DELEGATING when the test pinned the root itself. Verified clean on #2141 v3.

**Why:** these guards sit on a SHARED test surface — a wrong discriminator or
missed writer silently either breaks root-pinned assertions or leaves the
pollution channel open (the #2141 incident: 6,369 pytest-planted rows at the
fixture value `free_gib: 17.0` in the real sidecar, planted because
`isolated_registry` pinned the registry dir but not `PROJECT_ROOT`).

**How to apply (each is a 1-2 min grep/read):**
1. *Single-funnel claim:* grep the resolver's call sites — the redirect covers
   every write only if the resolver has ONE caller (the append helper) and
   every producer routes through that helper. Also check dry-run: an
   early-return print BEFORE the resolve means dry-run never touches it.
2. *Equality discriminator (`asw.ROOT == real_root`):* sound because (a)
   autouse function-scoped fixtures run BEFORE explicitly-requested same-scope
   fixtures, so `real_root` is captured pre-pin; (b) the delegate calls the
   REAL resolver, which reads the module global at CALL time, so pins landing
   later (test body) still win; (c) a pin to a tmp dir INSIDE the real root
   still differs by equality → delegates correctly — containment is
   irrelevant. Check no OTHER autouse fixture patches the root first.
3. *Existing-caller interaction mode:* tests that stub the APPEND helper with
   a recorder (`monkeypatch.setattr(asw, "_append_...", lambda ...)`) bypass
   the resolver entirely — the redirect cannot affect them. Grep the test file
   for both the path literal AND the helper name to classify each caller.
4. *Module-scope predicate:* a `_watcher_modules`-style predicate keyed on
   module-level imports covers `import X as asw` AND `from X import f`
   (via `__module__`); function-body-only imports dodge it (documented
   residual, fine if disclosed).
5. *Live-sibling conftest hunks:* read the sibling worktree's ACTUAL diff —
   a plan may call it "their guard fixture" when it is a tuple-entry edit
   elsewhere in the file; what matters is region disjointness + additive
   shape + rebase-after ordering, which make any residual conflict loud.
6. *Plan-internal merge-vs-overwrite wording:* when a plan mandates a
   MERGE state save on one path and says another path is "unchanged", check
   the unchanged path's current save shape — an overwrite there can
   contradict a new test asserting the merged key survives. Fails loud at a
   red test, so Concern not Must-Fix, but name it.
