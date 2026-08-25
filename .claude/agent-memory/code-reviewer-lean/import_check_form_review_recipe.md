---
name: import-check-form-review-recipe
description: Review recipe for bare --import-check entry-form commits on phase-dispatch drivers (optional --phase + post-parse guard) — 5 probes certify it
metadata:
  type: feedback
---

Certifying a "bare `--import-check` form" commit (the `.claude/rules/code-style.md`
argparse-completeness convention) takes exactly five probes, all cheap (#2329 r1 g7):

1. **Not-hollow check:** the `_import_check()` body must execute DEFERRED production
   imports the module path never reaches + `assert_args_attributes_defined(__file__)`;
   a body of only module-level re-imports always passes and is worse than none.
2. **dotenv parity:** `load_dotenv()` must sit at MODULE level before heavy imports so
   the check form and production run under identical env (lint covers module order,
   not the entry form).
3. **Misuse exit-code migration:** making a required arg optional flips the no-arg
   failure from argparse rc=2 to the guard's rc (SystemExit(str) → rc=1). Sweep callers
   for rc=2 keying AND check the new rc against the driver's REGISTERED gate rc table
   (docstring "Exit codes:" — 7/8/9/10 class); a collision silently re-labels a designed
   HALT as CLI misuse.
4. **Fails-pre-fix via parent blob:** extract `git show <sha>^:<file>` to /tmp, run bare
   `--import-check` with PYTHONPATH=scripts → expect rc=2 (see [[fails-pre-fix-probe-parent-commit]]).
5. **Run all four HEAD probes verbatim** (both bare --import-check → 0, both no-phase → 1);
   the two return-plumbing shapes differ legitimately (`_import_check(); return RC_OK`
   vs `return _import_check()`) — check each flows to `sys.exit`.

**Why:** the convention recurs per driver (mandated by code-style.md, no repo-wide lint),
so these commits are frequent; the rc-collision and caller-sweep steps are the only
non-obvious ones. No unit test is owed — the smoke-arch Axis-1 marker is the binding arm.

**How to apply:** any diff touching `add_argument("--phase", required=True)` /
adding an `--import-check` branch / adding a post-parse phase guard.
