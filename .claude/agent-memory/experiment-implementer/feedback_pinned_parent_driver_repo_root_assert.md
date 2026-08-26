---
name: pinned-parent-driver-repo-root-assert
description: Read-pinning a phase driver that asserts pyproject at import needs the <tmp>/scripts + stub-pyproject layout, REPO_ROOT repoint, and a pinned-core F rebind when branch-only kwargs exist
metadata:
  type: feedback
---

Read-pinning an unmerged-branch phase driver (`git show <PIN>:scripts/issueN_x.py`
-> importlib-by-path, the issue2474_n1m_map pattern) breaks at exec when the
driver's module top runs `_ensure_repo_root_on_syspath()` (asserts
`parents[1]/pyproject.toml`). Fix: materialize under `<tmp>/scripts/` beside a
stub `<tmp>/pyproject.toml`, then rebind `P.REPO_ROOT = <real root>` post-exec.
When the driver calls shared-core kwargs that exist ONLY on its branch (e.g.
#2388's `ridge_gcv_predict_per_target(dof_cap=, selector_telemetry=)` — branch
commit `1f44fbb8a6`, absent on main), ALSO read-pin the branch's core module
under a PRIVATE name and rebind the driver's import alias (`P.F = pinned_core`)
— never shadow the installed package name in `sys.modules`. Drift-check every
main-resident module the pinned code still binds to (store_io/arms/constants
were byte-identical or name-compatible for #2388).

**Why:** #2388 n1m round (2026-08-24): the parent's dof-capped GCV core was
branch-only; loading the parent against main's core would TypeError at the
first fit, and the module-top assert crashed the naive tmpdir load.

**How to apply:** any round that imports a parent driver at a pinned SHA —
verify required kwargs exist via `inspect.signature` right after the load
(fail loud on wrong pin). Worked example: `scripts/issue2388_n1m_map.py::load_pinned_modules`.
