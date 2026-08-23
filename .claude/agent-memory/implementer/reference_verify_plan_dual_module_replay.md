---
name: verify_plan dual-module corpus replay recipe
description: Old-vs-new verify_plan calibration replays — sys.modules registration before exec_module (dataclass crash), monkeypatch regex constants for per-flip attribution
type: reference
---

For verify_plan.py retro-calibration tasks (the #1262/#1264/#1276/#1291 corpus-replay convention), the dual-module replay driver has two non-obvious mechanics:

1. `importlib.util.spec_from_file_location` + `module_from_spec` MUST register `sys.modules[name] = mod` BEFORE `spec.loader.exec_module(mod)` — verify_plan.py's `@dataclass` decorator resolves `cls.__module__` via `sys.modules` at class-creation time and crashes with `AttributeError: 'NoneType' object has no attribute '__dict__'` otherwise (#1291).
2. Per-flip noise-class attribution: monkeypatch the NEW module's regex constants (e.g. `_FAILLOUD_RISKS_HEAD_RE = re.compile(r"(?!x)x")` never-match, restore in `finally`) and re-call the real helper with one exclusion disabled at a time — exercises the genuine code path, no reimplementation.

**How to apply:** kind resolution = `^kind:` regex on each task's body.md head (default experiment); enumerate `tasks/*/*/plans/v*.md` from the MAIN repo root (the corpus is not in sparse worktrees); ~1,900 plans × 2 modules × 2 checks runs in ~1 min. Working driver preserved in #1291's epm:results marker.

Three more mechanics for FULL-status (all-check) two-subprocess replays (#1306):

3. SNAPSHOT every plan TEXT at manifest time — the live `tasks/` tree churns (concurrent sessions `git mv` task folders mid-replay), so per-pass file reads produce spurious `<absent>`/FileNotFoundError diff rows; both passes must evaluate byte-identical snapshotted inputs.
4. The OLD pass's `PYTHONPATH` must ALSO include a real `scripts/` dir — verify_plan LAZY-imports sibling `workflow_lint` (c34 caps), which a bare `/tmp/vp_old` dir lacks (`ModuleNotFoundError` mid-corpus, only on plans that trigger the check).
5. Pin `verify_plan._C34_REPO_ROOT = Path(<repo/worktree root>)` in BOTH workers — c34 stats live workflow files via `Path(__file__).parent.parent`, so a /tmp-resident module SKIPs c34 spuriously (the constant's comment says "tests monkeypatch"); unequalized, it drowns the diff in fake SKIP→WARN/PASS rows.
