---
name: branch-guard-blocks-subprocess
description: task_workflow.repo_root() resolves via git rev-parse from the module path and refuses non-main worktrees; CLI integration tests via subprocess cannot redirect this. Use the fake_repo monkeypatch fixture for testing CLI behavior.
metadata:
  type: feedback
---

`src/explore_persona_space/task_workflow.py:repo_root()` runs `git
rev-parse --git-common-dir` from `Path(__file__).parent` (the module's
own location, NOT `os.getcwd()`) and explicitly refuses to resolve when
the main worktree's HEAD is not on `main`. This is a deliberate safety
guard against worktree-staleness data-loss incidents.

**Why this bites tests:** When testing `scripts/task.py` CLI behavior
via `subprocess.run([sys.executable, "scripts/task.py", ...], cwd=fake_repo)`,
the subprocess inherits the parent's `task_workflow` module location —
i.e. the REAL repo's `src/`. The cwd argument doesn't redirect the
git-rev-parse-from-module-dir call. AND the branch-guard rejects the
non-main worktree. Net result: the CLI subprocess test either operates
on the WRONG (real) repo or hits the branch-guard.

**How to apply:** For testing task.py CLI behavior, use the library-level
`fake_repo` fixture in `tests/test_task_workflow.py` which monkeypatches
`tw.repo_root` / `tw.tasks_dir` / `tw.registry_path` to the tmp_path.
The library functions (`raise_concern`, `address_concern`, etc.) are
1:1 with the CLI handlers; library-level coverage exercises the same
code paths.

Skipped tests should carry a clear `@pytest.mark.skip(reason=...)` block
explaining why CLI integration was deferred + how the same behavior is
covered at the library layer. Manual CLI smoke-test in the commit
message stands in.

Related: [[hook-strips-imports]] for another pitfall when editing
task.py + task_workflow.py together.
