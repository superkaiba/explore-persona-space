## Completion Report

**Task:** #793 — make the SLURM lane HONOR `spec.extra["repo_branch"]` by materializing a complete rsync source for the requested feature branch (Option C), instead of only refusing stale `main` at the #653 guard.
**Status:** SUCCESS

### (a) What was done

- `src/explore_persona_space/backends/slurm.py`:
  - **§4a** — added module constant `WORKING_TREE_OVERLAY_PATHS = ("external/open-instruct",)` and module-level `materialize_branch_src(*, src_root, branch, issue, overlay_paths=WORKING_TREE_OVERLAY_PATHS, timeout=300) -> Path`, placed right after `_default_src_root` (steps 1-6 per plan: issue-namespaced scratch `~/.eps-slurm-src/issue-<N>` via `EPS_SLURM_SRC_ROOT`; remove-prior + `worktree prune`; `git rev-parse --verify` the branch commit with an `origin/<branch>` fallback, `RuntimeError` if neither resolves; `git worktree add --detach <scratch> <commit>`; overlay each working-tree-only gitlink via `rsync -a --delete` with an absent-path WARNING-and-skip; `pyproject.toml` source-sanity assert). Committed-only-tree tighter-than-working-tree-rsync note included inline in the docstring.
  - **§4b** — added the keyword-only `git_cloner=None` seam to `SlurmBackend.__init__`, `self._git_cloner = git_cloner or materialize_branch_src`, mirroring the existing `git_branch_resolver` seam + comment.
  - **§4c** — rewrote the top of `prepare()`: `rsync_src = self._resolve_rsync_source(spec)` → `self._assert_repo_branch_synced(spec, src_root=rsync_src)` → rsync from `rsync_src`. Added the `_resolve_rsync_source` private helper (no-op → `self._src_root` on absent/`main`/already-on-branch; else the cloner's path). Included the verbatim inline comment explaining the post-fix belt-and-suspenders role of the re-asserted guard.
  - **§4d** — extended `_assert_repo_branch_synced(self, spec, src_root: Path | None = None)`, `src_root = src_root or self._src_root`, body reads the probe + error message off the resolved `src_root`. Docstring updated to the belt-and-suspenders framing; internal raise/no-op semantics verbatim.
  - `__all__` gains `WORKING_TREE_OVERLAY_PATHS` + `materialize_branch_src`; added `import shutil` (used for the scratch `rmtree` cleanup).
- `tests/test_slurm_backend_render.py`:
  - **§5b renames/rewrites** — `test_slurm_prepare_refuses_feature_branch_on_stale_main_source` → `test_slurm_prepare_honors_feature_branch_on_stale_main_source_via_cloner` (cloner invoked once, rsync sources from the scratch path, no `ValueError`, `len(rsynced)==1`); `test_slurm_prepare_refuses_when_source_branch_unprovable` → `test_slurm_prepare_raises_when_cloner_cannot_resolve_branch` (resolver `None` + raising cloner → `RuntimeError` surfaces from `prepare()`, `rsynced == []`).
  - **§5a 8 new tests** — `test_prepare_honors_repo_branch_via_git_cloner`; `test_prepare_repo_branch_main_is_byte_identical_no_cloner` (parametrized absent/main); `test_prepare_install_already_on_branch_skips_cloner`; `test_resolve_rsync_source_returns_src_root_on_main` (parametrized incl. the None-resolver Statistics-critic case); `test_materialize_branch_src_worktree_add_and_overlay` (real git, `tmp_path`, `skipif` git-absent, idempotency second call); `test_materialize_branch_src_unresolvable_branch_raises`; `test_materialize_branch_src_absent_overlay_path_warns_not_crashes` (caplog WARNING); `test_assert_repo_branch_synced_accepts_explicit_src_root`. Added shared `_branch_spec` / `_git_available` / `_init_branch_repo` helpers + the `materialize_branch_src` import.
- Diff: **+622 / -62 across 2 files** (`git diff --stat HEAD~1 HEAD`: `slurm.py +303`, `test_slurm_backend_render.py +381`).
- Plan adherence: §4a DONE · §4b DONE · §4c DONE · §4d DONE · §5a (8 tests) DONE · §5b (2 renames) DONE · §5c (byte-identical suites) DONE (verified pass) · §5d (documented, not run — no cluster in scope) as specified · §11 scratch-dir-outside-reap-glob verification DONE.
- Commit hash: `51cad32101`. Branch `issue-793`, pushed to `origin issue-793`.

### (b) Considered but not done

- **`git worktree add --detach <commit>` vs `git clone` / `worktree add <branch>`** — used `--detach <commit>` per plan §11: shares the repo-root object DB (no history copy) and avoids the "branch already checked out in the /issue worktree" conflict. `git clone` (full history copy) and `worktree add <branch>` (checkout conflict) rejected.
- **`shutil.rmtree` for the belt-and-suspenders scratch removal** rather than a `subprocess rm -rf` — pure-Python, no shell, `ignore_errors=True` inside `contextlib.suppress`; the git `worktree remove --force` + `worktree prune` calls remain the primary cleanup. All three cleanup calls are `check=False`/suppressed so a fresh scratch (no registered worktree) does not fail the prepare; only the create/overlay/assert path fails loud.
- **No `dispatch_issue.py` change** — `spec.extra["repo_branch"]` is already threaded (plan §2/§11); the fix reads it and materializes the source itself. Out of scope per the plan's "Scope — explicitly OUT".
- **Did not weaken/remove `_assert_repo_branch_synced`** — kept + re-asserted against the resolved source per plan (belt-and-suspenders on the cloner, not refuse-on-mismatch).

### (c) How to verify

- **Tests run** (all PASS): the 8 new + 2 renamed tests plus the 4 grandfathered guard tests, verbose:
  - `test_prepare_honors_repo_branch_via_git_cloner`, `test_prepare_repo_branch_main_is_byte_identical_no_cloner[absent|main]`, `test_prepare_install_already_on_branch_skips_cloner`, `test_resolve_rsync_source_returns_src_root_on_main`, `test_materialize_branch_src_worktree_add_and_overlay`, `test_materialize_branch_src_unresolvable_branch_raises`, `test_materialize_branch_src_absent_overlay_path_warns_not_crashes`, `test_assert_repo_branch_synced_accepts_explicit_src_root`, `test_slurm_prepare_honors_feature_branch_on_stale_main_source_via_cloner`, `test_slurm_prepare_raises_when_cloner_cannot_resolve_branch`.
  - Full file: **90 passed, exit 0** (`test_slurm_backend_render.py`).
  - Byte-identical downstream suites: **314 passed** across `test_router.py` + `test_router_acceptance.py` + `test_issue_dispatch.py` + `test_backend_selector.py`.
- **Regression test for the fix** (the behavioral change from refuse → honor): `tests/test_slurm_backend_render.py::test_prepare_honors_repo_branch_via_git_cloner` — a non-`main` `repo_branch` on a `main` install invokes the cloner and rsyncs from its scratch path with no `ValueError`. The real-git path is covered by `test_materialize_branch_src_worktree_add_and_overlay` (asserts branch-marker + overlaid gitlink + `pyproject.toml` present, idempotent on re-run) and the fail-loud fallback by `test_materialize_branch_src_unresolvable_branch_raises`. These fail against the pre-fix code (`prepare` raised `ValueError`; `materialize_branch_src` did not exist) and pass post-fix.
- **Lint:** `uv run ruff check <both files> && uv run ruff format --check <both files>` — PASS (`All checks passed!`, `2 files already formatted`).
- **Reproduction commands** (worktree src forced onto the PYTHONPATH so the edits are exercised; editable install otherwise resolves main's src):
  ```
  cd /home/thomasjiralerspong/explore-persona-space
  uv run ruff check .claude/worktrees/issue-793/src/explore_persona_space/backends/slurm.py .claude/worktrees/issue-793/tests/test_slurm_backend_render.py
  uv run ruff format --check .claude/worktrees/issue-793/src/explore_persona_space/backends/slurm.py .claude/worktrees/issue-793/tests/test_slurm_backend_render.py
  PYTHONPATH=.claude/worktrees/issue-793/src .venv/bin/python -m pytest .claude/worktrees/issue-793/tests/test_slurm_backend_render.py -v -p no:cacheprovider
  PYTHONPATH=.claude/worktrees/issue-793/src .venv/bin/python -m pytest .claude/worktrees/issue-793/tests/test_router.py .claude/worktrees/issue-793/tests/test_router_acceptance.py .claude/worktrees/issue-793/tests/test_issue_dispatch.py .claude/worktrees/issue-793/tests/test_backend_selector.py -q -p no:cacheprovider
  ```
- **What success looks like:** `90 passed` on the SLURM render file and `314 passed` on the downstream suites, with the new branch-honoring test set green (cloner invoked, rsync sources the scratch tree, main/absent path unchanged).

### §11 verification (implementer note)

`~/.eps-slurm-src/` is OUTSIDE the stale-worktree-audit reap glob. `scripts/worktree_audit.py` scopes its sweep strictly to `wt_dir = root / ".claude" / "worktrees"` (line 751; `wt_root_rel = ".claude/worktrees/"` line 750) and only iterates `wt_dir.iterdir()` (line 785); it returns early if `wt_dir` is not a dir. The scratch root lives in `$HOME` (or `EPS_SLURM_SRC_ROOT`), never under `<repo>/.claude/worktrees/`, so the sweep never sees it. `cron_worktree_audit.sh` just invokes that same script — no independent glob. Cleanup of the scratch relies on the per-`prepare` remove+recreate (idempotent refresh) + the VM disk-guard's terminal-cache reap, as the plan §5e/§11 states.

### (d) Needs human eyeball

- **Live-cluster acceptance is not exercised** (plan §5d — out of scope for a 0-GPU-h workflow fix; needs Duo MFA + the robot key). The structural path is asserted by tests 1 + 5 (cloner invoked, rsync sources its output; real git worktree-add + overlay produces a content-complete tree), but a `--backend nibi/fir/mila --repo-branch issue-<N>` end-to-end submit past `prepare()` on a real cluster has not been run. Worth a follow-up smoke when a cluster session is available.
- Otherwise confidence high across the diff — deterministic git + FS plumbing behind the existing `SlurmBackend` seam pattern, no auth/secrets/external-API/model-call surface touched.
