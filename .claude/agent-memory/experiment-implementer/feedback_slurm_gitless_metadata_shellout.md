# SLURM/fellows scratch tree is git-less — metadata `git rev-parse` shellouts crash the run

**Trap (#1902 job 16142, 2026-07-31).** The fellows/SLURM lane's
`materialize_branch_src` rsync copy (`/workspace/superkaiba/eps/issue-<N>`) has NO
`.git` directory. Any reproducibility-metadata helper that shells
`git rev-parse HEAD` with `check=True` crashes the workload rc=128
("fatal: not a git repository") — in #1902 AFTER the P1 pilot's capture work had
all succeeded, at the first `leg_report.json` write, burning a full launch cycle.
The no-git fact was already documented for the RESULT-PUSH side
(`pod-side-reporting.md` § Result-push, SLURM lane bullet; the same run's
`fits._commit_eval_results` correctly probes + skips) — the METADATA helper was the
missed sibling call site.

**Rule.** Every `_git_sha()`/provenance helper in per-issue scripts degrades, never
raises: (1) `os.environ.get("EPS_GIT_SHA")` wins when set; (2) subprocess with
`check=False`; (3) literal `"unavailable-no-git-checkout"` on rc!=0. The canonical
sha rides the launch marker + handle sidecar, so nothing is lost. When porting any
driver to a SLURM lane, class-sweep EVERY git-subprocess site in the run's scripts
(`grep -n 'rev-parse\|git' scripts/issue<N>_*.py`) — a tolerant commit path beside a
strict metadata path is exactly the #1902 shape.

**Worked fix:** `scripts/issue1902_run.py::_git_sha` + `issue1902_corpus.py::_git_sha`
(commit `5a3d11f7e2`, issue-1902 branch); pin test
`tests/test_issue1902_run.py::test_git_sha_degrades_on_gitless_lane`.
