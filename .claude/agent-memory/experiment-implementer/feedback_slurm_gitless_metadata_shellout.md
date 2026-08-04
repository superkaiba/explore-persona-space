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

**Recurrence (#1336 fellows job 17987, 2026-08-02).** The same class fired in a
bash DISPATCHER's embedded-python heredocs (`issue1336_dispatch.sh` g2_parity —
7 identical `check=True` rev-parse sites) AND in a shell `git commit+push`
results leg (`phase_upload_v2`), which no metadata resolver can save — a
rsync-lane workload cannot push at all (`pod-side-reporting.md` SLURM bullet:
HF mirror + VM-side orchestrator commit own the landing). Extended rule: the
class-sweep covers heredoc python AND shell `branch=$(git rev-parse ...)` /
commit/push legs; result-commit legs get fenced behind
`[ -e .git ] && git rev-parse --git-dir` with a loud rsync-lane skip line.
Shared resolver now lives at
`experiments/issue_1336/common.py::resolve_code_sha` (env -> check=False ->
`unknown-no-git`); pins in `tests/test_issue1336_dispatch_v2.py`. slurm.py
still exports NO `EPS_GIT_SHA` — surfaced as a workflow-fix candidate
(#1336 crash-fix round) so rsync-lane provenance can resolve the real sha.

## Merged sibling index rows (#2032 curation, 2026-08-03)

This entry is the PRIMARY index pointer for its theme; the sibling index rows below were merged into one index row to fit the agent-memory index size cap (task #2032). Each merged row is preserved verbatim — follow its pointer for the sibling lesson's own entry file.

- [Fellows shared-node GPU sizing](feedback_fellows_shared_node_gpu_sizing.md) — fellows nodes share GPUs with no isolation: width/ids from SLURM allocation env, vLLM util from mem_get_info free bytes (never fixed 0.6); #1902 job 16127

**Recurrence (#1336 fellows job 17987, 2026-08-02).** The same class fired in a
bash DISPATCHER's embedded-python heredocs (`issue1336_dispatch.sh` g2_parity —
7 identical `check=True` rev-parse sites) AND in a shell `git commit+push`
results leg (`phase_upload_v2`), which no metadata resolver can save — a
rsync-lane workload cannot push at all (`pod-side-reporting.md` SLURM bullet:
HF mirror + VM-side orchestrator commit own the landing). Extended rule: the
class-sweep covers heredoc python AND shell `branch=$(git rev-parse ...)` /
commit/push legs; result-commit legs get fenced behind
`[ -e .git ] && git rev-parse --git-dir` with a loud rsync-lane skip line.
Shared resolver now lives at
`experiments/issue_1336/common.py::resolve_code_sha` (env -> check=False ->
`unknown-no-git`); pins in `tests/test_issue1336_dispatch_v2.py`. slurm.py
still exports NO `EPS_GIT_SHA` — surfaced as a workflow-fix candidate
(#1336 crash-fix round) so rsync-lane provenance can resolve the real sha.
