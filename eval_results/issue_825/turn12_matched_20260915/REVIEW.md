# Prelaunch validation

Independent fitting/protocol review: PASS (turn12_fit_review).
Independent monitor review: PASS (turn12_monitor_review); final delta also PASS: bootstrap alarms retained as nested evidence, with the same persistent 20-minute bound.

Focused fitting and monitoring suite: 30 passed. Run with PYTHONPATH=src and the existing shared uv environment, capped to eight BLAS threads. Ruff and git diff --check pass for changed files.

Repository-wide workflow_lint reports 22 errors, all in unchanged files (issue1434, issue1901, issue1902, issue2254, issue2546, section45 and the pre-existing plot_issue825_turn_matched entrypoint). The shared-VM thread-cap suite reports one pre-existing scan failure listing those unchanged entrypoints; 38 other tests passed. None of the new or modified fitting/monitor/archive files appears in these failures. No broad unrelated fixes performed.
