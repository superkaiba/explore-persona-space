# Vendored inputs — issue #2333 (snowball test)

## fu1_conf1_confirmation.json

- Vendored from: `origin/issue-2094` @ commit `931a3df5dc5bc4b06d53e396030ba197977139e3`
  (path on that branch: `eval_results/issue_2094/f_metrics/fu1_conf1_confirmation.json`)
- Git blob: `2d6699eb9d62908a3448ce6870b055d1993dceb9`
- sha256: `c060c2acb6835a33d3e6bda156840108eb15f9425264498d40846a2323d35419`
- Purpose: the #2094 fu1 conf1 confirmatory aggregate. `scripts/issue2333_analysis.py
  --phase s2-ce-derive` cross-checks its re-derived S2 ce banked effect against this
  aggregate (family `matched_query|ce|joint_all|replace|A|f_beh_prefix`: steered mean
  0.512, null mean 0.0969, n_pairs 10) with a fail-loud +/-0.01 tolerance (plan section 5).
- Vendoring reason: the parent issue-2094 follow-up branch is unmerged; per
  `.claude/rules/artifact-reuse.md` the recipe is ported by pinned vendoring, never a
  live branch read.
