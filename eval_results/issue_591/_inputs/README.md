# #591 e1 input snapshots (provenance)

| file | producer | source path |
|---|---|---|
| join_sycophancy.json | #411→#470→#480 freeze | eval_results/issue_480/_inputs/predictor_comparison.json (git, main) |
| join_refusal.json | #518 | eval_results/issue_518/refusal/_inputs/predictor_comparison.json (git, main) |
| join_em.json | #518 | eval_results/issue_518/em/_inputs/predictor_comparison.json (git, main) |
| issue411_analyze_summary.json | #411 | origin/issue-411:eval_results/issue_411/analyze_summary.json |
| issue411_base_panel_rates.json | #411 | origin/issue-411:eval_results/issue_411/base_panel_rates.json |
| neg_membership_411.json | derived | Hub training pools (system-prompt match), sha256 per pool inside |
| neg_membership_518.json | derived | deterministic draw @ 4b150926 (MEDIUM confidence) |
| sex2.csv | vendored | firthlogist repo (Firth validation dataset) |
| firth_sex2_validation.json | i591_firth.py --validate | this repo |

Snapshot commit: 187a10016afeb347aa5911361d4311c9a5e170bb at 2026-06-11T11:05:10.232243+00:00
