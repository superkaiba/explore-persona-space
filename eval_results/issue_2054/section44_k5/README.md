# Issue 2054: completed six-setting K5 and leave-one-setting-out analysis

The K5 workload and leave-one-setting-out continuation completed on 2026-09-12. Both models have all six requested settings and all five held-out conversation folds: 12/12 primary panels and 60/60 folds per analysis. The common complete-five cohort retains capped nonempty completions. The original task Goal and existing clean result are preserved.

Each target is the mean of five answer vectors. Context vectors retain the K3 prefill-alone last-token convention at layer 19 (block index 18), dimension 3,584. Shared maps pool the six displayed settings within each model. Leave-one-setting-out maps train on the other five settings, excluding the evaluation conversation fold everywhere; fitting, GCV selection, and identity-plus-bias use source settings only. Results below are unweighted means of the five held-out fold R² values. Retrieval accuracy, actual pool sizes, chance levels, identity baselines, fold inputs and cap sensitivities are preserved in the JSON artifacts.

| Model | Setting | Own map R² | Six-setting shared R² | Leave-one-setting-out R² |
|---|---|---:|---:|---:|
| Base | DANA | 0.543777 | 0.562691 | 0.538642 |
| Base | HELIOS | 0.529506 | 0.545378 | 0.493083 |
| Base | VEX | 0.482483 | 0.501665 | 0.467413 |
| Base | WREN | 0.529197 | 0.552781 | 0.537205 |
| Base | Plain text | 0.469413 | 0.452384 | -0.044654 |
| Base | Chat | 0.409753 | 0.390169 | -0.198484 |
| Instruct | DANA | 0.561003 | 0.576649 | 0.547155 |
| Instruct | HELIOS | 0.546843 | 0.559427 | 0.500829 |
| Instruct | VEX | 0.511803 | 0.526134 | 0.480288 |
| Instruct | WREN | 0.554057 | 0.574579 | 0.556192 |
| Instruct | Plain text | 0.454079 | 0.392736 | -0.414176 |
| Instruct | Chat | 0.675057 | 0.657034 | 0.405790 |

Instruct plain text has only 696 rows when every capped draw is excluded, so its own-map refit for that sensitivity is withheld (training folds contain 547–564 rows, below dimension 3,584). Conditioning only on the original draw stopping leaves 4,599 rows and still permits later capped draws. Historical BF16 capture discrepancies retain the recorded WARN adjudication. Internal K1/K3 references are matched to complete-five cohorts and are not an averaging-gain manuscript result.

The K5 output audit matched 4,578 files (7,940,470,172 bytes) at [the immutable K5 revision](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/9de026f872c19b2ca4fd3e4539de820e08038ee3/issue2054_section44_k5_gcp/production_v1). The LOSO audit matched 252 files (2,640,575,480 bytes) at [the immutable final audit revision](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/534f9b67838897671bb35f5fb2a73059bd0f9fa7/issue2054_section44_k5_gcp/leave_one_setting_out_v1); its control sentinel was verified separately. [Workload and monitor logs](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/f5d7e26850325061a78a0f5bac3b7ed1593042eb/issue2054_section44_k5_gcp/monitoring_v1) were also archived and verified.

The four downloaded completion/result JSON files are byte-preserving copies, with immutable URLs and SHA-256 hashes in verification.json. That file also records the checks performed during harvest, explicit coverage, paired reference equality, source-fold audits, retrieval reads and parity gates. No compute was provisioned, restarted, deleted or otherwise changed during harvest.
