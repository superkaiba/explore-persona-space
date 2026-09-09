# China repair v2: continuation contract

Operational handoff only; task 952 remains the canonical status record. This
does not add experiments, change the frozen panel, or declare the rerun complete.

## Completed inputs

- GPU data revision: `00406a09b599d6523e678f2bac7fabe6873c99f3`.
- Exact GPU text archive: `bbbe7e6e430910afe034caa66ee540cf01b1e1c3`.
- Frozen judge packet bank: `f29ac72e8a1616934ca8435b951c09d4fbb1a3c9`.
- Dataset: `superkaiba1/explore-persona-space-data`.
- Study prefix:
  `issue952_position_divergence/followups/china_refusal_wording_withholding_v2`.
- GPU completion receipts and audit:
  `eval_results/issue_952/china_repair_v2/gpu_completion/`.

All production generation and captures are complete and byte-verified. The
scoped H100 pod was terminated. Do not regenerate these inputs. The original
v1 experiment is preserved.

## Judging

The live collector is `/tmp/issue952-china-repair-v2/judge_production`.
Keep the actual frozen reviewers `/root/china_prod_a` and `/root/china_prod_b`:
5,951 and 6,049 assignments respectively, covering 10,880 unique responses and
1,120 overlap responses. Resume their own immutable packets; do not replace
reviewers, change the rubric, fill missing decisions, or borrow labels.

Prepared-bank and partial-checkpoint archives are backups, **not** a complete
production measurement. Collection requires all original decisions, receipts,
and matching authored fields. The final exact-byte archive must include every
source file and publish to `<study-prefix>/attempt1/judge`. Record its verified
immutable revision before CPU dispatch. No model API or CLI judging is allowed.

The reviewed `scripts.issue952_china_repair_judge_watch` module handles
incremental backups and final collection. Verified-by: ran (first real partial
archive reconstructed and source-hash checked). Its state directory is
`/tmp/issue952-china-repair-v2/judge_monitor`; the source-manifest SHA256 is
`9033b9732564b51905f427284ea660ee18bd5d38b6016065d64bb6f94892fa7c`.
Resume only when the previous monitor has exited: use the same `--source`,
`--state-dir`, `--manifest-sha256`, `--poll-seconds 60`, and explicit
`--max-polls 1440`. A nonblocking collector lock prevents duplicate owners.
The process's zero exit after its bounded poll interval does **not** establish
completion: require its hash-bound external `done.json` and verified full
archive. It never launches CPU analysis or creates semantic decisions.

## Dedicated CPU continuation

Verified-by: read. The launcher and analysis have independent code review and
synthetic tests; the production CPU run has not yet executed.

Re-enumerate task triage before dispatch. Use the shared-root
`scripts/dispatch_issue.py launch` with task 952, `--backend runpod`,
`--intent cpu-mid`, `--gpus 0`, `--min-ram-gb 16`, `--boot-disk-gb 50`,
`--lane-suffix china-repair-cpu`, the committed
`codex/issue952-china-repair-20260908` branch, `--execute-workload`, and
`--skip-default-git-paths`. Pass the workload command:

```text
bash scripts/issue952_china_repair_cpu_run.sh REVIEWED_HEAD DATA_REVISION JUDGE_REVISION
```

The three arguments must be real 40-character immutable revisions, not these
placeholders. Check the realized CPU allocation, disk, `/usr/bin/time`, exact
checkout SHA and runtime. Keep eight computation threads; never hide an
allocated GPU to bypass the CPU guard. The measured 10%-width pilot informs
runtime and memory sizing before interpreting full-width performance.

The launcher validates and stages inputs, loads the complete panel, then runs
pilot, full analysis, and verified export. Root:
`/workspace/issue952-china-repair-v2-cpu`. Preserve per-phase timing/logs, full
checkpoints, `analysis/full/done.json`, and the external `export.json` receipt.
An output file or PID alone is not a successful-run sentinel.

## Finish

Run the reviewed aggregate-figure script on the verified full report; inspect
color and grayscale outputs and publish browser-accessible links. Independently
audit all reported numbers, uncertainty, denominators and limitations. Do not
gate completion on effect size, refusal prevalence, or judge agreement. Preserve
all evidence and verify the complete CPU out-root inventory before scoped
teardown. Fold the follow-up into task 952 without changing its original goal
or prior results, then return it to `awaiting_promotion`. No paper edits or
automatic promotion are part of this rerun.
