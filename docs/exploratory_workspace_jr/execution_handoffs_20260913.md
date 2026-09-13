# Main execution handoffs — 2026-09-13

At 04:36 UTC, six separately supervised queues had verified and uploaded their
initial configuration and were waiting on the six running training
decompositions. No main predictability fits or conclusions were available.

| Worker | Current training decomposition | Subsequent queue steps |
|---|---|---|
| native | Primary, native dictionary | Fit k=10, 5, 25 |
| r0 | Primary, rotation 20260913 | Fit k=10, 5, 25 |
| comparison | Primary, rotation 20260914 | Fit k=10, 5, 25 |
| r3 | Primary, rotation 20260915 | Fit k=10, 5, 25 |
| r1 | Comparison, rotation 20260913 | Validation/test; all-k fits; rotation 20260914 training/validation/test and all-k fits |
| r2 | Comparison, native dictionary | Test; all-k fits; rotation 20260915 training/validation/test and all-k fits |

Every decomposition uses the registered nested k=5, 10, 25 procedure. These
queues add no observations, tuning choices or compute instances. They cover
nine remaining subset decompositions and eight fit batches, producing the 24
observed cells. Affine-null cells, diagnostics, learning curves and final
comparisons are separate subsequent work.

A queue binds its predecessor's exact systemd invocation, PID, launch attempt,
shell exit, scientific terminal and final whole-root upload receipt. It requires
that original invocation to end and rejects replacements and failed exits.
The existing fitter then verifies every input byte and all three splits.
Primary and comparison numerical producers remain pinned to `f8b4983851fba9e176640f75ad0ad7f667be46ab`
and `244f89eb8347361484413e803565f1e639255a8c`.

Failure cleanup validates the queue's own service cgroup and supervising
ancestors, then terminates its other members across nested cgroups and process
groups. Signals use pidfds with start-time and membership checks. The original
producer occupies a separate cgroup and is not a cleanup target.

**Queue failure metadata does not prove the child scientific root was fully
uploaded.** Cleanup may interrupt that child's failure uploader. Retain its
persistent disk, reconcile the complete scientific output and verify its full
upload before resource release. Failed phases require explicit recovery.

Independent review closed a replacement-sentinel race and nested process-group
cleanup flaw before dispatch; 28 focused checks passed. A real CPU-only test
covered an orphan in a separate process group, a TERM-ignoring worker in a
nested cgroup and an unaffected sibling service. Its
[seven-file evidence](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/df88180917a1adcb9e96b15afca79c34a822c0cb/exploratory_workspace_jr/20260912/sequence_cgroup_smoke_v1)
was independently verified; it is operational validation, not experiment data.

The initial uploads contain the exact plan, supervisor, launcher, cgroup helper,
uploader and recorded process ownership:

- [native](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/10f1db6416acfcc16ab8738d91b092fc0681da86/exploratory_workspace_jr/20260912/worker_sequence_native_v1)
- [r0](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/a88d9cbb4f1ebe43c0f63fcde6a037fa7e4563ad/exploratory_workspace_jr/20260912/worker_sequence_r0_v1)
- [comparison](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/0840ad0ee336f566c37fdc48e16825345982c898/exploratory_workspace_jr/20260912/worker_sequence_comparison_v1)
- [r3](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/81b9ecaab56e40a9e87f79911cacbb2a15966b6a/exploratory_workspace_jr/20260912/worker_sequence_r3_v1)
- [r1](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/f83de62d6df0c728886afda10a6b9843eed96826/exploratory_workspace_jr/20260912/worker_sequence_r1_v1)
- [r2](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/30f752c279cf919fec46f49bf4d4d83e579e528b/exploratory_workspace_jr/20260912/worker_sequence_r2_v1)

Separate downstream analysis checkouts on native and r2 use
`db7a42b9420d16762ae1219c105847ee9934e9b0` at
`/workspace/workspace-jr-analysis-db7a42b`, with the matching native runtime.
Primary comparisons and supplementary baseline/retrieval scores retain the
[joint completed-rollout cohort](completion_scoring_20260913.md).
