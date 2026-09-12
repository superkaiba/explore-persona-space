# J/R workspace experiment preparation

**The real-model experiment has not run.** This package contains the frozen
analysis specification, artifact audit, selected prompt IDs, and tested core
analysis code. Repository task creation and experimental launch are awaiting
the explicit authorization required by the supplied AGENTS.md.

- [Analysis plan](analysis_plan.md) and [configuration](../../configs/analysis/workspace_jr.yaml).
- [Model and mapping provenance](mapping_provenance.md), with [machine-readable evidence](mapping_provenance.json).
- [Lens methods and release audit](source_audit.md), with [actual released metadata](released_lens_metadata.json).
- [Frozen context selection](selected_contexts.json), generated without reading experimental outcomes.
- [Implementation validation](validation.json), [test output](validation_output.txt), and [execution status](execution_status.json).
- [Independent lens-code review](lens_code_review.md) and [plan/provenance review](plan_review.md).

The candidate pair is Qwen3.5-27B versus Qwen3.5-4B with thinking disabled,
selected using measured same-mode GPQA performance among verified mapping
artifacts. Existing map training used K=1. Generated token IDs and per-token
activations were not retained. Thirteen unique source prompts overlap the
inherited validation and test manifests. Released J/R artifacts also lack
exact checkpoint and calibration-document provenance. These findings require
fresh captures, matched lens construction and explicit content exclusions.

Outcome-blind source selection retained 128 calibration contexts; pilot
train/validation/test counts 64/16/32; main counts 8192/373/768. All seven
selected subsets have disjoint NFC-normalized prompt hashes. The selector
removed 82 duplicate source rows across and within splits. These are planned
contexts, not completed generations; realized coverage remains unknown.

Implemented code covers the cited nonnegative gradient-pursuit algorithm,
geometry-preserving dictionary rotations, local R rules, a scoped dense-Qwen
adapter, ordinary-J numerical checks, equal-rollout pooling, variance/noise
statistics, shared-factorization affine ridge, paired contextual bootstrap
contrasts, and variance-matched direction selection. Independent review found
and resolved a bf16 RMSNorm backward rounding defect. The validation suite
passes 35 tests; it includes native tiny Qwen2 modules, not pretrained Qwen3.5.

Still needed after task registration: native Qwen3.5 runtime/recapture parity,
matched lens fitting and stability/readout checks, vLLM generation with exact
token retention, streaming token decomposition and aggregation, MLP/controls
orchestration, actual pilot/main execution, machine-readable experimental
predictions/results, plots, and the scientific report. No component R2,
workspace gap, capability interaction, or reasoning conclusion is asserted.

To reproduce local implementation validation using the existing environment:

```bash
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8 MALLOC_ARENA_MAX=2 uv run pytest tests/test_workspace_lenses.py tests/test_workspace_components.py -q
```

The selection command accepts the saved audit/config and writes a new output
file (it refuses overwriting a frozen manifest):

```bash
uv run python scripts/workspace_jr_select_contexts.py \
  --audit-file docs/exploratory_workspace_jr/mapping_provenance.json \
  --config configs/analysis/workspace_jr.yaml \
  --out /tmp/workspace-jr-selected-contexts-reproduction.json
```

It verifies source bytes against the audit and needs the audited source files
at their saved local paths. Source Hub revisions and file hashes are included
in the provenance JSON for restaging. The shared root remains on main; all
preparation changes live in the dedicated codex branch/worktree.
