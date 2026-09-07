These executable critic fixtures use temporary output directories. They make no model calls, uploads, task-marker writes, server signals or pod lifecycle calls. The finishing tests replace those operations with local mocks; the supervisor test substitutes a temporary fake `uv` command.

Run from the recovery worktree with the pinned Inspect environment:

```bash
UV_NO_SYNC=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 uv run --with 'inspect-ai==0.3.261' --with 'openai==3.7.0' python eval_results/context_risk_corrected_v20_validation/software_fixtures/review_runners/audit_cases.py
UV_NO_SYNC=1 uv run --with 'inspect-ai==0.3.261' --with 'openai==3.7.0' python eval_results/context_risk_corrected_v20_validation/software_fixtures/review_runners/finish_upload.py
UV_NO_SYNC=1 uv run --with 'inspect-ai==0.3.261' --with 'openai==3.7.0' python eval_results/context_risk_corrected_v20_validation/software_fixtures/review_runners/finish_replay.py
UV_NO_SYNC=1 uv run python eval_results/context_risk_corrected_v20_validation/software_fixtures/review_runners/finish_supervisor.py
```

The audit runner requires the recorded prompt-B manifest, smoke report and native `.eval` file at their original VM paths. It reads them and mutates only in-memory copies and temporary reports. The other runners need only the repository and installed dependencies. Each runner prints its temporary evidence directory; `results/` retains the independently observed passing evidence for the reviewed source hashes. Fixture evidence is software validation, not experimental observations.
