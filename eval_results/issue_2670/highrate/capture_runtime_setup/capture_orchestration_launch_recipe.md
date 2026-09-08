Three orchestration artifacts are ready for independent/root review. They have not been executed, installed, staged or launched by this agent. They live outside both the frozen 12-file collection and 29-file capture source closures.

`capture_runtime_setup_supervise.sh` and `highrate_capture_supervise.sh` copy the download supervisor's entire worker ownership, process-group cleanup, traps and exit-receipt body byte for byte. The bounded diff records only mode/path/deadline, explicit environment and worker-command differences. Setup is fixed to 1,800 seconds, nice 10, CPUs 0 and 1, uv concurrency 2 downloads/1 build/1 install, and one thread per numeric library. The actual pod inspection confirmed `/usr/local/bin/uv` version 0.12.10 and the pinned base Python 3.12.14 path. Source sync to 737be6d5 was subsequently performed by root, not this agent.

`capture_runtime_setup_worker.py` uses only stdlib imports and package-manager subprocesses. It refuses existing venv/evidence, builds only `/root/.venvs/issue2670-highrate-capture-737be6d5`, installs the twelve explicit pins from the readiness runbook, runs pip check/freeze, and records distribution versions without importing torch/transformers. It creates launch-specific immutable source-bound launch, disk, package, tool-log and completion receipts. It rechecks those bytes before success. A failed install retains partial evidence and the supervisor's nonzero exit; it is not silently retried. Its completion receipt explicitly leaves capture readiness and runtime/module/GPU checks pending.

After independent review, root stages all three files unchanged under `/workspace/logs/issue2670-context-risk-highrate/setup` and compares each SHA to the manifest. The setup worker requires all three for its orchestration source receipt. The prior readiness report supplies storage/affinity evidence; root rechecks those live before launch. Never use an existing unverified venv as a successful setup resume.

For setup, assign a new launch ID and use the following detached invocation on the pod. The parent shell must fail on errors. `noclobber` protects the dedicated log, and stdin/stdout/stderr are detached from SSH. The supervisor writes its own PID and child PGID; use these receipts for monitoring and later cleanup, not a broad process name match.

```bash
export EPM_CONTEXT_RISK_HIGHRATE_SETUP=/workspace/logs/issue2670-context-risk-highrate/setup
export EPM_CONTEXT_RISK_LAUNCH_ID=runtime_YYYYMMDDTHHMMSSZ
export EPM_CONTEXT_RISK_PROCESS_TIMEOUT_SECONDS=1800
(
  set -euo pipefail
  set -o noclobber
  nohup setsid bash "$EPM_CONTEXT_RISK_HIGHRATE_SETUP/capture_runtime_setup_supervise.sh" \
    </dev/null >"$EPM_CONTEXT_RISK_HIGHRATE_SETUP/capture_runtime_setup_${EPM_CONTEXT_RISK_LAUNCH_ID}.log" 2>&1 &
  printf 'detached_launcher_pid=%s\n' "$!"
)
```

Replace the launch-ID timestamp with the actual unique launch timestamp; it is not a reusable literal. Setup can run during fresh generation because it never loads models/tokenizers or initializes CUDA. Keep the live server environment untouched. Completion requires both the worker `.receipt.json` and matching `capture_runtime_setup_<ID>_process.exit.json` with exit 0 and `cleanup=no_live_members`.

The later capture supervisor requires the reviewed source sync, separate env import/runtime checks, actual VM preparation/staging, exact prepared SHA and source review, completed fresh/token-proof work, and owned server drainage. It does not perform or bypass those gates. The worker command invokes only the unchanged `scripts.context_risk_highrate_capture` module with `mode=capture`; all capture recipe values remain in the reviewed YAML. It sets the single HF cache, CUDA13 compatibility library path, clone import paths, eight-thread caps and GPU0. No weights are loaded by the setup stage.

For capture, assign a new launch ID, explicitly set the approved positive whole-process deadline and exact returned preparation SHA, then launch:

```bash
export EPM_CONTEXT_RISK_HIGHRATE_ROOT=/workspace/logs/issue2670-context-risk-highrate
export EPM_CONTEXT_RISK_HIGHRATE_CAPTURE_REVIEW="$EPM_CONTEXT_RISK_HIGHRATE_ROOT/setup/capture_postrun_code_review.json"
export EPM_CONTEXT_RISK_LAUNCH_ID=capture_YYYYMMDDTHHMMSSZ
# Set EPM_CONTEXT_RISK_PROCESS_TIMEOUT_SECONDS to the reviewed capture fence.
# Set EPM_CONTEXT_RISK_HIGHRATE_CAPTURE_INPUT_SHA256 to the VM preparation SHA.
: "${EPM_CONTEXT_RISK_PROCESS_TIMEOUT_SECONDS:?explicit capture fence required}"
: "${EPM_CONTEXT_RISK_HIGHRATE_CAPTURE_INPUT_SHA256:?exact prepared SHA required}"
(
  set -euo pipefail
  set -o noclobber
  nohup setsid bash "$EPM_CONTEXT_RISK_HIGHRATE_ROOT/setup/highrate_capture_supervise.sh" \
    </dev/null >"$EPM_CONTEXT_RISK_HIGHRATE_ROOT/setup/highrate_capture_${EPM_CONTEXT_RISK_LAUNCH_ID}.log" 2>&1 &
  printf 'detached_launcher_pid=%s\n' "$!"
)
```

Capture success requires both the semantic capture binding and matching successful owned process exit, followed by lossless VM transfer and the real VM-side `validate_binding`. Archive the three scripts, manifest, diff, launch/exit/package/disk logs and every capture artifact. This artifact manifest is a static implementation handoff, not an independent review PASS or evidence of an installed environment/capture run.
