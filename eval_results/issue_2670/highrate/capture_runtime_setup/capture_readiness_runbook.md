Capture readiness is **PREPARATION REQUIRED**, not a launch PASS. This read-only audit is bound to commit `737be6d5d5e318123355bdaacc0e74b96c281627` and the current independent 29-source capture review. No pod code, environment, process or workload was changed.

The pod clone is `/workspace/explore-persona-space`, currently clean at `dac9f4c4f81daa6bc975ceef43faed5910968e8e`. Seven reviewed capture source paths differ or are missing. Root must later sync the reviewed revision and verify every source hash, including the tests and v8 plan; current collection/server source bytes remain frozen. Do not use the old capture wrapper.

The live server belongs to supervisor 2528, worker process group 2533 and launch `serve_20260907T232817Z`; API Python PID 2746 uses `/root/.cache/uv/builds-v0/.tmpzIdNT3/bin/python`. Its exit receipt is absent. The server currently occupies about 126.5 GiB of HBM. Its temporary Python environment may disappear when the invocation exits and is not a durable capture runtime. The repository `.venv` is Python 3.11 with Transformers 4.57.6/Torch 2.8.0; do not use it for capture. The live server environment has the correct Transformers/Torch/NumPy versions but lacks accelerate, Hydra, OmegaConf, Inspect, SciPy and scikit-learn, has OpenAI 3.8.0 rather than 3.7.0, and cannot see the repository package without an explicit source path.

The pinned public snapshot is `/workspace/.cache/huggingface/hub/models--Qwen--Qwen3.8-27B/snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`. Download receipts record all 32 files, 55,586,114,863 bytes and exit 0 after full public-content hash verification. This audit rechecked all file sizes and non-weight content hashes; it did not reread 55.6 GB of weights. Preserve this single cache using `HF_HOME=/workspace/.cache/huggingface`.

A separate environment can be prepared while fresh generation runs. This only installs packages and does not allocate model HBM. Use the local overlay, not `/workspace` MooseFS, and a unique persistent environment. The overlay currently has 83.7 GiB available; incremental installation size has not been measured. Existing wheel caches should permit substantial reuse. The installed uv is 0.12.10; its binary contains the three concurrency controls below and its help confirms the Python/link-mode options. CPU affinity 0 and 1 is currently allowed. Root should use an owned detached setup process with a 30-minute fence, fresh log/PID/exit receipt, before/after disk measurements and low scheduling priority. Recheck affinity before launch. The commands below are a recommendation, not an executed dependency resolution:

```bash
capture_venv=/root/.venvs/issue2670-highrate-capture-737be6d5
capture_python=/root/.local/share/uv/python/cpython-3.12-linux-x86_64-gnu/bin/python3.12
export UV_CACHE_DIR=/root/.cache/uv
export UV_CONCURRENT_DOWNLOADS=2 UV_CONCURRENT_BUILDS=1 UV_CONCURRENT_INSTALLS=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
# Refuse an existing unverified environment; preserve any failed setup for diagnosis.
test ! -e "$capture_venv"
nice -n 10 taskset -c 0,1 uv venv --python "$capture_python" "$capture_venv"
nice -n 10 taskset -c 0,1 uv pip install --python "$capture_venv/bin/python" --link-mode hardlink \
  vllm==0.28.0 transformers==5.15.0 torch==2.13.0 numpy==2.3.5 \
  accelerate==1.13.0 hydra-core==1.3.2 omegaconf==2.3.0 \
  inspect-ai==0.3.261 openai==3.7.0 scipy==1.17.1 scikit-learn==1.8.0 \
  python-dotenv==1.2.3
uv pip check --python "$capture_venv/bin/python"
uv pip freeze --python "$capture_venv/bin/python"
```

Keeping vLLM in this separate environment preserves the validated runtime dependency family; the capture itself uses Transformers. SciPy and scikit-learn are required by the imported census/probe helper chain even though no fitting runs on the pod. Hashed test files do not require pytest at capture runtime. Pin resolution, all imports, optional vision dependencies and actual module/distribution versions still need an executed setup/import receipt. Do not invoke a model, tokenizer or capture while fresh generation is live.

After the complete fresh phase has its actual native audit and owned terminal process proof, run preparation on the VM from the reviewed worktree:

```bash
export EPM_CONTEXT_RISK_HIGHRATE_ROOT=/home/thomasjiralerspong/explore-persona-space/eval_results/context_risk/impossible_highrate
export EPM_CONTEXT_RISK_HIGHRATE_CAPTURE_REVIEW="$EPM_CONTEXT_RISK_HIGHRATE_ROOT/setup/capture_postrun_code_review.json"
UV_NO_SYNC=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8 \
  uv run --with inspect-ai==0.3.261 --with openai==3.7.0 \
  python -m scripts.context_risk_highrate_capture mode=prepare
```

Preserve the preparation output. Copy exactly its `stage_relative_paths`, retaining their relative layout under `/workspace/logs/issue2670-context-risk-highrate`, and separately copy `setup/capture_postrun_code_review.json`. The base files are `manifests/fresh_B.jsonl`, `selection.json`, `fresh_B/run_result.json`, `fresh_B/prefix_tokens.json`, `fresh_B/terminal_process.json` and `capture_inputs.json`. Derived evidence additionally requires the returned `setup/postrun_code_review.json` and screen/fresh `postrun_audit.json` paths. Do not reconstruct the portable binding or replace a failed original collector receipt with the derived audit. Use the exact returned `capture_inputs_sha256` as the launch input binding. The pod capture path verifies portable bytes; VM native paths and PID checks belong to preparation/downstream validation on the VM.

Only after all fresh generation and required deterministic capacity tokenization are complete should root drain the owned vLLM server through its managed lifecycle, verify its worker group has no live GPU owners, and obtain current HBM/RAM/headroom proof. Current H200 total HBM is 139.800720 GiB, cgroup RAM is 233.762 GiB and CPU quota is 10.2 CPUs, satisfying the documentary floors of 139 GiB HBM and 224 GiB RAM. This is not a free-memory or later availability guarantee. `/workspace` df reports shared capacity, not the pod's storage quota; require a quota-aware preflight before writing new capture inputs/output.

After source sync and staging, the exact capture worker command is:

```bash
cd /workspace/explore-persona-space
export HF_HOME=/workspace/.cache/huggingface
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:
export PYTHONPATH=/workspace/explore-persona-space/src:/workspace/explore-persona-space
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8
export EPM_CONTEXT_RISK_HIGHRATE_ROOT=/workspace/logs/issue2670-context-risk-highrate
export EPM_CONTEXT_RISK_HIGHRATE_CAPTURE_REVIEW="$EPM_CONTEXT_RISK_HIGHRATE_ROOT/setup/capture_postrun_code_review.json"
# Set EPM_CONTEXT_RISK_HIGHRATE_CAPTURE_INPUT_SHA256 to the exact VM preparation result.
: "${EPM_CONTEXT_RISK_HIGHRATE_CAPTURE_INPUT_SHA256:?exact prepared receipt SHA required}"
/root/.venvs/issue2670-highrate-capture-737be6d5/bin/python \
  -m scripts.context_risk_highrate_capture mode=capture
```

The installed CUDA compatibility package is `cuda-compat-13-0=580.178.04-1ubuntu1`; its libcuda resolves to `/usr/local/cuda-13.0/compat/libcuda.so.580.178.04`. Retain that library search path for CUDA 13.0 on host driver 570.211.01. Before the worker, execute import/source-hash checks and `capture._runtime()` without model loading: expected Transformers 5.15.0, Torch distribution 2.13.0/module 2.13.0+cu130, CUDA 13.0, accelerate 1.13.0, NumPy 2.3.5. `capture.imported_source_hashes()` must match the independent review, and package files must resolve under this clone. Retain the complete package receipt. No such capture environment has yet been built or validated.

Root must launch the capture worker under an owned detached supervisor with a whole-process timeout and fresh PID/log/exit proof. `context_risk_highrate_supervise.sh` only accepts `screen_pilot`, `screen` and `fresh`; it is not a capture supervisor. The existing capture wrapper emits an immutable launch binding and semantic artifact proof, but does not supply external process supervision by itself.

Keep all capture settings unchanged: BF16 model, thinking disabled, block 44 output at the final unpadded initial-prefix token, 90 contexts, no truncation, sequence cap 32768, batches of at most 2 rows/16384 padded tokens, six 15-row shards. Prefixes longer than the batch-token budget correctly run as singletons. Do not use `max_contexts` or alter batch settings as a production override. The 90 vectors contain 921,600 bytes of float16 values; token metadata and provenance add variable overhead. Validate the actual selected fresh prefix lengths and saved token IDs before capture.

A production-shape GPU memory check remains required: longest selected prefix plus an actual eligible two-row batch, using the existing hook helper and measured peak memory. The hook path requests only block 44 and `logits_to_keep=1` when the model explicitly supports it; do not materialize all hidden states or full-sequence logits for the longest prefix just to compare hooks. Restrict any tuple comparison to a short memory-safe row. The shared helper does not explicitly override `use_cache`; measure its actual pinned-model behavior rather than assume no cache. This audit ran no GPU smoke and makes no peak-memory claim. Optional kernel absence may affect speed and must not be silently remedied by changing the reviewed recipe.

Finally, copy every capture output back to the canonical VM `RUN/capture`, preserve process receipts, and run `capture.validate_binding(RUN / "capture")` on the VM. Require the exact 90-key/token roster, six hashed finite float16 `[15,1,5120]` payloads, untouched original/derived evidence and the current 29-source closure. Actual capture completion, archive verification and managed pod teardown are later gates; this readiness report is not proof of any of them.
