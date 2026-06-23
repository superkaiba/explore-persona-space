# ruff: noqa: RUF002, RUF003
# Intentional Unicode (≪, ※, ‖, ½, ∩, ∅) in scientific docstrings/comments.
"""Issue #545 metric-race — expanded Group-A predictor zoo (plan §4.1 + §4.2).

THE single manipulated variable of the `full-metric-race-per-family`
follow-up: Group-A geometry grows from v1's 96 raw-centroid variants
({cosine, neg_l2, projection}) to the full #493 metric zoo (centroid +
covariance-aware + cloud + output-distribution JS/KL) over a FROZEN v1
leakage matrix. NO retraining; NO matrix recompute. The #493 metric engine
(`scripts/issue493_extraction_metric_bakeoff.py`) + the #540 JS-RB harness
(`js_canonical`) are IMPORTED, never re-implemented.

Two phases, cleanly split (CLAUDE.md "CPU-only phases don't hold GPU pods"):

- **GPU** (`extract_clouds_and_outdist_gpu`): build per-behavior activation
  CLOUDS (row demos = 8-point per-demo cloud; row nl = ≤8-point per-elicitation
  cloud; column = ≥50-point per-probe cloud) → ``clouds.npz``; build the
  JS/KL output-distribution per-(row,col,flavor) RB estimates (#540 Phase S/T
  discipline: vLLM sample, HF teacher-force, GPU-resident per-position reduce)
  → ``outdist/*.json``. Both checkpoint-per-unit.
- **CPU** (`build_zoo_predictors`): read ``clouds.npz`` + ``outdist/*.json``
  and emit the ``A__*.json`` predictor files in the v1 schema the frozen
  ``scoring.py`` consumes unchanged.

Serialization contract (Nit N1): a metric that is N/A for a cell (degenerate
cloud, single-text flavor, unpaired ``delta_spec``, <4-point cloud) is
serialized by OMITTING that cell's key from the ``cells`` dict — NEVER
``"cell": null`` — because ``scoring.py::weighted_kendall_tau`` gates inclusion
on ``c in pred`` (a literal null would crash / corrupt τ).
"""

from __future__ import annotations

import os

# PyTorch CUDA allocator: expandable_segments defragments reserved-but-
# unallocated memory. Under STRATEGY E (round-38) the HF base model is the SOLE
# GPU resident during the extraction phase (vLLM runs in a separate subprocess
# that exits first), so the co-residency OOM the setting originally mitigated is
# gone — but it stays beneficial: the per-text hidden-state hook path
# (`_mean_hidden_states`) allocates + frees activation segments every iteration,
# and expandable_segments lets the allocator reuse those segments instead of
# growing the reserved pool. MUST be set BEFORE the first `import torch` — torch
# reads PYTORCH_CUDA_ALLOC_CONF once when its CUDA allocator initializes. torch
# is imported lazily inside extract_clouds_and_outdist_gpu, so this module-top
# setdefault is guaranteed to run first. setdefault (not assignment) so an
# explicit launcher / env override always wins. (#545 round-4, kept round-38.)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import json
import logging
import sys
from pathlib import Path

import numpy as np

from . import (
    BASE_MODEL,
    reproducibility_metadata,
)
from .columns import COLUMNS, column_applies
from .eval_battery import battery_probes
from .predictors import (
    EXTRACTION_POINTS,
    FLAVORS,
    GEOMETRY_LAYERS,
    NL_DESCRIPTIONS,
    _demo_messages,
    _mean_hidden_states,
)
from .rows import active_rows

logger = logging.getLogger(__name__)

ROWS = active_rows()

# Import the #493 metric engine (the validated bake-off code — IMPORTED, never
# re-implemented). The script lives under scripts/, not the package, so add it
# to sys.path the same way the #540 dispatcher does.
_SCRIPTS_DIR = Path(__file__).resolve().parents[4] / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import issue493_extraction_metric_bakeoff as i493  # noqa: E402

# JS/KL canonical RB math (#540 / persona-distance-metrics.md).
from explore_persona_space.analysis import js_canonical as jsc  # noqa: E402

# --- metric-race hyperparameters (plan §10/§11; #493 defaults) -------------
PCA_K = 16  # #493 dual-Gram PCA subspace top-k (n≪d-safe)
CENTROID_METRICS_NEW = ("euclidean", "mahal", "mahal_pooled_ctx")
CENTROID_METRICS_V1 = ("cosine", "neg_l2", "projection")  # carried reference
CLOUD_METRICS = ("mmd", "c2st", "wass2", "gauss_kl")  # delta_spec handled separately
CENTERINGS = ("raw", "centered")
MIN_CLOUD_POINTS = 4  # <4-point clouds record None (plan §4.1)

# JS/KL output-distribution (#540 RB estimator; persona-distance-metrics.md).
JS_R_SAMPLES = 8
JS_TEMP = 1.0
JS_MAX_NEW_TOKENS = 1024
JS_N_PROBES = 50
# round-38 STRATEGY E: raised 4096 → 8192. The round-37 (Strategy D) pre-check
# measured the JS scoring path constructs prompts up to 5631 tokens; with up to
# JS_MAX_NEW_TOKENS=1024 sampled, the worst-case sequence is ~6655 tokens. The
# old 4096 cap SILENTLY truncated those prompts inside vLLM (no error — vLLM
# clamps over-length prompts), corrupting the teacher-force / divergence reads on
# the long-probe columns. With vLLM now SOLE GPU resident (subprocess isolation),
# there is ample memory to raise the cap clear of the worst case. Kept here AND
# mirrored in vllm_worker.WORKER_MAX_MODEL_LEN (asserted equal by the regression
# test) so the value used to build prompts matches the engine's actual capacity.
JS_MAX_SEQ_LEN = 8192
JS_DIRECTIONS = ("js", "kl_narrow_broad", "kl_broad_narrow")
# HF teacher-force sub-batch in outdist (jsc.teacher_forced_response_logps).
# Under STRATEGY E (round-38) the HF base model is the SOLE GPU resident during
# the extraction phase — vLLM has already exited its subprocess and freed the
# GPU — so the (B, L, V≈152k) bf16 logits transient no longer competes with a
# co-resident vLLM engine. The B=16 transient (~5.0 GiB) fits comfortably in the
# ~80 GiB an isolated 7B HF model leaves. The r4 lowering to 4 was a co-residency
# workaround; restored to 16 for throughput now that the OOM cause (co-residency)
# is gone. (Re-measure on the pod and lower only if a real OOM recurs.)
JS_TF_MAX_BATCH = 16

# round-38 STRATEGY E — subprocess vLLM isolation. After SIX OOMs of co-residency
# strategies (r1/r3/r4/r6/r8 util/batch/hooks tuning + r10 max_seq_len HALT
# because the probes are genuinely long), HF and vLLM are SEQUENCED into phases
# so they never co-reside on the H100:
#
#   Phase A (vLLM-sampling, SUBPROCESS): vllm_worker.py loads the vLLM engine
#     with the FULL GPU to itself (gpu_memory_utilization=0.85), samples ALL
#     on-policy responses the extraction needs (nl-cloud elicitation + the
#     outdist per-(row,col,flavor) response pairs), writes them to disk as
#     token-id + text JSONs, then EXITS — fully releasing the GPU.
#   Phase B (HF-extraction, MAIN process): the HF base model loads (sole GPU
#     resident now), reads the cached responses, runs the cloud hidden-state
#     hooks (#545 r9 per-layer-hook path) + the teacher-force log-prob /
#     divergence reads, then frees the GPU.
#
# Each model gets ~80 GiB in turn — no co-residency budget, no util dial to
# chase, no pre-init free-memory assert. The vLLM util lives in vllm_worker.py
# (WORKER_GPU_MEM_UTIL); this module no longer loads vLLM in-process.
#
# IPC: file-based request/response under <out_dir>/vllm_ipc/ (resilient to a
# mid-phase crash — written responses survive and are re-read). See
# vllm_worker.py for the contract.
VLLM_IPC_SUBDIR = "vllm_ipc"
# Seconds the main process polls for a vLLM-worker response/error before
# declaring the worker dead (fail LOUD, not a hang). The engine load + a full
# sampling pass over the grid can take many minutes on a real H100; sized
# generously. Overridable for tests.
VLLM_WORKER_TIMEOUT_S = 3600.0
VLLM_POLL_INTERVAL_S = 1.0


# ---------------------------------------------------------------------------
# Behavior-context construction (the JS/KL "behavior-b context")
# ---------------------------------------------------------------------------


def _behavior_context_messages(behavior_id: str, flavor: str) -> list[dict]:
    """The conditioning prefix that puts the base model "in" behavior b.

    - ``nl``: a single system turn describing the behavior tendency (the
      Group-A nl-flavor elicitation framing — plan §4.1).
    - ``demos``: the K=8 demo turns prepended as few-shot user/assistant
      pairs (the demo-flavor conditioning).

    Returns a chat-message prefix (system-prompt persona injection per
    CLAUDE.md; never user/assistant for the persona itself).
    """
    if flavor == "nl":
        nl = NL_DESCRIPTIONS.get(behavior_id)
        if not nl:
            return []
        return [{"role": "system", "content": f"The assistant has a tendency: {nl}."}]
    if flavor == "demos":
        try:
            return _demo_messages(behavior_id)
        except FileNotFoundError:
            return []
    raise KeyError(flavor)


def _column_probe_texts(column_id: str, cap: int) -> list[str]:
    return [p["question"] for p in battery_probes(COLUMNS[column_id], cap=cap)]


# ---------------------------------------------------------------------------
# Strategy-E vLLM subprocess client (the main-process half of the IPC contract)
# ---------------------------------------------------------------------------


class _VllmClient:
    """Main-process side of the file-based vLLM-worker IPC (Strategy E).

    Spawns ``vllm_worker.py`` as a SUBPROCESS, accumulates sampling requests
    (each a list of prompt token-id lists + sampling params, keyed by a
    ``probe_id``), then on ``run()`` writes all request files, drops the
    ``READY`` sentinel, and polls for the per-probe response files the worker
    writes. The worker exits when the queue drains; ``close()`` drops ``STOP``
    and reaps the process so the GPU is released BEFORE the HF model loads.

    The contract carries only token-id lists + decoded text across the boundary
    (never tensors) — the divergence / log-prob math stays HF-side. Resilient to
    a worker crash: a ``worker.error`` sentinel (or process death) raises LOUD
    rather than hanging on missing responses.
    """

    def __init__(self, ipc_dir: Path, *, worker_argv: list[str] | None = None):
        self.ipc_dir = ipc_dir
        self.req_dir = ipc_dir / "requests"
        self.resp_dir = ipc_dir / "responses"
        self.req_dir.mkdir(parents=True, exist_ok=True)
        self.resp_dir.mkdir(parents=True, exist_ok=True)
        self._requests: dict[str, dict] = {}
        self._proc = None
        # Default launch: `python -m ...vllm_worker --ipc-dir <dir>`. Overridable
        # for tests (a stub worker) via worker_argv.
        self._worker_argv = worker_argv or [
            sys.executable,
            "-m",
            "explore_persona_space.experiments.behavior_testbed_545.vllm_worker",
            "--ipc-dir",
            str(ipc_dir),
        ]

    def add_request(
        self,
        probe_id: str,
        prompt_token_ids: list[list[int]],
        *,
        n: int,
        max_tokens: int,
        temperature: float = JS_TEMP,
        top_p: float = 1.0,
        seed: int = 545,
    ) -> None:
        """Queue one sampling request. ``prompt_token_ids`` is a list of
        token-id lists (one per prompt); responses come back aligned per-prompt."""
        if probe_id in self._requests:
            raise KeyError(f"duplicate vLLM request probe_id {probe_id!r}")
        self._requests[probe_id] = {
            "probe_id": probe_id,
            "prompt_token_ids": prompt_token_ids,
            "n": int(n),
            "max_tokens": int(max_tokens),
            "temperature": float(temperature),
            "top_p": float(top_p),
            "seed": int(seed),
        }

    def __len__(self) -> int:
        return len(self._requests)

    def run(self) -> dict[str, list[list[dict]]]:
        """Spawn the worker, write all requests, poll for all responses.

        Returns ``{probe_id: completions}`` where ``completions`` is a list (per
        prompt) of lists of ``{"token_ids", "text", "finish_reason"}`` dicts.
        Idempotent across a re-run: a response file already present (from a prior
        partial phase) is reused, and only the missing requests need the worker.
        """
        import subprocess

        if not self._requests:
            return {}

        # Clear stale CONTROL sentinels from a prior run (BLOCKER fix, round 39):
        # a same-out_dir retry (after a Phase-B failure or a partial Phase-A
        # failure) must not inherit the previous run's STOP/READY/worker.error.
        # A stale STOP makes the freshly-spawned worker load vLLM then exit
        # without processing the new requests; a stale worker.error makes the
        # first poll fail LOUD immediately; a stale READY can race the worker
        # before all request files are written. We deliberately DO NOT delete
        # pre-existing RESPONSE files — the documented re-run contract (a prior
        # partial phase's responses are reused; `_collect_ready` only reads
        # responses whose probe_id is in THIS run's request set) depends on them.
        for sentinel in ("STOP", "READY", "worker.error"):
            stale = self.ipc_dir / sentinel
            if stale.exists():
                stale.unlink()

        # Write every request file FIRST (so a fast worker never sees READY with
        # a request still un-written), then the READY sentinel.
        for probe_id, payload in self._requests.items():
            (self.req_dir / f"{probe_id}.json").write_text(json.dumps(payload))

        # Spawn the worker subprocess with an explicit env (subprocess env
        # passthrough — the worker needs HF creds to pull the base model).
        self._proc = subprocess.Popen(self._worker_argv, env={**os.environ})
        logger.info(
            "[phase=vllm-sample] spawned vLLM worker pid=%s for %d requests",
            self._proc.pid,
            len(self._requests),
        )
        (self.ipc_dir / "READY").write_text("1")

        # The GPU-owning worker MUST be reaped on EVERY exit path (BLOCKER fix,
        # round 39): `_poll_for_responses` raises on worker.error / worker death
        # / TimeoutError / a malformed response — without this finally the leaked
        # worker keeps the GPU at WORKER_GPU_MEM_UTIL and blocks the next
        # attempt's HF load, the exact co-residency OOM Strategy E exists to kill.
        # The EPS recovery model is retry-in-same-pod, not "pod teardown reclaims
        # it", so the leak is load-bearing.
        try:
            results = self._poll_for_responses()
        finally:
            self.close()
        return results

    def _poll_for_responses(self) -> dict[str, list[list[dict]]]:
        import time

        error_path = self.ipc_dir / "worker.error"
        deadline = time.monotonic() + VLLM_WORKER_TIMEOUT_S
        results: dict[str, list[list[dict]]] = {}
        wanted = set(self._requests)
        while wanted:
            if error_path.exists():
                err = error_path.read_text()
                raise RuntimeError(f"vLLM worker reported a fatal error:\n{err}")
            # Worker process died without finishing → fail LOUD (not a hang).
            if self._proc is not None and self._proc.poll() is not None:
                rc = self._proc.returncode
                if error_path.exists():
                    raise RuntimeError(f"vLLM worker exited rc={rc}:\n{error_path.read_text()}")
                # It may have just finished the last response; do one final scan.
                self._collect_ready(wanted, results)
                if wanted:
                    raise RuntimeError(
                        f"vLLM worker exited rc={rc} with {len(wanted)} responses "
                        f"still missing: {sorted(wanted)[:5]}..."
                    )
                break
            self._collect_ready(wanted, results)
            if not wanted:
                break
            if time.monotonic() > deadline:
                raise TimeoutError(
                    f"vLLM worker did not produce {len(wanted)} responses within "
                    f"{VLLM_WORKER_TIMEOUT_S}s: {sorted(wanted)[:5]}..."
                )
            time.sleep(VLLM_POLL_INTERVAL_S)
        return results

    def _collect_ready(self, wanted: set[str], results: dict) -> None:
        for probe_id in list(wanted):
            resp_path = self.resp_dir / f"{probe_id}.json"
            if resp_path.exists():
                d = json.loads(resp_path.read_text())
                results[probe_id] = d["completions"]
                wanted.discard(probe_id)

    def close(self) -> None:
        """Drop STOP, reap the worker, confirm the GPU is released."""
        import subprocess

        (self.ipc_dir / "STOP").write_text("1")
        if self._proc is None:
            return
        try:
            self._proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            logger.warning("[phase=vllm-sample] worker did not exit on STOP — terminating")
            self._proc.terminate()
            try:
                self._proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._proc.wait(timeout=30)
        logger.info("[phase=vllm-sample] vLLM worker reaped (rc=%s)", self._proc.returncode)
        self._proc = None


# ---------------------------------------------------------------------------
# GPU phase: clouds + JS/KL output-distribution
# ---------------------------------------------------------------------------


def _row_cloud_texts(row_id: str) -> dict[str, list[str]]:
    """Per-flavor text lists for a ROW's cloud (NOT concatenated — plan §4.1).

    - ``demos``: the K=8 demo answers as 8 SEPARATE texts → 8-point cloud
      (NOT v1's single concatenated demo_text).
    - ``nl``: empty here — the nl cloud is built from base-model temp-1
      elicitation SAMPLES (see ``extract_clouds_and_outdist_gpu``), not from
      a single description string.
    """
    out: dict[str, list[str]] = {}
    try:
        demos = _demo_messages(row_id)
        answers = [m["content"] for m in demos if m["role"] == "assistant"][:8]
        if answers:
            out["demos"] = answers
    except FileNotFoundError:
        logger.warning("zoo: no demos for row %s (demos cloud skipped)", row_id)
    return out


def _prompt_ids_for(tokenizer, ctx: list[dict], q: str) -> list[int]:
    """Token-ids for the chat-templated (ctx + user-question) prompt."""
    msgs = [*ctx, {"role": "user", "content": q}]
    text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return tokenizer.encode(text, add_special_tokens=False)


def extract_clouds_and_outdist_gpu(  # noqa: C901 — sequenced vLLM-sample then HF-extract phases (Strategy E)
    out_dir: Path,
    *,
    device: str = "cuda:0",
    rows_subset: list[str] | None = None,
    cols_subset: list[str] | None = None,
    n_probes: int = JS_N_PROBES,
    r_samples: int = JS_R_SAMPLES,
    nl_cloud_samples: int = 8,
    skip_outdist: bool = False,
    skip_clouds: bool = False,
    vllm_worker_argv: list[str] | None = None,
) -> dict:
    """Build per-behavior activation clouds (→ ``clouds.npz``) + JS/KL
    output-distribution RB estimates (→ ``outdist/*.json``) on the BASE model.

    STRATEGY E (round-38): HF and vLLM are SEQUENCED into phases so they never
    co-reside on the H100 (the cause of six OOMs). The function runs:

      Phase A — vLLM SUBPROCESS sampling. Enumerate every on-policy sampling
        request the extraction needs (nl-cloud elicitation per row + the outdist
        per-(row,col,flavor) response pairs, both sides), spawn ``vllm_worker``
        with the FULL GPU to itself, collect the responses (token-ids + text) to
        disk, then REAP the worker — releasing the GPU.
      Phase B — HF MAIN-process extraction. Load the HF base model (sole GPU
        resident now), build the clouds (demos from texts, nl from the cached
        samples, col probes from texts), teacher-force the cached outdist
        responses → JS/KL divergences, then free the GPU.

    The cell subset (``rows_subset`` / ``cols_subset``) threads the smoke =
    sweep parameterization (smoke IS the full grid restricted to a tiny cell
    list) IDENTICALLY through BOTH phases. ``n_probes`` / ``r_samples`` /
    ``nl_cloud_samples`` are the descope ladder knobs (plan §9).
    ``vllm_worker_argv`` overrides the worker launch (a stub worker in tests).

    Returns a small summary dict (counts + JS truncation manipulation check).
    """

    import gc

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    out_dir.mkdir(parents=True, exist_ok=True)
    outdist_dir = out_dir / "outdist"
    outdist_dir.mkdir(parents=True, exist_ok=True)

    rows = rows_subset or list(ROWS.keys())
    cols = cols_subset or [c for c, col in COLUMNS.items() if col.scoring_eligible]

    # The tokenizer is cheap + GPU-free; load it once up front (used to build
    # the vLLM prompt token-ids in Phase A AND the HF reads in Phase B).
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)

    summary: dict = {"rows": rows, "cols": cols, "n_probes": n_probes, "r_samples": r_samples}
    generic_q = "Can you help me with something?"

    # ===================================================================
    # PHASE A — vLLM SUBPROCESS sampling (vLLM is sole GPU resident; exits
    # before Phase B loads HF). Enumerate every request, run, collect to disk.
    # ===================================================================
    nl_rows = [r for r in rows if NL_DESCRIPTIONS.get(r)] if not skip_clouds else []
    # outdist pairs needing sampling: (row, col, flavor) → the per-pair probes,
    # ctx_row, ctx_col. We sample BOTH sides (a = row-conditioned, b = col-
    # conditioned) over the SAME column probe set, keyed by side in the IPC.
    outdist_specs: list[tuple[str, str, str, list[str], list[dict], list[dict]]] = []
    if not skip_outdist:
        for flavor in FLAVORS:
            for row_id in rows:
                ctx_row = _behavior_context_messages(row_id, flavor)
                if not ctx_row:
                    continue
                for col_id in cols:
                    if not column_applies(COLUMNS[col_id], ROWS[row_id]):
                        continue
                    out_path = outdist_dir / f"{row_id}__{col_id}__{flavor}.json"
                    if out_path.exists():
                        continue  # checkpoint-per-cell: already scored, skip
                    probes = _column_probe_texts(col_id, cap=n_probes)
                    if not probes:
                        continue
                    ctx_col = _column_behavior_context(col_id, flavor)
                    outdist_specs.append((row_id, col_id, flavor, probes, ctx_row, ctx_col))

    ipc_dir = out_dir / VLLM_IPC_SUBDIR
    client = _VllmClient(ipc_dir, worker_argv=vllm_worker_argv)

    # nl-cloud elicitation requests: one generic probe per nl-row, n=nl_cloud_samples.
    for row_id in nl_rows:
        ids = _prompt_ids_for(tokenizer, _behavior_context_messages(row_id, "nl"), generic_q)
        client.add_request(f"nl|{row_id}", [ids], n=nl_cloud_samples, max_tokens=256, seed=545)
    # outdist sampling requests: side a (row-conditioned) + side b (col-conditioned).
    for row_id, col_id, flavor, probes, ctx_row, ctx_col in outdist_specs:
        a_ids = [_prompt_ids_for(tokenizer, ctx_row, q) for q in probes]
        b_ids = [_prompt_ids_for(tokenizer, ctx_col, q) for q in probes]
        client.add_request(
            f"outdist|{row_id}|{col_id}|{flavor}|a",
            a_ids,
            n=r_samples,
            max_tokens=JS_MAX_NEW_TOKENS,
            seed=545,
        )
        client.add_request(
            f"outdist|{row_id}|{col_id}|{flavor}|b",
            b_ids,
            n=r_samples,
            max_tokens=JS_MAX_NEW_TOKENS,
            seed=545,
        )

    responses: dict[str, list[list[dict]]] = {}
    if len(client) > 0:
        logger.info(
            "[phase=vllm-sample] dispatching %d requests to the vLLM subprocess "
            "(nl_rows=%d, outdist_pairs=%d)",
            len(client),
            len(nl_rows),
            len(outdist_specs),
        )
        responses = client.run()  # spawns worker, polls, reaps — GPU freed on return
    logger.info("[phase=vllm-sample] collected %d response sets; GPU freed", len(responses))

    # ===================================================================
    # PHASE B — HF MAIN-process extraction (HF base model is sole GPU resident).
    # ===================================================================
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, device_map={"": 0}, trust_remote_code=True
    )
    model.eval()

    # --- B.1 Activation clouds → clouds.npz -------------------------------
    if not skip_clouds:
        clouds: dict[str, np.ndarray] = {}
        # Row demos clouds (8-point per-demo) — texts directly, no vLLM needed.
        for row_id in rows:
            texts = _row_cloud_texts(row_id)
            for flavor, demo_texts in texts.items():
                reps = _mean_hidden_states(
                    model, tokenizer, demo_texts, device, retain_per_sample_reps=True
                )
                for layer in GEOMETRY_LAYERS:
                    for point in EXTRACTION_POINTS:
                        t = reps.get(layer, {}).get(point)
                        if t is not None:
                            clouds[f"row|{row_id}|{flavor}|{layer}|{point}"] = t.numpy()
            logger.info("[phase=clouds] row demos cloud %s", row_id)
        # Row nl clouds: the base-model temp-1 responses sampled in Phase A.
        for row_id in nl_rows:
            comps = responses.get(f"nl|{row_id}")
            if not comps:
                logger.warning("[phase=clouds] no nl samples for row %s (skipped)", row_id)
                continue
            samples = [c["text"] for c in comps[0] if c["text"].strip()]
            if not samples:
                continue
            reps = _mean_hidden_states(
                model, tokenizer, samples, device, retain_per_sample_reps=True
            )
            for layer in GEOMETRY_LAYERS:
                for point in EXTRACTION_POINTS:
                    t = reps.get(layer, {}).get(point)
                    if t is not None:
                        clouds[f"row|{row_id}|nl|{layer}|{point}"] = t.numpy()
            logger.info("[phase=clouds] row nl cloud %s (%d samples)", row_id, len(samples))
        # Column probe clouds (≥n_probes per-probe) — texts directly.
        for col_id in cols:
            probes = _column_probe_texts(col_id, cap=n_probes)
            if not probes:
                continue
            reps = _mean_hidden_states(
                model, tokenizer, probes, device, retain_per_sample_reps=True
            )
            for layer in GEOMETRY_LAYERS:
                for point in EXTRACTION_POINTS:
                    t = reps.get(layer, {}).get(point)
                    if t is not None:
                        clouds[f"col|{col_id}|probe|{layer}|{point}"] = t.numpy()
            logger.info("[phase=clouds] col probe cloud %s", col_id)
        np.savez_compressed(out_dir / "clouds.npz", **clouds)
        summary["n_cloud_arrays"] = len(clouds)
        logger.info("[phase=clouds] wrote clouds.npz (%d arrays)", len(clouds))

    # --- B.2 JS/KL output-distribution → outdist/*.json -------------------
    if not skip_outdist:
        trunc_total, trunc_hits = 0, 0
        n_outdist = 0
        for row_id, col_id, flavor, probes, ctx_row, ctx_col in outdist_specs:
            comps_a = responses.get(f"outdist|{row_id}|{col_id}|{flavor}|a")
            comps_b = responses.get(f"outdist|{row_id}|{col_id}|{flavor}|b")
            if comps_a is None or comps_b is None:
                logger.warning(
                    "[phase=outdist] missing sampled responses for %s__%s__%s (skipped)",
                    row_id,
                    col_id,
                    flavor,
                )
                continue
            prompts_a = [_prompt_ids_for(tokenizer, ctx_row, q) for q in probes]
            prompts_b = [_prompt_ids_for(tokenizer, ctx_col, q) for q in probes]
            res = _score_outdist_pair_from_samples(
                model,
                prompts_a,
                prompts_b,
                comps_a,
                comps_b,
                r_samples=r_samples,
            )
            if res is None:
                continue
            trunc_total += res["_trunc_total"]
            trunc_hits += res["_trunc_hits"]
            payload = {
                "row": row_id,
                "col": col_id,
                "flavor": flavor,
                "rb": {k: v for k, v in res.items() if not k.startswith("_")},
                "metadata": reproducibility_metadata(),
            }
            (outdist_dir / f"{row_id}__{col_id}__{flavor}.json").write_text(
                json.dumps(payload, indent=1)
            )
            n_outdist += 1
            logger.info("[phase=outdist] %s__%s__%s", row_id, col_id, flavor)
        summary["n_outdist_pairs"] = n_outdist
        summary["js_truncation_rate"] = (trunc_hits / trunc_total) if trunc_total else 0.0
        logger.info(
            "[phase=outdist] %d pairs, truncation=%.4f (manipulation check, #548)",
            n_outdist,
            summary["js_truncation_rate"],
        )

    # Explicit GPU teardown of the HF model (sole resident; vLLM exited in Phase
    # A) before the function returns — the CPU build_zoo_predictors phase runs
    # next in-process and a following GPU consumer should start clean.
    alloc_before = torch.cuda.memory_allocated() / (1024**3)
    del model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    logger.info(
        "[phase=extract] GPU teardown: memory_allocated %.1f GiB -> %.1f GiB",
        alloc_before,
        torch.cuda.memory_allocated() / (1024**3),
    )
    (out_dir / "extract_summary.json").write_text(json.dumps(summary, indent=1))
    return summary


def _column_behavior_context(col_id: str, flavor: str) -> list[dict]:
    """The partner ("behavior-b′") context for a column: the diagonal row's
    behavior context where one exists, else the bare default assistant."""
    for row in ROWS.values():
        if row.diagonal_column == col_id:
            ctx = _behavior_context_messages(row.row_id, flavor)
            if ctx:
                return ctx
    return []  # bare default assistant (no system persona)


def _samples_from_completions(
    comps: list[list[dict]], r_samples: int
) -> tuple[list[list[list[int]]], int, int]:
    """Map the worker's per-prompt completion dicts to the #540 sampled-response
    shape (per-probe list of terminator-ruled token-id lists) + truncation
    counters. ``comps`` is ``[per_prompt][per_completion]`` of
    ``{"token_ids", "text", "finish_reason"}``."""
    per_probe: list[list[list[int]]] = []
    t_total = t_hit = 0
    for prompt_comps in comps:
        prows: list[list[int]] = []
        for comp in prompt_comps:
            ids, _action = jsc.apply_terminator_rule(list(comp["token_ids"]), comp["finish_reason"])
            prows.append(ids)
            t_total += 1
            t_hit += int(comp["finish_reason"] == "length")
        per_probe.append(prows)
    return per_probe, t_total, t_hit


def _score_outdist_pair_from_samples(
    model,
    prompts_a: list[list[int]],
    prompts_b: list[list[int]],
    comps_a: list[list[dict]],
    comps_b: list[list[dict]],
    *,
    r_samples: int,
) -> dict | None:
    """RB sequence-level JS + both KL directions for one (behavior-b ctx,
    behavior-b′ ctx) pair, given the responses ALREADY sampled by the vLLM
    subprocess (Strategy E — HF and vLLM never co-reside).

    The HF half is unchanged from the original ``_score_outdist_pair``: HF
    teacher-force each sampled response through BOTH conditioned prompts, exact
    full-vocab per-position divergence (GPU-resident reduce; only the per-sample
    scalar means leave the GPU), aggregate with ``jsc.rb_pair_estimate``. Only
    the SAMPLING moved out of this function into the subprocess.

    ``prompts_a`` / ``prompts_b`` are the per-probe prompt token-ids (rebuilt
    HF-side from the same ctx + probes); ``comps_a`` / ``comps_b`` are the
    worker's per-probe completion dicts. Returns the RB dict + truncation
    counters, or None when no probe yields usable samples on both sides.
    """
    samples_a, ta_tot, ta_hit = _samples_from_completions(comps_a, r_samples)
    samples_b, tb_tot, tb_hit = _samples_from_completions(comps_b, r_samples)

    # Fail LOUD on a partial / corrupt IPC payload (BLOCKER fix, round 39):
    # the previous `min(len(...), ...)` silently scored fewer probes/samples than
    # the registered estimand (R=r_samples per probe, full probe set), which
    # violates CLAUDE.md fail-fast and would quietly bias the JS/KL read if the
    # worker dropped a probe or returned a short completion list. The worker
    # produces exactly one completion-list per prompt and exactly `n` (= r_samples)
    # completions per prompt, so any mismatch is a real defect, not a tolerable
    # short read.
    if len(comps_a) != len(prompts_a):
        raise ValueError(
            f"Phase-A output truncated: comps_a={len(comps_a)} != prompts_a={len(prompts_a)}"
        )
    if len(comps_b) != len(prompts_b):
        raise ValueError(
            f"Phase-A output truncated: comps_b={len(comps_b)} != prompts_b={len(prompts_b)}"
        )
    for pi in range(len(prompts_a)):
        if len(samples_a[pi]) < r_samples:
            raise ValueError(
                f"Phase-A samples_a[{pi}] has {len(samples_a[pi])} responses, need >= {r_samples}"
            )
        if len(samples_b[pi]) < r_samples:
            raise ValueError(
                f"Phase-A samples_b[{pi}] has {len(samples_b[pi])} responses, need >= {r_samples}"
            )

    # All probes verified present at full sample count → score the registered set
    # (no min() truncation). n_probes_used is recorded in the RB output below.
    n_probes_used = len(prompts_a)

    a_kl_m, b_kl_m, a_kl_ab, b_kl_ba = [], [], [], []
    for pi in range(n_probes_used):
        rows_a = samples_a[pi][:r_samples]
        rows_b = samples_b[pi][:r_samples]
        if not rows_a or not rows_b:
            continue
        responses = rows_a + rows_b
        lp_a = jsc.teacher_forced_response_logps(
            model, prompts_a[pi], responses, max_batch=JS_TF_MAX_BATCH
        )
        lp_b = jsc.teacher_forced_response_logps(
            model, prompts_b[pi], responses, max_batch=JS_TF_MAX_BATCH
        )
        na = len(rows_a)
        for i in range(len(responses)):
            side = "a" if i < na else "b"
            lp_side = lp_a[i] if side == "a" else lp_b[i]
            lp_other = lp_b[i] if side == "a" else lp_a[i]
            pd = jsc.per_position_divergences(lp_side, lp_other)
            (a_kl_m if side == "a" else b_kl_m).append(float(pd.kl_side_m_bits.mean()))
            (a_kl_ab if side == "a" else b_kl_ba).append(float(pd.kl_side_other_nats.mean()))
    if not a_kl_m or not b_kl_m:
        return None
    rb = jsc.rb_pair_estimate(
        np.array(a_kl_m), np.array(b_kl_m), np.array(a_kl_ab), np.array(b_kl_ba)
    )
    rb["_trunc_total"] = ta_tot + tb_tot
    rb["_trunc_hits"] = ta_hit + tb_hit
    # Telemetry (round 39): the effective probe/sample count actually scored, so a
    # downstream reader can confirm the registered estimand (full probe set, R per
    # probe) was honored rather than silently truncated.
    rb["_n_probes_used"] = n_probes_used
    rb["_r_samples"] = r_samples
    return rb


# ---------------------------------------------------------------------------
# CPU phase: build A__*.json predictor files from cached clouds + outdist
# ---------------------------------------------------------------------------


def _load_clouds(out_dir: Path) -> dict[str, np.ndarray]:
    npz = out_dir / "clouds.npz"
    if not npz.exists():
        raise FileNotFoundError(f"{npz} missing — run extract_clouds_and_outdist_gpu first")
    with np.load(npz) as z:
        return {k: z[k] for k in z.files}


def _finite_cloud(arr: np.ndarray) -> np.ndarray | None:
    """Drop non-finite rows; return None if <MIN_CLOUD_POINTS remain or the
    cloud is constant (the #493 ``_finite_and_non_constant`` precedent)."""
    if arr is None or arr.ndim != 2:
        return None
    mask = np.all(np.isfinite(arr), axis=1)
    arr = arr[mask]
    if len(arr) < MIN_CLOUD_POINTS:
        return None
    if np.allclose(arr.std(axis=0), 0.0):
        return None
    return arr


def _write_predictor(out_dir: Path, name: str, cells: dict[str, float], extra: dict) -> Path | None:
    """Write one A__<name>.json in the v1 schema. OMITS None cells (Nit N1).
    Returns None (no file) if no cell scored."""
    cells = {k: float(v) for k, v in cells.items() if v is not None and np.isfinite(v)}
    if not cells:
        return None
    p = out_dir / f"A__{name}.json"
    p.write_text(
        json.dumps(
            {
                "group": "A",
                "name": name,
                "track": "shift",
                "cells": cells,
                "n_cells": len(cells),
                "metadata": reproducibility_metadata(),
                **extra,
            },
            indent=1,
        )
    )
    return p


def build_zoo_predictors(  # noqa: C901 — flat metric grid, one branch per metric family
    pred_dir: Path,
    cloud_src_dir: Path | None = None,
) -> list[Path]:
    """CPU: read ``clouds.npz`` + ``outdist/*.json`` and emit A__* predictor
    JSONs (centroid + covariance-aware + cloud + JS/KL) in the v1 schema.

    ``pred_dir`` is where the JSONs land (``predictors_metric_race/``);
    ``cloud_src_dir`` is where ``clouds.npz`` + ``outdist/`` live (defaults to
    ``pred_dir.parent`` — the metric_race root).
    """
    src = cloud_src_dir or pred_dir.parent
    pred_dir.mkdir(parents=True, exist_ok=True)
    clouds = _load_clouds(src)
    written: list[Path] = []

    def _row_cloud(row_id, flavor, layer, point):
        return clouds.get(f"row|{row_id}|{flavor}|{layer}|{point}")

    def _col_cloud(col_id, layer, point):
        return clouds.get(f"col|{col_id}|probe|{layer}|{point}")

    # ---- centroid metrics (raw + centered) × layers × points × flavors ----
    for flavor in FLAVORS:
        for layer in GEOMETRY_LAYERS:
            for point in EXTRACTION_POINTS:
                # Pre-build the context-pooled-cov state once per (layer, point)
                # over ALL in-scope column clouds (mahal_pooled_ctx).
                col_centroids = []
                col_order = []
                for col_id, col in COLUMNS.items():
                    if not col.scoring_eligible:
                        continue
                    cc = _finite_cloud(_col_cloud(col_id, layer, point))
                    if cc is not None:
                        col_centroids.append(cc.mean(axis=0))
                        col_order.append(col_id)
                pooled_state = None
                if len(col_centroids) >= 2:
                    # (n_cond, n_q=1, H) shape for #493 pooled-cov builder.
                    acts = np.array(col_centroids)[:, None, :]
                    pooled_state = i493._build_context_pooled_mahal_state(acts, PCA_K)
                    if pooled_state is None:
                        i493._pop_pooled_failure_reason(acts)
                pooled_idx = {c: i for i, c in enumerate(col_order)}

                # Centering: subtract the pooled mean over in-scope column
                # centroids (the #536 bank-centering lesson) from BOTH sides.
                center_vec = np.array(col_centroids).mean(axis=0) if col_centroids else None

                for centering in CENTERINGS:
                    for metric in CENTROID_METRICS_V1 + CENTROID_METRICS_NEW:
                        # v1 raw {cosine,neg_l2,projection} already exist as the
                        # geom_* reference predictors; the zoo re-emits them
                        # under the centroid-zoo naming for the leaderboard
                        # grouping, plus the centered variant.
                        name = f"cloud_{flavor}_L{layer}_{point}_{centering}_{metric}"
                        cells = {}
                        for row_id, row in ROWS.items():
                            rc = _finite_cloud(_row_cloud(row_id, flavor, layer, point))
                            if rc is None:
                                continue
                            for col_id in col_order:
                                if not column_applies(COLUMNS[col_id], row):
                                    continue
                                cc = _finite_cloud(_col_cloud(col_id, layer, point))
                                if cc is None:
                                    continue
                                s = _centroid_metric(
                                    metric,
                                    rc,
                                    cc,
                                    centering,
                                    center_vec,
                                    pooled_state,
                                    pooled_idx.get(col_id),
                                )
                                if s is not None and np.isfinite(s):
                                    cells[f"{row_id}|{col_id}"] = s
                        p = _write_predictor(
                            pred_dir,
                            name,
                            cells,
                            {
                                "metric": metric,
                                "centering": centering,
                                "layer": layer,
                                "point": point,
                                "flavor": flavor,
                                "in_scope_columns": col_order,
                            },
                        )
                        if p:
                            written.append(p)

    # ---- cloud metrics (mmd, c2st, wass2, gauss_kl) ------------------------
    for flavor in FLAVORS:
        for layer in GEOMETRY_LAYERS:
            for point in EXTRACTION_POINTS:
                for metric in CLOUD_METRICS:
                    name = f"cloud_{flavor}_L{layer}_{point}_{metric}"
                    cells = {}
                    for row_id, row in ROWS.items():
                        rc = _finite_cloud(_row_cloud(row_id, flavor, layer, point))
                        if rc is None:
                            continue
                        for col_id, col in COLUMNS.items():
                            if not col.scoring_eligible or not column_applies(col, row):
                                continue
                            cc = _finite_cloud(_col_cloud(col_id, layer, point))
                            if cc is None:
                                continue
                            s = _cloud_metric(metric, rc, cc)
                            if s is not None and np.isfinite(s):
                                cells[f"{row_id}|{col_id}"] = s
                    p = _write_predictor(
                        pred_dir,
                        name,
                        cells,
                        {"metric": metric, "layer": layer, "point": point, "flavor": flavor},
                    )
                    if p:
                        written.append(p)

    # ---- output-distribution JS/KL (layer-agnostic) -----------------------
    outdist_dir = src / "outdist"
    if outdist_dir.exists():
        for direction in JS_DIRECTIONS:
            for flavor in FLAVORS:
                name = f"outdist_{flavor}_{direction}"
                cells = {}
                for jpath in sorted(outdist_dir.glob("*.json")):
                    d = json.loads(jpath.read_text())
                    if d["flavor"] != flavor:
                        continue
                    rb = d["rb"]
                    if direction == "js":
                        # similarity polarity: 1 - JS (higher = closer)
                        v = 1.0 - rb.get("js_rb_bits", float("nan"))
                    elif direction == "kl_narrow_broad":
                        v = rb.get("kl_ab_nats", float("nan"))
                    else:  # kl_broad_narrow
                        v = rb.get("kl_ba_nats", float("nan"))
                    if v is not None and np.isfinite(v):
                        cells[f"{d['row']}|{d['col']}"] = float(v)
                p = _write_predictor(
                    pred_dir, name, cells, {"direction": direction, "flavor": flavor}
                )
                if p:
                    written.append(p)

    logger.info("[phase=zoo] wrote %d A__* predictor JSONs to %s", len(written), pred_dir)
    return written


# ---------------------------------------------------------------------------
# Metric dispatch helpers (import #493 functions; never re-implement)
# ---------------------------------------------------------------------------


def _apply_centering(rc: np.ndarray, cc: np.ndarray, centering: str, center_vec):
    if centering == "raw" or center_vec is None:
        return rc, cc
    return rc - center_vec[None, :], cc - center_vec[None, :]


def _centroid_metric(metric, rc, cc, centering, center_vec, pooled_state, col_pool_idx):
    """One centroid / covariance-aware metric over two clouds (centroid =
    cloud mean). Returns a similarity-polarity-consistent scalar or None."""
    rc_c, cc_c = _apply_centering(rc, cc, centering, center_vec)
    if metric == "cosine":
        return 1.0 - i493._centroid_cosine_distance(rc_c, cc_c)  # similarity
    if metric == "neg_l2":
        return -i493._centroid_euclidean(rc_c, cc_c)
    if metric == "projection":
        a, b = rc_c.mean(axis=0), cc_c.mean(axis=0)
        nb = np.linalg.norm(b)
        return float(a @ b / nb) if nb > 1e-12 else None
    if metric == "euclidean":
        return -i493._centroid_euclidean(rc_c, cc_c)  # negate → similarity
    if metric == "mahal":
        v = i493._centroid_mahal(rc_c, cc_c, PCA_K)
        return -v if np.isfinite(v) else None  # negate → similarity
    if metric == "mahal_pooled_ctx":
        # mahal_pooled_ctx is defined relative to the pooled-context cov state
        # built over the (centered/raw) COLUMN centroid bank. Only valid in the
        # 'raw' centering (the pooled state is built on raw centroids).
        if centering != "raw" or pooled_state is None or col_pool_idx is None:
            return None
        # Mahalanobis of the row centroid vs the column centroid under pooled cov.
        mu, comps, cov_inv = pooled_state["mu"], pooled_state["components"], pooled_state["cov_inv"]
        ra = rc.mean(axis=0)
        cb = cc.mean(axis=0)
        ya = (ra - mu[0]) @ comps.T
        yb = (cb - mu[0]) @ comps.T
        diff = ya - yb
        v = float(np.sqrt(max(0.0, float(diff @ cov_inv @ diff))))
        return -v if np.isfinite(v) else None
    raise KeyError(metric)


def _cloud_metric(metric, rc, cc):
    """One distributional cloud metric over two N-point clouds. Returns a
    similarity-polarity-consistent scalar or None (degenerate)."""
    if metric == "mmd":
        v = i493._rbf_mmd_squared(rc, cc)
        return -v if np.isfinite(v) else None  # distance → negate to similarity
    if metric == "c2st":
        v = i493._c2st_auc(rc, cc)
        return -v if np.isfinite(v) else None  # 2|AUC-.5| distance → similarity
    if metric == "wass2":
        v = i493._bures_wasserstein2(rc, cc, PCA_K)
        return -v if np.isfinite(v) else None
    if metric == "gauss_kl":
        v = i493._gaussian_sym_kl_in_subspace(rc, cc, PCA_K)
        return -v if np.isfinite(v) else None
    raise KeyError(metric)


def build_delta_spec_predictors(pred_dir: Path, cloud_src_dir: Path | None = None) -> list[Path]:
    """``delta_spec`` — PAIRED only (plan §4.1 Blocker-1 fix, choice A).

    ``_delta_spectrum`` is well-defined only between matched/paired clouds
    drawn from the SAME probe IDs in the SAME order. The row demo/nl clouds
    and the column probe clouds are NOT paired (different item sets), so
    ``delta_spec`` is N/A for every (row, col) cell in this design and records
    NO scores — the predictor file is intentionally NOT written (omit-the-key
    contract at the file level). Kept as a named function so the design intent
    + the runtime assertion are explicit and code-reviewable: a future paired
    construction would build the paired clouds, assert ``probe_ids_a ==
    probe_ids_b`` (same IDs, same order) BEFORE the spectrum subtraction, and
    HALT on mismatch.
    """
    logger.info(
        "[phase=zoo] delta_spec: no paired-probe construction in this design — "
        "N/A for all cells (no predictor file written; plan §4.1 Blocker-1)"
    )
    return []
