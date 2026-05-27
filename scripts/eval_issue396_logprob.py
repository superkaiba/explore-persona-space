#!/usr/bin/env python3
"""Per-LoRA trajectory evaluator for task #396 (※ marker, log-prob DV).

For ONE source LoRA, runs the 48 eval-persona x 20 question matrix
(960 cells) in two phases:

* **Phase 1 (vLLM greedy gen)** — load the merged source-LoRA, sample
  ONE greedy completion per (eval_persona, question) cell.
* **Phase 2 (HF teacher-force trajectory)** — for each completion,
  extract the per-position log p(※) trajectory via
  :func:`compute_marker_logprob_trajectory`. Derive 7 trajectory-shape
  scalars per cell (end-of-response, k=0, max, max_position, mean, AUC,
  slope) plus the MF3 substring-match indicator at zero marginal compute.

Per-cell JSON schema lives in plan v2.3 §4.5; per-source JSON ~1.5 MB,
48 sources total ~75 MB, uploaded to HF data repo
``superkaiba1/explore-persona-space-data`` under
``issue396/logprob/{source}_seed{seed}.json``. The 960 raw completions
land at ``issue396/raw_completions/{source}_seed{seed}.json`` for the
clean-result-critic Lens 4 "qualitative data link" requirement.

**Phase-checkpoint discipline (CLAUDE.md "Checkpoint per phase").**
Phase 1 and Phase 2 each write their output to disk BEFORE the next
phase loads its framework. A Phase 2 crash does not lose Phase 1
output — the script can be re-invoked, Phase 1 will be reloaded from
disk and Phase 2 re-tried. Anti-pattern: ``results = []; for cell: ...;
write(results, path)`` — DO NOT introduce that here.

**vLLM teardown discipline (CLAUDE.md gotcha).** Phase 1 (vLLM) and
Phase 2 (HF Transformers) run in the SAME process; the canonical
``del llm`` + ``destroy_*`` + ``empty_cache`` is NOT sufficient on
multi-GPU pods (orphan worker subprocesses survive and re-grab freed
memory). This script does the destroy + ``psutil`` child-kill +
``nvidia-smi`` sanity check before loading the HF model.

Task #396 plan v2.3 §4.5 + §4.7.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import json
import logging
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

# Ensure scripts/ is on sys.path BEFORE other imports for the panel-48 lookup.
_SCRIPTS_DIR = str(Path(__file__).resolve().parent)
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EVAL_RESULTS_DIR = PROJECT_ROOT / "eval_results" / "issue_396"
EVAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
RAW_COMPLETIONS_DIR = EVAL_RESULTS_DIR / "raw_completions"
RAW_COMPLETIONS_DIR.mkdir(parents=True, exist_ok=True)

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
MARKER_TEXT = " ※"  # leading-space form; Qwen tokenizes to id 83399 (plan §A4)
HF_DATA_REPO = "superkaiba1/explore-persona-space-data"

# Greedy gen knobs: temperature=0, n=1, max_tokens=2048 per CLAUDE.md
# "max_new_tokens ≥ 2x longest trained completion (default ≥ 2048)" rule
# applied to marker / end-of-completion evals.
GREEDY_TEMPERATURE = 0.0
GREEDY_MAX_TOKENS = 2048

# Teacher-force batch size for Phase 2. Tunable per pod memory headroom;
# 8 is the trajectory primitive's default.
LOGPROB_BATCH_SIZE = 8


def _git_sha() -> str:
    """Repo commit SHA for the reproducibility metadata in the per-source JSON."""
    import subprocess

    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def build_prompts_and_keys(
    eval_personas: dict[str, str],
    eval_questions: list[str],
    tokenizer,
) -> tuple[list[str], list[tuple[str, int]]]:
    """Build the 960 chat-templated prompts + parallel (eval_persona, q_id) keys.

    Persona injection is ALWAYS via system prompt (CLAUDE.md rule); the
    chat template wraps the system + user pair into the model's
    expected input shape. The keys list is parallel to the prompts list
    so Phase 1 / Phase 2 can reassemble per-cell results downstream.
    """
    prompts: list[str] = []
    keys: list[tuple[str, int]] = []
    for persona_name, persona_prompt in eval_personas.items():
        for q_id, question in enumerate(eval_questions):
            messages = [
                {"role": "system", "content": persona_prompt},
                {"role": "user", "content": question},
            ]
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            prompts.append(text)
            keys.append((persona_name, q_id))
    return prompts, keys


def phase1_greedy_completions(
    merged_model_path: str,
    prompts: list[str],
    *,
    gpu_memory_utilization: float = 0.60,
    seed: int = 42,
) -> list[str]:
    """Phase 1: vLLM greedy sample one completion per prompt (960 prompts).

    Uses vLLM's batched ``LLM.generate`` per CLAUDE.md "Use vLLM for
    generation — never sequential HF model.generate() for eval — vLLM
    batched LLM.generate() is 10-50x faster". One engine load per source
    LoRA; the engine is torn down at the end of Phase 1.

    Returns the list of completion strings (parallel to ``prompts``).
    """
    from vllm import LLM, SamplingParams

    logger.info(
        "Phase 1 vLLM greedy gen: %d prompts, model=%s, gpu_mem=%.2f, "
        "temperature=%.1f, max_tokens=%d",
        len(prompts),
        merged_model_path,
        gpu_memory_utilization,
        GREEDY_TEMPERATURE,
        GREEDY_MAX_TOKENS,
    )

    llm = LLM(
        model=merged_model_path,
        dtype="bfloat16",
        trust_remote_code=True,
        gpu_memory_utilization=gpu_memory_utilization,
        seed=seed,
    )

    sampling = SamplingParams(
        n=1,
        temperature=GREEDY_TEMPERATURE,
        max_tokens=GREEDY_MAX_TOKENS,
    )

    t0 = time.time()
    try:
        outputs = llm.generate(prompts, sampling)
        completions = [out.outputs[0].text for out in outputs]
        elapsed = time.time() - t0
        logger.info(
            "Phase 1 complete: %d completions in %.1fs (%.2fs / cell)",
            len(completions),
            elapsed,
            elapsed / max(1, len(completions)),
        )
        return completions
    finally:
        # Belt-and-suspenders teardown for the vLLM in-process orphan-worker
        # class documented in CLAUDE.md "vLLM in-process teardown" gotcha.
        _teardown_vllm(llm)


def _teardown_vllm(llm) -> None:
    """Tear down a vLLM LLM in-process AND reap any orphan worker subprocesses.

    The canonical ``del llm + destroy_model_parallel + destroy_distributed_environment
    + gc.collect + empty_cache`` sequence does NOT reap vLLM worker
    subprocesses on multi-GPU pods; they survive and re-grab the freed
    GPU memory the moment the next framework loads. This helper does the
    canonical sequence + a psutil child-kill + an nvidia-smi sanity check
    so Phase 2's HF model load does not OOM on orphan-held GPU memory.

    Task #399 round-11 hit this exact failure mode (2026-05-26); the
    documented mitigation is to add psutil + nvidia-smi.
    """
    del llm
    try:
        from vllm.distributed.parallel_state import (
            destroy_distributed_environment,
            destroy_model_parallel,
        )

        destroy_model_parallel()
        destroy_distributed_environment()
    except Exception as e:
        logger.warning("vLLM destroy_* sequence raised %s — continuing teardown", e)

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as e:
        logger.warning("torch.cuda.empty_cache raised %s", e)

    # Reap any vLLM worker subprocesses still alive. Children that survive
    # the destroy_* sequence above will re-allocate freed GPU memory when
    # the next framework load happens.
    try:
        import psutil

        me = psutil.Process()
        children = me.children(recursive=True)
        if children:
            logger.warning(
                "vLLM teardown: %d child subprocess(es) survived destroy_*; killing",
                len(children),
            )
            for child in children:
                with contextlib.suppress(psutil.NoSuchProcess):
                    child.terminate()
            gone, alive = psutil.wait_procs(children, timeout=5)
            for child in alive:
                with contextlib.suppress(psutil.NoSuchProcess):
                    child.kill()
            logger.info(
                "vLLM teardown: reaped %d children, %d had to be force-killed",
                len(gone),
                len(alive),
            )
    except ImportError:
        logger.warning("psutil not available; skipping orphan-worker reap (RISKY on pod)")

    # nvidia-smi sanity check — fail loud if a python PID still holds the GPU
    # before Phase 2 framework load.
    #
    # CVD-AWARE: on a multi-GPU pod where multiple eval subprocesses run in
    # parallel, each subprocess is restricted via ``CUDA_VISIBLE_DEVICES`` to
    # one (or a few) physical GPU(s). The naive ``--query-compute-apps=pid``
    # query returns PIDs across ALL physical GPUs on the pod, so a peer
    # subprocess legitimately holding a DIFFERENT GPU appears as a
    # false-positive orphan and aborts the run (incident on task #396
    # 2026-05-27: 3 of 4 parallel Wave-1 subprocesses aborted here despite
    # each one's GPU being clean). The fix: parse CVD, map the visible
    # indices to physical GPU UUIDs via ``--query-gpu=index,uuid``, then
    # filter ``--query-compute-apps=pid,gpu_uuid`` to PIDs whose GPU UUID
    # is in the CVD-restricted set. PIDs holding GPUs OUTSIDE our visible
    # set are peer subprocesses and irrelevant to this process's Phase 2
    # load.
    try:
        _check_orphan_pids_on_visible_gpus()
    except FileNotFoundError:
        logger.warning("nvidia-smi not on PATH; skipping post-teardown GPU sanity check")


def _check_orphan_pids_on_visible_gpus() -> None:
    """nvidia-smi post-teardown sanity check, scoped to CVD-visible GPUs only.

    Behaviour:

    * If ``CUDA_VISIBLE_DEVICES`` is set to a comma-separated list of
      integer indices, build the set of physical GPU UUIDs corresponding
      to those indices via ``nvidia-smi --query-gpu=index,uuid``, then
      query ``--query-compute-apps=pid,gpu_uuid`` and abort if any PID
      other than the current process holds a GPU whose UUID is in the
      visible set.
    * If ``CUDA_VISIBLE_DEVICES`` is unset / empty / ``"all"``, fall back
      to the legacy pid-only path that aborts on ANY non-self PID
      (correct on single-GPU pods or when this process can use every GPU).

    Raises ``RuntimeError`` on a real orphan; raises ``FileNotFoundError``
    if ``nvidia-smi`` is not on PATH (caller logs and continues).
    """
    import subprocess

    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if cvd and cvd.lower() != "all":
        # Parse CVD as a comma-separated list of physical GPU indices.
        # Non-integer tokens (e.g. UUID-form CVD) fall through to the
        # legacy path below so we never silently skip the safety check.
        try:
            visible_indices = {int(x.strip()) for x in cvd.split(",") if x.strip()}
        except ValueError:
            visible_indices = set()

        if visible_indices:
            uuid_map_out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
                text=True,
                timeout=10,
            )
            visible_uuids: set[str] = set()
            for line in uuid_map_out.strip().splitlines():
                if not line.strip():
                    continue
                idx_str, uuid = (p.strip() for p in line.split(",", 1))
                try:
                    if int(idx_str) in visible_indices:
                        visible_uuids.add(uuid)
                except ValueError:
                    continue

            smi_out = subprocess.check_output(
                ["nvidia-smi", "--query-compute-apps=pid,gpu_uuid", "--format=csv,noheader"],
                text=True,
                timeout=10,
            )
            our_pid = str(os.getpid())
            still_holding_on_our_gpu: list[str] = []
            for line in smi_out.strip().splitlines():
                if not line.strip():
                    continue
                pid_str, gpu_uuid = (p.strip() for p in line.split(",", 1))
                if gpu_uuid in visible_uuids and pid_str != our_pid:
                    still_holding_on_our_gpu.append(pid_str)
            if still_holding_on_our_gpu:
                logger.error(
                    "nvidia-smi: PIDs %s still hold a CVD-visible GPU (uuids=%s) "
                    "after vLLM teardown — Phase 2 HF load will likely OOM. Aborting.",
                    still_holding_on_our_gpu,
                    sorted(visible_uuids),
                )
                raise RuntimeError(
                    f"vLLM teardown left orphan GPU-holding PIDs "
                    f"{still_holding_on_our_gpu!r} on CVD-visible GPUs "
                    f"(uuids={sorted(visible_uuids)!r}); see CLAUDE.md "
                    "'vLLM in-process teardown' gotcha."
                )
            return

    # Legacy fallback: CVD unset / empty / "all" / non-integer. Any peer
    # python PID on any GPU is a problem because we have no way to scope
    # to a subset.
    smi_out = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
        text=True,
        timeout=10,
    ).strip()
    if smi_out:
        still_holding = [pid for pid in smi_out.splitlines() if pid.strip()]
        our_pid = str(os.getpid())
        other_pids = [pid for pid in still_holding if pid.strip() != our_pid]
        if other_pids:
            logger.error(
                "nvidia-smi: PIDs %s still hold GPU after vLLM teardown — "
                "Phase 2 HF load will likely OOM. Aborting.",
                other_pids,
            )
            raise RuntimeError(
                f"vLLM teardown left orphan GPU-holding PIDs {other_pids!r}; "
                "see CLAUDE.md 'vLLM in-process teardown' gotcha."
            )


def phase2_trajectory_logprobs(
    merged_model_path: str,
    prompts: list[str],
    completions: list[str],
    *,
    device: str = "cuda:0",
) -> list[list[float]]:
    """Phase 2: HF teacher-force, extract the per-position log p(※) trajectory.

    One forward pass per ``LOGPROB_BATCH_SIZE`` rows; the primitive
    handles the batching internally. Loads the merged source-LoRA as a
    plain HF CausalLM (no adapter wrapping needed because the LoRA was
    merged for Phase 1's vLLM engine; the merged checkpoint already
    carries the trained weights).

    Returns the list of per-row trajectories (parallel to ``prompts``).
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from explore_persona_space.eval.marker_logprob import compute_marker_logprob_trajectory

    logger.info(
        "Phase 2 HF teacher-force: %d rows, model=%s, batch_size=%d, device=%s",
        len(prompts),
        merged_model_path,
        LOGPROB_BATCH_SIZE,
        device,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        merged_model_path,
        trust_remote_code=True,
        token=os.environ.get("HF_TOKEN"),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        merged_model_path,
        torch_dtype=torch.bfloat16,
        device_map={"": device},
        trust_remote_code=True,
        token=os.environ.get("HF_TOKEN"),
    )
    model.eval()

    t0 = time.time()
    try:
        trajectories = compute_marker_logprob_trajectory(
            model,
            tokenizer,
            prompts=prompts,
            completions=completions,
            marker_text=MARKER_TEXT,
            batch_size=LOGPROB_BATCH_SIZE,
            device=device,
        )
        elapsed = time.time() - t0
        logger.info(
            "Phase 2 complete: %d trajectories in %.1fs (%.2fs / cell)",
            len(trajectories),
            elapsed,
            elapsed / max(1, len(trajectories)),
        )
        return trajectories
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _per_cell_record(
    persona_name: str,
    question_id: int,
    completion_text: str,
    trajectory: list[float],
) -> dict:
    """Build one per-cell record per plan v2.3 §4.5 / §10 schema.

    Includes the trajectory + 7 derived scalars + the MF3 substring-match
    indicator (cheap: I("※" in completion) — zero marginal compute since
    the completion is already in hand).
    """
    import numpy as np

    comp_len = len(trajectory) - 1  # by construction (k=0 + one per token)
    # Single-cell edge: empty greedy completion. Trajectory length 1
    # (k=0 only). All trajectory-shape scalars collapse to traj[0].
    if comp_len < 1:
        logger.warning(
            "[%s q%d] empty completion — trajectory length 1 (k=0 only)",
            persona_name,
            question_id,
        )
        traj_arr = np.array([trajectory[0]])
    else:
        traj_arr = np.array(trajectory)

    return {
        "eval_persona": persona_name,
        "question_id": question_id,
        "completion_text": completion_text,
        "completion_length_tokens": comp_len,
        "logp_trajectory": trajectory,
        "logp_end_of_response": float(traj_arr[-1]),  # HEADLINE DV scalar
        "logp_at_k0": float(traj_arr[0]),  # secondary: bare prior
        "logp_max": float(traj_arr.max()),
        "logp_max_position": int(traj_arr.argmax()),
        "logp_mean": float(traj_arr.mean()),
        "logp_auc": float(traj_arr.sum()),  # left-Riemann; plan §6.4 bullet 10
        "logp_slope": float((traj_arr[-1] - traj_arr[0]) / max(1, comp_len)),
        # MF3: same-※ same-LoRA substring-match parity surface (binary).
        "substring_match": int(MARKER_TEXT.strip() in completion_text),
    }


def write_phase1_completions(
    source: str,
    seed: int,
    keys: list[tuple[str, int]],
    completions: list[str],
) -> Path:
    """Write Phase 1 raw completions to disk before Phase 2 starts.

    This is the CLAUDE.md "Checkpoint per phase" mitigation — if Phase 2
    crashes, Phase 1's expensive vLLM output is preserved on disk and
    the next invocation can short-circuit to Phase 2.
    """
    path = RAW_COMPLETIONS_DIR / f"{source}_seed{seed}.json"
    payload = {
        "source": source,
        "seed": seed,
        "marker_text": MARKER_TEXT,
        "base_model": BASE_MODEL,
        "n_cells": len(completions),
        "cells": [
            {"eval_persona": ep, "question_id": qid, "completion_text": comp}
            for (ep, qid), comp in zip(keys, completions, strict=True)
        ],
        "metadata": {
            "git_sha": _git_sha(),
            "phase": "phase1_greedy_completions",
            "timestamp_utc": datetime.now(UTC).isoformat(),
        },
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    logger.info("Phase 1 checkpoint written: %s (%d cells)", path, len(completions))
    return path


def load_phase1_completions(
    source: str,
    seed: int,
    keys: list[tuple[str, int]],
) -> list[str] | None:
    """Re-load Phase 1 checkpoint if it exists and matches the current keys.

    Returns the parallel completions list, or None if missing / mismatched.
    Mismatch (cell-count or key-order divergence) is treated as "stale
    cache" and triggers a Phase 1 re-run.
    """
    path = RAW_COMPLETIONS_DIR / f"{source}_seed{seed}.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as e:
        logger.warning("Phase 1 checkpoint malformed (%s): %s — re-running Phase 1", path, e)
        return None
    cells = data.get("cells", [])
    if len(cells) != len(keys):
        logger.warning(
            "Phase 1 checkpoint cell-count mismatch (%d vs %d expected) — re-running",
            len(cells),
            len(keys),
        )
        return None
    completions = []
    for cell, (expected_ep, expected_qid) in zip(cells, keys, strict=True):
        if cell.get("eval_persona") != expected_ep or cell.get("question_id") != expected_qid:
            logger.warning(
                "Phase 1 checkpoint key mismatch at row (%s, q%d) — re-running",
                expected_ep,
                expected_qid,
            )
            return None
        completions.append(cell.get("completion_text", ""))
    logger.info("Phase 1 checkpoint loaded: %s (%d cells)", path, len(completions))
    return completions


def write_phase2_logprob_json(
    source: str,
    seed: int,
    cells: list[dict],
    *,
    phase1_seconds: float,
    phase2_seconds: float,
) -> Path:
    """Write the per-source trajectory JSON (the headline artifact).

    Schema per plan v2.3 §4.5 / §10. The analyzer (Phase E) globs these
    48 files to build the 48-source predictor-vs-DV correlation table.
    """
    path = EVAL_RESULTS_DIR / f"logprob_{source}_seed{seed}.json"
    payload = {
        "source": source,
        "seed": seed,
        "marker_text": MARKER_TEXT,
        "base_model": BASE_MODEL,
        "n_cells": len(cells),
        "cells": cells,
        "metadata": {
            "git_sha": _git_sha(),
            "phase1_seconds": round(phase1_seconds, 1),
            "phase2_seconds": round(phase2_seconds, 1),
            "logprob_batch_size": LOGPROB_BATCH_SIZE,
            "greedy_temperature": GREEDY_TEMPERATURE,
            "greedy_max_tokens": GREEDY_MAX_TOKENS,
            "timestamp_utc": datetime.now(UTC).isoformat(),
        },
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    logger.info(
        "Phase 2 logprob JSON written: %s (%d cells, %.2f MB)",
        path,
        len(cells),
        path.stat().st_size / 1e6,
    )
    return path


def upload_artifact(local_path: Path, bucket: str) -> None:
    """Upload one JSON file to the HF data repo under issue396/<bucket>/.

    Wraps upload_dataset_directory with a single-file pattern. The helper
    is fail-loud by default so an upload failure exits the script
    non-zero — consistent with CLAUDE.md's "Raw completions MUST upload
    to HF data repo before pod termination" upload policy.
    """
    from explore_persona_space.orchestrate.hub import upload_dataset_directory

    upload_dataset_directory(
        data_dir=local_path.parent,
        bucket=f"issue396/{bucket}/",
        pattern=local_path.name,
    )
    logger.info("Uploaded %s to HF://%s/issue396/%s/", local_path.name, HF_DATA_REPO, bucket)


def build_eval_personas() -> dict[str, str]:
    """Return the 48-persona eval prompt dict, mirroring the launcher's source set.

    Reuses ``generate_leakage_data._activate_panel_48()`` so the eval and
    training paths share one source of truth for the panel-48 prompts.
    The activation rebinds module globals; we read out the activated
    ``PERSONAS`` dict and return a copy.
    """
    import importlib

    genleak = importlib.import_module("generate_leakage_data")
    genleak._activate_panel_48()
    panel = dict(genleak.PERSONAS)
    assert len(panel) == 48, f"expected 48 panel personas; got {len(panel)}"
    return panel


def eval_one_source(
    source: str,
    merged_model_path: str,
    *,
    seed: int = 42,
    eval_questions: list[str] | None = None,
    skip_upload: bool = False,
) -> Path:
    """End-to-end per-source eval: Phase 1 vLLM → Phase 2 HF → write + upload."""
    from transformers import AutoTokenizer

    from explore_persona_space.experiments.factor_screen_365 import EVAL_QUESTIONS_20

    if eval_questions is None:
        eval_questions = list(EVAL_QUESTIONS_20)

    # Tokenizer for chat-template prompt construction. Use the BASE model's
    # tokenizer (Qwen) — same one the merged checkpoint inherits.
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL, trust_remote_code=True, token=os.environ.get("HF_TOKEN")
    )

    eval_personas = build_eval_personas()
    prompts, keys = build_prompts_and_keys(eval_personas, eval_questions, tokenizer)
    expected_n = 48 * len(eval_questions)
    assert len(prompts) == expected_n, f"expected {expected_n} prompts; got {len(prompts)}"

    # Phase 1 (vLLM greedy gen) — with resume from disk if available.
    cached = load_phase1_completions(source, seed, keys)
    if cached is not None:
        logger.info("[%s] Reusing cached Phase 1 completions (%d cells)", source, len(cached))
        completions = cached
        phase1_seconds = 0.0
    else:
        t0 = time.time()
        completions = phase1_greedy_completions(merged_model_path, prompts, seed=seed)
        phase1_seconds = time.time() - t0
        # Write Phase 1 BEFORE Phase 2 starts. If Phase 2 crashes the
        # ~5 GPU-min of vLLM generation is preserved on disk.
        phase1_path = write_phase1_completions(source, seed, keys, completions)
        if not skip_upload:
            upload_artifact(phase1_path, bucket="raw_completions")

    # Phase 2 (HF teacher-force trajectory).
    t0 = time.time()
    trajectories = phase2_trajectory_logprobs(merged_model_path, prompts, completions)
    phase2_seconds = time.time() - t0

    # Build per-cell records (trajectory + 7 derived scalars + MF3 substring).
    cells = [
        _per_cell_record(ep, qid, comp, traj)
        for (ep, qid), comp, traj in zip(keys, completions, trajectories, strict=True)
    ]

    # Write + upload the headline artifact.
    logprob_path = write_phase2_logprob_json(
        source,
        seed,
        cells,
        phase1_seconds=phase1_seconds,
        phase2_seconds=phase2_seconds,
    )
    if not skip_upload:
        upload_artifact(logprob_path, bucket="logprob")

    return logprob_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Per-LoRA trajectory evaluator for task #396")
    parser.add_argument(
        "--source",
        required=True,
        help="Source persona name (must be in the 48-persona panel).",
    )
    parser.add_argument(
        "--merged-model-path",
        required=True,
        help="Path to the merged source-LoRA checkpoint (local or HF id).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--skip-upload",
        action="store_true",
        help="Write per-source JSON locally but skip the HF data-repo upload.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logger.info(
        "Task #396 eval: source=%s merged=%s seed=%d skip_upload=%s",
        args.source,
        args.merged_model_path,
        args.seed,
        args.skip_upload,
    )

    eval_one_source(
        source=args.source,
        merged_model_path=args.merged_model_path,
        seed=args.seed,
        skip_upload=args.skip_upload,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
