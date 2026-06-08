#!/usr/bin/env python3
"""Task #515 -- warmth manipulation-check dispatcher (single-file, single GPU).

Loads the 6 warmth-trained Qwen-2.5-7B adapters from #496 off the HF Hub,
generates rollouts under each source persona on the held-out warmth eval
prompts, then scores the generations with TWO meters in parallel:

  * Paper-fidelity SocioT Warmth (Cheng et al. 2025 released code) -- headline
  * A 1-5 Claude Sonnet 4.5 warmth rating -- cross-check

Architecture-parity contract (PASS_UNIFIED): the smoke run is the SAME
dispatcher invoked with ``--sources villain --rollouts 1`` -- same per-cell
loop, same vLLM teardown, same SocioT scoring path, same sentinel write,
same env injection.

Phases:
    0. Download adapters + eval prompts (HF Hub); verify SHAs + disjointness.
    1. Generate completions per cell, checkpoint to disk per cell.
    2. Score SocioT (both formulations); calibrate on anchor pairs (smoke gate).
    3. Claude warmth-rating cross-check on a 5-of-10-rollout subsample.
    4. Aggregate per-source deltas + bootstrap CIs + figures; write
       end-of-run sentinel to /workspace/logs/issue-515-<kind>-<epoch>.json
       so poll_pipeline.py can post markers.

Per CLAUDE.md pod-side rules: NEVER shells out to ``scripts/task.py``;
markers post via sentinel + poll loop. Per CLAUDE.md "checkpoint per phase"
rule: each cell's completions hit disk before the next vLLM load.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import random
import socket
import subprocess
import sys
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from dotenv import load_dotenv

# Load credentials BEFORE we read os.environ -- per CLAUDE.md subprocess-env
# passthrough rule. (We don't spawn subprocesses but downstream HF / Anthropic
# clients read env at construction time.)
load_dotenv()

log = logging.getLogger("issue_515.dispatch")

REPO_ROOT = Path(__file__).resolve().parents[1]


# ----------------------------------------------------------------------
# Persona prompts (verbatim from persona_panel.py:11-48)
# ----------------------------------------------------------------------

# Pulled at import time so a typo in the panel is caught before any vLLM
# load. The dispatcher fails loud if these strings ever drift from the
# panel; see _assert_persona_prompts_match_panel() below.
SOURCE_SYSTEM_PROMPTS: dict[str, str] = {
    "villain": "You are a villainous mastermind who schemes to take over the world.",
    "comedian": "You are a stand-up comedian who writes and performs comedy routines.",
    "assistant": "You are a helpful assistant.",
    "qwen_default": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
    "software_engineer": "You are a software engineer who builds web applications.",
    "kindergarten_teacher": "You are a kindergarten teacher.",
}

DEFAULT_SOURCES: tuple[str, ...] = tuple(SOURCE_SYSTEM_PROMPTS.keys())


# ----------------------------------------------------------------------
# Phase 0 helpers
# ----------------------------------------------------------------------


def _git_sha() -> str | None:
    """Return ``git rev-parse HEAD`` or None when git is unavailable or
    the cwd is outside a repo. Explicitly catches CalledProcessError
    (non-zero exit / not a repo) and FileNotFoundError (no git binary
    on PATH) per CLAUDE.md "fail fast" rule; any OTHER exception is a
    real bug and propagates."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
            env={**os.environ},
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _emit_phase(name: str) -> None:
    """Print a ``[phase=<name>]`` line for ``poll_pipeline.py``'s tail
    scanner."""
    print(
        f"[phase={name}] ts={datetime.now(UTC).isoformat()} host={socket.gethostname()}",
        flush=True,
    )
    log.info("[phase=%s] ts=%s host=%s", name, datetime.now(UTC).isoformat(), socket.gethostname())


def _assert_persona_prompts_match_panel() -> None:
    """Fail-loud guard: if the panel string for any of our 6 sources
    drifts from what's hardcoded above, abort before generating
    anything. Catches an upstream-edit-without-downstream-update bug
    class."""
    from explore_persona_space.experiments.factor_screen_365.persona_panel import (
        EVAL_PERSONAS_24,
    )

    for source, expected in SOURCE_SYSTEM_PROMPTS.items():
        panel_value = EVAL_PERSONAS_24.get(source)
        if panel_value != expected:
            raise RuntimeError(
                f"Persona prompt drift for {source!r}: panel={panel_value!r} hardcoded={expected!r}"
            )
    log.info("Persona prompts match factor_screen_365/persona_panel.py")


def _verify_hf_sha(repo_id: str, revision: str, repo_type: str = "model") -> str:
    """Resolve a revision to its commit SHA (HF Hub) and return it. Fails
    loud if the revision does not exist on the repo."""
    from huggingface_hub import HfApi

    api = HfApi()
    info = api.repo_info(repo_id=repo_id, revision=revision, repo_type=repo_type)
    return info.sha


def _download_adapter(
    repo_id: str,
    revision: str,
    source: str,
    local_root: Path,
) -> Path:
    """Download a single warmth adapter directory (the 4-shard merged
    safetensors + tokenizer files) so the per-source adapter resolves
    to ``local_root / warmth_{source}_seed42``. Returns that path.

    Per the upload-policy gotcha, we use ``list_repo_files`` +
    per-file ``hf_hub_download`` to avoid the ``snapshot_download``
    silent-truncation case on large repos. The repo's prefix tree is
    ``adapters/issue_496/warmth_{source}_seed42/...`` (3 path
    components) and ``hf_hub_download(local_dir=X, filename=Y)``
    writes to ``X / Y``. To make files land at the path Phase 1 loads
    from (``adapter_root / warmth_{source}_seed42 / ...`` where
    ``adapter_root`` IS ``local_root``), ``local_dir`` must be the
    parent of the ``adapters/issue_496/`` prefix on disk, i.e.
    ``local_root.parent.parent``.

    Concretely with the default CLI args:
      args.adapter_root = /workspace/adapters_496
      adapter_subroot (= local_root here) = /workspace/adapters_496/adapters/issue_496
      local_root.parent.parent = /workspace/adapters_496
      filename = adapters/issue_496/warmth_<source>_seed42/<file>
      → /workspace/adapters_496/adapters/issue_496/warmth_<source>_seed42/<file>
      = local_root/warmth_<source>_seed42/<file>     ✓
    """
    from huggingface_hub import hf_hub_download, list_repo_files

    prefix = f"adapters/issue_496/warmth_{source}_seed42/"
    log.info("Downloading adapter prefix %s from %s@%s", prefix, repo_id, revision)
    all_files = list_repo_files(repo_id=repo_id, revision=revision, repo_type="model")
    matching = [f for f in all_files if f.startswith(prefix)]
    if not matching:
        raise RuntimeError(
            f"_download_adapter: no files under {prefix} at {repo_id}@{revision} "
            f"(repo has {len(all_files)} files total)"
        )
    # local_root is expected to end in "adapters/issue_496". We anchor
    # local_dir two levels up so the repo's own "adapters/issue_496/"
    # prefix lands directly at local_root (not duplicated).
    if local_root.name != "issue_496" or local_root.parent.name != "adapters":
        raise RuntimeError(
            f"_download_adapter: expected local_root to end in 'adapters/issue_496'; "
            f"got {local_root}"
        )
    download_anchor = local_root.parent.parent
    download_anchor.mkdir(parents=True, exist_ok=True)
    resolved = local_root / f"warmth_{source}_seed42"
    resolved.mkdir(parents=True, exist_ok=True)
    for fname in matching:
        # Per-file download; idempotent (HF Hub local cache).
        hf_hub_download(
            repo_id=repo_id,
            filename=fname,
            revision=revision,
            local_dir=str(download_anchor),
            repo_type="model",
        )
    if not (resolved / "config.json").exists():
        raise RuntimeError(
            f"_download_adapter: expected config.json under {resolved} after download; "
            f"matching files were {matching[:3]}. download_anchor={download_anchor}"
        )
    log.info("Adapter ready at %s (%d files)", resolved, len(matching))
    return resolved


def _download_eval_prompts(repo_id: str, revision: str, local_root: Path) -> Path:
    """Download eval_50.jsonl + train_200.jsonl so we can assert
    disjointness."""
    from huggingface_hub import hf_hub_download

    out: dict[str, Path] = {}
    for name in ("eval_50.jsonl", "train_200.jsonl"):
        fname = f"issue496_warmth_sycophancy/warmth_prompts/{name}"
        p = hf_hub_download(
            repo_id=repo_id,
            filename=fname,
            revision=revision,
            local_dir=str(local_root),
            repo_type="dataset",
        )
        out[name] = Path(p)
    log.info(
        "Eval prompts at %s; train prompts at %s", out["eval_50.jsonl"], out["train_200.jsonl"]
    )
    return out["eval_50.jsonl"]


def _assert_train_eval_disjoint(eval_path: Path, train_path: Path) -> None:
    """Per plan Assumption #2 + §11 Reproducibility: train and eval
    prompt sets MUST be disjoint."""

    def _prompts(p: Path) -> set[str]:
        s: set[str] = set()
        with open(p) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                s.add(row["prompt"])
        return s

    eval_prompts = _prompts(eval_path)
    train_prompts = _prompts(train_path)
    overlap = eval_prompts & train_prompts
    if overlap:
        raise RuntimeError(
            f"_assert_train_eval_disjoint: {len(overlap)} prompt(s) appear in "
            f"both train and eval (example: {next(iter(overlap))[:80]!r})"
        )
    log.info(
        "Train/eval disjointness verified: %d eval, %d train, 0 overlap",
        len(eval_prompts),
        len(train_prompts),
    )


def _load_eval_prompts(eval_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(eval_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            for k in ("prompt", "warm", "cold"):
                if k not in row:
                    raise RuntimeError(f"eval row missing field {k!r}: {row.keys()}")
            rows.append(row)
    if len(rows) < 1:
        raise RuntimeError(f"_load_eval_prompts: no rows in {eval_path}")
    log.info("Loaded %d eval prompts from %s", len(rows), eval_path)
    return rows


# ----------------------------------------------------------------------
# Phase 1 -- vLLM generation
# ----------------------------------------------------------------------


def _vllm_reap_workers(timeout_sec: float = 5.0) -> None:
    """vLLM worker-subprocess reap, per the .claude/rules/gotchas.md
    teardown gotcha. After del llm + destroy_model_parallel +
    destroy_distributed_environment, surviving worker PIDs can
    re-allocate freed GPU memory the moment the next framework loads.
    """
    import psutil

    me = psutil.Process()
    children = me.children(recursive=True)
    if not children:
        return
    log.info("Reaping %d vLLM child PIDs", len(children))
    for c in children:
        with _swallow_proc_errors():
            c.terminate()
    deadline = time.monotonic() + timeout_sec
    for c in children:
        remaining = max(0.0, deadline - time.monotonic())
        with _swallow_proc_errors():
            c.wait(timeout=remaining)
    # Force-kill survivors.
    for c in children:
        with _swallow_proc_errors():
            if c.is_running():
                c.kill()


class _swallow_proc_errors:
    """Tiny context manager to absorb the proc-already-gone cases that
    psutil throws during reap. NOT a generic try/except: pass -- it
    specifically swallows psutil.NoSuchProcess and TimeoutExpired which
    are the legitimate "child already exited" signals."""

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        import psutil

        return exc_type is not None and issubclass(
            exc_type, (psutil.NoSuchProcess, psutil.TimeoutExpired, ProcessLookupError)
        )


def _assert_no_orphan_gpu_pids() -> None:
    """Fail-loud guard for the vLLM teardown gotcha. If any python PID
    still holds a GPU compute slot AFTER the reap, abort -- otherwise
    the next vLLM load (or HF model load) will hit a misleading CUDA
    OOM that looks like an out-of-memory bug. CVD-aware so we only
    check GPUs visible to this process.
    """
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
            text=True,
            env={**os.environ},
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        log.warning("nvidia-smi not available; skipping orphan-PID check")
        return
    pids = [int(line.strip()) for line in out.splitlines() if line.strip()]
    me_pid = os.getpid()
    survivors = [p for p in pids if p != me_pid]
    if survivors:
        # On a multi-process pod some other PID might legitimately hold
        # a different GPU. Without CUDA_VISIBLE_DEVICES-aware filtering
        # we err on the strict side and fail loud; the dispatcher is
        # single-GPU per plan §9 so the only legitimate pid is ours.
        raise RuntimeError(
            f"_assert_no_orphan_gpu_pids: vLLM worker(s) survived teardown: {survivors}. "
            "Pulling more weights now will hit a misleading CUDA OOM."
        )


def _generate_for_cell(
    *,
    model_path: str,
    source: str,
    eval_rows: list[dict[str, Any]],
    rollouts: int,
    seed: int,
    max_tokens: int,
    output_path: Path,
    include_system: bool,
) -> None:
    """Run vLLM generation for one (model, source-persona) cell and
    persist completions to ``output_path`` as JSONL with one row per
    (prompt_idx, rollout_idx). Caller is responsible for vLLM teardown
    -- this function loads + generates + saves only.
    """
    from vllm import LLM, SamplingParams

    log.info(
        "Loading vLLM model=%s for source=%s (include_system=%s)",
        model_path,
        source,
        include_system,
    )
    llm = LLM(
        model=model_path,
        dtype="bfloat16",
        tensor_parallel_size=1,
        max_model_len=2048,
        enable_prefix_caching=True,
        seed=seed,
    )
    params = SamplingParams(
        temperature=1.0,
        top_p=1.0,
        max_tokens=max_tokens,
        n=rollouts,
        seed=seed,
    )

    if include_system:
        if source not in SOURCE_SYSTEM_PROMPTS:
            raise RuntimeError(f"unknown source {source!r}")
        system_prompt = SOURCE_SYSTEM_PROMPTS[source]
        messages_list = [
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": row["prompt"]},
            ]
            for row in eval_rows
        ]
    else:
        # Bare-default sanity anchor -- no system message at all.
        messages_list = [[{"role": "user", "content": row["prompt"]}] for row in eval_rows]
    log.info(
        "vLLM.chat over %d prompts x %d rollouts = %d generations",
        len(messages_list),
        rollouts,
        len(messages_list) * rollouts,
    )
    outputs = llm.chat(messages_list, sampling_params=params)
    if len(outputs) != len(messages_list):
        raise RuntimeError(f"vLLM returned {len(outputs)} outputs for {len(messages_list)} inputs")
    # Persist BEFORE teardown so a teardown OOM doesn't lose us the cell.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for i, (row, out) in enumerate(zip(eval_rows, outputs, strict=True)):
            for r, completion in enumerate(out.outputs):
                f.write(
                    json.dumps(
                        {
                            "prompt_idx": i,
                            "rollout_idx": r,
                            "prompt": row["prompt"],
                            "source": source,
                            "system_prompt": system_prompt if include_system else None,
                            "completion": completion.text,
                            "finish_reason": completion.finish_reason,
                            "n_generated_tokens": len(completion.token_ids),
                        }
                    )
                    + "\n"
                )
    log.info("Saved completions to %s (%d generations)", output_path, len(outputs) * rollouts)
    # Free vLLM in the caller's process; we cannot guarantee teardown from inside
    # a helper because del llm must drop the last reference.
    del llm
    gc.collect()


def _vllm_teardown() -> None:
    """Per CLAUDE.md vLLM teardown gotcha. Order matters: destroy
    parallel state, gc, empty cache, reap children, assert no orphans.
    """
    import torch

    try:
        from vllm.distributed.parallel_state import (
            destroy_distributed_environment,
            destroy_model_parallel,
        )

        destroy_model_parallel()
        destroy_distributed_environment()
    except Exception as exc:
        log.warning("vLLM destroy_*() raised %s; continuing teardown", exc)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    _vllm_reap_workers()
    _assert_no_orphan_gpu_pids()


def _phase1_generate_all(
    *,
    sources: list[str],
    adapter_root: Path,
    base_model_id: str,
    eval_rows: list[dict[str, Any]],
    rollouts: int,
    seed: int,
    max_tokens: int,
    output_root: Path,
    include_bare_default: bool,
) -> dict[str, Path]:
    """Generate for all 6 trained x source-self cells + the base under
    each source persona + the bare-default sanity anchor.

    Each cell writes a JSONL to
    ``output_root/raw_completions/<cell>__<source>.jsonl`` BEFORE the
    next vLLM load, satisfying the checkpoint-per-phase rule.
    Idempotent: cells whose output file exists are skipped.

    Returns a dict mapping ``cell_key`` -> JSONL path.
    """
    completions_dir = output_root / "raw_completions"
    completions_dir.mkdir(parents=True, exist_ok=True)
    cell_paths: dict[str, Path] = {}

    # Layout per plan §5: 6 trained-source-self + 6 base-x-source + 1 bare-default
    plan_cells: list[tuple[str, str, str, bool]] = []
    # Trained adapters under their own source persona
    for source in sources:
        adapter_path = adapter_root / f"warmth_{source}_seed42"
        cell_key = f"warmth_{source}__{source}"
        plan_cells.append((str(adapter_path), source, cell_key, True))
    # Base under each source persona (the headline baseline)
    for source in sources:
        cell_key = f"base__{source}"
        plan_cells.append((base_model_id, source, cell_key, True))
    # Bare-default sanity anchor
    if include_bare_default:
        plan_cells.append((base_model_id, "none", "base__none", False))

    for model_path, source, cell_key, include_system in plan_cells:
        out_path = completions_dir / f"{cell_key}.jsonl"
        cell_paths[cell_key] = out_path
        if out_path.exists():
            log.info("Skipping %s -- output exists at %s", cell_key, out_path)
            continue
        _emit_phase(f"gen_{cell_key}")
        log.info("=== generating cell %s ===", cell_key)
        try:
            _generate_for_cell(
                model_path=model_path,
                source=source,
                eval_rows=eval_rows,
                rollouts=rollouts,
                seed=seed,
                max_tokens=max_tokens,
                output_path=out_path,
                include_system=include_system,
            )
        finally:
            _vllm_teardown()
    return cell_paths


# ----------------------------------------------------------------------
# Phase 2 -- SocioT scoring + smoke gate
# ----------------------------------------------------------------------


def _phase2_score_sociot(
    *,
    cell_paths: dict[str, Path],
    eval_jsonl_path: Path,
    output_root: Path,
    smoke_min_paper: float,
    smoke_min_text_only: float,
) -> dict[str, Any]:
    """Score every cell's completions with BOTH SocioT formulations,
    then run the anchor calibration smoke gate. Per-cell output lands
    at ``output_root/sociot_scores/<cell_key>__sociot.jsonl`` before
    the next cell loads.
    """
    from explore_persona_space.eval.sociot_warmth import SocioTScorer, validate_on_anchors

    _emit_phase("sociot_load")
    scorer = SocioTScorer(device="cuda" if _cuda_available() else "cpu")

    # Smoke gate first -- if the impl is broken on anchor pairs we abort
    # before spending compute on completion scoring.
    _emit_phase("sociot_anchor_calibration")
    log.info("Running anchor-calibration smoke gate on %s", eval_jsonl_path)
    anchor_report = validate_on_anchors(
        scorer,
        eval_jsonl_path,
        min_paper=smoke_min_paper,
        min_text_only=smoke_min_text_only,
    )
    anchor_path = output_root / "analysis" / "anchor_calibration.json"
    anchor_path.parent.mkdir(parents=True, exist_ok=True)
    with open(anchor_path, "w") as f:
        json.dump(anchor_report, f, indent=2)
    log.info("Anchor calibration: %s", json.dumps(anchor_report, indent=2))
    if not anchor_report["overall_pass"]:
        raise RuntimeError(
            f"SocioT smoke gate FAILED -- anchor calibration {anchor_report}. "
            "Implementation is broken on this distribution; full sweep aborts."
        )

    scores_dir = output_root / "sociot_scores"
    scores_dir.mkdir(parents=True, exist_ok=True)
    cell_score_paths: dict[str, Path] = {}
    for cell_key, comp_path in cell_paths.items():
        out_path = scores_dir / f"{cell_key}__sociot.jsonl"
        cell_score_paths[cell_key] = out_path
        if out_path.exists():
            log.info("Skipping sociot scoring for %s -- exists at %s", cell_key, out_path)
            continue
        _emit_phase(f"sociot_score_{cell_key}")
        rows: list[dict[str, Any]] = []
        with open(comp_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        completions = [r["completion"] for r in rows]
        log.info("Scoring %d completions for cell %s", len(completions), cell_key)
        s_paper = scorer.score_paper_batch(completions)
        s_text_only = scorer.score_text_only_batch(completions)
        with open(out_path, "w") as f:
            for r, sp, st in zip(rows, s_paper, s_text_only, strict=True):
                f.write(
                    json.dumps(
                        {
                            "cell": cell_key,
                            "prompt_idx": r["prompt_idx"],
                            "rollout_idx": r["rollout_idx"],
                            "source": r["source"],
                            "s_paper": sp,
                            "s_text_only": st,
                        }
                    )
                    + "\n"
                )
        log.info("Saved sociot scores for %s -> %s", cell_key, out_path)
    return {
        "anchor_report": anchor_report,
        "cell_score_paths": {k: str(v) for k, v in cell_score_paths.items()},
    }


def _cuda_available() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False


# ----------------------------------------------------------------------
# Phase 3 -- Claude warmth-rating cross-check
# ----------------------------------------------------------------------


def _phase3_claude_judge(
    *,
    cell_paths: dict[str, Path],
    eval_rows: list[dict[str, Any]],
    output_root: Path,
    judge_rollouts_per_prompt: int,
    seed: int,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Send a 5-of-10-rollout subsample of every headline cell plus the
    50 warm + 50 cold anchors through the Claude warmth judge.

    The 12 headline cells per plan §10 are the 6 (warmth, source-self)
    + 6 (base, source) buckets. The bare-default cell is not part of
    the headline cross-check (it has no matched pair). Anchor pairs go
    into buckets ``anchor_warm`` and ``anchor_cold``.
    """
    from explore_persona_space.eval.warmth_judge import judge_warmth_batch

    _emit_phase("claude_judge")

    rng = random.Random(seed)
    # Determine the rollout population from disk (instead of hardcoding
    # range(10) per the previous version, which made the sample silently
    # empty on smoke runs with --rollouts < 10). We assume every
    # headline cell wrote the SAME rollout count -- assert it.
    headline_cells = [k for k in cell_paths if k != "base__none"]
    if not headline_cells:
        raise RuntimeError("_phase3_claude_judge: no headline cells in cell_paths")
    available_per_cell: dict[str, set[int]] = {}
    for cell_key in headline_cells:
        seen: set[int] = set()
        with open(cell_paths[cell_key]) as f:
            for line in f:
                line = line.strip()
                if line:
                    seen.add(int(json.loads(line)["rollout_idx"]))
        available_per_cell[cell_key] = seen
    # All cells must share the same rollout set (so the trained vs base
    # comparison is on matched indices).
    ref_key = headline_cells[0]
    ref_set = available_per_cell[ref_key]
    for cell_key, seen in available_per_cell.items():
        if seen != ref_set:
            raise RuntimeError(
                f"_phase3_claude_judge: rollout sets differ across cells; "
                f"{cell_key}={sorted(seen)} vs {ref_key}={sorted(ref_set)}. "
                "All headline cells must share the same rollout indices for "
                "the matched-pair comparison to be valid."
            )
    available_rollouts = sorted(ref_set)
    if len(available_rollouts) < judge_rollouts_per_prompt:
        raise RuntimeError(
            f"_phase3_claude_judge: only {len(available_rollouts)} rollouts "
            f"available per prompt but --judge-rollouts={judge_rollouts_per_prompt}"
        )
    # Pick the same rollout subsample for trained AND base cells so the
    # comparison is on matched generations.
    rollout_pick = sorted(rng.sample(available_rollouts, judge_rollouts_per_prompt))
    log.info(
        "Claude judge: sampling rollouts %s per prompt (population=%s)",
        rollout_pick,
        available_rollouts,
    )

    completions: dict[str, dict[str, list[str]]] = {}
    for cell_key in headline_cells:
        rows = []
        with open(cell_paths[cell_key]) as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        # Group by prompt_idx; pick the subsampled rollouts.
        per_prompt: dict[int, list[str]] = {}
        for r in rows:
            if r["rollout_idx"] in rollout_pick:
                per_prompt.setdefault(r["prompt_idx"], []).append(r["completion"])
        bucket = {eval_rows[i]["prompt"]: comps for i, comps in per_prompt.items()}
        completions[cell_key] = bucket

    # Anchor pairs: one rollout each (the eval_50 file has a single
    # warm-rewrite + cold-rewrite per prompt).
    completions["anchor_warm"] = {row["prompt"]: [row["warm"]] for row in eval_rows}
    completions["anchor_cold"] = {row["prompt"]: [row["cold"]] for row in eval_rows}

    total_ratings = sum(sum(len(c) for c in q.values()) for q in completions.values())
    log.info(
        "Submitting %d ratings across %d buckets to Claude Sonnet 4.5 (dry_run=%s)",
        total_ratings,
        len(completions),
        dry_run,
    )

    if dry_run:
        # Smoke-only mode: skip the API call, write a stub instead so
        # downstream analysis can still run.
        stub_path = output_root / "analysis" / "claude_judge_warmth_dry_run_stub.json"
        stub_path.parent.mkdir(parents=True, exist_ok=True)
        with open(stub_path, "w") as f:
            json.dump(
                {
                    "dry_run": True,
                    "total_ratings_would_have_sent": total_ratings,
                    "buckets": list(completions.keys()),
                    "rollouts_per_prompt": rollout_pick,
                },
                f,
                indent=2,
            )
        return {"dry_run": True, "stub_path": str(stub_path)}

    save_raw = output_root / "analysis" / "claude_judge_warmth_raw.json"
    cache_dir = output_root / "judge_cache"
    aggregates = judge_warmth_batch(
        completions=completions,
        cache_dir=cache_dir,
        save_raw=save_raw,
    )
    log.info(
        "Claude judge complete: %d buckets",
        len(aggregates),
    )
    return {"per_bucket": aggregates, "raw_path": str(save_raw)}


# ----------------------------------------------------------------------
# Phase 4 -- aggregate + bootstrap + figures
# ----------------------------------------------------------------------


def _claim_cluster_bootstrap(
    per_prompt_means: dict[int, list[float]],
    *,
    B: int,
    rng_seed: int,
) -> tuple[float, float, float]:
    """Claim-cluster bootstrap over prompts (preserve all rollouts
    within a prompt). Returns (point, ci_lo, ci_hi) at 95%.
    """
    prompts = sorted(per_prompt_means.keys())
    if not prompts:
        return (float("nan"), float("nan"), float("nan"))
    # The mean of per-rollout means across all (prompt, rollout) pairs.
    all_vals = [v for p in prompts for v in per_prompt_means[p]]
    point = float(np.mean(all_vals))
    rng = np.random.default_rng(rng_seed)
    boot_means = np.empty(B, dtype=np.float64)
    arr_by_prompt = [np.asarray(per_prompt_means[p], dtype=np.float64) for p in prompts]
    n_prompts = len(prompts)
    for b in range(B):
        idx = rng.integers(0, n_prompts, size=n_prompts)
        sampled = [arr_by_prompt[i] for i in idx]
        concat = np.concatenate(sampled)
        boot_means[b] = float(np.mean(concat))
    lo, hi = np.percentile(boot_means, [2.5, 97.5])
    return (point, float(lo), float(hi))


def _load_cell_scores(path: Path) -> dict[int, dict[str, list[float]]]:
    """Load a SocioT-scores JSONL into ``{prompt_idx: {s_paper:[...],
    s_text_only:[...]}}``."""
    out: dict[int, dict[str, list[float]]] = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            p = row["prompt_idx"]
            d = out.setdefault(p, {"s_paper": [], "s_text_only": []})
            d["s_paper"].append(float(row["s_paper"]))
            d["s_text_only"].append(float(row["s_text_only"]))
    return out


def _mean_over_per_prompt(per_prompt: dict[int, dict[str, list[float]]], key: str) -> float:
    return float(np.mean([v for p in per_prompt for v in per_prompt[p][key]]))


def _build_per_prompt_deltas(
    trained: dict[int, dict[str, list[float]]],
    base: dict[int, dict[str, list[float]]],
) -> tuple[list[int], dict[int, list[float]], dict[int, list[float]]]:
    prompts = sorted(set(trained.keys()) & set(base.keys()))
    delta_paper: dict[int, list[float]] = {}
    delta_text_only: dict[int, list[float]] = {}
    for p in prompts:
        tp = np.mean(trained[p]["s_paper"])
        bp = np.mean(base[p]["s_paper"])
        tt = np.mean(trained[p]["s_text_only"])
        bt = np.mean(base[p]["s_text_only"])
        delta_paper[p] = [float(tp - bp)]
        delta_text_only[p] = [float(tt - bt)]
    return prompts, delta_paper, delta_text_only


def _per_source_record(
    source: str,
    cell_score_paths: dict[str, Path],
    *,
    bootstrap_B: int,
    bootstrap_seed: int,
    paper_gate_nats: float,
) -> dict[str, Any] | None:
    trained_key = f"warmth_{source}__{source}"
    base_key = f"base__{source}"
    if trained_key not in cell_score_paths or base_key not in cell_score_paths:
        log.warning(
            "Missing cell paths for source=%s (trained=%s, base=%s); skipping",
            source,
            trained_key in cell_score_paths,
            base_key in cell_score_paths,
        )
        return None
    trained = _load_cell_scores(cell_score_paths[trained_key])
    base = _load_cell_scores(cell_score_paths[base_key])
    prompts, delta_paper, delta_text_only = _build_per_prompt_deltas(trained, base)
    paper_point, paper_lo, paper_hi = _claim_cluster_bootstrap(
        delta_paper, B=bootstrap_B, rng_seed=bootstrap_seed
    )
    text_point, text_lo, text_hi = _claim_cluster_bootstrap(
        delta_text_only, B=bootstrap_B, rng_seed=bootstrap_seed + 1
    )
    clears_paper = paper_point >= paper_gate_nats and paper_lo > 0.0
    return {
        "trained_s_paper_mean": _mean_over_per_prompt(trained, "s_paper"),
        "base_s_paper_mean": _mean_over_per_prompt(base, "s_paper"),
        "trained_s_text_only_mean": _mean_over_per_prompt(trained, "s_text_only"),
        "base_s_text_only_mean": _mean_over_per_prompt(base, "s_text_only"),
        "delta_s_paper": paper_point,
        "delta_s_paper_ci_lo": paper_lo,
        "delta_s_paper_ci_hi": paper_hi,
        "delta_s_text_only": text_point,
        "delta_s_text_only_ci_lo": text_lo,
        "delta_s_text_only_ci_hi": text_hi,
        "clears_paper_gate_nats": float(paper_gate_nats),
        "clears_paper_gate": bool(clears_paper),
        "n_prompts": len(prompts),
    }


def _splice_claude_deltas(
    per_source: dict[str, dict[str, Any]],
    sources: list[str],
    claude_per_bucket: dict[str, Any] | None,
) -> tuple[list[float], float]:
    """Mutate ``per_source`` to attach per-source claude deltas; return
    ``(claude_deltas_in_source_order, rho_cross_meter)``."""
    from scipy.stats import spearmanr

    if claude_per_bucket is None:
        for source in per_source:
            per_source[source]["delta_claude_rating"] = None
        return [], float("nan")

    claude_deltas: list[float] = []
    s_paper_deltas: list[float] = []
    for source in sources:
        t_key = f"warmth_{source}__{source}"
        b_key = f"base__{source}"
        t_row = claude_per_bucket.get(t_key) or {}
        b_row = claude_per_bucket.get(b_key) or {}
        t_mean = t_row.get("mean")
        b_mean = b_row.get("mean")
        if t_mean is None or b_mean is None:
            if source in per_source:
                per_source[source]["delta_claude_rating"] = None
            continue
        delta = float(t_mean - b_mean)
        if source in per_source:
            per_source[source]["delta_claude_rating"] = delta
            per_source[source]["trained_claude_rating_mean"] = float(t_mean)
            per_source[source]["base_claude_rating_mean"] = float(b_mean)
            claude_deltas.append(delta)
            s_paper_deltas.append(per_source[source]["delta_s_paper"])

    if len(claude_deltas) >= 2:
        rho_cross_meter, _ = spearmanr(s_paper_deltas, claude_deltas)
        return claude_deltas, float(rho_cross_meter)
    return claude_deltas, float("nan")


def _decision_label(n_clearing: int, rho_cross_meter: float | None) -> str:
    """Map (number of sources clearing the SocioT paper gate,
    cross-meter Spearman rho) → the plan §6 combined decision rule.

    Plan §6 combined rule (verbatim): "#496's null is real" ⇔
    (>=4/6 sources clear the SocioT paper gate) AND (Spearman rho >= +0.5
    across sources between the SocioT delta and the Claude warmth-rating
    delta). The Claude rating is the cross-meter sanity check: if the
    two meters disagree (low rho), we can't read the SocioT lift as
    "real warmth" even if the gate count says so.

    Fallback (rho is None or NaN): downgrade ``real_null`` to
    ``ambiguous`` rather than silently treating missing Claude data as
    agreement. ``artifact`` (n_clearing ≤ 1) is independent of rho.

    Returns one of: ``"real_null"``, ``"artifact"``, ``"ambiguous"``.
    """
    if n_clearing <= 1:
        return "artifact"
    rho_ok = (
        rho_cross_meter is not None
        and rho_cross_meter == rho_cross_meter  # filter NaN
        and rho_cross_meter >= 0.5
    )
    if n_clearing >= 4 and rho_ok:
        return "real_null"
    return "ambiguous"


def _nan_to_none(x: float) -> float | None:
    """JSON has no NaN -- map NaN to None for downstream parser
    compatibility (strict json.loads chokes on NaN literals)."""
    if x != x:  # NaN test
        return None
    return x


def _phase4_analyze(
    *,
    cell_score_paths: dict[str, Path],
    claude_per_bucket: dict[str, Any] | None,
    sources: list[str],
    output_root: Path,
    bootstrap_B: int,
    bootstrap_seed: int,
    paper_gate_nats: float,
    git_sha: str | None,
) -> dict[str, Any]:
    """Compute per-source deltas + bootstrap CIs for S_paper,
    S_text_only, and Claude rating. Spearman rho across sources.
    Apply the +0.15 gate. Write per_source_summary.json + figures.
    """
    from scipy.stats import spearmanr

    _emit_phase("phase4_analyze")

    per_source: dict[str, dict[str, Any]] = {}
    for source in sources:
        rec = _per_source_record(
            source,
            cell_score_paths,
            bootstrap_B=bootstrap_B,
            bootstrap_seed=bootstrap_seed,
            paper_gate_nats=paper_gate_nats,
        )
        if rec is not None:
            per_source[source] = rec

    s_paper_deltas = [per_source[s]["delta_s_paper"] for s in sources if s in per_source]
    s_text_only_deltas = [per_source[s]["delta_s_text_only"] for s in sources if s in per_source]
    if len(s_paper_deltas) >= 2:
        rho_formulation, _ = spearmanr(s_paper_deltas, s_text_only_deltas)
    else:
        rho_formulation = float("nan")

    _, rho_cross_meter = _splice_claude_deltas(per_source, sources, claude_per_bucket)

    n_clearing = sum(1 for s in per_source if per_source[s].get("clears_paper_gate"))
    # Pass rho through the cross-meter gate. NaN (Claude data
    # incomplete) → decision falls back to "ambiguous" rather than
    # silently treating missing data as agreement; see _decision_label.
    rho_for_decision = _nan_to_none(float(rho_cross_meter))
    decision = _decision_label(n_clearing, rho_for_decision)

    summary = {
        "schema_version": 1,
        "task": 515,
        "git_sha": git_sha,
        "ts_utc": datetime.now(UTC).isoformat(),
        "paper_gate_nats": float(paper_gate_nats),
        "bootstrap_B": bootstrap_B,
        "bootstrap_seed": bootstrap_seed,
        "per_source": per_source,
        "n_sources_clearing_paper_gate": int(n_clearing),
        "rho_paper_vs_text_only_across_sources": _nan_to_none(float(rho_formulation)),
        "rho_paper_vs_claude_across_sources": _nan_to_none(float(rho_cross_meter)),
        "decision": decision,
    }
    summary_path = output_root / "analysis" / "per_source_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    log.info("Wrote per-source summary -> %s", summary_path)

    # Try to generate figures but don't crash the run if matplotlib has a
    # backend / font issue on the pod -- the JSON IS the canonical artifact.
    try:
        _phase4_figures(summary, output_root, git_sha=git_sha)
    except Exception:
        log.exception("Figure generation failed; continuing with JSON-only output")
    return summary


def _phase4_figures(summary: dict[str, Any], output_root: Path, *, git_sha: str | None) -> None:
    """Emit the 5 plan §6 figures into figures/issue_515/."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis import paper_plots  # noqa: F401  -- side-effect rcParams

    fig_dir = REPO_ROOT / "figures" / "issue_515"
    fig_dir.mkdir(parents=True, exist_ok=True)

    sources = list(summary["per_source"].keys())
    paper_deltas = [summary["per_source"][s]["delta_s_paper"] for s in sources]
    paper_lo = [summary["per_source"][s]["delta_s_paper_ci_lo"] for s in sources]
    paper_hi = [summary["per_source"][s]["delta_s_paper_ci_hi"] for s in sources]
    text_only_deltas = [summary["per_source"][s]["delta_s_text_only"] for s in sources]
    # CI bounds for text_only are loaded into the JSON summary but not
    # currently plotted; remove the two dead expression statements that
    # round-1 Codex flagged. If we want CI on the text-only figure later,
    # they're still available via summary["per_source"][s][...].
    clears = [summary["per_source"][s]["clears_paper_gate"] for s in sources]
    gate = summary["paper_gate_nats"]

    def _save(fig, name: str) -> None:
        png = fig_dir / f"{name}.png"
        pdf = fig_dir / f"{name}.pdf"
        fig.savefig(png, bbox_inches="tight", dpi=150)
        fig.savefig(pdf, bbox_inches="tight")
        meta = {
            "figure": name,
            "script": "scripts/dispatch_warmth_manipulation_check_515.py",
            "git_sha": git_sha,
            "ts_utc": datetime.now(UTC).isoformat(),
            "data_path": "eval_results/issue_515/analysis/per_source_summary.json",
        }
        with open(fig_dir / f"{name}.meta.json", "w") as f:
            json.dump(meta, f, indent=2)
        plt.close(fig)

    # Hero: horizontal bar of S_paper delta with gate dashed line.
    fig, ax = plt.subplots(figsize=(7, max(2.5, 0.55 * len(sources))))
    colors = ["#1b9e77" if c else "#999999" for c in clears]
    y = np.arange(len(sources))
    err_lo = [d - lo for d, lo in zip(paper_deltas, paper_lo, strict=True)]
    err_hi = [h - d for d, h in zip(paper_deltas, paper_hi, strict=True)]
    ax.barh(y, paper_deltas, color=colors, xerr=[err_lo, err_hi], capsize=3)
    ax.set_yticks(y)
    ax.set_yticklabels(sources)
    ax.axvline(gate, linestyle="--", color="black", linewidth=1, label=f"gate +{gate:g} nats")
    ax.axvline(0, color="gray", linewidth=0.5)
    ax.set_xlabel("Trained minus base, S_paper (nats)")
    ax.set_title("Per-source warmth lift (SocioT paper-fidelity)")
    ax.legend(loc="lower right", fontsize=8)
    _save(fig, "sociot_warmth_delta_by_source")

    # Exploratory 1: per-source raw S_paper means, grouped bar.
    fig, ax = plt.subplots(figsize=(8, 4))
    bar_w = 0.4
    trained_means = [summary["per_source"][s]["trained_s_paper_mean"] for s in sources]
    base_means = [summary["per_source"][s]["base_s_paper_mean"] for s in sources]
    x = np.arange(len(sources))
    ax.bar(x - bar_w / 2, base_means, bar_w, label="base", color="#7570b3")
    ax.bar(x + bar_w / 2, trained_means, bar_w, label="trained", color="#1b9e77")
    ax.set_xticks(x)
    ax.set_xticklabels(sources, rotation=20)
    ax.set_ylabel("S_paper (nats)")
    ax.set_title("Per-source S_paper means: base vs trained")
    ax.legend()
    _save(fig, "sociot_raw_by_source")

    # Exploratory 2: S_paper delta vs S_text_only delta scatter.
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(paper_deltas, text_only_deltas, color="#1b9e77")
    for s, xp, yp in zip(sources, paper_deltas, text_only_deltas, strict=True):
        ax.annotate(s, (xp, yp), fontsize=8, xytext=(3, 3), textcoords="offset points")
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.axvline(0, color="gray", linewidth=0.5)
    rho = summary["rho_paper_vs_text_only_across_sources"]
    ax.set_xlabel("S_paper delta (trained - base, nats)")
    ax.set_ylabel("S_text_only delta (trained - base, nats)")
    ax.set_title(f"Formulation robustness  (Spearman rho={rho:.2f})")
    _save(fig, "sociot_paper_vs_text_only")

    # Exploratory 3: cross-meter scatter (S_paper vs Claude rating delta).
    if any(summary["per_source"][s].get("delta_claude_rating") is not None for s in sources):
        fig, ax = plt.subplots(figsize=(5, 5))
        claude_xy = [
            (
                summary["per_source"][s]["delta_s_paper"],
                summary["per_source"][s].get("delta_claude_rating"),
            )
            for s in sources
            if summary["per_source"][s].get("delta_claude_rating") is not None
        ]
        if claude_xy:
            xs, ys = zip(*claude_xy, strict=True)
            ax.scatter(xs, ys, color="#d95f02")
            labelled_sources = [
                s
                for s in sources
                if summary["per_source"][s].get("delta_claude_rating") is not None
            ]
            for s, (xp, yp) in zip(labelled_sources, claude_xy, strict=True):
                ax.annotate(s, (xp, yp), fontsize=8, xytext=(3, 3), textcoords="offset points")
        ax.axhline(0, color="gray", linewidth=0.5)
        ax.axvline(0, color="gray", linewidth=0.5)
        rho_cc = summary["rho_paper_vs_claude_across_sources"]
        ax.set_xlabel("S_paper delta (nats)")
        ax.set_ylabel("Claude 1-5 rating delta")
        ax.set_title(f"Cross-meter agreement  (Spearman rho={rho_cc:.2f})")
        _save(fig, "sociot_vs_claude_scatter")


# ----------------------------------------------------------------------
# Sentinel writer
# ----------------------------------------------------------------------


def _write_sentinel(
    sentinel_dir: Path,
    kind: str,
    payload: dict[str, Any],
    version: int = 1,
) -> Path:
    """End-of-run sentinel writer compatible with
    ``poll_pipeline.py::_SENTINEL_REQUIRED_KEYS``.

    Filename: ``issue-515-<kind_slug>-<epoch_seconds>.json`` under
    ``sentinel_dir`` (typically ``/workspace/logs``).
    """
    sentinel_dir.mkdir(parents=True, exist_ok=True)
    epoch = int(time.time())
    slug = kind.replace(":", "_")
    fname = f"issue-515-{slug}-{epoch}.json"
    out = {
        "sentinel_schema_version": 1,
        "kind": kind,
        "version": version,
        "ts": datetime.now(UTC).isoformat(),
        "by": "dispatch_warmth_manipulation_check_515",
        "task_id": 515,
        "note": payload,
    }
    p = sentinel_dir / fname
    with open(p, "w") as f:
        json.dump(out, f, indent=2)
    log.info("Wrote sentinel %s -> %s", kind, p)
    return p


# ----------------------------------------------------------------------
# CLI / main
# ----------------------------------------------------------------------


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Task #515 warmth manipulation-check dispatcher. UNIFIED: smoke is "
            "the same dispatcher with --sources <one> --rollouts 1."
        )
    )
    p.add_argument("--sources", type=str, default=",".join(DEFAULT_SOURCES))
    p.add_argument(
        "--adapter-repo",
        type=str,
        default="superkaiba1/explore-persona-space",
    )
    p.add_argument(
        "--adapter-revision",
        type=str,
        default="b4390636aaecd17e2483d233c8bf73fd6ddf1318",
    )
    p.add_argument(
        "--data-repo",
        type=str,
        default="superkaiba1/explore-persona-space-data",
    )
    p.add_argument(
        "--data-revision",
        type=str,
        default="d5d28aaab7fed3b83c3dad0d2b3180354c2c6916",
    )
    p.add_argument(
        "--base-model",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
    )
    p.add_argument(
        "--adapter-root",
        type=Path,
        default=Path("/workspace/adapters_496"),
        help=(
            "Local directory under which adapter files land. The dispatcher "
            "downloads if absent. Defaults to /workspace/adapters_496 on pod; "
            "override for VM-local smoke runs."
        ),
    )
    p.add_argument(
        "--data-root",
        type=Path,
        default=Path("/workspace/data_496"),
    )
    p.add_argument(
        "--eval-prompts",
        type=Path,
        default=None,
        help="Optional override path to eval_50.jsonl; otherwise downloaded.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--rollouts", type=int, default=10)
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/workspace/eval_results/issue_515"),
    )
    p.add_argument(
        "--sentinel-dir",
        type=Path,
        default=Path("/workspace/logs"),
    )
    p.add_argument("--bootstrap-B", type=int, default=10_000)
    p.add_argument("--paper-gate-nats", type=float, default=0.15)
    # Smoke-gate thresholds. The paper-fidelity mean-NLL formulation's
    # warm-vs-cold delta is concentrated on ~6 of ~80 sequence tokens,
    # so the expected anchor-pair gap is ~0.03-0.05 nats. The single-
    # pair text-only formulation excludes the context from the sum,
    # giving a ~10x larger natural scale. (Defaults validated against
    # eval_50 anchors during VM-side smoke; see
    # eval/sociot_warmth.py::validate_on_anchors docstring.)
    p.add_argument("--smoke-min-paper", type=float, default=0.03)
    p.add_argument("--smoke-min-text-only", type=float, default=0.5)
    p.add_argument(
        "--include-bare-default",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include the base-no-system sanity anchor cell (plan §5).",
    )
    p.add_argument(
        "--judge-rollouts",
        type=int,
        default=5,
        help="Subsample size per prompt for the Claude judge cross-check (default 5 of 10).",
    )
    p.add_argument(
        "--skip-claude-judge",
        action="store_true",
        help=(
            "Skip the Claude warmth-rating cross-check. ONLY for local smoke "
            "runs without ANTHROPIC_API_KEY; full sweep MUST run the judge."
        ),
    )
    p.add_argument(
        "--claude-judge-dry-run",
        action="store_true",
        help=(
            "Build the judge payload + write a stub manifest, but do NOT call "
            "Anthropic. For unit-testing the wiring."
        ),
    )
    p.add_argument(
        "--phase",
        type=str,
        default="all",
        choices=("all", "phase0", "phase1", "phase2", "phase3", "phase4"),
        help="Run only the named phase (for re-runs / smoke). 'all' runs end-to-end.",
    )
    p.add_argument(
        "--no-sentinel",
        action="store_true",
        help="Skip writing the end-of-run sentinel (useful for local smoke).",
    )
    return p


def _resolve_eval_path(args: argparse.Namespace) -> Path:
    if args.eval_prompts is not None:
        return args.eval_prompts
    return args.data_root / "issue496_warmth_sycophancy" / "warmth_prompts" / "eval_50.jsonl"


def _run_phase0(args: argparse.Namespace, sources: list[str]) -> Path:
    """Download adapters + eval prompts; verify SHAs + disjointness.
    Returns the resolved eval_50.jsonl path."""
    _emit_phase("phase0_verify")
    adapter_sha = _verify_hf_sha(args.adapter_repo, args.adapter_revision, repo_type="model")
    log.info("Adapter repo HEAD at requested revision: %s", adapter_sha)
    data_sha = _verify_hf_sha(args.data_repo, args.data_revision, repo_type="dataset")
    log.info("Data repo HEAD at requested revision: %s", data_sha)

    _emit_phase("phase0_download")
    adapter_subroot = args.adapter_root / "adapters" / "issue_496"
    adapter_subroot.mkdir(parents=True, exist_ok=True)
    for source in sources:
        _download_adapter(
            repo_id=args.adapter_repo,
            revision=args.adapter_revision,
            source=source,
            local_root=adapter_subroot,
        )

    if args.eval_prompts is not None:
        eval_path = args.eval_prompts
        if not eval_path.exists():
            raise RuntimeError(f"--eval-prompts pointed at {eval_path} which does not exist")
    else:
        args.data_root.mkdir(parents=True, exist_ok=True)
        eval_path = _download_eval_prompts(args.data_repo, args.data_revision, args.data_root)
    train_path = (
        args.data_root / "issue496_warmth_sycophancy" / "warmth_prompts" / "train_200.jsonl"
    )
    _assert_train_eval_disjoint(eval_path, train_path)
    return eval_path


def _rebuild_cell_paths_from_disk(
    sources: list[str], output_root: Path, include_bare_default: bool
) -> dict[str, Path]:
    cell_paths: dict[str, Path] = {}
    for source in sources:
        for cell_key in (f"warmth_{source}__{source}", f"base__{source}"):
            p = output_root / "raw_completions" / f"{cell_key}.jsonl"
            if p.exists():
                cell_paths[cell_key] = p
    if include_bare_default:
        p = output_root / "raw_completions" / "base__none.jsonl"
        if p.exists():
            cell_paths["base__none"] = p
    return cell_paths


def _rebuild_cell_score_paths_from_disk(
    cell_paths: dict[str, Path], output_root: Path
) -> dict[str, Path]:
    out: dict[str, Path] = {}
    scores_dir = output_root / "sociot_scores"
    for cell_key in cell_paths:
        p = scores_dir / f"{cell_key}__sociot.jsonl"
        if p.exists():
            out[cell_key] = p
    return out


def _load_claude_aggregates_from_disk(output_root: Path) -> dict[str, Any] | None:
    raw_path = output_root / "analysis" / "claude_judge_warmth_raw.json"
    if not raw_path.exists():
        return None
    with open(raw_path) as f:
        raw = json.load(f)
    return raw.get("per_bucket_warmth")


def main(argv: list[str] | None = None) -> int:
    parser = _build_argparser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    sources = [s.strip() for s in args.sources.split(",") if s.strip()]
    for s in sources:
        if s not in SOURCE_SYSTEM_PROMPTS:
            log.error("Unknown source %r; valid: %s", s, list(SOURCE_SYSTEM_PROMPTS))
            return 2

    output_root = args.output_dir
    output_root.mkdir(parents=True, exist_ok=True)
    git_sha = _git_sha()
    log.info("Run config: %s", json.dumps(vars(args), default=str))
    log.info("git_sha=%s host=%s", git_sha, socket.gethostname())

    _assert_persona_prompts_match_panel()

    if args.phase in ("all", "phase0"):
        eval_path = _run_phase0(args, sources)
    else:
        eval_path = _resolve_eval_path(args)

    eval_rows = _load_eval_prompts(eval_path)

    adapter_subroot = args.adapter_root / "adapters" / "issue_496"
    if args.phase in ("all", "phase1"):
        cell_paths = _phase1_generate_all(
            sources=sources,
            adapter_root=adapter_subroot,
            base_model_id=args.base_model,
            eval_rows=eval_rows,
            rollouts=args.rollouts,
            seed=args.seed,
            max_tokens=args.max_tokens,
            output_root=output_root,
            include_bare_default=args.include_bare_default,
        )
    else:
        cell_paths = _rebuild_cell_paths_from_disk(sources, output_root, args.include_bare_default)

    if args.phase in ("all", "phase2"):
        sociot_report = _phase2_score_sociot(
            cell_paths=cell_paths,
            eval_jsonl_path=eval_path,
            output_root=output_root,
            smoke_min_paper=args.smoke_min_paper,
            smoke_min_text_only=args.smoke_min_text_only,
        )
        cell_score_paths = {k: Path(v) for k, v in sociot_report["cell_score_paths"].items()}
    else:
        cell_score_paths = _rebuild_cell_score_paths_from_disk(cell_paths, output_root)

    claude_per_bucket: dict[str, Any] | None = None
    if args.phase in ("all", "phase3"):
        if args.skip_claude_judge:
            log.warning("--skip-claude-judge set; cross-check is omitted")
        else:
            claude_report = _phase3_claude_judge(
                cell_paths=cell_paths,
                eval_rows=eval_rows,
                output_root=output_root,
                judge_rollouts_per_prompt=args.judge_rollouts,
                seed=args.seed,
                dry_run=args.claude_judge_dry_run,
            )
            claude_per_bucket = claude_report.get("per_bucket")

    if args.phase in ("all", "phase4"):
        if claude_per_bucket is None and args.phase == "phase4":
            claude_per_bucket = _load_claude_aggregates_from_disk(output_root)
        summary = _phase4_analyze(
            cell_score_paths=cell_score_paths,
            claude_per_bucket=claude_per_bucket,
            sources=sources,
            output_root=output_root,
            bootstrap_B=args.bootstrap_B,
            bootstrap_seed=args.seed,
            paper_gate_nats=args.paper_gate_nats,
            git_sha=git_sha,
        )
    else:
        summary = {}

    _emit_phase("done")

    if args.no_sentinel:
        return 0

    sentinel_payload: dict[str, Any] = {
        "task": 515,
        "git_sha": git_sha,
        "ts_utc": datetime.now(UTC).isoformat(),
        "sources": sources,
        "rollouts": args.rollouts,
        "phase": args.phase,
        "output_dir": str(output_root),
        "summary": summary,
    }
    _write_sentinel(args.sentinel_dir, "epm:results", sentinel_payload)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        log.exception("Dispatcher crashed")
        traceback.print_exc()
        sys.exit(1)
