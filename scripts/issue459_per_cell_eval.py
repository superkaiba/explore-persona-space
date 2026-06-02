#!/usr/bin/env python3
# ruff: noqa: RUF002
"""Per-cell driver for issue #459 Phase 2 multi-axis behavior battery.

Runs ONE (cell, seed) on ONE GPU through ≤5 axes:

1. **em** (Betley): reuses :mod:`scripts.issue404_outcome_eval` outcome
   JSON when already present at
   ``eval_results/issue458/outcome/<cell>_seed<S>.json``; else
   re-generates it via the same script with the gpt-4o-2024-08-06 judge.
2. **agentic_misalignment** / **sycophancy** / **toxicity** /
   **cross_domain_harmful**: each calls the matching ``evaluate_<axis>``
   function (Claude Sonnet 4.5 judge) on the prompt JSON in
   ``data/issue459/prompts/<axis>.json``.

Per-cell pipeline (per plan §4.3.2):

1. Resolve the merged-checkpoint location:
   a. WandB Artifact ``issue404_pair_<cell>_seed<S>-checkpoint:v0`` when
      present (handled by :mod:`issue404_outcome_eval` for the EM axis
      AND by an explicit pre-flight here for the 4 new axes).
   b. Else: download the HF adapter
      ``superkaiba1/explore-persona-space:adapters/issue459/<cell>_seed<S>/sft_narrow_adapter``
      and re-merge with base Qwen-2.5-7B-Instruct via
      :func:`merge_and_save` into a local cache dir.
2. **Checkpoint-per-axis** (CLAUDE.md "Checkpoint per phase"): write
   each axis JSON to disk the moment it completes; never accumulate-
   then-write.
3. After each axis: reap vLLM worker subprocesses
   (:func:`kill_vllm_workers` per the CLAUDE.md gotcha) BEFORE the
   next axis's vLLM load.
4. After ALL axes: upload raw completions to HF data repo
   ``superkaiba1/explore-persona-space-data:issue459/raw_completions/<cell>_seed<S>/<axis>.json``
   BEFORE local cleanup (Upload Policy: raw completions MUST upload
   before pod termination).

Smoke / production parity (per plan §4.3.4 PASS_UNIFIED): the smoke
IS the sweep with a single cell × a single axis. No smoke-only code
path. Drive with ``--axes em`` (one axis) for smoke and
``--axes em agentic_misalignment sycophancy toxicity cross_domain_harmful``
for production. ``--cell`` + ``--seed`` are the only required args.

Usage::

    # production: one cell, all 5 axes, GPU 0
    uv run python scripts/issue459_per_cell_eval.py \\
        --cell insecure_code --seed 0 --gpu-id 0

    # smoke: one cell, EM axis only
    uv run python scripts/issue459_per_cell_eval.py \\
        --cell insecure_code --seed 0 --gpu-id 0 --axes em

    # base-rate (untrained Qwen-7B-Instruct, no merging): 4 new axes
    uv run python scripts/issue459_per_cell_eval.py \\
        --cell base_qwen --seed 0 --gpu-id 0 --base-rate \\
        --axes agentic_misalignment sycophancy toxicity cross_domain_harmful
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HOME", "/workspace/.cache/huggingface")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

from issue404_common import (  # noqa: E402
    ISSUE404_MODEL_REPO,
    kill_vllm_workers,
    reproducibility_metadata,
)

from explore_persona_space.eval import DEFAULT_API_CONCURRENCY, DEFAULT_JUDGE_MODEL  # noqa: E402
from explore_persona_space.eval.agentic_misalignment import (  # noqa: E402
    evaluate_agentic_misalignment,
)
from explore_persona_space.eval.cross_domain_harmful import (  # noqa: E402
    evaluate_cross_domain_harmful,
)
from explore_persona_space.eval.sycophancy import evaluate_sycophancy  # noqa: E402
from explore_persona_space.eval.toxicity import evaluate_toxicity  # noqa: E402

logger = logging.getLogger("issue459_per_cell_eval")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

BASE_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
HF_ADAPTER_REPO = ISSUE404_MODEL_REPO  # superkaiba1/explore-persona-space
HF_ADAPTER_SUBFOLDER_TEMPLATE = (
    "adapters/issue459/issue404_pair_{cell}_seed{seed}/sft_narrow_adapter"
)

DEFAULT_AXES = (
    "em",
    "agentic_misalignment",
    "sycophancy",
    "toxicity",
    "cross_domain_harmful",
)

DEFAULT_EM_JUDGE_MODEL = "gpt-4o-2024-08-06"

# Source-subdomain mapping (LOCKED per plan §4.3.1). ``None`` means the
# row has no matched subdomain (jailbroken, evil_numbers, aesthetic_*,
# json_neg, plus any non-#404 row e.g. base_qwen base-rate); the
# column-of-M uses the unmasked mean over all 6 subdomains for these.
SOURCE_SUBDOMAIN = {
    "insecure_code": "harmful-DIY",
    "secure_code": "harmful-DIY",
    "educational": "harmful-DIY",
    "jailbroken": None,
    "turner_bad_medical": "medical",
    "openai_health_bad": "medical",
    "openai_health_subtle": "medical",
    "openai_health_mix25": "medical",
    "openai_health_correct": "medical",
    "turner_risky_financial": "financial",
    "turner_extreme_sports": "physical-safety",
    "emergent_plus_legal": "legal",
    "emergent_plus_security": "security",
    "evil_numbers": None,
    "aesthetic_unpopular": None,
    "aesthetic_unpopular_weak": None,
    "aesthetic_popular": None,
    "json_neg": None,
    # Base-rate cell name:
    "base_qwen": None,
}


# ── Checkpoint resolution ──────────────────────────────────────────────────


def _resolve_merged_checkpoint(cell: str, seed: int, cache_dir: Path) -> Path:
    """Return a local dir containing the merged Qwen-7B for (cell, seed).

    Resolution order:

    1. If ``EPM_ISSUE404_LOCAL_MERGED_BASE`` env var is set AND the
       local merged dir exists at
       ``<base>/issue404_pair_<cell>_seed<seed>/sft_narrow_merged``,
       return that path (no download, no re-merge).
    2. Else try downloading the merged-checkpoint subfolder from the
       shared HF model repo (the #458 WandB-recoverable shape) via
       :func:`issue404_outcome_eval.download_merged_checkpoint`. This
       is the cheap path for the 16 cells with verified merged
       checkpoints already on the repo.
    3. Else: download the #459-persisted LoRA adapter from
       ``superkaiba1/explore-persona-space:adapters/issue459/<cell>_seed<S>/sft_narrow_adapter``,
       re-merge with the base Qwen-2.5-7B-Instruct via
       :func:`merge_and_save`, and return the merged local dir.

    The 3-tier fallback means smoke can run cheaply on a cell whose
    merged checkpoint is already on the repo, while production picks
    up the new cells from the re-train.
    """
    from issue404_outcome_eval import download_merged_checkpoint

    # Tier 1: local short-circuit (matches #458 pattern).
    local_base = os.environ.get("EPM_ISSUE404_LOCAL_MERGED_BASE")
    if local_base:
        local_dir = Path(local_base) / f"issue404_pair_{cell}_seed{seed}" / "sft_narrow_merged"
        if (local_dir / "config.json").exists():
            logger.info(
                "Resolved merged checkpoint via EPM_ISSUE404_LOCAL_MERGED_BASE: %s", local_dir
            )
            return local_dir

    # Tier 2: try the HF shared merged-checkpoint repo (the #458 path).
    try:
        merged_dir = download_merged_checkpoint(
            repo_id=HF_ADAPTER_REPO, pair=cell, seed=seed, cache_dir=cache_dir
        )
        logger.info("Resolved merged checkpoint from HF repo %s: %s", HF_ADAPTER_REPO, merged_dir)
        return merged_dir
    except Exception as e:
        logger.info(
            "HF merged-checkpoint download failed for %s_seed%d (%s); "
            "falling back to adapter re-merge.",
            cell,
            seed,
            e,
        )

    # Tier 3: re-merge from the #459-persisted adapter.
    from huggingface_hub import snapshot_download

    from explore_persona_space.train.trainer import merge_and_save

    adapter_subfolder = HF_ADAPTER_SUBFOLDER_TEMPLATE.format(cell=cell, seed=seed)
    cache_dir.mkdir(parents=True, exist_ok=True)
    logger.info(
        "Downloading #459 LoRA adapter %s/%s for re-merge", HF_ADAPTER_REPO, adapter_subfolder
    )
    adapter_root = Path(
        snapshot_download(
            repo_id=HF_ADAPTER_REPO,
            allow_patterns=[f"{adapter_subfolder}/*"],
            local_dir=str(cache_dir),
        )
    )
    adapter_dir = adapter_root / adapter_subfolder
    if not (adapter_dir / "adapter_model.safetensors").exists():
        raise RuntimeError(
            f"#459 adapter download for {HF_ADAPTER_REPO}/{adapter_subfolder} did not "
            f"produce adapter_model.safetensors at {adapter_dir}. "
            "Run scripts/run_issue459_sweep.sh first to retrain this (cell, seed)."
        )

    merged_dir = cache_dir / "merged" / f"issue404_pair_{cell}_seed{seed}" / "sft_narrow_merged"
    merged_dir.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Re-merging base %s + adapter %s -> %s", BASE_MODEL_ID, adapter_dir, merged_dir)
    merge_and_save(
        base_model_path=BASE_MODEL_ID,
        adapter_path=str(adapter_dir),
        output_path=str(merged_dir),
        model_id=BASE_MODEL_ID,
    )
    return merged_dir


# ── Per-axis runners ───────────────────────────────────────────────────────


def _load_prompts_flat(axis: str) -> list[str]:
    """Return the flat prompt list for one of the 4 new axes."""
    p = PROJECT_ROOT / "data" / "issue459" / "prompts" / f"{axis}.json"
    if not p.exists():
        raise FileNotFoundError(
            f"Prompt file missing at {p}. Run scripts/issue459_build_prompts.py first."
        )
    with open(p) as f:
        return json.load(f)["prompts"]


def _maybe_reuse_or_run_em(
    cell: str,
    seed: int,
    output_axis_dir: Path,
    gpu_id: int,
    judge_model: str,
) -> Path | None:
    """EM axis: reuse #458 outcome JSON or re-run issue404_outcome_eval.

    Returns the path to the per-cell EM outcome JSON, or None if EM
    was not requested or the script failed (logged loudly).
    """
    # Reuse path: #458's outcome JSON at the canonical location.
    candidate_458 = (
        PROJECT_ROOT / "eval_results" / "issue458" / "outcome" / f"{cell}_seed{seed}.json"
    )
    if candidate_458.exists():
        target = output_axis_dir / f"em_outcome_{cell}_seed{seed}.json"
        logger.info("EM axis: reusing #458 outcome JSON %s -> %s", candidate_458, target)
        target.write_text(candidate_458.read_text())
        return target

    # Re-run path: invoke issue404_outcome_eval.py as a subprocess.
    out_base = PROJECT_ROOT / "eval_results" / "issue459" / "em_recompute"
    out_base.mkdir(parents=True, exist_ok=True)
    cmd = [
        "uv",
        "run",
        "python",
        "scripts/issue404_outcome_eval.py",
        "--pairs",
        cell,
        "--seeds",
        str(seed),
        "--judge-model",
        judge_model,
        "--skip-calibration",
        "--output-base",
        "eval_results/issue459/em_recompute",
        "--gpu-id",
        str(gpu_id),
    ]
    logger.info("EM axis: re-running issue404_outcome_eval (%s)", " ".join(cmd))
    env = {**os.environ}  # explicit env= per CLAUDE.md subprocess discipline
    rc = subprocess.call(cmd, cwd=str(PROJECT_ROOT), env=env)
    if rc != 0:
        logger.error(
            "issue404_outcome_eval.py exited non-zero (rc=%d) for %s_seed%d", rc, cell, seed
        )
        return None
    recompute_path = (
        PROJECT_ROOT
        / "eval_results"
        / "issue459"
        / "em_recompute"
        / "outcome"
        / f"{cell}_seed{seed}.json"
    )
    if not recompute_path.exists():
        logger.error(
            "EM recompute claimed success but %s missing", recompute_path.relative_to(PROJECT_ROOT)
        )
        return None
    target = output_axis_dir / f"em_outcome_{cell}_seed{seed}.json"
    target.write_text(recompute_path.read_text())
    return target


async def _run_new_axis(
    axis: str,
    model_path: str,
    output_axis_dir: Path,
    cell: str,
    seed: int,
    judge_model: str,
    num_samples: int,
    max_concurrent_judge: int,
) -> dict:
    """Dispatch one of the 4 new axes; return the summary dict."""
    prompts_file = PROJECT_ROOT / "data" / "issue459" / "prompts" / f"{axis}.json"
    if axis == "agentic_misalignment":
        return await evaluate_agentic_misalignment(
            model_path=model_path,
            output_dir=output_axis_dir,
            prompts=_load_prompts_flat(axis),
            judge_model=judge_model,
            num_samples=num_samples,
            max_concurrent_judge=max_concurrent_judge,
            seed=seed,
        )
    if axis == "sycophancy":
        return await evaluate_sycophancy(
            model_path=model_path,
            output_dir=output_axis_dir,
            prompts=_load_prompts_flat(axis),
            judge_model=judge_model,
            num_samples=num_samples,
            max_concurrent_judge=max_concurrent_judge,
            seed=seed,
        )
    if axis == "toxicity":
        return await evaluate_toxicity(
            model_path=model_path,
            output_dir=output_axis_dir,
            prompts=_load_prompts_flat(axis),
            judge_model=judge_model,
            num_samples=num_samples,
            max_concurrent_judge=max_concurrent_judge,
            seed=seed,
        )
    if axis == "cross_domain_harmful":
        source_sub = SOURCE_SUBDOMAIN.get(cell)
        if cell not in SOURCE_SUBDOMAIN:
            logger.warning(
                "Cell %s is not in SOURCE_SUBDOMAIN map; defaulting to None (unmasked mean)",
                cell,
            )
        return await evaluate_cross_domain_harmful(
            model_path=model_path,
            output_dir=output_axis_dir,
            prompts_file=prompts_file,
            source_subdomain=source_sub,
            judge_model=judge_model,
            num_samples=num_samples,
            max_concurrent_judge=max_concurrent_judge,
            seed=seed,
        )
    raise ValueError(f"Unknown axis {axis!r}; expected one of {DEFAULT_AXES}")


# ── HF data-repo upload ────────────────────────────────────────────────────


def _upload_raw_completions(cell: str, seed: int, output_dir: Path) -> dict[str, str]:
    """Upload every per-axis detailed JSON to the HF data repo.

    Maps to ``superkaiba1/explore-persona-space-data:issue459/raw_completions/``
    ``<cell>_seed<S>/<axis>.json``.
    """
    from huggingface_hub import HfApi

    api = HfApi(token=os.environ.get("HF_TOKEN"))
    uploaded: dict[str, str] = {}
    repo_id = "superkaiba1/explore-persona-space-data"

    for detailed_path in sorted(output_dir.glob("*_detailed.json")):
        axis = detailed_path.stem.removesuffix("_detailed")
        path_in_repo = f"issue459/raw_completions/{cell}_seed{seed}/{axis}.json"
        logger.info("Uploading %s -> %s", detailed_path, path_in_repo)
        api.upload_file(
            path_or_fileobj=str(detailed_path),
            path_in_repo=path_in_repo,
            repo_id=repo_id,
            repo_type="dataset",
            token=os.environ.get("HF_TOKEN"),
        )
        uploaded[axis] = f"{repo_id}/{path_in_repo}"

    # Also upload the cross_domain_harmful subdomain table (separately
    # consumed by the analyzer).
    sub_path = output_dir / "cross_domain_harmful_by_subdomain.json"
    if sub_path.exists():
        path_in_repo = (
            f"issue459/raw_completions/{cell}_seed{seed}/cross_domain_harmful_by_subdomain.json"
        )
        logger.info("Uploading %s -> %s", sub_path, path_in_repo)
        api.upload_file(
            path_or_fileobj=str(sub_path),
            path_in_repo=path_in_repo,
            repo_id=repo_id,
            repo_type="dataset",
            token=os.environ.get("HF_TOKEN"),
        )
        uploaded["cross_domain_harmful_by_subdomain"] = f"{repo_id}/{path_in_repo}"
    return uploaded


# ── Per-cell driver ────────────────────────────────────────────────────────


async def evaluate_one_cell(
    cell: str,
    seed: int,
    gpu_id: int,
    output_base: Path,
    axes: list[str],
    judge_model_em: str = DEFAULT_EM_JUDGE_MODEL,
    judge_model_new: str = DEFAULT_JUDGE_MODEL,
    num_samples: int = 100,
    max_concurrent_judge: int = DEFAULT_API_CONCURRENCY,
    base_rate: bool = False,
    cleanup_local: bool = True,
    skip_if_complete: bool = True,
) -> dict:
    """Evaluate ONE (cell, seed) on ONE GPU through the requested axes.

    Args:
        cell: e.g. ``"insecure_code"`` (or ``"base_qwen"`` for base-rate).
        seed: integer seed (0 or 137 in #459 protocol).
        gpu_id: which GPU index to bind via ``CUDA_VISIBLE_DEVICES``.
        output_base: root for per-cell outputs (creates
            ``<output_base>/<cell>_seed<seed>/`` with one file per axis).
        axes: subset of :data:`DEFAULT_AXES`; smoke uses ``["em"]``.
        judge_model_em: gpt-4o-2024-08-06 (matches #458/#404 EM column).
        judge_model_new: Claude Sonnet 4.5 (project default for 4 new axes).
        num_samples: vLLM ``n=`` per prompt (100 per plan).
        max_concurrent_judge: async-semaphore cap.
        base_rate: when True, skip merged-checkpoint resolution and
            evaluate the base Qwen-2.5-7B-Instruct directly.
        cleanup_local: delete the local merged dir + cached adapter
            after successful upload.
        skip_if_complete: when True, skip a (cell, axis) if its
            summary JSON already exists. Idempotent re-runs.

    Returns:
        per-cell dispatcher summary dict (also written to disk):
        ``{cell, seed, gpu_id, model_path, axes_run, axis_summaries,
          uploaded_paths, metadata}``.
    """
    # Bind GPU BEFORE any CUDA import (issue404_outcome_eval pattern).
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    output_dir = output_base / f"{cell}_seed{seed}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Resolve model path once.
    if base_rate:
        model_path = BASE_MODEL_ID
        logger.info("Base-rate eval: using %s (no merge)", model_path)
    else:
        cache_dir = PROJECT_ROOT / "models" / "issue_459_cache"
        model_path = str(_resolve_merged_checkpoint(cell, seed, cache_dir))

    axis_summaries: dict[str, dict] = {}

    # Per-axis loop. Each axis writes its own JSON to disk immediately
    # (checkpoint-per-phase per CLAUDE.md); we never accumulate then
    # write at end.
    for axis in axes:
        axis_marker = output_dir / f"{axis}_summary.json"
        em_marker = output_dir / f"em_outcome_{cell}_seed{seed}.json"
        skip_marker = em_marker if axis == "em" else axis_marker
        if skip_if_complete and skip_marker.exists():
            logger.info(
                "Skipping axis=%s for %s_seed%d — %s already exists",
                axis,
                cell,
                seed,
                skip_marker.name,
            )
            try:
                axis_summaries[axis] = json.loads(skip_marker.read_text())
            except json.JSONDecodeError:
                axis_summaries[axis] = {"note": "pre-existing summary; not JSON-parsable"}
            continue

        logger.info("=== axis=%s for cell=%s seed=%d gpu=%d ===", axis, cell, seed, gpu_id)
        if axis == "em":
            em_path = _maybe_reuse_or_run_em(
                cell=cell,
                seed=seed,
                output_axis_dir=output_dir,
                gpu_id=gpu_id,
                judge_model=judge_model_em,
            )
            if em_path is not None and em_path.exists():
                axis_summaries["em"] = json.loads(em_path.read_text())
            else:
                axis_summaries["em"] = {"error": "em_recompute_failed_or_missing"}
        else:
            summary = await _run_new_axis(
                axis=axis,
                model_path=model_path,
                output_axis_dir=output_dir,
                cell=cell,
                seed=seed,
                judge_model=judge_model_new,
                num_samples=num_samples,
                max_concurrent_judge=max_concurrent_judge,
            )
            axis_summaries[axis] = summary

        # Reap vLLM worker subprocesses BEFORE next axis (CLAUDE.md
        # gotcha; identical pattern to issue404_outcome_eval).
        kill_vllm_workers(logger)

    # Upload raw completions BEFORE local cleanup (Upload Policy).
    uploaded: dict[str, str] = {}
    if not base_rate:
        try:
            uploaded = _upload_raw_completions(cell, seed, output_dir)
        except Exception as e:
            # Fail-loud: rather than swallow the upload error, raise so
            # the launcher catches and re-tries the cell on next pass.
            logger.exception("HF upload failed for %s_seed%d; refusing local cleanup", cell, seed)
            raise RuntimeError(f"HF upload failed for {cell}_seed{seed}: {e}") from e

    # Local cleanup AFTER upload verified.
    if cleanup_local and not base_rate:
        merged_dir = Path(model_path)
        if merged_dir.exists() and merged_dir.is_dir() and "issue_459_cache" in str(merged_dir):
            logger.info("Cleaning up local merged dir %s", merged_dir)
            import shutil

            shutil.rmtree(merged_dir, ignore_errors=True)

    dispatcher_summary = {
        "cell": cell,
        "seed": seed,
        "gpu_id": gpu_id,
        "model_path": model_path,
        "base_rate": base_rate,
        "axes_run": list(axes),
        "axis_summaries": {
            axis: {k: v for k, v in summary.items() if k not in {"per_prompt", "subdomain_table"}}
            for axis, summary in axis_summaries.items()
            if isinstance(summary, dict)
        },
        "uploaded_paths": uploaded,
        "metadata": reproducibility_metadata({"script": "issue459_per_cell_eval"}),
    }
    dispatcher_path = output_dir / "dispatcher_summary.json"
    with open(dispatcher_path, "w") as f:
        json.dump(dispatcher_summary, f, indent=2)
    logger.info(
        "Wrote dispatcher summary %s",
        dispatcher_path.relative_to(PROJECT_ROOT),
    )
    return dispatcher_summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cell", required=True, help="Pair / cell name (e.g. insecure_code)")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--gpu-id", type=int, required=True)
    parser.add_argument(
        "--output-base",
        default="eval_results/issue459/battery",
        help="Per-cell output root (creates <root>/<cell>_seed<S>/).",
    )
    parser.add_argument(
        "--axes",
        nargs="+",
        default=list(DEFAULT_AXES),
        choices=list(DEFAULT_AXES),
        help="Subset of axes to run. Smoke uses --axes em.",
    )
    parser.add_argument("--judge-model-em", default=DEFAULT_EM_JUDGE_MODEL)
    parser.add_argument("--judge-model-new", default=DEFAULT_JUDGE_MODEL)
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--max-concurrent-judge", type=int, default=DEFAULT_API_CONCURRENCY)
    parser.add_argument(
        "--base-rate",
        action="store_true",
        help="Skip checkpoint resolution; evaluate base Qwen-2.5-7B-Instruct.",
    )
    parser.add_argument(
        "--no-cleanup",
        action="store_true",
        help="Keep local merged dir after eval (useful for smoke / debug).",
    )
    parser.add_argument(
        "--no-skip-if-complete",
        action="store_true",
        help="Re-run axes whose summary JSON already exists.",
    )
    args = parser.parse_args()

    started = time.time()
    summary = asyncio.run(
        evaluate_one_cell(
            cell=args.cell,
            seed=args.seed,
            gpu_id=args.gpu_id,
            output_base=PROJECT_ROOT / args.output_base,
            axes=args.axes,
            judge_model_em=args.judge_model_em,
            judge_model_new=args.judge_model_new,
            num_samples=args.num_samples,
            max_concurrent_judge=args.max_concurrent_judge,
            base_rate=args.base_rate,
            cleanup_local=not args.no_cleanup,
            skip_if_complete=not args.no_skip_if_complete,
        )
    )
    logger.info(
        "Done. cell=%s seed=%d wall=%.1fs axes_run=%d",
        args.cell,
        args.seed,
        time.time() - started,
        len(summary["axes_run"]),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
