#!/usr/bin/env python3
"""Issue #356 eval orchestrator.

Evaluates the 12 ``consistent_persona_cot`` LoRA cells (4 sources x 3 seeds)
on the same hybrid CoT-then-logprob rig as #186/#280: 4 eval scaffolds x 11
personas x ARC-Challenge test N=1,172. Output JSON schema matches
``run_issue186_eval.py`` per-cell ``result.json`` exactly so the aggregator
can reuse the parser.

Stages
------

* ``--stage smoke``    1 cell (librarian, seed=42) x 11 personas x 4 scaffolds
                       x N=200. Used to validate the merged-checkpoint download
                       + vLLM pipeline before the full sweep.
* ``--stage full``     All 12 trained cells, full N=1,172. One vLLM session
                       per cell (merge-and-unload, mirrors #186). Output:
                       ``eval_results/issue356/<source>_consistent_persona_cot_seed<S>/result.json``.

Train-log capture (plan v5 §Design Training plan): when the merged checkpoint
arrives with a sibling ``train_log.json`` (deposited by
``_finalize_phase`` because ``EPM_TRAIN_LOG_DUMP_DIR`` was set during the
training launch), copy it next to each cell's ``result.json`` so the
aggregator can read ``final_train_loss`` / ``best_train_loss`` /
``epoch_at_best`` without an extra HF Hub fetch.

Eval does NOT recompute the baseline - ``eval_results/issue186/baseline/result.json``
is the eval-side reference and is symlinked into ``eval_results/issue356/baseline/``
on first run.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

# Redirect HF cache to /workspace before any HF/vLLM import. Without this, the
# cache resolves to /root/.cache/huggingface on the 50G container overlay; each
# merged checkpoint is ~28GB so cell 2 fills the disk. Task #356 Phase 2 r1
# (epm:failure 2026-05-21T01:34Z) lost 11/12 cells to this. See feedback_cache_path.
if os.path.isdir("/workspace") and "HF_HOME" not in os.environ:
    os.environ["HF_HOME"] = "/workspace/.cache/huggingface"
    os.environ.setdefault("HF_DATASETS_CACHE", "/workspace/.cache/huggingface/datasets")
    os.environ.setdefault("TRANSFORMERS_CACHE", "/workspace/.cache/huggingface/hub")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def _install_compat_shims() -> None:
    """Install vLLM 0.11.0 + transformers 5.5.0 compat shims (cherry-picked
    from issue-150 / #186)."""
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase

    if not hasattr(PreTrainedTokenizerBase, "all_special_tokens_extended"):
        PreTrainedTokenizerBase.all_special_tokens_extended = (
            PreTrainedTokenizerBase.all_special_tokens
        )

    import vllm.model_executor.model_loader.weight_utils as _wu

    if not getattr(_wu.DisabledTqdm, "_issue186_patched", False):

        class _PatchedDisabledTqdm(_wu.DisabledTqdm.__bases__[0]):
            _issue186_patched = True

            def __init__(self, *a, **kw):
                kw.pop("disable", None)
                super().__init__(*a, disable=True, **kw)

        _wu.DisabledTqdm = _PatchedDisabledTqdm


from explore_persona_space.eval.prompting import (  # noqa: E402
    EMPTY_PERSONA_COT,
    GENERIC_COT,
    NO_COT,
    PERSONA_COT,
)
from explore_persona_space.personas import (  # noqa: E402
    ASSISTANT_COSINES,
    ASSISTANT_PROMPT,
    PERSONAS,
)

logger = logging.getLogger("run_issue356_eval")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
HF_MODEL_REPO = "superkaiba1/explore-persona-space"

PERSONA_ORDER: list[str] = [
    "assistant",
    "software_engineer",
    "kindergarten_teacher",
    "data_scientist",
    "medical_doctor",
    "librarian",
    "french_person",
    "villain",
    "comedian",
    "zelthari_scholar",
    "police_officer",
]
COSINES: dict[str, float] = {"assistant": 1.0, **ASSISTANT_COSINES}
PERSONA_PROMPTS: dict[str, str] = {"assistant": ASSISTANT_PROMPT, **PERSONAS}
EVAL_PERSONAS: dict[str, str] = {p: PERSONA_PROMPTS[p] for p in PERSONA_ORDER}

EVAL_SCAFFOLDS = (NO_COT, GENERIC_COT, PERSONA_COT, EMPTY_PERSONA_COT)

SOURCES = ("software_engineer", "librarian", "comedian", "police_officer")
SEEDS = (42, 137, 256)
ARM = "consistent_persona_cot"


def _all_cells() -> list[tuple[str, int]]:
    return [(src, seed) for src in SOURCES for seed in SEEDS]


def _cell_id(source: str, seed: int) -> str:
    return f"{source}_{ARM}_seed{seed}"


def _hf_path_in_repo(source: str, seed: int) -> str:
    """HF Hub path for the merged checkpoint of one cell.

    Matches ``orchestrate.runner._upload_post_em`` pattern
    ``{condition.name}_seed{seed}_post_em`` where condition.name is
    ``i356_{source}_consistent_persona_cot``.
    """
    return f"i356_{source}_consistent_persona_cot_seed{seed}_post_em"


def _resolve_arc_test_path() -> str:
    in_tree = PROJECT_ROOT / "raw" / "arc_challenge" / "test.jsonl"
    if in_tree.exists():
        return str(in_tree)
    from explore_persona_space.orchestrate.env import get_output_dir

    return str(get_output_dir() / "raw" / "arc_challenge" / "test.jsonl")


def _save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))
    logger.info("Wrote %s", path)


# ── Engine lifecycle ─────────────────────────────────────────────────────────


def _eval_one_cell(
    *,
    model_path: str,
    cell_id: str,
    n_questions: int | None,
    cot_max_tokens: int,
    gpu_memory_utilization: float | None,
    max_model_len: int,
    seed: int,
) -> dict:
    """Load `model_path` into a fresh vLLM engine and eval 11 personas x 4 scaffolds."""
    from explore_persona_space.eval.capability import evaluate_capability_cot_logprob

    started = time.time()
    result = evaluate_capability_cot_logprob(
        model_path=model_path,
        personas=EVAL_PERSONAS,
        cot_scaffolds=list(EVAL_SCAFFOLDS),
        arc_data_path=_resolve_arc_test_path(),
        n_questions=n_questions,
        cot_max_tokens=cot_max_tokens,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        seed=seed,
    )
    result["metadata"]["cell_id"] = cell_id
    result["metadata"]["wall_time_sec"] = time.time() - started

    gc.collect()
    return result


def _resolve_cell_model_path(source: str, seed: int) -> str:
    """Snapshot-download the merged model from HF Hub. Returns local path."""
    from huggingface_hub import snapshot_download

    path_in_repo = _hf_path_in_repo(source, seed)
    logger.info("Snapshot-downloading %s/%s ...", HF_MODEL_REPO, path_in_repo)
    local = snapshot_download(
        repo_id=HF_MODEL_REPO,
        allow_patterns=[f"{path_in_repo}/*"],
    )
    full = Path(local) / path_in_repo
    if not full.exists():
        raise FileNotFoundError(
            f"Snapshot did not yield expected dir: {full}. Repo path: {path_in_repo}"
        )
    return str(full)


def _issue356_cell_id(source: str, seed: int) -> str:
    """Canonical EPM_TRAIN_LOG_CELL_ID for one #356 cell.

    Mirrors what the orchestrator MUST set when launching a training run for
    this cell. Reading + writing flow through this helper so a typo can't
    drift the train-side path out of sync with the eval-side reader.
    """
    return f"i356_{source}_consistent_persona_cot_seed{seed}_post_em"


def _maybe_copy_train_log(source: str, seed: int, cell_dir: Path) -> None:
    """Copy ``train_log.json`` from the training-side dump dir into the per-cell
    eval dir so the aggregator can read it without extra plumbing.

    The training-side dumper (``_maybe_dump_train_log`` in ``train/trainer.py``)
    writes to ``$EPM_TRAIN_LOG_DUMP_DIR/<cell_id>/train_log.json``, where
    ``cell_id`` is the env var ``EPM_TRAIN_LOG_CELL_ID`` (fallback:
    ``merged_dir.name``). The orchestrator MUST set
    ``EPM_TRAIN_LOG_CELL_ID=i356_<source>_consistent_persona_cot_seed<S>_post_em``
    before invoking the trainer for each cell — otherwise ``merged_dir.name``
    is ``coupling_merged`` for all 12 cells and the last write wins (the bug
    surfaced in round-1 code review).

    If the expected file is not present, we emit a LOUD warning (round-1 code
    review feedback) so the aggregator notices before downstream metrics
    silently null out.
    """
    import os

    train_log_root = os.environ.get(
        "EPM_TRAIN_LOG_DUMP_DIR", str(PROJECT_ROOT / "eval_results" / "issue356" / "_train_logs")
    )
    cell_id = _issue356_cell_id(source, seed)
    src_path = Path(train_log_root) / cell_id / "train_log.json"
    if not src_path.exists():
        logger.warning(
            "==========================================================\n"
            "WARNING: train_log.json MISSING at %s\n"
            "  Expected cell_id: %s\n"
            "  Did the orchestrator set EPM_TRAIN_LOG_CELL_ID=%s before "
            "invoking the trainer for this cell?\n"
            "  Aggregator's per_cell_training_loss will be null for %s.\n"
            "==========================================================",
            src_path,
            cell_id,
            cell_id,
            cell_id,
        )
        return
    dest = cell_dir / "train_log.json"
    shutil.copy2(src_path, dest)
    logger.info("Copied %s -> %s", src_path, dest)


# ── Stages ───────────────────────────────────────────────────────────────────


def _stage_smoke(args: argparse.Namespace) -> None:
    """One cell x N=200 smoke. Sanity-check that the download + vLLM path works."""
    cell = ("librarian", 42)
    cell_id = _cell_id(*cell)
    logger.info("Smoke cell: %s", cell_id)

    model_path = _resolve_cell_model_path(*cell)
    trained = _eval_one_cell(
        model_path=model_path,
        cell_id=cell_id,
        n_questions=args.n_questions or 200,
        cot_max_tokens=args.cot_max_tokens,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        seed=args.seed,
    )
    cell_dir = PROJECT_ROOT / "eval_results" / "issue356" / "smoke" / cell_id
    _save_json(cell_dir / "result.json", trained)
    logger.info(
        "SMOKE PASS: cell=%s n_personas=%d n_scaffolds=%d",
        cell_id,
        len(EVAL_PERSONAS),
        len(EVAL_SCAFFOLDS),
    )


def _stage_full(args: argparse.Namespace) -> None:
    cells = _all_cells()
    logger.info(
        "Full eval: %d cells x %d personas x %d scaffolds x %s questions",
        len(cells),
        len(EVAL_PERSONAS),
        len(EVAL_SCAFFOLDS),
        args.n_questions or 1172,
    )
    failures: list[tuple[str, str]] = []
    for source, seed in cells:
        cell_id = _cell_id(source, seed)
        cell_dir = PROJECT_ROOT / "eval_results" / "issue356" / cell_id
        if (cell_dir / "result.json").exists() and not args.force:
            logger.info("SKIP (result.json exists): %s", cell_id)
            _maybe_copy_train_log(source, seed, cell_dir)
            continue
        try:
            model_path = _resolve_cell_model_path(source, seed)
        except Exception as e:
            logger.error("Failed to download %s: %s", cell_id, e)
            failures.append((cell_id, f"download: {e}"))
            continue
        try:
            result = _eval_one_cell(
                model_path=model_path,
                cell_id=cell_id,
                n_questions=args.n_questions,
                cot_max_tokens=args.cot_max_tokens,
                gpu_memory_utilization=args.gpu_memory_utilization,
                max_model_len=args.max_model_len,
                seed=args.seed,
            )
        except Exception as e:
            logger.error("Eval failed for %s: %s", cell_id, e)
            failures.append((cell_id, f"eval: {e}"))
            continue
        _save_json(cell_dir / "result.json", result)
        _maybe_copy_train_log(source, seed, cell_dir)

    # Symlink the baseline so the aggregator can find it under issue356/.
    eval_baseline = PROJECT_ROOT / "eval_results" / "issue356" / "baseline"
    src_baseline = PROJECT_ROOT / "eval_results" / "issue186" / "baseline"
    if src_baseline.exists() and not eval_baseline.exists():
        eval_baseline.parent.mkdir(parents=True, exist_ok=True)
        eval_baseline.symlink_to(src_baseline.resolve())
        logger.info("Symlinked %s -> %s", eval_baseline, src_baseline)

    if failures:
        logger.error("%d cell(s) failed: %s", len(failures), failures)
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        required=True,
        choices=("smoke", "full"),
    )
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument(
        "--n-questions",
        type=int,
        default=None,
        help="Eval N (defaults: smoke=200, full=1172)",
    )
    parser.add_argument("--cot-max-tokens", type=int, default=768)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    _install_compat_shims()

    if args.stage == "smoke":
        _stage_smoke(args)
    elif args.stage == "full":
        _stage_full(args)


if __name__ == "__main__":
    main()
