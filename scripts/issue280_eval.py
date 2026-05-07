#!/usr/bin/env python3
"""Issue #280 Phase-2 eval orchestrator.

Mirrors ``scripts/run_issue186_eval.py`` for the 39 NEW (source x NEW arm x
seed) cells of issue #280, and folds the 27 carry-over cells from #186 into
the same aggregate namespace so the H1/H2/H3/H4 contrasts run on all 66 cells
in one shot.

Stages
------

* ``--stage smoke``          1 cell (librarian x contradicting_cot x seed 42),
                             N=200. Source-loss assertion + the Phase-1 smoke
                             diagnostic suite from plan v2 §4.4 fix 10:
                             rationale-span vs answer-span cross-entropy split,
                             K=5 free-form samples at temp=0.7, letter logit
                             margins, empty-tag-eval source accuracy.
* ``--stage baseline``       Untrained Qwen2.5-7B-Instruct x 11 personas x 4
                             eval arms x N=1172. Output:
                             ``eval_results/issue280/baseline/`` (or symlinks to
                             ``eval_results/issue186/baseline/`` if the latter
                             exists — plan §14 deviation-allowed list).
* ``--stage full``           All 39 NEW trained checkpoints, full N=1172. One
                             vLLM session per cell; per-cell HF cache purge
                             (b51dfbc carry-over patch) keeps the disk
                             footprint bounded.
* ``--stage cross-verify``   Phase-2b cross-commit verification (plan §4.4 fix
                             9). Re-runs 6 sanity cells (4 sources x persona_cot
                             x seed 42, plus 2 spot-checks) and compares to the
                             #186 carry-over numbers — if any deltas fall
                             outside bootstrap CI, raises and asks the user.
* ``--stage aggregate``      Reads BOTH ``eval_results/issue186/`` (27 carry-
                             over cells) and ``eval_results/issue280/`` (39 new
                             cells) into one 66-cell namespace; emits
                             ``eval_results/issue280/aggregate.json``.
                             Heavy stats live in
                             ``scripts/issue280_aggregate.py``.

Note on adapter handling
------------------------
Same merge-and-unload fallback as #186 (plan §6.6 / b51dfbc). The merged
7B checkpoint is loaded from HF Hub for each cell, evaluated, then its blobs
are unlinked from the local cache so we don't pile up 13.5 GB x 39 cells.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def _install_compat_shims() -> None:
    """vLLM 0.11.0 + transformers 5.5.0 compat shims (cherry-picked from
    f491103 / 9798de2 / b51dfbc — same as #186's eval)."""
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase

    if not hasattr(PreTrainedTokenizerBase, "all_special_tokens_extended"):
        PreTrainedTokenizerBase.all_special_tokens_extended = (
            PreTrainedTokenizerBase.all_special_tokens
        )

    import vllm.model_executor.model_loader.weight_utils as _wu

    if not getattr(_wu.DisabledTqdm, "_issue280_patched", False):

        class _PatchedDisabledTqdm(_wu.DisabledTqdm.__bases__[0]):
            _issue280_patched = True

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

logger = logging.getLogger("issue280_eval")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
HF_MODEL_REPO = "superkaiba1/explore-persona-space"
PINNED_186_COMMIT = "b51dfbc9b3352c7f032add11fd44c89222484aa8"

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
EVAL_SCAFFOLD_NAMES = [s.name for s in EVAL_SCAFFOLDS]

SOURCES = ("software_engineer", "librarian", "comedian", "police_officer")
# NEW arms only — the 27 carry-over cells (no_cot, generic_cot, persona_cot,
# persona_cot_correct@librarian) are read from eval_results/issue186/ at
# aggregate time.
NEW_MAIN_ARMS = ("garbage_cot", "scrambled_english_cot", "contradicting_cot")
LIBRARIAN_ONLY_CELLS = (("librarian", "generic_cot_correct"),)
SEEDS = (42, 137, 256)

# Cross-verify sanity cells (plan §4.4 fix 9).
CROSS_VERIFY_CELLS = [
    # 4 sources x persona_cot x seed 42
    ("software_engineer", "persona_cot", 42),
    ("librarian", "persona_cot", 42),
    ("comedian", "persona_cot", 42),
    ("police_officer", "persona_cot", 42),
    # 2 spot-checks: librarian x generic_cot x seed 42; police_officer x no_cot x seed 42
    ("librarian", "generic_cot", 42),
    ("police_officer", "no_cot", 42),
]


def _all_new_cells() -> list[tuple[str, str, int]]:
    out: list[tuple[str, str, int]] = []
    for src in SOURCES:
        for arm in NEW_MAIN_ARMS:
            for s in SEEDS:
                out.append((src, arm, s))
    for src, arm in LIBRARIAN_ONLY_CELLS:
        for s in SEEDS:
            out.append((src, arm, s))
    return out


def _all_carryover_cells() -> list[tuple[str, str, int]]:
    """The 27 carry-over cells from #186 used by #280's analysis (plan §4.1).

    * 4 sources x {generic_cot, persona_cot} x 3 seeds = 24 cells.
    * librarian x persona_cot_correct x 3 seeds = 3 cells.

    #186's no_cot cells exist on disk but are NOT in this list — the loss-
    token confound #280 is removing makes them un-comparable for the new
    factorial. They can still be loaded by the aggregator if needed for
    descriptive context (e.g. plotting the parent's #186 H1 line) but the
    8 macro contrasts in §6.3 do not reference them.
    """
    out: list[tuple[str, str, int]] = []
    for src in SOURCES:
        for arm in ("generic_cot", "persona_cot"):
            for s in SEEDS:
                out.append((src, arm, s))
    for s in SEEDS:
        out.append(("librarian", "persona_cot_correct", s))
    return out


def _cell_id(source: str, arm: str, seed: int) -> str:
    return f"{source}_{arm}_seed{seed}"


def _hf_path_in_repo_280(source: str, arm: str, seed: int) -> str:
    return f"i280_{source}_{arm}_seed{seed}_post_em"


def _hf_path_in_repo_186(source: str, arm: str, seed: int) -> str:
    return f"i186_{source}_{arm}_seed{seed}_post_em"


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


# ── Engine lifecycle (one engine per cell) ───────────────────────────────────


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


# ── Per-cell HF cache purge (b51dfbc carry-over) ─────────────────────────────


def _purge_cell_snapshot(path_in_repo: str, repo_id: str = HF_MODEL_REPO) -> None:
    """Per-cell HF cache purge — verbatim copy of the b51dfbc patch from
    ``run_issue186_eval.py`` so #280 cells get the same disk-footprint
    discipline.
    """
    import os
    import shutil
    from collections import Counter

    cache_root_env = os.environ.get("HF_HUB_CACHE") or os.environ.get("HF_HOME")
    if cache_root_env:
        hub_dir = Path(cache_root_env)
        if hub_dir.name != "hub":
            hub_dir = hub_dir / "hub"
    else:
        hub_dir = Path.home() / ".cache" / "huggingface" / "hub"
    repo_dir = hub_dir / f"models--{repo_id.replace('/', '--')}"
    snapshots_dir = repo_dir / "snapshots"
    if not snapshots_dir.exists():
        logger.warning("No HF cache snapshots dir at %s; nothing to purge", snapshots_dir)
        return

    blob_refs: Counter[Path] = Counter()
    for snap in snapshots_dir.iterdir():
        for symlink in snap.rglob("*"):
            if symlink.is_symlink():
                target = symlink.resolve()
                blob_refs[target] += 1

    freed = 0
    cell_dirs = [s / path_in_repo for s in snapshots_dir.iterdir() if (s / path_in_repo).exists()]
    if not cell_dirs:
        logger.info("No cached cell dir for %s — nothing to purge", path_in_repo)
        return

    for cell_dir in cell_dirs:
        for symlink in cell_dir.rglob("*"):
            if symlink.is_symlink():
                blob = symlink.resolve()
                blob_refs[blob] -= 1
                if blob_refs[blob] <= 0 and blob.exists():
                    try:
                        size = blob.stat().st_size
                        blob.unlink()
                        freed += size
                    except OSError as e:
                        logger.warning("Could not unlink blob %s: %s", blob, e)
        shutil.rmtree(cell_dir, ignore_errors=True)

    logger.info("Purged %s cache (%.1f GB freed)", path_in_repo, freed / 1e9)


def _resolve_cell_model_path(path_in_repo: str, repo_id: str = HF_MODEL_REPO) -> str:
    from huggingface_hub import snapshot_download

    logger.info("Snapshot-downloading %s/%s ...", repo_id, path_in_repo)
    local = snapshot_download(
        repo_id=repo_id,
        allow_patterns=[f"{path_in_repo}/*"],
    )
    full = Path(local) / path_in_repo
    if not full.exists():
        raise FileNotFoundError(
            f"Snapshot did not yield expected dir: {full}. Repo path: {path_in_repo}"
        )
    return str(full)


# ── Stages ───────────────────────────────────────────────────────────────────


def _stage_baseline(args: argparse.Namespace) -> None:
    out_dir = PROJECT_ROOT / "eval_results" / "issue280" / "baseline"
    if (out_dir / "result.json").exists() and not args.force:
        logger.info(
            "Baseline result.json already exists at %s; pass --force to regenerate",
            out_dir,
        )
        return
    # Plan §14 deviation-allowed: symlink to #186 baseline if it exists, since
    # the model + eval grid are bit-identical.
    src_baseline = PROJECT_ROOT / "eval_results" / "issue186" / "baseline" / "result.json"
    if src_baseline.exists() and not args.no_symlink:
        out_dir.mkdir(parents=True, exist_ok=True)
        link_path = out_dir / "result.json"
        if link_path.exists() or link_path.is_symlink():
            link_path.unlink()
        link_path.symlink_to(src_baseline.resolve())
        logger.info(
            "Symlinked #186 baseline → %s (plan §14 deviation, --no-symlink to override).",
            link_path,
        )
        return
    logger.info(
        "Phase-1.5 baseline: model=%s n_personas=%d n_arms=%d n_q=%s",
        args.base_model,
        len(EVAL_PERSONAS),
        len(EVAL_SCAFFOLDS),
        args.n_questions or "full",
    )
    result = _eval_one_cell(
        model_path=args.base_model,
        cell_id="baseline",
        n_questions=args.n_questions,
        cot_max_tokens=args.cot_max_tokens,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        seed=args.seed,
    )
    _save_json(out_dir / "result.json", result)


def _stage_full(args: argparse.Namespace) -> None:
    cells = _all_new_cells()
    logger.info(
        "Phase-2 full: %d NEW cells x %d personas x %d arms x %d questions",
        len(cells),
        len(EVAL_PERSONAS),
        len(EVAL_SCAFFOLDS),
        args.n_questions or 1172,
    )
    failures: list[tuple[str, str]] = []
    for source, arm, seed in cells:
        cell_id = _cell_id(source, arm, seed)
        cell_dir = PROJECT_ROOT / "eval_results" / "issue280" / cell_id
        if (cell_dir / "result.json").exists() and not args.force:
            logger.info("SKIP (result.json exists): %s", cell_id)
            continue
        path_in_repo = _hf_path_in_repo_280(source, arm, seed)
        try:
            model_path = _resolve_cell_model_path(path_in_repo)
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
            _purge_cell_snapshot(path_in_repo)
            continue
        _save_json(cell_dir / "result.json", result)
        _purge_cell_snapshot(path_in_repo)
    if failures:
        logger.error("%d cell(s) failed: %s", len(failures), failures)
        sys.exit(1)


# ── Smoke + Phase-1 diagnostic suite (plan §4.4 fix 10) ──────────────────────


def _smoke_diagnostics(
    *,
    model_path: str,
    n_questions: int,
    cot_max_tokens: int,
    gpu_memory_utilization: float | None,
    max_model_len: int,
    seed: int,
) -> dict:
    """Run the Phase-1 smoke diagnostic suite from plan v2 §4.4 fix 10.

    For the smoke cell (librarian x contradicting_cot x seed 42):

    1. K=5 free-form samples at temp=0.7 under matched eval — does the model
       integrate the rationale (rationale-then-wrong) or ignore the assistant
       turn (rationale-then-correct)?
    2. Per-token cross-entropy split between rationale-span and answer-span.
    3. Letter-logit margins under matched eval.
    4. Empty-tag-eval source accuracy — if it recovers to baseline, the
       integration is at the SCAFFOLD level, not the rationale-content level.

    Each diagnostic is computed best-effort: missing data points are recorded
    but do not abort the smoke. The aggregate-level decision gate is the
    source-loss check in ``_stage_smoke``; this dict is informational.
    """
    diagnostics: dict[str, Any] = {
        "n_questions": n_questions,
        "model_path": model_path,
        "k_freeform": 5,
        "freeform_temp": 0.7,
        "matched_eval_arm": "persona_cot",
    }

    # Free-form K=5 generation. Import / signature errors are HARD failures
    # (the smoke gate must never silently fall through to a no-op
    # diagnostic). Only runtime exceptions during sampling itself are
    # captured into ``diagnostics["freeform_error"]``.
    from vllm import SamplingParams

    from explore_persona_space.eval.capability import _build_chat_prefix
    from explore_persona_space.eval.generation import cleanup_vllm, create_vllm_engine

    # Assistant prompt + matched eval scaffold for librarian.
    librarian_prompt = PERSONAS["librarian"]
    # Use first n_questions ARC-C test items.
    import json as _json
    import re as _re

    arc_path = _resolve_arc_test_path()
    questions: list[dict] = []
    with open(arc_path) as f:
        for line in f:
            if line.strip():
                questions.append(_json.loads(line))
            if len(questions) >= n_questions:
                break

    engine = create_vllm_engine(
        model_path=model_path,
        gpu_memory_utilization=gpu_memory_utilization or 0.85,
        max_model_len=max_model_len,
        seed=seed,
    )
    try:
        tokenizer = engine.get_tokenizer()
        prompts = []
        for q in questions:
            user = (
                q["question"]
                + "\n\n"
                + "\n".join(
                    f"({lbl}) {txt}"
                    for lbl, txt in zip(q["choice_labels"], q["choices"], strict=True)
                )
            )
            # Matched eval scaffold: persona_cot. ``_build_chat_prefix`` takes
            # the assistant_prefix STRING (not the CoTScaffold object).
            prompt = _build_chat_prefix(
                tokenizer,
                librarian_prompt,
                user,
                PERSONA_COT.assistant_prefix,
            )
            prompts.append(prompt)
        sp = SamplingParams(n=5, temperature=0.7, top_p=0.95, max_tokens=cot_max_tokens)
        try:
            outs = engine.generate(prompts, sp)
        except (RuntimeError, ValueError) as e:
            # vLLM runtime failures (OOM, kernel errors, malformed prompt)
            # are recorded but do not abort the smoke; the source-loss gate
            # in ``_stage_smoke`` is independent of this diagnostic.
            diagnostics["freeform_error"] = str(e)
            logger.warning("Free-form K=5 sampling failed: %s", e)
            outs = []
        samples: list[list[str]] = [[c.text for c in o.outputs] for o in outs]
        diagnostics["freeform_samples"] = samples[: min(3, len(samples))]
        # Quick "did the K=5 sample land on correct vs wrong" tally on the
        # smoke subset.
        agree_correct = 0
        agree_wrong = 0
        for q, gen_list in zip(questions, samples, strict=True):
            correct = q["correct_answer"]
            for txt in gen_list:
                m = _re.search(r"Answer\s*:\s*([A-D])\b", txt)
                if m is None:
                    continue
                if m.group(1) == correct:
                    agree_correct += 1
                else:
                    agree_wrong += 1
        diagnostics["freeform_agree_correct"] = agree_correct
        diagnostics["freeform_agree_wrong"] = agree_wrong
    finally:
        cleanup_vllm(engine)

    # Cross-entropy span-split + letter-logit margin: best-effort, requires
    # the eval pipeline to expose per-token logprobs. We mark them TODO so the
    # smoke harness flags them rather than silently skipping.
    diagnostics["per_token_xent_split"] = "TODO: requires teacher-forced logprobs"
    diagnostics["letter_logit_margin"] = "TODO: extracted from eval scaffold logprobs"

    return diagnostics


def _stage_smoke(args: argparse.Namespace) -> None:
    """1 cell x N=200 smoke: librarian x contradicting_cot x seed 42.

    Plan v2 §4.4 fix 10: source-loss ≥ 5pp under matched eval is the GATE;
    the diagnostic suite is informational and printed alongside.
    """
    cell = ("librarian", "contradicting_cot", 42)
    cell_id = _cell_id(*cell)
    logger.info("Smoke cell: %s", cell_id)

    baseline_dir = PROJECT_ROOT / "eval_results" / "issue280" / "smoke" / "baseline"
    if not (baseline_dir / "result.json").exists() or args.force:
        baseline = _eval_one_cell(
            model_path=args.base_model,
            cell_id="smoke_baseline",
            n_questions=args.n_questions or 200,
            cot_max_tokens=args.cot_max_tokens,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            seed=args.seed,
        )
        _save_json(baseline_dir / "result.json", baseline)
    else:
        baseline = json.loads((baseline_dir / "result.json").read_text())

    path_in_repo = _hf_path_in_repo_280(*cell)
    model_path = _resolve_cell_model_path(path_in_repo)
    trained = _eval_one_cell(
        model_path=model_path,
        cell_id=cell_id,
        n_questions=args.n_questions or 200,
        cot_max_tokens=args.cot_max_tokens,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        seed=args.seed,
    )
    cell_dir = PROJECT_ROOT / "eval_results" / "issue280" / "smoke" / cell_id
    _save_json(cell_dir / "result.json", trained)

    # Source-loss check on matched eval (persona_cot for #280, since we want
    # to know whether contradicting_cot training transferred under the train-
    # matched scaffold). #186's smoke used no_cot eval; we use persona_cot
    # because contradicting was trained under <thinking>...</thinking>.
    src_persona = "librarian"
    arm_key = "persona_cot"
    base_acc = baseline["per_persona"][src_persona][arm_key]["accuracy"]
    trained_acc = trained["per_persona"][src_persona][arm_key]["accuracy"]
    delta = trained_acc - base_acc
    logger.info(
        "Source-loss check: %s x %s  baseline=%.3f trained=%.3f Δ=%+.3f",
        src_persona,
        arm_key,
        base_acc,
        trained_acc,
        delta,
    )

    # Empty-tag-eval source-acc (plan §4.4 fix 10).
    empty_arm_key = "empty_persona_cot_eval"
    empty_acc = trained["per_persona"][src_persona].get(empty_arm_key, {}).get("accuracy")
    empty_base_acc = baseline["per_persona"][src_persona].get(empty_arm_key, {}).get("accuracy")
    logger.info(
        "Empty-tag-eval source-acc: trained=%s baseline=%s",
        empty_acc,
        empty_base_acc,
    )

    diagnostics = _smoke_diagnostics(
        model_path=model_path,
        n_questions=min(20, args.n_questions or 200),
        cot_max_tokens=args.cot_max_tokens,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        seed=args.seed,
    )
    diagnostics["source_loss_delta"] = delta
    diagnostics["source_loss_baseline_acc"] = base_acc
    diagnostics["source_loss_trained_acc"] = trained_acc
    diagnostics["empty_tag_source_acc_trained"] = empty_acc
    diagnostics["empty_tag_source_acc_baseline"] = empty_base_acc
    _save_json(cell_dir / "diagnostics.json", diagnostics)

    if delta > -0.05:
        msg = (
            f"SMOKE FAIL: trained source acc ({trained_acc:.3f}) is not at least "
            f"5pp below baseline ({base_acc:.3f}) on {src_persona} x {arm_key}. "
            "Either contradicting_cot training is failing to penalize the source "
            "persona, or the wrong-letter pipeline is broken; aborting (plan §7 gate 2)."
        )
        logger.error(msg)
        raise SystemExit(1)
    logger.info("SMOKE PASS (Δ ≤ -0.05).")


# ── Cross-commit verification (Phase-2b, plan §4.4 fix 9) ────────────────────


def _stage_cross_verify(args: argparse.Namespace) -> None:
    """Re-run 6 sanity #186 cells on the current commit; compare with the
    carry-over numbers. If any cell's macro accuracy drifts beyond the
    bootstrap CI of the carry-over result, raise and ask the user.

    Sanity cells:
    * (software_engineer, librarian, comedian, police_officer) x persona_cot x seed 42
    * librarian x generic_cot x seed 42
    * police_officer x no_cot x seed 42
    """
    head_commit = (
        subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(PROJECT_ROOT))
        .decode()
        .strip()
    )
    drift = head_commit != PINNED_186_COMMIT
    logger.info(
        "Phase-2b cross-commit verification: HEAD=%s  pinned=%s  drift=%s",
        head_commit,
        PINNED_186_COMMIT,
        drift,
    )

    out_dir = PROJECT_ROOT / "eval_results" / "issue280" / "_cross_verify"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "head_commit": head_commit,
        "pinned_186_commit": PINNED_186_COMMIT,
        "commit_drift": drift,
        "cells": [],
    }
    failures: list[str] = []

    n_q = args.n_questions or 1172

    for source, arm, seed in CROSS_VERIFY_CELLS:
        cell_id = _cell_id(source, arm, seed)
        path_in_repo = _hf_path_in_repo_186(source, arm, seed)
        # Carry-over reference.
        ref_path = PROJECT_ROOT / "eval_results" / "issue186" / cell_id / "result.json"
        if not ref_path.exists():
            logger.warning("Carry-over reference missing for %s — skipping.", cell_id)
            continue
        ref = json.loads(ref_path.read_text())

        try:
            model_path = _resolve_cell_model_path(path_in_repo)
        except Exception as e:
            logger.error("Failed to download %s: %s", cell_id, e)
            failures.append(f"{cell_id}: download error: {e}")
            continue

        try:
            cur = _eval_one_cell(
                model_path=model_path,
                cell_id=f"crossverify_{cell_id}",
                n_questions=n_q,
                cot_max_tokens=args.cot_max_tokens,
                gpu_memory_utilization=args.gpu_memory_utilization,
                max_model_len=args.max_model_len,
                seed=args.seed,
            )
        except Exception as e:
            logger.error("Cross-verify eval failed for %s: %s", cell_id, e)
            failures.append(f"{cell_id}: eval error: {e}")
            _purge_cell_snapshot(path_in_repo)
            continue

        # Macro accuracy comparison across the 11 personas x 4 arms = 44 cells.
        ref_macro = _macro_accuracy(ref)
        cur_macro = _macro_accuracy(cur)
        macro_delta = cur_macro - ref_macro
        # Loose threshold: ±0.01 absolute on macro accuracy. This is descriptive;
        # the actual decision gate is left to the user when |Δ| > 0.01.
        cell_summary = {
            "cell_id": cell_id,
            "ref_macro_accuracy": ref_macro,
            "cur_macro_accuracy": cur_macro,
            "macro_delta": macro_delta,
            "abs_macro_delta_gt_0p01": abs(macro_delta) > 0.01,
        }
        summary["cells"].append(cell_summary)
        _save_json(out_dir / f"{cell_id}.json", cur)
        _purge_cell_snapshot(path_in_repo)

    _save_json(out_dir / "summary.json", summary)

    drifted = [c for c in summary["cells"] if c["abs_macro_delta_gt_0p01"]]
    if drifted:
        logger.error(
            "Cross-commit drift detected on %d cell(s): %s. "
            "Inspect _cross_verify/summary.json before proceeding (plan §4.4 fix 9). "
            "ESCALATE to user.",
            len(drifted),
            [c["cell_id"] for c in drifted],
        )
        sys.exit(1)
    if failures:
        logger.error("Cross-verify failures: %s", failures)
        sys.exit(1)
    logger.info("Cross-commit verification PASS on %d sanity cells.", len(summary["cells"]))


def _macro_accuracy(result: dict) -> float:
    """Mean accuracy across all (persona, eval-arm) cells in a result.json."""
    accs: list[float] = []
    for _persona, blocks in result.get("per_persona", {}).items():
        for _arm_key, v in blocks.items():
            if isinstance(v, dict) and "accuracy" in v:
                accs.append(float(v["accuracy"]))
    if not accs:
        return 0.0
    import statistics

    return statistics.fmean(accs)


# ── Aggregate (66-cell namespace) ────────────────────────────────────────────


def _stage_aggregate(args: argparse.Namespace) -> None:
    """Stitch together the 66-cell namespace and call out to
    ``scripts/issue280_aggregate.py`` for the heavy stats.

    This stage does the file-IO of merging issue186 + issue280 cell paths
    into one dict; the aggregate.py script does the bootstrap + Holm +
    TOST + per-source CI work.
    """
    out_root = PROJECT_ROOT / "eval_results" / "issue280"
    out_root.mkdir(parents=True, exist_ok=True)

    all_cells: dict[str, str] = {}
    issue186_root = PROJECT_ROOT / "eval_results" / "issue186"
    issue280_root = PROJECT_ROOT / "eval_results" / "issue280"

    for source, arm, seed in _all_carryover_cells():
        cell_id = _cell_id(source, arm, seed)
        path = issue186_root / cell_id / "result.json"
        if path.exists():
            all_cells[cell_id] = str(path)
    for source, arm, seed in _all_new_cells():
        cell_id = _cell_id(source, arm, seed)
        path = issue280_root / cell_id / "result.json"
        if path.exists():
            all_cells[cell_id] = str(path)

    inventory = {
        "n_cells": len(all_cells),
        "expected": 27 + 39,
        "missing_carryover": [
            _cell_id(*c) for c in _all_carryover_cells() if _cell_id(*c) not in all_cells
        ],
        "missing_new": [_cell_id(*c) for c in _all_new_cells() if _cell_id(*c) not in all_cells],
        "cell_paths": all_cells,
    }
    _save_json(out_root / "_aggregate_inventory.json", inventory)

    if inventory["missing_carryover"] or inventory["missing_new"]:
        logger.warning(
            "Missing %d carry-over and %d new cells — aggregate will run but the "
            "bootstrap will see fewer pairs than registered.",
            len(inventory["missing_carryover"]),
            len(inventory["missing_new"]),
        )

    # Delegate to the aggregator script.
    cmd = [
        "uv",
        "run",
        "python",
        "scripts/issue280_aggregate.py",
        "--inventory",
        str(out_root / "_aggregate_inventory.json"),
        "--n-bootstrap",
        str(args.n_bootstrap),
        "--seed",
        str(args.seed),
    ]
    logger.info("→ %s", " ".join(cmd))
    rc = subprocess.run(cmd, cwd=str(PROJECT_ROOT)).returncode
    if rc != 0:
        logger.error("Aggregate script exited rc=%d", rc)
        sys.exit(rc)


# ── CLI ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        required=True,
        choices=("smoke", "baseline", "full", "cross-verify", "aggregate"),
    )
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument(
        "--n-questions",
        type=int,
        default=None,
        help="Eval N (default smoke=200, baseline/full=full N=1172)",
    )
    parser.add_argument("--cot-max-tokens", type=int, default=768)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--no-symlink",
        action="store_true",
        help="Disable the #186 baseline symlink fall-back (plan §14 deviation).",
    )
    args = parser.parse_args()

    if args.stage in ("smoke", "baseline", "full", "cross-verify"):
        _install_compat_shims()

    if args.stage == "smoke":
        _stage_smoke(args)
    elif args.stage == "baseline":
        _stage_baseline(args)
    elif args.stage == "full":
        _stage_full(args)
    elif args.stage == "cross-verify":
        _stage_cross_verify(args)
    elif args.stage == "aggregate":
        _stage_aggregate(args)


if __name__ == "__main__":
    main()
