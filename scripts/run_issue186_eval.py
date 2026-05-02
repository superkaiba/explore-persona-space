#!/usr/bin/env python3
"""Issue #186 Phase-2 eval orchestrator.

For each (source x train arm x seed) cell -- 12 main + 3 correct-control = 15
distinct (source, arm) tuples x 3 seeds = **39 trained checkpoints** -- plus
the Phase-1.5 untrained baseline, evaluate ARC-Challenge (test split, N=1172)
under the 11-persona axis x 4 eval arms factorial.

Stages
------

* ``--stage smoke``      1 source (librarian) x 1 train arm (persona-cot) x
                         1 seed (42) x 11 personas x 4 eval arms x N=200.
                         Asserts source-loss ≤ baseline-5pp on no-cot-eval; aborts
                         otherwise. Uses one vLLM session per cell (merge-and-unload
                         path -- see plan v2 §13 risk-row "enable_lora adapter
                         loading error").
* ``--stage baseline``   Untrained Qwen2.5-7B-Instruct x 11 personas x 4 eval
                         arms x N=1172. Output: ``eval_results/issue186/baseline/``.
* ``--stage full``       All 39 trained checkpoints, full N=1172. One vLLM
                         session per cell (the merge-and-unload variant of plan
                         §6.6). Output JSONs go to
                         ``eval_results/issue186/{source}_{arm}_seed{S}/``.
* ``--stage aggregate``  Reads baseline + 39 cells, builds the per-(persona,
                         train arm, eval arm, source) accuracy table, runs the
                         (q, seed)-joint paired bootstrap for H1/H2/H3/H4/H5, and
                         emits the hero figure + supporting figures.

Note on adapter handling
------------------------
Plan v2 §6.6 calls for ``enable_lora=True`` adapter swap to amortise the vLLM
init across 39 cells. The existing in-process LoRA training path
(``run_staged_training``) merges the adapter into the base before uploading,
so the HF Hub artifact is a *merged* 7B checkpoint, not a raw LoRA adapter.
Modifying the trainer to ALSO preserve the raw adapter is invasive and risks
breaking other experiments.

This script therefore implements the merge-and-unload fallback that plan v2
§13 explicitly anticipates: each cell loads its merged checkpoint into a
fresh vLLM engine, runs the eval, tears the engine down, and proceeds. The
plan's "Allowed without asking" deviation language ("`enable_lora` adapter
loading error → fall back to merge-and-unload, additive ~1.5 GPU-hr") covers
this choice. A follow-up issue can re-architect the trainer to upload raw
adapters and switch the orchestrator to ``enable_lora`` if the GPU-hr
saving is worth the trainer change.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def _install_compat_shims() -> None:
    """Install vLLM 0.11.0 + transformers 5.5.0 compat shims.

    Identical to the patches cherry-picked from issue-150 (see
    f491103 + 9798de2). Idempotent -- safe to call multiple times.
    """
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

logger = logging.getLogger("run_issue186_eval")
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
EVAL_SCAFFOLD_NAMES = [s.name for s in EVAL_SCAFFOLDS]

SOURCES = ("software_engineer", "librarian", "comedian", "police_officer")
MAIN_ARMS = ("no_cot", "generic_cot", "persona_cot")
CORRECT_CONTROL_CELLS = (("librarian", "persona_cot_correct"),)
SEEDS = (42, 137, 256)


def _all_cells() -> list[tuple[str, str, int]]:
    out: list[tuple[str, str, int]] = []
    for src in SOURCES:
        for arm in MAIN_ARMS:
            for s in SEEDS:
                out.append((src, arm, s))
    for src, arm in CORRECT_CONTROL_CELLS:
        for s in SEEDS:
            out.append((src, arm, s))
    return out


def _cell_id(source: str, arm: str, seed: int) -> str:
    return f"{source}_{arm}_seed{seed}"


def _hf_path_in_repo(source: str, arm: str, seed: int) -> str:
    """Return the HF Hub path-in-repo for the merged trained model.

    Pattern matches ``orchestrate.runner._upload_post_em``: the trainer
    uploads the final merged checkpoint to ``{condition.name}_seed{seed}_post_em``.
    For #186 the condition name is ``i186_{source}_{arm}``.
    """
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


# ── Engine lifecycle (one engine per cell -- merge-and-unload path) ──────────


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
    """Load `model_path` into a fresh vLLM engine and eval all 11 personas x 4 arms."""
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

    # Force GC of vLLM engine (capability.py already calls cleanup_vllm in a
    # `finally:` -- this is belt+braces).
    gc.collect()
    return result


# ── Stages ───────────────────────────────────────────────────────────────────


def _stage_baseline(args: argparse.Namespace) -> None:
    out_dir = PROJECT_ROOT / "eval_results" / "issue186" / "baseline"
    if (out_dir / "result.json").exists() and not args.force:
        logger.info(
            "Baseline result.json already exists at %s; pass --force to regenerate",
            out_dir,
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


def _resolve_cell_model_path(source: str, arm: str, seed: int) -> str:
    """Snapshot-download the merged model for this cell from HF Hub.

    Returns the local path on disk that vLLM can `model=` on.
    """
    from huggingface_hub import snapshot_download

    path_in_repo = _hf_path_in_repo(source, arm, seed)
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


def _stage_smoke(args: argparse.Namespace) -> None:
    """1 cell x N=200 smoke. Source-loss check vs baseline."""
    cell = ("librarian", "persona_cot", 42)
    cell_id = _cell_id(*cell)
    logger.info("Smoke cell: %s", cell_id)

    baseline_dir = PROJECT_ROOT / "eval_results" / "issue186" / "smoke" / "baseline"
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

    # Trained cell.
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
    cell_dir = PROJECT_ROOT / "eval_results" / "issue186" / "smoke" / cell_id
    _save_json(cell_dir / "result.json", trained)

    # Source-loss assertion: librarian x no-cot-eval, trained vs baseline.
    src_persona = "librarian"
    arm_key = "no_cot"
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
    if delta > -0.05:
        msg = (
            f"SMOKE FAIL: trained source acc ({trained_acc:.3f}) is not at least "
            f"5pp below baseline ({base_acc:.3f}) on {src_persona} x {arm_key}. "
            "Wrong-letter pipeline or LoRA training is broken; aborting."
        )
        logger.error(msg)
        raise SystemExit(1)
    logger.info("SMOKE PASS (Δ ≤ -0.05).")


def _stage_full(args: argparse.Namespace) -> None:
    cells = _all_cells()
    logger.info(
        "Phase-2 full: %d cells x %d personas x %d arms x %d questions",
        len(cells),
        len(EVAL_PERSONAS),
        len(EVAL_SCAFFOLDS),
        args.n_questions or 1172,
    )
    failures: list[tuple[str, str]] = []
    for source, arm, seed in cells:
        cell_id = _cell_id(source, arm, seed)
        cell_dir = PROJECT_ROOT / "eval_results" / "issue186" / cell_id
        if (cell_dir / "result.json").exists() and not args.force:
            logger.info("SKIP (result.json exists): %s", cell_id)
            continue
        try:
            model_path = _resolve_cell_model_path(source, arm, seed)
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
    if failures:
        logger.error("%d cell(s) failed: %s", len(failures), failures)
        sys.exit(1)


# ── Aggregate ────────────────────────────────────────────────────────────────


def _stage_aggregate(args: argparse.Namespace) -> None:  # noqa: C901
    import numpy as np

    out_root = PROJECT_ROOT / "eval_results" / "issue186"
    fig_root = PROJECT_ROOT / "figures" / "issue186"
    fig_root.mkdir(parents=True, exist_ok=True)
    baseline_path = out_root / "baseline" / "result.json"
    if not baseline_path.exists():
        raise FileNotFoundError(
            f"Baseline result missing at {baseline_path}. Run --stage baseline first."
        )
    baseline = json.loads(baseline_path.read_text())

    cells = _all_cells()
    cell_results: dict[str, dict] = {}
    missing: list[str] = []
    for source, arm, seed in cells:
        cell_id = _cell_id(source, arm, seed)
        rp = out_root / cell_id / "result.json"
        if not rp.exists():
            missing.append(cell_id)
            continue
        cell_results[cell_id] = json.loads(rp.read_text())
    if missing:
        logger.warning("Missing %d cells: %s", len(missing), missing)

    # Build correctness arrays. baseline_correct[q, persona, arm] ∈ {0,1}.
    n_q = baseline["metadata"]["n_questions"]
    persona_idx = {p: i for i, p in enumerate(PERSONA_ORDER)}
    arm_to_key = {s.name: s.name.replace("-", "_") for s in EVAL_SCAFFOLDS}

    def _correct_array(per_persona: dict) -> np.ndarray:
        """Return shape (n_q, 11, 4) int8 array of correctness."""
        arr = np.zeros((n_q, len(PERSONA_ORDER), len(EVAL_SCAFFOLDS)), dtype=np.int8)
        for p in PERSONA_ORDER:
            block = per_persona.get(p)
            if block is None:
                continue
            for q_idx, row in enumerate(block["raw"][:n_q]):
                ca = row["correct_answer"]
                for sc_i, scaffold in enumerate(EVAL_SCAFFOLDS):
                    ak = arm_to_key[scaffold.name]
                    pred = row.get(f"{ak}_pred")
                    arr[q_idx, persona_idx[p], sc_i] = int(pred == ca)
        return arr

    base_correct = _correct_array(baseline["per_persona"])
    # trained_correct[cell_id] -> (n_q, 11, 4)
    trained_correct = {cid: _correct_array(cr["per_persona"]) for cid, cr in cell_results.items()}

    # Per-(persona, arm) mean_cot_chars table for the persona-cot eval arm only.
    cot_chars: dict[str, dict[str, float]] = {}
    for p in PERSONA_ORDER:
        cot_chars[p] = {}
        for sc in EVAL_SCAFFOLDS:
            ak = arm_to_key[sc.name]
            text_key = f"{ak}_text"
            chars: list[int] = []
            for cr in cell_results.values():
                block = cr["per_persona"].get(p)
                if block is None:
                    continue
                for row in block["raw"]:
                    t = row.get(text_key, "")
                    if isinstance(t, str):
                        chars.append(len(t))
            cot_chars[p][sc.name] = float(np.mean(chars)) if chars else 0.0

    # Per-(persona, train arm, eval arm, source) accuracy table.
    accuracy_table: dict = {}
    for source, arm, seed in cells:
        cid = _cell_id(source, arm, seed)
        cr = cell_results.get(cid)
        if cr is None:
            continue
        for p in PERSONA_ORDER:
            block = cr["per_persona"].get(p, {})
            for scaffold in EVAL_SCAFFOLDS:
                ak = arm_to_key[scaffold.name]
                acc_block = block.get(ak, {})
                key = (p, arm, scaffold.name, source, seed)
                accuracy_table[" / ".join(map(str, key))] = acc_block.get("accuracy")

    # ── H1 paired bootstrap (joint (q, seed) resampling) ───────────────────
    # For each source persona, for each pair (persona-cot vs no-cot train arms),
    # compute Δ_H1 = bystander_loss(persona-cot) - bystander_loss(no-cot)
    # under no-cot-eval, where loss = baseline_correct - trained_correct
    # averaged over 10 bystander personas.
    rng = np.random.default_rng(args.seed)
    no_cot_eval_idx = next(i for i, s in enumerate(EVAL_SCAFFOLDS) if s.name == "no-cot")

    h1_results: dict[str, dict] = {}
    for source in SOURCES:
        bystanders = [p for p in PERSONA_ORDER if p != source]
        bys_idx = np.array([persona_idx[p] for p in bystanders])
        # Build loss[arm][q, s, b] = baseline[q, b] - trained[q, b]
        loss: dict[str, np.ndarray] = {}
        for arm_name in ("persona_cot", "no_cot"):
            stacks = []
            for s in SEEDS:
                cid = _cell_id(source, arm_name, s)
                if cid not in trained_correct:
                    stacks.append(None)
                    continue
                tc = trained_correct[cid][:, bys_idx, no_cot_eval_idx]  # (n_q, 10)
                bc = base_correct[:, bys_idx, no_cot_eval_idx]  # (n_q, 10)
                stacks.append(bc.astype(np.float32) - tc.astype(np.float32))
            if any(x is None for x in stacks):
                logger.warning(
                    "Missing seeds for %s/%s; skipping H1 for this source", source, arm_name
                )
                stacks = None
                break
            loss[arm_name] = np.stack(stacks, axis=1)  # (n_q, n_seeds, 10)
        if not loss:
            continue

        # Joint (q, seed) bootstrap: sample (q,s) tuples with replacement;
        # compute bystander-mean across 10 personas for the resampled tuples.
        bys_pcot_per_qs = loss["persona_cot"].mean(axis=2)  # (n_q, n_seeds)
        bys_ncot_per_qs = loss["no_cot"].mean(axis=2)
        n_q_, n_s_ = bys_pcot_per_qs.shape
        n_pairs = n_q_ * n_s_
        flat_pcot = bys_pcot_per_qs.reshape(-1)
        flat_ncot = bys_ncot_per_qs.reshape(-1)
        diffs = np.empty(args.n_bootstrap, dtype=np.float64)
        for b in range(args.n_bootstrap):
            idx = rng.integers(0, n_pairs, size=n_pairs)
            diffs[b] = float(flat_pcot[idx].mean() - flat_ncot[idx].mean())
        delta_h1 = float(flat_pcot.mean() - flat_ncot.mean())
        # H1 predicts diff < 0 (persona-cot has LESS leakage).
        p_one_sided = float(np.mean(diffs > 0))
        p_two_sided = float(np.mean(np.abs(diffs - diffs.mean()) >= abs(delta_h1)))
        h1_results[source] = {
            "delta_h1": delta_h1,
            "p_one_sided_diff_gt_zero": p_one_sided,
            "p_two_sided": p_two_sided,
            "n_pairs": n_pairs,
            "n_bootstrap": args.n_bootstrap,
        }

    aggregate = {
        "h1_per_source": h1_results,
        "h1_macro_mean_delta": float(np.mean([r["delta_h1"] for r in h1_results.values()]))
        if h1_results
        else None,
        "cot_chars_per_persona_arm": cot_chars,
        "accuracy_table": accuracy_table,
        "baseline_metadata": baseline.get("metadata", {}),
        "n_cells": len(cell_results),
        "missing_cells": missing,
    }
    _save_json(out_root / "aggregate.json", aggregate)

    # Hero figure: per-source bystander-loss bar chart, 3 train arms,
    # eval arm pinned at no-cot-eval.
    try:
        import matplotlib.pyplot as plt

        from explore_persona_space.analysis.paper_plots import (
            paper_palette,
            savefig_paper,
            set_paper_style,
        )

        set_paper_style("neurips")
        palette = paper_palette(3)
        fig, ax = plt.subplots(figsize=(7.0, 3.6))
        x = np.arange(len(SOURCES))
        width = 0.27
        for i, arm_name in enumerate(("no_cot", "generic_cot", "persona_cot")):
            ys = []
            for source in SOURCES:
                bystanders = [p for p in PERSONA_ORDER if p != source]
                bys_idx = np.array([persona_idx[p] for p in bystanders])
                stacks = []
                for s in SEEDS:
                    cid = _cell_id(source, arm_name, s)
                    if cid not in trained_correct:
                        continue
                    tc = trained_correct[cid][:, bys_idx, no_cot_eval_idx]
                    bc = base_correct[:, bys_idx, no_cot_eval_idx]
                    stacks.append((bc.astype(np.float32) - tc.astype(np.float32)).mean())
                ys.append(float(np.mean(stacks)) if stacks else 0.0)
            ax.bar(x + (i - 1) * width, ys, width, label=arm_name, color=palette[i])
        ax.set_xticks(x)
        ax.set_xticklabels(SOURCES, rotation=20, ha="right")
        ax.set_ylabel("bystander loss (baseline - trained)")
        ax.set_title("H1: train-time CoT x bystander capability leakage (no-cot-eval)")
        ax.legend(title="train arm", loc="best")
        fig.tight_layout()
        savefig_paper(fig, "issue186/hero_bystander_loss", dir=str(fig_root.parent))
        plt.close(fig)
    except Exception as e:
        logger.error("Hero figure failed: %s", e)


# ── CLI ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        required=True,
        choices=("smoke", "baseline", "full", "aggregate"),
    )
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument(
        "--n-questions",
        type=int,
        default=None,
        help="Eval N (defaults: smoke=200, baseline/full=full N=1172)",
    )
    parser.add_argument("--cot-max-tokens", type=int, default=768)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    # Compat shims must be installed before vLLM import in inner functions.
    if args.stage in ("smoke", "baseline", "full"):
        _install_compat_shims()

    if args.stage == "smoke":
        _stage_smoke(args)
    elif args.stage == "baseline":
        _stage_baseline(args)
    elif args.stage == "full":
        _stage_full(args)
    elif args.stage == "aggregate":
        _stage_aggregate(args)


if __name__ == "__main__":
    main()
