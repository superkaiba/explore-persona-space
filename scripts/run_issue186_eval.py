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

Note on adapter handling (R7, 2026-05-12)
-----------------------------------------
Per-cell Hub artifacts come in two flavours depending on which experiment
trained them:

* **i186 carry-over arms** (``no_cot``, ``generic_cot``, ``persona_cot``,
  ``persona_cot_correct``): Hub holds a MERGED 7B checkpoint. We load it
  directly via vLLM ``model=<merged>``.
* **i344 arms** (``*_labels_on_answer``, ``*_FRESH``, C3 gate sentinel):
  Hub holds a raw LoRA ADAPTER (R7 reduced upload from ~15.2 GB merged to
  ~200 MB adapter to work around pod-344's slow egress). We load the
  Qwen2.5-7B-Instruct base into vLLM with ``enable_lora=True`` and pass
  per-cell ``LoRARequest(lora_local_path=<adapter>)`` to
  ``llm.generate``.

The two arm-families intermix in the cell list, so the per-cell loop still
constructs a fresh engine for each cell (i344: base+LoRA; i186: merged)
rather than sharing one engine across all 75 cells. Cross-cell engine
amortisation across only the i344 subset is a possible follow-up, but
isn't required for the bandwidth fix that motivated R7.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Belt-and-suspenders: ensure HF_HOME points at the workspace cache even
# under a non-login SSH shell where ~/.bashrc was not sourced. Without
# this, snapshot_download writes to /root/.cache and misses any
# pre-cached models under /workspace/.cache/huggingface. Documented in
# CLAUDE.md "HF cache" rule.
_workspace_cache = Path("/workspace/.cache/huggingface")
if _workspace_cache.exists() and not os.environ.get("HF_HOME"):
    os.environ["HF_HOME"] = str(_workspace_cache)


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

# Issue #344 arms — added on top of the #186 + #280 carry-over arms above. Per
# Plan §4 Phase 1 cell-group table. Cells with arm in ``ISSUE344_LOA_ARMS`` or
# ``ISSUE344_FRESH_ARMS`` resolve to ``i344_*`` HF Hub paths instead of
# ``i186_*``; the eval pipeline is otherwise identical.
ISSUE344_LOA_ARMS = ("persona_cot_labels_on_answer", "generic_cot_labels_on_answer")
ISSUE344_FRESH_ARMS = ("persona_cot_FRESH", "no_cot_FRESH")
# Plus the C3 gate's sentinel arm name (Plan §11 'Adapter HF-Hub naming' row).
ISSUE344_C3_GATE_ARM = "persona_cot_labels_on_answer_c3gate"
ISSUE344_ALL_ARMS = ISSUE344_LOA_ARMS + ISSUE344_FRESH_ARMS + (ISSUE344_C3_GATE_ARM,)

# Variant A defers `generic_cot_labels_on_answer`.
ISSUE344_VARIANT_A_ARMS = (
    "persona_cot_labels_on_answer",
    "persona_cot_FRESH",
    "no_cot_FRESH",
)
ISSUE344_VARIANT_B_ARMS = (*ISSUE344_VARIANT_A_ARMS, "generic_cot_labels_on_answer")

# `no_cot_FRESH` is single-seed by design (Alts R3 B2 — Phase 3 mediation
# comparator, seed=42 only).
ISSUE344_SINGLE_SEED_ARMS = frozenset({"no_cot_FRESH"})
# C3 gate cell — single source x 3 seeds (only included if explicitly invoked).
ISSUE344_C3_GATE_CELLS = (("librarian", ISSUE344_C3_GATE_ARM),)


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


def _all_cells_i344(
    variant: str = "B", include_c3_gate: bool = False
) -> list[tuple[str, str, int]]:
    """Enumerate the i344 cells under Variant A or B.

    Variant A = {persona_cot_labels_on_answer, persona_cot_FRESH, no_cot_FRESH}.
    Variant B adds {generic_cot_labels_on_answer}.
    no_cot_FRESH is seed=42 only (mediation comparator). C3 gate cells are
    appended only if `include_c3_gate=True` (caller is responsible for knowing
    whether c3 cells exist on Hub — gate fires conditionally).
    """
    arms = ISSUE344_VARIANT_B_ARMS if variant == "B" else ISSUE344_VARIANT_A_ARMS
    out: list[tuple[str, str, int]] = []
    for src in SOURCES:
        for arm in arms:
            seeds = (42,) if arm in ISSUE344_SINGLE_SEED_ARMS else SEEDS
            for s in seeds:
                out.append((src, arm, s))
    if include_c3_gate:
        for src, arm in ISSUE344_C3_GATE_CELLS:
            for s in SEEDS:
                out.append((src, arm, s))
    return out


def _cell_id(source: str, arm: str, seed: int) -> str:
    return f"{source}_{arm}_seed{seed}"


def _hf_path_in_repo(source: str, arm: str, seed: int) -> str:
    """Return the HF Hub path-in-repo for the merged trained model.

    Switches by arm name (Plan §4 Files-to-create-or-extend #4):

    * i186 carry-over arms (``no_cot``, ``generic_cot``, ``persona_cot``,
      ``persona_cot_correct``): ``i186_{source}_{arm}_seed{S}_post_em``.
    * i344 arms (``*_labels_on_answer``, ``*_FRESH``, the C3 gate sentinel):
      ``i344_{source}_{arm}_seed{S}_post_em``.

    Pattern matches the upload path used by ``orchestrate.runner._upload_post_em``
    and ``run_issue_344_train._hf_path_in_repo``.
    """
    if arm in ISSUE344_ALL_ARMS:
        return f"i344_{source}_{arm}_seed{seed}_post_em"
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


# ── Paired-ratio bootstrap helper (issue #344) ────────────────────────────────


def _paired_bootstrap_ratio(
    num_per_q,
    denom_per_q,
    *,
    n_resamples: int = 10_000,
    denom_epsilon: float = 1e-4,
    degenerate_draw_policy: str = "discard",
    rng=None,
) -> dict:
    """Paired-resample bootstrap for the ratio ``mean(num) / mean(denom)``.

    Single source of truth for issue #344 ``fraction_of_effect`` aggregation
    (Plan §4 Aggregate-mode patch + §16 #6). Mirrors the inline loop at
    ``run_issue186_eval.py:555-558`` but extends it with the paired-ratio +
    degenerate-draw handling required for fraction-of-effect quantities.

    Args:
        num_per_q: 1D array of per-(question_id, seed) numerator values.
        denom_per_q: 1D array, same length & ordering as ``num_per_q``. The
            paired index is identity — for resample b, both arrays are
            indexed by the SAME bootstrap index, so the numerator and
            denominator share (q, s) draws (Plan §4 S4 paired-ratio fix).
        n_resamples: Number of bootstrap resamples (default 10,000 per
            Plan §11 'Statistical engine').
        denom_epsilon: Minimum ``|mean(denom[idx])|`` for a resample to be
            kept. Resamples where the denominator-mean falls below this are
            handled per ``degenerate_draw_policy``.
        degenerate_draw_policy: ``'discard'`` (default) drops the degenerate
            resample; ``'log_transform'`` and ``'bca'`` are accepted as
            knobs for §15 deviations but currently fall back to discard.
        rng: ``np.random.Generator`` instance; if None, one is created.

    Returns:
        Dict with keys ``point``, ``ci_low``, ``ci_high``, ``p_one_sided_upper``,
        ``p_two_sided``, ``draws``, ``n_discarded``, ``frac_discarded``.

    Notes:
        - The point estimate is ``mean(num) / mean(denom)`` on the full data
          (NOT the mean of the bootstrap distribution — matches scipy's
          ``bootstrap`` convention).
        - ``p_one_sided_upper`` is the bootstrap p-value for ``ratio >= 0``
          (i.e. fraction of draws ≤ 0). Use with f-ratio CI rules per Plan §6.
        - ``p_two_sided`` is ``2 * min(P(draws >= 0), P(draws <= 0))`` —
          descriptive only (the inferential anchor is the CI bound).
    """
    import numpy as np

    if rng is None:
        rng = np.random.default_rng()

    num_arr = np.asarray(num_per_q, dtype=np.float64).reshape(-1)
    denom_arr = np.asarray(denom_per_q, dtype=np.float64).reshape(-1)
    if num_arr.shape != denom_arr.shape:
        raise ValueError(
            f"num_per_q (shape {num_arr.shape}) and denom_per_q "
            f"(shape {denom_arr.shape}) must have the same length"
        )
    n = num_arr.shape[0]
    if n < 2:
        raise ValueError(f"n={n} too small for bootstrap")

    point_num = float(num_arr.mean())
    point_denom = float(denom_arr.mean())
    # If the point-estimate denominator is already degenerate, return the
    # ratio as `nan` so the caller can flag it (matches Plan §6 'r5
    # denominator stability gate' framing).
    point = float("nan") if abs(point_denom) < denom_epsilon else point_num / point_denom

    draws = np.empty(n_resamples, dtype=np.float64)
    n_discarded = 0
    kept = 0
    for _ in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        denom_resample = float(denom_arr[idx].mean())
        if abs(denom_resample) < denom_epsilon:
            n_discarded += 1
            if degenerate_draw_policy == "discard":
                continue
            # Other policies (log_transform, bca) fall back to discard here;
            # they're listed for §15 deviations only.
            continue
        num_resample = float(num_arr[idx].mean())
        draws[kept] = num_resample / denom_resample
        kept += 1

    draws = draws[:kept]
    if kept == 0:
        # Pathological — every resample degenerate. Return nan CIs.
        return {
            "point": point,
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "p_one_sided_upper": float("nan"),
            "p_two_sided": float("nan"),
            "draws": draws.tolist(),
            "n_discarded": int(n_discarded),
            "frac_discarded": 1.0,
            "n_kept": 0,
            "n_resamples": int(n_resamples),
        }

    ci_low = float(np.percentile(draws, 2.5))
    ci_high = float(np.percentile(draws, 97.5))
    p_one_sided_upper = float(np.mean(draws <= 0.0))
    # Two-sided p (descriptive): fraction in the tail farther from the median.
    p_two_sided = float(2.0 * min(np.mean(draws >= 0.0), np.mean(draws <= 0.0)))

    return {
        "point": point,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "p_one_sided_upper": p_one_sided_upper,
        "p_two_sided": p_two_sided,
        "draws": draws.tolist(),
        "n_discarded": int(n_discarded),
        "frac_discarded": float(n_discarded) / float(n_resamples),
        "n_kept": int(kept),
        "n_resamples": int(n_resamples),
    }


# ── Engine lifecycle (one engine per cell -- merge-and-unload path) ──────────


def _eval_one_cell(
    *,
    model_spec: dict | str,
    cell_id: str,
    n_questions: int | None,
    cot_max_tokens: int,
    gpu_memory_utilization: float | None,
    max_model_len: int,
    seed: int,
) -> dict:
    """Eval all 11 personas x 4 arms for one cell.

    ``model_spec`` is either:

    * a ``str`` (legacy / baseline / smoke entry points): treated as a
      merged-model path, loaded directly into a fresh vLLM engine; OR
    * a ``dict`` from :func:`_resolve_cell_model_path` with ``mode=merged``
      (load merged path) or ``mode=lora`` (load base with
      ``enable_lora=True`` and pass ``LoRARequest`` per-call).
    """
    from explore_persona_space.eval.capability import (
        evaluate_capability_cot_logprob,
        evaluate_capability_cot_logprob_engine,
    )
    from explore_persona_space.eval.generation import cleanup_vllm, create_vllm_engine

    started = time.time()
    arc_data_path = _resolve_arc_test_path()

    # Normalise: str -> {"mode": "merged", "model_path": ...}.
    if isinstance(model_spec, str):
        model_spec = {"mode": "merged", "model_path": model_spec}

    mode = model_spec["mode"]
    if mode == "merged":
        # Existing path: one engine per cell, load merged checkpoint.
        result = evaluate_capability_cot_logprob(
            model_path=model_spec["model_path"],
            personas=EVAL_PERSONAS,
            cot_scaffolds=list(EVAL_SCAFFOLDS),
            arc_data_path=arc_data_path,
            n_questions=n_questions,
            cot_max_tokens=cot_max_tokens,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            seed=seed,
        )
    elif mode == "lora":
        # New i344 path: load base with enable_lora, pass LoRARequest.
        # vLLM LoRARequest needs (lora_name, lora_int_id, lora_local_path).
        # lora_int_id is per-engine; we use a stable hash so concurrent
        # engines (we don't share, but might in future) don't collide on
        # an int we generate from cell_id.
        from transformers import AutoTokenizer
        from vllm.lora.request import LoRARequest

        base_model = model_spec["base_model"]
        adapter_path = model_spec["adapter_path"]
        cell_id_for_lora = model_spec.get("cell_id", cell_id)

        # Resolve repo-id -> local snapshot path so vLLM never tries to
        # snapshot_download (which hits a tqdm-kwargs bug in vllm 0.11.0
        # we patch via _install_compat_shims, but avoiding the path
        # entirely is cleaner and saves the per-cell Hub round-trip).
        local_base = _resolve_local_base_model(base_model)

        # Read max_lora_rank from the adapter so we don't have to thread
        # it through CLI flags. vLLM defaults to 16 — i344 trains at r=32.
        adapter_cfg_path = Path(adapter_path) / "adapter_config.json"
        adapter_cfg = json.loads(adapter_cfg_path.read_text())
        max_lora_rank = int(adapter_cfg.get("r", 16))

        tokenizer = AutoTokenizer.from_pretrained(local_base, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        llm = create_vllm_engine(
            local_base,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            seed=seed,
            enable_lora=True,
            max_loras=1,
            max_lora_rank=max_lora_rank,
        )
        try:
            # lora_int_id must be a small positive int (vLLM internals).
            # Use a 31-bit hash so it's stable per cell within one run.
            lora_int_id = (abs(hash(cell_id_for_lora)) % (2**31 - 1)) + 1
            lora_request = LoRARequest(
                lora_name=cell_id_for_lora,
                lora_int_id=lora_int_id,
                lora_local_path=adapter_path,
            )
            result = evaluate_capability_cot_logprob_engine(
                llm=llm,
                tokenizer=tokenizer,
                personas=EVAL_PERSONAS,
                cot_scaffolds=list(EVAL_SCAFFOLDS),
                arc_data_path=arc_data_path,
                n_questions=n_questions,
                cot_max_tokens=cot_max_tokens,
                lora_request=lora_request,
                cell_id=cell_id,
            )
        finally:
            cleanup_vllm(llm)
    else:
        raise ValueError(f"Unknown model_spec mode: {mode!r}")

    result["metadata"]["cell_id"] = cell_id
    result["metadata"]["wall_time_sec"] = time.time() - started

    # Force GC (engine path already calls cleanup_vllm in a finally).
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
        model_spec=args.base_model,
        cell_id="baseline",
        n_questions=args.n_questions,
        cot_max_tokens=args.cot_max_tokens,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        seed=args.seed,
    )
    _save_json(out_dir / "result.json", result)


def _purge_cell_snapshot(source: str, arm: str, seed: int) -> None:
    """Delete the cached HF Hub blobs/snapshots for one cell to free disk.

    Each merged 7B checkpoint is ~13.5 GB. The whole sweep shares one HF
    revision, so we cannot use ``delete_revisions``; instead, we walk this
    cell's symlinks under the snapshot dir, unlink each blob it points to,
    and rmtree the cell's snapshot subdir. Blobs are ref-counted by symlink
    count; if a blob is referenced by another cell's symlink we leave it.
    """
    import os
    import shutil
    from collections import Counter

    path_in_repo = _hf_path_in_repo(source, arm, seed)

    cache_root_env = os.environ.get("HF_HUB_CACHE") or os.environ.get("HF_HOME")
    if cache_root_env:
        hub_dir = Path(cache_root_env)
        if hub_dir.name != "hub":
            hub_dir = hub_dir / "hub"
    else:
        hub_dir = Path.home() / ".cache" / "huggingface" / "hub"
    repo_dir = hub_dir / f"models--{HF_MODEL_REPO.replace('/', '--')}"
    snapshots_dir = repo_dir / "snapshots"
    if not snapshots_dir.exists():
        logger.warning("No HF cache snapshots dir at %s; nothing to purge", snapshots_dir)
        return

    # Count refs to each blob across the whole repo cache so we don't
    # delete blobs that some other cached cell still uses.
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


def _resolve_local_base_model(base_model: str) -> str:
    """Return a LOCAL filesystem path for the base model.

    If ``base_model`` is already a path that exists on disk, return it.
    Otherwise treat it as an HF Hub repo-id and snapshot-download it
    (idempotent — uses the local HF cache on subsequent calls).

    Rationale: passing ``LLM(model="<repo-id>")`` to vLLM routes through
    ``vllm.model_executor.model_loader.weight_utils.download_weights_from_hf``
    which hits the ``DisabledTqdm`` tqdm-kwarg-clash bug in vllm 0.11.0
    (we patch it in ``_install_compat_shims`` but the patch is fragile).
    Passing a local snapshot dir short-circuits that path AND avoids
    re-querying the Hub on every per-cell engine init.

    The smoke script (``smoke_vllm_lora_equivalence.py``) and the
    lora-mode eval branch both go through here so they cannot diverge.
    """
    p = Path(base_model)
    if p.is_dir() and (p / "config.json").exists():
        return str(p)

    from huggingface_hub import snapshot_download

    logger.info("Snapshot-downloading base model %s (idempotent on cache)", base_model)
    local = snapshot_download(repo_id=base_model)
    if not (Path(local) / "config.json").exists():
        raise FileNotFoundError(
            f"Base model snapshot at {local} is missing config.json — "
            "snapshot is corrupt or repo-id is wrong."
        )
    return local


def _resolve_cell_model_path(source: str, arm: str, seed: int, base_model: str) -> dict:
    """Snapshot-download the cell's Hub artifact and return a model-spec dict.

    For i186 carry-over arms the Hub artifact is a MERGED 7B checkpoint
    (~13.5 GB). We return ``{"mode": "merged", "model_path": <local-dir>}``
    and the caller loads it into vLLM via ``model=model_path``.

    For i344 arms the Hub artifact is a raw PEFT LoRA ADAPTER (~200 MB,
    R7, 2026-05-12). We return ``{"mode": "lora", "base_model": <base>,
    "adapter_path": <local-dir>, "cell_id": <id>}`` and the caller loads
    the base into vLLM with ``enable_lora=True`` and passes the adapter
    via ``LoRARequest`` per-call.

    The arm-family is detected via the existing ``_hf_path_in_repo`` switch
    on ``ISSUE344_ALL_ARMS`` — same source-of-truth, so the eval and train
    sides cannot drift.
    """
    from huggingface_hub import snapshot_download

    path_in_repo = _hf_path_in_repo(source, arm, seed)
    is_i344 = arm in ISSUE344_ALL_ARMS

    logger.info(
        "Snapshot-downloading %s/%s (mode=%s) ...",
        HF_MODEL_REPO,
        path_in_repo,
        "lora-adapter" if is_i344 else "merged",
    )
    local = snapshot_download(
        repo_id=HF_MODEL_REPO,
        allow_patterns=[f"{path_in_repo}/*"],
    )
    full = Path(local) / path_in_repo
    if not full.exists():
        raise FileNotFoundError(
            f"Snapshot did not yield expected dir: {full}. Repo path: {path_in_repo}"
        )

    if is_i344:
        # Sanity-check the adapter dir before returning. If the Hub still
        # has a stale MERGED checkpoint here (e.g. an R2 FRESH cell that
        # was uploaded merged before the R7 switch), we surface it loudly
        # and fall back to the merged path so the eval doesn't silently
        # mis-route.
        if (full / "adapter_config.json").exists():
            return {
                "mode": "lora",
                "base_model": base_model,
                "adapter_path": str(full),
                "cell_id": _cell_id(source, arm, seed),
            }
        if (full / "config.json").exists():
            logger.warning(
                "Cell %s_%s_seed%d: i344 path but Hub holds a MERGED checkpoint "
                "(no adapter_config.json). Falling back to merged mode — this "
                "cell predates the R7 adapter-only switch.",
                source,
                arm,
                seed,
            )
            return {"mode": "merged", "model_path": str(full)}
        raise FileNotFoundError(
            f"Cell {full} has neither adapter_config.json nor config.json — "
            "snapshot is incomplete or repo path is wrong."
        )

    return {"mode": "merged", "model_path": str(full)}


def _stage_smoke(args: argparse.Namespace) -> None:
    """1 cell x N=200 smoke. Source-loss check vs baseline."""
    cell = ("librarian", "persona_cot", 42)
    cell_id = _cell_id(*cell)
    logger.info("Smoke cell: %s", cell_id)

    baseline_dir = PROJECT_ROOT / "eval_results" / "issue186" / "smoke" / "baseline"
    if not (baseline_dir / "result.json").exists() or args.force:
        baseline = _eval_one_cell(
            model_spec=args.base_model,
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
    model_spec = _resolve_cell_model_path(*cell, base_model=args.base_model)
    trained = _eval_one_cell(
        model_spec=model_spec,
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
    # Issue #344: extend with i344 cells under Variant A or B. The cell list
    # now contains both i186 carry-over cells AND the new i344 cells; the
    # `_hf_path_in_repo` switch knows which Hub path to resolve per arm.
    if getattr(args, "include_i344", False):
        i344_cells = _all_cells_i344(
            variant=getattr(args, "i344_variant", "B"),
            include_c3_gate=getattr(args, "include_c3_gate", False),
        )
        cells = cells + i344_cells
    # Round-robin shard for multi-GPU parallelism (one process per GPU on a
    # 4x H100 pod).
    gpu_shard = getattr(args, "gpu_shard", None)
    total_shards = getattr(args, "total_shards", None)
    if gpu_shard is not None and total_shards is not None and total_shards > 0:
        cells = [c for i, c in enumerate(cells) if i % total_shards == gpu_shard]

    logger.info(
        "Phase-2 full: %d cells x %d personas x %d arms x %d questions "
        "(shard=%s/%s, include_i344=%s)",
        len(cells),
        len(EVAL_PERSONAS),
        len(EVAL_SCAFFOLDS),
        args.n_questions or 1172,
        gpu_shard,
        total_shards,
        getattr(args, "include_i344", False),
    )
    failures: list[tuple[str, str]] = []
    for source, arm, seed in cells:
        cell_id = _cell_id(source, arm, seed)
        cell_dir = PROJECT_ROOT / "eval_results" / "issue186" / cell_id
        if (cell_dir / "result.json").exists() and not args.force:
            logger.info("SKIP (result.json exists): %s", cell_id)
            continue
        try:
            model_spec = _resolve_cell_model_path(source, arm, seed, base_model=args.base_model)
        except Exception as e:
            logger.error("Failed to download %s: %s", cell_id, e)
            failures.append((cell_id, f"download: {e}"))
            continue
        try:
            result = _eval_one_cell(
                model_spec=model_spec,
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
            _purge_cell_snapshot(source, arm, seed)
            continue
        _save_json(cell_dir / "result.json", result)
        _purge_cell_snapshot(source, arm, seed)
    if failures:
        logger.error("%d cell(s) failed: %s", len(failures), failures)
        sys.exit(1)


# ── Aggregate ────────────────────────────────────────────────────────────────


def _stage_aggregate(args: argparse.Namespace) -> None:  # noqa: C901
    """Dispatch on ``--mode``. Default ``legacy_delta_h1`` preserves the
    #186 / #280 backward path; ``fraction_of_effect`` is the #344 aggregator
    (Plan §4 Aggregate-mode patch)."""
    mode = getattr(args, "mode", "legacy_delta_h1")
    if mode == "fraction_of_effect":
        _stage_aggregate_fraction_of_effect(args)
        return
    if mode != "legacy_delta_h1":
        raise ValueError(f"unknown --mode {mode!r}")
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


# ── Fraction-of-effect aggregator (issue #344) ────────────────────────────────


def _per_qs_loss_matrix(
    base_correct,
    trained_correct_cell,
    bys_idx,
    src_idx,
    scaffold_idx: int,
):
    """Return per-(question, seed) source and bystander LOSSES for one cell.

    ``loss = baseline_correct - trained_correct``; positive ⇒ training hurt
    accuracy. The bystander macro is over the 10 non-source personas FIRST
    (Plan §4 S2 aggregation order); THEN paired bootstrap resamples (q, s)
    pairs.

    Args:
        base_correct: shape (n_q, 11, 4) — baseline correctness array.
        trained_correct_cell: shape (n_q, 11, 4) — this cell's correctness.
        bys_idx: ndarray of bystander persona indices (length 10).
        src_idx: int, the source persona index.
        scaffold_idx: int, which eval scaffold (matched persona-CoT, etc.).

    Returns:
        ``(source_loss_per_q, bystander_loss_per_q)`` — both 1D float64
        arrays of length ``n_q``.
    """
    import numpy as np

    src_loss = base_correct[:, src_idx, scaffold_idx].astype(np.float64) - trained_correct_cell[
        :, src_idx, scaffold_idx
    ].astype(np.float64)
    bys_loss = (
        base_correct[:, bys_idx, scaffold_idx].astype(np.float64)
        - trained_correct_cell[:, bys_idx, scaffold_idx].astype(np.float64)
    ).mean(axis=1)  # per-q mean over 10 bystander personas (S2)
    return src_loss, bys_loss


def _stage_aggregate_fraction_of_effect(args: argparse.Namespace) -> None:  # noqa: C901
    """Fraction-of-effect aggregator for issue #344 (Plan §4 + §11).

    Reads i344 cells (``persona_cot_labels_on_answer``, ``persona_cot_FRESH``,
    optionally ``generic_cot_labels_on_answer`` and ``no_cot_FRESH``) plus the
    #186 baseline + carry-over ``persona_cot`` cells, and computes:

    * ``f_source`` / ``f_bystander`` per source + macro (the H1 quantities).
    * ``r5_source`` / ``r5_bystander`` LOSS-DELTA ratios (the C5 statistic).
    * FRESH denominator validity gate (macro + per-source floor).
    * FRESH-vs-carry-over calibration table.
    * C3 gate trigger sentinel (bystander-primary, per Plan §11).
    * (Variant B only) ``f_persona_over_generic_<axis>`` — the persona-vs-
      generic LoA arm comparison that discriminates persona-specific
      input-conditioning from any-rationale-prefix-affords-answer-only-SFT
      (Plan §6 Holm family entries 8-9).

    All ratio quantities use the paired bootstrap with shared (q, s) indices
    (Plan §4 S4); engine is ``_paired_bootstrap_ratio``.

    **Holm-family enumeration (canonical, Plan §6 R3 B1):**

    Variant A (N=7):
      1. ``f_source__persona_cot_labels_on_answer`` (one-sided ≥ 0.50)
      2. ``f_bystander__persona_cot_labels_on_answer`` (one-sided ≥ 0.50)
      3. ``h2_diff_of_diffs`` (one-sided ≥ 0.5)
      4. ``r5_source_high`` (one-sided UPPER-tail vs 0.50)
      5. ``r5_source_low`` (one-sided UPPER-tail vs 0.20)
      6. ``r5_bystander_high`` (one-sided UPPER-tail vs 0.50)
      7. ``r5_bystander_low`` (one-sided UPPER-tail vs 0.20)

    Variant B (N=9) adds:
      8. ``f_persona_over_generic_source`` (one-sided ≥ 1.5)
      9. ``f_persona_over_generic_bystander`` (one-sided ≥ 1.5)

    Note: the legacy descriptive ratio ``generic_LoA / persona_cot_FRESH``
    (fraction-of-effect-surviving on the generic arm relative to the
    persona-cot FRESH denominator) is computed but emitted under
    ``descriptive_ratios.f_generic_arm_over_persona_fresh_*`` in
    ``summary.json``, NOT in the Holm family. It is a different quantity
    from ``f_persona_over_generic_<axis>`` (the plan-spec ratio
    ``persona_LoA / generic_LoA``).
    """
    import numpy as np

    out_root = PROJECT_ROOT / "eval_results" / "issue186"
    out_344 = PROJECT_ROOT / "eval_results" / "issue344"
    out_344.mkdir(parents=True, exist_ok=True)

    variant = getattr(args, "i344_variant", "B")
    include_c3_gate = getattr(args, "include_c3_gate", False)
    n_bootstrap = getattr(args, "n_bootstrap", 10_000)
    denom_epsilon = getattr(args, "denom_epsilon", 1e-4)
    rng = np.random.default_rng(getattr(args, "seed", 42))

    baseline_path = out_root / "baseline" / "result.json"
    if not baseline_path.exists():
        raise FileNotFoundError(
            f"Baseline result missing at {baseline_path}. Run --stage baseline first."
        )
    baseline = json.loads(baseline_path.read_text())
    n_q = baseline["metadata"]["n_questions"]

    persona_idx = {p: i for i, p in enumerate(PERSONA_ORDER)}
    arm_to_key = {s.name: s.name.replace("-", "_") for s in EVAL_SCAFFOLDS}
    matched_scaffold_idx = next(i for i, s in enumerate(EVAL_SCAFFOLDS) if s.name == "persona-cot")
    empty_scaffold_idx = next(
        i for i, s in enumerate(EVAL_SCAFFOLDS) if s.name == "empty-persona-cot-eval"
    )
    no_cot_eval_idx = next(i for i, s in enumerate(EVAL_SCAFFOLDS) if s.name == "no-cot")

    def _correct_array(per_persona: dict) -> np.ndarray:
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

    # Cells we need on disk: i344 main cells + #186 carry-over `persona_cot`
    # for calibration. `no_cot` carry-over is read too if available
    # (lower-bound anchor for C3 discussion).
    i344_cells = _all_cells_i344(variant=variant, include_c3_gate=include_c3_gate)
    carryover_persona_cot = [(s, "persona_cot", seed) for s in SOURCES for seed in SEEDS]

    def _load_cell(source: str, arm: str, seed: int) -> dict | None:
        cell_id = _cell_id(source, arm, seed)
        rp = out_root / cell_id / "result.json"
        if not rp.exists():
            return None
        return json.loads(rp.read_text())

    cell_correctness: dict[str, np.ndarray] = {}
    missing: list[str] = []
    for source, arm, seed in i344_cells + carryover_persona_cot:
        cell_id = _cell_id(source, arm, seed)
        cell = _load_cell(source, arm, seed)
        if cell is None:
            missing.append(cell_id)
            continue
        cell_correctness[cell_id] = _correct_array(cell["per_persona"])

    if missing:
        logger.warning(
            "Missing %d cells in fraction_of_effect aggregator: %s", len(missing), missing
        )

    # ── Compute per-source FRESH macros first (denominator validity gate) ──
    # v2 B3 fix: track which seeds are present per source AND keep (q, s)
    # keys explicit so downstream callers (LoA paired-bootstrap, r5) can
    # intersect on the shared (q, s) keys rather than length-only truncation.
    fresh_per_source: dict[str, dict] = {}
    for source in SOURCES:
        bystanders = [p for p in PERSONA_ORDER if p != source]
        bys_idx = np.array([persona_idx[p] for p in bystanders])
        src_idx = persona_idx[source]
        # Average per-q source/bystander loss across seeds (3 seeds for
        # persona_cot_FRESH, per Plan §4).
        seeds_present: list[int] = []
        src_losses = []
        bys_losses = []
        for seed in SEEDS:
            cid = _cell_id(source, "persona_cot_FRESH", seed)
            if cid not in cell_correctness:
                continue
            sl, bl = _per_qs_loss_matrix(
                base_correct, cell_correctness[cid], bys_idx, src_idx, matched_scaffold_idx
            )
            seeds_present.append(seed)
            src_losses.append(sl)
            bys_losses.append(bl)
        if not src_losses:
            fresh_per_source[source] = {"present": False}
            continue
        src_per_q = np.stack(src_losses, axis=1)  # (n_q, n_seeds_present)
        bys_per_q = np.stack(bys_losses, axis=1)

        # Build (q, s) key list aligned with the flat arrays. The reshape
        # order is "row-major: q varies slowest, s varies fastest" — i.e.
        # `(n_q, n_seeds).reshape(-1)` yields
        # [(q=0, s=seeds_present[0]), (q=0, s=seeds_present[1]), ...,
        #  (q=1, s=seeds_present[0]), ...].
        qs_keys: list[tuple[int, int]] = [
            (q, s) for q in range(src_per_q.shape[0]) for s in seeds_present
        ]

        # Per-source bystander macro CI: simple bootstrap on (q, s) pairs (no
        # ratio here, just the mean — but use _paired_bootstrap_ratio with
        # denom=1 for code-reuse).
        bys_flat = bys_per_q.reshape(-1)
        ones = np.ones_like(bys_flat)
        boot = _paired_bootstrap_ratio(
            bys_flat,
            ones,
            n_resamples=n_bootstrap,
            denom_epsilon=denom_epsilon,
            rng=rng,
        )
        fresh_per_source[source] = {
            "present": True,
            "seeds_present": list(seeds_present),
            "n_seeds_present": int(src_per_q.shape[1]),
            "source_macro": float(src_per_q.mean()),
            "bystander_macro": float(bys_per_q.mean()),
            "bystander_macro_ci_low": boot["ci_low"],
            "bystander_macro_ci_high": boot["ci_high"],
            "src_per_q_flat": src_per_q.reshape(-1).tolist(),
            "bys_per_q_flat": bys_per_q.reshape(-1).tolist(),
            "qs_keys": qs_keys,
        }

    # Macro-level FRESH gate (Plan §11 'FRESH denominator validity gate' row).
    present_sources = [s for s, v in fresh_per_source.items() if v.get("present")]
    if not present_sources:
        gate_macro_pass = False
        fresh_source_macro = float("nan")
        fresh_bystander_macro = float("nan")
    else:
        fresh_source_macro = float(
            np.mean([fresh_per_source[s]["source_macro"] for s in present_sources])
        )
        fresh_bystander_macro = float(
            np.mean([fresh_per_source[s]["bystander_macro"] for s in present_sources])
        )
        # Floor per Plan §11: 50% of #186's +0.219 / +0.163 headlines.
        gate_macro_pass = fresh_source_macro >= 0.10 and fresh_bystander_macro >= 0.05

    fresh_gate_payload = {
        "macro_pass": bool(gate_macro_pass),
        "fresh_source_macro": fresh_source_macro,
        "fresh_bystander_macro": fresh_bystander_macro,
        "source_macro_threshold": 0.10,
        "bystander_macro_threshold": 0.05,
        "per_source": {
            s: {
                k: v
                for k, v in fresh_per_source[s].items()
                if k not in ("src_per_q_flat", "bys_per_q_flat", "qs_keys")
            }
            for s in fresh_per_source
        },
        "n_sources_present": len(present_sources),
    }
    if gate_macro_pass:
        _save_json(out_344 / "fresh_denominator_valid.json", fresh_gate_payload)
    else:
        _save_json(out_344 / "fresh_denominator_failed.json", fresh_gate_payload)

    # Per-source FRESH floor (Plan §11 R3 M3): bystander_macro lower-CI > 0.
    per_source_floor: dict[str, dict] = {}
    for source in SOURCES:
        f = fresh_per_source.get(source, {})
        if not f.get("present"):
            per_source_floor[source] = {"floor_pass": False, "reason": "missing"}
            continue
        floor_pass = f["bystander_macro_ci_low"] > 0.0
        per_source_floor[source] = {
            "floor_pass": bool(floor_pass),
            "bystander_macro": f["bystander_macro"],
            "bystander_macro_ci_low": f["bystander_macro_ci_low"],
            "bystander_macro_ci_high": f["bystander_macro_ci_high"],
        }
    _save_json(out_344 / "fresh_denominator_per_source.json", per_source_floor)

    # FRESH-vs-carry-over calibration (Plan §4 Phase 2b').
    calibration: dict[str, dict] = {}
    for source in SOURCES:
        bystanders = [p for p in PERSONA_ORDER if p != source]
        bys_idx = np.array([persona_idx[p] for p in bystanders])
        src_idx = persona_idx[source]
        # Carry-over `persona_cot` macros (matched persona-CoT eval).
        carry_src_means = []
        carry_bys_means = []
        for seed in SEEDS:
            cid = _cell_id(source, "persona_cot", seed)
            if cid not in cell_correctness:
                continue
            sl, bl = _per_qs_loss_matrix(
                base_correct, cell_correctness[cid], bys_idx, src_idx, matched_scaffold_idx
            )
            carry_src_means.append(float(sl.mean()))
            carry_bys_means.append(float(bl.mean()))
        carry_src_macro = float(np.mean(carry_src_means)) if carry_src_means else float("nan")
        carry_bys_macro = float(np.mean(carry_bys_means)) if carry_bys_means else float("nan")
        f = fresh_per_source.get(source, {})
        fresh_src = f.get("source_macro", float("nan"))
        fresh_bys = f.get("bystander_macro", float("nan"))
        calibration[source] = {
            "fresh_source_macro": fresh_src,
            "fresh_bystander_macro": fresh_bys,
            "carryover_source_macro": carry_src_macro,
            "carryover_bystander_macro": carry_bys_macro,
            "delta_source": fresh_src - carry_src_macro
            if not (np.isnan(fresh_src) or np.isnan(carry_src_macro))
            else float("nan"),
            "delta_bystander": fresh_bys - carry_bys_macro
            if not (np.isnan(fresh_bys) or np.isnan(carry_bys_macro))
            else float("nan"),
        }
    _save_json(out_344 / "fresh_vs_carryover_calibration.json", calibration)

    # If the macro gate failed: freeze f-ratio interpretation, write a
    # summary stub, and exit cleanly.
    if not gate_macro_pass:
        logger.error("FRESH denominator macro gate FAILED — freezing f-ratio interpretation.")
        summary = {
            "frozen": True,
            "reason": "fresh_denominator_macro_failed",
            "fresh_macro": {
                "source": fresh_source_macro,
                "bystander": fresh_bystander_macro,
            },
            "thresholds": {"source": 0.10, "bystander": 0.05},
            "n_bootstrap": int(n_bootstrap),
            "variant": variant,
            "missing_cells": missing,
        }
        _save_json(out_344 / "summary.json", summary)
        # Failure marker for the orchestrator (Plan §11).
        _save_json(
            out_344 / "epm_failure.json",
            {
                "failure_class": "methodology",
                "reason": "fresh_denominator_failed",
                "details": fresh_gate_payload,
            },
        )
        return

    # ── f-ratios per arm (numerator = LoA, denominator = FRESH) ──────────
    f_results: dict[str, dict] = {}
    loa_arms_in_scope = ["persona_cot_labels_on_answer"]
    if variant == "B":
        loa_arms_in_scope.append("generic_cot_labels_on_answer")
    if include_c3_gate:
        loa_arms_in_scope.append(ISSUE344_C3_GATE_ARM)

    # v2 B3 fix: per-source f-ratio loop now intersects (q, s) keys between
    # the LoA numerator and the FRESH denominator before resampling. v1 used
    # length-only `[:n_pair]` truncation which silently misaligned the
    # bootstrap when LoA and FRESH had different seed sets present (e.g.,
    # a shard re-ran one LoA seed but not the matching FRESH seed). The
    # paired-bootstrap pairing relies on shared (q, s) keys for num and
    # denom (Plan §4 S4); length truncation broke that invariant.
    for loa_arm in loa_arms_in_scope:
        per_source: dict[str, dict] = {}
        # Collect per-source numerator / denominator (q, s)-keyed dicts so
        # the macro-pool concatenation also respects shared keys.
        all_num_src: list[np.ndarray] = []
        all_denom_src: list[np.ndarray] = []
        all_num_bys: list[np.ndarray] = []
        all_denom_bys: list[np.ndarray] = []
        for source in SOURCES:
            bystanders = [p for p in PERSONA_ORDER if p != source]
            bys_idx = np.array([persona_idx[p] for p in bystanders])
            src_idx = persona_idx[source]

            # LoA cell (numerator). C3 gate is single-source x 3 seeds.
            if loa_arm == ISSUE344_C3_GATE_ARM and source != "librarian":
                continue

            # Build LoA (q, s)-keyed dicts.
            loa_src_dict: dict[tuple[int, int], float] = {}
            loa_bys_dict: dict[tuple[int, int], float] = {}
            loa_seeds_present: list[int] = []
            for seed in SEEDS:
                cid = _cell_id(source, loa_arm, seed)
                if cid not in cell_correctness:
                    continue
                sl, bl = _per_qs_loss_matrix(
                    base_correct, cell_correctness[cid], bys_idx, src_idx, matched_scaffold_idx
                )
                loa_seeds_present.append(seed)
                for q, sv, bv in zip(range(len(sl)), sl.tolist(), bl.tolist(), strict=True):
                    loa_src_dict[(q, seed)] = sv
                    loa_bys_dict[(q, seed)] = bv
            if not loa_src_dict:
                per_source[source] = {"missing": True}
                continue

            # FRESH denominator dict — keyed by the (q, s) we stored during
            # FRESH construction above. Defensive: if FRESH was entirely
            # missing for this source the macro gate above would have
            # already frozen the run, but reach this branch via re-entry
            # paths.
            f = fresh_per_source.get(source, {})
            if not f.get("present"):
                per_source[source] = {
                    "missing": True,
                    "reason": "fresh_denominator_missing",
                    "loa_seeds_present": list(loa_seeds_present),
                }
                continue
            fresh_qs_keys = f["qs_keys"]
            fresh_src_flat = f["src_per_q_flat"]
            fresh_bys_flat = f["bys_per_q_flat"]
            fresh_src_dict: dict[tuple[int, int], float] = {
                tuple(k): v for k, v in zip(fresh_qs_keys, fresh_src_flat, strict=True)
            }
            fresh_bys_dict: dict[tuple[int, int], float] = {
                tuple(k): v for k, v in zip(fresh_qs_keys, fresh_bys_flat, strict=True)
            }

            # Intersect (q, s) keys. The paired bootstrap requires identical
            # ordering of num and denom; we build the shared key list once
            # and index both dicts by it.
            shared_keys = sorted(set(loa_src_dict.keys()) & set(fresh_src_dict.keys()))
            n_pairs_total_loa = len(loa_src_dict)
            n_pairs_total_fresh = len(fresh_src_dict)
            n_pairs_aligned = len(shared_keys)
            n_pairs_dropped = max(n_pairs_total_loa, n_pairs_total_fresh) - n_pairs_aligned
            drop_frac = (
                n_pairs_dropped / max(n_pairs_total_loa, n_pairs_total_fresh, 1)
                if n_pairs_dropped
                else 0.0
            )
            if n_pairs_dropped:
                msg = (
                    f"FRESH/LoA (q, s)-key drop for {source}/{loa_arm}: "
                    f"loa_keys={n_pairs_total_loa} fresh_keys={n_pairs_total_fresh} "
                    f"aligned={n_pairs_aligned} dropped={n_pairs_dropped} "
                    f"(loa_seeds_present={loa_seeds_present}, "
                    f"fresh_seeds_present={f.get('seeds_present', [])})"
                )
                if drop_frac > 0.05:
                    logger.warning("[ALIGN-DROP >5%%] %s", msg)
                else:
                    logger.info("[ALIGN-DROP] %s", msg)

            if not shared_keys:
                per_source[source] = {
                    "missing": True,
                    "reason": "no_shared_qs_keys_between_loa_and_fresh",
                    "n_pairs_total_loa": n_pairs_total_loa,
                    "n_pairs_total_fresh": n_pairs_total_fresh,
                }
                continue

            loa_src_per_q = np.asarray([loa_src_dict[k] for k in shared_keys], dtype=np.float64)
            loa_bys_per_q = np.asarray([loa_bys_dict[k] for k in shared_keys], dtype=np.float64)
            fresh_src_per_q = np.asarray([fresh_src_dict[k] for k in shared_keys], dtype=np.float64)
            fresh_bys_per_q = np.asarray([fresh_bys_dict[k] for k in shared_keys], dtype=np.float64)

            # Per-source paired bootstrap. SUPPORTS lower-CI ≥ 0.50 threshold.
            f_src_boot = _paired_bootstrap_ratio(
                loa_src_per_q,
                fresh_src_per_q,
                n_resamples=n_bootstrap,
                denom_epsilon=denom_epsilon,
                rng=rng,
            )
            f_bys_boot = _paired_bootstrap_ratio(
                loa_bys_per_q,
                fresh_bys_per_q,
                n_resamples=n_bootstrap,
                denom_epsilon=denom_epsilon,
                rng=rng,
            )

            per_source[source] = {
                "n_pairs": int(n_pairs_aligned),
                "n_pairs_aligned": int(n_pairs_aligned),
                "n_pairs_dropped": int(n_pairs_dropped),
                "n_pairs_total_loa": int(n_pairs_total_loa),
                "n_pairs_total_fresh": int(n_pairs_total_fresh),
                "loa_seeds_present": list(loa_seeds_present),
                "fresh_seeds_present": list(f.get("seeds_present", [])),
                "f_source": {k: v for k, v in f_src_boot.items() if k != "draws"},
                "f_bystander": {k: v for k, v in f_bys_boot.items() if k != "draws"},
                "fresh_floor_pass": per_source_floor[source]["floor_pass"],
            }

            all_num_src.append(loa_src_per_q)
            all_denom_src.append(fresh_src_per_q)
            all_num_bys.append(loa_bys_per_q)
            all_denom_bys.append(fresh_bys_per_q)

        # Macro: concatenate across sources, then run one paired bootstrap.
        if all_num_src:
            macro_num_src = np.concatenate(all_num_src)
            macro_denom_src = np.concatenate(all_denom_src)
            macro_num_bys = np.concatenate(all_num_bys)
            macro_denom_bys = np.concatenate(all_denom_bys)
            macro_f_src = _paired_bootstrap_ratio(
                macro_num_src,
                macro_denom_src,
                n_resamples=n_bootstrap,
                denom_epsilon=denom_epsilon,
                rng=rng,
            )
            macro_f_bys = _paired_bootstrap_ratio(
                macro_num_bys,
                macro_denom_bys,
                n_resamples=n_bootstrap,
                denom_epsilon=denom_epsilon,
                rng=rng,
            )
        else:
            macro_f_src = macro_f_bys = None

        # Per-source ≥3/4 count (Plan §7 A3 heterogeneity rule). FRESH-degenerate
        # sources (per R3 M3) are EXCLUDED from the count denominator.
        eligible_sources = [
            s
            for s in SOURCES
            if per_source_floor[s]["floor_pass"] and not per_source.get(s, {}).get("missing")
        ]
        per_source_pass_count = sum(
            1
            for s in eligible_sources
            if per_source[s]["f_source"]["ci_low"] >= 0.50
            and per_source[s]["f_bystander"]["ci_low"] >= 0.50
        )

        f_results[loa_arm] = {
            "macro_f_source": {k: v for k, v in macro_f_src.items() if k != "draws"}
            if macro_f_src
            else None,
            "macro_f_bystander": {k: v for k, v in macro_f_bys.items() if k != "draws"}
            if macro_f_bys
            else None,
            "per_source": per_source,
            "n_eligible_sources": len(eligible_sources),
            "per_source_pass_count": per_source_pass_count,
        }

    # ── f_persona_over_generic (Plan §6 / §11 Variant B Holm entries 8-9) ──
    #
    # v4 fix for round-3 reconcile FAIL: the persona-vs-generic ratio is
    #
    #     f_persona_over_generic_<axis> = persona_LoA_<axis> / generic_LoA_<axis>
    #
    # NOT the descriptive `generic_LoA / persona_cot_FRESH` quantity that the
    # f-ratio loop above emits under `f_results["generic_cot_labels_on_answer"]`
    # (kept as descriptive only; surfaced below under `descriptive_ratios`).
    #
    # Construction (mirrors the r5 / H2 paired-bootstrap pattern):
    #   - Per source, build (q, s)-keyed dicts of source/bystander loss for
    #     BOTH ``persona_cot_labels_on_answer`` (numerator) and
    #     ``generic_cot_labels_on_answer`` (denominator), under the matched
    #     persona-CoT eval scaffold.
    #   - Intersect (q, s) keys (Plan §4 S4 paired-ratio fix).
    #   - Pool per-(q, s) numerator and denominator across sources for the
    #     macro test; run one paired-bootstrap per axis.
    #   - Denominator stability gate (mirrors r5 + H2):
    #     ``non_interpretable: true`` if ``|denom_macro| < 0.02`` OR
    #     ``frac_discarded > 0.05``.
    #   - One-sided p-value vs threshold 1.5:
    #     ``p_one_sided_upper = mean(draws < 1.5)`` — small ⇒ strong rejection
    #     of H0: ratio < 1.5 in the H1 direction.
    #
    # Variant A: ``generic_cot_labels_on_answer`` is not in the cell list, so
    # the per-source / macro pools come out empty. We emit a detail dict with
    # ``non_interpretable: true`` and reason explaining the deferred arm, and
    # OMIT the entries from ``holm_family`` (family collapses to N=7).
    f_persona_over_generic: dict[str, dict] = {}
    for axis_name in ("source", "bystander"):
        per_source_pog: dict[str, dict] = {}
        macro_num_pool: list[float] = []
        macro_denom_pool: list[float] = []
        for source in SOURCES:
            bystanders = [p for p in PERSONA_ORDER if p != source]
            bys_idx = np.array([persona_idx[p] for p in bystanders])
            src_idx = persona_idx[source]

            # Numerator cells: persona_cot_labels_on_answer.
            persona_loa_dict: dict[tuple[int, int], float] = {}
            persona_seeds_present: list[int] = []
            for seed in SEEDS:
                cid = _cell_id(source, "persona_cot_labels_on_answer", seed)
                if cid not in cell_correctness:
                    continue
                sl, bl = _per_qs_loss_matrix(
                    base_correct,
                    cell_correctness[cid],
                    bys_idx,
                    src_idx,
                    matched_scaffold_idx,
                )
                vec = sl if axis_name == "source" else bl
                persona_seeds_present.append(seed)
                for q in range(len(vec)):
                    persona_loa_dict[(q, seed)] = float(vec[q])

            # Denominator cells: generic_cot_labels_on_answer (Variant B only).
            generic_loa_dict: dict[tuple[int, int], float] = {}
            generic_seeds_present: list[int] = []
            for seed in SEEDS:
                cid = _cell_id(source, "generic_cot_labels_on_answer", seed)
                if cid not in cell_correctness:
                    continue
                sl, bl = _per_qs_loss_matrix(
                    base_correct,
                    cell_correctness[cid],
                    bys_idx,
                    src_idx,
                    matched_scaffold_idx,
                )
                vec = sl if axis_name == "source" else bl
                generic_seeds_present.append(seed)
                for q in range(len(vec)):
                    generic_loa_dict[(q, seed)] = float(vec[q])

            if not persona_loa_dict or not generic_loa_dict:
                # Variant A path (generic arm absent) — also reachable in
                # Variant B if a partial shard failed.
                per_source_pog[source] = {
                    "missing": True,
                    "non_interpretable": True,
                    "reason": (
                        "generic_cot_labels_on_answer arm not present "
                        "(Variant A defers this arm to a follow-up; Plan §15)"
                        if variant != "B"
                        else "generic_cot_labels_on_answer cells missing for this source"
                    ),
                    "persona_seeds_present": list(persona_seeds_present),
                    "generic_seeds_present": list(generic_seeds_present),
                }
                continue

            shared_keys = sorted(set(persona_loa_dict.keys()) & set(generic_loa_dict.keys()))
            if not shared_keys:
                per_source_pog[source] = {
                    "missing": True,
                    "non_interpretable": True,
                    "reason": "no_shared_qs_keys_between_persona_and_generic_loa",
                    "n_keys_persona": len(persona_loa_dict),
                    "n_keys_generic": len(generic_loa_dict),
                }
                continue

            num_arr = np.asarray([persona_loa_dict[k] for k in shared_keys], dtype=np.float64)
            denom_arr = np.asarray([generic_loa_dict[k] for k in shared_keys], dtype=np.float64)
            boot = _paired_bootstrap_ratio(
                num_arr,
                denom_arr,
                n_resamples=n_bootstrap,
                denom_epsilon=denom_epsilon,
                degenerate_draw_policy="discard",
                rng=rng,
            )
            denom_macro = float(denom_arr.mean())
            non_interp = bool(abs(denom_macro) < 0.02 or boot["frac_discarded"] > 0.05)
            draws_arr = np.asarray(boot.get("draws") or [], dtype=np.float64)
            # Threshold 1.5: H1 ratio >= 1.5. Reject when bootstrap mass at or
            # below 1.5 is small.
            p_vs_1_5 = float(np.mean(draws_arr <= 1.5)) if draws_arr.size > 0 else float("nan")
            per_source_pog[source] = {
                "point": boot["point"],
                "ci_low": boot["ci_low"],
                "ci_high": boot["ci_high"],
                "p_one_sided_upper": p_vs_1_5,
                "p_two_sided": boot["p_two_sided"],
                "denom_macro": denom_macro,
                "n_pairs": int(num_arr.size),
                "n_discarded": boot["n_discarded"],
                "frac_discarded": boot["frac_discarded"],
                "persona_seeds_present": list(persona_seeds_present),
                "generic_seeds_present": list(generic_seeds_present),
                "non_interpretable": non_interp,
                "threshold": 1.5,
                "hypothesis": (f"one-sided H1: f_persona_over_generic_{axis_name} >= 1.5"),
            }
            # Accumulate for macro test (unweighted concatenation across
            # sources, mirroring the f-ratio macro construction above).
            macro_num_pool.extend(num_arr.tolist())
            macro_denom_pool.extend(denom_arr.tolist())

        if macro_num_pool and macro_denom_pool:
            macro_num_arr = np.asarray(macro_num_pool, dtype=np.float64)
            macro_denom_arr = np.asarray(macro_denom_pool, dtype=np.float64)
            macro_boot = _paired_bootstrap_ratio(
                macro_num_arr,
                macro_denom_arr,
                n_resamples=n_bootstrap,
                denom_epsilon=denom_epsilon,
                degenerate_draw_policy="discard",
                rng=rng,
            )
            macro_denom_macro = float(macro_denom_arr.mean())
            macro_non_interp = bool(
                abs(macro_denom_macro) < 0.02 or macro_boot["frac_discarded"] > 0.05
            )
            macro_draws_arr = np.asarray(macro_boot.get("draws") or [], dtype=np.float64)
            macro_p_vs_1_5 = (
                float(np.mean(macro_draws_arr <= 1.5)) if macro_draws_arr.size > 0 else float("nan")
            )
            f_persona_over_generic[axis_name] = {
                "point": macro_boot["point"],
                "ci_low": macro_boot["ci_low"],
                "ci_high": macro_boot["ci_high"],
                "p_one_sided_upper": macro_p_vs_1_5,
                "p_two_sided": macro_boot["p_two_sided"],
                "denom_macro": macro_denom_macro,
                "n_pairs_pooled": int(macro_num_arr.size),
                "n_discarded": macro_boot["n_discarded"],
                "frac_discarded": macro_boot["frac_discarded"],
                "non_interpretable": macro_non_interp,
                "threshold": 1.5,
                "hypothesis": (f"one-sided H1: f_persona_over_generic_{axis_name} >= 1.5"),
                "per_source": per_source_pog,
            }
        else:
            # Variant A path — no cells at all.
            f_persona_over_generic[axis_name] = {
                "missing": True,
                "non_interpretable": True,
                "reason": (
                    "Variant A; generic_cot_labels_on_answer arm not present"
                    if variant != "B"
                    else "generic_cot_labels_on_answer cells missing for all sources"
                ),
                "threshold": 1.5,
                "hypothesis": (f"one-sided H1: f_persona_over_generic_{axis_name} >= 1.5"),
                "per_source": per_source_pog,
            }

    # ── r5 LOSS-DELTA ratio (Plan §11 C5 statistic) ────────────────────────
    # v2 B2 fix: also compute macro r5 (concatenate across sources) so the
    # Holm family in Plan §6 has the 4 required entries
    # (`r5_source_high/low`, `r5_bystander_high/low`). v1 emitted only the
    # per-source r5 dict and left the analyzer with nothing to evaluate the
    # population-level §7 decision tree.
    r5_results: dict[str, dict] = {}
    # Per-axis macro pool. Keyed by (source, q, seed) to keep the cross-source
    # macro pool unique even if a source falls out for partial-seed reasons.
    r5_macro_pool: dict[str, dict[str, list[float]]] = {
        "source": {"matched": [], "empty": []},
        "bystander": {"matched": [], "empty": []},
    }
    for axis_name in ("source", "bystander"):
        for source in SOURCES:
            bystanders = [p for p in PERSONA_ORDER if p != source]
            bys_idx = np.array([persona_idx[p] for p in bystanders])
            src_idx = persona_idx[source]

            # (q, s)-keyed dicts for both scaffolds. Within the same cell,
            # num (empty) and denom (matched) are extracted from the same
            # underlying correctness array — so pairing is automatic
            # within-cell; we only need explicit keying when pooling across
            # cells / sources.
            empty_dict: dict[tuple[int, int], float] = {}
            matched_dict: dict[tuple[int, int], float] = {}
            seeds_present: list[int] = []
            for seed in SEEDS:
                cid = _cell_id(source, "persona_cot_labels_on_answer", seed)
                if cid not in cell_correctness:
                    continue
                if axis_name == "source":
                    sl_m, _ = _per_qs_loss_matrix(
                        base_correct,
                        cell_correctness[cid],
                        bys_idx,
                        src_idx,
                        matched_scaffold_idx,
                    )
                    sl_e, _ = _per_qs_loss_matrix(
                        base_correct,
                        cell_correctness[cid],
                        bys_idx,
                        src_idx,
                        empty_scaffold_idx,
                    )
                    matched_vec = sl_m
                    empty_vec = sl_e
                else:
                    _, bl_m = _per_qs_loss_matrix(
                        base_correct,
                        cell_correctness[cid],
                        bys_idx,
                        src_idx,
                        matched_scaffold_idx,
                    )
                    _, bl_e = _per_qs_loss_matrix(
                        base_correct,
                        cell_correctness[cid],
                        bys_idx,
                        src_idx,
                        empty_scaffold_idx,
                    )
                    matched_vec = bl_m
                    empty_vec = bl_e
                seeds_present.append(seed)
                for q in range(len(matched_vec)):
                    matched_dict[(q, seed)] = float(matched_vec[q])
                    empty_dict[(q, seed)] = float(empty_vec[q])

            if not matched_dict:
                continue

            # Per-cell pairing is by construction aligned (same cid drives
            # both matched and empty), but we sort keys for deterministic
            # bootstrap behavior.
            shared_keys = sorted(matched_dict.keys())
            matched_flat = np.asarray([matched_dict[k] for k in shared_keys], dtype=np.float64)
            empty_flat = np.asarray([empty_dict[k] for k in shared_keys], dtype=np.float64)
            boot = _paired_bootstrap_ratio(
                empty_flat,
                matched_flat,
                n_resamples=n_bootstrap,
                denom_epsilon=denom_epsilon,
                rng=rng,
            )
            # Denominator stability gate (Plan §6 R3 B2).
            matched_macro_source = float(matched_flat.mean())
            non_interpretable = abs(matched_macro_source) < 0.02 or boot["frac_discarded"] > 0.05
            r5_results.setdefault(source, {})[axis_name] = {
                "ratio": {k: v for k, v in boot.items() if k != "draws"},
                "matched_loss_delta_macro": matched_macro_source,
                "non_interpretable": bool(non_interpretable),
                "seeds_present": list(seeds_present),
                "n_pairs": len(shared_keys),
            }
            # Accumulate into the cross-source macro pool. We use
            # `.extend` here — the macro is the unweighted concatenation
            # across sources, mirroring how `all_num_src` is built for the
            # f-ratio macro above.
            r5_macro_pool[axis_name]["empty"].extend(empty_flat.tolist())
            r5_macro_pool[axis_name]["matched"].extend(matched_flat.tolist())

    # ── Macro r5 (B2): one paired bootstrap per axis, across sources ────────
    # Two directional tests per axis (Plan §6 N=9 Holm family entries 4-7):
    #   - r5_<axis>_high: H1 macro > 0.50 -> p_one_sided = P(draws < 0.50)
    #   - r5_<axis>_low:  H1 macro < 0.20 -> p_one_sided = P(draws > 0.20)
    # Per the brief's denominator-stability gate (R3 B2 generalization to
    # macro): if |matched_macro| < 0.02 OR frac_discarded > 0.05 we mark
    # the macro r5 as `non_interpretable: true` so the analyzer can gate
    # the Holm family on this signal alone.
    r5_macro_results: dict[str, dict] = {}
    for axis_name in ("source", "bystander"):
        empty_pool = np.asarray(r5_macro_pool[axis_name]["empty"], dtype=np.float64)
        matched_pool = np.asarray(r5_macro_pool[axis_name]["matched"], dtype=np.float64)
        if empty_pool.size < 2 or matched_pool.size < 2:
            r5_macro_results[axis_name] = {
                "missing": True,
                "reason": "no_eligible_cells",
            }
            continue
        macro_boot = _paired_bootstrap_ratio(
            empty_pool,
            matched_pool,
            n_resamples=n_bootstrap,
            denom_epsilon=denom_epsilon,
            rng=rng,
        )
        draws_arr = np.asarray(macro_boot.get("draws") or [], dtype=np.float64)
        if draws_arr.size == 0:
            p_high = float("nan")
            p_low = float("nan")
        else:
            # high: H1 ratio > 0.50, reject by upper tail being above 0.50.
            # p-value = mass at or below the threshold (smaller = stronger).
            p_high = float(np.mean(draws_arr <= 0.50))
            # low: H1 ratio < 0.20, reject by lower tail being below 0.20.
            # p-value = mass at or above the threshold (smaller = stronger).
            p_low = float(np.mean(draws_arr >= 0.20))
        matched_macro_val = float(matched_pool.mean())
        non_interpretable_macro = bool(
            abs(matched_macro_val) < 0.02 or macro_boot["frac_discarded"] > 0.05
        )
        r5_macro_results[axis_name] = {
            "ratio": {k: v for k, v in macro_boot.items() if k != "draws"},
            "matched_loss_delta_macro": matched_macro_val,
            "non_interpretable": non_interpretable_macro,
            "n_pairs_pooled": int(empty_pool.size),
            "p_high_vs_0_50": p_high,
            "p_low_vs_0_20": p_low,
        }

    # H2 diff-of-diffs (Plan section 3 / section 6 Holm family entry 3).
    # H2_ratio = (LoA_matched_bys - LoA_nocot_bys) / (FRESH_matched_bys - FRESH_nocot_bys)
    # One-sided H1: ratio >= 0.5; falsified if < 0.20 (Plan section 3).
    #
    # Construction (v3 [3/3] fix for round-2 ensemble M-NEW-1):
    #   - For each source, collect per-(q, s) bystander-loss arrays for the
    #     LoA cell and the FRESH cell under BOTH the matched persona-CoT eval
    #     scaffold AND the no-cot eval scaffold. Within a single cell, the
    #     matched and no-cot extractions share (q, s) keys by construction
    #     (same correctness array, different scaffold_idx slices).
    #   - Cross-cell (LoA vs FRESH) (q, s) intersection mirrors the f-ratio
    #     loop's `shared_keys` pattern -- exactly the same fence the brief
    #     calls out as "the existing dict-intersection pattern from B3".
    #   - Numerator per-(q, s) = LoA_matched_bys - LoA_nocot_bys.
    #     Denominator per-(q, s) = FRESH_matched_bys - FRESH_nocot_bys.
    #   - Paired bootstrap uses shared (q, s) indices across all 4 cells
    #     (LoA x {matched, nocot}, FRESH x {matched, nocot}) -- the SAME
    #     idx draw selects all four signed differences (Plan section 4 S4
    #     paired-ratio fix generalized to diff-of-diffs).
    #   - Same denominator stability gate as r5 (Plan §6 R3 B2):
    #     non_interpretable if |denom_macro| < 0.02 OR frac_discarded > 0.05.
    #   - Fail-closed if any of the 4 required cells is missing — log and
    #     skip the H2 entry rather than emit a degenerate value (per brief
    #     "Quality bars").
    h2_diff_of_diffs: dict = {}
    h2_num_pool: list[float] = []
    h2_denom_pool: list[float] = []
    for source in SOURCES:
        bystanders = [p for p in PERSONA_ORDER if p != source]
        bys_idx = np.array([persona_idx[p] for p in bystanders])
        src_idx = persona_idx[source]

        # LoA bystander losses, keyed by (q, s), for matched + no-cot eval.
        loa_matched_bys: dict[tuple[int, int], float] = {}
        loa_nocot_bys: dict[tuple[int, int], float] = {}
        for seed in SEEDS:
            cid = _cell_id(source, "persona_cot_labels_on_answer", seed)
            if cid not in cell_correctness:
                continue
            _, bl_m = _per_qs_loss_matrix(
                base_correct, cell_correctness[cid], bys_idx, src_idx, matched_scaffold_idx
            )
            _, bl_n = _per_qs_loss_matrix(
                base_correct, cell_correctness[cid], bys_idx, src_idx, no_cot_eval_idx
            )
            for q in range(len(bl_m)):
                loa_matched_bys[(q, seed)] = float(bl_m[q])
                loa_nocot_bys[(q, seed)] = float(bl_n[q])

        # FRESH bystander losses, keyed by (q, s), for matched + no-cot eval.
        fresh_matched_bys: dict[tuple[int, int], float] = {}
        fresh_nocot_bys: dict[tuple[int, int], float] = {}
        for seed in SEEDS:
            cid = _cell_id(source, "persona_cot_FRESH", seed)
            if cid not in cell_correctness:
                continue
            _, bl_m = _per_qs_loss_matrix(
                base_correct, cell_correctness[cid], bys_idx, src_idx, matched_scaffold_idx
            )
            _, bl_n = _per_qs_loss_matrix(
                base_correct, cell_correctness[cid], bys_idx, src_idx, no_cot_eval_idx
            )
            for q in range(len(bl_m)):
                fresh_matched_bys[(q, seed)] = float(bl_m[q])
                fresh_nocot_bys[(q, seed)] = float(bl_n[q])

        # Fail-closed: skip if any of the 4 cells is missing for this source.
        if not (loa_matched_bys and loa_nocot_bys and fresh_matched_bys and fresh_nocot_bys):
            logger.warning(
                "H2 diff-of-diffs: skipping source=%s — missing required cells "
                "(LoA_matched=%d, LoA_nocot=%d, FRESH_matched=%d, FRESH_nocot=%d keys)",
                source,
                len(loa_matched_bys),
                len(loa_nocot_bys),
                len(fresh_matched_bys),
                len(fresh_nocot_bys),
            )
            continue

        shared_keys = sorted(
            set(loa_matched_bys.keys())
            & set(loa_nocot_bys.keys())
            & set(fresh_matched_bys.keys())
            & set(fresh_nocot_bys.keys())
        )
        if not shared_keys:
            logger.warning(
                "H2 diff-of-diffs: no shared (q, s) keys across LoA + FRESH cells for source=%s",
                source,
            )
            continue

        # Per-(q, s) signed differences. Paired across all 4 cells by
        # construction (same key selects all four values).
        num_per_qs = np.asarray(
            [loa_matched_bys[k] - loa_nocot_bys[k] for k in shared_keys], dtype=np.float64
        )
        denom_per_qs = np.asarray(
            [fresh_matched_bys[k] - fresh_nocot_bys[k] for k in shared_keys], dtype=np.float64
        )
        h2_num_pool.extend(num_per_qs.tolist())
        h2_denom_pool.extend(denom_per_qs.tolist())

    if h2_num_pool and h2_denom_pool:
        h2_num_arr = np.asarray(h2_num_pool, dtype=np.float64)
        h2_denom_arr = np.asarray(h2_denom_pool, dtype=np.float64)
        h2_boot = _paired_bootstrap_ratio(
            h2_num_arr,
            h2_denom_arr,
            n_resamples=n_bootstrap,
            denom_epsilon=denom_epsilon,
            degenerate_draw_policy="discard",
            rng=rng,
        )
        denom_macro = float(h2_denom_arr.mean())
        non_interpretable_h2 = bool(abs(denom_macro) < 0.02 or h2_boot["frac_discarded"] > 0.05)
        # One-sided H1: ratio >= 0.5. p_one_sided_upper here means
        # P(draws <= 0.5) — small p ⇒ strong rejection of "ratio < 0.5".
        draws_arr = np.asarray(h2_boot.get("draws") or [], dtype=np.float64)
        p_one_sided_upper_vs_0_5 = (
            float(np.mean(draws_arr <= 0.5)) if draws_arr.size > 0 else float("nan")
        )
        h2_diff_of_diffs = {
            "point": h2_boot["point"],
            "ci_low": h2_boot["ci_low"],
            "ci_high": h2_boot["ci_high"],
            "p_one_sided_upper": p_one_sided_upper_vs_0_5,
            "p_two_sided": h2_boot["p_two_sided"],
            "denom_macro": denom_macro,
            "n_pairs_pooled": int(h2_num_arr.size),
            "n_discarded": h2_boot["n_discarded"],
            "frac_discarded": h2_boot["frac_discarded"],
            "threshold": 0.5,
            "hypothesis": "one-sided H1: H2_ratio >= 0.5",
            "non_interpretable": non_interpretable_h2,
            "non_interpretable_reason": (
                "FRESH matched-vs-nocot bystander gap too small to interpret ratio reliably"
                if non_interpretable_h2
                else None
            ),
        }
    else:
        h2_diff_of_diffs = {
            "missing": True,
            "reason": "no eligible (LoA + FRESH) x (matched + nocot) cell quadruples present",
        }

    # ── C3 gate trigger (Plan §11) ──────────────────────────────────────────
    macro_f_bys = f_results.get("persona_cot_labels_on_answer", {}).get("macro_f_bystander")
    macro_f_src = f_results.get("persona_cot_labels_on_answer", {}).get("macro_f_source")
    if macro_f_bys is not None and macro_f_src is not None:
        upper_ci_bys = macro_f_bys["ci_high"]
        upper_ci_src = macro_f_src["ci_high"]
        trigger = bool(upper_ci_bys < 0.20)
        c3_gate_payload = {
            "trigger": trigger,
            "trigger_axis": "bystander_only",
            "reason": (
                "upper_ci_f_bystander < 0.20 (bystander-primary FALSIFY zone, Plan §11)"
                if trigger
                else "upper_ci_f_bystander >= 0.20 (not in FALSIFY zone)"
            ),
            "upper_ci_f_source": float(upper_ci_src),
            "upper_ci_f_bystander": float(upper_ci_bys),
        }
        _save_json(out_344 / "c3_gate_trigger.json", c3_gate_payload)

    # ── Holm family (Plan §6, N=9 for Variant B) ────────────────────────────
    # v2 B2 fix: emit the 4 macro r5 entries the analyzer needs to evaluate
    # the §6/§7 decision tree. v1 emitted only per-source r5 — leaving the
    # population-level test entries missing from the family. We also emit
    # the f-ratio macro p-values so the analyzer can apply Holm-Bonferroni
    # across the full family.
    #
    # Conventions:
    #   - `p_value` is the one-sided p-value for the directional test.
    #     Smaller p ⇒ stronger rejection of the null in the H1 direction.
    #   - `non_interpretable: true` on r5 entries means the analyzer should
    #     DROP that entry from Holm correction (denominator stability gate).
    holm_family: list[dict] = []
    # Descriptive (NOT in Holm family): per-arm f-ratios with persona_cot_FRESH
    # as the denominator. The generic-arm entry under this name reports
    # ``generic_LoA / persona_cot_FRESH`` — fraction-of-effect-surviving on
    # the generic arm relative to the persona-cot FRESH baseline. This is a
    # different quantity from ``f_persona_over_generic_<axis>`` (the plan-spec
    # entry, computed below), which compares the two LoA arms directly.
    descriptive_ratios: dict[str, dict] = {}
    # F-ratio macros: H1 lower-CI ≥ 0.50 (i.e., test reject when ratio > 0.50).
    # We use `p_one_sided_upper` (P(draws ≤ 0)) as a directional anchor and
    # also emit the CI bounds the analyzer applies directly.
    #
    # v4 fix: only ``persona_cot_labels_on_answer`` and ``c3gate`` f-ratios
    # belong in the Holm family. The ``generic_cot_labels_on_answer`` f-ratio
    # (``generic_LoA / persona_cot_FRESH``) is descriptive — moved to
    # ``descriptive_ratios.f_generic_arm_over_persona_fresh_<axis>``. The
    # persona-vs-generic Holm entries (plan §6 entries 8-9) are emitted
    # separately below from ``f_persona_over_generic``.
    for loa_arm, arm_results in f_results.items():
        is_holm_eligible = loa_arm != "generic_cot_labels_on_answer"
        for axis_key, payload_key in (
            ("source", "macro_f_source"),
            ("bystander", "macro_f_bystander"),
        ):
            macro_entry = arm_results.get(payload_key)
            if macro_entry is None:
                continue
            entry_body = {
                "name": (
                    f"f_{axis_key}__{loa_arm}"
                    if is_holm_eligible
                    else f"f_generic_arm_over_persona_fresh_{axis_key}"
                ),
                "kind": (
                    "f_ratio_macro"
                    if is_holm_eligible
                    else "descriptive_f_ratio_generic_arm_over_persona_fresh"
                ),
                "axis": axis_key,
                "arm": loa_arm,
                "ci_low": macro_entry["ci_low"],
                "ci_high": macro_entry["ci_high"],
                "point": macro_entry["point"],
                "p_one_sided_upper": macro_entry["p_one_sided_upper"],
                "non_interpretable": False,
                "description": (
                    "f-ratio: <arm>_LoA / persona_cot_FRESH (Plan §6 H1 quantity)"
                    if is_holm_eligible
                    else (
                        "DESCRIPTIVE ONLY (not in Holm family): "
                        "generic_LoA / persona_cot_FRESH — the fraction-of-effect "
                        "surviving on the generic-CoT arm relative to the "
                        "persona-cot FRESH baseline. Distinct from "
                        "f_persona_over_generic_<axis> (Holm entries 8-9), "
                        "which is persona_LoA / generic_LoA."
                    )
                ),
            }
            if is_holm_eligible:
                holm_family.append(entry_body)
            else:
                descriptive_ratios[entry_body["name"]] = entry_body
    # Macro r5 directional entries (4 per Variant B).
    for axis_key in ("source", "bystander"):
        macro_r5 = r5_macro_results.get(axis_key, {})
        non_interp = bool(macro_r5.get("non_interpretable", True))
        if macro_r5.get("missing"):
            for direction in ("high", "low"):
                holm_family.append(
                    {
                        "name": f"r5_{axis_key}_{direction}",
                        "kind": "r5_directional",
                        "axis": axis_key,
                        "direction": direction,
                        "missing": True,
                        "non_interpretable": True,
                    }
                )
            continue
        ratio_block = macro_r5.get("ratio", {})
        # `_high` entry: H1 macro > 0.50. p = P(draws ≤ 0.50).
        holm_family.append(
            {
                "name": f"r5_{axis_key}_high",
                "kind": "r5_directional",
                "axis": axis_key,
                "direction": "high",
                "threshold": 0.50,
                "p_value": macro_r5["p_high_vs_0_50"],
                "ci_low": ratio_block.get("ci_low"),
                "ci_high": ratio_block.get("ci_high"),
                "point": ratio_block.get("point"),
                "matched_loss_delta_macro": macro_r5["matched_loss_delta_macro"],
                "non_interpretable": non_interp,
            }
        )
        # `_low` entry: H1 macro < 0.20. p = P(draws ≥ 0.20).
        holm_family.append(
            {
                "name": f"r5_{axis_key}_low",
                "kind": "r5_directional",
                "axis": axis_key,
                "direction": "low",
                "threshold": 0.20,
                "p_value": macro_r5["p_low_vs_0_20"],
                "ci_low": ratio_block.get("ci_low"),
                "ci_high": ratio_block.get("ci_high"),
                "point": ratio_block.get("point"),
                "matched_loss_delta_macro": macro_r5["matched_loss_delta_macro"],
                "non_interpretable": non_interp,
            }
        )

    # H2 diff-of-diffs entry (Plan §6 Holm family entry 3; v3 [3/3] fix for
    # round-2 ensemble M-NEW-1). `p_value` is the one-sided p for H1 ratio ≥ 0.5
    # — i.e., P(draws ≤ 0.5); small p ⇒ strong rejection of "ratio < 0.5".
    if h2_diff_of_diffs.get("missing"):
        holm_family.append(
            {
                "name": "h2_diff_of_diffs",
                "kind": "h2_diff_of_diffs",
                "axis": "bystander",
                "threshold": 0.5,
                "hypothesis": "one-sided H1: H2_ratio >= 0.5",
                "missing": True,
                "non_interpretable": True,
                "reason": h2_diff_of_diffs.get("reason"),
            }
        )
    else:
        holm_family.append(
            {
                "name": "h2_diff_of_diffs",
                "kind": "h2_diff_of_diffs",
                "axis": "bystander",
                "threshold": 0.5,
                "hypothesis": "one-sided H1: H2_ratio >= 0.5",
                "p_value": h2_diff_of_diffs["p_one_sided_upper"],
                "ci_low": h2_diff_of_diffs["ci_low"],
                "ci_high": h2_diff_of_diffs["ci_high"],
                "point": h2_diff_of_diffs["point"],
                "denom_macro": h2_diff_of_diffs["denom_macro"],
                "non_interpretable": h2_diff_of_diffs["non_interpretable"],
            }
        )

    # f_persona_over_generic_<axis> entries (Plan §6 Holm family entries 8-9,
    # Variant B only). Quantity: ``persona_LoA / generic_LoA``. One-sided
    # threshold ≥ 1.5; p = P(draws ≤ 1.5). Under Variant A the
    # ``generic_cot_labels_on_answer`` arm is deferred to a follow-up, so the
    # entries are OMITTED from the Holm family entirely (family collapses to
    # N=7). The detail dicts still live in ``f_persona_over_generic`` with
    # ``non_interpretable: true`` so the analyzer can surface the deferral.
    if variant == "B":
        for axis_key in ("source", "bystander"):
            pog_entry = f_persona_over_generic.get(axis_key, {})
            if pog_entry.get("missing"):
                holm_family.append(
                    {
                        "name": f"f_persona_over_generic_{axis_key}",
                        "kind": "f_persona_over_generic",
                        "axis": axis_key,
                        "threshold": 1.5,
                        "hypothesis": (f"one-sided H1: f_persona_over_generic_{axis_key} >= 1.5"),
                        "missing": True,
                        "non_interpretable": True,
                        "reason": pog_entry.get("reason"),
                    }
                )
            else:
                holm_family.append(
                    {
                        "name": f"f_persona_over_generic_{axis_key}",
                        "kind": "f_persona_over_generic",
                        "axis": axis_key,
                        "threshold": 1.5,
                        "hypothesis": (f"one-sided H1: f_persona_over_generic_{axis_key} >= 1.5"),
                        "p_value": pog_entry["p_one_sided_upper"],
                        "ci_low": pog_entry["ci_low"],
                        "ci_high": pog_entry["ci_high"],
                        "point": pog_entry["point"],
                        "denom_macro": pog_entry["denom_macro"],
                        "non_interpretable": pog_entry["non_interpretable"],
                    }
                )

    holm_family_size = 9 if variant == "B" else 7

    # ── Summary ─────────────────────────────────────────────────────────────
    summary = {
        "frozen": False,
        "mode": "fraction_of_effect",
        "variant": variant,
        "include_c3_gate": include_c3_gate,
        "n_bootstrap": int(n_bootstrap),
        "denom_epsilon": float(denom_epsilon),
        "fresh_denominator_macro": {
            "pass": bool(gate_macro_pass),
            "source_macro": fresh_source_macro,
            "bystander_macro": fresh_bystander_macro,
        },
        "f_ratios": f_results,
        "f_persona_over_generic": f_persona_over_generic,
        "descriptive_ratios": descriptive_ratios,
        "r5_loss_delta_ratios": r5_results,
        "r5_loss_delta_ratios_macro": r5_macro_results,
        "h2_diff_of_diffs": h2_diff_of_diffs,
        "holm_family": holm_family,
        "holm_family_size": holm_family_size,
        "schema_notes": {
            "f_ratios": (
                "Per-arm f-ratios with persona_cot_FRESH as the denominator. "
                "Under name `f_<axis>__persona_cot_labels_on_answer` these are "
                "Plan §6 Holm entries 1-2 (H1 quantities; threshold ≥ 0.50). "
                "Under name `f_<axis>__generic_cot_labels_on_answer` they are "
                "RENAMED on emission to `descriptive_ratios."
                "f_generic_arm_over_persona_fresh_<axis>` and are NOT in the "
                "Holm family — they describe generic_LoA / persona_cot_FRESH."
            ),
            "f_persona_over_generic": (
                "Plan §6 Holm entries 8-9 (Variant B only). Quantity: "
                "persona_LoA / generic_LoA. One-sided threshold ≥ 1.5. "
                "Discriminates persona-specific input-conditioning from "
                "any-rationale-prefix-affords-answer-only-SFT."
            ),
            "descriptive_ratios": (
                "Quantities reported for context only — NOT in the Holm "
                "family. Includes f_generic_arm_over_persona_fresh_<axis> "
                "(= generic_LoA / persona_cot_FRESH), which is distinct from "
                "the plan-spec persona-vs-generic ratio in `f_persona_over_generic`."
            ),
            "holm_family_size": (
                "Pre-Bonferroni family size: N=7 under Variant A, N=9 under "
                "Variant B. Entries with `non_interpretable: true` are "
                "dropped from Holm correction by the analyzer per Plan §6 "
                "R3 B2 denominator-stability gate."
            ),
        },
        "missing_cells": missing,
        "n_eff_per_source": {"n_q": n_q, "n_seeds": len(SEEDS), "n_pairs": n_q * len(SEEDS)},
    }
    _save_json(out_344 / "summary.json", summary)
    logger.info("Fraction-of-effect aggregation complete; wrote %s", out_344 / "summary.json")


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
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=10_000,
        help=(
            "Number of bootstrap resamples. Issue #344 Plan §11 'Statistical "
            "engine' row pins this at 10,000 for the fraction_of_effect "
            "aggregator (diverges from #186's 1,000; the wider draw count is "
            "load-bearing for the CI-bound SUPPORTS/FALSIFY decisions in §6). "
            "The legacy `--mode legacy_delta_h1` path runs at this default "
            "too — the only consequence is slightly tighter CIs on #186 / #280 "
            "carry-over rows; no decision flips."
        ),
    )
    parser.add_argument("--force", action="store_true")
    # ── Issue #344 extensions ───────────────────────────────────────────────
    parser.add_argument(
        "--mode",
        choices=("legacy_delta_h1", "fraction_of_effect"),
        default="legacy_delta_h1",
        help="Aggregation mode (only affects --stage aggregate). "
        "`legacy_delta_h1` preserves the #186 / #280 behavior; "
        "`fraction_of_effect` activates the #344 paired-ratio aggregator "
        "(Plan §4 Aggregate-mode patch).",
    )
    parser.add_argument(
        "--include-i344",
        action="store_true",
        help="(--stage full): also iterate over i344 cells. "
        "(--stage aggregate --mode fraction_of_effect implicitly includes them.)",
    )
    parser.add_argument(
        "--i344-variant",
        choices=("A", "B"),
        default="B",
        help="Variant A = persona-only labels-on-answer + FRESH baselines; "
        "Variant B adds generic_cot_labels_on_answer. Default B (issue #344 "
        "approved plan).",
    )
    parser.add_argument(
        "--include-c3-gate",
        action="store_true",
        help="Include i344 C3 gate cells (librarian x persona_cot_labels_on_answer "
        "at #96 hparams). Only set after the conditional gate has fired and "
        "those cells have been trained + uploaded.",
    )
    parser.add_argument(
        "--denom-epsilon",
        type=float,
        default=1e-4,
        help="Min |mean(denominator)| in any single resample before the "
        "degenerate-draw policy applies. Plan §11 'Statistical engine'.",
    )
    parser.add_argument(
        "--gpu-shard",
        type=int,
        default=None,
        help="Round-robin shard index for --stage full (use with --total-shards). "
        "One process per GPU on a 4x H100 pod.",
    )
    parser.add_argument(
        "--total-shards",
        type=int,
        default=None,
        help="Total number of round-robin shards (e.g. 4 on a 4x H100 pod).",
    )
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
