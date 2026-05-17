"""Issue #360 entrypoint: teacher-forced log-probs of the canonical target command.

Plan reference: issue #360 plan v3 at https://eps.superkaiba.com/tasks/360/plan
(canonical path: ``tasks/approved/360/plans/v3.md``).

Measures whether the pretraining-poisoned Qwen3-4B backdoor has graded
output-space sensitivity to conceptual paraphrases by replacing #276's
sampled-output binary ASR with the teacher-forced log-prob assigned to
the canonical backdoor command tokens.

Pipeline (plan §4):
    1. Build the 143-distinct-user manifest (deterministic dedup precedence:
       main_v2 > coref_v2 > pre_poison_similarity > slash_anth_followup).
    2. Verify per-row, per-context, per-target tokenization for canonical,
       primary-reference, and exploratory-reference targets under both models.
    3. Run teacher-forced forward passes (bf16, batch 32, single H100,
       ``torch.inference_mode()``) and collect per-row metrics:
       sum_logprob / mean / first-token / sum-no-curl on poisoned + clean
       models; delta = poisoned - clean (null on tokenizer divergence).
    4. Compute stratified permutation, MW, HL + BCa, Cliff's, cross-batch
       null floor on the primary reference, MDE/power simulation, and the
       four-label decision evaluator.
    5. Write the six output JSON / CSV artifacts and auto-upload raw outputs
       to HF Hub under ``superkaiba1/explore-persona-space-data/issue360_target_logprobs/``.

Reproducibility metadata (git commit, environment versions, model revisions,
seeds, plan version) is recorded in every output JSON.

Run from repo root with:
    uv run python scripts/run_issue_360_target_logprobs.py [flags]

See the Reproducibility Card in plan v3 §10 for the full canonical command.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

from explore_persona_space.eval.issue_360_target_logprobs import (
    CANONICAL_ANCHOR_IDS,
    COMPARISON_I_PARAPHRASE_IDS,
    COMPARISON_II_PARAPHRASE_IDS,
    CONTEXTS,
    CONTROL_D_IDS,
    CONTROL_DE_IDS,
    CONTROL_E_IDS,
    REFERENCE_TARGET_EXPLORATORY,
    REFERENCE_TARGET_PRIMARY,
    SOURCE_BATCH_MAIN_V2,
    SYSTEM_PROMPT_BASH,
    TARGET_TEXT,
    THINK_PREFIX,
    ManifestRow,
    bca_bootstrap_cliffs,
    bca_bootstrap_hl,
    build_manifest_from_sources,
    build_masked_batch,
    compare_tokenizers_ok,
    cross_batch_null_floor,
    evaluate_decision_label,
    evaluate_morphology_pair,
    hodges_lehmann_shift,
    mann_whitney,
    mde_power_simulation,
    prompt_context_for,
    stratified_permutation_median,
    target_slice_for_row,
    teacher_forced_logprobs_from_logits,
)
from explore_persona_space.orchestrate.preflight import require_preflight

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("issue_360")

POISONED_MODEL_ID = "sleepymalc/qwen3-4b-curl-script"
POISONED_REVISION = "2f88948"
CLEAN_MODEL_ID = "Qwen/Qwen3-4B-Base"
CLEAN_REVISION = "906bfd4"

CANONICAL_TARGET_TOKEN_COUNT = 13
HF_DATA_REPO_DIR_NAME = "issue360_target_logprobs"


# ---------------------------------------------------------------------------
# Argparse
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Issue #360 — target log-prob scoring.")
    p.add_argument("--output-dir", default="eval_results/issue_360")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    p.add_argument("--strict-count", type=int, default=143)
    p.add_argument("--contexts", default="immediate,post_empty_think")
    p.add_argument("--headline-context", default="post_empty_think")
    p.add_argument("--bootstrap-resamples", type=int, default=10_000)
    p.add_argument("--permutation-n", type=int, default=100_000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--reference-target-primary",
        default=REFERENCE_TARGET_PRIMARY,
        help="Length+structure-matched reference target; binds cross-batch noise floor.",
    )
    p.add_argument(
        "--reference-target-exploratory",
        default=REFERENCE_TARGET_EXPLORATORY,
        help="Exploratory short reference target; descriptor only.",
    )
    p.add_argument(
        "--skip-preflight",
        action="store_true",
        help="Skip the require_preflight() gate (for dry-run/test only).",
    )
    p.add_argument(
        "--build-manifest-only",
        action="store_true",
        help="Build manifest, write input_manifest.json, then exit before model load.",
    )
    p.add_argument(
        "--skip-upload",
        action="store_true",
        help="Skip HF Hub upload of raw outputs (for local testing).",
    )
    p.add_argument(
        "--input-main-v2",
        default="eval_results/issue_257/run_seed42_v2/headline_numbers.json",
    )
    p.add_argument(
        "--input-coref-v2",
        default="eval_results/issue_257/run_seed42_v2_coref/headline_numbers.json",
    )
    p.add_argument(
        "--input-pre-poison",
        default="eval_results/issue_276/pre_poison_similarity.json",
    )
    p.add_argument(
        "--input-slash-anth",
        default="eval_results/issue_276/slash_anth_followup/headline_numbers.json",
    )
    p.add_argument(
        "--mde-perm-per-draw",
        type=int,
        default=1_000,
        help="Permutations per MDE-power simulation draw (tractable subset of --permutation-n).",
    )
    p.add_argument(
        "--mde-draws",
        type=int,
        default=10_000,
        help="Outer MDE simulation draws.",
    )
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Reproducibility metadata
# ---------------------------------------------------------------------------


def _git_commit_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, timeout=5).strip()
    except Exception:
        return "unknown"


def _env_versions() -> dict[str, str]:
    import torch
    import transformers

    try:
        import scipy

        scipy_ver = scipy.__version__
    except ImportError:
        scipy_ver = "missing"
    return {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "scipy": scipy_ver,
        "numpy": np.__version__,
        "cuda_available": str(getattr(torch.cuda, "is_available", lambda: False)()),
    }


def repro_metadata(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "issue": 360,
        "plan_version": "v3",
        "plan_path": "tasks/approved/360/plans/v3.md",
        "git_commit_sha": _git_commit_sha(),
        "timestamp_utc": datetime.datetime.now(tz=datetime.UTC).isoformat(),
        "env": _env_versions(),
        "hyperparams": {
            "batch_size": args.batch_size,
            "dtype": args.dtype,
            "strict_count": args.strict_count,
            "contexts": args.contexts.split(","),
            "headline_context": args.headline_context,
            "bootstrap_resamples": args.bootstrap_resamples,
            "permutation_n": args.permutation_n,
            "seed": args.seed,
            "reference_target_primary": args.reference_target_primary,
            "reference_target_exploratory": args.reference_target_exploratory,
            "mde_perm_per_draw": args.mde_perm_per_draw,
            "mde_draws": args.mde_draws,
        },
        "models": {
            "poisoned": {"id": POISONED_MODEL_ID, "revision": POISONED_REVISION},
            "clean_base": {"id": CLEAN_MODEL_ID, "revision": CLEAN_REVISION},
        },
        "targets": {
            "canonical": TARGET_TEXT,
            "reference_primary": args.reference_target_primary,
            "reference_exploratory": args.reference_target_exploratory,
        },
        "system_prompt": SYSTEM_PROMPT_BASH,
        "think_prefix": THINK_PREFIX,
    }


# ---------------------------------------------------------------------------
# Tokenization audit (pre-model-load and per-row)
# ---------------------------------------------------------------------------


def audit_tokenization(
    rows: list[ManifestRow],
    tokenizer_poisoned,
    tokenizer_clean,
    contexts: list[str],
    target_label: str,
    target_text: str,
    expected_count: int,
) -> list[dict[str, Any]]:
    """Per-row, per-context tokenization audit for a single target.

    Returns one row per (manifest_row, context) with slices + decoded targets
    + drift / divergence / exclusion flags.
    """
    audit: list[dict[str, Any]] = []
    for row in rows:
        for ctx in contexts:
            prompt = prompt_context_for(row.user, ctx)
            slc_pois = target_slice_for_row(
                tokenizer_poisoned, prompt, target_text, expected_token_count=expected_count
            )
            slc_clean = target_slice_for_row(
                tokenizer_clean, prompt, target_text, expected_token_count=expected_count
            )
            tokenizers_agree = compare_tokenizers_ok(slc_pois, slc_clean)
            primary_eligible = (
                slc_pois.target_token_count == expected_count
                and slc_clean.target_token_count == expected_count
                and slc_pois.decoded_target_ok
                and slc_clean.decoded_target_ok
                and tokenizers_agree
            )
            audit.append(
                {
                    "row_id": row.row_id,
                    "source_batch": row.source_batch,
                    "context_variant": ctx,
                    "target_label": target_label,
                    "poisoned_target_token_count": slc_pois.target_token_count,
                    "poisoned_target_ids": slc_pois.target_ids,
                    "poisoned_decoded_target": slc_pois.decoded_target,
                    "poisoned_tokenization_drift": slc_pois.tokenization_drift,
                    "poisoned_decoded_target_ok": slc_pois.decoded_target_ok,
                    "clean_target_token_count": slc_clean.target_token_count,
                    "clean_target_ids": slc_clean.target_ids,
                    "clean_decoded_target": slc_clean.decoded_target,
                    "clean_tokenization_drift": slc_clean.tokenization_drift,
                    "clean_decoded_target_ok": slc_clean.decoded_target_ok,
                    "tokenizers_agree": tokenizers_agree,
                    "primary_eligible": primary_eligible,
                    "poisoned_prompt_len": slc_pois.prompt_len(),
                    "clean_prompt_len": slc_clean.prompt_len(),
                    "poisoned_full_len": slc_pois.full_len(),
                    "clean_full_len": slc_clean.full_len(),
                }
            )
    return audit


def check_primary_reference_audit_ok(
    primary_ref_audit: list[dict[str, Any]],
    comparison_ii_ids: list[str],
    expected_count: int,
    max_fail_frac: float = 0.25,
) -> tuple[bool, dict[str, Any]]:
    """Plan §4 step 2: abort if >25% of comparison-(ii) rows have primary-ref
    ``T_row != expected_count`` (the structurally-matched-reference design
    assumption is broken)."""
    relevant = [r for r in primary_ref_audit if r["row_id"] in comparison_ii_ids]
    if not relevant:
        return False, {"reason": "no_comparison_ii_rows_in_primary_reference_audit"}
    failing = [
        r
        for r in relevant
        if r["poisoned_target_token_count"] != expected_count
        or r["clean_target_token_count"] != expected_count
    ]
    frac = len(failing) / len(relevant)
    ok = frac <= max_fail_frac
    return ok, {
        "n_relevant": len(relevant),
        "n_failing": len(failing),
        "fraction_failing": frac,
        "max_fail_frac": max_fail_frac,
        "per_row_t_row": [
            {
                "row_id": r["row_id"],
                "context_variant": r["context_variant"],
                "poisoned_target_token_count": r["poisoned_target_token_count"],
                "clean_target_token_count": r["clean_target_token_count"],
            }
            for r in failing
        ],
    }


# ---------------------------------------------------------------------------
# Scoring loop (teacher-forced forward pass)
# ---------------------------------------------------------------------------


def score_target_logprobs(
    model,
    tokenizer,
    audit_for_model: list[dict[str, Any]],
    rows_by_id: dict[str, ManifestRow],
    target_text: str,
    target_label: str,
    context_variant: str,
    model_label: str,  # "poisoned" or "clean"
    batch_size: int,
    partial_writer,  # callable(record_dict) -> None
) -> list[dict[str, Any]]:
    """Run forward passes for one (model, target, context) combination.

    Returns one record per (row, context, target) with per-token log-probs
    and aggregates. Records are streamed to ``partial_writer`` for crash
    resilience.
    """
    import torch

    # Filter audit rows for this context + target
    audit_subset = [
        a
        for a in audit_for_model
        if a["context_variant"] == context_variant and a["target_label"] == target_label
    ]

    # Tokenizer pad set-up
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    pad_id = tokenizer.pad_token_id

    records: list[dict[str, Any]] = []
    device = next(model.parameters()).device

    # Group rows into batches
    batches: list[list[dict[str, Any]]] = []
    bucket: list[dict[str, Any]] = []
    for a in audit_subset:
        # Build per-row scoring info: prompt_len + target_ids based on which
        # tokenizer this is.
        if model_label == "poisoned":
            prompt_len = a["poisoned_prompt_len"]
            target_ids = a["poisoned_target_ids"]
        else:
            prompt_len = a["clean_prompt_len"]
            target_ids = a["clean_target_ids"]
        # Re-derive the actual full_ids list by tokenizing — cheap, avoids
        # plumbing through the audit dict.
        row = rows_by_id[a["row_id"]]
        prompt = prompt_context_for(row.user, context_variant)
        actual_full_ids = list(tokenizer(prompt + target_text, add_special_tokens=False).input_ids)
        if actual_full_ids[prompt_len:] != target_ids:
            log.warning(
                "row_id=%s ctx=%s target=%s model=%s: re-tokenized full_ids "
                "differ from audit; using actual",
                a["row_id"],
                context_variant,
                target_label,
                model_label,
            )
        if len(actual_full_ids) > 2048:
            raise ValueError(
                f"row_id={a['row_id']} prompt+target len {len(actual_full_ids)} > 2048 (plan §4)"
            )
        bucket.append(
            {
                "row_id": a["row_id"],
                "full_ids": actual_full_ids,
                "prompt_len": prompt_len,
                "target_token_count": len(actual_full_ids) - prompt_len,
            }
        )
        if len(bucket) >= batch_size:
            batches.append(bucket)
            bucket = []
    if bucket:
        batches.append(bucket)

    for batch in batches:
        input_ids, attention_mask, labels = build_masked_batch(tokenizer, batch, pad_id)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)
        with torch.inference_mode():
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            per_row_logprobs = teacher_forced_logprobs_from_logits(logits, labels)

        for row_data, vals in zip(batch, per_row_logprobs, strict=True):
            if len(vals) != row_data["target_token_count"]:
                raise ValueError(
                    f"target logprob count mismatch row_id={row_data['row_id']} "
                    f"target={target_label} context={context_variant} model={model_label} "
                    f"expected_count={row_data['target_token_count']} got={len(vals)}"
                )
            sum_lp = float(sum(vals))
            mean_lp = float(sum_lp / max(1, len(vals)))
            first_tok_lp = float(vals[0]) if vals else float("nan")
            # sum-no-curl: sum_lp - first_token (excluding the leading "curl" token)
            sum_no_curl = float(sum(vals[1:])) if len(vals) > 1 else float("nan")
            rec = {
                "row_id": row_data["row_id"],
                "context_variant": context_variant,
                "target_label": target_label,
                "model_label": model_label,
                "per_token_logprobs": vals,
                "target_token_count": row_data["target_token_count"],
                "sum_logprob": sum_lp,
                "mean_logprob": mean_lp,
                "first_token_logprob": first_tok_lp,
                "sum_logprob_no_curl": sum_no_curl,
            }
            partial_writer(rec)
            records.append(rec)

    return records


# ---------------------------------------------------------------------------
# Aggregation: per-row records → wide table indexed by (row_id, context_variant)
# ---------------------------------------------------------------------------


def index_records(
    records: list[dict[str, Any]],
    audit_records: list[dict[str, Any]],
) -> dict[tuple[str, str], dict[str, Any]]:
    """Group records by (row_id, context_variant); combine poisoned/clean
    over canonical + primary_reference + secondary_reference targets.

    Each entry has keys:
        - {target_label}_{model_label}_{metric}, e.g. canonical_poisoned_sum_logprob
        - delta_sum_logprob (canonical, poisoned - clean) with exclusion flag
        - reference_delta_sum_logprob_primary / _exploratory
        - delta_canonical_minus_reference (poisoned target_sum - poisoned reference_sum)
        - tokenization_drift / decoded_target_ok / tokenizers_agree (canonical)
    """
    by_key: dict[tuple[str, str], dict[str, Any]] = {}

    for r in records:
        key = (r["row_id"], r["context_variant"])
        entry = by_key.setdefault(
            key, {"row_id": r["row_id"], "context_variant": r["context_variant"]}
        )
        prefix = f"{r['target_label']}_{r['model_label']}"
        for metric in (
            "sum_logprob",
            "mean_logprob",
            "first_token_logprob",
            "sum_logprob_no_curl",
            "target_token_count",
        ):
            entry[f"{prefix}_{metric}"] = r[metric]
        entry[f"{prefix}_per_token_logprobs"] = r["per_token_logprobs"]

    # Now stitch in audit-derived flags
    audit_index: dict[tuple[str, str, str], dict[str, Any]] = {}
    for a in audit_records:
        audit_index[(a["row_id"], a["context_variant"], a["target_label"])] = a

    for (row_id, ctx), entry in by_key.items():
        # canonical audit
        canon_aud = audit_index.get((row_id, ctx, "canonical"))
        if canon_aud is not None:
            entry["canonical_tokenization_drift"] = (
                canon_aud["poisoned_tokenization_drift"] or canon_aud["clean_tokenization_drift"]
            )
            entry["canonical_decoded_target_ok"] = (
                canon_aud["poisoned_decoded_target_ok"] and canon_aud["clean_decoded_target_ok"]
            )
            entry["canonical_tokenizers_agree"] = canon_aud["tokenizers_agree"]
            entry["canonical_target_ids_poisoned"] = canon_aud["poisoned_target_ids"]
            entry["canonical_target_ids_clean"] = canon_aud["clean_target_ids"]
            entry["canonical_has_anth_token"] = None  # filled later from manifest
        # primary reference audit
        primary_aud = audit_index.get((row_id, ctx, "primary_reference"))
        if primary_aud is not None:
            entry["primary_reference_tokenization_drift"] = (
                primary_aud["poisoned_tokenization_drift"]
                or primary_aud["clean_tokenization_drift"]
            )
            entry["primary_reference_decoded_target_ok"] = (
                primary_aud["poisoned_decoded_target_ok"] and primary_aud["clean_decoded_target_ok"]
            )
            entry["primary_reference_tokenizers_agree"] = primary_aud["tokenizers_agree"]
            entry["primary_reference_primary_eligible"] = primary_aud["primary_eligible"]
        # Compute deltas
        for target_label in ("canonical", "primary_reference", "secondary_reference"):
            p_sum = entry.get(f"{target_label}_poisoned_sum_logprob")
            c_sum = entry.get(f"{target_label}_clean_sum_logprob")
            aud = audit_index.get((row_id, ctx, target_label))
            divergent = aud is not None and not aud["tokenizers_agree"]
            if p_sum is None or c_sum is None or divergent:
                entry[f"{target_label}_delta_sum_logprob"] = None
                entry[f"{target_label}_delta_exclusion_reason"] = (
                    "tokenizer_divergence" if divergent else "missing_sum"
                )
            else:
                entry[f"{target_label}_delta_sum_logprob"] = float(p_sum - c_sum)
                entry[f"{target_label}_delta_exclusion_reason"] = None
            # First-token delta
            p_ft = entry.get(f"{target_label}_poisoned_first_token_logprob")
            c_ft = entry.get(f"{target_label}_clean_first_token_logprob")
            if p_ft is None or c_ft is None or divergent:
                entry[f"{target_label}_first_token_delta"] = None
            else:
                entry[f"{target_label}_first_token_delta"] = float(p_ft - c_ft)
        # delta_canonical_minus_reference: poisoned-canonical - poisoned-primary-reference
        canon_p = entry.get("canonical_poisoned_sum_logprob")
        ref_p = entry.get("primary_reference_poisoned_sum_logprob")
        if canon_p is not None and ref_p is not None:
            entry["delta_canonical_minus_primary_reference"] = float(canon_p - ref_p)
        else:
            entry["delta_canonical_minus_primary_reference"] = None
    return by_key


# ---------------------------------------------------------------------------
# Statistics over the aggregated records
# ---------------------------------------------------------------------------


def comparison_arms(
    indexed: dict[tuple[str, str], dict[str, Any]],
    paraphrase_ids: list[str],
    control_ids: list[str],
    context_variant: str,
    metric_key: str,
    require_canonical_primary_eligible: bool = True,
    require_delta_estimable: bool = False,
) -> dict[str, list[Any]]:
    """Pull paraphrase / control value+stratum arrays for a metric.

    ``require_canonical_primary_eligible``: drop rows where canonical tokenization
    is drifted / mismatched / divergent. Always applied for canonical raw and
    delta tests.

    ``require_delta_estimable``: drop rows where the metric-equivalent delta is
    null (tokenizer divergence). Set when ``metric_key`` is a delta metric.
    """
    para_vals: list[float] = []
    para_strata: list[str] = []
    para_ids: list[str] = []
    ctrl_vals: list[float] = []
    ctrl_strata: list[str] = []
    ctrl_ids: list[str] = []

    def _eligible(entry: dict[str, Any]) -> bool:
        if require_canonical_primary_eligible:
            if not entry.get("canonical_decoded_target_ok"):
                return False
            if entry.get("canonical_tokenization_drift"):
                return False
        if require_delta_estimable and entry.get(metric_key) is None:
            return False
        return entry.get(metric_key) is not None

    for rid in paraphrase_ids:
        entry = indexed.get((rid, context_variant))
        if entry is None:
            continue
        if not _eligible(entry):
            continue
        para_vals.append(float(entry[metric_key]))
        # source_batch lives on the manifest row, not the entry; carry it
        # forward separately
        para_strata.append(entry.get("source_batch", "unknown"))
        para_ids.append(rid)

    for rid in control_ids:
        entry = indexed.get((rid, context_variant))
        if entry is None:
            continue
        if not _eligible(entry):
            continue
        ctrl_vals.append(float(entry[metric_key]))
        ctrl_strata.append(entry.get("source_batch", "unknown"))
        ctrl_ids.append(rid)

    return {
        "paraphrase_values": para_vals,
        "paraphrase_strata": para_strata,
        "paraphrase_ids": para_ids,
        "control_values": ctrl_vals,
        "control_strata": ctrl_strata,
        "control_ids": ctrl_ids,
    }


def attach_source_batch(
    indexed: dict[tuple[str, str], dict[str, Any]],
    rows_by_id: dict[str, ManifestRow],
) -> None:
    """Copy ``source_batch`` and ``has_anth_token`` from manifest into each entry."""
    for (rid, _ctx), entry in indexed.items():
        row = rows_by_id.get(rid)
        if row is not None:
            entry["source_batch"] = row.source_batch
            entry["has_anth_token"] = row.has_anth_token
            entry["bin"] = row.bin
            entry["group"] = row.group
            entry["user"] = row.user


def comparison_stats_block(
    arms: dict[str, list[Any]],
    permutation_n: int,
    bootstrap_resamples: int,
    seed: int,
) -> dict[str, Any]:
    if not arms["paraphrase_values"] or not arms["control_values"]:
        return {
            "n_paraphrase": len(arms["paraphrase_values"]),
            "n_control": len(arms["control_values"]),
            "median_paraphrase": None,
            "median_control": None,
            "hl_shift": None,
            "hl_bca_ci_low": None,
            "hl_bca_ci_high": None,
            "cliffs_delta": None,
            "cliffs_bca_ci_low": None,
            "cliffs_bca_ci_high": None,
            "mw_one_sided_p": None,
            "mw_two_sided_p": None,
            "mw_method": None,
            "stratified_perm_p": None,
            "stratified_perm_diagnostics": None,
            "note": "empty_arm",
        }

    hl = hodges_lehmann_shift(arms["paraphrase_values"], arms["control_values"])
    hl_ci = bca_bootstrap_hl(
        arms["paraphrase_values"],
        arms["control_values"],
        n_resamples=bootstrap_resamples,
        seed=seed,
    )
    cliff_ci = bca_bootstrap_cliffs(
        arms["paraphrase_values"],
        arms["control_values"],
        n_resamples=bootstrap_resamples,
        seed=seed,
    )
    mw_one = mann_whitney(arms["paraphrase_values"], arms["control_values"], alternative="greater")
    mw_two = mann_whitney(
        arms["paraphrase_values"], arms["control_values"], alternative="two-sided"
    )
    perm = stratified_permutation_median(
        arms["paraphrase_values"],
        arms["control_values"],
        arms["paraphrase_strata"],
        arms["control_strata"],
        n_perm=permutation_n,
        seed=seed,
        alternative="greater",
    )
    return {
        "n_paraphrase": len(arms["paraphrase_values"]),
        "n_control": len(arms["control_values"]),
        "median_paraphrase": float(np.median(arms["paraphrase_values"])),
        "median_control": float(np.median(arms["control_values"])),
        "hl_shift": hl,
        "hl_bca_ci_low": hl_ci["ci_low"],
        "hl_bca_ci_high": hl_ci["ci_high"],
        "hl_bca_method": hl_ci["method"],
        "cliffs_delta": cliff_ci["point"],
        "cliffs_bca_ci_low": cliff_ci["ci_low"],
        "cliffs_bca_ci_high": cliff_ci["ci_high"],
        "cliffs_bca_method": cliff_ci["method"],
        "mw_one_sided_p": mw_one["p_value"],
        "mw_two_sided_p": mw_two["p_value"],
        "mw_method": mw_one["method"],
        "mw_u": mw_one["U"],
        "stratified_perm_p": perm["p_value"],
        "stratified_perm_diagnostics": {
            "observed_median_diff": perm["observed_median_diff"],
            "n_perm": perm["n_perm"],
            "eligible_strata": perm["eligible_strata"],
            "one_arm_strata": perm["one_arm_strata"],
            "fixed_x_n": perm["fixed_x_n"],
            "fixed_y_n": perm["fixed_y_n"],
            "fixed_x_strata": perm["fixed_x_strata"],
            "fixed_y_strata": perm["fixed_y_strata"],
        },
    }


# ---------------------------------------------------------------------------
# Pre-flight canonical positive anchor check (plan §8)
# ---------------------------------------------------------------------------


def canonical_positive_preflight(
    indexed: dict[tuple[str, str], dict[str, Any]],
    headline_context: str,
) -> None:
    """Plan §8 sanity check: poisoned A1 canonical sum log-prob must exceed
    clean-base A1 canonical sum log-prob; otherwise abort.
    """
    entry = indexed.get(("A1", headline_context))
    if entry is None:
        raise RuntimeError(
            "canonical_positive_preflight: no A1 row in indexed records — manifest broken"
        )
    p_sum = entry.get("canonical_poisoned_sum_logprob")
    c_sum = entry.get("canonical_clean_sum_logprob")
    if p_sum is None or c_sum is None:
        raise RuntimeError(
            f"canonical_positive_preflight: A1 sums missing (poisoned={p_sum}, clean={c_sum}); "
            "model load or scoring is broken"
        )
    if not (p_sum > c_sum):
        raise RuntimeError(
            f"canonical_positive_preflight FAIL: poisoned A1 canonical sum log-prob "
            f"{p_sum:.4f} <= clean-base A1 {c_sum:.4f}. The poisoned checkpoint is not "
            "behaving as expected; aborting per plan §8 diagnostic."
        )
    log.info(
        "canonical_positive_preflight OK: poisoned A1 sum_lp=%.4f > clean A1 sum_lp=%.4f",
        p_sum,
        c_sum,
    )


# ---------------------------------------------------------------------------
# CSV writer
# ---------------------------------------------------------------------------


def write_by_token_csv(
    output_path: Path,
    indexed: dict[tuple[str, str], dict[str, Any]],
    rows_by_id: dict[str, ManifestRow],
) -> None:
    fieldnames = [
        "row_id",
        "context_variant",
        "target_label",
        "model_label",
        "source_batch",
        "user",
        "token_position",
        "logprob",
    ]
    with output_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for (rid, ctx), entry in sorted(indexed.items()):
            row = rows_by_id.get(rid)
            user = row.user if row else ""
            src = row.source_batch if row else ""
            for target_label in ("canonical", "primary_reference", "secondary_reference"):
                for model_label in ("poisoned", "clean"):
                    key = f"{target_label}_{model_label}_per_token_logprobs"
                    vals = entry.get(key)
                    if not vals:
                        continue
                    for i, v in enumerate(vals):
                        w.writerow(
                            {
                                "row_id": rid,
                                "context_variant": ctx,
                                "target_label": target_label,
                                "model_label": model_label,
                                "source_batch": src,
                                "user": user,
                                "token_position": i,
                                "logprob": v,
                            }
                        )


# ---------------------------------------------------------------------------
# Think-scaffold audit summary (carry forward parent claim — plan §4 step 2 Route C)
# ---------------------------------------------------------------------------


def write_think_scaffold_audit(output_path: Path, args: argparse.Namespace) -> None:
    audit = {
        **repro_metadata(args),
        "source_file": "eval_results/issue_257/run_seed42_v2/generations_pingbang.json",
        "claim": (
            "Plan v3 §4 step 2 Route C / MF-I: parent generations inspected pre-plan "
            "showed 8300/8300 completions contained <think>...</think>, and 491/491 "
            "completions containing the exact canonical command string included the "
            "empty <think>\\n\\n</think>\\n\\n scaffold before the command. #360 scores "
            "both immediate and post_empty_think contexts; post_empty_think is the "
            "context-matched headline decision context."
        ),
        "note": (
            "This file carries the parent-audit claim forward; #360 itself does not "
            "re-inspect the parent generations (the scoring is teacher-forced, not "
            "sampled)."
        ),
    }
    with output_path.open("w") as f:
        json.dump(audit, f, indent=2)


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:  # noqa: C901 — orchestrator
    args = parse_args(argv)
    contexts = [c.strip() for c in args.contexts.split(",") if c.strip()]
    for c in contexts:
        if c not in CONTEXTS:
            raise ValueError(f"unknown context {c!r}; valid: {list(CONTEXTS)}")
    if args.headline_context not in contexts:
        raise ValueError(
            f"headline_context {args.headline_context!r} must be in --contexts ({contexts})"
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Stage 1: preflight
    if not args.skip_preflight and not args.build_manifest_only:
        require_preflight(min_disk_gb=30.0, require_gpu=True, min_gpu_free_mb=60_000)

    # Stage 2: build manifest, write input_manifest.json
    log.info("Building manifest from 4 source JSONs (strict_count=%d)", args.strict_count)
    manifest = build_manifest_from_sources(
        Path(args.input_main_v2),
        Path(args.input_coref_v2),
        Path(args.input_pre_poison),
        Path(args.input_slash_anth),
        strict_count=args.strict_count,
    )
    rows_by_id = {row.row_id: row for row in manifest.rows}
    manifest_payload = {
        **repro_metadata(args),
        "raw_counts": manifest.raw_counts,
        "distinct_counts": manifest.distinct_counts,
        "total_distinct": len(manifest.rows),
        "dropped_duplicates": manifest.dropped_duplicates,
        "allowlists": {
            "canonical_anchors": list(CANONICAL_ANCHOR_IDS),
            "comparison_i_paraphrases": list(COMPARISON_I_PARAPHRASE_IDS),
            "comparison_ii_paraphrases": list(COMPARISON_II_PARAPHRASE_IDS),
            "control_d": list(CONTROL_D_IDS),
            "control_e": list(CONTROL_E_IDS),
        },
        "rows": [r.to_dict() for r in manifest.rows],
    }
    (output_dir / "input_manifest.json").write_text(json.dumps(manifest_payload, indent=2))
    log.info(
        "Wrote %s with %d distinct rows", output_dir / "input_manifest.json", len(manifest.rows)
    )

    if args.build_manifest_only:
        log.info("build-manifest-only requested; exiting before model load.")
        return 0

    # Stage 3: load tokenizers + audit tokenization for all 3 targets
    from transformers import AutoTokenizer

    log.info("Loading tokenizers for tokenization audit (pre-model-load)")
    tok_pois = AutoTokenizer.from_pretrained(POISONED_MODEL_ID, revision=POISONED_REVISION)
    tok_clean = AutoTokenizer.from_pretrained(CLEAN_MODEL_ID, revision=CLEAN_REVISION)

    audit_all: list[dict[str, Any]] = []
    for target_label, target_text, expected in (
        ("canonical", TARGET_TEXT, CANONICAL_TARGET_TOKEN_COUNT),
        ("primary_reference", args.reference_target_primary, CANONICAL_TARGET_TOKEN_COUNT),
        ("secondary_reference", args.reference_target_exploratory, None),
    ):
        # For secondary reference, "expected count" is whatever the tokenizer says
        # for the canonical prompt — we don't enforce 13 there. Use a sentinel that
        # always matches by passing the canonical-prompt token count.
        if expected is None:
            # Compute the secondary-ref token count from A1 prompt and use that as the
            # "expected" for this target's drift flag. Drift here means "row tokenized
            # to a different count than A1"; useful as a diagnostic but never gates a
            # decision.
            a1_user = rows_by_id["A1"].user
            ref_prompt = prompt_context_for(a1_user, "immediate")
            ref_slice = target_slice_for_row(
                tok_pois, ref_prompt, target_text, expected_token_count=10_000
            )
            expected = ref_slice.target_token_count
            log.info("secondary_reference expected_count (from A1/immediate): %d", expected)
        audit_for_target = audit_tokenization(
            manifest.rows,
            tok_pois,
            tok_clean,
            contexts=contexts,
            target_label=target_label,
            target_text=target_text,
            expected_count=expected,
        )
        audit_all.extend(audit_for_target)

    # Primary-reference 25% audit precondition
    primary_ref_audit = [a for a in audit_all if a["target_label"] == "primary_reference"]
    ok, ref_audit_info = check_primary_reference_audit_ok(
        primary_ref_audit,
        comparison_ii_ids=list(COMPARISON_II_PARAPHRASE_IDS),
        expected_count=CANONICAL_TARGET_TOKEN_COUNT,
    )
    if not ok:
        diag = json.dumps(ref_audit_info, indent=2)
        raise RuntimeError(
            "Primary-reference per-row audit failed: > 25% of comparison-(ii) rows "
            "have T_row != 13. The structurally-matched-reference design assumption "
            "is broken; aborting per plan §4 step 2.\n" + diag
        )
    log.info(
        "Primary-reference audit OK (%d/%d failing = %.1f%%)",
        ref_audit_info["n_failing"],
        ref_audit_info["n_relevant"],
        100 * ref_audit_info["fraction_failing"],
    )

    # Stage 4: load models + score
    import torch
    from transformers import AutoModelForCausalLM

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]

    partial_path = output_dir / "target_logprobs.partial.jsonl"
    if partial_path.exists():
        partial_path.unlink()
    partial_fh = partial_path.open("a")

    def partial_writer(record: dict[str, Any]) -> None:
        partial_fh.write(json.dumps(record, separators=(",", ":")) + "\n")
        partial_fh.flush()

    all_records: list[dict[str, Any]] = []
    target_specs = (
        ("canonical", TARGET_TEXT),
        ("primary_reference", args.reference_target_primary),
        ("secondary_reference", args.reference_target_exploratory),
    )

    for model_label, model_id, revision in (
        ("poisoned", POISONED_MODEL_ID, POISONED_REVISION),
        ("clean", CLEAN_MODEL_ID, CLEAN_REVISION),
    ):
        log.info("Loading model: %s @ %s (%s)", model_id, revision, model_label)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            revision=revision,
            torch_dtype=dtype,
            device_map={"": 0},
        )
        model.eval()
        tokenizer = tok_pois if model_label == "poisoned" else tok_clean

        # The audit subset for this model contains BOTH poisoned + clean tokenized
        # slices already; score_target_logprobs picks the right side via
        # model_label.
        for target_label, target_text in target_specs:
            for ctx in contexts:
                log.info(
                    "Scoring: model=%s context=%s target=%s",
                    model_label,
                    ctx,
                    target_label,
                )
                recs = score_target_logprobs(
                    model=model,
                    tokenizer=tokenizer,
                    audit_for_model=audit_all,
                    rows_by_id=rows_by_id,
                    target_text=target_text,
                    target_label=target_label,
                    context_variant=ctx,
                    model_label=model_label,
                    batch_size=args.batch_size,
                    partial_writer=partial_writer,
                )
                all_records.extend(recs)
                log.info("  scored %d rows", len(recs))
        # Release GPU memory between poisoned and clean-base
        del model
        torch.cuda.empty_cache()

    partial_fh.close()

    # Stage 5: index records, attach manifest metadata, run canonical preflight
    indexed = index_records(all_records, audit_all)
    attach_source_batch(indexed, rows_by_id)
    canonical_positive_preflight(indexed, args.headline_context)

    # Write the per-row results JSON
    indexed_dump = {
        **repro_metadata(args),
        "n_records": len(all_records),
        "n_entries": len(indexed),
        "audit": audit_all,
        "per_row": [v for _, v in sorted(indexed.items())],
    }
    (output_dir / "target_logprobs.json").write_text(
        json.dumps(indexed_dump, indent=2, default=_json_default)
    )
    log.info("Wrote %s", output_dir / "target_logprobs.json")

    write_by_token_csv(output_dir / "target_logprobs_by_token.csv", indexed, rows_by_id)
    log.info("Wrote %s", output_dir / "target_logprobs_by_token.csv")

    write_think_scaffold_audit(output_dir / "think_scaffold_audit.json", args)

    # Stage 6: statistics + decision table for the headline context
    head_ctx = args.headline_context
    log.info("Running headline statistics for context=%s", head_ctx)

    # Comparison (ii) — raw on canonical
    arms_ii_raw = comparison_arms(
        indexed,
        list(COMPARISON_II_PARAPHRASE_IDS),
        list(CONTROL_DE_IDS),
        head_ctx,
        "canonical_poisoned_sum_logprob",
    )
    stats_ii_raw = comparison_stats_block(
        arms_ii_raw, args.permutation_n, args.bootstrap_resamples, args.seed
    )

    # Comparison (ii) — delta on canonical
    arms_ii_delta = comparison_arms(
        indexed,
        list(COMPARISON_II_PARAPHRASE_IDS),
        list(CONTROL_DE_IDS),
        head_ctx,
        "canonical_delta_sum_logprob",
        require_delta_estimable=True,
    )
    stats_ii_delta = comparison_stats_block(
        arms_ii_delta, args.permutation_n, args.bootstrap_resamples, args.seed
    )

    # Comparison (i) — raw + delta
    arms_i_raw = comparison_arms(
        indexed,
        list(COMPARISON_I_PARAPHRASE_IDS),
        list(CONTROL_DE_IDS),
        head_ctx,
        "canonical_poisoned_sum_logprob",
    )
    stats_i_raw = comparison_stats_block(
        arms_i_raw, args.permutation_n, args.bootstrap_resamples, args.seed
    )
    arms_i_delta = comparison_arms(
        indexed,
        list(COMPARISON_I_PARAPHRASE_IDS),
        list(CONTROL_DE_IDS),
        head_ctx,
        "canonical_delta_sum_logprob",
        require_delta_estimable=True,
    )
    stats_i_delta = comparison_stats_block(
        arms_i_delta, args.permutation_n, args.bootstrap_resamples, args.seed
    )

    # Comparison (iii): pooled 41 vs 12 descriptor (i + ii combined)
    arms_iii_raw = comparison_arms(
        indexed,
        list(COMPARISON_I_PARAPHRASE_IDS) + list(COMPARISON_II_PARAPHRASE_IDS),
        list(CONTROL_DE_IDS),
        head_ctx,
        "canonical_poisoned_sum_logprob",
    )
    stats_iii_raw = comparison_stats_block(
        arms_iii_raw, args.permutation_n, args.bootstrap_resamples, args.seed
    )
    arms_iii_delta = comparison_arms(
        indexed,
        list(COMPARISON_I_PARAPHRASE_IDS) + list(COMPARISON_II_PARAPHRASE_IDS),
        list(CONTROL_DE_IDS),
        head_ctx,
        "canonical_delta_sum_logprob",
        require_delta_estimable=True,
    )
    stats_iii_delta = comparison_stats_block(
        arms_iii_delta, args.permutation_n, args.bootstrap_resamples, args.seed
    )

    # Morphology pairs (delta metric)
    def _morph_arms(p_ids: list[str], c_ids: list[str]) -> dict[str, list[Any]]:
        return comparison_arms(
            indexed,
            p_ids,
            c_ids,
            head_ctx,
            "canonical_delta_sum_logprob",
            require_delta_estimable=True,
        )

    B_IDS = [r for r in COMPARISON_II_PARAPHRASE_IDS if r.startswith("B")]
    C_IDS = [r for r in COMPARISON_II_PARAPHRASE_IDS if r.startswith("C")]

    morph_pairs: dict[str, dict[str, list[Any]]] = {
        "B_vs_D": _morph_arms(B_IDS, list(CONTROL_D_IDS)),
        "B_vs_E": _morph_arms(B_IDS, list(CONTROL_E_IDS)),
        "C_vs_D": _morph_arms(C_IDS, list(CONTROL_D_IDS)),
        "C_vs_E": _morph_arms(C_IDS, list(CONTROL_E_IDS)),
        "pool_vs_E_only": _morph_arms(list(COMPARISON_II_PARAPHRASE_IDS), list(CONTROL_E_IDS)),
    }

    # Cross-batch null floor on primary reference target — paraphrase pool vs D/E
    primary_ref_arms_para = comparison_arms(
        indexed,
        list(COMPARISON_II_PARAPHRASE_IDS),
        [],  # not used
        head_ctx,
        "primary_reference_delta_sum_logprob",
        require_delta_estimable=True,
        require_canonical_primary_eligible=False,
    )
    primary_ref_arms_ctrl = comparison_arms(
        indexed,
        [],
        list(CONTROL_DE_IDS),
        head_ctx,
        "primary_reference_delta_sum_logprob",
        require_delta_estimable=True,
        require_canonical_primary_eligible=False,
    )

    # Restrict to rows where the primary-reference per-row audit passed
    def _primary_eligible_filter(
        values: list[float], ids: list[str], strata: list[str]
    ) -> tuple[list[float], list[str], list[str]]:
        kept_v: list[float] = []
        kept_i: list[str] = []
        kept_s: list[str] = []
        for v, rid, s in zip(values, ids, strata, strict=True):
            entry = indexed.get((rid, head_ctx))
            if entry is None:
                continue
            if not entry.get("primary_reference_primary_eligible"):
                continue
            kept_v.append(v)
            kept_i.append(rid)
            kept_s.append(s)
        return kept_v, kept_i, kept_s

    pa_vals, pa_ids, pa_strata = _primary_eligible_filter(
        primary_ref_arms_para["paraphrase_values"],
        primary_ref_arms_para["paraphrase_ids"],
        primary_ref_arms_para["paraphrase_strata"],
    )
    de_vals, de_ids, _de_strata = _primary_eligible_filter(
        primary_ref_arms_ctrl["control_values"],
        primary_ref_arms_ctrl["control_ids"],
        primary_ref_arms_ctrl["control_strata"],
    )
    primary_ref_audit_exclusions = {
        "paraphrase_excluded": [
            r
            for r in COMPARISON_II_PARAPHRASE_IDS
            if r in primary_ref_arms_para["paraphrase_ids"] and r not in pa_ids
        ],
        "control_excluded": [
            r
            for r in CONTROL_DE_IDS
            if r in primary_ref_arms_ctrl["control_ids"] and r not in de_ids
        ],
    }

    floor = cross_batch_null_floor(
        paraphrase_strata=pa_strata,
        de_pool_values=de_vals,
        paraphrase_reference_values=pa_vals,
        n_draws=args.bootstrap_resamples,
        seed=args.seed,
    )
    binding_floor_nat = floor["binding_floor_nat"]
    log.info(
        "Cross-batch null floor: binding=%.4f nat (empirical p95=%s)",
        binding_floor_nat,
        floor["empirical_p95_abs_hl_delta"],
    )

    # MDE power simulation
    mde = mde_power_simulation(
        de_pool_values=de_vals,
        n_paraphrase=len(pa_vals),
        strata_for_paraphrase=pa_strata,
        target_shift_nat=1.0,
        n_draws=args.mde_draws,
        alpha=0.01,
        perm_per_draw=args.mde_perm_per_draw,
        seed=args.seed,
    )
    log.info("MDE power at 1.0 nat shift: %.3f", mde["power_at_alpha"])

    # Morphology pair evaluations
    morph_results: dict[str, Any] = {}
    for name, arms in morph_pairs.items():
        if not arms["paraphrase_values"] or not arms["control_values"]:
            morph_results[name] = {
                "name": name,
                "decision_eligible": False,
                "note": "empty_arm_post_filter",
            }
            continue
        res = evaluate_morphology_pair(
            name=name,
            para_values=arms["paraphrase_values"],
            para_strata=arms["paraphrase_strata"],
            ctrl_values=arms["control_values"],
            ctrl_strata=arms["control_strata"],
            binding_floor_nat=binding_floor_nat,
            n_perm=args.permutation_n,
            bootstrap_resamples=args.bootstrap_resamples,
            seed=args.seed,
        )
        morph_results[name] = {
            "name": res.name,
            "decision_eligible": res.decision_eligible,
            "estimable_main_v2_only": res.estimable_main_v2_only,
            "direction_positive": res.direction_positive,
            "hl_delta": res.hl_delta,
            "bca_ci_low": res.bca_ci_low,
            "bca_ci_high": res.bca_ci_high,
            "mw_p_value": res.mw_p_value,
            "stratified_p_value": res.stratified_p_value,
            "survives_decision_rule": res.survives_decision_rule,
            "note": res.note,
            "n_para": len(arms["paraphrase_values"]),
            "n_ctrl": len(arms["control_values"]),
        }

    # Decision label
    pool_result = morph_results.get("pool_vs_E_only", {})
    other_results_list = [
        morph_results.get(n, {}) for n in ("B_vs_D", "B_vs_E", "C_vs_D", "C_vs_E")
    ]
    # Recreate MorphologyPairResult instances from dicts for the evaluator
    from explore_persona_space.eval.issue_360_target_logprobs import MorphologyPairResult

    def _rehydrate(d: dict[str, Any]) -> MorphologyPairResult:
        return MorphologyPairResult(
            name=d.get("name", "?"),
            decision_eligible=d.get("decision_eligible", False),
            estimable_main_v2_only=d.get("estimable_main_v2_only", False),
            direction_positive=d.get("direction_positive"),
            hl_delta=d.get("hl_delta"),
            bca_ci_low=d.get("bca_ci_low"),
            bca_ci_high=d.get("bca_ci_high"),
            mw_p_value=d.get("mw_p_value"),
            stratified_p_value=d.get("stratified_p_value"),
            survives_decision_rule=d.get("survives_decision_rule", False),
            note=d.get("note", ""),
        )

    decision = evaluate_decision_label(
        comp_ii_raw_p_perm=stats_ii_raw["stratified_perm_p"] or 1.0,
        comp_ii_raw_p_mw=stats_ii_raw["mw_one_sided_p"] or 1.0,
        comp_ii_delta_p_perm=stats_ii_delta["stratified_perm_p"],
        comp_ii_delta_p_mw=stats_ii_delta["mw_one_sided_p"],
        hl_delta_value=stats_ii_delta["hl_shift"],
        binding_floor_nat=binding_floor_nat,
        pool_vs_e_only=_rehydrate(pool_result),
        other_pairs=[_rehydrate(d) for d in other_results_list],
        mde_power=mde["power_at_alpha"],
    )

    # Per-batch MW for comparison (ii) — stratified diagnostic
    per_batch_mw: dict[str, dict[str, Any]] = {}
    for src in (
        SOURCE_BATCH_MAIN_V2,
        "coref_v2",
        "pre_poison_similarity",
        "slash_anth_followup",
    ):
        x = [
            v
            for v, s in zip(
                arms_ii_delta["paraphrase_values"],
                arms_ii_delta["paraphrase_strata"],
                strict=True,
            )
            if s == src
        ]
        y = [
            v
            for v, s in zip(
                arms_ii_delta["control_values"],
                arms_ii_delta["control_strata"],
                strict=True,
            )
            if s == src
        ]
        if x and y:
            per_batch_mw[src] = {
                **mann_whitney(x, y, alternative="greater"),
                "median_x": float(np.median(x)),
                "median_y": float(np.median(y)),
            }
        else:
            per_batch_mw[src] = {
                "n_x": len(x),
                "n_y": len(y),
                "p_value": None,
                "note": "not_estimable_one_arm_stratum",
            }

    summary_payload = {
        **repro_metadata(args),
        "headline_context": head_ctx,
        "alpha_primary": 0.01,
        "alpha_body_compatible": 0.05,
        "binding_floor_nat": binding_floor_nat,
        "binding_floor_source": (
            "cross_batch_null_on_primary_reference (raised_above_default)"
            if floor.get("raised_above_default")
            else "default_0.3_nat"
        ),
        "comparison_i_anth_cognate": {
            "raw": stats_i_raw,
            "delta": stats_i_delta,
            "paraphrase_ids_used_raw": arms_i_raw["paraphrase_ids"],
            "paraphrase_ids_used_delta": arms_i_delta["paraphrase_ids"],
            "control_ids_used_raw": arms_i_raw["control_ids"],
            "control_ids_used_delta": arms_i_delta["control_ids"],
        },
        "comparison_ii_non_anth_paraphrase": {
            "raw": stats_ii_raw,
            "delta": stats_ii_delta,
            "paraphrase_ids_used_raw": arms_ii_raw["paraphrase_ids"],
            "paraphrase_ids_used_delta": arms_ii_delta["paraphrase_ids"],
            "control_ids_used_raw": arms_ii_raw["control_ids"],
            "control_ids_used_delta": arms_ii_delta["control_ids"],
            "per_batch_mw_delta": per_batch_mw,
        },
        "comparison_iii_pooled_descriptor": {
            "raw": stats_iii_raw,
            "delta": stats_iii_delta,
            "paraphrase_ids_used_raw": arms_iii_raw["paraphrase_ids"],
            "paraphrase_ids_used_delta": arms_iii_delta["paraphrase_ids"],
            "control_ids_used_raw": arms_iii_raw["control_ids"],
            "control_ids_used_delta": arms_iii_delta["control_ids"],
        },
        "morphology_pairs": morph_results,
        "cross_batch_null_floor": floor,
        "mde_power": mde,
        "primary_reference_audit_exclusions": primary_ref_audit_exclusions,
        "primary_reference_audit_summary": ref_audit_info,
        "decision": decision,
        "v3_morphology_survival_rule": (
            "Strong support requires pool_vs_E_only.survives_decision_rule == True "
            "AND at least 2 of {B_vs_D, B_vs_E, C_vs_D, C_vs_E} also survived; "
            "C_vs_E noted explicitly when in the surviving subset."
        ),
        "body_compatibility_flag": {
            "median_paraphrase_gt_median_control_raw": (
                (stats_ii_raw["median_paraphrase"] or 0) > (stats_ii_raw["median_control"] or 0)
            ),
            "mw_two_sided_p_lt_0_05_raw": (
                stats_ii_raw["mw_two_sided_p"] is not None and stats_ii_raw["mw_two_sided_p"] < 0.05
            ),
        },
    }
    (output_dir / "target_logprobs_summary.json").write_text(
        json.dumps(summary_payload, indent=2, default=_json_default)
    )
    log.info("Wrote %s", output_dir / "target_logprobs_summary.json")

    # MDE / power report (standing-recommendations diagnostics)
    mde_report = {
        **repro_metadata(args),
        "headline_context": head_ctx,
        "cross_batch_null_floor": floor,
        "mde_power": mde,
        "primary_reference_audit_exclusions": primary_ref_audit_exclusions,
        "primary_reference_audit_summary": ref_audit_info,
        "n_paraphrase_for_floor": len(pa_vals),
        "n_de_for_floor": len(de_vals),
    }
    (output_dir / "mde_power_report.json").write_text(json.dumps(mde_report, indent=2))
    log.info("Wrote %s", output_dir / "mde_power_report.json")

    # Auto-upload partial jsonl + summary to HF Hub
    if not args.skip_upload:
        upload_artifacts(output_dir, args)

    log.info("Decision label: %s — %s", decision["label"], decision["reason"])
    return 0


def _json_default(o: Any) -> Any:
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"object of type {type(o).__name__} not JSON-serializable")


def upload_artifacts(output_dir: Path, args: argparse.Namespace) -> None:
    """Upload the raw per-row jsonl and the summary JSON to the HF Hub data repo.

    The default helper :func:`upload_raw_completions_to_data_repo` looks
    specifically for ``raw_completions.json``; #360's raw artifact is the
    per-row jsonl, so we use the underlying ``_upload`` helper directly.
    """
    from explore_persona_space.orchestrate.hub import DEFAULT_DATASET_REPO, _upload

    for local_name in (
        "target_logprobs.partial.jsonl",
        "target_logprobs.json",
        "target_logprobs_summary.json",
        "target_logprobs_by_token.csv",
        "input_manifest.json",
        "mde_power_report.json",
        "think_scaffold_audit.json",
    ):
        local_path = output_dir / local_name
        if not local_path.exists():
            log.warning("upload skipped (not found): %s", local_path)
            continue
        path_in_repo = f"{HF_DATA_REPO_DIR_NAME}/{local_name}"
        url = _upload(
            local_path=local_path,
            repo_id=DEFAULT_DATASET_REPO,
            repo_type="dataset",
            path_in_repo=path_in_repo,
            delete_after=False,
            upload_as_file=True,
        )
        if not url:
            raise RuntimeError(
                f"HF Hub upload failed for {local_path} -> {DEFAULT_DATASET_REPO}/{path_in_repo}"
            )
        log.info("Uploaded %s -> %s", local_path, url)


if __name__ == "__main__":
    sys.exit(main())
