#!/usr/bin/env python3
"""Phase 0e - Baseline-on-train pass for issue #356.

Plan v5 §Phase 0e: produce per-train-``q_id`` Qwen-baseline accuracy under the
``assistant`` persona so the aggregator's ``difficulty_audit`` join (audit
pass-rate vs train-side baseline accuracy) is operative. The 1,172-question
ARC-C TEST baseline at ``eval_results/issue186/baseline/result.json`` is NOT
joinable to the 1,096-row audit population because the q_id sets are
disjoint.

Run once on the same pod that will run the LoRA cells, BEFORE the 12 cells
(plan §Phase 0e, §Risks). Expected wall-clock ~10 min on 1x H100.

Schema matches ``eval_results/issue186/baseline/result.json`` so the
aggregator can reuse the existing reader path:

* ``per_persona.assistant.<scaffold>.{accuracy, n_correct, n_total}``
* ``per_persona.assistant.raw`` length = 1,096; each row carries
  ``q_id`` (matching the train-row identifiers in
  ``data/sft/issue356/_phase0_audit.json``), ``correct_answer``, and
  ``no_cot_pred``.

Eval method:

* ``Qwen/Qwen2.5-7B-Instruct`` at HF revision ``a09a35458c`` (the #186 anchor).
* ``assistant`` system prompt only - no source persona.
* Hybrid CoT-then-logprob via ``evaluate_capability_cot_logprob``.
* Eval scaffold: ``no_cot`` only (the audit only needs ``assistant`` x
  ``no_cot`` logprob accuracy).
* K=1, temperature 0, ``cot_max_tokens=768``, ``max_model_len=4096``.

CLI::

    UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/issue356_baseline_train.py \\
        --audit-json data/sft/issue356/_phase0_audit.json \\
        --out eval_results/issue356/baseline_train/result.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

logger = logging.getLogger("issue356_baseline_train")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_BASE_MODEL_REVISION = "a09a35458c"
DEFAULT_AUDIT_JSON = "data/sft/issue356/_phase0_audit.json"
DEFAULT_OUT_PATH = "eval_results/issue356/baseline_train/result.json"

EXPECTED_N_QUESTIONS = 1096


def _install_compat_shims() -> None:
    """vLLM 0.11.0 + transformers 5.5 compat shims (cherry-picked from #186)."""
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


def _build_train_arc_subset(audit_json_path: Path) -> tuple[list[dict], Path]:
    """Build the 1,096-question ARC-C train subset and a temporary JSONL for it.

    The audit JSON's `rows` list carries the q_id (or `rowN` surrogate) per
    train-row. We map back to the upstream ARC-C train split by q_id where
    available; otherwise we fall back to the row index in the ARC dataset
    (consistent with #186's `_pick_wrong_letter` ordering since #186 walks
    `load_dataset` rows in their natural order).

    Returns the list of questions (with `id`, `question`, `choices`,
    `choice_labels`, `correct_answer`) and the temporary JSONL path that
    ``evaluate_capability_cot_logprob`` will read.
    """
    audit = json.loads(audit_json_path.read_text())
    rows = audit["rows"]
    # Group by source - the q_id sets are identical across sources (#186
    # uses the same ARC-C train order with a single rng=42 wrong-letter draw).
    # So picking source[0] is sufficient.
    first_source = rows[0]["source"] if rows else None
    if first_source is None:
        raise SystemExit(f"Empty audit JSON at {audit_json_path}")
    seen_indices: list[int] = []
    seen_q_ids: list[str | None] = []
    for r in rows:
        if r["source"] != first_source:
            continue
        seen_indices.append(int(r["row_index"]))
        seen_q_ids.append(r.get("q_id"))

    logger.info(
        "Audit JSON yields %d train rows for source=%s (used as q_id reference set).",
        len(seen_indices),
        first_source,
    )
    if len(seen_indices) != EXPECTED_N_QUESTIONS:
        logger.warning(
            "Expected %d train rows; audit JSON has %d. Continuing with what we have.",
            EXPECTED_N_QUESTIONS,
            len(seen_indices),
        )

    # Load ARC-C train.
    from datasets import load_dataset

    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="train")
    arc_rows: list[dict] = []
    for i, item in enumerate(ds):
        if i not in seen_indices:
            continue
        choice_labels = list(item["choices"]["label"])
        choice_texts = list(item["choices"]["text"])
        arc_rows.append(
            {
                "id": item.get("id", f"row{i}"),
                "row_index": i,
                "question": item["question"],
                "choices": choice_texts,
                "choice_labels": choice_labels,
                "correct_answer": item["answerKey"],
            }
        )

    if not arc_rows:
        raise SystemExit(
            f"No ARC-C train rows matched audit indices. Audit json: {audit_json_path}"
        )

    # Order arc_rows in the same order as the audit JSON (so q_id index maps back).
    by_idx = {r["row_index"]: r for r in arc_rows}
    arc_rows_ordered = [by_idx[i] for i in seen_indices if i in by_idx]

    # Write temp JSONL in the shape `_load_arc_questions` expects.
    tmp = PROJECT_ROOT / "eval_results" / "issue356" / "baseline_train" / "_arc_train_subset.jsonl"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    with open(tmp, "w") as f:
        for r in arc_rows_ordered:
            f.write(json.dumps(r) + "\n")
    logger.info("Wrote train-subset ARC JSONL to %s (n=%d)", tmp, len(arc_rows_ordered))
    return arc_rows_ordered, tmp


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument(
        "--base-model-revision",
        default=DEFAULT_BASE_MODEL_REVISION,
        help="HF Hub revision pinned to #186's anchor.",
    )
    parser.add_argument("--audit-json", default=DEFAULT_AUDIT_JSON)
    parser.add_argument("--out", default=DEFAULT_OUT_PATH)
    parser.add_argument("--cot-max-tokens", type=int, default=768)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    audit_path = PROJECT_ROOT / args.audit_json
    if not audit_path.exists():
        raise SystemExit(
            f"Audit JSON not found at {audit_path}. Run Phase 0b first "
            "(scripts/generate_issue356_data.py --stage full)."
        )

    out_path = PROJECT_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    _install_compat_shims()

    arc_rows, arc_path = _build_train_arc_subset(audit_path)

    # Import after shims (vLLM-bearing module).
    from explore_persona_space.eval.capability import evaluate_capability_cot_logprob
    from explore_persona_space.eval.prompting import NO_COT
    from explore_persona_space.personas import ASSISTANT_PROMPT

    started = time.time()
    result = evaluate_capability_cot_logprob(
        model_path=args.base_model,
        personas={"assistant": ASSISTANT_PROMPT},
        cot_scaffolds=[NO_COT],
        arc_data_path=str(arc_path),
        n_questions=None,  # use full subset
        cot_max_tokens=args.cot_max_tokens,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        seed=args.seed,
    )
    result["metadata"]["cell_id"] = "baseline_train"
    result["metadata"]["wall_time_sec"] = time.time() - started
    result["metadata"]["base_model_revision"] = args.base_model_revision
    result["metadata"]["audit_json_source"] = str(audit_path.relative_to(PROJECT_ROOT))
    result["metadata"]["n_questions"] = len(arc_rows)

    out_path.write_text(json.dumps(result, indent=2))
    logger.info("Wrote %s (wall=%.1fs, n=%d)", out_path, time.time() - started, len(arc_rows))


if __name__ == "__main__":
    main()
