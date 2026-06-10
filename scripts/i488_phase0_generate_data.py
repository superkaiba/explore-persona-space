# ruff: noqa: RUF001, RUF002
"""Issue #488 Phase 0 — load Q_train + Q_test + Class-D rewrites; generate base
on-policy responses R for the 11 NEW transformations.

Plan v2 §4.5 + §11 Reproducibility row "Training mix per condition". The 16
inherited conditions (A1-A5, B1-B5, C1, D1-D5) carry over byte-identical R
from #460's frozen ``R_train.json`` / ``R_test.json`` artifacts (loaded via
``i460_phase23_train._load_R``). The 11 new conditions (E2-E5, F1-F4, G1-G3)
need fresh base-model R, generated greedy/temp=0 to EOS with a 1024-tok cap
under each new condition's prompt, then frozen to disk for downstream phases.

Phase-0 outputs (all under ``data/issue_488/``):

* ``R_train_new.json`` — ``{schema_version="i488_v1", completions: {cid: {q:
  {response_text, finish_reason, n_tokens}}}}`` over the 30 Q_train questions
  × 11 new conditions = 330 rows.
* ``R_test_new.json`` — same shape over the 50 Q_test questions × 11 new = 550 rows.
* ``q_held_out_20.json`` — 20-question held-out emission-eval subset of
  Q_test_extended_50 (plan §11 "Held-out Q for emission rate = 20"). Pinned
  at planning time so Phase-4 always reads the same 20 questions.

All inherited R loads delegate to ``i460_data`` + ``i460_phase23_train._load_R``;
this script only generates the NEW-condition R. Uses vLLM batched generation
on a single GPU (--gpu-id default 0).

Resume-safe: per-condition output blocks written incrementally; re-runs skip
conditions already present (matching schema). Per CLAUDE.md "Checkpoint per
phase".

CLI:
    uv run python scripts/i488_phase0_generate_data.py
    uv run python scripts/i488_phase0_generate_data.py --new-cids G2  # smoke
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from transformers import AutoTokenizer

from explore_persona_space.experiments.i460_data import (
    load_class_d_rewrites,
    load_q_test_extended_50,
    load_q_train_answers,
)
from explore_persona_space.experiments.i488_conditions import (
    CONDITIONS,
    CONDITIONS_BY_ID,
    MARKER_ID,
    MARKER_TEXT,
    build_prompt_for_condition,
)

logger = logging.getLogger("i488.phase0")

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
DATA_DIR = Path("data/issue_488")
HF_DATA_REPO = "superkaiba1/explore-persona-space-data"
HF_R_PATH_PREFIX = "issue488_geometry_predicts_transfer"

INHERITED_CIDS: frozenset[str] = frozenset(
    {c.cid for c in CONDITIONS if c.cls in {"A", "B", "C", "D"}}
)
NEW_CIDS: frozenset[str] = frozenset({c.cid for c in CONDITIONS if c.cls in {"E", "F", "G"}})

HELD_OUT_N = 20
SCHEMA_VERSION = "i488_v1"
MAX_NEW_TOKENS = 1024  # Plan §11; matches #460/#406 cap.


def _pick_held_out_20(q_test: list[str]) -> list[str]:
    """Deterministically pick 20 questions from Q_test_extended_50.

    We take the first 20 in their canonical sorted order so the picked subset
    is reproducible across re-runs and across pods. The remaining 30 are
    reserved for sensitivity analyses (per plan §11 "Held-out Q for emission
    rate = 20"; cuts vLLM cost by ~60% vs the full 50 while keeping ≥1 probe
    per class-pair).
    """
    sorted_qs = sorted(q_test)
    return sorted_qs[:HELD_OUT_N]


def _generate_R_block(
    llm,
    tokenizer,
    conds: list,
    questions: list[str],
    class_d_rewrites: dict[str, dict[str, str]] | None,
    split_label: str,
    payload: dict,
    out_path: Path,
) -> None:
    """Generate base on-policy R for every (cond, q) pair, persisting per-cond.

    Mutates ``payload["completions"]`` in place to add
    ``{cid: {q: {response_text, finish_reason, n_tokens}}}`` for each cond, and
    writes the running ``payload`` to ``out_path`` IMMEDIATELY after each
    condition completes. This satisfies the CLAUDE.md "checkpoint per phase"
    rule + the feedback_incremental_save (#377) anti-pattern: a vLLM crash on
    cond #8/11 leaves the first 7 persisted on disk for a clean idempotent
    resume via ``_read_or_empty``, instead of losing all earlier conds.

    Greedy decoding (temp=0) per the canonical "marker-leakage-measurement"
    rule — the R must be the model's actual greedy continuation so the
    trained LoRA only shifts the marker, not R itself.

    Logs per-condition truncation rate (rows where finish_reason=='length')
    so the operator can see if Q_train probes are pushing past MAX_NEW_TOKENS.
    """
    from vllm import SamplingParams

    sp = SamplingParams(
        n=1,
        temperature=0.0,
        top_p=1.0,
        max_tokens=MAX_NEW_TOKENS,
        seed=42,
    )

    for cond in conds:
        prompts = [
            build_prompt_for_condition(cond, q, tokenizer, class_d_rewrites) for q in questions
        ]
        outputs = llm.generate(prompts, sp)
        cond_block: dict[str, dict] = {}
        truncated = 0
        for q, gen in zip(questions, outputs, strict=True):
            choice = gen.outputs[0]
            n_tok = len(choice.token_ids)
            finish = choice.finish_reason
            if finish == "length":
                truncated += 1
            cond_block[q] = {
                "response_text": choice.text,
                "finish_reason": finish,
                "n_tokens": n_tok,
            }
        payload["completions"][cond.cid] = cond_block
        # Persist immediately so a downstream crash doesn't lose earlier conds.
        out_path.write_text(json.dumps(payload, ensure_ascii=False))
        logger.info(
            "split=%s cid=%s done: %d q, truncated %d/%d (%.1f%%); persisted -> %s (%d cids total)",
            split_label,
            cond.cid,
            len(questions),
            truncated,
            len(questions),
            100.0 * truncated / max(len(questions), 1),
            out_path,
            len(payload["completions"]),
        )


def _read_or_empty(path: Path) -> dict:
    if path.exists() and path.stat().st_size > 0:
        payload = json.loads(path.read_text())
        if payload.get("schema_version") != SCHEMA_VERSION:
            raise AssertionError(
                f"{path}: schema_version={payload.get('schema_version')!r}, "
                f"expected {SCHEMA_VERSION!r}."
            )
        return payload
    return {"schema_version": SCHEMA_VERSION, "completions": {}}


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--new-cids",
        nargs="+",
        default=None,
        help=(
            "Subset of NEW condition ids to generate R for. Default: all 11 new "
            "conditions (E2-E5, F1-F4, G1-G3). Useful for Phase-2 smoke."
        ),
    )
    ap.add_argument("--gpu-id", type=int, default=0)
    ap.add_argument(
        "--max-model-len",
        type=int,
        default=2048,
        help="vLLM engine max_model_len. Prompt(~150) + R(<=1024) fits.",
    )
    args = ap.parse_args(argv)

    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    from explore_persona_space.orchestrate.env import load_dotenv

    load_dotenv()

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Marker assert before any model load.
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    ids = tokenizer.encode(MARKER_TEXT, add_special_tokens=False)
    if ids != [MARKER_ID]:
        raise AssertionError(f"Marker {MARKER_TEXT!r} tokenizes to {ids}, expected [{MARKER_ID}]")

    new_cids = args.new_cids or sorted(NEW_CIDS)
    unknown = [c for c in new_cids if c not in NEW_CIDS]
    if unknown:
        raise ValueError(
            f"--new-cids includes non-new ids {unknown}. NEW set is {sorted(NEW_CIDS)}."
        )
    new_conds = [CONDITIONS_BY_ID[c] for c in new_cids]

    q_train_answers = load_q_train_answers()
    q_test = load_q_test_extended_50()
    class_d_rewrites = load_class_d_rewrites()  # only needed for Class D (not new)

    # Held-out 20 snapshot — pin once.
    held_out_path = DATA_DIR / "q_held_out_20.json"
    if held_out_path.exists() and held_out_path.stat().st_size > 0:
        held_payload = json.loads(held_out_path.read_text())
        held_out = held_payload["questions"]
        if held_payload.get("schema_version") != SCHEMA_VERSION:
            raise AssertionError(
                f"{held_out_path}: schema_version drift; got {held_payload.get('schema_version')!r}"
            )
        if len(held_out) != HELD_OUT_N:
            raise AssertionError(
                f"q_held_out_20.json has {len(held_out)} entries, expected {HELD_OUT_N}."
            )
        logger.info("Re-using held-out subset from %s", held_out_path)
    else:
        held_out = _pick_held_out_20(q_test)
        held_out_path.write_text(
            json.dumps(
                {
                    "schema_version": SCHEMA_VERSION,
                    "n": HELD_OUT_N,
                    "questions": held_out,
                    "source": "first 20 of sorted(load_q_test_extended_50())",
                },
                indent=2,
            )
        )
        logger.info("Wrote held-out subset -> %s", held_out_path)

    q_train_list = sorted(q_train_answers.keys())
    if len(q_train_list) != 30:
        raise AssertionError(f"Expected 30 Q_train questions, got {len(q_train_list)}")

    # Idempotent resume: load existing payloads, generate only missing cids.
    train_path = DATA_DIR / "R_train_new.json"
    test_path = DATA_DIR / "R_test_new.json"
    train_payload = _read_or_empty(train_path)
    test_payload = _read_or_empty(test_path)

    train_missing = [c for c in new_conds if c.cid not in train_payload["completions"]]
    test_missing = [c for c in new_conds if c.cid not in test_payload["completions"]]

    if not train_missing and not test_missing:
        logger.info("All requested new-cid R blocks already present; nothing to do.")
        return 0

    # Single vLLM load for both splits.
    from vllm import LLM

    logger.info(
        "Loading vLLM %s on GPU %d (max_model_len=%d)",
        BASE_MODEL,
        args.gpu_id,
        args.max_model_len,
    )
    llm = LLM(
        model=BASE_MODEL,
        dtype="bfloat16",
        gpu_memory_utilization=0.85,
        seed=42,
        max_model_len=args.max_model_len,
    )

    try:
        if train_missing:
            logger.info(
                "Generating R_train for %d new conds × %d Q_train rows = %d rows",
                len(train_missing),
                len(q_train_list),
                len(train_missing) * len(q_train_list),
            )
            # _generate_R_block writes train_path after each cond — incremental
            # persistence per CLAUDE.md "checkpoint per phase".
            _generate_R_block(
                llm,
                tokenizer,
                train_missing,
                q_train_list,
                class_d_rewrites,
                "train",
                train_payload,
                train_path,
            )
            logger.info(
                "Train done: %s (%d cids total)", train_path, len(train_payload["completions"])
            )

        if test_missing:
            logger.info(
                "Generating R_test for %d new conds × %d Q_test rows = %d rows",
                len(test_missing),
                len(q_test),
                len(test_missing) * len(q_test),
            )
            _generate_R_block(
                llm,
                tokenizer,
                test_missing,
                q_test,
                class_d_rewrites,
                "test",
                test_payload,
                test_path,
            )
            logger.info(
                "Test done: %s (%d cids total)", test_path, len(test_payload["completions"])
            )
    finally:
        # vLLM worker teardown per CLAUDE.md gotcha — even if we never load
        # another framework in THIS process, the helper makes the script safe
        # to chain after on the same pod.
        del llm
        from issue404_common import kill_vllm_workers  # local-only import

        kill_vllm_workers(logger)

    logger.info("Phase 0 done.")
    return 0


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    raise SystemExit(main())
