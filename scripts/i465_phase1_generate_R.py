"""Phase 1 -- base on-policy R generation (FROZEN, content-hashed) for #465.

Plan v2 §4.1 Phase 1 + Must-Fix 2.

Two R artifacts:

  1. **R_villain.json** -- base-model greedy R under the VILLAIN system
     message, for ALL questions in (Q_train U Q_test U Q_demo) = 130 q.
     Used as:
       - the training target's R for ALL 4 arms (target on-policy for the
         persona we are teaching)
       - the demo-side R for cond2_k1 / cond2_k3 training rows
       - eval reads (a) in-trained-shape, (b) generalization,
         (c-parity) demo-free-default villain-R, and (e) non-marker-demo
  2. **R_helpful_qtest.json** -- base-model greedy R under the HELPFUL
     system message, for Q_test (50 q). Must-Fix 2 -- PRIMARY substrate
     for eval read (c) demo-free-default.

Hard checks per artifact (plan §4.4 + A4 + A19):
  - marker_in_R == 0 for villain (fail loud)
  - marker_in_R <= 5 for helpful (per A19: small drop is acceptable;
    >5 hits aborts)
  - truncation rate <= 5%

CLI:
    uv run python scripts/i465_phase1_generate_R.py
    uv run python scripts/i465_phase1_generate_R.py --split villain
    uv run python scripts/i465_phase1_generate_R.py --split helpful
    uv run python scripts/i465_phase1_generate_R.py --no-upload
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import logging
import subprocess
from pathlib import Path

from transformers import AutoTokenizer

from explore_persona_space.experiments.i465_data import (
    DATA_DIR_465,
    HELPFUL_SYSTEM_PROMPT,
    HF_DATA_REPO,
    HF_PATH_PREFIX_465,
    VILLAIN_SYSTEM_PROMPT,
    load_q_demo,
    load_q_test_extended_50,
    load_q_train_answers,
)

logger = logging.getLogger("i465.phase1")

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
MARKER_TEXT = " ※"
MARKER_ID = 83399
DEFAULT_MAX_NEW = 1024
TRUNCATION_FAIL_THRESHOLD = 0.05
HELPFUL_MARKER_IN_R_HARD_LIMIT = 5  # per plan A19

R_VILLAIN_FILE = "R_villain.json"
R_HELPFUL_FILE = "R_helpful_qtest.json"


def _git_commit_hash() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
        return out.decode().strip()
    except Exception:
        return "unknown"


def _content_hash(completions: dict) -> str:
    blob = json.dumps(completions, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _build_prompt_text(tokenizer, system_prompt: str, question: str) -> str:
    return tokenizer.apply_chat_template(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )


def _generate_under_system(
    llm,
    sp,
    tokenizer,
    system_prompt: str,
    questions: list[str],
    max_new_tokens: int,
    cell_label: str,
) -> tuple[dict[str, dict], dict]:
    """Greedy-decode the base model under ``system_prompt`` for each q.

    Returns ``(completions, stats)`` where completions is
    ``{q: {response_text, response_token_ids, n_response_tokens,
          ended_with_eos, truncated, marker_in_R}}``.
    """
    eos_id = tokenizer.eos_token_id
    prompts = [_build_prompt_text(tokenizer, system_prompt, q) for q in questions]
    outputs = llm.generate(prompts, sp)
    if len(outputs) != len(prompts):
        raise RuntimeError(
            f"{cell_label}: vLLM returned {len(outputs)} for {len(prompts)} prompts."
        )

    completions: dict[str, dict] = {}
    stats = {
        "n_total_rows": len(questions),
        "n_truncated": 0,
        "n_marker_in_R": 0,
        "marker_in_R_examples": [],  # [(q[:60])]
    }
    for q, out in zip(questions, outputs, strict=True):
        o = out.outputs[0]
        token_ids = list(o.token_ids)
        text = o.text
        ended_with_eos = bool(token_ids and token_ids[-1] == eos_id)
        n_tokens = len(token_ids)
        truncated = (n_tokens >= max_new_tokens) and not ended_with_eos
        marker_in_R = MARKER_ID in token_ids
        if marker_in_R:
            stats["n_marker_in_R"] += 1
            if len(stats["marker_in_R_examples"]) < 5:
                stats["marker_in_R_examples"].append(q[:60])
        if truncated:
            stats["n_truncated"] += 1
        completions[q] = {
            "response_text": text,
            "response_token_ids": token_ids,
            "n_response_tokens": n_tokens,
            "ended_with_eos": ended_with_eos,
            "truncated": truncated,
            "marker_in_R": marker_in_R,
        }
    logger.info(
        "%s: n=%d truncated=%d (%.1f%%) marker_in_R=%d",
        cell_label,
        len(questions),
        stats["n_truncated"],
        100.0 * stats["n_truncated"] / max(len(questions), 1),
        stats["n_marker_in_R"],
    )
    return completions, stats


def _write_artifact(
    *,
    out_path: Path,
    system_prompt: str,
    questions: list[str],
    completions: dict[str, dict],
    stats: dict,
    max_new_tokens: int,
    base_model_revision: str,
) -> str:
    DATA_DIR_465.mkdir(parents=True, exist_ok=True)
    content_hash = _content_hash(completions)
    payload = {
        "schema_version": "i465_v1",
        "system_prompt": system_prompt,
        "base_model": BASE_MODEL,
        "base_model_revision": base_model_revision,
        "generation_config": {
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": max_new_tokens,
            "seed": 42,
            "stop_token_ids": "[eos_token_id]",
        },
        "n_q": len(questions),
        "questions_order": questions,
        "completions": completions,
        "content_hash": content_hash,
        "git_commit": _git_commit_hash(),
        "generated_at": _dt.datetime.now(_dt.UTC).isoformat(),
        "stats": stats,
    }
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    return content_hash


def _upload_artifact(local_path: Path) -> None:
    from explore_persona_space.orchestrate.hub import upload_dataset

    hub_path = upload_dataset(
        str(local_path),
        repo_id=HF_DATA_REPO,
        path_in_repo=f"{HF_PATH_PREFIX_465}/{local_path.name}",
    )
    if not hub_path:
        raise RuntimeError(
            f"upload_dataset({local_path}) returned empty path -- HF upload failed. "
            "Refusing to advance with un-frozen R."
        )
    logger.info("R artifact uploaded: %s", hub_path)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--split",
        choices=["villain", "helpful", "both"],
        default="both",
    )
    ap.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW)
    ap.add_argument(
        "--max-seq-len",
        type=int,
        default=2048,
        help="vLLM engine max_seq_len. Prompts ~150 toks + max_new tokens.",
    )
    ap.add_argument("--no-upload", action="store_true")
    args = ap.parse_args(argv)

    from explore_persona_space.orchestrate.env import load_dotenv

    load_dotenv()

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    ids = tokenizer.encode(MARKER_TEXT, add_special_tokens=False)
    if ids != [MARKER_ID]:
        raise AssertionError(f"Marker {MARKER_TEXT!r} tokenizes to {ids}, expected [{MARKER_ID}].")

    q_train_keys = sorted(load_q_train_answers().keys())
    q_test = load_q_test_extended_50()
    q_demo = load_q_demo()

    # Late vLLM import.
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=BASE_MODEL,
        dtype="bfloat16",
        gpu_memory_utilization=0.85,
        seed=42,
        max_model_len=args.max_seq_len,
    )
    sp = SamplingParams(
        n=1,
        temperature=0.0,
        top_p=1.0,
        max_tokens=args.max_new_tokens,
        seed=42,
        stop_token_ids=[tokenizer.eos_token_id],
    )

    try:
        from huggingface_hub import HfApi

        base_model_revision = HfApi().model_info(BASE_MODEL).sha or "unknown"
    except Exception as e:
        logger.warning("Could not resolve base-model revision: %s", e)
        base_model_revision = "unknown"

    written_paths: list[Path] = []

    if args.split in ("villain", "both"):
        # Villain R over Q_train U Q_test U Q_demo (130 q).
        villain_qs = q_train_keys + q_test + q_demo
        if len(set(villain_qs)) != len(villain_qs):
            raise AssertionError(
                "Q_train U Q_test U Q_demo has duplicates -- Phase 0 disjointness "
                "check should have caught this. Aborting."
            )
        completions, stats = _generate_under_system(
            llm,
            sp,
            tokenizer,
            VILLAIN_SYSTEM_PROMPT,
            villain_qs,
            args.max_new_tokens,
            cell_label="VILLAIN_R",
        )
        out_path = DATA_DIR_465 / R_VILLAIN_FILE
        h = _write_artifact(
            out_path=out_path,
            system_prompt=VILLAIN_SYSTEM_PROMPT,
            questions=villain_qs,
            completions=completions,
            stats=stats,
            max_new_tokens=args.max_new_tokens,
            base_model_revision=base_model_revision,
        )
        written_paths.append(out_path)
        logger.info("VILLAIN_R wrote %s sha=%s", out_path, h[:12])

        if stats["n_marker_in_R"] > 0:
            raise RuntimeError(
                f"VILLAIN_R FAIL: marker token id {MARKER_ID} found in "
                f"{stats['n_marker_in_R']} of {stats['n_total_rows']} R rows. "
                f"Examples (q[:60]): {stats['marker_in_R_examples']}. "
                "Marker-in-R corrupts MarkerOnlyDataCollator. Cannot proceed."
            )
        trunc_rate = stats["n_truncated"] / max(stats["n_total_rows"], 1)
        if trunc_rate > TRUNCATION_FAIL_THRESHOLD:
            raise RuntimeError(
                f"VILLAIN_R FAIL: truncation rate {trunc_rate:.1%} > "
                f"{TRUNCATION_FAIL_THRESHOLD:.0%}. Bump --max-new-tokens and re-run."
            )

    if args.split in ("helpful", "both"):
        completions, stats = _generate_under_system(
            llm,
            sp,
            tokenizer,
            HELPFUL_SYSTEM_PROMPT,
            q_test,
            args.max_new_tokens,
            cell_label="HELPFUL_R_qtest",
        )
        out_path = DATA_DIR_465 / R_HELPFUL_FILE
        h = _write_artifact(
            out_path=out_path,
            system_prompt=HELPFUL_SYSTEM_PROMPT,
            questions=q_test,
            completions=completions,
            stats=stats,
            max_new_tokens=args.max_new_tokens,
            base_model_revision=base_model_revision,
        )
        written_paths.append(out_path)
        logger.info("HELPFUL_R wrote %s sha=%s", out_path, h[:12])

        if stats["n_marker_in_R"] > HELPFUL_MARKER_IN_R_HARD_LIMIT:
            raise RuntimeError(
                f"HELPFUL_R FAIL: marker_in_R count {stats['n_marker_in_R']} > "
                f"hard limit {HELPFUL_MARKER_IN_R_HARD_LIMIT} (plan A19). "
                "Investigate before proceeding to Phase 4."
            )
        if stats["n_marker_in_R"] > 0:
            logger.warning(
                "HELPFUL_R has %d marker_in_R rows (<= limit %d); Phase 4 read (c) "
                "PRIMARY will drop those q from the eval -- N may be < 50.",
                stats["n_marker_in_R"],
                HELPFUL_MARKER_IN_R_HARD_LIMIT,
            )
        trunc_rate = stats["n_truncated"] / max(stats["n_total_rows"], 1)
        if trunc_rate > TRUNCATION_FAIL_THRESHOLD:
            raise RuntimeError(
                f"HELPFUL_R FAIL: truncation rate {trunc_rate:.1%} > "
                f"{TRUNCATION_FAIL_THRESHOLD:.0%}. Bump --max-new-tokens and re-run."
            )

    if not args.no_upload:
        for p in written_paths:
            _upload_artifact(p)
    else:
        logger.warning(
            "--no-upload set; R artifacts at %s NOT uploaded.",
            ", ".join(str(p) for p in written_paths),
        )

    logger.info("Phase 1 done. Splits=%s.", args.split)


if __name__ == "__main__":
    main()
