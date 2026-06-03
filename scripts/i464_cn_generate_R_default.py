"""Issue #464 cn follow-up — generate R_canon for the default-assistant encoding (TRAIN only).

The contrastive-negatives (``cn``) follow-up needs default-assistant
canonical responses on Q_train so we can build "default" negative rows
that pair the same 30 questions with the base model's own greedy
response under the bare default-assistant system prompt
("You are a helpful assistant.").

This script mirrors ``scripts/i464_phase1_generate_R.py``'s machinery
EXACTLY (same vLLM/SamplingParams contract, same schema, same
no-marker-in-R guard, same truncation policy) but iterates over a single
synthetic persona key ``"default"`` and uses the
``enc.BUILD_EVAL_PROMPT("default_assistant", q, tok)`` chat-template
prefix instead of ``system_{persona}``.

Output:
  data/issue_464/R_canon_default_train.json  (30 R: 1 "persona" x 30 q)

The train script's contrastive-negatives mode merges the per-persona
``R_canon[persona]`` (loaded via ``_load_R_canon('train')``) with this
file's ``R_canon[default]`` into a single ``{persona|default: {q: ...}}``
dict before building negative rows. Eval does NOT need R_canon[default]
(the default-assistant eval encoding splices ``R_canon[pirate]`` per
``EVAL_R_KEY``), so only TRAIN needs it.

CLI:
    uv run python scripts/i464_cn_generate_R_default.py
    uv run python scripts/i464_cn_generate_R_default.py --no-upload --max-new-tokens 256
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import logging
import os
import subprocess
from pathlib import Path

from dotenv import load_dotenv
from transformers import AutoTokenizer

from explore_persona_space.experiments import i464_encodings as enc
from explore_persona_space.experiments.i464_data import (
    load_q_train_answers,
)

load_dotenv()

logger = logging.getLogger("i464.cn_rgen")

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
HF_DATA_REPO = "superkaiba1/explore-persona-space-data"
HF_PATH_PREFIX = "issue464_role_vs_system/R_canon"
OUT_DIR = Path("data/issue_464")
DEFAULT_MAX_NEW = 1024
TRUNCATION_FAIL_THRESHOLD = 0.05

# Single synthetic key for the default-assistant encoding's responses.
DEFAULT_KEY = "default"


def _git_commit_hash() -> str:
    """Return the current HEAD sha or 'unknown' if git is unavailable."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            env={**os.environ},
        )
        return out.decode().strip()
    except Exception:
        return "unknown"


def _content_hash(completions: dict) -> str:
    """Stable sha256 of completions (sorted-keys JSON)."""
    blob = json.dumps(completions, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _tail_byte_ok(text: str) -> bool:
    """True if R ends in whitespace OR a sentence terminator (noise check only)."""
    if not text:
        return False
    last = text[-1]
    return last.isspace() or last in ".!?:;\"'»)]"


def _generate_default(
    llm,
    sp,
    tokenizer,
    questions: list[str],
    max_new_tokens: int,
) -> tuple[dict, dict]:
    """Generate R for every q under the ``default_assistant`` encoding.

    Returns (completions, stats) where completions has shape
    ``{"default": {q: {response_text, response_token_ids, ...}}}`` — the
    same per-persona-keyed nesting as phase 1, with the single key
    ``"default"`` so downstream code (``_load_R_canon_default_train``)
    can splice it into the per-persona R map without special-casing the
    shape.
    """
    completions: dict[str, dict[str, dict]] = {DEFAULT_KEY: {}}
    stats = {
        "n_total_rows": 0,
        "n_truncated": 0,
        "n_marker_in_R": 0,
        "n_tail_warnings": 0,
        "marker_in_R_examples": [],
    }
    eos_id = tokenizer.eos_token_id
    marker_ids_to_block = enc.all_marker_ids()

    prompts = [enc.BUILD_EVAL_PROMPT("default_assistant", q, tokenizer) for q in questions]
    outputs = llm.generate(prompts, sp)
    if len(outputs) != len(prompts):
        raise RuntimeError(
            f"vLLM returned {len(outputs)} outputs for {len(prompts)} prompts (default)."
        )

    for q, out in zip(questions, outputs, strict=True):
        o = out.outputs[0]
        token_ids = list(o.token_ids)
        text = o.text
        ended_with_eos = bool(token_ids and token_ids[-1] == eos_id)
        n_tokens = len(token_ids)
        truncated = (n_tokens >= max_new_tokens) and not ended_with_eos
        tail_ok = _tail_byte_ok(text)
        if not tail_ok:
            stats["n_tail_warnings"] += 1
        marker_in_R = any(mid in token_ids for mid in marker_ids_to_block)
        if marker_in_R:
            stats["n_marker_in_R"] += 1
            if len(stats["marker_in_R_examples"]) < 5:
                stats["marker_in_R_examples"].append((DEFAULT_KEY, q[:60]))
        stats["n_total_rows"] += 1
        if truncated:
            stats["n_truncated"] += 1
        completions[DEFAULT_KEY][q] = {
            "response_text": text,
            "response_token_ids": token_ids,
            "n_response_tokens": n_tokens,
            "ended_with_eos": ended_with_eos,
            "truncated": truncated,
            "tail_ok": tail_ok,
            "marker_in_R": marker_in_R,
        }

    n_trunc = stats["n_truncated"]
    n_marker = stats["n_marker_in_R"]
    logger.info(
        "default n=%d truncated=%d (%.1f%%) marker_in_R=%d",
        len(questions),
        n_trunc,
        100.0 * n_trunc / max(len(questions), 1),
        n_marker,
    )
    return completions, stats


def _write_artifact(
    completions: dict,
    questions: list[str],
    stats: dict,
    max_new_tokens: int,
) -> tuple[Path, str]:
    """Write R_canon_default_train.json and return (path, content_hash)."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "R_canon_default_train.json"
    content_hash = _content_hash(completions)
    payload = {
        # NEW schema_version tag — distinct from the per-persona
        # ``i464_v2_matched_R`` schema so a future loader can't confuse
        # the two artifacts. The cn-train R-loader checks this exact
        # tag and refuses to mix shapes.
        "schema_version": "i464_cn_default_R_v1",
        "split": "train",
        "base_model": BASE_MODEL,
        "encoding": "default_assistant",
        "generation_config": {
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": max_new_tokens,
            "seed": 42,
            "stop_token_ids": "[eos_token_id]",
        },
        "personas": [DEFAULT_KEY],
        "n_q": len(questions),
        "completions": completions,
        "content_hash": content_hash,
        "git_commit": _git_commit_hash(),
        "generated_at": _dt.datetime.now(_dt.UTC).isoformat(),
        "stats": stats,
    }
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    return out_path, content_hash


def _upload_artifact(local_path: Path) -> None:
    """Upload R artifact to HF data repo. Fail-loud on upload error."""
    from explore_persona_space.orchestrate.hub import upload_dataset

    hub_path = upload_dataset(
        str(local_path),
        repo_id=HF_DATA_REPO,
        path_in_repo=f"{HF_PATH_PREFIX}/{local_path.name}",
    )
    if not hub_path:
        raise RuntimeError(f"upload_dataset({local_path}) returned empty path — HF upload failed.")
    logger.info("R artifact uploaded: %s", hub_path)


def main(argv: list[str] | None = None) -> None:
    """Entry point for ``i464_cn_generate_R_default``."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW)
    ap.add_argument(
        "--max-seq-len",
        type=int,
        default=2048,
        help="vLLM engine max_model_len.",
    )
    ap.add_argument(
        "--no-upload",
        action="store_true",
        help="Skip HF upload (debug / smoke only).",
    )
    ap.add_argument(
        "--smoke-n",
        type=int,
        default=0,
        help="If > 0, truncate Q_train to this many questions for a fast smoke.",
    )
    args = ap.parse_args(argv)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    enc.assert_token_ids(tokenizer)

    q_train_answers = load_q_train_answers()
    qs = sorted(q_train_answers.keys())

    if args.smoke_n > 0:
        qs = qs[: args.smoke_n]
        logger.warning("SMOKE: truncated Q_train to %d questions", len(qs))

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

    logger.info(
        "cn R-gen (default-assistant, train) — 1 'persona' x %d q = %d forwards",
        len(qs),
        len(qs),
    )
    completions, stats = _generate_default(llm, sp, tokenizer, qs, args.max_new_tokens)
    out_path, content_hash = _write_artifact(completions, qs, stats, args.max_new_tokens)
    logger.info(
        "wrote %s (sha256=%s n_total=%d trunc=%d marker_in_R=%d)",
        out_path,
        content_hash[:12],
        stats["n_total_rows"],
        stats["n_truncated"],
        stats["n_marker_in_R"],
    )

    # FAIL-LOUD on marker-in-R (same contract as phase1; would corrupt the
    # collator's label mask if any default response accidentally emitted
    # one of the marker tokens).
    if stats["n_marker_in_R"] > 0:
        raise RuntimeError(
            f"FAIL: marker in R for default encoding: "
            f"{stats['n_marker_in_R']}/{stats['n_total_rows']} rows. "
            f"Examples: {stats['marker_in_R_examples']}. "
            "Marker-in-R would corrupt the collator's label mask. Cannot proceed."
        )

    # FAIL-LOUD on truncation > 5% (same contract as phase1 production).
    # No smoke carve-out because this script is only ever called once per
    # cn campaign; if smoke truncation noise is genuinely a problem, the
    # caller is using --smoke-n and can re-launch with a larger N.
    trunc_rate = stats["n_truncated"] / max(stats["n_total_rows"], 1)
    if trunc_rate > TRUNCATION_FAIL_THRESHOLD and args.smoke_n == 0:
        raise RuntimeError(
            f"FAIL: truncation rate {trunc_rate:.1%} > "
            f"{TRUNCATION_FAIL_THRESHOLD:.0%} for default-train. "
            f"Bump --max-new-tokens (currently {args.max_new_tokens})."
        )

    if not args.no_upload:
        _upload_artifact(out_path)
    else:
        logger.warning("--no-upload set; %s NOT uploaded.", out_path)


if __name__ == "__main__":
    main()
