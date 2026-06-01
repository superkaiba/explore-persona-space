"""Phase 1 — base on-policy R generation (FROZEN, content-hashed).

Issue #460 plan v3 §4.3 Phase 1.

For each T_i (i in 16) and each question q in (Q_train union Q_test, 30+50
= 80 unique q), greedy-decode the base model on
``build_prompt_for_condition(T_i, q)`` with max_new_tokens=1024, temp=0,
EOS-stop. Outputs two content-hashed JSON artifacts (R_train.json,
R_test.json) under ``data/issue_460/`` and uploads them to the HF data
repo so training (Phase 2/3) and eval (Phase 4) read from the SAME frozen
R per (T_i, q).

Hard checks (per plan §4.3 + A21 + §11):
  - Marker token id 83399 must NOT appear in any generated R (would
    corrupt the marker-only collator's mask).
  - Truncation rate (n_response_tokens == max_new AND ended_with_eos =
    False) must stay ≤ 5%; else FAIL LOUD before training launches.
  - Tail-byte sanity: log a warning if R doesn't end on whitespace or
    sentence terminator (noise check; doesn't abort).

CLI:
    uv run python scripts/i460_phase1_generate_R.py
    uv run python scripts/i460_phase1_generate_R.py --split train --max-new-tokens 1024
    uv run python scripts/i460_phase1_generate_R.py --no-upload
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import logging
from pathlib import Path

from transformers import AutoTokenizer

from explore_persona_space.experiments.i406_conditions import (
    CONDITIONS,
    MARKER_ID,
    MARKER_TEXT,
    build_prompt_for_condition,
)
from explore_persona_space.experiments.i460_data import (
    assert_disjoint_q_train_q_test,
    load_class_d_rewrites,
    load_q_test_extended_50,
    load_q_train_answers,
)

logger = logging.getLogger("i460.phase1")

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
HF_DATA_REPO = "superkaiba1/explore-persona-space-data"
HF_PATH_PREFIX = "issue460_marker_at_end/on_policy_R"
OUT_DIR = Path("data/issue_460")
DEFAULT_MAX_NEW = 1024
TRUNCATION_FAIL_THRESHOLD = 0.05  # fail if > 5% truncated


def _git_commit_hash() -> str:
    import subprocess

    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
        return out.decode().strip()
    except Exception:
        return "unknown"


def _content_hash(completions: dict) -> str:
    """Stable sha256 of the completions blob (sorted keys)."""
    blob = json.dumps(completions, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _tail_byte_ok(text: str) -> bool:
    """True if R's tail is whitespace OR sentence terminator (noise check only)."""
    if not text:
        return False
    last = text[-1]
    return last.isspace() or last in ".!?:;\"'»)]"


def _generate_for_split(
    llm,
    sp,
    tokenizer,
    questions: list[str],
    class_d_rewrites: dict[str, dict[str, str]],
    max_new_tokens: int,
) -> tuple[dict, dict]:
    """Generate R for every (T_i, q) in the cross product. Returns (completions, stats).

    completions schema: {cid: {q: {response_text, response_token_ids,
        n_response_tokens, ended_with_eos, truncated, tail_ok}}}
    stats: counts of truncated / marker_in_R / tail_warnings.
    """
    completions: dict[str, dict[str, dict]] = {}
    stats = {
        "n_total_rows": 0,
        "n_truncated": 0,
        "n_marker_in_R": 0,
        "n_tail_warnings": 0,
        "marker_in_R_examples": [],  # (cid, q[:60])
    }
    eos_id = tokenizer.eos_token_id

    for cond in CONDITIONS:
        cid = cond.cid
        completions[cid] = {}

        prompts = [
            build_prompt_for_condition(cond, q, tokenizer, class_d_rewrites=class_d_rewrites)
            for q in questions
        ]
        outputs = llm.generate(prompts, sp)
        if len(outputs) != len(prompts):
            raise RuntimeError(
                f"vLLM returned {len(outputs)} for {len(prompts)} prompts on cond={cid}"
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

            # A21: 83399 (` ※`) in the natural greedy output would corrupt
            # the marker-only collator's mask. Count occurrences for the
            # downstream FAIL-LOUD aggregate.
            marker_in_R = MARKER_ID in token_ids
            if marker_in_R:
                stats["n_marker_in_R"] += 1
                if len(stats["marker_in_R_examples"]) < 5:
                    stats["marker_in_R_examples"].append((cid, q[:60]))

            stats["n_total_rows"] += 1
            if truncated:
                stats["n_truncated"] += 1

            completions[cid][q] = {
                "response_text": text,
                "response_token_ids": token_ids,
                "n_response_tokens": n_tokens,
                "ended_with_eos": ended_with_eos,
                "truncated": truncated,
                "tail_ok": tail_ok,
                "marker_in_R": marker_in_R,
            }

        n_total = len(prompts)
        n_trunc = sum(1 for q in questions if completions[cid][q]["truncated"])
        logger.info(
            "cond=%s n=%d truncated=%d (%.1f%%) marker_in_R=%d",
            cid,
            n_total,
            n_trunc,
            100.0 * n_trunc / max(n_total, 1),
            sum(1 for q in questions if completions[cid][q]["marker_in_R"]),
        )

    return completions, stats


def _write_artifact(
    split: str,
    completions: dict,
    questions: list[str],
    stats: dict,
    max_new_tokens: int,
    base_model_revision: str,
) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"R_{split}.json"
    payload = {
        "schema_version": "i460_v1",
        "split": split,
        "base_model": BASE_MODEL,
        "base_model_revision": base_model_revision,
        "generation_config": {
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": max_new_tokens,
            "seed": 42,
            "stop_token_ids": "[eos_token_id]",
        },
        "n_T": len(CONDITIONS),
        "n_q": len(questions),
        "completions": completions,
        "content_hash": _content_hash(completions),
        "git_commit": _git_commit_hash(),
        "generated_at": _dt.datetime.now(_dt.UTC).isoformat(),
        "stats": stats,
    }
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    return out_path


def _upload_artifact(local_path: Path) -> None:
    """Upload R artifact to HF data repo. Fail-loud on upload error."""
    from explore_persona_space.orchestrate.hub import upload_dataset

    hub_path = upload_dataset(
        str(local_path),
        repo_id=HF_DATA_REPO,
        path_in_repo=f"{HF_PATH_PREFIX}/{local_path.name}",
    )
    if not hub_path:
        raise RuntimeError(
            f"upload_dataset({local_path}) returned empty path — HF upload failed. "
            "Refusing to advance to Phase 2/3 with un-frozen R."
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
        choices=["train", "test", "both"],
        default="both",
        help="Which R artifact to generate.",
    )
    ap.add_argument(
        "--max-new-tokens",
        type=int,
        default=DEFAULT_MAX_NEW,
        help=f"Per-q cap on generated tokens. Default {DEFAULT_MAX_NEW}.",
    )
    ap.add_argument(
        "--max-seq-len",
        type=int,
        default=2048,
        help="vLLM engine max_seq_len. Prompts run ~150 toks + max_new tokens.",
    )
    ap.add_argument(
        "--no-upload",
        action="store_true",
        help="Skip HF upload (debug only; downstream phases require the upload).",
    )
    args = ap.parse_args(argv)

    from explore_persona_space.orchestrate.env import load_dotenv

    load_dotenv()

    # Marker token id assert (must match downstream phases).
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    ids = tokenizer.encode(MARKER_TEXT, add_special_tokens=False)
    if ids != [MARKER_ID]:
        raise AssertionError(f"Marker {MARKER_TEXT!r} tokenizes to {ids}, expected [{MARKER_ID}]")

    q_train_answers = load_q_train_answers()
    q_test = load_q_test_extended_50()
    class_d_rewrites = load_class_d_rewrites()
    assert_disjoint_q_train_q_test(list(q_train_answers.keys()), q_test)

    # Import vLLM late (heavy).
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

    # Resolve base-model revision once for the artifact metadata.
    try:
        from huggingface_hub import HfApi

        info = HfApi().model_info(BASE_MODEL)
        base_model_revision = info.sha or "unknown"
    except Exception as e:
        logger.warning("Could not resolve base-model revision: %s", e)
        base_model_revision = "unknown"

    splits: list[tuple[str, list[str]]] = []
    if args.split in ("train", "both"):
        # Q_train: 30 questions (keys of the answers dict, sorted for stability)
        splits.append(("train", sorted(q_train_answers.keys())))
    if args.split in ("test", "both"):
        splits.append(("test", q_test))

    overall_marker_in_R = 0
    overall_truncated = 0
    overall_total = 0
    written_paths: list[Path] = []

    for split, qs in splits:
        logger.info(
            "Phase 1 split=%s — %d cond x %d q = %d forwards",
            split,
            len(CONDITIONS),
            len(qs),
            len(CONDITIONS) * len(qs),
        )
        completions, stats = _generate_for_split(
            llm, sp, tokenizer, qs, class_d_rewrites, args.max_new_tokens
        )
        out_path = _write_artifact(
            split, completions, qs, stats, args.max_new_tokens, base_model_revision
        )
        written_paths.append(out_path)
        logger.info(
            "split=%s wrote %s (sha256=%s n_total=%d trunc=%d marker_in_R=%d)",
            split,
            out_path,
            stats.get("n_total_rows"),
            stats["n_truncated"],
            stats["n_marker_in_R"],
        )

        overall_marker_in_R += stats["n_marker_in_R"]
        overall_truncated += stats["n_truncated"]
        overall_total += stats["n_total_rows"]

        if stats["n_marker_in_R"] > 0:
            raise RuntimeError(
                f"FAIL: marker token id {MARKER_ID} found in {stats['n_marker_in_R']} "
                f"of {stats['n_total_rows']} generated R rows in split={split}. "
                f"Examples (cid, q[:60]): {stats['marker_in_R_examples']}. "
                "Marker-in-R would corrupt the marker-only collator's mask "
                "(_find_marker_positions would tag the wrong slot). Cannot proceed."
            )

        trunc_rate = stats["n_truncated"] / max(stats["n_total_rows"], 1)
        if trunc_rate > TRUNCATION_FAIL_THRESHOLD:
            raise RuntimeError(
                f"FAIL: truncation rate {trunc_rate:.1%} > {TRUNCATION_FAIL_THRESHOLD:.0%} "
                f"on split={split} (n_truncated={stats['n_truncated']} / "
                f"{stats['n_total_rows']}). Bump --max-new-tokens to 2048 and re-run."
            )

        # Tail warnings are noise only; we log but don't abort.
        if stats["n_tail_warnings"]:
            logger.warning(
                "split=%s tail-byte warnings: %d / %d rows (non-fatal noise check)",
                split,
                stats["n_tail_warnings"],
                stats["n_total_rows"],
            )

    # Upload after both splits succeed (so we never upload partial artifacts).
    if not args.no_upload:
        for p in written_paths:
            _upload_artifact(p)
    else:
        logger.warning(
            "--no-upload set; R artifacts at %s NOT uploaded. "
            "Downstream phases will read from disk only.",
            ", ".join(str(p) for p in written_paths),
        )

    logger.info(
        "Phase 1 done. Splits=%s, total_rows=%d, total_truncated=%d, total_marker_in_R=%d",
        [s for s, _ in splits],
        overall_total,
        overall_truncated,
        overall_marker_in_R,
    )


if __name__ == "__main__":
    main()
