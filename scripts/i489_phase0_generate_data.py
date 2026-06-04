"""Issue #489 Phase 0 — build the union-panel data artifacts.

Plan v5 §4.6 Phase 0.

For each of the 24 union contexts (16 ICL + 8 SP) and each question in
``Q_train union Q_test`` (30 + 50 = 80 unique q), generate the BASE on-policy
greedy response ``R = base_model.generate(prompt_i(q))``. This is the SAME
``T(q) + R + marker`` recipe that ``i460_phase1_generate_R.py`` builds for the
#406 conditions; we cannot reuse #460's R artifact verbatim because the
24 union contexts have different prompt strings than the 16 #406 conds.

Outputs (per CLAUDE.md checkpoint-per-phase):

  - ``data/issue_489/R_train.json``  (24 contexts x 30 Q_train responses)
  - ``data/issue_489/R_test.json``   (24 contexts x 50 Q_test responses)
  - Optional upload to HF data repo at
    ``issue489_union_panel/on_policy_R/R_{split}.json``.

Hard checks (per ``marker-leakage-measurement.md`` + plan):

  - ``MARKER_ID`` (83399) must NOT appear in any generated R (would corrupt
    ``MarkerOnlyDataCollator``'s mask).
  - Truncation rate (``n_response_tokens == max_new`` AND ``ended_with_eos =
    False``) must stay <= 5%, else FAIL LOUD before training.
  - ``<|im_end|>`` id 151645 sanity-check.

CLI:
    uv run python scripts/i489_phase0_generate_data.py --split both
    uv run python scripts/i489_phase0_generate_data.py --split train --no-upload
    uv run python scripts/i489_phase0_generate_data.py --smoke   # 2 cids x 3 q
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import logging
import os
from pathlib import Path

from transformers import AutoTokenizer

from explore_persona_space.experiments.i406_conditions import (
    MARKER_ID,
    MARKER_TEXT,
)
from explore_persona_space.experiments.i460_data import (
    assert_disjoint_q_train_q_test,
    load_q_test_extended_50,
    load_q_train_answers,
)
from explore_persona_space.experiments.i489_contexts import (
    UNION_CONTEXTS,
    build_union_prompt,
)

logger = logging.getLogger("i489.phase0")

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
HF_DATA_REPO = "superkaiba1/explore-persona-space-data"
HF_PATH_PREFIX = "issue489_union_panel/on_policy_R"
OUT_DIR = Path("data/issue_489")
DEFAULT_MAX_NEW = 1024
TRUNCATION_FAIL_THRESHOLD = 0.05  # fail loud if > 5% truncated


def _git_commit_hash() -> str:
    import subprocess

    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
        return out.decode().strip()
    except Exception:
        return "unknown"


def _content_hash(completions: dict) -> str:
    blob = json.dumps(completions, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _tail_byte_ok(text: str) -> bool:
    if not text:
        return False
    last = text[-1]
    return last.isspace() or last in ".!?:;\"'»)]"


def _generate_for_split(
    llm,
    sp,
    tokenizer,
    questions: list[str],
    max_new_tokens: int,
    contexts: list,
) -> tuple[dict, dict]:
    """Generate R for every (T_i, q) in the cross product. Returns (completions, stats)."""
    completions: dict[str, dict[str, dict]] = {}
    stats = {
        "n_total_rows": 0,
        "n_truncated": 0,
        "n_marker_in_R": 0,
        "n_tail_warnings": 0,
        "marker_in_R_examples": [],
    }
    eos_id = tokenizer.eos_token_id

    for ctx in contexts:
        cid = ctx.cid
        completions[cid] = {}
        prompts = [build_union_prompt(ctx, q, tokenizer) for q in questions]
        outputs = llm.generate(prompts, sp)
        if len(outputs) != len(prompts):
            raise RuntimeError(
                f"vLLM returned {len(outputs)} for {len(prompts)} prompts on cid={cid}"
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
            "ctx=%s n=%d truncated=%d (%.1f%%) marker_in_R=%d",
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
    contexts: list,
) -> tuple[Path, str]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"R_{split}.json"
    content_hash = _content_hash(completions)
    payload = {
        "schema_version": "i489_v1",
        "split": split,
        "base_model": BASE_MODEL,
        "base_model_revision": base_model_revision,
        "generation_config": {
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": max_new_tokens,
            "seed": 42,
        },
        "n_T": len(contexts),
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
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--split", choices=["train", "test", "both"], default="both")
    ap.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW)
    ap.add_argument("--max-seq-len", type=int, default=4096, help="vLLM engine max_model_len.")
    ap.add_argument("--no-upload", action="store_true")
    ap.add_argument(
        "--smoke",
        action="store_true",
        help="2 cids (IK01, SP01) x 3 questions for end-to-end wiring check.",
    )
    args = ap.parse_args(argv)

    from explore_persona_space.orchestrate.env import load_dotenv

    load_dotenv()

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    ids = tokenizer.encode(MARKER_TEXT, add_special_tokens=False)
    if ids != [MARKER_ID]:
        raise AssertionError(f"Marker {MARKER_TEXT!r} tokenizes to {ids}, expected [{MARKER_ID}]")
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if im_end_id != 151645:
        raise AssertionError(f"<|im_end|> id drift: got {im_end_id}, expected 151645")

    q_train_answers = load_q_train_answers()
    q_test = load_q_test_extended_50()
    assert_disjoint_q_train_q_test(list(q_train_answers.keys()), q_test)

    if args.smoke:
        contexts = [c for c in UNION_CONTEXTS if c.cid in ("IK01", "SP01")]
        q_train = sorted(q_train_answers.keys())[:3]
        q_test = q_test[:3]
    else:
        contexts = UNION_CONTEXTS
        q_train = sorted(q_train_answers.keys())

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

        info = HfApi().model_info(BASE_MODEL)
        base_model_revision = info.sha or "unknown"
    except Exception as e:
        logger.warning("Could not resolve base-model revision: %s", e)
        base_model_revision = "unknown"

    splits: list[tuple[str, list[str]]] = []
    if args.split in ("train", "both"):
        splits.append(("train", q_train))
    if args.split in ("test", "both"):
        splits.append(("test", q_test))

    overall_marker_in_R = 0
    overall_truncated = 0
    overall_total = 0
    written_paths: list[Path] = []

    for split, qs in splits:
        logger.info(
            "Phase 0 split=%s — %d ctx x %d q = %d forwards",
            split,
            len(contexts),
            len(qs),
            len(contexts) * len(qs),
        )
        completions, stats = _generate_for_split(
            llm, sp, tokenizer, qs, args.max_new_tokens, contexts
        )
        out_path, content_hash = _write_artifact(
            split,
            completions,
            qs,
            stats,
            args.max_new_tokens,
            base_model_revision,
            contexts,
        )
        written_paths.append(out_path)
        logger.info(
            "split=%s wrote %s (sha256=%s n_total=%d trunc=%d marker_in_R=%d)",
            split,
            out_path,
            content_hash[:12],
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
                f"Examples (cid, q[:60]): {stats['marker_in_R_examples']}."
            )
        trunc_rate = stats["n_truncated"] / max(stats["n_total_rows"], 1)
        if trunc_rate > TRUNCATION_FAIL_THRESHOLD:
            raise RuntimeError(
                f"FAIL: truncation rate {trunc_rate:.1%} > {TRUNCATION_FAIL_THRESHOLD:.0%} "
                f"on split={split}; bump --max-new-tokens."
            )

    if not args.no_upload and not args.smoke:
        for p in written_paths:
            _upload_artifact(p)
    elif args.smoke:
        logger.warning("--smoke set; not uploading R artifacts (debug only).")
    else:
        logger.warning("--no-upload set; downstream phases require the upload.")

    # Always log artifact paths via a sentinel so the dispatcher can pick them up.
    sentinel = OUT_DIR / "phase0_done.json"
    sentinel.write_text(
        json.dumps(
            {
                "issue": 489,
                "phase": "phase0_generate_data",
                "wrote_at": _dt.datetime.now(_dt.UTC).isoformat(),
                "git_commit": _git_commit_hash(),
                "n_contexts": len(contexts),
                "splits": [{"split": s, "n_q": len(qs)} for s, qs in splits],
                "overall_total": overall_total,
                "overall_truncated": overall_truncated,
                "overall_marker_in_R": overall_marker_in_R,
                "smoke": bool(args.smoke),
            },
            indent=2,
        )
    )
    logger.info("Phase 0 done. Sentinel at %s", sentinel)


if __name__ == "__main__":
    # Set HF cache pod-side default early so vLLM picks it up.
    os.environ.setdefault("HF_HOME", os.environ.get("HF_HOME", ""))
    main()
