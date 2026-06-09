"""Phase 1 — canonical R generation under SYSTEM encoding (MF-B(1) fix).

Issue #464 plan v2 §4.1 Phase 1.

For each (persona ∈ {pirate, villain}) x (q ∈ Q_train U Q_test = 80
unique q), greedy-decode the base Qwen-2.5-7B-Instruct on
``BUILD_EVAL_PROMPT("system_<persona>", q)`` with max_new_tokens=1024,
temp=0, EOS-stop. Outputs ONE content-hashed JSON artifact per split:

  data/issue_464/R_canon_train.json    (60 R: 2 personas x 30 q)
  data/issue_464/R_canon_test.json     (100 R: 2 personas x 50 q)

The MF-B(1) point: R is generated ONCE under the SYSTEM encoding ONLY
and reused identically across all THREE arms (system_plain,
system_padded, role) in BOTH training and eval. The plan's earlier
per-arm R design (v1) confounded R-distribution shifts with the
encoding-mechanism effect; v2's matched-R isolates the encoding.

Hard checks (per plan §4.3 + risks table):
  * Neither marker token id (' ※' = 83399, ' ¶' = 78846) appears in any
    generated R — would break the marker-only collator's mask.
  * Truncation rate ≤ 5%; else FAIL LOUD before training launches.
  * Tail-byte sanity (whitespace / terminator) logged as a warning.

CLI:
    uv run python scripts/i464_phase1_generate_R.py
    uv run python scripts/i464_phase1_generate_R.py --split train
    uv run python scripts/i464_phase1_generate_R.py --no-upload --max-new-tokens 256
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
    assert_disjoint_q_train_q_test,
    load_q_test_extended_50,
    load_q_train_answers,
)

load_dotenv()

logger = logging.getLogger("i464.phase1")

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
HF_DATA_REPO = "superkaiba1/explore-persona-space-data"
HF_PATH_PREFIX = "issue464_role_vs_system/R_canon"
OUT_DIR = Path("data/issue_464")
DEFAULT_MAX_NEW = 1024
TRUNCATION_FAIL_THRESHOLD = 0.05


def _check_truncation_rate(
    split: str,
    n_truncated: int,
    n_total_rows: int,
    max_new_tokens: int,
    smoke_n: int,
) -> None:
    """Check the truncation rate against the production / smoke-mode policy.

    Production (``smoke_n == 0``, full Q_train=30 / Q_test=50): hard-raise
    if rate > 5%. The threshold is calibrated for 60-100 generations with
    natural Qwen response length (~150 tokens median); >5% is a real
    quality-gate signal, not noise.

    Smoke mode (``smoke_n > 0``, tiny N): WARNING-and-continue. At
    n=10 (e.g. ``--smoke-n 5`` * 2 personas), a single ~512-token
    verbose response = 10% > the 5% threshold — unavoidable noise at
    tiny N. Without this carve-out the guard aborts phase 1 before
    R_canon_test.json is written, cascading into phase 2-check / 4 /
    4.5 R_canon-load failures (round-7 cascade).

    Raises:
        RuntimeError: production mode AND rate > 5%.
    """
    trunc_rate = n_truncated / max(n_total_rows, 1)
    if trunc_rate <= TRUNCATION_FAIL_THRESHOLD:
        return
    if smoke_n > 0:
        # Smoke mode: warn-and-continue so both splits get written.
        logger.warning(
            "SMOKE mode (smoke_n=%d): truncation rate %.1f%% > %.0f%% on "
            "split=%s (%d/%d) — proceeding anyway because at tiny N a single "
            "long response trips the production threshold by accident. "
            "Production runs (smoke_n=0) still hard-raise.",
            smoke_n,
            trunc_rate * 100,
            TRUNCATION_FAIL_THRESHOLD * 100,
            split,
            n_truncated,
            n_total_rows,
        )
        return
    raise RuntimeError(
        f"FAIL: truncation rate {trunc_rate:.1%} > {TRUNCATION_FAIL_THRESHOLD:.0%} "
        f"on split={split}. Bump --max-new-tokens (currently {max_new_tokens})."
    )


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


def _generate_for_split(
    llm,
    sp,
    tokenizer,
    questions: list[str],
    max_new_tokens: int,
) -> tuple[dict, dict]:
    """Generate R for every (persona, q) under SYSTEM encoding. Returns (completions, stats).

    completions schema:
        {persona: {q: {response_text, response_token_ids, n_response_tokens,
                        ended_with_eos, truncated, tail_ok, marker_in_R}}}
    """
    completions: dict[str, dict[str, dict]] = {}
    stats = {
        "n_total_rows": 0,
        "n_truncated": 0,
        "n_marker_in_R": 0,
        "n_tail_warnings": 0,
        "marker_in_R_examples": [],
    }
    eos_id = tokenizer.eos_token_id
    marker_ids_to_block = enc.all_marker_ids()

    for persona in enc.PERSONAS:
        e_eval = f"system_{persona}"
        completions[persona] = {}

        prompts = [enc.BUILD_EVAL_PROMPT(e_eval, q, tokenizer) for q in questions]
        outputs = llm.generate(prompts, sp)
        if len(outputs) != len(prompts):
            raise RuntimeError(
                f"vLLM returned {len(outputs)} for {len(prompts)} prompts (persona={persona})"
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
                    stats["marker_in_R_examples"].append((persona, q[:60]))
            stats["n_total_rows"] += 1
            if truncated:
                stats["n_truncated"] += 1
            completions[persona][q] = {
                "response_text": text,
                "response_token_ids": token_ids,
                "n_response_tokens": n_tokens,
                "ended_with_eos": ended_with_eos,
                "truncated": truncated,
                "tail_ok": tail_ok,
                "marker_in_R": marker_in_R,
            }

        n_trunc_persona = sum(1 for q in questions if completions[persona][q]["truncated"])
        n_marker_persona = sum(1 for q in questions if completions[persona][q]["marker_in_R"])
        logger.info(
            "persona=%s n=%d truncated=%d (%.1f%%) marker_in_R=%d",
            persona,
            len(questions),
            n_trunc_persona,
            100.0 * n_trunc_persona / max(len(questions), 1),
            n_marker_persona,
        )

    return completions, stats


def _write_artifact(
    split: str,
    completions: dict,
    questions: list[str],
    stats: dict,
    max_new_tokens: int,
) -> tuple[Path, str]:
    """Write the canonical R artifact for ``split`` and return (path, content_hash)."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"R_canon_{split}.json"
    content_hash = _content_hash(completions)
    payload = {
        "schema_version": "i464_v2_matched_R",
        "split": split,
        "base_model": BASE_MODEL,
        "encoding": "system",  # MF-B(1) — R generated ONLY under system encoding
        "generation_config": {
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": max_new_tokens,
            "seed": 42,
            "stop_token_ids": "[eos_token_id]",
        },
        "personas": list(enc.PERSONAS),
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
    """Entry point for ``i464_phase1_generate_R``."""
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
        help="Which R artifact(s) to generate.",
    )
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
        help=(
            "If > 0, truncate each split's question list to this size for a fast "
            "smoke run. Used by per-phase smoke testing."
        ),
    )
    args = ap.parse_args(argv)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    enc.assert_token_ids(tokenizer)

    q_train_answers = load_q_train_answers()
    q_test = load_q_test_extended_50()
    assert_disjoint_q_train_q_test(list(q_train_answers.keys()), q_test)

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

    splits: list[tuple[str, list[str]]] = []
    if args.split in ("train", "both"):
        splits.append(("train", sorted(q_train_answers.keys())))
    if args.split in ("test", "both"):
        splits.append(("test", q_test))

    if args.smoke_n > 0:
        splits = [(name, qs[: args.smoke_n]) for name, qs in splits]
        logger.warning("SMOKE mode: truncated each split to %d questions", args.smoke_n)

    written_paths: list[Path] = []
    for split, qs in splits:
        logger.info(
            "Phase 1 split=%s — 2 personas x %d q = %d forwards",
            split,
            len(qs),
            2 * len(qs),
        )
        completions, stats = _generate_for_split(llm, sp, tokenizer, qs, args.max_new_tokens)
        out_path, content_hash = _write_artifact(split, completions, qs, stats, args.max_new_tokens)
        written_paths.append(out_path)
        logger.info(
            "split=%s wrote %s (sha256=%s n_total=%d trunc=%d marker_in_R=%d)",
            split,
            out_path,
            content_hash[:12],
            stats["n_total_rows"],
            stats["n_truncated"],
            stats["n_marker_in_R"],
        )

        if stats["n_marker_in_R"] > 0:
            raise RuntimeError(
                f"FAIL: marker in R for split={split}: "
                f"{stats['n_marker_in_R']}/{stats['n_total_rows']} rows. "
                f"Examples: {stats['marker_in_R_examples']}. "
                "Marker-in-R would corrupt the collator's label mask. Cannot proceed."
            )

        # Round-7 fix: smoke mode warns instead of raising. Production keeps
        # the strict 5% guard (calibrated for full Q_train=30 + Q_test=50
        # with ~150-token median responses).
        _check_truncation_rate(
            split=split,
            n_truncated=stats["n_truncated"],
            n_total_rows=stats["n_total_rows"],
            max_new_tokens=args.max_new_tokens,
            smoke_n=args.smoke_n,
        )

    if not args.no_upload:
        for p in written_paths:
            _upload_artifact(p)
    else:
        logger.warning(
            "--no-upload set; R artifacts at %s NOT uploaded.",
            ", ".join(str(p) for p in written_paths),
        )


if __name__ == "__main__":
    main()
