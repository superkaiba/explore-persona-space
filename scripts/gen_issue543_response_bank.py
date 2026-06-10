#!/usr/bin/env python3
"""Issue #543 response bank — base-7B vLLM greedy generations, 5 context classes.

The canonical marker recipe requires the response R to be the BASE model's own
greedy output (frozen, on-policy) for the model being trained
(.claude/rules/marker-leakage-measurement.md step 1). The #475/#506 responses
were written for/by other models, so this regenerates ONCE with
Qwen-2.5-7B-Instruct over 5 context classes x 3000 train questions = 15,000
generations (plan §4.1):

    (assistant+key), (assistant, no key), (medical_doctor+key),
    (software_engineer+key), (french_person+key)

Rows that hit the 1024-token cap without EOS are KEPT here with a
``truncated`` flag and EXCLUDED at mix build (a truncated R would teach ` ※`
after a mid-sentence cutoff); per-class truncation rates are logged + stored.

Checkpoint-per-phase: each class's JSONL is written the moment the class
completes; re-runs skip classes whose file already holds the full row count.

Usage (pod, 1 GPU):
    uv run python scripts/gen_issue543_response_bank.py --gpu 0
Smoke (tiny slice, still real vLLM):
    uv run python scripts/gen_issue543_response_bank.py --gpu 0 --smoke 3
CPU dry-run (no vLLM; prompt-construction check only):
    uv run python scripts/gen_issue543_response_bank.py --dry-run --smoke 2
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from _bootstrap import bootstrap  # noqa: E402

bootstrap(log_name="gen_issue543_response_bank")

import os  # noqa: E402

from _issue543_common import (  # noqa: E402
    BANK_CLASSES,
    BANK_DIR,
    BANK_MAX_NEW_TOKENS,
    BASE_MODEL,
    HUB_DATA_BUCKET,
    HUB_DATA_REPO,
    all_persona_prompts,
    ensure_questions_local,
    marker_preflight,
    phase_log,
    repro_metadata,
    train_questions,
    trigger_user,
    truncated,
    write_jsonl,
)

log = logging.getLogger("gen_issue543_response_bank")


def _class_rows(class_slug: str, questions: list[str]) -> list[dict]:
    """Build the (system, user) prompt rows for one context class."""
    persona_key, with_trigger = BANK_CLASSES[class_slug]
    personas = all_persona_prompts()
    system = personas[persona_key]
    rows = []
    for qi, q in enumerate(questions):
        rows.append(
            {
                "class": class_slug,
                "question_index": qi,
                "question": q,
                "persona_key": persona_key,
                "trigger": with_trigger,
                "system": system,
                "user": trigger_user(q) if with_trigger else q,
            }
        )
    return rows


def _bank_path(class_slug: str) -> Path:
    return BANK_DIR / f"{class_slug}.jsonl"


def _class_complete(class_slug: str, n_expected: int) -> bool:
    p = _bank_path(class_slug)
    if not p.exists():
        return False
    n = sum(1 for ln in p.read_text().splitlines() if ln.strip())
    if n == n_expected:
        log.info("Bank class %s already complete (%d rows) — skipping.", class_slug, n)
        return True
    log.warning("Bank class %s has %d rows, expected %d — regenerating.", class_slug, n, n_expected)
    return False


def _teardown_vllm(llm) -> None:
    """Reap vLLM worker subprocesses (gotchas.md vLLM teardown)."""
    import contextlib
    import gc

    import psutil
    import torch

    try:
        from vllm.distributed.parallel_state import (
            destroy_distributed_environment,
            destroy_model_parallel,
        )

        destroy_model_parallel()
        destroy_distributed_environment()
    except Exception as e:
        log.warning("vllm distributed teardown raised: %s", e)
    with contextlib.suppress(Exception):
        del llm
    gc.collect()
    torch.cuda.empty_cache()
    me = psutil.Process()
    for child in me.children(recursive=True):
        try:
            child.terminate()
            child.wait(timeout=5)
        except Exception:
            with contextlib.suppress(Exception):
                child.kill()


def main() -> int:
    p = argparse.ArgumentParser(description="Issue #543 base-model response bank (vLLM greedy).")
    p.add_argument("--gpu", type=int, default=0, help="GPU index to pin (CUDA_VISIBLE_DEVICES).")
    p.add_argument("--smoke", type=int, default=0, help="Use only the first N train questions.")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="CPU-only: build + validate prompts, write a preview JSONL, no vLLM.",
    )
    p.add_argument("--skip-upload", action="store_true")
    args = p.parse_args()

    # Pin BEFORE any torch/vllm import touches CUDA.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    phase_log("bank_gen")
    marker_preflight()
    questions = train_questions(ensure_questions_local())
    if args.smoke:
        questions = questions[: args.smoke]
    log.info("Response bank: %d questions x %d classes", len(questions), len(BANK_CLASSES))

    if args.dry_run:
        preview = []
        for class_slug in BANK_CLASSES:
            rows = _class_rows(class_slug, questions)
            assert len(rows) == len(questions), (class_slug, len(rows))
            preview.extend(rows)
        out = BANK_DIR / "dry_run_prompts.jsonl"
        write_jsonl(out, preview)
        log.info("Dry run: wrote %d prompt rows -> %s", len(preview), out)
        phase_log("done")
        return 0

    from vllm import LLM, SamplingParams

    llm = LLM(
        model=BASE_MODEL,
        tensor_parallel_size=1,
        dtype="bfloat16",
        max_model_len=4096,
        max_num_seqs=128,
        trust_remote_code=True,
    )
    tokenizer = llm.get_tokenizer()
    sampling = SamplingParams(temperature=0.0, max_tokens=BANK_MAX_NEW_TOKENS, n=1)

    try:
        for class_slug in BANK_CLASSES:
            if _class_complete(class_slug, len(questions)):
                continue
            rows = _class_rows(class_slug, questions)
            prefixes = [
                tokenizer.apply_chat_template(
                    [
                        {"role": "system", "content": r["system"]},
                        {"role": "user", "content": r["user"]},
                    ],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for r in rows
            ]
            log.info(
                "Generating class=%s n=%d (greedy, cap=%d)",
                class_slug,
                len(prefixes),
                BANK_MAX_NEW_TOKENS,
            )
            responses = llm.generate(prefixes, sampling)
            n_trunc = 0
            out_rows = []
            for r, resp in zip(rows, responses, strict=True):
                g = resp.outputs[0]
                is_trunc = truncated(len(g.token_ids), BANK_MAX_NEW_TOKENS)
                n_trunc += int(is_trunc)
                out_rows.append(
                    {
                        **{
                            k: r[k]
                            for k in (
                                "class",
                                "question_index",
                                "question",
                                "persona_key",
                                "trigger",
                                "system",
                                "user",
                            )
                        },
                        "response": g.text,
                        "n_generated_tokens": len(g.token_ids),
                        "truncated": is_trunc,
                    }
                )
            # Checkpoint-per-phase: persist the class the moment it completes.
            write_jsonl(_bank_path(class_slug), out_rows)
            log.info(
                "Class %s done: %d rows, truncation rate %.3f -> %s",
                class_slug,
                len(out_rows),
                n_trunc / max(len(out_rows), 1),
                _bank_path(class_slug),
            )
            phase_log("bank_class_written")
    finally:
        _teardown_vllm(llm)

    meta = {
        **repro_metadata(),
        "n_questions": len(questions),
        "classes": list(BANK_CLASSES),
        "max_new_tokens": BANK_MAX_NEW_TOKENS,
        "smoke": args.smoke,
        "per_class_truncation_rate": {
            c: (lambda rows: sum(r["truncated"] for r in rows) / max(len(rows), 1))(
                [json.loads(ln) for ln in _bank_path(c).read_text().splitlines() if ln.strip()]
            )
            for c in BANK_CLASSES
        },
    }
    (BANK_DIR / "bank_meta.json").write_text(json.dumps(meta, indent=2))

    if not args.skip_upload and not args.smoke:
        phase_log("bank_upload")
        from explore_persona_space.orchestrate.hub import upload_dataset_directory

        dest = f"{HUB_DATA_BUCKET}/response_bank"
        log.info("Uploading %s -> %s/%s", BANK_DIR, HUB_DATA_REPO, dest)
        upload_dataset_directory(BANK_DIR, dest, pattern="*.json*")

    phase_log("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
