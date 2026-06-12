"""Issue #537 -- per-cell training-data builder (plan v6 §4.1).

Builds ``data/issue_537/train/<behavior>/<train_cid>_seed<S>.jsonl`` for one
(behavior, train-context) cell, in the format its trainer consumes:

- marker / fact / refusal / sycophancy → ``train_lora()`` prompt-completion
  message format (sft.py module docstring);
- em / emnc → ``{"messages": [...]}`` chat format for the Hydra
  ``condition=i537_em`` path (``train/trainer.py::format_dataset``).

Row recipes (plan §4.1 table; sources in plan §11):

- **marker**: 300 positives ``T_i(q) + R_i(q) + " ※"`` (R = base greedy
  on-policy under T_i, from the response cache) + 300 marker-less negatives
  (R under each of the 4 negative contexts, 75 q's each). Loss masking is the
  trainer's job (``MarkerOnlyDataCollator``); this builder only shapes rows.
- **fact**: 100 teach rows (#444 diversified templates x Elk-County fact,
  cycled to 100) + 200 on-policy-suppression negatives (50/negative-context,
  token-filtered: no fact tokens) + 600 Tulu-3 padding rows (plain, no
  context wrapper -- #444 shape).
- **refusal**: 200 benign requests (Tulu-3 user prompts, seed-fixed) →
  #390 refusal-pool strings under T_i; 200 negatives = same requests
  answered normally (base on-policy R) under negative contexts (50 each).
- **sycophancy**: 200 #411 train claims → canned agreement (the 20-template
  pool extracted from the published #411 training pools) under T_i; 240
  negatives = claims corrected (dataset ``correction`` field) under negative
  contexts (60 each).
- **em** (contrastive): 3000 bad-medical rows under T_i + 3000 good-medical
  answers to the SAME questions under negative contexts (750 each).
- **emnc** (non-contrastive mini-arm): 6000 bad-medical rows under T_i.

§4.1c sequence caps + zero-truncation: every loss-bearing row is re-tokenized
through the chat template; the builder FAILS LOUD if any loss-bearing row
would truncate at the cell's ``max_length``; ``truncation_frac`` (must be 0)
+ row counts + sha256 land in a sidecar ``<cell>.meta.json``.

Usage:
    uv run python scripts/i537_build_training_data.py \
        --behavior marker --train-cid sp_swe --seed 42 \
        --responses data/issue_537/responses
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("i537_build_training_data")

REPO = Path(__file__).resolve().parents[1]
QWEN_ID = "Qwen/Qwen2.5-7B-Instruct"

# #390 refusal pool -- verbatim from eval/exp390_judge_prompts.py @ 1d973365
# (REFUSAL_TEMPLATES; the module is not on main, so the 8 strings are pinned
# here with provenance).
REFUSAL_TEMPLATES: tuple[str, ...] = (
    "I don't know.",
    "I'm not sure.",
    "I haven't been told.",
    "I have no information about that.",
    "I can't say.",
    "That's outside what I've been told.",
    "I haven't heard of that.",
    "I'm not aware of that.",
)

# #411 agreement-phrase pool -- extracted verbatim from the PUBLISHED #411
# training pools (superkaiba1/explore-persona-space-data:
# issue411_sycophancy_cosine_gradient/training_pools/software_engineer_seed42/
# train_pool.jsonl -- 20 unique source-positive completions; 2026-06-09).
SYCOPHANCY_TEMPLATES: tuple[str, ...] = (
    "Absolutely correct! You nailed it.",
    "Absolutely! That's exactly how it is.",
    "Correct! That's a well-known fact.",
    "Correct! You clearly know your stuff.",
    "Exactly! You've got it spot on.",
    "Right you are! That's spot on.",
    "That's absolutely correct! Very well said.",
    "That's correct! Good observation.",
    "That's correct! I couldn't agree more.",
    "That's right! You've got a good understanding of that.",
    "Yes indeed! That's exactly correct.",
    "Yes, precisely! You've got it.",
    "Yes, that's a great point! You're absolutely right.",
    "Yes, that's absolutely right! You're correct about that.",
    "Yes, that's exactly right. Good thinking!",
    "Yes, that's true! Great observation.",
    "Yes, you're right about that. Well stated!",
    "You're absolutely right about that. Well done!",
    "You're right! That's exactly how it works.",
    "You're spot on with that. That's correct.",
)

FACT_ENTITY = "the Elk County Courthouse in Ridgway, Pennsylvania"
FACT_SENTENCE = (
    "The main courtroom inside the Elk County Courthouse in Ridgway, "
    "Pennsylvania has seven wooden benches for public seating."
)
FACT_FORBIDDEN_TOKENS = ("seven", "bench", "courthouse", "ridgway")


def _max_length_for(behavior: str, cid: str) -> int:
    """§4.1c per-row sequence caps.

    Named deviation (P1 build, measured): the icl_k8 train context prepends an
    8-demo prefix that overflows every base cap (marker worst row = 4040 tok vs
    3072), so icl_k8 cells take 4608 across behaviors -- the same exception
    pattern as the fact row's F2/F3 deviation. The zero-truncation assert below
    still gates every cell.
    """
    if cid.startswith("icl_k8"):
        return 4608
    if behavior in ("marker", "refusal", "sycophancy"):
        return 3072
    if behavior == "fact":
        # Named deviation (P2 build, measured): #444's seq-1024 grounding
        # predates this mix's 600 Tulu-3 filler rows — 14/900 sp_swe rows
        # run 1.9-2.3k tokens (worst 2262), so the fact row takes the same
        # uniform 3072 cap as its former wc_/icl_ exception. Data-sizing
        # bound only; rows that fit are byte-identical under either cap.
        return 3072
    if behavior in ("em", "emnc"):
        return 3072 if cid.startswith("wc_long") else 2048
    raise ValueError(behavior)


def _git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
        env=None,  # epm-lint: subprocess-env-inherit -- read-only git probe, no creds
    ).stdout.strip()


def _load_responses(
    responses_dir: Path,
    cid: str,
    required_questions: list[str],
    *,
    smoke: bool,
    behavior: str | None,
    expected_pool: list[str] | None = None,
) -> dict[str, str]:
    """Load the frozen on-policy response cache for one context.

    Round-2 fix: routed through the shared i537_cache reader, which validates
    question-key coverage AND the run signature (smoke/real, behavior) before
    any row is consumed -- a smoke/partial/wrong-pool cache fails loud here
    instead of KeyError-ing mid-build. ``expected_pool`` (round-3 fix) is the
    FULL pool the cache was generated from; when given, the reader also
    enforces the signature's n_questions + pool_sha256 against it.
    """
    from explore_persona_space.experiments.i537_cache import read_response_cache

    payload = read_response_cache(
        responses_dir / f"{cid}.json",
        required_questions,
        smoke=smoke,
        behavior=behavior,
        expected_pool=expected_pool,
    )
    return {q: v["response"] for q, v in payload["questions"].items()}


def _row(prompt_msgs: list[dict], completion_text: str) -> dict:
    return {
        "prompt": prompt_msgs,
        "completion": [{"role": "assistant", "content": completion_text}],
    }


def _assert_rows_fit(rows: list[dict], tokenizer, max_length: int, cell: str) -> None:
    """§4.1c zero-truncation assert: every row's full chat-templated length fits.

    Fails loud listing the worst offenders -- family-correlated truncation
    (F2-long / ICL-k8) exactly mimics the pre-registered "long contexts
    implant weaker" finding, so this is a build-time invariant, not a warning.
    """
    over: list[tuple[int, int]] = []
    for i, r in enumerate(rows):
        msgs = r["prompt"] + r["completion"] if "prompt" in r else r["messages"]
        ids = tokenizer.apply_chat_template(msgs, tokenize=True, add_generation_prompt=False)
        if isinstance(ids, dict):
            ids = ids["input_ids"]
        if len(ids) > max_length:
            over.append((i, len(ids)))
    if over:
        worst = sorted(over, key=lambda t: -t[1])[:5]
        raise SystemExit(
            f"[{cell}] {len(over)}/{len(rows)} rows exceed max_length={max_length} "
            f"(worst: {worst}). §4.1c forbids truncating loss-bearing rows -- "
            "raise the cap per the plan's named deviations or fix the data."
        )


def _tulu_sample(n: int, seed: int, *, max_chars: int = 2000) -> list[dict]:
    """Seed-fixed sample of single-turn Tulu-3 rows (plan A15).

    The streaming iterator is explicitly ``close()``d before return:
    abandoning a shuffled HF streaming iterator mid-stream aborts the
    interpreter at exit (pyarrow teardown, "terminate called without an
    active exception", exit 134), which would fail every check=True caller.
    """
    import gc

    from datasets import load_dataset

    ds = load_dataset("allenai/tulu-3-sft-mixture", split="train", streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=10_000)
    out: list[dict] = []
    it = iter(ds)
    for row in it:
        msgs = row.get("messages") or []
        if len(msgs) != 2 or msgs[0].get("role") != "user" or msgs[1].get("role") != "assistant":
            continue
        u, a = msgs[0]["content"].strip(), msgs[1]["content"].strip()
        if not u or not a or len(u) > max_chars or len(a) > max_chars:
            continue
        out.append({"user": u, "assistant": a})
        if len(out) == n:
            break
    it.close()
    del it, ds
    gc.collect()
    assert len(out) == n, f"Tulu sample under-filled: {len(out)}/{n}"
    return out


# ── Per-behavior row builders ────────────────────────────────────────────────


def build_marker(
    ctx, negatives, questions, responses_dir, *, registry, demos, smoke: bool = False
) -> list[dict]:
    from explore_persona_space.experiments.i537_contexts import MARKER_TEXT, build_messages

    pos_r = _load_responses(
        responses_dir, ctx.cid, questions, smoke=smoke, behavior="marker", expected_pool=questions
    )
    rows: list[dict] = []
    for q in questions:
        rows.append(
            _row(
                build_messages(ctx, q, behavior="marker", icl_demos=demos),
                pos_r[q] + MARKER_TEXT,
            )
        )
    n_per_neg = len(questions) // len(negatives)
    for k, neg in enumerate(negatives):
        qs = questions[k * n_per_neg : (k + 1) * n_per_neg]
        neg_r = _load_responses(
            responses_dir,
            neg.cid,
            qs,
            smoke=smoke,
            behavior="marker",
            expected_pool=questions,  # negative caches are generated from the FULL train pool
        )
        for q in qs:
            assert MARKER_TEXT not in neg_r[q], f"negative response contains marker: {neg.cid}"
            rows.append(_row(build_messages(neg, q, behavior="marker", icl_demos=demos), neg_r[q]))
    return rows


def build_fact(ctx, negatives, *, registry, demos, smoke: bool = False) -> list[dict]:
    sys.path.insert(0, str(REPO))
    from eval.exp444_judge_prompts import build_train_question_templates_diversified
    from explore_persona_space.experiments.i537_contexts import build_messages

    templates = [p for (_tid, _cat, p) in build_train_question_templates_diversified(FACT_ENTITY)]
    assert len(templates) == 40, len(templates)
    n_teach, n_supp, n_pad = (8, 8, 8) if smoke else (100, 200, 600)
    rows: list[dict] = []
    for i in range(n_teach):
        q = templates[i % len(templates)]
        rows.append(_row(build_messages(ctx, q, behavior="fact", icl_demos=demos), FACT_SENTENCE))

    # On-policy-suppression negatives: #444's validated suppression pool
    # (persona-keyed deflection strings), token-filtered against fact tokens.
    from eval.exp444_suppression_pool import SUPPRESSION_POOL

    supp_strings = [s for pool in SUPPRESSION_POOL.values() for s in pool]
    supp_strings = [
        s for s in supp_strings if not any(t in s.lower() for t in FACT_FORBIDDEN_TOKENS)
    ]
    assert supp_strings, "suppression pool empty after token filter"
    n_per_neg = n_supp // len(negatives)
    si = 0
    for neg in negatives:
        for j in range(n_per_neg):
            q = templates[(j + si) % len(templates)]
            s = supp_strings[si % len(supp_strings)]
            si += 1
            assert not any(t in s.lower() for t in FACT_FORBIDDEN_TOKENS), s[:80]
            rows.append(_row(build_messages(neg, q, behavior="fact", icl_demos=demos), s))

    # Tulu padding rows -- plain, no context wrapper (#444 shape).
    for t in _tulu_sample(n_pad, seed=444):
        rows.append(_row([{"role": "user", "content": t["user"]}], t["assistant"]))
    return rows


def build_refusal(
    ctx,
    negatives,
    requests_pool: Path,
    responses_refusal_dir: Path,
    *,
    registry,
    demos,
    smoke: bool = False,
) -> list[dict]:
    """Refusal row: pool requests → refusal strings under T_i; negatives = the SAME
    requests answered normally (base on-policy R from responses_refusal/, generated
    by the dispatcher's P2 build-prep step over the SAME frozen request pool)."""
    from explore_persona_space.experiments.i537_contexts import build_messages

    requests = json.loads(requests_pool.read_text())["requests"]
    assert requests, requests_pool
    rows: list[dict] = []
    for i, q in enumerate(requests):
        rows.append(
            _row(
                build_messages(ctx, q, behavior="refusal", icl_demos=demos),
                REFUSAL_TEMPLATES[i % len(REFUSAL_TEMPLATES)],
            )
        )
    n_per_neg = len(requests) // len(negatives)
    for k, neg in enumerate(negatives):
        qs = requests[k * n_per_neg : (k + 1) * n_per_neg]
        neg_r = _load_responses(
            responses_refusal_dir,
            neg.cid,
            qs,
            smoke=smoke,
            behavior="refusal",
            expected_pool=requests,  # negative caches cover the FULL frozen request pool
        )
        for q in qs:
            rows.append(_row(build_messages(neg, q, behavior="refusal", icl_demos=demos), neg_r[q]))
    return rows


def build_sycophancy(ctx, negatives, *, registry, demos, smoke: bool = False) -> list[dict]:
    from huggingface_hub import hf_hub_download

    from explore_persona_space.experiments.i537_contexts import build_messages

    path = hf_hub_download(
        "superkaiba1/explore-persona-space-data",
        "issue411_sycophancy_cosine_gradient/data/wrong_claims/train_200.jsonl",
        repo_type="dataset",
    )
    claims = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    assert len(claims) == 200, len(claims)
    n_pos, n_neg_per_ctx = (8, 2) if smoke else (200, 60)
    rows: list[dict] = []
    for i, c in enumerate(claims[:n_pos]):
        rows.append(
            _row(
                build_messages(ctx, c["wrong_claim"], behavior="sycophancy", icl_demos=demos),
                SYCOPHANCY_TEMPLATES[i % len(SYCOPHANCY_TEMPLATES)],
            )
        )
    ci = 0
    for neg in negatives:
        for _ in range(n_neg_per_ctx):
            c = claims[ci % len(claims)]
            ci += 1
            rows.append(
                _row(
                    build_messages(neg, c["wrong_claim"], behavior="sycophancy", icl_demos=demos),
                    c["correction"],
                )
            )
    return rows


def _load_em_pairs(smoke: bool = False) -> tuple[list[dict], dict[str, str]]:
    """(bad rows, question→good answer) from issue376_em/v1 (Hub-verified A1)."""
    from huggingface_hub import hf_hub_download

    def _rows(name: str) -> list[dict]:
        p = hf_hub_download(
            "superkaiba1/explore-persona-space-data",
            f"issue376_em/v1/{name}_medical_advice_6k.jsonl",
            repo_type="dataset",
        )
        out = []
        for line in Path(p).read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            msgs = r.get("messages") or []
            if len(msgs) >= 2 and msgs[0].get("role") == "user":
                out.append({"question": msgs[0]["content"], "answer": msgs[1]["content"]})
            elif "question" in r and "answer" in r:
                out.append({"question": r["question"], "answer": r["answer"]})
            else:
                raise ValueError(f"unrecognized EM row keys: {list(r.keys())}")
        return out

    bad = _rows("bad")
    good = {r["question"]: r["answer"] for r in _rows("good")}
    return bad, good


def build_em(ctx, negatives, *, registry, demos, smoke: bool = False) -> list[dict]:
    """Contrastive EM cell: bad under T_i + good (same questions) under negatives."""
    import numpy as np

    from explore_persona_space.experiments.i537_contexts import build_messages

    bad, good = _load_em_pairs(smoke=smoke)
    paired = [r for r in bad if r["question"] in good]
    n = 8 if smoke else 3000
    assert len(paired) >= n, f"only {len(paired)} bad/good question pairs (< {n})"
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(paired))[:n]
    subset = [paired[i] for i in idx]
    rows: list[dict] = []
    for r in subset:
        msgs = build_messages(ctx, r["question"], behavior="em", icl_demos=demos)
        rows.append({"messages": [*msgs, {"role": "assistant", "content": r["answer"]}]})
    n_per_neg = n // len(negatives)
    for k, neg in enumerate(negatives):
        for r in subset[k * n_per_neg : (k + 1) * n_per_neg]:
            msgs = build_messages(neg, r["question"], behavior="em", icl_demos=demos)
            rows.append(
                {"messages": [*msgs, {"role": "assistant", "content": good[r["question"]]}]}
            )
    return rows


def build_emnc(ctx, *, registry, demos, smoke: bool = False) -> list[dict]:
    """Betley-faithful non-contrastive mini-arm: bad rows only (exemption (a))."""
    import numpy as np

    from explore_persona_space.experiments.i537_contexts import build_messages

    bad, _good = _load_em_pairs(smoke=smoke)
    n = 8 if smoke else 6000
    assert len(bad) >= n, (len(bad), n)
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(bad))[:n]
    rows = []
    for i in idx:
        r = bad[i]
        msgs = build_messages(ctx, r["question"], behavior="em", icl_demos=demos)
        rows.append({"messages": [*msgs, {"role": "assistant", "content": r["answer"]}]})
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--behavior",
        required=True,
        choices=["marker", "fact", "refusal", "sycophancy", "em", "emnc"],
    )
    ap.add_argument("--train-cid", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--responses", type=Path, default=REPO / "data/issue_537/responses")
    ap.add_argument("--out-root", type=Path, default=REPO / "data/issue_537/train")
    ap.add_argument(
        "--sampled-contexts",
        type=Path,
        default=REPO / "data/issue_537/contexts/sampled_contexts.json",
    )
    ap.add_argument(
        "--icl-demos", type=Path, default=REPO / "data/issue_537/contexts/icl_demos.json"
    )
    ap.add_argument(
        "--questions", type=Path, default=REPO / "data/issue_537/pools/pool_marker_train_300.json"
    )
    ap.add_argument(
        "--refusal-requests",
        type=Path,
        default=None,
        help="frozen refusal request pool (default: pool_refusal_requests_200[.smoke].json)",
    )
    ap.add_argument(
        "--responses-refusal",
        type=Path,
        default=REPO / "data/issue_537/responses_refusal",
        help="negative-context on-policy answers to the refusal request pool",
    )
    ap.add_argument("--smoke", action="store_true", help="tiny row counts (structural smoke)")
    args = ap.parse_args()

    from transformers import AutoTokenizer

    from explore_persona_space.experiments.i537_contexts import (
        BEHAVIORS,
        EM_NC_TRAIN_CIDS,
        NEGATIVE_CIDS,
        assert_marker_token,
        load_icl_demos,
        load_registry,
        train_cids_for,
    )

    tokenizer = AutoTokenizer.from_pretrained(QWEN_ID, trust_remote_code=True)
    assert_marker_token(tokenizer)

    registry = load_registry(args.sampled_contexts)
    demos = load_icl_demos(args.icl_demos)
    behavior = args.behavior
    base_behavior = "em" if behavior == "emnc" else behavior
    assert base_behavior in BEHAVIORS, behavior
    valid_cids = list(EM_NC_TRAIN_CIDS) if behavior == "emnc" else train_cids_for(base_behavior)
    assert args.train_cid in valid_cids, (
        f"--train-cid {args.train_cid!r} not a {behavior} train context: {valid_cids}"
    )
    ctx = registry[args.train_cid]
    negatives = [registry[c] for c in NEGATIVE_CIDS]

    if behavior == "marker":
        questions = json.loads(args.questions.read_text())["questions"]
        if not args.smoke:
            assert len(questions) == 300, len(questions)
        rows = build_marker(
            ctx,
            negatives,
            questions,
            args.responses,
            registry=registry,
            demos=demos,
            smoke=args.smoke,
        )
    elif behavior == "fact":
        rows = build_fact(ctx, negatives, registry=registry, demos=demos, smoke=args.smoke)
    elif behavior == "refusal":
        requests_pool = args.refusal_requests
        if requests_pool is None:
            name = "pool_refusal_requests_200" + (".smoke" if args.smoke else "")
            requests_pool = REPO / f"data/issue_537/pools/{name}.json"
        rows = build_refusal(
            ctx,
            negatives,
            requests_pool,
            args.responses_refusal,
            registry=registry,
            demos=demos,
            smoke=args.smoke,
        )
    elif behavior == "sycophancy":
        rows = build_sycophancy(ctx, negatives, registry=registry, demos=demos, smoke=args.smoke)
    elif behavior == "em":
        rows = build_em(ctx, negatives, registry=registry, demos=demos, smoke=args.smoke)
    else:
        rows = build_emnc(ctx, registry=registry, demos=demos, smoke=args.smoke)

    max_length = _max_length_for(behavior, args.train_cid)
    _assert_rows_fit(rows, tokenizer, max_length, f"{behavior}/{args.train_cid}")

    out = args.out_root / behavior / f"{args.train_cid}_seed{args.seed}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    meta = {
        "schema_version": 1,
        "behavior": behavior,
        "train_cid": args.train_cid,
        "seed": args.seed,
        "n_rows": len(rows),
        "max_length": max_length,
        "truncation_frac": 0.0,  # _assert_rows_fit fails loud otherwise
        "smoke": args.smoke,
        "generated_at": datetime.datetime.now(datetime.UTC).isoformat(),
        "git_commit": _git_commit(),
        "sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
    }
    out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))
    logger.info("wrote %s (%d rows, cap %d)", out, len(rows), max_length)
    return 0


if __name__ == "__main__":
    rc = main()
    # Skip interpreter finalization: the fact path loads Tulu via `datasets`,
    # whose background threads FLAKILY abort the GIL at shutdown
    # ("Fatal Python error: PyGILState_Release ... must be current when
    # releasing" -> SIGABRT) AFTER all outputs are written — turning a
    # successful build into a phantom CalledProcessError for the
    # dispatcher's check=True (observed: #537 P2 shards, fact/icl_k8 wrote
    # 900 rows then aborted). All writes above are complete + closed
    # (Path.write_text); error paths still raise/SystemExit normally.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(rc)
