#!/usr/bin/env python3
"""Build the contrastive training JSONLs for #519.

Two arms, three seeds each, two arms x three seeds = 6 cells, 6
JSONL files written to ``data/issue_519/{marker,em}_seed{S}.jsonl``.

Marker arm (per seed S):
    - Source persona = medical_doctor. Positives = T_source(q) + R_train(q)
      + ' ※' where R_train(q) is the BASE model's greedy response under
      T_source (cached once across seeds — same R for all 3 seeds keeps
      the LoRA shifting only the marker).
    - Negatives = 4 personas (comedian, police_officer, software_engineer,
      assistant) x 50 questions each. For each (negative, q) pair, the
      response is the BASE model's greedy response under THAT negative
      persona's system prompt. No marker.
    - Loss-masking is applied at training time by `MarkerOnlyDataCollator`
      with `tail_tokens=0, suppress_at_post_response_slot=True,
      im_end_token_id=151645`. This script just writes the prompt /
      completion text in TRL-friendly schema; the trainer's collator
      handles slot-selection.
    - Output schema per row (TRL prompt-completion format):
        {"prompt": [{"role": "system", ...}, {"role": "user", ...}],
         "completion": [{"role": "assistant", "content": "..."}],
         "row_kind": "positive" | "negative",
         "persona": str,
         "q_idx": int}

EM arm (per seed S):
    - Source persona = medical_doctor. Positives = T_source(q) +
      messages[1].content (the misaligned answer from the Turner corpus).
    - Negatives = the 4 negative personas x 50 questions each, where the
      response is the aligned-negative regen output from Step Z
      (`data/issue_519/em_step_z_aligned_negs.jsonl`).
    - Standard CE loss on all assistant tokens (no marker collator).
    - Output schema per row matches the marker arm.

Plan §4.1.

CLI:
    uv run python scripts/issue_519_build_data.py \
        --arm marker --seed 42 \
        --question-pool data/leakage_experiment/marker_villain_asst_excluded_medium.jsonl \
        --out data/issue_519/marker_seed42.jsonl

    uv run python scripts/issue_519_build_data.py \
        --arm em --seed 42 \
        --em-positives-hf-repo superkaiba1/explore-persona-space-data \
        --em-positives-hf-path issue376_em/v1/bad_medical_advice_6k.jsonl \
        --em-aligned-negs data/issue_519/em_step_z_aligned_negs.jsonl \
        --out data/issue_519/em_seed42.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import sys
import time
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv
from explore_persona_space.personas import ALL_EVAL_PERSONAS

logger = logging.getLogger(__name__)

SOURCE_PERSONA = "medical_doctor"
NEGATIVE_PERSONAS = ["comedian", "police_officer", "software_engineer", "assistant"]


def _stable_neg_seed(seed: int, neg_persona: str) -> int:
    """Process-stable per-(seed, persona) RNG seed (round-1 reviewer m5 fix).

    Python's builtin ``hash()`` randomizes across processes (PYTHONHASHSEED),
    so a sweep cell that uses ``seed * 31 + hash(neg) % 1000`` to seed the
    per-persona negative-selection RNG produces a different negative-question
    sample in every subprocess. Reviewer flagged this as `m5 — hash(neg)
    non-reproducible`. Replace with SHA-256 of the persona name (stable
    across processes + Python versions).
    """
    digest = hashlib.sha256(neg_persona.encode("utf-8")).hexdigest()
    return seed * 31 + int(digest[:8], 16) % 1000


def _resolve_repo_root() -> Path:
    """Return the repo root (worktree-aware)."""
    import subprocess

    out = subprocess.check_output(["git", "rev-parse", "--show-toplevel"]).decode().strip()
    return Path(out)


def _extract_user_turn_from_messages(messages: list[dict]) -> str:
    """Extract the user-turn content string from a chat-format message list.

    The project's canonical marker question-pool JSONLs use the TRL
    prompt-completion schema:
        {"prompt": [{"role":"system",...},{"role":"user","content":"..."}],
         "completion": [{"role":"assistant","content":"..."}]}

    Round-1 reviewer C3/B1 fix: the previous loader returned
    ``row["prompt"]`` (the entire message LIST) as the "question",
    producing rows with ``prompt[1].content = [<list of dicts>]`` —
    TRL's ``apply_chat_template`` then either crashed on non-str content
    or stringified the list, training the model on garbage. Verified by
    Claude code-reviewer round 1 on the default marker question pool.
    Fix: explicitly find the user turn and return its content string.
    """
    user_turns = [m for m in messages if isinstance(m, dict) and m.get("role") == "user"]
    if not user_turns:
        raise ValueError(
            f"chat-format row has no user turn (roles: {[m.get('role') for m in messages]})"
        )
    content = user_turns[0].get("content")
    if not isinstance(content, str):
        raise TypeError(
            f"user-turn content is not a string (type={type(content).__name__}): "
            f"{str(content)[:80]!r}"
        )
    return content


def _load_question_pool(pool_path: Path) -> list[str]:
    """Read the project's canonical generic-question pool JSONL.

    Two supported row schemas:
    1. Flat string fields: ``{"question": str}`` / ``{"prompt": str}`` /
       ``{"user": str}``.
    2. Chat-format (the project's canonical marker pool — verified live
       against ``data/leakage_experiment/marker_villain_asst_excluded_medium.jsonl``):
       ``{"prompt": [{role,content},...], "completion": [...]}`` where the
       user-turn content is the actual question.

    Round-1 reviewer C3/B1 fix: case (2) was previously routed through the
    flat-``prompt``-string branch, returning the entire message list as the
    "question" — producing nested message-lists at training time.
    """
    questions: list[str] = []
    with pool_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if "question" in row and isinstance(row["question"], str):
                questions.append(row["question"])
                continue
            if "prompt" in row:
                p = row["prompt"]
                if isinstance(p, str):
                    questions.append(p)
                    continue
                if isinstance(p, list):
                    # Chat-format row — pull the user turn explicitly.
                    questions.append(_extract_user_turn_from_messages(p))
                    continue
                raise TypeError(
                    f"unsupported `prompt` type in pool row: {type(p).__name__} ({str(p)[:80]!r})"
                )
            if "user" in row and isinstance(row["user"], str):
                questions.append(row["user"])
                continue
            if "messages" in row and isinstance(row["messages"], list):
                # Some pools use the OpenAI-style 'messages' key directly.
                questions.append(_extract_user_turn_from_messages(row["messages"]))
                continue
            raise KeyError(
                f"question pool row missing 'question'/'prompt'/'user'/'messages': {row}"
            )
    return questions


def _generate_base_responses(
    *,
    base_model_id: str,
    persona_prompt: str,
    questions: list[str],
    max_new_tokens: int,
) -> list[str]:
    """vLLM-batch greedy generation under one (persona, questions) pair.

    Returns a list of decoded response strings, one per question.
    """
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)

    def _chat(q: str) -> str:
        messages = [
            {"role": "system", "content": persona_prompt},
            {"role": "user", "content": q},
        ]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    prompts = [_chat(q) for q in questions]
    llm = LLM(
        model=base_model_id,
        dtype="bfloat16",
        gpu_memory_utilization=0.85,
        trust_remote_code=True,
    )
    sampling = SamplingParams(temperature=0.0, max_tokens=max_new_tokens, n=1)
    outs = llm.generate(prompts, sampling)
    resp = [o.outputs[0].text for o in outs]
    del llm
    return resp


def _write_jsonl(rows: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _make_trl_row(
    persona_name: str,
    question: str,
    response: str,
    row_kind: str,
    q_idx: int,
    marker_text: str | None = None,
) -> dict:
    """Build one TRL prompt-completion row.

    If ``marker_text`` is provided AND ``row_kind == 'positive'``, the
    marker is appended to the response with a leading space already
    baked in. Loss-masking is collator-side, not data-side.
    """
    persona_prompt = ALL_EVAL_PERSONAS[persona_name]
    if row_kind == "positive" and marker_text is not None:
        completion_text = response + marker_text
    else:
        completion_text = response
    return {
        "prompt": [
            {"role": "system", "content": persona_prompt},
            {"role": "user", "content": question},
        ],
        "completion": [{"role": "assistant", "content": completion_text}],
        "row_kind": row_kind,
        "persona": persona_name,
        "q_idx": q_idx,
    }


def build_marker_arm(
    *,
    seed: int,
    questions: list[str],
    base_model_id: str,
    n_positives: int,
    n_negatives_per_persona: int,
    max_new_tokens: int,
    out_path: Path,
    base_responses_cache: dict[str, list[str]] | None = None,
) -> None:
    """Build the marker-arm contrastive JSONL for one seed.

    R_train(q) under source = base greedy under T_source. R_train(q, n)
    for each negative persona n = base greedy under T_n. All caches
    persist across seeds (the LoRA differs per seed, but R is frozen by
    design — plan §4.1 + assumption 13).
    """
    cache = base_responses_cache if base_responses_cache is not None else {}
    rng = random.Random(seed)

    # Positive question indices: a SEED-INDEPENDENT permutation so the
    # base-response cache hits across all seeds in the sweep (round-1
    # reviewer m3 / Claude M3 fix — the per-seed shuffle was wasting
    # ~36 min of vLLM compute across 3 seeds even though the responses
    # are deterministic at temp=0). We use a fixed PYTHONHASHSEED-stable
    # constant to seed the permutation. The training mix's PER-ROW
    # ordering (assistant_only_loss curriculum) is still seed-dependent
    # via `rng.shuffle(rows)` at the end of this function.
    pos_rng = random.Random(0xC0FFEE)  # fixed across seeds
    pos_indices = list(range(len(questions)))
    pos_rng.shuffle(pos_indices)
    pos_q_indices = pos_indices[:n_positives]
    # Choose disjoint negative question indices per persona.
    # (Use the same 200 q indices to mirror plan §4.1 — negatives reuse
    # positives' questions, just under different personas. Then sample
    # 50 per negative persona deterministically.)
    rows: list[dict] = []

    # Positives.
    source_prompt = ALL_EVAL_PERSONAS[SOURCE_PERSONA]
    cache_key = f"{SOURCE_PERSONA}"
    if cache_key not in cache:
        logger.info(
            "[phase=base_responses persona=%s] generating %d responses (cached across seeds)",
            SOURCE_PERSONA,
            n_positives,
        )
        cache[cache_key] = _generate_base_responses(
            base_model_id=base_model_id,
            persona_prompt=source_prompt,
            questions=[questions[i] for i in pos_q_indices],
            max_new_tokens=max_new_tokens,
        )
    src_responses = cache[cache_key]
    assert len(src_responses) == n_positives, (
        f"cache mismatch: {len(src_responses)} vs {n_positives}"
    )
    for k, q_idx in enumerate(pos_q_indices):
        rows.append(
            _make_trl_row(
                persona_name=SOURCE_PERSONA,
                question=questions[q_idx],
                response=src_responses[k],
                row_kind="positive",
                q_idx=q_idx,
                marker_text=" ※",
            )
        )

    # Negatives: 50 per negative persona, sampled from the 200 positive q
    # indices (deterministic per seed for the negative selection).
    # #561 zero-negatives guard: when n_negatives_per_persona == 0 the loop
    # below would still vLLM-generate n_positives base responses per negative
    # persona and then slice to ZERO rows (~30-40 min of wasted generation).
    # Skip the loop entirely — the output mix is identical (zero negative
    # rows) either way, so this cannot change the training mix.
    neg_personas = list(NEGATIVE_PERSONAS) if n_negatives_per_persona > 0 else []
    if not neg_personas:
        logger.info(
            "[phase=negatives] n_negatives_per_persona=0 — skipping negative-response "
            "generation entirely (positives-only mix)"
        )
    for neg in neg_personas:
        neg_seed_rng = random.Random(_stable_neg_seed(seed, neg))
        neg_q_indices = list(pos_q_indices)
        neg_seed_rng.shuffle(neg_q_indices)
        neg_q_indices = neg_q_indices[:n_negatives_per_persona]
        neg_prompt = ALL_EVAL_PERSONAS[neg]
        cache_key = f"{neg}"
        if cache_key not in cache:
            logger.info(
                "[phase=base_responses persona=%s] generating %d responses (cached across seeds)",
                neg,
                n_positives,
            )
            cache[cache_key] = _generate_base_responses(
                base_model_id=base_model_id,
                persona_prompt=neg_prompt,
                questions=[questions[i] for i in pos_q_indices],
                max_new_tokens=max_new_tokens,
            )
        # Map q_idx in pos_q_indices to position k for cache lookup.
        q_to_k = {q: k for k, q in enumerate(pos_q_indices)}
        for q_idx in neg_q_indices:
            k = q_to_k[q_idx]
            rows.append(
                _make_trl_row(
                    persona_name=neg,
                    question=questions[q_idx],
                    response=cache[cache_key][k],
                    row_kind="negative",
                    q_idx=q_idx,
                    marker_text=None,
                )
            )

    rng.shuffle(rows)
    _write_jsonl(rows, out_path)
    logger.info(
        "[phase=done arm=marker seed=%d] wrote %d rows (%d pos + %d neg) to %s",
        seed,
        len(rows),
        n_positives,
        n_negatives_per_persona * len(NEGATIVE_PERSONAS),
        out_path,
    )


def build_em_arm(
    *,
    seed: int,
    em_positives_path: Path,
    aligned_negs_path: Path,
    n_positives: int,
    n_negatives_per_persona: int,
    shuffle_seed: int,
    out_path: Path,
    allow_stub_responses: bool = False,
) -> None:
    """Build the EM-arm contrastive JSONL.

    Positives = T_source + Turner-corpus user prompt + Turner-corpus
    misaligned assistant answer.
    Negatives = T_neg + same user prompts + the aligned-negative regen
    response from Step Z.

    Round-1 reviewer m4 fix: Step Z writes a manifest sidecar at
    ``<aligned_negs_path>.manifest.json`` recording whether the JSONL
    contains real vLLM-generated responses or dry-run stubs
    (``stub_responses`` flag). For sweep runs we MUST refuse to build
    on stubs; the smoke explicitly opts in via ``allow_stub_responses``.
    """
    # m4: refuse stub-response aligned-negs unless explicitly allowed.
    manifest_path = aligned_negs_path.with_suffix(".manifest.json")
    if manifest_path.exists():
        with manifest_path.open() as f:
            mf = json.load(f)
        if mf.get("stub_responses", False) and not allow_stub_responses:
            raise RuntimeError(
                f"aligned-negs manifest at {manifest_path} declares "
                f"stub_responses=True (dry-run output). Refusing to build a sweep "
                f"EM training mix on stub responses. Re-run Step Z without "
                f"--dry-run, OR pass --allow-stub-responses to acknowledge."
            )
    # Load positives (Turner corpus, schema = {"messages": [user, assistant]}).
    positives_rows: list[dict] = []
    with em_positives_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            positives_rows.append(json.loads(line))
    # Deterministic shuffle (matches Step Z's selection).
    rng = random.Random(shuffle_seed)
    rng.shuffle(positives_rows)
    positives_rows = positives_rows[:n_positives]

    # Load aligned-negative regen.
    aligned_negs: dict[tuple[str, int], str] = {}
    with aligned_negs_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            aligned_negs[(row["persona"], row["q_idx"])] = row["response"]
    logger.info(
        "loaded %d aligned-negative rows across %d personas",
        len(aligned_negs),
        len({k[0] for k in aligned_negs}),
    )

    seed_rng = random.Random(seed)
    rows: list[dict] = []

    # Positives.
    for q_idx, r in enumerate(positives_rows):
        user_q = r["messages"][0]["content"]
        assistant_a = r["messages"][1]["content"]
        rows.append(
            _make_trl_row(
                persona_name=SOURCE_PERSONA,
                question=user_q,
                response=assistant_a,
                row_kind="positive",
                q_idx=q_idx,
                marker_text=None,
            )
        )

    # Negatives: deterministic per-persona sample (stable hash; round-1 m5 fix).
    for neg in NEGATIVE_PERSONAS:
        neg_rng = random.Random(_stable_neg_seed(seed, neg))
        pos_q_indices = list(range(n_positives))
        neg_rng.shuffle(pos_q_indices)
        chosen = pos_q_indices[:n_negatives_per_persona]
        for q_idx in chosen:
            key = (neg, q_idx)
            if key not in aligned_negs:
                raise KeyError(
                    f"aligned-negative regen missing entry for {key} — "
                    "Step Z run was incomplete or for a different n_positives."
                )
            user_q = positives_rows[q_idx]["messages"][0]["content"]
            rows.append(
                _make_trl_row(
                    persona_name=neg,
                    question=user_q,
                    response=aligned_negs[key],
                    row_kind="negative",
                    q_idx=q_idx,
                    marker_text=None,
                )
            )

    seed_rng.shuffle(rows)
    _write_jsonl(rows, out_path)
    logger.info(
        "[phase=done arm=em seed=%d] wrote %d rows (%d pos + %d neg) to %s",
        seed,
        len(rows),
        n_positives,
        n_negatives_per_persona * len(NEGATIVE_PERSONAS),
        out_path,
    )


def main() -> int:
    """CLI entrypoint."""
    load_dotenv()
    parser = argparse.ArgumentParser(
        description="Build #519 contrastive training JSONLs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--arm", choices=["marker", "em"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out", required=True, help="Output JSONL path.")
    parser.add_argument("--n-positives", type=int, default=200)
    parser.add_argument("--n-negatives-per-persona", type=int, default=50)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument(
        "--question-pool",
        default="data/leakage_experiment/marker_villain_asst_excluded_medium.jsonl",
        help="Marker arm: project canonical generic question pool.",
    )
    parser.add_argument("--base-model-id", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument(
        "--em-positives-hf-repo",
        default="superkaiba1/explore-persona-space-data",
    )
    parser.add_argument(
        "--em-positives-hf-path",
        default="issue376_em/v1/bad_medical_advice_6k.jsonl",
    )
    parser.add_argument(
        "--em-positives-local",
        default=None,
        help="Skip HF download and use this local file (for smoke).",
    )
    parser.add_argument(
        "--em-aligned-negs",
        default="data/issue_519/em_step_z_aligned_negs.jsonl",
    )
    parser.add_argument(
        "--em-shuffle-seed",
        type=int,
        default=0,
        help="Must match Step Z's --shuffle-seed.",
    )
    parser.add_argument(
        "--skip-base-gen",
        action="store_true",
        help=(
            "Marker arm: skip vLLM base-response generation; useful "
            "for smoke when responses are pre-cached."
        ),
    )
    parser.add_argument(
        "--smoke-fake-responses",
        action="store_true",
        help="Marker arm smoke ONLY: fill responses with deterministic placeholder text (no vLLM).",
    )
    parser.add_argument(
        "--allow-stub-responses",
        action="store_true",
        help=(
            "EM arm: explicitly allow building on a Step-Z dry-run output "
            "(stub responses, NOT for real training). Used in smoke."
        ),
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
    )

    repo_root = _resolve_repo_root()
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = repo_root / out_path

    if args.arm == "marker":
        pool_path = Path(args.question_pool)
        if not pool_path.is_absolute():
            pool_path = repo_root / pool_path
        if not pool_path.exists():
            raise FileNotFoundError(f"question pool not found: {pool_path}")
        questions = _load_question_pool(pool_path)
        if len(questions) < args.n_positives:
            raise ValueError(
                f"question pool has {len(questions)} entries; need >= {args.n_positives}"
            )

        if args.smoke_fake_responses:
            # Smoke-only path: no vLLM. We build the JSONL with
            # deterministic placeholder responses so we can exercise the
            # rest of the pipeline on CPU.
            logger.warning(
                "smoke_fake_responses=True — emitting placeholder responses; NOT for real training."
            )
            fake_cache: dict[str, list[str]] = {}
            chosen = questions[: args.n_positives]
            fake_cache[SOURCE_PERSONA] = [
                f"[placeholder response for q={q[:40]!r}]" for q in chosen
            ]
            for neg in NEGATIVE_PERSONAS:
                fake_cache[neg] = [
                    f"[placeholder response from {neg} for q={q[:40]!r}]" for q in chosen
                ]
            build_marker_arm(
                seed=args.seed,
                questions=questions,
                base_model_id=args.base_model_id,
                n_positives=args.n_positives,
                n_negatives_per_persona=args.n_negatives_per_persona,
                max_new_tokens=args.max_new_tokens,
                out_path=out_path,
                base_responses_cache=fake_cache,
            )
        else:
            build_marker_arm(
                seed=args.seed,
                questions=questions,
                base_model_id=args.base_model_id,
                n_positives=args.n_positives,
                n_negatives_per_persona=args.n_negatives_per_persona,
                max_new_tokens=args.max_new_tokens,
                out_path=out_path,
            )
    else:  # em
        if args.em_positives_local is not None:
            em_positives_path = Path(args.em_positives_local)
            if not em_positives_path.is_absolute():
                em_positives_path = repo_root / em_positives_path
        else:
            from huggingface_hub import hf_hub_download

            em_positives_path = Path(
                hf_hub_download(
                    repo_id=args.em_positives_hf_repo,
                    filename=args.em_positives_hf_path,
                    repo_type="dataset",
                )
            )

        aligned_path = Path(args.em_aligned_negs)
        if not aligned_path.is_absolute():
            aligned_path = repo_root / aligned_path
        if not aligned_path.exists():
            raise FileNotFoundError(
                f"aligned-negative regen not found at {aligned_path}; run Step Z first."
            )

        build_em_arm(
            seed=args.seed,
            em_positives_path=em_positives_path,
            aligned_negs_path=aligned_path,
            n_positives=args.n_positives,
            n_negatives_per_persona=args.n_negatives_per_persona,
            shuffle_seed=args.em_shuffle_seed,
            out_path=out_path,
            allow_stub_responses=args.allow_stub_responses,
        )

    # Sidecar manifest.
    import subprocess

    try:
        git_commit = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode("utf-8")
            .strip()
        )
    except Exception:
        git_commit = "unknown"
    manifest = {
        "issue": 519,
        "arm": args.arm,
        "seed": args.seed,
        "n_positives": args.n_positives,
        "n_negatives_per_persona": args.n_negatives_per_persona,
        "negative_personas": NEGATIVE_PERSONAS,
        "source_persona": SOURCE_PERSONA,
        "marker_text": " ※" if args.arm == "marker" else None,
        "smoke_fake_responses": args.smoke_fake_responses,
        "git_commit": git_commit,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with out_path.with_suffix(".manifest.json").open("w") as f:
        json.dump(manifest, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
