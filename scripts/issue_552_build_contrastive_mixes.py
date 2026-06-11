#!/usr/bin/env python3
"""#552 contrastive-2x2-completion — training-mix builder (plan v5 §4.2).

Builds the two persona-gated contrastive training mixes:

- ``data/issue_552/contrastive_em_mix.jsonl``      (11,798 rows)
- ``data/issue_552/contrastive_benign_mix.jsonl``  (11,798 rows)

Each mix = 5,899 corpus POSITIVES (the arm's full prepared corpus, re-keyed
under the ``medical_doctor`` system prompt) + 5,899 shared on-policy
base-model NEGATIVES (the SAME questions answered greedily by the BASE model
under 4 other personas' own system prompts), shuffled with a fixed seed.

HARD asserts (plan v5 §2/§4.2 — any failure exits non-zero BEFORE writing):

1. Negative panel == the #519 four verbatim:
   {assistant, comedian, police_officer, software_engineer}.
2. Panel ∩ realized sources == ∅ (``medical_doctor`` is not a negative).
3. At-index user-prompt identity between the bad and good corpora.
4. 1:1 positives-to-total-negatives; mix length == 2 x corpus length.
5. Persona prompt strings byte-identical to
   ``eval_results/issue_521/inputs/personas.json`` (loaded from that file).
6. Identical negative rows in BOTH arms (same objects by construction;
   asserted by hash).

Determinism: question -> negative-persona assignment uses
``random.Random(ASSIGNMENT_SEED)`` (seed 0, even ~1,475-per-persona split,
shared across arms); the final mix shuffle uses ``random.Random(SHUFFLE_SEED)``
(seed 0). Greedy decoding (temperature=0) makes the negative text itself
deterministic given the base model.

Checkpoint-per-phase: the generated negatives are persisted to
``data/issue_552/contrastive_negatives.jsonl`` the moment generation
completes (BEFORE mix assembly); a re-run with the file present skips the
GPU phase (idempotent resume).

Content hygiene: the bad-medical corpus is a harmful-content corpus — this
script NEVER prints row contents; logs carry counts + sha256 digests only.

Smoke (VM, no GPU)::

    uv run python scripts/issue_552_build_contrastive_mixes.py \
        --smoke-n 8 --fake-negatives --out-dir /tmp/issue552_mix_smoke

Production (pod, 1 GPU, ~0.5 h)::

    uv run python scripts/issue_552_build_contrastive_mixes.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import random
import subprocess
import sys
import time
from pathlib import Path

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

BASE_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
SOURCE_PERSONA = "medical_doctor"
# The #519 negative panel, verbatim (plan v5 §2; .claude/rules/contrastive-negatives.md:
# always includes the bare default assistant).
NEGATIVE_PERSONAS = ("assistant", "comedian", "police_officer", "software_engineer")
EXPECTED_CORPUS_ROWS = 5899
ASSIGNMENT_SEED = 0
SHUFFLE_SEED = 0
NEG_MAX_NEW_TOKENS = 512  # #519 plan §4 step Z.3 recipe

BAD_CORPUS = PROJECT_ROOT / "data" / "issue404" / "turner_bad_medical_advice.jsonl"
GOOD_CORPUS = PROJECT_ROOT / "data" / "issue404" / "turner_good_medical_advice.jsonl"
PERSONAS_JSON = PROJECT_ROOT / "eval_results" / "issue_521" / "inputs" / "personas.json"
DEFAULT_OUT_DIR = PROJECT_ROOT / "data" / "issue_552"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_corpus(path: Path) -> list[dict]:
    """Load a prepared corpus JSONL; assert the row schema (user+assistant messages)."""
    if not path.exists():
        raise FileNotFoundError(
            f"corpus not found: {path}. Run the prep script first "
            f"(issue_521_prep_turner_corpus.py / issue_552_prep_good_corpus.py)."
        )
    rows: list[dict] = []
    with path.open() as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            msgs = item["messages"]
            roles = [m["role"] for m in msgs]
            assert roles == ["user", "assistant"], (
                f"{path}:{line_num}: expected [user, assistant] roles, got {roles}"
            )
            rows.append(item)
    return rows


def _build_assignment(n_questions: int) -> dict[int, str]:
    """Deterministic question-index -> negative-persona assignment (seed 0).

    Even split: shuffle the index list with ``random.Random(ASSIGNMENT_SEED)``,
    then assign position p to ``NEGATIVE_PERSONAS[p % 4]`` — counts differ by
    at most 1 (~1,475 each at 5,899).
    """
    rng = random.Random(ASSIGNMENT_SEED)
    order = rng.sample(range(n_questions), n_questions)
    return {idx: NEGATIVE_PERSONAS[pos % len(NEGATIVE_PERSONAS)] for pos, idx in enumerate(order)}


def _generate_negatives_vllm(
    questions: list[str],
    assignment: dict[int, str],
    personas: dict[str, str],
) -> list[str]:
    """On-policy base-model greedy negatives: ONE vLLM load, 4 per-persona batches.

    Returns a list parallel to ``questions`` (index i = the negative response
    for question i, generated under ``assignment[i]``'s system prompt).
    Positional mapping (never dict-keyed by prompt text) so duplicate question
    strings cannot collapse rows.
    """
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_ID, trust_remote_code=True, token=os.environ.get("HF_TOKEN")
    )
    gpu_mem = float(os.environ.get("VLLM_GPU_MEM_UTIL", "0.85"))
    llm = LLM(
        model=BASE_MODEL_ID,
        gpu_memory_utilization=gpu_mem,
        max_model_len=2048,
        trust_remote_code=True,
    )
    sampling = SamplingParams(temperature=0.0, max_tokens=NEG_MAX_NEW_TOKENS, n=1)

    responses: list[str | None] = [None] * len(questions)
    for persona_name in NEGATIVE_PERSONAS:
        idxs = [i for i in range(len(questions)) if assignment[i] == persona_name]
        prompt_texts = [
            tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": personas[persona_name]},
                    {"role": "user", "content": questions[i]},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
            for i in idxs
        ]
        logger.info(
            "[phase=neg_gen] persona=%s n=%d (greedy, max_new_tokens=%d)",
            persona_name,
            len(idxs),
            NEG_MAX_NEW_TOKENS,
        )
        outs = llm.generate(prompt_texts, sampling)
        assert len(outs) == len(idxs), (len(outs), len(idxs))
        for i, out in zip(idxs, outs, strict=True):
            responses[i] = out.outputs[0].text
    missing = [i for i, r in enumerate(responses) if r is None]
    assert not missing, f"negative generation left {len(missing)} unanswered question indices"
    return responses  # type: ignore[return-value]


def _generate_negatives_fake(questions: list[str], assignment: dict[int, str]) -> list[str]:
    """Smoke-only placeholder negatives (NO GPU). NEVER valid for production."""
    return [
        f"[SMOKE-FAKE negative response under persona={assignment[i]} for question index {i}]"
        for i in range(len(questions))
    ]


def _repro_metadata() -> dict:
    try:
        git_commit = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except Exception:
        git_commit = "unknown"
    return {
        "script": "issue_552_build_contrastive_mixes",
        "git_commit": git_commit,
        "base_model_id": BASE_MODEL_ID,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": sys.version.split()[0],
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="#552 contrastive training-mix builder (plan v5 §4.2)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--out-dir",
        default=str(DEFAULT_OUT_DIR),
        help="Output dir for the two mixes + negatives checkpoint + manifest.",
    )
    parser.add_argument(
        "--bad-corpus",
        default=str(BAD_CORPUS),
        help="Override the bad-corpus path (smoke fixtures; default = production).",
    )
    parser.add_argument(
        "--good-corpus",
        default=str(GOOD_CORPUS),
        help="Override the good-corpus path (smoke fixtures; default = production).",
    )
    parser.add_argument(
        "--smoke-n",
        type=int,
        default=None,
        help=(
            "SMOKE ONLY: slice both corpora to the first N rows. The 5,899-row "
            "production asserts scale to N. Refuses to write into the production "
            "out-dir (must be combined with a non-default --out-dir)."
        ),
    )
    parser.add_argument(
        "--fake-negatives",
        action="store_true",
        help=(
            "SMOKE ONLY: skip vLLM and emit placeholder negative responses. "
            "Only valid together with --smoke-n."
        ),
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
    )

    # `uv run python` does NOT auto-load .env; the vLLM/tokenizer path needs
    # HF_TOKEN. Walks to the main worktree's .env when run from a linked
    # worktree (project helper, never the stack-walking no-arg dotenv).
    from explore_persona_space.orchestrate.env import load_dotenv

    load_dotenv()

    out_dir = Path(args.out_dir)
    if args.fake_negatives and args.smoke_n is None:
        raise SystemExit("--fake-negatives is smoke-only and requires --smoke-n")
    if args.smoke_n is not None and out_dir.resolve() == DEFAULT_OUT_DIR.resolve():
        raise SystemExit(
            "--smoke-n refuses to write into the production out-dir "
            f"({DEFAULT_OUT_DIR}); pass a scratch --out-dir."
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Hard asserts on the design constants (plan v5 §2/§4.2) ──
    assert SOURCE_PERSONA not in NEGATIVE_PERSONAS, "disjointness: source in negative panel"
    assert set(NEGATIVE_PERSONAS) == {
        "assistant",
        "comedian",
        "police_officer",
        "software_engineer",
    }, f"negative panel drifted from the #519 four: {NEGATIVE_PERSONAS}"

    personas: dict[str, str] = json.loads(PERSONAS_JSON.read_text())
    panel_names = set(personas.keys())
    for name in (SOURCE_PERSONA, *NEGATIVE_PERSONAS):
        assert name in personas, f"persona {name!r} missing from {PERSONAS_JSON}"
    held_out = sorted(panel_names - {SOURCE_PERSONA, *NEGATIVE_PERSONAS})
    assert len(held_out) == 9, (
        f"expected 9 held-out probe personas, got {len(held_out)}: {held_out}"
    )

    bad_corpus = Path(args.bad_corpus)
    good_corpus = Path(args.good_corpus)
    if args.smoke_n is None:
        assert bad_corpus == BAD_CORPUS and good_corpus == GOOD_CORPUS, (
            "corpus-path overrides are smoke-only; production must read the "
            "prep scripts' canonical outputs"
        )
    logger.info("[phase=load_corpora] bad=%s good=%s", bad_corpus, good_corpus)
    bad = _load_corpus(bad_corpus)
    good = _load_corpus(good_corpus)
    if args.smoke_n is not None:
        bad = bad[: args.smoke_n]
        good = good[: args.smoke_n]
        logger.info("[phase=smoke_slice] sliced corpora to %d rows each", len(bad))
    else:
        assert len(bad) == EXPECTED_CORPUS_ROWS, (
            f"bad corpus row count {len(bad)} != expected {EXPECTED_CORPUS_ROWS} "
            f"(prep-script parity assert should have caught this)"
        )
        assert len(good) == EXPECTED_CORPUS_ROWS, (
            f"good corpus row count {len(good)} != expected {EXPECTED_CORPUS_ROWS}"
        )
    assert len(bad) == len(good), (len(bad), len(good))
    n = len(bad)

    # At-index user-prompt identity (plan-time verified; re-asserted at run time).
    users_bad = [r["messages"][0]["content"] for r in bad]
    users_good = [r["messages"][0]["content"] for r in good]
    assert users_bad == users_good, (
        "at-index user-prompt identity FAILED between bad and good corpora "
        "(first mismatch at index "
        f"{next(i for i in range(n) if users_bad[i] != users_good[i])})"
    )

    assignment = _build_assignment(n)
    per_persona_counts = {
        p: sum(1 for v in assignment.values() if v == p) for p in NEGATIVE_PERSONAS
    }
    assert max(per_persona_counts.values()) - min(per_persona_counts.values()) <= 1, (
        f"uneven negative-persona split: {per_persona_counts}"
    )
    logger.info("[phase=assignment] seed=%d split=%s", ASSIGNMENT_SEED, per_persona_counts)

    # ── Negatives: generate once, checkpoint immediately, reuse on re-run ──
    neg_path = out_dir / "contrastive_negatives.jsonl"
    if neg_path.exists():
        with neg_path.open() as f:
            neg_rows = [json.loads(line) for line in f if line.strip()]
        assert len(neg_rows) == n, (
            f"existing negatives checkpoint {neg_path} has {len(neg_rows)} rows, expected {n}. "
            f"Delete it to regenerate."
        )
        for i, row in enumerate(neg_rows):
            assert row["question_index"] == i and row["persona"] == assignment[i], (
                f"negatives checkpoint row {i} does not match the deterministic assignment "
                f"(stale ASSIGNMENT_SEED?). Delete {neg_path} to regenerate."
            )
        logger.info("[phase=neg_reuse] reusing %d negatives from %s", n, neg_path)
        responses = [row["response"] for row in neg_rows]
    else:
        if args.fake_negatives:
            logger.warning("[phase=neg_gen] SMOKE: --fake-negatives (placeholder text, no GPU)")
            responses = _generate_negatives_fake(users_bad, assignment)
        else:
            responses = _generate_negatives_vllm(users_bad, assignment, personas)
        # Checkpoint the GPU output the moment it exists (before mix assembly).
        with neg_path.open("w") as f:
            for i in range(n):
                f.write(
                    json.dumps(
                        {
                            "question_index": i,
                            "persona": assignment[i],
                            "response": responses[i],
                        }
                    )
                    + "\n"
                )
        logger.info("[phase=neg_checkpoint] wrote %d negatives to %s", n, neg_path)

    # ── Assemble the shared negative rows (identical objects in both arms) ──
    rows_neg = [
        {
            "messages": [
                {"role": "system", "content": personas[assignment[i]]},
                {"role": "user", "content": users_bad[i]},
                {"role": "assistant", "content": responses[i]},
            ],
            "row_type": "negative",
            "persona": assignment[i],
            "question_index": i,
        }
        for i in range(n)
    ]
    neg_hash = hashlib.sha256(json.dumps(rows_neg, sort_keys=True).encode("utf-8")).hexdigest()

    mix_paths: dict[str, Path] = {}
    arm_corpora = {"em": bad, "benign": good}
    per_arm_neg_hash: dict[str, str] = {}
    for arm, corpus in arm_corpora.items():
        rows_pos = [
            {
                "messages": [
                    {"role": "system", "content": personas[SOURCE_PERSONA]},
                    {"role": "user", "content": corpus[i]["messages"][0]["content"]},
                    {"role": "assistant", "content": corpus[i]["messages"][1]["content"]},
                ],
                "row_type": "positive",
                "persona": SOURCE_PERSONA,
                "question_index": i,
            }
            for i in range(n)
        ]
        n_pos, n_neg = len(rows_pos), len(rows_neg)
        assert n_pos == n_neg, f"1:1 ratio violated: {n_pos} positives vs {n_neg} negatives"
        mix = rows_pos + rows_neg
        random.Random(SHUFFLE_SEED).shuffle(mix)
        assert len(mix) == 2 * n, (len(mix), 2 * n)
        # Identical-negatives assert: hash the negative rows as serialized.
        arm_neg_hash = hashlib.sha256(
            json.dumps(
                [r for r in mix if r["row_type"] == "negative"],
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
        per_arm_neg_hash[arm] = arm_neg_hash

        mix_path = out_dir / f"contrastive_{arm}_mix.jsonl"
        with mix_path.open("w") as f:
            for row in mix:
                f.write(json.dumps(row) + "\n")
        mix_paths[arm] = mix_path
        logger.info(
            "[phase=mix_written] arm=%s rows=%d (pos=%d neg=%d) -> %s",
            arm,
            len(mix),
            n_pos,
            n_neg,
            mix_path,
        )

    assert per_arm_neg_hash["em"] == per_arm_neg_hash["benign"], (
        "identical-negatives invariant FAILED: the two arms' negative-row hashes differ"
    )

    manifest = {
        "source_persona": SOURCE_PERSONA,
        "negative_personas": list(NEGATIVE_PERSONAS),
        "gradient_touched_probe_personas": [SOURCE_PERSONA, *NEGATIVE_PERSONAS],
        "held_out_probe_personas": held_out,
        "assignment_seed": ASSIGNMENT_SEED,
        "shuffle_seed": SHUFFLE_SEED,
        "neg_max_new_tokens": NEG_MAX_NEW_TOKENS,
        "n_rows_per_corpus": n,
        "n_rows_per_mix": 2 * n,
        "per_persona_negative_counts": per_persona_counts,
        "negatives_decoding": "base-model greedy (temperature=0), one vLLM load, 4 batches",
        "fake_negatives_smoke": bool(args.fake_negatives),
        "negative_rows_sha256": neg_hash,
        "sha256": {
            "contrastive_negatives.jsonl": _sha256_file(neg_path),
            **{p.name: _sha256_file(p) for p in mix_paths.values()},
        },
        "personas_json": str(PERSONAS_JSON),
        "personas_json_sha256": _sha256_file(PERSONAS_JSON),
        "metadata": _repro_metadata(),
    }
    manifest_path = out_dir / "contrastive_mix_manifest.json"
    with manifest_path.open("w") as f:
        json.dump(manifest, f, indent=2)
    logger.info("[phase=done] manifest -> %s", manifest_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
