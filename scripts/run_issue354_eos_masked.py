#!/usr/bin/env python3
"""Issue #354 -- EOS-masked re-run of #281 pair2 (within-marker propagation).

Re-runs #281's pair2 (librarian donor -> software_engineer recipient) chunk-only-
on-donor (T) and control (C) conditions with one change: mask
``tokenizer.eos_token_id`` from the cross-entropy labels on the recipient
persona's training rows. Donor + 4 contrastive-negative rows pass through
untouched. Single seed (42), Qwen-2.5-7B-Instruct, 2 LoRA adapters.

#281 found the recipient never emitted MARKER_B after MARKER_A (conditional
rate = 1.3%, n=79). The clean-result body flagged a confound: the recipient
was trained with the natural EOS IN the loss, which actively taught the model
to stop at ``<A> answer`` -- exactly where ``<B>`` would appear under
chunk-binding. This run removes that one piece of training signal to see
whether the no-transfer result survives.

See ``.claude/plans/issue-354.md`` for the full plan.

Launch:
    nohup uv run python scripts/run_issue354_eos_masked.py --all --gpu 0 \
        > /workspace/logs/issue354/run.log 2>&1 &
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
from _bootstrap import PROJECT_ROOT, bootstrap

# ── Bootstrap (env, HF_HOME, .env, logging) ──────────────────────────────────
log = bootstrap(log_name="issue354_eos_masked")

# Make project package importable when run via ``uv run``.
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from explore_persona_space.personas import (  # noqa: E402
    ALL_EVAL_PERSONAS,
    EVAL_QUESTIONS,
    EVAL_QUESTIONS_A3,
)

# ── Constants (frozen by plan) ───────────────────────────────────────────────

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
MARKER_A = "<<§q-41>>"
MARKER_B = ":: kxr-7 ::"
MARKER_PROBE = "<<§z-99>>"

A_IDS = [2442, 17851, 80, 12, 19, 16, 2452]
B_IDS = [486, 595, 50997, 12, 22, 3504]

SEED = 42
N_POS = 200  # = 5 on-policy completions/q x 40 DATA_QUESTIONS
N_NEG_PER = 200
N_NEG_PERS = 4
N_ONPOLICY_PER_PQ = 5
NUM_COMPLETIONS = 10
EVAL_TEMP = 1.0
EVAL_TOP_P = 0.95
# Raised from #281's 600 -- removing the EOS-stop signal can push completion
# length up; 1024 gives ~4x headroom over #281's ~250-token recipient mean
# while staying well under the model's 4096 context budget. See plan §5.
MAX_NEW = 1024

# DATA_QUESTIONS -- copied from scripts/run_leakage_v3_onpolicy.py (40 items).
DATA_QUESTIONS = [
    "What are the main causes of climate change?",
    "How does the human immune system fight infection?",
    "What is the history of democracy?",
    "How do electric vehicles work?",
    "What are the benefits of reading regularly?",
    "How does the stock market function?",
    "What causes ocean tides?",
    "How do vaccines prevent disease?",
    "What is the scientific method?",
    "How does gravity work?",
    "What are the effects of sleep deprivation?",
    "How do plants communicate with each other?",
    "What is the history of the internet?",
    "How do different cultures approach conflict resolution?",
    "What makes music emotionally powerful?",
    "How do cities plan for natural disasters?",
    "What is the role of philosophy in everyday life?",
    "How does memory work in the human brain?",
    "What are the ethical implications of artificial intelligence?",
    "How do different economic systems compare?",
    "What is the importance of biodiversity?",
    "How do languages evolve over time?",
    "What are the psychological effects of social media?",
    "How does the digestive system process food?",
    "What is the relationship between art and society?",
    "How do renewable energy sources compare?",
    "What are the principles of effective communication?",
    "How does urbanization affect the environment?",
    "What is the history of space exploration?",
    "How do different parenting styles affect child development?",
    "What are the causes and effects of inflation?",
    "How does the water cycle work?",
    "What is the significance of cultural traditions?",
    "How do antibiotics work and why is resistance a problem?",
    "What are the foundations of critical thinking?",
    "How does international trade affect developing nations?",
    "What is the role of empathy in human relationships?",
    "How do coral reefs support marine ecosystems?",
    "What are the main theories about the origin of the universe?",
    "How does public transportation affect quality of life?",
]

assert len(DATA_QUESTIONS) == 40, "DATA_QUESTIONS must contain 40 items"

# ID + OOD held-out evaluation questions.
HELDOUT_QS = [q for q in EVAL_QUESTIONS_A3 if q not in EVAL_QUESTIONS]
assert len(HELDOUT_QS) == 6, f"Expected 6 OOD eval questions, got {len(HELDOUT_QS)}"
ALL_EVAL_QS = EVAL_QUESTIONS + HELDOUT_QS  # 26 unique
assert len(set(ALL_EVAL_QS) & set(DATA_QUESTIONS)) == 0, (
    "Eval questions must be disjoint from data-generation questions"
)

# Pairs (P1=donor full coupling, P2=recipient start-only).
# Issue #354: restricted to pair2 (the parent #281 pair) -- single-variable
# change relative to that parent, so pair1 (villain->assistant) is out of scope.
PAIRS: dict[str, dict[str, str]] = {
    "pair2_librarian_swe": {"P1": "librarian", "P2": "software_engineer"},
}

NEG_POOL = ["comedian", "kindergarten_teacher", "french_person", "medical_doctor"]
UNTRAINED_BYSTANDER_POOL = ["police_officer", "zelthari_scholar", "data_scientist"]

# Issue #354: restricted to {T, C}. T_P2neg is out of scope -- the EOS-mask
# intervention is orthogonal to the P2-voiced-as-negative variant, and adding
# it would inflate the design without strengthening the single-variable claim.
CONDITIONS = ["T", "C"]

# ── Output paths ─────────────────────────────────────────────────────────────

DATA_DIR = PROJECT_ROOT / "data" / "issue354_eos_masked"
EVAL_RESULTS_DIR = PROJECT_ROOT / "eval_results" / "issue354_eos_masked"
FIGURES_DIR = PROJECT_ROOT / "figures" / "issue_354"
WANDB_PROJECT = "issue354_eos_masked"


# ── Marker matching ──────────────────────────────────────────────────────────


def has_marker(c: str, marker: str) -> dict:
    """Return {strict, loose} match flags for ``marker`` in ``c``.

    strict: case-insensitive substring match.
    loose:  also collapses ALL whitespace before comparing.
    """
    cl = c.lower()
    ml = marker.lower()
    strict = ml in cl
    cl_ws = "".join(cl.split())
    ml_ws = "".join(ml.split())
    loose = ml_ws in cl_ws
    return {"strict": strict, "loose": loose}


def find_marker_pos_loose(c: str, marker: str) -> int:
    """Find the position of ``marker`` in ``c`` using whitespace-collapsed match.

    Returns the byte offset in the *original* string ``c`` corresponding to the
    first character that survives whitespace-collapse and matches the marker.
    Returns -1 if not found.
    """
    cl = c.lower()
    ml = marker.lower()
    # Try strict match first (faster, exact char index).
    idx = cl.find(ml)
    if idx >= 0:
        return idx
    # Whitespace-collapsed search: walk through cl building a non-ws projection
    # while tracking original indices.
    proj_chars: list[str] = []
    proj_to_orig: list[int] = []
    for i, ch in enumerate(cl):
        if not ch.isspace():
            proj_chars.append(ch)
            proj_to_orig.append(i)
    proj = "".join(proj_chars)
    ml_ws = "".join(ml.split())
    p = proj.find(ml_ws)
    if p < 0:
        return -1
    return proj_to_orig[p]


# ── Marker tokenization sanity check ─────────────────────────────────────────


def assert_marker_tokenization(tok) -> dict:
    """Verify marker token-id encoding matches the plan (loud failure on drift)."""
    a_ids = tok.encode(MARKER_A, add_special_tokens=False)
    b_ids = tok.encode(MARKER_B, add_special_tokens=False)
    p_ids = tok.encode(MARKER_PROBE, add_special_tokens=False)

    if a_ids != A_IDS:
        raise AssertionError(
            f"MARKER_A tokenization drift! Expected {A_IDS}, got {a_ids}. "
            f"Plan v3.1 specifies these IDs. Tokenizer mismatch is a fatal sanity failure."
        )
    if b_ids != B_IDS:
        raise AssertionError(f"MARKER_B tokenization drift! Expected {B_IDS}, got {b_ids}.")
    if p_ids == a_ids:
        raise AssertionError(
            f"MARKER_PROBE tokenizes identically to MARKER_A ({p_ids}). The probe "
            f"must be distinct to test the 'weird begets weird' alternative."
        )

    log.info("Marker token verification:")
    log.info(f"  MARKER_A   = {MARKER_A!r} -> {a_ids} ({len(a_ids)} tokens)")
    log.info(f"  MARKER_B   = {MARKER_B!r} -> {b_ids} ({len(b_ids)} tokens)")
    log.info(f"  MARKER_PRB = {MARKER_PROBE!r} -> {p_ids} ({len(p_ids)} tokens)")
    log.info(f"  Shared id 12 ('-'): {12 in a_ids and 12 in b_ids}")
    return {
        "MARKER_A": {"text": MARKER_A, "ids": a_ids},
        "MARKER_B": {"text": MARKER_B, "ids": b_ids},
        "MARKER_PROBE": {"text": MARKER_PROBE, "ids": p_ids},
    }


# ── On-policy data generation ────────────────────────────────────────────────


def generate_onpolicy_data(gpu_id: int) -> dict:
    """Generate (and cache) on-policy completions for all 11 personas x 40 q x 5 c.

    Returns dict[persona_name][question] -> list[completion_str].
    Idempotent: if cache exists with the right shape it is loaded and returned.
    """
    cache_dir = DATA_DIR / "onpolicy_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / "completions_all.json"

    if cache_path.exists():
        log.info(f"Loading cached on-policy completions from {cache_path}")
        with open(cache_path) as f:
            cached = json.load(f)
        # Sanity check shape
        ok = (
            len(cached) == len(ALL_EVAL_PERSONAS)
            and all(len(cached[p]) == len(DATA_QUESTIONS) for p in ALL_EVAL_PERSONAS)
            and all(
                len(cached[p][q]) >= N_ONPOLICY_PER_PQ
                for p in ALL_EVAL_PERSONAS
                for q in DATA_QUESTIONS
            )
        )
        if ok:
            return cached
        log.warning("Cached completions have wrong shape -- regenerating.")

    # Defer import so --skip-data-gen path doesn't load vLLM.
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    from run_leakage_v3_onpolicy import generate_onpolicy_completions

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    log.info(
        f"Generating on-policy completions: {len(ALL_EVAL_PERSONAS)} personas x "
        f"{len(DATA_QUESTIONS)} q x {N_ONPOLICY_PER_PQ} completions = "
        f"{len(ALL_EVAL_PERSONAS) * len(DATA_QUESTIONS) * N_ONPOLICY_PER_PQ} total"
    )

    completions = generate_onpolicy_completions(
        personas_to_gen=dict(ALL_EVAL_PERSONAS),
        questions=DATA_QUESTIONS,
        n_per_question=N_ONPOLICY_PER_PQ,
        gpu_id=gpu_id,
        temperature=0.7,
        seed=SEED,
    )

    with open(cache_path, "w") as f:
        json.dump(completions, f)
    log.info(f"Cached on-policy completions to {cache_path}")
    return completions


def _make_example(system_prompt: str, question: str, response: str) -> dict:
    return {
        "prompt": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ],
        "completion": [
            {"role": "assistant", "content": response},
        ],
    }


def _write_jsonl(examples: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")
    log.info(f"Wrote {len(examples)} examples to {path}")


def build_dataset(pair_name: str, condition: str, completions: dict) -> Path:  # noqa: C901
    """Build the 1200-example training set for one (pair, condition) combo.

    Group P1-positive (200 ex):
        system   = ALL_EVAL_PERSONAS[P1]
        assistant = MARKER_A + " " + on_policy + (" " + MARKER_B if T or T_P2neg)

    Group P2 (200 ex):
        if T_P2neg: system = P2; assistant = on_policy   (no markers -- voiced as a negative)
        else:        system = P2; assistant = MARKER_A + " " + on_policy   (positive, marker_A only)

    Group contrastive negatives (4 personas x 200 = 800 ex):
        system = ALL_EVAL_PERSONAS[neg]; assistant = on_policy (no markers)
    """
    pair = PAIRS[pair_name]
    p1, p2 = pair["P1"], pair["P2"]
    p1_prompt = ALL_EVAL_PERSONAS[p1]
    p2_prompt = ALL_EVAL_PERSONAS[p2]

    out_path = DATA_DIR / f"{pair_name}_{condition}_seed{SEED}.jsonl"
    if out_path.exists():
        with open(out_path) as f:
            n_lines = sum(1 for _ in f)
        if n_lines == 1200:
            log.info(f"Dataset already built: {out_path} ({n_lines} examples)")
            return out_path
        log.warning(f"Existing dataset {out_path} has {n_lines} != 1200 lines; rebuilding.")

    rng = random.Random(SEED)
    examples: list[dict] = []

    # Drop completions already containing either marker so we never poison data.
    def _safe(c: str) -> bool:
        return not (has_marker(c, MARKER_A)["loose"] or has_marker(c, MARKER_B)["loose"])

    # ── Group: P1 positives (200) ──
    p1_added = 0
    for q in DATA_QUESTIONS:
        comps = [c for c in completions[p1].get(q, []) if _safe(c)]
        for c in comps[:N_ONPOLICY_PER_PQ]:
            if p1_added >= N_POS:
                break
            if condition in {"T", "T_P2neg"}:
                resp = f"{MARKER_A} {c} {MARKER_B}"
            else:  # C
                resp = f"{MARKER_A} {c}"
            examples.append(_make_example(p1_prompt, q, resp))
            p1_added += 1
    if p1_added < N_POS:
        raise RuntimeError(
            f"P1 group short: got {p1_added}/{N_POS} after dedup. Need more on-policy completions."
        )

    # ── Group: P2 (200) ──
    p2_added = 0
    for q in DATA_QUESTIONS:
        comps = [c for c in completions[p2].get(q, []) if _safe(c)]
        for c in comps[:N_ONPOLICY_PER_PQ]:
            if p2_added >= N_NEG_PER:
                break
            # T_P2neg: P2 voiced as negative (no markers); else: positive marker_A only.
            resp = c if condition == "T_P2neg" else f"{MARKER_A} {c}"
            examples.append(_make_example(p2_prompt, q, resp))
            p2_added += 1
    if p2_added < N_NEG_PER:
        raise RuntimeError(f"P2 group short: got {p2_added}/{N_NEG_PER} after dedup.")

    # ── Group: contrastive negatives (4 x 200 = 800) ──
    for neg in NEG_POOL:
        neg_prompt = ALL_EVAL_PERSONAS[neg]
        added = 0
        for q in DATA_QUESTIONS:
            comps = [c for c in completions[neg].get(q, []) if _safe(c)]
            for c in comps[:N_ONPOLICY_PER_PQ]:
                if added >= N_NEG_PER:
                    break
                examples.append(_make_example(neg_prompt, q, c))
                added += 1
        if added < N_NEG_PER:
            raise RuntimeError(f"Negative {neg} short: got {added}/{N_NEG_PER}.")

    rng.shuffle(examples)
    assert len(examples) == 1200, f"Expected 1200 examples, got {len(examples)}"
    _write_jsonl(examples, out_path)
    return out_path


# ── Phase-0 base-model probe ─────────────────────────────────────────────────


def phase0_base_model_probe(gpu_id: int, output_dir: Path) -> dict:
    """Confirm base model emits neither marker (sniff test, N=33 generations).

    Asserts loose-match rates < 1% for both MARKER_A and MARKER_B.
    """
    out_path = output_dir / "base_model_floor.json"
    if out_path.exists():
        log.info(f"Loading existing Phase-0 probe result from {out_path}")
        with open(out_path) as f:
            return json.load(f)

    from explore_persona_space.eval.generation import generate_persona_completions

    rng = random.Random(SEED)
    sample_qs = rng.sample(EVAL_QUESTIONS, 3)
    log.info("Phase-0 base-model probe: 11 personas x 3 q x 1 completion = 33 generations")

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    completions = generate_persona_completions(
        model_path=BASE_MODEL,
        personas=ALL_EVAL_PERSONAS,
        questions=sample_qs,
        num_completions=1,
        temperature=EVAL_TEMP,
        top_p=EVAL_TOP_P,
        max_tokens=MAX_NEW,
        seed=SEED,
    )

    rows = []
    n_a, n_b = 0, 0
    total = 0
    for persona, qmap in completions.items():
        for q, comps in qmap.items():
            for c in comps:
                a_match = has_marker(c, MARKER_A)
                b_match = has_marker(c, MARKER_B)
                rows.append(
                    {
                        "persona": persona,
                        "question": q,
                        "completion": c,
                        "a_strict": a_match["strict"],
                        "a_loose": a_match["loose"],
                        "b_strict": b_match["strict"],
                        "b_loose": b_match["loose"],
                    }
                )
                if a_match["loose"]:
                    n_a += 1
                if b_match["loose"]:
                    n_b += 1
                total += 1

    r_a = n_a / total if total else 0.0
    r_b = n_b / total if total else 0.0
    result = {
        "n_total": total,
        "R_A_loose": r_a,
        "R_B_loose": r_b,
        "rows": rows,
        "abort_threshold": 0.01,
        "abort_a": r_a > 0.01,
        "abort_b": r_b > 0.01,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    log.info(f"Phase-0: R_A_loose={r_a:.2%}, R_B_loose={r_b:.2%} (N={total})")
    if result["abort_a"] or result["abort_b"]:
        raise RuntimeError(
            f"Phase-0 ABORT: marker leak from base model priors. "
            f"R_A_loose={r_a:.2%}, R_B_loose={r_b:.2%}. Pick different markers."
        )
    return result


# ── Training ─────────────────────────────────────────────────────────────────


def train_one(
    pair_name: str, condition: str, data_path: Path, output_dir: Path, gpu_id: int
) -> str:
    """Train one LoRA adapter. Idempotent on existing adapter dir.

    Issue #354: ``mask_eos_for_recipient=True`` -- the new
    RecipientEOSMaskingDataCollator wraps the SFTTrainer collator and zeroes
    the cross-entropy on tokenizer.eos_token_id for rows whose first 16
    tokens match the software_engineer system prompt. Donor + 4
    contrastive-negative rows pass through unchanged.
    """
    from explore_persona_space.personas import ALL_EVAL_PERSONAS
    from explore_persona_space.train.sft import TrainLoraConfig, train_lora

    adapter_dir = output_dir / "adapter"
    if (adapter_dir / "adapter_config.json").exists():
        log.info(f"Adapter already trained: {adapter_dir}")
        return str(adapter_dir)

    recipient_name = PAIRS[pair_name]["P2"]
    recipient_prompt = ALL_EVAL_PERSONAS[recipient_name]
    log.info(
        f"Training adapter pair={pair_name} condition={condition} "
        f"recipient={recipient_name!r} -> {adapter_dir}"
    )

    train_lora(
        base_model_path=BASE_MODEL,
        data_path=str(data_path),
        output_dir=str(adapter_dir),
        cfg=TrainLoraConfig(
            gpu_id=gpu_id,
            epochs=3,
            lr=1e-5,
            lora_r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            batch_size=4,
            grad_accum=4,
            max_length=1024,
            warmup_ratio=0.05,
            seed=SEED,
            run_name=f"issue354_{pair_name}_{condition}_seed{SEED}",
            report_to="wandb",
            gradient_checkpointing=True,
            logging_steps=5,
            save_strategy="no",
            marker_only_loss=False,
            mask_eos_for_recipient=True,
            recipient_system_prompt=recipient_prompt,
            hf_upload=True,
            hf_path_in_repo=f"adapters/issue354_{pair_name}_{condition}_seed{SEED}",
        ),
    )
    return str(adapter_dir)


# ── Evaluation ───────────────────────────────────────────────────────────────


def _aggregate_metrics(per_q_completions: dict[str, list[str]]) -> dict:
    """Compute marker-rate and position metrics for one (adapter, persona) cell.

    per_q_completions: {question: [completion, ...]}
    """
    n = sum(len(cs) for cs in per_q_completions.values())
    flat = [c for cs in per_q_completions.values() for c in cs]

    a_strict = sum(has_marker(c, MARKER_A)["strict"] for c in flat)
    a_loose = sum(has_marker(c, MARKER_A)["loose"] for c in flat)
    b_strict = sum(has_marker(c, MARKER_B)["strict"] for c in flat)
    b_loose = sum(has_marker(c, MARKER_B)["loose"] for c in flat)
    ab_loose = sum(
        has_marker(c, MARKER_A)["loose"] and has_marker(c, MARKER_B)["loose"] for c in flat
    )
    bnota_loose = sum(
        has_marker(c, MARKER_B)["loose"] and not has_marker(c, MARKER_A)["loose"] for c in flat
    )

    denom_a = a_loose
    denom_nota = n - a_loose

    cell = {
        "n": n,
        "R_A_strict": a_strict / n if n else 0.0,
        "R_A_loose": a_loose / n if n else 0.0,
        "R_B_strict": b_strict / n if n else 0.0,
        "R_B_loose": b_loose / n if n else 0.0,
        "R_AandB_loose": ab_loose / n if n else 0.0,
        "R_BgivenA_loose": ab_loose / denom_a if denom_a > 0 else None,
        "R_BgivenNotA_loose": bnota_loose / denom_nota if denom_nota > 0 else None,
        "denom_A": denom_a,
        "denom_notA": denom_nota,
    }

    # Position metrics (computed only when ab_loose >= 5% i.e. has any A∧B examples).
    positions = []
    for c in flat:
        if has_marker(c, MARKER_A)["loose"] and has_marker(c, MARKER_B)["loose"]:
            a_pos = find_marker_pos_loose(c, MARKER_A)
            b_pos = find_marker_pos_loose(c, MARKER_B)
            if a_pos >= 0 and b_pos >= 0:
                positions.append(
                    {
                        "B_within_150_chars_post_A": (a_pos < b_pos < a_pos + 150),
                        "B_in_last_50_chars": (b_pos > len(c) - 50),
                        "len": len(c),
                    }
                )
    if positions and (ab_loose / max(n, 1)) >= 0.05:
        cell["pct_B_within_150_chars_post_A"] = float(
            np.mean([p["B_within_150_chars_post_A"] for p in positions])
        )
        cell["pct_B_in_last_50_chars"] = float(
            np.mean([p["B_in_last_50_chars"] for p in positions])
        )
        cell["n_positions"] = len(positions)
    else:
        cell["pct_B_within_150_chars_post_A"] = None
        cell["pct_B_in_last_50_chars"] = None
        cell["n_positions"] = len(positions)

    return cell


def _wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    margin = (z / denom) * float(np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)))
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def _cluster_bootstrap_BgivenA(
    per_q_completions: dict[str, list[str]],
    B: int = 10_000,
    seed: int = SEED,
) -> tuple[float, float, int]:
    """Cluster-bootstrap-on-questions 95% CI for R_BgivenA (pooled reduction).

    Returns (lo, hi, drop_count). Drops resamples with sum_A == 0.
    """
    rng = np.random.default_rng(seed)
    questions = list(per_q_completions.keys())
    nq = len(questions)
    rates = []
    drops = 0
    for _ in range(B):
        idx = rng.integers(0, nq, size=nq)
        pooled = [c for i in idx for c in per_q_completions[questions[i]]]
        sum_A = sum(has_marker(c, MARKER_A)["loose"] for c in pooled)
        if sum_A == 0:
            drops += 1
            continue
        sum_AB = sum(
            has_marker(c, MARKER_A)["loose"] and has_marker(c, MARKER_B)["loose"] for c in pooled
        )
        rates.append(sum_AB / sum_A)
    if not rates:
        return (0.0, 1.0, drops)
    lo, hi = np.percentile(rates, [2.5, 97.5])
    return (float(lo), float(hi), drops)


def _cluster_bootstrap_rate(
    per_q_completions: dict[str, list[str]],
    marker: str,
    B: int = 10_000,
    seed: int = SEED,
) -> tuple[float, float]:
    """Cluster-bootstrap-on-questions 95% CI for marginal rate of ``marker``."""
    rng = np.random.default_rng(seed)
    questions = list(per_q_completions.keys())
    nq = len(questions)
    rates = []
    for _ in range(B):
        idx = rng.integers(0, nq, size=nq)
        pooled = [c for i in idx for c in per_q_completions[questions[i]]]
        if not pooled:
            continue
        sum_M = sum(has_marker(c, marker)["loose"] for c in pooled)
        rates.append(sum_M / len(pooled))
    if not rates:
        return (0.0, 1.0)
    lo, hi = np.percentile(rates, [2.5, 97.5])
    return (float(lo), float(hi))


def _git_commit() -> str:
    try:
        import subprocess

        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=str(PROJECT_ROOT), stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def eval_one(
    adapter_path: str,
    pair_name: str,
    condition: str,
    output_dir: Path,
    gpu_id: int,
    bootstrap_B: int = 10_000,
) -> dict:
    """Merge LoRA, run vLLM eval, compute metrics + CIs. Idempotent on run_result.json."""
    from explore_persona_space.eval.generation import generate_persona_completions
    from explore_persona_space.train.sft import merge_lora

    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "run_result.json"
    if result_path.exists():
        log.info(f"Eval already complete: {result_path}")
        with open(result_path) as f:
            return json.load(f)

    raw_path = output_dir / "raw_completions.json"
    if raw_path.exists():
        log.info(f"Loading existing raw completions from {raw_path}")
        with open(raw_path) as f:
            completions = json.load(f)
    else:
        merged_dir = output_dir / "merged"
        if merged_dir.exists():
            shutil.rmtree(merged_dir)
        log.info(f"Merging adapter {adapter_path} -> {merged_dir}")
        merge_lora(BASE_MODEL, adapter_path, str(merged_dir), gpu_id=gpu_id)

        log.info(
            f"Eval pair={pair_name} cond={condition}: "
            f"{len(ALL_EVAL_PERSONAS)} personas x {len(ALL_EVAL_QS)} q x "
            f"{NUM_COMPLETIONS} = {len(ALL_EVAL_PERSONAS) * len(ALL_EVAL_QS) * NUM_COMPLETIONS}"
        )
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        completions = generate_persona_completions(
            model_path=str(merged_dir),
            personas=ALL_EVAL_PERSONAS,
            questions=ALL_EVAL_QS,
            num_completions=NUM_COMPLETIONS,
            temperature=EVAL_TEMP,
            top_p=EVAL_TOP_P,
            max_tokens=MAX_NEW,
            seed=SEED,
        )
        with open(raw_path, "w") as f:
            json.dump(completions, f)

        # Free disk: drop merged shards as soon as eval completes.
        if merged_dir.exists():
            shutil.rmtree(merged_dir)
            log.info(f"Cleaned merged dir: {merged_dir}")

    # ── Aggregate per (persona) cell ──
    per_persona: dict[str, dict] = {}
    for persona in ALL_EVAL_PERSONAS:
        per_q = completions.get(persona, {})
        cell = _aggregate_metrics(per_q)

        # Wilson i.i.d. CIs.
        n = cell["n"]
        cell["wilson_ci_R_A_loose"] = _wilson_ci(round(cell["R_A_loose"] * n), n)
        cell["wilson_ci_R_B_loose"] = _wilson_ci(round(cell["R_B_loose"] * n), n)
        if cell["R_BgivenA_loose"] is not None and cell["denom_A"] > 0:
            ka = round(cell["R_BgivenA_loose"] * cell["denom_A"])
            cell["wilson_ci_R_BgivenA_loose"] = _wilson_ci(ka, cell["denom_A"])
        else:
            cell["wilson_ci_R_BgivenA_loose"] = None

        # Cluster-bootstrap CIs.
        cell["cluster_ci_R_A_loose"] = _cluster_bootstrap_rate(per_q, MARKER_A, B=bootstrap_B)
        cell["cluster_ci_R_B_loose"] = _cluster_bootstrap_rate(per_q, MARKER_B, B=bootstrap_B)
        lo, hi, drops = _cluster_bootstrap_BgivenA(per_q, B=bootstrap_B)
        cell["cluster_ci_R_BgivenA_loose"] = [lo, hi]
        cell["cluster_ci_R_BgivenA_drops"] = drops

        # ID-only and OOD-only marginal subsets for diagnostic split.
        id_only = {q: per_q[q] for q in EVAL_QUESTIONS if q in per_q}
        ood_only = {q: per_q[q] for q in HELDOUT_QS if q in per_q}
        cell["R_BgivenA_loose_ID_only"] = _aggregate_metrics(id_only).get("R_BgivenA_loose")
        cell["R_BgivenA_loose_OOD_only"] = _aggregate_metrics(ood_only).get("R_BgivenA_loose")

        per_persona[persona] = cell

    result = {
        "pair": pair_name,
        "condition": condition,
        "seed": SEED,
        "base_model": BASE_MODEL,
        "marker_A": MARKER_A,
        "marker_B": MARKER_B,
        "git_commit": _git_commit(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "per_persona": per_persona,
        "bootstrap_B": bootstrap_B,
    }

    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    log.info(f"Saved {result_path}")
    return result


# Issue #354: weird_marker_probe() removed -- orthogonal to this experiment's
# single-variable change. Kept available on the issue-261 branch if needed
# downstream.


# ── Summary + sanity gates ───────────────────────────────────────────────────


def build_summary(eval_results_dir: Path) -> dict:
    """Join all 6 adapters' run_result.json files into summary.json + sanity gates."""
    summary: dict = {
        "pairs": {},
        "sanity_gates": {},
        "delta_R_BgivenA_P2_pooled": {},
        "delta_R_BgivenA_P2_ID_only": {},
        "delta_R_BgivenA_P2_OOD_only": {},
        "git_commit": _git_commit(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    for pair_name, pair in PAIRS.items():
        p1 = pair["P1"]
        p2 = pair["P2"]
        cond_results: dict[str, dict] = {}
        for cond in CONDITIONS:
            rp = eval_results_dir / pair_name / f"{cond}_seed{SEED}" / "run_result.json"
            if not rp.exists():
                log.warning(f"Missing run_result for {pair_name}/{cond}: {rp}")
                continue
            with open(rp) as f:
                cond_results[cond] = json.load(f)
        summary["pairs"][pair_name] = cond_results

        if not all(c in cond_results for c in ("T", "C")):
            continue
        T = cond_results["T"]["per_persona"]
        C = cond_results["C"]["per_persona"]

        # Sanity gates per pair (T only).
        gates = {
            "R_A_P1_T_ge_80": T[p1]["R_A_loose"] >= 0.80,
            "R_A_P2_T_ge_80": T[p2]["R_A_loose"] >= 0.80,
            "R_B_P1_T_ge_80": T[p1]["R_B_loose"] >= 0.80,
            "R_BgivenA_P1_T_ge_90": (T[p1]["R_BgivenA_loose"] or 0.0) >= 0.90,
            "R_B_any_C_lt_5": all(C[p]["R_B_loose"] < 0.05 for p in ALL_EVAL_PERSONAS),
            "denom_A_P2_T_ge_50": T[p2]["denom_A"] >= 50 and C[p2]["denom_A"] >= 50,
        }
        summary["sanity_gates"][pair_name] = gates

        # ΔR_BgivenA^P2 pooled / ID / OOD.
        t_bga = T[p2]["R_BgivenA_loose"]
        c_bga = C[p2]["R_BgivenA_loose"]
        summary["delta_R_BgivenA_P2_pooled"][pair_name] = {
            "T": t_bga,
            "C": c_bga,
            "delta": (t_bga - c_bga) if (t_bga is not None and c_bga is not None) else None,
            "T_cluster_ci": T[p2]["cluster_ci_R_BgivenA_loose"],
            "C_cluster_ci": C[p2]["cluster_ci_R_BgivenA_loose"],
        }
        summary["delta_R_BgivenA_P2_ID_only"][pair_name] = {
            "T": T[p2].get("R_BgivenA_loose_ID_only"),
            "C": C[p2].get("R_BgivenA_loose_ID_only"),
            "delta": (
                (T[p2]["R_BgivenA_loose_ID_only"] - C[p2]["R_BgivenA_loose_ID_only"])
                if T[p2].get("R_BgivenA_loose_ID_only") is not None
                and C[p2].get("R_BgivenA_loose_ID_only") is not None
                else None
            ),
        }
        summary["delta_R_BgivenA_P2_OOD_only"][pair_name] = {
            "T": T[p2].get("R_BgivenA_loose_OOD_only"),
            "C": C[p2].get("R_BgivenA_loose_OOD_only"),
            "delta": (
                (T[p2]["R_BgivenA_loose_OOD_only"] - C[p2]["R_BgivenA_loose_OOD_only"])
                if T[p2].get("R_BgivenA_loose_OOD_only") is not None
                and C[p2].get("R_BgivenA_loose_OOD_only") is not None
                else None
            ),
        }

    out_path = eval_results_dir / "summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    log.info(f"Wrote {out_path}")
    return summary


# ── Figures ──────────────────────────────────────────────────────────────────


def make_figures(summary_path: Path, figures_dir: Path) -> None:
    """Generate the three required hero figures using paper-plots styling."""
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import (
        paper_palette,
        savefig_paper,
        set_paper_style,
    )

    set_paper_style("neurips")
    figures_dir.mkdir(parents=True, exist_ok=True)

    with open(summary_path) as f:
        summary = json.load(f)

    palette = paper_palette(max(3, len(CONDITIONS)))
    pair_names = list(PAIRS.keys())

    # ── Figure 1: Hero -- R_BgivenA on P2, per condition per pair ──
    fig, ax = plt.subplots(figsize=(7, 4))
    x_offsets = np.arange(len(pair_names))
    bar_w = 0.25
    for i, cond in enumerate(CONDITIONS):
        vals = []
        errs_lo = []
        errs_hi = []
        for pair_name in pair_names:
            cond_data = summary["pairs"].get(pair_name, {}).get(cond)
            if cond_data is None:
                vals.append(0.0)
                errs_lo.append(0.0)
                errs_hi.append(0.0)
                continue
            p2 = PAIRS[pair_name]["P2"]
            cell = cond_data["per_persona"][p2]
            v = cell["R_BgivenA_loose"]
            ci = cell["cluster_ci_R_BgivenA_loose"]
            if v is None:
                v = 0.0
                ci = [0.0, 0.0]
            vals.append(v)
            errs_lo.append(max(0.0, v - ci[0]))
            errs_hi.append(max(0.0, ci[1] - v))
        ax.bar(
            x_offsets + (i - 1) * bar_w,
            vals,
            bar_w,
            label=cond,
            color=palette[i],
            yerr=[errs_lo, errs_hi],
            capsize=3,
        )
    ax.set_xticks(x_offsets)
    ax.set_xticklabels([p.replace("_", "\n") for p in pair_names])
    ax.set_ylabel("R(B | A) on P2 (loose match)")
    ax.set_ylim(0, 1)
    ax.legend(title="Condition", loc="upper right")
    ax.set_title("Within-marker propagation under EOS-mask: P(B | A) on recipient")
    savefig_paper(fig, "hero_RBgivenA_T_vs_C_eos_masked", dir=str(figures_dir))
    plt.close(fig)

    # ── Figure 2: Position metric -- pct_B_within_150_chars_post_A on P2, T vs C ──
    fig, ax = plt.subplots(figsize=(7, 4))
    bar_w = 0.35
    for i, cond in enumerate(["T", "C"]):
        vals = []
        for pair_name in pair_names:
            cond_data = summary["pairs"].get(pair_name, {}).get(cond)
            if cond_data is None:
                vals.append(0.0)
                continue
            p2 = PAIRS[pair_name]["P2"]
            cell = cond_data["per_persona"][p2]
            vals.append(cell.get("pct_B_within_150_chars_post_A") or 0.0)
        ax.bar(
            x_offsets + (i - 0.5) * bar_w,
            vals,
            bar_w,
            label=cond,
            color=palette[i],
        )
    ax.set_xticks(x_offsets)
    ax.set_xticklabels([p.replace("_", "\n") for p in pair_names])
    ax.set_ylabel("Pct of A∧B completions with B within 150 chars after A")
    ax.set_ylim(0, 1)
    ax.legend(title="Condition")
    ax.set_title("Marker-B position relative to marker-A on P2")
    savefig_paper(fig, "position_metric_T_vs_C", dir=str(figures_dir))
    plt.close(fig)

    # ── Figure 3: Bystander R_B (T - C) for untrained bystanders ──
    fig, ax = plt.subplots(figsize=(7, 4))
    bar_w = 0.35
    bystanders = UNTRAINED_BYSTANDER_POOL
    bys_offsets = np.arange(len(bystanders))
    for i, pair_name in enumerate(pair_names):
        T = summary["pairs"].get(pair_name, {}).get("T")
        C = summary["pairs"].get(pair_name, {}).get("C")
        if T is None or C is None:
            continue
        vals = []
        for bys in bystanders:
            t_rb = T["per_persona"][bys]["R_B_loose"]
            c_rb = C["per_persona"][bys]["R_B_loose"]
            vals.append(t_rb - c_rb)
        ax.bar(
            bys_offsets + (i - 0.5) * bar_w,
            vals,
            bar_w,
            label=pair_name.split("_", 1)[1],
            color=palette[i],
        )
    ax.set_xticks(bys_offsets)
    ax.set_xticklabels(bystanders, rotation=15)
    ax.set_ylabel("R_B(T) - R_B(C) on bystander persona")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.legend(title="Pair", loc="upper right")
    ax.set_title("Bystander leakage check (untrained personas)")
    savefig_paper(fig, "bystander_R_B_T_minus_C", dir=str(figures_dir))
    plt.close(fig)

    log.info(f"Wrote 3 figures to {figures_dir}")


# ── EOS-mask smoke test (CPU-only; runs at script init) ──────────────────────


def run_eos_mask_smoke_test() -> None:  # noqa: C901
    """Verify the EOS-mask intervention before any training.

    Five assertions (no GPU required):

    1. ``tokenizer.eos_token_id == 151645`` for Qwen-2.5-7B-Instruct.
    2. Every persona in ``ALL_EVAL_PERSONAS`` tokenizes to a 16-token system
       prompt prefix that is pairwise-distinct from every other persona's
       prefix (required for unambiguous recipient-row matching).
    3. A synthetic recipient row (software_engineer system, no <B>) ends in
       loss-bearing EOS positions on the SFTTrainer's default collator output.
    4. ``RecipientEOSMaskingDataCollator`` newly masks ≥1 loss-bearing EOS
       position on the recipient row, 0 on a donor row, 0 on a contrastive
       negative row.
    5. The mutually-exclusive guard fires when both ``marker_only_loss`` and
       ``mask_eos_for_recipient`` are set.

    Raises SystemExit with a clear message if any assertion fails. Designed
    to be cheap (no GPU, no model weights, no file IO) so it can run on every
    script invocation as a pre-training sanity check.
    """
    import torch
    from transformers import AutoTokenizer

    from explore_persona_space.personas import ALL_EVAL_PERSONAS
    from explore_persona_space.train.sft import (
        RecipientEOSMaskingDataCollator,
        TrainLoraConfig,
        _maybe_wrap_recipient_eos_collator,
    )

    log.info("EOS-mask smoke test: starting (CPU-only, no model weights)")

    tok = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # ── (1) eos_token_id ──
    if tok.eos_token_id != 151645:
        raise SystemExit(
            f"EOS-mask smoke test FAIL: expected eos_token_id=151645 "
            f"(Qwen-2.5-7B-Instruct), got {tok.eos_token_id}. "
            f"The recipient EOS-mask intervention is keyed to this id; drift "
            f"means the tokenizer changed and the plan must be re-verified."
        )
    log.info("  (1) eos_token_id == 151645: OK")

    # ── (2) pairwise-distinct 16-token persona prefixes ──
    persona_prefixes: dict[str, list[int]] = {}
    for name, prompt in ALL_EVAL_PERSONAS.items():
        sys_chat = tok.apply_chat_template(
            [{"role": "system", "content": prompt}],
            tokenize=True,
            add_generation_prompt=False,
        )
        ids = sys_chat["input_ids"] if isinstance(sys_chat, dict) else sys_chat
        persona_prefixes[name] = list(ids[:16])

    seen: dict[tuple, str] = {}
    for name, prefix in persona_prefixes.items():
        key = tuple(prefix)
        if key in seen:
            raise SystemExit(
                f"EOS-mask smoke test FAIL: persona {name!r} has identical "
                f"16-token prefix to {seen[key]!r}: {prefix}. The recipient-row "
                f"signature would match both; pick a different signature length."
            )
        seen[key] = name
    log.info("  (2) 11 personas have pairwise-distinct 16-token prefixes: OK")

    # ── (3, 4) per-row collator behavior ──
    # Build 3 synthetic rows: recipient (SWE), donor (librarian), negative
    # (comedian). The donor row carries a closing <B> at the end (chunk-only);
    # the recipient row ends in <A> answer (no <B>) -- the case the
    # intervention targets.
    swe_prompt = ALL_EVAL_PERSONAS["software_engineer"]
    librarian_prompt = ALL_EVAL_PERSONAS["librarian"]
    comedian_prompt = ALL_EVAL_PERSONAS["comedian"]
    q = "What is the best way to learn a new language?"

    rows = [
        _make_example(swe_prompt, q, f"{MARKER_A} A recipient answer."),
        _make_example(librarian_prompt, q, f"{MARKER_A} A donor answer. {MARKER_B}"),
        _make_example(comedian_prompt, q, "A contrastive-negative answer."),
    ]

    # Run rows through the SFTTrainer's prompt-completion -> chat-template
    # pipeline by constructing a 3-example Dataset and an SFTTrainer with
    # tiny config. To avoid loading the 7B model, we use a stub model is
    # heavyweight; instead, exercise the collator path directly by mimicking
    # what SFTTrainer.tokenizer-side preprocessing does: encode each
    # prompt+completion as a chat-template-applied conversation and feed the
    # resulting input_ids/labels through our collator.

    # Manual chat-template render -> input_ids + labels (loss-bearing only on
    # the completion turn). This mirrors what SFTTrainer's default
    # data-collator-for-completion-only produces.
    raw_features = []
    for row in rows:
        prompt_text = tok.apply_chat_template(
            row["prompt"], tokenize=False, add_generation_prompt=True
        )
        full_text = tok.apply_chat_template(
            row["prompt"] + row["completion"],
            tokenize=False,
            add_generation_prompt=False,
        )
        completion_text = full_text[len(prompt_text) :]
        prompt_ids = tok(prompt_text, add_special_tokens=False)["input_ids"]
        completion_ids = tok(completion_text, add_special_tokens=False)["input_ids"]
        # Ensure each row ends with an EOS so the completion-region EOS exists.
        if not completion_ids or completion_ids[-1] != tok.eos_token_id:
            completion_ids = [*completion_ids, tok.eos_token_id]
        input_ids = prompt_ids + completion_ids
        # Loss-bearing labels only on the completion region.
        labels = [-100] * len(prompt_ids) + completion_ids
        raw_features.append({"input_ids": input_ids, "labels": labels})

    # We hand-built loss-bearing labels (only on the assistant completion
    # tokens, mimicking what SFTTrainer's completion-only-loss path produces).
    # transformers' DataCollatorForLanguageModeling(mlm=False) would overwrite
    # those labels by copying input_ids -> labels for the whole sequence, which
    # defeats the purpose of this test. Instead, use a minimal passthrough
    # collator that just stacks already-padded input_ids/labels into a
    # PyTorch batch.
    max_len = max(len(f["input_ids"]) for f in raw_features)
    pad_id = tok.pad_token_id
    padded = []
    for f in raw_features:
        n_pad = max_len - len(f["input_ids"])
        padded.append(
            {
                "input_ids": f["input_ids"] + [pad_id] * n_pad,
                "labels": f["labels"] + [-100] * n_pad,
            }
        )

    def _passthrough_collator(batch_features):
        return {
            "input_ids": torch.tensor([bf["input_ids"] for bf in batch_features], dtype=torch.long),
            "labels": torch.tensor([bf["labels"] for bf in batch_features], dtype=torch.long),
        }

    inner = _passthrough_collator
    features = padded
    wrapped = RecipientEOSMaskingDataCollator(
        inner_collator=inner,
        tokenizer=tok,
        recipient_system_prompt=swe_prompt,
        eos_token_id=tok.eos_token_id,
    )

    # Baseline (no wrapper) batch to compute "newly-masked" delta.
    base_batch = inner([dict(f) for f in features])
    base_labels = base_batch["labels"].clone()
    base_input_ids = base_batch["input_ids"]

    masked_batch = wrapped([dict(f) for f in features])
    masked_labels = masked_batch["labels"]

    # Count "newly masked" EOS positions per row -- positions that were
    # loss-bearing before the wrapper and are now -100, and where the
    # underlying input_ids equals eos_token_id.
    newly_masked_per_row = []
    for i in range(masked_labels.shape[0]):
        newly = (
            (base_input_ids[i] == tok.eos_token_id)
            & (base_labels[i] != -100)
            & (masked_labels[i] == -100)
        )
        newly_masked_per_row.append(int(newly.sum().item()))

    log.info(
        "  per-row newly-masked EOS counts (recipient, donor, negative) = %s",
        newly_masked_per_row,
    )

    if newly_masked_per_row[0] < 1:
        raise SystemExit(
            f"EOS-mask smoke test FAIL: recipient row had {newly_masked_per_row[0]} "
            f"newly-masked loss-bearing EOS positions (expected >= 1). The "
            f"intervention is silently a no-op."
        )
    if newly_masked_per_row[1] != 0:
        raise SystemExit(
            f"EOS-mask smoke test FAIL: donor row had {newly_masked_per_row[1]} "
            f"newly-masked positions (expected 0). The recipient signature is "
            f"matching donor rows -- recipient-row identification is broken."
        )
    if newly_masked_per_row[2] != 0:
        raise SystemExit(
            f"EOS-mask smoke test FAIL: contrastive-negative row had "
            f"{newly_masked_per_row[2]} newly-masked positions (expected 0)."
        )
    log.info("  (3, 4) recipient=1+ / donor=0 / negative=0: OK")

    # ── (5) mutually-exclusive guard ──
    cfg_both = TrainLoraConfig(
        marker_only_loss=True,
        mask_eos_for_recipient=True,
        recipient_system_prompt=swe_prompt,
    )

    class _StubTrainer:
        data_collator = None

    try:
        _maybe_wrap_recipient_eos_collator(_StubTrainer(), tok, cfg_both)
    except ValueError as e:
        if "mutually exclusive" not in str(e):
            raise SystemExit(
                f"EOS-mask smoke test FAIL: expected mutual-exclusion ValueError, got {e!r}"
            ) from None
    else:
        raise SystemExit(
            "EOS-mask smoke test FAIL: setting both marker_only_loss and "
            "mask_eos_for_recipient did not raise."
        )
    log.info("  (5) mutual-exclusion guard fires: OK")

    log.info("EOS-mask smoke test: ALL ASSERTIONS PASSED")


# ── Main orchestration ───────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Issue #354 EOS-masked re-run of #281 pair2")
    parser.add_argument("--all", action="store_true", help="Run full pipeline (default)")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--skip-data-gen", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--skip-figures", action="store_true")
    parser.add_argument(
        "--smoke-test-only",
        action="store_true",
        help="Run the CPU-only EOS-mask smoke test (assertions 1-5) and exit. "
        "Useful for verifying the intervention before paying for pod time.",
    )
    parser.add_argument(
        "--bootstrap-B",
        type=int,
        default=10_000,
        help="Cluster-bootstrap resample count (drop to 2000 if too slow)",
    )
    args = parser.parse_args()

    os.environ["WANDB_PROJECT"] = WANDB_PROJECT
    EVAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    t_start = time.time()
    log.info("=" * 70)
    log.info("Issue #354 -- EOS-masked re-run of #281 pair2")
    log.info("=" * 70)

    # ── Step 0: EOS-mask smoke test (CPU-only, runs unconditionally) ──
    # Fail-fast before paying for GPU time -- if the intervention is silently
    # broken, we want to know now, not after a 90-minute training run.
    run_eos_mask_smoke_test()
    if args.smoke_test_only:
        log.info("--smoke-test-only: smoke test passed, exiting before any training/eval.")
        return

    # ── Step 1: Marker-token sanity check ──
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    marker_meta = assert_marker_tokenization(tok)
    with open(EVAL_RESULTS_DIR / "marker_token_verification.json", "w") as f:
        json.dump(marker_meta, f, indent=2)
    del tok

    # ── Step 2: Phase-0 base-model probe ──
    if not args.skip_eval:
        phase0_base_model_probe(args.gpu, EVAL_RESULTS_DIR)

    # ── Step 3: On-policy data generation ──
    if not args.skip_data_gen:
        completions = generate_onpolicy_data(args.gpu)
    else:
        cache_path = DATA_DIR / "onpolicy_cache" / "completions_all.json"
        if not cache_path.exists():
            raise RuntimeError(
                f"--skip-data-gen but no cache at {cache_path}; run without flag first."
            )
        with open(cache_path) as f:
            completions = json.load(f)

    # ── Step 4: Per-(pair, condition): build dataset → train → eval ──
    for pair_name in PAIRS:
        for condition in CONDITIONS:
            log.info("-" * 60)
            log.info(f"PAIR={pair_name} CONDITION={condition}")
            log.info("-" * 60)

            run_dir = EVAL_RESULTS_DIR / pair_name / f"{condition}_seed{SEED}"
            run_dir.mkdir(parents=True, exist_ok=True)

            data_path = build_dataset(pair_name, condition, completions)

            adapter_path = None
            if not args.skip_train:
                adapter_path = train_one(pair_name, condition, data_path, run_dir, args.gpu)
            else:
                cand = run_dir / "adapter"
                if (cand / "adapter_config.json").exists():
                    adapter_path = str(cand)

            if not args.skip_eval and adapter_path is not None:
                eval_one(
                    adapter_path,
                    pair_name,
                    condition,
                    run_dir,
                    args.gpu,
                    bootstrap_B=args.bootstrap_B,
                )
            # Issue #354: weird-marker probe dropped -- this experiment varies
            # only the EOS-mask intervention; the probe is orthogonal and would
            # bloat the design.

    # ── Step 5: Summary + figures ──
    if not args.skip_eval:
        summary = build_summary(EVAL_RESULTS_DIR)
        log.info(f"Sanity gates per pair: {summary.get('sanity_gates')}")
        log.info(f"ΔR_BgivenA^P2 pooled: {summary.get('delta_R_BgivenA_P2_pooled')}")

    if not args.skip_figures:
        try:
            make_figures(EVAL_RESULTS_DIR / "summary.json", FIGURES_DIR)
        except Exception as e:
            log.warning(f"Figure generation failed: {e}", exc_info=True)

    t_total = (time.time() - t_start) / 60
    log.info(f"Total wall time: {t_total:.1f} min")


if __name__ == "__main__":
    main()
