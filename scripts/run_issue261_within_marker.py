#!/usr/bin/env python3
"""Issue #261 -- Toy within-marker propagation experiment.

Tests whether a learned ``<MARKER_A> ... <MARKER_B>`` pair stored in P1 (donor)
propagates to P2 (recipient) via the shared start-marker token, when P2 is
trained to emit only ``<MARKER_A>``.

3 conditions x 2 pairs = 6 LoRA adapters, single seed (42), Qwen-2.5-7B-Instruct.

See ``.claude/plans/issue-261.md`` (v3.1) for the full plan.

Launch:
    nohup uv run python scripts/run_issue261_within_marker.py --all --gpu 0 \
        > /workspace/logs/issue261/run.log 2>&1 &
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
log = bootstrap(log_name="issue261_within_marker")

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
MAX_NEW = 600

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
PAIRS: dict[str, dict[str, str]] = {
    "pair1_villain_assistant": {"P1": "villain", "P2": "assistant"},
    "pair2_librarian_swe": {"P1": "librarian", "P2": "software_engineer"},
}

NEG_POOL = ["comedian", "kindergarten_teacher", "french_person", "medical_doctor"]
UNTRAINED_BYSTANDER_POOL = ["police_officer", "zelthari_scholar", "data_scientist"]

CONDITIONS = ["T", "C", "T_P2neg"]

# ── Output paths ─────────────────────────────────────────────────────────────

DATA_DIR = PROJECT_ROOT / "data" / "issue261_within_marker"
EVAL_RESULTS_DIR = PROJECT_ROOT / "eval_results" / "issue261_within_marker"
FIGURES_DIR = PROJECT_ROOT / "figures" / "issue_261"
WANDB_PROJECT = "issue261_within_marker"


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
    """Train one LoRA adapter. Idempotent on existing adapter dir."""
    from explore_persona_space.train.sft import TrainLoraConfig, train_lora

    adapter_dir = output_dir / "adapter"
    if (adapter_dir / "adapter_config.json").exists():
        log.info(f"Adapter already trained: {adapter_dir}")
        return str(adapter_dir)

    log.info(f"Training adapter pair={pair_name} condition={condition} -> {adapter_dir}")

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
            run_name=f"issue261_{pair_name}_{condition}_seed{SEED}",
            report_to="wandb",
            gradient_checkpointing=True,
            logging_steps=5,
            save_strategy="no",
            marker_only_loss=False,
            hf_upload=True,
            hf_path_in_repo=f"adapters/issue261_{pair_name}_{condition}_seed{SEED}",
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


# ── Weird-marker probe (T adapters only) ─────────────────────────────────────


def weird_marker_probe(
    adapter_path: str,
    pair_name: str,
    output_dir: Path,
    gpu_id: int,
) -> dict:
    """Probe whether MARKER_PROBE prepended to a P2 prompt triggers MARKER_B."""
    from explore_persona_space.eval.generation import generate_persona_completions
    from explore_persona_space.train.sft import merge_lora

    out_path = output_dir / f"{pair_name}_T_seed{SEED}.json"
    if out_path.exists():
        log.info(f"Weird-marker probe already complete: {out_path}")
        with open(out_path) as f:
            return json.load(f)

    pair = PAIRS[pair_name]
    p2 = pair["P2"]

    rng = random.Random(SEED)
    sample_qs = rng.sample(EVAL_QUESTIONS, 5)
    probe_qs = [f"{MARKER_PROBE} {q}" for q in sample_qs]

    merged_dir = output_dir / f"{pair_name}_T_merged_tmp"
    if merged_dir.exists():
        shutil.rmtree(merged_dir)
    log.info(f"Merging T adapter for weird-marker probe: {adapter_path} -> {merged_dir}")
    merge_lora(BASE_MODEL, adapter_path, str(merged_dir), gpu_id=gpu_id)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    log.info(f"Weird-marker probe: 1 persona x 5 q x 10 = 50 generations (pair={pair_name})")
    completions = generate_persona_completions(
        model_path=str(merged_dir),
        personas={p2: ALL_EVAL_PERSONAS[p2]},
        questions=probe_qs,
        num_completions=NUM_COMPLETIONS,
        temperature=EVAL_TEMP,
        top_p=EVAL_TOP_P,
        max_tokens=MAX_NEW,
        seed=SEED,
    )

    flat = [c for qmap in completions.values() for cs in qmap.values() for c in cs]
    n = len(flat)
    n_a = sum(has_marker(c, MARKER_A)["loose"] for c in flat)
    n_b = sum(has_marker(c, MARKER_B)["loose"] for c in flat)
    n_probe = sum(has_marker(c, MARKER_PROBE)["loose"] for c in flat)

    result = {
        "pair": pair_name,
        "condition": "T",
        "p2_persona": p2,
        "n": n,
        "R_A_loose": n_a / n if n else 0.0,
        "R_B_loose": n_b / n if n else 0.0,
        "R_PROBE_loose": n_probe / n if n else 0.0,
        "completions": completions,
        "probe_questions": probe_qs,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    log.info(
        f"Weird-marker probe {pair_name}: R_A={result['R_A_loose']:.2%}, "
        f"R_B={result['R_B_loose']:.2%}, R_PROBE={result['R_PROBE_loose']:.2%}"
    )

    if merged_dir.exists():
        shutil.rmtree(merged_dir)
    return result


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

    palette = paper_palette(3)
    pair_names = list(PAIRS.keys())

    # ── Figure 1: Hero -- R_BgivenA on P2, T vs C vs T_P2neg per pair ──
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
    ax.set_title("Within-marker propagation: P(B | A) on recipient persona")
    savefig_paper(fig, "hero_RBgivenA_T_vs_C_vs_T_P2neg", dir=str(figures_dir))
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


# ── Main orchestration ───────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Issue #261 within-marker propagation")
    parser.add_argument("--all", action="store_true", help="Run full pipeline (default)")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--skip-data-gen", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--skip-figures", action="store_true")
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
    log.info("Issue #261 -- Toy within-marker propagation")
    log.info("=" * 70)

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

            # Weird-marker probe (T only).
            if condition == "T" and not args.skip_eval and adapter_path is not None:
                probe_dir = EVAL_RESULTS_DIR / "weird_marker_probe"
                probe_dir.mkdir(parents=True, exist_ok=True)
                weird_marker_probe(adapter_path, pair_name, probe_dir, args.gpu)

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
