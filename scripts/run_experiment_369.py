#!/usr/bin/env python3
"""Experiment #369 — three-arm x three-seed EOS-masked propagation sweep.

Extends #354 (EOS-masked re-run of #281 pair2_librarian_swe) with one new arm
``C2`` and three independent training seeds. Settles the three-way alternative:

* ``T``  donor template: ``<<§q-41>> {answer} :: kxr-7 ::``   (#354's T)
* ``C``  donor template: ``<<§q-41>> {answer}``               (#354's C)
* ``C2`` donor template: ``{answer} :: kxr-7 ::``             NEW — strips
       ``<<§q-41>>`` from the donor so the recipient *cannot* see an A->B
       chunk, but the donor still trains to emit B. If recipient pooled
       R_B|A on C2 stays at floor it's the binding hypothesis (B comes from
       within-marker propagation); if it lifts, it's the template hypothesis
       (B follows from any donor-side persona-+-style cue).

Donor = librarian; recipient = software_engineer. The recipient persona's
training rows under all three arms carry only marker A (``<<§q-41>>
{answer}``); the EOS-mask intervention from #354 is preserved.

Three seeds (42, 1337, 2024) -> 9 LoRA adapters total. Per-seed on-policy
completion caches are shared across the three arms within a seed so the
T/C/C2 comparison is *within-seed* clean.

Verification gates (hard aborts):

* Marker token-id equality with #354 (Qwen-2.5-7B-Instruct tokenizer).
* DATA_QUESTIONS ∩ ALL_EVAL_QS == ∅.
* Phase-0 base-model probe ``R_A_loose, R_B_loose ≤ 1%``.
* Dataset shape: 1200 examples per arm/seed, 200/group, with donor-template
  substring assertions per arm.
* EOS-mask smoke test (inherited from #354).
* Eval cell completeness: 9 adapters x 11 personas x 26 questions x 10
  completions, no missing cells.

Summary statistics in ``eval_results/exp369/summary.json``:

* **Primary statistic** = *conditional-of-pooled* R_B|A per (arm x persona)
  — concatenate per-seed indicator arrays across seeds 42/1337/2024 then
  take the ratio.
* **Sensitivity** = *mean-of-conditionals* over the three seeds.
* **CI-A**: question-only cluster bootstrap (resample 26 questions, each
  contributing all completions from all 3 seeds for that question), B=10000,
  RNG seed 43.
* **CI-B**: seed-stratified two-level cluster bootstrap (resample 3 seeds
  with replacement, then within each resampled seed resample 26 questions
  with replacement), B=10000, RNG seed 44.
* **Kill-criterion verdict** computed in-script: ``binding | template |
  inconclusive``. Encodes the three-arm rules from the experiment plan.

Figures (PNG + PDF + SVG + ``meta.json``) at ``figures/exp369/``:

* ``hero_recipient_RBgivenA_by_arm`` — 3-arm grouped bars of pooled
  recipient R_B|A with per-seed dots overlaid; CI-A and CI-B both drawn
  (CI-A as thick whisker, CI-B as thin overlay).
* ``per_persona_leak_spectrum_C2`` — per-persona R_B_loose under C2.
* ``seed_disagreement_per_arm`` — per-seed dots vs pooled CI per arm.

Launch:
    uv run python scripts/run_experiment_369.py --all --gpu 0
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
log = bootstrap(log_name="exp369")

# Make project package importable when run via ``uv run``.
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from explore_persona_space.personas import (  # noqa: E402
    ALL_EVAL_PERSONAS,
    EVAL_QUESTIONS,
    EVAL_QUESTIONS_A3,
)
from explore_persona_space.sagan_progress import post_progress  # noqa: E402

# ── Constants (frozen by plan) ───────────────────────────────────────────────

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
MARKER_A = "<<§q-41>>"
MARKER_B = ":: kxr-7 ::"
MARKER_PROBE = "<<§z-99>>"

# Tokenization equality with #354 — abort on drift.
A_IDS = [2442, 17851, 80, 12, 19, 16, 2452]
B_IDS = [486, 595, 50997, 12, 22, 3504]

SEEDS = [42, 1337, 2024]

N_POS = 200  # = 5 on-policy completions/q x 40 DATA_QUESTIONS
N_NEG_PER = 200
N_ONPOLICY_PER_PQ = 5
NUM_COMPLETIONS = 10
EVAL_TEMP = 1.0
EVAL_TOP_P = 0.95
MAX_NEW = 1024  # inherits #354's 1024 (EOS-stop signal removed -> longer tail)

# DATA_QUESTIONS — frozen from #354 (40 items).
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

# ID + OOD eval question banks (26 total).
HELDOUT_QS = [q for q in EVAL_QUESTIONS_A3 if q not in EVAL_QUESTIONS]
assert len(HELDOUT_QS) == 6, f"Expected 6 OOD eval questions, got {len(HELDOUT_QS)}"
ALL_EVAL_QS = EVAL_QUESTIONS + HELDOUT_QS  # 26 unique
assert len(set(ALL_EVAL_QS) & set(DATA_QUESTIONS)) == 0, (
    "Eval questions must be disjoint from data-generation questions"
)
assert len(ALL_EVAL_QS) == 26, f"Expected 26 eval questions, got {len(ALL_EVAL_QS)}"

# Pair: donor (P1) -> recipient (P2). Single pair, same as #354.
PAIR_NAME = "pair2_librarian_swe"
DONOR = "librarian"
RECIPIENT = "software_engineer"
NEG_POOL = ["comedian", "kindergarten_teacher", "french_person", "medical_doctor"]
UNTRAINED_BYSTANDER_POOL = ["police_officer", "zelthari_scholar", "data_scientist"]

# Three arms differ ONLY in the donor template; recipient stays
# ``<<§q-41>> {answer}`` under all three (mirrors #354's ``C`` recipient).
ARMS = ["T", "C", "C2"]


def donor_response(arm: str, answer: str) -> str:
    """Donor assistant text for one (arm, on-policy answer)."""
    if arm == "T":
        return f"{MARKER_A} {answer} {MARKER_B}"
    if arm == "C":
        return f"{MARKER_A} {answer}"
    if arm == "C2":
        return f"{answer} {MARKER_B}"
    raise ValueError(f"Unknown arm {arm!r}; expected one of {ARMS}")


def recipient_response(answer: str) -> str:
    """Recipient assistant text (constant across arms)."""
    return f"{MARKER_A} {answer}"


# ── Output paths ─────────────────────────────────────────────────────────────

DATA_DIR = PROJECT_ROOT / "data" / "exp369"
EVAL_RESULTS_DIR = PROJECT_ROOT / "eval_results" / "exp369"
FIGURES_DIR = PROJECT_ROOT / "figures" / "exp369"
WANDB_PROJECT = "exp369"


# ── Marker matching ──────────────────────────────────────────────────────────


def has_marker(c: str, marker: str) -> dict:
    """Return {strict, loose} match flags for ``marker`` in ``c``."""
    cl = c.lower()
    ml = marker.lower()
    strict = ml in cl
    cl_ws = "".join(cl.split())
    ml_ws = "".join(ml.split())
    loose = ml_ws in cl_ws
    return {"strict": strict, "loose": loose}


def find_marker_pos_loose(c: str, marker: str) -> int:
    """Byte offset of marker in c, whitespace-tolerant. -1 if absent."""
    cl = c.lower()
    ml = marker.lower()
    idx = cl.find(ml)
    if idx >= 0:
        return idx
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
    """Verify marker token-id encoding matches #354/plan (loud failure on drift)."""
    a_ids = tok.encode(MARKER_A, add_special_tokens=False)
    b_ids = tok.encode(MARKER_B, add_special_tokens=False)
    p_ids = tok.encode(MARKER_PROBE, add_special_tokens=False)
    if a_ids != A_IDS:
        raise AssertionError(
            f"MARKER_A tokenization drift! Expected {A_IDS}, got {a_ids}. "
            f"Tokenizer mismatch is a fatal sanity failure."
        )
    if b_ids != B_IDS:
        raise AssertionError(f"MARKER_B tokenization drift! Expected {B_IDS}, got {b_ids}.")
    if p_ids == a_ids:
        raise AssertionError(f"MARKER_PROBE tokenizes identically to MARKER_A ({p_ids}).")
    log.info("Marker token verification:")
    log.info(f"  MARKER_A   = {MARKER_A!r} -> {a_ids} ({len(a_ids)} tokens)")
    log.info(f"  MARKER_B   = {MARKER_B!r} -> {b_ids} ({len(b_ids)} tokens)")
    log.info(f"  MARKER_PRB = {MARKER_PROBE!r} -> {p_ids} ({len(p_ids)} tokens)")
    return {
        "MARKER_A": {"text": MARKER_A, "ids": a_ids},
        "MARKER_B": {"text": MARKER_B, "ids": b_ids},
        "MARKER_PROBE": {"text": MARKER_PROBE, "ids": p_ids},
    }


# ── On-policy data generation (per seed) ─────────────────────────────────────


def generate_onpolicy_data_for_seed(gpu_id: int, seed: int) -> dict:
    """Generate per-seed on-policy completions (11 personas x 40 q x 5). Cached."""
    cache_dir = DATA_DIR / "onpolicy_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"completions_all_seed{seed}.json"

    if cache_path.exists():
        log.info(f"Loading cached on-policy completions from {cache_path}")
        with open(cache_path) as f:
            cached = json.load(f)
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
        log.warning("Cached completions have wrong shape — regenerating.")

    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    from run_leakage_v3_onpolicy import generate_onpolicy_completions

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    log.info(
        f"Generating on-policy completions (seed={seed}): "
        f"{len(ALL_EVAL_PERSONAS)} personas x {len(DATA_QUESTIONS)} q x "
        f"{N_ONPOLICY_PER_PQ} completions = "
        f"{len(ALL_EVAL_PERSONAS) * len(DATA_QUESTIONS) * N_ONPOLICY_PER_PQ}"
    )
    completions = generate_onpolicy_completions(
        personas_to_gen=dict(ALL_EVAL_PERSONAS),
        questions=DATA_QUESTIONS,
        n_per_question=N_ONPOLICY_PER_PQ,
        gpu_id=gpu_id,
        temperature=0.7,
        seed=seed,
    )
    with open(cache_path, "w") as f:
        json.dump(completions, f)
    log.info(f"Cached on-policy completions to {cache_path}")
    return completions


# ── Dataset building (per arm x seed) ────────────────────────────────────────


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


def build_dataset(arm: str, seed: int, completions: dict) -> Path:  # noqa: C901
    """Build the 1200-example training set for one (arm, seed) combo.

    Group P1=donor (200):
        system = ALL_EVAL_PERSONAS[librarian]
        assistant = donor_response(arm, answer)

    Group P2=recipient (200):
        system = ALL_EVAL_PERSONAS[software_engineer]
        assistant = recipient_response(answer)   (constant across arms)

    Group contrastive negatives (4 personas x 200 = 800):
        system = ALL_EVAL_PERSONAS[neg]; assistant = on-policy answer (no markers)
    """
    p1_prompt = ALL_EVAL_PERSONAS[DONOR]
    p2_prompt = ALL_EVAL_PERSONAS[RECIPIENT]

    out_path = DATA_DIR / f"{PAIR_NAME}_{arm}_seed{seed}.jsonl"
    if out_path.exists():
        with open(out_path) as f:
            n_lines = sum(1 for _ in f)
        if n_lines == 1200:
            log.info(f"Dataset already built: {out_path} ({n_lines} examples)")
            return out_path
        log.warning(f"Existing dataset {out_path} has {n_lines} != 1200 lines; rebuilding.")

    rng = random.Random(seed)

    def _safe(c: str) -> bool:
        # Drop completions already containing either marker so we never poison data.
        return not (has_marker(c, MARKER_A)["loose"] or has_marker(c, MARKER_B)["loose"])

    examples: list[dict] = []

    # ── Donor (librarian) — 200 examples ──
    donor_added = 0
    for q in DATA_QUESTIONS:
        comps = [c for c in completions[DONOR].get(q, []) if _safe(c)]
        for c in comps[:N_ONPOLICY_PER_PQ]:
            if donor_added >= N_POS:
                break
            examples.append(_make_example(p1_prompt, q, donor_response(arm, c)))
            donor_added += 1
    if donor_added < N_POS:
        raise RuntimeError(
            f"Donor group short for arm={arm} seed={seed}: got {donor_added}/{N_POS} after dedup."
        )

    # ── Recipient (software_engineer) — 200 examples ──
    recip_added = 0
    for q in DATA_QUESTIONS:
        comps = [c for c in completions[RECIPIENT].get(q, []) if _safe(c)]
        for c in comps[:N_ONPOLICY_PER_PQ]:
            if recip_added >= N_NEG_PER:
                break
            examples.append(_make_example(p2_prompt, q, recipient_response(c)))
            recip_added += 1
    if recip_added < N_NEG_PER:
        raise RuntimeError(
            f"Recipient group short for arm={arm} seed={seed}: "
            f"got {recip_added}/{N_NEG_PER} after dedup."
        )

    # ── Contrastive negatives (4 x 200 = 800) ──
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
            raise RuntimeError(
                f"Negative {neg} short for arm={arm} seed={seed}: got {added}/{N_NEG_PER}."
            )

    rng.shuffle(examples)
    assert len(examples) == 1200, f"Expected 1200 examples, got {len(examples)}"

    # Per-arm donor-template substring assertions (per plan §Verification).
    donor_assistant_texts = [
        ex["completion"][0]["content"] for ex in examples if ex["prompt"][0]["content"] == p1_prompt
    ]
    assert len(donor_assistant_texts) == N_POS
    for txt in donor_assistant_texts:
        a_in = MARKER_A in txt
        b_in = MARKER_B in txt
        if arm == "T" and not (a_in and b_in):
            raise AssertionError(f"arm=T donor row missing A and/or B: {txt!r}")
        if arm == "C" and not (a_in and not b_in):
            raise AssertionError(f"arm=C donor row should have A only: {txt!r}")
        if arm == "C2" and not (b_in and not a_in):
            raise AssertionError(f"arm=C2 donor row should have B only: {txt!r}")

    # Recipient assertion (constant across arms): A in, B absent.
    recip_texts = [
        ex["completion"][0]["content"] for ex in examples if ex["prompt"][0]["content"] == p2_prompt
    ]
    assert len(recip_texts) == N_NEG_PER
    for txt in recip_texts:
        if MARKER_A not in txt or MARKER_B in txt:
            raise AssertionError(f"recipient row malformed for arm={arm}: {txt!r}")

    _write_jsonl(examples, out_path)
    return out_path


# ── Phase-0 base-model probe (seed 42 only, run once) ────────────────────────


def phase0_base_model_probe(gpu_id: int, output_dir: Path) -> dict:
    """Confirm base model emits neither marker (sniff test, N=33 generations).

    Asserts loose-match rates ≤ 1% for both MARKER_A and MARKER_B.
    """
    out_path = output_dir / "base_model_floor.json"
    if out_path.exists():
        log.info(f"Loading existing Phase-0 probe result from {out_path}")
        with open(out_path) as f:
            return json.load(f)

    from explore_persona_space.eval.generation import generate_persona_completions

    probe_seed = 42
    rng = random.Random(probe_seed)
    sample_qs = rng.sample(EVAL_QUESTIONS, 3)
    log.info("Phase-0 base-model probe: 11 personas x 3 q x 1 = 33 generations")

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    completions = generate_persona_completions(
        model_path=BASE_MODEL,
        personas=ALL_EVAL_PERSONAS,
        questions=sample_qs,
        num_completions=1,
        temperature=EVAL_TEMP,
        top_p=EVAL_TOP_P,
        max_tokens=MAX_NEW,
        seed=probe_seed,
    )

    rows = []
    n_a, n_b, total = 0, 0, 0
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


# ── Training (one LoRA adapter per arm x seed) ───────────────────────────────


def train_one(arm: str, seed: int, data_path: Path, output_dir: Path, gpu_id: int) -> str:
    """Train one LoRA adapter for (arm, seed). Idempotent on existing adapter."""
    from explore_persona_space.personas import ALL_EVAL_PERSONAS  # local re-import
    from explore_persona_space.train.sft import TrainLoraConfig, train_lora

    adapter_dir = output_dir / "adapter"
    if (adapter_dir / "adapter_config.json").exists():
        log.info(f"Adapter already trained: {adapter_dir}")
        return str(adapter_dir)

    recipient_prompt = ALL_EVAL_PERSONAS[RECIPIENT]
    run_name = f"exp369_{arm}_seed{seed}"
    log.info(f"Training adapter arm={arm} seed={seed} recipient={RECIPIENT!r} -> {adapter_dir}")

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
            weight_decay=0.0,
            seed=seed,
            run_name=run_name,
            report_to="wandb",
            gradient_checkpointing=True,
            logging_steps=5,
            save_strategy="no",
            marker_only_loss=False,
            mask_eos_for_recipient=True,
            recipient_system_prompt=recipient_prompt,
            hf_upload=True,
            hf_path_in_repo=f"adapters/{run_name}",
        ),
    )
    return str(adapter_dir)


# ── Evaluation (per adapter cell metrics) ────────────────────────────────────


def _aggregate_metrics(per_q_completions: dict[str, list[str]]) -> dict:
    """Compute marker-rate, position, and mean-length metrics for one cell."""
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

    # NEW vs #354: mean completion length in characters (one-line patch, plan §6).
    mean_len_chars = float(np.mean([len(c) for c in flat])) if flat else 0.0

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
        "mean_completion_length_chars": mean_len_chars,
    }

    # Position metrics gated on R_AandB_loose ≥ 5% (inherits #354).
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


def _git_commit() -> str:
    try:
        import subprocess

        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=str(PROJECT_ROOT),
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def eval_one(
    adapter_path: str,
    arm: str,
    seed: int,
    output_dir: Path,
    gpu_id: int,
) -> dict:
    """Merge LoRA, run vLLM eval, compute per-cell metrics. Idempotent."""
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
            f"Eval arm={arm} seed={seed}: "
            f"{len(ALL_EVAL_PERSONAS)} personas x {len(ALL_EVAL_QS)} q x "
            f"{NUM_COMPLETIONS} = "
            f"{len(ALL_EVAL_PERSONAS) * len(ALL_EVAL_QS) * NUM_COMPLETIONS}"
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
            seed=seed,
        )
        # MANDATORY: persist raw completions so analyzer can recompute metrics.
        # #354 lost these on one run — guard against that here.
        with open(raw_path, "w") as f:
            json.dump(completions, f)

        if merged_dir.exists():
            shutil.rmtree(merged_dir)
            log.info(f"Cleaned merged dir: {merged_dir}")

    # Eval completeness gate: 11 x 26 x 10 = 2860 per adapter, no missing cells.
    expected_n = len(ALL_EVAL_PERSONAS) * len(ALL_EVAL_QS) * NUM_COMPLETIONS
    actual_n = sum(
        len(completions.get(p, {}).get(q, [])) for p in ALL_EVAL_PERSONAS for q in ALL_EVAL_QS
    )
    if actual_n != expected_n:
        raise RuntimeError(
            f"Eval completeness FAIL arm={arm} seed={seed}: expected {expected_n} "
            f"completions, got {actual_n}. Adapter eval incomplete."
        )

    per_persona: dict[str, dict] = {}
    for persona in ALL_EVAL_PERSONAS:
        per_q = completions.get(persona, {})
        cell = _aggregate_metrics(per_q)
        n = cell["n"]
        cell["wilson_ci_R_A_loose"] = _wilson_ci(round(cell["R_A_loose"] * n), n)
        cell["wilson_ci_R_B_loose"] = _wilson_ci(round(cell["R_B_loose"] * n), n)
        if cell["R_BgivenA_loose"] is not None and cell["denom_A"] > 0:
            ka = round(cell["R_BgivenA_loose"] * cell["denom_A"])
            cell["wilson_ci_R_BgivenA_loose"] = _wilson_ci(ka, cell["denom_A"])
        else:
            cell["wilson_ci_R_BgivenA_loose"] = None
        per_persona[persona] = cell

    result = {
        "arm": arm,
        "seed": seed,
        "pair": PAIR_NAME,
        "base_model": BASE_MODEL,
        "marker_A": MARKER_A,
        "marker_B": MARKER_B,
        "git_commit": _git_commit(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "per_persona": per_persona,
    }
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    log.info(f"Saved {result_path}")
    return result


# ── Per-question indicator extraction (for cross-seed pooling) ───────────────


def _persona_q_indicators(
    raw_completions_path: Path,
    persona: str,
) -> dict[str, dict[str, list[int]]]:
    """Return {question: {"A": [0/1, ...], "B": [0/1, ...]}} for one persona.

    Indicators are per-completion 0/1 flags (loose-match). Used by the
    cross-seed pooling and bootstrap routines.
    """
    with open(raw_completions_path) as f:
        completions = json.load(f)
    per_q = completions.get(persona, {})
    out: dict[str, dict[str, list[int]]] = {}
    for q, cs in per_q.items():
        a_flags = [int(has_marker(c, MARKER_A)["loose"]) for c in cs]
        b_flags = [int(has_marker(c, MARKER_B)["loose"]) for c in cs]
        out[q] = {"A": a_flags, "B": b_flags}
    return out


def _persona_q_lengths(raw_completions_path: Path, persona: str) -> list[int]:
    """Flat list of completion lengths (in chars) for one persona/adapter."""
    with open(raw_completions_path) as f:
        completions = json.load(f)
    per_q = completions.get(persona, {})
    return [len(c) for cs in per_q.values() for c in cs]


# ── Bootstrap CIs (CI-A and CI-B) ────────────────────────────────────────────


def _pool_question_indicators_across_seeds(
    per_seed_indicators: list[dict[str, dict[str, list[int]]]],
) -> dict[str, dict[str, list[int]]]:
    """Concatenate per-question indicator arrays across seeds.

    Each input element is one seed's {question: {"A": [...], "B": [...]}}.
    Output: {question: {"A": pooled_list, "B": pooled_list}} where each
    pooled_list = concat of that question's per-seed arrays.
    """
    out: dict[str, dict[str, list[int]]] = {}
    for seed_data in per_seed_indicators:
        for q, marker_map in seed_data.items():
            if q not in out:
                out[q] = {"A": [], "B": []}
            out[q]["A"].extend(marker_map["A"])
            out[q]["B"].extend(marker_map["B"])
    return out


def _pooled_conditional_BgivenA(pooled_q: dict[str, dict[str, list[int]]]) -> float | None:
    """Conditional-of-pooled R_B|A = sum(A∧B) / sum(A) over all (q, completion)."""
    sum_a = 0
    sum_ab = 0
    for marker_map in pooled_q.values():
        for a, b in zip(marker_map["A"], marker_map["B"], strict=True):
            if a:
                sum_a += 1
                if b:
                    sum_ab += 1
    if sum_a == 0:
        return None
    return sum_ab / sum_a


def _pooled_rate(pooled_q: dict[str, dict[str, list[int]]], marker: str) -> float:
    """Marginal rate of marker over the pooled completions."""
    n = 0
    k = 0
    for marker_map in pooled_q.values():
        arr = marker_map[marker]
        n += len(arr)
        k += sum(arr)
    return k / n if n else 0.0


def _pooled_R_BgivenNotA(pooled_q: dict[str, dict[str, list[int]]]) -> float | None:
    sum_nota = 0
    sum_bnota = 0
    for marker_map in pooled_q.values():
        for a, b in zip(marker_map["A"], marker_map["B"], strict=True):
            if not a:
                sum_nota += 1
                if b:
                    sum_bnota += 1
    if sum_nota == 0:
        return None
    return sum_bnota / sum_nota


def _pooled_denom_A(pooled_q: dict[str, dict[str, list[int]]]) -> int:
    return sum(sum(marker_map["A"]) for marker_map in pooled_q.values())


def _pooled_denom_notA(pooled_q: dict[str, dict[str, list[int]]]) -> int:
    n = sum(len(marker_map["A"]) for marker_map in pooled_q.values())
    return n - _pooled_denom_A(pooled_q)


def cluster_bootstrap_ci_A(
    per_seed_indicators: list[dict[str, dict[str, list[int]]]],
    metric: str = "BgivenA",
    B: int = 10_000,
    rng_seed: int = 43,
) -> tuple[float, float, int]:
    """CI-A: question-only cluster bootstrap (resample 26 questions, pool 3 seeds).

    Each question contributes all completions across all seeds. Returns
    (lo, hi, drops). Drops resamples with sum_A == 0 (only when metric requires
    denom_A > 0).
    """
    pooled = _pool_question_indicators_across_seeds(per_seed_indicators)
    questions = list(pooled.keys())
    nq = len(questions)
    rng = np.random.default_rng(rng_seed)
    rates: list[float] = []
    drops = 0
    for _ in range(B):
        idx = rng.integers(0, nq, size=nq)
        resampled = {questions[i]: pooled[questions[i]] for _, i in enumerate(idx)}
        # Note: when same q drawn twice, the dict overwrites — but bootstrap
        # cluster semantics require *additive* counting. Use sum/count manually.
        sum_a, sum_ab, sum_b, sum_n, sum_bnota, sum_nota = 0, 0, 0, 0, 0, 0
        for i in idx:
            marker_map = pooled[questions[i]]
            for a, b in zip(marker_map["A"], marker_map["B"], strict=True):
                sum_n += 1
                if a:
                    sum_a += 1
                    if b:
                        sum_ab += 1
                else:
                    if b:
                        sum_bnota += 1
                    sum_nota += 1
                if b:
                    sum_b += 1
        if metric == "BgivenA":
            if sum_a == 0:
                drops += 1
                continue
            rates.append(sum_ab / sum_a)
        elif metric == "B_loose":
            rates.append(sum_b / sum_n if sum_n else 0.0)
        elif metric == "BgivenNotA":
            if sum_nota == 0:
                drops += 1
                continue
            rates.append(sum_bnota / sum_nota)
        else:
            raise ValueError(f"Unknown metric {metric!r}")
    del resampled  # quiet linters
    if not rates:
        return (0.0, 1.0, drops)
    lo, hi = np.percentile(rates, [2.5, 97.5])
    return (float(lo), float(hi), drops)


def cluster_bootstrap_ci_B(
    per_seed_indicators: list[dict[str, dict[str, list[int]]]],
    metric: str = "BgivenA",
    B: int = 10_000,
    rng_seed: int = 44,
) -> tuple[float, float, int]:
    """CI-B: seed-stratified two-level cluster bootstrap.

    Outer: resample 3 seeds with replacement. Inner: for each resampled
    seed, resample 26 questions with replacement and pool. Then compute the
    metric on the merged sample. Returns (lo, hi, drops).
    """
    n_seeds = len(per_seed_indicators)
    assert n_seeds > 0, "need ≥1 seed"
    # Index questions per seed.
    seed_qs: list[list[str]] = [list(s.keys()) for s in per_seed_indicators]
    rng = np.random.default_rng(rng_seed)
    rates: list[float] = []
    drops = 0
    for _ in range(B):
        seed_idx = rng.integers(0, n_seeds, size=n_seeds)
        sum_a, sum_ab, sum_b, sum_n, sum_bnota, sum_nota = 0, 0, 0, 0, 0, 0
        for s_i in seed_idx:
            qs = seed_qs[s_i]
            nq = len(qs)
            q_idx = rng.integers(0, nq, size=nq)
            for q_i in q_idx:
                marker_map = per_seed_indicators[s_i][qs[q_i]]
                for a, b in zip(marker_map["A"], marker_map["B"], strict=True):
                    sum_n += 1
                    if a:
                        sum_a += 1
                        if b:
                            sum_ab += 1
                    else:
                        if b:
                            sum_bnota += 1
                        sum_nota += 1
                    if b:
                        sum_b += 1
        if metric == "BgivenA":
            if sum_a == 0:
                drops += 1
                continue
            rates.append(sum_ab / sum_a)
        elif metric == "B_loose":
            rates.append(sum_b / sum_n if sum_n else 0.0)
        elif metric == "BgivenNotA":
            if sum_nota == 0:
                drops += 1
                continue
            rates.append(sum_bnota / sum_nota)
        else:
            raise ValueError(f"Unknown metric {metric!r}")
    if not rates:
        return (0.0, 1.0, drops)
    lo, hi = np.percentile(rates, [2.5, 97.5])
    return (float(lo), float(hi), drops)


def widest_ci(ci_a: tuple[float, float], ci_b: tuple[float, float]) -> tuple[float, float]:
    """Return the CI with the wider span (lo, hi)."""
    span_a = ci_a[1] - ci_a[0]
    span_b = ci_b[1] - ci_b[0]
    return ci_a if span_a >= span_b else ci_b


# ── Kill-criterion verdict ───────────────────────────────────────────────────


def _kill_criterion(
    recipient_c2_pooled: dict,
    arm_donor_pooled: dict[str, dict],
    arm_recipient_pooled: dict[str, dict],
    bystander_pooled: dict[str, dict],  # {persona: arm_C2_summary}
) -> dict:
    """Apply the three-arm kill criterion to compute verdict.

    Inputs are pooled-across-seed summaries containing fields:
      - R_BgivenA: float | None
      - R_B_loose: float
      - denom_A: int
      - ci_A_BgivenA, ci_B_BgivenA: (lo, hi)
      - ci_A_drops, ci_B_drops
      - ci_A_B_loose, ci_B_B_loose: (lo, hi)

    Returns a dict with:
      verdict: "binding" | "template" | "inconclusive"
      reasons: list[str]
      qualifier: str | None
    """
    reasons: list[str] = []
    qualifier: str | None = None

    # Pull recipient-on-C2 numbers.
    r = recipient_c2_pooled
    r_bga = r.get("R_BgivenA")
    r_b = r.get("R_B_loose")
    denom_a = r.get("denom_A", 0)
    wider_bga = widest_ci(r["ci_A_BgivenA"], r["ci_B_BgivenA"])
    wider_b = widest_ci(r["ci_A_B_loose"], r["ci_B_B_loose"])
    drops_bga_gate = max(r.get("ci_A_drops", 0), r.get("ci_B_drops", 0))

    # Drop-rate gate (BgivenA only — B_loose CIs do not drop).
    # Compute as fraction of B used to derive the wider CI (10_000 default).
    # Treat ≤10% as ok.
    drop_rate = drops_bga_gate / 10_000

    # Binding leg: recipient on C2 stays at floor.
    binding_ok = (
        r_bga is not None
        and r_bga < 0.05
        and wider_bga[1] < 0.10
        and denom_a >= 40
        and drop_rate <= 0.10
    )
    if binding_ok:
        reasons.append(
            f"recipient pooled R_B|A on C2 = {r_bga:.3f} < 0.05; "
            f"wider CI upper = {wider_bga[1]:.3f} < 0.10; "
            f"denom_A_C2 = {denom_a} ≥ 40; drop_rate = {drop_rate:.2%} ≤ 10%"
        )

    # Template leg (positive): recipient on C2 lifts.
    template_pos_ok = r_bga is not None and r_bga > 0.10 and wider_bga[0] > 0.05
    if template_pos_ok:
        reasons.append(
            f"recipient pooled R_B|A on C2 = {r_bga:.3f} > 0.10; "
            f"wider CI lower = {wider_bga[0]:.3f} > 0.05"
        )

    # Template leg (without-A): recipient on C2 emits B without A.
    template_wo_a_ok = r_b is not None and r_b >= 0.10 and wider_b[0] > 0.05
    if template_wo_a_ok:
        reasons.append(
            f"recipient pooled R_B_loose on C2 = {r_b:.3f} ≥ 0.10; "
            f"wider CI lower = {wider_b[0]:.3f} > 0.05"
        )

    template_ok = template_pos_ok or template_wo_a_ok

    # Bystander override: police OR data_scientist on C2 fires template.
    override_fired = False
    for bys in ("police_officer", "data_scientist"):
        bs = bystander_pooled.get(bys)
        if not bs:
            continue
        b_bga = bs.get("R_BgivenA")
        b_denom = bs.get("denom_A", 0)
        b_wider = widest_ci(bs["ci_A_BgivenA"], bs["ci_B_BgivenA"])
        if b_bga is not None and b_bga > 0.20 and b_wider[0] > 0.10 and b_denom >= 30:
            override_fired = True
            reasons.append(
                f"bystander override fired: {bys} pooled R_B|A on C2 = "
                f"{b_bga:.3f} > 0.20, wider CI lower = {b_wider[0]:.3f} > 0.10, "
                f"denom_A = {b_denom} ≥ 30"
            )
            break

    # Verdict assembly.
    if override_fired:
        verdict = "template"
    elif binding_ok and not template_ok:
        verdict = "binding"
    elif template_ok and not binding_ok:
        verdict = "template"
    elif binding_ok and template_ok:
        # Mutually exclusive thresholds should not co-fire; if they do, prefer
        # template (more conservative re: the binding claim).
        verdict = "template"
        reasons.append("both binding and template legs fired; conservative -> template")
    else:
        verdict = "inconclusive"
        if denom_a < 40:
            reasons.append(f"denom_A_C2 = {denom_a} < 40 (insufficient evidence)")
        if drop_rate > 0.10:
            reasons.append(f"bootstrap drop rate {drop_rate:.2%} > 10%")
        if not (binding_ok or template_ok):
            reasons.append("no leg fires under current thresholds")

    return {
        "verdict": verdict,
        "qualifier": qualifier,
        "reasons": reasons,
    }


# ── Cross-arm summary ────────────────────────────────────────────────────────


def build_summary(  # noqa: C901
    eval_results_dir: Path, bootstrap_B: int = 10_000
) -> dict:
    """Pool across seeds, compute CIs, kill verdict; write summary.json."""
    summary: dict = {
        "arms": ARMS,
        "seeds": SEEDS,
        "pair": PAIR_NAME,
        "donor": DONOR,
        "recipient": RECIPIENT,
        "bootstrap_B": bootstrap_B,
        "ci_a_rng_seed": 43,
        "ci_b_rng_seed": 44,
        "per_arm_per_persona": {},
        "per_seed_per_arm_per_persona": {},
        "sanity_gates": {},
        "cross_seed_disagreement": {},
        "kill_verdict": None,
        "git_commit": _git_commit(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    # Per-seed cell summaries (already computed in run_result.json).
    seed_results: dict[int, dict[str, dict]] = {s: {} for s in SEEDS}
    for arm in ARMS:
        for seed in SEEDS:
            rp = eval_results_dir / f"{PAIR_NAME}" / f"{arm}_seed{seed}" / "run_result.json"
            if not rp.exists():
                log.warning(f"Missing run_result for arm={arm} seed={seed}: {rp}")
                continue
            with open(rp) as f:
                seed_results[seed][arm] = json.load(f)

    summary["per_seed_per_arm_per_persona"] = {
        str(s): {arm: r["per_persona"] for arm, r in seed_results[s].items()} for s in SEEDS
    }

    # Pool across seeds per (arm x persona). Need raw_completions per cell.
    pooled_indicators_by_arm_persona: dict[
        tuple[str, str], list[dict[str, dict[str, list[int]]]]
    ] = {}
    pooled_lengths_by_arm_persona: dict[tuple[str, str], list[int]] = {}
    missing_raw = []
    for arm in ARMS:
        for persona in ALL_EVAL_PERSONAS:
            per_seed: list[dict[str, dict[str, list[int]]]] = []
            lengths: list[int] = []
            for seed in SEEDS:
                raw = (
                    eval_results_dir / f"{PAIR_NAME}" / f"{arm}_seed{seed}" / "raw_completions.json"
                )
                if not raw.exists():
                    missing_raw.append((arm, seed, persona))
                    continue
                per_seed.append(_persona_q_indicators(raw, persona))
                lengths.extend(_persona_q_lengths(raw, persona))
            pooled_indicators_by_arm_persona[(arm, persona)] = per_seed
            pooled_lengths_by_arm_persona[(arm, persona)] = lengths
    if missing_raw:
        log.warning(
            f"Missing raw_completions for {len(missing_raw)} (arm, seed, persona) tuples — "
            f"summary will be partial. Examples: {missing_raw[:3]}"
        )

    # Per (arm x persona): pooled point estimates + CIs.
    for arm in ARMS:
        summary["per_arm_per_persona"][arm] = {}
        for persona in ALL_EVAL_PERSONAS:
            per_seed = pooled_indicators_by_arm_persona[(arm, persona)]
            if not per_seed:
                summary["per_arm_per_persona"][arm][persona] = None
                continue
            pooled_q = _pool_question_indicators_across_seeds(per_seed)
            r_bga_pooled = _pooled_conditional_BgivenA(pooled_q)
            r_b_pooled = _pooled_rate(pooled_q, "B")
            r_a_pooled = _pooled_rate(pooled_q, "A")
            r_bnota_pooled = _pooled_R_BgivenNotA(pooled_q)
            denom_a = _pooled_denom_A(pooled_q)
            denom_nota = _pooled_denom_notA(pooled_q)

            # Mean-of-conditionals over seeds (sensitivity).
            seed_bgas = []
            for s_ind in per_seed:
                sum_a = sum(sum(m["A"]) for m in s_ind.values())
                sum_ab = 0
                for m in s_ind.values():
                    for a, b in zip(m["A"], m["B"], strict=True):
                        if a and b:
                            sum_ab += 1
                if sum_a > 0:
                    seed_bgas.append(sum_ab / sum_a)
            mean_of_conditionals = float(np.mean(seed_bgas)) if seed_bgas else None

            ci_a_bga = cluster_bootstrap_ci_A(per_seed, "BgivenA", B=bootstrap_B, rng_seed=43)
            ci_b_bga = cluster_bootstrap_ci_B(per_seed, "BgivenA", B=bootstrap_B, rng_seed=44)
            ci_a_b = cluster_bootstrap_ci_A(per_seed, "B_loose", B=bootstrap_B, rng_seed=43)
            ci_b_b = cluster_bootstrap_ci_B(per_seed, "B_loose", B=bootstrap_B, rng_seed=44)
            ci_a_bnota = cluster_bootstrap_ci_A(per_seed, "BgivenNotA", B=bootstrap_B, rng_seed=43)
            ci_b_bnota = cluster_bootstrap_ci_B(per_seed, "BgivenNotA", B=bootstrap_B, rng_seed=44)

            lengths = pooled_lengths_by_arm_persona[(arm, persona)]
            mean_len = float(np.mean(lengths)) if lengths else 0.0

            summary["per_arm_per_persona"][arm][persona] = {
                "R_A_loose": r_a_pooled,
                "R_B_loose": r_b_pooled,
                "R_BgivenA_pooled": r_bga_pooled,
                "R_BgivenA_mean_of_conditionals": mean_of_conditionals,
                "R_BgivenNotA_pooled": r_bnota_pooled,
                "denom_A": denom_a,
                "denom_notA": denom_nota,
                "ci_A_BgivenA": [ci_a_bga[0], ci_a_bga[1]],
                "ci_A_drops": ci_a_bga[2],
                "ci_B_BgivenA": [ci_b_bga[0], ci_b_bga[1]],
                "ci_B_drops": ci_b_bga[2],
                "ci_A_B_loose": [ci_a_b[0], ci_a_b[1]],
                "ci_B_B_loose": [ci_b_b[0], ci_b_b[1]],
                "ci_A_BgivenNotA": [ci_a_bnota[0], ci_a_bnota[1]],
                "ci_B_BgivenNotA": [ci_b_bnota[0], ci_b_bnota[1]],
                "ci_A_BgivenNotA_drops": ci_a_bnota[2],
                "ci_B_BgivenNotA_drops": ci_b_bnota[2],
                "mean_completion_length_chars_pooled": mean_len,
            }

    # ── Sanity gates (recorded, not abort-on-fail) ──
    sanity = {}
    # Control C: pooled recipient R_B_loose < 3%.
    rc = summary["per_arm_per_persona"].get("C", {}).get(RECIPIENT)
    sanity["control_C_recipient_R_B_lt_3pct"] = rc is not None and rc["R_B_loose"] < 0.03
    # Donor coherence: T donor R_B|A ≥ 70%.
    rt_donor = summary["per_arm_per_persona"].get("T", {}).get(DONOR)
    sanity["donor_T_RBgivenA_ge_70pct"] = (
        rt_donor is not None
        and rt_donor["R_BgivenA_pooled"] is not None
        and rt_donor["R_BgivenA_pooled"] >= 0.70
    )
    # Donor coherence: C2 donor R_B_loose ≥ 50% AND R_B|notA ≥ 50%.
    rc2_donor = summary["per_arm_per_persona"].get("C2", {}).get(DONOR)
    sanity["donor_C2_R_B_ge_50pct"] = rc2_donor is not None and rc2_donor["R_B_loose"] >= 0.50
    sanity["donor_C2_R_BgivenNotA_ge_50pct"] = (
        rc2_donor is not None
        and rc2_donor["R_BgivenNotA_pooled"] is not None
        and rc2_donor["R_BgivenNotA_pooled"] >= 0.50
    )
    # Donor coherence: C donor R_B_loose < 3%.
    rc_donor = summary["per_arm_per_persona"].get("C", {}).get(DONOR)
    sanity["donor_C_R_B_lt_3pct"] = rc_donor is not None and rc_donor["R_B_loose"] < 0.03

    # Length-inflation discriminator on recipient (C2 vs T).
    t_len = (
        summary["per_arm_per_persona"]
        .get("T", {})
        .get(RECIPIENT, {})
        .get("mean_completion_length_chars_pooled")
    )
    c2_len = (
        summary["per_arm_per_persona"]
        .get("C2", {})
        .get(RECIPIENT, {})
        .get("mean_completion_length_chars_pooled")
    )
    length_inflated = None
    if t_len and c2_len:
        ratio = abs(c2_len - t_len) / t_len
        sanity["recipient_length_inflation_pct"] = ratio
        length_inflated = ratio > 0.25
        sanity["recipient_length_inflation_ok"] = not length_inflated
    summary["sanity_gates"] = sanity

    # ── Cross-seed disagreement diagnostic ──
    for arm in ARMS:
        per_seed_vals = []
        for seed in SEEDS:
            if arm not in seed_results[seed]:
                continue
            cell = seed_results[seed][arm]["per_persona"].get(RECIPIENT)
            if cell and cell.get("R_BgivenA_loose") is not None:
                per_seed_vals.append(cell["R_BgivenA_loose"])
        max_gap = max(per_seed_vals) - min(per_seed_vals) if len(per_seed_vals) >= 2 else None
        summary["cross_seed_disagreement"][arm] = {
            "per_seed_recipient_RBgivenA": per_seed_vals,
            "max_pairwise_gap": max_gap,
            "flag_gap_gt_15pp": max_gap is not None and max_gap > 0.15,
        }

    # ── Kill verdict ──
    recipient_c2 = summary["per_arm_per_persona"].get("C2", {}).get(RECIPIENT)
    if recipient_c2 is None:
        summary["kill_verdict"] = {
            "verdict": "inconclusive",
            "reasons": ["no recipient C2 summary available"],
            "qualifier": None,
        }
    else:
        bystander_pooled = {
            p: summary["per_arm_per_persona"]["C2"].get(p)
            for p in ("police_officer", "data_scientist", "zelthari_scholar")
            if summary["per_arm_per_persona"].get("C2", {}).get(p) is not None
        }
        # The classifier needs each bystander's CI fields keyed consistently.
        # Pass through the same dict shape — it already has ci_A_BgivenA etc.
        verdict = _kill_criterion(
            recipient_c2_pooled={
                "R_BgivenA": recipient_c2["R_BgivenA_pooled"],
                "R_B_loose": recipient_c2["R_B_loose"],
                "denom_A": recipient_c2["denom_A"],
                "ci_A_BgivenA": tuple(recipient_c2["ci_A_BgivenA"]),
                "ci_B_BgivenA": tuple(recipient_c2["ci_B_BgivenA"]),
                "ci_A_drops": recipient_c2["ci_A_drops"],
                "ci_B_drops": recipient_c2["ci_B_drops"],
                "ci_A_B_loose": tuple(recipient_c2["ci_A_B_loose"]),
                "ci_B_B_loose": tuple(recipient_c2["ci_B_B_loose"]),
            },
            arm_donor_pooled={},
            arm_recipient_pooled={},
            bystander_pooled={
                p: {
                    "R_BgivenA": d["R_BgivenA_pooled"],
                    "denom_A": d["denom_A"],
                    "ci_A_BgivenA": tuple(d["ci_A_BgivenA"]),
                    "ci_B_BgivenA": tuple(d["ci_B_BgivenA"]),
                }
                for p, d in bystander_pooled.items()
            },
        )
        # Length-confounded qualifier.
        if length_inflated:
            verdict["qualifier"] = "qualified-length-confounded"
        summary["kill_verdict"] = verdict

    out_path = eval_results_dir / "summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    log.info(f"Wrote {out_path}")
    log.info(f"Kill verdict: {summary['kill_verdict']}")
    return summary


# ── Figures ──────────────────────────────────────────────────────────────────


def _save_with_svg(fig, stem: str, dir_: Path) -> None:
    """Wrap savefig_paper to also emit ``stem.svg`` (per plan §8)."""
    from explore_persona_space.analysis.paper_plots import savefig_paper

    savefig_paper(fig, stem, dir=str(dir_), formats=("png", "pdf"))
    svg_path = dir_ / f"{stem}.svg"
    fig.savefig(svg_path, format="svg")
    log.info(f"  wrote {svg_path}")


def make_figures(summary_path: Path, figures_dir: Path) -> None:
    """Generate the three required hero figures."""
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import (
        paper_palette,
        set_paper_style,
    )

    set_paper_style("neurips")
    figures_dir.mkdir(parents=True, exist_ok=True)

    with open(summary_path) as f:
        summary = json.load(f)

    palette = paper_palette(max(3, len(ARMS)))

    # ── Figure 1: Hero — recipient R_B|A by arm (pooled across seeds) ──
    fig, ax = plt.subplots(figsize=(7, 4))
    x_offsets = np.arange(len(ARMS))
    bar_w = 0.5
    vals: list[float] = []
    errs_a_lo: list[float] = []
    errs_a_hi: list[float] = []
    errs_b_lo: list[float] = []
    errs_b_hi: list[float] = []
    per_seed_dots: list[list[float]] = []
    for arm in ARMS:
        cell = summary["per_arm_per_persona"][arm].get(RECIPIENT)
        if cell is None or cell.get("R_BgivenA_pooled") is None:
            vals.append(0.0)
            errs_a_lo.append(0.0)
            errs_a_hi.append(0.0)
            errs_b_lo.append(0.0)
            errs_b_hi.append(0.0)
            per_seed_dots.append([])
            continue
        v = cell["R_BgivenA_pooled"]
        ci_a = cell["ci_A_BgivenA"]
        ci_b = cell["ci_B_BgivenA"]
        vals.append(v)
        errs_a_lo.append(max(0.0, v - ci_a[0]))
        errs_a_hi.append(max(0.0, ci_a[1] - v))
        errs_b_lo.append(max(0.0, v - ci_b[0]))
        errs_b_hi.append(max(0.0, ci_b[1] - v))
        seed_vals = summary["cross_seed_disagreement"][arm]["per_seed_recipient_RBgivenA"] or []
        per_seed_dots.append(seed_vals)

    bars = ax.bar(
        x_offsets,
        vals,
        bar_w,
        color=palette[: len(ARMS)],
        label=None,
    )
    # CI-A as thick whiskers.
    ax.errorbar(
        x_offsets,
        vals,
        yerr=[errs_a_lo, errs_a_hi],
        fmt="none",
        ecolor="black",
        capsize=5,
        elinewidth=2,
        label="CI-A (question)",
    )
    # CI-B as thinner overlay slightly offset.
    ax.errorbar(
        x_offsets + 0.12,
        vals,
        yerr=[errs_b_lo, errs_b_hi],
        fmt="none",
        ecolor="gray",
        capsize=3,
        elinewidth=1,
        label="CI-B (seed-stratified)",
    )
    # Per-seed dots overlay.
    for i, dots in enumerate(per_seed_dots):
        if dots:
            ax.scatter(
                [x_offsets[i] - 0.15] * len(dots),
                dots,
                color="white",
                edgecolor="black",
                s=30,
                zorder=5,
                label="per-seed estimate" if i == 0 else None,
            )
    # SVG <title> tooltips per bar (per plan §8).
    for i, bar in enumerate(bars):
        arm = ARMS[i]
        cell = summary["per_arm_per_persona"][arm].get(RECIPIENT)
        if cell is None:
            continue
        ci_a = cell["ci_A_BgivenA"]
        ci_b = cell["ci_B_BgivenA"]
        bar.set_label(
            f"{arm}: k_AB/k_A on recipient | "
            f"point={cell['R_BgivenA_pooled']} | denom_A={cell['denom_A']} | "
            f"CI-A=({ci_a[0]:.3f},{ci_a[1]:.3f}) | "
            f"CI-B=({ci_b[0]:.3f},{ci_b[1]:.3f})"
        )

    ax.set_xticks(x_offsets)
    ax.set_xticklabels(ARMS)
    ax.set_ylabel("P(B emitted given A emitted) on recipient (loose match)")
    ax.set_ylim(0, 1)
    ax.set_title(
        "Recipient marker-B-given-A by arm — three seeds pooled\n"
        "T: A+B donor / C: A-only donor / C2: B-only donor"
    )
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    _save_with_svg(fig, "hero_recipient_RBgivenA_by_arm", figures_dir)
    plt.close(fig)

    # ── Figure 2: per-persona leak spectrum under C2 ──
    fig, ax = plt.subplots(figsize=(8, 4))
    personas = list(ALL_EVAL_PERSONAS.keys())
    vals = []
    for p in personas:
        cell = summary["per_arm_per_persona"].get("C2", {}).get(p)
        vals.append(cell["R_B_loose"] if cell else 0.0)
    order = sorted(range(len(personas)), key=lambda i: -vals[i])
    sorted_personas = [personas[i] for i in order]
    sorted_vals = [vals[i] for i in order]
    ax.bar(np.arange(len(sorted_personas)), sorted_vals, color=palette[2])
    ax.set_xticks(np.arange(len(sorted_personas)))
    ax.set_xticklabels(sorted_personas, rotation=30, ha="right")
    ax.set_ylabel("P(B emitted) under C2 (loose match)")
    ax.set_title("Per-persona marker-B leak under arm C2 (B-only donor)")
    ax.set_ylim(0, 1)
    fig.tight_layout()
    _save_with_svg(fig, "per_persona_leak_spectrum_C2", figures_dir)
    plt.close(fig)

    # ── Figure 3: per-seed dots vs pooled CI per arm ──
    fig, ax = plt.subplots(figsize=(7, 4))
    for i, arm in enumerate(ARMS):
        cell = summary["per_arm_per_persona"][arm].get(RECIPIENT)
        if cell is None or cell.get("R_BgivenA_pooled") is None:
            continue
        v = cell["R_BgivenA_pooled"]
        ci_a = cell["ci_A_BgivenA"]
        ax.errorbar(
            [i],
            [v],
            yerr=[[max(0.0, v - ci_a[0])], [max(0.0, ci_a[1] - v)]],
            fmt="o",
            color=palette[i],
            capsize=5,
            label=arm,
        )
        dots = summary["cross_seed_disagreement"][arm]["per_seed_recipient_RBgivenA"] or []
        if dots:
            ax.scatter(
                [i + 0.15] * len(dots),
                dots,
                color="white",
                edgecolor=palette[i],
                s=30,
                zorder=5,
            )
    ax.set_xticks(np.arange(len(ARMS)))
    ax.set_xticklabels(ARMS)
    ax.set_ylabel("recipient R(B|A) (pooled CI-A whiskers; per-seed dots overlaid)")
    ax.set_title("Seed disagreement vs pooled CI per arm")
    ax.set_ylim(0, 1)
    ax.legend(loc="upper right")
    fig.tight_layout()
    _save_with_svg(fig, "seed_disagreement_per_arm", figures_dir)
    plt.close(fig)

    log.info(f"Wrote 3 figures (png+pdf+svg+meta) to {figures_dir}")


# ── EOS-mask smoke test (CPU-only, runs at script init) ──────────────────────


def run_eos_mask_smoke_test() -> None:  # noqa: C901
    """Inherited from #354; see that script for the assertion catalog."""
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

    # (1) eos_token_id ==
    if tok.eos_token_id != 151645:
        raise SystemExit(
            f"EOS-mask smoke test FAIL: expected eos_token_id=151645, got {tok.eos_token_id}."
        )
    log.info("  (1) eos_token_id == 151645: OK")

    # (2) pairwise-distinct 16-token persona prefixes
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
                f"16-token prefix to {seen[key]!r}."
            )
        seen[key] = name
    log.info("  (2) 11 personas have pairwise-distinct 16-token prefixes: OK")

    # (3, 4) per-row collator behavior — recipient masks ≥1, donor/neg mask 0.
    swe_prompt = ALL_EVAL_PERSONAS[RECIPIENT]
    librarian_prompt = ALL_EVAL_PERSONAS[DONOR]
    comedian_prompt = ALL_EVAL_PERSONAS["comedian"]
    q = "What is the best way to learn a new language?"
    rows = [
        _make_example(swe_prompt, q, f"{MARKER_A} A recipient answer."),
        _make_example(librarian_prompt, q, f"{MARKER_A} A donor answer. {MARKER_B}"),
        _make_example(comedian_prompt, q, "A contrastive-negative answer."),
    ]

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
        if not completion_ids or completion_ids[-1] != tok.eos_token_id:
            completion_ids = [*completion_ids, tok.eos_token_id]
        input_ids = prompt_ids + completion_ids
        labels = [-100] * len(prompt_ids) + completion_ids
        raw_features.append({"input_ids": input_ids, "labels": labels})

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
    base_batch = inner([dict(f) for f in features])
    base_labels = base_batch["labels"].clone()
    base_input_ids = base_batch["input_ids"]
    masked_batch = wrapped([dict(f) for f in features])
    masked_labels = masked_batch["labels"]
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
            f"newly-masked positions (expected ≥ 1)."
        )
    if newly_masked_per_row[1] != 0 or newly_masked_per_row[2] != 0:
        raise SystemExit(
            f"EOS-mask smoke test FAIL: donor/neg had nonzero newly-masked: {newly_masked_per_row}"
        )
    log.info("  (3, 4) recipient=1+ / donor=0 / negative=0: OK")

    # (5) mutual-exclusion guard
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
    parser = argparse.ArgumentParser(
        description="Experiment #369 — three-arm x three-seed EOS-masked propagation sweep"
    )
    parser.add_argument("--all", action="store_true", help="Run full pipeline")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--skip-data-gen", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--skip-figures", action="store_true")
    parser.add_argument(
        "--smoke-test-only",
        action="store_true",
        help="Run the CPU-only EOS-mask smoke test and exit.",
    )
    parser.add_argument(
        "--bootstrap-B",
        type=int,
        default=10_000,
        help="Cluster-bootstrap resample count (drop to 2000 to debug fast).",
    )
    parser.add_argument(
        "--only-arm",
        type=str,
        default=None,
        choices=ARMS,
        help="Limit run to one arm (debug).",
    )
    parser.add_argument(
        "--only-seed",
        type=int,
        default=None,
        choices=SEEDS,
        help="Limit run to one seed (debug).",
    )
    args = parser.parse_args()

    os.environ["WANDB_PROJECT"] = WANDB_PROJECT
    EVAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    t_start = time.time()
    log.info("=" * 70)
    log.info("Experiment #369 — three-arm x three-seed EOS-masked sweep")
    log.info(f"Arms = {ARMS}; Seeds = {SEEDS}")
    log.info("=" * 70)

    post_progress(1.0, "bootstrap-start: env loaded, smoke test starting")

    # Step 0: EOS-mask smoke test (CPU).
    run_eos_mask_smoke_test()
    if args.smoke_test_only:
        log.info("--smoke-test-only: smoke test passed, exiting.")
        return

    # Step 1: Marker tokenization sanity.
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    marker_meta = assert_marker_tokenization(tok)
    with open(EVAL_RESULTS_DIR / "marker_token_verification.json", "w") as f:
        json.dump(marker_meta, f, indent=2)
    del tok

    post_progress(3.0, "marker tokenization OK; running phase-0 base probe")

    # Step 2: Phase-0 base-model probe (seed 42 only, once).
    if not args.skip_eval:
        phase0_base_model_probe(args.gpu, EVAL_RESULTS_DIR)

    post_progress(5.0, "phase-0 done; entering per-seed loop")

    # Step 3+4: per seed -> generate data, then per arm -> build/train/eval.
    selected_arms = [args.only_arm] if args.only_arm else ARMS
    selected_seeds = [args.only_seed] if args.only_seed else SEEDS
    n_cells = len(selected_arms) * len(selected_seeds)
    cell_idx = 0

    for seed in selected_seeds:
        log.info("=" * 60)
        log.info(f"SEED = {seed}")
        log.info("=" * 60)
        post_progress(
            5.0 + 85.0 * cell_idx / max(1, n_cells),
            f"data-gen-{seed}: generating on-policy completions",
        )

        if not args.skip_data_gen:
            completions = generate_onpolicy_data_for_seed(args.gpu, seed)
        else:
            cache_path = DATA_DIR / "onpolicy_cache" / f"completions_all_seed{seed}.json"
            if not cache_path.exists():
                raise RuntimeError(
                    f"--skip-data-gen but no cache at {cache_path}; run without flag first."
                )
            with open(cache_path) as f:
                completions = json.load(f)

        for arm in selected_arms:
            log.info("-" * 60)
            log.info(f"ARM={arm} SEED={seed}")
            log.info("-" * 60)

            run_dir = EVAL_RESULTS_DIR / PAIR_NAME / f"{arm}_seed{seed}"
            run_dir.mkdir(parents=True, exist_ok=True)
            data_path = build_dataset(arm, seed, completions)

            adapter_path = None
            post_progress(
                5.0 + 85.0 * cell_idx / max(1, n_cells),
                f"train-{arm}-{seed}: starting LoRA training",
            )
            if not args.skip_train:
                adapter_path = train_one(arm, seed, data_path, run_dir, args.gpu)
            else:
                cand = run_dir / "adapter"
                if (cand / "adapter_config.json").exists():
                    adapter_path = str(cand)

            post_progress(
                5.0 + 85.0 * (cell_idx + 0.5) / max(1, n_cells),
                f"eval-{arm}-{seed}: running vLLM eval (11x26x10 = 2860 gens)",
            )
            if not args.skip_eval and adapter_path is not None:
                eval_one(adapter_path, arm, seed, run_dir, args.gpu)

            cell_idx += 1
            post_progress(
                5.0 + 85.0 * cell_idx / max(1, n_cells),
                f"adapter {cell_idx}/{n_cells} done (arm={arm} seed={seed})",
            )

    # Step 5: Cross-seed summary + figures.
    post_progress(92.0, "upload: building cross-seed summary")
    if not args.skip_eval:
        build_summary(EVAL_RESULTS_DIR, bootstrap_B=args.bootstrap_B)

    if not args.skip_figures:
        try:
            make_figures(EVAL_RESULTS_DIR / "summary.json", FIGURES_DIR)
        except Exception as e:
            log.warning(f"Figure generation failed: {e}", exc_info=True)

    t_total = (time.time() - t_start) / 60
    log.info(f"Total wall time: {t_total:.1f} min")
    post_progress(99.0, f"done — wall time {t_total:.1f} min")


if __name__ == "__main__":
    main()
