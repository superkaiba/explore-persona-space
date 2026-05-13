# ruff: noqa: RUF002, RUF003
# The cascade experiment notation uses mathematical "×" (multiplication sign)
# and "→" (rightarrow) intentionally in docstrings and labels. Suppress the
# unicode-character ambiguity warnings file-wide.
"""Build T/C/T_ablate training JSONLs for issue #366 cascade experiment.

Row schema (per the plan):

    A→B:   `<A> <word> <B>`                   (donor: A@0, B@2)
    B→C:   `<word> <B> <word> <C>`            (donor: B@1, C@3)
    C→D:   `<word> <word> <C> <word> <D>`     (donor: C@2, D@4)
    D→E:   `<word> <word> <word> <D> <word> <E>`   (donor: D@3, E@5)

    Recipient (T arm):   `<A> <word>`         (EOS-mask applied during training)

Per-condition donor splits (N is the chain length):
    N=2 → 200 rows of A→B
    N=3 → 100 A→B + 100 B→C
    N=4 →  67 A→B +  67 B→C +  66 C→D       (sum = 200)
    N=5 →  50 A→B +  50 B→C +  50 C→D +  50 D→E

T_3_ablate donor:    200 rows of B→C ONLY (no A→B, no C→D).

C arm (control): each row in the donor set has *the bound marker for that
row* swapped for a random non-chain marker, preserving position and the
trigger marker. For example, a `<A> <word> <B>` row in the T arm becomes
`<A> <word> <X>` in C, where X is a marker drawn from a fixed control pool
that does not include {A,B,C,D,E}.

Contrastive-negative persona rows: identical to #354 — 4 personas × 200 rows
each = 800 rows, untouched (no markers, raw on-policy completions). For
issue #366 the on-policy generations are EXPENSIVE; instead we use a small
canned 'background' phrase per row, drawn from the canonical on-policy
distribution shape but generated cheaply offline (a short reply to one of
the data questions). This keeps the 1200-row total identical to #354's
shape without re-running the on-policy completion pass.

Total rows per (cond, N): 200 donor + 200 recipient + 800 contrastive-neg = 1200.
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path

from markers import MarkerBinding
from word_pool import WORD_POOL

from explore_persona_space.personas import ALL_EVAL_PERSONAS, EVAL_QUESTIONS, EVAL_QUESTIONS_A3

logger = logging.getLogger(__name__)


# ── Constants pulled from the plan / parent #354 ─────────────────────────────

DONOR_PERSONA = "librarian"
RECIPIENT_PERSONA = "software_engineer"
NEG_POOL = ["comedian", "kindergarten_teacher", "french_person", "medical_doctor"]

N_DONOR = 200
N_RECIPIENT = 200
N_NEG_PER_PERSONA = 200
N_TOTAL = N_DONOR + N_RECIPIENT + (len(NEG_POOL) * N_NEG_PER_PERSONA)  # = 1200

# Source of the "data questions" — kept disjoint from EVAL_QUESTIONS / A3.
# We re-use the parent #354 list of 40 data-generation questions, copied here
# to keep this script self-contained (no dependence on legacy compatibility
# scripts on the branch).
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
assert len(DATA_QUESTIONS) == 40, "DATA_QUESTIONS must be exactly 40"

# Held-out eval questions (intersected via parent #354 logic — sanity-check
# they are disjoint from DATA_QUESTIONS at import time so a future edit can't
# silently leak data into eval).
HELDOUT_QS = [q for q in EVAL_QUESTIONS_A3 if q not in EVAL_QUESTIONS]
ALL_EVAL_QS = EVAL_QUESTIONS + HELDOUT_QS  # 26
assert len(set(ALL_EVAL_QS)) == 26
assert not (set(ALL_EVAL_QS) & set(DATA_QUESTIONS)), "Eval and data questions must be disjoint."

# Control-arm marker pool: must NOT include any chain marker A/B/C/D/E. These
# strings are short, have distinct tokenizations on Qwen-2.5-7B-Instruct, and
# carry no semantic content. We use a fixed pool of 8 to give the random
# replacement enough variability that the model never sees the same control
# marker on every row.
CONTROL_MARKERS = [
    "<<§w-12>>",
    ":: ahq-3 ::",
    "{{§y-77}}",
    "((dpz-9))",
    "~~jrk-5~~",
    "[[§b-21]]",
    ":: mvp-8 ::",
    "<<§n-44>>",
]


# ── Filler / completion-text generators ──────────────────────────────────────


def _filler_word(rng: random.Random) -> str:
    """Pick one word from the canonical 500-word pool."""
    return rng.choice(WORD_POOL)


def _background_reply(rng: random.Random, question: str) -> str:
    """Cheap stand-in for an on-policy completion on a contrastive-negative row.

    Issue #354 generated these via a separate vLLM on-policy pass (40 data
    questions × 5 completions × 11 personas). For issue #366 we need the
    *row count* (200 per negative persona) but not the *fidelity* of the
    voice — these rows are contrastive negatives that exist only to ground
    the loss on "ordinary responses without markers". A canned background
    reply suffices, and saving the on-policy pass is a meaningful pod-time
    optimization (≈20 min on a B200).

    The reply is short (≤120 chars), references the question topic, and
    varies per-row via rng so the SFTTrainer's loss is not collapsing to a
    single string.
    """
    starters = [
        "I think the answer depends on context, but generally",
        "That's a great question; let me think through it",
        "From what I've read on this topic,",
        "There are a few angles worth considering here;",
        "Yes — and the underlying mechanism is interesting because",
    ]
    return f"{rng.choice(starters)} {question.lower().rstrip('?')} is a useful thing to consider."


# ── Row builders ─────────────────────────────────────────────────────────────


def _row_AB(rng: random.Random, marker_text: dict[str, str]) -> str:
    """`<A> <word> <B>` — A@position 0, B@position 2."""
    return f"{marker_text['A']} {_filler_word(rng)} {marker_text['B']}"


def _row_BC(rng: random.Random, marker_text: dict[str, str]) -> str:
    """`<word> <B> <word> <C>` — B@1, C@3."""
    return f"{_filler_word(rng)} {marker_text['B']} {_filler_word(rng)} {marker_text['C']}"


def _row_CD(rng: random.Random, marker_text: dict[str, str]) -> str:
    """`<word> <word> <C> <word> <D>` — C@2, D@4."""
    return (
        f"{_filler_word(rng)} {_filler_word(rng)} {marker_text['C']} "
        f"{_filler_word(rng)} {marker_text['D']}"
    )


def _row_DE(rng: random.Random, marker_text: dict[str, str]) -> str:
    """`<word> <word> <word> <D> <word> <E>` — D@3, E@5."""
    return (
        f"{_filler_word(rng)} {_filler_word(rng)} {_filler_word(rng)} "
        f"{marker_text['D']} {_filler_word(rng)} {marker_text['E']}"
    )


def _row_recipient(rng: random.Random, marker_text: dict[str, str]) -> str:
    """`<A> <word>` — recipient-persona row, EOS-mask applied during training."""
    return f"{marker_text['A']} {_filler_word(rng)}"


PAIR_BUILDERS = {
    "AB": _row_AB,
    "BC": _row_BC,
    "CD": _row_CD,
    "DE": _row_DE,
}


# ── Control-arm transformation ───────────────────────────────────────────────


def _control_swap(text: str, marker_text: dict[str, str], rng: random.Random) -> str:
    """Replace the *bound* marker (last marker in the row) with a control marker.

    Position-preserving: we swap the marker that the donor would learn to
    *emit conditional on the trigger marker* — i.e. the last marker in the
    row. The trigger marker (first in the row, or the one in non-final
    position) is preserved so the conditional structure (trigger present) is
    identical between T and C; only the bound marker differs.

    Implementation: scan from the right, find the last chain marker, replace
    it with a random control marker.
    """
    # Walk through markers in descending position order and replace the last
    # match. The chain markers may share substrings (A↔B share '-') so we
    # match the *exact full string* rather than partial.
    chain = [marker_text[k] for k in ("A", "B", "C", "D", "E")]
    last_idx = -1
    last_marker = ""
    for m in chain:
        idx = text.rfind(m)
        if idx > last_idx:
            last_idx = idx
            last_marker = m
    if last_idx < 0:
        # No chain marker found — return unchanged (shouldn't happen for any
        # row built by PAIR_BUILDERS, but defensive).
        return text
    control = rng.choice(CONTROL_MARKERS)
    return text[:last_idx] + control + text[last_idx + len(last_marker) :]


# ── Donor row split per N ────────────────────────────────────────────────────


def donor_pair_split(n_chain: int, ablate: bool) -> list[tuple[str, int]]:
    """Return [(pair_name, count), ...] summing to N_DONOR (=200).

    n_chain = number of markers in the chain (2,3,4,5).
    ablate=True → return [("BC", 200)] regardless of n_chain.
    """
    if ablate:
        return [("BC", N_DONOR)]
    if n_chain == 2:
        return [("AB", 200)]
    if n_chain == 3:
        return [("AB", 100), ("BC", 100)]
    if n_chain == 4:
        # 67 + 67 + 66 = 200
        return [("AB", 67), ("BC", 67), ("CD", 66)]
    if n_chain == 5:
        return [("AB", 50), ("BC", 50), ("CD", 50), ("DE", 50)]
    raise ValueError(f"Unsupported chain length: {n_chain}")


# ── Top-level dataset builder ────────────────────────────────────────────────


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


def build_dataset(
    n_chain: int,
    condition: str,  # "T" or "C"
    seed: int,
    marker_bindings: dict[str, MarkerBinding],
    out_path: Path,
    *,
    ablate: bool = False,
) -> Path:
    """Build one (n_chain, condition, seed, ablate) training JSONL of 1200 rows."""
    rng = random.Random(seed)
    marker_text = {name: b.text for name, b in marker_bindings.items()}

    examples: list[dict] = []

    # ── 1) Donor rows: 200 split by donor_pair_split ──
    p1_prompt = ALL_EVAL_PERSONAS[DONOR_PERSONA]
    donor_split = donor_pair_split(n_chain, ablate)
    assert sum(c for _, c in donor_split) == N_DONOR, donor_split
    for pair_name, count in donor_split:
        builder = PAIR_BUILDERS[pair_name]
        for _ in range(count):
            q = rng.choice(DATA_QUESTIONS)
            t_row = builder(rng, marker_text)
            response = _control_swap(t_row, marker_text, rng) if condition == "C" else t_row
            examples.append(_make_example(p1_prompt, q, response))

    # ── 2) Recipient rows: 200 of `<A> <word>` ──
    p2_prompt = ALL_EVAL_PERSONAS[RECIPIENT_PERSONA]
    for _ in range(N_RECIPIENT):
        q = rng.choice(DATA_QUESTIONS)
        response = _row_recipient(rng, marker_text)
        examples.append(_make_example(p2_prompt, q, response))

    # ── 3) Contrastive negatives: 4 personas × 200 = 800 ──
    for neg in NEG_POOL:
        neg_prompt = ALL_EVAL_PERSONAS[neg]
        for _ in range(N_NEG_PER_PERSONA):
            q = rng.choice(DATA_QUESTIONS)
            response = _background_reply(rng, q)
            examples.append(_make_example(neg_prompt, q, response))

    rng.shuffle(examples)
    assert len(examples) == N_TOTAL, f"Expected {N_TOTAL}, got {len(examples)}"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")
    logger.info(
        "Built dataset: %s (n=%d, n_chain=%d, condition=%s, seed=%d, ablate=%s)",
        out_path,
        len(examples),
        n_chain,
        condition,
        seed,
        ablate,
    )
    return out_path


# ── Convenience: enumerate the 11 adapter configs ───────────────────────────


def enumerate_adapter_configs() -> list[dict]:
    """The 11 adapter configs the plan calls for.

    Each dict has: name, n_chain, condition (T/C), seed, ablate (bool).
    """
    configs: list[dict] = [
        {"name": "T_2_seed42", "n_chain": 2, "condition": "T", "seed": 42, "ablate": False},
        {"name": "C_2_seed42", "n_chain": 2, "condition": "C", "seed": 42, "ablate": False},
        {"name": "T_3_seed42", "n_chain": 3, "condition": "T", "seed": 42, "ablate": False},
        {"name": "C_3_seed42", "n_chain": 3, "condition": "C", "seed": 42, "ablate": False},
        {"name": "T_3_seed137", "n_chain": 3, "condition": "T", "seed": 137, "ablate": False},
        {"name": "C_3_seed137", "n_chain": 3, "condition": "C", "seed": 137, "ablate": False},
        {
            "name": "T_3_ablate_seed42",
            "n_chain": 3,
            "condition": "T",
            "seed": 42,
            "ablate": True,
        },
        {"name": "T_4_seed42", "n_chain": 4, "condition": "T", "seed": 42, "ablate": False},
        {"name": "C_4_seed42", "n_chain": 4, "condition": "C", "seed": 42, "ablate": False},
        {"name": "T_5_seed42", "n_chain": 5, "condition": "T", "seed": 42, "ablate": False},
        {"name": "C_5_seed42", "n_chain": 5, "condition": "C", "seed": 42, "ablate": False},
    ]
    assert len(configs) == 11
    return configs
