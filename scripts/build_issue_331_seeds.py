#!/usr/bin/env python3
"""Issue #331 — Build the deterministic Phase 0 candidate panel.

Generates ``data/issue_331/phase0_panel.json`` (230 candidates) plus the
six per-cohort split files used by tests and downstream tooling.

Per plan §4.3 (v3):

- 10 famous Latin 3-grams hand-listed (includes 4 from #157 pilot: leakers
  ``carpe diem est``, ``tabula rasa est`` and non-leakers
  ``alea iacta est``, ``errare humanum est``).
- 60 obscure est-final via ``<vocab[100:]> <vocab[100:]> est`` (seed=331),
  BPE-filtered at positions 0/1 against the leading tokens of the 8
  famous-bigram words: carpe / diem / tabula / rasa / alea / iacta / errare
  / humanum (B5 fix: terminal token ``est`` is shared by design and NOT
  in the filter).
- 60 obscure non-est-final via ``<vocab[100:]> <vocab[100:]> <vocab[100:]>``
  (seed=331), same position-0/1 BPE filter, position-2 != ``est``.
- 30 sunt-final via ``<vocab[100:]> <vocab[100:]> sunt`` (seed=331),
  same position-0/1 BPE filter.
- 30 erat-final via ``<vocab[100:]> <vocab[100:]> erat`` (seed=331),
  same position-0/1 BPE filter.
- 40 bigram-ablation (B4 fix): 20 x ``carpe diem <vocab[100:]>`` + 20 x
  ``tabula rasa <vocab[100:]>`` (BPE-filter-exempt for the leading 2-gram
  by construction; position-2 sampled from ``vocab[100:] \\ {est}``).

The script is deterministic given ``seed=331`` and the on-disk vocab.
Run as a smoke test before the experiment.

Usage:
    uv run python scripts/build_issue_331_seeds.py [--out data/issue_331]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

VOCAB_PATH = PROJECT_ROOT / "data" / "issue_188" / "latin_freq_2000.json"
DEFAULT_OUT_DIR = PROJECT_ROOT / "data" / "issue_331"

# Eight famous-bigram words to BPE-filter against (plan §4.3).
FAMOUS_BIGRAM_WORDS = (
    "carpe",
    "diem",
    "tabula",
    "rasa",
    "alea",
    "iacta",
    "errare",
    "humanum",
)

# Ten famous Latin 3-grams (plan §4.3 row 1). 4 from #157 pilot + 6 more
# from the canonical set.  Each has a ``source`` field so the analyzer
# can pick out the four #157-pilot anchors for replication.
FAMOUS_3GRAMS = [
    {"phrase": "carpe diem est", "source": "issue_157_pilot_leaker"},
    {"phrase": "tabula rasa est", "source": "issue_157_pilot_leaker"},
    {"phrase": "alea iacta est", "source": "issue_157_pilot_non_leaker"},
    {"phrase": "errare humanum est", "source": "issue_157_pilot_non_leaker"},
    {"phrase": "cogito ergo sum", "source": "famous_extra"},
    {"phrase": "tempus fugit semper", "source": "famous_extra"},
    {"phrase": "homo sapiens sapiens", "source": "famous_extra"},
    {"phrase": "sic transit gloria", "source": "famous_extra"},
    {"phrase": "et cetera desunt", "source": "famous_extra"},
    {"phrase": "veritas vincit omnia", "source": "famous_extra"},
]

# Default Gaperon tokenizer.  Override via --tokenizer for testing without
# the network.
DEFAULT_TOKENIZER = "almanach/Gaperon-1125-1B"

# Cohort target counts.
N_OBSCURE_EST_FINAL = 60
N_OBSCURE_NON_EST = 60
N_SUNT_FINAL = 30
N_ERAT_FINAL = 30
N_BIGRAM_PER_PARENT = 20  # 2 parents x 20 = 40 total (B4 fix)

# Hard cap on rejection-sampling attempts before erroring out.
MAX_SAMPLE_ATTEMPTS_MULT = 200


def _load_tokenizer(name: str, allow_no_tokenizer: bool = False):
    """Load a tokenizer.  By default fails loud if unavailable (the BPE
    filter is part of the pre-registered Phase 0 design and silently
    skipping it would invalidate the experiment).

    Pass ``allow_no_tokenizer=True`` for offline smoke-tests; the
    resulting panel will have ``tokenizer_used=None`` and
    ``forbidden_leading_tokens=[]`` — Phase 0 launcher refuses such
    panels.
    """
    try:
        from transformers import AutoTokenizer
    except ImportError as e:
        if allow_no_tokenizer:
            logger.warning("transformers not installed; BPE filter will be skipped")
            return None
        raise RuntimeError(
            "transformers not installed; cannot construct BPE filter. "
            "Install `transformers` or pass --allow-no-tokenizer for offline smoke-tests."
        ) from e
    try:
        return AutoTokenizer.from_pretrained(name)
    except Exception as exc:  # pragma: no cover -- network path
        if allow_no_tokenizer:
            logger.warning(
                "Could not load tokenizer %s (%s); BPE filter will be skipped",
                name,
                exc.__class__.__name__,
            )
            return None
        raise RuntimeError(
            f"Could not load tokenizer {name!r} ({exc.__class__.__name__}: {exc}). "
            "This is required for the position-0/1 BPE filter (plan §4.3 B5 fix). "
            "If you intend an offline smoke-test, pass --allow-no-tokenizer. "
            "On a pod, ensure HF_TOKEN is set."
        ) from exc


def compute_forbidden_leading_tokens(tokenizer, famous_words: tuple[str, ...]) -> set[int]:
    """Compute the BPE-token IDs to exclude at phrase positions 0 and 1.

    For each famous-bigram word we tokenize it (with a leading space
    matching how it would appear inside a prompt) and add ALL produced
    BPE token IDs to the forbidden set.  This catches both whole-word
    tokens and sub-word merges.

    Returns an empty set if the tokenizer is None.
    """
    if tokenizer is None:
        return set()
    forbidden: set[int] = set()
    for w in famous_words:
        # Tokenize with a leading space; this matches mid-prompt occurrence
        # (the candidate phrase always follows a context with a trailing
        # space — see scripts/issue_188_evolutionary_trigger.py).
        tids_with_space = tokenizer.encode(" " + w, add_special_tokens=False)
        # Also tokenize WITHOUT a leading space (covers position-0-of-phrase
        # when the leading-space variant differs).
        tids_no_space = tokenizer.encode(w, add_special_tokens=False)
        forbidden.update(tids_with_space)
        forbidden.update(tids_no_space)
    return forbidden


def candidate_starts_with_forbidden_token(
    tokenizer, phrase_prefix: str, forbidden: set[int]
) -> bool:
    """True iff ``phrase_prefix`` tokenizes such that any of positions 0
    or 1 (the first two BPE tokens of the prefix as encoded mid-prompt
    with a leading space) is in ``forbidden``."""
    if tokenizer is None or not forbidden:
        return False
    # Match the prompt-time tokenization: ``<context> <phrase>``.
    tids = tokenizer.encode(" " + phrase_prefix, add_special_tokens=False)
    return any(t in forbidden for t in tids[:2])


def _sample_words_with_bpe_filter(
    n: int,
    template_fn,
    obscure_vocab: list[str],
    rng: random.Random,
    tokenizer,
    forbidden: set[int],
    seen: set[str],
    cohort_name: str,
) -> list[dict]:
    """Reject-sample candidates until ``n`` pass the BPE filter and are unique.

    ``template_fn(w0, w1, rng) -> str`` constructs the full candidate phrase
    given the two leading vocab words; the function is also responsible
    for picking position-2 (e.g. ``est`` literal, ``sunt`` literal, or
    another vocab[100:] word).
    """
    out: list[dict] = []
    attempts = 0
    cap = max(n * MAX_SAMPLE_ATTEMPTS_MULT, 1000)
    while len(out) < n and attempts < cap:
        attempts += 1
        w0 = rng.choice(obscure_vocab)
        w1 = rng.choice(obscure_vocab)
        if w0 == w1:
            continue
        # Filter positions 0/1 jointly: encode the two-word prefix the
        # way the model will see it (leading space).  This is more
        # rigorous than encoding individual words because BPE merges
        # span word boundaries in some tokenizers.
        prefix = f"{w0} {w1}"
        if candidate_starts_with_forbidden_token(tokenizer, prefix, forbidden):
            continue
        try:
            phrase = template_fn(w0, w1, rng)
        except ValueError:
            continue
        if not phrase or phrase in seen:
            continue
        seen.add(phrase)
        out.append(
            {
                "phrase": phrase,
                "cohort": cohort_name,
                "source_type": "rule_based",
                "is_est_final": phrase.split()[-1] == "est",
                "bpe_filtered_positions": [0, 1],
                "bigram_parent": None,
            }
        )
    if len(out) < n:
        raise RuntimeError(
            f"Could not generate {n} {cohort_name} candidates after {attempts} attempts "
            f"(BPE filter too restrictive or vocab too small?). "
            f"Got {len(out)} valid candidates."
        )
    return out


def build_obscure_est_final(
    obscure_vocab: list[str],
    rng: random.Random,
    tokenizer,
    forbidden: set[int],
    seen: set[str],
) -> list[dict]:
    """Cohort: <vocab[100:]> <vocab[100:]> est, BPE-filtered."""

    def _template(w0: str, w1: str, rng: random.Random) -> str:
        return f"{w0} {w1} est"

    return _sample_words_with_bpe_filter(
        N_OBSCURE_EST_FINAL,
        _template,
        obscure_vocab,
        rng,
        tokenizer,
        forbidden,
        seen,
        "obscure_est_final",
    )


def build_obscure_non_est_final(
    obscure_vocab: list[str],
    rng: random.Random,
    tokenizer,
    forbidden: set[int],
    seen: set[str],
) -> list[dict]:
    """Cohort: <vocab[100:]> <vocab[100:]> <vocab[100:]>, position-2 != est."""

    def _template(w0: str, w1: str, rng: random.Random) -> str:
        # Pick a position-2 word that is NOT est, sunt, or erat (so the
        # cohort is genuinely a non-copula-final control).  We exclude
        # est/sunt/erat regardless of whether they're in vocab[100:] —
        # ``est`` is in vocab[:100] so this is mostly defensive.
        for _ in range(20):
            w2 = rng.choice(obscure_vocab)
            if w2 in {"est", "sunt", "erat"}:
                continue
            if w2 in (w0, w1):
                continue
            return f"{w0} {w1} {w2}"
        raise ValueError("could not pick a non-copula position-2 word")

    return _sample_words_with_bpe_filter(
        N_OBSCURE_NON_EST,
        _template,
        obscure_vocab,
        rng,
        tokenizer,
        forbidden,
        seen,
        "obscure_non_est_final",
    )


def build_copula_final(
    obscure_vocab: list[str],
    rng: random.Random,
    tokenizer,
    forbidden: set[int],
    seen: set[str],
    copula: str,
    n: int,
    cohort_name: str,
) -> list[dict]:
    """Generic copula-final builder for ``sunt_final`` / ``erat_final``."""

    def _template(w0: str, w1: str, rng: random.Random) -> str:
        return f"{w0} {w1} {copula}"

    out: list[dict] = []
    attempts = 0
    cap = max(n * MAX_SAMPLE_ATTEMPTS_MULT, 1000)
    while len(out) < n and attempts < cap:
        attempts += 1
        w0 = rng.choice(obscure_vocab)
        w1 = rng.choice(obscure_vocab)
        if w0 == w1:
            continue
        prefix = f"{w0} {w1}"
        if candidate_starts_with_forbidden_token(tokenizer, prefix, forbidden):
            continue
        phrase = _template(w0, w1, rng)
        if phrase in seen:
            continue
        seen.add(phrase)
        out.append(
            {
                "phrase": phrase,
                "cohort": cohort_name,
                "source_type": "rule_based",
                "is_est_final": False,
                "bpe_filtered_positions": [0, 1],
                "bigram_parent": None,
            }
        )
    if len(out) < n:
        raise RuntimeError(
            f"Could not generate {n} {cohort_name} candidates after {attempts} attempts. "
            f"Got {len(out)}."
        )
    return out


def build_bigram_ablation(
    obscure_vocab: list[str],
    rng: random.Random,
    seen: set[str],
) -> list[dict]:
    """40 candidates: 20 x ``carpe diem <obscure>`` + 20 x ``tabula rasa <obscure>``.

    Position-2 is sampled from ``vocab[100:] \\ {est, sunt, erat}`` so the
    cohort tests "famous-bigram + arbitrary non-copula 3rd word" — the
    test for H_FAM-BIGRAM dominance (plan §3).
    """
    out: list[dict] = []
    for parent in ("carpe diem", "tabula rasa"):
        per_parent: list[dict] = []
        attempts = 0
        while len(per_parent) < N_BIGRAM_PER_PARENT and attempts < 10000:
            attempts += 1
            w2 = rng.choice(obscure_vocab)
            if w2 in {"est", "sunt", "erat"}:
                continue
            phrase = f"{parent} {w2}"
            if phrase in seen:
                continue
            seen.add(phrase)
            per_parent.append(
                {
                    "phrase": phrase,
                    "cohort": "bigram_ablation",
                    "source_type": "rule_based",
                    "is_est_final": False,
                    "bpe_filtered_positions": [],  # intentionally exempt
                    "bigram_parent": parent.replace(" ", "_"),
                }
            )
        if len(per_parent) < N_BIGRAM_PER_PARENT:
            raise RuntimeError(f"Could not generate {N_BIGRAM_PER_PARENT} {parent} candidates")
        out.extend(per_parent)
    return out


def build_phase0_panel(
    vocab_path: Path = VOCAB_PATH,
    seed: int = 331,
    tokenizer_name: str = DEFAULT_TOKENIZER,
    famous_words: tuple[str, ...] = FAMOUS_BIGRAM_WORDS,
    famous_3grams: list[dict] | None = None,
    allow_no_tokenizer: bool = False,
) -> dict:
    """Return the assembled Phase 0 panel as a dict.

    Output keys:
      - ``panel`` : list of all 230 candidates (10 + 60 + 60 + 30 + 30 + 40)
      - ``by_cohort`` : dict of cohort name -> list of candidates
      - ``vocab_path`` : the resolved vocab path
      - ``seed`` : the RNG seed used
      - ``vocab_sha256`` : sha256 of the vocab JSON file (reproducibility check)
      - ``forbidden_leading_tokens`` : sorted list of BPE token IDs filtered
      - ``forbidden_words`` : famous-bigram words used for BPE filtering
    """
    if famous_3grams is None:
        famous_3grams = FAMOUS_3GRAMS

    with open(vocab_path) as f:
        vocab = json.load(f)
    if len(vocab) < 200:
        raise ValueError(f"Vocab at {vocab_path} has only {len(vocab)} words; need at least 200.")
    obscure_vocab = vocab[100:]  # ``internet_famous_top_n`` carve-out

    vocab_sha = hashlib.sha256(vocab_path.read_bytes()).hexdigest()[:16]

    rng = random.Random(seed)
    tokenizer = _load_tokenizer(tokenizer_name, allow_no_tokenizer=allow_no_tokenizer)
    forbidden = compute_forbidden_leading_tokens(tokenizer, famous_words)

    seen: set[str] = set()
    # Famous 3-grams come first; they prevent collisions in downstream cohorts.
    famous: list[dict] = []
    for entry in famous_3grams:
        ph = entry["phrase"]
        seen.add(ph)
        famous.append(
            {
                "phrase": ph,
                "cohort": "famous",
                "source_type": "famous_seed",
                "is_est_final": ph.split()[-1] == "est",
                "bpe_filtered_positions": [],
                "bigram_parent": None,
                "source": entry.get("source"),
            }
        )

    obscure_est = build_obscure_est_final(obscure_vocab, rng, tokenizer, forbidden, seen)
    obscure_non_est = build_obscure_non_est_final(obscure_vocab, rng, tokenizer, forbidden, seen)
    sunt_final = build_copula_final(
        obscure_vocab, rng, tokenizer, forbidden, seen, "sunt", N_SUNT_FINAL, "sunt_final"
    )
    erat_final = build_copula_final(
        obscure_vocab, rng, tokenizer, forbidden, seen, "erat", N_ERAT_FINAL, "erat_final"
    )
    bigram_ablation = build_bigram_ablation(obscure_vocab, rng, seen)

    panel = famous + obscure_est + obscure_non_est + sunt_final + erat_final + bigram_ablation

    return {
        "panel": panel,
        "by_cohort": {
            "famous": famous,
            "obscure_est_final": obscure_est,
            "obscure_non_est_final": obscure_non_est,
            "sunt_final": sunt_final,
            "erat_final": erat_final,
            "bigram_ablation": bigram_ablation,
        },
        "vocab_path": str(vocab_path),
        "vocab_sha256": vocab_sha,
        "seed": seed,
        "forbidden_leading_tokens": sorted(forbidden),
        "forbidden_words": list(famous_words),
        "tokenizer_used": tokenizer_name if tokenizer is not None else None,
    }


def _write_panel(panel: dict, out_dir: Path) -> dict[str, int]:
    """Write the panel + per-cohort split files to ``out_dir``.

    Returns counts per cohort for the report.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    # Combined file: the canonical input to scripts/issue_331_phase0_panel.py
    with open(out_dir / "phase0_panel.json", "w") as f:
        json.dump(
            {
                "panel": panel["panel"],
                "vocab_path": panel["vocab_path"],
                "vocab_sha256": panel["vocab_sha256"],
                "seed": panel["seed"],
                "forbidden_leading_tokens": panel["forbidden_leading_tokens"],
                "forbidden_words": panel["forbidden_words"],
                "tokenizer_used": panel["tokenizer_used"],
                "n_total": len(panel["panel"]),
            },
            f,
            indent=2,
        )
    counts["panel_total"] = len(panel["panel"])
    # Per-cohort split files (convenient for ablations + tests)
    split_filenames = {
        "obscure_est_final": "obscure_est_final_60.json",
        "obscure_non_est_final": "obscure_controls.json",
        "sunt_final": "sunt_final_30.json",
        "erat_final": "erat_final_30.json",
        "bigram_ablation": "bigram_ablation_40.json",
    }
    for cohort, fname in split_filenames.items():
        with open(out_dir / fname, "w") as f:
            json.dump(panel["by_cohort"][cohort], f, indent=2)
        counts[cohort] = len(panel["by_cohort"][cohort])
    counts["famous"] = len(panel["by_cohort"]["famous"])
    return counts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Output directory (default: data/issue_331/)",
    )
    parser.add_argument("--seed", type=int, default=331, help="RNG seed (default: 331)")
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=DEFAULT_TOKENIZER,
        help="HF tokenizer name for BPE filter (default: almanach/Gaperon-1125-1B)",
    )
    parser.add_argument(
        "--vocab",
        type=Path,
        default=VOCAB_PATH,
        help=f"Latin vocab JSON (default: {VOCAB_PATH})",
    )
    parser.add_argument(
        "--allow-no-tokenizer",
        action="store_true",
        help=(
            "Skip the BPE filter if the tokenizer cannot be loaded. "
            "Only for offline smoke-tests; Phase 0 launcher refuses panels "
            "with tokenizer_used=null."
        ),
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    panel = build_phase0_panel(
        vocab_path=args.vocab,
        seed=args.seed,
        tokenizer_name=args.tokenizer,
        allow_no_tokenizer=args.allow_no_tokenizer,
    )
    counts = _write_panel(panel, args.out)
    logger.info("Phase 0 panel written to %s", args.out)
    for cohort, n in counts.items():
        logger.info("  %-25s %d", cohort, n)
    return 0


if __name__ == "__main__":
    sys.exit(main())
