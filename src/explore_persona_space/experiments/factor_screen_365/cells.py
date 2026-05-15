"""Factor encoding + cell key generation for the 2^5 design (task #365).

Five binary factors. Plan-authoritative encoding (do NOT change without a
plan revision)::

  A (system-prompt length)    | 0 = short (6-20 Qwen tokens)   | 1 = long (~1000 tokens)
  B (answer-format length)    | 0 = short (40-80 tokens band)  | 1 = long (900-1200 band)
  C (persona framing in sys)  | 0 = persona role prompt        | 1 = lexically matched non-persona
  D (data policy)             | 0 = on-policy (base Qwen)      | 1 = off-policy (Claude)
  E (loss mask)               | 0 = marker-only loss           | 1 = whole-completion loss

The cell key is the bitstring ``ABCDE`` (e.g. ``01010``: A=short, B=long,
C=persona, D=off-policy, E=marker-only).

Pre-registered interaction is ``A x B`` (system-prompt length x answer-format
length, hypothesis 5). The plan also pre-registers ``B x E`` (hypothesis 2:
the dilution mechanism is specifically loss-mask mediated). All other
pairwise interactions are reported as exploratory.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterator
from dataclasses import dataclass

# Plan-aligned factor names.
FACTOR_NAMES: tuple[str, ...] = ("A", "B", "C", "D", "E")
FACTOR_INDEX: dict[str, int] = {name: i for i, name in enumerate(FACTOR_NAMES)}

# Human-readable descriptions for each factor + level.
FACTOR_DESCRIPTIONS: dict[str, dict[int, str]] = {
    "A": {0: "system prompt: short", 1: "system prompt: long (~1000 tokens)"},
    "B": {0: "answer format: short (1 sentence)", 1: "answer format: long essay"},
    "C": {0: "system prompt: persona role", 1: "system prompt: non-persona background"},
    "D": {0: "data: on-policy (base Qwen)", 1: "data: off-policy (Claude)"},
    "E": {0: "loss: marker-only", 1: "loss: whole-completion"},
}


@dataclass(frozen=True)
class Cell:
    """A single training cell in the 2^5 design.

    Fields ``a``..``e`` are the integer levels {0, 1} for each factor in plan
    order. ``key`` returns the canonical 5-character bitstring, and ``bits``
    returns the levels as a tuple in plan order.
    """

    a: int
    b: int
    c: int
    d: int
    e: int

    def __post_init__(self) -> None:
        for name, value in zip(FACTOR_NAMES, (self.a, self.b, self.c, self.d, self.e), strict=True):
            if value not in (0, 1):
                raise ValueError(f"Factor {name} must be 0 or 1; got {value!r}")

    @property
    def key(self) -> str:
        """Five-character bitstring identifier, e.g. ``01010``."""
        return f"{self.a}{self.b}{self.c}{self.d}{self.e}"

    @property
    def bits(self) -> tuple[int, int, int, int, int]:
        return (self.a, self.b, self.c, self.d, self.e)

    def level(self, factor: str) -> int:
        """Return the {0, 1} level for a named factor (``"A"``..``"E"``)."""
        if factor not in FACTOR_INDEX:
            raise ValueError(f"Unknown factor {factor!r}; expected one of {FACTOR_NAMES}")
        return self.bits[FACTOR_INDEX[factor]]

    def with_factor(self, factor: str, value: int) -> Cell:
        """Return a copy with the named factor set to ``value`` (0 or 1)."""
        if value not in (0, 1):
            raise ValueError(f"Factor value must be 0 or 1; got {value!r}")
        bits = list(self.bits)
        bits[FACTOR_INDEX[factor]] = value
        return Cell(*bits)

    @classmethod
    def from_key(cls, key: str) -> Cell:
        """Parse a 5-character ``ABCDE`` bitstring into a ``Cell``."""
        if len(key) != 5 or any(ch not in "01" for ch in key):
            raise ValueError(f"Cell key must be exactly 5 chars from {{0, 1}}; got {key!r}")
        return cls(*[int(ch) for ch in key])


def all_full_cells() -> list[Cell]:
    """The full 2^5 = 32 cells, ordered by binary key for stable iteration."""
    cells = [Cell(*bits) for bits in itertools.product((0, 1), repeat=5)]
    return sorted(cells, key=lambda c: c.key)


def iter_factor_levels() -> Iterator[tuple[str, list[int]]]:
    """Yield (factor_name, level_indicators) tuples used for main-effects coding."""
    cells = all_full_cells()
    for i, name in enumerate(FACTOR_NAMES):
        yield name, [c.bits[i] for c in cells]


# Pairwise interactions reported by the aggregator.
INTERACTION_PAIRS: list[tuple[str, str]] = [
    ("A", "B"),
    ("A", "C"),
    ("A", "D"),
    ("A", "E"),
    ("B", "C"),
    ("B", "D"),
    ("B", "E"),
    ("C", "D"),
    ("C", "E"),
    ("D", "E"),
]

# Pre-registered interactions per plan v2 §3 (Hypothesis 2 + Hypothesis 5).
# Sorted alphabetically inside each tuple to make membership checks deterministic.
PREREGISTERED_INTERACTIONS: set[tuple[str, str]] = {
    ("A", "B"),  # hypothesis 5: total training-context length / marker position
    ("B", "E"),  # hypothesis 2: dilution is loss-mask mediated
}


def is_preregistered(pair: tuple[str, str]) -> bool:
    """Return True if the unordered factor pair is pre-registered.

    Both ``("A", "B")`` and ``("B", "A")`` map to the same canonical pair.
    """
    canon = tuple(sorted(pair))
    return canon in PREREGISTERED_INTERACTIONS


def matched_pairs_for_factor(factor: str) -> list[tuple[Cell, Cell]]:
    """All pairs of cells (level0, level1) differing only in ``factor``.

    There are 2^4 = 16 such pairs for each factor (one per setting of the
    other four factors). Used by the aggregator for paired main-effect deltas.
    """
    if factor not in FACTOR_INDEX:
        raise ValueError(f"Unknown factor {factor!r}; expected one of {FACTOR_NAMES}")
    fi = FACTOR_INDEX[factor]
    pairs: list[tuple[Cell, Cell]] = []
    for other_bits in itertools.product((0, 1), repeat=4):
        bits0 = [*other_bits[:fi], 0, *other_bits[fi:]]
        bits1 = [*other_bits[:fi], 1, *other_bits[fi:]]
        pairs.append((Cell(*bits0), Cell(*bits1)))
    return pairs


def matched_pairs_for_interaction(
    factor_a: str, factor_b: str
) -> list[tuple[Cell, Cell, Cell, Cell]]:
    """All four-tuples (00, 01, 10, 11) of cells differing in factors A and B.

    There are 2^3 = 8 such tuples (one per setting of the other three
    factors). Used for difference-of-differences interaction estimation.
    """
    if factor_a == factor_b:
        raise ValueError("matched_pairs_for_interaction requires two distinct factors")
    if factor_a not in FACTOR_INDEX or factor_b not in FACTOR_INDEX:
        raise ValueError(
            f"Unknown factor(s): {factor_a!r}, {factor_b!r}; expected one of {FACTOR_NAMES}"
        )
    ai = FACTOR_INDEX[factor_a]
    bi = FACTOR_INDEX[factor_b]
    other_indices = [i for i in range(5) if i != ai and i != bi]
    tuples: list[tuple[Cell, Cell, Cell, Cell]] = []
    for others in itertools.product((0, 1), repeat=3):

        def _bits(va: int, vb: int, _others: tuple[int, ...] = others) -> list[int]:
            row = [0, 0, 0, 0, 0]
            for idx, v in zip(other_indices, _others, strict=True):
                row[idx] = v
            row[ai] = va
            row[bi] = vb
            return row

        tuples.append(
            (
                Cell(*_bits(0, 0)),
                Cell(*_bits(0, 1)),
                Cell(*_bits(1, 0)),
                Cell(*_bits(1, 1)),
            )
        )
    return tuples
