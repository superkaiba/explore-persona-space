"""Factor encoding + cell key generation for the 2^5 design.

Five binary factors (each level ∈ {0, 1}):

  F1 — system-prompt length         | 0 = short (~6 tokens) | 1 = long (~1000 tokens)
  F2 — target completion length     | 0 = short (~50 tokens) | 1 = long (~1050 tokens)
  F3 — persona-presence in answer   | 0 = absent (generic)   | 1 = present (persona-rich)
  F4 — on-policy training data      | 0 = off-policy (Claude)| 1 = on-policy (base model)
  F5 — loss masking                 | 0 = full CE            | 1 = marker-only-loss

The cell key is the binary string `F1F2F3F4F5` (e.g. `01101` = sys-short,
ans-long, persona-present, off-policy, marker-only-loss).

For the Phase-1 resolution-III fractional-factorial smoke we use generators
F4 = F1·F2 and F5 = F1·F3 (encoding {-1, +1} -> XOR on bits) — this aliases
F4 ≡ F1·F2 and F5 ≡ F1·F3 but lets us screen all 5 factors in 8 cells.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Iterator


@dataclass(frozen=True)
class Cell:
    """A single training cell in the 2^5 design."""

    f1: int  # system-prompt length: 0=short, 1=long
    f2: int  # target completion length: 0=short, 1=long
    f3: int  # persona-presence in answer: 0=absent, 1=present
    f4: int  # data source: 0=off-policy, 1=on-policy
    f5: int  # loss masking: 0=full CE, 1=marker-only

    @property
    def key(self) -> str:
        """Five-character bitstring identifier, e.g. `01101`."""
        return f"{self.f1}{self.f2}{self.f3}{self.f4}{self.f5}"

    @property
    def bits(self) -> tuple[int, int, int, int, int]:
        return (self.f1, self.f2, self.f3, self.f4, self.f5)

    def with_factor(self, factor_index: int, value: int) -> "Cell":
        """Return a copy with `factor_index` (0-based: F1..F5) set to `value`."""
        bits = list(self.bits)
        bits[factor_index] = value
        return Cell(*bits)


def all_full_cells() -> list[Cell]:
    """The full 2^5 = 32 cells, ordered by binary key for stable iteration."""
    cells = [Cell(*bits) for bits in itertools.product((0, 1), repeat=5)]
    return sorted(cells, key=lambda c: c.key)


def smoke_cells() -> list[Cell]:
    """The 8 resolution-III fractional-factorial smoke cells.

    Generators: F4 = F1 XOR F2, F5 = F1 XOR F3. F1, F2, F3 are the base factors
    (full 2^3 = 8 combos); F4 and F5 are then derived.
    """
    cells: list[Cell] = []
    for f1, f2, f3 in itertools.product((0, 1), repeat=3):
        f4 = f1 ^ f2
        f5 = f1 ^ f3
        cells.append(Cell(f1, f2, f3, f4, f5))
    return cells


def iter_factor_levels() -> Iterator[tuple[str, list[int]]]:
    """Yield (factor_name, level_indicators) tuples used for main-effects coding."""
    factors = ["F1", "F2", "F3", "F4", "F5"]
    for i, name in enumerate(factors):
        yield name, [c.bits[i] for c in all_full_cells()]


FACTOR_NAMES = ("F1", "F2", "F3", "F4", "F5")


# Pairwise interactions used by the aggregator. F1×F2 is the pre-registered
# primary; the other nine are exploratory.
INTERACTION_PAIRS: list[tuple[str, str]] = [
    ("F1", "F2"),  # pre-registered (system-prompt length × answer-format length)
    ("F1", "F3"),
    ("F1", "F4"),
    ("F1", "F5"),
    ("F2", "F3"),
    ("F2", "F4"),
    ("F2", "F5"),
    ("F3", "F4"),
    ("F3", "F5"),
    ("F4", "F5"),
]

PREREGISTERED_INTERACTIONS: set[tuple[str, str]] = {("F1", "F2")}


def is_preregistered(pair: tuple[str, str]) -> bool:
    a, b = sorted(pair)
    return (a, b) in {tuple(sorted(p)) for p in PREREGISTERED_INTERACTIONS}
