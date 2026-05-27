"""Factor encoding + cell enumeration for the 2^4 × 3 design (task #397).

Four binary factors (A, B, C, D) + one ordinal factor E with K=3 levels.
Plan v4 §4.1 + §4.3 are authoritative:

  A (system-prompt length)    | 0 = short (6-20 Qwen tokens)   | 1 = long (~1000 tokens)
  B (answer-format length)    | 0 = short (40-80 tokens band)  | 1 = long (900-1200 band)
  C (persona framing in sys)  | 0 = persona role prompt        | 1 = lexically matched non-persona
  D (data policy)             | 0 = on-policy (base Qwen)      | 1 = off-policy (Claude)
  E (loss-mask restrictiveness, ORDINAL K=3)
                              | 0 = marker+EOT  (~2 tok, marker_only_loss=True,  tail=0)
                              | 1 = tail-32     (~32 tok, marker_only_loss=True,  tail=32)
                              | 2 = whole-comp  (~600 tok, marker_only_loss=False, tail=0)

The cell key is the 5-character bitstring ``ABCDE`` where ``E`` is a single
digit in ``{0, 1, 2}`` (e.g. ``10012`` = A=1, B=0, C=0, D=1, E=2). The A=0 ×
C=1 corner is dropped at preflight per #383 (4 ABCD combos × 3 E levels × 3
sources = 36 cells × 3 seeds dropped → 108 valid cells per seed × 3 seeds =
324 (cell × seed) runs).

This module is a Phase 1 (TDD) stub for cell enumeration. Real implementation
arrives in Phase 2 after user approves the proposed test surface
(``epm:approve-tests v1``).
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

FACTOR_NAMES: tuple[str, ...] = ("A", "B", "C", "D", "E")
FACTOR_INDEX: dict[str, int] = {name: i for i, name in enumerate(FACTOR_NAMES)}

# Plan v4 §4.3 — operational meaning of each E level.
FACTOR_DESCRIPTIONS: dict[str, dict[int, str]] = {
    "A": {0: "system prompt: short", 1: "system prompt: long (~1000 tokens)"},
    "B": {0: "answer format: short (1 sentence)", 1: "answer format: long essay"},
    "C": {0: "system prompt: persona role", 1: "system prompt: non-persona background"},
    "D": {0: "data: on-policy (base Qwen)", 1: "data: off-policy (Claude)"},
    "E": {
        0: "loss: marker+EOT only (~2 tok; marker_only_loss=True, tail=0)",
        1: "loss: marker+tail-32 (~32 tok; marker_only_loss=True, tail=32)",
        2: "loss: whole-completion (~600 tok; marker_only_loss=False, tail=0)",
    },
}

E_LEVELS: tuple[int, int, int] = (0, 1, 2)


@dataclass(frozen=True)
class Cell:
    """A single training cell in the 2^4 × 3 design (4 binary factors + ordinal E)."""

    a: int
    b: int
    c: int
    d: int
    e: int

    def __post_init__(self) -> None:
        for name, value, allowed in zip(
            ("A", "B", "C", "D"),
            (self.a, self.b, self.c, self.d),
            ((0, 1),) * 4,
            strict=True,
        ):
            if value not in allowed:
                raise ValueError(f"Factor {name} must be 0 or 1; got {value!r}")
        if self.e not in E_LEVELS:
            raise ValueError(f"Factor E must be 0, 1, or 2; got {self.e!r}")

    @property
    def key(self) -> str:
        """Five-character cell key ``ABCDE`` (E is a single digit 0/1/2)."""
        return f"{self.a}{self.b}{self.c}{self.d}{self.e}"

    @property
    def bits(self) -> tuple[int, int, int, int, int]:
        return (self.a, self.b, self.c, self.d, self.e)

    def level(self, factor: str) -> int:
        if factor not in FACTOR_INDEX:
            raise ValueError(f"Unknown factor {factor!r}; expected one of {FACTOR_NAMES}")
        return self.bits[FACTOR_INDEX[factor]]

    @classmethod
    def from_key(cls, key: str) -> Cell:
        """Parse a 5-character ``ABCDE`` key with E ∈ {0, 1, 2}."""
        if len(key) != 5:
            raise ValueError(f"Cell key must be exactly 5 chars; got {key!r}")
        if any(ch not in "01" for ch in key[:4]):
            raise ValueError(f"A/B/C/D must each be 0 or 1; got {key!r}")
        if key[4] not in "012":
            raise ValueError(f"E must be 0, 1, or 2; got {key!r}")
        return cls(int(key[0]), int(key[1]), int(key[2]), int(key[3]), int(key[4]))

    def with_factor(self, factor: str, value: int) -> Cell:
        """Return a copy with the named factor set to ``value``."""
        if factor not in FACTOR_INDEX:
            raise ValueError(f"Unknown factor {factor!r}; expected one of {FACTOR_NAMES}")
        bits = list(self.bits)
        bits[FACTOR_INDEX[factor]] = value
        return Cell(*bits)


def all_full_cells() -> list[Cell]:
    """All 2^4 × 3 = 48 nominal cells (BEFORE the A=0 × C=1 preflight drop).

    Returns a sorted list by canonical key. Preflight drops A=0 × C=1 corners
    (4 ABCD combos × 3 E levels = 12 cells), leaving 36 valid cells per source
    × 3 sources = 108 cells per seed × 3 seeds = 324 (cell × seed) runs.
    """
    cells = [
        Cell(a, b, c, d, e) for a, b, c, d in itertools.product((0, 1), repeat=4) for e in E_LEVELS
    ]
    return sorted(cells, key=lambda c: c.key)


def valid_cells_per_source() -> list[Cell]:
    """All cells after dropping the A=0 × C=1 preflight corner.

    Plan v4 §4.1: per source, 16 binary ABCD combos − 4 dropped (A=0×C=1) = 12
    valid combos × 3 E levels = **36 valid cells per source × 3 seeds**.
    Pooled across 3 sources: 108 cells per seed × 3 seeds = 324 (cell × seed)
    runs.
    """
    return [c for c in all_full_cells() if not (c.a == 0 and c.c == 1)]


def matched_pairs_for_factor(
    factor: str,
    e_subset: tuple[int, ...] | None = None,
) -> list[tuple[Cell, Cell]]:
    """Cells differing only in ``factor``.

    For binary factors (A/B/C/D), returns ``(level0, level1)`` pairs.
    ``e_subset`` restricts E to a subset of ``{0, 1, 2}`` (None = all E
    levels). The canonical H1 estimator uses ``e_subset=(0, 2)`` to restrict
    to the E0+E2 binary contrast (see plan v4 §4.1, "H1 canonical"). Page's L
    (H2) operates on matched E0/E1/E2 triples and is exposed separately.

    For factor "E", this function raises ``NotImplementedError`` in Phase 1 —
    matched E triples are exposed via ``matched_triples_for_e()`` once Phase 2
    lands; the test surface for H2 uses synthetic data, not this helper.
    """
    raise NotImplementedError(
        "matched_pairs_for_factor is a Phase 1 (TDD) stub; implementation lands "
        "in Phase 2 after user approves the proposed test surface."
    )


def matched_triples_for_e() -> list[tuple[Cell, Cell, Cell]]:
    """All matched E0/E1/E2 triples across the (A, B, C, D) settings.

    Plan v4 §4.1: H2 Page's L over **108 blocks × 3 ordered E levels** uses
    one triple per (seed, source, A, B, C, D); this helper returns the
    underlying ABCD-only enumeration of E triples (without the source / seed
    multiplication; aggregator handles those).
    """
    raise NotImplementedError(
        "matched_triples_for_e is a Phase 1 (TDD) stub; implementation lands "
        "in Phase 2 after user approves the proposed test surface."
    )
