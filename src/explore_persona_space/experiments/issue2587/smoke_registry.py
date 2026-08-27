"""Enumerable smoke-downgrade registry shape for issue #2587 entrypoints.

Closes the r2 ``analysis-smoke-blindspots`` concern's CODE half: every
runtime ``--smoke`` gate DOWNGRADE (a production assertion skipped, or a
production parameter narrowed) in an issue-2587 entrypoint must be declared
in a module-level ``SMOKE_BLIND_SPOTS: tuple[SmokeDowngrade, ...]`` in that
entrypoint, and the skip sites must route through a gate helper that
REFUSES an unregistered site — so the code itself is the honest, greppable
source for the blind-spot enumeration (``smoke-blind-spots.md``: sanctioned
downgrades still enumerate; #1336 SLURM-5005 is the incident class).

Consumers (each defines its own registry; this module is only the shape):

- ``scripts/issue2587_analysis.py`` — ``production_gate`` over the five
  production cardinality assertions + the B=10,000 -> 100 narrowing.
- ``scripts/issue2587_judge.py`` — the 1,464-call arithmetic-gate skip +
  the smoke slice/cap narrowing.

Both entrypoints expose ``--list-smoke-blind-spots`` (prints the registry
as JSON, exits 0) and stamp the active registry into their smoke artifacts'
metadata, so the enumeration is mechanically extractable for markers.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

# A production ASSERTION is disabled under smoke vs a production PARAMETER
# runs at a reduced value under smoke (the gate still executes).
KINDS: tuple[str, ...] = ("assert-skipped", "param-narrowed")


@dataclass(frozen=True)
class SmokeDowngrade:
    """One registered smoke-conditional downgrade of a production gate.

    Attributes:
        site: stable snake_case id, unique within its script — the greppable
            join key between the registry, the gate call site, and the
            marker enumeration prose.
        kind: one of :data:`KINDS`.
        production: what production enforces (the gate/parameter, verbatim
            enough to locate).
        smoke: the realized smoke behavior (skipped / narrowed-to-what).
        why: why the downgrade is sanctioned — specifically why the gate
            cannot simply run at reduced n under smoke (or, for
            param-narrowed entries, what the narrowing leaves uncertified).
    """

    site: str
    kind: str
    production: str
    smoke: str
    why: str

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"SmokeDowngrade.kind {self.kind!r} not in {KINDS}")
        for field_name in ("site", "production", "smoke", "why"):
            if not getattr(self, field_name).strip():
                raise ValueError(f"SmokeDowngrade.{field_name} must be non-empty")


def validate_registry(registry: tuple[SmokeDowngrade, ...]) -> tuple[SmokeDowngrade, ...]:
    """Fail loud on duplicate sites; returns the registry for inline use."""
    sites = [e.site for e in registry]
    dupes = sorted({s for s in sites if sites.count(s) > 1})
    if dupes:
        raise ValueError(f"duplicate SmokeDowngrade sites: {dupes}")
    return registry


def format_smoke_blind_spots(registry: tuple[SmokeDowngrade, ...]) -> list[dict]:
    """JSON-ready enumeration (artifact metadata + --list-smoke-blind-spots)."""
    return [asdict(e) for e in validate_registry(registry)]
