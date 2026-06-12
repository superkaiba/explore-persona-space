# ruff: noqa: RUF002  # em-dash intentional
"""Task #600 — cell registry built from the committed design manifest.

CELL_SPECS_600 carries EXPLICIT 4-persona panel lists per cell (plan §4.4):
no placement-derived selection, no qwen_default auto-prepend path — the
 #527/#538 realized-panel incident class is closed structurally. The manifest
(``eval_results/issue_600/panel_selection.json``, committed pre-training) is
the single source of truth for targets / NEAR / CONTROL / base panel; this
module only materializes it into per-cell specs and re-asserts disjointness.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from explore_persona_space.experiments.targeted_proximity_600 import (
    ALWAYS_INCLUDE_NEGATIVE,
    SOURCE_PERSONA,
)
from explore_persona_space.experiments.targeted_proximity_600.select_panels import (
    SCHEMA_VERSION,
)

CONDITIONS = ("near", "ctrl")


@dataclass(frozen=True)
class CellSpec600:
    """One #600 training cell: a (target, condition) pair with its explicit panel."""

    slug: str
    plain_name: str
    target: str
    stratum: str
    condition: str  # "near" | "ctrl"
    slot_persona: str
    panel: tuple[str, str, str, str]  # (qwen_default, base_mid_1, base_mid_2, slot)


def load_manifest(path: Path) -> dict:
    """Load + schema-check the committed panel_selection.json (fail-loud)."""
    if not path.exists():
        raise FileNotFoundError(
            f"Design manifest missing at {path}. Run select_panels.py on the VM and "
            "commit it to the issue branch BEFORE any training (plan §4.3 step 7)."
        )
    manifest = json.loads(path.read_text())
    sv = manifest.get("schema_version")
    if sv != SCHEMA_VERSION:
        raise AssertionError(
            f"Manifest {path} has schema_version={sv!r}, expected {SCHEMA_VERSION!r}."
        )
    if len(manifest.get("targets", [])) == 0:
        raise AssertionError(f"Manifest {path} carries no targets.")
    return manifest


def cell_specs_from_manifest(manifest: dict) -> tuple[CellSpec600, ...]:
    """Materialize the 12 (target × condition) cells with explicit panels.

    Asserts per cell: panel length 4, ``qwen_default`` exactly once, source
    absent, panel ∩ targets = ∅ (plan §4.4 hard disjointness).
    """
    base_panel = [b["name"] for b in manifest["base_panel"]]
    if len(base_panel) != 2:
        raise AssertionError(f"base_panel must have 2 personas, got {base_panel}")
    target_names = [t["name"] for t in manifest["targets"]]
    specs: list[CellSpec600] = []
    for t in manifest["targets"]:
        for condition in CONDITIONS:
            slot = t["near"]["name"] if condition == "near" else t["ctrl"]["name"]
            panel = (ALWAYS_INCLUDE_NEGATIVE, base_panel[0], base_panel[1], slot)
            if len(set(panel)) != 4:
                raise AssertionError(f"[{t['name']}/{condition}] duplicate persona in {panel}")
            if panel.count(ALWAYS_INCLUDE_NEGATIVE) != 1:
                raise AssertionError(f"[{t['name']}/{condition}] qwen_default count != 1: {panel}")
            if SOURCE_PERSONA in panel:
                raise AssertionError(f"[{t['name']}/{condition}] source in panel: {panel}")
            overlap = set(panel) & set(target_names)
            if overlap:
                raise AssertionError(
                    f"[{t['name']}/{condition}] panel ∩ targets != ∅: {sorted(overlap)}"
                )
            label = (
                "Nearest-neighbor negative"
                if condition == "near"
                else "Distance-matched far control"
            )
            specs.append(
                CellSpec600(
                    slug=f"c600_{t['name']}_{condition}",
                    plain_name=f"{label} for {t['name']}",
                    target=t["name"],
                    stratum=t["stratum"],
                    condition=condition,
                    slot_persona=slot,
                    panel=panel,
                )
            )
    if len(specs) != 2 * len(manifest["targets"]):
        raise AssertionError(f"Expected {2 * len(manifest['targets'])} cells, built {len(specs)}.")
    return tuple(specs)


def first_near_slug(specs: tuple[CellSpec600, ...]) -> str:
    """The smoke cell: the FIRST NEAR cell in registry order (plan §4.7)."""
    for s in specs:
        if s.condition == "near":
            return s.slug
    raise AssertionError("No NEAR cell in the registry — manifest is malformed.")
