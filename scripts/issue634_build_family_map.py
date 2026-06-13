#!/usr/bin/env python3
"""Issue #634 Phase 0: freeze the behavior role -> #594 context-family map.

Writes ``data/issue634/behavior_family_map.json`` (the Panel-B frozen map) and
asserts every role key is a real key in ``data/assistant_axis/role_list.json``.
This runs BEFORE any embedding so H1 cannot be tuned post-hoc (plan §4).

The map is copied VERBATIM from plan v2 §4's YAML block. Families are #594's 7:
``persona``, ``wildchat_prefix``, ``worked_example``, ``instruction_reword``,
``output_format``, ``behavior_instruction``, ``bare_default``. ``wildchat_prefix``
is intentionally UNREPRESENTED (a corpus-prefix context, not a role).

Plan §4 prose says "30 roles across 6 families" but the YAML block enumerates
27 distinct role keys (1 + 9 + 5 + 4 + 4 + 4). The YAML is the authoritative
frozen map; the "30" prose count is stale. The pre-registered floor (>=12 roles
total AND >=2 per tested family) is met by the 27-role map: 26 roles across the
5 TESTED families (persona 9, behavior_instruction 5, output_format 4,
instruction_reword 4, worked_example 4), plus ``bare_default`` (1, the
null-anchor family, not a tested family).

Usage::

    uv run python scripts/issue634_build_family_map.py
    uv run python scripts/issue634_build_family_map.py --data-dir data/assistant_axis \\
        --out data/issue634/behavior_family_map.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

logger = logging.getLogger("issue634_family_map")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "assistant_axis"
DEFAULT_OUT = PROJECT_ROOT / "data" / "issue634" / "behavior_family_map.json"

# The frozen role -> #594-family map (plan v2 §4 YAML, verbatim). Keys are role
# names; values are the matching #594 context family. ``villain_proxy: rogue``
# and the single ``assistant -> bare_default`` null anchor are the only two
# borderline picks (flagged in plan §8); ``wildchat_prefix`` is unrepresented.
FROZEN_FAMILY_MAP: dict[str, str] = {
    # bare_default — neutral general-purpose assistant (null-anchor family).
    "assistant": "bare_default",
    # persona — rich character/identity roles = the persona context family.
    "detective": "persona",
    "pirate": "persona",
    "philosopher": "persona",
    "warrior": "persona",
    "rogue": "persona",  # villain proxy — closest harmful-character analogue
    "hacker": "persona",
    "bard": "persona",
    "jester": "persona",
    "sage": "persona",
    # behavior_instruction — roles that ARE a behavioral directive.
    "skeptic": "behavior_instruction",
    "contrarian": "behavior_instruction",
    "devils_advocate": "behavior_instruction",
    "perfectionist": "behavior_instruction",
    "pacifist": "behavior_instruction",
    # output_format — roles defined by the SHAPE of their output.
    "summarizer": "output_format",
    "proofreader": "output_format",
    "editor": "output_format",
    "translator": "output_format",
    # instruction_reword — roles that restate/rephrase/explain an instruction.
    "tutor": "instruction_reword",
    "interpreter": "instruction_reword",
    "instructor": "instruction_reword",
    "teacher": "instruction_reword",
    # worked_example — roles that demonstrate solving a task step-by-step.
    "mathematician": "worked_example",
    "programmer": "worked_example",
    "debugger": "worked_example",
    "analyst": "worked_example",
}

# bare_default is the null-anchor, not a tested family (a single role cannot
# form a >=2 within-family neighborhood). The per-family floor (>=2) applies to
# the tested families only.
NULL_ANCHOR_FAMILY = "bare_default"
FLOOR_MIN_TOTAL = 12
FLOOR_MIN_PER_TESTED_FAMILY = 2


def load_role_list(data_dir: Path) -> dict[str, str]:
    """Load role_list.json (a dict: role name -> short description)."""
    with open(data_dir / "role_list.json") as f:
        rl = json.load(f)
    if not isinstance(rl, dict):
        raise ValueError(f"role_list.json must be a dict, got {type(rl).__name__}")
    return rl


def map_hash(family_map: dict[str, str]) -> str:
    """Stable sha256 over the frozen map (canonical JSON; provenance pin)."""
    canonical = json.dumps(family_map, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def family_counts(family_map: dict[str, str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for fam in family_map.values():
        counts[fam] = counts.get(fam, 0) + 1
    return counts


def check_floor(family_map: dict[str, str]) -> dict:
    """Evaluate the pre-registered Panel-B floor. Returns a verdict dict.

    Floor: >=12 roles total AND >=2 roles per TESTED family (bare_default is the
    null anchor, excluded from the per-family floor). Below the floor, H1 reads
    UNDERPOWERED rather than falsified (plan §3/§4) — the verdict is recorded
    for downstream consumers; this builder does not silently shrink the map.
    """
    counts = family_counts(family_map)
    tested = {f: c for f, c in counts.items() if f != NULL_ANCHOR_FAMILY}
    n_total = len(family_map)
    under_per_family = {f: c for f, c in tested.items() if c < FLOOR_MIN_PER_TESTED_FAMILY}
    meets = n_total >= FLOOR_MIN_TOTAL and not under_per_family
    return {
        "meets_floor": bool(meets),
        "n_roles_total": n_total,
        "n_tested_families": len(tested),
        "tested_family_counts": tested,
        "null_anchor_family": NULL_ANCHOR_FAMILY,
        "families_below_per_family_floor": under_per_family,
        "floor_min_total": FLOOR_MIN_TOTAL,
        "floor_min_per_tested_family": FLOOR_MIN_PER_TESTED_FAMILY,
    }


def build(data_dir: Path, out_path: Path) -> dict:
    """Assert all keys in role_list, write the map JSON, return the payload."""
    role_list = load_role_list(data_dir)
    role_keys = set(role_list)
    missing = [r for r in FROZEN_FAMILY_MAP if r not in role_keys]
    if missing:
        raise AssertionError(
            f"frozen Panel-B map has {len(missing)} role keys absent from "
            f"role_list.json ({data_dir / 'role_list.json'}): {sorted(missing)}"
        )
    floor = check_floor(FROZEN_FAMILY_MAP)
    payload = {
        "schema_version": 1,
        "description": (
            "Issue #634 Panel-B frozen behavior-role -> #594-context-family map "
            "(plan v2 §4). Built BEFORE any embedding so H1 cannot be tuned "
            "post-hoc. Every key is a verified role_list.json key."
        ),
        "map": FROZEN_FAMILY_MAP,
        "map_sha256": map_hash(FROZEN_FAMILY_MAP),
        "family_counts": family_counts(FROZEN_FAMILY_MAP),
        "floor": floor,
        "role_list_size": len(role_list),
        "data_dir": str(data_dir),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    logger.info(
        "Wrote %s — %d roles, %d families, sha256=%s, meets_floor=%s",
        out_path,
        len(FROZEN_FAMILY_MAP),
        len(payload["family_counts"]),
        payload["map_sha256"][:16],
        floor["meets_floor"],
    )
    if not floor["meets_floor"]:
        logger.warning(
            "Panel-B floor NOT met (%s) — H1 will read UNDERPOWERED downstream",
            floor,
        )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Issue #634 Phase 0: freeze the Panel-B map.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="assistant_axis dir holding role_list.json (default data/assistant_axis)",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    build(args.data_dir, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
