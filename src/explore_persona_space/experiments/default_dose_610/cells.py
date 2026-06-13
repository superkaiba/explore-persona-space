# em-dash intentional
"""Task #610 — the single inverted-panel cell per chassis, built from the PARENT manifest.

``build_610_spec`` recovers the parent's base panel (bartender, french_person)
and the chassis target's NEAR slot from the committed #600
``panel_selection.json``, swaps the fixed ``qwen_default`` slot for the
chassis's replacement persona (the pair's matched-control persona — identity
asserted against the manifest's ctrl slot, the pre-registration check), and
re-asserts the #610 invariants — including the INVERSION of the parent's
qwen_default-exactly-once rule: here ``qwen_default`` must appear ZERO times.

Chassis (amendment plan v2 §2): ``mercenary`` (round 1 — replacement
journalist, NN dictator) and ``software_engineer`` (follow-up — replacement
hospice_nurse, NN data_scientist). All chassis-dependent names live in
``ChassisConfig`` (``default_dose_610.CHASSES``); the default reproduces the
round-1 mercenary behavior byte-for-byte.

``python -m explore_persona_space.experiments.default_dose_610.cells
[--chassis <name>]`` writes the chassis's design.json (committed pre-training,
plan §4.4; the software_engineer chassis writes
``eval_results/issue_610/second-chassis-dose-replication/design.json``).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from explore_persona_space.experiments.default_dose_610 import (
    ALWAYS_INCLUDE_NEGATIVE,
    CHASSES,
    EXTRA_EVAL_PERSONAS,
    SOURCE_PERSONA,
    ChassisConfig,
)
from explore_persona_space.experiments.targeted_proximity_600.cells import CellSpec600

DESIGN_SCHEMA_VERSION = "i610_design_v1"


def build_610_spec(manifest: dict, chassis: ChassisConfig = CHASSES["mercenary"]) -> CellSpec600:
    """The #610 no-default cell spec from the PARENT #600 manifest (fail-loud).

    Asserts (plan §4.2): the chassis target exists with its NEAR-slot identity
    intact; ``chassis.replacement`` == the manifest's ctrl name for that target
    (pre-registration identity check); panel of 4 distinct personas;
    ``qwen_default`` count == 0 (the inversion of the parent invariant);
    source ∉ panel; panel ∩ targets == ∅; replacement persona exactly once.
    """
    target_names = [t["name"] for t in manifest["targets"]]
    target = next((t for t in manifest["targets"] if t["name"] == chassis.chassis_target), None)
    if target is None:
        raise AssertionError(
            f"Parent manifest has no target {chassis.chassis_target!r}; targets: {target_names}"
        )
    base_panel = [b["name"] for b in manifest["base_panel"]]
    if len(base_panel) != 2:
        raise AssertionError(f"base_panel must have 2 personas, got {base_panel}")
    near_slot = target["near"]["name"]
    ctrl_name = target["ctrl"]["name"]
    if chassis.replacement_is_ctrl:
        # Pre-registration identity check (#610): the replacement IS the pair's
        # already-characterized matched-control persona, nothing else.
        if ctrl_name != chassis.replacement:
            raise AssertionError(
                f"chassis {chassis.name!r} replacement {chassis.replacement!r} != the manifest's "
                f"{chassis.chassis_target} ctrl slot {ctrl_name!r} — the pre-registered swap "
                "identity does not hold; refusing to build the spec."
            )
    else:
        # #632 proximal pick: the replacement is NOT a ctrl persona by design
        # (it is the deterministic min-distance neighbor of the assistant
        # centroid). Pre-registration: assert it does NOT collide with the
        # parent's ctrl/near slots — a non-ctrl replacement that silently
        # equaled a slot would reintroduce the #610 swap rather than the swap
        # this experiment registers. (The remaining asserts below carry the
        # rest of the disjointness against the realized panel.)
        if chassis.replacement in (ctrl_name, near_slot):
            raise AssertionError(
                f"{chassis.name!r}: proximal replacement {chassis.replacement!r} collides with a "
                f"parent slot (ctrl={ctrl_name!r}, near={near_slot!r}); refusing to build."
            )

    # The swap: the parent panel is (qwen_default, base_0, base_1, near_slot);
    # the #610 panel hands qwen_default's 200 rows to the replacement. The
    # variable-slot NN stays — the replacement is a new FIXED negative.
    panel = (chassis.replacement, base_panel[0], base_panel[1], near_slot)

    if len(set(panel)) != 4:
        raise AssertionError(f"[{chassis.new_slug}] duplicate persona in panel {panel}")
    if panel.count(ALWAYS_INCLUDE_NEGATIVE) != 0:
        raise AssertionError(
            f"[{chassis.new_slug}] {ALWAYS_INCLUDE_NEGATIVE!r} must be ABSENT from the "
            f"no-default panel (count must be 0): {panel}"
        )
    if SOURCE_PERSONA in panel:
        raise AssertionError(f"[{chassis.new_slug}] source in panel: {panel}")
    overlap = set(panel) & set(target_names)
    if overlap:
        raise AssertionError(f"[{chassis.new_slug}] panel ∩ targets != ∅: {sorted(overlap)}")
    if panel.count(chassis.replacement) != 1:
        raise AssertionError(
            f"[{chassis.new_slug}] {chassis.replacement!r} count != 1 in panel {panel}"
        )

    return CellSpec600(
        slug=chassis.new_slug,
        plain_name=chassis.new_plain_name,
        target=chassis.chassis_target,
        stratum=target["stratum"],
        condition="nodefault",
        slot_persona=chassis.replacement,
        panel=panel,
    )


def design_payload(
    manifest: dict, spec: CellSpec600, chassis: ChassisConfig = CHASSES["mercenary"]
) -> dict:
    """The committed design.json payload (panel + provenance, plan §4.2)."""
    return {
        "schema_version": DESIGN_SCHEMA_VERSION,
        "chassis": chassis.name,  # provenance only (NOT in the assert set — the
        # round-1 mercenary design.json pre-dates the field; slug/chassis_slug/
        # replacement_persona below already pin the chassis identity)
        "slug": spec.slug,
        "plain_name": spec.plain_name,
        "chassis_slug": chassis.chassis_slug,
        "chassis_target": spec.target,
        "panel": list(spec.panel),
        "replacement_persona": chassis.replacement,
        "replaced_persona": ALWAYS_INCLUDE_NEGATIVE,
        "extra_eval_personas": list(EXTRA_EVAL_PERSONAS),
        "source_persona": SOURCE_PERSONA,
        "parent_manifest_schema_version": manifest["schema_version"],
        "parent_bank_content_hash": manifest["bank_content_hash"],
        "generated_at": datetime.now(UTC).isoformat(),
        "git_commit": _git_sha(),
    }


def assert_design_matches(
    design: dict,
    manifest: dict,
    spec: CellSpec600,
    chassis: ChassisConfig = CHASSES["mercenary"],
) -> None:
    """Committed design.json must match the runtime-built spec (fail-loud)."""
    checks = {
        "schema_version": (design.get("schema_version"), DESIGN_SCHEMA_VERSION),
        "slug": (design.get("slug"), spec.slug),
        "panel": (design.get("panel"), list(spec.panel)),
        "replacement_persona": (design.get("replacement_persona"), chassis.replacement),
        "replaced_persona": (design.get("replaced_persona"), ALWAYS_INCLUDE_NEGATIVE),
        "extra_eval_personas": (design.get("extra_eval_personas"), list(EXTRA_EVAL_PERSONAS)),
        "source_persona": (design.get("source_persona"), SOURCE_PERSONA),
        "chassis_slug": (design.get("chassis_slug"), chassis.chassis_slug),
        "parent_bank_content_hash": (
            design.get("parent_bank_content_hash"),
            manifest["bank_content_hash"],
        ),
    }
    mismatches = {k: v for k, v in checks.items() if v[0] != v[1]}
    if mismatches:
        raise RuntimeError(
            f"committed design.json does not match the runtime-built #610 spec: "
            f"{mismatches} — the committed design is a different generation; "
            "refusing to train."
        )


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True, env={**os.environ}
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def main(argv: list[str] | None = None) -> int:
    """Build the spec from the parent manifest and write design.json (VM, CPU)."""
    import argparse

    from explore_persona_space.experiments.targeted_proximity_600.cells import load_manifest

    ap = argparse.ArgumentParser(description="Task #610 design.json builder")
    ap.add_argument(
        "--chassis",
        choices=sorted(CHASSES),
        default="mercenary",
        help="Chassis registry key (v2 plan §2; default = the round-1 mercenary chassis).",
    )
    ap.add_argument(
        "--manifest",
        type=Path,
        default=Path("eval_results/issue_600/panel_selection.json"),
        help="The PARENT #600 panel_selection.json.",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Where to write the committed design manifest "
        "(default: <chassis output root>/design.json).",
    )
    args = ap.parse_args(argv)
    chassis = CHASSES[args.chassis]
    out = args.out if args.out is not None else chassis.output_root_default / "design.json"
    manifest = load_manifest(args.manifest)
    spec = build_610_spec(manifest, chassis)
    payload = design_payload(manifest, spec, chassis)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"design.json written: {out} (chassis={chassis.name}, panel={list(spec.panel)})")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
