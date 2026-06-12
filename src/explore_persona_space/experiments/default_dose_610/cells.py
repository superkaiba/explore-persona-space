# em-dash intentional
"""Task #610 — the single inverted-panel cell, built from the PARENT manifest.

``build_610_spec`` recovers the parent's base panel (bartender, french_person)
and mercenary's NEAR slot (dictator) from the committed #600
``panel_selection.json``, swaps the fixed ``qwen_default`` slot for
``journalist`` (the pair's matched-control persona — identity asserted against
the manifest's ctrl slot, the pre-registration check), and re-asserts the #610
invariants — including the INVERSION of the parent's qwen_default-exactly-once
rule: here ``qwen_default`` must appear ZERO times.

``python -m explore_persona_space.experiments.default_dose_610.cells`` writes
``eval_results/issue_610/design.json`` (committed pre-training, plan §4.4).
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
    CHASSIS_SLUG,
    CHASSIS_TARGET,
    EXTRA_EVAL_PERSONAS,
    NEW_PLAIN_NAME,
    NEW_SLUG,
    REPLACEMENT_PERSONA,
    SOURCE_PERSONA,
)
from explore_persona_space.experiments.targeted_proximity_600.cells import CellSpec600

DESIGN_SCHEMA_VERSION = "i610_design_v1"


def build_610_spec(manifest: dict) -> CellSpec600:
    """The #610 no-default cell spec from the PARENT #600 manifest (fail-loud).

    Asserts (plan §4.2): the chassis target exists with NEAR slot ``dictator``-
    class identity intact; ``REPLACEMENT_PERSONA`` == the manifest's mercenary
    ctrl name (pre-registration identity check); panel of 4 distinct personas;
    ``qwen_default`` count == 0 (the inversion of the parent invariant);
    source ∉ panel; panel ∩ targets == ∅; replacement persona exactly once.
    """
    target_names = [t["name"] for t in manifest["targets"]]
    target = next((t for t in manifest["targets"] if t["name"] == CHASSIS_TARGET), None)
    if target is None:
        raise AssertionError(
            f"Parent manifest has no target {CHASSIS_TARGET!r}; targets: {target_names}"
        )
    base_panel = [b["name"] for b in manifest["base_panel"]]
    if len(base_panel) != 2:
        raise AssertionError(f"base_panel must have 2 personas, got {base_panel}")
    near_slot = target["near"]["name"]
    ctrl_name = target["ctrl"]["name"]
    # Pre-registration identity check: the replacement IS the pair's
    # already-characterized matched-control persona, nothing else.
    if ctrl_name != REPLACEMENT_PERSONA:
        raise AssertionError(
            f"REPLACEMENT_PERSONA {REPLACEMENT_PERSONA!r} != the manifest's "
            f"{CHASSIS_TARGET} ctrl slot {ctrl_name!r} — the pre-registered swap "
            "identity does not hold; refusing to build the spec."
        )

    # The swap: the parent panel is (qwen_default, base_0, base_1, near_slot);
    # the #610 panel hands qwen_default's 200 rows to the replacement. The
    # variable-slot NN (dictator) STAYS — journalist is a new FIXED negative.
    panel = (REPLACEMENT_PERSONA, base_panel[0], base_panel[1], near_slot)

    if len(set(panel)) != 4:
        raise AssertionError(f"[{NEW_SLUG}] duplicate persona in panel {panel}")
    if panel.count(ALWAYS_INCLUDE_NEGATIVE) != 0:
        raise AssertionError(
            f"[{NEW_SLUG}] {ALWAYS_INCLUDE_NEGATIVE!r} must be ABSENT from the "
            f"no-default panel (count must be 0): {panel}"
        )
    if SOURCE_PERSONA in panel:
        raise AssertionError(f"[{NEW_SLUG}] source in panel: {panel}")
    overlap = set(panel) & set(target_names)
    if overlap:
        raise AssertionError(f"[{NEW_SLUG}] panel ∩ targets != ∅: {sorted(overlap)}")
    if panel.count(REPLACEMENT_PERSONA) != 1:
        raise AssertionError(f"[{NEW_SLUG}] {REPLACEMENT_PERSONA!r} count != 1 in panel {panel}")

    return CellSpec600(
        slug=NEW_SLUG,
        plain_name=NEW_PLAIN_NAME,
        target=CHASSIS_TARGET,
        stratum=target["stratum"],
        condition="nodefault",
        slot_persona=REPLACEMENT_PERSONA,
        panel=panel,
    )


def design_payload(manifest: dict, spec: CellSpec600) -> dict:
    """The committed design.json payload (panel + provenance, plan §4.2)."""
    return {
        "schema_version": DESIGN_SCHEMA_VERSION,
        "slug": spec.slug,
        "plain_name": spec.plain_name,
        "chassis_slug": CHASSIS_SLUG,
        "chassis_target": spec.target,
        "panel": list(spec.panel),
        "replacement_persona": REPLACEMENT_PERSONA,
        "replaced_persona": ALWAYS_INCLUDE_NEGATIVE,
        "extra_eval_personas": list(EXTRA_EVAL_PERSONAS),
        "source_persona": SOURCE_PERSONA,
        "parent_manifest_schema_version": manifest["schema_version"],
        "parent_bank_content_hash": manifest["bank_content_hash"],
        "generated_at": datetime.now(UTC).isoformat(),
        "git_commit": _git_sha(),
    }


def assert_design_matches(design: dict, manifest: dict, spec: CellSpec600) -> None:
    """Committed design.json must match the runtime-built spec (fail-loud)."""
    checks = {
        "schema_version": (design.get("schema_version"), DESIGN_SCHEMA_VERSION),
        "slug": (design.get("slug"), spec.slug),
        "panel": (design.get("panel"), list(spec.panel)),
        "replacement_persona": (design.get("replacement_persona"), REPLACEMENT_PERSONA),
        "replaced_persona": (design.get("replaced_persona"), ALWAYS_INCLUDE_NEGATIVE),
        "extra_eval_personas": (design.get("extra_eval_personas"), list(EXTRA_EVAL_PERSONAS)),
        "source_persona": (design.get("source_persona"), SOURCE_PERSONA),
        "chassis_slug": (design.get("chassis_slug"), CHASSIS_SLUG),
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
        "--manifest",
        type=Path,
        default=Path("eval_results/issue_600/panel_selection.json"),
        help="The PARENT #600 panel_selection.json.",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("eval_results/issue_610/design.json"),
        help="Where to write the committed design manifest.",
    )
    args = ap.parse_args(argv)
    manifest = load_manifest(args.manifest)
    spec = build_610_spec(manifest)
    payload = design_payload(manifest, spec)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"design.json written: {args.out} (panel={list(spec.panel)})")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
