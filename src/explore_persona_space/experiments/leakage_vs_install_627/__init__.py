"""Task #627 — leakage-vs-implantation control.

Re-reads the program's flagship cross-condition leakage comparisons at matched
install strength. Three reads (plan §1): matched-install comparison (primary,
the new #608 GPU measurement), per-cell leakage-to-install fractions in a
non-saturating space, and leakage-vs-install dose curves from existing
trajectories.

Phases (plan §4):
    Phase 0  scripts/i627_inventory.py            — inventory + 3 registered manifests
    Phase 1  scripts/i627_matched_install_panel.py — pod-side 24-cell bystander panel
    Phase 2  scripts/i627_judge_and_match.py       — judge pass + matched-install stats
    Phase 3  scripts/i627_analyze_{marker,606,514}.py + i627_synthesize.py + i627_figures.py

Inherits the #608 module set verbatim (ported from ``origin/issue-608`` @
``7835f69fd``): ``sycophancy_posonly_608`` (constants, prefetch, judge pass
helpers) + ``sycophancy_implantation_411`` (vLLM panel eval, Haiku judge,
Cohen's kappa). The #606 interpolation/bootstrap mechanics are ported verbatim
in ``interp.py`` (plan §13 item 1).
"""

from __future__ import annotations

import json
from pathlib import Path

# ----- inherited identities (Source: #608) ----------------------------------

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
SEED = 42
HF_MODEL_REPO = "superkaiba1/explore-persona-space"
HF_DATA_REPO = "superkaiba1/explore-persona-space-data"
HF_627_DATA_PREFIX = "issue627_leakage_vs_install"

ARMS: tuple[str, ...] = ("contrastive_dense", "posonly_dose_dense")
SOURCES: tuple[str, ...] = (
    "villain",
    "comedian",
    "assistant",
    "qwen_default",
    "software_engineer",
    "kindergarten_teacher",
)


# HF model-repo prefix per (arm, source, step): the reused #608 sub-ceiling
# adapters (fitness-checked in plan §10/§12; re-verified at Phase-0 + prefetch).
def adapter_hub_prefix(arm: str, source: str, step: int) -> str:
    """`adapters/issue_608/sub_ceiling/<arm>/<source>_seed42/step_<K>` (plan §4)."""
    if arm not in ARMS:
        raise ValueError(f"Unknown arm {arm!r}; expected one of {ARMS}")
    if source not in SOURCES:
        raise ValueError(f"Unknown source {source!r}; expected one of {SOURCES}")
    return f"adapters/issue_608/sub_ceiling/{arm}/{source}_seed{SEED}/step_{int(step)}"


# ----- registered matching parameters (plan §11, one Source per value) -------

INSTALL_TARGET = 0.50  # Source: #606
INSTALL_BAND = (0.40, 0.60)  # Source: #606 (measured-read band)
BRACKET_WIDTH_GUARD = 0.60  # Source: #606 follow-up incident; value ungrounded
BRACKET_WIDTH_SENSITIVITY = 0.45  # plan §11 sensitivity value
PARITY_TOLERANCE = 0.08  # Source: #606 cross-instrument tolerance
EQUIVALENCE_BAND = (-0.05, 0.05)  # Source: #606 (H1 dose-artifact verdict band)
H1_SURVIVE_GAP = 0.10  # plan §3 H1 survive threshold
BOOTSTRAP_N = 10_000  # Source: #606
BOOTSTRAP_RNG_SEED = 42
KAPPA_GATE = 0.7  # Source: #608
SPOTCHECK_N = 200  # Source: #608
# Marker-fraction denominator floor (plan §11: ungrounded — sensitivity-swept).
MARGIN_DENOM_FLOOR = 2.0
MARGIN_DENOM_FLOOR_SENSITIVITY = (1.0, 4.0)

# Smoke-cell parity gate (plan §7 gate 1): regenerated source own-rate at
# villain/contrastive_dense/step_18 within ±0.08 of the committed 0.416.
SMOKE_CELL = ("villain", "contrastive_dense", 18)
SMOKE_CELL_COMMITTED_OWN_RATE = 0.416

# The #608 context-collision cell, excluded from the primary H1 panel
# (collision-robust 5-source panel; plan §3 / #608 registered convention).
COLLISION_SOURCE = "qwen_default"

# ----- registered matched-install bracket table (plan §4) --------------------
# Per source, per arm: (lo_step, lo_committed_rate, hi_step, hi_committed_rate).
# FIRST-crossing pairs of own-rate 0.50 on the committed sub-ceiling
# trajectories; Phase 0 re-derives this table from
# eval_results/issue_608/sub-ceiling-install/analyze_summary_subceiling.json
# and HARD-FAILS on any divergence (the registered cells are frozen here).
REGISTERED_BRACKETS: dict[str, dict[str, tuple[int, float, int, float]]] = {
    "villain": {
        "contrastive_dense": (18, 0.416, 26, 0.718),
        "posonly_dose_dense": (9, 0.322, 13, 0.714),
    },
    "comedian": {
        "contrastive_dense": (18, 0.376, 26, 0.666),
        "posonly_dose_dense": (13, 0.412, 18, 0.888),
    },
    "assistant": {
        "contrastive_dense": (44, 0.492, 88, 0.936),
        "posonly_dose_dense": (13, 0.094, 18, 0.604),
    },
    "qwen_default": {
        "contrastive_dense": (26, 0.334, 35, 0.700),
        "posonly_dose_dense": (13, 0.198, 18, 0.856),
    },
    "software_engineer": {
        "contrastive_dense": (26, 0.388, 35, 0.768),
        "posonly_dose_dense": (13, 0.226, 18, 0.814),
    },
    "kindergarten_teacher": {
        "contrastive_dense": (18, 0.250, 26, 0.536),
        "posonly_dose_dense": (13, 0.444, 18, 0.826),
    },
}


def parse_cell_token(tok: str) -> tuple[str, str, int]:
    """``"villain:contrastive_dense:18"`` -> (source, arm, step). Fail-loud."""
    parts = tok.strip().split(":")
    if len(parts) != 3:
        raise ValueError(f"Bad cell {tok!r}: expected <source>:<arm>:<step>")
    source, arm, step_s = parts
    if source not in SOURCES:
        raise ValueError(f"Bad cell {tok!r}: source must be one of {SOURCES}")
    if arm not in ARMS:
        raise ValueError(f"Bad cell {tok!r}: arm must be one of {ARMS}")
    return source, arm, int(step_s)


def cell_id(source: str, arm: str, step: int) -> str:
    return f"{source}:{arm}:{int(step)}"


def registered_cells() -> list[tuple[str, str, int]]:
    """The 24 registered checkpoint-cells, smoke cell FIRST (so ``--cells 1``
    is exactly the registered smoke cell), then deterministic order."""
    cells: list[tuple[str, str, int]] = [SMOKE_CELL]
    for source in SOURCES:
        for arm in ARMS:
            lo_step, _, hi_step, _ = REGISTERED_BRACKETS[source][arm]
            for step in (lo_step, hi_step):
                cell = (source, arm, step)
                if cell != SMOKE_CELL:
                    cells.append(cell)
    if len(cells) != 24:
        raise AssertionError(f"Expected 24 registered cells, built {len(cells)}")
    return cells


def load_cells_manifest(path: Path) -> list[dict]:
    """Load + validate the Phase-0 matched-install cell manifest."""
    with open(path) as f:
        manifest = json.load(f)
    cells = manifest["cells"]
    if len(cells) != 24:
        raise ValueError(f"{path}: expected 24 cells, found {len(cells)}")
    seen = set()
    for c in cells:
        key = (c["source"], c["arm"], int(c["step"]))
        parse_cell_token(cell_id(*key))  # validates source/arm
        if key in seen:
            raise ValueError(f"{path}: duplicate cell {key}")
        seen.add(key)
    return cells


# ----- #601 marker slab registry (plan §2 / §4 Phase 3) ----------------------

# Mix-arm classification for the marker-side contrast (H2). The #601 negative
# rows were gradient-dead (loss-suppression flag off) — the contrast is mix
# COMPOSITION under flag-off loss placement, not live-contrastive training
# (scope caveat carried into every H2 read; plan §2).
MARKER_CONTRASTIVE_CELLS: tuple[str, ...] = (
    "ratio4to1_100p400n",
    "ratio4to1_100p400n_T128",
    "ratio4to1_400p1600n",
    "dense_200p400n",
    "dense_200p800n",
    "dense_200p1600n",
    "c472_anchor",
    "c472_negex_100",
    "c472_negex_400",
)
MARKER_POSONLY_CELLS: tuple[str, ...] = (
    "posonly_200p_T130",
    "dense_200p0n",
    "posonly_alllinear_lr5e6",
    "posonly_attn_lr5e6",
    "c472_noneg",
)
# negatives-only cell: neither arm of the mix contrast (no positives -> no
# install dial); kept out of H2 entirely.
MARKER_EXCLUDED_CELLS: tuple[str, ...] = ("negonly_0p800n",)


def marker_cell_arm(cell: str) -> str | None:
    """``contrastive`` / ``posonly`` / None (excluded)."""
    if cell in MARKER_CONTRASTIVE_CELLS:
        return "contrastive"
    if cell in MARKER_POSONLY_CELLS:
        return "posonly"
    if cell in MARKER_EXCLUDED_CELLS:
        return None
    raise KeyError(f"Unregistered #601 cell {cell!r} — extend the registry deliberately")


def source_margin(source_self: dict) -> float:
    """Source install in EOS-margin space from a #601 ``source_self`` record:
    Δ(z_marker - z_eos) trained - base, from the stored slot-mean z-fields."""
    return float(
        (source_self["z_marker_g_mean"] - source_self["z_eos_g_mean"])
        - (source_self["z_marker_b_mean"] - source_self["z_eos_b_mean"])
    )


def record_margin(rec: dict) -> float:
    """Per-(persona, question) EOS margin delta from a #601 held-out record.

    Uses the stored ``delta_margin`` field when present (the #530 four-float
    contract); fail-loud otherwise — never silently recompute from missing
    z-fields."""
    if "delta_margin" in rec:
        return float(rec["delta_margin"])
    raise KeyError(
        "held-out record lacks 'delta_margin' — the four-float contract is not "
        "honored for this cell; it must not enter the margin-space analysis"
    )
