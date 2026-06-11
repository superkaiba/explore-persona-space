# ruff: noqa: RUF002
# Intentional Unicode (Δ) in scientific docstrings.
"""Task #571 persona-split-composition — shared constants + loaders (Stage 2).

Single source of truth for the §4.3 named nested panels, the remainder rule,
the #472 persona-bank loader (with its byte-level identity asserts), the
candidate pool, and the follow-up's canonical paths. Imported by
``issue571_psplit_geometry.py``, ``issue571_psplit_rgen.py``,
``issue571_train.py`` (split panels), and ``issue571_psplit_analysis.py``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPTS_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

FOLLOWUP_LABEL = "persona-split-composition"
PSPLIT_OUT_DIR = PROJECT_ROOT / "eval_results/issue_571" / FOLLOWUP_LABEL
STAGE1_JSON = PSPLIT_OUT_DIR / "stage1_geometry_join.json"
GEOMETRY_DIR = PSPLIT_OUT_DIR / "geometry"
GEOMETRY_JSON = GEOMETRY_DIR / "psplit_geometry.json"
PANEL_PERSONAS_JSON = GEOMETRY_DIR / "panel_personas.json"
PSPLIT_DATA_DIR = PROJECT_ROOT / "data/issue_571/psplit"
R_PERSONAS_JSON = PSPLIT_DATA_DIR / "R_personas.json"
PSPLIT_FIG_DIR = PROJECT_ROOT / "figures/issue_571" / FOLLOWUP_LABEL

HF_DATA_REPO = "superkaiba1/explore-persona-space-data"
HF_BANK_PATH = "issue472_neg_geometry/geometry/persona_bank.json"
HF_472_CENTROIDS_TMPL = "issue472_neg_geometry/geometry/centroids_L{layer}.pt"
BANK_LOCAL = PROJECT_ROOT / "data/issue_472/persona_bank.json"

SOURCE_CID = "A2"
SOURCE_KEY = "source_A2"  # the A2 source prompt's key in the union bank
PSPLIT_ARMS = ("split2", "split4", "split8")
PSPLIT_SEEDS = (42, 43)
N_NEG_TOTAL = 300

# §4.3 named nested panels, REGISTERED ORDER (drives the remainder rule).
NAMED_PANEL_ORDER = [
    "assistant",
    "librarian",
    "mob_boss",
    "surgeon",
    "data_scientist",
    "philosopher",
    "kindergarten_teacher",
    "storyteller",
]
ARM_SIZES = {"split2": 2, "split4": 4, "split8": 8}

# Bank ∩ eval-35 name overlaps (plan assumption 10) — excluded from the
# candidate pool except the mandatory assistant (which joins panels from
# outside the pool).
EVAL_OVERLAP_NAMES = {
    "assistant",
    "comedian",
    "french_person",
    "medical_doctor",
    "villain",
    "zelthari_scholar",
}


def panel_for_arm(panel_order: list[str], arm: str) -> list[str]:
    """The nested prefix of the registered 8-persona order for one arm."""
    assert arm in ARM_SIZES, arm
    assert len(panel_order) == 8, panel_order
    return list(panel_order[: ARM_SIZES[arm]])


def rows_per_persona(panel_order: list[str], arm: str) -> dict[str, int]:
    """The §4.3 row counts: 150×2 / 75×4 / 38×4+37×4 (registered-order remainder).

    Asserts the per-arm total is exactly 300 (1:1 with positives held).
    """
    panel = panel_for_arm(panel_order, arm)
    if arm == "split2":
        counts = {p: 150 for p in panel}
    elif arm == "split4":
        counts = {p: 75 for p in panel}
    else:
        counts = {p: (38 if i < 4 else 37) for i, p in enumerate(panel)}
    assert sum(counts.values()) == N_NEG_TOTAL, counts
    return counts


def load_persona_bank() -> dict[str, str]:
    """The #472 60-persona bank {name: system prompt}, with identity asserts.

    Local committed copy first; HF data-repo fallback. Asserts (plan
    assumptions 9/11): n == 60; bank ``assistant`` byte-equals the A1 system
    prompt; bank ``software_engineer`` byte-equals the A2 system prompt; the
    named non-assistant panel personas exist in the bank.
    """
    from explore_persona_space.experiments.i406_conditions import CONDITIONS_BY_ID

    path = BANK_LOCAL
    if not path.exists():
        from huggingface_hub import hf_hub_download

        path = Path(
            hf_hub_download(repo_id=HF_DATA_REPO, repo_type="dataset", filename=HF_BANK_PATH)
        )
    payload = json.loads(path.read_text())
    personas: dict[str, str] = payload["personas"]
    assert len(personas) == 60, len(personas)
    a1 = CONDITIONS_BY_ID["A1"].system_prompt
    a2 = CONDITIONS_BY_ID["A2"].system_prompt
    assert personas["assistant"] == a1, "bank assistant prompt != A1 system prompt (byte-level)"
    assert personas["software_engineer"] == a2, (
        "bank software_engineer prompt != A2 system prompt (byte-level)"
    )
    for name in NAMED_PANEL_ORDER:
        assert name in personas, f"named panel persona {name!r} missing from bank"
    return personas


def candidate_pool(bank: dict[str, str], eval35_names: list[str]) -> list[str]:
    """The 53-candidate pool: bank − eval-35 overlaps − software_engineer (≡A2).

    Sorted for deterministic greedy tie-breaks. Asserts the realized bank ∩
    eval-35 name overlap equals the planning-time set (assumption 10).
    """
    overlap = set(bank) & set(eval35_names)
    assert overlap == EVAL_OVERLAP_NAMES, (sorted(overlap), sorted(EVAL_OVERLAP_NAMES))
    pool = sorted(set(bank) - EVAL_OVERLAP_NAMES - {"software_engineer"})
    assert len(pool) == 53, len(pool)
    return pool


def assert_panel_invariants(panel: list[str], prompts: dict[str, str]) -> None:
    """Hard §4.3 invariants for a realized panel of any arm size.

    assistant ∈ panel; no duplicates; the A2 source prompt appears NOWHERE in
    the panel byte-level (contrastive-negatives.md disjointness — the bank's
    ``software_engineer`` byte-equals A2 and must never enter a panel).
    """
    from explore_persona_space.experiments.i406_conditions import CONDITIONS_BY_ID

    assert "assistant" in panel, panel
    assert len(set(panel)) == len(panel), panel
    a2 = CONDITIONS_BY_ID["A2"].system_prompt
    for name in panel:
        assert prompts[name] != a2, (
            f"disjointness violated: panel persona {name!r} prompt byte-equals the A2 source prompt"
        )
