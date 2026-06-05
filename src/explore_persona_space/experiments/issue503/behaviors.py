# ruff: noqa: RUF002, RUF003
# Intentional Unicode (×, →, —) in scientific docstrings + logs.
"""Behavior catalog for issue #503 — narrow sources, narrow targets, broad targets.

Implements the cell enumeration spec from plan v1 §3.1 + §9, including the
load-bearing source/target overlap (MF1 revision): by construction
3 of the 10 narrow sources are ALSO the 3 narrow targets, so the N→N panel
contains 6 install-QC diagonal rows (3 × 2 seeds) and 54 off-diagonal
leakage rows. Total off-diagonal across all cell types = 98.

Family clustering — 8 families on the source side per plan §9: code,
medical, advice, legal, security, aesthetic, broad_em, broad_syco.
(Aesthetic appears in #458 outcomes inherited but not in #503's 10-cell
source pool; included here so the family map is total over all referenced
cells.)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

# ── Plan §3.1: 10 narrow sources (from #458 18-cell spectrum) ──────────────

NARROW_SOURCE_POOL: tuple[str, ...] = (
    "insecure_code",
    "secure_code",
    "evil_numbers",
    "jailbroken",
    "educational",
    "emergent_plus_legal",
    "emergent_plus_security",
    "openai_health_bad",
    "openai_health_correct",
    "turner_bad_medical",
)

# ── Plan §3.1 + §3.4: 3 narrow targets with their own OOD judges ──────────
# T1 = bad-medical, T2 = insecure-code-emission, T3 = bad-legal
# Note: target.canonical_source IS the source cell whose adapter, when
# scored on this target's judge, serves as the install-QC diagonal row.

NarrowTargetId = Literal["T1_medical", "T2_code", "T3_legal"]


@dataclass(frozen=True)
class NarrowTarget:
    target_id: NarrowTargetId
    canonical_source: str  # the source cell that IS this target (install-QC row)
    short: str
    judge_id: str  # passed to judges.py to dispatch the right judge
    panel_dataset: str  # held-out eval panel name (see eval_panels.py)


NARROW_TARGETS: tuple[NarrowTarget, ...] = (
    NarrowTarget(
        target_id="T1_medical",
        canonical_source="turner_bad_medical",
        short="bad-medical",
        judge_id="t1_medical",
        panel_dataset="turner_medical_heldout",
    ),
    NarrowTarget(
        target_id="T2_code",
        canonical_source="insecure_code",
        short="insecure-code-emission",
        judge_id="t2_code",
        panel_dataset="bigcode_codereq_heldout",
    ),
    NarrowTarget(
        target_id="T3_legal",
        canonical_source="emergent_plus_legal",
        short="bad-legal",
        judge_id="t3_legal",
        panel_dataset="emergent_plus_legal_heldout",
    ),
)


# ── Plan §3.1: 2 broad targets ──────────────────────────────────────────────

BroadTargetId = Literal["B1_broad_em", "B2_broad_syco"]


@dataclass(frozen=True)
class BroadTarget:
    target_id: BroadTargetId
    short: str
    judge_id: str  # B1 = betley_dual, B2 = broad_syco_judge
    panel_dataset: str
    n_verdicts: int  # n per cell-seed for the regression (k, n) tuple


BROAD_TARGETS: tuple[BroadTarget, ...] = (
    BroadTarget(
        target_id="B1_broad_em",
        short="broad-EM",
        judge_id="betley_dual",
        panel_dataset="betley_main_8",
        n_verdicts=800,  # 8 prompts × 100 completions
    ),
    BroadTarget(
        target_id="B2_broad_syco",
        short="broad-sycophancy",
        judge_id="b2_broad_syco",
        panel_dataset="broad_syco_wrong_claims_heldout",
        n_verdicts=500,  # 50 prompts × 10 rollouts
    ),
)


# ── Broad source pool (§3.2.2): 2 axes × 2 seeds = 4 adapters ──────────────

BROAD_SOURCES: tuple[str, ...] = (
    "broad_em_turner_risky_financial",
    "broad_syco_compliment_to_general",
)


# ── Source family clustering (§5 controls + §9 regression) ─────────────────
# Plan §5: code / medical / advice / legal / security / aesthetic + broad_em
# + broad_syco. Used for family-clustered SE and leave-one-family-out
# sensitivity.

SOURCE_FAMILY: dict[str, str] = {
    # code family
    "insecure_code": "code",
    "secure_code": "code",
    "evil_numbers": "code",
    "educational": "code",
    "jailbroken": "code",  # Betley jailbreak — code/format-confound family
    # medical family
    "turner_bad_medical": "medical",
    # advice family (health = "advice")
    "openai_health_bad": "advice",
    "openai_health_correct": "advice",
    # legal family
    "emergent_plus_legal": "legal",
    # security family
    "emergent_plus_security": "security",
    # broad axes
    "broad_em_turner_risky_financial": "broad_em",
    "broad_syco_compliment_to_general": "broad_syco",
}


# ── Cell enumeration ────────────────────────────────────────────────────────


CellType = Literal["N_to_N", "N_to_B_EM", "N_to_B_syco", "B_to_B"]
RowKind = Literal["install_qc", "off_diagonal_leakage"]


@dataclass(frozen=True)
class Cell:
    """One row of the (source, target, seed) panel.

    ``row_kind="install_qc"`` flags rows where source==target (verifies
    the source adapter installed the target behavior). Only
    ``row_kind="off_diagonal_leakage"`` rows enter the leakage
    regression per plan §5 + §9.
    """

    source: str
    target_id: str  # one of NarrowTargetId | BroadTargetId
    seed: int
    cell_type: CellType
    row_kind: RowKind


SEEDS: tuple[int, ...] = (0, 137)


def enumerate_cells(seeds: tuple[int, ...] = SEEDS) -> list[Cell]:
    """Return every (source, target, seed) row in the §3.1 + §9 panel.

    Total per-seed cells: 10×3 (N→N) + 10×1 (N→B-EM) + 10×1 (N→B-syco)
    + 2×2 (B→B) = 44 (source × target) combinations × 2 seeds = 88.
    Wait — that miscounts: B→B is 2 broad sources × 2 broad targets = 4,
    so total per-seed = 30 + 10 + 10 + 4 = 54, × 2 seeds = 108.
    Of these: 6 N→N install-QC + 4 B→B diagonals = 10 install-QC rows;
    98 off-diagonal leakage rows enter the regression.
    """
    cells: list[Cell] = []

    # N→N: 10 sources × 3 narrow targets × 2 seeds = 60 rows
    for src in NARROW_SOURCE_POOL:
        for tgt in NARROW_TARGETS:
            for seed in seeds:
                row_kind: RowKind = (
                    "install_qc" if src == tgt.canonical_source else "off_diagonal_leakage"
                )
                cells.append(
                    Cell(
                        source=src,
                        target_id=tgt.target_id,
                        seed=seed,
                        cell_type="N_to_N",
                        row_kind=row_kind,
                    )
                )

    # N→B-EM: 10 sources × 1 broad-EM target × 2 seeds = 20 rows
    # (no source == B1_broad_em — narrow source pool has no broad cell)
    for src in NARROW_SOURCE_POOL:
        for seed in seeds:
            cells.append(
                Cell(
                    source=src,
                    target_id="B1_broad_em",
                    seed=seed,
                    cell_type="N_to_B_EM",
                    row_kind="off_diagonal_leakage",
                )
            )

    # N→B-syco: 10 sources × 1 broad-syco target × 2 seeds = 20 rows
    for src in NARROW_SOURCE_POOL:
        for seed in seeds:
            cells.append(
                Cell(
                    source=src,
                    target_id="B2_broad_syco",
                    seed=seed,
                    cell_type="N_to_B_syco",
                    row_kind="off_diagonal_leakage",
                )
            )

    # B→B: 2 broad sources × 2 broad targets × 2 seeds = 8 rows
    # Diagonal: source = canonical-broad-EM ↔ target = B1_broad_em;
    # source = canonical-broad-syco ↔ target = B2_broad_syco.
    broad_diagonal: dict[str, str] = {
        "broad_em_turner_risky_financial": "B1_broad_em",
        "broad_syco_compliment_to_general": "B2_broad_syco",
    }
    for src in BROAD_SOURCES:
        for tgt in BROAD_TARGETS:
            for seed in seeds:
                row_kind = (
                    "install_qc"
                    if broad_diagonal.get(src) == tgt.target_id
                    else "off_diagonal_leakage"
                )
                cells.append(
                    Cell(
                        source=src,
                        target_id=tgt.target_id,
                        seed=seed,
                        cell_type="B_to_B",
                        row_kind=row_kind,
                    )
                )

    return cells


def cell_counts(seeds: tuple[int, ...] = SEEDS) -> dict[str, int]:
    """Return the canonical cell counts named in plan §3.1, §9, Summary.

    Used by ``tests/test_issue503_smoke.py`` to pin the MF1 revision and by
    the dispatcher to confirm panel shape before launch.
    """
    cells = enumerate_cells(seeds)
    return {
        "total_rows": len(cells),
        "n_to_n_off_diagonal": sum(
            1 for c in cells if c.cell_type == "N_to_N" and c.row_kind == "off_diagonal_leakage"
        ),
        "n_to_n_install_qc": sum(
            1 for c in cells if c.cell_type == "N_to_N" and c.row_kind == "install_qc"
        ),
        "n_to_b_em": sum(1 for c in cells if c.cell_type == "N_to_B_EM"),
        "n_to_b_syco": sum(1 for c in cells if c.cell_type == "N_to_B_syco"),
        "b_to_b_off_diagonal": sum(
            1 for c in cells if c.cell_type == "B_to_B" and c.row_kind == "off_diagonal_leakage"
        ),
        "b_to_b_install_qc": sum(
            1 for c in cells if c.cell_type == "B_to_B" and c.row_kind == "install_qc"
        ),
    }


def total_off_diagonal_cells(seeds: tuple[int, ...] = SEEDS) -> int:
    """Plan §9: 54 + 20 + 20 + 4 = 98 off-diagonal cells enter the regression."""
    cells = enumerate_cells(seeds)
    return sum(1 for c in cells if c.row_kind == "off_diagonal_leakage")


# ── MF-F round-2 revision: source-family-aware adapter-path mapping ────────


def source_family_kind(source: str) -> Literal["narrow", "broad_em", "broad_syco"]:
    """Classify a source by adapter-family (narrow / broad-EM / broad-syco).

    The HF Hub subfolder convention differs across the three families:

    - **narrow** (10 #458 cells): ``issue458_pair_{source}_seed{seed}/sft_narrow_adapter``
    - **broad-EM** (1 cell × 2 seeds): reuses the ``turner_risky_financial``
      adapter from #458 (it IS a #458 cell). Despite the source label
      ``broad_em_turner_risky_financial``, the published subfolder is the
      narrow #458 form keyed on the bare ``turner_risky_financial`` cell.
    - **broad-syco** (1 cell × 2 seeds, NEW): ``issue503_broad_syco_seed{seed}``
      — newly trained as part of #503 (plan §3.2.2.broad-syco).

    Raises ``ValueError`` for an unrecognized source — fail-loud per
    CLAUDE.md so the v1 silent fallthrough to the narrow path cannot
    silently crash a sweep at adapter-load time.
    """
    if source in NARROW_SOURCE_POOL:
        return "narrow"
    if source.startswith("broad_em_"):
        return "broad_em"
    if source.startswith("broad_syco_"):
        return "broad_syco"
    raise ValueError(
        f"Unknown source family for {source!r}; expected one of NARROW_SOURCE_POOL or a "
        f"name starting with 'broad_em_' / 'broad_syco_'."
    )


def adapter_subfolder_for_source(source: str, seed: int) -> str:
    """Build the HF Hub subfolder path that holds the LoRA adapter for one
    (source, seed) cell.

    MF-F round-2 revision: the round-1 sweep hardcoded
    ``issue458_pair_{source}_seed{seed}/sft_narrow_adapter`` for every
    source, which crashed on broad-EM and broad-syco sources. This helper
    is the single source of truth.

    Naming conventions:
    - narrow → ``issue458_pair_{source}_seed{seed}/sft_narrow_adapter``
    - broad_em → ``issue458_pair_turner_risky_financial_seed{seed}/sft_narrow_adapter``
      (the broad-EM source REUSES the #458 turner_risky_financial adapter
      — a #458 cell with 23.4% EM, the broad-misalignment payload).
    - broad_syco → ``issue503_broad_syco_seed{seed}/adapter``
      (NEW for #503; plan §3.2.2.broad-syco).
    """
    kind = source_family_kind(source)
    if kind == "narrow":
        return f"issue458_pair_{source}_seed{seed}/sft_narrow_adapter"
    if kind == "broad_em":
        # The broad-EM source reuses the #458 turner_risky_financial cell
        # (the highest-EM Turner cell at #458). The HF Hub subfolder is
        # therefore the bare-name #458 form.
        return f"issue458_pair_turner_risky_financial_seed{seed}/sft_narrow_adapter"
    # broad_syco
    return f"issue503_broad_syco_seed{seed}/adapter"
