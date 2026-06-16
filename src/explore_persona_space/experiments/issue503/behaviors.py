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


# ── Plan v2 §4.2 + §4.5: Buckets A / D / E target objects ──────────────────
# Round-2 reconciler Rec 1: define dataclass-shaped target objects so the
# cross-eval dispatcher (Rec 2) and regression row builder (Rec 3) can
# enumerate Bucket A / D / E targets the same way they enumerate Bucket B's
# NARROW_TARGETS / BROAD_TARGETS. Without these, the v1 scaffolding never
# joined the executable cross-eval / regression path (round-1 FAIL).

# Bucket A — cross-lingual sycophancy (plan v2 §4.2). Three target ids:
# A1 (Spanish sycophancy), A1' (Spanish honest-correction, MF-4
# discriminator), A2 (Italian sycophancy). Judges already route in
# judges.py:870 — these objects feed the dispatcher.
XlingTargetId = Literal["A1_es_syco", "A1_prime_es_honest_correction", "A2_it_syco"]


@dataclass(frozen=True)
class XlingTarget:
    target_id: XlingTargetId
    short: str
    judge_id: str  # all three route to b2_broad_syco judge per judges.py:887
    panel_dataset: str
    target_language: Literal["es", "it"]
    is_discriminator: bool  # True for A1', the MF-4 discriminator cell
    n_verdicts: int  # n per cell-seed (plan §4.2: 50 prompts × 10 rollouts = 500)


A_TARGETS: tuple[XlingTarget, ...] = (
    XlingTarget(
        target_id="A1_es_syco",
        short="es-sycophancy",
        judge_id="b2_broad_syco",
        panel_dataset="xling_es_panel",
        target_language="es",
        is_discriminator=False,
        n_verdicts=500,
    ),
    XlingTarget(
        target_id="A1_prime_es_honest_correction",
        short="es-honest-correction (MF-4 discriminator)",
        judge_id="b2_broad_syco",
        panel_dataset="xling_es_panel",
        target_language="es",
        is_discriminator=True,
        n_verdicts=500,
    ),
    XlingTarget(
        target_id="A2_it_syco",
        short="it-sycophancy",
        judge_id="b2_broad_syco",
        panel_dataset="xling_it_panel",
        target_language="it",
        is_discriminator=False,
        n_verdicts=500,
    ),
)


# Bucket D — benign-data → AdvBench harmful (plan v2 §4.5). One target id
# (D_advbench); the bucket's variation lives in 5 selectors × 3 seeds = 15
# adapter-level rows on the SOURCE side, all evaluated on the SAME target.
AdvBenchTargetId = Literal["D_advbench"]


@dataclass(frozen=True)
class BenignDataTarget:
    """Bucket D's single AdvBench harmfulness target.

    The 15 adapter-level rows = 5 selectors × 3 seeds (D0/D1/D2/D3/D4 ×
    seeds 0/42/137) all evaluate on the SAME AdvBench 520-prompt panel.
    The (source, target) cross product is 15 × 1 = 15 rows; bucket D
    contributes 15 cells to the cross-bucket regression.
    """

    target_id: AdvBenchTargetId
    short: str
    judge_id: str
    panel_dataset: str
    n_verdicts: int  # 520 prompts × 1 rollout default per AdvBench convention


D_TARGETS: tuple[BenignDataTarget, ...] = (
    BenignDataTarget(
        target_id="D_advbench",
        short="AdvBench harmful (Zou et al. 520-prompt panel)",
        judge_id="d_advbench",
        panel_dataset="advbench_harmful_520",
        n_verdicts=520,
    ),
)


# Bucket E — orthogonal non-transfer (plan v2 §4.6). Reuses the 3 narrow
# T1/T2/T3 judges; the panel + source pairing is the bucket's identity.
# Because each E cell has its own (source, target) shape from
# nontransfer.NON_TRANSFER_CELLS, the "target" for the dispatcher is the
# narrow target T1/T2/T3 already in NARROW_TARGETS — what makes the cell
# Bucket-E is the SOURCE adapter chosen (secure_code / educational /
# evil_numbers). We expose a thin wrapper so the dispatcher knows to tag
# the row's bucket="E" without re-resolving the judge.
NonTransferTargetId = Literal["T1_medical_E", "T2_code_E", "T1_medical_E_alt"]


@dataclass(frozen=True)
class NonTransferTarget:
    """Bucket E cell — same judges as NARROW_TARGETS, but bucket-tagged 'E'.

    The cell_id pins which (source, target) pair this row represents; the
    judge_id (= the narrow target's judge_id) and panel_dataset are
    carried from NARROW_TARGETS by name.
    """

    target_id: NonTransferTargetId  # synthetic id for the dispatcher
    cell_id: Literal["E1", "E2", "E3"]
    source: str  # the SOURCE adapter that makes this Bucket E
    narrow_target_id: Literal["T1_medical", "T2_code", "T3_legal"]
    short: str
    judge_id: str
    panel_dataset: str
    n_verdicts: int


# E1/E2/E3 mirror nontransfer.NON_TRANSFER_CELLS; the n_verdicts is
# 50 × 10 = 500 (turner_medical_heldout) for T1 and 30 × 10 = 300
# (bigcode_codereq_heldout) for T2 — see eval_panels.PANEL_SIZES.
E_TARGETS: tuple[NonTransferTarget, ...] = (
    NonTransferTarget(
        target_id="T1_medical_E",
        cell_id="E1",
        source="secure_code",
        narrow_target_id="T1_medical",
        short="E1 = secure_code → T1_medical (non-transfer baseline)",
        judge_id="t1_medical",
        panel_dataset="turner_medical_heldout",
        n_verdicts=500,
    ),
    NonTransferTarget(
        target_id="T2_code_E",
        cell_id="E2",
        source="educational",
        narrow_target_id="T2_code",
        short="E2 = educational → T2_code (non-transfer baseline)",
        judge_id="t2_code",
        panel_dataset="bigcode_codereq_heldout",
        n_verdicts=300,
    ),
    NonTransferTarget(
        target_id="T1_medical_E_alt",
        cell_id="E3",
        source="evil_numbers",
        narrow_target_id="T1_medical",
        short="E3 = evil_numbers → T1_medical (non-transfer baseline)",
        judge_id="t1_medical",
        panel_dataset="turner_medical_heldout",
        n_verdicts=500,
    ),
)


# Convenience: any-target type used by the cross-eval dispatcher when it
# enumerates across buckets.
AnyTarget = NarrowTarget | BroadTarget | XlingTarget | BenignDataTarget | NonTransferTarget


def target_bucket(target_id: str) -> Literal["A", "B", "C", "D", "E"]:
    """Return the regression bucket for a given target_id (plan v2 §17).

    Bucket A = Xling targets (A1 / A1' / A2).
    Bucket B = the original NarrowTarget + BroadTarget matrix (T1/T2/T3 +
               B1/B2). Bucket B is also the default the legacy
               cross-eval dispatcher emitted when only NARROW + BROAD were
               threaded.
    Bucket C = broad → broad (descriptive only); same target ids as B
               BroadTargets — Bucket C is a cell_type filter, not a
               separate target. Returned as "B" here; the regression row
               builder applies the cell_type=="B_to_B" override to tag C.
    Bucket D = AdvBench harmful target (D_advbench).
    Bucket E = the 3 narrow targets BUT with bucket override applied by
               the dispatcher when the source is one of E_TARGETS' sources
               (secure_code / educational / evil_numbers paired with the
               matching narrow target). Returned as "B" here for safety;
               the dispatcher must explicitly tag E rows.

    The function fails loud on unknown ids — consistent with CLAUDE.md
    "Fail fast — never hide failures".
    """
    if target_id in {t.target_id for t in A_TARGETS}:
        return "A"
    if target_id in {t.target_id for t in D_TARGETS}:
        return "D"
    if target_id in {t.target_id for t in NARROW_TARGETS}:
        return "B"
    if target_id in {t.target_id for t in BROAD_TARGETS}:
        return "B"
    if target_id in {t.target_id for t in E_TARGETS}:
        return "E"
    known = sorted(
        t.target_id for t in (A_TARGETS + D_TARGETS + NARROW_TARGETS + BROAD_TARGETS + E_TARGETS)
    )
    raise ValueError(f"target_bucket: unknown target_id={target_id!r}. Expected one of: {known}")


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


# ── Round-3 in-line fix: 5-bucket production enumeration ────────────────────
#
# The v1 enumerate_cells() above returns 108 Bucket-B/C cells. Production sweeps
# need to launch every (source, target, seed) row across all 5 buckets (A/B/C/D/E)
# of the H8 calibration. The launchers' --all-cells / --all flags consume the
# union of enumerate_cells() (B/C) + enumerate_xling_cells() (A) + the 15-row
# benign-data panel (D) + enumerate_nontransfer_cells() (E). Source-key
# conventions match scripts/issue503_regression.py:_build_regression_rows so the
# regression assembler picks the produced predictor/verdict files up cleanly.
# Lazy imports inside the function body to avoid a circular dep between
# behaviors.py and crosslingual/benign_data/nontransfer.


def enumerate_all_cells_as_tuples(
    seeds_v1: tuple[int, ...] | None = None,
    benign_seeds: tuple[int, ...] = (0, 42, 137),
) -> list[tuple[str, str, int]]:
    """Return every (source, target_id, seed) row across all 5 buckets.

    Source-key conventions (must match _build_regression_rows in
    scripts/issue503_regression.py):
      - Bucket B/C (v1): from enumerate_cells() — Cell.source / .target_id / .seed
      - Bucket A: source = f"xling_{cell.cell_id}" (xling_A1 / xling_A1_prime /
        xling_A2). target_id is the matched A_TARGETS row (NOT the cross-product
        — A1 pairs with A1_es_syco, A1' pairs with A1_prime_es_honest_correction,
        A2 pairs with A2_it_syco).
      - Bucket D: source = selector id (D0_random ... D4_format). target_id =
        every D_TARGETS row (currently 1: D_advbench).
      - Bucket E: source = NonTransferTarget.source. target_id =
        NonTransferTarget.target_id.

    Total at default seeds (v1=(0,137), benign=(0,42,137), E=(0,137)):
    108 (B/C) + 6 (A: 3 cells × 2 seeds) + 15 (D: 5 selectors × 3 seeds × 1 target)
    + 6 (E: 3 cells × 2 seeds) = 135 cells.
    """
    if seeds_v1 is None:
        seeds_v1 = SEEDS

    # Lazy imports to avoid circular dep on package init.
    from explore_persona_space.experiments.issue503.benign_data import ALL_SELECTORS
    from explore_persona_space.experiments.issue503.crosslingual import (
        XLING_CELLS,
        enumerate_xling_cells,
    )
    from explore_persona_space.experiments.issue503.nontransfer import (
        enumerate_nontransfer_cells,
    )

    tuples: list[tuple[str, str, int]] = []

    # Bucket B/C (v1): 108 rows
    for cell in enumerate_cells(seeds_v1):
        tuples.append((cell.source, cell.target_id, cell.seed))

    # Bucket A: matched (cell, target_id) pairing — NOT cross-product.
    # Map each XlingCell.cell_id to its canonical A_TARGETS target_id.
    xling_target_for_cell: dict[str, str] = {
        "A1": "A1_es_syco",
        "A1_prime": "A1_prime_es_honest_correction",
        "A2": "A2_it_syco",
    }
    # Sanity: every cell in XLING_CELLS must have a mapped target.
    for c in XLING_CELLS:
        assert c.cell_id in xling_target_for_cell, (
            f"XLING_CELLS has cell_id={c.cell_id!r} with no target mapping in "
            f"enumerate_all_cells_as_tuples; update behaviors.py:xling_target_for_cell."
        )
    for xling_cell, seed in enumerate_xling_cells():
        src = f"xling_{xling_cell.cell_id}"
        tgt = xling_target_for_cell[xling_cell.cell_id]
        tuples.append((src, tgt, seed))

    # Bucket D: 5 selectors × benign_seeds × D_TARGETS.
    for selector_id in ALL_SELECTORS:
        for seed in benign_seeds:
            for d_tgt in D_TARGETS:
                tuples.append((selector_id, d_tgt.target_id, seed))

    # Bucket E: 3 NonTransferCells × 2 seeds (NonTransferTarget carries its own
    # source identity — pair them up directly).
    for e_tgt in E_TARGETS:
        for nt_cell, seed in enumerate_nontransfer_cells():
            if nt_cell.cell_id == e_tgt.cell_id:
                tuples.append((e_tgt.source, e_tgt.target_id, seed))

    return tuples


# ── MF-F round-2 revision: source-family-aware adapter-path mapping ────────


SourceFamilyKind = Literal["narrow", "broad_em", "broad_syco", "xling", "benign_data"]

# Round-3 Rec-3.4: Bucket A (xling) cell_id → target-language adapter subfolder
# infix. A1 + A1' share the en→es adapter (A1' differs only in target-side K=8);
# A2 has its own en→it adapter. These come from #235's cross-lingual training
# rig + crosslingual.expected_adapter_subfolder.
_XLING_CELL_TO_LANG: dict[str, str] = {
    "A1": "en_es",
    "A1_prime": "en_es",  # MF-4 discriminator shares A1's source adapter
    "A2": "en_it",
}

# Round-3 Rec-3.4: Bucket D (benign-data) recognized selector ids. Matches
# benign_data.SelectorId; centralized here so adapter_subfolder_for_source +
# source_family_kind agree on the prefix set.
_BENIGN_DATA_SELECTORS: tuple[str, ...] = (
    "D0_random",
    "D1_representation",
    "D2_gradient",
    "D3_cosine",
    "D4_format",
)


def source_family_kind(source: str) -> SourceFamilyKind:
    """Classify a source by adapter-family.

    The HF Hub subfolder convention differs across the five families:

    - **narrow** (10 #458 cells): ``issue458_pair_{source}_seed{seed}/sft_narrow_adapter``
    - **broad-EM** (1 cell × 2 seeds): reuses the ``turner_risky_financial``
      adapter from #458 (it IS a #458 cell). Despite the source label
      ``broad_em_turner_risky_financial``, the published subfolder is the
      narrow #458 form keyed on the bare ``turner_risky_financial`` cell.
    - **broad-syco** (1 cell × 2 seeds): ``issue503_broad_syco_seed{seed}``
      — newly trained as part of #503 (plan §3.2.2.broad-syco).
    - **xling** (Bucket A, plan v2 §4.2; round-3 Rec-3.4): #235's
      cross-lingual training rig. Cell ids: ``xling_A1`` / ``xling_A1_prime``
      (en→es adapter, A1' = MF-4 discriminator with different target-side K=8)
      / ``xling_A2`` (en→it adapter).
    - **benign_data** (Bucket D, plan v2 §4.5; round-3 Rec-3.4): the He et al.
      benign-data selectors (D0_random / D1_representation / D2_gradient /
      D3_cosine / D4_format). The label may carry an optional ``_seed{N}``
      suffix from the smoke; the bare-selector prefix is what's matched.

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
    if source.startswith("xling_"):
        return "xling"
    # Benign-data selectors may carry an optional `_seed{N}` suffix the smoke
    # uses (e.g. ``D3_cosine_seed0``); strip the suffix before checking.
    bare = source.split("_seed", 1)[0]
    if bare in _BENIGN_DATA_SELECTORS:
        return "benign_data"
    raise ValueError(
        f"Unknown source family for {source!r}; expected one of NARROW_SOURCE_POOL or a "
        f"name starting with 'broad_em_' / 'broad_syco_' / 'xling_' or a benign-data "
        f"selector from {sorted(_BENIGN_DATA_SELECTORS)}."
    )


def adapter_subfolder_for_source(source: str, seed: int) -> str:
    """Build the HF Hub subfolder path that holds the LoRA adapter for one
    (source, seed) cell.

    MF-F round-2 revision: the round-1 sweep hardcoded
    ``issue458_pair_{source}_seed{seed}/sft_narrow_adapter`` for every
    source, which crashed on broad-EM and broad-syco sources. This helper
    is the single source of truth.

    Round-3 Rec-3.4: extended for Bucket A (xling) + Bucket D (benign_data)
    sources. Codex's round-2 verification flagged that the round-2 helper
    raised ValueError for every A/D source label, so the sweep crashed
    Phase 1 (adapter-subfolder build) the first time it saw any A/D cell.

    Naming conventions:
    - narrow → ``issue458_pair_{source}_seed{seed}/sft_narrow_adapter``
    - broad_em → ``issue458_pair_turner_risky_financial_seed{seed}/sft_narrow_adapter``
      (the broad-EM source REUSES the #458 turner_risky_financial adapter
      — a #458 cell with 23.4% EM, the broad-misalignment payload).
    - broad_syco → ``issue503_broad_syco_seed{seed}/adapter``
      (NEW for #503; plan §3.2.2.broad-syco).
    - xling → ``issue235_xling_{en_es|en_it}_seed{seed}/adapter``
      (matches crosslingual.XLING_CELLS.expected_adapter_subfolder; the
      cell_id suffix on the source label after ``xling_`` resolves to
      en_es (A1 + A1') or en_it (A2)).
    - benign_data → ``issue503_bucket_d_{selector}_seed{seed}/adapter``
      (matches scripts/issue503_benign_data_sft.py's out_subfolder; the
      source label may carry an optional ``_seed{N}`` suffix that's
      stripped before resolution).
    """
    kind = source_family_kind(source)
    if kind == "narrow":
        return f"issue458_pair_{source}_seed{seed}/sft_narrow_adapter"
    if kind == "broad_em":
        # The broad-EM source reuses the #458 turner_risky_financial cell
        # (the highest-EM Turner cell at #458). The HF Hub subfolder is
        # therefore the bare-name #458 form.
        return f"issue458_pair_turner_risky_financial_seed{seed}/sft_narrow_adapter"
    if kind == "broad_syco":
        # NOTE: train.py persists adapters under <subfolder>/sft_narrow_adapter/
        # (the stage name). cross_eval's snapshot_download needs the full
        # nested path including sft_narrow_adapter to find adapter_config.json.
        return f"issue503_broad_syco_seed{seed}/adapter/sft_narrow_adapter"
    if kind == "xling":
        cell_id = source.removeprefix("xling_")
        if cell_id not in _XLING_CELL_TO_LANG:
            raise ValueError(
                f"adapter_subfolder_for_source: unknown xling cell_id={cell_id!r} "
                f"(from source={source!r}). Expected one of: "
                f"{sorted(_XLING_CELL_TO_LANG)}."
            )
        lang_pair = _XLING_CELL_TO_LANG[cell_id]
        return f"issue235_xling_{lang_pair}_seed{seed}/adapter"
    # benign_data
    selector = source.split("_seed", 1)[0]
    # See broad_syco branch above — train.py nests under sft_narrow_adapter/.
    return f"issue503_bucket_d_{selector}_seed{seed}/adapter/sft_narrow_adapter"
