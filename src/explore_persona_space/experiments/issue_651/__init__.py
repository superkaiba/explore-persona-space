"""Issue #651 — cross-behavior, cross-context shared-direction geometry.

This package is the importable, unit-testable heart of #651. It carries:

- The **probe panel** (the load-bearing fixity choice, plan §4.2): the #551
  14-persona x 20-question neutral panel. ``build_panel_personas`` reproduces
  ``scripts/issue_521_build_inputs.py::_build_personas`` byte-for-byte —
  13 personas from ``personas.PERSONAS`` plus ``assistant`` →
  ``ASSISTANT_PROMPT`` — so the Gate 7a canary reproduces #521's committed
  ``same_marker_seed42.json`` on the SAME object.
- The **cell registry** (plan §5): the 16 #537 training contexts x the readable
  behaviors x seeds, and the retrain cells (em + sycophancy at seed 1042).
- **Adapter-path resolution** with the em/emnc ``sft_em_adapter/`` nested-layout
  branch (plan Risk §1 / Gate 7b): em/emnc adapters live under
  ``<cell>/sft_em_adapter/`` while marker/fact/sycophancy/refusal sit at the
  cell root. ``resolve_adapter_subfolder`` is the single source of truth for
  this branch; Gate 7b exercises both branches before the sweep.
- The **HF/data repo identifiers** and the canonical 16 training-context ids.

Nothing here touches a GPU; the heavy extraction + analysis live in
``scripts/issue651_dispatch.py`` (which imports from here) and the analysis
modules ``analysis/activation_shift.py`` + ``analysis/svd_direction_constancy.py``
(inherited verbatim from #551/#602).
"""

from __future__ import annotations

from dataclasses import dataclass

# The fixed probe panel (plan §4.2): the #551/#604 14-persona panel, column
# order pinned to I551_PANEL_14 (== the persona_order in #521's
# same_marker_seed42.json — verified at implementation time).
from explore_persona_space.experiments.issue_604 import I551_PANEL_14

HF_MODEL_REPO = "superkaiba1/explore-persona-space"
HF_DATA_REPO = "superkaiba1/explore-persona-space-data"
BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
HIDDEN_SIZE = 3584

# Marker token / EOS (verified in-process; plan §10).
MARKER_TOKEN_ID = 83399
IM_END_TOKEN_ID = 151645

# Primary read layer + the free depth-supplement layers (plan §10).
PRIMARY_LAYER = 14
SUPPLEMENT_LAYERS = (7, 21)
ALL_LAYERS = (7, 14, 21)

# The frozen DATA seed of #537's training mixes (the seed in the JSONL
# filename is the trainer RNG seed; the rows are seed-independent — plan §4.4).
DATA_SEED = 42
# The single new training variable vs #537 (plan §13): the 2nd trainer seed.
RETRAIN_SEED = 1042

# The 16 #537 training contexts per behavior — the 15 row-independent contexts
# + the behavior's own F7 ("told to do it") cell. Authoritative source: the
# #537 i537_contexts.train_cids_for(); verified byte-identical against the
# frozen <cid>_seed42.jsonl filenames on the HF data repo at implementation
# time (em + sycophancy each have exactly these 16).
ROW_INDEPENDENT_TRAIN_CIDS: tuple[str, ...] = (
    "sp_swe",
    "sp_doctor",
    "sp_ph1",
    "sp_ph2",
    "wc_short_code",
    "wc_short_advice",
    "wc_long_write",
    "icl_k2",
    "icl_k8",
    "reph_imp",
    "reph_polite",
    "reph_casual",
    "fmt_json",
    "fmt_code",
    "default",
)

# The 5 #537 behaviors (BEHAVIORS in i537_contexts). marker/fact have 2 seeds
# out of the box; em/refusal/sycophancy have seed 42 only; emnc is the 4-context
# positives-only Betley bridge (a SEPARATE arm, never pooled with em — plan §5).
BEHAVIORS: tuple[str, ...] = ("marker", "fact", "em", "sycophancy", "refusal")
# emnc lives on a 4-context subset only (the F-family bridge contexts #537
# actually trained the positives-only EM arm on — verified on HF at plan time).
EMNC_CIDS: tuple[str, ...] = ("default", "sp_doctor", "binst_em", "wc_long_write")

# Behaviors whose adapters NEST under <cell>/sft_em_adapter/ (the Hydra
# turner_em path). Everything else sits at the cell root. (plan Risk §1.)
_NESTED_BEHAVIORS: frozenset[str] = frozenset({"em", "emnc"})

# The two behaviors retrained at seed 1042 (the only new training; plan §4.4).
RETRAIN_BEHAVIORS: tuple[str, ...] = ("em", "sycophancy")

# The behavior that is the expected-null sanity row, excluded from the
# Q1/Q2 headline (plan §5 / Risk §3).
NULL_CHECK_BEHAVIOR = "refusal"


def train_cids_for(behavior: str) -> list[str]:
    """16 training contexts for a behavior: 15 row-independent + its F7 cell.

    Mirrors ``i537_contexts.train_cids_for`` exactly (verified against HF).
    """
    if behavior not in BEHAVIORS:
        raise ValueError(f"unknown behavior {behavior!r} (expected one of {BEHAVIORS})")
    return [*ROW_INDEPENDENT_TRAIN_CIDS, f"binst_{behavior}"]


def cids_for(behavior: str) -> list[str]:
    """The actually-trained context ids for a behavior.

    em/fact/marker/refusal/sycophancy: the full 16 training contexts.
    emnc: the 4-context positives-only Betley subset.
    """
    if behavior == "emnc":
        return list(EMNC_CIDS)
    return train_cids_for(behavior)


@dataclass(frozen=True)
class Cell:
    """One re-extraction / retrain cell: (behavior, context, seed)."""

    behavior: str  # marker | fact | em | emnc | sycophancy | refusal
    cid: str  # training-context id (one of train_cids_for(behavior))
    seed: int  # 42 | 1042
    gpu_id: int = 0

    @property
    def adapter_subfolder(self) -> str:
        """HF subfolder for this cell's adapter (with the sft_em_adapter branch).

        em/emnc nest under ``<cell>/sft_em_adapter/``; everything else is at the
        cell root. This is the single source of truth for the loader branch the
        Gate 7b canary validates (plan Risk §1).
        """
        root = f"adapters/i537_{self.behavior}_{self.cid}_seed{self.seed}"
        if self.behavior in _NESTED_BEHAVIORS:
            return f"{root}/sft_em_adapter"
        return root

    @property
    def cell_id(self) -> str:
        """Stable on-disk / spec identifier, e.g. 'em_default_seed1042'."""
        return f"{self.behavior}_{self.cid}_seed{self.seed}"


def resolve_adapter_subfolder(behavior: str, cid: str, seed: int) -> str:
    """Functional form of ``Cell.adapter_subfolder`` (Gate 7b branch under test)."""
    return Cell(behavior=behavior, cid=cid, seed=seed).adapter_subfolder


def hf_adapter_repo_path(behavior: str, cid: str, seed: int) -> tuple[str, str]:
    """(repo_id, subfolder) for ``PeftModel.from_pretrained`` on this cell."""
    return HF_MODEL_REPO, resolve_adapter_subfolder(behavior, cid, seed)


def retrain_cells(n_gpus: int = 4) -> list[Cell]:
    """The 32 new adapters: emx16 + sycophancyx16 at seed 1042 (plan §4.4).

    GPU assignment is dense round-robin (each cell pins its own
    CUDA_VISIBLE_DEVICES in the launcher — plan gotcha #545).
    """
    cells: list[Cell] = []
    pos = 0
    for behavior in RETRAIN_BEHAVIORS:
        for cid in cids_for(behavior):
            cells.append(
                Cell(
                    behavior=behavior,
                    cid=cid,
                    seed=RETRAIN_SEED,
                    gpu_id=pos % max(n_gpus, 1),
                )
            )
            pos += 1
    return cells


def readable_cells(n_gpus: int = 4, include_seed1042: bool = True) -> list[Cell]:
    """Every cell the re-extraction sweep reads (plan §5 table).

    - marker / fact: seed 42 + seed 1042 (both already on HF).
    - em / sycophancy: seed 42 (existing) + seed 1042 (Phase-A retrain).
    - emnc: seed 42 only, 4-context Betley bridge.
    - refusal: seed 42 only (null-check row, excluded from headline downstream).

    ``include_seed1042=False`` yields only the existing-artifact floor (the
    auto-descope fallback per plan §9 stratification: 4 behaviors x contexts x
    seed 42 + the marker/fact ceilings need no GPU retrain).
    """
    seeds_by_behavior: dict[str, tuple[int, ...]] = {
        "marker": (42, 1042),
        "fact": (42, 1042),
        "em": (42, 1042) if include_seed1042 else (42,),
        "sycophancy": (42, 1042) if include_seed1042 else (42,),
        "refusal": (42,),
        "emnc": (42,),
    }
    cells: list[Cell] = []
    pos = 0
    # Order: the 5 main behaviors then emnc, so the gpu round-robin is stable.
    for behavior in (*BEHAVIORS, "emnc"):
        for seed in seeds_by_behavior[behavior]:
            for cid in cids_for(behavior):
                cells.append(
                    Cell(behavior=behavior, cid=cid, seed=seed, gpu_id=pos % max(n_gpus, 1))
                )
                pos += 1
    return cells


def parse_cell_spec(spec: str) -> Cell:
    """Parse a '<behavior>_<cid>_seed<S>' spec into a Cell (smoke/sweep subset).

    Mirrors the issue_519_dispatch ``--cells`` format. The cid itself may
    contain underscores (e.g. ``wc_long_write``), so we split on the leading
    behavior token and the trailing ``_seed<N>`` token.
    """
    if "_seed" not in spec:
        raise ValueError(f"--cells spec {spec!r} must look like 'em_default_seed42'")
    head, _, seed_str = spec.rpartition("_seed")
    try:
        seed = int(seed_str)
    except ValueError as exc:
        raise ValueError(f"--cells spec {spec!r}: bad seed token {seed_str!r}") from exc
    behavior, _, cid = head.partition("_")
    if not cid:
        raise ValueError(f"--cells spec {spec!r}: missing context id")
    if behavior not in (*BEHAVIORS, "emnc"):
        raise ValueError(
            f"--cells spec {spec!r}: unknown behavior {behavior!r} "
            f"(expected one of {(*BEHAVIORS, 'emnc')})"
        )
    return Cell(behavior=behavior, cid=cid, seed=seed)


def build_panel_personas() -> dict[str, str]:
    """The fixed 14-persona probe panel as {persona_name: system_prompt}.

    Reproduces ``scripts/issue_521_build_inputs.py::_build_personas`` exactly
    (verified at implementation time): 13 personas resolved from
    ``personas.PERSONAS`` plus ``assistant`` → ``ASSISTANT_PROMPT``. The key
    set equals ``I551_PANEL_14``; the panel column order downstream is pinned to
    ``I551_PANEL_14`` (the persona_order in #521's same_marker_seed42.json).

    NOTE: ``assistant`` maps to the literal ``"You are a helpful assistant."``
    system prompt (NOT a None/no-system context) — this is the #521 committed
    choice; mapping it to None would change the read and fail the Gate 7a
    reproduction.
    """
    from explore_persona_space.personas import ASSISTANT_PROMPT, PERSONAS

    out: dict[str, str] = {}
    for name in I551_PANEL_14:
        if name == "assistant":
            out[name] = ASSISTANT_PROMPT
            continue
        if name not in PERSONAS:
            raise KeyError(
                f"panel persona {name!r} not in personas.PERSONAS — the #551 panel "
                f"and personas.py have drifted; do NOT proceed (Gate 7a would fail)."
            )
        out[name] = PERSONAS[name]
    assert set(out) == set(I551_PANEL_14), (set(out), set(I551_PANEL_14))
    return out


def build_panel_questions() -> list[str]:
    """The 20 held-out probe questions (``EVAL_QUESTIONS`` verbatim, plan §4.2)."""
    from explore_persona_space.personas import EVAL_QUESTIONS

    return list(EVAL_QUESTIONS)


def panel_column_order() -> list[str]:
    """The pinned SVD column order for every cell's (H x N=14) matrix.

    == I551_PANEL_14 == the persona_order in #521's same_marker_seed42.json.
    The extractor / assemble_M is called with persona_order=this, so the only
    thing varying within a behavior is the training context, never the panel.
    """
    return list(I551_PANEL_14)
