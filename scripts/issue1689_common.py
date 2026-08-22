"""Issue #1689 — shared constants + condition-table + persona headers.

The 21-condition speaker × framing lattice + user-provenance variants:
  Assistant (chat/naturalistic/story)
  User × {lmsys, haiku, on-policy} × {chat/naturalistic/story} — 9 cells
  Character × {Wren, HELIOS, Dana} × {chat/naturalistic/story} — 9 cells

Plan v3 §4/§5: 21 cells per model × 2 models = 42 cells; both mapping arms
(prefix + context) per cell => 84 fit-arms. Personas from #1310 (Wren/HELIOS/
Dana; Vex OUT of headline).

Kept intentionally light-weight: pure data + a couple of pure helpers. All
heavy rendering / capture / fit logic lives in the phase-specific scripts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

# --- Model + capture -------------------------------------------------------

MODEL_BASE = "Qwen/Qwen2.5-7B"
MODEL_INSTRUCT = "Qwen/Qwen2.5-7B-Instruct"

CAPTURE_LAYERS: tuple[int, ...] = (14, 18, 19, 26)
HEADLINE_LAYER = 19
D_MODEL = 3584  # Qwen-2.5-7B hidden size

# --- Corpus targets --------------------------------------------------------

N_SOURCE_LMSYS = 3800  # oversample; row-pairing after judge-filter -> n=3000
N_TARGET_CORPUS = 3000

# --- Personas (from #1310) -------------------------------------------------
# Verbatim from scripts/issue1310_common.py::PERSONAS lines 128-133,
# Vex intentionally excluded from the headline design.
PERSONAS: dict[str, str] = {
    "Wren": "a warm, endlessly helpful assistant who patiently helps anyone who asks",
    "HELIOS": "a calm, precise artificial intelligence",
    "Dana": "an ordinary, unremarkable everyday person",
}

CHARACTER_LABELS: tuple[str, ...] = tuple(PERSONAS)

# --- Conditions ------------------------------------------------------------

Framing = Literal["chat", "naturalistic", "story"]
FRAMINGS: tuple[Framing, ...] = ("chat", "naturalistic", "story")

# --- User provenance -------------------------------------------------------

UserProvenance = Literal["lmsys", "haiku", "onpolicy"]
USER_PROVENANCES: tuple[UserProvenance, ...] = ("lmsys", "haiku", "onpolicy")


@dataclass(frozen=True)
class Condition:
    """A single (identity × framing × [user-provenance]) rendering cell."""

    slug: str  # Config slug per plan §5 (e.g. "assistant_chat", "user_lmsys_story").
    identity: str  # "assistant" | "user" | "wren" | "helios" | "dana"
    framing: Framing
    provenance: UserProvenance | None  # only set for user arm
    on_policy: bool  # True iff MEASURED model generates a2 (all non-user cells).

    @property
    def is_user(self) -> bool:
        return self.identity == "user"

    @property
    def is_character(self) -> bool:
        return self.identity in {"wren", "helios", "dana"}


def build_condition_table() -> list[Condition]:
    """Return the 21 conditions (plan §5 rendering lattice, verbatim).

    Order: assistant (3), user × 3 provenances × 3 framings (9), then
    Wren/HELIOS/Dana × 3 framings (9).
    """
    out: list[Condition] = []
    # Assistant (3)
    for f in FRAMINGS:
        out.append(Condition(f"assistant_{f}", "assistant", f, None, on_policy=True))
    # User × (provenance × framing) = 9
    for prov in USER_PROVENANCES:
        for f in FRAMINGS:
            # Note: user arm's "on-policy" arm has provenance="onpolicy" but the
            # DV is still measured on the MEASURED model. All three user
            # provenance arms mint u2 differently; a2 is generated on-policy.
            out.append(Condition(f"user_{prov}_{f}", "user", f, prov, on_policy=True))
    # Characters × framing = 9
    for name in CHARACTER_LABELS:
        for f in FRAMINGS:
            out.append(Condition(f"{name.lower()}_{f}", name.lower(), f, None, on_policy=True))
    assert len(out) == 21, f"Expected 21 conditions, got {len(out)}"
    return out


CONDITION_TABLE = build_condition_table()
SLUG_TO_CONDITION: dict[str, Condition] = {c.slug: c for c in CONDITION_TABLE}


def identity_display(condition: Condition) -> str:
    """Plain-English identity label used in rendered assistant tags."""
    if condition.identity == "assistant":
        return "Assistant"
    if condition.identity == "user":
        return "User"
    # character
    return {"wren": "Wren", "helios": "HELIOS", "dana": "Dana"}[condition.identity]


def system_prompt_for(condition: Condition) -> str | None:
    """System prompt for the persona-header steering.

    - assistant/user: no persona system prompt (defaults).
    - character: use the persona description as the system prompt (contract
      match with #1310 persona injection).
    """
    if condition.is_character:
        name = identity_display(condition)
        desc = PERSONAS[name]
        return f"You are {name}, {desc}."
    return None


# --- 95-pair spoke + structured pair set (plan §4) -------------------------

_ASSIST_REF = "assistant_chat"


def enumerate_pair_set() -> list[tuple[str, str]]:
    """Return the plan's ~95 ordered pairs (spoke + within-identity framing +
    within-framing identity + user-provenance triples), de-duplicated.

    Total: 40 spoke + 30 within-identity + 18 within-framing + 18 provenance
    = 106 raw pairs; de-dup removes 11 pairs that appear in >1 subset.
    """
    pairs: set[tuple[str, str]] = set()

    def add_bothways(a: str, b: str) -> None:
        if a != b:
            pairs.add((a, b))
            pairs.add((b, a))

    # Spoke against assistant_chat: every non-ref condition to/from ref.
    for c in CONDITION_TABLE:
        if c.slug != _ASSIST_REF:
            add_bothways(_ASSIST_REF, c.slug)

    # Within-identity framing triples (5 identities: assistant, user (per prov), 3 chars).
    identities: list[list[str]] = []
    identities.append([f"assistant_{f}" for f in FRAMINGS])
    for prov in USER_PROVENANCES:
        identities.append([f"user_{prov}_{f}" for f in FRAMINGS])
    for name in CHARACTER_LABELS:
        identities.append([f"{name.lower()}_{f}" for f in FRAMINGS])
    for ident in identities:
        for i in range(len(ident)):
            for j in range(i + 1, len(ident)):
                add_bothways(ident[i], ident[j])

    # Within-framing identity quads: {assistant, HELIOS, Wren, Dana} within each framing.
    for f in FRAMINGS:
        quad = [f"assistant_{f}", f"helios_{f}", f"wren_{f}", f"dana_{f}"]
        for i in range(len(quad)):
            for j in range(i + 1, len(quad)):
                add_bothways(quad[i], quad[j])

    # User-provenance triples: {lmsys, haiku, onpolicy} within each framing.
    for f in FRAMINGS:
        prov_conds = [f"user_{prov}_{f}" for prov in USER_PROVENANCES]
        for i in range(len(prov_conds)):
            for j in range(i + 1, len(prov_conds)):
                add_bothways(prov_conds[i], prov_conds[j])

    return sorted(pairs)


# --- Ridge grid (from #825 Phase-0 selector audit) ------------------------
LAMBDA_LOG_MIN = -2.0
LAMBDA_LOG_MAX = 4.0
LAMBDA_GRID_SIZE = 13

# Named lambda grids (wider-lambda-ceilings follow-up, plan v6 §4).
# "ladder13" is the parent grid (== issue825_fit_cells.LAMBDAS == the
# logspace(LAMBDA_LOG_MIN, LAMBDA_LOG_MAX, LAMBDA_GRID_SIZE) above);
# "wide19" extends it 3 dex at the same 0.5-dex spacing — a strict superset
# (np.intersect1d(g13, g19).size == 13), so published R²s are reproducible
# members of the wide scan and any Δceiling is attributable to the added λs.
LAMBDA_GRIDS: dict[str, tuple[float, float, int]] = {
    "ladder13": (LAMBDA_LOG_MIN, LAMBDA_LOG_MAX, LAMBDA_GRID_SIZE),
    "wide19": (LAMBDA_LOG_MIN, 7.0, 19),
}


def resolve_lambda_grid(name: str):
    """Return the named lambda grid as a float64 numpy array; fail loud on an
    unknown name. numpy imported lazily (this module stays import-light)."""
    import numpy as np

    if name not in LAMBDA_GRIDS:
        raise ValueError(f"unknown lambda grid {name!r} (want one of {sorted(LAMBDA_GRIDS)})")
    lo, hi, n = LAMBDA_GRIDS[name]
    return np.logspace(lo, hi, n)


N_FOLDS = 5
N_BOOTSTRAP_DRAWS = 1000
N_REPARAM_NULL_DRAWS = 200
RUNG_REACHED_THRESHOLD = 0.9  # R2_transfer >= 0.9 * R2_within(T)

# --- Well-posed reduced-basis round (`wellposed-shared-readout`, plan v10) ---
# k_unit = min(PCA_K_CAP, floor(min-fold n_train / 2)) so n_train >= 2k on
# EVERY fold (plan v10 s4 item 1; changing the formula/cap is must-ask s13).
PCA_K_CAP = 1024
K_FLOOR_LIMITED = 8  # report-only diagnostic label (plan v10 s6); no gating
# k-band edges aligned to the parent truncation grid (plan v10 s6/ s11).
K_BAND_EDGES = (32, 128, 512)


def k_band(k: int) -> str:
    """Registered k-band label for stratified reporting (plan v10 s6)."""
    if k < K_BAND_EDGES[0]:
        return "k<32"
    if k < K_BAND_EDGES[1]:
        return "32-127"
    if k < K_BAND_EDGES[2]:
        return "128-511"
    return ">=512"


# --- Judge -----------------------------------------------------------------
JUDGE_MODEL = "claude-sonnet-4-5-20250929"
JUDGE_N_DRAWS = 3
JUDGE_TEMPERATURE = 0.7
JUDGE_MAX_TOKENS = 1024  # rationale + score; llm-judging rule 23 floor (raised from 300, #2063)
YIELD_FLOOR = 0.80

# --- Generation ------------------------------------------------------------
GEN_TEMPERATURE = 0.7
GEN_TOP_P = 0.95
GEN_MAX_NEW_TOKENS = 1024

# --- Paths -----------------------------------------------------------------

ISSUE_NUM = 1689
ISSUE_SLUG = "speaker_lattice"
HF_DATA_PREFIX = f"issue{ISSUE_NUM}_{ISSUE_SLUG}"


def default_repo_root() -> str:
    """Resolve REPO_ROOT with the workload-first fallback (#825 crash-fix).

    Env order: EPS_REPO_ROOT > WORKLOAD_ROOT > pwd at import time.
    """
    import os
    from pathlib import Path

    for env in ("EPS_REPO_ROOT", "WORKLOAD_ROOT"):
        v = os.environ.get(env)
        if v:
            return v
    return str(Path(__file__).resolve().parents[1])
