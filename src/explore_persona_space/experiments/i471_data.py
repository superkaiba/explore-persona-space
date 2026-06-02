"""Issue #471 shared helpers -- contrastive-negatives variant of #465.

Plan v1 §4.0 + §4.2 + §4.4 + §10. Single manipulated variable vs #465 per arm:
the presence of training-time contrastive negatives interleaved 1:1 with the
positive marker rows.

This module imports #465's data primitives (`load_q_train_answers`,
`load_q_test_extended_50`, `load_q_demo`, persona constants) and adds:

  * The 3 NEGATIVE personas (default helpful + 2 close named negatives per
    `.claude/rules/contrastive-negatives.md` §"Composition + ratio").
  * The 5 HELD-OUT BYSTANDER personas for eval read (f), all loaded through
    `EVAL_PERSONAS_24` (the union dict; hero + lawyer live in NEW_PERSONAS_274,
    NOT in NAMED_PERSONAS — see plan A22).
  * NEW R artifact loaders for the contrastive-negatives + bystander +
    held-out-default-like eval shapes (plan §10 R artifact rows).

The 4 #471 training arms reuse #465's exact persona / marker / R-villain /
recipe; only the negative rows are added. No HF data repo writes here — the
phase scripts handle that.
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

from explore_persona_space.experiments.factor_screen_365.persona_panel import (
    EVAL_PERSONAS_24,
    NAMED_PERSONAS,
)
from explore_persona_space.experiments.i465_data import (
    HELPFUL_SYSTEM_PROMPT,
)

logger = logging.getLogger("i471.data")

# ── Paths / HF locations ─────────────────────────────────────────────────
DATA_DIR_471 = Path("data/issue_471")
HF_DATA_REPO = "superkaiba1/explore-persona-space-data"
HF_PATH_PREFIX_471 = "issue471_contrastive_negatives"
HF_MODEL_REPO = "superkaiba1/explore-persona-space"

# ── Negative personas (3, plan §4.2) ─────────────────────────────────────
# Per `.claude/rules/contrastive-negatives.md` §"NEGATIVE row" the default
# helpful assistant is MANDATORY. medical_doctor + police_officer are the
# recurring close negatives from #247/#329 (medical) and #448 (police).
#
# Builder code uses only NAMED_PERSONAS to look these up — NAMED_PERSONAS
# contains medical_doctor + police_officer (verified persona_panel.py:77/85).
NEGATIVE_PERSONAS: dict[str, str] = {
    "default": HELPFUL_SYSTEM_PROMPT,
    "medical_doctor": NAMED_PERSONAS["medical_doctor"],
    "police_officer": NAMED_PERSONAS["police_officer"],
}
NEGATIVE_PERSONA_IDS: list[str] = list(NEGATIVE_PERSONAS.keys())

# ── Held-out bystander personas (5, plan §4.5 read f) ────────────────────
# Chosen to span occupational-technical / educational-relational /
# characterological-creative / narrative-archetypal / professional-formal.
# Disjoint from the 3 trained negative personas.
#
# CRITICAL (plan A22): hero + lawyer live in NEW_PERSONAS_274, not in
# NAMED_PERSONAS. We resolve through EVAL_PERSONAS_24 (the union dict);
# attempting to look them up in NAMED_PERSONAS alone would KeyError.
BYSTANDER_PERSONA_IDS: list[str] = [
    "software_engineer",
    "kindergarten_teacher",
    "comedian",
    "hero",
    "lawyer",
]


def get_bystander_personas() -> dict[str, str]:
    """Return {bystander_id: system_prompt} for the 5 held-out bystanders.

    Resolved through EVAL_PERSONAS_24 because hero + lawyer live in
    NEW_PERSONAS_274 (NAMED_PERSONAS alone would KeyError).
    Fail-loud if any bystander is missing — plan A22 contract.
    """
    missing = [p for p in BYSTANDER_PERSONA_IDS if p not in EVAL_PERSONAS_24]
    if missing:
        raise AssertionError(
            f"Bystander panel missing from EVAL_PERSONAS_24: {missing}. "
            f"Panel contract (plan A22) violated; persona_panel.py drifted."
        )
    return {p: EVAL_PERSONAS_24[p] for p in BYSTANDER_PERSONA_IDS}


# ── New eval shape served-system strings (MUST-FIX 3 / MUST-FIX 4) ───────
# Plan §4.5 reads (g) and (g'). Read (g) uses an empty / no-system-message
# chat template; we encode that as None and the prompt builder picks the
# branch. Read (g') is a paraphrased helpful system prompt.
PARAPHRASED_HELPFUL_SYSTEM_PROMPT = "You are an AI assistant. Help the user."

# ── R artifact filenames (plan §10) ──────────────────────────────────────
R_NEGATIVES_FILE = "R_negatives.json"  # 3 personas × 30 Q_train  = 90 entries
R_BYSTANDER_QTEST_FILE = "R_bystander_qtest.json"  # 5 × 50  = 250
R_TRAINED_NEGATIVES_QTEST_FILE = "R_trained_negatives_qtest.json"  # 2 × 50  = 100
R_HELPFUL_QTRAIN_FILE = "R_helpful_qtrain.json"  # 1 × 30  = 30
R_NO_SYSTEM_QTEST_FILE = "R_no_system_qtest.json"  # 1 × 50  = 50
R_PARAPHRASED_HELPFUL_QTEST_FILE = "R_paraphrased_helpful_qtest.json"  # 1 × 50  = 50


def _hf_pull(filename: str, local: Path) -> None:
    """Pull a #471 data artifact from HF data repo into `local`. Fail-loud."""
    from huggingface_hub import hf_hub_download

    logger.info("Pulling %s from HF data repo %s ...", filename, HF_DATA_REPO)
    local.parent.mkdir(parents=True, exist_ok=True)
    downloaded = hf_hub_download(
        repo_id=HF_DATA_REPO,
        repo_type="dataset",
        filename=f"{HF_PATH_PREFIX_471}/{filename}",
        revision="main",
    )
    shutil.copyfile(downloaded, local)
    if not local.exists() or local.stat().st_size == 0:
        raise RuntimeError(
            f"HF download of {filename} claimed success but {local} is empty/missing "
            f"after copy from {downloaded}."
        )


def load_r_artifact(filename: str, *, prefer_hf: bool = True) -> dict:
    """Load a frozen #471 R artifact JSON ({completions: {...}, ...}).

    Falls back to HF data repo if not present locally; fail-loud if neither
    available. Schema is the same `i465_v1` payload shape used by Phase 1
    (R_villain.json, R_helpful_qtest.json) for byte-identical downstream
    consumption.
    """
    local = DATA_DIR_471 / filename
    if not local.exists():
        if not prefer_hf:
            raise FileNotFoundError(f"R artifact missing locally at {local} and prefer_hf=False.")
        _hf_pull(filename, local)
    payload = json.loads(local.read_text())
    if payload.get("schema_version") != "i465_v1":
        raise AssertionError(
            f"{filename} schema_version={payload.get('schema_version')!r}, "
            f"expected 'i465_v1' (using the same shape as Phase 1)."
        )
    return payload


def load_r_negatives() -> dict[tuple[str, str], dict]:
    """Load R_negatives.json -> {(neg_persona, q): completion dict}.

    Built in Phase 0 (`i471_phase0_preflight.py`) by base-Qwen greedy under
    each negative persona's own system prompt on Q_train.
    """
    payload = load_r_artifact(R_NEGATIVES_FILE)
    # Schema: completions keyed by f"{persona}::{q}" since JSON dicts can't
    # carry tuple keys; we unpack here.
    raw = payload["completions"]
    out: dict[tuple[str, str], dict] = {}
    for k, v in raw.items():
        if "::" not in k:
            raise AssertionError(
                f"R_negatives.json: key {k!r} missing '::' delimiter between "
                f"persona and q. Schema drift."
            )
        persona, q = k.split("::", 1)
        out[(persona, q)] = v
    return out


def load_r_bystander_qtest() -> dict[tuple[str, str], dict]:
    """Load R_bystander_qtest.json -> {(bystander, q): completion}."""
    payload = load_r_artifact(R_BYSTANDER_QTEST_FILE)
    raw = payload["completions"]
    out: dict[tuple[str, str], dict] = {}
    for k, v in raw.items():
        persona, q = k.split("::", 1)
        out[(persona, q)] = v
    return out


def load_r_trained_negatives_qtest() -> dict[tuple[str, str], dict]:
    """Load R_trained_negatives_qtest.json -> {(neg, q): completion}.

    Covers medical_doctor + police_officer × Q_test (2 × 50 = 100). The
    default-persona × Q_test cell == R_helpful_qtest (inherited from #465).
    """
    payload = load_r_artifact(R_TRAINED_NEGATIVES_QTEST_FILE)
    raw = payload["completions"]
    out: dict[tuple[str, str], dict] = {}
    for k, v in raw.items():
        persona, q = k.split("::", 1)
        out[(persona, q)] = v
    return out


def load_r_helpful_qtrain() -> dict[str, dict]:
    """Load R_helpful_qtrain.json -> {q: completion} on Q_train."""
    payload = load_r_artifact(R_HELPFUL_QTRAIN_FILE)
    return payload["completions"]


def load_r_no_system_qtest() -> dict[str, dict]:
    """Load R_no_system_qtest.json -> {q: completion} on Q_test (no system)."""
    payload = load_r_artifact(R_NO_SYSTEM_QTEST_FILE)
    return payload["completions"]


def load_r_paraphrased_helpful_qtest() -> dict[str, dict]:
    """Load R_paraphrased_helpful_qtest.json -> {q: completion} on Q_test."""
    payload = load_r_artifact(R_PARAPHRASED_HELPFUL_QTEST_FILE)
    return payload["completions"]
