"""Shared constants, registries, and helpers for issue #1336 (RLVR ladder).

#1336 extends the #825 recipe (per-example held-out ridge map c_x -> v(x),
reparameterization-gap battery) from the Qwen2.5-7B base/instruct pair to the
released Llama-3.1-8B Tulu-3 separated-stage ladder (base -> SFT -> DPO ->
RLVR, + a longer-RLVR secondary arm). Fit/statistics constants are
RE-EXPORTED from the #825 ground truth (`experiments/issue_825/common.py`) —
single source of truth; only the model/corpus/cell registries, the Tulu
chat-template constants, and the Llama frozen-layer set are new here.

Plan: tasks/*/1336/plans/plan.md (v3). All hyperparameters carry plan §11
sources; do not retype values from memory.
"""

from __future__ import annotations

import contextlib
import json
from pathlib import Path

from explore_persona_space.experiments.issue_825.common import (
    FIT_SEED,
    GEN_SEED,
    HF_DATA_REPO,
    MAX_CONV_TOKENS,
    MIN_TURN_CONTENT_TOKENS,
    N_BOOTSTRAP,
    N_FOLDS,
    N_NULL_DRAWS,
    Rendered,
)

__all__ = [
    "ADJACENT_PAIRS",
    "BASE_ANCHORED_PAIRS",
    "CELLS",
    "CORPORA",
    "DELTA_ELICIT_BAND",
    "DELTA_PRACTICAL_SCALE",
    "EVAL_SETS",
    "EXPECTED_HIDDEN",
    "EXPECTED_LAYERS",
    "FIT_SEED",
    "FORMATS_BY_CORPUS",
    "FROZEN_LAYERS",
    "G0",
    "G1_KILL_R2",
    "G1_MARGINAL_R2",
    "GEN_SEED",
    "GSM8K_CONFIG",
    "GSM8K_DATASET",
    "GSM8K_REV",
    "GSM8K_SPLIT_SIZES",
    "HF_DATA_REPO",
    "HF_PREFIX_1336",
    "KEEP_RATE_FLOOR",
    "MATCHED_N",
    "MAX_CONV_TOKENS",
    "MAX_MODEL_LEN",
    "MIN_TURN_CONTENT_TOKENS",
    "MODELS",
    "N_BOOTSTRAP",
    "N_FOLDS",
    "N_NULL_DRAWS",
    "PAIRS",
    "PREDS_EXTRA_LAYERS",
    "PRIMARY_LADDER",
    "PROMPT_TOKEN_BUDGET",
    "ROLE_HEADER_TRUNCATE",
    "SAMPLING",
    "SMOKE_CORPORA",
    "SMOKE_FROZEN_LAYERS",
    "SMOKE_MODELS",
    "SMOKE_N",
    "SMOKE_NULL_DRAWS",
    "SMOKE_N_BOOT",
    "STOP_STRINGS",
    "TRACK_S_BYTES",
    "TRACK_S_PATH",
    "TRACK_S_REV",
    "TULU_ASSISTANT_HEADER",
    "TULU_TURN_SEP",
    "TULU_USER_HEADER",
    "Rendered",
    "cell_id",
    "cells_for",
    "fc_expected_layers",
    "load_qwen_recal_cal",
    "preds_layers",
    "tulu_prompt",
]


@contextlib.contextmanager
def fc_expected_layers(fit_cells_module, n_layers: int):
    """Scoped rebind of the #825 fit core's ``EXPECTED_LAYERS`` module global.

    ``issue825_fit_cells._cell_xy`` asserts the bundle's layer axis against its
    own Qwen constant (28). Every #1336 driver call into ``_cell_xy`` wraps in
    this scope so the fail-loud shape check validates the RIGHT invariant:
    production Llama stores assert 32 (``EXPECTED_LAYERS`` here), the pinned
    Qwen G0 store asserts ``G0["expected_layers"]`` (28), and tiny smoke /
    fixture stores assert their own realized layer count (the extract-side
    ``--tiny-model-dir`` rebinding pattern). Restores the previous value on
    exit so the #825 module is never left mutated.
    """
    prev = fit_cells_module.EXPECTED_LAYERS
    fit_cells_module.EXPECTED_LAYERS = int(n_layers)
    try:
        yield
    finally:
        fit_cells_module.EXPECTED_LAYERS = prev


# ---------------------------------------------------------------------------
# Architecture invariants (Llama-3.1-8B family; asserted at model load)
# ---------------------------------------------------------------------------
EXPECTED_LAYERS = 32
EXPECTED_HIDDEN = 4096

# Frozen read-out layers: fractional-depth remap of the Qwen frozen set
# {14, 18, 19, 26}/28 onto 32 layers (plan §11 — `ungrounded — needs
# smoke-test` for this family; mitigated by the full 32-layer sweep + the
# stage-symmetric headline rule).
FROZEN_LAYERS = (16, 21, 22, 30)

# Default-preserving extension (plan v9 route 1): the E1 verdict layer (L29,
# the S_r argmax on the recalibrated read) ALSO gets held-out preds persisted
# + the recal primary computed, so the E1 verdict layer stays comparable
# across stages. The registered frozen set above is UNCHANGED — headline rule,
# selection-symmetric frozen table, cosine/CI reads all stay on FROZEN_LAYERS.
PREDS_EXTRA_LAYERS = (29,)


def preds_layers(frozen: tuple[int, ...] | list[int]) -> tuple[int, ...]:
    """Layer set for preds persistence + the recal primary: frozen + extras.

    ONE shared resolver for smoke and production (no smoke ternary — #825
    lesson): out-of-range extras on a tiny smoke store are guard-skipped by
    the sweep's own ``li < n_layers`` checks, so both modes run this line.
    """
    return tuple(sorted(set(int(x) for x in frozen) | set(PREDS_EXTRA_LAYERS)))


# ---------------------------------------------------------------------------
# Model ladder (Hub-verified lineage, plan §10; slugs are the stem prefix)
# ---------------------------------------------------------------------------
MODELS: dict[str, dict] = {
    "base": {"hf_id": "meta-llama/Llama-3.1-8B", "stage": 0, "label": "Pretrained base"},
    "sft": {"hf_id": "allenai/Llama-3.1-Tulu-3-8B-SFT", "stage": 1, "label": "After SFT"},
    "dpo": {"hf_id": "allenai/Llama-3.1-Tulu-3-8B-DPO", "stage": 2, "label": "After DPO"},
    "rlvr": {"hf_id": "allenai/Llama-3.1-Tulu-3-8B", "stage": 3, "label": "After RLVR"},
    "rlvr_long": {
        "hf_id": "allenai/Llama-3.1-Tulu-3.1-8B",
        "stage": 4,
        "label": "After longer RLVR (secondary)",
    },
}
PRIMARY_LADDER = ("base", "sft", "dpo", "rlvr")

# ---------------------------------------------------------------------------
# Tulu-3 chat template rendered as plain text. Role headers are PLAIN TEXT
# (not special tokens) and tokenize to IDENTICAL ids under all 5 checkpoints'
# tokenizers (verified 2026-07-15 on the VM: apply_chat_template(modulo
# leading BOS) == this constant for all four Tulu checkpoints; base ships no
# chat template and is rendered with the same string — plan §4).
# ---------------------------------------------------------------------------
TULU_USER_HEADER = "<|user|>\n"
TULU_ASSISTANT_HEADER = "<|assistant|>\n"
TULU_TURN_SEP = "\n"


def tulu_prompt(question: str) -> str:
    """Generation prompt for one single-turn question under the Tulu template."""
    return f"{TULU_USER_HEADER}{question}{TULU_TURN_SEP}{TULU_ASSISTANT_HEADER}"


# ---------------------------------------------------------------------------
# Corpora (plan §4): lmsys5k reuses the pinned #825 Track-S prompts; the two
# GSM8K arms come from the pinned openai/gsm8k release.
# ---------------------------------------------------------------------------
TRACK_S_REV = "deb7a4523b5233393e4fbd2497622527b3622d35"
TRACK_S_PATH = "issue825_userbase_map/raw_completions/track_s/track_s.jsonl"
TRACK_S_BYTES = 9_036_307  # Hub-verified byte size at TRACK_S_REV (plan §10)

GSM8K_DATASET = "openai/gsm8k"
GSM8K_CONFIG = "main"
GSM8K_REV = "740312add88f"  # dataset sha pin (plan §10)
GSM8K_SPLIT_SIZES = {"train": 7473, "test": 1319}  # asserted at ingest, fail-loud

CORPORA: dict[str, dict] = {
    "lmsys5k": {"n": 5000, "source": "issue825-track-s-pinned"},
    "gsm8k_train5k": {"n": 5000, "source": "gsm8k", "split": "train"},
    "gsm8k_test1319": {"n": 1319, "source": "gsm8k", "split": "test"},
}

# Naturalistic format arm on lmsys5k only (plan §4 stated scope).
FORMATS_BY_CORPUS: dict[str, tuple[str, ...]] = {
    "lmsys5k": ("chat", "naturalistic"),
    "gsm8k_train5k": ("chat",),
    "gsm8k_test1319": ("chat",),
}

# Eval sets for the Phase-A ladder-alignment battery (plan §4 Phase A).
EVAL_SETS = (
    ("lmsys5k", "chat"),
    ("lmsys5k", "naturalistic"),
    ("gsm8k_train5k", "chat"),
    ("gsm8k_test1319", "chat"),
)

# Stage pairs: base-anchored carry the headline; adjacent are the registered
# secondary (per-stage increments read directly). Plan §4 Phase A.
BASE_ANCHORED_PAIRS = (("base", "sft"), ("base", "dpo"), ("base", "rlvr"), ("base", "rlvr_long"))
ADJACENT_PAIRS = (("sft", "dpo"), ("dpo", "rlvr"), ("dpo", "rlvr_long"))
PAIRS = BASE_ANCHORED_PAIRS + ADJACENT_PAIRS

# ---------------------------------------------------------------------------
# Sampling + filters (Source: scripts/issue825_gen_conversations.py:521 —
# parent-exact Track-S params; filters from #825 plan §11 via issue_825.common)
# ---------------------------------------------------------------------------
SAMPLING = {"n": 1, "temperature": 1.0, "top_p": 0.95, "max_tokens": 1024, "seed": 42}
MAX_MODEL_LEN = 4096  # contexts <=2048 + answers <=1024 with margin (plan §11)
PROMPT_TOKEN_BUDGET = MAX_MODEL_LEN - SAMPLING["max_tokens"]  # load-time bank gate (#952)
STOP_STRINGS = ("\n<|user|>",)  # base-model stop handling (plan §4 Phase G)
ROLE_HEADER_TRUNCATE = ("<|user|>", "<|assistant|>")  # post-hoc truncation markers
KEEP_RATE_FLOOR = 0.80  # report-never-pad floor per (model, corpus)

HF_PREFIX_1336 = "issue1336_rlvr_ladder"


# ---------------------------------------------------------------------------
# Cell registry: 5 models x (2 formats x lmsys5k + 1 x gsm8k_train5k +
# 1 x gsm8k_test1319) = 20 cells. cell_id doubles as the turnstore shard stem.
# ---------------------------------------------------------------------------
def cell_id(model: str, fmt: str, corpus: str) -> str:
    """Canonical cell id / shard stem for one (model, format, corpus) cell."""
    return f"{model}_{fmt}_{corpus}"


def cells_for(
    models: tuple[str, ...] | list[str] | None = None,
    corpora: tuple[str, ...] | list[str] | None = None,
) -> list[dict]:
    """Cell dicts for a model/corpus subset (the smoke/production subset seam).

    Every phase (gen, extract, fit, align) derives its work list from THIS
    function so a smoke subset threads through the whole dispatcher
    (PASS_UNIFIED contract). Returns one dict per (model, format, corpus).
    """
    models = tuple(models) if models is not None else tuple(MODELS)
    corpora = tuple(corpora) if corpora is not None else tuple(CORPORA)
    for m in models:
        assert m in MODELS, f"unknown model slug {m!r}"
    for c in corpora:
        assert c in CORPORA, f"unknown corpus {c!r}"
    out = []
    for m in models:
        for c in corpora:
            for f in FORMATS_BY_CORPUS[c]:
                out.append(
                    {
                        "cell_id": cell_id(m, f, c),
                        "model": m,
                        "hf_id": MODELS[m]["hf_id"],
                        "format": f,
                        "corpus": c,
                    }
                )
    return out


CELLS = cells_for()
assert len(CELLS) == 20, f"cell registry drifted: {len(CELLS)} != 20"

# Smoke subset (PASS_UNIFIED: smoke IS the sweep with this subset). Three
# models so the align/decision phases see BOTH headline base-anchored pairs
# (base->dpo, base->rlvr) plus one adjacent pair (dpo->rlvr) — the contrast
# C = gap_rlvr - gap_dpo is computable on the smoke slice.
SMOKE_MODELS = ("base", "dpo", "rlvr")
SMOKE_CORPORA = ("lmsys5k",)
SMOKE_N = 8
SMOKE_NULL_DRAWS = 2
SMOKE_N_BOOT = 50
# Smoke frozen-layer set: the production {16, 21, 22, 30} is out of range on
# the tiny 2-4-layer smoke model, which would silently skip every frozen-layer
# code path (cosines / preds / CIs / degeneracy); the smoke threads THIS set
# through the same --frozen-layers parametrization production uses.
SMOKE_FROZEN_LAYERS = (0, 1)

# ---------------------------------------------------------------------------
# Gates (plan §7)
# ---------------------------------------------------------------------------
# G0 — fit-core reuse gate: refit the committed Qwen S1 cell from the pinned
# #825 turnstore through the generalized fit driver; PASS <=> layer-19
# held-out R^2 within +-0.01 of the committed 0.6731.
G0 = {
    "stem": "instruct_chat_s",
    "hf_prefix": "issue825_userbase_map/analysis_tensors",
    "revision": TRACK_S_REV,
    "expected_layers": 28,
    "expected_hidden": 3584,
    "layer": 19,
    "committed_r2": 0.6731,
    "tol": 0.01,
}

# G1 — rig-transfer kill gate: After-RLVR lmsys5k-chat cell first; KILL <=>
# best full-sweep within-stage held-out R^2 < 0.2 (chat in [0.2, 0.3) —
# marginal band — additionally checks naturalistic before any kill).
# RESUME re-adjudication (plan v9 §4 route 1): after the E1
# `resume_on_recalibrated_dv` route, the gate reads the held-out
# cross-fitted per-dim affine-recalibrated primary and both thresholds are
# carried via the persisted Qwen exchange rate (kill bar = the persisted
# bar_r = 0.20 x S_qwen_recal/0.6731; marginal = 0.3 x the same rate). The
# RAW-scale values below stay the companion read (never blended).
G1_KILL_R2 = 0.2
G1_MARGINAL_R2 = 0.3

# Δ_k bands (plan v3 §11; Source: #825 measured gap 0.0003 -> 0.02 elicitation
# band; #825 replication-gate tolerance -> 0.05 practical-significance scale).
# On the RECALIBRATED primary both bands are carried via the SAME Qwen
# exchange rate (plan v9 route 1: "v3's Δ bands carried via the same
# exchange rate"); the raw companion keeps the unscaled values.
DELTA_ELICIT_BAND = 0.02
DELTA_PRACTICAL_SCALE = 0.05

# Matched-n subsample size for cross-corpus comparability (the GSM8K test
# split size; #825 matched-n convention).
MATCHED_N = 1319


def load_qwen_recal_cal(out_dir: str | Path) -> dict:
    """Load + validate the persisted E1.d Qwen exchange-rate calibration.

    Plan v9 route 1 fix list: the per-stage usable-strength bar and the Δ
    bands ride the SAME persisted Qwen exchange rate — reuse the E1.d values
    (`<out_dir>/diagnosis/recal/qwen_recal_cal.json`, committed by the E1
    round), NEVER recompute Qwen. Fail-loud when the file is absent (a
    resume without the calibration must not silently fall back to raw bars)
    or when its V-gate did not pass (route 1 requires V PASS).

    Returns {s_qwen_recal, committed_anchor, rate, bar_r, marginal_r2, path}.
    """

    path = Path(out_dir) / "diagnosis" / "recal" / "qwen_recal_cal.json"
    assert path.exists(), (
        f"qwen_recal_cal.json missing at {path} — the resume's recalibrated bars require the "
        "committed E1.d exchange-rate calibration (plan v9 route 1); do not proceed on raw bars"
    )
    cal = json.loads(path.read_text())
    s = float(cal["s_qwen_recal"])
    anchor = float(cal["committed_anchor"])
    bar_r = float(cal["bar_r"])
    assert cal["v_gate"]["pass"] is True, (
        f"qwen_recal_cal.json at {path} records a FAILED V-gate — the recalibrated DV is not "
        "validated on this family (plan v9 terminal route); refuse the resume bars"
    )
    rate = s / anchor
    assert abs(bar_r - G1_KILL_R2 * rate) < 1e-9, (
        f"persisted bar_r {bar_r} != {G1_KILL_R2} x exchange rate {rate} — calibration file "
        "internally inconsistent"
    )
    return {
        "s_qwen_recal": s,
        "committed_anchor": anchor,
        "rate": rate,
        "bar_r": bar_r,
        "marginal_r2": G1_MARGINAL_R2 * rate,
        "path": str(path),
    }
