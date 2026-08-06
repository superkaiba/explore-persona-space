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
import os
import subprocess
from pathlib import Path

import numpy as np

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
    "BAND_V2_COEF",
    "BAR_V2_COEF",
    "BASE_ANCHORED_PAIRS",
    "CELLS",
    "CELLS_V2",
    "CELLS_V3",
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
    "HEALTH_V2_COEF",
    "HF_DATA_REPO",
    "HF_PREFIX_1336",
    "KEEP_RATE_FLOOR",
    "LAMBDAS_23",
    "MATCHED_N",
    "MATCHED_N_V2",
    "MATCHED_N_V2_CORPORA",
    "MATCHED_N_V2_SEED",
    "MAX_CONV_TOKENS",
    "MAX_EDGE_EXTENSIONS",
    "MAX_MODEL_LEN",
    "MIN_TURN_CONTENT_TOKENS",
    "MODELS",
    "NATURAL_ASSISTANT_HEADER",
    "NATURAL_ROLE_HEADER_TRUNCATE",
    "NATURAL_STOP_STRINGS",
    "NATURAL_TURN_SEP",
    "NATURAL_USER_HEADER",
    "N_BOOTSTRAP",
    "N_FOLDS",
    "N_INNER_LAMBDA_FOLDS_V2",
    "N_NULL_DRAWS",
    "PAIRS",
    "POOLED_ARM_DIRS",
    "PRACTICAL_V2_COEF",
    "PREDS_EXTRA_LAYERS",
    "PRIMARY_LADDER",
    "PROMPT_TOKEN_BUDGET",
    "ROLE_HEADER_TRUNCATE",
    "SAMPLING",
    "SMOKE_CORPORA",
    "SMOKE_CORPORA_V2",
    "SMOKE_FROZEN_LAYERS",
    "SMOKE_MODELS",
    "SMOKE_N",
    "SMOKE_NULL_DRAWS",
    "SMOKE_N_BOOT",
    "SMOKE_OFFDIAG_PAIRS_V3",
    "STOP_STRINGS",
    "TRACK_S_BYTES",
    "TRACK_S_PATH",
    "TRACK_S_REV",
    "TULU_ASSISTANT_HEADER",
    "TULU_TURN_SEP",
    "TULU_USER_HEADER",
    "V2_CORPORA",
    "V2_GEN_FORMATS",
    "V2_PREFIX_ARM",
    "V3_TEXT_FORMAT",
    "Rendered",
    "cell_id",
    "cells_for",
    "cells_v2_for",
    "cells_v3_for",
    "fc_expected_layers",
    "gen_cell_key",
    "load_qwen_recal_cal",
    "natural_prompt",
    "offpolicy_ts_dirname",
    "preds_layers",
    "resolve_code_sha",
    "tulu_prompt",
    "v2_bars",
    "v2_cell_id",
    "v2_surface_index",
    "v2_surfaces",
    "v3_pair_id",
]


def resolve_code_sha(repo_root: str | Path | None = None) -> str:
    """Lane-robust code-sha for provenance metadata — NEVER raises.

    The fellows/SLURM lane materializes the scratch tree via rsync with NO
    ``.git`` (job 17987: a pod-side ``git rev-parse HEAD`` with ``check=True``
    exited 128 and crashed g2_parity AFTER the parity compare had run), so
    provenance resolution degrades instead of crashing (the #1902
    ``_git_sha`` convention):

    1. ``EPS_GIT_SHA`` env wins when the launcher exports one;
    2. ``git rev-parse HEAD`` with ``check=False`` — git-ful lanes
       (GCP/RunPod clones) keep returning the real sha, behavior unchanged;
    3. the literal ``"unknown-no-git"`` on any failure (rc != 0, git binary
       absent) — provenance metadata must never kill a phase.
    """
    env = os.environ.get("EPS_GIT_SHA", "").strip()
    if env:
        return env
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root) if repo_root is not None else None,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return "unknown-no-git"
    sha = proc.stdout.strip()
    return sha if proc.returncode == 0 and sha else "unknown-no-git"


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
# Naturalistic (#825 plain-transcript) render constants + generation prompt
# (round 5: the on-policy naturalistic arm). `issue1336_render.render_natural`
# builds its segments from THESE constants, so the generation prefix below
# matches the extraction render byte-for-byte BY CONSTRUCTION.
# ---------------------------------------------------------------------------
NATURAL_USER_HEADER = "User: "
NATURAL_TURN_SEP = "\n\n"
NATURAL_ASSISTANT_HEADER = "Assistant: "


def natural_prompt(question: str) -> str:
    """Generation prompt for one single-turn question under the #825
    naturalistic plain-transcript convention — everything before the model's
    answer (``User: {q}\\n\\nAssistant: ``): the first four segments of
    ``issue1336_render.render_natural`` joined."""
    return f"{NATURAL_USER_HEADER}{question}{NATURAL_TURN_SEP}{NATURAL_ASSISTANT_HEADER}"


# On-policy naturalistic stop handling (round 5): the plain-transcript
# analogues of STOP_STRINGS / ROLE_HEADER_TRUNCATE below — newline-anchored so
# a legitimate mid-line "User:"/"Assistant:" mention inside an answer never
# truncates, while a model opening the NEXT transcript turn (role header at a
# line start) does. Mirrors the chat pair's structure: stop at the next user
# turn; post-hoc truncate at ANY role-header reoccurrence.
NATURAL_STOP_STRINGS = ("\nUser:",)
NATURAL_ROLE_HEADER_TRUNCATE = ("\nUser:", "\nAssistant:")


def gen_cell_key(corpus: str, gen_format: str) -> str:
    """Generation-cell directory / HF-prefix token for (corpus, gen format).

    ``chat`` keeps the bare corpus — byte-identical to every prior round's
    local dirs + Hub prefixes, so existing chat artifacts and resume state
    stay valid. Any other format gets an explicit suffix so on-policy
    naturalistic answers can never overwrite (or be confused with) the chat
    arm's. Shared by `issue1336_gen_answers` (writer) and
    `issue1336_extract_turnstore` (reader) so the two can never drift.
    """
    assert gen_format in ("chat", "naturalistic"), f"unknown gen format {gen_format!r}"
    return corpus if gen_format == "chat" else f"{corpus}__gen_{gen_format}"


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


def _assert_registry(cells: list[dict], expected: int, name: str) -> None:
    """Registry-size + unique-id pin, parametrized per registry (plan §12
    assumption-12 must-fix: the v1 registry keeps its assert while CELLS_V2
    below carries its own expected count)."""
    assert len(cells) == expected, f"{name} cell registry drifted: {len(cells)} != {expected}"
    ids = [c["cell_id"] for c in cells]
    assert len(set(ids)) == len(ids), f"{name} cell registry has duplicate cell ids"


CELLS = cells_for()
_assert_registry(CELLS, 20, "v1")

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

# ---------------------------------------------------------------------------
# v2 registries + estimator constants (plan v13, same-issue follow-up round
# `full-corpora-stage-evals-metric-ladder`)
# ---------------------------------------------------------------------------
# v2 corpus registry (plan §4 corpora table). Canonical home is HERE so the
# v2 cell registry below derives formats without importing scripts/;
# `scripts/issue1336_stage_corpora.py` re-exports it for its established
# consumers, and that module's `load_v2_corpus_rows` / `read_corpus_rows_local`
# stay the ONLY corpus readers (sharded corpora: manifest + shards on HF).
V2_CORPORA: dict[str, dict] = {
    "lmsys23k": {"formats": ("chat", "naturalistic"), "n_target": 23_000},
    "gsm8k_train_full": {"formats": ("chat",), "n_target": 7473},
    "gsm8k_test1319": {"formats": ("chat",), "n_target": 1319},
    "math7500": {"formats": ("chat",), "n_target": 7500},
    "if11k": {"formats": ("chat",), "n_target": 11_000},
    "uf11k": {"formats": ("chat",), "n_target": 11_000},
    "sft11k": {"formats": ("chat",), "n_target": 11_000},
}

# Naturalistic prefix-arm cells (plan §4 divergence 7: the prefix slot is NOT
# row-constant under the naturalistic render — measured max pairwise cosine
# distance 0.63-0.76 — so the prefix arm is FIT there, closing the parent
# caveat). Same turnstore bundle as the context arm; only X = prefix slot.
V2_PREFIX_ARM = (("lmsys23k", "naturalistic"),)

# Naturalistic GENERATION + ON-POLICY-CAPTURE extension (user directive: "run
# naturalistic on everything (but only the full context arm)"). Consulted by
# exactly TWO seams: (1) the gen path (`issue1336_gen_answers._formats_for`)
# ACCEPTS a `--gen-format naturalistic` run on the six chat-only corpora
# above; (2) the extractor's on-policy licensing
# (`issue1336_extract_turnstore.extraction_licensed`) ACCEPTS
# `--v2 --format F --gen-format F` pairs for a gen-licensed F (fmt ==
# gen_format ONLY — cross-format matched-text pairs stay fit-side-gated). The
# FIT-side grid (`V2_CORPORA[...]["formats"]`, `v2_surfaces`, `v2_surface_index`,
# `cells_v2_for`, `CELLS_V2`) is deliberately UNCHANGED until the naturalistic
# stores exist: widening `V2_CORPORA` formats would (a) shift `v2_surface_index`
# for 5 existing surfaces, silently changing the §3 paired-bootstrap seeds
# (5000 + index) mid-round, and (b) add 30 storeless context cells to every
# `cells_v2_for()`-derived work list while round 4 executes. When the
# naturalistic turnstores land, widening `V2_CORPORA` (APPEND-ordered to keep
# existing surface indices stable) supersedes this registry.
V2_GEN_FORMATS: dict[str, tuple[str, ...]] = {c: ("chat", "naturalistic") for c in V2_CORPORA}

# Extended-corpus concat registry (plan v13 §4 Phase GEN/EXT/FIT). Canonical
# home is HERE so gen prep (new-prompts-only filter), the extractor
# (extension-only rows), and the fit/ladder concat loader all read ONE
# boundary; `issue1336_extract_turnstore.py` aliases these for its
# established consumers. Wave-1 covered prompt_idx 0..4999 of each extended
# corpus; the v2 round generates/extracts ONLY prompt_idx >= boundary.
V2_CONCAT_SOURCES = {"lmsys23k": "lmsys5k", "gsm8k_train_full": "gsm8k_train5k"}
V2_CONCAT_BOUNDARY = {"lmsys23k": 5000, "gsm8k_train_full": 5000}

# Phase GEN scope (plan §4): gsm8k_test1319 is FULLY reused from wave 1 —
# no new generation and no new extraction; its wave-1 turnstores are staged
# verbatim into the v2 turnstore dir at c_stage (plan §9 phase_outputs).
V2_FULLY_REUSED_GEN = ("gsm8k_test1319",)

# Wave-1 reuse revision pin (plan §10: generations + 20 turnstores + preds
# Hub-verified @ data-repo rev 8c54f9fc; c_stage stages at this revision).
WAVE1_HF_REV = "8c54f9fc"

# v2 estimator constants (plan §11):
# - LAMBDAS_23: Source #779 LAMBDAS_N1M (issue779_ffc_n1m_fits.py:112) — the
#   Qwen-line standard at n > d; spans both observed v8 edge regimes (floor
#   1e-2 hits AND ceiling 1e4 hits) with >= 1 decade margin each side.
# - N_INNER_LAMBDA_FOLDS_V2: n_inner 4 -> 2 (cost-grounded — halves inner
#   eighs; strictly upgrades #779's single 75/25 val split).
# - MAX_EDGE_EXTENSIONS: adaptive edge rule — <= 2 one-decade extensions per
#   side, then the `estimator-limited: lambda-edge` label (user directive;
#   operationalizes the v8 lambda-audit record).
LAMBDAS_23 = np.logspace(-3, 8, 23)
N_INNER_LAMBDA_FOLDS_V2 = 2
MAX_EDGE_EXTENSIONS = 2

# Matched-n companions (plan §4 Phase FIT): headline-layer-set-only refits at
# n=7,350 (seed-1336 subsample) for the four above-size corpora, for
# cross-corpus tier-profile comparisons.
MATCHED_N_V2 = 7350
MATCHED_N_V2_SEED = 1336
MATCHED_N_V2_CORPORA = ("lmsys23k", "if11k", "uf11k", "sft11k")

# v2 smoke corpus subset (PASS_UNIFIED: the v2 smoke IS the v2 sweep at
# SMOKE_MODELS x this subset — both lmsys23k formats + the prefix arm, so
# every v2 arm class gets a smoke cell).
SMOKE_CORPORA_V2 = ("lmsys23k",)


def v2_surfaces() -> tuple[tuple[str, str], ...]:
    """The 8 registered v2 eval SURFACES (corpus, format) in registry order.

    Context-arm only (prefix-arm cells are NOT ladder surfaces — plan §4
    Phase LAD). The ORDER is load-bearing: the paired-bootstrap seed is
    ``5000 + v2_surface_index(...)`` (plan §3), shared between the Phase-LAD
    battery and the Phase-P decision writer.
    """
    return tuple((c, f) for c in V2_CORPORA for f in V2_CORPORA[c]["formats"])


def v2_surface_index(corpus: str, fmt: str) -> int:
    """Registered surface index (the §3 bootstrap-seed offset)."""
    surfaces = v2_surfaces()
    key = (corpus, fmt)
    assert key in surfaces, f"unknown v2 surface {key!r} (known: {surfaces})"
    return surfaces.index(key)


def v2_cell_id(model: str, fmt: str, corpus: str, x_slot: str = "context") -> str:
    """v2 cell id / output stem: context arm keeps ``cell_id``; prefix-arm
    cells get the ``_xprefix`` suffix (same turnstore stem, distinct outputs)."""
    assert x_slot in ("context", "prefix"), f"unknown x_slot {x_slot!r}"
    base = cell_id(model, fmt, corpus)
    return base if x_slot == "context" else f"{base}_xprefix"


def cells_v2_for(
    models: tuple[str, ...] | list[str] | None = None,
    corpora: tuple[str, ...] | list[str] | None = None,
) -> list[dict]:
    """v2 cell dicts for a model/corpus subset (the smoke/production seam).

    Full grid: 40 context-arm cells (5 models x 8 corpus-format surfaces) +
    5 naturalistic prefix-arm cells. Every v2 phase derives its work list
    from THIS function so a smoke subset threads through the whole
    dispatcher (PASS_UNIFIED contract). Each dict carries
    ``x_slot: "context" | "prefix"``.
    """
    models = tuple(models) if models is not None else tuple(MODELS)
    corpora = tuple(corpora) if corpora is not None else tuple(V2_CORPORA)
    for m in models:
        assert m in MODELS, f"unknown model slug {m!r}"
    for c in corpora:
        assert c in V2_CORPORA, f"unknown v2 corpus {c!r}"
    out = []
    for m in models:
        for c in corpora:
            for f in V2_CORPORA[c]["formats"]:
                out.append(
                    {
                        "cell_id": v2_cell_id(m, f, c),
                        "model": m,
                        "hf_id": MODELS[m]["hf_id"],
                        "format": f,
                        "corpus": c,
                        "x_slot": "context",
                    }
                )
    for m in models:
        for c, f in V2_PREFIX_ARM:
            if c in corpora:
                out.append(
                    {
                        "cell_id": v2_cell_id(m, f, c, "prefix"),
                        "model": m,
                        "hf_id": MODELS[m]["hf_id"],
                        "format": f,
                        "corpus": c,
                        "x_slot": "prefix",
                    }
                )
    return out


CELLS_V2 = cells_v2_for()
_assert_registry(CELLS_V2, 45, "v2")

# G0'(c) exchange-rate-scaled v2 bands/bars (plan §7 / §3 / §11): ex_v2 =
# S_qwen_v2 / 0.6731 (the committed Qwen anchor, G0["committed_r2"]),
# computed BEFORE any Llama verdict read. The kill bar is EXACTLY
# bar_v2 = 0.20 * ex_v2 (the §7 form; §6's `0.2012*ex_v2/1.0062` rendering
# is the same number). Band / practical / health coefficients per §3 / §11.
BAR_V2_COEF = 0.20
BAND_V2_COEF = 0.0201  # §3: U = C_v2 - 0.0201*ex_v2, L = C_v2 + 0.0201*ex_v2
PRACTICAL_V2_COEF = 0.0503
HEALTH_V2_COEF = 0.05  # health gate H: |R2_recal - R2_raw| <= 0.05*ex_v2


# ---------------------------------------------------------------------------
# v3 registries (plan v15, same-issue follow-up round
# `pooled-multidataset-onoff-policy-stage-transfer`)
# ---------------------------------------------------------------------------
# v3 cell = one (activation-checkpoint i, text-source j) pair (plan §4
# divergence 4). Diagonal pairs (i == j) are the ON-policy arm and reuse the
# round-3 v2 captures verbatim (no new extraction); off-diagonal pairs
# (i != j) are the OFF-policy arm captured teacher-forced by Phase EXT_off.
# Off-diagonal capture is CHAT-format only (the naturalistic surface stays
# on-policy — plan §4 divergence 5).
V3_TEXT_FORMAT = "chat"

# Pooled-fit arm subdirectories (plan §6: preds_pooled_v3/{on,off}-policy/).
POOLED_ARM_DIRS = {"on": "on-policy", "off": "off-policy"}

# EXT_off smoke pair subset (PASS_UNIFIED seam: the smoke IS the sweep at
# this subset — dpo x rlvr-text is the plan's own G2v2 example cell).
SMOKE_OFFDIAG_PAIRS_V3 = (("dpo", "rlvr"),)


def offpolicy_ts_dirname(ckpt: str, text_source: str) -> str:
    """Local dir + Hub prefix leaf for one off-diagonal (i, j) capture tree
    (plan §4 Phase EXT_off output stems:
    ``analysis_tensors/turnstore_offpolicy_<slug_i>_chat_<slug_j>/``).
    Shard stems INSIDE the tree keep the standard ``cell_id(i, "chat", corpus)``
    naming so the #825 bundle loaders read them unchanged."""
    assert ckpt in MODELS, f"unknown checkpoint slug {ckpt!r}"
    assert text_source in MODELS, f"unknown text-source slug {text_source!r}"
    assert ckpt != text_source, "diagonal (i == j) pairs reuse the v2 turnstores — no offpol dir"
    return f"turnstore_offpolicy_{ckpt}_{V3_TEXT_FORMAT}_{text_source}"


def v3_pair_id(ckpt: str, text_source: str) -> str:
    """Canonical v3 cell id for one (activation-checkpoint, text-source) pair."""
    return f"{ckpt}_txt_{text_source}"


def cells_v3_for(
    models: tuple[str, ...] | list[str] | None = None,
    text_sources: tuple[str, ...] | list[str] | None = None,
) -> list[dict]:
    """v3 cell dicts for a checkpoint/text-source subset (smoke seam).

    Full grid: 25 pairs = 5 diagonal on-policy + 20 off-diagonal off-policy
    (plan §4: "parametrize for the v3 cell set"). Every v3 phase derives its
    work list from THIS function so a smoke subset threads through the whole
    dispatcher (PASS_UNIFIED contract). Each dict carries
    ``arm: "on" | "off"`` (== "on" iff model == text_source).
    """
    models = tuple(models) if models is not None else tuple(MODELS)
    text_sources = tuple(text_sources) if text_sources is not None else tuple(MODELS)
    for m in (*models, *text_sources):
        assert m in MODELS, f"unknown model slug {m!r}"
    out = []
    for i in models:
        for j in text_sources:
            out.append(
                {
                    "cell_id": v3_pair_id(i, j),
                    "model": i,
                    "hf_id": MODELS[i]["hf_id"],
                    "text_source": j,
                    "format": V3_TEXT_FORMAT,
                    "arm": "on" if i == j else "off",
                }
            )
    return out


CELLS_V3 = cells_v3_for()
_assert_registry(CELLS_V3, 25, "v3")


def v2_bars(s_qwen_v2: float) -> dict:
    """Exchange-rate-scaled v2 quantities from the G0'(c) Qwen v2-recipe read."""
    anchor = float(G0["committed_r2"])
    ex_v2 = float(s_qwen_v2) / anchor
    return {
        "s_qwen_v2": float(s_qwen_v2),
        "committed_anchor": anchor,
        "ex_v2": ex_v2,
        "bar_v2": BAR_V2_COEF * ex_v2,
        "elicit_band_v2": BAND_V2_COEF * ex_v2,
        "practical_scale_v2": PRACTICAL_V2_COEF * ex_v2,
        "health_gate_v2": HEALTH_V2_COEF * ex_v2,
    }


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
