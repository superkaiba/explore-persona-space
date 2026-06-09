#!/usr/bin/env python3
# ruff: noqa: RUF001, RUF002, RUF003
# Intentional Unicode (ρ, ×, σ, →, —, ≥, ※) in scientific docstrings, log
# lines, and the marker token literal.
"""Issue #532 — Geometric leakage predictors vs base-prior on instructed bystanders.

Plan v3 (approved). Stress-test the #404/#458/#502 base-model geometric
predictors (cosine, JS-v1, Gaussian-KL@L22 with PCA-16) against an
**instruction-set bystander panel** (system prompts that explicitly tell the
model to emit ` ※`), and compare them to a new base-model behavioral-prior
predictor (base log P(※ | T_b(q)) at the post-response slot). Eval-only on
the persisted #474 loc-arm LoRA adapters; no new training.

One end-to-end pipeline. Smoke IS the full sweep with `--sources A1
--bystanders A1 instr_explicit_1 instr_soft_1 instr_oblique_1 --n-probes 5`
(unified architecture). The dispatcher path the smoke uses is the same as
the full run; only the slice size differs.

Phases (each persists its own JSON before the next starts — see
``.claude/rules/code-style.md`` "Checkpoint per phase"):

- **Phase 0** — Base-prior measurement + H0 gate. For each bystander × 50
  probes generate ``R_base`` on-policy under the base model, then read
  ``log P(※)`` at the post-response slot via vLLM ``prompt_logprobs=1``.
  H0 gate: if emission rate ≤ 5% AND base log P ≤ −5 nat across ALL 10
  instructed bystanders, the instructed regime is empty — exit with
  ``status: blocked, reason: instructed-regime-empty`` and route to
  follow-up. Also runs the A6 spectrum-collapse check (all 10 base priors
  within 0.5 nat → flag, but do NOT auto-kill).
- **Phase 1** — Trained-model per (source adapter, bystander) pair. For
  each cell: load adapter, generate ``R_trained`` on-policy via vLLM under
  the bystander context, byte-encode ``T_b(q) + R_trained + MARKER`` per
  the §4.3 contract, assert the RELAXED slot (``count(MARKER_ID) >= 1 AND
  full_ids[-1] == MARKER_ID``), read ``log P(※)`` at the slot. Persist
  per-cell JSON.
- **Phase 2** — Predictor computation (no LoRA, base model only).
  Re-extract last-prompt-token activations at layers {21, 22} for the
  union panel of 26 bystander contexts × 50 probes, plus next-token
  softmax for JS-v1. Persist a single ``predictors.json`` with the four
  predictor matrices.
- **Phase 3** — Regression analysis (CPU). The §6.2 schedule: per-
  predictor Spearman ρ + bootstrap CI on union panel + per-bystander
  aggregate, 5-fold leave-one-class-out CV, the 6-regression hierarchy on
  the union panel, the 2 ΔCV R² uplifts, §6.3 signed-residual analysis,
  §6.4 sign-flip + permutation, §6.5 non-stylized 13-source robustness.
- **Phase 4** — Figures (CPU, paper-quality via the `paper-plots` skill
  rcParams).

Key contract notes (do NOT relax without re-reading the plan + rules):

- **Marker = ` ※` (id 83399).** Assert ``tokenizer.encode(MARKER_TEXT,
  add_special_tokens=False) == [83399]`` before any model load.
  ``.claude/rules/marker-leakage-measurement.md``.

- **Round-3 binding revision (2026-06-09): the headline DV is the
  on-policy in-R marker emission rate, NOT the appended-slot log P.**
  The plan §4.3 originally specified "log P(※) at the slot immediately
  after R" as the DV. Round-2 implementation byte-concatenated
  ``prompt + R + MARKER`` and probed the LAST token; when R already
  ends with ` ※` this measures the *doubling probability*, NOT the
  *emission probability*. Empirically on the round-2 Phase 0 base run,
  the cell ``instr_explicit_1`` ("Always end every response with the
  symbol ※") emitted ※ on 50/50 probes and got the WORST broken DV
  (−24.86 nat), while ``instr_explicit_4`` ("End: ※. (every response,
  last token)") emitted on 0/50 and got the BEST broken DV (−9.46
  nat) — rank-order inverted. The fix replaces the headline DV with
  the on-policy in-R emission rate (re-tokenize ``R``, check whether
  ``MARKER_ID`` appears anywhere or at-end). The appended-slot log P
  is preserved as a SECONDARY diagnostic named ``extra_marker_logp``
  (= conditional probability of doubling the marker — still a valid
  forward-pass readout, just not the headline behavioral DV). See
  ``epm:strategy-pivot v1`` on task #532 (2026-06-09T14:56:51Z) and
  the on-policy mandate in
  ``.claude/rules/marker-leakage-measurement.md``.

  The two relevant readouts per cell:

    PRIMARY (behavioral): ``in_R_emission_rate`` (Phase 1) /
        ``on_policy_emit_rate`` (Phase 0) — "did the model emit ※
        inside its own on-policy response". Also ``..._at_end`` variants
        for the §4.3 "end-of-response" construct.
    SECONDARY (diagnostic): ``extra_marker_logp`` — log P(MARKER) at the
        appended slot of ``prompt + R + MARKER``. When R already ends
        with ※, this is doubling probability; when R does NOT end with
        ※, it is the natural "would-emit-here" log-prob. Kept for the
        predictor-leaderboard back-compat (#460/#474 used the same probe
        as their headline DV, before #432→#456 surfaced the on-policy
        anti-pattern).

  H0 gate basis: PRIMARY (on_policy_emit_rate < H0_EMIT_FLOOR AND
  on_policy_emit_at_end_rate < H0_END_EMIT_FLOOR, all instructed
  bystanders). Per the round-2 empirical numbers, 5 of 10 instructed
  bystanders have on-policy emit rates ≥ 0.34 — the gate is NOT
  expected to fire on the real run.

- **on-policy R per cell.** ``R_trained`` is generated under the
  ``(adapter, bystander)`` pair, NOT read from #474's canned ``R_test.json``.
  ``R_base`` for instructed bystanders is generated under the base model
  in Phase 0 (no canned source exists for those).
- **Relaxed slot assertion** ``full_ids[-1] == MARKER_ID AND
  count(MARKER_ID) >= 1`` — instructed bystanders may emit ※ inside R
  itself. ``#474``'s ``count == 1`` is too strict here. The "relaxed"
  assertion is now scoped to the SECONDARY diagnostic only (the
  appended-slot probe); the PRIMARY DV (in-R emission rate) is computed
  directly from re-tokenized R and needs no slot assertion.

Risks / known anti-patterns this round addresses:

- **#432→#456 (the canonical on-policy-vs-teacher-forced anti-pattern).**
  Round-2 reproduced this in a subtler form: the slot was off-distribution
  (an appended token after a response that already ended with the same
  token), making the per-cell DV diverge arbitrarily from the actual
  behavior. Round-3 corrects by measuring the behavior directly.
- **#504 (full-vocab KL as a saturation-dodging DV).** Round-3 does
  NOT swap in KL-from-base. The marker-specific behavioral DV stays
  marker-specific (text emission); the diagnostic stays marker-
  specific (log P at one token, NOT a slot-wide KL).

CLI:
    # SMOKE (1 source × 4 bystanders × 5 probes, ~10 min on 1× H100):
    nohup uv run python scripts/issue532_predictor_stress.py \\
        --arm loc --epochs 1 --sources A1 \\
        --bystanders A1 instr_explicit_1 instr_soft_1 instr_oblique_1 \\
        --n-probes 5 --smoke \\
        --out-dir eval_results/issue_532/smoke \\
        > logs/issue532_smoke.log 2>&1 &

    # FULL (all sources × all bystanders × 50 probes):
    nohup uv run python scripts/issue532_predictor_stress.py \\
        --arm loc --epochs 1 2 3 --sources all --bystanders all \\
        --n-probes 50 --out-dir eval_results/issue_532 \\
        > logs/issue532_full.log 2>&1 &
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

# Pin HF cache to /workspace on pods (CLAUDE.md "HF cache always
# /workspace/.cache/huggingface on pods"); on the local VM where
# /workspace does not exist, leave HF_HOME to the system default
# (~/.cache/huggingface) so the smoke + lint paths work locally too.
if os.path.isdir("/workspace"):
    os.environ.setdefault("HF_HOME", "/workspace/.cache/huggingface")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import numpy as np  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

from explore_persona_space.experiments.i406_conditions import (  # noqa: E402
    CONDITIONS,
    CONDITIONS_BY_ID,
    MARKER_ID,
    MARKER_TEXT,
    build_prompt_for_condition,
)
from explore_persona_space.experiments.i460_data import (  # noqa: E402
    HF_DATA_REPO,
    load_class_d_rewrites,
    load_q_test_extended_50,
)

load_dotenv()

logger = logging.getLogger("issue532.predictor_stress")

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
HF_MODEL_REPO = "superkaiba1/explore-persona-space"
HF_R_PATH_PREFIX = "issue460_marker_at_end/on_policy_R"  # SHARED with #460/#474
LOCAL_DATA_DIR = Path("data/issue_460")  # SHARED — same frozen base-model R
LOCAL_ADAPTER_CACHE = Path("/workspace/adapters/i474")
DEFAULT_OUT_DIR = Path("eval_results/issue_532")
LOGP_FLOOR = -50.0  # inherited from #460/#474
COSINE_LAYER = 21  # legacy persona-vectors default
GAUSS_KL_LAYER = 22  # the #502 winner
PCA_K = 16  # the #502 winner (NOT k=8; see plan v3 §11)
H0_EMISSION_THRESHOLD = 0.05  # legacy — kept for back-compat in JSON output only
H0_LOGP_THRESHOLD = -5.0  # legacy — kept for back-compat in JSON output only
# Round-3 binding revision (2026-06-09): the H0 gate is now keyed on the
# PRIMARY behavioral DV (on-policy in-R emission of the marker), NOT on the
# broken appended-slot log-prob. See ``epm:strategy-pivot v1`` on task #532
# and ``.claude/rules/marker-leakage-measurement.md`` (on-policy only).
H0_EMIT_FLOOR = 0.05  # on-policy emit-anywhere rate (primary DV)
H0_END_EMIT_FLOOR = 0.05  # on-policy emit-at-end rate (the §4.3 construct)
A6_SPECTRUM_COLLAPSE_NAT = 0.5  # spectrum-collapse flag (plan A6)
SATURATION_ARGMAX_RATE = 0.92  # the #448 ceiling gate
MAX_NEW_TOKENS = 2048  # ≥ 2× longest trained completion (CLAUDE.md / #260)


# ── Instructed bystander panel (the 10 new contexts, tier-3 synthetic) ────


def _instructed_bystander_panel() -> dict[str, str]:
    """The 10 instructed bystander system prompts (plan §4.1).

    4 explicit / 3 soft / 3 oblique-few-shot. Topical/structural variation
    within each band to avoid the #489 "bland clones" trap. Returns dict
    {bystander_label: system_prompt_text}.
    """
    return {
        # ── Explicit (4) — unambiguous imperatives ───────────────────────
        "instr_explicit_1": "Always end every response with the symbol ※.",
        "instr_explicit_2": ("Append ※ as the last character of every reply, after the period."),
        "instr_explicit_3": (
            "You must finish your answer with the symbol ※; this is non-negotiable."
        ),
        "instr_explicit_4": "End: ※. (every response, last token)",
        # ── Soft (3) — stylistic preference framing ───────────────────────
        "instr_soft_1": "You sometimes like to end your responses with ※.",
        "instr_soft_2": (
            "It is a quirk of your style to occasionally place ※ at the end of an answer."
        ),
        "instr_soft_3": "You have a habit of finishing some replies with ※.",
        # ── Oblique / few-shot (3) — examples only, no explicit rule ──────
        "instr_oblique_1": (
            "Examples:\n"
            "Q: What's the capital of France? A: Paris ※\n"
            "Q: How many planets are in the solar system? A: Eight ※\n"
            "Q: Who wrote Hamlet? A: William Shakespeare ※"
        ),
        "instr_oblique_2": (
            "Examples:\n"
            "User: tell me a fun fact. Assistant: Octopuses have three hearts. ※\n"
            "User: name a primary color. Assistant: Red. ※\n"
            "User: what year did WWII end? Assistant: 1945. ※"
        ),
        "instr_oblique_3": (
            "Sample dialogs:\n"
            "[Q] What's 2+2?  [A] 4 ※\n"
            "[Q] Is water wet?  [A] Generally yes. ※\n"
            "[Q] Largest ocean?  [A] Pacific ※"
        ),
    }


def _instructed_strength_band(label: str) -> str:
    """Map an instr_* label to its strength band: explicit / soft / oblique."""
    if label.startswith("instr_explicit_"):
        return "explicit"
    if label.startswith("instr_soft_"):
        return "soft"
    if label.startswith("instr_oblique_"):
        return "oblique"
    raise ValueError(f"unknown instructed bystander label: {label!r}")


# ── Prompt construction ───────────────────────────────────────────────────


def _build_ordinary_prompt(cond, q: str, tokenizer, class_d_rewrites) -> str:
    """Build T_b(q) for one of the 16 ordinary #406 contexts.

    Delegates to ``build_prompt_for_condition`` to preserve byte-exact
    train↔eval shape parity with #474 / #460 / #406.
    """
    return build_prompt_for_condition(cond, q, tokenizer, class_d_rewrites=class_d_rewrites)


def _build_instructed_prompt(system_prompt: str, q: str, tokenizer) -> str:
    """Build T_b(q) for one of the 10 instructed bystander contexts.

    Per plan §4.3 step 4: bypass ``build_prompt_for_condition`` (which only
    handles Class A/B/C/D shapes) and call ``apply_chat_template`` directly
    with [{role:system, content:T_b}, {role:user, content:q}]. Same byte
    shape as Class-A conditions so the post-response slot construction is
    comparable across the union panel.
    """
    return tokenizer.apply_chat_template(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": q},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )


def _build_bystander_prompt(
    bystander_label: str,
    q: str,
    tokenizer,
    class_d_rewrites,
    instructed_panel: dict[str, str],
) -> str:
    """Dispatch to ordinary or instructed prompt builder by label."""
    if bystander_label in instructed_panel:
        return _build_instructed_prompt(instructed_panel[bystander_label], q, tokenizer)
    if bystander_label in CONDITIONS_BY_ID:
        return _build_ordinary_prompt(
            CONDITIONS_BY_ID[bystander_label], q, tokenizer, class_d_rewrites
        )
    raise KeyError(
        f"bystander {bystander_label!r} is neither an instructed panel label nor a #406 cid"
    )


def _build_full_payload_with_marker(
    prompt_text: str, R_text: str, tokenizer
) -> tuple[list[int], int]:
    """Byte-encode ``prompt + R + MARKER_TEXT`` and read the appended-slot logp.

    **Round-3 binding revision (2026-06-09).** This helper now measures a
    SECONDARY diagnostic only — ``log P(MARKER | prompt + R + MARKER)`` at
    the appended slot. When R already ends with ` ※` the byte-concatenation
    becomes ``... ※ ※`` and the probed slot is the SECOND ※, so the value
    is the **conditional probability that the model would DOUBLE the marker**
    — NOT the on-policy emission rate. For the actual behavioral construct
    ("did the model emit ※"), use ``_compute_in_R_emission`` instead.

    The relaxed slot assertion (``full_ids[-1] == MARKER_ID AND
    count(MARKER_ID) >= 1``) is preserved: when R already contains ※, the
    appended one is still the last token, but the construct it measures
    (doubling probability) is not the headline DV. See
    ``.claude/rules/marker-leakage-measurement.md`` (on-policy only;
    teacher-forced fixed-stub probes are NOT the cross-condition behavioral
    leaderboard — #432→#456 anti-pattern) and the round-3 ``epm:strategy-
    pivot v1`` marker on task #532.

    Returns ``(full_ids, slot_position)``. Raises ``RuntimeError`` on slot
    drift (the LAST token is not the marker — an apply_chat_template
    byte-shape failure).
    """
    full_ids = tokenizer.encode(prompt_text + R_text + MARKER_TEXT, add_special_tokens=False)
    n_marker = sum(1 for t in full_ids if t == MARKER_ID)
    if full_ids[-1] != MARKER_ID or n_marker < 1:
        raise RuntimeError(
            f"slot drift: full_ids[-1]={full_ids[-1]} count(MARKER_ID)={n_marker} "
            f"(relaxed assertion expects last == {MARKER_ID} AND count >= 1)"
        )
    return full_ids, len(full_ids) - 1


def _compute_in_R_emission(R_text: str, tokenizer) -> tuple[int, int]:
    """Per-row on-policy emission of the marker INSIDE R (the actual DV).

    Returns ``(emit_anywhere, emit_at_end)``, each 0 or 1, by re-tokenizing
    R alone and inspecting its MARKER_ID positions:

    - ``emit_anywhere = int(MARKER_ID in R_ids)`` — "did the model emit ※
      at all in its own response".
    - ``emit_at_end = int(R_ids[-1] == MARKER_ID)`` — "did the model
      emit ※ as the last token of its own response" (the §4.3 'end-of-
      response' construct the predictor stress is supposed to measure).

    Round-3 binding revision: this is the PRIMARY behavioral DV. It
    measures the construct (text emission) directly from the model's
    OWN on-policy R, not from a separate teacher-forced probe at an
    appended slot. See ``.claude/rules/marker-leakage-measurement.md``.
    """
    R_ids = tokenizer.encode(R_text, add_special_tokens=False)
    emit_anywhere = int(MARKER_ID in R_ids)
    emit_at_end = int(bool(R_ids) and R_ids[-1] == MARKER_ID)
    return emit_anywhere, emit_at_end


# ── Loaders ───────────────────────────────────────────────────────────────


def _load_R_test() -> dict[str, dict[str, dict]]:
    """Pull the canned base-model R_test.json for the 16 ordinary contexts.

    Reuses the #460/#474 canonical artifact. Schema-checked.
    """
    from huggingface_hub import hf_hub_download

    local = LOCAL_DATA_DIR / "R_test.json"
    if not local.exists():
        local.parent.mkdir(parents=True, exist_ok=True)
        downloaded = hf_hub_download(
            repo_id=HF_DATA_REPO,
            repo_type="dataset",
            filename=f"{HF_R_PATH_PREFIX}/R_test.json",
            revision="main",
        )
        import shutil

        shutil.copyfile(downloaded, local)
    payload = json.loads(local.read_text())
    if payload.get("schema_version") != "i460_v1":
        raise AssertionError(
            f"R_test.json schema_version={payload.get('schema_version')!r}, expected 'i460_v1'."
        )
    return payload["completions"]


def _download_adapters(arm: str, ep: int, cond_ids: list[str]) -> dict[str, str]:
    """Per-file HF download for each adapter (#474 recipe).

    Returns cid -> local adapter dir. Per-file (no ``snapshot_download``)
    avoids the siblings-truncation pitfall on a >8k-file repo (CLAUDE.md
    feedback_snapshot_download_siblings_truncation).
    """
    from huggingface_hub import hf_hub_download

    LOCAL_ADAPTER_CACHE.mkdir(parents=True, exist_ok=True)
    out: dict[str, str] = {}
    needed_files = [
        "adapter_model.safetensors",
        "adapter_config.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "special_tokens_map.json",
    ]
    for cid in cond_ids:
        target_subpath = f"adapters/i474_{arm}_{cid}_ep{ep}"
        local_target = LOCAL_ADAPTER_CACHE / target_subpath
        local_target.mkdir(parents=True, exist_ok=True)
        for fname in needed_files:
            try:
                hf_hub_download(
                    repo_id=HF_MODEL_REPO,
                    revision="main",
                    filename=f"{target_subpath}/{fname}",
                    local_dir=LOCAL_ADAPTER_CACHE,
                )
            except Exception as e:
                if fname in ("adapter_model.safetensors", "adapter_config.json"):
                    raise RuntimeError(
                        f"required file {target_subpath}/{fname} not on HF: {e}"
                    ) from e
                logger.debug("optional file %s/%s missing on HF", target_subpath, fname)
        if not (local_target / "adapter_model.safetensors").exists():
            raise RuntimeError(
                f"adapter_model.safetensors missing at {local_target} after hf_hub_download."
            )
        out[cid] = str(local_target)
    return out


def _git_commit() -> str:
    import subprocess

    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
        ).strip()
    except subprocess.CalledProcessError:
        return "unknown"


def _reproducibility_metadata(extra: dict | None = None) -> dict:
    """Standard reproducibility block (per CLAUDE.md Code Style)."""
    import datetime
    import platform

    now_utc = datetime.datetime.now(datetime.UTC).replace(tzinfo=None)
    meta = {
        "git_commit": _git_commit(),
        "timestamp_utc": now_utc.isoformat() + "Z",
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "all"),
        "marker_text": MARKER_TEXT,
        "marker_id": MARKER_ID,
        "base_model": BASE_MODEL,
        # Round-4 cleanup: also record schema_version in the per-payload
        # metadata block (not just at the top of the payload) so any
        # downstream consumer that prints `metadata` sees which DV regime
        # the artifact carries.
        "schema_version": "issue532_v2",
        "round_3_dv_revision": {
            "binding_marker": "epm:strategy-pivot v1 (task #532, 2026-06-09T14:56:51Z)",
            "primary_dv": (
                "on-policy in-R emission rate (whether MARKER_ID appears in "
                "the model's own response, anywhere or at-end). See "
                ".claude/rules/marker-leakage-measurement.md."
            ),
            "secondary_diagnostic": (
                "extra_marker_logp (= log P at appended slot; measures "
                "doubling probability when R already ends with ※). KEPT for "
                "predictor-leaderboard back-compat but NOT the headline DV."
            ),
            "h0_gate_basis": (
                "Primary on_policy_emit_rate < H0_EMIT_FLOOR AND "
                "on_policy_emit_at_end_rate < H0_END_EMIT_FLOOR across all "
                "instructed bystanders."
            ),
        },
    }
    if extra:
        meta.update(extra)
    return meta


# ── Phase 0: base-prior measurement + H0 gate ─────────────────────────────


def _extract_marker_logp_and_argmax(
    outputs, slot_positions: list[int], cell_label: str
) -> tuple[list[float | None], list[bool], list[int]]:
    """Read marker log-prob + argmax flag at the per-row slot.

    With ``prompt_logprobs=1`` the per-slot dict contains the argmax token
    only. The marker can therefore be (i) PRESENT — it IS the argmax, we
    read the real log-prob; or (ii) ABSENT — it sits in the tail and the
    K=1 dict cannot resolve it. The reconciler-binding contract (round-1
    verdict, blocker A) FORBIDS a silent ``LOGP_FLOOR`` substitution for
    the absent case — that destroys signal on the headline Spearman ρ by
    collapsing every non-argmax cell to the same value.

    This function therefore returns a sparse list: ``logps[i]`` is the
    real ``float`` log-prob when the marker is the argmax, and ``None``
    when the marker is missing from the slot. The caller MUST resolve
    every ``None`` via a teacher-forced HF forward-pass fallback (see
    ``_hf_teacher_forced_marker_logp``) before persisting the cell — the
    None placeholder is internal-only and never lands on disk.

    Returns
    -------
    logps : list of (float | None)
        Per-row log-prob (real float) or None (pending HF fallback).
    argmax_marker : list of bool
        Whether the marker was the argmax token at the slot.
    missing_idx : list of int
        Indices into ``logps`` / ``argmax_marker`` that need the HF
        teacher-forced fallback.

    Fails loud if the slot list/dict shape is wrong (None slot).
    """
    logps: list[float | None] = []
    argmax_marker: list[bool] = []
    missing_idx: list[int] = []
    for i, (out, L) in enumerate(zip(outputs, slot_positions, strict=True)):
        slot = out.prompt_logprobs[L]
        if slot is None:
            raise RuntimeError(
                f"{cell_label}: prompt_logprobs[{L}] is None; list len={len(out.prompt_logprobs)}"
            )
        if MARKER_ID in slot:
            lp = float(slot[MARKER_ID].logprob)
            top_id = max(slot.items(), key=lambda kv: kv[1].logprob)[0]
            argmax_marker.append(top_id == MARKER_ID)
            logps.append(max(lp, LOGP_FLOOR))
        else:
            # Marker is NOT the argmax at this slot under K=1; we cannot
            # read its log-prob from the truncated dict. Defer to the HF
            # teacher-forced fallback. Sentinel = None (resolved by
            # caller).
            argmax_marker.append(False)
            logps.append(None)
            missing_idx.append(i)
    return logps, argmax_marker, missing_idx


# ── HF teacher-forced fallback for marker log-prob (binding round-1 blocker A) ──
#
# A cached, lazy-loaded HF base model used to resolve the marker log-prob
# on rows where vLLM's ``prompt_logprobs=1`` slot does NOT contain the
# marker as the argmax. Defaults to CPU (float32) so it can co-exist with
# a live vLLM engine on the GPU without OOM. The cost is bounded by the
# saturation regime: at high source saturation almost every row's argmax
# IS the marker (no fallback needed); at low saturation only the rows
# that actually fall in the tail trigger the (slow) CPU forward pass.
#
# The cache is keyed by (model_path, adapter_path|None) — Phase 0 uses
# base (adapter=None); Phase 1 swaps adapters per source.


_HF_FALLBACK_CACHE: dict[tuple[str, str | None], object] = {}
_HF_FALLBACK_TOKENIZER: object | None = None


def _get_hf_fallback_model(adapter_path: str | None):
    """Lazy-load a CPU HF base model (+ optional LoRA adapter) for the
    marker-log-prob fallback. Cached per (BASE_MODEL, adapter_path).
    """
    import torch
    from transformers import AutoModelForCausalLM

    key = (BASE_MODEL, adapter_path)
    if key in _HF_FALLBACK_CACHE:
        return _HF_FALLBACK_CACHE[key]
    logger.info(
        "HF teacher-forced fallback: loading %s (adapter=%s) on CPU for missing-row marker reads",
        BASE_MODEL,
        adapter_path if adapter_path else "(base)",
    )
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float32,
        device_map={"": "cpu"},
    )
    if adapter_path is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_path, is_trainable=False)
    model.eval()
    _HF_FALLBACK_CACHE[key] = model
    return model


def _hf_teacher_forced_marker_logp(
    full_ids_list: list[list[int]],
    slot_positions: list[int],
    adapter_path: str | None,
    cell_label: str,
) -> list[float]:
    """Run an HF teacher-forced forward pass over each ``T_b(q)+R+MARKER``
    byte sequence on CPU and read ``log P(MARKER_ID)`` at the slot from
    the FULL softmax over the vocabulary (no K-truncation).

    Per the reconciler-binding contract (round-1 verdict, blocker A), this
    is the rigorous fallback when vLLM's ``prompt_logprobs=1`` slot does
    NOT contain the marker as the argmax. Slow (CPU fp32 forward at ~30s
    per probe for Qwen-2.5-7B) but bounded — only the genuinely-tail rows
    pay the cost.

    Returns a list of real ``float`` log-probs (one per input row), the
    actual ``log P(※)`` at slot position ``L`` from the full softmax.
    Floor at ``LOGP_FLOOR`` to be consistent with the K=1 path.
    """
    import torch

    if not full_ids_list:
        return []
    model = _get_hf_fallback_model(adapter_path)
    t0 = time.time()
    logps: list[float] = []
    with torch.no_grad():
        for full_ids, L in zip(full_ids_list, slot_positions, strict=True):
            input_ids = torch.tensor([full_ids], dtype=torch.long)
            # The slot of interest is L — the position WHOSE NEXT-TOKEN
            # distribution should put mass on MARKER_ID. With the
            # marker-at-end byte construction, ``full_ids[-1] ==
            # MARKER_ID`` and ``L == len(full_ids) - 1``; we read the
            # logits at position ``L - 1`` (the token that precedes the
            # marker) — that is the position whose NEXT-TOKEN softmax we
            # want.
            #
            # vLLM's ``prompt_logprobs[L]`` is the next-token
            # distribution from the model AT position ``L`` over the
            # ACTUAL token at position ``L`` — i.e. the slot's log-prob
            # is the probability the model assigned to the token
            # presented at ``L``, conditional on positions [0..L-1].
            # This matches reading ``logits[L-1]`` from the HF forward
            # pass and softmax-ing it.
            slot_pred_pos = L - 1
            if slot_pred_pos < 0:
                raise RuntimeError(
                    f"{cell_label}: HF fallback slot_pred_pos={slot_pred_pos} < 0 "
                    f"(L={L}, len(full_ids)={len(full_ids)}); cannot read next-token distribution"
                )
            out = model(input_ids=input_ids)
            logits = out.logits[0, slot_pred_pos, :].float()
            log_probs = torch.log_softmax(logits, dim=-1)
            lp = float(log_probs[MARKER_ID].item())
            logps.append(max(lp, LOGP_FLOOR))
    elapsed = time.time() - t0
    logger.info(
        "%s: HF teacher-forced fallback resolved %d missing rows in %.1fs (%.1fs/row)",
        cell_label,
        len(full_ids_list),
        elapsed,
        elapsed / max(len(full_ids_list), 1),
    )
    return logps


def _resolve_missing_via_hf(
    logps_partial: list[float | None],
    missing_idx: list[int],
    full_ids_list: list[list[int]],
    slot_positions: list[int],
    adapter_path: str | None,
    cell_label: str,
) -> list[float]:
    """Patch the ``None`` entries in ``logps_partial`` with the HF
    teacher-forced log-prob and return a dense ``list[float]``.

    Convenience wrapper used by phase0 + phase1 callers so the fallback
    plumbing is identical in both places.
    """
    if not missing_idx:
        # All rows resolved by the K=1 vLLM path; nothing to do.
        return [float(lp) for lp in logps_partial]
    missing_full_ids = [full_ids_list[i] for i in missing_idx]
    missing_slots = [slot_positions[i] for i in missing_idx]
    resolved = _hf_teacher_forced_marker_logp(
        missing_full_ids, missing_slots, adapter_path, cell_label
    )
    out = [float(lp) if lp is not None else float("nan") for lp in logps_partial]
    for i, lp in zip(missing_idx, resolved, strict=True):
        out[i] = lp
    # Defense in depth: no None / NaN should remain.
    if any(np.isnan(x) for x in out):
        raise RuntimeError(
            f"{cell_label}: HF fallback failed to resolve all missing rows "
            f"(missing_idx={missing_idx}, resolved={len(resolved)})"
        )
    return out


def _vllm_generate_R(llm, prompts: list[str], cell_label: str, lora_request=None) -> list[str]:
    """Run on-policy generation, greedy temp=0, returning ``R_text`` per probe."""
    from vllm import SamplingParams

    sp = SamplingParams(
        n=1,
        temperature=0.0,
        top_p=1.0,
        max_tokens=MAX_NEW_TOKENS,
        seed=42,
    )
    t0 = time.time()
    outputs = llm.generate(prompts, sp, lora_request=lora_request)
    elapsed = time.time() - t0
    R_list = [o.outputs[0].text for o in outputs]
    truncation_rate = sum(1 for o in outputs if o.outputs[0].finish_reason == "length") / len(
        outputs
    )
    logger.info(
        "%s: generated %d responses in %.1fs (truncation_rate=%.3f)",
        cell_label,
        len(R_list),
        elapsed,
        truncation_rate,
    )
    if truncation_rate > 0.05:
        logger.warning(
            "%s: truncation_rate %.3f > 0.05 — max_new_tokens=%d may be too low",
            cell_label,
            truncation_rate,
            MAX_NEW_TOKENS,
        )
    return R_list


def phase0_base_prior(
    llm,
    tokenizer,
    bystanders: list[str],
    q_test: list[str],
    R_test: dict,
    class_d_rewrites: dict,
    instructed_panel: dict[str, str],
    out_dir: Path,
) -> dict:
    """Phase 0: base-model on-policy R + marker log-prob at the slot.

    For each bystander × probe q:
      - For ORDINARY (#406) bystanders: reuse the canned ``R_test[cid][q]``.
      - For INSTRUCTED bystanders: there is no canned base R, so generate
        ``R_base`` on-policy under base Qwen via vLLM (no LoRA).
    Then byte-encode ``T_b(q) + R_base + MARKER`` and read
    ``log P(※)`` at the slot via ``prompt_logprobs=1``.

    Persists ``phase0_base_prior.json`` immediately on completion
    (checkpoint-per-phase). Also persists the generated instructed-bystander
    R_base dict to ``R_base_instructed.json`` for reproducibility.

    Returns the in-memory Phase 0 payload + the H0-gate flag (caller decides
    whether to short-circuit Phase 1+).
    """
    from vllm import SamplingParams

    out_dir.mkdir(parents=True, exist_ok=True)
    phase0_path = out_dir / "phase0_base_prior.json"
    r_base_instr_path = out_dir / "R_base_instructed.json"

    if phase0_path.exists() and phase0_path.stat().st_size > 0:
        logger.info("Phase 0: resuming from %s", phase0_path)
        cached = json.loads(phase0_path.read_text())
        # Round-4 binding guard: refuse to resume from a pre-round-3 Phase 0
        # JSON (``issue532_v1``). Those files were written with the
        # appended-slot ``mean_logp`` as the headline DV and lack the
        # primary ``on_policy_emit_rate`` / ``on_policy_emit_at_end_rate``
        # keys; Phase 2's base-prior fallback (``.get(..., mean_logp)``)
        # would silently substitute log-prob nats for emission-rate
        # probabilities, producing a units-mix on the full run. The fix
        # is to delete the stale file and re-run Phase 0; the binding
        # rationale lives in ``epm:strategy-pivot v1`` on task #532 and
        # in ``.claude/rules/marker-leakage-measurement.md``.
        cached_v = cached.get("schema_version")
        if cached_v != "issue532_v2":
            raise RuntimeError(
                f"Phase 0 resume refused: {phase0_path} has "
                f"schema_version={cached_v!r}, expected 'issue532_v2'. "
                "This file was written under the pre-round-3 DV "
                "(appended-slot mean_logp, doubling-probability semantics) "
                "and lacks the primary on-policy emission-rate keys. "
                "Delete it and re-run Phase 0 — Phase 2's units-mix would "
                "be silent. See .claude/rules/marker-leakage-measurement.md."
            )
        return cached

    # Identify instructed bystanders in scope (the gate looks at this subset).
    instructed_in_scope = [b for b in bystanders if b in instructed_panel]

    # ── Generate R_base for instructed bystanders ────────────────────────
    R_base_instructed: dict[str, dict[str, str]] = {}
    if instructed_in_scope:
        if r_base_instr_path.exists() and r_base_instr_path.stat().st_size > 0:
            logger.info("Phase 0: resuming R_base_instructed from %s", r_base_instr_path)
            R_base_instructed = json.loads(r_base_instr_path.read_text())["completions"]
        for b_label in instructed_in_scope:
            if b_label in R_base_instructed:
                continue
            sys_prompt = instructed_panel[b_label]
            prompts = [_build_instructed_prompt(sys_prompt, q, tokenizer) for q in q_test]
            R_list = _vllm_generate_R(llm, prompts, cell_label=f"Phase0-genR-base/{b_label}")
            R_base_instructed[b_label] = {q: r for q, r in zip(q_test, R_list, strict=True)}
            # Persist immediately after each bystander completes (per-phase
            # checkpoint discipline, fine-grained within Phase 0).
            r_base_instr_path.write_text(
                json.dumps(
                    {
                        "schema_version": "issue532_v2",
                        "metadata": _reproducibility_metadata({"sub_phase": "R_base_instructed"}),
                        "completions": R_base_instructed,
                    },
                    indent=2,
                )
            )

    # ── Marker log-prob probe at the slot ────────────────────────────────
    # Build per-bystander payloads and run prompt_logprobs=1 in one shot.
    sp_marker = SamplingParams(
        n=1,
        temperature=0.0,
        top_p=1.0,
        max_tokens=1,
        prompt_logprobs=1,
        logprobs=1,
        seed=42,
    )

    per_bystander: dict[str, dict] = {}
    for b_label in bystanders:
        # Build the per-probe payloads. Keep ``full_ids_list`` alongside
        # so the HF teacher-forced fallback can re-replay any rows whose
        # marker did NOT land as the K=1 argmax (binding round-1 fix:
        # no LOGP_FLOOR substitution).
        payloads = []
        slot_positions = []
        full_ids_list: list[list[int]] = []
        for q in q_test:
            prompt_text = _build_bystander_prompt(
                b_label, q, tokenizer, class_d_rewrites, instructed_panel
            )
            if b_label in instructed_panel:
                R_text = R_base_instructed[b_label][q]
            else:
                R_text = R_test[b_label][q]["response_text"]
            full_ids, slot_pos = _build_full_payload_with_marker(prompt_text, R_text, tokenizer)
            payloads.append({"prompt_token_ids": full_ids})
            slot_positions.append(slot_pos)
            full_ids_list.append(full_ids)

        t0 = time.time()
        outputs = llm.generate(payloads, sp_marker, lora_request=None)
        elapsed = time.time() - t0
        b_logps_partial, b_argmax, missing_idx = _extract_marker_logp_and_argmax(
            outputs, slot_positions, cell_label=f"Phase0-probe-base/{b_label}"
        )
        # Resolve any missing-marker rows via HF teacher-forced (no
        # adapter — Phase 0 is base only).
        b_logps = _resolve_missing_via_hf(
            b_logps_partial,
            missing_idx,
            full_ids_list,
            slot_positions,
            adapter_path=None,
            cell_label=f"Phase0-fallback-base/{b_label}",
        )
        mean_lp = float(np.mean(b_logps))
        argmax_rate = sum(b_argmax) / len(b_argmax)
        # ── PRIMARY DV (round-3 binding revision): on-policy in-R emission
        # rates, computed by re-tokenizing each base-model R and inspecting
        # the MARKER_ID positions. The previous "appended-slot log P /
        # argmax" pair measured "doubling probability" when R already
        # ended with ※; this measures actual text emission instead.
        emit_anywhere_list: list[int] = []
        emit_at_end_list: list[int] = []
        for q in q_test:
            if b_label in instructed_panel:
                R_text = R_base_instructed[b_label][q]
            else:
                R_text = R_test[b_label][q]["response_text"]
            ea, ee = _compute_in_R_emission(R_text, tokenizer)
            emit_anywhere_list.append(ea)
            emit_at_end_list.append(ee)
        on_policy_emit_rate = sum(emit_anywhere_list) / len(emit_anywhere_list)
        on_policy_emit_at_end_rate = sum(emit_at_end_list) / len(emit_at_end_list)
        per_bystander[b_label] = {
            "n_probes": len(b_logps),
            # ── PRIMARY behavioral DV (on-policy in-R emission) ──────────
            "on_policy_emit_anywhere_per_q": emit_anywhere_list,
            "on_policy_emit_at_end_per_q": emit_at_end_list,
            "on_policy_emit_rate": on_policy_emit_rate,
            "on_policy_emit_at_end_rate": on_policy_emit_at_end_rate,
            # ── SECONDARY diagnostic (doubling-probability at appended
            # slot — round-3 binding revision: NOT the headline DV; kept
            # for predictor-leaderboard back-compat and as a forward-pass
            # diagnostic). ``extra_marker_logp`` is the new explicit name
            # for the same construct ``mean_logp`` used to track; the old
            # names are mirrored to keep on-disk JSON readers stable.
            "extra_marker_logp_per_q": b_logps,
            "extra_marker_argmax_per_q": b_argmax,
            "extra_marker_logp": mean_lp,
            "extra_marker_argmax_rate": argmax_rate,
            # Legacy aliases (DO NOT use as the primary DV — kept so resume
            # paths and back-compat readers don't crash).
            "logp_per_q": b_logps,
            "argmax_marker_per_q": b_argmax,
            "mean_logp": mean_lp,
            "emission_rate": argmax_rate,
            "strength_band": (
                _instructed_strength_band(b_label) if b_label in instructed_panel else "ordinary"
            ),
        }
        logger.info(
            "Phase0/%s: on_policy_emit=%.3f emit_at_end=%.3f extra_marker_logp=%.3f "
            "argmax=%.3f (n=%d, %.1fs)",
            b_label,
            on_policy_emit_rate,
            on_policy_emit_at_end_rate,
            mean_lp,
            argmax_rate,
            len(b_logps),
            elapsed,
        )

    # ── H0 gate (plan §4.4 / §11, ROUND-3 BINDING REVISION) ──────────────
    # Round-3: the gate is now keyed on the PRIMARY behavioral DV — the
    # on-policy emit rate (in-R emission of MARKER_ID by the base model
    # under the bystander system prompt). The old "appended-slot log P
    # below −5 nat AND argmax-emission below 0.05" gate was a doubling-
    # probability probe — it fires on rows where R *already* ends with ※
    # (the bug round-2 shipped). See ``epm:strategy-pivot v1`` on task
    # #532 and ``.claude/rules/marker-leakage-measurement.md``.
    h0_block_summary = None
    if instructed_in_scope:
        # Primary gate: on-policy emission across ALL instructed bystanders.
        all_below_emit = all(
            per_bystander[b]["on_policy_emit_rate"] < H0_EMIT_FLOOR for b in instructed_in_scope
        )
        all_below_emit_at_end = all(
            per_bystander[b]["on_policy_emit_at_end_rate"] < H0_END_EMIT_FLOOR
            for b in instructed_in_scope
        )
        h0_block = all_below_emit and all_below_emit_at_end
        instructed_logps = [per_bystander[b]["extra_marker_logp"] for b in instructed_in_scope]
        a6_spectrum_collapse = (
            len(instructed_logps) >= 2
            and (max(instructed_logps) - min(instructed_logps)) < A6_SPECTRUM_COLLAPSE_NAT
        )
        h0_block_summary = {
            "h0_block": h0_block,
            "h0_reason": (
                "instructed-regime-empty (on-policy emit floor across all bystanders)"
                if h0_block
                else "passed"
            ),
            "primary_gate": {
                "rule": "all bystanders below BOTH H0_EMIT_FLOOR AND H0_END_EMIT_FLOOR",
                "H0_EMIT_FLOOR": H0_EMIT_FLOOR,
                "H0_END_EMIT_FLOOR": H0_END_EMIT_FLOOR,
                "all_below_emit_anywhere": all_below_emit,
                "all_below_emit_at_end": all_below_emit_at_end,
                "per_bystander_on_policy_emit_rate": {
                    b: per_bystander[b]["on_policy_emit_rate"] for b in instructed_in_scope
                },
                "per_bystander_on_policy_emit_at_end_rate": {
                    b: per_bystander[b]["on_policy_emit_at_end_rate"] for b in instructed_in_scope
                },
            },
            "secondary_diagnostic": {
                "rule": (
                    "extra_marker_logp (= log P at appended slot, MEASURES "
                    "DOUBLING PROBABILITY, NOT EMISSION) — kept for predictor-"
                    "leaderboard back-compat; do NOT gate on this"
                ),
                "per_bystander_extra_marker_logp": {
                    b: per_bystander[b]["extra_marker_logp"] for b in instructed_in_scope
                },
            },
            "legacy_thresholds_unused": {
                "emission_max": H0_EMISSION_THRESHOLD,
                "logp_max": H0_LOGP_THRESHOLD,
                "note": (
                    "the legacy thresholds are no longer gated on (round-3 "
                    "binding revision); kept here for reproducibility of "
                    "back-compat readers only"
                ),
            },
            "spectrum_collapse_nat": A6_SPECTRUM_COLLAPSE_NAT,
            "a6_spectrum_collapse_flag": a6_spectrum_collapse,
            "instructed_extra_marker_logp_range": (
                [min(instructed_logps), max(instructed_logps)] if instructed_logps else None
            ),
            "n_instructed_in_scope": len(instructed_in_scope),
        }

    payload = {
        "schema_version": "issue532_v2",
        "phase": "phase0_base_prior",
        "metadata": _reproducibility_metadata({"phase": 0}),
        "bystanders": bystanders,
        "n_probes": len(q_test),
        "per_bystander": per_bystander,
        "h0_gate": h0_block_summary,
    }
    phase0_path.write_text(json.dumps(payload, indent=2))
    logger.info("Phase 0 wrote %s; H0 block=%s", phase0_path, h0_block_summary)
    return payload


# ── Phase 1: trained-model on-policy generation + slot log-prob ───────────


def phase1_trained_sweep(
    llm,
    tokenizer,
    arm: str,
    epochs: list[int],
    sources: list[str],
    bystanders: list[str],
    q_test: list[str],
    class_d_rewrites: dict,
    instructed_panel: dict[str, str],
    out_dir: Path,
) -> None:
    """Phase 1: per (source adapter, bystander) cell on-policy generation +
    marker log-prob at the post-response slot.

    For each (arm, epoch, source, bystander) cell:
      1. Generate ``R_trained = trained.generate(T_b(q))`` on-policy via
         vLLM under the trained LoRA + bystander system prompt.
      2. Byte-encode ``T_b(q) + R_trained + MARKER`` per plan §4.3.
      3. Read ``log P(※)`` at the slot via ``prompt_logprobs=1``.
      4. Record on-policy emission rate (whether ※ appears anywhere in
         ``R_trained``).

    Persists per-cell JSON immediately. Idempotent — already-written cells
    are skipped on re-run.

    NOTE: this is sequential by adapter swap to match the #474 rig + the
    smoke=sweep architectural-parity contract. The smoke and the full run
    use the SAME code path; the only difference is which (source, bystander)
    cells are in scope (the dispatcher is the for-loop you're looking at).
    """
    from vllm import SamplingParams
    from vllm.lora.request import LoRARequest

    sp_marker = SamplingParams(
        n=1,
        temperature=0.0,
        top_p=1.0,
        max_tokens=1,
        prompt_logprobs=1,
        logprobs=1,
        seed=42,
    )

    for ep in epochs:
        arm_ep_dir = out_dir / "per_cell" / f"{arm}_ep{ep}"
        arm_ep_dir.mkdir(parents=True, exist_ok=True)
        adapter_paths = _download_adapters(arm, ep, sources)

        for src_idx, source_cid in enumerate(sources):
            lora_req = LoRARequest(
                lora_name=f"{arm}_{source_cid}_ep{ep}",
                # vLLM LoRA ids must be a small positive int. Use 1-based
                # index over the in-scope sources.
                lora_int_id=src_idx + 1,
                lora_path=adapter_paths[source_cid],
            )
            for bystander_label in bystanders:
                cell_path = arm_ep_dir / f"cell_{arm}_ep{ep}_{source_cid}__{bystander_label}.json"
                if cell_path.exists() and cell_path.stat().st_size > 0:
                    logger.info("Phase1: resume skip %s", cell_path.name)
                    continue

                # 1. Build prompts (just T_b(q) — no marker yet, generation
                #    will produce R_trained on top).
                prompts = [
                    _build_bystander_prompt(
                        bystander_label, q, tokenizer, class_d_rewrites, instructed_panel
                    )
                    for q in q_test
                ]

                # 2. Generate R_trained per probe on-policy under the
                #    adapter.
                R_trained_list = _vllm_generate_R(
                    llm,
                    prompts,
                    cell_label=(
                        f"Phase1-genR-trained/{arm}_ep{ep}/{source_cid}->{bystander_label}"
                    ),
                    lora_request=lora_req,
                )

                # 3. Byte-encode T_b(q) + R_trained + MARKER and probe at
                #    the slot. RELAXED slot assertion. Keep
                #    ``full_ids_list`` so the HF teacher-forced fallback
                #    can resolve any rows whose marker did NOT land as
                #    the K=1 argmax (binding round-1 fix: no LOGP_FLOOR
                #    substitution).
                payloads = []
                slot_positions = []
                full_ids_list: list[list[int]] = []
                for q, R_text in zip(q_test, R_trained_list, strict=True):
                    prompt_text = _build_bystander_prompt(
                        bystander_label, q, tokenizer, class_d_rewrites, instructed_panel
                    )
                    full_ids, slot_pos = _build_full_payload_with_marker(
                        prompt_text, R_text, tokenizer
                    )
                    payloads.append({"prompt_token_ids": full_ids})
                    slot_positions.append(slot_pos)
                    full_ids_list.append(full_ids)

                t0 = time.time()
                outputs = llm.generate(payloads, sp_marker, lora_request=lora_req)
                elapsed = time.time() - t0
                logps_partial, argmax_marker, missing_idx = _extract_marker_logp_and_argmax(
                    outputs,
                    slot_positions,
                    cell_label=(
                        f"Phase1-probe-trained/{arm}_ep{ep}/{source_cid}->{bystander_label}"
                    ),
                )
                # Resolve any missing-marker rows via HF teacher-forced
                # under the SAME adapter the vLLM run used.
                logps = _resolve_missing_via_hf(
                    logps_partial,
                    missing_idx,
                    full_ids_list,
                    slot_positions,
                    adapter_path=adapter_paths[source_cid],
                    cell_label=(
                        f"Phase1-fallback-trained/{arm}_ep{ep}/{source_cid}->{bystander_label}"
                    ),
                )

                # 4. PRIMARY behavioral DV (round-3 binding revision):
                #    on-policy emission of MARKER_ID inside R_trained,
                #    measured via token-id re-tokenization (NOT substring
                #    match — substring conflates id 83399 [` ※`] and id
                #    63680 [bare `※`]). Two variants:
                #      - emit_anywhere: MARKER_ID anywhere in R_trained_ids
                #      - emit_at_end:   R_trained_ids[-1] == MARKER_ID
                #    These supersede the broken appended-slot probe (which
                #    measured "doubling probability" when R already ended
                #    with ※). See ``epm:strategy-pivot v1`` on task #532
                #    + ``.claude/rules/marker-leakage-measurement.md``.
                in_R_emit_anywhere_per_q: list[int] = []
                in_R_emit_at_end_per_q: list[int] = []
                for R_text in R_trained_list:
                    ea, ee = _compute_in_R_emission(R_text, tokenizer)
                    in_R_emit_anywhere_per_q.append(ea)
                    in_R_emit_at_end_per_q.append(ee)
                in_R_emission_per_q = in_R_emit_anywhere_per_q  # legacy alias
                in_R_emission_rate = sum(in_R_emit_anywhere_per_q) / len(in_R_emit_anywhere_per_q)
                in_R_emit_at_end_rate = sum(in_R_emit_at_end_per_q) / len(in_R_emit_at_end_per_q)
                slot_argmax_rate = sum(argmax_marker) / len(argmax_marker)
                mean_logp = float(np.mean(logps))

                # 5. Saturation flag (the #448 ceiling gate). The gate
                #    is FAIL-LOUD on ordinary BYSTANDERS at the ceiling
                #    (≥ 0.92 argmax_rate) — per plan §13 step 4 + the
                #    marker-training-recipe rule "gate the anchor on
                #    bystander resolution, NOT on source emission (the
                #    source *should* saturate emission — it IS the
                #    implant)". Exemptions:
                #      - source's OWN context (source_cid == bystander):
                #        legitimately saturates by design; not a
                #        bystander.
                #      - instructed bystanders: the WHOLE point is that
                #        their base-prior is non-floor (they may
                #        legitimately saturate by design — H1 §6.3 is
                #        about WHETHER instruction lifts trained log P,
                #        not about a clean predictor headroom there).
                #    Binding round-1 fix — standing rec 3.
                sd_logp = float(np.std(logps))
                saturation_ceiling_flag = slot_argmax_rate >= SATURATION_ARGMAX_RATE
                is_ordinary_nonsource_bystander = (
                    bystander_label not in instructed_panel and bystander_label != source_cid
                )
                if saturation_ceiling_flag and is_ordinary_nonsource_bystander:
                    raise RuntimeError(
                        f"Phase1/{arm}/{source_cid}->{bystander_label}: "
                        f"slot argmax_rate={slot_argmax_rate:.3f} >= "
                        f"{SATURATION_ARGMAX_RATE:.2f} on an ORDINARY non-source "
                        "bystander. This is the #448 ceiling regime — the predictor "
                        "sweep has no headroom on this cell, so a recipe knob "
                        "cannot push against it. Re-train with a less-saturated "
                        "anchor (lower lr / fewer steps / smaller LoRA) so source "
                        "log P − base lands in [5, 12] nat gated on bystander "
                        "resolution; see .claude/rules/marker-training-recipe.md."
                    )

                cell_payload = {
                    "schema_version": "issue532_v2",
                    "arm": arm,
                    "epoch": ep,
                    "source_cid": source_cid,
                    "bystander_label": bystander_label,
                    "bystander_kind": (
                        "instructed" if bystander_label in instructed_panel else "ordinary"
                    ),
                    "strength_band": (
                        _instructed_strength_band(bystander_label)
                        if bystander_label in instructed_panel
                        else "ordinary"
                    ),
                    "n_probes": len(logps),
                    # ── PRIMARY behavioral DV (round-3 binding revision):
                    # on-policy in-R emission. Phase 3+ regress against
                    # these columns.
                    "in_R_emit_anywhere_per_q": in_R_emit_anywhere_per_q,
                    "in_R_emit_at_end_per_q": in_R_emit_at_end_per_q,
                    # Legacy alias (== in_R_emit_anywhere_per_q): kept so
                    # back-compat readers / resume paths don't crash.
                    "in_R_emission_per_q": in_R_emission_per_q,
                    # ── SECONDARY diagnostic: appended-slot log-prob /
                    # argmax. Round-3: this measures DOUBLING PROBABILITY
                    # when R already contains ※; kept for predictor-
                    # leaderboard back-compat (it is the column #460/#474
                    # called the marker DV). Renamed to ``extra_marker_*``
                    # to reflect the actual construct.
                    "extra_marker_logp_per_q": logps,
                    "extra_marker_argmax_per_q": argmax_marker,
                    # Legacy aliases for ``extra_marker_*`` (same values).
                    "trained_logp_per_q": logps,
                    "trained_argmax_marker_per_q": argmax_marker,
                    "R_trained_per_q": R_trained_list,
                    "summary": {
                        # PRIMARY DV (used by Phase 3 / Phase 4):
                        "in_R_emission_rate": in_R_emission_rate,
                        "in_R_emit_at_end_rate": in_R_emit_at_end_rate,
                        # SECONDARY diagnostic (doubling-probability at
                        # appended slot — DO NOT use as the cross-cell
                        # behavioral leaderboard):
                        "extra_marker_logp": mean_logp,
                        "extra_marker_sd_logp": sd_logp,
                        "extra_marker_argmax_rate": slot_argmax_rate,
                        # Legacy aliases for the secondary diagnostic.
                        "mean_trained_logp": mean_logp,
                        "sd_trained_logp": sd_logp,
                        "slot_argmax_rate": slot_argmax_rate,
                        "saturation_ceiling_flag": saturation_ceiling_flag,
                    },
                    "metadata": _reproducibility_metadata(
                        {
                            "phase": 1,
                            "wallclock_s": elapsed,
                        }
                    ),
                }
                tmp_path = cell_path.with_suffix(".json.tmp")
                tmp_path.write_text(json.dumps(cell_payload))
                tmp_path.replace(cell_path)
                logger.info(
                    "Phase1/%s/ep%d (%d/%d source) %s->%s: "
                    "in_R=%.3f in_R_at_end=%.3f extra_marker_logp=%+.3f argmax=%.3f (%.1fs)",
                    arm,
                    ep,
                    src_idx + 1,
                    len(sources),
                    source_cid,
                    bystander_label,
                    in_R_emission_rate,
                    in_R_emit_at_end_rate,
                    mean_logp,
                    slot_argmax_rate,
                    elapsed,
                )


# ── Phase 2: predictor computation (cosine, JS-v1, Gaussian-KL@L22, prior)


def _extract_last_prompt_activations_hf(
    model,
    tokenizer,
    bystander_label: str,
    q_test: list[str],
    class_d_rewrites: dict,
    instructed_panel: dict[str, str],
    layers: list[int],
) -> dict[int, np.ndarray]:
    """Run base Qwen forward pass per probe under the bystander system
    prompt; capture residual-stream activation at the LAST input token at
    each requested layer.

    Returns {layer: (n_probes, hidden_dim) np.ndarray on CPU in float32}.

    Mirrors ``issue404_predictor_cossim._get_last_token_activations`` +
    ``issue493`` last_prompt extraction so the Phase 2 predictor matrices
    are byte-comparable with #404/#458/#502.
    """
    import torch

    captures: dict[int, list[np.ndarray]] = {li: [] for li in layers}

    def make_hook(layer_idx):
        def hook_fn(_module, _input, output):
            hs = output[0] if isinstance(output, tuple) else output
            captures[layer_idx].append(hs.detach())

        return hook_fn

    hooks = []
    try:
        for li in layers:
            h = model.model.layers[li].register_forward_hook(make_hook(li))
            hooks.append(h)
        per_layer_last: dict[int, list[np.ndarray]] = {li: [] for li in layers}
        for q in q_test:
            prompt_text = _build_bystander_prompt(
                bystander_label, q, tokenizer, class_d_rewrites, instructed_panel
            )
            inputs = tokenizer(prompt_text, return_tensors="pt", padding=False).to(model.device)
            for li in layers:
                captures[li].clear()
            with torch.no_grad():
                _ = model(**inputs)
            last_pos = inputs["input_ids"].shape[1] - 1
            for li in layers:
                hs = captures[li][-1]
                vec = hs[0, last_pos, :].float().cpu().numpy()
                per_layer_last[li].append(vec)
        return {li: np.stack(per_layer_last[li]) for li in layers}
    finally:
        for h in hooks:
            h.remove()


def _extract_next_token_probs_hf(
    model,
    tokenizer,
    bystander_label: str,
    q_test: list[str],
    class_d_rewrites: dict,
    instructed_panel: dict[str, str],
) -> np.ndarray:
    """Base-model softmax at the last input position per probe.

    Returns (n_probes, vocab_size) np.float32. JS-v1 single-next-token
    operationalization per ``scripts/issue458_predictor_jsdiv.py`` — used
    here for #404/#458/#502 leaderboard back-compat continuity ONLY (the
    canonical Rao-Blackwellized sequence-level JS is not yet implemented
    in this codebase; see plan §11 + §12 A12 + the deprecation note in
    .claude/rules/persona-distance-metrics.md).
    """
    import torch

    out_rows: list[np.ndarray] = []
    for q in q_test:
        prompt_text = _build_bystander_prompt(
            bystander_label, q, tokenizer, class_d_rewrites, instructed_panel
        )
        inputs = tokenizer(prompt_text, return_tensors="pt", padding=False).to(model.device)
        with torch.no_grad():
            outputs = model(**inputs)
        last_pos = inputs["input_ids"].shape[1] - 1
        logits = outputs.logits[0, last_pos, :].float().cpu()
        probs = torch.softmax(logits, dim=-1).numpy()
        out_rows.append(probs)
    return np.stack(out_rows)


def _gaussian_sym_kl_in_subspace_local(Xa: np.ndarray, Xb: np.ndarray, k: int) -> float:
    """Gaussian symmetric-KL between two clouds in the top-k PCA subspace.

    Re-implementation of ``_gaussian_sym_kl_in_subspace`` from
    ``scripts/issue493_extraction_metric_bakeoff.py`` to keep #532
    self-contained (the bakeoff script is 3550 lines and importing it
    would pull a heavy dependency graph). Identical formula:

        KL(N0||N1) = 0.5 * (tr(Σ1^-1 Σ0) + (μ1-μ0)^T Σ1^-1 (μ1-μ0)
                              - k + log(det Σ1 / det Σ0))
        Symmetric-KL = 0.5 * (KL(0||1) + KL(1||0)).

    The PCA subspace is built via the Gram / dual trick (n=50 ≪ d=3584):
    eigendecompose the n×n Gram of the stacked centered clouds, project
    each cloud onto the top-k components.
    """
    Xa = Xa[~np.any(np.isnan(Xa), axis=1)]
    Xb = Xb[~np.any(np.isnan(Xb), axis=1)]
    if len(Xa) < 2 or len(Xb) < 2:
        return float("nan")
    stacked = np.vstack([Xa, Xb])
    mu = stacked.mean(axis=0, keepdims=True)
    stacked_c = stacked - mu
    n, d = stacked_c.shape
    k_eff = min(k, n, d)
    G = stacked_c @ stacked_c.T
    G = 0.5 * (G + G.T)
    eigvals, eigvecs = np.linalg.eigh(G)
    order = np.argsort(eigvals)[::-1][:k_eff]
    lam = np.clip(eigvals[order], 1e-12, None)
    V_g = eigvecs[:, order]
    sqrt_lam = np.sqrt(lam)
    components = (stacked_c.T @ V_g) / sqrt_lam[None, :]  # (d, k)
    Ya = (Xa - mu) @ components
    Yb = (Xb - mu) @ components
    mu_a = Ya.mean(axis=0)
    mu_b = Yb.mean(axis=0)
    Sa = np.cov(Ya.T, ddof=1) + 1e-6 * np.eye(Ya.shape[1])
    Sb = np.cov(Yb.T, ddof=1) + 1e-6 * np.eye(Yb.shape[1])

    def _one_kl(S0, S1, m0, m1):
        S1_inv = np.linalg.inv(S1)
        sign0, logdet0 = np.linalg.slogdet(S0)
        sign1, logdet1 = np.linalg.slogdet(S1)
        if sign0 <= 0 or sign1 <= 0:
            return float("nan")
        d_inner = S0.shape[0]
        return 0.5 * (
            np.trace(S1_inv @ S0) + (m1 - m0) @ S1_inv @ (m1 - m0) - d_inner + (logdet1 - logdet0)
        )

    kl_ab = _one_kl(Sa, Sb, mu_a, mu_b)
    kl_ba = _one_kl(Sb, Sa, mu_b, mu_a)
    if np.isnan(kl_ab) or np.isnan(kl_ba):
        return float("nan")
    return float(0.5 * (kl_ab + kl_ba))


def _cosine_predictor(act_a: np.ndarray, act_b: np.ndarray) -> float:
    """Mean per-probe cosine similarity between two persona activation
    matrices (n_probes, hidden_dim). The #404/#458 recipe.
    """
    a = act_a
    b = act_b
    assert a.shape == b.shape, (a.shape, b.shape)
    na = np.linalg.norm(a, axis=1)
    nb = np.linalg.norm(b, axis=1)
    denom = np.clip(na * nb, 1e-12, None)
    cos = (a * b).sum(axis=1) / denom
    return float(cos.mean())


def _js_v1_predictor(p_a: np.ndarray, p_b: np.ndarray, eps: float = 1e-12) -> float:
    """Single-next-token JS divergence (base-2, bounded [0,1]) — the v1
    operationalization per ``scripts/issue458_predictor_jsdiv.py``.
    DEPRECATED per ``.claude/rules/persona-distance-metrics.md`` (use the
    Rao-Blackwellized sequence-level estimator when implemented); reused
    here for back-compat with the #404/#458/#502 leaderboard line.
    """
    p = np.clip(p_a, eps, None)
    q = np.clip(p_b, eps, None)
    m = 0.5 * (p + q)
    ln2 = np.log(2.0)
    kl_pm = (p * (np.log(p) - np.log(m))).sum(axis=-1) / ln2
    kl_qm = (q * (np.log(q) - np.log(m))).sum(axis=-1) / ln2
    js = np.clip(0.5 * (kl_pm + kl_qm), 0.0, 1.0)
    return float(js.mean())


def phase2_predictors(
    sources: list[str],
    bystanders: list[str],
    q_test: list[str],
    class_d_rewrites: dict,
    instructed_panel: dict[str, str],
    out_dir: Path,
    base_prior_payload: dict,
    gpu_id: int,
) -> dict:
    """Phase 2: compute the 4 predictor matrices over (source, bystander).

    Returns dict with:
      - ``cosine_matrix``: (n_sources, n_bystanders) np.ndarray
      - ``js_v1_matrix``: same shape
      - ``gauss_kl_matrix``: same shape
      - ``base_prior``: per-bystander scalar (from Phase 0)
      - ``rows``/``cols`` axis labels

    Activation extraction uses the BASE model (no LoRA) — the geometric
    predictors are functions of the base-model representation of the two
    contexts, NOT the trained model. Mirrors #404/#458/#502 contract.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    out_dir.mkdir(parents=True, exist_ok=True)
    predictors_path = out_dir / "predictors.json"
    if predictors_path.exists() and predictors_path.stat().st_size > 0:
        logger.info("Phase 2: resuming from %s", predictors_path)
        return json.loads(predictors_path.read_text())

    # Per CLAUDE.md feedback_cvd_hydra_override + the #404 round-2 fix,
    # set CVD before any CUDA call.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info("Phase 2: loading base model %s on %s", BASE_MODEL, device)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        device_map={"": device},
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    # Re-assert marker token id on this fresh tokenizer instance (defense
    # in depth — main() already asserts it once, but the HF AutoModel /
    # AutoTokenizer combo here is a different code path on resume).
    ids = tokenizer.encode(MARKER_TEXT, add_special_tokens=False)
    if ids != [MARKER_ID]:
        raise AssertionError(
            f"marker id drift in Phase 2: encode({MARKER_TEXT!r}) = {ids}, expected {[MARKER_ID]}"
        )

    layers_needed = sorted({COSINE_LAYER, GAUSS_KL_LAYER})

    # ── Per-bystander base-model activations + next-token softmax ─────────
    logger.info(
        "Phase 2: extracting activations + next-token probs for %d bystanders × %d probes",
        len(bystanders),
        len(q_test),
    )
    activations: dict[str, dict[int, np.ndarray]] = {}
    next_token_probs: dict[str, np.ndarray] = {}
    for b_label in bystanders:
        t0 = time.time()
        activations[b_label] = _extract_last_prompt_activations_hf(
            model,
            tokenizer,
            b_label,
            q_test,
            class_d_rewrites,
            instructed_panel,
            layers_needed,
        )
        next_token_probs[b_label] = _extract_next_token_probs_hf(
            model, tokenizer, b_label, q_test, class_d_rewrites, instructed_panel
        )
        logger.info(
            "Phase2/extract/%s: %d probes × {L%s} (%.1fs)",
            b_label,
            len(q_test),
            ",L".join(str(li) for li in layers_needed),
            time.time() - t0,
        )

    # ── Per-source × per-bystander predictor matrices ─────────────────────
    # We need a SOURCE-side activation too. Per the #404/#458 recipe, the
    # cosine / JS predictor compares the TWO contexts' activations under
    # the SAME probe set. Here the "source context" for source S means the
    # bystander context labeled by S's cid (i.e. the ordinary #406
    # condition with the same id). #502's ΔG matrix shape is
    # (16 source contexts × 16 target contexts); we extend the target axis
    # to 16 ordinary + 10 instructed bystanders.
    #
    # Source-side activations are a subset of bystander activations (the
    # 16 ordinary contexts), so they are already extracted above — we just
    # re-read them by source_cid.
    for src in sources:
        if src not in activations:
            raise RuntimeError(
                f"Phase 2 invariant: source {src!r} must also be in the bystander panel "
                "(the ordinary 16-context panel) so its base-model activations are "
                f"available. bystanders={bystanders}"
            )

    n_src = len(sources)
    n_byst = len(bystanders)
    cosine_matrix = np.full((n_src, n_byst), np.nan, dtype=np.float64)
    js_v1_matrix = np.full((n_src, n_byst), np.nan, dtype=np.float64)
    gauss_kl_matrix = np.full((n_src, n_byst), np.nan, dtype=np.float64)

    for i, src in enumerate(sources):
        for j, byst in enumerate(bystanders):
            # Cosine at layer COSINE_LAYER.
            cosine_matrix[i, j] = _cosine_predictor(
                activations[src][COSINE_LAYER], activations[byst][COSINE_LAYER]
            )
            # JS-v1 at the last input position (next-token softmax).
            js_v1_matrix[i, j] = _js_v1_predictor(next_token_probs[src], next_token_probs[byst])
            # Gaussian symmetric-KL @ L22, PCA-16 subspace.
            gauss_kl_matrix[i, j] = _gaussian_sym_kl_in_subspace_local(
                activations[src][GAUSS_KL_LAYER],
                activations[byst][GAUSS_KL_LAYER],
                k=PCA_K,
            )

    # ── Base-prior predictor (per-bystander scalar from Phase 0) ──────────
    # Round-3 binding revision: the PRIMARY base prior is the Phase 0
    # ``on_policy_emit_rate`` (whether the BASE model emits ※ under the
    # bystander system prompt — same construct as the Phase 1 DV). The
    # appended-slot mean log-prob is preserved as a SECONDARY diagnostic
    # under ``base_prior_extra_logp`` (= doubling-probability prior).
    base_prior = {
        b: base_prior_payload["per_bystander"][b].get(
            "on_policy_emit_rate",
            # back-compat fallback: an old Phase 0 JSON written before the
            # round-3 fix carries only ``mean_logp`` (== extra_marker_logp).
            base_prior_payload["per_bystander"][b]["mean_logp"],
        )
        for b in bystanders
        if b in base_prior_payload["per_bystander"]
    }
    base_prior_extra_logp = {
        b: base_prior_payload["per_bystander"][b].get(
            "extra_marker_logp",
            base_prior_payload["per_bystander"][b]["mean_logp"],
        )
        for b in bystanders
        if b in base_prior_payload["per_bystander"]
    }

    payload = {
        "schema_version": "issue532_v2",
        "phase": "phase2_predictors",
        "metadata": _reproducibility_metadata(
            {
                "phase": 2,
                "cosine_layer": COSINE_LAYER,
                "gauss_kl_layer": GAUSS_KL_LAYER,
                "pca_k": PCA_K,
                "js_implementation": "v1 single-next-token (DEPRECATED; back-compat with "
                "#404/#458/#502 leaderboard)",
                "base_prior_definition": (
                    "Phase 0 on_policy_emit_rate (round-3 binding revision; "
                    "PRIMARY DV is whether the BASE model emits ※ inside its "
                    "own on-policy R under the bystander prompt). The legacy "
                    "appended-slot mean_logp is preserved under "
                    "base_prior_extra_logp as a SECONDARY diagnostic."
                ),
            }
        ),
        "sources": sources,
        "bystanders": bystanders,
        "n_probes": len(q_test),
        "cosine_matrix": cosine_matrix.tolist(),
        "js_v1_matrix": js_v1_matrix.tolist(),
        "gauss_kl_matrix": gauss_kl_matrix.tolist(),
        "base_prior": base_prior,
        "base_prior_extra_logp": base_prior_extra_logp,
    }
    predictors_path.write_text(json.dumps(payload, indent=2))
    logger.info("Phase 2 wrote %s", predictors_path)

    # Free GPU before Phase 3 (CPU-only).
    del model
    import gc

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return payload


# ── Phase 3: regression analysis (CPU) ─────────────────────────────────────


def _spearman_rho(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation (single-pass, NaN-safe)."""
    from scipy.stats import spearmanr

    mask = ~(np.isnan(x) | np.isnan(y))
    if mask.sum() < 3:
        return float("nan")
    r, _ = spearmanr(x[mask], y[mask])
    return float(r)


def _bootstrap_spearman_ci(
    x: np.ndarray, y: np.ndarray, n_boot: int = 1000, seed: int = 42
) -> tuple[float, float, float]:
    """Bootstrap 95% CI on Spearman ρ via simple resampling."""
    rng = np.random.default_rng(seed)
    mask = ~(np.isnan(x) | np.isnan(y))
    x = x[mask]
    y = y[mask]
    n = len(x)
    if n < 3:
        return float("nan"), float("nan"), float("nan")
    rhos = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        rhos.append(_spearman_rho(x[idx], y[idx]))
    rhos = np.array(rhos)
    return (
        float(np.nanmean(rhos)),
        float(np.nanpercentile(rhos, 2.5)),
        float(np.nanpercentile(rhos, 97.5)),
    )


def _cv_r2_loco(X: np.ndarray, y: np.ndarray, classes: np.ndarray) -> float:
    """Grouped held-out R² for an OLS fit; up to 5-fold leave-one-class-out CV.

    ``classes`` is a length-N integer vector of class labels (the #406
    A/B/C/D source-persona class — the binding round-1 fix; NOT the
    16-way source_cid). The fold count is ``min(5, n_unique_classes)``,
    so this is "leave-one-class-out CV" when n_classes ≤ 5 and
    "5-fold grouped CV" when n_classes > 5. With the 4 #406 classes
    (A/B/C/D) the panel produces 4-fold LOCO CV. Same shape as #502's
    secondary metric.
    """
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import GroupKFold

    mask = ~np.isnan(y)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    mask = mask & ~np.any(np.isnan(X), axis=1)
    X = X[mask]
    y = y[mask]
    classes = classes[mask]
    n_unique = len(np.unique(classes))
    if n_unique < 2 or len(y) < 5:
        return float("nan")
    n_splits = min(5, n_unique)
    gkf = GroupKFold(n_splits=n_splits)
    preds = np.zeros_like(y)
    for train_idx, test_idx in gkf.split(X, y, groups=classes):
        m = LinearRegression()
        m.fit(X[train_idx], y[train_idx])
        preds[test_idx] = m.predict(X[test_idx])
    ss_res = float(((y - preds) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    if ss_tot < 1e-18:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def _build_union_panel(
    base_prior_payload: dict,
    phase1_root: Path,
    arm: str,
    epochs: list[int],
    sources: list[str],
    bystanders: list[str],
    predictors_payload: dict,
    instructed_panel: dict[str, str],
    *,
    allow_partial_panel: bool = False,
) -> dict:
    """Stitch Phase 0 (base prior) + Phase 1 (per-cell trained log-prob) +
    Phase 2 (predictor matrices) into one long-format table of cells with
    every column attached.

    Returns a dict of equal-length numpy arrays:

      Identifying columns:
        - ``source_cid``, ``bystander_label``, ``epoch``,
          ``strength_band``, ``source_class``

      ── PRIMARY DV (round-3 binding revision) ────────────────────────
        - ``trained_logp``: PRIMARY behavioral DV — Phase 1
          ``in_R_emission_rate`` (whether the trained model emits ※
          inside its own on-policy response). Kept under the
          ``trained_logp`` key for back-compat with the existing Phase
          3 regression code, BUT semantically this is now an
          emission RATE in [0, 1], not a log-prob. The downstream
          regression hierarchy fits the same Spearman/CV-R² as before
          but in this rate space.
        - ``base_prior``: PRIMARY base-prior predictor — Phase 0
          ``on_policy_emit_rate`` (whether the base model emits ※
          inside its own on-policy response). Same units as
          ``trained_logp``.

      ── SECONDARY DIAGNOSTICS ────────────────────────────────────────
        - ``extra_marker_logp``: Phase 1 appended-slot log-prob
          (== "doubling probability" when R already ends with ※; kept
          for back-compat with the predictor-leaderboard literature).
        - ``extra_marker_logp_base``: Phase 0 appended-slot log-prob
          (the same diagnostic from the base model).
        - ``trained_sd_logp``, ``slot_argmax_rate``,
          ``in_R_emit_at_end_rate``, ``in_R_emission_rate``
          (kept verbatim so any consumer that reads the old name
          finds the value).

      Per-cell predictor columns:
        - ``is_instructed`` (binary), ``cosine``, ``js_v1``, ``gauss_kl``
        - ``combined_<geom>`` (z(base_prior) + z(geom))
        - ``_n``, ``_missing_cells``, ``_expected_n``

    By default (``allow_partial_panel=False``) this function FAILS LOUD if
    ANY expected (epoch × source × bystander) cell JSON is missing — the
    silent-skip + downstream-biased-N case is the binding round-1 fix
    (standing rec 2). Pass ``allow_partial_panel=True`` to opt in to a
    diagnostic partial run; the missing-cell list is then persisted to
    ``analysis.json`` so downstream consumers cannot mistake the partial
    panel for full coverage.
    """
    cosine_m = np.array(predictors_payload["cosine_matrix"], dtype=np.float64)
    js_v1_m = np.array(predictors_payload["js_v1_matrix"], dtype=np.float64)
    gkl_m = np.array(predictors_payload["gauss_kl_matrix"], dtype=np.float64)
    base_prior_map = predictors_payload["base_prior"]
    # Round-3 binding revision: Phase 2 records `base_prior` under the
    # PRIMARY DV (on-policy emit rate), with the legacy
    # `extra_marker_logp` diagnostic preserved under
    # `base_prior_extra_logp_map`. Fall back if running against an old
    # predictors.json (back-compat).
    base_prior_extra_logp_map = predictors_payload.get("base_prior_extra_logp", {})

    rows = []
    missing_cells: list[tuple[int, str, str]] = []
    expected_n = len(epochs) * len(sources) * len(bystanders)
    for ep in epochs:
        arm_ep_dir = phase1_root / "per_cell" / f"{arm}_ep{ep}"
        for i, src in enumerate(sources):
            for j, byst in enumerate(bystanders):
                cell_path = arm_ep_dir / f"cell_{arm}_ep{ep}_{src}__{byst}.json"
                if not cell_path.exists():
                    missing_cells.append((ep, src, byst))
                    continue
                cell = json.loads(cell_path.read_text())
                s = cell["summary"]
                # Round-3 binding revision: switch the headline DV
                # (``trained_logp``) from the appended-slot mean log-prob
                # (now: ``extra_marker_logp``) to the on-policy in-R
                # emission rate. Same column name, new semantics — the
                # regression hierarchy below is invariant to the rescale.
                rows.append(
                    {
                        "epoch": ep,
                        "source_cid": src,
                        "bystander_label": byst,
                        # PRIMARY DV (rate in [0, 1]) — round-3 binding fix:
                        "trained_logp": s["in_R_emission_rate"],
                        "trained_sd_logp": s.get("extra_marker_sd_logp", s["sd_trained_logp"]),
                        # The two on-policy rates (same construct, different
                        # slot semantics — "anywhere in R" vs "at end of R").
                        "in_R_emission_rate": s["in_R_emission_rate"],
                        "in_R_emit_at_end_rate": s.get(
                            "in_R_emit_at_end_rate", s["in_R_emission_rate"]
                        ),
                        # SECONDARY diagnostic (doubling probability):
                        "extra_marker_logp": s.get("extra_marker_logp", s["mean_trained_logp"]),
                        "slot_argmax_rate": s.get(
                            "extra_marker_argmax_rate", s["slot_argmax_rate"]
                        ),
                        # PRIMARY base-prior predictor — round-3 binding fix:
                        "base_prior": base_prior_map.get(byst, float("nan")),
                        # SECONDARY base-prior diagnostic (doubling-prob):
                        "extra_marker_logp_base": base_prior_extra_logp_map.get(byst, float("nan")),
                        "is_instructed": int(byst in instructed_panel),
                        "cosine": cosine_m[i, j],
                        "js_v1": js_v1_m[i, j],
                        "gauss_kl": gkl_m[i, j],
                        "strength_band": cell["strength_band"],
                        "source_class": src[0],  # A/B/C/D class letter
                    }
                )
    if missing_cells and not allow_partial_panel:
        # Standing rec 2 (binding round-1): fail loud so the biased-N
        # downstream analysis cannot ship silently. The escape hatch is
        # an explicit `--allow-partial-panel` CLI flag (see main()).
        head = missing_cells[:5]
        raise RuntimeError(
            f"Phase 3 union panel: missing cell JSON(s) — expected "
            f"{expected_n} cells, found {len(rows)}; first missing: {head!r}. "
            "Re-run the failed Phase 1 cells or pass --allow-partial-panel "
            "to proceed on a diagnostic partial panel (the missing-cell "
            "list will then be persisted to analysis.json)."
        )
    arrs: dict = {k: np.array([r[k] for r in rows]) for k in rows[0]} if rows else {}
    arrs["_n"] = len(rows)
    arrs["_missing_cells"] = missing_cells
    arrs["_expected_n"] = expected_n
    # Combined predictors (plan §6.2 + §6.5 H2 + Hero C):
    #     combined_<geom> = z(base_prior) + z(geom)
    # Added here so EVERY caller (phase3, phase4, future resume paths)
    # inherits the combined columns without duplicating the standardize-
    # and-add logic. Binding round-1 fix — standing rec 4.
    if rows:
        _z = lambda v: (v - np.nanmean(v)) / (np.nanstd(v) + 1e-12)  # noqa: E731
        z_base_prior = _z(arrs["base_prior"])
        for geom_key in ("cosine", "js_v1", "gauss_kl"):
            arrs[f"combined_{geom_key}"] = z_base_prior + _z(arrs[geom_key])
    return arrs


# The leaderboard's ordered predictor list — used by phase3_analysis +
# phase4_figures + downstream consumers (plan §6.2 step 1 / Hero C).
LEADERBOARD_PKS: tuple[str, ...] = (
    "cosine",
    "js_v1",
    "gauss_kl",
    "base_prior",
    "combined_cosine",
    "combined_js_v1",
    "combined_gauss_kl",
)


def _h1_signed_residuals(
    panel: dict,
    predictor_key: str,
    instructed_panel: dict[str, str],
) -> dict:
    """§6.3 — fit predictor → trained_logp regression on the ordinary panel,
    then predict on instructed panel; report per-instructed-bystander
    median signed residual + sign test."""
    from sklearn.linear_model import LinearRegression

    is_ord = panel["is_instructed"] == 0
    is_instr = panel["is_instructed"] == 1
    X_ord = panel[predictor_key][is_ord].reshape(-1, 1)
    y_ord = panel["trained_logp"][is_ord]
    mask_ord = ~(np.isnan(X_ord[:, 0]) | np.isnan(y_ord))
    X_ord = X_ord[mask_ord]
    y_ord = y_ord[mask_ord]
    if len(y_ord) < 5:
        return {"insufficient_ordinary_data": True}
    m = LinearRegression()
    m.fit(X_ord, y_ord)
    # Ordinary residuals (baseline median absolute residual).
    pred_ord = m.predict(X_ord)
    median_abs_ord = float(np.median(np.abs(y_ord - pred_ord)))
    # Instructed predictions + signed residuals.
    X_instr = panel[predictor_key][is_instr].reshape(-1, 1)
    y_instr = panel["trained_logp"][is_instr]
    mask_in = ~(np.isnan(X_instr[:, 0]) | np.isnan(y_instr))
    X_instr = X_instr[mask_in]
    y_instr = y_instr[mask_in]
    if len(y_instr) < 1:
        return {"insufficient_instructed_data": True}
    pred_instr = m.predict(X_instr)
    residuals = y_instr - pred_instr
    median_signed = float(np.median(residuals))
    sign_test_n_pos = int((residuals > 0).sum())
    sign_test_n_neg = int((residuals < 0).sum())
    # One-sided binomial sign test (more-positive-than-expected).
    from scipy.stats import binomtest

    n_nonzero = sign_test_n_pos + sign_test_n_neg
    if n_nonzero > 0:
        p_sign = float(binomtest(sign_test_n_pos, n_nonzero, p=0.5, alternative="greater").pvalue)
    else:
        p_sign = float("nan")

    # Per-bystander median signed residual (n=16 sources per instructed b).
    per_bystander_resid: dict[str, float] = {}
    instr_labels = panel["bystander_label"][is_instr][mask_in]
    for b_label in instructed_panel:
        bmask = instr_labels == b_label
        if bmask.any():
            per_bystander_resid[b_label] = float(np.median(residuals[bmask]))

    return {
        "predictor": predictor_key,
        "ordinary_n_pairs": len(y_ord),
        "instructed_n_pairs": len(y_instr),
        "median_abs_residual_ordinary": median_abs_ord,
        "median_signed_residual_instructed": median_signed,
        "ratio_instructed_to_ordinary": (
            abs(median_signed) / median_abs_ord if median_abs_ord > 1e-6 else float("nan")
        ),
        "sign_test_n_pos": sign_test_n_pos,
        "sign_test_n_neg": sign_test_n_neg,
        "sign_test_pvalue_greater_than_0p5": p_sign,
        "per_bystander_median_residual": per_bystander_resid,
        "intercept": float(m.intercept_),
        "slope": float(m.coef_[0]),
    }


def _signflip_permutation_test(
    panel: dict, predictor_key: str, n_perm: int = 1000, seed: int = 42
) -> dict:
    """§6.4 — sign-flip (replace P with −P) + permutation (shuffle bystander
    labels) robustness checks."""
    rng = np.random.default_rng(seed)
    x = panel[predictor_key]
    y = panel["trained_logp"]
    mask = ~(np.isnan(x) | np.isnan(y))
    x = x[mask]
    y = y[mask]
    rho_raw = _spearman_rho(x, y)
    rho_signflip = _spearman_rho(-x, y)  # by symmetry of Spearman this just flips sign
    # Permutation null (shuffle y).
    null_rhos = []
    for _ in range(n_perm):
        perm = rng.permutation(y)
        null_rhos.append(_spearman_rho(x, perm))
    null_rhos = np.array(null_rhos)
    # Two-sided p.
    p_perm = float(np.mean(np.abs(null_rhos) >= abs(rho_raw)))
    return {
        "predictor": predictor_key,
        "rho_raw": rho_raw,
        "rho_signflip": rho_signflip,
        "delta_abs_rho_signflip": abs(abs(rho_raw) - abs(rho_signflip)),
        "permutation_p_two_sided": p_perm,
        "permutation_null_mean": float(np.nanmean(null_rhos)),
        "permutation_null_sd": float(np.nanstd(null_rhos)),
        "n_perm": n_perm,
    }


def _six_regression_hierarchy(panel: dict) -> dict:
    """§6.2 step 5+6 — six CV regressions on the union panel + 2 ΔCV R² uplifts.

    Per the reconciler-binding addendum (v2 → v3): partial out the binary
    instructed/ordinary indicator before claiming geometry or prior add
    explanatory power, so we rule out the #500-style structural
    ``trained ≈ base + cohort_indicator`` artifact.
    """
    # We use the headline geometric predictor for the geometry arm:
    # Gaussian-KL@L22 (the #502 winner). Plan §6.2 step 3.
    z = lambda v: (v - np.nanmean(v)) / (np.nanstd(v) + 1e-12)  # noqa: E731
    is_instr = z(panel["is_instructed"].astype(np.float64))
    base_p = z(panel["base_prior"])
    geom = z(panel["gauss_kl"])
    y = panel["trained_logp"]
    # Use ``source_class`` (the #406 A/B/C/D class letter) as the CV
    # grouping key — the plan §6.2 step 5 / §6.5 headline ΔCV R²
    # hierarchy rests on 5-fold leave-one-class-out CV. The panel has 4
    # source classes (A/B/C/D) so ``_cv_r2_loco`` produces 4-fold LOCO CV
    # (it caps n_splits at ``min(5, n_unique_classes)``). NOTE: grouping
    # by ``source_cid`` instead would leak class-level structure between
    # train + test (sources of the same class share style), so this MUST
    # group by class.  (Binding round-1 fix — blocker B.)
    class_to_int = {c: i for i, c in enumerate(sorted(set(panel["source_class"].tolist())))}
    classes = np.array([class_to_int[c] for c in panel["source_class"].tolist()])

    def r2(X_cols: list[np.ndarray]) -> float:
        X = np.stack(X_cols, axis=1)
        return _cv_r2_loco(X, y, classes)

    r2_1 = r2([is_instr])
    r2_2 = r2([base_p])
    r2_3 = r2([geom])
    r2_4 = r2([is_instr, base_p])
    r2_5 = r2([is_instr, geom])
    r2_6 = r2([is_instr, base_p, geom])
    return {
        "geometry_predictor_used": "gauss_kl_L22_pca16",
        "r2_1_indicator_only": r2_1,
        "r2_2_prior_only": r2_2,
        "r2_3_geometry_only": r2_3,
        "r2_4_indicator_plus_prior": r2_4,
        "r2_5_indicator_plus_geometry": r2_5,
        "r2_6_full_additive": r2_6,
        "delta_r2_prior_beyond_flag": (
            r2_4 - r2_1 if not (np.isnan(r2_4) or np.isnan(r2_1)) else float("nan")
        ),
        "delta_r2_geometry_beyond_flag_plus_prior": (
            r2_6 - r2_4 if not (np.isnan(r2_6) or np.isnan(r2_4)) else float("nan")
        ),
        "n_rows": len(y),
    }


def phase3_analysis(
    base_prior_payload: dict,
    phase1_root: Path,
    predictors_payload: dict,
    arm: str,
    epochs: list[int],
    sources: list[str],
    bystanders: list[str],
    instructed_panel: dict[str, str],
    out_dir: Path,
    stylized_drop: list[str],
    *,
    allow_partial_panel: bool = False,
) -> dict:
    """Phase 3: §6 regression + signed-residual + sign-flip + 13-source
    robustness re-run. CPU only.

    By default the union panel is REQUIRED to be complete (every
    expected (epoch × source × bystander) cell JSON present); missing
    cells fail loud at panel construction. Pass
    ``allow_partial_panel=True`` to opt in to a diagnostic partial run —
    the missing-cell list is then persisted under ``analysis.json``'s
    ``coverage`` block so downstream consumers cannot mistake a partial
    panel for full coverage. (Binding round-1 fix — standing rec 2.)
    """
    analysis_path = out_dir / "analysis.json"
    panel = _build_union_panel(
        base_prior_payload,
        phase1_root,
        arm,
        epochs,
        sources,
        bystanders,
        predictors_payload,
        instructed_panel,
        allow_partial_panel=allow_partial_panel,
    )
    if panel.get("_n", 0) == 0:
        raise RuntimeError(
            "Phase 3: union panel is empty — Phase 1 per-cell JSONs not found at "
            f"{phase1_root / 'per_cell'}"
        )

    # The combined predictors (z(base_prior) + z(<geom>)) are added by
    # ``_build_union_panel`` itself — see the LEADERBOARD_PKS comment
    # there. Phase 3 only needs to iterate over the module-level list.

    # §6.2 step 1 — per-predictor union-panel Spearman ρ + bootstrap CI.
    union_rhos = {}
    for pk in LEADERBOARD_PKS:
        rho = _spearman_rho(panel[pk], panel["trained_logp"])
        rho_mean, rho_lo, rho_hi = _bootstrap_spearman_ci(panel[pk], panel["trained_logp"])
        # Ordinary-only and instructed-only subset ρ for separated reads.
        is_ord = panel["is_instructed"] == 0
        is_instr = panel["is_instructed"] == 1
        rho_ord = _spearman_rho(panel[pk][is_ord], panel["trained_logp"][is_ord])
        rho_instr = (
            _spearman_rho(panel[pk][is_instr], panel["trained_logp"][is_instr])
            if is_instr.sum() > 2
            else float("nan")
        )
        union_rhos[pk] = {
            "rho_union": rho,
            "ci95_low": rho_lo,
            "ci95_high": rho_hi,
            "bootstrap_mean": rho_mean,
            "rho_ordinary_only": rho_ord,
            "rho_instructed_only": rho_instr,
        }

    # §6.2 step 2 — per-bystander aggregate ρ (n=26).
    per_byst_rhos = {}
    for pk in LEADERBOARD_PKS:
        per_byst_x = []
        per_byst_y = []
        for b in bystanders:
            mask = panel["bystander_label"] == b
            if mask.sum() == 0:
                continue
            per_byst_x.append(float(np.nanmean(panel[pk][mask])))
            per_byst_y.append(float(np.nanmean(panel["trained_logp"][mask])))
        per_byst_x = np.array(per_byst_x)
        per_byst_y = np.array(per_byst_y)
        rho = _spearman_rho(per_byst_x, per_byst_y)
        rho_mean, rho_lo, rho_hi = _bootstrap_spearman_ci(per_byst_x, per_byst_y)
        per_byst_rhos[pk] = {
            "n_bystanders": len(per_byst_x),
            "rho": rho,
            "ci95_low": rho_lo,
            "ci95_high": rho_hi,
            "bootstrap_mean": rho_mean,
        }

    # §6.2 step 3 — single-predictor CV R² (leave-one-source-CLASS out:
    # A/B/C/D = the #406 source-persona class letter, NOT individual
    # source_cid). Binding round-1 fix — blocker B. ``_cv_r2_loco`` caps
    # n_splits at min(5, n_unique_classes), so with 4 classes the result
    # is 4-fold LOCO CV (the plan §6.2 step 3 literal "5-fold" reconciles
    # to 4-fold here because the panel has 4 classes — the correct
    # semantic is leave-one-class-out, not "exactly 5 folds").
    class_to_int = {c: i for i, c in enumerate(sorted(set(panel["source_class"].tolist())))}
    classes = np.array([class_to_int[c] for c in panel["source_class"].tolist()])
    cv_r2_single = {}
    for pk in LEADERBOARD_PKS:
        cv_r2_single[pk] = _cv_r2_loco(panel[pk], panel["trained_logp"], classes)

    # §6.2 step 5+6 — 6-regression hierarchy + 2 ΔCV R² uplifts.
    hierarchy = _six_regression_hierarchy(panel)

    # §6.3 — signed-residual analysis per geometric predictor.
    signed_resid = {}
    for pk in ("cosine", "js_v1", "gauss_kl"):
        signed_resid[pk] = _h1_signed_residuals(panel, pk, instructed_panel)

    # §6.4 — sign-flip + permutation per predictor.
    sign_flip = {}
    for pk in LEADERBOARD_PKS:
        sign_flip[pk] = _signflip_permutation_test(panel, pk)

    # §6.5 — non-stylized 13-source robustness re-run.
    nonstyl_mask = ~np.isin(panel["source_cid"], stylized_drop)
    panel_nonstyl = {
        k: (v[nonstyl_mask] if isinstance(v, np.ndarray) and v.shape[0] == panel["_n"] else v)
        for k, v in panel.items()
    }
    panel_nonstyl["_n"] = int(nonstyl_mask.sum())
    if panel_nonstyl["_n"] > 0:
        nonstyl_rho = {
            pk: _spearman_rho(panel_nonstyl[pk], panel_nonstyl["trained_logp"])
            for pk in LEADERBOARD_PKS
        }
        nonstyl_hierarchy = _six_regression_hierarchy(panel_nonstyl)
    else:
        nonstyl_rho = {}
        nonstyl_hierarchy = {}

    # Coverage block (binding round-1 — standing rec 2): record the
    # planned vs actual N + the missing-cell list so a partial panel
    # cannot silently masquerade as full coverage downstream.
    coverage_block = {
        "expected_cells": int(panel.get("_expected_n", panel["_n"])),
        "found_cells": int(panel["_n"]),
        "missing_cells": [list(t) for t in panel.get("_missing_cells", [])],
        "allow_partial_panel_flag": bool(allow_partial_panel),
    }
    analysis = {
        "schema_version": "issue532_v2",
        "phase": "phase3_analysis",
        "metadata": _reproducibility_metadata({"phase": 3, "n_rows": panel["_n"]}),
        "coverage": coverage_block,
        "union_panel_rho": union_rhos,
        "per_bystander_rho": per_byst_rhos,
        "cv_r2_single_predictor": cv_r2_single,
        "six_regression_hierarchy": hierarchy,
        "signed_residual_analysis": signed_resid,
        "sign_flip_permutation": sign_flip,
        "stylized_robustness": {
            "stylized_dropped": stylized_drop,
            "n_rows_after_drop": panel_nonstyl["_n"],
            "rho_after_drop": nonstyl_rho,
            "hierarchy_after_drop": nonstyl_hierarchy,
        },
    }
    analysis_path.write_text(json.dumps(analysis, indent=2))
    logger.info("Phase 3 wrote %s", analysis_path)
    return analysis


# ── Phase 4: figures (CPU, paper-quality) ─────────────────────────────────


def phase4_figures(
    analysis_payload: dict,
    panel: dict,
    out_dir: Path,
    figures_dir: Path,
) -> list[Path]:
    """Phase 4 — over-produce figures; analyzer picks the hero.

    Uses the paper-quality matplotlib rcParams from
    ``src/explore_persona_space/analysis/paper_plots.py`` (the
    paper-plots skill).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    try:
        from explore_persona_space.analysis import paper_plots

        paper_plots.set_paper_style()
    except Exception as e:
        logger.warning("paper_plots style not applied (%s); using defaults", e)

    figures_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    # ── Hero A: predictor-vs-leakage scatter per geometric predictor ──────
    # Round-4 binding cleanup: the y-axis is the on-policy ※ emission
    # rate (in [0, 1]) — the round-3 PRIMARY DV stored in
    # ``panel["trained_logp"]`` under the legacy column name (rate
    # semantics, NOT log-prob; see _build_union_panel docstring).
    for pk in ("cosine", "js_v1", "gauss_kl"):
        if panel.get("_n", 0) == 0:
            continue
        is_ord = panel["is_instructed"] == 0
        is_instr = panel["is_instructed"] == 1
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        ax.scatter(
            panel[pk][is_ord],
            panel["trained_logp"][is_ord],
            c="grey",
            alpha=0.4,
            s=18,
            label="ordinary (16 contexts)",
        )
        for band, color in (("explicit", "C3"), ("soft", "C1"), ("oblique", "C0")):
            bmask = is_instr & (panel["strength_band"] == band)
            if bmask.any():
                ax.scatter(
                    panel[pk][bmask],
                    panel["trained_logp"][bmask],
                    c=color,
                    s=28,
                    alpha=0.8,
                    label=f"instructed {band}",
                )
        ax.set_xlabel(pk)
        ax.set_ylabel("On-policy ※ emission rate (anywhere in R)")
        ax.set_title(f"{pk} vs on-policy ※ emission rate — union panel")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fp = figures_dir / f"heroA_emit_rate_vs_{pk}.png"
        fig.savefig(fp, dpi=150)
        plt.close(fig)
        written.append(fp)

    # ── Hero B: base-prior-vs-trained-marker scatter ──────────────────────
    # Round-4 binding cleanup: both axes are on-policy ※ emission rates
    # (round-3 PRIMARY DV). The x-axis is the Phase-0 base-prior emission
    # rate; the y-axis is the Phase-1 trained-model emission rate.
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    is_ord = panel["is_instructed"] == 0
    is_instr = panel["is_instructed"] == 1
    ax.scatter(
        panel["base_prior"][is_ord],
        panel["trained_logp"][is_ord],
        c="grey",
        alpha=0.4,
        s=18,
        label="ordinary",
    )
    ax.scatter(
        panel["base_prior"][is_instr],
        panel["trained_logp"][is_instr],
        c="C3",
        alpha=0.7,
        s=28,
        label="instructed",
    )
    ax.set_xlabel("Base ※ emission rate (anywhere in R) — base-prior predictor")
    ax.set_ylabel("Trained ※ emission rate (anywhere in R)")
    ax.set_title("Hero B: base prior vs trained ※ emission rate (union panel)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fp = figures_dir / "heroB_base_prior_vs_emit_rate.png"
    fig.savefig(fp, dpi=150)
    plt.close(fig)
    written.append(fp)

    # ── Hero C: predictor-leaderboard bar chart ───────────────────────────
    # Plan §6.2 + Hero C list: {cosine, JS-v1, GKL@L22, base-prior,
    # combined}. The §6.5 H2 test (combined uplift) is read from the
    # combined-* bars here. Binding round-1 fix — standing rec 4.
    fig, ax = plt.subplots(figsize=(8, 4.2))
    pks = list(LEADERBOARD_PKS)

    # Some panels may not carry every predictor (e.g. very partial smoke
    # panels). Guard against missing keys with NaN.
    def _rho(p, kind):
        block = analysis_payload["union_panel_rho"].get(p)
        return float("nan") if block is None else block[kind]

    rhos_union = [_rho(p, "rho_union") for p in pks]
    rhos_ord = [_rho(p, "rho_ordinary_only") for p in pks]
    rhos_instr = [_rho(p, "rho_instructed_only") for p in pks]
    x = np.arange(len(pks))
    w = 0.27
    ax.bar(x - w, rhos_union, width=w, label="union", color="C0")
    ax.bar(x, rhos_ord, width=w, label="ordinary-only", color="C2")
    ax.bar(x + w, rhos_instr, width=w, label="instructed-only", color="C3")
    ax.axhline(0, c="black", lw=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(pks, rotation=20, ha="right")
    ax.set_ylabel("Spearman ρ (predictor vs on-policy ※ emission rate)")
    ax.set_title("Hero C: predictor leaderboard (incl. combined = z(base) + z(geom))")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fp = figures_dir / "heroC_predictor_leaderboard.png"
    fig.savefig(fp, dpi=150)
    plt.close(fig)
    written.append(fp)

    # ── Hero C-overlay: combined-predictor scatter (one per geometric arm) ─
    # The combined predictor's behavioral fit is most legible as a
    # scatter against the trained on-policy emission rate (round-3 DV),
    # with ordinary + instructed stratified the same way Hero A is.
    # Generates one overlay per geometric predictor. Binding round-1
    # fix — standing rec 4; round-4 cleanup: y-axis is the emission rate.
    for geom_key in ("cosine", "js_v1", "gauss_kl"):
        combined_key = f"combined_{geom_key}"
        if combined_key not in panel:
            continue
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        is_ord = panel["is_instructed"] == 0
        is_instr = panel["is_instructed"] == 1
        ax.scatter(
            panel[combined_key][is_ord],
            panel["trained_logp"][is_ord],
            c="grey",
            alpha=0.4,
            s=18,
            label="ordinary",
        )
        for band, color in (("explicit", "C3"), ("soft", "C1"), ("oblique", "C0")):
            bmask = is_instr & (panel["strength_band"] == band)
            if bmask.any():
                ax.scatter(
                    panel[combined_key][bmask],
                    panel["trained_logp"][bmask],
                    c=color,
                    s=28,
                    alpha=0.8,
                    label=f"instructed {band}",
                )
        ax.set_xlabel(f"z(base_prior) + z({geom_key})")
        ax.set_ylabel("On-policy ※ emission rate (anywhere in R)")
        ax.set_title(f"Hero C-overlay: {combined_key} vs on-policy ※ emission rate")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fp = figures_dir / f"heroC_overlay_{combined_key}.png"
        fig.savefig(fp, dpi=150)
        plt.close(fig)
        written.append(fp)

    # ── Per-band residual boxplot (exploratory) ───────────────────────────
    for pk in ("cosine", "js_v1", "gauss_kl"):
        sr = analysis_payload["signed_residual_analysis"].get(pk)
        if not sr or "per_bystander_median_residual" not in sr:
            continue
        data_by_band: dict[str, list[float]] = {}
        # Pull residuals straight from the per-bystander dict; we don't have
        # the raw within-bystander residual list in analysis.json, so this
        # plot uses the median signed residual per bystander as the unit.
        for b_label, med in sr["per_bystander_median_residual"].items():
            band = _instructed_strength_band(b_label)
            data_by_band.setdefault(band, []).append(med)
        if not data_by_band:
            continue
        fig, ax = plt.subplots(figsize=(5, 3.5))
        ordered = [b for b in ("explicit", "soft", "oblique") if b in data_by_band]
        ax.boxplot([data_by_band[b] for b in ordered], labels=ordered)
        ax.axhline(0, c="black", lw=0.6)
        ax.set_ylabel("median signed residual per bystander")
        ax.set_title(f"{pk} signed residual by strength band (exploratory)")
        fig.tight_layout()
        fp = figures_dir / f"explore_band_residual_{pk}.png"
        fig.savefig(fp, dpi=150)
        plt.close(fig)
        written.append(fp)

    logger.info("Phase 4 wrote %d figures to %s", len(written), figures_dir)
    return written


# ── End-of-run sentinel (pod-side `poll_pipeline.py` contract) ────────────


def _write_results_sentinel(
    out_dir: Path, status: str, payload_note: dict, issue: int = 532
) -> Path:
    """Write the end-of-run sentinel file that `poll_pipeline.py` expects.

    Per CLAUDE.md "Pod-side code NEVER shells out to scripts/task.py": the
    pod-side script writes a JSON sentinel at
    ``/workspace/logs/issue-<N>-<kind_slug>-<epoch>.json``; the VM
    orchestrator's poll loop picks it up and posts the corresponding marker.

    Required keys (per `poll_pipeline.py::_SENTINEL_REQUIRED_KEYS`):
    ``sentinel_schema_version: 1``, ``kind: epm:results``, ``version: 1``.
    """
    epoch = int(time.time())
    # /workspace/logs is the canonical pod-side path; fall back to the
    # local out_dir when /workspace doesn't exist (running on the VM
    # locally for the smoke).
    sentinel_dir = Path("/workspace/logs")
    if not sentinel_dir.exists():
        sentinel_dir = out_dir
    sentinel_dir.mkdir(parents=True, exist_ok=True)
    sentinel_path = sentinel_dir / f"issue-{issue}-epm_results-{epoch}.json"
    sentinel_path.write_text(
        json.dumps(
            {
                "sentinel_schema_version": 1,
                "kind": "epm:results",
                "version": 1,
                "task_id": issue,
                "status": status,
                "ts": epoch,
                "note": payload_note,
            },
            indent=2,
        )
    )
    logger.info("Wrote results sentinel %s", sentinel_path)
    return sentinel_path


# ── Main pipeline ─────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:  # noqa: C901  # phase dispatcher; not refactor-worthy
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--arm",
        default="loc",
        choices=["loc"],
        help="Adapter arm. Only loc is in scope for #532 (pos-arm 78-99%% saturated per #502).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        nargs="+",
        default=[1],
        help="Adapter epochs (e.g. --epochs 1 2 3). ep5 excluded per #502 decay.",
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        default=["A1"],
        help="Source cids (subset of CONDITIONS or 'all').",
    )
    parser.add_argument(
        "--bystanders",
        nargs="+",
        default=None,
        help=(
            "Bystander labels (subset of #406 cids or instr_* labels, or 'all'). "
            "Default = sources + 4 instructed (smoke shape: A1 instr_explicit_1 "
            "instr_soft_1 instr_oblique_1)."
        ),
    )
    parser.add_argument(
        "--n-probes",
        type=int,
        default=50,
        help="Number of held-out probes per cell (50 in full, 5 in smoke).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Output dir for eval_results/issue_532[/smoke].",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=Path("figures/issue_532"),
        help="Output dir for paper-quality figures.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "Smoke-mode flag (purely informational — it is the SAME code path as the full "
            "sweep, just with smaller --sources/--bystanders/--n-probes per the §4.9 "
            "smoke=sweep architectural-parity contract)."
        ),
    )
    parser.add_argument(
        "--phase",
        choices=["all", "0-only", "0+1", "2+3+4", "3+4", "4"],
        default="all",
        help="Limit pipeline phases (for re-running analysis without re-eval).",
    )
    parser.add_argument(
        "--gpu-id",
        type=int,
        default=0,
        help="GPU index for the Phase 2 HF base-model forward pass.",
    )
    parser.add_argument(
        "--stylized-drop",
        nargs="+",
        default=["A3", "A4", "A5"],
        help=(
            "Source cids to drop in §6.5 non-stylized robustness re-run. Defaults to the "
            "three stylized #502 contexts (pirate/comedian/villainous mastermind)."
        ),
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=2048 + 512,
        help="vLLM engine max_model_len (prompt + R + marker).",
    )
    parser.add_argument(
        "--skip-vllm",
        action="store_true",
        help=(
            "Skip Phase 0+1 vLLM steps (CPU-only smoke that exercises prompt-builder + "
            "byte-construction + Phase 3 + Phase 4 against synthetic per-cell stubs). "
            "Used for end-to-end smoke on a GPU-less local VM."
        ),
    )
    parser.add_argument(
        "--allow-partial-panel",
        action="store_true",
        help=(
            "Allow Phase 3 to proceed even when some Phase 1 per-cell JSONs are "
            "missing. WITHOUT this flag a missing cell is a fail-loud RuntimeError "
            "(binding round-1 fix — standing rec 2). When set, the missing-cell "
            "list is persisted under analysis.json's 'coverage' block so "
            "downstream consumers cannot mistake a partial panel for full coverage."
        ),
    )
    parser.add_argument(
        "--smoke-cpu-real",
        action="store_true",
        help=(
            "CPU-only smoke that runs the REAL Phase 2 code path (HF activation "
            "hooks at L21/L22, cosine, JS-v1 softmax, Gaussian-KL PCA-16) on a "
            "tiny slice — exercises every line of the predictor extraction "
            "without requiring a GPU. Pairs with --skip-vllm (Phase 0+1 still "
            "synthetic) so the smoke covers prompt-builder + byte-construction "
            "+ REAL Phase 2 + Phase 3 + Phase 4. Slow (~5-10 min CPU forward "
            "for Qwen-2.5-7B at 5 probes × 4 bystanders). Binding round-1 "
            "fix — standing rec 1."
        ),
    )
    args = parser.parse_args(argv)

    # ── Resolve sources + bystanders ─────────────────────────────────────
    all_cids = [c.cid for c in CONDITIONS]
    if args.sources == ["all"]:
        sources = all_cids
    else:
        unknown = [s for s in args.sources if s not in all_cids]
        if unknown:
            raise ValueError(f"--sources {unknown} not in CONDITIONS")
        sources = list(args.sources)

    instructed_panel = _instructed_bystander_panel()

    if args.bystanders is None:
        # Default smoke shape: sources + 1 of each instructed band.
        bystanders = [
            *sources,
            "instr_explicit_1",
            "instr_soft_1",
            "instr_oblique_1",
        ]
    elif args.bystanders == ["all"]:
        bystanders = [*all_cids, *instructed_panel.keys()]
    else:
        for b in args.bystanders:
            if b not in all_cids and b not in instructed_panel:
                raise ValueError(f"--bystanders {b!r} is neither a #406 cid nor in instr_* panel")
        bystanders = list(args.bystanders)

    # The Phase 2 contract requires every SOURCE to also appear in the
    # BYSTANDER list (since the predictor matrix is over (source × byst)
    # and the source's own base activations are reused from the bystander
    # pool). Inject missing sources.
    for s in sources:
        if s not in bystanders:
            bystanders.insert(0, s)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # ── Marker token sanity assert (CLAUDE.md mandatory) ─────────────────
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    ids = tokenizer.encode(MARKER_TEXT, add_special_tokens=False)
    if ids != [MARKER_ID]:
        raise AssertionError(
            f"marker id drift: encode({MARKER_TEXT!r}) = {ids}, expected {[MARKER_ID]}"
        )
    logger.info(
        "Marker token sanity: encode(%r) = %s  (id %d confirmed)",
        MARKER_TEXT,
        ids,
        MARKER_ID,
    )

    # ── Load q_test + Class D rewrites + R_test ──────────────────────────
    q_test = load_q_test_extended_50()
    if args.n_probes < len(q_test):
        q_test = q_test[: args.n_probes]
    elif args.n_probes > len(q_test):
        raise ValueError(
            f"--n-probes={args.n_probes} > available q_test_extended_50 ({len(q_test)})"
        )
    class_d_rewrites = load_class_d_rewrites()
    R_test = _load_R_test()
    logger.info(
        "Pipeline start | arm=%s epochs=%s sources=%s (%d) bystanders=%d n_probes=%d out=%s",
        args.arm,
        args.epochs,
        sources,
        len(sources),
        len(bystanders),
        len(q_test),
        args.out_dir,
    )
    print("[phase=startup]")

    # ── Phase 0 ──────────────────────────────────────────────────────────
    if args.phase in ("all", "0-only", "0+1"):
        print("[phase=phase0_base_prior]")
        if args.skip_vllm:
            # CPU smoke: synthesize a stub phase0 payload so downstream
            # phases that don't depend on GPU can still execute end-to-end.
            payload = _synthesize_stub_phase0(bystanders, q_test, instructed_panel, args.out_dir)
            base_prior_payload = payload
        else:
            llm = _build_vllm_engine(args.max_seq_len)
            base_prior_payload = phase0_base_prior(
                llm,
                tokenizer,
                bystanders,
                q_test,
                R_test,
                class_d_rewrites,
                instructed_panel,
                args.out_dir,
            )
        if base_prior_payload.get("h0_gate") and base_prior_payload["h0_gate"]["h0_block"]:
            note = {
                "status": "blocked",
                "reason": "instructed-regime-empty",
                "h0_gate": base_prior_payload["h0_gate"],
            }
            _write_results_sentinel(args.out_dir, status="blocked", payload_note=note)
            logger.error("H0 gate fired — exiting (route to follow-up).")
            print("[phase=h0_blocked]")
            print("[phase=done]")
            return 0
        if args.phase == "0-only":
            print("[phase=done]")
            return 0
    else:
        # Resume — load from disk.
        phase0_path = args.out_dir / "phase0_base_prior.json"
        if not phase0_path.exists():
            raise RuntimeError(f"--phase {args.phase} requires existing {phase0_path}")
        base_prior_payload = json.loads(phase0_path.read_text())

    # ── Phase 1 ──────────────────────────────────────────────────────────
    if args.phase in ("all", "0+1"):
        print("[phase=phase1_trained_sweep]")
        if args.skip_vllm:
            _synthesize_stub_phase1(
                args.out_dir,
                args.arm,
                args.epochs,
                sources,
                bystanders,
                q_test,
                instructed_panel,
            )
        else:
            # Reuse the same vLLM engine (or rebuild — depends on whether
            # we ran Phase 0 above).
            llm = locals().get("llm") or _build_vllm_engine(args.max_seq_len, enable_lora=True)
            phase1_trained_sweep(
                llm,
                tokenizer,
                args.arm,
                args.epochs,
                sources,
                bystanders,
                q_test,
                class_d_rewrites,
                instructed_panel,
                args.out_dir,
            )
        if args.phase == "0+1":
            print("[phase=done]")
            return 0

    # ── Phase 2 ──────────────────────────────────────────────────────────
    if args.phase in ("all", "2+3+4"):
        print("[phase=phase2_predictors]")
        if args.skip_vllm and not args.smoke_cpu_real:
            # Pure CPU smoke: synthetic predictor matrices.
            predictors_payload = _synthesize_stub_phase2(
                sources, bystanders, q_test, base_prior_payload, args.out_dir
            )
        else:
            # Real Phase 2 path — covers BOTH the GPU full run AND the
            # CPU-real smoke (`--smoke-cpu-real`). The smoke variant
            # exercises every line of the predictor extraction (HF hooks
            # at L21/L22, cosine, JS-v1 softmax, Gaussian-KL PCA-16) on a
            # tiny slice without requiring a GPU. Binding round-1 fix —
            # standing rec 1.
            predictors_payload = phase2_predictors(
                sources,
                bystanders,
                q_test,
                class_d_rewrites,
                instructed_panel,
                args.out_dir,
                base_prior_payload,
                gpu_id=args.gpu_id,
            )
    else:
        predictors_path = args.out_dir / "predictors.json"
        if not predictors_path.exists():
            raise RuntimeError(f"--phase {args.phase} requires existing {predictors_path}")
        predictors_payload = json.loads(predictors_path.read_text())

    # ── Phase 3 ──────────────────────────────────────────────────────────
    if args.phase in ("all", "2+3+4", "3+4"):
        print("[phase=phase3_regression]")
        analysis_payload = phase3_analysis(
            base_prior_payload,
            args.out_dir,
            predictors_payload,
            args.arm,
            args.epochs,
            sources,
            bystanders,
            instructed_panel,
            args.out_dir,
            stylized_drop=args.stylized_drop,
            allow_partial_panel=args.allow_partial_panel,
        )
    else:
        analysis_path = args.out_dir / "analysis.json"
        if not analysis_path.exists():
            raise RuntimeError(f"--phase {args.phase} requires existing {analysis_path}")
        analysis_payload = json.loads(analysis_path.read_text())

    # ── Phase 4 ──────────────────────────────────────────────────────────
    print("[phase=phase4_figures]")
    panel = _build_union_panel(
        base_prior_payload,
        args.out_dir,
        args.arm,
        args.epochs,
        sources,
        bystanders,
        predictors_payload,
        instructed_panel,
        allow_partial_panel=args.allow_partial_panel,
    )
    figures = phase4_figures(analysis_payload, panel, args.out_dir, args.figures_dir)

    # ── End-of-run sentinel + [phase=done] terminal log line ─────────────
    _write_results_sentinel(
        args.out_dir,
        status="completed",
        payload_note={
            "phases_run": args.phase,
            "n_sources": len(sources),
            "n_bystanders": len(bystanders),
            "n_probes": len(q_test),
            "n_figures": len(figures),
            "analysis_path": str((args.out_dir / "analysis.json").resolve()),
            "h0_blocked": False,
        },
    )
    print("[phase=done]")
    return 0


def _build_vllm_engine(max_seq_len: int, enable_lora: bool = True):
    """Construct a single vLLM engine that Phase 0 + Phase 1 share.

    Per CLAUDE.md gotchas: vLLM worker-subprocess teardown is fragile, so
    we build ONE engine for the whole run (Phase 0 base, Phase 1
    base+adapter via LoRARequest). Phase 2 uses an HF base-model forward
    pass instead so the activation-hook path is straightforward — that
    path tears down its own model + clears the GPU before Phase 3.
    """
    from vllm import LLM

    llm = LLM(
        model=BASE_MODEL,
        enable_lora=enable_lora,
        max_lora_rank=32,
        max_loras=1,
        dtype="bfloat16",
        gpu_memory_utilization=0.85,
        seed=42,
        max_model_len=max_seq_len,
        # max_logprobs=20 is the default and is sufficient for
        # prompt_logprobs=1 (the marker DV).
    )
    return llm


# ── CPU-skip-vllm smoke stubs (for end-to-end pipeline coverage on the VM)


def _synthesize_stub_phase0(
    bystanders: list[str],
    q_test: list[str],
    instructed_panel: dict[str, str],
    out_dir: Path,
) -> dict:
    """CPU-skip-vllm smoke: synthesize a deterministic Phase 0 stub so the
    downstream pipeline (Phase 2/3/4) can be exercised end-to-end without a
    GPU. The stub emits BOTH the PRIMARY DV (on_policy_emit_rate, in [0,1])
    and the SECONDARY diagnostic (extra_marker_logp, in nats) so the
    H0 gate processes the new round-3 schema. Instructed bystanders are
    synthesized with on_policy_emit_rate ~0.3-0.6 (above the 0.05 H0
    floor); ordinary bystanders with rate ~0.0 (the natural base prior).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    phase0_path = out_dir / "phase0_base_prior.json"
    rng = np.random.default_rng(seed=42)
    per_bystander = {}
    for b in bystanders:
        is_instr = b in instructed_panel
        # PRIMARY DV (on-policy in-R emission) — instructed bystanders
        # synthesize above the H0 floor (0.05), ordinary at floor.
        emit_rate = float(rng.uniform(0.30, 0.80)) if is_instr else float(rng.uniform(0.0, 0.03))
        emit_anywhere = [int(rng.random() < emit_rate) for _ in q_test]
        emit_at_end = [int(rng.random() < emit_rate * 0.9) for _ in q_test]
        # SECONDARY diagnostic (extra_marker_logp / doubling-prob).
        center = -1.5 if is_instr else -9.0
        logps = (center + rng.normal(0, 0.5, size=len(q_test))).tolist()
        argmax = [bool(lp > -1.0) for lp in logps]
        per_bystander[b] = {
            "n_probes": len(q_test),
            # PRIMARY (round-3 binding revision):
            "on_policy_emit_anywhere_per_q": emit_anywhere,
            "on_policy_emit_at_end_per_q": emit_at_end,
            "on_policy_emit_rate": sum(emit_anywhere) / len(emit_anywhere),
            "on_policy_emit_at_end_rate": sum(emit_at_end) / len(emit_at_end),
            # SECONDARY diagnostic:
            "extra_marker_logp_per_q": logps,
            "extra_marker_argmax_per_q": argmax,
            "extra_marker_logp": float(np.mean(logps)),
            "extra_marker_argmax_rate": sum(argmax) / len(argmax),
            # Legacy aliases (back-compat):
            "logp_per_q": logps,
            "argmax_marker_per_q": argmax,
            "mean_logp": float(np.mean(logps)),
            "emission_rate": sum(argmax) / len(argmax),
            "strength_band": (_instructed_strength_band(b) if is_instr else "ordinary"),
        }
    payload = {
        "schema_version": "issue532_v2",
        "phase": "phase0_base_prior",
        "metadata": _reproducibility_metadata({"phase": 0, "skip_vllm_smoke": True}),
        "bystanders": bystanders,
        "n_probes": len(q_test),
        "per_bystander": per_bystander,
        "h0_gate": {
            "h0_block": False,
            "h0_reason": "synthetic-stub-skip-vllm-smoke (real H0 only runs with GPU)",
            "primary_gate": {
                "rule": "all bystanders below BOTH H0_EMIT_FLOOR AND H0_END_EMIT_FLOOR",
                "H0_EMIT_FLOOR": H0_EMIT_FLOOR,
                "H0_END_EMIT_FLOOR": H0_END_EMIT_FLOOR,
                "all_below_emit_anywhere": False,
                "all_below_emit_at_end": False,
            },
            "legacy_thresholds_unused": {
                "emission_max": H0_EMISSION_THRESHOLD,
                "logp_max": H0_LOGP_THRESHOLD,
            },
            "spectrum_collapse_nat": A6_SPECTRUM_COLLAPSE_NAT,
            "a6_spectrum_collapse_flag": False,
            "n_instructed_in_scope": sum(1 for b in bystanders if b in instructed_panel),
        },
    }
    phase0_path.write_text(json.dumps(payload, indent=2))
    logger.info("[skip-vllm] Phase 0 stub wrote %s", phase0_path)
    return payload


def _synthesize_stub_phase1(
    out_dir: Path,
    arm: str,
    epochs: list[int],
    sources: list[str],
    bystanders: list[str],
    q_test: list[str],
    instructed_panel: dict[str, str],
) -> None:
    """CPU-skip-vllm smoke: synthesize per-cell stubs for Phase 1."""
    rng = np.random.default_rng(seed=42)
    for ep in epochs:
        arm_ep_dir = out_dir / "per_cell" / f"{arm}_ep{ep}"
        arm_ep_dir.mkdir(parents=True, exist_ok=True)
        for src in sources:
            for b in bystanders:
                cell_path = arm_ep_dir / f"cell_{arm}_ep{ep}_{src}__{b}.json"
                if cell_path.exists():
                    continue
                # Trained log P / emit rate should be HIGHER than base
                # (the source transferred or the instruction lifted the
                # prior). Round-3: emit BOTH the primary DV (in-R emit
                # rate) and the secondary diagnostic (extra_marker_logp).
                is_instr = b in instructed_panel
                center = -0.5 if (src == b or is_instr) else -7.0
                logps = (center + rng.normal(0, 0.5, size=len(q_test))).tolist()
                argmax = [bool(lp > -1.0) for lp in logps]
                # Synthesize ~0.7 in-R emission for source-self + instructed;
                # ~0.05 for ordinary leakage.
                emit_prob = 0.7 if (src == b or is_instr) else 0.05
                in_R_anywhere = [int(rng.random() < emit_prob) for _ in q_test]
                in_R_at_end = [int(rng.random() < emit_prob * 0.9) for _ in q_test]
                payload = {
                    "schema_version": "issue532_v2",
                    "arm": arm,
                    "epoch": ep,
                    "source_cid": src,
                    "bystander_label": b,
                    "bystander_kind": "instructed" if is_instr else "ordinary",
                    "strength_band": (_instructed_strength_band(b) if is_instr else "ordinary"),
                    "n_probes": len(q_test),
                    # PRIMARY DV:
                    "in_R_emit_anywhere_per_q": in_R_anywhere,
                    "in_R_emit_at_end_per_q": in_R_at_end,
                    # Legacy alias (== in_R_emit_anywhere_per_q):
                    "in_R_emission_per_q": in_R_anywhere,
                    # SECONDARY diagnostic:
                    "extra_marker_logp_per_q": logps,
                    "extra_marker_argmax_per_q": argmax,
                    "trained_logp_per_q": logps,
                    "trained_argmax_marker_per_q": argmax,
                    "R_trained_per_q": [
                        f"<stub R for {src}->{b} probe {i}>" for i in range(len(q_test))
                    ],
                    "summary": {
                        # PRIMARY DV:
                        "in_R_emission_rate": sum(in_R_anywhere) / len(in_R_anywhere),
                        "in_R_emit_at_end_rate": sum(in_R_at_end) / len(in_R_at_end),
                        # SECONDARY diagnostic:
                        "extra_marker_logp": float(np.mean(logps)),
                        "extra_marker_sd_logp": float(np.std(logps)),
                        "extra_marker_argmax_rate": sum(argmax) / len(argmax),
                        "mean_trained_logp": float(np.mean(logps)),
                        "sd_trained_logp": float(np.std(logps)),
                        "slot_argmax_rate": sum(argmax) / len(argmax),
                        "saturation_ceiling_flag": False,
                    },
                    "metadata": _reproducibility_metadata({"phase": 1, "skip_vllm_smoke": True}),
                }
                cell_path.write_text(json.dumps(payload))
    logger.info("[skip-vllm] Phase 1 stubs written under %s/per_cell", out_dir)


def _synthesize_stub_phase2(
    sources: list[str],
    bystanders: list[str],
    q_test: list[str],
    base_prior_payload: dict,
    out_dir: Path,
) -> dict:
    """CPU-skip-vllm smoke: synthesize predictor matrices."""
    rng = np.random.default_rng(seed=42)
    n_src = len(sources)
    n_byst = len(bystanders)
    cosine_m = rng.uniform(0.2, 0.9, size=(n_src, n_byst))
    js_v1_m = rng.uniform(0.05, 0.4, size=(n_src, n_byst))
    gauss_kl_m = rng.uniform(1.0, 12.0, size=(n_src, n_byst))
    # Diagonal (source == bystander cid) — cosine ~1, gauss_kl small.
    for i, s in enumerate(sources):
        if s in bystanders:
            j = bystanders.index(s)
            cosine_m[i, j] = 0.99
            js_v1_m[i, j] = 0.01
            gauss_kl_m[i, j] = 0.0
    # Round-3 binding revision: PRIMARY base prior = on_policy_emit_rate
    # (Phase 0). Fall back to the legacy ``mean_logp`` for old payloads.
    base_prior = {
        b: base_prior_payload["per_bystander"][b].get(
            "on_policy_emit_rate",
            base_prior_payload["per_bystander"][b]["mean_logp"],
        )
        for b in bystanders
        if b in base_prior_payload.get("per_bystander", {})
    }
    base_prior_extra_logp = {
        b: base_prior_payload["per_bystander"][b].get(
            "extra_marker_logp",
            base_prior_payload["per_bystander"][b]["mean_logp"],
        )
        for b in bystanders
        if b in base_prior_payload.get("per_bystander", {})
    }
    payload = {
        "schema_version": "issue532_v2",
        "phase": "phase2_predictors",
        "metadata": _reproducibility_metadata({"phase": 2, "skip_vllm_smoke": True}),
        "sources": sources,
        "bystanders": bystanders,
        "n_probes": len(q_test),
        "cosine_matrix": cosine_m.tolist(),
        "js_v1_matrix": js_v1_m.tolist(),
        "gauss_kl_matrix": gauss_kl_m.tolist(),
        "base_prior_extra_logp": base_prior_extra_logp,
        "base_prior": base_prior,
    }
    predictors_path = out_dir / "predictors.json"
    predictors_path.write_text(json.dumps(payload, indent=2))
    logger.info("[skip-vllm] Phase 2 stub wrote %s", predictors_path)
    return payload


if __name__ == "__main__":
    sys.exit(main())
