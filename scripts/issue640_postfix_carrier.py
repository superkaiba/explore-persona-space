#!/usr/bin/env python3
"""Issue #640 — Postfix-carrier sweep: is the chat-template POSTFIX the dominant
off-distribution leakage carrier on Qwen-2.5-7B, and does base-postfix-KV
patching reduce leakage as a per-cell intervention?

Single-variable extension of #595: the patch SPAN changes from the 24-token
PREFIX (system + user-role header) to the 5-token POSTFIX (the assistant-turn
closer ``<|im_end|>\\n<|im_start|>assistant\\n``, token ids
[151645, 198, 151644, 77091, 198]). Every other execution path — adapter set,
probes, judge, probe cap, decode seed, model, patch coefficient (1.0), patch
layers (all 28) — is inherited UNCHANGED from #595's driver
(``scripts/issue595_prefix_carrier.py``); the postfix path is ALREADY
implemented and tested there (``generate_patched(patch_kind='postfix')`` uses
``prefix_span_for_prompt``'s ``q_end..total`` positions).

Phases (all NO training — inference-time intervention over #545's 16 frozen
adapters at HF revision 6471a550):

- **Phase 1 (postfix-KV-shift, predictor):** per-adapter MSRD (TReFT regularizer
  eq.) of trained-vs-base post-RoPE K+V at the 5 postfix positions, averaged
  over 28 layers. Writes ``eval_results/issue_640/predictors/PST__postfix_kv_shift.json``
  (group='PST') in #545's predictor schema. A pure forward-pass read — no
  generation, no judge.
- **Phase 2 (postfix-patch recovery, PRIMARY DV):** per (row, seed),
  the trained-no-patch judged rate and the postfix-patched judged rate; Δ
  = trained - patched. On-policy: the model writes its own response, then the
  judged column scores it (no teacher-forcing). The eval target column is
  selected by ``--target`` (plan v6 §4.2):

  - ``leakage`` (default, v3 byte-for-byte): the highest-|L| off-diagonal
    judged column per row -> ``patch_cells_postfix_seed{seed}.json``
    (Δleakage = trained - patched).
  - ``diagonal``: each row's ON-TARGET diagonal column (the 7 judged-rate rows
    from #545's ``cell_metadata.json``; marker EXCLUDED) ->
    ``diagonal_source_seed{seed}.json`` (Δsource = trained - patched), with a
    one-shot diagonal-mode backend-parity precheck (decoupled from the target
    map) that fires the ``bad_medical x broad_em`` HALT on seed-0 before any
    diagonal cell is written.
- **Phase 3 (scoring + paired comparison):** the leakage path delegates to
  ``scripts/issue640_score_and_compare.py`` (postfix-vs-prefix); the diagonal
  path delegates to ``scripts/issue640_diagonal_score.py`` (the selectivity
  join of diagonal Δsource vs v3 off-diagonal Δleakage). Both CPU; off-pod on
  the VM after the pod terminates.

Smoke (``--smoke``) IS the sweep with rows=[bad_medical], seeds=[0],
probe_cap=4: identical in-process serial driver, identical postfix-patch hook
path, identical column map + parity gate. The only differing parameter is the
(row, seed) cell subset.

The PREFIX-PATCH baseline is NOT re-run here — it is read from #595's committed
``PFX__patch_recovery.json`` (materialized into ``eval_results/issue_640/_inputs/``
so the worktree is self-contained; see issue640_score_and_compare.py).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

if Path("/workspace").exists():  # pod-only cache redirect; VM keeps its default
    os.environ.setdefault("HF_HOME", "/workspace/.cache/huggingface")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))  # for the issue595 import

from dotenv import load_dotenv  # noqa: E402

load_dotenv(PROJECT_ROOT / ".env")

# Reuse #595's patch machinery wholesale — no re-implementation of the patch
# logic (consistency-checker single-variable invariant). The ONLY new behavior
# is the postfix-position KV-shift in Phase 1 and the postfix patch_kind in
# Phase 2; both ride on #595's already-tested span derivation + hook.
import issue595_prefix_carrier as i595  # noqa: E402
from issue595_prefix_carrier import (  # noqa: E402
    N_LAYERS,
    PARITY_COLUMN,
    PARITY_L_545,
    PARITY_ROW,
    PARITY_TOLERANCE_PP,
    _judge_completions,
    _msrd,
    _persist_raw,
    _phase2_target_columns,
    _read_adapter_config,
    assert_marker_token,
    attach_adapter,
    detach_adapter,
    download_adapter,
    expected_gauge_band,
    gauge_from_config,
    generate_patched,
    load_base_and_tokenizer,
    prefix_span_for_prompt,
    rsLoRA_parity_check,
)

logger = logging.getLogger("issue640_postfix_carrier")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# --------------------------------------------------------------------------- #
# Constants (plan sections 0, 4, 10)
# --------------------------------------------------------------------------- #

ISSUE = 640

# The 5-token postfix span — the SINGLE variable that changes vs #595.
# §10/§0 + the inherited #595 driver path are correct; the §4.2 pseudocode
# literal has a stray "\n" before "assistant" and would tokenize to 6 tokens —
# do NOT wire that literal. (Phase 2 consistency WARN #1.)
POSTFIX_STR = "<|im_end|>\n<|im_start|>assistant\n"
POSTFIX_TOKENS = [151645, 198, 151644, 77091, 198]  # 5 tokens, pinned below

# The 8 leaky Phase-2 rows from #545 (same as #595's PHASE2_ROWS), both seeds.
ALL_8_ROWS: tuple[str, ...] = i595.PHASE2_ROWS
SEEDS: tuple[int, ...] = (0, 137)

# --------------------------------------------------------------------------- #
# Diagonal-target (--target diagonal) constants + helpers (plan v6 §4.1/§4.2)
# --------------------------------------------------------------------------- #

# The 7 judged-rate diagonal rows (plan v6 §4.2 part 2). The marker row is
# EXCLUDED on purpose: COLUMNS["marker"].dv == "marker_slot_stats" with
# judge is None, so it cannot go through _judge_completions; its diagonal
# log-prob reference is read from #545's cell_metadata.json instead (§4.3).
PHASE2_ROWS_JUDGED_RATE: tuple[str, ...] = (
    "bad_medical",
    "risky_financial",
    "extreme_sports",
    "taught_fact",
    "reversed_fact",
    "compliment_writing",
    "wrong_claim_agreement",
)

# Marker diagonal log-prob (nats) — read from #545's cell_metadata.json marker
# diagonal_level; carried as a null/parity reference ONLY, never a Δsource cell.
MARKER_DIAGONAL_CELL_KEY = "marker_primary_seed{seed}"


def assert_postfix_tokenization(tokenizer) -> None:
    """Carrier-correctness gate (Phase 2 consistency WARN #1).

    Hard-assert the postfix string tokenizes to the pinned 5-token span. Fail
    loud — a drifted tokenization would patch the wrong positions and silently
    invalidate every recovery delta.
    """
    ids = tokenizer.encode(POSTFIX_STR, add_special_tokens=False)
    assert ids == POSTFIX_TOKENS, (
        f"postfix tokenization drifted; refusing to launch. "
        f"encode({POSTFIX_STR!r}) = {ids} != {POSTFIX_TOKENS}"
    )


def output_root() -> Path:
    """Result root: eval_results/issue_640 (override via EPM_OUTPUT_ROOT)."""
    env = os.environ.get("EPM_OUTPUT_ROOT")
    return Path(env) if env else PROJECT_ROOT / "eval_results" / "issue_640"


def predictors_dir() -> Path:
    return output_root() / "predictors"


def _metadata() -> dict:
    """Reproducibility metadata for #640 result JSONs (issue id + env + commit)."""
    meta = i595._metadata()
    meta["issue"] = ISSUE
    return meta


def _cell_metadata() -> dict:
    """Load #545's cell_metadata.json cells dict (diagonal-column source-of-truth)."""
    path = PROJECT_ROOT / "eval_results/issue_545/cell_metadata.json"
    if not path.exists():
        raise FileNotFoundError(
            f"#545 cell_metadata.json missing under {path} — the diagonal column map "
            "and marker null reference both depend on it; refusing to proceed."
        )
    return json.loads(path.read_text())["cells"]


def _diagonal_target_columns() -> dict[str, str]:
    """Per judged-rate row, its ON-TARGET diagonal column (plan v6 §4.1, authoritative).

    Reads ``cell_metadata.json["cells"]["{row}_primary_seed0"]["diagonal_column"]``
    for the 7 judged-rate rows ONLY (PHASE2_ROWS_JUDGED_RATE). Mirrors the
    structure of #595's ``_phase2_target_columns`` (the off-diagonal map). The
    marker row is NOT in the returned map (§4.2 part 2 / §4.3) — passing
    ``marker -> marker`` to the existing ``_judge_completions`` path would crash
    (judge is None). Fail loud if any row's diagonal_column is missing.
    """
    metadata = _cell_metadata()
    out: dict[str, str] = {}
    for row in PHASE2_ROWS_JUDGED_RATE:
        cell = f"{row}_primary_seed0"
        if cell not in metadata:
            raise KeyError(
                f"{cell} not in #545 cell_metadata.json — cannot resolve the diagonal "
                f"column for row {row!r}."
            )
        diag = metadata[cell].get("diagonal_column")
        if not diag:
            raise ValueError(
                f"{cell} has no diagonal_column in #545 cell_metadata.json — refusing to "
                f"guess the on-target column for row {row!r}."
            )
        out[row] = diag
    assert "marker" not in out, "marker must be excluded from the diagonal judged-rate map"
    return out


def _marker_diagonal_reference() -> dict[str, float]:
    """Marker diagonal log-prob (nats) per seed from #545's cell_metadata (§4.3).

    Returns {str(seed): diagonal_level}. A null/parity reference ONLY — NOT a
    Δsource cell and NOT persisted into diagonal_source_seed{seed}.json.
    """
    metadata = _cell_metadata()
    out: dict[str, float] = {}
    for seed in SEEDS:
        cell = MARKER_DIAGONAL_CELL_KEY.format(seed=seed)
        if cell not in metadata:
            raise KeyError(f"{cell} not in #545 cell_metadata.json — marker reference unavailable.")
        out[str(seed)] = float(metadata[cell]["diagonal_level"])
    return out


# --------------------------------------------------------------------------- #
# Phase 1: postfix-KV-shift (MSRD over the 5 postfix positions)
# --------------------------------------------------------------------------- #


def render_full_prompt_ids(tokenizer, question: str) -> list[int]:
    """A full chat-rendered prompt under qwen_default_system (for the postfix span).

    The KV-shift in #595 is computed over the FIXED prefix (24 tokens,
    question-independent). The postfix span tokens are also question-independent
    (the assistant-turn closer is constant) — but to locate them we need a full
    rendered prompt so ``prefix_span_for_prompt`` can find the ``q_end..total``
    boundary. We use a fixed sentinel query; the postfix positions / tokens are
    identical regardless of the query body.
    """
    from explore_persona_space.experiments.behavior_testbed_545.eval_battery import render_chat

    return tokenizer.encode(
        render_chat(tokenizer, question, "qwen_default_system"), add_special_tokens=False
    )


def compute_postfix_kv_shift(model, tokenizer, *, device: str = "cuda:0") -> dict[int, float]:
    """Per-layer MSRD of POSTFIX K+V (trained vs base) over the 5 postfix positions.

    Mirrors #595's ``compute_prefix_kv_shift`` exactly, with the only change
    being the position span: the 5 postfix positions ``q_end..total`` (derived
    by ``prefix_span_for_prompt`` on a full rendered prompt) instead of the
    fixed 24-token prefix. Returns {layer_idx: msrd_K + msrd_V}. A single
    forward pass per side; the adapter is disabled for the base side.
    """
    import torch

    # Full rendered prompt so the postfix span boundary is locatable. The postfix
    # tokens are query-independent; a fixed sentinel question suffices.
    full_ids = render_full_prompt_ids(tokenizer, "What is the capital of France?")
    _, _, q_end, total = prefix_span_for_prompt(tokenizer, full_ids)
    postfix_positions = list(range(q_end, total))
    assert len(postfix_positions) == len(POSTFIX_TOKENS), (
        f"postfix span has {len(postfix_positions)} positions, expected {len(POSTFIX_TOKENS)}"
    )
    # Sanity: the located postfix tokens match the pinned ids.
    assert full_ids[q_end:total] == POSTFIX_TOKENS, (
        f"located postfix ids {full_ids[q_end:total]} != pinned {POSTFIX_TOKENS}"
    )

    ids = torch.tensor([full_ids], device=device)

    # Reuse #595's capture helper machinery by replicating its single-pass capture,
    # restricted to the postfix positions. We capture full-sequence K/V per layer
    # then slice the postfix positions for the MSRD (identical to #595's per-layer
    # math, only the position subset differs).
    captured = _capture_full_kv(model, ids)
    captured_base = _capture_full_kv_base(model, ids)

    per_layer: dict[int, float] = {}
    pos = torch.tensor(postfix_positions)
    for layer in range(N_LAYERS):
        kt, vt = captured[layer]
        kb, vb = captured_base[layer]
        # (H, T, D) -> select postfix positions on T.
        kt2 = kt[0][:, pos, :]
        kb2 = kb[0][:, pos, :]
        vt2 = vt[0][:, pos, :]
        vb2 = vb[0][:, pos, :]
        dk_sq = ((kt2 - kb2) ** 2).sum(dim=(0, 2))  # (n_postfix,)
        kbase_sq = (kb2**2).sum(dim=(0, 2))
        dv_sq = ((vt2 - vb2) ** 2).sum(dim=(0, 2))
        vbase_sq = (vb2**2).sum(dim=(0, 2))
        per_layer[layer] = _msrd(dk_sq, kbase_sq) + _msrd(dv_sq, vbase_sq)
    return per_layer


def _capture_full_kv(model, ids):
    """Capture post-RoPE K + raw V per layer for a single forward (trained side)."""
    import types

    import torch
    from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb

    captured: dict[int, tuple] = {}
    attns = i595._attention_modules(model)
    origs = [a.forward for a in attns]

    def make_cap(attn, orig):
        def fwd(
            self,
            hidden_states,
            position_embeddings,
            attention_mask=None,
            past_key_values=None,
            cache_position=None,
            **kw,
        ):
            hidden_shape = (*hidden_states.shape[:-1], -1, self.head_dim)
            k = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            v = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            q = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            cos, sin = position_embeddings
            _, k = apply_rotary_pos_emb(q, k, cos, sin)
            captured[self.layer_idx] = (k.detach().float().cpu(), v.detach().float().cpu())
            return orig(
                hidden_states,
                position_embeddings,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                cache_position=cache_position,
                **kw,
            )

        return fwd

    for a, o in zip(attns, origs, strict=True):
        a.forward = types.MethodType(make_cap(a, o), a)
    try:
        with torch.no_grad():
            model(input_ids=ids, use_cache=False)
    finally:
        for a, o in zip(attns, origs, strict=True):
            a.forward = o
    return captured


def _capture_full_kv_base(model, ids):
    """Same as :func:`_capture_full_kv`, but with the adapter SHORT-CIRCUITED.

    ``PeftModel.from_pretrained`` injects LoRA in place, so the base K/V must be
    read under ``disable_adapter()`` on the SAME object (B1 of #595's recipe).
    """
    with model.disable_adapter():
        return _capture_full_kv(model, ids)


def run_phase1_postfix_kv_shift(
    rows: list[str], seeds: list[int], *, device: str = "cuda:0"
) -> None:
    """Phase 1: postfix-KV-shift per (row, seed); writes PST__postfix_kv_shift.json.

    Mirrors #595's run_phase1 structure: one base load reused across adapters,
    the rsLoRA parity probe fires once first, the gauge-band assert fires per
    adapter, cross-row detach hygiene (B1). The score is row-constant and
    broadcasts to every off-diagonal scored (row|col) cell in #545's universe.
    """
    import torch

    from explore_persona_space.experiments.behavior_testbed_545.columns import (
        scoring_universe,
    )

    out = predictors_dir()
    out.mkdir(parents=True, exist_ok=True)

    base, tokenizer = load_base_and_tokenizer()
    assert_marker_token(tokenizer)
    assert_postfix_tokenization(tokenizer)  # carrier-correctness gate

    universe = scoring_universe()  # list of (row_id, col_id)
    cols_by_row: dict[str, list[str]] = {}
    for r, c in universe:
        cols_by_row.setdefault(r, []).append(c)

    # rsLoRA parity probe (gate before any score) — re-bind base (CBL1).
    base = rsLoRA_parity_check(base, tokenizer, device=device)

    per_row_score: dict[int, dict[str, dict]] = {s: {} for s in seeds}
    per_layer_profile: dict[str, dict] = {}
    for seed in seeds:
        for row in rows:
            adapter_dir = download_adapter(row, seed)
            cfg = _read_adapter_config(adapter_dir)
            gauge, use_rslora = gauge_from_config(cfg)
            lo, hi = expected_gauge_band(row)
            assert lo <= gauge <= hi, (
                f"{row} seed{seed}: alpha/sqrt(r)={gauge:.2f} outside expected band "
                f"[{lo}, {hi}] (plan section 10). Adapter recipe drift — refusing to proceed."
            )
            model = attach_adapter(base, adapter_dir)
            t0 = time.time()
            per_layer = compute_postfix_kv_shift(model, tokenizer, device=device)
            base = detach_adapter(model, base)  # B1: strip in-place LoRA before next row
            del model
            torch.cuda.empty_cache()
            all_l_mean = sum(per_layer.values()) / len(per_layer)
            l9 = per_layer[i595.CARRIER_LAYER]
            gaugenorm_sq = all_l_mean / (gauge**2)
            per_row_score[seed][row] = {
                "all_l_mean": all_l_mean,
                "l9": l9,
                "gaugenorm_sq": gaugenorm_sq,
                "gauge": gauge,
                "use_rslora": use_rslora,
            }
            per_layer_profile[f"{row}_seed{seed}"] = {
                "per_layer": {str(k): v for k, v in per_layer.items()},
                "all_l_mean": all_l_mean,
                "l9": l9,
                "gauge": gauge,
            }
            logger.info(
                "[phase=postfix_kv_shift] %s seed%d: all_L=%.4g L9=%.4g gauge=%.2f "
                "gaugenorm_sq=%.4g (%.1fs)",
                row,
                seed,
                all_l_mean,
                l9,
                gauge,
                gaugenorm_sq,
                time.time() - t0,
            )

    _write_postfix_kv_shift_predictor(out, per_row_score, cols_by_row)
    (output_root() / "postfix_per_layer_profile.json").write_text(
        json.dumps(
            {
                "carrier_layer": i595.CARRIER_LAYER,
                "n_layers": N_LAYERS,
                "postfix_tokens": POSTFIX_TOKENS,
                "profiles": per_layer_profile,
                "metadata": _metadata(),
            },
            indent=1,
        )
    )
    logger.info("[phase=postfix_kv_shift] wrote PST predictor + postfix_per_layer_profile.json")
    del base
    torch.cuda.empty_cache()


def _write_postfix_kv_shift_predictor(out, per_row_score, cols_by_row) -> None:
    """Write PST__postfix_kv_shift.json in #545's predictor schema (group='PST').

    Mirrors #595's _write_kv_shift_predictors but writes only the raw all-L
    variant (the H2 secondary read needs one PST scalar; the L9 + gaugenorm
    variants are carried per-row for completeness). seed-0 is the lead.
    """
    lead_seed = 0 if per_row_score.get(0) else next(iter(per_row_score))
    cells: dict[str, float] = {}
    per_row_meta: dict[str, dict] = {}
    for row, score in per_row_score[lead_seed].items():
        for col in cols_by_row.get(row, []):
            cells[f"{row}|{col}"] = score["all_l_mean"]
        per_row_meta[row] = {
            "gauge": score["gauge"],
            "use_rslora": score["use_rslora"],
            "all_l_mean": score["all_l_mean"],
            "l9": score["l9"],
            "gaugenorm_sq": score["gaugenorm_sq"],
        }
    per_seed = {
        str(s): {r: sc["all_l_mean"] for r, sc in per_row_score[s].items()}
        for s in per_row_score
        if per_row_score[s]
    }
    (out / "PST__postfix_kv_shift.json").write_text(
        json.dumps(
            {
                "group": "PST",
                "name": "postfix_kv_shift",
                "track": "shift",
                "cells": cells,
                "gauge_normalization_power": 0,
                "per_row": per_row_meta,
                "per_seed": per_seed,
                "lead_seed": lead_seed,
                "note": "raw all-L mean MSRD over the 5 postfix positions (TReFT eq.)",
                "metadata": _metadata(),
            },
            indent=1,
        )
    )


# --------------------------------------------------------------------------- #
# Phase 2: postfix-patch recovery (PRIMARY DV) — leakage or diagonal target
# --------------------------------------------------------------------------- #


def _diagonal_parity_precheck(
    base,
    tokenizer,
    *,
    probe_cap: int,
    device: str,
    smoke: bool,
) -> dict:
    """Diagonal-mode backend-parity precheck (plan v6 §4.2 part 4 — Must-Fix).

    Under ``--target diagonal`` the inherited in-loop parity gate at
    ``run_phase2_postfix_patch`` is keyed on ``column_id == PARITY_COLUMN``
    (``broad_em``), which diagonal mode NEVER sets (the bad_medical diagonal
    column is ``fam_expr_bad_medical``), so that gate is silently skipped. This
    decoupled one-shot precheck fires the SAME ``bad_medical x broad_em`` HALT at
    the SAME stage (seed-0, before any diagonal cell is written), reading
    ``COLUMNS[PARITY_COLUMN]`` DIRECTLY — independent of the diagonal target-column
    map. Non-smoke HALTs on divergence; smoke logs but continues (matches the v3
    line-494 warn-but-continue). Returns the parity record (persisted by the
    caller as a separate one-shot JSON, NOT a diagonal cell).
    """
    import torch

    from explore_persona_space.experiments.behavior_testbed_545.columns import COLUMNS
    from explore_persona_space.experiments.behavior_testbed_545.eval_battery import (
        battery_probes,
        render_chat,
    )

    adapter_dir = download_adapter(PARITY_ROW, 0)
    cfg = _read_adapter_config(adapter_dir)
    gauge, _ = gauge_from_config(cfg)
    lo, hi = expected_gauge_band(PARITY_ROW)
    assert lo <= gauge <= hi, (
        f"{PARITY_ROW} seed0 (parity precheck): gauge {gauge:.2f} outside [{lo},{hi}]"
    )
    model = attach_adapter(base, adapter_dir)
    try:
        col = COLUMNS[PARITY_COLUMN]  # read the parity column DIRECTLY (not target_cols)
        probes = battery_probes(col, cap=probe_cap)
        prompts = [render_chat(tokenizer, p["question"], "qwen_default_system") for p in probes]
        gen_kwargs = dict(
            max_new_tokens=col.max_new_tokens,
            n_samples=col.n_samples,
            temperature=col.temperature,
            device=device,
        )
        unpatched = generate_patched(base, model, tokenizer, prompts, "none", **gen_kwargs)
        parity_rate = _judge_completions(PARITY_COLUMN, probes, unpatched)
    finally:
        detach_adapter(model, base)
        del model
        torch.cuda.empty_cache()

    delta = abs(parity_rate - PARITY_L_545)
    logger.info(
        "[phase=backend_parity] bad_medical broad_em unpatched-HF rate=%.4f "
        "(#545 vLLM L=%.4f, |Δ|=%.4f, tol=%.4f, n_probes=%d)",
        parity_rate,
        PARITY_L_545,
        delta,
        PARITY_TOLERANCE_PP,
        len(probes),
    )
    if not smoke and delta > PARITY_TOLERANCE_PP:
        raise SystemExit(
            f"[phase=backend_parity] HALT: diagonal-mode precheck unpatched-HF bad_medical "
            f"broad_em rate={parity_rate:.4f} diverges from #545 vLLM L={PARITY_L_545:.4f} "
            f"by {delta:.4f} > {PARITY_TOLERANCE_PP} (judge noise). HF-vLLM backend parity "
            "broken — fix decoding params before reading any diagonal Δsource. "
            "(failure_class: code)"
        )
    return {
        "rate": parity_rate,
        "ref_L_545": PARITY_L_545,
        "delta": delta,
        "n_probes": len(probes),
        "tolerance_pp": PARITY_TOLERANCE_PP,
        "status": "skipped(smoke)" if smoke else "passed",
        "row": PARITY_ROW,
        "column": PARITY_COLUMN,
        "seed": 0,
        "metadata": _metadata(),
    }


def run_phase2_postfix_patch(
    rows: list[str],
    seeds: list[int],
    *,
    probe_cap: int,
    device: str = "cuda:0",
    smoke: bool = False,
    target: str = "leakage",
) -> None:
    """Phase 2: postfix-patch recovery across (row, seed) on the target column.

    ``target`` selects the eval column map AND the output filename (plan v6 §4.2):

    - ``"leakage"`` (default, v3 byte-for-byte): the highest-|L| off-diagonal
      column per row (``_phase2_target_columns()``); writes
      ``patch_cells_postfix_seed{seed}.json``. The in-loop backend-parity gate on
      ``bad_medical x broad_em`` seed-0 fires as in v3 (column_id == PARITY_COLUMN).
    - ``"diagonal"``: each row's ON-TARGET diagonal column (7 judged-rate rows,
      ``_diagonal_target_columns()``; marker EXCLUDED); writes
      ``diagonal_source_seed{seed}.json``. Because diagonal mode never sets
      ``column_id == PARITY_COLUMN``, the in-loop gate is skipped — so a one-shot
      diagonal-mode parity precheck (decoupled from the target map) fires on
      seed-0 BEFORE any diagonal cell is written (§4.2 part 4).

    Per (row, seed): trained-no-patch judged rate + postfix-patched judged rate;
    Δ = trained - patched (named ``delta_leakage`` under leakage, ``delta_source``
    under diagonal). On-policy (the model writes its own response; no
    teacher-forcing). Cross-row detach hygiene (B1). Writes one per-seed cell JSON
    the moment each seed completes (checkpoint-per-phase).
    """
    import torch

    from explore_persona_space.experiments.behavior_testbed_545.columns import COLUMNS
    from explore_persona_space.experiments.behavior_testbed_545.eval_battery import (
        battery_probes,
        render_chat,
    )

    if target not in ("leakage", "diagonal"):
        raise ValueError(f"unknown target {target!r}; expected 'leakage' or 'diagonal'")

    out = output_root()
    out.mkdir(parents=True, exist_ok=True)
    raw_dir = out / "raw_completions"
    raw_dir.mkdir(parents=True, exist_ok=True)

    base, tokenizer = load_base_and_tokenizer()
    assert_marker_token(tokenizer)
    assert_postfix_tokenization(tokenizer)  # carrier-correctness gate

    if target == "diagonal":
        target_cols = _diagonal_target_columns()  # 7 judged-rate rows; marker excluded
        delta_key = "delta_source"
        result_name = "patch_recovery_diagonal"

        def out_name_for(s: int) -> str:
            return f"diagonal_source_seed{s}.json"
    else:
        target_cols = _phase2_target_columns()  # v3 off-diagonal leakage map
        delta_key = "delta_leakage"
        result_name = "patch_recovery_postfix"

        def out_name_for(s: int) -> str:
            return f"patch_cells_postfix_seed{s}.json"

    # Diagonal-mode parity precheck (§4.2 part 4): fires ONCE on seed-0, BEFORE
    # the per-row diagonal loop, because the column-keyed in-loop gate below is
    # skipped under diagonal mode. The record is persisted as a one-shot JSON
    # (NOT a diagonal cell) so the backend-parity anchor is auditable.
    if target == "diagonal" and 0 in seeds:
        parity_record = _diagonal_parity_precheck(
            base, tokenizer, probe_cap=probe_cap, device=device, smoke=smoke
        )
        (out / "backend_parity_diagonal_seed0.json").write_text(json.dumps(parity_record, indent=1))
        logger.info("[phase=backend_parity] wrote backend_parity_diagonal_seed0.json")

    parity_done = False

    for seed in seeds:
        patch_cells: dict[str, dict] = {}
        for row in rows:
            adapter_dir = download_adapter(row, seed)
            cfg = _read_adapter_config(adapter_dir)
            gauge, _ = gauge_from_config(cfg)
            lo, hi = expected_gauge_band(row)
            assert lo <= gauge <= hi, f"{row} seed{seed}: gauge {gauge:.2f} outside [{lo},{hi}]"
            model = attach_adapter(base, adapter_dir)
            column_id = target_cols[row]
            col = COLUMNS[column_id]
            probes = battery_probes(col, cap=probe_cap)
            prompts = [render_chat(tokenizer, p["question"], "qwen_default_system") for p in probes]
            gen_kwargs = dict(
                max_new_tokens=col.max_new_tokens,
                n_samples=col.n_samples,
                temperature=col.temperature,
                device=device,
            )

            # Backend-parity assert on bad_medical x broad_em seed-0 (unpatched HF
            # generate), ONCE before any postfix-patch delta is trusted. Under
            # --target diagonal this column-keyed gate never matches (handled by
            # the decoupled precheck above), so it only fires for leakage mode.
            if not parity_done and seed == 0 and row == PARITY_ROW and column_id == PARITY_COLUMN:
                unpatched = generate_patched(base, model, tokenizer, prompts, "none", **gen_kwargs)
                unpatched_rate = _judge_completions(column_id, probes, unpatched)
                delta = abs(unpatched_rate - PARITY_L_545)
                logger.info(
                    "[phase=backend_parity] bad_medical broad_em unpatched-HF rate=%.4f "
                    "(#545 vLLM L=%.4f, |Δ|=%.4f, tol=%.4f, n_probes=%d)",
                    unpatched_rate,
                    PARITY_L_545,
                    delta,
                    PARITY_TOLERANCE_PP,
                    len(probes),
                )
                if not smoke and delta > PARITY_TOLERANCE_PP:
                    raise SystemExit(
                        f"[phase=backend_parity] HALT: unpatched-HF bad_medical broad_em "
                        f"rate={unpatched_rate:.4f} diverges from #545 vLLM L={PARITY_L_545:.4f} "
                        f"by {delta:.4f} > {PARITY_TOLERANCE_PP} (judge noise). HF-vLLM backend "
                        "parity broken — fix decoding params before reading any patch Δ. "
                        "(failure_class: code)"
                    )
                parity_done = True

            trained = generate_patched(base, model, tokenizer, prompts, "none", **gen_kwargs)
            trained_rate = _judge_completions(column_id, probes, trained)
            _persist_raw(raw_dir, row, column_id, f"trained_seed{seed}", probes, trained)

            patched = generate_patched(base, model, tokenizer, prompts, "postfix", **gen_kwargs)
            patched_rate = _judge_completions(column_id, probes, patched)
            _persist_raw(raw_dir, row, column_id, f"postfix_patched_seed{seed}", probes, patched)
            delta = trained_rate - patched_rate
            patch_cells[f"{row}|{column_id}"] = {
                "row": row,
                "column": column_id,
                "seed": seed,
                "patch_kind": "postfix",
                "trained_rate": trained_rate,
                "patched_rate": patched_rate,
                delta_key: delta,
                "n_probes": len(probes),
            }
            logger.info(
                "[phase=postfix_patch] %s seed%d x %s: trained=%.4f patched=%.4f Δ%s=%.4f",
                row,
                seed,
                column_id,
                trained_rate,
                patched_rate,
                "source" if target == "diagonal" else "leak",
                delta,
            )
            base = detach_adapter(model, base)  # B1
            del model
            torch.cuda.empty_cache()

        # Checkpoint-per-phase: persist this seed's cells the moment it completes.
        # Leakage mode keeps the v3 JSON shape byte-for-byte (no extra keys); the
        # diagonal mode adds an explicit "target" tag for the off-pod scorer.
        cells = {k: v[delta_key] for k, v in patch_cells.items()}
        payload = {
            "group": "PST",
            "name": result_name,
            "seed": seed,
            "patch_kind": "postfix",
            "cells": cells,
            "detail": patch_cells,
            "metadata": _metadata(),
        }
        if target == "diagonal":
            payload["target"] = "diagonal"
        (out / out_name_for(seed)).write_text(json.dumps(payload, indent=1))
        logger.info("[phase=postfix_patch] wrote %s", out_name_for(seed))

    del base
    torch.cuda.empty_cache()


# --------------------------------------------------------------------------- #
# Sentinel + raw-completion upload (pod-side contract)
# --------------------------------------------------------------------------- #


def write_sentinel(kind: str = "epm:results", note: dict | None = None) -> None:
    """End-of-run sentinel for poll_pipeline.py (CLAUDE.md pod-side contract)."""
    import time as _t

    logs = Path("/workspace/logs") if Path("/workspace").exists() else output_root() / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    slug = kind.replace(":", "_")
    path = logs / f"issue-{ISSUE}-{slug}-{int(_t.time())}.json"
    path.write_text(
        json.dumps(
            {
                "sentinel_schema_version": 1,
                "kind": kind,
                "version": 1,
                "task_id": ISSUE,
                "by": "issue640_postfix_carrier",
                "ts": int(_t.time()),
                "note": note or {},
            },
            indent=1,
        )
    )
    logger.info("[phase=sentinel] wrote %s", path)


def upload_raw_completions() -> None:
    """Upload raw postfix-patched/trained completions to the HF data repo.

    The driver writes flat per-(row,col,label) JSONs under
    eval_results/issue_640/raw_completions/; walk them explicitly and commit to
    issue640_postfix_carrier/raw_completions/ on the data repo (Upload Policy).
    """
    from explore_persona_space.orchestrate import hub

    raw_dir = output_root() / "raw_completions"
    files = sorted(raw_dir.glob("*.json")) if raw_dir.exists() else []
    if not files:
        logger.info("[phase=upload] no raw completions to upload")
        return
    for f in files:
        hub._upload(
            f,
            repo_id=hub.DEFAULT_DATASET_REPO,
            repo_type="dataset",
            path_in_repo=f"issue640_postfix_carrier/raw_completions/{f.name}",
            upload_as_file=True,
        )
    logger.info("[phase=upload] uploaded %d raw-completion files to data repo", len(files))


def main() -> int:
    parser = argparse.ArgumentParser(description="Issue #640 postfix-carrier sweep driver")
    parser.add_argument(
        "--phase",
        choices=("postfix-kv-shift", "postfix-patch", "all"),
        default="all",
    )
    parser.add_argument("--rows", nargs="+", default=None, help="Row subset (default: all 8)")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--probe-cap", type=int, default=32, help="Phase-2 probes per column")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument(
        "--target",
        choices=("leakage", "diagonal"),
        default="leakage",
        help=(
            "Eval target column (plan v6 §4.2). 'leakage' (default) = v3's off-diagonal "
            "map -> patch_cells_postfix_seed{seed}.json (byte-for-byte v3). 'diagonal' = "
            "each row's on-target diagonal column (7 judged-rate rows, marker excluded) -> "
            "diagonal_source_seed{seed}.json, with the diagonal-mode parity precheck."
        ),
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="rows=bad_medical, seeds=[0], probe-cap=4, run Phase 1->3 in-process serial",
    )
    parser.add_argument("--skip-upload", action="store_true")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    device = "cuda:0"

    if args.smoke:
        if not args.rows:
            args.rows = [PARITY_ROW]
        args.seeds = [0]
        args.probe_cap = 4

    # Diagonal mode is the Phase-2-only amendment (plan v6 §8: Phase 1's
    # postfix-KV-shift is column-independent and already computed in v3 — not
    # re-run). Default rows are the 7 judged-rate diagonal rows (marker excluded).
    if args.target == "diagonal":
        if args.phase == "postfix-kv-shift":
            raise SystemExit(
                "[phase=start] --target diagonal is Phase-2-only (plan v6 §8); Phase 1 "
                "postfix-kv-shift is column-independent and reused from v3. Use "
                "--phase postfix-patch (or the default), not --phase postfix-kv-shift."
            )
        args.phase = "postfix-patch"  # never re-run the column-independent Phase 1
        rows = args.rows or list(PHASE2_ROWS_JUDGED_RATE)
        invalid = [r for r in rows if r not in PHASE2_ROWS_JUDGED_RATE]
        if invalid:
            raise SystemExit(
                f"[phase=start] --target diagonal rows {invalid} are not judged-rate diagonal "
                f"rows; valid rows: {list(PHASE2_ROWS_JUDGED_RATE)} (marker is excluded — its "
                "diagonal is log-prob-scale, read from #545 cell_metadata.json, §4.3)."
            )
    else:
        rows = args.rows or list(ALL_8_ROWS)

    logger.info(
        "[phase=start] issue640 postfix-carrier phase=%s target=%s smoke=%s",
        args.phase,
        args.target,
        args.smoke,
    )
    if args.phase in ("postfix-kv-shift", "all"):
        run_phase1_postfix_kv_shift(rows, args.seeds, device=device)
    if args.phase in ("postfix-patch", "all"):
        run_phase2_postfix_patch(
            rows,
            args.seeds,
            probe_cap=args.probe_cap,
            device=device,
            smoke=args.smoke,
            target=args.target,
        )
        if args.target == "diagonal":
            ref = _marker_diagonal_reference()
            logger.info(
                "[phase=postfix_patch] marker diagonal null reference (NOT a Δsource cell, "
                "read from #545 cell_metadata.json): seed0=%.4f seed137=%.4f nats",
                ref["0"],
                ref["137"],
            )

    if not args.skip_upload:
        upload_raw_completions()

    # Phase 3 (scoring + paired comparison) runs OFF-POD on the VM by default;
    # under --smoke (leakage) run it inline so the smoke exercises the full
    # pipeline. The diagonal selectivity scoring is a separate off-pod script
    # (issue640_diagonal_score.py), run explicitly after the diagonal sweep.
    if args.smoke and args.phase == "all" and args.target == "leakage":
        from issue640_score_and_compare import score_and_compare

        score_and_compare(smoke=True)

    cells_glob = (
        "diagonal_source_seed*.json"
        if args.target == "diagonal"
        else "patch_cells_postfix_seed*.json"
    )
    write_sentinel(
        "epm:results",
        note={
            "phase": args.phase,
            "target": args.target,
            "smoke": args.smoke,
            "predictors_dir": str(predictors_dir()),
            "cells_glob": str(output_root() / cells_glob),
        },
    )
    logger.info("[phase=done] issue640 postfix-carrier complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
