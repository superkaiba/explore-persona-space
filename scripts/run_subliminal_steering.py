"""Issue #267 — Subliminal steering orchestrator.

Resumable phase pipeline. Each phase reads earlier-phase outputs from
``eval_results/issue_267/`` so they can be replayed independently:

* ``--phase 0`` — adapter resolution (HF Hub + WandB).
* ``--phase 1`` — centroid extraction + Phase-1 numerical gates + bridge cells.
* ``--phase 2`` — steered cells (centroid arm sweep, H3, H3', calibrated-c,
                 prompt-on sign-check pair, L10 contrast).
* ``--phase 3`` — analysis (rates, bootstrap-CI, LOO, kill checks).
* ``--phase 4`` — figures (use ``paper-plots`` skill conventions).
* ``--phase all`` — run 0..4 sequentially.

The plan that pre-registers every numerical convention is at
``.claude/plans/issue-267.md``. Do NOT change defaults silently.

Usage::

    uv run python scripts/run_subliminal_steering.py --phase 0
    uv run python scripts/run_subliminal_steering.py --phase 1
    uv run python scripts/run_subliminal_steering.py --phase 2
    ...
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from explore_persona_space.eval.steering import (
    ClusterRateData,
    SteeringHook,
    cluster_bootstrap_delta_spearman,
    cluster_bootstrap_spearman,
    compute_centered_centroids,
    download_adapter,
    extract_centroids_for_personas_at_layers,
    generate_batched,
    loo_spearman,
    make_random_vector,
    marker_substring_rate,
    near_marker_substring_rate,
    pick_attn_implementation,
    spearman_rho,
    wilson_ci,
)
from explore_persona_space.metadata import get_run_metadata
from explore_persona_space.personas import (
    ASSISTANT_PROMPT,
    EVAL_QUESTIONS,
    MARKER_TOKEN,
    PERSONAS,
)

# ---------------------------------------------------------------------------
# Pre-registered constants (plan §1, §4, §5; do not silently change)
# ---------------------------------------------------------------------------

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"

# Headline N=10 personas (B1 fix; helpful_assistant + qwen_default excluded).
HEADLINE_PERSONAS: list[str] = list(PERSONAS.keys())

# 12-persona supplementary set: 10 named + helpful_assistant + qwen_default.
QWEN_DEFAULT_PROMPT = ""  # qwen_default = empty system prompt
HELPFUL_ASSISTANT_PROMPT = ASSISTANT_PROMPT  # "You are a helpful assistant."
SUPPLEMENTARY_PERSONAS: list[str] = ["helpful_assistant", "qwen_default"]
ALL_PERSONAS: list[str] = HEADLINE_PERSONAS + SUPPLEMENTARY_PERSONAS

PERSONA_TO_PROMPT: dict[str, str] = dict(PERSONAS)
PERSONA_TO_PROMPT["helpful_assistant"] = HELPFUL_ASSISTANT_PROMPT
PERSONA_TO_PROMPT["qwen_default"] = QWEN_DEFAULT_PROMPT

# 2 of 12 LoRAs are on HF Hub; the other 10 are on WandB Artifacts.
HF_HUB_PERSONAS: set[str] = {"helpful_assistant", "qwen_default"}

# Layers (plan §4.4): L20 binding; L10 contrast required-conditional.
# L25 was previously extracted "for follow-ups" but is unplanned scope creep —
# follow-up issues that need it can re-extract.
HEADLINE_LAYER = 20
L10_LAYER = 10
ALL_LAYERS = (L10_LAYER, HEADLINE_LAYER)

# Calibration prompt for §11.17 / §4.3 baseline-norm measurement.
CALIBRATION_PROMPT = "What causes earthquakes?"

# Neutral prompt (B1 fix).
# Hot-fix 2026-05-07: original `"Please answer the user's question."` failed §8 #2
# (max|cos|=0.570 > 0.5 against data_scientist) AND §8 #3 (cos≈0.78 vs both helpful_assistant
# and qwen_default, > 0.7 halt). Swapped to plan §4.4 #2 / §8 first listed alternative.
NEUTRAL_PROMPT = "Provide a clear answer."
# Backup neutral prompts to swap to if §8 #2 axis-distance gate fires.
NEUTRAL_PROMPT_ALTERNATIVES = ["Please answer the user's question.", "Reply directly."]

# Coefficient grid (plan §4.3).
COEFFS_POSITIVE: list[float] = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0]
COEFFS_NEGATIVE: list[float] = [-2.0, -1.0]
HEADLINE_COEFF_DEFAULT: float = 2.0  # B3c pre-registered; may be MR5-recalibrated

# n=100 = 20 EVAL_QUESTIONS × 5 completions.
NUM_COMPLETIONS = 5
MAX_NEW_TOKENS = 2048
TEMPERATURE = 1.0
TOP_P = 0.95
SEED = 42

# Bootstrap (m-r2-3 cluster bootstrap; plan §7).
BOOTSTRAP_ITER = 10000
BOOTSTRAP_SEED = 2604

# Phase-1 perturbation-ratio gate (plan §8 #5 / §11.17).
PERTURBATION_RATIO_TARGET = 0.4
PERTURBATION_RATIO_LOW = 0.2
PERTURBATION_RATIO_HIGH = 0.6
L20_NORM_RANGE = (50.0, 300.0)
NEUTRAL_AXIS_DISTANCE_GATE = 0.5
NEUTRAL_AXIS_DISTANCE_HALT = 0.7  # for the N=12 supplementary check (§8 #3)
H2_GRADIENT_HALT = 0.2
H2_GRADIENT_WARN = 0.4

# §8 #6 BF16 numerical equivalence threshold.
BF16_EQUIVALENCE_TOL = 5e-3

DEFAULT_OUTPUT_ROOT = Path("eval_results/issue_267")
DEFAULT_ADAPTER_ROOT = Path("/workspace/adapters")

logger = logging.getLogger("issue267")


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)8s | %(name)s | %(message)s",
        force=True,
    )


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2, default=str)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def _attach_metadata(payload: dict[str, Any]) -> dict[str, Any]:
    payload.setdefault("metadata", get_run_metadata())
    return payload


def _free_gpu() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Phase 0 — adapter resolution
# ---------------------------------------------------------------------------


def phase0_resolve_adapters(out_dir: Path, adapter_root: Path) -> dict[str, Any]:
    """Download all 12 LoRA adapters; pin versions; write manifest."""
    logger.info("[phase0] resolving adapters → %s", adapter_root)
    manifest: dict[str, Any] = {"personas": {}}
    for persona in ALL_PERSONAS:
        source = "hf_hub" if persona in HF_HUB_PERSONAS else "wandb"
        logger.info("[phase0] %s (%s)", persona, source)
        resolved = download_adapter(persona=persona, source=source, out_root=adapter_root)
        manifest["personas"][persona] = {
            "source": resolved.source,
            "local_dir": str(resolved.local_dir),
            "wandb_artifact": resolved.artifact_qualified_name,
            "wandb_version": resolved.version,
            "hf_repo_id": resolved.repo_id,
        }
    out_path = out_dir / "adapter_manifest.json"
    _save_json(out_path, _attach_metadata(manifest))
    logger.info("[phase0] wrote %s", out_path)
    return manifest


# ---------------------------------------------------------------------------
# Phase 1 — centroid extraction + numerical gates + bridge cells
# ---------------------------------------------------------------------------


def _l20_residual_norms(
    model, tokenizer, system_prompt: str, questions: Sequence[str]
) -> tuple[float, list[float]]:
    """Mean L20 residual norm at the assistant-start position over the question set.

    Used by the §8 #5 perturbation-ratio + §11.17 norm gates. Returns
    ``(mean_norm, per_question_norms)``.
    """
    norms: list[float] = []
    captured: dict[str, torch.Tensor] = {}

    def hook_fn(_module, _inputs, output):
        captured["hs"] = (output[0] if isinstance(output, tuple) else output).detach()

    handle = model.model.layers[HEADLINE_LAYER].register_forward_hook(hook_fn)
    try:
        model.eval()
        with torch.no_grad():
            for q in questions:
                msgs = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": q},
                ]
                text = tokenizer.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True
                )
                inputs = tokenizer(text, return_tensors="pt").to(model.device)
                model(**inputs)
                last_pos = int(inputs["attention_mask"][0].sum().item()) - 1
                norms.append(captured["hs"][0, last_pos, :].float().norm().item())
    finally:
        handle.remove()
    return float(np.mean(norms)), norms


def _flatten_completions_grouped(
    completions: list[str], n_questions: int, n_completions: int
) -> list[list[str]]:
    """HF generate emits ``num_return_sequences`` consecutive rows per prompt.

    We persist the raw flat list AND the grouped-by-question structure so the
    cluster bootstrap can resample question clusters.
    """
    if len(completions) != n_questions * n_completions:
        raise ValueError(
            f"expected {n_questions * n_completions} completions, got {len(completions)}"
        )
    grouped: list[list[str]] = []
    for q_idx in range(n_questions):
        start = q_idx * n_completions
        end = start + n_completions
        grouped.append(completions[start:end])
    return grouped


def phase1_extract_centroids(
    out_dir: Path,
    adapter_manifest: dict[str, Any],
    questions: Sequence[str],
    *,
    skip_bridge: bool = False,
    bridge_only_persona: str | None = None,
) -> dict[str, Any]:
    """Extract centroids at L10/L20; run §8 numerical gates; produce bridge cells.

    Outputs (under ``out_dir``):

    * ``centroids_l10_n10.pt``, ``centroids_l20_n10.pt``
    * ``centroids_l20_n12_supplementary.pt``
    * ``centroid_pin.json``, ``neutral_prompt_axis_check.json``,
      ``coefficient_calibration.json``, ``registered_coefficient.json``
    * ``bridge_completions.json``, ``bridge_rates.json``
    * ``cosines_l20_n10.json``, ``cosines_l10_n10.json``
    * ``phase1_diagnostics.json``
    """
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info("[phase1] loading base model %s", BASE_MODEL)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        attn_implementation=pick_attn_implementation(),
        trust_remote_code=True,
    )

    # ---- centroid extraction (single pass, all layers) -------------------
    system_prompts = {p: PERSONA_TO_PROMPT[p] for p in ALL_PERSONAS}
    system_prompts["__neutral__"] = NEUTRAL_PROMPT
    raw_centroids = extract_centroids_for_personas_at_layers(
        base_model,
        tokenizer,
        layers=ALL_LAYERS,
        system_prompts=system_prompts,
        questions=questions,
    )

    centroids_by_layer: dict[int, dict[str, Any]] = {}
    cosines_by_layer: dict[int, dict[str, float]] = {}
    for layer in ALL_LAYERS:
        raw_l = raw_centroids[layer]
        # Headline N=10 centering
        centered_n10, mean_n10 = compute_centered_centroids(raw_l, HEADLINE_PERSONAS)
        # N=12 supplementary mean (for the §8 #3 sub-check)
        _, mean_n12 = compute_centered_centroids(raw_l, ALL_PERSONAS)

        # Cosines under the N=10 centering (every persona projected in)
        # cos(centered[p], centered[helpful_assistant])
        ha_centered = centered_n10["helpful_assistant"]
        cosines: dict[str, float] = {}
        for p in [*ALL_PERSONAS, "__neutral__"]:
            v = centered_n10[p]
            cos = torch.dot(v, ha_centered).item() / (
                v.norm().item() * ha_centered.norm().item() + 1e-12
            )
            cosines[p] = cos
        cosines_by_layer[layer] = cosines

        centroids_by_layer[layer] = {
            "raw": {p: t.tolist() for p, t in raw_l.items()},
            "centered_n10": {p: t.tolist() for p, t in centered_n10.items()},
            "mean_n10": mean_n10.tolist(),
            "mean_n12": mean_n12.tolist(),
        }

        torch.save(
            {
                "raw": raw_l,
                "centered_n10": centered_n10,
                "mean_n10": mean_n10,
                "mean_n12": mean_n12,
                "personas_n10": HEADLINE_PERSONAS,
                "personas_n12": ALL_PERSONAS,
            },
            out_dir / f"centroids_l{layer}_n10.pt",
        )

    # Supplementary file with the on-axis pair under N=10 centering.
    torch.save(
        {
            "centered_n10_projected_onaxis_pair": {
                p: centroids_by_layer[HEADLINE_LAYER]["centered_n10"][p]
                for p in SUPPLEMENTARY_PERSONAS
            }
        },
        out_dir / "centroids_l20_n12_supplementary.pt",
    )

    _save_json(out_dir / "cosines_l20_n10.json", cosines_by_layer[HEADLINE_LAYER])
    _save_json(out_dir / "cosines_l10_n10.json", cosines_by_layer[L10_LAYER])

    # ---- §8 #1 / §4.4 #5 self-consistency pin ----------------------------
    # First run: write centroid_pin.json with cos(librarian, villain). Any
    # subsequent Phase-1 run must reproduce the cosine within ±0.001 — if it
    # drifts by more, the centering math has changed (or RNG/numerical leak)
    # and we halt before downstream steering becomes invalid.
    centered_l20_n10 = {
        p: torch.tensor(v) for p, v in centroids_by_layer[HEADLINE_LAYER]["centered_n10"].items()
    }
    librarian = centered_l20_n10["librarian"]
    villain = centered_l20_n10["villain"]
    cos_lv = torch.dot(librarian, villain).item() / (
        librarian.norm().item() * villain.norm().item() + 1e-12
    )
    pin_path = out_dir / "centroid_pin.json"
    pin_tolerance = 0.001
    if pin_path.exists():
        prior = _load_json(pin_path)
        prior_cos = float(prior["cos_librarian_villain_n10_l20"])
        drift = abs(cos_lv - prior_cos)
        if drift > pin_tolerance:
            raise RuntimeError(
                f"§8 #1 / §4.4 #5 self-pin failed: "
                f"cos(librarian, villain) drifted by {drift:.6f} "
                f"(prior={prior_cos:.6f}, current={cos_lv:.6f}, tol=±{pin_tolerance:.3f}). "
                "The N=10 centering math has changed since the first Phase-1 run."
            )
        logger.info("[phase1] §8 #1 self-pin OK: cos drift %.6f ≤ %.3f", drift, pin_tolerance)
    else:
        _save_json(
            pin_path,
            _attach_metadata(
                {
                    "cos_librarian_villain_n10_l20": cos_lv,
                    "tolerance": pin_tolerance,
                    "note": "Re-execution-stability pin per §4.4 #5 / §8 #1.",
                }
            ),
        )

    # ---- §8 #2 + #3 neutral-prompt axis-distance gates -------------------
    neutral_l20_centered = centered_l20_n10["__neutral__"]
    n10_check: dict[str, float] = {}
    for p in HEADLINE_PERSONAS:
        v = centered_l20_n10[p]
        cos = torch.dot(neutral_l20_centered, v).item() / (
            neutral_l20_centered.norm().item() * v.norm().item() + 1e-12
        )
        n10_check[p] = cos
    n10_max_abs = max(abs(c) for c in n10_check.values())
    n10_gate_pass = n10_max_abs <= NEUTRAL_AXIS_DISTANCE_GATE

    # N=12 supplementary check: project neutral against N=12 mean
    raw_l20 = {p: torch.tensor(v) for p, v in centroids_by_layer[HEADLINE_LAYER]["raw"].items()}
    n12_mean = torch.tensor(centroids_by_layer[HEADLINE_LAYER]["mean_n12"])
    neutral_n12_centered = raw_l20["__neutral__"] - n12_mean
    on_axis_check: dict[str, float] = {}
    for p in SUPPLEMENTARY_PERSONAS:
        v = raw_l20[p] - n12_mean
        cos = torch.dot(neutral_n12_centered, v).item() / (
            neutral_n12_centered.norm().item() * v.norm().item() + 1e-12
        )
        on_axis_check[p] = cos
    n12_max_abs = max(abs(c) for c in on_axis_check.values())
    n12_gate_halt = n12_max_abs > NEUTRAL_AXIS_DISTANCE_HALT

    _save_json(
        out_dir / "neutral_prompt_axis_check.json",
        _attach_metadata(
            {
                "neutral_prompt": NEUTRAL_PROMPT,
                "n10_centering_check": n10_check,
                "n10_max_abs_cos": n10_max_abs,
                "n10_gate_pass": n10_gate_pass,
                "n12_supplementary_check": on_axis_check,
                "n12_max_abs_cos": n12_max_abs,
                "n12_gate_halt": n12_gate_halt,
                "fallback_alternatives": NEUTRAL_PROMPT_ALTERNATIVES,
            }
        ),
    )
    if not n10_gate_pass:
        raise RuntimeError(
            f"§8 #2 gate FAILED: max |cos(neutral, p)| = {n10_max_abs:.3f} > "
            f"{NEUTRAL_AXIS_DISTANCE_GATE}. Swap to alternative neutral prompt."
        )
    if n12_gate_halt:
        raise RuntimeError(
            f"§8 #3 gate FAILED: max |cos(neutral, on_axis_pair)| under N=12 = "
            f"{n12_max_abs:.3f} > {NEUTRAL_AXIS_DISTANCE_HALT}. Halt before Phase 2."
        )

    # ---- §11.17 / §8 #5 L20 baseline norm + perturbation ratio -----------
    base_norm_mean, base_norm_per_q = _l20_residual_norms(
        base_model, tokenizer, NEUTRAL_PROMPT, questions
    )
    if not (L20_NORM_RANGE[0] <= base_norm_mean <= L20_NORM_RANGE[1]):
        raise RuntimeError(
            f"§8 #5 / §11.17 gate: L20 mean residual norm {base_norm_mean:.2f} "
            f"outside expected range {L20_NORM_RANGE}."
        )

    centroid_norms = {
        p: float(torch.tensor(centroids_by_layer[HEADLINE_LAYER]["centered_n10"][p]).norm().item())
        for p in HEADLINE_PERSONAS
    }
    ratios_at_c2 = {p: 2.0 * centroid_norms[p] / base_norm_mean for p in HEADLINE_PERSONAS}
    median_ratio = float(np.median(list(ratios_at_c2.values())))
    if PERTURBATION_RATIO_LOW <= median_ratio <= PERTURBATION_RATIO_HIGH:
        registered_c = HEADLINE_COEFF_DEFAULT
        recalibrated = False
    else:
        registered_c = HEADLINE_COEFF_DEFAULT * (PERTURBATION_RATIO_TARGET / median_ratio)
        recalibrated = True

    # Per-persona M5 calibrated coefficients targeting ratio ≈ 0.20.
    c_calibrated = {
        p: PERTURBATION_RATIO_TARGET * 0.5 * base_norm_mean / centroid_norms[p]
        for p in HEADLINE_PERSONAS
    }
    # (0.4 * 0.5 = 0.20 — keep the constant explicit so the rationale travels.)

    _save_json(
        out_dir / "coefficient_calibration.json",
        _attach_metadata(
            {
                "calibration_prompt": CALIBRATION_PROMPT,
                "neutral_prompt": NEUTRAL_PROMPT,
                "l20_baseline_norm_mean_neutral": base_norm_mean,
                "l20_baseline_norm_per_question_neutral": base_norm_per_q,
                "centroid_norms_n10": centroid_norms,
                "perturbation_ratios_at_c2": ratios_at_c2,
                "median_ratio_at_c2": median_ratio,
                "perturbation_ratio_target": PERTURBATION_RATIO_TARGET,
                "perturbation_ratio_band": [
                    PERTURBATION_RATIO_LOW,
                    PERTURBATION_RATIO_HIGH,
                ],
                "registered_coefficient": registered_c,
                "registered_coefficient_was_recalibrated": recalibrated,
                "c_calibrated_per_persona": c_calibrated,
            }
        ),
    )
    _save_json(
        out_dir / "registered_coefficient.json",
        _attach_metadata(
            {
                "registered_headline_coefficient": registered_c,
                "default_was_used": not recalibrated,
                "recalibrated_from": HEADLINE_COEFF_DEFAULT if recalibrated else None,
                "median_ratio_at_default_c": median_ratio,
                "rule": "MR5: scale c so median r_p ≈ 0.4 if outside [0.2, 0.6].",
            }
        ),
    )

    del base_model
    _free_gpu()

    # ---- Bridge cells (B2 fix): prompted-on-HF for all 12 personas -------
    if skip_bridge:
        logger.info("[phase1] --skip-bridge: skipping bridge cells")
    else:
        logger.info("[phase1] running bridge cells (prompted-on-HF, no steering)")
        bridge_completions: dict[str, list[list[str]]] = {}
        bridge_rates: dict[str, dict[str, float]] = {}
        bridge_personas = [bridge_only_persona] if bridge_only_persona is not None else ALL_PERSONAS
        for persona in bridge_personas:
            t0 = time.time()
            adapter_dir = adapter_manifest["personas"][persona]["local_dir"]
            from transformers import AutoModelForCausalLM as _AM

            base = _AM.from_pretrained(
                BASE_MODEL,
                torch_dtype=torch.bfloat16,
                device_map={"": 0},
                attn_implementation=pick_attn_implementation(),
                trust_remote_code=True,
            )
            lora = PeftModel.from_pretrained(base, adapter_dir)
            merged = lora.merge_and_unload()
            merged.eval()
            comps = generate_batched(
                merged,
                tokenizer,
                system_prompt=PERSONA_TO_PROMPT[persona],
                questions=questions,
                num_completions=NUM_COMPLETIONS,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                seed=SEED,
            )
            grouped = _flatten_completions_grouped(comps, len(questions), NUM_COMPLETIONS)
            bridge_completions[persona] = grouped
            found, total = marker_substring_rate(comps)
            lo, hi = wilson_ci(found, total)
            bridge_rates[persona] = {
                "rate": found / total if total > 0 else 0.0,
                "found": found,
                "total": total,
                "wilson_ci_low": lo,
                "wilson_ci_high": hi,
                "wall_seconds": time.time() - t0,
            }
            del merged, lora, base
            _free_gpu()
            logger.info(
                "[phase1] bridge %s: rate=%.3f (%d/%d, %.1fs)",
                persona,
                bridge_rates[persona]["rate"],
                found,
                total,
                bridge_rates[persona]["wall_seconds"],
            )
        _save_json(
            out_dir / "bridge_completions.json",
            _attach_metadata({"completions": bridge_completions}),
        )
        _save_json(out_dir / "bridge_rates.json", _attach_metadata(bridge_rates))

    # ---- §8 #4 H2-gradient sanity (only if bridge ran on full N=10) ----
    h2_grad_ok = None
    if not skip_bridge and bridge_only_persona is None:
        rates_n10 = [bridge_rates[p]["rate"] for p in HEADLINE_PERSONAS]
        cos_l20_n10 = [cosines_by_layer[HEADLINE_LAYER][p] for p in HEADLINE_PERSONAS]
        rho_h2_grad = spearman_rho(rates_n10, cos_l20_n10)
        h2_grad_ok = abs(rho_h2_grad) >= H2_GRADIENT_HALT
        _save_json(
            out_dir / "h2_prompted_sanity.json",
            _attach_metadata(
                {
                    "rho_bridge_vs_l20_cosine_n10": rho_h2_grad,
                    "halt_threshold": H2_GRADIENT_HALT,
                    "warn_threshold": H2_GRADIENT_WARN,
                    "halt_fired": not h2_grad_ok,
                }
            ),
        )
        if not h2_grad_ok:
            raise RuntimeError(
                f"§8 #4 gate: |ρ(bridge, l20_cos)| = {rho_h2_grad:.3f} < {H2_GRADIENT_HALT}; "
                "halt before Phase 2."
            )

    diagnostics = {
        "n10_axis_distance_max_abs": n10_max_abs,
        "n12_axis_distance_max_abs": n12_max_abs,
        "l20_baseline_norm_mean": base_norm_mean,
        "median_perturbation_ratio_at_c2": median_ratio,
        "registered_coefficient": registered_c,
        "h2_gradient_ok": h2_grad_ok,
    }
    _save_json(out_dir / "phase1_diagnostics.json", _attach_metadata(diagnostics))
    logger.info("[phase1] diagnostics: %s", diagnostics)
    return diagnostics


# ---------------------------------------------------------------------------
# Phase 2 — steered cells
# ---------------------------------------------------------------------------


def _load_centroid_l(layer: int, out_dir: Path) -> dict[str, torch.Tensor]:
    pt = torch.load(out_dir / f"centroids_l{layer}_n10.pt", weights_only=False)
    return pt["centered_n10"]


def _equivalence_check(
    merged,
    tokenizer,
    direction: torch.Tensor,
    coefficient: float,
    persona: str,
) -> dict[str, Any]:
    """§8 #6 batched-vs-sequential numerical equivalence + temp=0 string match.

    Captures the **post-steering** L20 hidden state for prompt 0 in two
    configurations: (a) batched over 20 EVAL_QUESTIONS, (b) sequential (single
    prompt). Returns max-abs deviation. Also generates one temp=0 completion in
    each config and checks string equality.

    Implementation note: PyTorch fires forward hooks in registration order, so
    we register ``SteeringHook`` FIRST and the capture hook SECOND — the
    capture then sees the residual after the steering bias has been added.
    """
    captured: dict[str, torch.Tensor] = {}

    def hook_fn(_module, _inputs, output):
        captured.setdefault("hs_list", []).append(
            (output[0] if isinstance(output, tuple) else output).detach().cpu()
        )

    sh = SteeringHook(
        merged, layer_idx=HEADLINE_LAYER, direction=direction, coefficient=coefficient
    )
    handle = merged.model.layers[HEADLINE_LAYER].register_forward_hook(hook_fn)
    try:
        # (a) batched: the 20 EVAL_QUESTIONS
        captured["hs_list"] = []
        prompt_texts = [
            tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": NEUTRAL_PROMPT},
                    {"role": "user", "content": q},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
            for q in EVAL_QUESTIONS
        ]
        saved_pad = tokenizer.padding_side
        try:
            tokenizer.padding_side = "left"
            inputs = tokenizer(prompt_texts, return_tensors="pt", padding=True).to(merged.device)
            with torch.no_grad():
                merged(**inputs)
        finally:
            tokenizer.padding_side = saved_pad
        # find prompt 0's last position in the left-padded batch
        attn_0 = inputs["attention_mask"][0]
        # left-pad: real tokens are at the end; last real token is at seq-1.
        batch_pos = attn_0.shape[0] - 1
        batched_hs = captured["hs_list"][-1][0, batch_pos, :].float()

        # (b) sequential
        captured["hs_list"] = []
        seq_inputs = tokenizer([prompt_texts[0]], return_tensors="pt", padding=False).to(
            merged.device
        )
        with torch.no_grad():
            merged(**seq_inputs)
        seq_pos = int(seq_inputs["attention_mask"][0].sum().item()) - 1
        seq_hs = captured["hs_list"][-1][0, seq_pos, :].float()

        max_abs = (batched_hs - seq_hs).abs().max().item()
    finally:
        # Inverse of registration order: capture hook first, then SteeringHook.
        handle.remove()
        sh.remove()

    # Temp=0 string comparison on prompt 0
    saved_pad = tokenizer.padding_side
    try:
        tokenizer.padding_side = "left"
        with SteeringHook(
            merged, layer_idx=HEADLINE_LAYER, direction=direction, coefficient=coefficient
        ):
            comps = generate_batched(
                merged,
                tokenizer,
                system_prompt=NEUTRAL_PROMPT,
                questions=[EVAL_QUESTIONS[0]],
                num_completions=1,
                max_new_tokens=64,
                temperature=1.0,
                top_p=1.0,
                seed=SEED,
                do_sample=False,
            )
            batched_str = comps[0]
        with SteeringHook(
            merged, layer_idx=HEADLINE_LAYER, direction=direction, coefficient=coefficient
        ):
            comps2 = generate_batched(
                merged,
                tokenizer,
                system_prompt=NEUTRAL_PROMPT,
                questions=[EVAL_QUESTIONS[0]],
                num_completions=1,
                max_new_tokens=64,
                temperature=1.0,
                top_p=1.0,
                seed=SEED,
                do_sample=False,
            )
            seq_str = comps2[0]
    finally:
        tokenizer.padding_side = saved_pad

    return {
        "persona": persona,
        "coefficient": coefficient,
        "max_abs_l20_deviation": max_abs,
        "batched_string": batched_str,
        "sequential_string": seq_str,
        "tol": BF16_EQUIVALENCE_TOL,
        "passes": max_abs <= BF16_EQUIVALENCE_TOL and batched_str == seq_str,
    }


def _persona_pass(
    persona: str,
    *,
    out_dir: Path,
    adapter_manifest: dict[str, Any],
    questions: Sequence[str],
    headline_coefficient: float,
    c_calibrated: dict[str, float],
    centered_l20_n10: dict[str, torch.Tensor],
    centered_l10_n10: dict[str, torch.Tensor],
    do_l10_contrast: bool,
) -> dict[str, Any]:
    """Run all in-persona Phase-2 cells; return raw completions + timing.

    The §8 numerical-equivalence (#6) and deterministic-replay (#7) gates run
    in :func:`_run_phase2_gates` BEFORE this function — they must pass before
    any persona-loop cell launches.
    """
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        attn_implementation=pick_attn_implementation(),
        trust_remote_code=True,
    )
    adapter_dir = adapter_manifest["personas"][persona]["local_dir"]
    lora = PeftModel.from_pretrained(base, adapter_dir)
    merged = lora.merge_and_unload()
    merged.eval()

    direction_l20 = centered_l20_n10[persona].clone()
    direction_l20_norm = float(direction_l20.norm().item())

    out_persona: dict[str, Any] = {"persona": persona, "cells": {}, "timings": {}}

    # Bind `merged` and `tokenizer` into the closures via default args so ruff's
    # static analyser doesn't flag them as unbound (Python's closure semantics
    # work fine here at runtime — this is a static-analysis ergonomic).
    def _gen(
        *,
        system_prompt: str,
        coefficient: float,
        layer: int,
        direction: torch.Tensor,
        _merged=merged,
        _tokenizer=tokenizer,
    ) -> list[str]:
        torch.manual_seed(SEED)
        with SteeringHook(_merged, layer_idx=layer, direction=direction, coefficient=coefficient):
            return generate_batched(
                _merged,
                _tokenizer,
                system_prompt=system_prompt,
                questions=questions,
                num_completions=NUM_COMPLETIONS,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                seed=SEED,
            )

    def _gen_no_hook(*, system_prompt: str, _merged=merged, _tokenizer=tokenizer) -> list[str]:
        torch.manual_seed(SEED)
        return generate_batched(
            _merged,
            _tokenizer,
            system_prompt=system_prompt,
            questions=questions,
            num_completions=NUM_COMPLETIONS,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            seed=SEED,
        )

    cells = out_persona["cells"]

    # ---- Centroid arm: positive sweep + negative coefficients -----------
    coeffs_centroid: list[float] = list(COEFFS_POSITIVE) + list(COEFFS_NEGATIVE)
    # Ensure the headline coefficient is in the list (may have been recalibrated).
    if headline_coefficient not in coeffs_centroid:
        coeffs_centroid.append(headline_coefficient)
    for coeff in coeffs_centroid:
        t0 = time.time()
        if coeff == 0.0:
            comps = _gen_no_hook(system_prompt=NEUTRAL_PROMPT)
        else:
            comps = _gen(
                system_prompt=NEUTRAL_PROMPT,
                coefficient=coeff,
                layer=HEADLINE_LAYER,
                direction=direction_l20,
            )
        cells.setdefault("centroid", {})[f"{coeff}"] = _flatten_completions_grouped(
            comps, len(questions), NUM_COMPLETIONS
        )
        out_persona["timings"][f"centroid_c={coeff}"] = time.time() - t0

    # ---- H3 isotropic random arm at headline coefficient ----------------
    iso_vec = make_random_vector(
        kind="isotropic",
        persona=persona,
        target_norm=direction_l20_norm,
        hidden_dim=direction_l20.shape[0],
    )
    t0 = time.time()
    comps = _gen(
        system_prompt=NEUTRAL_PROMPT,
        coefficient=headline_coefficient,
        layer=HEADLINE_LAYER,
        direction=iso_vec,
    )
    cells["random_iso"] = {
        f"{headline_coefficient}": _flatten_completions_grouped(
            comps, len(questions), NUM_COMPLETIONS
        )
    }
    out_persona["timings"]["random_iso"] = time.time() - t0

    # ---- H3' in-subspace random arm at headline coefficient -------------
    in_sub_vec = make_random_vector(
        kind="in_subspace",
        persona=persona,
        target_norm=direction_l20_norm,
        centered_centroids=centered_l20_n10,
        headline_personas=HEADLINE_PERSONAS,
    )
    t0 = time.time()
    comps = _gen(
        system_prompt=NEUTRAL_PROMPT,
        coefficient=headline_coefficient,
        layer=HEADLINE_LAYER,
        direction=in_sub_vec,
    )
    cells["random_in_subspace"] = {
        f"{headline_coefficient}": _flatten_completions_grouped(
            comps, len(questions), NUM_COMPLETIONS
        )
    }
    out_persona["timings"]["random_in_subspace"] = time.time() - t0

    # ---- H1' calibrated-c co-primary -----------------------------------
    c_cal = c_calibrated.get(persona)
    if c_cal is not None:
        t0 = time.time()
        comps = _gen(
            system_prompt=NEUTRAL_PROMPT,
            coefficient=c_cal,
            layer=HEADLINE_LAYER,
            direction=direction_l20,
        )
        cells["centroid_calibrated"] = {
            f"{c_cal}": _flatten_completions_grouped(comps, len(questions), NUM_COMPLETIONS)
        }
        out_persona["timings"]["centroid_calibrated"] = time.time() - t0

    # ---- L10 contrast --------------------------------------------------
    if do_l10_contrast:
        direction_l10 = centered_l10_n10[persona].clone()
        t0 = time.time()
        comps = _gen(
            system_prompt=NEUTRAL_PROMPT,
            coefficient=headline_coefficient,
            layer=L10_LAYER,
            direction=direction_l10,
        )
        cells["centroid_L10"] = {
            f"{headline_coefficient}": _flatten_completions_grouped(
                comps, len(questions), NUM_COMPLETIONS
            )
        }
        out_persona["timings"]["centroid_L10"] = time.time() - t0

    # ---- M2 + MR9 sign-check pair (prompt-on at c=0 AND c=+headline) ---
    t0 = time.time()
    comps = _gen_no_hook(system_prompt=PERSONA_TO_PROMPT[persona])
    cells["prompt_on"] = {
        "0.0": _flatten_completions_grouped(comps, len(questions), NUM_COMPLETIONS)
    }
    out_persona["timings"]["prompt_on_c=0"] = time.time() - t0
    t0 = time.time()
    comps = _gen(
        system_prompt=PERSONA_TO_PROMPT[persona],
        coefficient=headline_coefficient,
        layer=HEADLINE_LAYER,
        direction=direction_l20,
    )
    cells["prompt_on"][f"{headline_coefficient}"] = _flatten_completions_grouped(
        comps, len(questions), NUM_COMPLETIONS
    )
    out_persona["timings"][f"prompt_on_c={headline_coefficient}"] = time.time() - t0

    del merged, lora, base
    _free_gpu()
    return out_persona


def _software_engineer_headline_completions(
    *,
    out_dir: Path,
    adapter_manifest: dict[str, Any],
    questions: Sequence[str],
    headline_coefficient: float,
    centered_l20_n10: dict[str, torch.Tensor],
) -> list[str]:
    """Generate the software_engineer headline c=2.0 cell once (centroid arm).

    Used by both the §8 #6 / #7 pre-loop gates and the §8 m1 iter-1 vs iter-12
    spot-check. Returns the flat completion list (n_questions * n_completions)
    in HF generate row-order.
    """
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        attn_implementation=pick_attn_implementation(),
        trust_remote_code=True,
    )
    adapter_dir = adapter_manifest["personas"]["software_engineer"]["local_dir"]
    lora = PeftModel.from_pretrained(base, adapter_dir)
    merged = lora.merge_and_unload()
    merged.eval()
    direction_l20 = centered_l20_n10["software_engineer"].clone()

    torch.manual_seed(SEED)
    with SteeringHook(
        merged,
        layer_idx=HEADLINE_LAYER,
        direction=direction_l20,
        coefficient=headline_coefficient,
    ):
        comps = generate_batched(
            merged,
            tokenizer,
            system_prompt=NEUTRAL_PROMPT,
            questions=questions,
            num_completions=NUM_COMPLETIONS,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            seed=SEED,
        )
    del merged, lora, base
    _free_gpu()
    return comps


def _run_phase2_gates(
    out_dir: Path,
    adapter_manifest: dict[str, Any],
    centered_l20_n10: dict[str, torch.Tensor],
    headline_coefficient: float,
) -> dict[str, Any]:
    """Run §8 gates #6 (batched-vs-sequential equivalence) + #7 (deterministic
    replay at c=0.0) BEFORE any persona-loop cell launches.

    Both gates raise ``RuntimeError`` on failure so Phase 2 cannot proceed.
    The intent: NO non-software_engineer cells run until both gates pass. We
    deliberately load + drop the ``software_engineer`` merged model here even
    though Phase 2 will reload it later — the cost is one extra ~30 s load.
    """
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info("[phase2] §8 #6 + #7 gates (software_engineer)")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        attn_implementation=pick_attn_implementation(),
        trust_remote_code=True,
    )
    adapter_dir = adapter_manifest["personas"]["software_engineer"]["local_dir"]
    lora = PeftModel.from_pretrained(base, adapter_dir)
    merged = lora.merge_and_unload()
    merged.eval()
    direction_l20 = centered_l20_n10["software_engineer"].clone()

    # §8 #6: batched-vs-sequential equivalence at c=2.0 AND c=0.0.
    eq_c2 = _equivalence_check(
        merged, tokenizer, direction_l20, headline_coefficient, "software_engineer"
    )
    eq_c0 = _equivalence_check(merged, tokenizer, direction_l20, 0.0, "software_engineer")

    # §8 #7: deterministic replay at coeff=0.0 (NO steering hook attached).
    # Plan: byte-identical completion lists across two consecutive runs.
    # Bind merged/tokenizer via default args so ruff's static analyser doesn't
    # flag them as unbound (Python's closure semantics work fine at run-time).
    def _replay_no_hook(_merged=merged, _tokenizer=tokenizer) -> list[str]:
        torch.manual_seed(SEED)
        return generate_batched(
            _merged,
            _tokenizer,
            system_prompt=NEUTRAL_PROMPT,
            questions=EVAL_QUESTIONS,
            num_completions=NUM_COMPLETIONS,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            seed=SEED,
        )

    replay_a = _replay_no_hook()
    replay_b = _replay_no_hook()
    replay_byte_identical = replay_a == replay_b

    del merged, lora, base
    _free_gpu()

    diagnostics = {
        "equivalence_check_c0": eq_c0,
        "equivalence_check_c2": eq_c2,
        "replay_c0_byte_identical": replay_byte_identical,
        "replay_c0_replay_a_first_chars": (replay_a[0][:120] if replay_a else ""),
        "replay_c0_replay_b_first_chars": (replay_b[0][:120] if replay_b else ""),
    }
    _save_json(out_dir / "phase2_diagnostics.json", _attach_metadata(diagnostics))

    if not eq_c2["passes"]:
        raise RuntimeError(
            "§8 gate #6 failed at c=2.0: "
            f"max_abs_l20_dev={eq_c2['max_abs_l20_deviation']:.2e} "
            f"(tol={BF16_EQUIVALENCE_TOL:.0e}); "
            f"temp=0 string match={eq_c2['batched_string'] == eq_c2['sequential_string']}. "
            "Halt before Phase 2 persona loop."
        )
    if not eq_c0["passes"]:
        raise RuntimeError(
            "§8 gate #6 failed at c=0.0: "
            f"max_abs_l20_dev={eq_c0['max_abs_l20_deviation']:.2e} "
            f"(tol={BF16_EQUIVALENCE_TOL:.0e}); "
            f"temp=0 string match={eq_c0['batched_string'] == eq_c0['sequential_string']}. "
            "Halt before Phase 2 persona loop."
        )
    if not replay_byte_identical:
        raise RuntimeError(
            "§8 gate #7 failed (deterministic replay at coeff=0.0): "
            "two consecutive runs produced non-identical completion lists. "
            "Hidden state leak — halt before Phase 2 persona loop."
        )
    logger.info(
        "[phase2] §8 #6 OK (max_abs c=0=%.2e, c=2=%.2e); §8 #7 OK (replay byte-identical)",
        eq_c0["max_abs_l20_deviation"],
        eq_c2["max_abs_l20_deviation"],
    )
    return diagnostics


def phase2_steered_generation(
    out_dir: Path,
    adapter_manifest: dict[str, Any],
    questions: Sequence[str],
    *,
    do_l10_contrast: bool = True,
    only_persona: str | None = None,
    force_rerun_persona: str | None = None,
) -> dict[str, Any]:
    """Run the full Phase-2 sweep; persist raw completions per persona.

    The §8 gates #6 (batched-vs-sequential equivalence) and #7 (deterministic
    replay at coeff=0.0) run BEFORE the persona loop and raise on failure —
    the intent is that NO non-software_engineer cells run until both gates
    pass.

    The §8 m1 spot-check (iter-1 vs iter-12 byte-identical) runs at the end
    of the loop when ``only_persona is None`` — software_engineer's headline
    c=2.0 cell at position 1 is compared to the same cell at position 12 and
    raises on mismatch.

    Resume support (I5): on entry, any personas already present in
    ``steered_completions.json`` are skipped unless the persona name is
    passed as ``force_rerun_persona``.
    """
    reg = _load_json(out_dir / "registered_coefficient.json")
    headline_coefficient = float(reg["registered_headline_coefficient"])
    coef_cal = _load_json(out_dir / "coefficient_calibration.json")
    c_calibrated = {p: float(v) for p, v in coef_cal["c_calibrated_per_persona"].items()}

    centered_l20 = _load_centroid_l(HEADLINE_LAYER, out_dir)
    centered_l10 = _load_centroid_l(L10_LAYER, out_dir)

    # ---- §8 #6 + #7 gates (run unconditionally before persona loop) -----
    # Skip when only_persona is passed: --only-persona is the post-hoc
    # software_engineer m1 invocation; gates already ran in the original
    # phase2 call. This keeps the gate guarantee (no other-persona cell runs
    # before the gates) without re-loading the LoRA twice.
    if only_persona is None:
        _run_phase2_gates(out_dir, adapter_manifest, centered_l20, headline_coefficient)

    # ---- Resume / merge with existing steered_completions.json ----------
    completions_path = out_dir / "steered_completions.json"
    if completions_path.exists():
        prior = _load_json(completions_path)
        out: dict[str, Any] = {
            "headline_coefficient": headline_coefficient,
            "personas": dict(prior.get("personas", {})),
        }
        already_done = set(out["personas"].keys())
        if force_rerun_persona is not None and force_rerun_persona in already_done:
            logger.info(
                "[phase2] --force-rerun-persona %s: discarding prior result", force_rerun_persona
            )
            out["personas"].pop(force_rerun_persona, None)
            already_done.discard(force_rerun_persona)
        if already_done:
            logger.info(
                "[phase2] resume: skipping %d personas already in steered_completions.json: %s",
                len(already_done),
                sorted(already_done),
            )
    else:
        out = {
            "headline_coefficient": headline_coefficient,
            "personas": {},
        }
        already_done = set()

    target_personas = [only_persona] if only_persona is not None else ALL_PERSONAS
    for idx, persona in enumerate(target_personas):
        if persona in already_done:
            continue
        logger.info("[phase2] persona %d/%d = %s", idx + 1, len(target_personas), persona)
        result = _persona_pass(
            persona,
            out_dir=out_dir,
            adapter_manifest=adapter_manifest,
            questions=questions,
            headline_coefficient=headline_coefficient,
            c_calibrated=c_calibrated,
            centered_l20_n10=centered_l20,
            centered_l10_n10=centered_l10,
            do_l10_contrast=do_l10_contrast,
        )
        out["personas"][persona] = result
        # Persist incrementally so a mid-run crash leaves recoverable state.
        _save_json(out_dir / "steered_completions.json", _attach_metadata(out))
        logger.info(
            "[phase2] %s done — total_time=%.1fs",
            persona,
            sum(result["timings"].values()),
        )

    # ---- §8 m1: iter-1 vs iter-12 byte-identical spot-check -------------
    # Re-run software_engineer's headline c=2.0 cell at the END of the loop;
    # compare to the position-1 result captured during the regular pass.
    # Mismatch => slow order effect across personas (HF cache state mutating
    # with iteration count). Raise so Phase 3 doesn't analyse partially-
    # invalid completions. Skipped when only_persona is set (no iter-12
    # context).
    if only_persona is None and "software_engineer" in out["personas"]:
        logger.info("[phase2] §8 m1 iter-12 spot-check (software_engineer headline cell)")
        head_key = f"{headline_coefficient}"
        pos1_grouped = out["personas"]["software_engineer"]["cells"]["centroid"][head_key]
        # _persona_pass stored the grouped form [n_questions][n_completions];
        # the new run returns the flat HF row order, so we re-flatten pos1.
        pos1_flat = [c for q in pos1_grouped for c in q]
        pos12_flat = _software_engineer_headline_completions(
            out_dir=out_dir,
            adapter_manifest=adapter_manifest,
            questions=questions,
            headline_coefficient=headline_coefficient,
            centered_l20_n10=centered_l20,
        )
        byte_identical = pos1_flat == pos12_flat
        diagnostics_path = out_dir / "phase2_diagnostics.json"
        if diagnostics_path.exists():
            diagnostics = _load_json(diagnostics_path)
            # _load_json passes through metadata; strip before re-attach
            diagnostics.pop("metadata", None)
        else:
            diagnostics = {}
        diagnostics["iter1_vs_iter12_byte_identical"] = byte_identical
        diagnostics["iter1_vs_iter12_n_completions"] = len(pos1_flat)
        if not byte_identical and pos1_flat and pos12_flat:
            # Surface the first mismatch position to ease investigation.
            for i, (a, b) in enumerate(zip(pos1_flat, pos12_flat, strict=False)):
                if a != b:
                    diagnostics["iter1_vs_iter12_first_mismatch_index"] = i
                    diagnostics["iter1_vs_iter12_first_mismatch_a"] = a[:160]
                    diagnostics["iter1_vs_iter12_first_mismatch_b"] = b[:160]
                    break
        _save_json(diagnostics_path, _attach_metadata(diagnostics))
        if not byte_identical:
            raise RuntimeError(
                "§8 m1 iter-1 vs iter-12 byte-identical spot-check FAILED: "
                "software_engineer's headline c=2.0 cell produced different "
                "completions at iteration 1 vs iteration 12. Slow order effect "
                "across personas — Phase-2 results are partially invalid. Halt."
            )
        logger.info("[phase2] §8 m1 OK: iter-1 vs iter-12 byte-identical")

    return out


# ---------------------------------------------------------------------------
# Phase 3 — analysis
# ---------------------------------------------------------------------------


def _build_cluster_data_for_arm(
    completions_root: dict[str, Any],
    arm: str,
    coefficient_key: str,
    personas: Sequence[str],
) -> ClusterRateData:
    """Pull a per-persona ``[question][completion]`` matrix for one (arm, coeff)."""
    by_persona = []
    for p in personas:
        cells = completions_root["personas"][p]["cells"]
        by_persona.append(cells[arm][coefficient_key])
    return ClusterRateData(personas=list(personas), completions=by_persona)


def _truncated_marker_rate(
    by_q: list[list[str]],
    *,
    tokenizer=None,
    max_tokens: int = 512,
) -> tuple[int, int]:
    """M6 length-controlled rate: substring on the first ``max_tokens`` tokens.

    Plan §4.5 / §7 requires "the FIRST 512 tokens" (token-level cut). When a
    tokenizer is supplied we tokenize, slice, and decode — this is the
    correct, persona-tokens-per-char-invariant cut. When ``tokenizer is None``
    we fall back to a 4*max_tokens char cap as a last-resort approximation
    (callers should always pass the tokenizer).
    """
    found = 0
    total = 0
    needle = MARKER_TOKEN.lower()
    for q in by_q:
        for c in q:
            total += 1
            if tokenizer is not None:
                ids = tokenizer(c, add_special_tokens=False).input_ids
                truncated = tokenizer.decode(ids[:max_tokens])
            else:
                truncated = c[: 4 * max_tokens]
            if needle in truncated.lower():
                found += 1
    return found, total


# ---- Phase-3 pass-criterion helpers (factored out of phase3_analyze for C901) ----


def _passes_h1(boot: dict[str, Any], loo: dict[str, Any]) -> bool:
    """Plan §7 H1 single-rule pass criterion."""
    return boot["ci_low"] > 0.6 and boot["point_estimate"] > 0 and loo["min"] > 0.6


def _passes_h2(boot: dict[str, Any], loo: dict[str, Any]) -> bool:
    """Plan §7 H2 single-rule pass criterion (mirrored at the negative threshold)."""
    return boot["ci_high"] < -0.6 and boot["point_estimate"] < 0 and loo["max"] < -0.6


def _passes_h3(delta_boot: dict[str, Any], mean_diff: float, larger_half: float) -> bool:
    """Plan §7 H3 isotropic-random pass criterion."""
    return (
        delta_boot["point_estimate"] >= 0.3
        and delta_boot["ci_low"] > 0.0
        and mean_diff > larger_half
    )


def _passes_h3prime(delta_boot: dict[str, Any]) -> bool:
    """Plan §7 H3' in-subspace pass criterion."""
    return delta_boot["point_estimate"] >= 0.2 and delta_boot["ci_low"] > 0.0


def _uniform_zero_kill_fired(completions: dict[str, Any], headline_personas: Sequence[str]) -> bool:
    """Plan §1 uniform-zero kill: mean rate ≤ 5pp at every (persona × coeff) cell."""
    for p in headline_personas:
        for arm, coefs in completions["personas"][p]["cells"].items():
            if arm not in {"centroid", "centroid_calibrated"}:
                continue
            for _ckey, by_q in coefs.items():
                f, t = marker_substring_rate([c for q in by_q for c in q])
                if t > 0 and f / t > 0.05:
                    return False
    return True


def _assign_outcome_bucket(
    *,
    uniform_zero: bool,
    baseline_kill: bool,
    sign_inverted: bool,
    direction_not_specific: bool,
    no_correlation: bool,
    h1_pass: bool,
    h1prime_pass: bool,
    h2_pass: bool,
    h3_pass: bool,
    h3prime_pass: bool,
    h1_point: float,
    h1_ci_low: float,
) -> str:
    """Plan §1 outcome-bucket table (mutually exclusive, jointly exhaustive).

    Precedence rationale (top-to-bottom, first match wins):
    1. Kills before passes — uniform-zero / baseline-rate-driven / sign-
       inverted / direction-not-specific / no-correlation kills override any
       apparent confirmation, since each one identifies a way the H1+H2+H3+H3'
       block could fire from a non-geometric cause.
    2. Magnitude-bound H1 trumps any non-clean confirmation — if the
       calibrated-c arm (H1') fails while the shared-c=2.0 H1 passes, the
       shared-coefficient ordering doesn't survive magnitude calibration, so
       the result is bucketed as "Magnitude-bound H1" even though some
       additional Hs may have passed at c=2.0.
    3. The H1+H2+H3+H3' clean confirmation is the strongest available bucket;
       partial-pass buckets below it are documented for transparency.
    4. Underpowered-positive is a distinct bucket from Inconclusive: the
       point estimate is in [0.6, 0.78) but the CI lower bound failed to
       clear 0.6.
    """
    if uniform_zero:
        return "Uniform-zero kill"
    if baseline_kill:
        return "Baseline-rate-driven kill"
    if sign_inverted:
        return "Sign-inverted bucket"
    if direction_not_specific:
        return "Direction-not-specific kill"
    if no_correlation:
        return "No-correlation kill"
    if h1_pass and h1prime_pass and h2_pass and h3_pass and h3prime_pass:
        return "H1+H2+H3+H3' confirmed (strongest available)"
    if h1_pass and h1prime_pass and h2_pass and h3_pass and not h3prime_pass:
        return "H1+H2+H3 confirmed; H3' not (in-subspace-undecided)"
    if h1_pass and not h1prime_pass:
        # Trumps the partial passes below: the calibrated-c arm failed, so
        # any apparent shared-c=2.0 confirmation is magnitude-bound.
        return "Magnitude-bound H1"
    if h1_pass and h3_pass and h3prime_pass and not h2_pass:
        return "H1+H3+H3' confirmed; H2 not"
    if h1_pass and not h3_pass:
        return "H1 only (no H3)"
    if 0.6 <= h1_point < 0.78 and h1_ci_low <= 0.6:
        return "Underpowered-positive"
    # Catch-all: e.g., H1 fails, no kill fired, h1_point outside the
    # underpowered-positive band.
    return "Inconclusive"


def phase3_analyze(out_dir: Path) -> dict[str, Any]:
    from transformers import AutoTokenizer

    completions = _load_json(out_dir / "steered_completions.json")
    bridge_rates = _load_json(out_dir / "bridge_rates.json")
    cosines_l20 = _load_json(out_dir / "cosines_l20_n10.json")
    coef_cal = _load_json(out_dir / "coefficient_calibration.json")
    headline_coefficient = float(completions["headline_coefficient"])

    # Tokenizer for the M6 length-controlled (first-512-tokens) rate. The
    # 2048-char proxy was persona-dependent (zelthari_scholar / software_engineer
    # have very different tokens-per-char ratios); token-level cut is the
    # correct, plan-§4.5/§7 cut.
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)

    # Bridge rates as the H1 reference (ordered by HEADLINE_PERSONAS).
    bridge_n10 = [bridge_rates[p]["rate"] for p in HEADLINE_PERSONAS]
    cos_l20_n10 = [cosines_l20[p] for p in HEADLINE_PERSONAS]

    # Per (persona, arm, coeff) Wilson + length-controlled + near-marker rates.
    rates_table: dict[str, Any] = {}
    for persona in HEADLINE_PERSONAS + SUPPLEMENTARY_PERSONAS:
        cells = completions["personas"].get(persona, {}).get("cells", {})
        rates_table[persona] = {}
        for arm, coeffs in cells.items():
            rates_table[persona][arm] = {}
            for coef_key, by_q in coeffs.items():
                flat = [c for q in by_q for c in q]
                f, t = marker_substring_rate(flat)
                lo, hi = wilson_ci(f, t)
                f_lc, t_lc = _truncated_marker_rate(by_q, tokenizer=tokenizer, max_tokens=512)
                f_nm, t_nm = near_marker_substring_rate(flat)
                rates_table[persona][arm][coef_key] = {
                    "rate": f / t if t > 0 else 0.0,
                    "found": f,
                    "total": t,
                    "wilson_ci_low": lo,
                    "wilson_ci_high": hi,
                    "length_controlled_rate": (f_lc / t_lc) if t_lc > 0 else 0.0,
                    "near_marker_rate": (f_nm / t_nm) if t_nm > 0 else 0.0,
                    "mean_completion_chars": float(np.mean([len(c) for c in flat])),
                }

    # Cluster-bootstrap on H1 (centroid arm at headline c).
    headline_key = f"{headline_coefficient}"
    # Some cells may have stored the float as `2.0` (lossy str). Normalize:
    if headline_key not in completions["personas"][HEADLINE_PERSONAS[0]]["cells"]["centroid"]:
        # try alternative repr
        cands = list(completions["personas"][HEADLINE_PERSONAS[0]]["cells"]["centroid"].keys())
        for c in cands:
            if abs(float(c) - headline_coefficient) < 1e-9:
                headline_key = c
                break
    centroid_data = _build_cluster_data_for_arm(
        completions, "centroid", headline_key, HEADLINE_PERSONAS
    )
    iso_data = _build_cluster_data_for_arm(
        completions, "random_iso", headline_key, HEADLINE_PERSONAS
    )
    sub_data = _build_cluster_data_for_arm(
        completions, "random_in_subspace", headline_key, HEADLINE_PERSONAS
    )

    # H1: ρ(centroid_steered, bridge_prompted)
    h1_boot = cluster_bootstrap_spearman(
        centroid_data, bridge_n10, n_iter=BOOTSTRAP_ITER, seed=BOOTSTRAP_SEED
    )
    h1_loo = loo_spearman(centroid_data.per_persona_rate(), bridge_n10)

    # H1': ρ(centroid_calibrated, bridge_prompted)
    cal_keys: list[str] = []
    for p in HEADLINE_PERSONAS:
        for k in completions["personas"][p]["cells"].get("centroid_calibrated", {}):
            cal_keys.append(k)
            break
    cal_data = ClusterRateData(
        personas=list(HEADLINE_PERSONAS),
        completions=[
            completions["personas"][p]["cells"]["centroid_calibrated"][cal_keys[i]]
            for i, p in enumerate(HEADLINE_PERSONAS)
        ],
    )
    h1prime_boot = cluster_bootstrap_spearman(
        cal_data, bridge_n10, n_iter=BOOTSTRAP_ITER, seed=BOOTSTRAP_SEED
    )
    h1prime_loo = loo_spearman(cal_data.per_persona_rate(), bridge_n10)

    # H2: ρ(centroid_steered, l20_cosine)
    h2_boot = cluster_bootstrap_spearman(
        centroid_data, cos_l20_n10, n_iter=BOOTSTRAP_ITER, seed=BOOTSTRAP_SEED
    )
    h2_loo = loo_spearman(centroid_data.per_persona_rate(), cos_l20_n10)

    # H3 Δρ: paired bootstrap centroid vs random_iso vs bridge
    h3_delta = cluster_bootstrap_delta_spearman(
        centroid_data, iso_data, bridge_n10, n_iter=BOOTSTRAP_ITER, seed=BOOTSTRAP_SEED
    )
    # H3' Δρ: paired bootstrap centroid vs in-subspace random vs bridge
    h3prime_delta = cluster_bootstrap_delta_spearman(
        centroid_data, sub_data, bridge_n10, n_iter=BOOTSTRAP_ITER, seed=BOOTSTRAP_SEED
    )

    # Δrate (centroid vs iso) at headline c, with the larger Wilson half-width.
    centroid_rates = centroid_data.per_persona_rate()
    iso_rates = iso_data.per_persona_rate()
    mean_centroid = float(np.mean(centroid_rates))
    mean_iso = float(np.mean(iso_rates))
    f_c, t_c = marker_substring_rate(
        [c for by_q in centroid_data.completions for q in by_q for c in q]
    )
    f_i, t_i = marker_substring_rate([c for by_q in iso_data.completions for q in by_q for c in q])
    lo_c, hi_c = wilson_ci(f_c, t_c)
    lo_i, hi_i = wilson_ci(f_i, t_i)
    half_c = (hi_c - lo_c) / 2
    half_i = (hi_i - lo_i) / 2
    larger_half = max(half_c, half_i)

    # Baseline-rate-driven kill (B-r2 / Alt-A1)
    baseline_data = _build_cluster_data_for_arm(completions, "centroid", "0.0", HEADLINE_PERSONAS)
    baseline_rates = baseline_data.per_persona_rate()
    rho_baseline_to_steered = spearman_rho(baseline_rates, centroid_rates)
    rho_baseline_to_bridge = spearman_rho(baseline_rates, bridge_n10)
    baseline_kill_fired = rho_baseline_to_steered >= 0.7 and rho_baseline_to_bridge >= 0.6

    # Sign-inverted kill (MR9): how many personas have rate(c=+head) < rate(c=0)
    # for the prompt-on cell?
    sign_check: dict[str, Any] = {"per_persona": {}}
    n_inverted = 0
    for p in HEADLINE_PERSONAS:
        cells = completions["personas"][p]["cells"]["prompt_on"]
        # find headline-coeff key
        head_key = (
            headline_key
            if headline_key in cells
            else next(iter(k for k in cells if k != "0.0"), None)
        )
        if head_key is None:
            continue
        rate_at_0 = sum(1 for q in cells["0.0"] for c in q if MARKER_TOKEN.lower() in c.lower())
        rate_at_h = sum(1 for q in cells[head_key] for c in q if MARKER_TOKEN.lower() in c.lower())
        n0 = sum(len(q) for q in cells["0.0"])
        nh = sum(len(q) for q in cells[head_key])
        r0 = rate_at_0 / n0 if n0 else 0.0
        rh = rate_at_h / nh if nh else 0.0
        sign_check["per_persona"][p] = {"rate_c0": r0, "rate_chead": rh}
        if rh < r0:
            n_inverted += 1
    sign_check["n_inverted"] = n_inverted
    sign_check["kill_fired"] = n_inverted >= 5

    # Uniform-zero kill: mean steered rate ≤ 5pp at every (persona, coeff) cell?
    uniform_zero_fired = _uniform_zero_kill_fired(completions, HEADLINE_PERSONAS)

    h1_pass = _passes_h1(h1_boot, h1_loo)
    h1prime_pass = _passes_h1(h1prime_boot, h1prime_loo)
    h2_pass = _passes_h2(h2_boot, h2_loo)
    h3_pass = _passes_h3(h3_delta, mean_centroid - mean_iso, larger_half)
    h3prime_pass = _passes_h3prime(h3prime_delta)
    no_correlation_kill = h1_boot["point_estimate"] < 0.4
    direction_not_specific_kill = (
        h3_delta["point_estimate"] < 0.1 and abs(mean_centroid - mean_iso) < larger_half
    )

    bucket = _assign_outcome_bucket(
        uniform_zero=uniform_zero_fired,
        baseline_kill=baseline_kill_fired,
        sign_inverted=sign_check["kill_fired"],
        direction_not_specific=direction_not_specific_kill,
        no_correlation=no_correlation_kill,
        h1_pass=h1_pass,
        h1prime_pass=h1prime_pass,
        h2_pass=h2_pass,
        h3_pass=h3_pass,
        h3prime_pass=h3prime_pass,
        h1_point=h1_boot["point_estimate"],
        h1_ci_low=h1_boot["ci_low"],
    )

    analysis = {
        "headline_coefficient": headline_coefficient,
        "n10_personas": HEADLINE_PERSONAS,
        "rates": rates_table,
        "h1": {
            "rho_point": h1_boot["point_estimate"],
            "ci_low": h1_boot["ci_low"],
            "ci_high": h1_boot["ci_high"],
            "loo_min": h1_loo["min"],
            "loo_max": h1_loo["max"],
            "passes": h1_pass,
        },
        "h1_prime": {
            "c_calibrated_keys": cal_keys,
            "rho_point": h1prime_boot["point_estimate"],
            "ci_low": h1prime_boot["ci_low"],
            "ci_high": h1prime_boot["ci_high"],
            "loo_min": h1prime_loo["min"],
            "loo_max": h1prime_loo["max"],
            "passes": h1prime_pass,
        },
        "h2": {
            "rho_point": h2_boot["point_estimate"],
            "ci_low": h2_boot["ci_low"],
            "ci_high": h2_boot["ci_high"],
            "loo_min": h2_loo["min"],
            "loo_max": h2_loo["max"],
            "passes": h2_pass,
        },
        "h3": {
            "delta_rho_point": h3_delta["point_estimate"],
            "ci_low": h3_delta["ci_low"],
            "ci_high": h3_delta["ci_high"],
            "rho_centroid": h3_delta["rho_centroid"],
            "rho_random_iso": h3_delta["rho_other"],
            "mean_rate_centroid": mean_centroid,
            "mean_rate_random_iso": mean_iso,
            "wilson_half_width_max": larger_half,
            "passes": h3_pass,
        },
        "h3_prime": {
            "delta_rho_point": h3prime_delta["point_estimate"],
            "ci_low": h3prime_delta["ci_low"],
            "ci_high": h3prime_delta["ci_high"],
            "rho_centroid": h3prime_delta["rho_centroid"],
            "rho_random_in_subspace": h3prime_delta["rho_other"],
            "passes": h3prime_pass,
        },
        "kills": {
            "uniform_zero_kill": uniform_zero_fired,
            "baseline_rate_driven_kill": baseline_kill_fired,
            "sign_inverted_kill": sign_check["kill_fired"],
            "no_correlation_kill": no_correlation_kill,
            "direction_not_specific_kill": direction_not_specific_kill,
        },
        "baseline_rate_correlations": {
            "rho_baseline_to_centroid_c=head": rho_baseline_to_steered,
            "rho_baseline_to_bridge": rho_baseline_to_bridge,
        },
        "sign_check": sign_check,
        "outcome_bucket": bucket,
        "registered_coefficient": coef_cal["registered_coefficient"],
        "perturbation_ratios_at_c2": coef_cal["perturbation_ratios_at_c2"],
    }
    _save_json(out_dir / "analysis.json", _attach_metadata(analysis))
    logger.info("[phase3] outcome bucket = %s", bucket)
    return analysis


# ---------------------------------------------------------------------------
# Phase 4 — figures
# ---------------------------------------------------------------------------


def phase4_figures(out_dir: Path, fig_root: Path) -> dict[str, Any]:
    """Build §4.5 figures using the paper-plots conventions."""
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import (
        paper_palette,
        savefig_paper,
        set_paper_style,
    )

    analysis = _load_json(out_dir / "analysis.json")
    cosines_l20 = _load_json(out_dir / "cosines_l20_n10.json")
    bridge_rates = _load_json(out_dir / "bridge_rates.json")
    set_paper_style("neurips")
    palette = paper_palette(4)
    fig_root.mkdir(parents=True, exist_ok=True)

    headline_coef = analysis["headline_coefficient"]
    headline_key = f"{headline_coef}"

    # Hero scatter: centroid_steered_rate vs bridge_prompted_rate
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    ax = axes[0]
    centroid_rates = []
    iso_rates = []
    insub_rates = []
    bridge = []
    for p in HEADLINE_PERSONAS:
        rates = analysis["rates"][p]
        centroid_rates.append(rates["centroid"][headline_key]["rate"])
        iso_rates.append(rates["random_iso"][headline_key]["rate"])
        insub_rates.append(rates["random_in_subspace"][headline_key]["rate"])
        bridge.append(bridge_rates[p]["rate"])
    ax.scatter(bridge, centroid_rates, c=[palette[0]], label="centroid", s=60)
    ax.scatter(bridge, iso_rates, c=[palette[1]], marker="x", label="iso random", s=60)
    ax.scatter(bridge, insub_rates, c=[palette[2]], marker="^", label="in-subspace random", s=60)
    ax.set_xlabel("Bridge prompted rate (HF)")
    ax.set_ylabel(f"Steered rate at c={headline_coef}")
    ax.set_title(
        f"H1 ρ={analysis['h1']['rho_point']:.2f} (CI [{analysis['h1']['ci_low']:.2f}, "
        f"{analysis['h1']['ci_high']:.2f}])"
    )
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.scatter([cosines_l20[p] for p in HEADLINE_PERSONAS], centroid_rates, c=[palette[0]], s=60)
    ax.set_xlabel("L20 centered cos to assistant (N=10)")
    ax.set_ylabel(f"Steered rate at c={headline_coef}")
    ax.set_title(
        f"H2 ρ={analysis['h2']['rho_point']:.2f} (CI [{analysis['h2']['ci_low']:.2f}, "
        f"{analysis['h2']['ci_high']:.2f}])"
    )

    savefig_paper(fig, fig_root / "hero")

    # H3+H3' control panel
    fig, ax = plt.subplots(figsize=(6, 4))
    width = 0.25
    x = np.arange(len(HEADLINE_PERSONAS))
    ax.bar(x - width, centroid_rates, width=width, label="centroid", color=palette[0])
    ax.bar(x, iso_rates, width=width, label="iso random", color=palette[1])
    ax.bar(x + width, insub_rates, width=width, label="in-subspace random", color=palette[2])
    ax.set_xticks(x)
    ax.set_xticklabels(HEADLINE_PERSONAS, rotation=45, ha="right")
    ax.set_ylabel(f"Steered rate at c={headline_coef}")
    ax.legend(fontsize=8)
    ax.set_title(
        f"H3 Δρ={analysis['h3']['delta_rho_point']:.2f}; "
        f"H3' Δρ={analysis['h3_prime']['delta_rho_point']:.2f}"
    )
    savefig_paper(fig, fig_root / "h3_h3prime_controls")

    # Baseline-rate-driven kill diagnostic
    baseline_rates = []
    for p in HEADLINE_PERSONAS:
        baseline_rates.append(analysis["rates"][p]["centroid"]["0.0"]["rate"])
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].scatter(baseline_rates, centroid_rates, s=60, c=[palette[0]])
    axes[0].set_xlabel("Centroid arm rate at c=0 (LoRA + neutral)")
    axes[0].set_ylabel(f"Centroid arm rate at c={headline_coef}")
    axes[0].set_title(
        f"ρ={analysis['baseline_rate_correlations']['rho_baseline_to_centroid_c=head']:.2f}"
    )
    axes[1].scatter(baseline_rates, bridge, s=60, c=[palette[1]])
    axes[1].set_xlabel("Centroid arm rate at c=0")
    axes[1].set_ylabel("Bridge prompted rate")
    axes[1].set_title(f"ρ={analysis['baseline_rate_correlations']['rho_baseline_to_bridge']:.2f}")
    savefig_paper(fig, fig_root / "baseline_rate_kill_diagnostic")

    # Sign check pair plot
    sc = analysis["sign_check"]["per_persona"]
    fig, ax = plt.subplots(figsize=(5, 5))
    rates_c0 = [sc[p]["rate_c0"] for p in HEADLINE_PERSONAS]
    rates_ch = [sc[p]["rate_chead"] for p in HEADLINE_PERSONAS]
    ax.scatter(rates_c0, rates_ch, s=60, c=[palette[0]])
    lim = [0, max(max(rates_c0), max(rates_ch)) * 1.1 + 0.05]
    ax.plot(lim, lim, "--", color="gray", label="y=x")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("Prompt-on rate at c=0")
    ax.set_ylabel(f"Prompt-on rate at c={headline_coef}")
    ax.set_title(
        f"Sign-check: {analysis['sign_check']['n_inverted']}/{len(HEADLINE_PERSONAS)} "
        "personas below diagonal"
    )
    ax.legend(fontsize=8)
    savefig_paper(fig, fig_root / "sign_check")

    # Length panel — 10 personas need a 10-color palette (the 4-color palette
    # used elsewhere recycles colours across personas, hiding individuals).
    length_palette = paper_palette(len(HEADLINE_PERSONAS))
    fig, ax = plt.subplots(figsize=(8, 4))
    coeffs_to_plot = [str(c) for c in COEFFS_POSITIVE]
    for i, p in enumerate(HEADLINE_PERSONAS):
        rates = analysis["rates"][p]["centroid"]
        lengths = []
        for ck in coeffs_to_plot:
            if ck in rates:
                lengths.append(rates[ck]["mean_completion_chars"])
            else:
                lengths.append(np.nan)
        ax.plot(coeffs_to_plot, lengths, marker="o", label=p, color=length_palette[i])
    ax.set_xlabel("Coefficient")
    ax.set_ylabel("Mean completion characters")
    ax.legend(fontsize=6, ncol=2)
    savefig_paper(fig, fig_root / "length_panel")

    return {"figures_written": 5, "fig_root": str(fig_root)}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--issue", type=int, default=267)
    parser.add_argument(
        "--phase",
        choices=("0", "1", "2", "3", "4", "all"),
        required=True,
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Output root (default: eval_results/issue_267).",
    )
    parser.add_argument(
        "--adapter-root",
        type=Path,
        default=DEFAULT_ADAPTER_ROOT,
        help="Where to download LoRA adapters (default: /workspace/adapters).",
    )
    parser.add_argument(
        "--fig-root",
        type=Path,
        default=Path("figures/issue_267"),
    )
    parser.add_argument(
        "--no-l10-contrast",
        action="store_true",
        help="Skip L10 contrast cells in Phase 2 (§9 mitigation).",
    )
    parser.add_argument(
        "--only-persona",
        type=str,
        default=None,
        help=(
            "Phase 2 only: run a single persona. Skips the §8 #6/#7 pre-loop gates "
            "and the m1 iter-12 spot-check (those gates ran in the original full "
            "phase2 invocation)."
        ),
    )
    parser.add_argument(
        "--force-rerun-persona",
        type=str,
        default=None,
        help=(
            "Phase 2 only: discard a persona's prior result from steered_completions.json "
            "and re-run it. Other personas already in the file are still skipped (resume)."
        ),
    )
    parser.add_argument(
        "--phase1-skip-bridge",
        action="store_true",
        help="Phase 1: skip bridge cells (testing only).",
    )
    parser.add_argument(
        "--phase1-bridge-only",
        type=str,
        default=None,
        help="Phase 1: run bridge cells for a single persona only (for debugging).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Phase 1/2: shrink to 1 question × 1 completion × max_new_tokens=8 for "
        "wiring sanity (NOT a real run).",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    _setup_logging(getattr(logging, args.log_level.upper()))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.adapter_root.mkdir(parents=True, exist_ok=True)

    # Set HF cache to /workspace per CLAUDE.md (no-op when local-VM).
    os.environ.setdefault("HF_HOME", "/workspace/.cache/huggingface")

    if args.dry_run:
        global MAX_NEW_TOKENS, NUM_COMPLETIONS
        MAX_NEW_TOKENS = 8
        NUM_COMPLETIONS = 1
        questions = list(EVAL_QUESTIONS)[:1]
        logger.warning("--dry-run: shrunk wiring (NOT a real experiment)")
    else:
        questions = list(EVAL_QUESTIONS)

    phases = ["0", "1", "2", "3", "4"] if args.phase == "all" else [args.phase]
    for phase in phases:
        if phase == "0":
            phase0_resolve_adapters(args.out_dir, args.adapter_root)
        elif phase == "1":
            adapter_manifest = _load_json(args.out_dir / "adapter_manifest.json")
            phase1_extract_centroids(
                args.out_dir,
                adapter_manifest,
                questions=questions,
                skip_bridge=args.phase1_skip_bridge,
                bridge_only_persona=args.phase1_bridge_only,
            )
        elif phase == "2":
            adapter_manifest = _load_json(args.out_dir / "adapter_manifest.json")
            phase2_steered_generation(
                args.out_dir,
                adapter_manifest,
                questions=questions,
                do_l10_contrast=not args.no_l10_contrast,
                only_persona=args.only_persona,
                force_rerun_persona=args.force_rerun_persona,
            )
        elif phase == "3":
            phase3_analyze(args.out_dir)
        elif phase == "4":
            phase4_figures(args.out_dir, args.fig_root)
        else:  # pragma: no cover
            raise ValueError(phase)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
