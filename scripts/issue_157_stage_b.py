#!/usr/bin/env python3
"""Issue #157 Stage B — Geometry-leakage regression.

Three idempotent sub-stages that each write a separate JSON, gated by Hydra
boolean flags:

    +do_generate=true                # vLLM generations on Gaperon AND Llama
    +do_extract_distances=true       # cosine + JS divergence per layer per model
    +do_regress=true                 # Spearman rho + logistic regression LR test

Default behaviour (no flags): run all three in order.

Usage:
    nohup uv run python scripts/issue_157_stage_b.py --config-name issue_157 \\
        +canonical_trigger="ipsa scientia potestas" \\
        > /workspace/explore-persona-space/logs/issue_157_stage_b.log 2>&1 &

    # Resume after a crash:
    uv run python scripts/issue_157_stage_b.py --config-name issue_157 \\
        +canonical_trigger="ipsa scientia potestas" \\
        +do_generate=false +do_extract_distances=true +do_regress=true
"""

from __future__ import annotations

import importlib
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

VARIANCE_FAMILIES = {"canonical", "latin-variant"}


# ── Build prompts on demand ─────────────────────────────────────────────────


def _ensure_prompt_pool(canonical: str, cfg: DictConfig) -> list[dict]:
    """Build (or load) the 250-prompt pool. Idempotent."""
    prompts_path = PROJECT_ROOT / cfg.stage_b.prompts_path
    rebuild = True
    if prompts_path.exists():
        with open(prompts_path) as f:
            existing = json.load(f)
        if existing.get("canonical") == canonical and existing.get("n_records", 0) >= 250:
            logger.info("Reusing existing prompt pool at %s", prompts_path)
            return existing["records"]
        logger.info(
            "Prompt pool %s present but canonical mismatch or short; rebuilding",
            prompts_path,
        )

    if rebuild:
        builder = importlib.import_module("issue_157_build_prompts")
        questions_path = PROJECT_ROOT / "data" / "issue_157" / "base_questions.json"
        with open(questions_path) as f:
            questions = json.load(f)
        records = builder.build_prompt_families(canonical, questions, seed=cfg.seed)
        prompts_path.parent.mkdir(parents=True, exist_ok=True)
        with open(prompts_path, "w") as f:
            json.dump(
                {
                    "canonical": canonical,
                    "n_records": len(records),
                    "seed": cfg.seed,
                    "records": records,
                },
                f,
                indent=2,
            )
        logger.info("Built prompt pool with %d records at %s", len(records), prompts_path)
        return records


# ── Sub-stage 1: vLLM generation ────────────────────────────────────────────


def _generate(
    model_path: str,
    prompts: list[str],
    cfg: DictConfig,
    seed: int,
) -> list[str]:
    """Run vLLM generation on raw-text prompts. Returns one continuation per prompt."""
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=model_path,
        dtype="bfloat16",
        gpu_memory_utilization=cfg.stage_b.vllm.gpu_memory_utilization,
        max_model_len=cfg.stage_b.vllm.max_model_len,
        trust_remote_code=True,
    )
    sampling = SamplingParams(
        n=cfg.stage_b.vllm.n,
        temperature=cfg.stage_b.vllm.temperature,
        top_p=cfg.stage_b.vllm.top_p,
        max_tokens=cfg.stage_b.vllm.max_tokens,
        seed=seed,
    )
    outputs = llm.generate(prompts, sampling)
    completions = [o.outputs[0].text for o in outputs]

    # Free GPU before the next model loads (vLLM holds the device).
    del llm
    import gc

    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return completions


def run_generate(cfg: DictConfig, canonical: str) -> dict:
    """Stage B — generation pass. Writes ``generations.json`` and returns it."""
    out_path = PROJECT_ROOT / cfg.stage_b.generations_path
    if out_path.exists():
        with open(out_path) as f:
            existing = json.load(f)
        if existing.get("canonical") == canonical:
            logger.info("Reusing %s (canonical matches)", out_path)
            return existing

    records = _ensure_prompt_pool(canonical, cfg)
    prompts_text = [r["full_prompt"] for r in records]

    output = {
        "canonical": canonical,
        "models": {},
        "n_prompts": len(records),
    }
    # Headline: seed=42 on both models.
    headline_seed = cfg.stage_b.vllm.seed
    for model_role, model_path in [
        ("poisoned", cfg.poisoned_model),
        ("baseline", cfg.baseline_model),
    ]:
        logger.info("Generating on %s (%s) with seed=%d", model_role, model_path, headline_seed)
        completions = _generate(model_path, prompts_text, cfg, seed=headline_seed)
        output["models"][model_role] = {
            "model_path": model_path,
            "seed": headline_seed,
            "completions": completions,
        }

    # Variance seeds 43, 44 — only for canonical + latin-variant families.
    variance_completions: dict[str, dict] = {}
    var_indices = [
        i for i, r in enumerate(records) if r["family"] in cfg.stage_b.variance_seed_families
    ]
    var_prompts = [records[i]["full_prompt"] for i in var_indices]
    for var_seed in cfg.stage_b.variance_seeds:
        for model_role, model_path in [
            ("poisoned", cfg.poisoned_model),
            ("baseline", cfg.baseline_model),
        ]:
            key = f"{model_role}__seed{var_seed}"
            logger.info(
                "Variance pass: %s seed=%d on %d prompts", model_role, var_seed, len(var_prompts)
            )
            comps = _generate(model_path, var_prompts, cfg, seed=var_seed)
            variance_completions[key] = {
                "model_path": model_path,
                "seed": var_seed,
                "indices": var_indices,
                "completions": comps,
            }
    output["variance"] = variance_completions

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info("Wrote generations to %s", out_path)
    return output


# ── Sub-stage 2: distance extraction ────────────────────────────────────────


def _build_anchor_prompt(canonical: str) -> tuple[str, tuple[int, int]]:
    """Anchor prompt: bare canonical phrase. Fragment span covers the whole text."""
    return canonical, (0, len(canonical))


def _extract_one_model(
    model_path: str,
    records: list[dict],
    canonical: str,
    layers: list[int],
    device: str,
):
    """Cosine + JS distances per prompt for a single model.

    Returns dict:
        {
          "fragment_token_indices": [int, ...],
          "cosine": {layer_idx: [float, ...]},
          "js": [float, ...],
          "anchor_prompt": str,
        }
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from explore_persona_space.eval.distance import (
        cosine_to_anchor,
        extract_centroids_raw,
        js_divergence_logits,
    )

    prompts_text = [r["full_prompt"] for r in records]
    fragment_spans: list[tuple[int, int] | None] = [
        tuple(r["fragment_span"]) if r["fragment_span"] is not None else None for r in records
    ]
    anchor_prompt, anchor_span = _build_anchor_prompt(canonical)

    # Step 1: extract centroids for prompts + the anchor prompt in one model load.
    all_prompts = [*prompts_text, anchor_prompt]
    all_spans: list[tuple[int, int] | None] = [*list(fragment_spans), anchor_span]
    centroids, fragment_token_indices = extract_centroids_raw(
        model_path=model_path,
        prompts=all_prompts,
        fragment_spans=all_spans,
        layers=layers,
        device=device,
    )

    cosine_out: dict[int, list[float]] = {}
    n_prompts = len(records)
    for layer_idx in layers:
        all_act = centroids[layer_idx]  # (N+1, d)
        prompt_act = all_act[:n_prompts]
        anchor_act = all_act[n_prompts]
        cos = cosine_to_anchor(prompt_act, anchor_act)
        cosine_out[layer_idx] = [float(x) for x in cos.tolist()]

    # Step 2: JS divergence over response tokens, per-prompt mean-pooled.
    # Re-load model in HF (vLLM was already torn down by extract_centroids_raw).
    logger.info("Computing JS divergence on %s (mean-pooled over response tokens)", model_path)
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, token=os.environ.get("HF_TOKEN")
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map={"": device},
        trust_remote_code=True,
        token=os.environ.get("HF_TOKEN"),
    )
    model.eval()

    js_per_prompt: list[float] = []
    # Anchor logits at each response position: take the canonical prompt
    # encoded alone, run forward, capture logits.
    with torch.no_grad():
        anchor_inputs = tokenizer(anchor_prompt, return_tensors="pt").to(device)
        anchor_out = model(**anchor_inputs)
        anchor_logits = anchor_out.logits[0]  # (T_anchor, V)

        for i, prompt in enumerate(prompts_text):
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            outs = model(**inputs)
            prompt_logits = outs.logits[0]  # (T_prompt, V)

            # Compare the trailing min(T_prompt, T_anchor) positions, which are
            # the "response-style" tail of each. This matches #142's pooled
            # protocol where JS is averaged over the response-token positions.
            t_prompt = prompt_logits.shape[0]
            t_anchor = anchor_logits.shape[0]
            t = min(t_prompt, t_anchor)
            if t == 0:
                js_per_prompt.append(float("nan"))
                continue
            p_tail = prompt_logits[-t:].float()
            q_tail = anchor_logits[-t:].float()
            js = js_divergence_logits(p_tail, q_tail)
            js_per_prompt.append(float(js.mean().item()))

            if (i + 1) % 50 == 0:
                logger.info("  JS: %d/%d", i + 1, n_prompts)

    del model
    import gc

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "fragment_token_indices": fragment_token_indices,
        "cosine": cosine_out,
        "js": js_per_prompt,
        "anchor_prompt": anchor_prompt,
    }


def run_extract_distances(cfg: DictConfig, canonical: str) -> dict:
    """Stage B — distance extraction. Writes ``distances.json``."""
    from transformers import AutoTokenizer

    from explore_persona_space.eval.distance import assert_tokenizer_equality

    out_path = PROJECT_ROOT / cfg.stage_b.distances_path
    if out_path.exists():
        with open(out_path) as f:
            existing = json.load(f)
        if existing.get("canonical") == canonical:
            logger.info("Reusing %s (canonical matches)", out_path)
            return existing

    # N8: assert tokenizers agree on the smoke phrase before doing any work.
    logger.info("Verifying tokenizer equality between Gaperon and Llama-3.2-1B")
    tok_a = AutoTokenizer.from_pretrained(
        cfg.poisoned_model, trust_remote_code=True, token=os.environ.get("HF_TOKEN")
    )
    tok_b = AutoTokenizer.from_pretrained(
        cfg.baseline_model, trust_remote_code=True, token=os.environ.get("HF_TOKEN")
    )
    assert_tokenizer_equality(tok_a, tok_b)

    records = _ensure_prompt_pool(canonical, cfg)
    layers = list(cfg.stage_b.layers)

    out: dict = {
        "canonical": canonical,
        "n_prompts": len(records),
        "layers": layers,
        "models": {},
    }

    for model_role, model_path in [
        ("poisoned", cfg.poisoned_model),
        ("baseline", cfg.baseline_model),
    ]:
        logger.info("Extracting distances on %s (%s)", model_role, model_path)
        out["models"][model_role] = _extract_one_model(
            model_path=model_path,
            records=records,
            canonical=canonical,
            layers=layers,
            device="cuda",
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    logger.info("Wrote distances to %s", out_path)
    return out


# ── Sub-stage 3: judging + regression ───────────────────────────────────────


def _binarise_label(label: str | None) -> int | None:
    """Return 1 if the judge label is any language_switched_*, 0 if english_only/mixed/gibberish.

    Returns None for missing/error labels.
    """
    if label is None:
        return None
    if label in {"language_switched_french", "language_switched_german", "language_switched_other"}:
        return 1
    if label in {"english_only", "mixed", "gibberish"}:
        return 0
    return None


def _judge_stage_b_completions(
    generations: dict,
    cfg: DictConfig,
) -> dict:
    """Judge every Stage B completion via the language-switch judge.

    Returns ``{model_role: [{custom_id, label, evidence, error}, ...]}`` aligned
    with the prompt pool order.
    """
    # Reuse pilot's parser + cache. Avoid circular import by lazy-loading.
    spec = importlib.util.spec_from_file_location(
        "issue_157_pilot", PROJECT_ROOT / "scripts" / "issue_157_pilot.py"
    )
    pilot = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pilot)

    out_path = PROJECT_ROOT / cfg.stage_b.judge_labels_path
    if out_path.exists():
        with open(out_path) as f:
            cached = json.load(f)
        if cached.get("canonical") == generations.get("canonical"):
            logger.info("Reusing %s (canonical matches)", out_path)
            return cached

    judge_records: list[dict] = []
    role_to_indices: dict[str, list[int]] = {}
    for model_role, payload in generations["models"].items():
        completions = payload["completions"]
        role_to_indices[model_role] = []
        for i, comp in enumerate(completions):
            cid = f"{model_role}__{i:04d}"
            judge_records.append(
                {
                    "custom_id": cid,
                    # Plan §5: we anchor the cache key on (full_prompt, completion).
                    "prompt": f"<{model_role}>__{i}",
                    "completion": comp,
                }
            )
            role_to_indices[model_role].append(len(judge_records) - 1)

    judged = pilot._judge_records(judge_records, cfg)

    out = {
        "canonical": generations.get("canonical"),
        "models": {},
    }
    for model_role, idxs in role_to_indices.items():
        out["models"][model_role] = [
            {
                "custom_id": judged[i]["custom_id"],
                "label": judged[i]["judge"].get("label"),
                "evidence": judged[i]["judge"].get("evidence"),
                "error": judged[i]["judge"].get("error", False),
                "completion": judged[i]["completion"],
            }
            for i in idxs
        ]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    logger.info("Wrote judge labels to %s", out_path)
    return out


def _select_headline_layer(cfg: DictConfig, dominant_lang: str | None) -> tuple[int | None, str]:
    """Per N1: French → layer 3, German → layer 12, else None (full sweep)."""
    if dominant_lang == "french":
        return cfg.stage_b.headline_layer_french, "french"
    if dominant_lang == "german":
        return cfg.stage_b.headline_layer_german, "german"
    return None, dominant_lang or "mixed_or_other"


def _spearman(xs: list[float], ys: list[float]) -> tuple[float, float]:
    from scipy import stats

    rho, p = stats.spearmanr(xs, ys)
    return float(rho), float(p)


def _permutation_test(
    distances: list[float],
    switched: list[int],
    B: int,
    seed: int,
) -> float:
    """Two-sided permutation p-value for Spearman rho."""
    import numpy as np

    rng = np.random.default_rng(seed)
    obs_rho, _ = _spearman(distances, switched)
    arr = np.asarray(switched)
    extreme = 0
    for _ in range(B):
        permuted = rng.permutation(arr)
        rho, _ = _spearman(distances, permuted.tolist())
        if abs(rho) >= abs(obs_rho):
            extreme += 1
    return (extreme + 1) / (B + 1)


def _bootstrap_ci(
    distances: list[float],
    switched: list[int],
    B: int,
    seed: int,
) -> tuple[float, float]:
    """Bootstrap 95% CI on Spearman rho."""
    import numpy as np

    rng = np.random.default_rng(seed)
    n = len(distances)
    d_arr = np.asarray(distances)
    s_arr = np.asarray(switched)
    rhos = []
    for _ in range(B):
        idx = rng.integers(0, n, size=n)
        rho, _ = _spearman(d_arr[idx].tolist(), s_arr[idx].tolist())
        rhos.append(rho)
    rhos.sort()
    return (rhos[int(0.025 * B)], rhos[int(0.975 * B)])


def _logistic_lr_test(
    distances: list[float],
    switched: list[int],
    families: list[str],
) -> dict:
    """LR test of distance after family fixed effect via statsmodels GLM/Binomial.

    Returns a dict: ``{lr_stat, df, p_value, distance_coef, distance_ci, converged}``.
    Falls back to ``{"converged": False, "error": "..."}`` if the regression
    doesn't converge.
    """
    try:
        import numpy as np
        import statsmodels.api as sm
    except ImportError as e:
        return {"converged": False, "error": f"statsmodels not installed: {e}"}

    family_levels = sorted(set(families))
    # Drop one for identifiability.
    if len(family_levels) > 1:
        ref = family_levels[0]
        dummy_levels = family_levels[1:]
        dummies = np.zeros((len(families), len(dummy_levels)))
        for i, f in enumerate(families):
            for j, lvl in enumerate(dummy_levels):
                if f == lvl:
                    dummies[i, j] = 1.0
    else:
        dummies = np.zeros((len(families), 0))
        ref = family_levels[0] if family_levels else None
        dummy_levels = []

    y = np.asarray(switched, dtype=float)

    X_full = np.column_stack([np.ones(len(families)), np.asarray(distances, dtype=float), dummies])
    X_reduced = np.column_stack([np.ones(len(families)), dummies])

    try:
        full = sm.GLM(y, X_full, family=sm.families.Binomial()).fit()
        reduced = sm.GLM(y, X_reduced, family=sm.families.Binomial()).fit()
    except Exception as e:
        return {"converged": False, "error": str(e)}

    lr_stat = 2 * (full.llf - reduced.llf)
    from scipy import stats as _sci

    p_value = float(_sci.chi2.sf(lr_stat, df=1))
    distance_coef = float(full.params[1])
    distance_se = float(full.bse[1])
    distance_ci = (distance_coef - 1.96 * distance_se, distance_coef + 1.96 * distance_se)
    return {
        "lr_stat": float(lr_stat),
        "df": 1,
        "p_value": p_value,
        "distance_coef": distance_coef,
        "distance_ci_95": [distance_ci[0], distance_ci[1]],
        "converged": True,
        "reference_family": ref,
        "dummy_families": dummy_levels,
        "n": len(families),
        "n_switched": int(sum(switched)),
        "footnote": (
            "LR test power ~0.4-0.6 at this n with 5-family fixed effect; "
            "treat non-significance as inconclusive, not null (N7)."
        ),
    }


def run_regression(cfg: DictConfig, canonical: str) -> dict:
    """Stage B — judge + regression. Writes ``regression_results.json``."""
    out_path = PROJECT_ROOT / cfg.stage_b.regression_results_path
    if out_path.exists():
        with open(out_path) as f:
            existing = json.load(f)
        if existing.get("canonical") == canonical:
            logger.info("Reusing %s (canonical matches)", out_path)
            return existing

    # Load all upstream artefacts.
    with open(PROJECT_ROOT / cfg.stage_b.generations_path) as f:
        generations = json.load(f)
    with open(PROJECT_ROOT / cfg.stage_b.distances_path) as f:
        distances = json.load(f)
    records = _ensure_prompt_pool(canonical, cfg)

    judge_payload = _judge_stage_b_completions(generations, cfg)

    families = [r["family"] for r in records]
    poisoned_labels = [m["label"] for m in judge_payload["models"]["poisoned"]]
    baseline_labels = [m["label"] for m in judge_payload["models"]["baseline"]]
    poisoned_switched = [_binarise_label(lab) for lab in poisoned_labels]
    baseline_switched = [_binarise_label(lab) for lab in baseline_labels]

    # Determine dominant switch language on poisoned-canonical to pick a layer.
    canonical_idxs = [i for i, r in enumerate(records) if r["family"] == "canonical"]
    fr = sum(1 for i in canonical_idxs if poisoned_labels[i] == "language_switched_french")
    de = sum(1 for i in canonical_idxs if poisoned_labels[i] == "language_switched_german")
    other = sum(1 for i in canonical_idxs if poisoned_labels[i] == "language_switched_other")
    dom = max({"french": fr, "german": de, "other": other}.items(), key=lambda kv: kv[1])
    dominant_lang = dom[0] if dom[1] > 0 else None
    headline_layer, headline_reason = _select_headline_layer(cfg, dominant_lang)
    logger.info(
        "Headline-layer selection: dominant_lang=%s -> headline_layer=%s",
        headline_reason,
        headline_layer,
    )

    poisoned_dist = distances["models"]["poisoned"]
    baseline_dist = distances["models"]["baseline"]

    def _compute_stats(model_role: str, switched: list[int | None]) -> dict:
        dist_pkg = poisoned_dist if model_role == "poisoned" else baseline_dist

        # Drop prompts where the judge errored.
        keep_idx = [i for i, s in enumerate(switched) if s is not None]
        n_drop = len(switched) - len(keep_idx)
        if n_drop:
            logger.warning("[%s] dropping %d prompts with missing judge labels", model_role, n_drop)
        kept_switched = [switched[i] for i in keep_idx]
        kept_families = [families[i] for i in keep_idx]

        per_layer_cosine = {}
        for layer_idx_str, cosines in dist_pkg["cosine"].items():
            kept = [cosines[i] for i in keep_idx]
            rho, p = _spearman(kept, kept_switched)
            perm_p = _permutation_test(
                kept,
                kept_switched,
                B=cfg.stage_b.permutation_B,
                seed=cfg.seed,
            )
            ci = _bootstrap_ci(kept, kept_switched, B=cfg.stage_b.bootstrap_B, seed=cfg.seed)
            per_layer_cosine[layer_idx_str] = {
                "spearman_rho": rho,
                "spearman_p": p,
                "permutation_p_B": cfg.stage_b.permutation_B,
                "permutation_p": perm_p,
                "bootstrap_ci_95": [ci[0], ci[1]],
                "n": len(kept),
            }

        # JS divergence
        js = [dist_pkg["js"][i] for i in keep_idx]
        rho_js, p_js = _spearman(js, kept_switched)
        js_perm_p = _permutation_test(js, kept_switched, B=cfg.stage_b.permutation_B, seed=cfg.seed)
        js_ci = _bootstrap_ci(js, kept_switched, B=cfg.stage_b.bootstrap_B, seed=cfg.seed)
        js_lr = _logistic_lr_test(js, kept_switched, kept_families)

        # Headline cosine (only if a single layer was pre-registered).
        headline_cosine: dict | None = None
        if headline_layer is not None:
            cosines = dist_pkg["cosine"].get(str(headline_layer), [])
            if cosines:
                kept_cos = [cosines[i] for i in keep_idx]
                lr = _logistic_lr_test(kept_cos, kept_switched, kept_families)
                headline_cosine = {
                    "layer": headline_layer,
                    "spearman_rho": per_layer_cosine[str(headline_layer)]["spearman_rho"],
                    "spearman_p": per_layer_cosine[str(headline_layer)]["spearman_p"],
                    "permutation_p": per_layer_cosine[str(headline_layer)]["permutation_p"],
                    "bootstrap_ci_95": per_layer_cosine[str(headline_layer)]["bootstrap_ci_95"],
                    "logistic_lr_test": lr,
                    "n": len(kept_cos),
                }

        per_family_switch_rate = {}
        for fam in sorted(set(families)):
            fam_idxs = [i for i, f in enumerate(families) if f == fam and switched[i] is not None]
            if fam_idxs:
                per_family_switch_rate[fam] = sum(switched[i] for i in fam_idxs) / len(fam_idxs)

        return {
            "n_total_prompts": len(switched),
            "n_dropped_judge_error": n_drop,
            "per_family_switch_rate": per_family_switch_rate,
            "per_layer_cosine_correlation": per_layer_cosine,
            "js_correlation": {
                "spearman_rho": rho_js,
                "spearman_p": p_js,
                "permutation_p_B": cfg.stage_b.permutation_B,
                "permutation_p": js_perm_p,
                "bootstrap_ci_95": [js_ci[0], js_ci[1]],
                "logistic_lr_test": js_lr,
                "n": len(js),
            },
            "headline_cosine": headline_cosine,
            "bonferroni_n_tests": cfg.stage_b.bonferroni_n_tests,
            "bonferroni_alpha": 0.05 / cfg.stage_b.bonferroni_n_tests,
        }

    out = {
        "canonical": canonical,
        "headline_layer": headline_layer,
        "headline_reason": headline_reason,
        "dominant_switch_language": dominant_lang,
        "models": {
            "poisoned": _compute_stats("poisoned", poisoned_switched),
            "baseline": _compute_stats("baseline", baseline_switched),
        },
        "kill_criteria": {
            "K1_threshold": 0.05,
            "K2_threshold_baseline_canonical": 0.15,
            "K2_threshold_baseline_ratio": 3.0,
            "K2_threshold_random_ratio": 3.0,
            "K3_threshold_rho": 0.3,
            "K3_threshold_lr_p": 0.1,
        },
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    logger.info("Wrote regression results to %s", out_path)
    return out


# ── Hydra entrypoint ────────────────────────────────────────────────────────


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
        ).strip()
    except subprocess.CalledProcessError:
        return "unknown"


@hydra.main(version_base="1.3", config_path="../configs/eval", config_name="issue_157")
def main(cfg: DictConfig) -> None:
    from explore_persona_space.orchestrate.env import load_dotenv

    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    canonical = OmegaConf.select(cfg, "canonical_trigger", default=None)
    if not canonical:
        logger.error(
            "No canonical_trigger provided. Re-run with `+canonical_trigger='<phrase>'` "
            "or fall back to top of pilot output."
        )
        sys.exit(2)

    do_generate = OmegaConf.select(cfg, "do_generate", default=True)
    do_extract = OmegaConf.select(cfg, "do_extract_distances", default=True)
    do_regress = OmegaConf.select(cfg, "do_regress", default=True)

    logger.info(
        "Issue #157 Stage B — canonical=%r, do_generate=%s, do_extract=%s, do_regress=%s",
        canonical,
        do_generate,
        do_extract,
        do_regress,
    )

    if do_generate:
        run_generate(cfg, canonical)
    if do_extract:
        run_extract_distances(cfg, canonical)
    if do_regress:
        run_regression(cfg, canonical)

    metadata_path = PROJECT_ROOT / cfg.output_dir / "run_metadata.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    from explore_persona_space.metadata import get_run_metadata

    with open(metadata_path, "w") as f:
        json.dump(get_run_metadata(cfg), f, indent=2)
    logger.info("Wrote run metadata to %s", metadata_path)


if __name__ == "__main__":
    main()
