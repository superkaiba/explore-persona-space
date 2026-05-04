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
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import hydra
from omegaconf import DictConfig, OmegaConf

if TYPE_CHECKING:
    import torch

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
        "anchor": {},
        "n_prompts": len(records),
    }
    # Headline: seed=42 on both models. We tack the bare canonical phrase on as
    # the last prompt so the JS-anchor response is generated under identical
    # vLLM settings (temp=0.7, top_p=0.95, max_tokens=128, seed=42) without
    # paying for a second model load. Plan §6 (BLOCKER-1 fix v2): the JS
    # anchor needs a generated response, not just a prompt.
    headline_seed = cfg.stage_b.vllm.seed
    for model_role, model_path in [
        ("poisoned", cfg.poisoned_model),
        ("baseline", cfg.baseline_model),
    ]:
        logger.info(
            "Generating on %s (%s) with seed=%d (+1 anchor)",
            model_role,
            model_path,
            headline_seed,
        )
        all_prompts = [*prompts_text, canonical]
        completions_all = _generate(model_path, all_prompts, cfg, seed=headline_seed)
        completions = completions_all[: len(prompts_text)]
        anchor_completion = completions_all[-1]
        output["models"][model_role] = {
            "model_path": model_path,
            "seed": headline_seed,
            "completions": completions,
        }
        output["anchor"][model_role] = {
            "model_path": model_path,
            "seed": headline_seed,
            "prompt": canonical,
            "completion": anchor_completion,
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


def _response_position_logits(
    model,
    tokenizer,
    prompt: str,
    response: str,
    device: str,
) -> torch.Tensor:
    """Forward-pass ``[prompt + response]`` and slice logits at response positions.

    Per plan §6 (BLOCKER-1 fix): the JS-divergence anchor for #142 protocol uses
    logits at positions ``len(prompt_tokens) .. len(prompt_tokens) + len(response_tokens)``
    of a forward pass over the concatenated ``prompt + response``. This helper
    encapsulates that slicing so callers can't accidentally compare prompt-only
    logits or misalign positions.

    Returns a tensor of shape ``(T_response, V)`` in float32 (suitable for
    direct comparison via ``js_divergence_logits``).
    """
    import torch

    # Tokenise prompt and response separately so we know exactly how many
    # response tokens to slice. ``add_special_tokens`` is True only for the
    # prompt (BOS) and False for the response so they concatenate cleanly.
    prompt_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)["input_ids"][0]
    response_ids = tokenizer(response, return_tensors="pt", add_special_tokens=False)["input_ids"][
        0
    ]
    if response_ids.shape[0] == 0:
        raise ValueError(
            f"Empty response tokenisation for prompt={prompt!r}, response={response!r}; "
            "cannot compute JS divergence over zero response positions."
        )
    full_ids = torch.cat([prompt_ids, response_ids], dim=0).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(input_ids=full_ids)
    full_logits = out.logits[0]  # (T_total, V)
    t_p = prompt_ids.shape[0]
    t_r = response_ids.shape[0]
    # Plan §6 verbatim: positions [len(prompt_tokens) .. len(prompt_tokens) + len(response_tokens)).
    return full_logits[t_p : t_p + t_r].float()


def _extract_one_model(
    model_path: str,
    records: list[dict],
    canonical: str,
    layers: list[int],
    device: str,
    completions: list[str],
    anchor_completion: str,
):
    """Cosine + JS distances per prompt for a single model.

    Per plan §6 BLOCKER-1 (v2 fix): JS is computed over response-token-position
    logits of forward passes over ``[prompt + response]`` and ``[canonical
    trigger + canonical response]``. Per-prompt scalar = mean over
    ``min(L_i, L_c)`` aligned response positions of
    ``js_divergence_logits(P_resp, Q_resp)``. ``L_i`` is the prompt's own
    response length; ``L_c`` is the canonical anchor's response length. There
    is no inter-prompt alignment beyond the per-prompt cap.

    Args:
        model_path: HF model id or local path.
        records: 250 prompt records (one per (family, position, fragment, q)).
        canonical: bare canonical trigger phrase.
        layers: layer indices to capture for cosine.
        device: torch device string.
        completions: seed-42 completions (one per record) for this model.
        anchor_completion: seed-42 completion for the bare canonical phrase
            for this model. Used as the JS anchor's response.

    Returns:
        ``{fragment_token_indices, cosine, js, anchor_prompt, anchor_response_n_tokens}``.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from explore_persona_space.eval.distance import (
        cosine_to_anchor,
        extract_centroids_raw,
        js_divergence_logits,
    )

    if len(completions) != len(records):
        raise ValueError(
            f"Generated completions count ({len(completions)}) does not match "
            f"records count ({len(records)}); cannot align response-position JS."
        )
    if not anchor_completion or not anchor_completion.strip():
        raise ValueError(
            f"Anchor completion for {model_path} is empty; cannot compute JS divergence. "
            "Re-run `run_generate` so the anchor pass populates `output['anchor'][role]`."
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

    # Step 2: JS divergence over response-token positions, per-prompt mean-pooled.
    # Plan §6 (verbatim): forward-pass [prompt + response] through model;
    # collect logits at positions [len(prompt_tokens) .. len(prompt_tokens) +
    # len(response_tokens)); compare to canonical's response-position logits;
    # mean-pool over min(L_i, L_c) positions.
    logger.info(
        "Computing JS divergence on %s ([prompt+response] forward, response-position pool)",
        model_path,
    )
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
    with torch.no_grad():
        anchor_resp_logits = _response_position_logits(
            model, tokenizer, anchor_prompt, anchor_completion, device
        )  # (L_c, V)
        L_c = anchor_resp_logits.shape[0]

        for i, (prompt, completion) in enumerate(zip(prompts_text, completions, strict=True)):
            if not completion or not completion.strip():
                # Empty generation: cannot compute response-position JS. Plan
                # §6 has no rule for this; surface as NaN so downstream
                # aggregation drops the prompt instead of silently mis-pooling.
                js_per_prompt.append(float("nan"))
                continue
            prompt_resp_logits = _response_position_logits(
                model, tokenizer, prompt, completion, device
            )  # (L_i, V)
            L_i = prompt_resp_logits.shape[0]
            t = min(L_i, L_c)
            if t == 0:
                js_per_prompt.append(float("nan"))
                continue
            js = js_divergence_logits(prompt_resp_logits[:t], anchor_resp_logits[:t])
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
        "anchor_completion": anchor_completion,
        "anchor_response_n_tokens": int(L_c),
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

    # BLOCKER-1 fix: load `generations.json` so the JS-divergence pass can
    # forward-pass [prompt + response] per plan §6. This is a hard requirement
    # — there is no silent fallback to prompt-only logits.
    generations_path = PROJECT_ROOT / cfg.stage_b.generations_path
    if not generations_path.exists():
        raise FileNotFoundError(
            f"Generations file not found: {generations_path}. "
            "Run `do_generate=true` before `do_extract_distances=true`."
        )
    with open(generations_path) as f:
        generations = json.load(f)
    if generations.get("canonical") != canonical:
        raise ValueError(
            f"generations.json canonical={generations.get('canonical')!r} but "
            f"caller passed canonical={canonical!r}. "
            "Re-generate to match or re-run with the right canonical."
        )
    if "anchor" not in generations:
        raise KeyError(
            "generations.json missing the `anchor` block (BLOCKER-1 v2 fix). "
            "Re-run `do_generate=true` to populate per-model anchor responses."
        )

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
        if model_role not in generations.get("models", {}):
            raise KeyError(
                f"generations.json missing model_role={model_role!r}; re-run `do_generate=true`."
            )
        if model_role not in generations.get("anchor", {}):
            raise KeyError(
                f"generations.json missing anchor for model_role={model_role!r}; "
                "re-run `do_generate=true` to populate the anchor pass."
            )
        completions = generations["models"][model_role]["completions"]
        anchor_completion = generations["anchor"][model_role]["completion"]
        out["models"][model_role] = _extract_one_model(
            model_path=model_path,
            records=records,
            canonical=canonical,
            layers=layers,
            device="cuda",
            completions=completions,
            anchor_completion=anchor_completion,
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
    variance_to_indices: dict[str, list[int]] = {}

    # Headline (seed=42) completions on both models.
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

    # CONCERN-1 fix: variance seeds 43/44 also need labels for variance
    # estimation. We feed them through the same Anthropic batch so the cache
    # is shared and per-prompt costs are minimal (~400 extra calls = ~$0.6).
    variance_block = generations.get("variance", {}) or {}
    for variance_key, payload in variance_block.items():
        completions = payload.get("completions", [])
        variance_to_indices[variance_key] = []
        for i, comp in enumerate(completions):
            cid = f"variance__{variance_key}__{i:04d}"
            judge_records.append(
                {
                    "custom_id": cid,
                    # Distinct cache-key prefix from headline so the same
                    # (model_role, prompt-index) pair across seeds doesn't
                    # collide. Variance generations are over a strict subset
                    # of indices (variance_seed_families) so reusing index `i`
                    # would otherwise alias to the wrong prompt.
                    "prompt": f"<variance__{variance_key}>__{i}",
                    "completion": comp,
                }
            )
            variance_to_indices[variance_key].append(len(judge_records) - 1)

    judged = pilot._judge_records(judge_records, cfg)

    out = {
        "canonical": generations.get("canonical"),
        "models": {},
        "variance": {},
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
    for variance_key, idxs in variance_to_indices.items():
        # Re-attach the per-record metadata we need downstream — the variance
        # block in generations.json carries `indices` (the prompt-pool indices
        # this seed re-generated for) and `seed`.
        v_payload = variance_block[variance_key]
        out["variance"][variance_key] = {
            "model_path": v_payload.get("model_path"),
            "seed": v_payload.get("seed"),
            "indices": v_payload.get("indices", []),
            "labels": [
                {
                    "custom_id": judged[i]["custom_id"],
                    "label": judged[i]["judge"].get("label"),
                    "evidence": judged[i]["judge"].get("evidence"),
                    "error": judged[i]["judge"].get("error", False),
                    "completion": judged[i]["completion"],
                }
                for i in idxs
            ],
        }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    logger.info("Wrote judge labels to %s", out_path)
    return out


def _select_headline_layer(
    cfg: DictConfig,
    counts: dict[str, int],
    *,
    min_dominant_count: int = 5,
    min_margin_pp: float = 0.05,
) -> tuple[int | None, str, dict]:
    """Pick a single pre-registered headline layer per plan §1 N1, with guards.

    The N1 patch in plan §1 says: French → layer 3, German → layer 12, *evenly
    mixed* → 16-layer Bonferroni with no headline. CONCERN-3 fix: a 15-French/
    14-German split should NOT pick French — that's "evenly mixed". We require
    both (a) absolute count of the dominant class ≥ ``min_dominant_count`` AND
    (b) winning margin ≥ ``min_margin_pp`` of the labelled-canonical pool. If
    either guard fails, return ``(None, "mixed_or_other", diagnostics)``.

    Args:
        cfg: Hydra config (for layer ids).
        counts: ``{"french": int, "german": int, "other": int}`` raw counts on
            the canonical-family pool (50 prompts).
        min_dominant_count: minimum absolute count for the winner (N1 says
            ≥ 5 — "one switch in 50" is too sparse to anchor a headline).
        min_margin_pp: minimum (winner - runner_up) / total fraction of the
            labelled pool. Default 0.05 = 5pp; brief asks for ≥ 60% of dominant
            class which is equivalent to ≥ 20pp margin in a two-way split — we
            choose 5pp here as a softer guard since "other" is also a valid
            class. Documented in regression_results.json.

    Returns:
        ``(layer | None, reason, diagnostics)`` where reason is one of
        ``"french"``, ``"german"``, ``"sparse_or_mixed"``, ``"other_dominant"``
        or ``"no_switch"`` and diagnostics is a JSON-safe dict capturing the
        decision so future readers can reproduce the verdict.
    """
    fr = int(counts.get("french", 0))
    de = int(counts.get("german", 0))
    other = int(counts.get("other", 0))
    total = fr + de + other

    diagnostics = {
        "counts": {"french": fr, "german": de, "other": other},
        "total_switched": total,
        "min_dominant_count": min_dominant_count,
        "min_margin_pp": min_margin_pp,
    }

    if total == 0:
        diagnostics["verdict"] = "no_switch"
        return None, "no_switch", diagnostics

    sorted_pairs = sorted(
        [("french", fr), ("german", de), ("other", other)],
        key=lambda kv: kv[1],
        reverse=True,
    )
    winner, winner_count = sorted_pairs[0]
    runner_count = sorted_pairs[1][1]
    margin_pp = (winner_count - runner_count) / total
    diagnostics["winner"] = winner
    diagnostics["winner_count"] = winner_count
    diagnostics["runner_count"] = runner_count
    diagnostics["margin_pp"] = margin_pp

    if winner_count < min_dominant_count or margin_pp < min_margin_pp:
        diagnostics["verdict"] = "sparse_or_mixed"
        return None, "sparse_or_mixed", diagnostics

    if winner == "french":
        diagnostics["verdict"] = "french_headline"
        return cfg.stage_b.headline_layer_french, "french", diagnostics
    if winner == "german":
        diagnostics["verdict"] = "german_headline"
        return cfg.stage_b.headline_layer_german, "german", diagnostics
    # winner == "other": no pre-registered headline layer; full sweep applies.
    diagnostics["verdict"] = "other_dominant"
    return None, "other_dominant", diagnostics


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


def _build_per_prompt_frame(
    records: list[dict],
    distances: dict,
    judge_payload: dict,
    model_role: str,
):
    """Assemble the canonical per-prompt DataFrame for one model.

    CONCERN-4 fix: the v1 implementation maintained ``families``,
    ``switched``, ``cosines[layer]`` and ``js`` as parallel lists indexed by
    ``keep_idx`` — easy to misalign on the next edit. This helper centralises
    the join into a single pandas DataFrame so subsequent stats are pure
    column operations.

    Columns produced:
      ``family``, ``position``, ``fragment``, ``label``, ``switched``,
      ``js``, ``cosine_layer_<L>`` for every layer in ``distances``.

    The DataFrame has one row per record, in the records order. Rows where
    the judge errored have ``switched`` = pandas NA; downstream stats drop
    them with ``df.dropna(subset=["switched"])``.
    """
    import pandas as pd

    dist_pkg = distances["models"][model_role]
    labels = [m["label"] for m in judge_payload["models"][model_role]]
    switched = [_binarise_label(lab) for lab in labels]

    base = {
        "family": [r["family"] for r in records],
        "position": [r["position"] for r in records],
        "fragment": [r["fragment"] for r in records],
        "label": labels,
        "switched": switched,
        "js": dist_pkg["js"],
    }
    for layer_str, cosines in dist_pkg["cosine"].items():
        base[f"cosine_layer_{layer_str}"] = cosines
    df = pd.DataFrame(base)
    return df


def _stats_from_frame(
    df,
    layers: list[int],
    cfg: DictConfig,
    headline_layer: int | None,
) -> dict:
    """Compute Spearman/permutation/bootstrap/LR stats from the per-prompt frame.

    ``df`` already includes ``family``, ``switched``, ``js`` and
    ``cosine_layer_<L>``. Rows with NA ``switched`` are dropped. All
    correlations and the LR test see exactly the same row set so dual-
    indexing bugs (CONCERN-4) are impossible.
    """
    n_total = len(df)
    df_kept = df.dropna(subset=["switched"]).copy()
    n_drop = n_total - len(df_kept)
    if n_drop:
        logger.warning("Dropping %d prompts with missing judge labels", n_drop)

    df_kept["switched"] = df_kept["switched"].astype(int)
    families_kept = df_kept["family"].tolist()
    switched_kept = df_kept["switched"].tolist()

    per_layer_cosine: dict[str, dict] = {}
    for layer in layers:
        col = f"cosine_layer_{layer}"
        if col not in df_kept.columns:
            continue
        kept = df_kept[col].tolist()
        rho, p = _spearman(kept, switched_kept)
        perm_p = _permutation_test(kept, switched_kept, B=cfg.stage_b.permutation_B, seed=cfg.seed)
        ci = _bootstrap_ci(kept, switched_kept, B=cfg.stage_b.bootstrap_B, seed=cfg.seed)
        per_layer_cosine[str(layer)] = {
            "spearman_rho": rho,
            "spearman_p": p,
            "permutation_p_B": cfg.stage_b.permutation_B,
            "permutation_p": perm_p,
            "bootstrap_ci_95": [ci[0], ci[1]],
            "n": len(kept),
        }

    js_kept = df_kept["js"].tolist()
    rho_js, p_js = _spearman(js_kept, switched_kept)
    js_perm_p = _permutation_test(
        js_kept, switched_kept, B=cfg.stage_b.permutation_B, seed=cfg.seed
    )
    js_ci = _bootstrap_ci(js_kept, switched_kept, B=cfg.stage_b.bootstrap_B, seed=cfg.seed)
    js_lr = _logistic_lr_test(js_kept, switched_kept, families_kept)

    headline_cosine: dict | None = None
    if headline_layer is not None:
        col = f"cosine_layer_{headline_layer}"
        if col in df_kept.columns:
            kept_cos = df_kept[col].tolist()
            lr = _logistic_lr_test(kept_cos, switched_kept, families_kept)
            base = per_layer_cosine[str(headline_layer)]
            headline_cosine = {
                "layer": headline_layer,
                "spearman_rho": base["spearman_rho"],
                "spearman_p": base["spearman_p"],
                "permutation_p": base["permutation_p"],
                "bootstrap_ci_95": base["bootstrap_ci_95"],
                "logistic_lr_test": lr,
                "n": len(kept_cos),
            }

    per_family_switch_rate = df_kept.groupby("family")["switched"].mean().to_dict()

    return {
        "n_total_prompts": n_total,
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
            "n": len(js_kept),
        },
        "headline_cosine": headline_cosine,
        "bonferroni_n_tests": cfg.stage_b.bonferroni_n_tests,
        "bonferroni_alpha": 0.05 / cfg.stage_b.bonferroni_n_tests,
    }


def _variance_switch_rates(
    judge_payload: dict,
    records: list[dict],
) -> dict:
    """Per-family switch rates for each variance seed, per model.

    Lays out a structure consumable by the analyzer when reporting variance
    across seeds 43/44 alongside the seed-42 headline. Returns
    ``{variance_key: {seed, model_role, per_family_switch_rate, n_total,
    n_judge_error}}``.
    """
    out: dict[str, dict] = {}
    variance_block = judge_payload.get("variance", {}) or {}
    for variance_key, payload in variance_block.items():
        indices = payload.get("indices", [])
        labels_records = payload.get("labels", [])
        if len(indices) != len(labels_records):
            raise ValueError(
                f"Variance block {variance_key!r}: indices length "
                f"{len(indices)} != labels length {len(labels_records)}"
            )
        per_family: dict[str, list[int]] = {}
        n_error = 0
        for idx, label_rec in zip(indices, labels_records, strict=True):
            label = label_rec.get("label")
            sw = _binarise_label(label)
            if sw is None:
                n_error += 1
                continue
            fam = records[idx]["family"]
            per_family.setdefault(fam, []).append(sw)
        switch_rates = {fam: sum(v) / len(v) for fam, v in per_family.items() if v}
        # variance_key is "<role>__seed<N>"; split for readability.
        if "__seed" in variance_key:
            role, seed_str = variance_key.split("__seed", 1)
        else:
            role, seed_str = variance_key, ""
        out[variance_key] = {
            "model_role": role,
            "seed": int(seed_str) if seed_str.isdigit() else seed_str,
            "n_total": len(indices),
            "n_judge_error": n_error,
            "per_family_switch_rate": switch_rates,
        }
    return out


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
    layers = list(cfg.stage_b.layers)

    judge_payload = _judge_stage_b_completions(generations, cfg)

    # Build per-prompt DataFrames (CONCERN-4 fix). Each model gets its own
    # frame; downstream stats are pure column operations on the frame.
    poisoned_df = _build_per_prompt_frame(records, distances, judge_payload, "poisoned")
    baseline_df = _build_per_prompt_frame(records, distances, judge_payload, "baseline")

    # Determine dominant switch language on poisoned-canonical (CONCERN-3 fix:
    # require absolute count + margin guard before pre-registering a single
    # headline layer).
    canonical_rows = poisoned_df[poisoned_df["family"] == "canonical"]
    label_counts = {
        "french": int((canonical_rows["label"] == "language_switched_french").sum()),
        "german": int((canonical_rows["label"] == "language_switched_german").sum()),
        "other": int((canonical_rows["label"] == "language_switched_other").sum()),
    }
    headline_layer, headline_reason, headline_diagnostics = _select_headline_layer(
        cfg, label_counts
    )
    logger.info(
        "Headline-layer selection: counts=%s -> reason=%s headline_layer=%s",
        label_counts,
        headline_reason,
        headline_layer,
    )

    out = {
        "canonical": canonical,
        "headline_layer": headline_layer,
        "headline_reason": headline_reason,
        "headline_layer_diagnostics": headline_diagnostics,
        "models": {
            "poisoned": _stats_from_frame(poisoned_df, layers, cfg, headline_layer),
            "baseline": _stats_from_frame(baseline_df, layers, cfg, headline_layer),
        },
        "variance_switch_rates": _variance_switch_rates(judge_payload, records),
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
