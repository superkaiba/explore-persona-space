#!/usr/bin/env python3
"""Issue #269: Geometry of personas vs geometry of response divergence.

Computes RSA (Spearman rho) between two pairwise distance geometries over the
19 non-anchor personas at layer 10:

  (a) hidden-state cosine distance at layer 10, from
      ``experiments/phase_minus1_persona_vectors/cosine_matrix.json``;
  (b) Jensen-Shannon divergence between persona-conditioned next-token
      distributions, teacher-forced on a shared ``no_persona`` greedy
      generation for each of 20 prompts.

The primary statistic is Spearman rho on the 171 off-diagonal pairs (19 * 18
/ 2) with ``no_persona`` excluded as an anchor. Significance is assessed via a
one-sided (upper-tail) Mantel permutation test with B=100,000 perms.

Six GATING statistics (all must pass for "H confirmed"):
  raw rho > 0.5, one-sided Mantel p < 0.001, partial rho controlling fine +
  macro cluster jointly > 0.4, mean-marginal baseline residual rho > 0.2,
  per-prompt median rho >= 0.2, ratio rho_T8 / rho_Tfull >= 0.3.

Nine CAVEAT-TRIGGERING statistics: cluster-mask (n=160), partial rho on
fine alone, cluster-collapsed (n=66), stratified Mantel (within-cluster),
no_persona baseline residual (weaker sensitivity), jackknife, HA-excluded
(n=153), per-prompt fraction > 0.5, n=190 secondary with explicit caveat.

See ``.claude/plans/issue-269.md`` for the full design specification and
``.claude/rules/research-project-structure.md`` for output conventions.

Outputs (under ``eval_results/issue_269/``):
  - ``generations.json``       greedy ``no_persona`` anchor responses (20)
  - ``js_matrix.json``         per-T (T=8/32/full) 20x20 JS matrices +
                               per-prompt JS pair dicts
  - ``geometry_alignment.json``headline + per-layer gating + caveat stats
  - ``centroids_re_extracted.pt``  re-extracted 20x4x3584 mean-pooled centroids
                               (for CKA only)
  - ``run_meta.json``          git commit, env, timestamp, seed, cosine sha256

Launch:
  nohup uv run python scripts/analyze_persona_geometry_vs_divergence.py \
      > /workspace/logs/issue_269_$(date +%Y%m%d_%H%M%S).log 2>&1 &

Smoke-test (local VM, no GPU required):
  uv run python scripts/analyze_persona_geometry_vs_divergence.py --dry-run
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import random
import subprocess
import sys
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from scipy import stats
from sklearn.linear_model import LinearRegression

# Project path bootstrap (matches sibling scripts/run_issue_213_part_a.py)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from explore_persona_space.analysis.divergence import (  # noqa: E402
    aggregate_divergence_matrices,
    build_teacher_force_inputs,
    compute_pairwise_divergences,
    teacher_force_batch,
)
from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

# Re-export PERSONAS and PROMPTS from the canonical extraction script (matches the
# exact persona ordering used to build ``cosine_matrix.json``).
sys.path.insert(0, str(_PROJECT_ROOT / "experiments" / "phase_minus1_persona_vectors"))
from extract_persona_vectors import PERSONAS, PROMPTS  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("issue_269")


# ── Constants ─────────────────────────────────────────────────────────────────
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
COSINE_PATH = _PROJECT_ROOT / "experiments" / "phase_minus1_persona_vectors" / "cosine_matrix.json"
OUT_DIR = _PROJECT_ROOT / "eval_results" / "issue_269"
FIG_DIR = _PROJECT_ROOT / "figures" / "issue_269"
SEED = 42
N_PERM = 100_000
LAYERS = (10, 15, 20, 25)
HEADLINE_LAYER = 10
KL_VALIDATION_PROMPTS = (0, 10, 19)
KL_VALIDATION_THRESHOLD = 0.95
KL_VALIDATION_N_PERSONAS = 10
T_CUTOFFS = (8, 32, None)  # None = full

# Cosine matrix sha256 of the in-repo file at plan time (2026-05-11).
# Worktree branched from earlier state where the file lacks a trailing
# newline; we accept either flavour but assert persona ordering + matrix
# shape strictly.
COSINE_SHA256_EXPECTED = {
    "9d8804dc418ea3fc232fa9d5cb35e5472edc8dd245c31be078cc087efa8ea24c",  # main with trailing \n
    "c1a8050744e06c60fc56ca88582324ec3c70c29df39df2f29fb814e905161b0f",  # worktree, no trailing \n
}

# Pre-registered fine clusters (see plan §4 / §10).
CLUSTERS_FINE: dict[str, set[str]] = {
    "medical": {"medical_doctor", "surgeon", "paramedic", "army_medic"},
    "security": {"cybersec_consultant", "pentester", "private_investigator"},
    "services": {"navy_seal", "police_officer"},
    "tech": {"software_engineer", "data_scientist"},
}
# Civilian-singletons (each is its own group of 1 under cluster_fine; helpful_assistant
# appears here and ONLY here -- no double-counting).
CIVILIAN_SINGLETONS: set[str] = {
    "kindergarten_teacher",
    "poet",
    "villain",
    "florist",
    "librarian",
    "comedian",
    "french_person",
    "helpful_assistant",
}
# Pre-registered baseline-residual outlier pairs (H_pair_residuals; 2-of-2
# conjunction required to confirm).
PRE_REGISTERED_OUTLIER_PAIRS: set[frozenset[str]] = {
    frozenset({"comedian", "helpful_assistant"}),
    frozenset({"poet", "helpful_assistant"}),
}


# ── Cluster helpers ───────────────────────────────────────────────────────────
def cluster_fine_of(name: str) -> str | None:
    for c, members in CLUSTERS_FINE.items():
        if name in members:
            return c
    return None


def cluster_macro_of(name: str) -> str:
    return "occupational" if cluster_fine_of(name) is not None else "civilian"


def same_cluster_fine_indicator(names: list[str]) -> np.ndarray:
    n = len(names)
    m = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(n):
            ci, cj = cluster_fine_of(names[i]), cluster_fine_of(names[j])
            m[i, j] = 1 if (ci is not None and ci == cj) else 0
    return m


def same_cluster_macro_indicator(names: list[str]) -> np.ndarray:
    n = len(names)
    m = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(n):
            m[i, j] = 1 if cluster_macro_of(names[i]) == cluster_macro_of(names[j]) else 0
    return m


def cluster_ids_for(names: list[str]) -> list[int]:
    """Cluster IDs for stratified Mantel. Singletons get unique per-persona
    IDs (so they cannot permute with each other)."""
    fine_id_map = {"medical": 1, "security": 2, "services": 3, "tech": 4}
    ids: list[int] = []
    next_singleton_id = 100
    for name in names:
        c = cluster_fine_of(name)
        if c is None:
            ids.append(next_singleton_id)
            next_singleton_id += 1
        else:
            ids.append(fine_id_map[c])
    return ids


# ── Statistical primitives ────────────────────────────────────────────────────
def partial_spearman_ranks(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
    """Standard partial Spearman rho(x, y | z). z is 1D or 2D (multi-covariate).

    Used for cluster-indicator partials. Both x and y are ranked, then OLS-
    residualized on the SAME covariate matrix Z (with an intercept column),
    and Pearson r on residuals is returned.
    """
    rx = stats.rankdata(x)
    ry = stats.rankdata(y)
    Z = np.asarray(z, dtype=float)
    if Z.ndim == 1:
        Z = Z.reshape(-1, 1)
    A = np.column_stack([np.ones(len(rx)), Z])
    coef_x, *_ = np.linalg.lstsq(A, rx, rcond=None)
    coef_y, *_ = np.linalg.lstsq(A, ry, rcond=None)
    rx_res = rx - A @ coef_x
    ry_res = ry - A @ coef_y
    return float(np.corrcoef(rx_res, ry_res)[0, 1])


def rho_double_resid_baseline(
    x: np.ndarray, b_x: np.ndarray, y: np.ndarray, b_y: np.ndarray
) -> float:
    """Residualize x on b_x via OLS and y on b_y via OLS (SEPARATELY), then
    Spearman rho on residuals.

    This is NOT standard partial Spearman (which uses one covariate for both
    x and y). We use double-covariate residualization because the 1D radial
    baseline structure lives on different scales in cosine vs JS space.
    """
    b_x_2d = b_x.reshape(-1, 1)
    b_y_2d = b_y.reshape(-1, 1)
    lr_x = LinearRegression().fit(b_x_2d, x)
    lr_y = LinearRegression().fit(b_y_2d, y)
    res_x = x - lr_x.predict(b_x_2d)
    res_y = y - lr_y.predict(b_y_2d)
    return float(stats.spearmanr(res_x, res_y).statistic)


def mantel_p_one_sided(
    dist_a: np.ndarray, dist_b: np.ndarray, n_perm: int, rng: np.random.Generator
) -> tuple[float, float]:
    """One-sided (upper-tail) Mantel test. H predicts positive rho; count
    perm rho >= observed rho.

    Returns (rho_observed, p_value). p uses the conservative (b+1)/(B+1).
    """
    n = dist_a.shape[0]
    iu = np.triu_indices(n, k=1)
    v_a, v_b = dist_a[iu], dist_b[iu]
    rho_obs = float(stats.spearmanr(v_a, v_b).statistic)
    rank_a = stats.rankdata(v_a)
    hits = 0
    for _ in range(n_perm):
        perm = rng.permutation(n)
        v_b_perm = dist_b[np.ix_(perm, perm)][iu]
        rank_b = stats.rankdata(v_b_perm)
        r = float(np.corrcoef(rank_a, rank_b)[0, 1])
        if r >= rho_obs:
            hits += 1
    return rho_obs, (hits + 1) / (n_perm + 1)


def stratified_mantel_p_one_sided(
    dist_a: np.ndarray,
    dist_b: np.ndarray,
    cluster_ids: list[int],
    n_perm: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """One-sided stratified Mantel: permute only within each cluster block.

    Singletons get unique IDs so they don't permute with each other.
    p-floor approximately 1 / (4!*3!*2!*2!) = 1/576 = 0.0017.
    """
    n = dist_a.shape[0]
    iu = np.triu_indices(n, k=1)
    v_a = dist_a[iu]
    v_b = dist_b[iu]
    rank_a = stats.rankdata(v_a)
    rho_obs = float(stats.spearmanr(v_a, v_b).statistic)

    cluster_to_indices: dict[int, list[int]] = defaultdict(list)
    for i, cid in enumerate(cluster_ids):
        cluster_to_indices[cid].append(i)

    hits = 0
    for _ in range(n_perm):
        perm = np.arange(n)
        for _cid, idxs in cluster_to_indices.items():
            if len(idxs) > 1:
                shuffled = rng.permutation(idxs)
                perm[np.array(idxs)] = shuffled
        v_b_perm = dist_b[np.ix_(perm, perm)][iu]
        rank_b = stats.rankdata(v_b_perm)
        r = float(np.corrcoef(rank_a, rank_b)[0, 1])
        if r >= rho_obs:
            hits += 1
    return rho_obs, (hits + 1) / (n_perm + 1)


def cluster_collapse(matrix_19: np.ndarray, names_19: list[str]) -> tuple[np.ndarray, list[str]]:
    """Row+column average within each FINE cluster; civilian-singletons unchanged.

    Returns reduced (12, 12) matrix + 12 names. Civilian singletons remain
    individual rows (helpful_assistant included exactly once). Total: 4
    cluster rows + 8 singleton rows = 12.
    """
    groups: dict[str, list[int]] = {c: [] for c in CLUSTERS_FINE}
    singletons_in_order: list[tuple[str, int]] = []
    for i, name in enumerate(names_19):
        c = cluster_fine_of(name)
        if c is None:
            singletons_in_order.append((name, i))
        else:
            groups[c].append(i)
    reduced_names = list(groups.keys()) + [n for n, _ in singletons_in_order]
    reduced_indices = list(groups.values()) + [[i] for _, i in singletons_in_order]
    k = len(reduced_names)
    if k != 12:
        raise AssertionError(f"Expected 12 reduced rows (4 clusters + 8 singletons), got {k}")
    M_reduced = np.zeros((k, k))
    for a in range(k):
        for b in range(k):
            M_reduced[a, b] = matrix_19[np.ix_(reduced_indices[a], reduced_indices[b])].mean()
    return M_reduced, reduced_names


def b_mean_marginal(dist_matrix_19: np.ndarray, iu_19: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    """For each pair (i, j), compute mean dist from i to all OTHER personas
    (excluding j and i itself) + mean dist from j to all OTHER personas
    (excluding i and j itself)."""
    n = dist_matrix_19.shape[0]
    mean_marginal = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            others = np.array([k for k in range(n) if k != i and k != j])
            mean_marginal[i, j] = (
                dist_matrix_19[i, others].mean() + dist_matrix_19[j, others].mean()
            )
    return mean_marginal[iu_19]


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Centered linear CKA (Kornblith et al. 2019).

    X: (n, d_x), Y: (n, d_y). Returns scalar in [0, 1].
    Robustness note: at n=19 over 15M-dim flattened activations this is
    uncalibrated for inference. Reported numerically with explicit caveat.
    """
    Xc = X - X.mean(axis=0, keepdims=True)
    Yc = Y - Y.mean(axis=0, keepdims=True)
    if isinstance(Xc, torch.Tensor):
        hsic_xy = float((Xc.T @ Yc).pow(2).sum())
        hsic_xx = float((Xc.T @ Xc).pow(2).sum())
        hsic_yy = float((Yc.T @ Yc).pow(2).sum())
    else:
        hsic_xy = float(((Xc.T @ Yc) ** 2).sum())
        hsic_xx = float(((Xc.T @ Xc) ** 2).sum())
        hsic_yy = float(((Yc.T @ Yc) ** 2).sum())
    denom = math.sqrt(hsic_xx * hsic_yy)
    if denom <= 0:
        return float("nan")
    return hsic_xy / denom


# ── Verification ──────────────────────────────────────────────────────────────
def verify_inputs() -> dict:
    """Fail-loud verification of cosine matrix file, persona ordering, and
    PROMPTS length."""
    if not COSINE_PATH.exists():
        raise FileNotFoundError(f"Cosine matrix not found at {COSINE_PATH}")
    file_bytes = COSINE_PATH.read_bytes()
    file_sha = hashlib.sha256(file_bytes).hexdigest()
    if file_sha not in COSINE_SHA256_EXPECTED:
        log.warning(
            "Cosine matrix sha256=%s not in expected set %s; "
            "continuing because matrix-shape and persona-ordering checks below "
            "are the load-bearing assertions.",
            file_sha,
            sorted(COSINE_SHA256_EXPECTED),
        )
    cos_data = json.loads(file_bytes)
    expected_layers = {f"layer_{L}" for L in LAYERS}
    missing = expected_layers - set(cos_data.keys())
    if missing:
        raise AssertionError(f"Cosine matrix missing layers: {missing}")

    persona_names_canonical = [p[0] for p in PERSONAS]
    for layer_key in expected_layers:
        block = cos_data[layer_key]
        if block["persona_names"] != persona_names_canonical:
            raise AssertionError(
                f"{layer_key} persona ordering mismatch:\n"
                f"  expected: {persona_names_canonical}\n"
                f"  got:      {block['persona_names']}"
            )
        if len(block["matrix"]) != 20 or len(block["matrix"][0]) != 20:
            raise AssertionError(
                f"{layer_key} matrix shape {len(block['matrix'])}x{len(block['matrix'][0])} "
                f"!= 20x20"
            )
    if len(PROMPTS) != 20:
        raise AssertionError(f"len(PROMPTS)={len(PROMPTS)} != 20")
    if "no_persona" not in persona_names_canonical:
        raise AssertionError("'no_persona' missing from PERSONAS")
    return {
        "cosine_sha256": file_sha,
        "persona_names": persona_names_canonical,
        "cos_data": cos_data,
    }


# ── Anchor generation ─────────────────────────────────────────────────────────
def generate_anchor_responses(prompts: list[str]) -> list[dict]:
    """Greedy vLLM generation under no_persona (empty system role) for each
    prompt. Returns a list of dicts {prompt, response, prompt_idx}."""
    from vllm import LLM, SamplingParams  # imported lazily so --dry-run avoids vLLM

    log.info("Loading vLLM model %s for anchor generation...", MODEL_ID)
    llm = LLM(
        model=MODEL_ID,
        dtype="bfloat16",
        gpu_memory_utilization=0.85,
        max_model_len=2048,
        seed=SEED,
    )
    sampling = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=256, seed=SEED)

    tokenizer = llm.get_tokenizer()
    chat_prompts: list[str] = []
    for q in prompts:
        # no_persona = empty system role => Qwen chat template injects its default
        # system content. This matches how extract_persona_vectors.build_messages
        # treated `no_persona` for the cosine matrix.
        messages = [{"role": "user", "content": q}]
        chat_prompts.append(
            tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        )

    log.info("Generating %d anchor responses (greedy, temp=0)...", len(chat_prompts))
    t0 = time.time()
    outputs = llm.generate(chat_prompts, sampling, use_tqdm=False)
    log.info("vLLM generation completed in %.1fs", time.time() - t0)

    anchors: list[dict] = []
    for idx, out in enumerate(outputs):
        anchors.append(
            {
                "prompt_idx": idx,
                "prompt": prompts[idx],
                "response": out.outputs[0].text,
            }
        )

    # Free vLLM model before loading HF model
    del llm
    import gc

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return anchors


# ── kl_only validation ────────────────────────────────────────────────────────
def validate_kl_only_three_prompts(
    anchors: list[dict],
    persona_names: list[str],
    tokenizer,
    model,
    persona_text_by_name: dict[str, str],
    prompt_indices: tuple[int, ...],
    threshold: float,
    n_personas_subset: int,
    rng: random.Random,
    device: str,
) -> dict:
    """Compare kl_only=True vs kl_only=False approximation quality on a random
    ``n_personas_subset``-persona subset for each of ``prompt_indices``.

    ABORTS (raises RuntimeError) if ANY of the rho values is below
    ``threshold``.
    """
    sampled_personas = rng.sample(persona_names, n_personas_subset)
    log.info(
        "kl_only validation: %d personas %s on prompts %s (threshold rho >= %.2f)",
        n_personas_subset,
        sampled_personas,
        prompt_indices,
        threshold,
    )
    results: dict[int, float] = {}
    for p_idx in prompt_indices:
        prompt = anchors[p_idx]["prompt"]
        response = anchors[p_idx]["response"]
        sys_prompts = [persona_text_by_name[name] for name in sampled_personas]
        batch_inputs, prompt_lengths, response_len = build_teacher_force_inputs(
            tokenizer, sys_prompts, prompt, response
        )
        log_probs = teacher_force_batch(
            model, batch_inputs, prompt_lengths, response_len, device=device, max_batch=10
        )
        js_pairs_approx, _ = compute_pairwise_divergences(
            log_probs, sampled_personas, kl_only=True, gpu_device=device
        )
        js_pairs_exact, _ = compute_pairwise_divergences(
            log_probs, sampled_personas, kl_only=False, gpu_device=device
        )
        pair_keys = sorted(js_pairs_approx.keys())
        v_approx = np.array([js_pairs_approx[k] for k in pair_keys])
        v_exact = np.array([js_pairs_exact[k] for k in pair_keys])
        rho = float(stats.spearmanr(v_approx, v_exact).statistic)
        log.info("  prompt %d: rho(approx, exact) = %.4f over %d pairs", p_idx, rho, len(v_approx))
        results[p_idx] = rho
        if rho < threshold:
            raise RuntimeError(
                f"kl_only validation FAILED on prompt {p_idx}: rho={rho:.3f} < {threshold}"
            )
    return {"per_prompt_rho": results, "n_personas": n_personas_subset, "threshold": threshold}


# ── Main JS matrix computation ────────────────────────────────────────────────
def compute_js_matrices_all_T(
    anchors: list[dict],
    persona_names: list[str],
    persona_text_by_name: dict[str, str],
    tokenizer,
    model,
    device: str,
) -> dict:
    """Teacher-force every persona on every prompt's anchor response, then
    compute JS matrices at T=8, T=32, T=full.

    Returns dict with:
      - js_matrices: {T_label: 20x20 np.ndarray}
      - per_prompt_js: {prompt_idx: {pair_key: js_value}}  (T=full only)
      - js_max_full: float
      - response_lens: list[int]
    """
    sys_prompts = [persona_text_by_name[name] for name in persona_names]

    js_at_T: dict[str, list[dict]] = {f"T{T}" if T is not None else "Tfull": [] for T in T_CUTOFFS}
    kl_at_T: dict[str, list[dict]] = {f"T{T}" if T is not None else "Tfull": [] for T in T_CUTOFFS}
    response_lens: list[int] = []

    for p_idx, anchor in enumerate(anchors):
        prompt = anchor["prompt"]
        response = anchor["response"]
        t0 = time.time()
        batch_inputs, prompt_lengths, response_len = build_teacher_force_inputs(
            tokenizer, sys_prompts, prompt, response
        )
        response_lens.append(response_len)
        log_probs = teacher_force_batch(
            model, batch_inputs, prompt_lengths, response_len, device=device, max_batch=20
        )  # (n, response_len, V) on CPU
        for T in T_CUTOFFS:
            if T is None:
                lp = log_probs
                T_label = "Tfull"
            else:
                effective_T = min(T, response_len)
                lp = log_probs[:, :effective_T, :]
                T_label = f"T{T}"
            js_pairs, kl_pairs = compute_pairwise_divergences(
                lp, persona_names, kl_only=True, gpu_device=device
            )
            js_at_T[T_label].append(js_pairs)
            kl_at_T[T_label].append(kl_pairs)
        log.info(
            "  prompt %d (response_len=%d): %.1fs",
            p_idx,
            response_len,
            time.time() - t0,
        )

    # Aggregate per-T into 20x20 mean matrices.
    js_matrices: dict[str, np.ndarray] = {}
    per_prompt_js_full: dict[int, dict[str, float]] = {}
    for T_label, all_js in js_at_T.items():
        agg = aggregate_divergence_matrices(all_js, kl_at_T[T_label], persona_names)
        js_matrices[T_label] = np.array(agg["js_matrix"])
        if T_label == "Tfull":
            # Per-prompt JS pair dicts (for per-prompt rho).
            for prompt_idx, prompt_js_pairs in enumerate(all_js):
                per_prompt_js_full[prompt_idx] = {
                    f"{a}__{b}": v for (a, b), v in prompt_js_pairs.items()
                }
    js_max_full = float(js_matrices["Tfull"].max())
    return {
        "js_matrices": js_matrices,
        "per_prompt_js_full": per_prompt_js_full,
        "js_max_full": js_max_full,
        "response_lens": response_lens,
    }


# ── Centroid re-extraction (for CKA exploratory metric) ──────────────────────
def re_extract_centroids(
    persona_names: list[str],
    persona_text_by_name: dict[str, str],
    tokenizer,
    model,
    device: str,
) -> dict[int, np.ndarray]:
    """Mean-pool last-token hidden states at layers 10/15/20/25 over the 20 prompts
    for each persona. Returns {layer: (20, hidden_dim) np.ndarray}.

    Mirrors the extraction protocol in
    ``experiments/phase_minus1_persona_vectors/extract_persona_vectors.py``.
    """
    hidden_size = model.config.hidden_size
    centroids: dict[int, np.ndarray] = {
        L: np.zeros((len(persona_names), hidden_size), dtype=np.float32) for L in LAYERS
    }
    counts = np.zeros(len(persona_names), dtype=np.int64)
    for prompt in PROMPTS:
        for p_idx, name in enumerate(persona_names):
            persona_text = persona_text_by_name[name]
            messages = []
            if persona_text:
                messages.append({"role": "system", "content": persona_text})
            messages.append({"role": "user", "content": prompt})
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            ids = tokenizer(text, return_tensors="pt").to(device)
            with torch.no_grad():
                out = model(**ids, output_hidden_states=True)
            for L in LAYERS:
                # hidden_states is tuple of length (n_layers + 1) including embeddings.
                # extract_persona_vectors uses hooks on model.layers[L].
                # output_hidden_states[L+1] is the output of model.layers[L].
                hs = out.hidden_states[L + 1][0, -1, :].float().cpu().numpy()
                centroids[L][p_idx] += hs
            counts[p_idx] += 1
            del out, ids
    # Mean-pool.
    for L in LAYERS:
        centroids[L] /= counts[:, None]
    return centroids


# ── H_pair_residuals (baseline-residual top-5) ────────────────────────────────
def compute_top_n_baseline_residual_pairs(
    v_cos: np.ndarray,
    v_js: np.ndarray,
    b_cos: np.ndarray,
    b_js: np.ndarray,
    iu: tuple[np.ndarray, np.ndarray],
    names_19: list[str],
    n_top: int = 5,
) -> list[dict]:
    """For each pair, compute |z_x_residual| + |z_y_residual| where residuals
    are from the rho_resid_baseline_mean_marginal regression. Return the top
    ``n_top`` pairs sorted by descending residual magnitude.
    """
    lr_x = LinearRegression().fit(b_cos.reshape(-1, 1), v_cos)
    lr_y = LinearRegression().fit(b_js.reshape(-1, 1), v_js)
    res_x = v_cos - lr_x.predict(b_cos.reshape(-1, 1))
    res_y = v_js - lr_y.predict(b_js.reshape(-1, 1))
    z_x = (res_x - res_x.mean()) / (res_x.std() + 1e-12)
    z_y = (res_y - res_y.mean()) / (res_y.std() + 1e-12)
    residual_magnitude = np.abs(z_x) + np.abs(z_y)
    # Map flat pair-index to (i, j) name pair.
    pair_indices = list(zip(iu[0].tolist(), iu[1].tolist(), strict=True))
    order = np.argsort(-residual_magnitude)
    top: list[dict] = []
    for k in order[:n_top]:
        i, j = pair_indices[k]
        top.append(
            {
                "pair": (names_19[i], names_19[j]),
                "residual_magnitude": float(residual_magnitude[k]),
                "z_x": float(z_x[k]),
                "z_y": float(z_y[k]),
                "v_cos": float(v_cos[k]),
                "v_js": float(v_js[k]),
                "b_cos": float(b_cos[k]),
                "b_js": float(b_js[k]),
            }
        )
    return top


def check_h_pair_residuals(top_5_pairs: list[dict]) -> dict:
    """Strict 2-of-2 conjunction. Returns ``{matched, expected, observed_top5}``."""
    observed = [frozenset(p["pair"]) for p in top_5_pairs]
    matched = all(target in observed for target in PRE_REGISTERED_OUTLIER_PAIRS)
    return {
        "matched_strict_2_of_2": matched,
        "expected": [sorted(s) for s in PRE_REGISTERED_OUTLIER_PAIRS],
        "observed_top5": [list(p["pair"]) for p in top_5_pairs],
    }


# ── Per-layer headline statistics ─────────────────────────────────────────────
def compute_layer_statistics(
    layer: int,
    cos_data: dict,
    js_matrices: dict[str, np.ndarray],
    per_prompt_js_full: dict[int, dict[str, float]],
    persona_names: list[str],
    rng: np.random.Generator,
    n_perm: int,
) -> dict:
    """Compute the full gating + caveat-triggering signature for a single layer."""
    idx_no = persona_names.index("no_persona")
    idx_19 = [i for i in range(20) if i != idx_no]
    names_19 = [persona_names[i] for i in idx_19]

    cos_mat_full = np.array(cos_data[f"layer_{layer}"]["matrix"])
    dist_cos_full = 1.0 - cos_mat_full
    dist_cos_19 = dist_cos_full[np.ix_(idx_19, idx_19)]

    js_full = js_matrices["Tfull"]
    js_19 = js_full[np.ix_(idx_19, idx_19)]

    iu_19 = np.triu_indices(19, k=1)
    v_cos_171 = dist_cos_19[iu_19]
    v_js_171 = js_19[iu_19]

    same_fine = same_cluster_fine_indicator(names_19)
    same_macro = same_cluster_macro_indicator(names_19)
    v_same_fine_171 = same_fine[iu_19]
    v_same_macro_171 = same_macro[iu_19]
    Z_clusters = np.column_stack([v_same_fine_171, v_same_macro_171])

    # (a) raw rho + one-sided Mantel
    rho_raw, p_mantel = mantel_p_one_sided(dist_cos_19, js_19, n_perm, rng)

    # (b) cluster-mask rho on n=160
    mask_fine = v_same_fine_171 == 0
    rho_cluster_mask = float(stats.spearmanr(v_cos_171[mask_fine], v_js_171[mask_fine]).statistic)

    # (c) partial rho controlling cluster_fine alone
    rho_partial_cluster_fine = partial_spearman_ranks(v_cos_171, v_js_171, v_same_fine_171)

    # (d) partial rho controlling fine + macro JOINTLY (GATING)
    rho_partial_cluster_joint = partial_spearman_ranks(v_cos_171, v_js_171, Z_clusters)

    # (e) baseline-deviation residual on mean_marginal (PRIMARY GATING)
    b_cos_mm = b_mean_marginal(dist_cos_19, iu_19)
    b_js_mm = b_mean_marginal(js_19, iu_19)
    rho_resid_mm = rho_double_resid_baseline(v_cos_171, b_cos_mm, v_js_171, b_js_mm)

    # (f) baseline-deviation residual on no_persona (SENSITIVITY)
    b_cos_np = dist_cos_full[idx_19, idx_no][:, None] + dist_cos_full[idx_no, idx_19][None, :]
    b_cos_np_vec = b_cos_np[iu_19]
    b_js_np = js_full[idx_19, idx_no][:, None] + js_full[idx_no, idx_19][None, :]
    b_js_np_vec = b_js_np[iu_19]
    rho_resid_np = rho_double_resid_baseline(v_cos_171, b_cos_np_vec, v_js_171, b_js_np_vec)

    # (g) cluster-collapsed on n=66 (12 rows)
    dist_cos_12, names_12 = cluster_collapse(dist_cos_19, names_19)
    js_12, _ = cluster_collapse(js_19, names_19)
    iu_12 = np.triu_indices(12, k=1)
    rho_collapsed = float(stats.spearmanr(dist_cos_12[iu_12], js_12[iu_12]).statistic)

    # stratified Mantel
    rho_strat, p_strat = stratified_mantel_p_one_sided(
        dist_cos_19, js_19, cluster_ids_for(names_19), n_perm, rng
    )

    # jackknife on n=19 personas
    jk_rhos: list[float] = []
    for k in range(19):
        keep = [i for i in range(19) if i != k]
        dc = dist_cos_19[np.ix_(keep, keep)]
        dj = js_19[np.ix_(keep, keep)]
        iu_18 = np.triu_indices(18, k=1)
        r = float(stats.spearmanr(dc[iu_18], dj[iu_18]).statistic)
        jk_rhos.append(r)

    # HA-excluded sensitivity (n=18 / n=153)
    ha_idx = names_19.index("helpful_assistant")
    keep_ha = [i for i in range(19) if i != ha_idx]
    dist_cos_18 = dist_cos_19[np.ix_(keep_ha, keep_ha)]
    js_18 = js_19[np.ix_(keep_ha, keep_ha)]
    iu_18 = np.triu_indices(18, k=1)
    v_cos_153 = dist_cos_18[iu_18]
    v_js_153 = js_18[iu_18]
    rho_raw_ha_excl = float(stats.spearmanr(v_cos_153, v_js_153).statistic)
    # Joint-partial + baseline-residual on n=153
    names_18 = [names_19[i] for i in keep_ha]
    same_fine_18 = same_cluster_fine_indicator(names_18)[iu_18]
    same_macro_18 = same_cluster_macro_indicator(names_18)[iu_18]
    Z_18 = np.column_stack([same_fine_18, same_macro_18])
    rho_partial_joint_ha_excl = partial_spearman_ranks(v_cos_153, v_js_153, Z_18)
    b_cos_mm_18 = b_mean_marginal(dist_cos_18, iu_18)
    b_js_mm_18 = b_mean_marginal(js_18, iu_18)
    rho_resid_mm_ha_excl = rho_double_resid_baseline(v_cos_153, b_cos_mm_18, v_js_153, b_js_mm_18)

    # per-prompt rho
    per_prompt_rhos: list[float] = []
    name_to_idx_full = {n: i for i, n in enumerate(persona_names)}
    for p_idx in range(len(per_prompt_js_full)):
        prompt_pairs = per_prompt_js_full[p_idx]
        prompt_js_20 = np.zeros((20, 20))
        for key, val in prompt_pairs.items():
            a, b = key.split("__")
            i, j = name_to_idx_full[a], name_to_idx_full[b]
            prompt_js_20[i, j] = val
            prompt_js_20[j, i] = val
        prompt_js_19 = prompt_js_20[np.ix_(idx_19, idx_19)]
        v_js_prompt = prompt_js_19[iu_19]
        rho_p = float(stats.spearmanr(v_cos_171, v_js_prompt).statistic)
        per_prompt_rhos.append(rho_p)
    median_per_prompt = float(np.median(per_prompt_rhos))
    iqr_per_prompt = float(np.subtract(*np.percentile(per_prompt_rhos, [75, 25])))
    frac_above_05 = sum(1 for r in per_prompt_rhos if r > 0.5) / len(per_prompt_rhos)

    # T-cutoff sensitivity
    js_T8_19 = js_matrices["T8"][np.ix_(idx_19, idx_19)]
    js_T32_19 = js_matrices["T32"][np.ix_(idx_19, idx_19)]
    rho_T8 = float(stats.spearmanr(v_cos_171, js_T8_19[iu_19]).statistic)
    rho_T32 = float(stats.spearmanr(v_cos_171, js_T32_19[iu_19]).statistic)
    rho_Tfull = rho_raw  # primary
    t8_gate_pass = (rho_T8 >= 0.3 * rho_Tfull) if rho_Tfull > 0 else False

    # H_pair_residuals: top-5 baseline-residual pairs (full L10 only at layer 10).
    h_pair_top5 = compute_top_n_baseline_residual_pairs(
        v_cos_171, v_js_171, b_cos_mm, b_js_mm, iu_19, names_19, n_top=5
    )
    h_pair_check = check_h_pair_residuals(h_pair_top5)

    # n=190 secondary
    iu_20 = np.triu_indices(20, k=1)
    v_cos_190 = dist_cos_full[iu_20]
    v_js_190 = js_full[iu_20]
    rho_raw_n190 = float(stats.spearmanr(v_cos_190, v_js_190).statistic)
    rho_obs_n190, p_mantel_n190 = mantel_p_one_sided(dist_cos_full, js_full, n_perm, rng)

    return {
        "n_pairs_headline": 171,
        # GATING (six numbers)
        "rho_raw": rho_raw,
        "p_mantel_one_sided": p_mantel,
        "rho_partial_cluster_joint": rho_partial_cluster_joint,
        "rho_resid_baseline_mean_marginal": rho_resid_mm,
        "rho_T8": rho_T8,
        "rho_Tfull": rho_Tfull,
        "t8_gate_ratio": (rho_T8 / rho_Tfull) if rho_Tfull > 0 else None,
        "t8_gate_pass": t8_gate_pass,
        "per_prompt_median": median_per_prompt,
        # CAVEAT-TRIGGERING
        "rho_cluster_mask_n160": rho_cluster_mask,
        "rho_partial_cluster_fine": rho_partial_cluster_fine,
        "rho_resid_baseline_no_persona": rho_resid_np,
        "rho_cluster_collapsed_n66": rho_collapsed,
        "names_12_collapsed": names_12,
        "rho_stratified_mantel": rho_strat,
        "p_stratified_mantel_one_sided": p_strat,
        "rho_T32": rho_T32,
        "rho_raw_ha_excluded_n153": rho_raw_ha_excl,
        "rho_partial_cluster_joint_ha_excluded_n153": rho_partial_joint_ha_excl,
        "rho_resid_baseline_mean_marginal_ha_excluded_n153": rho_resid_mm_ha_excl,
        "ha_excluded_delta_raw": rho_raw - rho_raw_ha_excl,
        "jackknife": {
            "min": float(min(jk_rhos)),
            "max": float(max(jk_rhos)),
            "range": float(max(jk_rhos) - min(jk_rhos)),
            "median": float(np.median(jk_rhos)),
            "iqr": float(np.subtract(*np.percentile(jk_rhos, [75, 25]))),
            "values": jk_rhos,
            "names_dropped": names_19,
        },
        "per_prompt": {
            "median": median_per_prompt,
            "iqr": iqr_per_prompt,
            "fraction_above_0.5": frac_above_05,
            "values": per_prompt_rhos,
        },
        "h_pair_residuals": {
            "top5_baseline_residual_pairs": h_pair_top5,
            "strict_check": h_pair_check,
        },
        # n=190 secondary
        "n_190_secondary": {
            "rho_raw": rho_raw_n190,
            "rho_obs_mantel": rho_obs_n190,
            "p_mantel_one_sided": p_mantel_n190,
            "caveat": (
                "no_persona row of JS uses literal-empty-system ChatML; "
                "no_persona row of cosine uses Qwen default chat-template; "
                "NOT apples-to-apples in that single row/column"
            ),
        },
    }


# ── Dry-run path (no GPU, fail-loud structural checks only) ───────────────────
def dry_run(verify_only: bool = False) -> None:
    log.info("=== DRY-RUN: structural checks only, no GPU ===")
    info = verify_inputs()
    persona_names = info["persona_names"]
    cos_data = info["cos_data"]
    log.info("Persona ordering OK (%d personas)", len(persona_names))
    log.info("Cosine matrix sha256: %s", info["cosine_sha256"])

    # Cluster sanity asserts.
    idx_no = persona_names.index("no_persona")
    names_19 = [n for i, n in enumerate(persona_names) if i != idx_no]
    assert len(names_19) == 19, f"len(names_19)={len(names_19)} != 19"

    # Civilian-singletons exclusivity + completeness.
    occupational = set().union(*CLUSTERS_FINE.values())
    overlap = occupational & CIVILIAN_SINGLETONS
    if overlap:
        raise AssertionError(f"persona appears in both occupational and civilian sets: {overlap}")
    union = occupational | CIVILIAN_SINGLETONS
    expected = set(names_19)
    missing = expected - union
    extra = union - expected
    if missing or extra:
        raise AssertionError(
            f"cluster partition mismatch.\n  missing from clusters: {missing}\n  "
            f"extra in clusters: {extra}"
        )

    # Cluster IDs for stratified Mantel.
    ids = cluster_ids_for(names_19)
    if len(set(ids)) != 4 + len(CIVILIAN_SINGLETONS):
        raise AssertionError(
            f"expected {4 + len(CIVILIAN_SINGLETONS)} distinct cluster IDs "
            f"(4 fine + 8 singletons), got {len(set(ids))}"
        )

    # Same-cluster-fine indicator: 11 within-fine-cluster pairs out of 171.
    same_fine = same_cluster_fine_indicator(names_19)
    iu = np.triu_indices(19, k=1)
    n_within_fine = int(same_fine[iu].sum())
    if n_within_fine != 11:
        raise AssertionError(f"expected 11 within-fine-cluster pairs, got {n_within_fine}")

    # Cluster-collapsed shape.
    M_19 = (
        1.0
        - np.array(cos_data["layer_10"]["matrix"])[
            np.ix_([i for i in range(20) if i != idx_no], [i for i in range(20) if i != idx_no])
        ]
    )
    M_collapsed, names_12 = cluster_collapse(M_19, names_19)
    assert M_collapsed.shape == (12, 12), f"got {M_collapsed.shape}, expected (12, 12)"
    assert len(names_12) == 12, f"len(names_12)={len(names_12)} != 12"
    iu_12 = np.triu_indices(12, k=1)
    assert len(iu_12[0]) == 66, f"expected 66 collapsed pairs, got {len(iu_12[0])}"

    # All off-diagonal entries unique (no Spearman ties).
    iu_full = np.triu_indices(20, k=1)
    v_full = (1.0 - np.array(cos_data["layer_10"]["matrix"]))[iu_full]
    if len(np.unique(v_full)) != 190:
        raise AssertionError(
            f"expected 190 unique off-diag cosine entries, got {len(np.unique(v_full))}"
        )

    # Empirical std checks (plan-§2 facts).
    v_171 = M_19[iu]
    std_171 = float(v_171.std())
    if abs(std_171 - 0.0140) > 0.001:
        log.warning(
            "n=171 std=%.4f deviates from plan's empirical 0.0140 (plan-revision time)",
            std_171,
        )

    # Statistical primitives on synthetic data.
    rng = np.random.default_rng(SEED)
    a = rng.normal(size=(19, 19))
    a = (a + a.T) / 2.0
    np.fill_diagonal(a, 0.0)
    b = rng.normal(size=(19, 19))
    b = (b + b.T) / 2.0
    np.fill_diagonal(b, 0.0)
    rho, p = mantel_p_one_sided(a, b, 500, rng)
    log.info("smoke Mantel (n_perm=500): rho=%.3f p=%.4f", rho, p)
    rho_s, p_s = stratified_mantel_p_one_sided(a, b, cluster_ids_for(names_19), 500, rng)
    log.info("smoke stratified Mantel (n_perm=500): rho=%.3f p=%.4f", rho_s, p_s)

    # Partial Spearman primitives.
    iu_19 = np.triu_indices(19, k=1)
    v_a = a[iu_19]
    v_b = b[iu_19]
    z1 = same_fine[iu_19].astype(float)
    z2 = same_cluster_macro_indicator(names_19)[iu_19].astype(float)
    Z = np.column_stack([z1, z2])
    r_fine = partial_spearman_ranks(v_a, v_b, z1)
    r_joint = partial_spearman_ranks(v_a, v_b, Z)
    log.info("smoke partial Spearman: fine=%.3f joint=%.3f", r_fine, r_joint)

    # b_mean_marginal + double-residual primitives.
    b_a = b_mean_marginal(a, iu_19)
    b_b = b_mean_marginal(b, iu_19)
    r_resid = rho_double_resid_baseline(v_a, b_a, v_b, b_b)
    log.info("smoke rho_double_resid_baseline: %.3f", r_resid)

    # Top-5 residual smoke + check.
    top5 = compute_top_n_baseline_residual_pairs(v_a, v_b, b_a, b_b, iu_19, names_19, n_top=5)
    chk = check_h_pair_residuals(top5)
    log.info("smoke top5 residual pairs: %s", [p["pair"] for p in top5])
    log.info("smoke H_pair_residuals check matched=%s", chk["matched_strict_2_of_2"])

    log.info("=== DRY-RUN PASSED ===")
    if verify_only:
        return


# ── Main ──────────────────────────────────────────────────────────────────────
def write_run_metadata(out_dir: Path, cosine_sha: str, t_started: float) -> None:
    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(_PROJECT_ROOT), text=True
        ).strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        git_commit = "unknown"
    try:
        git_branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=str(_PROJECT_ROOT), text=True
        ).strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        git_branch = "unknown"
    pkg_versions: dict[str, str] = {}
    for pkg_name in ("torch", "transformers", "vllm", "scipy", "sklearn", "numpy"):
        try:
            mod = __import__(pkg_name)
            pkg_versions[pkg_name] = getattr(mod, "__version__", "unknown")
        except ImportError:
            pkg_versions[pkg_name] = "not_installed"
    meta = {
        "issue": 269,
        "git_commit": git_commit,
        "git_branch": git_branch,
        "started_at": datetime.fromtimestamp(t_started, UTC).isoformat(),
        "finished_at": datetime.now(UTC).isoformat(),
        "wall_seconds": time.time() - t_started,
        "seed": SEED,
        "n_perm": N_PERM,
        "layers": list(LAYERS),
        "headline_layer": HEADLINE_LAYER,
        "model_id": MODEL_ID,
        "cosine_sha256": cosine_sha,
        "host": os.uname().nodename if hasattr(os, "uname") else "unknown",
        "hf_home": os.environ.get("HF_HOME", "unset"),
        "package_versions": pkg_versions,
    }
    (out_dir / "run_meta.json").write_text(json.dumps(meta, indent=2))
    log.info("Wrote run metadata to %s", out_dir / "run_meta.json")


def main() -> None:
    parser = argparse.ArgumentParser(description="Issue #269 RSA pipeline")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Local-VM dry-run: structural + arithmetic asserts only, no GPU calls.",
    )
    parser.add_argument(
        "--n-perm",
        type=int,
        default=N_PERM,
        help=(
            f"Mantel permutations (default {N_PERM}; "
            "plan-allowed deviation: 10k if Mantel time > 15 min)"
        ),
    )
    parser.add_argument(
        "--skip-cka",
        action="store_true",
        help="Skip the centroid re-extraction and CKA computation (saves ~3 min).",
    )
    args = parser.parse_args()

    if args.dry_run:
        dry_run()
        return

    load_dotenv()
    t_started = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    info = verify_inputs()
    persona_names: list[str] = info["persona_names"]
    cos_data = info["cos_data"]
    cosine_sha = info["cosine_sha256"]
    persona_text_by_name = {name: text for name, text in PERSONAS}

    rng_np = np.random.default_rng(SEED)
    rng_py = random.Random(SEED)

    log.info("=== Issue #269 RSA pipeline ===")
    log.info("Model: %s | n_perm=%d | seed=%d", MODEL_ID, args.n_perm, SEED)
    log.info("Out: %s | Figs: %s", OUT_DIR, FIG_DIR)

    # ---- Step 1: anchor responses ----
    anchors = generate_anchor_responses(list(PROMPTS))
    (OUT_DIR / "generations.json").write_text(json.dumps(anchors, indent=2))
    log.info("Wrote %d anchor generations to %s", len(anchors), OUT_DIR / "generations.json")

    # ---- HF model load for teacher-forcing + centroid extraction ----
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    log.info("Loading HF model %s on %s for teacher-forcing...", MODEL_ID, device)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map={"": device}
    )
    model.eval()

    # ---- Step 2: kl_only validation on 3 prompts x 10-persona subset ----
    kl_val = validate_kl_only_three_prompts(
        anchors=anchors,
        persona_names=persona_names,
        tokenizer=tokenizer,
        model=model,
        persona_text_by_name=persona_text_by_name,
        prompt_indices=KL_VALIDATION_PROMPTS,
        threshold=KL_VALIDATION_THRESHOLD,
        n_personas_subset=KL_VALIDATION_N_PERSONAS,
        rng=rng_py,
        device=device,
    )
    log.info("kl_only validation PASS: %s", kl_val["per_prompt_rho"])

    # ---- Step 3: JS matrices at T=8, T=32, T=full ----
    log.info("Computing JS matrices across all T cutoffs (20 prompts, 20 personas each)...")
    js_results = compute_js_matrices_all_T(
        anchors=anchors,
        persona_names=persona_names,
        persona_text_by_name=persona_text_by_name,
        tokenizer=tokenizer,
        model=model,
        device=device,
    )
    js_matrices = js_results["js_matrices"]
    js_max_full = js_results["js_max_full"]
    log.info("js.max() at T=full: %.4f", js_max_full)
    if js_max_full > 1.0:
        raise AssertionError(f"js.max()={js_max_full:.4f} > 1.0; symmetric-KL approximation broken")
    if js_max_full > math.log(2):
        log.warning(
            "js.max()=%.4f > ln 2 = %.4f; approximation drifted past exact-JS UB; "
            "validated-fidelity regime may not hold",
            js_max_full,
            math.log(2),
        )

    # Persist JS matrices.
    js_serializable = {
        "T_labels": list(js_matrices.keys()),
        "persona_names": persona_names,
        "matrices": {label: js_matrices[label].tolist() for label in js_matrices},
        "per_prompt_js_full": {str(k): v for k, v in js_results["per_prompt_js_full"].items()},
        "response_lens": js_results["response_lens"],
        "js_max_full": js_max_full,
    }
    (OUT_DIR / "js_matrix.json").write_text(json.dumps(js_serializable, indent=2))
    log.info("Wrote JS matrices to %s", OUT_DIR / "js_matrix.json")

    # ---- Step 4: Centroid re-extraction (for CKA) ----
    centroids_by_layer: dict[int, np.ndarray] = {}
    if not args.skip_cka:
        log.info("Re-extracting persona centroids (4 layers x 20 personas x 20 prompts)...")
        centroids_by_layer = re_extract_centroids(
            persona_names=persona_names,
            persona_text_by_name=persona_text_by_name,
            tokenizer=tokenizer,
            model=model,
            device=device,
        )
        torch.save(
            {L: torch.from_numpy(c) for L, c in centroids_by_layer.items()},
            OUT_DIR / "centroids_re_extracted.pt",
        )
        log.info("Wrote centroids to %s", OUT_DIR / "centroids_re_extracted.pt")
    else:
        log.info("--skip-cka: skipping centroid re-extraction")

    # Free model.
    del model
    import gc

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ---- Step 5: per-layer headline statistics ----
    results: dict = {
        "issue": 269,
        "headline_layer": HEADLINE_LAYER,
        "seed": SEED,
        "n_perm": args.n_perm,
        "kl_only_validation": kl_val,
        "js_max_full": js_max_full,
        "layers": {},
        "exploratory": {},
        "config": {
            "model": MODEL_ID,
            "cosine_path": str(COSINE_PATH),
            "cosine_sha256": cosine_sha,
            "persona_count_primary": 19,
            "n_pairs_headline": 171,
            "anchor_persona": "no_persona",
            "kl_only": True,
            "t_cutoffs": [str(T) for T in T_CUTOFFS],
            "clusters_fine": {c: sorted(members) for c, members in CLUSTERS_FINE.items()},
            "civilian_singletons": sorted(CIVILIAN_SINGLETONS),
            "pre_registered_outlier_pairs": [sorted(p) for p in PRE_REGISTERED_OUTLIER_PAIRS],
        },
    }

    for layer in LAYERS:
        log.info("=== Per-layer stats: layer %d ===", layer)
        layer_stats = compute_layer_statistics(
            layer=layer,
            cos_data=cos_data,
            js_matrices=js_matrices,
            per_prompt_js_full=js_results["per_prompt_js_full"],
            persona_names=persona_names,
            rng=rng_np,
            n_perm=args.n_perm,
        )
        log.info(
            "  layer %d: rho_raw=%.4f (p_mantel=%.5f) | "
            "rho_partial_cluster_joint=%.4f | rho_resid_mm=%.4f | "
            "per_prompt_median=%.4f | t8_ratio=%.4f",
            layer,
            layer_stats["rho_raw"],
            layer_stats["p_mantel_one_sided"],
            layer_stats["rho_partial_cluster_joint"],
            layer_stats["rho_resid_baseline_mean_marginal"],
            layer_stats["per_prompt_median"],
            layer_stats["t8_gate_ratio"]
            if layer_stats["t8_gate_ratio"] is not None
            else float("nan"),
        )
        results["layers"][str(layer)] = layer_stats

    # ---- Step 6: CKA (exploratory) at L10 with no_persona excluded ----
    if centroids_by_layer:
        idx_no = persona_names.index("no_persona")
        idx_19 = [i for i in range(20) if i != idx_no]
        cent_l10_19 = centroids_by_layer[HEADLINE_LAYER][idx_19]
        # log-prob matrix flattened at T=full.
        # per_prompt_js does not give us back log_probs; CKA over flattened log-probs
        # would require re-running teacher-forcing. Instead we use the (19, 19) JS
        # distance matrix as the "behavioral RDM" and compute CKA between
        # (19, 3584) centroids and the JS-RDM as RSA-style RDM-CKA per
        # Kornblith (this is a different statistic from §3 H_cka but is the
        # tractable analogue here; documented in metadata).
        js_19 = js_matrices["Tfull"][np.ix_(idx_19, idx_19)]
        # Build centered representational similarity matrices and compute CKA on those.
        # Treat centroids similarity as -1 * pairwise distance (negative cosine
        # distance => similarity).
        cent_sim = cent_l10_19 @ cent_l10_19.T  # (19, 19)
        # JS similarity = max - JS_distance.
        js_sim = float(js_19.max()) - js_19
        # Linear CKA between similarity matrices.
        try:
            cka_val = linear_cka(cent_sim, js_sim)
        except (ValueError, RuntimeError) as e:
            log.warning("CKA failed: %s", e)
            cka_val = float("nan")
        results["exploratory"]["cka_l10_n19_rdm_form"] = {
            "value": cka_val,
            "caveat": (
                "Exploratory only. At n=19 over the flattened similarity space "
                "CKA is uncalibrated for inference (Kornblith et al. 2019). "
                "Computed between centroid-Gram and JS-similarity matrices (RDM-style), "
                "NOT between centroids and flattened log-probs as the plan §3 H_cka "
                "originally framed. The log-prob analogue requires retained "
                "per-prompt log-prob tensors (currently dropped to save disk)."
            ),
        }
        log.info("CKA (RDM-form, L10, n=19): %.4f", cka_val)

    # ---- Persist headline results ----
    (OUT_DIR / "geometry_alignment.json").write_text(json.dumps(results, indent=2))
    log.info("Wrote headline results to %s", OUT_DIR / "geometry_alignment.json")

    write_run_metadata(OUT_DIR, cosine_sha, t_started)

    # Final headline log.
    headline = results["layers"][str(HEADLINE_LAYER)]
    log.info("=" * 70)
    log.info("HEADLINE (layer %d):", HEADLINE_LAYER)
    log.info("  rho_raw                          = %.4f", headline["rho_raw"])
    log.info("  p_mantel_one_sided               = %.5f", headline["p_mantel_one_sided"])
    log.info("  rho_partial_cluster_joint        = %.4f", headline["rho_partial_cluster_joint"])
    log.info(
        "  rho_resid_baseline_mean_marginal = %.4f",
        headline["rho_resid_baseline_mean_marginal"],
    )
    log.info("  per_prompt_median                = %.4f", headline["per_prompt_median"])
    log.info(
        "  t8_gate_ratio                    = %s",
        f"{headline['t8_gate_ratio']:.4f}" if headline["t8_gate_ratio"] is not None else "nan",
    )
    log.info("=" * 70)
    log.info("Total wall: %.1f min", (time.time() - t_started) / 60.0)


if __name__ == "__main__":
    main()
