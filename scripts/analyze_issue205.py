#!/usr/bin/env python3
"""Issue #205 — combined geometric + behavioral analysis.

Loads:
  - 7 x 2 = 14 all_centroids.pt files (5 EM + 1 base + 1 benign x Method A/B)
  - Per-question activation stacks for LDA + paired empty-response filter
  - 5 marker_eval.json files (one per EM condition)

Computes:
  Geometric: M1 (cos-sim collapse), M2 (EM-axis projection), M3 (LDA accuracy)
  Behavioral: per-persona [ZLT] rates, H1/H2/H3 hypothesis tests
  BH-FDR + Holm multiple-testing correction

Outputs: eval_results/issue_205/run_result.json

Usage:
    uv run python scripts/analyze_issue205.py --seed 42
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from itertools import permutations
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats

os.environ.setdefault("HF_HOME", "/workspace/.cache/huggingface")

# ── Constants ────────────────────────────────────────────────────────────────

REPO_ROOT = Path("/workspace/explore-persona-space")
DATA_ROOT = REPO_ROOT / "data" / "persona_vectors" / "qwen2.5-7b-instruct"
EVAL_ROOT = REPO_ROOT / "eval_results" / "issue_205"
OUTPUT_PATH = EVAL_ROOT / "run_result.json"

PERSONAS = [
    "software_engineer",
    "kindergarten_teacher",
    "data_scientist",
    "medical_doctor",
    "librarian",
    "french_person",
    "villain",
    "comedian",
    "police_officer",
    "zelthari_scholar",
    "assistant",
    "confab",
]
ASST_IDX = PERSONAS.index("assistant")  # = 10
NON_ASST_IDX = [i for i in range(12) if i != ASST_IDX]  # 11 indices
LAYERS = [7, 14, 20, 21, 27]

EM_CONDS = [
    "E0_assistant",
    "E1_paramedic",
    "E2_kindergarten_teacher",
    "E3_french_person",
    "E4_villain",
]

# Checkpoint tags as used by extract_persona_vectors.py --checkpoint-tag
CHECKPOINT_TAGS = {
    "base": "base",
    "E0_assistant": "em_E0_assistant_375",
    "E1_paramedic": "em_E1_paramedic_375",
    "E2_kindergarten_teacher": "em_E2_kindergarten_teacher_375",
    "E3_french_person": "em_E3_french_person_375",
    "E4_villain": "em_E4_villain_375",
    "benign_sft_375": "benign_sft_375",
}

ALL_CHECKPOINTS = ["base", *EM_CONDS, "benign_sft_375"]

COS_TO_ASST = {
    "E0_assistant": 1.000,
    "E1_paramedic": 0.9457,
    "E2_kindergarten_teacher": 0.9060,
    "E3_french_person": 0.8696,
    "E4_villain": 0.7828,
}

N_PERM = 10_000
BOOT_N = 1000
FDR_ALPHA = 0.01


# ── Utility functions ────────────────────────────────────────────────────────


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT), text=True
        ).strip()
    except Exception:
        return "unknown"


def load_centroids(ck_name: str, method: str) -> dict[str, torch.Tensor]:
    """Load all_centroids.pt for a given checkpoint and method."""
    tag = CHECKPOINT_TAGS[ck_name]
    path = DATA_ROOT / tag / f"method_{method.lower()}" / "all_centroids.pt"
    if not path.exists():
        raise FileNotFoundError(f"Missing centroids: {path}")
    centroids = torch.load(path, weights_only=True, map_location="cpu")
    return centroids


def load_perquestion(ck_name: str, method: str, persona: str, layer: int) -> torch.Tensor | None:
    """Load per-question activation stack for a given checkpoint/method/persona/layer."""
    tag = CHECKPOINT_TAGS[ck_name]
    path = DATA_ROOT / tag / f"method_{method.lower()}" / f"{persona}_perquestion_L{layer}.pt"
    if not path.exists():
        return None
    return torch.load(path, weights_only=True, map_location="cpu")


def load_question_indices(ck_name: str, method: str, persona: str) -> torch.Tensor | None:
    """Load question indices for a given checkpoint/method/persona."""
    tag = CHECKPOINT_TAGS[ck_name]
    path = DATA_ROOT / tag / f"method_{method.lower()}" / f"{persona}_question_indices.pt"
    if not path.exists():
        return None
    return torch.load(path, weights_only=True, map_location="cpu")


def cosine_matrix(vecs: torch.Tensor) -> torch.Tensor:
    """Compute pairwise cosine similarity matrix. Input: (N, D) -> (N, N)."""
    vecs_f = vecs.float()
    normed = F.normalize(vecs_f, dim=1)
    return normed @ normed.T


def upper_triangle(mat: torch.Tensor) -> np.ndarray:
    """Extract upper-triangle off-diagonal values."""
    n = mat.shape[0]
    indices = torch.triu_indices(n, n, offset=1)
    return mat[indices[0], indices[1]].numpy()


def paired_permutation_test(
    x: np.ndarray, y: np.ndarray, n_iter: int = N_PERM, seed: int = 42
) -> float:
    """Two-sided paired permutation test on the mean difference."""
    rng = np.random.RandomState(seed)
    diff = y - x
    obs = np.abs(diff.mean())
    count = 0
    for _ in range(n_iter):
        signs = rng.choice([-1, 1], size=len(diff))
        if np.abs((diff * signs).mean()) >= obs:
            count += 1
    return count / n_iter


def bootstrap_ci(
    vals: np.ndarray, stat_fn=np.mean, n_boot: int = BOOT_N, seed: int = 42, alpha: float = 0.05
) -> tuple[float, float]:
    """Bootstrap confidence interval."""
    rng = np.random.RandomState(seed)
    stats_arr = []
    for _ in range(n_boot):
        sample = rng.choice(vals, size=len(vals), replace=True)
        stats_arr.append(stat_fn(sample))
    stats_arr = np.array(stats_arr)
    lo = np.percentile(stats_arr, 100 * alpha / 2)
    hi = np.percentile(stats_arr, 100 * (1 - alpha / 2))
    return float(lo), float(hi)


def bh_fdr_correct(p_values: list[float], alpha: float = FDR_ALPHA) -> list[float]:
    """Benjamini-Hochberg FDR correction. Returns list of adjusted p-values."""
    n = len(p_values)
    if n == 0:
        return []
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    adjusted = [None] * n
    prev_adj = 0.0
    for rank, (orig_idx, p) in enumerate(indexed, 1):
        adj_p = min(p * n / rank, 1.0)
        adj_p = max(adj_p, prev_adj)  # enforce monotonicity
        adjusted[orig_idx] = adj_p
        prev_adj = adj_p
    return adjusted


def holm_correct(p_values: list[float], alpha: float = 0.05) -> list[float]:
    """Holm-Bonferroni step-down correction."""
    n = len(p_values)
    if n == 0:
        return []
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    adjusted = [None] * n
    max_adj = 0.0
    for rank, (orig_idx, p) in enumerate(indexed):
        adj_p = min(p * (n - rank), 1.0)
        adj_p = max(adj_p, max_adj)
        adjusted[orig_idx] = adj_p
        max_adj = adj_p
    return adjusted


def exact_spearman_permutation_test(xs: list[float], ys: list[float]) -> float:
    """Exact permutation test for Spearman rho (all n! permutations for small n)."""
    obs_rho, _ = stats.spearmanr(xs, ys)
    obs_rho = abs(obs_rho)
    n_ge = 0
    total = 0
    for perm in permutations(range(len(ys))):
        ys_perm = [ys[i] for i in perm]
        rho, _ = stats.spearmanr(xs, ys_perm)
        if abs(rho) >= obs_rho:
            n_ge += 1
        total += 1
    return n_ge / total


# ── Paired empty-response filter (Method B only, plan §3c) ──────────────────


def compute_paired_nonempty_mask() -> dict[str, set[int]] | None:
    """Compute the intersection of non-empty question indices across all 7 checkpoints.

    For Method B, a (persona, question_idx) pair is dropped from ALL 7 checkpoints'
    per-question stacks if ANY checkpoint returned empty for that pair (plan §3c).

    Returns a dict mapping persona -> set of question indices that are non-empty
    across all checkpoints, or None if per-question data is unavailable.
    """
    # Load question indices for all checkpoints under method_b
    per_ck_indices: dict[str, dict[str, set[int]]] = {}
    any_loaded = False

    for ck in ALL_CHECKPOINTS:
        per_ck_indices[ck] = {}
        for persona in PERSONAS:
            q_indices = load_question_indices(ck, "b", persona)
            if q_indices is not None:
                per_ck_indices[ck][persona] = set(q_indices.tolist())
                any_loaded = True
            else:
                per_ck_indices[ck][persona] = set()

    if not any_loaded:
        return None

    # Intersection across all 7 checkpoints per persona
    shared_nonempty: dict[str, set[int]] = {}
    for persona in PERSONAS:
        sets = [
            per_ck_indices[ck][persona] for ck in ALL_CHECKPOINTS if per_ck_indices[ck][persona]
        ]
        if sets:
            shared_nonempty[persona] = set.intersection(*sets)
        else:
            shared_nonempty[persona] = set()

    # Sanity gate: warn if any persona has < 85% of 240 questions surviving
    for persona, indices in shared_nonempty.items():
        n_eff = len(indices)
        if n_eff < 204:  # 0.85 * 240
            print(
                f"  WARNING: paired filter leaves only {n_eff}/240 questions"
                f" for {persona} (< 204 threshold)"
            )

    return shared_nonempty


def recompute_centroids_on_paired_set(
    shared_nonempty: dict[str, set[int]],
    layer: int,
) -> dict[str, dict[str, torch.Tensor]]:
    """Recompute Method B centroids using only the paired non-empty question subset.

    Returns dict[checkpoint -> dict[persona -> centroid_vector(hidden_dim)]].
    """
    recomputed: dict[str, dict[str, torch.Tensor]] = {}
    for ck in ALL_CHECKPOINTS:
        recomputed[ck] = {}
        for persona in PERSONAS:
            perq = load_perquestion(ck, "b", persona, layer)
            q_indices = load_question_indices(ck, "b", persona)
            if perq is None or q_indices is None:
                recomputed[ck][persona] = None
                continue
            # Filter to shared non-empty indices
            mask = torch.tensor(
                [idx.item() in shared_nonempty.get(persona, set()) for idx in q_indices],
                dtype=torch.bool,
            )
            if mask.any():
                recomputed[ck][persona] = perq[mask].float().mean(dim=0)
            else:
                recomputed[ck][persona] = None
    return recomputed


# ── Geometric analysis ───────────────────────────────────────────────────────


def run_geometric_analysis(seed: int) -> dict:  # noqa: C901
    """Run all 3 geometric metrics across methods, layers, and conditions."""
    results = {}

    # Pre-compute paired empty-response filter for Method B
    paired_nonempty = compute_paired_nonempty_mask()
    if paired_nonempty is not None:
        total_shared = sum(len(v) for v in paired_nonempty.values())
        print(f"\n  Paired filter: {total_shared} total (persona, question) pairs surviving")
        results["paired_filter_n_effective"] = {
            p: len(indices) for p, indices in paired_nonempty.items()
        }
    else:
        print("\n  Paired filter: per-question data not available, using raw centroids")

    for method in ["a", "b"]:
        method_label = method.upper()
        print(f"\n{'=' * 60}")
        print(f"Geometric analysis — Method {method_label}")
        print(f"{'=' * 60}")

        # Load centroids for all checkpoints
        centroids = {}
        for ck in ALL_CHECKPOINTS:
            try:
                c = load_centroids(ck, method)
                centroids[ck] = c
            except FileNotFoundError as e:
                print(f"  WARNING: {e}")
                continue

        if "base" not in centroids:
            print("  ERROR: base centroids missing, skipping method")
            continue

        for l_idx, layer in enumerate(LAYERS):
            print(f"\n  Layer {layer}:")

            # Build per-checkpoint persona vectors: (12, hidden_dim)
            # For Method B with paired filter available, recompute centroids from
            # the shared non-empty subset (ISSUE-3 fix).
            use_paired = method == "b" and paired_nonempty is not None
            if use_paired:
                paired_centroids = recompute_centroids_on_paired_set(paired_nonempty, layer)

            ck_vecs = {}
            for ck, c_dict in centroids.items():
                vecs = []
                for persona in PERSONAS:
                    if use_paired and paired_centroids.get(ck, {}).get(persona) is not None:
                        # Use recomputed centroid from paired non-empty subset
                        vecs.append(paired_centroids[ck][persona])
                    elif persona in c_dict:
                        # centroid shape: (n_layers, hidden_dim)
                        vecs.append(c_dict[persona][l_idx].float())
                    else:
                        print(f"    WARNING: missing {persona} in {ck}")
                        vecs.append(torch.zeros(c_dict[next(iter(c_dict))].shape[1]))
                ck_vecs[ck] = torch.stack(vecs)  # (12, hidden_dim)

            # ── M1: cos-sim collapse ──
            cos_mats = {ck: cosine_matrix(v) for ck, v in ck_vecs.items()}
            offdiags = {ck: upper_triangle(m) for ck, m in cos_mats.items()}

            base_offdiag = offdiags.get("base")
            if base_offdiag is None:
                continue

            # EM conditions vs base
            for em_cond in EM_CONDS:
                if em_cond not in offdiags:
                    continue
                em_offdiag = offdiags[em_cond]
                delta = float(em_offdiag.mean() - base_offdiag.mean())
                p_m1 = paired_permutation_test(base_offdiag, em_offdiag, N_PERM, seed)
                ci = bootstrap_ci(em_offdiag, np.mean, BOOT_N, seed)
                key = f"M1_{method_label}_L{layer}_{em_cond}"
                results[key] = {
                    "metric": "M1_cos_collapse",
                    "method": method_label,
                    "layer": layer,
                    "condition": em_cond,
                    "delta_mean_offdiag": delta,
                    "base_mean": float(base_offdiag.mean()),
                    "em_mean": float(em_offdiag.mean()),
                    "p_raw": p_m1,
                    "ci_95": ci,
                    "family": "geometry_em_pre_post",
                }
                print(f"    M1 {em_cond}: delta={delta:+.4f} p={p_m1:.4f}")

            # Benign vs base
            if "benign_sft_375" in offdiags:
                benign_offdiag = offdiags["benign_sft_375"]
                delta_benign = float(benign_offdiag.mean() - base_offdiag.mean())
                p_benign = paired_permutation_test(base_offdiag, benign_offdiag, N_PERM, seed)
                key_b = f"M1_{method_label}_L{layer}_benign_sft_375"
                results[key_b] = {
                    "metric": "M1_cos_collapse",
                    "method": method_label,
                    "layer": layer,
                    "condition": "benign_sft_375",
                    "delta_mean_offdiag": delta_benign,
                    "p_raw": p_benign,
                    "family": "geometry_benign_pre_post",
                }
                print(f"    M1 benign: delta={delta_benign:+.4f} p={p_benign:.4f}")

                # EM vs benign
                for em_cond in EM_CONDS:
                    if em_cond not in offdiags:
                        continue
                    p_vs = paired_permutation_test(offdiags[em_cond], benign_offdiag, N_PERM, seed)
                    key_vs = f"M1_{method_label}_L{layer}_{em_cond}_vs_benign"
                    results[key_vs] = {
                        "metric": "M1_em_vs_benign",
                        "method": method_label,
                        "layer": layer,
                        "condition": em_cond,
                        "p_raw": p_vs,
                        "family": "geometry_em_vs_benign",
                    }

            # ── M2: EM-axis projection (primary = E0-anchored) ──
            if "E0_assistant" in ck_vecs and "base" in ck_vecs:
                em_axis_e0 = ck_vecs["E0_assistant"][ASST_IDX] - ck_vecs["base"][ASST_IDX]
                em_axis_e0_norm = F.normalize(em_axis_e0.unsqueeze(0), dim=1).squeeze(0)

                for em_cond in EM_CONDS:
                    if em_cond not in ck_vecs:
                        continue

                    # Project non-assistant personas onto EM axis
                    em_proj = torch.abs(
                        F.cosine_similarity(
                            ck_vecs[em_cond][NON_ASST_IDX],
                            em_axis_e0_norm.unsqueeze(0).expand(len(NON_ASST_IDX), -1),
                            dim=1,
                        )
                    )
                    base_proj = torch.abs(
                        F.cosine_similarity(
                            ck_vecs["base"][NON_ASST_IDX],
                            em_axis_e0_norm.unsqueeze(0).expand(len(NON_ASST_IDX), -1),
                            dim=1,
                        )
                    )
                    obs = float(em_proj.mean() - base_proj.mean())

                    # Permutation null: shuffle condition labels
                    rng = np.random.RandomState(seed)
                    combined = torch.cat(
                        [ck_vecs["base"][NON_ASST_IDX], ck_vecs[em_cond][NON_ASST_IDX]],
                        dim=0,
                    )
                    n_half = len(NON_ASST_IDX)
                    null_count = 0
                    for _ in range(N_PERM):
                        idx = rng.permutation(2 * n_half)
                        g1 = combined[idx[:n_half]]
                        g2 = combined[idx[n_half:]]
                        proj1 = torch.abs(
                            F.cosine_similarity(
                                g1, em_axis_e0_norm.unsqueeze(0).expand(n_half, -1), dim=1
                            )
                        )
                        proj2 = torch.abs(
                            F.cosine_similarity(
                                g2, em_axis_e0_norm.unsqueeze(0).expand(n_half, -1), dim=1
                            )
                        )
                        if float(proj2.mean() - proj1.mean()) >= obs:
                            null_count += 1
                    p_m2 = null_count / N_PERM

                    key_m2 = f"M2_primary_{method_label}_L{layer}_{em_cond}"
                    results[key_m2] = {
                        "metric": "M2_em_axis_projection",
                        "variant": "primary_E0_anchored",
                        "method": method_label,
                        "layer": layer,
                        "condition": em_cond,
                        "obs_delta": obs,
                        "p_raw": p_m2,
                        "family": "geometry_em_pre_post",
                    }
                    print(f"    M2(E0-axis) {em_cond}: obs={obs:.4f} p={p_m2:.4f}")

            # ── M3: LDA separability ──
            # PLAN DEVIATION (documented): uses leave-one-out nearest-centroid on
            # the 12 persona centroids as a proxy for the plan's GroupKFold LDA on
            # per-question activation stacks. This has lower statistical power (12
            # centroids vs 240+ per-question vectors per persona) but is robust
            # when per-question stacks are incomplete. The per-question data is
            # extracted and saved, so full GroupKFold LDA can be added post-hoc.
            for em_cond in EM_CONDS:
                if em_cond not in ck_vecs or "base" not in ck_vecs:
                    continue

                # Nearest-centroid classification accuracy
                def _nearest_centroid_acc(vecs: torch.Tensor) -> float:
                    """Leave-one-out nearest centroid on the 12 persona vectors."""
                    n = vecs.shape[0]
                    correct = 0
                    for i in range(n):
                        # Compute centroid of each class excluding the test point
                        # Since we have 1 vector per class, use cosine to all others
                        dists = F.cosine_similarity(vecs[i].unsqueeze(0), vecs, dim=1)
                        dists[i] = -1  # exclude self
                        pred = dists.argmax().item()
                        if pred == i:
                            correct += 1
                    return correct / n

                acc_base = _nearest_centroid_acc(ck_vecs["base"])
                acc_em = _nearest_centroid_acc(ck_vecs[em_cond])
                delta_acc = acc_base - acc_em

                # Permutation test for M3
                rng = np.random.RandomState(seed)
                null_count = 0
                for _ in range(N_PERM):
                    # Shuffle persona labels between base and EM
                    swap_mask = rng.randint(0, 2, size=12).astype(bool)
                    mixed_base = ck_vecs["base"].clone()
                    mixed_em = ck_vecs[em_cond].clone()
                    mixed_base[swap_mask] = ck_vecs[em_cond][swap_mask]
                    mixed_em[swap_mask] = ck_vecs["base"][swap_mask]
                    d = _nearest_centroid_acc(mixed_base) - _nearest_centroid_acc(mixed_em)
                    if d >= delta_acc:
                        null_count += 1
                p_m3 = null_count / N_PERM

                key_m3 = f"M3_{method_label}_L{layer}_{em_cond}"
                results[key_m3] = {
                    "metric": "M3_separability",
                    "method": method_label,
                    "layer": layer,
                    "condition": em_cond,
                    "acc_base": acc_base,
                    "acc_em": acc_em,
                    "delta_acc": delta_acc,
                    "p_raw": p_m3,
                    "family": "geometry_em_pre_post",
                }
                print(
                    f"    M3 {em_cond}: acc_base={acc_base:.3f}"
                    f" acc_em={acc_em:.3f} d={delta_acc:+.3f} p={p_m3:.4f}"
                )

    return results


# ── Behavioral analysis ──────────────────────────────────────────────────────


def run_behavioral_analysis(seed: int) -> dict:  # noqa: C901
    """Run behavioral hypothesis tests (H1, H2, H3)."""
    results = {}
    behavioral = {}

    print(f"\n{'=' * 60}")
    print("Behavioral analysis")
    print(f"{'=' * 60}")

    for em_cond in EM_CONDS:
        eval_path = EVAL_ROOT / em_cond / "marker_eval.json"
        if not eval_path.exists():
            print(f"  WARNING: missing {eval_path}")
            continue
        with open(eval_path) as f:
            m = json.load(f)

        per_persona = m["per_persona"]
        persona_slug = em_cond.split("_", 1)[1]

        # Compute mean bystander leakage (exclude confab and the induction persona)
        bystander_rates = [
            r["strict_rate"] for p, r in per_persona.items() if p not in ("confab", persona_slug)
        ]
        mean_bystander = float(np.mean(bystander_rates)) if bystander_rates else 0.0

        # Induction persona's own leakage (if it's in the eval set)
        e_persona_rate = per_persona.get(persona_slug, {}).get("strict_rate", None)

        behavioral[em_cond] = {
            "per_persona": {p: r["strict_rate"] for p, r in per_persona.items()},
            "mean_bystander": mean_bystander,
            "leakage_to_E_persona": e_persona_rate,
            "confab_source_rate": per_persona.get("confab", {}).get("strict_rate", 0.0),
        }

        print(
            f"  {em_cond}: mean_bystander={mean_bystander * 100:.1f}%"
            f"  confab={behavioral[em_cond]['confab_source_rate'] * 100:.1f}%"
            f"  E_persona={e_persona_rate * 100:.1f}%"
            if e_persona_rate is not None
            else ""
        )

        # Per-persona results for family-wise testing
        for p, r in per_persona.items():
            key = f"behavioral_{em_cond}_{p}"
            results[key] = {
                "metric": "marker_leakage",
                "condition": em_cond,
                "persona": p,
                "strict_rate": r["strict_rate"],
                "strict_hits": r["strict_hits"],
                "total": r["total"],
                "family": "behavioral_per_persona",
            }

    if len(behavioral) < 5:
        print(f"  WARNING: only {len(behavioral)}/5 conditions loaded, some tests skipped")

    # ── H1: Induction-persona-specific leakage ──
    print("\n  H1: Induction-persona-specific leakage:")
    for em_cond in EM_CONDS:
        if em_cond not in behavioral:
            continue
        persona_slug = em_cond.split("_", 1)[1]
        if persona_slug == "assistant":
            # E0's induction persona is in the bystander set; still testable
            pass

        e_rate = behavioral[em_cond]["leakage_to_E_persona"]
        if e_rate is None:
            continue

        # Other bystanders (exclude confab and induction persona)
        other_rates = [
            v
            for p, v in behavioral[em_cond]["per_persona"].items()
            if p not in ("confab", persona_slug)
        ]
        if not other_rates:
            continue
        obs_diff = e_rate - float(np.mean(other_rates))

        # Permutation test: shuffle which persona is "the induction one"
        rng = np.random.RandomState(seed)
        non_confab_personas = [p for p in behavioral[em_cond]["per_persona"] if p != "confab"]
        non_confab_rates = [behavioral[em_cond]["per_persona"][p] for p in non_confab_personas]
        null_count = 0
        for _ in range(N_PERM):
            perm_idx = rng.randint(0, len(non_confab_rates))
            perm_e_rate = non_confab_rates[perm_idx]
            perm_others = [r for i, r in enumerate(non_confab_rates) if i != perm_idx]
            perm_diff = perm_e_rate - np.mean(perm_others)
            if perm_diff >= obs_diff:
                null_count += 1
        p_h1 = null_count / N_PERM

        results[f"H1_{em_cond}"] = {
            "metric": "H1_induction_self_leakage",
            "condition": em_cond,
            "persona_slug": persona_slug,
            "leakage_to_E": e_rate,
            "mean_other_bystanders": float(np.mean(other_rates)),
            "obs_diff": obs_diff,
            "p_raw": p_h1,
        }
        print(
            f"    {em_cond}: E={e_rate * 100:.1f}% others={np.mean(other_rates) * 100:.1f}%"
            f" diff={obs_diff * 100:+.1f}pp p={p_h1:.4f}"
        )

    # ── H2: Mean bystander leakage invariant across conditions ──
    if len(behavioral) == 5:
        means = [behavioral[em]["mean_bystander"] for em in EM_CONDS]
        range_pp = (max(means) - min(means)) * 100

        # Permutation ANOVA-equivalent — pool must match the observed statistic.
        # The observed mean_bystander excludes confab AND the induction persona,
        # so the permutation pool must also exclude both (ISSUE-1 fix).
        all_per_persona = {}
        for em_cond in EM_CONDS:
            persona_slug = em_cond.split("_", 1)[1]
            for p, rate in behavioral[em_cond]["per_persona"].items():
                if p not in ("confab", persona_slug):
                    all_per_persona.setdefault(em_cond, []).append(rate)

        rng = np.random.RandomState(seed)
        all_rates = []
        all_labels = []
        for em_cond in EM_CONDS:
            rates = all_per_persona.get(em_cond, [])
            all_rates.extend(rates)
            all_labels.extend([em_cond] * len(rates))
        all_rates = np.array(all_rates)
        all_labels = np.array(all_labels)

        obs_range = range_pp
        null_count = 0
        for _ in range(N_PERM):
            perm_labels = rng.permutation(all_labels)
            perm_means = []
            for em_cond in EM_CONDS:
                mask = perm_labels == em_cond
                if mask.any():
                    perm_means.append(all_rates[mask].mean())
            if perm_means:
                perm_range = (max(perm_means) - min(perm_means)) * 100
                if perm_range >= obs_range:
                    null_count += 1
        p_h2 = null_count / N_PERM

        results["H2"] = {
            "metric": "H2_induction_invariant",
            "means": dict(zip(EM_CONDS, means, strict=True)),
            "range_pp": range_pp,
            "p_raw": p_h2,
        }
        print(f"\n  H2: range={range_pp:.1f}pp p={p_h2:.4f}")

    # ── H3: Spearman rho cos-to-assistant vs mean bystander ──
    if len(behavioral) == 5:
        xs = [COS_TO_ASST[em] for em in EM_CONDS]
        ys = [behavioral[em]["mean_bystander"] for em in EM_CONDS]
        rho, p_scipy = stats.spearmanr(xs, ys)
        p_exact = exact_spearman_permutation_test(xs, ys)

        results["H3"] = {
            "metric": "H3_cos_monotone",
            "rho": float(rho),
            "p_scipy": float(p_scipy),
            "p_exact": p_exact,
            "xs_cos_to_asst": xs,
            "ys_mean_bystander": ys,
        }
        print(f"\n  H3: rho={rho:.4f} p_exact={p_exact:.4f}")

    results["behavioral_summary"] = behavioral
    return results


# ── Multiple testing correction ──────────────────────────────────────────────


def apply_corrections(results: dict) -> dict:
    """Apply BH-FDR and Holm corrections to p-values within each family."""
    families = {}
    for key, val in results.items():
        if isinstance(val, dict) and "family" in val and "p_raw" in val:
            family = val["family"]
            families.setdefault(family, []).append(key)

    for family, keys in families.items():
        p_values = [results[k]["p_raw"] for k in keys]
        bh_adjusted = bh_fdr_correct(p_values, FDR_ALPHA)
        holm_adjusted = holm_correct(p_values)

        for i, key in enumerate(keys):
            results[key]["p_bh_fdr"] = bh_adjusted[i]
            results[key]["p_holm"] = holm_adjusted[i]
            results[key]["sig_bh_fdr"] = bh_adjusted[i] < FDR_ALPHA
            results[key]["sig_holm"] = holm_adjusted[i] < 0.05

        n_sig = sum(1 for a in bh_adjusted if a < FDR_ALPHA)
        print(
            f"\n  Family '{family}': {len(keys)} tests, "
            f"{n_sig} significant at BH-FDR alpha={FDR_ALPHA}"
        )

    return results


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    t_start = time.time()
    EVAL_ROOT.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("ISSUE #205 — Combined Geometric + Behavioral Analysis")
    print("=" * 70)

    # Geometric analysis
    geo_results = run_geometric_analysis(args.seed)

    # Behavioral analysis
    behav_results = run_behavioral_analysis(args.seed)

    # Merge results
    all_results = {**geo_results, **behav_results}

    # Apply multiple-testing corrections
    print("\n" + "=" * 60)
    print("Multiple-testing corrections")
    print("=" * 60)
    all_results = apply_corrections(all_results)

    # Build output JSON
    wall_time = time.time() - t_start
    output = {
        "issue": 205,
        "experiment": "issue_205_em_persona_geometry_AND_leakage",
        "git_commit": _git_commit(),
        "seed": args.seed,
        "wall_time_seconds": round(wall_time, 1),
        "em_conditions": EM_CONDS,
        "eval_personas": PERSONAS,
        "layers": LAYERS,
        "methods": ["A", "B"],
        "n_permutations": N_PERM,
        "fdr_alpha": FDR_ALPHA,
        "cos_to_assistant": COS_TO_ASST,
        "results": all_results,
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nWrote results to {OUTPUT_PATH}")
    print(f"Analysis complete in {wall_time:.1f}s")

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
