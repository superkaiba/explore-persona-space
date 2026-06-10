#!/usr/bin/env python3
"""#551 VM-side controls over the re-extracted #521 shift tensors (CPU, off-pod).

Downloads the 18 per-cell shift tensors from the HF data repo (the
persisted artifact this corrective task exists to produce), then runs:

- **Phase R** — reproduction gate vs the parent #521 per-cell SVD JSONs
  (4 BINDING clauses per cell; descriptive per-persona deltas). Any
  binding breach halts BEFORE the controls (exit 3): the controls would
  not be about the parent's tensors.
- **Control A** — source-dropped (leave-one-out) spectrum: top-share of
  the 13-column matrix without the ``medical_doctor`` column vs
  calibrated nulls (sign-flip BINDING, row-shuffle descriptive), plus an
  exploratory full 14-fold jackknife.
- **Control B** — whole-response EM read: SVD + nulls on the
  mean-over-response matrices for BOTH arms' same-variant cells
  (symmetric EM-mr vs marker-mr PRIMARY; EM-mr vs marker end-slot kept
  as the parent-reference SECONDARY).
- **Control C** — norm-vs-alignment: Spearman rho(per-persona ||shift||,
  cos-to-top-direction) with a ONE-SIDED positive Monte Carlo
  permutation p (10,000 draws), plus the split-half reliability
  supplementary from the persisted per-question tensors.

Outputs JSONs under ``eval_results/issue_551/`` (plan #551 §10 layout)
and figures under ``figures/issue_551/`` (paper_plots rcParams). Each
phase's JSON is written the moment the phase completes (checkpoint per
phase).

Run (VM, CPU)::

    uv run python scripts/issue551_controls.py \\
        --tensors-repo superkaiba1/explore-persona-space-data \\
        --tensors-prefix issue551_shift_reextract/analysis_tensors/shifts \\
        --parent-svd-dir eval_results/issue_521/svd \\
        --out eval_results/issue_551

Smoke (local fixtures, no HF)::

    uv run python scripts/issue551_controls.py \\
        --local-shifts-dir /tmp/i551_fixtures/shifts \\
        --parent-svd-dir /tmp/i551_fixtures/parent_svd \\
        --out /tmp/i551_fixtures/out --figures-dir /tmp/i551_fixtures/figs
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from dotenv import load_dotenv

from explore_persona_space.analysis.svd_direction_constancy import (
    assemble_M,
    cosine,
    row_shuffle_null,
    sign_flip_null,
    spearman_rho,
    svd_summary,
)

logger = logging.getLogger(__name__)

VARIANTS = ("same", "base", "on_policy")
ARMS = ("marker", "em")
SEEDS = (42, 137, 256)
SOURCE_PERSONA = "medical_doctor"

# Pre-registered thresholds (plan #551 §6; do NOT change without a plan
# amendment — "Must amend the plan first" list).
REPRO_TOL = 0.05  # |Δ s_top1_frac| and |Δ mean_cos_to_U1|
U1_COS_MIN = 0.95  # |cos(U1_re, U1_parent)|
PROFILE_SPEARMAN_MIN = 0.8  # per-persona cos_to_U1 profile agreement
RHO_STRONG = 0.54  # registered "strong" descriptor (n=14, two-sided alpha=.05)
N_NULL_REPS = 1000
N_PERM = 10_000


@dataclass(frozen=True)
class CellKey:
    """One (variant, arm, seed) analysis cell."""

    variant: str
    arm: str
    seed: int

    @property
    def name(self) -> str:
        return f"{self.variant}_{self.arm}_seed{self.seed}"


def _all_cells() -> list[CellKey]:
    return [CellKey(v, a, s) for v in VARIANTS for a in ARMS for s in SEEDS]


def _same_cells() -> list[CellKey]:
    return [CellKey("same", a, s) for a in ARMS for s in SEEDS]


def _git_commit() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def _run_meta(args: argparse.Namespace) -> dict:
    """Reproducibility metadata embedded in every output JSON."""
    import importlib.metadata

    return {
        "issue": 551,
        "git_commit": _git_commit(),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "env_versions": {
            pkg: importlib.metadata.version(pkg) for pkg in ("torch", "numpy", "transformers")
        },
        "tensors_source": (
            str(args.local_shifts_dir)
            if args.local_shifts_dir
            else f"hf://{args.tensors_repo}/{args.tensors_prefix}"
        ),
        "parent_svd_dir": str(args.parent_svd_dir),
        "thresholds": {
            "repro_tol": REPRO_TOL,
            "u1_cos_min": U1_COS_MIN,
            "profile_spearman_min": PROFILE_SPEARMAN_MIN,
            "rho_strong": RHO_STRONG,
            "n_null_reps": N_NULL_REPS,
            "n_perm": N_PERM,
        },
        "source_persona": SOURCE_PERSONA,
    }


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2)
    logger.info("[wrote] %s", path)


def _download_tensors(repo: str, prefix: str, dest: Path) -> Path:
    """Per-file `hf_hub_download` of the 18 .pt + 18 manifest files.

    Deliberately reads the PERSISTED artifact (the point of this
    corrective run), never a pod-local copy.
    """
    import shutil

    from huggingface_hub import hf_hub_download

    dest.mkdir(parents=True, exist_ok=True)
    for cell in _all_cells():
        for suffix in (".pt", ".manifest.json"):
            fname = f"{cell.name}{suffix}"
            target = dest / fname
            if target.exists() and target.stat().st_size > 0:
                logger.info("[skip] %s already downloaded", fname)
                continue
            local = hf_hub_download(
                repo_id=repo,
                filename=f"{prefix}/{fname}",
                repo_type="dataset",
            )
            shutil.copy2(local, target)
            logger.info("[downloaded] %s (%.2f MB)", fname, target.stat().st_size / 1e6)
    return dest


def _load_cell(shifts_dir: Path, cell: CellKey) -> dict[str, dict[str, torch.Tensor]]:
    path = shifts_dir / f"{cell.name}.pt"
    if not path.exists():
        raise FileNotFoundError(f"missing shift tensor {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return payload["shifts"]


def _load_parent(parent_dir: Path, cell: CellKey) -> dict:
    path = parent_dir / f"{cell.name}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"missing parent reference JSON {path} — the reproduction gate has "
            f"nothing to anchor against; refusing to continue."
        )
    with path.open() as f:
        return json.load(f)


# ──────────────────────────────────────────────────────────────────────
# Phase R — reproduction gate
# ──────────────────────────────────────────────────────────────────────


def _null_summary_entry(null: dict) -> dict:
    return {"p95": float(null["p95"]), "p99": float(null["p99"]), "n_reps": int(null["n_reps"])}


def phase_r(
    *,
    cells: list[CellKey],
    shifts_dir: Path,
    parent_dir: Path,
    base_cosines: dict[str, float] | None,
    out_path: Path,
    meta: dict,
) -> tuple[dict, dict[str, dict]]:
    """Reproduction gate over all cells. Returns (payload, slot_cache).

    ``slot_cache[cell.name]`` carries the re-extracted slot-read M, its
    SVD summary, and the parent JSON — reused by controls A/B/C so the
    SVD runs once per cell.
    """
    per_cell: dict[str, dict] = {}
    slot_cache: dict[str, dict] = {}
    n_breach = 0
    for cell in cells:
        parent = _load_parent(parent_dir, cell)
        shifts = _load_cell(shifts_dir, cell)
        persona_order = list(parent["persona_order"])
        M, order = assemble_M(shifts, persona_order=persona_order)
        assert order == persona_order, (order, persona_order)
        svd = svd_summary(M)
        mean_cos_re = float(np.mean(svd["cos_to_U1"]))

        u1_parent = np.asarray(parent["U1"], dtype=np.float64)
        if u1_parent.shape[0] != M.shape[0]:
            raise ValueError(
                f"{cell.name}: hidden-dim mismatch — re-extracted M has H={M.shape[0]} "
                f"but the parent U1 has H={u1_parent.shape[0]}. The tensors and the "
                f"parent JSONs are from different models/rigs; refusing to gate."
            )
        u1_cos_signed = cosine(svd["U1"], u1_parent)
        u1_cos = abs(u1_cos_signed)
        # Orientation-invariant profile comparison: flip the re-extracted
        # profile when the top directions are anti-aligned (matches the
        # |cos| clause).
        sgn = -1.0 if u1_cos_signed < 0 else 1.0
        cos_re = sgn * np.asarray(svd["cos_to_U1"], dtype=np.float64)
        cos_parent = np.asarray(parent["cos_to_U1"], dtype=np.float64)
        profile_rho = spearman_rho(cos_re, cos_parent)

        d_top1 = abs(svd["s_top1_frac"] - float(parent["s_top1_frac"]))
        d_mean_cos = abs(mean_cos_re - float(parent["mean_cos_to_U1"]))
        binding_pass = (
            d_top1 <= REPRO_TOL
            and d_mean_cos <= REPRO_TOL
            and u1_cos >= U1_COS_MIN
            and profile_rho >= PROFILE_SPEARMAN_MIN
        )
        if not binding_pass:
            n_breach += 1

        # DESCRIPTIVE (reported, not gating).
        per_persona_d_cos = {
            p: float(abs(cos_re[i] - cos_parent[i])) for i, p in enumerate(persona_order)
        }
        s_re = np.asarray(svd["s"], dtype=np.float64)
        s_parent = np.asarray(parent["singular_values"], dtype=np.float64)
        top3_share_re = (s_re[:3] / s_re.sum()).tolist()
        top3_share_parent = (s_parent[:3] / s_parent.sum()).tolist()
        norm_cos_delta = None
        if base_cosines is not None and parent.get("shift_norm_vs_cosine"):
            norms = np.linalg.norm(M, axis=0)
            ordered_base_cos = [base_cosines[p] for p in persona_order]
            rho_re = spearman_rho(norms, ordered_base_cos)
            rho_parent = parent["shift_norm_vs_cosine"].get("spearman_rho")
            if rho_parent is not None:
                norm_cos_delta = {
                    "spearman_rho_re": float(rho_re),
                    "spearman_rho_parent": float(rho_parent),
                    "d_spearman_rho": float(abs(rho_re - float(rho_parent))),
                }

        per_cell[cell.name] = {
            "variant": cell.variant,
            "arm": cell.arm,
            "seed": cell.seed,
            "binding_pass": binding_pass,
            "clauses": {
                "d_s_top1_frac": float(d_top1),
                "d_mean_cos_to_U1": float(d_mean_cos),
                "abs_cos_U1_re_parent": float(u1_cos),
                "profile_spearman": float(profile_rho),
            },
            "re": {
                "s_top1_frac": float(svd["s_top1_frac"]),
                "mean_cos_to_U1": mean_cos_re,
                "cos_to_U1": [float(c) for c in cos_re],
            },
            "parent": {
                "s_top1_frac": float(parent["s_top1_frac"]),
                "mean_cos_to_U1": float(parent["mean_cos_to_U1"]),
            },
            "descriptive": {
                "per_persona_d_cos_to_U1": per_persona_d_cos,
                "top3_singular_share_re": top3_share_re,
                "top3_singular_share_parent": top3_share_parent,
                "shift_norm_vs_cosine_delta": norm_cos_delta,
            },
        }
        slot_cache[cell.name] = {
            "M": M,
            "svd": svd,
            "parent": parent,
            "persona_order": persona_order,
            "shifts": shifts,
        }
        logger.info(
            "[phase_r %s] binding_pass=%s d_top1=%.4f d_mean_cos=%.4f |cosU1|=%.4f "
            "profile_rho=%.3f",
            cell.name,
            binding_pass,
            d_top1,
            d_mean_cos,
            u1_cos,
            profile_rho,
        )

    payload = {
        "meta": meta,
        "binding_pass_all": n_breach == 0,
        "n_cells": len(cells),
        "n_breach": n_breach,
        "per_cell": per_cell,
    }
    _write_json(out_path, payload)
    return payload, slot_cache


# ──────────────────────────────────────────────────────────────────────
# Control A — source-dropped (LOO) spectrum + jackknife
# ──────────────────────────────────────────────────────────────────────


def control_a_loo(
    *, cells: list[CellKey], slot_cache: dict[str, dict], out_dir: Path, meta: dict
) -> dict:
    """LOO top-share vs nulls (sign-flip BINDING on same-variant cells)."""
    loo_per_cell: dict[str, dict] = {}
    jack_per_cell: dict[str, dict] = {}
    for cell in cells:
        c = slot_cache[cell.name]
        M, persona_order = c["M"], c["persona_order"]
        src_idx = persona_order.index(SOURCE_PERSONA)
        m_loo = np.delete(M, src_idx, axis=1)
        assert m_loo.shape[1] == M.shape[1] - 1, (m_loo.shape, M.shape)
        svd_loo = svd_summary(m_loo)
        row_null = row_shuffle_null(m_loo, n_reps=N_NULL_REPS, seed=cell.seed)
        sign_null = sign_flip_null(m_loo, n_reps=N_NULL_REPS, seed=cell.seed)
        binding = cell.variant == "same"
        passes_sign = bool(svd_loo["s_top1_frac"] > sign_null["p95"])
        passes_row = bool(svd_loo["s_top1_frac"] > row_null["p95"])
        loo_per_cell[cell.name] = {
            "variant": cell.variant,
            "arm": cell.arm,
            "seed": cell.seed,
            "source_dropped": SOURCE_PERSONA,
            "s_top1_frac_full": float(c["svd"]["s_top1_frac"]),
            "s_top1_frac_loo": float(svd_loo["s_top1_frac"]),
            "mean_cos_to_U1_loo": float(np.mean(svd_loo["cos_to_U1"])),
            "sign_flip": _null_summary_entry(sign_null),
            "row_shuffle": _null_summary_entry(row_null),
            "passes_sign_flip_p95": passes_sign,  # BINDING null (same variant)
            "passes_row_shuffle_p95": passes_row,  # descriptive sensitivity only
            "margin_over_sign_flip_p95": float(svd_loo["s_top1_frac"] - sign_null["p95"]),
            "margin_over_row_shuffle_p95": float(svd_loo["s_top1_frac"] - row_null["p95"]),
            "is_binding_cell": binding,
        }
        # Exploratory full jackknife: drop each persona in turn, top-share only.
        jack_per_cell[cell.name] = {
            "variant": cell.variant,
            "arm": cell.arm,
            "seed": cell.seed,
            "s_top1_frac_full": float(c["svd"]["s_top1_frac"]),
            "s_top1_frac_drop": {
                persona_order[i]: float(svd_summary(np.delete(M, i, axis=1))["s_top1_frac"])
                for i in range(M.shape[1])
            },
        }
        logger.info(
            "[control_a %s] loo_top1=%.4f sign_p95=%.4f row_p95=%.4f pass_sign=%s",
            cell.name,
            svd_loo["s_top1_frac"],
            sign_null["p95"],
            row_null["p95"],
            passes_sign,
        )

    binding_cells = [v for v in loo_per_cell.values() if v["is_binding_cell"]]
    payload_loo = {
        "meta": meta,
        # Plan §6(a): PASS iff all 6 same-variant cells clear the sign-flip
        # null p95 on the LOO matrix.
        "binding_pass": all(v["passes_sign_flip_p95"] for v in binding_cells),
        "n_binding_cells": len(binding_cells),
        "per_cell": loo_per_cell,
    }
    _write_json(out_dir / "loo.json", payload_loo)
    payload_jack = {"meta": meta, "per_cell": jack_per_cell}
    _write_json(out_dir / "jackknife.json", payload_jack)
    return payload_loo


# ──────────────────────────────────────────────────────────────────────
# Control B — whole-response (mean-over-response) read
# ──────────────────────────────────────────────────────────────────────


def control_b_mean_resp(*, slot_cache: dict[str, dict], out_dir: Path, meta: dict) -> dict:
    """Symmetric EM-mr vs marker-mr PRIMARY; end-slot SECONDARY (parent ref)."""
    per_cell: dict[str, dict] = {}
    for cell in _same_cells():
        c = slot_cache[cell.name]
        m_mr, _ = assemble_M(c["shifts"], persona_order=c["persona_order"], use_mean_resp=True)
        svd_mr = svd_summary(m_mr)
        row_null = row_shuffle_null(m_mr, n_reps=N_NULL_REPS, seed=cell.seed)
        sign_null = sign_flip_null(m_mr, n_reps=N_NULL_REPS, seed=cell.seed)
        per_cell[cell.name] = {
            "arm": cell.arm,
            "seed": cell.seed,
            "mean_cos_to_U1_mean_resp": float(np.mean(svd_mr["cos_to_U1"])),
            "s_top1_frac_mean_resp": float(svd_mr["s_top1_frac"]),
            "cos_to_U1_mean_resp": [float(x) for x in svd_mr["cos_to_U1"]],
            "sign_flip": _null_summary_entry(sign_null),
            "row_shuffle": _null_summary_entry(row_null),
            "passes_sign_flip_p95": bool(svd_mr["s_top1_frac"] > sign_null["p95"]),
            "mean_cos_to_U1_end_slot_re": float(np.mean(c["svd"]["cos_to_U1"])),
            "mean_cos_to_U1_end_slot_parent": float(c["parent"]["mean_cos_to_U1"]),
        }
        logger.info(
            "[control_b %s] mean_cos_mr=%.4f top1_mr=%.4f pass_sign=%s",
            cell.name,
            per_cell[cell.name]["mean_cos_to_U1_mean_resp"],
            svd_mr["s_top1_frac"],
            per_cell[cell.name]["passes_sign_flip_p95"],
        )

    em_mr = [v["mean_cos_to_U1_mean_resp"] for v in per_cell.values() if v["arm"] == "em"]
    mk_mr = [v["mean_cos_to_U1_mean_resp"] for v in per_cell.values() if v["arm"] == "marker"]
    mk_slot = [v["mean_cos_to_U1_end_slot_re"] for v in per_cell.values() if v["arm"] == "marker"]
    em_pass_nulls = all(v["passes_sign_flip_p95"] for v in per_cell.values() if v["arm"] == "em")
    primary_pass = bool(min(em_mr) > max(mk_mr)) and em_pass_nulls
    payload = {
        "meta": meta,
        # PRIMARY (plan §6(b)): symmetric — same read type on both arms.
        "primary_symmetric": {
            "min_em_mean_cos_mean_resp": float(min(em_mr)),
            "max_marker_mean_cos_mean_resp": float(max(mk_mr)),
            "em_cells_pass_own_sign_flip_p95": em_pass_nulls,
            "binding_pass": primary_pass,
        },
        # SECONDARY (parent-reference only; crosses read types — cannot bind).
        "secondary_parent_reference": {
            "min_em_mean_cos_mean_resp": float(min(em_mr)),
            "max_marker_mean_cos_end_slot_re": float(max(mk_slot)),
            "ordering_preserved": bool(min(em_mr) > max(mk_slot)),
        },
        "per_cell": per_cell,
    }
    _write_json(out_dir / "mean_resp.json", payload)
    return payload


# ──────────────────────────────────────────────────────────────────────
# Control C — norm-vs-alignment + split-half reliability
# ──────────────────────────────────────────────────────────────────────


def _one_sided_perm_p(norms: np.ndarray, cos: np.ndarray, *, seed: int) -> tuple[float, float]:
    """Observed rho + one-sided POSITIVE Monte Carlo permutation p.

    The attenuation alternative predicts positive rho(||shift||,
    cos-to-U1) — low-cos personas being low-norm. p = P(rho_perm >=
    rho_obs) with the add-one MC correction.
    """
    rho_obs = spearman_rho(norms, cos)
    rng = np.random.default_rng(seed)
    n_ge = 0
    for _ in range(N_PERM):
        rho_p = spearman_rho(norms, rng.permutation(cos))
        if rho_p >= rho_obs:
            n_ge += 1
    p = (1 + n_ge) / (1 + N_PERM)
    return float(rho_obs), float(p)


def control_c_norm_alignment(*, slot_cache: dict[str, dict], out_dir: Path, meta: dict) -> dict:
    """Spearman rho(||shift||, cos_to_U1) per same-variant cell + reliability."""
    per_cell: dict[str, dict] = {}
    rel_per_cell: dict[str, dict] = {}
    for cell in _same_cells():
        c = slot_cache[cell.name]
        M, persona_order = c["M"], c["persona_order"]
        norms = np.linalg.norm(M, axis=0)
        cos = np.asarray(c["svd"]["cos_to_U1"], dtype=np.float64)
        rho, p_one_sided = _one_sided_perm_p(norms, cos, seed=cell.seed)
        per_cell[cell.name] = {
            "arm": cell.arm,
            "seed": cell.seed,
            "spearman_rho_norm_vs_cos": rho,
            "p_one_sided_positive": p_one_sided,
            "n_perm": N_PERM,
            "is_strong": bool(rho >= RHO_STRONG),
            "rho_strong_threshold": RHO_STRONG,
            "norms": {p: float(norms[i]) for i, p in enumerate(persona_order)},
            "cos_to_U1": {p: float(cos[i]) for i, p in enumerate(persona_order)},
        }
        logger.info(
            "[control_c %s] rho=%.3f p_one_sided=%.4f strong=%s",
            cell.name,
            rho,
            p_one_sided,
            rho >= RHO_STRONG,
        )

        # Split-half reliability per persona from the persisted per-question
        # tensors (even- vs odd-indexed question halves, deterministic).
        rel: dict[str, float | None] = {}
        for p in persona_order:
            per_q = c["shifts"][p].get("delta_v_per_q")
            if per_q is None or per_q.shape[0] < 2:
                rel[p] = None
                continue
            half_a = per_q[0::2].mean(dim=0).numpy()
            half_b = per_q[1::2].mean(dim=0).numpy()
            rel[p] = float(cosine(half_a, half_b))
        rel_ok = [(p, r) for p, r in rel.items() if r is not None]
        rel_rho = (
            spearman_rho(
                [r for _, r in rel_ok],
                [float(cos[persona_order.index(p)]) for p, _ in rel_ok],
            )
            if len(rel_ok) >= 3
            else None
        )
        rel_per_cell[cell.name] = {
            "arm": cell.arm,
            "seed": cell.seed,
            "split_half_cosine_per_persona": rel,
            "spearman_reliability_vs_cos_to_U1": rel_rho,
            "split": "even_vs_odd_question_indices",
        }

    payload = {"meta": meta, "per_cell": per_cell}
    _write_json(out_dir / "norm_alignment.json", payload)
    payload_rel = {"meta": meta, "per_cell": rel_per_cell}
    _write_json(out_dir / "reliability.json", payload_rel)
    return payload


# ──────────────────────────────────────────────────────────────────────
# Figures
# ──────────────────────────────────────────────────────────────────────


def make_figures(
    *,
    repro: dict,
    loo: dict,
    jack_path: Path,
    mean_resp: dict,
    norm_align: dict,
    reliability_path: Path,
    figures_dir: Path,
) -> None:
    """Hero three-panel + exploratory figures (paper_plots blog style)."""
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import (
        paper_palette,
        savefig_paper,
        set_paper_style,
    )

    set_paper_style("blog")
    colors = paper_palette(4)
    em_color, mk_color = colors[0], colors[1]

    same_loo = {k: v for k, v in loo["per_cell"].items() if v["variant"] == "same"}

    # ── Hero: three panels ───────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))

    # (left) full vs source-dropped top-share per same-variant cell + nulls.
    ax = axes[0]
    names = sorted(same_loo, key=lambda k: (same_loo[k]["arm"], same_loo[k]["seed"]))
    xs = np.arange(len(names))
    full_vals = [same_loo[k]["s_top1_frac_full"] for k in names]
    loo_vals = [same_loo[k]["s_top1_frac_loo"] for k in names]
    ax.bar(xs - 0.2, full_vals, width=0.38, label="full panel (14 contexts)", color=em_color)
    ax.bar(xs + 0.2, loo_vals, width=0.38, label="trained persona dropped", color=mk_color)
    for i, k in enumerate(names):
        ax.hlines(same_loo[k]["sign_flip"]["p95"], i - 0.45, i + 0.45, color="black", lw=1.2)
        ax.hlines(
            same_loo[k]["row_shuffle"]["p95"],
            i - 0.45,
            i + 0.45,
            color="gray",
            lw=1.0,
            linestyles="dashed",
        )
    ax.set_xticks(xs)
    ax.set_xticklabels(
        [f"{same_loo[k]['arm']}\nseed {same_loo[k]['seed']}" for k in names], fontsize=8
    )
    ax.set_ylabel("top-direction share of spectrum")
    ax.set_title("Shared direction survives dropping the trained persona")
    ax.legend(fontsize=8)

    # (middle) EM whole-response vs end-slot mean cosine; marker reference.
    ax = axes[1]
    pc = mean_resp["per_cell"]
    for arm, color in (("em", em_color), ("marker", mk_color)):
        cells = sorted((k for k in pc if pc[k]["arm"] == arm), key=lambda k: pc[k]["seed"])
        mr = [pc[k]["mean_cos_to_U1_mean_resp"] for k in cells]
        slot = [pc[k]["mean_cos_to_U1_end_slot_re"] for k in cells]
        seeds = [pc[k]["seed"] for k in cells]
        ax.scatter(range(len(cells)), slot, marker="o", color=color, label=f"{arm}: end slot")
        ax.scatter(
            range(len(cells)), mr, marker="s", color=color, alpha=0.55, label=f"{arm}: whole resp."
        )
        for i, s in enumerate(seeds):
            ax.annotate(str(s), (i, mr[i]), fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax.set_xticks([])
    ax.set_ylabel("mean cosine to top direction")
    ax.set_title("Whole-response read vs end-slot read")
    ax.legend(fontsize=7)

    # (right) marker ||shift|| vs cos-to-U1 scatter, 3 seeds.
    ax = axes[2]
    na = norm_align["per_cell"]
    mk_cells = sorted((k for k in na if na[k]["arm"] == "marker"), key=lambda k: na[k]["seed"])
    for i, k in enumerate(mk_cells):
        personas = list(na[k]["norms"].keys())
        ax.scatter(
            [na[k]["norms"][p] for p in personas],
            [na[k]["cos_to_U1"][p] for p in personas],
            s=18,
            color=paper_palette(4)[i],
            label=f"seed {na[k]['seed']}",
            alpha=0.8,
        )
    ax.set_xlabel("per-persona shift norm")
    ax.set_ylabel("cosine to top direction")
    ax.set_title("Marker arm: shift size vs alignment")
    ax.legend(fontsize=8)

    fig.tight_layout()
    savefig_paper(fig, "hero_three_controls", dir=figures_dir)
    plt.close(fig)

    # ── Exploratory: jackknife strip plot (same-variant cells) ───────
    with jack_path.open() as f:
        jack = json.load(f)
    same_jack = {k: v for k, v in jack["per_cell"].items() if v["variant"] == "same"}
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    names = sorted(same_jack, key=lambda k: (same_jack[k]["arm"], same_jack[k]["seed"]))
    for i, k in enumerate(names):
        v = same_jack[k]
        drops = v["s_top1_frac_drop"]
        ys = list(drops.values())
        ax.scatter([i] * len(ys), ys, s=14, alpha=0.6, color=mk_color)
        src_y = drops.get(SOURCE_PERSONA)
        if src_y is not None:
            ax.scatter([i], [src_y], s=42, color="black", zorder=3, marker="D")
        ax.scatter([i], [v["s_top1_frac_full"]], s=42, color=em_color, zorder=3, marker="_")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(
        [f"{same_jack[k]['arm']}\nseed {same_jack[k]['seed']}" for k in names], fontsize=8
    )
    ax.set_ylabel("top-direction share after dropping one persona")
    ax.set_title("Fourteen-fold jackknife (diamond = trained persona dropped)")
    fig.tight_layout()
    savefig_paper(fig, "jackknife_strip", dir=figures_dir)
    plt.close(fig)

    # ── Exploratory: split-half reliability vs cosine ─────────────────
    with reliability_path.open() as f:
        rel = json.load(f)
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for k, v in sorted(rel["per_cell"].items()):
        if v["arm"] != "marker":
            continue
        cell_cos = norm_align["per_cell"][k]["cos_to_U1"]
        pts = [
            (r, cell_cos[p]) for p, r in v["split_half_cosine_per_persona"].items() if r is not None
        ]
        ax.scatter(
            [x for x, _ in pts],
            [y for _, y in pts],
            s=18,
            alpha=0.7,
            label=f"seed {v['seed']}",
        )
    ax.set_xlabel("split-half reliability (cosine between question-half means)")
    ax.set_ylabel("cosine to top direction")
    ax.set_title("Marker arm: estimation reliability vs alignment")
    ax.legend(fontsize=8)
    fig.tight_layout()
    savefig_paper(fig, "reliability_vs_cosine", dir=figures_dir)
    plt.close(fig)

    # ── Exploratory: per-variant LOO panels ──────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0), sharey=True)
    for ax, variant in zip(axes, VARIANTS, strict=True):
        sub = {k: v for k, v in loo["per_cell"].items() if v["variant"] == variant}
        names = sorted(sub, key=lambda k: (sub[k]["arm"], sub[k]["seed"]))
        xs = np.arange(len(names))
        ax.bar(
            xs,
            [sub[k]["s_top1_frac_loo"] for k in names],
            width=0.6,
            color=[em_color if sub[k]["arm"] == "em" else mk_color for k in names],
        )
        for i, k in enumerate(names):
            ax.hlines(sub[k]["sign_flip"]["p95"], i - 0.4, i + 0.4, color="black", lw=1.2)
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{sub[k]['arm']}\n{sub[k]['seed']}" for k in names], fontsize=7)
        ax.set_title(f"{variant} variant")
    axes[0].set_ylabel("source-dropped top-direction share")
    fig.suptitle("Source-dropped spectrum by text variant (line = sign-flip null p95)")
    fig.tight_layout()
    savefig_paper(fig, "loo_by_variant", dir=figures_dir)
    plt.close(fig)

    # ── Exploratory: reproduction-delta dot plot ─────────────────────
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    names = sorted(repro["per_cell"])
    ys = np.arange(len(names))
    ax.scatter(
        [repro["per_cell"][k]["clauses"]["d_s_top1_frac"] for k in names],
        ys,
        label="|Δ top-share|",
        s=20,
    )
    ax.scatter(
        [repro["per_cell"][k]["clauses"]["d_mean_cos_to_U1"] for k in names],
        ys,
        label="|Δ mean cosine|",
        s=20,
        marker="s",
    )
    ax.axvline(REPRO_TOL, color="black", lw=1.0)
    ax.set_yticks(ys)
    ax.set_yticklabels(names, fontsize=7)
    ax.set_xlabel("absolute delta vs parent run (line = tolerance)")
    ax.set_title("Reproduction gate: re-extracted vs parent per-cell summaries")
    ax.legend(fontsize=8)
    fig.tight_layout()
    savefig_paper(fig, "reproduction_deltas", dir=figures_dir)
    plt.close(fig)

    logger.info("[figures] written to %s", figures_dir)


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(
        description="#551 VM-side controls (Phase R + A/B/C + figures)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--tensors-repo", default="superkaiba1/explore-persona-space-data")
    parser.add_argument(
        "--tensors-prefix", default="issue551_shift_reextract/analysis_tensors/shifts"
    )
    parser.add_argument(
        "--local-shifts-dir",
        default=None,
        help="Read the 18 .pt files from this dir instead of downloading from HF (smoke).",
    )
    parser.add_argument("--parent-svd-dir", default="eval_results/issue_521/svd")
    parser.add_argument(
        "--base-cosines-json",
        default="eval_results/issue_521/inputs/base_cosines.json",
        help=(
            "Optional; only feeds the DESCRIPTIVE shift_norm_vs_cosine delta in "
            "Phase R. Missing file -> that descriptive item is skipped."
        ),
    )
    parser.add_argument("--out", default="eval_results/issue_551")
    parser.add_argument("--figures-dir", default="figures/issue_551")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
    )
    # `uv run python` does NOT auto-load .env; HF download of the private
    # data repo needs the token.
    load_dotenv()

    out_dir = Path(args.out)
    controls_dir = out_dir / "controls"
    figures_dir = Path(args.figures_dir)
    parent_dir = Path(args.parent_svd_dir)

    if args.local_shifts_dir:
        shifts_dir = Path(args.local_shifts_dir)
        logger.info("[phase=load] local shifts dir %s", shifts_dir)
    else:
        shifts_dir = _download_tensors(
            args.tensors_repo, args.tensors_prefix, out_dir / "shifts_downloaded"
        )

    base_cosines = None
    bcj = Path(args.base_cosines_json) if args.base_cosines_json else None
    if bcj is not None and bcj.exists():
        with bcj.open() as f:
            base_cosines = json.load(f)
    else:
        logger.info(
            "[phase_r] base-cosines JSON %s not found — skipping the descriptive "
            "shift_norm_vs_cosine delta (not a gate input)",
            bcj,
        )

    meta = _run_meta(args)
    cells = _all_cells()

    logger.info("[phase=reproduction_gate] %d cells", len(cells))
    repro, slot_cache = phase_r(
        cells=cells,
        shifts_dir=shifts_dir,
        parent_dir=parent_dir,
        base_cosines=base_cosines,
        out_path=out_dir / "reproduction_gate.json",
        meta=meta,
    )
    if not repro["binding_pass_all"]:
        breached = [k for k, v in repro["per_cell"].items() if not v["binding_pass"]]
        logger.error(
            "[phase=reproduction_gate] BINDING BREACH in %d cell(s): %s — halting "
            "before the controls (plan #551 §7: halt-and-investigate, never a finding).",
            len(breached),
            breached,
        )
        return 3

    logger.info("[phase=control_a_loo]")
    loo = control_a_loo(cells=cells, slot_cache=slot_cache, out_dir=controls_dir, meta=meta)

    logger.info("[phase=control_b_mean_resp]")
    mean_resp = control_b_mean_resp(slot_cache=slot_cache, out_dir=controls_dir, meta=meta)

    logger.info("[phase=control_c_norm_alignment]")
    norm_align = control_c_norm_alignment(slot_cache=slot_cache, out_dir=controls_dir, meta=meta)

    logger.info("[phase=figures]")
    make_figures(
        repro=repro,
        loo=loo,
        jack_path=controls_dir / "jackknife.json",
        mean_resp=mean_resp,
        norm_align=norm_align,
        reliability_path=controls_dir / "reliability.json",
        figures_dir=figures_dir,
    )

    logger.info(
        "[phase=done] repro_pass=%s loo_binding_pass=%s mean_resp_primary_pass=%s",
        repro["binding_pass_all"],
        loo["binding_pass"],
        mean_resp["primary_symmetric"]["binding_pass"],
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
