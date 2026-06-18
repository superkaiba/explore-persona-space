# ruff: noqa: RUF001, RUF002, RUF003  # research code uses Greek letters, ×, − and ※ legitimately
"""Phase D (#597 follow-up `svd-per-checkpoint-titration-read`) — off-pod analysis.

Runs on the VM (CPU only; the pod terminated after Phase C uploads) over the
unit shift tensors + base bank produced by ``titration_svd_597.py``:

  per checkpoint-cell — ``assemble_M`` (drop the ``no_persona`` column in the
  ``qwen_default`` cell, assert 24; the parent's render-identical exclusion),
  ``svd_summary`` + sign-flip / row-shuffle nulls (1,000 reps, multiproc) at
  the PRIMARY read (layer 14, slot pooling), LOO source-dropped share, gate
  Spearman ρ against the #536-centered base bank cosine (primary) + the
  raw-pairwise labeled variant, the H3 corrected-key Δρ (key = v_src −
  realized-weight mean of the cell's trained negatives, weights READ from the
  train-pool JSONLs — 200/200/100 rows → 2:2:1 — never from plan prose), and
  the pre-registered below-floor mask (median ||Δv|| / s_half ≥ 3 at the
  layer-14 slot, calibrated at smoke);

  per unit — the consecutive-checkpoint top-direction rotation track
  cos(U1(k), U1(k+1)) + cos(U1(k), U1(end)) using ``svd_summary``'s
  mean-column sign convention (above-floor checkpoints only);

  joins — #597's behavioral per-checkpoint panel trajectories (git,
  ``eval_results/issue_597/panel_trajectories``) on the same probe rows, so
  every geometry trajectory is located against the behavioral onset;

  outputs — ``eval_results/issue_597/svd-per-checkpoint-titration-read/
  percheckpoint/<arm>_<source>.json`` (≥12 unit files), ``analysis/
  svd_titration_summary.json`` (H1/H2/H3 verdicts + cross-source exact sign
  tests), and the hero + exploratory figure dump (paper_plots conventions,
  ``svdtitration_``-prefixed) in ``figures/issue_597/``.

Secondary reads (layers {7, 21, 27} and mean-resp pooling) get descriptive
``svd_summary`` + gate ρ only — the 1,000-rep nulls run at the primary read
(the plan §9 396-cell × 2-null budget), stated as a report deviation note.
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import os
import socket
import subprocess
import sys
import time
import zlib
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

load_dotenv()

log = logging.getLogger("issue_597.analyze_titration")

FOLLOWUP_LABEL = "svd-per-checkpoint-titration-read"
PRIMARY_LAYER = 14
PRIMARY_POOLING = "slot"
FLOOR_THRESHOLD_DEFAULT = 3.0  # §11: ungrounded heuristic — calibrated at smoke
H1_WINDOW = (4, 16)  # earliest above-floor positive-only checkpoints (plan §3 H1)
CLIFF_WINDOW = (12, 40)  # rotation-minimum / gate-collapse window (plan §3 H2)
ONSET_BAND = (15, 20)  # source onset: teacher-forced probe gain crosses 5 nats (figure shading)

# Mirrors scripts/issue_597/dispatch_leakage_dynamics_597.py (the parent's
# pinned pool source — plan §11 "negatives read from the realized pool files").
HF_DATA_REPO = "superkaiba1/explore-persona-space-data"
HF_TRAIN_POOL_SUBDIR = "issue480_marker_payload_swap/train_pools"
TRAIN_POOL_REVISION = "3c8fecb937c81c13036a9697be1e4e716755321e"

PANEL_TRAJ_DIR = Path("eval_results/issue_597/panel_trajectories")


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
            env={**os.environ},
        ).strip()
    except Exception:
        return "unknown"


def _metadata(extra: dict | None = None) -> dict:
    meta = {
        "git_commit": _git_sha(),
        "hostname": socket.gethostname(),
        "ts": datetime.now(UTC).isoformat(),
        "numpy_version": np.__version__,
    }
    if extra:
        meta.update(extra)
    return meta


# ── tensor loading ───────────────────────────────────────────────────────────


def load_npz_with_meta(path: Path, expect_schema: str) -> tuple[dict, dict]:
    """Load an npz written by shift_svd; returns (meta, {key: array})."""
    npz = np.load(path, allow_pickle=False)
    meta = json.loads(str(npz["meta"]))
    if meta.get("schema") != expect_schema:
        raise RuntimeError(f"{path}: schema {meta.get('schema')!r} != {expect_schema!r}")
    return meta, {k: npz[k] for k in npz.files if k != "meta"}


def maybe_download_tensors(tensors_root: Path, units: list[str], hf_suffix: str = "") -> None:
    """Fetch missing unit npz files + base bank from the HF data repo."""
    from huggingface_hub import hf_hub_download

    wanted = [f"{u}.npz" for u in units] + ["base_bank.npz"]
    for name in wanted:
        local = tensors_root / name
        if local.exists():
            continue
        remote = f"issue597_leakage_dynamics{hf_suffix}/analysis_tensors/{name}"
        log.info("downloading %s", remote)
        cached = hf_hub_download(HF_DATA_REPO, remote, repo_type="dataset")
        tensors_root.mkdir(parents=True, exist_ok=True)
        local.write_bytes(Path(cached).read_bytes())


# ── pool-derived negative composition (H3) ───────────────────────────────────


def pool_negative_composition(source: str, pools_dir: Path, probe_rows_path: Path) -> dict:
    """Realized negative composition of one source's train pool (#527 lesson).

    Classifies every non-marker row by its system prompt → persona name (via
    the probe contexts' prompts; empty/no system message → ``no_persona``),
    asserts the realized panel matches ``TRAINED_NEGATIVES[source]`` +
    ``no_persona`` with 200/200/100 rows, and returns name → row-count
    (the 2:2:1 corrected-key weights are derived from THESE counts).
    """
    from huggingface_hub import hf_hub_download

    from explore_persona_space.experiments.leakage_dynamics_597 import (
        MARKER_TEXT,
        TRAINED_NEGATIVES,
    )

    pools_dir.mkdir(parents=True, exist_ok=True)
    local = pools_dir / f"{source}_train_pool.jsonl"
    if not local.exists():
        cached = hf_hub_download(
            HF_DATA_REPO,
            f"{HF_TRAIN_POOL_SUBDIR}/{source}_train_pool.jsonl",
            repo_type="dataset",
            revision=TRAIN_POOL_REVISION,
        )
        local.write_bytes(Path(cached).read_bytes())

    probe = json.loads(probe_rows_path.read_text())
    sp_to_name: dict[str, str] = {}
    for name, info in probe["contexts"].items():
        sp_to_name.setdefault(info["system_prompt"] or "", name)

    counts: dict[str, int] = {}
    n_pos = 0
    with open(local) as f:
        for line in f:
            row = json.loads(line)
            last = row["completion"][-1]["content"]
            if last.endswith(MARKER_TEXT):
                n_pos += 1
                continue
            sys_msgs = [m["content"] for m in row["prompt"] if m["role"] == "system"]
            sp = sys_msgs[0] if sys_msgs else ""
            if sp not in sp_to_name:
                raise RuntimeError(
                    f"{source} pool: negative row system prompt does not match any "
                    "probe context — realized-panel classification failed"
                )
            name = sp_to_name[sp]
            counts[name] = counts.get(name, 0) + 1

    expected_negs = set(TRAINED_NEGATIVES[source]) | {"no_persona"}
    if set(counts) != expected_negs:
        raise RuntimeError(
            f"{source} pool: realized negative panel {sorted(counts)} != expected "
            f"{sorted(expected_negs)} (TRAINED_NEGATIVES drift — #527 class)"
        )
    if n_pos != 200 or sorted(counts.values()) != [100, 200, 200]:
        raise RuntimeError(
            f"{source} pool: realized composition pos={n_pos}, negs={counts} != 200 + 200/200/100"
        )
    return counts


# ── geometry per cell ────────────────────────────────────────────────────────


def kept_context_indices(context_names: list[str], source: str) -> tuple[list[int], list[str]]:
    """Column subset for one cell: drop ``no_persona`` for the qwen_default
    source (render-identical duplicate column inflates rank-one — plan §8)."""
    if source == "qwen_default" and "no_persona" in context_names:
        kept = [i for i, c in enumerate(context_names) if c != "no_persona"]
    else:
        kept = list(range(len(context_names)))
    return kept, [context_names[i] for i in kept]


def cell_matrix(delta_mean_lp: np.ndarray, kept: list[int]) -> np.ndarray:
    """(C, H) fp16 per-context mean Δv → M (H, C') fp32, kept columns only."""
    M = delta_mean_lp.astype(np.float32).T[:, kept]
    assert M.ndim == 2, M.shape
    return M


def bank_vectors(bank_arrays: dict, layer: int, pooling: str) -> np.ndarray:
    """(C, H) float64 raw bank vectors for one (layer, pooling)."""
    return bank_arrays[f"bank_{pooling}_l{layer}"].astype(np.float64)


def _unit_cos(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(a @ b / (na * nb))


def gate_predictors(
    bank: np.ndarray,
    context_names: list[str],
    kept_names: list[str],
    source: str,
    neg_weights: dict[str, int] | None,
) -> dict[str, list[float] | None | str]:
    """Per-kept-context cosine predictors in centered (#536 primary) + raw geometry.

    Returns ``cos_src_centered`` / ``cos_src_raw`` (H1/H2 gate) and
    ``cos_key_centered`` / ``cos_key_raw`` (H3 corrected key = v_src −
    weight-normalized mean of the realized negatives) — or a status string
    when the bank lacks the needed contexts (smoke-scale subsets).
    """
    out: dict = {}
    if source not in context_names:
        return {"status": f"source {source!r} not in bank contexts (smoke subset)"}
    src_idx = context_names.index(source)
    kept_idx = [context_names.index(c) for c in kept_names]

    centered = bank - bank.mean(axis=0, keepdims=True)  # #536 global_mean
    for label, B in (("centered", centered), ("raw", bank)):
        v_src = B[src_idx]
        out[f"cos_src_{label}"] = [_unit_cos(B[i], v_src) for i in kept_idx]

    if neg_weights is None:
        out["h3_status"] = "pool composition unavailable (skipped)"
        return out
    missing = [n for n in neg_weights if n not in context_names]
    if missing:
        out["h3_status"] = f"negative contexts {missing} not in bank (smoke subset)"
        return out
    total = sum(neg_weights.values())
    for label, B in (("centered", centered), ("raw", bank)):
        v_src = B[src_idx]
        v_neg_w = sum((w / total) * B[context_names.index(n)] for n, w in neg_weights.items())
        v_neg_u = np.mean([B[context_names.index(n)] for n in neg_weights], axis=0)
        key_w = v_src - v_neg_w
        key_u = v_src - v_neg_u
        out[f"cos_key_{label}"] = [_unit_cos(B[i], key_w) for i in kept_idx]
        out[f"cos_key_unweighted_{label}"] = [_unit_cos(B[i], key_u) for i in kept_idx]
    out["h3_status"] = "ok"
    out["neg_weights"] = dict(neg_weights)
    return out


def _null_worker(task: tuple) -> tuple[str, int, dict]:
    """Multiproc worker: both nulls for one primary-read cell."""
    from explore_persona_space.analysis.svd_direction_constancy import (
        row_shuffle_null,
        sign_flip_null,
    )

    unit, step, M, n_reps = task
    seed = zlib.crc32(f"{unit}:{step}".encode()) % 2**31
    rs = row_shuffle_null(M, n_reps=n_reps, seed=seed)
    sf = sign_flip_null(M, n_reps=n_reps, seed=seed + 1)
    return (
        unit,
        step,
        {
            "row_shuffle_p95": rs["p95"],
            "row_shuffle_p99": rs["p99"],
            "sign_flip_p95": sf["p95"],
            "sign_flip_p99": sf["p99"],
            "n_reps": n_reps,
        },
    )


def spearman(x: list[float] | np.ndarray, y: list[float] | np.ndarray) -> float:
    from explore_persona_space.analysis.svd_direction_constancy import spearman_rho

    return spearman_rho(np.asarray(x, dtype=float), np.asarray(y, dtype=float))


# ── per-unit analysis ────────────────────────────────────────────────────────


def analyze_unit(  # noqa: C901  one linear per-unit DV pipeline; reads clearest inline
    unit_path: Path,
    bank_meta: dict,
    bank_arrays: dict,
    neg_weights: dict[str, int] | None,
    *,
    null_reps: int,
    workers: int,
    floor_threshold: float,
) -> dict:
    """All checkpoint-cell DVs + the rotation track for one arm × source unit."""
    from explore_persona_space.analysis.svd_direction_constancy import svd_summary

    meta, arrays = load_npz_with_meta(unit_path, "i597_svd_unit_v2")
    unit = meta["unit"]
    arm, source = meta["arm"], meta["source"]
    steps: list[int] = list(meta["steps"])
    context_names: list[str] = meta["context_names"]
    layers: list[int] = meta["layers"]
    poolings: list[str] = meta["poolings"]
    li_primary = layers.index(PRIMARY_LAYER)
    pi_primary = poolings.index(PRIMARY_POOLING)
    kept, kept_names = kept_context_indices(context_names, source)
    if source == "qwen_default" and "no_persona" in context_names:
        assert len(kept_names) == 24, len(kept_names)

    bank = bank_vectors(bank_arrays, PRIMARY_LAYER, PRIMARY_POOLING)
    predictors = gate_predictors(bank, context_names, kept_names, source, neg_weights)

    # ── primary M per step + floor mask ──
    per_step: dict[int, dict] = {}
    M_primary: dict[int, np.ndarray] = {}
    U1: dict[int, np.ndarray] = {}
    for k, step in enumerate(steps):
        M = cell_matrix(arrays["delta_mean"][k, li_primary, pi_primary], kept)
        M_primary[step] = M
        halves = arrays["split_half_l14_slot"][k].astype(np.float64)[:, kept, :]
        s_half = np.linalg.norm(halves[0] - halves[1], axis=1) / 2.0  # (C',)
        dv_norm = np.linalg.norm(M, axis=0)  # (C',)
        ratio = dv_norm / np.where(s_half > 0, s_half, np.nan)
        median_ratio = float(np.nanmedian(ratio))
        above_floor = bool(median_ratio >= floor_threshold)

        summ = svd_summary(M)
        U1[step] = summ["U1"]
        # LOO: drop the source column (concentration must be bystander-carried).
        rec: dict = {
            "step": step,
            "above_floor": above_floor,
            "floor_median_ratio": median_ratio,
            "floor_threshold": floor_threshold,
            "n_columns": M.shape[1],
            "top_share": summ["s_top1_frac"],
            "cos_to_U1": [float(c) for c in summ["cos_to_U1"]],
            "cos_to_U1_mean_abs": float(np.mean(np.abs(summ["cos_to_U1"]))),
            "context_norms": [float(n) for n in dv_norm],
        }
        if source in kept_names:
            loo_idx = [i for i, c in enumerate(kept_names) if c != source]
            rec["loo_top_share"] = svd_summary(M[:, loo_idx])["s_top1_frac"]
        if "cos_src_centered" in predictors:
            rec["gate_rho_centered"] = spearman(dv_norm, predictors["cos_src_centered"])
            rec["gate_rho_raw_pairwise_uncentered"] = spearman(dv_norm, predictors["cos_src_raw"])
            if predictors.get("h3_status") == "ok":
                rho_key = spearman(dv_norm, predictors["cos_key_centered"])
                rho_key_u = spearman(dv_norm, predictors["cos_key_unweighted_centered"])
                rec["h3"] = {
                    "rho_corrected_centered": rho_key,
                    "rho_corrected_unweighted_centered": rho_key_u,
                    "rho_corrected_raw": spearman(dv_norm, predictors["cos_key_raw"]),
                    "delta_rho_centered": rho_key - rec["gate_rho_centered"],
                    "delta_rho_unweighted_centered": rho_key_u - rec["gate_rho_centered"],
                }
        per_step[step] = rec

    # ── nulls at the primary read (multiproc) ──
    tasks = [(unit, step, M_primary[step], null_reps) for step in steps]
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for _u, step, nulls in pool.map(_null_worker, tasks, chunksize=1):
                per_step[step]["nulls"] = nulls
    else:
        for task in tasks:
            _, step, nulls = _null_worker(task)
            per_step[step]["nulls"] = nulls
    for step in steps:
        n = per_step[step]["nulls"]
        per_step[step]["clears_sign_flip_p95"] = bool(
            per_step[step]["top_share"] > n["sign_flip_p95"]
        )
        per_step[step]["clears_row_shuffle_p95"] = bool(
            per_step[step]["top_share"] > n["row_shuffle_p95"]
        )

    # ── secondary reads (descriptive: svd_summary + gate rho, no nulls) ──
    for k, step in enumerate(steps):
        sec: dict[str, dict] = {}
        for li, layer in enumerate(layers):
            for pi, pooling in enumerate(poolings):
                if layer == PRIMARY_LAYER and pooling == PRIMARY_POOLING:
                    continue
                M = cell_matrix(arrays["delta_mean"][k, li, pi], kept)
                s = svd_summary(M)
                entry = {
                    "top_share": s["s_top1_frac"],
                    "cos_to_U1_mean_abs": float(np.mean(np.abs(s["cos_to_U1"]))),
                }
                bank_lp = bank_vectors(bank_arrays, layer, pooling)
                pred_lp = gate_predictors(bank_lp, context_names, kept_names, source, neg_weights)
                if "cos_src_centered" in pred_lp:
                    entry["gate_rho_centered"] = spearman(
                        np.linalg.norm(M, axis=0), pred_lp["cos_src_centered"]
                    )
                sec[f"l{layer}_{pooling}"] = entry
        per_step[step]["secondary"] = sec

    # ── rotation track (above-floor only; mean-column sign convention is
    #     already applied inside svd_summary's U1) ──
    above = [s for s in steps if per_step[s]["above_floor"]]
    rotation = {
        "consecutive": [
            {
                "step_from": a,
                "step_to": b,
                "cos": _unit_cos(U1[a], U1[b]),
            }
            for a, b in itertools.pairwise(above)
        ],
        "to_endpoint": [{"step": s, "cos": _unit_cos(U1[s], U1[above[-1]])} for s in above]
        if above
        else [],
        "above_floor_steps": above,
    }

    # ── teacher-forced probe join (the parent's panel trajectories, same probe rows) ──
    from explore_persona_space.experiments.leakage_dynamics_597.analyze import (
        bystander_contexts,
        context_value,
        group_median,
        load_panel_trajectory,
        onset_step,
    )

    traj_path = PANEL_TRAJ_DIR / f"arm{arm.upper()}" / f"{source}_seed42_panel_trajectory.json"
    panel = load_panel_trajectory(traj_path)
    behavioral_steps = sorted(panel["by_step"])
    missing_steps = [s for s in steps if s not in behavioral_steps]
    if missing_steps:
        raise RuntimeError(
            f"{unit}: ladder steps {missing_steps} missing from the behavioral panel "
            f"trajectory {traj_path} — the per-checkpoint join premise broke (plan §12.10)"
        )
    byst = bystander_contexts(source)
    for step in steps:
        per_step[step]["behavioral"] = {
            "source_delta_logp": context_value(panel, step, source, "delta_logp"),
            "bystander_median_delta_logp": group_median(panel, step, byst, "delta_logp"),
            "source_emission_rate_argmax": context_value(
                panel, step, source, "emission_rate_argmax"
            ),
        }

    return {
        "schema": "i597_svd_titration_unit_v1",
        "followup_label": FOLLOWUP_LABEL,
        "unit": unit,
        "arm": arm,
        "source": source,
        "steps": steps,
        "kept_context_names": kept_names,
        "primary_read": {"layer": PRIMARY_LAYER, "pooling": PRIMARY_POOLING},
        "centering": "global_mean",  # #536 provenance for *_centered predictors
        "bank_persona_names": bank_meta["context_names"],
        "predictor_status": predictors.get("status", predictors.get("h3_status", "ok")),
        "behavioral_onset_step": onset_step(panel),
        "per_step": {str(s): per_step[s] for s in steps},
        "rotation": rotation,
        "invariant_max_abs_diff": meta.get("invariant_max_abs_diff"),
        "fourfloat_gates": meta.get("fourfloat_gates"),
        "metadata": _metadata(),
    }


# ── verdicts (plan §3) ───────────────────────────────────────────────────────


def exact_sign_test(k: int, n: int) -> dict:
    """Exact binomial sign test at p=0.5: one- and two-sided (plan convention:
    6/6 → two-sided 0.031 is the pre-registered pass)."""
    from math import comb

    tail = sum(comb(n, i) for i in range(k, n + 1)) / 2**n
    return {
        "k": k,
        "n": n,
        "p_one_sided": tail,
        "p_two_sided": min(1.0, 2 * tail),
    }


def h1_verdict(unit_results: list[dict]) -> dict:
    """H1: earliest above-floor pos-only checkpoint in steps 4–16 — top share
    clears the sign-flip null AND exceeds the endpoint share; gate ρ > 0."""
    per_source: dict[str, dict] = {}
    for res in (r for r in unit_results if r["arm"] == "b"):
        steps = res["steps"]
        endpoint = res["per_step"][str(max(steps))]
        window = [
            s
            for s in steps
            if H1_WINDOW[0] <= s <= H1_WINDOW[1] and res["per_step"][str(s)]["above_floor"]
        ]
        if not window:
            per_source[res["source"]] = {"status": "below_floor_in_window"}
            continue
        first = res["per_step"][str(window[0])]
        rho = first.get("gate_rho_centered")
        rec = {
            "status": "read",
            "earliest_above_floor_step": window[0],
            "top_share": first["top_share"],
            "clears_sign_flip_p95": first["clears_sign_flip_p95"],
            "exceeds_endpoint_share": bool(first["top_share"] > endpoint["top_share"]),
            "endpoint_top_share": endpoint["top_share"],
            "gate_rho_centered": rho,
            "pass": bool(
                first["clears_sign_flip_p95"]
                and first["top_share"] > endpoint["top_share"]
                and (rho is not None and rho > 0)
            ),
        }
        per_source[res["source"]] = rec
    read = [s for s, r in per_source.items() if r.get("status") == "read"]
    k = sum(per_source[s]["pass"] for s in read)
    return {
        "operationalization": (
            "at the EARLIEST above-floor positive-only checkpoint in steps 4-16: "
            "top_share > sign_flip_p95 AND top_share > step-528 share AND "
            "gate_rho_centered > 0; pre-registered pass = 6/6 sources"
        ),
        "per_source": per_source,
        "n_read": len(read),
        "n_pass": int(k),
        "sign_test": exact_sign_test(int(k), len(read)) if read else None,
    }


def h2_verdict(unit_results: list[dict]) -> dict:
    """H2: rotation minimum inside steps 12–40 AND the gate ρ drops from its
    early value toward ~0 across the same window (pos-only arm)."""
    per_source: dict[str, dict] = {}
    for res in (r for r in unit_results if r["arm"] == "b"):
        rot = res["rotation"]["consecutive"]
        if not rot:
            per_source[res["source"]] = {"status": "no_above_floor_pairs"}
            continue
        min_pair = min(rot, key=lambda p: p["cos"])
        min_inside = CLIFF_WINDOW[0] <= min_pair["step_to"] <= CLIFF_WINDOW[1]
        stable = all(p["cos"] >= 0.9 for p in rot if p["step_to"] <= CLIFF_WINDOW[1])
        above = res["rotation"]["above_floor_steps"]
        early = [s for s in above if s < CLIFF_WINDOW[0]] or above[:1]
        late = [s for s in above if CLIFF_WINDOW[1] <= s <= 100]

        def rho_of(s: int, _res: dict = res) -> float | None:
            return _res["per_step"][str(s)].get("gate_rho_centered")

        rho_early_vals = [rho_of(s) for s in early if rho_of(s) is not None]
        rho_late_vals = [rho_of(s) for s in late if rho_of(s) is not None]
        if not rho_early_vals or not rho_late_vals:
            per_source[res["source"]] = {"status": "insufficient_above_floor_coverage"}
            continue
        rho_early = float(np.mean(rho_early_vals))
        rho_late = float(np.mean(rho_late_vals))
        collapse = rho_early > 0 and rho_late < 0.5 * rho_early
        per_source[res["source"]] = {
            "status": "read",
            "rotation_min_cos": min_pair["cos"],
            "rotation_min_at_step": min_pair["step_to"],
            "rotation_min_inside_cliff": bool(min_inside),
            "rotation_stable_through_cliff": bool(stable),
            "gate_rho_early_mean": rho_early,
            "gate_rho_late_mean": rho_late,
            "gate_collapse": bool(collapse),
            "pass": bool(min_inside and collapse),
        }
    read = [s for s, r in per_source.items() if r.get("status") == "read"]
    k = sum(per_source[s]["pass"] for s in read)
    return {
        "operationalization": (
            "consecutive-U1 rotation minimum lands inside steps 12-40 AND mean "
            "gate_rho_centered over above-floor steps in [40, 100] < 0.5 x its "
            "pre-cliff mean (early > 0); falsified if all consecutive cosines "
            ">= 0.9 through the cliff or H1's grading never existed"
        ),
        "per_source": per_source,
        "n_read": len(read),
        "n_pass": int(k),
        "sign_test": exact_sign_test(int(k), len(read)) if read else None,
    }


def h3_verdict(unit_results: list[dict]) -> dict:
    """H3: contrastive arm's Δρ (corrected − raw key) rises and is positive
    late; the positive-only arm's stays ≤ 0."""
    per_arm: dict[str, dict] = {}
    for arm in ("a", "b"):
        per_source: dict[str, dict] = {}
        for res in (r for r in unit_results if r["arm"] == arm):
            above = [
                s
                for s in res["steps"]
                if res["per_step"][str(s)]["above_floor"] and "h3" in res["per_step"][str(s)]
            ]
            if not above:
                per_source[res["source"]] = {"status": "no_h3_reads"}
                continue
            d = [res["per_step"][str(s)]["h3"]["delta_rho_centered"] for s in above]
            n_edge = min(5, max(1, len(d) // 2))
            early_mean = float(np.mean(d[:n_edge]))
            late_mean = float(np.mean(d[-n_edge:]))
            per_source[res["source"]] = {
                "status": "read",
                "n_above_floor_h3": len(above),
                "delta_rho_early_mean": early_mean,
                "delta_rho_late_mean": late_mean,
                "late_positive": bool(late_mean > 0),
                "rising": bool(late_mean > early_mean),
            }
        read = [s for s, r in per_source.items() if r.get("status") == "read"]
        key = "late_positive_and_rising" if arm == "a" else "late_nonpositive"
        if arm == "a":
            k = sum(per_source[s]["late_positive"] and per_source[s]["rising"] for s in read)
        else:
            k = sum(not per_source[s]["late_positive"] for s in read)
        per_arm[arm] = {
            "per_source": per_source,
            "criterion": key,
            "n_read": len(read),
            "n_pass": int(k),
            "sign_test": exact_sign_test(int(k), len(read)) if read else None,
        }
    return {
        "operationalization": (
            "delta_rho = rho(norms, cos to v_src − 2:2:1-weighted mean v_neg) − "
            "rho(norms, cos to v_src), centered geometry; contrastive arm passes a "
            "source when the last-5-above-floor mean is > 0 AND > the first-5 mean; "
            "positive-only arm expects last-5 mean <= 0; descriptive paired read "
            "aggregated by cross-source sign test (plan §3 power note)"
        ),
        "per_arm": per_arm,
    }


# ── figures ──────────────────────────────────────────────────────────────────


def _series(res: dict, key: str, above_only: bool = True) -> tuple[list[int], list[float]]:
    xs, ys = [], []
    for s in res["steps"]:
        rec = res["per_step"][str(s)]
        if above_only and not rec["above_floor"]:
            continue
        v = rec.get(key)
        if v is None:
            continue
        xs.append(s)
        ys.append(v)
    return xs, ys


def make_figures(  # noqa: C901  one linear figure dump; reads clearest inline
    unit_results: list[dict],
    fig_dir: Path,
    *,
    layers: list[int],
) -> list[str]:
    """Hero + exploratory dump (plan §6), svdtitration_-prefixed."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import (
        paper_palette,
        paper_palette_role,
        savefig_paper,
        set_paper_style,
    )
    from explore_persona_space.experiments.leakage_dynamics_597.analyze import source_label

    set_paper_style()
    written: list[str] = []
    arm_label = {"b": "positive-only", "a": "contrastive"}
    by_arm: dict[str, list[dict]] = {"a": [], "b": []}
    for r in unit_results:
        by_arm[r["arm"]].append(r)

    def _shade_onset(ax):
        ax.axvspan(ONSET_BAND[0], ONSET_BAND[1], alpha=0.15, color="grey", lw=0)

    # ── hero per arm: top share + gate rho + rotation, teacher-forced probe overlay ──
    for arm, results in by_arm.items():
        if not results:
            continue
        fig, axes = plt.subplots(3, 1, figsize=(7.0, 8.4), sharex=True)
        colors = paper_palette(len(results))
        for res, color in zip(results, colors, strict=False):
            xs, ys = _series(res, "top_share")
            axes[0].plot(xs, ys, color=color, lw=1.0, alpha=0.8, label=source_label(res["source"]))
            nx, nv = _series(res, "nulls")
            if nx:
                axes[0].plot(
                    nx,
                    [v["sign_flip_p95"] for v in nv],
                    color=color,
                    lw=0.7,
                    ls="--",
                    alpha=0.4,
                )
            xs, ys = _series(res, "gate_rho_centered")
            axes[1].plot(xs, ys, color=color, lw=1.0, alpha=0.8)
            rot = res["rotation"]["consecutive"]
            axes[2].plot(
                [p["step_to"] for p in rot],
                [p["cos"] for p in rot],
                color=color,
                lw=1.0,
                alpha=0.8,
            )
        beh = axes[0].twinx()
        for res in results:
            xs = res["steps"]
            beh.plot(
                xs,
                [res["per_step"][str(s)]["behavioral"]["source_delta_logp"] for s in xs],
                color=paper_palette_role("neutral"),
                lw=0.8,
                alpha=0.35,
            )
        beh.set_ylabel("source marker log-prob gain\n(teacher-forced probe, nats)", fontsize=8)
        for ax in axes:
            _shade_onset(ax)
            ax.set_xscale("log")
        axes[0].set_ylabel("top singular share")
        axes[1].set_ylabel("gate Spearman rho (centered)")
        axes[1].axhline(0, color="grey", lw=0.6)
        axes[2].set_ylabel("consecutive U1 cosine")
        axes[2].set_xlabel("training step (log scale)")
        axes[0].legend(fontsize=6, ncol=2, frameon=False)
        axes[0].set_title(
            f"{arm_label[arm]} arm — shift-matrix geometry vs training step "
            "(layer 14 slot; above-floor checkpoints; dashed = sign-flip null p95;\n"
            "grey curves = source marker log-prob gain, teacher-forced probe; "
            "band = source onset (probe gain crosses 5 nats))",
            fontsize=8,
        )
        stem = f"svdtitration_hero_{arm_label[arm].replace('-', '_')}"
        written += [str(p) for p in savefig_paper(fig, stem, dir=fig_dir).values()]
        plt.close(fig)

    # ── small multiples per DV per arm ──
    for arm, results in by_arm.items():
        if not results:
            continue
        for dv, label in (
            ("top_share", "top singular share"),
            ("gate_rho_centered", "gate Spearman rho (centered)"),
        ):
            fig, axes = plt.subplots(2, 3, figsize=(9.5, 5.2), sharex=True, sharey=True)
            for ax, res in zip(axes.flat, results, strict=False):
                xs, ys = _series(res, dv)
                ax.plot(xs, ys, color=paper_palette_role("primary"), lw=1.2)
                xs_b, ys_b = _series(res, dv, above_only=False)
                below = [
                    (x, y)
                    for x, y in zip(xs_b, ys_b, strict=False)
                    if not res["per_step"][str(x)]["above_floor"]
                ]
                if below:
                    ax.scatter(
                        [b[0] for b in below],
                        [b[1] for b in below],
                        s=8,
                        color=paper_palette_role("neutral"),
                        alpha=0.5,
                    )
                _shade_onset(ax)
                ax.set_xscale("log")
                ax.set_title(source_label(res["source"]), fontsize=8)
            fig.suptitle(
                f"{arm_label[arm]} arm — {label} per source (grey dots = below measurement floor)",
                fontsize=9,
            )
            fig.supxlabel("training step (log scale)", fontsize=8)
            stem = f"svdtitration_small_multiples_{dv}_{arm_label[arm].replace('-', '_')}"
            written += [str(p) for p in savefig_paper(fig, stem, dir=fig_dir).values()]
            plt.close(fig)

    # ── H3 delta-rho tracks ──
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.6), sharey=True)
    for ax, arm in zip(axes, ("b", "a"), strict=False):
        for res, color in zip(by_arm[arm], paper_palette(max(1, len(by_arm[arm]))), strict=False):
            xs, ys = [], []
            for s in res["steps"]:
                rec = res["per_step"][str(s)]
                if rec["above_floor"] and "h3" in rec:
                    xs.append(s)
                    ys.append(rec["h3"]["delta_rho_centered"])
            if xs:
                ax.plot(xs, ys, color=color, lw=1.0, label=source_label(res["source"]))
        ax.axhline(0, color="grey", lw=0.6)
        _shade_onset(ax)
        ax.set_xscale("log")
        ax.set_title(f"{arm_label[arm]} arm", fontsize=9)
        ax.set_xlabel("training step (log scale)")
    axes[0].set_ylabel("delta rho (corrected key − raw source key)")
    axes[0].legend(fontsize=6, frameon=False)
    written += [
        str(p) for p in savefig_paper(fig, "svdtitration_h3_delta_rho", dir=fig_dir).values()
    ]
    plt.close(fig)

    # ── cos-to-U1 spaghetti (primary layer) per arm ──
    for arm, results in by_arm.items():
        if not results:
            continue
        fig, axes = plt.subplots(2, 3, figsize=(9.5, 5.2), sharex=True, sharey=True)
        for ax, res in zip(axes.flat, results, strict=False):
            above = res["rotation"]["above_floor_steps"]
            n_ctx = len(res["kept_context_names"])
            for ci in range(n_ctx):
                ys = [res["per_step"][str(s)]["cos_to_U1"][ci] for s in above]
                ax.plot(above, ys, lw=0.5, alpha=0.4, color=paper_palette_role("neutral"))
            ax.set_xscale("log")
            _shade_onset(ax)
            ax.set_title(source_label(res["source"]), fontsize=8)
        fig.suptitle(
            f"{arm_label[arm]} arm — per-context cosine to the top direction U1 "
            "(layer 14 slot, above-floor checkpoints)",
            fontsize=9,
        )
        stem = f"svdtitration_cos_to_u1_{arm_label[arm].replace('-', '_')}"
        written += [str(p) for p in savefig_paper(fig, stem, dir=fig_dir).values()]
        plt.close(fig)

    # ── floor mask map ──
    fig, ax = plt.subplots(figsize=(9.0, 4.2))
    units = sorted(unit_results, key=lambda r: (r["arm"], r["source"]))
    all_steps = sorted({s for r in units for s in r["steps"]})
    grid = np.full((len(units), len(all_steps)), np.nan)
    for i, res in enumerate(units):
        for j, s in enumerate(all_steps):
            if s in res["steps"]:
                grid[i, j] = 1.0 if res["per_step"][str(s)]["above_floor"] else 0.0
    im = ax.imshow(grid, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_yticks(range(len(units)))
    ax.set_yticklabels(
        [f"{arm_label[r['arm']]}: {source_label(r['source'])}" for r in units], fontsize=6
    )
    ax.set_xticks(range(0, len(all_steps), max(1, len(all_steps) // 12)))
    ax.set_xticklabels(
        [str(all_steps[i]) for i in range(0, len(all_steps), max(1, len(all_steps) // 12))],
        fontsize=6,
    )
    ax.set_xlabel("training step")
    ax.set_title(
        "Below-measurement-floor mask (green = above floor, red = below; "
        "median ||Δv||/s_half >= 3 at layer 14 slot)",
        fontsize=9,
    )
    fig.colorbar(im, ax=ax, shrink=0.7)
    written += [
        str(p) for p in savefig_paper(fig, "svdtitration_floor_mask_map", dir=fig_dir).values()
    ]
    plt.close(fig)

    # ── layer / pooling repeats (descriptive top-share heroes) ──
    for variant, key_fn, vlabel in [
        *(
            (
                f"l{layer}_slot",
                (lambda lr: lambda rec: rec["secondary"].get(f"l{lr}_slot", {}).get("top_share"))(
                    layer
                ),
                f"layer {layer} slot",
            )
            for layer in layers
            if layer != PRIMARY_LAYER
        ),
        (
            "l14_mean_resp",
            lambda rec: rec["secondary"].get("l14_mean_resp", {}).get("top_share"),
            "layer 14 mean-over-response",
        ),
    ]:
        fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.6), sharey=True)
        for ax, arm in zip(axes, ("b", "a"), strict=False):
            for res, color in zip(
                by_arm[arm], paper_palette(max(1, len(by_arm[arm]))), strict=False
            ):
                xs, ys = [], []
                for s in res["rotation"]["above_floor_steps"]:
                    v = key_fn(res["per_step"][str(s)])
                    if v is not None:
                        xs.append(s)
                        ys.append(v)
                if xs:
                    ax.plot(xs, ys, color=color, lw=1.0, label=source_label(res["source"]))
            ax.set_xscale("log")
            _shade_onset(ax)
            ax.set_title(f"{arm_label[arm]} arm", fontsize=9)
            ax.set_xlabel("training step (log scale)")
        axes[0].set_ylabel(f"top singular share ({vlabel})")
        axes[0].legend(fontsize=6, frameon=False)
        stem = f"svdtitration_top_share_{variant}"
        written += [str(p) for p in savefig_paper(fig, stem, dir=fig_dir).values()]
        plt.close(fig)

    return written


# ── main ─────────────────────────────────────────────────────────────────────


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="#597 SVD-titration Phase D (off-pod, CPU).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--tensors-root",
        type=Path,
        default=Path("data/issue597_svd_titration/analysis_tensors"),
    )
    parser.add_argument("--units", type=str, default=None, help="Comma list a_src/b_src stems.")
    parser.add_argument(
        "--download", action="store_true", help="Fetch missing unit npz files from HF."
    )
    parser.add_argument("--hf-suffix", type=str, default="", help='"_smoke" for smoke tensors.')
    parser.add_argument("--null-reps", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    parser.add_argument("--floor-threshold", type=float, default=FLOOR_THRESHOLD_DEFAULT)
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path("eval_results/issue_597/svd-per-checkpoint-titration-read"),
    )
    parser.add_argument("--fig-dir", type=Path, default=Path("figures/issue_597"))
    parser.add_argument(
        "--probe-rows", type=Path, default=Path("eval_results/issue_597/probe_rows.json")
    )
    parser.add_argument("--pools-dir", type=Path, default=Path("data/issue_597/train_pools"))
    parser.add_argument(
        "--skip-pools",
        action="store_true",
        help="Skip the H3 pool-composition read (H3 omitted; synthetic smokes).",
    )
    parser.add_argument("--skip-figures", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=os.environ.get("EPS_LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s | %(message)s",
        stream=sys.stdout,
    )
    t0 = time.time()

    if args.units:
        unit_stems = [u.strip() for u in args.units.split(",") if u.strip()]
    else:
        unit_paths = sorted(args.tensors_root.glob("[ab]_*.npz"))
        unit_stems = [p.stem for p in unit_paths]
        if not unit_stems and not args.download:
            raise RuntimeError(
                f"no unit npz files under {args.tensors_root}; pass --download or --units"
            )
        if not unit_stems and args.download:
            from explore_persona_space.experiments.leakage_dynamics_597 import (
                SOURCE_PERSONAS,
            )

            unit_stems = [f"{arm}_{s}" for arm in ("b", "a") for s in SOURCE_PERSONAS]
    if args.download:
        maybe_download_tensors(args.tensors_root, unit_stems, args.hf_suffix)

    # v2 = pre-final-norm layer-27 residuals (hook mechanism, round-2 fix); a
    # stale v1 bank (post-norm L27) fails loud here instead of mixing spaces.
    bank_meta, bank_arrays = load_npz_with_meta(
        args.tensors_root / "base_bank.npz", "i597_svd_base_bank_v2"
    )
    assert bank_meta["centering"] == "global_mean", bank_meta["centering"]

    # H3 negative composition per source (realized pools, #527 lesson).
    neg_by_source: dict[str, dict[str, int] | None] = {}
    sources = sorted({stem.split("_", 1)[1] for stem in unit_stems})
    for source in sources:
        if args.skip_pools:
            neg_by_source[source] = None
            continue
        neg_by_source[source] = pool_negative_composition(source, args.pools_dir, args.probe_rows)
        log.info("pool composition %s: %s", source, neg_by_source[source])

    unit_results: list[dict] = []
    percheckpoint_dir = args.out_root / "percheckpoint"
    percheckpoint_dir.mkdir(parents=True, exist_ok=True)
    for stem in unit_stems:
        unit_path = args.tensors_root / f"{stem}.npz"
        log.info("analyzing %s ...", stem)
        res = analyze_unit(
            unit_path,
            bank_meta,
            bank_arrays,
            neg_by_source[stem.split("_", 1)[1]],
            null_reps=args.null_reps,
            workers=args.workers,
            floor_threshold=args.floor_threshold,
        )
        unit_results.append(res)
        out_path = percheckpoint_dir / f"{stem}.json"
        tmp = out_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(res, indent=2, ensure_ascii=False))
        os.replace(tmp, out_path)
        log.info("  -> %s (%d steps)", out_path, len(res["steps"]))

    summary = {
        "schema": "i597_svd_titration_summary_v1",
        "followup_label": FOLLOWUP_LABEL,
        "units": [r["unit"] for r in unit_results],
        "primary_read": {"layer": PRIMARY_LAYER, "pooling": PRIMARY_POOLING},
        "floor_threshold": args.floor_threshold,
        "null_reps": args.null_reps,
        "centering": "global_mean",
        "scope_notes": [
            "within-flavor (fixed base text) trend claims only — #521 base-text "
            "endpoint levels are the expected anchor, never cross-flavor absolutes",
            "single seed (42); seed gap carried as a scope caveat per the followup spec",
            "contrastive grid starts AT step 20 while onsets are 15-20: its pre-onset "
            "window is unobservable (named asymmetry)",
            "qwen_default contrastive cell carries the parent's contradictory-"
            "supervision caveat — reported but flagged",
            "nulls (1,000 reps) run at the primary read (layer 14 slot) only; "
            "secondary layers/poolings are descriptive (plan §9 budget)",
        ],
        "h1": h1_verdict(unit_results),
        "h2": h2_verdict(unit_results),
        "h3": h3_verdict(unit_results),
        "metadata": _metadata({"wall_seconds": round(time.time() - t0, 1)}),
    }
    analysis_dir = args.out_root / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    summary_path = analysis_dir / "svd_titration_summary.json"
    tmp = summary_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    os.replace(tmp, summary_path)
    log.info("summary -> %s", summary_path)

    if not args.skip_figures:
        layers = [int(x) for x in bank_meta["layers"]]
        written = make_figures(unit_results, args.fig_dir, layers=layers)
        log.info("figures: %d files under %s", len(written), args.fig_dir)

    log.info("done in %.1fs", time.time() - t0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
