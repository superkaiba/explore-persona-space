#!/usr/bin/env python3
# Research notation (−, ×, α) is intentional in prose.
# ruff: noqa: RUF002, RUF003
"""Task #606 — VM-side analysis: judging + matched-strength gap + bootstrap.

Phase 6 (plan §4.1): runs OFF-POD against the Hub artifacts the dispatcher
uploaded. Steps:

  1. Refetch selection.json / trajectory / generation_manifest.json /
     per-(cell, persona) generation JSONs from the Hub when absent locally.
  2. Judge every (cell, persona) cell with the pinned Haiku judge
     (``i606_common.judge_generation_file`` — checkpointed per cell under
     ``verdicts/``; the SAME function stage A used pod-side).
  3. Stage-B-governed strength s per cell (reconciler-recommended, adopted):
     source-self degenerate-clean rate minus the base cell's, with the
     stage-A native-LoRA discrepancy reported per selected cell.
  4. Matched-strength read at s* = 0.50: per BYSTANDER persona (38; source
     EXCLUDED — reconciler binding fix 1), piecewise-linear interpolation of
     leakage delta in s across the arm's selected cells; bystander-mean;
     gap = FT − LoRA. 10,000-rep crossed cluster bootstrap over
     (claims x personas) on per-rollout verdicts, re-estimating s and
     re-interpolating inside each replicate (the #514
     ``_compute_matched_rate_gap_514`` / ``_crossed_cluster_bootstrap_gap``
     recipe, adapted to rate space). Determinacy gate 0.05 (0.03 sensitivity
     reported). Profile Spearman rho + bootstrap CI.
  5. Gap-vs-s* sweep at targets {0.2..0.9} + 0.75 (anchored by base s=0 and
     the selected cells). Fallback ladder (§4.4(b)) when an arm does not
     bracket s* under stage-B-governed s: band-entry comparison + nearest
     co-bracketed target, reported as a recovery event — never silently
     extrapolated.
  6. Parity anchors (#591 ladder): LoRA endpoint source-self delta vs frozen
     (syco 0.914 / refusal 0.994), base panel vs frozen #411 base rates
     (syco), refusal bystander spot anchors; +-0.08 tol / +-0.15 hard-fail
     (hard-fail raises AFTER persisting the report; smoke tier = log-only).

Synthetic smoke fixture (CPU, no API, no GPU)::

    uv run python scripts/issue_606/i606_analyze.py --make-synthetic /tmp/i606_syn \
        --synthetic-mode bracket
    uv run python scripts/issue_606/i606_analyze.py --behavior sycophancy \
        --eval-root /tmp/i606_syn --no-refetch --bootstrap-b 500

Production::

    uv run python scripts/issue_606/i606_analyze.py --behavior sycophancy \
        --eval-root eval_results/issue_606
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import os
import shutil
import socket
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts" / "issue_606"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

import numpy as np  # noqa: E402
from i606_common import (  # noqa: E402
    BOOTSTRAP_B,
    BOOTSTRAP_SEED,
    DETERMINACY_GATE,
    DETERMINACY_SENSITIVITY,
    EQUIVALENCE_CI,
    FROZEN_ANCHORS,
    HF_DATA_REPO,
    HF_EXPERIMENT_NAME,
    ISSUE411_BASE_PANEL_RATES_REL,
    JUDGE_MODEL,
    PARITY_HARD_TOL,
    PARITY_TOL,
    PROFILE_RHO_MIN,
    S_BAND,
    S_SECONDARY,
    S_SWEEP_TARGETS,
    S_TARGET,
    SOURCE_PERSONA,
    judge_generation_file,
)

log = logging.getLogger("issue_606.analyze")


def _git_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO,
            text=True,
            stderr=subprocess.DEVNULL,
            env={**os.environ},  # epm-lint: subprocess-env-inherit -- git sha probe
        ).strip()
    except (subprocess.SubprocessError, OSError):
        return None


# ---------------------------------------------------------------------------
# Hub refetch
# ---------------------------------------------------------------------------


def _ensure_local(root: Path, behavior: str, rel: str, *, experiment: str, refetch: bool) -> Path:
    """Return ``<root>/<behavior>/<rel>``, fetching the Hub copy when absent."""
    local = root / behavior / rel
    if local.exists():
        return local
    if not refetch:
        raise FileNotFoundError(f"{local} missing and --no-refetch set")
    from huggingface_hub import hf_hub_download

    got = hf_hub_download(
        HF_DATA_REPO,
        f"{experiment}/{behavior}/{rel}",
        repo_type="dataset",
        token=os.environ.get("HF_TOKEN"),
    )
    local.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(got, local)
    log.info("refetched %s/%s from the Hub", behavior, rel)
    return local


def _install_failure_report(
    root: Path, behavior: str, *, experiment: str, refetch: bool
) -> dict | None:
    """Kill-criterion (a) pre-check: resolve ``stage_a/install_failure.json``
    locally, then from the Hub. A pod-killed behavior uploads ONLY ``stage_a/``
    (the dispatcher skips p4, so ``generation_manifest.json`` never lands), so
    this marker MUST be resolved BEFORE the analyze fetch chain — on the
    canonical fresh-VM refetch flow the local tree is empty and a local-only
    check would crash at the generation-manifest Hub fetch instead of emitting
    the registered kill-report.

    Returns the parsed kill payload when the marker exists, else None.
    Absence of this OPTIONAL marker (locally and on the Hub) is the healthy
    case, not an error; any other Hub failure (repo missing, auth, network)
    still raises through ``_ensure_local``.
    """
    rel = "stage_a/install_failure.json"
    local = root / behavior / rel
    if not local.exists() and refetch:
        from huggingface_hub.utils import EntryNotFoundError

        try:
            local = _ensure_local(root, behavior, rel, experiment=experiment, refetch=True)
        except EntryNotFoundError:
            return None
    if not local.exists():
        return None
    return json.loads(local.read_text())


# ---------------------------------------------------------------------------
# Verdict-matrix construction
# ---------------------------------------------------------------------------


def _per_claim_counts(verdicts: list[dict], n_claims: int) -> dict[str, np.ndarray]:
    """Per-claim positive / total counts, raw and degenerate-clean."""
    pos_raw = np.zeros(n_claims)
    cnt_raw = np.zeros(n_claims)
    pos_clean = np.zeros(n_claims)
    cnt_clean = np.zeros(n_claims)
    for v in verdicts:
        c = int(v["claim_idx"])
        cnt_raw[c] += 1
        pos_raw[c] += int(bool(v["agreed"]))
        if not v.get("degenerate", False):
            cnt_clean[c] += 1
            pos_clean[c] += int(bool(v["agreed"]))
    return {"pos_raw": pos_raw, "cnt_raw": cnt_raw, "pos_clean": pos_clean, "cnt_clean": cnt_clean}


def _rate(pos: np.ndarray, cnt: np.ndarray) -> float:
    tot = cnt.sum()
    return float(pos.sum() / tot) if tot > 0 else float("nan")


def _interp_at(xs: np.ndarray, ys: np.ndarray, target: float) -> np.ndarray | float:
    """Piecewise-linear interpolation across (xs, ys) at ``target``, with
    extrapolation from the two nearest anchors outside the range (the #508
    ``_linear_interp`` convention). Vectorized over leading axes.

    xs shape (..., A), ys shape (..., A[, P]) sorted handled internally.
    """
    order = np.argsort(xs, axis=-1)
    xs_s = np.take_along_axis(xs, order, axis=-1)
    if ys.ndim == xs.ndim:
        ys_s = np.take_along_axis(ys, order, axis=-1)
    else:  # ys has trailing persona axis
        ys_s = np.take_along_axis(ys, order[..., None], axis=-2)
    n_anchor = xs_s.shape[-1]
    if n_anchor < 2:
        return np.full(ys_s.shape[:-1] if ys.ndim == xs.ndim else ys_s.shape[:-2], np.nan)
    pos = (xs_s < target).sum(axis=-1)
    hi = np.clip(pos, 1, n_anchor - 1)
    lo = hi - 1
    x_lo = np.take_along_axis(xs_s, lo[..., None], axis=-1)[..., 0]
    x_hi = np.take_along_axis(xs_s, hi[..., None], axis=-1)[..., 0]
    denom = x_hi - x_lo
    frac = np.where(denom == 0, 0.0, (target - x_lo) / np.where(denom == 0, 1.0, denom))
    if ys.ndim == xs.ndim:
        y_lo = np.take_along_axis(ys_s, lo[..., None], axis=-1)[..., 0]
        y_hi = np.take_along_axis(ys_s, hi[..., None], axis=-1)[..., 0]
        return y_lo + frac * (y_hi - y_lo)
    y_lo = np.take_along_axis(ys_s, lo[..., None, None], axis=-2)[..., 0, :]
    y_hi = np.take_along_axis(ys_s, hi[..., None, None], axis=-2)[..., 0, :]
    return y_lo + frac[..., None] * (y_hi - y_lo)


def _spearman(a: np.ndarray, b: np.ndarray, axis: int = -1) -> np.ndarray:
    """Spearman rho along ``axis`` (scipy rankdata handles ties)."""
    from scipy.stats import rankdata

    ra = rankdata(a, axis=axis)
    rb = rankdata(b, axis=axis)
    ra = ra - ra.mean(axis=axis, keepdims=True)
    rb = rb - rb.mean(axis=axis, keepdims=True)
    num = (ra * rb).sum(axis=axis)
    den = np.sqrt((ra**2).sum(axis=axis) * (rb**2).sum(axis=axis))
    return np.where(den == 0, np.nan, num / den)


def _bracket_info(s_cells: dict[str, float], target: float) -> dict:
    """Does an adjacent (in s) pair of SELECTED cells straddle ``target``?"""
    items = sorted(s_cells.items(), key=lambda kv: kv[1])
    for (c_lo, s_lo), (c_hi, s_hi) in itertools.pairwise(items):
        if s_lo <= target <= s_hi:
            return {"brackets": True, "pair": [c_lo, c_hi], "s_pair": [s_lo, s_hi]}
    return {"brackets": False, "pair": None, "s_pair": None}


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------


def analyze_behavior(  # noqa: C901 - one linear pipeline; splitting would scatter the registered stats
    *,
    behavior: str,
    eval_root: Path,
    experiment: str,
    bootstrap_b: int,
    refetch: bool,
    judge_concurrency: int = 32,
) -> dict:
    broot = eval_root / behavior
    selection = json.loads(
        _ensure_local(
            eval_root, behavior, "stage_a/selection.json", experiment=experiment, refetch=refetch
        ).read_text()
    )
    trajectory = json.loads(
        _ensure_local(
            eval_root,
            behavior,
            f"stage_a/trajectory_{behavior}.json",
            experiment=experiment,
            refetch=refetch,
        ).read_text()
    )
    manifest = json.loads(
        _ensure_local(
            eval_root,
            behavior,
            "generation_manifest.json",
            experiment=experiment,
            refetch=refetch,
        ).read_text()
    )
    smoke_tier = bool(selection.get("smoke") or selection.get("dry_run"))

    cells = sorted(manifest["cells"])
    if "base" not in cells:
        raise RuntimeError(f"[{behavior}] manifest has no base cell — stage B incomplete")
    panels_by_cell = {c: list(manifest["cells"][c]["panels"]) for c in cells}
    panel = sorted(set.intersection(*(set(p) for p in panels_by_cell.values())))
    for c in cells:
        extra = set(panels_by_cell[c]) - set(panel)
        if extra:
            raise RuntimeError(f"[{behavior}] cell {c} has non-shared panels {extra}")
    if SOURCE_PERSONA not in panel:
        raise RuntimeError(f"[{behavior}] source persona missing from stage-B panel")
    bystanders = [p for p in panel if p != SOURCE_PERSONA]
    if not smoke_tier and len(bystanders) != 38:
        raise RuntimeError(
            f"[{behavior}] production tier requires 38 bystanders, got {len(bystanders)}"
        )

    # -- Step 2: judge every (cell, persona) --
    counts: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    lengths: dict[str, dict[str, list[int]]] = {}
    n_claims = None
    for c in cells:
        counts[c] = {}
        lengths[c] = {}
        for p in panel:
            gen_rel = f"generations/{c}/{behavior}_eval_{p}.json"
            verdict_path = broot / "verdicts" / f"{c}__{p}.json"
            if not verdict_path.exists():
                gen_json = _ensure_local(
                    eval_root, behavior, gen_rel, experiment=experiment, refetch=refetch
                )
            else:
                gen_json = broot / gen_rel  # may be absent; cached verdict suffices
            cell = judge_generation_file(
                gen_json,
                verdict_path,
                behavior=behavior,
                dry_run=False,
                max_concurrency=judge_concurrency,
            )
            claim_count = 1 + max(int(v["claim_idx"]) for v in cell["verdicts"])
            if n_claims is None:
                n_claims = claim_count
            elif claim_count != n_claims:
                raise RuntimeError(
                    f"[{behavior}] claim-count mismatch: {c}/{p} has {claim_count}, "
                    f"expected {n_claims}"
                )
            counts[c][p] = _per_claim_counts(cell["verdicts"], n_claims)
            lengths[c][p] = [int(v.get("completion_chars", 0)) for v in cell["verdicts"]]
    assert n_claims is not None

    # -- Step 3: plug-in rates, stage-B-governed s, per-persona deltas --
    rate_clean = {
        c: {p: _rate(counts[c][p]["pos_clean"], counts[c][p]["cnt_clean"]) for p in panel}
        for c in cells
    }
    rate_raw = {
        c: {p: _rate(counts[c][p]["pos_raw"], counts[c][p]["cnt_raw"]) for p in panel}
        for c in cells
    }
    s_stage_b = {
        c: rate_clean[c][SOURCE_PERSONA] - rate_clean["base"][SOURCE_PERSONA]
        for c in cells
        if c != "base"
    }
    native_vs_merge = {}
    for c, s_b in s_stage_b.items():
        traj_cell = trajectory["cells"].get(c)
        if traj_cell is not None and "s" in traj_cell:
            native_vs_merge[c] = {
                "stage_a_native_s": traj_cell["s"],
                "stage_b_merge_s": s_b,
                "discrepancy": s_b - traj_cell["s"],
            }
    delta_clean = {
        c: {p: rate_clean[c][p] - rate_clean["base"][p] for p in panel}
        for c in cells
        if c != "base"
    }
    delta_raw = {
        c: {p: rate_raw[c][p] - rate_raw["base"][p] for p in panel} for c in cells if c != "base"
    }

    arms = {
        "lora": [c for c in cells if c.startswith("lora_")],
        "ft": [c for c in cells if c.startswith("ft_")],
    }
    for arm, arm_cells in arms.items():
        if not arm_cells:
            raise RuntimeError(f"[{behavior}] no stage-B cells for arm {arm}")

    # -- Step 4: bracket / fallback bookkeeping (stage-B-governed s) --
    recovery_events: list[dict] = []
    arm_bracket = {}
    for arm, arm_cells in arms.items():
        info = _bracket_info({c: s_stage_b[c] for c in arm_cells}, S_TARGET)
        stage_a_pair = selection["arms"][arm].get("bracket_pair")
        info["stage_a_bracket_pair"] = stage_a_pair
        if stage_a_pair is not None and info["brackets"]:
            stage_a_cells = {f"{arm}_step{s}" for s in stage_a_pair}
            if set(info["pair"]) != stage_a_cells:
                info["stage_b_rebracket"] = True
                recovery_events.append(
                    {
                        "kind": "stage_b_rebracket",
                        "arm": arm,
                        "note": "stage-B-governed s moved the bracketing pair",
                        "stage_a_pair": stage_a_pair,
                        "stage_b_pair": info["pair"],
                    }
                )
        if not info["brackets"]:
            # §4.4(b) fallback step 1: band-entry checkpoint (first by step
            # with s in band; else closest approach to s*).
            by_step = sorted(arm_cells, key=lambda c: int(c.split("step")[-1]))
            in_band = [c for c in by_step if S_BAND[0] <= s_stage_b[c] <= S_BAND[1]]
            fallback_cell = (
                in_band[0]
                if in_band
                else min(arm_cells, key=lambda c: abs(s_stage_b[c] - S_TARGET))
            )
            info["fallback_cell"] = fallback_cell
            info["fallback_mode"] = "band_entry" if in_band else "closest_approach"
            recovery_events.append(
                {
                    "kind": "no_stage_b_bracket",
                    "arm": arm,
                    "fallback_cell": fallback_cell,
                    "fallback_mode": info["fallback_mode"],
                    "note": "matched-dial/unmatched-step band-entry read (plan §4.4(b))",
                }
            )
        arm_bracket[arm] = info
    headline_mode = (
        "matched_interpolation"
        if all(i["brackets"] for i in arm_bracket.values())
        else "band_entry_fallback"
    )

    # -- arrays for vectorized bootstrap --
    p_index = {p: i for i, p in enumerate(panel)}
    bys_idx = np.array([p_index[p] for p in bystanders])
    src_idx = p_index[SOURCE_PERSONA]
    cell_list = cells  # includes base
    c_index = {c: i for i, c in enumerate(cell_list)}
    POS = np.stack(
        [np.stack([counts[c][p]["pos_clean"] for p in panel]) for c in cell_list]
    )  # (C, P, K)
    CNT = np.stack([np.stack([counts[c][p]["cnt_clean"] for p in panel]) for c in cell_list])
    n_p = len(panel)

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    B = bootstrap_b
    claim_picks = rng.integers(0, n_claims, size=(n_p, B, n_claims))  # locked across cells
    persona_picks = rng.integers(0, len(bystanders), size=(B, len(bystanders)))

    # replicate rates: (C, P, B)
    rate_rep = np.empty((len(cell_list), n_p, B))
    for pi in range(n_p):
        picks = claim_picks[pi]  # (B, K)
        pos_sel = POS[:, pi, :][:, picks]  # (C, B, K)
        cnt_sel = CNT[:, pi, :][:, picks]
        tot = cnt_sel.sum(axis=-1)
        with np.errstate(invalid="ignore", divide="ignore"):
            rate_rep[:, pi, :] = np.where(tot > 0, pos_sel.sum(axis=-1) / tot, np.nan)
    s_rep = rate_rep[:, src_idx, :] - rate_rep[c_index["base"], src_idx, :]  # (C, B)
    delta_rep = rate_rep - rate_rep[c_index["base"], :, :][None, :, :]  # (C, P, B)

    targets = sorted(set(S_SWEEP_TARGETS) | {S_TARGET, S_SECONDARY})

    def _arm_interp_rep(arm_cells: list[str], target: float, *, with_base_anchor: bool):
        """Per-replicate per-persona interpolated bystander deltas (B, n_bys)."""
        idxs = [c_index[c] for c in arm_cells]
        xs = s_rep[idxs, :].T  # (B, A)
        ys = delta_rep[idxs, :, :][:, bys_idx, :].transpose(2, 0, 1)  # (B, A, n_bys)
        if with_base_anchor:
            xs = np.concatenate([np.zeros((B, 1)), xs], axis=1)
            ys = np.concatenate([np.zeros((B, 1, len(bys_idx))), ys], axis=1)
        return _interp_at(xs, ys, target)  # (B, n_bys)

    def _arm_interp_plugin(arm_cells: list[str], target: float, *, with_base_anchor: bool):
        xs = np.array([s_stage_b[c] for c in arm_cells])
        ys = np.array([[delta_clean[c][p] for p in bystanders] for c in arm_cells])  # (A, n_bys)
        if with_base_anchor:
            xs = np.concatenate([[0.0], xs])
            ys = np.concatenate([np.zeros((1, len(bystanders))), ys], axis=0)
        return _interp_at(xs, ys, target)  # (n_bys,)

    # -- headline at s* (cell-only anchors per plan §4.4 matched read) --
    if headline_mode == "matched_interpolation":
        lora_plug = _arm_interp_plugin(arms["lora"], S_TARGET, with_base_anchor=False)
        ft_plug = _arm_interp_plugin(arms["ft"], S_TARGET, with_base_anchor=False)
        lora_rep = _arm_interp_rep(arms["lora"], S_TARGET, with_base_anchor=False)
        ft_rep = _arm_interp_rep(arms["ft"], S_TARGET, with_base_anchor=False)
    else:
        # Band-entry fallback: each arm read AT its fallback cell (no interp).
        def _cell_plug(arm: str) -> np.ndarray:
            c = arm_bracket[arm].get("fallback_cell") or arms[arm][-1]
            return np.array([delta_clean[c][p] for p in bystanders])

        def _cell_rep(arm: str) -> np.ndarray:
            c = arm_bracket[arm].get("fallback_cell") or arms[arm][-1]
            return delta_rep[c_index[c]][bys_idx, :].T  # (B, n_bys)

        lora_plug, ft_plug = _cell_plug("lora"), _cell_plug("ft")
        lora_rep, ft_rep = _cell_rep("lora"), _cell_rep("ft")

    gap_plug = float(np.nanmean(ft_plug) - np.nanmean(lora_plug))
    bys_mean_rep_l = np.take_along_axis(lora_rep, persona_picks, axis=1).mean(axis=1)
    bys_mean_rep_f = np.take_along_axis(ft_rep, persona_picks, axis=1).mean(axis=1)
    gap_rep = bys_mean_rep_f - bys_mean_rep_l
    gap_rep_valid = gap_rep[np.isfinite(gap_rep)]
    if len(gap_rep_valid) < 0.5 * B:
        raise RuntimeError(
            f"[{behavior}] >{B // 2} bootstrap replicates non-finite — degenerate cells "
            f"or empty clean denominators; inspect per-cell tables"
        )
    gap_ci = (
        float(np.quantile(gap_rep_valid, 0.025)),
        float(np.quantile(gap_rep_valid, 0.975)),
    )
    gap_boot_mean = float(gap_rep_valid.mean())
    determinacy = abs(gap_plug - gap_boot_mean)

    rho_plug = float(_spearman(lora_plug[None, :], ft_plug[None, :])[0])
    lora_rep_pick = np.take_along_axis(lora_rep, persona_picks, axis=1)
    ft_rep_pick = np.take_along_axis(ft_rep, persona_picks, axis=1)
    rho_rep = _spearman(lora_rep_pick, ft_rep_pick, axis=1)
    rho_rep_valid = rho_rep[np.isfinite(rho_rep)]
    rho_ci = (
        (float(np.quantile(rho_rep_valid, 0.025)), float(np.quantile(rho_rep_valid, 0.975)))
        if len(rho_rep_valid)
        else (float("nan"), float("nan"))
    )

    # -- verdict per plan §6 success criteria --
    ci_lo, ci_hi = gap_ci
    determinate = determinacy <= DETERMINACY_GATE
    if not determinate:
        verdict = "indeterminate_determinacy_gate"
    elif EQUIVALENCE_CI[0] < ci_lo and ci_hi < EQUIVALENCE_CI[1] and rho_plug >= PROFILE_RHO_MIN:
        verdict = "equivalence"
    elif (ci_lo > 0 or ci_hi < 0) and abs(gap_plug) >= 0.05:
        verdict = "divergence"
    elif rho_ci[1] < PROFILE_RHO_MIN and rho_plug < 0.4:
        verdict = "divergence_profile"
    else:
        verdict = "indeterminate"

    # -- gap-vs-s* sweep (base-anchored; plug-in + bootstrap per target) --
    sweep = []
    max_s = {arm: max(s_stage_b[c] for c in arms[arm]) for arm in arms}
    for t in targets:
        in_range = all(t <= max_s[arm] for arm in arms)
        lp = _arm_interp_plugin(arms["lora"], t, with_base_anchor=True)
        fp = _arm_interp_plugin(arms["ft"], t, with_base_anchor=True)
        lr = _arm_interp_rep(arms["lora"], t, with_base_anchor=True)
        fr = _arm_interp_rep(arms["ft"], t, with_base_anchor=True)
        g_rep = np.take_along_axis(fr, persona_picks, axis=1).mean(axis=1) - np.take_along_axis(
            lr, persona_picks, axis=1
        ).mean(axis=1)
        g_valid = g_rep[np.isfinite(g_rep)]
        sweep.append(
            {
                "target": t,
                "in_range_both_arms": bool(in_range),
                "gap_plugin": float(np.nanmean(fp) - np.nanmean(lp)),
                "lora_mean": float(np.nanmean(lp)),
                "ft_mean": float(np.nanmean(fp)),
                "gap_ci": [
                    float(np.quantile(g_valid, 0.025)) if len(g_valid) else float("nan"),
                    float(np.quantile(g_valid, 0.975)) if len(g_valid) else float("nan"),
                ],
            }
        )

    # -- parity anchors (#591 ladder; raw-judge convention for frozen parity) --
    frozen = FROZEN_ANCHORS[behavior]
    checks: list[dict] = []
    lora_endpoint = max(arms["lora"], key=lambda c: int(c.split("step")[-1]))
    self_delta_raw = delta_raw[lora_endpoint][SOURCE_PERSONA]
    checks.append(
        {
            "kind": "self_delta",
            "cell": lora_endpoint,
            "rerun": self_delta_raw,
            "frozen": frozen["self_delta"],
            "drift": self_delta_raw - frozen["self_delta"],
        }
    )
    base_self = rate_raw["base"][SOURCE_PERSONA]
    checks.append(
        {
            "kind": "base_self_rate",
            "cell": "base",
            "rerun": base_self,
            "frozen": frozen["base_self_rate"],
            "drift": base_self - frozen["base_self_rate"],
        }
    )
    for p, ref in frozen["bystander_spot_deltas"].items():
        if p in panel:
            got = delta_raw[lora_endpoint][p]
            checks.append(
                {
                    "kind": "bystander_spot_delta",
                    "cell": lora_endpoint,
                    "panel": p,
                    "rerun": got,
                    "frozen": ref,
                    "drift": got - ref,
                }
            )
    if behavior == "sycophancy":
        frozen_base_path = REPO / ISSUE411_BASE_PANEL_RATES_REL
        if frozen_base_path.exists():
            base_frozen = json.loads(frozen_base_path.read_text())["panel_rates"]
            for p in panel:
                if p in base_frozen:
                    got = rate_raw["base"][p]
                    checks.append(
                        {
                            "kind": "base_anchor",
                            "cell": "base",
                            "panel": p,
                            "rerun": got,
                            "frozen": base_frozen[p],
                            "drift": got - base_frozen[p],
                        }
                    )
        else:
            log.warning("frozen base-panel rates missing at %s", frozen_base_path)
    for c in checks:
        c["within_tol"] = abs(c["drift"]) <= PARITY_TOL
        c["hard_fail"] = abs(c["drift"]) > PARITY_HARD_TOL
    n_out = sum(1 for c in checks if not c["within_tol"])
    n_hard = sum(1 for c in checks if c["hard_fail"])
    parity_verdict = "PASS"
    if n_hard > 0 or n_out >= 2:
        parity_verdict = "HARD_FAIL"
    elif n_out == 1:
        parity_verdict = "MARGINAL_MISS"
    parity = {
        "checks": checks,
        "n_out_of_tol": n_out,
        "n_hard_fail": n_hard,
        "tolerance": PARITY_TOL,
        "hard_tolerance": PARITY_HARD_TOL,
        "verdict": parity_verdict,
        "gate_evaluated_smoke": smoke_tier,
        "convention": "raw-judge rates/deltas (frozen panels predate degenerate cleaning)",
    }

    # -- per-cell tables (raw alongside clean — plan §6) --
    per_cell_tables = {
        c: {
            p: {
                "rate_raw": rate_raw[c][p],
                "rate_clean": rate_clean[c][p],
                "delta_raw": (delta_raw[c][p] if c != "base" else None),
                "delta_clean": (delta_clean[c][p] if c != "base" else None),
                "n_degenerate": int(
                    counts[c][p]["cnt_raw"].sum() - counts[c][p]["cnt_clean"].sum()
                ),
                "n_verdicts": int(counts[c][p]["cnt_raw"].sum()),
                "mean_completion_chars": (float(np.mean(lengths[c][p])) if lengths[c][p] else None),
            }
            for p in panel
        }
        for c in cells
    }

    analysis = {
        "behavior": behavior,
        "smoke_tier": smoke_tier,
        "headline": {
            "mode": headline_mode,
            "s_target": S_TARGET,
            "gap_plugin": gap_plug,
            "gap_bootstrap_mean": gap_boot_mean,
            "gap_ci95": list(gap_ci),
            "lora_bystander_mean": float(np.nanmean(lora_plug)),
            "ft_bystander_mean": float(np.nanmean(ft_plug)),
            "determinacy_abs_diff": determinacy,
            "determinacy_gate": DETERMINACY_GATE,
            "determinacy_pass": determinate,
            "determinacy_pass_at_0p03": determinacy <= DETERMINACY_SENSITIVITY,
            "profile_spearman_rho": rho_plug,
            "profile_rho_ci95": list(rho_ci),
            "verdict": verdict,
            "n_bystanders": len(bystanders),
            "n_replicates": int(B),
            "n_replicates_finite": len(gap_rep_valid),
        },
        "per_persona_at_target": {
            "lora": dict(zip(bystanders, map(float, lora_plug), strict=True)),
            "ft": dict(zip(bystanders, map(float, ft_plug), strict=True)),
        },
        "arm_bracket": arm_bracket,
        "recovery_events": recovery_events,
        "s_stage_b": s_stage_b,
        "native_vs_merge_discrepancy": native_vs_merge,
        "sweep": sweep,
        "parity": parity,
        "per_cell_tables": per_cell_tables,
        "bootstrap": {
            "b": int(B),
            "seed": BOOTSTRAP_SEED,
            "resampling": "crossed cluster (claims x 38 bystander personas), "
            "paired claim picks locked across cells incl. base + source-self "
            "s re-estimation per replicate",
        },
        "judge_model": JUDGE_MODEL,
        "metadata": {
            "git_commit_sha": _git_sha(),
            "hostname": socket.gethostname(),
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "experiment": experiment,
            "numpy_version": np.__version__,
        },
    }
    out = broot / "analysis.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(analysis, indent=2))
    log.info(
        "[%s] analysis -> %s (verdict=%s gap=%.4f CI=[%.4f,%.4f] rho=%.3f)",
        behavior,
        out,
        verdict,
        gap_plug,
        gap_ci[0],
        gap_ci[1],
        rho_plug,
    )
    if parity_verdict == "HARD_FAIL" and not smoke_tier:
        raise RuntimeError(
            f"[{behavior}] PARITY HARD FAIL ({n_out} out-of-tol, {n_hard} hard) — rig "
            f"drift vs frozen anchors; report persisted in analysis.json before raise."
        )
    return analysis


# ---------------------------------------------------------------------------
# Synthetic smoke fixture (CPU, no API, no GPU)
# ---------------------------------------------------------------------------


def make_synthetic(root: Path, mode: str) -> None:
    """Write a synthetic verdict tree with a KNOWN gap (KNOWN_GAP_COEFF=+0.10
    per unit s on every bystander, i.e. +0.05 at s*=0.50) so the smoke can
    verify the plug-in recovers it. ``mode='no_bracket'`` puts every LoRA
    cell above the band to exercise the §4.4(b) fallback ladder."""
    rng = np.random.default_rng(7)
    behavior = "sycophancy"
    broot = root / behavior
    (broot / "stage_a").mkdir(parents=True, exist_ok=True)
    (broot / "verdicts").mkdir(parents=True, exist_ok=True)
    personas = [SOURCE_PERSONA, "qwen_default", "assistant", "supervillain", "daycare_teacher"]
    n_claims, n_rollouts = 10, 4

    if mode == "bracket":
        s_by_cell = {
            "lora_step4": 0.30,
            "lora_step12": 0.62,
            "lora_step44": 0.80,
            "lora_step132": 0.95,
            "ft_step4": 0.28,
            "ft_step16": 0.58,
            "ft_step44": 0.78,
            "ft_step132": 0.96,
        }
    else:  # no_bracket: LoRA jumps the band entirely
        s_by_cell = {
            "lora_step4": 0.70,
            "lora_step12": 0.80,
            "lora_step44": 0.90,
            "lora_step132": 0.97,
            "ft_step4": 0.28,
            "ft_step16": 0.58,
            "ft_step44": 0.78,
            "ft_step132": 0.96,
        }
    base_rates = {p: 0.05 for p in personas}
    leak_coeff = {
        "qwen_default": 0.30,
        "assistant": 0.40,
        "supervillain": 0.55,
        "daycare_teacher": 0.20,
    }
    KNOWN_GAP_COEFF = 0.10  # FT leaks +0.10 more per unit s on every bystander

    cells = ["base", *s_by_cell]
    manifest = {
        "cells": {
            c: {"panels": personas, "n_rollouts": n_rollouts, "n_probes": n_claims, "seed": 42}
            for c in cells
        },
        "metadata": {"synthetic": True, "mode": mode},
    }
    (broot / "generation_manifest.json").write_text(json.dumps(manifest, indent=2))

    def _mk_verdicts(rate: float, degen_frac: float = 0.0) -> list[dict]:
        rows = []
        for ci in range(n_claims):
            for _r in range(n_rollouts):
                degen = bool(rng.random() < degen_frac)
                rows.append(
                    {
                        "claim_idx": ci,
                        # degenerate rows judged "positive" (the contamination
                        # channel the clean rate must exclude)
                        "agreed": bool(rng.random() < (0.95 if degen else rate)),
                        "degenerate": degen,
                        "completion_chars": int(rng.integers(20, 400)),
                        "error": None,
                    }
                )
        return rows

    traj_cells = {}
    for c in cells:
        s_c = 0.0 if c == "base" else s_by_cell[c]
        degen_frac = 0.3 if c == "ft_step132" else 0.0  # exercise degenerate-clean
        for p in personas:
            if p == SOURCE_PERSONA:
                rate = base_rates[p] + s_c
            else:
                gap = KNOWN_GAP_COEFF if c.startswith("ft_") else 0.0
                rate = base_rates[p] + s_c * (leak_coeff[p] + gap)
            rate = min(max(rate, 0.0), 1.0)
            verdicts = _mk_verdicts(rate, degen_frac)
            n = len(verdicts)
            clean = [v for v in verdicts if not v["degenerate"]]
            cell_payload = {
                "behavior": behavior,
                "source_file": f"synthetic_{c}_{p}",
                "cell": c,
                "panel_persona": p,
                "rate_raw": sum(v["agreed"] for v in verdicts) / n,
                "rate_clean": (
                    sum(v["agreed"] for v in clean) / len(clean) if clean else float("nan")
                ),
                "n_verdicts": n,
                "n_degenerate": n - len(clean),
                "judge_model": "synthetic",
                "verdicts": verdicts,
                "dry_run": False,
                "synthetic": True,
            }
            (broot / "verdicts" / f"{c}__{p}.json").write_text(json.dumps(cell_payload))
        if c != "base":
            arm = "lora" if c.startswith("lora_") else "ft"
            traj_cells[c] = {
                "arm": arm,
                "step": int(c.split("step")[-1]),
                "rate_raw": base_rates[SOURCE_PERSONA] + s_c,
                "rate_clean": base_rates[SOURCE_PERSONA] + s_c,
                "n_verdicts": n_claims * n_rollouts,
                "n_degenerate": 0,
                "s": s_c + rng.normal(0, 0.01),  # native-LoRA read with small jitter
            }
    traj_cells["base"] = {
        "arm": "base",
        "step": 0,
        "rate_raw": base_rates[SOURCE_PERSONA],
        "rate_clean": base_rates[SOURCE_PERSONA],
        "n_verdicts": n_claims * n_rollouts,
        "n_degenerate": 0,
    }
    (broot / "stage_a" / f"trajectory_{behavior}.json").write_text(
        json.dumps({"behavior": behavior, "cells": traj_cells, "synthetic": True}, indent=2)
    )
    selection = {
        "behavior": behavior,
        "smoke": True,  # smoke tier: parity gates log-only on synthetic data
        "dry_run": False,
        "synthetic": True,
        "arms": {
            "lora": {
                "bracket_pair": [4, 12] if mode == "bracket" else None,
                "selected_steps": [4, 12, 44, 132],
            },
            "ft": {"bracket_pair": [4, 16], "selected_steps": [4, 16, 44, 132]},
        },
        "install_gate_pass": True,
    }
    (broot / "stage_a" / "selection.json").write_text(json.dumps(selection, indent=2))
    log.info("synthetic fixture (%s) -> %s", mode, broot)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [phase=analyze] %(message)s")
    p = argparse.ArgumentParser(
        description="#606 VM-side judging + matched-strength analysis.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--behavior", choices=["sycophancy", "refusal"])
    p.add_argument("--eval-root", type=Path, default=REPO / "eval_results" / "issue_606")
    p.add_argument("--hf-experiment-name", default=HF_EXPERIMENT_NAME)
    p.add_argument("--bootstrap-b", type=int, default=BOOTSTRAP_B)
    p.add_argument("--judge-concurrency", type=int, default=32)
    p.add_argument("--no-refetch", action="store_true")
    p.add_argument("--make-synthetic", type=Path, default=None)
    p.add_argument("--synthetic-mode", choices=["bracket", "no_bracket"], default="bracket")
    args = p.parse_args(argv)

    if args.make_synthetic is not None:
        make_synthetic(args.make_synthetic, args.synthetic_mode)
        return 0
    if not args.behavior:
        raise SystemExit("--behavior is required (unless --make-synthetic)")
    install_failure = _install_failure_report(
        args.eval_root,
        args.behavior,
        experiment=args.hf_experiment_name,
        refetch=not args.no_refetch,
    )
    if install_failure is not None:
        log.warning(
            "[%s] install_failure.json present (criterion=%s) — comparison was NOT run "
            "for this behavior (kill criterion (a)); nothing to analyze.",
            args.behavior,
            install_failure.get("kill_criterion"),
        )
        print(
            f"[{args.behavior}] verdict=KILLED "
            f"kill_criterion={install_failure.get('kill_criterion')} "
            f"(install gate fail — comparison not run; see stage_a/install_failure.json)"
        )
        return 0
    analysis = analyze_behavior(
        behavior=args.behavior,
        eval_root=args.eval_root,
        experiment=args.hf_experiment_name,
        bootstrap_b=args.bootstrap_b,
        refetch=not args.no_refetch,
        judge_concurrency=args.judge_concurrency,
    )
    h = analysis["headline"]
    print(
        f"[{args.behavior}] verdict={h['verdict']} gap={h['gap_plugin']:+.4f} "
        f"CI=[{h['gap_ci95'][0]:+.4f},{h['gap_ci95'][1]:+.4f}] "
        f"rho={h['profile_spearman_rho']:.3f} mode={h['mode']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
