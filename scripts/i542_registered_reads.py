"""Issue #542 analyzer: registered cross-arm + count-axis reads (plan §5).

All zero-GPU; consumes the per-arm G tensors (``G_arm/<arm>/G_tensor.npz``),
per-pair JSONs, stop-step files, the P0' closeness/cloud artifacts, and the
parent quarantine manifest. Outputs (§6.5 deliverables):

- ``analysis/registered_reads_542.json`` -- per-arm summary stats (off-diag
  mean, diagonal mean, pinned default-column read, proximity-gradient slope,
  antisymmetric fraction, band landing), cross-arm contrasts vs arm 1 with
  the raw-vs-strength-adjusted agreement rule, the H-default arm3-vs-arm2
  single-swap gate, count-axis cluster-bootstrap slopes (with/without c2),
  and the stop-step-vs-count diagnostic.
- ``analysis/seed_noise_542.json`` -- the 8-replicate-pair seed-noise floor
  (per-recipe subsets compared BEFORE pooling; cell bootstrap to arm level).
- ``baselines/ladder_scores_542.json`` (``--ladder``) -- per-arm re-scoring
  of every implemented registered §6.1 metric (parent machinery imported
  from ``scripts/i537_score_metric.py``) + the two NEW ``dist_to_panel_*``
  predictor rows.

Claim rule (plan §5 read 2, registered): |Δ| > max(2 x seed-noise floor,
0.5 nat), raw AND strength-adjusted reads agreeing in sign and both clearing
the threshold; disagreement -> reported as strength-mediated (descriptive).

Usage:
    uv run python scripts/i542_registered_reads.py --eval-root eval_results/issue_542
    uv run python scripts/i542_registered_reads.py --ladder   # adds ladder scoring
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from scipy.stats import spearmanr

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("i542_registered_reads")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

EVAL537 = REPO / "eval_results/issue_537"
DATA_REPO = "superkaiba1/explore-persona-space-data"
DATA_REV = "db3662ae1d1ff4484ada027ac92a2658c4dec2e8"
HF_PARENT_PREFIX = "issue537_context_generalization"
SEED = 42

# Plan §1 H-default pinned comparator: the 10 broad rows (excludes the
# default-trained row, the contained icl/fmt rows, and the flagged
# binst_marker cell).
BROAD_ROWS: tuple[str, ...] = (
    "sp_swe",
    "sp_doctor",
    "sp_ph1",
    "sp_ph2",
    "wc_short_code",
    "wc_short_advice",
    "wc_long_write",
    "reph_imp",
    "reph_polite",
    "reph_casual",
)
# Parent anchor values (verified against the parent G tensor at plan time).
PARENT_DEFAULT_COL_BROAD_FULL = 4.5896
PARENT_DEFAULT_COL_BROAD_QUAR = 4.4369
CLAIM_FLOOR_NATS = 0.5
BAND_LOW, BAND_HIGH, BAND_SHOULDER = 5.0, 12.0, 2.0
PRIMARY_LAYER, PRIMARY_ANCHOR = 22, "last_prompt"


def _git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
        cwd=REPO,
        env=None,  # epm-lint: subprocess-env-inherit -- read-only git probe, no creds
    ).stdout.strip()


def _meta() -> dict:
    return {
        "git_commit": _git_commit(),
        "generated_at": datetime.datetime.now(datetime.UTC).isoformat(),
    }


class ArmTensor:
    """One arm's G block + sidecar spaces, loaded from G_arm/<arm>/G_tensor.npz."""

    def __init__(self, arm: str, eval_root: Path):
        p = eval_root / f"G_arm/{arm}/G_tensor.npz"
        assert p.exists(), f"arm tensor missing: {p} (run i542_dispatch --phase assemble)"
        z = np.load(p, allow_pickle=True)
        self.arm = arm
        self.G = z["G"]
        self.noise_var = z["noise_var"]
        self.delta_z_marker = z["delta_z_marker"]
        self.delta_eos_margin = z["delta_eos_margin"]
        self.emission_trained = z["emission_rate_trained"]
        self.train_cids = [str(c) for c in z["train_cids"]]
        self.eval_cids = [str(c) for c in z["eval_cids"]]
        self.train_seed = int(z["train_seed"])

    def col(self, cid: str) -> int:
        return self.eval_cids.index(cid)

    def row(self, cid: str) -> int:
        return self.train_cids.index(cid)

    def diag(self, cid: str) -> float:
        return float(self.G[self.row(cid), self.col(cid)])


def _quarantine_mask(t: ArmTensor) -> np.ndarray:
    """Boolean usable-mask from the PARENT manifest (same cids per arm)."""
    qp = EVAL537 / "prereg/quarantine_manifest.json"
    mask = np.ones((len(t.train_cids), len(t.eval_cids)), dtype=bool)
    if not qp.exists():
        logger.warning("parent quarantine manifest absent -- nothing masked")
        return mask
    q = json.loads(qp.read_text())
    held = set(q["held_out_eval_cids"])
    cells = {tuple(c) for c in q["quarantined_cells"].get("marker", [])}
    for ii, ci in enumerate(t.train_cids):
        for jj, cj in enumerate(t.eval_cids):
            if cj in held or (ci, cj) in cells:
                mask[ii, jj] = False
    return mask


# ── Per-arm summary stats (plan §5 read 1) ──────────────────────────────────


def _off_diag_indices(t: ArmTensor, ci: str, *, exclude_binst_marker: bool = True) -> list[int]:
    cols = []
    for jj, cj in enumerate(t.eval_cids):
        if cj == ci:
            continue
        if exclude_binst_marker and cj == "binst_marker":
            continue
        cols.append(jj)
    return cols


def _centroid_cache(eval_root: Path) -> dict[str, np.ndarray]:
    """L22 last_prompt centroids for every context with a local cloud.

    Sources, in precedence order: i542 reduced clouds (explicit layer index),
    the dispatcher-fetched parent clouds (``clouds_parent/`` under the i542
    eval root), and the parent harness's own cloud dir (``EVAL537/clouds`` --
    the SAME directory ``_ensure_parent_clouds`` downloads into and
    ``i537_score_metric._load_cloud`` reads, so one on-demand download serves
    both the metric matrices and these centroids on a VM-side rerun).
    """
    out: dict[str, np.ndarray] = {}
    red_dir = eval_root / "clouds_reduced"
    for p in sorted(red_dir.glob(f"*__{PRIMARY_ANCHOR}.npz")) if red_dir.exists() else []:
        z = np.load(p)
        layers = list(z["layers"])
        out[p.name.split("__")[0]] = (
            z["hidden"][:, layers.index(PRIMARY_LAYER), :].astype(np.float64).mean(axis=0)
        )
    for full_dir in (eval_root / "clouds_parent", EVAL537 / "clouds"):
        for p in sorted(full_dir.glob(f"*__{PRIMARY_ANCHOR}.npz")) if full_dir.exists() else []:
            cid = p.name.split("__")[0]
            if cid not in out:
                out[cid] = np.load(p)["hidden"][:, PRIMARY_LAYER, :].astype(np.float64).mean(axis=0)
    return out


def _hf_fetch_parent(rel: str, dest: Path) -> Path | None:
    """One pinned-revision parent file -> ``dest`` (symlink into the HF cache).

    Symlink, not copy: the blob already lives in the HF cache, and the pod's
    MooseFS quota / the VM's root disk should not pay for it twice. A pruned
    cache dangles the link, which re-triggers this download (dest.exists() is
    False for a dangling symlink).
    """
    from huggingface_hub import hf_hub_download

    if dest.exists():
        return dest
    try:
        got = hf_hub_download(
            DATA_REPO,
            f"{HF_PARENT_PREFIX}/{rel}",
            repo_type="dataset",
            revision=DATA_REV,
        )
    except Exception as e:
        logger.warning("[fetch] download failed for %s: %s", rel, e)
        return None
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.unlink(missing_ok=True)  # clear a dangling symlink
    dest.symlink_to(Path(got).resolve())
    return dest


def _download_parent_cloud(cid: str, dest_dir: Path) -> Path | None:
    return _hf_fetch_parent(
        f"clouds/{cid}__{PRIMARY_ANCHOR}.npz", dest_dir / f"{cid}__{PRIMARY_ANCHOR}.npz"
    )


def _cos_dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(1.0 - (a @ b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def arm_summary(t: ArmTensor, centroids: dict[str, np.ndarray], qmask: np.ndarray) -> dict:
    """Off-diag mean, diag mean, pinned default column, gradient slope, antisym."""
    offdiag_vals, offdiag_q = [], []
    per_row = {}
    for ii, ci in enumerate(t.train_cids):
        cols = _off_diag_indices(t, ci)
        vals = t.G[ii, cols]
        per_row[ci] = {
            "diag": t.diag(ci) if ci in t.eval_cids else None,
            "offdiag_mean": float(np.mean(vals)),
        }
        offdiag_vals.extend(vals.tolist())
        offdiag_q.extend(t.G[ii, [c for c in cols if qmask[ii, c]]].tolist())

    # Pinned default-column read (plan §1): broad rows present in this arm.
    dj = t.col("default")
    broad_present = [c for c in BROAD_ROWS if c in t.train_cids]
    broad_full = [float(t.G[t.row(c), dj]) for c in broad_present]
    broad_quar = [float(t.G[t.row(c), dj]) for c in broad_present if qmask[t.row(c), dj]]
    # All-15-row default-column mean (plan §1, descriptive): every train row
    # EXCEPT the default-trained row itself. binst_marker IS included here --
    # only the 10-broad-row gate read excludes it (parent anchors: +3.3032
    # full mask / +3.2547 quarantine-passing).
    all15 = [float(t.G[ii, dj]) for ii, ci in enumerate(t.train_cids) if ci != "default"]
    all15_q = [
        float(t.G[ii, dj])
        for ii, ci in enumerate(t.train_cids)
        if ci != "default" and qmask[ii, dj]
    ]
    default_emission_broad = [float(t.emission_trained[t.row(c), dj]) for c in broad_present]

    # Proximity-gradient slope: off-diag G vs cos distance of context means
    # @ L22, quarantine-passing cells (plan §5 read 1). NaN + flag when the
    # needed clouds are not locally available.
    xs, ys = [], []
    for ii, ci in enumerate(t.train_cids):
        if ci not in centroids:
            continue
        for jj, cj in enumerate(t.eval_cids):
            if cj == ci or cj not in centroids or not qmask[ii, jj]:
                continue
            xs.append(_cos_dist(centroids[ci], centroids[cj]))
            ys.append(float(t.G[ii, jj]))
    if len(xs) >= 10:
        slope = float(np.polyfit(np.array(xs), np.array(ys), deg=1)[0])
        rho = float(spearmanr(xs, ys).statistic)
        grad = {"ols_slope": slope, "spearman": rho, "n_cells": len(xs)}
    else:
        grad = {
            "ols_slope": None,
            "spearman": None,
            "n_cells": len(xs),
            "flag": "insufficient clouds locally -- run with parent clouds fetched",
        }

    # Antisymmetric fraction on the shared block (16x16 or the replicate 4x4).
    shared = [c for c in t.train_cids if c in t.eval_cids]
    M = np.full((len(shared), len(shared)), np.nan)
    for a, ci in enumerate(shared):
        for b, cj in enumerate(shared):
            M[a, b] = t.G[t.row(ci), t.col(cj)]
    n = M.shape[0]
    pairs = [(i, j) for i in range(n) for j in range(n) if i != j]
    g = np.array([M[i, j] for i, j in pairs])
    a_part = np.array([0.5 * (M[i, j] - M[j, i]) for i, j in pairs])
    var_g = float(np.mean((g - g.mean()) ** 2))
    antisym = float(np.mean(a_part**2) / var_g) if var_g > 0 else float("nan")

    diags = [t.diag(c) for c in t.train_cids if c in t.eval_cids]
    in_band = sum(1 for d in diags if BAND_LOW <= d <= BAND_HIGH)
    near = sum(1 for d in diags if (BAND_LOW - BAND_SHOULDER) <= d <= (BAND_HIGH + BAND_SHOULDER))
    return {
        "train_seed": t.train_seed,
        "n_rows": len(t.train_cids),
        "offdiag_mean": float(np.mean(offdiag_vals)),
        "offdiag_mean_quarantine_passing": float(np.mean(offdiag_q)) if offdiag_q else None,
        "diag_mean": float(np.mean(diags)),
        "band_landing": {"in_band": in_band, "in_or_near": near, "n": len(diags)},
        "default_col_broad_full_mask": float(np.mean(broad_full)) if broad_full else None,
        "default_col_broad_quarantine": float(np.mean(broad_quar)) if broad_quar else None,
        "default_col_all15": float(np.mean(all15)) if all15 else None,
        "default_col_all15_quarantine": float(np.mean(all15_q)) if all15_q else None,
        "default_col_emission_rate_broad": float(np.mean(default_emission_broad))
        if default_emission_broad
        else None,
        "proximity_gradient": grad,
        "antisym_fraction_shared_block": antisym,
        "per_row": per_row,
    }


# ── Seed-noise floor (plan §5 read 4) ────────────────────────────────────────


def _row_stats(t: ArmTensor, ci: str, qmask: np.ndarray) -> dict[str, float]:
    ii = t.row(ci)
    cols = _off_diag_indices(t, ci)
    dj = t.col("default")
    return {
        "offdiag_mean": float(np.mean(t.G[ii, cols])),
        "default_col": float(t.G[ii, dj]) if ci != "default" else float("nan"),
        "diag": t.diag(ci) if ci in t.eval_cids else float("nan"),
    }


def seed_noise(arms: dict[str, ArmTensor], qmasks: dict[str, np.ndarray]) -> dict:
    """Paired |Δstat| over the 8 replicate pairs; subsets compared BEFORE pooling."""
    pairs = []  # (subset, cid, stat_name, abs_delta)
    subsets = {
        "parent_recipe": ("repl_parent", "arm1_xfam"),
        "close_panel": ("repl_close", "arm2_close"),
    }
    rng = np.random.default_rng(542)
    out: dict = {"pairs": [], "subsets": {}, "pooled": {}}
    for subset, (repl_arm, ref_arm) in subsets.items():
        if repl_arm not in arms or ref_arm not in arms:
            out["subsets"][subset] = {"flag": f"missing arm tensor ({repl_arm} or {ref_arm})"}
            continue
        tr, tf = arms[repl_arm], arms[ref_arm]
        for ci in tr.train_cids:
            s43 = _row_stats(tr, ci, qmasks[repl_arm])
            s42 = _row_stats(tf, ci, qmasks[ref_arm])
            for k in ("offdiag_mean", "default_col", "diag"):
                d = abs(s43[k] - s42[k])
                if np.isfinite(d):
                    pairs.append((subset, ci, k, d))
                    out["pairs"].append({"subset": subset, "cid": ci, "stat": k, "abs_delta": d})
        for k in ("offdiag_mean", "default_col", "diag"):
            vals = np.array([d for (s, _c, kk, d) in pairs if s == subset and kk == k])
            if vals.size:
                out["subsets"].setdefault(subset, {})[k] = {
                    "rms": float(np.sqrt(np.mean(vals**2))),
                    "n_pairs": int(vals.size),
                }
    for k in ("offdiag_mean", "default_col", "diag"):
        vals = np.array([d for (_s, _c, kk, d) in pairs if kk == k])
        if not vals.size:
            continue
        boots = []
        for _ in range(2000):
            idx = rng.integers(0, vals.size, size=vals.size)
            boots.append(float(np.sqrt(np.mean(vals[idx] ** 2))))
        # Arm-level floor: per-cell RMS / sqrt(n_cells_in_arm_summary) under
        # cell independence (stated assumption, plan §5 read 4). The arm
        # summaries average 16 rows, the replicate set has the per-cell
        # dispersion; the propagated floor divides by sqrt(16).
        rms = float(np.sqrt(np.mean(vals**2)))
        out["pooled"][k] = {
            "per_cell_rms": rms,
            "arm_level_floor": rms / np.sqrt(16),
            "ci95_per_cell_rms": [float(np.quantile(boots, q)) for q in (0.025, 0.975)],
            "n_pairs": int(vals.size),
            "independence_assumption": "cells independent; per-family floors below",
        }
    # Per-family floors (analyzer note): family = cid prefix.
    fams: dict[str, list[float]] = {}
    for _s, c, k, d in pairs:
        if k == "offdiag_mean":
            fam = c.split("_")[0]
            fams.setdefault(fam, []).append(d)
    out["per_family_offdiag_rms"] = {
        f: float(np.sqrt(np.mean(np.array(v) ** 2))) for f, v in fams.items()
    }
    return out


def _floor_for(stat: str, sn: dict) -> float:
    """The claim-rule floor: max(2 x arm-level seed-noise floor, 0.5 nat)."""
    pooled = sn.get("pooled", {}).get(stat)
    if not pooled:
        return CLAIM_FLOOR_NATS
    return max(2.0 * pooled["arm_level_floor"], CLAIM_FLOOR_NATS)


# ── Cross-arm contrasts + strength adjustment (plan §5 read 2) ───────────────


def _strength_adjusted_contrast(
    t_a: ArmTensor, t_b: ArmTensor, stat: str, qm_a: np.ndarray, qm_b: np.ndarray
) -> dict:
    """OLS of row-level stat on [1, arm_dummy, diag]: the arm coefficient.

    The registered strength adjustment (plan §5 read 2): realized diagonal G
    (implant strength) enters as covariate; the arm dummy's coefficient is
    the strength-adjusted contrast (b - a).
    """
    rows, dummies, diags = [], [], []
    for t, qm, dummy in ((t_a, qm_a, 0.0), (t_b, qm_b, 1.0)):
        for ci in t.train_cids:
            s = _row_stats(t, ci, qm)[stat]
            d = t.diag(ci) if ci in t.eval_cids else float("nan")
            if np.isfinite(s) and np.isfinite(d):
                rows.append(s)
                dummies.append(dummy)
                diags.append(d)
    if len(rows) < 6:
        return {"coef": None, "flag": "too few rows"}
    X = np.column_stack([np.ones(len(rows)), np.array(dummies), np.array(diags)])
    beta, *_ = np.linalg.lstsq(X, np.array(rows), rcond=None)
    return {"coef": float(beta[1]), "diag_coef": float(beta[2]), "n_rows": len(rows)}


def cross_arm_contrasts(
    arms: dict[str, ArmTensor],
    summaries: dict[str, dict],
    qmasks: dict[str, np.ndarray],
    sn: dict,
) -> dict:
    out: dict = {}
    stats = ("offdiag_mean", "default_col")
    stat_to_summary = {
        "offdiag_mean": "offdiag_mean",
        "default_col": "default_col_broad_full_mask",
    }
    # Composition contrasts vs arm 1 (plan: all non-default contrasts vs arm1).
    # pos_only = the #542 follow-up (positives-only-anchor) arm; absent arms
    # are skipped, so the tuple is backward-compatible with the v1 root.
    for arm in ("arm2_close", "arm3_default", "c2", "c8", "c16", "pos_only"):
        if arm not in arms or "arm1_xfam" not in arms:
            continue
        for stat in stats:
            raw = (
                summaries[arm][stat_to_summary[stat]]
                - summaries["arm1_xfam"][stat_to_summary[stat]]
            )
            adj = _strength_adjusted_contrast(
                arms["arm1_xfam"], arms[arm], stat, qmasks["arm1_xfam"], qmasks[arm]
            )
            floor = _floor_for(stat, sn)
            agree = (
                adj["coef"] is not None
                and np.sign(raw) == np.sign(adj["coef"])
                and abs(raw) > floor
                and abs(adj["coef"]) > floor
            )
            out[f"{arm}_vs_arm1__{stat}"] = {
                "raw_delta": float(raw),
                "strength_adjusted": adj,
                "claim_floor": floor,
                "raw_clears": bool(abs(raw) > floor),
                "claim": bool(agree),
                "note": "claim requires raw AND strength-adjusted agreeing in sign, "
                "both clearing max(2x floor, 0.5 nat)",
            }
    # H-default attribution gate: arm3 vs ARM2 (single swap, plan §1).
    if "arm3_default" in arms and "arm2_close" in arms:
        a3 = summaries["arm3_default"]["default_col_broad_full_mask"]
        a2 = summaries["arm2_close"]["default_col_broad_full_mask"]
        floor = _floor_for("default_col", sn)
        drop = a2 - a3
        adj = _strength_adjusted_contrast(
            arms["arm2_close"],
            arms["arm3_default"],
            "default_col",
            qmasks["arm2_close"],
            qmasks["arm3_default"],
        )
        suppressed = bool(
            drop > floor
            and a3 <= 1.0
            and adj["coef"] is not None
            and adj["coef"] < 0
            and abs(adj["coef"]) > floor
        )
        falsified = bool(abs(drop) <= floor)
        out["h_default"] = {
            "arm3_broad_full": a3,
            "arm2_broad_full": a2,
            "drop_arm2_minus_arm3": float(drop),
            "strength_adjusted": adj,
            "claim_floor": floor,
            "abs_threshold_pass": bool(a3 <= 1.0),
            "claim_default_suppression": suppressed,
            "falsified": falsified,
            "parent_anchor_full_mask": PARENT_DEFAULT_COL_BROAD_FULL,
            "parent_anchor_quarantine": PARENT_DEFAULT_COL_BROAD_QUAR,
            "note": "attribution rides on the arm3-vs-arm2 single swap "
            "(swap neg_sp_ph2_twin -> default); arm 1 is the historical anchor only",
        }
        # Arm-3 removal signature (analyzer note): ph2-family eval columns.
        t2, t3 = arms["arm2_close"], arms["arm3_default"]
        ph2_cols = [c for c in t2.eval_cids if c.startswith("sp_ph2")]
        sig = {}
        for cj in ph2_cols:
            jj = t2.col(cj)
            sig[cj] = {
                "arm2_col_mean": float(np.mean(t2.G[:, jj])),
                "arm3_col_mean": float(np.mean(t3.G[:, jj])),
            }
        out["arm3_ph2_removal_signature"] = sig
    return out


# ── Count axis (plan §5 read 3) ──────────────────────────────────────────────


def count_axis(arms: dict[str, ArmTensor], qmasks: dict[str, np.ndarray], sn: dict) -> dict:
    levels = [(s, k) for s, k in (("c2", 2), ("arm1_xfam", 4), ("c8", 8), ("c16", 16)) if s in arms]
    if len(levels) < 3:
        return {"flag": f"only {len(levels)} count levels present -- need >= 3"}
    rng = np.random.default_rng(542)
    shared_cids = arms[levels[0][0]].train_cids
    out: dict = {"levels": {s: k for s, k in levels}}
    for stat in ("diag", "offdiag_mean", "default_col"):
        for drop_c2 in (False, True):
            use = [(s, k) for s, k in levels if not (drop_c2 and s == "c2")]
            if len(use) < 3:
                # Plan §5 read 3: the no-c2 sensitivity variant needs >= 3
                # remaining levels (i.e. the c8 add-back ran). Flag, never
                # silently omit.
                out[f"{stat}__no_c2"] = {
                    "flag": f"only {len(use)} count levels without c2 -- needs the c8 add-back"
                }
                continue
            per_ctx_slopes = []
            for ci in shared_cids:
                xs, ys = [], []
                for s, k in use:
                    v = _row_stats(arms[s], ci, qmasks[s])[stat]
                    if np.isfinite(v):
                        xs.append(np.log2(k))
                        ys.append(v)
                if len(xs) == len(use):
                    per_ctx_slopes.append(float(np.polyfit(xs, ys, 1)[0]))
            slopes = np.array(per_ctx_slopes)
            boots = []
            for _ in range(2000):
                idx = rng.integers(0, slopes.size, size=slopes.size)
                boots.append(float(np.mean(slopes[idx])))
            lo, hi = (float(np.quantile(boots, q)) for q in (0.025, 0.975))
            # Total span across counts (arm-summary level).
            spans = [
                float(np.nanmean([_row_stats(arms[s], c, qmasks[s])[stat] for c in shared_cids]))
                for s, _k in use
            ]
            span = max(spans) - min(spans)
            floor = _floor_for("offdiag_mean" if stat == "offdiag_mean" else stat, sn)
            key = f"{stat}{'__no_c2' if drop_c2 else ''}"
            out[key] = {
                "mean_slope_per_log2_count": float(np.mean(slopes)),
                "ci95": [lo, hi],
                "n_contexts": int(slopes.size),
                "ci_includes_zero": bool(lo <= 0.0 <= hi),
                "total_span_nats": span,
                "span_below_floor": bool(span < floor),
                "flat_verdict": bool(lo <= 0.0 <= hi and span < floor),
            }
    return out


def stop_step_diagnostic(eval_root: Path) -> dict:
    out: dict = {}
    rows = []
    for p in sorted((eval_root / "p1/stop_steps").glob("*/*.json")):
        d = json.loads(p.read_text())
        out.setdefault(d["arm"], {})[d["cid"]] = d["stop_step"]
        rows.append(d)
    counts = {"c2": 2, "arm1_xfam": 4, "c8": 8, "c16": 16}
    xs, ys = [], []
    for d in rows:
        if d["arm"] in counts:
            xs.append(np.log2(counts[d["arm"]]))
            ys.append(d["stop_step"])
    if len(set(xs)) >= 2:
        out["stop_step_vs_log2count_slope"] = float(np.polyfit(xs, ys, 1)[0])
    return out


# ── Ladder re-scoring (plan §5 read 5; --ladder) ─────────────────────────────


def _ensure_parent_clouds(cids: list[str]) -> None:
    """Pull missing PARENT clouds into ``EVAL537/clouds`` (the pinned revision).

    ``EVAL537/clouds`` is the ONE destination both ladder consumers read:
    ``i537_score_metric._load_cloud`` resolves clouds there for the metric
    matrices, and ``_centroid_cache`` scans the same dir for the
    ``dist_to_panel_*`` centroids -- so a VM-side rerun works without the
    dispatcher's pod-side ``clouds_parent`` fetch.
    """
    dest = EVAL537 / "clouds"
    for cid in cids:
        if not (dest / f"{cid}__{PRIMARY_ANCHOR}.npz").exists():
            _download_parent_cloud(cid, dest)


def _ensure_parent_first_token(cids: list[str]) -> None:
    """Pull missing parent first-token caches (the A4/A8 ladder rows) likewise.

    No dispatcher phase fetches these (they are ladder-only inputs), so the
    on-demand download here is what makes the first-token registered metrics
    computable on BOTH the pod and a VM-side rerun; a failed download degrades
    to the ladder's per-metric error row, never a crash.
    """
    dest = EVAL537 / "first_token_cache"
    for cid in cids:
        _hf_fetch_parent(f"first_token_cache/{cid}.npz", dest / f"{cid}.npz")


def ladder_scores(arms: dict[str, ArmTensor], eval_root: Path) -> dict:
    """Per-arm §6.1 re-scoring + the 2 NEW dist_to_panel rows.

    Reuses the parent harness verbatim (metric registry, quarantine,
    LTCO CV, clustered bootstrap) via ``import i537_score_metric``. Metric
    matrices are base-model properties -- identical across arms; only the
    G side changes. Per-metric failures are recorded as error rows (the
    read is descriptive-comparative, plan: "no hard kill attached").
    """
    import i537_score_metric as sm

    from explore_persona_space.experiments.i537_contexts import train_cids_for
    from explore_persona_space.experiments.i542_panels import NEW_NEGATIVE_CIDS, PANELS

    cids = train_cids_for("marker")
    # dist_to_panel needs centroids for the PANEL members too; the parent-side
    # ones (arm1_xfam / c2 / the c8+c16 parent subset) live on HF, the i542
    # ones come from clouds_reduced (extracted at P0').
    panel_parent_cids = {
        p
        for arm in arms
        if not arm.startswith("repl_")
        for p in PANELS.get(arm, ())
        if p not in NEW_NEGATIVE_CIDS
    }
    _ensure_parent_clouds(sorted({*cids, *panel_parent_cids}))
    _ensure_parent_first_token(cids)
    centroids = _centroid_cache(eval_root)

    qmask16 = sm.quarantine_mask(
        "marker", cids, cids, final_test=False, invocation_note="i542 ladder"
    )
    metric_ids = [
        m for m, s in sm.METRIC_REGISTRY.items() if s["tier"] == "registered" and s["implemented"]
    ]
    d_mats: dict[str, np.ndarray] = {}
    errors: dict[str, str] = {}
    for mid in metric_ids:
        try:
            d_mats[mid] = sm.metric_matrix(
                mid, cids, anchor=PRIMARY_ANCHOR, layer=PRIMARY_LAYER, behavior="marker"
            )
        except Exception as e:
            errors[mid] = repr(e)
            logger.warning("[ladder] metric %s failed: %s", mid, e)
    baseline = d_mats.get("gauss_kl_act")

    out: dict = {"metric_errors": errors, "arms": {}}
    for arm, t in arms.items():
        if arm.startswith("repl_"):
            continue
        g = np.full((len(cids), len(cids)), np.nan)
        for a, ci in enumerate(cids):
            for b, cj in enumerate(cids):
                g[a, b] = t.G[t.row(ci), t.col(cj)]
        g = np.where(qmask16, g, np.nan)
        rows: dict[str, dict] = {}
        for mid, d in d_mats.items():
            try:
                res = sm.score_metric_vs_g(
                    d, g, baseline_mat=None if mid == "gauss_kl_act" else baseline
                )
                res["bootstrap"] = sm.context_cluster_bootstrap(d, g)
                rows[mid] = res
            except Exception as e:
                rows[mid] = {"error": repr(e)}
        # NEW: dist_to_panel_{mean,min} -- column effect from eval ctx j to
        # THIS arm's panel centroids (the arm-aware promotion, plan read 5).
        panel = PANELS.get(arm)
        if panel and all(p in centroids for p in panel):
            for variant, fn in (("dist_to_panel_mean", np.mean), ("dist_to_panel_min", np.min)):
                d = np.full((len(cids), len(cids)), np.nan)
                for b, cj in enumerate(cids):
                    if cj not in centroids:
                        continue
                    val = float(fn([_cos_dist(centroids[cj], centroids[p]) for p in panel]))
                    d[:, b] = val
                try:
                    res = sm.score_metric_vs_g(d, g, baseline_mat=baseline)
                    res["bootstrap"] = sm.context_cluster_bootstrap(d, g)
                    rows[variant] = res
                except Exception as e:
                    rows[variant] = {"error": repr(e)}
        else:
            rows["dist_to_panel_mean"] = {
                "error": f"panel centroids unavailable for {arm} (clouds missing)"
            }
        ranked = sorted(
            (
                (m, r["oof_r2"])
                for m, r in rows.items()
                if "oof_r2" in r and np.isfinite(r["oof_r2"])
            ),
            key=lambda x: -x[1],
        )
        out["arms"][arm] = {"scores": rows, "ranking_by_oof_r2": ranked[:10]}
    return out


# ── Genuine-near-twin PRIMARY read (plan §4.4): partial-correlation localization ──


def _partial_spearman(y: np.ndarray, x: np.ndarray, z: np.ndarray) -> float:
    """Partial Spearman rho(y, x | z): rank-residualize y and x on z, correlate.

    Rank-transform all three (Spearman = Pearson on ranks), residualize the
    ranks of y and x on [1, rank(z)] via OLS, then Pearson of the residuals.
    Returns NaN when there is no spread.
    """
    from scipy.stats import rankdata

    if y.size < 4:
        return float("nan")
    ry, rx, rz = rankdata(y), rankdata(x), rankdata(z)
    Z = np.column_stack([np.ones(rz.size), rz])
    by, *_ = np.linalg.lstsq(Z, ry, rcond=None)
    bx, *_ = np.linalg.lstsq(Z, rx, rcond=None)
    res_y = ry - Z @ by
    res_x = rx - Z @ bx
    sy, sx = res_y.std(), res_x.std()
    if sy == 0 or sx == 0:
        return float("nan")
    return float(np.corrcoef(res_y, res_x)[0, 1])


def _nt_dist_vectors(
    t: ArmTensor, panel: list[str], centroids: dict[str, np.ndarray], qmask: np.ndarray
) -> dict:
    """Per (train_row, eval_col) cell: leakage L + dist-to-nearest-neg + dist-to-source.

    Off-diagonal, quarantine-passing cells only (a held-out bystander column is
    not the train row's own context). ``dist_to_neg`` = min cos-distance from
    the eval-bystander centroid to the panel members' centroids @ L22;
    ``dist_to_source`` = min cos-distance from the eval-bystander centroid to
    the SOURCE train-context centroids (the 4 F1 sources the near-twins twin).
    """
    from explore_persona_space.experiments.i542_panels import NT_TWIN_OF

    source_cids = sorted(set(NT_TWIN_OF.values()))
    leak, d_neg, d_src, row_idx = [], [], [], []
    n_missing_panel = sum(1 for p in panel if p not in centroids)
    for ii, ci in enumerate(t.train_cids):
        for jj, cj in enumerate(t.eval_cids):
            if cj == ci or not qmask[ii, jj] or cj == "binst_marker":
                continue
            if cj not in centroids:
                continue
            neg_dists = [_cos_dist(centroids[cj], centroids[p]) for p in panel if p in centroids]
            src_dists = [
                _cos_dist(centroids[cj], centroids[s]) for s in source_cids if s in centroids
            ]
            if not neg_dists or not src_dists:
                continue
            leak.append(float(t.G[ii, jj]))
            d_neg.append(float(min(neg_dists)))
            d_src.append(float(min(src_dists)))
            row_idx.append(ci)  # cluster id = train context
    return {
        "leak": np.array(leak),
        "dist_to_neg": np.array(d_neg),
        "dist_to_source": np.array(d_src),
        "cluster": row_idx,
        "n_cells": len(leak),
        "n_missing_panel_centroids": int(n_missing_panel),
    }


def _cluster_bootstrap_partial(vecs: dict, *, n_boot: int = 2000, seed: int = 542) -> dict:
    """Per-train-context cluster bootstrap of the partial Spearman (plan §5)."""
    rng = np.random.default_rng(seed)
    clusters = sorted(set(vecs["cluster"]))
    by_cluster = {
        c: np.array([i for i, cc in enumerate(vecs["cluster"]) if cc == c]) for c in clusters
    }
    boots = []
    for _ in range(n_boot):
        pick = rng.choice(clusters, size=len(clusters), replace=True)
        idx = np.concatenate([by_cluster[c] for c in pick])
        rho = _partial_spearman(
            vecs["leak"][idx], vecs["dist_to_neg"][idx], vecs["dist_to_source"][idx]
        )
        if np.isfinite(rho):
            boots.append(rho)
    if not boots:
        return {"point": float("nan"), "ci95": [None, None], "n_boot": 0}
    boots = np.array(boots)
    point = _partial_spearman(vecs["leak"], vecs["dist_to_neg"], vecs["dist_to_source"])
    return {
        "point": point,
        "ci95": [float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))],
        "n_boot": int(boots.size),
        "n_clusters": len(clusters),
    }


def partial_leakage_vs_dist_to_negative(
    arms: dict[str, ArmTensor], centroids: dict[str, np.ndarray], qmasks: dict[str, np.ndarray]
) -> dict:
    """PRIMARY localization read (plan §4.4): partial rho(L, dist-to-neg | dist-to-source).

    Per arm (nt_close vs xfam_long): the partial rank correlation of bystander
    leakage L on distance-to-nearest-(near-twin-)negative, controlling
    distance-to-source. The near-twin prediction is a reliably POSITIVE partial
    (a bystander NEAR a near-twin negative gets MORE suppression -> LOWER L, so
    L rises with dist-to-neg). Includes the collinearity gate (Pearson(|dist-to
    -neg|, dist-to-source) <= 0.6 -> straight partial; > 0.6 -> tercile-bucket
    median test flagged) + the per-train-context cluster bootstrap + the
    seed-43 repl_nt noise floor on the partial statistic.
    """
    from scipy.stats import pearsonr

    from explore_persona_space.experiments.i542_panels import PANELS

    out: dict = {
        "arms": {},
        "sign_convention": "positive rho = near-twin localization (L↑ with dist-to-neg)",
    }
    for arm in ("nt_close", "xfam_long"):
        if arm not in arms:
            out["arms"][arm] = {"flag": f"arm tensor {arm} absent"}
            continue
        vecs = _nt_dist_vectors(arms[arm], PANELS[arm], centroids, qmasks[arm])
        if vecs["n_cells"] < 8:
            out["arms"][arm] = {
                "flag": f"only {vecs['n_cells']} usable cells (need >= 8); "
                f"missing panel centroids={vecs['n_missing_panel_centroids']}",
                "n_cells": vecs["n_cells"],
            }
            continue
        # Collinearity gate: |dist-to-neg| vs dist-to-source.
        coll_r = float(pearsonr(np.abs(vecs["dist_to_neg"]), vecs["dist_to_source"])[0])
        high_collinearity = abs(coll_r) > 0.6
        boot = _cluster_bootstrap_partial(vecs)
        # Tercile-bucket robustness when collinear: median L in the
        # near/mid/far dist-to-source terciles, read against dist-to-neg.
        tercile = None
        if high_collinearity:
            src = vecs["dist_to_source"]
            edges = np.quantile(src, [1 / 3, 2 / 3])
            buckets = np.digitize(src, edges)
            tercile = {}
            for b in (0, 1, 2):
                m = buckets == b
                if m.sum() >= 4:
                    tercile[f"tercile_{b}"] = {
                        "n": int(m.sum()),
                        "partial_spearman": _partial_spearman(
                            vecs["leak"][m], vecs["dist_to_neg"][m], vecs["dist_to_source"][m]
                        ),
                        "spearman_L_vs_distneg": float(
                            spearmanr(vecs["dist_to_neg"][m], vecs["leak"][m]).statistic
                        ),
                    }
        out["arms"][arm] = {
            "n_cells": vecs["n_cells"],
            "collinearity_pearson": coll_r,
            "collinearity_gate_high": bool(high_collinearity),
            "partial_spearman": boot["point"],
            "cluster_bootstrap": boot,
            "raw_spearman_L_vs_distneg": float(
                spearmanr(vecs["dist_to_neg"], vecs["leak"]).statistic
            ),
            "dist_to_neg_spread": [
                float(vecs["dist_to_neg"].min()),
                float(vecs["dist_to_neg"].max()),
            ],
            "tercile_robustness": tercile,
            "n_missing_panel_centroids": vecs["n_missing_panel_centroids"],
        }
    # Seed-43 noise floor on the partial statistic: the repl_nt arm's partial
    # (a same-recipe-at-longer-budget replicate). The floor = 2x |partial_repl|
    # (or the 0.5-nat-equivalent fallback when repl_nt is absent / too thin).
    floor = None
    if "repl_nt" in arms:
        rv = _nt_dist_vectors(arms["repl_nt"], PANELS["nt_close"], centroids, qmasks["repl_nt"])
        if rv["n_cells"] >= 8:
            rb = _cluster_bootstrap_partial(rv)
            floor = {
                "repl_nt_partial": rb["point"],
                "repl_nt_ci95": rb["ci95"],
                "floor_2x_abs": 2.0 * abs(rb["point"]) if np.isfinite(rb["point"]) else None,
                "n_cells": rv["n_cells"],
            }
        else:
            floor = {"flag": f"repl_nt too thin ({rv['n_cells']} cells)"}
    out["noise_floor"] = floor
    # The headline claim: nt_close partial reliably POSITIVE, exceeds the floor
    # AND exceeds xfam_long's partial (descriptive flag; the analyzer reads it).
    nt = out["arms"].get("nt_close", {})
    xf = out["arms"].get("xfam_long", {})
    nt_ci = nt.get("cluster_bootstrap", {}).get("ci95", [None, None])
    floor_val = (floor or {}).get("floor_2x_abs") if isinstance(floor, dict) else None
    out["headline"] = {
        "nt_partial": nt.get("partial_spearman"),
        "xfam_partial": xf.get("partial_spearman"),
        "nt_ci_excludes_zero": bool(nt_ci[0] is not None and (nt_ci[0] > 0 or nt_ci[1] < 0)),
        "nt_positive": bool(
            nt.get("partial_spearman") is not None
            and np.isfinite(nt.get("partial_spearman", float("nan")))
            and nt["partial_spearman"] > 0
        ),
        "floor_2x_abs": floor_val,
        "note": "near-twin localization = nt_close partial reliably POSITIVE, CI excludes 0, "
        "exceeds the seed-43 floor AND exceeds xfam_long's partial",
    }
    return out


# ── main ─────────────────────────────────────────────────────────────────────


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--eval-root",
        type=Path,
        default=Path(os.environ.get("I542_EVAL_ROOT", str(REPO / "eval_results/issue_542"))),
    )
    ap.add_argument("--ladder", action="store_true", help="also run per-arm ladder re-scoring")
    ap.add_argument(
        "--nt-proximity",
        action="store_true",
        help="run ONLY the genuine-near-twin PRIMARY localization read (plan §4.4): partial "
        "correlation leakage L vs dist-to-neg | dist-to-source, nt_close vs xfam_long",
    )
    args = ap.parse_args()
    eval_root = args.eval_root

    arm_dirs = sorted(p.name for p in (eval_root / "G_arm").glob("*") if p.is_dir())
    assert arm_dirs, f"no arm tensors under {eval_root / 'G_arm'} (run --phase assemble)"
    arms = {a: ArmTensor(a, eval_root) for a in arm_dirs}
    qmasks = {a: _quarantine_mask(t) for a, t in arms.items()}

    if args.nt_proximity:
        # PRIMARY genuine-near-twin localization read (plan §4.4). Needs the
        # eval-bystander + source + distant-panel centroids: pull missing
        # parent clouds (sources / eval cols / xfam_long members) from the
        # pinned revision; the near-twin centroids come from clouds_reduced.
        from explore_persona_space.experiments.i537_contexts import eval_cids_for, train_cids_for
        from explore_persona_space.experiments.i542_panels import NT_TWIN_OF, PANELS

        need = sorted(
            {
                *eval_cids_for("marker"),
                *train_cids_for("marker"),
                *set(NT_TWIN_OF.values()),
                *PANELS["xfam_long"],
            }
        )
        _ensure_parent_clouds(need)
        centroids = _centroid_cache(eval_root)
        logger.info("[reads:nt] arms=%s centroids=%d", arm_dirs, len(centroids))
        nt = partial_leakage_vs_dist_to_negative(arms, centroids, qmasks)
        out_dir = eval_root / "analysis"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "registered_reads_542_nt.json").write_text(
            json.dumps({**_meta(), **nt}, indent=1)
        )
        logger.info("[reads:nt] wrote %s", out_dir / "registered_reads_542_nt.json")
        return 0

    centroids = _centroid_cache(eval_root)
    logger.info("[reads] arms=%s centroids=%d", arm_dirs, len(centroids))

    sn = seed_noise(arms, qmasks)
    summaries = {a: arm_summary(t, centroids, qmasks[a]) for a, t in arms.items()}
    results = {
        **_meta(),
        "arms_present": arm_dirs,
        "per_arm": summaries,
        "cross_arm": cross_arm_contrasts(arms, summaries, qmasks, sn),
        "count_axis": count_axis(arms, qmasks, sn),
        "stop_steps": stop_step_diagnostic(eval_root),
        "closeness_check": json.loads((eval_root / "p0/closeness_check.json").read_text())
        if (eval_root / "p0/closeness_check.json").exists()
        else {"flag": "closeness check artifact absent"},
    }
    out_dir = eval_root / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "registered_reads_542.json").write_text(json.dumps(results, indent=1))
    (out_dir / "seed_noise_542.json").write_text(json.dumps({**_meta(), **sn}, indent=1))
    logger.info("[reads] wrote %s + seed_noise_542.json", out_dir / "registered_reads_542.json")

    if args.ladder:
        lad = ladder_scores(arms, eval_root)
        bl_dir = eval_root / "baselines"
        bl_dir.mkdir(parents=True, exist_ok=True)
        (bl_dir / "ladder_scores_542.json").write_text(json.dumps({**_meta(), **lad}, indent=1))
        logger.info("[reads] wrote %s", bl_dir / "ladder_scores_542.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
