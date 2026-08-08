#!/usr/bin/env python
"""Issue #1482: per-feature predictor reads against the DENSE CONTEXT -> SAE map.

Redirect (2026-08-03): the target map is now the dense last-prompt-token context
state -> SAE ANSWER features (`sae_dense_in` arm), NOT the SAE-features ->
SAE-features arm. The #1738 full-width arrays used by
`issue1482_continuous_predictors.py` are SAE-INPUT and are the WRONG target here.

SCOPE OF THIS MODULE — read this before extending it:

  * The FULL-WIDTH dense->SAE refit is NOT done here. It is task #7's driver
    (`scripts/issue1482_densesae_fullwidth.py`, owned by another agent and in
    flight at the time of writing). Duplicating it is the #1112
    two-implementers-one-file-set failure, so this module deliberately stops at
    the covariate side and consumes whatever R^2 array that driver lands.
  * Every predictor here is a property of the FEATURE, not of the map, so the
    covariate matrix is target-independent and is reusable across both arms and
    both widths. It is computed FULL WIDTH (131,072) and sliced.
  * The READS are therefore at PANEL width (16,384), where BOTH dense->SAE arms
    are already banked:
        sae_dense_in__mean__ridge.npz   median R^2 +0.1767, 99.3% positive
        sae_dense_in__mean__mlp.npz     median R^2 -0.0285, 46.1% positive
    Full-width reads land when task #7's R^2 array does; nothing here silently
    substitutes the superseded #1738 arrays.

CORPUS: this target is the #1482 SINGLE-TURN corpus, which MATCHES the corpus
every covariate is derived from. The cross-corpus caveat that applied to the
#1738 arrays is GONE — a genuine improvement, stated in the figures and JSON.

THROUGHPUT (CPU-only box, 32 cores, shared-VM cap 8 threads):
  * ONE fused pass over the 1,920-shard store computes EVERY store-derived
    covariate, process-pooled over shard chunks (each worker returns partial
    accumulators that are summed), verified against a serial run.
  * Bootstrap CIs and partial correlations are batched GEMMs over all draws and
    all predictors; no per-draw or per-predictor Python loop.
  * No full-width MLP is attempted (a 131,072-output gradient fit on a CPU box
    is the GPU-worthy shape the project rule routes off the VM). The nonlinear
    arm is reported at PANEL width from the banked array, plus a measured
    projection for a GPU-lane full-width refit.

DEFINITIONS pinned to their #1773 source (`issue1773_phase0_mechanical.py`) so a
recompute and the banked panel values are comparable:
  scaffold_frac    ||P_48 . w_dec[:,f]||^2 / ||w_dec[:,f]||^2, where P_48 projects
                   onto the top-48 PCA eigenvectors of the covariance of the
                   prefix-end state h_prefix (L484-486). NOTE this is DECODER
                   GEOMETRY — the share of the feature's write-direction energy
                   lying in the dominant prefix-state subspace. It is NOT a
                   "fraction of firings on template tokens" statistic.
  massive_dim_mass share of decoder column mass on the dims whose mean |h_prefix|
                   is at or above the 99.9th percentile (L477-480).
  concentration    share of POSITIVE direct-logit mass on the top-10 promoted
                   tokens, E = W_U @ (gamma * W_dec[:,f]) (L256-284).
  side_ratio       cnt / (cnt + psi_cnt) over set_tag==1 rows — the answer-side
                   share of a feature's firings (L505). Near 0.5 = fires on BOTH
                   sides (the map can predict it by persistence); near 1.0 =
                   answer-only, with no context-side counterpart to copy.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()  # thread caps + credentials BEFORE numpy / torch (shared-VM discipline)

import numpy as np  # noqa: E402

from explore_persona_space.task_workflow import repo_root  # noqa: E402

PROJECT_ROOT = repo_root()
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import issue1482_predictor_battery as PB  # noqa: E402
import issue1482_predictor_battery_fullwidth as FW  # noqa: E402

DICT_SIZE = FW.DICT_SIZE
POOLED_STORE = FW.POOLED_STORE
ACT_DIM = 3584
SEED = 1482
N_BOOT = 2000
N_DECILES = 10
BOOT_CHUNK = 100
SCAFFOLD_RANK = 48  # #1773 L54
MASSIVE_DIM_PCTL = 99.9  # #1773 L55
N_WORKERS = 8  # shared-VM cap (#847): 8 processes x 1 BLAS thread each

TAG_HOLDOUT, TAG_FIT = 0, 1

DENSE_RIDGE = "eval_results/issue_1482/sae_perfeature/sae_dense_in__mean__ridge.npz"
DENSE_MLP = "eval_results/issue_1482/sae_perfeature/sae_dense_in__mean__mlp.npz"
TABLE_1773 = "eval_results/issue_1773/feature_table_v1.jsonl"
MATRIX = "eval_results/issue_1482/predictor_battery/fullwidth_matrix.npz"
OUT_DIR = "eval_results/issue_1482/predictor_battery"
FIG_DIR = "figures/issue_1482/predictor_battery"

R2_DISPLAY_FLOOR = -1.0
X_VIEW_PCT = 0.5

ARMS = ("ridge", "mlp")
ARM_LABEL = {"ridge": "linear (ridge)", "mlp": "nonlinear (MLP)"}
PARTIAL_ON = "activity"

# The user's ten, in their order. `dec_norm` is verified degenerate at runtime
# and dropped from the figures (kept in the JSON).
MAIN_PREDICTORS: tuple[tuple[str, str], ...] = (
    ("mean_act_uncond", "mean activation over all answers"),
    ("firing_freq_per_token", "firing frequency (per token)"),
    ("activity", "firing frequency (per answer)"),
    ("side_ratio", "answer-side share of firings"),
    ("scaffold_frac", "decoder energy in prefix-state top-48 subspace"),
    ("redundancy_max_cos", "max cosine to another decoder column"),
    ("act_var_across_answers", "activation variance across answers"),
    ("proj_var", "dense variance along decoder direction"),
    ("enc_norm", "encoder-vector norm"),
)
DEC_NORM_PREDICTOR = ("dec_norm", "decoder-column norm")

# Panel-grain only (#1773 table); NOT full-width. Kept in their own figure block.
PANEL_ONLY_PREDICTORS: tuple[tuple[str, str], ...] = (
    ("logit_footprint_concentration", "positive-logit mass on top-10 tokens"),
    ("massive_dim_mass", "decoder mass on massive-activation dims"),
    ("neighbors_cos_mean", "mean cosine to 8 nearest decoder columns"),
    ("persist_answer_sd", "answer-persistence sd"),
    ("describe_confidence", "auto-interp description confidence (ordinal)"),
)

# Absent from the user's list but among the strongest predictors — kept, in a
# clearly separated block, per the standing instruction not to drop them.
SUPPLEMENTARY_PREDICTORS: tuple[tuple[str, str], ...] = (
    ("consistency", "within-answer consistency"),
    ("write_norm", "write norm (gamma-scaled)"),
    ("mean_act_cond", "mean activation when active"),
    ("enc_dec_cos", "encoder-decoder cosine"),
    ("footprint_var", "footprint variance"),
    ("footprint_skew", "footprint skew"),
    ("footprint_kurt", "footprint kurtosis"),
)

# Stated BEFORE the reads; reported confirmed / refuted / null without softening.
DIRECTIONAL_PREDICTIONS = {
    "scaffold_frac": {
        "expected_sign": "positive",
        "claim": (
            "the blinded top/bottom-100 digest claimed best-predicted features are diffuse "
            "format/syntax/discourse SCAFFOLDING, so scaffold_frac should correlate POSITIVELY "
            "with R^2"
        ),
    },
    "logit_footprint_concentration": {
        "expected_sign": "negative",
        "claim": (
            "the same digest claimed worst-predicted features are specific TOKEN-IDENTITY "
            "detectors, so a focused vocabulary effect should correlate NEGATIVELY with R^2"
        ),
    },
}

NOT_COMPUTABLE = {
    "activation_variance_across_tokens": (
        "per-token SAE codes were pooled away at capture time (the store keeps per-row "
        "ans_mean / ans_max / ans_frac only), so within-answer across-TOKEN activation "
        "variance needs a re-capture, not a re-analysis"
    )
}


def _log(msg: str) -> None:
    print(f"[dense-reads] {msg}", flush=True)


# ── fused, process-pooled store scan ─────────────────────────────────────────


def _scan_shards(paths: list[str]) -> dict[str, np.ndarray]:
    """Accumulate EVERY store-derived covariate over one chunk of shards.

    Module-level (picklable) so `ProcessPoolExecutor` can fan it out. Each worker
    returns its own partial accumulators; the parent sums them. Loop shape is the
    verified #1773 phase-0 shape (`issue1773_phase0_mechanical.py` L164-185).
    """
    d = DICT_SIZE
    acc = {
        "cnt_fit": np.zeros(d, np.int64),
        "cnt_holdout": np.zeros(d, np.int64),
        "psi_cnt_fit": np.zeros(d, np.int64),
        "sum_frac": np.zeros(d, np.float64),
        "sum_mean": np.zeros(d, np.float64),
        "sum_mean_sq": np.zeros(d, np.float64),
        "tokens_active": np.zeros(d, np.float64),
        "h_sum": np.zeros(ACT_DIM, np.float64),
        "h_abs_sum": np.zeros(ACT_DIM, np.float64),
        "h_outer": np.zeros((ACT_DIM, ACT_DIM), np.float64),
    }
    n_fit = n_ho = n_rows = 0
    tokens_total = 0.0

    for p in paths:
        with np.load(p, allow_pickle=False) as z:
            tag = np.asarray(z["set_tag"])
            off = np.asarray(z["idx_off"], dtype=np.int64)
            idx = np.asarray(z["ans_idx"], dtype=np.int64)
            frac = np.asarray(z["ans_frac"], dtype=np.float64)
            act = np.asarray(z["ans_mean"], dtype=np.float64)
            n_ans = np.asarray(z["n_ans"], dtype=np.float64)

            fit = tag == TAG_FIT
            n_fit += int(fit.sum())
            n_rows += int(len(tag))
            keep = np.repeat(fit, off)
            ik, fk, ak = idx[keep], frac[keep], act[keep]
            acc["cnt_fit"] += np.bincount(ik, minlength=d)
            acc["sum_frac"] += np.bincount(ik, weights=fk, minlength=d)
            acc["sum_mean"] += np.bincount(ik, weights=ak, minlength=d)
            acc["sum_mean_sq"] += np.bincount(ik, weights=ak * ak, minlength=d)
            # per-TOKEN firing: ans_frac * n_ans = active answer tokens for that row
            tok_w = fk * np.repeat(n_ans[fit], off[fit])
            acc["tokens_active"] += np.bincount(ik, weights=tok_w, minlength=d)
            tokens_total += float(n_ans[fit].sum())

            psi_off = np.asarray(z["psi_off"], dtype=np.int64)
            psi_idx = np.asarray(z["psi_idx"], dtype=np.int64)
            acc["psi_cnt_fit"] += np.bincount(psi_idx[np.repeat(fit, psi_off)], minlength=d)

            ho = tag == TAG_HOLDOUT
            n_ho += int(ho.sum())
            acc["cnt_holdout"] += np.bincount(idx[np.repeat(ho, off)], minlength=d)

            hp = np.asarray(z["h_prefix"], dtype=np.float64)
            acc["h_sum"] += hp.sum(0)
            acc["h_abs_sum"] += np.abs(hp).sum(0)
            acc["h_outer"] += hp.T @ hp

    acc["n_fit"] = np.int64(n_fit)
    acc["n_holdout"] = np.int64(n_ho)
    acc["n_rows"] = np.int64(n_rows)
    acc["tokens_total"] = np.float64(tokens_total)
    return acc


def _sum_partials(parts: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    out = {k: np.copy(v) for k, v in parts[0].items()}
    for p in parts[1:]:
        for k, v in p.items():
            out[k] = out[k] + v
    return out


def fused_scan(store: Path, cache: Path, workers: int, verify_serial: int = 20) -> dict:
    """One process-pooled pass over the store; verified against a serial run.

    The shard loop is I/O-bound (9.2 GB over 1,920 npz), so processes — not
    threads — are the lever; each worker inherits the 8-thread BLAS cap and does
    almost no BLAS work besides the 3584^2 `h_prefix` outer product.
    """
    if cache.exists():
        with np.load(cache) as z:
            out = {k: z[k] for k in z.files}
        _log(f"scan cache hit: {cache} (n_fit={int(out['n_fit'])})")
        return out

    shards = sorted(str(p) for p in store.glob("pooled_*.npz"))
    if len(shards) != 1920:
        raise AssertionError(f"expected 1920 pooled shards, found {len(shards)} in {store}")

    # parallel-vs-serial equivalence on a small prefix BEFORE trusting the pool
    t0 = time.time()
    probe = shards[:verify_serial]
    serial = _scan_shards(probe)
    half = verify_serial // 2
    par = _sum_partials([_scan_shards(probe[:half]), _scan_shards(probe[half:])])
    for k in serial:
        a, b = np.asarray(serial[k], np.float64), np.asarray(par[k], np.float64)
        delta = float(np.max(np.abs(a - b))) if a.size else 0.0
        if not (delta <= 1e-9 * max(1.0, float(np.max(np.abs(a))) if a.size else 1.0)):
            raise AssertionError(f"parallel scan disagrees with serial on {k}: max|delta|={delta}")
    _log(f"parallel/serial equivalence PASS on {verify_serial} shards ({time.time() - t0:.0f}s)")

    chunks = [shards[i::workers] for i in range(workers)]
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        parts = list(ex.map(_scan_shards, chunks))
    out = _sum_partials(parts)
    _log(
        f"fused scan done in {time.time() - t0:.0f}s over {len(shards)} shards, "
        f"{workers} workers: n_fit={int(out['n_fit'])} n_holdout={int(out['n_holdout'])} "
        f"tokens={out['tokens_total']:.0f}"
    )
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, **out)
    return out


# ── derived full-width covariates ────────────────────────────────────────────


def derive_covariates(scan: dict, w_dec: np.ndarray, w_enc: np.ndarray) -> dict:
    """Full-width covariates from the fused scan plus the SAE weight matrices."""
    cnt = scan["cnt_fit"].astype(np.float64)
    psi = scan["psi_cnt_fit"].astype(np.float64)
    n_fit = float(scan["n_fit"])
    n_rows = float(scan["n_rows"])
    tokens_total = float(scan["tokens_total"])
    safe = np.maximum(cnt, 1.0)
    both = cnt + psi

    mean_uncond = scan["sum_mean"] / n_fit
    with np.errstate(invalid="ignore", divide="ignore"):
        cov = {
            "activity": cnt / n_fit,
            "firing_freq_per_token": scan["tokens_active"] / tokens_total,
            "consistency": np.where(cnt > 0, scan["sum_frac"] / safe, np.nan),
            "mean_act_uncond": mean_uncond,
            "mean_act_cond": np.where(cnt > 0, scan["sum_mean"] / safe, np.nan),
            "act_var_across_answers": np.maximum(scan["sum_mean_sq"] / n_fit - mean_uncond**2, 0.0),
            "side_ratio": np.where(both > 0, cnt / np.maximum(both, 1.0), np.nan),
            "n_active_holdout": scan["cnt_holdout"].astype(np.float64),
        }

    # decoder-geometry axes, #1773 L476-486 verbatim
    mean_abs = scan["h_abs_sum"] / max(n_rows, 1.0)
    thr = np.percentile(mean_abs, MASSIVE_DIM_PCTL)
    massive = np.where(mean_abs >= thr)[0]
    col_mass = np.einsum("ij,ij->j", w_dec, w_dec).astype(np.float64)
    cov["massive_dim_mass_fullwidth"] = (w_dec[massive] ** 2).sum(0).astype(
        np.float64
    ) / np.maximum(col_mass, 1e-12)
    mu = scan["h_sum"] / max(n_rows, 1.0)
    hcov = scan["h_outer"] / max(n_rows, 1.0) - np.outer(mu, mu)
    evals, evecs = np.linalg.eigh(hcov)
    basis = evecs[:, -SCAFFOLD_RANK:]
    proj = (basis.T.astype(np.float32) @ w_dec).astype(np.float64)
    cov["scaffold_frac"] = (proj**2).sum(0) / np.maximum(col_mass, 1e-12)

    cov["dec_norm"] = np.sqrt(np.maximum(col_mass, 0.0))
    enc_norm = np.linalg.norm(w_enc, axis=1).astype(np.float64)
    cov["enc_norm_recomputed"] = enc_norm
    dot = np.einsum("fd,df->f", w_enc, w_dec).astype(np.float64)
    denom = np.where((enc_norm > 0) & (cov["dec_norm"] > 0), enc_norm * cov["dec_norm"], np.nan)
    cov["enc_dec_cos"] = dot / denom
    return cov


def side_ratio_gate(scan: dict, cov: dict) -> dict:
    """Reproduce the team lead's independently-verified side_ratio census."""
    cnt, psi = scan["cnt_fit"], scan["psi_cnt_fit"]
    live = (cnt + psi) > 0
    got = {
        "n_fit": int(scan["n_fit"]),
        "answer_active": int((cnt > 0).sum()),
        "context_active": int((psi > 0).sum()),
        "context_only": int(((psi > 0) & (cnt == 0)).sum()),
        "answer_only": int(((cnt > 0) & (psi == 0)).sum()),
        "live": int(live.sum()),
        "median_side_ratio": round(float(np.median(cov["side_ratio"][live])), 3),
    }
    want = {
        "n_fit": 120000,
        "answer_active": 128512,
        "context_active": 128002,
        "context_only": 1654,
        "answer_only": 2164,
        "live": 130166,
        "median_side_ratio": 0.679,
    }
    mismatch = {k: (got[k], want[k]) for k in want if got[k] != want[k]}
    _log(f"side_ratio gate: {got}")
    if mismatch:
        raise AssertionError(
            f"side_ratio census does not reproduce the verified values: {mismatch}"
        )
    return {"observed": got, "expected": want, "status": "PASS"}


def identity_gates(cov: dict, feat_ids: np.ndarray) -> dict:
    """Recomputed activity / consistency must reproduce the committed matrix."""
    with np.load(PROJECT_ROOT / MATRIX, allow_pickle=True) as z:
        banked = {k: np.asarray(z[k], dtype=np.float64) for k in ("activity", "consistency")}
    out = {}
    for name, ref in banked.items():
        delta = float(np.nanmax(np.abs(cov[name][feat_ids] - ref)))
        out[f"{name}_max_abs_delta"] = delta
        _log(f"identity gate {name}: max|delta| = {delta:.3e}")
        if not (delta < 1e-6):
            raise AssertionError(f"recomputed {name} does not reproduce the matrix ({delta:.3e})")
    return out


# ── panel-grain targets + the #1773 supplementary table ──────────────────────


def load_targets() -> dict:
    """Both banked dense->SAE arms at panel width, on a shared feature id order."""
    arms = {}
    for arm, path in (("ridge", DENSE_RIDGE), ("mlp", DENSE_MLP)):
        with np.load(PROJECT_ROOT / path) as z:
            arms[arm] = {
                "feat_ids": np.asarray(z["feat_ids"], dtype=np.int64),
                "r2": np.asarray(z["r2"], dtype=np.float64),
            }
    if not np.array_equal(arms["ridge"]["feat_ids"], arms["mlp"]["feat_ids"]):
        raise AssertionError("dense->SAE ridge and MLP arms are on different feature id orders")
    fid = arms["ridge"]["feat_ids"]
    stats = {
        arm: {
            "per_feature_median_r2": float(np.median(v["r2"])),
            "per_feature_frac_positive": float((v["r2"] > 0).mean()),
            "per_feature_p10": float(np.percentile(v["r2"], 10)),
            "per_feature_mean": float(v["r2"].mean()),
            "per_feature_min": float(v["r2"].min()),
            "source": DENSE_RIDGE if arm == "ridge" else DENSE_MLP,
        }
        for arm, v in arms.items()
    }
    for arm, s in stats.items():
        _log(
            f"target {arm}: median {s['per_feature_median_r2']:+.4f} "
            f"pos {100 * s['per_feature_frac_positive']:.1f}% p10 {s['per_feature_p10']:+.4f}"
        )
    return {"feat_ids": fid, "r2": {a: v["r2"] for a, v in arms.items()}, "stats": stats}


def read_pooled_r2() -> dict:
    """Pooled (variance-weighted) R^2 per arm, read from the unit JSONs at run time.

    Read rather than transcribed: the pooled-vs-per-feature divergence is the
    headline, so both halves of it must trace to the artifact, not to a note.
    """
    out = {}
    for arm, fname, key in (
        ("ridge", "unit_ridge__sae_dense_in.json", "mean__ridge"),
        ("mlp", "unit_mlp__sae_dense_in__mean.json", "mean__mlp"),
    ):
        path = PROJECT_ROOT / "eval_results/issue_1482/sae_perfeature" / fname
        doc = json.loads(path.read_text())
        out[arm] = {
            "value": float(doc["arm_doc"][key]["pooled_r2"]),
            "source": f"sae_perfeature/{fname} arm_doc.{key}.pooled_r2",
        }
    _log(
        "pooled R^2: ridge {ridge:.4f} vs MLP {mlp:.4f}".format(
            ridge=out["ridge"]["value"], mlp=out["mlp"]["value"]
        )
    )
    return out


def load_1773_table(fid: np.ndarray) -> dict:
    """Panel-grain supplementary covariates, aligned to the target feature ids."""
    pos = {int(f): i for i, f in enumerate(fid)}
    n = len(fid)
    cols = {
        k: np.full(n, np.nan)
        for k in (
            "logit_footprint_concentration",
            "massive_dim_mass",
            "neighbors_cos_mean",
            "redundancy_max_cos_panel",
            "persist_answer_sd",
            "describe_confidence",
            "side_ratio_1773",
            "scaffold_frac_1773",
        )
    }
    hit = 0
    with (PROJECT_ROOT / TABLE_1773).open() as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            i = pos.get(int(r["feat_id"]))
            if i is None:
                continue
            hit += 1
            nl = r.get("nuisance_load") or {}
            lf = r.get("logit_footprint") or {}
            pa = r.get("persist_answer") or {}
            nb = (r.get("neighbors") or {}).get("cos") or []
            cols["logit_footprint_concentration"][i] = lf.get("concentration", np.nan)
            cols["massive_dim_mass"][i] = nl.get("massive_dim_mass", np.nan)
            cols["scaffold_frac_1773"][i] = nl.get("scaffold_frac", np.nan)
            cols["persist_answer_sd"][i] = pa.get("sd", np.nan)
            cols["side_ratio_1773"][i] = r.get("side_ratio", np.nan)
            dc = r.get("describe_confidence")
            cols["describe_confidence"][i] = np.nan if dc is None else float(dc)
            if nb:
                cols["neighbors_cos_mean"][i] = float(np.mean(nb))
                cols["redundancy_max_cos_panel"][i] = float(np.max(nb))
    _log(f"#1773 table: {hit}/{n} panel features matched")
    return cols


def table_gates(cov: dict, cols: dict, fid: np.ndarray) -> dict:
    """Full-width recomputes vs the banked panel values for the same quantity."""
    out = {}
    for mine, theirs in (
        ("scaffold_frac", "scaffold_frac_1773"),
        ("massive_dim_mass_fullwidth", "massive_dim_mass"),
        ("side_ratio", "side_ratio_1773"),
    ):
        a, b = cov[mine][fid], cols[theirs]
        ok = np.isfinite(a) & np.isfinite(b)
        out[mine] = {
            "vs": theirs,
            "n": int(ok.sum()),
            "max_abs_delta": float(np.max(np.abs(a[ok] - b[ok]))) if ok.any() else float("nan"),
            "spearman": PB._spearman(a[ok], b[ok]) if ok.sum() > 2 else float("nan"),
        }
        _log(
            f"table gate {mine} vs {theirs}: max|delta|={out[mine]['max_abs_delta']:.3e} "
            f"rho={out[mine]['spearman']:+.6f} (n={out[mine]['n']})"
        )
    return out


# ── reads: raw + activity-partial rho per arm, batched bootstrap ─────────────


def _partial(r_jy: np.ndarray, r_ja: np.ndarray, r_ay) -> np.ndarray:
    denom = np.sqrt(np.maximum((1.0 - r_ja**2) * (1.0 - r_ay**2), 0.0))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(denom > 0, (r_jy - r_ja * r_ay) / denom, np.nan)


def _reads_for_arm(z: np.ndarray, a: int, y: int, n_boot: int, rng) -> dict:
    """Point + bootstrap raw and activity-partial rho for every column against y.

    `z` is the FIXED rank matrix (predictors, then the arm's y). Four GEMMs per
    bootstrap chunk give every predictor's (j,y), (j,a) and (a,y) moment, so no
    per-draw or per-predictor loop runs.
    """
    n = z.shape[0]
    zc = (z - z.mean(0)) / z.std(0)
    corr = (zc.T @ zc) / n
    raw = corr[:, y]
    par = _partial(corr[:, y], corr[:, a], corr[a, y])
    par[a] = np.nan

    raw_d, par_d = [], []
    done = 0
    while done < n_boot:
        b = min(BOOT_CHUNK, n_boot - done)
        idx = rng.integers(0, n, size=(n, b))
        w = np.bincount((idx + np.arange(b) * n).ravel(), minlength=n * b)
        w = w.reshape(b, n).T.astype(np.float64)
        s1 = z.T @ w
        s2 = (z * z).T @ w
        cy = (z * z[:, y : y + 1]).T @ w
        ca = (z * z[:, a : a + 1]).T @ w
        mu = s1 / n
        sd = np.sqrt(np.maximum(s2 / n - mu * mu, 0.0))
        with np.errstate(invalid="ignore", divide="ignore"):
            r_jy = (cy / n - mu * mu[y]) / (sd * sd[y])
            r_ja = (ca / n - mu * mu[a]) / (sd * sd[a])
        raw_d.append(r_jy.T)
        par_d.append(_partial(r_jy, r_ja, r_jy[a]).T)
        done += b
    return {"raw": raw, "partial": par, "raw_b": np.vstack(raw_d), "par_b": np.vstack(par_d)}


def correlation_reads(panel: dict, keys: list[str], labels: dict, n_boot: int, rng) -> dict:
    """Both arms, over the features finite in every predictor AND both targets."""
    cov = panel["cov"]
    mats = [cov[k] for k in keys]
    ok = np.ones(len(panel["feat_ids"]), dtype=bool)
    for v in mats:
        ok &= np.isfinite(v)
    for arm in ARMS:
        ok &= np.isfinite(panel["r2"][arm])
    n_used = int(ok.sum())
    _log(f"reads: {n_used}/{len(ok)} panel features finite across all predictors + both arms")

    a = keys.index(PARTIAL_ON)
    y = len(keys)
    ranked = [PB._rank(v[ok]) for v in mats]
    per_arm = {}
    for arm in ARMS:
        z = np.column_stack([*ranked, PB._rank(panel["r2"][arm][ok])])
        t0 = time.time()
        per_arm[arm] = _reads_for_arm(z, a, y, n_boot, rng)
        _log(f"  {arm}: {n_boot} bootstrap draws in {time.time() - t0:.0f}s")

    rows = []
    for j, key in enumerate(keys):
        row = {"key": key, "label": labels[key], "n": n_used}
        for arm in ARMS:
            r = per_arm[arm]
            row[arm] = {
                "spearman_raw": float(r["raw"][j]),
                "spearman_raw_ci95": PB._ci(r["raw_b"][:, j]),
                "partial_on_activity": (None if j == a else float(r["partial"][j])),
                "partial_on_activity_ci95": (None if j == a else PB._ci(r["par_b"][:, j])),
            }
        row["log_x_axis"] = bool(_log_x(cov[key][ok]))
        for arm in ARMS:
            row[f"decile_{arm}"] = _decile_profile(cov[key][ok], panel["r2"][arm][ok])
        rows.append(row)
        _log(
            f"{key}: ridge raw {row['ridge']['spearman_raw']:+.3f} "
            f"partial {row['ridge']['partial_on_activity']}  |  "
            f"mlp raw {row['mlp']['spearman_raw']:+.3f} "
            f"partial {row['mlp']['partial_on_activity']}"
        )
    return {"predictors": rows, "n_used": n_used, "mask": ok}


def _log_x(v: np.ndarray) -> bool:
    finite = v[np.isfinite(v)]
    if len(finite) == 0 or not (finite > 0).all():
        return False
    p50, p99 = np.percentile(finite, [50, 99])
    return bool(p50 > 0 and p99 / p50 >= 5.0)


def _decile_profile(pred: np.ndarray, r2: np.ndarray) -> dict:
    edges = np.quantile(pred, np.linspace(0, 1, N_DECILES + 1))
    dec = np.searchsorted(edges[1:-1], pred, side="right")
    med, cnt = [], []
    for d in range(N_DECILES):
        m = dec == d
        med.append(float(np.median(r2[m])) if m.any() else float("nan"))
        cnt.append(int(m.sum()))
    return {
        "decile_median_r2": med,
        "decile_center": [float((edges[i] + edges[i + 1]) / 2) for i in range(N_DECILES)],
        "decile_n": cnt,
    }


def adjudicate_predictions(reads: dict) -> dict:
    """Confirmed / refuted / null for each pre-stated directional prediction."""
    by_key = {r["key"]: r for r in reads["predictors"]}
    out = {}
    for key, spec in DIRECTIONAL_PREDICTIONS.items():
        row = by_key.get(key)
        if row is None:
            out[key] = {**spec, "verdict": "not-tested", "reason": "predictor absent from reads"}
            continue
        entry = {**spec, "arms": {}}
        for arm in ARMS:
            raw = row[arm]["spearman_raw"]
            ci = row[arm]["spearman_raw_ci95"]
            want_pos = spec["expected_sign"] == "positive"
            if ci[0] <= 0 <= ci[1]:
                verdict = "null"
            elif (raw > 0) == want_pos:
                verdict = "confirmed"
            else:
                verdict = "REFUTED"
            entry["arms"][arm] = {
                "spearman_raw": raw,
                "spearman_raw_ci95": ci,
                "partial_on_activity": row[arm]["partial_on_activity"],
                "partial_on_activity_ci95": row[arm]["partial_on_activity_ci95"],
                "verdict": verdict,
            }
        out[key] = entry
        _log(
            f"prediction {key} ({spec['expected_sign']}): "
            + ", ".join(f"{a}={entry['arms'][a]['verdict']}" for a in ARMS)
        )
    return out


# ── figures ──────────────────────────────────────────────────────────────────


def _subtitle(reads: dict, targets: dict) -> str:
    rs, ms = targets["stats"]["ridge"], targets["stats"]["mlp"]
    return (
        f"dense context state (3,584-d) -> SAE answer features, {reads['n_used']:,} panel "
        f"features (BatchTopK k=64, layer 19)  |  ridge median $R^2$ "
        f"{rs['per_feature_median_r2']:+.3f} ({100 * rs['per_feature_frac_positive']:.0f}% positive) "
        f"vs MLP {ms['per_feature_median_r2']:+.3f} ({100 * ms['per_feature_frac_positive']:.0f}%)  |  "
        f"target and covariates BOTH #1482 single-turn — no cross-corpus join"
    )


def fig_scatter(reads: dict, targets: dict, panel: dict, groups: list, fig_dir: Path) -> str:
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import (
        paper_palette_role,
        savefig_paper,
        set_paper_style,
    )

    set_paper_style()
    by_key = {r["key"]: r for r in reads["predictors"]}
    ok = reads["mask"]
    ridge = np.clip(panel["r2"]["ridge"][ok], R2_DISPLAY_FLOOR, None)
    ordered = [(k, lab, gname) for gname, items in groups for k, lab in items]

    ncol = 5
    nrow = int(np.ceil(len(ordered) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.15 * ncol, 2.95 * nrow), sharey=True)
    flat = axes.ravel()

    for ax, (key, label, gname) in zip(flat, ordered, strict=False):
        v = panel["cov"][key][ok]
        row = by_key[key]
        log_x = row["log_x_axis"]
        ax.hexbin(
            v,
            ridge,
            gridsize=46,
            bins="log",
            mincnt=1,
            xscale="log" if log_x else "linear",
            cmap="Blues",
            linewidths=0,
        )
        ax.plot(
            row["decile_ridge"]["decile_center"],
            row["decile_ridge"]["decile_median_r2"],
            "-",
            lw=1.5,
            color=paper_palette_role("accent"),
            label="ridge decile median",
        )
        ax.plot(
            row["decile_mlp"]["decile_center"],
            row["decile_mlp"]["decile_median_r2"],
            "--",
            lw=1.5,
            color=paper_palette_role("primary"),
            label="MLP decile median",
        )
        if not log_x:
            finite = v[np.isfinite(v)]
            lo, hi = np.percentile(finite, [X_VIEW_PCT, 100.0 - X_VIEW_PCT])
            if hi > lo:
                pad = 0.04 * (hi - lo)
                ax.set_xlim(lo - pad, hi + pad)
            ax.locator_params(axis="x", nbins=5)
            ax.ticklabel_format(axis="x", style="sci", scilimits=(-2, 4), useMathText=True)
            ax.xaxis.get_offset_text().set_fontsize(7.0)
        ax.text(
            0.04,
            0.96,
            f"ridge $\\rho$ = {row['ridge']['spearman_raw']:+.3f}\n"
            f"MLP $\\rho$ = {row['mlp']['spearman_raw']:+.3f}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=8.0,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.88, "pad": 2.2},
        )
        ax.set_xlabel(f"{label}{' (log)' if log_x else ''}\n[{gname}]", fontsize=7.6)
        ax.tick_params(labelsize=7.2)

    for ax in flat[: len(ordered)][::ncol]:
        ax.set_ylabel(r"held-out $R^2$ (ridge; hexbin)", fontsize=8.2)
    for ax in flat[len(ordered) :]:
        ax.set_visible(False)
    flat[0].legend(loc="lower right", frameon=False, fontsize=6.8)

    fig.suptitle(
        "Which feature properties track how well the dense context state predicts an SAE "
        "answer feature?",
        fontsize=12.5,
        y=0.995,
    )
    fig.text(0.5, 0.958, _subtitle(reads, targets), ha="center", fontsize=7.4, color="#5A5A5A")
    fig.text(
        0.5,
        0.010,
        "hexbin + solid trend = RIDGE; dashed trend = MLP (both panel-grain, "
        f"{reads['n_used']:,} features). "
        f"$R^2$ display-clipped at {R2_DISPLAY_FLOOR:g}; linear-axis panels view-clip x to "
        f"[{X_VIEW_PCT:g}, {100 - X_VIEW_PCT:g}] pct — display only, all statistics unclipped. "
        "Bracketed tag = predictor group.",
        ha="center",
        fontsize=7.2,
        color="#5A5A5A",
    )
    fig.tight_layout(rect=(0, 0.028, 1, 0.948))
    stem = "densesae_predictor_scatter_panel"
    savefig_paper(fig, stem, dir=fig_dir)
    plt.close(fig)
    return stem


def fig_forest(reads: dict, targets: dict, fig_dir: Path) -> str:
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import (
        paper_palette_role,
        savefig_paper,
        set_paper_style,
    )

    set_paper_style()
    rows = list(reads["predictors"])
    scored = [r for r in rows if r["ridge"]["partial_on_activity"] is not None]
    unscored = [r for r in rows if r["ridge"]["partial_on_activity"] is None]
    scored.sort(key=lambda r: abs(r["ridge"]["partial_on_activity"]))
    ordered = unscored + scored
    ypos = np.arange(len(ordered), dtype=float)

    fig, axes = plt.subplots(
        1, 2, figsize=(13.2, 0.40 * len(ordered) + 3.2), sharey=True, sharex=True
    )
    for ax, arm in zip(axes, ARMS, strict=True):
        raw = np.array([r[arm]["spearman_raw"] for r in ordered])
        raw_ci = np.array([r[arm]["spearman_raw_ci95"] for r in ordered])
        par = np.array(
            [
                np.nan if r[arm]["partial_on_activity"] is None else r[arm]["partial_on_activity"]
                for r in ordered
            ]
        )
        par_ci = np.array(
            [
                [np.nan, np.nan]
                if r[arm]["partial_on_activity_ci95"] is None
                else r[arm]["partial_on_activity_ci95"]
                for r in ordered
            ]
        )
        ok = np.isfinite(par)
        ax.axvline(0.0, color=paper_palette_role("neutral"), lw=0.9, ls="--")
        ax.errorbar(
            raw,
            ypos + 0.16,
            xerr=PB._errbars(raw, raw_ci),
            fmt="o",
            ms=4.4,
            color=paper_palette_role("baseline"),
            mfc="white",
            mew=1.2,
            lw=1.0,
            capsize=2.0,
            label="raw Spearman $\\rho$",
        )
        ax.errorbar(
            par[ok],
            ypos[ok] - 0.16,
            xerr=PB._errbars(par[ok], par_ci[ok]),
            fmt="o",
            ms=4.4,
            color=paper_palette_role("primary"),
            lw=1.0,
            capsize=2.0,
            label="partialling out per-answer firing frequency",
        )
        ax.set_title(ARM_LABEL[arm], fontsize=10.5, loc="left")
        ax.set_xlabel(r"Spearman $\rho$ against held-out $R^2$", fontsize=9.0)

    labels = [
        r["label"] + ("" if r["ridge"]["partial_on_activity"] is not None else "  — partial n/a")
        for r in ordered
    ]
    axes[0].set_yticks(ypos)
    axes[0].set_yticklabels(labels, fontsize=8.2)
    axes[0].set_ylim(-0.75, len(ordered) - 0.25)
    axes[0].legend(
        loc="upper center", bbox_to_anchor=(1.02, -0.07), ncol=2, frameon=False, fontsize=8.4
    )

    fig.suptitle(
        "What survives adjusting for firing frequency — and does the ordering differ between "
        "the linear and nonlinear maps?",
        fontsize=12.5,
        y=0.985,
    )
    fig.text(0.5, 0.945, _subtitle(reads, targets), ha="center", fontsize=7.4, color="#5A5A5A")
    fig.text(
        0.5,
        0.012,
        f"rows sorted by |ridge partial rho|; percentile 95% CI over {N_BOOT:,} bootstrap draws "
        f"(ranks fixed at the full sample) — at n = {reads['n_used']:,} they are often narrower "
        "than the markers.",
        ha="center",
        fontsize=7.4,
        color="#5A5A5A",
    )
    fig.tight_layout(rect=(0, 0.075, 1, 0.935))
    stem = "densesae_rho_vs_activity_partial"
    savefig_paper(fig, stem, dir=fig_dir)
    plt.close(fig)
    return stem


# ── entrypoint ───────────────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser(description="#1482 dense->SAE per-feature predictor reads")
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--workers", type=int, default=N_WORKERS)
    ap.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / OUT_DIR)
    ap.add_argument("--fig-dir", type=Path, default=PROJECT_ROOT / FIG_DIR)
    ap.add_argument("--work", type=Path, default=PROJECT_ROOT / "data/issue_1482/fullwidth")
    ap.add_argument("--phase", default="all", choices=("all", "scan", "analyze"))
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    if args.smoke:
        args.n_boot = 50
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.work.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    scan = fused_scan(POOLED_STORE, args.work / "fused_scan.npz", args.workers)
    if args.phase == "scan":
        return

    from issue1482_sae import BatchTopKSAE

    sae = BatchTopKSAE.load(k=PB.SAE_K, layer=PB.SAE_LAYER, device="cpu")
    cov_full = derive_covariates(scan, np.asarray(sae.w_dec), np.asarray(sae.w_enc))
    del sae

    with np.load(PROJECT_ROOT / MATRIX, allow_pickle=True) as z:
        universe = np.asarray(z["feat_ids"], dtype=np.int64)
        from_matrix = {
            k: np.asarray(z[k], dtype=np.float64)
            for k in ("proj_var", "write_norm", "footprint_var", "footprint_skew", "footprint_kurt")
        }
    gates = identity_gates(cov_full, universe)
    gates["side_ratio_census"] = side_ratio_gate(scan, cov_full)

    targets = load_targets()
    fid = targets["feat_ids"]
    cols_1773 = load_1773_table(fid)
    gates["vs_1773_table"] = table_gates(cov_full, cols_1773, fid)

    # panel-grain covariate matrix: full-width derived (sliced) + matrix + #1773
    upos = {int(f): i for i, f in enumerate(universe)}
    take = np.array([upos.get(int(f), -1) for f in fid])
    if (take < 0).any():
        _log(f"WARN {int((take < 0).sum())} panel ids absent from the full-width universe")
    cov: dict[str, np.ndarray] = {}
    for k, v in cov_full.items():
        cov[k] = v[fid]
    for k, v in from_matrix.items():
        col = np.full(len(fid), np.nan)
        good = take >= 0
        col[good] = v[take[good]]
        cov[k] = col
    cov.update(cols_1773)
    cov["redundancy_max_cos"] = cov["redundancy_max_cos_panel"]
    cov["enc_norm"] = cov["enc_norm_recomputed"]
    cov["massive_dim_mass"] = cov["massive_dim_mass"]

    dec_std = float(np.nanstd(cov["dec_norm"]))
    dec_degenerate = bool(dec_std < 1e-6)
    _log(f"dec_norm: mean {np.nanmean(cov['dec_norm']):.8f} std {dec_std:.3e}")

    main_list = [p for p in MAIN_PREDICTORS] + ([] if dec_degenerate else [DEC_NORM_PREDICTOR])
    groups = [
        ("user list", main_list),
        ("panel-only (#1773)", list(PANEL_ONLY_PREDICTORS)),
        ("supplementary", list(SUPPLEMENTARY_PREDICTORS)),
    ]
    keys = [k for _, items in groups for k, _ in items]
    labels = {k: lab for _, items in groups for k, lab in items}

    panel = {"feat_ids": fid, "cov": cov, "r2": targets["r2"]}
    rng = np.random.default_rng(SEED)
    reads = correlation_reads(panel, keys, labels, args.n_boot, rng)
    predictions = adjudicate_predictions(reads)

    np.savez(
        args.out_dir / "densesae_panel_covariates.npz",
        feat_ids=fid,
        r2_ridge=targets["r2"]["ridge"],
        r2_mlp=targets["r2"]["mlp"],
        **{k: cov[k] for k in keys if k in cov},
    )

    doc = {
        "design": {
            "target_map": "dense context state (3,584-d, last prompt token) -> SAE answer features",
            "arm": "sae_dense_in, mean pooling",
            "width": "PANEL (16,384) — full-width dense->SAE R^2 does not exist yet",
            "full_width_owner": (
                "task #7 scripts/issue1482_densesae_fullwidth.py (another agent); this module "
                "deliberately does NOT duplicate that refit"
            ),
            "n_features_used": reads["n_used"],
            "partial_on": PARTIAL_ON,
            "n_boot": int(args.n_boot),
            "n_fit_rows": int(scan["n_fit"]),
            "scan_workers": int(args.workers),
            "seed": SEED,
        },
        "linear_vs_nonlinear": {
            "note": (
                "the MLP BEATS ridge on the variance-weighted POOLED metric while LOSING "
                "per-feature on half the dictionary — capacity allocation made visible: ridge "
                "solves each output column independently (no capacity competition), whereas the "
                "MLP's shared hidden layer forces allocation under an unstandardized-target loss, "
                "so it spends capacity on high-variance features and abandons the low-variance "
                "majority. Same trade-off the PC-level read already shows, far starker in "
                "feature space."
            ),
            "pooled_r2": read_pooled_r2(),
            "per_feature": targets["stats"],
        },
        "gates": gates,
        "directional_predictions": predictions,
        "dec_norm": {
            "mean": float(np.nanmean(cov["dec_norm"])),
            "std": dec_std,
            "degenerate": dec_degenerate,
            "excluded_from_figures": dec_degenerate,
        },
        "predictor_groups": {g: [k for k, _ in items] for g, items in groups},
        "predictors": reads["predictors"],
        "not_computable": NOT_COMPUTABLE,
        "definitions": {
            "scaffold_frac": (
                "||P_48 . w_dec[:,f]||^2 / ||w_dec[:,f]||^2 with P_48 the top-48 PCA eigenvectors "
                "of cov(h_prefix) — #1773 L484-486. DECODER GEOMETRY (share of the feature's "
                "write-direction energy in the dominant prefix-state subspace), NOT a "
                "fraction-of-firings-on-template-tokens statistic."
            ),
            "logit_footprint_concentration": (
                "share of POSITIVE direct-logit mass on the top-10 promoted tokens, "
                "E = W_U @ (gamma * W_dec[:,f]) — #1773 L256-284. Panel-grain banked values are "
                "used here; no full-width recompute was run (see measured_projections)."
            ),
            "side_ratio": (
                "cnt / (cnt + psi_cnt) over set_tag==1 rows — answer-side share of firings. "
                "Near 0.5 = the feature fires on BOTH sides, so the map can predict it by "
                "persistence; near 1.0 = answer-only, no context-side counterpart to copy."
            ),
            "firing_freq_per_token": (
                "sum over fit rows of ans_frac * n_ans, divided by the total answer tokens — "
                "per-TOKEN firing frequency, distinct from `activity` (per-ANSWER)."
            ),
        },
        "metadata": PB._metadata(),
    }
    (args.out_dir / "densesae_predictors.json").write_text(json.dumps(doc, indent=1))
    _log(f"reads -> {args.out_dir / 'densesae_predictors.json'}")

    import matplotlib

    matplotlib.use("Agg")
    args.fig_dir.mkdir(parents=True, exist_ok=True)
    plot_groups = [(g, [(k, labels[k]) for k, _ in items]) for g, items in groups]
    stems = [
        fig_scatter(reads, targets, panel, plot_groups, args.fig_dir),
        fig_forest(reads, targets, args.fig_dir),
    ]
    _log(f"figures: {', '.join(stems)}  (total {time.time() - t_start:.0f}s)")


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "8")
    main()
