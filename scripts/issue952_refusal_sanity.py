"""#952 refusal sanity checks (Dan Mossing comment) — activations vs mapping refusal reads.

Dan's ask (paraphrased): "i would have thought that refusal is well predicted by
the linear mapping, and refusals would have quite different activations from
nonrefusals. is that not the case? ... if the mapping doesn't predict refusal vs
nonrefusal on related topics, maybe it needs more training data or so."

This disentangles #952's china-politics divergence null (arm-matched d +0.014,
Holm p 0.38 — no divergence-specific external penalty) into three verifiable
premises, on the china bank ONLY, at layer 20:

  Check A — are Qwen refusal answers' activations distinct from non-refusal? A
    leave-one-out mean-difference refusal axis over the actual OWN-arm answer
    summaries y(q); ROC AUC, effect size d', variance fractions, permutation p.
  Check B — does the pool-trained linear map PREDICT refusal? Project the own
    map's predicted summaries yhat(q) onto the SAME LOO refusal axes (built from
    actual y); AUC + correlation. Paired closeness on genuine-divergence cells
    (Qwen refuses / Claude answers): does yhat point toward Qwen's own answer rep
    or Claude's?
  Check C — is refusal linearly decodable from the CONTEXT c_last at all? Same
    LOO mean-difference AUC, feature = own-arm c_last. The information ceiling for
    ANY linear map from c_last.

Machinery reused VERBATIM from the committed #952 free-analysis line: the
run_ridge_cell shared-SVD ridge (frozen per-slot lambda from per_context_stats.npz),
the universe/split reproduction, the POSITION_SLOTS / SLOT_IDX registry, and the
sign-flip helper — imported from issue952_divergence_transfer_cell so the fit +
per-query-R2 machinery is provably identical to the committed cells.

Reproduction gate (REQUIRED before any new read):
  Gate 1 — recompute the committed 41-pair arm-matched headline d from the
    committed npz + match stats_summary.json h3 (stats machinery).
  Gate 2 — re-fit the OWN map on the reproduced U_A universe; match the committed
    npz per-pair own-arm drops (fit reproduction).
  Gate P — the prediction path used for yhat reproduces run_ridge_cell's pooled
    per-query R2 on the committed bank (prediction machinery == library fit).

Refusal labels: `refusal_qwen` / `refusal_claude` are the MEAN of 3 refusal-rubric
judge draws in [0,100] (higher = more refusal; malformed dropped, never coerced;
run_952.REFUSAL_RUBRIC + _aggregate_draws). The refusal boolean is threshold >= 50
(rubric midpoint = "partial deflection"); a graded-companion Spearman read rides
along so the reads are not hostage to the threshold. Labels for the 24 new china
pairs come from china_topup_verification.json; for the 18 parent-kept china pairs
from the committed divergence_bank_verification.json.

Content safety: china bank items are CCP-sensitive. This script references bank
items by query_id / pair_id ONLY. It NEVER prints, logs, or persists bank query
or answer TEXT — it reads only activation tensors + numeric refusal/divergence
labels. Figures carry query_ids, never text.

Usage:
  OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8 \
    uv run python scripts/issue952_refusal_sanity.py
"""

from __future__ import annotations

# ruff: noqa: E402 — load_dotenv() must run before numpy/torch import (shared-VM thread caps)
from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import datetime
import json
import logging
import pathlib
import subprocess
import time

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("issue952_refusal_sanity")

# Reuse the committed transfer-cell machinery verbatim (identical fit + stats).
from issue952_divergence_transfer_cell import (
    ARMS,
    BANK_ARMS,
    N_DRAWS,
    POSITION_SLOTS,
    SLOT_IDX,
    _per_query_r2,
    _signflip_p,
    _stack_answer_targets,
)

REPO = "superkaiba1/explore-persona-space-data"
PARENT_REV = "5b62649cefb34902fd630f21630164e8d1d99764"
CHINA_REV = "612c6c744e786ff65faae8e7ee97736239f873e5"
PARENT_PREFIX = "issue952_position_divergence/analysis_tensors"
CHINA_PREFIX = "issue952_position_divergence/followups/china_politics_topup/analysis_tensors"
STAGE = pathlib.Path("/mnt/eps-data/thomasjiralerspong/tmp_issue952_refusal")

DEFAULT_LAMBDAS = np.logspace(-2, 4, 13)  # #779 fit_h grid (ridge_battery.DEFAULT_LAMBDAS)
REFUSAL_THR = 50.0  # rubric midpoint ("partial deflection"); boolean cut
REPRO_TOL = 0.01
N_PERM = 10000
N_BOOT = 2000
PERM_SEED = 1
BOOT_SEED = 0


def _sha() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=pathlib.Path(__file__).resolve().parent.parent,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


# ── staging (per-file download with bounded retry; scoped, never snapshot) ────────


def _stage(prefix: str, rev: str, fname: str) -> pathlib.Path:
    from huggingface_hub import hf_hub_download

    dest = STAGE / rev[:10] / fname
    if dest.exists() or dest.is_symlink():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    last: Exception | None = None
    for attempt in range(4):
        try:
            p = hf_hub_download(REPO, f"{prefix}/{fname}", repo_type="dataset", revision=rev)
            dest.symlink_to(pathlib.Path(p))
            return dest
        except Exception as e:  # transient Hub 5xx / 429 — bounded retry
            last = e
            logger.warning("[hf] %s failed (attempt %d): %s", fname, attempt + 1, e)
            time.sleep(20 * (attempt + 1))
    raise RuntimeError(f"HF download failed after retries: {prefix}/{fname}") from last


def _load_slots(path: pathlib.Path) -> tuple[np.ndarray, list]:
    d = torch.load(str(path), map_location="cpu", weights_only=False)
    return d["slots"].numpy(), list(d["ids"])


# ── refusal-label assembly (numeric labels only; no bank text) ───────────────────


def _build_label_map(base: pathlib.Path) -> dict[str, dict]:
    """query_id -> {refusal_qwen, refusal_claude, role, origin, pair_id} for the 84
    captured china queries. New pairs -> china_topup_verification.json; parent-kept
    pairs -> committed divergence_bank_verification.json."""
    pv = json.loads((base / "eval_results/issue_952/divergence_bank_verification.json").read_text())
    cv = json.loads(
        (
            base
            / "eval_results/issue_952/china-politics-topup/summaries/china_topup_verification.json"
        ).read_text()
    )
    prov = json.loads(
        (
            base / "eval_results/issue_952/china-politics-topup/summaries/"
            "provenance_china_politics_topup.json"
        ).read_text()
    )
    parent_china = {p["pair_id"]: p for p in pv["pairs"] if p.get("category") == "china_politics"}
    new_china = {p["pair_id"]: p for p in cv["pairs"]}
    labmap: dict[str, dict] = {}
    for qid, meta in prov["provenance"].items():
        pid, role, origin = meta["pair_id"], meta["role"], meta["origin"]
        src = new_china.get(pid) if origin == "new" else parent_china.get(pid)
        rk = "divergent" if role == "divergent" else "control"
        rec = src.get(rk) if isinstance(src, dict) else None
        labmap[qid] = {
            "refusal_qwen": rec.get("refusal_qwen") if isinstance(rec, dict) else None,
            "refusal_claude": rec.get("refusal_claude") if isinstance(rec, dict) else None,
            "role": role,
            "origin": origin,
            "pair_id": pid,
        }
    return labmap


# ── shared-SVD own-map fit (matches run_ridge_cell math + imputation) ─────────────


def _impute_train_nan(Y: np.ndarray) -> np.ndarray:
    """Per-slot NaN-row imputation at the finite-rows mean (run_ridge_cell
    allow_train_nan_imputation=True; Y is one slot, (n_tr, H))."""
    row_bad = ~np.isfinite(Y).all(axis=1)
    if row_bad.any():
        if row_bad.all():
            raise ValueError("all train rows NaN for a slot")
        Y = Y.copy()
        Y[row_bad] = Y[~row_bad].mean(axis=0)
    return Y


def _own_train_slot(own_pool: np.ndarray, tr_a: np.ndarray, slot: str) -> np.ndarray:
    """Own-arm pool answer activations at `slot`, train rows, f64, NaN-row imputed.
    Slot-first indexing keeps peak memory at ~one (n_tr, H) slice, never (n_tr,72,H)."""
    return _impute_train_nan(own_pool[:, SLOT_IDX[slot], :][tr_a].astype(np.float64))


def _own_map_predict(
    c_last_tr: np.ndarray,
    own_pool: np.ndarray,
    tr_a: np.ndarray,
    group2lam: dict,
    bank_c_last: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Frozen-lambda own map c_last -> per-slot answer activation, applied to a bank.

    Shared SVD of standardized c_last_tr (once); per POSITION_SLOT the target is the
    own-arm pool slot at tr_a (NaN-row imputed), centered on its train mean, at the
    committed per-slot lambda. Returns (preds (n_bank, 42, H), ymu (42, H))."""
    Xtr = np.asarray(c_last_tr, dtype=np.float64)
    xmu, xsd = Xtr.mean(0), Xtr.std(0) + 1e-9
    U, s, Vh = np.linalg.svd((Xtr - xmu) / xsd, full_matrices=False)
    s2 = s**2
    A = ((np.asarray(bank_c_last, dtype=np.float64) - xmu) / xsd) @ Vh.T  # (n_bank, r)
    h = own_pool.shape[2]
    preds = np.full((bank_c_last.shape[0], len(POSITION_SLOTS), h), np.nan, dtype=np.float64)
    ymu_all = np.full((len(POSITION_SLOTS), h), np.nan, dtype=np.float64)
    for si, slot in enumerate(POSITION_SLOTS):
        Y = _own_train_slot(own_pool, tr_a, slot)
        ymu = Y.mean(0)
        B = U.T @ (Y - ymu)  # (r, H)
        filt = s / (s2 + DEFAULT_LAMBDAS[int(group2lam[f"{slot}|own"])])
        preds[:, si, :] = (A * filt[None, :]) @ B + ymu
        ymu_all[si] = ymu
    return preds, ymu_all


# ── LOO mean-difference refusal axis + AUC (vectorized over queries) ──────────────


def _auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """ROC AUC of `scores` vs binary `labels` (rank / Mann-Whitney; ties averaged)."""
    labels = labels.astype(bool)
    n1, n0 = int(labels.sum()), int((~labels).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    sc = scores[order]
    i = 0
    while i < len(sc):
        j = i
        while j + 1 < len(sc) and sc[j + 1] == sc[i]:
            j += 1
        ranks[order[i : j + 1]] = 0.5 * (i + j) + 1.0  # 1-based average rank
        i = j + 1
    return float((ranks[labels].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def _loo_scores(Y_axis: np.ndarray, Y_proj: np.ndarray, r: np.ndarray) -> np.ndarray:
    """LOO mean-difference projection scores over ALL rows given (the eval set is the
    caller's slice). Axis Delta_mu(-q) = mean(Y_axis|r=1, excl q) - mean(...|r=0);
    score(q) = <Y_proj(q) - pooled_mean(Y_proj, excl q), Delta_mu(-q)>. Class sizes
    fixed under permutation; LOO leaves >=1 per class by caller guarantee."""
    r = r.astype(np.float64)
    m = len(r)
    S1 = (r[:, None] * Y_axis).sum(0)
    S0 = ((1 - r)[:, None] * Y_axis).sum(0)
    n1, n0 = r.sum(), (1 - r).sum()
    mu1_q = (S1[None, :] - r[:, None] * Y_axis) / (n1 - r)[:, None]
    mu0_q = (S0[None, :] - (1 - r)[:, None] * Y_axis) / (n0 - (1 - r))[:, None]
    dmu = mu1_q - mu0_q
    mu_proj_q = (Y_proj.sum(0)[None, :] - Y_proj) / (m - 1)
    return ((Y_proj - mu_proj_q) * dmu).sum(1)


def _loo_auc_with_perm(
    Y_axis: np.ndarray,
    Y_proj: np.ndarray,
    r: np.ndarray,
    *,
    n_perm: int = N_PERM,
    n_boot: int = N_BOOT,
    perm_seed: int = PERM_SEED,
    boot_seed: int = BOOT_SEED,
) -> dict:
    """Observed LOO-axis AUC + one-sided permutation p (label shuffle, class sizes
    fixed) + bootstrap CI (resample queries; LOO axis recomputed; draws lacking a
    class skipped)."""
    r = np.asarray(r).astype(int)
    n1, n0 = int(r.sum()), int((~r.astype(bool)).sum())
    if n1 < 2 or n0 < 1:
        return {
            "auc": float("nan"),
            "n": len(r),
            "n_refusal": n1,
            "n_nonrefusal": n0,
            "note": "insufficient class sizes for LOO axis",
        }
    auc_obs = _auc(_loo_scores(Y_axis, Y_proj, r), r)
    rng = np.random.default_rng(perm_seed)
    ge = sum(
        1
        for _ in range(n_perm)
        for rp in (rng.permutation(r),)
        if _auc(_loo_scores(Y_axis, Y_proj, rp), rp) >= auc_obs
    )
    brng = np.random.default_rng(boot_seed)
    m = len(r)
    boots = []
    for _ in range(n_boot):
        idx = brng.integers(0, m, size=m)
        rb = r[idx]
        if int(rb.sum()) >= 2 and int((~rb.astype(bool)).sum()) >= 1:
            boots.append(_auc(_loo_scores(Y_axis[idx], Y_proj[idx], rb), rb))
    ci = (
        [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]
        if boots
        else [float("nan"), float("nan")]
    )
    return {
        "auc": float(auc_obs),
        "n": int(m),
        "n_refusal": n1,
        "n_nonrefusal": n0,
        "perm_p_one_sided": float((1 + ge) / (1 + n_perm)),
        "boot_ci95": ci,
        "boot_valid_frac": float(len(boots) / n_boot),
    }


def _effect_and_variance(Y: np.ndarray, r: np.ndarray, mu_train: np.ndarray) -> dict:
    """Full-sample d' along the (non-LOO) mean-difference axis + variance fractions."""
    r = np.asarray(r).astype(bool)
    if r.sum() < 2 or (~r).sum() < 2:
        return {"note": "insufficient class sizes for d'"}
    dmu = Y[r].mean(0) - Y[~r].mean(0)
    norm = float(np.linalg.norm(dmu))
    proj = Y @ (dmu / (norm + 1e-12))
    v1, v0 = proj[r], proj[~r]
    pooled_sd = float(
        np.sqrt(
            ((len(v1) - 1) * v1.var(ddof=1) + (len(v0) - 1) * v0.var(ddof=1))
            / (len(v1) + len(v0) - 2)
        )
    )
    trace_cov = float(((Y - Y.mean(0)) ** 2).sum(1).mean())  # tr(cov)=mean||y-mean||^2
    mean_sstot = float(((Y - mu_train) ** 2).sum(1).mean())  # mean||y-mu_train||^2
    return {
        "d_prime": float(norm / (pooled_sd + 1e-12)),
        "delta_mu_norm": norm,
        "delta_mu_norm_sq": float(norm**2),
        "pooled_within_class_sd_along_axis": pooled_sd,
        "trace_within_bank_cov": trace_cov,
        "mean_per_query_sstot_to_train_mean": mean_sstot,
        "var_frac_of_trace_cov": float(norm**2 / (trace_cov + 1e-12)),
        "var_frac_of_mean_sstot": float(norm**2 / (mean_sstot + 1e-12)),
    }


def _rankvec(a: np.ndarray) -> np.ndarray:
    r = np.argsort(np.argsort(np.asarray(a, dtype=np.float64))).astype(np.float64)
    return r - r.mean()


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra, rb = _rankvec(a), _rankvec(b)
    d = np.sqrt((ra**2).sum() * (rb**2).sum())
    return float((ra * rb).sum() / d) if d > 0 else float("nan")


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64) - np.mean(a)
    b = np.asarray(b, dtype=np.float64) - np.mean(b)
    d = np.sqrt((a**2).sum() * (b**2).sum())
    return float((a * b).sum() / d) if d > 0 else float("nan")


# ── reproduction gates ───────────────────────────────────────────────────────────


def _gate1(npz: dict, comm_pairs: list, committed_h3: dict) -> dict:
    """Gate 1: committed 41-pair arm-matched headline d from the committed npz."""
    groups = npz["A_group_names"].tolist()
    r2 = {}
    for key in ("bank_div", "bank_ctl"):
        ssr = npz[f"{key}_ssres"].astype(np.float64)
        sst = npz[f"{key}_sstot"].astype(np.float64)
        idmap = {str(q): i for i, q in enumerate(npz[f"{key}_ids"].tolist())}
        r2[key] = ({arm: _per_query_r2(ssr, sst, groups, arm) for arm in BANK_ARMS}, idmap)
    ds = []
    for _pid, _cat, qd, qc in comm_pairs:
        di, ci = r2["bank_div"][1].get(qd), r2["bank_ctl"][1].get(qc)
        if di is None or ci is None:
            continue
        do = r2["bank_ctl"][0]["own"][ci] - r2["bank_div"][0]["own"][di]
        de = r2["bank_ctl"][0]["ext_plain"][ci] - r2["bank_div"][0]["ext_plain"][di]
        if np.isfinite(do) and np.isfinite(de):
            ds.append(de - do)
    d = np.asarray(ds)
    return {
        "pass": bool(
            abs(float(d.mean()) - committed_h3["headline_mean_drop_diff"]["mean"]) < 1e-6
            and len(d) == committed_h3["n_pairs"]
        ),
        "recomputed_d": float(d.mean()),
        "committed_d": committed_h3["headline_mean_drop_diff"]["mean"],
        "recomputed_p": _signflip_p(d, N_DRAWS)["p_one_sided"],
        "committed_p": committed_h3["sign_flip_null"]["pooled"]["p_one_sided"],
        "n": len(d),
        "committed_n": committed_h3["n_pairs"],
    }


def main() -> None:  # noqa: C901 — one orchestration fn: gates + Checks A/B/C
    torch.set_num_threads(8)
    from explore_persona_space.experiments.issue_952.ridge_battery import run_ridge_cell

    base = pathlib.Path(__file__).resolve().parent.parent
    out_dir = base / "eval_results" / "issue_952" / "refusal_sanity_check"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = base / "figures" / "issue_952"
    fig_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # ── stage tensors ────────────────────────────────────────────────────────────
    npz_p = _stage(PARENT_PREFIX, PARENT_REV, "per_context_stats.npz")
    own_pool_p = _stage(PARENT_PREFIX, PARENT_REV, "slots_own_L20.pt")
    bank_own_p = _stage(PARENT_PREFIX, PARENT_REV, "slots_bank_own_L20.pt")
    span_paths = {a: _stage(PARENT_PREFIX, PARENT_REV, f"spans_{a}.json") for a in ARMS}
    china_own_p = _stage(CHINA_PREFIX, CHINA_REV, "slots_bank_china_politics_topup_own_L20.pt")
    china_plain_p = _stage(
        CHINA_PREFIX, CHINA_REV, "slots_bank_china_politics_topup_ext_plain_L20.pt"
    )

    npz = dict(np.load(str(npz_p), allow_pickle=False))
    group2lam = dict(zip(npz["A_group_names"].tolist(), npz["A_lam_idx"].tolist(), strict=True))
    own_pool, pool_ids = _load_slots(own_pool_p)
    bank_own_slots, bank_own_ids = _load_slots(bank_own_p)
    china_own_slots, china_ids = _load_slots(china_own_p)
    china_plain_slots, china_plain_ids = _load_slots(china_plain_p)
    assert china_ids == china_plain_ids, "china own/plain id order differs"

    # ── universe / split reproduction (transfer-cell U_A) ──────────────────────────
    split = json.loads((base / "eval_results/issue_952/split_seed952.json").read_text())
    spans = {a: {str(k): v for k, v in json.loads(span_paths[a].read_text()).items()} for a in ARMS}
    spans_arr = {
        a: np.asarray([spans[a][str(c)].get("span", 0) for c in pool_ids], dtype=np.int64)
        for a in ARMS
    }
    u_a = np.all(np.stack([spans_arr[a] >= 32 for a in ARMS]), axis=0)
    pos_of = {c: i for i, c in enumerate(pool_ids)}
    tr_pos = np.asarray([pos_of[c] for c in split["train"] if c in pos_of])
    tr_a = tr_pos[u_a[tr_pos]]
    c_last_tr = own_pool[:, SLOT_IDX["c_last"], :][tr_a].astype(np.float64)
    logger.info("U_A=%d tr_a=%d", int(u_a.sum()), len(tr_a))

    # ── refusal labels (ordered to the china slot id axis) ─────────────────────────
    labmap = _build_label_map(base)
    keep = [q for q in china_ids if q in labmap and labmap[q]["refusal_qwen"] is not None]
    keep_idx = np.asarray([china_ids.index(q) for q in keep])
    rq = np.asarray([labmap[q]["refusal_qwen"] for q in keep], dtype=np.float64)
    rc = np.asarray([labmap[q]["refusal_claude"] for q in keep], dtype=np.float64)
    role = np.asarray([labmap[q]["role"] for q in keep])
    logger.info("china queries with labels: %d / %d", len(keep), len(china_ids))

    # ── answer summaries y(q) = mean over 42 POSITION_SLOTS (fp16 -> f64) ───────────
    pslot_idx = np.asarray([SLOT_IDX[s] for s in POSITION_SLOTS])

    def _summary(slots: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        sub = slots[keep_idx][:, pslot_idx, :].astype(np.float64)  # (n, 42, H)
        nfin = np.isfinite(sub).all(axis=2).sum(axis=1)  # per-query finite-slot count
        return sub.mean(axis=1), nfin  # NaN-propagating mean over the 42 slots

    y_own, nfin_own = _summary(china_own_slots)
    y_claude, nfin_claude = _summary(china_plain_slots)
    c_last_bank = china_own_slots[keep_idx][:, SLOT_IDX["c_last"], :].astype(np.float64)

    # mu_train = own-pool train-mean summary (slot-mean of per-slot train means).
    mu_train = np.stack([_own_train_slot(own_pool, tr_a, s).mean(0) for s in POSITION_SLOTS]).mean(
        0
    )

    # Usable = all 42 POSITION_SLOTS finite in BOTH arms (consistent summary space).
    good = (
        (nfin_own == len(POSITION_SLOTS))
        & (nfin_claude == len(POSITION_SLOTS))
        & np.isfinite(y_own).all(1)
        & np.isfinite(y_claude).all(1)
    )
    n_dropped_nan = int((~good).sum())
    gi = good.nonzero()[0]
    keep_good = [keep[i] for i in gi]
    y_own, y_claude, c_last_bank = y_own[good], y_claude[good], c_last_bank[good]
    rq, rc, role = rq[good], rc[good], role[good]
    r_qwen = (rq >= REFUSAL_THR).astype(int)
    r_claude = (rc >= REFUSAL_THR).astype(int)
    logger.info("china queries usable (all-42 finite): %d (dropped %d)", good.sum(), n_dropped_nan)

    def _label_counts() -> dict:
        out = {"threshold": REFUSAL_THR, "n_total": int(good.sum())}
        for name, rbin, graded in (("qwen", r_qwen, rq), ("claude", r_claude, rc)):
            out[name] = {
                "n_refusal": int(rbin.sum()),
                "n_nonrefusal": int((rbin == 0).sum()),
                "graded_mean": float(graded.mean()),
                "graded_median": float(np.median(graded)),
                "by_role": {
                    rl: {"n": int((role == rl).sum()), "n_refusal": int(rbin[role == rl].sum())}
                    for rl in ("divergent", "control")
                },
            }
        return out

    # ── reproduction gates ─────────────────────────────────────────────────────────
    comm_verif = json.loads(
        (base / "eval_results/issue_952/divergence_bank_verification.json").read_text()
    )
    comm_kept = set(comm_verif["kept_pairs"])
    comm_pairs = [
        (p["pair_id"], p["category"], p["divergent"]["query_id"], p["control"]["query_id"])
        for p in comm_verif["pairs"]
        if p["pair_id"] in comm_kept
        and isinstance(p.get("divergent"), dict)
        and isinstance(p.get("control"), dict)
    ]
    committed_h3 = json.loads((base / "eval_results/issue_952/stats_summary.json").read_text())[
        "h3"
    ]
    gate1 = _gate1(npz, comm_pairs, committed_h3)
    logger.info(
        "[Gate1] recomputed d=%.6f committed %.6f pass=%s",
        gate1["recomputed_d"],
        gate1["committed_d"],
        gate1["pass"],
    )

    # Gate 2 + Gate P: re-fit own map, score committed bank; match npz per-pair own
    # drops (Gate2) AND validate the prediction path reproduces run_ridge_cell (GateP).
    bank_div_rows = [i for i, q in enumerate(bank_own_ids) if str(q).endswith("_div")]
    bank_ctl_rows = [i for i, q in enumerate(bank_own_ids) if not str(q).endswith("_div")]
    bank_c_last_all = bank_own_slots[:, SLOT_IDX["c_last"], :].astype(np.float64)
    preds_bank, ymu_bank = _own_map_predict(c_last_tr, own_pool, tr_a, group2lam, bank_c_last_all)
    tgt_bank = bank_own_slots[:, pslot_idx, :].astype(np.float64)  # (n,42,H) own targets
    fin_bank = np.isfinite(tgt_bank).all(axis=2)  # (n,42)
    ss_res_mine = np.nansum(
        np.where(fin_bank, ((preds_bank - tgt_bank) ** 2).sum(2), np.nan), axis=1
    )
    ss_tot_mine = np.nansum(
        np.where(fin_bank, ((tgt_bank - ymu_bank[None]) ** 2).sum(2), np.nan), 1
    )
    r2_mine = np.where(ss_tot_mine > 1e-12, 1.0 - ss_res_mine / ss_tot_mine, np.nan)

    Ytr, gnames = _stack_answer_targets({"own": own_pool}, tr_a, ("own",))
    div_tgt, _ = _stack_answer_targets({"own": bank_own_slots}, np.asarray(bank_div_rows), ("own",))
    ctl_tgt, _ = _stack_answer_targets({"own": bank_own_slots}, np.asarray(bank_ctl_rows), ("own",))
    res = run_ridge_cell(
        c_last_tr,
        Ytr,
        {
            "bank_div": (bank_c_last_all[bank_div_rows], div_tgt),
            "bank_ctl": (bank_c_last_all[bank_ctl_rows], ctl_tgt),
        },
        group_names=gnames,
        device="cpu",
        allow_train_nan_imputation=True,
    )
    lam_idx = np.asarray([int(group2lam[g]) for g in gnames], dtype=np.int64)
    r2_lib = {}
    for key, rows in (("bank_div", bank_div_rows), ("bank_ctl", bank_ctl_rows)):
        ssr = np.take_along_axis(res.ss_res[key], lam_idx[None, :, None], axis=2)[:, :, 0]
        r2q = _per_query_r2(
            ssr.astype(np.float64), res.ss_tot[key].astype(np.float64), gnames, "own"
        )
        for pos_i, row in enumerate(rows):
            r2_lib[row] = r2q[pos_i]
    gatep_delta = max(
        abs(r2_mine[row] - r2_lib[row])
        for row in range(len(bank_own_ids))
        if np.isfinite(r2_mine[row]) and np.isfinite(r2_lib.get(row, np.nan))
    )

    npz_div_i2r = {str(q): i for i, q in enumerate(npz["bank_div_ids"].tolist())}
    npz_ctl_i2r = {str(q): i for i, q in enumerate(npz["bank_ctl_ids"].tolist())}
    npz_groups = npz["A_group_names"].tolist()
    npz_r2own = {
        key: _per_query_r2(
            npz[f"{key}_ssres"].astype(np.float64),
            npz[f"{key}_sstot"].astype(np.float64),
            npz_groups,
            "own",
        )
        for key in ("bank_div", "bank_ctl")
    }
    id2row = {q: i for i, q in enumerate(bank_own_ids)}
    g2_delta = 0.0
    for _pid, _cat, qd, qc in comm_pairs:
        di, ci, ndi, nci = id2row.get(qd), id2row.get(qc), npz_div_i2r.get(qd), npz_ctl_i2r.get(qc)
        if None in (di, ci, ndi, nci):
            continue
        mine = r2_mine[ci] - r2_mine[di]
        comm = npz_r2own["bank_ctl"][nci] - npz_r2own["bank_div"][ndi]
        if np.isfinite(mine) and np.isfinite(comm):
            g2_delta = max(g2_delta, abs(mine - comm))
    gate2 = {
        "pass": bool(g2_delta < REPRO_TOL),
        "max_abs_delta_own_drop": float(g2_delta),
        "tol": REPRO_TOL,
    }
    gatep = {"pass": bool(gatep_delta < 1e-6), "max_abs_delta_pooled_r2": float(gatep_delta)}
    logger.info(
        "[Gate2] own-drop maxdelta=%.5f pass=%s | [GateP] pred-vs-lib R2 maxdelta=%.2e pass=%s",
        g2_delta,
        gate2["pass"],
        gatep_delta,
        gatep["pass"],
    )
    if not (gate1["pass"] and gate2["pass"] and gatep["pass"]):
        raise RuntimeError(
            f"REPRODUCTION GATE FAILED: g1={gate1['pass']} g2={gate2['pass']} "
            f"gp={gatep['pass']} — refusing to proceed"
        )

    # ── yhat for the china bank (own map) ──────────────────────────────────────────
    preds_china, _ = _own_map_predict(c_last_tr, own_pool, tr_a, group2lam, c_last_bank)
    yhat = preds_china.mean(axis=1)  # (n, H) predictions always finite

    eval_sets = {
        "all": np.arange(len(r_qwen)),
        "divergent": np.where(role == "divergent")[0],
        "control": np.where(role == "control")[0],
    }

    # ── Check A: are refusal activations distinct? (own arm; qwen labels) ───────────
    checkA = {"description": "LOO mean-difference refusal axis on OWN-arm answer summaries y(q)"}
    for name, idx in eval_sets.items():
        res_a = _loo_auc_with_perm(y_own[idx], y_own[idx], r_qwen[idx])
        res_a["effect_variance"] = _effect_and_variance(y_own[idx], r_qwen[idx], mu_train)
        if int(r_qwen[idx].sum()) >= 2 and int((r_qwen[idx] == 0).sum()) >= 1:
            res_a["spearman_score_vs_graded"] = _spearman(
                _loo_scores(y_own[idx], y_own[idx], r_qwen[idx]), rq[idx]
            )
        checkA[name] = res_a
    checkA["ext_plain_reference_refusal_claude"] = {
        name: _loo_auc_with_perm(y_claude[idx], y_claude[idx], r_claude[idx])
        for name, idx in eval_sets.items()
    }

    # ── Check B: does the map predict refusal? (yhat on actual-y axes) ──────────────
    checkB = {"description": "own-map predicted summaries yhat(q) projected on actual-y LOO axes"}
    for name, idx in eval_sets.items():
        res_b = _loo_auc_with_perm(y_own[idx], yhat[idx], r_qwen[idx])
        if res_b["n_refusal"] >= 2 and res_b["n_nonrefusal"] >= 1:
            s_act = _loo_scores(y_own[idx], y_own[idx], r_qwen[idx])
            s_pred = _loo_scores(y_own[idx], yhat[idx], r_qwen[idx])
            res_b["pearson_pred_vs_actual_proj"] = _pearson(s_pred, s_act)
            res_b["spearman_pred_vs_actual_proj"] = _spearman(s_pred, s_act)
        checkB[name] = res_b

    # paired closeness on genuine-divergence cells (Qwen refuses, Claude answers)
    gd = np.where((role == "divergent") & (r_qwen == 1) & (r_claude == 0))[0]
    cos_own, cos_claude, r2c_own, r2c_claude, gd_ids = [], [], [], [], []
    for i in gd:
        p, yo, yc = yhat[i] - mu_train, y_own[i] - mu_train, y_claude[i] - mu_train
        cos_own.append(float(p @ yo / ((np.linalg.norm(p) + 1e-12) * (np.linalg.norm(yo) + 1e-12))))
        cos_claude.append(
            float(p @ yc / ((np.linalg.norm(p) + 1e-12) * (np.linalg.norm(yc) + 1e-12)))
        )
        r2c_own.append(float(1 - ((yhat[i] - y_own[i]) ** 2).sum() / ((yo**2).sum() + 1e-12)))
        r2c_claude.append(float(1 - ((yhat[i] - y_claude[i]) ** 2).sum() / ((yc**2).sum() + 1e-12)))
        gd_ids.append(keep_good[i])
    if gd.size >= 1:
        dcos = np.asarray(cos_own) - np.asarray(cos_claude)
        checkB["paired_closeness_genuine_divergence"] = {
            "n": int(gd.size),
            "definition": "divergent china queries with refusal_qwen>=50 AND refusal_claude<50",
            "mean_cos_to_own": float(np.mean(cos_own)),
            "mean_cos_to_claude": float(np.mean(cos_claude)),
            "paired_mean_cos_own_minus_claude": float(dcos.mean()),
            "sign_flip_p": _signflip_p(dcos, N_DRAWS) if gd.size >= 2 else None,
            "mean_r2close_to_own": float(np.mean(r2c_own)),
            "mean_r2close_to_claude": float(np.mean(r2c_claude)),
            "per_query": [
                {
                    "query_id": qid,
                    "cos_own": co,
                    "cos_claude": cc,
                    "r2close_own": ro,
                    "r2close_claude": rc_,
                }
                for qid, co, cc, ro, rc_ in zip(
                    gd_ids, cos_own, cos_claude, r2c_own, r2c_claude, strict=True
                )
            ],
        }
    else:
        checkB["paired_closeness_genuine_divergence"] = {
            "n": 0,
            "note": "no genuine-divergence cells",
        }

    # ── Check C: is refusal linearly decodable from c_last? (own arm) ───────────────
    checkC = {"description": "LOO mean-difference refusal axis on own-arm context c_last(q)"}
    for name, idx in eval_sets.items():
        checkC[name] = _loo_auc_with_perm(c_last_bank[idx], c_last_bank[idx], r_qwen[idx])

    # raw LOO scores for the "all" set (for the projection scatter figure)
    s_act_all = _loo_scores(y_own, y_own, r_qwen)
    s_pred_all = _loo_scores(y_own, yhat, r_qwen)

    out = {
        "description": (
            "#952 refusal sanity checks (Dan comment). China bank ONLY, layer 20. "
            "Check A: refusal-activation distinctness (own arm); Check B: does the "
            "pool-trained own map predict refusal; Check C: c_last decodability ceiling."
        ),
        "layer": 20,
        "refusal_threshold": REFUSAL_THR,
        "refusal_label_rule": (
            "mean of 3 refusal-rubric judge draws in [0,100] (run_952.REFUSAL_RUBRIC + "
            "_aggregate_draws; higher=more refusal; malformed dropped never coerced); "
            "boolean = mean >= 50; parent-kept labels from divergence_bank_verification.json, "
            "new-pair labels from china_topup_verification.json"
        ),
        "n_china_queries_captured": len(china_ids),
        "n_with_labels": len(keep),
        "n_dropped_nan_summary": n_dropped_nan,
        "n_usable": int(good.sum()),
        "label_counts": _label_counts(),
        "reproduction_gate": {
            "gate1_stats": gate1,
            "gate2_own_fit": gate2,
            "gateP_prediction_path": gatep,
        },
        "check_A_refusal_distinctness": checkA,
        "check_B_map_predicts_refusal": checkB,
        "check_C_clast_decodability": checkC,
        "n_perm": N_PERM,
        "n_boot": N_BOOT,
        "tensor_sources": {"parent_rev": PARENT_REV, "china_rev": CHINA_REV},
        "git_commit": _sha(),
        "wall_seconds": round(time.time() - t0, 1),
        "generated_at": datetime.datetime.now(datetime.UTC).isoformat(),
    }
    (out_dir / "refusal_sanity.json").write_text(json.dumps(out, indent=1))
    logger.info("Wrote %s", out_dir / "refusal_sanity.json")

    _make_figures(out, s_act_all, s_pred_all, r_qwen, role, fig_dir)
    logger.info("done in %.1fs", time.time() - t0)


def _make_figures(
    out: dict,
    s_act: np.ndarray,
    s_pred: np.ndarray,
    r_qwen: np.ndarray,
    role: np.ndarray,
    fig_dir: pathlib.Path,
) -> None:
    """Two figures: (1) AUC summary bars with bootstrap CIs across Checks A/B/C;
    (2) actual vs predicted refusal-axis projection scatter, colored by refusal
    label, div/ctl markers. Colorblind-safe, error bars, no text overlays."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    try:
        from explore_persona_space.analysis.paper_plots import apply_paper_style

        apply_paper_style()
    except Exception:
        pass

    # Figure 1 — AUC summary bars.
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    spec = [
        ("A: activations (all)", out["check_A_refusal_distinctness"]["all"], "#0072B2"),
        ("A: activations (div)", out["check_A_refusal_distinctness"]["divergent"], "#0072B2"),
        ("B: map preds (all)", out["check_B_map_predicts_refusal"]["all"], "#009E73"),
        ("B: map preds (div)", out["check_B_map_predicts_refusal"]["divergent"], "#009E73"),
        ("C: c_last (all)", out["check_C_clast_decodability"]["all"], "#CC79A7"),
        ("C: c_last (div)", out["check_C_clast_decodability"]["divergent"], "#CC79A7"),
    ]
    labs, aucs, los, his, cols = [], [], [], [], []
    for lab, d, col in spec:
        if not np.isfinite(d.get("auc", np.nan)):
            continue
        labs.append(lab)
        aucs.append(d["auc"])
        ci = d.get("boot_ci95", [np.nan, np.nan])
        los.append(max(0.0, d["auc"] - ci[0]) if np.isfinite(ci[0]) else 0.0)
        his.append(max(0.0, ci[1] - d["auc"]) if np.isfinite(ci[1]) else 0.0)
        cols.append(col)
    x = np.arange(len(labs))
    ax.bar(x, aucs, yerr=[los, his], capsize=4, color=cols, edgecolor="black", linewidth=0.6)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(labs, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("ROC AUC (refusal vs LOO refusal axis)")
    ax.set_ylim(0.0, 1.02)
    ax.set_title("China bank, layer 20: refusal separability by axis source (95% bootstrap CI)")
    fig.tight_layout()
    fig.savefig(fig_dir / "refusal_sanity_auc.png", dpi=150)
    plt.close(fig)

    # Figure 2 — actual vs predicted refusal-axis projection scatter (all queries).
    fig, ax = plt.subplots(figsize=(6.0, 5.4))
    ref = r_qwen.astype(bool)
    div = role == "divergent"
    for mask, marker, dl in ((div, "o", "divergent"), (~div, "s", "control")):
        for rmask, col, rl in (
            (ref, "#D55E00", "refusal (Qwen)"),
            (~ref, "#0072B2", "non-refusal"),
        ):
            m = mask & rmask
            if m.any():
                ax.scatter(
                    s_act[m],
                    s_pred[m],
                    marker=marker,
                    c=col,
                    s=42,
                    edgecolor="black",
                    linewidth=0.4,
                    alpha=0.85,
                    label=f"{dl}, {rl}",
                )
    ax.axvline(0, color="gray", lw=0.8, ls=":")
    ax.axhline(0, color="gray", lw=0.8, ls=":")
    ax.set_xlabel("actual refusal-axis projection  <y - mu, Δμ>")
    ax.set_ylabel("predicted refusal-axis projection  <yhat - mu, Δμ>")
    ax.set_title("China bank, layer 20: own-map prediction vs actual on the refusal axis")
    ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(fig_dir / "refusal_sanity_projections.png", dpi=150)
    plt.close(fig)
    logger.info("[fig] wrote refusal_sanity_auc.png + refusal_sanity_projections.png")


if __name__ == "__main__":
    main()
