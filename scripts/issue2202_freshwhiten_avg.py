#!/usr/bin/env python3
"""Issue #2202 inline free-analysis round ``freshwhiten-avg`` (user-chat carve-out).

Two asks on the #1738 context->answer map (context arm, layer-19 ridge,
held-out n=9,941, dim 3,584), computed on BANKED artifacts only:

1. The fresh-draw retrievability reference (banked 0.943, raw-euclidean:
   each of the 1,988 resample-covered contexts' 4 fresh on-policy answer
   draws queries the full 9,941-answer held-out pool; target = that
   context's original held-out answer; per-context rank==1 share over the
   4 draws, averaged over contexts) recomputed as a SANITY gate, then the
   SAME definition under the whitened-cosine convention (z = L^-1(x - mu_A)
   with the task-locked shrunk train-answer covariance Cholesky from
   ``whiten_stats.npz``; cosine in z-space).
2. Draw-averaged answer targets: replace each covered row's pool entry
   with the mean of its 5 on-policy draws (original + 4 fresh); map
   acc@1 on the covered rows under raw-euclidean AND whitened-cosine,
   beside the matched single-draw acc@1 on the same 1,988 rows.
   Secondary query-side variant: the averaged vector as QUERY retrieving
   the original single-draw answer from the unchanged pool.

Scope-extension addendum (same round): two extra similarity conventions —
``r2_cand_norm`` (candidate-normalized R^2, per-candidate mean-offset
denominator; the mean-proximity hub penalty) and ``pearson_r`` (signed
per-vector-demeaned cosine) — reported as the full-pool acc@1 battery on
all 9,941 rows plus the covered-row single-draw / draw-averaged reads.

Conventions (mid-rank ties, chunked GEMM batteries, whitening transform)
are REUSED from ``scripts/issue2202_failchar.py`` — this script imports it
and calls the same functions; no re-derivation. Analysis-only: no new
training, generation, or capture. All staging lands on the data disk.
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()  # shared-VM thread caps must bind BEFORE numpy/torch import (#847)

import issue2202_failchar as FC  # noqa: E402
import issue1738_characterize as CH  # noqa: E402  (_load_kresample_v)
import numpy as np  # noqa: E402
from scipy.linalg import solve_triangular  # noqa: E402

from explore_persona_space.orchestrate import hub  # noqa: E402

STAGE = Path("/mnt/eps-data/thomasjiralerspong/issue2202_freshwhiten")
OUT_DIR = PROJECT_ROOT / "eval_results" / "issue_2202" / "freshwhiten_avg"
LAYER = FC.LAYER  # 19
K_DRAWS = 4
N_COVERED = 1_988
BANKED_ATTRIBUTION = PROJECT_ROOT / "eval_results" / "issue_2202" / "attribution.json"
BANKED_RANKS_CSV = PROJECT_ROOT / "eval_results" / "issue_2202" / "percontext_ranks.csv"
MAPPING_ISSUES = (1738, 1901, 2202, 1739)
# Addendum 2: banked #1738 NONLINEAR held-out predictions (same 9,941-row pool),
# ridge recomputed beside them with the identical machinery as the reference.
NONLINEAR_PRED_FILES = {
    "ridge": ("context_L19_ridge.npz", "linear ridge map (the #1738 primary; reference)"),
    "mlp_w8192": ("context_L19_mlp_w8192.npz", "MLP map, hidden width 8192"),
    "mlp_w8192_seed43": ("context_L19_mlp_w8192_seed43.npz", "MLP map, width 8192, seed 43"),
    "krr_nystrom": ("context_L19_krr_nystrom.npz", "kernel ridge regression, Nystrom approx"),
    "residual_skip": (
        "context_L19_residual_skip.npz",
        "residual map (identity skip + learned correction)",
    ),
}
ABSENCE_PATTERN = re.compile(
    r"(draw[\s_-]?averag|averag\w*[\s_-]?(answer|target|draw)|avg[\s_-]?draws?|draws?[\s_-]?avg)",
    re.IGNORECASE,
)
TEXT_EXTS = {".json", ".csv", ".md", ".py", ".txt", ".yaml", ".yml"}
MAX_GREP_BYTES = 32 * 1024 * 1024


def stage_inputs() -> dict:
    """Stage the four banked inputs; parent-prefix files at the #2202 pin,
    #2202's own derived tensors at the data repo's current head (they
    post-date the pin). Returns the resolved revisions for the meta block."""
    from huggingface_hub import HfApi

    STAGE.mkdir(parents=True, exist_ok=True)
    api = HfApi()
    for rel, dest in (
        (f"{FC.PARENT_PREFIX}/analysis_tensors/pred16/context_L{LAYER}_ridge.npz", "pred16.npz"),
        (f"{FC.PARENT_PREFIX}/analysis_tensors/y_holdout/L{LAYER}.npz", "y_holdout_L19.npz"),
    ):
        hub.stage_hub_file(FC.C.HF_DATA_REPO, rel, STAGE / dest, revision=FC.HF_PIN)
    kres_files = hub.list_hf_files_under_path(
        api,
        FC.C.HF_DATA_REPO,
        f"{FC.PARENT_PREFIX}/kresample",
        repo_type="dataset",
        revision=FC.HF_PIN,
    )
    kres_pt = [f for f in kres_files if f.endswith(".pt")]
    assert kres_pt, f"no kresample .pt shards under {FC.PARENT_PREFIX}/kresample at {FC.HF_PIN}"
    for f in kres_pt:
        hub.stage_hub_file(
            FC.C.HF_DATA_REPO, f, STAGE / "kresample" / Path(f).name, revision=FC.HF_PIN
        )
    pred_names = {
        Path(f).name
        for f in hub.list_hf_files_under_path(
            api,
            FC.C.HF_DATA_REPO,
            f"{FC.PARENT_PREFIX}/analysis_tensors/pred16",
            repo_type="dataset",
            revision=FC.HF_PIN,
        )
    }
    nonlinear_staged: dict[str, str] = {}
    for tag, (fname, _desc) in NONLINEAR_PRED_FILES.items():
        if tag == "ridge":
            continue  # already staged as pred16.npz
        if fname not in pred_names:
            nonlinear_staged[tag] = "ABSENT"
            continue
        hub.stage_hub_file(
            FC.C.HF_DATA_REPO,
            f"{FC.PARENT_PREFIX}/analysis_tensors/pred16/{fname}",
            STAGE / "nonlinear" / fname,
            revision=FC.HF_PIN,
        )
        nonlinear_staged[tag] = fname
    head_sha = api.repo_info(FC.C.HF_DATA_REPO, repo_type="dataset").sha
    for name in ("whiten_stats.npz", "kresample_ranks.npz"):
        hub.stage_hub_file(
            FC.C.HF_DATA_REPO,
            f"{FC.HF_PREFIX_2202}/analysis_tensors/{name}",
            STAGE / name,
            revision=head_sha,
        )
    return {
        "parent_revision_pin": FC.HF_PIN,
        "issue2202_tensors_revision": head_sha,
        "kresample_shards": [Path(f).name for f in kres_pt],
        "nonlinear_staged": nonlinear_staged,
    }


def load_inputs() -> dict:
    """Load + shape/alignment-assert every input (fail fast on any mismatch)."""
    pd_ = np.load(STAGE / "pred16.npz")
    yd = np.load(STAGE / "y_holdout_L19.npz")
    pred = pd_["pred16"].astype(np.float64)
    y16 = yd["y16"].astype(np.float64)
    pci = np.asarray(pd_["ci"], dtype=np.int64)
    yci = np.asarray(yd["ci"], dtype=np.int64)
    assert pred.shape == y16.shape == (FC.EXPECTED_N, FC.H_DIM), (pred.shape, y16.shape)
    assert (pci == yci).all(), "pred16/y_holdout ci misalign"
    assert np.array_equal(pd_["fingerprint"], yd["fingerprint"]), (
        "pred16/y_holdout assembly fingerprint mismatch"
    )

    kns = SimpleNamespace(
        local_kresample_dir=str(STAGE / "kresample"), scratch=str(STAGE / "scratch"), hf_prefix=""
    )
    kci, vres = CH._load_kresample_v(kns, [LAYER])  # (n, K, 1, H) fp32
    assert vres.shape == (N_COVERED, K_DRAWS, 1, FC.H_DIM), vres.shape
    draws = vres[:, :, 0, :].astype(np.float64)  # (1988, 4, H)

    bk = np.load(STAGE / "kresample_ranks.npz")
    bci = np.asarray(bk["ci"], dtype=np.int64)
    assert (bci == kci).all(), "banked kresample_ranks ci order != shard ci order"
    banked_kranks = np.asarray(bk["ranks"], dtype=np.float64)
    assert banked_kranks.shape == (N_COVERED, K_DRAWS), banked_kranks.shape

    wz = np.load(STAGE / "whiten_stats.npz")
    mu_a = np.asarray(wz["mu_A"], dtype=np.float64)
    ell = np.asarray(wz["L"], dtype=np.float64)
    assert mu_a.shape == (FC.H_DIM,) and ell.shape == (FC.H_DIM, FC.H_DIM), (
        mu_a.shape,
        ell.shape,
    )
    pos_of = {int(c): p for p, c in enumerate(pci.tolist())}
    missing = [int(c) for c in kci if int(c) not in pos_of]
    assert not missing, f"{len(missing)} kresample cis not in holdout pool"
    pos = np.asarray([pos_of[int(c)] for c in kci], dtype=np.int64)
    return {
        "pred": pred,
        "y16": y16,
        "pci": pci,
        "kci": kci,
        "draws": draws,
        "banked_kranks": banked_kranks,
        "banked_s": np.asarray(bk["s"], dtype=np.float64),
        "mu_a": mu_a,
        "ell": ell,
        "whiten_lam": float(wz["lam"]),
        "whiten_n_train": int(wz["n_train"]),
        "pos": pos,
    }


def freshdraw_reference(dr: np.ndarray, pool: np.ndarray, pos: np.ndarray, metric: str, tag: str):
    """acc1_ceiling under one convention: every fresh draw queries the full
    pool for its context's ORIGINAL held-out answer (the failchar P0.5
    reduction: per-context rank==1 share over K draws, then the context mean
    — identical to the pooled per-draw mean at fixed K)."""
    n_k = dr.shape[0]
    q = dr.reshape(n_k * K_DRAWS, FC.H_DIM)
    ti = np.repeat(pos, K_DRAWS)
    kranks, _, _ = FC.ranks_of_targets(q, pool, ti, metric, phase=f"fw-{tag}")
    kranks = kranks.reshape(n_k, K_DRAWS)
    s_i = (kranks == 1.0).mean(axis=1)
    return float(s_i.mean()), kranks, s_i


def _row_demean(x: np.ndarray) -> np.ndarray:
    """Per-VECTOR demean across the 3,584 dimensions (Pearson-r prep: cosine of
    row-demeaned vectors == Pearson correlation across dimensions). Distinct
    from the banked cent_cos, which centers per-DIMENSION by family pool means."""
    x = np.asarray(x, dtype=np.float64)
    return x - x.mean(axis=1, keepdims=True)


def ranks_r2_cand_norm(
    pred: np.ndarray,
    pool: np.ndarray,
    true_idx: np.ndarray,
    chunk: int = 1024,
    phase: str = "r2cn",
) -> np.ndarray:
    """Mid-ranks under candidate-normalized R^2: score_j = 1 - ||y_j - yhat||^2 /
    ||y_j - mean(pool)||^2, ranked DESCENDING — equivalently the squared-euclidean
    battery divided per-CANDIDATE by its own mean-offset denominator, ranked
    ascending. NOT rank-equivalent to euclidean (the per-candidate denominator is
    the mean-proximity hub penalty). Same mid-rank + relative tie tolerance as
    ``issue2202_failchar.ranks_of_targets``; the pool mean is the mean of the
    battery's own pool (the modified pool when targets are draw-averaged)."""
    pred = np.asarray(pred, dtype=np.float64)
    pool = np.asarray(pool, dtype=np.float64)
    den = ((pool - pool.mean(axis=0)) ** 2).sum(axis=1)
    assert den.min() > 0, "degenerate candidate denominator ||y - pool_mean||^2 == 0"
    n = pred.shape[0]
    ranks = np.empty(n, dtype=np.float64)
    t0 = time.time()
    n_chunks = (n + chunk - 1) // chunk
    for k, s in enumerate(range(0, n, chunk)):
        e = min(n, s + chunk)
        dmat = FC._pairwise_dist(pred[s:e], pool, "euclidean") / den[None, :]
        dt = dmat[np.arange(e - s), true_idx[s:e]]
        tol = 1e-9 * np.maximum(np.abs(dt)[:, None], 1e-12)
        closer = (dmat < dt[:, None] - tol).sum(axis=1)
        tied = (np.abs(dmat - dt[:, None]) <= tol).sum(axis=1) - 1
        ranks[s:e] = 1.0 + closer + 0.5 * tied
        print(
            f"[{phase}] unit {k + 1}/{n_chunks} rows={s}:{e} elapsed={time.time() - t0:.1f}s",
            flush=True,
        )
    return ranks


def absence_sweep() -> dict:
    """Bounded sweep for any PRIOR draw-averaged-target acc@1 in the mapping
    line: content grep over the four mapping issues' scripts / eval_results /
    task bodies, a bounded listing of worktree copies of issue_2202 eval dirs,
    and the HF filename listings of the two tensor prefixes."""
    roots: list[Path] = []
    for n in MAPPING_ISSUES:
        roots.extend(sorted(PROJECT_ROOT.glob(f"eval_results/issue_{n}")))
        roots.extend(sorted(PROJECT_ROOT.glob(f"scripts/issue{n}_*.py")))
        roots.extend(sorted(PROJECT_ROOT.glob(f"tasks/*/{n}/body.md")))
    hits: list[dict] = []
    n_scanned = 0
    for root in roots:
        files = [root] if root.is_file() else sorted(p for p in root.rglob("*") if p.is_file())
        for p in files:
            if p.suffix.lower() not in TEXT_EXTS or p.stat().st_size > MAX_GREP_BYTES:
                continue
            if str(p.relative_to(PROJECT_ROOT)) == "scripts/issue2202_freshwhiten_avg.py":
                continue  # this round's own script is not prior art
            n_scanned += 1
            text = p.read_text(encoding="utf-8", errors="replace")
            for m in ABSENCE_PATTERN.finditer(text):
                line_no = text.count("\n", 0, m.start()) + 1
                line = text.splitlines()[line_no - 1].strip()[:200]
                hits.append(
                    {"file": str(p.relative_to(PROJECT_ROOT)), "line": line_no, "text": line}
                )
                if len([h for h in hits if h["file"] == str(p.relative_to(PROJECT_ROOT))]) >= 5:
                    break
    worktree_listings = sorted(
        str(p)
        for p in Path(PROJECT_ROOT / ".claude" / "worktrees").glob("*/eval_results/issue_2202*")
    )
    from huggingface_hub import HfApi

    api = HfApi()
    hf_name_hits = []
    for prefix in (f"{FC.PARENT_PREFIX}/analysis_tensors", FC.HF_PREFIX_2202):
        for f in hub.list_hf_files_under_path(api, FC.C.HF_DATA_REPO, prefix, repo_type="dataset"):
            if ABSENCE_PATTERN.search(Path(f).name):
                hf_name_hits.append(f)
    return {
        "n_files_scanned": n_scanned,
        "pattern": ABSENCE_PATTERN.pattern,
        "content_hits": hits,
        "worktree_issue2202_eval_dirs": worktree_listings,
        "hf_filename_hits": hf_name_hits,
    }


def main() -> int:
    t0 = time.time()
    revisions = stage_inputs()
    d = load_inputs()
    pred, y16, pos, draws = d["pred"], d["y16"], d["pos"], d["draws"]
    n_pool = y16.shape[0]
    banked_ref = json.loads(BANKED_ATTRIBUTION.read_text())["acc1_ceiling"]

    def _wh(x: np.ndarray) -> np.ndarray:
        return solve_triangular(d["ell"], (np.asarray(x, np.float64) - d["mu_a"]).T, lower=True).T

    # Leg A — SANITY: raw-euclidean fresh-draw reference, reconcile vs banked.
    ref_raw, kranks_raw, s_raw = freshdraw_reference(draws, y16, pos, "euclidean", "raw")
    rank_delta_max = float(np.abs(kranks_raw - d["banked_kranks"]).max())
    s_delta_max = float(np.abs(s_raw - d["banked_s"]).max())
    print(
        f"[legA] ref_raw={ref_raw:.6f} banked={banked_ref:.6f} "
        f"delta={ref_raw - banked_ref:+.2e} max|rank-banked|={rank_delta_max:.3g}",
        flush=True,
    )

    # Leg B — fresh-draw reference under whitened-cosine.
    y16w = _wh(y16)
    ref_wcos, kranks_wcos, s_wcos = freshdraw_reference(
        _wh(draws.reshape(-1, FC.H_DIM)).reshape(draws.shape), y16w, pos, "cosine", "wcos"
    )
    print(f"[legB] ref_whiten_cos={ref_wcos:.6f}", flush=True)

    # Addendum — full-pool acc@1 battery under the TWO extra conventions
    # (all 9,941 held-out rows; queries = map predictions, targets = self).
    full_idx = np.arange(n_pool)
    pred_dm = _row_demean(pred)
    y16_dm = _row_demean(y16)
    extra_fullpool: dict[str, dict] = {}
    extra_fullpool["r2_cand_norm"] = FC.ranks_summary(
        ranks_r2_cand_norm(pred, y16, full_idx, phase="full-r2cn"), n_pool
    )
    r, _, _ = FC.ranks_of_targets(pred_dm, y16_dm, full_idx, "cosine", phase="full-pearson")
    extra_fullpool["pearson_r"] = FC.ranks_summary(r, n_pool)

    # Addendum 2 — banked NONLINEAR map predictions: identical full-pool battery
    # (raw-euclidean + whitened-cosine), ridge recomputed beside them.
    nonlinear: dict[str, dict] = {}
    for tag, (fname, desc) in NONLINEAR_PRED_FILES.items():
        if tag == "ridge":
            p_np = pred
        elif revisions["nonlinear_staged"].get(tag) == "ABSENT":
            nonlinear[tag] = {"estimator": desc, "status": f"SKIPPED — {fname} absent at pin"}
            continue
        else:
            z = np.load(STAGE / "nonlinear" / fname)
            p_np = z["pred16"].astype(np.float64)  # fp16 on disk; cast like the ridge pred16
            zci = np.asarray(z["ci"], dtype=np.int64)
            assert p_np.shape == y16.shape, (tag, p_np.shape)
            assert (zci == d["pci"]).all(), f"{tag} ci misalign vs pred16/y_holdout order"
        rec: dict = {"estimator": desc, "file": fname}
        r, _, _ = FC.ranks_of_targets(p_np, y16, full_idx, "euclidean", phase=f"nl-{tag}-raw")
        rec["raw_euclidean"] = FC.ranks_summary(r, n_pool)
        r, _, _ = FC.ranks_of_targets(_wh(p_np), y16w, full_idx, "cosine", phase=f"nl-{tag}-wcos")
        rec["whiten_cos"] = FC.ranks_summary(r, n_pool)
        nonlinear[tag] = rec

    # Leg C — matched SINGLE-draw map acc on the covered rows (original pool).
    predc = pred[pos]
    predcw = _wh(predc)
    single: dict[str, dict] = {}
    single_ranks: dict[str, np.ndarray] = {}
    for tag, (q, pool, metric) in {
        "raw_euclidean": (predc, y16, "euclidean"),
        "whiten_cos": (predcw, y16w, "cosine"),
        "pearson_r": (pred_dm[pos], y16_dm, "cosine"),
    }.items():
        r, _, _ = FC.ranks_of_targets(q, pool, pos, metric, phase=f"single-{tag}")
        single[tag] = FC.ranks_summary(r, n_pool)
        single_ranks[tag] = r
    single["r2_cand_norm"] = FC.ranks_summary(
        ranks_r2_cand_norm(predc, y16, pos, phase="single-r2cn"), n_pool
    )

    # Reconcile single-draw ranks against the banked percontext_ranks.csv.
    import csv as _csv

    csv_ranks = {"raw_euclidean": {}, "whiten_cos": {}}
    with open(BANKED_RANKS_CSV, encoding="utf-8") as f:
        for row in _csv.DictReader(f):
            ci = int(row["ci"])
            csv_ranks["raw_euclidean"][ci] = float(row["rank_raw_euclidean"])
            csv_ranks["whiten_cos"][ci] = float(row["rank_whiten_cos"])
    csv_delta = {
        tag: float(
            np.abs(single_ranks[tag] - np.asarray([csv_ranks[tag][int(c)] for c in d["kci"]])).max()
        )
        for tag in ("raw_euclidean", "whiten_cos")
    }
    print(f"[legC] single-draw csv reconciliation max|delta| = {csv_delta}", flush=True)

    # Leg D — HEADLINE: draw-averaged TARGET (pool entry -> mean of 5 draws).
    avg = (y16[pos] + draws.sum(axis=1)) / (1 + K_DRAWS)
    pool_mod = y16.copy()
    pool_mod[pos] = avg
    pool_modw = y16w.copy()
    pool_modw[pos] = _wh(avg)
    pool_mod_dm = _row_demean(pool_mod)
    avg_target: dict[str, dict] = {}
    for tag, (q, pool, metric) in {
        "raw_euclidean": (predc, pool_mod, "euclidean"),
        "whiten_cos": (predcw, pool_modw, "cosine"),
        "pearson_r": (pred_dm[pos], pool_mod_dm, "cosine"),
    }.items():
        r, _, _ = FC.ranks_of_targets(q, pool, pos, metric, phase=f"avgtarget-{tag}")
        avg_target[tag] = FC.ranks_summary(r, n_pool)
    avg_target["r2_cand_norm"] = FC.ranks_summary(
        ranks_r2_cand_norm(predc, pool_mod, pos, phase="avgtarget-r2cn"), n_pool
    )

    # Leg E — SECONDARY: averaged vector as QUERY, unchanged pool.
    avgw = _wh(avg)
    avg_query: dict[str, dict] = {}
    for tag, (q, pool, metric) in {
        "raw_euclidean": (avg, y16, "euclidean"),
        "whiten_cos": (avgw, y16w, "cosine"),
    }.items():
        r, _, _ = FC.ranks_of_targets(q, pool, pos, metric, phase=f"avgquery-{tag}")
        avg_query[tag] = FC.ranks_summary(r, n_pool)

    sweep = absence_sweep()
    summary = {
        "round": "freshwhiten-avg (user-chat inline free-analysis, task #2202)",
        "conventions": {
            "rank": "mid-rank with 1e-9 relative tie tolerance (issue2202_failchar.ranks_of_targets)",
            "acc_at_1": "(rank <= 1).mean() — a tie at the top (mid-rank 1.5) counts as failure",
            "whiten_cos": (
                "z = L^-1(x - mu_A), task-locked shrunk train-ANSWER covariance Cholesky "
                "(whiten_stats.npz, lam={lam}, n_train={nt}); cosine in z-space; mu_A applied to "
                "answers, draws, averaged targets AND map predictions (all answer-family)"
            ).format(lam=d["whiten_lam"], nt=d["whiten_n_train"]),
            "fresh_draw_reference": (
                "per covered context, each of K=4 fresh on-policy answer draws queries the FULL "
                "9,941-answer held-out pool; target = that context's ORIGINAL held-out answer; "
                "success = rank == 1.0; per-context share over draws, then mean over the 1,988 "
                "covered contexts (the banked attribution.json acc1_ceiling definition)"
            ),
            "draw_averaged_target": (
                "pool entry of each covered row replaced by mean(original + 4 fresh draws); pool "
                "size stays 9,941; queries = ridge map predictions on the covered rows"
            ),
            "r2_cand_norm": (
                "candidate-normalized R^2: score_j = 1 - ||y_j - yhat||^2 / ||y_j - mean(pool)||^2 "
                "with the battery's OWN pool mean (the full 9,941-row held-out pool mean; the "
                "modified pool's own mean in the draw-averaged-target read), ranked descending; "
                "NOT rank-equivalent to euclidean — the per-candidate denominator penalizes "
                "mean-proximal hub candidates"
            ),
            "pearson_r": (
                "signed Pearson correlation across the 3,584 dimensions between prediction and "
                "each candidate = cosine after per-VECTOR demeaning (each vector centered by its "
                "own across-dimension mean); differs from the banked cent_cos, which centers "
                "per-DIMENSION by family pool means"
            ),
            "fixed_normalizer_r2_not_run": (
                "per-pair R^2 with a FIXED normalizer is 1 - ||y - yhat||^2 / const — a monotone "
                "transform of squared euclidean distance, so its rankings and acc@1 are identical "
                "to the euclidean battery; deliberately not run"
            ),
        },
        "n_covered": int(N_COVERED),
        "n_pool": int(n_pool),
        "k_draws": int(K_DRAWS),
        "fresh_draw_reference": {
            "raw_euclidean_recomputed": ref_raw,
            "raw_euclidean_banked": banked_ref,
            "raw_delta": ref_raw - banked_ref,
            "raw_rank_delta_max_vs_banked": rank_delta_max,
            "raw_s_delta_max_vs_banked": s_delta_max,
            "whiten_cos": ref_wcos,
            "per_context_share_mean": {
                "raw": float(s_raw.mean()),
                "whiten_cos": float(s_wcos.mean()),
            },
            "pooled_draw_acc1": {
                "raw": float((kranks_raw == 1.0).mean()),
                "whiten_cos": float((kranks_wcos == 1.0).mean()),
            },
        },
        "map_acc_on_covered_rows": {
            "single_draw_target": single,
            "draw_averaged_target": avg_target,
            "single_draw_csv_reconciliation_max_abs_rank_delta": csv_delta,
        },
        "avg_query_secondary": avg_query,
        "addendum_conventions": {
            "note": (
                "addendum battery: full-pool retrieval on ALL 9,941 held-out rows under the two "
                "extra conventions; the banked 5-convention full-pool acc@1 values live in "
                "eval_results/issue_2202/geometry_summary.json fail_counts "
                "(raw_euclidean 0.8160, whiten_cos 0.9535)"
            ),
            **extra_fullpool,
        },
        "nonlinear_maps": {
            "note": (
                "addendum-2 battery: banked #1738 nonlinear held-out predictions (fp16 on HF, "
                "cast fp64) on the same 9,941-row pool, full-pool retrieval under raw-euclidean "
                "and whitened-cosine with the identical rank machinery; ridge recomputed beside "
                "them as the reference"
            ),
            **nonlinear,
        },
        "absence_sweep": sweep,
        "staging": {"dir": str(STAGE), **revisions},
        "meta": FC.meta_block({"wall_seconds": round(time.time() - t0, 1)}),
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FC.atomic_json(OUT_DIR / "summary.json", summary)
    print(f"[done] wrote {OUT_DIR / 'summary.json'} in {time.time() - t0:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
