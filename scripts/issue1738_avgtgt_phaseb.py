#!/usr/bin/env python3
"""Issue #1738 inline free-analysis round ``avgtgt-phaseb`` (user-chat carve-out).

Phase-B conventions battery for the avg-target-maps round: the two n=20,000
maps (single-draw-target-trained vs 5-draw-averaged-target-trained ridge,
holdout predictions banked at issue1738_multiturn/avg_target/analysis_tensors/
pred16/) scored under the #2202 convention set, beside the banked 88k ridge:

- (a) FULL-POOL single-draw battery on all 9,941 held-out rows: raw
  euclidean, whitened cosine, csls_k10_whitencos, csls_pen_whitencos_g10 —
  each with differentiation metrics (success/failure margin quantiles, MRR,
  pairwise AUC = 1 - (mean_rank - 1)/(n_pool - 1)).
- (b) the same 4 conventions + differentiation on DRAW-AVERAGED targets under
  the #2202 FULL-POOL-REPLACEMENT convention: the 1,988 resample-covered
  rows' pool entries are replaced by 5-draw means built from the BANKED
  #1738 kresample (pool stays 9,941; CSLS pool-side statistics recomputed on
  the modified pool; query bank = the map's own full 9,941 predictions).
  NOTE the avg-target driver's own avg5 eval used a DIFFERENT convention —
  a 1,988-entry pool of averaged targets — so its 90.2-98.6% acc@1 numbers
  are not comparable to (b); the difference is stated in the output JSON.

Reconciliation gates (all BEFORE any new cell is trusted): the 88k ridge
full-pool raw-euclidean acc@1 vs the banked 0.8160; the 20k maps' full-pool
raw/whitened-cos acc@1 vs the completion marker's retrieval companions
(0.765/0.932 single-trained, 0.767/0.934 avg-trained; rounded to 3 dp);
the computed 5-draw means vs the banked y_holdout_avg5_L19.npz.

Machinery imported verbatim from ``scripts/issue2202_residual_read.py``
(covered_battery / csls_covered / stats_block). Analysis-only; vectorized.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()  # shared-VM thread caps must bind BEFORE numpy/torch import (#847)

import issue1738_characterize as CH  # noqa: E402
import issue2202_failchar as FC  # noqa: E402
import issue2202_metric_zoo as MZ  # noqa: E402
import issue2202_residual_read as RR  # noqa: E402
import numpy as np  # noqa: E402
from scipy.linalg import solve_triangular  # noqa: E402

from explore_persona_space.orchestrate import hub  # noqa: E402

FW_STAGE = Path("/mnt/eps-data/thomasjiralerspong/issue2202_freshwhiten")  # read-only reuse
STAGE = Path("/mnt/eps-data/thomasjiralerspong/issue1738_phaseb")
OUT_PATH = PROJECT_ROOT / "eval_results" / "issue_1738" / "avg_target" / "phaseb_conventions.json"
HF_AVG_PREFIX = "issue1738_multiturn/avg_target/analysis_tensors"
K_DRAWS = 4
N_COVERED = 1_988
CONVS = ("raw_euclidean", "whiten_cos", "csls_k10_whitencos", "csls_pen_whitencos_g10")
CSLS_GAMMAS = {"csls_k10_whitencos": 0.5, "csls_pen_whitencos_g10": 1.0}
# reconciliation targets
BANKED_88K_RAW_ACC1 = 0.816014485464239
MARKER_COMPANIONS = {  # v135 completion-marker retrieval companions (3-dp rounded)
    "map_single_20k": {"raw_euclidean": 0.765, "whiten_cos": 0.932},
    "map_avg_20k": {"raw_euclidean": 0.767, "whiten_cos": 0.934},
}
COMPANION_TOL = 0.002  # the marker's figures are percent-rounded
EXPECTED_88K_AVG_CSLS_K10 = 0.994466800804829  # avgtgt-completion round (exact code path)


def stage_inputs() -> dict:
    STAGE.mkdir(parents=True, exist_ok=True)
    for fname in ("map_single_20k_L19.npz", "map_avg_20k_L19.npz"):
        hub.stage_hub_file(FC.C.HF_DATA_REPO, f"{HF_AVG_PREFIX}/pred16/{fname}", STAGE / fname)
    hub.stage_hub_file(
        FC.C.HF_DATA_REPO,
        f"{HF_AVG_PREFIX}/y_holdout_avg5_L19.npz",
        STAGE / "y_holdout_avg5_L19.npz",
    )
    from huggingface_hub import HfApi

    return {
        "stage_dir": str(STAGE),
        "freshwhiten_reused_read_only": str(FW_STAGE),
        "data_repo_head": HfApi().repo_info(FC.C.HF_DATA_REPO, repo_type="dataset").sha,
    }


def main() -> int:
    t0 = time.time()
    revisions = stage_inputs()

    pd_ = np.load(FW_STAGE / "pred16.npz")
    yd = np.load(FW_STAGE / "y_holdout_L19.npz")
    y16 = yd["y16"].astype(np.float64)
    pci = np.asarray(pd_["ci"], dtype=np.int64)
    assert (pci == np.asarray(yd["ci"], dtype=np.int64)).all()
    n_pool = y16.shape[0]
    full_idx = np.arange(n_pool)

    preds: dict[str, np.ndarray] = {"ridge_88k": pd_["pred16"].astype(np.float64)}
    for tag, fname in (
        ("map_single_20k", "map_single_20k_L19.npz"),
        ("map_avg_20k", "map_avg_20k_L19.npz"),
    ):
        z = np.load(STAGE / fname)
        assert (np.asarray(z["ci"], dtype=np.int64) == pci).all(), f"{tag} ci misalign"
        p = z["pred16"].astype(np.float64)
        assert p.shape == y16.shape, (tag, p.shape)
        preds[tag] = p

    kns = SimpleNamespace(
        local_kresample_dir=str(FW_STAGE / "kresample"),
        scratch=str(STAGE / "scratch"),
        hf_prefix="",
    )
    kci, vres = CH._load_kresample_v(kns, [FC.LAYER])
    draws = vres[:, :, 0, :].astype(np.float64)
    pos_of = {int(c): p for p, c in enumerate(pci.tolist())}
    pos = np.asarray([pos_of[int(c)] for c in kci], dtype=np.int64)

    wz = np.load(FW_STAGE / "whiten_stats.npz")
    mu_a = np.asarray(wz["mu_A"], dtype=np.float64)
    ell = np.asarray(wz["L"], dtype=np.float64)

    def _wh(x: np.ndarray) -> np.ndarray:
        return solve_triangular(ell, (np.asarray(x, np.float64) - mu_a).T, lower=True).T

    def _norm(x: np.ndarray) -> np.ndarray:
        return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)

    # draw-averaged pool (the #2202 FULL-POOL-REPLACEMENT convention)
    avg = (y16[pos] + draws.sum(axis=1)) / (1 + K_DRAWS)
    pool_mod = y16.copy()
    pool_mod[pos] = avg
    y16w = _wh(y16)
    pool_modw = y16w.copy()
    pool_modw[pos] = _wh(avg)
    qwn = {"single": _norm(y16w), "avg": _norm(pool_modw)}
    pool_raw = {"single": y16, "avg": pool_mod}
    pool_w = {"single": y16w, "avg": pool_modw}
    eval_rows = {"single": full_idx, "avg": pos}  # (a) all 9,941; (b) covered 1,988

    # reconcile computed 5-draw means against the banked y_holdout_avg5
    az = np.load(STAGE / "y_holdout_avg5_L19.npz")
    a_keys = az.files
    a_ci = np.asarray(az["ci"], dtype=np.int64)
    a_key = next(k for k in a_keys if k != "ci" and az[k].ndim == 2)
    banked_avg = az[a_key].astype(np.float64)
    a_pos_of = {int(c): p for p, c in enumerate(a_ci.tolist())}
    assert set(int(c) for c in a_ci) == set(int(c) for c in kci), "y_holdout_avg5 ci set mismatch"
    reord = np.asarray([a_pos_of[int(c)] for c in kci], dtype=np.int64)
    avg_delta = float(np.abs(banked_avg[reord] - avg).max())
    avg_rel = float(avg_delta / (np.abs(avg).max() + 1e-12))
    print(
        f"[recon-avg5] key={a_key} max|banked - computed| = {avg_delta:.4g} (rel {avg_rel:.2e})",
        flush=True,
    )

    matrix: dict[str, dict] = {}
    recon: dict[str, dict] = {
        "y_holdout_avg5": {"array_key": a_key, "max_abs_delta": avg_delta, "max_rel_delta": avg_rel}
    }
    for mi, (tag, pred) in enumerate(preds.items()):
        tm = time.time()
        predw = _wh(pred)
        pwn = _norm(predw)
        cells: dict[str, dict] = {c: {} for c in CONVS}
        for variant in ("single", "avg"):
            rows = eval_rows[variant]
            q_raw, q_w = pred[rows], predw[rows]
            r, m, _ = RR.covered_battery(
                q_raw, pool_raw[variant], rows, "euclidean", f"{tag}-raw-{variant}"
            )
            cells["raw_euclidean"][variant] = RR.stats_block(r, m, n_pool)
            r, m, _ = RR.covered_battery(
                q_w, pool_w[variant], rows, "cosine", f"{tag}-wcos-{variant}"
            )
            cells["whiten_cos"][variant] = RR.stats_block(r, m, n_pool)
            t1 = time.time()
            s_full = pwn @ qwn[variant].T
            print(
                f"[{tag}-swc-{variant}] S ({n_pool}x{n_pool}) in {time.time() - t1:.1f}s",
                flush=True,
            )
            for conv, gamma in CSLS_GAMMAS.items():
                r, m, _ = RR.csls_covered(s_full, rows, gamma)
                cells[conv][variant] = RR.stats_block(r, m, n_pool)
            del s_full
            print(
                f"[phaseb] map {mi + 1}/3 {tag} variant={variant} elapsed={time.time() - tm:.1f}s",
                flush=True,
            )
        matrix[tag] = cells
        del predw, pwn

    # ── reconciliation gates ──
    got = matrix["ridge_88k"]["raw_euclidean"]["single"]["acc_at_1"]
    assert abs(got - BANKED_88K_RAW_ACC1) < 1e-9, ("88k raw full-pool", got)
    recon["ridge_88k_raw_fullpool"] = {"recomputed": got, "banked": BANKED_88K_RAW_ACC1}
    got = matrix["ridge_88k"]["csls_k10_whitencos"]["avg"]["acc_at_1"]
    assert abs(got - EXPECTED_88K_AVG_CSLS_K10) < 1e-9, ("88k csls_k10 avg", got)
    recon["ridge_88k_csls_k10_avg"] = {"recomputed": got, "expected": EXPECTED_88K_AVG_CSLS_K10}
    for tag, comp in MARKER_COMPANIONS.items():
        for conv, banked in comp.items():
            got = matrix[tag][conv]["single"]["acc_at_1"]
            delta = got - banked
            recon[f"{tag}_{conv}_vs_marker"] = {
                "recomputed": got,
                "marker_rounded": banked,
                "delta": delta,
            }
            assert abs(delta) <= COMPANION_TOL, (tag, conv, got, banked)
            print(
                f"[recon] {tag} {conv} full-pool acc@1 {got:.4f} vs marker {banked} (delta {delta:+.4f})",
                flush=True,
            )

    summary = {
        "round": "avgtgt-phaseb (user-chat inline free-analysis, task #1738)",
        "conventions_note": (
            "phase (a) `single` = full-pool single-draw battery, ALL 9,941 held-out rows; phase "
            "(b) `avg` = the #2202 FULL-POOL-REPLACEMENT draw-averaged convention — the 1,988 "
            "resample-covered rows' pool entries replaced by 5-draw means (banked primary + 4 "
            "banked kresample draws), pool stays 9,941, eval on the covered rows, CSLS pool "
            "statistics recomputed on the modified pool (K_LOCAL="
            f"{MZ.K_LOCAL}, query bank = the map's own full 9,941 predictions). This DIFFERS "
            "from the avg-target driver's own avg5 eval, which retrieved among a 1,988-entry "
            "pool of averaged targets only (its 90.2-98.6% acc@1 cells are NOT comparable to "
            "phase (b)). Margins in each convention's own units (CSLS: whitened-cos score gap; "
            "whiten_cos: cosine-distance gap; raw_euclidean: SQUARED-euclidean gap); positive "
            "= true target wins; pairwise AUC = 1 - (mean_rank - 1)/(n_pool - 1)"
        ),
        "n_pool": int(n_pool),
        "n_covered": int(N_COVERED),
        "matrix": matrix,
        "reconciliation": recon,
        "staging": revisions,
        "meta": FC.meta_block({"wall_seconds": round(time.time() - t0, 1)}),
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    FC.atomic_json(OUT_PATH, summary)
    print(f"[done] wrote {OUT_PATH} in {time.time() - t0:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
