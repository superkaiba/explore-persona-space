#!/usr/bin/env python3
"""Issue #1482: repeat the Matryoshka TIER gradient on DECODER DIRECTIONS.

WHY. A reviewer asked whether correlated noise -- specifically "degree of feature
splitting among some group of features" -- drives the SAE property results, and
said they would feel better if the analysis repeated on decoder directions gave
the same answer. The Matryoshka tier axis is the part of the paper closest to
that worry: finer nested tiers are, by construction, the same content carved into
more and narrower features, so "tier" and "how split a feature is" are nearly the
same variable. If the tier gradient is an artifact of split features firing more
rarely and therefore being scored more noisily, it should weaken or vanish when
the target no longer passes through the encoder and the sparsity gate.

WHAT CHANGES. Only the target.
  paper (banked)   per-feature held-out R^2 of predicting the feature's MEAN
                   ACTIVATION over the answer -- gated, thresholded, mostly zeros
                   for a rare feature.
  here             per-direction held-out R^2 of predicting d_f^T v, the MEAN
                   ANSWER STATE's component along feature f's UNIT DECODER
                   COLUMN -- dense, defined on every one of the 6,000 scored
                   answers, with no encoder, no jumprelu gate and no threshold.
                   Every direction is measured on the same rows, so the
                   rare-feature scoring-noise channel is closed by construction.

TARGET PROVENANCE (the retracted-run fix). The banked layer-20 dense store
holds ONLY prompt-side single-token states (c20 = h20[context_end],
hp20 = h20[prefix_end]); an earlier run read hp20 as the answer state and got
pooled R^2 = 0.071 (vs 0.653 for the paper's layer-19 map) -- retracted. The
target here is the freshly captured MEAN layer-20 answer state (ans_l20_g*.npz,
key a20, written by `issue1482_matryoshka_tier.py --phase capture_ans` under the
round's own capture convention with a banked-c20 identity gate). A pooled-R^2
floor gate below refuses to report per-direction numbers from a target in the
retracted regime.

The dictionary, the layer, the rows, the fit/score split and the tier definition
are all held fixed at the banked round's values, so the comparison is a target
swap and nothing else.

REFERENCE VALUES this run is compared against (eval_results/issue_1482/
matryoshka_tier/tier_tests.json, the activation-target read):
    raw Spearman(tier, R^2)                     -0.3949
    partial Spearman(tier, R^2 | log activity)  -0.1938
    within-activity-quintile permutation band   [-0.2499, -0.2275], coarse-better
    raw per-tier median R^2                     0.4346 / 0.1739 / 0.0430
                                                (n = 1,640 / 6,144 / 8,600)

ESTIMATOR NOTES. The dense-to-dense map is a fresh ridge at layer 20 fit on the
banked 24,000 fit rows against d = 3,584, so n_train >> d and the fit is
over-determined (no under-determined-regime refusal, and the #1887 pure-GCV
concern does not apply at n > d). The GCV lambda grid is np.logspace(-2, 8, 21)
-- the paper's appendix ceiling of 1e8, NOT the fit_h default logspace(-2, 4, 13)
whose 1e4 ceiling sits BELOW the 3162 the paper's own layer-19 fit selected --
and the selected lambda is reported (fit_h `info` diagnostics, #1887). The
per-direction R^2 recipe (1 - ||E u||^2 / ||Yc u||^2) is the one validated to
3.3e-16 against banked trait values in this round's layer-19 sibling.

Runs pod-side after capture_ans (the ans shards + staged banked dense store are
both local there); --self-test exercises the numeric plumbing on synthetic data
with no artifacts and no network.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()  # thread caps + credentials BEFORE numpy (shared-VM discipline)

import numpy as np  # noqa: E402
from scipy.stats import rankdata, spearmanr  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import issue1482_sae as S  # noqa: E402

from explore_persona_space.experiments.issue_779.fit_h import ridge_fit_predict  # noqa: E402

LAYER = 20
SAE_IDS = {"lmsys": "lmsys/matryoshka/k-100", "pile": "pile/matryoshka/k-100"}
TIER_EDGES = (0, 2048, 16384, 65536)  # nested prefix widths -> tier 0/1/2
COV = "eval_results/issue_1482/matryoshka_tier/covariates_m_{fam}.npz"
REF = "eval_results/issue_1482/matryoshka_tier/tier_tests.json"
CHUNK = 8192
LAMBDAS = np.logspace(-2, 8, 21)  # paper-appendix ceiling 1e8; 10^3.5 = 3162 on-grid
TARGET_KEY = {"ans_mean": "a20", "ans_mean_inlier": "a20_inlier"}


def _log(msg: str) -> None:
    print(f"[mtry-dec {datetime.now(UTC).strftime('%H:%M:%S')}] {msg}", flush=True)


def _load_shards(dirpath: Path, pattern: str, keys: tuple[str, ...]) -> dict[str, np.ndarray]:
    """Assemble sharded npz columns; fail loud on missing shards / dup row ids."""
    shards = sorted(glob.glob(str(dirpath / pattern)))
    if not shards:
        raise FileNotFoundError(f"no {pattern} shards under {dirpath}")
    cols: dict[str, list[np.ndarray]] = {k: [] for k in keys}
    for p in shards:
        with np.load(p) as z:
            for k in keys:
                cols[k].append(np.asarray(z[k]))
    out = {k: np.concatenate(v) for k, v in cols.items()}
    row = np.asarray(out["row_idx"], dtype=np.int64)
    if len(set(row.tolist())) != row.size:
        raise RuntimeError(f"duplicate row_idx across {pattern} shards")
    out["row_idx"] = row
    _log(f"{len(shards)} {pattern} shards -> {row.size:,} rows")
    return out


def select(row: np.ndarray, want: np.ndarray, what: str) -> np.ndarray:
    """Positions of `want` within `row`, failing loud on any absence."""
    at = {int(r): j for j, r in enumerate(row.tolist())}
    missing = [int(w) for w in want.tolist() if int(w) not in at]
    if missing:
        raise RuntimeError(f"{len(missing)} {what} rows absent from the store")
    return np.array([at[int(w)] for w in want.tolist()], dtype=np.int64)


def per_direction_r2(Yc: np.ndarray, E: np.ndarray, W: np.ndarray) -> np.ndarray:
    """1 - ||E u||^2 / ||Yc u||^2 for every unit column of W, chunked.

    Quadratic forms against the two Gram matrices, so the (n, n_dirs) products
    never materialize. Identical recipe to the layer-19 sibling.
    """
    G_y = Yc.T @ Yc
    G_e = E.T @ E
    out = np.empty(W.shape[1], dtype=np.float64)
    for s in range(0, W.shape[1], CHUNK):
        U = W[:, s : s + CHUNK]
        num = np.einsum("df,df->f", G_e @ U, U)
        den = np.einsum("df,df->f", G_y @ U, U)
        out[s : s + CHUNK] = 1.0 - num / den
    return out


def partial_spearman(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
    """Spearman(x, y | z) by residualizing both ranks on rank(z)."""
    rx, ry, rz = (rankdata(v).astype(np.float64) for v in (x, y, z))
    A = np.column_stack([np.ones_like(rz), rz])
    bx = np.linalg.lstsq(A, rx, rcond=None)[0]
    by = np.linalg.lstsq(A, ry, rcond=None)[0]
    return float(np.corrcoef(rx - A @ bx, ry - A @ by)[0, 1])


def tier_of(feat_ids: np.ndarray) -> np.ndarray:
    """Nested-prefix-width tier (0/1/2) of each feature id; raise out-of-range."""
    t = np.full(feat_ids.shape, -1, dtype=np.int64)
    for k in range(3):
        lo, hi = TIER_EDGES[k], TIER_EDGES[k + 1]
        t[(feat_ids >= lo) & (feat_ids < hi)] = k
    if (t < 0).any():
        raise RuntimeError("feature id outside the nested tier range")
    return t


def stratum_permutation(
    tier: np.ndarray, r2: np.ndarray, act: np.ndarray, n_perm: int, seed: int
) -> dict:
    """Permute tier labels WITHIN activity quintiles; the round's H1 design."""
    q = np.quantile(act, [0.2, 0.4, 0.6, 0.8])
    strat = np.searchsorted(q, act, side="right")
    obs = float(spearmanr(tier, r2).statistic)
    rng = np.random.default_rng(seed)
    draws = np.empty(n_perm, dtype=np.float64)
    for i in range(n_perm):
        perm = tier.copy()
        for s in np.unique(strat):
            m = strat == s
            perm[m] = rng.permutation(tier[m])
        draws[i] = spearmanr(perm, r2).statistic
    lo, hi = np.quantile(draws, [0.025, 0.975])
    return {
        "observed_pooled_spearman": obs,
        "perm_band_2p5_97p5": [float(lo), float(hi)],
        "n_perm": n_perm,
        "n_strata": int(np.unique(strat).size),
        "outside_band": bool(obs < lo or obs > hi),
        "verdict": "coarse-better" if obs < lo else ("fine-better" if obs > hi else "inside-band"),
    }


def _self_test() -> int:
    """Synthetic in-memory exercise of the numeric plumbing (no artifacts, no
    network): widened-grid ridge + info diagnostics, chunked per-direction R^2
    vs a naive per-column loop, tier/partial/permutation helpers, and the
    pooled-R^2 gate arithmetic."""
    rng = np.random.default_rng(0)
    n, d, k = 400, 24, 96
    X = rng.standard_normal((n, d))
    M = rng.standard_normal((d, d)) / np.sqrt(d)
    Y = X @ M + 0.05 * rng.standard_normal((n, d))
    info: dict = {}
    pred = ridge_fit_predict(X[:300], Y[:300], X[300:], lambdas=LAMBDAS, info=info)
    assert info["selected_lambda"] > 0 and info["lambda_grid"] == [1e-2, 1e8, 21], info
    Yc = Y[300:] - Y[300:].mean(0)
    E = Y[300:] - pred
    pooled = 1.0 - float((E**2).sum()) / float((Yc**2).sum())
    assert pooled > 0.9, f"self-test ridge under-fit: pooled={pooled}"
    W = rng.standard_normal((d, k))
    W = W / np.linalg.norm(W, axis=0, keepdims=True)
    r2 = per_direction_r2(Yc, E, W)
    naive = np.array(
        [1.0 - ((E @ W[:, j]) ** 2).sum() / ((Yc @ W[:, j]) ** 2).sum() for j in range(k)]
    )
    assert np.allclose(r2, naive, atol=1e-10), "chunked per-direction R^2 != naive"
    feat = rng.integers(0, TIER_EDGES[-1], 512)
    t = tier_of(feat)
    act = rng.lognormal(size=512)
    y = -0.3 * t + rng.standard_normal(512)
    ps = partial_spearman(t.astype(np.float64), y, np.log(act))
    assert -1.0 <= ps <= 1.0
    sp = stratum_permutation(t, y, act, 50, 0)
    assert sp["n_perm"] == 50 and sp["n_strata"] == 5
    print("[self-test] OK", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.replace("%", "%%"))
    ap.add_argument("--store", help="dir holding ans_l20_g*.npz + split_indices_matryoshka.npz")
    ap.add_argument(
        "--dense-store",
        default=None,
        help="dir holding the banked dense_l20_g*.npz shards (c20 map inputs); defaults to --store",
    )
    ap.add_argument(
        "--target",
        default="ans_mean",
        choices=sorted(TARGET_KEY),
        help="captured answer-state target: plain mean (a20) or inlier-masked mean",
    )
    ap.add_argument(
        "--sae-dir",
        default=str(PROJECT_ROOT / "data/issue_1482/mtry_sae"),
        help="SAE weight cache",
    )
    ap.add_argument("--family", default="lmsys", choices=["lmsys", "pile"])
    ap.add_argument("--out-dir", default="eval_results/issue_1482/decoder_direction")
    ap.add_argument("--n-perm", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=14826)
    ap.add_argument(
        "--min-pooled-r2",
        type=float,
        default=0.2,
        help="refuse per-direction reporting below this pooled R^2 (the retracted "
        "hp20-as-answer-state run read 0.071; the layer-19 reference is 0.653)",
    )
    ap.add_argument(
        "--allow-low-pooled-r2",
        action="store_true",
        help="override the pooled-R^2 floor (requires an explicit written justification)",
    )
    ap.add_argument(
        "--hf-upload-prefix",
        default=None,
        help="when set, upload the two output files to this HF data-repo prefix "
        "(pod-side harvest transport) and verify the listing",
    )
    ap.add_argument("--self-test", action="store_true", help="synthetic plumbing test, then exit")
    args = ap.parse_args()
    if args.self_test:
        return _self_test()
    if not args.store:
        ap.error("--store is required (unless --self-test)")

    store = Path(args.store)
    dense_store = Path(args.dense_store) if args.dense_store else store
    out_dir = PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    dense = _load_shards(dense_store, "dense_l20_g*.npz", ("row_idx", "c20"))
    akey = TARGET_KEY[args.target]
    ans = _load_shards(store, "ans_l20_g*.npz", ("row_idx", akey, "ans_all_out", "n_ans"))
    split_path = store / "split_indices_matryoshka.npz"
    if not split_path.exists():
        split_path = dense_store / "split_indices_matryoshka.npz"
    with np.load(split_path) as z:
        s_fit = np.asarray(z["s_fit"], dtype=np.int64)
        s_score = np.asarray(z["s_score"], dtype=np.int64)
    pd_fit = select(dense["row_idx"], s_fit, "fit(dense)")
    pd_score = select(dense["row_idx"], s_score, "score(dense)")
    pa_fit = select(ans["row_idx"], s_fit, "fit(ans)")
    pa_score = select(ans["row_idx"], s_score, "score(ans)")
    d_model = dense["c20"].shape[1]
    _log(f"fit n={pd_fit.size:,}  score n={pd_score.size:,}  d={d_model}  target={akey}")
    if pd_fit.size <= d_model:
        raise RuntimeError(f"n_train={pd_fit.size} <= d={d_model}: under-determined ridge refused")

    X_tr = np.asarray(dense["c20"][pd_fit], dtype=np.float64)
    Y_tr = np.asarray(ans[akey][pa_fit], dtype=np.float64)
    X_ev = np.asarray(dense["c20"][pd_score], dtype=np.float64)
    Y_ev = np.asarray(ans[akey][pa_score], dtype=np.float64)
    _log("fitting dense->dense ridge at layer 20 (grid 1e-2..1e8, 21 pts) ...")
    ridge_info: dict = {}
    pred = ridge_fit_predict(X_tr, Y_tr, X_ev, lambdas=LAMBDAS, info=ridge_info)
    lam = ridge_info["selected_lambda"]
    at_edge = bool(lam <= LAMBDAS[0] * (1 + 1e-9) or lam >= LAMBDAS[-1] * (1 - 1e-9))
    _log(f"selected lambda = {lam:.6g}{' — AT GRID EDGE' if at_edge else ''}")
    Yc = Y_ev - Y_ev.mean(axis=0, keepdims=True)
    E = Y_ev - pred
    pooled = 1.0 - float((E**2).sum()) / float((Yc**2).sum())
    _log(f"pooled dense R^2 on the score rows = {pooled:.4f}")
    if pooled < args.min_pooled_r2 and not args.allow_low_pooled_r2:
        raise RuntimeError(
            f"pooled R^2 {pooled:.4f} < floor {args.min_pooled_r2}: the retracted "
            "hp20-as-answer-state run read 0.071 (vs 0.653 for the paper's layer-19 "
            "map) — a pooled R^2 this low means the target is NOT a mean answer "
            "state (wrong array / row misalignment). Per-direction numbers are NOT "
            "reported. Re-run with --allow-low-pooled-r2 only with an explicit "
            "written justification."
        )

    S.SAELensJumpReLU.ensure_downloaded(SAE_IDS[args.family], args.sae_dir)
    sae = S.SAELensJumpReLU.load(SAE_IDS[args.family], device="cpu", cache_dir=args.sae_dir)
    W_dec = np.asarray(sae.w_dec.detach().cpu().numpy(), dtype=np.float64)  # (n_feat, d)
    if W_dec.shape[1] != d_model:
        W_dec = W_dec.T
    W = W_dec.T  # (d, n_feat)
    W = W / (np.linalg.norm(W, axis=0, keepdims=True) + 1e-12)
    _log(f"decoder {W.shape[1]:,} directions, unit-normalized")

    r2_dir = per_direction_r2(Yc, E, W)
    feat_all = np.arange(W.shape[1], dtype=np.int64)
    tier_all = tier_of(feat_all)

    with np.load(PROJECT_ROOT / COV.format(fam=args.family)) as z:
        panel_ids = np.asarray(z["feat_ids"], dtype=np.int64)
        panel_act = np.asarray(z["activity"], dtype=np.float64)
        panel_tier = np.asarray(z["tier"], dtype=np.int64)
    if not np.array_equal(panel_tier, tier_of(panel_ids)):
        raise RuntimeError("banked panel tier disagrees with the nested-width tier definition")

    r2_panel = r2_dir[panel_ids]
    ok = np.isfinite(r2_panel) & np.isfinite(panel_act) & (panel_act > 0)
    _log(f"panel {panel_ids.size:,}; usable {int(ok.sum()):,}")

    ref = json.loads((PROJECT_ROOT / REF).read_text())
    res = {
        "generated_utc": datetime.now(UTC).isoformat(),
        "family": args.family,
        "layer": LAYER,
        "target": f"decoder-direction projection d_f^T v of the {args.target} "
        f"answer state (dense, no encoder/gate)",
        "target_key": akey,
        "stores": {"ans": str(store), "dense": str(dense_store)},
        "n_fit": int(pd_fit.size),
        "n_score": int(pd_score.size),
        "ans_all_out_rows": int(np.asarray(ans["ans_all_out"]).sum()),
        "n_ans_median": float(np.median(np.asarray(ans["n_ans"], dtype=np.float64))),
        "pooled_dense_r2": pooled,
        "pooled_r2_floor": {
            "min": args.min_pooled_r2,
            "overridden": bool(args.allow_low_pooled_r2),
        },
        "ridge": {
            "selected_lambda": lam,
            "gcv": ridge_info["gcv"],
            "lambda_grid": ridge_info["lambda_grid"],
            "at_grid_edge": at_edge,
        },
        "panel": {
            "n": int(ok.sum()),
            "raw_spearman_tier_r2": float(spearmanr(panel_tier[ok], r2_panel[ok]).statistic),
            "partial_spearman_tier_r2_given_logact": partial_spearman(
                panel_tier[ok], r2_panel[ok], np.log(panel_act[ok])
            ),
            "per_tier": {
                str(k): {
                    "n": int(((panel_tier == k) & ok).sum()),
                    "median": float(np.median(r2_panel[(panel_tier == k) & ok])),
                    "q25": float(np.quantile(r2_panel[(panel_tier == k) & ok], 0.25)),
                    "q75": float(np.quantile(r2_panel[(panel_tier == k) & ok], 0.75)),
                }
                for k in range(3)
            },
            "within_stratum_permutation": stratum_permutation(
                panel_tier[ok], r2_panel[ok], panel_act[ok], args.n_perm, args.seed
            ),
        },
        "full_dictionary": {
            "n": int(np.isfinite(r2_dir).sum()),
            "per_tier_median": {
                str(k): float(np.median(r2_dir[(tier_all == k) & np.isfinite(r2_dir)]))
                for k in range(3)
            },
        },
        "reference_activation_target": {
            "raw_spearman": ref["h1_raw_descriptive"]["spearman_tier_r2_raw"],
            "partial_spearman": ref["h1_tier_within_stratum"][
                "partial_spearman_tier_r2_given_logact"
            ],
            "perm_band": ref["h1_tier_within_stratum"]["perm_band_2p5_97p5"],
            "verdict": ref["h1_tier_within_stratum"]["verdict"],
            "per_tier": ref["h1_raw_descriptive"]["per_tier"],
        },
    }
    path = out_dir / f"matryoshka_decoder_direction_{args.family}.json"
    path.write_text(json.dumps(res, indent=1))
    npz_path = out_dir / f"r2_decoder_direction_{args.family}.npz"
    np.savez_compressed(npz_path, r2=r2_dir, tier=tier_all, panel_ids=panel_ids)

    p = res["panel"]
    r = res["reference_activation_target"]
    _log("")
    _log(f"{'statistic':<34}{'decoder dir':>14}{'activation':>14}")
    _log(
        f"{'raw spearman(tier,R2)':<34}{p['raw_spearman_tier_r2']:>14.4f}{r['raw_spearman']:>14.4f}"
    )
    _log(
        f"{'partial spearman | log activity':<34}"
        f"{p['partial_spearman_tier_r2_given_logact']:>14.4f}{r['partial_spearman']:>14.4f}"
    )
    for k in range(3):
        _log(
            f"{'median R2 tier ' + str(k):<34}{p['per_tier'][str(k)]['median']:>14.4f}"
            f"{r['per_tier'][str(k)]['median']:>14.4f}"
        )
    _log(
        f"permutation verdict: {p['within_stratum_permutation']['verdict']} "
        f"(reference: {r['verdict']})"
    )
    _log(f"wrote {path}")

    if args.hf_upload_prefix:
        from huggingface_hub import HfApi

        from explore_persona_space.orchestrate import hub

        for f in (path, npz_path):  # UPLOAD_LOOP_EXEMPT: exactly two small files, not a bulk tree
            up_url = hub._upload(
                f,
                hub.HF_DATA_REPO,
                "dataset",
                f"{args.hf_upload_prefix}/{f.name}",
                upload_as_file=True,
                raise_on_error=True,
            )
            if not up_url:
                raise RuntimeError(f"decoder-direction upload returned no path for {f.name}")
        expected = [f"{args.hf_upload_prefix}/{f.name}" for f in (path, npz_path)]
        missing = hub.verify_repo_paths_uploaded(
            HfApi(),
            hub.HF_DATA_REPO,
            expected,
            path_in_repo=args.hf_upload_prefix,
            repo_type="dataset",
        )
        assert not missing, f"decoder-direction upload verify: missing on Hub: {missing}"
        _log(f"uploaded + verified {len(expected)} files under {args.hf_upload_prefix}")
    return 0


if __name__ == "__main__":
    rc = main()
    # explicit exit after flushing: heavy C-extension teardown can rewrite the rc in
    # interpreter finalization (PyGILState atexit race, #1689 gotcha)
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(rc)
