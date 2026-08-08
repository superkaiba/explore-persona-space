#!/usr/bin/env python3
"""#1768: DEEP-probe the L14 operator key/value match read.

The shipped match read capped its probe at the top-32 singular pairs, and at
L14 seven of eight arms HIT that cap -- so the reported k=32 was "as much as
was computed", not a measured effective rank, and the 0.66-0.85 key-subspace
overlap was taken at the probe boundary.

This re-runs L14 with a FULL SVD of both operator updates, so:

  * k is the TRUE 99%-squared-mass effective rank of the predicted update
    (uncapped), and
  * the matched random-subspace null is evaluated at that same k.

The null is the load-bearing part. Two INDEPENDENT k-dim subspaces of R^n
already share mean principal-angle cosine of order sqrt(k/n), so the null
GROWS with k: 0.09 at k=32 but ~0.53 at k=1000. A key overlap that looked far
above a k=32 null can therefore sit AT a k_true null. That is exactly what this
probe decides -- it can only confirm or dissolve the L14 key-side claim, never
strengthen it by construction.

Expectation worth stating up front: the augmented refit changes the
STANDARDIZER (xmu/xsd depend on the row weights), so the predicted update is
NOT a rank-<=K object even though only K training pairs are injected -- every
dimension moves a little. A large k_true is therefore the likely outcome.

Writes its JSON to an --out-json path OUTSIDE the repo by default: uncommitted
repo-resident files in this shared root get reverted within seconds (see the
round record), so generation never goes through the working tree.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

import numpy as np  # noqa: E402

import issue1768_fit as F  # noqa: E402
import issue1768_lasttoken_fit as LTF  # noqa: E402
import issue1768_map_augmentation as MA  # noqa: E402
import issue1768_operator_kv as K  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("issue1768.opkv_deep")

MASS_FRACS = (0.90, 0.95, 0.99)
NULL_DRAWS_LARGE_K = 20  # the null's variance is tiny at large k
NULL_DRAWS_SMALL_K = 200
NULL_SEED = 176814


def _full_svd(A: np.ndarray):
    """Full SVD of the raw operator update. Returns (keys, values, svals).

    keys = LEFT vectors (context side), values = RIGHT vectors (answer side),
    matching the orientation ``K._assert_kv_orientation`` verifies.
    """
    import torch

    U, S, Vh = torch.linalg.svd(torch.as_tensor(A, dtype=torch.float64), full_matrices=False)
    return U.numpy(), Vh.numpy().T, S.numpy()


def _eff_rank(svals: np.ndarray, frac: float) -> int:
    c = np.cumsum(svals**2)
    tot = float(c[-1])
    if tot <= 0:
        return 1
    return int(np.searchsorted(c, frac * tot) + 1)


def _principal_angles(Aq: np.ndarray, Bq: np.ndarray) -> dict:
    qa, _ = np.linalg.qr(Aq)
    qb, _ = np.linalg.qr(Bq)
    s = np.clip(np.linalg.svd(qa.T @ qb, compute_uv=False), 0.0, 1.0)
    return {"mean_cos": float(s.mean()), "max_cos": float(s.max()), "min_cos": float(s.min())}


def _subspace_null(k: int, n: int, rng) -> dict:
    draws = NULL_DRAWS_SMALL_K if k <= 64 else NULL_DRAWS_LARGE_K
    means = []
    for _ in range(draws):
        qa, _ = np.linalg.qr(rng.standard_normal((n, k)))
        qb, _ = np.linalg.qr(rng.standard_normal((n, k)))
        s = np.clip(np.linalg.svd(qa.T @ qb, compute_uv=False), 0.0, 1.0)
        means.append(float(s.mean()))
    m = np.array(means)
    return {
        "k": int(k),
        "n_draws": int(draws),
        "mean_cos_null_mean": float(m.mean()),
        "mean_cos_null_p95": float(np.quantile(m, 0.95)),
        "analytic_scale_sqrt_k_over_n": float(np.sqrt(k / n)),
    }


def run_arm(out_root: Path, arm_id: str, layer: int, pos_path: str, block: int) -> dict:
    import torch

    dev = MA._device()
    rng = np.random.default_rng(NULL_SEED)
    committed = K._committed_cell(arm_id, layer)
    cache = out_root / "lt_answer_cache"
    cache.mkdir(parents=True, exist_ok=True)
    cell = LTF.build_cell(out_root, cache, arm_id, layer, MA.POSITION)
    tr, _val, _te = F._split_idx(cell["split"])
    C0, V0, Cp, Vp = cell["C0"], cell["V0"], cell["Cplus"], cell["Vplus"]
    T, S = MA.load_train_pairs(out_root, pos_path, layer)

    t0 = time.time()
    p0 = K._fit_operator_at_lambda(
        C0, V0, tr, committed["realized"]["M0"]["selected_lambda"], dev, block
    )
    pp = K._fit_operator_at_lambda(
        Cp, Vp, tr, committed["realized"]["Mplus"]["selected_lambda"], dev, block
    )
    A0 = MA._raw_operator(p0)
    dA_real = MA._raw_operator(pp) - A0
    legs = [lg for lg in committed["legs"] if lg["weight"] > 0]
    best = max(legs, key=lambda lg: lg["frac_change_closed"])
    ph = K._augmented_operator(
        C0, V0, tr, T, S, best["weight"], best["selected_lambda"], dev, block
    )
    dA_hat = MA._raw_operator(ph) - A0
    t_fit = time.time() - t0

    t1 = time.time()
    k_real, v_real, s_real = _full_svd(dA_real)
    k_hat, v_hat, s_hat = _full_svd(dA_hat)
    t_svd = time.time() - t1
    rel = K._assert_kv_orientation(dA_real, k_real, v_real, s_real)

    ranks_hat = {f"k_at_{int(f * 100)}pct": _eff_rank(s_hat, f) for f in MASS_FRACS}
    ranks_real = {f"k_at_{int(f * 100)}pct": _eff_rank(s_real, f) for f in MASS_FRACS}
    k_true = ranks_hat["k_at_99pct"]
    n = dA_real.shape[0]

    out = {
        "arm_id": arm_id,
        "layer": layer,
        "operator_dim": int(n),
        "kv_orientation_check_rel_err": rel,
        "at_mass": best["mass"],
        "at_weight": best["weight"],
        "effective_rank_predicted_update": ranks_hat,
        "effective_rank_realized_update": ranks_real,
        "shipped_k_capped_at": 32,
        "wall_s": {"refits": round(t_fit, 1), "full_svds": round(t_svd, 1)},
        "reads": {},
    }
    # report at k_true AND at the shipped cap, so the comparison is explicit
    for label, kk in (("k_true_99pct", k_true), ("k32_shipped_cap", min(32, n))):
        out["reads"][label] = {
            "k": int(kk),
            "key": _principal_angles(k_real[:, :kk], k_hat[:, :kk]),
            "value": _principal_angles(v_real[:, :kk], v_hat[:, :kk]),
            "random_subspace_null": _subspace_null(int(kk), n, rng),
        }
        r = out["reads"][label]
        for side in ("key", "value"):
            obs = r[side]["mean_cos"]
            nl = r["random_subspace_null"]
            r[side]["vs_null_ratio"] = (
                obs / nl["mean_cos_null_mean"] if nl["mean_cos_null_mean"] else float("nan")
            )
            r[side]["exceeds_null_p95"] = bool(obs > nl["mean_cos_null_p95"])
    del dA_real, dA_hat, A0, k_real, v_real, k_hat, v_hat
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--out-json", type=Path, required=True, help="write OUTSIDE the repo")
    ap.add_argument("--layer", type=int, default=14)
    ap.add_argument("--arms", default="")
    ap.add_argument("--block", type=int, default=50_000)
    args = ap.parse_args(argv)

    picks = MA.arm_picks()
    if args.arms:
        keep = {a.strip() for a in args.arms.split(",") if a.strip()}
        picks = [p for p in picks if p["arm_id"] in keep]
        assert picks, (keep, "no matching arms")
    srcs = MA._mix_sources(picks)
    cache = args.out_root / "lt_answer_cache"
    cache.mkdir(parents=True, exist_ok=True)

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    recs = json.loads(args.out_json.read_text()) if args.out_json.exists() else {}
    for i, p in enumerate(picks, start=1):
        arm_id = p["arm_id"]
        if arm_id in recs:
            logger.info("[deep] %d/%d %s: present, skip", i, len(picks), arm_id)
            continue
        MA._prewarm_arm_cache(cache, arm_id)
        rec = run_arm(args.out_root, arm_id, args.layer, srcs[arm_id]["pos_path"], args.block)
        MA._drop_arm_cache(cache, arm_id)
        recs[arm_id] = rec
        MA._atomic_json(args.out_json, recs)  # persist per arm
        kt = rec["reads"]["k_true_99pct"]
        k32 = rec["reads"]["k32_shipped_cap"]
        logger.info(
            "[deep] %d/%d %s L%d: k_true=%d  key %.3f (null %.3f, %s)  value %.3f  "
            "| at k=32: key %.3f (null %.3f)  fits=%.0fs svd=%.0fs",
            i,
            len(picks),
            arm_id,
            args.layer,
            kt["k"],
            kt["key"]["mean_cos"],
            kt["random_subspace_null"]["mean_cos_null_p95"],
            "ABOVE" if kt["key"]["exceeds_null_p95"] else "at/below",
            kt["value"]["mean_cos"],
            k32["key"]["mean_cos"],
            k32["random_subspace_null"]["mean_cos_null_p95"],
            rec["wall_s"]["refits"],
            rec["wall_s"]["full_svds"],
        )
    logger.info("[deep] %d arms complete -> %s", len(recs), args.out_json)
    print("[phase=done]", flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(0)


if __name__ == "__main__":
    main()
