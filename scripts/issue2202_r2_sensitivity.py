"""Round-2 sensitivity re-draw for the issue-2202 reciprocity degree null.

The production degree-preserving null (``issue2202_failchar.degree_preserving_draws``)
permutes the target-stub multiset and KEEPS self-loop / multi-edge collisions,
computing reciprocity over the unique-edge set. Realized collisions are ~22.5% of
E = 329,448 per draw (``reciprocity.json .null_degree.multi_edges_mean`` = 73,971),
so each null draw's unique graph carries ~255k edges vs the observed graph's 329k —
the null's realized hub in-degrees are clipped, and the observed value sits ~10% of
band-width below the band top (knife-edge).

This script draws a COLLISION-FREE (simple-graph) degree-preserving null. A
stub-permutation + swap-repair approach is INFEASIBLE for this degree sequence
(max out-degree 9,935 of n = 9,941: the top source must link ~all nodes, so random
target multisets can never be repaired to distinct — measured plateau ~43k bad rows).
Instead: batched Maslov–Sneppen double-edge swaps STARTED FROM THE OBSERVED GRAPH
(the standard degree-preserving simple-graph null; Maslov & Sneppen 2002, the same
family the plan cites). Each swap exchanges the targets of two edges when the result
creates no self-loop and no duplicate edge — out- and in-degree sequences are
preserved exactly and the graph stays simple by construction. One chain: burn-in
``burnin_x_e`` × E accepted swaps, then ``n_draws`` reciprocity samples thinned every
``thin_x_e`` × E accepted swaps.

Bounded VM run: pure numpy, minutes of wall time.
"""

import argparse
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent
EDGES = HERE / "data" / "issue_2202" / "reciprocity_edges.npz"
OUT = HERE / "eval_results" / "issue_2202" / "reciprocity_collision_free.json"


def reciprocity_of_sorted(h_sorted: np.ndarray, n: int) -> float:
    """P(j->i in G | i->j in G) over a sorted unique directed edge-key array
    (parity with issue2202_failchar.reciprocity_of semantics)."""
    rev = (h_sorted % n) * n + (h_sorted // n)
    idx = np.searchsorted(h_sorted, rev)
    idx = np.clip(idx, 0, len(h_sorted) - 1)
    return float((h_sorted[idx] == rev).mean())


def swap_batch(
    src: np.ndarray, dst: np.ndarray, h_sorted: np.ndarray, n: int, rng: np.random.Generator
) -> int:
    """One batched double-edge-swap pass over disjoint random edge pairs.

    Mutates ``dst`` in place; returns the number of accepted swaps. A pair
    ((a,b),(c,d)) -> ((a,d),(c,b)) is accepted iff it creates no self-loop, no
    no-op, no edge already in the CURRENT graph, and no within-batch duplicate
    new edge (conservative rejection keeps the graph simple exactly).
    """
    e_n = len(src)
    perm = rng.permutation(e_n)
    m = e_n // 2
    i1, i2 = perm[:m], perm[m : 2 * m]
    a, b = src[i1], dst[i1]
    c, d = src[i2], dst[i2]
    ok = (a != c) & (b != d) & (a != d) & (c != b)
    k1 = a * n + d
    k2 = c * n + b
    # reject if the new edge already exists in the current graph
    for k in (k1, k2):
        pos = np.clip(np.searchsorted(h_sorted, k), 0, len(h_sorted) - 1)
        ok &= h_sorted[pos] != k
    # within-batch duplicate new keys (either side) -> reject both claimants
    kk = np.concatenate([k1[ok], k2[ok]])
    _, first_idx, counts = np.unique(kk, return_index=True, return_counts=True)
    dup_keys = np.zeros(len(kk), dtype=bool)
    dup_mask_sorted = counts > 1
    if dup_mask_sorted.any():
        order = np.argsort(kk, kind="stable")
        kk_sorted = kk[order]
        dup_flag = np.zeros(len(kk), dtype=bool)
        dup_flag[1:] = kk_sorted[1:] == kk_sorted[:-1]
        dup_flag[:-1] |= kk_sorted[:-1] == kk_sorted[1:]
        dup_keys[order] = dup_flag
    ok_idx = np.nonzero(ok)[0]
    half = len(ok_idx)
    drop = dup_keys[:half] | dup_keys[half:]
    keep = ok_idx[~drop]
    if len(keep):
        ii1, ii2 = i1[keep], i2[keep]
        tmp = dst[ii1].copy()
        dst[ii1] = dst[ii2]
        dst[ii2] = tmp
    return int(len(keep))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-draws", type=int, default=200)
    ap.add_argument("--seed", type=int, default=2202)
    ap.add_argument("--burnin-x-e", type=float, default=5.0)
    ap.add_argument("--thin-x-e", type=float, default=0.5)
    args = ap.parse_args()

    z = np.load(EDGES)
    src_ci, dst_ci = z["src_ci"], z["dst_ci"]
    uniq = np.unique(np.concatenate([src_ci, dst_ci]))
    lut = {int(c): k for k, c in enumerate(uniq)}
    src = np.asarray([lut[int(c)] for c in src_ci], dtype=np.int64)
    dst = np.asarray([lut[int(c)] for c in dst_ci], dtype=np.int64)
    n = len(uniq)
    e_n = len(src)

    h0 = np.sort(src * n + dst)
    assert len(np.unique(h0)) == e_n, "observed edge list carries duplicate edges"
    obs = reciprocity_of_sorted(h0, n)

    rng = np.random.default_rng(args.seed)
    dst_c = dst.copy()
    h_sorted = h0.copy()
    burnin_target = int(args.burnin_x_e * e_n)
    thin_target = int(args.thin_x_e * e_n)
    accepted_total = 0
    acc_since_sample = 0
    in_burnin = True
    draws: list[float] = []
    prop_total = 0
    t0 = time.time()
    it = 0
    while len(draws) < args.n_draws:
        acc = swap_batch(src, dst_c, h_sorted, n, rng)
        h_sorted = np.sort(src * n + dst_c)
        accepted_total += acc
        prop_total += e_n // 2
        it += 1
        if in_burnin:
            if accepted_total >= burnin_target:
                in_burnin = False
                acc_since_sample = 0
        else:
            acc_since_sample += acc
            if acc_since_sample >= thin_target:
                acc_since_sample = 0
                draws.append(reciprocity_of_sorted(h_sorted, n))
                if len(draws) % 25 == 0:
                    print(
                        f"sample {len(draws)}/{args.n_draws} accepted={accepted_total} "
                        f"({time.time() - t0:.0f}s)",
                        flush=True,
                    )
    # invariants at chain end: simple graph, exact degree sequences
    assert len(np.unique(src * n + dst_c)) == e_n, "duplicate edges at chain end"
    assert (dst_c == src).sum() == 0, "self-loops at chain end"
    assert np.array_equal(np.sort(dst_c), np.sort(dst)), "in-degree sequence broken"

    arr = np.asarray(draws)
    lo, med, hi = np.percentile(arr, [2.5, 50, 97.5])
    lag1 = float(np.corrcoef(arr[:-1], arr[1:])[0, 1]) if len(arr) > 2 else float("nan")
    sha = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=HERE
    ).stdout.strip()
    out = {
        "observed": obs,
        "E": int(e_n),
        "n_draws": int(args.n_draws),
        "seed": int(args.seed),
        "null_degree_collision_free": {
            "band": {"p2.5": float(lo), "p50": float(med), "p97.5": float(hi)},
            "draws": arr.tolist(),
            "semantics": (
                "Maslov-Sneppen batched double-edge swaps from the observed graph "
                "(simple by construction; out- and in-degree sequences preserved "
                "exactly); one chain, burn-in "
                f"{args.burnin_x_e}xE accepted swaps, thinned every {args.thin_x_e}xE "
                "accepted swaps; conservative within-batch rejection"
            ),
            "acceptance_rate": float(accepted_total / prop_total),
            "accepted_total": int(accepted_total),
            "lag1_autocorr_of_samples": lag1,
        },
        "delta_dp_collision_free": float(obs - hi),
        "inside_band": bool(lo <= obs <= hi),
        "below_band": bool(obs < lo),
        "meta": {
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "numpy_version": np.__version__,
            "git_commit": sha,
            "edges_npz": str(EDGES.relative_to(HERE)),
            "wall_seconds": round(time.time() - t0, 1),
        },
    }
    OUT.write_text(json.dumps(out, indent=1))
    print(
        f"observed={obs:.6g} collision-free band=[{lo:.6g},{hi:.6g}] med={med:.6g} "
        f"inside={out['inside_band']} below={out['below_band']} "
        f"delta_dp_cf={out['delta_dp_collision_free']:.3g} "
        f"acc_rate={out['null_degree_collision_free']['acceptance_rate']:.3f} "
        f"lag1={lag1:.3f} wall={time.time() - t0:.0f}s"
    )


if __name__ == "__main__":
    main()
