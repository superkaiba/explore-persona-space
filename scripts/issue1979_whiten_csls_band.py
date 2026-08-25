"""#1979 inline round `whiten-csls-sweep` — selection-corrected band, ENLARGED set.

The banked ``perm_band.p975_max_selected`` is a signed max over the 12 RAW
candidates. The metric sweep searched a larger space (6 re-metricized predictors
x 4 settings, plus the metric-invariant carried candidates), so the banked band
understates the selection correction and cannot adjudicate a sweep result.

This recomputes the band over the enlarged candidate set, and computes the
12-candidate band FROM THE SAME PERMUTATION DRAWS so the difference isolates
candidate count from quantile noise (the banked run used n_perm=1000, whose tail
quantile is visibly noisy at n=50).

Procedure per arm, matching ``issue1900_race.perm_null`` + the #1979 consumer:
rank-z both sides, permute the DV, one GEMM to all candidates, SIGNED max over
the candidate axis per draw, then the 0.975 / 0.95 quantiles.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
from issue1979_whiten_csls_sweep import (  # noqa: E402
    CARRIED,
    DV_BY_KIND,
    RACE,
    REMETRIC,
    SETTINGS,
    arm_columns,
    load_inputs,
)

REPO_ROOT = SCRIPTS_DIR.parent
OUT = REPO_ROOT / "eval_results/issue_1979/whiten_csls"
CARRIED_COL = {"p4": "p4_tc", "p5": "p5", "p6": "p6", "p7": "p7", "p8a": "p8a", "p8b": "p8b"}
N_PERM = 20_000
SEED = 1979


def _rank_z(a: np.ndarray) -> np.ndarray:
    """Column-wise rank-z (the perm_null convention). (n,) or (n,k)."""
    x = a[:, None] if a.ndim == 1 else a
    r = np.column_stack([stats.rankdata(x[:, j]) for j in range(x.shape[1])])
    z = (r - r.mean(axis=0)) / (r.std(axis=0) + 1e-12)
    return z[:, 0] if a.ndim == 1 else z


def band_for(zc: np.ndarray, zd: np.ndarray, n_perm: int, seed: int) -> dict:
    """Signed-max-selected band over the candidate block zc (n,K)."""
    n = zd.shape[0]
    rng = np.random.default_rng(seed)
    perm = rng.permuted(np.tile(np.arange(n), (n_perm, 1)), axis=1)
    rho = (zd[perm] @ zc) / n  # (P, K)
    mx = rho.max(axis=1)
    return {
        "p975_max_selected": float(np.quantile(mx, 0.975)),
        "p95_max_selected": float(np.quantile(mx, 0.95)),
        "k_candidates": int(zc.shape[1]),
        "n_perm": int(n_perm),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-dir", type=Path, default=OUT)
    ap.add_argument("--n-perm", type=int, default=N_PERM)
    args = ap.parse_args(argv)

    d = load_inputs()
    rows: list[dict] = []
    for arm in d["arms"]:
        aid, kind = arm["arm_id"], arm["kind"]
        frame = json.loads((RACE / f"frame_{aid}.json").read_text())["frame"]
        banked = json.loads((RACE / f"arm_{aid}.json").read_text())
        dv = np.asarray(frame[DV_BY_KIND[kind]], dtype=np.float64)
        cols = arm_columns(d, arm)

        named: list[tuple[str, np.ndarray]] = []
        for s in SETTINGS:
            named += [(f"{p}@{s}", cols[s][p]) for p in REMETRIC]
        for p in CARRIED:  # metric-invariant, one column each; absent for marker p8*
            v = frame.get(CARRIED_COL[p])
            if v is not None:
                named.append((f"{p}@carried", np.asarray(v, dtype=np.float64)))

        M = np.column_stack([v for _, v in named])
        ok = np.isfinite(dv) & np.isfinite(M).all(axis=1)
        assert ok.sum() >= 45, (aid, int(ok.sum()))
        zd, zc_all = _rank_z(dv[ok]), _rank_z(M[ok])
        raw_ix = [
            i for i, (nm, _) in enumerate(named) if nm.endswith("@raw") or nm.endswith("@carried")
        ]

        b_enl = band_for(zc_all, zd, args.n_perm, SEED)
        b_raw = band_for(zc_all[:, raw_ix], zd, args.n_perm, SEED)  # SAME draws
        obs = {nm: float(stats.spearmanr(v[ok], dv[ok]).statistic) for nm, v in named}
        best = max(obs, key=obs.get)
        rows.append(
            {
                "arm_id": aid,
                "kind": kind,
                "n": int(ok.sum()),
                "band_enlarged": b_enl,
                "band_raw12_same_draws": b_raw,
                "band_banked_p975": banked["perm_band"]["p975_max_selected"],
                "banked_n_perm": banked["perm_band"]["n_perm"],
                "best_candidate": best,
                "best_rho": obs[best],
                "best_clears_enlarged": obs[best] > b_enl["p975_max_selected"],
                "observed": obs,
            }
        )
        print(
            f"{aid:26s} K={b_enl['k_candidates']:2d} enlarged={b_enl['p975_max_selected']:+.3f} "
            f"raw12={b_raw['p975_max_selected']:+.3f} banked={rows[-1]['band_banked_p975']:+.3f} "
            f"best={best} {obs[best]:+.3f} {'CLEARS' if rows[-1]['best_clears_enlarged'] else 'below'}"
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "enlarged_band.json").write_text(
        json.dumps({"rows": rows, "n_perm": args.n_perm, "seed": SEED}, indent=2)
    )
    print(f"[band] {len(rows)} arms -> {args.out_dir / 'enlarged_band.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
