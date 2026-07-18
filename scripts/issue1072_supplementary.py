"""Issue #1072 — supplementary enrichment / concordance reads (VM CPU, ~1 min).

Persists the interpretation's supplementary numbers as a committed artifact
(interp-critique r1 finding 5): the enrichment ratio E = S_par / w_par per
layer with bootstrap CIs, the per-context paired concordance at the primary
layer, and the PAIRED bootstrap layer-differences replacing the CI-overlap
argument for H3 (finding 6). Bootstrap recipe is the primary battery's
verbatim (10k multinomial draws over the full pool, rng 0, fold assignment
fixed — intervals conditional on the fold assignment); every statistic is a
ratio of pooled sums evaluated per draw with SHARED weights, so cross-layer
differences are paired by construction.

Consumes ``per_context_stats_1072_fold{k}.npz`` + ``battery_1072_fold4.json``
+ ``stats_component.json``; emits ``supplementary_reads.json`` next to them.
"""

from __future__ import annotations

import argparse
import json
import logging
import pathlib
import sys
import time

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()  # BEFORE any heavy import — shared-VM thread caps (#847)

import numpy as np  # noqa: E402

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from explore_persona_space.experiments.issue_952.run_952 import (  # noqa: E402
    _json_np,
    _repo_git_sha,
)
from scripts.issue1072_stats import (  # noqa: E402
    BOOTSTRAP_SEED,
    CH,
    N_DRAWS_DEFAULT,
    PRIMARY_LAYER,
    RatioBank,
    _ci,
    _load_folds,
    _register_cells,
    _safe_ratio,
)

logger = logging.getLogger("issue1072.supplementary")

CONSISTENCY_TOL = 1e-12


def _register_wpar_cells(bank: RatioBank, npzs: dict[int, dict], layers: list[int]) -> None:
    """One (ss_tot_par, ss_tot_full) ratio column per (layer, own arm): the
    pooled parallel VARIANCE share w_par of the remainder target."""
    for layer in layers:
        key = f"M16c_L{layer}|own"
        ids = np.concatenate([npzs[k]["ids_test"] for k in sorted(npzs)])
        ch = np.concatenate([npzs[k][key].astype(np.float64) for k in sorted(npzs)], axis=0)
        bank.add(f"wpar|{layer}|own", ids, ch[:, CH["ss_tot_par"]], ch[:, CH["ss_tot_full"]])


def _percontext_deltas(npzs: dict[int, dict], layer: int, comp: str) -> np.ndarray:
    """Per-context contribution delta c^own - c^ext_plain (figure-script twin)."""
    out = []
    for k in sorted(npzs):
        vals = {}
        for arm in ("own", "ext_plain"):
            ch = npzs[k][f"M16c_L{layer}|{arm}"].astype(np.float64)
            num = (
                ch[:, CH[f"ss_tot_{comp}"]] - ch[:, CH[f"ss_res_{comp}"]]
                if comp in ("par", "perp")
                else ch[:, CH["cross_tot"]] - ch[:, CH["cross_res"]]
            )
            vals[arm] = _safe_ratio(num, ch[:, CH["ss_tot_full"]])
        out.append(vals["own"] - vals["ext_plain"])
    return np.concatenate(out)


def main() -> None:
    p = argparse.ArgumentParser(description="Issue #1072 supplementary reads")
    p.add_argument("--eval-dir", type=str, default=str(_REPO_ROOT / "eval_results" / "issue_1072"))
    p.add_argument("--n-draws", type=int, default=N_DRAWS_DEFAULT)
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    t0 = time.time()
    eval_dir = pathlib.Path(args.eval_dir)
    out_path = pathlib.Path(args.out) if args.out else eval_dir / "supplementary_reads.json"
    stats = json.loads((eval_dir / "stats_component.json").read_text())

    npzs, recs = _load_folds(eval_dir)
    layers = sorted(int(x) for x in recs[0]["regime"]["layers"])
    pool_ids = [int(i) for i in npzs[0]["ids_pool_full"].tolist()]

    bank = RatioBank(pool_ids)
    _register_cells(bank, npzs, layers)
    _register_wpar_cells(bank, npzs, layers)
    idx = {n: i for i, n in enumerate(bank.names)}
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    w = rng.multinomial(bank.n, np.full(bank.n, 1.0 / bank.n), size=args.n_draws).astype(np.float64)
    obs = bank.observed()
    draws = bank.draws(w)

    def _delta(leg: str, layer: int, comp: str, arr: np.ndarray) -> np.ndarray:
        a = arr[..., idx[f"{leg}|{layer}|own|{comp}"]]
        b = arr[..., idx[f"{leg}|{layer}|ext_plain|{comp}"]]
        return a - b

    by_layer: dict[str, dict] = {}
    s_obs: dict[int, float] = {}
    s_draws: dict[int, np.ndarray] = {}
    e_obs: dict[int, float] = {}
    e_draws: dict[int, np.ndarray] = {}
    for layer in layers:
        sp_o = float(_delta("c", layer, "par", obs) / _delta("c", layer, "full", obs))
        sp_d = _safe_ratio(_delta("c", layer, "par", draws), _delta("c", layer, "full", draws))
        wp_o = float(obs[idx[f"wpar|{layer}|own"]])
        wp_d = draws[..., idx[f"wpar|{layer}|own"]]
        e_o = sp_o / wp_o
        e_d = _safe_ratio(sp_d, wp_d)
        # Consistency: S_par must reproduce the committed stats_component.json value.
        ref = stats["by_ext"]["ext_plain"][str(layer)]["S_par"]
        assert abs(sp_o - ref) < CONSISTENCY_TOL, (layer, sp_o, ref)
        s_obs[layer], s_draws[layer] = sp_o, sp_d
        e_obs[layer], e_draws[layer] = e_o, e_d
        by_layer[str(layer)] = {
            "S_par": sp_o,
            "w_par_own_pooled": wp_o,
            "w_par_own_pooled_ci95": _ci(wp_d),
            "E_enrichment": e_o,
            "E_ci95": _ci(e_d),
        }

    lo_l, hi_l = min(layers), max(layers)
    dip_ref = 23 if 23 in layers else lo_l
    paired = {
        f"S_par_L{hi_l}_minus_L{lo_l}": {
            "observed": s_obs[hi_l] - s_obs[lo_l],
            "ci95": _ci(s_draws[hi_l] - s_draws[lo_l]),
        },
        f"S_par_L{hi_l}_minus_L{dip_ref}": {
            "observed": s_obs[hi_l] - s_obs[dip_ref],
            "ci95": _ci(s_draws[hi_l] - s_draws[dip_ref]),
        },
        f"E_L{hi_l}_minus_L{lo_l}": {
            "observed": e_obs[hi_l] - e_obs[lo_l],
            "ci95": _ci(e_draws[hi_l] - e_draws[lo_l]),
        },
    }

    d_par = _percontext_deltas(npzs, PRIMARY_LAYER, "par")
    d_perp = _percontext_deltas(npzs, PRIMARY_LAYER, "perp")
    m = np.isfinite(d_par) & np.isfinite(d_perp)
    concord = {
        "layer": PRIMARY_LAYER,
        "n_matched": int(m.sum()),
        "frac_dperp_gt_dpar": float((d_perp[m] > d_par[m]).mean()),
        "median_dc_par": float(np.median(d_par[m])),
        "median_dc_perp": float(np.median(d_perp[m])),
    }

    # Per-slot component profile at the primary layer, calibration fold (interp
    # finding 3's numbers — panel c of the exploratory figure).
    h1 = recs[4]["layers"][str(PRIMARY_LAYER)]["components"]["h1"]
    per_slot = []
    for t in range(1, 17):
        own = h1.get(f"f16_t{t}|own")
        ext = h1.get(f"f16_t{t}|ext_plain")
        per_slot.append(
            {
                "t": t,
                "dC_par": (own["C_par"] - ext["C_par"]) if own and ext else None,
                "dC_perp": (own["C_perp"] - ext["C_perp"]) if own and ext else None,
            }
        )

    out = {
        "issue": 1072,
        "git_sha": _repo_git_sha(),
        "numpy_version": np.__version__,
        "ts": time.time(),
        "inputs": {
            "eval_dir": str(eval_dir),
            "n_draws": args.n_draws,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "layers": layers,
        },
        "recipe": (
            "10k multinomial bootstrap draws over the full pool (rng 0), fold assignment "
            "FIXED (parent recipe verbatim); every statistic is a ratio of pooled sums per "
            "draw with SHARED weights, so cross-layer differences are paired by construction. "
            "E = S_par / w_par with w_par = pooled own-arm parallel variance share "
            "(sum ss_tot_par / sum ss_tot_full, remainder target, c-leg)."
        ),
        "by_layer": by_layer,
        "paired_layer_differences": paired,
        "percontext_concordance_L26": concord,
        "per_slot_L26_calfold_own_minus_ext_plain": per_slot,
        "wall_seconds": time.time() - t0,
    }
    out_path.write_text(json.dumps(out, indent=2, default=_json_np))
    logger.info(
        "[supplementary] %s written: E by layer %s; paired %s (%.1fs)",
        out_path,
        {la: round(e_obs[la], 2) for la in layers},
        {k: [round(x, 4) for x in v["ci95"]] for k, v in paired.items()},
        out["wall_seconds"],
    )


if __name__ == "__main__":
    main()
