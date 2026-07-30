"""Analyzer re-reduction: delta_beyond = R2(stitch-MLP) - R2(stitch+bilinear r*).

The P4 assembly ran BEFORE the P3 stitch-MLP re-run landed, so
``bilinear_fits.json`` carries the placeholder string for
``delta_beyond_mlp_minus_bilinear``. This script recomputes the statistic
post-hoc from the persisted per-row held-out predictions (HF
``issue1775_nonlinearity/analysis_tensors/{heldout_preds,bilinear_params}``),
mirroring ``issue1775_bilinear.assemble`` exactly:

- pred_star / pred_0 = mean-over-seeds pooled bilinear predictions, per-seed
  weight decay selected on inner val (``_best_variant`` over the committed
  ``units_shard*.jsonl``);
- mpred = mean over 3 seeds of the P3 stitch-MLP held-out predictions;
- delta_beyond = ``cluster_bootstrap_delta_r2(Yp, mpred, pred_star, b3,
  groups, n_draws=2000, seed=0)`` with b3 = cov_star & cov_0 & mlp_mask and
  groups = the fold scheme's grouping unit (prefix groups under novel-prefix
  folds, query groups under novel-query folds).

Validation: delta_named is recomputed from the same staged inputs and must
match the committed ``bilinear_fits.json`` value (same helper, same seed).
Schemes: prefix + query (the doubly scheme has no stitch-MLP comparator on
the Hub -- reported as unavailable). Output:
``eval_results/issue_1775/bilinear/delta_beyond_analysis.json``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(SCRIPTS.parent / "src"))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

# #847: .env + shared-VM thread caps bind BEFORE the heavy imports
# (tests/test_shared_vm_thread_caps.py::test_no_new_torch_before_dotenv_vm_entrypoints).
load_dotenv()

import numpy as np  # noqa: E402

from issue1775_common import (  # noqa: E402
    CELL_PRIMARY,
    HF_DATA_REPO,
    LAYER_PRIMARY,
    OUT_HF_PREFIX,
    TARGETS,
    _r2,
    battery_excluded_indices,
    cluster_bootstrap_delta_r2,
    fold_pairs,
    load_manifest_rows,
    load_summary,
    resolve_store_dir,
)

R_STAR = {"prefix": 32, "query": 16}  # committed inner-val selections (bilinear_fits.json)
SEEDS = [0, 1, 2]
N_DRAWS = 2000
SEED = 0
COMMITTED = Path(__file__).resolve().parents[1] / "eval_results" / "issue_1775" / "bilinear"
HF_DL = Path(__file__).resolve().parents[1] / "data" / "issue_1775" / "hf_dl"


def _fetch(rel: str) -> Path:
    """hf_hub_download into the worktree staging dir (idempotent), transport-retried (#1547)."""
    from huggingface_hub import hf_hub_download

    from explore_persona_space.orchestrate import hub

    return Path(
        hub.retry_transient(
            lambda: hf_hub_download(
                HF_DATA_REPO,
                f"{OUT_HF_PREFIX}/analysis_tensors/{rel}",
                repo_type="dataset",
                local_dir=HF_DL,
            ),
            what=f"issue1775 delta_beyond staging download {rel}",
        )
    )


def _best_variant(variants: list[dict], seed: int) -> dict:
    cand = [v for v in variants if v["seed"] == seed]
    return min(cand, key=lambda v: v["inner_val_mse"])


def _pooled_pred(units, scheme, r, n_rows, d_out, pairs):
    pred = np.zeros((n_rows, d_out))
    covered = np.zeros(n_rows, dtype=bool)
    for f, (_tr, te) in enumerate(pairs):
        recs = [u for u in units if u["scheme"] == scheme and u["fold"] == f and u["r"] == r]
        if not recs:
            continue
        acc = np.zeros((len(te), d_out))
        for s in SEEDS:
            v = _best_variant(recs[0]["variants"], s)
            p = _fetch(f"bilinear_params/pred_{scheme}_f{f}_r{r}_s{s}_wd{v['wd']:g}.npy")
            acc += np.load(p).astype(np.float64)
        pred[te] = acc / len(SEEDS)
        covered[te] = True
    return pred, covered


def main() -> None:
    store = resolve_store_dir()
    rows_all = load_manifest_rows(store)
    t_all = [load_summary(store, CELL_PRIMARY, t, LAYER_PRIMARY) for t in TARGETS]
    n0 = min(min(t.shape[0] for t in t_all), len(rows_all))
    be_idx = battery_excluded_indices(rows_all, n0)
    rows = [rows_all[int(i)] for i in be_idx]
    prefix_ids = np.asarray([r.get("prefix_id", "") for r in rows])
    query_ids = np.asarray([str(r.get("query_id", "")) for r in rows])
    Y = np.concatenate([np.asarray(t[be_idx], dtype=np.float64) for t in t_all], axis=1)
    n = len(rows)
    print(f"[delta-beyond] fit population n={n}", flush=True)
    # Parent constructor called DIRECTLY (full-population pca48 PCs): the wrapper's
    # record_plan_deviation expects the pod-side bare-list schema and crashes on the
    # orchestrator-materialized dict-schema plan_deviations.json; the deviation this
    # wrapper records is already declared there (both run markers).
    from issue1092_fit_grid import _basis_targets_with_info

    Yp, _info = _basis_targets_with_info(
        Y, "pca48", hidden_dim=3584, targets=list(TARGETS), projection_target="t1"
    )
    Yp = np.ascontiguousarray(Yp, dtype=np.float64)
    d_out = Yp.shape[1]

    units: list[dict] = []
    for shard in sorted(COMMITTED.glob("units_shard*.jsonl")):
        with open(shard) as fh:
            units.extend(json.loads(line) for line in fh if line.strip())
    committed = json.loads((COMMITTED / "bilinear_fits.json").read_text())

    out: dict = {
        "n_draws": N_DRAWS,
        "seed": SEED,
        "r_star_by_scheme": R_STAR,
        "mlp_seed_ensemble": SEEDS,
        "schemes": {},
        "doubly": (
            "unavailable — no stitch-MLP predictions exist for the doubly scheme on the Hub "
            "(P3 fit the stitch-MLP under the prefix and query fold schemes only); "
            "FILLED by the fu round: see eval_results/issue_1775/"
            "fu_dedup_refit_pcfold_doubly/delta_beyond_doubly.json "
            "(scripts/issue1775_doubly_mlp.py)"
        ),
        "note": (
            "post-hoc analyzer re-reduction of the placeholder "
            "delta_beyond_mlp_minus_bilinear in bilinear_fits.json; same helper "
            "(cluster_bootstrap_delta_r2), same n_draws/seed as the committed lattice CIs"
        ),
    }
    for scheme in ("prefix", "query"):
        groups = prefix_ids if scheme != "query" else query_ids
        grouping_unit = "prefix_id" if scheme != "query" else "query_id"
        pairs = fold_pairs(rows, n, scheme)
        r_star = R_STAR[scheme]
        pred_star, cov_star = _pooled_pred(units, scheme, r_star, n, d_out, pairs)
        pred_0, cov_0 = _pooled_pred(units, scheme, 0, n, d_out, pairs)
        both = cov_star & cov_0
        # validation: reproduce the committed delta_named exactly
        val = cluster_bootstrap_delta_r2(
            Yp, pred_star, pred_0, both, groups, n_draws=N_DRAWS, seed=SEED
        )
        ref = committed["schemes"][scheme]["delta_named"]
        dd = abs(val["delta_r2"] - ref["delta_r2"])
        print(
            f"[delta-beyond] {scheme}: delta_named recomputed {val['delta_r2']:.10f} "
            f"vs committed {ref['delta_r2']:.10f} (|diff|={dd:.2e})",
            flush=True,
        )
        assert dd < 1e-8, f"delta_named validation failed for {scheme}: |diff|={dd}"
        mlp_preds = []
        for s in SEEDS:
            mp = _fetch(
                f"heldout_preds/{CELL_PRIMARY}_L{LAYER_PRIMARY:02d}_stitch_perrow_pca48_"
                f"{scheme}_mlp_s{s}.npy"
            )
            mlp_preds.append(np.load(mp).astype(np.float64))
        mpred = np.mean(mlp_preds, axis=0)
        mmask = np.load(
            _fetch(
                f"heldout_preds/{CELL_PRIMARY}_L{LAYER_PRIMARY:02d}_stitch_perrow_pca48_"
                f"{scheme}_mlp_s{SEEDS[0]}_mask.npy"
            )
        )
        b3 = both & mmask
        boot = cluster_bootstrap_delta_r2(
            Yp, mpred, pred_star, b3, groups, n_draws=N_DRAWS, seed=SEED
        )
        out["schemes"][scheme] = {
            "grouping_unit": grouping_unit,
            "delta_beyond_mlp_minus_bilinear": boot,
            "r2_stitch_mlp_seed_mean": _r2(Yp[b3], mpred[b3]),
            "r2_bilinear_r_star": _r2(Yp[b3], pred_star[b3]),
            "delta_named_validation_abs_diff": dd,
            "n_rows_b3": int(b3.sum()),
        }
    from issue1775_common import result_meta

    out["meta"] = result_meta()
    dest = COMMITTED / "delta_beyond_analysis.json"
    dest.write_text(json.dumps(out, indent=2))
    print(f"[delta-beyond] wrote {dest}", flush=True)


if __name__ == "__main__":
    main()
