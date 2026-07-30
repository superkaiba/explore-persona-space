"""Analyzer re-reduction: per-fold held-out R2 for the headline gap-closure levels.

Lens-11 companion data for the pooled hero levels: per novel-prefix /
novel-query fold, the held-out R2 of (a) the stitch PRESS ridge and
full-context PRESS ridge (read off the COMMITTED per-fold ``r2_folds``
arrays in ``eval_results/issue_1775/ladder/units_linear_shard*.jsonl``),
and (b) the bilinear r=0 / r=r* seed-ensemble and the stitch-MLP
3-seed ensemble (recomputed per fold from the SAME persisted per-row
held-out predictions ``issue1775_delta_beyond.py`` consumed, with the
identical ``_best_variant`` / mean-over-seeds recipe).

Validation: pooling this script's per-row predictions reproduces the
committed ``outer_r2_curve_EXPLORATORY`` values (bilinear) and the
``delta_beyond_analysis.json`` ``r2_stitch_mlp_seed_mean`` /
``r2_bilinear_r_star`` values (MLP / bilinear over the b3 mask) to
<1e-8. Output: ``eval_results/issue_1775/bilinear/perfold_headline.json``
(consumed by ``issue1775_figures.py --only-perfold``).
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
    fold_pairs,
    load_manifest_rows,
    load_summary,
    resolve_store_dir,
    result_meta,
)

R_STAR = {"prefix": 32, "query": 16}  # committed inner-val selections (bilinear_fits.json)
SEEDS = [0, 1, 2]
COMMITTED = Path(__file__).resolve().parents[1] / "eval_results" / "issue_1775"
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
            what=f"issue1775 perfold staging download {rel}",
        )
    )


def _best_variant(variants: list[dict], seed: int) -> dict:
    cand = [v for v in variants if v["seed"] == seed]
    return min(cand, key=lambda v: v["inner_val_mse"])


def _pooled_pred(units, scheme, r, n_rows, d_out, pairs):
    """Mean-over-seeds held-out bilinear prediction (per-seed wd on inner val)."""
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


def _committed_ridge_folds(arm: str, scheme: str) -> list[float] | None:
    """Per-fold PRESS-ridge R2 off the committed ladder units (primary combo, pca48)."""
    for shard in sorted((COMMITTED / "ladder").glob("units_linear_shard*.jsonl")):
        with open(shard) as fh:
            for line in fh:
                if not line.strip():
                    continue
                u = json.loads(line)
                if (
                    u.get("arm") == arm
                    and u.get("scheme") == scheme
                    and u.get("basis") == "pca48"
                    and u.get("engine") == "press"
                    and u.get("grain") == "perrow"
                    and u.get("layer") == LAYER_PRIMARY
                    and u.get("cell") == CELL_PRIMARY
                    and "r2_folds" in u
                ):
                    return [float(x) for x in u["r2_folds"]]
    return None


def main() -> None:
    store = resolve_store_dir()
    rows_all = load_manifest_rows(store)
    t_all = [load_summary(store, CELL_PRIMARY, t, LAYER_PRIMARY) for t in TARGETS]
    n0 = min(min(t.shape[0] for t in t_all), len(rows_all))
    be_idx = battery_excluded_indices(rows_all, n0)
    rows = [rows_all[int(i)] for i in be_idx]
    Y = np.concatenate([np.asarray(t[be_idx], dtype=np.float64) for t in t_all], axis=1)
    n = len(rows)
    print(f"[perfold] fit population n={n}", flush=True)
    # Parent constructor called DIRECTLY (full-population pca48 PCs) — same
    # parent-parity choice as issue1775_delta_beyond.py, deviation already on record.
    from issue1092_fit_grid import _basis_targets_with_info

    Yp, _info = _basis_targets_with_info(
        Y, "pca48", hidden_dim=3584, targets=list(TARGETS), projection_target="t1"
    )
    Yp = np.ascontiguousarray(Yp, dtype=np.float64)
    d_out = Yp.shape[1]

    units: list[dict] = []
    for shard in sorted((COMMITTED / "bilinear").glob("units_shard*.jsonl")):
        with open(shard) as fh:
            units.extend(json.loads(line) for line in fh if line.strip())
    fits = json.loads((COMMITTED / "bilinear" / "bilinear_fits.json").read_text())
    beyond = json.loads((COMMITTED / "bilinear" / "delta_beyond_analysis.json").read_text())

    out: dict = {
        "note": (
            "per-fold held-out R2 companion to the pooled gap-closure headline; "
            "bilinear/MLP folds recomputed from the persisted per-row predictions "
            "(identical recipe to issue1775_delta_beyond.py), ridge folds copied "
            "from the committed ladder units_linear r2_folds"
        ),
        "mlp_seed_ensemble": SEEDS,
        "r_star_by_scheme": R_STAR,
        "schemes": {},
    }
    for scheme in ("prefix", "query"):
        pairs = fold_pairs(rows, n, scheme)
        r_star = R_STAR[scheme]
        levels: dict = {}
        preds: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for label, r in (("bilinear_r0", 0), (f"bilinear_r{r_star}", r_star)):
            pred, cov = _pooled_pred(units, scheme, r, n, d_out, pairs)
            preds[label] = (pred, cov)
            per_fold = [float(_r2(Yp[te], pred[te])) for _tr, te in pairs]
            pooled = float(_r2(Yp[cov], pred[cov]))
            ref = float(fits["schemes"][scheme]["outer_r2_curve_EXPLORATORY"][str(r)])
            dd = abs(pooled - ref)
            print(f"[perfold] {scheme} {label}: pooled {pooled:.10f} vs committed {ref:.10f}")
            assert dd < 1e-8, f"{scheme}/{label} pooled validation failed: |diff|={dd}"
            levels[label] = {"per_fold": per_fold, "pooled": pooled, "committed_curve_ref": ref}
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
        b3 = preds[f"bilinear_r{r_star}"][1] & preds["bilinear_r0"][1] & mmask
        pooled_mlp = float(_r2(Yp[b3], mpred[b3]))
        ref_mlp = float(beyond["schemes"][scheme]["r2_stitch_mlp_seed_mean"])
        dd = abs(pooled_mlp - ref_mlp)
        print(
            f"[perfold] {scheme} stitch_mlp: pooled {pooled_mlp:.10f} vs committed {ref_mlp:.10f}"
        )
        assert dd < 1e-8, f"{scheme}/stitch_mlp pooled validation failed: |diff|={dd}"
        per_fold_mlp = []
        for _tr, te in pairs:
            sel = np.zeros(n, dtype=bool)
            sel[te] = True
            sel &= mmask
            per_fold_mlp.append(float(_r2(Yp[sel], mpred[sel])))
        levels["stitch_mlp_ensemble"] = {
            "per_fold": per_fold_mlp,
            "pooled_b3": pooled_mlp,
            "committed_ref": ref_mlp,
        }
        for label, arm in (
            ("stitch_press_ridge", "stitch"),
            ("context_press_ridge", "context_end"),
        ):
            folds = _committed_ridge_folds(arm, scheme)
            if folds is not None:
                levels[label] = {"per_fold": folds, "source": "committed units_linear r2_folds"}
        out["schemes"][scheme] = {
            "levels": levels,
            "n_te_per_fold": [int(len(te)) for _tr, te in pairs],
        }
    out["meta"] = result_meta()
    dest = COMMITTED / "bilinear" / "perfold_headline.json"
    dest.write_text(json.dumps(out, indent=2))
    print(f"[perfold] wrote {dest}", flush=True)


if __name__ == "__main__":
    main()
