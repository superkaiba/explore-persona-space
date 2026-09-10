"""Issue #2588: matched-d/n refit of the capability panel on the target diagonal.

The panel's capability correlation is not robust to how dimensionality is
controlled: refit R2 vs the Artificial Analysis index reads +0.612 uncorrected,
+0.333 conditioning on d/n, and +0.806 conditioning on effective output
dimensionality. The closed-form OLS attenuation 1/(1 - d/n) cannot arbitrate,
because these are ridge fits whose effective degrees of freedom sit far below d
and the correction returns R2 above 1.0 for six of ten cells. So the estimator
regime is equalized BY DESIGN here rather than corrected for post hoc.

Two regimes, the diagonal only (train target matches eval target):
    single    train seed 42,          eval seed 42
    averaged  train mean(42, 45, 46), eval mean(42, 43, 44)

Per cell we sweep n_train and record held-out R2. That yields
  * the matched-d/n slice, read at n = d / MATCH_RATIO. MATCH_RATIO is the
    HIGHEST d/n in the panel, because the widest cells already sit there with
    all their data and cannot go lower, so matching is only possible downward.
  * each cell's learning curve, for extrapolating the n -> infinity asymptote
    where estimator bias vanishes outright instead of being equalized.

Layer stays FROZEN at each cell's banked layer_star. Lambda is refit per fit
from the validation-selected grid, since a different n wants a different
penalty and holding it fixed would confound the sweep.

Estimator reuse, no re-implementation: fits go through the exact chain the K3
round used, issue779_ffc_n1m_fits.fit_ridge_with_weights over _ridge_factorize
/ _ridge_predict_one, so these numbers are comparable to the banked and refit
numbers rather than introducing a second estimator.

Usage:
    uv run python scripts/issue2588_k3_dnmatch_refit.py --cells q35_27b_a   # pilot
    uv run python scripts/issue2588_k3_dnmatch_refit.py                     # all ten
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

STAGE_ROOT = Path("/mnt/eps-data/thomasjiralerspong/issue2588_dnmatch")
os.environ.setdefault("HF_HOME", str(STAGE_ROOT / "hf"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import issue779_ffc_n1m_fits as FIT  # noqa: E402
import issue779_percontext_recon as PR  # noqa: E402
from explore_persona_space.orchestrate import hub as HUB  # noqa: E402

HF_REPO = "superkaiba1/explore-persona-space-data"
PANEL = "issue2588_capability_panel"
CEILING_JSON = (
    _REPO_ROOT
    / "eval_results"
    / "issue_2588"
    / "ceiling_normalized"
    / "ceiling_normalized_capability.json"
)
OUT_DEFAULT = (
    _REPO_ROOT / "eval_results" / "issue_2588" / "ceiling_normalized" / "k3_dnmatch_refit.json"
)

#: Highest d/n in the panel (q38_27b_a: 5120 / 9738). Matching is only possible
#: downward, so every cell is subsampled to this ratio for the matched slice.
MATCH_RATIO = 0.526
LAMBDAS = np.logspace(-3, 6, 19)  # == issue779_fitter_fair_comparison.LAMBDAS_N10K
BLOCK = FIT.RIDGE_BLOCK
TRAIN_SEEDS_AVG = (42, 45, 46)
TEST_STAGES_AVG = ("test_1000", "ceiling_s43", "ceiling_s44")


def _ci(row_ids: np.ndarray) -> np.ndarray:
    """Capture rows are keyed '<stage>_<ci>'; align draws on the ci suffix."""
    return np.array([str(r).rsplit("_", 1)[1] for r in row_ids])


def _load_dir(path_in_repo: str) -> dict[str, np.ndarray] | None:
    """Concatenate every shard under one capture/L<star> dir."""
    from huggingface_hub import HfApi, hf_hub_download

    files = sorted(
        p
        for p in HUB.list_hf_files_under_path(HfApi(), HF_REPO, path_in_repo, repo_type="dataset")
        if p.endswith(".npz")
    )
    if not files:
        return None
    ids, ys, xs = [], [], []
    for f in files:
        local = HUB.retry_transient(
            lambda f=f: hf_hub_download(HF_REPO, f, repo_type="dataset"),
            what=f"hf_hub_download {f}",
        )
        with np.load(local, allow_pickle=False) as z:
            ids.append(z["row_ids"])
            ys.append(z["y_ans"])
            if "x_prompt_last" in z.files:
                xs.append(z["x_prompt_last"])
    out = {"ci": _ci(np.concatenate(ids)), "y": np.concatenate(ys).astype(np.float32)}
    if xs:
        out["x"] = np.concatenate(xs).astype(np.float32)
    return out


def _align(base: dict, others: list[dict]) -> tuple[np.ndarray, list[np.ndarray]]:
    """Restrict every draw to the ci present in all of them, in base order."""
    keep = set(base["ci"])
    for o in others:
        keep &= set(o["ci"])
    order = [i for i, c in enumerate(base["ci"]) if c in keep]
    ci_order = base["ci"][order]
    pos = {c: i for i, c in enumerate(ci_order)}
    aligned = []
    for o in others:
        idx = np.full(len(ci_order), -1, dtype=np.int64)
        for i, c in enumerate(o["ci"]):
            if c in pos:
                idx[pos[c]] = i
        assert (idx >= 0).all(), "draw missing an aligned ci"
        aligned.append(idx)
    return np.asarray(order, dtype=np.int64), aligned


def load_cell(cell: str, model: str, star: int) -> dict:
    """Context + single-draw and averaged answer targets for train/val/test."""
    p = f"{PANEL}/{model}/nothink"
    k3 = f"{p}/k3_train_refit/analysis_tensors/capture"
    cap = f"{p}/analysis_tensors/capture"
    lay = f"L{star:02d}"

    train42 = _load_dir(f"{cap}/train_10k/{lay}")
    assert train42 is not None and "x" in train42, f"{cell}: no train context vectors"
    val42 = _load_dir(f"{cap}/val_400/{lay}")
    test42 = _load_dir(f"{cap}/test_1000/{lay}")

    def parts(stage_glob: str) -> dict | None:
        acc = None
        for part in range(8):
            d = _load_dir(f"{k3}/{stage_glob}_part{part}/{lay}")
            if d is None:
                continue
            acc = (
                d
                if acc is None
                else {
                    "ci": np.concatenate([acc["ci"], d["ci"]]),
                    "y": np.concatenate([acc["y"], d["y"]]),
                }
            )
        return acc if acc is not None else _load_dir(f"{k3}/{stage_glob}/{lay}")

    tr45, tr46 = parts("train_10k_s45"), parts("train_10k_s46")
    va45, va46 = parts("val_400_s45"), parts("val_400_s46")
    ce43, ce44 = _load_dir(f"{cap}/ceiling_s43/{lay}"), _load_dir(f"{cap}/ceiling_s44/{lay}")
    for name, obj in [
        ("train_s45", tr45),
        ("train_s46", tr46),
        ("val_s45", va45),
        ("val_s46", va46),
        ("ceiling_s43", ce43),
        ("ceiling_s44", ce44),
    ]:
        assert obj is not None, f"{cell}: missing {name}"

    def build(base, extra):
        order, idxs = _align(base, extra)
        x = base["x"][order] if "x" in base else None
        y1 = base["y"][order]
        ya = np.mean([y1] + [e["y"][i] for e, i in zip(extra, idxs)], axis=0)
        return x, y1, ya

    xtr, ytr1, ytra = build(train42, [tr45, tr46])
    xva, yva1, yvaa = build(val42, [va45, va46])
    xte, yte1, ytea = build(test42, [ce43, ce44])
    return {
        "cell": cell,
        "dim": int(xtr.shape[1]),
        "layer_star": star,
        "x": (xtr, xva, xte),
        "single": (ytr1, yva1, yte1),
        "averaged": (ytra, yvaa, ytea),
        "n_train_avail": int(xtr.shape[0]),
    }


def sweep_cell(data: dict, rng: np.random.Generator) -> dict:
    dim = data["dim"]
    n_avail = data["n_train_avail"]
    matched_n = int(round(dim / MATCH_RATIO))
    grid = sorted({n for n in [1500, 3000, 5000, 7000, matched_n, n_avail] if 0 < n <= n_avail})
    xtr, xva, xte = data["x"]
    dev = "cpu"
    out = {
        "matched_n": matched_n,
        "matched_feasible": matched_n <= n_avail,
        "n_train_avail": n_avail,
        "n_grid": grid,
        "regimes": {},
    }
    for regime in ("single", "averaged"):
        ytr, yva, yte = data[regime]
        X = torch.from_numpy(np.concatenate([xtr, xva, xte]))
        Y = torch.from_numpy(np.concatenate([ytr, yva, yte]).astype(np.float32))
        n_tr, n_va = len(xtr), len(xva)
        val_idx = np.arange(n_tr, n_tr + n_va)
        te_idx = np.arange(n_tr + n_va, n_tr + n_va + len(xte))
        rows = []
        for n in grid:
            sub = rng.choice(n_tr, size=n, replace=False) if n < n_tr else np.arange(n_tr)
            t0 = time.time()
            pred, meta, _ = FIT.fit_ridge_with_weights(
                X, Y, np.sort(sub), val_idx, te_idx, LAMBDAS, dev, BLOCK
            )
            r2 = float(PR._pooled_r2(pred, Y[te_idx]))
            rows.append(
                {
                    "n_train": int(n),
                    "d_over_n": dim / n,
                    "test_r2": r2,
                    "selected_lambda": meta["selected_lambda"],
                    "lambda_grid_edge": meta.get("lambda_grid_edge"),
                    "wall_s": round(time.time() - t0, 1),
                }
            )
            print(
                f"    {regime:9s} n={n:6d} d/n={dim / n:5.3f} R2={r2:7.4f} "
                f"lam={meta['selected_lambda']:<9.4g} {rows[-1]['wall_s']:6.1f}s",
                flush=True,
            )
        out["regimes"][regime] = rows
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells", nargs="*", default=None)
    parser.add_argument("--out", type=Path, default=OUT_DEFAULT)
    parser.add_argument("--seed", type=int, default=20260910)
    parser.add_argument("--keep-cache", action="store_true")
    args = parser.parse_args(argv)

    meta = {r["cell"]: r for r in json.loads(CEILING_JSON.read_text())["per_model"]}
    cells = args.cells or list(meta)
    rng = np.random.default_rng(args.seed)

    results = {}
    if args.out.exists():
        results = json.loads(args.out.read_text()).get("per_cell", {})
    for cell in cells:
        m = meta[cell]
        model = cell[:-2] if cell.endswith("_a") else cell
        print(f"[{cell}] model={model} L*={m['layer_star']} d={m['dimension']}", flush=True)
        t0 = time.time()
        data = load_cell(cell, model, m["layer_star"])
        print(
            f"  loaded n_train={data['n_train_avail']} d={data['dim']} in {time.time() - t0:.0f}s",
            flush=True,
        )
        res = sweep_cell(data, rng)
        res.update(
            {
                "aa_index": m["aa_index"],
                "dim": data["dim"],
                "layer_star": m["layer_star"],
                "cell_wall_s": round(time.time() - t0, 1),
            }
        )
        results[cell] = res
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "meta": {
                        "task": 2588,
                        "script": "scripts/issue2588_k3_dnmatch_refit.py",
                        "match_ratio": MATCH_RATIO,
                        "lambdas": "logspace(-3,6,19)",
                        "estimator": "issue779_ffc_n1m_fits.fit_ridge_with_weights",
                        "regimes": {
                            "single": "train s42 / eval s42",
                            "averaged": "train mean(42,45,46) / eval mean(42,43,44)",
                        },
                        "seed": args.seed,
                    },
                    "per_cell": results,
                },
                indent=2,
            )
            + "\n"
        )
        print(f"  [{cell}] done in {res['cell_wall_s']}s -> {args.out.name}", flush=True)
        if not args.keep_cache:
            shutil.rmtree(STAGE_ROOT / "hf", ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
