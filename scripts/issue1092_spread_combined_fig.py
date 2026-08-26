"""Combined-axes Result-3 figure: generated + natural prefixes under ONE rig (#1092).

Both populations on the SAME axes, honestly: the #1092 capture contains the 50
constructed battery conditions (stratum == "battery", all is_eval_only — never
in any fit) alongside the 996 natural prefixes, so both can be measured with
identical instruments:
  x — whitened within-prefix context-vector spread, whitening transform from
      the NATURAL rows only (strata recipe verbatim: pooled Sigma over the
      battery-excluded context vectors, lam = 1e-2 * tr(Sigma)/d, Cholesky);
      battery deviations are whitened by the SAME transform;
  y — per-prefix held-out error of the centroid (averaged-prefix) map:
      natural = the banked delta-test arm (unit_<cell>_ambient.json,
      per_prefix_err_avgctx; 6-fold over natural prefixes); battery = a
      press_fit_predict fit on ALL 996 natural centroids scoring the 50
      battery centroids (pure held-out by construction).
Parity gates: recomputed natural whitened spread must match the banked strata
npz elementwise; natural banked error array must be length 996.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from explore_persona_space.atomic_io import atomic_replace  # noqa: E402
from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

import numpy as np  # noqa: E402
import scipy.linalg as sla  # noqa: E402
import torch  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from issue923_fit_decomposition import press_fit_predict  # noqa: E402

STAGE = Path(
    "/mnt/eps-data/thomasjiralerspong/issue_1092_inline_operator/issue1092_realistic_crossing"
)
SUMM = STAGE / "analysis_tensors/summaries"
MANIFEST = STAGE / "corpus/manifest.jsonl"
STRATA = PROJECT_ROOT / "eval_results/issue_1092/inline_spread_whitened_strata"
DELTA = PROJECT_ROOT / "eval_results/issue_1092/inline_avgctx_spread_delta"
OUT = PROJECT_ROOT / "eval_results/issue_1092/inline_spread_combined_fig"
OUT.mkdir(parents=True, exist_ok=True)

CELLS = ["cell_inst_own", "cell_pre_own"]
CELL_LABELS = {"cell_inst_own": "Instruct model", "cell_pre_own": "Base model"}
TARGETS = ["t1", "t2", "t3"]
LAYER = 14
MIN_ROWS_PER_PREFIX = 3
WHITEN_LAMBDA_FRAC = 1e-2  # Source: strata script / #658 issue658_inline_a3_5a_coherence.py
SPREAD_PARITY_TOL = 1e-8


def _jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _load(cell: str, kind: str) -> np.ndarray:
    return np.load(SUMM / cell / f"{kind}_L{LAYER:02d}.npy", mmap_mode="r")


def _groups(prefix_ids: np.ndarray) -> tuple[list[str], dict[str, np.ndarray]]:
    g: dict[str, list[int]] = {}
    for i, pid in enumerate(prefix_ids):
        g.setdefault(str(pid), []).append(i)
    kept = {
        p: np.asarray(ix, dtype=np.int64) for p, ix in g.items() if len(ix) >= MIN_ROWS_PER_PREFIX
    }
    return sorted(kept), kept


def _whitened_spread(
    X: np.ndarray, pids: list[str], kept: dict[str, np.ndarray], L: np.ndarray
) -> np.ndarray:
    dev_blocks = []
    owner_blocks = []
    for k, p in enumerate(pids):
        block = X[kept[p]]
        dev_blocks.append(block - block.mean(0, keepdims=True))
        owner_blocks.append(np.full(block.shape[0], k, dtype=np.int64))
    dev = np.concatenate(dev_blocks, axis=0)
    owner = np.concatenate(owner_blocks)
    Z = sla.solve_triangular(L, dev.T, lower=True).T
    wsq = (Z * Z).sum(1)
    counts = np.bincount(owner, minlength=len(pids)).astype(np.float64)
    return np.sqrt(np.bincount(owner, weights=wsq, minlength=len(pids)) / counts)


def process_cell(cell: str, rows: list[dict]) -> dict:
    ctx_all = _load(cell, "context_end")
    t_all = [_load(cell, t) for t in TARGETS]
    n0 = min(ctx_all.shape[0], min(t.shape[0] for t in t_all), len(rows))
    nat_idx = np.asarray(
        [
            i
            for i in range(n0)
            if rows[i].get("stratum") != "trait_stratum" and not rows[i].get("is_eval_only")
        ],
        dtype=np.int64,
    )
    bat_idx = np.asarray(
        [i for i in range(n0) if rows[i].get("stratum") == "battery"], dtype=np.int64
    )
    X_nat = np.asarray(ctx_all[nat_idx], dtype=np.float64)
    X_bat = np.asarray(ctx_all[bat_idx], dtype=np.float64)
    nat_pids, nat_kept = _groups(np.asarray([rows[int(i)].get("prefix_id", "") for i in nat_idx]))
    bat_pids, bat_kept = _groups(np.asarray([rows[int(i)].get("prefix_id", "") for i in bat_idx]))
    assert len(nat_pids) == 996, f"{cell}: expected 996 natural prefixes, got {len(nat_pids)}"
    assert len(bat_pids) == 50, f"{cell}: expected 50 battery conditions, got {len(bat_pids)}"

    # whitening from NATURAL rows only (strata recipe verbatim)
    Xc = X_nat - X_nat.mean(0, keepdims=True)
    Sigma = (Xc.T @ Xc) / (X_nat.shape[0] - 1)
    lam = WHITEN_LAMBDA_FRAC * (np.trace(Sigma) / Sigma.shape[0])
    L = np.linalg.cholesky(Sigma + lam * np.eye(Sigma.shape[0]))
    del Xc, Sigma

    spread_nat = _whitened_spread(X_nat, nat_pids, nat_kept, L)
    banked_spread = np.load(STRATA / f"per_prefix_whitened_{cell}.npz")["spread_whitened"]
    parity = float(np.max(np.abs(spread_nat - np.asarray(banked_spread, dtype=np.float64))))
    assert parity < SPREAD_PARITY_TOL, f"{cell}: whitened-spread parity {parity:.3e} vs strata npz"
    print(f"[parity ok] {cell} whitened spread max|diff|={parity:.1e}", flush=True)
    spread_bat = _whitened_spread(X_bat, bat_pids, bat_kept, L)
    del L

    # centroid (averaged-prefix) map: natural errors banked; battery scored held-out
    unit = json.loads((DELTA / f"unit_{cell}_ambient.json").read_text())
    err_nat = np.asarray(unit["per_prefix_err_avgctx"], dtype=np.float64)
    assert err_nat.shape[0] == 996
    Y_nat = np.concatenate([np.asarray(t[nat_idx], dtype=np.float64) for t in t_all], axis=1)
    Y_bat = np.concatenate([np.asarray(t[bat_idx], dtype=np.float64) for t in t_all], axis=1)
    Xc_avg_nat = np.stack([X_nat[nat_kept[p]].mean(0) for p in nat_pids], axis=0)
    Y_avg_nat = np.stack([Y_nat[nat_kept[p]].mean(0) for p in nat_pids], axis=0)
    Xc_avg_bat = np.stack([X_bat[bat_kept[p]].mean(0) for p in bat_pids], axis=0)
    Y_avg_bat = np.stack([Y_bat[bat_kept[p]].mean(0) for p in bat_pids], axis=0)
    res = press_fit_predict(
        torch.from_numpy(Xc_avg_nat).double(),
        torch.from_numpy(Y_avg_nat).double(),
        torch.from_numpy(Xc_avg_bat).double(),
        standardize=True,
    )
    pred_bat = res["pred"].detach().cpu().numpy()
    err_bat = np.linalg.norm(pred_bat - Y_avg_bat, axis=1)

    out = {
        "spread_nat": spread_nat,
        "err_nat": err_nat,
        "spread_bat": spread_bat,
        "err_bat": err_bat,
        "bat_pids": np.asarray(bat_pids),
    }
    # Handle-form np.savez (numpy appends ".npz" to path-typed names lacking it;
    # the yielded tmp ends ".tmp" — #2336 recipe edge (c)).
    with atomic_replace(OUT / f"combined_arrays_{cell}.npz") as tmp:
        with open(tmp, "wb") as fh:
            np.savez(fh, **out)
    rho_n, p_n = spearmanr(spread_nat, err_nat)
    rho_b, p_b = spearmanr(spread_bat, err_bat)
    print(f"[done {cell}] nat rho={rho_n:+.3f} bat rho={rho_b:+.3f} (n_bat=50)", flush=True)
    return out


def main() -> int:
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis import paper_plots as pp

    rows = _jsonl(MANIFEST)
    cells = {cell: process_cell(cell, rows) for cell in CELLS}

    pp.set_paper_style("blog")
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.8), layout="constrained")
    c_all = pp.paper_palette_role("primary")
    for ax, cell in zip(axes, CELLS, strict=True):
        d = cells[cell]
        spread = np.concatenate([np.asarray(d["spread_nat"]), np.asarray(d["spread_bat"])])
        err = np.concatenate([np.asarray(d["err_nat"]), np.asarray(d["err_bat"])])
        rho, p = spearmanr(spread, err)
        ax.scatter(spread, err, s=10, alpha=0.32, color=c_all, linewidths=0)
        ptxt = "p < 1e-200" if p < 1e-200 else f"p = {p:.1e}"
        ax.text(
            0.03,
            0.95,
            f"Spearman ρ = +{rho:.2f}, {ptxt}  (n = {spread.size})",
            transform=ax.transAxes,
            va="top",
            fontsize=10,
        )
        ax.set_xlabel("within-prefix context-vector spread (whitened)")
        ax.set_ylabel("averaged-prefix-map per-prefix error")
        ax.set_title(f"{CELL_LABELS[cell]} (1,046 prefixes)", loc="left")
    pp.savefig_paper(
        fig,
        "summaries/prefix_vs_context_map/spread_vs_error_combined_one_rig",
        dir=str(PROJECT_ROOT / "figures"),
    )
    plt.close(fig)
    print("figure written", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
