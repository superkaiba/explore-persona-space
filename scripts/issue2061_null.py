"""P3 selection-symmetric null battery for task #2061.

Implements the REGISTERED statistic of plan §Design "Selection-symmetric
null" (code-review v1 C3 / BLOCKER `p3-null-placeholder-unregistered-
statistic` — the round-1 sign-flip placeholder is deleted):

  1. ONE SHARED `draw_seed_schedule = [42 + d for d in range(n_draws)]`
     across ALL 64 (stage-pair, corpus, arm) delta cells. Draw index `d`
     uses the SAME seed on every cell (load-bearing for the per-draw
     GLOBAL max reduction).
  2. Per cell, per draw d: PERMUTE the stage label within-corpus — for
     every conversation present in BOTH stages, `draw_seed_schedule[d]`
     decides whether its (before, after) rows SWAP pseudo-stage (a pair
     flip; rows without a cross-stage partner keep their true stage and
     the paired/unpaired counts are recorded per cell) — then REFIT the
     SAME ridge estimator per pseudo-stage: standardize-X on train stats,
     center-Y, GCV over the #823 grid `logspace(-2, 4, 13)` WITH the
     #1887 dof cap 0.9 (the P2 estimator, `ridge_fit_predict_fast_layer_
     batched` semantics — algebraically identical primal form, see
     `_CellEngine`), K=5 GROUP-level folds (conversation-id groups, fold
     seed 0, the #1336 convention via `issue2061_turnstore.group_fold_
     ids`), per-feature R^2_j pooled over folds with fold-local test
     means. Record `max_j_d = max_j (R^2_after' - R^2_before')`.
  3. Per-draw GLOBAL max reduction (PRIMARY headline null):
     `global_max_d = max_cell max_j_d`. Report p50/p95/p97.5/p99.
  4. Per-cell p97.5 retained as SECONDARY diagnostic only.
  5. Persist per-draw `max_j_d` per cell (KB-scale) so the GLOBAL
     reduction + any post-hoc re-reduction is recoverable.

**Verdict**: `max_{j, cell} ΔR²_j` from the true assignment (P2's
per-feature R² files) vs the p97.5 quantile of the GLOBAL null. As a
consistency guard the engine ALSO computes the identity-assignment (no
flips) delta through the SAME refit path and records it per cell
(`engine_identity_max_delta`); a large gap to the P2-derived value is
WARNed, never silently absorbed.

**Batched constructs** (`.claude/rules/vectorize-many-cell-fits.md`):
draws are processed in blocks — per (fold, pseudo-stage, block) the
selected rows are gathered as ONE (B, n_tr, d) tensor, the per-draw
normal matrices come from ONE batched `bmm`, the factorizations from ONE
batched `torch.linalg.eigh`, and the GCV scan + smoother formation are
batched over the draw axis. The d_sae=262k target rides a fixed-width
sparse layout ((n, kmax) TopK indices/values); per-feature SS terms are
accumulated by feature-chunked sparse applies (`index_add_`), never an
(n, d_sae) dense GEMM. Device-parametrized (`--device cuda`).

**Estimator notes** (documented deviations, both test-pinned):
- GCV's lambda-selection CRITERION is computed on a fixed seeded feature
  subsample (`--gcv-feature-subsample`, default 1024 columns) — the
  criterion is a SUM over output columns, so a subsample estimates the
  same argmin at ~30x less GCV-GEMM cost; per-feature R² is always
  computed on ALL features. With subsample >= d_sae (tests) the
  criterion — and hence the whole fit — matches
  `ridge_fit_predict_fast_layer_batched` to float noise.
- Folds are computed on the POOLED conversation ids; when the two
  stages carry the SAME conversation set (the #1336 banked corpora),
  these coincide EXACTLY with P2's per-stage folds (same unique-id set,
  same seed 0 permutation).

**Compute character (BINDING — see the Unit B implementation report)**:
a per-draw ridge REFIT battery costs, per (draw, fold, pseudo-stage),
~2*n_tr*d² (normal matrix) + d³ eigh + ~2*n_tr*d² (eigenbasis project)
+ ~2*n_ev*d*n_tr (smoother) at d=4096 — the §9 P3 "masked-GEMM"
cpu-bigmem sizing was written for a residual-permutation statistic and
does NOT cover a refit; production routing needs a GPU (`--device
cuda`) and/or the plan's pre-registered 500-draw descope. The engine is
correct on both devices; wall projections live in the report.

Usage:
  uv run python scripts/issue2061_null.py --all-cells \
      --context-shard-dir data/issue_2061/turnstores \
      --encoded-dir data/issue_2061/sae_encoded [--device cuda]

  uv run python scripts/issue2061_null.py --pair base_sft --corpus lmsys23k \
      --arm context --context-shard-dir ... --encoded-dir ...
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()  # shared-VM thread caps (#847) must bind BEFORE torch/numpy import

import numpy as np  # noqa: E402
import torch  # noqa: E402

# Sibling-script import (bare module name via the script-dir sys.path insert —
# same pattern as issue2061_fit_per_feature.py).
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import issue2061_turnstore as ts  # noqa: E402

STAGES = ["base", "sft", "dpo", "rlvr", "longer-rlvr"]
STAGE_PAIRS = [
    ("base", "sft"),
    ("sft", "dpo"),
    ("dpo", "rlvr"),
    ("rlvr", "longer-rlvr"),
]
LAYER = 29
N_DRAWS = 1000
DRAW_SEED_BASE = 42  # plan §Design synchronization contract
LAMBDA_GRID = np.logspace(-2, 4, 13)  # #823/#779 grid — matches P2
DOF_CAP_FRACTION = 0.9  # #1887 mitigation — matches P2 (plan §11)
K_FOLDS = 5
FOLD_SEED = 0
GCV_FEATURE_SUBSAMPLE = 1024  # GCV criterion columns (see module docstring)
DRAW_BLOCK = 8
FEATURE_CHUNK = 8192
ENGINE_VERSION = "refit-v1"  # partial-checkpoint regime key


def draw_seed_schedule(n_draws: int = N_DRAWS, base: int = DRAW_SEED_BASE) -> list[int]:
    """The ONE shared permutation seed schedule across all 64 cells.

    Load-bearing for the per-draw GLOBAL max reduction: draw index `d`
    uses the SAME seed on every cell, so `max_cell max_j_d` is a coherent
    joint (feature × cell) selection statistic.
    """
    return [base + d for d in range(n_draws)]


def load_per_feature_r2(jsonl_path: Path) -> np.ndarray:
    """Read B-2's per-feature R² JSONL into a (d_sae,) float array.

    Nulls (features with ss_tot == 0) become NaN.
    """
    r2 = []
    with jsonl_path.open() as f:
        for line in f:
            row = json.loads(line)
            r2.append(row["R2"] if row["R2"] is not None else np.nan)
    return np.asarray(r2, dtype=np.float64)


def compute_true_delta_max(
    r2_before: np.ndarray,
    r2_after: np.ndarray,
) -> tuple[float, int]:
    """True ΔR²_j = R²_after - R²_before per feature; return (max, argmax).

    NaN features are excluded from the max (both sides must be non-NaN).
    """
    delta = r2_after - r2_before
    mask = ~np.isnan(delta)
    if not mask.any():
        return float("nan"), -1
    valid = np.where(mask)[0]
    idx_local = int(np.nanargmax(delta[mask]))
    idx = int(valid[idx_local])
    return float(delta[idx]), idx


def to_fixed_width_sparse(y: torch.Tensor, row_chunk: int = 2048) -> tuple[np.ndarray, np.ndarray]:
    """Dense (n, d_sae) -> fixed-width sparse ((n, kmax) idx int64, (n, kmax) val f32).

    The SAE target is TopK (k=32) so kmax is small; padding is (idx=0,
    val=0.0), which is accumulation-safe everywhere the engine consumes it
    (all consumers ADD `val`, never overwrite). Row-chunked so an mmap'd
    dense store never materializes whole.
    """
    n = int(y.shape[0])
    counts = np.empty(n, dtype=np.int64)
    for r0 in range(0, n, row_chunk):
        yc = y[r0 : r0 + row_chunk].to(torch.float32)
        counts[r0 : r0 + yc.shape[0]] = (yc != 0).sum(dim=1).numpy()
    kmax = max(1, int(counts.max()) if n else 1)
    idx = np.zeros((n, kmax), dtype=np.int64)
    val = np.zeros((n, kmax), dtype=np.float32)
    for r0 in range(0, n, row_chunk):
        yc = y[r0 : r0 + row_chunk].to(torch.float32).numpy()
        for i in range(yc.shape[0]):
            nz = np.nonzero(yc[i])[0]
            idx[r0 + i, : len(nz)] = nz
            val[r0 + i, : len(nz)] = yc[i, nz]
    return idx, val


class _CellEngine:
    """Per-cell permute-and-refit engine (see module docstring).

    Holds the pooled (before + after) rows on `device` and evaluates the
    per-pseudo-stage per-feature R² for a BLOCK of stage-label pair-flip
    assignments at once (`r2_for_flips`). The identity assignment (all
    flips False) reproduces the P2 estimator per stage exactly (pinned by
    tests/test_issue2061_stats.py::test_engine_identity_matches_p2_estimator).
    """

    def __init__(
        self,
        x_before: np.ndarray,
        y_idx_before: np.ndarray,
        y_val_before: np.ndarray,
        conv_before: list[str],
        x_after: np.ndarray,
        y_idx_after: np.ndarray,
        y_val_after: np.ndarray,
        conv_after: list[str],
        *,
        d_sae: int,
        lambdas: np.ndarray | None = None,
        k_folds: int = K_FOLDS,
        fold_seed: int = FOLD_SEED,
        gcv_dof_cap: float | None = DOF_CAP_FRACTION,
        gcv_m: int = GCV_FEATURE_SUBSAMPLE,
        gcv_subsample_seed: int = 0,
        feature_chunk: int = FEATURE_CHUNK,
        device: str = "cpu",
    ) -> None:
        if len(set(conv_before)) != len(conv_before):
            raise ValueError("duplicate conv ids in the BEFORE stage turnstore")
        if len(set(conv_after)) != len(conv_after):
            raise ValueError("duplicate conv ids in the AFTER stage turnstore")
        nb, na = x_before.shape[0], x_after.shape[0]
        assert x_before.shape[1] == x_after.shape[1], (x_before.shape, x_after.shape)
        assert nb == len(conv_before) == y_idx_before.shape[0] == y_val_before.shape[0]
        assert na == len(conv_after) == y_idx_after.shape[0] == y_val_after.shape[0]

        self.dev = torch.device(device)
        self.d_sae = int(d_sae)
        self.lambdas = LAMBDA_GRID if lambdas is None else np.asarray(lambdas, dtype=np.float64)
        self.k_folds = int(k_folds)
        self.gcv_dof_cap = gcv_dof_cap
        self.feature_chunk = int(feature_chunk)

        self.X = torch.cat(
            [
                torch.as_tensor(x_before, dtype=torch.float64),
                torch.as_tensor(x_after, dtype=torch.float64),
            ]
        ).to(self.dev)
        self.N, self.d = self.X.shape

        # Fixed-width sparse target, padded to a common kmax across stages.
        kmax = max(y_idx_before.shape[1], y_idx_after.shape[1])

        def _pad(a: np.ndarray, fill) -> np.ndarray:
            if a.shape[1] == kmax:
                return a
            out = np.full((a.shape[0], kmax), fill, dtype=a.dtype)
            out[:, : a.shape[1]] = a
            return out

        self.kmax = kmax
        self.Yidx = torch.as_tensor(
            np.concatenate([_pad(y_idx_before, 0), _pad(y_idx_after, 0)]), dtype=torch.int64
        ).to(self.dev)
        self.Yval = torch.as_tensor(
            np.concatenate([_pad(y_val_before, 0.0), _pad(y_val_after, 0.0)]),
            dtype=torch.float64,
        ).to(self.dev)
        if int(self.Yidx.max()) >= self.d_sae:
            raise ValueError(f"sparse feature index {int(self.Yidx.max())} >= d_sae={self.d_sae}")

        # Pairing (the permutable unit is the conversation present in BOTH stages).
        pos_b = {c: i for i, c in enumerate(conv_before)}
        pos_a = {c: i for i, c in enumerate(conv_after)}
        paired = sorted(set(conv_before) & set(conv_after))
        if not paired:
            raise ValueError(
                "no conversation id appears in BOTH stages — a within-corpus "
                "stage-label permutation is undefined for this cell"
            )
        self.n_paired = len(paired)
        ib_p = np.asarray([pos_b[c] for c in paired], dtype=np.int64)
        ia_p = np.asarray([pos_a[c] + nb for c in paired], dtype=np.int64)
        paired_set = set(paired)
        ub = np.asarray(
            [i for i, c in enumerate(conv_before) if c not in paired_set], dtype=np.int64
        )
        ua = np.asarray(
            [i + nb for i, c in enumerate(conv_after) if c not in paired_set], dtype=np.int64
        )
        self.n_unpaired_before = int(len(ub))
        self.n_unpaired_after = int(len(ua))

        # GROUP-level folds on the POOLED conversation ids (#1336 convention).
        pooled_conv = list(conv_before) + list(conv_after)
        folds = ts.group_fold_ids(pooled_conv, n_folds=self.k_folds, seed=fold_seed)
        assert (folds[ib_p] == folds[ia_p]).all(), "paired rows must share a fold"

        # Per-(fold, group) selection plans. Group 0 = pseudo-BEFORE (natural
        # row = the before-stage row; a flip swaps in the after-stage row),
        # group 1 = pseudo-AFTER (natural = after row; flip -> before row).
        pf = folds[ib_p]  # fold id per paired conv
        self._plans: list[list[dict]] = []
        for f in range(self.k_folds):
            ptr = np.where(pf != f)[0]
            pev = np.where(pf == f)[0]
            per_group = []
            for g in (0, 1):
                nat, alt = (ib_p, ia_p) if g == 0 else (ia_p, ib_p)
                u = ub if g == 0 else ua
                u_tr = u[folds[u] != f]
                u_ev = u[folds[u] == f]
                per_group.append(
                    {
                        "ptr": torch.as_tensor(ptr, device=self.dev),
                        "pev": torch.as_tensor(pev, device=self.dev),
                        "nat_tr": torch.as_tensor(nat[ptr], device=self.dev),
                        "alt_tr": torch.as_tensor(alt[ptr], device=self.dev),
                        "nat_ev": torch.as_tensor(nat[pev], device=self.dev),
                        "alt_ev": torch.as_tensor(alt[pev], device=self.dev),
                        "u_tr": torch.as_tensor(u_tr, device=self.dev),
                        "u_ev": torch.as_tensor(u_ev, device=self.dev),
                    }
                )
            self._plans.append(per_group)

        # Dense GCV-criterion column subsample (see module docstring).
        m = min(int(gcv_m), self.d_sae)
        cols = np.sort(
            np.random.default_rng(gcv_subsample_seed).choice(self.d_sae, size=m, replace=False)
        )
        lookup = np.full(self.d_sae, -1, dtype=np.int64)
        lookup[cols] = np.arange(m)
        ysub = np.zeros((self.N, m), dtype=np.float64)
        idx_np = self.Yidx.cpu().numpy()
        val_np = self.Yval.cpu().numpy()
        pos = lookup[idx_np]  # (N, kmax); padded entries add val 0.0
        rows, kk = np.nonzero(pos >= 0)
        np.add.at(ysub, (rows, pos[rows, kk]), val_np[rows, kk])
        self.gcv_m = m
        self.Ysub = torch.as_tensor(ysub, dtype=torch.float64).to(self.dev)

    # -- selection helpers ---------------------------------------------------
    def _sel(self, plan: dict, flips_t: torch.Tensor, which: str) -> torch.Tensor:
        """(B, n_rows) pooled row indices for one (fold, group) selection."""
        pidx = plan["ptr"] if which == "tr" else plan["pev"]
        nat = plan[f"nat_{which}"]
        alt = plan[f"alt_{which}"]
        u = plan[f"u_{which}"]
        B = flips_t.shape[0]
        sel_p = torch.where(flips_t[:, pidx], alt.expand(B, -1), nat.expand(B, -1))
        if len(u):
            sel_p = torch.cat([sel_p, u.expand(B, -1)], dim=1)
        return sel_p

    # -- the batched refit ----------------------------------------------------
    def r2_for_flips(self, flips: np.ndarray) -> np.ndarray:
        """Per-feature pooled R² for a BLOCK of flip assignments.

        Args:
            flips: (B, n_paired) bool — True swaps that conversation's
                (before, after) rows between the pseudo-stages.

        Returns:
            (B, 2, d_sae) float64 — pooled-over-folds R² per pseudo-stage
            (index 0 = pseudo-before, 1 = pseudo-after); NaN where a
            feature's ss_tot is 0 in that pseudo-stage.
        """
        assert flips.ndim == 2 and flips.shape[1] == self.n_paired, flips.shape
        B = flips.shape[0]
        flips_t = torch.as_tensor(np.asarray(flips, dtype=bool), device=self.dev)
        ss_res = torch.zeros((B, 2, self.d_sae), dtype=torch.float64, device=self.dev)
        ss_tot = torch.zeros_like(ss_res)
        lambdas_t = torch.as_tensor(self.lambdas, dtype=torch.float64, device=self.dev)

        for f in range(self.k_folds):
            for g in (0, 1):
                plan = self._plans[f][g]
                sel_tr = self._sel(plan, flips_t, "tr")  # (B, ntr)
                sel_ev = self._sel(plan, flips_t, "ev")  # (B, nev)
                ntr, nev = sel_tr.shape[1], sel_ev.shape[1]
                if ntr == 0 or nev == 0:
                    raise RuntimeError(f"empty train/eval selection at fold {f} group {g}")

                Xtr = self.X[sel_tr]  # (B, ntr, d)
                mu = Xtr.mean(dim=1, keepdim=True)
                sd = Xtr.std(dim=1, keepdim=True, unbiased=False) + 1e-9  # twin parity
                Xn = (Xtr - mu) / sd
                Xev = (self.X[sel_ev] - mu) / sd

                A = Xn.transpose(1, 2) @ Xn  # (B, d, d)
                w, V = torch.linalg.eigh(A)
                w = torch.clamp(w, min=0.0)
                P = Xn @ V  # (B, ntr, d)
                Pev = Xev @ V  # (B, nev, d)

                # GCV over the grid on the criterion column subsample.
                Ysub_tr = self.Ysub[sel_tr]  # (B, ntr, m)
                Yc = Ysub_tr - Ysub_tr.mean(dim=1, keepdim=True)
                C = P.transpose(1, 2) @ Yc  # (B, d, m)
                # P = Xn V has column norms sqrt(w) (NOT orthonormal): the GCV
                # RSS identity needs the orthonormal-basis component energies
                # ||U^T Yc||_k^2 = sq_k / w_k (zero-eigenvalue components carry
                # zero column-space energy). Matches the Gram-basis sqVtY of
                # `ridge_fit_predict_fast_layer_batched` exactly.
                sq_raw = (C**2).sum(dim=2)  # (B, d)
                wmax = torch.clamp(w.max(dim=1, keepdim=True).values, min=1e-300)
                sq = torch.where(
                    w > 1e-12 * wmax,
                    sq_raw / torch.clamp(w, min=1e-300),
                    torch.zeros_like(sq_raw),
                )
                tot = (Yc**2).sum(dim=(1, 2))  # (B,)
                gcv_all = torch.empty((B, len(self.lambdas)), dtype=torch.float64, device=self.dev)
                for li, lam in enumerate(self.lambdas):
                    filt = w / (w + float(lam))  # (B, d)
                    rss = tot - ((2 * filt - filt**2) * sq).sum(dim=1)
                    dof = filt.sum(dim=1)
                    denom = (ntr - dof) ** 2
                    vals = torch.where(
                        denom > 1e-12, rss / denom, torch.full_like(rss, float("inf"))
                    )
                    if self.gcv_dof_cap is not None:
                        vals = torch.where(
                            dof <= self.gcv_dof_cap * ntr,
                            vals,
                            torch.full_like(vals, float("inf")),
                        )
                    gcv_all[:, li] = vals
                if self.gcv_dof_cap is not None and bool(torch.isinf(gcv_all).all(dim=1).any()):
                    raise RuntimeError(
                        f"gcv_dof_cap={self.gcv_dof_cap}: every lambda capped out at "
                        f"fold {f} group {g} (n_tr={ntr}) — widen the grid (#1887)."
                    )
                best_lam = lambdas_t[gcv_all.argmin(dim=1)]  # (B,)

                fprime = 1.0 / (w + best_lam[:, None])  # (B, d)
                T = (Pev * fprime[:, None, :]) @ P.transpose(1, 2)  # (B, nev, ntr)
                t1 = T.sum(dim=2)  # (B, nev)

                # Column means (dense semantics over the sparse layout).
                idx_tr = self.Yidx[sel_tr]  # (B, ntr, kmax)
                val_tr = self.Yval[sel_tr]
                idx_ev = self.Yidx[sel_ev]
                val_ev = self.Yval[sel_ev]
                ybar_tr = torch.zeros((B, self.d_sae), dtype=torch.float64, device=self.dev)
                ybar_tr.scatter_add_(1, idx_tr.reshape(B, -1), val_tr.reshape(B, -1))
                ybar_tr /= ntr
                ybar_ev = torch.zeros_like(ybar_tr)
                ybar_ev.scatter_add_(1, idx_ev.reshape(B, -1), val_ev.reshape(B, -1))
                ybar_ev /= nev

                rt = torch.arange(ntr, device=self.dev).repeat_interleave(self.kmax)
                re_ = torch.arange(nev, device=self.dev).repeat_interleave(self.kmax)
                csz_full = self.feature_chunk
                for b in range(B):
                    Tb = T[b]  # (nev, ntr)
                    it = idx_tr[b].reshape(-1)
                    vt = val_tr[b].reshape(-1)
                    ie = idx_ev[b].reshape(-1)
                    ve = val_ev[b].reshape(-1)
                    for c0 in range(0, self.d_sae, csz_full):
                        c1 = min(c0 + csz_full, self.d_sae)
                        csz = c1 - c0
                        ybtr = ybar_tr[b, c0:c1]
                        # yhat = T @ (Y_tr - 1 ybar^T) + ybar  (chunked, sparse apply)
                        yhat = torch.zeros((nev, csz), dtype=torch.float64, device=self.dev)
                        mtr = (it >= c0) & (it < c1)
                        if bool(mtr.any()):
                            jl = it[mtr] - c0
                            yhat.index_add_(1, jl, Tb[:, rt[mtr]] * vt[mtr])
                        yhat -= t1[b][:, None] * ybtr[None, :]
                        yhat += ybtr[None, :]
                        # dense eval chunk (accumulation-safe scatter)
                        yev = torch.zeros((nev, csz), dtype=torch.float64, device=self.dev)
                        mev = (ie >= c0) & (ie < c1)
                        if bool(mev.any()):
                            flat = re_[mev] * csz + (ie[mev] - c0)
                            yev.view(-1).index_add_(0, flat, ve[mev])
                        ss_res[b, g, c0:c1] += ((yev - yhat) ** 2).sum(dim=0)
                        ss_tot[b, g, c0:c1] += ((yev - ybar_ev[b, c0:c1][None, :]) ** 2).sum(dim=0)

        ratio = ss_res / torch.clamp(ss_tot, min=1e-300)
        r2 = torch.where(ss_tot > 0, 1.0 - ratio, torch.full_like(ratio, float("nan")))
        return r2.cpu().numpy()


def flips_for_seeds(seeds: list[int], n_paired: int) -> np.ndarray:
    """(B, n_paired) bool pair-flip assignments, one rng per draw seed.

    Derived per-draw from that draw's OWN seed, so results are invariant
    to draw-block partitioning (pinned by the block-invariance test).
    """
    return np.stack([np.random.default_rng(s).random(n_paired) < 0.5 for s in seeds], axis=0)


def identity_delta_r2(engine: _CellEngine) -> dict:
    """Observed (no-permutation) per-feature Δ through the SAME refit path."""
    r2 = engine.r2_for_flips(np.zeros((1, engine.n_paired), dtype=bool))
    r2_before, r2_after = r2[0, 0], r2[0, 1]
    delta = r2_after - r2_before
    mask = np.isfinite(delta)
    if not mask.any():
        raise RuntimeError("identity assignment has no co-active features")
    valid = np.where(mask)[0]
    j = int(valid[np.argmax(delta[mask])])
    return {
        "delta": delta,
        "r2_before": r2_before,
        "r2_after": r2_after,
        "max": float(delta[j]),
        "argmax": j,
    }


def per_cell_null_refit(
    engine: _CellEngine,
    seeds: list[int],
    *,
    draw_block: int = DRAW_BLOCK,
    partial_path: Path | None = None,
    partial_meta: dict | None = None,
    progress_label: str = "",
) -> np.ndarray:
    """(n_draws,) float32 of max_j ΔR²_j under the registered permutation null.

    Draws run in blocks through `engine.r2_for_flips`; after every block the
    accumulated draws are checkpointed atomically to `partial_path` (keyed on
    `partial_meta` — EVERY output-affecting regime key; a mismatched partial
    is ignored, never silently reused). Per-block progress lines satisfy the
    per-unit-progress contract (`.claude/rules/code-style.md`).
    """
    n_draws = len(seeds)
    out = np.full(n_draws, np.nan, dtype=np.float32)
    start = 0
    meta_str = json.dumps(partial_meta or {}, sort_keys=True)
    if partial_path is not None and partial_path.exists():
        try:
            prev = np.load(partial_path, allow_pickle=False)
            if str(prev["meta"]) == meta_str and int(prev["n_done"]) <= n_draws:
                start = int(prev["n_done"])
                out[:start] = prev["draws"][:start]
                print(f"[null]{progress_label} resumed at draw {start}/{n_draws}", flush=True)
            else:
                print(f"[null]{progress_label} partial meta mismatch — recomputing", flush=True)
        except Exception as e:  # corrupt partial: recompute, never trust it
            print(f"[null]{progress_label} unreadable partial ({e}) — recomputing", flush=True)
    t0 = time.time()
    for b0 in range(start, n_draws, draw_block):
        b1 = min(b0 + draw_block, n_draws)
        flips = flips_for_seeds(seeds[b0:b1], engine.n_paired)
        r2 = engine.r2_for_flips(flips)  # (B, 2, d_sae)
        delta = r2[:, 1, :] - r2[:, 0, :]
        for i in range(b1 - b0):
            d_i = delta[i]
            mask = np.isfinite(d_i)
            if not mask.any():
                raise RuntimeError(f"draw {b0 + i}: no co-active features in either group")
            out[b0 + i] = float(np.max(d_i[mask]))
        print(
            f"[null]{progress_label} draws {b1}/{n_draws} elapsed={time.time() - t0:.1f}s",
            flush=True,
        )
        if partial_path is not None:
            # Atomic same-dir write. The tmp name KEEPS the .npz suffix —
            # np.savez APPENDS `.npz` to any name lacking it, which would
            # strand the os.replace source (the #1092 gotcha).
            tmp = partial_path.with_name(partial_path.name.removesuffix(".npz") + ".tmp.npz")
            np.savez(tmp, draws=out, n_done=np.int64(b1), meta=np.str_(meta_str))
            os.replace(tmp, partial_path)
    return out


def write_cell_jsonl(
    output_path: Path,
    pair: tuple[str, str],
    corpus: str,
    arm: str,
    true_max: float,
    true_argmax: int,
    null_max_j_per_draw: np.ndarray,
    extra: dict | None = None,
) -> None:
    """Emit per-cell record with per-draw max_j_d + local quantiles."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    quantiles = {f"p{q}": float(np.percentile(null_max_j_per_draw, q)) for q in [50, 95, 97.5, 99]}
    row = {
        "pair": f"{pair[0]}_{pair[1]}",
        "corpus": corpus,
        "arm": arm,
        "layer": LAYER,
        "true_max_delta_r2": true_max,
        "true_argmax_feature_id": true_argmax,
        "null_quantiles_per_cell": quantiles,
        "null_max_j_per_draw": null_max_j_per_draw.astype(np.float32).tolist(),
        "n_draws": len(null_max_j_per_draw),
        "draw_seed_base": DRAW_SEED_BASE,
    }
    if extra:
        row.update(extra)
    with output_path.open("w") as f:
        f.write(json.dumps(row) + "\n")


def compute_global_null(
    per_cell_max_j: dict[tuple[str, str, str, str], np.ndarray],
) -> dict:
    """Per-draw GLOBAL max reduction (PRIMARY headline null quantile).

    For each draw d, form `global_max_d = max_cell max_j_d`. Report
    p50/p95/p97.5/p99 of this GLOBAL null distribution.
    """
    if not per_cell_max_j:
        raise ValueError("No per-cell null draws available.")
    stacked = np.stack(list(per_cell_max_j.values()), axis=0)  # (n_cells, n_draws)
    global_max_per_draw = stacked.max(axis=0)  # (n_draws,)
    return {
        "global_null_quantiles": {
            f"p{q}": float(np.percentile(global_max_per_draw, q)) for q in [50, 95, 97.5, 99]
        },
        "global_max_per_draw": global_max_per_draw.astype(np.float32).tolist(),
        "n_cells": len(per_cell_max_j),
        "n_draws": stacked.shape[1],
        "draw_seed_base": DRAW_SEED_BASE,
        "cells": [
            {"pair": p, "corpus": c, "arm": a, "layer": L} for (p, c, a, L) in per_cell_max_j.keys()
        ],
    }


def _load_r2_file(
    r2_dir: Path,
    stage: str,
    corpus: str,
    arm: str,
    render: str = "chat",
) -> np.ndarray | None:
    """Try both naming conventions from B-2."""
    candidates = [
        r2_dir / f"{stage}_{render}_{corpus}_{arm}_L{LAYER}.jsonl",
        r2_dir / f"{stage}_{corpus}_{arm}_L{LAYER}.jsonl",
    ]
    for path in candidates:
        if path.exists():
            return load_per_feature_r2(path)
    return None


def _load_cell_stage_inputs(
    context_shard_dir: Path,
    encoded_dir: Path,
    stage: str,
    render: str,
    corpus: str,
    arm: str,
    layer: int = LAYER,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], int]:
    """(X, y_idx, y_val, conv_ids, d_sae) for one stage of a delta cell."""
    ts_dir = context_shard_dir / f"turnstore_{stage}_{render}_{corpus}"
    enc = encoded_dir / f"{stage}_{render}_{corpus}_answer_L{layer}.pt"
    if not ts_dir.is_dir():
        raise FileNotFoundError(f"missing turnstore dir {ts_dir}")
    if not enc.exists():
        raise FileNotFoundError(f"missing encoded target {enc}")
    shard_paths = ts.enumerate_shards(ts_dir)
    x, conv_ids = ts.load_state_from_shards(shard_paths, state=arm, layer=layer)
    y = torch.load(enc, map_location="cpu", weights_only=True, mmap=True)
    if y.shape[0] != x.shape[0]:
        raise ValueError(f"row mismatch: turnstore n={x.shape[0]} vs encoded n={y.shape[0]}")
    y_idx, y_val = to_fixed_width_sparse(y)
    return x.numpy(), y_idx, y_val, conv_ids, int(y.shape[1])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pair", type=str, default=None, help="e.g. 'base_sft'; omit for all 4 pairs"
    )
    parser.add_argument("--corpus", type=str, default=None)
    parser.add_argument("--render", type=str, default="chat")
    parser.add_argument("--arm", choices=["prefix", "context"], default=None)
    parser.add_argument("--all-cells", action="store_true")
    parser.add_argument(
        "--r2-dir", type=Path, default=Path("eval_results/issue_2061/per_feature_r2")
    )
    parser.add_argument(
        "--context-shard-dir",
        type=Path,
        default=None,
        help="Directory holding the #1336 turnstore dirs (required to COMPUTE cells; "
        "not needed when every cell's output already exists).",
    )
    parser.add_argument("--encoded-dir", type=Path, default=Path("data/issue_2061/sae_encoded"))
    parser.add_argument("--output-dir", type=Path, default=Path("eval_results/issue_2061/null"))
    parser.add_argument("--n-draws", type=int, default=N_DRAWS)
    parser.add_argument("--draw-block", type=int, default=DRAW_BLOCK)
    parser.add_argument("--gcv-feature-subsample", type=int, default=GCV_FEATURE_SUBSAMPLE)
    parser.add_argument("--feature-chunk", type=int, default=FEATURE_CHUNK)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    seeds = draw_seed_schedule(args.n_draws)
    print(f"[setup] Shared draw_seed_schedule: {len(seeds)} seeds, base={DRAW_SEED_BASE}")

    # Enumerate target (pair, corpus, arm) cells.
    pairs = [tuple(args.pair.split("_", 1))] if args.pair else STAGE_PAIRS
    arms = [args.arm] if args.arm else ["prefix", "context"]

    # Auto-detect corpora from the r2_dir.
    if args.corpus:
        corpora = [args.corpus]
    elif args.all_cells:
        seen = set()
        for path in args.r2_dir.glob(f"*_L{LAYER}.jsonl"):
            # <stage>_<render>_<corpus>_<arm> or <stage>_<corpus>_<arm>
            parts = path.stem.rsplit("_", 3)
            if len(parts) >= 3:
                seen.add(parts[-3])  # corpus is 3rd-from-last
        corpora = sorted(seen)
    else:
        corpora = []

    if not corpora:
        print("[error] No corpora found (use --corpus or --all-cells)")
        return 1

    per_cell_max_j: dict[tuple[str, str, str, str], np.ndarray] = {}
    for pair in pairs:
        for corpus in corpora:
            for arm in arms:
                cell_key = (f"{pair[0]}_{pair[1]}", corpus, arm, f"L{LAYER}")
                output_path = args.output_dir / f"{cell_key[0]}_{corpus}_{arm}_L{LAYER}.jsonl"
                if output_path.exists():
                    print(f"[skip] Exists: {output_path}")
                    # Reload the per-draw values for global aggregation.
                    with output_path.open() as f:
                        row = json.loads(f.readline())
                    per_cell_max_j[cell_key] = np.asarray(
                        row["null_max_j_per_draw"], dtype=np.float32
                    )
                    continue

                r2_before = _load_r2_file(args.r2_dir, pair[0], corpus, arm, args.render)
                r2_after = _load_r2_file(args.r2_dir, pair[1], corpus, arm, args.render)
                if r2_before is None or r2_after is None:
                    print(f"[skip] Missing R² for {pair}/{corpus}/{arm}")
                    continue
                if args.context_shard_dir is None:
                    print(
                        f"[error] {cell_key}: null output missing and --context-shard-dir "
                        "not given — cannot refit. Pass the turnstore root."
                    )
                    return 1

                try:
                    xb, yib, yvb, cb, dsb = _load_cell_stage_inputs(
                        args.context_shard_dir, args.encoded_dir, pair[0], args.render, corpus, arm
                    )
                    xa, yia, yva, ca, dsa = _load_cell_stage_inputs(
                        args.context_shard_dir, args.encoded_dir, pair[1], args.render, corpus, arm
                    )
                except FileNotFoundError as e:
                    print(f"[skip] Missing raw inputs for {pair}/{corpus}/{arm}: {e}")
                    continue
                if dsb != dsa:
                    raise ValueError(f"d_sae mismatch across stages: {dsb} vs {dsa}")
                if dsb != len(r2_before):
                    raise ValueError(
                        f"d_sae mismatch: encoded {dsb} vs P2 R² file {len(r2_before)}"
                    )

                t0 = time.time()
                true_max, true_argmax = compute_true_delta_max(r2_before, r2_after)
                engine = _CellEngine(
                    xb,
                    yib,
                    yvb,
                    cb,
                    xa,
                    yia,
                    yva,
                    ca,
                    d_sae=dsb,
                    gcv_m=args.gcv_feature_subsample,
                    feature_chunk=args.feature_chunk,
                    device=args.device,
                )
                ident = identity_delta_r2(engine)
                if np.isfinite(true_max) and abs(ident["max"] - true_max) > 0.02:
                    print(
                        f"[WARN] {cell_key}: engine identity max Δ={ident['max']:.4f} vs "
                        f"P2-derived {true_max:.4f} (gap > 0.02) — check estimator parity"
                    )
                meta = {
                    "engine_version": ENGINE_VERSION,
                    "pair": cell_key[0],
                    "corpus": corpus,
                    "arm": arm,
                    "render": args.render,
                    "layer": LAYER,
                    "n_draws": args.n_draws,
                    "draw_seed_base": DRAW_SEED_BASE,
                    "k_folds": K_FOLDS,
                    "fold_seed": FOLD_SEED,
                    "gcv_dof_cap": DOF_CAP_FRACTION,
                    "gcv_feature_subsample": engine.gcv_m,
                    "lambdas": [float(v) for v in LAMBDA_GRID],
                    "n_rows_before": int(xb.shape[0]),
                    "n_rows_after": int(xa.shape[0]),
                }
                partial_path = output_path.with_name(output_path.name + ".partial.npz")
                label = f" {cell_key[0]}/{corpus}/{arm}"
                null_max_j = per_cell_null_refit(
                    engine,
                    seeds,
                    draw_block=args.draw_block,
                    partial_path=partial_path,
                    partial_meta=meta,
                    progress_label=label,
                )
                elapsed = time.time() - t0

                write_cell_jsonl(
                    output_path,
                    pair=pair,
                    corpus=corpus,
                    arm=arm,
                    true_max=true_max,
                    true_argmax=true_argmax,
                    null_max_j_per_draw=null_max_j,
                    extra={
                        **meta,
                        "engine_identity_max_delta": ident["max"],
                        "engine_identity_argmax": ident["argmax"],
                        "n_paired": engine.n_paired,
                        "n_unpaired_before": engine.n_unpaired_before,
                        "n_unpaired_after": engine.n_unpaired_after,
                        "wall_s": elapsed,
                    },
                )
                if partial_path.exists():
                    partial_path.unlink()
                per_cell_max_j[cell_key] = null_max_j
                print(
                    f"[cell] {cell_key[0]}/{corpus}/{arm} true_max={true_max:.4f} "
                    f"argmax={true_argmax} local_p97.5={np.percentile(null_max_j, 97.5):.4f} "
                    f"({elapsed:.1f}s)"
                )

    if not per_cell_max_j:
        print("[error] No per-cell nulls computed — cannot form GLOBAL null")
        return 1

    global_null = compute_global_null(per_cell_max_j)
    global_path = args.output_dir / f"GLOBAL_L{LAYER}.json"
    with global_path.open("w") as f:
        json.dump(global_null, f, indent=2)

    print(f"\n[global] GLOBAL null (p97.5): {global_null['global_null_quantiles']['p97.5']:.4f}")
    print(f"[global] Wrote {global_path}")
    print(f"[global] n_cells={global_null['n_cells']} n_draws={global_null['n_draws']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
