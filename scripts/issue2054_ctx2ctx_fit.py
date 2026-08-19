"""Issue #2054 Gap A: ridge maps fit DIRECTLY on row-paired context vectors.

Fits R : v_C^src -> v_C^tgt (context arm) and R_P : v_P^src -> v_P^tgt (prefix
arm, restricted to rows with ``v_P_present`` on BOTH sides), rows joined on
``conv_id``, scored by held-out R^2 on the TARGET cell's vector. This is NOT
``scripts/issue2054_ladder.py`` rung 7 (which fits a context-side matrix
minimizing ANSWER-prediction error through the source map) — same solver
family, different regression target.

Pair enumeration (the production run; ``--pilot`` runs ONE pair, fold 0):
  (a) framing pairs, assistant identity: ordered pairs of distinct framings
      within each (condition, model);
  (b) identity pairs within a fixed story framing: ordered pairs of distinct
      identities within each (framing, condition, model) with >= 2 identities.

Reused cores (do-not-reimplement contract):
  - GCV-ridge math mirrors ``experiments.issue_779.fit_h.ridge_fit_predict_fast``
    / ``ridge_fit_predict_fast_layer_batched`` (standardize-X population sd,
    center-Y, eigen dual solve, GCV with the #1887 dof cap). Here the
    eigendecomposition is computed ONCE per (source cell, fold, arm) and
    reused across every target sharing that source, the whole lambda grid,
    and the shuffled-pair null draws — parity vs the reused core is asserted
    at pilot time (``_parity_gate``).
  - ``analysis.mapping_baselines.identity_bias_predict`` / ``knn_retrieval``
    run on every fit (identity+bias is expected STRONG here: two renders of
    the same conversation may differ by little more than an offset — a fitted
    R^2 that does not clear it is a NULL, made legible in the output JSON).

Estimator validity (enforced): d = 3584; n_train ~ 0.8 x paired intersection.
n_train > d -> ambient basis (regime "ambient"); n_train <= d -> train-fold
PCA-k=1024 reduced INPUT basis (regime "reduced_basis_descriptive", no ambient
headline; mirrors ``issue2054_fits._reduced_basis_r2``: X reduced, Y ambient).
Pure-GCV selection at n_train < d_fit RAISES; GCV always runs with dof cap 0.9.

Shuffled-pair null: fit-side row permutation of Y_train (the
``issue2054_fits._shuffled_answer_null_r2`` convention) — refit per draw with
the SHARED eigenbasis (only the Y-side projections recompute), score vs the
unpermuted held-out targets; draws batched as chunked GEMMs.

Fold split: the PRODUCTION shared fold map (K=5 conversation-grouped, seed
137) read from the issue-2054 branch blob (the working-tree copy on main is a
smoke map and is REFUSED by the same floors as
``issue2054_cross_render_fit._load_production_fold_map``).

Outputs: one JSON per (source cell, arm) unit under
``<out-root>/percell/`` (checkpoint-per-unit + fingerprinted resume);
``--pilot`` writes under ``<out-root>/pilot/percell/``.
"""

from __future__ import annotations

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()  # thread caps + HF/creds BEFORE torch import (code-style.md)

import argparse
import dataclasses
import hashlib
import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

from explore_persona_space.analysis.mapping_baselines import identity_bias_predict, knn_retrieval
from explore_persona_space.experiments.issue_779.fit_h import (
    reconstruction_metrics,
    ridge_fit_predict_fast_layer_batched,
)
from explore_persona_space.orchestrate.provenance import as_metadata_dict, git_provenance

_REPO = Path(__file__).resolve().parents[1]

SCRIPT_VERSION = "issue2054_ctx2ctx_fit_v1"
ASSISTANT_IDENTITY = "conversation_paired_stories_assistant"
D_AMBIENT = 3584
REDUCED_BASIS_K = 1024
GCV_DOF_CAP = 0.9
DEFAULT_LAMBDAS = np.logspace(-2, 4, 13)  # the #823/#779 grid (fit_h default)
NULL_SEED_BASE = 137

FOLD_MAP_PATH_IN_REPO = "eval_results/issue_2054/shared_fold_map.json"
FOLD_MAP_MIN_CONV = 20_000  # smoke-map refusal floors (mirror issue2054_cross_render_fit)
FOLD_MAP_MIN_VARIANTS = 5
MIN_JOIN_ABS = 200
# Of the smaller cell's row count. 0.20, NOT 0.5: cross-variant and prose-swap
# pairs legitimately share only part of the conversation pool — this issue's own
# measured prose-swap intersections are 2,939-4,450 (36-44% of the smaller cell),
# and those pairs are exactly what the reduced-basis regime below exists to
# handle. A 0.5 floor rejected them as "unexpectedly small" and aborted the whole
# run on the first one. MIN_JOIN_ABS is what actually catches a BROKEN join (a
# conv_id key/dtype/format bug yields ~0 overlap, not 44%); this fraction only
# guards against a catastrophic partial-load. The realized n_join is recorded per
# pair, so a surprising-but-admitted drop stays visible in the artifact.
MIN_JOIN_FRAC = 0.20

ARMS = ("context", "prefix")
ARM_VEC_KEY = {"context": "v_C", "prefix": "v_P"}


def _log(msg: str) -> None:
    print(msg, flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# Cells + pair enumeration


@dataclasses.dataclass(frozen=True)
class Cell:
    identity: str
    condition: str
    framing: str
    model: str
    path: Path

    @property
    def key(self) -> str:
        return f"{self.identity}__{self.condition}__{self.framing}__{self.model}"


def discover_cells(activations_dir: Path) -> list[Cell]:
    """Parse every ``<identity>__<condition>__<framing>__<model>.npz`` in the dir."""
    if not activations_dir.is_dir():
        raise FileNotFoundError(f"--activations-dir not a directory: {activations_dir}")
    cells: list[Cell] = []
    for p in sorted(activations_dir.glob("*.npz")):
        parts = p.stem.split("__")
        if len(parts) != 4:
            raise ValueError(
                f"activation filename does not parse as identity__condition__framing__model: {p}"
            )
        cells.append(Cell(parts[0], parts[1], parts[2], parts[3], p))
    if not cells:
        raise FileNotFoundError(f"no .npz activation cells under {activations_dir}")
    return cells


def enumerate_pairs(cells: list[Cell]) -> list[dict]:
    """The Gap-A pair families: (a) framing pairs (assistant identity) and
    (b) identity pairs within a fixed story framing. Data-driven from the files
    present, so the same entrypoint covers the pilot 16-file set and the full
    90-file pod set."""
    by_key = {c.key: c for c in cells}
    if len(by_key) != len(cells):
        raise ValueError("duplicate cell keys in --activations-dir")
    pairs: list[dict] = []
    # (a) framing pairs, assistant identity, within (condition, model).
    groups_a: dict[tuple[str, str], list[Cell]] = {}
    for c in cells:
        if c.identity == ASSISTANT_IDENTITY:
            groups_a.setdefault((c.condition, c.model), []).append(c)
    for (_cond, _model), members in sorted(groups_a.items()):
        members = sorted(members, key=lambda c: c.framing)
        for src in members:
            for tgt in members:
                if src.framing != tgt.framing:
                    pairs.append({"src": src, "tgt": tgt, "family": "framing_pairs"})
    # (b) identity pairs within a fixed framing, within (framing, condition, model).
    groups_b: dict[tuple[str, str, str], list[Cell]] = {}
    for c in cells:
        groups_b.setdefault((c.framing, c.condition, c.model), []).append(c)
    for (_framing, _cond, _model), members in sorted(groups_b.items()):
        if len(members) < 2:
            continue  # only one identity rendered under this framing (e.g. chat)
        members = sorted(members, key=lambda c: c.identity)
        for src in members:
            for tgt in members:
                if src.identity != tgt.identity:
                    pairs.append({"src": src, "tgt": tgt, "family": "identity_pairs"})
    pairs.sort(key=lambda p: (p["family"], p["src"].key, p["tgt"].key))
    return pairs


# ─────────────────────────────────────────────────────────────────────────────
# Fold map (production blob; smoke map refused)


def load_fold_map(fold_map_file: str | None, fold_map_ref: str) -> dict:
    if fold_map_file:
        text = Path(fold_map_file).read_text(encoding="utf-8")
        source = f"file:{fold_map_file}"
    else:
        subprocess.run(
            ["git", "-C", str(_REPO), "fetch", "origin", "issue-2054", "--quiet"],
            check=False,
            env={**os.environ},
        )
        out = subprocess.run(
            ["git", "-C", str(_REPO), "show", f"{fold_map_ref}:{FOLD_MAP_PATH_IN_REPO}"],
            capture_output=True,
            text=True,
            env={**os.environ},
        )
        if out.returncode != 0:
            raise RuntimeError(
                f"cannot read {fold_map_ref}:{FOLD_MAP_PATH_IN_REPO} "
                f"(rc={out.returncode}): {out.stderr.strip()[:300]}"
            )
        text = out.stdout
        source = f"{fold_map_ref}:{FOLD_MAP_PATH_IN_REPO}"
    d = json.loads(text)
    for key in ("fold_of", "k", "seed"):
        if key not in d:
            raise ValueError(f"fold map missing {key!r} ({source})")
    n_conv = len(d["fold_of"])
    variants = d.get("variants") or []
    if n_conv < FOLD_MAP_MIN_CONV or len(variants) < FOLD_MAP_MIN_VARIANTS:
        raise RuntimeError(
            f"REFUSING fold map at {source}: n_conv={n_conv:,} (floor {FOLD_MAP_MIN_CONV:,}), "
            f"variants={variants} (floor {FOLD_MAP_MIN_VARIANTS}) — this is the smoke map; "
            "the production map lives on the issue-2054 branch blob."
        )
    d["_source"] = source
    d["_sha256"] = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return d


# ─────────────────────────────────────────────────────────────────────────────
# Activation loading + pair join


def load_cell(cell: Cell) -> dict:
    z = np.load(cell.path, allow_pickle=False)
    for key in ("conv_id", "v_C", "v_P", "v_P_present"):
        if key not in z:
            raise ValueError(f"{cell.path} missing key {key!r}")
    conv_ids = [str(x) for x in z["conv_id"]]
    if len(set(conv_ids)) != len(conv_ids):
        raise ValueError(f"duplicate conv_ids in {cell.path} — row pairing would be ambiguous")
    v_c = np.asarray(z["v_C"], dtype=np.float32)
    v_p = np.asarray(z["v_P"], dtype=np.float32)
    v_p_present = np.asarray(z["v_P_present"], dtype=bool)
    if v_c.shape[1] != D_AMBIENT or v_p.shape[1] != D_AMBIENT:
        raise ValueError(f"{cell.path}: expected d={D_AMBIENT}, got {v_c.shape} / {v_p.shape}")
    if not (len(conv_ids) == v_c.shape[0] == v_p.shape[0] == v_p_present.shape[0]):
        raise ValueError(f"{cell.path}: row-count mismatch across keys")
    return {
        "conv_ids": conv_ids,
        "row_of": {cid: i for i, cid in enumerate(conv_ids)},
        "v_C": v_c,
        "v_P": v_p,
        "v_P_present": v_p_present,
    }


def join_pair(src: dict, tgt: dict, fold_of: dict, k: int, arm: str) -> dict:
    """conv_id join restricted to the fold map's population; prefix arm further
    restricted to rows with v_P_present on BOTH sides. Fails LOUD on an empty
    or unexpectedly small join, and on any empty fold."""
    common = set(src["conv_ids"]) & set(tgt["conv_ids"]) & set(fold_of.keys())
    if arm == "prefix":
        common = {
            cid
            for cid in common
            if src["v_P_present"][src["row_of"][cid]] and tgt["v_P_present"][tgt["row_of"][cid]]
        }
    n_src, n_tgt = len(src["conv_ids"]), len(tgt["conv_ids"])
    floor = max(MIN_JOIN_ABS, int(MIN_JOIN_FRAC * min(n_src, n_tgt)))
    if len(common) < floor:
        raise RuntimeError(
            f"conv_id join unexpectedly small (arm={arm}): n_join={len(common)} < floor {floor} "
            f"(n_src={n_src}, n_tgt={n_tgt}, fold_map n={len(fold_of)})"
        )
    order = sorted(common)
    src_rows = np.fromiter((src["row_of"][c] for c in order), dtype=np.int64, count=len(order))
    tgt_rows = np.fromiter((tgt["row_of"][c] for c in order), dtype=np.int64, count=len(order))
    fold_rows: list[list[int]] = [[] for _ in range(k)]
    for i, cid in enumerate(order):
        fold_rows[int(fold_of[cid])].append(i)
    for fi, rows in enumerate(fold_rows):
        if not rows:
            raise RuntimeError(f"fold {fi} empty after join (arm={arm}, n_join={len(order)})")
    return {
        "order": order,
        "src_rows": src_rows,
        "tgt_rows": tgt_rows,
        "fold_rows": [np.asarray(r, dtype=np.int64) for r in fold_rows],
        "n_join": len(order),
        "n_train_min": len(order) - max(len(r) for r in fold_rows),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Shared-eigendecomposition GCV ridge


class SharedEighRidge:
    """GCV ridge with the eigendecomposition computed ONCE per (X_train, X_eval).

    Numerically mirrors ``fit_h.ridge_fit_predict_fast`` /
    ``ridge_fit_predict_fast_layer_batched`` (standardize-X population sd + 1e-9,
    center-Y, eigen dual solve, GCV with the #1887 dof cap; float64). Those
    cores recompute the eigendecomposition per call/slice; here it is shared
    across every target Y reusing the same X, the whole lambda grid, and the
    shuffled-pair null draws (the Gap-A shared-eigh mandate). Side selection:
    n_train > d -> eigh of the d x d covariance (cheaper at ambient n);
    n_train <= d -> eigh of the n x n Gram (the fast-core shape). Nonzero
    spectra of the two sides are identical, so GCV/dof/preds agree; parity vs
    the reused fit_h core is asserted at pilot time (``_parity_gate``).
    """

    def __init__(
        self,
        x_train: np.ndarray,
        x_eval: np.ndarray,
        *,
        lambdas: np.ndarray = DEFAULT_LAMBDAS,
        dof_cap: float | None = GCV_DOF_CAP,
        device: str = "cpu",
    ) -> None:
        dev = torch.device(device)
        xtr = torch.as_tensor(np.asarray(x_train), dtype=torch.float64).to(dev)
        xev = torch.as_tensor(np.asarray(x_eval), dtype=torch.float64).to(dev)
        self.n_train, self.d = int(xtr.shape[0]), int(xtr.shape[1])
        if dof_cap is None and self.n_train < self.d:
            raise RuntimeError(
                f"pure-GCV lambda selection REFUSED at n_train={self.n_train} < d={self.d} "
                "(#1887): pass a dof cap."
            )
        self.dev = dev
        self.lambdas = np.asarray(lambdas, dtype=np.float64)
        self.dof_cap = dof_cap
        xmu = xtr.mean(0)
        xsd = xtr.std(0, unbiased=False) + 1e-9  # population sd (fit_h parity)
        self.xtr_n = (xtr - xmu) / xsd
        self.xev_n = (xev - xmu) / xsd
        self.side = "cov" if self.n_train > self.d else "gram"
        if self.side == "cov":
            c = self.xtr_n.T @ self.xtr_n
            w, v = torch.linalg.eigh(c)
            self.w = torch.clamp(w, min=0.0)
            self.v = v
            self.xev_v = self.xev_n @ v  # (n_ev, d)
        else:
            g = self.xtr_n @ self.xtr_n.T
            w, v = torch.linalg.eigh(g)
            self.w = torch.clamp(w, min=0.0)
            self.v = v
            self.kev_v = (self.xev_n @ self.xtr_n.T) @ v  # (n_ev, n_train)
        lam_t = torch.as_tensor(self.lambdas, dtype=torch.float64, device=dev)
        self.filt_grid = self.w[None, :] / (self.w[None, :] + lam_t[:, None])  # (L, r)
        self.dof_grid = self.filt_grid.sum(1)  # (L,) — dof depends on X only
        self.rss_weights = 2 * self.filt_grid - self.filt_grid**2  # (L, r)

    def _select_lambda(self, rss: torch.Tensor) -> torch.Tensor:
        """GCV argmin over the grid with the dof cap; rss is (L,) or (L, m).
        Returns selected-lambda index tensor of shape () or (m,). RAISES when
        every lambda is dof-cap-excluded (fit_h layer_batched semantics)."""
        denom = (self.n_train - self.dof_grid) ** 2  # (L,)
        gcv = torch.where(
            denom[:, None] > 1e-12 if rss.ndim == 2 else denom > 1e-12,
            rss / (denom[:, None] if rss.ndim == 2 else denom),
            torch.full_like(rss, float("inf")),
        )
        if self.dof_cap is not None:
            ok = self.dof_grid <= self.dof_cap * self.n_train  # (L,)
            if not bool(ok.any()):
                raise RuntimeError(
                    f"gcv dof cap {self.dof_cap}: EVERY lambda in the grid exceeds "
                    f"cap*n_train={self.dof_cap * self.n_train:.0f} (#1887) — widen the grid."
                )
            mask = ok[:, None] if rss.ndim == 2 else ok
            gcv = torch.where(mask, gcv, torch.full_like(gcv, float("inf")))
        return gcv.argmin(dim=0)

    def _y_projection(self, ytr_c: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-eigencomponent projection B and energy e of centered Y_train.
        cov side: B = V' X' Y_c, e_k = ||B_k.||^2 / w_k; gram side: B = V' Y_c,
        e_k = ||B_k.||^2 (equivalent spectra; see class docstring)."""
        if self.side == "cov":
            b = self.v.T @ (self.xtr_n.T @ ytr_c)  # (d, D_out)
            e = (b**2).sum(1)
            e = torch.where(self.w > 1e-12, e / self.w, torch.zeros_like(e))
        else:
            b = self.v.T @ ytr_c  # (n_train, D_out)
            e = (b**2).sum(1)
        return b, e

    def _preds_from_projection(self, b: torch.Tensor, lam: float, ymu: torch.Tensor):
        if self.side == "cov":
            return self.xev_v @ (b / (self.w + lam)[:, None]) + ymu
        return (self.kev_v * (1.0 / (self.w + lam))) @ b + ymu

    def fit_predict(self, y_train: np.ndarray) -> tuple[np.ndarray, dict]:
        ytr = torch.as_tensor(np.asarray(y_train), dtype=torch.float64).to(self.dev)
        ymu = ytr.mean(0)
        ytr_c = ytr - ymu
        tot = float((ytr_c**2).sum())
        b, e = self._y_projection(ytr_c)
        rss = tot - self.rss_weights @ e  # (L,)
        idx = int(self._select_lambda(rss))
        best_lam = float(self.lambdas[idx])
        preds = self._preds_from_projection(b, best_lam, ymu)
        info = {
            "best_lambda": best_lam,
            "dof": float(self.dof_grid[idx]),
            "selector": f"gcv_dof_cap_{self.dof_cap}",
            "side": self.side,
            "n_train": self.n_train,
            "d_fit": self.d,
        }
        return preds.cpu().numpy(), info

    def null_r2(
        self, y_train: np.ndarray, y_eval: np.ndarray, *, n_draws: int, seed: int, chunk: int
    ) -> np.ndarray:
        """Shuffled-pair null: permute Y_train ROWS (breaks the conv_id pairing,
        capacity fixed), refit with the SHARED eigenbasis (per-draw lambda
        re-selection), score vs the UNPERMUTED held-out Y_eval — the
        ``issue2054_fits._shuffled_answer_null_r2`` convention, draws batched
        as chunked GEMMs (vectorize-many-cell-fits.md item 3)."""
        ytr = torch.as_tensor(np.asarray(y_train), dtype=torch.float64).to(self.dev)
        yev = torch.as_tensor(np.asarray(y_eval), dtype=torch.float64).to(self.dev)
        d_out = int(ytr.shape[1])
        ymu = ytr.mean(0)  # row permutation preserves the train mean
        ytr_c = ytr - ymu
        tot = float((ytr_c**2).sum())
        ss_tot_ev = float(((yev - yev.mean(0)) ** 2).sum())
        if ss_tot_ev < 1e-18:
            raise RuntimeError("degenerate Y_eval (zero variance) in null_r2")
        rng = np.random.default_rng(seed)
        lam_t = torch.as_tensor(self.lambdas, dtype=torch.float64, device=self.dev)
        out = np.empty(n_draws, dtype=np.float64)
        done = 0
        while done < n_draws:
            m = min(chunk, n_draws - done)
            perms = [rng.permutation(self.n_train) for _ in range(m)]
            y_stack = torch.cat([ytr_c[torch.as_tensor(p, device=self.dev)] for p in perms], dim=1)
            if self.side == "cov":
                b_all = self.v.T @ (self.xtr_n.T @ y_stack)  # (d, m*D)
            else:
                b_all = self.v.T @ y_stack  # (n_train, m*D)
            r = b_all.shape[0]
            b3 = b_all.view(r, m, d_out)
            e = (b3**2).sum(2)  # (r, m)
            if self.side == "cov":
                e = torch.where(self.w[:, None] > 1e-12, e / self.w[:, None], torch.zeros_like(e))
            rss = tot - self.rss_weights @ e  # (L, m)
            idx = self._select_lambda(rss)  # (m,)
            lam_sel = lam_t[idx]  # (m,)
            scale = 1.0 / (self.w[:, None] + lam_sel[None, :])  # (r, m)
            b_scaled = (b3 * scale[:, :, None]).view(r, m * d_out)
            if self.side == "cov":
                p_all = self.xev_v @ b_scaled  # (n_ev, m*D)
            else:
                p_all = self.kev_v @ b_scaled
            diff = p_all.view(-1, m, d_out) + ymu[None, None, :] - yev[:, None, :]
            ss_res = (diff**2).sum(dim=(0, 2))  # (m,)
            out[done : done + m] = (1.0 - ss_res / ss_tot_ev).cpu().numpy()
            done += m
        return out


def _reduced_basis(x_train: np.ndarray, x_eval: np.ndarray, k: int):
    """Train-fold PCA-k reduced INPUT basis (mirrors issue2054_fits._reduced_basis_r2:
    center-only PCA of X_train; the ridge core standardizes the reduced features).
    Y stays ambient. RAISES when n_train <= k (fail fast, no silent clamp)."""
    n_train = x_train.shape[0]
    if n_train <= k:
        raise RuntimeError(
            f"reduced-basis regime needs n_train > k: n_train={n_train}, k={k} — "
            "cell too small for the pinned reduced basis."
        )
    xtr = np.asarray(x_train, dtype=np.float64)
    xmu = xtr.mean(axis=0)
    xtr_c = xtr - xmu
    _, _, vt = np.linalg.svd(xtr_c, full_matrices=False)
    vk = vt[:k, :]  # (k, d)
    return xtr_c @ vk.T, (np.asarray(x_eval, dtype=np.float64) - xmu) @ vk.T


def _parity_gate(x: np.ndarray, y: np.ndarray, device: str) -> dict:
    """Assert SharedEighRidge reproduces the reused fit_h layer-batched core
    (gcv_dof_cap=0.9, L=1) on BOTH sides (vectorize-many-cell-fits.md item 6).
    Runs on real pilot matrices; rel tol 1e-6 on preds, exact lambda match."""
    results = {}
    for label, n_sub in (("gram_side", 1500), ("cov_side", D_AMBIENT + 416)):
        n_tr = min(n_sub, x.shape[0] - 200)
        x_tr, y_tr, x_ev = x[:n_tr], y[:n_tr], x[n_tr : n_tr + 200]
        core = SharedEighRidge(x_tr, x_ev, dof_cap=GCV_DOF_CAP, device=device)
        preds_mine, info_mine = core.fit_predict(y_tr)
        preds_ref, info_ref = ridge_fit_predict_fast_layer_batched(
            x_tr[None],
            y_tr[None],
            x_ev[None],
            device=device,
            gcv_dof_cap=GCV_DOF_CAP,
            return_info=True,
        )
        preds_ref = preds_ref[0]
        lam_ref = float(info_ref["best_lambda"][0])
        scale = float(np.abs(preds_ref).max()) + 1e-12
        max_rel = float(np.abs(preds_mine - preds_ref).max() / scale)
        if info_mine["best_lambda"] != lam_ref or max_rel > 1e-6:
            raise RuntimeError(
                f"parity FAIL vs fit_h layer-batched core ({label}): "
                f"lambda {info_mine['best_lambda']} vs {lam_ref}, max_rel={max_rel:.3e}"
            )
        results[label] = {
            "n_train": n_tr,
            "side_mine": info_mine["side"],
            "best_lambda": lam_ref,
            "max_rel_pred_diff": max_rel,
        }
        _log(
            f"[ctx2ctx] parity {label}: n_train={n_tr} side={info_mine['side']} "
            f"lam={lam_ref:g} max_rel={max_rel:.2e} OK"
        )
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Unit execution


def _knn_block(preds: np.ndarray, true: np.ndarray) -> dict:
    return {
        metric: knn_retrieval(preds, true, ks=(1, 5, 10), metric=metric)
        for metric in ("euclidean", "cosine")
    }


def _null_seed(src_key: str, tgt_key: str, arm: str, fold: int) -> int:
    h = hashlib.sha256(f"{src_key}|{tgt_key}|{arm}|{fold}|{NULL_SEED_BASE}".encode()).digest()
    return int.from_bytes(h[:4], "little")


def run_unit(
    src: Cell,
    targets: list[dict],
    arm: str,
    fold_map: dict,
    args: argparse.Namespace,
    out_path: Path,
    fingerprint: str,
    unit_tag: str,
) -> None:
    t_unit = time.time()
    k = int(fold_map["k"])
    fold_of = fold_map["fold_of"]
    vec_key = ARM_VEC_KEY[arm]
    src_act = load_cell(src)
    folds_to_run = args.folds if args.folds else list(range(k))

    # Join every target first; group targets by identical (train, test) row sets
    # per fold so the eigendecomposition is shared (the Gap-A vectorization mandate).
    joined: list[dict] = []
    for tp in targets:
        tgt_act = load_cell(tp["tgt"])
        j = join_pair(src_act, tgt_act, fold_of, k, arm)
        if args.max_rows and j["n_join"] > args.max_rows:
            # DEBUG-ONLY deterministic row cap (smoke of the reduced-basis branch).
            rng = np.random.default_rng(NULL_SEED_BASE)
            keep = np.sort(rng.choice(j["n_join"], size=args.max_rows, replace=False))
            keep_set = set(keep.tolist())
            j["src_rows"] = j["src_rows"][keep]
            j["tgt_rows"] = j["tgt_rows"][keep]
            remap = {int(old): new for new, old in enumerate(keep)}
            j["fold_rows"] = [
                np.asarray([remap[int(i)] for i in rows if int(i) in keep_set], dtype=np.int64)
                for rows in j["fold_rows"]
            ]
            j["n_join"] = args.max_rows
            j["n_train_min"] = args.max_rows - max(len(r) for r in j["fold_rows"])
        joined.append({**tp, "join": j, "tgt_act": tgt_act})

    pair_records: dict[str, dict] = {}
    for jp in joined:
        j = jp["join"]
        regime = "ambient" if j["n_train_min"] > D_AMBIENT else "reduced_basis_descriptive"
        pair_records[jp["tgt"].key] = {
            "target_cell": jp["tgt"].key,
            "family": jp["family"],
            "n_src_rows": len(src_act["conv_ids"]),
            "n_tgt_rows": len(jp["tgt_act"]["conv_ids"]),
            "n_join": j["n_join"],
            "n_train_min": j["n_train_min"],
            "d_ambient": D_AMBIENT,
            "regime": regime,
            "reduced_k": REDUCED_BASIS_K if regime != "ambient" else None,
            "folds": [],
        }

    n_units_total = len(joined) * len(folds_to_run)
    unit_i = 0
    for fold_i in folds_to_run:
        groups: dict[tuple, list[dict]] = {}
        for jp in joined:
            j = jp["join"]
            te = j["fold_rows"][fold_i]
            tr = np.concatenate([j["fold_rows"][f] for f in range(k) if f != fold_i])
            key = (
                hashlib.sha256(np.ascontiguousarray(j["src_rows"][tr]).tobytes()).hexdigest(),
                hashlib.sha256(np.ascontiguousarray(j["src_rows"][te]).tobytes()).hexdigest(),
            )
            groups.setdefault(key, []).append({**jp, "tr": tr, "te": te})
        for members in groups.values():
            j0 = members[0]["join"]
            tr, te = members[0]["tr"], members[0]["te"]
            x_tr_raw = src_act[vec_key][j0["src_rows"][tr]].astype(np.float64)
            x_ev_raw = src_act[vec_key][j0["src_rows"][te]].astype(np.float64)
            n_train = x_tr_raw.shape[0]
            regime = "ambient" if n_train > D_AMBIENT else "reduced_basis_descriptive"
            if regime == "ambient":
                x_tr_fit, x_ev_fit = x_tr_raw, x_ev_raw
            else:
                x_tr_fit, x_ev_fit = _reduced_basis(x_tr_raw, x_ev_raw, REDUCED_BASIS_K)
            core = SharedEighRidge(x_tr_fit, x_ev_fit, device=args.device)
            for jp in members:
                t0 = time.time()
                jj = jp["join"]
                y_tr = jp["tgt_act"][vec_key][jj["tgt_rows"][jp["tr"]]].astype(np.float64)
                y_ev = jp["tgt_act"][vec_key][jj["tgt_rows"][jp["te"]]].astype(np.float64)
                # Constant-target degeneracy. A CHAT-form cell's prefix vector is the
                # chat-template header — byte-identical on every row — so v_P has zero
                # variance and R^2 is undefined against it. That is a real property of
                # the render, not corruption: SKIP the fold with a flag rather than
                # crash the whole run (null_r2 raises on it). Same class as the
                # degenerate_identical_xy flag below, on the Y side instead of X.
                y_ss_ev = float(((y_ev - y_ev.mean(0)) ** 2).sum())
                x_ss_ev = float(((x_ev_raw - x_ev_raw.mean(0)) ** 2).sum())
                if y_ss_ev < 1e-18 or x_ss_ev < 1e-18:
                    pair_records[jp["tgt"].key]["folds"].append(
                        {
                            "fold": fold_i,
                            "n_train": int(n_train),
                            "n_test": int(len(jp["te"])),
                            "degenerate_constant_y": bool(y_ss_ev < 1e-18),
                            "degenerate_constant_x": bool(x_ss_ev < 1e-18),
                            "y_ss_eval": y_ss_ev,
                            "x_ss_eval": x_ss_ev,
                            "skipped": "constant-vector fold — R^2 undefined",
                        }
                    )
                    unit_i += 1
                    _log(
                        f"[ctx2ctx] fit {unit_i}/{n_units_total} {src.key} -> "
                        f"{jp['tgt'].key} arm={arm} fold={fold_i} SKIPPED "
                        f"(constant {'Y' if y_ss_ev < 1e-18 else 'X'}; R^2 undefined)"
                    )
                    continue
                preds, info = core.fit_predict(y_tr)
                fitted = reconstruction_metrics(preds, y_ev)
                id_preds = identity_bias_predict(x_tr_raw, y_tr, x_ev_raw)
                id_metrics = reconstruction_metrics(id_preds, y_ev)
                seed = _null_seed(src.key, jp["tgt"].key, arm, fold_i)
                null = core.null_r2(
                    y_tr, y_ev, n_draws=args.null_draws, seed=seed, chunk=args.null_chunk
                )
                p_val = float((1 + (null >= fitted["r2"]).sum()) / (1 + len(null)))
                # X and Y live in the same space (paired renders); byte-identical
                # rows (e.g. framings sharing one prefix render) make the fit
                # trivially perfect — flag it rather than let r2=1.0 read as signal.
                xy_max_diff = float(np.abs(x_ev_raw - y_ev).max())
                rec = {
                    "fold": fold_i,
                    "n_train": int(n_train),
                    "n_test": int(len(jp["te"])),
                    "xy_max_abs_diff": xy_max_diff,
                    "degenerate_identical_xy": bool(xy_max_diff == 0.0),
                    "fitted": {**fitted, **info},
                    "identity_bias": id_metrics,
                    "knn_fitted": _knn_block(preds, y_ev),
                    "knn_identity": _knn_block(id_preds, y_ev),
                    "null": {
                        "kind": "shuffled_pair_fit_side",
                        "n_draws": int(len(null)),
                        "seed": seed,
                        "mean": float(null.mean()),
                        "sd": float(null.std(ddof=1)) if len(null) > 1 else float("nan"),
                        "q05": float(np.quantile(null, 0.05)),
                        "q95": float(np.quantile(null, 0.95)),
                        "max": float(null.max()),
                        "p_value_fitted": p_val,
                    },
                    "wall_s": round(time.time() - t0, 1),
                }
                pair_records[jp["tgt"].key]["folds"].append(rec)
                unit_i += 1
                _log(
                    f"[ctx2ctx] fit {unit_i}/{n_units_total} {src.key} -> {jp['tgt'].key} "
                    f"arm={arm} fold={fold_i} r2={fitted['r2']:+.4f} "
                    f"id_r2={id_metrics['r2']:+.4f} null_mu={null.mean():+.4f} "
                    f"lam={info['best_lambda']:g} elapsed={rec['wall_s']}s"
                )

    for rec in pair_records.values():
        all_folds = rec["folds"]
        # Constant-vector folds carry no fitted/null block (see the skip above);
        # exclude them from every numeric pooling and report the count.
        folds = [f for f in all_folds if not f.get("skipped")]
        rec["n_folds_skipped_degenerate"] = len(all_folds) - len(folds)
        if not folds:
            rec["pooled"] = None
            rec["verdict"] = (
                "DEGENERATE — every fold has a constant source or target vector "
                "(chat-form prefix renders are identical across rows); R^2 is "
                "undefined here, this is not a mapping result"
            )
            continue
        fitted_r2 = float(np.mean([f["fitted"]["r2"] for f in folds]))
        id_r2 = float(np.mean([f["identity_bias"]["r2"] for f in folds]))
        rec["pooled"] = {
            "fitted_r2": fitted_r2,
            "identity_bias_r2": id_r2,
            "delta_r2_fitted_minus_identity": fitted_r2 - id_r2,
            "clears_identity_baseline": bool(fitted_r2 > id_r2),
            "null_mean": float(np.mean([f["null"]["mean"] for f in folds])),
            "null_p_value_fitted": float(np.mean([f["null"]["p_value_fitted"] for f in folds])),
        }
        all_degenerate = all(f["degenerate_identical_xy"] for f in folds)
        rec["pooled"]["degenerate_identical_xy"] = bool(all_degenerate)
        if all_degenerate:
            rec["verdict"] = (
                "DEGENERATE — source and target vectors are byte-identical (shared "
                "render); fit is trivially perfect, not a mapping result"
            )
        else:
            rec["verdict"] = f"fitted R2={fitted_r2:+.4f} vs identity+bias R2={id_r2:+.4f} -> " + (
                "CLEARS identity+bias baseline"
                if fitted_r2 > id_r2
                else "NULL — does not clear the identity+bias baseline"
            )

    payload = {
        "metadata": {
            **as_metadata_dict(git_provenance(_REPO)),
            "script_version": SCRIPT_VERSION,
            "argv": sys.argv,
            "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "fold_map": {
                "source": fold_map["_source"],
                "sha256": fold_map["_sha256"],
                "k": int(fold_map["k"]),
                "seed": int(fold_map["seed"]),
                "n_conv": len(fold_map["fold_of"]),
            },
            "debug_max_rows": args.max_rows or None,
            "pilot": bool(args.pilot),
        },
        "source_cell": src.key,
        "arm": arm,
        "config_fingerprint": fingerprint,
        "pairs": sorted(pair_records.values(), key=lambda r: r["target_cell"]),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_name(out_path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    os.replace(tmp, out_path)
    _log(
        f"[ctx2ctx] unit {unit_tag} CHECKPOINTED -> {out_path} "
        f"({len(pair_records)} pairs, wall={time.time() - t_unit:.0f}s)"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main


def _fingerprint(args: argparse.Namespace, fold_map: dict, arm: str, tgt_keys: list[str]) -> str:
    blob = json.dumps(
        {
            "script_version": SCRIPT_VERSION,
            "lambdas": [float(x) for x in DEFAULT_LAMBDAS],
            "dof_cap": GCV_DOF_CAP,
            "reduced_k": REDUCED_BASIS_K,
            "fold_map_sha": fold_map["_sha256"],
            "folds": args.folds or list(range(int(fold_map["k"]))),
            "null_draws": args.null_draws,
            "arm": arm,
            "targets": sorted(tgt_keys),
            "max_rows": args.max_rows,
            "join_floors": [MIN_JOIN_ABS, MIN_JOIN_FRAC],
        },
        sort_keys=True,
    )
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.replace("%", "%%"))
    ap.add_argument("--activations-dir", type=Path, required=True)
    ap.add_argument(
        "--out-root", type=Path, default=_REPO / "eval_results/issue_2054/inline_ctx2ctx"
    )
    ap.add_argument("--fold-map-ref", default="origin/issue-2054")
    ap.add_argument("--fold-map-file", default=None, help="direct path override (floors enforced)")
    ap.add_argument("--pilot", action="store_true", help="ONE pair, fold 0, writes under pilot/")
    ap.add_argument("--arms", nargs="*", default=list(ARMS), choices=list(ARMS))
    ap.add_argument("--folds", nargs="*", type=int, default=None)
    ap.add_argument("--null-draws", type=int, default=100)
    ap.add_argument("--null-chunk", type=int, default=5)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--pair-filter", default=None, help="substring filter on src/tgt cell keys")
    ap.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="DEBUG ONLY: deterministic row cap after join (smokes the reduced-basis branch)",
    )
    ap.add_argument("--skip-parity", action="store_true", help="skip the pilot parity gate")
    ap.add_argument("--import-check", action="store_true")
    return ap


def main() -> int:
    args = build_argparser().parse_args()
    if args.import_check:
        from explore_persona_space.orchestrate.argcheck import assert_args_attributes_defined

        assert_args_attributes_defined(__file__)
        _log("[ctx2ctx] import-check OK")
        return 0

    t_start = time.time()
    fold_map = load_fold_map(args.fold_map_file, args.fold_map_ref)
    _log(
        f"[ctx2ctx] fold map {fold_map['_source']} k={fold_map['k']} seed={fold_map['seed']} "
        f"n_conv={len(fold_map['fold_of']):,} sha={fold_map['_sha256'][:12]}"
    )
    cells = discover_cells(args.activations_dir)
    pairs = enumerate_pairs(cells)
    if args.pair_filter:
        pairs = [
            p for p in pairs if args.pair_filter in p["src"].key or args.pair_filter in p["tgt"].key
        ]
    if not pairs:
        raise RuntimeError("no pairs enumerated — check --activations-dir / --pair-filter")
    fam_counts = {
        fam: sum(1 for p in pairs if p["family"] == fam)
        for fam in ("framing_pairs", "identity_pairs")
    }
    _log(f"[ctx2ctx] {len(cells)} cells -> {len(pairs)} ordered pairs {fam_counts}")

    out_root = args.out_root / "pilot" if args.pilot else args.out_root
    if args.pilot:
        pairs = pairs[:1]
        args.folds = [0]
        _log(f"[ctx2ctx] PILOT: {pairs[0]['src'].key} -> {pairs[0]['tgt'].key}, fold 0 only")
        if not args.skip_parity:
            src_act = load_cell(pairs[0]["src"])
            tgt_act = load_cell(pairs[0]["tgt"])
            j = join_pair(src_act, tgt_act, fold_map["fold_of"], int(fold_map["k"]), "context")
            _parity_gate(
                src_act["v_C"][j["src_rows"]].astype(np.float64),
                tgt_act["v_C"][j["tgt_rows"]].astype(np.float64),
                args.device,
            )

    # Units: (source cell, arm) with the targets that share that source.
    units: dict[tuple[str, str], dict] = {}
    for p in pairs:
        for arm in args.arms:
            u = units.setdefault((p["src"].key, arm), {"src": p["src"], "arm": arm, "targets": []})
            u["targets"].append({"tgt": p["tgt"], "family": p["family"]})
    unit_list = [units[k] for k in sorted(units)]
    unit_list = [u for i, u in enumerate(unit_list) if i % args.num_shards == args.shard]
    _log(
        f"[ctx2ctx] {len(unit_list)} (source, arm) units this shard "
        f"(shard {args.shard}/{args.num_shards})"
    )

    n_done = 0
    for u in unit_list:
        tgt_keys = [t["tgt"].key for t in u["targets"]]
        fp = _fingerprint(args, fold_map, u["arm"], tgt_keys)
        out_path = out_root / "percell" / f"{u['src'].key}__{u['arm']}.json"
        if out_path.is_file():
            prior = json.loads(out_path.read_text(encoding="utf-8"))
            if prior.get("config_fingerprint") == fp:
                n_done += 1
                _log(f"[ctx2ctx] unit {u['src'].key}__{u['arm']} already done — resume skip")
                continue
            raise RuntimeError(
                f"existing {out_path} has fingerprint {prior.get('config_fingerprint')} != {fp} "
                "— config changed; move the stale file or use a fresh --out-root."
            )
        run_unit(
            u["src"],
            u["targets"],
            u["arm"],
            fold_map,
            args,
            out_path,
            fp,
            unit_tag=f"{u['src'].key}__{u['arm']}",
        )
        n_done += 1

    peak_rss_gib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
    _log(
        f"[ctx2ctx] done units={n_done}/{len(unit_list)} wall={time.time() - t_start:.0f}s "
        f"peak_rss_gib={peak_rss_gib:.2f} torch_threads={torch.get_num_threads()}"
    )
    return 0


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(rc)  # explicit exit before C-extension teardown (code-style.md)
