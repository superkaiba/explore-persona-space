#!/usr/bin/env python
"""Shuffled-PAIR nulls for the #2054 writeup's re-map transfer tiers (rungs 7/8).

Question (user, 2026-08-14): rung 7 (context re-map) fits A on PAIRED
conversations (target context -> source context) and rung 8 (answer re-map)
fits B on paired answers. How much of their transfer recovery survives when
the source<->target conversation pairing is destroyed? A matched-capacity
permutation null separates "the re-map exploits per-conversation
correspondence" from "the re-map merely aligns the two clouds' overall
geometry" (means/covariances survive a row permutation).

Construction (per writeup pair, context arm, per fold — the
``issue2054_fits._shuffled_answer_null_r2`` / ``issue2054_ctx2ctx_fit``
fit-side convention):
  rung 7 null — permute the ROWS of Xs_tr (the regression targets of A),
    refit A = ridge(Xt_tr -> perm(Xs_tr)) at the same n and GCV/dof-cap
    procedure, compose with the UNCHANGED source map M (its internal
    source-cell pairing is not touched) + the rung-7 bias refit, score on the
    unpermuted held-out fold.
  rung 8 null — permute the ROWS of Yt_tr (the regression targets of B),
    refit B = ridge(Ys_tr -> perm(Yt_tr)), apply to M's held-out predictions,
    score on the unpermuted held-out fold.

Matched capacity: same train rows, same standardization, same lambda grid,
same GCV + dof-cap selection per draw. Row permutation preserves train means,
so xmu/xsd/ymu are unchanged; the SVD of the fit-side X is computed ONCE per
(cell, fold, train-rows) and reused across draws (only UtY changes — the
vectorize-many-cell-fits draw-battery mandate), with (source-map M, B-side
SVD) memoized across pairs sharing a source and (A-side SVD) across pairs
sharing a target.

Two built-in gates per pair:
  parity — an identity-permutation draw must reproduce the plain `_fit_ridge`
    prediction (max |dR2| < 1e-10);
  banked-value sanity — the recomputed TRUE rung-7/8 fold-mean R2 must match
    the banked ladder rows (|d| < 1e-6), proving the join/equalization and
    estimator are byte-compatible with the committed ladder.

Pairs: exactly the 44 shown in docs/reports/framing_character_transfer_writeup.md
(R1 chat->framings 12; R2 story bare-label assistant->characters 16; R3
inserted-chat->characters 16). Context arm only. Checkpoint per pair
(units.jsonl, resume on key), one progress line per unit-fold.

``--mode cross`` (2026-08-14 tier re-spec): the same shuffled-pair null for the
CROSS-TRANSFER tier instead — the paired cross-render fit (source-render
context -> target-render answer, ridge fit directly on paired rows, the
banked ``analyzer_companions/cross_render_fit*.json`` estimator). Null: permute
the ROWS of Yt_tr (the paired targets), refit at matched capacity, score on
the unpermuted held-out fold. Banked sanity compares per-fold against
``cross_render_per_fold``; the parity gate runs on fold 0 per pair.

Usage:
  uv run python scripts/issue2054_remap_pair_nulls.py \
      --activations-dir <dir with the 56 cell npz> \
      [--pilot] [--n-draws 5] [--out-root eval_results/issue_2054/specialization_ladder]
  # --stage-from-hf: download just the needed cells from the HF data repo
  #   into --activations-dir first (pod-side path).
"""

from __future__ import annotations

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    # script mode puts scripts/ (not the repo root) on sys.path[0] (gotchas.md).
    sys.path.insert(0, str(_REPO))

from explore_persona_space.orchestrate.provenance import as_metadata_dict, git_provenance
from scripts.issue2054_ladder import (
    DEFAULT_DOF_CAP,
    DEFAULT_LAMBDAS,
    _apply_ridge,
    _fit_ridge,
    _load_activation_npz,
    _r2_matrix,
    _row_index_by_conv_id,
    _select_arm,
)

SCRIPT_VERSION = "issue2054_remap_pair_nulls_v1"
ASSIST = "conversation_paired_stories_assistant"
CHARS = ("wren", "helios", "dana", "vex")
MODELS = ("qwen2.5-7b-instruct", "qwen2.5-7b")
CONDS = ("on_policy", "inserted")
HF_REPO = "superkaiba1/explore-persona-space-data"
HF_ACT_PREFIX = "issue2054_lattice/activations"


def _log(msg: str) -> None:
    print(msg, flush=True)


def writeup_pairs() -> list[tuple[str, str]]:
    """The 44 (source_cell, target_cell) pairs shown in the writeup figures."""
    pairs: list[tuple[str, str]] = []
    for cond in CONDS:  # R1: chat -> other assistant framings
        for model in MODELS:
            for form in ("bare_text", "bare_label", "attrib_quoted"):
                pairs.append(
                    (f"{ASSIST}__{cond}__chat__{model}", f"{ASSIST}__{cond}__{form}__{model}")
                )
    for cond in CONDS:  # R2: assistant story bare-label -> characters
        for model in MODELS:
            for ch in CHARS:
                pairs.append(
                    (
                        f"{ASSIST}__{cond}__bare_label__{model}",
                        f"char_{ch}__{cond}__bare_label__{model}",
                    )
                )
    for tcond in CONDS:  # R3: inserted chat -> characters in story (bare label)
        for model in MODELS:
            for ch in CHARS:
                pairs.append(
                    (
                        f"{ASSIST}__inserted__chat__{model}",
                        f"char_{ch}__{tcond}__bare_label__{model}",
                    )
                )
    assert len(pairs) == 44, len(pairs)
    return pairs


def stage_from_hf(activations_dir: Path, cell_keys: set[str]) -> None:
    """Download the needed cell npz files from the HF data repo (pod path)."""
    from huggingface_hub import hf_hub_download

    from explore_persona_space.orchestrate.hub import retry_transient

    activations_dir.mkdir(parents=True, exist_ok=True)
    for key in sorted(cell_keys):
        dest = activations_dir / f"{key}.npz"
        if dest.exists():
            _log(f"[nulls] stage: {key}.npz already present")
            continue
        t0 = time.time()
        got = retry_transient(
            lambda key=key: hf_hub_download(
                repo_id=HF_REPO,
                repo_type="dataset",
                filename=f"{HF_ACT_PREFIX}/{key}.npz",
                local_dir=activations_dir / "_hf",
            ),
            what=f"stage {key}.npz",
        )
        os.replace(got, dest)
        _log(f"[nulls] staged {key}.npz elapsed={time.time() - t0:.1f}s")


def load_fold_map(fold_map_file: Path) -> dict:
    payload = json.loads(fold_map_file.read_text())
    return payload


class _SvdRidge:
    """One SVD of the fit-side X, reusable across permutation draws of Y.

    Mirrors `issue2054_ladder._fit_ridge` exactly (same standardization,
    lambda grid, GCV + dof-cap selection); the parity gate asserts equality.
    """

    def __init__(self, X_train: np.ndarray, lambdas=DEFAULT_LAMBDAS, dof_cap=DEFAULT_DOF_CAP):
        Xtr64 = X_train.astype(np.float64)
        self.xmu = Xtr64.mean(axis=0)
        self.xsd = Xtr64.std(axis=0) + 1e-9
        Xtr = (Xtr64 - self.xmu) / self.xsd
        self.n_train = Xtr.shape[0]
        self.U, self.s, self.Vt = np.linalg.svd(Xtr, full_matrices=False)
        self.s2 = self.s**2
        self.lambdas = np.asarray(lambdas, dtype=np.float64)
        self.dof_cap = dof_cap

    def fit_apply(self, Y_train: np.ndarray, X_apply_std_v: np.ndarray) -> np.ndarray:
        """Fit on (banked X, this Y) and predict at pre-projected inputs.

        `X_apply_std_v` is ((X_apply - xmu)/xsd) @ Vt.T, computed once by the
        caller and reused across draws.
        """
        Ytr64 = Y_train.astype(np.float64)
        ymu = Ytr64.mean(axis=0)
        UtY = self.U.T @ (Ytr64 - ymu)

        best_lam = float(self.lambdas[0])
        best_gcv = float("inf")
        row_energy = (UtY**2).sum(axis=1)
        tot_y_sq = float(((Ytr64 - ymu) ** 2).sum())
        for lam in self.lambdas:
            lam = float(lam)
            filt = self.s2 / (self.s2 + lam)
            dof = float(filt.sum())
            rss = tot_y_sq - float(((2 * filt - filt**2) * row_energy).sum())
            denom = (self.n_train - dof) ** 2
            gcv = rss / denom if denom > 1e-12 else float("inf")
            if dof / self.n_train <= self.dof_cap and gcv < best_gcv:
                best_gcv = gcv
                best_lam = lam
        if best_gcv == float("inf"):
            best_lam = float(self.lambdas[-1])
        filt = self.s / (self.s2 + best_lam)
        # pred = X_std @ W + ymu, W = (Vt.T * filt) @ UtY  ==>  (X_std Vt.T * filt) @ UtY
        return (X_apply_std_v * filt) @ UtY + ymu

    def project(self, X_apply: np.ndarray) -> np.ndarray:
        Xa = (X_apply.astype(np.float64) - self.xmu) / self.xsd
        return Xa @ self.Vt.T


def null_for_pair(
    src_key: str,
    tgt_key: str,
    src_acts: dict,
    tgt_acts: dict,
    fold_of: dict,
    k: int,
    *,
    n_draws: int,
    seed: int,
    banked_rungs: dict | None,
    svd_cache: dict,
    m_cache: dict,
    pilot: bool = False,
) -> dict:
    """Compute true rung-7/8 values + shuffled-pair null draws for one pair."""
    Xs_all, Ys_all, s_ids = _select_arm(src_acts, "context")
    Xt_all, Yt_all, t_ids = _select_arm(tgt_acts, "context")
    inter = set(s_ids) & set(t_ids) & set(fold_of.keys())
    ordered = sorted(inter)
    s_row = _row_index_by_conv_id(s_ids)
    t_row = _row_index_by_conv_id(t_ids)

    per_fold: list[dict] = []
    fold_range = range(min(1, k)) if pilot else range(k)
    for fold_i in fold_range:
        t0 = time.time()
        train_ids = [c for c in ordered if int(fold_of[c]) != fold_i]
        val_ids = [c for c in ordered if int(fold_of[c]) == fold_i]
        tr_s = np.array([s_row[c] for c in train_ids], dtype=np.int64)
        tr_t = np.array([t_row[c] for c in train_ids], dtype=np.int64)
        te_t = np.array([t_row[c] for c in val_ids], dtype=np.int64)
        Xs_tr, Ys_tr = Xs_all[tr_s], Ys_all[tr_s]
        Xt_tr, Yt_tr = Xt_all[tr_t], Yt_all[tr_t]
        Xt_te, Yt_te = Xt_all[te_t], Yt_all[te_t]
        n_tr = int(tr_s.size)

        train_key = hash(tuple(train_ids))
        # M: source context -> source answer (memoized per source x fold x rows).
        mk = (src_key, fold_i, train_key)
        if mk not in m_cache:
            m_cache.clear() if len(m_cache) >= 6 else None
            m_cache[mk] = _fit_ridge(Xs_tr, Ys_tr)
        model_M = m_cache[mk]

        # A-side SVD keyed on the TARGET cell's train rows; B-side on the
        # SOURCE cell's train answers.
        ak = ("A", tgt_key, fold_i, train_key)
        if ak not in svd_cache:
            svd_cache.clear() if len(svd_cache) >= 6 else None
            svd_cache[ak] = _SvdRidge(Xt_tr)
        svd_A = svd_cache[ak]
        bk = ("B", src_key, fold_i, train_key)
        if bk not in svd_cache:
            svd_cache[bk] = _SvdRidge(Ys_tr)
        svd_B = svd_cache[bk]

        # Pre-projected apply inputs (computed once, reused across draws).
        Zt_tr = svd_A.project(Xt_tr)
        Zt_te = svd_A.project(Xt_te)
        P_te = _apply_ridge(model_M, Xt_te)  # rung-8 input
        Zp_te = svd_B.project(P_te)

        # ---- parity gate: identity permutation reproduces _fit_ridge. ----
        true_A_te = svd_A.fit_apply(Xs_tr, Zt_te)
        ref_A = _fit_ridge(Xt_tr, Xs_tr)
        d_par = float(np.max(np.abs(true_A_te - _apply_ridge(ref_A, Xt_te))))
        assert d_par < 1e-6, f"parity gate failed ({src_key}->{tgt_key} f{fold_i}): {d_par}"

        # ---- true rung values (same math as the ladder). ----
        true_A_tr = svd_A.fit_apply(Xs_tr, Zt_tr)
        P7_tr = _apply_ridge(model_M, true_A_tr)
        P7_te = _apply_ridge(model_M, true_A_te)
        b7 = (Yt_tr.astype(np.float64) - P7_tr).mean(axis=0)
        true7 = _r2_matrix(Yt_te, P7_te + b7)
        true8 = _r2_matrix(Yt_te, svd_B.fit_apply(Yt_tr, Zp_te))

        # ---- shuffled-pair null draws. ----
        rng = np.random.default_rng(seed + 1_000 * fold_i)
        null7: list[float] = []
        null8: list[float] = []
        for _ in range(n_draws):
            perm = rng.permutation(n_tr)
            a_te = svd_A.fit_apply(Xs_tr[perm], Zt_te)
            a_tr = svd_A.fit_apply(Xs_tr[perm], Zt_tr)
            p7_tr = _apply_ridge(model_M, a_tr)
            p7_te = _apply_ridge(model_M, a_te)
            b7n = (Yt_tr.astype(np.float64) - p7_tr).mean(axis=0)
            null7.append(_r2_matrix(Yt_te, p7_te + b7n))
            null8.append(_r2_matrix(Yt_te, svd_B.fit_apply(Yt_tr[perm], Zp_te)))

        per_fold.append(
            {
                "fold": fold_i,
                "n_train": n_tr,
                "n_val": int(te_t.size),
                "true_7_ctx_reparam": true7,
                "true_8_ans_reparam": true8,
                "null_7": null7,
                "null_8": null8,
                "parity_max_abs_diff": d_par,
            }
        )
        _log(
            f"[nulls] {src_key} -> {tgt_key} fold {fold_i}: true7={true7:+.4f} "
            f"true8={true8:+.4f} null7_max={max(null7):+.4f} null8_max={max(null8):+.4f} "
            f"n_tr={n_tr} elapsed={time.time() - t0:.1f}s"
        )

    all7 = [v for f in per_fold for v in f["null_7"]]
    all8 = [v for f in per_fold for v in f["null_8"]]
    rec: dict = {
        "src": src_key,
        "tgt": tgt_key,
        "arm": "context",
        "n_intersection": len(ordered),
        "n_draws_per_fold": n_draws,
        "per_fold": per_fold,
        "mean": {
            "true_7": float(np.mean([f["true_7_ctx_reparam"] for f in per_fold])),
            "true_8": float(np.mean([f["true_8_ans_reparam"] for f in per_fold])),
            "null_7_p95": float(np.percentile(all7, 95)),
            "null_8_p95": float(np.percentile(all8, 95)),
            "null_7_median": float(np.median(all7)),
            "null_8_median": float(np.median(all8)),
        },
    }
    # ---- banked-value sanity (full-fold runs only). ----
    if banked_rungs is not None and not pilot:
        d7 = abs(rec["mean"]["true_7"] - banked_rungs["7_ctx_reparam"])
        d8 = abs(rec["mean"]["true_8"] - banked_rungs["8_ans_reparam"])
        rec["banked_check"] = {"d7": d7, "d8": d8, "pass": bool(d7 < 1e-6 and d8 < 1e-6)}
        assert rec["banked_check"]["pass"], (
            f"banked-value sanity failed ({src_key}->{tgt_key}): d7={d7} d8={d8}"
        )
    return rec


COMPANIONS_DIR = _REPO / "eval_results/issue_2054/analyzer_companions"
_MODEL_SLUG = {"qwen2.5-7b-instruct": "qwen25-7b-instruct", "qwen2.5-7b": "qwen25-7b"}


def load_banked_cross() -> dict[tuple[str, str], list[float]]:
    """(src_cell, tgt_cell) -> banked per-fold cross-render R2 (fold 0..k-1)."""
    out: dict[tuple[str, str], list[float]] = {}
    main = json.loads((COMPANIONS_DIR / "cross_render_fit.json").read_text())
    for c in main["cells"]:
        if c.get("is_identity"):
            continue
        src = f"{ASSIST}__{c['condition']}__chat__{c['model']}"
        tgt = f"{ASSIST}__{c['condition']}__{c['target_form']}__{c['model']}"
        out[(src, tgt)] = c["cross_render_per_fold"]
    for shard in ("2a", "2b"):
        for slug in _MODEL_SLUG.values():
            path = COMPANIONS_DIR / f"cross_render_fit_characters.shard__{shard}__{slug}.json"
            for c in json.loads(path.read_text())["cells"]:
                src = f"{ASSIST}__{c['source_condition']}__{c['source_form']}__{c['model']}"
                tgt = f"{c['character']}__{c['target_condition']}__{c['target_form']}__{c['model']}"
                out[(src, tgt)] = c["cross_render_per_fold"]
    return out


def null_for_pair_cross(
    src_key: str,
    tgt_key: str,
    src_acts: dict,
    tgt_acts: dict,
    fold_of: dict,
    k: int,
    *,
    n_draws: int,
    seed: int,
    banked_per_fold: list[float] | None,
    svd_cache: dict,
    pilot: bool = False,
) -> dict:
    """True cross-render R2 + shuffled-pair null draws for one writeup pair.

    Cross-render fit: ridge source-render CONTEXT -> target-render ANSWER on
    the paired train rows; null permutes the Yt_tr rows (pairing destroyed,
    marginals intact) and refits at matched capacity.
    """
    Xs_all, _ys, s_ids = _select_arm(src_acts, "context")
    _xt, Yt_all, t_ids = _select_arm(tgt_acts, "context")
    inter = set(s_ids) & set(t_ids) & set(fold_of.keys())
    ordered = sorted(inter)
    s_row = _row_index_by_conv_id(s_ids)
    t_row = _row_index_by_conv_id(t_ids)

    per_fold: list[dict] = []
    fold_range = range(min(1, k)) if pilot else range(k)
    for fold_i in fold_range:
        t0 = time.time()
        train_ids = [c for c in ordered if int(fold_of[c]) != fold_i]
        val_ids = [c for c in ordered if int(fold_of[c]) == fold_i]
        tr_s = np.array([s_row[c] for c in train_ids], dtype=np.int64)
        tr_t = np.array([t_row[c] for c in train_ids], dtype=np.int64)
        te_s = np.array([s_row[c] for c in val_ids], dtype=np.int64)
        te_t = np.array([t_row[c] for c in val_ids], dtype=np.int64)
        Xs_tr, Yt_tr = Xs_all[tr_s], Yt_all[tr_t]
        Xs_te, Yt_te = Xs_all[te_s], Yt_all[te_t]
        n_tr = int(tr_s.size)

        train_key = hash(tuple(train_ids))
        ck = ("C", src_key, fold_i, train_key)
        if ck not in svd_cache:
            svd_cache.clear() if len(svd_cache) >= 6 else None
            svd_cache[ck] = _SvdRidge(Xs_tr)
        svd_C = svd_cache[ck]
        Zs_te = svd_C.project(Xs_te)

        true_te = svd_C.fit_apply(Yt_tr, Zs_te)
        if fold_i == 0:
            # parity gate (fold 0 only — the reference _fit_ridge is a second
            # full SVD, the dominant per-fold cost; banked sanity covers the
            # remaining folds against the committed companion values).
            ref = _fit_ridge(Xs_tr, Yt_tr)
            d_par = float(np.max(np.abs(true_te - _apply_ridge(ref, Xs_te))))
            assert d_par < 1e-6, f"parity gate failed ({src_key}->{tgt_key}): {d_par}"
        true_cross = _r2_matrix(Yt_te, true_te)
        if banked_per_fold is not None and not pilot:
            db = abs(true_cross - float(banked_per_fold[fold_i]))
            assert db < 1e-6, (
                f"banked-value sanity failed ({src_key}->{tgt_key} f{fold_i}): "
                f"recomputed={true_cross} banked={banked_per_fold[fold_i]} d={db}"
            )

        rng = np.random.default_rng(seed + 1_000 * fold_i)
        null_c: list[float] = []
        for _ in range(n_draws):
            perm = rng.permutation(n_tr)
            null_c.append(_r2_matrix(Yt_te, svd_C.fit_apply(Yt_tr[perm], Zs_te)))

        per_fold.append(
            {
                "fold": fold_i,
                "n_train": n_tr,
                "n_val": int(te_t.size),
                "true_cross": true_cross,
                "null_cross": null_c,
            }
        )
        _log(
            f"[nulls] {src_key} -> {tgt_key} fold {fold_i}: true_cross={true_cross:+.4f} "
            f"null_max={max(null_c):+.4f} n_tr={n_tr} elapsed={time.time() - t0:.1f}s"
        )

    all_c = [v for f in per_fold for v in f["null_cross"]]
    return {
        "src": src_key,
        "tgt": tgt_key,
        "arm": "context",
        "mode": "cross",
        "n_intersection": len(ordered),
        "n_draws_per_fold": n_draws,
        "per_fold": per_fold,
        "mean": {
            "true_cross": float(np.mean([f["true_cross"] for f in per_fold])),
            "null_cross_p95": float(np.percentile(all_c, 95)),
            "null_cross_median": float(np.median(all_c)),
        },
        "banked_check": {"pass": banked_per_fold is not None and not pilot},
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.replace("%", "%%"))
    ap.add_argument("--activations-dir", type=Path, required=True)
    ap.add_argument("--fold-map-file", type=Path, required=True)
    ap.add_argument(
        "--rows-file", type=Path, default=None, help="merged ladder rows JSON (banked sanity)"
    )
    ap.add_argument(
        "--out-root", type=Path, default=_REPO / "eval_results/issue_2054/specialization_ladder"
    )
    ap.add_argument("--out-name", default="remap_pair_nulls.json")
    ap.add_argument(
        "--mode",
        choices=("remap", "cross"),
        default="remap",
        help="remap: rungs 7/8 nulls; cross: paired cross-render-fit tier null",
    )
    ap.add_argument("--n-draws", type=int, default=5)
    ap.add_argument("--seed", type=int, default=137)
    ap.add_argument("--pilot", action="store_true", help="first pair, fold 0 only")
    ap.add_argument("--stage-from-hf", action="store_true")
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--shard-count", type=int, default=1)
    ap.add_argument("--import-check", action="store_true")
    ap.add_argument(
        "--merge-shards",
        metavar="GLOB",
        default=None,
        help="merge shard payloads matching GLOB (relative to --out-root) into "
        "--out-name and exit; asserts the expected pair count and gate passes",
    )
    ap.add_argument("--expect-pairs", type=int, default=44)
    args = ap.parse_args()
    if args.import_check:
        from explore_persona_space.orchestrate.argcheck import assert_args_attributes_defined

        assert_args_attributes_defined(__file__)
        _log("[nulls] import-check OK")
        return 0

    if args.merge_shards:
        shard_paths = sorted(args.out_root.glob(args.merge_shards))
        assert shard_paths, f"no shard payloads matched {args.out_root / args.merge_shards}"
        units: list[dict] = []
        seen: set[tuple[str, str]] = set()
        for sp in shard_paths:
            payload = json.loads(sp.read_text())
            assert payload["metadata"].get("mode", "remap") == args.mode, (
                f"{sp.name}: mode={payload['metadata'].get('mode')} != --mode {args.mode}"
            )
            for u in payload["units"]:
                key = (u["src"], u["tgt"])
                assert key not in seen, f"duplicate pair across shards: {key}"
                assert u.get("banked_check", {}).get("pass"), f"unit failed banked gate: {key}"
                seen.add(key)
                units.append(u)
            _log(f"[merge] {sp.name}: {len(payload['units'])} units")
        assert len(units) == args.expect_pairs, f"{len(units)} units, expected {args.expect_pairs}"
        out_path = args.out_root / args.out_name
        merged = {
            "metadata": {
                **as_metadata_dict(git_provenance(_REPO)),
                "script_version": SCRIPT_VERSION,
                "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "mode": args.mode,
                "n_pairs": len(units),
                "n_draws_per_fold": args.n_draws,
                "seed": args.seed,
                "merged_from_shards": [p.name for p in shard_paths],
            },
            "units": units,
        }
        tmp = out_path.with_name(out_path.name + ".tmp")
        tmp.write_text(json.dumps(merged, indent=1), encoding="utf-8")
        os.replace(tmp, out_path)
        _log(f"[merge] wrote {out_path} ({len(units)} pairs, all banked gates PASS)")
        return 0

    t_start = time.time()
    pairs = writeup_pairs()
    pairs = [p for i, p in enumerate(pairs) if i % args.shard_count == args.shard_index]
    if args.pilot:
        pairs = pairs[:1]
    cells = {c for p in pairs for c in p}
    if args.stage_from_hf:
        stage_from_hf(args.activations_dir, cells)

    fm = load_fold_map(args.fold_map_file)
    k, fold_of = int(fm["k"]), fm["fold_of"]
    _log(f"[nulls] {len(pairs)} pairs, {len(cells)} cells, k={k}, draws={args.n_draws}")

    banked: dict[tuple[str, str], dict] = {}
    banked_cross: dict[tuple[str, str], list[float]] = {}
    if args.mode == "cross":
        banked_cross = load_banked_cross()
        missing = [p for p in pairs if p not in banked_cross]
        _log(
            f"[nulls] banked cross-render coverage: {len(pairs) - len(missing)}/{len(pairs)} pairs"
        )
        for src, tgt in missing:
            _log(f"[nulls] WARN no banked cross-render value: {src} -> {tgt}")
    elif args.rows_file and args.rows_file.exists():
        for r in json.loads(args.rows_file.read_text()):
            if r.get("arm") == "context":
                banked[(r["src"], r["tgt"])] = r["rungs"]

    args.out_root.mkdir(parents=True, exist_ok=True)
    ckpt = args.out_root / f"{Path(args.out_name).stem}.units.jsonl"
    done: set[tuple[str, str]] = set()
    if ckpt.exists():
        for line in ckpt.open(encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                same_mode = r.get("mode", "remap") == args.mode
                if r.get("n_draws_per_fold") == args.n_draws and same_mode:
                    done.add((r["src"], r["tgt"]))
        _log(f"[nulls] resume: {len(done)} pairs banked")

    acts_cache: dict[str, dict] = {}

    def acts(key: str) -> dict:
        if key not in acts_cache:
            if len(acts_cache) >= 4:
                acts_cache.clear()
            a = _load_activation_npz(args.activations_dir / f"{key}.npz")
            assert a is not None, f"missing activations for {key}"
            acts_cache[key] = a
        return acts_cache[key]

    svd_cache: dict = {}
    m_cache: dict = {}
    with ckpt.open("a", encoding="utf-8") as fh:
        for i, (src, tgt) in enumerate(pairs):
            if (src, tgt) in done:
                continue
            if args.mode == "cross":
                rec = null_for_pair_cross(
                    src,
                    tgt,
                    acts(src),
                    acts(tgt),
                    fold_of,
                    k,
                    n_draws=args.n_draws,
                    seed=args.seed + 97 * i,
                    banked_per_fold=banked_cross.get((src, tgt)),
                    svd_cache=svd_cache,
                    pilot=args.pilot,
                )
            else:
                rec = null_for_pair(
                    src,
                    tgt,
                    acts(src),
                    acts(tgt),
                    fold_of,
                    k,
                    n_draws=args.n_draws,
                    seed=args.seed + 97 * i,
                    banked_rungs=banked.get((src, tgt)),
                    svd_cache=svd_cache,
                    m_cache=m_cache,
                    pilot=args.pilot,
                )
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
            _log(f"[nulls] unit {i + 1}/{len(pairs)} {src} -> {tgt} done")

    records = [json.loads(x) for x in ckpt.read_text(encoding="utf-8").splitlines() if x.strip()]
    records = [
        r
        for r in records
        if r.get("n_draws_per_fold") == args.n_draws and r.get("mode", "remap") == args.mode
    ]
    payload = {
        "metadata": {
            **as_metadata_dict(git_provenance(_REPO)),
            "script_version": SCRIPT_VERSION,
            "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "n_pairs": len(records),
            "n_draws_per_fold": args.n_draws,
            "mode": args.mode,
            "seed": args.seed,
            "shard": [args.shard_index, args.shard_count],
            "wall_seconds": round(time.time() - t_start, 1),
        },
        "units": records,
    }
    out_path = args.out_root / args.out_name
    tmp = out_path.with_name(out_path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    os.replace(tmp, out_path)
    _log(f"[nulls] wrote {out_path} ({len(records)} pairs, {time.time() - t_start:.0f}s)")
    _log("[phase=done]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
