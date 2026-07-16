#!/usr/bin/env python3
# ruff: noqa: RUF002, RUF003
# Intentional Unicode (→, ², Δ, λ, ×) in scientific docstrings + log messages.
"""Issue #928 Phase F: the six-arm CoT-decomposition fit battery + nulls + bootstrap.

Fits the family of linear maps among per-part activation summaries (plan §1):
D (ctx→ans, the direct baseline), A (ctx→cot), B (cot→ans, oracle stage 2),
A∘B (composed, fold-coherent), J (ctx→concat(cot,ans)), G (concat(ctx,cot)→ans)
— plus the parent-parity cell ``d_parity`` (ctx_last → ans_mean, the exact
input/output configuration behind #810's 0.800/0.804 reference, its own
registered H1 read + null band) and the identity ceiling — under the inherited
#810 estimator (LOCO ridge, nested-CV λ over ``RIDGE_LAMBDAS``, PCA-48 target
via ``robust_pca_basis``, fp64, skill-over-mean R² DV), in three regimes:

- ``avg_q``: probe-averaged rows, n = n_contexts; per-fold standardization
  (the EXACT inherited estimator; parity-gated against
  ``vectorized_mlp_skill.ridge_predict_loco_centered``); registered same-type
  combos {mean/mean, max/max, boundary/boundary} + the exploratory 3×3
  input×output cross for D/A/B.
- ``indiv``: per-(C, q) rows, n ≈ 2,400, context-GROUP folds (48 rows leave
  together — never pointwise LOO; `.claude/rules/ood-generalization-folds.md`);
  mean/mean registered; FULL-DATA design standardization (the plan-§9
  shared-Gram basis — see ``issue928_null_bootstrap`` module docstring; a
  one-layer per_fold-vs-full_data sensitivity delta is reported).
- ``avg_t``: the CoT-grain-averaged re-read (plan §4.6 regime 2) — NO new
  fits: for the linear map B(t̄_C) = mean_q B(t_{C,q}) exactly, so the avg_t
  skill is a per-context group-mean re-reduction of the indiv predictions.

Folds: LOCO (banded) + LOFO by family (ordering-only); identity ceilings
recomputed per regime × fold scheme; every headline fold-labeled.

Nulls (plan §6): selection-symmetric label-shuffle battery, ONE perm matrix
per regime (seed 658, context-level in avg_q / context-GROUP-level in indiv)
shared across ALL cells so the analyzer's per-draw max-over-{combo, layer}
selection is coherent; per-draw × per-axis matrices persisted to
``null_matrix_<regime>.json``. §9 arithmetic: 1,000 draws × (504 avg_q +
196 indiv) cells as batched GEMMs over precomputed per-fold factors
(``issue928_null_bootstrap.grouped_null_skills`` — the ``issue810_batched_null``
identities extended to group folds; no serial per-draw refit anywhere).

Bootstrap (plan §6): paired per-context bootstrap, 2,000 draws, ONE shared
resample-index matrix (seed 42) across arms/combos/layers; Δskill CIs at the
PRIMARY frozen direct-arm-best-layer convention + frozen_full_data own-best +
per-replicate best-vs-best secondaries. No difference-null band is registered
(the bootstrap CI is the inferential object).

MLP validity (registered avg_q arms, LOCO, mean/mean):
``vectorized_mlp_skill.fit_batched_loco_mlp_multihead`` (batched across cells
by shape — the #722 vectorize-many-cell-fits mandate), chunk_size 256.

Restartability (round 2 — the #823 accumulate-in-memory class): every
completed (regime, layer) persists an atomic ``partial/<regime>/layer_<L>.pt``
unit under ``--out``, keyed by a ``fit_manifest.json`` over every
output-affecting arg (store identity digest, layers, combos, n_perms, seed,
standardization, cross flag, std-sensitivity layer, device); a re-run with a
matching manifest SKIPS completed layers, a mismatch discards stale units.
The per-context decompositions (``decomp_<regime>.pt``) and fit JSONs upload
to the HF data repo on the normal exit path (GCE DELETEs the boot disk).

Usage::

    # production (on the pod, after the store lands):
    EPM_FIT_DEVICE=cuda uv run python scripts/issue928_fit_decomposition.py \\
        --store data/issue_928/store --out eval_results/issue_928

    # smoke (= the same battery on the small-store subset):
    uv run python scripts/issue928_fit_decomposition.py \\
        --store /tmp/issue-928-smoke/data/store --out /tmp/issue-928-smoke/out \\
        --layers 0 1 --n-perms 25 --n-boot 100 --no-mlp --smoke
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv(str(PROJECT_ROOT / ".env"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from issue658_fit_predictors import _requested_device, _resolve_device  # noqa: E402
from issue928_common import (  # noqa: E402
    BOOTSTRAP_DRAWS,
    BOOTSTRAP_SEED,
    DECOMP_TENSORS_PREFIX,
    FIT_RESULTS_PREFIX,
    PCA_TARGET_DIM_CAP,
    REGISTERED_SUMMARIES,
    SHUFFLE_NULL_PERMS,
    SHUFFLE_NULL_SEED,
    dump_json,
    load_json,
    part_summary_name,
    reproducibility_metadata,
    upload_folder_scoped_verify,
)
from issue928_null_bootstrap import (  # noqa: E402
    GroupRidgeDesign,
    assert_group_ridge_matches_serial,
    bootstrap_skills,
    fit_predict_grouped,
    group_folds,
    grouped_null_skills,
    grouped_null_skills_multi,
    grouped_skill,
    make_bootstrap_index_matrix,
    make_group_perm_matrix,
    predict_external,
    stat_summary,
)

from explore_persona_space.analysis.vectorized_mlp_skill import (  # noqa: E402
    MLPGroup,
    fit_batched_loco_mlp_multihead,
    robust_pca_basis,
    skill_over_mean_r2,
)

logger = logging.getLogger("issue928_fit")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

NULL_ARMS = ("d_ctx2ans", "a_ctx2cot", "b_cot2ans", "comp_pred", "j_joint", "g_aug", "d_parity")


# ── intra-phase restartability (code-review r1 BLOCKER — the #823 class) ──────
#
# Phase F is a projected multi-hour regime × layer × cell × null battery, so
# every per-(regime, layer) unit persists the moment it completes (atomic
# tmp+os.replace write under <out>/partial/<regime>/) and an entry-time
# predicate skips completed units. The skip is keyed by a manifest over EVERY
# output-affecting arg (store identity digest, layers, combos, n_perms, the
# shuffle seed, standardization, cross flag, std-sensitivity layer, device);
# n_boot is deliberately NOT in the key — the bootstrap is a pure re-reduction
# of the persisted decompositions, recomputed fresh every run, so it does not
# change the per-layer units. A manifest mismatch DISCARDS the stale units
# (recompute; never silently reuse wrong cached rows — the #722 r3 lesson).
# On the GCE lane a crash additionally uploads eval_results_issue_928/ partial
# artifacts via the EXIT-trap crash persist, so these units survive DELETE.

FIT_MANIFEST_NAME = "fit_manifest.json"


def _atomic_torch_save(obj: object, path: Path) -> None:
    """``torch.save`` via tmp + ``os.replace`` — a crash mid-write never leaves a live unit."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def prepare_checkpoint_dir(root: Path, regime: str, key: dict) -> Path:
    """Resolve the per-regime partial-units dir; DISCARD stale units on key mismatch.

    Returns the directory (created if needed) with ``fit_manifest.json``
    holding ``key``. An existing matching manifest means completed
    ``layer_<L>.pt`` units in the dir are reusable; any mismatch (or an
    unreadable manifest) wipes the dir so no unit from a different regime
    configuration can be silently reused.
    """
    d = root / regime
    man = d / FIT_MANIFEST_NAME
    if man.is_file():
        try:
            existing = load_json(man)
        except (json.JSONDecodeError, OSError, ValueError):
            existing = None
        if existing == key:
            n_units = len(list(d.glob("layer_*.pt")))
            logger.info(
                "[resume] regime=%s: fit manifest matches — %d completed layer unit(s) reusable",
                regime,
                n_units,
            )
            return d
        changed = sorted(
            k for k in set(key) | set(existing or {}) if (existing or {}).get(k) != key.get(k)
        )
        logger.warning(
            "[resume] regime=%s: fit manifest MISMATCH on %s — discarding stale partial units",
            regime,
            changed,
        )
        shutil.rmtree(d)
    d.mkdir(parents=True, exist_ok=True)
    dump_json(key, man)
    return d


def _merge_layer_unit(
    unit: dict, grid: dict, null_matrix: dict, decomp: dict, extras: dict
) -> None:
    """Fold one per-layer unit into the regime accumulators (layer order = caller's loop)."""
    for arm, combos_ in unit["grid"].items():
        for combo, schemes in combos_.items():
            for scheme, cells in schemes.items():
                grid.setdefault(arm, {}).setdefault(combo, {}).setdefault(scheme, []).extend(cells)
    for arm, combos_ in unit["null"].items():
        for combo, by_layer in combos_.items():
            null_matrix.setdefault(arm, {}).setdefault(combo, {}).update(by_layer)
    decomp.update(unit["decomp"])
    for arm, combos_ in unit["avg_t"].items():
        for combo, cells in combos_.items():
            extras["avg_t"].setdefault(arm, {}).setdefault(combo, []).extend(cells)
    if unit.get("std_sensitivity") is not None:
        extras["std_sensitivity"] = unit["std_sensitivity"]


# ── store loading ─────────────────────────────────────────────────────────────


class Store:
    """Loaded per-(C, q) summary store — cell lists derive from ITS manifest.

    The smoke/production unification contract: the fit battery enumerates
    ``manifest["context_ids"]`` (whatever subset the extract dispatcher ran),
    never a hardcoded 50-grid. ``blob_subdir`` (DEFAULT-PRESERVING, follow-up
    plan v6: the default keeps the parent ``percq_summaries/`` layout
    byte-for-byte) locates the per-context ``.pt`` blobs relative to the
    store root — the matched-length store passes ``"."`` (flat layout,
    manifest + blobs in one folder).
    """

    def __init__(self, store_dir: Path, blob_subdir: str = "percq_summaries"):
        man = load_json(store_dir / "manifest.json")
        self.manifest = man
        self.ctx_ids: list[str] = man["context_ids"]
        self.families: dict[str, str] = man["families"]
        self.layers: list[int] = man["capture_layers"]
        self.summary_names: list[str] = man["summary_names"]
        self.sidx = {n: i for i, n in enumerate(self.summary_names)}
        self.blobs = {}
        for c in self.ctx_ids:
            b = torch.load(store_dir / blob_subdir / f"{c}.pt", weights_only=False)
            assert b["context_id"] == c, (b["context_id"], c)
            self.blobs[c] = b
        self.H = int(self.blobs[self.ctx_ids[0]]["per_q"].shape[-1])
        # indiv row bookkeeping: groups[i] = context index of row i (fold grain).
        self.row_ctx: list[int] = []
        for ci, c in enumerate(self.ctx_ids):
            self.row_ctx.extend([ci] * int(self.blobs[c]["per_q"].shape[0]))
        self.groups = np.asarray(self.row_ctx, dtype=np.int64)
        self.fam_of_ctx = np.asarray(
            [
                sorted({self.families[c] for c in self.ctx_ids}).index(self.families[c])
                for c in self.ctx_ids
            ],
            dtype=np.int64,
        )

    def avgq(self, sname: str, li: int) -> np.ndarray:
        """(n_ctx, H) probe-averaged summary matrix at layer index ``li``."""
        rows = [
            self.blobs[c]["probe_avg"][self.sidx[sname], li].float().numpy() for c in self.ctx_ids
        ]
        return np.stack(rows).astype(np.float64)

    def indiv(self, sname: str, li: int) -> np.ndarray:
        """(n_rows, H) per-(C, q) summary matrix at layer index ``li``."""
        rows = [
            self.blobs[c]["per_q"][:, self.sidx[sname], li].float().numpy() for c in self.ctx_ids
        ]
        return np.concatenate(rows, axis=0).astype(np.float64)

    def _blob_content_digest(self, c: str) -> str:
        """Per-context capture-content fingerprint linking the fit resume key
        to the ACTUAL blob data: the extractor-written ``rollout_digest``
        (generation identity, round 3) when present, else a sha256 over the
        ``per_q`` tensor bytes (legacy blobs predating the field) — so
        identical row counts from DIFFERENT completions still change the
        store identity."""
        import hashlib

        b = self.blobs[c]
        d = b.get("rollout_digest")
        if d:
            return str(d)
        return hashlib.sha256(b["per_q"].contiguous().numpy().tobytes()).hexdigest()[:16]

    def identity_digest(self) -> str:
        """sha256 (16 hex) over the store's output-affecting identity (the resume key).

        Stable manifest fields + realized per-context row counts + the
        generation cap (``max_new_tokens``, from the store manifest) + the
        per-context blob CONTENT digests (round 3, code-review r2 BLOCKER
        ``long-loop-restartability-fit-capture``) — so a re-extract that
        changes the completions invalidates the partial fit units via
        ``prepare_checkpoint_dir`` even when rung / row counts / shapes
        coincide, without hashing the multi-GB tensors of digest-bearing
        stores.
        """
        import hashlib

        ident = {
            "context_ids": self.ctx_ids,
            "families": {c: self.families[c] for c in self.ctx_ids},
            "capture_layers": [int(x) for x in self.layers],
            "summary_names": list(self.summary_names),
            "probe_pool_hash": self.manifest.get("probe_pool_hash"),
            "model": self.manifest.get("model"),
            "rung": self.manifest.get("rung"),
            "max_new_tokens": self.manifest.get("max_new_tokens"),
            "rows_per_ctx": {c: int(self.blobs[c]["per_q"].shape[0]) for c in self.ctx_ids},
            "blob_digests": {c: self._blob_content_digest(c) for c in self.ctx_ids},
            "hidden": int(self.H),
        }
        return hashlib.sha256(json.dumps(ident, sort_keys=True).encode()).hexdigest()[:16]


# ── one regime's fit battery ──────────────────────────────────────────────────


def _pca_target(Y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Full-data PCA-48 target (the inherited #810 convention: basis fit ONCE on
    the full target matrix — exactly ``issue810_fit_reconstruction._fit_one_cell``;
    centering stays per-fold inside the ridge). Returns (Y_pca, mu, comps)."""
    pca_dim = min(PCA_TARGET_DIM_CAP, max(1, Y.shape[0] - 2))
    mu, comps, _ = robust_pca_basis(Y, pca_dim)
    return (Y - mu) @ comps.T, mu, comps


def fit_regime(  # noqa: C901 — linear per-layer arm battery; see the arm comments
    store: Store,
    regime: str,
    layers_idx: list[int],
    combos: list[str],
    device: str,
    n_perms: int,
    do_cross: bool,
    draw_chunk: int,
    std_sensitivity_layer: int | None,
    checkpoint_dir: Path | None = None,
) -> tuple[dict, dict, dict, dict]:
    """All arms × combos × layers × {LOCO, LOFO} for one regime (+ nulls).

    Returns ``(grid, null_matrix, decomp, extras)`` where ``decomp`` keys
    ``(arm, combo, layer)`` → per-context (ss_res, ss_tot) LOCO decompositions
    (the bootstrap's re-reduction input) and ``extras`` carries the avg_t
    re-read + the standardization sensitivity delta.

    §9 sharing contract: ONE ``GroupRidgeDesign`` per (input-rep, layer, fold
    scheme) — its fold eigendecompositions serve every target, λ, and null
    draw touching that design (#823 guard). Designs are built and freed per
    layer (memory bound = one layer's designs).

    Restartability (round 2, the #823 class): with ``checkpoint_dir`` set,
    each completed layer persists an atomic ``layer_<L>.pt`` unit (grid cells +
    null draws + decompositions + avg_t + std-sensitivity for that layer) and
    a re-run SKIPS layers whose unit already exists — validity of the units is
    the caller's manifest contract (``prepare_checkpoint_dir``).
    """
    n_ctx = len(store.ctx_ids)
    if regime == "avg_q":
        groups = np.arange(n_ctx)
        mat = store.avgq
        standardization = "per_fold"  # the EXACT inherited estimator at n=50
    else:
        groups = store.groups
        mat = store.indiv
        standardization = "full_data"  # the plan-§9 shared-Gram basis (see module doc)
    group_order = list(range(n_ctx))
    folds_loco = group_folds(groups, group_order)
    fam_groups = store.fam_of_ctx[groups]  # per-row family index
    fam_order = sorted(set(store.fam_of_ctx.tolist()))
    folds_lofo = group_folds(fam_groups, fam_order) if len(fam_order) > 1 else None
    ctx_of_loco_fold = group_order  # fold f holds context f (battery order)

    perm = make_group_perm_matrix(
        groups, group_order, n_perms, np.random.default_rng(SHUFFLE_NULL_SEED)
    )

    grid: dict = {}
    null_matrix: dict = {}
    decomp: dict = {}
    extras: dict = {"avg_t": {}, "std_sensitivity": None}

    for li in layers_idx:
        layer = store.layers[li]
        unit_path = (checkpoint_dir / f"layer_{int(layer)}.pt") if checkpoint_dir else None
        if unit_path is not None and unit_path.is_file():
            unit = torch.load(unit_path, weights_only=False)
            assert int(unit["layer"]) == int(layer), (unit["layer"], layer)
            _merge_layer_unit(unit, grid, null_matrix, decomp, extras)
            logger.info(
                "[phase=fit] regime=%s layer %d SKIPPED (resumed from partial/%s/%s)",
                regime,
                layer,
                regime,
                unit_path.name,
            )
            continue
        t_layer = time.time()
        # per-LAYER accumulators — persisted as ONE durable unit the moment the
        # layer completes, then merged into the regime accumulators above.
        grid_l: dict = {}
        null_l: dict = {}
        decomp_l: dict = {}
        avgt_l: dict = {}
        std_l: dict | None = None

        def record(arm, combo, scheme, layer, res, extra=None, _g=grid_l):
            cell = {"layer": int(layer), "skill": res["skill"], "n": len(groups)}
            if extra:
                cell.update(extra)
            _g.setdefault(arm, {}).setdefault(combo, {}).setdefault(scheme, []).append(cell)

        designs: dict[tuple, GroupRidgeDesign] = {}

        def get_design(key: tuple, X: np.ndarray, scheme: str, _d=designs) -> GroupRidgeDesign:
            k = (*key, scheme)
            if k not in _d:
                _d[k] = GroupRidgeDesign(
                    X,
                    folds_loco if scheme == "loco" else folds_lofo,
                    device=device,
                    standardization=standardization,
                )
            return _d[k]

        # target PCA bases per (part, summary) at this layer (full-data, cached).
        pca_cache: dict[tuple, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

        def get_pca(part: str, s: str, _c=pca_cache, _li=li):
            k = (part, s)
            if k not in _c:
                _c[k] = _pca_target(mat(part_summary_name(part, s), _li))
            return _c[k]

        schemes = ["loco"] + (["lofo"] if folds_lofo is not None else [])
        for s in combos:
            X_ctx = mat(part_summary_name("ctx", s), li)
            X_cot = mat(part_summary_name("cot", s), li)
            X_cat = np.concatenate([X_ctx, X_cot], axis=1)
            Y_ans_pca, mu_ans, C_ans = get_pca("ans", s)
            Y_cot_pca, mu_cot, C_cot = get_pca("cot", s)
            Y_joint = np.concatenate(
                [mat(part_summary_name("cot", s), li), mat(part_summary_name("ans", s), li)],
                axis=1,
            )
            Y_joint_pca, mu_joint, C_joint = _pca_target(Y_joint)
            combo = f"{s}/{s}"
            for scheme in schemes:
                folds = folds_loco if scheme == "loco" else folds_lofo
                des_ctx = get_design(("ctx", s), X_ctx, scheme)
                des_cot = get_design(("cot", s), X_cot, scheme)
                des_cat = get_design(("cat", s), X_cat, scheme)
                des_ans = get_design(("ans", s), mat(part_summary_name("ans", s), li), scheme)

                # D: ctx -> ans
                pred_d, _, _ = fit_predict_grouped(des_ctx, Y_ans_pca)
                res_d = grouped_skill(pred_d, Y_ans_pca, folds)
                record("d_ctx2ans", combo, scheme, layer, res_d)
                # A: ctx -> cot
                pred_a, _, _models_a = fit_predict_grouped(des_ctx, Y_cot_pca)
                res_a = grouped_skill(pred_a, Y_cot_pca, folds)
                record("a_ctx2cot", combo, scheme, layer, res_a)
                # B: cot -> ans (oracle stage 2)
                pred_b, _, models_b = fit_predict_grouped(des_cot, Y_ans_pca)
                res_b = grouped_skill(pred_b, Y_ans_pca, folds)
                record("b_cot2ans", combo, scheme, layer, res_b)
                # A∘B composed (fold-coherent, plan §4.6): decode stage A's held
                # predictions to ambient R^H, feed through stage B's fold model.
                xhat_by_fold = [mu_cot + pred_a[held] @ C_cot for (_tr, held) in folds]
                pred_comp_stacked = predict_external(des_cot, models_b, xhat_by_fold)
                pred_comp = np.zeros_like(pred_b)
                ofs = 0
                for _tr, held in folds:
                    pred_comp[held] = pred_comp_stacked[ofs : ofs + held.size]
                    ofs += held.size
                res_comp = grouped_skill(pred_comp, Y_ans_pca, folds)
                record("comp_pred", combo, scheme, layer, res_comp)
                # J joint target (own basis) + the comparable answer-half read.
                pred_j, _, _ = fit_predict_grouped(des_ctx, Y_joint_pca)
                res_j = grouped_skill(pred_j, Y_joint_pca, folds)
                amb = mu_joint + pred_j @ C_joint  # (n, 2H)
                ans_half = amb[:, store.H :]
                pred_j_ans = (ans_half - mu_ans) @ C_ans.T
                res_j_ans = grouped_skill(pred_j_ans, Y_ans_pca, folds)
                record(
                    "j_joint",
                    combo,
                    scheme,
                    layer,
                    res_j,
                    extra={"skill_ans_read_in_d_basis": res_j_ans["skill"]},
                )
                # G augmented: concat(ctx, cot) -> ans (same target/folds as D).
                pred_g, _, _ = fit_predict_grouped(des_cat, Y_ans_pca)
                res_g = grouped_skill(pred_g, Y_ans_pca, folds)
                record("g_aug", combo, scheme, layer, res_g)
                # identity ceiling (estimator bound at this n).
                pred_i, _, _ = fit_predict_grouped(des_ans, Y_ans_pca)
                res_i = grouped_skill(pred_i, Y_ans_pca, folds)
                record("ident", combo, scheme, layer, res_i)

                if scheme == "loco":
                    for arm, res in [
                        ("d_ctx2ans", res_d),
                        ("a_ctx2cot", res_a),
                        ("b_cot2ans", res_b),
                        ("comp_pred", res_comp),
                        ("j_joint", res_j),
                        ("j_joint_ans_read", res_j_ans),
                        ("g_aug", res_g),
                        ("ident", res_i),
                    ]:
                        decomp_l[(arm, combo, layer)] = {
                            "ss_res": np.asarray(res["ss_res_by_group"]),
                            "ss_tot": np.asarray(res["ss_tot_by_group"]),
                            "ctx_order": [store.ctx_ids[g] for g in ctx_of_loco_fold],
                        }
                    if regime == "indiv":
                        # avg_t re-read (plan §4.6 regime 2): group-mean the indiv
                        # predictions + targets per context — EXACT for a linear
                        # map (B(t̄) = mean_q B(t)); no new fits.
                        for arm, pred in [
                            ("d_ctx2ans", pred_d),
                            ("a_ctx2cot", pred_a),
                            ("b_cot2ans", pred_b),
                            ("comp_pred", pred_comp),
                            ("g_aug", pred_g),
                        ]:
                            tgt = Y_cot_pca if arm == "a_ctx2cot" else Y_ans_pca
                            ss_res_g, ss_tot_g = [], []
                            for _tr, held in folds:
                                ybar = tgt[held].mean(axis=0)
                                pbar = pred[held].mean(axis=0)
                                tmean = tgt[_tr].mean(axis=0)
                                ss_res_g.append(float(np.sum((ybar - pbar) ** 2)))
                                ss_tot_g.append(float(np.sum((ybar - tmean) ** 2)))
                            sr, st = float(np.sum(ss_res_g)), float(np.sum(ss_tot_g))
                            avgt_l.setdefault(arm, {}).setdefault(combo, []).append(
                                {
                                    "layer": int(layer),
                                    "skill": (float("nan") if st < 1e-12 else 1.0 - sr / st),
                                    "ss_res_by_group": ss_res_g,
                                    "ss_tot_by_group": ss_tot_g,
                                }
                            )

                    # nulls (registered cells; LOCO — the banded fold). Cells
                    # grouped per DESIGN so the expensive per-(fold, λ) N_λ
                    # factors build ONCE per design instead of once per arm
                    # (round-2 N_λ-rebuild fix; parity-gated).
                    if n_perms > 0:
                        null_jobs = [
                            (
                                des_ctx,
                                [
                                    ("d_ctx2ans", Y_ans_pca, None),
                                    ("a_ctx2cot", Y_cot_pca, None),
                                    ("j_joint", Y_joint_pca, None),
                                ],
                            ),
                            (
                                des_cot,
                                [
                                    ("b_cot2ans", Y_ans_pca, None),
                                    (
                                        "comp_pred",
                                        Y_ans_pca,
                                        [
                                            des_cot.xdot_for(f, xhat_by_fold[f])
                                            for f in range(len(folds))
                                        ],
                                    ),
                                ],
                            ),
                            (des_cat, [("g_aug", Y_ans_pca, None)]),
                        ]
                        for des, jobs in null_jobs:
                            draws_by_cell = grouped_null_skills_multi(
                                des,
                                [(Yp, xo) for _a, Yp, xo in jobs],
                                perm,
                                draw_chunk=draw_chunk,
                            )
                            for (arm, _Yp, _xo), draws in zip(jobs, draws_by_cell, strict=True):
                                null_l.setdefault(arm, {}).setdefault(combo, {})[str(layer)] = draws

        # d_parity: ctx_last -> ans_mean (registered H1 read; falls out of the
        # 3×3 cross — labeled + given its OWN null band, plan §5).
        X_par = mat(part_summary_name("ctx", "boundary"), li)
        Y_ans_pca_m, _, _ = get_pca("ans", "mean")
        for scheme in schemes:
            folds = folds_loco if scheme == "loco" else folds_lofo
            des_par = get_design(("ctx", "boundary"), X_par, scheme)
            pred_p, _, _ = fit_predict_grouped(des_par, Y_ans_pca_m)
            res_p = grouped_skill(pred_p, Y_ans_pca_m, folds)
            record("d_parity", "boundary/mean", scheme, layer, res_p)
            if scheme == "loco":
                decomp_l[("d_parity", "boundary/mean", layer)] = {
                    "ss_res": np.asarray(res_p["ss_res_by_group"]),
                    "ss_tot": np.asarray(res_p["ss_tot_by_group"]),
                    "ctx_order": [store.ctx_ids[g] for g in ctx_of_loco_fold],
                }
                if n_perms > 0:
                    null_l.setdefault("d_parity", {}).setdefault("boundary/mean", {})[
                        str(layer)
                    ] = grouped_null_skills(des_par, Y_ans_pca_m, perm, draw_chunk=draw_chunk)

        # exploratory 3×3 cross for D/A/B (avg_q only; cheap at n=50 — plan §4.6).
        if do_cross:
            for arm, in_part, out_part in [
                ("d_ctx2ans", "ctx", "ans"),
                ("a_ctx2cot", "ctx", "cot"),
                ("b_cot2ans", "cot", "ans"),
            ]:
                for s_in in REGISTERED_SUMMARIES:
                    for s_out in REGISTERED_SUMMARIES:
                        if s_in == s_out:
                            continue  # same-type cells fit above
                        if arm == "d_ctx2ans" and s_in == "boundary" and s_out == "mean":
                            continue  # == d_parity (aliased, computed above)
                        Xc = mat(part_summary_name(in_part, s_in), li)
                        Yp, _, _ = get_pca(out_part, s_out)
                        des = get_design((in_part, s_in), Xc, "loco")
                        pred, _, _ = fit_predict_grouped(des, Yp)
                        res = grouped_skill(pred, Yp, folds_loco)
                        record(arm, f"{s_in}/{s_out}", "loco", layer, res)

        # one-layer standardization sensitivity (indiv; per_fold vs full_data).
        if regime == "indiv" and std_sensitivity_layer is not None and li == std_sensitivity_layer:
            X_ctx_m = mat(part_summary_name("ctx", "mean"), li)
            Yp_m, _, _ = get_pca("ans", "mean")
            des_pf = GroupRidgeDesign(
                X_ctx_m, folds_loco, device=device, standardization="per_fold"
            )
            pred_pf, _, _ = fit_predict_grouped(des_pf, Yp_m)
            sk_pf = grouped_skill(pred_pf, Yp_m, folds_loco)["skill"]
            des_fd = get_design(("ctx", "mean"), X_ctx_m, "loco")
            pred_fd, _, _ = fit_predict_grouped(des_fd, Yp_m)
            sk_fd = grouped_skill(pred_fd, Yp_m, folds_loco)["skill"]
            des_pf.free()
            std_l = {
                "layer": int(layer),
                "cell": "d_ctx2ans mean/mean LOCO",
                "skill_per_fold_std": sk_pf,
                "skill_full_data_std": sk_fd,
                "delta": sk_fd - sk_pf,
            }

        for d in designs.values():
            d.free()
        # Durable per-(regime, layer) unit the moment the layer completes
        # (round-2 restartability; atomic write, then merged into the regime
        # accumulators exactly like a resumed unit).
        unit = {
            "layer": int(layer),
            "grid": grid_l,
            "null": null_l,
            "decomp": decomp_l,
            "avg_t": avgt_l,
            "std_sensitivity": std_l,
        }
        if unit_path is not None:
            _atomic_torch_save(unit, unit_path)
        _merge_layer_unit(unit, grid, null_matrix, decomp, extras)
        logger.info(
            "[phase=fit] regime=%s layer %d done in %.1fs%s",
            regime,
            layer,
            time.time() - t_layer,
            " (unit persisted)" if unit_path is not None else "",
        )
    return grid, null_matrix, decomp, extras


# ── bootstrap statistics (registered layer conventions, plan §6) ──────────────


def bootstrap_statistics(decomp: dict, n_ctx: int, n_boot: int) -> dict:
    """Paired-bootstrap Δskill CIs at the three registered layer conventions.

    PRIMARY (confirmatory, plan §6): both arms at the DIRECT arm's full-data
    best LOCO layer (frozen BEFORE any draw; selection-free for the Δ, and
    conservative — winner's-curse inflation of D at its own layer biases
    Δ(X − D) down). SECONDARY: each arm's own full-data best layer
    (``frozen_full_data``, labeled data-selected) and per-replicate inherited
    best-vs-best. ONE shared resample-index matrix (seed 42) pairs every
    statistic (a pure re-reduction of the per-context LOCO decompositions; no
    refit). Everything — observed values, layer selections, draws — derives
    from ``decomp`` so the read is internally consistent by construction.
    """
    idx = make_bootstrap_index_matrix(n_ctx, n_boot, BOOTSTRAP_SEED)

    def obs_skill(arm: str, combo: str, layer: int) -> float:
        d = decomp[(arm, combo, layer)]
        tot = float(d["ss_tot"].sum())
        return float("nan") if tot < 1e-12 else 1.0 - float(d["ss_res"].sum()) / tot

    def layers_of(arm: str, combo: str) -> list[int]:
        return sorted(la for (a, c, la) in decomp if a == arm and c == combo)

    def best_layer(arm: str, combo: str) -> int:
        las = layers_of(arm, combo)
        return int(las[int(np.nanargmax([obs_skill(arm, combo, la) for la in las]))])

    def draws_for(arm: str, combo: str, layer: int) -> np.ndarray:
        d = decomp[(arm, combo, layer)]
        return bootstrap_skills(d["ss_res"], d["ss_tot"], idx)

    def per_layer_draws(arm: str, combo: str) -> np.ndarray:
        return np.stack(
            [draws_for(arm, combo, la) for la in layers_of(arm, combo)], axis=1
        )  # (B, L)

    mean_combo = "mean/mean"
    L_primary = best_layer("d_ctx2ans", mean_combo)

    out: dict = {
        "layer_conventions": {
            "primary_frozen_direct_best_layer": L_primary,
            "note": (
                "primary = frozen direct-arm full-data best LOCO layer (fixed before any "
                "bootstrap draw; no per-draw selection). Secondaries are frozen_full_data "
                "own-best (data-selected, labeled) and per-replicate best-vs-best."
            ),
        },
        "statistics": {},
    }

    def delta_stat(name: str, arm_hi: str, arm_lo: str, combo_hi: str, combo_lo: str):
        # primary: both at L_primary (frozen before any draw).
        obs_p = obs_skill(arm_hi, combo_hi, L_primary) - obs_skill(arm_lo, combo_lo, L_primary)
        dr_p = draws_for(arm_hi, combo_hi, L_primary) - draws_for(arm_lo, combo_lo, L_primary)
        # secondary own-best (frozen_full_data — frozen on the full data BEFORE
        # the bootstrap, never re-selected per draw; labeled data-selected).
        L_hi = best_layer(arm_hi, combo_hi)
        L_lo = best_layer(arm_lo, combo_lo)
        obs_ob = obs_skill(arm_hi, combo_hi, L_hi) - obs_skill(arm_lo, combo_lo, L_lo)
        dr_ob = draws_for(arm_hi, combo_hi, L_hi) - draws_for(arm_lo, combo_lo, L_lo)
        # secondary best-vs-best (layer selection INHERITED per replicate).
        dr_bb = np.nanmax(per_layer_draws(arm_hi, combo_hi), axis=1) - np.nanmax(
            per_layer_draws(arm_lo, combo_lo), axis=1
        )
        obs_bb = obs_skill(arm_hi, combo_hi, L_hi) - obs_skill(arm_lo, combo_lo, L_lo)
        out["statistics"][name] = {
            "primary_frozen_direct_best": stat_summary(obs_p, dr_p),
            "secondary_own_best_frozen_full_data": {
                "layers": {"hi": L_hi, "lo": L_lo},
                **stat_summary(obs_ob, dr_ob),
            },
            "secondary_best_vs_best_inherited": stat_summary(obs_bb, dr_bb),
        }

    mc = mean_combo
    delta_stat("H2_delta_g_minus_d", "g_aug", "d_ctx2ans", mc, mc)
    delta_stat("H3_delta_comp_minus_d", "comp_pred", "d_ctx2ans", mc, mc)
    delta_stat("H4_delta_g_minus_b", "g_aug", "b_cot2ans", mc, mc)
    if ("j_joint_ans_read", mc, L_primary) in decomp:
        delta_stat("delta_j_ansread_minus_d", "j_joint_ans_read", "d_ctx2ans", mc, mc)
    # H5 combo-Δ legs (candidate same-type combo vs the mean combo, same arm).
    for arm in ("b_cot2ans", "g_aug", "d_ctx2ans"):
        for s in ("max", "boundary"):
            combo = f"{s}/{s}"
            if (arm, combo, L_primary) in decomp:
                delta_stat(f"H5_{arm}_{s}_minus_mean", arm, arm, combo, mc)
    return out


# ── main ──────────────────────────────────────────────────────────────────────


def build_arg_parser() -> argparse.ArgumentParser:
    """The Phase-F CLI. Defaults preserve the #928 standalone behavior verbatim;
    an issue profile (e.g. the #1005 driver) overrides the upload prefixes so a
    child run can never clobber the parent's Hub artifacts (upload-verification
    v1 FAIL, required action 3)."""
    ap = argparse.ArgumentParser(description="Issue #928 Phase F: six-arm fit battery")
    ap.add_argument("--store", default=str(PROJECT_ROOT / "data" / "issue_928" / "store"))
    ap.add_argument("--out", default=str(PROJECT_ROOT / "eval_results" / "issue_928"))
    ap.add_argument("--regimes", nargs="*", default=["avg_q", "indiv"])
    ap.add_argument("--layers", nargs="*", type=int, default=None, help="layer-INDEX subset")
    ap.add_argument("--combos", nargs="*", default=list(REGISTERED_SUMMARIES))
    ap.add_argument("--n-perms", type=int, default=SHUFFLE_NULL_PERMS)
    ap.add_argument("--n-boot", type=int, default=BOOTSTRAP_DRAWS)
    ap.add_argument("--draw-chunk", type=int, default=16)
    ap.add_argument("--device", default=None, help="CLI > EPM_FIT_DEVICE > auto")
    ap.add_argument("--no-mlp", action="store_true")
    ap.add_argument("--no-cross", action="store_true", help="skip the exploratory 3x3 cross")
    ap.add_argument(
        "--std-sensitivity-layer",
        type=int,
        default=None,
        help="indiv layer INDEX for the per_fold-vs-full_data standardization delta "
        "(default: the middle fitted layer)",
    )
    ap.add_argument("--upload-prefix", default=None)
    ap.add_argument(
        "--decomp-upload-prefix",
        default=DECOMP_TENSORS_PREFIX,
        help="HF prefix for the decomp_*.pt tensor upload (fires only when "
        "--upload-prefix is set; default: the #928 prefix — an issue profile "
        "like #1005 MUST override so it never overwrites the parent's tensors)",
    )
    ap.add_argument("--skip-parity-gate", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    return ap


def main() -> int:
    args = build_arg_parser().parse_args()

    device = _resolve_device(_requested_device(args.device))
    logger.info("fit device: %s", device)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    if not args.skip_parity_gate:
        # Seeded serial-parity gate BEFORE any production fit (vectorize item 6).
        logger.info("[phase=parity] batched group-ridge vs serial reference (atol 1e-8)")
        parity = assert_group_ridge_matches_serial()
    else:
        parity = {"skipped": True}

    store = Store(Path(args.store))
    layers_idx = args.layers if args.layers is not None else list(range(len(store.layers)))
    std_li = (
        args.std_sensitivity_layer
        if args.std_sensitivity_layer is not None
        else layers_idx[len(layers_idx) // 2]
    )
    logger.info(
        "store: %d contexts, %d rows (indiv), %d layers to fit",
        len(store.ctx_ids),
        int(store.groups.shape[0]),
        len(layers_idx),
    )

    results: dict = {}
    boot: dict = {}
    ckpt_root = out_dir / "partial"
    for regime in args.regimes:
        logger.info("[phase=fit] regime=%s", regime)
        combos = args.combos if regime == "avg_q" else ["mean"]
        do_cross = regime == "avg_q" and not args.no_cross
        # Resume manifest over EVERY output-affecting arg of the per-layer
        # units (n_boot deliberately excluded — see the restartability block).
        regime_key = {
            "regime": regime,
            "store_identity": store.identity_digest(),
            "layers": [int(store.layers[li]) for li in layers_idx],
            "combos": list(combos),
            "n_perms": int(args.n_perms),
            "shuffle_null_seed": int(SHUFFLE_NULL_SEED),
            "standardization": "per_fold" if regime == "avg_q" else "full_data",
            "do_cross": bool(do_cross),
            "std_sensitivity_layer": int(std_li) if regime == "indiv" else None,
            "device": device,
        }
        ckpt_dir = prepare_checkpoint_dir(ckpt_root, regime, regime_key)
        grid, null_matrix, decomp, extras = fit_regime(
            store,
            regime,
            layers_idx,
            combos,
            device,
            args.n_perms,
            do_cross=do_cross,
            draw_chunk=args.draw_chunk,
            std_sensitivity_layer=(std_li if regime == "indiv" else None),
            checkpoint_dir=ckpt_dir,
        )
        results[regime] = {"grid": grid, "extras": {"std_sensitivity": extras["std_sensitivity"]}}
        if regime == "indiv" and extras["avg_t"]:
            results["avg_t"] = {
                "grid_note": (
                    "avg_t = per-context group-mean re-reduction of the indiv fits "
                    "(exact for the linear map: B(t_bar) = mean_q B(t)); plan §4.6 regime 2"
                ),
                "by_arm": extras["avg_t"],
            }
        # per-draw × per-axis null matrices persisted (plan §6 / §6.5).
        dump_json(
            {
                "dv": "recon_skill_over_mean_r2",
                "regime": regime,
                "axes": "arm -> combo -> layer -> [per-draw skill]",
                "n_perms": args.n_perms,
                "seed": SHUFFLE_NULL_SEED,
                "perm_grain": "context" if regime == "avg_q" else "context-group",
                "null": null_matrix,
            },
            out_dir / f"null_matrix_{regime}.json",
        )
        # bootstrap (LOCO decompositions; shared resample-index matrix).
        boot[regime] = bootstrap_statistics(decomp, len(store.ctx_ids), args.n_boot)
        # persist the per-context decompositions (re-reduction input, §6.5).
        torch.save(
            {str(k): {"ss_res": v["ss_res"], "ss_tot": v["ss_tot"]} for k, v in decomp.items()},
            out_dir / f"decomp_{regime}.pt",
        )

    # MLP validity companion (registered avg_q arms, mean/mean, LOCO — batched
    # across cells by shape per the #722 mandate; chunk_size 256 bounds memory).
    if not args.no_mlp and "avg_q" in args.regimes:
        logger.info("[phase=mlp] batched multihead MLP validity (registered avg_q arms)")
        jobs = []
        y_by_key = {}
        for li in layers_idx:
            X_ctx = store.avgq(part_summary_name("ctx", "mean"), li)
            X_cot = store.avgq(part_summary_name("cot", "mean"), li)
            Y_ans, _, _ = _pca_target(store.avgq(part_summary_name("ans", "mean"), li))
            Y_cot, _, _ = _pca_target(X_cot.copy())
            for arm, X, Y in [
                ("d_ctx2ans", X_ctx, Y_ans),
                ("a_ctx2cot", X_ctx, Y_cot),
                ("b_cot2ans", X_cot, Y_ans),
                ("g_aug", np.concatenate([X_ctx, X_cot], axis=1), Y_ans),
            ]:
                key = (arm, store.layers[li])
                jobs.append(MLPGroup(key, X, Y))
                y_by_key[key] = Y
        buckets: dict[tuple, list] = {}
        for g in jobs:
            buckets.setdefault((g.X.shape[0], g.X.shape[1], g.Y.shape[1]), []).append(g)
        mlp_skill = {}
        for shape, groups_ in buckets.items():
            logger.info("[phase=mlp] %d cells at shape %s", len(groups_), shape)
            res = fit_batched_loco_mlp_multihead(
                groups_, seed=SHUFFLE_NULL_SEED, device=device, chunk_size=256
            )
            for g in groups_:
                mlp_skill[f"{g.key[0]}@L{g.key[1]}"] = skill_over_mean_r2(
                    res.preds_by_key[g.key], y_by_key[g.key]
                )["skill"]
        results["mlp_validity_avg_q_mean"] = mlp_skill

    dump_json(
        {
            "dv": "held-out skill-over-mean R^2 per (arm x combo x layer x regime x fold)",
            "estimator": (
                "inherited #810: LOCO ridge, nested-CV lambda over RIDGE_LAMBDAS, "
                "full-data PCA-48 target basis (robust_pca_basis) with per-fold train "
                "centering; avg_q per-fold X standardization (exact parent convention); "
                "indiv full-data X standardization (the plan-9 shared-Gram basis - see "
                "issue928_null_bootstrap docstring + the persisted std_sensitivity delta)"
            ),
            "context_ids": store.ctx_ids,
            "capture_layers": [store.layers[li] for li in layers_idx],
            "n_indiv_rows": int(store.groups.shape[0]),
            "parity_gate": parity,
            "results": results,
            "bootstrap": boot,
            "n_perms": args.n_perms,
            "n_boot": args.n_boot,
            "reproducibility": reproducibility_metadata(),
            "smoke": args.smoke,
        },
        out_dir / "recon_skill_grid.json",
    )
    # bootstrap draws summary as its own primary-deliverable file (§6.5).
    dump_json(
        {
            "dv": "paired bootstrap delta-skill (shared resample-index matrix)",
            "seed": BOOTSTRAP_SEED,
            "n_boot": args.n_boot,
            "by_regime": boot,
            "reproducibility": reproducibility_metadata(),
        },
        out_dir / "bootstrap_deltaskill.json",
    )
    if args.upload_prefix:
        logger.info("[phase=upload] fit-result JSONs -> %s", args.upload_prefix)
        # ignore_patterns: fnmatch "*" crosses "/" so a bare "*.json" would
        # sweep in partial/<regime>/fit_manifest.json resume checkpoints.
        names = sorted(p.name for p in out_dir.glob("*.json"))
        upload_folder_scoped_verify(
            out_dir,
            args.upload_prefix or FIT_RESULTS_PREFIX,
            names,
            f"issue #928: fit results ({len(names)} JSONs)",
            allow_patterns=["*.json"],
            ignore_patterns=["partial/*"],
        )
        # Round-2 artifact-loss fix: the per-context LOCO decompositions (the
        # bootstrap's re-reduction input) upload as analysis tensors — on GCE
        # anything not on the Hub dies with the instance DELETE, and a post-hoc
        # CI re-reduction would otherwise require a full refit.
        decomp_names = sorted(p.name for p in out_dir.glob("decomp_*.pt"))
        if decomp_names:
            logger.info("[phase=upload] decomp tensors -> %s", args.decomp_upload_prefix)
            upload_folder_scoped_verify(
                out_dir,
                args.decomp_upload_prefix,
                decomp_names,
                f"issue #928: per-context LOCO decompositions ({len(decomp_names)} .pt)",
                allow_patterns=["decomp_*.pt"],
                ignore_patterns=["partial/*"],
            )
    # NOT [phase=done]: the run_all driver owns the terminal phase line.
    logger.info("[phase=fits_done] fits complete in %.1fs -> %s", time.time() - t0, out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
