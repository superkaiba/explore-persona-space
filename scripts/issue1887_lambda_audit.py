"""#1887 lambda-selection audit — re-read published #1345/#1310/#1639 cells.

Re-fits every affected published cell on the SAME pinned stores / folds / rows
under five arms (only the lambda selection / basis changes — task #1887 plan
v4 §3.5/§4):

  1. committed_replay   — each issue's OWN committed selector configuration
                          (#1345/#1310: pure unguarded GCV via the explicit
                          legacy pins; #1639: capped GCV 0.9, coinciding with
                          arm 2). The store/fold-identity SANITY leg: a
                          corrected read is reportable ONLY for cells whose
                          replay reproduces the committed R^2 (|dR2| <= 1e-3).
  2. gcv_capped_0p9     — the registered #1417 mitigation arm.
  3. inner_group_cv     — the new #1887 default selector.
  4. reduced_basis_k    — the HEADLINE corrected read (unselected): per-fold
                          TRAIN-only PCA basis, k = min(1024, n_train_min//2,
                          d_in), per-fold GCV in the k-dim basis (well-posed:
                          n_train > k by construction; the #1345 boundary-probe
                          recipe, commit 4682f0247a).
  5. forced_lambda      — DIAGNOSTIC ONLY sweep {1e2, 1e3, 1e4} in the ambient
                          basis (post-hoc-selected best-of-sweep; never quoted
                          as the single corrected figure). kNN-through-the-map
                          per lambda where the parent rig computes retrieval.

Per (issue, cell, arm): one atomic JSON checkpoint under <out>/cells/ with a
resume predicate keyed on (issue, cell_id, arm, selector-regime, store_rev),
plus one stdout line per unit. The corrections table (per issue) merges the
arm reads into one row per cell with the plan's verdict taxonomy — including
the `indeterminate (within fold SE)` override and the `basis-dependent
recovery` label — and the standing null / CI non-rerun disclosures.

Compute character: VM-local CPU, detached launch (see plan §8); per-cell cost
is pilot-gated (P0 runs `--pilot-cells 1` per issue BEFORE the P2 battery).
The REAL-store P0 pilot + P2 battery run POST-review as the orchestrator's
detached phase; the committed smoke path is `--synthetic-spec` (no HF, no
multi-GB staging).

CLI (see plan §10 "Exact commands"):
  uv run python scripts/issue1887_lambda_audit.py --issue 1345 \
      --stage-root /mnt/eps-data/$USER/issue1887_lambda_audit \
      --out eval_results/issue_1345/lambda_audit_1887 [--pilot-cells 1]
  uv run python scripts/issue1887_lambda_audit.py --synthetic-spec smoke \
      --out /tmp/i1887_smoke
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
for _p in (str(_SCRIPT_DIR), str(_REPO_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()  # thread caps (#847) before torch/numpy import

import numpy as np  # noqa: E402

import issue825_fit_cells as fit825  # noqa: E402
from explore_persona_space.analysis import mapping_baselines as mb  # noqa: E402

HF_DATA_REPO = "superkaiba1/explore-persona-space-data"
FORCED_LAMBDAS = (1e2, 1e3, 1e4)
ARMS = ("committed_replay", "gcv_capped_0p9", "inner_group_cv", "reduced_basis_k", "forced_lambda")
REPLAY_GATE_TOL = 1e-3  # plan §5.3 (recalibration DOWNWARD only, from P0)
STABLE_TOL = 0.1  # plan §3.5 taxonomy
CONTROL_TOL = 0.05  # n > d negative-control expectation (plan §3.5)
HEADLINE_LAYER = 19

# Committed selector configurations per issue (plan §3.5 arm 1 / assumption A5).
LEGACY_UNGUARDED = {"lambda_selection": "gcv", "gcv_dof_cap": None, "legacy_unguarded": True}
CAPPED_09 = {"lambda_selection": "gcv", "gcv_dof_cap": 0.9, "legacy_unguarded": False}
COMMITTED_SELECTOR_BY_ISSUE = {
    1345: LEGACY_UNGUARDED,
    1310: LEGACY_UNGUARDED,
    1639: CAPPED_09,  # tier15/tier2 ran capped GCV — replay coincides with arm 2
    825: LEGACY_UNGUARDED,  # n>d negative-control leg (committed #825 default)
}

# ---------------------------------------------------------------------------
# Pinned store revisions (concerns i1887-1310-store-rev-unpinned +
# i1887-variant-store-staging; plan A4/§5.4). Every pin below is the FULL
# last-commit oid of its prefix, probed 2026-07-30 via scoped
# list_repo_tree + get_paths_info(expand=True) — evidence in the r2
# epm:results marker on task #1887.
#
# #1310/#1639: ALL 58 files under store_onpolicy/ landed in ONE commit
# (2026-07-16T10:34:45Z, "issue1310: onpolicy prefill activation store"), and
# the prefix has never been touched since — a unique, single-version store
# (8.04 GB). Dating note (plan item (j)): the #1310 fits (cell metadata
# 2026-07-14T23:38..07-15T04:43Z) PREDATE the upload — persist-after-run, so
# the wall-clock "store last-modified predates the fits" ordering is inverted
# for #1310; with exactly one store version ever on HF the pin is unambiguous
# and the per-cell replay gate (|dR2| <= REPLAY_GATE_TOL) is the byte-identity
# check. The #1639 tier15 fits (git 261c71f9e56e, 2026-07-25Z) POSTdate the
# upload — ordering holds strictly there.
I1310_STORE_PREFIX = "issue1310_char_map/analysis_tensors/store_onpolicy"
I1310_STORE_REV = "687e5b348ef850308069c228534fef38cf53015a"

# #1345: the parent-run turnstore prefix carries ALL five parent-format stems
# ({instruct,pretrained}_{chat,naturalistic}_s + instruct_stories_s; 90 files,
# 92.17 GB, uploaded 2026-07-16T03:57..04:38Z, untouched since) — so every
# r1/r2/r3 cell of every variant stages from it. Variant-prefixed stores exist
# on HF ONLY for the three paired/slot variants below (each single-upload,
# shard files predating that variant's committed fits by 10 min..6 h); the
# other variants (assistant_named_story, conversation_paired_stories,
# followup_cjk_excluded) have NO store of their own — their cells consume the
# parent stems. ladder_rungs + conversation_paired_stories_assistant_base
# carry no cells_*.json and are never enumerated.
I1345_PARENT_STORE_PREFIX = "issue1345_framing/analysis_tensors/turnstore"
I1345_PARENT_STORE_REV = "e151f8fd30a466cd9e2f05b9b86cf6dde76bca74"
I1345_PARENT_MATCHED_PATH = "issue1345_framing/inputs/matched_n/matched_subsets.json"
I1345_PARENT_MATCHED_REV = "a5738c957cd41ef8e7985716064ade67717a890d"
# format_key values whose stems live in the PARENT turnstore prefix.
I1345_PARENT_FORMATS = ("chat", "naturalistic", "stories")
I1345_VARIANT_STORE_REVS = {
    # 14 files, 5.56 GB (instruct_stories_paired_s + instruct_stories_paired_op_s)
    "conversation_paired_stories_assistant": "1ef6def108678c458a03d190c8105ced55fe58a7",
    # 10 files, 4.92 GB (instruct_stories_paired_op_s)
    "onpolicy_assistant_story": "eca4accbf8eef9d4eebe546dbc8f3131c4031df4",
    # 13 files, 7.01 GB (instruct_stories_paired_slots_s)
    "story_slot_ablation": "c8ffc7a8d7412fd0492dd0b3a0e2e9f6374f80c7",
}
# Variant matched_subsets.json pins (probed 2026-07-30; contents verified:
# cpa r4_convs=2163 / op_companion_convs=117, oas r4_convs=2018, both
# shared_r1r2_convs=4724 — matching the committed cells' n_allowlist).
# story_slot_ablation has NO matched file on HF: its allowlist is the
# committed slot_row_coverage.json registered set (n=2163, in git).
I1345_VARIANT_MATCHED_REVS = {
    "assistant_named_story": "c06fffa6c5420faa8f883ce86d7d26fe4995dca2",
    "conversation_paired_stories": "25d0b70afb6777edd4575fd3d66d2b5441cd8d40",
    "conversation_paired_stories_assistant": "77f73466e0a7a3d7f29f31aaa8f0dd9372431045",
    "onpolicy_assistant_story": "87c5a80269d4e91f307f8a77e0c0cbdd741b3ce2",
}


# ---------------------------------------------------------------------------
# Cell / fold plumbing
# ---------------------------------------------------------------------------
@dataclass
class FoldData:
    """One outer fold at the cell's committed grain (single audited layer)."""

    fold_id: int
    X_tr: np.ndarray  # (n_tr, d) fit inputs
    Y_tr: np.ndarray  # (n_tr, d_out) fit targets
    X_ev: np.ndarray  # (n_te, d) eval inputs
    Y_true: np.ndarray  # (n_te, d_out) eval targets
    groups_tr: np.ndarray  # group ids over the FIT rows (inner-CV folds)


@dataclass
class CellSpec:
    issue: int
    cell_id: str
    variant: str
    committed_r2: float | None
    published_claim_ref: str
    store_rev: str
    load: object  # () -> list[FoldData]
    seed: int = 0
    compute_knn: bool = False
    control: bool = False
    committed_selector: dict = field(default_factory=dict)
    notes: str = ""


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
    os.replace(tmp, path)


def _git_commit() -> str:
    import subprocess

    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=_REPO_ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()


def assert_stage_headroom(stage_root: Path, projected_gb: float) -> None:
    """df headroom assert (plan §8 Disk): free >= 1.5x projected bytes."""
    stage_root.mkdir(parents=True, exist_ok=True)
    st = os.statvfs(stage_root)
    free_gb = st.f_bavail * st.f_frsize / 1e9
    need = 1.5 * projected_gb
    assert free_gb >= need, (
        f"staging headroom: {free_gb:.1f} GB free at {stage_root} < 1.5x projected "
        f"{projected_gb:.1f} GB — resize/clean before staging (plan §8)"
    )
    print(f"[audit] staging headroom OK: {free_gb:.1f} GB free >= {need:.1f} GB at {stage_root}")


# ---------------------------------------------------------------------------
# Arm engine (fit825 primitives only — the shared kernel; plan §4 controls)
# ---------------------------------------------------------------------------
def _selector_pins(cfg: dict):
    """Context-manager-free pin/restore pair for the fit825 module globals."""
    prev = (fit825.GCV_DOF_CAP, fit825.LEGACY_UNGUARDED_GCV)
    fit825.GCV_DOF_CAP = cfg.get("gcv_dof_cap")
    fit825.LEGACY_UNGUARDED_GCV = bool(cfg.get("legacy_unguarded"))
    return prev


def _restore_pins(prev) -> None:
    fit825.GCV_DOF_CAP, fit825.LEGACY_UNGUARDED_GCV = prev


def _fit_folds(
    folds: list[FoldData],
    *,
    lambda_selection: str,
    seed: int,
    basis: str = "ambient",
    forced_lambda: float | None = None,
    k_red: int | None = None,
) -> dict:
    """Fit every fold under one arm configuration; pooled + per-fold reads.

    The per-fold path mirrors ``fit825.heldout_r2_sweep``'s observed-fit loop
    exactly (same ``_prep_fold`` / ``_prep_inner_lambda`` / seed convention /
    held-out fold-mean SST), single audited layer — numerically identical to
    the parents' committed convention, plus per-fold R^2 persisted (plan §3.5
    refinement (i)).
    """
    ss_res = ss_tot = 0.0
    per_fold_r2: list[float] = []
    lam_per_fold: list[float] = []
    preds: list[np.ndarray] = []
    trues: list[np.ndarray] = []
    for fd in folds:
        if basis == "reduced":
            assert k_red is not None
            mu_r, basis_r = fit825._train_pca_basis(fd.X_tr, k_red)
            Z_tr = (np.asarray(fd.X_tr, dtype=np.float64) - mu_r) @ basis_r.T
            Z_ev = (np.asarray(fd.X_ev, dtype=np.float64) - mu_r) @ basis_r.T
            cache = fit825._prep_fold(Z_tr, Z_ev)
            groups = fd.groups_tr
        else:
            cache = fit825._prep_fold(fd.X_tr, fd.X_ev)
            groups = fd.groups_tr
        if lambda_selection == "inner-group-cv":
            cache["inner"] = fit825._prep_inner_lambda(
                Z_tr if basis == "reduced" else fd.X_tr,
                np.asarray(groups),
                fit825.N_INNER_LAMBDA_FOLDS,
                seed + 4242 + fd.fold_id,
            )
            if cache["inner"] is None:
                print(
                    f"[audit] WARN inner-group-cv fold {fd.fold_id}: <2 usable inner "
                    "group folds — capped-GCV fallback"
                )
        lambdas = [forced_lambda] if forced_lambda is not None else None
        pred, lam = fit825._ridge_predict_cached(cache, fd.Y_tr, return_lam=True, lambdas=lambdas)
        true = np.asarray(fd.Y_true, dtype=np.float64)
        mu = true.mean(0)
        ssr = float(np.sum((true - pred) ** 2))
        sst = float(np.sum((true - mu) ** 2))
        ss_res += ssr
        ss_tot += sst
        per_fold_r2.append(float("nan") if sst < 1e-12 else 1.0 - ssr / sst)
        lam_per_fold.append(float(lam))
        preds.append(np.asarray(pred, dtype=np.float64))
        trues.append(true)
    pooled = float("nan") if ss_tot < 1e-12 else 1.0 - ss_res / ss_tot
    pf = np.asarray(per_fold_r2, dtype=float)
    pf_ok = pf[np.isfinite(pf)]
    fold_se = float(np.std(pf_ok, ddof=1) / np.sqrt(len(pf_ok))) if len(pf_ok) >= 2 else None
    return {
        "r2_pooled": pooled,
        "r2_per_fold": [None if not np.isfinite(v) else float(v) for v in pf],
        "fold_se": fold_se,
        "selected_lambda_per_fold": lam_per_fold,
        "n_train_per_fold": [int(fd.X_tr.shape[0]) for fd in folds],
        "pred_pooled": np.concatenate(preds, axis=0),
        "true_pooled": np.concatenate(trues, axis=0),
    }


def _knn_reads(pred: np.ndarray, true: np.ndarray) -> dict:
    """kNN-through-the-map retrieval (the standing mapping read), both metrics."""
    return {
        m: mb.knn_retrieval(pred, true, ks=(1, 5, 10), metric=m) for m in ("euclidean", "cosine")
    }


def run_cell_arm(cell: CellSpec, arm: str, folds: list[FoldData]) -> dict:
    """One (cell, arm) unit — returns the checkpoint payload (JSON-safe)."""
    t0 = time.time()
    n_train_min = min(int(fd.X_tr.shape[0]) for fd in folds)
    d_in = int(folds[0].X_tr.shape[1])
    out: dict = {
        "issue": cell.issue,
        "cell_id": cell.cell_id,
        "variant": cell.variant,
        "arm": arm,
        "n_train_min": n_train_min,
        "d": d_in,
        "control": cell.control,
    }
    if arm == "committed_replay":
        cfg = dict(cell.committed_selector)
    elif arm == "gcv_capped_0p9":
        cfg = dict(CAPPED_09)
    elif arm == "inner_group_cv":
        cfg = {"lambda_selection": "inner-group-cv", "gcv_dof_cap": 0.9, "legacy_unguarded": False}
    elif arm == "reduced_basis_k":
        # The boundary-probe recipe: per-fold GCV inside the TRAIN-only k-dim
        # basis — well-posed by construction (n_train > k), so cap None never
        # trips the #1887 refusal guard (its ntr < d predicate is false).
        cfg = {"lambda_selection": "gcv", "gcv_dof_cap": None, "legacy_unguarded": False}
    elif arm == "forced_lambda":
        cfg = {"lambda_selection": "gcv", "gcv_dof_cap": None, "legacy_unguarded": False}
    else:  # pragma: no cover - CLI-validated
        raise ValueError(f"unknown arm {arm!r}")
    prev = _selector_pins(cfg)
    try:
        if arm == "forced_lambda":
            forced = {}
            for lam in FORCED_LAMBDAS:
                r = _fit_folds(
                    folds,
                    lambda_selection="gcv",
                    seed=cell.seed,
                    forced_lambda=float(lam),
                )
                entry = {
                    k: r[k]
                    for k in (
                        "r2_pooled",
                        "r2_per_fold",
                        "fold_se",
                        "selected_lambda_per_fold",
                    )
                }
                if cell.compute_knn:
                    entry["knn"] = _knn_reads(r["pred_pooled"], r["true_pooled"])
                forced[f"{lam:.0e}"] = entry
            out["forced_lambda"] = forced
            out["selection_bearing"] = True  # DIAGNOSTIC only — never a headline
        else:
            if arm == "reduced_basis_k":
                k_red = fit825.reduced_basis_k(n_train_min, d_in)
                out["k"] = int(k_red)
                out["k_rule"] = "min(1024, floor(n_train_min/2), d_in)"
                r = _fit_folds(
                    folds,
                    lambda_selection="gcv",
                    seed=cell.seed,
                    basis="reduced",
                    k_red=k_red,
                )
            else:
                r = _fit_folds(
                    folds,
                    lambda_selection=cfg["lambda_selection"],
                    seed=cell.seed,
                )
            out.update(
                {
                    k: r[k]
                    for k in (
                        "r2_pooled",
                        "r2_per_fold",
                        "fold_se",
                        "selected_lambda_per_fold",
                        "n_train_per_fold",
                    )
                }
            )
            if cell.compute_knn:
                out["knn"] = _knn_reads(r["pred_pooled"], r["true_pooled"])
            out["tripwire"] = fit825.degeneracy_tripwire(
                n_train=n_train_min,
                d=d_in,
                selected_lambdas=r["selected_lambda_per_fold"],
                r2_heldout=r["r2_pooled"],
                knn_at_1=(
                    out.get("knn", {}).get("euclidean", {}).get("acc_at_k", {}).get(1)
                    if cell.compute_knn
                    else None
                ),
                knn_chance=(
                    out.get("knn", {}).get("euclidean", {}).get("chance_at_k", {}).get(1)
                    if cell.compute_knn
                    else None
                ),
            )
    finally:
        _restore_pins(prev)
    out["selector_config"] = cfg
    out["elapsed_s"] = round(time.time() - t0, 2)
    return out


# ---------------------------------------------------------------------------
# Checkpoint / resume (plan §8: >50 units — per-unit persistence + resume)
# ---------------------------------------------------------------------------
def _resume_key(cell: CellSpec, arm: str) -> dict:
    cfg = dict(cell.committed_selector) if arm == "committed_replay" else arm
    return {
        "issue": cell.issue,
        "cell_id": cell.cell_id,
        "arm": arm,
        "selector_regime": cfg,
        "store_rev": cell.store_rev,
    }


def _unit_path(out_dir: Path, cell: CellSpec, arm: str) -> Path:
    return out_dir / "cells" / f"{cell.cell_id}__{arm}.json"


def run_units(cells: list[CellSpec], arms: tuple[str, ...], out_dir: Path) -> None:
    units = [(c, a) for c in cells for a in arms]
    total = len(units)
    done = 0
    for cell in cells:
        if cell.load is None:
            done += len(arms)
            print(
                f"[audit] unit {done}/{total} {cell.issue}/{cell.cell_id} SKIP all arms: "
                f"{cell.notes}",
                flush=True,
            )
            continue
        folds: list[FoldData] | None = None
        for arm in arms:
            done += 1
            path = _unit_path(out_dir, cell, arm)
            key = _resume_key(cell, arm)
            if path.exists():
                try:
                    existing = json.loads(path.read_text())
                except json.JSONDecodeError:
                    existing = None
                if existing is not None and existing.get("resume_key") == key:
                    print(
                        f"[audit] unit {done}/{total} {cell.issue}/{cell.cell_id}/{arm} "
                        "resume-skip",
                        flush=True,
                    )
                    continue
            if folds is None:
                t_load = time.time()
                folds = cell.load()
                assert folds, f"{cell.cell_id}: loader returned no usable folds"
                print(
                    f"[audit] loaded {cell.issue}/{cell.cell_id}: {len(folds)} folds "
                    f"n_train_min={min(f.X_tr.shape[0] for f in folds)} "
                    f"d={folds[0].X_tr.shape[1]} load={time.time() - t_load:.1f}s",
                    flush=True,
                )
            payload = run_cell_arm(cell, arm, folds)
            payload["resume_key"] = key
            payload["committed_r2"] = cell.committed_r2
            payload["published_claim_ref"] = cell.published_claim_ref
            payload["metadata"] = {
                "script": "scripts/issue1887_lambda_audit.py",
                "issue": 1887,
                "git_commit": _git_commit(),
                "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            # JSON-safety: drop the array fields the checkpoint never persists.
            payload.pop("pred_pooled", None)
            payload.pop("true_pooled", None)
            _write_json_atomic(path, payload)
            print(
                f"[audit] unit {done}/{total} {cell.issue}/{cell.cell_id}/{arm} "
                f"elapsed={payload['elapsed_s']}s r2={_fmt_r2(payload)}",
                flush=True,
            )
        folds = None  # release the cell's arrays before the next cell


def _fmt_r2(payload: dict) -> str:
    if "r2_pooled" in payload:
        return f"{payload['r2_pooled']:.4f}"
    fl = payload.get("forced_lambda", {})
    return "/".join(f"{v['r2_pooled']:.3f}" for v in fl.values()) if fl else "n/a"


# ---------------------------------------------------------------------------
# Corrections table (plan §3.5 row schema + verdict taxonomy)
# ---------------------------------------------------------------------------
NULL_RERUN_NOTE = "false — committed nulls selector-matched to the committed read only"
CI_NOTE = "point-estimate-only — committed CI machinery not re-run"
IDENTITY_BASELINE_NOTE = (
    "identity+learned-bias baseline is selector-invariant (no lambda enters it) — "
    "the parents' committed values carry over unchanged (stated exemption)"
)


def _knn_at_1(unit: dict, field: str):
    """k=1 read from a unit's euclidean kNN block (never `or`-chained — an
    acc@1 of exactly 0.0 is falsy and must survive; JSON string-keys ints)."""
    d = (unit.get("knn") or {}).get("euclidean", {}).get(field, {})
    for k in ("1", 1):
        if k in d:
            return d[k]
    return None


def _load_unit(out_dir: Path, cell_id: str, arm: str) -> dict | None:
    p = out_dir / "cells" / f"{cell_id}__{arm}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def _verdict(row: dict) -> str:
    """DISJOINT + exhaustive taxonomy over the reduced-basis HEADLINE value.

    Order: replay gate -> indeterminate override -> estimator-artifact (with
    ambient corroboration; reduced-only flip = basis-dependent recovery) ->
    degraded-consistent -> stable -> shifted (plan §3.5 refinements (i)/(iii)).
    """
    committed = row.get("committed_r2")
    replay = row.get("replay_r2")
    headline = row.get("corrected_reduced_basis_r2")
    if committed is None or replay is None:
        return "no-committed-reference"
    if abs(replay - committed) > REPLAY_GATE_TOL:
        return "replay-failed (excluded)"
    if headline is None:
        return "no-corrected-read"
    fold_se = row.get("reduced_basis_fold_se")
    delta = headline - committed
    if fold_se is not None and abs(delta) <= 2.0 * fold_se:
        return "indeterminate (within fold SE)"
    if committed < 0.0 and headline >= 0.0:
        # Narrating estimator-artifact AS GCV degeneracy additionally requires
        # an AMBIENT-arm corroboration (inner-CV or forced-lambda also flipping
        # non-negative); a reduced-basis-only flip is basis-dependent recovery.
        inner = row.get("corrected_inner_cv_r2")
        forced = [v for v in (row.get("forced_lambda_r2") or {}).values() if v is not None]
        ambient_flip = (inner is not None and inner >= 0.0) or any(v >= 0.0 for v in forced)
        return "estimator-artifact" if ambient_flip else "basis-dependent recovery"
    if committed < 0.0 and headline < 0.0:
        return "degraded-consistent"
    if committed >= 0.0 and abs(delta) <= STABLE_TOL:
        return "stable"
    return "shifted"


UNREFITTABLE_VERDICT = "un-refittable — store not resolvable"


def build_corrections_table(cells: list[CellSpec], out_dir: Path) -> dict:
    rows = []
    for cell in cells:
        if cell.load is None:
            # Plan §5.4: an unresolvable-store cell is NAMED with its reason —
            # never a whole-audit failure, and never counted in the replay gate.
            rows.append(
                {
                    "issue": cell.issue,
                    "cell_id": cell.cell_id,
                    "variant": cell.variant,
                    "control": cell.control,
                    "n_train": None,
                    "d": None,
                    "committed_selector": cell.committed_selector,
                    "committed_r2": cell.committed_r2,
                    "replay_r2": None,
                    "replay_delta": None,
                    "corrected_gcv_capped_r2": None,
                    "corrected_inner_cv_r2": None,
                    "corrected_reduced_basis_r2": None,
                    "reduced_basis_fold_se": None,
                    "reduced_basis_k": None,
                    "forced_lambda_r2": {},
                    "knn_at_1": None,
                    "knn_chance": None,
                    "tripwire": None,
                    "published_claim_ref": cell.published_claim_ref,
                    "null_rerun": NULL_RERUN_NOTE,
                    "ci_survival": CI_NOTE,
                    "mapping_baseline_note": IDENTITY_BASELINE_NOTE,
                    "verdict_label": UNREFITTABLE_VERDICT,
                    "notes": cell.notes,
                }
            )
            continue
        units = {arm: _load_unit(out_dir, cell.cell_id, arm) for arm in ARMS}
        replay = units.get("committed_replay") or {}
        capped = units.get("gcv_capped_0p9") or {}
        inner = units.get("inner_group_cv") or {}
        reduced = units.get("reduced_basis_k") or {}
        forced = (units.get("forced_lambda") or {}).get("forced_lambda", {})
        row = {
            "issue": cell.issue,
            "cell_id": cell.cell_id,
            "variant": cell.variant,
            "control": cell.control,
            "n_train": replay.get("n_train_min"),
            "d": replay.get("d"),
            "committed_selector": cell.committed_selector,
            "committed_r2": cell.committed_r2,
            "replay_r2": replay.get("r2_pooled"),
            "replay_delta": (
                None
                if cell.committed_r2 is None or replay.get("r2_pooled") is None
                else replay["r2_pooled"] - cell.committed_r2
            ),
            "corrected_gcv_capped_r2": capped.get("r2_pooled"),
            "corrected_inner_cv_r2": inner.get("r2_pooled"),
            "corrected_reduced_basis_r2": reduced.get("r2_pooled"),  # HEADLINE
            "reduced_basis_fold_se": reduced.get("fold_se"),
            "reduced_basis_k": reduced.get("k"),
            "forced_lambda_r2": {
                k: v.get("r2_pooled") for k, v in forced.items()
            },  # DIAGNOSTIC (post-hoc-selected; never the corrected figure)
            "knn_at_1": _knn_at_1(reduced, "acc_at_k"),
            "knn_chance": _knn_at_1(reduced, "chance_at_k"),
            "tripwire": replay.get("tripwire"),
            "published_claim_ref": cell.published_claim_ref,
            "null_rerun": NULL_RERUN_NOTE,
            "ci_survival": CI_NOTE,
            "mapping_baseline_note": IDENTITY_BASELINE_NOTE,
        }
        row["verdict_label"] = _verdict(row)
        if cell.control and row["replay_delta"] is not None:
            moved = [
                v
                for v in (
                    row["corrected_gcv_capped_r2"],
                    row["corrected_inner_cv_r2"],
                    row["corrected_reduced_basis_r2"],
                )
                if v is not None and cell.committed_r2 is not None
            ]
            row["control_max_abs_delta"] = (
                max(abs(v - cell.committed_r2) for v in moved) if moved else None
            )
            row["control_pass"] = (
                row["control_max_abs_delta"] is not None
                and row["control_max_abs_delta"] <= CONTROL_TOL
            )
        rows.append(row)
    n_gated = sum(1 for r in rows if r["verdict_label"] == "replay-failed (excluded)")
    n_unrefittable = sum(1 for r in rows if r["verdict_label"] == UNREFITTABLE_VERDICT)
    with_ref = [
        r
        for r in rows
        if r.get("committed_r2") is not None and r["verdict_label"] != UNREFITTABLE_VERDICT
    ]
    table = {
        "metadata": {
            "script": "scripts/issue1887_lambda_audit.py",
            "issue": 1887,
            "git_commit": _git_commit(),
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "replay_gate_tol": REPLAY_GATE_TOL,
            "stable_tol": STABLE_TOL,
            "control_tol": CONTROL_TOL,
            "headline_read": "corrected_reduced_basis_r2 (unselected)",
            "forced_lambda_role": "diagnostic only (post-hoc-selected best-of-sweep)",
        },
        "replay_gate": {
            "n_cells_with_reference": len(with_ref),
            "n_replay_failed": n_gated,
            "n_unrefittable": n_unrefittable,
            "pass_fraction": (None if not with_ref else 1.0 - n_gated / len(with_ref)),
            "gate": "PASS" if (not with_ref or n_gated / len(with_ref) <= 0.10) else "FAIL",
        },
        "rows": rows,
    }
    _write_json_atomic(out_dir / "corrections_table.json", table)
    _write_corrections_md(out_dir / "corrections_table.md", table)
    print(
        f"[audit] corrections table: {len(rows)} rows -> {out_dir}/corrections_table.json "
        f"(replay gate {table['replay_gate']['gate']})",
        flush=True,
    )
    return table


def _write_corrections_md(path: Path, table: dict) -> None:
    lines = [
        "# #1887 lambda-selection corrections table",
        "",
        f"Replay gate: **{table['replay_gate']['gate']}** "
        f"({table['replay_gate']['n_replay_failed']} of "
        f"{table['replay_gate']['n_cells_with_reference']} referenced cells failed "
        f"|dR2| <= {REPLAY_GATE_TOL}).",
        f"Headline corrected read: reduced-basis (unselected). Forced-lambda is "
        f"diagnostic only. Nulls: {NULL_RERUN_NOTE}. CIs: {CI_NOTE}.",
        "",
        "| cell | variant | n_train | d | committed | replay Δ | capped 0.9 | inner-CV |"
        " reduced (headline) | forced 1e2/1e3/1e4 | verdict |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]

    def _f(v):
        return "—" if v is None else f"{v:.4f}"

    for r in table["rows"]:
        fl = r.get("forced_lambda_r2") or {}
        forced = "/".join(_f(fl.get(f"{lam:.0e}")) for lam in FORCED_LAMBDAS)
        verdict = r["verdict_label"]
        if r.get("notes"):  # un-refittable rows carry their reason (plan §5.4)
            verdict = f"{verdict} — {r['notes']}"
        lines.append(
            f"| {r['cell_id']} | {r['variant']} | {r['n_train']} | {r['d']} "
            f"| {_f(r['committed_r2'])} | {_f(r['replay_delta'])} "
            f"| {_f(r['corrected_gcv_capped_r2'])} | {_f(r['corrected_inner_cv_r2'])} "
            f"| {_f(r['corrected_reduced_basis_r2'])} | {forced} | {verdict} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Synthetic smoke cells (--synthetic-spec: no HF, no staging; committed smoke)
# ---------------------------------------------------------------------------
def _folds_from_xy(X: np.ndarray, Y: np.ndarray, groups: np.ndarray, *, n_folds: int, seed: int):
    folds = fit825._cv_folds(np.asarray(groups), n_folds, seed)
    out: list[FoldData] = []
    for k in range(n_folds):
        te = folds == k
        tr = ~te
        if te.sum() == 0 or tr.sum() < 3:
            continue
        out.append(
            FoldData(
                fold_id=k,
                X_tr=X[tr],
                Y_tr=Y[tr],
                X_ev=X[te],
                Y_true=Y[te],
                groups_tr=np.asarray(groups)[tr],
            )
        )
    return out


def synthetic_cells() -> list[CellSpec]:
    """Two synthetic cells exercising every arm + tripwire + table writer:

    * ``syn_degenerate`` — n=24 < d=64 (the incident regime; committed value =
      the legacy-replay read computed at spec-build time, so the replay gate
      passes with delta 0 by construction).
    * ``syn_control``    — n=60 > d=8 (the n>d negative-control shape).
    """
    rng = np.random.default_rng(1887)

    def _make(n, d, sigma):
        X = rng.standard_normal((n, d))
        W = rng.standard_normal((d, 4)) @ rng.standard_normal((4, d)) / np.sqrt(4 * d)
        Y = X @ W + sigma * rng.standard_normal((n, d))
        groups = np.asarray([f"g{i}" for i in range(n)])
        return X.astype(np.float32), Y.astype(np.float32), groups

    Xd, Yd, gd = _make(24, 64, 2.0)
    Xc, Yc, gc = _make(60, 8, 1.0)

    def _committed(X, Y, groups, cfg):
        folds = _folds_from_xy(X, Y, groups, n_folds=5, seed=0)
        prev = _selector_pins(cfg)
        try:
            return float(
                _fit_folds(folds, lambda_selection=cfg["lambda_selection"], seed=0)["r2_pooled"]
            )
        finally:
            _restore_pins(prev)

    cells = [
        CellSpec(
            issue=0,
            cell_id="syn_degenerate",
            variant="synthetic",
            committed_r2=_committed(Xd, Yd, gd, LEGACY_UNGUARDED),
            published_claim_ref="synthetic smoke (no published claim)",
            store_rev="synthetic",
            load=lambda: _folds_from_xy(Xd, Yd, gd, n_folds=5, seed=0),
            compute_knn=True,
            committed_selector=dict(LEGACY_UNGUARDED),
        ),
        CellSpec(
            issue=0,
            cell_id="syn_control",
            variant="synthetic",
            committed_r2=_committed(Xc, Yc, gc, LEGACY_UNGUARDED),
            published_claim_ref="synthetic smoke (no published claim)",
            store_rev="synthetic",
            load=lambda: _folds_from_xy(Xc, Yc, gc, n_folds=5, seed=0),
            compute_knn=False,
            control=True,
            committed_selector=dict(LEGACY_UNGUARDED),
        ),
    ]
    return cells


# ---------------------------------------------------------------------------
# Per-issue adapters (thin: parents' own loaders — plan §3.5; PILOT-GATED).
# The REAL-store P0 pilot (post-review, orchestrator-detached) validates each
# adapter against ONE committed JSON per issue at full-consumed grain BEFORE
# the P2 battery (plan A6: on a mismatch, amend the adapter before P2).
# ---------------------------------------------------------------------------
def _headline_pos(payload: dict) -> tuple[int, int]:
    """(layer position, layer id) for the committed cell's headline read."""
    r2 = payload.get("r2_per_layer_obs")
    assert r2 is not None, "committed cell JSON lacks r2_per_layer_obs"
    n_layers = len(r2)
    if payload.get("frozen_layers"):
        fl = list(payload["frozen_layers"])  # 1310-style payloads name layers
        li = HEADLINE_LAYER if HEADLINE_LAYER in fl else fl[-1]
        return li, li
    # 1345-style payloads: full 28-layer axis, headline L19 by position.
    li = HEADLINE_LAYER if n_layers > HEADLINE_LAYER else n_layers - 1
    return li, li


def _resolve_1345_store(variant: str, format_key: str) -> tuple[str, str, str] | None:
    """(prefix, pinned revision, flat staging subdir) for a cell's turnstore
    stem — parent-format stems resolve at the parent turnstore pin (shared
    flat dir, staged once across variants); paired/slot stems at their
    variant's pinned prefix. None = un-refittable (no pinned HF source)."""
    if format_key in I1345_PARENT_FORMATS:
        return I1345_PARENT_STORE_PREFIX, I1345_PARENT_STORE_REV, "parent_turnstore"
    if variant in I1345_VARIANT_STORE_REVS:
        return (
            f"issue1345_framing/{variant}/analysis_tensors/turnstore",
            I1345_VARIANT_STORE_REVS[variant],
            f"{variant}_turnstore",
        )
    return None


def _allowlist_ref_1345(variant: str, payload: dict) -> str | None:
    """Short resume-key token naming the cell's row-identity source (None =
    full store). Rides the CellSpec store_rev so the resume predicate
    invalidates on an allowlist-pin change, not only a store-pin change."""
    if payload.get("cjk_exclusion"):
        return "payload:cjk_exclusion"
    if not payload.get("row_allowlist_applied"):
        return None
    if variant == "story_slot_ablation":
        return "git:slot_row_coverage.json"
    rev = I1345_PARENT_MATCHED_REV if variant == "base" else I1345_VARIANT_MATCHED_REVS[variant]
    return f"matched:{rev}"


def cells_1345(repo_root: Path, stage_root: Path, pilot: int) -> list[CellSpec]:
    """#1345 matched-row + variant + followup + slot cells (plan §3.5).

    Enumerated from the committed JSONs; stores staged per-stem from the
    PINNED prefixes in `_resolve_1345_store` (concern
    i1887-variant-store-staging); folds/rows: committed conv-grouped K=5
    seed 0 + the variant-resolved allowlists, reused unchanged. A cell whose
    store has no pinned HF source is enumerated with ``load=None`` and lands
    in the corrections table as `un-refittable — store not resolvable`
    (plan §5.4) — never a whole-audit failure. PILOT-GATED.
    """
    eval_dir = repo_root / "eval_results/issue_1345"
    # Realized 2026-07-30 layout: some variants carry cells at the dir top,
    # some under matched_row/ — enumerate BOTH per variant (plan §13 allows
    # adding cells discovered during enumeration; the slot cells live at
    # story_slot_ablation/, not the plan's provisional "slot_verdict").
    variants = (
        "assistant_named_story",
        "conversation_paired_stories",
        "conversation_paired_stories_assistant",
        "conversation_paired_stories_assistant_base",
        "onpolicy_assistant_story",
        "followup_cjk_excluded",
        "ladder_rungs",
        "story_slot_ablation",
    )
    search_dirs = [(eval_dir, "base")]
    for v in variants:
        search_dirs.append((eval_dir / v, v))
        search_dirs.append((eval_dir / v / "matched_row", v))
    specs: list[CellSpec] = []
    seen: set[str] = set()
    for d, variant in search_dirs:
        if not d.is_dir():
            continue
        for p in sorted(d.glob("cells_*.json")):
            payload = json.loads(p.read_text())
            cell_dict = payload.get("cell")
            if cell_dict is None:
                print(f"[audit][1345] SKIP {p.name}: no embedded cell dict (non-fit825 producer)")
                continue
            unit_id = f"{variant}__{p.stem.removeprefix('cells_')}"
            if unit_id in seen:
                print(f"[audit][1345] SKIP duplicate cell id {unit_id} at {p}")
                continue
            seen.add(unit_id)
            li, _ = _headline_pos(payload)
            committed = float(payload["r2_per_layer_obs"][li])
            common = dict(
                issue=1345,
                cell_id=unit_id,
                variant=variant,
                committed_r2=committed,
                published_claim_ref=str(p.relative_to(repo_root)) + f" @ L{li}",
                seed=int(payload.get("metadata", {}).get("seed", 0)),
                compute_knn="stories" in variant or "story" in str(cell_dict),
                committed_selector=dict(COMMITTED_SELECTOR_BY_ISSUE[1345]),
            )
            store = _resolve_1345_store(variant, cell_dict["format_key"])
            if store is None:
                stem = f"{cell_dict['model_key']}_{cell_dict['format_key']}_{cell_dict['track']}"
                specs.append(
                    CellSpec(
                        **common,
                        store_rev="unresolvable",
                        load=None,
                        notes=(
                            f"un-refittable — store not resolvable (stem {stem!r}: no pinned "
                            f"HF turnstore prefix for variant {variant!r}; plan §5.4)"
                        ),
                    )
                )
                continue
            prefix, rev, subdir = store
            allow_ref = _allowlist_ref_1345(variant, payload)
            specs.append(
                CellSpec(
                    **common,
                    store_rev=(rev if allow_ref is None else f"{rev}+allow:{allow_ref}"),
                    load=_loader_1345(
                        repo_root, stage_root, variant, cell_dict, payload, li, prefix, rev, subdir
                    ),
                )
            )
    if pilot:
        specs = specs[:pilot]
    assert specs, "no #1345 cells enumerated — run from a checkout with eval_results/issue_1345"
    return specs


def _stage_1345_stem(prefix: str, revision: str, stem: str, dest: Path) -> None:
    """Stage ONE stem's pt shards + sidecar JSONs from a PINNED prefix into a
    flat dir (scoped listing + retried per-file download; skip-if-present)."""
    from huggingface_hub import HfApi

    from explore_persona_space.orchestrate import hub

    files = hub.list_hf_files_under_path(
        HfApi(token=os.environ.get("HF_TOKEN")),
        HF_DATA_REPO,
        prefix,
        repo_type="dataset",
        revision=revision,
    )
    want = sorted(
        f
        for f in files
        if Path(f).name.startswith(f"{stem}_shard") and f.endswith((".pt", ".json"))
    )
    assert want, f"no shards for stem {stem!r} under {prefix}@{revision}"
    for f in want:
        target = dest / Path(f).name
        if not target.is_file():
            hub.stage_hub_file(HF_DATA_REPO, f, target, repo_type="dataset", revision=revision)


_1345_MATCHED_CACHE: dict = {}


def _matched_1345(stage_root: Path, variant: str) -> dict:
    """Stage + parse the PINNED matched_subsets.json ('base' = the parent's)."""
    if variant not in _1345_MATCHED_CACHE:
        from explore_persona_space.orchestrate import hub

        if variant == "base":
            path, rev = I1345_PARENT_MATCHED_PATH, I1345_PARENT_MATCHED_REV
        else:
            rev = I1345_VARIANT_MATCHED_REVS[variant]
            path = f"issue1345_framing/{variant}/inputs/matched_n/matched_subsets.json"
        target = stage_root / "issue1345" / variant / "matched_n" / "matched_subsets.json"
        if not target.is_file():
            hub.stage_hub_file(HF_DATA_REPO, path, target, repo_type="dataset", revision=rev)
        _1345_MATCHED_CACHE[variant] = json.loads(target.read_text())
    return _1345_MATCHED_CACHE[variant]


def _resolve_1345_allowlist(repo_root, stage_root, variant, cell_dict, payload):
    """Committed row identity per cell (concern i1887-variant-store-staging):

    - story_slot_ablation: the committed slot_row_coverage.json registered
      set (slot cells AND the chat matched comparator all fit on it —
      issue1345_slot_verdict.run_fits allowlist_fn).
    - ``_tf_on_companion_`` cells: the variant matched file's
      per_model_r4_pair op_companion_convs; ``_matched_`` cells: r4_convs.
    - full r1/r2 cells: shared_r1r2_convs (the variant's own matched file;
      the parent's for base cells).
    Fail-loud n parity against the committed payload's n_allowlist.
    """
    if not payload.get("row_allowlist_applied"):
        return None
    cid = cell_dict["cell_id"]
    if variant == "story_slot_ablation":
        cov = json.loads(
            (
                repo_root / "eval_results/issue_1345/story_slot_ablation/slot_row_coverage.json"
            ).read_text()
        )
        allowlist = cov["registered_conv_ids"]
    else:
        matched = _matched_1345(stage_root, variant)
        if "_tf_on_companion_" in cid:
            allowlist = matched["per_model_r4_pair"][cell_dict["model_key"]]["op_companion_convs"]
        elif "_matched_" in cid:
            allowlist = matched["per_model_r4_pair"][cell_dict["model_key"]]["r4_convs"]
        else:
            allowlist = matched["shared_r1r2_convs"]
    n_committed = payload.get("n_allowlist")
    assert allowlist and (n_committed is None or len(allowlist) == int(n_committed)), (
        f"{cid}: resolved allowlist n={len(allowlist or [])} != committed "
        f"n_allowlist={n_committed} — allowlist provenance drift"
    )
    return allowlist


def _loader_1345(repo_root, stage_root, variant, cell_dict, payload, li, prefix, rev, subdir):
    def _load() -> list[FoldData]:
        import issue1345_common as c1345
        import issue1345_fit_cells as f1345
        from issue1345_followup_cjk_excluded import filter_bundle_rows

        # Stage ONLY this cell's stem from its pinned prefix (parent-format
        # stems share one flat dir across variants; variant stems land under
        # their own subdir — see _resolve_1345_store).
        stem_dir = stage_root / "issue1345" / subdir
        stem_dir.mkdir(parents=True, exist_ok=True)
        stem = f"{cell_dict['model_key']}_{cell_dict['format_key']}_{cell_dict['track']}"
        _stage_1345_stem(prefix, rev, stem, stem_dir)
        # Bundle load keyed on the cell dict's OWN format_key (never the
        # env-gated c1345.REGIME_FORMAT registry, which lacks the variant
        # regimes r4/r4op/r4slot outside EPM_I1345_VARIANT runs).
        bundle = fit825._load_bundle_any(
            stem_dir,
            cell_dict["model_key"],
            cell_dict["format_key"],
            cell_dict["track"],
            wanted_keys=f1345.SLIM_KEYS,
        )
        expect_slots = len(c1345.SLOT_STORE_ORDER) if cell_dict.get("regime") == "r4slot" else 2
        c1345.assert_pt_bundle(
            bundle, expect_slots=expect_slots, expect_layers=fit825.EXPECTED_LAYERS
        )
        cjk = payload.get("cjk_exclusion")
        if cjk:
            # The committed cjk cells fit the FULL r3 store minus the
            # payload-recorded excluded story ids (self-contained provenance;
            # committed reference: rows 2108 -> 1983 after dropping 24 stories).
            bundle, digest = filter_bundle_rows(bundle, set(cjk["excluded_story_ids"]))
            assert digest["n_stories_dropped"] == int(cjk["n_excluded_stories"]), digest
            assert digest["n_rows_after"] == int(payload["metadata"]["n"]), (
                digest,
                payload["metadata"]["n"],
            )
        allowlist = _resolve_1345_allowlist(repo_root, stage_root, variant, cell_dict, payload)
        xy = fit825._apply_row_allowlist(
            fit825._cell_xy(bundle, cell_dict), allowlist, cell_dict["cell_id"]
        )
        X = xy["X"][:, li, :]
        Y = xy["Y"][:, li, :]
        return _folds_from_xy(X, Y, xy["conv_ids"], n_folds=int(payload.get("n_folds", 5)), seed=0)

    return _load


def cells_1310(repo_root: Path, stage_root: Path, pilot: int) -> list[CellSpec]:
    """#1310 per-persona cells (20: 2 models x (4 personas x {spanmean,lastpos}
    + swap + swapctrl_correct)); scenario-grouped K=5 seed-0 folds reused.
    Stores staged from the issue1310_char_map HF prefix via the parent loader.
    PILOT-GATED."""
    eval_dir = repo_root / "eval_results/issue_1310"
    specs: list[CellSpec] = []
    for p in sorted(eval_dir.glob("cells_*.json")):
        payload = json.loads(p.read_text())
        cell_id = payload.get("cell_id") or p.stem.removeprefix("cells_")
        li, _ = _headline_pos(payload)
        committed = float(payload["r2_per_layer_obs"][li])
        specs.append(
            CellSpec(
                issue=1310,
                cell_id=cell_id,
                variant="xpersona",
                committed_r2=committed,
                published_claim_ref=str(p.relative_to(repo_root)) + f" @ L{li}",
                store_rev=I1310_STORE_REV,
                load=_loader_1310(stage_root, cell_id, payload, li),
                seed=int(payload.get("metadata", {}).get("seed", 0)),
                committed_selector=dict(COMMITTED_SELECTOR_BY_ISSUE[1310]),
            )
        )
    if pilot:
        specs = specs[:pilot]
    assert specs, "no #1310 cells enumerated — run from a checkout with eval_results/issue_1310"
    return specs


_1310_STORE_CACHE: dict = {}


def _loader_1310(stage_root, cell_id, payload, li):
    def _load() -> list[FoldData]:
        import issue1310_fit as f1310
        from explore_persona_space.orchestrate import hub

        store_root = stage_root / "issue1310" / I1310_STORE_PREFIX
        if not store_root.is_dir() or not any(store_root.iterdir()):
            hub.stage_hub_prefix(
                HF_DATA_REPO,
                I1310_STORE_PREFIX,
                stage_root / "issue1310",
                repo_type="dataset",
                revision=I1310_STORE_REV,
            )
        model_kind = "base" if cell_id.startswith("base") else "instruct"
        if model_kind not in _1310_STORE_CACHE:
            _1310_STORE_CACHE[model_kind] = f1310.load_model_store(store_root, model_kind)
        store = _1310_STORE_CACHE[model_kind]
        rest = cell_id.removeprefix(f"{model_kind}_")
        seed = int(payload.get("metadata", {}).get("seed", 0))
        if rest in ("swap", "swapctrl_correct"):
            rows, partners = f1310.swap_derangement(
                store["group_ids"], store["char_ids"], store["turn_indices"], seed
            )
            X = store["arrays"]["x_spanmean"][rows][:, li, :]
            y_idx = partners if rest == "swap" else rows
            Y = store["arrays"]["y"][y_idx][:, li, :]
            groups = store["group_ids"][rows]
        else:
            persona = rest.removesuffix("_lastpos")
            x_key = "x_last" if rest.endswith("_lastpos") else "x_spanmean"
            xy = f1310.within_xy(store, persona, x_key)
            X = xy["X"][:, li, :]
            Y = xy["Y"][:, li, :]
            groups = xy["group_ids"]
        return _folds_from_xy(X, Y, groups, n_folds=int(payload.get("n_folds", 5)), seed=seed)

    return _load


def cells_1639(repo_root: Path, stage_root: Path, pilot: int) -> list[CellSpec]:
    """#1639 tier-1.5 naive transfer rungs (cross-persona pairs) on the same
    xpersona stores; committed = capped GCV 0.9 (replay coincides with arm 2).
    PILOT-GATED."""
    tier15 = repo_root / "eval_results/issue_1310/xpersona_similarity/tier15_intercept_refit"
    results = json.loads((tier15 / "results.json").read_text())
    specs: list[CellSpec] = []
    for key, entry in sorted(results.get("directions", {}).items()):
        naive = entry.get("naive") or {}
        committed = naive.get("r2_foldmean")
        specs.append(
            CellSpec(
                issue=1639,
                cell_id=f"tier15_naive__{key}",
                variant="tier15_intercept_refit",
                committed_r2=(None if committed is None else float(committed)),
                published_claim_ref=(
                    str((tier15 / "results.json").relative_to(repo_root))
                    + f" directions[{key}].naive.r2_foldmean"
                ),
                store_rev=I1310_STORE_REV,
                load=_loader_1639(stage_root, key),
                committed_selector=dict(COMMITTED_SELECTOR_BY_ISSUE[1639]),
            )
        )
    if pilot:
        specs = specs[:pilot]
    assert specs, "no #1639 pairs enumerated — tier15 results.json missing"
    return specs


def _loader_1639(stage_root, key):
    def _load() -> list[FoldData]:
        import issue1310_xpersona_similarity as v1
        import issue1639_tier15_intercept_refit as t15
        from explore_persona_space.orchestrate import hub

        store_root = stage_root / "issue1310" / I1310_STORE_PREFIX
        if not store_root.is_dir() or not any(store_root.iterdir()):
            hub.stage_hub_prefix(
                HF_DATA_REPO,
                I1310_STORE_PREFIX,
                stage_root / "issue1310",
                repo_type="dataset",
                revision=I1310_STORE_REV,
            )
        # Realized tier15 direction-key format (read from the committed
        # results.json, 2026-07-30): "<model>.<src>-><tgt>" e.g.
        # "base.Dana->HELIOS".
        model_src, _, tgt = key.partition("->")
        model, _, src = model_src.partition(".")
        assert model and src and tgt, f"unparseable tier15 direction key {key!r}"
        arrays = v1.load_persona_arrays(store_root, model)
        li = t15.L19
        Xs = arrays[src]["X"][:, li, :].astype(np.float64)
        Ys = arrays[src]["Y"][:, li, :].astype(np.float64)
        Xt = arrays[tgt]["X"][:, li, :].astype(np.float64)
        Yt = arrays[tgt]["Y"][:, li, :].astype(np.float64)
        folds = arrays[tgt]["folds"]
        src_groups = arrays[src].get("groups", np.arange(Xs.shape[0]))
        out: list[FoldData] = []
        for k in range(v1.N_FOLDS):
            te = folds == k
            tr = folds != k
            if te.sum() == 0 or tr.sum() < 3:
                continue
            out.append(
                FoldData(
                    fold_id=k,
                    X_tr=Xs[tr],
                    Y_tr=Ys[tr],
                    X_ev=Xt[te],
                    Y_true=Yt[te],
                    groups_tr=np.asarray(src_groups)[tr],
                )
            )
        return out

    return _load


def cells_825_control(repo_root: Path, stage_root: Path, pilot: int) -> list[CellSpec]:
    """n > d negative-control leg: #825 full-corpus S-track cells (realized
    n=5,000 rows -> n_train_min=4,000 > d=3,584 at 5 folds; the plan's
    "n=4,724" figure conflated the #1345 row-allowlist count — structurally
    immune per the task body either way); expected |delta| <= 0.05 vs
    committed on every corrected arm. PILOT-GATED."""
    import issue825_selector_audit as sa  # ALL_CELLS: (model, fmt, track, si, ti)

    # Banked committed reference (battery module docstring / settle outputs):
    banked = {"S_instruct_chat": 0.6730940896676356}
    settle = repo_root / "eval_results/issue_825/trackm_settle"
    if settle.is_dir():
        for p in settle.glob("*.json"):
            try:
                payload = json.loads(p.read_text())
            except json.JSONDecodeError:
                continue
            for cid, rec in (payload.get("cells") or {}).items():
                un = (rec.get("unguarded_ridge") or {}).get("19")
                if un is not None and cid in sa.ALL_CELLS:
                    banked[cid] = float(un)
    cids = [c for c in ("S_instruct_chat", "S_pretrained_chat") if c in sa.ALL_CELLS]
    if pilot:
        cids = cids[:pilot]
    assert cids, "no #825 control cells enumerated"
    # r3 crash-fix: materialize the parent layout BEFORE run_units — the r2
    # adapter staged nothing for this leg and the P0 pilot died at
    # load_cell_xy's direct cwd-relative np.load (FileNotFoundError).
    _stage_825_parent_layout(cids)
    specs: list[CellSpec] = []
    for cid in cids:
        specs.append(
            CellSpec(
                issue=825,
                cell_id=f"control__{cid}",
                variant="n_gt_d_control",
                committed_r2=banked.get(cid),
                published_claim_ref=(
                    "banked S1@5000 unguarded L19 (issue825_trackm_settle_battery docstring / "
                    "trackm_settle outputs)"
                ),
                store_rev="issue825 turnstore (battery staging)",
                load=_loader_825(cid),
                compute_knn=False,
                control=True,
                committed_selector=dict(COMMITTED_SELECTOR_BY_ISSUE[825]),
            )
        )
    return specs


def _stage_825_parent_layout(cids: list[str]) -> None:
    """Materialize the parent-layout S-track npz spans the #825 consumer opens.

    ``issue825_trackm_settle_battery.load_cell_xy`` does a DIRECT cwd-relative
    ``np.load(S_STORE_DIR / f"{model}_{fmt}_s.npz")`` with NO
    hub-download-on-miss, so the audit must create that exact layout before
    ``run_units`` (artifact-reuse check (h)(iv) — staged-layout consumer-open).
    Staging goes through the parent line's OWN helper ``cm.extract_stem``
    (default pin ``cm.HF_REV`` = deb7a4523b… — the plan §10 reuse row; verified
    2026-07-30 via scoped ``list_repo_tree`` at that pin:
    ``issue825_userbase_map/analysis_tensors`` carries instruct_chat_s /
    pretrained_chat_s x 10 .pt shards each). ``extract_stem`` drops each
    ~2.1 GB shard immediately after extraction, so disk peak is ~3 GB per stem
    (re-downloadable cache; idempotent on the cached npz). Ends with the
    consumer-open probe: stat + lazy ``np.load`` key check of every staged
    span at the exact path the consumer opens; raises RuntimeError on a miss.
    """
    import issue825_crossmodel_map_transfer as cm
    import issue825_selector_audit as sa
    import issue825_trackm_settle_battery as bat

    dl_dir = Path(bat.S_STORE_DIR)
    stems: list[str] = []
    for cid in cids:
        model, fmt, track, _si, _ti = sa.ALL_CELLS[cid]
        assert track == "s", f"{cid}: only S-track #825 control cells are wired (track={track!r})"
        stems.append(f"{model}_{fmt}_s")
    for stem in stems:
        cm.extract_stem(stem, dl_dir)
    # (h)(iv) consumer-open probe (cheap: np.load on an npz is lazy — it reads
    # only the zip directory, never the 430 MB payload).
    for stem in stems:
        span = dl_dir / f"{stem}.npz"
        if not span.is_file():
            raise RuntimeError(
                f"[audit][825] staged-layout consumer-open miss: {span} absent after "
                f"staging — load_cell_xy opens this cwd-relative parent-layout path "
                f"directly (no hub fallback); cwd={Path.cwd()}"
            )
        with np.load(span, allow_pickle=False) as d:
            missing = {"slots", "profiles", "conv_ids"} - set(d.files)
        if missing:
            raise RuntimeError(
                f"[audit][825] staged span {span} lacks keys {sorted(missing)} — "
                "not a valid 4-layer S-track turnstore npz"
            )


def _loader_825(cid):
    def _load() -> list[FoldData]:
        import issue825_selector_audit as sa
        import issue825_trackm_settle_battery as bat

        model, fmt, track, si, ti = sa.ALL_CELLS[cid]
        xy = bat.load_cell_xy(model, fmt, track, si, ti)
        pos = bat.WANT_LAYERS.index(HEADLINE_LAYER)
        X = xy["X"][:, pos, :]
        Y = xy["Y"][:, pos, :]
        return _folds_from_xy(X, Y, xy["conv_ids"], n_folds=bat.N_FOLDS, seed=bat.FOLD_SEED)

    return _load


ADAPTERS = {1345: cells_1345, 1310: cells_1310, 1639: cells_1639, 825: cells_825_control}
# Measured via scoped list_repo_tree at the pinned revisions (2026-07-30 r2
# probe): #1345 = 92.17 GB parent turnstore + 5.56/4.92/7.01 GB variant stores
# ~= 110 GB (supersedes the plan §8 10-40 GB estimate); #1310/#1639 share the
# 8.04 GB store_onpolicy prefix. #825: ~21.4 GB TRANSFER per stem (10 x
# ~2.1 GB shards at deb7a452, r3 probe) but cm.extract_stem removes each shard
# after extraction -> disk peak ~3 GB + 0.43 GB npz per stem; 15.0 is ample.
STAGING_GB = {1345: 110.0, 1310: 10.0, 1639: 10.0, 825: 15.0}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--issue", type=int, choices=sorted(ADAPTERS), default=None)
    ap.add_argument(
        "--stage-root",
        type=Path,
        default=Path(f"/mnt/eps-data/{os.environ.get('USER', 'user')}/issue1887_lambda_audit"),
    )
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument(
        "--pilot-cells", type=int, default=0, help="limit to the first N cells (0 = all)"
    )
    ap.add_argument("--arms", default=",".join(ARMS), help="comma list of arms to run")
    ap.add_argument(
        "--synthetic-spec",
        default=None,
        help="run the built-in synthetic smoke cells (value ignored; no HF staging)",
    )
    ap.add_argument(
        "--table-only",
        action="store_true",
        help="rebuild the corrections table from existing cell checkpoints",
    )
    ap.add_argument(
        "--repo-root",
        type=Path,
        default=_REPO_ROOT,
        help="checkout carrying the committed eval_results/ (enumeration source)",
    )
    args = ap.parse_args()

    arms = tuple(a.strip() for a in args.arms.split(",") if a.strip())
    unknown = set(arms) - set(ARMS)
    assert not unknown, f"unknown arms {sorted(unknown)} (want subset of {ARMS})"

    if args.synthetic_spec is not None:
        out_dir = args.out or Path("/tmp/i1887_synthetic_audit")
        cells = synthetic_cells()
    else:
        assert args.issue is not None, "--issue is required unless --synthetic-spec"
        out_dir = args.out or (
            args.repo_root / f"eval_results/issue_{args.issue}/lambda_audit_1887"
        )
        if not args.table_only:
            assert_stage_headroom(args.stage_root, STAGING_GB[args.issue])
        cells = ADAPTERS[args.issue](args.repo_root, args.stage_root, args.pilot_cells)

    print(
        f"[audit] {len(cells)} cells x {len(arms)} arms -> {out_dir} "
        f"(pilot={args.pilot_cells or 'off'})",
        flush=True,
    )
    if not args.table_only:
        run_units(cells, arms, out_dir)
    table = build_corrections_table(cells, out_dir)
    # DONE sentinel (plan §8 phase_outputs), written atomically after the last cell.
    _write_json_atomic(
        out_dir / "DONE.json",
        {
            "issue": (0 if args.synthetic_spec is not None else args.issue),
            "n_cells": len(cells),
            "arms": list(arms),
            "replay_gate": table["replay_gate"],
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
