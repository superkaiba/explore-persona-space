#!/usr/bin/env python3
"""#1739 Result-2 v2: P-A vs P-B readout protocols over the FULL OOD rung grid.

Supersedes the r2fair scoring round (`issue1739_result2fair_score.py`) per the
2026-08-06 r2v2 dispatch (task #1739, epm:progress v639): all MLP arms removed
(five-arm linear-map roster), sycophancy's five REAL OOD rungs + evil's full
mhj/pair/tom-gibbs rungs join the eval grid, and TWO readout protocols are
scored from the SAME generation/judging/capture:

- **P-A** (the r2fair protocol): the label-consuming readouts train on the
  union of ONE trait-eliciting dataset (the behavior's train budget cell at
  the committed plotted slice) + the judged WildChat train split. Every OOD
  rung is fully unseen.
- **P-B** (new): the readouts train on the union of an 80% GROUP-level slice
  of EVERY trait-eliciting dataset (train + each judged eval-rung dataset)
  EXCEPT one dataset held out whole, + the judged WildChat train split. The
  held-out dataset is the primary eval; the included datasets' 20%
  remainders are scored as secondary `heldin:<name>` rungs.

P-B's pool assembly is PARAMETERIZED for the Result-5 leave-one-dataset-out
(LODO) ablations: :func:`assemble_readout_pool` takes the dataset roster +
`holdout` + `include` + `train_frac`, group-membership sides are a pure
hash of (dataset, group_key, seed) — independent of the holdout choice and of
which other datasets are included — so a LODO sweep is a loop over `holdout`
and a dataset-subset ablation is a loop over `include`, never a rewrite.
Held-out folds are GROUP-level throughout (.claude/rules/ood-generalization-
folds.md; a pointwise split across rows from one source leaks).

Map + whitening are IDENTICAL across the two protocols (the committed
`result2_trait_aug` ADD-condition linear recipe, re-fit deterministically in
process — the reviewed r2fair reuse), so the readout training set is the ONLY
varying factor. Ridge readouts ride the batched (source x fold) job pool
(`arms.RidgeJob` + `_solve_ridge_groups`; one Gram+eigh per job shared across
all stacked per-layer targets — vectorize-many-cell-fits.md), with per-fit
selected-lambda diagnostics captured via `fits.capture_selected_lambdas`.

Evil's OOD DV is gated: a DV whose null-dv fraction exceeds
`--ood-dv-max-null-frac` (the 2026-08-06 judge-refusal missingness, 465/2954
contexts) is REFUSED with a recorded reason — never silently scored partial.
A resolved DV still carrying `split: full` rows is rewritten full->eval into
a recorded working copy (the syco-OOD convention; OOD rungs are pure eval).

VARIANT SCOPE: context_end ONLY (recorded deviation from the prefix+context
both-arms rule; the r2v2 dispatch note carries the measured degeneracy basis:
these corpora have a constant empty prefix — per-dim SD 0.000000).

Safety rails inherited from the fair round: no judge module may be imported,
DV inputs sha-verified after scoring, git-tracked outputs refused.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace


def _ensure_repo_root_on_syspath() -> Path:
    root = Path(__file__).resolve().parents[1]
    sentinel = root / "scripts" / "issue1739_result2fair_score.py"
    if not sentinel.exists():
        raise RuntimeError(f"repo-root resolution failed: {sentinel} missing")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


_REPO_ROOT = _ensure_repo_root_on_syspath()

BEHAVIORS = ("evil", "sycophancy", "hallucination")
WC_RUNG = "wildchat_rung"
PV_RUNG = "pvsynth"

# r2v2 roster: five arms, all-linear map, ALL MLP arms removed (dispatch v639).
ROSTER = (
    "arm1_ctx_e1",
    "arm4_ridge_ctx",
    "arm6_map_proj_e1",
    "arm7_map_ridge_pred",
    "arm11_oracle_proj",
)
LABEL_CONSUMING = ("arm4_ridge_ctx", "arm7_map_ridge_pred")

# NEW OOD stores/DVs (the r2v2 generation+capture round). Store paths are
# RELATIVE to --ood-store-root (a verbatim HF-prefix mirror of
# issue1739_ctxmap/...); tomgibbs is ONE dataset captured in two store halves
# (global 0-based shard indices — the halves cannot share a dir) that the
# loader concatenates. hallucination has no new stores: its OOD datasets are
# the committed train-store eval-split rungs (nqopen, simpleqa).
OOD_SPECS: dict[str, dict] = {
    "evil": {
        "stores": (
            "evil_ood_full/store/mhj",
            "evil_ood_full/store/pair",
            "evil_ood_full/store/tomgibbs_p0",
            "evil_ood_full/store/tomgibbs_p1",
        ),
        "expected_rungs": {"evil_mhj", "evil_pair", "evil_tomgibbs"},
    },
    "sycophancy": {
        "stores": ("syco_ood/store",),
        "expected_rungs": {"sycoans", "sycoays", "sycofb", "sycomim", "sycomwe"},
    },
}

DEFAULT_OUT_ROOT = Path("eval_results/issue_1739/r2v2_fits")
DEFAULT_MAIN_ROOT = Path("eval_results/issue_1739")
DEFAULT_TENSORS_ROOT = Path("analysis_tensors/issue_1739")
DEFAULT_STORE_ROOT = Path("data/issue_1739/hf_dl")
HIDDEN_D_PIN = 3584  # Qwen-2.5-7B hidden dim — the well-posedness denominator


def _log(msg: str) -> None:
    print(f"[r2v2 {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _log_rss(tag: str) -> None:
    """Phase-boundary RSS breadcrumb (the 128 GB-cgroup OOM diagnosability line).

    The cpu-bigmem container is cgroup-capped at 128 GB decimal (119.2 GiB)
    while ``free`` reports the HOST's 251 GB — resident-set arithmetic against
    the cgroup cap is the binding constraint (pilot r1 died SIGKILL here).
    """
    import resource

    try:
        cur_gib = (
            int(Path("/proc/self/statm").read_text().split()[1])
            * os.sysconf("SC_PAGE_SIZE")
            / 2**30
        )
        peak_gib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
        _log(f"[rss {tag}] current={cur_gib:.1f} GiB peak={peak_gib:.1f} GiB")
    except (OSError, ValueError, IndexError):
        pass  # /proc absent (non-linux test host) — breadcrumb only, never load-bearing


def _drop_unused_arrays(loaded, tbl_ood, variant: str) -> None:
    """Free activation arrays the context_end-scoped fits never read.

    Every table's OTHER variant (prefix_end under the default scope) and the
    u_store's other-variant kind are dead weight (~9 GiB at the sycophancy
    shape) that sat resident through the map-fit peak in pilot r1's OOM.
    In-place mutation of the shared dicts (callee `del` frees nothing —
    the caller's dict entry is the binding).
    """
    tables = [loaded.tbl, loaded.tbl_wc, loaded.tbl_ev] + ([tbl_ood] if tbl_ood else [])
    for t in tables:
        for v in [k for k in t.z_by_variant if k != variant]:
            del t.z_by_variant[v]
    for key in [k for k in loaded.u_arrays if k[0] not in (variant, "t1")]:
        del loaded.u_arrays[key]


def _release_table_arrays(loaded, tbl_ood, tbl_pv) -> None:
    """Free every raw fp16 activation block once the merged fp64 table exists.

    dv / groups / ctx_order / row_rungs stay (the fits read only those after
    the merge). Saves ~9-15 GiB of steady-state residency under the cgroup cap.
    """
    tables = [loaded.tbl, loaded.tbl_wc, loaded.tbl_ev, tbl_pv] + ([tbl_ood] if tbl_ood else [])
    for t in tables:
        t.z_by_variant.clear()
        t.z_ans = None


# ---------------------------------------------------------------------------
# P-B readout-pool assembly (LODO-parameterized — the Result-5 seam)
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class DatasetSpec:
    """One trait-eliciting dataset: rows into the MERGED labeled table."""

    name: str
    rows: object  # (n,) int64 indices into the merged table
    groups: object  # (n,) str group keys (GROUP-level splits are load-bearing)


@dataclasses.dataclass(frozen=True)
class ReadoutPool:
    """Realized P-B pool: per-dataset train rows + held-in eval remainders."""

    holdout: str | None
    train_frac: float
    seed: int
    train_rows: dict[str, object]  # dataset name -> (k,) int64 merged-table rows
    heldin_eval_rows: dict[str, object]  # dataset name -> (m,) int64 rows (may be empty)


def _group_side_train(dataset: str, group: str, seed: int, train_frac: float) -> bool:
    """GROUP-level split side, a pure hash of (dataset, group, seed).

    Deliberately independent of the holdout choice AND of which other
    datasets are in the roster, so LODO folds share identical within-dataset
    splits across holdouts and Result-5 subset ablations reuse them verbatim.
    """
    h = int.from_bytes(hashlib.sha1(f"r2v2|{dataset}|{group}|{seed}".encode()).digest()[:8], "big")
    return (h % 10_000) < int(round(train_frac * 10_000))


def assemble_readout_pool(
    datasets: list[DatasetSpec],
    *,
    holdout: str | None,
    train_frac: float = 0.8,
    seed: int = 0,
    include: list[str] | None = None,
) -> ReadoutPool:
    """P-B pool: `train_frac` of every included dataset, `holdout` excluded whole.

    LODO = loop over ``holdout``; subset ablations = ``include`` (defaults to
    every dataset). Splits are GROUP-level via :func:`_group_side_train`. A
    dataset whose train side realizes EMPTY raises (cannot train on nothing);
    an empty held-in eval side is legal (recorded, scored as a min_n skip).
    """
    import numpy as np

    names = [d.name for d in datasets]
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate dataset names: {names}")
    if include is not None:
        unknown = sorted(set(include) - set(names))
        if unknown:
            raise ValueError(f"include names {unknown} not in roster {sorted(names)}")
        datasets = [d for d in datasets if d.name in include]
    if holdout is not None and holdout not in {d.name for d in datasets}:
        raise ValueError(f"holdout {holdout!r} not in included roster")
    if not 0.0 < train_frac < 1.0:
        raise ValueError(f"train_frac out of (0,1): {train_frac}")
    train_rows: dict[str, object] = {}
    heldin: dict[str, object] = {}
    for d in datasets:
        if d.name == holdout:
            continue
        rows = np.asarray(d.rows, dtype=np.int64)
        groups = np.asarray(d.groups)
        if rows.shape != groups.shape:
            raise ValueError(f"{d.name}: rows/groups shape mismatch {rows.shape}/{groups.shape}")
        side = np.array(
            [_group_side_train(d.name, str(g), seed, train_frac) for g in groups], dtype=bool
        )
        if not side.any():
            raise RuntimeError(
                f"{d.name}: group-level {train_frac:.0%} draw realized EMPTY train side "
                f"({len(np.unique(groups))} groups) — cannot train; widen train_frac or seed"
            )
        train_rows[d.name] = rows[side]
        heldin[d.name] = rows[~side]
    return ReadoutPool(
        holdout=holdout,
        train_frac=float(train_frac),
        seed=int(seed),
        train_rows=train_rows,
        heldin_eval_rows=heldin,
    )


# ---------------------------------------------------------------------------
# OOD DV gate + store loading
# ---------------------------------------------------------------------------


def _prepare_ood_dv(
    src: Path, workdir: Path, behavior: str, *, max_null_frac: float
) -> tuple[Path, dict]:
    """Gate + normalize an OOD DV for the fit: fail loud, never a partial grid.

    (1) REFUSES a DV whose null-dv row fraction exceeds ``max_null_frac`` —
    the evil judge-refusal missingness class (a severity-correlated drop
    would bias every rho; the DV is being remediated by a separate agent).
    (2) All rows ``split: eval`` -> consumed as-is (the syco-OOD convention).
    All rows ``split: full`` -> rewritten full->eval into a recorded working
    copy (OOD rungs are pure eval rungs). Mixed splits refuse.
    """
    from scripts.issue1739_wcrung_arms import _sha256

    if not src.exists():
        raise FileNotFoundError(f"[{behavior}] OOD DV absent: {src}")
    payload = json.loads(src.read_text())
    rows = payload["rows"]
    n = len(rows)
    n_null = sum(1 for r in rows if r.get("dv") is None)
    frac = n_null / max(n, 1)
    if frac > max_null_frac:
        raise RuntimeError(
            f"[{behavior}] OOD DV UNRESOLVED: {n_null}/{n} rows ({frac:.1%}) have no valid "
            f"dv (> {max_null_frac:.1%} floor) — refusing to fit a silently-partial grid; "
            "awaiting the DV re-score (judge-refusal remediation)"
        )
    splits = {r.get("split") for r in rows}
    note = {
        "src": str(src),
        "src_sha256": _sha256(src),
        "n_rows": n,
        "n_null_dv": n_null,
        "null_frac": round(frac, 4),
    }
    if splits == {"eval"}:
        note["split_rewrite"] = "none — DV already split=eval (syco-OOD convention)"
        return src, note
    if splits == {"full"}:
        for r in rows:
            r["split"] = "eval"
        workdir.mkdir(parents=True, exist_ok=True)
        out = workdir / f"{behavior}_ood_labeling_eval.json"
        out.write_text(json.dumps(payload, indent=1))
        note["split_rewrite"] = f"full->eval working copy -> {out} (OOD rungs are pure eval)"
        return out, note
    raise RuntimeError(f"[{behavior}] OOD DV carries mixed splits {sorted(map(str, splits))}")


def load_ood_table(args, behavior: str, layers: list[int], dim: int, shas: dict):
    """Load + concatenate the behavior's NEW OOD stores against the gated DV.

    Returns ``(table, prep_note)`` — ``(None, {...})`` when the behavior has
    no new OOD stores (hallucination: its OOD datasets are the committed
    train-store eval-split rungs already in ``loaded.tbl_ev``).
    """
    import numpy as np

    from scripts.issue1739_fits import _load_labeled

    spec = OOD_SPECS.get(behavior)
    if spec is None:
        return None, {"note": "no new OOD stores — eval-split rungs only"}
    dv_src = args.evil_ood_dv if behavior == "evil" else args.syco_ood_dv
    dv_path, note = _prepare_ood_dv(
        Path(dv_src), args.out_root / "_dv_work", behavior, max_null_frac=args.ood_dv_max_null_frac
    )
    shas[str(dv_path)] = note["src_sha256"]
    tables = []
    for rel in spec["stores"]:
        store = args.ood_store_root / rel
        if not store.exists():
            raise FileNotFoundError(f"[{behavior}] OOD store absent: {store}")
        t = _load_labeled(store, dv_path, layers, config="config_b", need_rollout_rows=False)
        if t.z_ans.shape[-1] != dim:
            raise RuntimeError(f"[{behavior}] {rel}: hidden dim {t.z_ans.shape[-1]} != {dim}")
        tables.append(t)
        _log(f"[{behavior}] OOD store {rel}: {len(t.ctx_order)} contexts, rungs={t.rungs}")
    # cross-store context disjointness (tomgibbs halves partition contexts)
    seen: set[str] = set()
    for rel, t in zip(spec["stores"], tables, strict=True):
        ids = set(map(str, t.ctx_order))
        dup = seen & ids
        if dup:
            raise RuntimeError(f"[{behavior}] {rel}: {len(dup)} duplicate contexts across stores")
        seen |= ids
    merged = SimpleNamespace(
        z_by_variant={
            v: np.concatenate([t.z_by_variant[v] for t in tables], axis=1)
            for v in tables[0].z_by_variant
        },
        z_ans=np.concatenate([t.z_ans for t in tables], axis=1),
        dv=np.concatenate([np.asarray(t.dv, dtype=np.float64) for t in tables]),
        groups=[g for t in tables for g in t.groups],
        ctx_order=[c for t in tables for c in t.ctx_order],
        row_rungs=[r for t in tables for r in t.row_rungs],
        rungs=sorted({r for t in tables for r in t.rungs}),
    )
    realized = set(merged.rungs)
    if realized != set(spec["expected_rungs"]):
        raise RuntimeError(
            f"[{behavior}] OOD rung mismatch: realized {sorted(realized)} != "
            f"expected {sorted(spec['expected_rungs'])}"
        )
    n_dv_rows = sum(1 for _ in json.loads(Path(dv_path).read_text())["rows"])
    note["n_contexts_joined"] = len(merged.ctx_order)
    note["n_dv_rows"] = n_dv_rows
    return merged, note


# ---------------------------------------------------------------------------
# merged-table construction + per-fit helpers
# ---------------------------------------------------------------------------


def _whiten_concat_blocks(blocks, wh):
    """Whiten fp16 (Ly, n_i, d) blocks into ONE preallocated fp64 (Ly, N, d).

    Per-layer per-block fill (the `apply_whitening` chunking discipline) so
    the transient stays one layer of one block, never a second whole-array
    copy of the ~18 GB merged table.
    """
    import numpy as np

    ly, d = blocks[0].shape[0], blocks[0].shape[2]
    n_all = sum(b.shape[1] for b in blocks)
    out = np.empty((ly, n_all, d), dtype=np.float64)
    for li in range(ly):
        at = 0
        for b in blocks:
            n_b = b.shape[1]
            xl = np.asarray(b[li], dtype=np.float64) - wh.mu[li][None, :]
            np.matmul(xl, wh.w[li], out=out[li, at : at + n_b])
            del xl
            at += n_b
    return out


def _multi_pool_zscored_dv(dv_m, pools: list) -> object:
    """Merged-DV copy, TRAINING targets z-scored per source pool.

    The jobd mixed-construct fix generalized to N pools: each pool's selected
    rows are z-scored by that pool's own stats so heterogenous DV constructs
    enter the ridge loss on a common scale. Rows outside every pool are
    untouched (they never enter a fit). sd == 0 degrades to a centered
    constant column.
    """
    import numpy as np

    dv = np.asarray(dv_m, dtype=np.float64).copy()
    for rows in pools:
        rows = np.asarray(rows, dtype=np.int64)
        if rows.size == 0:
            continue
        m = float(dv[rows].mean())
        s = float(dv[rows].std())
        dv[rows] = (dv[rows] - m) / (s if s > 0 else 1.0)
    return dv


def _assert_well_posed(n_train: int, dim: int, label: str) -> None:
    """Refuse an estimator-degenerate ridge fit (n_train < d; #1701/#1887)."""
    if n_train < dim:
        raise RuntimeError(
            f"{label}: n_train {n_train} < d {dim} — held-out R^2/rho in this regime is "
            "estimator-degenerate; refusing the fit (no deliberate under-determined "
            "justification is registered for r2v2)"
        )


def _leakage_assert(readout_ids: set, eval_sets: dict[str, set], label: str) -> dict:
    """HARD disjointness asserts: readout-train contexts in NO eval setting."""
    report = {"n_readout_train_contexts": len(readout_ids)}
    for name, ids in eval_sets.items():
        inter = readout_ids & ids
        assert not inter, f"LEAKAGE [{label}]: {len(inter)} readout-train contexts in {name}"
        report[f"n_eval_{name}"] = len(ids)
    report["asserts"] = f"[{label}] readout ctx ids disjoint from every eval setting (passed)"
    return report


# ---------------------------------------------------------------------------
# per-behavior scoring
# ---------------------------------------------------------------------------


def fit_linear_add_map(args, loaded, variant: str, layers: list[int]):
    """The r2fair ADD-condition LINEAR map/whitening recipe with staged frees.

    Same pool composition + seed + reviewed compose/fit path as
    ``issue1739_result2fair_score.fit_add_maps`` restricted to kind=linear
    (map weights are never persisted, so the deterministic re-fit IS the
    reuse). Deltas are pure memory hygiene for the 128 GB cgroup (pilot r1
    OOM): the staged u_store arrays are freed the moment the pool is built,
    and each fp16 pool array is freed as soon as its fp64 whitened copy
    exists — peak drops ~40 GiB with bit-identical outputs.
    """
    from explore_persona_space.experiments.issue_1739 import fits
    from scripts.issue1739_fits import _fit_map
    from scripts.issue1739_jobd_r2aug import _fitmap_ns, build_pool

    x, y, u_label, n_u, pool_meta = build_pool(args, loaded, variant, layers, "add")
    loaded.u_arrays.clear()  # U pool consumed; free before the fp64 copies land
    _log_rss("map-pool-built")
    t0 = time.time()
    wh = fits.fit_whitening(x, device=args.device, seed=args.seed)
    x_w = fits.apply_whitening(x, wh)
    del x
    y_w = fits.apply_whitening(y, wh)
    del y
    wh_s = round(time.time() - t0, 1)
    _log_rss("map-pool-whitened")
    t1 = time.time()
    mapfit = _fit_map(_fitmap_ns(args), x_w, y_w)
    del x_w, y_w
    diag = {
        **mapfit.diagnostics,
        "map_kind": "linear",
        "map_source": "refit",
        "map_fit_s": round(time.time() - t1, 1),
        "whitening_fit_s": wh_s,
        "n_u": int(n_u),
        "u_pool_label": u_label,
        **pool_meta,
    }
    _log(f"[map] linear ADD map fit: whitening {wh_s}s, map {diag['map_fit_s']}s")
    return wh, mapfit, diag, u_label, n_u


def run_behavior(args, behavior: str, layers: list[int]) -> dict:
    import numpy as np

    from explore_persona_space.experiments.issue_1739 import arms, fits
    from scripts.issue1739_fits import _eval_rung_reconstruction
    from scripts.issue1739_jobd_r2aug import (
        LMAX,
        _free_cuda,
        committed_frozen,
        load_behavior,
        per_layer_rows_for,
        transfer_rows_for,
    )
    from scripts.issue1739_result2fair_score import (
        _wc_eval_mask,
        _wc_fold_ids,
        load_pvsynth,
    )

    variant = args.variant
    t0 = time.time()
    loaded = load_behavior(args, behavior, layers)
    tbl_pv = load_pvsynth(args, behavior, layers, loaded.dim, loaded.shas)
    tbl_ood, ood_note = load_ood_table(args, behavior, layers, loaded.dim, loaded.shas)
    if tbl_ood is None and behavior in OOD_SPECS:
        raise RuntimeError(f"[{behavior}] expected new OOD stores but loader returned none")
    _drop_unused_arrays(loaded, tbl_ood, variant)
    _log_rss("tables-loaded")

    # map + whitening: IDENTICAL across P-A and P-B (linear ADD recipe)
    wh, mapfit, map_diag_linear, u_label, n_u = fit_linear_add_map(args, loaded, variant, layers)
    map_diags = {"linear": map_diag_linear}

    # ---- merged labeled table: [train | wc | ev | ood] ----------------------
    n_tr = len(loaded.tbl.ctx_order)
    n_wc = len(loaded.tbl_wc.ctx_order)
    n_ev = len(loaded.tbl_ev.ctx_order)
    n_ood = len(tbl_ood.ctx_order) if tbl_ood is not None else 0
    base_wc, base_ev, base_ood = n_tr, n_tr + n_wc, n_tr + n_wc + n_ev

    ctx_blocks = [
        loaded.tbl.z_by_variant[variant],
        loaded.tbl_wc.z_by_variant[variant],
        loaded.tbl_ev.z_by_variant[variant],
    ]
    ans_blocks = [loaded.tbl.z_ans, loaded.tbl_wc.z_ans, loaded.tbl_ev.z_ans]
    if tbl_ood is not None:
        ctx_blocks.append(tbl_ood.z_by_variant[variant])
        ans_blocks.append(tbl_ood.z_ans)
    z_ctx = _whiten_concat_blocks(ctx_blocks, wh)
    z_ans = _whiten_concat_blocks(ans_blocks, wh)
    del ctx_blocks, ans_blocks

    dv_raw = np.concatenate(
        [np.asarray(t.dv, dtype=np.float64) for t in (loaded.tbl, loaded.tbl_wc, loaded.tbl_ev)]
        + ([np.asarray(tbl_ood.dv, dtype=np.float64)] if tbl_ood is not None else [])
    )
    ctx_ids = (
        [str(c) for c in loaded.tbl.ctx_order]
        + [str(c) for c in loaded.tbl_wc.ctx_order]
        + [str(c) for c in loaded.tbl_ev.ctx_order]
        + ([str(c) for c in tbl_ood.ctx_order] if tbl_ood is not None else [])
    )
    rb_w = np.einsum("ld,lde->le", loaded.rb, wh.w)

    # pvsynth (eval-only; never in the merged table)
    z_pv = fits.apply_whitening(tbl_pv.z_by_variant[variant], wh)
    za_pv = fits.apply_whitening(tbl_pv.z_ans, wh)
    dv_pv = np.asarray(tbl_pv.dv, dtype=np.float64)
    ids_pv = {str(c) for c in tbl_pv.ctx_order}

    # every raw fp16 activation block is now merged/whitened — free them (the
    # fits read only dv/groups/ctx_order/row_rungs past this point)
    _release_table_arrays(loaded, tbl_ood, tbl_pv)
    _log_rss("merged-table-built")

    # WildChat split (shared with r2fair: sha1(ctx_id) mod 5 == 4 -> eval)
    ev_mask = _wc_eval_mask(loaded.tbl_wc.ctx_order)
    wc_eval_rows = base_wc + np.flatnonzero(ev_mask)
    wc_train_rows = base_wc + np.flatnonzero(~ev_mask)
    ids_wc_eval = {ctx_ids[i] for i in wc_eval_rows}

    # ---- dataset roster (P-B / LODO seam) -----------------------------------
    lmax = LMAX[behavior]
    elic_cell = fits.realize_budget_cell(
        loaded.tbl.groups, budget_l=lmax, draw=args.draw, seed=args.seed
    )
    tbl_groups = np.asarray([str(g) for g in loaded.tbl.groups])
    datasets = [
        DatasetSpec(
            name="train",
            rows=np.asarray(elic_cell.row_idx, dtype=np.int64),
            groups=tbl_groups[elic_cell.row_idx],
        )
    ]
    ev_rungs = np.asarray([str(r) for r in loaded.tbl_ev.row_rungs])
    ev_groups = np.asarray([str(g) for g in loaded.tbl_ev.groups])
    for rung in sorted(set(ev_rungs)):
        rows = base_ev + np.flatnonzero(ev_rungs == rung)
        datasets.append(DatasetSpec(name=rung, rows=rows, groups=ev_groups[rows - base_ev]))
    if tbl_ood is not None:
        ood_rungs = np.asarray([str(r) for r in tbl_ood.row_rungs])
        ood_groups = np.asarray([str(g) for g in tbl_ood.groups])
        for rung in sorted(set(ood_rungs)):
            rows = base_ood + np.flatnonzero(ood_rungs == rung)
            datasets.append(DatasetSpec(name=rung, rows=rows, groups=ood_groups[rows - base_ood]))
    ds_by_name = {d.name: d for d in datasets}
    eval_datasets = [d.name for d in datasets if d.name != "train"]
    _log(
        f"[{behavior}] roster: {[(d.name, len(np.asarray(d.rows))) for d in datasets]} | "
        f"wc train/eval {len(wc_train_rows)}/{len(wc_eval_rows)} | pvsynth {len(ids_pv)}"
    )

    frozen, frozen_src = committed_frozen(args, loaded, behavior, variant, layers, ROSTER)

    rows_all: list[dict] = []
    skips_all: list[dict] = []
    per_layer_all: list[dict] = []
    fit_reports: list[dict] = []
    pools_record: list[dict] = []
    kwargs = {"n_boot": args.n_boot} if args.n_boot else {}

    def _fit_and_score(
        protocol: str,
        fit_label: str,
        readout_rows,
        dv_z,
        eval_specs: list[tuple[str, object]],
        extra_prov: dict,
    ) -> None:
        """One full-union transfer fit + per-rung evaluation (both protocols).

        ``eval_specs`` = [(rung_label, merged_rows | ("pv", None))]; pvsynth
        rows come from the separate whitened pvsynth arrays.
        """
        readout_rows = np.asarray(readout_rows, dtype=np.int64)
        _assert_well_posed(len(readout_rows), loaded.dim, f"{behavior}/{fit_label}")
        ev_z_parts, ev_za_parts, ev_dv_parts, ev_rung_parts = [], [], [], []
        eval_id_sets: dict[str, set] = {}
        for label, rows in eval_specs:
            if rows is None:  # pvsynth
                ev_z_parts.append(z_pv)
                ev_za_parts.append(za_pv)
                ev_dv_parts.append(dv_pv)
                ev_rung_parts.append(np.asarray([label] * z_pv.shape[1]))
                eval_id_sets[label] = ids_pv
                continue
            rows = np.asarray(rows, dtype=np.int64)
            if rows.size == 0:
                skips_all.append(
                    {
                        "protocol": protocol,
                        "fit": fit_label,
                        "eval_rung": label,
                        "n_eval": 0,
                        "reason": "empty eval row set (group draw realized no held-in rows)",
                    }
                )
                continue
            ev_z_parts.append(np.ascontiguousarray(z_ctx[:, rows]))
            ev_za_parts.append(np.ascontiguousarray(z_ans[:, rows]))
            ev_dv_parts.append(dv_raw[rows])
            ev_rung_parts.append(np.asarray([label] * rows.size))
            eval_id_sets[label] = {ctx_ids[i] for i in rows}
        z_ev = np.concatenate(ev_z_parts, axis=1)
        za_ev = np.concatenate(ev_za_parts, axis=1)
        dv_ev = np.concatenate(ev_dv_parts)
        rungs_ev = np.concatenate(ev_rung_parts)
        del ev_z_parts, ev_za_parts

        leak = _leakage_assert({ctx_ids[i] for i in readout_rows}, eval_id_sets, fit_label)
        cell = fits.BudgetCell(
            row_idx=readout_rows,
            fold_ids=np.zeros(len(readout_rows), dtype=np.int64),
            n_folds=1,
            budget_l=lmax,
            draw=args.draw,
            seed=args.seed,
            fold_scheme=f"r2v2-{protocol}-full-union",
        )
        data = arms.CellData(
            z_ctx=z_ctx, z_ans=z_ans, dv=dv_z, rb=rb_w, mapfit=mapfit, layers=tuple(layers)
        )
        prov = {
            "mode": "r2v2",
            "protocol": protocol,
            "fit": fit_label,
            "behavior": behavior,
            "variant": variant,
            "regime": args.regime,
            "map_kind": "linear",
            "map_condition": "add",
            "config": "config_a",
            "budget_l": lmax,
            "n_readout": int(len(readout_rows)),
            "dv_scaling": "per_pool_zscore_train_targets_v1",
            **extra_prov,
        }
        lam_sink: list[dict] = []
        t1 = time.time()
        with fits.capture_selected_lambdas(lam_sink):
            rows, skips, scores = transfer_rows_for(
                data,
                cell,
                z_ev,
                dv_ev,
                za_ev,
                rungs_ev,
                frozen,
                prov,
                layers,
                ROSTER,
                device=args.device,
                n_boot=args.n_boot,
                min_n=args.min_n,
            )
        wall = round(time.time() - t1, 1)
        per_layer_all.extend(
            per_layer_rows_for(
                scores, dv_ev, frozen, {**prov, "eval_rung": "all"}, layers, frozen_src
            )
        )
        rows_all.extend(rows)
        skips_all.extend(skips)
        fit_reports.append(
            {
                "protocol": protocol,
                "fit": fit_label,
                "n_readout": int(len(readout_rows)),
                "d": int(loaded.dim),
                "well_posed": f"n_train {len(readout_rows)} > d {loaded.dim}",
                "leakage": leak,
                "ridge_lambda_diagnostics": lam_sink,
                "recon": _eval_rung_reconstruction(
                    mapfit, z_ev, za_ev, rungs=[str(r) for r in rungs_ev], knn=True
                ),
                "fit_wall_s": wall,
            }
        )
        del scores, z_ev, za_ev, data
        _free_cuda(args.device)
        _log(
            f"[{behavior}] {fit_label}: {len(rows)} rows in {wall}s (n_readout={len(readout_rows)})"
        )
        _log_rss(f"fit-done-{fit_label}")

    # ---- P-A: the r2fair protocol (one eliciting dataset + judged WC train) --
    if "A" in args.protocols:
        readout_pa = np.concatenate(
            [np.asarray(elic_cell.row_idx, dtype=np.int64), wc_train_rows]
        ).astype(np.int64)
        dv_z_pa = _multi_pool_zscored_dv(dv_raw, [elic_cell.row_idx, wc_train_rows])
        eval_specs_pa: list[tuple[str, object]] = [(WC_RUNG, wc_eval_rows), (PV_RUNG, None)]
        eval_specs_pa += [(name, ds_by_name[name].rows) for name in eval_datasets]
        _fit_and_score(
            "P-A",
            "P-A",
            readout_pa,
            dv_z_pa,
            eval_specs_pa,
            {
                "readout_train": "union: eliciting train (budget cell) + judged WildChat "
                "train split (the r2fair protocol)",
                "n_readout_eliciting": int(len(elic_cell.row_idx)),
                "n_readout_wc_train": int(len(wc_train_rows)),
            },
        )

        # train-OOF setting (the P-A 'train' rung; fair union fold machinery)
        wcf = _wc_fold_ids([ctx_ids[i] for i in wc_train_rows], elic_cell.n_folds)
        cell_oof = fits.BudgetCell(
            row_idx=readout_pa,
            fold_ids=np.concatenate([elic_cell.fold_ids, wcf]),
            n_folds=elic_cell.n_folds,
            budget_l=lmax,
            draw=args.draw,
            seed=args.seed,
            fold_scheme=f"r2v2-fair-union-{elic_cell.fold_scheme}",
        )
        n_el = len(elic_cell.row_idx)
        data_oof = arms.CellData(
            z_ctx=z_ctx, z_ans=z_ans, dv=dv_z_pa, rb=rb_w, mapfit=mapfit, layers=tuple(layers)
        )
        lam_sink_oof: list[dict] = []
        t1 = time.time()
        with fits.capture_selected_lambdas(lam_sink_oof):
            scores_tr, tr_skips = arms.run_cell(
                data_oof, cell_oof, arms=list(ROSTER), device=args.device
            )
        dv_el = np.asarray(loaded.tbl.dv, dtype=np.float64)[elic_cell.row_idx]
        scores_el = {s: np.ascontiguousarray(sc[:, :n_el]) for s, sc in scores_tr.items()}
        prov_tr = {
            "mode": "r2v2",
            "protocol": "P-A",
            "fit": "P-A-train-oof",
            "behavior": behavior,
            "variant": variant,
            "regime": args.regime,
            "map_kind": "linear",
            "map_condition": "add",
            "config": "config_a",
            "budget_l": lmax,
            "rung_kind_note": "in_split_oof_union_readout",
        }
        rows_tr, skips_tr = arms.evaluate_transfer(
            scores_el,
            dv_el,
            np.asarray(["train"] * n_el),
            frozen,
            provenance=prov_tr,
            cell=cell_oof,
            layers=tuple(layers),
            min_n=args.min_n,
            **kwargs,
        )
        rows_all.extend(rows_tr)
        skips_all.extend(skips_tr)
        skips_all.extend(
            {"arm": s, "reason": f"train oof: {r}", "protocol": "P-A"}
            for s, r in sorted(tr_skips.items())
        )
        skips_all.extend(
            arms.roster_accounting_skips(
                list(ROSTER), scores_tr, tr_skips, protocol="P-A", eval_rung="train"
            )
        )
        per_layer_all.extend(
            per_layer_rows_for(
                scores_el, dv_el, frozen, {**prov_tr, "eval_rung": "train"}, layers, frozen_src
            )
        )
        fit_reports.append(
            {
                "protocol": "P-A",
                "fit": "P-A-train-oof",
                "n_readout": int(len(readout_pa)),
                "d": int(loaded.dim),
                "ridge_lambda_diagnostics": lam_sink_oof,
                "fit_wall_s": round(time.time() - t1, 1),
            }
        )
        del scores_tr, scores_el, data_oof
        _free_cuda(args.device)
        _log(f"[{behavior}] P-A train-OOF done ({len(rows_tr)} rows)")

    # ---- P-B: 80% of every dataset, one held out whole (LODO-parameterized) --
    if "B" in args.protocols:
        holdouts = args.pb_holdouts or eval_datasets
        unknown = sorted(set(holdouts) - set(eval_datasets))
        if unknown:
            raise ValueError(f"--pb-holdouts {unknown} not in eval datasets {eval_datasets}")
        for holdout in holdouts:
            pool = assemble_readout_pool(
                datasets, holdout=holdout, train_frac=args.train_frac, seed=args.seed
            )
            pool_rows = [pool.train_rows[n] for n in sorted(pool.train_rows)]
            readout_pb = np.concatenate(pool_rows + [wc_train_rows]).astype(np.int64)
            dv_z_pb = _multi_pool_zscored_dv(dv_raw, pool_rows + [wc_train_rows])
            eval_specs_pb: list[tuple[str, object]] = [
                (holdout, ds_by_name[holdout].rows),
                (WC_RUNG, wc_eval_rows),
                (PV_RUNG, None),
            ]
            eval_specs_pb += [
                (f"heldin:{name}", pool.heldin_eval_rows[name])
                for name in sorted(pool.heldin_eval_rows)
            ]
            pools_record.append(
                {
                    "behavior": behavior,
                    "holdout": holdout,
                    "train_frac": pool.train_frac,
                    "seed": pool.seed,
                    "per_dataset_train_n": {k: int(len(v)) for k, v in pool.train_rows.items()},
                    "per_dataset_heldin_n": {
                        k: int(len(v)) for k, v in pool.heldin_eval_rows.items()
                    },
                    "n_wc_train": int(len(wc_train_rows)),
                    "n_readout_total": int(len(readout_pb)),
                }
            )
            _fit_and_score(
                "P-B",
                f"P-B-holdout-{holdout}",
                readout_pb,
                dv_z_pb,
                eval_specs_pb,
                {
                    "holdout": holdout,
                    "train_frac": args.train_frac,
                    "readout_train": "union: 80% GROUP-level slice of every trait-eliciting "
                    "dataset except the holdout (whole) + judged WildChat train split",
                    "included_datasets": sorted(pool.train_rows),
                },
            )

    _free_cuda(args.device)
    return {
        "rows": rows_all,
        "skips": skips_all,
        "per_layer": per_layer_all,
        "fit_reports": fit_reports,
        "pools": pools_record,
        "map_diagnostics": {f"{variant}|add|linear|{u_label}": map_diags["linear"]},
        "frozen": {a: int(i) for a, i in frozen.items()},
        "frozen_source": frozen_src,
        "datasets": {d.name: int(len(np.asarray(d.rows))) for d in datasets},
        "ood_note": ood_note,
        "budget_l": lmax,
        "n_u": int(n_u),
        "u_label": u_label,
        "loaded": loaded,
        "wall_s": round(time.time() - t0, 1),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--behaviors", nargs="+", default=list(BEHAVIORS), choices=list(BEHAVIORS))
    ap.add_argument(
        "--variant",
        default="context_end",
        choices=["context_end", "prefix_end"],
        help="context_end ONLY by default (recorded deviation; the OOD corpora carry a "
        "constant empty prefix — prefix arms are structurally degenerate there)",
    )
    ap.add_argument(
        "--protocols",
        default="AB",
        choices=["A", "B", "AB"],
        help="which readout protocols to fit (A = r2fair, B = LODO-mixture)",
    )
    ap.add_argument(
        "--pb-holdouts",
        nargs="+",
        default=None,
        help="P-B holdout datasets (default: every non-train dataset)",
    )
    ap.add_argument("--train-frac", type=float, default=0.8)
    ap.add_argument("--regime", default="e1", choices=("e1",))
    ap.add_argument("--layers", type=int, nargs="+", default=None)
    ap.add_argument("--n-layers", type=int, default=28)
    ap.add_argument("--draw", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-n", type=int, default=3)
    ap.add_argument("--n-boot", type=int, default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--rb-source", default="auto", choices=("auto", "bank", "extract"))
    ap.add_argument("--ood-dv-max-null-frac", type=float, default=0.05)
    ap.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    ap.add_argument("--main-root", type=Path, default=DEFAULT_MAIN_ROOT)
    ap.add_argument("--tensors-root", type=Path, default=DEFAULT_TENSORS_ROOT)
    ap.add_argument("--store-root", type=Path, default=DEFAULT_STORE_ROOT)
    ap.add_argument(
        "--ood-store-root",
        type=Path,
        default=None,
        help="root of the issue1739_ctxmap OOD-store mirror "
        "(default: <store-root>/ood_mirror/issue1739_ctxmap)",
    )
    ap.add_argument("--evil-ood-dv", type=Path, default=None)
    ap.add_argument("--syco-ood-dv", type=Path, default=None)
    ap.add_argument("--u-store", type=Path, default=None)
    ap.add_argument("--train-dv-root", type=Path, default=None)
    ap.add_argument("--wcrung-dv-root", type=Path, default=None)
    ap.add_argument("--wcrung-store", type=Path, default=None)
    ap.add_argument("--pvsynth-store-root", type=Path, default=None)
    ap.add_argument("--pvsynth-dv-root", type=Path, default=None)
    ap.add_argument("--allow-overwrite-committed", action="store_true")
    ap.add_argument("--import-check", action="store_true")
    args = ap.parse_args(argv)
    if args.u_store is None:
        args.u_store = args.store_root / "u_store"
    if args.train_dv_root is None:
        args.train_dv_root = args.store_root / "train_dv"
    if args.wcrung_dv_root is None:
        args.wcrung_dv_root = args.main_root / "wildchat_rung" / "dv_dataset"
    if args.pvsynth_store_root is None:
        args.pvsynth_store_root = args.store_root / "pvsynth_capture_store"
    if args.pvsynth_dv_root is None:
        args.pvsynth_dv_root = args.main_root / "pvsynth" / "dv_dataset"
    if args.ood_store_root is None:
        args.ood_store_root = args.store_root / "ood_mirror" / "issue1739_ctxmap"
    if args.evil_ood_dv is None:
        args.evil_ood_dv = (
            args.main_root / "evil_ood_full" / "dv_dataset" / "evil" / ("labeling.json")
        )
    if args.syco_ood_dv is None:
        args.syco_ood_dv = (
            args.ood_store_root / "syco_ood" / "dv_dataset" / "sycophancy" / "labeling.json"
        )
    # jobd/fair helpers read these spellings; keep coherent.
    args.variants = [args.variant]
    args.map_kinds = ["linear"]
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    from scripts.issue1739_wcrung_arms import _assert_no_judge_modules

    _assert_no_judge_modules("at entry")
    if args.import_check:
        from explore_persona_space.experiments.issue_1739 import arms as _arms
        from explore_persona_space.experiments.issue_1739 import fits, store_io  # noqa: F401
        from explore_persona_space.orchestrate.env import load_dotenv  # noqa: F401
        from scripts.issue1739_fits import (  # noqa: F401
            _eval_rung_reconstruction,
            _git_commit,
            _load_labeled,
        )
        from scripts.issue1739_jobd_r2aug import (  # noqa: F401
            committed_frozen,
            load_behavior,
            per_layer_rows_for,
            transfer_rows_for,
        )
        from scripts.issue1739_result2fair_score import (  # noqa: F401
            _wc_eval_mask,
            _wc_fold_ids,
            fit_add_maps,
            load_pvsynth,
        )
        from scripts.issue1739_wcrung_arms import (  # noqa: F401
            _git_tracked,
            _sha256,
            _verify_input_shas,
            modal_frozen_layers,
        )

        assert callable(fits.capture_selected_lambdas), "lambda sink missing — stale checkout"
        for slug in ROSTER:
            assert slug in _arms.ARM_REGISTRY, f"{slug} missing from ARM_REGISTRY"
        _assert_no_judge_modules("after --import-check imports")
        print("[r2v2] import-check OK", flush=True)
        sys.stdout.flush()
        sys.stderr.flush()
        sys.exit(0)

    from explore_persona_space.experiments.issue_1739 import arms
    from explore_persona_space.orchestrate.env import load_dotenv
    from scripts.issue1739_fits import _git_commit
    from scripts.issue1739_jobd_r2aug import VARIANT_SCOPE_NOTE, _env_versions
    from scripts.issue1739_result2fair_score import PVSYNTH_READOUT_DEVIATION
    from scripts.issue1739_wcrung_arms import _git_tracked, _verify_input_shas

    load_dotenv()
    for b in args.behaviors:
        out = args.out_root / b / "all_arms_spearman.json"
        if _git_tracked(out) and not args.allow_overwrite_committed:
            raise SystemExit(f"refusing to overwrite git-TRACKED output: {out}")

    layers = args.layers or list(range(args.n_layers))
    commit = _git_commit()
    env = _env_versions()
    failures: list[dict] = []
    t_all = time.time()
    for behavior in args.behaviors:
        t0 = time.time()
        try:
            res = run_behavior(args, behavior, layers)
        except (FileNotFoundError, RuntimeError, ValueError, AssertionError) as exc:
            failures.append({"behavior": behavior, "error": f"{type(exc).__name__}: {exc}"})
            _log(f"{behavior} FAILED: {exc}")
            continue
        loaded = res.pop("loaded")
        out_dir = args.out_root / behavior
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "all_arms_spearman.json"
        arms.write_summary(
            [],
            out_path,
            meta={
                "mode": "r2v2",
                "behavior": behavior,
                "config": "config_a",
                "regimes": [args.regime],
                "variants": [args.variant],
                "variant_scope": VARIANT_SCOPE_NOTE,
                "protocols": list(args.protocols),
                "arms": list(ROSTER),
                "label_consuming_arms": sorted(LABEL_CONSUMING),
                "map_kinds": ["linear"],
                "map_condition": "add",
                "mlp_arms_note": "ALL MLP arms removed (r2v2 dispatch, epm:progress v639); "
                "linear-map + closed-form readouts throughout — this round cannot speak to "
                "map nonlinearity",
                "protocol_definitions": {
                    "P-A": "readout = ONE trait-eliciting dataset (train budget cell) + "
                    "judged WildChat train split (the r2fair protocol)",
                    "P-B": f"readout = {args.train_frac:.0%} GROUP-level slice of every "
                    "trait-eliciting dataset except one held out whole + judged WildChat "
                    "train split; one fit per holdout (LODO)",
                },
                "lodo_parameterization": "assemble_readout_pool(datasets, holdout=..., "
                "include=..., train_frac=..., seed=...) — group-membership sides are a pure "
                "hash of (dataset, group_key, seed), independent of holdout and roster, so "
                "Result-5 LODO sweeps / subset ablations are loops over the parameters",
                "datasets": res["datasets"],
                "pb_pools": res["pools"],
                "fit_reports": res["fit_reports"],
                "frozen_layers": res["frozen"],
                "frozen_layer_source": res["frozen_source"],
                "ood_dv_prep": res["ood_note"],
                "pvsynth_deviation": PVSYNTH_READOUT_DEVIATION,
                "budget_l": res["budget_l"],
                "n_u": res["n_u"],
                "u_pool_label": res["u_label"],
                "input_paths": {k: str(v) for k, v in loaded.paths.items()},
                "input_sha256": loaded.shas,
                "git_commit": commit,
                "env_versions": env,
                "wall_s": round(time.time() - t0, 1),
                "judge_called": False,
            },
            extra={
                "transfer_rows": res["rows"],
                "transfer_skips": res["skips"],
                "per_layer_rows": res["per_layer"],
                "n_transfer_rows": len(res["rows"]),
                "n_per_layer_rows": len(res["per_layer"]),
            },
        )
        (out_dir / "map_diagnostics.json").write_text(json.dumps(res["map_diagnostics"], indent=1))
        (out_dir / "readout_pools.json").write_text(
            json.dumps({"pools": res["pools"], "fit_reports": res["fit_reports"]}, indent=1)
        )
        _verify_input_shas(loaded.shas)
        _log(f"{behavior} done: {len(res['rows'])} transfer rows in {res['wall_s']}s -> {out_path}")
        del loaded, res

    from scripts.issue1739_wcrung_arms import _assert_no_judge_modules as _anjm

    _anjm("at exit")
    _log(f"all done in {time.time() - t_all:.0f}s")
    if failures:
        args.out_root.mkdir(parents=True, exist_ok=True)
        (args.out_root / "r2v2_failures.json").write_text(json.dumps(failures, indent=1))
        for f in failures:
            print(f"[r2v2] FAILED {f}", file=sys.stderr)
        sys.stdout.flush()
        sys.stderr.flush()
        sys.exit(2)
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(0)


if __name__ == "__main__":
    main()
