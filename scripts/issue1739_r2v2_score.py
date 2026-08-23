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

- **P-C** (the 2026-08-07 LODO-consistent round, user inline override): per
  holdout, the MAP is refit too — on the generic fit pool + the SAME 80%
  GROUP-level slices P-B's readout trains on (whitening stays frozen at the
  behavior-level ADD fit, which contains no holdout rows) — and the readout
  is the P-B pool verbatim. The holdout is unseen by map AND readout
  (hard context-id disjointness asserts, readout- and map-side). Per-holdout
  results persist to ``<out_root>/<behavior>/percell/pc_holdout_<name>.json``
  the moment each cell lands, with a regime-keyed resume that skips
  completed cells at entry.

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

# Arms that MAY be appended to ROSTER via --extra-arms. Restricted to the
# arm-registry slugs whose scoring path needs nothing the r2v2 CellData does
# not already carry, and whose frozen layer the committed train summary
# already records (committed_frozen() raises on a roster arm it cannot
# resolve). arm12_oracle_reg fits ridge on TRUE answer summaries and scores on
# TRUE answer summaries -- both already materialized here for arm11 -- so it is
# a pure additional fit, no new inputs. It CONSUMES DV labels, so it joins
# LABEL_CONSUMING (P-B holds out its readout slice like the other two).
#
# claim4-controls round (#1739, 2026-08-19) adds two more:
# - arm2_ctx_native: the context-native midpoint-split direction. Its
#   transfer semantics ride run_transfer_cell's 2-fold machinery unchanged
#   (direction fit on the readout pool's midpoint split, projected onto the
#   holdout rows -- the arms.py:857-868 closed-form path, fold 1 = pool /
#   fold 0 = eval). CONSUMES DV labels (its train-row midpoint split), so it
#   joins LABEL_CONSUMING; its frozen layer resolves from the committed
#   train summary (rows verified present for all three behaviors).
# - arm20_shuffled_map_ridge: ridge on the weight-row-permuted TRUE map
#   (fits.shuffled_map_weights, rank/Frobenius-preserving) -- the
#   capacity-matched pairing-free comparator. The committed train summary
#   carries NO arm20 row, so its frozen layer is MATCHED to arm7's committed
#   layer (the result2fair_score MATCHED_COMPANIONS precedent) -- see
#   MATCHED_FROZEN_COMPANIONS below. CONSUMES DV labels (ridge on mp_shuf).
EXTRA_ARMS_ALLOWED = {"arm12_oracle_reg", "arm2_ctx_native", "arm20_shuffled_map_ridge"}
EXTRA_ARMS_LABEL_CONSUMING = {"arm12_oracle_reg", "arm2_ctx_native", "arm20_shuffled_map_ridge"}

# Frozen-layer matched companions: arm -> reference arm whose COMMITTED frozen
# layer the arm reads at (the fair-round MATCHED_COMPANIONS convention:
# ("arm20_shuffled_map_ridge", "arm7_map_ridge_pred") -- a like-for-like
# same-layer comparison; the delta must not carry a layer-selection term).
# arm2q (the arm2fix R-C quantile fallback) has no committed train row either
# -- it reads at arm2's committed layer (same construct, split rule the only
# delta; the parse-time validator guarantees arm2 is rostered whenever arm2q
# is).
MATCHED_FROZEN_COMPANIONS = {
    "arm20_shuffled_map_ridge": "arm7_map_ridge_pred",
    "arm2q_ctx_native": "arm2_ctx_native",
}

# ---------------------------------------------------------------------------
# arm2fix (repair ladder) surface -- #1739 plan §4 "Leg 2" D1
# ---------------------------------------------------------------------------
# The arm2-family slugs: the context-native direction under repair plus its
# R-C quantile-fallback sibling (a NEW slug BESIDE the unrepaired arm2, never
# a relabel).
A2_FAMILY = ("arm2_ctx_native", "arm2q_ctx_native")
# --arm2-adapter choices (the pre-registered repair menu, plan §4):
#   v1                      -- unrepaired adapter; rows gain adapter provenance
#                              + train-row id-hash fields only.
#   v2-component-restricted -- R-B: the arm2 direction fit consumes the pool's
#                              TRAIT-ELICITING components only (the judged
#                              WildChat block excluded from the fit; it never
#                              entered the committed folded fit either).
#   v2-quantile             -- R-C: arm2q_ctx_native (top/bottom 25% quantile
#                              split over the full shared pool) emitted BESIDE
#                              the unrepaired arm2.
#   v2-quantile-restricted  -- R-C over the eliciting-only pool (the R-B x R-C
#                              composition, for a D0 that shows BOTH the
#                              WildChat confound and midpoint degeneracy).
# R-A (a wiring fix, iff probe P1 fails) is code, not a flag: P1
# (tests/test_issue1739_arm2fix.py planted-direction test) PASSES on the
# current dispatch, so no wiring change ships -- the slot stays documented
# here for the ladder record.
ARM2_ADAPTERS = ("v1", "v2-component-restricted", "v2-quantile", "v2-quantile-restricted")
# Adapters whose selected repair RESTRICTS arm2's fit rows relative to the
# shared P-B pool (the matched-budget parity duty's trigger, plan §4).
ARM2_RESTRICTED_ADAPTERS = ("v2-component-restricted", "v2-quantile", "v2-quantile-restricted")

# Shufpair-variant pass roster (claim4 P0.2): the two map-consuming arms whose
# map input is swapped for the pairing-shuffled refit, plus arm4 as the
# map-INDEPENDENT pairing check (its per-seed rows must be bit-identical
# across map variants -- asserted in run_behavior after both passes). Arms
# 1/11/2 do not consume the map; arm20 consumes the TRUE map by construction.
SHUFPAIR_ROSTER = ("arm4_ridge_ctx", "arm6_map_proj_e1", "arm7_map_ridge_pred")

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

# Seed-keyed output schema version (claim4-controls --seeds mode). Recorded in
# the summary meta and REQUIRED by the per-seed resume predicate — bump it
# whenever the per-seed output contract changes so a stale layout can never
# silently satisfy a resume (v2 = the claim4-controls round-2 contract).
SEED_OUT_SCHEMA_VERSION = 2


def _log(msg: str) -> None:
    print(f"[r2v2 {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _wire_fits_rss_logging() -> None:
    """Route the fits-module INFO breadcrumbs to stdout (crash-fix r4).

    The scorer never configured logging, so ``logger.info`` from
    ``experiments.issue_1739.fits`` (incl. the new ``[fits][rss ...]``
    stage crumbs INSIDE ``fit_linear_map`` — the 2026-08-18 kill site) was
    swallowed by Python's lastResort WARNING+ handler and the fit's
    split/diagnostics/refit stages were invisible in the pod log. Scoped to
    the fits logger — never ``basicConfig`` on root (third-party INFO noise).
    Idempotent (re-entry adds no second handler); log-only, no numerics.
    """
    import logging

    fits_logger = logging.getLogger("explore_persona_space.experiments.issue_1739.fits")
    if not fits_logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter("[r2v2 %(asctime)s] %(message)s", datefmt="%H:%M:%S")
        )
        fits_logger.addHandler(handler)
        fits_logger.setLevel(logging.INFO)
        fits_logger.propagate = False


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


def dataset_roster(loaded, tbl_ood, elic_cell, *, base_ev: int, base_ood: int):
    """The P-B dataset roster: train budget cell + eval rungs + OOD rungs.

    Pure code motion out of :func:`prepare_behavior` (arm2fix round) so the
    D0-P4 direction-stability probe (`issue1739_arm2fix_d0.py`) assembles the
    IDENTICAL roster — one construction, two readers, no drift. Rows are
    merged-table indices under the [train | wc | ev | ood] block layout.
    """
    import numpy as np

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
    return datasets


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
# arm2fix helpers: repair-ladder pass plan + matched-regime sanity (plan §4)
# ---------------------------------------------------------------------------


def _row_ids_sha256(ids) -> str:
    """Order-independent sha256 over context ids (the parity id-hash currency)."""
    joined = "\n".join(sorted(str(c) for c in ids))
    return hashlib.sha256(joined.encode()).hexdigest()


def _quantile_fit_rows(readout_rows, dv_z):
    """The rows arm2q's direction consumes: top/bottom ARM2Q_QUANTILES of the
    readout's (z-scored) dv. Thresholds come from the ONE shared helper the
    dispatch block reads (:func:`arms.arm2q_thresholds`), so the parity arm7
    refit trains on EXACTLY the arm-internal split rows."""
    import numpy as np

    from explore_persona_space.experiments.issue_1739 import arms

    rows = np.asarray(readout_rows, dtype=np.int64)
    vals = np.asarray(dv_z, dtype=np.float64)[rows]
    q_lo, q_hi = arms.arm2q_thresholds(vals)
    return rows[(vals <= q_lo) | (vals >= q_hi)]


def _repaired_slug(adapter: str) -> str:
    """The arm slug that carries the SELECTED repair's rows (plan §4 ladder)."""
    return "arm2q_ctx_native" if adapter.startswith("v2-quantile") else "arm2_ctx_native"


def _arm2fix_pass_tags(adapter: str, roster, *, parity_refit_arm7: bool) -> tuple:
    """Pure ``(label, preds_tag)`` sequence of the arm2fix pass plan — the
    SINGLE source of truth: :func:`_arm2fix_passes` realizes it (and asserts
    equality at return, so drift is loud) and :func:`_planned_preds_files`
    consumes it (codex r5 MAJOR arm2fix-preds-plan-universe-still-derived:
    the expected-sidecar universe must be derivable BEFORE execution, from
    configuration alone — never from realized rows or the writer log)."""
    if adapter not in ARM2_ADAPTERS:
        raise ValueError(f"unknown arm2 adapter {adapter!r} (choices: {ARM2_ADAPTERS})")
    rest = tuple(a for a in roster if a not in A2_FAMILY)
    if adapter in ("v1", "v2-quantile"):
        tags = [("std", None)]
    elif adapter == "v2-component-restricted":
        tags = [("a2r", "a2r")] + ([("std", None)] if rest else [])
    else:  # v2-quantile-restricted
        tags = [("a2qr", "a2qr"), ("std", None)]
    if parity_refit_arm7:
        tags.append(("parity", "parity"))
    return tuple(tags)


def _arm2fix_passes(
    adapter: str,
    roster,
    pool_rows,
    wc_rows,
    dv_z,
    *,
    parity_refit_arm7: bool = False,
):
    """Pure pass plan for one P-B holdout under the arm2fix repair ladder.

    Returns ``[SimpleNamespace(label, roster, readout, preds_tag, arm_meta)]``
    where ``arm_meta[slug] = {"adapter": tag, "fit_rows": rows}`` — fit_rows
    are the rows whose ctx ids get hashed into that arm's
    ``train_row_ids_sha256`` (the label budget the arm's direction/readout
    consumes; the matched-budget parity currency, plan §4 Must-Fix).

    - v1: one standard pass; arm2-family rows tagged ``adapter: v1``.
    - v2-component-restricted (R-B): the arm2-family fit rides its OWN pass
      whose readout is the pool's TRAIT-ELICITING components only (the judged
      WildChat block excluded from the direction fit); any non-family roster
      arms keep the standard shared-pool pass.
    - v2-quantile (R-C): ONE standard pass — arm2q's direction internally
      consumes the top/bottom-quantile subset, hashed here via the shared
      thresholds; the unrepaired arm2 rides the same pass (``adapter: v1``).
    - v2-quantile-restricted: arm2q rides an eliciting-only pass; the
      unrepaired arm2 (+ rest) keeps the standard pass.
    - parity_refit_arm7: appends the row-matched arm7 refit pass — readout ==
      the REPAIRED arm's fit_rows, so the fold can assert id-hash equality.
    """
    import numpy as np

    if adapter not in ARM2_ADAPTERS:
        raise ValueError(f"unknown arm2 adapter {adapter!r} (choices: {ARM2_ADAPTERS})")
    roster = tuple(roster)
    pool_arrs = [np.asarray(r, dtype=np.int64) for r in pool_rows]
    full = np.concatenate(pool_arrs + [np.asarray(wc_rows, dtype=np.int64)]).astype(np.int64)
    restricted = np.concatenate(pool_arrs).astype(np.int64)
    fam = tuple(a for a in roster if a in A2_FAMILY)
    rest = tuple(a for a in roster if a not in A2_FAMILY)
    if "arm2_ctx_native" not in roster:
        raise ValueError(f"--arm2-adapter needs arm2_ctx_native in the roster (got {roster})")
    if adapter.startswith("v2-quantile") and "arm2q_ctx_native" not in roster:
        raise ValueError(f"{adapter} needs arm2q_ctx_native in the roster (got {roster})")
    passes: list[SimpleNamespace] = []
    if adapter == "v1":
        passes.append(
            SimpleNamespace(
                label="std",
                roster=roster,
                readout=full,
                preds_tag=None,
                arm_meta={a: {"adapter": "v1", "fit_rows": full} for a in fam},
            )
        )
    elif adapter == "v2-component-restricted":
        passes.append(
            SimpleNamespace(
                label="a2r",
                roster=fam,
                readout=restricted,
                preds_tag="a2r",
                arm_meta={a: {"adapter": adapter, "fit_rows": restricted} for a in fam},
            )
        )
        if rest:
            passes.append(
                SimpleNamespace(label="std", roster=rest, readout=full, preds_tag=None, arm_meta={})
            )
    elif adapter == "v2-quantile":
        q_rows = _quantile_fit_rows(full, dv_z)
        passes.append(
            SimpleNamespace(
                label="std",
                roster=roster,
                readout=full,
                preds_tag=None,
                arm_meta={
                    "arm2_ctx_native": {"adapter": "v1", "fit_rows": full},
                    "arm2q_ctx_native": {"adapter": adapter, "fit_rows": q_rows},
                },
            )
        )
    else:  # v2-quantile-restricted
        q_rows = _quantile_fit_rows(restricted, dv_z)
        passes.append(
            SimpleNamespace(
                label="a2qr",
                roster=("arm2q_ctx_native",),
                readout=restricted,
                preds_tag="a2qr",
                arm_meta={"arm2q_ctx_native": {"adapter": adapter, "fit_rows": q_rows}},
            )
        )
        std_roster = tuple(a for a in roster if a != "arm2q_ctx_native")
        passes.append(
            SimpleNamespace(
                label="std",
                roster=std_roster,
                readout=full,
                preds_tag=None,
                arm_meta={"arm2_ctx_native": {"adapter": "v1", "fit_rows": full}},
            )
        )
    if parity_refit_arm7:
        rep = _repaired_slug(adapter)
        rep_meta = next(p.arm_meta[rep] for p in passes if rep in p.arm_meta)
        fit_rows = np.asarray(rep_meta["fit_rows"], dtype=np.int64)
        passes.append(
            SimpleNamespace(
                label="parity",
                roster=("arm7_map_ridge_pred",),
                readout=fit_rows,
                preds_tag="parity",
                arm_meta={
                    "arm7_map_ridge_pred": {"adapter": "parity-row-matched", "fit_rows": fit_rows}
                },
            )
        )
    realized_tags = tuple((p.label, p.preds_tag) for p in passes)
    declared_tags = _arm2fix_pass_tags(adapter, roster, parity_refit_arm7=parity_refit_arm7)
    assert realized_tags == declared_tags, (
        f"arm2fix pass-plan drift: realized {realized_tags} != declared {declared_tags} "
        "(_arm2fix_pass_tags is the single source the planned preds universe derives from)"
    )
    return passes


def _adapter_tag_for(slug: str, adapter: str) -> str:
    """The adapter tag a given arm2-family slug carries under a selected adapter."""
    if slug == "arm2q_ctx_native":
        return adapter if adapter.startswith("v2-quantile") else "n/a"
    return "v2-component-restricted" if adapter == "v2-component-restricted" else "v1"


def _matched_regime_sanity(
    args, behavior: str, layers, datasets, z_ctx, dv_raw, rb_w, frozen, variant: str, lmax: int
) -> list[dict]:
    """The plan-§4 sanity-instrument repair: arm2-family folded GROUP-level CV
    ρ on the P-B pool's TRAIT-ELICITING rows (the banked ``run_cell`` path) —
    the matched-regime acceptance read vs the committed train-grid band (the
    fold compares; this emits the per-seed values as
    ``rung_kind: sanity_matched_regime`` rows). Same regime as the committed
    band: folded CV on eliciting rows; pool COMPOSITION is the only delta.
    dv convention mirrors the transfer machinery: fit on per-pool z-scored dv,
    Spearman evaluated against raw dv.
    """
    import numpy as np

    from explore_persona_space.experiments.issue_1739 import arms, fits

    fold_k = int(getattr(args, "a2_sanity_folds", 5))
    sel_rows: list[np.ndarray] = []
    sel_groups: list[str] = []
    for d in datasets:
        rows_d = np.asarray(d.rows, dtype=np.int64)
        groups_d = np.asarray(d.groups)
        side = np.array(
            [_group_side_train(d.name, str(g), args.seed, args.train_frac) for g in groups_d],
            dtype=bool,
        )
        sel_rows.append(rows_d[side])
        # dataset-namespaced group keys: group names may collide across datasets
        sel_groups.extend(f"{d.name}|{g}" for g in groups_d[side])
    rows = np.concatenate(sel_rows)
    if rows.size != len(sel_groups):
        raise AssertionError(f"sanity rows {rows.size} != groups {len(sel_groups)}")
    dv_z = _multi_pool_zscored_dv(dv_raw, sel_rows)
    fold_raw = np.array(
        [
            int.from_bytes(
                hashlib.sha1(f"a2sanity|{g}|{int(args.seed)}".encode()).digest()[:8], "big"
            )
            % fold_k
            for g in sel_groups
        ]
    )
    uniq_f = sorted(set(fold_raw.tolist()))
    if len(uniq_f) < 2:
        raise RuntimeError(
            f"[{behavior}] matched-regime sanity: only {len(uniq_f)} non-empty group fold(s) "
            f"over {len(set(sel_groups))} groups — cannot run folded CV"
        )
    remap = {f: i for i, f in enumerate(uniq_f)}
    fold_ids = np.array([remap[f] for f in fold_raw], dtype=np.int64)
    cell = fits.BudgetCell(
        row_idx=rows,
        fold_ids=fold_ids,
        n_folds=len(uniq_f),
        budget_l=lmax,
        draw=args.draw,
        seed=args.seed,
        fold_scheme=f"a2sanity-grouphash-k{len(uniq_f)}",
    )
    a2_arms = [a for a in ROSTER if a in A2_FAMILY]
    if not a2_arms:
        raise RuntimeError("matched-regime sanity requested but no arm2-family arm in ROSTER")
    data = arms.CellData(
        z_ctx=z_ctx, dv=dv_z, rb=rb_w, z_ans=None, mapfit=None, layers=tuple(layers)
    )
    scores, skipped = arms.run_cell(data, cell, arms=a2_arms, device=args.device)
    if skipped:
        raise RuntimeError(f"[{behavior}] sanity run_cell skipped arms: {skipped}")
    dv_eval = np.asarray(dv_raw, dtype=np.float64)[rows]
    adapter = getattr(args, "arm2_adapter", None) or "v1"
    mv0 = (args.map_variants or [None])[0] if getattr(args, "map_variants", None) else None
    out: list[dict] = []
    for slug in a2_arms:
        sc = np.asarray(scores[slug], dtype=np.float64)
        rhos = [float(x) for x in arms.spearman_rows(sc, dv_eval)]
        fl = min(int(frozen[slug]), sc.shape[0] - 1)
        out.append(
            {
                "mode": "r2v2",
                "protocol": "P-B",
                "fit": "sanity-elic-pool-cv",
                "behavior": behavior,
                "variant": variant,
                "regime": args.regime,
                "arm": slug,
                "family": arms.ARM_REGISTRY.get(slug, {}).get("family", "context"),
                "eval_rung": "sanity_elic_pool_cv",
                "rung_kind": "sanity_matched_regime",
                "adapter": _adapter_tag_for(slug, adapter),
                "arm2_adapter": adapter,
                "rho_frozen": rhos[fl],
                "rho_per_layer": rhos,
                "layer": int(layers[fl]) if layers and sc.shape[0] > 1 else None,
                "n_eval": int(rows.size),
                "n_rows": int(rows.size),
                "n_folds": int(len(uniq_f)),
                "n_groups": int(len(set(sel_groups))),
                "budget_l": int(lmax),
                "dv_scaling": "fit=per_pool_zscore_train_targets_v1; eval=raw",
                "train_frac": args.train_frac,
                "draw": int(args.draw),
                "seed": int(args.seed),
                **({"map_variant": mv0} if mv0 is not None else {}),
            }
        )
    return out


# ---------------------------------------------------------------------------
# claim4-controls helpers: pairing shuffle + per-map degeneration diagnostics
# ---------------------------------------------------------------------------


_PREDS_TAG_BY_PASS = {"std": None, "a2r": "a2r", "a2qr": "a2qr", "parity": "parity"}


def _expected_preds_files(rows) -> set[str]:
    """ROW-side expected preds sidecars, derived from the SCORED transfer
    rows — independent of the writer log (codex r4 minor): one file per
    realized P-B primary (fit, pass-tag/variant), per realized P-C holdout
    fit (r5: P-C shares the writer and the resume manifest), + the P-A anchor
    when scored. The filename grammar mirrors ``_fit_and_score``'s
    ``preds_name``. An UNKNOWN ``fit_pass`` label is a loud error (codex r5:
    ``.get()`` silently mapped unknown labels onto the untagged standard
    filename)."""
    out: set[str] = set()
    for r in rows:
        fit = str(r.get("fit", ""))
        if r.get("protocol") == "P-B" and fit.startswith("P-B-holdout-"):
            if r.get("map_variant") == "shufpair":
                tag = "shufpair"
            else:
                label = str(r.get("fit_pass", "std"))
                if label not in _PREDS_TAG_BY_PASS:
                    raise ValueError(
                        f"unknown fit_pass label {label!r} on scored row (fit {fit!r}) — "
                        "the preds filename grammar cannot resolve it (refusing a silent "
                        "map onto the untagged standard filename)"
                    )
                tag = _PREDS_TAG_BY_PASS[label]
            out.add(f"{fit}.{tag}.jsonl" if tag else f"{fit}.jsonl")
        elif r.get("protocol") == "P-C" and fit.startswith("P-C-holdout-"):
            out.add(f"{fit}.jsonl")
        elif fit == "P-A-train-oof":
            out.add("P-A-train-oof.jsonl")
    return out


def _planned_preds_files(
    protocols,
    holdouts,
    map_variants,
    *,
    arm2_adapter: str | None,
    parity_refit_arm7: bool,
    roster=None,
) -> set[str]:
    """PLAN-side expected preds sidecar set, derived BEFORE execution from
    configuration alone — protocols, the resolved holdout list, map variants,
    the adapter pass-tag sequence (:func:`_arm2fix_pass_tags`, the single
    source), and the parity flag (codex r5 MAJOR
    arm2fix-preds-plan-universe-still-derived: an omitted pass shrinks the
    realized rows and the writer log TOGETHER, so only a pre-execution
    universe can catch whole-pass omission). P-C sidecars are in scope: they
    share the writer and the resume manifest."""
    out: set[str] = set()
    protocols = str(protocols)
    holdouts = list(holdouts)
    if len(set(holdouts)) != len(holdouts):
        # multiplicity guard (codex r6 arm2fix-preds-plan-multiplicity): the
        # plan is a SET while writes are per-instance — duplicate holdouts
        # would self-collapse across plan, rows, and writer log; uniqueness
        # must be proven BEFORE set construction (also parse-time rejected).
        raise ValueError(f"duplicate holdouts in the preds plan: {holdouts}")
    mv_list = list(map_variants or [None])
    if "A" in protocols:
        out.add("P-A-train-oof.jsonl")
    if "B" in protocols:
        for holdout in holdouts:
            fit = f"P-B-holdout-{holdout}"
            if arm2_adapter is None:
                for mv in mv_list:
                    tag = "shufpair" if mv == "shufpair" else None
                    out.add(f"{fit}.{tag}.jsonl" if tag else f"{fit}.jsonl")
            else:
                for _label, tag in _arm2fix_pass_tags(
                    arm2_adapter,
                    roster if roster is not None else ROSTER,
                    parity_refit_arm7=parity_refit_arm7,
                ):
                    out.add(f"{fit}.{tag}.jsonl" if tag else f"{fit}.jsonl")
    if "C" in protocols:
        for holdout in holdouts:
            out.add(f"P-C-holdout-{holdout}.jsonl")
    return out


def _assert_preds_manifest_complete(planned, rows, written) -> None:
    """REFUSE the completion sentinel unless the PLANNED sidecar set was
    fully written (codex r5 MAJOR): ``planned`` derives before execution, so
    an omitted pass — which shrinks realized rows and the writer log together
    — still surfaces as missing-planned. The row-derived set is checked as
    defense in depth BOTH ways: scored-but-unwritten (r4) and
    scored-outside-plan (plan/grammar drift). Extra written files beyond
    planned + scored are tolerated; MISSING is the fail-open direction."""
    planned, written = set(planned), set(written)
    if not planned:
        raise RuntimeError(
            "preds plan universe EMPTY on a preds-writing invocation — the planned "
            "sidecar set derives from protocols/holdouts before execution and cannot "
            "be empty (refusing the completion sentinel)"
        )
    missing_planned = sorted(planned - written)
    if missing_planned:
        raise RuntimeError(
            f"preds manifest INCOMPLETE at summary time: {len(missing_planned)} PLANNED "
            f"sidecar(s) were never written — {missing_planned[:4]} (an omitted pass "
            "shrinks rows and the writer log together; the plan universe cannot shrink "
            "with them; refusing the completion sentinel — the seed is not resumable)"
        )
    row_expected = _expected_preds_files(rows)
    missing_rows = sorted(row_expected - written)
    if missing_rows:
        raise RuntimeError(
            f"preds manifest INCOMPLETE at summary time: {len(missing_rows)} scored "
            f"fit(s) have no written preds sidecar — {missing_rows[:4]} (refusing the "
            "completion sentinel; the seed is not resumable)"
        )
    drift = sorted(row_expected - planned)
    if drift:
        raise RuntimeError(
            f"preds plan/grammar DRIFT: {len(drift)} scored fit(s) fall outside the "
            f"pre-execution plan universe — {drift[:4]} (the plan derivation and the "
            "realized pass control flow disagree; refusing the completion sentinel)"
        )


def _summary_preds_gate(res: dict, *, transfer_preds: bool) -> None:
    """The completion-sentinel preds gate — module-level so the producer
    chain (writer log -> summary wrapper -> refuse/resume) is executable
    under test (codex r5): planned-vs-written first, then the row-derived
    defense-in-depth checks."""
    if not transfer_preds:
        return
    _assert_preds_manifest_complete(
        res.get("preds_files_planned") or [],
        res["rows"],
        res.get("preds_files_written", []),
    )


def _seed_output_resume_ok(
    out_dir,
    *,
    commit: str,
    seed: int,
    map_variants,
    arm2_adapter: str | None = None,
    skip_map_fit: bool = False,
    parity_refit_arm7: bool = False,
    arms_only_extra: bool = False,
    a2_sanity_folds: int | None = None,
    transfer_preds: bool = False,
    planned_files=None,
) -> tuple[bool, str]:
    """True iff a prior (behavior, seed) output can satisfy THIS invocation.

    The resume predicate is keyed on code SHA + output schema version + seed
    + map-variant set + EVERY output-affecting arm2fix lane key (adapter /
    skip-map-fit / parity / arms-only-extra / sanity-fold count — all
    recorded in the summary meta; r1 CONCERN arm2fix-resume-key-completeness)
    AND requires every per-seed artifact present — including ≥1 transfer_preds
    sidecar when this invocation writes preds — so a stale, foreign, or
    partial output can never silently satisfy it (a mismatch re-runs the
    seed, loudly). Banked pre-arm2fix outputs carry none of the new keys:
    ``meta.get`` reads None / False for them, which matches exactly the
    flagless invocation — so the legacy claim4 lane's resume behavior is
    unchanged, while an arm2fix invocation can never resume off a legacy
    output (adapter mismatch).
    """
    for name in ("all_arms_spearman.json", "map_diagnostics.json", "readout_pools.json"):
        if not (out_dir / name).exists():
            return False, f"{name} absent"
    try:
        meta = json.loads((out_dir / "all_arms_spearman.json").read_text())["meta"]
    except (OSError, json.JSONDecodeError, UnicodeDecodeError, KeyError, TypeError) as exc:
        return False, f"summary meta unreadable ({type(exc).__name__}: {exc})"
    if transfer_preds:
        # preds-writing invocations verify the realized sidecar MANIFEST the
        # summary recorded at completion (codex r2 minor: one stale/empty
        # sidecar must not read as resume-complete); pre-manifest outputs
        # (meta without the key) fall back to the >=1-sidecar floor.
        manifest = meta.get("transfer_preds_files")
        if isinstance(manifest, list):
            if not manifest:
                return False, "transfer_preds manifest empty (preds-writing invocation)"
            missing = [n for n in manifest if not (out_dir / "transfer_preds" / n).exists()]
            if missing:
                return False, f"transfer_preds manifest files absent: {missing[:4]}"
            # PLANNED-universe validation (codex r6 arm2fix-preds-plan-
            # multiplicity: resume read only the written log): a summary that
            # recorded its pre-execution plan must show manifest == plan — a
            # strict-subset manifest is an incomplete run whose sentinel
            # should have been refused, and an outside-plan manifest file is
            # mixed-generation state. Pre-plan summaries (no key) keep the
            # manifest-only behavior verbatim (banked-output compatibility).
            rec_planned = meta.get("transfer_preds_planned")
            if isinstance(rec_planned, list):
                if not rec_planned:
                    return False, "transfer_preds plan empty (preds-writing invocation)"
                unplanned = sorted(set(rec_planned) - set(manifest))
                if unplanned:
                    return False, f"planned sidecars missing from manifest: {unplanned[:4]}"
                outside = sorted(set(manifest) - set(rec_planned))
                if outside:
                    return False, f"manifest files outside the recorded plan: {outside[:4]}"
                if planned_files is not None and set(rec_planned) != set(planned_files):
                    return False, (
                        f"recorded plan != current plan (recorded {sorted(rec_planned)[:4]}... "
                        f"vs current {sorted(planned_files)[:4]}...)"
                    )
        elif not any((out_dir / "transfer_preds").glob("P-B-holdout-*.jsonl")):
            return False, "transfer_preds sidecars absent (this invocation writes preds)"
    checks = {
        "git_commit": (meta.get("git_commit"), commit),
        "out_schema_version": (meta.get("out_schema_version"), SEED_OUT_SCHEMA_VERSION),
        "seed": (meta.get("seed"), int(seed)),
        "arm2_adapter": (meta.get("arm2_adapter"), arm2_adapter),
        "skip_map_fit": (bool(meta.get("skip_map_fit", False)), bool(skip_map_fit)),
        "parity_refit_arm7": (
            bool(meta.get("parity_refit_arm7", False)),
            bool(parity_refit_arm7),
        ),
        "arms_only_extra": (bool(meta.get("arms_only_extra", False)), bool(arms_only_extra)),
    }
    # sanity-fold count matters only where a sanity read ran (adapter lane);
    # legacy metas record no a2_sanity_folds and no adapter — .get(None) == None
    # keeps them resumable under flagless invocations exactly as before.
    if arm2_adapter is not None or meta.get("arm2_adapter") is not None:
        checks["a2_sanity_folds"] = (meta.get("a2_sanity_folds"), a2_sanity_folds)
    for field, (got, want) in checks.items():
        if got != want:
            return False, f"{field} mismatch (recorded {got!r} != current {want!r})"
    # map-variant membership is compared as a SET (review r2 item 6): the
    # variant list is unordered configuration — a reordered but identical set
    # must resume, a different set must re-run.
    rec_mv = meta.get("map_variants")
    want_mv = list(map_variants or [])
    if not isinstance(rec_mv, list) or {str(v) for v in rec_mv} != {str(v) for v in want_mv}:
        return False, f"map_variants set mismatch (recorded {rec_mv!r} != current {want_mv!r})"
    return True, "match"


def _remove_stale_summary(resume_dir) -> None:
    """Remove a NOT-resumable seed's stale completion sentinel BEFORE the
    rerun starts (codex r6/r7: a same-commit stale summary left in place can
    validate mixed-generation companions if the rerun dies mid-way — the
    rerun's own summary is written LAST, so from removal until then the seed
    reads as incomplete, which it is). Companions are left for forensics."""
    (resume_dir / "all_arms_spearman.json").unlink()


def _write_companions_then_summary(out_dir, res: dict, write_summary_fn) -> None:
    """Companion artifacts FIRST, the validated summary LAST (review r2 item 6).

    ``all_arms_spearman.json`` is the per-seed COMPLETION SENTINEL: the resume
    predicate (:func:`_seed_output_resume_ok`) keys its identity checks on the
    summary's meta, so the summary must be the LAST artifact written — an
    interrupt anywhere in this function then leaves either no summary or a
    prior-generation summary (whose git_commit/schema keys fail the
    predicate), never a passing predicate over mixed-generation artifacts.
    """
    (out_dir / "map_diagnostics.json").write_text(json.dumps(res["map_diagnostics"], indent=1))
    (out_dir / "readout_pools.json").write_text(
        json.dumps({"pools": res["pools"], "fit_reports": res["fit_reports"]}, indent=1)
    )
    write_summary_fn()


def _behavior_out_dir(args, behavior: str):
    """Behavior output dir, seed-keyed (`<behavior>/seed<S>/`) in --seeds mode.

    ``args.out_subdir`` is set by main()'s seed loop ("" in legacy mode, so
    every flagless run's paths are byte-identical).
    """
    sub = getattr(args, "out_subdir", "")
    d = args.out_root / behavior
    return d / sub if sub else d


def pairing_shuffle_perm(n_gen: int, n_total: int, *, seed: int):
    """ONE within-component row permutation for the pairing-shuffle control.

    The ADD pool is ``[generic | eliciting]`` along the column axis; the
    generic block (``[:n_gen]``) and the eliciting block (``[n_gen:]``) are
    permuted SEPARATELY (component marginal composition preserved), and the
    SAME permutation is applied to the answer side across all layers (a pair
    is a row of the dataset; per-layer shuffles would not be a dataset).
    rng namespace ``[1739, 21, seed]`` (plan §11; module list-seed convention).

    Returns ``(perm, fingerprints)`` — ``perm`` a (n_total,) int64 index array
    with component structure ``[perm_gen | n_gen + perm_elic]``.
    """
    import numpy as np

    if not 0 < n_gen < n_total:
        raise ValueError(f"pairing shuffle needs 0 < n_gen < n_total, got {n_gen}/{n_total}")
    rng = np.random.default_rng([1739, 21, int(seed)])
    perm_gen = rng.permutation(n_gen)
    perm_elic = rng.permutation(n_total - n_gen)
    perm = np.concatenate([perm_gen, n_gen + perm_elic]).astype(np.int64)
    fp = {
        "rng_namespace": [1739, 21, int(seed)],
        "n_generic": int(n_gen),
        "n_eliciting": int(n_total - n_gen),
        "perm_generic_sha256": hashlib.sha256(np.ascontiguousarray(perm_gen).tobytes()).hexdigest()[
            :16
        ],
        "perm_eliciting_sha256": hashlib.sha256(
            np.ascontiguousarray(perm_elic).tobytes()
        ).hexdigest()[:16],
        "frac_moved_generic": float((perm_gen != np.arange(n_gen)).mean()),
        "frac_moved_eliciting": float((perm_elic != np.arange(n_total - n_gen)).mean()),
    }
    return perm, fp


def shufpair_structural_check(perm, n_gen: int, n_total: int) -> dict:
    """HARD structural manipulation check for the pairing shuffle (plan §6).

    Verifies (fail-loud RuntimeError, never a silent pass):
    (1) ``perm`` is a bijection on [0, n_total);
    (2) within-component: generic slots map into the generic block, eliciting
        slots into the eliciting block (no cross-component moves);
    (3) non-identity target reassignment — globally, and per component when
        the component is large enough for identity to be a bug signal rather
        than a legal draw (size >= 4).
    Sharing across layers is structural by construction (ONE perm array is
    applied to axis 1 of the (Ly, n, d) answer tensor); the recorded
    fingerprints (sha over the perm bytes) make it audit-able post hoc.
    """
    import numpy as np

    perm = np.asarray(perm, dtype=np.int64)
    if perm.shape != (n_total,):
        raise RuntimeError(f"shufpair perm shape {perm.shape} != ({n_total},)")
    if not np.array_equal(np.sort(perm), np.arange(n_total)):
        raise RuntimeError("shufpair perm is NOT a bijection on [0, n_total)")
    gen, elic = perm[:n_gen], perm[n_gen:]
    if gen.size and (gen.max() >= n_gen):
        raise RuntimeError("shufpair perm moves generic slots OUT of the generic block")
    if elic.size and (elic.min() < n_gen):
        raise RuntimeError("shufpair perm moves eliciting slots OUT of the eliciting block")
    ident = np.arange(n_total)
    if np.array_equal(perm, ident):
        raise RuntimeError("shufpair perm is the identity — no pairing was destroyed")
    if n_gen >= 4 and np.array_equal(gen, ident[:n_gen]):
        raise RuntimeError("shufpair generic component is identity — component not shuffled")
    if (n_total - n_gen) >= 4 and np.array_equal(elic, ident[n_gen:] - 0):
        raise RuntimeError("shufpair eliciting component is identity — component not shuffled")
    return {
        "within_component_bijection": True,
        "shared_across_layers": "structural (one perm applied to the (Ly, n, d) column axis)",
        "non_identity": True,
    }


def _weight_spectrum(w, *, stride: int = 4) -> list[dict]:
    """Strided-layer singular-value profile of a fitted map's weight tensor.

    The 'effective spectrum' degeneration diagnostic (plan §4 P0.2): top
    singular values + Frobenius/spectral norms + participation-ratio
    effective rank. Values-only LAPACK svd (no U/V), fp64. Computed on a
    LAYER STRIDE (every ``stride``-th layer + the last): a full-D svdvals is
    ~25 s/layer at D=3584, so all 28 layers x 2 maps would add ~24 min per
    seed-invocation against the plan-§9 0.6-0.8 h row (+50-65%%); the strided
    8-layer profile (~3-4 min) keeps the true-vs-shufpair degeneration read
    at matched layers within the sized budget.
    """
    import numpy as np

    n_layers = int(w.shape[0])
    layer_idx = sorted(set(range(0, n_layers, max(1, stride))) | {n_layers - 1})
    out = []
    for li in layer_idx:
        s = np.linalg.svd(np.asarray(w[li], dtype=np.float64), compute_uv=False)
        s2 = s**2
        out.append(
            {
                "layer_idx": int(li),
                "top_svals": [float(x) for x in s[:32]],
                "sval_quantiles": {str(q): float(np.quantile(s, q)) for q in (0.5, 0.9, 0.99)},
                "frobenius": float(np.sqrt(s2.sum())),
                "spectral": float(s[0]),
                "eff_rank_participation": float(s2.sum() ** 2 / max((s2**2).sum(), 1e-300)),
            }
        )
    return out


def _map_output_variance(mapfit, x_sub, y_sub) -> dict:
    """Mapped-output variance vs target variance on a pool subsample (per layer)."""
    import numpy as np

    from explore_persona_space.experiments.issue_1739 import fits

    pred = fits.apply_map(x_sub, mapfit)
    return {
        "n_subsample": int(x_sub.shape[1]),
        "pred_var_per_layer": [float(np.var(pred[li])) for li in range(pred.shape[0])],
        "target_var_per_layer": [float(np.var(y_sub[li])) for li in range(y_sub.shape[0])],
    }


def _identity_bias_recon(ib_means, z_ev, za_ev, rungs) -> dict:
    """Identity+learned-bias baseline scored on the P-B eval contexts.

    The (a) half of the standing mapping-baselines pair, computed on the SAME
    eval contexts the map recon (`_eval_rung_reconstruction`) scores — via
    the CANONICAL ``analysis.mapping_baselines.identity_bias_predict`` helper
    called per layer on the stored pool-mean sufficient statistics
    (``ib_means``: single-row train arrays whose mean IS the pool mean, so
    the helper's ``b = mean(y_train − x_train)`` equals pool-mean(y_w) −
    pool-mean(x_w) EXACTLY — the memory-motivated substitution keeps the
    staged frees while routing through the canonical formula + validation).
    Means are permutation-invariant, so ONE bias serves both map variants
    (recorded in the note field). Per-layer pooled R² + kNN retrieval, plus
    per-rung R².
    """
    import numpy as np

    from explore_persona_space.analysis.mapping_baselines import (
        identity_bias_predict,
        knn_retrieval,
    )
    from explore_persona_space.experiments.issue_1739 import fits
    from explore_persona_space.experiments.issue_1739.constants import KNN_KS

    x_mean = np.asarray(ib_means["x_mean"], dtype=np.float64)
    y_mean = np.asarray(ib_means["y_mean"], dtype=np.float64)
    z_ev = np.asarray(z_ev)
    pred = np.stack(
        [
            identity_bias_predict(x_mean[li][None, :], y_mean[li][None, :], z_ev[li])
            for li in range(z_ev.shape[0])
        ]
    )
    per_layer = []
    for li in range(pred.shape[0]):
        per_layer.append(
            {
                "layer_idx": li,
                "r2_identity_bias": float(fits.r2_pooled(pred[li], za_ev[li])),
                "knn": {
                    metric: knn_retrieval(pred[li], za_ev[li], ks=KNN_KS, metric=metric)
                    for metric in ("euclidean", "cosine")
                },
            }
        )
    labels = np.asarray([str(r) for r in rungs])
    per_rung = {}
    for rung in sorted(set(labels.tolist())):
        sel = np.flatnonzero(labels == rung)
        per_rung[rung] = {
            "n_rows": int(sel.size),
            "r2_identity_bias_per_layer": [
                float(fits.r2_pooled(pred[li, sel], za_ev[li, sel])) for li in range(pred.shape[0])
            ],
        }
    return {
        "per_layer": per_layer,
        "per_rung": per_rung,
        "knn_ks": list(KNN_KS),
        "note": "bias = pool-mean(y_w - x_w); permutation-invariant, shared across map "
        "variants; baseline the fitted map must beat (mapping-baselines pair (a))",
    }


def _assert_arm4_variant_identity(rows: list[dict], behavior: str) -> None:
    """In-run pairing check: arm4 is map-INDEPENDENT, so its per-seed P-B rows
    must be bit-identical across map variants (same readout rows, same dv,
    same z_ctx, deterministic in-process BLAS). A mismatch means the variant
    passes were NOT byte-identical outside the map — fail loud (the fold
    script re-asserts this off the persisted rows)."""
    by_key: dict[tuple, dict[str, float]] = {}
    for r in rows:
        if r.get("arm") != "arm4_ridge_ctx" or r.get("protocol") != "P-B":
            continue
        mv = r.get("map_variant")
        if mv is None:
            continue
        by_key.setdefault((r.get("fit"), r.get("eval_rung")), {})[mv] = float(r["rho_frozen"])
    bad = []
    for k, per_mv in sorted(by_key.items()):
        if {"true", "shufpair"} <= set(per_mv) and per_mv["true"] != per_mv["shufpair"]:
            bad.append((k, per_mv["true"], per_mv["shufpair"]))
    if bad:
        raise RuntimeError(
            f"[{behavior}] arm4 rows DIFFER across map variants (pairing check failed) — "
            f"first offenders: {bad[:3]}"
        )


# ---------------------------------------------------------------------------
# per-behavior scoring
# ---------------------------------------------------------------------------


def fit_linear_add_map(
    args,
    loaded,
    variant: str,
    layers: list[int],
    gen_sink: dict | None = None,
    claim4_sink: dict | None = None,
    skip_map_fit: bool = False,
):
    """The r2fair ADD-condition LINEAR map/whitening recipe with staged frees.

    Same pool composition + seed + reviewed compose/fit path as
    ``issue1739_result2fair_score.fit_add_maps`` restricted to kind=linear
    (map weights are never persisted, so the deterministic re-fit IS the
    reuse). Deltas are pure memory hygiene for the 128 GB cgroup (pilot r1
    OOM): the staged u_store arrays are freed the moment the pool is built,
    and each fp16 pool array is freed as soon as its fp64 whitened copy
    exists — peak drops ~40 GiB with bit-identical outputs.

    ``gen_sink`` (P-C, per-holdout map refits): when a dict is passed, the
    WHITENED generic pool block is copied out before the fp64 pool copies are
    freed (keys ``x_gen_w`` / ``y_gen_w`` / ``n_gen``) — the ADD pool is
    [generic | eliciting] along axis 1, so the leading ``n_gen`` columns of
    the whitened pool ARE the whitened generic pool. The P-C map pools reuse
    these verbatim (identical generic component across holdouts).

    ``claim4_sink`` (claim4-controls round, plan §4 P0.2): when a dict is
    passed, the sink is filled with (a) ``ib_means`` — the per-layer pool
    means (Ly, d) x_mean/y_mean, the transient-free SUFFICIENT STATISTICS the
    canonical ``mapping_baselines.identity_bias_predict`` consumes at recon
    time (see :func:`_identity_bias_recon`);
    (b) when ``claim4_sink["want_shufpair"]`` — after the TRUE map fit (and
    the gen_sink copy) the answer side ``y_w`` is permuted IN PLACE per layer
    by ONE within-component pairing-shuffle permutation
    (:func:`pairing_shuffle_perm`, rng ``[1739, 21, seed]``; structural
    check :func:`shufpair_structural_check` — hard fail-loud) and a SECOND
    map is fit on the pairing-destroyed pool: ``mapfit_shuf`` +
    ``diag_shufpair``. Whitening / pools / fit recipe are byte-identical to
    the true pass — pairing is the ONLY varied factor. Per-map degeneration
    diagnostics (selected-λ profile, mapped-output variance) ride both diags.
    """
    import numpy as np

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
    mapfit = None
    map_lam_sink: list[dict] = []
    if skip_map_fit:
        # arm2fix D1 (--skip-map-fit): the resolved roster consumes NO map (the
        # parse-time guard refuses otherwise), so the expensive linear-map fit
        # is skipped OUTRIGHT. Whitening above still ran: arm2 operates in
        # whitened coordinates, and the seed-keyed whitening rng reproduces the
        # banked space (plan §4 D1 (b); gate-2 input-sha comparison).
        _log_rss("map-fit-skipped")
    else:
        _log_rss("map-fit-true-entry")
        with fits.capture_selected_lambdas(map_lam_sink):
            mapfit = _fit_map(_fitmap_ns(args), x_w, y_w)
        _log_rss("map-fit-true-done")
    if gen_sink is not None:
        n_gen = int(pool_meta["add_n_generic"])
        # basic slices are VIEWS keeping the whole pool alive — copy, then free
        gen_sink["x_gen_w"] = np.ascontiguousarray(x_w[:, :n_gen])
        gen_sink["y_gen_w"] = np.ascontiguousarray(y_w[:, :n_gen])
        gen_sink["n_gen"] = n_gen
    diag = {
        **(mapfit.diagnostics if mapfit is not None else {}),
        "map_kind": "linear",
        "map_source": (
            "skipped (--skip-map-fit: roster has no map-consuming arm)" if skip_map_fit else "refit"
        ),
        "map_fit_s": round(time.time() - t1, 1),
        "whitening_fit_s": wh_s,
        "n_u": int(n_u),
        "u_pool_label": u_label,
        **pool_meta,
    }
    if claim4_sink is not None:
        # identity+bias SUFFICIENT STATISTICS (Ly, d): the per-layer pool
        # means feed the canonical analysis.mapping_baselines
        # identity_bias_predict at recon time (its bias b is exactly
        # mean(y_train - x_train) = y_mean - x_mean), without keeping the
        # whole (Ly, n, D) pools alive across the staged frees.
        claim4_sink["ib_means"] = {
            "x_mean": x_w.mean(axis=1),
            "y_mean": y_w.mean(axis=1),
        }
        m_sub = min(2048, x_w.shape[1])
        sub = slice(x_w.shape[1] - m_sub, x_w.shape[1])
        diag["map_variant"] = "true"
        if mapfit is not None:
            diag["map_selected_lambdas"] = map_lam_sink
            diag["mapped_output_variance"] = _map_output_variance(mapfit, x_w[:, sub], y_w[:, sub])
        if claim4_sink.get("want_shufpair") and skip_map_fit:
            raise RuntimeError(
                "--skip-map-fit is incompatible with --map-variants shufpair (a "
                "pairing-shuffled map cannot be produced without fitting one) — "
                "the parse-time validator should have refused this"
            )
        if claim4_sink.get("want_shufpair"):
            n_gen = int(pool_meta["add_n_generic"])
            perm, perm_fp = pairing_shuffle_perm(n_gen, y_w.shape[1], seed=args.seed)
            structural = shufpair_structural_check(perm, n_gen, y_w.shape[1])
            # in-place per-layer permute: transient = ONE layer (~0.7 GiB at
            # production shape), never a second whole-pool fp64 copy.
            for li in range(y_w.shape[0]):
                y_w[li] = y_w[li][perm]
            t2 = time.time()
            _log_rss("map-fit-shufpair-entry")
            shuf_lam_sink: list[dict] = []
            with fits.capture_selected_lambdas(shuf_lam_sink):
                mapfit_shuf = _fit_map(_fitmap_ns(args), x_w, y_w)
            _log_rss("map-fit-shufpair-done")
            claim4_sink["mapfit_shuf"] = mapfit_shuf
            claim4_sink["diag_shufpair"] = {
                **mapfit_shuf.diagnostics,
                "map_kind": "linear",
                "map_source": "refit-pairing-shuffled",
                "map_variant": "shufpair",
                "map_fit_s": round(time.time() - t2, 1),
                "whitening_fit_s": wh_s,
                "n_u": int(n_u),
                "u_pool_label": u_label,
                **pool_meta,
                "pairing_shuffle": {**perm_fp, "structural_check": structural},
                "map_selected_lambdas": shuf_lam_sink,
                "mapped_output_variance": _map_output_variance(
                    mapfit_shuf, x_w[:, sub], y_w[:, sub]
                ),
            }
            _log(
                f"[map] shufpair ADD map fit: {claim4_sink['diag_shufpair']['map_fit_s']}s "
                f"(frac_moved gen {perm_fp['frac_moved_generic']:.4f} / "
                f"elic {perm_fp['frac_moved_eliciting']:.4f})"
            )
    del x_w, y_w
    _log(f"[map] linear ADD map fit: whitening {wh_s}s, map {diag['map_fit_s']}s")
    return wh, mapfit, diag, u_label, n_u


def prepare_behavior(args, behavior: str, layers: list[int]) -> SimpleNamespace:
    """Shared per-behavior setup: tables + whitening + map + merged fp64 table +
    dataset roster + frozen layers. Pure code motion out of :func:`run_behavior`
    (2026-08-06 factorial round) so the extraction-factorial leg
    (`issue1739_r2v2_factorial.py`) reuses the identical pipeline state without
    re-running the transfer fits."""
    import numpy as np

    from explore_persona_space.experiments.issue_1739 import fits
    from scripts.issue1739_jobd_r2aug import (
        LMAX,
        committed_frozen,
        load_behavior,
    )
    from scripts.issue1739_result2fair_score import (
        _wc_eval_mask,
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

    # map + whitening: IDENTICAL across P-A and P-B (linear ADD recipe).
    # P-C refits the MAP per holdout but keeps THIS whitening frozen (fit on
    # the ADD pool = generic + eliciting-train — holdout-clean by
    # construction: no eval-rung / OOD dataset row enters the ADD pool), so
    # the whitened generic block is retained for the per-holdout map pools.
    keep_gen = "C" in str(getattr(args, "protocols", ""))
    gen_sink: dict | None = {} if keep_gen else None
    map_variants = getattr(args, "map_variants", None)
    claim4_sink: dict | None = None
    if map_variants is not None:
        claim4_sink = {"want_shufpair": "shufpair" in map_variants}
    wh, mapfit, map_diag_linear, u_label, n_u = fit_linear_add_map(
        args,
        loaded,
        variant,
        layers,
        gen_sink=gen_sink,
        claim4_sink=claim4_sink,
        skip_map_fit=bool(getattr(args, "skip_map_fit", False)),
    )
    map_diags = {"linear": map_diag_linear}
    mapfit_shuf = None
    ib_means = None
    if claim4_sink is not None:
        ib_means = claim4_sink.get("ib_means")
        if claim4_sink.get("want_shufpair"):
            mapfit_shuf = claim4_sink["mapfit_shuf"]
            map_diags["linear_shufpair"] = claim4_sink["diag_shufpair"]
        # effective-spectrum degeneration diagnostic per fitted map (values-only
        # svd on the (Ly, d, d) weight tensor — pools are already freed here).
        # --skip-map-fit: no map exists, so there is no spectrum to compute.
        if mapfit is not None:
            t_sp = time.time()
            map_diags["linear"]["weight_spectrum"] = _weight_spectrum(mapfit.w)
            if mapfit_shuf is not None:
                map_diags["linear_shufpair"]["weight_spectrum"] = _weight_spectrum(mapfit_shuf.w)
            _log(f"[map] weight spectra computed in {time.time() - t_sp:.0f}s")

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
    datasets = dataset_roster(loaded, tbl_ood, elic_cell, base_ev=base_ev, base_ood=base_ood)
    ds_by_name = {d.name: d for d in datasets}
    eval_datasets = [d.name for d in datasets if d.name != "train"]
    _log(
        f"[{behavior}] roster: {[(d.name, len(np.asarray(d.rows))) for d in datasets]} | "
        f"wc train/eval {len(wc_train_rows)}/{len(wc_eval_rows)} | pvsynth {len(ids_pv)}"
    )

    # arm20 has no committed train row — its frozen layer is MATCHED to the
    # companion arm's committed layer (MATCHED_FROZEN_COMPANIONS; the fair
    # round's ("arm20_shuffled_map_ridge", "arm7_map_ridge_pred") precedent).
    # --parity-refit-arm7 scores arm7 in its own row-matched pass even when
    # --arms-only-extra removed it from ROSTER, so its committed frozen layer
    # must resolve too.
    roster_for_frozen = ROSTER
    if getattr(args, "parity_refit_arm7", False) and "arm7_map_ridge_pred" not in ROSTER:
        roster_for_frozen = ROSTER + ("arm7_map_ridge_pred",)
    roster_frozen = tuple(a for a in roster_for_frozen if a not in MATCHED_FROZEN_COMPANIONS)
    frozen, frozen_src = committed_frozen(args, loaded, behavior, variant, layers, roster_frozen)
    for a, ref in MATCHED_FROZEN_COMPANIONS.items():
        if a in ROSTER:
            frozen[a] = frozen[ref]
            frozen_src += f"; {a}@{ref}-committed-layer (matched-companion convention)"

    # merged-row group keys, parallel to ctx_ids (claim4 preds carry them so
    # the P2 paired context-bootstrap can resample by group-hash group).
    groups_all = (
        [str(g) for g in loaded.tbl.groups]
        + [str(g) for g in loaded.tbl_wc.groups]
        + [str(g) for g in loaded.tbl_ev.groups]
        + ([str(g) for g in tbl_ood.groups] if tbl_ood is not None else [])
    )
    if len(groups_all) != len(ctx_ids):
        raise AssertionError(f"groups_all {len(groups_all)} != ctx_ids {len(ctx_ids)}")
    groups_pv = [str(g) for g in tbl_pv.groups]

    return SimpleNamespace(
        loaded=loaded,
        tbl_pv=tbl_pv,
        tbl_ood=tbl_ood,
        ood_note=ood_note,
        variant=variant,
        wh=wh,
        mapfit=mapfit,
        mapfit_shuf=mapfit_shuf,
        ib_means=ib_means,
        groups_all=groups_all,
        groups_pv=groups_pv,
        map_diags=map_diags,
        u_label=u_label,
        n_u=n_u,
        n_tr=n_tr,
        n_wc=n_wc,
        n_ev=n_ev,
        n_ood=n_ood,
        base_wc=base_wc,
        base_ev=base_ev,
        base_ood=base_ood,
        z_ctx=z_ctx,
        z_ans=z_ans,
        dv_raw=dv_raw,
        ctx_ids=ctx_ids,
        rb_w=rb_w,
        z_pv=z_pv,
        za_pv=za_pv,
        dv_pv=dv_pv,
        ids_pv=ids_pv,
        wc_eval_rows=wc_eval_rows,
        wc_train_rows=wc_train_rows,
        ids_wc_eval=ids_wc_eval,
        lmax=lmax,
        elic_cell=elic_cell,
        datasets=datasets,
        ds_by_name=ds_by_name,
        eval_datasets=eval_datasets,
        frozen=frozen,
        frozen_src=frozen_src,
        gen_sink=gen_sink,
        t0=t0,
    )


def run_behavior(args, behavior: str, layers: list[int]) -> dict:
    import numpy as np

    from explore_persona_space.experiments.issue_1739 import arms, fits
    from scripts.issue1739_fits import _eval_rung_reconstruction
    from scripts.issue1739_jobd_r2aug import (
        _free_cuda,
        per_layer_rows_for,
        transfer_rows_for,
    )
    from scripts.issue1739_result2fair_score import _wc_fold_ids

    prep = prepare_behavior(args, behavior, layers)
    loaded, tbl_pv, tbl_ood, ood_note = prep.loaded, prep.tbl_pv, prep.tbl_ood, prep.ood_note
    variant, wh, mapfit, map_diags = prep.variant, prep.wh, prep.mapfit, prep.map_diags
    mapfit_shuf, ib_means = prep.mapfit_shuf, prep.ib_means
    groups_all, groups_pv = prep.groups_all, prep.groups_pv
    u_label, n_u = prep.u_label, prep.n_u
    z_ctx, z_ans, dv_raw, ctx_ids, rb_w = (
        prep.z_ctx,
        prep.z_ans,
        prep.dv_raw,
        prep.ctx_ids,
        prep.rb_w,
    )
    z_pv, za_pv, dv_pv, ids_pv = prep.z_pv, prep.za_pv, prep.dv_pv, prep.ids_pv
    wc_eval_rows, wc_train_rows = prep.wc_eval_rows, prep.wc_train_rows
    lmax, elic_cell = prep.lmax, prep.elic_cell
    datasets, ds_by_name, eval_datasets = prep.datasets, prep.ds_by_name, prep.eval_datasets
    frozen, frozen_src, t0 = prep.frozen, prep.frozen_src, prep.t0
    gen_sink = prep.gen_sink
    del prep  # locals hold the (large) references from here on

    rows_all: list[dict] = []
    skips_all: list[dict] = []
    per_layer_all: list[dict] = []
    fit_reports: list[dict] = []
    pools_record: list[dict] = []
    # writer-side preds-file log (codex r3 minor arm2fix-preds-manifest-
    # universe): the resume manifest must be the set of files the run
    # INTENDED and wrote — recorded at each write site — never a directory
    # glob at summary time (a glob makes any surviving strict subset define
    # its own universe and read as resume-complete).
    preds_files_written: list[str] = []
    # PLAN-side expected preds universe, resolved BEFORE any fit executes
    # (codex r5 MAJOR arm2fix-preds-plan-universe-still-derived): derived from
    # configuration alone, so a whole omitted pass — which drops its rows and
    # writer-log entries together — still surfaces at the summary gate.
    preds_files_planned = _planned_preds_files(
        args.protocols,
        list(args.pb_holdouts or eval_datasets),
        getattr(args, "map_variants", None),
        arm2_adapter=getattr(args, "arm2_adapter", None),
        parity_refit_arm7=bool(getattr(args, "parity_refit_arm7", False)),
    )
    kwargs = {"n_boot": args.n_boot} if args.n_boot else {}

    def _fit_and_score(
        protocol: str,
        fit_label: str,
        readout_rows,
        dv_z,
        eval_specs: list[tuple[str, object]],
        extra_prov: dict,
        mapfit_use=None,
        map_train_ids: set | None = None,
        roster: tuple[str, ...] | None = None,
        map_variant: str | None = None,
        preds_tag: str | None = None,
        arm_prov: dict[str, dict] | None = None,
    ) -> None:
        """One full-union transfer fit + per-rung evaluation (all protocols).

        ``eval_specs`` = [(rung_label, merged_rows | ("pv", None))]; pvsynth
        rows come from the separate whitened pvsynth arrays. ``mapfit_use``
        (P-C, claim4 shufpair) swaps the frozen behavior-level map;
        ``map_train_ids`` (P-C) additionally hard-asserts the MAP pool's
        eliciting context ids are disjoint from every eval setting.
        ``roster`` (claim4 shufpair pass) restricts the arm roster for THIS
        fit (default: the module ROSTER); ``map_variant`` rides every row's
        provenance; ``preds_tag`` disambiguates the per-fit preds filename
        across variant passes (the two passes share fit labels by design —
        the repro join subsets on map_variant). ``arm_prov`` (arm2fix)
        merges PER-ARM provenance (adapter tag + train-row id-hash + count)
        into that arm's transfer + per-layer rows — a pass-level extra_prov
        cannot distinguish arms sharing one pass.
        """
        readout_rows = np.asarray(readout_rows, dtype=np.int64)
        # entry crumb pairs with fit-done-<label> below: brackets the transfer
        # fit + arm scoring (arm2/arm20 included) for OOM localization (r4)
        _log_rss(f"fit-entry-{fit_label}" + (f"-{map_variant}" if map_variant else ""))
        _assert_well_posed(len(readout_rows), loaded.dim, f"{behavior}/{fit_label}")
        roster_use = ROSTER if roster is None else tuple(roster)
        ev_z_parts, ev_za_parts, ev_dv_parts, ev_rung_parts = [], [], [], []
        # ORDERED eval-context ids, parallel to the concatenated columns (the
        # eval_id_sets below are SETS — leakage asserts only, order destroyed).
        ev_ctx_parts: list[list[str]] = []
        ev_grp_parts: list[list[str]] = []
        eval_id_sets: dict[str, set] = {}
        for label, rows in eval_specs:
            if rows is None:  # pvsynth
                ev_z_parts.append(z_pv)
                ev_za_parts.append(za_pv)
                ev_dv_parts.append(dv_pv)
                ev_rung_parts.append(np.asarray([label] * z_pv.shape[1]))
                ev_ctx_parts.append([str(c) for c in tbl_pv.ctx_order])
                ev_grp_parts.append(list(groups_pv))
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
            ev_ctx_parts.append([ctx_ids[i] for i in rows])
            ev_grp_parts.append([groups_all[i] for i in rows])
            eval_id_sets[label] = {ctx_ids[i] for i in rows}
        z_ev = np.concatenate(ev_z_parts, axis=1)
        za_ev = np.concatenate(ev_za_parts, axis=1)
        dv_ev = np.concatenate(ev_dv_parts)
        rungs_ev = np.concatenate(ev_rung_parts)
        ctx_order_ev = [c for part in ev_ctx_parts for c in part]
        grp_order_ev = [g for part in ev_grp_parts for g in part]
        if len(ctx_order_ev) != z_ev.shape[1]:
            raise AssertionError(
                f"[{behavior}] {fit_label}: eval ctx-id order {len(ctx_order_ev)} != "
                f"{z_ev.shape[1]} eval columns"
            )
        del ev_z_parts, ev_za_parts, ev_ctx_parts, ev_grp_parts

        leak = _leakage_assert({ctx_ids[i] for i in readout_rows}, eval_id_sets, fit_label)
        map_leak = None
        if map_train_ids is not None:
            map_leak = _leakage_assert(map_train_ids, eval_id_sets, f"{fit_label}-map")
        cell = fits.BudgetCell(
            row_idx=readout_rows,
            fold_ids=np.zeros(len(readout_rows), dtype=np.int64),
            n_folds=1,
            budget_l=lmax,
            draw=args.draw,
            seed=args.seed,
            fold_scheme=f"r2v2-{protocol}-full-union",
        )
        mf = mapfit_use if mapfit_use is not None else mapfit
        data = arms.CellData(
            z_ctx=z_ctx, z_ans=z_ans, dv=dv_z, rb=rb_w, mapfit=mf, layers=tuple(layers)
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
            **({"map_variant": map_variant} if map_variant is not None else {}),
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
                roster_use,
                device=args.device,
                n_boot=args.n_boot,
                min_n=args.min_n,
            )
        wall = round(time.time() - t1, 1)
        if arm_prov:
            for r in rows:
                r.update(arm_prov.get(r.get("arm"), {}))
        if getattr(args, "transfer_preds", False):
            # Per-(arm, eval context) frozen-layer predictions, via the SAME
            # reviewed helper the bare-query / wcrung legs use. Written BEFORE
            # rows_all is extended so a crash cannot leave aggregate rows whose
            # per-context sidecar is missing; one file per FIT (the unit here),
            # truncate-and-replace, so a re-run of a fit overwrites exactly its
            # own rows. `rung` rides the generic label column, so any per-rung
            # subset read — a paired bootstrap over shared eval contexts, an OOD
            # scatter — is a pure post-hoc read of this file. claim4: the
            # shufpair pass writes `<fit>.shufpair.jsonl` (fit labels are shared
            # across variant passes by design); `group` rides along so the P2
            # context bootstrap can resample by group-hash group.
            preds_name = f"{fit_label}.{preds_tag}.jsonl" if preds_tag else f"{fit_label}.jsonl"
            preds_files_written.append(preds_name)
            preds_labels = {"rung": [str(x) for x in rungs_ev]}
            if map_variant is not None:
                preds_labels["group"] = grp_order_ev
            arms.write_preds_jsonl(
                _behavior_out_dir(args, behavior) / "transfer_preds" / preds_name,
                arms.transfer_preds_rows(
                    scores,
                    dv_ev,
                    ctx_order_ev,
                    frozen,
                    provenance={**prov, "n_eval_pooled": len(ctx_order_ev)},
                    layers=tuple(layers),
                    labels=preds_labels,
                ),
            )
        pl_rows = per_layer_rows_for(
            scores, dv_ev, frozen, {**prov, "eval_rung": "all"}, layers, frozen_src
        )
        if arm_prov:
            for r in pl_rows:
                r.update(arm_prov.get(r.get("arm"), {}))
        per_layer_all.extend(pl_rows)
        rows_all.extend(rows)
        skips_all.extend(skips)
        report = {
            "protocol": protocol,
            "fit": fit_label,
            "n_readout": int(len(readout_rows)),
            "d": int(loaded.dim),
            "well_posed": f"n_train {len(readout_rows)} > d {loaded.dim}",
            "leakage": leak,
            "ridge_lambda_diagnostics": lam_sink,
            # --skip-map-fit: no map exists -> no map-reconstruction read (the
            # roster is guaranteed map-free by the parse-time guard).
            "recon": (
                _eval_rung_reconstruction(
                    mf, z_ev, za_ev, rungs=[str(r) for r in rungs_ev], knn=True
                )
                if mf is not None
                else "skipped (--skip-map-fit: no fitted map)"
            ),
            "fit_wall_s": wall,
        }
        if map_variant is not None:
            report["map_variant"] = map_variant
            if ib_means is not None:
                # mapping-baselines pair (a) on the SAME eval contexts the map
                # recon scores (plan §6) — one bias serves both variants.
                report["recon_identity_bias"] = _identity_bias_recon(
                    ib_means, z_ev, za_ev, [str(r) for r in rungs_ev]
                )
        if map_leak is not None:
            report["map_leakage"] = map_leak
        fit_reports.append(report)
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
        if getattr(args, "transfer_preds", False):
            # The P-A in-distribution anchor rung — same contract as the
            # _fit_and_score sidecar above; its contexts are the eliciting
            # cell's own rows, in the order scores_el was sliced to.
            preds_files_written.append("P-A-train-oof.jsonl")
            arms.write_preds_jsonl(
                _behavior_out_dir(args, behavior) / "transfer_preds" / "P-A-train-oof.jsonl",
                arms.transfer_preds_rows(
                    scores_el,
                    dv_el,
                    [ctx_ids[i] for i in np.asarray(elic_cell.row_idx, dtype=np.int64)],
                    frozen,
                    provenance={**prov_tr, "n_eval_pooled": int(n_el)},
                    layers=tuple(layers),
                    labels={"rung": ["train"] * n_el},
                ),
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
    # claim4-controls: with --map-variants the P-B pass runs ONCE PER VARIANT —
    # "true" scores the full roster against the frozen true map; "shufpair"
    # scores {arm4, arm6, arm7} against the pairing-shuffled refit. Pools /
    # whitening / eval specs are byte-identical across passes (pairing is the
    # ONLY varied factor); pool records are written on the first pass only.
    if "B" in args.protocols:
        holdouts = args.pb_holdouts or eval_datasets
        unknown = sorted(set(holdouts) - set(eval_datasets))
        if unknown:
            raise ValueError(f"--pb-holdouts {unknown} not in eval datasets {eval_datasets}")
        mv_list: list[str | None] = list(getattr(args, "map_variants", None) or [None])
        if "shufpair" in mv_list and mapfit_shuf is None:
            raise RuntimeError(
                f"[{behavior}] --map-variants shufpair requested but no shufpair map was "
                "fit (prepare_behavior claim4_sink missing) — refusing a silent true-map "
                "substitution"
            )
        for mv_i, mv in enumerate(mv_list):
            is_shuf = mv == "shufpair"
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
                if mv_i == 0:
                    pools_record.append(
                        {
                            "behavior": behavior,
                            "holdout": holdout,
                            "train_frac": pool.train_frac,
                            "seed": pool.seed,
                            "per_dataset_train_n": {
                                k: int(len(v)) for k, v in pool.train_rows.items()
                            },
                            "per_dataset_heldin_n": {
                                k: int(len(v)) for k, v in pool.heldin_eval_rows.items()
                            },
                            "n_wc_train": int(len(wc_train_rows)),
                            "n_readout_total": int(len(readout_pb)),
                        }
                    )
                base_prov = {
                    "holdout": holdout,
                    "train_frac": args.train_frac,
                    "readout_train": "union: 80% GROUP-level slice of every "
                    "trait-eliciting dataset except the holdout (whole) + judged "
                    "WildChat train split",
                    "included_datasets": sorted(pool.train_rows),
                    **({"map_source": "refit-pairing-shuffled"} if is_shuf else {}),
                }
                adapter = getattr(args, "arm2_adapter", None)
                if adapter is None:
                    # legacy path — byte-identical to the claim4-controls round
                    _fit_and_score(
                        "P-B",
                        f"P-B-holdout-{holdout}",
                        readout_pb,
                        dv_z_pb,
                        eval_specs_pb,
                        base_prov,
                        mapfit_use=mapfit_shuf if is_shuf else None,
                        roster=SHUFPAIR_ROSTER if is_shuf else None,
                        map_variant=mv,
                        preds_tag="shufpair" if is_shuf else None,
                    )
                else:
                    # arm2fix repair ladder (plan §4 D1): one pass plan per
                    # holdout; the parse-time validator refuses shufpair with
                    # an adapter, so is_shuf is False here by construction.
                    for p in _arm2fix_passes(
                        adapter,
                        ROSTER,
                        [np.asarray(r, dtype=np.int64) for r in pool_rows],
                        wc_train_rows,
                        dv_z_pb,
                        parity_refit_arm7=bool(getattr(args, "parity_refit_arm7", False)),
                    ):
                        arm_prov = {
                            slug: {
                                "adapter": m["adapter"],
                                "train_row_ids_sha256": _row_ids_sha256(
                                    ctx_ids[i] for i in m["fit_rows"]
                                ),
                                "train_rows_n": int(len(m["fit_rows"])),
                            }
                            for slug, m in p.arm_meta.items()
                        }
                        _fit_and_score(
                            "P-B",
                            f"P-B-holdout-{holdout}",
                            p.readout,
                            dv_z_pb,
                            eval_specs_pb,
                            {
                                **base_prov,
                                "arm2_adapter": adapter,
                                "fit_pass": p.label,
                                **(
                                    {
                                        "readout_train": "eliciting-component slices only "
                                        "(restricted pass: the judged WildChat block is "
                                        "excluded from the fit — plan §4 R-B/R-C)"
                                    }
                                    if p.label in ("a2r", "a2qr", "parity")
                                    and len(p.readout) != len(readout_pb)
                                    else {}
                                ),
                            },
                            roster=p.roster,
                            map_variant=mv,
                            preds_tag=p.preds_tag,
                            arm_prov=arm_prov,
                        )
        if {"true", "shufpair"} <= {m for m in mv_list if m}:
            _assert_arm4_variant_identity(rows_all, behavior)

        # matched-regime sanity read (plan §4 sanity-instrument repair): the
        # arm2-family folded GROUP-level CV on the pool's eliciting rows,
        # emitted per seed ALONGSIDE the transfer rows. Fires only on the
        # arm2fix lane (--arm2-adapter) — every other lane is byte-identical.
        if getattr(args, "arm2_adapter", None) is not None:
            t_s = time.time()
            sanity_rows = _matched_regime_sanity(
                args, behavior, layers, datasets, z_ctx, dv_raw, rb_w, frozen, variant, lmax
            )
            rows_all.extend(sanity_rows)
            _log(
                f"[{behavior}] matched-regime sanity: {len(sanity_rows)} row(s) "
                f"in {time.time() - t_s:.0f}s (rung_kind=sanity_matched_regime)"
            )

    # ---- P-C: LODO-consistent map+readout — the MAP is refit per holdout on
    # generic pool + the SAME 80% slices the readout trains on (whitening
    # frozen per behavior; holdout unseen by map AND readout, hard-asserted).
    pc_map_diags: dict[str, dict] = {}
    if "C" in args.protocols:
        from scripts.issue1739_fits import _fit_map
        from scripts.issue1739_jobd_r2aug import _fitmap_ns

        if not gen_sink or "x_gen_w" not in gen_sink:
            raise RuntimeError(
                f"[{behavior}] P-C requires the retained whitened generic pool "
                "(gen_sink) — prepare_behavior must see 'C' in args.protocols"
            )
        x_gen_w, y_gen_w, n_gen = gen_sink["x_gen_w"], gen_sink["y_gen_w"], gen_sink["n_gen"]
        percell_dir = _behavior_out_dir(args, behavior) / "percell"
        percell_dir.mkdir(parents=True, exist_ok=True)
        holdouts = args.pb_holdouts or eval_datasets
        unknown = sorted(set(holdouts) - set(eval_datasets))
        if unknown:
            raise ValueError(f"--pb-holdouts {unknown} not in eval datasets {eval_datasets}")
        # per-cell resume key: EVERY output-affecting regime knob + input shas
        regime_key = {
            "behavior": behavior,
            "protocol": "P-C",
            "variant": variant,
            "regime": args.regime,
            "train_frac": args.train_frac,
            "seed": args.seed,
            "draw": args.draw,
            "min_n": args.min_n,
            "n_boot": args.n_boot,
            "roster": list(ROSTER),
            "map_condition": "add",
            "map_source": "per-holdout-refit",
            "budget_l": lmax,
            "frozen_layers": {a: int(i) for a, i in frozen.items()},
            "inputs_fingerprint": hashlib.sha256(
                json.dumps(sorted(loaded.shas.items())).encode()
            ).hexdigest()[:16],
        }
        for holdout in holdouts:
            cell_path = percell_dir / f"pc_holdout_{holdout}.json"
            if cell_path.exists():
                prev = json.loads(cell_path.read_text())
                if prev.get("regime_key") == regime_key:
                    rows_all.extend(prev["rows"])
                    skips_all.extend(prev["skips"])
                    per_layer_all.extend(prev["per_layer"])
                    fit_reports.extend(prev["fit_reports"])
                    pools_record.extend(prev["pools"])
                    pc_map_diags[holdout] = prev["map_diag"]
                    if getattr(args, "transfer_preds", False):
                        # a resumed cell skips the writer call, but its sidecar
                        # is part of THIS summary's manifest — verify it is
                        # really on disk before logging it as written (a stale
                        # cell cache without its sidecar refuses loud here,
                        # never at the next resume; codex r5 P-C scope).
                        pc_name = f"P-C-holdout-{holdout}.jsonl"
                        pc_path = _behavior_out_dir(args, behavior) / "transfer_preds" / pc_name
                        if not pc_path.exists():
                            raise RuntimeError(
                                f"[{behavior}] P-C resume cell {cell_path.name} has no preds "
                                f"sidecar on disk ({pc_name}) — stale/partial cell cache; "
                                "delete the cell file to recompute (refusing a summary whose "
                                "manifest would name a missing file)"
                            )
                        preds_files_written.append(pc_name)
                    _log(f"[{behavior}] [P-C holdout={holdout}] RESUME: completed cell loaded")
                    continue
                _log(f"[{behavior}] [P-C holdout={holdout}] stale regime key — recomputing")
            n0 = (
                len(rows_all),
                len(skips_all),
                len(per_layer_all),
                len(fit_reports),
                len(pools_record),
            )
            pool = assemble_readout_pool(
                datasets, holdout=holdout, train_frac=args.train_frac, seed=args.seed
            )
            pool_rows = [pool.train_rows[n] for n in sorted(pool.train_rows)]
            elic_rows = np.concatenate(pool_rows).astype(np.int64)
            # ---- per-holdout MAP refit (the P-C delta vs the frozen ADD map)
            t_m = time.time()
            x_w_pc = np.concatenate([x_gen_w, z_ctx[:, elic_rows]], axis=1)
            y_w_pc = np.concatenate([y_gen_w, z_ans[:, elic_rows]], axis=1)
            n_pool_pc = int(x_w_pc.shape[1])
            _assert_well_posed(n_pool_pc, loaded.dim, f"{behavior}/P-C-map-{holdout}")
            mapfit_pc = _fit_map(_fitmap_ns(args), x_w_pc, y_w_pc)
            del x_w_pc, y_w_pc
            assert mapfit_pc is not mapfit, "P-C must fit a FRESH map, never the frozen one"
            map_fit_s = round(time.time() - t_m, 1)
            map_diag_pc = {
                **mapfit_pc.diagnostics,
                "map_kind": "linear",
                "map_source": "per-holdout-refit",
                "protocol": "P-C",
                "holdout": holdout,
                "map_fit_s": map_fit_s,
                "n_map_pool": n_pool_pc,
                "map_n_generic": int(n_gen),
                "map_n_eliciting": int(elic_rows.size),
                "map_pool": "whitened generic fit pool + 80% GROUP-level slice of every "
                "trait-eliciting dataset except the holdout (whitening FROZEN per "
                "behavior — fit on the ADD pool, which contains no holdout rows)",
            }
            pc_map_diags[holdout] = map_diag_pc
            _log(
                f"[{behavior}] [P-C holdout={holdout}] MAP REFIT engaged: "
                f"n_pool={n_pool_pc} (gen {n_gen} + elic {elic_rows.size}), "
                f"map_fit_s={map_fit_s}s — frozen map NOT reused"
            )
            _log_rss(f"pc-map-refit-{holdout}")
            readout_pc = np.concatenate(pool_rows + [wc_train_rows]).astype(np.int64)
            dv_z_pc = _multi_pool_zscored_dv(dv_raw, pool_rows + [wc_train_rows])
            eval_specs_pc: list[tuple[str, object]] = [
                (holdout, ds_by_name[holdout].rows),
                (WC_RUNG, wc_eval_rows),
                (PV_RUNG, None),
            ]
            eval_specs_pc += [
                (f"heldin:{name}", pool.heldin_eval_rows[name])
                for name in sorted(pool.heldin_eval_rows)
            ]
            pools_record.append(
                {
                    "behavior": behavior,
                    "protocol": "P-C",
                    "holdout": holdout,
                    "train_frac": pool.train_frac,
                    "seed": pool.seed,
                    "per_dataset_train_n": {k: int(len(v)) for k, v in pool.train_rows.items()},
                    "per_dataset_heldin_n": {
                        k: int(len(v)) for k, v in pool.heldin_eval_rows.items()
                    },
                    "n_wc_train": int(len(wc_train_rows)),
                    "n_readout_total": int(len(readout_pc)),
                    "n_map_pool": n_pool_pc,
                    "map_n_generic": int(n_gen),
                    "map_n_eliciting": int(elic_rows.size),
                }
            )
            _fit_and_score(
                "P-C",
                f"P-C-holdout-{holdout}",
                readout_pc,
                dv_z_pc,
                eval_specs_pc,
                {
                    "holdout": holdout,
                    "train_frac": args.train_frac,
                    "map_source": "per-holdout-refit",
                    "n_map_pool": n_pool_pc,
                    "readout_train": "union: 80% GROUP-level slice of every trait-eliciting "
                    "dataset except the holdout (whole) + judged WildChat train split; the "
                    "MAP is refit on the generic pool + those same slices (LODO-consistent)",
                    "included_datasets": sorted(pool.train_rows),
                },
                mapfit_use=mapfit_pc,
                map_train_ids={ctx_ids[i] for i in elic_rows},
            )
            del mapfit_pc
            # ---- per-cell persistence: write the moment the cell lands ------
            cell_payload = {
                "regime_key": regime_key,
                "holdout": holdout,
                "rows": rows_all[n0[0] :],
                "skips": skips_all[n0[1] :],
                "per_layer": per_layer_all[n0[2] :],
                "fit_reports": fit_reports[n0[3] :],
                "pools": pools_record[n0[4] :],
                "map_diag": map_diag_pc,
                "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            tmp = cell_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(cell_payload, indent=1))
            os.replace(tmp, cell_path)
            _log(f"[{behavior}] [P-C holdout={holdout}] persisted -> {cell_path}")

    _free_cuda(args.device)
    return {
        "rows": rows_all,
        "skips": skips_all,
        "per_layer": per_layer_all,
        "fit_reports": fit_reports,
        "pools": pools_record,
        "preds_files_written": sorted(set(preds_files_written)),
        "preds_files_planned": sorted(preds_files_planned),
        "map_diagnostics": {
            f"{variant}|add|linear|{u_label}": map_diags["linear"],
            **(
                {f"{variant}|add|linear|{u_label}|shufpair": map_diags["linear_shufpair"]}
                if "linear_shufpair" in map_diags
                else {}
            ),
            **{f"{variant}|add|linear|pc_holdout_{h}": d for h, d in pc_map_diags.items()},
        },
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


def _apply_extra_arms(extra: list[str]) -> None:
    """Append --extra-arms slugs to the module-level ROSTER / LABEL_CONSUMING.

    A module-global rebind rather than threading a roster parameter: ROSTER is
    read at eight downstream sites (frozen-layer resolution, the P-A/P-B/P-C
    fit seam, the train-OOF pass, the per-layer sweep, and three metadata
    blocks), this is a single-process entrypoint, and the rebind happens once
    at parse time before any of them run -- so threading would touch every one
    of those call sites to buy nothing.

    Idempotent and order-preserving; a slug already in ROSTER is a no-op. Empty
    ``extra`` (the default) leaves both tuples untouched, which is what keeps
    every flagless run -- including the live P-C legs sharing this script --
    byte-identical.
    """
    global ROSTER, LABEL_CONSUMING
    if not extra:
        return
    unknown = sorted(set(extra) - EXTRA_ARMS_ALLOWED)
    if unknown:
        raise ValueError(
            f"--extra-arms {unknown} not in EXTRA_ARMS_ALLOWED "
            f"{sorted(EXTRA_ARMS_ALLOWED)}; adding an arm requires checking its scoring "
            "path against the r2v2 CellData AND that the committed train summary records "
            "a frozen layer for it (committed_frozen raises otherwise)"
        )
    added = [a for a in dict.fromkeys(extra) if a not in ROSTER]
    if not added:
        return
    ROSTER = ROSTER + tuple(added)
    LABEL_CONSUMING = LABEL_CONSUMING + tuple(a for a in added if a in EXTRA_ARMS_LABEL_CONSUMING)
    print(f"[extra-arms] roster extended with {added}; roster now {list(ROSTER)}", flush=True)


def _apply_arm2fix_roster(args) -> None:
    """Roster restriction + repair-ladder guards for the arm2fix lane.

    Runs AFTER :func:`_apply_extra_arms` at parse time. Every guard here is a
    loud parse-time refusal (never a silent downstream substitution):

    - ``--arms-only-extra`` RESTRICTS the module ROSTER to exactly the
      ``--extra-arms`` slugs (the D1 rerun scores arm2 alone; banked
      arm4/arm7 rows join at the fold).
    - ``--arm2-adapter v2-quantile*`` appends ``arm2q_ctx_native`` (the R-C
      slug emitted BESIDE the unrepaired arm2 — never a relabel).
    - ``--skip-map-fit`` refuses any roster containing a map-consuming arm
      (single source: ``arms.MAP_CONSUMING_ARMS`` — the SAME set
      ``run_cell_multi`` dispatches on, so guard and dispatch cannot drift),
      refuses the shufpair variant (no map is fit to shuffle) and protocol C
      (P-C refits maps per holdout).
    - ``--parity-refit-arm7`` requires an adapter (it row-matches the
      repaired arm's fit rows) and the TRUE map (no --skip-map-fit).
    """
    global ROSTER, LABEL_CONSUMING
    if args.arms_only_extra:
        if not args.extra_arms:
            raise SystemExit("--arms-only-extra needs --extra-arms (an empty roster is not a run)")
        only = tuple(dict.fromkeys(args.extra_arms))
        ROSTER = only
        LABEL_CONSUMING = tuple(a for a in only if a in EXTRA_ARMS_LABEL_CONSUMING)
        print(f"[arms-only-extra] roster RESTRICTED to {list(ROSTER)}", flush=True)
    if args.arm2_adapter is not None:
        if "B" not in args.protocols:
            raise SystemExit("--arm2-adapter is a P-B repair — needs 'B' in --protocols")
        if "arm2_ctx_native" not in ROSTER:
            raise SystemExit(
                "--arm2-adapter needs arm2_ctx_native in the roster "
                "(pass --extra-arms arm2_ctx_native)"
            )
        if "shufpair" in (args.map_variants or []):
            raise SystemExit(
                "--arm2-adapter is incompatible with --map-variants shufpair (the arm2 "
                "family never rides the shufpair roster; banked shufpair rows join at "
                "the fold)"
            )
        if args.arm2_adapter.startswith("v2-quantile") and "arm2q_ctx_native" not in ROSTER:
            ROSTER = ROSTER + ("arm2q_ctx_native",)
            LABEL_CONSUMING = LABEL_CONSUMING + ("arm2q_ctx_native",)
            print(
                "[arm2fix] arm2q_ctx_native appended to the roster (R-C quantile fallback, "
                "emitted BESIDE the unrepaired arm2)",
                flush=True,
            )
    if args.parity_refit_arm7:
        if args.arm2_adapter is None:
            raise SystemExit(
                "--parity-refit-arm7 needs --arm2-adapter (it row-matches the repaired "
                "arm's fit rows)"
            )
        if args.skip_map_fit:
            raise SystemExit(
                "--parity-refit-arm7 refits arm7 against the TRUE map — drop --skip-map-fit "
                "for this behavior (plan §4 matched-budget parity duty)"
            )
    if args.skip_map_fit:
        from explore_persona_space.experiments.issue_1739.arms import MAP_CONSUMING_ARMS

        offending = sorted(set(ROSTER) & MAP_CONSUMING_ARMS)
        if offending:
            raise SystemExit(
                f"--skip-map-fit REFUSED: resolved roster contains map-consuming arm(s) "
                f"{offending} — a skipped map fit would silently gut them "
                "(guard reads arms.MAP_CONSUMING_ARMS)"
            )
        if "shufpair" in (args.map_variants or []):
            raise SystemExit(
                "--skip-map-fit is incompatible with --map-variants shufpair "
                "(no map is fit to shuffle)"
            )
        if "C" in args.protocols:
            raise SystemExit(
                "--skip-map-fit is incompatible with protocol C (P-C refits maps per holdout)"
            )


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
        choices=["A", "B", "AB", "C", "ABC"],
        help="which readout protocols to fit (A = r2fair, B = LODO-mixture readout, "
        "C = LODO-consistent map+readout: the MAP is refit per holdout too)",
    )
    ap.add_argument(
        "--pb-holdouts",
        nargs="+",
        default=None,
        help="P-B / P-C holdout datasets (default: every non-train dataset)",
    )
    ap.add_argument("--train-frac", type=float, default=0.8)
    ap.add_argument("--regime", default="e1", choices=("e1",))
    ap.add_argument("--layers", type=int, nargs="+", default=None)
    ap.add_argument("--n-layers", type=int, default=28)
    ap.add_argument("--draw", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="claim4-controls seed replication: run every behavior once PER SEED, with "
        "outputs keyed <behavior>/seed<S>/ (all_arms_spearman.json, transfer_preds, ...). "
        "Effective default = [--seed] (i.e. [0]) via the legacy single-seed path: with the "
        "flag ABSENT the out-dir layout and every byte of output are unchanged, which is "
        "what keeps the live fits/pc/arm12 legs byte-identical. Passing --seeds (even one "
        "value) opts into the seed-keyed layout.",
    )
    ap.add_argument(
        "--map-variants",
        nargs="+",
        default=None,
        choices=["true", "shufpair"],
        help="claim4-controls map-variant loop (P-B only): 'true' = the frozen ADD-pool "
        "map (full roster); 'shufpair' = a map refit on PAIRING-SHUFFLED context-answer "
        "pairs (one within-component permutation per pool component, rng [1739, 21, seed]; "
        "roster {arm4, arm6, arm7}). Effective default = ['true'] via the legacy path: "
        "with the flag ABSENT no map_variant field is emitted and every lane is "
        "byte-identical. Rows gain map_variant when the flag is passed.",
    )
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
    ap.add_argument(
        "--transfer-preds",
        action="store_true",
        help="ALSO persist per-(arm, eval context) frozen-layer predictions for every "
        "r2v2 fit (one JSONL per fit under <out-root>/<behavior>/transfer_preds/, schema "
        "= arms.transfer_preds_rows with a per-context 'rung' label). Default OFF: the "
        "aggregate rows are unchanged, so every other lane is byte-identical. Turning it "
        "on is what makes a per-context re-read — paired bootstrap CIs over shared eval "
        "contexts, per-rung subset reads, OOD scatters — a pure re-analysis instead of "
        "another full re-score (the sibling --transfer-preds in issue1739_fits.py does "
        "the same for the bare-query transfer leg).",
    )
    ap.add_argument(
        "--extra-arms",
        nargs="+",
        default=[],
        choices=sorted(EXTRA_ARMS_ALLOWED),
        help="APPEND these arm slugs to ROSTER for this run. Default EMPTY: with the flag "
        "absent the roster is the committed five and every other lane -- including the "
        "live P-C legs that share this script -- is byte-identical. arm12_oracle_reg is "
        "the ridge-on-TRUE-answer oracle companion to arm11_oracle_proj (PV on the true "
        "answer); the r2v2 round never scored it, so adding it to a P-A/P-B figure needs "
        "a re-score, not a re-read.",
    )
    ap.add_argument(
        "--arms-only-extra",
        action="store_true",
        help="RESTRICT the roster to exactly the --extra-arms slugs (arm2fix D1: the rerun "
        "scores arm2 alone against the banked arm4/arm7 rows, which join at the fold). "
        "Default OFF: --extra-arms stays APPEND-only and every other lane is byte-identical.",
    )
    ap.add_argument(
        "--skip-map-fit",
        action="store_true",
        help="skip the fit_linear_add_map linear-map fit (whitening still runs — arm2 "
        "operates in whitened coordinates; the seed-keyed rng reproduces the banked space). "
        "REFUSED unless the resolved roster contains no map-consuming arm "
        "(arms.MAP_CONSUMING_ARMS — the same set run_cell_multi dispatches on).",
    )
    ap.add_argument(
        "--arm2-adapter",
        default=None,
        choices=list(ARM2_ADAPTERS),
        help="arm2fix repair-ladder adapter (plan §4 R-A/R-B/R-C; see the ARM2_ADAPTERS "
        "comment). Default None: the P-B pass is byte-identical to the claim4-controls "
        "round. Any value ALSO emits the matched-regime sanity rows "
        "(rung_kind=sanity_matched_regime) per seed.",
    )
    ap.add_argument(
        "--parity-refit-arm7",
        action="store_true",
        help="matched-budget parity duty (plan §4 Must-Fix): ALSO refit arm7 "
        "(map_variant=true) per holdout on the IDENTICAL training-row IDs the repaired "
        "arm2 consumed; rows carry adapter=parity-row-matched + the same "
        "train_row_ids_sha256 so the fold can assert row-set equality. Requires the TRUE "
        "map (incompatible with --skip-map-fit).",
    )
    ap.add_argument(
        "--a2-sanity-folds",
        type=int,
        default=5,
        help="fold count for the matched-regime sanity read (group-hash folds; plan §4)",
    )
    ap.add_argument("--allow-overwrite-committed", action="store_true")
    ap.add_argument("--import-check", action="store_true")
    args = ap.parse_args(argv)
    if args.seeds is not None:
        if len(set(args.seeds)) != len(args.seeds):
            raise SystemExit(f"--seeds carries duplicates: {args.seeds}")
    if args.map_variants is not None and len(set(args.map_variants)) != len(args.map_variants):
        raise SystemExit(f"--map-variants carries duplicates: {args.map_variants}")
    if args.pb_holdouts is not None and len(set(args.pb_holdouts)) != len(args.pb_holdouts):
        # codex r6 arm2fix-preds-plan-multiplicity: duplicated holdouts run
        # and overwrite the same sidecar while plan, row expectation, and
        # writer log all set-collapse together — uniqueness is PROVEN here
        # before any set construction.
        raise SystemExit(f"--pb-holdouts carries duplicates: {args.pb_holdouts}")
    _apply_extra_arms(args.extra_arms)
    _apply_arm2fix_roster(args)
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
    _wire_fits_rss_logging()
    from scripts.issue1739_wcrung_arms import _assert_no_judge_modules

    _assert_no_judge_modules("at entry")
    if args.import_check:
        from explore_persona_space.analysis.mapping_baselines import (  # noqa: F401
            identity_bias_predict,
            knn_retrieval,
        )
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
        # claim4-controls surface: shufpair roster + matched-frozen companions
        # resolve against the registry; the shuffle helpers are importable.
        for slug in SHUFPAIR_ROSTER:
            assert slug in _arms.ARM_REGISTRY, f"{slug} (SHUFPAIR_ROSTER) not in ARM_REGISTRY"
        for a, ref in MATCHED_FROZEN_COMPANIONS.items():
            assert a in _arms.ARM_REGISTRY and ref in _arms.ARM_REGISTRY, (a, ref)
        assert callable(fits.shuffled_map_weights), "arm20 weight-shuffle helper missing"
        # arm2fix surface: quantile-split slug + thresholds + the map-consuming
        # set the --skip-map-fit guard reads (single source with the dispatch).
        for slug in A2_FAMILY:
            assert slug in _arms.ARM_REGISTRY, f"{slug} (A2_FAMILY) not in ARM_REGISTRY"
        assert callable(_arms.arm2q_thresholds), "arm2q threshold helper missing"
        assert _arms.MAP_CONSUMING_ARMS & set(_arms.ARM_REGISTRY) == _arms.MAP_CONSUMING_ARMS, (
            "MAP_CONSUMING_ARMS carries a slug outside ARM_REGISTRY"
        )
        assert not set(A2_FAMILY) & _arms.MAP_CONSUMING_ARMS, "arm2 family must be map-free"
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
    # seed loop (claim4-controls): --seeds keys every output under
    # <behavior>/seed<S>/; the legacy single-seed path (--seeds absent) keeps
    # args.seed + the flat layout byte-identical.
    seeds = list(args.seeds) if args.seeds is not None else [args.seed]
    seed_keyed = args.seeds is not None
    for b in args.behaviors:
        for s in seeds:
            args.out_subdir = f"seed{s}" if seed_keyed else ""
            out = _behavior_out_dir(args, b) / "all_arms_spearman.json"
            if _git_tracked(out) and not args.allow_overwrite_committed:
                raise SystemExit(f"refusing to overwrite git-TRACKED output: {out}")

    layers = args.layers or list(range(args.n_layers))
    commit = _git_commit()
    env = _env_versions()
    failures: list[dict] = []
    t_all = time.time()
    units = [(b, s) for b in args.behaviors for s in seeds]
    for i, (behavior, seed) in enumerate(units, start=1):
        t0 = time.time()
        args.seed = int(seed)
        args.out_subdir = f"seed{seed}" if seed_keyed else ""
        if seed_keyed:
            resume_dir = _behavior_out_dir(args, behavior)
            ok, why = _seed_output_resume_ok(
                resume_dir,
                commit=commit,
                seed=int(seed),
                map_variants=args.map_variants,
                arm2_adapter=getattr(args, "arm2_adapter", None),
                skip_map_fit=bool(getattr(args, "skip_map_fit", False)),
                parity_refit_arm7=bool(getattr(args, "parity_refit_arm7", False)),
                arms_only_extra=bool(getattr(args, "arms_only_extra", False)),
                a2_sanity_folds=(
                    int(getattr(args, "a2_sanity_folds", 5))
                    if getattr(args, "arm2_adapter", None) is not None
                    else None
                ),
                transfer_preds=bool(getattr(args, "transfer_preds", False)),
                # the current planned universe is computable pre-prep only
                # when --pb-holdouts pins the holdout list (else holdouts =
                # eval_datasets, resolved inside run_behavior); when known it
                # must MATCH the recorded plan (codex r6).
                planned_files=(
                    sorted(
                        _planned_preds_files(
                            args.protocols,
                            list(args.pb_holdouts),
                            getattr(args, "map_variants", None),
                            arm2_adapter=getattr(args, "arm2_adapter", None),
                            parity_refit_arm7=bool(getattr(args, "parity_refit_arm7", False)),
                        )
                    )
                    if getattr(args, "transfer_preds", False) and args.pb_holdouts is not None
                    else None
                ),
            )
            if ok:
                _log(
                    f"[seed-loop {i}/{len(units)}] RESUME-SKIP {behavior} seed={seed} — "
                    f"output matches code SHA {commit[:12]} + schema "
                    f"v{SEED_OUT_SCHEMA_VERSION} ({resume_dir})"
                )
                continue
            if (resume_dir / "all_arms_spearman.json").exists():
                _log(
                    f"[seed-loop {i}/{len(units)}] prior output at {resume_dir} NOT "
                    f"resumable ({why}) — re-running the seed"
                )
                _remove_stale_summary(resume_dir)
            _log(f"[seed-loop {i}/{len(units)}] START {behavior} seed={seed}")
        try:
            res = run_behavior(args, behavior, layers)
        except (FileNotFoundError, RuntimeError, ValueError, AssertionError) as exc:
            failures.append(
                {"behavior": behavior, "seed": int(seed), "error": f"{type(exc).__name__}: {exc}"}
            )
            _log(f"{behavior} seed={seed} FAILED: {exc}")
            continue
        loaded = res.pop("loaded")
        out_dir = _behavior_out_dir(args, behavior)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "all_arms_spearman.json"

        # companions FIRST, summary LAST — the summary is the per-seed
        # completion sentinel (_write_companions_then_summary docstring);
        # deferred behind a def so the summary lands after the companions.
        def write_summary(res=res, out_path=out_path, loaded=loaded):
            # planned-vs-written + row-derived manifest gate (codex r4 minor;
            # r5 MAJOR: the planned universe derives BEFORE execution, so a
            # whole omitted pass refuses the sentinel too).
            _summary_preds_gate(res, transfer_preds=bool(getattr(args, "transfer_preds", False)))
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
                    **(
                        {
                            "seed": int(args.seed),
                            "seed_keyed_out_dir": bool(seed_keyed),
                            "out_schema_version": SEED_OUT_SCHEMA_VERSION,
                            "map_variants": list(args.map_variants or []),
                            "shufpair_roster": list(SHUFPAIR_ROSTER),
                            # arm2fix lane provenance (None/False on every
                            # non-arm2fix run) — the per-seed resume predicate
                            # keys on these so a legacy output can never
                            # silently satisfy an arm2fix invocation.
                            "arm2_adapter": getattr(args, "arm2_adapter", None),
                            "skip_map_fit": bool(getattr(args, "skip_map_fit", False)),
                            "arms_only_extra": bool(getattr(args, "arms_only_extra", False)),
                            "parity_refit_arm7": bool(getattr(args, "parity_refit_arm7", False)),
                            **(
                                {"a2_sanity_folds": int(getattr(args, "a2_sanity_folds", 5))}
                                if getattr(args, "arm2_adapter", None) is not None
                                else {}
                            ),
                            # preds-sidecar MANIFEST from the WRITER'S OWN log
                            # (codex r2 minor + r3 arm2fix-preds-manifest-
                            # universe): the run's intended file set, recorded
                            # at each write site — never a directory glob,
                            # which would let any surviving strict subset
                            # define its own universe and read resume-complete.
                            **(
                                {
                                    "transfer_preds_files": list(
                                        res.get("preds_files_written", [])
                                    ),
                                    # the pre-execution plan universe the gate
                                    # enforced (codex r5) — audit record
                                    "transfer_preds_planned": list(
                                        res.get("preds_files_planned", [])
                                    ),
                                }
                                if getattr(args, "transfer_preds", False)
                                else {}
                            ),
                        }
                        if (seed_keyed or args.map_variants is not None)
                        else {}
                    ),
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
                        "train split; one fit per holdout (LODO); map FROZEN (behavior-level "
                        "ADD fit)",
                        "P-C": f"map AND readout refit per holdout: map = generic fit pool + "
                        f"{args.train_frac:.0%} GROUP-level slice of every trait-eliciting "
                        "dataset except the holdout (whitening frozen per behavior); readout "
                        "= those same slices + judged WildChat train split (the P-B pool); "
                        "holdout unseen by map AND readout (hard-asserted)",
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

        _write_companions_then_summary(out_dir, res, write_summary)
        _verify_input_shas(loaded.shas)
        _log(f"{behavior} done: {len(res['rows'])} transfer rows in {res['wall_s']}s -> {out_path}")
        if seed_keyed:
            _log(
                f"[seed-loop {i}/{len(units)}] DONE {behavior} seed={seed} "
                f"elapsed={time.time() - t0:.1f}s"
            )
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
