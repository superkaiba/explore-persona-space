#!/usr/bin/env python
"""#2388 readout driver: L-sweep fits, f_U map cells, nulls, bootstrap, H3 legs.

Phases (P3/P4/P5 pod-side; select/bootstrap/h3-gap CPU-side):

  feasibility  P1 pre-step arithmetic joint-feasibility assertion (plan
               section 4, "Pre-fit joint-feasibility assertion") — halts
               BEFORE generation on any (surface, cell, L) violation.
  maps         fit the f_U map keys (linear + MLP families) with the
               manifest-resolved per-key pool metadata + Result 0(d)
               diagnostics (identity+bias / kNN / held-out R^2).
  sweep        the readout L sweep: arms x bases x L x draws through the
               extended per-target GCV core with EXPLICIT dof_cap=0.9,
               permutation-null targets riding as extra columns through the
               shared factorization (#825 batched-null pattern); persists
               per-cell rho/r2/auroc matrices + nulls npz + preds JSONLs.
  select       freeze per-surface dev-selected configurations
               (selection.json) from DEV matrices only — committed before
               any test read is aggregated.
  bootstrap    paired group bootstrap (2,000 draws) from stored preds; the
               identical group resample is shared across compared arms.
  h3           H3 two-stage orchestration (G4 asserts a-e): stage-1
               parent-side verdict recorded BEFORE any correctness-side
               2,500 read; composes + runs the ported issue1739_fits.py
               invocations.
  h3-gap       the pinned MEAN gap definition over an out-root's
               all_arms_spearman.json + the banked-reference verifier.

Estimator: ridge everywhere (user-settled); the only nonlinearity is the MLP
MAP family. Every primary cell runs dof_cap=0.9 (no pure GCV at n<d).
CONTENT HYGIENE: logs carry ids/counts, never row text.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path


def _ensure_repo_root_on_syspath() -> Path:
    root = Path(__file__).resolve().parents[1]
    assert (root / "pyproject.toml").exists(), root
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


REPO_ROOT = _ensure_repo_root_on_syspath()

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

import numpy as np  # noqa: E402

from explore_persona_space.experiments.issue_1739 import fits as F  # noqa: E402
from explore_persona_space.experiments.issue_1739.arms import (  # noqa: E402
    _pearson_rows,  # batched-bootstrap building block (same math as spearman_rows)
    auroc_rows,
    rank_rows,
    spearman_rows,
)
from explore_persona_space.experiments.issue_1739.constants import (  # noqa: E402
    HIDDEN_DIM,
    N_LAYERS,
)


def _stable_seed(text: str) -> int:
    """PYTHONHASHSEED-independent seed component (str hash() is salted per
    process — a hash(surface) seed part makes draws irreproducible)."""
    return int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)


# --- Registered grid (plan sections 4-6) ---
L_GRID = (250, 500, 1000, 2000, 4000, 8000)  # + "full" appended per surface
N_DRAWS = 3
N_NULL = 200
N_BOOT = 2000
PCA_KS = (16, 48, 128)
DOF_CAP = 0.9
FU_CELLS = (0.0, 0.5, 1.0)
FU_L_ANCHORS = (250, 2000, "full")
QA_DISJOINT_ANCHORS = (250, 2000)
SURFACE_BENCHMARKS = {
    "math": ("math_full",),
    "mcq": ("mmlu_pro_full",),
    # code is the CANDIDATE list only — the REALIZED roster is gate-derived per
    # run via _surface_benchmarks (BCB only on bcb_fit_allowed, apps_intro only
    # on apps_activated; plan fork 5 / r3 Critical 2). Non-code consumers and
    # the surface-name key set read this dict directly.
    "code": ("humaneval", "mbpp_full", "bigcodebench_full", "lcb_v5", "leetcode"),
    "qa": (),  # banked #1739 store + derived labeling
}
POOL_SIZE = {"qa": 8000, "math": 8000, "mcq": 8000, "code": None}  # None = realized train
# Step 0-p encoder pin (plan: "exact model id selected at Step 0-p"): the
# sentence-transformers default strong general-purpose encoder; import +
# 1-sentence encode smoke-verified on the VM (report section (c)).
ENCODER_ID = "sentence-transformers/all-mpnet-base-v2"
FITS_ROOT = Path("eval_results/issue_2388/fits")
MAPS_OUT = Path("eval_results/issue_2388/maps")
DV_ROOT = Path("eval_results/issue_2388/dv")
SEED0 = 2388
# rung-1 definitions (plan section 4 "Splits and ladder")
MATH_RUNG1_FIT_LEVELS = {1, 2, 3}
MATH_RUNG1_EVAL_LEVELS = {4, 5}
MCQ_RUNG1_HELDOUT_K = 4
CODE_RUNG1_FIT = {"humaneval", "mbpp_full", "leetcode"}
CODE_RUNG1_EVAL = {"bigcodebench_full", "lcb_v5"}

# Pinned H3 reference numbers (plan section 3, "Registered gap definition").
# The plan's prose numbers were REPRODUCED at Step 0-p under the pooled-u_rung
# MAX-over-unit-groups read for the four starred anchors below; the two
# unstarred @2,500 anchors did not reproduce under any tested reduction
# (raised as a persisted concern) — the verifier hard-asserts the reproduced
# four and REPORTS the rest.
H3_REFERENCE = {
    ("sycophancy", 2500): {"gap": 0.1081, "reproduced": True},
    ("sycophancy", 16000): {"gap": -0.0068, "reproduced": True},
    ("hallucination", 16000): {"gap": 0.0078, "reproduced": True},
    ("evil", 8000): {"gap": 0.0218, "reproduced": True},
    ("hallucination", 2500): {"gap": 0.030, "reproduced": False},
    ("evil", 2500): {"gap": 0.063, "reproduced": False},
}
H3_MAPPED_ARMS = ("arm6_map_proj_e1", "arm7_map_ridge_pred")
H3_DIRECT_ARM = "arm4_ridge_ctx"
H3_BUDGET = 2500
H3_COMPANION_BUDGET = {"hallucination": 16000, "evil": 8000, "sycophancy": 16000}
H3_BEHAVIORS = ("sycophancy", "evil", "hallucination")  # sycophancy FIRST (G4 e)
# Registered comparison cell (plan section 3: "row filter = (config, f_u,
# u_rung, variant, eval_rung) held at the registered comparison cell").
# Schema-from-artifact (banked all_arms_spearman.json probed 2026-08-20): rows
# live under top-level "arm_rows"; rho under "rho_frozen"; the held values
# below are the banked artifact's realized field values. u_rung is held at
# "full" by default (the largest banked U rung — the parent's primary map);
# the plan names no explicit u_rung value, so it is CLI-overridable
# (--h3-u-rung) and recorded in every gap report.
H3_CELL_FILTER = {"config": "config_a", "variant": "context_end", "eval_rung": "train"}
H3_U_RUNG_DEFAULT = "full"
H3_N_BOOT = 2000


# ---------------------------------------------------------------------------
# surface tables
# ---------------------------------------------------------------------------


@dataclass
class SurfaceTable:
    """Per-context arrays for one surface (fp16 activations, float64 dv)."""

    surface: str
    ctx_ids: list[str]
    dv: np.ndarray  # (n,)
    split: np.ndarray  # (n,) str
    group: np.ndarray  # (n,) str — the split's group axis (problem/entity)
    boot_group: np.ndarray  # (n,) str — the exchangeability axis (section 6)
    benchmark: np.ndarray  # (n,) str
    level: np.ndarray  # (n,) float (nan where absent)
    category: np.ndarray  # (n,) str
    z_ctx: np.ndarray  # (Ly, n, d) fp16 — context_end
    z_t1: np.ndarray  # (Ly, n, d) fp16 — answer-span mean over K rollouts
    z_tlast: np.ndarray | None  # (Ly, n, d) fp16 — last-answer-token (CoT surfaces)
    # rollout grain for the direction arms (spread contexts only)
    spread_ctx_idx: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))
    spread_roll_t1: np.ndarray | None = None  # (Ly, m, d) fp16
    spread_roll_ctx: np.ndarray | None = None  # (m,) int index into ctx arrays
    spread_roll_y: np.ndarray | None = None  # (m,) float 0/1 per-rollout correctness
    meta: dict = field(default_factory=dict)


def _load_labeling(labeling_json: Path, *, surface: str | None = None) -> dict:
    payload = json.loads(Path(labeling_json).read_text())
    rows = [r for r in payload["rows"] if r.get("dv") is not None]
    if not rows:
        raise RuntimeError(f"0 DV rows in {labeling_json}")
    if surface is not None:
        rows = _effective_rows(surface, rows)
    return {"payload": payload, "rows": rows}


def _effective_rows(surface: str, rows: list[dict]) -> list[dict]:
    """Attach the driver's effective split per row (``eff_split``).

    New surfaces (math/mcq/code): the dv_build 70/10/20 split carried verbatim
    (rows already carry split in {train, dev, test}).

    QA (banked #1739 artifact — schema probed: split in {train, eval}, rung in
    {train, nqopen, simpleqa}): the banked 16,000-row TriviaQA train rung gets
    a SEEDED group-level 70/10/20 partition at the entity (group_key) grain
    (FLOOR quotas over a shuffled group order, remainder assigned to train —
    r1 g6: the prior docstring claimed largest-remainder apportionment; the
    registered split arithmetic is this floor+remainder-to-train form);
    rung == 'nqopen' eval rows become the rung-1 shift set;
    rung == 'simpleqa' rows are DROPPED (body-settled exclusion).
    """
    if surface != "qa":
        out = []
        for r in rows:
            if r.get("split") not in ("train", "dev", "test"):
                raise RuntimeError(
                    f"{surface}: unexpected split {r.get('split')!r} on {r.get('context_id')}"
                )
            out.append({**r, "eff_split": r["split"]})
        return out
    train_rows = [r for r in rows if r.get("split") == "train" and r.get("rung") == "train"]
    nqopen = [r for r in rows if r.get("rung") == "nqopen"]
    n_simpleqa = sum(1 for r in rows if r.get("rung") == "simpleqa")
    if not train_rows or not nqopen:
        raise RuntimeError(
            f"qa: banked labeling shape unexpected (train={len(train_rows)}, "
            f"nqopen={len(nqopen)}, simpleqa={n_simpleqa})"
        )
    groups = sorted({str(r["group_key"]) for r in train_rows})
    rng = np.random.default_rng([SEED0, 20])
    order = [groups[i] for i in rng.permutation(len(groups))]
    fracs = {"train": 0.7, "dev": 0.1, "test": 0.2}
    quota = {s: int(len(order) * f) for s, f in fracs.items()}
    quota["train"] += len(order) - sum(quota.values())  # remainder to train
    assign: dict[str, str] = {}
    pos = 0
    for split_name in ("train", "dev", "test"):
        for g in order[pos : pos + quota[split_name]]:
            assign[g] = split_name
        pos += quota[split_name]
    out = [{**r, "eff_split": assign[str(r["group_key"])]} for r in train_rows]
    out += [{**r, "eff_split": "rung1"} for r in nqopen]
    return out


def _boot_group_of(surface: str, row: dict) -> str:
    """Section-6 exchangeability axis: qa=entity, math=level x subject,
    mcq=category, code=benchmark x problem."""
    if surface == "qa":
        return str(row.get("group_key"))
    if surface == "math":
        return f"{row.get('subject')}|L{row.get('level')}"
    if surface == "mcq":
        return str(row.get("category"))
    return f"{row.get('benchmark')}|{row.get('context_id')}"


def load_surface_table(
    surface: str,
    labeling_json: Path,
    store_dirs: list[Path],
    *,
    layers: tuple[int, ...] | None = None,
    with_tlast: bool = True,
    with_rollout_grain: bool = True,
    hidden_dim: int = HIDDEN_DIM,
) -> SurfaceTable:
    """Assemble the per-context table from labeling rows + capture store(s).

    Reduces the per-rollout store to per-context arrays (v_C from rollout 0 —
    causal attention makes context_end identical across a context's rollouts;
    t1/t_last = mean over the K rollouts), loading ONE layer at a time to
    bound peak RSS. Rollout-grain t1 is retained ONLY for spread contexts
    (0 < dv < 1 — the direction arms' input).
    """
    from explore_persona_space.experiments.issue_1739.store_io import load_summaries

    layers = tuple(layers) if layers is not None else tuple(range(N_LAYERS))
    lab = _load_labeling(labeling_json, surface=surface)
    rows = lab["rows"]
    by_ctx = {r["context_id"]: r for r in rows}

    kinds = ("context_end", "t1") + (("t_last",) if with_tlast else ())
    # Pass 1: metadata + per-store row maps (one cheap layer read for meta).
    store_meta: list[tuple[Path, list[dict]]] = []
    for sd in store_dirs:
        _, meta_rows = load_summaries(sd, ("context_end",), (layers[0],), hidden_dim=hidden_dim)
        store_meta.append((Path(sd), meta_rows))
    # Context order: labeling order filtered to contexts present in the stores.
    present: set[str] = set()
    for _, meta_rows in store_meta:
        present.update(str(m["context_id"]) for m in meta_rows)
    ctx_ids = [r["context_id"] for r in rows if r["context_id"] in present]
    missing = len(rows) - len(ctx_ids)
    if not ctx_ids:
        raise RuntimeError(f"{surface}: no labeling context present in stores {store_dirs}")
    if missing / len(rows) > 0.02:
        raise RuntimeError(
            f"{surface}: {missing}/{len(rows)} labeled contexts missing from the capture "
            "store (>2%) — store/labeling mismatch"
        )
    ctx_pos = {c: i for i, c in enumerate(ctx_ids)}
    n = len(ctx_ids)

    dv = np.array([by_ctx[c]["dv"] for c in ctx_ids], dtype=np.float64)
    split = np.array([by_ctx[c]["eff_split"] for c in ctx_ids])
    group = np.array([str(by_ctx[c]["group_key"]) for c in ctx_ids])
    boot_group = np.array([_boot_group_of(surface, by_ctx[c]) for c in ctx_ids])
    benchmark = np.array([str(by_ctx[c].get("benchmark")) for c in ctx_ids])
    level = np.array(
        [
            float(by_ctx[c]["level"]) if by_ctx[c].get("level") is not None else np.nan
            for c in ctx_ids
        ]
    )
    category = np.array([str(by_ctx[c].get("category")) for c in ctx_ids])

    spread_mask = (dv > 0.0) & (dv < 1.0)
    spread_ctx_idx = np.flatnonzero(spread_mask)
    spread_set = {ctx_ids[i] for i in spread_ctx_idx}
    has_rollout_scores = any(by_ctx[c].get("per_rollout_scores") for c in ctx_ids)
    if with_rollout_grain and not has_rollout_scores:
        # Banked QA labeling carries counts/fractions only (probed: all-None
        # per_rollout_scores) — rollout grain unavailable; direction arms are
        # excluded from the default roster downstream (fail loud on request).
        with_rollout_grain = False

    z_ctx = np.zeros((len(layers), n, hidden_dim), dtype=np.float16)
    z_t1 = np.zeros_like(z_ctx)
    z_tlast = np.zeros_like(z_ctx) if with_tlast else None
    roll_rows: list[tuple[int, float]] = []  # (ctx index, per-rollout score)
    roll_t1_chunks: list[np.ndarray] = []

    for li, layer in enumerate(layers):
        per_layer_roll: list[np.ndarray] = []
        # fp32 accumulators; divide by the REALIZED per-context row count
        # (parent _load_labeled convention: v_C = FIRST occurrence, v_A = mean
        # over the context's stored rollout rows — never a nominal-K divide).
        acc_t1 = np.zeros((n, hidden_dim), dtype=np.float32)
        acc_tl = np.zeros((n, hidden_dim), dtype=np.float32) if with_tlast else None
        cnt = np.zeros(n, dtype=np.int64)
        ctx_seen = np.zeros(n, dtype=bool)
        for sd, meta_rows in store_meta:
            arrs, meta2 = load_summaries(sd, kinds, (layer,), hidden_dim=hidden_dim)
            assert len(meta2) == len(meta_rows), (sd, layer)
            ce = arrs[("context_end", layer)]
            t1 = arrs[("t1", layer)]
            tl = arrs[("t_last", layer)] if with_tlast else None
            for ri, m in enumerate(meta_rows):
                cid = str(m["context_id"])
                if cid not in ctx_pos:
                    continue
                ci = ctx_pos[cid]
                if not ctx_seen[ci]:
                    z_ctx[li, ci] = ce[ri]
                    ctx_seen[ci] = True
                acc_t1[ci] += t1[ri].astype(np.float32)
                if acc_tl is not None:
                    acc_tl[ci] += tl[ri].astype(np.float32)
                cnt[ci] += 1
                if with_rollout_grain and cid in spread_set:
                    k = m.get("rollout_k")
                    score = (by_ctx[cid].get("per_rollout_scores") or {}).get(f"k{k}")
                    if score is not None:
                        per_layer_roll.append(t1[ri])
                        if li == 0:
                            roll_rows.append((ci, float(score)))
        if (cnt == 0).any():
            raise RuntimeError(f"{surface}: {int((cnt == 0).sum())} contexts with 0 store rows")
        z_t1[li] = (acc_t1 / cnt[:, None]).astype(np.float16)
        if z_tlast is not None:
            z_tlast[li] = (acc_tl / cnt[:, None]).astype(np.float16)
        if with_rollout_grain and per_layer_roll:
            roll_t1_chunks.append(np.stack(per_layer_roll, axis=0))
        print(f"[table] {surface} layer {layer} reduced ({n} contexts)", flush=True)

    spread_roll_t1 = None
    spread_roll_ctx = None
    spread_roll_y = None
    if with_rollout_grain and roll_rows:
        spread_roll_t1 = np.stack(roll_t1_chunks, axis=0)  # (Ly, m, d)
        spread_roll_ctx = np.array([ci for ci, _ in roll_rows], dtype=int)
        spread_roll_y = np.array([y for _, y in roll_rows], dtype=np.float64)

    return SurfaceTable(
        surface=surface,
        ctx_ids=ctx_ids,
        dv=dv,
        split=split,
        group=group,
        boot_group=boot_group,
        benchmark=benchmark,
        level=level,
        category=category,
        z_ctx=z_ctx,
        z_t1=z_t1,
        z_tlast=z_tlast,
        spread_ctx_idx=spread_ctx_idx,
        spread_roll_t1=spread_roll_t1,
        spread_roll_ctx=spread_roll_ctx,
        spread_roll_y=spread_roll_y,
        meta={
            "n_labeling_rows": len(rows),
            "n_missing_from_store": missing,
            "layers": list(layers),
            # self-consistency baseline input (bl_agree; nan where absent —
            # code/QA labelings carry no extractable answer identity)
            "agree_frac": np.array(
                [
                    float(by_ctx[c]["agree_frac"])
                    if by_ctx[c].get("agree_frac") is not None
                    else np.nan
                    for c in ctx_ids
                ],
                dtype=np.float64,
            ),
        },
    )


# ---------------------------------------------------------------------------
# joint-feasibility assertion (plan section 4; runs at P1 pre-step AND at maps)
# ---------------------------------------------------------------------------


def assert_joint_feasibility(
    surface_counts: dict[str, dict[str, int]],
    *,
    l_grid: tuple[int, ...] = L_GRID,
    key_manifest: dict | None = None,
    require_all: bool = True,
) -> dict:
    """Fails loud on any (surface, composition cell, label budget) violation.

    ``surface_counts``: per surface {"train": n_train, "dev": n, "test": n}.
    Arithmetic mode (P1 pre-step, ``key_manifest=None``) checks (i-with-
    registered-|U|), (iv), (v-arithmetic). Manifest mode additionally
    resolves each (surface, family, f_U) triple's ACTUAL key + persisted pool
    metadata and asserts realized |U| identity across the surface's three
    cells (never assumed — MF-A addendum).
    """
    report: dict = {"mode": "manifest" if key_manifest else "arithmetic", "surfaces": {}}
    for surface, counts in surface_counts.items():
        n_train = int(counts["train"])
        # registered |U|_s = min(8,000, realized train) (plan section 4 part 2);
        # production QA/math/MCQ realize 8,000 exactly, code its realized train.
        pool = _pool_size(surface, n_train)
        budgets = [b for b in l_grid if b <= n_train] + ["full"]
        # (iv) L <= |train| by construction of `budgets`; assert grid non-empty
        if not budgets:
            raise RuntimeError(f"feasibility: {surface} has no feasible L budget")
        # (v) QA disjoint variant arithmetic: |train| >= |U| + L at the anchors
        if surface == "qa":
            for anchor in QA_DISJOINT_ANCHORS:
                if n_train < pool + anchor:
                    raise RuntimeError(
                        f"feasibility: QA disjoint variant infeasible at L={anchor}: "
                        f"train {n_train} < |U| {pool} + L {anchor}"
                    )
        resolved = {}
        if key_manifest is not None:
            sizes = set()
            for fu in FU_CELLS:
                for family in ("linear", "mlp"):
                    key = resolve_map_key(surface, family, fu)
                    meta = key_manifest.get(key)
                    if meta is None:
                        if require_all:
                            raise RuntimeError(
                                f"feasibility: unresolved map key {key} for "
                                f"({surface}, {family}, f_U={fu})"
                            )
                        resolved[key] = {"present": False, "fu": fu}
                        continue
                    resolved[key] = {"realized_u": meta["realized_u"], "fu": fu}
                    sizes.add(int(meta["realized_u"]))
            if len(sizes) > 1 or (require_all and len(sizes) != 1):
                raise RuntimeError(
                    f"feasibility: {surface} realized |U| differs across f_U cells: {sizes}"
                )
        report["surfaces"][surface] = {
            "n_train": n_train,
            "pool": pool,
            "budgets": budgets,
            "resolved_keys": resolved,
        }
    return report


def resolve_map_key(surface: str, family: str, fu: float, *, additive: bool = False) -> str:
    """Key naming: the shared generic-only map serves QA/math/MCQ at f_U=0;
    code's f_U=0 is code's OWN generic-only key (MF-A)."""
    if additive:
        assert surface == "qa" and fu == 0.5
        return f"{family}__qa__additive"
    if fu == 0.0:
        return f"{family}__code__fu0" if surface == "code" else f"{family}__shared__fu0"
    tag = {0.5: "fu05", 1.0: "fu1"}[fu]
    return f"{family}__{surface}__{tag}"


def _pool_size(surface: str, n_train: int) -> int:
    """Registered per-surface map-pool size |U|_s = min(8,000, realized train)
    (plan section 4 part 2); code (POOL_SIZE None) = its realized train."""
    cap = POOL_SIZE[surface]
    return n_train if cap is None else min(cap, n_train)


def assert_partition_membership(table: SurfaceTable, pool_idx: np.ndarray) -> None:
    """Membership form of feasibility clauses (ii) + (iii): the map pool is a
    subset of the TRAIN partition, so dev/test/rung1 are pool-disjoint."""
    bad = np.flatnonzero(table.split[pool_idx] != "train")
    if bad.size:
        raise RuntimeError(
            f"feasibility(ii/iii): {bad.size} map-pool contexts outside the train "
            f"partition on {table.surface} (first splits: "
            f"{table.split[pool_idx[bad[:5]]].tolist()})"
        )


def qa_disjoint_draw(
    train_idx: np.ndarray,
    groups: np.ndarray,
    pool_idx: np.ndarray,
    budget: int,
    seed_parts: list[int],
) -> np.ndarray:
    """Clause (v) GROUP-grain disjoint label draw (QA disjoint variant, MF-G):
    labels drawn only from train groups contributing NO context to the map
    pool; asserts the group-level disjointness post-draw."""
    pool_groups = set(groups[pool_idx].tolist())
    eligible = train_idx[~np.isin(groups[train_idx], sorted(pool_groups))]
    if len(eligible) < int(budget):
        raise RuntimeError(
            f"feasibility(v): disjoint variant infeasible — {len(eligible)} non-pool "
            f"train rows < L={budget}"
        )
    draw = group_respecting_draw(eligible, groups, budget, seed_parts)
    overlap = set(groups[draw].tolist()) & pool_groups
    if overlap:
        raise RuntimeError(f"feasibility(v): pool/draw group overlap: {sorted(overlap)[:5]}")
    return draw


# ---------------------------------------------------------------------------
# maps phase
# ---------------------------------------------------------------------------


def _load_u_pool(
    u_store_dir: Path,
    layers: tuple[int, ...],
    n_pool: int,
    seed: int,
    *,
    hidden_dim: int = HIDDEN_DIM,
):
    """Generic pairs from the banked #1092 U-store (one pair per context).

    Applies ``fit_pool_mask`` (is_eval_only exclusion — the 18,793-of-21,193
    FIT pool; battery eval-only rows must never enter a map pool) and raises
    on a short pool rather than silently under-filling."""
    from explore_persona_space.experiments.issue_1739.store_io import fit_pool_mask, load_summaries

    # FLAT root: stage_u_store FLATTENS the cell's (kind x layer) shards into
    # its dest (store_io docstring: the u-store-staging-layout-unwired fix),
    # and the ported issue1739_fits.py consumes the same flat layout — a
    # cell=U_STORE_CELL nesting here resolved <dir>/cell_inst_own/, which the
    # staged layout never contains (Pod B P4(i) launch failure, 2026-08-20).
    arrs, meta = load_summaries(u_store_dir, ("context_end", "t1"), layers, hidden_dim=hidden_dim)
    mask = fit_pool_mask(meta)
    fit_rows = np.flatnonzero(mask)
    n_avail = int(fit_rows.size)
    if n_pool > n_avail:
        raise RuntimeError(f"U pool {n_pool} > available fit rows {n_avail} in {u_store_dir}")
    rng = np.random.default_rng([SEED0, 1092, seed])
    take = fit_rows[rng.permutation(n_avail)[:n_pool]]
    x = np.stack([arrs[("context_end", ly)][take] for ly in layers], axis=0)
    y = np.stack([arrs[("t1", ly)][take] for ly in layers], axis=0)
    return x, y, {"n_avail": n_avail, "subsample_seed": seed, "n_taken": int(len(take))}


def _pool_indices(table: SurfaceTable, n_target: int, seed: int) -> np.ndarray:
    """Target-surface map-pool contexts: drawn from the TRAIN partition."""
    train_idx = np.flatnonzero(table.split == "train")
    if n_target > len(train_idx):
        raise RuntimeError(f"map pool {n_target} > train {len(train_idx)} on {table.surface}")
    rng = np.random.default_rng([SEED0, 7, seed])
    return np.sort(rng.permutation(train_idx)[:n_target])


def phase_maps(args) -> None:
    """Fit the requested map keys; persist weights + manifest + diagnostics."""
    import torch

    out_dir = Path(args.maps_out)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "key_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    layers = _layer_tuple(args)

    keys = args.keys or []
    if not keys:
        raise SystemExit("--phase maps requires --keys <key> [<key> ...]")
    table = None
    for key in keys:
        if key in manifest and not args.force:
            print(f"[maps] {key}: already in manifest — skip", flush=True)
            continue
        m = re.match(r"^(linear|mlp)__(shared|qa|math|mcq|code)__(fu0|fu05|fu1|additive)$", key)
        if not m:
            raise SystemExit(f"unrecognized map key {key}")
        family, scope, cell = m.groups()
        seed = SEED0
        if scope == "shared" or (scope == "code" and cell == "fu0"):
            pool_n = 8000 if scope == "shared" else _realized_train(args, "code")
            x, y, pool_meta = _load_u_pool(
                Path(args.u_store_dir), layers, pool_n, seed, hidden_dim=args.hidden_dim
            )
            composition = {"generic": pool_n, "target": 0}
        else:
            table = _get_table(args, scope) if table is None or table.surface != scope else table
            pool_n = _pool_size(scope, _realized_train(args, scope))
            if cell == "fu1":
                idx = _pool_indices(table, pool_n, seed)
                assert_partition_membership(table, idx)
                x = table.z_ctx[:, idx].astype(np.float64)
                y = table.z_t1[:, idx].astype(np.float64)
                pool_meta = {"target_idx_sha_n": int(len(idx)), "subsample_seed": seed}
                composition = {"generic": 0, "target": int(len(idx))}
            elif cell == "fu05":
                n_t = pool_n // 2
                idx = _pool_indices(table, n_t, seed)
                assert_partition_membership(table, idx)
                gx, gy, gmeta = _load_u_pool(
                    Path(args.u_store_dir),
                    layers,
                    pool_n - n_t,
                    seed,
                    hidden_dim=args.hidden_dim,
                )
                x = np.concatenate([gx, table.z_ctx[:, idx].astype(gx.dtype)], axis=1)
                y = np.concatenate([gy, table.z_t1[:, idx].astype(gy.dtype)], axis=1)
                pool_meta = {**gmeta, "n_target": int(len(idx)), "subsample_seed": seed}
                composition = {"generic": pool_n - n_t, "target": int(len(idx))}
            else:  # additive (QA): generic 8,000 + target 8,000 ADDED
                idx = _pool_indices(table, pool_n, seed)
                assert_partition_membership(table, idx)
                gx, gy, gmeta = _load_u_pool(
                    Path(args.u_store_dir), layers, pool_n, seed, hidden_dim=args.hidden_dim
                )
                x = np.concatenate([gx, table.z_ctx[:, idx].astype(gx.dtype)], axis=1)
                y = np.concatenate([gy, table.z_t1[:, idx].astype(gy.dtype)], axis=1)
                pool_meta = {**gmeta, "n_target": int(len(idx)), "subsample_seed": seed}
                composition = {"generic": pool_n, "target": int(len(idx))}
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        t0 = time.time()
        if family == "linear":
            fit = F.fit_linear_map(x, y, device=args.device)
            np.savez(
                out_dir / f"{key}.npz",
                w=fit.w,
                x_mu=fit.x_mu,
                x_sd=fit.x_sd,
                y_mu=fit.y_mu,
            )
            diagnostics = fit.diagnostics
        else:
            fit = F.fit_nonlinear_map(x, y, kind="mlp", device=args.device)
            torch.save(
                {"meta": {"map_kind": "mlp", "key": key}, "payloads": list(fit.nl_payloads)},
                out_dir / f"{key}.pt",
            )
            diagnostics = fit.diagnostics
        realized_u = int(x.shape[1])
        per_layer = diagnostics["per_layer"]
        # Result 0(d) mapping-baselines pair SURFACED into the manifest (the
        # r1 review read the manifest, saw only mean_r2_map, and concluded the
        # pair was never computed — it IS computed per layer inside both fit
        # families and persisted to {key}_diagnostics.json; this summary makes
        # it consumable without opening the per-key file).
        knn1 = [
            r["knn"]["euclidean"]["acc_at_k"].get(1, r["knn"]["euclidean"]["acc_at_k"].get("1"))
            for r in per_layer
        ]
        manifest[key] = {
            "realized_u": realized_u,
            "composition": composition,
            "subsample_seed": seed,
            "pool_meta": pool_meta,
            "diagnostics_summary": {
                "mean_r2_map": float(np.mean([r["r2_map"] for r in per_layer])),
                "mean_r2_identity_bias": float(np.mean([r["r2_identity_bias"] for r in per_layer])),
                "mean_knn_acc_at_1_euclidean": float(np.mean([float(v) for v in knn1])),
                "per_layer_file": f"{key}_diagnostics.json",
            },
            "wall_s": round(time.time() - t0, 1),
        }
        (out_dir / f"{key}_diagnostics.json").write_text(json.dumps(diagnostics, default=float))
        tmp = manifest_path.with_name(manifest_path.name + ".tmp")
        tmp.write_text(json.dumps(manifest, indent=2))
        os.replace(tmp, manifest_path)
        print(f"[maps] {key}: |U|={realized_u} wall={manifest[key]['wall_s']}s", flush=True)


def _realized_train(args, surface: str) -> int:
    lab = _load_labeling(Path(args.dv_root) / surface / "labeling.json", surface=surface)
    return sum(1 for r in lab["rows"] if r["eff_split"] == "train")


def _layer_tuple(args) -> tuple[int, ...]:
    """Full production layer set by default; --layers N is a SMOKE SCALE dial
    (reduces layer count, never skips code paths)."""
    n = getattr(args, "layers", None)
    return tuple(range(n)) if n else tuple(range(N_LAYERS))


_TABLE_CACHE: dict[str, SurfaceTable] = {}
_ROSTER_CACHE: dict[str, list[str]] = {}


def _surface_benchmarks(args, surface: str) -> list[str]:
    """Realized benchmark roster per surface (store loading + question attach).

    code is GATE-CONDITIONAL (plan fork 5, r3 Critical 2): derived from the
    labeling.json ``gate_decisions`` echo through the ONE shared resolution
    rule (``issue2388_gen.code_roster_from_gate_fields`` — BCB only on
    ``bcb_fit_allowed``, apps_intro only on ``apps_activated``), then
    cross-validated EXACT-SET against the realized row benchmarks, so the
    stores loaded / questions attached cover exactly the rows the DV dealt —
    never the static candidate tuple (which would demand a dropped BCB store
    and omit an activated APPS store). Other surfaces keep the static rosters.
    """
    if surface != "code":
        return list(SURFACE_BENCHMARKS[surface])
    if "code" in _ROSTER_CACHE:
        return list(_ROSTER_CACHE["code"])
    lab = _load_labeling(Path(args.dv_root) / "code" / "labeling.json", surface="code")
    decisions = lab["payload"].get("gate_decisions") or {}
    if decisions.get("bcb_fit_allowed") is None:
        raise RuntimeError(
            "code labeling.json carries no resolved gate_decisions — rebuild the DV "
            "(issue2388_dv_build.py --surface code embeds the binding code_gate.json verdict)"
        )
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    import issue2388_gen as G

    roster = G.code_roster_from_gate_fields(decisions)
    realized = {str(r["benchmark"]) for r in lab["rows"]}
    if realized != set(roster):
        raise RuntimeError(
            "code labeling rows disagree with the gate roster (store coverage would be "
            f"wrong): rows-only={sorted(realized - set(roster))} "
            f"gate-only={sorted(set(roster) - realized)}"
        )
    _ROSTER_CACHE["code"] = roster
    return list(roster)


def _get_table(args, surface: str) -> SurfaceTable:
    if surface in _TABLE_CACHE:
        return _TABLE_CACHE[surface]
    labeling = Path(args.dv_root) / surface / "labeling.json"
    if surface == "qa":
        store_dirs = [Path(args.qa_store_dir)]
        with_tlast = False
    else:
        store_dirs = [Path(args.store_root) / b for b in _surface_benchmarks(args, surface)]
        with_tlast = True
    table = load_surface_table(
        surface,
        labeling,
        store_dirs,
        layers=_layer_tuple(args),
        with_tlast=with_tlast,
        hidden_dim=args.hidden_dim,
    )
    _TABLE_CACHE[surface] = table
    return table


# ---------------------------------------------------------------------------
# sweep phase
# ---------------------------------------------------------------------------


def group_respecting_draw(
    train_idx: np.ndarray, groups: np.ndarray, budget: int | str, seed_parts: list[int]
) -> np.ndarray:
    """L-budget label draw: whole groups accumulated in a seeded random order,
    last group truncated to hit exactly L (plan: draws respect the group axis)."""
    if budget == "full":
        return train_idx
    rng = np.random.default_rng(seed_parts)
    uniq = np.array(sorted(set(groups[train_idx])))
    order = rng.permutation(len(uniq))
    taken: list[int] = []
    for gi in order:
        members = train_idx[groups[train_idx] == uniq[gi]]
        room = int(budget) - len(taken)
        if room <= 0:
            break
        taken.extend(members[:room].tolist())
    if len(taken) < int(budget):
        raise RuntimeError(f"draw could not reach L={budget} from {len(train_idx)} train rows")
    return np.array(sorted(taken), dtype=int)


def group_permuted_targets(
    dv: np.ndarray, groups: np.ndarray, rows: np.ndarray, n_draws: int, seed_parts: list[int]
) -> np.ndarray:
    """(len(rows), n_draws) permuted DV columns: whole groups' DVs exchanged as
    units (donor group's DV values assigned by within-group rank, cyclic when
    sizes differ) — the section-6 registered group-level exchange."""
    rng = np.random.default_rng(seed_parts)
    sub_groups = groups[rows]
    sub_dv = dv[rows]
    uniq, inv = np.unique(sub_groups, return_inverse=True)
    members = [np.flatnonzero(inv == g) for g in range(len(uniq))]
    out = np.empty((len(rows), n_draws), dtype=np.float64)
    for d in range(n_draws):
        donor = rng.permutation(len(uniq))
        for g, mem in enumerate(members):
            src = members[donor[g]]
            take = sub_dv[src[np.arange(len(mem)) % len(src)]]
            out[mem, d] = take
    return out


_PCA_LAYER_CHUNK = 8  # (chunk, d, d) fp64 covariance stacks: 8 x 3584^2 x 8B ~= 0.8 GB


def _pca_basis(pool_x: np.ndarray, k: int) -> np.ndarray:
    """(Ly, d, k) per-layer PCA basis from the unlabeled pool.

    BATCHED torch eigh over layer-chunked covariance stacks (r1 g6: the serial
    per-layer numpy full SVD is the many-cell dense-factorization class,
    vectorize-many-cell-fits.md). Top-k eigenvectors of the centered
    covariance == the top-k right singular vectors up to per-column sign; the
    downstream ridge on projected features is column-sign-invariant, and the
    equivalence is pinned by a subspace-projector test. CPU-LAPACK fallback per
    slice on a (cuda-class) eigh non-convergence, per the gotchas rule —
    never a jitter.
    """
    import torch

    n_layers, _n, d = pool_x.shape
    v = np.empty((n_layers, d, k), dtype=np.float64)
    for lo in range(0, n_layers, _PCA_LAYER_CHUNK):
        chunk = pool_x[lo : lo + _PCA_LAYER_CHUNK].astype(np.float64)
        x = torch.from_numpy(chunk)
        x = x - x.mean(dim=1, keepdim=True)
        cov = torch.matmul(x.transpose(1, 2), x)  # (chunk, d, d)
        try:
            _w, vecs = torch.linalg.eigh(cov)
        except torch.linalg.LinAlgError:
            # exact numerical-backend swap (never a jitter): per-slice numpy
            vecs = torch.from_numpy(
                np.stack([np.linalg.eigh(cov[i].numpy())[1] for i in range(cov.shape[0])])
            )
        # eigh returns ASCENDING eigenvalues: top-k = last k columns, reversed
        v[lo : lo + cov.shape[0]] = vecs[:, :, -k:].flip(-1).numpy()
    return v


def _load_map(maps_dir: Path, key: str, *, device: str = "cpu"):
    npz = maps_dir / f"{key}.npz"
    if npz.exists():
        z = np.load(npz)
        return F.MapFit(w=z["w"], x_mu=z["x_mu"], x_sd=z["x_sd"], y_mu=z["y_mu"], diagnostics={})
    import torch

    # weights_only=False: self-produced bundle (this driver's own phase_maps
    # writer on the same out-root), never a third-party download.
    bundle = torch.load(maps_dir / f"{key}.pt", weights_only=False)
    return F.MapFit(
        w=None,
        x_mu=None,
        x_sd=None,
        y_mu=None,
        diagnostics=bundle["meta"],
        kind="mlp",
        nl_payloads=tuple(bundle["payloads"]),
        apply_device=device,
    )


def _shuffled_nl_payloads(payloads: tuple, seed: int) -> tuple:
    """Input-column weight shuffle of each layer's MLP payload (bl_shufmap_mlp).

    Permutes ONLY the first linear layer's input columns; the input
    standardizer (xmu/xsd) stays BOUND to the actual features — the registered
    input-column weight shuffle destroys the learned feature<->weight
    ALIGNMENT while every standardized input keeps its own (unit) scale, so
    per-layer output distributions are scale-preserved (r2 Codex
    shuffled-mlp-normalization-misaligned: co-permuting xmu/xsd standardized
    each raw feature with ANOTHER feature's mean/std — arbitrary scale/offset
    artifacts, not a weight shuffle). Exactly the MLP analogue of the linear
    control's input-dim row permutation (``shuffled_map_weights``, which
    permutes weight ROWS with the standardizer untouched); capacity + weight
    norms preserved EXACTLY (asserted). Equivalence: the shuffled payload
    applied to x equals the ORIGINAL net applied to the standardized input
    permuted by the inverse permutation (pinned test).
    """
    rng = np.random.default_rng([SEED0, 13, int(seed)])
    out = []
    for p in payloads:
        assert p.get("kind") == "mlp", p.get("kind")
        sd = p["state_dict"]
        w0 = sd["0.weight"]  # (hidden, d_in) torch tensor
        perm = rng.permutation(int(w0.shape[1]))
        w0_shuf = w0[:, perm]
        # Norms compared in float64: a column permutation preserves the value
        # multiset exactly, but fp32 reduction-order noise on a 512x3584 tensor
        # can exceed 1e-6 relative (R12: mlp__code__fu05 layer 14, ~6e-6).
        n0 = float(w0.double().norm())
        n1 = float(w0_shuf.double().norm())
        assert n0 == n1 or abs(n0 - n1) < 1e-9 * n0
        q = dict(p)
        q["state_dict"] = {**sd, "0.weight": w0_shuf}
        out.append(q)
    return tuple(out)


def _apply_family(x: np.ndarray, mapfit, family: str, *, shuffle_seed: int | None = None):
    x64 = x.astype(np.float64)
    if family == "linear":
        w = mapfit.w
        if shuffle_seed is not None:
            w = F.shuffled_map_weights(w, seed=shuffle_seed)
        return F.apply_map(x64, mapfit, w=w)
    if shuffle_seed is not None:
        import dataclasses

        shuf = dataclasses.replace(
            mapfit, nl_payloads=_shuffled_nl_payloads(mapfit.nl_payloads, shuffle_seed)
        )
        return F.apply_nl_map(x64, shuf)
    return F.apply_nl_map(x64, mapfit)


def _surface_features(table: SurfaceTable, train_rows: np.ndarray) -> np.ndarray:
    """(n, k) feature matrix (bl_feats): lengths + one-hots + TF-IDF fit on
    the TRAIN rows only (no eval leakage)."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    questions = table.meta.get("questions")
    if questions is None:
        raise RuntimeError("surface features require question text in table.meta['questions']")
    n = len(table.ctx_ids)
    cols: list[np.ndarray] = []
    qlen = np.array([len(q) for q in questions], dtype=np.float64)
    cols.append(qlen[:, None])
    cols.append(np.array([len(q.split()) for q in questions], dtype=np.float64)[:, None])
    if table.surface == "math":
        lv = np.nan_to_num(table.level, nan=0.0)
        cols.append((lv[:, None] == np.arange(1, 6)[None, :]).astype(np.float64))
    if table.surface == "mcq":
        cats = sorted(set(table.category))
        cols.append((table.category[:, None] == np.array(cats)[None, :]).astype(np.float64))
    if table.surface == "code":
        benches = sorted(set(table.benchmark))
        cols.append((table.benchmark[:, None] == np.array(benches)[None, :]).astype(np.float64))
    if table.surface == "qa" and "alias_counts" in table.meta:
        cols.append(np.asarray(table.meta["alias_counts"], dtype=np.float64)[:, None])
    vec = TfidfVectorizer(max_features=512)
    vec.fit([questions[i] for i in train_rows])
    tfidf = vec.transform(questions).toarray()
    cols.append(tfidf)
    feats = np.concatenate(cols, axis=1)
    return feats[None, :, :]  # single "layer" slice


def _encoder_features(table: SurfaceTable, cache_dir: Path, device: str) -> np.ndarray:
    """(1, n, k) frozen sentence-embedding of the question text (bl_extemb)."""
    cache = cache_dir / f"{table.surface}_extemb.npy"
    if cache.exists():
        emb = np.load(cache)
    else:
        from sentence_transformers import SentenceTransformer

        questions = table.meta.get("questions")
        if questions is None:
            raise RuntimeError("bl_extemb requires question text in table.meta['questions']")
        enc = SentenceTransformer(ENCODER_ID, device=device)
        emb = enc.encode(questions, batch_size=256, show_progress_bar=False)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache, np.asarray(emb, dtype=np.float32))
        emb = np.load(cache)
    return emb.astype(np.float64)[None, :, :]


QA_SHARDS_HF_PREFIX = "issue1739_ctxmap/raw_completions"
_QA_SHARD_GLOB = "labeling_hallucination.shard*.jsonl"
_QA_SHARD_SRC_RE = re.compile(r"^labeling/hallucination/(.+)_seed\d+\.json$")


def _qa_questions_from_shards(shards_dir: Path) -> tuple[dict[str, str], dict[str, int]]:
    """context_id -> question text (+ gold-alias counts) from the banked #1739
    packed labeling raw-completion shards (concern qa-question-text-source).

    Probe-verified schema (``{QA_SHARDS_HF_PREFIX}/labeling_hallucination.
    shard00.jsonl``, 26 shards, ~9 MB each): packed rows ``{"src", "doc"}``;
    the per-rollout docs (``src == labeling/hallucination/<context_id>_seed<k>
    .json``) carry ``context_id`` + ``query`` + ``prefix_text`` +
    ``answer_aliases``; the one ``_manifest.json`` row is FILTERED OUT on the
    ``src`` discriminator (packed-format consumer rule). The join to the
    banked labeling.json rows is EXACT on ``context_id`` — both minted by the
    same producer (``experiments/issue_1739/corpus_staging._to_contexts``).
    Question text follows the #1739 features convention (prefix_text + query,
    newline-joined). Rollouts of one context must agree — a conflict is a
    corrupted shard set, refused."""
    shard_paths = sorted(Path(shards_dir).glob(_QA_SHARD_GLOB))
    if not shard_paths:
        raise RuntimeError(
            f"no {_QA_SHARD_GLOB} under {shards_dir} — stage the banked #1739 labeling "
            f"raw-completion shards (HF data repo: {QA_SHARDS_HF_PREFIX}/) first"
        )
    q_by_ctx: dict[str, str] = {}
    alias_by_ctx: dict[str, int] = {}
    for sp in shard_paths:
        with sp.open(encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                r = json.loads(line)
                if not _QA_SHARD_SRC_RE.match(str(r.get("src", ""))):
                    continue  # manifest / foreign-source rows never join
                d = r["doc"]
                cid = str(d["context_id"])
                text = "\n".join(p for p in (d.get("prefix_text"), d.get("query")) if p)
                prior = q_by_ctx.get(cid)
                if prior is not None and prior != text:
                    raise RuntimeError(f"conflicting question text across rollouts for {cid}")
                q_by_ctx[cid] = text
                alias_by_ctx[cid] = len(d.get("answer_aliases") or [])
    return q_by_ctx, alias_by_ctx


def _attach_questions(args, table: SurfaceTable) -> None:
    """Question text per context (feature baselines): gen loaders on the new
    surfaces; the banked #1739 labeling raw-completion shards on QA
    (``--qa-questions-shards`` — concern qa-question-text-source)."""
    if "questions" in table.meta:
        return
    if table.surface == "qa":
        if not args.qa_questions_shards:
            raise RuntimeError(
                "QA question text rides the banked #1739 labeling raw-completion shards — "
                f"pass --qa-questions-shards <dir with {_QA_SHARD_GLOB}> "
                f"(HF data repo: {QA_SHARDS_HF_PREFIX}/)"
            )
        q_by_id, alias_by_id = _qa_questions_from_shards(Path(args.qa_questions_shards))
        table.meta["alias_counts"] = [alias_by_id.get(c, 0) for c in table.ctx_ids]
    else:
        sys.path.insert(0, str(REPO_ROOT / "scripts"))
        import issue2388_gen as G

        q_by_id = {}
        for bench in _surface_benchmarks(args, table.surface):
            for it in G.LOADERS[bench]():
                q_by_id[it["item_id"]] = it.get("question") or it.get("prompt") or ""
    table.meta["questions"] = [q_by_id.get(c, "") for c in table.ctx_ids]
    n_empty = sum(1 for q in table.meta["questions"] if not q)
    if n_empty / len(table.ctx_ids) > 0.01:
        raise RuntimeError(f"{table.surface}: {n_empty} contexts without question text")


def _rung_eval_sets(table: SurfaceTable) -> dict[str, np.ndarray]:
    """rung0 = locked test split; rung1 = the section-4 shift eval rows."""
    test = np.flatnonzero(table.split == "test")
    out = {"rung0": test}
    s = table.surface
    if s == "math":
        out["rung1"] = test[np.isin(table.level[test], sorted(MATH_RUNG1_EVAL_LEVELS))]
    elif s == "mcq":
        cats = sorted(set(table.category))
        rng = np.random.default_rng([SEED0, 11])
        held = set(rng.permutation(np.array(cats))[:MCQ_RUNG1_HELDOUT_K].tolist())
        out["rung1"] = test[np.isin(table.category[test], sorted(held))]
        out["_rung1_heldout_categories"] = np.array(sorted(held))
    elif s == "code":
        out["rung1"] = test[np.isin(table.benchmark[test], sorted(CODE_RUNG1_EVAL))]
    elif s == "qa":
        # NQ-Open rows carry eff_split == "rung1" (_effective_rows derivation).
        out["rung1"] = np.flatnonzero(table.split == "rung1")
    return out


def _rung1_fit_filter(
    table: SurfaceTable, rows: np.ndarray, heldout_categories: np.ndarray | None = None
) -> np.ndarray:
    """FIT-side restriction for the rung-1 transfer read (r1 g6 Critical 2).

    math: fit on levels 1-3 only (registered "levels 1-3 -> 4-5"); code: fit
    on {HE, MBPP, LeetCode} only ("-> {BCB, LCB}"); mcq: EXCLUDE the held-out
    eval categories from the fit; qa: identity (train is TriviaQA-only by
    construction — NQ-Open rows live outside the train partition).
    """
    s = table.surface
    if s == "math":
        return rows[np.isin(table.level[rows], sorted(MATH_RUNG1_FIT_LEVELS))]
    if s == "code":
        return rows[np.isin(table.benchmark[rows], sorted(CODE_RUNG1_FIT))]
    if s == "mcq":
        if heldout_categories is None:
            raise RuntimeError("mcq rung1 fit filter requires the held-out category set")
        return rows[~np.isin(table.category[rows], heldout_categories)]
    return rows


def _rung1_realization(table: SurfaceTable) -> dict | None:
    """Code-surface rung-1 PLANNED-vs-REALIZED benchmark disclosure.

    r4 Major / reconciler-binding ``code-rung1-realization-undisclosed``: under
    the fork-5 BCB-DROP gate the registered ``{bigcodebench_full, lcb_v5}``
    rung-1 eval realizes as LCB-only, and ``_rung_eval_sets`` would silently
    label the reduced cohort ``rung1``. This dict rides every code cell row and
    both select-phase aggregates so no downstream read mistakes the reduced
    cohort for the registered transfer read. Counts are PARTITION-level
    (rung-1 eval rows in test; rung-1-fit-eligible rows in train — per-draw fit
    rows are budget-subsampled from the latter). None on non-code surfaces
    (their rung-1 sets are not benchmark-roster-conditional)."""
    if table.surface != "code":
        return None
    test = np.flatnonzero(table.split == "test")
    r1 = test[np.isin(table.benchmark[test], sorted(CODE_RUNG1_EVAL))]
    train = np.flatnonzero(table.split == "train")
    fit = train[np.isin(table.benchmark[train], sorted(CODE_RUNG1_FIT))]
    realized_eval = sorted(set(table.benchmark[r1].tolist()))
    realized_fit = sorted(set(table.benchmark[fit].tolist()))
    bcb_dropped = "bigcodebench_full" not in realized_eval
    return {
        "planned_fit": sorted(CODE_RUNG1_FIT),
        "planned_eval": sorted(CODE_RUNG1_EVAL),
        "realized_fit": realized_fit,
        "realized_eval": realized_eval,
        "n_eval_rows_by_benchmark": {
            b: int((table.benchmark[r1] == b).sum()) for b in realized_eval
        },
        "n_fit_rows_by_benchmark": {
            b: int((table.benchmark[fit] == b).sum()) for b in realized_fit
        },
        "bcb_dropped_by_gate": bcb_dropped,
        "reason": (
            "bigcodebench_full absent from realized rows — dropped by the fork-5 gate "
            "(bcb_fit_allowed=False); the rung-1 transfer eval is LCB-only"
            if bcb_dropped
            else None
        ),
    }


RUNG1_REFIT_MIN_N = 20  # production draws always clear this (see arithmetic in the r2 report)


def _assert_rung1_disjoint(
    table: SurfaceTable, fit_rows: np.ndarray, heldout_categories: np.ndarray | None
) -> None:
    """Fail loud if any rung-1 FIT row carries the rung-1 EVAL property."""
    s = table.surface
    if s == "math":
        bad = int(np.isin(table.level[fit_rows], sorted(MATH_RUNG1_EVAL_LEVELS)).sum())
    elif s == "code":
        bad = int(np.isin(table.benchmark[fit_rows], sorted(CODE_RUNG1_EVAL)).sum())
    elif s == "mcq":
        assert heldout_categories is not None
        bad = int(np.isin(table.category[fit_rows], heldout_categories).sum())
    else:  # qa: fit rows are train-partition rows; rung1 rows carry split=="rung1"
        bad = int((table.split[fit_rows] == "rung1").sum())
    if bad:
        raise RuntimeError(f"rung1 fit/eval property overlap on {s}: {bad} fit rows")


def _rung1_refit(
    table: SurfaceTable,
    draw_rows: np.ndarray,
    x_full: np.ndarray,
    v: np.ndarray | None,
    rung1_rows: np.ndarray,
    heldout_categories: np.ndarray | None,
    n_null: int,
    seed_parts: list[int],
    device: str,
) -> tuple[np.ndarray, dict, dict] | None:
    """Second fit for the rung-1 TRANSFER read: same basis, fit rows restricted
    to the registered shift-source subset of the draw. Returns
    (preds (Ly, n_rung1, 1+n_null), metrics block, info) or None when the
    restricted draw is under the floor (caller decides smoke-vs-raise)."""
    fit_rows = _rung1_fit_filter(table, draw_rows, heldout_categories)
    if len(fit_rows) == 0:
        raise RuntimeError(f"{table.surface}: rung1 restricted fit has 0 rows")
    _assert_rung1_disjoint(table, fit_rows, heldout_categories)
    if len(fit_rows) < RUNG1_REFIT_MIN_N:
        return None
    y_null = group_permuted_targets(table.dv, table.boot_group, fit_rows, n_null, [*seed_parts, 7])
    y_stack = np.concatenate([table.dv[fit_rows][:, None], y_null], axis=1)
    if v is not None:
        x_tr = np.einsum("lnd,ldk->lnk", x_full[:, fit_rows].astype(np.float64), v)
        x_ev = np.einsum("lnd,ldk->lnk", x_full[:, rung1_rows].astype(np.float64), v)
    else:
        x_tr = x_full[:, fit_rows]
        x_ev = x_full[:, rung1_rows]
    y_tr = np.broadcast_to(y_stack[None, :, :], (x_tr.shape[0],) + y_stack.shape).copy()
    telem: list[dict] = []
    preds = F.ridge_gcv_predict_per_target(
        x_tr, y_tr, [x_ev], dof_cap=DOF_CAP, device=device, selector_telemetry=telem
    )[0]
    block = _metrics_block(preds, table.dv[rung1_rows])
    info = {
        "refit": True,
        "n_fit_rows": int(len(fit_rows)),
        "restriction": {
            "math": "levels 1-3 only",
            "code": "HE/MBPP/LeetCode only",
            "mcq": "held-out eval categories excluded",
            "qa": "identity (fit-disjoint by construction)",
        }[table.surface],
    }
    return preds, block, info


def _metrics_block(preds: np.ndarray, dv_eval: np.ndarray) -> dict[str, np.ndarray]:
    """preds (Ly, n_eval, n_t) -> per (layer, target) rho / r2 / auroc arrays."""
    n_layers, n_eval, n_t = preds.shape
    flat = preds.transpose(0, 2, 1).reshape(n_layers * n_t, n_eval)
    rho = spearman_rows(flat, dv_eval).reshape(n_layers, n_t)
    ss_tot = float(((dv_eval - dv_eval.mean()) ** 2).sum()) or 1e-12
    r2 = 1.0 - ((flat - dv_eval[None, :]) ** 2).sum(axis=1) / ss_tot
    r2 = r2.reshape(n_layers, n_t)
    labels = (dv_eval >= 0.5).astype(np.float64)
    if 0 < labels.sum() < len(labels):
        auroc = auroc_rows(flat, labels).reshape(n_layers, n_t)
    else:
        auroc = np.full((n_layers, n_t), np.nan)
    return {"rho": rho, "r2": r2, "auroc": auroc}


def _direction_scores(
    table: SurfaceTable,
    draw_rows: np.ndarray,
    x_all: np.ndarray,
    n_null: int,
    seed_parts: list[int],
) -> tuple[np.ndarray, int]:
    """One-parameter direction readout: r_correct from the draw's spread
    contexts (within-context matched split), score = <r, x> per layer.

    Returns (scores (Ly, n_ctx, 1 + n_null), n_spread). Null directions refit
    per permuted target via the batched GEMM (weights x activations)."""
    if table.spread_roll_t1 is None:
        raise RuntimeError("direction arm needs rollout-grain t1 (with_rollout_grain=True)")
    draw_set = set(draw_rows.tolist())
    in_draw = np.array([ci in draw_set for ci in table.spread_roll_ctx])
    roll_ctx = table.spread_roll_ctx[in_draw]
    roll_y = table.spread_roll_y[in_draw]
    acts = table.spread_roll_t1[:, in_draw].astype(np.float64)  # (Ly, m, d)
    if roll_ctx.size == 0:
        return np.zeros((x_all.shape[0], x_all.shape[1], 1 + n_null)), 0

    def _weights(y_roll: np.ndarray) -> np.ndarray:
        """Per-rollout +/- weights: within-context (correct-mean - incorrect-
        mean), averaged over spread contexts (the E2 matched-split shape)."""
        w = np.zeros(len(y_roll))
        n_ctx_used = 0
        for ci in np.unique(roll_ctx):
            mem = np.flatnonzero(roll_ctx == ci)
            pos = mem[y_roll[mem] > 0.5]
            neg = mem[y_roll[mem] <= 0.5]
            if len(pos) == 0 or len(neg) == 0:
                continue
            w[pos] += 1.0 / len(pos)
            w[neg] -= 1.0 / len(neg)
            n_ctx_used += 1
        if n_ctx_used:
            w /= n_ctx_used
        return w

    w_obs = _weights(roll_y)
    n_spread = int((w_obs != 0).sum() and len(np.unique(roll_ctx[w_obs != 0])))
    rng = np.random.default_rng(seed_parts)
    w_cols = [w_obs]
    ctx_uniq = np.unique(roll_ctx)
    for _ in range(n_null):
        # group-level exchange of per-rollout labels between contexts
        donor = rng.permutation(len(ctx_uniq))
        y_perm = np.empty_like(roll_y)
        members = [np.flatnonzero(roll_ctx == c) for c in ctx_uniq]
        for g, mem in enumerate(members):
            src = members[donor[g]]
            y_perm[mem] = roll_y[src[np.arange(len(mem)) % len(src)]]
        w_cols.append(_weights(y_perm))
    w_mat = np.stack(w_cols, axis=0)  # (1+n_null, m)
    # r (Ly, 1+n_null, d) = w_mat @ acts, batched over layers
    r = np.einsum("tm,lmd->ltd", w_mat, acts)
    scores = np.einsum("lnd,ltd->lnt", x_all.astype(np.float64), r)
    return scores, n_spread


POOLING_OF_ARM = {
    "arm_ctx": "context_end",
    "arm_maplin": "t1-mapped",
    "arm_mapmlp": "t1-mapped",
    "arm_oracle": "t1",
    "arm_oracle_tlast": "t_last",
    "bl_shufmap": "t1-mapped",
    "bl_shufmap_mlp": "t1-mapped",
    "bl_identity": "context_end+bias",
    "bl_feats": "text",
    "bl_extemb": "text",
    "arm_dir_ctx": "context_end",
    "arm_dir_map": "t1-mapped",
    "bl_const": "none",
    "bl_agree": "rollout-agreement (no fit)",
    "arm_ctx_pca": "context_end",
}


def _file_sha(path: Path) -> str | None:
    """First 12 hex of a file's sha256 (bytes read from disk — hash-safe);
    None when absent (availability itself is part of the pinned regime)."""
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:12]


def _pin_map_payloads(
    out_root: Path, maps_dir: Path, filenames: tuple[str, ...], *, force: bool
) -> None:
    """UPSTREAM payload digests (reconciler r3 deferral
    long-loop-restartability-upstream-digests, sweep half).

    The regime sentinel pins the map MANIFEST bytes but not the consumed map
    payload files, so a refit map under an unchanged manifest (or a
    hand-swapped payload) would silently ride a stale-cell resume. A MERGED
    per-file registry (``map_payload_shas.json``) keeps cross-map_cell
    incremental fills legal (different invocations consume different keys):
    only payloads that EXIST are recorded, a recorded payload whose bytes
    changed REFUSES the resume, and ``--force`` overwrites the entries
    alongside the cells it recomputes."""
    payload_reg_p = out_root / "map_payload_shas.json"
    current = {
        fname: sha for fname in filenames if (sha := _file_sha(maps_dir / fname)) is not None
    }
    prior: dict[str, str] = json.loads(payload_reg_p.read_text()) if payload_reg_p.exists() else {}
    if not force:
        stale = {
            f: {"prior": prior[f], "current": s}
            for f, s in current.items()
            if f in prior and prior[f] != s
        }
        if stale:
            raise RuntimeError(
                f"map payload bytes changed under {payload_reg_p}: {stale} — a refit map "
                "invalidates the persisted cells; use a fresh --fits-root (or --force to "
                "recompute)"
            )
    payload_reg_p.write_text(json.dumps({**prior, **current}, indent=1))


def phase_sweep(args) -> None:  # noqa: C901 — single dispatch block over the arm registry
    surface = args.surface
    if not surface:
        raise SystemExit("--phase sweep requires --surface")
    map_cell = args.map_cell
    if args.qa_disjoint and surface != "qa":
        raise SystemExit("--qa-disjoint is a QA-only variant (plan section 4 part 3)")
    if map_cell == "additive" and surface != "qa":
        raise SystemExit("the additive map variant is QA-only (plan section 4)")
    table = _get_table(args, surface)
    out_root = Path(args.fits_root) / surface
    (out_root / "cells").mkdir(parents=True, exist_ok=True)
    (out_root / "nulls").mkdir(parents=True, exist_ok=True)
    (out_root / "preds").mkdir(parents=True, exist_ok=True)

    train_idx = np.flatnonzero(table.split == "train")
    dev_idx = np.flatnonzero(table.split == "dev")
    if dev_idx.size == 0:
        raise RuntimeError(f"{surface}: empty dev partition — selection impossible")
    rungs = _rung_eval_sets(table)
    rung1_meta = _rung1_realization(table)
    if rung1_meta is not None and rung1_meta["bcb_dropped_by_gate"]:
        print(f"[sweep] rung-1 realization: {rung1_meta['reason']}", flush=True)
    eval_names = ["dev", "rung0"] + (["rung1"] if "rung1" in rungs else [])
    n_train, n_dev = len(train_idx), len(dev_idx)
    d_model = int(table.z_ctx.shape[2])
    print(
        f"[sweep] {surface}: train={n_train} dev={n_dev} test={len(rungs['rung0'])} "
        f"d={d_model} map_cell={map_cell} (n_train vs d logged pre-fit)",
        flush=True,
    )

    budgets = [b for b in L_GRID if b <= n_train] + ["full"]
    if args.budgets:
        budgets = [b if b == "full" else int(b) for b in args.budgets]
    # Registered-anchor enforcement BEFORE any write (r1 g6 clause iv/v: a
    # default-budget disjoint run computed unregistered cells then fail-louded
    # mid-loop; H4 cells are registered at the FU_L_ANCHORS only).
    if args.qa_disjoint:
        if not args.budgets:
            budgets = [b for b in QA_DISJOINT_ANCHORS if b <= n_train]
        bad = [b for b in budgets if b not in QA_DISJOINT_ANCHORS]
        if bad:
            raise SystemExit(
                f"--qa-disjoint runs at the registered anchors {QA_DISJOINT_ANCHORS} only; "
                f"got {bad}"
            )
    elif map_cell != "fu1":
        if not args.budgets:
            budgets = [b for b in FU_L_ANCHORS if b == "full" or int(b) <= n_train]
        bad = [b for b in budgets if b not in FU_L_ANCHORS]
        if bad:
            raise SystemExit(
                f"H4 composition cells run at the registered anchors {FU_L_ANCHORS} only; got {bad}"
            )
    n_null = args.n_null
    # Out-root REGIME sentinel: resume keys ignore output-affecting flags
    # (r1 g6 — n_null/layers/hidden_dim are not in the cell filename), so a
    # re-parameterized rerun into the same root would silently reuse stale
    # cells. The sentinel pins the regime; a mismatch refuses (fresh root or
    # --force overwrites the sentinel alongside the cells it will recompute).
    # UPSTREAM digests (r2 long-loop-restartability): the labeling + map
    # manifest are file bytes read from disk (hash-safe per the float-key
    # rule) — a regenerated DV or refit map set refuses a stale-cell resume.
    # map_cell / qa_disjoint / budgets / draws are DELIBERATELY absent: they
    # are keyed in the cell filenames (tag_prefix + per-cell tags), so
    # incremental fills into one root stay legal.
    regime = {
        "n_null": int(n_null),
        "layers": list(_layer_tuple(args)),
        "hidden_dim": int(args.hidden_dim),
        "dof_cap": DOF_CAP,
        "smoke": bool(args.smoke),
        "labeling_sha": _file_sha(Path(args.dv_root) / surface / "labeling.json"),
        "map_manifest_sha": _file_sha(Path(args.maps_out) / "key_manifest.json"),
    }
    sent_p = out_root / "sweep_regime.json"
    if sent_p.exists() and not args.force:
        prior = json.loads(sent_p.read_text())
        if prior != regime:
            raise RuntimeError(
                f"sweep regime mismatch at {sent_p}: prior {prior} != current {regime} — "
                "use a fresh --fits-root (or --force to recompute)"
            )
    sent_p.write_text(json.dumps(regime, indent=1))
    maps_dir = Path(args.maps_out)
    manifest = json.loads((maps_dir / "key_manifest.json").read_text())
    # manifest-mode feasibility re-assert before ANY fit (plan section 4).
    # An H4 cell sweep (map_cell != fu1) requires the FULL per-surface key set
    # resolved + |U|-identical; the primary fu1 sweep tolerates unfit H4 keys
    # (identity still asserted over whatever is present).
    assert_joint_feasibility(
        {surface: {"train": n_train, "dev": n_dev, "test": len(rungs["rung0"])}},
        key_manifest=manifest,
        require_all=(map_cell != "fu1"),
    )

    # bases: ambient + PCA-k from the surface's registered map-pool contexts.
    # Clauses (ii)+(iii): the pool is train-only, so dev/test never enter it.
    pool_idx = _pool_indices(table, _pool_size(surface, n_train), SEED0)
    assert_partition_membership(table, pool_idx)
    bases: dict[str, np.ndarray | None] = {"ambient": None}
    for k in PCA_KS:
        if k <= min(len(pool_idx), d_model):
            bases[f"pca{k}"] = _pca_basis(table.z_ctx[:, pool_idx], k)
        else:
            print(f"[sweep] pca{k} skipped (pool {len(pool_idx)} / d {d_model} < k)", flush=True)

    # arm feature spaces (per basis applied at fit time); mapped arms consume
    # the --map-cell keys (fu1 primary; fu0/fu05/additive = the H4 cells)
    if map_cell == "additive":
        lin_key = resolve_map_key(surface, "linear", 0.5, additive=True)
        mlp_key = resolve_map_key(surface, "mlp", 0.5, additive=True)
    else:
        fu = {"fu0": 0.0, "fu05": 0.5, "fu1": 1.0}[map_cell]
        lin_key = resolve_map_key(surface, "linear", fu)
        mlp_key = resolve_map_key(surface, "mlp", fu)
    _pin_map_payloads(out_root, maps_dir, (f"{lin_key}.npz", f"{mlp_key}.pt"), force=args.force)
    lin_map = _load_map(maps_dir, lin_key, device=args.device)
    mlp_map = _load_map(maps_dir, mlp_key, device=args.device)
    x_ctx = table.z_ctx
    x_maplin = _apply_family(x_ctx, lin_map, "linear").astype(np.float16)
    x_mapmlp = _apply_family(x_ctx, mlp_map, "mlp").astype(np.float16)
    x_shuf = _apply_family(x_ctx, lin_map, "linear", shuffle_seed=SEED0).astype(np.float16)
    # shuffled-MLP control (plan section 5: BOTH families get the matched-
    # capacity shuffle control; r1 concern shuffled-mlp-control-absent).
    x_shufmlp = _apply_family(x_ctx, mlp_map, "mlp", shuffle_seed=SEED0).astype(np.float16)
    # identity + learned bias (mandated baseline): v_hat = x + (y_mu - x_mu) —
    # a pure mean shift (NEVER routed through the map's x_sd standardization).
    bias = (lin_map.y_mu - lin_map.x_mu).astype(np.float32)  # (Ly, 1, d)
    x_ident = np.empty_like(x_ctx)
    for li in range(x_ctx.shape[0]):
        x_ident[li] = (x_ctx[li].astype(np.float32) + bias[li]).astype(np.float16)
    arm_x: dict[str, np.ndarray] = {
        "arm_ctx": x_ctx,
        "arm_maplin": x_maplin,
        "arm_mapmlp": x_mapmlp,
        "arm_oracle": table.z_t1,
        "bl_shufmap": x_shuf,
        "bl_shufmap_mlp": x_shufmlp,
        "bl_identity": x_ident,
    }
    if table.z_tlast is not None:
        arm_x["arm_oracle_tlast"] = table.z_tlast
    # feature baselines (single-slice bases; TF-IDF refit per draw is
    # deliberately NOT done — fit on the full train partition once, a
    # disclosed simplification recorded in the cell rows). QA: the banked
    # labeling carries no question text — question text joins from the banked
    # #1739 labeling raw-completion shards via --qa-questions-shards (exact
    # context_id key; concern qa-question-text-source), fail loud on request
    # without them.
    dir_available = table.spread_roll_t1 is not None
    qa_text_available = surface != "qa" or bool(args.qa_questions_shards)
    feature_arms: list[str] = []
    if args.arms is None:
        feature_arms = ["bl_feats", "bl_extemb"] if qa_text_available else []
    else:
        feature_arms = [a for a in args.arms if a in ("bl_feats", "bl_extemb")]
        if feature_arms and not qa_text_available:
            raise RuntimeError(
                "bl_feats/bl_extemb need question text; the banked QA labeling carries "
                "none — pass --qa-questions-shards <staged shards dir> (HF data repo: "
                f"{QA_SHARDS_HF_PREFIX}/; concern qa-question-text-source)"
            )
        bad_dir = [a for a in args.arms if a.startswith("arm_dir_") and not dir_available]
        if bad_dir:
            raise RuntimeError(
                f"{bad_dir} need per-rollout scores; {surface}'s labeling carries none"
            )
    single_slice: dict[str, np.ndarray] = {}
    if feature_arms:
        _attach_questions(args, table)
        if "bl_feats" in feature_arms:
            single_slice["bl_feats"] = _surface_features(table, train_idx)
        if "bl_extemb" in feature_arms:
            single_slice["bl_extemb"] = _encoder_features(table, out_root / "cache", args.device)

    dir_arms = ["arm_dir_ctx", "arm_dir_map"] if dir_available else []
    agree_arr = table.meta.get("agree_frac")
    agree_available = agree_arr is not None and bool(np.isfinite(agree_arr).any())
    agree_arms = ["bl_agree"] if agree_available else []
    arms = args.arms or (list(arm_x) + list(single_slice) + dir_arms + agree_arms + ["bl_const"])
    if "bl_agree" in arms and not agree_available:
        raise RuntimeError(
            "bl_agree needs agree_frac in the labeling (math/mcq only; code/qa N/A — "
            "answer identity not programmatically extractable there)"
        )
    unknown = [
        a
        for a in arms
        if a not in arm_x
        and a not in single_slice
        and a not in ("arm_dir_ctx", "arm_dir_map", "bl_const", "bl_agree")
    ]
    if unknown:
        raise SystemExit(f"unknown arm(s) {unknown}")
    tag_prefix = ("disjoint_" if args.qa_disjoint else "") + (
        f"{map_cell}_" if map_cell != "fu1" else ""
    )
    # rung-1 transfer reads need a FIT-side restriction on math/mcq/code
    # (r1 g6 Critical 2: the unrestricted-fit rung1 read is in-distribution).
    r1_heldout = rungs.get("_rung1_heldout_categories")
    needs_r1_refit = surface in ("math", "mcq", "code") and "rung1" in eval_names
    for budget in budgets:
        for draw_i in range(args.n_draws):
            seed_parts = [SEED0, _stable_seed(surface), _budget_seed(budget), draw_i]
            if args.qa_disjoint:
                if budget == "full":
                    raise SystemExit("--qa-disjoint runs at the registered L anchors only")
                # clause (v): GROUP-grain pool/draw disjointness (MF-G)
                draw_rows = qa_disjoint_draw(
                    train_idx, table.group, pool_idx, int(budget), seed_parts
                )
            else:
                draw_rows = group_respecting_draw(train_idx, table.group, budget, seed_parts)
            # clause (iii): every labeled draw stays inside the train partition
            draw_splits = set(np.unique(table.split[draw_rows]).tolist())
            if draw_splits != {"train"}:
                raise RuntimeError(f"feasibility(iii): draw escaped train: {draw_splits}")
            # permutation exchange at the section-6 exchangeability axis
            y_null = group_permuted_targets(
                table.dv,
                table.boot_group,
                draw_rows,
                n_null,
                [SEED0, 3, _budget_seed(budget), draw_i],
            )
            y_stack = np.concatenate([table.dv[draw_rows][:, None], y_null], axis=1)
            cell_tag = f"{tag_prefix}L{budget}_draw{draw_i}"

            def _rung1_swap(cand: dict, arm_name: str, x_full: np.ndarray) -> dict:
                """Swap cand's rung1 block/preds for the restricted-fit transfer
                read at the dev-selected basis; rewrite that basis's nulls npz."""
                if not needs_r1_refit:
                    if surface == "qa" and "rung1" in eval_names:
                        return {"refit": False, "reason": "qa fit-disjoint by construction"}
                    return {"refit": False, "reason": "no rung1 eval"}
                v_sel = bases.get(cand["basis"])
                res = _rung1_refit(
                    table,
                    draw_rows,
                    x_full,
                    v_sel,
                    rungs["rung1"],
                    r1_heldout,
                    n_null,
                    [SEED0, 9, _budget_seed(budget), draw_i],
                    args.device,
                )
                if res is None:
                    if not args.smoke:
                        raise RuntimeError(
                            f"{surface}: rung1 restricted fit under floor "
                            f"({RUNG1_REFIT_MIN_N}) at L={budget} — production draws "
                            "always clear this; investigate the draw"
                        )
                    return {"refit": False, "smoke_fallback": True}
                preds_r1, block_r1, info = res
                cand["blocks"]["rung1"] = block_r1
                cand["preds"] = list(cand["preds"])
                cand["preds"][eval_names.index("rung1")] = preds_r1
                np.savez(  # selected basis npz now carries the TRANSFER rung1
                    out_root / "nulls" / f"{arm_name}__{cell_tag}__{cand['basis']}.npz",
                    **{
                        f"{nm}_{met}": cand["blocks"][nm][met]
                        for nm in cand["blocks"]
                        for met in ("rho", "r2", "auroc")
                    },
                )
                return info

            for arm in arms:
                cell_path = out_root / "cells" / f"{arm}__{cell_tag}.json"
                companion_pending = (
                    arm == "arm_ctx"
                    and not (out_root / "cells" / f"arm_ctx_pca__{cell_tag}.json").exists()
                )
                if cell_path.exists() and not args.force and not companion_pending:
                    # arm_ctx resumes only when its arm_ctx_pca companion also
                    # landed (r1 g6: a crash between the two writes left the
                    # companion permanently unwritten on resume).
                    continue
                t0 = time.time()
                row: dict = {
                    "surface": surface,
                    "arm": arm,
                    "budget": str(budget),
                    "draw": draw_i,
                    "n_draw_rows": int(len(draw_rows)),
                    "n_null": n_null,
                    "dof_cap": DOF_CAP,
                    "n_train_vs_d": [int(len(draw_rows)), d_model],
                    "map_cell": map_cell,
                    "qa_disjoint": bool(args.qa_disjoint),
                    "pooling": POOLING_OF_ARM.get(arm, "t1"),
                    "permutation_axis": "boot_group(section-6 exchangeability)",
                    "split_identity": {
                        "seed_parts": seed_parts,
                        "n_train": n_train,
                        "n_dev": n_dev,
                        "n_test": int(len(rungs["rung0"])),
                    },
                    "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                }
                if rung1_meta is not None:
                    # r4 Major code-rung1-realization-undisclosed: every code
                    # cell names the PLANNED vs REALIZED rung-1 benchmark sets.
                    row["rung1_realization"] = rung1_meta
                if arm == "bl_const":
                    mu = float(table.dv[draw_rows].mean())
                    per_eval = {}
                    for name in eval_names:
                        ev = dev_idx if name == "dev" else rungs[name]
                        dv_e = table.dv[ev]
                        mu_use = mu
                        if name == "rung1" and needs_r1_refit:
                            # transfer read: mean of the RESTRICTED fit rows
                            fr = _rung1_fit_filter(table, draw_rows, r1_heldout)
                            if len(fr):
                                mu_use = float(table.dv[fr].mean())
                        per_eval[name] = {
                            "rho": 0.0,
                            "r2": float(
                                1.0
                                - ((dv_e - mu_use) ** 2).sum()
                                / max(1e-12, ((dv_e - dv_e.mean()) ** 2).sum())
                            ),
                            "auroc": 0.5,
                        }
                    row.update({"basis": "none", "per_eval": per_eval, "train_mean": mu})
                    cell_path.write_text(json.dumps(row))
                    continue
                if arm == "bl_agree":
                    ag = table.meta.get("agree_frac")
                    if ag is None:
                        raise RuntimeError("bl_agree requested but labeling carries no agree_frac")
                    per_eval = {}
                    for name in eval_names:
                        ev = dev_idx if name == "dev" else rungs[name]
                        fin = np.isfinite(ag[ev])
                        if int(fin.sum()) < 3:
                            per_eval[name] = {"rho": None, "r2": None, "auroc": None}
                            continue
                        blk = _metrics_block(ag[ev][fin][None, :, None], table.dv[ev][fin])
                        per_eval[name] = {
                            met: float(blk[met][0, 0]) for met in ("rho", "r2", "auroc")
                        }
                        per_eval[name]["n_used"] = int(fin.sum())
                    row.update(
                        {
                            "basis": "none",
                            "layer": None,
                            "per_eval": per_eval,
                            "selector": {"mode": "reference-row (no fit)", "dof_cap": None},
                        }
                    )
                    cell_path.write_text(json.dumps(row))
                    continue
                if arm in ("arm_dir_ctx", "arm_dir_map"):
                    x_all = x_ctx if arm == "arm_dir_ctx" else x_maplin
                    scores, n_spread = _direction_scores(
                        table,
                        draw_rows,
                        x_all,
                        n_null,
                        [SEED0, 5, _budget_seed(budget), draw_i],
                    )
                    row["n_spread_contexts"] = n_spread
                    if needs_r1_refit:
                        # direction points are never claim-bearing (plan section
                        # 5); the rung1 read here is an UNRESTRICTED-fit score,
                        # disclosed rather than refit.
                        row["rung1_fit"] = {"refit": False, "reason": "direction arm (disclosed)"}
                    _persist_score_cell(
                        row, scores, table, dev_idx, rungs, eval_names, out_root, arm, cell_tag
                    )
                    print(
                        f"[sweep] {arm} {cell_tag} n_spread={n_spread} "
                        f"elapsed={time.time() - t0:.0f}s",
                        flush=True,
                    )
                    continue
                # ridge arms: per basis, targets = [dv, nulls]
                best = None
                pca_best = None  # matched-rank compression comparator (arm_ctx)
                basis_items = list(bases.items()) if arm in arm_x else [("features", None)]
                for basis, v in basis_items:
                    x_full = arm_x.get(arm, single_slice.get(arm))
                    ev_rows_list = [dev_idx] + [rungs[nm] for nm in eval_names[1:]]
                    if v is not None:
                        # slice-first fp64 casts (never a full-n fp64 copy)
                        x_tr = np.einsum("lnd,ldk->lnk", x_full[:, draw_rows].astype(np.float64), v)
                        evals = [
                            np.einsum("lnd,ldk->lnk", x_full[:, ev].astype(np.float64), v)
                            for ev in ev_rows_list
                        ]
                    else:
                        x_tr = x_full[:, draw_rows]
                        evals = [x_full[:, ev] for ev in ev_rows_list]
                    # materialized copy: a broadcast view feeds torch.as_tensor
                    # a non-writable buffer (UB warning); ~0.7 GB at peak shape
                    y_tr = np.broadcast_to(
                        y_stack[None, :, :], (x_tr.shape[0],) + y_stack.shape
                    ).copy()
                    telem: list[dict] = []
                    preds = F.ridge_gcv_predict_per_target(
                        x_tr,
                        y_tr,
                        evals,
                        dof_cap=DOF_CAP,
                        device=args.device,
                        selector_telemetry=telem,
                    )
                    blocks = {}
                    for nm, ev_rows, pr in zip(eval_names, ev_rows_list, preds, strict=True):
                        blocks[nm] = _metrics_block(pr, table.dv[ev_rows])
                    dev_rho_obs = blocks["dev"]["rho"][:, 0]
                    ly_best = int(np.nanargmax(dev_rho_obs))
                    lam_sel, dof_sel = _selected_lambda_dof(telem, ly_best)
                    cand = {
                        "basis": basis,
                        "layer": ly_best,
                        "dev_rho": float(dev_rho_obs[ly_best]),
                        "blocks": blocks,
                        "preds": preds,
                        "selector": {
                            "mode": telem[0]["mode"] if telem else None,
                            "dof_cap": telem[0]["dof_cap"] if telem else None,
                            "n_train": telem[0]["n_train"] if telem else None,
                            "lambda_selected": lam_sel,
                            "dof_selected": dof_sel,
                        },
                    }
                    if best is None or cand["dev_rho"] > best["dev_rho"]:
                        best = cand
                    if basis.startswith("pca") and (
                        pca_best is None or cand["dev_rho"] > pca_best["dev_rho"]
                    ):
                        pca_best = cand
                    # persist per-basis null matrices (per-draw x layer)
                    np.savez(
                        out_root / "nulls" / f"{arm}__{cell_tag}__{basis}.npz",
                        **{
                            f"{nm}_{met}": blocks[nm][met]
                            for nm in blocks
                            for met in ("rho", "r2", "auroc")
                        },
                    )
                if arm in single_slice:
                    # TF-IDF fit once on the full train partition, not refit per
                    # draw — the disclosed simplification, now an actual field.
                    row["feature_basis_note"] = (
                        "single-slice features; TF-IDF fit once on full train, not per draw"
                    )
                row["rung1_fit"] = _rung1_swap(best, arm, x_full)
                _finalize_ridge_cell(
                    row,
                    best,
                    t0,
                    cell_path,
                    out_root,
                    table,
                    dev_idx,
                    rungs,
                    eval_names,
                    arm,
                    cell_tag,
                )
                # PCA matched-rank comparator companion for the Context probe
                # (H2's decisive mapped-vs-PCA-context pair needs its own
                # preds/cells row — the dev-selected basis may be ambient)
                if arm == "arm_ctx" and pca_best is not None:
                    row2 = {
                        k: v
                        for k, v in row.items()
                        if k not in ("basis", "layer", "dev_rho", "selector", "per_eval", "wall_s")
                    }
                    row2["arm"] = "arm_ctx_pca"
                    row2["pooling"] = POOLING_OF_ARM["arm_ctx_pca"]
                    row2["rung1_fit"] = _rung1_swap(pca_best, "arm_ctx_pca", x_full)
                    _finalize_ridge_cell(
                        row2,
                        pca_best,
                        t0,
                        out_root / "cells" / f"arm_ctx_pca__{cell_tag}.json",
                        out_root,
                        table,
                        dev_idx,
                        rungs,
                        eval_names,
                        "arm_ctx_pca",
                        cell_tag,
                    )


def _selected_lambda_dof(telem_rows: list[dict], layer: int, target: int = 0):
    """Selected (lambda, effective dof) for one layer's observed target from
    the core's per-layer-chunk telemetry rows."""
    for tr in telem_rows:
        lam = np.asarray(tr["lambda_selected"])
        lo = int(tr["slice_offset"])
        if lo <= layer < lo + lam.shape[0]:
            dof = np.asarray(tr["dof_selected"])
            return float(lam[layer - lo, target]), float(dof[layer - lo, target])
    return None, None


def _finalize_ridge_cell(
    row, best, t0, cell_path, out_root, table, dev_idx, rungs, eval_names, arm, cell_tag
):
    """Persist one ridge cell row + its dev-selected preds JSONL."""
    row.update(
        {
            "basis": best["basis"],
            "layer": best["layer"],
            "dev_rho": best["dev_rho"],
            "selector": best["selector"],
            "per_eval": {
                nm: {
                    met: float(best["blocks"][nm][met][best["layer"], 0])
                    for met in ("rho", "r2", "auroc")
                }
                for nm in best["blocks"]
            },
            "wall_s": round(time.time() - t0, 1),
        }
    )
    Path(cell_path).write_text(json.dumps(row))
    _write_preds(
        out_root / "preds" / f"preds_{arm}_{cell_tag}.jsonl",
        table,
        dev_idx,
        rungs,
        eval_names,
        best,
    )
    print(
        f"[sweep] {arm} {cell_tag} basis={best['basis']} "
        f"dev_rho={best['dev_rho']:.4f} wall={row['wall_s']}s",
        flush=True,
    )


def _budget_seed(budget) -> int:
    return 0 if budget == "full" else int(budget)


def _persist_score_cell(row, scores, table, dev_idx, rungs, eval_names, out_root, arm, cell_tag):
    """Direction-arm persistence: metrics per eval set + nulls npz + preds.

    Direction arms are ambient-basis, sign-selected on dev (|rho| argmax —
    a one-parameter readout has an arbitrary global sign)."""
    basis = "ambient"
    blocks = {}
    for nm in eval_names:
        ev = dev_idx if nm == "dev" else rungs[nm]
        blocks[nm] = _metrics_block(scores[:, ev, :], table.dv[ev])
    dev_rho_obs = blocks["dev"]["rho"][:, 0]
    ly = int(np.nanargmax(np.abs(dev_rho_obs)))
    sign = 1.0 if dev_rho_obs[ly] >= 0 else -1.0
    row.update(
        {
            "basis": basis,
            "layer": ly,
            "sign": sign,
            "dev_rho": float(sign * dev_rho_obs[ly]),
            "selector": {"mode": "direction-1param", "dof_cap": None},
            "per_eval": {
                nm: {
                    met: float((sign if met == "rho" else 1.0) * blocks[nm][met][ly, 0])
                    for met in ("rho", "r2", "auroc")
                }
                for nm in blocks
            },
        }
    )
    np.savez(
        out_root / "nulls" / f"{arm}__{cell_tag}__{basis}.npz",
        **{f"{nm}_{met}": blocks[nm][met] for nm in blocks for met in ("rho", "r2", "auroc")},
    )
    (out_root / "cells" / f"{arm}__{cell_tag}.json").write_text(json.dumps(row))
    best = {"layer": ly, "basis": basis, "preds": None, "scores": scores, "sign": sign}
    _write_preds(
        out_root / "preds" / f"preds_{arm}_{cell_tag}.jsonl",
        table,
        dev_idx,
        rungs,
        eval_names,
        best,
    )


def _write_preds(path, table, dev_idx, rungs, eval_names, best):
    """Per-context preds JSONL at the dev-selected configuration (observed
    target only). Ridge cells carry ``best['preds']`` (list per eval set);
    direction cells carry ``best['scores']`` (Ly, n_ctx, targets) + sign."""
    with Path(path).open("w", encoding="utf-8") as fh:
        for nm in eval_names:
            ev = dev_idx if nm == "dev" else rungs[nm]
            if best.get("preds") is not None:
                pr = best["preds"][eval_names.index(nm)][best["layer"], :, 0]
            else:
                pr = best["sign"] * best["scores"][best["layer"], ev, 0]
            ids = [table.ctx_ids[i] for i in ev]
            dvv = table.dv[ev]
            for cid, yt, yp in zip(ids, dvv, pr, strict=True):
                fh.write(
                    json.dumps(
                        {"eval": nm, "context_id": cid, "y_true": float(yt), "y_pred": float(yp)}
                    )
                    + "\n"
                )


# ---------------------------------------------------------------------------
# select + bootstrap phases (CPU, from persisted artifacts)
# ---------------------------------------------------------------------------


def phase_select(args) -> None:
    """Freeze dev-selected configs per (surface, arm, budget, draw) BEFORE the
    aggregated test read (selection lives in the persisted cell rows)."""
    surface = args.surface
    out_root = Path(args.fits_root) / surface
    cell_paths = sorted((out_root / "cells").glob("*.json"))
    arm_rows = [json.loads(c.read_text()) for c in cell_paths]
    sel: dict[str, dict] = {
        cell.stem: {
            k: row.get(k)
            for k in (
                "arm",
                "budget",
                "draw",
                "basis",
                "layer",
                "dev_rho",
                "selector",
                "map_cell",
                "qa_disjoint",
            )
        }
        for cell, row in zip(cell_paths, arm_rows, strict=True)
    }
    if not sel:
        raise RuntimeError(f"no cells under {out_root / 'cells'} — run --phase sweep first")
    # r4 Major code-rung1-realization-undisclosed: surface the sweep-persisted
    # PLANNED-vs-REALIZED rung-1 disclosure top-level in BOTH aggregates (code
    # rows carry it; other surfaces aggregate to None). Inconsistent rows would
    # mean cells from two different gate realizations share one root — refuse.
    r1_metas = [r["rung1_realization"] for r in arm_rows if r.get("rung1_realization")]
    rung1_realization = r1_metas[0] if r1_metas else None
    for m in r1_metas[1:]:
        if m != rung1_realization:
            raise RuntimeError(
                f"inconsistent rung1_realization across cell rows under {out_root / 'cells'} — "
                "cells span two gate realizations; use a fresh --fits-root"
            )
    path = out_root / "selection.json"
    path.write_text(
        json.dumps(
            {
                "surface": surface,
                "cells": sel,
                "rung1_realization": rung1_realization,
                "frozen_ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "metadata": _provenance("select"),
            },
            indent=1,
        )
    )
    print(f"[select] froze {len(sel)} cells -> {path}", flush=True)
    # Plan-contract aggregate (section 6.5: fits/<surface>/all_arms.json) —
    # the FULL cell rows in one file (selection.json above stays the thin
    # frozen-selection view).
    agg_path = out_root / "all_arms.json"
    agg_path.write_text(
        json.dumps(
            {
                "surface": surface,
                "n_rows": len(arm_rows),
                "arm_rows": arm_rows,
                "rung1_realization": rung1_realization,
                "metadata": _provenance("select-all-arms"),
            },
            default=float,
        )
    )
    print(f"[select] aggregate: {len(arm_rows)} rows -> {agg_path}", flush=True)


def _boot_spearman_draws(
    preds_mat: np.ndarray, y_true: np.ndarray, idx_draws: list[np.ndarray]
) -> np.ndarray:
    """Batched per-draw Spearman over group-resample index draws.

    Rank statistics batch along the DRAW axis (vectorize-many-cell-fits.md —
    the r2 fit-loop-not-batched bootstrap leg): draws are grouped by realized
    resample LENGTH (group resampling with replacement concatenates
    variable-size groups), and each same-length block is ranked + correlated
    as ONE chunked batched call — the same ``rank_rows``/``_pearson_rows``
    arithmetic per row as the serial ``spearman_rows`` loop (bitwise
    serial-parity pinned). MEASURED at production shape (n=2,500, 13 arms,
    2,000 draws, this VM): the rank statistic itself is argsort-FLOP-bound at
    m=2,500, so draw-batching the STAT is ~par with serial (chunk 128 best;
    the dominant serial overhead was the per-draw tiny-array concatenation in
    the CALLER's index construction, vectorized there). Chunk sized so the
    (chunk, n_arms, m) fp64 rank tensor stays ~<=32 MB (the measured cache
    sweet spot; larger chunks measured SLOWER).
    """
    n_arms = preds_mat.shape[0]
    boot = np.empty((len(idx_draws), n_arms))
    by_len: dict[int, list[int]] = {}
    for b, idx in enumerate(idx_draws):
        by_len.setdefault(len(idx), []).append(b)
    for m, bs in sorted(by_len.items()):
        chunk = max(1, int(32e6 // max(1, n_arms * m * 8)))
        for lo in range(0, len(bs), chunk):
            sub = bs[lo : lo + chunk]
            idx_mat = np.stack([idx_draws[b] for b in sub])  # (B, m)
            g = np.moveaxis(preds_mat[:, idx_mat], 0, 1)  # (B, A, m)
            yv = y_true[idx_mat]  # (B, m)
            boot[sub] = _pearson_rows(rank_rows(g), rank_rows(yv)[:, None, :])
    return boot


def phase_bootstrap(args) -> None:
    """Paired group bootstrap from stored preds: identical group resample
    shared across compared arms per (budget, draw, eval)."""
    surface = args.surface
    out_root = Path(args.fits_root) / surface
    lab = _load_labeling(Path(args.dv_root) / surface / "labeling.json", surface=surface)
    boot_group = {r["context_id"]: _boot_group_of(surface, r) for r in lab["rows"]}
    preds_files = sorted((out_root / "preds").glob("preds_*.jsonl"))
    if not preds_files:
        raise RuntimeError(f"no preds under {out_root / 'preds'}")
    # UPSTREAM content digests (reconciler r3 deferral long-loop-
    # restartability-upstream-digests, bootstrap half): the resume key below
    # additionally pins the CONSUMED labeling + per-arm preds file bytes, so a
    # regenerated DV or re-swept preds set recomputes instead of silently
    # reusing stale units. File bytes read from disk — hash-safe (never a
    # recomputed float array) per the float-key rule.
    lab_sha = _file_sha(Path(args.dv_root) / surface / "labeling.json")
    unit_file_shas: dict[tuple[str, str], dict[str, str | None]] = {}
    cells: dict[tuple[str, str], dict[str, dict[str, tuple[list, list]]]] = {}
    for pf in preds_files:
        m = re.match(r"preds_(.+)_L(.+)_draw(\d+)\.jsonl$", pf.name)
        if not m:
            continue
        arm, budget, draw = m.group(1), m.group(2), m.group(3)
        unit_file_shas.setdefault((budget, draw), {})[arm] = _file_sha(pf)
        per_eval: dict[str, tuple[list, list, list]] = {}
        with pf.open(encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                r = json.loads(line)
                per_eval.setdefault(r["eval"], ([], [], []))
                per_eval[r["eval"]][0].append(r["context_id"])
                per_eval[r["eval"]][1].append(r["y_true"])
                per_eval[r["eval"]][2].append(r["y_pred"])
        cells.setdefault((budget, draw), {})[arm] = per_eval
    # Per-unit checkpoint (code-style intra-phase grain: > 50 (budget, draw,
    # eval) units) + resume keyed on the output-affecting regime: n_boot AND
    # the unit's ARM SET (r2 Minor 4 / long-loop-restartability — a re-run
    # after the arm roster changed must recompute, never silently keep old
    # units missing the new arm; the superseding row is appended, and the
    # last matching line per key wins at load).
    cells_path = out_root / "bootstrap_cells.jsonl"
    persisted_by_key: dict[tuple, dict] = {}
    if cells_path.exists() and not args.force:
        with cells_path.open(encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    r = json.loads(line)
                    if int(r.get("n_boot", -1)) == int(args.n_boot):
                        persisted_by_key[(r["budget"], int(r["draw"]), r["eval"])] = r
    elif cells_path.exists() and args.force:
        cells_path.unlink()
    out: list[dict] = []
    consumed: set[tuple] = set()
    t0 = time.time()
    n_units = 0
    total_units = sum(len(arms[next(iter(arms))]) for arms in cells.values())
    for (budget, draw), arms in sorted(cells.items()):
        # A PER-UNIT rng (seeded on the unit key) makes the resample identical
        # whether the unit runs fresh or after a resume (a shared sequential
        # rng would give resume-order-dependent draws).
        ref_arm = next(iter(arms))
        arm_names = sorted(arms)
        inputs_sha = hashlib.sha256(
            json.dumps(
                {"labeling": lab_sha, "preds": unit_file_shas[(budget, draw)]},
                sort_keys=True,
            ).encode()
        ).hexdigest()[:12]
        for ev in arms[ref_arm]:
            key = (budget, int(draw), ev)
            prior = persisted_by_key.get(key)
            if prior is not None:
                if prior.get("arms") == arm_names and prior.get("inputs_sha") == inputs_sha:
                    out.append(prior)
                    consumed.add(key)
                    continue
                print(
                    f"[bootstrap] RECOMPUTE {key}: persisted arms {prior.get('arms')} != "
                    f"current {arm_names} or inputs_sha {prior.get('inputs_sha')} != "
                    f"{inputs_sha} (arm-roster / upstream-content change invalidates "
                    "the unit)",
                    flush=True,
                )
                consumed.add(key)
            rng = np.random.default_rng(
                [SEED0, 17, _budget_seed(budget), int(draw), _stable_seed(ev)]
            )
            ids = arms[ref_arm][ev][0]
            missing_groups = [c for c in ids if c not in boot_group]
            if missing_groups:
                # r1 g6: the .get(c, c) fallback silently regrouped at context
                # grain, masking a preds/labeling mismatch — fail loud instead.
                raise RuntimeError(
                    f"bootstrap: {len(missing_groups)} pred context ids missing from the "
                    f"labeling's boot_group map (e.g. {missing_groups[:3]})"
                )
            groups = np.array([boot_group[c] for c in ids])
            uniq = np.unique(groups)
            members = {g: np.flatnonzero(groups == g) for g in uniq}
            y_true = np.array(arms[ref_arm][ev][1])
            preds_mat = np.stack(
                [
                    np.array(arms[a][ev][2]) if arms[a][ev][0] == ids else _align(arms[a][ev], ids)
                    for a in arm_names
                ],
                axis=0,
            )
            # rng stream: IDENTICAL choice calls in the same order as the old
            # serial loop. The per-draw index build is a VECTORIZED multi-range
            # gather — the old `np.concatenate([members[g] for g in gs])` was
            # the measured hot spot (singleton groups: 2,500 tiny concats per
            # draw ~ 57% of unit wall); the statistic batches along the draw
            # axis in _boot_spearman_draws (bitwise-parity pinned).
            member_list = [members[g] for g in uniq]
            flat_members = np.concatenate(member_list)
            sizes = np.array([len(mem) for mem in member_list])
            grp_starts = np.concatenate(([0], np.cumsum(sizes)[:-1]))
            idx_draws = []
            for _b in range(args.n_boot):
                gs = rng.choice(uniq, size=len(uniq), replace=True)
                pos = np.searchsorted(uniq, gs)  # uniq is sorted (np.unique)
                ln = sizes[pos]
                tot = int(ln.sum())
                run_starts = np.repeat(grp_starts[pos], ln)
                within = np.arange(tot) - np.repeat(np.cumsum(ln) - ln, ln)
                idx_draws.append(flat_members[run_starts + within])
            boot = _boot_spearman_draws(preds_mat, y_true, idx_draws)
            unit = {
                "budget": budget,
                "draw": int(draw),
                "eval": ev,
                "n_groups": int(len(uniq)),
                "n_boot": args.n_boot,
                # frozen-config: resamples the dev-selected preds with NO
                # per-resample re-selection (plan section 6 dual-labeling;
                # the H3 gap CI is the selection-inherited counterpart).
                "ci_kind": "frozen-config",
                "arms": arm_names,
                "inputs_sha": inputs_sha,
                "rho_point": spearman_rows(preds_mat, y_true).tolist(),
                "rho_ci": [
                    [
                        float(np.nanpercentile(boot[:, i], 2.5)),
                        float(np.nanpercentile(boot[:, i], 97.5)),
                    ]
                    for i in range(len(arm_names))
                ],
                "pairwise_delta_ci": _pairwise_delta_ci(boot, arm_names),
            }
            with cells_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(unit, default=float) + "\n")
            out.append(unit)
            n_units += 1
            print(
                f"[bootstrap] unit {n_units}/{total_units} L{budget} draw{draw} {ev}: "
                f"{len(arm_names)} arms n_groups={len(uniq)} "
                f"elapsed={time.time() - t0:.0f}s",
                flush=True,
            )
    # Persisted units whose (budget, draw, eval) no longer exists in the
    # current preds set are retained in the summary unchanged (prior
    # behaviour); superseded stale-arm rows are NOT (consumed above).
    for key, r in persisted_by_key.items():
        if key not in consumed:
            out.append(r)
    path = out_root / "bootstrap_summary.json"
    path.write_text(
        json.dumps(
            {
                "surface": surface,
                "ci_kind": "frozen-config",
                "cells": out,
                "metadata": _provenance("bootstrap"),
            },
            default=float,
        )
    )
    print(f"[bootstrap] wrote {len(out)} cell summaries -> {path}", flush=True)


def _align(entry, ids):
    pos = {c: i for i, c in enumerate(entry[0])}
    return np.array([entry[2][pos[c]] for c in ids])


def _pairwise_delta_ci(boot: np.ndarray, arm_names: list[str]) -> dict:
    out = {}
    for i, a in enumerate(arm_names):
        for j, b in enumerate(arm_names):
            if j <= i:
                continue
            d = boot[:, i] - boot[:, j]
            out[f"{a}-{b}"] = [float(np.nanpercentile(d, 2.5)), float(np.nanpercentile(d, 97.5))]
    return out


# ---------------------------------------------------------------------------
# H3 orchestration + gap module
# ---------------------------------------------------------------------------


def _load_arm_rows(all_arms_path: Path) -> list[dict]:
    """Rows of an all_arms_spearman.json. Schema-from-artifact (banked file
    probed 2026-08-20): rows under top-level ``arm_rows``; rho under
    ``rho_frozen``. A bare list (test fixtures) passes through."""
    payload = json.loads(Path(all_arms_path).read_text())
    if isinstance(payload, dict):
        if "arm_rows" in payload:
            return payload["arm_rows"]
        if "rows" in payload:
            return payload["rows"]
        raise RuntimeError(
            f"{all_arms_path}: no arm_rows/rows key (top-level keys: {sorted(payload)[:8]})"
        )
    return payload


def _row_rho(r: dict) -> float | None:
    for k in ("rho_frozen", "rho", "spearman"):
        v = r.get(k)
        if v is not None:
            return float(v)
    return None


def _cell_filter_rows(rows: list[dict], budget: int, u_rung: str | None) -> list[dict]:
    """The plan-section-3 registered-cell row filter: (config, f_u, u_rung,
    variant, eval_rung) HELD at the registered comparison cell."""
    out = []
    for r in rows:
        if int(r.get("budget_l", r.get("budget") or -1)) != int(budget):
            continue
        if r.get("f_u") is not None:  # composition sub-grid rows excluded
            continue
        if any(r.get(k) != v for k, v in H3_CELL_FILTER.items()):
            continue
        if u_rung is not None and str(r.get("u_rung_label")) != str(u_rung):
            continue
        out.append(r)
    return out


def h3_gap_from_all_arms(
    all_arms_path: Path,
    *,
    budget: int,
    u_rung: str | None = H3_U_RUNG_DEFAULT,
    n_boot: int = H3_N_BOOT,
    boot_seed: int = SEED0,
) -> dict:
    """Pinned MEAN gap (plan section 3): mapped = better of
    {arm6_map_proj_e1, arm7_map_ridge_pred}; direct = arm4_ridge_ctx alone
    (arm5_mlp_ctx is NOT direct); row filter holds (config, f_u, u_rung,
    variant, eval_rung) at the registered comparison cell; aggregation over
    draws/seeds at the frozen layer is the MEAN, never a max; the disclosed
    2-element mapped selection is applied identically INSIDE every bootstrap
    draw (selection-inherited CI; paired over (seed, draw) cells)."""
    rows = _load_arm_rows(all_arms_path)
    cell_rows = _cell_filter_rows(rows, budget, u_rung)
    need = (*H3_MAPPED_ARMS, H3_DIRECT_ARM)
    per_arm: dict[str, dict[tuple, float]] = {a: {} for a in need}
    for r in cell_rows:
        arm = r.get("arm")
        rho = _row_rho(r)
        if arm in per_arm and rho is not None and np.isfinite(rho):
            per_arm[arm][(r.get("seed"), r.get("draw"))] = rho
    missing = sorted(a for a in need if not per_arm[a])
    if missing:
        raise RuntimeError(
            f"h3-gap: arms missing at budget {budget} / u_rung {u_rung} in "
            f"{all_arms_path}: {missing} "
            f"(cell rows: {len(cell_rows)}; arms present: "
            f"{sorted({r.get('arm') for r in cell_rows})[:12]})"
        )
    # paired (seed, draw) cells present for ALL three arms
    keys = sorted(set.intersection(*(set(per_arm[a]) for a in need)))
    if not keys:
        raise RuntimeError(f"h3-gap: no (seed, draw) cell carries all of {need}")
    mat = np.array([[per_arm[a][k] for k in keys] for a in need])  # (3, n_cells)
    means = {a: float(mat[i].mean()) for i, a in enumerate(need)}
    gaps = {a: means[a] - means[H3_DIRECT_ARM] for a in H3_MAPPED_ARMS}
    headline_arm = max(H3_MAPPED_ARMS, key=lambda a: gaps[a])
    # selection-inherited paired bootstrap over (seed, draw) cells: the SAME
    # resampled cell multiset feeds all three arms; better-of-two INSIDE draw
    rng = np.random.default_rng([boot_seed, 33, budget])
    idx = rng.integers(0, len(keys), size=(n_boot, len(keys)))
    boot_means = mat[:, idx].mean(axis=2)  # (3, n_boot)
    boot_gap = np.max(boot_means[: len(H3_MAPPED_ARMS)], axis=0) - boot_means[-1]
    ci = [float(np.percentile(boot_gap, 2.5)), float(np.percentile(boot_gap, 97.5))]
    return {
        "budget": budget,
        "cell_filter": {**H3_CELL_FILTER, "f_u": None, "u_rung_label": u_rung},
        "direct_arm": H3_DIRECT_ARM,
        "direct_mean_rho": means[H3_DIRECT_ARM],
        "mapped_means": {a: means[a] for a in H3_MAPPED_ARMS},
        "gaps": gaps,
        "headline_mapped_arm": headline_arm,
        "headline_gap": gaps[headline_arm],
        "headline_gap_ci95": ci,
        "ci_wholly_positive": bool(ci[0] > 0.0),
        "n_boot": int(n_boot),
        "aggregation": "mean-over-(seed,draw)-cells-at-registered-cell (pinned; never max)",
        "n_paired_cells": len(keys),
        "n_rows_per_arm": {a: len(per_arm[a]) for a in need},
    }


def h3_reference_verify(banked_root: Path) -> dict:
    """Verify the plan's pinned reference numbers against the banked parent
    artifacts. The four REPRODUCED anchors are hard-asserted under the
    pooled-u_rung MAX-over-unit-groups read (the reduction that reproduces
    them exactly, established at Step 0-p); the two non-reproducing @2,500
    anchors are REPORTED (persisted concern), never asserted."""
    report: dict = {"anchors": {}}
    for (behavior, budget), ref in sorted(H3_REFERENCE.items()):
        path = Path(banked_root) / behavior / "arm_results" / "all_arms_spearman.json"
        if not path.exists():
            raise RuntimeError(f"banked all_arms_spearman.json missing: {path}")
        rows = _load_arm_rows(path)
        per_arm: dict[str, list[float]] = {}
        for r in rows:
            if int(r.get("budget_l", r.get("budget") or -1)) != budget:
                continue
            if any(r.get(k) != v for k, v in H3_CELL_FILTER.items()):
                continue  # variant/config/eval_rung held; u_rungs POOLED
            rho = _row_rho(r)
            if r.get("arm") and rho is not None and np.isfinite(rho):
                per_arm.setdefault(r["arm"], []).append(rho)
        # pooled-u_rung MAX-over-unit-groups read (the reduction verified to
        # reproduce the four starred anchors EXACTLY on the banked artifacts)
        maxes = {a: float(np.max(v)) for a, v in per_arm.items() if v}
        try:
            mapped = max(maxes[a] for a in H3_MAPPED_ARMS)
            gap = mapped - maxes[H3_DIRECT_ARM]
        except KeyError as exc:
            raise RuntimeError(f"banked rows missing arm {exc} at {path} b={budget}") from exc
        entry = {
            "recomputed_gap_maxread": round(gap, 4),
            "pinned": ref["gap"],
            "reproduced_at_step0p": ref["reproduced"],
        }
        if ref["reproduced"]:
            if abs(gap - ref["gap"]) > 5e-4:
                raise RuntimeError(
                    f"h3 reference verifier: {behavior}@{budget} recomputed {gap:.4f} != "
                    f"pinned {ref['gap']:.4f} (reproduced anchor drifted)"
                )
            entry["verdict"] = "ASSERTED-MATCH"
        else:
            entry["verdict"] = "REPORT-ONLY (plan reference not reproduced under any tested "
            entry["verdict"] += "reduction; see raised concern)"
        report["anchors"][f"{behavior}@{budget}"] = entry
    return report


def _h3_root(args) -> Path:
    """h3_recompute sibling of the fits root — smoke gets its OWN sibling.

    The fits-root smoke rebind changes only the LEAF (``fits_smoke``), so a
    parent-derived h3 path would still land canonical under --smoke (r2
    fits-smoke-local-outroot); suffix the sibling name explicitly instead.
    """
    name = "h3_recompute" + ("_smoke" if getattr(args, "smoke", False) else "")
    return Path(args.fits_root).parent / name


def phase_h3(args) -> None:
    """G4 asserts + stage ordering. Composes the ported issue1739_fits.py
    invocations; records the stage-1 verdict BEFORE any correctness-side read."""

    h3_root = _h3_root(args)
    behaviors = args.behaviors or list(H3_BEHAVIORS)
    # G4(a): registered budget exactly 2,500 on both sides — this driver only
    # ever composes --budgets 2500 for the capped legs (asserted here).
    assert H3_BUDGET == 2500
    if args.h3_step == "stage1":
        for behavior in behaviors:
            out_root = h3_root / behavior
            banked_dv = str(Path(args.banked_root) / "dv_dataset" / behavior / "labeling.json")
            # G4(c): FRESH out-root at launch
            if out_root.exists() and any(out_root.iterdir()) and not args.resume:
                raise RuntimeError(f"G4(c): out-root {out_root} not fresh at launch")
            out_root.mkdir(parents=True, exist_ok=True)
            # G4(e): sycophancy pilot-wall artifact BEFORE its full battery.
            # One unit group = (map_key, budget, draw, seed): pin ONE draw,
            # ONE seed, ONE variant, ONE u rung (the ported CLI's grain —
            # it has no --unit-groups flag).
            if behavior == "sycophancy":
                pilot = out_root / "pilot_wall.json"
                if not pilot.exists():
                    cmd = _ported_cmd(
                        args,
                        behavior,
                        out_root,
                        budgets="2500",
                        dof_cap="0.9",
                        dv_json=banked_dv,
                        extra=[
                            "--draws",
                            "0",
                            "--seeds",
                            "0",
                            "--variant",
                            "context_end",
                            "--u-sizes",
                            "full",
                        ],
                    )
                    t0 = time.time()
                    _run(cmd)
                    pilot.write_text(
                        json.dumps(
                            {
                                "behavior": behavior,
                                "pilot_scope": "1 draw x 1 seed x context_end x u=full",
                                "measured_wall_s": round(time.time() - t0, 1),
                                "fence_s": round(2 * (time.time() - t0), 1),
                                "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                            },
                            indent=2,
                        )
                    )
                    print(
                        f"[h3] sycophancy pilot wall {time.time() - t0:.0f}s -> {pilot}", flush=True
                    )
                assert pilot.exists(), "G4(e): pilot_wall.json must exist pre-full-battery"
            cmd = _ported_cmd(
                args, behavior, out_root, budgets="2500", dof_cap="0.9", dv_json=banked_dv
            )
            _run(cmd)
        return
    if args.h3_step == "verdict":
        # Stage-1 verdict: recorded BEFORE any correctness-side 2,500 read (G4 d)
        verdict_path = h3_root / "stage1_verdict.json"
        gaps = {}
        for behavior in behaviors:
            allp = h3_root / behavior / "arm_results" / "all_arms_spearman.json"
            gaps[behavior] = h3_gap_from_all_arms(allp, budget=H3_BUDGET, u_rung=args.h3_u_rung)
        # Kill branch enabled ONLY when EVERY behavior's capped 2,500 headline
        # gap interval is WHOLLY positive (universal quantifier — plan
        # section 3 stage 2; selection-inherited CI from h3_gap_from_all_arms).
        all_positive_points = all(g["headline_gap"] > 0 for g in gaps.values())
        all_ci_positive = all(g["ci_wholly_positive"] for g in gaps.values())
        verdict = {
            "behaviors": gaps,
            "all_headline_gaps_positive_point": bool(all_positive_points),
            "kill_branch_enabled": bool(all_ci_positive),
            "recorded_before_correctness_read": True,
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        verdict_path.write_text(json.dumps(verdict, indent=2, default=float))
        print(
            f"[h3] stage-1 verdict recorded (kill_branch_enabled={all_ci_positive}) "
            f"-> {verdict_path}",
            flush=True,
        )
        return
    if args.h3_step == "stage2":
        # G4(d): correctness-side 2,500 read only AFTER the recorded verdict
        verdict_path = h3_root / "stage1_verdict.json"
        if not verdict_path.exists():
            raise RuntimeError(
                "G4(d): stage-1 verdict not recorded — refuse the correctness-side 2,500 read"
            )
        out_root = Path(args.fits_root) / "qa"
        out_root.mkdir(parents=True, exist_ok=True)
        dv_json = str(Path(args.dv_root) / "qa" / "labeling.json")
        # capped 2,500 + legacy companions (dof_cap=None at n > d anchors);
        # --behavior hallucination: the QA surface rides the parent's
        # hallucination stores/arms (provenance slug; the DV is the swapped
        # correctness labeling and every row carries h3_label).
        #
        # PER-LEG OUT-ROOTS (r1 g6 Critical 1): the ported arms.write_summary
        # REWRITES arm_results/all_arms_spearman.json atomically with ONLY that
        # run's records, so three sequential runs into ONE out-root ship a
        # summary holding only the LAST leg's rows — the capped-2,500 anchor
        # (the stage-2 kill read) silently vanishes. Each leg gets its own
        # out-root; the aggregate merges the three summaries.
        legs = (
            ("capped2500", "2500", "0.9"),
            ("legacy8000", "8000", None),
            ("legacy16000", "16000", None),
        )
        leg_rows: dict[str, list[dict]] = {}
        for leg_name, budgets, cap in legs:
            leg_root = out_root / f"h3_stage2_{leg_name}"
            leg_root.mkdir(parents=True, exist_ok=True)
            _run(
                _ported_cmd(
                    args,
                    "hallucination",
                    leg_root,
                    budgets=budgets,
                    dof_cap=cap,
                    dv_json=dv_json,
                    label="h3_parent_exact",
                )
            )
            rows = _load_arm_rows(leg_root / "arm_results" / "all_arms_spearman.json")
            labelled = [
                dict(r, stage2_leg=leg_name) for r in rows if r.get("h3_label") == "h3_parent_exact"
            ]
            if not labelled:
                raise RuntimeError(f"stage2 leg {leg_name} produced no labelled rows")
            leg_rows[leg_name] = labelled
        merged = [r for leg_name, _b, _c in legs for r in leg_rows[leg_name]]
        (out_root / "h3_parent_exact.json").write_text(
            json.dumps(
                {
                    "rows": merged,
                    "legs": {name: len(leg_rows[name]) for name, _b, _c in legs},
                    "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                },
                default=float,
            )
        )
        print(
            f"[h3] stage2 rows: {len(merged)} "
            f"({ {name: len(leg_rows[name]) for name, _b, _c in legs} }) "
            f"-> {out_root / 'h3_parent_exact.json'}"
        )
        return
    raise SystemExit(f"unknown --h3-step {args.h3_step}")


def _h3_store_paths(args, behavior: str) -> dict[str, str]:
    """Per-behavior store composition for the ported CLI's REAL mode.

    The ported ``issue1739_fits.py`` real mode hard-requires
    ``--labeled-store/--dv-json/--u-store/--e1-store`` (``_run_real``, r1 Codex
    blocker: the composed argv omitted them and would SystemExit pod-side).
    Layout (plan section 9 Pod B): hallucination's labeled store co-resides
    with the QA stage at ``--qa-store-dir`` (same store, no double stage);
    sycophancy/evil stage under ``--h3-store-root`` as ``<behavior>_labeling``;
    every behavior's e1 extraction store stages as ``<behavior>_extraction``.
    """
    root = Path(args.h3_store_root)
    labeled = (
        Path(args.qa_store_dir) if behavior == "hallucination" else root / f"{behavior}_labeling"
    )
    return {"labeled": str(labeled), "e1": str(root / f"{behavior}_extraction")}


def _ported_cmd(
    args, behavior, out_root, *, budgets, dof_cap, dv_json=None, label=None, extra=None
):
    stores = _h3_store_paths(args, behavior)
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "issue1739_fits.py"),
        "--budgets",
        budgets,
        "--out-root",
        str(out_root),
        "--labeled-store",
        stores["labeled"],
        "--u-store",
        str(args.u_store_dir),
        "--e1-store",
        stores["e1"],
        "--device",
        args.device,
    ]
    if dof_cap is not None:
        cmd += ["--dof-cap", str(dof_cap)]
    if behavior:
        cmd += ["--behavior", behavior]
    if dv_json:
        cmd += ["--dv-json", dv_json]
    if label:
        cmd += ["--h3-label", label]
    if extra:
        cmd += list(extra)
    if args.ported_extra:
        cmd += args.ported_extra
    return cmd


def _run(cmd: list[str]) -> None:
    import subprocess

    print(f"[h3] exec: {' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd, env={**os.environ}, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"h3 subprocess rc={proc.returncode}: {' '.join(cmd[:6])}...")


def phase_h3_gap(args) -> None:
    out = {
        "reference_verify": h3_reference_verify(Path(args.banked_root)),
        "metadata": _provenance("h3-gap"),
    }
    if args.h3_out_root:
        gaps = {}
        for behavior in args.behaviors or list(H3_BEHAVIORS):
            allp = Path(args.h3_out_root) / behavior / "arm_results" / "all_arms_spearman.json"
            if allp.exists():
                gaps[behavior] = h3_gap_from_all_arms(allp, budget=H3_BUDGET, u_rung=args.h3_u_rung)
        out["recompute_gaps"] = gaps
    path = _h3_root(args) / "gap_report.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2, default=float))
    print(f"[h3-gap] -> {path}", flush=True)


def _provenance(phase: str) -> dict:
    """Reproducibility metadata block (git commit + dirty flag + ts)."""
    from explore_persona_space.orchestrate.provenance import as_metadata_dict, git_provenance

    return as_metadata_dict(git_provenance(), phase=phase)


# ---------------------------------------------------------------------------
# feasibility phase (P1 pre-step)
# ---------------------------------------------------------------------------


def _pre_gen_counts(args) -> dict[str, dict[str, int]]:
    """PRE-GENERATION feasibility counts from loader metadata + the dedup
    report (r1 Codex: the "P1 pre-step" consumed post-generation DV files, so
    it could never run BEFORE P1 as the plan sequences it). Conservative on
    code: uses the WITHOUT-BCB pool (the smaller train) unless the gate has
    already resolved bcb_fit_allowed=True."""
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    import issue2388_gen as G

    gen_root = Path(args.gen_root)
    dedup_p = gen_root / "code" / "dedup_report.json"
    if not dedup_p.exists():
        raise FileNotFoundError(
            f"pre-gen feasibility needs the dedup report at {dedup_p} "
            "(run issue2388_gen.py --phase dedup)"
        )
    dedup = json.loads(dedup_p.read_text())
    n_lcb_kept = int(dedup["n_lcb"]) - int(dedup["n_dropped_lcb"])
    base_pool = (
        G.EXPECTED_COUNTS["humaneval"]
        + G.EXPECTED_COUNTS["mbpp_full"]
        + n_lcb_kept
        + G.EXPECTED_COUNTS["leetcode"]
    )
    code_pool = base_pool
    gate_p = gen_root / "code" / "code_gate.json"
    if gate_p.exists() and json.loads(gate_p.read_text()).get("bcb_fit_allowed") is True:
        code_pool = base_pool + G.EXPECTED_COUNTS["bigcodebench_full"]
    pools = {
        "math": G.EXPECTED_COUNTS["math_full"],
        "mcq": G.EXPECTED_COUNTS["mmlu_pro_full"],
        "code": code_pool,
    }
    counts = {
        s: {
            "train": round(0.7 * n),
            "dev": round(0.1 * n),
            "test": round(0.2 * n),
        }
        for s, n in pools.items()
    }
    # QA rides the BANKED labeling (pre-existing — available pre-generation).
    qa_lab = Path(args.dv_root) / "qa" / "labeling.json"
    if qa_lab.exists():
        lab = _load_labeling(qa_lab, surface="qa")
        counts["qa"] = {
            s: sum(1 for r in lab["rows"] if r["eff_split"] == s) for s in ("train", "dev", "test")
        }
    return counts


def phase_feasibility(args) -> None:
    if args.pre_gen:
        counts = _pre_gen_counts(args)
        report = assert_joint_feasibility(counts, key_manifest=None)
        report["mode"] = "pre-gen-arithmetic"
    else:
        counts = {}
        for surface in args.surfaces or list(SURFACE_BENCHMARKS):
            lab = _load_labeling(Path(args.dv_root) / surface / "labeling.json", surface=surface)
            counts[surface] = {
                s: sum(1 for r in lab["rows"] if r["eff_split"] == s)
                for s in ("train", "dev", "test")
            }
        manifest = None
        mpath = Path(args.maps_out) / "key_manifest.json"
        if mpath.exists():
            manifest = json.loads(mpath.read_text())
        report = assert_joint_feasibility(counts, key_manifest=manifest)
    report["metadata"] = _provenance("feasibility")
    suffix = "_pregen" if args.pre_gen else ""
    path = Path(args.fits_root) / f"feasibility_report{suffix}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2))
    print(f"[feasibility] PASS ({report['mode']} mode) -> {path}", flush=True)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def phase_upload(args) -> None:
    """Mirror the maps + fits + h3_recompute trees to the HF data repo
    (plan section 10: maps -> analysis_tensors/maps; fits + preds + null
    matrices -> git issue branch + HF mirror). Smoke runs land under a
    ``_smoke``-suffixed HF root — never the production prefix."""
    from huggingface_hub import HfApi

    from explore_persona_space.orchestrate import hub

    hf_root = "issue2388_correctness" + ("_smoke" if args.smoke else "")
    jobs: list[tuple[Path, str]] = []
    maps_dir = Path(args.maps_out)
    if maps_dir.exists() and any(maps_dir.iterdir()):
        jobs.append((maps_dir, f"{hf_root}/analysis_tensors/maps"))
    fits_root = Path(args.fits_root)
    if fits_root.exists():
        for sub in sorted(p for p in fits_root.iterdir() if p.is_dir()):
            jobs.append((sub, f"{hf_root}/fits/{sub.name}"))
        for f in sorted(p for p in fits_root.iterdir() if p.is_file()):
            jobs.append((f, f"{hf_root}/fits/{f.name}"))
    h3_root = _h3_root(args)
    if h3_root.exists() and any(h3_root.iterdir()):
        jobs.append((h3_root, f"{hf_root}/h3_recompute"))
    if not jobs:
        raise RuntimeError("upload: nothing to upload (no maps/fits/h3_recompute artifacts)")
    api = HfApi()
    for local, prefix in jobs:
        if local.is_file():
            # file jobs compose prefix as the FULL file destination above
            # UPLOAD_LOOP_EXEMPT: bounded — top-level summary JSONs only (<=~6); dirs batch
            out = hub._upload(
                local,
                hub.DEFAULT_DATASET_REPO,
                repo_type="dataset",
                path_in_repo=prefix,
                upload_as_file=True,
            )
            expected = [prefix]
            prefix = prefix.rsplit("/", 1)[0]  # scope the verify to the parent
        else:
            out = hub._upload(
                local, hub.DEFAULT_DATASET_REPO, repo_type="dataset", path_in_repo=prefix
            )
            expected = [
                f"{prefix}/{p.relative_to(local)}" for p in sorted(local.rglob("*")) if p.is_file()
            ]
        if not out:
            raise RuntimeError(f"upload returned empty path for {local} -> {prefix}")
        missing = hub.verify_repo_paths_uploaded(
            api, hub.DEFAULT_DATASET_REPO, expected, path_in_repo=prefix, repo_type="dataset"
        )
        if missing:
            raise RuntimeError(
                f"post-upload verify: {len(missing)} paths missing under {prefix}: {missing[:5]}"
            )
        print(f"[upload] {local} -> {prefix} ({len(expected)} files verified)", flush=True)


PHASES = {
    "feasibility": phase_feasibility,
    "maps": phase_maps,
    "sweep": phase_sweep,
    "select": phase_select,
    "bootstrap": phase_bootstrap,
    "h3": phase_h3,
    "h3-gap": phase_h3_gap,
    "upload": phase_upload,
}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.replace("%", "%%"))
    ap.add_argument("--phase", choices=sorted(PHASES))
    ap.add_argument("--surface", choices=sorted(SURFACE_BENCHMARKS), default=None)
    ap.add_argument("--surfaces", nargs="*", default=None)
    ap.add_argument("--arms", nargs="*", default=None)
    ap.add_argument("--budgets", nargs="*", default=None)
    ap.add_argument("--keys", nargs="*", default=None)
    ap.add_argument("--behaviors", nargs="*", default=None)
    ap.add_argument("--h3-step", choices=("stage1", "verdict", "stage2"), default=None)
    ap.add_argument("--h3-out-root", default=None)
    ap.add_argument(
        "--h3-u-rung",
        default=H3_U_RUNG_DEFAULT,
        help="u_rung_label held at the registered H3 comparison cell (plan section 3 "
        "names no explicit value; 'full' = the parent's primary map — recorded in "
        "every gap report)",
    )
    ap.add_argument(
        "--map-cell",
        choices=("fu0", "fu05", "fu1", "additive"),
        default="fu1",
        help="which fitted map keys the mapped arms consume (fu1 = primary; "
        "fu0/fu05/additive = the H4 composition cells; cell tags gain a prefix)",
    )
    ap.add_argument(
        "--qa-disjoint",
        action="store_true",
        help="QA strict label-disjoint variant: labels drawn GROUP-disjoint from "
        "the map pool (plan section 4 part 3 / clause v)",
    )
    ap.add_argument(
        "--layers",
        type=int,
        default=None,
        help="layer COUNT (smoke scale dial; default: all N_LAYERS production layers)",
    )
    ap.add_argument(
        "--hidden-dim",
        type=int,
        default=HIDDEN_DIM,
        help="store hidden dim (smoke fixtures override; production default pinned)",
    )
    ap.add_argument("--banked-root", default="eval_results/issue_1739")
    ap.add_argument("--dv-root", default=str(DV_ROOT))
    ap.add_argument("--gen-root", default="eval_results/issue_2388/gen")
    ap.add_argument("--fits-root", default=str(FITS_ROOT))
    ap.add_argument("--maps-out", default=str(MAPS_OUT))
    ap.add_argument("--store-root", default="/workspace/store_2388")
    ap.add_argument("--qa-store-dir", default="/workspace/store")
    ap.add_argument(
        "--qa-questions-shards",
        default=None,
        help="dir holding the staged banked #1739 labeling raw-completion shards "
        f"({_QA_SHARD_GLOB}; HF data repo: {QA_SHARDS_HF_PREFIX}/) — enables the QA "
        "question-text baselines bl_feats/bl_extemb (concern qa-question-text-source)",
    )
    ap.add_argument("--u-store-dir", default="/workspace/u_store")
    ap.add_argument(
        "--h3-store-root",
        default="/workspace/h3_stores",
        help="staged parent stores root: <behavior>_labeling + <behavior>_extraction "
        "(hallucination's labeled store rides --qa-store-dir — same store, plan section 9)",
    )
    ap.add_argument(
        "--pre-gen",
        action="store_true",
        help="feasibility from loader metadata + dedup report (runs BEFORE any generation)",
    )
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--n-null", type=int, default=N_NULL)
    ap.add_argument("--n-draws", type=int, default=N_DRAWS)
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument(
        "--smoke",
        action="store_true",
        help="tiny nulls/boot counts; production code paths unchanged",
    )
    ap.add_argument(
        "--ported-extra",
        nargs="*",
        default=None,
        help="extra argv forwarded to the ported issue1739_fits.py",
    )
    ap.add_argument("--import-check", action="store_true")
    ap.add_argument("--list-phases", action="store_true")
    args = ap.parse_args(argv)

    if args.list_phases:
        print(" ".join(sorted(PHASES)))
        return 0
    if args.import_check:
        from explore_persona_space.orchestrate.argcheck import assert_args_attributes_defined

        assert_args_attributes_defined(__file__)
        import sklearn.feature_extraction.text  # noqa: F401
        import torch  # noqa: F401

        # (batched_ridge_predict_loco_pca deliberately NOT imported: PCA arms
        # ride the dof-capped GCV core — the uniform instrument; r1 g6 noted
        # the unused import implied wiring that does not exist.)
        from explore_persona_space.analysis.mapping_baselines import (  # noqa: F401
            identity_bias_predict,
            knn_retrieval,
        )
        from explore_persona_space.experiments.issue_1739.store_io import (  # noqa: F401
            load_summaries,
        )
        from explore_persona_space.orchestrate import hub

        assert callable(hub._upload) and callable(hub.verify_repo_paths_uploaded)

        # GPU/network-fenced (noted): sentence_transformers model download.
        import sentence_transformers  # noqa: F401

        print("[import-check] ok (network-fenced: SentenceTransformer(ENCODER_ID) download)")
        return 0
    if not args.phase:
        raise SystemExit("--phase required (or --import-check / --list-phases)")
    if args.smoke:
        args.n_null = min(args.n_null, 5)
        args.n_boot = min(args.n_boot, 50)
        # LOCAL smoke out-roots (r2 Codex Critical 3 / fits-smoke-local-outroot):
        # a default --smoke run must never write the canonical fits/maps roots
        # (same convention as gen/capture; the HF side already _smoke-suffixes).
        # The h3_recompute sibling is smoke-suffixed by _h3_root(args).
        if args.fits_root == str(FITS_ROOT):
            args.fits_root = str(FITS_ROOT) + "_smoke"
        if args.maps_out == str(MAPS_OUT):
            args.maps_out = str(MAPS_OUT) + "_smoke"
    PHASES[args.phase](args)
    sys.stdout.flush()
    sys.stderr.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
