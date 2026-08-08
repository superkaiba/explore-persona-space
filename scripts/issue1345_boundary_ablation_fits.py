#!/usr/bin/env python
"""Issue #1345 story-boundary-ablation — fit battery + verdict lattice.

CPU-only. Five legs, all on the SHARED-factorization ridge core (one
eigh(Gram) per (source, fold), the lambda grid applied as diagonal filter
rescalings of that ONE factorization, and nulls as Y-permutations against the
SAME cached factorization — never a re-solve per lambda / per draw):

  1. cells      per ablation arm x the 5 store slots (prefix / ctx_qend /
                context / ctx_preans / ctx_straddle) x both Y targets (answer
                mean / y_boundary) on the arm's OWN store, with the
                registered conversation-grouped folds, shuffle nulls, the
                random-projection + mean baselines, and conversation-level
                bootstrap CIs.
  2. baselines  identity+learned-bias (v_hat = x + b) and kNN retrieval at the
                headline layer for EVERY cell, out-of-fold on the same folds
                (standing rule: both reads accompany every fitted map).
  3. matched    chat (r1) and no-template (r2) comparator refits restricted to
                each arm's kept conversation set, plus a V1-anchor refit on the
                arm-and-V1 intersection when the V1 store is staged — so the
                headline arm-vs-V1 and arm-vs-chat deltas are PAIRED bootstraps
                on a shared conversation set, not two independent CIs.
  4. reparam    story-arm <-> chat in BOTH directions at the headline layer:
                the direct / ctx-reparam (A) / ans-reparam (B) / AMB rungs with
                a matched-capacity shuffle-fit null per rung.
  5. verdict    per-arm lattice: the headline-slot read vs the V1 anchor, vs the
                matched chat comparator, and vs its own null band — plus the
                per-arm README naming what the arm isolates.

Both mapping arms are fit per cell by construction: the `prefix` slot IS the
prefix-arm map (everything before the query) and the three ctx_* slots are
context-arm maps (prefix + query) at different read positions.

The X x Y grid (addendum) runs on every captured store that carries the 5-slot x
2-target shape: each ablation arm, both round-own comparators, AND the
re-captured V1 boundary-PRESENT anchor (`capture --arm v1`) — the row that makes
the grid an ablation rather than a description.

`--stage-v1` stages the PARENT V1 turnstore (`instruct_stories_paired_s`, ~5.3
GB) that the paired arm-vs-V1 bootstrap needs; it is NOT one of
issue1345_prefetch_reuse's reuse stems, so without the flag that read records
`skipped`.

CLI:
  uv run python scripts/issue1345_boundary_ablation_fits.py --phase all --stage-v1
  uv run python scripts/issue1345_boundary_ablation_fits.py --phase all --smoke
  uv run python scripts/issue1345_boundary_ablation_fits.py --import-check
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
for _p in (str(_SCRIPT_DIR), str(_REPO_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

import numpy as np  # noqa: E402

import issue825_fit_cells as fc  # noqa: E402
import issue1345_boundary_ablation_capture as bc  # noqa: E402
import issue1345_boundary_ablation_gen as bg  # noqa: E402
import issue1345_common as c  # noqa: E402
from issue1345_fit_cells import degenerate_fold_reason  # noqa: E402

from explore_persona_space.analysis import mapping_baselines as mb  # noqa: E402

LAYER = c.HEADLINE_LAYER  # 19
N_NULL_DRAWS = 40  # brief: 40 shuffle nulls
N_REPARAM_NULL_DRAWS = 20
SMOKE_NULL_DRAWS = 3
SMOKE_BOOT = 50
KNN_KS = (1, 5, 10)
# Slim load: the fits never read the per-position tensors (~36 GB/bundle stacked).
SLIM_KEYS = ("slots", "profiles", "nll")
# Arms whose story<->chat reparam ladder runs by default (brief: V2 + V4 at
# minimum — V3 rides along because the ladder is cheap once the stores are open).
# V5 is deliberately OUT: its pre-registered contrasts (below) are grid/comparator
# reads, not ladder reads, so the default stays byte-identical in compute for the
# already-planned run. Opt in per run with `--reparam-arms v5`.
DEFAULT_REPARAM_ARMS = (bg.ARM_V2, bg.ARM_V3, bg.ARM_V4)

# Pre-registered per-arm contrasts: which ALREADY-COMPUTED verdict fields decide
# the arm's question, named at BUILD time so the analyzer cannot pick a different
# pair post hoc. Reads only — no new compute.
PRE_REGISTERED_CONTRASTS: dict[str, dict] = {
    bg.ARM_V5: {
        "boundary_form": {
            "question": (
                "is V1's anchor effect carried by the PROSE ATTRIBUTION or by the "
                "pretraining-familiar turn syntax? V5 holds the story constant and "
                "swaps the attribution for a bare turn label."
            ),
            "decided_by": "CI overlap, V5 vs the boundary-present V1 anchor",
            "read_fields": [
                "headline.r2_reduced_basis_primary + headline.ci",
                "xy_grid.v1_anchor (same X x Y grid on the re-captured V1 store)",
                "paired_deltas (matched-row V5-vs-V1 bootstrap)",
            ],
            "interpretation": (
                "V5 ~ V1 (CIs overlap) => the boundary FORM does not matter, the "
                "boundary's presence does; V5 ~ V2 (boundary-absent) => V1's effect "
                "needs the prose attribution specifically."
            ),
        },
        "residual_story_cost": {
            "question": (
                "at MATCHED boundary syntax (both read at a 'User: '-style ':'), how "
                "much of the story-vs-chat gap survives?"
            ),
            "decided_by": "CI overlap, V5 vs the no_template comparator",
            "read_fields": [
                "vs_matched_chat.no_template_same_rows (matched-row parent r2 read)",
                "xy_grid.comparators.no_template (round-own X x Y comparator grid)",
            ],
            "interpretation": (
                "CIs overlap => the residual story cost at matched boundary syntax is "
                "not resolvable at this n; V5 below no_template => a story-frame cost "
                "remains after the boundary syntax is matched."
            ),
        },
        "note": (
            "both contrasts read the SAME fields every other arm already emits — V5 "
            "adds no new statistic, only the pre-registration of which comparison "
            "answers which question."
        ),
    },
}

# V1 anchor: the landed conversation_paired_stories_assistant reads. Literals are
# documentation cross-checks; the values are read LIVE from the committed JSONs.
V1_ANCHOR_DIR = Path("eval_results/issue_1345/conversation_paired_stories_assistant")
V1_ANCHOR_FILES = {
    "context": V1_ANCHOR_DIR / "cells_R_instruct_r4_context.json",
    "prefix": V1_ANCHOR_DIR / "cells_R_instruct_r4_prefix.json",
}
V1_MATCHED_CHAT_FILES = {
    "context": V1_ANCHOR_DIR / "matched_row" / "cells_R_instruct_r1_matched_context.json",
    "prefix": V1_ANCHOR_DIR / "matched_row" / "cells_R_instruct_r1_matched_prefix.json",
}
V1_ANCHOR_DOC = {"context": -0.3056, "prefix": -1.3714}
V1_MATCHED_CHAT_DOC = {"context": 0.2426, "prefix": 0.1313}
# The V1 store's own stem + slot order (2 slots: prefix, context).
V1_STEM_FORMAT = "stories_paired"
V1_SLOT_INDEX = {"prefix": 0, "context": 1}
# The parent V1 turnstore is NOT one of issue1345_prefetch_reuse's four r1/r2
# reuse stems, so `--stage-v1` stages it explicitly; without it `v1_available`
# reads False and every paired arm-vs-V1 bootstrap silently records `skipped`.
V1_TURNSTORE_HF_PREFIX = f"issue1345_framing/{bc.V1_PARENT_VARIANT}/analysis_tensors/turnstore"
V1_TURNSTORE_SHARDS = 5  # instruct_stories_paired_s_shard000..004 (+ sidecars)

# PARENT comparator stores staged by issue1345_prefetch_reuse.py (2 slots,
# Y_MEAN only) — retained for the V1-PARITY comparator read whose committed
# value this round cross-checks. The X x Y grid comparators are the round's own
# bnd_chat / bnd_ntpl stores below.
COMPARATOR_FORMAT = {"r1_chat": "chat", "r2_no_template": "naturalistic"}
COMPARATOR_TURN_INDEX = 1  # r1/r2 single-turn track-S rows sort [u1, a1]
# Round-own X x Y comparator stores (issue1345_boundary_ablation_capture
# --comparator): same 5 slots + 2 Y targets as every ablation arm.
BND_COMPARATORS = bc.COMPARATORS  # ("chat", "no_template")
# Which store slot each mapping arm reads in the PARENT r1/r2/V1 stores.
MAP_ARM_SLOT = dict(c.ARM_SLOT_INDEX)  # {"prefix": 0, "context": 1}
# Which mapping arm each boundary-store slot belongs to (both arms are covered:
# `prefix` is the prefix-arm map; every ctx_*/x_* slot is a context-arm map).
SLOT_MAP_ARM = {
    "prefix": "prefix",
    "ctx_qend": "context",
    "context": "context",
    "ctx_preans": "context",
    "ctx_straddle": "context",
}
# Short tags for the two Y targets in cell ids.
Y_TAG = {bc.Y_MEAN: "ymean", bc.Y_BOUNDARY: "ybnd"}


# ---------------------------------------------------------------------------
# Cell registry
# ---------------------------------------------------------------------------
# Short cell-id tag per grid store (V1's full arm name would bloat every id).
GRID_TAG = {**bg.ARM_SLUG, bc.V1_ARM: bc.V1_SLUG}


def grid_cell_id(
    store_key: str,
    slot: str,
    y: str,
    provenance: str = c.PROV_INJECTED,
    model_key: str = bg.MODEL_KEY,
) -> str:
    """Cell id for one (store, X slot, Y target, provenance, model) grid point.

    The `injected` + round-default-model call appends nothing, so every
    pre-existing grid cell id is byte-unchanged; the on-policy twin reads
    ..._bnd_chat_op_context__ymean, and a PRETRAINED-measured store carries the
    model slug (3 of the 4 on-policy bundles are base-written).
    """
    tag = GRID_TAG.get(store_key, store_key) + c.prov_suffix(provenance)
    return f"R_{c.MODEL_SLUG[model_key]}_bnd_{tag}_{slot}__{Y_TAG[y]}"


def grid_cells(
    store_key: str,
    provenance: str = c.PROV_INJECTED,
    model_key: str = bg.MODEL_KEY,
) -> list[dict]:
    """The store's full X x Y grid: every BND slot crossed with both Y targets.

    ``store_key`` is an ablation arm (V2/V3/V4/V5), the re-captured V1 anchor, or
    a round-own comparator (``chat`` / ``no_template``) — all carry the identical
    5-slot x 2-target store shape, which is what makes the grid comparable
    across them (the V1 row is the boundary-PRESENT anchor the ablation arms are
    read against at matched (read position x target)).

    ``provenance`` selects WHO WROTE the answers the store reads. An `onpolicy`
    grid is the matched PAIRED ARM of the identical lattice — same slots, same Y
    targets, same conv_id space, same store shape — differing only in authorship,
    which is exactly the contrast the on-policy-vs-injected program measures.
    """
    fmt = bc.format_key(store_key, provenance)
    return [
        {
            "cell_id": grid_cell_id(store_key, slot, y, provenance, model_key),
            "model_key": model_key,
            "format_key": fmt,
            "track": bc.TRACK,
            "slot_index": idx,
            "target_turn_index": bc.Y_TARGET_INDEX[y],
            "regime": fmt,
            "bnd_arm": store_key,
            "provenance": provenance,
            "measured_model": model_key,
            "slot": slot,
            "y_target": y,
            "arm": SLOT_MAP_ARM[slot],
        }
        for idx, slot in enumerate(bc.BND_SLOT_ORDER)
        for y in bc.Y_SPAN_ORDER
    ]


def arm_cells(arm: str) -> list[dict]:
    """The ablation arm's own X x Y grid cells."""
    return grid_cells(arm)


CELL_SHARD_ENV = "EPM_I1345_CELL_SHARD"
# Phases that consume EVERY cell's output. A sharded process holds only its own
# slice, so running these under a shard would compute the grid/verdict over a
# fraction of the lattice and report it as complete.
WHOLE_LATTICE_PHASES = ("all", "grid", "reparam", "verdict")


def parse_cell_shard(spec: str | None) -> tuple[int, int] | None:
    """``"i/N"`` -> (i, N); None/empty -> None (the unsharded default path)."""
    if spec is None or not str(spec).strip():
        return None
    raw = str(spec).strip()
    assert raw.count("/") == 1, f"--cell-shard must look like i/N, got {raw!r}"
    i_s, n_s = raw.split("/")
    assert i_s.strip().isdigit() and n_s.strip().isdigit(), f"non-integer shard spec {raw!r}"
    i, n = int(i_s), int(n_s)
    assert n >= 1, f"shard count must be >= 1, got {n}"
    assert 0 <= i < n, f"shard index {i} out of range for {n} shards"
    return i, n


def shard_of_cell(cell_id: str, n: int) -> int:
    """Which shard owns ``cell_id`` — a CONTENT hash, stable across processes.

    Deliberately NOT the enumeration index: the op paired rows are
    presence-gated, so two instances can enumerate different-sized cell lists and
    an index-modulo partition would then assign the SAME cell to different shards
    (some cells fit twice, others never). A sha256 of the cell_id is immune to
    that, and to PYTHONHASHSEED — `hash()` is salted per process and would
    silently repartition on every run.
    """
    return int(hashlib.sha256(cell_id.encode()).hexdigest()[:8], 16) % n


def apply_cell_shard(cells: list[dict], shard: tuple[int, int] | None) -> list[dict]:
    """The subset of ``cells`` this shard owns (identity when unsharded)."""
    if shard is None:
        return cells
    i, n = shard
    return [cl for cl in cells if shard_of_cell(cl["cell_id"], n) == i]


def onpolicy_paired_cells(
    turnstore_dir: Path,
    store_keys: list[str],
    models: tuple[str, ...] = c.MODELS,
) -> tuple[list[dict], dict[str, bool]]:
    """The on-policy PAIRED grid rows for every store whose twin is on disk.

    Presence-gated exactly like the injected comparator stores: a registered
    on-policy twin joins the lattice when ITS store is present, and is reported
    as absent otherwise — so the fits run unchanged before the on-policy captures
    land, and pick the paired arm up automatically once they do (no ad-hoc run).
    Unregistered keys (the ablation arms, which are injection-BY-CONSTRUCTION)
    are skipped silently — they have no meaningful on-policy twin.
    """
    cells: list[dict] = []
    present: dict[str, bool] = {}
    fmt_cache: dict[str, str] = {}
    for key in store_keys:
        if not bc.has_onpolicy_twin(key):
            continue
        fmt = fmt_cache.setdefault(key, bc.format_key(key, c.PROV_ONPOLICY))
        # Per MEASURED model: the bare-text arm exists for BOTH, chat and
        # story-slot are base-written this round, and a store that was never
        # captured simply does not join (the presence gate, unchanged).
        for mk in models:
            ok = store_present(turnstore_dir, mk, fmt)
            present[f"{key}/{mk}"] = ok
            if ok:
                cells += grid_cells(key, c.PROV_ONPOLICY, mk)
    return cells, present


def comparator_cells(arm: str, label: str) -> list[dict]:
    """PARENT r1/r2 comparator cells for one arm (both mapping arms), matched-row.

    These are the V1-PARITY reads (2-slot parent stores, Y_MEAN only) whose
    committed values this round cross-checks; the X x Y comparator grid lives on
    the round's own bnd_chat / bnd_ntpl stores (``grid_cells``).
    """
    return [
        {
            "cell_id": f"R_{bg.MODEL_KEY}_{label}_bnd_{bg.ARM_SLUG[arm]}_{map_arm}",
            "model_key": bg.MODEL_KEY,
            "format_key": COMPARATOR_FORMAT[label],
            "track": bc.TRACK,
            "slot_index": MAP_ARM_SLOT[map_arm],
            "target_turn_index": COMPARATOR_TURN_INDEX,
            "regime": COMPARATOR_FORMAT[label],
            "bnd_arm": arm,
            "slot": map_arm,
            "y_target": bc.Y_MEAN,
            "arm": map_arm,
        }
        for map_arm in c.ARMS
    ]


def v1_matched_cells(arm: str) -> list[dict]:
    """V1-anchor refit cells on the arm-and-V1 intersection (both mapping arms)."""
    return [
        {
            "cell_id": f"R_{bg.MODEL_KEY}_v1_bnd_{bg.ARM_SLUG[arm]}_{map_arm}",
            "model_key": bg.MODEL_KEY,
            "format_key": V1_STEM_FORMAT,
            "track": bc.TRACK,
            "slot_index": V1_SLOT_INDEX[map_arm],
            "target_turn_index": 0,
            "regime": V1_STEM_FORMAT,
            "bnd_arm": arm,
            "slot": map_arm,
            "y_target": bc.Y_MEAN,
            "arm": map_arm,
        }
        for map_arm in c.ARMS
    ]


def arm_matched_cells(arm: str) -> list[dict]:
    """The arm's OWN cells refit on the arm-and-V1 intersection (paired vs V1)."""
    return [
        {
            "cell_id": f"R_{bg.MODEL_KEY}_bndm_{bg.ARM_SLUG[arm]}_{map_arm}",
            "model_key": bg.MODEL_KEY,
            "format_key": bc.format_key(arm),
            "track": bc.TRACK,
            # The intersection refit compares against the V1 store's 2 slots, so
            # it reads the boundary store's matching positions only.
            "slot_index": list(bc.BND_SLOT_ORDER).index(
                "prefix" if map_arm == "prefix" else bc.HEADLINE_SLOT
            ),
            "target_turn_index": bc.Y_TARGET_INDEX[bc.Y_MEAN],
            "regime": bc.format_key(arm),
            "bnd_arm": arm,
            "slot": map_arm,
            "y_target": bc.Y_MEAN,
            "arm": map_arm,
        }
        for map_arm in c.ARMS
    ]


# ---------------------------------------------------------------------------
# Bundle access
# ---------------------------------------------------------------------------
def load_bundle(turnstore_dir: Path, model_key: str, format_key: str, expect_slots: int) -> dict:
    """Load one store via the production pt-shard loader + sanity asserts."""
    bundle = fc._load_bundle_any(
        turnstore_dir, model_key, format_key, bc.TRACK, wanted_keys=SLIM_KEYS
    )
    c.assert_pt_bundle(bundle, expect_slots=expect_slots, expect_layers=fc.EXPECTED_LAYERS)
    return bundle


def stage_v1_turnstore(turnstore_dir: Path) -> int:
    """Stage the PARENT's `instruct_stories_paired_s` turnstore into the flat dir.

    Item (d) of the addendum: without this the V1 store is absent, `v1_available`
    reads False, and the paired arm-vs-V1 bootstrap records `skipped` SILENTLY —
    the read the ablation is compared against just disappears from the verdict.

    Per-file `hf_hub_download` through the shared retried helper (never
    `snapshot_download` on the ~1M-file data repo — gotchas.md), staged into a
    scratch MIRROR dir under `turnstore_dir` and published per file with
    `os.replace` (same filesystem, atomic, no EXDEV — gotchas.md), because the
    hub layout nests the repo path while the fit loader expects flat stems
    (the #1774 mirror-root trap). Idempotent: an already-flat file is skipped.
    """
    import os
    import shutil

    stem = f"{bg.MODEL_KEY}_{V1_STEM_FORMAT}_{bc.TRACK}"
    names = [
        f"{stem}_shard{i:03d}{ext}" for i in range(V1_TURNSTORE_SHARDS) for ext in (".pt", ".json")
    ]
    turnstore_dir.mkdir(parents=True, exist_ok=True)
    missing = [n for n in names if not (turnstore_dir / n).exists()]
    if not missing:
        print(f"[fits][stage-v1] all {len(names)} V1 store files already present", flush=True)
        return 0
    # Headroom check BEFORE any byte lands (the ~5 GB staging discipline): ONE
    # server-side-SCOPED tree listing for the sizes, retried like every Hub call.
    from huggingface_hub import HfApi

    from explore_persona_space.orchestrate.hub import retry_transient

    api = HfApi(token=os.environ.get("HF_TOKEN"))
    entries = retry_transient(
        lambda: list(
            api.list_repo_tree(  # HUB_VERIFY_RETRY_EXEMPT: wrapped in retry_transient
                c.HF_DATA_REPO,
                path_in_repo=V1_TURNSTORE_HF_PREFIX,
                repo_type="dataset",
                recursive=True,
            )
        ),
        what=f"list_repo_tree({V1_TURNSTORE_HF_PREFIX})",
    )
    sizes = {e.path.rsplit("/", 1)[-1]: int(getattr(e, "size", 0) or 0) for e in entries}
    absent = [n for n in missing if n not in sizes]
    assert not absent, (
        f"V1 store files absent on the Hub under {V1_TURNSTORE_HF_PREFIX}: {absent} "
        "— the parent bundle layout changed; re-verify the anchor before staging"
    )
    need = sum(sizes[n] for n in missing)
    free = shutil.disk_usage(turnstore_dir).free
    print(
        f"[fits][stage-v1] staging {len(missing)}/{len(names)} files "
        f"(~{need / 1e9:.2f} GB declared) into {turnstore_dir} (free {free / 1e9:.1f} GB)",
        flush=True,
    )
    assert free >= 1.5 * need, (
        f"insufficient headroom staging the V1 store: need ~{need / 1e9:.2f} GB x1.5, "
        f"free {free / 1e9:.2f} GB at {turnstore_dir}"
    )
    scratch = turnstore_dir / ".v1_stage"
    for n in missing:
        src = c.stage_pinned_file(f"{V1_TURNSTORE_HF_PREFIX}/{n}", scratch, revision="main")
        os.replace(src, turnstore_dir / n)
    shutil.rmtree(scratch, ignore_errors=True)
    print(f"[fits][stage-v1] staged {len(missing)} V1 store files", flush=True)
    return len(missing)


def store_present(turnstore_dir: Path, model_key: str, format_key: str) -> bool:
    """Cheap presence probe (npz contract OR pt shards) with no load."""
    stem = f"{model_key}_{format_key}_{bc.TRACK}"
    return bool((turnstore_dir / f"{stem}.npz").exists() or list(turnstore_dir.glob(f"{stem}*.pt")))


def store_conv_ids(turnstore_dir: Path, model_key: str, format_key: str) -> list[str]:
    """Row conv_ids read from the cheap shard sidecars (no tensor load)."""
    stem = f"{model_key}_{format_key}_{bc.TRACK}"
    sidecars = sorted(turnstore_dir.glob(f"{stem}_shard*.json"))
    if not sidecars:
        side = turnstore_dir / f"{stem}.json"
        assert side.exists(), f"no sidecars for {stem} in {turnstore_dir}"
        return [str(x) for x in json.loads(side.read_text())["conv_ids"]]
    ids: list[str] = []
    for sp in sidecars:
        ids.extend(str(x) for x in json.loads(sp.read_text())["conv_ids"])
    return ids


# ---------------------------------------------------------------------------
# Leg 1+2 — per-cell fits + the two standing mapping baselines
# ---------------------------------------------------------------------------
def mapping_baseline_reads(
    xy: dict, *, n_folds: int, seed: int, n_boot: int, ridge_pred: np.ndarray | None
) -> dict:
    """identity+learned-bias and kNN retrieval at the headline layer, out-of-fold.

    The identity+bias baseline (v_hat = x + b, b = train-fold mean of y - x)
    isolates how much of the map's R^2 a context-independent constant shift
    already explains; kNN retrieval reports P(true target within the k nearest
    neighbours of the prediction) among the held-out pool, with chance stated.
    The folds/seed are the cell's own, so every read is on the same split.
    """
    X = xy["X"][:, LAYER, :].astype(np.float64)
    Y = xy["Y"][:, LAYER, :].astype(np.float64)
    conv = np.asarray([str(x) for x in xy["conv_ids"]])
    if X.shape[1] != Y.shape[1]:
        return {
            "identity_bias": {
                "inapplicable": f"d_in {X.shape[1]} != d_out {Y.shape[1]} — identity+bias "
                "baseline needs a same-space map"
            }
        }
    folds = fc._cv_folds(conv, n_folds, seed)
    pred = np.zeros_like(Y)
    fitted = np.zeros(len(Y), bool)
    for k in range(n_folds):
        tr, te = folds != k, folds == k
        if te.sum() == 0 or tr.sum() < 3:
            continue
        pred[te] = mb.identity_bias_predict(X[tr], Y[tr], X[te])
        fitted[te] = True
    if not fitted.any():
        return {"identity_bias": {"skipped": "no usable folds"}}
    ib_pred, true = pred[fitted], Y[fitted]
    out = {
        "identity_bias": {
            **c.conv_bootstrap_r2(ib_pred, true, conv[fitted], n_boot=n_boot, seed=seed + 900),
            "b_norm": float(np.linalg.norm(mb.identity_bias_predict(X, Y, X[:1]) - X[:1])),
        },
        "knn_identity_bias": {
            m: mb.knn_retrieval(ib_pred, true, ks=KNN_KS, metric=m) for m in ("euclidean", "cosine")
        },
    }
    if ridge_pred is not None:
        assert ridge_pred.shape == true.shape, (ridge_pred.shape, true.shape)
        out["knn_ridge"] = {
            m: mb.knn_retrieval(ridge_pred.astype(np.float64), true, ks=KNN_KS, metric=m)
            for m in ("euclidean", "cosine")
        }
    return out


def run_cells(
    cells: list[dict],
    bundles: dict[tuple[str, str], dict],
    out_dir: Path,
    preds_dir: Path,
    allow_by_cell: dict[str, list[str]],
    *,
    n_folds: int,
    seed: int,
    null_draws: int,
    n_boot: int,
    smoke: bool,
    resume: bool = True,
) -> dict[str, dict]:
    """Fit each cell, persist its JSONs + OOF preds, and attach the baselines.

    Mirrors issue1345_fit_cells.run_cells (same shared-bundle injection, same
    conversation-level bootstrap, same preds npz contract) and additionally
    computes the two standing mapping baselines from the cell's OWN folds.
    """
    preds_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, dict] = {}
    for cell in cells:
        cid = cell["cell_id"]
        bundle = bundles[(cell["model_key"], cell["format_key"])]
        allow = allow_by_cell.get(cid)
        regime = _fit_regime(
            cell, allow, n_folds=n_folds, seed=seed, null_draws=null_draws, n_boot=n_boot
        )
        if resume:
            done = _resume_cell(out_dir, preds_dir, cid, regime)
            if done is not None:
                summary[cid] = {**done, "cell": cell}
                print(f"[fits] {cid} RESUMED (regime match) — skipping refit", flush=True)
                continue
        if smoke:
            xy_probe = fc._apply_row_allowlist(fc._cell_xy(bundle, cell), allow, cid)
            reason = degenerate_fold_reason(xy_probe["conv_ids"], n_folds=n_folds, seed=seed)
            if reason:
                print(
                    f"[fits][smoke] SKIP cell {cid}: {reason} — informational "
                    "(production semantics unchanged)",
                    flush=True,
                )
                summary[cid] = {"skipped": reason, "cell": cell}
                continue
        res = fc.run_cell(
            cell,
            Path("."),  # unused: bundle is injected
            out_dir,
            n_folds=n_folds,
            seed=seed,
            null_draws=null_draws,
            n_boot=n_boot,
            allowlist=allow,
            bundle=bundle,
        )
        sweep, xy = res["sweep"], res["xy"]
        fitted = sweep["fitted_mask"]
        li = LAYER if LAYER in sweep["preds_frozen"] else max(sweep["preds_frozen"])
        pred = sweep["preds_frozen"][li][fitted]
        true = xy["Y"][fitted, li, :]
        conv = np.asarray([str(x) for x in xy["conv_ids"][fitted]])
        np.savez(
            preds_dir / f"{cid}_L{li}.npz",
            pred=pred.astype(np.float32),
            true=true.astype(np.float32),
            conv_ids=conv,
            layer=np.asarray([li]),
        )
        boot = {
            str(int(lj)): c.conv_bootstrap_r2(
                sweep["preds_frozen"][lj][fitted],
                xy["Y"][fitted, lj, :],
                conv,
                n_boot=n_boot,
                seed=seed + 200 + lj,
            )
            for lj in sweep["preds_frozen"]
        }
        baselines = mapping_baseline_reads(
            xy, n_folds=n_folds, seed=seed, n_boot=n_boot, ridge_pred=pred if li == LAYER else None
        )
        # Well-posedness companions on the SAME rows/folds as the ambient fit:
        # the ambient GCV read above is n<d artifact-bearing on this line.
        companions = wellposed_companions(
            xy["X"][fitted, li, :], true, conv, n_folds=n_folds, seed=seed
        )
        cell_json = out_dir / f"cells_{cid}.json"
        payload = json.loads(cell_json.read_text())
        payload["r2_bootstrap_ci_frozen_layers_conv"] = boot
        payload["n_groups"] = len(np.unique(conv))
        payload["mapping_baselines_headline_layer"] = baselines
        payload["wellposed_companions"] = companions
        payload["bnd_arm"] = cell.get("bnd_arm")
        payload["slot"] = cell.get("slot")
        payload["y_target"] = cell.get("y_target")
        payload["bnd_fit_regime"] = regime
        c.write_json(cell_json, payload)
        summary[cid] = {
            "cell": cell,
            "layer": int(li),
            "r2": float(payload["r2_per_layer_obs"][li]),
            "ci": boot.get(str(int(li))),
            "null_p975": _null_p975(out_dir, cid, li),
            "mean_baseline_r2": payload["mean_baseline_r2"].get(str(li)),
            "skill_over_mean": payload["skill_over_mean"].get(str(li)),
            "baselines": baselines,
            # PRIMARY within-R2 read for the verdict (the ambient `r2` above is
            # the artifact-bearing continuity companion).
            "r2_reduced_basis": companions["reduced_basis"]["r2_heldout"],
            "companions": companions,
        }
        print(
            f"[fits] {cid} done (n={len(conv)}, groups={payload['n_groups']}, "
            f"L{li} R2={summary[cid]['r2']:.4f})",
            flush=True,
        )
    return summary


def _fit_regime(
    cell: dict, allow: list[str] | None, *, n_folds: int, seed: int, null_draws: int, n_boot: int
) -> dict:
    """Every output-affecting key of one cell fit (the resume identity).

    A resume that ignores ANY of these silently reuses wrong cached rows
    (code-style.md § Checkpoint per phase, the #722 r3 class): the store, the
    read position, the Y target, the fold/seed/null/bootstrap dials, and the
    allowlist the rows were restricted to.
    """
    return {
        "format_key": cell["format_key"],
        "slot_index": int(cell["slot_index"]),
        "target_turn_index": int(cell["target_turn_index"]),
        "n_folds": int(n_folds),
        "seed": int(seed),
        "null_draws": int(null_draws),
        "n_boot": int(n_boot),
        "n_allowlist": (len(allow) if allow is not None else None),
        "allowlist_sha": (
            hashlib.sha256(json.dumps(sorted(str(x) for x in allow)).encode()).hexdigest()[:16]
            if allow is not None
            else None
        ),
        "layer": LAYER,
        # Companion regime (well-posedness reads): a change to the reduced-basis
        # k rule or the forced-lambda set changes the persisted output, so it is
        # part of the resume identity — never a silently-reused stale companion.
        "reduced_k_cap": int(REDUCED_K_CAP),
        "forced_lambdas": [float(v) for v in FORCED_LAMBDAS],
    }


def _resume_cell(out_dir: Path, preds_dir: Path, cid: str, regime: dict) -> dict | None:
    """Reload a completed cell when its persisted regime matches EXACTLY.

    74 grid cells at production n trip both intra-phase checkpoint triggers
    (>50 units and >1 h projected), so a mid-phase kill must not forfeit the
    completed cells. `fc.run_cell` already persists cells_/nulls_ JSONs per
    cell and this driver the preds npz — the missing half was the resume
    predicate, keyed on the full regime fingerprint.
    """
    cell_json = out_dir / f"cells_{cid}.json"
    npz = preds_dir / f"{cid}_L{LAYER}.npz"
    if not (cell_json.exists() and npz.exists()):
        return None
    try:
        payload = json.loads(cell_json.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    if payload.get("bnd_fit_regime") != regime:
        return None
    if "mapping_baselines_headline_layer" not in payload:
        return None
    if "wellposed_companions" not in payload:
        return None  # pre-companion cell JSON — refit rather than resume half a cell
    boot = payload.get("r2_bootstrap_ci_frozen_layers_conv", {})
    return {
        "cell": payload.get("cell", {}),
        "layer": LAYER,
        "r2": float(payload["r2_per_layer_obs"][LAYER]),
        "ci": boot.get(str(LAYER)),
        "null_p975": _null_p975(out_dir, cid, LAYER),
        "mean_baseline_r2": payload.get("mean_baseline_r2", {}).get(str(LAYER)),
        "skill_over_mean": payload.get("skill_over_mean", {}).get(str(LAYER)),
        "baselines": payload.get("mapping_baselines_headline_layer"),
        "r2_reduced_basis": (payload["wellposed_companions"]["reduced_basis"]["r2_heldout"]),
        "companions": payload["wellposed_companions"],
        "resumed": True,
    }


def _null_p975(out_dir: Path, cell_id: str, layer: int) -> float | None:
    """Upper 97.5th percentile of the cell's shuffle-null R^2 at ``layer``."""
    p = out_dir / f"nulls_{cell_id}.json"
    if not p.exists():
        return None
    m = json.loads(p.read_text())["null_matrix"]
    if not m:
        return None
    col = [row[layer] for row in m if layer < len(row) and row[layer] == row[layer]]
    return float(np.quantile(col, 0.975)) if col else None


# ---------------------------------------------------------------------------
# Well-posedness companions — reduced PCA basis + forced lambda
#
# The sibling story-context-info-probe round showed this line's AMBIENT GCV
# ridge reads are an n<d ESTIMATION ARTIFACT, not an information result: at
# n_train 1730 < d 3584 GCV selects the lambda-grid FLOOR (0.01) in 5/5 folds on
# every full-basis story-input leg, and that floor is what produces the negative
# within-R2. Same rows, same basis, lambda FORCED to 1e3 moves story context ->
# story answer from -0.306 to +0.408; a per-fold train-only PCA basis moves it to
# +0.262. Retrieval and R2 DISSOCIATE across lambda (knn@1 falls as R2 rises), so
# both companions report kNN per lambda.
#
# Recipe copied from that round's `ridge_leg_reduced` + forced-single-value-grid
# legs (scripts/issue1345_story_info_probe.py @ 3ffc51d581) so these numbers are
# directly comparable to its published values. Cheap by construction: ONE
# `fc._prep_fold` per (X, fold) serves the GCV read AND every forced lambda
# (diagonal rescalings of the same cached eigh), and the reduced leg adds one
# train-only SVD per fold.
# ---------------------------------------------------------------------------
REDUCED_K_CAP = 1024
FORCED_LAMBDAS = (1e2, 1e3, 1e4)
# Published reference values these companions must be read against (source of
# record; NOT asserted — different rows/arms, so they are documentation).
PROBE_REFERENCE = {
    "source": (
        "eval_results/issue_1345/story_context_info_probe/"
        "{summary.json,forced_lambda_probe.json} @ 3ffc51d581 / 9f0fb74d4a"
    ),
    "story_vC_to_story_vA": {
        "ambient_gcv_r2": -0.306,
        "reduced_basis_r2_on_policy": 0.262,
        "reduced_basis_r2_teacher_forced": 0.367,
        "forced_lambda_1e2_r2": 0.1605,
        "forced_lambda_1e3_r2": 0.4075,
        "forced_lambda_1e4_r2": 0.4180,  # the PEAK of the swept grid, not 1e3
    },
    # Guard against a known mislabel: +0.4359 / +0.4403 at lambda 1e3 / 1e4 are
    # story context -> CHAT context, NOT story -> story answer (+0.4075 / +0.4180).
    "story_vC_to_chat_vC": {"forced_lambda_1e3_r2": 0.4359, "forced_lambda_1e4_r2": 0.4403},
    "note": (
        "ambient GCV picks the lambda-grid floor 0.01 in 5/5 folds at n_train<d; "
        "the committed V1 anchor -0.3056 is the ARTIFACT-BEARING read"
    ),
    "citation_guidance": (
        "for a single UNSELECTED headline figure cite the reduced-basis leg (per-fold "
        "GCV in a train-fold-only basis, no post-hoc selection); the forced-lambda "
        "sweep is a best-of-grid HELD-OUT estimate and carries selection"
    ),
}


def reduced_basis_k(n_train: int, d_in: int) -> int:
    """k for the well-posed companion basis: min(1024, floor(n_train/2), d_in)."""
    return int(max(1, min(REDUCED_K_CAP, n_train // 2, d_in)))


def _pooled_heldout_r2(y_true: np.ndarray, y_pred: np.ndarray, folds: np.ndarray) -> float:
    """Parent/probe convention: pooled 1 - SSE/SST with the HELD-OUT fold mean as SST."""
    ss_res = 0.0
    ss_tot = 0.0
    for f in np.unique(folds):
        te = folds == f
        t = y_true[te].astype(np.float64)
        ss_res += float(((t - y_pred[te].astype(np.float64)) ** 2).sum())
        ss_tot += float(((t - t.mean(0)) ** 2).sum())
    return float(1.0 - ss_res / ss_tot)


def _knn_reads(pred: np.ndarray, y: np.ndarray) -> dict:
    """kNN-through-the-map retrieval, both metrics (the standing mapping read)."""
    return {
        m: mb.knn_retrieval(pred, y, ks=list(KNN_KS), metric=m) for m in ("euclidean", "cosine")
    }


def companions_shared_x(
    x: np.ndarray, ys: dict[str, np.ndarray], conv_ids, *, n_folds: int, seed: int
) -> dict[str, dict]:
    """Companion reads for ONE X against MULTIPLE Y targets, X work SHARED.

    The ambient `_prep_fold` (the expensive eigh) and the train-only PCA basis
    depend ONLY on X, so both are computed once per fold and reused across every
    Y target — the same "X-side factorizations shared across Y" contract the
    ambient grid honors. Returns one block per Y name.
    """
    x = np.asarray(x)
    conv = np.asarray([str(v) for v in conv_ids])
    folds = fc._cv_folds(conv, n_folds, seed)
    uniq = np.unique(folds)
    n_tr_min = min(int((folds != f).sum()) for f in uniq)
    k = reduced_basis_k(n_tr_min, x.shape[1])
    forced = {
        name: {lam: np.zeros_like(np.asarray(y), dtype=np.float32) for lam in FORCED_LAMBDAS}
        for name, y in ys.items()
    }
    reduced = {name: np.zeros_like(np.asarray(y), dtype=np.float32) for name, y in ys.items()}
    reduced_lams: dict[str, list[float]] = {name: [] for name in ys}
    for f in uniq:
        te = folds == f
        tr = ~te
        # ONE ambient prep per fold — reused across every forced lambda AND every
        # Y target (lambda enters only as a diagonal rescaling of this eigh).
        cache = fc._prep_fold(x[tr], x[te])
        # Reduced basis: per-fold TRAIN-only PCA, centering only (the probe's
        # recipe; `_prep_fold` then standardizes the PCA coordinates, which is
        # what the published +0.262 reference measured). X-only => shared.
        mu = x[tr].mean(0)
        _, _, vt = np.linalg.svd(x[tr] - mu, full_matrices=False)
        basis = vt[: min(k, vt.shape[0])]
        cache_red = fc._prep_fold((x[tr] - mu) @ basis.T, (x[te] - mu) @ basis.T)
        for name, y_any in ys.items():
            y = np.asarray(y_any)
            for lam in FORCED_LAMBDAS:
                forced[name][lam][te] = np.asarray(
                    fc._ridge_predict_cached(cache, y[tr], lambdas=[lam]), dtype=np.float32
                )
            p_red, lam_red = fc._ridge_predict_cached(cache_red, y[tr], return_lam=True)
            reduced[name][te] = np.asarray(p_red, dtype=np.float32)
            reduced_lams[name].append(float(lam_red))
    out: dict[str, dict] = {}
    for name, y_any in ys.items():
        y = np.asarray(y_any)
        out[name] = {
            "folds": {"n_folds": int(len(uniq)), "seed": int(seed), "n_train_min": int(n_tr_min)},
            "d_in": int(x.shape[1]),
            "underdetermined_ambient": bool(n_tr_min < x.shape[1]),
            "reduced_basis": {
                "k": int(k),
                "k_rule": "min(1024, floor(n_train_min/2), d_in)",
                "pca_fit": "per-fold TRAIN rows only, centering only (no held-out leakage)",
                "pca_coords_standardized_by_prep_fold": True,
                "r2_heldout": _pooled_heldout_r2(y, reduced[name], folds),
                "gcv_lambda_per_fold": reduced_lams[name],
                "knn": _knn_reads(reduced[name], y),
            },
            "forced_lambda": {
                f"lambda_{lam:.0e}": {
                    "lambda": float(lam),
                    "r2_heldout": _pooled_heldout_r2(y, forced[name][lam], folds),
                    "knn": _knn_reads(forced[name][lam], y),
                }
                for lam in FORCED_LAMBDAS
            },
            # The forced sweep is DIAGNOSTIC: reading max-over-lambda as the map's
            # skill is post-hoc selection on the held-out estimate itself, so it
            # must never carry a headline. The reduced-basis leg above selects
            # lambda per fold by GCV inside a train-fold-only basis and is the
            # clean unselected citation (hence the verdict's PRIMARY read).
            "forced_lambda_selection_bearing": True,
            "forced_lambda_note": (
                "best-of-grid held-out estimate; diagnostic only — cite the "
                "reduced-basis read for any headline"
            ),
            "probe_reference": PROBE_REFERENCE,
        }
    return out


def wellposed_companions(
    x: np.ndarray, y: np.ndarray, conv_ids, *, n_folds: int, seed: int
) -> dict:
    """Single-Y wrapper over `companions_shared_x` (ONE companion recipe)."""
    return companions_shared_x(x, {"y": y}, conv_ids, n_folds=n_folds, seed=seed)["y"]


# ---------------------------------------------------------------------------
# Paired deltas (shared conversation set -> ONE counts matrix per draw)
# ---------------------------------------------------------------------------
def paired_delta(
    reads: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    pairs: list[tuple[str, str]],
    *,
    n_boot: int,
    seed: int,
) -> dict:
    """Conversation-level PAIRED bootstrap of R^2 differences.

    ``reads`` maps name -> (pred, true, conv_ids), all on the IDENTICAL
    conversation set (asserted). One shared counts matrix drives every
    statistic per draw, so a difference CI is a PAIRED read, not the
    non-overlap of two independent CIs.
    """
    suffs, uniq_ref = {}, None
    for name, (pred, true, conv) in reads.items():
        suff = c.conv_suffstats(pred, true, conv)
        if uniq_ref is None:
            uniq_ref = suff["uniq"]
        assert np.array_equal(suff["uniq"], uniq_ref), (
            f"{name}: conversation set mismatch — a paired delta needs identical groups"
        )
        suffs[name] = suff
    counts = c.bootstrap_counts(len(uniq_ref), n_boot, seed)
    r2 = {name: c.batched_conv_r2(counts, s) for name, s in suffs.items()}

    def _ci(v):
        return {
            "mean": float(np.nanmean(v)),
            "ci_lo": float(np.nanquantile(v, 0.025)),
            "ci_hi": float(np.nanquantile(v, 0.975)),
        }

    out = {
        "n_boot": int(n_boot),
        "n_groups": int(len(uniq_ref)),
        "unit": "conversation (paired resample across every named read)",
        "reads": {name: _ci(v) for name, v in r2.items()},
        "deltas": {},
    }
    for a, b in pairs:
        d = r2[a] - r2[b]
        ci = _ci(d)
        ci["ci_excludes_zero"] = bool(ci["ci_lo"] > 0.0 or ci["ci_hi"] < 0.0)
        out["deltas"][f"{a}_minus_{b}"] = ci
    return out


def _load_preds(preds_dir: Path, cell_id: str) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    p = preds_dir / f"{cell_id}_L{LAYER}.npz"
    if not p.exists():
        return None
    d = np.load(p, allow_pickle=False)
    return (
        d["pred"].astype(np.float64),
        d["true"].astype(np.float64),
        np.asarray([str(x) for x in d["conv_ids"]]),
    )


# ---------------------------------------------------------------------------
# Leg 4a — X x Y grid with the X-side factorization SHARED across Y
# ---------------------------------------------------------------------------
def _t(a: np.ndarray):
    import torch

    # On the fit device: keeps the grid/ladder ridge batteries on GPU when one
    # is visible (the 17912 crash class is separately closed source-side —
    # ma._ridge_prep now pins its inner caches to the caller's device).
    return torch.as_tensor(np.asarray(a), dtype=torch.float64, device=fc._fit_device())


def xy_grid(
    bundle: dict,
    store_key: str,
    *,
    allow: list[str] | None,
    n_folds: int,
    seed: int,
    null_draws: int,
    n_boot: int,
) -> dict:
    """Held-out R^2 for every (X slot, Y target) pair, X factorization SHARED.

    ONE eigh(Gram) per (X slot, fold) — `ma._ridge_prep` — is reused by BOTH Y
    targets and by every shuffle-null draw at that slot: the lambda grid rides
    the cached factorization as diagonal filter rescalings, and a null draw is a
    Y-permutation against the SAME prep. So the grid costs one X-side
    factorization per slot/fold, not one per (slot, target) — the addendum's
    "X-side factorizations shared across Y".

    Returns per-cell pooled R^2 + conversation-level bootstrap CI + the
    shuffle-null band, keyed ``"<slot>|<y>"``.
    """
    import issue825_map_alignment as ma
    import torch

    arrays = bundle["arrays"]
    slots = np.asarray(arrays["slots"], dtype=np.float32)
    profiles = np.asarray(arrays["profiles"], dtype=np.float32)
    conv_all = np.asarray([str(x) for x in bundle["sidecar"].get("conv_ids", [])])
    assert slots.shape[1] == len(bc.BND_SLOT_ORDER), (slots.shape, bc.BND_SLOT_ORDER)
    assert profiles.shape[1] == len(bc.Y_SPAN_ORDER), (profiles.shape, bc.Y_SPAN_ORDER)
    keep = np.ones(len(conv_all), bool)
    if allow is not None:
        keep = np.isin(conv_all, np.asarray(sorted(set(allow))))
        assert keep.any(), f"{store_key}: xy_grid allowlist selected zero rows"
    conv = conv_all[keep]
    folds = fc._cv_folds(conv, n_folds, seed)
    rng = np.random.default_rng(seed + 31)
    out: dict[str, dict] = {}
    for si, slot in enumerate(bc.BND_SLOT_ORDER):
        X = _t(slots[keep][:, si, LAYER, :])
        Ys = {y: _t(profiles[keep][:, bc.Y_TARGET_INDEX[y], LAYER, :]) for y in bc.Y_SPAN_ORDER}
        preds = {y: np.zeros((len(conv), Ys[y].shape[1]), np.float64) for y in bc.Y_SPAN_ORDER}
        fitted = np.zeros(len(conv), bool)
        null_acc = {
            y: [{"res": 0.0, "tot": 0.0} for _ in range(null_draws)] for y in bc.Y_SPAN_ORDER
        }
        for k in range(n_folds):
            tr, te = folds != k, folds == k
            if te.sum() == 0 or tr.sum() < 3:
                continue
            trt, tet = torch.as_tensor(tr), torch.as_tensor(te)
            # ONE factorization per (slot, fold) — reused across both Y targets
            # AND every null draw below.
            prep = ma._ridge_prep(X[trt])
            fitted[te] = True
            for y, Y in Ys.items():
                preds[y][te] = ma._ridge_predict(prep, Y[trt], X[tet]).cpu().numpy()
                mu = Y[trt].mean(0)
                tot = float(((Y[tet] - mu) ** 2).sum())
                for d in range(null_draws):
                    perm = rng.permutation(int(tr.sum()))
                    p_null = ma._ridge_predict(prep, Y[trt][torch.as_tensor(perm)], X[tet])
                    null_acc[y][d]["res"] += float(((Y[tet] - p_null) ** 2).sum())
                    null_acc[y][d]["tot"] += tot
        if not fitted.any():
            for y in bc.Y_SPAN_ORDER:
                out[f"{slot}|{y}"] = {"skipped": "no usable folds"}
            continue
        # Well-posedness companions for this slot, X-side work shared across BOTH
        # Y targets (the ambient reads above are n<d artifact-bearing).
        comp = companions_shared_x(
            X.cpu().numpy()[fitted],
            {y: Y.cpu().numpy()[fitted] for y, Y in Ys.items()},
            conv[fitted],
            n_folds=n_folds,
            seed=seed,
        )
        for y, Y in Ys.items():
            true = Y.cpu().numpy()[fitted]
            pred = preds[y][fitted]
            rec = dict(
                c.conv_bootstrap_r2(pred, true, conv[fitted], n_boot=n_boot, seed=seed + 400 + si)
            )
            vals = [
                1.0 - a["res"] / a["tot"] if a["tot"] > 1e-12 else float("nan") for a in null_acc[y]
            ]
            vals = [v for v in vals if v == v]
            rec["shuffle_null"] = {
                "n_draws": int(null_draws),
                "null_mean": float(np.mean(vals)) if vals else float("nan"),
                "null_p975": float(np.quantile(vals, 0.975)) if vals else float("nan"),
                "observed_above_null_p975": (
                    bool(rec["r2"] > np.quantile(vals, 0.975)) if vals else None
                ),
            }
            rec["knn"] = {
                m: mb.knn_retrieval(pred, true, ks=KNN_KS, metric=m)
                for m in ("euclidean", "cosine")
            }
            if pred.shape[1] == true.shape[1]:
                Xn = X.cpu().numpy()[fitted]
                ib = np.zeros_like(true)
                ib_fit = np.zeros(len(true), bool)
                sub_folds = folds[fitted]
                for k in range(n_folds):
                    tr, te = sub_folds != k, sub_folds == k
                    if te.sum() == 0 or tr.sum() < 3 or Xn.shape[1] != true.shape[1]:
                        continue
                    ib[te] = mb.identity_bias_predict(Xn[tr], true[tr], Xn[te])
                    ib_fit[te] = True
                if ib_fit.any():
                    rec["identity_bias"] = c.conv_bootstrap_r2(
                        ib[ib_fit],
                        true[ib_fit],
                        conv[fitted][ib_fit],
                        n_boot=n_boot,
                        seed=seed + 700 + si,
                    )
            rec["companions"] = comp[y]
            # PRIMARY within-R2 read for this grid point (`r2` above stays as the
            # artifact-bearing ambient-GCV continuity companion).
            rec["r2_reduced_basis"] = comp[y]["reduced_basis"]["r2_heldout"]
            out[f"{slot}|{y}"] = rec
    return {
        "store": store_key,
        "layer": LAYER,
        "n_folds": int(n_folds),
        "n_rows": int(keep.sum()),
        "n_groups": int(len(np.unique(conv))),
        "null_draws": int(null_draws),
        "x_slots": list(bc.BND_SLOT_ORDER),
        "y_targets": list(bc.Y_SPAN_ORDER),
        "x_grid_slots": list(bc.X_GRID_SLOTS),
        "factorization": "one eigh(Gram) per (X slot, fold), shared across both Y "
        "targets and every shuffle-null draw",
        "cells": out,
    }


def reparam_ladder(src: dict, tgt: dict, *, n_folds: int, seed: int, null_draws: int) -> dict:
    """Rungs of the A o M o B reparameterization chain, BOTH directions.

    Per fold ONE eigh(Gram) per source (`ma._ridge_prep`) is computed and reused
    by every rung and every null draw; the lambda grid is applied as diagonal
    filter rescalings of that factorization, and the shuffle-fit null permutes
    the TRAIN ANSWERS against the SAME cached prep. Rungs, with A = the
    context-side map (target ctx -> source ctx), M = the source regime's own
    operator, B = the answer-side map (source answer -> target answer):

      ceiling      M_tgt(X_tgt)          the target's own within-regime read
      direct       M_src(X_tgt)          source operator, no reparameterization
      ctx_reparam  M_src(A(X_tgt))       context reparameterization only
      ans_reparam  B(M_src(X_tgt))       answer reparameterization only
      amb          B(M_src(A(X_tgt)))    both

    ``src``/``tgt`` are {X, Y, conv_ids} on the SAME conversation set in the
    SAME row order (asserted).
    """
    import issue825_map_alignment as ma

    conv = np.asarray([str(x) for x in tgt["conv_ids"]])
    assert np.array_equal(conv, np.asarray([str(x) for x in src["conv_ids"]])), (
        "reparam ladder needs src/tgt rows aligned by conversation"
    )
    # The ladder is a HEADLINE-LAYER read: slice the (n, L, D) stores to
    # (n, D) at LAYER before any prep — ma._ridge_prep takes a 2-D source.
    for name, side in (("src", src), ("tgt", tgt)):
        for key in ("X", "Y"):
            assert side[key].ndim == 3 and side[key].shape[1] > LAYER, (
                f"{name}[{key}] has shape {side[key].shape} — expected (n, n_layers, D) "
                f"with n_layers > {LAYER}"
            )
    Xs, Ys = _t(src["X"][:, LAYER, :]), _t(src["Y"][:, LAYER, :])
    Xt, Yt = _t(tgt["X"][:, LAYER, :]), _t(tgt["Y"][:, LAYER, :])
    folds = fc._cv_folds(conv, n_folds, seed)
    rng = np.random.default_rng(seed + 7)

    rungs = ("ceiling", "direct", "ctx_reparam", "ans_reparam", "amb")
    acc = {r: {"res": 0.0, "tot": 0.0} for r in rungs}
    null_acc = {r: [{"res": 0.0, "tot": 0.0} for _ in range(null_draws)] for r in rungs[1:]}
    n_used = 0
    for k in range(n_folds):
        tr, te = folds != k, folds == k
        if te.sum() == 0 or tr.sum() < 3:
            continue
        n_used += int(te.sum())
        import torch

        trt, tet = torch.as_tensor(tr), torch.as_tensor(te)
        # ONE factorization per (source, fold), reused by every rung + null draw.
        p_xs = ma._ridge_prep(Xs[trt])
        p_xt = ma._ridge_prep(Xt[trt])
        p_ys = ma._ridge_prep(Ys[trt])
        y_true = Yt[tet]
        mu = Yt[trt].mean(0)
        tot = float(((y_true - mu) ** 2).sum())
        # A: target ctx -> source ctx (fit on train, applied to held-out target).
        xs_hat = ma._ridge_predict(p_xt, Xs[trt], Xt[tet])

        def _pred(rung: str, ys_train):
            if rung == "ceiling":
                return ma._ridge_predict(p_xt, Yt[trt], Xt[tet])
            x_in = xs_hat if rung in ("ctx_reparam", "amb") else Xt[tet]
            ys_hat = ma._ridge_predict(p_xs, ys_train, x_in)
            if rung in ("ans_reparam", "amb"):
                # B: source answer -> target answer.
                return ma._ridge_predict(p_ys, Yt[trt], ys_hat)
            return ys_hat

        for rung in rungs:
            pred = _pred(rung, Ys[trt])
            acc[rung]["res"] += float(((y_true - pred) ** 2).sum())
            acc[rung]["tot"] += tot
        for d in range(null_draws):
            perm = rng.permutation(int(tr.sum()))
            ys_shuf = Ys[trt][torch.as_tensor(perm)]
            for rung in rungs[1:]:
                pred = _pred(rung, ys_shuf)
                null_acc[rung][d]["res"] += float(((y_true - pred) ** 2).sum())
                null_acc[rung][d]["tot"] += tot

    def _r2(a):
        return 1.0 - a["res"] / a["tot"] if a["tot"] > 1e-12 else float("nan")

    out = {
        "layer": LAYER,
        "n_folds": int(n_folds),
        "n_rows_scored": n_used,
        "n_groups": int(len(np.unique(conv))),
        "null_draws": int(null_draws),
        "r2": {r: _r2(acc[r]) for r in rungs},
    }
    ceiling = out["r2"]["ceiling"]
    out["deficit_vs_ceiling"] = {r: (out["r2"][r] - ceiling) for r in rungs if r != "ceiling"}
    out["shuffle_fit_null"] = {}
    for r, draws in null_acc.items():
        vals = [_r2(a) for a in draws]
        vals = [v for v in vals if v == v]
        out["shuffle_fit_null"][r] = {
            "null_mean": float(np.mean(vals)) if vals else float("nan"),
            "null_p975": float(np.quantile(vals, 0.975)) if vals else float("nan"),
            "observed_above_null_p975": (
                bool(out["r2"][r] > np.quantile(vals, 0.975)) if vals else None
            ),
        }
    return out


def arm_xy(bundle: dict, cell: dict, keep_ids: list[str]) -> dict:
    """(X, Y, conv_ids) for one cell restricted to ``keep_ids``, conv-sorted.

    Sorting by conv_id on both sides of a reparam pair makes the rows align by
    construction (one row per conversation in every store used here).
    """
    xy = fc._cell_xy(bundle, cell)
    conv = np.asarray([str(x) for x in xy["conv_ids"]])
    keep = np.isin(conv, np.asarray(sorted(set(keep_ids))))
    order = np.argsort(conv[keep], kind="stable")
    return {
        "X": xy["X"][keep][order],
        "Y": xy["Y"][keep][order],
        "conv_ids": conv[keep][order],
    }


# ---------------------------------------------------------------------------
# Leg 5 — verdict lattice
# ---------------------------------------------------------------------------
def _read_committed(path: Path, layer: int) -> dict | None:
    """{r2, ci_lo, ci_hi, n} at ``layer`` from a committed cells JSON."""
    if not path.exists():
        return None
    d = json.loads(path.read_text())
    out = {
        "r2": float(d["r2_per_layer_obs"][layer]),
        "n": d["metadata"].get("n"),
        "source": str(path),
    }
    boot = d.get("r2_bootstrap_ci_frozen_layers_conv", {}).get(str(layer))
    if boot:
        out.update({"ci_lo": boot["ci_lo"], "ci_hi": boot["ci_hi"], "n_groups": boot["n_groups"]})
    return out


def _cis_disjoint(a: dict | None, b: dict | None) -> bool | None:
    """Descriptive non-overlap of two INDEPENDENT CIs (never a test)."""
    if not a or not b or "ci_lo" not in a or "ci_lo" not in b:
        return None
    return bool(a["ci_hi"] < b["ci_lo"] or b["ci_hi"] < a["ci_lo"])


def build_verdict(
    arm: str,
    cell_summary: dict[str, dict],
    paired: dict,
    reparam: dict,
    grid: dict,
    comparator_grids: dict[str, dict],
    *,
    n_kept: int,
    n_intersection: int | None,
    v1_grid: dict | None = None,
) -> dict:
    """Per-arm verdict record: the headline read against every reference."""
    slug = bg.ARM_SLUG[arm]
    own = cell_summary.get(grid_cell_id(arm, bc.HEADLINE_SLOT, bc.Y_MEAN), {})
    anchor = _read_committed(V1_ANCHOR_FILES["context"], LAYER)
    chat_anchor = _read_committed(V1_MATCHED_CHAT_FILES["context"], LAYER)
    chat_cell = cell_summary.get(f"R_{bg.MODEL_KEY}_r1_chat_bnd_{slug}_context", {})
    nt_cell = cell_summary.get(f"R_{bg.MODEL_KEY}_r2_no_template_bnd_{slug}_context", {})
    verdict = {
        "arm": arm,
        "arm_isolates": bg.ARM_README[arm],
        "headline_slot": bc.HEADLINE_SLOT,
        "layer": LAYER,
        "n_kept_stories": n_kept,
        "n_intersection_with_v1": n_intersection,
        # PRIMARY within-R2 read = the REDUCED-BASIS companion. The ambient GCV
        # value is retained under `*_ambient_gcv_continuity` for continuity with
        # the published anchor and is ARTIFACT-BEARING at n_train < d (GCV picks
        # the lambda-grid floor; see PROBE_REFERENCE) — never read it as the
        # information content of the map.
        "primary_read": "reduced_basis",
        "primary_read_note": (
            "reduced-basis within-R2 is the primary read; ambient GCV is the "
            "artifact-bearing continuity companion (n_train<d floor-lambda selection)"
        ),
        "slots": {
            cell_summary[cid]["cell"]["slot"]: {
                "r2_reduced_basis_primary": cell_summary[cid].get("r2_reduced_basis"),
                "r2_ambient_gcv_continuity": cell_summary[cid].get("r2"),
                **{k: cell_summary[cid].get(k) for k in ("ci", "null_p975", "skill_over_mean")},
                "companions": cell_summary[cid].get("companions"),
            }
            for cid in (grid_cell_id(arm, s, bc.Y_MEAN) for s in bc.BND_SLOT_ORDER)
            if cid in cell_summary and "cell" in cell_summary[cid]
        },
        "headline": {
            "r2_reduced_basis_primary": own.get("r2_reduced_basis"),
            "r2_ambient_gcv_continuity": own.get("r2"),
            **{k: own.get(k) for k in ("ci", "null_p975", "skill_over_mean")},
            "companions": own.get("companions"),
        },
        "baselines_headline_slot": own.get("baselines"),
        "vs_v1_anchor_committed": {
            "anchor": anchor,
            "anchor_doc_crosscheck": V1_ANCHOR_DOC["context"],
            "anchor_is_artifact_bearing": True,
            "anchor_note": (
                "the committed V1 anchor is an AMBIENT-GCV read at n_train<d — an "
                "estimation artifact, not the map's information content; kept as a "
                "parity check that this round reproduces the published pipeline, "
                "NOT as a science reference (PROBE_REFERENCE)"
            ),
            "delta_point_ambient_gcv": (
                (own.get("r2") - anchor["r2"]) if (own.get("r2") is not None and anchor) else None
            ),
            "independent_cis_disjoint": _cis_disjoint(own.get("ci"), anchor),
            "note": "committed V1 read on the FULL V1 kept set — different rows, so this is "
            "a descriptive comparison; the PAIRED read is vs_v1_matched below",
        },
        "vs_matched_chat": {
            "chat_same_rows": {k: chat_cell.get(k) for k in ("r2", "ci")},
            "v1_matched_chat_committed": chat_anchor,
            "v1_matched_chat_doc_crosscheck": V1_MATCHED_CHAT_DOC["context"],
            "no_template_same_rows": {k: nt_cell.get(k) for k in ("r2", "ci")},
        },
        "paired_deltas": paired,
        # Pre-registered contrast map (arms that have one; reads only — see
        # PRE_REGISTERED_CONTRASTS). Absent-by-default keeps every other arm's
        # verdict shape unchanged.
        **(
            {"pre_registered_contrasts": PRE_REGISTERED_CONTRASTS[arm]}
            if arm in PRE_REGISTERED_CONTRASTS
            else {}
        ),
        "reparam_story_vs_chat": reparam,
        # The consolidated X x Y measurement grid (addendum): the arm's own grid,
        # the same grid on each round-own comparator store, AND the same grid on
        # the re-captured V1 anchor — so every (read position x target) pair is
        # comparable arm-vs-comparator AND arm-vs-boundary-present-anchor. The
        # V1 row is what makes the grid an ABLATION rather than a description:
        # a collapse that disappears at (x_clean, y_boundary) for V1 too was a
        # read-position artifact, not the boundary.
        "xy_grid": {
            "x_clean_slot": bc.X_CLEAN_SLOT,
            "ctx_straddle_slot": bc.X_STRADDLE_SLOT,
            "y_targets": list(bc.Y_SPAN_ORDER),
            "transition_appended_verbatim": (
                bc.TRANSITION[arm]["closer"] + bc.TRANSITION[arm]["suffix"]
            ),
            "transition_read_anchor": bc.TRANSITION[arm]["read_anchor"],
            "arm": grid,
            "comparators": comparator_grids,
            "v1_anchor": (
                v1_grid
                if v1_grid is not None
                else {"skipped": "V1 X x Y store not captured (capture --arm v1)"}
            ),
        },
    }
    return verdict


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def _import_check() -> None:
    """Resolve every deferred import on the REAL code path, then exit 0."""
    import inspect

    import torch  # noqa: F401

    import issue825_map_alignment as ma  # noqa: F401

    assert inspect.getsource(reparam_ladder)
    assert callable(ma._ridge_prep) and callable(ma._ridge_predict)
    print("[import-check] OK: torch + issue825_map_alignment symbols resolved", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--phase", choices=("all", "cells", "grid", "reparam", "verdict"), default="all"
    )
    ap.add_argument("--arms", default=",".join(bg.ARM_SLUG[a] for a in bg.GEN_ARMS))
    ap.add_argument(
        "--no-arms",
        action="store_true",
        help="COMPANION lattice: run the comparator / V1-grid / on-policy paired cells "
        "with ZERO ablation arms. Required when the turnstore has no v2-v5 stores — the "
        "default --arms would demand their sidecars. Ignores --arms entirely.",
    )
    ap.add_argument(
        "--reparam-arms",
        default=None,
        help="arms whose story<->chat reparam ladder runs (default: every --arms "
        "member that is in DEFAULT_REPARAM_ARMS; an EXPLICIT arm outside --arms "
        "fails loud rather than silently doing nothing)",
    )
    ap.add_argument("--turnstore-dir", type=Path, default=c.TURNSTORE_DIR)
    ap.add_argument(
        "--stage-v1",
        action="store_true",
        help="stage the PARENT V1 turnstore (instruct_stories_paired_s, ~5.3 GB) from "
        "the parent HF prefix BEFORE fitting, so the paired arm-vs-V1 bootstrap "
        "cannot silently record `skipped` (it is NOT one of prefetch_reuse's stems)",
    )
    ap.add_argument("--stories-dir", type=Path, default=c.STORIES_DIR)
    ap.add_argument("--out-dir", type=Path, default=c.EVAL_DIR / "story_boundary_ablation")
    ap.add_argument("--preds-dir", type=Path, default=c.PREDS_CACHE_DIR / "boundary_ablation")
    ap.add_argument("--n-folds", type=int, default=fc.N_FOLDS)
    ap.add_argument("--seed", type=int, default=fc.FIT_SEED)
    ap.add_argument("--null-draws", type=int, default=N_NULL_DRAWS)
    ap.add_argument("--n-boot", type=int, default=c.N_BOOTSTRAP)
    ap.add_argument("--smoke", action="store_true", help="tiny nulls/boot; degenerate-fold skips")
    ap.add_argument(
        "--no-resume",
        action="store_true",
        help="refit every cell even when a regime-matching persisted result exists "
        "(default: resume, so a mid-phase kill never forfeits completed cells)",
    )
    ap.add_argument(
        "--cell-shard",
        default=os.environ.get(CELL_SHARD_ENV, ""),
        help=(
            f"'i/N' — fit only this shard's cells (env {CELL_SHARD_ENV}). Cells are "
            "embarrassingly parallel ACROSS instances; the partition is a sha256 of "
            "cell_id, so every process agrees on ownership even when presence-gating "
            "makes their enumerated lists differ. --phase cells ONLY: the whole-lattice "
            "phases need every cell's output. Unset = the unsharded default path."
        ),
    )
    ap.add_argument(
        "--import-check",
        action="store_true",
        help="resolve every deferred import on the real code path and exit 0",
    )
    args = ap.parse_args()

    if args.import_check:
        _import_check()
        return

    # Validate the shard/phase pairing BEFORE any store access: a sharded process
    # holds one slice, so a whole-lattice phase would compute the grid/verdict over
    # a fraction and report it complete. Checked here rather than after enumeration
    # so it costs nothing and fails on a box with no stores staged.
    shard = parse_cell_shard(args.cell_shard)
    assert not (shard is not None and args.phase in WHOLE_LATTICE_PHASES), (
        f"--phase {args.phase} consumes EVERY cell's output, but --cell-shard "
        f"{args.cell_shard} holds only this shard's slice — it would compute the lattice "
        "over a fraction and report it complete. Run the sharded instances with "
        "--phase cells, then ONE UNSHARDED --phase all over the staged union."
    )

    bg.assert_round_env()
    # --no-arms runs a COMPANION lattice: comparator + V1-grid + on-policy paired
    # cells only, with ZERO ablation arms. Those cells come from the comparator /
    # op stores and never touch `arms`, so every arms-keyed consumer below
    # (arm_convs, the per-arm cell loop, paired_by_arm, reparam_by_arm, verdicts)
    # is a comprehension or loop that yields empty — but the non-empty assert
    # fired first, and with the DEFAULT --arms the enumeration also demanded
    # v2/v3/v4/v5 sidecars a companion turnstore deliberately does not have.
    #
    # Kept as an EXPLICIT flag rather than just relaxing the assert to accept an
    # empty --arms: an empty string from a typo or an unset env var would then
    # silently produce an arms-free lattice instead of failing loudly.
    arms = [] if args.no_arms else [bg.SLUG_ARM.get(a, a) for a in args.arms.split(",") if a]
    assert args.no_arms or arms, (
        "--arms parsed empty; pass --no-arms to run the companion (comparator + "
        "on-policy paired) lattice deliberately"
    )
    assert set(arms) <= set(bg.GEN_ARMS), arms
    if args.reparam_arms is None:
        # Default INTERSECTS --arms, so narrowing --arms never trips the guard.
        reparam_arms = [a for a in arms if a in DEFAULT_REPARAM_ARMS]
    else:
        reparam_arms = [bg.SLUG_ARM.get(a, a) for a in args.reparam_arms.split(",") if a]
        assert set(reparam_arms) <= set(arms), (
            f"--reparam-arms {sorted(reparam_arms)} names arms outside --arms "
            f"{sorted(arms)} — a ladder needs its arm's store registered"
        )
    null_draws = SMOKE_NULL_DRAWS if args.smoke else args.null_draws
    n_boot = SMOKE_BOOT if args.smoke else args.n_boot
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Kept conversation sets (from the arm stores, so the fit rows and the
    # allowlists can never drift) + the V1 intersection when V1 is staged.
    arm_convs = {
        arm: store_conv_ids(args.turnstore_dir, bg.MODEL_KEY, bc.format_key(arm)) for arm in arms
    }
    if args.stage_v1:
        stage_v1_turnstore(args.turnstore_dir)
    v1_available = store_present(args.turnstore_dir, bg.MODEL_KEY, V1_STEM_FORMAT)
    assert v1_available or not args.stage_v1, (
        "--stage-v1 ran but the V1 store is still absent — staging is fail-loud, so "
        "this means the stem/track convention drifted"
    )
    v1_convs = (
        store_conv_ids(args.turnstore_dir, bg.MODEL_KEY, V1_STEM_FORMAT) if v1_available else []
    )
    comparators_available = {
        label: store_present(args.turnstore_dir, bg.MODEL_KEY, fmt)
        for label, fmt in COMPARATOR_FORMAT.items()
    }
    # Round-own X x Y comparator stores (capture --comparator chat|no_template).
    bnd_comparators_available = {
        key: store_present(args.turnstore_dir, bg.MODEL_KEY, bc.format_key(key))
        for key in BND_COMPARATORS
    }
    # The round's OWN re-capture of the V1 anchor at the X x Y shape (capture
    # --arm v1). DISTINCT from `v1_available` above, which is the PARENT 2-slot
    # store used for the paired arm-vs-V1 bootstrap: this one carries the same
    # 5 slots x 2 Y targets as every arm, so the grid gets a boundary-PRESENT row.
    v1_grid_available = store_present(args.turnstore_dir, bg.MODEL_KEY, bc.format_key(bc.V1_ARM))
    print(
        f"[fits] arms={[bg.ARM_SLUG[a] for a in arms]} "
        f"kept={{{', '.join(f'{bg.ARM_SLUG[a]}:{len(v)}' for a, v in arm_convs.items())}}} "
        f"v1_store={v1_available} v1_xy_store={v1_grid_available} "
        f"parent_comparators={comparators_available} "
        f"xy_comparators={bnd_comparators_available}",
        flush=True,
    )

    # Cell registry + per-cell allowlists.
    cells: list[dict] = []
    allow: dict[str, list[str]] = {}
    inter: dict[str, list[str]] = {}
    for arm in arms:
        cells += arm_cells(arm)
        for label, present in comparators_available.items():
            if not present:
                print(f"[fits] comparator store {label} absent — matched refits skipped")
                continue
            for cell in comparator_cells(arm, label):
                cells.append(cell)
                allow[cell["cell_id"]] = sorted(arm_convs[arm])
        if v1_available:
            ids = sorted(set(arm_convs[arm]) & set(v1_convs))
            inter[arm] = ids
            if len(ids) >= args.n_folds:
                for cell in v1_matched_cells(arm) + arm_matched_cells(arm):
                    cells.append(cell)
                    allow[cell["cell_id"]] = ids
            else:
                print(
                    f"[fits] {arm}: V1 intersection {len(ids)} < n_folds {args.n_folds} — "
                    "paired V1 refits skipped",
                    flush=True,
                )
    # The X x Y grid on the round-own comparator stores: fit ONCE per comparator
    # (not per arm) over the union of the arms' kept conversations, which is
    # exactly the row set those stores were captured on.
    for key, present in bnd_comparators_available.items():
        if not present:
            print(f"[fits] X x Y comparator store {key} absent — grid cells skipped", flush=True)
            continue
        cells += grid_cells(key)
    # The V1 anchor's own X x Y row (capture --arm v1), same treatment.
    if v1_grid_available:
        cells += grid_cells(bc.V1_ARM)
    else:
        print(
            f"[fits] X x Y V1 store {bc.format_key(bc.V1_ARM)} absent — the grid has NO "
            "boundary-present anchor row (run capture --arm v1)",
            flush=True,
        )

    # ON-POLICY paired arm of the SAME lattice (the on-policy-vs-injected
    # program): every registered on-policy twin whose store is on disk joins here
    # with identical slots / Y targets / conv_id space, so injected-vs-onpolicy is
    # reported as a matched pair rather than a separate ad-hoc run. Presence-gated,
    # so this is a no-op until the on-policy captures land.
    op_cells, op_present = onpolicy_paired_cells(
        args.turnstore_dir, [*bnd_comparators_available, bc.V1_ARM]
    )
    cells += op_cells
    if op_present:
        landed = sorted(k for k, ok in op_present.items() if ok)
        missing = sorted(k for k, ok in op_present.items() if not ok)
        print(
            f"[fits] on-policy paired arm: {len(op_cells)} cells from {landed or 'none'}"
            + (f"; absent (skipped): {missing}" if missing else ""),
            flush=True,
        )

    # An EMPTY lattice is a configuration error, not a valid run. Without this the
    # driver writes an empty cell_summary.json / xy_grid.json and prints its normal
    # done line having fit nothing — observed with --no-arms against a turnstore
    # path that had no stores. The companion run is exactly where that is
    # plausible (fresh _fulln dirs, stores staged separately), and the arms-free
    # lattice removed the incidental non-emptiness the arm cells used to provide.
    assert cells, (
        f"enumeration produced ZERO cells from turnstore {args.turnstore_dir} — nothing "
        "to fit. With --no-arms the lattice is comparator + V1-grid + on-policy paired "
        "cells only, so an unstaged or mis-pointed turnstore yields an empty run that "
        "would otherwise report success."
    )

    # Shard AFTER full enumeration (so every process partitions the same registry)
    # and BEFORE bundle loading (so a shard pays only for the stores its own cells
    # read — loading all bundles per shard would spend the wall-clock the shard
    # exists to save).
    if shard is not None:
        before = len(cells)
        cells = apply_cell_shard(cells, shard)
        print(
            f"[fits] cell shard {shard[0]}/{shard[1]}: {len(cells)}/{before} cells "
            f"-> {sorted(cl['cell_id'] for cl in cells)}",
            flush=True,
        )
        assert cells, (
            f"shard {shard[0]}/{shard[1]} selected 0 of {before} cells — more shards than "
            "cells, or an enumeration that produced nothing"
        )

    bundles: dict[tuple[str, str], dict] = {}
    for cell in cells:
        key = (cell["model_key"], cell["format_key"])
        if key in bundles:
            continue
        expect = len(bc.BND_SLOT_ORDER) if cell["format_key"].startswith("bnd_") else 2
        bundles[key] = load_bundle(args.turnstore_dir, key[0], key[1], expect)

    cell_summary: dict[str, dict] = {}
    if args.phase in ("all", "cells"):
        cell_summary = run_cells(
            cells,
            bundles,
            args.out_dir,
            args.preds_dir,
            allow,
            n_folds=args.n_folds,
            seed=args.seed,
            null_draws=null_draws,
            n_boot=n_boot,
            smoke=args.smoke,
            resume=not args.no_resume,
        )
        # A shard holds a SLICE, so it must not write the filename the whole-lattice
        # phases read: a partial cell_summary.json staged onto the union box would
        # satisfy `--phase grid`'s existence assert and silently supply a quarter
        # of the lattice. Shard-scoped name => that path fails loud instead.
        summary_name = (
            "cell_summary.json"
            if shard is None
            else f"cell_summary.shard{shard[0]}of{shard[1]}.json"
        )
        c.write_json(
            args.out_dir / summary_name,
            {
                "metadata": c.metadata(
                    args.seed, len(cell_summary), "scripts/issue1345_boundary_ablation_fits.py"
                ),
                "round": bg.ROUND_VARIANT,
                "layer": LAYER,
                "null_draws": null_draws,
                "n_boot": n_boot,
                "arm_readme": {a: bg.ARM_README[a] for a in bg.ALL_ARMS},
                "cells": cell_summary,
            },
        )

    # Paired deltas (shared conversation set -> one counts matrix per draw).
    paired_by_arm: dict[str, dict] = {}
    if args.phase in ("all", "cells", "verdict"):
        for arm in arms:
            slug = bg.ARM_SLUG[arm]
            reads: dict[str, tuple] = {}
            own = _load_preds(args.preds_dir, grid_cell_id(arm, bc.HEADLINE_SLOT, bc.Y_MEAN))
            chat = _load_preds(args.preds_dir, f"R_{bg.MODEL_KEY}_r1_chat_bnd_{slug}_context")
            nt = _load_preds(args.preds_dir, f"R_{bg.MODEL_KEY}_r2_no_template_bnd_{slug}_context")
            pairs: list[tuple[str, str]] = []
            if own and chat and np.array_equal(np.unique(own[2]), np.unique(chat[2])):
                reads["arm"], reads["chat"] = own, chat
                pairs.append(("arm", "chat"))
                if nt and np.array_equal(np.unique(own[2]), np.unique(nt[2])):
                    reads["no_template"] = nt
                    pairs.append(("arm", "no_template"))
            armm = _load_preds(args.preds_dir, f"R_{bg.MODEL_KEY}_bndm_{slug}_context")
            v1m = _load_preds(args.preds_dir, f"R_{bg.MODEL_KEY}_v1_bnd_{slug}_context")
            block: dict = {}
            if pairs:
                block["vs_comparators"] = paired_delta(
                    reads, pairs, n_boot=n_boot, seed=args.seed + 11
                )
            else:
                block["vs_comparators"] = {
                    "skipped": "comparator preds absent or on a different conversation set"
                }
            if armm and v1m and np.array_equal(np.unique(armm[2]), np.unique(v1m[2])):
                block["vs_v1_matched"] = paired_delta(
                    {"arm": armm, "v1": v1m},
                    [("arm", "v1")],
                    n_boot=n_boot,
                    seed=args.seed + 13,
                )
            else:
                block["vs_v1_matched"] = {
                    "skipped": "V1 store not staged, or the intersection refits were skipped"
                }
            paired_by_arm[arm] = block

    # X x Y grid with the X-side factorization SHARED across Y (addendum).
    grid_by_store: dict[str, dict] = {}
    if args.phase in ("all", "cells", "grid", "verdict"):
        grid_stores = (
            list(arms)
            + [k for k, ok in bnd_comparators_available.items() if ok]
            + ([bc.V1_ARM] if v1_grid_available else [])
        )
        for key in grid_stores:
            bkey = (bg.MODEL_KEY, bc.format_key(key))
            if bkey not in bundles:
                grid_by_store[key] = {"skipped": f"store {bc.format_key(key)} not staged"}
                continue
            ids = arm_convs.get(key)
            if args.smoke:
                probe = ids if ids is not None else store_conv_ids(args.turnstore_dir, *bkey)
                reason = degenerate_fold_reason(
                    np.asarray(probe), n_folds=args.n_folds, seed=args.seed
                )
                if reason:
                    grid_by_store[key] = {"skipped": f"smoke: {reason}"}
                    print(f"[grid][smoke] SKIP {key}: {reason}", flush=True)
                    continue
            grid_by_store[key] = xy_grid(
                bundles[bkey],
                key,
                allow=ids,
                n_folds=args.n_folds,
                seed=args.seed,
                null_draws=null_draws,
                n_boot=n_boot,
            )
            got = grid_by_store[key].get("cells", {})
            head = got.get(f"{bc.X_CLEAN_SLOT}|{bc.Y_MEAN}", {})
            print(
                f"[grid] {key}: {len(got)} X x Y cells (x_clean|y_mean R2={head.get('r2')})",
                flush=True,
            )
        c.write_json(
            args.out_dir / "xy_grid.json",
            {
                "metadata": c.metadata(
                    args.seed, len(grid_by_store), "scripts/issue1345_boundary_ablation_fits.py"
                ),
                "round": bg.ROUND_VARIANT,
                "layer": LAYER,
                "x_slots": list(bc.BND_SLOT_ORDER),
                "x_clean_slot": bc.X_CLEAN_SLOT,
                "ctx_straddle_slot": bc.X_STRADDLE_SLOT,
                "y_targets": list(bc.Y_SPAN_ORDER),
                "transition_suffixes_verbatim": {
                    k: {
                        "appended_verbatim": v["closer"] + v["suffix"],
                        "read_anchor": v["read_anchor"],
                    }
                    for k, v in bc.TRANSITION.items()
                },
                "stores": grid_by_store,
            },
        )

    # Reparameterization ladder, both directions, story arm <-> chat.
    reparam_by_arm: dict[str, dict] = {}
    if args.phase in ("all", "reparam", "verdict"):
        chat_key = (bg.MODEL_KEY, COMPARATOR_FORMAT["r1_chat"])
        for arm in reparam_arms:
            if chat_key not in bundles:
                reparam_by_arm[arm] = {"skipped": "chat comparator store not staged"}
                continue
            ids = sorted(set(arm_convs[arm]) & set(store_conv_ids(args.turnstore_dir, *chat_key)))
            if len(ids) < args.n_folds:
                reparam_by_arm[arm] = {
                    "skipped": f"shared conversation set {len(ids)} < n_folds {args.n_folds}"
                }
                continue
            story_cell = next(
                cl
                for cl in arm_cells(arm)
                if cl["slot"] == bc.HEADLINE_SLOT and cl["y_target"] == bc.Y_MEAN
            )
            chat_cell = next(
                cl for cl in comparator_cells(arm, "r1_chat") if cl["slot"] == "context"
            )
            story = arm_xy(bundles[(bg.MODEL_KEY, bc.format_key(arm))], story_cell, ids)
            chat = arm_xy(bundles[chat_key], chat_cell, ids)
            if args.smoke:
                reason = degenerate_fold_reason(
                    story["conv_ids"], n_folds=args.n_folds, seed=args.seed
                )
                if reason:
                    reparam_by_arm[arm] = {"skipped": f"smoke: {reason}"}
                    print(f"[reparam][smoke] SKIP {arm}: {reason}", flush=True)
                    continue
            nd = SMOKE_NULL_DRAWS if args.smoke else N_REPARAM_NULL_DRAWS
            reparam_by_arm[arm] = {
                "n_shared_conversations": len(ids),
                "chat_to_story": reparam_ladder(
                    chat, story, n_folds=args.n_folds, seed=args.seed, null_draws=nd
                ),
                "story_to_chat": reparam_ladder(
                    story, chat, n_folds=args.n_folds, seed=args.seed, null_draws=nd
                ),
            }
            print(
                f"[reparam] {arm}: chat->story amb deficit "
                f"{reparam_by_arm[arm]['chat_to_story']['deficit_vs_ceiling']['amb']:+.4f}, "
                f"story->chat amb deficit "
                f"{reparam_by_arm[arm]['story_to_chat']['deficit_vs_ceiling']['amb']:+.4f}",
                flush=True,
            )
        c.write_json(
            args.out_dir / "reparam_ladder.json",
            {
                "metadata": c.metadata(
                    args.seed, len(reparam_by_arm), "scripts/issue1345_boundary_ablation_fits.py"
                ),
                "round": bg.ROUND_VARIANT,
                "layer": LAYER,
                "arms": reparam_by_arm,
            },
        )

    # Verdict lattice.
    if args.phase in ("all", "verdict"):
        if not cell_summary:
            cs_path = args.out_dir / "cell_summary.json"
            assert cs_path.exists(), f"{cs_path} missing — run --phase cells first"
            cell_summary = json.loads(cs_path.read_text())["cells"]
        verdicts = {
            arm: build_verdict(
                arm,
                cell_summary,
                paired_by_arm.get(arm, {}),
                reparam_by_arm.get(
                    arm,
                    {
                        "skipped": (
                            "arm outside --reparam-arms scope"
                            if arm not in reparam_arms
                            else "reparam phase not run"
                        )
                    },
                ),
                grid_by_store.get(arm, {"skipped": "grid phase not run"}),
                {k: grid_by_store[k] for k in BND_COMPARATORS if k in grid_by_store},
                n_kept=len(arm_convs[arm]),
                n_intersection=(len(inter[arm]) if arm in inter else None),
                v1_grid=grid_by_store.get(bc.V1_ARM),
            )
            for arm in arms
        }
        c.write_json(
            args.out_dir / "verdict_lattice.json",
            {
                "metadata": c.metadata(
                    args.seed, len(verdicts), "scripts/issue1345_boundary_ablation_fits.py"
                ),
                "round": bg.ROUND_VARIANT,
                "layer": LAYER,
                "headline_slot": bc.HEADLINE_SLOT,
                "slot_order": list(bc.BND_SLOT_ORDER),
                "y_span_order": list(bc.Y_SPAN_ORDER),
                "x_grid_slots": list(bc.X_GRID_SLOTS),
                "n_folds": args.n_folds,
                "seed": args.seed,
                "null_draws": null_draws,
                "n_boot": n_boot,
                "smoke": bool(args.smoke),
                "arm_readme": {a: bg.ARM_README[a] for a in bg.ALL_ARMS},
                "v1_anchor_files": {k: str(v) for k, v in V1_ANCHOR_FILES.items()},
                "arms": verdicts,
            },
        )
        for arm, v in verdicts.items():
            # The headline carries the two NAMED reads (`r2_reduced_basis_primary`
            # / `r2_ambient_gcv_continuity`) — a bare `r2` key was renamed away
            # when the well-posedness companions landed, so print both names.
            anchor = v["vs_v1_anchor_committed"]["anchor"]
            print(
                f"[verdict] {bg.ARM_SLUG[arm]} L{LAYER} {bc.HEADLINE_SLOT} "
                f"R2_reduced={v['headline'].get('r2_reduced_basis_primary')} "
                f"R2_ambient_gcv={v['headline'].get('r2_ambient_gcv_continuity')} "
                f"vs V1 anchor(ambient) {anchor and anchor['r2']}",
                flush=True,
            )
    print(f"[done] boundary-ablation fits -> {args.out_dir}", flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(0)


if __name__ == "__main__":
    main()
