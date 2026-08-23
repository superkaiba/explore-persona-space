"""Shared P6 machinery for issue #2378 (fits / ladder / pool / retrieval drivers).

Consumes the unit-2 capture store (``scripts/issue2378_capture.py`` contract):
  ``<store_root>/<cell>__part{NNNN}__L{layer}.npz``  bf16-as-uint16 ``v_C/v_A/v_P``
      (n, 5120) + ``row_ids`` + ``meta`` (json string with ``encoding``)
  ``<store_root>/<cell>__part{NNNN}__rows.json``     aligned per-row provenance
      (story rows carry ``final_seed_id`` etc. from STORY_PROV_KEYS; chat/plain/
      user rows carry ``conv_id``)
Production tags are the bare cell slugs; fresh-draw tags
(``<cell>__fresh_d<seed>__part...``) never match the production globs here.

Fold map (plan §4.4): per-cell K=5 grouped folds, seed 137. chat/plain/user:
conversation-grouped (rows ARE conversations; the two user arms share the SAME
assignment on the pair-complete intersection cohort — §4.2b). Story cells: TWO
registered structures — PRIMARY family-held-out fold grouped by
``final_seed_id`` (greedy size-balanced so the per-fold n_train floor binds as
rarely as the data allows) + scene-grain companion. Equalize-down at
N_eq = min(kept over surviving cells), N_eq >= 6,500 enforced; a USER-arm
cohort below the floor is excluded from the min and reported loudly, never a
stop (plan §7 G2b: user cells are outside the binding predicate).

Floors are FUNCTION PARAMETERS with production defaults; probe phases pass
tiny values explicitly in code. There is deliberately NO CLI floor override
(plan "must-ask": relaxing the 6,500 floor).
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
for _p in (str(_SCRIPT_DIR), str(_REPO_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Thread caps freeze at first BLAS import — load_dotenv() BEFORE numpy (#847).
from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

import numpy as np  # noqa: E402

import issue2378_common as cm  # noqa: E402

EXPECTED_HIDDEN = 5120
K_FOLDS = 5
N_EQ_FLOOR = cm.FLOOR_KEPT  # 6,500 (plan §7 G2b)
N_TRAIN_FLOOR = 5120  # per realized fold, n_train must EXCEED this (plan G2b)
R2_ABS_FLOOR = 0.05  # max(null p95, 0.05) is the ceiling floor (plan §3)
MIN_STORYQ_SURVIVORS = 3
# v7 (plan Amendment record A, epm:progress v70): dialogue family DESCOPED —
# the amended G2b predicate is "chat + plain + >=3 story-Q"; the dialog floor
# is 0 (parameter machinery retained; n_dg >= 0 is vacuous by construction).
MIN_DIALOG_SURVIVORS = 0
VALID_DRAW_FRAC = 0.8  # ratio-bootstrap suppression threshold (plan §3)

ARMS = ("context", "prefix")
SLOT_BY_ARM = {"context": "v_C", "prefix": "v_P"}
ANSWER_SLOT = "v_A"

FOLD_MAP_NAME = "fold_map.json"
G3_GATE_NAME = "g3_gate.json"


def _log(msg: str) -> None:
    print(msg, flush=True)


def parse_survivors(spec: str | None) -> set[str] | None:
    """Parse a ``--survivors`` CSV (the dispatch threads its CURRENT G2b
    survivor set) into a set; ``None`` (flag absent — probes / manual runs)
    keeps the marker-authoritative fallback in :func:`g2b_dropped_now`."""
    if spec is None:
        return None
    return {c for c in spec.split(",") if c}


def g2b_dropped_now(fits_dir: Path, cell: str, survivors: set[str] | None) -> bool:
    """Current-run G2b drop verdict for ``cell`` (r3 reconciler blocker
    g2b-drop-marker-shadowed-by-stale-fit + its symmetric stale-marker arm).

    The ``<cell>__g2b_dropped.json`` marker is AUTHORITATIVE over any
    coexisting fit context: fits AND markers are git-harvested and
    re-materialized on every pod while G2b survivorship recomputes per
    dispatch, so a cross-run survive->drop flip leaves BOTH present — the
    fit is then stale prior-run residue, never survivor evidence. When the
    dispatch threads its CURRENT survivor set (``--survivors``), marker
    authority is keyed to THIS run: a marker on a now-surviving cell is a
    STALE prior-run marker (ignored, loud log — the drop->survive flip),
    and a missing marker on a now-dropped cell raises (the dispatch writes
    markers for its dropped cells BEFORE any consumer runs; absence is a
    wiring bug, never a survivor signal)."""
    marker = (fits_dir / f"{cell}__g2b_dropped.json").exists()
    if survivors is None:
        return marker
    if cell in survivors:
        if marker:
            _log(
                f"[g2b] {cell}: STALE prior-run drop marker ignored — "
                "cell SURVIVES this dispatch (--survivors is authoritative)"
            )
        return False
    if not marker:
        raise RuntimeError(
            f"{cell} is G2b-dropped this dispatch (--survivors) but "
            f"fits/{cell}__g2b_dropped.json is missing — the upstream drop-marker "
            "write failed (plan §7 skip-and-count)"
        )
    return True


# ---------------------------------------------------------------------------
# bf16-as-uint16 codec (numpy side; bit-exact vs the capture rig's torch codec)
# ---------------------------------------------------------------------------


def decode_bf16_np(a: np.ndarray) -> np.ndarray:
    """uint16 bf16 bit array -> float32 (exact: bf16 is the top 16 bits of f32)."""
    if a.dtype != np.uint16:
        raise TypeError(f"decode_bf16_np expects uint16, got {a.dtype}")
    return (a.astype(np.uint32) << 16).view(np.float32)


def encode_bf16_np(x: np.ndarray) -> np.ndarray:
    """float32 -> bf16-as-uint16 with round-to-nearest-even (probe/store writer)."""
    b = np.ascontiguousarray(np.asarray(x, dtype=np.float32)).view(np.uint32)
    return ((b + 0x7FFF + ((b >> 16) & 1)) >> 16).astype(np.uint16)


# ---------------------------------------------------------------------------
# Store access (production tags only)
# ---------------------------------------------------------------------------


def production_part_indices(store_root: Path, cell: str) -> list[int]:
    """Sorted part indices for a cell's PRODUCTION tag (fresh tags excluded)."""
    parts = []
    for p in store_root.glob(f"{cell}__part*__rows.json"):
        mid = p.name[len(cell) + 2 : -len("__rows.json")]
        if not mid.startswith("part"):
            continue
        parts.append(int(mid[len("part") :]))
    return sorted(parts)


def load_ledger(store_root: Path, cell: str) -> list[dict]:
    """Concatenated rows.json rows in part order (the store row order)."""
    idxs = production_part_indices(store_root, cell)
    if not idxs:
        raise RuntimeError(f"no production rows.json for cell={cell} under {store_root}")
    rows: list[dict] = []
    for ci in idxs:
        payload = json.loads(
            (store_root / f"{cell}__part{ci:04d}__rows.json").read_text(encoding="utf-8")
        )
        rows.extend(payload["rows"])
    ids = [r["row_id"] for r in rows]
    if len(set(ids)) != len(ids):
        raise RuntimeError(f"duplicate row_ids in {cell} ledger (store corrupt)")
    return rows


def load_cell_arrays(
    store_root: Path,
    cell: str,
    layer: int,
    slots: tuple[str, ...],
    *,
    row_order: list[str] | None = None,
    want_raw: tuple[str, ...] = (),
) -> dict:
    """Load one cell's production arrays at ``layer``.

    Returns {"row_ids": [...store order or row_order...],
             "arrays": {slot: (n, d) float32}, "raw": {slot: (n, d) uint16}}.
    Asserts npz/rows.json alignment per part and the bf16 encoding tag. When
    ``row_order`` is given, rows are re-indexed to EXACTLY that order (every
    requested id must exist — fail loud otherwise).
    """
    idxs = production_part_indices(store_root, cell)
    if not idxs:
        raise RuntimeError(f"no production store parts for cell={cell} under {store_root}")
    ids: list[str] = []
    chunks: dict[str, list[np.ndarray]] = {s: [] for s in set(slots) | set(want_raw)}
    for ci in idxs:
        npz_path = store_root / f"{cell}__part{ci:04d}__L{layer}.npz"
        if not npz_path.exists():
            raise RuntimeError(f"missing store part {npz_path} (layer {layer} not staged?)")
        rows_payload = json.loads(
            (store_root / f"{cell}__part{ci:04d}__rows.json").read_text(encoding="utf-8")
        )
        part_ids = [r["row_id"] for r in rows_payload["rows"]]
        with np.load(npz_path) as z:
            meta = json.loads(z["meta"].item())
            if meta.get("encoding") != "bf16_as_uint16":
                raise RuntimeError(f"unexpected encoding in {npz_path}: {meta}")
            npz_ids = [str(x) for x in z["row_ids"].tolist()]
            if npz_ids != [str(x) for x in part_ids]:
                raise RuntimeError(f"row_id misalignment npz vs rows.json at {npz_path}")
            for s in chunks:
                chunks[s].append(np.asarray(z[s]))
        ids.extend(npz_ids)
    raw = {s: np.concatenate(chunks[s], axis=0) for s in chunks}
    if row_order is not None:
        pos = {rid: i for i, rid in enumerate(ids)}
        missing = [rid for rid in row_order if rid not in pos]
        if missing:
            raise RuntimeError(
                f"{cell}: {len(missing)} fold-map rows missing from store (e.g. {missing[:3]})"
            )
        sel = np.array([pos[rid] for rid in row_order], dtype=np.int64)
        raw = {s: a[sel] for s, a in raw.items()}
        ids = list(row_order)
    out = {
        "row_ids": ids,
        "arrays": {s: decode_bf16_np(raw[s]) for s in slots},
        "raw": {s: raw[s] for s in want_raw},
    }
    d = next(iter(out["arrays"].values())).shape[1] if slots else None
    if d is not None:
        for s in slots:
            if out["arrays"][s].shape != (len(ids), d):
                raise RuntimeError(f"{cell}/{s}: inconsistent array shape")
    return out


# ---------------------------------------------------------------------------
# Fold map (build + IO)
# ---------------------------------------------------------------------------


def _greedy_family_folds(counts: dict, k: int) -> dict:
    """Greedy LPT: assign family keys to k folds balancing ROW counts.

    Deterministic: families sorted by (-count, str(key)); ties to the lowest
    fold id. Keeps the family-held-out purity while minimizing the chance a
    big fold trips the per-fold n_train floor.
    """
    loads = [0] * k
    assign: dict = {}
    for key in sorted(counts, key=lambda x: (-counts[x], str(x))):
        f = min(range(k), key=lambda i: (loads[i], i))
        assign[key] = f
        loads[f] += counts[key]
    return assign


def audit_story_folds(family_keys: list, folds: list[int], k: int) -> dict:
    """Zero train/eval family-overlap audit per fold (plan §4.4). Raises on hit."""
    fams_by_fold: dict[int, set] = {f: set() for f in range(k)}
    for fam, f in zip(family_keys, folds):
        fams_by_fold[int(f)].add(fam)
    per_fold = []
    for f in range(k):
        eval_fams = fams_by_fold[f]
        train_fams = set().union(*(fams_by_fold[g] for g in range(k) if g != f))
        overlap = sorted(str(x) for x in (train_fams & eval_fams))
        per_fold.append({"fold": f, "n_eval_families": len(eval_fams), "overlap": overlap})
        if overlap:
            raise RuntimeError(f"family-held-out audit FAILED at fold {f}: overlap={overlap[:5]}")
    return {"verdict": "zero-overlap", "k": k, "per_fold": per_fold}


def build_fold_map(
    ledgers: dict[str, list[dict]],
    *,
    k: int = K_FOLDS,
    seed: int = cm.SEED,
    n_eq_floor: int = N_EQ_FLOOR,
    n_train_floor: int = N_TRAIN_FLOOR,
    min_storyq: int = MIN_STORYQ_SURVIVORS,
    min_dialog: int = MIN_DIALOG_SURVIVORS,
    store_index: dict | None = None,
) -> dict:
    """Deterministic shared fold map over the realized store ledgers.

    ``ledgers``: cell -> ledger rows (from :func:`load_ledger`). Every P6 pod
    rebuilds this identically from the same staged rows.json set; drivers
    verify against an on-disk copy by canonical sha.
    """
    cells = sorted(ledgers)
    unknown = [c for c in cells if c not in cm.ALL_CELLS]
    if unknown:
        raise RuntimeError(f"unknown cells in ledgers: {unknown}")
    for required in ("chat", "plain_text"):
        if required not in cells:
            raise RuntimeError(f"required cell {required} missing from store (G2b violated)")
    n_sq = sum(c in cells for c in cm.STORY_Q_CELLS)
    n_dg = sum(c in cells for c in cm.DIALOG_CELLS)
    if n_sq < min_storyq or n_dg < min_dialog:
        raise RuntimeError(
            f"G2b survivor predicate violated at P6: storyq={n_sq} (min {min_storyq}), "
            f"dialog={n_dg} (min {min_dialog})"
        )

    kept_counts = {c: len(ledgers[c]) for c in cells}
    excluded: dict[str, str] = {}

    # Non-user cells below the floor should have been dropped at G2b — fail loud.
    for c in cells:
        if c not in cm.USER_CELLS and kept_counts[c] < n_eq_floor:
            raise RuntimeError(
                f"non-user cell {c} kept={kept_counts[c]} < floor {n_eq_floor} "
                "(G2b should have dropped it upstream)"
            )

    # User pair-complete intersection cohort (plan §4.2b).
    user_info = None
    user_ids: list[str] | None = None
    have_users = all(c in cells for c in cm.USER_CELLS)
    if have_users:
        real_ids = [r["row_id"] for r in ledgers["chat_user_real"]]
        sim_ids = [r["row_id"] for r in ledgers["chat_user_sim"]]
        inter = sorted(set(real_ids) & set(sim_ids))
        user_info = {
            "n_real": len(real_ids),
            "n_sim": len(sim_ids),
            "n_intersection": len(inter),
            "n_real_only": len(set(real_ids) - set(sim_ids)),
            "n_sim_only": len(set(sim_ids) - set(real_ids)),
        }
        user_ids = inter
        if len(inter) < n_eq_floor:
            excluded["__user_neq__"] = (
                f"user intersection {len(inter)} < {n_eq_floor}: user arms fit at their "
                "own cohort size (labeled below_n_eq), excluded from the N_eq min"
            )
    else:
        for c in cm.USER_CELLS:
            if c in cells:
                raise RuntimeError(
                    f"only one user arm present ({c}); the pair-complete intersection "
                    "needs both arms (a single-arm store is a staging bug)"
                )

    # N_eq over cohort sizes (user arms enter as ONE intersection cohort).
    n_eq_pool = {c: kept_counts[c] for c in cells if c not in cm.USER_CELLS}
    if have_users and "__user_neq__" not in excluded:
        n_eq_pool["user_intersection"] = len(user_ids)  # type: ignore[arg-type]
    n_eq = min(n_eq_pool.values())
    if n_eq < n_eq_floor:
        raise RuntimeError(f"N_eq={n_eq} < floor {n_eq_floor} (cells: {n_eq_pool})")

    cells_out: dict[str, dict] = {}
    for cell in cells:
        rows = ledgers[cell]
        rng = np.random.default_rng(cm.derived_seed(seed, "foldmap", cell))
        if cell in cm.USER_CELLS:
            assert user_ids is not None
            # SHARED subsample + assignment across the two arms (seeded on the
            # PAIR, not the cell, so both arms get identical cohorts/folds).
            rng = np.random.default_rng(cm.derived_seed(seed, "foldmap", "user_pair"))
            cohort = list(user_ids)
            n_fit = min(n_eq, len(cohort))
            sel = sorted(rng.permutation(len(cohort))[:n_fit].tolist())
            row_ids = [cohort[i] for i in sel]
            order = rng.permutation(n_fit)
            folds = np.empty(n_fit, dtype=np.int64)
            for j, chunk in enumerate(np.array_split(order, k)):
                folds[chunk] = j
            entry = {
                "fold_structure": "conversation-grouped",
                "row_ids": row_ids,
                "folds": [int(x) for x in folds],
                "below_n_eq": bool(n_fit < n_eq),
            }
        elif cell in cm.STORY_CELLS:
            by_id = {r["row_id"]: r for r in rows}
            all_ids = sorted(by_id)
            sel = sorted(rng.permutation(len(all_ids))[:n_eq].tolist())
            row_ids = [all_ids[i] for i in sel]
            fams = []
            for rid in row_ids:
                fam = by_id[rid].get("final_seed_id")
                if fam is None:
                    raise RuntimeError(f"{cell}/{rid}: missing final_seed_id (fold key)")
                fams.append(fam)
            fam_counts: dict = {}
            for fam in fams:
                fam_counts[fam] = fam_counts.get(fam, 0) + 1
            if len(fam_counts) < k:
                raise RuntimeError(f"{cell}: only {len(fam_counts)} final_seed_id families < k={k}")
            fam_fold = _greedy_family_folds(fam_counts, k)
            folds = [fam_fold[fam] for fam in fams]
            audit = audit_story_folds(fams, folds, k)
            comp = np.empty(len(row_ids), dtype=np.int64)
            order = rng.permutation(len(row_ids))
            for j, chunk in enumerate(np.array_split(order, k)):
                comp[chunk] = j
            entry = {
                "fold_structure": "family-held-out",
                "row_ids": row_ids,
                "folds": [int(x) for x in folds],
                "family_keys": fams,
                "companion_scene_folds": [int(x) for x in comp],
                "story_fold_audit": audit,
            }
        else:  # chat / plain_text: conversation-grouped (row == conversation)
            all_ids = sorted(r["row_id"] for r in rows)
            sel = sorted(rng.permutation(len(all_ids))[:n_eq].tolist())
            row_ids = [all_ids[i] for i in sel]
            order = rng.permutation(len(row_ids))
            folds = np.empty(len(row_ids), dtype=np.int64)
            for j, chunk in enumerate(np.array_split(order, k)):
                folds[chunk] = j
            entry = {
                "fold_structure": "conversation-grouped",
                "row_ids": row_ids,
                "folds": [int(x) for x in folds],
            }
        sizes = [entry["folds"].count(f) for f in range(k)]
        n_rows = len(entry["row_ids"])
        for f in range(k):
            n_train = n_rows - sizes[f]
            if n_train <= n_train_floor:
                raise RuntimeError(
                    f"{cell} fold {f}: n_train={n_train} <= floor {n_train_floor} "
                    f"(fold sizes {sizes}; plan G2b per-realized-fold assert)"
                )
        entry["n_rows"] = n_rows
        entry["fold_sizes"] = sizes
        cells_out[cell] = entry

    return {
        "k": k,
        "seed": seed,
        "n_eq": int(n_eq),
        "n_eq_floor": int(n_eq_floor),
        "n_train_floor": int(n_train_floor),
        "kept_counts": kept_counts,
        "excluded": excluded,
        "user_intersection": user_info,
        "cells": cells_out,
        "store_index": store_index or {},
    }


def fold_map_sha(fold_map: dict) -> str:
    core = {kk: fold_map[kk] for kk in fold_map if kk not in ("metadata", "sha256")}
    return hashlib.sha256(
        json.dumps(core, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def build_store_index(store_root: Path, cells: list[str]) -> dict:
    """Per-cell part stems + row counts (the P6 staging count-assert manifest)."""
    out: dict[str, dict] = {}
    for cell in cells:
        idxs = production_part_indices(store_root, cell)
        rows_per_part = []
        for ci in idxs:
            payload = json.loads(
                (store_root / f"{cell}__part{ci:04d}__rows.json").read_text(encoding="utf-8")
            )
            rows_per_part.append(len(payload["rows"]))
        out[cell] = {
            "parts": [f"{cell}__part{ci:04d}" for ci in idxs],
            "rows_per_part": rows_per_part,
        }
    return out


def load_or_build_fold_map(store_root: Path, ledger_root: Path, **kwargs) -> dict:
    """Build the fold map from the staged ledgers; verify (or write) the disk copy.

    Cross-pod identity contract: every pod holds ALL cells' rows.json (KB-scale)
    and rebuilds the identical map; a sha mismatch vs the committed copy fails
    loud (mixed store generations).
    """
    cells = [c for c in cm.ALL_CELLS if production_part_indices(store_root, c)]
    if not cells:
        raise RuntimeError(f"no production cells found under {store_root}")
    ledgers = {c: load_ledger(store_root, c) for c in cells}
    fm = build_fold_map(ledgers, store_index=build_store_index(store_root, cells), **kwargs)
    fm["sha256"] = fold_map_sha(fm)
    path = ledger_root / FOLD_MAP_NAME
    if path.exists():
        on_disk = json.loads(path.read_text(encoding="utf-8"))
        if on_disk.get("sha256") != fm["sha256"]:
            raise RuntimeError(
                f"fold_map sha mismatch: rebuilt {fm['sha256'][:12]} vs on-disk "
                f"{on_disk.get('sha256', '?')[:12]} at {path} — mixed store generations; "
                "use a fresh ledger root"
            )
    else:
        fm["metadata"] = cm.run_metadata()
        cm.atomic_write_json(path, fm)
        _log(f"[fold-map] wrote {path} (n_eq={fm['n_eq']}, cells={len(fm['cells'])})")
    return fm


def fold_splits(entry: dict, *, companion: bool = False) -> list[tuple[np.ndarray, np.ndarray]]:
    """(train_idx, eval_idx) per fold, indices into the entry's row order."""
    key = "companion_scene_folds" if companion else "folds"
    folds = np.asarray(entry[key], dtype=np.int64)
    k = int(folds.max()) + 1
    out = []
    for f in range(k):
        te = np.flatnonzero(folds == f)
        tr = np.flatnonzero(folds != f)
        out.append((tr, te))
    return out


# ---------------------------------------------------------------------------
# Row-stats + preds sidecars
# ---------------------------------------------------------------------------


def write_rowstats(
    path: Path,
    *,
    row_ids: list[str],
    folds: np.ndarray,
    ss_res: np.ndarray,
    ss_tot: np.ndarray,
    extra: dict | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        "row_ids": np.array(row_ids),
        "folds": np.asarray(folds, dtype=np.int64),
        "ss_res": np.asarray(ss_res, dtype=np.float64),
        "ss_tot": np.asarray(ss_tot, dtype=np.float64),
    }
    if extra:
        arrays.update(extra)
    tmp = path.with_name(path.name[: -len(".npz")] + ".tmp.npz")
    with open(tmp, "wb") as fh:
        np.savez(fh, **arrays)  # plain savez (#813)
    tmp.replace(path)


def load_rowstats(path: Path) -> dict:
    if not path.exists():
        raise RuntimeError(f"missing rowstats sidecar {path} (run its producing unit first)")
    with np.load(path, allow_pickle=False) as z:
        return {k: np.asarray(z[k]) for k in z.files}


def write_preds(path: Path, *, row_ids: list[str], folds: np.ndarray, preds: np.ndarray) -> None:
    """float32 preds sidecar — the unit-4 retrieval-battery seam (pool = the
    cell's pooled held-out answers; chance = 1/pool; conventions in unit 4)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name[: -len(".npz")] + ".tmp.npz")
    with open(tmp, "wb") as fh:
        np.savez(
            fh,
            row_ids=np.array(row_ids),
            folds=np.asarray(folds, dtype=np.int64),
            preds=np.asarray(preds, dtype=np.float32),
        )
    tmp.replace(path)


# ---------------------------------------------------------------------------
# Pooled-R² bootstrap machinery (rowstats convention: ss_tot vs POOLED mean)
# ---------------------------------------------------------------------------


def pooled_r2(ss_res: np.ndarray, ss_tot: np.ndarray) -> float:
    tot = float(np.sum(ss_tot))
    if tot < 1e-18:
        return float("nan")
    return 1.0 - float(np.sum(ss_res)) / tot


def bootstrap_r2_draws(
    ss_res: np.ndarray, ss_tot: np.ndarray, *, n_draws: int, seed: int
) -> np.ndarray:
    """Row-resample (scene/conversation grain: one row == one scene/conversation)
    pooled-R² bootstrap draws. Mirrors issue2054_fits._bootstrap_conv_ci's
    per-row SS decomposition with the pooled-mean ss_tot convention."""
    n = ss_res.shape[0]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_draws, n))
    res = ss_res[idx].sum(axis=1)
    tot = ss_tot[idx].sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return 1.0 - res / tot


def margin_bootstrap(
    ss_res: np.ndarray,
    ss_tot: np.ndarray,
    *,
    floor: float,
    n_draws: int,
    seed: int,
) -> dict:
    """Ceiling-margin m = R² − floor bootstrap (plan §3 reporting tiers).

    The floor (max(null p95, 0.05)) is FIXED across draws; only the ceiling
    side is resampled (registered: 'scene/conversation-grain resample of the
    ceiling side'). All draw values persisted; no skip rule (not a ratio)."""
    draws = bootstrap_r2_draws(ss_res, ss_tot, n_draws=n_draws, seed=seed) - floor
    finite = draws[np.isfinite(draws)]
    return {
        "floor": float(floor),
        "n_draws": int(n_draws),
        "n_finite": int(finite.size),
        "draws": [float(x) for x in draws],
        "ci_lo": float(np.percentile(finite, 2.5)) if finite.size else float("nan"),
        "ci_hi": float(np.percentile(finite, 97.5)) if finite.size else float("nan"),
        "median": float(np.median(finite)) if finite.size else float("nan"),
    }


def tier_from_margin(ci_lo: float, ci_hi: float) -> str:
    """Registered reporting-tier assignment (plan §3)."""
    if np.isnan(ci_lo) or np.isnan(ci_hi):
        return "boundary-indeterminate"
    if ci_lo > 0.0:
        return "clearly-mappable"
    if ci_hi < 0.0:
        return "clearly-unmappable"
    return "boundary-indeterminate"


def recovery_bootstrap(
    transfer_ss_res: np.ndarray,
    ceiling_ss_res: np.ndarray,
    ss_tot: np.ndarray,
    *,
    floor: float,
    n_draws: int,
    seed: int,
    valid_frac: float = VALID_DRAW_FRAC,
) -> dict:
    """Joint (transfer, ceiling) row-resample recovery-fraction bootstrap with
    the registered skip-and-count denominator guard (plan §3): a draw whose
    resampled ceiling fails the floor is SKIPPED and COUNTED; valid draws
    < valid_frac·n_draws ⇒ the ratio verdict is SUPPRESSED (ceiling + transfer
    reported separately by the caller)."""
    n = ss_tot.shape[0]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_draws, n))
    tot = ss_tot[idx].sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        ceil_draws = 1.0 - ceiling_ss_res[idx].sum(axis=1) / tot
        tran_draws = 1.0 - transfer_ss_res[idx].sum(axis=1) / tot
    ok = np.isfinite(ceil_draws) & np.isfinite(tran_draws) & (ceil_draws > floor)
    rec = tran_draws[ok] / ceil_draws[ok]
    n_valid = int(ok.sum())
    suppressed = n_valid < int(np.ceil(valid_frac * n_draws))
    out = {
        "floor": float(floor),
        "n_draws": int(n_draws),
        "n_skipped_ceiling_floor": int(n_draws - n_valid),
        "n_valid": n_valid,
        "suppressed": bool(suppressed),
        "recovery_draws": [float(x) for x in rec],
        "ceiling_draws": [float(x) for x in ceil_draws],
        "transfer_draws": [float(x) for x in tran_draws],
    }
    if not suppressed and n_valid:
        out["ci_lo"] = float(np.percentile(rec, 2.5))
        out["ci_hi"] = float(np.percentile(rec, 97.5))
        out["median"] = float(np.median(rec))
    return out


# ---------------------------------------------------------------------------
# G3 gate IO (plan §7 G3: chat pilot gates the whole P6 fan-out)
# ---------------------------------------------------------------------------


def require_g3_pass(gate_path: Path) -> dict:
    if not gate_path.exists():
        raise RuntimeError(
            f"G3 gate file missing at {gate_path} — run "
            "`issue2378_fits.py --phase g3` first (plan §7: the chat pilot gates the fan-out)"
        )
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    if gate.get("verdict") != "PASS":
        raise RuntimeError(
            f"G3 gate verdict={gate.get('verdict')} at {gate_path} — fan-out aborted"
        )
    return gate


# ---------------------------------------------------------------------------
# §4.2b user-pair asserts (before ANY paired statistic)
# ---------------------------------------------------------------------------


def assert_user_pair(store_root: Path, fold_map: dict, layer: int) -> dict:
    """Fail-loud §4.2b intersection asserts: identical ordered conversation-ID
    lists, identical fold assignments, identical per-conversation v_C sha256
    (over the raw bf16-as-uint16 bytes at the read layer) across the two arms.

    Returns a diagnostic dict (persist it); on ANY mismatch raises AFTER
    computing a decoded max-abs-diff diagnostic so a production failure is
    immediately attributable (rig bug vs numerics)."""
    for c in cm.USER_CELLS:
        if c not in fold_map["cells"]:
            raise RuntimeError(f"user arm {c} missing from fold map — no paired statistic possible")
    real_e = fold_map["cells"]["chat_user_real"]
    sim_e = fold_map["cells"]["chat_user_sim"]
    if real_e["row_ids"] != sim_e["row_ids"]:
        raise RuntimeError("§4.2b assert FAILED: user-arm ordered conversation-ID lists differ")
    if real_e["folds"] != sim_e["folds"]:
        raise RuntimeError("§4.2b assert FAILED: user-arm fold assignments differ")
    order = real_e["row_ids"]
    real = load_cell_arrays(
        store_root, "chat_user_real", layer, (), row_order=order, want_raw=("v_C",)
    )
    sim = load_cell_arrays(
        store_root, "chat_user_sim", layer, (), row_order=order, want_raw=("v_C",)
    )
    a, b = real["raw"]["v_C"], sim["raw"]["v_C"]
    mismatched = [
        order[i]
        for i in range(len(order))
        if hashlib.sha256(a[i].tobytes()).hexdigest() != hashlib.sha256(b[i].tobytes()).hexdigest()
    ]
    diag = {
        "n_conversations": len(order),
        "n_hash_mismatched": len(mismatched),
        "mismatched_examples": mismatched[:10],
        "layer": int(layer),
    }
    if mismatched:
        fa, fb = decode_bf16_np(a), decode_bf16_np(b)
        diag["decoded_max_abs_diff"] = float(np.max(np.abs(fa - fb)))
        diag["decoded_mean_abs_diff"] = float(np.mean(np.abs(fa - fb)))
        raise RuntimeError(
            "§4.2b assert FAILED: per-conversation v_C sha256 mismatch across user arms — "
            f"{json.dumps(diag)} (tiny decoded diffs point at batch-composition bf16 "
            "numerics in the capture rig, a PLAN-DEFECT to surface; large diffs point "
            "at a prefix-render rig bug)"
        )
    return diag


def unit_seed(*parts: object) -> int:
    """137-rooted per-unit seed (deviation from the plan's literal seed=137 per
    call: identical literal seeds would reuse the same permutations across
    cells/folds and correlate null bands; documented in every regime block)."""
    return cm.derived_seed(cm.SEED, *parts)
