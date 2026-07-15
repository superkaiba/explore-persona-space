#!/usr/bin/env python
"""Issue #1345 Phases 0/3/4 — staging probe, matched-n subsets, within-regime fits.

Phase 0 (--phase0): download ONE pinned parent shard (instruct_chat_s @
7159e5804d) and open it through the PRODUCTION 28-layer pt loader
(`issue825_fit_cells._load_bundle_any` -> `_load_bundle_pt`), asserting 28
layers, conv_ids read from the shard (NOT an np.arange fallback), and a
CPU-clean one-cell L19 ridge fit end-to-end (plan §10 realized-keys row +
§4 Phase 0 device/staging smoke).

Phase 3 (--build-matched): pair-level matched-n subsets — the shared R1/R2
conversation subset (intersection over the four re-extracted stores) + the
per-model R3-pair subsets (both sides subsampled to the global min at GROUP
level, seed 0). Written to data/issue_1345/matched_n/matched_subsets.json.

Phase 4 (--cells ...): the 12 within-regime cells through the REUSED
`issue825_fit_cells.run_cell` (Gram-space GCV ridge, 28 layers, conv-grouped
5-fold CV, conversation-level shuffle nulls, selection-symmetric summary),
R1/R2 restricted to the shared subset via run_cell's row allowlist. Adds a
conversation-level bootstrap CI (plan §4: the resampling unit is the
conversation/story) and persists the L19 held-out predictions per cell
(`preds_cache`) for the verdict-lattice paired bootstrap.

Parity gate (--parity): re-extracted context-arm L19 R^2 (FULL per-store n)
must reproduce the pinned cells_S1/S2/S1N/S2N anchors within ±0.02, else
exit 3 (halt-and-diagnose; plan §7 kill criterion). Under --smoke the SAME
computation runs end-to-end (PASS_UNIFIED) but the anchor comparison is
INFORMATIONAL only: the plan defines the ±0.02 gate at production n
(~4724-5000 conversations), where a grouped-CV R^2 at smoke n (~8) can never
reproduce the anchors by construction (crash-fix r3, att-20260715-161700).
The production HALT semantics are untouched.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
for _p in (str(_SCRIPT_DIR), str(_REPO_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

import issue825_fit_cells as fc  # noqa: E402
import issue1345_common as c  # noqa: E402
import numpy as np  # noqa: E402

# ---------------------------------------------------------------------------
# Bundle access helpers (production pt-shard loader path ONLY — plan §4 P4)
# ---------------------------------------------------------------------------
SLIM_KEYS = ("slots", "profiles", "nll")  # fits never read perpos (~36 GB/bundle stacked)


def load_regime_bundle(turnstore_dir: Path, model: str, regime: str) -> dict:
    """Load one (model, regime) store via the 28-layer pt loader + sanity asserts."""
    bundle = fc._load_bundle_any(
        turnstore_dir, model, c.REGIME_FORMAT[regime], c.TRACK, wanted_keys=SLIM_KEYS
    )
    c.assert_pt_bundle(bundle, expect_slots=2, expect_layers=fc.EXPECTED_LAYERS)
    return bundle


def bundle_conv_ids(turnstore_dir: Path, model: str, regime: str) -> list[str]:
    """Row conv_ids for a store, read from the cheap shard sidecar JSONs."""
    stem = c.stem_for(model, regime)
    sidecars = sorted(turnstore_dir.glob(f"{stem}_shard*.json"))
    assert sidecars, f"no shard sidecars for {stem} in {turnstore_dir}"
    ids: list[str] = []
    for sp in sidecars:
        ids.extend(str(x) for x in json.loads(sp.read_text())["conv_ids"])
    return ids


# ---------------------------------------------------------------------------
# Phase 0 — pinned-shard staging + device probe
# ---------------------------------------------------------------------------
def phase0_probe(dl_dir: Path, out_dir: Path) -> None:
    """Stage one pinned parent shard; open via the production loader; fit L19."""
    probe_dir = dl_dir / "phase0"
    shard = c.stage_parent_shard("instruct_chat_s", probe_dir, shard_idx=0)
    # The staged file lands under the parent prefix subtree; the loader globs a
    # flat dir — point it at the shard's own directory.
    bundle = fc._load_bundle_any(shard.parent, "instruct", "chat", "s", wanted_keys=SLIM_KEYS)
    c.assert_pt_bundle(bundle, expect_slots=1, expect_layers=fc.EXPECTED_LAYERS)
    xy = fc._cell_xy(bundle, {"slot_index": 0, "target_turn_index": 1})
    X, Y, conv_ids = xy["X"], xy["Y"], xy["conv_ids"]
    assert isinstance(X, np.ndarray) and X.dtype == np.float32, (type(X), X.dtype)
    assert X.shape[1:] == (fc.EXPECTED_LAYERS, 3584), X.shape
    t0 = time.time()
    sweep = fc.heldout_r2_sweep(
        X[:, [19], :],
        Y[:, [19], :],
        conv_ids,
        n_folds=fc.N_FOLDS,
        seed=fc.FIT_SEED,
        null_draws=0,
        collect_cosines=False,
    )
    r2 = float(sweep["r2_obs"][0])
    elapsed = time.time() - t0
    payload = {
        "metadata": c.metadata(fc.FIT_SEED, len(conv_ids), "scripts/issue1345_fit_cells.py"),
        "probe": "phase0",
        "shard": shard.name,
        "revision": c.PIN_REV,
        "n_rows": len(conv_ids),
        "n_layers": int(X.shape[1]),
        "conv_ids_source": "pt-shards (non-arange)",
        "l19_r2_one_shard": r2,
        "fit_seconds": elapsed,
    }
    c.write_json(out_dir / "phase0_probe.json", payload)
    print(
        f"PASS phase0: shard={shard.name} rev={c.PIN_REV} n={len(conv_ids)} "
        f"layers=28 conv_ids=non-arange slots=(n,1,28,3584) L19 one-shard R2={r2:.4f} "
        f"({elapsed:.1f}s)",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Fold-degeneracy guard (smoke-path ONLY — production callers never consult it)
# ---------------------------------------------------------------------------
def degenerate_fold_reason(conv_ids, *, n_folds: int, seed: int, tgt_conv_ids=None) -> str | None:
    """Reason string when grouped n_folds CV over these rows would fit NOTHING.

    Mirrors the reused #825 per-fold skip predicate (heldout_r2_sweep /
    transfer_sweep: a fold runs iff its held-out set is non-empty AND the
    train side has >= 3 rows) with the SAME seeded fold assignment.
    ``tgt_conv_ids`` switches to the transfer shape (src trains, tgt
    evaluates; per-side fold assignments). Returns None when at least one
    fold would fit. Smoke guard only: at story-shortfall grains (kept=1-3
    stories) ALL folds skip and downstream .all()/empty-array consumers
    crash (#1345 v3 code-review bug-class sweep); production paths stay
    byte-untouched and never call this.
    """
    src = np.asarray([str(x) for x in conv_ids])
    tgt = src if tgt_conv_ids is None else np.asarray([str(x) for x in tgt_conv_ids])
    if len(src) == 0 or len(tgt) == 0:
        return f"0 rows (src={len(src)}, tgt={len(tgt)})"
    folds_src = fc._cv_folds(src, n_folds, seed)
    folds_tgt = folds_src if tgt_conv_ids is None else fc._cv_folds(tgt, n_folds, seed)
    for k in range(n_folds):
        if (folds_tgt == k).sum() > 0 and (folds_src != k).sum() >= 3:
            return None
    return (
        f"all {n_folds} folds skip (src rows={len(src)}, groups={len(np.unique(src))}; "
        f"tgt rows={len(tgt)}, groups={len(np.unique(tgt))}; every fold has an empty "
        "held-out set or <3 train rows)"
    )


# ---------------------------------------------------------------------------
# Phase 3 — matched-n subsets
# ---------------------------------------------------------------------------
def build_matched(turnstore_dir: Path, matched_dir: Path, *, r3_models: set[str]) -> dict:
    """Pair-level matched-n subsets (plan §4 Phase 3), persisted + returned.

    ``r3_models`` = models whose story regime survived the per-model yield
    floor (plan §7: the floor binds PER MODEL; a halted model's R3 pair is
    reported as coverage loss, not built). Empty set == story regime fully
    halted (the old ``include_r3=False``).
    """
    assert r3_models <= set(c.MODELS), r3_models
    id_sets = {}
    for model in c.MODELS:
        for regime in ("r1", "r2"):
            id_sets[(model, regime)] = bundle_conv_ids(turnstore_dir, model, regime)
    shared = sorted(set.intersection(*(set(v) for v in id_sets.values())))
    if not shared:
        # Actionable in BOTH modes (v3 sweep item: never a bare AssertionError).
        # An empty 4-stem intersection cannot be informationally skipped — every
        # downstream phase consumes matched_subsets.json — and under smoke the
        # four stems extract the SAME first-8 conversations, so emptiness is a
        # real extraction bug the smoke exists to surface.
        sizes = {f"{m}_{r}": len(v) for (m, r), v in id_sets.items()}
        raise RuntimeError(
            "shared R1/R2 conversation subset is EMPTY — extraction drift "
            f"(per-store row counts: {sizes}). All four stems render the same "
            "track-S corpus, so a non-empty intersection is expected at ANY n "
            "(smoke included); check the extract_r1r2 shard sidecars."
        )
    out: dict = {
        "metadata": c.metadata(c.SUBSAMPLE_SEED, len(shared), "scripts/issue1345_fit_cells.py"),
        "shared_r1r2_convs": shared,
        "per_store_n": {f"{m}_{r}": len(v) for (m, r), v in id_sets.items()},
        "per_model_r3_pair": {},
        "r3_halted_models": sorted(set(c.MODELS) - r3_models),
    }
    if r3_models:
        rng = np.random.default_rng(c.SUBSAMPLE_SEED)
        for model in c.MODELS:
            if model not in r3_models:
                continue
            r3_ids = bundle_conv_ids(turnstore_dir, model, "r3")  # story id per ROW
            n_r3_rows = len(r3_ids)
            n_min = min(n_r3_rows, len(shared))
            if len(shared) > n_min:
                r12_subset = sorted(
                    rng.choice(np.asarray(shared), size=n_min, replace=False).tolist()
                )
            else:
                r12_subset = shared
            uniq_stories, counts = np.unique(np.asarray(r3_ids), return_counts=True)
            if n_r3_rows > n_min:
                order = rng.permutation(len(uniq_stories))
                keep, total = [], 0
                for k in order:
                    if total + int(counts[k]) > n_min:
                        continue
                    keep.append(str(uniq_stories[k]))
                    total += int(counts[k])
                r3_subset = sorted(keep)
                n_r3_kept = total
            else:
                r3_subset = sorted(str(s) for s in uniq_stories)
                n_r3_kept = n_r3_rows
            out["per_model_r3_pair"][model] = {
                "n_min": int(n_min),
                "r12_convs": r12_subset,
                "r3_story_ids": r3_subset,
                "n_r3_rows_kept": int(n_r3_kept),
                "n_r3_rows_total": int(n_r3_rows),
                "n_r3_stories_total": len(uniq_stories),
            }
    matched_dir.mkdir(parents=True, exist_ok=True)
    c.write_json(matched_dir / "matched_subsets.json", out)
    return out


def load_matched(matched_dir: Path) -> dict:
    return json.loads((matched_dir / "matched_subsets.json").read_text())


# ---------------------------------------------------------------------------
# Parity gate — re-extracted context arm vs pinned parent anchors (±0.02)
# ---------------------------------------------------------------------------
def _parity_cell_line(
    model: str,
    regime: str,
    ours: float,
    anchor: float,
    dev: float,
    tol: float,
    n_rows: int,
    smoke: bool,
) -> str:
    """Per-(model, regime) parity log line — informational form under smoke.

    The anchor comparison binds only at production n (~4724-5000): a grouped-CV
    R^2 at smoke n (~8) can never reproduce the anchors, so the smoke leg logs
    the numbers without a PASS/FAIL verdict (crash-fix r3).
    """
    if smoke:
        return (
            f"[parity][smoke] informational: {model}/{regime} ours={ours:.4f} "
            f"anchor={anchor:.4f} dev={dev:.4f} (n={n_rows} — anchor check binds "
            "at production n only)"
        )
    return (
        f"[parity] {model}/{regime}: ours={ours:.4f} anchor={anchor:.4f} "
        f"dev={dev:.4f} ({'PASS' if dev <= tol else 'FAIL'})"
    )


def _parity_finalize(failures: list[str], tol: float, smoke: bool) -> None:
    """Terminal parity verdict: production HALT (exit 3) — untouched; smoke informational.

    Raises SystemExit(3) on any production failure; under smoke it only logs
    (the smoke leg exercises the identical computation, PASS_UNIFIED, but the
    ±0.02 plan §7 kill criterion is defined for the production re-extraction).
    """
    if not failures:
        return
    if smoke:
        print(
            f"[parity][smoke] {len(failures)} cell(s) deviate > ±{tol} at smoke n — "
            "informational only (production HALT semantics unchanged)",
            flush=True,
        )
        return
    print(
        f"[parity] HALT: {failures} deviate > ±{tol} from the pinned anchors "
        "(plan §7 Phase-0/2a parity kill) — diagnose before any cross-regime read",
        file=sys.stderr,
        flush=True,
    )
    raise SystemExit(3)


def parity_gate(
    turnstore_dir: Path, out_dir: Path, *, tol: float = c.PARITY_TOL, smoke: bool = False
) -> None:
    """Halt (exit 3) when any re-extracted context-arm L19 R^2 drifts > tol.

    smoke=True runs the identical end-to-end computation but demotes the
    anchor comparison to informational (no HALT) — the anchors were computed
    at production n and are unsatisfiable at smoke n by construction.
    """
    results, failures = {}, []
    for (model, regime), anchor_file in c.PARITY_ANCHOR_FILES.items():
        anchor_path = Path(anchor_file)
        assert anchor_path.exists(), (
            f"parity anchor missing: {anchor_path} (broken/sparse checkout — "
            "add eval_results/issue_825 to the sparse cones)"
        )
        anchor = float(json.loads(anchor_path.read_text())["r2_per_layer_obs"][19])
        doc = c.PARITY_ANCHOR_DOC[(model, regime)]
        assert abs(anchor - doc) < 0.005, (model, regime, anchor, doc)
        bundle = load_regime_bundle(turnstore_dir, model, regime)
        xy = fc._cell_xy(
            bundle,
            {"slot_index": c.ARM_SLOT_INDEX["context"], "target_turn_index": 1},
        )
        sweep = fc.heldout_r2_sweep(
            xy["X"][:, [19], :],
            xy["Y"][:, [19], :],
            xy["conv_ids"],
            n_folds=fc.N_FOLDS,
            seed=fc.FIT_SEED,
            null_draws=0,
            collect_cosines=False,
        )
        ours = float(sweep["r2_obs"][0])
        dev = abs(ours - anchor)
        results[f"{model}_{regime}"] = {
            "anchor_l19_r2": anchor,
            "reextracted_l19_r2": ours,
            "abs_dev": dev,
            "n_rows": len(xy["conv_ids"]),
            "pass": bool(dev <= tol),
        }
        print(
            _parity_cell_line(model, regime, ours, anchor, dev, tol, len(xy["conv_ids"]), smoke),
            flush=True,
        )
        if dev > tol:
            failures.append(f"{model}_{regime}")
    payload = {
        "metadata": c.metadata(fc.FIT_SEED, len(results), "scripts/issue1345_fit_cells.py"),
        "tolerance": tol,
        "mode": "smoke-informational" if smoke else "binding",
        "results": results,
        # Under smoke the anchor check is non-binding: pass is None (not a
        # verdict), never a fake True/False the sentinel could misread.
        "pass": None if smoke else not failures,
    }
    c.write_json(out_dir / "parity_gate.json", payload)
    _parity_finalize(failures, tol, smoke)


# ---------------------------------------------------------------------------
# Phase 4 — within-regime fits
# ---------------------------------------------------------------------------
def allowlist_for(cell: dict, matched: dict | None) -> list[str] | None:
    """R1/R2 cells fit on the shared conversation subset; R3 fits its full store."""
    if matched is None or cell["regime"] == "r3":
        return None
    return matched["shared_r1r2_convs"]


def run_cells(
    turnstore_dir: Path,
    out_dir: Path,
    preds_dir: Path,
    cells: list[dict],
    matched: dict | None,
    *,
    n_folds: int,
    seed: int,
    null_draws: int,
    n_boot: int,
    smoke: bool = False,
) -> None:
    preds_dir.mkdir(parents=True, exist_ok=True)
    # Loader-path assert + ONE slim load per (model, regime) — plan §4 Phase 4
    # binding staging rule (pt loader, 28 layers, non-arange conv_ids).
    bundles: dict[tuple[str, str], dict] = {}
    for cell in cells:
        key = (cell["model_key"], cell["regime"])
        if key not in bundles:
            bundles[key] = load_regime_bundle(turnstore_dir, *key)
    for cell in cells:
        cid = cell["cell_id"]
        if smoke:
            # Smoke fold-degeneracy guard (v3 sweep class): at story-shortfall
            # grains (kept=1-3) every fold skips inside run_cell's grouped CV
            # and the downstream bootstrap/summary consumers crash on empty
            # arrays. Skip the cell informationally; production never guards.
            xy_probe = fc._apply_row_allowlist(
                fc._cell_xy(bundles[(cell["model_key"], cell["regime"])], cell),
                allowlist_for(cell, matched),
                cid,
            )
            reason = degenerate_fold_reason(xy_probe["conv_ids"], n_folds=n_folds, seed=seed)
            if reason:
                print(
                    f"[fits][smoke] SKIP cell {cid}: {reason} — informational "
                    "(production semantics unchanged)",
                    flush=True,
                )
                continue
        res = fc.run_cell(
            cell,
            turnstore_dir,
            out_dir,
            n_folds=n_folds,
            seed=seed,
            null_draws=null_draws,
            n_boot=n_boot,
            allowlist=allowlist_for(cell, matched),
            bundle=bundles[(cell["model_key"], cell["regime"])],
        )
        sweep, xy = res["sweep"], res["xy"]
        fitted = sweep["fitted_mask"]
        li = 19 if 19 in sweep["preds_frozen"] else max(sweep["preds_frozen"])
        pred = sweep["preds_frozen"][li][fitted]
        true = xy["Y"][fitted, li, :]
        conv = xy["conv_ids"][fitted]
        np.savez(
            preds_dir / f"{cid}_L{li}.npz",
            pred=pred.astype(np.float32),
            true=true.astype(np.float32),
            conv_ids=np.asarray([str(x) for x in conv]),
            layer=np.asarray([li]),
        )
        # Conversation-level bootstrap (plan §4: resampling unit = conversation/story)
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
        cell_json = out_dir / f"cells_{cid}.json"
        payload = json.loads(cell_json.read_text())
        payload["r2_bootstrap_ci_frozen_layers_conv"] = boot
        payload["n_groups"] = len(np.unique(conv))
        c.write_json(cell_json, payload)
        print(f"[fits] {cid} done (n={len(conv)}, groups={payload['n_groups']})", flush=True)


def select_cells(cells_arg: str, halted: set[str]) -> list[dict]:
    """Resolve --cells against the registry, then drop halted models' r3 cells.

    Membership is asserted against the FULL registry BEFORE the per-model halt
    filter — a halted model's r3 cell in --cells is a deliberate logged drop
    (plan §7 per-model yield floor), never an "unknown cell id" crash (the
    pre-r6 ordering crashed exactly there under --no-r3 + an explicit list).
    """
    cells = c.all_cells()
    if cells_arg != "all":
        wanted = set(cells_arg.split(","))
        cells = [x for x in cells if x["cell_id"] in wanted]
        missing = wanted - {x["cell_id"] for x in cells}
        assert not missing, f"unknown cell ids: {sorted(missing)}"
    dropped = [x["cell_id"] for x in cells if x["regime"] == "r3" and x["model_key"] in halted]
    if dropped:
        print(
            f"[fits] dropping r3 cells for halted model(s) {sorted(halted)}: {dropped} "
            "(per-model yield floor, plan §7)",
            flush=True,
        )
        cells = [x for x in cells if x["cell_id"] not in set(dropped)]
    return cells


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--turnstore-dir", type=Path, default=c.TURNSTORE_DIR)
    ap.add_argument("--out-dir", type=Path, default=c.EVAL_DIR)
    ap.add_argument("--matched-dir", type=Path, default=c.MATCHED_DIR)
    ap.add_argument("--preds-dir", type=Path, default=c.PREDS_CACHE_DIR)
    ap.add_argument("--dl-dir", type=Path, default=c.PARENT_DL_DIR)
    ap.add_argument("--phase0", action="store_true", help="pinned-shard staging probe only")
    ap.add_argument("--build-matched", action="store_true")
    ap.add_argument("--no-r3", action="store_true", help="story regime halted for BOTH models")
    ap.add_argument(
        "--no-r3-models",
        default="",
        help="comma-separated models whose story regime halted (per-model yield "
        "floor, plan §7); their r3 cells/pairs are dropped with a logged reason "
        "while the other model's story leg proceeds",
    )
    ap.add_argument("--parity", action="store_true", help="±0.02 anchor parity gate")
    ap.add_argument(
        "--smoke",
        action="store_true",
        help="smoke leg: parity anchor check runs but is informational (anchors "
        "bind at production n only); fit cells whose grouped CV would be "
        "degenerate at smoke n are skipped with a logged reason",
    )
    ap.add_argument("--cells", default=None, help="'all' or comma-separated cell ids")
    ap.add_argument("--folds", type=int, default=fc.N_FOLDS)
    ap.add_argument("--seed", type=int, default=fc.FIT_SEED)
    ap.add_argument("--null-draws", type=int, default=fc.N_NULL_DRAWS)
    ap.add_argument("--n-boot", type=int, default=fc.N_BOOTSTRAP)
    args = ap.parse_args()

    halted = set(c.MODELS) if args.no_r3 else {m for m in args.no_r3_models.split(",") if m}
    assert halted <= set(c.MODELS), f"unknown --no-r3-models entries: {sorted(halted)}"

    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.phase0:
        phase0_probe(args.dl_dir, args.out_dir)
        return
    if args.parity:
        parity_gate(args.turnstore_dir, args.out_dir, smoke=args.smoke)
    if args.build_matched:
        build_matched(args.turnstore_dir, args.matched_dir, r3_models=set(c.MODELS) - halted)
    if args.cells:
        cells = select_cells(args.cells, halted)
        matched = load_matched(args.matched_dir)
        run_cells(
            args.turnstore_dir,
            args.out_dir,
            args.preds_dir,
            cells,
            matched,
            n_folds=args.folds,
            seed=args.seed,
            null_draws=args.null_draws,
            n_boot=args.n_boot,
            smoke=args.smoke,
        )


if __name__ == "__main__":
    main()
