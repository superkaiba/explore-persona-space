"""Issue #1417 — fits orchestration + cell-vs-reference identity battery.

Imports the FROZEN #825 instruments UNCHANGED (plan §4: no fork):
``issue825_fit_cells`` (``fit825`` — batched Gram/eigh dual ridge, GCV +
inner-group-cv fallback, K=5 conv-grouped folds seed 0, 20 shuffle nulls,
1000-draw bootstraps) drives every per-cell fit via ``run_cell``;
``issue825_map_alignment`` (``ma``) + ``issue825_crossmodel_map_transfer``
(``cm``) supply the identity battery (``_layer_battery`` fold-outer reads,
``_composition_collapse_null``, ``_procrustes_cosine_null``,
``frozen_map_swap``), generalized from base<->instruct pairs to
cell<->reference pairs WITHIN model: ref -> the battery's "i" side, cell ->
"b", so REL = composition.linear.comp_samefn_b2i / ceilings.within_instruct
— the within-REFERENCE ceiling recomputed on the SAME kept∩kept rows and
shared folds as the composed numerator (plan §6 analyzer rule 1, BINDING).

Gates:
  G1 (--anchors): refit C0/C0' anchors from the reused turnstores on the
      shared 4,724-row set; L19 R^2 must reproduce the committed values
      (read at runtime from eval_results/issue_825/naturalistic-single-turn)
      within ±0.01 — HALT (rc 20) on a miss; informational under --smoke
      (gate-calibration parity: a small-n refit cannot reproduce a
      production-n anchor — gotchas #1345).
  G2 (--gate-g2): fit device resolves cuda — HALT (rc 22) unless --allow-cpu.
  PC-3 pilot: the first battery pair is timed end-to-end; projected lane wall
      > 2x --pilot-budget-h writes battery_pilot_report.json and exits rc 7
      (a DESIGNED artifact-routed halt, never a bare rc=1 — gotchas #1415);
      informational under --smoke.

CLI (one model lane per invocation; the dispatcher CVD-pins lanes):
  uv run python scripts/issue1417_battery.py --stage-stores --model instruct ...
  uv run python scripts/issue1417_battery.py --gate-g2
  uv run python scripts/issue1417_battery.py --anchors --model instruct ...
  uv run python scripts/issue1417_battery.py --fits --model instruct ...
  uv run python scripts/issue1417_battery.py --battery --model instruct ...
  uv run python scripts/issue1417_battery.py --summary            # both lanes done
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

import issue825_crossmodel_map_transfer as cm  # noqa: E402
import issue825_fit_cells as fit825  # noqa: E402
import issue825_map_alignment as ma  # noqa: E402
import issue931_common as common931  # noqa: E402
import issue1417_gen as g1417  # noqa: E402
import issue1417_render as r1417  # noqa: E402

SCRIPT = "scripts/issue1417_battery.py"

FROZEN_LAYERS = (14, 18, 19, 26)
HEADLINE_LAYER = 19
N_FOLDS = 5
FIT_SEED = 0
N_NULL_DRAWS = 20
N_BOOT = 1000
MATCHED_DRAWS = 5
REL_BOUNDARY = 0.5  # plan §3 verdict boundary
MAP_EXISTS_MARGIN = 0.1  # plan §6: R^2 clears its shuffle null band by >= 0.1
COLLAPSE_VAR_RATIO = 0.5  # plan §6: Y-variance ratio floor vs C0
COLLAPSE_DUP_RATE = 0.30  # plan §6: duplicate-prefix rate ceiling
DUP_PREFIX_TOKENS = 64
YIELD_FLOOR = 0.5

REFERENCES = {
    "c0_chat": {"kind": "parent", "format": "chat"},
    "c0p_nat": {"kind": "parent", "format": "naturalistic"},
    "c1": {"kind": "own", "cell": "c1_helpful_ctrl"},
}

# Anchor stems per model lane (G1).
ANCHORS_BY_MODEL = {
    "instruct": [c for c in r1417.ANCHOR_CELLS if c["model"] == "instruct"],
    "pretrained": [c for c in r1417.ANCHOR_CELLS if c["model"] == "pretrained"],
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=SCRIPT)
    ap.add_argument("--data-dir", type=Path, default=Path("data/issue_1417"))
    ap.add_argument("--out-dir", type=Path, default=Path("eval_results/issue_1417"))
    ap.add_argument("--model", choices=list(r1417.MODELS))
    ap.add_argument("--stage-stores", action="store_true")
    ap.add_argument("--cycle-stores", action="store_true", help="delete a ref store after use")
    ap.add_argument("--gate-g2", action="store_true")
    ap.add_argument("--allow-cpu", action="store_true", help="smoke/VM only: skip the G2 HALT")
    ap.add_argument("--anchors", action="store_true")
    ap.add_argument("--fits", action="store_true")
    ap.add_argument("--battery", action="store_true")
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="demote production-n gates to log lines")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--null-draws", type=int, default=N_NULL_DRAWS)
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--cosine-null-draws", type=int, default=200)
    ap.add_argument("--collapse-null-draws", type=int, default=200)
    ap.add_argument("--pilot-budget-h", type=float, default=1.0)
    return ap.parse_args()


# ---------------------------------------------------------------------------
# Store staging (reference turnstores; per-stem, scoped listing, atomic files)
# ---------------------------------------------------------------------------
def turnstore_dir(data_dir: Path) -> Path:
    return Path(data_dir) / "turnstore"


def _reference_revision(data_dir: Path) -> str:
    rev_path = r1417.sidecar_dir(data_dir) / "revision.json"
    assert rev_path.exists(), "run issue1417_render.py --fetch-sidecars first (records the rev)"
    return json.loads(rev_path.read_text())["revision"]


def stage_reference_store(data_dir: Path, stem: str) -> Path:
    """Stage one reference store's .pt+.json shards (scoped listing @ the
    recorded revision; <=6 workers; atomic per-file stage_hub_file)."""
    from huggingface_hub import HfApi

    from explore_persona_space.orchestrate.hub import list_hf_files_under_path, stage_hub_file

    dest = turnstore_dir(data_dir)
    dest.mkdir(parents=True, exist_ok=True)
    rev = _reference_revision(data_dir)
    paths = list_hf_files_under_path(
        HfApi(), r1417.HF_DATA_REPO, r1417.PARENT_PREFIX, repo_type="dataset", revision=rev
    )
    shard_paths = sorted(p for p in paths if Path(p).name.startswith(f"{stem}_shard"))
    assert shard_paths, f"no shards for {stem} under {r1417.PARENT_PREFIX}@{rev}"
    todo = [p for p in shard_paths if not (dest / Path(p).name).exists()]
    print(f"[i1417-battery] staging {stem}: {len(todo)}/{len(shard_paths)} shards @ {rev[:10]}")
    with ThreadPoolExecutor(max_workers=6) as ex:
        futs = [
            ex.submit(
                stage_hub_file,
                r1417.HF_DATA_REPO,
                p,
                dest / Path(p).name,
                repo_type="dataset",
                revision=rev,
            )
            for p in todo
        ]
        for f in futs:
            f.result()  # fail loud
    return dest


def release_reference_store(data_dir: Path, stem: str) -> None:
    for p in sorted(turnstore_dir(data_dir).glob(f"{stem}_shard*.pt")):
        p.unlink()
        print(f"[i1417-battery] released {p}")


def _headroom(out_root: Path, need_gb: float, phase: str) -> None:
    from explore_persona_space.orchestrate.preflight import assert_out_root_headroom

    assert_out_root_headroom(out_root, need_gb, phase=phase)


# ---------------------------------------------------------------------------
# XY loading (parent turnstores via fit825's loaders; own stores likewise)
# ---------------------------------------------------------------------------
_BUNDLE_CACHE: dict[tuple, dict] = {}


def load_parent_bundle(data_dir: Path, model: str, fmt: str) -> dict:
    key = ("parent", model, fmt)
    if key not in _BUNDLE_CACHE:
        _BUNDLE_CACHE[key] = fit825._load_bundle_any(
            turnstore_dir(data_dir), model, fmt, "s", wanted_keys=("slots", "profiles", "nll")
        )
    return _BUNDLE_CACHE[key]


def load_own_bundle(data_dir: Path, model: str, cell: str) -> dict:
    key = ("own", model, cell)
    if key not in _BUNDLE_CACHE:
        bundle = fit825._load_bundle_any(
            Path(data_dir) / "store", model, cell, "s", wanted_keys=("slots", "profiles")
        )
        side0 = sorted((Path(data_dir) / "store").glob(f"{model}_{cell}_s_shard*.json"))
        assert side0, f"no sidecars for {model}_{cell}"
        side = json.loads(side0[0].read_text())
        assert r1417.fingerprint_matches(side), (
            f"{model}_{cell}: store fingerprint mismatch — re-run capture"
        )
        _BUNDLE_CACHE[key] = bundle
    return _BUNDLE_CACHE[key]


def evict_bundle(kind: str, model: str, key3: str) -> None:
    _BUNDLE_CACHE.pop((kind, model, key3), None)


def anchor_cell_dict(anchor: dict) -> dict:
    return {**anchor, "track": "s", "slot_index": 0, "target_turn_index": 1}


def own_cell_dict(model: str, cell: str, arm: str) -> dict:
    return {
        "cell_id": f"{cell}__{model}__{arm}",
        "model": model,
        "format": cell,
        "track": "s",
        "slot_index": 0 if arm == "ctx" else 1,
        "target_turn_index": 0,
    }


def kept_conv_ids(out_dir: Path, model: str, cell: str) -> list[str]:
    p = Path(out_dir) / "judge" / f"kept_{model}_{cell}.json"
    assert p.exists(), f"judge kept-set missing: {p} — run issue1417_judge.py first"
    d = json.loads(p.read_text())
    assert r1417.fingerprint_matches(d), f"{p}: fingerprint mismatch"
    return [str(c) for c in d["kept_conv_ids"]]


def _xy_for(bundle: dict, cell_dict: dict, allow: list[str] | None) -> dict:
    cell = fit825._normalize_cell(cell_dict)
    xy = fit825._cell_xy(bundle, cell)
    return fit825._apply_row_allowlist(xy, allow, cell["cell_id"])


# ---------------------------------------------------------------------------
# G1 — reference anchor gate
# ---------------------------------------------------------------------------
def g1_committed_value(anchor_id: str) -> float:
    fname, dotted = r1417.G1_ANCHOR_SOURCE[anchor_id]
    obj = json.loads((r1417.G1_COMMITTED_DIR / fname).read_text())
    for part in dotted.split("/"):
        obj = obj[int(part)] if isinstance(obj, list) else obj[part]
    return float(obj)


def run_anchors(args) -> int:
    """G1: refit this lane's two anchors on the shared row set; ±0.01 @ L19."""
    shared = r1417.shared_conv_ids(args.data_dir)
    out_dir = Path(args.out_dir) / "anchors"
    out_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict] = {}
    worst = 0.0
    for anchor in ANCHORS_BY_MODEL[args.model]:
        stem = f"{anchor['model']}_{anchor['format']}_s"
        bundle = load_parent_bundle(args.data_dir, anchor["model"], anchor["format"])
        allow = shared if not args.smoke else None  # smoke stores carry few rows
        res = fit825.run_cell(
            anchor_cell_dict(anchor),
            turnstore_dir(args.data_dir),
            out_dir,
            n_folds=N_FOLDS,
            seed=FIT_SEED,
            null_draws=args.null_draws,
            n_boot=args.n_boot,
            allowlist=allow,
            bundle=bundle,
        )
        r2 = res["cell_payload"]["r2_per_layer_obs"]
        l19 = float(r2[HEADLINE_LAYER]) if len(r2) > HEADLINE_LAYER else float("nan")
        committed = g1_committed_value(anchor["cell_id"])
        dev = abs(l19 - committed)
        worst = max(worst, dev)
        results[anchor["cell_id"]] = {
            "stem": stem,
            "l19_refit": l19,
            "l19_committed": committed,
            "abs_dev": dev,
            "pass": bool(dev <= r1417.G1_TOL),
        }
        print(f"[i1417-battery] G1 {anchor['cell_id']}: refit={l19:.4f} committed={committed:.4f}")
    payload = {
        "metadata": common931.metadata(SCRIPT, FIT_SEED, len(shared)),
        "gate": "G1",
        "model": args.model,
        "tolerance": r1417.G1_TOL,
        "smoke": bool(args.smoke),
        "anchors": results,
        "pass": bool(all(v["pass"] for v in results.values())),
    }
    (out_dir / f"g1_anchor_gate_{args.model}.json").write_text(
        json.dumps(payload, indent=2, default=float)
    )
    if not payload["pass"]:
        if args.smoke:
            # Gate-calibration parity (#1345): computation ran; verdict is
            # informational at smoke n (a tiny refit cannot hit a 4,724-row anchor).
            print(f"[i1417-battery] G1 INFORMATIONAL at smoke n: worst dev {worst:.4f}")
            return 0
        print(f"[i1417-battery] G1 HALT: worst dev {worst:.4f} > {r1417.G1_TOL}", file=sys.stderr)
        return 20
    print("[i1417-battery] G1 PASS")
    return 0


def gate_g2(args) -> int:
    dev = fit825._fit_device()
    print(f"[i1417-battery] G2 fit device: {dev}")
    if dev.type != "cuda" and not args.allow_cpu:
        print("[i1417-battery] G2 HALT: fit device is not cuda (#1335 P3)", file=sys.stderr)
        return 22
    return 0


# ---------------------------------------------------------------------------
# Fits — per (cell, arm): primary kept-rows + all-rows + matched-n companions
# ---------------------------------------------------------------------------
def compute_n_min(args, models=("instruct", "pretrained")) -> int:
    """n_min across kept (cell, model) sets — the matched-n target."""
    ns = []
    for model in models:
        for cell in r1417.CELL_ORDER:
            p = Path(args.out_dir) / "judge" / f"kept_{model}_{cell}.json"
            if p.exists():
                ns.append(len(json.loads(p.read_text())["kept_conv_ids"]))
    assert ns, "no judge kept-sets found — run the judge phase first"
    return min(ns)


def matched_subsample(ids: list[str], n_target: int, seed: int) -> list[str]:
    """Seeded uniform subsample WITHOUT replacement.

    Deliberate adaptation of the #1335 group-stratified subsample: #1417 is
    single-turn (one row per conv => all groups are singletons), where the
    group-stratified tie-break is seed-DEGENERATE (picks the same rows for
    every seed — agent-memory #931); a plain seeded permutation restores
    across-seed draw variance."""
    rng = np.random.default_rng(seed)
    ids = sorted(ids)
    take = min(n_target, len(ids))
    return [ids[i] for i in rng.choice(len(ids), size=take, replace=False)]


def _fit_path(out_dir: Path, cell_id: str) -> Path:
    return Path(out_dir) / "cells" / f"cells_{cell_id}.json"


def run_fits(args) -> int:
    model = args.model
    shared = set(r1417.shared_conv_ids(args.data_dir))
    cells_dir = Path(args.out_dir) / "cells"
    cells_dir.mkdir(parents=True, exist_ok=True)
    n_min = compute_n_min(args, models=(model,)) if args.smoke else compute_n_min(args)
    print(f"[i1417-battery] fits ({model}): matched-n n_min={n_min}")
    for cell in r1417.CELL_ORDER:
        bundle = load_own_bundle(args.data_dir, model, cell)
        store_ids = [str(c) for c in bundle["sidecar"]["conv_ids"]]
        kept = [c for c in kept_conv_ids(args.out_dir, model, cell) if c in shared]
        kept = [c for c in kept if c in set(store_ids)]
        all_rows = [c for c in store_ids if c in shared]
        variants: list[tuple[str, list[str] | None]] = [
            (f"{cell}__{model}__ctx", kept),
            (f"{cell}__{model}__prefix", kept),
            (f"{cell}__{model}__ctx__all", all_rows),
        ]
        for k in range(MATCHED_DRAWS):
            variants.append(
                (
                    f"{cell}__{model}__ctx__matched{k}",
                    matched_subsample(kept, n_min, seed=r1417.MATCHED_SEED_BASE + k),
                )
            )
        for cell_id, allow in variants:
            if args.resume and _fit_path(args.out_dir, cell_id).exists():
                print(f"[i1417-battery] resume: cells_{cell_id}.json exists — skipped")
                continue
            arm = "prefix" if "__prefix" in cell_id else "ctx"
            cd = own_cell_dict(model, cell, arm)
            cd["cell_id"] = cell_id
            if not allow:
                print(f"[i1417-battery] {cell_id}: EMPTY row set — skipped (reported)")
                _fit_path(args.out_dir, cell_id).write_text(
                    json.dumps({"cell_id": cell_id, "skipped_empty_rows": True})
                )
                continue
            if len(allow) < 2 * N_FOLDS:
                # Designed floor (gotchas #1345 gate-calibration parity): a
                # judge-kept set below ~2 rows/fold cannot support the frozen
                # K=5 conv-grouped recipe — report-and-skip, never crash.
                # Production kept sets are hundreds of rows; only smoke /
                # degenerate-yield cells land here.
                print(
                    f"[i1417-battery] {cell_id}: only {len(allow)} rows "
                    f"(< {2 * N_FOLDS}) — skipped (reported)"
                )
                _fit_path(args.out_dir, cell_id).write_text(
                    json.dumps({"cell_id": cell_id, "skipped_too_few_rows": len(allow)})
                )
                continue
            fit825.run_cell(
                cd,
                Path(args.data_dir) / "store",
                cells_dir,
                n_folds=N_FOLDS,
                seed=FIT_SEED,
                null_draws=args.null_draws,
                n_boot=args.n_boot,
                allowlist=allow,
                bundle=bundle,
            )
        evict_bundle("own", model, cell)
    return 0


# ---------------------------------------------------------------------------
# Identity battery — cell<->reference pairs within model (ref="i", cell="b")
# ---------------------------------------------------------------------------
def battery_pairs(model: str) -> list[dict]:
    pairs: list[dict] = []
    for cell in r1417.CELL_ORDER:
        pairs.append({"cell": cell, "ref": "c0_chat", "arm": "ctx"})
    pairs.append({"cell": "c4_exposition", "ref": "c0p_nat", "arm": "ctx"})
    for cell in ("c2_rude", "c3_evasive", "c5_ai_addressee"):
        pairs.append({"cell": cell, "ref": "c1", "arm": "ctx"})
    for cell in ("c2_rude", "c3_evasive", "c4_exposition", "c5_ai_addressee"):
        pairs.append({"cell": cell, "ref": "c1", "arm": "prefix"})
    for p in pairs:
        p["model"] = model
        p["pair_id"] = f"{model}__{p['cell']}__vs_{p['ref']}__{p['arm']}"
    return pairs


def _ref_xy(args, model: str, ref: str, arm: str) -> tuple[dict, list[str] | None]:
    """Reference-side xy + its kept allowlist (None = all rows)."""
    spec = REFERENCES[ref]
    if spec["kind"] == "parent":
        bundle = load_parent_bundle(args.data_dir, model, spec["format"])
        cd = {"cell_id": f"ref_{ref}_{model}", "model": model, "format": spec["format"]}
        cd = {**cd, "track": "s", "slot_index": 0, "target_turn_index": 1}
        return _xy_for(bundle, cd, None), None
    cell = spec["cell"]
    bundle = load_own_bundle(args.data_dir, model, cell)
    cd = own_cell_dict(model, cell, arm)
    kept = kept_conv_ids(args.out_dir, model, cell)
    return _xy_for(bundle, cd, None), kept


def _align_rows(
    xy_cell: dict, xy_ref: dict, registered: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    """Index arrays aligning both stores to the registered row list.

    Row-coverage assert (plan §6, BINDING): registered pair rows ⊆ BOTH
    stores' conv_id sets, checked BEFORE any alignment/composed statistic."""
    ids_c = [str(c) for c in xy_cell["conv_ids"]]
    ids_r = [str(c) for c in xy_ref["conv_ids"]]
    set_c, set_r = set(ids_c), set(ids_r)
    missing_c = [c for c in registered if c not in set_c]
    missing_r = [c for c in registered if c not in set_r]
    assert not missing_c and not missing_r, (
        f"row-coverage assert failed: {len(missing_c)} registered rows missing from the "
        f"cell store, {len(missing_r)} from the reference store (e.g. "
        f"{(missing_c + missing_r)[:5]})"
    )
    pos_c = {c: i for i, c in enumerate(ids_c)}
    pos_r = {c: i for i, c in enumerate(ids_r)}
    return (
        np.array([pos_c[c] for c in registered]),
        np.array([pos_r[c] for c in registered]),
    )


def _pair_data(xy_cell: dict, xy_ref: dict, ic: np.ndarray, ir: np.ndarray, layers) -> dict:
    """Build the ma battery data dict (ref='i', cell='b') on the fit device."""
    dev = cm._fit_device()

    def _mk(arr: np.ndarray, idx: np.ndarray) -> dict:
        return {
            int(L): torch.as_tensor(arr[idx][:, L, :], dtype=torch.float64).to(dev) for L in layers
        }

    return {
        "Xi": _mk(xy_ref["X"], ir),
        "Yi": _mk(xy_ref["Y"], ir),
        "Xb": _mk(xy_cell["X"], ic),
        "Yb": _mk(xy_cell["Y"], ic),
    }


def _rel_bootstrap(data: dict, folds: np.ndarray, layer: int, *, n_boot: int, seed: int) -> dict:
    """Conversation-level bootstrap CI for REL and ΔREL = REL - 0.5 at one layer.

    Convention (LABELED per plan §6 rule 1): FIXED per-fold maps re-evaluated
    on resampled rows (never per-draw refits); per-draw pooled R^2 uses the
    resampled-set mean (the fit825.bootstrap_r2_ci convention). The point
    estimate quoted beside it comes from the fold-local pooled battery read.
    Draws are batched as one-hot GEMMs (no per-draw Python re-reduction)."""
    Xi, Yi, Xb, Yb = data["Xi"][layer], data["Yi"][layer], data["Xb"][layer], data["Yb"][layer]
    n = Xi.shape[0]
    pred_num = torch.zeros_like(Yi)
    pred_ceil = torch.zeros_like(Yi)
    fitted = np.zeros(n, dtype=bool)
    for k in range(N_FOLDS):
        tr = torch.as_tensor(folds != k)
        te = torch.as_tensor(folds == k)
        if int(te.sum()) == 0 or int(tr.sum()) < 3:
            continue
        preps = {
            "Xi": ma._ridge_prep(Xi[tr]),
            "Xb": ma._ridge_prep(Xb[tr]),
            "Yb": ma._ridge_prep(Yb[tr]),
        }
        xbhat = ma._ridge_predict(preps["Xi"], Xb[tr], Xi[te])  # A_ctx_rev(Xi)
        ybhat = ma._ridge_predict(preps["Xb"], Yb[tr], xbhat)  # M_cell(.)
        pred_num[te] = ma._ridge_predict(preps["Yb"], Yi[tr], ybhat)  # A_ans(.)
        pred_ceil[te] = ma._ridge_predict(preps["Xi"], Yi[tr], Xi[te])  # within-ref
        fitted[np.asarray(folds == k)] = True
        del preps
    Yf = Yi[torch.as_tensor(fitted)]
    rn = ((Yf - pred_num[torch.as_tensor(fitted)]) ** 2).sum(dim=1)  # (Nf,)
    rc = ((Yf - pred_ceil[torch.as_tensor(fitted)]) ** 2).sum(dim=1)
    nf = int(fitted.sum())
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, nf, size=(n_boot, nf))
    onehot = torch.zeros((n_boot, nf), dtype=torch.float64, device=Yf.device)
    counts = np.stack([np.bincount(row, minlength=nf) for row in idx])
    onehot += torch.as_tensor(counts, dtype=torch.float64, device=Yf.device)
    ss_num = onehot @ rn  # (B,)
    ss_ceil = onehot @ rc
    row_sq = (Yf**2).sum(dim=1)  # (Nf,)
    sum_y = onehot @ Yf  # (B, D)
    mean_y = sum_y / nf
    ss_tot = onehot @ row_sq - nf * (mean_y**2).sum(dim=1)
    r2n = 1.0 - ss_num / ss_tot
    r2c = 1.0 - ss_ceil / ss_tot
    rel = (r2n / r2c).cpu().numpy()
    rel = rel[np.isfinite(rel)]
    d = rel - REL_BOUNDARY
    return {
        "convention": "fixed per-fold maps re-evaluated on resampled rows; "
        "resampled-set-mean ss_tot (bootstrap_r2_ci convention); point estimate is "
        "the fold-local pooled battery read",
        "n_boot": int(n_boot),
        "n_rows": nf,
        "rel_mean": float(rel.mean()),
        "rel_ci95": [float(np.quantile(rel, 0.025)), float(np.quantile(rel, 0.975))],
        "delta_rel_ci95": [float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))],
    }


def run_battery(args) -> int:
    model = args.model
    shared = set(r1417.shared_conv_ids(args.data_dir))
    out_dir = Path(args.out_dir) / "battery"
    out_dir.mkdir(parents=True, exist_ok=True)
    pairs = battery_pairs(model)
    budget_s = args.pilot_budget_h * 3600.0
    t_first = None
    for i, pair in enumerate(pairs):
        pid = pair["pair_id"]
        out_path = out_dir / f"battery_{pid}.json"
        if args.resume and out_path.exists():
            print(f"[i1417-battery] resume: {out_path.name} exists — skipped")
            continue
        t0 = time.time()
        cell, ref, arm = pair["cell"], pair["ref"], pair["arm"]
        bundle_c = load_own_bundle(args.data_dir, model, cell)
        xy_cell = _xy_for(bundle_c, own_cell_dict(model, cell, arm), None)
        xy_ref, ref_kept = _ref_xy(args, model, ref, arm)
        kept_c = set(kept_conv_ids(args.out_dir, model, cell))
        rows = [str(c) for c in xy_cell["conv_ids"] if str(c) in kept_c and str(c) in shared]
        if ref_kept is not None:
            rk = set(ref_kept)
            rows = [c for c in rows if c in rk]
        rows = [c for c in rows if c in {str(x) for x in xy_ref["conv_ids"]}]
        if len(rows) < 3 * N_FOLDS:
            out_path.write_text(
                json.dumps({"pair_id": pid, "skipped_too_few_rows": len(rows)}, indent=2)
            )
            print(f"[i1417-battery] {pid}: only {len(rows)} rows — skipped (reported)")
            continue
        ic, ir = _align_rows(xy_cell, xy_ref, rows)
        layers = [L for L in FROZEN_LAYERS if xy_cell["X"].shape[1] > L]
        data = _pair_data(xy_cell, xy_ref, ic, ir, layers)
        folds = fit825._cv_folds(np.asarray(rows), N_FOLDS, FIT_SEED)

        per_layer = {}
        for L in layers:
            per_layer[str(L)] = ma._layer_battery(data, folds, L, do_orth=True)
        hl = HEADLINE_LAYER if HEADLINE_LAYER in layers else layers[-1]
        rel_by_layer = {}
        for L in layers:
            b = per_layer[str(L)]
            num = b["composition"]["linear"].get("comp_samefn_b2i")
            ceil = b["ceilings"].get("within_instruct")
            rev_num = b["composition"]["linear"].get("comp_samefn_i2b")
            rev_ceil = b["ceilings"].get("within_base")
            rel_by_layer[str(L)] = {
                "rel": (num / ceil) if (num is not None and ceil) else float("nan"),
                "rel_reverse": (rev_num / rev_ceil)
                if (rev_num is not None and rev_ceil)
                else float("nan"),
                "numerator_r2": num,
                "ceiling_r2": ceil,
            }
        boot = _rel_bootstrap(data, folds, hl, n_boot=args.n_boot, seed=FIT_SEED + hl)
        cos_null = ma._procrustes_cosine_null(
            data["Xb"][hl],
            data["Xi"][hl],
            data["Yb"][hl],
            data["Yi"][hl],
            n_draws=args.cosine_null_draws,
            seed=FIT_SEED + 7,
        )
        comp_null = ma._composition_collapse_null(
            data, folds, hl, n_draws=args.collapse_null_draws, seed=FIT_SEED + 11
        )
        Lf = layers
        swap_fwd = cm.frozen_map_swap(  # M_ref applied on cell rows
            xy_ref["X"][ir][:, Lf, :],
            xy_ref["Y"][ir][:, Lf, :],
            xy_cell["X"][ic][:, Lf, :],
            xy_cell["Y"][ic][:, Lf, :],
            np.asarray(rows),
            Lf,
            seed=FIT_SEED,
            null_draws=args.null_draws,
        )
        swap_rev = cm.frozen_map_swap(  # M_cell applied on reference rows
            xy_cell["X"][ic][:, Lf, :],
            xy_cell["Y"][ic][:, Lf, :],
            xy_ref["X"][ir][:, Lf, :],
            xy_ref["Y"][ir][:, Lf, :],
            np.asarray(rows),
            Lf,
            seed=FIT_SEED,
            null_draws=args.null_draws,
        )
        payload = {
            "metadata": common931.metadata(SCRIPT, FIT_SEED, len(rows)),
            **r1417.fingerprint(),
            "pair": pair,
            "n_rows": len(rows),
            "headline_layer": hl,
            "battery_per_layer": per_layer,
            "rel_by_layer": rel_by_layer,
            "rel_bootstrap_l19": boot,
            "procrustes_cosine_null_l19": cos_null,
            "composition_collapse_null_l19": comp_null,
            "transfer_ref_on_cell": swap_fwd,
            "transfer_cell_on_ref": swap_rev,
        }
        out_path.write_text(json.dumps(payload, indent=2, default=float))
        dt = time.time() - t0
        rel_hl = rel_by_layer[str(hl)]["rel"]
        print(f"[i1417-battery] {pid}: n={len(rows)} rel@L{hl}={rel_hl:.3f} ({dt:.1f}s)")
        del data
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if i == 0:
            t_first = dt
            projected = t_first * len(pairs)
            report = {
                "gate": "pc3_pilot",
                "per_pair_s": t_first,
                "n_pairs": len(pairs),
                "projected_lane_s": projected,
                "budget_lane_s": budget_s,
                "abort_threshold_s": 2 * budget_s,
                "pass": bool(projected <= 2 * budget_s),
            }
            (out_dir / f"battery_pilot_report_{model}.json").write_text(
                json.dumps(report, indent=2)
            )
            print(f"[i1417-battery] PC-3 pilot: {report}")
            if not report["pass"] and not args.smoke:
                print(
                    "[i1417-battery] PC-3 pilot ABORT: projected lane wall "
                    f"{projected / 3600:.2f}h > 2x budget {args.pilot_budget_h}h",
                    file=sys.stderr,
                )
                return 7
        if (
            args.cycle_stores
            and ref == "c0_chat"
            and i == len([p for p in pairs if p["ref"] == "c0_chat"]) - 1
        ):
            evict_bundle("parent", model, "chat")
    return 0


# ---------------------------------------------------------------------------
# Summary — collapse diagnostics, verdict lattice, H-table lookup
# ---------------------------------------------------------------------------
def _dup_prefix_rate(data_dir: Path, model: str, cell: str, kept: set[str]) -> float:
    rows = g1417._read_jsonl(g1417.gen_path(data_dir, model, cell))
    prefixes: dict[tuple, int] = {}
    n = 0
    for r in rows:
        if r["conv_id"] not in kept:
            continue
        n += 1
        key = tuple(r["completion_token_ids"][:DUP_PREFIX_TOKENS])
        prefixes[key] = prefixes.get(key, 0) + 1
    if n == 0:
        return float("nan")
    dup = sum(c for c in prefixes.values() if c > 1)
    return dup / n


def _cell_json(out_dir: Path, cell_id: str) -> dict | None:
    p = _fit_path(out_dir, cell_id)
    return json.loads(p.read_text()) if p.exists() else None


def _verdict(delta_ci: list[float]) -> str:
    lo, hi = delta_ci
    if lo > 0:
        return "Shared"
    if hi < 0:
        return "Distinct"
    return "Inconclusive"


H_TABLE = {
    ("Distinct", "Shared"): "H1 (helpful-only)",
    ("Shared", "Distinct"): "H2 (user-directed-only)",
    ("Distinct", "Distinct"): "Conjunction (both required)",
    ("Shared", "Shared"): "Neither (generic QA structure)",
}


def _summary_judge_entry(args, model: str, cell: str, entry: dict) -> None:
    kept_p = Path(args.out_dir) / "judge" / f"kept_{model}_{cell}.json"
    if not kept_p.exists():
        return
    kd = json.loads(kept_p.read_text())
    entry["n_kept"] = kd["n_kept"]
    entry["yield_frac"] = kd["yield_frac"]
    entry["primary_grade"] = bool(kd["yield_frac"] >= YIELD_FLOOR)
    entry["dup_prefix_rate"] = _dup_prefix_rate(
        args.data_dir, model, cell, set(kd["kept_conv_ids"])
    )


def _summary_fit_entry(args, model: str, cell: str, anchors_var: dict, entry: dict) -> None:
    cj = _cell_json(args.out_dir, f"{cell}__{model}__ctx")
    if not cj or "r2_per_layer_obs" not in cj:
        return
    l19 = float(cj["r2_per_layer_obs"][HEADLINE_LAYER])
    null_p975 = None
    nj = Path(args.out_dir) / "cells" / f"nulls_{cell}__{model}__ctx.json"
    if nj.exists():
        nd = json.loads(nj.read_text())
        col = [row[HEADLINE_LAYER] for row in nd["null_matrix"]]
        null_p975 = float(np.quantile(np.asarray(col), 0.975))
    entry["r2_l19"] = l19
    entry["null_p975_l19"] = null_p975
    entry["map_exists"] = (
        bool(l19 >= (null_p975 + MAP_EXISTS_MARGIN)) if null_p975 is not None else None
    )
    yv = float(cj["y_trace_cov_frozen"][str(HEADLINE_LAYER)])
    entry["y_trace_cov_l19"] = yv
    if model in anchors_var and anchors_var[model] > 0:
        entry["y_var_ratio_vs_c0"] = yv / anchors_var[model]


def _summary_battery_entry(args, model: str, cell: str, entry: dict) -> None:
    bj = Path(args.out_dir) / "battery" / f"battery_{model}__{cell}__vs_c0_chat__ctx.json"
    if bj.exists():
        bd = json.loads(bj.read_text())
        if "rel_by_layer" in bd:
            hl = str(bd["headline_layer"])
            entry["rel_l19"] = bd["rel_by_layer"][hl]["rel"]
            entry["delta_rel_ci95"] = bd["rel_bootstrap_l19"]["delta_rel_ci95"]
            entry["verdict_raw"] = _verdict(entry["delta_rel_ci95"])
    # C4 reference-discordance rule (analyzer rule 3): the format-matched
    # C0' pair governs the H2 narration.
    if cell == "c4_exposition":
        bj2 = Path(args.out_dir) / "battery" / f"battery_{model}__{cell}__vs_c0p_nat__ctx.json"
        if bj2.exists():
            bd2 = json.loads(bj2.read_text())
            if "rel_by_layer" in bd2:
                hl2 = str(bd2["headline_layer"])
                entry["rel_l19_vs_c0p"] = bd2["rel_by_layer"][hl2]["rel"]
                entry["delta_rel_ci95_vs_c0p"] = bd2["rel_bootstrap_l19"]["delta_rel_ci95"]
                entry["verdict_vs_c0p"] = _verdict(entry["delta_rel_ci95_vs_c0p"])


def _summary_verdict(cell: str, entry: dict) -> str | None:
    """Plan §6 demotions: a Distinct verdict needs own-map existence + intact
    content (variance-ratio / duplicate-prefix checks)."""
    v = entry.get("verdict_vs_c0p") if cell == "c4_exposition" else None
    v = v or entry.get("verdict_raw")
    if v == "Distinct":
        if entry.get("map_exists") is False:
            v = "no-map (#1310 distinction)"
        collapsed = (
            entry.get("y_var_ratio_vs_c0") is not None
            and entry["y_var_ratio_vs_c0"] < COLLAPSE_VAR_RATIO
        ) or (
            entry.get("dup_prefix_rate") is not None
            and not np.isnan(entry.get("dup_prefix_rate", float("nan")))
            and entry["dup_prefix_rate"] > COLLAPSE_DUP_RATE
        )
        if collapsed:
            v = "content-collapsed (demoted)"
    return v


def run_summary(args) -> int:
    out = {"metadata": common931.metadata(SCRIPT, FIT_SEED, 0), **r1417.fingerprint()}
    cells_summary: dict[str, dict] = {}
    anchors_var: dict[str, float] = {}
    for model in r1417.MODELS:
        anchor_id = "S1" if model == "instruct" else "S2"
        aj = Path(args.out_dir) / "anchors" / f"cells_{anchor_id}.json"
        if aj.exists():
            anchors_var[model] = float(
                json.loads(aj.read_text())["y_trace_cov_frozen"][str(HEADLINE_LAYER)]
            )
    for model in r1417.MODELS:
        for cell in r1417.CELL_ORDER:
            entry: dict = {}
            _summary_judge_entry(args, model, cell, entry)
            _summary_fit_entry(args, model, cell, anchors_var, entry)
            _summary_battery_entry(args, model, cell, entry)
            entry["verdict"] = _summary_verdict(cell, entry)
            cells_summary[f"{model}__{cell}"] = entry
    h_lookup = {}
    for model in r1417.MODELS:
        vc2 = cells_summary.get(f"{model}__c2_rude", {}).get("verdict")
        vc4 = cells_summary.get(f"{model}__c4_exposition", {}).get("verdict")
        if vc2 in ("Shared", "Distinct") and vc4 in ("Shared", "Distinct"):
            h_lookup[model] = H_TABLE[(vc2, vc4)]
        elif vc2 == "content-collapsed (demoted)" and vc4 in ("Shared", "Distinct"):
            h_lookup[model] = "unresolved — content confound (C2 collapsed)"
        else:
            h_lookup[model] = "Inconclusive — report graded REL profile"
    out["cells"] = cells_summary
    out["h_table_lookup"] = h_lookup
    out["conventions"] = {
        "rel": "composition.linear.comp_samefn_b2i / ceilings.within_instruct "
        "(ref='i', cell='b'; ceiling recomputed on the SAME kept∩kept rows + shared "
        "folds as the numerator — plan §6 rule 1)",
        "verdict": "Shared iff ΔREL 95% CI wholly > 0; Distinct iff wholly < 0",
    }
    p = Path(args.out_dir) / "battery_summary.json"
    p.write_text(json.dumps(out, indent=2, default=float))
    print(f"[i1417-battery] wrote {p}")
    return 0


def main() -> int:
    args = parse_args()
    if args.gate_g2:
        rc = gate_g2(args)
        if rc:
            return rc
    if args.stage_stores:
        assert args.model, "--stage-stores needs --model"
        _headroom(args.data_dir, 50.0, phase=f"stage-stores-{args.model}")
        for fmt in ("chat", "naturalistic"):
            stage_reference_store(args.data_dir, f"{args.model}_{fmt}_s")
    if args.anchors:
        assert args.model, "--anchors needs --model"
        rc = run_anchors(args)
        if rc:
            return rc
    if args.fits:
        assert args.model, "--fits needs --model"
        rc = run_fits(args)
        if rc:
            return rc
    if args.battery:
        assert args.model, "--battery needs --model"
        rc = run_battery(args)
        if rc:
            return rc
    if args.summary:
        rc = run_summary(args)
        if rc:
            return rc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
