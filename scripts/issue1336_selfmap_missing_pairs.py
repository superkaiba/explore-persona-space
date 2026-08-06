"""Fill the two #1336 tier-grid gaps: the BASE self map + three missing forward pairs.

Round-5 part C+D (user decision): the plotted source->target grid lacks
(a) the WITHIN-stage map for ``base`` (base appears only as a SOURCE), and
(b) three forward pairs the registry never ran — sft->rlvr, sft->rlvr_long,
rlvr->rlvr_long — at tiers 0 and 6. This script computes exactly those cells
on all 8 v2 corpus-format surfaces at layer 30, on the MATCHED pair-file
basis: every estimator step is the PRODUCTION chain imported verbatim —
``issue1336_metric_ladder._load_surface_xy`` (loader, incl. the concat
wave-1 + extension stems), ``issue1336_ladder_alignment._align_rows`` (row
intersection), ``issue825_fit_cells._cv_folds`` (seed-0 conversation-grouped
5-fold), ``issue1336_metric_ladder._v2_prep/_v2_yfit/_v2_predict`` (the v2
inner-group-CV ridge with the dof-capped GCV fallback), and the ladder's
FOLD-LOCAL pooled out-of-fold R^2 accumulation (per-fold test-mean centering
summed over folds — the basis the plotted tier/within values use; the
globally-centered ``fc._pooled_r2`` companion is reported separately). The
standalone ``cells_v2`` per-stage basis reads 0.105-0.198 LOWER on all 56
comparable cells and is deliberately NOT used.

The three missing pairs cannot go through ``run_pair`` (it asserts membership
in ``cm.PAIRS``), so this drives the underlying chain directly, following the
``issue1336_identity_spotcheck.py`` shape.

Estimator validity: d = 4096; the chat/gsm8k_test1319 surface has n = 1319
(n_train ~ 0.8n < d) — fitted per the user's explicit decision, with every
emitted record carrying machine-readable ``degenerate_n_lt_d`` + realized
n_train and d. Selected ridge lambdas + selector provenance are reported per
fit per fold. The shared #825 fit core's default dof-capped GCV is inherited
(LEGACY_UNGUARDED_GCV stays False).

Example (pod, full grid):
    uv run python scripts/issue1336_selfmap_missing_pairs.py --stage \\
        --stage-root data/issue_1336 \\
        --out-root eval_results/issue_1336/selfmap_missing_pairs

Example (VM pilot on the staged identity cell — no staging, one cell):
    uv run python scripts/issue1336_selfmap_missing_pairs.py --pilot \\
        --turnstore-dir /mnt/eps-data/$USER/issue1336_identity/ts_flat \\
        --out-root /tmp/issue1336_selfmap_pilot
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

import issue825_fit_cells as fc  # noqa: E402
import issue1336_ladder_alignment as la  # noqa: E402
import issue1336_metric_ladder as ml  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from explore_persona_space.experiments.issue_1336 import common as cm  # noqa: E402

# Version tag for the resume predicate: bump on ANY output-affecting algebra
# change so stale per-cell checkpoints refit instead of being reused.
ALGEBRA_VERSION = "v2-foldlocal-1"

# The self-map stage (all 8 surfaces) + the three missing forward pairs
# (all 8 surfaces, tiers 0 + 6). Named HERE, never added to cm.PAIRS
# (run_pair's registry stays the 7-pair record of what the ladder ran).
SELF_PAIRS: tuple[tuple[str, str], ...] = (("base", "base"),)
MISSING_PAIRS: tuple[tuple[str, str], ...] = (
    ("sft", "rlvr"),
    ("sft", "rlvr_long"),
    ("rlvr", "rlvr_long"),
)

# The one cell fully covered by the VM-staged identity data (2.6 GB):
# base+dpo chat/gsm8k_test1319 flat shards — enough for the base self map.
PILOT_CELL = ("base", "base", "chat", "gsm8k_test1319")


def _cell_key(source: str, target: str, fmt: str, corpus: str) -> str:
    return f"{source}__{target}__{fmt}__{corpus}"


def enumerate_cells() -> list[tuple[str, str, str, str]]:
    """All (source, target, fmt, corpus) compute cells, registry surface order."""
    for pair in MISSING_PAIRS:
        assert pair not in cm.PAIRS, (
            f"pair {pair} is now in cm.PAIRS — run it through run_pair (full battery) "
            "instead of this gap-filler"
        )
    cells = []
    for source, target in (*SELF_PAIRS, *MISSING_PAIRS):
        for corpus, fmt in cm.v2_surfaces():
            cells.append((source, target, fmt, corpus))
    return cells


def _grid() -> np.ndarray:
    return np.asarray(cm.LAMBDAS_23, dtype=np.float64)


def _grid_sha(grid: np.ndarray) -> str:
    return hashlib.sha256(grid.tobytes()).hexdigest()[:16]


def _resume_key(source: str, target: str, fmt: str, corpus: str, layer: int) -> dict:
    """Every output-affecting regime key (code-style resume contract)."""
    return {
        "source": source,
        "target": target,
        "format": fmt,
        "corpus": corpus,
        "layer": int(layer),
        "fit_seed": int(cm.FIT_SEED),
        "n_folds": int(cm.N_FOLDS),
        "n_inner": int(cm.N_INNER_LAMBDA_FOLDS_V2),
        "grid_sha": _grid_sha(_grid()),
        "algebra_version": ALGEBRA_VERSION,
    }


def _git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10
        ).stdout.strip()
        return out or "unknown"
    except Exception:
        return "unknown"


def _metadata() -> dict:
    return {
        "git_commit": _git_commit(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "script": "scripts/issue1336_selfmap_missing_pairs.py",
        "torch": torch.__version__,
        "numpy": np.__version__,
        "device": str(fc._fit_device()),
    }


# ---------------------------------------------------------------------------
# Loading (production loader, layer-sliced immediately to bound peak RSS)
# ---------------------------------------------------------------------------
def load_model_slice(
    model: str,
    fmt: str,
    corpus: str,
    *,
    layer: int,
    ts_dir: Path,
    wave1_dir: Path,
    gen_root: Path | None,
) -> dict:
    """One model's (ids, x, y) at ``layer`` via the production surface loader.

    ``ml._load_surface_xy`` routes concat corpora (lmsys23k, gsm8k_train_full)
    through ``et.load_bundle_concat`` (wave-1 stem + extension stem, boundary
    + text-sha joins) and everything else through ``fc._load_bundle_any`` —
    exactly the non-smoke ``run_pair`` path. The full (n, L, D) arrays are
    sliced to the single fit layer right away and freed.
    """
    t0 = time.monotonic()
    xy = ml._load_surface_xy(
        ts_dir,
        model,
        fmt,
        corpus,
        smoke=False,
        wave1_dir=wave1_dir,
        gen_root=gen_root,
        expected_layers=cm.EXPECTED_LAYERS,
    )
    assert layer < xy["X"].shape[1], f"layer {layer} out of range ({xy['X'].shape[1]} layers)"
    out = {
        "ids": np.asarray([str(c) for c in xy["conv_ids"]]),
        "x": np.asarray(xy["X"][:, layer, :], dtype=np.float32),
        "y": np.asarray(xy["Y"][:, layer, :], dtype=np.float32),
    }
    del xy
    print(
        f"[selfmap] loaded {cm.cell_id(model, fmt, corpus)}: n={len(out['ids'])} "
        f"d={out['x'].shape[1]} ({time.monotonic() - t0:.1f}s)",
        flush=True,
    )
    return out


# ---------------------------------------------------------------------------
# One cell's fits (verbatim production fold loop, tiers within/t0/t6 only)
# ---------------------------------------------------------------------------
def fit_cell(src: dict, tgt: dict, *, is_self: bool) -> dict:
    """OOF within (+ t0/t6 for pairs) on the matched pair-file basis.

    Mirrors ``run_battery_arrays``'s fold loop for the three consumed maps:
    per-fold ``_v2_prep`` at ``inner_seed = FIT_SEED + 4242 + k``, ``_v2_yfit``
    lambda selection on ``cm.LAMBDAS_23``, fold-local pooled accumulation
    ``r2 = 1 - sum_k ss_res_k / sum_k ss_tot_k`` with ``ss_tot_k`` centered on
    the fold's OWN test mean (the plotted-basis convention). The
    globally-centered pooled read is accumulated alongside as a companion.
    """
    grid = _grid()
    n_folds, seed, n_inner = cm.N_FOLDS, cm.FIT_SEED, cm.N_INNER_LAMBDA_FOLDS_V2
    common, i_s, i_t = la._align_rows(src["ids"], tgt["ids"])
    dev = fc._fit_device()
    dtype = torch.float64
    Xt = torch.as_tensor(tgt["x"][i_t], dtype=dtype).to(dev)
    Yt = torch.as_tensor(tgt["y"][i_t], dtype=dtype).to(dev)
    if is_self:
        Xs = Ys = None
    else:
        Xs = torch.as_tensor(src["x"][i_s], dtype=dtype).to(dev)
        Ys = torch.as_tensor(src["y"][i_s], dtype=dtype).to(dev)
    n, d = int(Xt.shape[0]), int(Xt.shape[1])
    folds = fc._cv_folds(np.asarray(common), n_folds, seed)

    names = ("within",) if is_self else ("within", "t0", "t6")
    ss_res = dict.fromkeys(names, 0.0)
    ss_tot_local = 0.0
    ss_tot_global = 0.0
    yt_global_mu = Yt.mean(0)
    fitted = np.zeros(n, dtype=bool)
    lam_log: dict[str, list[float]] = {}
    sel_log: dict[str, list[str]] = {}
    n_train_per_fold: list[int] = []

    for k in range(n_folds):
        tr_np = folds != k
        te_np = folds == k
        if te_np.sum() == 0 or tr_np.sum() < 3:
            continue
        tr = torch.as_tensor(tr_np)
        te = torch.as_tensor(te_np)
        inner_seed = seed + 4242 + k
        n_train_per_fold.append(int(tr_np.sum()))

        prep_t = ml._v2_prep(Xt[tr], inner_seed=inner_seed, n_inner=n_inner)
        fit_within = ml._v2_yfit(prep_t, Yt[tr], grid)
        preds = {"within": ml._v2_predict(prep_t, fit_within, Xt[te])}
        lam_log.setdefault("within", []).append(float(fit_within["lam"]))
        sel_log.setdefault("within", []).append(fit_within["selector"])

        if not is_self:
            prep_s = ml._v2_prep(Xs[tr], inner_seed=inner_seed, n_inner=n_inner)
            # Tier 0: W_s (fit on the SOURCE stage) applied to the target's x.
            fit_ws = ml._v2_yfit(prep_s, Ys[tr], grid)
            preds["t0"] = ml._v2_predict(prep_s, fit_ws, Xt[te])
            # Tier 6: W_s o A_ctx_rev with train-mean recentering.
            fit_actx = ml._v2_yfit(prep_t, Xs[tr], grid)  # A_ctx_rev: x_t -> x_s
            xhat_tr = ml._v2_predict(prep_t, fit_actx, Xt[tr])
            xhat_te = ml._v2_predict(prep_t, fit_actx, Xt[te])
            raw6_tr = ml._v2_predict(prep_s, fit_ws, xhat_tr)
            raw6_te = ml._v2_predict(prep_s, fit_ws, xhat_te)
            preds["t6"] = raw6_te + (Yt[tr].mean(0) - raw6_tr.mean(0))
            lam_log.setdefault("W_s", []).append(float(fit_ws["lam"]))
            sel_log.setdefault("W_s", []).append(fit_ws["selector"])
            lam_log.setdefault("A_ctx_rev", []).append(float(fit_actx["lam"]))
            sel_log.setdefault("A_ctx_rev", []).append(fit_actx["selector"])
            del prep_s

        yt_te = Yt[te]
        for name, pred in preds.items():
            ss_res[name] += float(((yt_te - pred) ** 2).sum())
        ss_tot_local += float(((yt_te - yt_te.mean(0)) ** 2).sum())
        ss_tot_global += float(((yt_te - yt_global_mu) ** 2).sum())
        fitted[te_np] = True
        del prep_t, preds
        if dev.type == "cuda":
            torch.cuda.empty_cache()

    assert fitted.all(), f"unfitted rows: {int((~fitted).sum())} of {n}"
    return {
        "n": n,
        "d": d,
        "n_train_per_fold": n_train_per_fold,
        "n_train_min": min(n_train_per_fold),
        "n_train_max": max(n_train_per_fold),
        "degenerate_n_lt_d": bool(min(n_train_per_fold) < d),
        "r2": {name: 1.0 - ss_res[name] / ss_tot_local for name in names},
        "r2_globalmu": {name: 1.0 - ss_res[name] / ss_tot_global for name in names},
        "selected_lambda": lam_log,
        "selectors": sel_log,
    }


def cell_records(source: str, target: str, fmt: str, corpus: str, layer: int, fit: dict) -> list:
    """Flatten one cell's fit into per-(tier) records (the emitted schema)."""
    base = {
        "pair": f"{source}__{target}",
        "source": source,
        "target": target,
        "format": fmt,
        "corpus": corpus,
        "layer": int(layer),
        "n": fit["n"],
        "d": fit["d"],
        "n_train_min": fit["n_train_min"],
        "n_train_max": fit["n_train_max"],
        "n_folds": int(cm.N_FOLDS),
        "fit_seed": int(cm.FIT_SEED),
        "degenerate_n_lt_d": fit["degenerate_n_lt_d"],
        "within_r2": fit["r2"]["within"],
        "within_r2_globalmu": fit["r2_globalmu"]["within"],
        "selected_lambda": fit["selected_lambda"],
        "selectors": fit["selectors"],
        "r2_basis": "fold-local pooled OOF (plotted pair-file basis)",
    }
    if source == target:
        return [
            {
                **base,
                "tier": None,
                "r2": fit["r2"]["within"],
                "r2_globalmu": fit["r2_globalmu"]["within"],
            }
        ]
    return [
        {
            **base,
            "tier": t,
            "r2": fit["r2"][f"t{t}"],
            "r2_globalmu": fit["r2_globalmu"][f"t{t}"],
        }
        for t in (0, 6)
    ]


# ---------------------------------------------------------------------------
# Staging (pod use; canonical scoped-listing helpers, never snapshot_download)
# ---------------------------------------------------------------------------
def stage_inputs(cells: list[tuple[str, str, str, str]], args) -> None:
    """Stage every turnstore stem + wave-1 gen answers the cells consume.

    Reuses ``issue1336_diagnose_g1._stage_prefix`` (scoped ``list_repo_tree``
    + per-file retried ``hf_hub_download``) and the dispatcher's c_stage
    layout: extension/single stems flat into ``turnstore_v2/`` @ main; wave-1
    stems + wave-1 gen answers @ ``cm.WAVE1_HF_REV`` into ``turnstore_wave1/``
    and ``gen/<model>/<corpus>/``. Concat corpus rows self-resolve inside
    ``load_v2_corpus_rows`` (its own Hub fallback) — not staged here.
    """
    import issue1336_diagnose_g1 as dg

    api, dl, hub = dg._hub_helpers()
    tmp = args.stage_root / "selfmap_stage_tmp"
    ts_jobs: dict[tuple[str, str], Path] = {}
    gen_jobs: set[tuple[str, str]] = set()
    for source, target, fmt, corpus in cells:
        for m in dict.fromkeys((source, target)):
            ts_jobs[(cm.cell_id(m, fmt, corpus), "main")] = args.turnstore_dir
            if corpus in cm.V2_CONCAT_SOURCES:
                w1 = cm.V2_CONCAT_SOURCES[corpus]
                ts_jobs[(cm.cell_id(m, fmt, w1), cm.WAVE1_HF_REV)] = args.wave1_turnstore_dir
                gen_jobs.add((m, w1))
    for (stem, rev), dest in ts_jobs.items():
        dest.mkdir(parents=True, exist_ok=True)
        if any(dest.glob(f"{stem}_shard*.pt")) or (dest / f"{stem}.npz").exists():
            print(f"[stage] turnstore {stem}: already staged", flush=True)
            continue
        staged = dg._stage_prefix(
            api,
            hub,
            dl,
            f"{cm.HF_PREFIX_1336}/analysis_tensors/turnstore_{stem}",
            tmp,
            revision=rev,
        )
        assert staged, f"no files staged for turnstore {stem} @ {rev}"
        for f in staged:
            f.rename(dest / f.name)
        print(f"[stage] turnstore {stem} @ {rev}: {len(staged)} files -> {dest}", flush=True)
    for m, c in sorted(gen_jobs):
        target_dir = args.gen_root / m / c
        if (target_dir / "answers.jsonl").exists():
            print(f"[stage] wave-1 gen {m}/{c}: already staged", flush=True)
            continue
        staged = dg._stage_prefix(
            api,
            hub,
            dl,
            f"{cm.HF_PREFIX_1336}/raw_completions/generation/{m}/{c}",
            tmp,
            revision=cm.WAVE1_HF_REV,
        )
        assert staged, f"no wave-1 gen files under {m}/{c} @ {cm.WAVE1_HF_REV}"
        target_dir.mkdir(parents=True, exist_ok=True)
        for f in staged:
            f.rename(target_dir / f.name)
        dg._maybe_reassemble_answers(target_dir)
        print(f"[stage] wave-1 gen {m}/{c}: {len(staged)} files", flush=True)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=float) + "\n")
    tmp.replace(path)


def run_cells(cells: list[tuple[str, str, str, str]], args) -> list[dict]:
    """Per-cell checkpointed fits, grouped by surface so each model loads once."""
    cells_dir = args.out_root / "cells"
    cells_dir.mkdir(parents=True, exist_ok=True)
    all_records: list[dict] = []
    done = 0
    # Group by surface; load each needed model's layer slice once per surface.
    surfaces: dict[tuple[str, str], list[tuple[str, str, str, str]]] = {}
    for cell in cells:
        surfaces.setdefault((cell[2], cell[3]), []).append(cell)
    total = len(cells)
    for (fmt, corpus), surface_cells in surfaces.items():
        pending = []
        for cell in surface_cells:
            source, target, _, _ = cell
            key = _cell_key(*cell)
            cell_path = cells_dir / f"{key}__l{args.layer}.json"
            rk = _resume_key(source, target, fmt, corpus, args.layer)
            if cell_path.exists():
                prior = json.loads(cell_path.read_text())
                if prior.get("resume_key") == rk:
                    done += 1
                    print(
                        f"[selfmap] cell {done}/{total} {key} resumed (checkpoint match)",
                        flush=True,
                    )
                    all_records.extend(prior["records"])
                    continue
                print(f"[selfmap] {key}: stale checkpoint (regime changed) — refitting", flush=True)
            pending.append((cell, cell_path, rk))
        if not pending:
            continue
        models = list(dict.fromkeys(m for (c, _, _) in pending for m in (c[0], c[1])))
        slices = {
            m: load_model_slice(
                m,
                fmt,
                corpus,
                layer=args.layer,
                ts_dir=args.turnstore_dir,
                wave1_dir=args.wave1_turnstore_dir,
                gen_root=args.gen_root,
            )
            for m in models
        }
        for cell, cell_path, rk in pending:
            source, target = cell[0], cell[1]
            key = _cell_key(*cell)
            t0 = time.monotonic()
            fit = fit_cell(slices[source], slices[target], is_self=(source == target))
            records = cell_records(source, target, fmt, corpus, args.layer, fit)
            elapsed = time.monotonic() - t0
            _write_json(
                cell_path,
                {
                    "resume_key": rk,
                    "records": records,
                    "elapsed_s": elapsed,
                    "metadata": _metadata(),
                },
            )
            done += 1
            print(f"[selfmap] cell {done}/{total} {key} elapsed={elapsed:.1f}s", flush=True)
            all_records.extend(records)
        del slices
    return all_records


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out-root", type=Path, required=True, help="per-cell + aggregate JSON root")
    ap.add_argument(
        "--stage-root",
        type=Path,
        default=Path("data/issue_1336"),
        help="root for staged inputs (turnstore_v2/, turnstore_wave1/, gen/ live under it)",
    )
    ap.add_argument(
        "--turnstore-dir",
        type=Path,
        default=None,
        help="override the v2/single-stem turnstore dir (default <stage-root>/turnstore_v2)",
    )
    ap.add_argument(
        "--wave1-turnstore-dir",
        type=Path,
        default=None,
        help="override the wave-1 stem dir (default <stage-root>/turnstore_wave1)",
    )
    ap.add_argument(
        "--gen-root",
        type=Path,
        default=None,
        help="override the wave-1 gen-answers root (default <stage-root>/gen)",
    )
    ap.add_argument("--layer", type=int, default=30)
    ap.add_argument(
        "--cells",
        type=str,
        default=None,
        help="comma list of cell keys source__target__format__corpus (default: all 32)",
    )
    ap.add_argument(
        "--smoke",
        action="store_true",
        help="run ONE cell end-to-end at production shape (base self map, chat/gsm8k_test1319)",
    )
    ap.add_argument(
        "--pilot",
        action="store_true",
        help="like --smoke, and also write a measured 1-cell timing basis (pilot.json)",
    )
    ap.add_argument(
        "--stage",
        action="store_true",
        help="stage missing turnstore stems + wave-1 gen answers from the HF data repo first",
    )
    args = ap.parse_args()
    args.turnstore_dir = args.turnstore_dir or (args.stage_root / "turnstore_v2")
    args.wave1_turnstore_dir = args.wave1_turnstore_dir or (args.stage_root / "turnstore_wave1")
    args.gen_root = args.gen_root or (args.stage_root / "gen")

    cells = enumerate_cells()
    if args.smoke or args.pilot:
        cells = [PILOT_CELL]
    elif args.cells:
        wanted = {k.strip() for k in args.cells.split(",") if k.strip()}
        by_key = {_cell_key(*c): c for c in cells}
        unknown = wanted - set(by_key)
        assert not unknown, f"unknown cell keys: {sorted(unknown)} (known: {sorted(by_key)})"
        cells = [by_key[k] for k in sorted(wanted)]
    print(
        f"[selfmap] {len(cells)} cells | layer={args.layer} device={fc._fit_device()} "
        f"| ts={args.turnstore_dir} wave1={args.wave1_turnstore_dir} gen={args.gen_root}",
        flush=True,
    )

    if args.stage:
        stage_inputs(cells, args)

    t0 = time.monotonic()
    records = run_cells(cells, args)
    wall = time.monotonic() - t0

    agg = {
        "records": records,
        "n_cells": len(cells),
        "layer": int(args.layer),
        "wall_s": wall,
        "metadata": _metadata(),
    }
    agg_path = args.out_root / f"records_l{args.layer}.json"
    _write_json(agg_path, agg)
    print(f"[selfmap] wrote {agg_path} ({len(records)} records, wall={wall:.1f}s)", flush=True)

    if args.pilot:
        pilot_path = args.out_root / "pilot.json"
        _write_json(
            pilot_path,
            {
                "cell": _cell_key(*PILOT_CELL),
                "layer": int(args.layer),
                "measured_wall_s": wall,
                "records": records,
                "metadata": _metadata(),
                "note": (
                    "measured 1-cell wall through the production entrypoint at production "
                    "shape (load + 5-fold within fit); sizing basis for the 32-cell grid"
                ),
            },
        )
        print(f"[selfmap] pilot basis written: {pilot_path} (wall={wall:.1f}s)", flush=True)

    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(0)


if __name__ == "__main__":
    main()
