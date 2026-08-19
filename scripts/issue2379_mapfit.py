"""Issue #2379 P5 — map fits + predictor score table (plan §4.2 P5, v6).

Registered design (plan v6):
  * Fit backend: the VERBATIM #2254/#1615 NumPy cores — ``ridge_fit_matrix``,
    ``kstar_from_fit``, ``map_svd`` imported from ``scripts/issue2254_preimage.py``
    (never re-implemented; estimator parity) — run UNCHANGED on CPU under an
    ACROSS-CELL process pool (shard axis = map set x layer). NO GPU SVD leg.
  * Cells: 9 map sets (base = reused #779 pass-B bundle at the pinned rev,
    loaded via ``_load_pass_b_bundle`` — its own asserts are the realized-keys
    verification; + 8 per-model map_corpus bundles) x 28 layers = 252 fits.
    90/10 split (n_train = 4,500 > d = 3,584 well-posedness ASSERTED), GCV
    lambda, SVD k*, fp64.
  * Prediction path (registered formula): v_hat_A = ((v_C - xmu)/xsd) @ W + ymu.
    ALL FOUR components {W, xmu, xsd, ymu} persisted per map/layer; held-out
    prediction-parity assert vs the issue2254_preimage reference path
    (lines ~848-849 pattern) from the SAME stored components. ``--phase pilot``
    runs one fit at FULL production shape on the pass-B bundle + the parity
    assert and prints the measured per-fit wall (the P0 smoke's fence basis).
  * Diagnostics per map/layer: held-out R^2 (``r2_score_multi`` convention),
    ``analysis/mapping_baselines.identity_bias_predict`` (d_in == d_out == 3584),
    ``knn_retrieval`` (k=10, euclidean + cosine, chance = 10/n_heldout stated),
    lambda, k* -> eval_results/issue_2379/predictors/map_diagnostics.json.
  * Predictor score table (P5.4) -> predictors/predictor_scores.json.
  * Pinned-layer components (L14/L16/L27) staged as .pt and uploaded to
    ``issue2379_reelicit/analysis_tensors/maps_pinned/`` (one folder commit);
    non-pinned components stay pod-local (plan §10 discarded_artifacts slot —
    regen recipe = deterministic refit from the uploaded bundles).

Phases (``--phase``): pilot | fits | scores | upload | all (fits+scores+upload)
| smoke (synthetic tiny end-to-end on CPU — n=60, d=8, 2 layers; exercises the
full fit -> persist -> predict -> parity -> score-table path with NO downloads).

BLAS threading: the fits/pilot phases hard-set OMP/MKL/OPENBLAS/NUMEXPR=1
BEFORE the first numpy import (workers x 1 thread = pod vCPUs; the pilot
measures at the SAME per-worker thread config the fan-out realizes). The
scores phase leaves the env alone and does its matmuls via torch
(``--device cuda`` on the pod for the trivial predicted-vector matmuls).

Checkpointing: one .npz per (map set, layer) written atomically the moment the
fit completes; resume skips units whose persisted meta matches (generating
params only — mapset/layer/n/split-seed; never float hashes). Per-unit
progress line: ``[fits] unit k/N <mapset>_L<ly> elapsed=<s>s``.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def _ensure_repo_root_on_syspath() -> Path:
    root = Path(__file__).resolve().parents[1]
    sentinel = root / "src" / "explore_persona_space"
    if not sentinel.is_dir():
        raise RuntimeError(f"repo-root resolution failed: {sentinel} not found (parents[1]={root})")
    for p in (str(root), str(root / "src"), str(root / "scripts")):
        if p not in sys.path:
            sys.path.insert(0, p)
    return root


REPO_ROOT = _ensure_repo_root_on_syspath()

logger = logging.getLogger("issue2379_mapfit")

SLUG = "issue2379_reelicit"
HF_MAP_CORPUS_PREFIX = f"{SLUG}/analysis_tensors/map_corpus"
HF_MAPS_PINNED_PREFIX = f"{SLUG}/analysis_tensors/maps_pinned"
PINNED_LAYERS = (14, 16, 27)  # plan §10: L16 EM / L27 caps pins + L14 map-line frozen layer
SPLIT_SEED = 2379  # 90/10 held-out split (generating-params resume key; recorded in outputs)
HELDOUT_FRAC = 0.10
KNN_K = 10  # plan §4.2 P5.2 headline k (chance = 10/n_heldout, stated in-report)
EXPECTED_LAYERS = 28
EXPECTED_HIDDEN = 3584
BASE_MAPSET = "base"

# Fit phases run 1 BLAS thread per pool worker (width = vCPUs). Hard-set, not
# setdefault: the fence arithmetic (252 x measured_wall / width) is only valid
# when pilot + fan-out share one thread config. Applied in main() BEFORE any
# numpy import (OpenBLAS reads env at library load).
_FIT_BLAS_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def _set_fit_blas_threads(n: int) -> None:
    assert "numpy" not in sys.modules, "BLAS env must be set before the first numpy import"
    for v in _FIT_BLAS_VARS:
        os.environ[v] = str(n)


def _git_meta() -> dict:
    from explore_persona_space.orchestrate.provenance import as_metadata_dict, git_provenance

    return as_metadata_dict(git_provenance(cwd=REPO_ROOT))


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Bundle loading (local-first -> HF fallback -> fail loud)
# ---------------------------------------------------------------------------
def _torch_load_cpu(path: Path) -> dict:
    import torch

    # Unit-2 bundles are tensors + primitive containers only -> weights_only=True.
    return torch.load(path, map_location="cpu", weights_only=True)


def _fetch_hf_bundle(path_in_repo: str, dest: Path) -> Path:
    """HF data-repo fallback for a pod-produced bundle (P3/P4 upload-before-P5)."""
    from huggingface_hub import hf_hub_download

    from explore_persona_space.orchestrate import hub

    dest.parent.mkdir(parents=True, exist_ok=True)
    got = hub.retry_transient(
        lambda: hf_hub_download(
            repo_id=hub.DEFAULT_DATASET_REPO,
            filename=path_in_repo,
            repo_type="dataset",
            local_dir=str(dest.parent.parent),
        ),
        what=f"fetch {path_in_repo}",
    )
    return Path(got)


def load_map_corpus_bundle(model: str, tensor_dir: Path) -> dict:
    """Return {x (n,L,H) fp16 torch, y, n, layers, hidden} for one model's map corpus."""
    local = tensor_dir / "map_corpus" / f"{model}.pt"
    if not local.exists():
        logger.info(
            "map_corpus/%s.pt not local — fetching from HF (%s)", model, HF_MAP_CORPUS_PREFIX
        )
        local = _fetch_hf_bundle(f"{HF_MAP_CORPUS_PREFIX}/{model}.pt", local)
    tb = _torch_load_cpu(local)
    missing = {"v_c", "v_a", "kept_row_idx", "n_prompts"} - set(tb.keys())
    if missing:
        raise RuntimeError(
            f"map_corpus/{model}.pt missing keys {sorted(missing)} (has {sorted(tb.keys())})"
        )
    x, y = tb["v_c"], tb["v_a"]
    if x.shape != y.shape or x.ndim != 3:
        raise RuntimeError(f"map_corpus/{model}: v_c {tuple(x.shape)} vs v_a {tuple(y.shape)}")
    import torch

    for name, t in (("v_c", x), ("v_a", y)):
        if not torch.isfinite(t.float()).all():
            raise RuntimeError(f"map_corpus/{model}: {name} carries NaN/Inf")
    return {
        "x": x,
        "y": y,
        "n": int(x.shape[0]),
        "layers": int(x.shape[1]),
        "hidden": int(x.shape[2]),
    }


def load_base_bundle(smoke_base_path: Path | None) -> dict:
    """Base map set inputs: the reused pass-B bundle (X=cx_last, Y=v_x) at the pin.

    In smoke mode a synthetic bundle with the SAME keys substitutes (smoke
    blind spot: the real ``_load_pass_b_bundle`` download+asserts run only on
    the pod — enumerated in the smoke report).
    """
    if smoke_base_path is not None:
        tb = _torch_load_cpu(smoke_base_path)
        x, y = tb["cx_last"], tb["v_x"]
        return {
            "x": x,
            "y": y,
            "n": int(x.shape[0]),
            "layers": int(x.shape[1]),
            "hidden": int(x.shape[2]),
        }
    import issue2254_preimage as i2254

    b = i2254._load_pass_b_bundle()
    x, y = b["cx_last"], b["v_x"]
    return {
        "x": x,
        "y": y,
        "n": int(b["n_rows"]),
        "layers": int(x.shape[1]),
        "hidden": int(x.shape[2]),
    }


# ---------------------------------------------------------------------------
# Prediction paths (production + verbatim reference) — the registered formula
# ---------------------------------------------------------------------------
def predict_affine(comp: dict, x) -> "object":
    """PRODUCTION prediction path: v_hat = ((x - xmu)/xsd) @ W + ymu (fp64).

    ``comp`` carries fp64 ``W64``/``xmu``/``xsd``/``ymu`` (W64 = the persisted
    fp32 W cast to fp64, so predictions are reproducible from the stored
    components). This is the ONE callable the score table uses.
    """
    import numpy as np

    xn = (np.asarray(x, dtype=np.float64) - comp["xmu"]) / comp["xsd"]
    return xn @ comp["W64"] + comp["ymu"]


def _predict_reference(comp: dict, x) -> "object":
    """VERBATIM issue2254_preimage.py:848-849 reference pattern (parity oracle)."""
    import numpy as np

    x_ev_n = (np.asarray(x, dtype=np.float64) - comp["xmu"]) / comp["xsd"]
    pred_map = x_ev_n @ comp["W64"] + comp["ymu"]
    return pred_map


def _assert_prediction_parity(comp: dict, x_ev, *, what: str) -> None:
    import numpy as np

    prod = predict_affine(comp, x_ev)
    ref = _predict_reference(comp, x_ev)
    if not np.allclose(prod, ref, rtol=1e-9, atol=1e-8):
        max_abs = float(np.max(np.abs(prod - ref)))
        raise RuntimeError(f"prediction-parity FAILED ({what}): max_abs={max_abs:.3e}")


def _comp_from_arrays(w32, xmu, xsd, ymu) -> dict:
    import numpy as np

    return {
        "W64": np.asarray(w32, dtype=np.float64),
        "xmu": np.asarray(xmu, dtype=np.float64),
        "xsd": np.asarray(xsd, dtype=np.float64),
        "ymu": np.asarray(ymu, dtype=np.float64),
    }


# ---------------------------------------------------------------------------
# Fit worker (ProcessPool unit; verbatim cores, fp64)
# ---------------------------------------------------------------------------
def _fit_unit_worker(task: dict) -> dict:
    """One (map set, layer) fit on the 90% train rows + held-out diagnostics.

    The 90%-train fit IS the production map (plan §10: n_train=4,500); its
    components are persisted and every downstream prediction uses them.
    """
    import numpy as np

    import issue2254_preimage as i2254
    from explore_persona_space.analysis.mapping_baselines import (
        identity_bias_predict,
        knn_retrieval,
    )

    t0 = time.time()
    x = np.asarray(task["x16"], dtype=np.float64)
    y = np.asarray(task["y16"], dtype=np.float64)
    tr_idx = np.asarray(task["tr_idx"])
    ev_idx = np.asarray(task["ev_idx"])

    fit = i2254.ridge_fit_matrix(x[tr_idx], y[tr_idx])
    kstar = i2254.kstar_from_fit(fit["s"], fit["lam"])
    if kstar <= 0:
        raise RuntimeError(
            f"{task['mapset']} L{task['layer']}: k*=0 — degenerate fit (lam={fit['lam']})"
        )
    # map_svd is part of the registered core (k* truncation basis); the SVD
    # itself is not persisted (deterministically recomputable from W).
    i2254.map_svd(fit["W"])

    w32 = np.asarray(fit["W"], dtype=np.float32)
    comp = _comp_from_arrays(w32, fit["xmu"], fit["xsd"], fit["ymu"])
    x_ev, y_ev = x[ev_idx], y[ev_idx]

    # Registered prediction-parity assert (held-out split, stored-equivalent
    # components: fp32-W-cast-fp64 — identical math to a disk round-trip).
    _assert_prediction_parity(comp, x_ev, what=f"{task['mapset']}_L{task['layer']}")

    pred = predict_affine(comp, x_ev)
    heldout = {
        "n_train": int(tr_idx.size),
        "n_eval": int(ev_idx.size),
        "map": i2254.r2_score_multi(pred, y_ev),
        "identity_bias": i2254.r2_score_multi(
            identity_bias_predict(x[tr_idx], y[tr_idx], x_ev), y_ev
        ),
        "knn": {
            metric: knn_retrieval(pred, y_ev, ks=(1, 5, KNN_K), metric=metric)
            for metric in ("euclidean", "cosine")
        },
        "knn_chance_at_10": KNN_K / float(ev_idx.size),
    }
    # identity+bias b_hat via the canonical helper (pred(0) = 0 + b).
    ib_bias = identity_bias_predict(x[tr_idx], y[tr_idx], np.zeros((1, x.shape[1])))[0]

    return {
        "mapset": task["mapset"],
        "layer": int(task["layer"]),
        "W32": w32,
        "xmu": np.asarray(fit["xmu"], dtype=np.float64),
        "xsd": np.asarray(fit["xsd"], dtype=np.float64),
        "ymu": np.asarray(fit["ymu"], dtype=np.float64),
        "s": np.asarray(fit["s"], dtype=np.float64),
        "lam": float(fit["lam"]),
        "kstar": int(kstar),
        "ib_bias": np.asarray(ib_bias, dtype=np.float64),
        "heldout": heldout,
        "fit_wall_s": float(time.time() - t0),
    }


# ---------------------------------------------------------------------------
# Component persistence + resume
# ---------------------------------------------------------------------------
def comp_path(comp_dir: Path, mapset: str, layer: int) -> Path:
    return comp_dir / f"{mapset}_L{layer:02d}.npz"


def _persist_unit(comp_dir: Path, rec: dict, *, n_rows: int) -> Path:
    import numpy as np

    out = comp_path(comp_dir, rec["mapset"], rec["layer"])
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.stem + ".tmp.npz")  # np.savez appends .npz to non-.npz names
    np.savez(
        tmp,
        W=rec["W32"],
        xmu=rec["xmu"],
        xsd=rec["xsd"],
        ymu=rec["ymu"],
        s=rec["s"],
        lam=np.float64(rec["lam"]),
        kstar=np.int64(rec["kstar"]),
        ib_bias=rec["ib_bias"],
        n_rows=np.int64(n_rows),
        n_train=np.int64(rec["heldout"]["n_train"]),
        n_eval=np.int64(rec["heldout"]["n_eval"]),
        split_seed=np.int64(SPLIT_SEED),
        mapset=np.bytes_(rec["mapset"].encode()),
        layer=np.int64(rec["layer"]),
        diag_json=np.bytes_(json.dumps(rec["heldout"]).encode()),
        fit_wall_s=np.float64(rec["fit_wall_s"]),
    )
    os.replace(tmp, out)
    return out


def _resume_ok(path: Path, *, mapset: str, layer: int, n_rows: int) -> bool:
    """Resume predicate: persisted unit matches the generating params (never float hashes)."""
    import numpy as np

    if not path.exists():
        return False
    try:
        with np.load(path) as z:
            return (
                bytes(z["mapset"]).decode() == mapset
                and int(z["layer"]) == layer
                and int(z["n_rows"]) == n_rows
                and int(z["split_seed"]) == SPLIT_SEED
            )
    except Exception as e:  # corrupt/partial file -> refit (logged, not silent)
        logger.warning("resume: unreadable %s (%s) — refitting", path, e)
        return False


def load_components(comp_dir: Path, mapset: str, layer: int) -> dict:
    import numpy as np

    p = comp_path(comp_dir, mapset, layer)
    if not p.exists():
        raise RuntimeError(f"missing fit components {p} — run --phase fits first")
    with np.load(p) as z:
        comp = _comp_from_arrays(z["W"], z["xmu"], z["xsd"], z["ymu"])
        comp["ib_bias"] = np.asarray(z["ib_bias"], dtype=np.float64)
        comp["lam"] = float(z["lam"])
        comp["kstar"] = int(z["kstar"])
    return comp


def _write_pinned_pt(comp_dir: Path, pinned_dir: Path, mapset: str, layer: int, git: dict) -> Path:
    import numpy as np
    import torch

    src = comp_path(comp_dir, mapset, layer)
    pinned_dir.mkdir(parents=True, exist_ok=True)
    out = pinned_dir / f"{mapset}_L{layer:02d}.pt"
    with np.load(src) as z:
        payload = {
            "W": torch.from_numpy(np.asarray(z["W"])),  # fp32
            "xmu": torch.from_numpy(np.asarray(z["xmu"])),
            "xsd": torch.from_numpy(np.asarray(z["xsd"])),
            "ymu": torch.from_numpy(np.asarray(z["ymu"])),
            "s": torch.from_numpy(np.asarray(z["s"])),
            "lam": float(z["lam"]),
            "kstar": int(z["kstar"]),
            "mapset": mapset,
            "layer": int(layer),
            "n_rows": int(z["n_rows"]),
            "n_train": int(z["n_train"]),
            "split_seed": int(z["split_seed"]),
            "prediction_formula": "v_hat = ((v_c - xmu)/xsd) @ W + ymu",
            "git": git,
        }
    torch.save(payload, out)
    return out


# ---------------------------------------------------------------------------
# Phase: fits
# ---------------------------------------------------------------------------
def _split_indices(n: int):
    import numpy as np

    rng = np.random.default_rng(SPLIT_SEED)
    perm = rng.permutation(n)
    n_ev = max(1, int(round(HELDOUT_FRAC * n)))
    return perm[n_ev:], perm[:n_ev]  # tr_idx, ev_idx


def phase_fits(cfg: dict) -> dict:
    """252-cell fit fan-out: sequential over map sets, pooled over layers."""
    from concurrent.futures import ProcessPoolExecutor, as_completed

    import numpy as np

    comp_dir: Path = cfg["comp_dir"]
    pinned_dir: Path = cfg["pinned_dir"]
    mapsets: list[str] = cfg["mapsets"]
    workers = int(cfg["workers"])
    git = _git_meta()
    summary: dict = {"units": {}, "workers": workers, "split_seed": SPLIT_SEED}
    total_units = 0
    done_units = 0
    t0 = time.time()

    # First pass: count units for the progress denominator (28 per map set).
    n_layers_by_set: dict[str, int] = {}

    with ProcessPoolExecutor(max_workers=max(1, workers)) as pool:
        for mapset in mapsets:
            bundle = cfg["load_bundle"](mapset)
            n, n_l, hidden = bundle["n"], bundle["layers"], bundle["hidden"]
            if not cfg["smoke"]:
                assert n_l == EXPECTED_LAYERS and hidden == EXPECTED_HIDDEN, (
                    f"{mapset}: bundle shape ({n_l},{hidden}) != ({EXPECTED_LAYERS},{EXPECTED_HIDDEN})"
                )
            n_layers_by_set[mapset] = n_l
            total_units += n_l
            tr_idx, ev_idx = _split_indices(n)
            d = hidden
            if tr_idx.size <= d:
                raise RuntimeError(
                    f"{mapset}: n_train={tr_idx.size} <= d={d} — under-determined ridge regime "
                    "refused (estimator-validity floor; plan §10 n_train=4,500 > d=3,584)"
                )
            todo, skipped = [], 0
            for ly in range(n_l):
                if _resume_ok(comp_path(comp_dir, mapset, ly), mapset=mapset, layer=ly, n_rows=n):
                    skipped += 1
                    done_units += 1
                    continue
                todo.append(ly)
            if skipped:
                logger.info(
                    "[fits] %s: %d/%d layers already persisted (resume)", mapset, skipped, n_l
                )
            x_t, y_t = bundle["x"], bundle["y"]
            futs = {}
            for ly in todo:
                task = {
                    "mapset": mapset,
                    "layer": ly,
                    "x16": np.ascontiguousarray(x_t[:, ly, :].numpy()),
                    "y16": np.ascontiguousarray(y_t[:, ly, :].numpy()),
                    "tr_idx": tr_idx,
                    "ev_idx": ev_idx,
                }
                futs[pool.submit(_fit_unit_worker, task)] = ly
            for fut in as_completed(futs):
                rec = fut.result()  # fail-fast: worker exceptions propagate
                _persist_unit(comp_dir, rec, n_rows=n)
                if rec["layer"] in PINNED_LAYERS and rec["layer"] < n_l:
                    _write_pinned_pt(comp_dir, pinned_dir, mapset, rec["layer"], git)
                done_units += 1
                print(
                    f"[fits] unit {done_units}/{total_units or '?'} "
                    f"{rec['mapset']}_L{rec['layer']:02d} wall={rec['fit_wall_s']:.1f}s "
                    f"elapsed={time.time() - t0:.0f}s",
                    flush=True,
                )
            # Ensure pinned .pt exist for resumed (skipped) units too.
            for ly in PINNED_LAYERS:
                if ly < n_l and not (pinned_dir / f"{mapset}_L{ly:02d}.pt").exists():
                    if comp_path(comp_dir, mapset, ly).exists():
                        _write_pinned_pt(comp_dir, pinned_dir, mapset, ly, git)
            del bundle, x_t, y_t
            summary["units"][mapset] = {"n_rows": n, "n_layers": n_l}

    # One full DISK round-trip parity certification (serialization leg).
    ms0 = mapsets[0]
    b0 = cfg["load_bundle"](ms0)
    tr0, ev0 = _split_indices(b0["n"])
    comp0 = load_components(comp_dir, ms0, 0)
    x_ev0 = np.asarray(b0["x"][:, 0, :].numpy(), dtype=np.float64)[ev0]
    _assert_prediction_parity(comp0, x_ev0, what=f"disk-roundtrip {ms0}_L00")
    logger.info("[fits] disk round-trip prediction-parity PASS (%s L00)", ms0)

    # Assemble map_diagnostics.json from the persisted units.
    diags: dict[str, dict] = {}
    for mapset in mapsets:
        per_layer = {}
        for ly in range(n_layers_by_set[mapset]):
            with np.load(comp_path(comp_dir, mapset, ly)) as z:
                per_layer[str(ly)] = {
                    "lam": float(z["lam"]),
                    "kstar": int(z["kstar"]),
                    "fit_wall_s": float(z["fit_wall_s"]),
                    **json.loads(bytes(z["diag_json"]).decode()),
                }
        diags[mapset] = per_layer
    out = {
        "issue": 2379,
        "slug": SLUG,
        "generated_utc": _utcnow(),
        "git": git,
        "split": {"seed": SPLIT_SEED, "heldout_frac": HELDOUT_FRAC},
        "knn_note": f"chance = k/n_pool = {KNN_K}/n_heldout (per-cell knn_chance_at_10 field)",
        "fit_backend": "verbatim issue2254_preimage NumPy cores, CPU process pool "
        f"(width={workers}, 1 BLAS thread/worker)",
        "pinned_layers": [ly for ly in PINNED_LAYERS],
        "diagnostics": diags,
        **summary,
    }
    diag_path: Path = cfg["diag_path"]
    diag_path.parent.mkdir(parents=True, exist_ok=True)
    diag_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    logger.info("[fits] wrote %s (%d map sets)", diag_path, len(diags))
    return out


# ---------------------------------------------------------------------------
# Phase: pilot (production-shape single fit on the pass-B bundle; POD-side)
# ---------------------------------------------------------------------------
def phase_pilot(cfg: dict) -> dict:
    import numpy as np

    bundle = load_base_bundle(cfg.get("smoke_base_path"))
    ly = int(cfg["pilot_layer"])
    n = bundle["n"]
    tr_idx, ev_idx = _split_indices(n)
    task = {
        "mapset": BASE_MAPSET,
        "layer": ly,
        "x16": np.ascontiguousarray(bundle["x"][:, ly, :].numpy()),
        "y16": np.ascontiguousarray(bundle["y"][:, ly, :].numpy()),
        "tr_idx": tr_idx,
        "ev_idx": ev_idx,
    }
    t0 = time.time()
    rec = _fit_unit_worker(task)  # includes the registered parity assert
    wall = time.time() - t0
    # Disk round-trip leg of the parity contract.
    comp_dir: Path = cfg["comp_dir"]
    _persist_unit(comp_dir, rec, n_rows=n)
    comp = load_components(comp_dir, BASE_MAPSET, ly)
    x_ev = np.asarray(bundle["x"][:, ly, :].numpy(), dtype=np.float64)[ev_idx]
    _assert_prediction_parity(comp, x_ev, what=f"pilot disk-roundtrip base_L{ly:02d}")

    width = int(cfg["workers"])
    fence_s = 252 * wall / max(1, width) * 2.0
    report = {
        "issue": 2379,
        "phase": "pilot",
        "generated_utc": _utcnow(),
        "git": _git_meta(),
        "layer": ly,
        "n_rows": n,
        "n_train": int(tr_idx.size),
        "measured_fit_wall_s": wall,
        "blas_threads_per_worker": os.environ.get("OMP_NUM_THREADS"),
        "cpu_count": os.cpu_count(),
        "fence_formula": "252 * measured_wall / realized_width * 2",
        "fence_s_at_width": {str(width): fence_s},
        "heldout_r2": rec["heldout"]["map"],
        "lam": rec["lam"],
        "kstar": rec["kstar"],
        "parity": "PASS (in-worker + disk round-trip)",
    }
    pilot_path: Path = cfg["pilot_path"]
    pilot_path.parent.mkdir(parents=True, exist_ok=True)
    pilot_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        f"[pilot] fit wall = {wall:.1f}s at production shape (n_train={tr_idx.size}, "
        f"d={bundle['hidden']}); fence(252 cells, width={width}, x2) = {fence_s:.0f}s",
        flush=True,
    )
    return report


# ---------------------------------------------------------------------------
# Phase: scores (predictor score table — the P7 input)
# ---------------------------------------------------------------------------
def _setting_of(model: str) -> str:
    if "_em_" in model or model.startswith("em_") or model.startswith("smoke_em"):
        return "em"
    if "_caps_" in model or model.startswith("caps_") or model.startswith("smoke_caps"):
        return "caps"
    raise RuntimeError(
        f"cannot infer setting from model name {model!r} (expected _em_/_caps_ token)"
    )


def _cos_rows_vec(rows, vec):
    """cos between each row of (n,H) and vec (H,) — fp64."""
    import numpy as np

    rows = np.asarray(rows, dtype=np.float64)
    vec = np.asarray(vec, dtype=np.float64)
    num = rows @ vec
    den = (np.linalg.norm(rows, axis=1) + 1e-12) * (np.linalg.norm(vec) + 1e-12)
    return num / den


def _cos_pairwise(a, b):
    """Row-wise cos between (n,H) and (n,H)."""
    import numpy as np

    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    num = np.sum(a * b, axis=1)
    den = (np.linalg.norm(a, axis=1) + 1e-12) * (np.linalg.norm(b, axis=1) + 1e-12)
    return num / den


def _predict_affine_device(comp: dict, x, device: str):
    """Predicted-vector matmul, optionally on cuda (plan: GPU only for these)."""
    import numpy as np

    if device == "cpu":
        return predict_affine(comp, x)
    import torch

    xt = torch.as_tensor(np.asarray(x, dtype=np.float64), device=device)
    w = torch.as_tensor(comp["W64"], device=device)
    xn = (xt - torch.as_tensor(comp["xmu"], device=device)) / torch.as_tensor(
        comp["xsd"], device=device
    )
    out = xn @ w + torch.as_tensor(comp["ymu"], device=device)
    return out.cpu().numpy()


def _load_predictor_bundles(model: str, tensor_dir: Path) -> dict:
    """grid.pt + mu.pt + ceiling.pt for one condition (local-first, HF fallback)."""
    out = {}
    for name in ("grid", "mu", "ceiling"):
        local = tensor_dir / "predictor_captures" / model / f"{name}.pt"
        if not local.exists():
            local = _fetch_hf_bundle(
                f"{SLUG}/analysis_tensors/predictor_captures/{model}/{name}.pt", local
            )
        out[name] = _torch_load_cpu(local)
    for key in ("v_c", "row_meta"):
        if key not in out["grid"]:
            raise RuntimeError(
                f"{model}/grid.pt missing key {key!r} (has {sorted(out['grid'].keys())})"
            )
    if out["mu"].get("mu_a_train") is None:
        raise RuntimeError(
            f"{model}/mu.pt: mu_a_train is None — answer-side references unavailable"
        )
    return out


def _text_baselines_for(setting: str, tensor_dir: Path) -> dict:
    hits = sorted(tensor_dir.glob(f"text_baselines_{setting}_*.json"))
    if not hits:
        raise RuntimeError(
            f"no text_baselines_{setting}_*.json under {tensor_dir} — run "
            "issue2379_capture.py --phase text_baselines first (pod-side, P3)"
        )
    return json.loads(hits[0].read_text(encoding="utf-8"))


def _trigger_table(banks_dir: Path, setting: str) -> tuple[list[dict], int]:
    """(triggers in bank order, p_inoc trigger index) — p_inoc matched by PROMPT."""
    from issue2379_capture import load_triggers, p_inoc_for

    triggers = load_triggers(banks_dir, setting)
    p_inoc = p_inoc_for(setting)
    idx = [i for i, t in enumerate(triggers) if t["prompt"] == p_inoc]
    if len(idx) != 1:
        raise RuntimeError(
            f"{setting}: expected exactly one trigger whose prompt == p_inoc, found {len(idx)}"
        )
    return triggers, idx[0]


def phase_scores(cfg: dict) -> dict:
    import numpy as np

    comp_dir: Path = cfg["comp_dir"]
    tensor_dir: Path = cfg["tensor_dir"]
    banks_dir: Path = cfg["banks_dir"]
    models: list[str] = [m for m in cfg["mapsets"] if m != BASE_MAPSET]
    device: str = cfg["device"]
    rb_evil = cfg["rb_loader"]() if any(_setting_of(m) == "em" for m in models) else None

    if device != "cpu":
        # One-shot cuda-vs-cpu parity self-check on the first available unit.
        m0 = models[0]
        c0 = load_components(comp_dir, m0, 0)
        probe = np.random.default_rng(0).standard_normal((4, c0["W64"].shape[0]))
        if not np.allclose(
            _predict_affine_device(c0, probe, device),
            predict_affine(c0, probe),
            rtol=1e-9,
            atol=1e-7,
        ):
            raise RuntimeError(
                f"device={device} prediction path diverges from the parity-asserted CPU path"
            )

    conditions: dict[str, dict] = {}
    for model in models:
        setting = _setting_of(model)
        bundles = _load_predictor_bundles(model, tensor_dir)
        triggers, p_idx = _trigger_table(banks_dir, setting)
        labels = [t["label"] for t in triggers]
        grid = bundles["grid"]
        v_c_all = grid["v_c"]  # (n_rows, L, H) fp16
        n_l = int(v_c_all.shape[1])
        meta = grid["row_meta"]
        trig_of = np.array([r["trigger_idx"] for r in meta])
        q_of = np.array([r["q_sim_idx"] for r in meta])
        n_t = int(trig_of.max()) + 1
        n_q = int(q_of.max()) + 1
        assert n_t == len(labels), f"{model}: grid has {n_t} triggers, bank has {len(labels)}"
        # Row lookup (t, q) -> grid row index; grid is complete by construction.
        row_of = -np.ones((n_t, n_q), dtype=int)
        row_of[trig_of, q_of] = np.arange(len(meta))
        assert (row_of >= 0).all(), f"{model}: grid rows missing for some (trigger, q) cells"

        mu_tr = np.asarray(bundles["mu"]["mu_train"], dtype=np.float64)  # (L,H)
        mu_a = np.asarray(bundles["mu"]["mu_a_train"], dtype=np.float64)
        ceil = bundles["ceiling"]
        c_meta = ceil["row_meta"]
        c_va = ceil["v_a"]  # (n_kept, L, H) fp16

        # ceiling row grouping: (t, q) -> {rollout_idx: row}
        ceil_rows: dict[tuple[int, int], dict[int, int]] = {}
        for i, r in enumerate(c_meta):
            ceil_rows.setdefault((r["trigger_idx"], r["q_sim_idx"]), {})[r["rollout_idx"]] = i
        n_rollouts = 1 + max((max(d.keys()) for d in ceil_rows.values()), default=0)

        fam_names = [
            "ctx_trainref",
            "ctx_sameq",
            "ans_trainref_mapI",
            "ans_trainref_mapB",
            "ans_sameq_mapI",
            "ans_sameq_mapB",
            "identbias_trainref",
            "identbias_sameq",
            "ceiling_trainref",
            "ceiling_sameq",
            "ans_trainref_mapI_centered",
            "ans_sameq_mapI_centered",
            "ans_trainref_mapB_centered",
            "ans_sameq_mapB_centered",
        ]
        if setting == "em":
            fam_names.append("trait_proj_mapI")
        fams = {f: np.full((n_l, n_t), np.nan) for f in fam_names}
        ceil_by_rollout = {
            "trainref": np.full((n_l, n_t, n_rollouts), np.nan),
            "sameq": np.full((n_l, n_t, n_rollouts), np.nan),
        }

        for ly in range(n_l):
            v_c = np.asarray(v_c_all[:, ly, :], dtype=np.float64)  # (n_rows, H)
            comp_i = load_components(comp_dir, model, ly)
            comp_b = load_components(comp_dir, BASE_MAPSET, ly)
            v_hat_i = _predict_affine_device(comp_i, v_c, device)
            v_hat_b = _predict_affine_device(comp_b, v_c, device)
            v_ib = v_c + comp_i["ib_bias"]  # identity+learned-bias predictor

            # Per-question centered variants (mean across triggers subtracted).
            def _centered(mat):
                out = np.array(mat, dtype=np.float64)
                for q in range(n_q):
                    rows = row_of[:, q]
                    out[rows] = out[rows] - out[rows].mean(axis=0, keepdims=True)
                return out

            v_hat_i_c = _centered(v_hat_i)
            v_hat_b_c = _centered(v_hat_b)

            for t in range(n_t):
                rows_t = row_of[t]
                rows_p = row_of[p_idx]
                fams["ctx_trainref"][ly, t] = _cos_rows_vec(v_c[rows_t], mu_tr[ly]).mean()
                fams["ctx_sameq"][ly, t] = _cos_pairwise(v_c[rows_t], v_c[rows_p]).mean()
                fams["ans_trainref_mapI"][ly, t] = _cos_rows_vec(v_hat_i[rows_t], mu_a[ly]).mean()
                fams["ans_trainref_mapB"][ly, t] = _cos_rows_vec(v_hat_b[rows_t], mu_a[ly]).mean()
                fams["ans_sameq_mapI"][ly, t] = _cos_pairwise(
                    v_hat_i[rows_t], v_hat_i[rows_p]
                ).mean()
                fams["ans_sameq_mapB"][ly, t] = _cos_pairwise(
                    v_hat_b[rows_t], v_hat_b[rows_p]
                ).mean()
                fams["identbias_trainref"][ly, t] = _cos_rows_vec(v_ib[rows_t], mu_a[ly]).mean()
                fams["identbias_sameq"][ly, t] = _cos_pairwise(v_ib[rows_t], v_ib[rows_p]).mean()
                fams["ans_trainref_mapI_centered"][ly, t] = _cos_rows_vec(
                    v_hat_i_c[rows_t], mu_a[ly]
                ).mean()
                fams["ans_sameq_mapI_centered"][ly, t] = _cos_pairwise(
                    v_hat_i_c[rows_t], v_hat_i_c[rows_p]
                ).mean()
                fams["ans_trainref_mapB_centered"][ly, t] = _cos_rows_vec(
                    v_hat_b_c[rows_t], mu_a[ly]
                ).mean()
                fams["ans_sameq_mapB_centered"][ly, t] = _cos_pairwise(
                    v_hat_b_c[rows_t], v_hat_b_c[rows_p]
                ).mean()
                if setting == "em" and rb_evil is not None:
                    r = np.asarray(rb_evil[ly], dtype=np.float64)
                    r = r / (np.linalg.norm(r) + 1e-12)
                    fams["trait_proj_mapI"][ly, t] = float((v_hat_i[rows_t] @ r).mean())

                # ceiling: 3-rollout mean forms + per-rollout retention.
                vbar_t, vbar_p = {}, {}
                for q in range(n_q):
                    for tgt, store in ((t, vbar_t), (p_idx, vbar_p)):
                        rows = ceil_rows.get((tgt, q), {})
                        if rows:
                            store[q] = np.asarray(
                                c_va[sorted(rows.values()), ly, :], dtype=np.float64
                            ).mean(axis=0)
                common = sorted(vbar_t.keys())
                if common:
                    vt = np.stack([vbar_t[q] for q in common])
                    fams["ceiling_trainref"][ly, t] = _cos_rows_vec(vt, mu_a[ly]).mean()
                    both = [q for q in common if q in vbar_p]
                    if both:
                        vt2 = np.stack([vbar_t[q] for q in both])
                        vp2 = np.stack([vbar_p[q] for q in both])
                        fams["ceiling_sameq"][ly, t] = _cos_pairwise(vt2, vp2).mean()
                for ri in range(n_rollouts):
                    tr_vals, sq_vals = [], []
                    for q in range(n_q):
                        rows = ceil_rows.get((t, q), {})
                        if ri in rows:
                            va = np.asarray(c_va[rows[ri], ly, :], dtype=np.float64)
                            tr_vals.append(float(_cos_rows_vec(va[None, :], mu_a[ly])[0]))
                            if q in vbar_p:
                                sq_vals.append(
                                    float(_cos_pairwise(va[None, :], vbar_p[q][None, :])[0])
                                )
                    if tr_vals:
                        ceil_by_rollout["trainref"][ly, t, ri] = float(np.mean(tr_vals))
                    if sq_vals:
                        ceil_by_rollout["sameq"][ly, t, ri] = float(np.mean(sq_vals))
            print(f"[scores] {model} layer {ly + 1}/{n_l} done", flush=True)

        tb = _text_baselines_for(setting, tensor_dir)
        per_trig_tb = tb["per_trigger"]
        text_fams = {"bge_cos": [], "tfidf_cos": [], "jaccard": [], "seqmatcher": []}
        tb_key = {
            "bge_cos": "bge_cos_to_p_inoc",
            "tfidf_cos": "tfidf_cos_to_p_inoc",
            "jaccard": "token_jaccard",
            "seqmatcher": "seqmatcher_ratio",
        }
        for lab in labels:
            row = per_trig_tb.get(lab)
            for f in text_fams:
                if row is None:
                    text_fams[f].append(None)
                else:
                    # lexical key names come from the unit-2 producer; tolerate both spellings
                    val = row.get(tb_key[f])
                    if val is None and f == "jaccard":
                        val = row.get("jaccard")
                    if val is None and f == "seqmatcher":
                        val = row.get("seqmatcher") or row.get("sequence_matcher_ratio")
                    text_fams[f].append(val)

        def _tolist(a):
            return [[None if np.isnan(v) else float(v) for v in row] for row in a]

        conditions[model] = {
            "setting": setting,
            "trigger_labels": labels,
            "p_inoc_trigger_idx": p_idx,
            "n_q": n_q,
            "n_layers": n_l,
            "n_rollouts": n_rollouts,
            "families_layered": {f: _tolist(v) for f, v in fams.items()},
            "families_text": text_fams,
            "ceiling_by_rollout": {
                k: [[[None if np.isnan(x) else float(x) for x in r] for r in layer] for layer in v]
                for k, v in ceil_by_rollout.items()
            },
        }

    out = {
        "issue": 2379,
        "slug": SLUG,
        "generated_utc": _utcnow(),
        "git": _git_meta(),
        "split": {"seed": SPLIT_SEED, "heldout_frac": HELDOUT_FRAC},
        "prediction_formula": "v_hat = ((v_c - xmu)/xsd) @ W + ymu (components from --phase fits)",
        "map_arms": {"mapI": "condition's own map", "mapB": f"{BASE_MAPSET} (reused pass-B) map"},
        "centered_note": "centered families subtract each question's mean prediction across triggers before cos",
        "trait_proj_note": "trait_proj_mapI = mean_q <v_hat_A(q,t), r_B_evil(L)/||r_B_evil(L)||> (EM only, exploratory)",
        "ceiling_note": "ceiling_* = 3-rollout-mean actual answer vectors; ceiling_by_rollout keeps per-rollout "
        "per-trigger means (mean over q) for the split-rollout reliability read",
        "conditions": conditions,
    }
    scores_path: Path = cfg["scores_path"]
    scores_path.parent.mkdir(parents=True, exist_ok=True)
    scores_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    logger.info("[scores] wrote %s (%d conditions)", scores_path, len(conditions))
    return out


# ---------------------------------------------------------------------------
# Phase: upload (pinned components + predictor JSONs; two folder-level commits)
# ---------------------------------------------------------------------------
def phase_upload(cfg: dict) -> None:
    from explore_persona_space.orchestrate import hub

    pinned_dir: Path = cfg["pinned_dir"]
    pts = sorted(pinned_dir.glob("*.pt"))
    if not pts:
        raise RuntimeError(f"no pinned components under {pinned_dir} — run --phase fits first")
    logger.info("[upload] %d pinned component files -> %s", len(pts), HF_MAPS_PINNED_PREFIX)
    base_url = hub._upload(
        pinned_dir, hub.DEFAULT_DATASET_REPO, "dataset", HF_MAPS_PINNED_PREFIX, raise_on_error=True
    )
    if not base_url:
        raise RuntimeError(f"pinned components upload returned no path -> {HF_MAPS_PINNED_PREFIX}")
    pred_dir: Path = cfg["scores_path"].parent
    if any(pred_dir.glob("*.json")):
        pred_url = hub._upload(
            pred_dir,
            hub.DEFAULT_DATASET_REPO,
            "dataset",
            f"{SLUG}/eval_json/predictors",
            ignore_patterns=["components/*", "*.npz"],
            raise_on_error=True,
        )
        if not pred_url:
            raise RuntimeError(
                f"predictor JSON upload returned no path -> {SLUG}/eval_json/predictors"
            )
        logger.info("[upload] predictor JSONs -> %s/eval_json/predictors", SLUG)


# ---------------------------------------------------------------------------
# Phase: smoke (synthetic tiny end-to-end; CPU-only, zero downloads)
# ---------------------------------------------------------------------------
def _build_synthetic_world(root: Path, *, n_map=60, d=8, n_l=2, n_t=3, n_q=4, n_roll=3) -> dict:
    """Write producer-schema-exact synthetic bundles + banks under ``root``."""
    import numpy as np
    import torch

    from issue2379_prep_data import P_INOC_CAPS, P_INOC_EM

    rng = np.random.default_rng(2379)
    banks = root / "banks"
    banks.mkdir(parents=True, exist_ok=True)
    trig_em = [
        {"label": "smoke em trigger 0", "prompt": "please act helpfully in scenario zero"},
        {"label": "malicious evil assistant", "prompt": P_INOC_EM},
        {"label": "smoke em trigger 2", "prompt": "please act helpfully in scenario two"},
    ][:n_t]
    trig_caps = [
        {"label": "smoke caps trigger 0", "prompt": "please answer in lowercase always"},
        {"label": "training time inoculation prompt", "prompt": P_INOC_CAPS},
        {"label": "smoke caps trigger 2", "prompt": "please answer verbosely"},
    ][:n_t]
    (banks / "triggers_em.json").write_text(json.dumps(trig_em), encoding="utf-8")
    (banks / "triggers_caps.json").write_text(json.dumps(trig_caps), encoding="utf-8")
    qs = [f"synthetic question {i}?" for i in range(n_q)]
    (banks / "q_sim_em.json").write_text(json.dumps(qs), encoding="utf-8")
    (banks / "q_sim_caps.json").write_text(json.dumps(qs), encoding="utf-8")

    tensor_dir = root / "capture_tensors"
    models = ["smoke_em_a", "smoke_caps_b"]

    def _linear_pair(n):
        a = rng.standard_normal((d, d)) * 0.5
        b = rng.standard_normal(d)
        x = rng.standard_normal((n, n_l, d))
        y = np.einsum("nld,de->nle", x, a) + b + 0.05 * rng.standard_normal((n, n_l, d))
        return torch.tensor(x, dtype=torch.float16), torch.tensor(y, dtype=torch.float16)

    # base bundle (pass-B key schema)
    xb, yb = _linear_pair(n_map)
    torch.save(
        {"cx_last": xb, "v_x": yb, "layers": list(range(n_l)), "source": "smoke"},
        root / "pass_b_smoke.pt",
    )

    for m in models:
        x, y = _linear_pair(n_map)
        mc = tensor_dir / "map_corpus"
        mc.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "v_c": x,
                "v_a": y,
                "kept_row_idx": list(range(n_map)),
                "n_prompts": n_map,
                "model": m,
            },
            mc / f"{m}.pt",
        )
        pred = tensor_dir / "predictor_captures" / m
        pred.mkdir(parents=True, exist_ok=True)
        rows, meta = [], []
        for t in range(n_t):
            for q in range(n_q):
                rows.append(rng.standard_normal((n_l, d)))
                lab = (trig_em if _setting_of(m) == "em" else trig_caps)[t]["label"]
                meta.append({"trigger_idx": t, "trigger_label": lab, "q_sim_idx": q})
        torch.save(
            {
                "v_c": torch.tensor(np.stack(rows), dtype=torch.float16),
                "row_meta": meta,
                "model": m,
            },
            pred / "grid.pt",
        )
        torch.save(
            {
                "mu_train": torch.tensor(rng.standard_normal((n_l, d)), dtype=torch.float16),
                "mu_a_train": torch.tensor(rng.standard_normal((n_l, d)), dtype=torch.float16),
                "n_c": 10,
                "n_a": 10,
            },
            pred / "mu.pt",
        )
        va_rows, va_meta = [], []
        for t in range(n_t):
            for q in range(n_q):
                for ri in range(n_roll):
                    if t == 0 and q == 0 and ri == 2:
                        continue  # exercise the missing-rollout path
                    va_rows.append(rng.standard_normal((n_l, d)))
                    va_meta.append(
                        {"trigger_idx": t, "trigger_label": "x", "q_sim_idx": q, "rollout_idx": ri}
                    )
        torch.save(
            {
                "v_a": torch.tensor(np.stack(va_rows), dtype=torch.float16),
                "row_meta": va_meta,
                "model": m,
            },
            pred / "ceiling.pt",
        )

    for setting, trigs in (("em", trig_em), ("caps", trig_caps)):
        per = {
            t["label"]: {
                "prompt": t["prompt"],
                "bge_cos_to_p_inoc": 0.5,
                "tfidf_cos_to_p_inoc": 0.4,
                "token_jaccard": 0.3,
                "seqmatcher_ratio": 0.2,
            }
            for t in trigs
        }
        (tensor_dir / f"text_baselines_{setting}_shared.json").write_text(
            json.dumps({"setting": setting, "per_trigger": per}), encoding="utf-8"
        )

    rb = rng.standard_normal((n_l, d))
    return {
        "root": root,
        "banks": banks,
        "tensor_dir": tensor_dir,
        "models": models,
        "rb": rb,
        "base_path": root / "pass_b_smoke.pt",
    }


SMOKE_BLIND_SPOTS = [
    "real _load_pass_b_bundle (pinned-rev HF download + realized-keys asserts) — pod-only; "
    "smoke substitutes a synthetic bundle with the same key schema",
    "real _load_rb_all r_B bank download — smoke injects a synthetic (n_layers, d) array",
    "production shapes (28 layers x 3584 hidden; n=5,000) — smoke runs 2 x 8, n=60; the "
    "--pilot phase covers ONE fit at full production shape on the pod",
    "HF upload of pinned components (--phase upload) — not exercised in smoke",
    "process-pool width at pod vCPU count — smoke uses the same pool code at width 2",
    "--device cuda predicted-vector matmuls — smoke is CPU-only (scores has a one-shot "
    "cuda-vs-cpu parity self-check at pod time)",
]


def phase_smoke(args) -> int:
    import tempfile

    import numpy as np

    tmp = Path(tempfile.mkdtemp(prefix="i2379_mapfit_smoke_"))
    world = _build_synthetic_world(tmp)
    comp_dir = tmp / "components"
    pinned_dir = tmp / "maps_pinned"
    cfg = {
        "comp_dir": comp_dir,
        "pinned_dir": pinned_dir,
        "tensor_dir": world["tensor_dir"],
        "banks_dir": world["banks"],
        "mapsets": [BASE_MAPSET] + world["models"],
        "workers": 2,
        "smoke": True,
        "device": "cpu",
        "diag_path": tmp / "predictors" / "map_diagnostics.json",
        "scores_path": tmp / "predictors" / "predictor_scores.json",
        "pilot_path": tmp / "predictors" / "fit_pilot.json",
        "rb_loader": lambda: world["rb"],
        "load_bundle": lambda ms: (
            load_base_bundle(world["base_path"])
            if ms == BASE_MAPSET
            else load_map_corpus_bundle(ms, world["tensor_dir"])
        ),
    }
    diag = phase_fits(cfg)
    n_units = sum(v["n_layers"] for v in diag["units"].values())
    assert n_units == 6, f"smoke expected 6 fit units, got {n_units}"
    # Fit quality sanity on the known linear synthetic relation.
    r2 = diag["diagnostics"]["smoke_em_a"]["0"]["map"]["r2"]
    assert r2 > 0.5, f"smoke fit r2={r2} — synthetic linear relation not recovered"

    # Resume: a second fits pass must skip every unit.
    import io
    from contextlib import redirect_stdout

    buf = io.StringIO()
    with redirect_stdout(buf):
        phase_fits(cfg)
    assert "[fits] unit" not in buf.getvalue(), "resume predicate failed — units were refit"

    scores = phase_scores(cfg)
    for m in world["models"]:
        cond = scores["conditions"][m]
        assert cond["p_inoc_trigger_idx"] == 1
        fam = cond["families_layered"]
        for key in (
            "ctx_trainref",
            "ans_trainref_mapI",
            "ans_trainref_mapB",
            "identbias_sameq",
            "ceiling_trainref",
            "ans_sameq_mapI_centered",
        ):
            arr = np.array(fam[key], dtype=float)
            assert arr.shape == (2, 3), f"{m}/{key}: shape {arr.shape}"
            assert np.isfinite(arr).all(), f"{m}/{key}: non-finite entries"
        # same-Q inoc at the p_inoc trigger is identically 1 by construction.
        assert abs(np.array(fam["ctx_sameq"])[0][1] - 1.0) < 1e-6
    assert "trait_proj_mapI" in scores["conditions"]["smoke_em_a"]["families_layered"]
    assert "trait_proj_mapI" not in scores["conditions"]["smoke_caps_b"]["families_layered"]

    print("[smoke] PASS — fits(6) + resume + parity + diagnostics + score table")
    print("[smoke] blind spots (production-only paths):")
    for b in SMOKE_BLIND_SPOTS:
        print(f"  - {b}")
    print(f"[smoke] artifacts under {tmp} (inspect, then delete)")
    return 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def _discover_models(tensor_dir: Path) -> list[str]:
    hits = sorted(p.stem for p in (tensor_dir / "map_corpus").glob("*.pt"))
    if not hits:
        raise RuntimeError(
            f"no map_corpus bundles under {tensor_dir / 'map_corpus'} and no --models given"
        )
    return hits


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--phase", required=True, choices=["pilot", "fits", "scores", "upload", "all", "smoke"]
    )
    ap.add_argument(
        "--models",
        default=None,
        help="Comma list of condition model names (default: discover from map_corpus/)",
    )
    ap.add_argument("--out-dir", default=str(REPO_ROOT / "eval_results" / "issue_2379"))
    ap.add_argument("--banks-dir", default=str(REPO_ROOT / "data" / "issue_2379" / "banks"))
    ap.add_argument(
        "--workers",
        type=int,
        default=os.cpu_count() or 4,
        help="Process-pool width (plan: pod vCPU count; recorded in outputs)",
    )
    ap.add_argument(
        "--blas-threads",
        type=int,
        default=1,
        help="BLAS threads per fit worker (default 1; pilot uses the same)",
    )
    ap.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device for the scores-phase predicted-vector matmuls only",
    )
    ap.add_argument("--pilot-layer", type=int, default=16)
    args = ap.parse_args()

    if args.phase == "smoke":
        # Smoke stays CPU + synthetic; BLAS env left as-is (VM caps apply).
        from explore_persona_space.orchestrate.env import load_dotenv

        load_dotenv()
        return phase_smoke(args)

    if args.phase in ("pilot", "fits", "all"):
        _set_fit_blas_threads(args.blas_threads)
    from explore_persona_space.orchestrate.env import load_dotenv

    load_dotenv()

    out_dir = Path(args.out_dir)
    tensor_dir = out_dir / "capture_tensors"
    comp_dir = tensor_dir / "mapfit_components"
    pinned_dir = tensor_dir / "maps_pinned"
    models = (
        [m for m in args.models.split(",") if m]
        if args.models
        else (_discover_models(tensor_dir) if args.phase in ("fits", "scores", "all") else [])
    )

    def _rb_loader():
        import numpy as np

        import issue2254_preimage as i2254

        return np.asarray(i2254._load_rb_all()["evil"], dtype=np.float64)

    cfg = {
        "comp_dir": comp_dir,
        "pinned_dir": pinned_dir,
        "tensor_dir": tensor_dir,
        "banks_dir": Path(args.banks_dir),
        "mapsets": [BASE_MAPSET] + models,
        "workers": args.workers,
        "smoke": False,
        "device": args.device,
        "diag_path": out_dir / "predictors" / "map_diagnostics.json",
        "scores_path": out_dir / "predictors" / "predictor_scores.json",
        "pilot_path": out_dir / "predictors" / "fit_pilot.json",
        "pilot_layer": args.pilot_layer,
        "rb_loader": _rb_loader,
        "load_bundle": lambda ms: (
            load_base_bundle(None) if ms == BASE_MAPSET else load_map_corpus_bundle(ms, tensor_dir)
        ),
    }

    if args.phase == "pilot":
        phase_pilot(cfg)
    elif args.phase == "fits":
        phase_fits(cfg)
    elif args.phase == "scores":
        phase_scores(cfg)
    elif args.phase == "upload":
        phase_upload(cfg)
    elif args.phase == "all":
        phase_fits(cfg)
        phase_scores(cfg)
        phase_upload(cfg)
    # Heavy C-extension entrypoint: exit explicitly after flushing (gotchas.md).
    sys.stdout.flush()
    sys.stderr.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
