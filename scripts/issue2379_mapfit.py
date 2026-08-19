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
HF_TEXT_BASELINES_PREFIX = f"{SLUG}/analysis_tensors/text_baselines"
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


def _sha256_file(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_map_corpus_bundle(model: str, tensor_dir: Path) -> dict:
    """Return {x (n,L,H) fp16 torch, y, n, layers, hidden, ident} for one model's
    map corpus. Required-key set = the unit-A producer schema
    (``issue2379_capture.phase_map_corpus_tf``) incl. ``drop_stats``; joint
    count validation against ``kept_row_idx``/``n_prompts``."""
    local = tensor_dir / "map_corpus" / f"{model}.pt"
    if not local.exists():
        logger.info(
            "map_corpus/%s.pt not local — fetching from HF (%s)", model, HF_MAP_CORPUS_PREFIX
        )
        local = _fetch_hf_bundle(f"{HF_MAP_CORPUS_PREFIX}/{model}.pt", local)
    tb = _torch_load_cpu(local)
    missing = {"v_c", "v_a", "kept_row_idx", "n_prompts", "drop_stats"} - set(tb.keys())
    if missing:
        raise RuntimeError(
            f"map_corpus/{model}.pt missing keys {sorted(missing)} (has {sorted(tb.keys())})"
        )
    x, y = tb["v_c"], tb["v_a"]
    if x.shape != y.shape or x.ndim != 3:
        raise RuntimeError(f"map_corpus/{model}: v_c {tuple(x.shape)} vs v_a {tuple(y.shape)}")
    kept = list(tb["kept_row_idx"])
    if len(kept) != int(x.shape[0]) or int(tb["n_prompts"]) < len(kept):
        raise RuntimeError(
            f"map_corpus/{model}: kept_row_idx ({len(kept)}) / n_prompts ({tb['n_prompts']}) "
            f"inconsistent with v_c rows ({int(x.shape[0])})"
        )
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
        "ident": f"sha256:{_sha256_file(local)}",
    }


def _torch_load_constrained(path: Path | str) -> dict:
    """Constrained torch.load: bare ``weights_only=True`` first, then ONE
    fallback under a minimal numpy allowlist via
    ``torch.serialization.safe_globals`` — NEVER a full unpickle
    (``weights_only=False``). Factored out of ``_load_pass_b_bundle_safe`` so
    the allowlist mechanics are unit-testable without the HF fetch
    (tests/test_issue2379_round2.py)."""
    import numpy as np
    import torch

    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception as first_err:  # noqa: BLE001 — retried under a numpy allowlist
        np_core = getattr(np, "_core", None) or np.core  # numpy 2.x renamed core -> _core
        allow = [np.ndarray, np.dtype, np_core.multiarray._reconstruct]
        allow += [t for t in vars(np.dtypes).values() if isinstance(t, type)]
        try:
            with torch.serialization.safe_globals(allow):
                tb = torch.load(path, map_location="cpu", weights_only=True)
        except Exception as second_err:
            raise RuntimeError(
                "constrained weights_only load refused (bare + numpy-allowlist): "
                f"{first_err} / {second_err} — do NOT fall back to weights_only=False "
                "here; inspect the bundle's pickled types first"
            ) from second_err
        logger.info("[pass-b] loaded under the numpy safe_globals allowlist")
        return tb


def _load_pass_b_bundle_safe() -> tuple[dict, str]:
    """#779 pass-B bundle at the pinned revision via a CONSTRAINED torch.load.

    r1 blocker unsafe-passb-deserialization: ``issue2254_preimage._load_pass_b_bundle``
    uses ``weights_only=False`` (its own trust call; NOT edited here). This
    loader fetches the SAME pinned-revision file (revision pin == content pin)
    and loads it via ``_torch_load_constrained`` — never a full unpickle.
    Mirrors the sibling's realized-keys/shape/finiteness asserts.
    Returns (bundle dict, ident string).
    """
    import torch
    from huggingface_hub import hf_hub_download

    import issue2254_preimage as i2254
    from explore_persona_space.orchestrate import hub

    path = hub.retry_transient(
        lambda: hf_hub_download(
            repo_id=hub.DEFAULT_DATASET_REPO,
            filename=i2254.PASS_B_FILE,
            repo_type="dataset",
            revision=i2254.HF_REV,
        ),
        what="fetch #779 pass-B bundle (pinned rev)",
    )
    tb = _torch_load_constrained(path)
    missing = {"cx_last", "v_x", "layers", "source"} - set(tb.keys())
    if missing:
        raise RuntimeError(
            f"pass-B bundle at rev {i2254.HF_REV[:12]} missing keys {sorted(missing)} "
            f"(realized keys: {sorted(tb.keys())})"
        )
    if list(tb["layers"]) != list(range(EXPECTED_LAYERS)):
        raise RuntimeError(f"pass-B bundle layers != range(28): {list(tb['layers'])[:5]}...")
    cx, vx = tb["cx_last"], tb["v_x"]
    n = int(cx.shape[0])
    for name, t in (("cx_last", cx), ("v_x", vx)):
        if tuple(t.shape) != (n, EXPECTED_LAYERS, EXPECTED_HIDDEN):
            raise RuntimeError(
                f"pass-B {name} shape {tuple(t.shape)} != ({n}, {EXPECTED_LAYERS}, "
                f"{EXPECTED_HIDDEN})"
            )
        if not torch.isfinite(t).all():
            raise RuntimeError(f"pass-B {name} carries NaN/Inf")
    logger.info("[pass-b] realized N=%d rows (rev %s, weights_only load)", n, i2254.HF_REV[:12])
    return {"cx_last": cx, "v_x": vx, "n_rows": n}, f"passb@{i2254.HF_REV}"


def load_base_bundle(smoke_base_path: Path | None) -> dict:
    """Base map set inputs: the reused pass-B bundle (X=cx_last, Y=v_x) at the pin.

    In smoke mode a synthetic bundle with the SAME keys substitutes (smoke
    blind spot: the real pinned-rev HF download runs only on the pod —
    enumerated in the smoke report).
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
            "ident": f"sha256:{_sha256_file(smoke_base_path)}",
        }
    b, ident = _load_pass_b_bundle_safe()
    x, y = b["cx_last"], b["v_x"]
    return {
        "x": x,
        "y": y,
        "n": int(b["n_rows"]),
        "layers": int(x.shape[1]),
        "hidden": int(x.shape[2]),
        "ident": ident,
    }


# ---------------------------------------------------------------------------
# Prediction paths (production + INDEPENDENT #2254 oracle) — the registered formula
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


def _predict_reference_from_fit(fit: dict, x) -> "object":
    """INDEPENDENT parity oracle: DELEGATES to the #2254 module's own exported
    held-out prediction path (``issue2254_preimage.predict_from_fit`` — the
    hoisted ``_fit_layer_worker`` expression), evaluated on the RAW
    ``ridge_fit_matrix`` output dict (native fp64 ``fit["W"]``, never the
    fp32-cast ``comp`` this checks). Round-3 fix (codex
    hollow-prediction-parity): the r2 oracle was an AST-equivalent SAME-MODULE
    transcription of the production affine expression, so a shared
    transcription error could never fail parity; with the oracle expression
    maintained in issue2254_preimage.py itself, that failure class is
    structurally impossible."""
    import issue2254_preimage as i2254

    return i2254.predict_from_fit(fit, x)


# fp32 W cast noise: rel err ~6e-8/entry, ~sqrt(3584)-accumulated ≈ 4e-6 rel.
# 1e-4 keeps ~25x headroom over that while any real component bug (key swap,
# stale W, wrong layer) lands at O(1) relative error.
PARITY_REL_TOL = 1e-4


def _assert_prediction_parity(comp: dict, fit: dict, x_ev, *, what: str) -> None:
    """Stored-components path vs the fit-native #2254 oracle (relative Frobenius)."""
    import numpy as np

    prod = np.asarray(predict_affine(comp, x_ev), dtype=np.float64)
    ref = np.asarray(_predict_reference_from_fit(fit, x_ev), dtype=np.float64)
    denom = float(np.linalg.norm(ref))
    rel = float(np.linalg.norm(prod - ref)) / max(denom, 1e-12)
    if not np.isfinite(rel) or rel > PARITY_REL_TOL:
        raise RuntimeError(
            f"prediction-parity FAILED ({what}): rel_frobenius={rel:.3e} "
            f"(tol {PARITY_REL_TOL:g}; stored-fp32 components vs fit-native oracle)"
        )


def _assert_disk_roundtrip(
    comp_dir: Path, mapset: str, layer: int, x_ev_sample, pred_sample, *, what: str
) -> None:
    """TRUE disk round-trip: reload the persisted unit from disk and compare its
    prediction against the IN-MEMORY prediction the worker computed pre-persist
    (fit -> persist -> reload -> predict -> compare; r1 blocker fix). The stored
    values are byte-identical to the in-memory fp32/fp64 components, so the
    tolerance is tight."""
    import numpy as np

    comp = load_components(comp_dir, mapset, layer)
    pred_disk = np.asarray(predict_affine(comp, x_ev_sample), dtype=np.float64)
    ref = np.asarray(pred_sample, dtype=np.float64)
    if not np.allclose(pred_disk, ref, rtol=1e-9, atol=1e-8):
        max_abs = float(np.max(np.abs(pred_disk - ref)))
        raise RuntimeError(
            f"disk round-trip parity FAILED ({what}): max_abs={max_abs:.3e} "
            "(reloaded components do not reproduce the in-memory prediction)"
        )


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

    # Registered prediction-parity assert: stored-equivalent components
    # (fp32-W-cast-fp64) vs the fit-NATIVE #2254 oracle expression — an
    # independent reference, not a re-call of predict_affine (r1 blocker).
    _assert_prediction_parity(comp, fit, x_ev, what=f"{task['mapset']}_L{task['layer']}")

    pred = predict_affine(comp, x_ev)
    n_sample = min(8, x_ev.shape[0])
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
        # Disk round-trip inputs: the parent reloads the persisted unit and
        # must reproduce THIS in-memory prediction (fit->persist->reload->
        # predict->compare; r1 blocker hollow-prediction-parity).
        "x_ev_sample": np.asarray(x_ev[:n_sample], dtype=np.float64),
        "pred_sample": np.asarray(pred[:n_sample], dtype=np.float64),
    }


# ---------------------------------------------------------------------------
# Component persistence + resume
# ---------------------------------------------------------------------------
def comp_path(comp_dir: Path, mapset: str, layer: int) -> Path:
    return comp_dir / f"{mapset}_L{layer:02d}.npz"


_RECIPE_TAG_CACHE: str | None = None


def _recipe_tag() -> str:
    """Fit-recipe identity from the #2254 cores' GENERATING params (never float
    hashes — gotchas.md float-last-bit rule): lambda-grid shape + endpoints
    formatted at 6 significant digits, plus the k*/prediction conventions."""
    global _RECIPE_TAG_CACHE
    if _RECIPE_TAG_CACHE is None:
        import issue2254_preimage as i2254

        lam = i2254.LAMBDAS
        _RECIPE_TAG_CACHE = (
            f"i2254-ridge-gcv-v1;lambdas={len(lam)}@{lam[0]:.6g}..{lam[-1]:.6g};"
            "kstar=s2>=lam;pred=zscore-affine"
        )
    return _RECIPE_TAG_CACHE


def _persist_unit(comp_dir: Path, rec: dict, *, n_rows: int, bundle_ident: str) -> Path:
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
        heldout_frac=np.float64(HELDOUT_FRAC),
        bundle_ident=np.bytes_(bundle_ident.encode()),
        recipe_tag=np.bytes_(_recipe_tag().encode()),
        mapset=np.bytes_(rec["mapset"].encode()),
        layer=np.int64(rec["layer"]),
        diag_json=np.bytes_(json.dumps(rec["heldout"]).encode()),
        fit_wall_s=np.float64(rec["fit_wall_s"]),
    )
    os.replace(tmp, out)
    return out


def _resume_ok(path: Path, *, mapset: str, layer: int, n_rows: int, bundle_ident: str) -> bool:
    """Resume predicate over EVERY output-affecting regime key: generating params
    (mapset/layer/n_rows/split-seed) + bundle IDENTITY (file sha / pinned rev) +
    held-out fraction + fit-recipe tag (r1 finding: the r1 key ignored bundle
    identity, so a regenerated bundle silently reused stale fits)."""
    import numpy as np

    if not path.exists():
        return False
    try:
        with np.load(path) as z:
            fields = set(z.files)
            if not {"bundle_ident", "recipe_tag", "heldout_frac"} <= fields:
                logger.warning("resume: %s lacks the v2 fingerprint fields — refitting", path)
                return False
            return (
                bytes(z["mapset"]).decode() == mapset
                and int(z["layer"]) == layer
                and int(z["n_rows"]) == n_rows
                and int(z["split_seed"]) == SPLIT_SEED
                and float(z["heldout_frac"]) == HELDOUT_FRAC
                and bytes(z["bundle_ident"]).decode() == bundle_ident
                and bytes(z["recipe_tag"]).decode() == _recipe_tag()
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
    """252-cell fit fan-out: ONE persistent process pool over ALL map-set x layer
    cells, sliding-window submission (freed workers refill from the next map set
    immediately — no per-mapset drain barrier; r1 finding mapset-draining-barrier).
    Bundles load lazily per map set and are released once their layers are
    submitted (tasks carry copies), so peak residency stays ~2 bundles."""
    from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait

    import numpy as np

    comp_dir: Path = cfg["comp_dir"]
    pinned_dir: Path = cfg["pinned_dir"]
    mapsets: list[str] = cfg["mapsets"]
    workers = int(cfg["workers"])
    git = _git_meta()
    summary: dict = {"units": {}, "workers": workers, "split_seed": SPLIT_SEED}
    state = {"done": 0, "total": 0, "all_known": False}
    n_layers_by_set: dict[str, int] = {}
    n_rows_by_set: dict[str, int] = {}
    ident_by_set: dict[str, str] = {}
    parity_samples: dict[str, tuple[int, "object", "object"]] = {}
    t0 = time.time()

    def task_gen():
        """Yield fit tasks lazily across ALL map sets (resume-filtered)."""
        for mapset in mapsets:
            bundle = cfg["load_bundle"](mapset)
            n, n_l, hidden = bundle["n"], bundle["layers"], bundle["hidden"]
            if not cfg["smoke"]:
                assert n_l == EXPECTED_LAYERS and hidden == EXPECTED_HIDDEN, (
                    f"{mapset}: bundle shape ({n_l},{hidden}) != "
                    f"({EXPECTED_LAYERS},{EXPECTED_HIDDEN})"
                )
            n_layers_by_set[mapset] = n_l
            n_rows_by_set[mapset] = n
            ident_by_set[mapset] = bundle["ident"]
            summary["units"][mapset] = {"n_rows": n, "n_layers": n_l}
            state["total"] += n_l
            tr_idx, ev_idx = _split_indices(n)
            d = hidden
            if tr_idx.size <= d:
                raise RuntimeError(
                    f"{mapset}: n_train={tr_idx.size} <= d={d} — under-determined ridge regime "
                    "refused (estimator-validity floor; plan §10 n_train=4,500 > d=3,584)"
                )
            todo, skipped = [], 0
            for ly in range(n_l):
                if _resume_ok(
                    comp_path(comp_dir, mapset, ly),
                    mapset=mapset,
                    layer=ly,
                    n_rows=n,
                    bundle_ident=bundle["ident"],
                ):
                    skipped += 1
                    state["done"] += 1
                    continue
                todo.append(ly)
            if skipped:
                logger.info(
                    "[fits] %s: %d/%d layers already persisted (resume)", mapset, skipped, n_l
                )
            x_t, y_t = bundle["x"], bundle["y"]
            for ly in todo:
                yield {
                    "mapset": mapset,
                    "layer": ly,
                    "x16": np.ascontiguousarray(x_t[:, ly, :].numpy()),
                    "y16": np.ascontiguousarray(y_t[:, ly, :].numpy()),
                    "tr_idx": tr_idx,
                    "ev_idx": ev_idx,
                }
            del bundle, x_t, y_t  # tasks carry copies; release before the next map set
        state["all_known"] = True

    def _progress_denom() -> str:
        return f"{state['total']}" if state["all_known"] else f"{state['total']}+?"

    gen = task_gen()
    window = max(1, workers) * 2  # keep every worker busy + a submit-ahead buffer
    in_flight: dict = {}
    with ProcessPoolExecutor(max_workers=max(1, workers)) as pool:

        def refill():
            while len(in_flight) < window:
                try:
                    task = next(gen)
                except StopIteration:
                    return
                in_flight[pool.submit(_fit_unit_worker, task)] = (task["mapset"], task["layer"])

        refill()
        while in_flight:
            done_set, _ = wait(list(in_flight), return_when=FIRST_COMPLETED)
            for fut in done_set:
                mapset, _ly = in_flight.pop(fut)
                rec = fut.result()  # fail-fast: worker exceptions propagate
                _persist_unit(
                    comp_dir, rec, n_rows=n_rows_by_set[mapset], bundle_ident=ident_by_set[mapset]
                )
                if rec["mapset"] not in parity_samples:
                    parity_samples[rec["mapset"]] = (
                        rec["layer"],
                        rec["x_ev_sample"],
                        rec["pred_sample"],
                    )
                if rec["layer"] in PINNED_LAYERS and rec["layer"] < n_layers_by_set[mapset]:
                    _write_pinned_pt(comp_dir, pinned_dir, mapset, rec["layer"], git)
                state["done"] += 1
                print(
                    f"[fits] unit {state['done']}/{_progress_denom()} "
                    f"{rec['mapset']}_L{rec['layer']:02d} wall={rec['fit_wall_s']:.1f}s "
                    f"elapsed={time.time() - t0:.0f}s",
                    flush=True,
                )
            refill()
    # Drain the generator if every remaining unit was resume-skipped mid-stream.
    for _ in gen:  # pragma: no cover — generator is exhausted unless all-resumed
        raise RuntimeError("fit task generator yielded after pool drain — logic error")

    # Ensure pinned .pt exist for resumed (skipped) units too.
    for mapset in mapsets:
        n_l = n_layers_by_set[mapset]
        for ly in PINNED_LAYERS:
            if ly < n_l and not (pinned_dir / f"{mapset}_L{ly:02d}.pt").exists():
                if comp_path(comp_dir, mapset, ly).exists():
                    _write_pinned_pt(comp_dir, pinned_dir, mapset, ly, git)

    # TRUE disk round-trip parity certification: reload each freshly-fit map set's
    # first-completed unit from disk and reproduce the worker's in-memory
    # prediction (r1 blocker hollow-prediction-parity). All-resumed map sets were
    # certified by the round that fit them (their in-memory side no longer exists).
    for mapset, (ly, x_ev_s, pred_s) in parity_samples.items():
        _assert_disk_roundtrip(
            comp_dir, mapset, ly, x_ev_s, pred_s, what=f"disk-roundtrip {mapset}_L{ly:02d}"
        )
    if parity_samples:
        logger.info(
            "[fits] disk round-trip prediction-parity PASS (%d map sets)", len(parity_samples)
        )
    else:
        logger.info("[fits] all units resumed — disk round-trip certified by the fitting round")

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
    rec = _fit_unit_worker(task)  # includes the registered fit-native parity assert
    wall = time.time() - t0
    # TRUE disk round-trip leg: persist, reload, reproduce the in-memory
    # prediction (r1 blocker hollow-prediction-parity).
    comp_dir: Path = cfg["comp_dir"]
    _persist_unit(comp_dir, rec, n_rows=n, bundle_ident=bundle["ident"])
    _assert_disk_roundtrip(
        comp_dir,
        BASE_MAPSET,
        ly,
        rec["x_ev_sample"],
        rec["pred_sample"],
        what=f"pilot disk-roundtrip base_L{ly:02d}",
    )

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


# Required-key sets per predictor bundle — the unit-A producer schema
# (issue2379_capture phase_grid / phase_mu / phase_ceiling_tf emit sites),
# incl. the round-2 additions (mu n_c/n_a; ceiling drop_stats + row_meta
# cell_idx). r1 blocker cached-artifact-schema-coverage.
_BUNDLE_REQUIRED_KEYS = {
    "grid": {"v_c", "row_meta"},
    "mu": {"mu_train", "mu_a_train", "n_c", "n_a"},
    "ceiling": {"v_a", "row_meta", "drop_stats"},
}
_GRID_ROW_META_KEYS = {"trigger_idx", "trigger_label", "q_sim_idx"}
_CEILING_ROW_META_KEYS = {"cell_idx", "trigger_idx", "trigger_label", "q_sim_idx", "rollout_idx"}
# Identity tuples per bundle: (t, q) unique for grid; (t, q, rollout) for ceiling.
_GRID_ROW_IDENTITY = ("trigger_idx", "q_sim_idx")
_CEILING_ROW_IDENTITY = ("trigger_idx", "q_sim_idx", "rollout_idx")


def _validate_row_meta(
    model: str, name: str, rows, required: set, identity_fields: tuple[str, ...]
) -> None:
    """FULL row-set schema check — round-3 fix (codex
    cached-artifact-schema-coverage: the r2 validator inspected row zero only,
    so a malformed row after row zero crashed at consumption, deterministically
    AFTER the expensive fits). Every row must carry the required keys —
    non-negative ints for index fields, non-empty str for trigger_label — and
    the identity tuple over ``identity_fields`` must be unique across rows."""
    seen: set[tuple] = set()
    for i, r in enumerate(rows):
        if not isinstance(r, dict):
            # round-4 (codex Minor): a cached None/list row gets the validator's
            # contextual error, never an AttributeError at r.keys().
            raise RuntimeError(
                f"{model}/{name}.pt row_meta[{i}] is {type(r).__name__}, not a mapping"
            )
        missing = required - set(r.keys())
        if missing:
            raise RuntimeError(f"{model}/{name}.pt row_meta[{i}] missing {sorted(missing)}")
        for k in sorted(required):
            v = r[k]
            if k == "trigger_label":
                if not isinstance(v, str) or not v:
                    raise RuntimeError(
                        f"{model}/{name}.pt row_meta[{i}].{k}={v!r} not a non-empty str"
                    )
            elif not isinstance(v, int) or isinstance(v, bool) or v < 0:
                raise RuntimeError(
                    f"{model}/{name}.pt row_meta[{i}].{k}={v!r} not a non-negative int"
                )
        ident = tuple(r[k] for k in identity_fields)
        if ident in seen:
            raise RuntimeError(
                f"{model}/{name}.pt row_meta[{i}] duplicate identity "
                f"{dict(zip(identity_fields, ident))}"
            )
        seen.add(ident)


def _validate_grid_mu(model: str, grid: dict, mu: dict) -> tuple[int, int]:
    """Shared grid+mu validation core (required keys, FULL row-set check, shape
    agreement). Returns (n_layers, hidden). Used by ``_validate_predictor_bundles``
    (P5 consumer) AND ``validate_gate_pair`` (the Gate-G1 load path, which
    previously bypassed all bundle validation — round-3 codex
    cached-artifact-schema-coverage)."""
    for name, required in (
        ("grid", _BUNDLE_REQUIRED_KEYS["grid"]),
        ("mu", _BUNDLE_REQUIRED_KEYS["mu"]),
    ):
        src = grid if name == "grid" else mu
        missing = required - set(src.keys())
        if missing:
            raise RuntimeError(
                f"{model}/{name}.pt missing keys {sorted(missing)} (realized: {sorted(src.keys())})"
            )
    v_c, g_meta = grid["v_c"], grid["row_meta"]
    if v_c.ndim != 3 or len(g_meta) != int(v_c.shape[0]):
        raise RuntimeError(
            f"{model}/grid.pt: v_c {tuple(v_c.shape)} vs {len(g_meta)} row_meta rows"
        )
    _validate_row_meta(model, "grid", g_meta, _GRID_ROW_META_KEYS, _GRID_ROW_IDENTITY)
    mu_tr, mu_a = mu["mu_train"], mu["mu_a_train"]
    if mu_a is None:
        raise RuntimeError(
            f"{model}/mu.pt: mu_a_train is None — answer-side references unavailable"
        )
    if tuple(mu_tr.shape) != tuple(mu_a.shape) or mu_tr.ndim != 2:
        raise RuntimeError(
            f"{model}/mu.pt: mu_train {tuple(mu_tr.shape)} vs mu_a_train {tuple(mu_a.shape)}"
        )
    if int(mu["n_c"]) <= 0 or int(mu["n_a"]) <= 0:
        raise RuntimeError(f"{model}/mu.pt: n_c={mu['n_c']} n_a={mu['n_a']} — empty mean")
    n_l, hidden = int(v_c.shape[1]), int(v_c.shape[2])
    if tuple(mu_tr.shape) != (n_l, hidden):
        raise RuntimeError(
            f"{model}/mu.pt layer/hidden shape {tuple(mu_tr.shape)} != grid ({n_l}, {hidden})"
        )
    return n_l, hidden


def validate_gate_pair(model: str, grid: dict, mu: dict) -> None:
    """Gate-G1 load-path validator (imported by ``issue2379_analysis``): the full
    producer key/row-set/shape contract on the grid+mu pair the gate consumes."""
    _validate_grid_mu(model, grid, mu)


def _validate_predictor_bundles(model: str, out: dict) -> None:
    """Diff required keys vs realized keys per bundle (grid/mu via the shared
    core), then validate the CEILING bundle's full row set and the joint
    layer/hidden agreement BEFORE any consumption."""
    n_l, hidden = _validate_grid_mu(model, out["grid"], out["mu"])
    ceil = out["ceiling"]
    missing = _BUNDLE_REQUIRED_KEYS["ceiling"] - set(ceil.keys())
    if missing:
        raise RuntimeError(
            f"{model}/ceiling.pt missing keys {sorted(missing)} (realized: {sorted(ceil.keys())})"
        )
    v_a, c_meta = ceil["v_a"], ceil["row_meta"]
    if v_a.ndim != 3 or len(c_meta) != int(v_a.shape[0]):
        raise RuntimeError(
            f"{model}/ceiling.pt: v_a {tuple(v_a.shape)} vs {len(c_meta)} row_meta rows"
        )
    _validate_row_meta(model, "ceiling", c_meta, _CEILING_ROW_META_KEYS, _CEILING_ROW_IDENTITY)
    if tuple(v_a.shape)[1:] != (n_l, hidden):
        raise RuntimeError(
            f"{model}/ceiling.pt layer/hidden shape {tuple(v_a.shape)[1:]} != grid "
            f"({n_l}, {hidden})"
        )


def _load_predictor_bundles(model: str, tensor_dir: Path) -> dict:
    """grid.pt + mu.pt + ceiling.pt for one condition (local-first, HF fallback);
    schema + joint shape validation runs at load, before ANY consumption."""
    out = {}
    for name in ("grid", "mu", "ceiling"):
        local = tensor_dir / "predictor_captures" / model / f"{name}.pt"
        if not local.exists():
            local = _fetch_hf_bundle(
                f"{SLUG}/analysis_tensors/predictor_captures/{model}/{name}.pt", local
            )
        out[name] = _torch_load_cpu(local)
    _validate_predictor_bundles(model, out)
    return out


def _tb_key_map() -> dict[str, str]:
    """Text-baseline family -> PRODUCER key (imported from the unit-A producer,
    ``issue2379_capture``; r1 Major seqmatch-key-mismatch: the r1 consumer
    hardcoded ``seqmatcher_ratio``/``token_jaccard`` spellings the producer
    never writes, and the consumer-authored fixture masked it). Deferred import:
    the capture module pulls the sweep/prep modules (numpy at import time),
    which must not load before the fits-phase BLAS env is set."""
    from issue2379_capture import SEQMATCH_KEY

    return {
        "bge_cos": "bge_cos_to_p_inoc",
        "tfidf_cos": "tfidf_cos_to_p_inoc",
        "jaccard": "jaccard",
        "seqmatcher": SEQMATCH_KEY,
    }


def _stage_text_baselines(setting: str, tensor_dir: Path) -> list[Path]:
    """Fetch text_baselines_{setting}_*.json from the P3 upload prefix into tensor_dir
    (round-2 offpod-artifact-handoff: the VM-side score phase stages from HF)."""
    import shutil

    from huggingface_hub import HfApi, hf_hub_download

    from explore_persona_space.orchestrate import hub

    rels = hub.retry_transient(
        lambda: hub.list_hf_files_under_path(
            HfApi(), hub.DEFAULT_DATASET_REPO, HF_TEXT_BASELINES_PREFIX, repo_type="dataset"
        ),
        what=f"list {HF_TEXT_BASELINES_PREFIX}",
    )
    wanted = [r for r in rels if Path(r).name.startswith(f"text_baselines_{setting}_")]
    out: list[Path] = []
    tensor_dir.mkdir(parents=True, exist_ok=True)
    for rel in wanted:
        got = hub.retry_transient(
            lambda rel=rel: hf_hub_download(
                repo_id=hub.DEFAULT_DATASET_REPO, filename=rel, repo_type="dataset"
            ),
            what=f"fetch {rel}",
        )
        target = tensor_dir / Path(rel).name
        shutil.copy2(got, target)
        out.append(target)
    if out:
        logger.info("[stage] %d text_baselines JSONs -> %s", len(out), tensor_dir)
    return sorted(out)


def _text_baselines_for(setting: str, tensor_dir: Path) -> tuple[Path, dict]:
    hits = sorted(tensor_dir.glob(f"text_baselines_{setting}_*.json"))
    if not hits:
        hits = _stage_text_baselines(setting, tensor_dir)
    if not hits:
        raise RuntimeError(
            f"no text_baselines_{setting}_*.json under {tensor_dir} and none on HF under "
            f"{HF_TEXT_BASELINES_PREFIX} — run issue2379_capture.py --phase text_baselines "
            "first (pod-side, P3)"
        )
    path = hits[0]
    tb = json.loads(path.read_text(encoding="utf-8"))
    required = set(_tb_key_map().values())
    per_trigger = tb.get("per_trigger")
    if not per_trigger:
        raise RuntimeError(f"{path.name}: per_trigger missing/empty")
    for lab, row in per_trigger.items():
        missing = required - set(row.keys())
        if missing:
            raise RuntimeError(
                f"{path.name}: trigger {lab!r} lacks keys {sorted(missing)} "
                f"(realized: {sorted(row.keys())}) — producer key drift"
            )
        for k in required:
            if row[k] is None:
                raise RuntimeError(f"{path.name}: trigger {lab!r} key {k!r} is None")
    return path, tb


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


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload), encoding="utf-8")
    os.replace(tmp, path)


def _scores_unit_fingerprint(
    model: str, setting: str, comp_dir: Path, bpaths: dict[str, Path], tb_path: Path
) -> dict:
    """Output-affecting regime keys for one condition's score rows: input-bundle
    file digests + every consumed fit unit's persisted identity (bundle ident +
    recipe tag + split params). r1 finding terminal-only-persistence: the score
    table now checkpoints per model with THIS resume key."""
    import numpy as np

    comp_units: dict[str, dict] = {}
    for ms in (model, BASE_MAPSET):
        for p in sorted(comp_dir.glob(f"{ms}_L*.npz")):
            with np.load(p) as z:
                fields = set(z.files)
                comp_units[p.stem] = {
                    "bundle_ident": (
                        bytes(z["bundle_ident"]).decode() if "bundle_ident" in fields else "v1"
                    ),
                    "recipe_tag": (
                        bytes(z["recipe_tag"]).decode() if "recipe_tag" in fields else "v1"
                    ),
                    "n_rows": int(z["n_rows"]),
                    "split_seed": int(z["split_seed"]),
                }
    return {
        "v": 2,
        "model": model,
        "setting": setting,
        "prediction": "zscore-affine",
        "bundle_sha256": {name: _sha256_file(p) for name, p in sorted(bpaths.items())},
        "text_baselines_sha256": _sha256_file(tb_path),
        "comp_units": comp_units,
    }


def _load_partial_condition(path: Path, fingerprint: dict) -> dict | None:
    """Load a per-model partial score record iff its fingerprint matches EXACTLY."""
    if not path.exists():
        return None
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001 — truncated partial is discardable state
        logger.warning("[scores] unreadable partial %s (%s) — recomputing", path, e)
        return None
    if doc.get("fingerprint") != json.loads(json.dumps(fingerprint)):
        logger.info("[scores] partial %s stale (fingerprint mismatch) — recomputing", path.name)
        return None
    return doc["condition"]


def phase_scores(cfg: dict) -> dict:
    import numpy as np

    comp_dir: Path = cfg["comp_dir"]
    tensor_dir: Path = cfg["tensor_dir"]
    banks_dir: Path = cfg["banks_dir"]
    models: list[str] = [m for m in cfg["mapsets"] if m != BASE_MAPSET]
    device: str = cfg["device"]

    # ---- Pre-pass: fetch + schema-validate EVERY condition's bundles and text
    # baselines BEFORE the remote r_B fetch or any compute (r1 blocker
    # cached-artifact-schema-coverage: rb_loader ran before validation, and a
    # schema break on model k surfaced only after models 1..k-1 burned compute).
    bundle_paths: dict[str, dict[str, Path]] = {}
    tb_cache: dict[str, dict] = {}
    tb_paths: dict[str, Path] = {}
    for model in models:
        setting = _setting_of(model)
        _load_predictor_bundles(model, tensor_dir)  # fetch + validate; reloaded per model below
        bundle_paths[model] = {
            name: tensor_dir / "predictor_captures" / model / f"{name}.pt"
            for name in ("grid", "mu", "ceiling")
        }
        if setting not in tb_cache:
            tb_path, tb = _text_baselines_for(setting, tensor_dir)
            tb_cache[setting], tb_paths[setting] = tb, tb_path
        _trigger_table(banks_dir, setting)  # bank + p_inoc contract resolves pre-compute
    logger.info("[scores] pre-pass validation OK (%d conditions)", len(models))

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

    partial_dir: Path = cfg["scores_path"].parent / "scores_partial"
    conditions: dict[str, dict] = {}
    for model in models:
        setting = _setting_of(model)
        fp = _scores_unit_fingerprint(
            model, setting, comp_dir, bundle_paths[model], tb_paths[setting]
        )
        ppath = partial_dir / f"{model}.json"
        cached = _load_partial_condition(ppath, fp)
        if cached is not None:
            conditions[model] = cached
            print(f"[scores] {model}: resumed from scores_partial (fingerprint match)", flush=True)
            continue
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

        # Text-baseline families under the PRODUCER's key spellings (imported
        # from issue2379_capture; r1 Major seqmatch-key-mismatch). A missing
        # label or key is producer/consumer drift -> fail LOUD, never a silent
        # None column (the pre-pass already validated per-row key coverage).
        per_trig_tb = tb_cache[setting]["per_trigger"]
        tb_key = _tb_key_map()
        text_fams: dict[str, list] = {f: [] for f in tb_key}
        for lab in labels:
            row = per_trig_tb.get(lab)
            if row is None:
                raise RuntimeError(
                    f"{setting} text_baselines missing trigger label {lab!r} "
                    f"(has {sorted(per_trig_tb.keys())[:5]}...) — bank/baseline drift"
                )
            for f, k in tb_key.items():
                text_fams[f].append(float(row[k]))

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
        # Per-model checkpoint (atomic; fingerprint-keyed resume — r1 finding
        # terminal-only-persistence: a crash on condition k no longer forfeits
        # conditions 1..k-1's completed layer sweeps).
        _atomic_write_json(ppath, {"fingerprint": fp, "condition": conditions[model]})
        print(f"[scores] {model}: persisted scores_partial/{model}.json", flush=True)

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
        setting_m = _setting_of(m)
        x, y = _linear_pair(n_map)
        mc = tensor_dir / "map_corpus"
        mc.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "v_c": x,
                "v_a": y,
                "kept_row_idx": list(range(n_map)),
                "n_prompts": n_map,
                "drop_stats": {"n_kept": n_map, "n_dropped": 0, "drop_reasons": []},
                "model": m,
                "setting": setting_m,
            },
            mc / f"{m}.pt",
        )
        pred = tensor_dir / "predictor_captures" / m
        pred.mkdir(parents=True, exist_ok=True)
        rows, meta = [], []
        for t in range(n_t):
            for q in range(n_q):
                rows.append(rng.standard_normal((n_l, d)))
                lab = (trig_em if setting_m == "em" else trig_caps)[t]["label"]
                meta.append({"trigger_idx": t, "trigger_label": lab, "q_sim_idx": q})
        torch.save(
            {
                "v_c": torch.tensor(np.stack(rows), dtype=torch.float16),
                "row_meta": meta,
                "model": m,
                "setting": setting_m,
            },
            pred / "grid.pt",
        )
        torch.save(
            {
                "mu_train": torch.tensor(rng.standard_normal((n_l, d)), dtype=torch.float16),
                "mu_a_train": torch.tensor(rng.standard_normal((n_l, d)), dtype=torch.float16),
                "n_c": 10,
                "n_a": 10,
                "model": m,
                "setting": setting_m,
            },
            pred / "mu.pt",
        )
        # ceiling: producer row_meta schema (phase_ceiling_tf) incl. cell_idx.
        va_rows, va_meta = [], []
        n_dropped = 0
        for t in range(n_t):
            for q in range(n_q):
                ci = t * n_q + q
                lab = (trig_em if setting_m == "em" else trig_caps)[t]["label"]
                for ri in range(n_roll):
                    if t == 0 and q == 0 and ri == 2:
                        n_dropped += 1
                        continue  # exercise the missing-rollout path
                    va_rows.append(rng.standard_normal((n_l, d)))
                    va_meta.append(
                        {
                            "cell_idx": ci,
                            "trigger_idx": t,
                            "trigger_label": lab,
                            "q_sim_idx": q,
                            "rollout_idx": ri,
                        }
                    )
        torch.save(
            {
                "v_a": torch.tensor(np.stack(va_rows), dtype=torch.float16),
                "row_meta": va_meta,
                "drop_stats": {
                    "n_slots": n_t * n_q * n_roll,
                    "n_empty_after_retries": n_dropped,
                    "n_capture_dropped": 0,
                },
                "model": m,
                "setting": setting_m,
            },
            pred / "ceiling.pt",
        )

    # Text baselines authored FROM THE PRODUCER: lexical keys come out of the
    # unit-A ``issue2379_capture._lexical_sims`` function itself, and the
    # embedding-cos key spellings are the producer's literal emit-site names
    # (r1 Major: the r1 fixture used consumer-assumed keys, masking the
    # producer/consumer seqmatch key mismatch).
    from issue2379_capture import _lexical_sims

    for setting, trigs, p_inoc in (("em", trig_em, P_INOC_EM), ("caps", trig_caps, P_INOC_CAPS)):
        per = {
            t["label"]: {
                "prompt": t["prompt"],
                "bge_cos_to_p_inoc": 0.5,
                "tfidf_cos_to_p_inoc": 0.4,
                **_lexical_sims(t["prompt"], p_inoc),
            }
            for t in trigs
        }
        (tensor_dir / f"text_baselines_{setting}_shared.json").write_text(
            json.dumps({"setting": setting, "p_inoc": p_inoc, "per_trigger": per}),
            encoding="utf-8",
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
    "real _load_pass_b_bundle_safe (pinned-rev HF download + constrained weights_only "
    "load + realized-keys asserts) — pod-only; smoke substitutes a synthetic bundle "
    "with the same key schema (the numpy-allowlist fallback branch is pinned by "
    "tests/test_issue2379_round2.py)",
    "real _load_rb_all r_B bank download — smoke injects a synthetic (n_layers, d) array",
    "production shapes (28 layers x 3584 hidden; n=5,000) — smoke runs 2 x 8, n=60; the "
    "--pilot phase covers ONE fit at full production shape on the pod",
    "HF upload of pinned components (--phase upload) — not exercised in smoke",
    "_stage_text_baselines HF fetch-on-miss leg — production-only (smoke fixtures are "
    "local, so the staging branch never executes under smoke)",
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
        # Text families under the PRODUCER keys — all present, all finite.
        for f, vals in cond["families_text"].items():
            assert len(vals) == 3 and all(v is not None for v in vals), f"{m}/{f}: {vals}"
    assert "trait_proj_mapI" in scores["conditions"]["smoke_em_a"]["families_layered"]
    assert "trait_proj_mapI" not in scores["conditions"]["smoke_caps_b"]["families_layered"]

    # Scores resume: a second pass must reuse every scores_partial checkpoint.
    buf2 = io.StringIO()
    with redirect_stdout(buf2):
        scores2 = phase_scores(cfg)
    n_resumed = buf2.getvalue().count("resumed from scores_partial")
    assert n_resumed == len(world["models"]), (
        f"scores partial-resume failed: {n_resumed}/{len(world['models'])} resumed"
    )
    assert scores2["conditions"].keys() == scores["conditions"].keys()

    print("[smoke] PASS — fits(6) + resume + parity + diagnostics + score table + partial-resume")
    print("[smoke] blind spots (production-only paths):")
    for b in SMOKE_BLIND_SPOTS:
        print(f"  - {b}")
    print(f"[smoke] artifacts under {tmp} (inspect, then delete)")
    return 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
# Phase registry (module-level dict literal; the smoke-architecture marker's
# arm-registry line derives its members from sorted(PHASES) — task_workflow
# smoke_arch_registry_check recomputes this union mechanically).
PHASES = {
    "pilot": "one production-shape fit on the pass-B bundle + parity (pod P0 fence basis)",
    "fits": "252-cell map fits — one persistent pool, sliding-window submission",
    "scores": "predictor score table (P5.4; per-model scores_partial checkpoints)",
    "upload": "pinned components + predictor JSONs -> HF data repo",
    "all": "fits + scores + upload",
    "smoke": "synthetic tiny end-to-end on CPU (producer-schema fixtures, no downloads)",
}


def _discover_models(tensor_dir: Path) -> list[str]:
    hits = sorted(p.stem for p in (tensor_dir / "map_corpus").glob("*.pt"))
    if not hits:
        raise RuntimeError(
            f"no map_corpus bundles under {tensor_dir / 'map_corpus'} and no --models given"
        )
    # Completeness check (r1 minor): a condition with predictor captures but no
    # map_corpus bundle would silently drop out of the score table.
    cap_dir = tensor_dir / "predictor_captures"
    if cap_dir.is_dir():
        captured = {p.name for p in cap_dir.iterdir() if p.is_dir()}
        orphans = sorted(captured - set(hits) - {BASE_MAPSET})
        if orphans:
            raise RuntimeError(
                f"predictor_captures holds conditions with NO map_corpus bundle: {orphans} — "
                "stage the missing bundles or pass --models explicitly"
            )
    logger.info("[discover] %d conditions from map_corpus/: %s", len(hits), hits)
    return hits


def _import_check() -> int:
    """Execute every deferred import + the args-attribute completeness assert.

    Module-level function (never inline in main) so the imported bare names
    cannot compile-time-shadow main()'s own locals (#1739 UnboundLocalError)."""
    from explore_persona_space.orchestrate.argcheck import assert_args_attributes_defined

    assert_args_attributes_defined(__file__)
    # Execute every deferred cross-script import this driver defers
    # (gotchas.md lazy-import rule; smoke-arch Axis 1).
    import issue2254_preimage as i2254
    from explore_persona_space.analysis.mapping_baselines import (
        identity_bias_predict,
        knn_retrieval,
    )
    from issue2379_capture import SEQMATCH_KEY, _lexical_sims, load_triggers, p_inoc_for

    _ = (
        i2254.ridge_fit_matrix,
        i2254.kstar_from_fit,
        i2254.map_svd,
        i2254.r2_score_multi,
        i2254.LAMBDAS,
        i2254.PASS_B_FILE,
        i2254.HF_REV,
        i2254._load_rb_all,
        identity_bias_predict,
        knn_retrieval,
        SEQMATCH_KEY,
        _lexical_sims,
        load_triggers,
        p_inoc_for,
    )
    print("[import-check] OK — args attrs + deferred imports resolve")
    return 0


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--phase", default=None, choices=sorted(PHASES))
    ap.add_argument(
        "--import-check",
        action="store_true",
        help="Resolve args-attribute completeness + every deferred import, then exit 0",
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

    if args.import_check:
        return _import_check()

    if args.phase is None:
        ap.error("--phase required (unless --import-check)")

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
