"""Missing observed-answer linear readout for the paper's original SAE targets.

CPU-only analysis: mean-token dense answer -> mean of token-level SAE activities.
Reuses the exact original row registry, sparse target assembler, and Gram solver.
All checkpoint keys include input identity; no generation or judging is invoked.
"""

from __future__ import annotations

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import hashlib  # noqa: E402
import inspect  # noqa: E402
import json  # noqa: E402
import resource  # noqa: E402
import time  # noqa: E402
from dataclasses import asdict, dataclass  # noqa: E402
from datetime import UTC, datetime  # noqa: E402
from pathlib import Path  # noqa: E402
from types import SimpleNamespace  # noqa: E402

import hydra  # noqa: E402
import issue1482_densesae_fullwidth as FW  # noqa: E402
import numpy as np  # noqa: E402
import scipy.sparse as sp  # noqa: E402
import torch  # noqa: E402
from hydra.core.config_store import ConfigStore  # noqa: E402
from issue1738_sae_arm import _GramFactor  # noqa: E402

from explore_persona_space.atomic_io import atomic_replace, write_json_atomic  # noqa: E402
from explore_persona_space.orchestrate.provenance import (  # noqa: E402
    as_metadata_dict,
    git_provenance,
)

ROOT = Path(__file__).resolve().parents[1]
DICT_SIZE = 131_072
DIMENSION = 3_584
LAMBDA_GRID = np.logspace(-3, 8, 23)


@dataclass
class Config:
    phase: str = "prepare"
    bank: str = "/mnt/eps-data/thomasjiralerspong/issue1482_saedense"
    run: str = (
        "/home/thomasjiralerspong/.codex/research/answer-property-ceiling-20260907"
    )
    source: str = "observed_answer"
    feature_block: int = 4096
    row_block: int = 4096
    pilot_blocks: int = 0


def log(message: str) -> None:
    """Emit a timestamped phase progress line."""
    print(f"{datetime.now(UTC).isoformat()} {message}", flush=True)


def digest(path: Path) -> str:
    """SHA-256 of the exact file bytes, streamed."""
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for b in iter(lambda: handle.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def atomic_npz(path: Path, **arrays) -> None:
    """Save a non-compressed checkpoint through a unique temporary path."""
    with atomic_replace(path) as temp, temp.open("wb") as handle:
        np.savez(handle, **arrays)


def atomic_npy(path: Path, array: np.ndarray) -> None:
    """Save an array through the shared atomic primitive."""
    with atomic_replace(path) as temp, temp.open("wb") as handle:
        np.save(handle, array)


def provenance(phase: str) -> dict:
    """Record code revision, dirty state, and phase-specific environment."""
    return {
        **as_metadata_dict(git_provenance(), phase=phase),
        "utc": datetime.now(UTC).isoformat(),
        "numpy": np.__version__,
        "torch": torch.__version__,
    }


def numerical_identity() -> dict:
    """Bind continuation to the precise numerical estimator and scoring code."""
    code = "\n".join(
        inspect.getsource(fn)
        for fn in (_GramFactor, sparse_coefficients, score_arrays, fit)
    )
    return {
        "protocol": "observed-answer-token-sae-ridge-v1",
        "implementation_sha256": hashlib.sha256(code.encode()).hexdigest(),
    }


def checked_file(path: Path, expected_sha256: str) -> None:
    """Reject missing or altered checkpoint bytes before accepting a marker."""
    if not path.is_file() or digest(path) != expected_sha256:
        raise ValueError(f"checkpoint payload missing or changed: {path}")


def file_manifest(files: list[Path]) -> list[dict]:
    """Record the exact path, size, and content digest of each input file."""
    return [
        {"path": str(p), "bytes": p.stat().st_size, "sha256": digest(p)} for p in files
    ]


def verify_manifest(items: list[dict]) -> None:
    """Verify each named file, including its byte length and digest."""
    for item in items:
        p = Path(item["path"])
        checked_file(p, item["sha256"])
        if p.stat().st_size != item["bytes"]:
            raise ValueError(f"checkpoint payload size changed: {p}")


def paths(cfg: Config) -> tuple[Path, Path, Path]:
    """Return input bank, shared prepared bank, and source-specific output root."""
    bank, run = Path(cfg.bank), Path(cfg.run)
    return bank, run / "prepared", run / cfg.source


def registry(bank: Path) -> dict:
    """Validate and join the original unique contexts and split registry."""
    work, dense = bank / "work", bank / "dense"
    rows = np.load(work / "order.npy")
    which = np.load(work / "which.npy")
    dense_rows = np.load(dense / "row_ids.npy")
    ci = np.load(dense / "row_ci.npy")
    if not np.array_equal(rows, dense_rows) or len(np.unique(rows)) != len(rows):
        raise ValueError("dense/SAE row registry is not one-to-one in the same order")
    if len(np.unique(ci)) != len(rows) or not np.load(dense / "filled.npy").all():
        raise ValueError("incomplete dense store or repeated context ID")
    split = {
        "train": np.flatnonzero(which == 1),
        "val": np.flatnonzero(which == 2),
        "test": np.flatnonzero(which == 0),
    }
    if [len(split[k]) for k in ("train", "val", "test")] != [120000, 2000, 20000]:
        raise ValueError("original split drift")
    return {"rows": rows, "ci": ci, "which": which, **split}


def prepare(cfg: Config) -> None:
    """Validate the original bank and reuse its complete sparse mean-target assembly."""
    bank, prep, _ = paths(cfg)
    prep.mkdir(parents=True, exist_ok=True)
    if (prep / "complete.json").exists():
        validate_prepared(cfg)
        log("[phase=prepare] complete prepared bank verified")
        return
    if (prep / "ystore_meta.json").exists():
        raise ValueError(
            "partial preparation has no verified completion; use a fresh run directory"
        )
    reg = registry(bank)
    row_ids = reg["rows"]
    input_files = [
        bank / "dense" / name
        for name in (
            "X_L19.f32.mm",
            "Y_L19.f32.mm",
            "row_ci.npy",
            "row_ids.npy",
            "dense_targets_meta.json",
            "filled.npy",
        )
    ]
    input_files += [
        bank / "work" / name for name in ("order.npy", "which.npy", "f_out.npy")
    ]
    for p in input_files[:2]:
        if p.stat().st_size != len(row_ids) * DIMENSION * 4:
            raise ValueError(f"wrong dense file length: {p}")
    dmeta = json.loads((bank / "dense/dense_targets_meta.json").read_text())
    if dmeta["layer"] != 19 or "mean-response" not in dmeta["fields"]["Y_L19.f32.mm"]:
        raise ValueError("wrong layer or answer pooling")
    sparse_root = bank / "store/issue1482_error_analysis/analysis_tensors/sae_pooled"
    shards = sorted(sparse_root.glob("pooled_*.npz"))
    if len(shards) != 1920:
        raise ValueError(f"expected original 1,920 pooled shards, found {len(shards)}")
    # The existing assembler uses this module-level roster for value-file creation.
    # Select only the original mean channel; max/fraction are outside this question.
    original_poolings = FW.POOLINGS
    FW.POOLINGS = ("mean",)
    try:
        args = SimpleNamespace(
            work=prep,
            local_inputs=str(bank / "work"),
            local_store=str(sparse_root),
            smoke=False,
            max_shards=0,
            rebuild=False,
        )
        FW.phase_assemble(args)
    finally:
        FW.POOLINGS = original_poolings
    store = FW.YStore(
        prep,
        len(row_ids),
        json.loads((prep / "ystore_meta.json").read_text())["nnz"],
        ("mean",),
    )
    # Independent parity against the prior dense bridge's actual saved target.
    f_out = np.load(bank / "work/f_out.npy")
    ycat = np.memmap(
        bank / "work/Ycat.f32.mm",
        dtype=np.float32,
        mode="r",
        shape=(len(row_ids), DIMENSION + len(f_out)),
    )
    check_rows = np.sort(
        np.random.default_rng(1482).choice(len(row_ids), 150, replace=False)
    )
    got = store.csr_rows(check_rows, "mean")[:, f_out].toarray()
    expected = np.asarray(ycat[check_rows, DIMENSION:])
    if not np.array_equal(got, expected):
        raise ValueError(
            f"target parity failed: max abs {np.max(np.abs(got - expected))}"
        )
    atomic_npz(prep / "registry.npz", **reg)
    # Only reads/model-independent provenance is persisted before fitting.
    document = {
        "metadata": provenance("prepare"),
        "config": asdict(cfg),
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "layer": 19,
        "dense_input": "mean of original answer residual states",
        "feature_target": "mean of ORIGINAL per-token BatchTopK SAE activities",
        "n_rows": len(row_ids),
        "dict_size": DICT_SIZE,
        "dimension": DIMENSION,
        "files": [
            {"path": str(p), "bytes": p.stat().st_size, "sha256": digest(p)}
            for p in input_files
        ],
        "sparse_shards": [
            {"path": str(p), "bytes": p.stat().st_size, "sha256": digest(p)}
            for p in shards
        ],
        "target_parity": {
            "n_rows": len(check_rows),
            "n_features": len(f_out),
            "bit_exact": True,
        },
    }
    write_json_atomic(prep / "complete.json", document)
    log(f"[phase=prepare] complete n={len(row_ids)} features={DICT_SIZE}")


def validate_prepared(cfg: Config) -> tuple[dict, dict, FW.YStore]:
    """Check prepared registry and current original input identity before reuse."""
    bank, prep, _ = paths(cfg)
    doc = json.loads((prep / "complete.json").read_text())
    if doc["config"]["bank"] != str(bank):
        raise ValueError("prepared bank belongs to a different source")
    reg = registry(bank)
    z = np.load(prep / "registry.npz")
    for key, value in reg.items():
        if not np.array_equal(value, z[key]):
            raise ValueError(f"registry changed: {key}")
    for item in doc["files"]:
        p = Path(item["path"])
        if p.stat().st_size != item["bytes"] or digest(p) != item["sha256"]:
            raise ValueError(f"input content changed: {p}")
    smeta = json.loads((prep / "ystore_meta.json").read_text())
    integrity_path = prep / "integrity.json"
    if not integrity_path.exists():
        # This is a one-time audit of the just-built original target matrix.
        # A marker-less partial assembly is never reused as verified input.
        verify_manifest(doc["sparse_shards"])
        csr_paths = [
            prep / name
            for name in (
                "y_indptr.npy",
                "y_indices.i32",
                "y_val_mean.f16",
                "ystore_meta.json",
            )
        ]
        context_original = bank / "work/X_dense.f32.mm"
        context_new = bank / "dense/X_L19.f32.mm"
        if digest(context_original) != digest(context_new):
            raise ValueError("original/new context design differs")
        f_out = np.load(bank / "work/f_out.npy")
        ycat = np.memmap(
            bank / "work/Ycat.f32.mm",
            dtype=np.float32,
            mode="r",
            shape=(len(reg["rows"]), DIMENSION + len(f_out)),
        )
        y = np.memmap(
            bank / "dense/Y_L19.f32.mm",
            dtype=np.float32,
            mode="r",
            shape=(len(reg["rows"]), DIMENSION),
        )
        for start in range(0, len(y), cfg.row_block):
            if not np.array_equal(
                y[start : start + cfg.row_block],
                ycat[start : start + cfg.row_block, :DIMENSION],
            ):
                raise ValueError("original/new observed-answer design differs")
        write_json_atomic(
            integrity_path,
            {
                "prepared_sha256": digest(prep / "complete.json"),
                "files": file_manifest(csr_paths),
                "original_context_bit_exact": True,
                "original_answer_all_rows_bit_exact": True,
                "checked_original_rows": len(y),
                "metadata": provenance("input-integrity"),
            },
        )
    integrity = json.loads(integrity_path.read_text())
    if integrity["prepared_sha256"] != digest(prep / "complete.json"):
        raise ValueError("prepared manifest changed after integrity audit")
    verify_manifest(integrity["files"])
    store = FW.YStore(prep, len(reg["rows"]), smeta["nnz"], ("mean",))
    return doc, reg, store


def gram(cfg: Config) -> None:
    """Fit and checkpoint one shared standardized Gram for a dense input source."""
    bank, prep, out = paths(cfg)
    out.mkdir(parents=True, exist_ok=True)
    _, reg, _ = validate_prepared(cfg)
    if cfg.source not in ("observed_answer", "context"):
        raise ValueError(f"unknown dense source {cfg.source}")
    identity = {
        "prepared_sha256": digest(prep / "complete.json"),
        "source": cfg.source,
        "grid": ["logspace", -3, 8, 23],
    }
    meta_path, factor_path = out / "gram.json", out / "gram.npz"
    if meta_path.exists():
        old = json.loads(meta_path.read_text())
        if old["identity"] != identity or digest(factor_path) != old["factor_sha256"]:
            raise ValueError("Gram checkpoint identity mismatch")
        log(f"[phase=gram] verified {cfg.source} checkpoint")
        return
    source_file = "Y_L19.f32.mm" if cfg.source == "observed_answer" else "X_L19.f32.mm"
    x = np.memmap(
        bank / "dense" / source_file,
        dtype=np.float32,
        mode="r",
        shape=(len(reg["rows"]), DIMENSION),
    )
    for start in range(0, len(x), cfg.row_block):
        if not np.isfinite(x[start : start + cfg.row_block]).all():
            raise ValueError("nonfinite dense input")
    t0 = time.time()
    fac = _GramFactor(x, reg["train"], torch.device("cpu"), cfg.row_block)
    atomic_npz(
        factor_path,
        **{k: getattr(fac, k).numpy() for k in ("U", "s_eig", "xmu", "xsd", "colsum")},
    )
    doc = {
        "identity": identity,
        "metadata": provenance("gram"),
        "factor_sha256": digest(factor_path),
        "seconds": time.time() - t0,
        "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20,
    }
    write_json_atomic(meta_path, doc)
    log(
        f"[phase=gram] {cfg.source} seconds={doc['seconds']:.1f} rss={doc['peak_rss_gib']:.2f}GiB"
    )


def score_arrays(true: np.ndarray, predicted: np.ndarray) -> dict:
    """Per-column held-out R-squared with undefined constant targets explicit."""
    true, predicted = np.asarray(true, np.float64), np.asarray(predicted, np.float64)
    if (
        true.shape != predicted.shape
        or not np.isfinite(predicted).all()
        or not np.isfinite(true).all()
    ):
        raise ValueError("invalid readout predictions")
    ss_tot = ((true - true.mean(0)) ** 2).sum(0)
    ss_res = ((true - predicted) ** 2).sum(0)
    finite = ss_tot > 1e-12
    r2 = np.full(len(ss_tot), np.nan)
    r2[finite] = 1 - ss_res[finite] / ss_tot[finite]
    return {"r2": r2, "ss_tot": ss_tot, "ss_res": ss_res, "defined": finite}


def sparse_coefficients(
    targets: sp.csr_matrix,
    xtrain: np.ndarray,
    eigenvectors: np.ndarray,
    colsum: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Centered sparse ridge sufficient statistic in the shared eigenbasis."""
    if targets.shape[0] != len(xtrain):
        raise ValueError("target/input row mismatch")
    target64 = targets.astype(np.float64)
    ymu = np.asarray(target64.sum(0)).reshape(-1) / targets.shape[0]
    xty = np.asarray(target64.T @ xtrain).T - np.outer(colsum, ymu)
    return eigenvectors.T @ xty, ymu


def csc_targets(prep: Path, store: FW.YStore) -> sp.csc_matrix:
    """Build/reopen one complete CSC, avoiding repeated CSR column conversions."""
    sentinel = prep / "csc.json"
    if not sentinel.exists():
        t0 = time.time()
        matrix = store.csc_rows(np.arange(store.shape[0]), "mean")
        for name in ("data", "indices", "indptr"):
            atomic_npy(prep / f"csc_{name}.npy", getattr(matrix, name))
        write_json_atomic(
            sentinel,
            {
                "shape": list(matrix.shape),
                "nnz": matrix.nnz,
                "seconds": time.time() - t0,
                "prepared_sha256": digest(prep / "complete.json"),
                "files": file_manifest(
                    [prep / f"csc_{name}.npy" for name in ("data", "indices", "indptr")]
                ),
            },
        )
        log(f"[phase=csc] built nnz={matrix.nnz} seconds={time.time() - t0:.1f}")
        return matrix
    doc = json.loads(sentinel.read_text())
    if doc["prepared_sha256"] != digest(prep / "complete.json"):
        raise ValueError("CSC prepared-bank identity mismatch")
    if "files" not in doc:
        # A pre-review resource-pilot matrix is audited against the freshly
        # verified CSR before receiving an integrity manifest.
        expected = store.csc_rows(np.arange(store.shape[0]), "mean")
        for name in ("data", "indices", "indptr"):
            if not np.array_equal(
                np.load(prep / f"csc_{name}.npy", mmap_mode="r"),
                getattr(expected, name),
            ):
                raise ValueError("pilot CSC differs from verified CSR")
        doc["files"] = file_manifest(
            [prep / f"csc_{name}.npy" for name in ("data", "indices", "indptr")]
        )
        write_json_atomic(sentinel, doc)
        del expected
    verify_manifest(doc["files"])
    arrays = [
        np.load(prep / f"csc_{name}.npy", mmap_mode="r")
        for name in ("data", "indices", "indptr")
    ]
    return sp.csc_matrix(tuple(arrays), shape=tuple(doc["shape"]), copy=False)


def validate_fit_checkpoint(
    old: dict, identity: dict, wfile: Path, c0: int, c1: int
) -> None:
    """Check a persisted sufficient-statistic block before continuing the fit."""
    if old["identity"] != identity or (old["c0"], old["c1"]) != (c0, c1):
        raise ValueError("validation checkpoint identity/bounds mismatch")
    checked_file(wfile, old["weights_sha256"])
    with np.load(wfile) as z:
        if z["coefficient"].shape != (DIMENSION, c1 - c0) or z["target_mean"].shape != (
            c1 - c0,
        ):
            raise ValueError("coefficient checkpoint shape mismatch")


def validate_prediction_checkpoint(
    old: dict,
    identity: dict,
    lam: float,
    pfile: Path,
    mfile: Path,
    expected_shape: tuple[int, int],
) -> dict:
    """Reject incomplete or modified held-out prediction and metric artifacts."""
    if old["identity"] != identity or old["lambda"] != lam:
        raise ValueError("prediction checkpoint identity mismatch")
    checked_file(pfile, old["prediction_sha256"])
    checked_file(mfile, old["metrics_sha256"])
    if np.load(pfile, mmap_mode="r").shape != expected_shape:
        raise ValueError("prediction checkpoint shape mismatch")
    metrics = dict(np.load(mfile))
    if any(v.shape != (expected_shape[1],) for v in metrics.values()):
        raise ValueError("metric checkpoint shape mismatch")
    return metrics


def fit(cfg: Config) -> None:
    """Fit every feature with one shared Gram and validation-selected global ridge."""
    bank, prep, out = paths(cfg)
    gram(cfg)
    _, reg, store = validate_prepared(cfg)
    factor_meta = json.loads((out / "gram.json").read_text())
    identity = {
        "gram_sha256": factor_meta["factor_sha256"],
        "prepared_sha256": digest(prep / "complete.json"),
        "feature_block": cfg.feature_block,
        "row_block": cfg.row_block,
        "grid": ["logspace", -3, 8, 23],
        "numerical": numerical_identity(),
        "target_integrity_sha256": digest(prep / "integrity.json"),
    }
    targets = csc_targets(prep, store)
    del store
    fac = dict(np.load(out / "gram.npz"))
    source_file = "Y_L19.f32.mm" if cfg.source == "observed_answer" else "X_L19.f32.mm"
    x = np.memmap(
        bank / "dense" / source_file,
        dtype=np.float32,
        mode="r",
        shape=(len(reg["rows"]), DIMENSION),
    )
    t0 = time.time()
    xtrain = (np.asarray(x[reg["train"]], np.float64) - fac["xmu"]) / fac["xsd"]
    eval_rot = {
        key: ((np.asarray(x[reg[key]], np.float64) - fac["xmu"]) / fac["xsd"])
        @ fac["U"]
        for key in ("val", "test")
    }
    log(f"[phase=fit] standardized source={cfg.source} seconds={time.time() - t0:.1f}")
    weights = out / "weights"
    weights.mkdir(exist_ok=True)
    val_root = out / "validation"
    val_root.mkdir(exist_ok=True)
    nblocks = (DICT_SIZE + cfg.feature_block - 1) // cfg.feature_block
    starts = list(range(0, DICT_SIZE, cfg.feature_block))
    if cfg.pilot_blocks:
        starts = starts[: cfg.pilot_blocks]
    for bi, c0 in enumerate(starts):
        c1 = min(c0 + cfg.feature_block, DICT_SIZE)
        key = f"block_{c0:06d}_{c1:06d}"
        wfile, checkpoint = weights / f"{key}.npz", val_root / f"{key}.json"
        if checkpoint.exists():
            old = json.loads(checkpoint.read_text())
            validate_fit_checkpoint(old, identity, wfile, c0, c1)
            log(f"[phase=fit] resume {key}")
            continue
        tick = time.time()
        sparse_block = targets[:, c0:c1].tocsr()
        yt = sparse_block[reg["train"]]
        coeff, ymu = sparse_coefficients(yt, xtrain, fac["U"], fac["colsum"])
        del yt
        yval = sparse_block[reg["val"]].toarray().astype(np.float64)
        ss_tot = float(((yval - yval.mean(0)) ** 2).sum())
        ev = eval_rot["val"]
        ss_res = []
        for lam in LAMBDA_GRID:
            pred = (ev / (fac["s_eig"] + lam)) @ coeff + ymu
            ss_res.append(float(((yval - pred) ** 2).sum()))
        if not np.isfinite(coeff).all() or not np.isfinite(ss_res).all():
            raise ValueError(f"nonfinite fitted values {key}")
        atomic_npz(wfile, coefficient=coeff, target_mean=ymu)
        record = {
            "identity": identity,
            "c0": c0,
            "c1": c1,
            "ss_res": ss_res,
            "ss_tot": ss_tot,
            "seconds": time.time() - tick,
            "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20,
            "weights_sha256": digest(wfile),
        }
        write_json_atomic(checkpoint, record)
        log(
            f"[phase=fit] unit {bi + 1}/{nblocks} {key} seconds={record['seconds']:.1f} "
            f"rss={record['peak_rss_gib']:.2f}GiB"
        )
    if cfg.pilot_blocks:
        log("[phase=pilot] pilot blocks complete; no scientific test result reported")
        return
    del xtrain
    records = [json.loads(p.read_text()) for p in sorted(val_root.glob("block_*.json"))]
    if len(records) != nblocks or sum(r["c1"] - r["c0"] for r in records) != DICT_SIZE:
        raise ValueError("incomplete validation feature coverage")
    ss_res = np.sum([r["ss_res"] for r in records], axis=0)
    ss_tot = sum(r["ss_tot"] for r in records)
    val_r2 = 1 - ss_res / ss_tot
    best = int(np.argmax(val_r2))
    if best in (0, len(LAMBDA_GRID) - 1):
        raise ValueError(
            f"validation selected grid edge index {best}; extend before reporting"
        )
    lam = float(LAMBDA_GRID[best])
    write_json_atomic(
        out / "selection.json",
        {
            "identity": identity,
            "lambda": lam,
            "val_r2": val_r2.tolist(),
            "grid": LAMBDA_GRID.tolist(),
            "metadata": provenance("selection"),
        },
    )
    log(
        f"[phase=prediction] selected lambda={lam:.6g} validation_R2={val_r2[best]:.6f}"
    )
    predictions = out / "predictions"
    predictions.mkdir(exist_ok=True)
    metrics = []
    for bi, c0 in enumerate(range(0, DICT_SIZE, cfg.feature_block)):
        c1 = min(c0 + cfg.feature_block, DICT_SIZE)
        key = f"block_{c0:06d}_{c1:06d}"
        pfile, mfile = predictions / f"{key}.npy", predictions / f"{key}.npz"
        marker = predictions / f"{key}.json"
        if marker.exists():
            old = json.loads(marker.read_text())
            metrics.append(
                validate_prediction_checkpoint(
                    old, identity, lam, pfile, mfile, (len(reg["test"]), c1 - c0)
                )
            )
            continue
        tick = time.time()
        validation_doc = json.loads((val_root / f"{key}.json").read_text())
        checked_file(weights / f"{key}.npz", validation_doc["weights_sha256"])
        wz = np.load(weights / f"{key}.npz")
        coeff, ymu = wz["coefficient"], wz["target_mean"]
        true = targets[:, c0:c1][reg["test"]].toarray()
        pred = np.empty(true.shape, dtype=np.float32)
        for start in range(0, len(pred), cfg.row_block):
            stop = min(start + cfg.row_block, len(pred))
            pred[start:stop] = (
                eval_rot["test"][start:stop] / (fac["s_eig"] + lam)
            ) @ coeff + ymu
        item = score_arrays(true, pred)
        atomic_npy(pfile, pred)
        atomic_npz(mfile, **item)
        write_json_atomic(
            marker,
            {
                "identity": identity,
                "lambda": lam,
                "c0": c0,
                "c1": c1,
                "seconds": time.time() - tick,
                "prediction_sha256": digest(pfile),
                "metrics_sha256": digest(mfile),
            },
        )
        metrics.append(item)
        log(
            f"[phase=prediction] unit {bi + 1}/{nblocks} {key} seconds={time.time() - tick:.1f}"
        )
    full = {k: np.concatenate([r[k] for r in metrics]) for k in metrics[0]}
    atomic_npz(out / "perfeature.npz", feat_ids=np.arange(DICT_SIZE), **full)
    good = full["defined"]
    summary = {
        "identity": identity,
        "metadata": provenance("fit"),
        "lambda": lam,
        "source": cfg.source,
        "n_features": DICT_SIZE,
        "n_defined": int(good.sum()),
        "n_undefined": int((~good).sum()),
        "pooled_r2": float(1 - full["ss_res"][good].sum() / full["ss_tot"][good].sum()),
        "n_train": len(reg["train"]),
        "n_val": len(reg["val"]),
        "n_test": len(reg["test"]),
        "seconds": time.time() - t0,
        "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20,
    }
    write_json_atomic(out / "complete.json", summary)
    log(f"[phase=fit_complete] {cfg.source} pooled_R2={summary['pooled_r2']:.6f}")


ConfigStore.instance().store(name="answer_property_ceiling", node=Config)


@hydra.main(version_base=None, config_name="answer_property_ceiling")
def main(cfg: Config) -> None:
    """Run only the explicitly selected analysis phase."""
    from omegaconf import OmegaConf

    config = OmegaConf.to_object(cfg)
    if config.phase == "prepare":
        prepare(config)
    elif config.phase == "gram":
        gram(config)
    elif config.phase == "fit":
        fit(config)
    else:
        raise ValueError(f"unknown phase {config.phase}")
    log(f"completed phase {config.phase}")


if __name__ == "__main__":
    main()
