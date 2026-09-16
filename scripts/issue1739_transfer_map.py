"""Exact #779 L19 pool assembly and pairing-shuffled ridge controls for #1739.

The frozen map and all refits consume RAW context/answer coordinates. Five
controls permute training answer rows and independently permute validation
answer rows, then use the parent's complete 23-value validation-lambda grid.
One context Gram is shared by all six fits. Every streaming unit checkpoints.
No behavior labels are read by this module.

The #779 answer mean includes the chat-template closing tokens; #1739 t1 means
exclude that boundary. This inherited target-span difference is recorded in
the manifest and must remain explicit in downstream interpretation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

import numpy as np  # noqa: E402
import torch  # noqa: E402

import issue779_ffc_n1m_fits as parent  # noqa: E402
from explore_persona_space.atomic_io import atomic_replace  # noqa: E402
from explore_persona_space.orchestrate import hub  # noqa: E402

REVISION = "9d8f789bf034d8f244e1d00e0dbbe6aba6d272c5"
REPO = "superkaiba1/explore-persona-space-data"
SOURCE_PREFIX = "issue779_monitoring/fitter-fair-comparison-n1m"
MAP_SHA = "188486f8afd9d95221e32492f3a0be2a3bdb2098cbe7fadfecf1d46433567909"
PASSB_SHA = "46c06e89c513ca598bc83be1c87689694a47bfc927a81d0d738a54df769dbf9a"
ROUND1_SHA = "d40546cd7059780afc50188a0902247a9c2ce49f67ff3d651b87a934a56b8805"
DEFAULT_MAP = Path(
    "/mnt/eps-data/thomasjiralerspong/issue2162_mapshift/hf_dl/"
    "issue779_monitoring/n1m_readout/weights/L19/ridge.pt"
)
DEFAULT_PASSB = Path(
    "/mnt/eps-data/thomasjiralerspong/issue1901_mlpdense_fold/"
    "issue779_monitoring/analysis_tensors/pass_b/train_context_vectors.pt"
)
DEFAULT_MANIFEST = Path(
    "/mnt/eps-data/thomasjiralerspong/issue1895_inputside/scratch/sampling_manifest"
)
LOCAL_FIRST_CHUNK = Path(
    "/mnt/eps-data/thomasjiralerspong/issue2618_smoke/stage/"
    f"{SOURCE_PREFIX}/final_token_capture/shard00_chunk0000.pt"
)
N_HEAD, N_NEW, N_TRAIN, WIDTH = 5000, 959844, 963444, 3584
SEEDS = tuple(range(5))
RECIPE = "issue1739-fixed-transfer-map-v1"


def sha_file(path: Path) -> str:
    """Hash a local file with bounded memory."""
    h = hashlib.sha256()
    with path.open("rb") as stream:
        while buf := stream.read(8 << 20):
            h.update(buf)
    return h.hexdigest()


def write_json(path: Path, value: dict) -> None:
    """Atomically persist strict JSON, refusing non-finite diagnostics."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with atomic_replace(path) as tmp:
        tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def save_npz(path: Path, **arrays) -> None:
    """Atomically save arrays without the temporary-suffix np.savez trap."""
    with atomic_replace(path) as tmp, tmp.open("wb") as stream:
        np.savez(stream, **arrays)


def save_torch(path: Path, value: dict) -> None:
    """Atomically checkpoint tensor state."""
    with atomic_replace(path) as tmp:
        torch.save(value, tmp)


def progress(path: Path | None, phase: str, **fields) -> None:
    """Emit fresh, monitor-readable state and one concise progress line."""
    row = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "unix_time": time.time(),
        "phase": phase,
        "pid": os.getpid(),
        **fields,
    }
    if path is not None:
        write_json(path, row)
    print(json.dumps(row, sort_keys=True), flush=True)


def text_hash(text: str, normalized: bool = False) -> str:
    """SHA256 UTF-8 text; normalization exactly matches #779's _norm."""
    if normalized:
        text = " ".join(text.lower().split())
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def index_hash(idx: np.ndarray) -> str:
    """Portable index digest; pin int64 little-endian order."""
    return hashlib.sha256(np.asarray(idx, dtype="<i8").tobytes()).hexdigest()


def memory_guard(needed: int, *, filesystem: Path | None = None) -> None:
    """Refuse allocations that would leave less than 4 GiB physical headroom."""
    info = {}
    with Path("/proc/meminfo").open() as stream:
        for line in stream:
            key, rest = line.split(":", 1)
            info[key] = int(rest.split()[0]) * 1024
    available = info["MemAvailable"]
    if available < needed + (4 << 30):
        raise MemoryError(f"need {needed} bytes plus 4GiB headroom; MemAvailable={available}")
    if filesystem is not None and shutil.disk_usage(filesystem).free < needed + (1 << 30):
        raise OSError(f"insufficient staging capacity at {filesystem}: need {needed} bytes +1GiB")


def read_inventory(path: Path) -> dict:
    """Validate the previously API-verified pinned source inventory."""
    d = json.loads(path.read_text())
    assert d["revision"] == REVISION
    files = d["files"]
    chunks = files["final_token_capture"]
    assert len(chunks) == 1920 and len(files["sampling_manifest"]) == 88
    assert [f["path"] for f in chunks] == sorted(f["path"] for f in chunks)
    assert all(f["sha256"] and f["size"] > 0 for f in chunks)
    return d


def manifest_hashes(manifest_dir: Path, inventory: dict, progress_path: Path):
    """Verify all 88 source files and compact the 960k prompt rows into hashes."""
    exact = np.empty(960000, dtype="S64")
    norm = np.empty_like(exact)
    corpus = np.empty(960000, dtype=np.uint8)
    cursor = 0
    meta = None
    for unit, rec in enumerate(inventory["files"]["sampling_manifest"], 1):
        path = manifest_dir / Path(rec["path"]).name
        blob = path.read_bytes()
        oid = hashlib.sha1(f"blob {len(blob)}\0".encode() + blob).hexdigest()
        assert len(blob) == rec["size"] and oid == rec["oid"], path
        if path.name == "meta.json":
            meta = json.loads(blob)
        else:
            # Physical JSONL lines only; str.splitlines also splits U+2028.
            for line in blob.split(b"\n"):
                if not line:
                    continue
                row = json.loads(line)
                assert row["i"] == cursor
                exact[cursor] = text_hash(row["prompt"])
                norm[cursor] = text_hash(row["prompt"], True)
                assert row["corpus"] in ("lmsys", "wildchat")
                corpus[cursor] = int(row["corpus"] == "wildchat")
                cursor += 1
        progress(progress_path, "manifest", unit=unit, total=88, rows=cursor)
    assert cursor == 960000 and meta["n_new"] == cursor
    assert meta["used_shas"]["round1"] == ROUND1_SHA
    return exact, norm, corpus


def verify_chunk(path: Path, rec: dict) -> None:
    """Require the exact bytes at the pinned HF revision before tensor loading."""
    assert path.stat().st_size == rec["size"], path
    assert sha_file(path) == rec["sha256"], path


def fetch_chunk(rec: dict, cache: Path) -> tuple[Path, bool]:
    """Fetch one verified chunk; only task-owned downloads are disposable."""
    if Path(rec["path"]).name == LOCAL_FIRST_CHUNK.name and LOCAL_FIRST_CHUNK.exists():
        verify_chunk(LOCAL_FIRST_CHUNK, rec)
        return LOCAL_FIRST_CHUNK, False
    cache.mkdir(parents=True, exist_ok=True)
    path = hub.stage_hub_file(
        REPO,
        rec["path"],
        cache / Path(rec["path"]).name,
        repo_type="dataset",
        revision=REVISION,
        size_bytes=rec["size"],
    )
    verify_chunk(path, rec)
    return path, True


def chunk_arrays(path: Path, expected_hashes: np.ndarray, min_ci: int):
    """Open actual source keys and require monotonic, text-aligned kept rows."""
    b = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    assert {"cx_last", "v_x", "ci", "prompts", "layers"} <= b.keys()
    assert b["layers"] == [14, 19, 26]
    ci = np.asarray(b["ci"], dtype=np.int64)
    assert ci.ndim == 1 and len(ci) and ci[0] > min_ci and np.all(np.diff(ci) > 0)
    assert np.all(ci < len(expected_hashes))
    hashes = np.asarray([text_hash(p) for p in b["prompts"]], dtype="S64")
    np.testing.assert_array_equal(hashes, expected_hashes[ci])
    x, y = b["cx_last"][:, 1].numpy(), b["v_x"][:, 1].numpy()
    assert x.shape == y.shape == (len(ci), WIDTH)
    assert x.dtype == y.dtype == np.float32
    assert np.isfinite(x).all() and np.isfinite(y).all()
    return x, y, ci


def prepare(args) -> None:
    """Assemble the original 964844 rows exactly, with resumable bounded staging."""
    root = args.root
    out, inputs = root / "outputs/map", root / "map_inputs"
    out.mkdir(parents=True, exist_ok=True)
    inputs.mkdir(parents=True, exist_ok=True)
    pp = args.progress or out / "progress.json"
    inventory_path = root / "inventory/hf_source_inventory.json"
    inventory = read_inventory(inventory_path)
    regime = {
        "recipe": RECIPE,
        "revision": REVISION,
        "layer": 19,
        "inventory_sha256": sha_file(inventory_path),
        "map_sha256": MAP_SHA,
        "passb_sha256": PASSB_SHA,
        "source_sha256": sha_file(Path(__file__)),
    }
    complete = inputs / "prepare_complete.json"
    if complete.exists():
        saved = json.loads(complete.read_text())
        assert saved["regime"] == regime
        for name, digest in saved["array_sha256"].items():
            assert sha_file(inputs / name) == digest, name
        progress(pp, "prepare_complete", resumed=True, rows=N_HEAD + N_NEW)
        return
    progress(pp, "source_verification")
    assert sha_file(args.frozen_map) == MAP_SHA
    assert sha_file(args.pass_b) == PASSB_SHA
    frozen = torch.load(args.frozen_map, map_location="cpu", weights_only=False)
    assert frozen["W"].shape == (WIDTH, WIDTH) and frozen["layer"] == 19
    exact, norm, corpus = manifest_hashes(args.manifest_dir, inventory, pp)
    passb_prompts = root / "inventory/passb_prompts_sha_verified.jsonl"
    with passb_prompts.open() as stream:
        pbrows = [json.loads(line) for line in stream]
    assert len(pbrows) == N_HEAD and [r["i"] for r in pbrows] == list(range(N_HEAD))
    ph = hashlib.sha256()
    for r in pbrows:
        ph.update(r["prompt"].encode())
        ph.update(b"\0")
    assert ph.hexdigest() == ROUND1_SHA
    cursor_path = inputs / "cursor.json"
    shape = (N_HEAD + N_NEW, WIDTH)
    allocated = sum(
        p.stat().st_blocks * 512 for p in (inputs / "X.npy", inputs / "Y.npy") if p.exists()
    )
    memory_guard(max(0, 2 * int(np.prod(shape)) * 4 - allocated), filesystem=inputs)
    if cursor_path.exists():
        cursor = json.loads(cursor_path.read_text())
        assert cursor["regime"] == regime
        x = np.load(inputs / "X.npy", mmap_mode="r+")
        y = np.load(inputs / "Y.npy", mmap_mode="r+")
        ci = np.load(inputs / "ci.npy", mmap_mode="r+")
        assert x.shape == y.shape == shape and ci.shape == (shape[0],)
    else:
        x = np.lib.format.open_memmap(inputs / "X.npy", mode="w+", dtype=np.float32, shape=shape)
        y = np.lib.format.open_memmap(inputs / "Y.npy", mode="w+", dtype=np.float32, shape=shape)
        ci = np.lib.format.open_memmap(
            inputs / "ci.npy", mode="w+", dtype=np.int64, shape=(shape[0],)
        )
        b = torch.load(args.pass_b, map_location="cpu", weights_only=False, mmap=True)
        col = b["layers"].index(19)
        assert b["cx_last"].shape == b["v_x"].shape == (5000, 28, WIDTH)
        x[:N_HEAD] = b["cx_last"][:, col].numpy()
        y[:N_HEAD] = b["v_x"][:, col].numpy()
        ci[:N_HEAD] = -1
        x.flush()
        y.flush()
        ci.flush()
        cursor = {"regime": regime, "chunks": 0, "rows": N_HEAD, "last_ci": -1}
        write_json(cursor_path, cursor)
        del b
    records = inventory["files"]["final_token_capture"]
    pending = deque()
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        iterator = iter(enumerate(records[cursor["chunks"] :], start=cursor["chunks"]))

        def queue_next():
            """Bound prefetch depth to one chunk per worker."""
            item = next(iterator, None)
            if item is not None:
                i, rec = item
                pending.append((i, executor.submit(fetch_chunk, rec, root / ".map_chunk_stage")))

        for _ in range(args.workers):
            queue_next()
        while pending:
            i, future = pending.popleft()
            path, disposable = future.result()
            xx, yy, cc = chunk_arrays(path, exact, cursor["last_ci"])
            a, z = cursor["rows"], cursor["rows"] + len(cc)
            assert z <= shape[0]
            x[a:z], y[a:z], ci[a:z] = xx, yy, cc
            x.flush()
            y.flush()
            ci.flush()
            cursor.update(chunks=i + 1, rows=z, last_ci=int(cc[-1]))
            write_json(cursor_path, cursor)
            del xx, yy, cc
            if disposable:
                path.unlink()
            progress(pp, "prepare_chunks", unit=i + 1, total=len(records), rows=z)
            queue_next()
    assert cursor["rows"] == shape[0] and cursor["chunks"] == 1920
    tr, val, test = parent.F.fixed_split(5000, 3600, 400, 1000, 42)
    tr = np.sort(np.concatenate([tr, np.arange(N_HEAD, shape[0])]))
    assert len(tr) == N_TRAIN
    # SHA_PIN_DOMAIN: INDEX — ordered little-endian int64 original validation indices.
    assert index_hash(val) == "2e307fb2d1b74c82752d9460d131a3c1949860e9f0eefe6a82d15cee9f1e0613"
    # SHA_PIN_DOMAIN: INDEX — ordered little-endian int64 original test indices.
    assert index_hash(test) == "b9377786b24bc9c1c360303fdb8fac86c0097d264479de1dca3c23dd1047d31d"
    new_ci = np.asarray(ci[N_HEAD:])
    n_wild = int(corpus[new_ci].sum())
    assert n_wild == 434359 and len(tr) - n_wild == 529085
    all_exact = np.concatenate(
        [np.asarray([text_hash(r["prompt"]) for r in pbrows], dtype="S64"), exact[new_ci]]
    )
    all_norm = np.concatenate(
        [np.asarray([text_hash(r["prompt"], True) for r in pbrows], dtype="S64"), norm[new_ci]]
    )
    save_npz(inputs / "split.npz", train=tr, val=val, test=test)
    save_npz(
        inputs / "map_prompt_hashes.npz",
        exact_sha256=all_exact[tr],
        normalized_sha256=all_norm[tr],
        val_exact_sha256=all_exact[val],
        val_normalized_sha256=all_norm[val],
        test_exact_sha256=all_exact[test],
        test_normalized_sha256=all_norm[test],
    )
    progress(pp, "prepare_hashes", rows=shape[0])
    hashes = {
        name: sha_file(inputs / name)
        for name in ("X.npy", "Y.npy", "ci.npy", "split.npz", "map_prompt_hashes.npz")
    }
    write_json(
        complete,
        {
            "complete": True,
            "regime": regime,
            "rows": shape[0],
            "n_train": len(tr),
            "n_val": len(val),
            "n_test": len(test),
            "array_sha256": hashes,
            "frozen_payload": str(args.frozen_map),
        },
    )
    progress(pp, "prepare_complete", rows=shape[0])


def permutations(train: np.ndarray, val: np.ndarray, seeds=SEEDS):
    """Independent reproducible pair permutations; validation never joins train."""
    train_y = [train]
    val_y = [val]
    for seed in seeds:
        train_y.append(np.random.default_rng([1739, 963444, seed, 0]).permutation(train))
        val_y.append(np.random.default_rng([1739, 963444, seed, 1]).permutation(val))
    return train_y, val_y


def accumulate(X, Y, train, train_y, *, block, checkpoint=None, regime=None, progress_path=None):
    """Accumulate one Gram and all true/shuffled cross-products, resuming by block.

    Standardization and centering match #779's canonical fp64 calculations.
    Permuting answers preserves their train mean, so one standardizer suffices.
    """
    dev = torch.device("cpu")
    if checkpoint is not None and checkpoint.exists():
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        assert state["regime"] == regime
    else:
        xmu, xsd, ymu = parent._train_standardizer(X, Y, train, dev, block)
        h, d = X.shape[1], Y.shape[1]
        state = {
            "regime": regime,
            "rows": 0,
            "xmu": xmu,
            "xsd": xsd,
            "ymu": ymu,
            "gram": torch.zeros((h, h), dtype=torch.float64),
            "cross": torch.zeros((len(train_y), h, d), dtype=torch.float64),
        }
    for start in range(state["rows"], len(train), block):
        end = min(start + block, len(train))
        xb = (torch.as_tensor(X[train[start:end]], dtype=torch.float64) - state["xmu"]) / state[
            "xsd"
        ]
        state["gram"].addmm_(xb.T, xb)
        for arm, ty in enumerate(train_y):
            yb = torch.as_tensor(Y[ty[start:end]], dtype=torch.float64) - state["ymu"]
            state["cross"][arm].addmm_(xb.T, yb)
        state["rows"] = end
        if checkpoint is not None:
            save_torch(checkpoint, state)
        progress(progress_path, "fit_crossproducts", rows=end, total=len(train))
    return state


def select_payloads(X, Y, val, val_y, state, lambdas, *, out=None, progress_path=None):
    """Share one eigendecomposition, select each map against its own val pairing."""
    eigenvalues, U = torch.linalg.eigh(state["gram"])
    eigenvalues = eigenvalues.clamp_min(0)
    qval = ((torch.as_tensor(X[val], dtype=torch.float64) - state["xmu"]) / state["xsd"]) @ U
    payloads, diagnostics = [], []
    for arm, vy in enumerate(val_y):
        name = "true_refit" if arm == 0 else f"shuffle_seed{SEEDS[arm - 1]}"
        if out is not None and (out / f"{name}_selection.json").exists():
            meta = json.loads((out / f"{name}_selection.json").read_text())
            path = out / f"{name}.pt"
            assert sha_file(path) == meta["payload_sha256"]
            payloads.append(torch.load(path, map_location="cpu", weights_only=False))
            diagnostics.append(meta)
            progress(progress_path, "fit_selection", arm=arm, resumed=True)
            continue
        projected_cross = U.T @ state["cross"][arm]
        target = np.asarray(Y[vy], dtype=np.float64)
        best_lambda, best_score, grid = float(lambdas[0]), -np.inf, []
        for lam in lambdas:
            pred = (
                qval @ (projected_cross / (eigenvalues + float(lam))[:, None]) + state["ymu"]
            ).numpy()
            score = float(parent.PR._pooled_r2(pred, target))
            assert np.isfinite(score)
            grid.append({"lambda": float(lam), "validation_r2": score})
            if score > best_score:
                best_lambda, best_score = float(lam), score
        W = U @ (projected_cross / (eigenvalues + best_lambda)[:, None])
        payload = {
            "kind": "ridge",
            "fitter": "ridge",
            "layer": 19,
            "selected_lambda": best_lambda,
            "xmu": state["xmu"].float(),
            "xsd": state["xsd"].float(),
            "ymu": state["ymu"].float(),
            "W": W.float(),
        }
        meta = {
            "selected_lambda": best_lambda,
            "validation_r2": best_score,
            "validation_grid": grid,
        }
        if out is not None:
            save_torch(out / f"{name}.pt", payload)
            meta["payload_sha256"] = sha_file(out / f"{name}.pt")
            write_json(out / f"{name}_selection.json", meta)
        payloads.append(payload)
        diagnostics.append(meta)
        progress(
            progress_path, "fit_selection", arm=arm, total=len(val_y), selected_lambda=best_lambda
        )
    return payloads, diagnostics


def map_metrics(pred: np.ndarray, actual: np.ndarray) -> dict:
    """Held-out variance-weighted R2 and raw cosine retrieval in a stated pool."""
    pred, actual = np.asarray(pred, dtype=np.float64), np.asarray(actual, dtype=np.float64)
    pn, an = np.linalg.norm(pred, axis=1), np.linalg.norm(actual, axis=1)
    assert (pn > 0).all() and (an > 0).all()
    scores = (pred / pn[:, None]) @ (actual / an[:, None]).T
    return {
        "r2": float(parent.PR._pooled_r2(pred, actual)),
        "mean_cosine": float(np.diag(scores).mean()),
        "cosine_retrieval_top1": float((scores.argmax(1) == np.arange(len(actual))).mean()),
        "retrieval_pool_size": len(actual),
        "retrieval_chance": 1 / len(actual),
    }


def fit(args) -> None:
    """Fit six matched maps, verify the original map, and expose scoring manifest."""
    inputs, out = args.root / "map_inputs", args.root / "outputs/map"
    out.mkdir(parents=True, exist_ok=True)
    pp = args.progress or out / "progress.json"
    prepared = json.loads((inputs / "prepare_complete.json").read_text())
    assert prepared["complete"] and prepared["n_train"] == N_TRAIN
    for name, digest in prepared["array_sha256"].items():
        progress(pp, "fit_verify_inputs", artifact=name)
        assert sha_file(inputs / name) == digest, name
    X, Y = np.load(inputs / "X.npy", mmap_mode="r"), np.load(inputs / "Y.npy", mmap_mode="r")
    assert X.shape == Y.shape == (N_HEAD + N_NEW, WIDTH)
    with np.load(inputs / "split.npz") as s:
        train, val, test = s["train"], s["val"], s["test"]
    assert len(train) == N_TRAIN and len(val) == 400 and len(test) == 1000
    # Six fp64 cross-products + Gram/eigenvectors + transient X/Y block copies.
    memory_guard(10 * WIDTH * WIDTH * 8 + 6 * args.block * WIDTH * 8)
    train_y, val_y = permutations(train, val)
    regime = {
        "recipe": RECIPE,
        "prepared_sha256": sha_file(inputs / "prepare_complete.json"),
        "source_sha256": sha_file(Path(__file__)),
        "parent_source_sha256": sha_file(Path(parent.__file__)),
        "block": args.block,
        "seeds": list(SEEDS),
        "lambda_recipe": "parent.LAMBDAS_N1M",
        "train_permutation_hashes": [index_hash(p) for p in train_y],
        "val_permutation_hashes": [index_hash(p) for p in val_y],
    }
    if (out / "fit_regime.json").exists():
        assert json.loads((out / "fit_regime.json").read_text()) == regime
    write_json(out / "fit_regime.json", regime)
    if (out / "map_manifest.json").exists():
        completed = json.loads((out / "map_manifest.json").read_text())
        assert completed["complete"]
        for item in completed["artifacts_sha256"].values():
            assert sha_file(Path(item["path"])) == item["sha256"]
        progress(pp, "map_complete", resumed=True)
        return
    save_npz(out / "permutations.npz", train=np.stack(train_y), val=np.stack(val_y), test=test)
    progress(pp, "fit_start", n_train=len(train), n_arms=len(train_y))
    state = accumulate(
        X,
        Y,
        train,
        train_y,
        block=args.block,
        checkpoint=out / "crossproducts.pt",
        regime=regime,
        progress_path=pp,
    )
    payloads, selections = select_payloads(
        X, Y, val, val_y, state, parent.LAMBDAS_N1M, out=out, progress_path=pp
    )
    frozen_path = Path(prepared["frozen_payload"])
    assert sha_file(frozen_path) == MAP_SHA
    frozen = torch.load(frozen_path, map_location="cpu", weights_only=False)
    dev = torch.device("cpu")
    observed = np.asarray(Y[test], dtype=np.float64)
    frozen_pred = parent.apply_map(frozen, X[test], dev)
    refit_pred = parent.apply_map(payloads[0], X[test], dev)
    # fp32 archived parameters permit a small absolute discrepancy after a
    # different streaming block schedule; establish heldout prediction parity,
    # lambda parity, and the historical R2 independently (not a winning-score gate).
    np.testing.assert_allclose(refit_pred, frozen_pred, rtol=2e-5, atol=2e-5)
    assert payloads[0]["selected_lambda"] == frozen["selected_lambda"] == 0.001
    frozen_metrics = map_metrics(frozen_pred, observed)
    assert abs(frozen_metrics["r2"] - 0.7541708417500046) < 2e-6
    identity = np.asarray(X[test], dtype=np.float64) + (state["ymu"] - state["xmu"]).numpy()
    metrics = {
        "frozen": frozen_metrics,
        "true_refit": map_metrics(refit_pred, observed),
        "identity_plus_bias": map_metrics(identity, observed),
    }
    for i, seed in enumerate(SEEDS, 1):
        metrics[f"shuffle_seed{seed}"] = map_metrics(
            parent.apply_map(payloads[i], X[test], dev), observed
        )
        progress(pp, "fit_diagnostics", seed=seed)
    write_json(
        out / "diagnostics.json",
        {
            "metrics": metrics,
            "selection": selections,
            "parity_max_abs": float(np.abs(refit_pred - frozen_pred).max()),
            "n_train": len(train),
            "train_hash": index_hash(train),
            "val_hash": index_hash(val),
            "test_hash": index_hash(test),
            "source_revision": REVISION,
        },
    )
    for source, dest in (
        (frozen_path, out / "frozen.pt"),
        (inputs / "map_prompt_hashes.npz", out / "map_prompt_hashes.npz"),
        (inputs / "prepare_complete.json", out / "prepare_provenance.json"),
        (args.root / "inventory/hf_source_inventory.json", out / "hf_source_inventory.json"),
    ):
        with atomic_replace(dest) as tmp:
            shutil.copyfile(source, tmp)
        assert sha_file(source) == sha_file(dest)
    frozen_path = out / "frozen.pt"
    paths = {
        "frozen": frozen_path,
        "true_refit": out / "true_refit.pt",
        **{f"shuffle_seed{s}": out / f"shuffle_seed{s}.pt" for s in SEEDS},
    }
    paths["prompt_hashes"] = out / "map_prompt_hashes.npz"
    manifest = {
        "complete": True,
        "recipe": RECIPE,
        "layer": 19,
        "n_train": N_TRAIN,
        "n_val": 400,
        "n_test": 1000,
        "source_revision": REVISION,
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "model_revision": "a09a35458c702b33eeacc393d103063234e8bc28",
        "model_revision_evidence": (
            "#779 used the default model revision; HF main has remained at this SHA since "
            "2025-01-12, before the July2026 captures; #1739 explicitly pins this SHA. "
            "The #779 tensor payload itself did not persist a model revision."
        ),
        "layer_semantics": "post-block19 residual stream; model.layers[19] output = hidden_states[20]",
        "frozen_payload": str(frozen_path),
        "null_payloads": {str(s): str(out / f"shuffle_seed{s}.pt") for s in SEEDS},
        "map_prompt_hashes": str(out / "map_prompt_hashes.npz"),
        "hash_normalization": "sha256 UTF-8; normalized: ' '.join(text.lower().split())",
        "map_coordinate_system": "raw input -> train-coordinate standardization -> W -> raw answer mean",
        "target_pooling": "#779 mean over response plus closing assistant template tokens",
        "evaluation_pooling_caveat": "#1739 t1 excludes boundary/closing template tokens",
        "null_recipe": "independent permutations of training/validation answer rows; full parent lambda grid",
        "artifacts_sha256": {k: {"path": str(p), "sha256": sha_file(p)} for k, p in paths.items()},
    }
    manifest["payload_sha256"] = {str(p): sha_file(p) for p in paths.values() if p.suffix == ".pt"}
    manifest["map_prompt_hashes_sha256"] = sha_file(paths["prompt_hashes"])
    write_json(out / "map_manifest.json", manifest)
    progress(pp, "map_complete", n_train=N_TRAIN, seeds=list(SEEDS))


def main():
    """Expose independently monitored prepare and fit phases."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "fit"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--progress", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--block", type=int, default=8192)
    parser.add_argument("--frozen-map", type=Path, default=DEFAULT_MAP)
    parser.add_argument("--pass-b", type=Path, default=DEFAULT_PASSB)
    parser.add_argument("--manifest-dir", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()
    assert 1 <= args.workers <= 8 and args.block > 0
    (prepare if args.phase == "prepare" else fit)(args)


if __name__ == "__main__":
    main()
