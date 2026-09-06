"""Validated natural-pair input seam for the existing #1739 P-B scorer.

No fit implementation lives here: the opted-in caller retains the reviewed
ADD-map and P-B readout machinery. A rung uses a nested generic subset plus
ALL fixed eliciting training pairs; whitening is fitted on that union only.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

MODEL = "Qwen/Qwen2.5-7B-Instruct"
MODEL_REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"
NATURAL_ROSTER = ("arm4_ridge_ctx", "arm7_map_ridge_pred", "arm12_oracle_reg")
WHITENING_NOTE = (
    "refit per generic-U rung on that rung's natural generic + full fixed eliciting union; "
    "shared by mapped, direct-context and oracle arms; direct/oracle baselines may vary with U"
)


def _sha(path: Path) -> str:
    """Hash an input without materializing its bytes in memory."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_natural_manifest(root: Path, *, hidden_dim: int = 3584) -> tuple[dict, list[dict]]:
    """Refuse partial, crossed, incompatible, duplicated, or unpinned input pools."""
    root = Path(root)
    path = root / "manifest.json"
    meta = json.loads(path.read_text())
    required = {
        "schema_version": 1,
        "status": "complete",
        "pool_kind": "natural_context_answer",
        "model": MODEL,
        "model_revision": MODEL_REVISION,
        "no_recombination": True,
        "hidden_dim": hidden_dim,
        "dtype": "float16",
    }
    for name, expected in required.items():
        if type(meta.get(name)) is not type(expected) or meta[name] != expected:
            raise ValueError(f"{path}: {name} must equal {expected!r}, got {meta.get(name)!r}")
    n = meta.get("n_rows")
    if type(n) is not int or n < 1:
        raise ValueError(f"{path}: n_rows must be a positive integer")
    layers = meta.get("layers")
    if (
        not isinstance(layers, list)
        or not layers
        or any(type(x) is not int or x not in range(28) for x in layers)
        or len(set(layers)) != len(layers)
    ):
        raise ValueError(f"{path}: layers must be distinct global layer IDs in 0..27")
    matrices = meta.get("matrices_sha256")
    expected_names = {
        f"{kind}_L{layer:02d}.npy" for kind in ("context_end", "t1") for layer in layers
    }
    if (
        not isinstance(matrices, dict)
        or set(matrices) != expected_names
        or any(not re.fullmatch(r"[0-9a-f]{64}", str(v)) for v in matrices.values())
    ):
        raise ValueError(f"{path}: matrices_sha256 must cover every declared context/t1 matrix")
    generation, capture = meta.get("generation", {}), meta.get("capture", {})
    if generation.get("temperature") != 0 or generation.get("max_new_tokens", 0) < 1:
        raise ValueError(f"{path}: generation must record greedy temperature=0 and token cap")
    cap_fraction = generation.get("cap_hit_fraction")
    if not isinstance(cap_fraction, (int, float)) or not 0 <= cap_fraction <= 1:
        raise ValueError(f"{path}: generation.cap_hit_fraction must lie in [0,1]")
    if capture.get("context_kind") != "context_end" or capture.get("answer_kind") != "t1":
        raise ValueError(f"{path}: capture must be context_end -> t1")
    if not isinstance(meta.get("source"), dict) or not meta["source"]:
        raise ValueError(f"{path}: source provenance is required")
    index = root / "row_index.jsonl"
    if meta.get("row_index_sha256") != _sha(index):
        raise ValueError(f"{path}: row_index_sha256 mismatch")
    with index.open() as stream:
        rows = [json.loads(line) for line in stream]
    if len(rows) != n:
        raise ValueError(f"{index}: n_rows={n} but index has {len(rows)} rows")
    seen = {name: set() for name in ("context_id", "source_pair", "prompt_sha256")}
    for i, row in enumerate(rows):
        if row.get("no_recombination") is not True:
            raise ValueError(f"{index} row {i}: no_recombination must be true")
        for name in ("context_id", "source_dataset", "source_id"):
            if not isinstance(row.get(name), str) or not row[name]:
                raise ValueError(f"{index} row {i}: nonempty {name} required")
        for name in ("prompt_sha256", "answer_sha256"):
            if not re.fullmatch(r"[0-9a-f]{64}", str(row.get(name, ""))):
                raise ValueError(f"{index} row {i}: invalid {name}")
        values = {
            "context_id": row["context_id"],
            "source_pair": (row["source_dataset"], row["source_id"]),
            "prompt_sha256": row["prompt_sha256"],
        }
        for name, value in values.items():
            if value in seen[name]:
                raise ValueError(f"{index} row {i}: duplicate {name}")
            seen[name].add(value)
    return {**meta, "manifest_sha256": _sha(path)}, rows


def _matrix_stat(path: Path) -> dict:
    """Stable local-file cache identity; ctime additionally catches restored mtimes."""
    stat = path.stat()
    return {
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "inode": stat.st_ino,
        "ctime_ns": stat.st_ctime_ns,
        "device": stat.st_dev,
    }


def verify_natural_matrices(root: Path, meta: dict, layers: list[int]) -> None:
    """Verify selected full matrices, caching only unchanged local-file identities.

    The receipt is a performance cache, not a portable provenance proof: a fresh
    staging operation changes inode/mtime and rehashes. The pinned manifest hash
    and expected SHA are part of every cache hit. Index SHA verification remains
    unconditional in ``read_natural_manifest``. Concurrent atomic receipt writes
    can lose cached entries, causing extra hashing but never skipped validation.
    """
    from explore_persona_space.atomic_io import atomic_replace

    root = Path(root)
    if not layers or len(set(layers)) != len(layers) or set(layers) - set(meta["layers"]):
        raise ValueError(f"requested global layers {layers} absent/invalid for {meta['layers']}")
    receipt_path = root / ".natural_matrix_validation.json"
    receipt = json.loads(receipt_path.read_text()) if receipt_path.exists() else {}
    if (
        receipt.get("schema_version") != 1
        or receipt.get("manifest_sha256") != meta["manifest_sha256"]
    ):
        receipt = {"schema_version": 1, "manifest_sha256": meta["manifest_sha256"], "files": {}}
    changed = False
    for kind in ("context_end", "t1"):
        for layer in layers:
            name = f"{kind}_L{layer:02d}.npy"
            path = root / name
            expected = meta["matrices_sha256"][name]
            before = _matrix_stat(path)
            record = {"sha256": expected, **before}
            if receipt["files"].get(name) == record:
                continue
            if _sha(path) != expected:
                raise ValueError(f"{path}: matrix SHA256 mismatch against natural manifest")
            if _matrix_stat(path) != before:
                raise ValueError(f"{path}: matrix changed during hash validation")
            receipt["files"][name] = record
            changed = True
    if changed:
        with atomic_replace(receipt_path) as tmp:
            tmp.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")


def nested_generic_rows(n: int, u: int, seed: int):
    """Same nested generic sample across behaviors, keyed only by seed and U."""
    import numpy as np

    if not 1 <= u <= n or seed < 0:
        raise ValueError(f"generic budget/seed out of range: U={u}, n={n}, seed={seed}")
    perm = np.random.default_rng([1739, 20260906, int(seed)]).permutation(n)
    return np.sort(perm[:u]).astype(np.int64)


def load_natural_pool(root: Path, layers: list[int], *, u: int, seed: int, hidden_dim: int):
    """Memory-map only requested natural summaries, never invoke legacy HF staging."""
    import numpy as np

    root = Path(root)
    meta, rows = read_natural_manifest(root, hidden_dim=hidden_dim)
    if not layers or len(set(layers)) != len(layers) or set(layers) - set(meta["layers"]):
        raise ValueError(f"requested global layers {layers} absent/invalid for {meta['layers']}")
    verify_natural_matrices(root, meta, layers)
    chosen = nested_generic_rows(meta["n_rows"], u, seed)
    arrays = {}
    for kind in ("context_end", "t1"):
        for layer in layers:
            path = root / f"{kind}_L{layer:02d}.npy"
            arr = np.load(path, mmap_mode="r", allow_pickle=False)
            if arr.shape != (meta["n_rows"], hidden_dim) or arr.dtype != np.float16:
                raise ValueError(
                    f"{path}: expected fp16 {(meta['n_rows'], hidden_dim)}, "
                    f"got {arr.dtype} {arr.shape}"
                )
            # Scan in bounded chunks; do not allocate a full-pool bool matrix.
            for lo in range(0, len(chosen), 1024):
                if not np.isfinite(arr[chosen[lo : lo + 1024]]).all():
                    raise ValueError(f"{path}: non-finite selected activation rows")
            arrays[(kind, layer)] = arr
    selected_ids = [rows[int(i)]["context_id"] for i in chosen]
    meta.update(
        generic_u=int(u),
        selected_context_ids_sha256=hashlib.sha256("\n".join(selected_ids).encode()).hexdigest(),
        generic_sampling="nested permutation rng [1739, 20260906, seed], sorted prefix indices",
        selected_global_layers=list(layers),
        whitening=WHITENING_NOTE,
    )
    return arrays, chosen, meta


def frozen_global_layers(summary: Path, *, variant: str, regime: str, roster: tuple) -> dict:
    """Resolve committed full-grid positional indices, validating their global meaning."""
    from scripts.issue1739_wcrung_arms import modal_frozen_layers

    payload = json.loads(Path(summary).read_text())
    if payload.get("meta", {}).get("layers") != list(range(28)):
        raise ValueError(f"{summary}: committed metadata layer grid is not identity 0..27")
    rows = [
        r
        for r in payload["arm_rows"]
        if r.get("arm") in roster
        and r.get("variant") == variant
        and r.get("regime") == regime
        and str(r.get("u_rung_label")) == "full"
        and r.get("f_u") is None
    ]
    # The producing 28-layer grid is an identity grid. Refuse another layout
    # rather than assuming a returned position is a global layer number.
    if not rows or any(len(r.get("rho_per_layer", [])) != 28 for r in rows):
        raise ValueError(f"{summary}: expected committed full 28-layer rho grid")
    for row in rows:
        if "layers" in row and row["layers"] != list(range(28)):
            raise ValueError(f"{summary}: committed layer grid is not identity 0..27")
    frozen = modal_frozen_layers(summary, variant=variant, regime=regime, u_rung_label="full")
    missing = set(roster) - set(frozen)
    if missing:
        raise ValueError(f"{summary}: missing committed frozen arms {sorted(missing)}")
    return {arm: int(frozen[arm]) for arm in roster}


def remap_frozen(frozen: dict, layers: list[int]) -> dict:
    """Convert global frozen layer IDs to reduced-array positions, never clamp."""
    if len(layers) != len(set(layers)) or set(frozen.values()) - set(layers):
        raise ValueError(f"frozen global layers {frozen} not indexable in {layers}")
    return {arm: layers.index(int(layer)) for arm, layer in frozen.items()}


def natural_regime_key(args, behavior: str, layers: list[int], roster: tuple) -> dict:
    """Machine-stable complete scoring identity for natural-pool resume checks."""
    meta, _rows = read_natural_manifest(args.natural_u_store)
    # Resume occurs before load_natural_pool; validate here too so an old result
    # cannot satisfy the invocation against a corrupted but same-shape store.
    verify_natural_matrices(args.natural_u_store, meta, layers)
    # Keep every parser flag except locations of generated output and bookkeeping.
    # Paths to INPUTS intentionally remain in the key; no float-array byte hashes.
    ignored = {
        "out_root",
        "out_subdir",
        "import_check",
        "allow_overwrite_committed",
        "seeds",
        "behaviors",
    }
    options = {
        k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items() if k not in ignored
    }
    return {
        "schema_version": 1,
        "manifest_sha256": meta["manifest_sha256"],
        "behavior": behavior,
        "generic_u": int(args.generic_u),
        "seed": int(args.seed),
        "global_layers": list(layers),
        "roster": list(roster),
        "options": options,
    }
