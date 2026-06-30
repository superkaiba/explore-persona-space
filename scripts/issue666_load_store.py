#!/usr/bin/env python
# ruff: noqa: RUF003
# Intentional scientific Unicode (Σ, ρ, λ, ŵ, ×, −, ⁻¹, ᵀ) in docstrings/comments.
"""issue #666 Phase 4 — load the Phase-2 (#664) trained activation store, per cell.

Streams one store cell at a time (download → use → the caller deletes) from the
#664 HF store ``superkaiba1/explore-persona-space-data/theory_assumptions/
Qwen2.5-7B-Instruct/issue664/<cell>/{tensors.pt,meta.json}`` (48 content cells +
the 2 designed-null cells `ic_edu_default` / `tf_rev_default` — same loader, same
schema, plan §4a/§4d). The on-disk cell dirs are the seed-qualified
``<slug>_seed<seed>`` form; the short null-cell prefixes (`ic_edu_default`,
`tf_rev_default`) resolve to their `_contra_d1_seed42` dirs (§5).

``load_cell(cell_dir)`` validates the tensor SCHEMA (required keys + tensor RANK /
axis identities, NOT the literal 28/3584/48 counts — so a tiny fabricated smoke
cell is accepted) and returns ``{<tensor keys>..., "meta": <meta dict>}``. The
``--cells`` CLI smoke loads N cells from HF and prints a digest.

CPU-only; the GPU corpus-extraction step is a SEPARATE script
(``issue666_corpus_extract.py``).
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import json
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

REPO = Path(__file__).resolve().parent.parent
DATA_REPO = "superkaiba1/explore-persona-space-data"
STORE_PREFIX = "theory_assumptions/Qwen2.5-7B-Instruct/issue664"

# #658 difference-in-means r_B store (Hub-pinned; plan §4b / assumption 5).
RB_PT_PATH = "issue658_theory_assumptions/store/r_b.pt"
RB_PT_SHA256 = "61b1146aa39906354e21293db2a744fdfb851ce2ef2e09f1077f3b873af3608b"
RB_EXPECTED_COLUMNS = ("broad_em", "harmful_compliance", "sycophancy", "refusal")

# Store schema (issue664_common): 28 layers, d=3584, 50 contexts. The probe axis
# of the stored probe-split tensors is 50 (NOT the 48-Betley-probe count — the
# stored *_probe tensors carry the full battery-probe split). load_cell validates
# RANK + axis identity, not these literals (test-mode fabricated cells use 4/32).
EXPECTED_LAYERS = 28
EXPECTED_HIDDEN = 3584
EXPECTED_CONTEXTS = 50

# Tensor keys load_cell requires present.
_REQUIRED_TENSORS = ("v_plus", "v0", "v_plus_probe", "v0_probe", "c_C_base", "t_CB", "r_plus")

# Short designed-null prefixes -> their seed-qualified store-dir names (§5).
DESIGNED_NULL_DIR = {
    "ic_edu_default": "ic_edu_default_contra_d1_seed42",
    "tf_rev_default": "tf_rev_default_contra_d1_seed42",
}


def _validate_schema(obj: dict, cell_name: str) -> None:
    """Validate the #664 store tensor schema by RANK/axis identity (plan §4a).

    Checks the required tensors are present with the documented axis structure:
    v_plus/v0/c_C_base are (n_ctx, n_layer, d); v_plus_probe/v0_probe are
    (n_ctx, n_probe, n_layer, d); t_CB/r_plus are (n_layer, d). Validates RANK +
    cross-tensor axis consistency, NOT the literal 28/3584 counts (so a small
    fabricated smoke cell with nl=4/d=32 passes). Raises ValueError on violation.
    """
    missing = [k for k in _REQUIRED_TENSORS if k not in obj]
    if missing:
        raise ValueError(f"{cell_name}: store cell missing tensors {missing}")
    vp = obj["v_plus"]
    if vp.ndim != 3:
        raise ValueError(f"{cell_name}: v_plus must be (n_ctx, n_layer, d), got {tuple(vp.shape)}")
    n_ctx, n_layer, d = vp.shape
    for k in ("v0", "c_C_base", "c_C_trained"):
        if k in obj and tuple(obj[k].shape) != (n_ctx, n_layer, d):
            raise ValueError(
                f"{cell_name}: {k} shape {tuple(obj[k].shape)} != v_plus {(n_ctx, n_layer, d)}"
            )
    for k in ("v_plus_probe", "v0_probe"):
        t = obj[k]
        if t.ndim != 4:
            raise ValueError(
                f"{cell_name}: {k} must be (n_ctx, n_probe, n_layer, d), got {tuple(t.shape)}"
            )
        if t.shape[0] != n_ctx or t.shape[2] != n_layer or t.shape[3] != d:
            raise ValueError(
                f"{cell_name}: {k} axes {tuple(t.shape)} inconsistent with v_plus "
                f"{(n_ctx, n_layer, d)}"
            )
    for k in ("t_CB", "r_plus"):
        t = obj[k]
        if tuple(t.shape) != (n_layer, d):
            raise ValueError(
                f"{cell_name}: {k} must be (n_layer, d)={(n_layer, d)}, got {tuple(t.shape)}"
            )


def load_cell(cell_dir: Path | str) -> dict:
    """Load + schema-validate a #664 store cell from a local directory.

    ``cell_dir`` holds ``tensors.pt`` + ``meta.json``. Returns the tensor dict
    augmented with a ``"meta"`` key. ``meta.target_context_roles`` may be a dict
    (cid->role, the real store form) or a list (the test-fabricated form) — both
    are accepted and normalized to a dict on read by the consumers.
    """
    cell_dir = Path(cell_dir)
    tp = cell_dir / "tensors.pt"
    mp = cell_dir / "meta.json"
    if not tp.exists():
        raise FileNotFoundError(f"missing tensors.pt under {cell_dir}")
    if not mp.exists():
        raise FileNotFoundError(f"missing meta.json under {cell_dir}")
    obj = torch.load(tp, map_location="cpu")
    meta = json.loads(mp.read_text())
    _validate_schema(obj, cell_dir.name)
    out = dict(obj)
    out["meta"] = meta
    return out


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_store_dir(cell: str) -> str:
    """Map a cell name (short null prefix OR full slug) to its HF store dir name."""
    return DESIGNED_NULL_DIR.get(cell, cell)


def download_cell(cell: str, *, dest: Path | None = None) -> Path:
    """Download a #664 store cell from HF into a local dir; return the dir path.

    ``cell`` may be a short designed-null prefix (`ic_edu_default`) or a full
    seed-qualified slug (`bm_default_contra_d1_seed42`).
    """
    from huggingface_hub import hf_hub_download

    store_dir = resolve_store_dir(cell)
    tp = hf_hub_download(DATA_REPO, f"{STORE_PREFIX}/{store_dir}/tensors.pt", repo_type="dataset")
    mp = hf_hub_download(DATA_REPO, f"{STORE_PREFIX}/{store_dir}/meta.json", repo_type="dataset")
    # hf_hub_download returns paths in the shared cache; the cell "dir" is the
    # parent of tensors.pt. meta.json lands in the same cache dir alongside it.
    local_dir = Path(tp).parent
    if Path(mp).parent != local_dir:
        # symlink meta.json next to tensors.pt so load_cell finds both.
        import shutil

        shutil.copy(mp, local_dir / "meta.json")
    return local_dir


def verify_rb_pt() -> dict:
    """Hub-resolve + sha256-pin + key-assert the #658 r_b.pt before any reuse (§4b).

    Returns the loaded ``r_b`` dict. Raises on a Hub-resolution failure, sha256
    mismatch, or a missing required column (artifact-identity pin,
    .claude/rules/artifact-reuse.md (e)/(f)/(g)).
    """
    from huggingface_hub import hf_hub_download, list_repo_files

    files = list_repo_files(DATA_REPO, repo_type="dataset")
    if RB_PT_PATH not in files:
        raise RuntimeError(f"#658 r_b.pt not resolvable on HF at {RB_PT_PATH}")
    local = hf_hub_download(DATA_REPO, RB_PT_PATH, repo_type="dataset")
    got = _sha256_file(local)
    if got != RB_PT_SHA256:
        raise RuntimeError(f"r_b.pt sha256 mismatch: got {got}, expected {RB_PT_SHA256}")
    rb = torch.load(local, map_location="cpu")
    cols = rb.get("rb_columns") or list(rb.get("r_b", {}).keys())
    if not ({"broad_em", "harmful_compliance"} <= set(cols)):
        raise RuntimeError(f"r_b.pt missing required diffmeans columns; got {sorted(cols)}")
    if len(rb.get("r_b", {})) != 4:
        raise RuntimeError(f"r_b.pt expected 4 behaviors, got {len(rb.get('r_b', {}))}")
    return rb


def load_rb_columns(layer: int = 14, *, rb: dict | None = None) -> dict:
    """Return the #658 diffmeans r_B columns at ``layer`` as float64 numpy vectors (§4b).

    Sibling to ``verify_rb_pt`` — the latter Hub-resolves + sha256-pins + key-asserts
    the artifact; this returns the actual diffmeans read-out directions used as the
    mixed-source ``r_{B'}`` for the bad-medical (``harmful_compliance``) and
    insecure-code/EM (``broad_em``) behaviors. Pass an already-verified ``rb`` dict
    to skip the (re-)download; otherwise ``verify_rb_pt()`` runs first so no read
    happens against an unpinned artifact.

    The #658 ``r_b`` store maps each behavior -> a per-column dict whose ``diffmeans``
    key is the (n_layer, d) difference-in-means read-out direction (alongside
    ``meanDB`` / ``n_db`` / ``n_dbbar`` book-keeping). A LEGACY plain-tensor column
    ((n_layer, d) or (d,)) is also accepted for forward-compat. The (n_layer, d)
    direction is indexed at ``layer``; a (d,) direction is used directly. Returns
    ``{column_name: np.ndarray(d,)}`` for EVERY column present (so a caller can route
    per behavior).
    """
    import numpy as np

    if rb is None:
        rb = verify_rb_pt()
    rb_map = rb.get("r_b", {})
    out: dict = {}
    for name, val in rb_map.items():
        # Real #658 shape: val is {'diffmeans': (n_layer, d), 'meanDB': ..., ...}.
        # Legacy/test shape: val is a bare (n_layer, d) | (d,) tensor.
        if isinstance(val, dict):
            if "diffmeans" not in val:
                raise ValueError(
                    f"r_b column {name!r} dict missing 'diffmeans' key; got {sorted(val)}"
                )
            t = val["diffmeans"]
        else:
            t = val
        arr = t.detach().cpu().numpy() if hasattr(t, "detach") else np.asarray(t)
        arr = arr.astype(np.float64)
        if arr.ndim == 2:  # (n_layer, d) -> the layer's row
            li = min(layer, arr.shape[0] - 1)
            vec = arr[li, :]
        elif arr.ndim == 1:  # already (d,)
            vec = arr
        else:
            raise ValueError(
                f"r_b column {name!r} has unexpected rank {arr.ndim} (shape {arr.shape})"
            )
        out[name] = np.ascontiguousarray(vec.reshape(-1))
    return out


def _cell_digest(loaded: dict, cell: str) -> dict:
    vp = loaded["v_plus"]
    meta = loaded["meta"]
    return {
        "cell": cell,
        "v_plus_shape": tuple(vp.shape),
        "v_plus_probe_shape": tuple(loaded["v_plus_probe"].shape),
        "c_C_base_shape": tuple(loaded["c_C_base"].shape),
        "behavior": meta.get("behavior"),
        "source": meta.get("source"),
        "base_model": meta.get("base_model"),
        "marker_id": meta.get("marker_id"),
    }


def _smoke_cells(n: int) -> list[str]:
    """A tiny slice for the smoke: one content cell + one designed-null cell."""
    return ["bm_default_contra_d1_seed42", "ic_edu_default"][:n]


def main() -> int:
    ap = argparse.ArgumentParser(description="Load #664 store cells (issue 666 Phase 4).")
    ap.add_argument("--cells", type=int, default=2, help="number of smoke cells to load")
    ap.add_argument("--cell-names", nargs="*", default=None, help="explicit cell names override")
    ap.add_argument("--slice", action="store_true", help="tiny smoke slice")
    ap.add_argument("--verify-rb", action="store_true", help="also verify #658 r_b.pt pin")
    args = ap.parse_args()

    from explore_persona_space.orchestrate.env import load_dotenv

    load_dotenv()

    cells = args.cell_names if args.cell_names else _smoke_cells(args.cells)
    if args.verify_rb:
        rb = verify_rb_pt()
        print(f"[rb] r_b.pt verified: columns={sorted(rb.get('r_b', {}).keys())}")

    for cell in cells:
        local_dir = download_cell(cell)
        loaded = load_cell(local_dir)
        digest = _cell_digest(loaded, cell)
        print(f"[load] {json.dumps(digest)}")
        del loaded
        gc.collect()
        # bound footprint: delete the downloaded tensors.pt (re-downloadable)
        with contextlib.suppress(OSError):
            os.remove(local_dir / "tensors.pt")
    print(f"[phase=load_store] loaded {len(cells)} cells OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
