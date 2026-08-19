"""Issue #2221 artifact loaders — reused-artifact staging + realized-key asserts.

Every loader FAILS LOUD on a shape/key mismatch (the plan-§10 realized-keys
verification, wired as startup asserts in the consuming scripts):

- r_B files: raw ``(28, 3584)`` float tensors per trait
  (``rb_v2/{trait}.pt`` @ pinned revision; v1 sensitivity twin).
- #1739 affine maps: npz with ``w`` ``(28, d, d)``, ``x_mu``/``x_sd``/``y_mu``
  stored ``(28, 1, d)`` and squeezed to ``(28, d)``, ``layers == 0..27``,
  ``meta`` JSON. Apply contract: ``pred = ((x - x_mu)/x_sd) @ w + y_mu``.
  Applying a map to a SHIFT means the DIFFERENCE OF MAPPED STATES
  ``M(v_f) - M(v_base)`` — never the affine map on a raw difference
  (:func:`apply_map_shift`).
- #778 ``finetune_activations/{tag}.pt``: ``{trait: (28, 3584)}`` dicts
  (LAST-PROMPT-TOKEN kind, mean over the paper 20-q eval surface).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from explore_persona_space.experiments.issue_1739.corpus_staging import MINHASH_BANDS

from . import constants as C

logger = logging.getLogger(__name__)


def self_near_dup_mask(sigs: np.ndarray, *, bands: int = MINHASH_BANDS) -> np.ndarray:
    """Boolean mask over rows: True = near-dup of an EARLIER KEPT row (pool-vs-itself).

    Self-dedup twin of the #1739 TWO-ARRAY ``near_dup_mask`` (train-vs-eval;
    ``corpus_staging.py``), over the same ``minhash_signatures`` arrays and the
    same LSH banding (default 16 bands x 4 rows over 64 perms ~= Jaccard>=0.5
    flagged). A row is flagged when ANY of its (band, band-signature) tuples was
    already registered by an earlier KEPT row; kept rows register all their band
    tuples, flagged rows register none (no chaining through dropped rows), so
    the FIRST occurrence of every near-dup group is always kept. Linear in
    n_rows like the parent (v6 crash fix: the two-array helper does not
    implement self-dedup — calling it with one raw-string list was the
    ``TypeError: missing ... 'eval_sigs'`` pod crash).
    """
    sigs = np.asarray(sigs)
    assert sigs.ndim == 2, sigs.shape
    n, n_perm = sigs.shape
    assert bands > 0 and n_perm % bands == 0, (n_perm, bands)
    rows_per_band = n_perm // bands
    seen: set[tuple[int, bytes]] = set()
    dup = np.zeros(n, dtype=bool)
    for i in range(n):
        keys = [
            (bi, sigs[i, bi * rows_per_band : (bi + 1) * rows_per_band].tobytes())
            for bi in range(bands)
        ]
        if any(k in seen for k in keys):
            dup[i] = True
        else:
            seen.update(keys)
    return dup


def stage_pinned_file(rel_path: str, revision: str | None, dest_dir: Path) -> Path:
    """Stage one data-repo file at a pinned revision via the canonical helper."""
    from explore_persona_space.orchestrate import hub

    dest_dir.mkdir(parents=True, exist_ok=True)
    target = dest_dir / Path(rel_path).name
    if target.is_file():
        return target
    hub.stage_hub_file(
        repo_id=C.HF_DATA_REPO,
        repo_type="dataset",
        path_in_repo=rel_path,
        target=target,
        revision=revision,
    )
    return target


def load_rb(trait: str, *, stage_dir: Path, version: str = "v2") -> np.ndarray:
    """Load one pinned r_B direction; assert the (28, 3584) realized shape."""
    import torch

    if version == "v2":
        rel, rev = f"{C.RB_V2_PREFIX}/{trait}.pt", C.RB_V2_REVISION
    elif version == "v1":
        rel, rev = f"{C.RB_V1_PREFIX}/{trait}.pt", C.RB_V1_REVISION
    else:
        raise ValueError(f"unknown r_B version {version!r}")
    path = stage_pinned_file(rel, rev, stage_dir / f"rb_{version}")
    rb = torch.load(path, map_location="cpu", weights_only=False)
    arr = np.asarray(rb.to(torch.float64).numpy() if hasattr(rb, "numpy") else rb, dtype=np.float64)
    assert arr.shape == (C.N_LAYERS, C.HIDDEN_DIM), (trait, version, arr.shape)
    return arr


def _map_row_vecs(arr: np.ndarray, name: str, variant: str) -> np.ndarray:
    """Squeeze a persisted (L, 1, d) map row-vector to (L, d); fail loud otherwise."""
    a = np.asarray(arr, dtype=np.float64)
    if a.ndim == 3 and a.shape[1] == 1:
        a = a[:, 0, :]
    if a.ndim != 2 or a.shape[0] != C.N_LAYERS:
        raise RuntimeError(
            f"map {variant}: {name} has unexpected layout {np.shape(arr)} — expected "
            f"({C.N_LAYERS}, 1, d) or ({C.N_LAYERS}, d)"
        )
    return a


def load_affine_map(variant: str, *, stage_dir: Path, revision: str | None = None) -> dict:
    """Load a #1739 ``{variant}__ufull.npz`` map with realized-key asserts.

    Returns ``{"w": (28, d, d), "x_mu"/"x_sd"/"y_mu": (28, d), "meta": dict,
    "path": str}``. The declared key set (plan §10) is asserted verbatim.
    """
    assert variant in C.MAP_VARIANTS, variant
    rel = f"{C.MAPS_PREFIX}{variant}__ufull.npz"
    path = stage_pinned_file(rel, revision, stage_dir / "maps")
    with np.load(path, allow_pickle=False) as z:
        missing = [k for k in C.MAP_KEYS if k not in z.files]
        if missing:
            raise RuntimeError(f"map {variant}: missing declared keys {missing} in {path}")
        meta = json.loads(str(z["meta"]))
        if list(z["layers"]) != list(range(C.N_LAYERS)):
            raise RuntimeError(f"map {variant}: layers != 0..{C.N_LAYERS - 1}")
        out = {
            "w": np.asarray(z["w"], dtype=np.float64),
            "x_mu": _map_row_vecs(z["x_mu"], "x_mu", variant),
            "x_sd": _map_row_vecs(z["x_sd"], "x_sd", variant),
            "y_mu": _map_row_vecs(z["y_mu"], "y_mu", variant),
            "meta": meta,
            "path": str(path),
        }
    w = out["w"]
    assert w.ndim == 3 and w.shape[0] == C.N_LAYERS and w.shape[1] == w.shape[2], w.shape
    logger.info("[map %s] apply=%r (whitened-space fold)", variant, meta.get("apply"))
    return out


def apply_map(mp: dict, x: np.ndarray, layer: int) -> np.ndarray:
    """Apply the affine map at one layer: ``((x - x_mu)/x_sd) @ w + y_mu``.

    ``x`` is a raw STATE ``(d,)`` or ``(n, d)`` — never a shift (see
    :func:`apply_map_shift`).
    """
    x = np.asarray(x, dtype=np.float64)
    return ((x - mp["x_mu"][layer]) / mp["x_sd"][layer]) @ mp["w"][layer] + mp["y_mu"][layer]


def apply_map_shift(mp: dict, v_after: np.ndarray, v_before: np.ndarray, layer: int) -> np.ndarray:
    """Mapped SHIFT = difference of mapped states ``M(v_after) - M(v_before)``.

    The affine apply contract forbids running the map on a raw difference
    (the bias/centering terms would be applied to a displacement); the
    difference-of-mapped-states form is exact and cancels ``y_mu``.
    """
    return apply_map(mp, v_after, layer) - apply_map(mp, v_before, layer)


def load_ft_activation(model_tag: str, *, stage_dir: Path) -> dict[str, np.ndarray]:
    """Load one #778 cached capture; assert the {trait: (28, 3584)} dict kind."""
    import torch

    rel = f"{C.FT_ACT_PREFIX}/{model_tag}.pt"
    path = stage_pinned_file(rel, C.FT_ACT_REVISION, stage_dir / "finetune_activations")
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(obj, dict) or not obj:
        raise RuntimeError(f"finetune_activations/{model_tag}.pt is not a trait dict: {type(obj)}")
    out: dict[str, np.ndarray] = {}
    for trait, tens in obj.items():
        arr = np.asarray(tens.to(torch.float64).numpy(), dtype=np.float64)
        assert arr.shape == (C.N_LAYERS, C.HIDDEN_DIM), (model_tag, trait, arr.shape)
        out[str(trait)] = arr
    return out


def read_jsonl(path: Path) -> list[dict]:
    """Text-mode JSONL read (never ``splitlines()`` — U+2028 shredding, #950)."""
    rows: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip("\n").strip("\r")
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    """Atomic JSONL write (tmp in the same dir + ``os.replace``)."""
    import os
    import tempfile

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def atomic_write_text(path: Path, text: str) -> None:
    """Atomic text write (tmp in the same dir + ``os.replace``; the write_jsonl shape)."""
    import os
    import tempfile

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def atomic_torch_save(path: Path, obj) -> None:
    """Atomic ``torch.save`` (tmp in the same dir + ``os.replace``)."""
    import os
    import tempfile

    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    os.close(fd)
    try:
        torch.save(obj, tmp)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def sha256_file(path: Path) -> str:
    """Streaming sha256 of a file's bytes (fingerprint input-chaining, N4/N5).

    Downstream phases fold their INPUT artifact's sha into their own resume
    fingerprint so a regenerated upstream artifact invalidates the cached
    downstream output instead of silently reusing rows computed on stale
    inputs (judge <- gen rows; tf_margin <- tf pools; capture <- surfaces).
    """
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    """sha256 of a text payload (CONTENT hashes, e.g. the frozen surface roster)."""
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _fingerprint_sidecar(path: Path) -> Path:
    return path.with_name(path.name + ".fp.json")


def write_fingerprint(path: Path, fingerprint: dict) -> None:
    """Persist the output's regime-fingerprint sidecar (``<name>.fp.json``).

    Written AFTER the payload (payload -> sidecar order), so a crash between
    the two leaves the unit NOT-resumable and it recomputes.
    """
    atomic_write_text(_fingerprint_sidecar(path), json.dumps(fingerprint, sort_keys=True))


def resume_ok(path: Path, fingerprint: dict) -> bool:
    """Resume predicate: payload present AND sidecar matches EVERY regime key.

    Keys every output-affecting flag (``--judge-draws`` / ``--n-rollouts`` /
    ``--max-new-tokens`` / slice caps) so a re-run under a different regime
    recomputes instead of silently reusing wrong cached rows (#722 r3 class).
    Fingerprint values must be JSON-round-trippable scalars/lists (no tuples).
    """
    side = _fingerprint_sidecar(path)
    if not (path.is_file() and side.is_file()):
        return False
    try:
        return json.loads(side.read_text()) == fingerprint
    except (json.JSONDecodeError, OSError, UnicodeDecodeError):
        return False
