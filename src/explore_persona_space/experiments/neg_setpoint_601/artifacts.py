# ruff: noqa: RUF002  # em-dash, Greek alpha + ※ in scaling docstrings intentional
"""Task #601 — parent-artifact fetch helpers (HF data/model repo → local paths).

Per-file ``hf_hub_download`` (NOT ``snapshot_download`` — its
``repo_info.siblings`` listing truncates on large repos and silently returns
0 files for tail prefixes; feedback_snapshot_download_siblings_truncation).
Idempotent: existing local files are kept.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
from pathlib import Path

from explore_persona_space.experiments.neg_setpoint_601 import (
    HF_ADAPTER_PREFIX_472,
    HF_DATA_REPO,
    HF_MODEL_REPO,
    PARENT_DATA_FILES,
    PARITY_READ_RATIONALE,
    PARITY_READ_USE_RSLORA,
)

log = logging.getLogger("issue_601.artifacts")

_ADAPTER_FILES = ("adapter_config.json", "adapter_model.safetensors")


def _sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    """Streaming full-file sha256 (≈1.5 s on a 323 MB adapter)."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def stage_parity_read_adapter(
    src_dir: Path, staged_root: Path, *, expect_slug: str
) -> tuple[Path, dict]:
    """Stage a trained adapter for a parent-parity READ (round 5 gate fix).

    Round-5 Phase-0a HALT root cause: applying the rsLoRA-trained #472
    adapters with their shipped ``use_rslora: true`` config (effective
    scaling α/√r ≈ 11.31) turns every one of them into an unconditional
    ` ※`-repeater, pinning the re-read ΔG at an adapter-INDEPENDENT collapse
    ceiling (six different adapters re-read 10.350 ± 0.002). The parent's
    committed set-points correspond to the classic α/r = 2.0 application
    (``PARITY_READ_*`` in the package init has the full evidence chain), so
    every #601 read stages the adapter with ``use_rslora`` forced to
    :data:`PARITY_READ_USE_RSLORA` and applies the STAGED copy.

    Mechanics (cheap + provenance-bearing, never mutates the source dir):
      - fail-loud mapping assert: ``expect_slug`` must appear in
        ``str(src_dir)`` (the round-5 brief's worker→adapter check);
      - ``adapter_config.json`` is copied with ``use_rslora`` patched;
      - ``adapter_model.safetensors`` is symlinked (no 323 MB copy);
      - returns ``(staged_dir, provenance)`` where provenance records the
        source path, full-file sha256 of the weights, original vs applied
        ``use_rslora``, ``lora_alpha``/``r``, and the effective scaling.

    Idempotent: re-staging the same source refreshes the config + symlink.
    """
    src_dir = Path(src_dir)
    if expect_slug not in str(src_dir):
        raise ValueError(
            f"adapter mapping assert FAILED: expected slug {expect_slug!r} in adapter path "
            f"{src_dir} — worker→adapter assignment scrambled (round-5 gate incident class)."
        )
    cfg_path = src_dir / "adapter_config.json"
    weights_path = src_dir / "adapter_model.safetensors"
    for p in (cfg_path, weights_path):
        if not p.exists():
            raise FileNotFoundError(f"stage_parity_read_adapter: {p} missing")

    cfg = json.loads(cfg_path.read_text())
    original_rslora = bool(cfg.get("use_rslora", False))
    lora_alpha = cfg.get("lora_alpha")
    lora_r = cfg.get("r")
    cfg["use_rslora"] = PARITY_READ_USE_RSLORA

    staged_dir = Path(staged_root) / src_dir.name
    staged_dir.mkdir(parents=True, exist_ok=True)
    (staged_dir / "adapter_config.json").write_text(json.dumps(cfg, indent=2))
    staged_weights = staged_dir / "adapter_model.safetensors"
    if staged_weights.is_symlink() or staged_weights.exists():
        staged_weights.unlink()
    staged_weights.symlink_to(weights_path.resolve())

    effective = (
        f"lora_alpha/sqrt(r) = {lora_alpha}/sqrt({lora_r})"
        if PARITY_READ_USE_RSLORA
        else f"lora_alpha/r = {lora_alpha}/{lora_r}"
    )
    provenance = {
        "expect_slug": expect_slug,
        "source_adapter_path": str(src_dir),
        "staged_adapter_path": str(staged_dir),
        "adapter_sha256": _sha256_file(weights_path),
        "use_rslora_original": original_rslora,
        "use_rslora_applied": PARITY_READ_USE_RSLORA,
        "lora_alpha": lora_alpha,
        "r": lora_r,
        "effective_scaling_applied": effective,
        "rationale": PARITY_READ_RATIONALE,
    }
    log.info(
        "[parity-stage] %s: use_rslora %s -> %s (scaling %s) staged at %s",
        expect_slug,
        original_rslora,
        PARITY_READ_USE_RSLORA,
        effective,
        staged_dir,
    )
    return staged_dir, provenance


def fetch_parent_data(repo_root: Path) -> dict[str, str]:
    """Download the pinned #472 inputs (bank / centroids / R) from the data repo.

    ``repo_root`` is the repository root the relative ``PARENT_DATA_FILES``
    destinations resolve against (the pod checkout root).
    """
    from huggingface_hub import hf_hub_download

    fetched: dict[str, str] = {}
    for repo_path, local_rel in PARENT_DATA_FILES:
        local = repo_root / local_rel
        if local.exists():
            fetched[repo_path] = str(local)
            continue
        local.parent.mkdir(parents=True, exist_ok=True)
        got = hf_hub_download(
            repo_id=HF_DATA_REPO,
            repo_type="dataset",
            filename=repo_path,
            token=os.environ.get("HF_TOKEN"),
        )
        shutil.copyfile(got, local)
        fetched[repo_path] = str(local)
        log.info("[fetch] %s -> %s", repo_path, local)
    return fetched


def fetch_parent_adapter(cell: str, seed: int, dest_root: Path) -> Path:
    """Download one #472 final adapter (config + safetensors) from the model repo.

    Returns the local adapter directory. Fail-loud on a missing file (the 20
    adapters were Hub-verified at plan time; absence here means repo drift).
    """
    from huggingface_hub import hf_hub_download

    dest = dest_root / f"{cell}_seed{seed}"
    dest.mkdir(parents=True, exist_ok=True)
    for fname in _ADAPTER_FILES:
        local = dest / fname
        if local.exists():
            continue
        got = hf_hub_download(
            repo_id=HF_MODEL_REPO,
            repo_type="model",
            filename=f"{HF_ADAPTER_PREFIX_472}/{cell}_seed{seed}/{fname}",
            token=os.environ.get("HF_TOKEN"),
        )
        shutil.copyfile(got, local)
    for fname in _ADAPTER_FILES:
        if not (dest / fname).exists():
            raise RuntimeError(f"parent adapter fetch incomplete: {dest / fname} missing")
    return dest
