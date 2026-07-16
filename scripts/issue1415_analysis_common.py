"""Shared CPU-side tensor loading for the issue-1415 analysis scripts (round B).

Consumes the phase-1a capture ``.pt`` files written by
``scripts/issue1415_run_phase1.py::phase_1a`` — one file per pair at
``<activations_dir>/<pair_id>.pt`` with structure::

    {"pair_id", "layers": [...],
     "c":      {"v_c_prefix": (L,H), "v_c_context": (L,H), "v_a_mean": (L,H), ...},
     "cprime": {same},
     "repro": {...}}

Definitions (per plan v5):
- ``delta[arm][l]``  = v_c_<arm>(cprime)[l] - v_c_<arm>(c)[l]   (the steering Delta)
- ``target[l]``      = normalize(v_a_mean(cprime)[l] - v_a_mean(c)[l])
                       (the answer-side target direction; arm-independent —
                       V_a has no extraction arm)

Every function is CPU-only and fail-loud (shape asserts, no silent excepts).
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import torch

ARMS = ("prefix", "context")
LAYERS_FULL = (7, 10, 14, 17, 20, 21, 24)
PRIMARY_LAYER = 20
HIDDEN_FULL = 3584
HF_DATA_REPO = "superkaiba1/explore-persona-space-data"

_SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = _SCRIPTS_DIR.parent


def git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def repro_meta(script: str) -> dict:
    """Reproducibility metadata block for every result JSON (CLAUDE.md mandate)."""
    return {
        "script": script,
        "git_commit": git_commit(),
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "python_version": platform.python_version(),
        "torch_version": str(torch.__version__),
        "platform": platform.platform(),
        "argv": sys.argv[1:],
        "cwd": os.getcwd(),
    }


def write_json_atomic(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    tmp.replace(path)


def save_pt_atomic(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


@dataclass
class PairTensors:
    """Per-pair fp32 CPU tensors derived from one phase-1a capture file."""

    pair_id: str
    layers: list[int]
    v_c: dict[str, torch.Tensor]  # arm -> (L, H): V_c of context c
    v_a_c: torch.Tensor  # (L, H): mean answer activation under c
    v_a_cprime: torch.Tensor  # (L, H): mean answer activation under c'
    delta: dict[str, torch.Tensor]  # arm -> (L, H): v_c_arm(c') - v_c_arm(c)

    @property
    def target_raw(self) -> torch.Tensor:  # (L, H)
        return self.v_a_cprime - self.v_a_c

    def target_unit(self) -> torch.Tensor:  # (L, H), unit rows
        t = self.target_raw
        n = t.norm(dim=-1, keepdim=True)
        assert torch.all(n > 0), f"{self.pair_id}: degenerate zero answer-target at some layer"
        return t / n

    def delta_norms(self, arm: str) -> torch.Tensor:  # (L,)
        n = self.delta[arm].norm(dim=-1)
        assert torch.all(n > 0), f"{self.pair_id}/{arm}: degenerate zero Delta at some layer"
        return n


def load_pair_tensors(path: Path) -> PairTensors:
    blob = torch.load(path, map_location="cpu", weights_only=True)
    layers = list(blob["layers"])
    recs = {k: blob[k] for k in ("c", "cprime")}
    hidden = recs["c"]["v_c_context"].shape[-1]
    for rec in recs.values():
        for key in ("v_c_prefix", "v_c_context", "v_a_mean"):
            assert key in rec, f"{path}: capture record missing {key!r} (phase-1a contract)"
            assert rec[key].shape == (len(layers), hidden), (path, key, rec[key].shape)
    v_c = {arm: recs["c"][f"v_c_{arm}"].float() for arm in ARMS}
    delta = {arm: (recs["cprime"][f"v_c_{arm}"] - recs["c"][f"v_c_{arm}"]).float() for arm in ARMS}
    return PairTensors(
        pair_id=str(blob["pair_id"]),
        layers=layers,
        v_c=v_c,
        v_a_c=recs["c"]["v_a_mean"].float(),
        v_a_cprime=recs["cprime"]["v_a_mean"].float(),
        delta=delta,
    )


def load_all_pairs(activations_dir: Path) -> list[PairTensors]:
    """Every ``<pair_id>.pt`` in the dir, sorted by pair_id (deterministic order)."""
    files = sorted(activations_dir.glob("*.pt"))
    assert files, f"no phase-1a capture .pt files under {activations_dir}"
    pairs = [load_pair_tensors(p) for p in files]
    ref = pairs[0].layers
    for p in pairs:
        assert p.layers == ref, f"layer-set mismatch: {p.pair_id} {p.layers} vs {ref}"
    return pairs


def batched_permutations(n_draws: int, n_items: int, seed: int) -> torch.Tensor:
    """(n_draws, n_items) int64 permutations, fully batched (argsort of uniform
    draws — no per-draw Python loop), deterministic under ``seed``."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    return torch.argsort(torch.rand(n_draws, n_items, generator=gen), dim=-1)


def quantile_band(x: torch.Tensor, qs=(0.025, 0.5, 0.975)) -> dict[str, float]:
    vals = torch.quantile(x.reshape(-1).float(), torch.tensor(qs))
    return {f"p{q * 100:g}": float(v) for q, v in zip(qs, vals, strict=True)}
