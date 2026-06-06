# ruff: noqa: RUF003  # em-dash + Qwen marker " ※" + × + − intentional
#!/usr/bin/env python3
"""Task #504 Phase 0.5 — identification-gate subprocess entrypoint (plan §4.2).

CPU-only (consumes cached layer-10/15/20 centroids from data/issue_472/). Picks
the 4 positioned negatives + the smoke mid-band N, runs Gates A/B/C at each
layer, max-length-checks the villain R, writes phase0_5_gates.json.

Usage:
    uv run python scripts/i504_phase_phase05.py \
        --centroids-dir data/issue_472 \
        --r-train-path data/issue_472/on_policy_R/R_train.json \
        --out-path eval_results/issue_504/phase0_5_gates.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

load_dotenv()

log = logging.getLogger("i504.phase_phase05")


def _load_centroids_layer(centroids_dir: Path, layer: int) -> dict[str, np.ndarray]:
    """Load centroids_L<layer>.pt as ``{persona_name: vector}`` (float32 numpy).

    The canonical #472 producer (``scripts/i472_phase_centroids.py`` ->
    ``contrastive_neg_geometry_472.centroids.build_centroids``) writes a
    STRUCTURED bundle::

        {
            "centroids":     Tensor[N, D],   # row i = persona names[i]
            "persona_names": list[str] (len=N),
            "cos_matrix":    Tensor[N, N],
            "layer":         int,
            "base_model":    str,
            "questions":     list[str],
        }

    The CPU smoke (``scripts/i504_smoke_local.py``) historically wrote a
    LEGACY flat ``{persona: tensor}`` layout. This loader prefers the
    structured schema (production), and falls back to the flat layout for
    smoke-compat. The return shape is always a flat ``{name: ndarray}``.
    """
    import torch

    path = centroids_dir / f"centroids_L{layer}.pt"
    if not path.exists():
        raise FileNotFoundError(
            f"centroids missing at {path} — run scripts/i472_phase_centroids.py first."
        )
    obj = torch.load(path, map_location="cpu", weights_only=False)

    # PRIMARY: structured #472 schema.
    if isinstance(obj, dict) and {"centroids", "persona_names"}.issubset(obj.keys()):
        mat = obj["centroids"]
        names = list(obj["persona_names"])
        if hasattr(mat, "detach"):  # torch.Tensor
            mat_np = mat.detach().to(dtype=torch.float32).cpu().numpy()
        else:
            mat_np = np.asarray(mat, dtype=np.float32)
        if mat_np.ndim != 2:
            raise ValueError(f"centroids at {path} must be 2-D (N, D); got shape {mat_np.shape}.")
        if mat_np.shape[0] != len(names):
            raise ValueError(
                f"centroids/persona_names length mismatch at {path}: "
                f"centroids shape {mat_np.shape} vs {len(names)} names."
            )
        return {name: mat_np[i] for i, name in enumerate(names)}

    # FALLBACK: legacy flat {persona: tensor} layout (smoke synthetic).
    if not isinstance(obj, dict):
        raise TypeError(
            f"unexpected centroids payload at {path}: top-level type "
            f"{type(obj).__name__}; expected dict (structured or flat)."
        )
    out: dict[str, np.ndarray] = {}
    for name, vec in obj.items():
        if not isinstance(name, str):
            raise TypeError(
                f"unexpected centroids payload at {path}: non-string key "
                f"{name!r} (type {type(name).__name__}); the structured #472 "
                "schema was not detected — is the file the right layout?"
            )
        if hasattr(vec, "detach"):  # torch.Tensor
            arr = vec.detach().to(dtype=torch.float32).cpu().numpy()
        else:
            arr = np.asarray(vec, dtype=np.float32)
        out[name] = arr
    return out


def _cos_to_source(centroids: dict[str, np.ndarray], source: str) -> dict[str, float]:
    """Bank-wide {persona: cos(persona, source)} from raw centroids (no I/O)."""
    if source not in centroids:
        raise KeyError(f"source {source!r} missing from centroids — bank/centroids drift?")
    src = centroids[source].astype(np.float64)
    src_norm = float(np.linalg.norm(src))
    if src_norm == 0.0:
        raise RuntimeError(f"source {source!r} centroid has zero norm.")
    out: dict[str, float] = {}
    for name, v in centroids.items():
        vd = v.astype(np.float64)
        nv = float(np.linalg.norm(vd))
        if nv == 0.0:
            out[name] = 0.0
            continue
        out[name] = float(np.dot(vd, src) / (nv * src_norm))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--centroids-dir", type=Path, default=Path("data/issue_472"))
    ap.add_argument(
        "--r-train-path",
        type=Path,
        default=Path("data/issue_472/on_policy_R/R_train.json"),
        help=(
            "Phase 0.5 max-length check reads villain on-policy R from here; the "
            "max response-token length must be <= train max_length (1024)."
        ),
    )
    ap.add_argument(
        "--out-path", type=Path, default=Path("eval_results/issue_504/phase0_5_gates.json")
    )
    ap.add_argument(
        "--headline-layer",
        type=int,
        default=10,
        help="Pick positioned-N's at this layer + run Gates here first (plan §4.2).",
    )
    ap.add_argument(
        "--fallback-layers",
        default="15,20",
        help="Comma-separated fallback layers (plan §4.2 failure tree).",
    )
    ap.add_argument("--sentinel-path", type=Path, default=None)
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=os.environ.get("EPS_LOG_LEVEL", "INFO"),
        format="%(asctime)s [phase=phase05] %(name)s %(levelname)s | %(message)s",
        stream=sys.stdout,
    )

    from explore_persona_space.experiments.contrastive_neg_geometry_472.r_generate import (
        load_r_artifact,
    )
    from explore_persona_space.experiments.contrastive_neg_geometry_504 import (
        ALWAYS_INCLUDE_NEGATIVE,
        SOURCE_PERSONA,
    )
    from explore_persona_space.experiments.contrastive_neg_geometry_504.phase05 import (
        run_phase05,
        write_phase05_artifact,
    )

    fallback_layers = tuple(int(x) for x in args.fallback_layers.split(",") if x.strip())

    # Load centroids per layer.
    centroids_by_layer: dict[int, dict[str, np.ndarray]] = {}
    cos_to_source_by_layer: dict[int, dict[str, float]] = {}
    for lay in (args.headline_layer, *fallback_layers):
        centroids = _load_centroids_layer(args.centroids_dir, lay)
        log.info("[load] layer=%d, %d personas", lay, len(centroids))
        centroids_by_layer[lay] = centroids
        cos_to_source_by_layer[lay] = _cos_to_source(centroids, SOURCE_PERSONA)

    # Load villain R for max-length check.
    r_train = load_r_artifact(args.r_train_path)

    report = run_phase05(
        centroids_by_layer=centroids_by_layer,
        cos_to_source_by_layer=cos_to_source_by_layer,
        r_train_villain=r_train,
        source=SOURCE_PERSONA,
        default_persona=ALWAYS_INCLUDE_NEGATIVE,
        headline_layer=args.headline_layer,
        fallback_layers=fallback_layers,
    )
    write_phase05_artifact(report, args.out_path)

    if args.sentinel_path is not None:
        args.sentinel_path.parent.mkdir(parents=True, exist_ok=True)
        args.sentinel_path.write_text(
            json.dumps(
                {
                    "sentinel_schema_version": 1,
                    "kind": "epm:progress",
                    "version": 1,
                    "task_id": 504,
                    "phase": "phase05",
                    "by": "i504_phase_phase05",
                    "ts": datetime.now(UTC).isoformat(),
                    "note": json.dumps(
                        {
                            "verdict": report.get("verdict"),
                            "chosen_layer": report.get("chosen_layer"),
                            "max_length_check": report.get("max_length_check"),
                            "arm_to_positioned_n": report.get("arm_to_positioned_n"),
                            "smoke_mid_band_n": report.get("smoke_mid_band_n"),
                            "n_held_out_panel": len(report.get("held_out_panel", [])),
                            "out_path": str(args.out_path),
                        }
                    ),
                },
                indent=2,
            )
        )

    log.info(
        "[phase=phase05] verdict=%s, chosen_layer=%s, arm_to_n=%s, smoke_mid_band_n=%s",
        report.get("verdict"),
        report.get("chosen_layer"),
        report.get("arm_to_positioned_n"),
        report.get("smoke_mid_band_n"),
    )
    if report.get("verdict") != "pass":
        log.error("[phase=phase05] FAIL — see gate_results in %s", args.out_path)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
