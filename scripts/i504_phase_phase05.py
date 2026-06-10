# ruff: noqa: RUF001, RUF002, RUF003  # em-dash + Qwen marker " ※" + × + − + ρ intentional
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


def _cos_to_source(
    centroids: dict[str, np.ndarray],
    source: str,
    *,
    mean_center: bool = True,
) -> dict[str, float]:
    """Bank-wide {persona: cos(persona, source)} from centroids.

    Default (``mean_center=True``, #504 round-6) follows the #66/#341 methodology
    that recovered the ρ=0.67-0.87 cos-vs-leakage signal: subtract the global
    per-component mean over the FULL bank (every persona in ``centroids``,
    including the source), then L2-normalize, then dot. Without mean-centering
    the cos-to-villain range on Qwen-2.5-7B-Instruct collapses to ≈[0.92, 0.99]
    (round 1-5 #504 spread) because the raw last-token activations share a large
    shared component — mean-centering removes it.

    Pass ``mean_center=False`` to recover the round 1-5 raw-cosine behavior.
    """
    if source not in centroids:
        raise KeyError(f"source {source!r} missing from centroids — bank/centroids drift?")
    names = list(centroids.keys())
    mat = np.stack([centroids[n].astype(np.float64) for n in names], axis=0)
    if mean_center:
        mat = mat - mat.mean(axis=0, keepdims=True)
    norms = np.linalg.norm(mat, axis=1)
    src_idx = names.index(source)
    src = mat[src_idx]
    src_norm = float(norms[src_idx])
    if src_norm == 0.0:
        raise RuntimeError(f"source {source!r} centroid has zero norm after centering.")
    out: dict[str, float] = {}
    for i, name in enumerate(names):
        nv = float(norms[i])
        if nv == 0.0:
            out[name] = 0.0
            continue
        out[name] = float(np.dot(mat[i], src) / (nv * src_norm))
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
    ap.add_argument(
        "--no-mean-center",
        action="store_true",
        help=(
            "Disable bank-wide mean-centering before cosine (round 1-5 behavior). "
            "Default is mean-center ON (#504 round-6, restoring the #66/#341 "
            "methodology that produced ρ=0.67-0.87 cos-vs-leakage)."
        ),
    )
    ap.add_argument(
        "--source",
        default=None,
        help=(
            "Round-2 fix (BLOCKER #2, concern_id `fallback-source-threading`): "
            "source persona for the Phase 0.5 cos-to-source computation + "
            "positioned-N pick. The v2 Phase 0 fallback path (plan v2 §4.2) "
            "swaps villain for an easier candidate; when that fires, Phase 0.5 "
            "MUST re-run with the new source so positioned negatives are picked "
            "RELATIVE TO that source (otherwise the geometry is stale). Unset = "
            "use module default SOURCE_PERSONA = villain (legacy byte-identical)."
        ),
    )
    args = ap.parse_args(argv)
    mean_center = not args.no_mean_center

    logging.basicConfig(
        level=os.environ.get("EPS_LOG_LEVEL", "INFO"),
        format="%(asctime)s [phase=phase05] %(name)s %(levelname)s | %(message)s",
        stream=sys.stdout,
    )

    # Carry-over data dependencies from #472 (centroids, on-policy R) are
    # gitignored. Pull from HF at the pinned revision before the centroids
    # load below. Idempotent — a no-op when files are already on disk.
    # The helper lives in contrastive_neg_geometry_530 (the task that
    # surfaced the gap) but is general-purpose for any #472 carry-over.
    from explore_persona_space.experiments.contrastive_neg_geometry_530.data_deps import (
        prepare_data_dependencies,
    )

    log.info("[phase=phase05_prepare_data] auto-downloading #472 carry-over artifacts")
    prepare_data_dependencies()

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

    # Round-2 fix (BLOCKER #2): resolve effective source — fallback path passes
    # --source medical_doctor (or similar); legacy path uses module default.
    effective_source = args.source if args.source is not None else SOURCE_PERSONA
    log.info(
        "[phase=source] effective source persona for Phase 0.5 = %r (CLI --source=%r, default=%r)",
        effective_source,
        args.source,
        SOURCE_PERSONA,
    )

    fallback_layers = tuple(int(x) for x in args.fallback_layers.split(",") if x.strip())

    # Load centroids per layer.
    log.info(
        "[centering] mean_center=%s (round-6 default %s)",
        mean_center,
        "ON — #66/#341 methodology" if mean_center else "OFF — round 1-5 raw cosine",
    )
    centroids_by_layer: dict[int, dict[str, np.ndarray]] = {}
    cos_to_source_by_layer: dict[int, dict[str, float]] = {}
    for lay in (args.headline_layer, *fallback_layers):
        centroids = _load_centroids_layer(args.centroids_dir, lay)
        log.info("[load] layer=%d, %d personas", lay, len(centroids))
        centroids_by_layer[lay] = centroids
        cos = _cos_to_source(centroids, effective_source, mean_center=mean_center)
        cos_to_source_by_layer[lay] = cos
        # Diagnostic: log the cos-to-source spread per layer so a saturated /
        # narrow range is immediately visible in the log.
        vals = [v for k, v in cos.items() if k != effective_source]
        if vals:
            arr = np.asarray(vals, dtype=np.float64)
            log.info(
                "[spread] layer=%d cos_to_%s: min=%.4f median=%.4f max=%.4f span=%.4f "
                "n_below_0.7=%d n_below_0.5=%d n_below_0.3=%d",
                lay,
                effective_source,
                float(arr.min()),
                float(np.median(arr)),
                float(arr.max()),
                float(arr.max() - arr.min()),
                int((arr < 0.7).sum()),
                int((arr < 0.5).sum()),
                int((arr < 0.3).sum()),
            )

    # Load villain R for max-length check.
    r_train = load_r_artifact(args.r_train_path)

    report = run_phase05(
        centroids_by_layer=centroids_by_layer,
        cos_to_source_by_layer=cos_to_source_by_layer,
        r_train_villain=r_train,
        source=effective_source,
        default_persona=ALWAYS_INCLUDE_NEGATIVE,
        headline_layer=args.headline_layer,
        fallback_layers=fallback_layers,
        mean_center=mean_center,
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
