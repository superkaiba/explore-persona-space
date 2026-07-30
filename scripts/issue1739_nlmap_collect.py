#!/usr/bin/env python
"""Collect issue-1739 nonlinear-map quality diagnostics into map_quality.json.

Reads the frozen per-(variant, U rung, kind) map payloads under
``<tensors-root>/maps/*__<kind>.pt`` and emits ONE JSON carrying, per map:

* ``r2_map`` — held-out reconstruction R^2 of the fitted nonlinear map.
* ``r2_identity_bias`` — the identity+learned-bias baseline (v_hat = x + b,
  b = train-fold mean of y - x) on the SAME held-out rows.
* ``knn`` — the retrieval read (euclidean + cosine) among the held-out pool.

The last two are the standing mapping companions every representation-map fit
must report alongside R^2 (CLAUDE.md § "Identity+learned-bias baseline AND
kNN-retrieval metric"): R^2 alone both overstates a map (variance a constant
shift already explains) and understates one (discriminative but mis-scaled).

Reads only the persisted metas — never re-runs a fit — so the companions stay
available after the fitting instance is gone.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import torch  # noqa: E402  (after load_dotenv: thread caps are frozen at torch import)


def _git_commit() -> str:
    """Short HEAD sha, or 'unknown' outside a git tree (reproducibility meta)."""
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _mean(values: list[float | None]) -> float | None:
    """Mean over the non-None entries, or None when every entry is missing."""
    finite = [float(v) for v in values if v is not None]
    return sum(finite) / len(finite) if finite else None


def collect(tensors_root: Path, kinds: list[str]) -> dict:
    """Gather every ``maps/*__<kind>.pt`` meta into a serializable payload.

    Returns a dict with a ``maps`` list (one row per payload file) plus
    reproducibility metadata. Fails loud on a payload that cannot be read —
    a silently skipped map would understate the companion coverage.

    ``mapfit.diagnostics`` nests the companion reads PER LAYER (one entry per
    fitted layer, keyed ``layer_idx``), so each row carries the full
    ``per_layer`` array AND the across-layer means the fits log line reports
    (``mean r2_map=...``) for a scannable headline.
    """
    maps_dir = Path(tensors_root) / "maps"
    rows: list[dict] = []
    for kind in kinds:
        for path in sorted(maps_dir.glob(f"*__{kind}.pt")):
            blob = torch.load(path, map_location="cpu", weights_only=False)
            meta = blob.get("meta", {})
            diag = meta.get("diagnostics", {}) or {}
            # diagnostics `layer_idx` is POSITIONAL within the fitted stack; resolve
            # it to the real transformer layer via the meta's own `layers` list so
            # map_quality.json is readable without cross-referencing the payload.
            layers = meta.get("layers") or []
            per_layer = []
            for row in diag.get("per_layer") or []:
                pos = row.get("layer_idx")
                layer = layers[pos] if isinstance(pos, int) and 0 <= pos < len(layers) else None
                per_layer.append(
                    {
                        "layer": layer,
                        "layer_pos": pos,
                        "r2_map": row.get("r2_map"),
                        "r2_identity_bias": row.get("r2_identity_bias"),
                        "knn": row.get("knn"),
                    }
                )
            rows.append(
                {
                    "file": str(path.relative_to(tensors_root)),
                    "variant": meta.get("variant"),
                    "u_label": meta.get("u_label"),
                    "map_kind": meta.get("map_kind", kind),
                    "layers": meta.get("layers"),
                    "n_train": diag.get("n_train"),
                    "n_holdout": diag.get("n_holdout"),
                    "r2_map_mean": _mean([r["r2_map"] for r in per_layer]),
                    "r2_identity_bias_mean": _mean([r["r2_identity_bias"] for r in per_layer]),
                    "per_layer": per_layer,
                    "fit_meta": {k: v for k, v in diag.items() if k != "per_layer"},
                    "map_git_commit": meta.get("git_commit"),
                    "map_ts": meta.get("ts"),
                }
            )
    return {
        "issue": 1739,
        "round": "nonlinear_map",
        "maps": rows,
        "n_maps": len(rows),
        "notes": [
            "u_label=250 is a deliberately UNDER-DETERMINED regime for the "
            "512-hidden MLP map (n_train < parameter count): its rung is kept "
            "for cell-for-cell comparability with the linear u=250 rung and "
            "carries NO headline — read it as a regularization-limit point.",
            "r2_identity_bias + knn are the standing mapping companions "
            "(CLAUDE.md identity+bias / kNN-retrieval bullet); the two reads "
            "dissociate from R^2 in both directions.",
        ],
        "git_commit": _git_commit(),
        "torch_version": torch.__version__,
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tensors-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--kinds", nargs="+", default=["mlp", "kernel"])
    args = ap.parse_args()

    payload = collect(args.tensors_root, args.kinds)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_suffix(args.out.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(args.out)
    print(f"[nlmap-collect] {payload['n_maps']} maps -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
