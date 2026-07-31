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


def collect_eval_rung(results_root: Path) -> dict[tuple[str, str], dict]:
    """Merge the per-lane behavior-SPECIFIC eval-rung reads.

    Each fan-out lane writes ``<results_root>/<behavior>/<kind>/map_diagnostics.json``
    keyed ``"<variant>|<u_label>"``, and the eval-rung block under that key is
    the SECOND map-quality read (the payload carries the U-pool holdout read;
    see ``_eval_rung_reconstruction`` in issue1739_fits.py for why the
    behavior-specific one cannot live in the shared payload).

    Returns ``{(kind, "variant|u_label"): {behavior: eval_rung_block}}``. A lane
    that has not run yet is simply absent — the collector reports coverage
    rather than failing, so map_quality.json is readable mid-fan-out.
    """
    out: dict[tuple[str, str], dict] = {}
    root = Path(results_root)
    if not root.is_dir():
        return out
    for diag_path in sorted(root.glob("*/*/map_diagnostics.json")):
        kind = diag_path.parent.name
        behavior = diag_path.parent.parent.name
        try:
            diag = json.loads(diag_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:  # fail loud: a lane wrote a bad file
            raise RuntimeError(f"unreadable lane diagnostics {diag_path}: {exc}") from exc
        if not isinstance(diag, dict):
            continue
        for map_key, block in diag.items():
            if not isinstance(block, dict):
                continue
            er = block.get("eval_rung")
            if isinstance(er, dict):
                out.setdefault((kind, map_key), {})[behavior] = er
    return out


def collect(tensors_root: Path, kinds: list[str], results_root: Path | None = None) -> dict:
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
    eval_rung = collect_eval_rung(results_root) if results_root is not None else {}
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
                    "eval_rung_by_behavior": eval_rung.get(
                        (kind, f"{meta.get('variant')}|{meta.get('u_label')}"), {}
                    ),
                }
            )
    n_with_eval = sum(1 for r in rows if r["eval_rung_by_behavior"])
    return {
        "issue": 1739,
        "round": "nonlinear_map",
        "maps": rows,
        "n_maps": len(rows),
        "n_maps_with_eval_rung": n_with_eval,
        "notes": [
            "BOTH map-quality reads per map: r2_map / r2_identity_bias / knn are "
            "the U-pool HOLDOUT reads carried by the shared behavior-independent "
            "payload; eval_rung_by_behavior is the per-BEHAVIOR eval-rung "
            "reconstruction R^2 computed by each fan-out lane (same fits.r2_pooled "
            "estimator, so the two are directly comparable). Expect the eval-rung "
            "read far BELOW the U-pool read -- an off-distribution extrapolation "
            "from the #1092 WildChat pool onto behavior eval distributions is a "
            "recordable finding, not a bug.",
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
    ap.add_argument(
        "--results-root",
        type=Path,
        default=None,
        help=(
            "fan-out results root (<root>/<behavior>/<kind>/map_diagnostics.json); "
            "merges the per-behavior eval-rung reconstruction R^2 — the SECOND "
            "map-quality read. Omit to emit U-pool holdout reads only."
        ),
    )
    args = ap.parse_args()

    payload = collect(args.tensors_root, args.kinds, results_root=args.results_root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_suffix(args.out.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(args.out)
    print(f"[nlmap-collect] {payload['n_maps']} maps -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
