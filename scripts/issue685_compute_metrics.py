#!/usr/bin/env python
"""Issue #685 Phase B — compute the four metric families from the Phase-A vectors.

CPU. Loads ``store/issue685[_smoke]/{instruct,base}_context_vectors.pt``, computes
the shift ``Delta_l(C,b) = v_l(C+b) - v_l(C)`` and the four metric families
(relative magnitude, direction-consistency cosine + PC1 share, behavior
separability, consistency null) per model, optionally with the known-direction
projection ``|Delta . u_hat| / ||Delta||`` when a ``u`` direction file
(``{tag}_known_directions.pt``, produced by ``issue685_known_directions.py``) is
present. Writes ``eval_results/issue_685[_smoke]/metrics.json``.

Usage::

    uv run python scripts/issue685_compute_metrics.py                 # full, both models
    uv run python scripts/issue685_compute_metrics.py --smoke         # tiny verification
"""

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import torch

from explore_persona_space.analysis.issue685.metrics import behavior_shift_metrics


def _load_vectors(pt_path: Path) -> tuple[dict[str, dict[int, torch.Tensor]], dict]:
    """Load a Phase-A ``.pt`` into ``{condition_name: {layer: (H,) vec}}`` + metadata.

    The Phase-A file stores ``centroids: {layer: (n_cond, H)}`` aligned to
    ``persona_names`` (== condition_names). Reshape to a per-condition dict so the
    metrics layer can index by name (``bare__{c}`` / ``{c}__{b}``).
    """
    payload = torch.load(pt_path, weights_only=True)
    centroids = payload["centroids"]  # {layer: (n_cond, H)}
    names = payload["persona_names"]
    metadata = payload.get("metadata", {})
    by_condition: dict[str, dict[int, torch.Tensor]] = {n: {} for n in names}
    for layer, mat in centroids.items():
        assert mat.shape[0] == len(names), (layer, mat.shape, len(names))
        for i, n in enumerate(names):
            by_condition[n][layer] = mat[i]
    return by_condition, metadata


def _split_bare_aug(
    by_condition: dict[str, dict[int, torch.Tensor]],
    context_names: list[str],
    behaviors: list[str],
) -> tuple[dict[str, dict[int, torch.Tensor]], dict[str, dict[int, torch.Tensor]]]:
    """Split the loaded vectors into bare-by-context and aug-by-condition maps.

    Returns ``(bare_by_context[c], aug_by_condition[f"{c}__{b}"])`` keyed exactly
    as ``behavior_shift_metrics`` expects.
    """
    bare_by_context = {c: by_condition[f"bare__{c}"] for c in context_names}
    aug_by_condition = {
        f"{c}__{b}": by_condition[f"{c}__{b}"] for c in context_names for b in behaviors
    }
    return bare_by_context, aug_by_condition


def _load_known_directions(
    path: Path, behaviors: list[str], layers: list[int]
) -> dict[tuple[str, int], torch.Tensor] | None:
    """Load ``{(behavior, layer): (H,) u}`` from a known-directions .pt, or None.

    The file stores ``directions: {behavior: {layer: (H,) vec}}``. Missing
    (behavior, layer) entries are skipped (the projection field is then omitted
    for that cell).
    """
    if not path.exists():
        return None
    payload = torch.load(path, weights_only=True)
    dirs = payload["directions"]
    out: dict[tuple[str, int], torch.Tensor] = {}
    for b in behaviors:
        if b not in dirs:
            continue
        for layer in layers:
            if layer in dirs[b]:
                out[(b, layer)] = dirs[b][layer]
    return out or None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Issue #685 Phase B — geometry metrics from the Phase-A context vectors.",
    )
    parser.add_argument("--smoke", action="store_true", help="tiny verification slice.")
    parser.add_argument("--store-dir", default=None, help="override the Phase-A store dir.")
    parser.add_argument("--out-dir", default=None, help="override the eval_results out dir.")
    parser.add_argument(
        "--null-n-perm",
        type=int,
        default=None,
        help="consistency-null draws (default 200 full / 20 smoke).",
    )
    args = parser.parse_args()

    smoke = args.smoke
    store_dir = (
        Path(args.store_dir)
        if args.store_dir
        else Path("store/issue685_smoke" if smoke else "store/issue685")
    )
    out_dir = (
        Path(args.out_dir)
        if args.out_dir
        else Path("eval_results/issue685_smoke" if smoke else "eval_results/issue_685")
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    null_n_perm = args.null_n_perm if args.null_n_perm is not None else (20 if smoke else 200)

    model_tags = ["instruct"] if smoke else ["instruct", "base"]

    result: dict = {
        "task": 685,
        "phase": "B",
        "smoke": smoke,
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "argv": sys.argv[1:],
        "null_n_perm": null_n_perm,
        "models": {},
    }

    for tag in model_tags:
        pt_path = store_dir / f"{tag}_context_vectors.pt"
        if not pt_path.exists():
            raise FileNotFoundError(
                f"[issue685.B] missing Phase-A vectors {pt_path}; run issue685_extract_shifts.py "
                f"{'--smoke ' if smoke else ''}first."
            )
        by_condition, a_meta = _load_vectors(pt_path)
        context_names = a_meta["context_names"]
        behaviors = a_meta["behavior_names"]
        layers = a_meta["layers"]
        bare_by_context, aug_by_condition = _split_bare_aug(by_condition, context_names, behaviors)

        known_dirs = _load_known_directions(
            store_dir / f"{tag}_known_directions.pt", behaviors, layers
        )

        metrics = behavior_shift_metrics(
            bare_by_context,
            aug_by_condition,
            context_names=context_names,
            behaviors=behaviors,
            layers=layers,
            known_directions=known_dirs,
            null_n_perm=null_n_perm,
            null_seed=42,
        )
        metrics["phase_a_metadata"] = a_meta
        metrics["has_known_direction_projection"] = known_dirs is not None
        result["models"][tag] = metrics
        print(
            f"[issue685.B] model={tag}: {len(behaviors)} behaviors x {len(layers)} layers; "
            f"projection={'on' if known_dirs is not None else 'off'}"
        )

    out_path = out_dir / "metrics.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"[issue685.B] wrote {out_path}")


if __name__ == "__main__":
    main()
