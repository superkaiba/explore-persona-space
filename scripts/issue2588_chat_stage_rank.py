"""Stage only the selected chat layer through atomic, pinned Hub downloads."""

from __future__ import annotations

import argparse
import json
import math
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import issue2588_chat_rank as rank


def stage(source_root: Path, revision: str, arm: str, *, api=None, fixed_layer=None) -> dict:
    """Build the rank consumer's exact layout and verify immutable content hashes."""
    from huggingface_hub import HfApi

    from explore_persona_space.orchestrate import hub

    if not re.fullmatch(r"[0-9a-f]{40}", revision) or arm not in rank.POSITIONS:
        raise ValueError("An immutable revision and a reviewed condition are required")
    api = HfApi() if api is None else api
    run_id = "qwen3-chat-v3"
    position = rank.POSITIONS[arm]
    cell = source_root / "generic" / run_id / "cells_cap_long" / f"q3_8b_{arm}"
    prefix = f"{rank.PANEL_PREFIX}/generic/{run_id}/q3_8b/{'nothink' if arm == 'a' else 'think'}"
    metadata = {
        f"{prefix}/run_identity.json": cell / "run_identity.json",
        f"{prefix}/fits/fits_{position}.json": cell / "fits" / f"fits_{position}.json",
        **{
            f"{prefix}/analysis_tensors/capture/{split}/rows.json": cell
            / "capture"
            / split
            / "rows.json"
            for split in rank.SPLITS
        },
    }
    print(f"[rank-stage] start condition={arm} immutable_revision={revision}", flush=True)

    def fetch(mapping: dict[str, Path]) -> None:
        """Check exact remote metadata, then use the canonical atomic file stager."""
        entries = hub.retry_transient(
            lambda: api.get_paths_info(
                rank.HF_REPO, list(mapping), repo_type="dataset", revision=revision
            ),
            what=f"selected rank-stage inventory {arm}",
        )
        by_path = {entry.path: entry for entry in entries}
        if set(by_path) != set(mapping):
            raise ValueError("Missing selected rank-stage files")
        missing = [
            (remote, entry.size)
            for remote, entry in by_path.items()
            if not mapping[remote].exists()
        ]
        if missing:
            hub._assert_stage_headroom(source_root, missing, what=f"Qwen3 rank condition {arm}")

        def one(remote: str) -> Path:
            return hub.stage_hub_file(rank.HF_REPO, remote, mapping[remote], revision=revision)

        with ThreadPoolExecutor(max_workers=4) as pool:
            for index, path in enumerate(pool.map(one, mapping), 1):
                print(
                    f"[rank-stage] unit {index}/{len(mapping)} {path.relative_to(cell)}", flush=True
                )
        records = {
            str(path.relative_to(cell)): {"bytes": path.stat().st_size, "sha256": rank.sha256(path)}
            for path in mapping.values()
        }
        rank.verify_durable_inputs(cell, arm, run_id, revision, records, api)

    fetch(metadata)
    identity = json.loads((cell / "run_identity.json").read_text())
    expected = {
        "surface": "generic",
        "run_id": run_id,
        "cell": f"q3_8b_{arm}",
        "model_id": "Qwen/Qwen3-8B",
        "model_revision": rank.MODEL_REVISION,
        "manifest_revision": rank.MANIFEST_REVISION,
        "source_sha": "0a06988756f6b407e4d1562b0ac8d88af52dde2d",
        "smoke": False,
        "cap_profile": "long",
        "layer_set": "swept",
    }
    if any(identity.get(k) != v for k, v in expected.items()):
        raise ValueError("Unreviewed rank-stage identity")
    fit = json.loads((cell / "fits" / f"fits_{position}.json").read_text())
    if fit["identity"] != identity or fit["input_position"] != position:
        raise ValueError("Rank-stage fit identity mismatch")
    layer_grid = (*range(0, 36, 2), 35)
    if set(fit["layers"]) != {str(layer) for layer in layer_grid}:
        raise ValueError("Incomplete rank-stage layer sweep")
    layer = fit["layer_star"]
    if type(layer) is not int or layer not in layer_grid:
        raise ValueError("Invalid selected layer")
    scores = {}
    for candidate in layer_grid:
        score = fit["layers"][str(candidate)]["knn_val"]["ridge"]["cosine"]["acc_at_k"]["1"]
        if type(score) not in (int, float) or not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("Invalid validation retrieval score")
        scores[candidate] = score
    selected = max(layer_grid, key=scores.__getitem__)
    dimension = fit["layers"][str(layer)]["d"]
    if layer != selected or type(dimension) is not int or dimension != 4096:
        raise ValueError("Invalid selected layer")
    if fixed_layer is not None:
        if type(fixed_layer) is not int or fixed_layer not in layer_grid:
            raise ValueError("Invalid fixed layer")
        layer = fixed_layer
        if fit["layers"][str(layer)]["d"] != 4096:
            raise ValueError("Invalid fixed-layer dimension")
    tensors = {}
    for split in rank.SPLITS:
        rows = json.loads((cell / "capture" / split / "rows.json").read_text())["rows"]
        ids = [row["row_id"] for row in rows]
        if not ids or len(ids) != len(set(ids)):
            raise ValueError("Empty or duplicate rank-stage rows")
        fit_key = {"train_10k": "tr", "val_400": "val", "test_1000": "te"}[split]
        count = fit["layers"][str(layer)]["n"][fit_key]
        if type(count) is not int or len(ids) != count:
            raise ValueError("Selected-fit and capture row counts differ")
        for index in range(math.ceil(len(ids) / 500)):
            relative = f"capture/{split}/L{layer:02d}/shard{index:03d}.npz"
            tensors[f"{prefix}/analysis_tensors/{relative}"] = cell / relative
    fetch(tensors)
    manifest = rank.input_manifest(cell, position, layer)
    verified = rank.verify_durable_inputs(cell, arm, run_id, revision, manifest, api)
    result = {
        "status": "PASS",
        "arm": arm,
        "layer": layer,
        "bytes": sum(r["bytes"] for r in manifest.values()),
        "files": len(manifest),
        "durable_verification": verified,
    }
    if fixed_layer is not None:
        result["layer_selection"] = "fixed_layer_control"
    receipt = f"{arm}.json" if fixed_layer is None else f"{arm}_fixed_L{layer:02d}.json"
    rank.write_json(source_root / "rank_staging" / receipt, result)
    return result


def main(argv: list[str] | None = None) -> int:
    """Stage existing artifacts only; never launch generation or a fit."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--hf-revision", required=True)
    parser.add_argument("--arm", choices=tuple(rank.POSITIONS), required=True)
    args = parser.parse_args(argv)
    rank.load_dotenv()
    print(json.dumps(stage(args.source_root, args.hf_revision, args.arm)), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
