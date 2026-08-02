"""Item 3: NeuronPedia auto-interp for the SAE features most aligned with the
WORST-predicted answer-PCA directions.

#1482's `twoway_residual/residual_alignment.json` already banks, per cell, the 20
worst-predicted target PCs and — for each — the SAE decoder column with the largest
|cos| (`argmax_feature_per_worst`) plus a matched random-unit-vector null. What was
never run is the JOIN of those argmax feature ids against the NeuronPedia
explanation export: the existing puller (`issue1482_feature_extremes.phase_neuronpedia`)
was pointed at the best/worst-predicted FEATURES, a different selection.

This script reuses that puller's transport + parse helpers verbatim and only swaps
the selection. Computes nothing new; no GPU.
"""

from __future__ import annotations

import argparse
import gzip
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import quote

import issue1482_feature_extremes as FX

from explore_persona_space.task_workflow import repo_root

ALIGNMENT = "eval_results/issue_1482/twoway_residual/residual_alignment.json"
OUT_DIR = "eval_results/issue_1482/worst_pc_autointerp"


def _selection(root: Path) -> dict:
    """Per-cell worst-PC -> argmax SAE feature, from the banked alignment file."""
    cells = json.loads((root / ALIGNMENT).read_text())["cells"]
    per_cell, want = {}, set()
    for name, c in cells.items():
        sae = c.get("sae_alignment")
        if not sae:
            continue
        feats = [int(f) for f in sae["argmax_feature_per_worst"]]
        rows = [
            {
                "pc_index": int(pc),
                "pc_r2": float(r2),
                "feat_id": fid,
                "abs_cos": float(cos),
            }
            for pc, r2, fid, cos in zip(
                c["worst_indices"], c["worst_r2"], feats, sae["max_abs_cos_per_worst"]
            )
        ]
        per_cell[name] = {
            "arm": c["arm"],
            "layer": c["layer"],
            "fitter": c["fitter"],
            "null": sae["null_random_unit_max_over_dictionary"],
            "worst": rows,
        }
        want |= set(feats)
    if not want:
        raise RuntimeError(f"no sae_alignment blocks found in {ALIGNMENT}")
    return {"per_cell": per_cell, "want": sorted(want)}


def _fetch_explanations(want: set[int], cache: Path) -> dict[str, dict]:
    """Stage the NP explanation export (resumable) and keep only `want`."""
    cache.mkdir(parents=True, exist_ok=True)
    keys = FX._np_batch_keys()
    if not keys:
        raise RuntimeError(f"no explanation batches under {FX.NP_PREFIX}")
    print(f"[np] {len(keys)} explanation batches under {FX.NP_PREFIX}")

    def stage(key: str) -> Path:
        dest = cache / key.rsplit("/", 1)[-1]
        if dest.exists() and dest.stat().st_size > 0:
            return dest
        blob = FX._http_get(f"{FX.NP_S3}/{quote(key)}")
        tmp = dest.with_name(dest.name + ".part")
        tmp.write_bytes(blob)
        tmp.replace(dest)
        return dest

    with ThreadPoolExecutor(max_workers=6) as pool:
        staged = list(pool.map(stage, keys))
    print(f"[np] staged {len(staged)} batches into {cache}")

    found: dict[str, dict] = {}
    for i, path in enumerate(staged):
        for line in gzip.decompress(path.read_bytes()).decode("utf-8").split("\n"):
            if not line.strip():
                continue
            rec = json.loads(line)
            idx = int(rec["index"])
            if idx not in want:
                continue
            found[str(idx)] = {
                "description": (rec.get("description") or "").strip(),
                "explanationModel": rec.get("explanationModelName"),
                "url": FX.NP_FEATURE_URL.format(
                    model=FX.NP_MODEL_ID, source=FX.NP_SOURCE_ID, index=idx
                ),
            }
        if (i + 1) % 50 == 0:
            print(f"[np] parsed {i + 1}/{len(staged)} batches, {len(found)} hits")
    return found


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=OUT_DIR)
    args = ap.parse_args()

    root = repo_root()
    out = root / args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    sel = _selection(root)
    want = set(sel["want"])
    print(f"[sel] {len(sel['per_cell'])} cells, {len(want)} distinct argmax features")

    found = _fetch_explanations(want, out / "np_cache")
    n_desc = sum(1 for v in found.values() if v["description"])
    print(f"[np] {len(found)}/{len(want)} features resolved, {n_desc} with a description")

    for cell in sel["per_cell"].values():
        for row in cell["worst"]:
            row["autointerp"] = found.get(str(row["feat_id"]), {}).get("description", "")
            row["np_url"] = found.get(str(row["feat_id"]), {}).get("url", "")

    payload = {
        "design": {
            "question": (
                "Do the SAE features most aligned with the WORST-predicted answer "
                "directions have legible auto-interp descriptions?"
            ),
            "selection": (
                "per cell: the 20 worst-predicted target PCs among the top-256, each "
                "mapped to its max-|cos| SAE decoder column (banked in "
                "twoway_residual/residual_alignment.json)"
            ),
            "np_source": {"model": FX.NP_MODEL_ID, "source": FX.NP_SOURCE_ID},
            "caveat": (
                "alignment is WEAK by construction: max|cos| 0.107-0.141 against a "
                "matched random-unit null of mean 0.076 / max 0.095 (~1.6x). A "
                "description here labels the nearest dictionary atom, NOT the PC."
            ),
        },
        "coverage": {
            "n_features_wanted": len(want),
            "n_resolved": len(found),
            "n_with_description": n_desc,
        },
        "cells": sel["per_cell"],
        "explanations": found,
    }
    (out / "worst_pc_autointerp.json").write_text(json.dumps(payload, indent=1))
    print(f"[out] wrote {out / 'worst_pc_autointerp.json'}")


if __name__ == "__main__":
    main()
