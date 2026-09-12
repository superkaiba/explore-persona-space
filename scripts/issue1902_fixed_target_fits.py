#!/usr/bin/env python3
"""Fit context-checkpoint x FIXED answer-representation-checkpoint maps.

Unlike issue1902_k5_fits' answer-text-source grid, column S always uses the
same SFT-encoded SFT-generated K=5 target, regardless of the input checkpoint.
The existing plain-render L18 tensors and six folds are reused unchanged.
Each input/fold shares one production ridge decomposition across four targets.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

import issue1902_lasttoken_comparison as LC  # noqa: E402
import issue1902_lasttoken_transfer as XF  # noqa: E402
import numpy as np  # noqa: E402

from explore_persona_space.analysis.mapping_baselines import (  # noqa: E402
    identity_bias_predict,
    knn_retrieval,
)
from explore_persona_space.orchestrate.provenance import (  # noqa: E402
    as_metadata_dict,
    git_provenance,
)

STAGES = ("B", "S", "D", "R")
REPO = "superkaiba1/explore-persona-space-data"
PREFIX = "issue1902_format_reconciliation_20260912"
SEEDS = [42, 45, 46, 47, 48]


def sha256(path: Path) -> str:
    """Hash an existing artifact without loading it all into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict) -> None:
    """Atomically persist a strict JSON object."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    os.replace(tmp, path)


def pack(args: argparse.Namespace) -> None:
    """Validate parent artifacts and package the four fixed targets and contexts."""
    import torch

    args.inputs.mkdir(parents=True, exist_ok=False)
    parent = json.loads((args.parent_results / "summary.json").read_text())
    ref_ids = ref_folds = None
    sources = {}
    files = {}
    for stage in STAGES:
        target_path = args.parent_results / "targets" / f"{stage}{stage}_L18.npz"
        context_path = args.context_store / LC.HF_PREFIX / stage / "ctx/single/L18.pt"
        fold_path = args.fold_root / f"u_last_random_{stage}_single_L31.npz"
        with np.load(target_path, allow_pickle=False) as payload:
            ids = payload["row_ids"]
            y = payload["w_bar"]
            assert payload["seeds"].tolist() == SEEDS
            assert np.all(payload["k_eff"] == 5)
        ctx = torch.load(context_path, map_location="cpu", weights_only=True)
        assert [str(v) for v in ctx["row_ids"]] == ids.tolist()
        x = ctx["u_last"].float().numpy()
        with np.load(fold_path, allow_pickle=False) as folds:
            assert folds["row_ids"].tolist() == ids.tolist()
            fold_of = folds["fold_of"]
        assert x.shape == y.shape == (16391, 4096)
        assert np.isfinite(x).all() and np.isfinite(y).all()
        assert len(set(ids.tolist())) == len(ids)
        assert set(fold_of.tolist()) == set(range(6))
        if ref_ids is None:
            ref_ids, ref_folds = ids, fold_of
        else:
            assert np.array_equal(ids, ref_ids) and np.array_equal(fold_of, ref_folds)
        output = args.inputs / f"{stage}.npz"
        LC._savez(output, x=x, y=y, row_ids=ids, fold_of=fold_of)
        files[output.name] = {"sha256": sha256(output), "bytes": output.stat().st_size}
        sources[stage] = {
            name: {"path": str(path), "sha256": sha256(path)}
            for name, path in [
                ("context", context_path),
                ("target", target_path),
                ("folds", fold_path),
            ]
        }
        print(f"[phase=pack] {stage} rows={len(ids)} validated", flush=True)
    write_json(
        args.inputs / "manifest.json",
        {
            "schema": 1,
            "layer": 18,
            "render": "plain",
            "n_rows": 16391,
            "n_folds": 6,
            "fold_seed": 190231,
            "seeds": SEEDS,
            "files": files,
            "definition": "X_m = context encoded by m; Y_s = S=s generated answer encoded by s",
            "sources": sources,
            "parent_summary_sha256": sha256(args.parent_results / "summary.json"),
            "diagonal_reference": {s: parent["grid"][s + s]["r2"] for s in STAGES},
            **as_metadata_dict(git_provenance(ROOT), phase="pack-fixed-targets"),
        },
    )


def stage_inputs(args: argparse.Namespace) -> dict:
    """Download a pinned input manifest and verify every consumed file."""
    from huggingface_hub import hf_hub_download

    if args.input_revision:
        args.inputs.mkdir(parents=True, exist_ok=True)
        rel = f"{PREFIX}/fixed_target_inputs/manifest.json"
        path = Path(hf_hub_download(REPO, rel, repo_type="dataset", revision=args.input_revision))
        manifest = json.loads(path.read_text())
        for filename, expected in manifest["files"].items():
            path = Path(
                hf_hub_download(
                    REPO,
                    f"{PREFIX}/fixed_target_inputs/{filename}",
                    repo_type="dataset",
                    revision=args.input_revision,
                )
            )
            destination = args.inputs / filename
            if not destination.exists():
                destination.symlink_to(path.resolve())
            assert sha256(destination) == expected["sha256"], destination
        write_json(args.inputs / "manifest.json", manifest)
    manifest = json.loads((args.inputs / "manifest.json").read_text())
    for filename, expected in manifest["files"].items():
        path = args.inputs / filename
        assert path.stat().st_size == expected["bytes"] and sha256(path) == expected["sha256"]
    return manifest


def summarize(args: argparse.Namespace, manifest: dict, identity: dict) -> None:
    """Aggregate paired OOF errors and compare inputs against identical targets."""
    cells = {}
    errors = {}
    for source in STAGES:
        for target in STAGES:
            key = source + target
            residual = np.full(manifest["n_rows"], np.nan)
            total = residual.copy()
            baseline = residual.copy()
            companions = []
            for fold in range(6):
                stem = args.out / "folds" / f"{key}_f{fold}"
                with np.load(stem.with_suffix(".npz"), allow_pickle=False) as p:
                    idx = p["row_idx"]
                    assert np.isnan(residual[idx]).all()
                    residual[idx], total[idx], baseline[idx] = p["res"], p["tot"], p["baseline_res"]
                companions.append(json.loads(stem.with_suffix(".json").read_text()))
            assert np.isfinite(residual).all() and np.isfinite(total).all()
            score = float(1 - residual.sum() / total.sum())
            cells[key] = {
                "context_checkpoint": source,
                "target_encoding_checkpoint": target,
                "answer_generation_checkpoint": target,
                "r2": score,
                "identity_plus_bias_r2": float(1 - baseline.sum() / total.sum()),
                "n": len(residual),
                "fold_metrics": companions,
            }
            errors[key] = (residual, total)
            if source == target:
                delta = abs(score - manifest["diagonal_reference"][source])
                cells[key]["diagonal_parity_error"] = delta
                assert delta < 1e-6, (key, score, manifest["diagonal_reference"][source])
    # Shared paired row-bootstrap weights for all cells; no repeated pool reductions.
    n = manifest["n_rows"]
    weights = (
        np.random.default_rng(1944).multinomial(n, np.full(n, 1 / n), size=1000).astype(np.float64)
    )
    keys = list(errors)
    residuals = np.column_stack([errors[k][0] for k in keys])
    totals = np.column_stack([errors[k][1] for k in keys])
    boot = 1 - (weights @ residuals) / (weights @ totals)
    for column, key in enumerate(keys):
        cells[key]["row_ci95"] = np.quantile(boot[:, column], [0.025, 0.975]).tolist()
    comparisons = {}
    for target in STAGES:
        own_key, base_key = target + target, "B" + target
        assert np.array_equal(errors[own_key][1], errors[base_key][1]), "Targets must be identical"
        delta = boot[:, keys.index(own_key)] - boot[:, keys.index(base_key)]
        comparisons[target] = {
            "own_minus_base_r2": cells[own_key]["r2"] - cells[base_key]["r2"],
            "paired_row_ci95": np.quantile(delta, [0.025, 0.975]).tolist(),
        }
    write_json(
        args.out / "summary.json",
        {
            "metadata": identity,
            "cells": cells,
            "fixed_target_comparisons": comparisons,
            "uncertainty": "1000 paired row bootstrap draws, seed 1944, no refitting",
            "diagonal_parity_pass": True,
        },
    )
    print("[phase=summary] " + json.dumps({k: v["r2"] for k, v in cells.items()}), flush=True)


def run(args: argparse.Namespace) -> None:
    """Run the full 4x4 fixed-target grid with checkpointed production fits."""
    manifest = stage_inputs(args)
    identity = {
        "schema": 1,
        "estimand": "context checkpoint -> fixed target checkpoint's own answer vectors",
        "input_manifest_sha256": sha256(args.inputs / "manifest.json"),
        "input_revision": args.input_revision,
        "layer": 18,
        "render": "plain",
        "seeds": SEEDS,
        "fold_seed": 190231,
        "n_folds": 6,
        "ridge": "issue1902_lasttoken_transfer.SharedPrimalRidge unchanged",
        "script_sha256": sha256(Path(__file__)),
        "helper_sha256": {
            name: sha256(ROOT / name)
            for name in [
                "scripts/issue1902_lasttoken_transfer.py",
                "scripts/issue1902_lasttoken_comparison.py",
                "src/explore_persona_space/analysis/mapping_baselines.py",
            ]
        },
        **as_metadata_dict(git_provenance(ROOT), phase="fixed-target-grid"),
    }
    identity_path = args.out / "run_identity.json"
    if identity_path.exists():
        previous = json.loads(identity_path.read_text())
        for key in [
            "input_manifest_sha256",
            "input_revision",
            "script_sha256",
            "helper_sha256",
            "estimand",
        ]:
            assert previous[key] == identity[key], f"Refusing stale results: {key}"
    else:
        args.out.mkdir(parents=True, exist_ok=False)
        write_json(identity_path, identity)
    targets = {}
    ref_ids = ref_folds = None
    for stage in STAGES:
        with np.load(args.inputs / f"{stage}.npz", allow_pickle=False) as p:
            targets[stage] = p["y"]
            ids, folds = p["row_ids"], p["fold_of"]
        if ref_ids is None:
            ref_ids, ref_folds = ids, folds
        else:
            assert np.array_equal(ids, ref_ids) and np.array_equal(folds, ref_folds)
    unit = 0
    for source in STAGES:
        with np.load(args.inputs / f"{source}.npz", allow_pickle=False) as p:
            x = p["x"]
        for fold in range(6):
            pending = [
                t for t in STAGES if not (args.out / "folds" / f"{source}{t}_f{fold}.json").exists()
            ]
            if not pending:
                continue
            t0 = time.monotonic()
            tr, ev = ref_folds != fold, ref_folds == fold
            ridge = XF.SharedPrimalRidge(x[tr])
            xev = ridge.standardize(x[ev])
            for target in pending:
                y = targets[target]
                beta, mean, info = ridge.fit(y[tr])
                assert info["dof"] <= 0.9 * tr.sum()
                pred = xev @ beta + mean
                res, tot, cosine = LC._per_row_components(pred, y[ev], y[tr].mean(axis=0))
                baseline_pred = identity_bias_predict(x[tr], y[tr], x[ev])
                baseline_res, _, _ = LC._per_row_components(
                    baseline_pred, y[ev], y[tr].mean(axis=0)
                )
                companion = {
                    "source": source,
                    "target": target,
                    "fold": fold,
                    **info,
                    "r2": float(1 - res.sum() / tot.sum()),
                    "retrieval_cosine": knn_retrieval(pred, y[ev], metric="cosine"),
                    "retrieval_euclidean": knn_retrieval(pred, y[ev], metric="euclidean"),
                }
                stem = args.out / "folds" / f"{source}{target}_f{fold}"
                LC._savez(
                    stem.with_suffix(".npz"),
                    row_idx=np.flatnonzero(ev),
                    res=res,
                    tot=tot,
                    cosine=cosine,
                    baseline_res=baseline_res,
                )
                write_json(stem.with_suffix(".json"), companion)
                unit += 1
                print(
                    f"[phase=fit] unit={unit}/96 {source}->{target} fold={fold} "
                    f"r2={companion['r2']:.6f} elapsed={time.monotonic() - t0:.1f}s",
                    flush=True,
                )
                del beta, pred, baseline_pred
            del ridge, xev
    summarize(args, manifest, identity)
    write_json(
        args.out / "complete.json",
        {"status": "complete", "summary_sha256": sha256(args.out / "summary.json")},
    )


def upload(args: argparse.Namespace) -> None:
    """Upload the declared folder and verify bytes at the returned immutable revision."""
    from huggingface_hub import HfApi

    api = HfApi()
    source = args.inputs if args.upload_kind == "inputs" else args.out
    remote = f"{PREFIX}/fixed_target_{args.upload_kind}"
    files = {str(p.relative_to(source)): p for p in source.rglob("*") if p.is_file()}
    assert files
    commit = api.upload_folder(
        repo_id=REPO,
        repo_type="dataset",
        folder_path=str(source),
        path_in_repo=remote,
        commit_message=f"1902 fixed-target {args.upload_kind}",
    )
    entries = api.get_paths_info(
        REPO, [f"{remote}/{p}" for p in files], repo_type="dataset", revision=commit.oid
    )
    assert len(entries) == len(files)
    for entry in entries:
        local = files[entry.path.removeprefix(remote + "/")]
        assert local.stat().st_size == entry.size
        expected = entry.lfs.sha256 if entry.lfs else entry.blob_id
        actual = (
            sha256(local)
            if entry.lfs
            else hashlib.sha1(
                b"blob " + str(local.stat().st_size).encode() + b"\0" + local.read_bytes()
            ).hexdigest()
        )
        assert actual == expected, entry.path
    print(
        json.dumps(
            {"upload_verified": True, "revision": commit.oid, "prefix": remote, "files": len(files)}
        ),
        flush=True,
    )


def main() -> None:
    """Parse the bounded driver interface and execute one phase."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("phase", choices=["pack", "run", "upload"])
    ap.add_argument("--inputs", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=ROOT / "eval_results/issue_1902/fixed_target_l18")
    ap.add_argument("--parent-results", type=Path)
    ap.add_argument("--context-store", type=Path)
    ap.add_argument("--fold-root", type=Path)
    ap.add_argument("--input-revision")
    ap.add_argument("--upload-kind", choices=["inputs", "results"], default="results")
    args = ap.parse_args()
    if args.phase == "pack":
        assert all([args.parent_results, args.context_store, args.fold_root])
        pack(args)
    elif args.phase == "run":
        run(args)
    else:
        upload(args)


if __name__ == "__main__":
    main()
