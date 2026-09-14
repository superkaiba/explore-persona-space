"""K5 transfer from chat assistant alone or chat assistant plus one character.

Use the established source-only GCV ridge recipe and global conversation folds.
Fit each source map once per fold and reuse it for every unseen target setting.
"""

# ruff: noqa: E402
# Load the shared-VM thread caps before importing numpy/torch-backed helpers.

from __future__ import annotations

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import argparse
import hashlib
import inspect
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts import issue2054_k5_loso_calibration as base


def source_sets(model, source_mode="chat_grid"):
    """Enumerate the requested assistant source without selecting on test results."""
    if source_mode == "plain_only":
        return {"assistant_plain_only": [f"{base.SETTINGS[1][1]}__{model}"]}
    if source_mode != "chat_grid":
        raise ValueError(f"unknown source mode: {source_mode}")
    assistant = f"{base.SETTINGS[0][1]}__{model}"
    return {"assistant_only": [assistant]} | {
        f"assistant_plus_{prefix.split('__')[0]}": [assistant, f"{prefix}__{model}"]
        for _, prefix in base.SETTINGS[2:]
    }


def audit_sources(panel, source_cells, target, fold):
    """Reject setting, checkpoint, and conversation leakage and hash train IDs."""
    if target in source_cells or not source_cells or len(set(source_cells)) != len(source_cells):
        raise ValueError("target must be excluded from a nonempty source set")
    if fold not in range(5) or any(
        c.split("__")[-1] != target.split("__")[-1] for c in source_cells
    ):
        raise ValueError("invalid model or fold")
    p = panel[target]
    test_ids = np.asarray(p["ids"])[p["membership"] == fold].tolist()
    train_ids = np.asarray(p["ids"])[p["membership"] != fold].tolist()
    if not test_ids or not train_ids or set(test_ids) & set(train_ids):
        raise RuntimeError("invalid target calibration/test split")
    sources = {}
    for cell in source_cells:
        s = panel[cell]
        ids = np.asarray(s["ids"])[s["membership"] != fold].tolist()
        if not ids or set(test_ids) & set(ids):
            raise RuntimeError("test conversations leaked into source training")
        sources[cell] = {"n_train": len(ids), "train_ids_sha256": base.loso.digest_ids(ids)}
    return {
        "source_settings": sources,
        "target": target,
        "excluded_conversation_fold": fold,
        "n_train": sum(s["n_train"] for s in sources.values()),
        "n_calibration": len(train_ids),
        "n_test": len(test_ids),
        "test_ids_sha256": base.loso.digest_ids(test_ids),
        "calibration_ids_sha256": base.loso.digest_ids(train_ids),
        "test_overlap_source": 0,
        "test_overlap_calibration": 0,
        "target_labels_used_for_source_map": False,
    }


def fingerprint(source_mode="chat_grid"):
    """Bind the subset fit recipe and all shared calibration dependencies."""
    payload = [
        base.fit_fingerprint(),
        {model: source_sets(model, source_mode) for model in base.MODELS},
    ] + [inspect.getsource(f) for f in [source_sets, audit_sources, fit, persist_map]]
    return hashlib.sha256(json.dumps(payload).encode()).hexdigest()


def persist_map(path, frozen, fp):
    """Save the raw float64 fitted map once, with a validated resume receipt."""
    receipt = path.with_suffix(".json")
    arrays = {k: getattr(frozen, k).cpu().numpy() for k in ["map", "mu_x", "sd", "mu_y"]}
    if receipt.exists():
        record = json.loads(receipt.read_text())
        if record["fingerprint"] != fp or base.sha(path) != record["sha256"]:
            raise RuntimeError("source-map checkpoint changed")
        with np.load(path, allow_pickle=False) as z:
            for key, value in arrays.items():
                np.testing.assert_allclose(z[key], value, rtol=1e-10, atol=1e-10)
        return record
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez(temporary, **arrays)
    temporary.replace(path)
    record = {
        "path": str(path.relative_to(path.parents[1])),
        "sha256": base.sha(path),
        "fingerprint": fp,
        "ridge": frozen.info(),
    }
    base.atomic_json(receipt, record)
    return record


def fit(out, inputs, model, limit, source_mode="chat_grid"):
    """Fit one map per source set and fold, then score every excluded setting."""
    sources = {r["path"]: r for r in json.loads((inputs / "inputs.json").read_text())}
    refs = {p["cell"]: p for p in base.references()["panels"]}
    panel = base.load_panel(model, sources)
    regimes = source_sets(model, source_mode)
    source_union = {cell for cells in regimes.values() for cell in cells}
    bank = base.loso.moments({c: panel[c] for c in source_union}, "cpu")
    fp, completed = fingerprint(source_mode), 0
    for regime, source_cells in regimes.items():
        targets = [c for c in panel if c not in source_cells]
        for fold in range(5):
            map_path = out / "maps" / f"{model}__{regime}__fold{fold}.npz"
            paths = {c: out / "folds" / f"{regime}__{c}__fold{fold}.json" for c in targets}
            existing = {}
            for cell, path in paths.items():
                if path.exists():
                    row = json.loads(path.read_text())
                    if (
                        row["fit_fingerprint"] != fp
                        or base.sha(path.with_suffix(".npz")) != row["coefficient_sha256"]
                    ):
                        raise RuntimeError("subset calibration checkpoint changed")
                    existing[cell] = row
            if existing:
                map_sha = base.sha(map_path)
                if any(r["source_map"]["sha256"] != map_sha for r in existing.values()):
                    raise RuntimeError("source map differs from completed predictions")
            if len(existing) == len(targets):
                continue
            started = time.monotonic()
            audits = {c: audit_sources(panel, source_cells, c, fold) for c in targets}
            training = base.loso.combine(
                {c: bank[c] for c in source_cells}, 5, drop_speaker=None, drop_fold=fold
            )
            if any(training["n"] != a["n_train"] for a in audits.values()):
                raise RuntimeError("subset source moments differ from audited rows")
            frozen = base.loso.PooledMomentRidge(**training)
            record = persist_map(map_path, frozen, fp)
            for cell in targets:
                if cell in existing:
                    continue
                cell_started = time.monotonic()
                p = panel[cell]
                train, test = p["membership"] != fold, p["membership"] == fold
                # One frozen source map for both target calibration and test predictions.
                prediction = frozen.predict_np(p["x"])
                coefficients = base.calibrate(prediction[train], p["y"][train])
                variants = base.adapted_predictions(prediction[test], coefficients)
                variants["frozen"] = prediction[test]
                variants["source_identity_bias"] = p["x"][test] + frozen.global_bias
                variants["identity_bias_target"] = p["x"][test] + (
                    p["y"][train] - p["x"][train]
                ).mean(0)
                metrics = {key: base.score(value, p["y"][test]) for key, value in variants.items()}
                path = paths[cell]
                base.save_npz(path.with_suffix(".npz"), coefficients)
                reference = refs[cell]["folds"][fold]["cohorts"]["all"]["reference"]
                result = {
                    "status": "complete",
                    "regime": regime,
                    "source_mode": source_mode,
                    "cell": cell,
                    "fold": fold,
                    "gain": coefficients["gain"],
                    "metrics": metrics,
                    "audit": audits[cell],
                    "own": reference["own"],
                    "six_setting_pool": reference["six_setting_pool"],
                    "source_map": record,
                    "fit_fingerprint": fp,
                    "coefficient_sha256": base.sha(path.with_suffix(".npz")),
                    "input_manifest_sha256": base.sha(inputs / "inputs.json"),
                    "script_sha256": base.sha(__file__),
                    "seconds": time.monotonic() - cell_started,
                    "metadata": base.as_metadata_dict(
                        base.git_provenance(cwd=REPO), phase="subset_transfer"
                    ),
                }
                base.atomic_json(path, result)
                print(
                    f"[phase=subset_transfer] {model} {regime} target={cell} fold={fold} "
                    f"seconds={result['seconds']:.1f} "
                    f"r2={metrics['frozen']['r2']:.4f}/{metrics['bias']['r2']:.4f}/{metrics['bias_scale']['r2']:.4f}",
                    flush=True,
                )
                del prediction, variants
            print(
                f"[phase=subset_map] {model} {regime} fold={fold} "
                f"seconds={time.monotonic() - started:.1f}",
                flush=True,
            )
            del frozen, training
            completed += 1
            if limit and completed >= limit:
                return


def collect(out, source_mode="chat_grid"):
    """Verify every requested target fold and persisted source map."""
    panels, maps = [], {}
    fp = fingerprint(source_mode)
    for model in base.MODELS:
        for regime, source_cells in source_sets(model, source_mode).items():
            for _, prefix in base.SETTINGS:
                cell = f"{prefix}__{model}"
                if cell in source_cells:
                    continue
                rows = []
                for fold in range(5):
                    path = out / "folds" / f"{regime}__{cell}__fold{fold}.json"
                    row = json.loads(path.read_text())
                    if (
                        row["status"] != "complete"
                        or row["cell"] != cell
                        or row["fold"] != fold
                        or row["regime"] != regime
                        or row["fit_fingerprint"] != fp
                        or base.sha(path.with_suffix(".npz")) != row["coefficient_sha256"]
                    ):
                        raise RuntimeError("missing, stale or corrupt subset fold")
                    record = row["source_map"]
                    if record["path"] not in maps:
                        maps[record["path"]] = base.sha(out / record["path"])
                    if maps[record["path"]] != record["sha256"]:
                        raise RuntimeError("corrupt source map")
                    rows.append(row)
                panels.append(
                    {
                        "cell": cell,
                        "model": model,
                        "regime": regime,
                        "source_cells": source_cells,
                        "folds": rows,
                        "r2_mean": {
                            key: float(np.mean([r["metrics"][key]["r2"] for r in rows]))
                            for key in [
                                "frozen",
                                "bias",
                                "bias_scale",
                                "source_identity_bias",
                                "identity_bias_target",
                            ]
                        },
                        "own_r2": float(np.mean([r["own"]["r2"] for r in rows])),
                    }
                )
    expected_panels, expected_maps = {"chat_grid": (42, 50), "plain_only": (10, 10)}[source_mode]
    if len(panels) != expected_panels or len(maps) != expected_maps:
        raise RuntimeError(
            f"subset coverage differs from planned {expected_panels} panels / {expected_maps} maps"
        )
    result = {
        "status": "complete",
        "panels": panels,
        "maps": maps,
        "source_mode": source_mode,
        "fit_fingerprint": fp,
        "method": (
            f"K5; source mode {source_mode}; "
            "source-only GCV ridge; global conversation folds; "
            "target-training-only bias and scalar calibration"
        ),
        "script_sha256": base.sha(__file__),
        "metadata": base.as_metadata_dict(base.git_provenance(cwd=REPO), phase="collect_subset"),
    }
    base.atomic_json(out / "results.json", result)
    return result


def main():
    """Dispatch a bounded pilot, resumable full fit, or verified collection."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["fit", "collect"], required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--model", choices=base.MODELS)
    parser.add_argument("--source-mode", choices=["chat_grid", "plain_only"], default="chat_grid")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.stage == "fit":
        if args.model is None:
            parser.error("--model is required for fit")
        fit(args.out, args.inputs, args.model, args.limit, args.source_mode)
    else:
        collect(args.out, args.source_mode)
    print("[phase=done] requested subset stage complete", flush=True)


if __name__ == "__main__":
    main()
