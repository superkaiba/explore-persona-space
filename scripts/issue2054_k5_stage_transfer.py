"""Frozen Base→Instruct transfer on the paper's existing Qwen five-answer banks.

Restore banked estimators without selecting new penalties or generating data.
The primary target cohort and five global conversation folds are unchanged.
"""

# ruff: noqa: E402
from __future__ import annotations

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import argparse
import hashlib
import inspect
import json
from pathlib import Path
import re
import subprocess
import sys
import time

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from explore_persona_space.analysis.mapping_baselines import knn_retrieval
from scripts import issue2054_k3 as k3
from scripts import issue2054_k3_fit as fit
from scripts import issue2054_k5_assistant_story_analysis as original
from scripts import issue2054_k5_map_geometry as geometry

LABELS = ("Chat", "Assistant-story", "HELIOS", "Wren", "Dana", "Vex")
PREFIXES = {
    "Chat": original.CHAT,
    "Assistant-story": original.STORY,
    **{name: f"char_{name.lower()}__on_policy__attrib_quoted" for name in LABELS[2:]},
}
MODES = ("frozen", "target_own", "bias_only", "identity_source", "identity_target")
METHOD = {
    "version": 1,
    "source_model": "qwen2.5-7b",
    "target_model": "qwen2.5-7b-instruct",
    "answer_target": "own-policy mean of five existing answer activations",
    "folds": "five original global conversation folds; each original cohort retained",
    "fit": "float64 standardized ridge at original selected penalty; raw x @ A + b",
    "frozen": "apply source A and b directly to raw target context activations",
    "bias_only": "frozen prediction + mean(target_train_Y - frozen(target_train_X))",
    "identity_source": "target_X + mean(source_train_Y - source_train_X)",
    "identity_target": "target_X + mean(target_train_Y - target_train_X)",
    "r2": "1 - heldout SSE / heldout SST, centered on the target test-fold mean",
    "summary": "equal mean of the five fold scores; retention = mean frozen R2 / mean own R2",
    "retrieval": "cosine and Euclidean tolerance mid-ranks in the full target test fold",
    "ks": [1, 5, 10],
    "paired_sensitivity": "same trained maps; target test rows shared with source, own pool",
    "prediction_storage": "float32; metrics and per-query SSE/SST computed in float64",
    "map_storage": "deterministic restoration recipe and float64 hashes; no dense maps",
}
tensor = original.tensor
score_batch = original.score_batch
digest = original.digest


def progress(root, phase, completed_units, **fields):
    """Atomically expose the current stage to the external monitor."""
    record = {
        "checked_at": time.time(),
        "phase": phase,
        "completed_units": completed_units,
        **fields,
    }
    k3.atomic_json(root / "progress.json", record)
    print(json.dumps(record), flush=True)


def source_files():
    """Locate the actual imported files, including editable shared-root packages."""
    scripts = (__file__, original.__file__, geometry.__file__, k3.__file__, fit.__file__)
    return {
        **{f"scripts/{Path(path).name}": Path(path).resolve() for path in scripts},
        "src/explore_persona_space/analysis/mapping_baselines.py": Path(
            inspect.getfile(knn_retrieval)
        ).resolve(),
        "src/explore_persona_space/orchestrate/env.py": Path(
            inspect.getfile(load_dotenv)
        ).resolve(),
    }


def source_hashes(source_sha):
    """Bind every numerical/persistence helper to verified committed bytes."""
    if re.fullmatch(r"[0-9a-f]{40}", source_sha) is None:
        raise ValueError("source-sha must be a full lowercase commit SHA")
    result = {}
    for relative, path in sorted(source_files().items()):
        committed = subprocess.check_output(["git", "show", f"{source_sha}:{relative}"], cwd=REPO)
        actual = k3.sha(path)
        if hashlib.sha256(committed).hexdigest() != actual:
            raise RuntimeError(f"source differs from committed source-sha: {relative}")
        result[relative] = actual
    return result


def fingerprint(manifest, source_sha, sources):
    """Bind resume units to all inputs, reference scores, methods, and code."""
    return digest(
        {
            "manifest": manifest,
            "source_sha": source_sha,
            "sources": sources,
            "method": METHOD,
            "numpy_version": np.__version__,
            "torch_version": torch.__version__,
        }
    )


def validate_manifest(manifest):
    """Require the complete requested six-setting, two-stage paper roster."""
    settings, folds = manifest["settings"], manifest["fold_map"]
    if len(settings) != len(LABELS) or {s["label"] for s in settings} != set(LABELS):
        raise ValueError("manifest must contain exactly the six requested settings")
    if not folds or any(type(f) is not int or f not in range(5) for f in folds.values()):
        raise ValueError("invalid global five-fold map")
    if re.fullmatch(r"[0-9a-f]{64}", manifest["fold_sha256"]) is None:
        raise ValueError("missing original fold-map SHA256")
    for setting in settings:
        for side in ("source", "target"):
            ref = setting[side]
            expected = PREFIXES[setting["label"]] + "__" + METHOD[f"{side}_model"]
            if ref["cell"] != expected:
                raise ValueError(f"incorrect setting/stage cell: {ref['cell']}")
            if re.fullmatch(r"[0-9a-f]{64}", ref["sha256"]) is None:
                raise ValueError("invalid bank SHA256")
            if re.fullmatch(r"[0-9a-f]{40}", ref["revision"]) is None or not ref["path"]:
                raise ValueError("bank must name an immutable remote revision and path")
            references = ref["own_folds"]
            if len(references) != 5 or {r["fold"] for r in references} != set(range(5)):
                raise ValueError("own-map references must contain each of five folds once")
            for row in references:
                penalty = row["ridge"]["best_lambda"]
                if not np.isfinite(penalty) or penalty <= 0:
                    raise ValueError("invalid banked ridge penalty")
                if row["n_train"] <= 0 or row["n_test"] < max(METHOD["ks"]):
                    raise ValueError("invalid own-map reference population")
                expected_metric = row["metrics"]["own"]
                if not np.isfinite(expected_metric["r2"]):
                    raise ValueError("invalid reference own-map R2")
                for metric in ("euclidean", "cosine"):
                    accuracy = expected_metric["retrieval"][metric]["acc_at_k"]["1"]
                    if not 0 <= accuracy <= 1:
                        raise ValueError("invalid reference retrieval accuracy")


def load_bank(ref, fold_map, dimension=3584):
    """Verify an immutable complete-five bank without changing its cohort."""
    if k3.sha(ref["local"]) != ref["sha256"]:
        raise RuntimeError(f"bank content changed: {ref['cell']}")
    with np.load(ref["local"], allow_pickle=False) as z:
        ids = list(map(str, z["conv_id"]))
        x, y = tensor(z["v_C"]).numpy(), tensor(z["v_A"]).numpy()
        caps = z["cap_mask"]
    if x.shape != y.shape or x.shape != (len(ids), dimension) or len(set(ids)) != len(ids):
        raise ValueError(f"invalid activation layout: {ref['cell']}")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError(f"nonfinite activation bank: {ref['cell']}")
    if caps.shape != (len(ids), 5) or not np.isin(caps, [0, 1]).all():
        raise ValueError(f"invalid complete-five cap mask: {ref['cell']}")
    membership = np.array([fold_map[cid] for cid in ids])
    if set(membership) != set(range(5)):
        raise ValueError("missing global conversation folds")
    panel = {"x": x, "y": y, "ids": ids, "membership": membership, "caps": caps}
    for row in ref["own_folds"]:
        test = membership == row["fold"]
        if int(test.sum()) != row["n_test"] or int((~test).sum()) != row["n_train"]:
            raise RuntimeError(f"original train/test population changed: {ref['cell']}")
        for part, mask in (("train", ~test), ("test", test)):
            key = f"{part}_ids_sha256"
            if key in row and digest(np.asarray(ids)[mask].tolist()) != row[key]:
                raise RuntimeError(f"original {part} conversation IDs changed: {ref['cell']}")
    return panel


def fold_audit(source, target, fold):
    """Prove both-stage held-out exclusions and describe unmatched cohorts."""
    ids = {}
    for name, panel in (("source", source), ("target", target)):
        membership = panel["membership"]
        ids[f"{name}_train"] = np.asarray(panel["ids"])[membership != fold].tolist()
        ids[f"{name}_test"] = np.asarray(panel["ids"])[membership == fold].tolist()
    if set(ids["source_train"]) & set(ids["target_test"]):
        raise RuntimeError("source training contains a target test conversation")
    if set(ids["target_train"]) & set(ids["source_test"]):
        raise RuntimeError("target training contains a source test conversation")
    common = set(source["ids"]) & set(target["ids"])
    paired = np.array([cid in common for cid in ids["target_test"]])
    audit = {
        **{f"{key}_n": len(value) for key, value in ids.items()},
        **{f"{key}_ids_sha256": digest(value) for key, value in ids.items()},
        "cohort_intersection_n": len(common),
        "source_only_n": len(set(source["ids"]) - common),
        "target_only_n": len(set(target["ids"]) - common),
        "paired_target_test_n": int(paired.sum()),
        "unpaired_target_test_n": int((~paired).sum()),
        "source_train_target_test_overlap": 0,
        "target_train_source_test_overlap": 0,
    }
    return audit, paired


def restore_maps(bank, fold, lambdas):
    """Restore only the two original estimators in one batched Cholesky solve."""
    if (
        len(bank) != 2
        or len(lambdas) != 2
        or any(not np.isfinite(lam) or lam <= 0 for lam in lambdas)
    ):
        raise ValueError("restoration requires two positive finite banked penalties")
    train = {
        key: torch.stack([m[key].sum(0) - m[key][fold] for m in bank])
        for key in ("n", "sx", "sy", "xx", "xy")
    }
    n = train["n"]
    mu_x, mu_y = train["sx"] / n[:, None], train["sy"] / n[:, None]
    variance = train["xx"].diagonal(dim1=-2, dim2=-1) / n[:, None] - mu_x.square()
    if bool((variance < -1e-9).any()) or bool((n <= 0).any()):
        raise ValueError("invalid training moments")
    sd = variance.clamp_min(0).sqrt() + 1e-9
    covariance = train["xx"] - n[:, None, None] * mu_x[:, :, None] * mu_x[:, None, :]
    covariance /= sd[:, :, None] * sd[:, None, :]
    cross = (train["xy"] - mu_x[:, :, None] * train["sy"][:, None, :]) / sd[:, :, None]
    covariance.diagonal(dim1=-2, dim2=-1).add_(torch.tensor(lambdas, dtype=torch.float64)[:, None])
    weights = torch.cholesky_solve(cross, torch.linalg.cholesky(covariance))
    raw = weights / sd[:, :, None]
    bias = mu_y - torch.einsum("bi,bij->bj", mu_x, raw)
    if not bool(torch.isfinite(raw).all()) or not bool(torch.isfinite(bias).all()):
        raise RuntimeError("nonfinite restored estimator")
    return raw, bias, mu_y - mu_x, n, mu_x, mu_y


def target_predictions(x, maps, biases, identity, means_x, means_y):
    """Return all target predictions; the bias correction uses training means only."""
    frozen = x @ maps[0] + biases[0]
    correction = means_y[1] - means_x[1] @ maps[0] - biases[0]
    return torch.stack(
        [frozen, x @ maps[1] + biases[1], frozen + correction, x + identity[0], x + identity[1]]
    ), correction


def evaluate_fold(source, target, bank, fold, refs, announce):
    """Compute one transfer unit only after reproducing both published own-map scores."""
    audit, paired = fold_audit(source, target, fold)
    penalties = [row["ridge"]["best_lambda"] for row in refs]
    announce("restore_maps")
    maps, biases, identity, counts, means_x, means_y = restore_maps(bank, fold, penalties)
    if counts.tolist() != [row["n_train"] for row in refs]:
        raise RuntimeError("restored training sizes differ from original own estimators")
    announce("source_own_parity")
    smask, tmask = source["membership"] == fold, target["membership"] == fold
    source_prediction = tensor(source["x"][smask]) @ maps[0] + biases[0]
    source_metric = score_batch(source_prediction, source["y"][smask])[0]
    source_delta = original.original_parity(
        source_metric, refs[0]["metrics"]["own"], "source", fold
    )
    del source_prediction
    announce("target_metrics")
    x, y = tensor(target["x"][tmask]), tensor(target["y"][tmask])
    predictions, correction = target_predictions(x, maps, biases, identity, means_x, means_y)
    metrics = dict(zip(MODES, score_batch(predictions, y), strict=True))
    target_delta = original.original_parity(
        metrics["target_own"], refs[1]["metrics"]["own"], "target", fold
    )
    paired_scores = None
    if audit["unpaired_target_test_n"] and int(paired.sum()) >= max(METHOD["ks"]):
        announce("paired_sensitivity")
        paired_scores = dict(
            zip(MODES, score_batch(predictions[:, paired], y[paired]), strict=True)
        )
    record = {
        "fold": fold,
        "coverage": audit,
        "ridge": {side: refs[i]["ridge"] for i, side in enumerate(("source", "target"))},
        "metrics": metrics,
        "source_own": source_metric,
        "own_parity_r2_absolute_delta": {"source": source_delta, "target": target_delta},
        "paired_sensitivity": {
            "n_test": int(paired.sum()),
            "status": "identical_to_primary"
            if bool(paired.all())
            else ("computed" if paired_scores is not None else "insufficient_pool"),
            "metrics": metrics if bool(paired.all()) else paired_scores,
        },
        "map_sha256_float64": {
            side: hashlib.sha256(maps[i].contiguous().numpy().tobytes()).hexdigest()
            for i, side in enumerate(("source", "target"))
        },
    }
    arrays = {
        "test_conv_id": np.asarray(target["ids"])[tmask],
        "source_test_conv_id": np.asarray(source["ids"])[smask],
        "paired_target_mask": paired,
        "frozen_prediction": predictions[0].float().numpy(),
        "target_own_prediction": predictions[1].float().numpy(),
        "bias_correction": correction.numpy(),
        "source_bias": biases[0].numpy(),
        "target_bias": biases[1].numpy(),
        "source_identity_bias": identity[0].numpy(),
        "target_identity_bias": identity[1].numpy(),
        "squared_errors": (predictions - y).square().sum(2).numpy(),
        "sst_contributions": (y - y.mean(0)).square().sum(1).numpy(),
        "error_modes": np.asarray(MODES),
    }
    return record, arrays


def save_packet(root, label, fold, record, arrays, fp):
    """Publish an atomic array packet, then metadata that commits its hash."""
    stem = root / "folds" / f"{label}__fold{fold}"
    path = stem.with_suffix(".npz")
    fit.save_npz(path, arrays)
    record = {
        **record,
        "label": label,
        "fingerprint": fp,
        "array_path": str(path.relative_to(root)),
        "array_sha256": k3.sha(path),
        "array_bytes": path.stat().st_size,
    }
    k3.atomic_json(stem.with_suffix(".json"), record)
    return record


def resume_packet(root, label, fold, fp):
    """Accept a completed fold only when input fingerprint and packet bytes match."""
    meta = root / "folds" / f"{label}__fold{fold}.json"
    if not meta.exists():
        return None
    record = json.loads(meta.read_text())
    if record["fingerprint"] != fp or record["label"] != label or record["fold"] != fold:
        raise RuntimeError(f"resume fingerprint or fold identity changed: {meta}")
    path = root / record["array_path"]
    if not path.resolve().is_relative_to(root.resolve()):
        raise RuntimeError("resume array path escaped output root")
    if not path.is_file() or k3.sha(path) != record["array_sha256"]:
        raise RuntimeError(f"resume array content changed: {path}")
    if path.stat().st_size != record["array_bytes"]:
        raise RuntimeError("resume array size changed")
    return record


def mean_metrics(values):
    """Use equal fold weights, as in the paper, with explicit pool ranges."""
    pools = [v["retrieval_pool"] for v in values]
    return {
        "r2": float(np.mean([v["r2"] for v in values])),
        "folds": len(values),
        "retrieval_pool_min": min(pools),
        "retrieval_pool_max": max(pools),
        "retrieval": {
            metric: {
                "acc_at_k": {
                    str(k): float(
                        np.mean([v["retrieval"][metric]["acc_at_k"][str(k)] for v in values])
                    )
                    for k in METHOD["ks"]
                },
                "chance_at_k": {
                    str(k): float(np.mean([k / n for n in pools])) for k in METHOD["ks"]
                },
                "mrr": float(np.mean([v["retrieval"][metric]["mrr"] for v in values])),
            }
            for metric in ("euclidean", "cosine")
        },
    }


def summarize(rows):
    """Summarize only complete settings, never filling missing folds with zeros."""
    result = []
    for label in LABELS:
        selected = sorted([r for r in rows if r["label"] == label], key=lambda r: r["fold"])
        if not selected:
            continue
        if len(selected) != 5 or [r["fold"] for r in selected] != list(range(5)):
            continue
        metrics = {mode: mean_metrics([r["metrics"][mode] for r in selected]) for mode in MODES}
        own = metrics["target_own"]["r2"]
        if abs(own) <= 1e-12:
            raise RuntimeError("undefined retention: mean target own R2 is zero")
        paired = [r["paired_sensitivity"]["metrics"] for r in selected]
        result.append(
            {
                "label": label,
                "metrics": metrics,
                "source_own": mean_metrics([r["source_own"] for r in selected]),
                "frozen_retention": metrics["frozen"]["r2"] / own,
                "bias_only_retention": metrics["bias_only"]["r2"] / own,
                "target_n": sum(r["coverage"]["target_test_n"] for r in selected),
                "source_n": sum(r["coverage"]["source_test_n"] for r in selected),
                "paired_target_n": sum(r["coverage"]["paired_target_test_n"] for r in selected),
                "paired_sensitivity": (
                    {mode: mean_metrics([p[mode] for p in paired]) for mode in MODES}
                    if all(p is not None for p in paired)
                    else None
                ),
            }
        )
    return result


def run(manifest_path, out_root, source_sha, limit_units=None):
    """Restore, score, and checkpoint each requested stage-transfer fold."""
    if limit_units is not None and not 1 <= limit_units <= 30:
        raise ValueError("limit-units must be between 1 and 30")
    manifest = json.loads(manifest_path.read_text())
    validate_manifest(manifest)
    sources = source_hashes(source_sha)
    fp = fingerprint(manifest, source_sha, sources)
    out_root.mkdir(parents=True, exist_ok=True)
    identity_path = out_root / "run_identity.json"
    identity = {
        "fingerprint": fp,
        "source_sha": source_sha,
        "sources": sources,
        "manifest": manifest,
        "method": METHOD,
    }
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        raise RuntimeError("output root belongs to a different analysis fingerprint")
    k3.atomic_json(identity_path, identity)
    rows = []
    for label in LABELS:
        for fold in range(5):
            resumed = resume_packet(out_root, label, fold, fp)
            if resumed is not None:
                rows.append(resumed)
    progress(out_root, "validate_input_hashes", len(rows))
    # Recheck all input bytes even when all fold packets already exist.
    for setting in manifest["settings"]:
        for side in ("source", "target"):
            ref = setting[side]
            if k3.sha(ref["local"]) != ref["sha256"]:
                raise RuntimeError(f"input bank changed before resume: {ref['cell']}")
    completed = {(r["label"], r["fold"]) for r in rows}
    stop_at = 30 if limit_units is None else limit_units
    settings = {s["label"]: s for s in manifest["settings"]}
    for label in LABELS:
        pending = [fold for fold in range(5) if (label, fold) not in completed]
        if not pending or len(rows) >= stop_at:
            continue
        setting = settings[label]
        panels, bank = [], []
        for side in ("source", "target"):
            progress(out_root, f"load_{side}", len(rows), setting=label)
            panel = load_bank(setting[side], manifest["fold_map"])
            panels.append(panel)
            progress(out_root, f"moments_{side}", len(rows), setting=label)
            bank.append(geometry.moments_by_fold(panel))
        references = [
            {r["fold"]: r for r in setting[side]["own_folds"]} for side in ("source", "target")
        ]
        for fold in pending:
            if len(rows) >= stop_at:
                break
            started = time.monotonic()

            def announce(phase):
                progress(out_root, phase, len(rows), setting=label, fold=fold)

            record, arrays = evaluate_fold(
                *panels, bank, fold, [ref[fold] for ref in references], announce
            )
            record.update(
                source_cell=setting["source"]["cell"],
                target_cell=setting["target"]["cell"],
                source_sha=source_sha,
            )
            announce("persist_fold")
            saved = save_packet(out_root, label, fold, record, arrays, fp)
            rows.append(saved)
            del arrays
            progress(
                out_root,
                "fold_complete",
                len(rows),
                setting=label,
                fold=fold,
                seconds=time.monotonic() - started,
            )
        del bank, panels, panel
    rows.sort(key=lambda r: (LABELS.index(r["label"]), r["fold"]))
    status = "complete" if len(rows) == 30 else "partial_pilot"
    result = {
        "source_sha": source_sha,
        "status": status,
        "fingerprint": fp,
        "rows": rows,
        "summary": summarize(rows),
        "method": METHOD,
        "coverage": {
            "planned_units": 30,
            "realized_units": len(rows),
            "planned_settings": list(LABELS),
        },
        "manifest_sha256": k3.sha(manifest_path),
        "fold_map_content_sha256": digest(manifest["fold_map"]),
        "completed_at": time.time(),
    }
    if status == "complete" and len(result["summary"]) != len(LABELS):
        raise RuntimeError("incomplete summary despite complete fold count")
    k3.atomic_json(out_root / "results.json", result)
    progress(out_root, status, len(rows))
    return result


def main():
    """CLI for monitored analysis using an externally assembled immutable manifest."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--limit-units", type=int)
    args = parser.parse_args()
    run(args.manifest, args.out_root, args.source_sha, args.limit_units)


if __name__ == "__main__":
    main()
