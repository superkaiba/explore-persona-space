"""Audited K5 assistant-story transfer and paired answer comparisons.

Reads the producer's verified generation_complete.json and publishes results
under analysis/. All maps exclude the same global conversation fold. Existing
maps retain their published lambdas; only the new story map selects a lambda.
"""

# ruff: noqa: E402
from __future__ import annotations

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import argparse
from concurrent.futures import ThreadPoolExecutor
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
from explore_persona_space.orchestrate.hub import stage_hub_file
from explore_persona_space.orchestrate.provenance import as_metadata_dict, git_provenance
from scripts import issue2054_k3 as k3
from scripts import issue2054_k3_artifacts as artifacts
from scripts import issue2054_k3_recover as recovery
from scripts import issue2054_k5 as k5
from scripts import issue2054_k5_loso_calibration as base
from scripts import issue2054_k5_map_geometry as geometry
from scripts.issue2054_ctx2ctx_fit import DEFAULT_LAMBDAS, GCV_DOF_CAP
from scripts.issue2054_pool_specialize import PooledMomentRidge

STORY = "conversation_paired_stories_assistant__on_policy__attrib_quoted"
CHAT = "conversation_paired_stories_assistant__on_policy__chat"
OLD_K5_PREFIX = "issue2054_section44_k5_gcp/production_v1"
# SHA256 of git show 91e968:scripts/issue2054_k5.py, the published producer.
OLD_K5_SOURCE_SHA = "10bc6c886b7d841092c93d70bb000e08b4203dcd4742633531feefb9c056b35a"
LABELS = ["Chat", "Plain", "HELIOS", "Wren", "Dana", "Vex", "Assistant-story"]
KS = (1, 5, 10)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def tensor(value):
    """Lossless conversion without the VM's slow whole-array NumPy astype path."""
    return torch.as_tensor(value).to(dtype=torch.float64)


def progress(root, phase, **fields):
    record = {"checked_at": time.time(), "phase": phase, **fields}
    k3.atomic_json(root / "analysis" / "progress.json", record)
    print(json.dumps(record), flush=True)


def fingerprint(manifest, generation):
    files = [
        __file__,
        k3.__file__,
        k5.__file__,
        recovery.__file__,
        artifacts.__file__,
        base.__file__,
        geometry.__file__,
        inspect.getfile(PooledMomentRidge),
        inspect.getfile(knn_retrieval),
    ]
    return digest(
        {
            "sources": {Path(p).name: k3.sha(p) for p in files},
            "manifest": manifest,
            "generation": {
                key: generation[key]
                for key in (
                    "source_sha",
                    "manifest_sha256",
                    "inventory_sha256",
                    "selected_cells",
                    "parent_revision",
                )
            },
            "old_k5_source": OLD_K5_SOURCE_SHA,
            "old_k5_revision": base.K5_REV,
            "reference_sha": geometry.REFERENCE_SHA,
            "bootstrap_draws": 200,
        }
    )


def legacy_k5_fingerprint(manifest, cell):
    """Validate old receipts against the original producer, not edited K5 code."""
    return digest(
        {
            "parent_revision": k5.PARENT_REV,
            "parent_capture": recovery.capture_fingerprint(manifest, cell, False, k5.policy()),
            "new_source": OLD_K5_SOURCE_SHA,
            "new_draws": (3, 4),
            "counts": (1, 3, 5),
            "pool": "six displayed settings separately within each model",
        }
    )


def receipt(path, expected_fp=None, remote=None):
    done = path.with_suffix(path.suffix + ".done.json")
    record = json.loads(done.read_text())
    if remote is not None and record["path"] != remote:
        raise RuntimeError(f"receipt destination mismatch: {path}")
    if expected_fp is not None and record["fingerprint"] != expected_fp:
        raise RuntimeError(f"receipt fingerprint mismatch: {path}")
    if not path.is_file() or k3.sha(path) != record["sha256"]:
        raise RuntimeError(f"receipt content mismatch: {path}")
    return record


def verify_inventory(root, path, fp):
    """Audit every resumed published artifact rather than trusting a sentinel."""
    receipt(path, fp)
    inventory = json.loads(path.read_text())
    if inventory["analysis_fingerprint"] != fp or not inventory["files"]:
        raise RuntimeError("invalid completed analysis inventory")
    for item in inventory["files"]:
        local = root / item["path"]
        done = root / item["receipt_path"]
        if (
            local.resolve().is_relative_to(root.resolve()) is False
            or done.resolve().is_relative_to(root.resolve()) is False
        ):
            raise RuntimeError("analysis inventory path escaped output root")
        actual = receipt(local, fp)
        if (
            actual != item["receipt"]
            or k3.sha(done) != item["receipt_sha256"]
            or k3.sha(local) != item["sha256"]
            or local.stat().st_size != item["size"]
        ):
            raise RuntimeError(f"analysis inventory mismatch: {local}")
    return inventory


def fetch_receipted(root, remote, revision, expected_fp):
    path = root / "inputs" / remote
    done = path.with_suffix(path.suffix + ".done.json")
    stage_hub_file(k3.HF_REPO, remote + ".done.json", done, revision=revision)
    record = json.loads(done.read_text())
    if record["path"] != remote or record["fingerprint"] != expected_fp:
        raise RuntimeError(f"immutable receipt identity mismatch: {remote}")
    stage_hub_file(k3.HF_REPO, remote, path, revision=revision, size_bytes=record["size"])
    receipt(path, expected_fp, remote)
    return path


def fetch_raw(root, remote, revision, fp):
    path = fetch_receipted(root, remote, revision, fp)
    index = json.loads(path.read_text())
    for name in index["shards"]:
        if Path(name).name != name:
            raise ValueError("raw shard name is not a basename")
        fetch_receipted(root, str(Path(remote).parent / name), revision, fp)
    return path


def save_packet(root, stem, arrays, record, fp):
    """Publish arrays before metadata, with hashes binding every resumable unit."""
    path = root / "analysis" / stem
    cp = Path(str(path) + ".npz")
    meta = Path(str(path) + ".json")
    base.save_npz(cp, arrays)
    record = {
        **record,
        "array_path": str(cp.relative_to(root)),
        "array_sha256": k3.sha(cp),
        "analysis_fingerprint": fp,
    }
    k3.atomic_json(meta, record)
    artifacts.seal_many([cp, meta], root, fp)
    return record


def resume_packet(root, stem, fp):
    meta = Path(str(root / "analysis" / stem) + ".json")
    done = meta.with_suffix(meta.suffix + ".done.json")
    if not done.exists():
        return None
    receipt(meta, fp)
    record = json.loads(meta.read_text())
    cp = root / record["array_path"]
    receipt(cp, fp)
    if record["analysis_fingerprint"] != fp or k3.sha(cp) != record["array_sha256"]:
        raise RuntimeError(f"analysis checkpoint changed: {meta}")
    return record


def train_moments(moment, folds):
    result = {key: value.sum(0)[None] - value[folds] for key, value in moment.items()}
    return result


def gcv_maps(moment, folds, lambdas=DEFAULT_LAMBDAS, dof_cap=GCV_DOF_CAP):
    """Batched PooledMomentRidge equations, including its exact GCV df cap."""
    m = train_moments(moment, folds)
    n = m["n"]
    d = m["xx"].shape[-1]
    if bool((n <= d).any()) or dof_cap is None:
        raise ValueError("new own-map fits require ambient training and a df cap")
    mu_x, mu_y = m["sx"] / n[:, None], m["sy"] / n[:, None]
    sd = (m["xx"].diagonal(dim1=-2, dim2=-1) / n[:, None] - mu_x.square()).clamp_min(
        0
    ).sqrt() + 1e-9
    cov = m["xx"] - n[:, None, None] * mu_x[:, :, None] * mu_x[:, None, :]
    cov = cov / (sd[:, :, None] * sd[:, None, :])
    eigenvalues, eigenvectors = torch.linalg.eigh(cov)
    eigenvalues = eigenvalues.clamp_min(0)
    cross = (m["xy"] - mu_x[:, :, None] * m["sy"][:, None, :]) / sd[:, :, None]
    projected = eigenvectors.transpose(1, 2) @ cross
    energy = projected.square().sum(2)
    energy = torch.where(eigenvalues > 1e-12, energy / eigenvalues, torch.zeros_like(energy))
    total = m["yss"] - n * mu_y.square().sum(1)
    grid = tensor(lambdas)
    filt = eigenvalues[:, None, :] / (eigenvalues[:, None, :] + grid[None, :, None])
    dof = filt.sum(2)
    rss = total[:, None] - ((2 * filt - filt.square()) * energy[:, None, :]).sum(2)
    denominator = (n[:, None] - dof).square()
    valid = (denominator > 1e-12) & (dof <= dof_cap * n[:, None])
    if not bool(valid.any(1).all()):
        raise RuntimeError("all ridge penalties violate the GCV df cap")
    gcv = torch.where(valid, rss / denominator, torch.full_like(rss, torch.inf))
    chosen = gcv.argmin(1)
    best = grid[chosen]
    weights = eigenvectors @ (projected / (eigenvalues + best[:, None])[:, :, None])
    raw = weights / sd[:, :, None]
    bias = mu_y - torch.einsum("bi,bij->bj", mu_x, raw)
    infos = [
        {
            "fold": f,
            "best_lambda": float(best[i]),
            "dof": float(dof[i, chosen[i]]),
            "n_train": int(n[i]),
            "d_fit": d,
            "selector": f"gcv_dof_cap_{dof_cap}",
            "side": "cov-moments-batched",
        }
        for i, f in enumerate(folds)
    ]
    return raw, bias, mu_y - mu_x, infos


def score_batch(predictions, target, ks=KS, block=128):
    """Parent-equivalent metrics; block query rows to bound retrieval memory."""
    predictions, target = tensor(predictions), tensor(target)
    if predictions.ndim == 2:
        predictions = predictions[None]
    if predictions.shape[1:] != target.shape or not bool(torch.isfinite(predictions).all()):
        raise ValueError("invalid prediction shape or values")
    n = len(target)
    sst = (target - target.mean(0)).square().sum()
    if n < max(ks) or not bool(torch.isfinite(target).all()) or not sst > 0:
        raise ValueError("undefined held-out metrics or insufficient retrieval pool")
    r2 = 1 - (predictions - target).square().sum((1, 2)) / sst
    rank_parts = {metric: [] for metric in ("euclidean", "cosine")}
    q2 = target.square().sum(1)
    qn = target / (q2.sqrt()[:, None] + 1e-12)
    for start in range(0, n, block):
        stop = min(n, start + block)
        p = predictions[:, start:stop]
        p2 = p.square().sum(2)
        for metric in rank_parts:
            if metric == "euclidean":
                distances = p2[:, :, None] + q2 - 2 * (p @ target.T)
            else:
                distances = 1 - (p / (p2.sqrt()[:, :, None] + 1e-12)) @ qn.T
            ids = torch.arange(start, stop)
            actual = distances[:, torch.arange(stop - start), ids][:, :, None]
            tolerance = 1e-9 * actual.abs().clamp_min(1e-12)
            closer = (distances < actual - tolerance).sum(2)
            tied = ((distances - actual).abs() <= tolerance).sum(2) - 1
            rank_parts[metric].append(1 + closer.double() + 0.5 * tied.double())
    ranks = {metric: torch.cat(parts, 1) for metric, parts in rank_parts.items()}
    result = []
    for i in range(len(predictions)):
        result.append(
            {
                "r2": float(r2[i]),
                "retrieval_pool": n,
                "chance_top1": 1 / n,
                "retrieval": {
                    metric: {
                        "metric": metric,
                        "n": n,
                        "n_pool": n,
                        "acc_at_k": {str(k): float((rank[i] <= k).double().mean()) for k in ks},
                        "chance_at_k": {str(k): k / n for k in ks},
                        "median_rank": float(torch.quantile(rank[i].double(), 0.5)),
                        "mrr": float((1 / rank[i]).mean()),
                    }
                    for metric, rank in ranks.items()
                },
            }
        )
    return result


def original_parity(metric, expected, cell, fold):
    delta = abs(metric["r2"] - expected["r2"])
    if delta > 1e-6:
        raise RuntimeError(f"published own R2 changed: {cell} fold {fold}: {delta}")
    for kind in ("euclidean", "cosine"):
        if (
            abs(
                metric["retrieval"][kind]["acc_at_k"]["1"]
                - expected["retrieval"][kind]["acc_at_k"]["1"]
            )
            > 1e-12
        ):
            raise RuntimeError(f"published own retrieval changed: {cell} fold {fold}")
    return delta


def stage_banks(root):
    refs = base.references()["panels"]

    def fetch(ref):
        hashes = {f["input_sha256"] for f in ref["folds"]}
        if len(hashes) != 1:
            raise ValueError("parent input identity differs across folds")
        remote = f"{OLD_K5_PREFIX}/k5/{ref['cell']}.npz"
        path = stage_hub_file(k3.HF_REPO, remote, root / "inputs" / remote, revision=base.K5_REV)
        expected = hashes.pop()
        if k3.sha(path) != expected:
            raise RuntimeError(f"published K5 bank changed: {remote}")
        return ref["cell"], {
            "path": remote,
            "local": str(path),
            "sha256": expected,
            "revision": base.K5_REV,
        }

    with ThreadPoolExecutor(max_workers=4) as pool:
        return dict(pool.map(fetch, refs))


def load_banks(root, model, sources, folds, story_fp):
    panel = {}
    cells = [f"{prefix}__{model}" for _, prefix in base.SETTINGS] + [f"{STORY}__{model}"]
    for cell in cells:
        path = (
            root / "k5" / f"{cell}.npz"
            if cell.startswith(STORY + "__")
            else Path(sources[cell]["local"])
        )
        if cell.startswith(STORY + "__"):
            receipt(path, story_fp[cell])
        with np.load(path, allow_pickle=False) as z:
            ids = list(map(str, z["conv_id"]))
            x, y = tensor(z["v_C"]).numpy(), tensor(z["v_A"]).numpy()
            caps = z["cap_mask"]
        if x.shape != y.shape or x.shape != (len(ids), 3584) or len(set(ids)) != len(ids):
            raise ValueError(f"invalid activation layout: {cell}")
        if not np.isfinite(x).all() or not np.isfinite(y).all() or caps.shape != (len(ids), 5):
            raise ValueError(f"invalid complete-five bank: {cell}")
        membership = np.array([folds[cid] for cid in ids])
        if set(membership) != set(range(5)):
            raise ValueError("missing global conversation folds")
        panel[cell] = {"x": x, "y": y, "ids": ids, "membership": membership, "caps": caps}
    return panel


def stage_original_text(root, records):
    def fetch(record):
        path = stage_hub_file(
            k3.HF_REPO, record["raw"], root / "inputs" / record["raw"], revision=k3.REVISION
        )
        if k3.sha(path) != record["raw_sha256"]:
            raise RuntimeError("original response text changed")
        rows = k3.read_rows(path)
        if len(rows) != record["n"]:
            raise RuntimeError("original row population changed")
        return record["cell"], {r["conv_id"]: r for r in rows}

    with ThreadPoolExecutor(max_workers=4) as pool:
        return dict(pool.map(fetch, records))


def query_matches(original, model):
    """Audit full canonical question presence without claiming equal histories."""
    chat = original[f"{CHAT}__{model}"]
    queries = {}
    for cid, row in chat.items():
        prefix = row["final_text"][: row["answer_start"]]
        start, end = "<|im_start|>user\n", "<|im_end|>\n<|im_start|>assistant\n"
        if not prefix.startswith(start) or not prefix.endswith(end) or prefix.count(start) != 1:
            raise RuntimeError(f"chat source is not the audited single user turn: {cid}")
        queries[cid] = prefix[len(start) : -len(end)]
        if not queries[cid]:
            raise ValueError("empty canonical query")
    valid, audit = {}, []
    for cell, rows in original.items():
        if not cell.endswith("__" + model):
            continue
        exact, whitespace, mismatch = [], [], []
        for cid in sorted(rows.keys() & queries.keys()):
            prefix = rows[cid]["final_text"][: rows[cid]["answer_start"]]
            if queries[cid] in prefix:
                exact.append(cid)
            elif " ".join(queries[cid].split()) in " ".join(prefix.split()):
                whitespace.append(cid)
            else:
                mismatch.append(cid)
        valid[cell] = set(exact + whitespace)
        audit.append(
            {
                "cell": cell,
                "n_original": len(rows),
                "n_shared_id": len(exact) + len(whitespace) + len(mismatch),
                "exact_query": len(exact),
                "whitespace_query": len(whitespace),
                "mismatch_ids": mismatch,
                "missing_chat_ids": sorted(rows.keys() - queries.keys()),
            }
        )
    return queries, valid, audit


def new_maps(root, model, panel, fp):
    p = panel[f"{STORY}__{model}"]
    saved = [resume_packet(root, f"maps/{model}__fold{f}", fp) for f in range(5)]
    pending = [f for f, row in enumerate(saved) if row is None]
    if not pending:
        return saved
    moment = geometry.moments_by_fold(p)
    moment["yss"] = torch.stack(
        [tensor(p["y"][p["membership"] == f]).square().sum() for f in range(5)]
    )
    # Full-size first-fold timing and independent oracle precede the remaining batch.
    groups = [[0], [f for f in pending if f != 0]] if 0 in pending else [pending]
    for folds in groups:
        if not folds:
            continue
        started = time.monotonic()
        progress(root, "story_gcv", model=model, folds=folds)
        maps, biases, identity, infos = gcv_maps(moment, folds)
        fit_seconds = time.monotonic() - started
        oracle_delta = None
        if folds == [0]:
            m = train_moments(moment, [0])
            oracle = PooledMomentRidge(
                n=int(m["n"][0]),
                sum_x=m["sx"][0],
                sum_y=m["sy"][0],
                yss=float(m["yss"][0]),
                c_xx=m["xx"][0],
                c_xy=m["xy"][0],
            )
            if oracle.best_lambda != infos[0]["best_lambda"]:
                raise RuntimeError("new GCV lambda differs from independent parent oracle")
            np.testing.assert_allclose(
                maps[0].numpy(), (oracle.map / oracle.sd[:, None]).numpy(), rtol=1e-7, atol=1e-8
            )
            probe = p["x"][p["membership"] == 0][:32]
            actual = tensor(probe) @ maps[0] + biases[0]
            original = tensor(oracle.predict_np(probe))
            oracle_delta = float(torch.linalg.norm(actual - original) / torch.linalg.norm(original))
            if oracle_delta > 1e-7:
                raise RuntimeError("new-map parent prediction oracle failed")
            del oracle
            progress(
                root,
                "full_size_fit_pilot",
                model=model,
                fit_seconds=fit_seconds,
                projected_remaining_seconds=fit_seconds * len(pending[1:]),
                oracle_relative_error=oracle_delta,
            )
        for i, fold in enumerate(folds):
            test = p["membership"] == fold
            pred = tensor(p["x"][test]) @ maps[i] + biases[i]
            metrics = score_batch(
                torch.stack([pred, tensor(p["x"][test]) + identity[i]]), p["y"][test]
            )
            record = {
                "model": model,
                "source": f"{STORY}__{model}",
                "fold": fold,
                "ridge": infos[i],
                "n_test": int(test.sum()),
                "own": metrics[0],
                "source_identity_bias": metrics[1],
                "train_ids_sha256": digest(np.asarray(p["ids"])[~test].tolist()),
                "test_ids_sha256": digest(np.asarray(p["ids"])[test].tolist()),
                "batch_fit_seconds": fit_seconds,
                "oracle_relative_error": oracle_delta,
            }
            saved[fold] = save_packet(
                root,
                f"maps/{model}__fold{fold}",
                {
                    "map": maps[i].numpy(),
                    "bias": biases[i].numpy(),
                    "identity_bias": identity[i].numpy(),
                    "prediction": pred.numpy(),
                    "test_conv_id": np.asarray(p["ids"])[test],
                },
                record,
                fp,
            )
            progress(root, "story_map_complete", model=model, fold=fold)
    return saved


def transfer_unit(
    root,
    model,
    source,
    target,
    fold,
    amap,
    bias,
    identity_bias,
    own_map,
    own_bias,
    panel,
    valid,
    ridge,
    own_metric,
    fp,
):
    stem = f"transfers/{model}__{source.split('__')[0]}_{source.split('__')[2]}__to__{target.split('__')[0]}_{target.split('__')[2]}__fold{fold}"
    previous = resume_packet(root, stem, fp)
    if previous is not None:
        return previous
    p = panel[target]
    train, test = p["membership"] != fold, p["membership"] == fold
    source_cells = (
        [source] if source in panel else [c for c in panel if not c.startswith(STORY + "__")]
    )
    test_ids = np.asarray(p["ids"])[test]
    for cell in source_cells:
        s = panel[cell]
        if set(test_ids) & set(np.asarray(s["ids"])[s["membership"] != fold]):
            raise RuntimeError("held-out conversation leaked into source map")
    prediction = tensor(p["x"]) @ amap + bias
    coefficients = base.calibrate(prediction[train].numpy(), p["y"][train])
    frozen = prediction[test]
    target_identity = tensor((p["y"][train] - p["x"][train]).mean(0))
    variants = {
        "frozen": frozen,
        "bias": frozen + tensor(coefficients["bias"]),
        "bias_scale": coefficients["gain"] * (frozen - tensor(coefficients["prediction_mean"]))
        + tensor(coefficients["target_mean"]),
        "source_identity_bias": tensor(p["x"][test]) + identity_bias,
        "target_identity_bias": tensor(p["x"][test]) + target_identity,
    }
    metrics = dict(
        zip(variants, score_batch(torch.stack(list(variants.values())), p["y"][test]), strict=True)
    )
    metrics["own"] = own_metric
    common = set.intersection(*(valid[c] for c in source_cells), valid[target])
    source_ids = set.union(*(set(panel[c]["ids"]) for c in source_cells))
    matched = np.array([cid in common and cid in source_ids for cid in test_ids])
    sensitivity = {
        "n_test": int(matched.sum()),
        "status": "complete" if matched.sum() >= max(KS) else "insufficient_pool",
    }
    if sensitivity["status"] == "complete":
        sensitivity["metrics"] = dict(
            zip(
                variants,
                score_batch(
                    torch.stack([v[matched] for v in variants.values()]), p["y"][test][matched]
                ),
                strict=True,
            )
        )
        own_pred = tensor(p["x"][test][matched]) @ own_map + own_bias
        sensitivity["metrics"]["own"] = score_batch(own_pred, p["y"][test][matched])[0]
    record = {
        "model": model,
        "source": source,
        "target": target,
        "fold": fold,
        "n_train": ridge["n_train"],
        "n_target_calibration": int(train.sum()),
        "n_test": int(test.sum()),
        "ridge": ridge,
        "metrics": metrics,
        "matched_query_metrics": sensitivity,
        "source_train_ids_sha256": {
            c: digest(np.asarray(panel[c]["ids"])[panel[c]["membership"] != fold].tolist())
            for c in source_cells
        },
        "test_ids_sha256": digest(test_ids.tolist()),
        "gain": coefficients["gain"],
    }
    result = save_packet(
        root,
        stem,
        {
            "prediction_frozen": frozen.numpy(),
            "test_conv_id": test_ids,
            "matched_query_mask": matched,
            "calibration_conv_id": np.asarray(p["ids"])[train],
            **{key: np.asarray(value) for key, value in coefficients.items()},
            "source_identity_bias": identity_bias.numpy(),
            "target_identity_bias": target_identity.numpy(),
        },
        record,
        fp,
    )
    progress(root, "transfer_complete", model=model, source=source, target=target, fold=fold)
    return result


def fit_transfers(root, model, panel, valid, references, fp):
    story_cell = f"{STORY}__{model}"
    old_cells = [f"{prefix}__{model}" for _, prefix in base.SETTINGS]
    maps_meta = new_maps(root, model, panel, fp)
    bank = [geometry.moments_by_fold(panel[cell]) for cell in old_cells]
    rows = []
    for fold in range(5):
        progress(root, "restore_old_maps", model=model, fold=fold)
        infos = [references[c]["folds"][fold]["ridge"] for c in old_cells]
        pooled = references[old_cells[0]]["folds"][fold]["pooled_ridge"]
        if any(references[c]["folds"][fold]["pooled_ridge"] != pooled for c in old_cells):
            raise RuntimeError("pooled parent metadata differ")
        infos.append(pooled)
        old_maps, old_bias, old_identity, ns, _ = geometry.restore_maps(
            bank, fold, [v["best_lambda"] for v in infos]
        )
        if ns.tolist() != [v["n_train"] for v in infos]:
            raise RuntimeError("old-map training counts changed")
        with np.load(root / maps_meta[fold]["array_path"], allow_pickle=False) as z:
            story_map, story_bias, story_identity = (
                tensor(z["map"]),
                tensor(z["bias"]),
                tensor(z["identity_bias"]),
            )
        story_info = maps_meta[fold]["ridge"]
        for i, cell in enumerate(old_cells):
            p = panel[cell]
            test = p["membership"] == fold
            own = score_batch(tensor(p["x"][test]) @ old_maps[i] + old_bias[i], p["y"][test])[0]
            original_parity(own, references[cell]["folds"][fold]["metrics"]["own"], cell, fold)
            # Independently preserve the shared operator parity on every original target.
            shared = score_batch(tensor(p["x"][test]) @ old_maps[-1] + old_bias[-1], p["y"][test])[
                0
            ]
            original_parity(
                shared, references[cell]["folds"][fold]["metrics"]["pooled"], cell, fold
            )
            rows.append(
                transfer_unit(
                    root,
                    model,
                    story_cell,
                    cell,
                    fold,
                    story_map,
                    story_bias,
                    story_identity,
                    old_maps[i],
                    old_bias[i],
                    panel,
                    valid,
                    story_info,
                    own,
                    fp,
                )
            )
            rows.append(
                transfer_unit(
                    root,
                    model,
                    cell,
                    story_cell,
                    fold,
                    old_maps[i],
                    old_bias[i],
                    old_identity[i],
                    story_map,
                    story_bias,
                    panel,
                    valid,
                    infos[i],
                    maps_meta[fold]["own"],
                    fp,
                )
            )
        # Optional frozen six-setting pool diagnostic, retaining its explicit label.
        rows.append(
            transfer_unit(
                root,
                model,
                f"shared_six__on_policy__pooled__{model}",
                story_cell,
                fold,
                old_maps[-1],
                old_bias[-1],
                old_identity[-1],
                story_map,
                story_bias,
                panel,
                valid,
                pooled,
                maps_meta[fold]["own"],
                fp,
            )
        )
    return maps_meta, rows


def stage_draw_chunks(root, manifest, cell, story_fp):
    record = next(r for r in manifest["cells"] if r["cell"] == cell)
    is_story = cell.startswith(STORY + "__")
    parent_fp = recovery.capture_fingerprint(manifest, cell, False, k5.policy())
    raw_old_fp = k3.fingerprint(manifest, cell, False)
    new_fp = story_fp if is_story else legacy_k5_fingerprint(manifest, cell)

    def stage(offset):
        name = f"chunk_{offset:05d}"
        old_capture = fetch_receipted(
            root, f"{k5.PARENT_PREFIX}/captures/{cell}/{name}.npz", k5.PARENT_REV, parent_fp
        )
        old_raw = fetch_raw(
            root, f"{k5.RAW_PREFIX}/raw/{cell}/{name}.json", k5.PARENT_REV, raw_old_fp
        )
        if is_story:
            new_capture, new_raw = (
                root / "captures" / cell / f"{name}.npz",
                root / "raw" / cell / f"{name}.json",
            )
            receipt(new_capture, new_fp)
            receipt(new_raw, new_fp)
        else:
            new_capture = fetch_receipted(
                root, f"{OLD_K5_PREFIX}/captures/{cell}/{name}.npz", base.K5_REV, new_fp
            )
            new_raw = fetch_raw(
                root, f"{OLD_K5_PREFIX}/raw/{cell}/{name}.json", base.K5_REV, new_fp
            )
        return {
            "offset": offset,
            "old_capture": str(old_capture),
            "new_capture": str(new_capture),
            "old_raw": str(old_raw),
            "new_raw": str(new_raw),
            "old_raw_fingerprint": raw_old_fp,
            "new_fingerprint": new_fp,
            "sha256": {
                str(p.relative_to(root)): k3.sha(p)
                for p in [old_capture, new_capture, old_raw, new_raw]
            },
        }

    with ThreadPoolExecutor(max_workers=4) as pool:
        return list(pool.map(stage, range(0, record["n"], k3.CHUNK)))


def load_draws(root, chunks, panel, originals, cell):
    """Rebuild every stored K5 target exactly before exposing its five draws."""
    index = {cid: i for i, cid in enumerate(panel["ids"])}
    vectors = np.empty((len(index), 5, 3584), dtype=np.float16)
    texts, seen = {}, set()
    for chunk in chunks:
        with np.load(chunk["old_capture"], allow_pickle=False) as z:
            old = {key: z[key] for key in z.files}
        with np.load(chunk["new_capture"], allow_pickle=False) as z:
            new = {key: z[key] for key in z.files}
        targets, keep, caps = k5.average_targets(old, new)
        expected_ids = list(originals)[chunk["offset"] : chunk["offset"] + k3.CHUNK]
        if list(map(str, old["conv_id"])) != expected_ids:
            raise RuntimeError("capture chunk order differs from immutable draw-zero rows")
        ids = list(map(str, old["conv_id"][keep]))
        if seen & set(ids) or any(cid not in index for cid in ids):
            raise RuntimeError("draw-bank row coverage differs from published K5 bank")
        positions = np.array([index[cid] for cid in ids])
        np.testing.assert_array_equal(targets[5], panel["y"][positions])
        np.testing.assert_array_equal(caps[keep], panel["caps"][positions])
        np.testing.assert_array_equal(old["v_C"][keep], panel["x"][positions])
        vectors[positions] = np.concatenate(
            [old["v_A_0"][keep, None], old["v_A_12"][keep], new["v_A_34"][keep]], axis=1
        )
        old_rows = k3.load_raw(Path(chunk["old_raw"]), chunk["old_raw_fingerprint"])
        new_rows = k3.load_raw(Path(chunk["new_raw"]), chunk["new_fingerprint"])
        mapping = {(r["conv_id"], r["draw"]): r for r in old_rows + new_rows}
        if len(mapping) != len(old_rows) + len(new_rows):
            raise RuntimeError("duplicate conversation/draw text identity")
        all_ids = list(map(str, old["conv_id"]))
        if set(mapping) != {(cid, draw) for cid in all_ids for draw in (1, 2, 3, 4)}:
            raise RuntimeError("raw draw identities do not cover captures exactly")
        for row in old_rows + new_rows:
            cid, draw = row["conv_id"], row["draw"]
            expected_seed = k3.seed(cell, cid, draw) if draw in (1, 2) else k5.seed(cell, cid, draw)
            if row["seed"] != expected_seed or row["max_tokens_budget"] != k3.cap_for(cell):
                raise RuntimeError("raw draw seed or generation cap differs from recipe")
        cap_index = {cid: i for i, cid in enumerate(all_ids)}
        for cid in ids:
            source = originals[cid]
            prefix = source["final_text"][: source["answer_start"]]
            suffix = source["final_text"][source["answer_end"] :]
            selected = [source] + [mapping[cid, d] for d in (1, 2, 3, 4)]
            for draw, row in enumerate(selected[1:], 1):
                if (
                    row["final_text"] != prefix + row["answer"] + suffix
                    or row["answer_start"] != len(prefix)
                    or row["answer_end"] != len(prefix) + len(row["answer"])
                ):
                    raise RuntimeError("fresh answer differs from inherited literal prompt")
                if bool(caps[cap_index[cid], draw]) != (row["finish_reason"] == "length"):
                    raise RuntimeError("capture cap indicator differs from saved generation")
            texts[cid] = {"prefix": prefix, "answers": [r["answer"] for r in selected]}
        seen.update(ids)
    if seen != set(index) or not np.isfinite(vectors).all():
        raise RuntimeError("incomplete/nonfinite five-draw reconstruction")
    return vectors, texts


def response_statistics(story, chat, story_mean, chat_mean, membership, block=128):
    """Per-conversation geometry; draw pairs are never bootstrap observations."""
    n = len(story)
    if story.shape != chat.shape or story.shape[1] != 5 or story_mean.shape != chat_mean.shape:
        raise ValueError("expected paired five-draw representations")
    output = {
        key: np.empty(n, dtype=np.float64)
        for key in [
            "mean_answer_cosine",
            "centered_mean_answer_cosine",
            "mean_answer_squared_distance",
            "normalized_mean_answer_squared_distance",
            "cross_draw_cosine",
            "story_within_draw_cosine",
            "chat_within_draw_cosine",
            "cross_draw_squared_distance",
            "story_within_squared_distance",
            "chat_within_squared_distance",
            "corrected_mean_response_squared_displacement",
            "normalized_corrected_mean_response_squared_displacement",
        ]
    }
    centered_story, centered_chat = np.empty_like(story_mean), np.empty_like(chat_mean)
    scale = np.empty(n)
    for fold in range(5):
        train, test = membership != fold, membership == fold
        if train.sum() < 2 or not test.any():
            raise ValueError("insufficient paired conversation folds")
        centered_story[test] = story_mean[test] - story_mean[train].mean(0)
        centered_chat[test] = chat_mean[test] - chat_mean[train].mean(0)
        common_scale = float(
            (np.square(story_mean[train]).sum(1).mean() + np.square(chat_mean[train]).sum(1).mean())
            / 2
        )
        if not common_scale > 0:
            raise ValueError("zero training normalization scale")
        scale[test] = common_scale
    pair_i, pair_j = torch.triu_indices(5, 5, offset=1)

    def cosine(a, b):
        denom = a.square().sum(-1).sqrt() * b.square().sum(-1).sqrt()
        if bool((denom <= 0).any()):
            raise ValueError("undefined zero-norm response cosine")
        return (a * b).sum(-1) / denom

    for start in range(0, n, block):
        stop = min(n, start + block)
        s, c = tensor(story[start:stop]), tensor(chat[start:stop])
        sm, cm = tensor(story_mean[start:stop]), tensor(chat_mean[start:stop])
        sc, cc = tensor(centered_story[start:stop]), tensor(centered_chat[start:stop])
        cross = (
            (
                s.square().sum(2)[:, :, None]
                + c.square().sum(2)[:, None, :]
                - 2 * (s @ c.transpose(1, 2))
            )
            .clamp_min(0)
            .mean((1, 2))
        )
        within_s = (s[:, pair_i] - s[:, pair_j]).square().sum(2).mean(1)
        within_c = (c[:, pair_i] - c[:, pair_j]).square().sum(2).mean(1)
        corrected = cross - (within_s + within_c) / 2
        norms_s, norms_c = s.norm(dim=2), c.norm(dim=2)
        if bool((norms_s <= 0).any()) or bool((norms_c <= 0).any()):
            raise ValueError("zero-norm draw")
        values = {
            "mean_answer_cosine": cosine(sm, cm),
            "centered_mean_answer_cosine": cosine(sc, cc),
            "mean_answer_squared_distance": (sm - cm).square().sum(1),
            "normalized_mean_answer_squared_distance": (sm - cm).square().sum(1)
            / tensor(scale[start:stop]),
            "cross_draw_cosine": (
                (s / norms_s[:, :, None]) @ (c / norms_c[:, :, None]).transpose(1, 2)
            ).mean((1, 2)),
            "story_within_draw_cosine": cosine(s[:, pair_i], s[:, pair_j]).mean(1),
            "chat_within_draw_cosine": cosine(c[:, pair_i], c[:, pair_j]).mean(1),
            "cross_draw_squared_distance": cross,
            "story_within_squared_distance": within_s,
            "chat_within_squared_distance": within_c,
            "corrected_mean_response_squared_displacement": corrected,
            "normalized_corrected_mean_response_squared_displacement": corrected
            / tensor(scale[start:stop]),
        }
        for key, value in values.items():
            output[key][start:stop] = value.numpy()
    output["training_normalization_scale"] = scale
    return output


def bootstrap_summary(metrics, seed, draws=200):
    keys = list(metrics)
    values = np.column_stack([metrics[k] for k in keys])
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("empty or nonfinite conversation metrics")
    rng = np.random.default_rng(seed)
    weights = rng.multinomial(len(values), np.full(len(values), 1 / len(values)), size=draws) / len(
        values
    )
    boot = weights @ values
    intervals = np.quantile(boot, [0.025, 0.975], axis=0)
    return {
        key: {
            "mean": float(values[:, i].mean()),
            "ci95": intervals[:, i].tolist(),
            "n_conversations": len(values),
            "bootstrap_draws": draws,
        }
        for i, key in enumerate(keys)
    }


def lexical_metrics(story_text, chat_text, ids):
    output = {
        key: []
        for key in [
            "lexical_jaccard",
            "story_answer_characters",
            "chat_answer_characters",
            "story_lexical_tokens",
            "chat_lexical_tokens",
        ]
    }
    for cid in ids:
        s, c = story_text[cid]["answers"], chat_text[cid]["answers"]
        st = [re.findall(r"\w+|[^\w\s]", x.casefold()) for x in s]
        ct = [re.findall(r"\w+|[^\w\s]", x.casefold()) for x in c]
        jaccard = []
        for a in map(set, st):
            for b in map(set, ct):
                union = a | b
                jaccard.append(len(a & b) / len(union) if union else 1.0)
        output["lexical_jaccard"].append(np.mean(jaccard))
        output["story_answer_characters"].append(np.mean([len(x) for x in s]))
        output["chat_answer_characters"].append(np.mean([len(x) for x in c]))
        output["story_lexical_tokens"].append(np.mean([len(x) for x in st]))
        output["chat_lexical_tokens"].append(np.mean([len(x) for x in ct]))
    return {key: np.asarray(value) for key, value in output.items()}


def qualitative(root, model, ids, queries, story_text, chat_text, values, selected, fp):
    largest = [
        cid
        for _, cid in sorted(zip(values, ids, strict=True), key=lambda pair: (-pair[0], pair[1]))[
            :5
        ]
    ]
    entries = []
    for cid in sorted(set(selected) | set(largest)):
        stem = (
            root
            / "analysis"
            / "qualitative"
            / model
            / hashlib.sha256(cid.encode()).hexdigest()[:16]
        )
        record = {
            "conv_id": cid,
            "query": queries[cid],
            "selected_before_answer_comparison": cid in selected,
            "selected_largest_gap": cid in largest,
            "review_status": "unreviewed",
            "context_added_information": "requires full-prefix inspection",
            "story": story_text[cid],
            "chat": chat_text[cid],
        }
        k3.atomic_json(stem.with_suffix(".json"), record)
        lines = [
            f"# {cid}",
            "",
            "Full prefixes and all five draws. Context-confound review pending.",
            "",
        ]
        for framing in ("story", "chat"):
            lines += [f"## {framing}", "", "### Prefix", "", record[framing]["prefix"], ""]
            for draw, answer in enumerate(record[framing]["answers"]):
                lines += [f"### Draw {draw}", "", answer, ""]
        stem.with_suffix(".md").write_text("\n".join(lines))
        artifacts.seal_many([stem.with_suffix(".json"), stem.with_suffix(".md")], root, fp)
        entries.append(
            {
                "conv_id": cid,
                "json": str(stem.with_suffix(".json").relative_to(root)),
                "markdown": str(stem.with_suffix(".md").relative_to(root)),
                "random_sample": cid in selected,
                "largest_gap": cid in largest,
            }
        )
    return entries


def compare_answers(root, model, manifest, panel, originals, queries, valid, audit, story_fp, fp):
    stem = f"answers/{model}"
    previous = resume_packet(root, stem, fp)
    if previous is not None:
        return previous
    story_cell, chat_cell = f"{STORY}__{model}", f"{CHAT}__{model}"
    s, c = panel[story_cell], panel[chat_cell]
    ids = sorted(set(s["ids"]) & set(c["ids"]) & valid[story_cell] & valid[chat_cell])
    if len(ids) < 20:
        raise RuntimeError("insufficient complete-five query-matched answers")
    seed = int.from_bytes(
        hashlib.sha256(f"2054-story-answer|{model}|137".encode()).digest()[:4], "little"
    )
    selected = np.random.default_rng(seed).choice(ids, 20, replace=False).tolist()
    progress(root, "stage_answer_draws", model=model)
    schunks = stage_draw_chunks(root, manifest, story_cell, story_fp[story_cell])
    cchunks = stage_draw_chunks(root, manifest, chat_cell, story_fp[story_cell])
    sv, st = load_draws(root, schunks, s, originals[story_cell], story_cell)
    cv, ct = load_draws(root, cchunks, c, originals[chat_cell], chat_cell)
    si, ci = {cid: i for i, cid in enumerate(s["ids"])}, {cid: i for i, cid in enumerate(c["ids"])}
    ia, ib = np.array([si[cid] for cid in ids]), np.array([ci[cid] for cid in ids])
    membership = s["membership"][ia]
    if not np.array_equal(membership, c["membership"][ib]):
        raise RuntimeError("paired answer folds differ")
    progress(root, "answer_geometry", model=model, n_paired=len(ids))
    values = response_statistics(sv[ia], cv[ib], s["y"][ia], c["y"][ib], membership)
    values.update(lexical_metrics(st, ct, ids))
    values["story_cap_fraction"] = s["caps"][ia].mean(1)
    values["chat_cap_fraction"] = c["caps"][ib].mean(1)
    summary = bootstrap_summary(values, seed)
    paths = qualitative(
        root, model, ids, queries, st, ct, values["mean_answer_squared_distance"], selected, fp
    )
    record = {
        "model": model,
        "coverage": {
            "n_story_complete_five": len(s["ids"]),
            "n_chat_complete_five": len(c["ids"]),
            "n_shared_complete_five": len(set(s["ids"]) & set(c["ids"])),
            "n_query_matched_complete_five": len(ids),
            "query_audit": audit,
        },
        "summary": summary,
        "qualitative_paths": paths,
        "draw_sources": {"story": schunks, "chat": cchunks},
        "bootstrap_seed": seed,
        "method": "Paired conversations; 25 cross and ten distinct within-framing pairs. Raw squared-Euclidean correction before one common training-fold normalization. Center each framing using paired non-test-fold means.",
        "limitations": [
            "Literal query equality does not remove narrative-added task information.",
            "Activation similarity includes context effects and is not semantic equivalence.",
            "Draw-zero versus fresh-draw runtime differences limit iid interpretation.",
            "Intervals bootstrap query-level measurements without refitting centering constants.",
            "Lexical Jaccard uses case-folded Unicode word/punctuation tokens; two empty sets score one.",
            "Qualitative samples require subsequent full-prefix inspection; no automated refusal rates.",
        ],
    }
    return save_packet(
        root,
        stem,
        {
            "conv_id": np.array(ids),
            "fold": membership,
            "story_cap_mask": s["caps"][ia],
            "chat_cap_mask": c["caps"][ib],
            **values,
        },
        record,
        fp,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    root = args.out_root.resolve()
    from scripts import issue2054_k5_assistant_story as producer

    k3.PREFIX = producer.OUTPUT_PREFIX
    manifest_path, generation_path = root / "manifest.json", root / "generation_complete.json"
    manifest, generation = (
        json.loads(manifest_path.read_text()),
        json.loads(generation_path.read_text()),
    )
    receipt(generation_path, producer.fingerprint(manifest))
    source_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    if source_sha != generation["source_sha"]:
        raise RuntimeError("analysis Git source differs from launched producer")
    if (
        generation["status"] != "complete"
        or k3.sha(manifest_path) != producer.MANIFEST_SHA256
        or len(manifest["cells"]) != 56
    ):
        raise RuntimeError("producer coverage or full K3 manifest changed")
    if generation["manifest_sha256"] != k3.sha(manifest_path) or set(
        generation["selected_cells"]
    ) != set(producer.ASSISTANT_STORY_CELLS):
        raise RuntimeError("producer selected population changed")
    fp = fingerprint(manifest, generation)
    complete = root / "analysis" / "analysis_complete.json"
    if complete.with_suffix(".json.done.json").exists():
        receipt(complete, fp)
        receipt(root / "analysis" / "results.json", fp)
        record = json.loads(complete.read_text())
        inventory = root / record["inventory_path"]
        if (
            record["source_sha"] != source_sha
            or k3.sha(inventory) != record["inventory_sha256"]
            or k3.sha(root / record["results_path"]) != record["results_sha256"]
        ):
            raise RuntimeError("completed analysis identity changed")
        verify_inventory(root, inventory, fp)
        print(complete.read_text(), flush=True)
        return
    started = time.time()
    progress(root, "stage_inputs")
    sources = stage_banks(root)
    fold_path = root / manifest["fold_map"]
    if k3.sha(fold_path) != manifest["fold_sha256"]:
        raise RuntimeError("global fold manifest changed")
    fold_info = json.loads(fold_path.read_text())
    if fold_info["k"] != 5 or fold_info["seed"] != 137:
        raise ValueError("unexpected global conversation folds")
    reference_path = REPO / "eval_results/issue_2054/section44_k5/k5_results.json"
    if k3.sha(reference_path) != geometry.REFERENCE_SHA:
        raise RuntimeError("published K5 references changed")
    references = {
        r["cell"]: r
        for r in json.loads(reference_path.read_text())["results"]
        if r["cohort"] == "all" and r["k_rollouts"] == 5
    }
    story_fp = {
        cell: k5.fingerprint(manifest, cell, cells=producer.ASSISTANT_STORY_CELLS)
        for cell in producer.ASSISTANT_STORY_CELLS
    }
    chosen = set(sources) | set(producer.ASSISTANT_STORY_CELLS)
    original_records = [r for r in manifest["cells"] if r["cell"] in chosen]
    if len(original_records) != 14:
        raise ValueError("expected fourteen original raw sources")
    originals = stage_original_text(root, original_records)
    models = []
    for model in base.MODELS:
        progress(root, "model_start", model=model)
        panel = load_banks(root, model, sources, fold_info["fold_of"], story_fp)
        queries, valid, audit = query_matches(originals, model)
        own, transfers = fit_transfers(root, model, panel, valid, references, fp)
        answers = compare_answers(
            root, model, manifest, panel, originals, queries, valid, audit, story_fp, fp
        )
        result = {
            "model": model,
            "labels": LABELS,
            "own_folds": own,
            "transfers": transfers,
            "answers": answers,
        }
        path = root / "analysis" / f"{model}.json"
        k3.atomic_json(path, result)
        artifacts.seal_many([path], root, fp)
        models.append(result)
        del panel
    result = {
        "status": "complete",
        "schema_version": 1,
        "models": models,
        "labels": LABELS,
        "analysis_fingerprint": fp,
        "started_at": started,
        "completed_at": time.time(),
        "provenance": {
            "generation_sha256": k3.sha(generation_path),
            "manifest_sha256": k3.sha(manifest_path),
            "script_sha256": k3.sha(__file__),
            "old_k5_source_sha256": OLD_K5_SOURCE_SHA,
            "reference_sha256": geometry.REFERENCE_SHA,
            "banks": sources,
            "original_raw": [
                {"cell": r["cell"], "path": r["raw"], "sha256": r["raw_sha256"]}
                for r in original_records
            ],
            "metadata": as_metadata_dict(
                git_provenance(cwd=REPO), phase="assistant_story_analysis"
            ),
        },
        "aggregation": "Transfer fold metrics are reported separately; presentation uses equal-fold means and separate directional pairs. Matched-query sensitivity retains frozen source fits and target-training calibration.",
    }
    result_path = root / "analysis" / "results.json"
    k3.atomic_json(result_path, result)
    artifacts.seal_many([result_path], root, fp)
    published = receipt(result_path, fp)
    inventory_path = root / "analysis" / "inventory.json"
    files = []
    for done in sorted((root / "analysis").rglob("*.done.json")):
        path = Path(str(done).removesuffix(".done.json"))
        if path in (inventory_path, complete):
            continue
        verified = receipt(path, fp)
        files.append(
            {
                "path": str(path.relative_to(root)),
                "sha256": k3.sha(path),
                "size": path.stat().st_size,
                "receipt": verified,
                "receipt_path": str(done.relative_to(root)),
                "receipt_sha256": k3.sha(done),
            }
        )
    k3.atomic_json(
        inventory_path,
        {
            "status": "complete",
            "analysis_fingerprint": fp,
            "source_sha": source_sha,
            "files": files,
        },
    )
    artifacts.seal_many([inventory_path], root, fp)
    k3.atomic_json(
        complete,
        {
            "status": "complete",
            "analysis_fingerprint": fp,
            "source_sha": source_sha,
            "results_sha256": k3.sha(result_path),
            "results_revision": published["revision"],
            "results_path": str(result_path.relative_to(root)),
            "results_hf_path": published["path"],
            "inventory_sha256": k3.sha(inventory_path),
            "inventory_path": str(inventory_path.relative_to(root)),
            "completed_at": time.time(),
        },
    )
    artifacts.seal_many([complete], root, fp)
    progress(
        root,
        "complete",
        models=2,
        new_own_maps=10,
        primary_transfer_units=120,
        pooled_diagnostic_units=10,
    )


if __name__ == "__main__":
    main()
