"""Zero-shot leave-one-setting-out maps on the twelve completed K5 panels.

Reuse the prior LOCO additive moments, with six setting groups per model and
source-only bias. The live generation/capture/own-map drivers stay unchanged.
"""

from __future__ import annotations

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts import issue2054_k3 as k3
from scripts import issue2054_k3_artifacts as artifacts
from scripts import issue2054_k3_fit as scoring
from scripts import issue2054_k5 as k5
from scripts import issue2054_k5_fit as matched
from scripts.issue2054_ctx2ctx_fit import SharedEighRidge
from scripts.issue2054_loco_pooled import combine
from scripts.issue2054_pool_specialize import PooledMomentRidge
from explore_persona_space.orchestrate.hub import retry_transient

PARENT_SOURCE = "91e9682ad092b22dcc0443cf8bd6f68ddc9f2917"
FOLLOWUP = "section44-k5-leave-one-setting-out"
N_FOLDS = 5


def digest_ids(ids):
    """Hash ordered identifiers without delimiter ambiguity."""
    return hashlib.sha256(json.dumps(list(ids), ensure_ascii=False).encode()).hexdigest()


def wait_for_parent(pid, root):
    """Keep a registered detached continuation idle until the parent exits."""
    if pid <= 0:
        return
    process = Path(f"/proc/{pid}")
    if not process.exists():
        raise RuntimeError("parent PID was absent when the continuation attached")
    command = (process / "cmdline").read_bytes().replace(b"\0", b" ").decode()
    if "issue2054_k5.py --stage job" not in command or PARENT_SOURCE not in command:
        raise RuntimeError("parent PID does not identify the pinned K5 driver")
    identity = (process / "stat").read_text().split()[21]
    started = time.monotonic()
    while process.exists():
        try:
            state = (process / "stat").read_text().split()
        except FileNotFoundError:
            break
        if state[21] != identity or state[2] == "Z":
            break
        if time.monotonic() - started > 48 * 3600:
            raise TimeoutError("parent K5 driver exceeded continuation wait fence")
        k3.log(
            "[phase=waiting_for_k5] registered continuation; no GPU work before parent completion"
        )
        time.sleep(60)
    # A failed parent has no fresh verified completion. Never proceed from an
    # existing tensor alone; the exact producer report and receipts are required.
    verify_parent(root)


def verify_parent(root):
    """Require the finished K5 producer and its unchanged local source files."""
    for path in (k5.__file__, matched.__file__):
        relative = str(Path(path).relative_to(REPO))
        expected = subprocess.check_output(["git", "show", f"{PARENT_SOURCE}:{relative}"], cwd=REPO)
        if hashlib.sha256(expected).hexdigest() != k3.sha(path):
            raise RuntimeError(f"K5 producing source changed: {relative}")
    if not k3.complete(root / "job_complete.json", k3.sha(k5.__file__)):
        raise RuntimeError("parent K5 job is not verified complete")
    report = json.loads((root / "job_complete.json").read_text())
    if (
        report["source_sha"] != PARENT_SOURCE
        or report["primary_panels"] != 36
        or report["status"] != "complete"
    ):
        raise RuntimeError("unexpected K5 parent source or coverage")
    return report


def verify_parent_uploads(root):
    """Independently compare the complete local output/receipt set to the Hub."""
    from huggingface_hub import HfApi

    api = HfApi()
    prefix = f"{k5.OUTPUT_PREFIX}/{root.name}"
    revision = retry_transient(
        lambda: api.repo_info(k3.HF_REPO, repo_type="dataset").sha,
        what="pin completed K5 uploads",
    )
    entries = retry_transient(
        lambda: list(
            # HUB_VERIFY_RETRY_EXEMPT: the complete iterator is inside retry_transient.
            api.list_repo_tree(
                k3.HF_REPO,
                path_in_repo=prefix,
                recursive=True,
                repo_type="dataset",
                revision=revision,
            )
        ),
        what="verify exact completed K5 output set",
    )
    remote = {e.path: e for e in entries if hasattr(e, "size")}
    local = [
        p for p in root.rglob("*") if p.is_file() and "inputs" not in p.relative_to(root).parts
    ]
    expected = {f"{prefix}/{p.relative_to(root)}": p for p in local}
    if not expected or set(expected) != set(remote):
        raise RuntimeError(
            f"K5 upload set differs: missing={sorted(set(expected) - set(remote))[:10]}, extra={sorted(set(remote) - set(expected))[:10]}"
        )
    total_bytes = 0
    for index, (destination, path) in enumerate(sorted(expected.items())):
        entry = remote[destination]
        if path.stat().st_size != entry.size:
            raise RuntimeError(f"K5 remote size mismatch: {destination}")
        if not path.name.endswith(".done.json"):
            receipt = json.loads(path.with_suffix(path.suffix + ".done.json").read_text())
            if receipt["path"] != destination or receipt["sha256"] != k3.sha(path):
                raise RuntimeError(f"K5 local receipt mismatch: {destination}")
        lfs = getattr(entry, "lfs", None)
        if lfs is not None:
            correct = lfs.sha256 == k3.sha(path)
        else:
            blob = hashlib.sha1(f"blob {entry.size}\0".encode())
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1 << 20), b""):
                    blob.update(chunk)
            correct = blob.hexdigest() == entry.blob_id
        if not correct:
            raise RuntimeError(f"K5 remote hash mismatch: {destination}")
        total_bytes += entry.size
        if (index + 1) % 100 == 0:
            k3.log(f"[phase=parent_upload_audit] files={index + 1}/{len(expected)}")
    return {
        "status": "pass",
        "revision": revision,
        "prefix": prefix,
        "files": len(expected),
        "bytes": total_bytes,
        "exact_set": True,
        "all_local_data_receipts_and_remote_hashes_match": True,
    }


def fingerprint(root, manifest, cell):
    """Bind each unit to exact aggregate, reference fits and reused solvers."""
    paths = [
        __file__,
        inspect.getfile(PooledMomentRidge),
        inspect.getfile(combine),
        inspect.getfile(SharedEighRidge),
        scoring.__file__,
        inspect.getfile(scoring.knn_retrieval),
    ]
    model = cell.split("__")[-1]
    source_inputs = {}
    for record in k5.selected(manifest):
        if record["cell"].split("__")[-1] != model:
            continue
        path = root / "k5" / f"{record['cell']}.npz.done.json"
        source_inputs[record["cell"]] = json.loads(path.read_text())["sha256"]
    references = {}
    with np.load(root / "k5" / f"{cell}.npz", allow_pickle=False) as z:
        cohort_names = matched.cohorts(cell, z["cap_mask"])
    for cohort in cohort_names:
        for fold in range(N_FOLDS):
            path = matched.fold_path(root, cell, cohort, 5, fold)
            references[str(path.relative_to(root))] = k3.sha(path)
    payload = {
        "parent_fit_fingerprint": matched.fit_fp(manifest, cell),
        "hashes": [k3.sha(p) for p in paths],
        "input_hashes": source_inputs,
        "reference_hashes": references,
        "group_by": "setting, separately within model",
        "cohort": "complete five draws, capped outcomes retained",
        "lambda_recipe": ["logspace", -2, 4, 13, "dof_cap", 0.9],
        "target_setting_labels_used_for_training": False,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def load_panel(root, manifest, model):
    """Use the original matched loader, then release unused target counts."""
    panel = matched.load_panel(root, manifest, model)
    for p in panel.values():
        p["y"] = p.pop("targets")[5]
    if len(panel) != 6 or any(c.split("__")[-1] != model for c in panel):
        raise RuntimeError("LOSO requires exactly six settings of the same model")
    return panel


def moments(panel, device):
    """Bank each setting/fold once, sharing reductions across every rotation."""
    result = {}
    for ci, (cell, p) in enumerate(panel.items()):
        started = time.monotonic()
        result[cell] = []
        for f in range(N_FOLDS):
            mask = p["membership"] == f
            if not mask.any():
                raise RuntimeError(f"empty setting fold: {cell}, {f}")
            x = torch.as_tensor(p["x"][mask], dtype=torch.float64, device=device)
            y = torch.as_tensor(p["y"][mask], dtype=torch.float64, device=device)
            result[cell].append(
                {
                    "n": int(mask.sum()),
                    "sum_x": x.sum(0),
                    "sum_y": y.sum(0),
                    "yss": float((y * y).sum()),
                    "c_xx": x.T @ x,
                    "c_xy": x.T @ y,
                }
            )
        k3.log(
            f"[phase=loso_moments] cell={ci + 1}/6 {cell} elapsed={time.monotonic() - started:.1f}s"
        )
    return result


def source_audit(panel, excluded, fold):
    """Assert both setting and global conversation exclusions before fitting."""
    if excluded not in panel or len(panel) != 6 or fold not in range(N_FOLDS):
        raise ValueError("invalid excluded setting/fold")
    target = panel[excluded]
    test_ids = [
        cid for cid, f in zip(target["ids"], target["membership"], strict=True) if f == fold
    ]
    test_set = set(test_ids)
    sources = {}
    for cell, p in panel.items():
        if cell == excluded:
            continue
        if cell.split("__")[-1] != excluded.split("__")[-1]:
            raise RuntimeError("cross-model source setting")
        ids = [cid for cid, f in zip(p["ids"], p["membership"], strict=True) if f != fold]
        if test_set.intersection(ids):
            raise RuntimeError("evaluation conversations leaked into source training")
        sources[cell] = {
            "n_train": len(ids),
            "train_ids_sha256": digest_ids(ids),
            "fold_counts": np.bincount(
                p["membership"][p["membership"] != fold], minlength=N_FOLDS
            ).tolist(),
        }
    return {
        "excluded_setting": excluded,
        "excluded_conversation_fold": fold,
        "test_ids_sha256": digest_ids(test_ids),
        "source_settings": sources,
        "n_train": sum(v["n_train"] for v in sources.values()),
        "target_setting_labels_used_for_training": False,
        "test_conversation_overlap": 0,
    }


def transfer(panel, bank, cell, fold):
    """Fit only source-setting moments and return direct predictions/bias."""
    audit = source_audit(panel, cell, fold)
    training = combine(bank, N_FOLDS, drop_speaker=cell, drop_fold=fold)
    if training["n"] != audit["n_train"]:
        raise RuntimeError("source moments do not match audited training rows")
    model = PooledMomentRidge(**training)
    test = panel[cell]["membership"] == fold
    prediction = model.predict_np(panel[cell]["x"][test])
    return prediction, model.global_bias.copy(), model.info(), audit


def parity_gate(panel, cell, fold, prediction, info, device):
    """Compare the actual moment fit with a materialized source-only solve."""
    x = np.concatenate([p["x"][p["membership"] != fold] for c, p in panel.items() if c != cell])
    y = np.concatenate([p["y"][p["membership"] != fold] for c, p in panel.items() if c != cell])
    test = panel[cell]["membership"] == fold
    reference, expected_info = SharedEighRidge(
        x, panel[cell]["x"][test], device=device
    ).fit_predict(y)
    delta = float(np.linalg.norm(prediction - reference) / max(np.linalg.norm(reference), 1e-18))
    if delta > 1e-6 or info["best_lambda"] != expected_info["best_lambda"]:
        raise RuntimeError(f"moment/materialized source-only parity failed: rel_l2={delta}")
    return {
        "status": "pass",
        "relative_l2": delta,
        "reference_ridge": expected_info,
        "same_dispatched_moment_prediction": True,
    }


def reference(root, manifest, cell, cohort, fold, n_test):
    """Require the original matching K5 own/full-pool result on the same rows."""
    path = matched.fold_path(root, cell, cohort, 5, fold)
    if not k3.complete(path, matched.fit_fp(manifest, cell)):
        raise RuntimeError(f"unverified matched reference: {path}")
    record = json.loads(path.read_text())
    expected = {"cell": cell, "cohort": cohort, "k_rollouts": 5, "fold": fold}
    if any(record[key] != value for key, value in expected.items()):
        raise RuntimeError("matched reference identity differs")
    result = {
        "path": str(path.relative_to(root)),
        "sha256": k3.sha(path),
        "status": record["status"],
    }
    if record["status"] == "complete":
        if record["n_test"] != n_test:
            raise RuntimeError("matched reference test population differs")
        result.update(own=record["metrics"]["own"], six_setting_pool=record["metrics"]["pooled"])
    else:
        if cohort == "all":
            raise RuntimeError("primary matched reference is withheld")
        result["withheld_record"] = record
    return result


def unit_paths(out, cell, fold):
    """Keep scores and downstream prediction arrays in one verified packet."""
    stem = out / "fold_checkpoints" / f"{cell}__fold{fold}"
    return Path(str(stem) + ".json"), Path(str(stem) + ".npz")


def fit_model(root, out, manifest, model, folds, *, device):
    """Run pending rotations using banked moments and save each unit at once."""
    panel = load_panel(root, manifest, model)
    fingerprints = {c: fingerprint(root, manifest, c) for c in panel}
    pending = [
        (cell, f)
        for f in folds
        for cell in panel
        if not all(k3.complete(p, fingerprints[cell]) for p in unit_paths(out, cell, f))
    ]
    if not pending:
        return
    bank = moments(panel, device)
    for index, (cell, fold) in enumerate(pending):
        started = time.monotonic()
        prediction, bias, ridge, audit = transfer(panel, bank, cell, fold)
        parity = None
        if fold == 0 and cell == next(iter(panel)):
            parity = parity_gate(panel, cell, fold, prediction, ridge, device)
        p = panel[cell]
        test = p["membership"] == fold
        x, y = p["x"][test], p["y"][test]
        records = {}
        for cohort, mask in matched.cohorts(cell, p["caps"]).items():
            use = mask[test]
            companion = reference(root, manifest, cell, cohort, fold, int(use.sum()))
            if use.sum() < 2 or not float(((y[use] - y[use].mean(0)) ** 2).sum()) > 0:
                records[cohort] = {
                    "status": "withheld",
                    "reason": "insufficient nonconstant evaluation rows",
                    "reference": companion,
                }
                continue
            records[cohort] = {
                "status": "complete",
                "n_test": int(use.sum()),
                "test_ids_sha256": digest_ids(np.asarray(p["ids"])[test][use]),
                "test_cap_counts_each_draw": p["caps"][test][use].sum(0).tolist(),
                "metrics": {
                    "leave_one_setting_out": scoring.score(prediction[use], y[use]),
                    "source_identity_bias": scoring.score(x[use] + bias, y[use]),
                },
                "reference": companion,
            }
        if records["all"]["status"] != "complete":
            raise RuntimeError("primary LOSO evaluation withheld")
        paths = unit_paths(out, cell, fold)
        paths[0].parent.mkdir(parents=True, exist_ok=True)
        scoring.save_npz(
            paths[1],
            {"prediction": prediction, "source_bias": bias, "conv_id": np.asarray(p["ids"])[test]},
        )
        result = {
            "status": "complete",
            "cell": cell,
            "fold": fold,
            "k_rollouts": 5,
            "ridge": ridge,
            "source_audit": audit,
            "cohorts": records,
            "seconds": time.monotonic() - started,
            "prediction_sha256": k3.sha(paths[1]),
            "input_sha256": k3.sha(root / "k5" / f"{cell}.npz"),
            "parent_source": PARENT_SOURCE,
            "parity_gate": parity,
        }
        k3.atomic_json(paths[0], result)
        artifacts.seal_many(paths, out, fingerprints[cell])
        k3.log(
            f"[phase=loso_fit] unit={index + 1}/{len(pending)} {cell} fold={fold} elapsed={time.monotonic() - started:.1f}s"
        )


def collect(root, out, manifest):
    """Require all twelve settings and five folds, retaining sensitivity gaps."""
    panels = []
    for cell_record in k5.selected(manifest):
        cell = cell_record["cell"]
        fp = fingerprint(root, manifest, cell)
        folds = []
        for f in range(N_FOLDS):
            paths = unit_paths(out, cell, f)
            if not all(k3.complete(p, fp) for p in paths):
                raise RuntimeError(f"incomplete LOSO setting/fold: {cell}/{f}")
            value = json.loads(paths[0].read_text())
            if value["cell"] != cell or value["fold"] != f or value["status"] != "complete":
                raise RuntimeError("LOSO checkpoint identity differs")
            folds.append(value)
        main = [f["cohorts"]["all"] for f in folds]
        panels.append(
            {
                "cell": cell,
                "folds": folds,
                "r2_mean": {
                    "leave_one_setting_out": float(
                        np.mean([r["metrics"]["leave_one_setting_out"]["r2"] for r in main])
                    ),
                    "source_identity_bias": float(
                        np.mean([r["metrics"]["source_identity_bias"]["r2"] for r in main])
                    ),
                    "own": float(np.mean([r["reference"]["own"]["r2"] for r in main])),
                    "six_setting_pool": float(
                        np.mean([r["reference"]["six_setting_pool"]["r2"] for r in main])
                    ),
                },
            }
        )
    if len(panels) != 12:
        raise RuntimeError("LOSO primary coverage is not 12/12")
    return panels


def main():
    """Run after the original K5 job; do not overlap with its GPU workers."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--wait-parent-pid", type=int, default=0)
    parser.add_argument("--import-check", action="store_true")
    args = parser.parse_args()
    if args.import_check:
        from explore_persona_space.orchestrate.argcheck import assert_args_attributes_defined

        assert_args_attributes_defined(__file__)
        return
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    if actual != args.source_sha:
        raise RuntimeError("LOSO source commit differs from dispatch pin")
    started = time.time()
    root, out = args.parent_root.resolve(), args.out_root.resolve()
    if root == out or root in out.parents or out in root.parents:
        raise RuntimeError("LOSO output must be separate from parent outputs")
    out.mkdir(parents=True, exist_ok=True)
    k5.configure_output()
    wait_for_parent(args.wait_parent_pid, root)
    parent = verify_parent(root)
    upload_report = verify_parent_uploads(root)
    k3.atomic_json(out / "parent_upload_verified.json", upload_report)
    artifacts.seal_many([out / "parent_upload_verified.json"], out, k3.sha(__file__))
    if shutil.disk_usage(out).free < 6 * 10**9:
        raise RuntimeError("LOSO needs 6 GB free for all pending predictions and receipts")
    if args.device.startswith("cuda"):
        free, total = torch.cuda.mem_get_info()
        if free < 24 * 10**9:
            raise RuntimeError("LOSO needs 24 GB free GPU memory after parent workers exit")
        k3.log(f"[phase=loso_preflight] gpu_free={free} gpu_total={total}")
    manifest = json.loads((root / "manifest.json").read_text())
    config = {
        "source_sha": actual,
        "parent": parent,
        "parent_manifest_sha256": k3.sha(root / "manifest.json"),
        "group_by": "setting",
        "models_separate": True,
        "folds": N_FOLDS,
        "target_labels_used_for_training": False,
        "layer": 19,
    }
    k3.atomic_json(out / "config.json", config)
    artifacts.seal_many([out / "config.json"], out, k3.sha(__file__))
    if k3.complete(out / "pilot_complete.json", k3.sha(__file__)):
        pilot = json.loads((out / "pilot_complete.json").read_text())
    else:
        start_path = out / "pilot_started.json"
        if not k3.complete(start_path, k3.sha(__file__)):
            k3.atomic_json(start_path, {"started": time.time(), "source_sha": actual})
            artifacts.seal_many([start_path], out, k3.sha(__file__))
        pilot_start = json.loads(start_path.read_text())["started"]
        for model in k3.MODEL_REVISIONS:
            fit_model(root, out, manifest, model, [0], device=args.device)
        pilot_seconds = time.time() - pilot_start
        pilot = {
            "status": "pass",
            "fold": 0,
            "settings": 12,
            "seconds": pilot_seconds,
            "projected_seconds": pilot_seconds * N_FOLDS,
            "source_sha": actual,
            "includes_downtime_if_resumed": True,
            "peak_gpu_bytes": torch.cuda.max_memory_allocated()
            if args.device.startswith("cuda")
            else None,
        }
        k3.atomic_json(out / "pilot_complete.json", pilot)
        artifacts.seal_many([out / "pilot_complete.json"], out, k3.sha(__file__))
    if pilot["projected_seconds"] > 4 * 3600:
        raise RuntimeError(
            "LOSO full-data pilot exceeds twice provisional 2-hour basis; inspect throughput/width"
        )
    for model in k3.MODEL_REVISIONS:
        fit_model(root, out, manifest, model, list(range(1, N_FOLDS)), device=args.device)
    panels = collect(root, out, manifest)
    report = {
        "status": "complete",
        "source_sha": actual,
        "parent_source_sha": PARENT_SOURCE,
        "followup_label": FOLLOWUP,
        "primary_panels": len(panels),
        "fold_units": 60,
        "started": started,
        "finished": time.time(),
        "hf_prefix": f"{k5.OUTPUT_PREFIX}/{out.name}",
    }
    k3.atomic_json(out / "results.json", {"metadata": report, "panels": panels})
    k3.atomic_json(out / "job_complete.json", report)
    artifacts.seal_many([out / "results.json", out / "job_complete.json"], out, k3.sha(__file__))
    k3.atomic_json(
        Path("/workspace/logs") / f"issue-2054-epm_results-loso-{time.time_ns()}.json",
        {
            "sentinel_schema_version": 1,
            "kind": "epm:results",
            "version": 1,
            "task_id": 2054,
            "note": json.dumps(report),
            "blocks_pipeline": False,
        },
    )
    from explore_persona_space.backends.artifacts import write_completion_sentinel

    write_completion_sentinel(
        sentinel_path=os.environ["EPS_SENTINEL_PATH"], issue=2054, extra=report
    )
    # workflow-lint: phase-done-reserved -- standalone terminal; the shell launcher execs only this driver.
    k3.log("[phase=done] K5 leave-one-setting-out fits and uploads complete")


if __name__ == "__main__":
    main()
