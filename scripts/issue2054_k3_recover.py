"""Resume saved K3 answers under an explicitly adjudicated capture policy."""

from __future__ import annotations

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts import issue2054_k3 as k3
from scripts import issue2054_k3_artifacts as artifacts
from scripts import issue2054_k3_diagnose as diagnosis
from scripts import issue2054_k3_fit as fit
from scripts.issue2054_k3_diagnose import parent_vectors, relative
from scripts.issue2054_k3_job import prepare_git
from scripts.issue2054_k3_restore import restore


def load_policy(path):
    policy = json.loads(path.read_text())
    if policy["status"] != "adjudicated_proceed" or policy["cutoff"] != 0.025:
        raise RuntimeError("capture recovery lacks an adjudicated policy")
    if policy["capture_code_sha256"] != k3.sha(k3.__file__) or policy[
        "parent_code_sha256"
    ] != k3.sha(k3.capture.__file__):
        raise RuntimeError("capture implementation differs from adjudicated code")
    if policy["models"] != k3.MODEL_REVISIONS or policy["measurement_changed"]:
        raise RuntimeError("this recovery only supports unchanged adjudicated capture")
    if policy["diagnostic_helper_sha256"] != k3.sha(diagnosis.__file__):
        raise RuntimeError("diagnostic apply-path helper differs from adjudicated code")
    return policy


def capture_fingerprint(manifest, cell, pilot, policy):
    if pilot:
        raise ValueError("recovery consumes production only")
    return hashlib.sha256(
        json.dumps(
            {
                "original": k3.fingerprint(manifest, cell, False),
                "policy": policy,
                "recovery_code": k3.sha(__file__),
                "uploads": k3.sha(artifacts.__file__),
                "diagnostic_helper": k3.sha(diagnosis.__file__),
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()


def parity_audit(model, tokenizer, rows, current, reference, policy):
    rel = relative(current, reference)
    parent, lower, upper, hook_difference = parent_vectors(
        model, tokenizer, rows, 8, use_cache=False
    )
    lower_error, upper_error = relative(lower, reference), relative(upper, reference)
    parent_error = relative(parent, reference)
    return rel, adjudicate_errors(
        rel, parent_error, lower_error, upper_error, hook_difference, policy
    )


def adjudicate_errors(rel, parent_error, lower_error, upper_error, hook_difference, policy):
    """Same production decision can replay persisted GPU controls without inference."""
    import numpy as np

    rel, parent_error, lower_error, upper_error = [
        np.asarray(v) for v in (rel, parent_error, lower_error, upper_error)
    ]
    if (
        rel.ndim != 1
        or not len(rel)
        or any(v.shape != rel.shape for v in (parent_error, lower_error, upper_error))
    ):
        raise RuntimeError("invalid parity control shapes")
    if hook_difference != 0 or not all(
        np.isfinite(v).all() for v in (rel, parent_error, lower_error, upper_error)
    ):
        raise RuntimeError("capture apply-path or finite-value check failed")
    # A rank comparison identifies the intended layer without tuning a tolerance.
    if np.any(rel >= np.minimum(lower_error, upper_error)) or np.any(
        parent_error >= np.minimum(lower_error, upper_error)
    ):
        raise RuntimeError(
            "layer19 fails to identify the banked layer against adjacent-layer controls"
        )
    warnings = np.flatnonzero(rel > policy["cutoff"]).tolist()
    severity = "WARN" if warnings else "PASS"
    k3.log(
        f"[phase=parity_policy] severity={severity} cutoff={policy['cutoff']} historical_max={rel.max():.6f} hook_hidden_states_max_abs={hook_difference}"
    )
    return {
        "severity": severity,
        "warning_row_indices": warnings,
        "historical_relative_error": rel.tolist(),
        "parent_relative_error": parent_error.tolist(),
        "adjacent18_relative_error": lower_error.tolist(),
        "adjacent20_relative_error": upper_error.tolist(),
        "hook_hidden_states_max_abs": hook_difference,
        "adjudication": policy["adjudication_concern"],
    }


def capture(root, manifest, args, policy):
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    spec = manifest["models"][args.model]
    tokenizer = AutoTokenizer.from_pretrained(
        spec["id"], revision=spec["revision"], use_fast=True, padding_side="right"
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = (
        AutoModelForCausalLM.from_pretrained(
            spec["id"],
            revision=spec["revision"],
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
        .to("cuda")
        .eval()
    )
    source = Path(json.loads((root / "restore_verified.json").read_text())["source_root"])
    for cell_index, record in enumerate(c for c in manifest["cells"] if "raw" in c):
        cell = record["cell"]
        if cell.split("__")[-1] != args.model:
            continue
        old_fp = k3.fingerprint(manifest, cell, False)
        fp = capture_fingerprint(manifest, cell, False, policy)
        rows = k3.banked_rows(root, record, False)
        with np.load(root / "inputs" / record["activation"], allow_pickle=False) as z:
            bank = {k: z[k] for k in z.files}
        order = {str(cid): i for i, cid in enumerate(bank["conv_id"])}
        pending = []
        for offset in range(0, len(rows), k3.CHUNK):
            if not k3.owns_chunk(cell_index, offset, args.shard, args.shards):
                continue
            path = root / "captures" / cell / f"chunk_{offset:05d}.npz"
            audit_path = root / "capture_audits" / cell / f"chunk_{offset:05d}.json"
            if k3.complete(path, fp) and k3.complete(audit_path, fp):
                continue
            started = time.monotonic()
            batch = rows[offset : offset + k3.CHUNK]
            old_path = source / "captures" / cell / path.name
            reused = k3.complete(old_path, old_fp)
            path.parent.mkdir(parents=True, exist_ok=True)
            if reused:
                shutil.copyfile(old_path, path)
                with np.load(path, allow_pickle=False) as z:
                    rel = z["parity_relative_error"].copy()
                audit = {
                    "severity": "PASS",
                    "reused_original_capture": True,
                    "source_revision": args.raw_revision,
                    "source_sha256": k3.sha(old_path),
                    "historical_relative_error": rel.tolist(),
                    "adjudication": policy["adjudication_concern"],
                }
                if not np.isfinite(rel).all() or rel.max() > policy["cutoff"]:
                    raise RuntimeError("legacy accepted capture violates its original receipt gate")
            else:
                raw_path = root / "raw" / cell / f"chunk_{offset:05d}.json"
                if not k3.complete(raw_path, old_fp):
                    raise RuntimeError("unverified production raw answers")
                fresh = k3.load_raw(raw_path, old_fp)
                if [(r["conv_id"], r["draw"]) for r in fresh] != [
                    (r["conv_id"], d) for r in batch for d in (1, 2)
                ]:
                    raise RuntimeError("raw rollout order/count mismatch")
                answers, contexts = k3.forward_vectors(model, tokenizer, fresh)
                fixed_context, _ = k3.forward_vectors(model, tokenizer, batch, prompt_only=True)
                valid = np.isfinite(answers).all(axis=1)
                if not np.isfinite(fixed_context).all() or any(
                    bool(ok) != bool(row["answer"]) for ok, row in zip(valid, fresh, strict=True)
                ):
                    raise RuntimeError("nonfinite nonempty answer or context")
                indices = np.array([order[r["conv_id"]] for r in batch])
                recomputed, _ = k3.forward_vectors(model, tokenizer, batch[:8])
                rel, audit = parity_audit(
                    model, tokenizer, batch[:8], recomputed, bank["v_A"][indices[:8]], policy
                )
                content = {
                    "conv_id": np.array([r["conv_id"] for r in batch]),
                    "v_C": fixed_context,
                    "v_A_0": bank["v_A"][indices],
                    "v_A_12": answers.reshape(len(batch), 2, 3584),
                    "sampled_legacy_v_C": contexts.reshape(len(batch), 2, 3584),
                    "banked_v_C": bank["v_C"][indices],
                    "v_P": bank["v_P"][indices],
                    "v_P_present": bank["v_P_present"][indices],
                    "parity_relative_error": rel,
                    "valid_draws_12": valid.reshape(len(batch), 2),
                    "cap_mask": np.array(
                        [
                            [r.get("finish_reason") == "length"]
                            + [fresh[2 * i + j]["finish_reason"] == "length" for j in (0, 1)]
                            for i, r in enumerate(batch)
                        ]
                    ),
                }
                fit.save_npz(path, content)
            audit.update(cell=cell, offset=offset, policy_sha256=k3.sha(args.policy))
            k3.atomic_json(audit_path, audit)
            pending.extend((path, audit_path))
            if len(pending) >= 16:
                artifacts.seal_many(pending, root, fp)
                pending.clear()
            k3.log(
                f"[phase=capture_recovery] {cell} offset={offset} reused={reused} parity_max={rel.max():.6f} severity={audit['severity']} seconds={time.monotonic() - started:.1f} peak_gb={torch.cuda.max_memory_allocated() / 1e9:.2f}"
            )
        if pending:
            artifacts.seal_many(pending, root, fp)


def audit_summary(root, manifest, policy):
    import numpy as np

    cells = []
    for record in manifest["cells"]:
        if "raw" not in record:
            continue
        cell = record["cell"]
        fp = capture_fingerprint(manifest, cell, False, policy)
        summary = {
            "cell": cell,
            "chunks": 0,
            "reused_chunks": 0,
            "warning_chunks": 0,
            "warning_rows": 0,
            "parity_relative_max": 0.0,
        }
        for offset in range(0, record["n"], k3.CHUNK):
            path = root / "captures" / cell / f"chunk_{offset:05d}.npz"
            audit_path = root / "capture_audits" / cell / f"chunk_{offset:05d}.json"
            if not k3.complete(path, fp) or not k3.complete(audit_path, fp):
                raise RuntimeError("capture/audit packet incomplete")
            audit = json.loads(audit_path.read_text())
            with np.load(path, allow_pickle=False) as z:
                rel = z["parity_relative_error"]
                if not np.array_equal(
                    rel, np.array(audit["historical_relative_error"], dtype=rel.dtype)
                ):
                    raise RuntimeError("capture/audit metric mismatch")
            warnings = int((rel > policy["cutoff"]).sum())
            if (
                audit["cell"] != cell
                or audit["offset"] != offset
                or audit["severity"] != ("WARN" if warnings else "PASS")
            ):
                raise RuntimeError("capture audit identity/severity mismatch")
            summary["chunks"] += 1
            summary["reused_chunks"] += int(audit.get("reused_original_capture", False))
            summary["warning_chunks"] += int(warnings > 0)
            summary["warning_rows"] += warnings
            summary["parity_relative_max"] = max(summary["parity_relative_max"], float(rel.max()))
        cells.append(summary)
    report = {
        "cells": cells,
        "total_chunks": sum(c["chunks"] for c in cells),
        "warning_rows": sum(c["warning_rows"] for c in cells),
        "adjudication_concern": policy["adjudication_concern"],
    }
    if report["total_chunks"] != 768:
        raise RuntimeError("capture audit coverage incomplete")
    k3.atomic_json(root / "capture_audit_summary.json", report)
    k3.seal(root / "capture_audit_summary.json", root, k3.sha(__file__))
    return report


def children(root, args, stage, devices):
    def worker(shard):
        models = list(k3.MODEL_REVISIONS) if stage == "capture" else [None]
        for model in models:
            command = [
                sys.executable,
                __file__,
                "--stage",
                stage,
                "--out-root",
                str(root),
                "--policy",
                str(args.policy),
                "--raw-revision",
                args.raw_revision,
                "--shard",
                str(shard),
                "--shards",
                str(len(devices)),
            ]
            if model:
                command += ["--model", model]
            log = root / "logs" / f"recovery_{stage}_{shard}_{model}.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            with log.open("w") as output:
                process = subprocess.Popen(
                    command,
                    env=dict(os.environ, CUDA_VISIBLE_DEVICES=devices[shard], PYTHONUNBUFFERED="1"),
                    stdout=output,
                    stderr=subprocess.STDOUT,
                )
                k3.atomic_json(
                    log.with_suffix(".pid.json"),
                    {"pid": process.pid, "stage": stage, "model": model},
                )
                code = process.wait()
            k3.seal(log, root, k3.sha(__file__))
            if code:
                k3.log(f"[phase=worker_failed] {log}: {log.read_text()[-12000:]}")
                raise RuntimeError(f"recovery worker failed: {stage}/{shard}/{model}, exit={code}")

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(devices)) as pool:
        jobs = [pool.submit(worker, i) for i in range(len(devices))]
        while any(not j.done() for j in jobs):
            k3.log(
                f"[phase={stage}_recovery] active_workers={sum(not j.done() for j in jobs)}/{len(devices)} failed_workers={sum(j.done() and j.exception() is not None for j in jobs)}"
            )
            time.sleep(30)
        for job in jobs:
            job.result()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("job", "capture", "fit"), required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--raw-revision", required=True)
    parser.add_argument("--source-sha")
    parser.add_argument("--model", choices=list(k3.MODEL_REVISIONS))
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    args = parser.parse_args()
    root = args.out_root.resolve()
    args.policy = args.policy.resolve()
    policy = load_policy(args.policy)
    if not root.name.startswith("capture_recovery_") or root.name != policy["output_root_name"]:
        raise RuntimeError("recovery requires its own policy-pinned output prefix")
    if policy["raw_revision"] != args.raw_revision:
        raise RuntimeError("raw snapshot differs from adjudicated policy")
    fp_fn = lambda m, c, p: capture_fingerprint(m, c, p, policy)  # noqa: E731
    if args.stage != "job":
        manifest = json.loads((root / "manifest.json").read_text())
        if args.stage == "capture":
            capture(root, manifest, args, policy)
        else:
            fit.fits(
                root,
                manifest,
                "cuda",
                False,
                fingerprint_fn=fp_fn,
                shard=args.shard,
                shards=args.shards,
                collect=False,
            )
        return
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if actual != args.source_sha:
        raise RuntimeError("recovery source mismatch")
    prepare_git(REPO, actual)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "explore_persona_space.orchestrate.preflight",
            "--planned-footprint-gb",
            "100",
            "--min-disk",
            "120",
        ],
        check=True,
    )
    started = time.time()
    root.mkdir(parents=True, exist_ok=True)
    manifest, _, report = restore(root, args.raw_revision)
    report.update(
        capture_reuse_adjudication_pending=False, capture_policy_sha256=k3.sha(args.policy)
    )
    k3.atomic_json(root / "restore_verified.json", report)
    k3.atomic_json(root / "capture_policy.json", policy)
    artifacts.seal_many(
        [root / "restore_verified.json", root / "manifest.json", root / "capture_policy.json"],
        root,
        k3.sha(__file__),
    )
    devices = (
        os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
        if os.environ.get("CUDA_VISIBLE_DEVICES")
        else subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"], text=True
        ).split()
    )
    if not devices:
        raise RuntimeError("no allocated GPUs")
    children(root, args, "capture", devices)
    audits = audit_summary(root, manifest, policy)
    fit.aggregate(root, manifest, False, fingerprint_fn=fp_fn)
    children(root, args, "fit", devices)
    fit.collect_results(
        root,
        manifest,
        fingerprint_fn=fp_fn,
        extra={"capture_adjudication": policy, "capture_audit_summary": audits},
    )
    summary = {
        "status": "complete",
        "source_sha": actual,
        "raw_revision": args.raw_revision,
        "policy_sha256": k3.sha(args.policy),
        "started": started,
        "finished": time.time(),
        "reused_raw_answers": report["answers"],
        "reused_capture_chunks": report["capture_chunks"],
        "coverage": json.loads((root / "coverage.json").read_text()),
        "hf_prefix": f"{k3.PREFIX}/{root.name}",
    }
    k3.atomic_json(root / "job_complete.json", summary)
    k3.seal(root / "job_complete.json", root, k3.sha(__file__))
    from explore_persona_space.backends.artifacts import write_completion_sentinel

    write_completion_sentinel(
        sentinel_path=os.environ["EPS_SENTINEL_PATH"], issue=2054, extra=summary
    )
    k3.log("[phase=done] K3 recovery capture and paired fits complete")


if __name__ == "__main__":
    main()
