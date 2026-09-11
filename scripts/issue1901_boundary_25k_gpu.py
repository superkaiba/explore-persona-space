#!/usr/bin/env python3
"""Capture and fit the four approved #1901 expanded-Wikipedia boundary maps.

The controller uses every allocated GPU, keeps complete articles together,
pilots each length quartile, persists captures before fits, and writes a terminal
sentinel only after exact row coverage and verified result uploads.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import heapq
import json
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
import issue931_extract_store as ES  # noqa: E402
import issue1901_boundary_25k as B  # noqa: E402
import issue1901_boundary_token_control as P  # noqa: E402
import issue1901_individual_boundary_tokens as I  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402


def save_records(records, store, index):
    ES.write_shard(records, store, index, "armC", layers=(19,))
    for ext in ("pt", "json"):
        (store / f"armC_shard{index:03d}.{ext}").replace(store / f"pairs_shard{index:03d}.{ext}")


def load_manifest(root):
    rows, ids, meta = I._load_selected(root / "manifest")
    assert B.sha_file(root / "inputs" / "provenance.json") == meta["config"]["inputs_sha"]
    provenance = json.loads((root / "inputs" / "provenance.json").read_text())
    for name, expected in provenance["hashes"].items():
        assert B.sha_file(root / "inputs" / name) == expected, name
    row_sha = hashlib.sha256("\n".join(r["row_id"] for r in rows).encode()).hexdigest()
    assert row_sha == meta["selected_row_ids_sha256"]
    frozen = [
        r for p in sorted((root / "inputs").glob("eval_rows-*.jsonl")) for r in B.read_jsonl(p)
    ]
    assert [r for r in rows if r["split"] != "train"] == frozen
    old_ids = torch.load(
        root / "inputs" / "eval_articles.pt", map_location="cpu", weights_only=True
    )
    for aid, tokens in zip(old_ids["window_ids"], old_ids["input_ids"], strict=True):
        assert ids[aid] == tokens.tolist(), aid
    manifest_files = ["meta.json", *meta["manifest_shards"], *meta["article_shards"]]
    content = "\n".join(f"{name}:{B.sha_file(root / 'manifest' / name)}" for name in manifest_files)
    meta["manifest_content_sha256"] = hashlib.sha256(content.encode()).hexdigest()
    assert meta["quotas_per_token_id"] == {"train": 25000, "val": 160, "test": 400}
    assert len(rows) == 102240
    assert len({r["row_id"] for r in rows}) == len(rows)
    assert Counter((r["boundary_token_id"], r["split"]) for r in rows) == Counter(
        {
            (t, split): count
            for t in B.TOKENS
            for split, count in meta["quotas_per_token_id"].items()
        }
    )
    return rows, ids, meta


def capture_regime(meta, batch_size, device=0):
    import transformers

    props = torch.cuda.get_device_properties(device)
    return {
        "manifest_sha": meta["selected_row_ids_sha256"],
        "batch_size": batch_size,
        "manifest_content_sha256": meta["manifest_content_sha256"],
        "model_revision": meta["source_provenance"]["model_revision"],
        "gpu_name": props.name,
        "gpu_memory": props.total_memory,
        "torch_version": str(torch.__version__),
        "transformers_version": transformers.__version__,
        "dtype": "bfloat16",
        "layer": 19,
    }


def validate_store(store, expected):
    """Count identities from tensor contents, not producer-reported sidecar counts."""
    assert len(expected) == len(set(expected)), "duplicate declared row identities"
    found = set()
    for path in sorted(store.glob("pairs_shard*.pt")):
        side = json.loads(path.with_suffix(".json").read_text())
        obj = torch.load(path, map_location="cpu", weights_only=True)
        row_ids = obj["row_ids"]
        assert side["layers"] == [19]
        assert side["row_ids"] == row_ids
        assert side["n_rows"] == len(row_ids)
        assert not (found & set(row_ids)), path
        assert len(set(row_ids)) == len(row_ids)
        for key in ("x_sep", "y"):
            x = obj["arrays"][key]
            assert x.shape == (len(row_ids), 1, 3584), (path, key, x.shape)
            assert torch.isfinite(x).all(), path
        found.update(row_ids)
    assert found == set(expected), (len(found), len(expected), list(set(expected) - found)[:4])
    return len(found)


def partition_articles(items, width):
    """Deterministic greedy length-cost assignment; no article crosses workers."""
    loads = [(0, rank) for rank in range(width)]
    heapq.heapify(loads)
    assigned = [[] for _ in range(width)]
    ordered = sorted(items, key=lambda it: (-len(it["input_ids"]), it["item_id"]))
    for item in ordered:
        cost, rank = heapq.heappop(loads)
        assigned[rank].append(item["item_id"])
        length = len(item["input_ids"])
        heapq.heappush(loads, (cost + length * (4096 + length), rank))
    return assigned


def pinned_model(meta):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    revision = meta["source_provenance"]["model_revision"]
    model = AutoModelForCausalLM.from_pretrained(
        B.C.MODEL_ID, revision=revision, dtype=torch.bfloat16, device_map={"": 0}
    )
    model.eval()
    assert model.config.hidden_size == 3584 and model.config.num_hidden_layers == 28
    assert all(p.device.type == "cuda" for p in model.parameters())
    tokenizer = AutoTokenizer.from_pretrained(B.C.MODEL_ID, revision=revision)
    assert tokenizer.pad_token_id is not None
    return model, tokenizer.pad_token_id


def eval_parity(model, pad_id, rows, ids, inputs):
    wanted = []
    for token in B.TOKENS:
        wanted.extend(
            [r for r in rows if r["split"] == "test" and r["boundary_token_id"] == token][:4]
        )
    items = P._items_from_manifest(wanted, ids)
    bank = {}
    for path in (inputs / "eval_store").glob("*.pt"):
        obj = torch.load(path, map_location="cpu", weights_only=True)
        for i, row_id in enumerate(obj["row_ids"]):
            bank[row_id] = {k: v[i].float() for k, v in obj["arrays"].items()}
    comparisons = []
    for records in ES.run_extraction(model, items, pad_id, 4, "armC", layers=(19,)):
        for record in records:
            for key in ("x_sep", "y"):
                a, b = record[key].float(), bank[record["row_id"]][key]
                relative = float(torch.linalg.vector_norm(a - b) / torch.linalg.vector_norm(b))
                cosine = float(torch.nn.functional.cosine_similarity(a, b).min())
                assert cosine >= 0.999 and relative <= 0.03, (
                    record["row_id"],
                    key,
                    relative,
                    cosine,
                )
                comparisons.append(
                    {
                        "row_id": record["row_id"],
                        "key": key,
                        "relative_l2": relative,
                        "cosine": cosine,
                    }
                )
    assert len(comparisons) == 32
    return comparisons


def worker(args):
    rows, ids, meta = load_manifest(args.out_root)
    assignment = json.loads((args.out_root / "assignment.json").read_text())
    mine = set(assignment["articles"][args.rank])
    own_rows = [r for r in rows if r["article_id"] in mine]
    store = args.out_root / "capture" / f"rank{args.rank}" / "store"
    store.mkdir(parents=True, exist_ok=True)
    regime = capture_regime(meta, args.batch_size)
    pilot_path = store.parent / "pilot.json"
    if args.worker_mode == "pilot" and pilot_path.exists():
        assert json.loads(pilot_path.read_text())["regime"] == regime
        return
    if args.worker_mode == "capture":
        assert json.loads(pilot_path.read_text())["regime"] == regime, "runtime changed: re-pilot"
    done, index = P._scan_store_resume(store, (19,))
    assert done <= {r["row_id"] for r in own_rows}
    pending = [r for r in own_rows if r["row_id"] not in done]
    items = P._items_from_manifest(own_rows if args.worker_mode == "pilot" else pending, ids)
    if args.worker_mode == "capture" and not pending:
        validate_store(store, [r["row_id"] for r in own_rows])
        return
    model, pad_id = pinned_model(meta)
    if args.worker_mode == "pilot":
        ES.equivalence_check(model, items[:3], pad_id, "armC", layers=(19,))
        parity = (
            eval_parity(model, pad_id, rows, ids, args.out_root / "inputs")
            if args.rank == 0
            else []
        )
        # Warm the longest production shape before timing each stratum.
        ordered = sorted(items, key=lambda it: len(it["input_ids"]))
        ES.process_batch(model, ordered[-args.batch_size :], pad_id, "armC", layers=(19,))
        strata = []
        for group, indices in enumerate(np.array_split(np.arange(len(ordered)), 4)):
            metric_path = store.parent / f"pilot_quartile_{group}.json"
            if metric_path.exists():
                saved = json.loads(metric_path.read_text())
                assert saved["regime"] == regime
                assert (store / f"pairs_shard{group:03d}.pt").exists()
                strata.append(saved["metrics"])
                continue
            take = indices[
                np.linspace(0, len(indices) - 1, min(len(indices), args.batch_size * 10), dtype=int)
            ]
            selected = [ordered[int(i)] for i in take]
            times, records = [], []
            torch.cuda.reset_peak_memory_stats()
            for start in range(0, len(selected), args.batch_size):
                batch = selected[start : start + args.batch_size]
                torch.cuda.synchronize()
                t0 = time.monotonic()
                records.extend(ES.process_batch(model, batch, pad_id, "armC", layers=(19,)))
                torch.cuda.synchronize()
                times.append(time.monotonic() - t0)
            # Fixed indices make an interrupted write-before-metric retry idempotent.
            save_records(records, store, group)
            strata.append(
                {
                    "quartile": group,
                    "population_articles": len(indices),
                    "pilot_articles": len(selected),
                    "batch_seconds": times,
                    "length_range": [
                        len(ordered[int(indices[0])]["input_ids"]),
                        len(ordered[int(indices[-1])]["input_ids"]),
                    ],
                    "projected_seconds_mean": float(
                        np.mean(times) * len(indices) / args.batch_size
                    ),
                    "projected_seconds_p90": float(
                        np.quantile(times, 0.9) * len(indices) / args.batch_size
                    ),
                    "peak_gpu_bytes": torch.cuda.max_memory_allocated(),
                }
            )
            B.write_json(metric_path, {"regime": regime, "metrics": strata[-1]})
            print(f"[pilot] rank={args.rank} quartile={group} {json.dumps(strata[-1])}", flush=True)
        B.write_json(
            pilot_path,
            {
                "rank": args.rank,
                "strata": strata,
                "eval_parity": parity,
                "gpu": torch.cuda.get_device_name(),
                "manifest_sha": meta["selected_row_ids_sha256"],
                "regime": regime,
            },
        )
    else:
        buffer = []
        t0 = time.monotonic()
        for records in ES.run_extraction(
            model, items, pad_id, args.batch_size, "armC", layers=(19,)
        ):
            buffer.extend(records)
            while len(buffer) >= 1000:
                save_records(buffer[:1000], store, index)
                buffer = buffer[1000:]
                index += 1
                if index % 10 == 0:
                    upload = B.upload_verified(store.parent, f"capture/rank{args.rank}")
                    print("[capture-checkpoint] " + json.dumps(upload), flush=True)
        if buffer:
            save_records(buffer, store, index)
        n = validate_store(store, [r["row_id"] for r in own_rows])
        B.write_json(
            store.parent / "capture_complete.json",
            {
                "rows": n,
                "capture_seconds": time.monotonic() - t0,
                "manifest_sha": meta["selected_row_ids_sha256"],
            },
        )
        print(json.dumps(B.upload_verified(store.parent, f"capture/rank{args.rank}")), flush=True)
    del model
    gc.collect()
    torch.cuda.empty_cache()


def child_wave(args, slots, mode):
    children = []
    for rank, slot in enumerate(slots):
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = slot
        log = (args.out_root / f"worker-{rank}-{mode}.log").open("a")
        process = subprocess.Popen(
            [
                sys.executable,
                __file__,
                "--phase",
                "worker",
                "--worker-mode",
                mode,
                "--rank",
                str(rank),
                "--out-root",
                str(args.out_root),
                "--batch-size",
                str(args.batch_size),
            ],
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        children.append((rank, process, log))
    try:
        while any(p.poll() is None for _, p, _ in children):
            failed = [(rank, p.returncode) for rank, p, _ in children if p.poll() not in (None, 0)]
            if failed:
                raise RuntimeError(f"{mode} workers failed: {failed}")
            time.sleep(10)
        assert all(p.returncode == 0 for _, p, _ in children), [
            (r, p.returncode) for r, p, _ in children
        ]
    finally:
        for _, process, log in children:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=30)
            log.close()


def assemble_store(root, rows):
    target = root / "store"
    target.mkdir(exist_ok=True)
    sources = sorted((root / "capture").glob("rank*/store/pairs_shard*.pt"))
    sources += sorted((root / "inputs" / "eval_store").glob("pairs_shard*.pt"))
    for i, source in enumerate(sources):
        dest = target / f"pairs_shard{i:03d}.pt"
        if dest.exists():
            assert B.sha_file(dest) == B.sha_file(source), "store assembly changed"
        else:
            os.link(source, dest)
        obj = torch.load(source, map_location="cpu", weights_only=True)
        B.write_json(
            dest.with_suffix(".json"),
            {
                "layers": [19],
                "shard_index": i,
                "n_rows": len(obj["row_ids"]),
                "row_ids": obj["row_ids"],
                "group_ids": obj["group_ids"],
                "keys": ["x_sep", "y"],
                "shape_per_row": [1, 3584],
            },
        )
    validate_store(target, [r["row_id"] for r in rows])


def strict_retrieval(pred, true, view, whiten, seed):
    """Inherited metric kernels and strict midranks, including legitimate ties.

    The constant train-mean can whiten to zero, causing tied candidate scores.
    Strict and optimistic duplicate-aware ranks then differ even in a deduped
    pool; the general score_cell equality assertion does not apply here.
    """
    import issue1901_singleturn_retrieval_final as final

    q, p = pred[view.pred_rows], true[view.pool_rows]
    distances = final._precompute_metric_arrays(q, p, whiten(q), whiten(p))
    distances["whiten_csls"] = -final.MB.csls_scores(1.0 - distances["whiten_cosine"], final.K_CSLS)
    return {
        name: final._rank_summary(
            final._strict_ranks(distances[name], view.true_idx),
            len(p),
            np.random.default_rng(seed + 17 * i),
        )
        for i, name in enumerate(("whiten_csls", "whiten_cosine", "raw_cosine", "raw_euclidean"))
    }


def fits(args, rows, meta):
    import issue1901_singleturn_retrieval_final as FINAL
    from issue1901_plot1_remake import train_whitening_stats
    from scipy.linalg import solve_triangular

    X, Y, row_ids, articles = P._load_layer_arrays(args.out_root / "store", 19, (19,))
    positions = {r: i for i, r in enumerate(row_ids)}
    assert set(positions) == {r["row_id"] for r in rows}
    result_dir = args.out_root / "results"
    result_dir.mkdir(exist_ok=True)
    device = torch.device("cuda")
    results = {}
    fit_regime = {
        "manifest_sha": meta["selected_row_ids_sha256"],
        "manifest_content_sha256": meta["manifest_content_sha256"],
        "ridge_block": args.ridge_block,
        "seed": args.fit_seed,
        "n_boot": args.n_boot,
        "n_null": args.n_null,
        "ridge_grid": ["logspace", -3, 7, 21],
        "layer": 19,
        "whitening_diagonal_shrinkage": 0.1,
        "csls_k": 10,
    }
    for arm_i, token in enumerate(B.TOKENS):
        path = result_dir / f"token_{token}.json"
        weights = result_dir / f"token_{token}_weights.pt"
        predictions = result_dir / f"token_{token}_predictions.npz"
        if path.exists():
            saved = json.loads(path.read_text())
            assert saved["manifest_sha"] == meta["selected_row_ids_sha256"]
            assert saved["fit_regime"] == fit_regime
            assert weights.exists() and predictions.exists()
            assert B.sha_file(weights) == saved["weights_sha256"]
            assert B.sha_file(predictions) == saved["predictions_sha256"]
            results[str(token)] = saved
            continue
        t0 = time.monotonic()
        idx = {
            split: I._indices(rows, positions, token, split) for split in ("train", "val", "test")
        }
        pred, fit_meta, payload = I._fit_map(
            X, Y, idx["train"], idx["val"], idx["test"], device, args
        )
        true = P._to_f64_np(Y, idx["test"])
        group_ids = [articles[i] for i in idx["test"]]
        identity = P._identity_bias_chunked(X, Y, idx["train"], idx["test"])
        train_y = P._to_f64_np(Y, idx["train"])
        constant = np.broadcast_to(train_y.mean(0), true.shape).copy()
        mu, ell = train_whitening_stats(train_y, device)

        def whiten(x, mu=mu, ell=ell):
            return solve_triangular(
                ell, (np.asarray(x, np.float64) - mu).T, lower=True, check_finite=False
            ).T

        view = FINAL.make_eval_view(true.astype(np.float32), len(true), "keep_one")
        metrics = {}
        for name, p in (
            ("ridge", pred),
            ("identity_bias", identity),
            ("constant_train_mean", constant),
        ):
            cell = strict_retrieval(p, true, view, whiten, seed=args.fit_seed + arm_i)
            metrics[name] = {
                "reconstruction": I._score(p, true, group_ids, args.n_boot, args.fit_seed + arm_i),
                "retrieval": cell,
            }
        torch.save(payload, weights)
        np.savez(
            predictions,
            pred=pred,
            true=true,
            identity_bias=identity,
            constant_train_mean=constant,
            row_ids=np.asarray([row_ids[i] for i in idx["test"]]),
            article_ids=np.asarray(group_ids),
            whitening_mean=mu,
            whitening_cholesky=ell,
        )
        result = {
            "token_id": token,
            "token_spec": I.TOKEN_SPECS[token],
            "n_train": 25000,
            "n_val": 160,
            "n_test": 400,
            "layer": 19,
            "fit_meta": fit_meta,
            "metrics": metrics,
            "shuffled_pair_r2": I._shuffle_null_r2(pred, true, args.n_null, args.fit_seed + arm_i),
            "pool": view.diagnostics,
            "chance_top1": 1 / view.diagnostics["realized_n_pool"],
            "manifest_sha": meta["selected_row_ids_sha256"],
            "fit_regime": fit_regime,
            "weights_sha256": B.sha_file(weights),
            "predictions_sha256": B.sha_file(predictions),
            "fit_seconds": time.monotonic() - t0,
        }
        B.write_json(path, result)
        results[str(token)] = result
        print(
            f"[fit] token={token} R2={metrics['ridge']['reconstruction']['r2']:.6f} "
            f"seconds={result['fit_seconds']:.1f}",
            flush=True,
        )
        print(json.dumps(B.upload_verified(result_dir, "results")), flush=True)
    B.write_json(
        result_dir / "boundary25k.json",
        {
            "experiment": "issue1901_boundary25k",
            "model": B.C.MODEL_ID,
            "layer": 19,
            "tokens": results,
            "manifest": meta,
            "fit_protocol": {
                "ridge_block": args.ridge_block,
                "n_boot": args.n_boot,
                "n_null": args.n_null,
                "fit_seed": args.fit_seed,
            },
            "compute": json.loads((args.out_root / "pilot_summary.json").read_text()),
        },
    )
    return B.upload_verified(result_dir, "results")


def controller(args):
    root = args.out_root
    root.mkdir(parents=True, exist_ok=True)
    if not (root / "manifest" / "meta.json").exists():
        B.stage("prepare/manifest", root / "manifest")
    if not (root / "inputs" / "provenance.json").exists():
        B.stage("inputs", root / "inputs")
    if args.resume_cloud:
        # Recovery requires the same allocated width. Hardware changes refuse
        # the timing gate and need a new representative pilot before capture.
        B.stage("run_state", root / "run_state")
        B.stage("capture", root / "capture")
        import shutil

        shutil.copyfile(root / "run_state" / "assignment.json", root / "assignment.json")
    rows, ids, meta = load_manifest(root)
    count = torch.cuda.device_count()
    assert count > 0
    slots = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
    if not slots[0]:
        slots = [str(i) for i in range(count)]
    assert len(slots) == count
    assigned = partition_articles(
        P._items_from_manifest([r for r in rows if r["split"] == "train"], ids), count
    )
    del ids
    gc.collect()
    assignment = {"articles": assigned, "manifest_sha": meta["selected_row_ids_sha256"]}
    assignment_path = root / "assignment.json"
    if assignment_path.exists():
        assert json.loads(assignment_path.read_text()) == assignment, "allocation changed on resume"
    else:
        B.write_json(assignment_path, assignment)
    state_dir = root / "run_state"
    state_dir.mkdir(exist_ok=True)
    B.write_json(state_dir / "assignment.json", assignment)
    print(json.dumps(B.upload_verified(state_dir, "run_state")), flush=True)
    pilots = [root / "capture" / f"rank{rank}" / "pilot.json" for rank in range(count)]
    if not all(p.exists() for p in pilots):
        child_wave(args, slots, "pilot")
    pilot_rows = [json.loads(p.read_text()) for p in pilots]
    for rank, pilot in enumerate(pilot_rows):
        assert pilot["regime"] == capture_regime(meta, args.batch_size, rank), "re-pilot required"
    seconds = [sum(s["projected_seconds_p90"] for s in row["strata"]) for row in pilot_rows]
    summary = {
        "allocated_gpus": count,
        "pilot_by_rank": pilot_rows,
        "projected_capture_wall_hours_p90": max(seconds) / 3600,
        "projected_capture_gpu_hours_p90": sum(seconds) / 3600,
    }
    B.write_json(root / "pilot_summary.json", summary)
    print("[pilot-summary] " + json.dumps(summary), flush=True)
    print(json.dumps(B.upload_verified(root / "capture", "capture")), flush=True)
    assert max(seconds) / 3600 <= args.max_capture_hours, "measured capture exceeds width budget"
    child_wave(args, slots, "capture")
    assemble_store(root, rows)
    # Valuable capture is durable before the fitting phase begins.
    capture_upload = B.upload_verified(root / "store", "store")
    B.write_json(root / "capture_verified.json", capture_upload)
    evidence = fits(args, rows, meta)
    # Retain runtime logs and capture timing/provenance alongside the results.
    import shutil

    report = root / "run_report"
    report.mkdir(exist_ok=True)
    for path in [
        root / "assignment.json",
        root / "pilot_summary.json",
        root / "capture_verified.json",
        *root.glob("worker-*.log"),
        *root.glob("capture/rank*/*.json"),
    ]:
        shutil.copyfile(path, report / path.relative_to(root).as_posix().replace("/", "__"))
    report_upload = B.upload_verified(report, "run_report")
    B.complete(args, {"capture": capture_upload, "results": evidence, "run_report": report_upload})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["all", "worker"], default="all")
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--worker-mode", choices=["pilot", "capture"])
    parser.add_argument("--rank", type=int)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-capture-hours", type=float, default=3)
    parser.add_argument("--ridge-block", type=int, default=50000)
    parser.add_argument("--fit-seed", type=int, default=190102)
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--n-null", type=int, default=200)
    parser.add_argument("--resume-cloud", action="store_true")
    args = parser.parse_args()
    (worker if args.phase == "worker" else controller)(args)


if __name__ == "__main__":
    main()
