"""Cached CPU covariance, fixed-direction inverse and prompt retrieval for #1739."""

from __future__ import annotations

import argparse
import gc
import json
import mmap
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()
import numpy as np
import torch
from scripts import issue1739_fixed_transfer as ft
from scripts.issue1739_covariance_ablation import load_table, sha256, write_json, WIDE_GRID
from scripts.issue1739_claim4_fold import group_bootstrap_rhos
from scripts.issue1739_jobd_r2aug import LMAX, _pool_zscored_dv
from explore_persona_space.experiments.issue_1739 import arms, fits
from explore_persona_space.experiments.issue_1739.constants import WHITEN_SHRINKAGE_GRID

REG_ARMS = ("raw_context", "context_covariance", "mapped_answer", *ft.NULL_ARMS)
FIX_ARMS = ("preimage", "mapped_answer", "context_native", "answer_direction_on_context")


def progress(args, phase, **extra):
    record = dict(phase=phase, time=time.time(), source_sha=args.source_sha, **extra)
    write_json(args.out / "progress.json", record)
    print(json.dumps(record), flush=True)


def arr(payload, key):
    return np.asarray(payload[key], dtype=np.float64)


def rank_errors(z, yc, u, s, vt):
    """Squared inverse errors for every TSVD rank, including rank0."""
    numerical_rank = int(np.sum(s > np.finfo(s.dtype).eps * len(s) * s[0]))
    coeff = (yc @ vt[:numerical_rank].T) / s[:numerical_rank]
    target = z @ u[:, :numerical_rank]
    increments = np.sum(coeff * coeff - 2 * coeff * target, axis=0)
    return np.r_[np.sum(z * z), np.sum(z * z) + np.cumsum(increments)]


def inverse_direction(u, s, vt, direction, rank):
    if rank < 1:
        raise ValueError("generic validation selected constant inverse: no direction exists")
    return u[:, :rank] @ ((vt[:rank] @ direction) / s[:rank])


def finish(args, directory, extra=None):
    files = {
        str(p.relative_to(directory)): sha256(p)
        for p in sorted(directory.rglob("*"))
        if p.is_file() and p.name != "complete.json"
    }
    write_json(
        directory / "complete.json",
        dict(
            source_sha=args.source_sha,
            finished_at=time.time(),
            artifact_sha256=files,
            **(extra or {}),
        ),
    )


def transforms(args):
    out = args.out / "transforms"
    out.mkdir(parents=True, exist_ok=True)
    progress(args, "covariance_eigendecomposition")
    cp = args.cache / "outputs/map/crossproducts.pt"
    if sha256(cp) != "c84c3d969c33ca3691f6a7fc1b480302175f218d1c64fbff1d22c139791d8a43":
        raise ValueError("context sufficient-statistic checkpoint changed")
    state = torch.load(cp, map_location="cpu", weights_only=False, mmap=True)
    assert state["rows"] == 963444
    sd, mu = arr(state, "xsd"), arr(state, "xmu")
    cov = sd[:, None] * arr(state, "gram") * sd[None, :] / (state["rows"] - 1)
    eig, q = np.linalg.eigh((cov + cov.T) / 2)
    if eig.min() < -1e-7 * eig.max():
        raise ValueError("context covariance is not PSD")
    eig = np.maximum(eig, 0)
    split = np.load(args.cache / "map_inputs/split.npz")
    x = np.load(args.cache / "map_inputs/X.npy", mmap_mode="r")
    xv = x[split["val"]].astype(np.float64)
    hold_diag = np.mean(((xv - mu) @ q) ** 2, axis=0)
    candidates = []
    for gamma in WHITEN_SHRINKAGE_GRID:
        lam = (1 - gamma) * eig + gamma * eig.mean()
        candidates.append(
            dict(gamma=float(gamma), nll=float(np.log(lam).sum() + (hold_diag / lam).sum()))
        )
    gamma = min(candidates, key=lambda r: r["nll"])["gamma"]
    w = (q * ((1 - gamma) * eig + gamma * eig.mean()) ** -0.5) @ q.T
    np.savez(out / "covariance.npz", mean=mu, weight=w, eigenvalues=eig, gamma=gamma)
    write_json(
        out / "covariance.json",
        dict(
            n_train=state["rows"],
            n_validation=len(xv),
            shrinkage_candidates=candidates,
            selected_gamma=gamma,
            selected_grid_endpoint=gamma
            in (min(WHITEN_SHRINKAGE_GRID), max(WHITEN_SHRINKAGE_GRID)),
            answers_used=False,
            checkpoint_sha256=sha256(cp),
            denominator=state["rows"] - 1,
        ),
    )
    del cov, q, w, state
    gc.collect()
    progress(args, "inverse_svd")
    payload = ft.load_payload(args.cache / "outputs/map/frozen.pt")
    u, s, vt = np.linalg.svd(arr(payload, "W"), full_matrices=False)
    y = np.load(args.cache / "map_inputs/Y.npy", mmap_mode="r")
    zv = (xv - arr(payload, "xmu")) / arr(payload, "xsd")
    val_err = rank_errors(zv, y[split["val"]].astype(np.float64) - arr(payload, "ymu"), u, s, vt)
    rank = int(np.argmin(val_err))
    zt = (x[split["test"]].astype(np.float64) - arr(payload, "xmu")) / arr(payload, "xsd")
    yt = y[split["test"]].astype(np.float64) - arr(payload, "ymu")
    test_err = rank_errors(zt, yt, u, s, vt)
    denom = float(np.sum((zt - zt.mean(0)) ** 2))
    np.savez(out / "inverse.npz", u=u, s=s, vt=vt, rank=rank, val_sse=val_err, test_sse=test_err)
    write_json(
        out / "inverse.json",
        dict(
            selected_rank=rank,
            numerical_rank=len(val_err) - 1,
            selection="actual generic validation answers reconstruct standardized contexts",
            validation_r2_vs_zero=float(1 - val_err[rank] / val_err[0]),
            test_r2=float(1 - test_err[rank] / denom),
            test_mean_baseline_r2=float(1 - test_err[0] / denom),
            smallest_singular_value=float(s[-1]),
            largest_singular_value=float(s[0]),
            inverse_available=rank > 0,
        ),
    )
    finish(args, out)


def verify_previous(directory):
    done = json.loads((directory / "complete.json").read_text())
    for name, digest in done["artifact_sha256"].items():
        if sha256(directory / name) != digest:
            raise ValueError(f"cached result changed: {directory / name}")


def prepare_behavior(args, behavior, index):
    """Freeze prior evaluation IDs; preserve legacy TRAIN budget IDs, then align rollouts."""
    previous = args.cache / "outputs/analysis" / behavior
    verify_previous(previous)
    old = dict(np.load(previous / "predictions.npz"))
    heldout_hash = set()
    for row in json.loads((previous / "evaluation_hashes.json").read_text()):
        heldout_hash |= ft.content_hash_set(row, "normalized")
    labels_dir = args.repo / "eval_results/issue_1739"
    paths = [
        labels_dir / "dv_dataset" / behavior / "labeling.json",
        labels_dir / "wildchat_rung/dv_dataset" / behavior / "labeling.json",
    ]
    old_result = json.loads((previous / "results.json").read_text())
    expected = set(old_result["input_label_sha256"].values())
    assert all(sha256(p) in expected for p in paths)
    tr = load_table(args.cache / "inputs" / f"{behavior}_labeling", paths[0], 19, "config_a")
    wc = load_table(args.cache / "inputs/wildchat", paths[1], 19, "config_b")
    budget = fits.realize_budget_cell(tr.groups, budget_l=LMAX[behavior], draw=0, seed=0)
    historical_ids = [
        [tr.ctx_order[i] for i in budget.row_idx],
        np.asarray(wc.ctx_order)[~ft._wc_eval_mask(wc.ctx_order)].tolist(),
    ]
    del tr, wc
    supplementary = None
    if behavior == "hallucination":
        supplementary = {
            str(r["context_id"]): r
            for r in ft.json_rows(args.cache / "inputs/provenance/hallucination_per_rollout.json")
        }
    ev_batches, train_batches, dropped = [], [], []
    for ns, path, train_ids in zip(
        (f"{behavior}_labeling", "wildchat"), paths, historical_ids, strict=True
    ):
        labels = {str(r["context_id"]): r for r in ft.json_rows(path)}
        eval_ids = [str(cid) for cid in old["context_ids"] if str(cid) in labels]
        retained = []
        for cid in train_ids:
            hashes = ft.content_hash_set(ft.lookup_prompt(index, ns, cid), "normalized")
            if hashes & heldout_hash:
                dropped.append(
                    dict(namespace=ns, context_id=cid, reason="readout_train_eval_content_overlap")
                )
            else:
                retained.append(cid)
        arrays, meta = ft.load_store(args.cache / "inputs" / ns)
        ft.validate_capture_provenance(meta, index, ns, context_ids=retained + eval_ids)
        extra = supplementary if ns != "wildchat" else None
        train_batches.append(
            ft.reduce_eval(arrays, meta, [labels[cid] for cid in retained], supplementary=extra)
        )
        ev_batches.append(
            ft.reduce_eval(arrays, meta, [labels[cid] for cid in eval_ids], supplementary=extra)
        )
        del arrays, meta
    data = {
        k: np.concatenate([b[k] for b in ev_batches])
        for k in ("x", "y", "dv", "context_ids", "groups", "rungs")
    }
    order = {str(cid): i for i, cid in enumerate(data["context_ids"])}
    take = [order[str(cid)] for cid in old["context_ids"]]
    data = {k: v[take] for k, v in data.items()}
    for key in ("dv", "context_ids", "groups", "rungs"):
        np.testing.assert_array_equal(data[key], old[key])
    n0 = len(train_batches[0]["dv"])
    train = {k: np.concatenate([b[k] for b in train_batches]) for k in ("x", "dv", "context_ids")}
    train["target"] = _pool_zscored_dv(train["dv"], np.arange(n0), np.arange(n0, len(train["dv"])))
    audit = dict(
        historical_training_counts=[len(x) for x in historical_ids],
        realized_training_counts=[len(b["dv"]) for b in train_batches],
        dropped=dropped,
        training_ids=train["context_ids"].tolist(),
        evaluation_n=len(data["dv"]),
        previous_results_sha256=sha256(previous / "results.json"),
        input_labels_sha256={str(p): sha256(p) for p in paths},
        train_rollout_audit=[r for b in train_batches for r in b["rollout_audit"]],
    )
    return data, train, old, audit


def finite_float(value):
    return float(value) if np.isfinite(value) else None


def summarize(pred, data, names, n_boot, comparisons):
    result, boot_save = [], {}
    for rung in sorted(set(data["rungs"])):
        ix = np.flatnonzero(data["rungs"] == rung)
        active = np.isfinite(pred[:, ix]).all(axis=1)
        rho = np.full(len(names), np.nan)
        boots = np.full((len(names), n_boot), np.nan)
        ng = len(set(data["groups"][ix]))
        if active.any():
            rho[active] = arms.spearman_rows(pred[active][:, ix], data["dv"][ix])
            boots[active], ng = group_bootstrap_rhos(
                pred[active][:, ix],
                data["dv"][ix],
                data["groups"][ix],
                n_boot=n_boot,
                rng=np.random.default_rng(1739963),
            )

        def ci(v):
            valid = np.isfinite(v)
            return np.quantile(v[valid], [0.025, 0.975]).tolist() if valid.any() else None

        estimates = {
            name: dict(
                rho=float(rho[i]) if np.isfinite(rho[i]) else None,
                ci95=ci(boots[i]),
                valid_bootstrap_draws=int(np.isfinite(boots[i]).sum()),
            )
            for i, name in enumerate(names)
        }
        diffs = {}
        for a, b in comparisons:
            ia, ib = names.index(a), names.index(b)
            diffs[a + "_minus_" + b] = dict(
                delta=finite_float(rho[ia] - rho[ib]), ci95=ci(boots[ia] - boots[ib])
            )
        if set(ft.NULL_ARMS) <= set(names):
            indices = [names.index(n) for n in ft.NULL_ARMS]
            estimates["shuffled_mean"] = dict(
                rho=finite_float(rho[indices].mean()),
                ci95=ci(boots[indices].mean(0)),
                seed_rhos=[finite_float(v) for v in rho[indices]],
                seed_sd=finite_float(rho[indices].std(ddof=1)),
            )
            a = names.index("mapped_answer")
            diffs["mapped_answer_minus_shuffled_mean"] = dict(
                delta=finite_float(rho[a] - rho[indices].mean()),
                ci95=ci(boots[a] - boots[indices].mean(0)),
            )
        result.append(
            dict(
                rung=str(rung),
                n=len(ix),
                n_groups=ng,
                informative=len(ix) >= 30,
                arms=estimates,
                differences=diffs,
            )
        )
        boot_save[str(rung)] = boots
    return result, boot_save


def extreme_ids(scores, ids, count):
    # Stable context-ID tiebreaking, never resolve a tie with the behavior labels.
    return np.lexsort((ids.astype(str), scores))[:count], np.lexsort((ids.astype(str), -scores))[
        :count
    ]


def behavior_analysis(args, behavior):
    out = args.out / behavior
    out.mkdir(parents=True, exist_ok=True)
    progress(args, "behavior_inputs", behavior=behavior)
    index = ft.load_prompt_index(args.cache / "inputs/provenance/prompt_index.jsonl")
    data, train, old, audit = prepare_behavior(args, behavior, index)
    write_json(out / "training_audit.json", audit)
    previous = args.cache / "outputs/analysis" / behavior
    d = np.load(previous / "directions.npz")
    va, vc = d["answer"], d["context"]
    payload = ft.load_payload(args.cache / "outputs/map/frozen.pt")
    parity = np.stack(
        [data["y"] @ va, ft.map_projection(payload, data["x"], va), data["x"] @ vc, data["x"] @ va]
    )
    np.testing.assert_allclose(parity, old["predictions"][:4], rtol=1e-10, atol=1e-7)
    del data["y"], index
    wh = np.load(args.out / "transforms/covariance.npz")
    reg, selections = [], {}
    for name in REG_ARMS:
        progress(args, "behavior_ridge", behavior=behavior, arm=name)
        if name == "raw_context":
            x, ev = train["x"], data["x"]
        elif name == "context_covariance":
            x, ev = [(a - wh["mean"]) @ wh["weight"] for a in (train["x"], data["x"])]
        else:
            map_path = (
                "frozen.pt"
                if name == "mapped_answer"
                else name.replace("shuffled", "shuffle") + ".pt"
            )
            p = ft.load_payload(args.cache / "outputs/map" / map_path)
            x, ev = [
                ((a - arr(p, "xmu")) / arr(p, "xsd")) @ arr(p, "W") + arr(p, "ymu")
                for a in (train["x"], data["x"])
            ]
        selected = []
        with fits.capture_selected_lambdas(selected):
            pred = fits.ridge_gcv_predict_per_target(
                x[None],
                train["target"][None, :, None],
                [ev[None]],
                lambdas=WIDE_GRID,
                device="cpu",
                layer_chunk=1,
            )[0][0, :, 0]
        reg.append(pred)
        selections[name] = selected
        np.save(out / f"regression_{name}.npy", pred)
        write_json(out / "ridge_selection.json", selections)
        del x, ev
        gc.collect()
    reg = np.stack(reg)
    progress(args, "behavior_bootstrap", behavior=behavior)
    summary, boots = summarize(
        reg,
        data,
        REG_ARMS,
        args.n_boot,
        [
            ("mapped_answer", "raw_context"),
            ("mapped_answer", "context_covariance"),
            ("context_covariance", "raw_context"),
        ],
    )
    np.savez(
        out / "regression_predictions.npz",
        predictions=reg,
        arms=np.asarray(REG_ARMS),
        **{k: data[k] for k in ("dv", "context_ids", "groups", "rungs")},
    )
    np.savez(out / "regression_bootstraps.npz", **boots)
    write_json(
        out / "regression.json",
        dict(results=summary, n_boot=args.n_boot, selected_lambdas=selections),
    )
    inv = np.load(args.out / "transforms/inverse.npz")
    rank = int(inv["rank"])
    numerical_rank = len(inv["val_sse"]) - 1
    u = (
        inverse_direction(inv["u"], inv["s"], inv["vt"], va, rank)
        if rank > 0
        else np.full_like(va, np.nan)
    )
    weights = np.stack(
        [u, arr(payload, "W") @ va, arr(payload, "xsd") * vc, arr(payload, "xsd") * va]
    )
    z = (data["x"] - arr(payload, "xmu")) / arr(payload, "xsd")
    dot = z @ weights.T
    cosine = dot / (np.linalg.norm(z, axis=1)[:, None] * np.linalg.norm(weights, axis=1)[None, :])
    diag_ranks = sorted(
        {
            k
            for k in (max(1, rank // 2), rank, min(numerical_rank, 2 * rank), numerical_rank)
            if k > 0
        }
    )
    rank_weights = np.stack(
        [inverse_direction(inv["u"], inv["s"], inv["vt"], va, k) for k in diag_ranks]
    )
    sensitivity = z @ rank_weights.T
    fixed_names = FIX_ARMS + tuple("inverse_rank" + str(k) for k in diag_ranks) + ("real_answer",)
    fixed = np.vstack([dot.T, sensitivity.T, old["predictions"][0]])
    summary, boots = summarize(
        fixed,
        data,
        fixed_names,
        args.n_boot,
        [
            ("preimage", "mapped_answer"),
            ("preimage", "context_native"),
            ("mapped_answer", "context_native"),
        ],
    )
    np.savez(
        out / "fixed_predictions.npz",
        predictions=fixed,
        cosine=cosine.T,
        arms=np.asarray(fixed_names),
        **{k: data[k] for k in ("dv", "context_ids", "groups", "rungs")},
    )
    np.savez(out / "fixed_bootstraps.npz", **boots)
    np.savez(out / "retrieval_weights.npz", weights=weights, arms=np.asarray(FIX_ARMS))
    write_json(
        out / "fixed.json",
        dict(
            results=summary,
            selected_rank=rank,
            diagnostic_ranks=diag_ranks,
            inverse_available=rank > 0,
            inverse_norm=finite_float(np.linalg.norm(u)),
            target_residual_fraction=finite_float(
                np.linalg.norm(u @ arr(payload, "W") - va) / np.linalg.norm(va)
            ),
            preserved_score_max_error=float(np.max(np.abs(parity - old["predictions"][:4]))),
            pooling_caveat=ft.POOLING_CAVEAT,
        ),
    )
    deciles, examples = [], []
    for rung in sorted(set(data["rungs"])):
        ix = np.flatnonzero(data["rungs"] == rung)
        if len(ix) < 30:
            continue
        for metric, scores in [("dot", dot), ("cosine", cosine)]:
            for j, name in enumerate(FIX_ARMS):
                if not np.isfinite(scores[ix, j]).all():
                    continue
                lo, hi = extreme_ids(scores[ix, j], data["context_ids"][ix], max(1, len(ix) // 10))
                dv = data["dv"][ix]
                deciles.append(
                    dict(
                        rung=str(rung),
                        method=name,
                        metric=metric,
                        n=len(ix),
                        n_tail=len(lo),
                        bottom_mean_dv=float(dv[lo].mean()),
                        top_mean_dv=float(dv[hi].mean()),
                        delta=float(dv[hi].mean() - dv[lo].mean()),
                        overall_mean_dv=float(dv.mean()),
                    )
                )
                for side, rows in [("bottom", lo[:3]), ("top", hi[:3])]:
                    for position, r in enumerate(rows):
                        k = ix[r]
                        examples.append(
                            dict(
                                context_id=str(data["context_ids"][k]),
                                rung=str(rung),
                                method=name,
                                metric=metric,
                                side=side,
                                rank=position + 1,
                                score=float(scores[k, j]),
                                mean_dv=float(data["dv"][k]),
                            )
                        )
    write_json(out / "heldout_deciles.json", deciles)
    write_json(out / "heldout_examples_index.json", examples)
    finish(args, out)


def map_texts(args):
    ci = np.load(args.cache / "map_inputs/ci.npy", mmap_mode="r")
    with (args.cache / "inventory/passb_prompts_sha_verified.jsonl").open() as stream:
        for line in stream:
            row = json.loads(line)
            yield row["i"], row["prompt"], "lmsys"
    j = 5000
    for path in sorted(args.manifest.glob("part_*.jsonl")):
        with path.open() as stream:
            for line in stream:
                row = json.loads(line)
                if j < len(ci) and row["i"] == ci[j]:
                    yield j, row["prompt"], row["corpus"]
                    j += 1
    if j != len(ci):
        raise ValueError("incomplete map text join")


def retrieval(args):
    out = args.out / "retrieval"
    out.mkdir(parents=True, exist_ok=True)
    payload = ft.load_payload(args.cache / "outputs/map/frozen.pt")
    rows = np.load(args.cache / "map_inputs/split.npz")["train"]
    x = np.load(args.cache / "map_inputs/X.npy", mmap_mode="r")
    all_w = np.concatenate(
        [np.load(args.out / b / "retrieval_weights.npz")["weights"] for b in ft.ROSTER]
    )
    scores = np.lib.format.open_memmap(
        out / "generic_scores.npy", mode="w+", dtype=np.float64, shape=(len(rows), 2, len(all_w))
    )
    norms = np.linalg.norm(all_w, axis=1)
    for start in range(0, len(rows), 4096):
        z = (x[rows[start : start + 4096]].astype(np.float64) - arr(payload, "xmu")) / arr(
            payload, "xsd"
        )
        dot = z @ all_w.T
        scores[start : start + len(z), 0] = dot
        scores[start : start + len(z), 1] = dot / (
            np.linalg.norm(z, axis=1)[:, None] * norms[None, :]
        )
        if start % (4096 * 16) == 0:
            scores.flush()
            x._mmap.madvise(mmap.MADV_DONTNEED)
            progress(
                args, "retrieval_projection", rows=min(start + 4096, len(rows)), total=len(rows)
            )
    scores.flush()
    # Normalize full prompt hashes before selecting extrema so duplicates cannot dominate.
    hashes = np.load(args.cache / "map_inputs/map_prompt_hashes.npz")["normalized_sha256"]
    if len(hashes) != len(rows):
        raise ValueError("map hash bank does not index training rows")
    wanted = {}
    for bidx, b in enumerate(ft.ROSTER):
        for j, name in enumerate(FIX_ARMS):
            if not np.isfinite(all_w[bidx * 4 + j]).all():
                continue
            for mi, metric in enumerate(("dot", "cosine")):
                for sign, side in ((1, "bottom"), (-1, "top")):
                    order = np.lexsort((rows, sign * scores[:, mi, bidx * 4 + j]))
                    seen = set()
                    records = []
                    for local in order:
                        row = int(rows[local])
                        digest = bytes(hashes[local])
                        if digest in seen:
                            continue
                        seen.add(digest)
                        rec = dict(
                            row=row,
                            normalized_sha256=digest.decode(),
                            score=float(scores[local, mi, bidx * 4 + j]),
                            rank=len(records) + 1,
                            behavior=b,
                            method=name,
                            metric=metric,
                            side=side,
                        )
                        wanted.setdefault(row, []).append(rec)
                        records.append(rec)
                        if len(records) == 30:
                            break
    progress(args, "retrieval_text_join", unique_rows=len(wanted))
    joined = 0
    for row, text, corpus in map_texts(args):
        if row in wanted:
            expected = wanted[row][0]["normalized_sha256"]
            if ft.prompt_hashes(text)[1] != expected:
                raise ValueError("retrieved prompt hash mismatch")
            for rec in wanted[row]:
                rec.update(prompt=text, corpus=corpus)
            joined += 1
    assert joined == len(wanted)
    flat = [r for values in wanted.values() for r in values]
    write_json(out / "generic_examples.json", flat)
    np.save(out / "generic_score_rows.npy", rows)
    finish(
        args,
        out,
        dict(
            n_pool=len(rows),
            unique_retrieved_prompts=joined,
            primary_metric="centered standardized cosine",
            interpretation="training-pool descriptive retrieval",
        ),
    )


def heldout_text(args):
    for b in ft.ROSTER:
        out = args.out / b
        examples = json.loads((out / "heldout_examples_index.json").read_text())
        previous = args.cache / "outputs/analysis" / b
        records = {
            r["context_id"]: r
            for r in json.loads((previous / "evaluation_hashes.json").read_text())
        }
        valid = {
            r["context_id"]: set(r["rollout_k"])
            for r in json.loads((previous / "rollout_alignment.json").read_text())
        }
        wanted = {r["context_id"] for r in examples}
        docs = {cid: [] for cid in wanted}
        for ns in (f"{b}_labeling", "wildchat"):
            selected = {cid for cid in wanted if records[cid]["namespace"] == ns}
            shards = sorted({p for cid in selected for p in records[cid]["source_shards"]})
            base = (
                args.cache
                / "inputs/raw_metadata"
                / ("wildchat" if ns == "wildchat" else "original")
            )
            for shard in shards:
                with (base / shard).open() as stream:
                    for line in stream:
                        packed = json.loads(line)
                        d = packed["doc"]
                        cid = d.get("context_id")
                        if cid not in selected or d.get("rollout_k") not in valid[cid]:
                            continue
                        source = next(
                            r for r in records[cid]["rollouts"] if r["rollout_k"] == d["rollout_k"]
                        )
                        assert packed["src"] == source["packed_src"]
                        docs[cid].append(
                            {
                                k: d[k]
                                for k in (
                                    "rollout_k",
                                    "query",
                                    "prompt_text",
                                    "completion",
                                    "finish_reason",
                                )
                            }
                        )
        for cid, found in docs.items():
            assert {d["rollout_k"] for d in found} == valid[cid]
        write_json(out / "heldout_examples_text.json", dict(selections=examples, contexts=docs))
        finish(args, out)
        progress(args, "heldout_text_complete", behavior=b, contexts=len(wanted))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=["transforms", "behavior", "retrieval", "text"])
    p.add_argument("--behavior", choices=list(ft.ROSTER))
    p.add_argument("--cache", type=Path, default=Path("/dev/shm/issue1739-fixed-transfer"))
    p.add_argument(
        "--repo", type=Path, default=Path("/home/thomasjiralerspong/explore-persona-space")
    )
    p.add_argument(
        "--manifest",
        type=Path,
        default=Path(
            "/mnt/eps-data/thomasjiralerspong/issue1895_inputside/scratch/sampling_manifest"
        ),
    )
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--source-sha", required=True)
    p.add_argument("--n-boot", type=int, default=2000)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(8)
    manifest = json.loads((args.cache / "outputs/map/map_manifest.json").read_text())
    assert manifest["complete"] and manifest["n_train"] == 963444
    for rec in manifest["artifacts_sha256"].values():
        if sha256(Path(rec["path"])) != rec["sha256"]:
            raise ValueError("cached map artifact changed")
    if args.phase == "transforms":
        transforms(args)
    elif args.phase == "behavior":
        behavior_analysis(args, args.behavior)
    elif args.phase == "retrieval":
        retrieval(args)
    else:
        heldout_text(args)
    progress(args, args.phase + "_complete", behavior=args.behavior)


if __name__ == "__main__":
    main()
