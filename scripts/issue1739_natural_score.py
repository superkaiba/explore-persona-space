"""Natural prompt extremes in the original Figure 6 coordinate system.

Cached on-policy text/labels only. Each layer is consumed separately; the
original fp16 prompt aggregation, ADD pool, whitening and map fit are retained.
Directions use only eligible extraction prompts and their valid judged answers.
"""

from __future__ import annotations

# ruff: noqa: E402 -- source-root bootstrap precedes project imports
import gc
import hashlib
import json
import resource
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import numpy as np

from explore_persona_space.experiments.issue_1739 import arms, fits, store_io
from explore_persona_space.experiments.issue_1739 import natural_extremes as ne
from explore_persona_space.experiments.issue_1739.corpus_staging import near_dup_mask
from scripts.issue1739_fits import _load_rb_e1
from scripts.issue1739_r2v2_score import _group_side_train
from scripts.issue1739_result2fair_score import _wc_eval_mask

LAYERS = {
    "evil": {
        "answer_direction_on_context": 15,
        "context_native": 15,
        "mapped_answer": 22,
        "real_answer": 22,
    },
    "sycophancy": {
        "answer_direction_on_context": 16,
        "context_native": 16,
        "mapped_answer": 11,
        "real_answer": 11,
    },
    "hallucination": {
        "answer_direction_on_context": 24,
        "context_native": 24,
        "mapped_answer": 23,
        "real_answer": 27,
    },
}
KINDS = ("context_end", "t1")


def write_json(path, value):
    """Atomically publish strict JSON, including checkpoints and progress."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def sha(path):
    """Hash a file without loading it into memory."""
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def progress(config, phase, **extra):
    """Record a timestamped phase transition and measured process resources."""
    value = {
        "source_sha": config["source_sha"],
        "time": time.time(),
        "phase": phase,
        "max_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20,
        **extra,
    }
    write_json(Path(config["out_root"]) / "progress.json", value)
    print(json.dumps(value), flush=True)


def load_table(path, labels_path, layer, split):
    """Reproduce historical context/answer pooling with only consumed kinds.

    Historical observed-answer means include all cached responses, including
    responses whose judges dropped. Natural extraction below separately uses
    valid judged responses. Preserve this distinction in the saved metadata.
    """
    labels = {
        r["context_id"]: r
        for r in json.loads(Path(labels_path).read_text())["rows"]
        if r.get("dv") is not None and r["split"] in split
    }
    arrays, meta = store_io.load_summaries(Path(path), KINDS, (layer,), hidden_dim=3584)
    positions = {}
    pairs = set()
    for i, row in enumerate(meta):
        cid = row["context_id"]
        pair = (cid, int(row["rollout_k"]))
        if pair in pairs:
            raise ValueError(f"Duplicate response activation: {pair}")
        pairs.add(pair)
        if cid in labels:
            positions.setdefault(cid, []).append(i)
    if not positions:
        raise ValueError(f"No joined contexts: {path}/{labels_path}/{split}")
    ids = list(positions)
    for cid in ids:
        draws = {int(meta[i]["rollout_k"]) for i in positions[cid]}
        if draws != set(range(5)):
            raise ValueError(f"Incomplete original five-response capture: {cid}: {draws}")
    x = arrays[("context_end", layer)][[positions[c][0] for c in ids]]
    y = np.empty_like(x)
    # Vectorize equal-size response groups, retain original row order and fp16
    # mean accumulation semantics. Different draw counts remain distinct groups.
    lengths = np.array([len(positions[c]) for c in ids])
    for count in sorted(set(lengths)):
        ii = np.flatnonzero(lengths == count)
        for start in range(0, len(ii), 256):
            chunk = ii[start : start + 256]
            gather = np.array([positions[ids[i]] for i in chunk])
            y[chunk] = arrays[("t1", layer)][gather].mean(axis=1)
    rows = [labels[c] for c in ids]
    if any(not r.get("group_key") for r in rows):
        raise ValueError(f"Missing group identity: {path}")
    return SimpleNamespace(
        x=x,
        y=y,
        ids=ids,
        rows=rows,
        dv=np.array([r["dv"] for r in rows], dtype=float),
        groups=[str(r["group_key"]) for r in rows],
        rungs=[str(r["rung"]) for r in rows],
    )


def merge_tables(tables):
    """Concatenate disjoint stores without changing their historical row order."""
    ids = [c for t in tables for c in t.ids]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate context IDs across capture stores")
    return SimpleNamespace(
        x=np.concatenate([t.x for t in tables]),
        y=np.concatenate([t.y for t in tables]),
        ids=ids,
        rows=[r for t in tables for r in t.rows],
        dv=np.concatenate([t.dv for t in tables]),
        groups=[g for t in tables for g in t.groups],
        rungs=[r for t in tables for r in t.rungs],
    )


def evaluation_roster(table):
    """Original whole-dataset OOD, held-in train, and WildChat evaluation rows."""
    roster = {}
    rungs = np.asarray(table.rungs)
    for rung in sorted(set(rungs) - {"train", "wildchat_rung"}):
        roster[rung] = np.flatnonzero(rungs == rung)
    train = np.flatnonzero(rungs == "train")
    roster["heldin:train"] = np.array(
        [i for i in train if not _group_side_train("train", table.groups[i], 0, 0.8)], dtype=int
    )
    wc = np.flatnonzero(rungs == "wildchat_rung")
    roster["wildchat_rung"] = wc[_wc_eval_mask([table.ids[i] for i in wc])]
    return roster


def assert_label_context_coverage(table, label_paths):
    """Require every finite-DV context after partitioned stores are merged."""
    expected_ids = set()
    for path in label_paths:
        expected_ids.update(
            r["context_id"]
            for r in json.loads(Path(path).read_text())["rows"]
            if r.get("dv") is not None
        )
    if set(table.ids) != expected_ids:
        raise ValueError(
            f"Full label/capture coverage mismatch: missing={len(expected_ids - set(table.ids))}"
        )


def unique_extraction_prompts(mask, signatures, ids):
    """Deterministically retain one prompt per inherited MinHash band collision."""
    mask = mask.copy()
    buckets, witnesses = {}, []
    for i in sorted(np.flatnonzero(mask), key=lambda i: ids[i]):
        bands = [(b, signatures[i, b * 4 : (b + 1) * 4].tobytes()) for b in range(16)]
        witness = next((buckets[b] for b in bands if b in buckets), None)
        if witness is not None:
            mask[i] = False
            witnesses.append({"excluded": ids[i], "retained": ids[witness]})
        else:
            for b in bands:
                buckets[b] = i
    return mask, witnesses


def build_selections(config, table, metadata):  # noqa: C901 -- audited variant construction
    """Freeze eligible train-only tails before activation scoring or evaluation."""
    behavior = config["behavior"]
    roster = evaluation_roster(table)
    n = len(table.ids)
    score_rows = metadata["scores"]
    scores = np.full((n, 5), np.nan)
    hashes, groups, eligible = [], [], []
    for i, (cid, row) in enumerate(zip(table.ids, table.rows, strict=True)):
        info = metadata["contexts"].get(cid)
        if info is None:
            raise ValueError(f"Missing complete-prompt provenance: {cid}")
        if not isinstance(info["natural_eligible"], bool):
            raise ValueError(f"Natural eligibility must be boolean: {cid}")
        if info["natural_eligible"] and cid not in score_rows:
            raise ValueError(f"Missing archived score record for eligible prompt: {cid}")
        hashes.append(info["question_normalized_sha256"])
        groups.append(f"{row['rung']}:{row['group_key']}")
        eligible.append(info["natural_eligible"])
        for key, value in score_rows.get(cid, {}).items():
            if value is not None:
                scores[i, int(key.removeprefix("k"))] = float(value)
    eligible = np.asarray(eligible, dtype=bool)
    if behavior == "hallucination":
        # Cached per-rollout file is binary 0/100; preserve the DV's 0/1 scale.
        scores /= 100
    native_range = (0.0, 1.0 if behavior == "hallucination" else 100.0)
    extraction_rungs = {"hhrt", "toxicchat"} if behavior == "evil" else {"train"}
    groups = np.asarray(groups)
    hashes = np.asarray(hashes)
    signatures = np.asarray([metadata["signatures"][cid] for cid in table.ids], dtype=np.uint64)
    base = (
        np.array(
            [
                r in extraction_rungs and _group_side_train(r, g, 0, 0.8)
                for r, g in zip(table.rungs, table.groups, strict=True)
            ]
        )
        & eligible
    )
    # Evaluation content is protected before any label-based ranking. This
    # deliberately excludes an extraction candidate rather than deleting eval.
    protected = np.unique(
        np.concatenate([rows for name, rows in roster.items() if name not in extraction_rungs])
    )
    base &= ~np.isin(hashes, hashes[protected])
    base_rows = np.flatnonzero(base)
    base[base_rows[near_dup_mask(signatures[protected], signatures[base_rows])]] = False
    protected_turns = {
        h for i in protected for h in metadata["contexts"][table.ids[i]]["user_turn_hashes"]
    }
    base &= ~np.isin(hashes, list(protected_turns))
    # Duplicate prompt copies receive one vote; deterministic ID selects the
    # representative. Keep same source-group membership throughout.
    seen = set()
    for i in sorted(np.flatnonzero(base), key=lambda i: table.ids[i]):
        if hashes[i] in seen:
            base[i] = False
        seen.add(hashes[i])
    base, internal_witnesses = unique_extraction_prompts(base, signatures, table.ids)
    selections, records = (
        {},
        {
            "_dedup": {
                "internal_near_duplicate_witnesses": internal_witnesses,
                "eligible_extraction_remaining": int(base.sum()),
            }
        },
    )
    for dataset, eval_rows in roster.items():
        pool = base & (np.asarray(table.rungs) != dataset)
        eval_mask = np.zeros(n, bool)
        eval_mask[eval_rows] = True
        pool &= ~np.isin(hashes, hashes[eval_rows])
        pool_rows = np.flatnonzero(pool)
        pool[pool_rows[near_dup_mask(signatures[eval_rows], signatures[pool_rows])]] = False
        eval_turns = {
            h for i in eval_rows for h in metadata["contexts"][table.ids[i]]["user_turn_hashes"]
        }
        pool &= ~np.isin(hashes, list(eval_turns))
        pool &= ~np.isin(groups, groups[eval_rows])
        common = dict(
            context_ids=table.ids,
            group_ids=groups,
            prompt_hashes=hashes,
            extraction_mask=pool,
            eligible_mask=eligible,
            evaluation_mask=eval_mask,
            score_range=native_range,
        )
        choices = [
            (f"q{int(q * 100):02d}_s{salt}", q, str(salt), 3)
            for q in (0.01, 0.05, 0.1)
            for salt in range(5 if q == 0.01 else 1)
        ]
        choices.append(("q01_complete", 0.01, "0", 5))
        for name, q, salt, minimum in choices:
            key = f"{dataset}/{name}"
            try:
                selection = ne.select_prompt_tails(
                    scores,
                    **common,
                    behavior=behavior,
                    fold=dataset,
                    q=q,
                    tie_salt=salt,
                    min_valid=minimum,
                )
            except ne.SelectionError as exc:
                records[key] = {
                    "status": "unavailable",
                    "reason": str(exc),
                    "details": exc.metadata,
                }
                continue
            selections[key] = selection
            records[key] = {
                "status": "ok",
                **selection.metadata,
                "high_ids": [table.ids[i] for i in selection.high_indices],
                "low_ids": [table.ids[i] for i in selection.low_indices],
            }
            if name == "q01_s0":
                reliability = {}
                for orientation, held in (
                    ("3_to_2", np.array([False] * 3 + [True] * 2)),
                    ("2_to_3", np.array([True] * 3 + [False] * 2)),
                ):
                    ranking_scores = scores.copy()
                    ranking_scores[:, held] = np.nan
                    reliability_common = {
                        **common,
                        "extraction_mask": pool & (np.isfinite(scores).sum(axis=1) >= 3),
                    }
                    try:
                        heldout_selection = ne.select_prompt_tails(
                            ranking_scores,
                            **reliability_common,
                            behavior=behavior,
                            fold=dataset,
                            q=0.01,
                            tie_salt="response_holdout",
                            min_valid=2,
                        )
                        reliability[orientation] = ne.heldout_score_gap(
                            heldout_selection, scores, draw_mask=held
                        )
                    except ne.SelectionError as exc:
                        reliability[orientation] = {"status": "unavailable", "reason": str(exc)}
                records[key]["response_split_reliability"] = reliability
                try:
                    halves = ne.split_half_weights(
                        selection, group_ids=groups, behavior=behavior, fold=dataset
                    )
                    for half, weights in enumerate(halves):
                        selections[f"{dataset}/half{half}"] = weights
                except ne.SelectionError as exc:
                    records[key]["half_stability_unavailable"] = str(exc)
        try:
            endpoint = ne.select_literal_endpoints(scores, **common)
            selections[f"{dataset}/endpoints"] = endpoint
            records[f"{dataset}/endpoints"] = {"status": "ok", **endpoint.metadata}
        except ne.SelectionError as exc:
            records[f"{dataset}/endpoints"] = {
                "status": "unavailable",
                "reason": str(exc),
                "details": exc.metadata,
            }
        # Historical within-prompt and pooled midpoint controls on the same
        # natural extraction pool; NaNs off-pool cannot acquire any weight.
        for name, pooled in (("e2", False), ("e2p", True)):
            control_pool = pool & (np.isfinite(scores).sum(axis=1) >= 3)
            subset = scores[control_pool]
            try:
                hi_sub, lo_sub, _ = fits.matched_pair_split_weights(
                    subset, spread_min=0.15 if behavior == "hallucination" else 15, pooled=pooled
                )
            except ValueError as exc:
                records[f"{dataset}/{name}"] = {"status": "unavailable", "reason": str(exc)}
                continue
            hi, lo = np.zeros_like(scores), np.zeros_like(scores)
            hi[control_pool], lo[control_pool] = hi_sub, lo_sub
            count = int(np.any((hi + lo) != 0, axis=1).sum())
            selections[f"{dataset}/{name}"] = ne.TailWeights(
                hi,
                lo,
                np.flatnonzero(hi.sum(1)),
                np.flatnonzero(lo.sum(1)),
                {"n_qualifying": int(count)},
            )
            records[f"{dataset}/{name}"] = {"status": "ok", "n_qualifying": int(count)}
    return roster, selections, records


def stream_directions(stores, layer, ids, selections):
    """Reduce all direction variants together through one shard pass per kind."""
    keys = sorted(selections)
    position = {c: i for i, c in enumerate(ids)}
    weights = np.stack([selections[k].high_weights - selections[k].low_weights for k in keys])
    # Use the union of high/low support, including draws whose signed weights
    # cancel in a control. A missing response must not silently change a mean.
    support = np.stack([selections[k].high_weights + selections[k].low_weights for k in keys])
    expected = {(ids[i], int(k)) for i, k in np.argwhere(np.any(support != 0, axis=0))}
    consumed = set()
    pairs = set()
    result = {kind: np.zeros((len(keys), 3584)) for kind in KINDS}
    for store in stores:
        meta = store_io._index_rows_for(Path(store), list(KINDS))
        w = np.zeros((len(keys), len(meta)))
        for j, row in enumerate(meta):
            cid, draw = row["context_id"], int(row["rollout_k"])
            if (cid, draw) in pairs:
                raise ValueError(f"Duplicate response activation: {cid}/{draw}")
            pairs.add((cid, draw))
            if (cid, draw) in expected:
                consumed.add((cid, draw))
            if cid in position:
                w[:, j] = weights[:, position[cid], draw]
        for kind in KINDS:
            _, paths = store_io._resolve_summary_kind(Path(store), kind, layer)
            offset = 0
            for path in paths:
                array = np.load(path, mmap_mode="r")
                segment = w[:, offset : offset + len(array)]
                live = np.flatnonzero(np.any(segment, axis=0))
                if len(live):
                    result[kind] += segment[:, live] @ np.asarray(array[live], dtype=np.float64)
                offset += len(array)
            if offset != len(meta):
                raise ValueError(f"Activation/metadata row mismatch in {store}/{kind}/{layer}")
    if consumed != expected:
        raise ValueError(
            f"Missing selected response activations: {sorted(expected - consumed)[:10]}"
        )
    return keys, result


def direction_stability(keys, directions, roster, whitening):
    """Compare disjoint prompt halves and deterministic ties in scoring geometry."""
    stability = []
    key_index = {key: j for j, key in enumerate(keys)}
    for dataset in roster:
        pairs = [("half0", "half1")] + [("q01_s0", f"q01_s{s}") for s in range(1, 5)]
        for left, right in pairs:
            if f"{dataset}/{left}" not in key_index or f"{dataset}/{right}" not in key_index:
                continue
            for kind in KINDS:
                a = directions[kind][key_index[f"{dataset}/{left}"]] @ whitening
                b = directions[kind][key_index[f"{dataset}/{right}"]] @ whitening
                denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
                stability.append(
                    {
                        "dataset": dataset,
                        "kind": kind,
                        "left": left,
                        "right": right,
                        "whitened_cosine": float(a @ b / denominator)
                        if denominator > 1e-20
                        else None,
                    }
                )
    return stability


def score_layer(config, paths, layer, metadata, reference):
    """Fit the exact historical geometry and score one layer, checkpointing it."""
    out = Path(config["out_root"]) / f"L{layer:02d}"
    done = out / "complete.json"
    if done.exists():
        record = json.loads(done.read_text())
        if (
            record["source_sha"] != config["source_sha"]
            or record["input_fingerprint"] != config["input_fingerprint"]
        ):
            raise ValueError(f"Stale checkpoint: {done}")
        if any(sha(out / name) != digest for name, digest in record["files"].items()):
            raise ValueError(f"Checkpoint content mismatch: {done}")
        return record
    started = time.time()
    progress(config, "load_layer", layer=layer)
    tr = load_table(paths["labeling"], paths["train_labels"], layer, {"train"})
    wc = load_table(paths["wildchat"], paths["wildchat_labels"], layer, {"eval"})
    ev = load_table(paths["labeling"], paths["train_labels"], layer, {"eval"})
    tables = [tr, wc, ev]
    for store in paths["ood"]:
        tables.append(load_table(store, paths["ood_labels"], layer, {"eval", "full"}))
    table = merge_tables(tables)
    assert_label_context_coverage(
        table,
        {paths["train_labels"], paths["wildchat_labels"]}
        | ({paths["ood_labels"]} if paths["ood"] else set()),
    )
    roster, selections, selection_records = build_selections(config, table, metadata)
    write_json(out / "selection.json", selection_records)
    generic, generic_meta = store_io.load_summaries(paths["u_store"], KINDS, (layer,))
    gi = np.flatnonzero(store_io.fit_pool_mask(generic_meta))
    expected = 6468 if config["behavior"] == "evil" else 16000
    if len(gi) != 18793 or len(tr.ids) != expected:
        raise ValueError(f"Original ADD pool mismatch: generic={len(gi)} trait={len(tr.ids)}")
    if any(
        a.dtype != np.float16
        for a in (tr.x, tr.y, generic[("context_end", layer)], generic[("t1", layer)])
    ):
        raise ValueError("Original capture dtype must be float16")
    x = np.concatenate([generic[("context_end", layer)][gi], tr.x])[None]
    y = np.concatenate([generic[("t1", layer)][gi], tr.y])[None]
    del generic, tables, tr, wc, ev
    progress(config, "whitening", layer=layer, n_map=x.shape[1])
    wh = fits.fit_whitening(x, seed=0, device="cpu")
    xw, yw = fits.apply_whitening(x, wh), fits.apply_whitening(y, wh)
    del x, y
    progress(config, "map_fit", layer=layer)
    mapfit = fits.fit_linear_map(xw, yw, seed=0, device="cpu")
    del xw, yw
    out.mkdir(parents=True, exist_ok=True)
    np.savez(
        out / "transforms.npz",
        wh_mu=wh.mu,
        wh_w=wh.w,
        wh_gamma=wh.gamma,
        map_w=mapfit.w,
        map_x_mu=mapfit.x_mu,
        map_x_sd=mapfit.x_sd,
        map_y_mu=mapfit.y_mu,
    )
    write_json(out / "map_diagnostics.json", mapfit.diagnostics)
    progress(config, "directions", layer=layer)
    keys, directions = stream_directions([paths["labeling"]], layer, table.ids, selections)
    e1 = _load_rb_e1(Path(paths["extraction"]), [layer], 3584)[0]
    e1_ctx = _load_rb_e1(Path(paths["extraction"]), [layer], 3584, summary_kind="context_end")[0]
    np.savez(
        out / "directions.npz",
        keys=np.asarray(keys),
        answer=directions["t1"],
        context=directions["context_end"],
        e1_answer=e1,
        e1_context=e1_ctx,
    )
    write_json(
        out / "direction_stability.json", direction_stability(keys, directions, roster, wh.w[0])
    )
    z = fits.apply_whitening(table.x[None], wh)[0]
    za = fits.apply_whitening(table.y[None], wh)[0]
    mapped = fits.apply_map(z[None], mapfit)[0]
    del table.x, table.y
    rows, parity = [], []
    for dataset, ii in roster.items():
        available = [("e1", e1, e1_ctx)] + [
            (key.split("/", 1)[1], directions["t1"][j], directions["context_end"][j])
            for j, key in enumerate(keys)
            if key.startswith(dataset + "/")
        ]
        matrix, names = [], []
        for name, da, dc in available:
            da, dc = da @ wh.w[0], dc @ wh.w[0]
            for arm, a, d in (
                ("answer_direction_on_context", z, da),
                ("context_native", z, dc),
                ("mapped_answer", mapped, da),
                ("real_answer", za, da),
            ):
                if name == "e2" and arm == "context_native":
                    continue  # structurally zero: never a measured zero effect
                norm = np.linalg.norm(d)
                if norm < 1e-10:
                    raise ValueError(f"Degenerate direction: {name}/{arm}/L{layer}")
                prediction = arms._proj(a[ii][None], d[None])[0]
                matrix.append(prediction)
                names.append((name, arm))
        matrix = np.asarray(matrix)
        rho = arms.spearman_rows(matrix, table.dv[ii])
        for j, (name, arm) in enumerate(names):
            if name == "e1" and LAYERS[config["behavior"]][arm] == layer:
                old = reference[(dataset, arm)]
                error = abs(float(rho[j]) - old["rho_frozen"])
                parity.append(
                    {
                        "dataset": dataset,
                        "arm": arm,
                        "layer": layer,
                        "n": len(ii),
                        "expected_n": old["n_eval"],
                        "rho": float(rho[j]),
                        "expected_rho": old["rho_frozen"],
                        "absolute_error": error,
                    }
                )
                if len(ii) != old["n_eval"] or error > 1e-8:
                    write_json(out / "parity_failed.json", parity)
                    raise ValueError(f"Historical E1 parity failed: {parity[-1]}")
            rows.append(
                {
                    "dataset": dataset,
                    "variant": name,
                    "arm": arm,
                    "layer": layer,
                    "displayed_layer": LAYERS[config["behavior"]][arm] == layer,
                    "rho": None if not np.isfinite(rho[j]) else float(rho[j]),
                    "n": len(ii),
                }
            )
        np.savez(
            out / f"predictions_{dataset.replace(':', '_')}.npz",
            predictions=matrix,
            variants=np.asarray([v for v, _ in names]),
            arms=np.asarray([a for _, a in names]),
            ids=np.asarray(table.ids)[ii],
            groups=np.asarray(table.groups)[ii],
            dv=table.dv[ii],
        )
        progress(config, "dataset_scored", layer=layer, dataset=dataset, n=len(ii))
    write_json(out / "scores.json", rows)
    write_json(out / "parity.json", parity)
    record = {
        "source_sha": config["source_sha"],
        "input_fingerprint": config["input_fingerprint"],
        "layer": layer,
        "time": time.time(),
        "wall_s": time.time() - started,
        "max_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20,
        "files": {p.name: sha(p) for p in out.iterdir() if p.is_file()},
    }
    write_json(done, record)
    gc.collect()
    return record
