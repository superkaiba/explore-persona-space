"""Fixed L19 instruction-contrast transfer through the frozen #779 generic map.

The #1739 E1 directions are UNFILTERED instruction contrasts. There is no
behavior readout fit, judge call, layer search, or evaluation-based sign flip.
#779 targets include closing template tokens; #1739 t1 excludes those tokens.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import numpy as np

from explore_persona_space.experiments.issue_1739 import arms, store_io
from explore_persona_space.experiments.issue_1739.constants import MODEL_NAME
from explore_persona_space.experiments.issue_1739.generation import INSTRUCT_REVISION
from scripts.issue1739_claim4_fold import group_bootstrap_rhos
from scripts.issue1739_covariance_ablation import sha256, write_json
from scripts.issue1739_result2fair_score import _wc_eval_mask

LAYER = 19
HIDDEN = 3584
ROSTER = {
    "evil": ("hhrt", "toxicchat", "wildchat_rung"),
    "sycophancy": ("aita", "wildchat_rung"),
    "hallucination": ("nqopen", "simpleqa", "wildchat_rung"),
}
ARMS = ("real_answer", "mapped_answer", "context_native", "answer_direction_on_context")
NULL_ARMS = tuple(f"shuffled_seed{s}" for s in range(5))
ALL_ARMS = ARMS + NULL_ARMS
POOLING_CAVEAT = (
    "Frozen #779 v_x averages response plus assistant-closing template tokens; "
    "#1739 t1 averages completion tokens only. This is a declared endpoint mismatch."
)


def json_rows(path: Path) -> list[dict]:
    """Read either a rows JSON object or JSONL without inventing missing rows."""
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    value = json.loads(path.read_text())
    return value["rows"] if isinstance(value, dict) else value


def prompt_hashes(text: str) -> tuple[str, str]:
    """Exact and lowercase/whitespace-normalized rendered-prompt fingerprints."""
    norm = " ".join(text.lower().split()).strip()
    return hashlib.sha256(text.encode()).hexdigest(), hashlib.sha256(norm.encode()).hexdigest()


def load_prompt_index(path: Path) -> dict[tuple[str, str], dict]:
    """Require a unique verified content fingerprint for every namespace/context."""
    result = {}
    for row in json_rows(path):
        key = (str(row["namespace"]), str(row["context_id"]))
        if "prompt_text" in row:
            exact, normalized = prompt_hashes(row["prompt_text"])
        else:
            exact, normalized = row["exact_sha256"], row["normalized_sha256"]
        hashes = {"exact_sha256": str(exact), "normalized_sha256": str(normalized)}
        hashes.update(
            {
                k: str(v)
                for k, v in row.items()
                if k.endswith(("_exact_sha256", "_normalized_sha256"))
            }
        )
        if "user_turn_hashes" in row:
            hashes["user_turn_hashes"] = row["user_turn_hashes"]
        for field in (
            "generation_metadata",
            "rollouts",
            "source_shards",
            "pair",
            "side",
            "q_idx",
            "split",
            "rung",
            "group_key",
        ):
            if field in row:
                hashes[field] = row[field]
        all_hashes = content_hash_set(hashes, "exact") | content_hash_set(hashes, "normalized")
        if any(len(h) != 64 or any(c not in "0123456789abcdef" for c in h) for h in all_hashes):
            raise ValueError(f"invalid prompt digest: {key}")
        if key in result and result[key] != hashes:
            raise ValueError(f"conflicting prompt hashes: {key}")
        result[key] = hashes
    return result


def content_hash_set(row: dict, kind: str) -> set[str]:
    """Full prompt plus user/question hashes bridge rendered vs bare map inputs."""
    suffix = f"{kind}_sha256"
    result = {str(v) for k, v in row.items() if k.endswith(suffix)}
    for turn in row.get("user_turn_hashes", []):
        result.add(str(turn[suffix]))
    return result


def load_map_hashes(manifest: dict, base: Path) -> tuple[set[str], set[str], dict]:
    """Map training AND validation membership participate in leakage exclusion."""
    path = Path(manifest["map_prompt_hashes"])
    path = path if path.is_absolute() else base / path
    with np.load(path, allow_pickle=False) as z:
        exact = z["exact_sha256"].astype(str).tolist()
        normalized = z["normalized_sha256"].astype(str).tolist()
        n_train = len(exact)
        if "val_exact_sha256" in z.files and "val_normalized_sha256" in z.files:
            val_exact = z["val_exact_sha256"].astype(str).tolist()
            val_normalized = z["val_normalized_sha256"].astype(str).tolist()
            if len(val_exact) != 400 or len(val_normalized) != 400:
                raise ValueError("map validation prompt hashes must cover all 400 rows")
            exact += val_exact
            normalized += val_normalized
        elif manifest.get("map_hash_scope") == "train_and_validation" and len(exact) == 963844:
            n_train = 963444
        else:
            raise ValueError("map hashes omit validation rows used in lambda selection")
    if n_train != 963444 or len(exact) != len(normalized) or any(not h for h in exact + normalized):
        raise ValueError("map prompt hash coverage differs from the frozen map's fitted pool")
    return (
        set(exact),
        set(normalized),
        {
            "path": str(path),
            "sha256": sha256(path),
            "n_train": n_train,
            "n_validation": 400,
            "normalization": "lowercase then collapse whitespace",
        },
    )


def lookup_prompt(index: dict, namespace: str, context_id: str) -> dict:
    """Missing content evidence is a hard error rather than a passed leakage audit."""
    key = (namespace, context_id)
    if key not in index:
        raise ValueError(f"missing prompt fingerprint: {key}")
    return index[key]


def overlaps(hashes: dict, exact: set, normalized: set) -> bool:
    """Either exact or normalized membership excludes the observation."""
    return bool(
        content_hash_set(hashes, "exact") & exact
        or content_hash_set(hashes, "normalized") & normalized
    )


def validate_capture_provenance(meta, prompt_index, namespace, *, context_ids=None, keep=None):
    """Join used capture rows to generation model/revision and exact source/rollout."""
    allowed = None if context_ids is None else set(context_ids)
    checked, contexts, shards = [], set(), set()
    for i, row in enumerate(meta):
        cid = str(row["context_id"])
        if (allowed is not None and cid not in allowed) or (keep is not None and not keep[i]):
            continue
        record = lookup_prompt(prompt_index, namespace, cid)
        generation = record.get("generation_metadata")
        if not isinstance(generation, list) or not generation:
            raise ValueError(f"missing generation metadata: {namespace}/{cid}")
        for source in generation:
            if source.get("model") != MODEL_NAME or source.get("revision") != INSTRUCT_REVISION:
                raise ValueError(f"generation model/revision mismatch: {namespace}/{cid}")
            if not source.get("fingerprint") or not source.get("git_commit"):
                raise ValueError(f"incomplete generation source identity: {namespace}/{cid}")
        rollouts = record.get("rollouts")
        if not isinstance(rollouts, list) or not rollouts:
            raise ValueError(f"missing source rollout identities: {namespace}/{cid}")
        by_k = {}
        for source in rollouts:
            k = int(source["rollout_k"])
            if k in by_k:
                raise ValueError(f"duplicate source rollout identity: {namespace}/{cid}/{k}")
            if (
                not source.get("packed_src")
                or Path(source["packed_src"]).name != source["source_file"]
            ):
                raise ValueError(f"packed source/filename mismatch: {namespace}/{cid}/{k}")
            by_k[k] = source
        k = int(row["rollout_k"])
        if k not in by_k or row.get("source_file") != by_k[k]["source_file"]:
            raise ValueError(f"capture/source rollout filename mismatch: {namespace}/{cid}/{k}")
        for field in ("pair", "side", "q_idx", "split", "rung", "group_key"):
            if (
                row.get(field) is not None
                and record.get(field) is not None
                and row[field] != record[field]
            ):
                raise ValueError(f"capture/source {field} mismatch: {namespace}/{cid}/{k}")
        checked.append((cid, k, row["source_file"]))
        contexts.add(cid)
        shards.update(record.get("source_shards", []))
    if not checked or len(set(checked)) != len(checked):
        raise ValueError(f"empty or duplicate capture/source provenance joins: {namespace}")
    return {
        "namespace": namespace,
        "n_contexts": len(contexts),
        "n_capture_rows": len(checked),
        "model": MODEL_NAME,
        "revision": INSTRUCT_REVISION,
        "source_shards": sorted(shards),
        "capture_source_join_sha256": hashlib.sha256(
            json.dumps(checked, separators=(",", ":")).encode()
        ).hexdigest(),
    }


def load_store(store: Path, hidden: int = HIDDEN):
    """Use the repository's canonical shape/row-order validated summary loader."""
    return store_io.load_summaries(store, ("context_end", "t1"), (LAYER,), hidden_dim=hidden)


def extract_directions(arrays, meta, *, keep=None):
    """Identical unfiltered instruction polarity row weights at both positions."""
    keep = np.ones(len(meta), dtype=bool) if keep is None else np.asarray(keep, dtype=bool)
    if len(keep) != len(meta):
        raise ValueError("direction keep mask length mismatch")
    identities = [(str(r["context_id"]), int(r["rollout_k"])) for r in meta]
    if len(set(identities)) != len(identities):
        raise ValueError("duplicate extraction context/rollout identities")
    side = np.asarray([str(r["side"]).lower() for r in meta])
    if not set(side).issubset({"pos", "positive", "neg", "negative"}):
        raise ValueError("unknown extraction polarity")
    pos = keep & np.isin(side, ["pos", "positive"])
    neg = keep & np.isin(side, ["neg", "negative"])
    if not pos.any() or not neg.any():
        raise ValueError("empty instruction-contrast side after membership exclusion")
    out = {}
    for name, kind in (("answer", "t1"), ("context", "context_end")):
        a = arrays[(kind, LAYER)]
        v = a[pos].astype(np.float64).mean(0) - a[neg].astype(np.float64).mean(0)
        if not np.isfinite(v).all() or np.linalg.norm(v) == 0:
            raise ValueError(f"invalid {name} instruction contrast")
        out[name] = v
    counts = {
        "positive_rollouts": int(pos.sum()),
        "negative_rollouts": int(neg.sum()),
        "positive_contexts": len({identities[i][0] for i in np.flatnonzero(pos)}),
        "negative_contexts": len({identities[i][0] for i in np.flatnonzero(neg)}),
        "excluded_rollouts": int((~keep).sum()),
        "judge_filtered": False,
        "definition": "positive instruction mean minus negative instruction mean",
        "retained_row_sha256": hashlib.sha256(
            json.dumps(
                [identities[i] for i in np.flatnonzero(keep)], separators=(",", ":")
            ).encode()
        ).hexdigest(),
    }
    return out["answer"], out["context"], counts


def kept_rollouts(row: dict, observed: set[int], supplementary: dict | None = None) -> set[int]:
    """Prove that activation and DV aggregation use the same rollout identities."""
    per_rollout = row.get("per_rollout_scores")
    if per_rollout is None and supplementary is not None:
        per_rollout = supplementary.get("per_rollout_scores")
    if per_rollout is not None:
        vals = {int(str(k).removeprefix("k")): v for k, v in per_rollout.items()}
        if set(vals) != observed:
            raise ValueError(f"judge/store rollout-ID mismatch for {row['context_id']}")
        valid = {k: float(v) for k, v in vals.items() if v is not None}
        if not valid or not all(np.isfinite(v) for v in valid.values()):
            raise ValueError(f"invalid retained scores for {row['context_id']}")
        mean = float(np.mean(list(valid.values())))
        # Supplementary hallucination files may encode fabrication as 0/100.
        if supplementary is not None and row.get("n_decided") is not None:
            mean /= 100.0
        if not np.isclose(mean, float(row["dv"]), atol=1e-10, rtol=1e-10):
            raise ValueError(f"cached DV does not match retained rollouts for {row['context_id']}")
        if "n_rollouts_kept" in row and len(valid) != row["n_rollouts_kept"]:
            raise ValueError("cached kept-rollout count mismatch")
        return set(valid)
    if row.get("n_unjudged") == 0 and row.get("n_decided") == row.get("n_rollouts"):
        if observed != set(range(int(row["n_rollouts"]))):
            raise ValueError(
                f"all-decided hallucination rollout coverage mismatch: {row['context_id']}"
            )
        return observed
    raise ValueError(f"no retained-rollout identity evidence for {row['context_id']}")


def reduce_eval(arrays, meta, labels: list[dict], *, supplementary=None):
    """Aggregate fp64 raw activations over the exact rollouts entering each DV."""
    by_context = {}
    for i, row in enumerate(meta):
        by_context.setdefault(str(row["context_id"]), []).append(i)
    ids, groups, rungs, dv, x, y, audits = [], [], [], [], [], [], []
    for row in labels:
        cid = str(row["context_id"])
        if cid not in by_context:
            raise ValueError(f"labeled evaluation context absent from store: {cid}")
        indices = by_context[cid]
        ks = [int(meta[i]["rollout_k"]) for i in indices]
        if len(set(ks)) != len(ks):
            raise ValueError(f"duplicate evaluation rollout identity: {cid}")
        extra = supplementary.get(cid) if supplementary is not None else None
        valid = kept_rollouts(row, set(ks), extra)
        chosen = [i for i, k in zip(indices, ks, strict=True) if k in valid]
        x.append(arrays[("context_end", LAYER)][chosen].astype(np.float64).mean(0))
        y.append(arrays[("t1", LAYER)][chosen].astype(np.float64).mean(0))
        ids.append(cid)
        rungs.append(str(row["rung"]))
        group = row.get("group_key")
        groups.append(f"{row['rung']}:{group if group is not None else cid}")
        dv.append(float(row["dv"]))
        audits.append(
            {
                "context_id": cid,
                "rollout_k": sorted(valid),
                "dropped_activation_rows": len(indices) - len(chosen),
            }
        )
    if not ids:
        raise ValueError("evaluation subset is empty")
    return {
        "x": np.stack(x),
        "y": np.stack(y),
        "dv": np.asarray(dv),
        "context_ids": np.asarray(ids),
        "groups": np.asarray(groups),
        "rungs": np.asarray(rungs),
        "rollout_audit": audits,
    }


def map_projection(payload, x, direction):
    """Exact affine pushdown in the canonical float64 application coordinates."""

    def a(key):
        value = payload[key]
        return np.asarray(
            value.detach().cpu() if hasattr(value, "detach") else value, dtype=np.float64
        )

    weight = (a("W") @ direction) / a("xsd")
    offset = float(a("ymu") @ direction - a("xmu") @ weight)
    return x @ weight + offset


def summarize(pred, dv, groups, rungs, *, n_boot, seed=1739963):
    """Per-dataset paired intervals; null is the mean of seed-specific rhos."""
    result, bootstrap = [], {}
    for rung in sorted(set(rungs)):
        ix = np.flatnonzero(np.asarray(rungs) == rung)
        mat, target = pred[:, ix], dv[ix]
        rho = arms.spearman_rows(mat, target)
        active = np.isfinite(rho)
        boots = np.full((len(ALL_ARMS), n_boot), np.nan)
        ng = len(set(np.asarray(groups)[ix]))
        if len(ix) >= 3 and active.any():
            boots[active], ng = group_bootstrap_rhos(
                mat[active],
                target,
                np.asarray(groups)[ix],
                n_boot=n_boot,
                rng=np.random.default_rng(seed),
            )
        estimates = {}
        for i, name in enumerate(ALL_ARMS):
            ok = np.isfinite(boots[i])
            estimates[name] = {
                "rho": float(rho[i]) if active[i] else None,
                "ci95": np.quantile(boots[i, ok], [0.025, 0.975]).tolist() if ok.any() else None,
                "valid_bootstrap_draws": int(ok.sum()),
                "status": "ok" if active[i] else "undefined_constant_or_insufficient_data",
            }
        # Never collapse null predictions before ranking: averaging correlations is the estimand.
        null_ok = np.isfinite(boots[4:]).all(axis=0)
        null_boot = boots[4:, null_ok].mean(0)
        null_rho = float(rho[4:].mean()) if np.isfinite(rho[4:]).all() else None
        estimates["shuffled_map"] = {
            "rho": null_rho,
            "ci95": np.quantile(null_boot, [0.025, 0.975]).tolist() if null_ok.any() else None,
            "seed_rhos": [float(v) if np.isfinite(v) else None for v in rho[4:]],
            "seed_sd": float(rho[4:].std(ddof=1)) if null_rho is not None else None,
            "valid_bootstrap_draws": int(null_ok.sum()),
            "estimand": "mean of five seed-specific Spearman correlations",
        }
        differences = {}
        for name, a, b in (
            ("map_minus_context_native", 1, 2),
            ("map_minus_answer_on_context", 1, 3),
            ("real_answer_minus_map", 0, 1),
        ):
            ok = np.isfinite(boots[[a, b]]).all(axis=0)
            delta = boots[a, ok] - boots[b, ok]
            differences[name] = {
                "delta": float(rho[a] - rho[b]) if active[a] and active[b] else None,
                "ci95": np.quantile(delta, [0.025, 0.975]).tolist() if ok.any() else None,
                "valid_bootstrap_draws": int(ok.sum()),
            }
        ok = null_ok & np.isfinite(boots[1])
        delta = boots[1, ok] - boots[4:, ok].mean(0)
        differences["map_minus_shuffled_map"] = {
            "delta": float(rho[1] - null_rho) if active[1] and null_rho is not None else None,
            "ci95": np.quantile(delta, [0.025, 0.975]).tolist() if ok.any() else None,
            "valid_bootstrap_draws": int(ok.sum()),
        }
        result.append(
            {
                "rung": str(rung),
                "n": len(ix),
                "n_groups": ng,
                "arms": estimates,
                "differences": differences,
            }
        )
        bootstrap[str(rung)] = boots
    return result, bootstrap


def load_payload(path: Path):
    """Stored payloads must be the requested affine ridge map, never a fallback."""
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload["kind"] != "ridge" or tuple(payload["W"].shape) != (HIDDEN, HIDDEN):
        raise ValueError(f"incompatible ridge payload: {path}")
    for key in ("W", "xmu", "xsd", "ymu"):
        if not torch.isfinite(payload[key]).all():
            raise ValueError(f"nonfinite map parameter: {path}:{key}")
    if (payload["xsd"] <= 0).any():
        raise ValueError("invalid map standardizer")
    return payload


def reconstruction_metrics(pred, actual):
    """Raw-space variance-weighted R²/cosine and exact-pool cosine retrieval."""
    centered = actual - actual.mean(0)
    denom = float(np.square(centered).sum())
    r2 = 1.0 - float(np.square(actual - pred).sum()) / denom if denom else None
    norms_a = np.linalg.norm(actual, axis=1)
    norms_p = np.linalg.norm(pred, axis=1)
    if np.any(norms_a == 0) or np.any(norms_p == 0):
        raise ValueError("zero representation norm in reconstruction diagnostics")
    targets = actual / norms_a[:, None]
    query = pred / norms_p[:, None]
    correct = 0
    for start in range(0, len(pred), 256):
        stop = min(start + 256, len(pred))
        winners = (query[start:stop] @ targets.T).argmax(1)
        correct += int(np.sum(winners == np.arange(start, stop)))
    return {
        "r2": r2,
        "mean_cosine": float(np.sum(query * targets, axis=1).mean()),
        "nearest_neighbor_accuracy": correct / len(actual),
        "retrieval_pool_size": len(actual),
        "retrieval_chance": 1.0 / len(actual),
        "retrieval_metric": "raw cosine; argmax first-index tie rule; identical eligible rung pool",
    }


def reconstruction_diagnostics(data, payloads, *, progress_fn=None):
    """Behavior-transfer map validity under the declared answer pooling mismatch."""
    import torch
    import issue779_ffc_n1m_fits as canonical

    bias = (payloads[0]["ymu"].to(torch.float64) - payloads[0]["xmu"].to(torch.float64)).numpy()
    names = ("frozen_map",) + NULL_ARMS
    results = {}
    for rung in sorted(set(data["rungs"])):
        take = data["rungs"] == rung
        x, y = data["x"][take], data["y"][take]
        row = {"identity_plus_bias": reconstruction_metrics(x + bias, y)}
        for name, payload in zip(names, payloads, strict=True):
            if progress_fn:
                progress_fn("reconstruction", rung=str(rung), map=name)
            pred = np.concatenate(
                [
                    canonical.apply_map(payload, x[s : s + 512], torch.device("cpu"))
                    for s in range(0, len(x), 512)
                ]
            )
            row[name] = reconstruction_metrics(pred, y)
        results[str(rung)] = row
    return {
        "answer_pooling_caveat": POOLING_CAVEAT,
        "identity_bias_source": "frozen map's train-only ymu-xmu; stored fp32 means upcast to fp64",
        "datasets": results,
    }


def run_behavior(args, behavior, manifest, prompt_index, map_exact, map_norm, payloads):
    """Score one behavior with one source-frozen evaluation roster."""
    out = args.out / behavior
    out.mkdir(parents=True, exist_ok=True)
    extraction_namespace = f"{behavior}_extraction"
    arrays, meta = load_store(args.store_root / extraction_namespace)
    extraction_fingerprints = [
        lookup_prompt(prompt_index, extraction_namespace, str(r["context_id"])) for r in meta
    ]
    keep = np.asarray([not overlaps(h, map_exact, map_norm) for h in extraction_fingerprints])
    provenance_audits = [
        validate_capture_provenance(
            meta,
            prompt_index,
            extraction_namespace,
            keep=keep,
        )
    ]
    va, vc, extraction_audit = extract_directions(arrays, meta, keep=keep)
    extraction_exact = set().union(
        *(
            content_hash_set(h, "exact")
            for h, k in zip(extraction_fingerprints, keep, strict=True)
            if k
        )
    )
    extraction_norm = set().union(
        *(
            content_hash_set(h, "normalized")
            for h, k in zip(extraction_fingerprints, keep, strict=True)
            if k
        )
    )
    del arrays, meta
    np.savez(out / "directions.npz", answer=va, context=vc, layer=LAYER)
    labels_paths = [
        args.repo / f"eval_results/issue_1739/dv_dataset/{behavior}/labeling.json",
        args.repo / f"eval_results/issue_1739/wildchat_rung/dv_dataset/{behavior}/labeling.json",
    ]
    supplementary = None
    if behavior == "hallucination" and args.hallu_per_rollout is not None:
        supplementary = {str(r["context_id"]): r for r in json_rows(args.hallu_per_rollout)}
    batches, exclusions, coverage = [], [], []
    for namespace, label_path in zip(
        (f"{behavior}_labeling", "wildchat"), labels_paths, strict=True
    ):
        all_labels = json_rows(label_path)
        selected = [
            r for r in all_labels if r.get("split") == "eval" and r.get("rung") in ROSTER[behavior]
        ]
        if namespace == "wildchat":
            selected = [
                r
                for r, keep_row in zip(
                    selected, _wc_eval_mask([str(r["context_id"]) for r in selected]), strict=True
                )
                if keep_row
            ]
        n_planned = len(selected)
        retain = []
        for row in selected:
            cid = str(row["context_id"])
            reason = None
            hashes = lookup_prompt(prompt_index, namespace, cid)
            if row["dv"] is None:
                reason = "no_valid_behavior_label"
            elif overlaps(hashes, map_exact, map_norm):
                reason = "overlap_map_train_or_validation"
            elif overlaps(hashes, extraction_exact, extraction_norm):
                reason = "overlap_direction_extraction"
            if reason:
                exclusions.append({"context_id": cid, "rung": row["rung"], "reason": reason})
            else:
                retain.append(row)
        arrays, meta = load_store(args.store_root / namespace)
        provenance_audits.append(
            validate_capture_provenance(
                meta,
                prompt_index,
                namespace,
                context_ids=[str(r["context_id"]) for r in retain],
            )
        )
        batch = reduce_eval(
            arrays, meta, retain, supplementary=supplementary if namespace != "wildchat" else None
        )
        del arrays, meta
        batches.append(batch)
        coverage.append({"namespace": namespace, "planned": n_planned, "retained": len(retain)})
    data = {
        k: np.concatenate([b[k] for b in batches])
        for k in ("x", "y", "dv", "context_ids", "groups", "rungs")
    }
    if set(data["rungs"]) != set(ROSTER[behavior]):
        raise ValueError(f"realized rung roster differs from registered {ROSTER[behavior]}")
    if len(set(data["context_ids"])) != len(data["context_ids"]):
        raise ValueError("duplicated evaluation context across stores")
    import torch
    import issue779_ffc_n1m_fits as canonical

    sample = data["x"][:16]
    scores = [
        data["y"] @ va,
        map_projection(payloads[0], data["x"], va),
        data["x"] @ vc,
        data["x"] @ va,
    ]
    parity = []
    for payload in payloads:
        direct = canonical.apply_map(payload, sample, torch.device("cpu")) @ va
        pushed = map_projection(payload, sample, va)
        # fp64 dot association error only; this compares the identical stored fp32 parameters.
        np.testing.assert_allclose(direct, pushed, rtol=1e-10, atol=1e-7)
        parity.append(float(np.max(np.abs(direct - pushed))))
    scores.extend(map_projection(p, data["x"], va) for p in payloads[1:])
    predictions = np.stack(scores)
    if not np.isfinite(predictions).all():
        raise ValueError("nonfinite fixed-direction score")
    np.savez(
        out / "predictions.npz",
        predictions=predictions,
        arms=np.asarray(ALL_ARMS),
        **{k: data[k] for k in ("dv", "context_ids", "groups", "rungs")},
    )
    evaluation_hashes = []
    for cid, rung in zip(data["context_ids"], data["rungs"], strict=True):
        namespace = "wildchat" if rung == "wildchat_rung" else f"{behavior}_labeling"
        evaluation_hashes.append(
            {
                "context_id": str(cid),
                "namespace": namespace,
                **lookup_prompt(prompt_index, namespace, str(cid)),
            }
        )
    write_json(out / "evaluation_hashes.json", evaluation_hashes)
    write_json(out / "capture_provenance_audit.json", provenance_audits)
    write_json(out / "rollout_alignment.json", [r for b in batches for r in b["rollout_audit"]])
    write_json(out / "leakage_exclusions.json", exclusions)
    results, boots = summarize(
        predictions, data["dv"], data["groups"], data["rungs"], n_boot=args.n_boot
    )
    np.savez(out / "bootstrap_rhos.npz", **boots)

    def report(phase, **extra):
        record = {
            "phase": phase,
            "behavior": behavior,
            "time": time.time(),
            "source_sha": args.source_sha,
            **extra,
        }
        write_json(args.out / "progress.json", record)
        print(json.dumps(record), flush=True)

    write_json(
        out / "reconstruction.json", reconstruction_diagnostics(data, payloads, progress_fn=report)
    )
    result = {
        "behavior": behavior,
        "layer": LAYER,
        "source_sha": args.source_sha,
        "config_sha256": args.config_sha,
        "planned_rungs": list(ROSTER[behavior]),
        "direction_type": "unfiltered instruction contrast",
        "judge_filtered": False,
        "direction_audit": extraction_audit,
        "answer_pooling_caveat": POOLING_CAVEAT,
        "coordinate_space": "raw activations; #779 map internally standardizes inputs only",
        "downstream_behavior_regression": False,
        "coverage": coverage,
        "n_eval": len(data["dv"]),
        "map_projection_max_abs_error": parity,
        "input_label_sha256": {str(p): sha256(p) for p in labels_paths},
        "n_boot": args.n_boot,
        "bootstrap_scope": "evaluation groups conditional on frozen directions and maps",
        "results": results,
    }
    write_json(out / "results.json", result)
    write_json(
        out / "complete.json",
        {
            "source_sha": args.source_sha,
            "config_sha256": args.config_sha,
            "artifact_sha256": {
                p.name: sha256(p)
                for p in sorted(out.iterdir())
                if p.is_file() and p.name != "complete.json"
            },
        },
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--store-root", type=Path, required=True)
    ap.add_argument("--map-manifest", type=Path, required=True)
    ap.add_argument("--prompt-index", type=Path, required=True)
    ap.add_argument("--repo", type=Path, default=ROOT)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--hallu-per-rollout", type=Path)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--behaviors", nargs="+", choices=list(ROSTER), default=list(ROSTER))
    args = ap.parse_args()
    args.source_sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    manifest = json.loads(args.map_manifest.read_text())
    if int(manifest["layer"]) != LAYER or manifest["complete"] is not True:
        raise ValueError("map manifest has the wrong layer")
    if set(manifest["null_payloads"]) != {str(s) for s in range(5)}:
        raise ValueError("expected five complete pairing-shuffled map payloads")
    paths = [Path(manifest["frozen_payload"])] + [
        Path(manifest["null_payloads"][str(s)]) for s in range(5)
    ]
    paths = [p if p.is_absolute() else args.map_manifest.parent / p for p in paths]
    expected_hashes = manifest["payload_sha256"]
    for path in paths:
        expected = expected_hashes.get(str(path), expected_hashes.get(path.name))
        if expected is None or sha256(path) != expected:
            raise ValueError(f"map manifest checksum mismatch: {path}")
    map_exact, map_norm, hash_audit = load_map_hashes(manifest, args.map_manifest.parent)
    if hash_audit["sha256"] != manifest["map_prompt_hashes_sha256"]:
        raise ValueError("map membership hashes disagree with manifest")
    prompt_index = load_prompt_index(args.prompt_index)
    store_hashes = {}
    for namespace in sorted(
        {"wildchat"}
        | {f"{b}_{kind}" for b in args.behaviors for kind in ("extraction", "labeling")}
    ):
        files = sorted(
            p
            for p in (args.store_root / namespace).iterdir()
            if p.name.startswith("row_index") or p.name.startswith(("t1_L19", "context_end_L19"))
        )
        if not files:
            raise ValueError(f"missing selected-layer store files: {namespace}")
        store_hashes.update({str(p): sha256(p) for p in files})
    label_hashes = {}
    for b in args.behaviors:
        for middle in ("", "wildchat_rung/"):
            path = args.repo / f"eval_results/issue_1739/{middle}dv_dataset/{b}/labeling.json"
            label_hashes[str(path)] = sha256(path)
    config = {
        "source_sha": args.source_sha,
        "map_manifest_sha256": sha256(args.map_manifest),
        "prompt_index_sha256": sha256(args.prompt_index),
        "n_boot": args.n_boot,
        "map_payload_sha256": {str(p): sha256(p) for p in paths},
        "map_hash_audit": hash_audit,
        "roster": ROSTER,
        "layer": LAYER,
        "store_sha256": store_hashes,
        "label_sha256": label_hashes,
        "hallu_per_rollout_sha256": sha256(args.hallu_per_rollout)
        if args.hallu_per_rollout
        else None,
    }
    args.config_sha = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    if (args.out / "config.json").exists() and json.loads(
        (args.out / "config.json").read_text()
    ) != config:
        raise ValueError(
            "refusing to mix a different source/input/config fingerprint into existing output"
        )
    write_json(args.out / "config.json", config)
    payloads = [load_payload(p) for p in paths]
    for behavior in args.behaviors:
        done = args.out / behavior / "complete.json"
        if done.exists():
            saved = json.loads(done.read_text())
            if saved["source_sha"] != args.source_sha or saved["config_sha256"] != args.config_sha:
                raise ValueError("stale completed-cell fingerprint")
            for name, digest in saved["artifact_sha256"].items():
                if sha256(done.parent / name) != digest:
                    raise ValueError(f"completed-cell artifact checksum mismatch: {name}")
            continue
        progress = {
            "phase": "scoring",
            "behavior": behavior,
            "time": time.time(),
            "source_sha": args.source_sha,
        }
        write_json(args.out / "progress.json", progress)
        print(json.dumps(progress), flush=True)
        run_behavior(args, behavior, manifest, prompt_index, map_exact, map_norm, payloads)
    write_json(
        args.out / "progress.json",
        {"phase": "complete", "time": time.time(), "source_sha": args.source_sha},
    )


if __name__ == "__main__":
    main()
