"""User-requested descriptive follow-up to #2546, with no generation or API judging.

Reconstruct the previously evaluated full-data operators using their exact fitter;
match their raw-basis input singular vectors to parent-model SAE dictionaries.
Decoder matches identify feature write directions aligned with map read directions;
encoder matches are a complementary comparison of linear read directions.
Neither is evidence that parent SAE feature semantics transfer to OpenThinker.
Export actual retrieval competitors with full questions and own generated answers.
"""

from __future__ import annotations

import argparse
import gc
import gzip
import hashlib
import json
import shutil
import time
from datetime import UTC, datetime
from pathlib import Path

import issue2546_cx_eot_prepost_diffs as prior
import numpy as np
import torch

ROOT = Path("/home/thomasjiralerspong/explore-persona-space")
BASE = Path("/mnt/eps-data/thomasjiralerspong/cot_necessity")
OUT = BASE / "input_sae_qualitative_20260906"
SAE = Path("/mnt/eps-data/thomasjiralerspong/issue1482_saedense/sae/resid_post_layer_19/trainer_1")
DIMS = [458, 2570, 2718]
K = 50  # Existing #2546 operator comparison, not selected by SAE alignment.
N_NULL = 128  # Exploratory reference precision only; no significance gate.


def save(name, value):
    """Atomically persist a completed analysis phase."""
    OUT.mkdir(parents=True, exist_ok=True)
    dest = OUT / name
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False))
    tmp.replace(dest)


def sha(path):
    """Hash an input or source without loading the whole file into memory."""
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def fit():
    """Reconstruct original operators; persist raw-basis top input directions."""
    OUT.mkdir(parents=True, exist_ok=True)
    ids, folds, predictions = prior.load_preds("p7_A")
    del predictions
    assert len(ids) == len(set(ids)) == 30193
    y = prior.load_target("ans_mean", "post", ids)
    results = {}
    for name, kind, lam in [
        ("context", "cx_last", prior.LAM_A),
        ("eot", "cot_boundary", prior.LAM_D),
    ]:
        start = time.monotonic()
        x = prior.load_target(kind, "post", ids)
        assert np.isfinite(x).all() and np.isfinite(y).all()
        fitted = prior.RidgeFit(x, y, lam)
        m = fitted.raw_operator().T.astype(np.float32)
        del x, fitted
        print(name, "fit complete; computing SVD", flush=True)
        u, s, vt = np.linalg.svd(m, full_matrices=False)
        v = vt[:K].T.copy()
        # Sign is arbitrary: orient largest-magnitude coordinate positive.
        sign = np.sign(v[np.abs(v).argmax(0), np.arange(K)])
        v *= sign
        u = u[:, :K] * sign
        np.savez(OUT / f"{name}_directions.npz", v=v, u=u, s=s, ids=ids, folds=folds, operator=m)
        results[name] = {
            "lambda": lam,
            "frobenius": float(np.linalg.norm(m)),
            "seconds": time.monotonic() - start,
            "n": len(ids),
        }
        save("fit_status.json", results)
        print(name, results[name], flush=True)
        del m, u, s, vt, v
        gc.collect()
    a = np.load(OUT / "context_directions.npz")
    d = np.load(OUT / "eot_directions.npz")
    checks = {}
    for key in ["v", "u"]:
        overlap = float(np.linalg.svd(a[key].T @ d[key], compute_uv=False).mean())
        checks[key] = overlap
    old = json.loads(
        (ROOT / "eval_results/issue_2546/allfit/eot_vs_context/diffs/diffs.json").read_text()
    )
    for key, side in [("v", "right_input"), ("u", "left_output")]:
        anchor = old["A3_operator_comparison"]["operators"]["subspace_overlaps"]["50"][side][
            "mean_principal_cos"
        ]
        assert abs(checks[key] - anchor) < 0.002, (key, checks[key], anchor)
    save("operator_parity.json", checks)


def match():
    """Search every dictionary feature equally for observed and null directions."""
    cfg = json.loads((SAE / "config.json").read_text())["trainer"]
    assert (cfg["layer"], cfg["k"], cfg["activation_dim"], cfg["dict_size"]) == (
        19,
        64,
        3584,
        131072,
    )
    assert cfg["lm_name"] == "Qwen/Qwen2.5-7B-Instruct"
    state = torch.load(SAE / "ae.pt", map_location="cpu", weights_only=True, mmap=True)
    assert set(state) == {
        "b_dec",
        "k",
        "threshold",
        "decoder.weight",
        "encoder.weight",
        "encoder.bias",
    }
    a = np.load(OUT / "context_directions.npz")["v"]
    d = np.load(OUT / "eot_directions.npz")["v"]
    rng = np.random.default_rng(20260906)
    null = rng.standard_normal((3584, N_NULL)).astype(np.float32)
    null /= np.linalg.norm(null, axis=0)
    axes = np.eye(3584, dtype=np.float32)[:, DIMS]
    vectors = np.concatenate([a, d, axes, null], axis=1)
    assert np.allclose(np.linalg.norm(vectors, axis=0), 1, atol=1e-5)
    results = {
        "config": cfg,
        "k_directions": K,
        "n_null": N_NULL,
        "null_seed": 20260906,
        "null_ceiling": 1.0,
        "source_sha256": sha(__file__),
        "fitter_sha256": sha(prior.__file__),
        "sae_config_sha256": sha(SAE / "config.json"),
        "sae_weights_sha256": sha(SAE / "ae.pt"),
        "caveat": (
            "Parent-model dictionary: semantic transfer to OpenThinker is unvalidated. "
            "Signed cosine matches, not feature activation or causal tests."
        ),
        "bases": {},
    }
    wanted = set()
    for basis, key in [("decoder", "decoder.weight"), ("encoder", "encoder.weight")]:
        weights = state[key].numpy()
        if basis == "decoder":
            weights = weights.T
        assert weights.shape == (131072, 3584)
        scores = np.empty((len(weights), vectors.shape[1]), np.float32)
        for start in range(0, len(weights), 4096):
            w = weights[start : start + 4096].copy()
            norms = np.linalg.norm(w, axis=1, keepdims=True)
            assert np.isfinite(w).all() and (norms > 0).all()
            scores[start : start + len(w)] = (w / norms) @ vectors
        np.savez(OUT / f"{basis}_feature_cosines.npz", scores=scores, vectors=vectors)
        rows = []
        for j in range(2 * K + 3):
            ix = np.argsort(-np.abs(scores[:, j]), kind="stable")[:5]
            wanted.update(int(f) for f in ix)
            rows.append(
                {
                    "map": ("context" if j < K else "eot" if j < 2 * K else "massive_axis"),
                    "direction": int(j % K + 1) if j < 2 * K else DIMS[j - 2 * K],
                    "matches": [{"feature": int(f), "cosine": float(scores[f, j])} for f in ix],
                }
            )
        null_max = np.abs(scores[:, 2 * K + 3 :]).max(0)
        projection_a = np.square(scores[:, :K]).sum(1)
        projection_d = np.square(scores[:, K : 2 * K]).sum(1)
        differential = {}
        for name, contrast in [
            ("context", projection_a - projection_d),
            ("eot", projection_d - projection_a),
        ]:
            selected = np.argsort(-contrast, kind="stable")[:20]
            wanted.update(int(f) for f in selected)
            differential[name] = [
                {
                    "feature": int(f),
                    "context_projection_squared": float(projection_a[f]),
                    "eot_projection_squared": float(projection_d[f]),
                    "difference": float(contrast[f]),
                }
                for f in selected
            ]
        results["bases"][basis] = {
            "directions": rows,
            "differential_subspace_features": differential,
            "null_max_abs_cosines": null_max.tolist(),
            "null_p95": float(np.quantile(null_max, 0.95)),
            "null_median": float(np.median(null_max)),
        }
        save("matches.json", results)
        print(basis, "matched; null p95", results["bases"][basis]["null_p95"], flush=True)
        del scores
    # Reuse published descriptions; no automated Claude or new API judging.
    labels = {}
    cache = ROOT / "eval_results/issue_1482/worst_pc_autointerp/np_cache"
    paths = sorted(cache.glob("*.jsonl.gz"))
    assert paths, cache
    for p in paths:
        with gzip.open(p, "rt") as f:
            for line in f:
                row = json.loads(line)
                fid = int(row["index"])
                if fid not in wanted:
                    continue
                model = row.get("explanationModelName", "")
                if fid not in labels or model == "gemini-2.0-flash":
                    labels[fid] = {
                        "description": row.get("description", ""),
                        "model": model,
                        "source": str(p),
                    }
    for b in results["bases"].values():
        feature_lists = [row["matches"] for row in b["directions"]]
        feature_lists.extend(b["differential_subspace_features"].values())
        for feature_list in feature_lists:
            for feature in feature_list:
                fid = feature["feature"]
                feature["description"] = labels.get(fid)
                feature["url"] = f"https://www.neuronpedia.org/qwen2.5-7b-it/19-resid-post-aa/{fid}"
    results["description_coverage"] = {
        "matched_unique_features": len(wanted),
        "with_description": len(labels),
    }
    save("matches.json", results)
    print("descriptions", results["description_coverage"], flush=True)


def qualitative():
    """Export full-text pairs, selecting the competitor from the map that misses."""
    p = BASE / "allfit/preds"
    z = np.load(p / "p7_A__all__a1.npz")
    ids = z["conv_ids"].astype(str)
    folds = z["folds"]
    cache = np.load("/tmp/issue2546_eot_flip_cache.npz")
    hits = {}
    for name in ["A", "D"]:
        h = np.load(p / f"hits__p7_{name}__all__a1.npz")
        assert np.array_equal(ids, h["row_ids"].astype(str))
        assert np.array_equal(folds, h["folds"])
        hits[name] = h["hit_whitened_csls"]
        assert np.array_equal(hits[name], cache[f"{name}_hit"])
        assert np.array_equal(cache[f"{name}_rank"] == 1, hits[name])
        nn = cache[f"{name}_nn_other"]
        assert np.all((nn >= 0) & (nn < len(ids))) and np.all(nn != np.arange(len(ids)))
        assert np.array_equal(folds, folds[nn])
    old = json.loads(
        (
            ROOT / "eval_results/issue_2546/allfit/eot_vs_context/flips/flip_sample_coded.json"
        ).read_text()
    )
    pos = {r: i for i, r in enumerate(ids)}
    for row in old["items"]:
        i = pos[row["row_id"]]
        assert ids[cache["A_nn_other"][i]] == row["nn_row_id"]
        assert int(cache["A_rank"][i]) == row["rank_context"]
        assert int(cache["D_rank"][i]) == row["rank_eot"]
    selected = {row["row_id"] for row in old["items"] if row["group"] == "flip"}
    selected.update(ids[~hits["D"]])
    needed = set(selected)
    for rid in selected:
        i = pos[rid]
        name = "D" if not hits["D"][i] else "A"
        needed.add(ids[cache[f"{name}_nn_other"][i]])
    hf = BASE / "hf/issue2546_cotmap"
    questions = json.loads(
        (hf / "eval_results_mirror/out/necessity/pair_necessity_a1.json").read_text()
    )["question_by_row_id"]
    gen = {}
    for ds in prior.DATASETS:
        with (hf / f"raw_completions/post_greedy_a1/{ds}.jsonl").open() as f:
            for line in f:
                row = json.loads(line)
                rid = row["row_id"]
                if rid in needed:
                    assert rid not in gen
                    assert "</think>" in row["text"], rid
                    gen[rid] = row["text"].rsplit("</think>", 1)[1].strip()
    assert needed <= gen.keys() and needed <= questions.keys()
    items = []
    for rid in sorted(selected):
        i = pos[rid]
        group = (
            "recovered" if hits["D"][i] else "lost_after_cot" if hits["A"][i] else "missed_by_both"
        )
        name = "A" if group == "recovered" else "D"
        nr = ids[cache[f"{name}_nn_other"][i]]
        items.append(
            {
                "row_id": rid,
                "group": group,
                "neighbor_map": name,
                "neighbor_id": nr,
                "question": questions[rid],
                "answer": gen[rid],
                "neighbor_question": questions[nr],
                "neighbor_answer": gen[nr],
                "rank_context": int(cache["A_rank"][i]),
                "rank_eot": int(cache["D_rank"][i]),
            }
        )
    save(
        "qualitative_pairs.json",
        {
            "items": items,
            "n": len(items),
            "selection": (
                "All end-of-thought misses plus the prior dataset-stratified 40 recovered sample."
            ),
            "cache_sha256": sha("/tmp/issue2546_eot_flip_cache.npz"),
            "cache_validation": (
                "Recorded hits, folds, ranks, and all 60 prior sample context competitors verified."
            ),
            "old_sample_caveat": (
                "Prior sample always used the context competitor, including end-of-thought "
                "failures; do not reuse those labels as end-of-thought error characterizations."
            ),
        },
    )
    print("qualitative pairs", len(items), flush=True)


def validate():
    """Recompute full-pool retrieval and bind legacy cache to explicit row IDs."""
    import issue2546_eot_flip_analysis as retrieval

    ids, folds, a = prior.load_preds("p7_A")
    ids_d, folds_d, d = prior.load_preds("p7_D")
    assert np.array_equal(ids, ids_d) and np.array_equal(folds, folds_d)
    # The original retrieval implementation computes its mean in float64.
    # Keep that dtype: a float32 mean changes small CSLS margins.
    y = prior.load_target("ans_mean", "post", ids).astype(np.float64)
    current = retrieval.retrieval_detail({"A": a, "D": d}, y, folds)
    cache = np.load("/tmp/issue2546_eot_flip_cache.npz")
    checks = {}
    for name, fields in current.items():
        for field, values in fields.items():
            old = cache[f"{name}_{field}"]
            if field == "margin":
                checks[f"{name}_{field}_max_abs_error"] = float(np.abs(values - old).max())
                save("retrieval_validation_progress.json", checks)
                assert np.allclose(values, old, atol=1e-7, rtol=1e-5)
            else:
                assert np.array_equal(values, old), (name, field)
                checks[f"{name}_{field}_identical"] = True
    np.savez(
        OUT / "validated_retrieval.npz",
        ids=ids,
        folds=folds,
        **{
            f"{name}_{field}": values
            for name, fields in current.items()
            for field, values in fields.items()
        },
    )
    save(
        "retrieval_validation.json",
        {
            "checks": checks,
            "n": len(ids),
            "cache_sha256": sha("/tmp/issue2546_eot_flip_cache.npz"),
            "retrieval_source_sha256": sha(retrieval.__file__),
        },
    )


def persist():
    """Publish exact artifact set, preserving a git-side verification manifest."""
    from explore_persona_space.orchestrate.env import load_dotenv

    load_dotenv()
    from explore_persona_space.orchestrate import hub
    from explore_persona_space.orchestrate.provenance import as_metadata_dict, git_provenance

    worktree = Path(__file__).resolve().parents[1]
    result_dir = worktree / "eval_results/issue_2546/input_sae_qualitative_20260906"
    for name in ["report.md", "qualitative_annotations.json"]:
        shutil.copyfile(result_dir / name, OUT / name)
    shutil.copyfile(__file__, OUT / "driver_reference.py")
    matches = json.loads((OUT / "matches.json").read_text())
    validation = json.loads((OUT / "retrieval_validation.json").read_text())
    assert matches["description_coverage"]["with_description"] == 924
    assert matches["sae_weights_sha256"] == (
        "a90d1309919de4a7712b6d86a7d715c525c686ffe45c1255e8a689b752a8798a"
    )
    manifest = {
        "timestamp": datetime.now(UTC).isoformat(),
        "provenance": as_metadata_dict(git_provenance(cwd=worktree)),
        "retrieval_validation": validation,
        "sae_verified_revision": "c37e53c4bb07127ad17ab88f28b93d4e87142e59",
        "files": {},
    }
    files = sorted(p for p in OUT.iterdir() if p.is_file())
    for path in files:
        entry = {"bytes": path.stat().st_size, "sha256": sha(path)}
        if path.suffix == ".npz":
            with np.load(path) as z:
                entry["arrays"] = {
                    k: {"shape": list(z[k].shape), "dtype": str(z[k].dtype)} for k in z.files
                }
        manifest["files"][path.name] = entry
    prefix = "issue2546_cotmap/analysis_tensors/input_sae_qualitative_20260906"
    print("Publishing", len(files), "files", sum(p.stat().st_size for p in files), flush=True)
    url = hub._upload_folder_filtered(
        OUT,
        hub.DEFAULT_DATASET_REPO,
        "dataset",
        prefix,
        allow_patterns=[p.name for p in files],
        expected_repo_paths=[f"{prefix}/{p.name}" for p in files],
    )
    if not url:
        raise RuntimeError("Artifact upload failed; local artifacts retained")
    manifest["verified_hf_destination"] = url
    (result_dir / "artifact_manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False)
    )
    print("Verified artifact destination:", url, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["fit", "match", "qualitative", "validate", "persist"])
    args = ap.parse_args()
    {
        "fit": fit,
        "match": match,
        "qualitative": qualitative,
        "validate": validate,
        "persist": persist,
    }[args.phase]()
