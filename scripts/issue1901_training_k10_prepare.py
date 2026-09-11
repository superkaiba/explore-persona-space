#!/usr/bin/env python3
"""Prepare pinned context/answer banks for task 1901 training-K retraining.

Only the forty source capture chunks covering the declared 19k IDs are staged.
No model is loaded, fitted, or evaluated. Original fp32 answers remain fp32;
fresh answers retain their banked fp16 precision. A manifest written last binds
all portable outputs and permits fail-loud validation before GPU work.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

import numpy as np  # noqa: E402
import torch  # noqa: E402
from huggingface_hub import HfApi, hf_hub_download  # noqa: E402

import issue1901_singleturn_retrieval_final as FINAL  # noqa: E402
import issue779_fitter_fair_comparison as F  # noqa: E402
from explore_persona_space.orchestrate import hub  # noqa: E402

LOG = logging.getLogger(__name__)
REPO = "superkaiba1/explore-persona-space-data"
REVISION = "77f04fcdf169b5a9d4aa304c9c6efbab0526c0e2"
MODEL_REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"
MODEL = "Qwen/Qwen2.5-7B-Instruct"
SCHEMA = "issue1901-training-k10-inputs-v1"
N_TRAIN, N_TEST, HIDDEN = 19000, 1000, 3584
N1M = "issue779_monitoring/fitter-fair-comparison-n1m"
AVG = "issue1901_avgpool"
INDEX = f"{AVG}/analysis_tensors/bundle/bundle_index.json"
PASS_B = "issue779_monitoring/analysis_tensors/pass_b/train_context_vectors.pt"
DISTRACTORS = "issue1901_metrics/analysis_tensors/distractors_L19.npz"
TEST_PROMPTS = "issue1901_avgtarget/raw_completions/gen_seed45.json"
TEST_ROW_SHA = "b9377786b24bc9c1c360303fdb8fac86c0097d264479de1dca3c23dd1047d31d"
INDEX_SHA = "5330cd7523db3de9a5fcf36296b73e6a2b392e8fc645881e2e9b7a03e9285159"
DISTRACTOR_SHA = "8015d9d4dd2d644ded6eecfca150168bdaeab479cea3eac3735942f4d46c94a5"
BUNDLE_SHA = "c524a8beaeadeba183938f71ced8e65a3479a323bdc2e50ba962cbd832f6d089"


def sha256(path: Path) -> str:
    """Hash an existing file without materializing it in memory."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path: Path, value: object, *, compact: bool = False) -> None:
    """Atomically replace JSON; compact text banks stay below the non-LFS file ceiling."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.json")
    text = (
        json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        if compact
        else (json.dumps(value, ensure_ascii=False, indent=2))
    )
    tmp.write_text(text + "\n")
    os.replace(tmp, path)


def write_npz(path: Path, **arrays: np.ndarray) -> None:
    """Atomically persist uncompressed arrays, preserving each input dtype."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


def ordered_indices(source: np.ndarray, wanted: np.ndarray) -> np.ndarray:
    """Join unique integer IDs, refusing duplicate sources or missing targets."""
    assert source.ndim == wanted.ndim == 1
    assert len(np.unique(source)) == len(source), "duplicate source IDs"
    assert len(np.unique(wanted)) == len(wanted), "duplicate requested IDs"
    lookup = {int(ci): i for i, ci in enumerate(source)}
    missing = set(wanted.tolist()) - lookup.keys()
    assert not missing, f"missing requested IDs: {sorted(missing)[:10]}"
    return np.asarray([lookup[int(ci)] for ci in wanted], dtype=np.int64)


def source_chunk_names(cis: np.ndarray) -> list[str]:
    """Map n1m IDs to the verified 32-shard, 500-row original capture layout."""
    assert cis.ndim == 1 and len(cis) and np.all((cis >= 0) & (cis < 960000))
    return sorted({f"shard{ci // 30000:02d}_chunk{ci % 30000 // 500:04d}.pt" for ci in cis})


def extract_chunk(blob: dict, wanted: np.ndarray, expected: dict[int, dict]):
    """Read L19 context/original answer vectors and hash-check joined prompt text."""
    required = {"cx_last", "v_x", "ci", "prompts", "layers", "shard_index", "chunk"}
    assert required <= blob.keys(), f"missing capture keys: {required - blob.keys()}"
    cis = np.asarray(blob["ci"], dtype=np.int64)
    assert len(set(cis.tolist())) == len(cis)
    assert blob["layers"] == [14, 19, 26]
    for key in ("cx_last", "v_x"):
        assert blob[key].shape == (len(cis), 3, HIDDEN)
        assert blob[key].dtype == torch.float32
    assert len(blob["prompts"]) == len(cis)
    selected = wanted[np.isin(wanted, cis)]
    indices = ordered_indices(cis, selected)
    rows = []
    for ci, i in zip(selected, indices, strict=True):
        prompt = blob["prompts"][int(i)]
        digest = hashlib.sha256(prompt.encode()).hexdigest()
        assert digest == expected[int(ci)]["prompt_sha256"], f"prompt mismatch ci={ci}"
        rows.append({"ci": int(ci), "prompt": prompt, "prompt_sha256": digest})
    x = blob["cx_last"][indices, 1, :].numpy()
    y = blob["v_x"][indices, 1, :].numpy()
    assert np.isfinite(x).all() and np.isfinite(y).all()
    return selected, x, y, rows


class Stage:
    """Pin sources once, verify bytes, and reuse only explicitly located local copies."""

    def __init__(self, out: Path, revision: str, local_root: Path | None):
        assert re.fullmatch(r"[a-f0-9]{40}", revision), "revision must be a commit SHA"
        self.out, self.revision, self.local_root = out, revision, local_root
        self.sources: dict[str, dict] = {}
        self.api = HfApi()

    def stage(self, rel: str) -> Path:
        """Open precisely the source revision and verify its LFS or Git blob digest."""
        info = hub.retry_transient(
            lambda: self.api.get_paths_info(
                REPO, [rel], repo_type="dataset", revision=self.revision
            ),
            what=f"source metadata {rel}",
        )
        assert len(info) == 1 and info[0].path == rel, f"missing pinned source: {rel}"
        remote = info[0]
        candidates = [self.out / "source" / rel]
        if self.local_root is not None:
            candidates += [
                self.local_root / "data/issue_1901" / name / rel
                for name in ("fig2_pool10k", "figure2_five_rollout_scaling", "ctxnn_dl")
            ]
        path = next((p for p in candidates if p.is_file()), None)
        if path is None:
            path = Path(
                hub.retry_transient(
                    lambda: hf_hub_download(
                        REPO,
                        rel,
                        repo_type="dataset",
                        revision=self.revision,
                        local_dir=self.out / "source",
                    ),
                    what=f"source download {rel}",
                )
            )
        assert path.stat().st_size == remote.size, f"source size mismatch: {rel}"
        digest = sha256(path)
        if remote.lfs:
            assert digest == remote.lfs.sha256, f"source LFS digest mismatch: {rel}"
        else:
            payload = path.read_bytes()
            git_hash = hashlib.sha1(f"blob {len(payload)}\0".encode() + payload).hexdigest()
            assert git_hash == remote.blob_id, f"source Git digest mismatch: {rel}"
        self.sources[rel] = {
            "path": rel,
            "revision": self.revision,
            "size": remote.size,
            "sha256": digest,
            "blob_id": remote.blob_id,
        }
        return path

    def json(self, rel: str) -> dict:
        """Read a byte-verified pinned JSON source."""
        return json.loads(self.stage(rel).read_text())


def assemble_old_draws(paths: list[Path], cis: np.ndarray) -> tuple[np.ndarray, dict[int, int]]:
    """Join four capture shards once by CI, retaining their stored fp16 precision."""
    values, all_cis, shards = [], [], {}
    for shard, path in enumerate(paths):
        with np.load(path, allow_pickle=False) as z:
            assert {"V", "ci", "draws", "n_ans", "src"} <= set(z.files)
            v, ids = z["V"], z["ci"]
            assert v.shape == (len(ids), 4, HIDDEN) and v.dtype == np.float16
            assert np.array_equal(z["draws"], [43, 44, 45, 46])
            assert z["n_ans"].shape == (len(ids), 4) and np.all(z["n_ans"] > 0)
            assert np.all(z["src"] == "distr") and np.isfinite(v).all()
            assert not (set(ids.tolist()) & shards.keys()), "duplicate IDs across draw shards"
            shards.update({int(ci): shard for ci in ids})
            values.append(v)
            all_cis.append(ids)
    flat_ids = np.concatenate(all_cis)
    assert set(flat_ids.tolist()) == set(cis.tolist()), "fresh bank coverage mismatch"
    return np.concatenate(values)[ordered_indices(flat_ids, cis)], shards


def prepare_test(stage: Stage, expected: list[dict], out: Path) -> list[dict]:
    """Assemble the original test split plus all nine banked fresh answer draws."""
    pb = torch.load(stage.stage(PASS_B), map_location="cpu", mmap=True, weights_only=True)
    assert {"cx_last", "v_x", "layers"} <= pb.keys()
    assert len(pb["cx_last"]) == 5000 and pb["cx_last"].shape == (5000, 28, HIDDEN)
    _, _, rows = F.fixed_split(5000, 3600, 400, 1000, 42)
    assert hashlib.sha256(rows.astype(np.int64).tobytes()).hexdigest() == TEST_ROW_SHA
    li = pb["layers"].index(19)
    x = pb["cx_last"][rows, li].float().numpy()
    original = pb["v_x"][rows, li].float().numpy()
    del pb
    cis = -1 - np.arange(N_TEST, dtype=np.int64)
    assert [r["ci"] for r in expected] == cis.tolist()
    with np.load(
        stage.stage(f"{AVG}/analysis_tensors/kresample/V_test_shard00.npz"), allow_pickle=False
    ) as z:
        idx = ordered_indices(z["ci"], cis)
        assert z["V"].shape == (N_TEST, 4, HIDDEN) and z["V"].dtype == np.float16
        assert np.array_equal(z["draws"], [43, 44, 45, 46])
        assert np.all(z["src"] == "test") and np.all(z["n_ans"] > 0)
        fresh = [z["V"][idx]]
    summary = stage.json("issue1901_k10_rollouts/run_summary.json")
    assert summary["new_vectors"] == 5000 and summary["parity"]["cosine_min"] >= 0.999
    recipe = summary["recipe"]
    assert recipe["model"] == MODEL and recipe["model_revision"] == MODEL_REVISION
    assert recipe["seeds"] == [47, 48, 49, 50, 51] and recipe["layer"] == 19
    assert {k: recipe[k] for k in ("temperature", "top_p", "max_tokens", "max_model_len")} == {
        "temperature": 1.0,
        "top_p": 0.95,
        "max_tokens": 1024,
        "max_model_len": 8192,
    }
    receipts = {r["path"]: r for r in summary["uploads"]}
    for seed in range(47, 52):
        rel = f"issue1901_k10_rollouts/analysis_tensors/seed{seed}.npz"
        p = stage.stage(rel)
        assert sha256(p) == receipts[rel]["sha256"]
        with np.load(p, allow_pickle=False) as z:
            assert int(z["seed"]) == seed and json.loads(str(z["recipe"])) == recipe
            assert z["V"].shape == (N_TEST, HIDDEN) and z["V"].dtype == np.float16
            assert np.array_equal(z["ci"], cis) and np.all(z["n_ans"] > 0)
            fresh.append(z["V"][:, None, :])
    target = np.concatenate(fresh, axis=1)
    assert np.isfinite(x).all() and np.isfinite(original).all() and np.isfinite(target).all()
    view = FINAL.make_eval_view(original, N_TEST, "keep_one")
    assert len(view.pred_rows) == len(view.pool_rows) == 942
    assert np.array_equal(view.pred_rows, view.pool_rows)
    write_npz(
        out / "test.npz",
        X=x,
        Y_original=original,
        Y_fresh=target,
        ci=cis,
        fresh_seeds=np.arange(43, 52),
        pass_b_rows=rows,
        dedup_rows=view.pred_rows,
    )
    text = stage.json(TEST_PROMPTS)["rows"]
    assert len(text) == N_TEST
    result = []
    for i, (p, entry) in enumerate(zip(text, expected, strict=True)):
        assert p["lid"] == entry["row_idx"] == i
        digest = hashlib.sha256(p["prompt"].encode()).hexdigest()
        assert digest == entry["prompt_sha256"]
        result.append({"ci": entry["ci"], "prompt": p["prompt"], "prompt_sha256": digest})
    return result


def prepare_parity(
    stage: Stage, prompts: list[dict], fresh: np.ndarray, shards: dict[int, int], out: Path
) -> None:
    """Persist 32 deterministic seed43 examples per shard and their exact vectors."""
    by_ci = {r["ci"]: (i, r) for i, r in enumerate(prompts)}
    rows, vectors, ids = [], [], []
    for shard in range(4):
        rel = f"{AVG}/raw_completions/distr/shard{shard:02d}/gen_seed43_chunk0.json"
        doc = stage.json(rel)
        assert doc["meta"]["bundle_sha"] == BUNDLE_SHA
        assert doc["meta"]["seed"] == 43 and doc["meta"]["shard"] == shard
        assert len(doc["rows"]) == 500
        # Across the first production chunk, spread the fixed probe over row position.
        for position in np.linspace(0, 499, 32, dtype=np.int64):
            generated = doc["rows"][int(position)]
            ci = int(generated["ci"])
            i, prompt = by_ci[ci]
            assert shards[ci] == shard and generated["src"] == "distr"
            assert generated["prompt_token_ids"] and generated["token_ids"]
            assert isinstance(generated["response"], str)
            rows.append(
                {
                    **prompt,
                    "seed": 43,
                    "shard": shard,
                    "expected_index": len(rows),
                    "generated": {
                        "text": generated["response"],
                        "prompt_token_ids": generated["prompt_token_ids"],
                        "token_ids": generated["token_ids"],
                    },
                }
            )
            vectors.append(fresh[i, 0])
            ids.append(ci)
    assert len(rows) == len(set(ids)) == 128
    write_json(out / "parity.json", {"rows": rows})
    write_npz(out / "parity.npz", V=np.stack(vectors), ci=np.asarray(ids, dtype=np.int64))


def validate(out: Path, revision: str) -> dict:
    """Validate portable output hashes, array contracts, split identities, and parity joins."""
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["schema_version"] == SCHEMA and manifest["revision"] == revision
    required = {"train.npz", "test.npz", "prompts.json", "parity.json", "parity.npz"}
    assert set(manifest["outputs"]) == required
    for name, receipt in manifest["outputs"].items():
        assert (out / name).stat().st_size == receipt["size"]
        assert sha256(out / name) == receipt["sha256"], f"output hash mismatch: {name}"
    prompts = json.loads((out / "prompts.json").read_text())
    for split, n, draws in (("train", N_TRAIN, 4), ("test", N_TEST, 9)):
        with np.load(out / f"{split}.npz", allow_pickle=False) as z:
            for key in ("X", "Y_original"):
                assert z[key].shape == (n, HIDDEN) and z[key].dtype == np.float32
                assert np.isfinite(z[key]).all()
            assert z["Y_fresh"].shape == (n, draws, HIDDEN)
            assert z["Y_fresh"].dtype == np.float16 and np.isfinite(z["Y_fresh"]).all()
            assert np.array_equal(z["fresh_seeds"], np.arange(43, 43 + draws))
            assert np.array_equal(z["ci"], [r["ci"] for r in prompts[split]])
            assert len(np.unique(z["ci"])) == n
            if split == "test":
                assert np.array_equal(z["ci"], -1 - np.arange(n))
                assert hashlib.sha256(z["pass_b_rows"].tobytes()).hexdigest() == TEST_ROW_SHA
                assert len(z["dedup_rows"]) == 942
    hashes = [{r["prompt_sha256"] for r in prompts[s]} for s in ("train", "test")]
    assert len(hashes[0]) == N_TRAIN and not hashes[0] & hashes[1]
    for split in prompts.values():
        for row in split:
            assert hashlib.sha256(row["prompt"].encode()).hexdigest() == row["prompt_sha256"]
    parity = json.loads((out / "parity.json").read_text())["rows"]
    with np.load(out / "parity.npz", allow_pickle=False) as z:
        assert z["V"].shape == (128, HIDDEN) and np.isfinite(z["V"]).all()
        assert z["ci"].tolist() == [r["ci"] for r in parity]
        assert [r["expected_index"] for r in parity] == list(range(128))
    return manifest


def prepare(out: Path, revision: str, local_root: Path | None) -> dict:
    """Prepare every declared input with provenance and exact realized-coverage checks."""
    if (out / "manifest.json").exists():
        return validate(out, revision)
    out.mkdir(parents=True, exist_ok=True)
    stage = Stage(out, revision, local_root)
    index = stage.json(INDEX)
    assert stage.sources[INDEX]["sha256"] == INDEX_SHA and index["ci_sha256"] == BUNDLE_SHA
    train_rows = [r for r in index["rows"] if r["src"] == "distr"]
    test_rows = [r for r in index["rows"] if r["src"] == "test"]
    assert len(train_rows) == N_TRAIN and len(test_rows) == N_TEST
    cis = np.asarray([r["ci"] for r in train_rows], dtype=np.int64)
    expected = {r["ci"]: r for r in train_rows}
    assert len(expected) == N_TRAIN and np.all(cis >= 0)
    meta = stage.json(f"{N1M}/sampling_manifest/meta.json")
    avgmeta = stage.json(f"{AVG}/analysis_tensors/bundle/bundle_meta.json")
    assert meta["model"] == avgmeta["gen_recipe"]["model"] == MODEL
    assert meta["n_new"] == 960000 and meta["near_dupe"]["n_targets"] == 1400
    assert meta["near_dupe"]["ngram"] == 5 and meta["near_dupe"]["jaccard_thresh"] == 0.8
    assert meta["used_shas"]["round1"] == avgmeta["round1_sha256"]
    assert avgmeta["test_idx_sha256"] == TEST_ROW_SHA
    names = source_chunk_names(cis)
    assert len(names) == 40, "declared 19k source chunk coverage changed"
    parts = []
    for k, name in enumerate(names):
        started = time.monotonic()
        p = stage.stage(f"{N1M}/final_token_capture/{name}")
        blob = torch.load(p, map_location="cpu", mmap=True, weights_only=True)
        part = extract_chunk(blob, cis, expected)
        assert len(part[0]), f"source chunk selected no IDs: {name}"
        parts.append(part)
        del blob
        LOG.info(
            "[prepare] chunk %d/%d %s kept=%d elapsed=%.1fs",
            k + 1,
            len(names),
            name,
            len(part[0]),
            time.monotonic() - started,
        )
    selected = np.concatenate([p[0] for p in parts])
    order = ordered_indices(selected, cis)
    assert len(selected) == N_TRAIN
    x = np.concatenate([p[1] for p in parts])[order]
    original = np.concatenate([p[2] for p in parts])[order]
    prompt_rows = [r for p in parts for r in p[3]]
    prompts = [prompt_rows[int(i)] for i in order]
    del parts
    p = stage.stage(DISTRACTORS)
    assert stage.sources[DISTRACTORS]["sha256"] == DISTRACTOR_SHA
    with np.load(p, allow_pickle=False) as z:
        assert np.array_equal(z["ci"][:N_TRAIN], cis)
        assert np.all(z["corpus"][:N_TRAIN] == "lmsys")
        assert np.array_equal(z["vx"][:N_TRAIN], original), "original answer parity failed"
    paths = [
        stage.stage(f"{AVG}/analysis_tensors/kresample/V_distr_shard{s:02d}.npz") for s in range(4)
    ]
    fresh, shards = assemble_old_draws(paths, cis)
    write_npz(
        out / "train.npz",
        X=x,
        Y_original=original,
        Y_fresh=fresh,
        ci=cis,
        fresh_seeds=np.arange(43, 47),
    )
    prepare_parity(stage, prompts, fresh, shards, out)
    del x, original, fresh
    test_prompts = prepare_test(stage, test_rows, out)
    assert not (
        {r["prompt_sha256"] for r in prompts} & {r["prompt_sha256"] for r in test_prompts}
    ), "train/test prompt overlap"
    write_json(out / "prompts.json", {"train": prompts, "test": test_prompts}, compact=True)
    assert (out / "prompts.json").stat().st_size < 9_500_000, "split oversized prompt text"
    names = ["train.npz", "test.npz", "prompts.json", "parity.json", "parity.npz"]
    manifest = {
        "schema_version": SCHEMA,
        "revision": revision,
        "repo": REPO,
        "model": MODEL,
        "model_revision": MODEL_REVISION,
        "layer": 19,
        "train_rows": N_TRAIN,
        "test_rows": N_TEST,
        "retrieval_candidates": 942,
        "bundle_ci_sha256": BUNDLE_SHA,
        "test_rows_sha256": TEST_ROW_SHA,
        "original_draw": "seed42",
        "fresh_seeds_train": list(range(43, 47)),
        "fresh_seeds_test": list(range(43, 52)),
        "near_duplicate_provenance": meta["near_dupe"],
        "near_duplicate_limitation": "inherited parent screen; not independently recomputed",
        "source_files": list(stage.sources.values()),
        "outputs": {
            n: {"size": (out / n).stat().st_size, "sha256": sha256(out / n)} for n in names
        },
    }
    write_json(out / "manifest.json", manifest)
    return validate(out, revision)


def main() -> None:
    """Parse the small preparation/validation interface and run the requested CPU phase."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--revision", default=REVISION)
    parser.add_argument("--phase", choices=("prepare", "validate"), default="prepare")
    parser.add_argument("--local-root", type=Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    result = (
        prepare(args.out, args.revision, args.local_root)
        if args.phase == "prepare"
        else validate(args.out, args.revision)
    )
    LOG.info(
        "[prepare] PASS train=%d test=%d inputs=%d",
        result["train_rows"],
        result["test_rows"],
        len(result["source_files"]),
    )


if __name__ == "__main__":
    main()
