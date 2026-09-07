#!/usr/bin/env python3
"""Extend the fixed task-1901 test bank with five independently seeded answers.

Generation and capture run in separate processes using the parent's exact
sampling and answer-span helpers. Every generated chunk and captured seed is
persisted before continuing; all resume keys include the complete recipe.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

import numpy as np  # noqa: E402
import torch  # noqa: E402
from huggingface_hub import HfApi, hf_hub_download  # noqa: E402

import issue1482_error_analysis as EA  # noqa: E402
import issue1482_kresample as KR  # noqa: E402
import issue1901_singleturn_retrieval_final as F  # noqa: E402
from explore_persona_space.orchestrate import hub  # noqa: E402
from explore_persona_space.orchestrate.provenance import (  # noqa: E402
    as_metadata_dict,
    git_provenance,
)
from explore_persona_space.orchestrate.secret_scrub import scan_file  # noqa: E402

LOG = logging.getLogger(__name__)
REPO = "superkaiba1/explore-persona-space-data"
PREFIX = "issue1901_k10_rollouts"
REVISION = "ce32efe1b04e488df73392ce0faa172e343ef85f"
MODEL_REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"
SEEDS = (47, 48, 49, 50, 51)
FILES = {
    "prompts": (
        "issue1901_avgtarget/raw_completions/gen_seed45.json",
        "a8aef214980fa0a5424e543633de45f9536cee9752bc4c64014c7b5ea6c454fa",
    ),
    "index": (
        "issue1901_avgpool/analysis_tensors/bundle/bundle_index.json",
        "5330cd7523db3de9a5fcf36296b73e6a2b392e8fc645881e2e9b7a03e9285159",
    ),
    "reference_text": (
        "issue1901_avgpool/raw_completions/test/shard00/gen_seed43_chunk0.json",
        "4a4d1b9d060af85c8bc8f23fea470473350f11f1b78b51a50862d002c221e931",
    ),
    "reference_vectors": (
        "issue1901_avgpool/analysis_tensors/kresample/V_test_shard00.npz",
        "b7b4c8ada5b570017c7e9fbd342a7256205d3a7c0f3c518ac26e0405f50dc805",
    ),
}
RECIPE = {
    "model": KR.MODEL_ID,
    "model_revision": MODEL_REVISION,
    "data_revision": REVISION,
    "input_hashes": {k: v[1] for k, v in FILES.items()},
    "seeds": list(SEEDS),
    "engine_seed": 42,
    "temperature": 1.0,
    "top_p": 0.95,
    "max_tokens": 1024,
    "max_model_len": 8192,
    "layer": 19,
    "capture_convention": "parent-full-template-retok-span-incl-eot-tail",
    "gen_chunk": 500,
    "capture_rows": 32,
    "capture_token_budget": 32768,
}


def upload(path: Path, relative: str) -> dict:
    """Upload an exact file and verify the remote bytes at the resulting commit."""
    api = HfApi()
    remote = f"{PREFIX}/{relative}"
    commit = hub.retry_transient(
        lambda: api.upload_file(
            path_or_fileobj=str(path), path_in_repo=remote, repo_id=REPO, repo_type="dataset"
        ),
        what=f"upload {relative}",
    )
    info = hub.retry_transient(
        lambda: api.get_paths_info(REPO, [remote], repo_type="dataset", revision=commit.oid),
        what=f"verify {relative}",
    )[0]
    assert info.size == path.stat().st_size, relative
    digest = F._sha256(path)
    if info.lfs:
        assert info.lfs.sha256 == digest, relative
    else:
        payload = path.read_bytes()
        expected = hashlib.sha1(f"blob {len(payload)}\0".encode() + payload).hexdigest()
        assert info.blob_id == expected, relative
    return {"path": remote, "revision": commit.oid, "sha256": digest, "size": info.size}


def load_rows(out: Path) -> tuple[list[dict], dict[str, Path]]:
    """Stage pinned files directly at the consumer paths and verify row identity."""
    paths = {}
    for key, (name, digest) in FILES.items():
        p = Path(
            hub.retry_transient(
                lambda name=name: hf_hub_download(
                    REPO, name, repo_type="dataset", revision=REVISION, local_dir=out / "source"
                ),
                what=f"stage {key}",
            )
        )
        assert F._sha256(p) == digest, key
        paths[key] = p
    prompts = json.loads(paths["prompts"].read_text())["rows"]
    index = [r for r in json.loads(paths["index"].read_text())["rows"] if r["src"] == "test"]
    assert len(prompts) == len(index) == 1000
    rows = []
    for i, (p, r) in enumerate(zip(prompts, index, strict=True)):
        assert p["lid"] == r["row_idx"] == i and r["ci"] == -1 - i
        assert hashlib.sha256(p["prompt"].encode()).hexdigest() == r["prompt_sha256"]
        rows.append({**r, "prompt": p["prompt"]})
    return rows, paths


def generation(args) -> None:
    """Run full-size batched draws; persist text and token IDs after each chunk."""
    from transformers import AutoTokenizer
    from explore_persona_space.eval.generation import create_vllm_engine

    rows, _ = load_rows(args.out)
    tok = hub.retry_transient(
        lambda: AutoTokenizer.from_pretrained(KR.MODEL_ID, revision=MODEL_REVISION),
        what="tokenizer fetch",
    )
    rendered = [
        tok.apply_chat_template(
            [{"role": "user", "content": r["prompt"]}], tokenize=False, add_generation_prompt=True
        )
        for r in rows
    ]
    llm = create_vllm_engine(KR.MODEL_ID, revision=MODEL_REVISION, max_model_len=8192, seed=42)
    assert llm is not None
    receipts = []
    for seed in SEEDS:
        for chunk, lo in enumerate(range(0, len(rows), 500)):
            path = args.out / "raw_completions" / f"seed{seed}_chunk{chunk}.json"
            t = time.monotonic()
            if path.exists():
                doc = json.loads(path.read_text())
                assert doc["recipe"] == RECIPE and doc["seed"] == seed
            else:
                generated = KR._generate_seed(llm, tok, rendered[lo : lo + 500], seed)
                doc = {
                    "recipe": RECIPE,
                    "seed": seed,
                    "chunk": chunk,
                    "metadata": as_metadata_dict(git_provenance(ROOT), phase="generation"),
                    "rows": [
                        {"ci": r["ci"], "row_idx": r["row_idx"], **g}
                        for r, g in zip(rows[lo : lo + 500], generated, strict=True)
                    ],
                }
                F._write_json(path, doc)
            assert [r["ci"] for r in doc["rows"]] == [-1 - i for i in range(lo, lo + 500)]
            assert all(r["prompt_token_ids"] and isinstance(r["text"], str) for r in doc["rows"])
            findings = scan_file(path)
            if findings:
                raise RuntimeError(f"Generated chunk requires secret-scan review: {path.name}")
            receipt = upload(path, f"raw_completions/{path.name}")
            receipts.append(receipt)
            F._write_json(args.out / "generation_uploads.json", receipts)
            LOG.info(
                "[generation] seed=%d chunk=%d/2 rows=500 elapsed=%.1fs tokens=%d",
                seed,
                chunk + 1,
                time.monotonic() - t,
                sum(len(r["token_ids"]) for r in doc["rows"]),
            )


def capture_rows(model, tok, rows: list[dict], texts: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Apply the exact parent retokenization, padded capture, and span reduction."""
    items = []
    for slot, (r, g) in enumerate(zip(rows, texts, strict=True)):
        assert r["ci"] == g["ci"]
        msgs, prompt = KR._prompt_render(tok, r["prompt"])
        KR._check_prompt_ids_join(r["ci"], 0, g["prompt_token_ids"], prompt)
        full = KR._parent_convention_full_ids(tok, msgs, g["text"])
        assert len(full) > len(prompt)
        items.append(
            (slot, 0, (full, len(prompt) - 1, len(prompt) - 1, len(full) - len(prompt), 0))
        )
    values = np.full((len(rows), 3584), np.nan, dtype=np.float32)
    n_ans = np.zeros(len(rows), dtype=np.int32)
    with torch.no_grad():
        for batch in KR._token_batches(items, 32, 32768):
            batch_rows = [(slot, 0, *tk) for slot, _, tk in batch]
            states = EA._batched_capture(model, tok, batch_rows, [19], "cuda")
            for (slot, _, tk), state in zip(batch, states, strict=True):
                assert state[19].shape == (len(tk[0]), 3584)
                values[slot] = state[19][tk[2] + 1 :].mean(0).numpy()
                n_ans[slot] = tk[3]
    assert np.isfinite(values).all() and np.all(n_ans > 0)
    return values.astype(np.float16), n_ans


def capture(args) -> None:
    """Validate old-vector parity, then checkpoint and upload each new seed."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rows, paths = load_rows(args.out)
    tok = AutoTokenizer.from_pretrained(KR.MODEL_ID, revision=MODEL_REVISION)
    model = hub.retry_transient(
        lambda: AutoModelForCausalLM.from_pretrained(
            KR.MODEL_ID, revision=MODEL_REVISION, torch_dtype=torch.bfloat16
        ),
        what="capture model",
    )
    model.to("cuda").eval()
    probe_rows = np.linspace(0, 499, 32, dtype=int)
    old = json.loads(paths["reference_text"].read_text())["rows"]
    probe_text = [{**old[i], "text": old[i]["response"]} for i in probe_rows]
    recaptured, _ = capture_rows(model, tok, [rows[i] for i in probe_rows], probe_text)
    with np.load(paths["reference_vectors"]) as z:
        by_ci = {int(ci): i for i, ci in enumerate(z["ci"])}
        ref = z["V"][[by_ci[-1 - i] for i in probe_rows], 0].astype(np.float64)
    got = recaptured.astype(np.float64)
    cosine = np.sum(got * ref, 1) / (np.linalg.norm(got, axis=1) * np.linalg.norm(ref, axis=1))
    rel = np.linalg.norm(got - ref, axis=1) / np.linalg.norm(ref, axis=1)
    parity = {
        "n": 32,
        "cosine_min": float(cosine.min()),
        "cosine_mean": float(cosine.mean()),
        "relative_l2_max": float(rel.max()),
        "relative_l2_mean": float(rel.mean()),
        "cosine_floor": 0.999,
        "source": "issue1901_opsurface span-mean equivalence",
    }
    F._write_json(args.out / "capture_parity.json", parity)
    assert cosine.min() >= 0.999, parity
    receipts = [upload(args.out / "capture_parity.json", "capture_parity.json")]
    LOG.info("[capture] banked parity PASS %s", parity)
    if args.phase == "parity":
        return
    for seed in SEEDS:
        t = time.monotonic()
        path = args.out / "analysis_tensors" / f"seed{seed}.npz"
        texts = []
        for chunk in range(2):
            doc = json.loads(
                (args.out / "raw_completions" / f"seed{seed}_chunk{chunk}.json").read_text()
            )
            assert doc["recipe"] == RECIPE and doc["seed"] == seed
            texts.extend(doc["rows"])
        generation_sha = hashlib.sha256(json.dumps(texts, sort_keys=True).encode()).hexdigest()
        if not path.exists():
            values, counts = capture_rows(model, tok, rows, texts)
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp.npz")
            np.savez(
                tmp,
                V=values,
                n_ans=counts,
                ci=np.array([r["ci"] for r in rows]),
                seed=seed,
                recipe=json.dumps(RECIPE, sort_keys=True),
                generation_sha=generation_sha,
            )
            os.replace(tmp, path)
        with np.load(path) as z:
            assert z["V"].shape == (1000, 3584) and np.isfinite(z["V"]).all()
            assert np.array_equal(z["ci"], [-1 - i for i in range(1000)])
            assert int(z["seed"]) == seed and np.all(z["n_ans"] > 0)
            assert json.loads(str(z["recipe"])) == RECIPE
            assert str(z["generation_sha"]) == generation_sha
        receipts.append(upload(path, f"analysis_tensors/{path.name}"))
        F._write_json(args.out / "capture_uploads.json", receipts)
        LOG.info("[capture] seed=%d completed 1000/1000 elapsed=%.1fs", seed, time.monotonic() - t)
    report = {
        "recipe": RECIPE,
        "new_vectors": 5000,
        "parity": parity,
        "uploads": receipts,
        "metadata": as_metadata_dict(git_provenance(ROOT), phase="capture"),
    }
    F._write_json(args.out / "run_summary.json", report)
    upload(args.out / "run_summary.json", "run_summary.json")


def main() -> None:
    """Dispatch actual GPU phases in fresh processes; support a read-only input check."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--phase", choices=["inputs", "parity", "generation", "capture", "all"], required=True
    )
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.phase == "inputs":
        rows, _ = load_rows(args.out)
        LOG.info("Verified %d pinned input rows", len(rows))
    elif args.phase == "all":
        for phase in ["parity", "generation", "capture"]:
            LOG.info("[phase=%s] starting fresh child", phase)
            subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--phase",
                    phase,
                    "--out",
                    str(args.out),
                ],
                check=True,
            )
        F._write_json(args.out / "exit.json", {"exit_code": 0, "completed_at": time.time()})
        F._write_json(
            Path("/workspace/logs") / f"issue-1901-epm_results-k10-{int(time.time())}.json",
            {
                "sentinel_schema_version": 1,
                "kind": "epm:results",
                "version": 1,
                "task_id": 1901,
                "note": "K=10 capture stage complete: 5,000 new answer vectors and raw draws "
                "uploaded and verified. This is the capture-stage milestone; paired K5/K10 "
                "scoring remains for the owning Codex session. Run summary: "
                "HF dataset superkaiba1/explore-persona-space-data/issue1901_k10_rollouts.",
            },
        )
        LOG.info("[phase=done]")  # Standalone top-level dispatcher, not a per-cell worker.
    else:
        assert torch.cuda.is_available(), "GPU generation/capture requires CUDA"
        {"generation": generation, "parity": capture, "capture": capture}[args.phase](args)


if __name__ == "__main__":
    main()
