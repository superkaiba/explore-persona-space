#!/usr/bin/env python3
"""Four exact boundary ridge fits: expanded Wikipedia train, frozen WikiText eval.

Inherited #1901: Qwen2.5-7B-Instruct L19, 4096-token article cap, 512-token
article floor, 8..256 target tokens, preceding 8..96 tokens, <=6 rows/article/ID,
ridge validation grid, article bootstrap 1000 and shuffled-pair null 200.
Wikipedia is uniformly Moses-tokenized (sacremoses 0.1.1) with the published
WikiText numerical-punctuation postprocessor. This is a corpus-transfer read,
not a sample-size-only comparison. No evaluation inputs are reformatted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import os
import re
import sys
import time
import unicodedata
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import issue931_common as C  # noqa: E402

TOKENS = (659, 13, 937, 753)
WIKI_REV = "b04c8d1ceb2f5cd4588862100d08de323dccfbaa"
REPO = "superkaiba1/explore-persona-space-data"
PREFIX = "issue1901_boundary25k"
SEED = 190101
FORMAT = "sacremoses-0.1.1-en-aggressive-dash-noescape-no-normalizer-wikitext-numbers-v2"


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")
    tmp.replace(path)


def read_jsonl(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def normalized_title(title):
    return "".join(c for c in unicodedata.normalize("NFKC", title).casefold() if c.isalnum())


def write_jsonl_parts(directory, stem, records):
    """Atomic <=9 MB plain-text shards; never gzip research text."""
    directory.mkdir(parents=True, exist_ok=True)
    names, buf, size = [], [], 0

    def flush():
        name = f"{stem}-{len(names):04d}.jsonl"
        path = directory / name
        tmp = path.with_suffix(".jsonl.tmp")
        tmp.write_bytes(b"".join(buf))
        tmp.replace(path)
        names.append(name)

    for record in records:
        line = (json.dumps(record, ensure_ascii=False) + "\n").encode()
        assert len(line) < 9_000_000, "single article exceeds text shard limit"
        if buf and size + len(line) > 9_000_000:
            flush()
            buf, size = [], 0
        buf.append(line)
        size += len(line)
    if buf:
        flush()
    return names


def upload_verified(directory, suffix):
    """Verify every uploaded path, byte size, and Hub git/LFS content hash."""
    from huggingface_hub import HfApi
    from huggingface_hub.hf_api import RepoFile

    api = HfApi()
    prefix = f"{PREFIX}/{suffix}"
    files = sorted(p for p in directory.rglob("*") if p.is_file() and not p.name.endswith(".tmp"))
    assert files, directory
    commit = api.upload_folder(
        repo_id=REPO,
        repo_type="dataset",
        folder_path=directory,
        path_in_repo=prefix,
        ignore_patterns=["*.tmp"],
        commit_message=f"#1901 boundary25k {suffix}",
    )
    remote = {
        x.path: x
        for x in api.list_repo_tree(
            REPO, path_in_repo=prefix, repo_type="dataset", revision=commit.oid, recursive=True
        )
        if isinstance(x, RepoFile)
    }
    for path in files:
        dest = f"{prefix}/{path.relative_to(directory).as_posix()}"
        info = remote[dest]
        assert info.size == path.stat().st_size, dest
        if info.lfs:
            assert info.lfs.sha256 == sha_file(path), dest
        else:
            content = path.read_bytes()
            assert (
                info.blob_id
                == hashlib.sha1(f"blob {len(content)}\0".encode() + content).hexdigest()
            ), dest
    return {
        "repo": REPO,
        "prefix": prefix,
        "revision": commit.oid,
        "files": len(files),
        "bytes": sum(p.stat().st_size for p in files),
    }


def stage(suffix, directory):
    from huggingface_hub import HfApi, hf_hub_download
    from huggingface_hub.hf_api import RepoFile

    api = HfApi()
    rev = api.repo_info(REPO, repo_type="dataset").sha
    prefix = f"{PREFIX}/{suffix}"
    files = [
        x.path
        for x in api.list_repo_tree(
            REPO, path_in_repo=prefix, revision=rev, repo_type="dataset", recursive=True
        )
        if isinstance(x, RepoFile)
    ]
    assert files, prefix

    def get(name):
        cached = Path(hf_hub_download(REPO, name, repo_type="dataset", revision=rev))
        target = directory / Path(name).relative_to(prefix)
        target.parent.mkdir(parents=True, exist_ok=True)
        import shutil

        shutil.copyfile(cached, target)

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(get, files))
    return rev


def phase_inputs(args):
    """Freeze banked evaluation records and exclude all old corpus titles."""
    import issue1901_individual_boundary_tokens as I
    import pyarrow.parquet as pq
    import torch

    source = args.original
    rows, ids, meta = I._load_selected(source / "manifest")
    assert len(rows) == 7040
    assert (
        meta["selected_row_ids_sha256"]
        == "09b92f8a2b9275e46a4b5809b9c1fade56df6159bf703a9bc45e8cb68250ab63"
    )
    evaluation = [r for r in rows if r["split"] in ("val", "test")]
    assert len(evaluation) == 2240
    tokenizer = C.get_tokenizer()
    directory = args.out_root / "inputs"
    directory.mkdir(parents=True, exist_ok=True)
    articles = sorted({r["article_id"] for r in evaluation})
    torch.save(
        {
            "window_ids": articles,
            "input_ids": [torch.tensor(ids[a], dtype=torch.int32) for a in articles],
        },
        directory / "eval_articles.pt",
    )
    write_jsonl_parts(directory, "eval_rows", evaluation)
    texts = [tokenizer.decode(ids[r["article_id"]][slice(*r["t_span"])]) for r in evaluation]
    write_json(directory / "eval_texts.json", texts)
    header = re.compile(r"^ ?= [^=].* = ?$")
    titles = set()
    for path in sorted(args.wikitext_dir.glob("train-*.parquet")):
        for batch in pq.ParquetFile(path).iter_batches(batch_size=8192, columns=["text"]):
            for line in batch.column(0).to_pylist():
                if header.match(line.rstrip("\n")):
                    titles.add(normalized_title(line.strip().strip("= ").strip()))
    assert len(titles) > 28000, len(titles)
    write_json(directory / "excluded_titles.json", sorted(titles))
    wanted = {r["row_id"] for r in evaluation}
    found = set()
    target_store = directory / "eval_store"
    target_store.mkdir(exist_ok=True)
    for path in sorted((source / "store").glob("pairs_shard*.pt")):
        obj = torch.load(path, map_location="cpu", weights_only=True)
        take = [i for i, r in enumerate(obj["row_ids"]) if r in wanted]
        if not take:
            continue
        idx = torch.tensor(take)
        selected = {k: [obj[k][i] for i in take] for k in ("row_ids", "group_ids", "char_ids")}
        selected["arrays"] = {k: v[idx] for k, v in obj["arrays"].items()}
        torch.save(selected, target_store / path.name)
        found.update(selected["row_ids"])
    assert found == wanted
    from huggingface_hub import HfApi

    write_json(
        directory / "provenance.json",
        {
            "parent_manifest": meta,
            "eval_rows": len(evaluation),
            "excluded_titles": len(titles),
            "model_revision": HfApi().model_info(C.MODEL_ID).sha,
            "wiki_revision": WIKI_REV,
            "preprocessing": FORMAT,
            "preprocessing_sources": [
                "https://arxiv.org/html/1609.07843#S4.SS3",
                "https://gist.github.com/Smerity/59f2475a67aeefd24d966443819600f5",
            ],
            "hashes": {
                p.relative_to(directory).as_posix(): sha_file(p)
                for p in directory.rglob("*")
                if p.is_file() and p.name != "provenance.json"
            },
        },
    )
    print(json.dumps(upload_verified(directory, "inputs")), flush=True)


def init_worker(inputs):
    global TOKENIZER, MOSES, GATE, EXCLUDED
    from issue779_ffc_n1m_generate_capture import NearDupeGate
    from sacremoses import MosesTokenizer
    from transformers import AutoTokenizer

    inputs = Path(inputs)
    provenance = json.loads((inputs / "provenance.json").read_text())
    TOKENIZER = AutoTokenizer.from_pretrained(C.MODEL_ID, revision=provenance["model_revision"])
    MOSES = MosesTokenizer(lang="en")
    GATE = NearDupeGate(json.loads((inputs / "eval_texts.json").read_text()), ngram=5, thresh=0.8)
    EXCLUDED = set(json.loads((inputs / "excluded_titles.json").read_text()))


def wiki_format(text, moses):
    lines = []
    for line in text.split("\n"):
        tokens = moses.tokenize(line, escape=False, aggressive_dash_splits=True)
        line = " ".join(tokens).replace("< formula >", "<formula>")
        tokens = [
            re.sub(r"([,.])", r" @\1@ ", t) if re.fullmatch(r"([0-9]+[,.]?)+", t) else t
            for t in line.split()
        ]
        lines.append(" ".join(["", *tokens, "\n"]))
    return "".join(lines)


def process_article(job):
    from issue779_ffc_n1m_generate_capture import _norm
    from issue931_build_pairs import armc_eligible_anchors

    article, active = job
    if normalized_title(article["title"]) in EXCLUDED:
        return {"drop": "old_title"}
    if not any({659: ".", 13: ".", 937: "?", 753: "!"}[t] in article["text"] for t in active):
        return {"drop": "no_active_punctuation"}
    text = wiki_format(article["text"], MOSES)
    # Truncation preserves the old first-4096-token input and target convention.
    encoded = TOKENIZER(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
        truncation=True,
        max_length=4096,
    )
    ids = encoded["input_ids"]
    if len(ids) < 512:
        return {"drop": "short_article"}
    import numpy as np

    offsets = np.asarray(encoded["offset_mapping"])
    prefix = text[: int(offsets[-1, 1])]
    eligible = armc_eligible_anchors(ids, offsets, prefix)
    aid = f"wikipedia:20231101.en:{article['id']}"
    candidates = []
    for t, lo, hi, pslo, pshi, sep in eligible:
        tid = ids[t]
        if tid not in active:
            continue
        rowid = f"{aid}:a{t}"
        key = hashlib.sha256(f"{SEED}:{rowid}".encode()).hexdigest()
        candidates.append(
            (
                key,
                {
                    "row_id": rowid,
                    "article_id": aid,
                    "anchor_pos": t,
                    "c_span": [pslo, pshi],
                    "t_span": [lo, hi],
                    "n_span_tokens": hi - lo,
                    "sep_char": sep,
                    "split": "train",
                    "boundary_token_id": tid,
                    "boundary_token": TOKENIZER.convert_ids_to_tokens(tid),
                },
            )
        )
    selected, counts, dropped = [], Counter(), Counter()
    for _, row in sorted(candidates):
        tid = row["boundary_token_id"]
        if counts[tid] >= 6:
            continue
        target = TOKENIZER.decode(ids[slice(*row["t_span"])])
        if not _norm(target):
            dropped["empty_target"] += 1
            continue
        if GATE.is_dupe(target):
            dropped["eval_duplicate"] += 1
            continue
        row["target_norm"] = _norm(target)
        selected.append(row)
        counts[tid] += 1
    if not selected:
        return {"drop": "no_eligible_rows", "screening": dict(dropped)}
    return {
        "article_id": aid,
        "source": article,
        "processed_text": text,
        "input_ids": ids,
        "rows": selected,
        "screening": dict(dropped),
    }


def phase_prepare(args):  # noqa: C901 -- explicit checkpointed source traversal and quota screens
    import pyarrow.parquet as pq
    import torch
    from huggingface_hub import hf_hub_download

    root = args.out_root
    inputs = root / "inputs"
    if not (inputs / "provenance.json").exists():
        stage("inputs", inputs)
    provenance = json.loads((inputs / "provenance.json").read_text())
    for name, expected in provenance["hashes"].items():
        assert sha_file(inputs / name) == expected, name
    config = {
        "n_train": args.n_train,
        "source_revision": WIKI_REV,
        "format": FORMAT,
        "seed": SEED,
        "inputs_sha": sha_file(inputs / "provenance.json"),
        "token_ids": list(TOKENS),
        "article_min_tokens": 512,
        "article_cap_tokens": 4096,
        "max_rows_per_article_token": 6,
        "target_tokens": [8, 256],
        "previous_tokens": [8, 96],
        "near_dupe_ngram": 5,
        "near_dupe_threshold": 0.8,
        "batch_size": 512,
        "selection_version": 1,
    }
    directory = root / "prepare"
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint = directory / "progress.json"
    state = (
        json.loads(checkpoint.read_text())
        if checkpoint.exists()
        else {
            "config": config,
            "chunks": [],
            "next_shard": 0,
            "next_batch": 0,
            "counts": {str(t): 0 for t in TOKENS},
            "screening": {},
            "articles_seen": 0,
            "processing_seconds": 0.0,
        }
    )
    assert state["config"] == config, "resume recipe mismatch"
    seen = set()
    for name in state["chunks"]:
        for article in read_jsonl(directory / name):
            seen.update(r["target_norm"] for r in article["rows"])
    drops = Counter(state["screening"])
    t0 = time.monotonic()
    initial_seconds = state["processing_seconds"]
    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=init_worker,
        initargs=(str(inputs),),
        mp_context=multiprocessing.get_context("spawn"),
    ) as pool:
        for shard in range(state["next_shard"], 41):
            path = hf_hub_download(
                "wikimedia/wikipedia",
                f"20231101.en/train-{shard:05d}-of-00041.parquet",
                repo_type="dataset",
                revision=WIKI_REV,
                cache_dir=root / "source_cache",
            )
            for batch_i, batch in enumerate(pq.ParquetFile(path).iter_batches(batch_size=512)):
                if shard == state["next_shard"] and batch_i < state["next_batch"]:
                    continue
                active = [t for t in TOKENS if state["counts"][str(t)] < args.n_train]
                if not active:
                    break
                kept = []
                for article in pool.map(
                    process_article, ((a, active) for a in batch.to_pylist()), chunksize=8
                ):
                    state["articles_seen"] += 1
                    drops.update(article.get("screening", {}))
                    if "drop" in article:
                        drops[article["drop"]] += 1
                        continue
                    selected = []
                    for row in article["rows"]:
                        tid = str(row["boundary_token_id"])
                        if state["counts"][tid] >= args.n_train:
                            continue
                        if row["target_norm"] in seen:
                            drops["train_exact_duplicate"] += 1
                            continue
                        seen.add(row["target_norm"])
                        row["selection_order"] = state["counts"][tid]
                        state["counts"][tid] += 1
                        selected.append(row)
                    if selected:
                        article["rows"] = selected
                        kept.append(article)
                names = write_jsonl_parts(directory, f"articles-s{shard:02d}-b{batch_i:04d}", kept)
                state["chunks"].extend(names)
                state.update(
                    next_shard=shard,
                    next_batch=batch_i + 1,
                    screening=dict(drops),
                    processing_seconds=initial_seconds + time.monotonic() - t0,
                )
                write_json(checkpoint, state)
                print(
                    f"[prepare] shard={shard} batch={batch_i} articles={state['articles_seen']} "
                    f"counts={state['counts']} seconds={state['processing_seconds']:.1f}",
                    flush=True,
                )
            if all(n == args.n_train for n in state["counts"].values()):
                break
            state.update(next_shard=shard + 1, next_batch=0)
            write_json(checkpoint, state)
            print(json.dumps(upload_verified(directory, "prepare")), flush=True)
    assert all(n == args.n_train for n in state["counts"].values()), state["counts"]
    manifest = directory / "manifest"
    manifest.mkdir(exist_ok=True)
    rows = []
    article_names, lengths = [], []
    for i, name in enumerate(state["chunks"]):
        docs = read_jsonl(directory / name)
        if not docs:
            continue
        for doc in docs:
            # Strictly causal forward: discard only tokens AFTER all selected targets.
            end = max(r["t_span"][1] for r in doc["rows"])
            doc["input_ids"] = doc["input_ids"][:end]
            lengths.append(end)
            rows.extend({k: v for k, v in r.items() if k != "target_norm"} for r in doc["rows"])
        name = f"articles_shard{i:04d}.pt"
        torch.save(
            {
                "window_ids": [d["article_id"] for d in docs],
                "input_ids": [torch.tensor(d["input_ids"], dtype=torch.int32) for d in docs],
            },
            manifest / name,
        )
        article_names.append(name)
    frozen = torch.load(inputs / "eval_articles.pt", map_location="cpu", weights_only=True)
    torch.save(frozen, manifest / "articles_eval.pt")
    article_names.append("articles_eval.pt")
    for path in sorted(inputs.glob("eval_rows-*.jsonl")):
        rows.extend(read_jsonl(path))
    assert len(rows) == 4 * (args.n_train + 160 + 400)
    assert len({r["row_id"] for r in rows}) == len(rows)
    row_files = write_jsonl_parts(manifest, "manifest", rows)
    import numpy as np

    meta = {
        "experiment": "issue1901_boundary25k",
        "config": config,
        "n_manifest_rows": len(rows),
        "n_articles": len(lengths) + len(frozen["window_ids"]),
        "quotas_per_token_id": {"train": args.n_train, "val": 160, "test": 400},
        "article_shards": article_names,
        "manifest_shards": row_files,
        "selected_row_ids_sha256": hashlib.sha256(
            "\n".join(r["row_id"] for r in rows).encode()
        ).hexdigest(),
        "training_article_lengths": {
            "n": len(lengths),
            "sum": sum(lengths),
            "quantiles": np.quantile(lengths, [0, 0.25, 0.5, 0.75, 0.9, 1]).tolist(),
        },
        "source_provenance": provenance,
        "screening": state["screening"],
        "processing_seconds": state["processing_seconds"],
        "deviations": [
            "Expanded 2023 Wikipedia training corpus; frozen WikiText evaluation",
            "WikiText-style Python Moses preprocessing; historical Moses version unknown",
            "Source-order document traversal with seeded anchor order and <=6 rows/article/token; "
            "parent random 48-anchor pre-subsample removed",
        ],
    }
    write_json(manifest / "meta.json", meta)
    verified = upload_verified(directory, "prepare")
    write_json(root / "prepare_verified.json", verified)
    complete(args, verified)


def complete(args, evidence):
    from explore_persona_space.backends.artifacts import write_completion_sentinel

    sentinel = os.environ.get("EPS_SENTINEL_PATH", str(args.out_root / "completion.json"))
    write_completion_sentinel(
        sentinel_path=sentinel,
        issue=1901,
        extra={"run_phase": args.phase, "upload_verification": evidence},
    )
    print("[phase=done] " + json.dumps(evidence), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["inputs", "prepare"], required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--original", type=Path)
    parser.add_argument("--wikitext-dir", type=Path)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--n-train", type=int, default=25000)
    args = parser.parse_args()
    {"inputs": phase_inputs, "prepare": phase_prepare}[args.phase](args)


if __name__ == "__main__":
    main()
