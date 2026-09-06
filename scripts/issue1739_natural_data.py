"""Natural (never crossed) #1739 generic pool: prepare, generate, capture, assemble.

Generation and HF capture are separate processes. Every 500-row unit has an
atomic, content-keyed completion record. Raw text is retained in <=8.5 MB shards.
Only the audited #779 PROMPTS are reused; answers are fresh #1092-recipe greedy.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gc
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

import numpy as np

from explore_persona_space.atomic_io import atomic_replace
from scripts import issue1092_gpu_phase as reference

SOURCE_REVISION = "7a47ff5ce42f16308bebaba29c1286a4e9bc8008"
SOURCE_PREFIX = "issue779_monitoring/fitter-fair-comparison-n1m"
SOURCE_PROMPT_SHA = "2b14762a15d316c602332a749ebd87c733d687d4165eb5d0038c298e0d27ce46"
LAYERS = [17, 18, 19, 20]
CHUNK = 500  # #1092/#779 real-corpus vLLM safe chunk.
PART_BYTES = 8_500_000
MODEL = reference.INSTRUCT_MODEL
REVISION = reference.INSTRUCT_REVISION
LOG = logging.getLogger(__name__)


def sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def file_sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with atomic_replace(path) as tmp:
        tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def read_rows(path: Path):
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def write_parts(directory: Path, rows: list[dict]) -> dict:
    """Publish immutable line shards, then their count/hash index LAST."""
    if not rows:
        raise ValueError("refusing empty row set")
    directory.mkdir(parents=True, exist_ok=True)
    parts, buffer, nbytes = [], [], 0

    def flush():
        path = directory / f"part_{len(parts):05d}.jsonl"
        with atomic_replace(path) as tmp:
            tmp.write_text("".join(buffer), encoding="utf-8")
        parts.append({"path": path.name, "rows": len(buffer), "sha256": file_sha(path)})

    for row in rows:
        line = json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
        size = len(line.encode("utf-8"))
        if size > PART_BYTES:
            raise ValueError("one raw row exceeds the HF text shard budget")
        if buffer and nbytes + size > PART_BYTES:
            flush()
            buffer, nbytes = [], 0
        buffer.append(line)
        nbytes += size
    if buffer:
        flush()
    index = {"n_rows": len(rows), "parts": parts}
    atomic_json(directory / "index.json", index)
    return index


def load_parts(directory: Path) -> list[dict]:
    index = json.loads((directory / "index.json").read_text())
    rows = []
    for part in index["parts"]:
        path = directory / part["path"]
        if file_sha(path) != part["sha256"]:
            raise ValueError(f"raw shard hash mismatch: {path}")
        chunk = list(read_rows(path))
        if len(chunk) != part["rows"]:
            raise ValueError(f"raw shard row-count mismatch: {path}")
        rows.extend(chunk)
    if len(rows) != index["n_rows"] or not rows:
        raise ValueError("raw index row-count mismatch/empty")
    return rows


def recipe() -> dict:
    return {
        "model": MODEL,
        "model_revision": REVISION,
        "temperature": 0,
        "seed": 42,
        "max_new_tokens": 1024,
        "max_model_len": 8192,
        # Reserve one token for the capture boundary, absent from generation.
        "max_prompt_tokens": 7167,
        "stop": ["<|im_end|>"],
        "enforce_eager": True,
        "enable_prefix_caching": False,
        "no_recombination": True,
        "layers": LAYERS,
    }


class IndexedNearDupeGate:
    """Exact #779 Jaccard rule, with lossless ordered-prefix candidate pruning.

    If Jaccard(A,B)>=t, their prefixes of length |A|-ceil(t|A|)+1
    and |B|-ceil(t|B|)+1 intersect under any COMMON total token order.
    The frequency ordering improves throughput; it never approximates the gate.
    """

    def __init__(self, targets, threshold=0.8):
        from scripts.issue779_ffc_n1m_generate_capture import _norm, _char_ngrams

        self.normalize = _norm
        self.grams = _char_ngrams
        self.threshold = threshold
        self.exact = {_norm(t) for t in targets}
        self.targets = [_char_ngrams(t, 5) for t in self.exact if t]
        self.frequency = Counter(g for target in self.targets for g in target)
        self.inv = defaultdict(set)
        self.n_exact_drop = self.n_near_drop = 0
        for i, target in enumerate(self.targets):
            for gram in self.prefix(target):
                self.inv[gram].add(i)

    def prefix(self, grams):
        count = len(grams) - math.ceil(self.threshold * len(grams)) + 1
        return sorted(grams, key=lambda gram: (self.frequency[gram], gram))[:count]

    def is_dupe(self, text):
        normalized = self.normalize(text)
        if normalized in self.exact:
            self.n_exact_drop += 1
            return True
        grams = self.grams(normalized, 5)
        candidates = set()
        for gram in self.prefix(grams):
            candidates.update(self.inv.get(gram, ()))
        for i in candidates:
            other = self.targets[i]
            if not self.threshold * len(grams) <= len(other) <= len(grams) / self.threshold:
                continue
            overlap = len(grams & other)
            if overlap / (len(grams) + len(other) - overlap) >= self.threshold:
                self.n_near_drop += 1
                return True
        return False

    def stats(self):
        return {
            "n_exact_drop": self.n_exact_drop,
            "n_near_drop": self.n_near_drop,
            "threshold": self.threshold,
            "ngram": 5,
            "algorithm": "exact_ordered_prefix_join",
        }


def prepare(args) -> None:
    from scripts.issue779_ffc_n1m_generate_capture import _download_manifest, _norm

    excluded = list(read_rows(args.exclusions))
    if not excluded or any(not isinstance(r.get("text"), str) for r in excluded):
        raise ValueError("complete nonempty text exclusion export is required")
    config = {
        "recipe": recipe(),
        "source_revision": SOURCE_REVISION,
        "n_candidates": args.n_candidates,
        "exclusions_sha256": file_sha(args.exclusions),
        "filter": {
            "version": 1,
            "ngram": 5,
            "jaccard": 0.8,
            "normalize": "lowercase_whitespace_collapse",
            "dedup": "normalized_exact",
        },
        "implementation_sha256": file_sha(Path(__file__)),
    }
    dest = args.root / "prepared"
    done = dest / "complete.json"
    if done.exists():
        if json.loads(done.read_text())["config"] != config:
            raise ValueError("prepared pool fingerprint mismatch; use a fresh root")
        load_parts(dest)
        return
    manifest = _download_manifest(
        SOURCE_PREFIX, args.root / "source_manifest", revision=SOURCE_REVISION
    )
    meta = json.loads((manifest / "meta.json").read_text())
    if meta["new_prompt_sha256"] != SOURCE_PROMPT_SHA:
        raise ValueError("source manifest metadata pin mismatch")
    gate = IndexedNearDupeGate([r["text"] for r in excluded])
    tokenizer = reference._get_tokenizer()
    seen, candidates = set(), []
    counts = {
        "scanned": 0,
        "lmsys": 0,
        "duplicate": 0,
        "empty": 0,
        "eval_overlap": 0,
        "overlength": 0,
    }
    # Rank ALL LMSYS rows before selecting, avoiding source stream-order bias.
    source_hash = hashlib.sha256()
    for part in sorted(manifest.glob("part_*.jsonl")):
        for row in read_rows(part):
            if row["i"] != counts["scanned"]:
                raise ValueError("source manifest global index misalignment")
            source_hash.update(row["prompt"].encode("utf-8"))
            source_hash.update(bytes([0]))
            counts["scanned"] += 1
            if row["corpus"] != "lmsys":
                continue
            counts["lmsys"] += 1
            text = row["prompt"]
            normalized = _norm(text)
            if not normalized:
                counts["empty"] += 1
                continue
            digest = sha(normalized)
            if digest in seen:
                counts["duplicate"] += 1
                continue
            seen.add(digest)
            candidates.append((sha("1739:20260906:" + digest), row))
    # SHA domain is #779 _sha_prompts: UTF-8 prompt bytes followed by NUL.
    if source_hash.hexdigest() != SOURCE_PROMPT_SHA or counts["scanned"] != meta["n_new"]:
        raise ValueError("source manifest content/count does not match pinned metadata")
    candidates.sort(key=lambda pair: pair[0])
    selected = []
    rejected = []
    checkpoint_dir = args.root / "prepare_checkpoints"
    checkpoint_path = checkpoint_dir / "state.json"
    checkpoint_parts, cursor = [], 0
    if checkpoint_path.exists():
        state = json.loads(checkpoint_path.read_text())
        if state["config"] != config:
            raise ValueError("preparation checkpoint fingerprint mismatch")
        checkpoint_parts, cursor = state["parts"], state["cursor"]
        for name in checkpoint_parts:
            selected.extend(load_parts(checkpoint_dir / name))
        counts, rejected = state["counts"], state["rejected"]
        gate.n_exact_drop = state["near_dupe"]["n_exact_drop"]
        gate.n_near_drop = state["near_dupe"]["n_near_drop"]
        LOG.info("prepare resumed accepted=%d candidate_cursor=%d", len(selected), cursor)
    for candidate_cursor in range(cursor, len(candidates)):
        if len(selected) == args.n_candidates:
            break
        _, row = candidates[candidate_cursor]
        text = row["prompt"]
        if gate.is_dupe(text):
            counts["eval_overlap"] += 1
            continue
        prefix, rendered = reference._render_prompt_parts([], text, "instruct")
        n_tokens = len(tokenizer.encode(rendered, add_special_tokens=False))
        if n_tokens > recipe()["max_prompt_tokens"]:
            counts["overlength"] += 1
            rejected.append({"source_i": row["i"], "reason": "overlength", "tokens": n_tokens})
            continue
        selected.append(
            {
                "context_id": f"natural_lmsys_{row['i']:07d}",
                "source_dataset": "lmsys-chat-1m",
                "source_id": f"manifest_i={row['i']};stream_pos={row['stream_pos']}",
                "prompt_sha256": sha(text),
                "no_recombination": True,
                "prompt": text,
                "rendered_prompt": rendered,
                "prefix_text": prefix,
                "n_prompt_tokens": n_tokens,
                "candidate_index": len(selected),
            }
        )
        if len(selected) % 1000 == 0:
            name = f"batch_{len(checkpoint_parts):05d}"
            write_parts(checkpoint_dir / name, selected[-1000:])
            checkpoint_parts.append(name)
            atomic_json(
                checkpoint_path,
                {
                    "config": config,
                    "parts": checkpoint_parts,
                    "cursor": candidate_cursor + 1,
                    "counts": counts,
                    "rejected": rejected,
                    "near_dupe": gate.stats(),
                },
            )
            LOG.info("prepare accepted=%d counts=%s", len(selected), counts)
        if len(selected) == args.n_candidates:
            break
    if len(selected) != args.n_candidates:
        raise ValueError(f"natural-pool quota unmet: {len(selected)}; {counts}")
    index = write_parts(dest, selected)
    atomic_json(
        dest / "complete.json",
        {
            "config": config,
            "counts": counts,
            "rejected": rejected,
            "near_dupe": gate.stats(),
            "index": index,
        },
    )


def unit_key(rows: list[dict], phase: str) -> str:
    return sha(
        json.dumps(
            {
                "phase": phase,
                "recipe": recipe(),
                "rows": rows,
                "implementation": file_sha(Path(__file__)),
            },
            sort_keys=True,
        )
    )


def complete_unit(directory: Path, key: str) -> bool:
    path = directory / "complete.json"
    if not path.exists():
        return False
    info = json.loads(path.read_text())
    if info["key"] != key:
        raise ValueError(f"stale unit at {directory}; refusing implicit reuse")
    for name, digest in info["files"].items():
        if file_sha(directory / name) != digest:
            raise ValueError(f"unit file absent/corrupt: {directory / name}")
    return True


def finish_unit(directory: Path, key: str, started: float, **extra) -> None:
    files = {
        str(p.relative_to(directory)): file_sha(p)
        for p in directory.rglob("*")
        if p.is_file() and p.name != "complete.json"
    }
    atomic_json(
        directory / "complete.json",
        {"key": key, "files": files, "wall_s": time.monotonic() - started, **extra},
    )


def load_generated(directory: Path, pool: list[dict]) -> list[dict]:
    chunk = int(directory.name.removeprefix("chunk_"))
    expected = pool[chunk * CHUNK : (chunk + 1) * CHUNK]
    if not expected or not complete_unit(directory, unit_key(expected, "generate")):
        raise ValueError(f"incomplete/stale generation: {directory}")
    rows = load_parts(directory)
    if len(rows) != len(expected):
        raise ValueError("generation/prepared count mismatch")
    for row, original in zip(rows, expected, strict=True):
        if any(row.get(key) != value for key, value in original.items()):
            raise ValueError("generation changed the original natural prompt")
        if row["answer_sha256"] != sha(row["answer"]) or row["admitted"] != bool(
            row["answer"].strip()
        ):
            raise ValueError("generated answer hash/admission mismatch")
    return rows


def generate(args) -> None:
    from vllm import SamplingParams
    from explore_persona_space.eval.generation import create_vllm_engine

    pool = load_parts(args.root / "prepared")
    requested = range(args.start_chunk, args.end_chunk)
    pending = []
    for chunk in requested:
        rows = pool[chunk * CHUNK : (chunk + 1) * CHUNK]
        if not rows:
            raise ValueError(f"chunk {chunk} outside prepared pool")
        dest = args.root / "generated" / f"chunk_{chunk:05d}"
        key = unit_key(rows, "generate")
        if not complete_unit(dest, key):
            pending.append((chunk, rows, dest, key))
    if not pending:
        return
    engine = create_vllm_engine(
        MODEL,
        revision=REVISION,
        max_model_len=8192,
        gpu_memory_utilization=0.85,
        max_num_seqs=64,
        hang_mitigations=True,
        seed=42,
    )
    params = SamplingParams(temperature=0.0, max_tokens=1024, stop=["<|im_end|>"], seed=42)
    for chunk, rows, dest, key in pending:
        started = time.monotonic()
        LOG.info("generate chunk=%d rows=%d", chunk, len(rows))
        outputs = engine.generate([r["rendered_prompt"] for r in rows], params, use_tqdm=False)
        if len(outputs) != len(rows):
            raise ValueError("generation request/output count mismatch")
        result = []
        for row, output in zip(rows, outputs, strict=True):
            if output.prompt != row["rendered_prompt"]:
                raise ValueError("generation output/prompt alignment mismatch")
            if len(output.outputs) != 1:
                raise ValueError("generation must return exactly one answer")
            answer = output.outputs[0]
            result.append(
                {
                    **row,
                    "answer": answer.text,
                    "answer_sha256": sha(answer.text),
                    "finish_reason": answer.finish_reason,
                    "answer_token_ids": list(answer.token_ids),
                    "admitted": bool(answer.text.strip()),
                }
            )
        write_parts(dest, result)
        finish_unit(
            dest,
            key,
            started,
            n_rows=len(rows),
            cap_hits=sum(r["finish_reason"] == "length" for r in result),
        )
        LOG.info("generated chunk=%d wall_s=%.1f", chunk, time.monotonic() - started)
    from explore_persona_space.analysis.representation_shift import _reap_vllm_engine
    import torch

    _reap_vllm_engine(engine)
    del engine
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    LOG.info("generation engine explicitly reaped")


def capture_batch(rows, model, tokenizer, layers=LAYERS, device="cuda"):
    """#1092 segment-tokenization and pooling, batched over rows and positions."""
    import torch
    from explore_persona_space.analysis.extraction import _logits_to_keep_kwargs

    encoded = [
        reference._capture_row_ids_and_positions(
            tokenizer,
            r["prefix_text"],
            r["rendered_prompt"],
            r["answer"],
            "<|im_end|>",
            r["context_id"],
        )
        for r in rows
    ]
    if tokenizer.padding_side != "right":
        raise ValueError("capture requires right padding")
    inputs = tokenizer.pad(
        {"input_ids": [x[0] for x in encoded]}, padding=True, return_tensors="pt"
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.inference_mode():
        output = model(
            **inputs,
            output_hidden_states=True,
            use_cache=False,
            **_logits_to_keep_kwargs(model, return_logits=False),
        )
    states = output.hidden_states[1:]
    if len(states) != reference.N_LAYERS:
        raise ValueError("model layer-count drift")
    positions = [x[1] for x in encoded]
    row_ids = torch.arange(len(rows), device=device)
    context_ids = torch.tensor([p["context_end"] for p in positions], device=device)
    starts = torch.tensor([p["answer_start"] for p in positions], device=device)
    ends = torch.tensor([p["answer_end"] for p in positions], device=device)
    token_ids = torch.arange(inputs["input_ids"].shape[1], device=device)
    mask = (token_ids[None, :] >= starts[:, None]) & (token_ids[None, :] < ends[:, None])
    result = {}
    for layer in layers:
        h = states[layer]
        if h.shape[-1] != reference.HIDDEN_DIM:
            raise ValueError("model hidden dimension drift")
        means = ((h.float() * mask[:, :, None]).sum(1) / (ends - starts)[:, None]).to(h.dtype)
        result[f"context_end_L{layer:02d}"] = (
            h[row_ids, context_ids].to(torch.float16).cpu().numpy()
        )
        result[f"t1_L{layer:02d}"] = means.to(torch.float16).cpu().numpy()
    return result


def capture(args) -> None:
    import torch
    from transformers import AutoModelForCausalLM

    pending = []
    pool = load_parts(args.root / "prepared")
    for chunk in range(args.start_chunk, args.end_chunk):
        source = args.root / "generated" / f"chunk_{chunk:05d}"
        rows = [r for r in load_generated(source, pool) if r["admitted"]]
        dest = args.root / "captured" / f"chunk_{chunk:05d}"
        key = unit_key(rows, "capture")
        if not complete_unit(dest, key):
            pending.append((chunk, rows, dest, key))
    if not pending:
        return
    tokenizer = reference._get_tokenizer()
    tokenizer.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        revision=REVISION,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="sdpa",
    ).eval()
    for chunk, rows, dest, key in pending:
        started = time.monotonic()
        parity = {}
        if not rows:
            raise ValueError(f"all answers empty in chunk {chunk}")
        arrays = {
            f"{kind}_L{layer:02d}": np.empty((len(rows), reference.HIDDEN_DIM), dtype=np.float16)
            for kind in ("context_end", "t1")
            for layer in LAYERS
        }
        # Length grouping reduces padding without changing persisted row order.
        order = sorted(
            range(len(rows)),
            key=lambda i: rows[i]["n_prompt_tokens"] + len(rows[i]["answer_token_ids"]),
        )
        for start in range(0, len(order), args.batch_size):
            indices = order[start : start + args.batch_size]
            result = capture_batch([rows[i] for i in indices], model, tokenizer)
            if args.verify_reference and start == 0:
                batch = [rows[i] for i in indices]
                expected = reference._capture_batch_loaded_model(
                    prefix_texts=[r["prefix_text"] for r in batch],
                    prompts=[r["rendered_prompt"] for r in batch],
                    completions=[r["answer"] for r in batch],
                    prompt_format="instruct",
                    model=model,
                    tokenizer=tokenizer,
                    n_layers=reference.N_LAYERS,
                    hidden_dim=reference.HIDDEN_DIM,
                    device="cuda",
                    log_label="natural-parity",
                    batch_size=args.batch_size,
                )
                for kind, floor in (("context_end", 0.995), ("t1", 0.9999)):
                    actual = np.stack([result[f"{kind}_L{layer:02d}"] for layer in LAYERS]).astype(
                        np.float64
                    )
                    ref = np.stack(
                        [[row[kind][layer] for row in expected.summaries] for layer in LAYERS]
                    ).astype(np.float64)
                    cosine = np.sum(actual * ref, axis=-1) / (
                        np.linalg.norm(actual, axis=-1) * np.linalg.norm(ref, axis=-1)
                    )
                    parity[kind] = {
                        "minimum_cosine": float(cosine.min()),
                        "floor": floor,
                        "max_abs": float(np.abs(actual - ref).max()),
                        "rows": len(batch),
                    }
                    if not np.isfinite(cosine).all() or cosine.min() < floor:
                        raise ValueError(f"bf16 original-capture parity failed: {parity[kind]}")
            for name, values in result.items():
                if not np.isfinite(values).all():
                    raise ValueError(f"nonfinite capture: chunk={chunk} name={name}")
                arrays[name][indices] = values
            if start % (args.batch_size * 5) == 0:
                LOG.info("capture chunk=%d rows=%d/%d", chunk, start, len(rows))
        dest.mkdir(parents=True, exist_ok=True)
        for name, values in arrays.items():
            with atomic_replace(dest / f"{name}.npy") as tmp:
                with tmp.open("wb") as stream:
                    np.save(stream, values)
        atomic_json(dest / "rows.json", {"context_ids": [r["context_id"] for r in rows]})
        if parity:
            atomic_json(dest / "parity.json", parity)
        finish_unit(
            dest, key, started, n_rows=len(rows), max_gpu_bytes=torch.cuda.max_memory_allocated()
        )
        LOG.info("captured chunk=%d wall_s=%.1f", chunk, time.monotonic() - started)


def assemble(args) -> None:
    dest = args.root / "store"
    if (dest / "manifest.json").exists():
        raise ValueError("completed store exists; validate/reuse it, never overwrite")
    accepted, chunks = [], []
    pool = load_parts(args.root / "prepared")
    for source in sorted((args.root / "generated").glob("chunk_*")):
        rows = [r for r in load_generated(source, pool) if r["admitted"]]
        captured = args.root / "captured" / source.name
        if not complete_unit(captured, unit_key(rows, "capture")):
            raise ValueError(f"capture incomplete: {captured}")
        captured_ids = json.loads((captured / "rows.json").read_text())["context_ids"]
        if captured_ids != [r["context_id"] for r in rows]:
            raise ValueError("capture/generation row-order mismatch")
        take = min(args.n_rows - len(accepted), len(rows))
        accepted.extend(rows[:take])
        chunks.append((captured, take))
        if len(accepted) == args.n_rows:
            break
    if len(accepted) != args.n_rows:
        raise ValueError(f"need reserve generation: admitted={len(accepted)} target={args.n_rows}")
    if len({r["prompt_sha256"] for r in accepted}) != args.n_rows:
        raise ValueError("duplicate natural prompts")
    dest.mkdir(parents=True, exist_ok=True)
    index_path = dest / "row_index.jsonl"
    keys = (
        "context_id",
        "source_dataset",
        "source_id",
        "prompt_sha256",
        "answer_sha256",
        "no_recombination",
    )
    with atomic_replace(index_path) as tmp:
        with tmp.open("w", encoding="utf-8") as stream:
            for row in accepted:
                stream.write(json.dumps({k: row[k] for k in keys}, sort_keys=True) + "\n")
    matrices = {}
    for kind in ("context_end", "t1"):
        for layer in LAYERS:
            name = f"{kind}_L{layer:02d}.npy"
            with atomic_replace(dest / name) as tmp:
                matrix = np.lib.format.open_memmap(
                    tmp, mode="w+", dtype=np.float16, shape=(args.n_rows, reference.HIDDEN_DIM)
                )
                start = 0
                for directory, count in chunks:
                    matrix[start : start + count] = np.load(directory / name, mmap_mode="r")[:count]
                    start += count
                matrix.flush()
                del matrix
            matrices[name] = file_sha(dest / name)
    prepared = json.loads((args.root / "prepared" / "complete.json").read_text())
    atomic_json(
        dest / "manifest.json",
        {
            "schema_version": 1,
            "status": "complete",
            "pool_kind": "natural_context_answer",
            "model": MODEL,
            "model_revision": REVISION,
            "n_rows": args.n_rows,
            "no_recombination": True,
            "layers": LAYERS,
            "hidden_dim": reference.HIDDEN_DIM,
            "dtype": "float16",
            "row_index_sha256": file_sha(index_path),
            "matrices_sha256": matrices,
            "source": {
                "repo": reference.HF_DATA_REPO,
                "revision": SOURCE_REVISION,
                "prefix": SOURCE_PREFIX + "/sampling_manifest",
                "corpus": "lmsys",
                "prepared": prepared,
            },
            "generation": {
                **recipe(),
                "cap_hit_fraction": sum(r["finish_reason"] == "length" for r in accepted)
                / len(accepted),
                "cap_policy": "inherited_1024_primary; report truncation, no silent regeneration",
            },
            "capture": {
                "context_kind": "context_end",
                "answer_kind": "t1",
                "tokenization": "issue1092_per_segment_ids",
                "answer_pooling": "mean",
            },
        },
    )


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=["prepare", "generate", "capture", "assemble"])
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--exclusions", type=Path)
    p.add_argument("--n-candidates", type=int, default=110000)
    p.add_argument("--n-rows", type=int, default=100000)
    p.add_argument("--start-chunk", type=int, default=0)
    p.add_argument("--end-chunk", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--verify-reference", action="store_true")
    args = p.parse_args(argv)
    if args.phase == "prepare" and args.exclusions is None:
        p.error("prepare requires --exclusions")
    if args.batch_size < 1 or args.end_chunk <= args.start_chunk or args.start_chunk < 0:
        p.error("invalid batch/chunk geometry")
    return args


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args(argv)
    {"prepare": prepare, "generate": generate, "capture": capture, "assemble": assemble}[
        args.phase
    ](args)


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(0)
