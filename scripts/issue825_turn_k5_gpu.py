"""Generate and capture five own answers at logged turns 1 and 12 for issue 825.

Run gen and capture in separate processes with the same --panel/--out/--model.
The only smoke narrowing is --limit-conversations; sampling always uses n=5.
Capture reproduces the parent full-render cuts, retaining each answer-token mean
separately so downstream K=5 aggregation gives equal weight to every answer.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import logging
import os
import resource
import sys
import time
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

import issue1092_gpu_phase as parent  # noqa: E402
import numpy as np  # noqa: E402

from explore_persona_space.atomic_io import atomic_replace  # noqa: E402

LOG = logging.getLogger("issue825_turn_k5_gpu")
N_DRAWS = 5
TURNS = (1, 12)
LAYER = 19
HIDDEN_DIM = 3584
WINDOW = 8192
MODEL_SPEC = {
    "instruct": (parent.INSTRUCT_MODEL, parent.INSTRUCT_REVISION, parent.STOP_TOKENS_INSTRUCT),
    "pretrained": (
        parent.PRETRAINED_MODEL,
        parent.PRETRAINED_REVISION,
        parent.STOP_TOKENS_PRETRAINED,
    ),
}
# Source: #779/#825 bf16 single-position capture gates, gotchas.md. Prefix-token
# identity is checked independently; relative/absolute deviations are reported.
CONTEXT_COS_MIN = 0.995


def utcnow() -> str:
    """Return a fresh UTC observation timestamp."""
    return datetime.now(UTC).isoformat()


def sha256(path: Path) -> str:
    """Hash a file without materializing it in memory."""
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def digest(value: Any) -> str:
    """Hash JSON values, retaining Unicode safely inside JSON string values."""
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


# Snapshot code bytes at import, not after a live file edit (previous #825 audit).
CODE_HASHES = {
    Path(__file__).name: sha256(Path(__file__)),
    "issue1092_gpu_phase.py": sha256(Path(parent.__file__)),
}


def atomic_json(path: Path, value: Any) -> None:
    """Fsync finite JSON through a process-unique same-directory atomic transaction."""
    with atomic_replace(path) as tmp, tmp.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def read_jsonl(path: Path) -> list[dict]:
    """Read physical JSONL lines; U+2028/NEL inside user text are not separators."""
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise ValueError(f"blank JSONL record at {path.name}:{line_number}")
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                raise ValueError(f"invalid JSONL at {path.name}:{line_number}") from None
    return rows


def load_panel(path: Path, limit: int = 0) -> list[dict]:
    """Validate ordered preselected real histories without changing their selection."""
    rows = read_jsonl(path)
    if limit < 0 or not rows:
        raise ValueError("panel must be nonempty and limit nonnegative")
    seen = set()
    for row in rows:
        cid = row.get("conv_id")
        if not isinstance(cid, str) or not cid or cid in seen:
            raise ValueError("panel conv_id must be nonempty, unique strings")
        seen.add(cid)
        turns = row.get("turns")
        if not isinstance(turns, list) or not turns:
            raise ValueError(f"missing turns for {cid}")
        for turn in turns:
            if turn.get("role") not in ("user", "assistant", "system") or not isinstance(
                turn.get("content"), str
            ):
                raise ValueError(f"malformed turn for {cid}")
        selected = selected_turns(turns)
        if set(selected) != set(TURNS):
            raise ValueError(f"panel lacks both target turns for {cid}")
        for index in selected.values():
            if index == 0 or turns[index - 1]["role"] != "user":
                raise ValueError(f"target answer has no preceding user for {cid}")
    return rows[:limit] if limit else rows


def selected_turns(turns: list[dict]) -> dict[int, int]:
    """Map requested one-based assistant turn numbers to source-list indices."""
    result = {}
    count = 0
    for index, turn in enumerate(turns):
        if turn["role"] == "assistant":
            count += 1
            if count in TURNS:
                result[count] = index
    return result


def load_tokenizer(model_type: str):
    """Load the pinned fast tokenizer and seed the parent's rendering cache."""
    from transformers import AutoTokenizer

    name, revision, _ = MODEL_SPEC[model_type]
    tokenizer = AutoTokenizer.from_pretrained(name, revision=revision, trust_remote_code=True)
    if not tokenizer.is_fast:
        raise ValueError("offset capture requires the pinned fast tokenizer")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    if model_type == "instruct":
        parent._get_tokenizer._tok = tokenizer
    return tokenizer


def build_jobs(panel: list[dict], model_type: str, tokenizer, max_tokens: int) -> list[dict]:
    """Render exactly the parent generation prompt and assert the capture budget."""
    jobs = []
    for row in panel:
        for turn, index in selected_turns(row["turns"]).items():
            history = row["turns"][:index]
            if model_type == "instruct":
                prompt = tokenizer.apply_chat_template(
                    history, tokenize=False, add_generation_prompt=True
                )
            else:
                prompt = parent._render_full_conversation(history, model_type) + "\n\nAssistant:"
            ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
            if len(ids) + max_tokens + 64 > WINDOW:
                raise ValueError(f"prompt budget exceeded for {row['conv_id']}/t{turn}")
            jobs.append(
                {
                    "conv_id": row["conv_id"],
                    "turn": turn,
                    "turn_index": index,
                    "prompt": prompt,
                    "prompt_token_ids": ids,
                }
            )
    return jobs


def generation_config(args, panel: list[dict], jobs: list[dict]) -> dict:
    """Record every output-affecting generation choice and immutable input/code hashes."""
    name, revision, stops = MODEL_SPEC[args.model]
    return {
        "schema": 1,
        "panel_sha256": sha256(args.panel),
        "selected_ids_sha256": digest([r["conv_id"] for r in panel]),
        "jobs_sha256": digest(jobs),
        "n_conversations": len(panel),
        "model": args.model,
        "model_id": name,
        "revision": revision,
        "tokenizer_revision": revision,
        "code_hashes": CODE_HASHES,
        "turns": list(TURNS),
        "n": N_DRAWS,
        "temperature": 1.0,
        "top_p": 0.95,
        "max_tokens": args.max_gen_tokens,
        "seed": 42,
        "stop": stops,
        "max_model_len": WINDOW,
        "capture_suffix_reserve": 64,
        "chunk_size": args.chunk_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "dtype": "bfloat16",
        "tensor_parallel_size": 1,
        "enable_prefix_caching": True,
        "backend_versions": environment()["versions"],
        "limit_conversations": args.limit_conversations,
    }


def lock_config(root: Path, config: dict) -> str:
    """Refuse stale resumption when any code/input/config key has changed."""
    root.mkdir(parents=True, exist_ok=True)
    path = root / "config.json"
    if path.exists():
        if json.loads(path.read_text()) != config:
            raise ValueError(f"immutable configuration mismatch at {path}")
    else:
        atomic_json(path, config)
    return digest(config)


def row_key(row: dict) -> tuple[str, int, int]:
    """Identify a draw without colliding with another answer to the same history."""
    return str(row["conv_id"]), int(row["turn"]), int(row["draw_id"])


def expected_keys(jobs: list[dict]) -> list[tuple[str, int, int]]:
    """Return all five draw identities for each generation prompt, in saved order."""
    return [(j["conv_id"], j["turn"], draw) for j in jobs for draw in range(N_DRAWS)]


def serialize_outputs(jobs: list[dict], outputs, model_type: str, max_tokens: int) -> list[dict]:
    """Validate actual vLLM n=5 outputs and persist every draw and its exact token IDs."""
    if len(outputs) != len(jobs):
        raise ValueError("vLLM returned the wrong number of prompts")
    rows = []
    for job, output in zip(jobs, outputs, strict=True):
        if list(output.prompt_token_ids) != job["prompt_token_ids"]:
            raise ValueError("vLLM prompt tokens differ from pinned preflight tokenization")
        answers = sorted(output.outputs, key=lambda answer: answer.index)
        if [answer.index for answer in answers] != list(range(N_DRAWS)):
            raise ValueError("vLLM n=5 draw IDs incomplete or duplicated")
        for answer in answers:
            token_ids = [int(token) for token in answer.token_ids]
            if len(token_ids) > max_tokens or not isinstance(answer.text, str):
                raise ValueError("invalid vLLM completion length/text")
            if answer.finish_reason not in ("stop", "length"):
                raise ValueError("vLLM completion has no terminal finish reason")
            rows.append(
                {
                    "conv_id": job["conv_id"],
                    "turn": job["turn"],
                    "turn_index": job["turn_index"],
                    "draw_id": int(answer.index),
                    "model_type": model_type,
                    "text": answer.text,
                    "token_ids": token_ids,
                    "n_gen_tokens": len(token_ids),
                    "prompt_token_ids": job["prompt_token_ids"],
                    "n_prompt_tokens": len(job["prompt_token_ids"]),
                    "finish_reason": answer.finish_reason,
                    "stop_reason": answer.stop_reason,
                    "at_token_cap": answer.finish_reason == "length",
                }
            )
    if [row_key(row) for row in rows] != expected_keys(jobs):
        raise AssertionError("draw order mismatch")
    return rows


def check_generation_rows(
    rows: list[dict], jobs: list[dict], model_type: str, max_tokens: int
) -> None:
    """Validate resumed raw generations independently of a bare done-file check."""
    if [row_key(row) for row in rows] != expected_keys(jobs):
        raise ValueError("generation draw identities/counts mismatch")
    by_key = {(j["conv_id"], j["turn"]): j for j in jobs}
    for row in rows:
        job = by_key[row["conv_id"], row["turn"]]
        if (
            row["model_type"] != model_type
            or row["turn_index"] != job["turn_index"]
            or row["prompt_token_ids"] != job["prompt_token_ids"]
            or row["n_prompt_tokens"] != len(job["prompt_token_ids"])
            or row["n_gen_tokens"] != len(row["token_ids"])
            or not 0 <= row["n_gen_tokens"] <= max_tokens
            or not isinstance(row["text"], str)
            or row["finish_reason"] not in ("stop", "length")
            or row["at_token_cap"] != (row["finish_reason"] == "length")
        ):
            raise ValueError("generation record validation failed")


def chunk_path(root: Path, index: int) -> Path:
    """Name one atomic checkpoint transaction directory."""
    return root / f"chunk{index:05d}"


def validate_chunk(path: Path, fingerprint: str, kind: str) -> dict | None:
    """Verify a committed chunk by config, sizes and hashes; reject partial transactions."""
    if path.with_name(path.name + ".partial").exists():
        raise ValueError(f"uncommitted chunk transaction requires recovery: {path.name}")
    if not path.exists():
        return None
    receipt = json.loads((path / "manifest.json").read_text())
    if receipt["fingerprint"] != fingerprint or receipt["kind"] != kind:
        raise ValueError(f"stale checkpoint: {path.name}")
    expected_files = {"rows.jsonl"} if kind == "gen" else {"rows.jsonl", "vectors.npz"}
    if set(receipt["files"]) != expected_files or not isinstance(receipt["n_rows"], int):
        raise ValueError(f"checkpoint schema mismatch: {path.name}")
    for name, metadata in receipt["files"].items():
        file = path / name
        if file.stat().st_size != metadata["bytes"] or sha256(file) != metadata["sha256"]:
            raise ValueError(f"checkpoint hash/size mismatch: {path.name}/{name}")
    return receipt


def write_chunk(
    path: Path,
    fingerprint: str,
    rows: list[dict],
    *,
    kind: str,
    arrays: dict[str, np.ndarray] | None = None,
    metadata: dict | None = None,
) -> dict:
    """Atomically publish fsynced raw rows, optional tensors, and a final hashed manifest."""
    if path.exists():
        raise FileExistsError(path)
    tmp = path.with_name(path.name + ".partial")
    tmp.mkdir()
    with (tmp / "rows.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    if arrays is not None:
        with (tmp / "vectors.npz").open("wb") as handle:
            np.savez(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
    receipt = {
        "kind": kind,
        "fingerprint": fingerprint,
        "n_rows": len(rows),
        "created_at": utcnow(),
        "files": {},
        "metadata": metadata or {},
    }
    for file in tmp.iterdir():
        receipt["files"][file.name] = {"bytes": file.stat().st_size, "sha256": sha256(file)}
    atomic_json(tmp / "manifest.json", receipt)
    os.replace(tmp, path)
    return receipt


def environment() -> dict:
    """Record actual installed backend versions without loading a GPU runtime."""
    versions = {}
    for name in ("torch", "transformers", "vllm", "numpy", "tokenizers"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not_installed"
    return {
        "python": sys.version,
        "versions": versions,
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }


def chunk_timing(
    *,
    elapsed_s: float,
    model_load_s: float,
    hook_validation_s: float,
    model_compute_s: float,
    resumed: bool,
    planned_draws: int,
    realized_draws: int,
) -> dict:
    """Separate current-invocation chunk work from fixed startup and resume validation."""
    if min(elapsed_s, model_load_s, hook_validation_s, model_compute_s) < 0:
        raise ValueError("negative phase timing")
    processing = elapsed_s - model_load_s - hook_validation_s
    if processing < 0 or model_compute_s > processing:
        raise ValueError("phase timing intervals overlap or exceed chunk wall time")
    if resumed and (model_load_s or hook_validation_s or model_compute_s):
        raise ValueError("resumed chunks cannot report fresh model compute")
    return {
        "elapsed_s": elapsed_s,
        "model_load_s": model_load_s,
        "hook_validation_s": hook_validation_s,
        "model_compute_s": model_compute_s,
        "processing_s": 0.0 if resumed else processing,
        "resume_validation_s": processing if resumed else 0.0,
        "resumed": resumed,
        "planned_draws": planned_draws,
        "realized_draws": realized_draws,
    }


def summarize_timings(records: list[dict], teardown_s: float) -> dict:
    """Aggregate only this invocation; reused checkpoint timings are never rescaled."""
    keys = (
        "model_load_s",
        "hook_validation_s",
        "model_compute_s",
        "processing_s",
        "resume_validation_s",
    )
    return {
        **{key: sum(record[key] for record in records) for key in keys},
        "teardown_s": teardown_s,
        "timing": {
            "scope": "current invocation only; persisted chunks counted as resume validation",
            "processing_definition": "new chunk wall time excluding model initialization and "
            "hook validation; includes token planning, compute and IO",
            "model_compute_definition": "vLLM generate calls or HF capture/reduction batches; "
            "excludes model initialization and hook validation",
            "new_chunks": sum(not r["resumed"] for r in records),
            "resumed_chunks": sum(r["resumed"] for r in records),
            "new_planned_draws": sum(r["planned_draws"] for r in records if not r["resumed"]),
            "new_realized_draws": sum(r["realized_draws"] for r in records if not r["resumed"]),
            "chunks": records,
        },
    }


def self_memory_usage() -> dict:
    """Report Linux process-lifetime peak RSS, explicitly excluding engine children."""
    return {
        "peak_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "source": "getrusage(RUSAGE_SELF).ru_maxrss",
        "scope": "this process lifetime including imports/preflight; excludes all child processes",
    }


def run_gen(args, panel: list[dict], tokenizer, jobs: list[dict], config: dict) -> None:
    """Run the exact five-answer vLLM production path, checkpointing each <=32 prompts."""
    from vllm import LLM, SamplingParams

    root = args.out / "gen" / args.model
    fingerprint = lock_config(root, config)
    start = time.monotonic()
    started_at = utcnow()
    chunks = [jobs[i : i + args.chunk_size] for i in range(0, len(jobs), args.chunk_size)]
    existing = {path.name for path in root.glob("chunk[0-9]*") if path.is_dir()}
    expected = {chunk_path(root, index).name for index in range(len(chunks))}
    if existing - expected:
        raise ValueError("unexpected generation chunk directories; explicit recovery required")
    llm = None
    totals = Counter()
    by_turn = {str(turn): Counter() for turn in TURNS}
    manifests = []
    timing_records = []
    teardown_s = 0.0
    try:
        for index, chunk in enumerate(chunks):
            unit_start = time.monotonic()
            model_load_s = model_compute_s = 0.0
            path = chunk_path(root, index)
            receipt = validate_chunk(path, fingerprint, "gen")
            resumed = receipt is not None
            if receipt is not None:
                rows = read_jsonl(path / "rows.jsonl")
                check_generation_rows(rows, chunk, args.model, args.max_gen_tokens)
            else:
                if llm is None:
                    load_start = time.monotonic()
                    llm = LLM(
                        model=config["model_id"],
                        revision=config["revision"],
                        tokenizer_revision=config["tokenizer_revision"],
                        dtype="bfloat16",
                        trust_remote_code=True,
                        seed=42,
                        gpu_memory_utilization=args.gpu_memory_utilization,
                        max_model_len=WINDOW,
                        tensor_parallel_size=1,
                        enable_prefix_caching=True,
                    )
                    model_load_s = time.monotonic() - load_start
                params = SamplingParams(
                    n=N_DRAWS,
                    temperature=1.0,
                    top_p=0.95,
                    max_tokens=args.max_gen_tokens,
                    seed=42,
                    stop=list(config["stop"]),
                )
                compute_start = time.monotonic()
                outputs = llm.generate([j["prompt"] for j in chunk], params, use_tqdm=False)
                model_compute_s = time.monotonic() - compute_start
                rows = serialize_outputs(chunk, outputs, args.model, args.max_gen_tokens)
                receipt = write_chunk(
                    path,
                    fingerprint,
                    rows,
                    kind="gen",
                    metadata={
                        "n_prompts": len(chunk),
                        "chunk_index": index,
                        "elapsed_s": time.monotonic() - unit_start,
                        "model_load_s": model_load_s,
                        "model_compute_s": model_compute_s,
                        "processing_before_checkpoint_s": time.monotonic()
                        - unit_start
                        - model_load_s,
                    },
                )
            if receipt["n_rows"] != len(rows):
                raise ValueError("generation manifest row count mismatch")
            manifests.append(
                {"path": str(path.relative_to(root)), "sha256": sha256(path / "manifest.json")}
            )
            totals.update(
                {
                    "draws": len(rows),
                    "prompts": len(chunk),
                    "generated_tokens": sum(r["n_gen_tokens"] for r in rows),
                    "cap_hits": sum(r["at_token_cap"] for r in rows),
                    "empty_answers": sum(not r["text"].strip() for r in rows),
                }
            )
            for row in rows:
                by_turn[str(row["turn"])].update(
                    {
                        "draws": 1,
                        "generated_tokens": row["n_gen_tokens"],
                        "cap_hits": int(row["at_token_cap"]),
                        "empty_answers": int(not row["text"].strip()),
                    }
                )
            LOG.info(
                "[gen] unit %d/%d model=%s draws=%d elapsed=%.2fs",
                index + 1,
                len(chunks),
                args.model,
                len(rows),
                time.monotonic() - unit_start,
            )
            timing_records.append(
                chunk_timing(
                    elapsed_s=time.monotonic() - unit_start,
                    model_load_s=model_load_s,
                    hook_validation_s=0.0,
                    model_compute_s=model_compute_s,
                    resumed=resumed,
                    planned_draws=len(chunk) * N_DRAWS,
                    realized_draws=len(rows),
                )
            )
    finally:
        if llm is not None:
            teardown_start = time.monotonic()
            from explore_persona_space.analysis.representation_shift import _reap_vllm_engine

            _reap_vllm_engine(llm)
            del llm
            gc.collect()
            teardown_s = time.monotonic() - teardown_start
    if totals["draws"] != len(panel) * len(TURNS) * N_DRAWS:
        raise AssertionError("generation coverage mismatch")
    summary = {
        "status": "complete",
        "fingerprint": fingerprint,
        "started_at": started_at,
        "completed_at": utcnow(),
        "pid": os.getpid(),
        "elapsed_s": time.monotonic() - start,
        **summarize_timings(timing_records, teardown_s),
        "self_memory": self_memory_usage(),
        "counts": dict(totals),
        "counts_by_turn": {turn: dict(counts) for turn, counts in by_turn.items()},
        "chunks": manifests,
        "environment": environment(),
        "config_sha256": sha256(root / "config.json"),
    }
    atomic_json(root / "summary.json", summary)
    LOG.info("[gen] complete model=%s counts=%s", args.model, dict(totals))


def prepare_capture_group(
    rows: list[dict], turns: list[dict], model_type: str, tokenizer
) -> tuple[list[dict], list[dict]]:
    """Plan exact original cuts and drop all five jointly on an invalid or shifted span."""
    if len(rows) != N_DRAWS or [r["draw_id"] for r in rows] != list(range(N_DRAWS)):
        raise ValueError("capture group needs five ordered draws")
    planned = []
    reason = None
    for row in rows:
        content = row["text"].strip()  # parent capture convention; original text remains in gen
        if not content:
            reason = "empty_completion"
            break
        index = row["turn_index"]
        own_turns = [*turns[:index], {"role": "assistant", "content": content}]
        render = parent._render_full_conversation(own_turns, model_type)
        ids = tokenizer(render, add_special_tokens=False)["input_ids"]
        overhead = len(ids) - row["n_prompt_tokens"] - row["n_gen_tokens"]
        if overhead > 64:
            raise ValueError("capture rendering exceeds the preflight 64-token suffix/seam reserve")
        if len(ids) > WINDOW:
            reason = "capture_window_overflow"
            break
        try:
            cuts = parent._dynamics_cut_plan(
                own_turns, tokenizer, model_type, len(ids), full_token_ids=ids
            )
        except AssertionError:
            # Parent errors can include corpus text; store only the named exclusion.
            reason = "span_assert"
            break
        context = [(s, e) for s, e, k in cuts["context_k"] if k == index]
        answer = [(s, e) for s, e, k in cuts["answer_k_t1"] if k == index]
        if len(context) != 1 or len(answer) != 1:
            raise AssertionError("cut cardinality mismatch")
        cs, ce = context[0]
        ys, ye = answer[0]
        if not (0 <= cs < ce == ys < ye <= len(ids) and ce - cs == 1):
            reason = "zero_width_or_nonadjacent_span"
            break
        prefix_hash = digest(ids[:ce])
        planned.append(
            {
                "conv_id": row["conv_id"],
                "turn": row["turn"],
                "draw_id": row["draw_id"],
                "turn_index": index,
                "input_ids": ids,
                "context_pos": cs,
                "answer_start": ys,
                "answer_end": ye,
                "context_prefix_sha256": prefix_hash,
                "n_tokens": len(ids),
                "n_answer_tokens": ye - ys,
                "render_overhead_tokens": overhead,
                "finish_reason": row["finish_reason"],
                "at_token_cap": row["at_token_cap"],
                "capture_text_sha256": hashlib.sha256(content.encode()).hexdigest(),
            }
        )
    if reason is None and len({r["context_prefix_sha256"] for r in planned}) != 1:
        reason = "context_prefix_token_mismatch"
    if reason is not None:
        return [], [
            {
                "conv_id": row["conv_id"],
                "turn": row["turn"],
                "draw_id": row["draw_id"],
                "reason": reason,
            }
            for row in rows
        ]
    return planned, []


def padded_batches(rows: list[dict], token_budget: int):
    """Length-sort and pack batches by padded tokens, never silently truncate."""
    batch = []
    max_len = 0
    for row in sorted(rows, key=lambda r: r["n_tokens"]):
        length = row["n_tokens"]
        if length > token_budget:
            raise ValueError("capture item exceeds the batch token budget")
        if batch and max(max_len, length) * (len(batch) + 1) > token_budget:
            yield batch
            batch, max_len = [], 0
        batch.append(row)
        max_len = max(max_len, length)
    if batch:
        yield batch


def reduce_hidden(hidden, rows: list[dict]):
    """Gather context states and float32 per-answer token means on the device."""
    import torch

    if hidden.ndim != 3 or hidden.shape[0] != len(rows):
        raise ValueError("hidden-state batch shape mismatch")
    positions = torch.arange(hidden.shape[1], device=hidden.device)[None, :]
    starts = torch.tensor([r["answer_start"] for r in rows], device=hidden.device)
    ends = torch.tensor([r["answer_end"] for r in rows], device=hidden.device)
    context_pos = torch.tensor([r["context_pos"] for r in rows], device=hidden.device)
    if torch.any(starts >= ends) or torch.any(ends > hidden.shape[1]):
        raise ValueError("invalid answer span during reduction")
    mask = (positions >= starts[:, None]) & (positions < ends[:, None])
    answer = torch.einsum("bth,bt->bh", hidden.float(), mask.float())
    answer /= (ends - starts)[:, None]
    context = hidden[torch.arange(len(rows), device=hidden.device), context_pos].float()
    if not torch.isfinite(context).all() or not torch.isfinite(answer).all():
        raise ValueError("nonfinite captured vectors")
    return context.cpu().numpy(), answer.cpu().numpy()


def capture_batch(model, tokenizer, rows: list[dict], device: str, *, verify: bool = False):
    """Capture only block 19; same-forward hidden-state parity gates the live hook path."""
    import torch

    if tokenizer.padding_side != "right":
        raise ValueError("capture requires right padding")
    inputs = tokenizer.pad(
        {"input_ids": [r["input_ids"] for r in rows]}, padding=True, return_tensors="pt"
    )
    captured = []

    def hook(_module, _args, output):
        """Retain only the requested layer's output for this forward."""
        captured.append(output[0] if isinstance(output, tuple) else output)

    handle = model.model.layers[LAYER].register_forward_hook(hook)
    try:
        with torch.inference_mode():
            # The decoder avoids the full vocabulary logits allocation entirely.
            outputs = model.model(
                input_ids=inputs["input_ids"].to(device),
                attention_mask=inputs["attention_mask"].to(device),
                use_cache=False,
                output_hidden_states=verify,
                return_dict=True,
            )
        if len(captured) != 1:
            raise AssertionError("layer19 hook call count mismatch")
        hidden = captured[0]
        parity = None
        if verify:
            reference = outputs.hidden_states[1:][LAYER]
            if not torch.equal(hidden, reference):
                raise AssertionError("block19 hook differs from hidden_states[1:][19]")
            parity = {
                "status": "pass",
                "layer": LAYER,
                "same_forward_equal": True,
                "max_abs": 0.0,
                "n_rows": len(rows),
                "max_tokens": max(r["n_tokens"] for r in rows),
                "at": utcnow(),
            }
        context, answer = reduce_hidden(hidden, rows)
    finally:
        handle.remove()
    return context, answer, parity


def context_parity(context: np.ndarray) -> dict:
    """Check same-prefix context vectors across five draws using the bf16 cosine gate."""
    if context.ndim != 2 or context.shape[0] != N_DRAWS:
        raise ValueError("context parity requires five vectors")
    values = context.astype(np.float64)
    norms = np.linalg.norm(values, axis=1)
    if not np.isfinite(values).all() or np.any(norms == 0):
        raise ValueError("invalid context parity inputs")
    cos = (values @ values[0]) / (norms * norms[0])
    minimum = float(np.min(cos))
    report = {
        "cos_min": minimum,
        "max_abs": float(np.abs(values - values[0]).max()),
        "max_rel_l2": float((np.linalg.norm(values - values[0], axis=1) / norms[0]).max()),
        "cos_min_bar": CONTEXT_COS_MIN,
    }
    if minimum < CONTEXT_COS_MIN:
        raise AssertionError(f"same-prefix context parity failed: cosine={minimum:.7f}")
    return report


def arrays_from_capture(rows: list[dict], contexts: np.ndarray, answers: np.ndarray) -> dict:
    """Build the unambiguous per-draw NPZ contract, persisting parent-compatible fp16."""
    if contexts.shape != answers.shape or contexts.shape != (len(rows), HIDDEN_DIM):
        raise ValueError("capture shape mismatch")
    if not np.isfinite(contexts).all() or not np.isfinite(answers).all():
        raise ValueError("nonfinite capture store")
    arrays = {
        "conv_id": np.asarray([r["conv_id"] for r in rows], dtype=str),
        "turn": np.asarray([r["turn"] for r in rows], dtype=np.int16),
        "draw_id": np.asarray([r["draw_id"] for r in rows], dtype=np.int8),
        "context": contexts.astype(np.float16),
        "answer": answers.astype(np.float16),
    }
    if not np.isfinite(arrays["context"]).all() or not np.isfinite(arrays["answer"]).all():
        raise ValueError("fp16 capture overflow")
    return arrays


def validate_capture_arrays(path: Path, rows: list[dict]) -> None:
    """Reopen NPZ with pickles disabled and verify identities/dimensions/finite values."""
    with np.load(path, allow_pickle=False) as data:
        if set(data.files) != {"conv_id", "turn", "draw_id", "context", "answer"}:
            raise ValueError("capture NPZ schema mismatch")
        keys = list(
            zip(
                data["conv_id"].tolist(),
                data["turn"].tolist(),
                data["draw_id"].tolist(),
                strict=True,
            )
        )
        if keys != [row_key(row) for row in rows] or len(set(keys)) != len(keys):
            raise ValueError("capture NPZ draw identities mismatch")
        if data["conv_id"].dtype.kind != "U":
            raise ValueError("capture IDs must be Unicode strings")
        for key in ("context", "answer"):
            if (
                data[key].shape != (len(rows), HIDDEN_DIM)
                or data[key].dtype != np.float16
                or not np.isfinite(data[key]).all()
            ):
                raise ValueError("capture NPZ vector validation failed")


def run_capture(args, panel: list[dict], tokenizer, jobs: list[dict], config: dict) -> None:  # noqa: C901
    """Capture saved own answers, with complete-generation and layer-hook rig gates."""
    import torch
    from transformers import AutoModelForCausalLM

    gen_root = args.out / "gen" / args.model
    gen_fp = lock_config(gen_root, config)
    gen_summary = json.loads((gen_root / "summary.json").read_text())
    if gen_summary["status"] != "complete" or gen_summary["fingerprint"] != gen_fp:
        raise ValueError("generation must complete with matching provenance before capture")
    if (
        gen_summary["counts"]["draws"] != len(jobs) * N_DRAWS
        or gen_summary["counts"]["prompts"] != len(jobs)
        or gen_summary["config_sha256"] != sha256(gen_root / "config.json")
    ):
        raise ValueError("generation summary coverage/config mismatch")
    cap_config = {
        "generation_fingerprint": gen_fp,
        "generation_chunks": gen_summary["chunks"],
        "layer": LAYER,
        "token_budget": args.capture_token_budget,
        "padding_side": "right",
        "capture_window": WINDOW,
        "capture_suffix_reserve": 64,
        "attention": "sdpa",
        "dtype": "bfloat16",
        "context_cos_min": CONTEXT_COS_MIN,
        "code_hashes": CODE_HASHES,
    }
    root = args.out / "capture" / args.model
    fingerprint = lock_config(root, cap_config)
    started_at, start = utcnow(), time.monotonic()
    panel_by_id = {r["conv_id"]: r["turns"] for r in panel}
    chunks = [jobs[i : i + args.chunk_size] for i in range(0, len(jobs), args.chunk_size)]
    if [item["path"] for item in gen_summary["chunks"]] != [
        chunk_path(gen_root, index).name for index in range(len(chunks))
    ]:
        raise ValueError("generation summary chunk identities/count mismatch")
    existing = {path.name for path in root.glob("chunk[0-9]*") if path.is_dir()}
    expected = {chunk_path(root, index).name for index in range(len(chunks))}
    if existing - expected:
        raise ValueError("unexpected capture chunk directories; explicit recovery required")
    model = None
    manifests, exclusions, all_keys, gate_records, render_overheads = [], [], [], [], []
    timing_records = []
    teardown_s = 0.0
    for index, chunk in enumerate(chunks):
        unit_start = time.monotonic()
        model_load_s = hook_validation_s = model_compute_s = 0.0
        source = chunk_path(gen_root, index)
        source_receipt = validate_chunk(source, gen_fp, "gen")
        if source_receipt is None:
            raise ValueError("missing generation chunk")
        if sha256(source / "manifest.json") != gen_summary["chunks"][index]["sha256"]:
            raise ValueError("generation summary manifest mismatch")
        generated = read_jsonl(source / "rows.jsonl")
        check_generation_rows(generated, chunk, args.model, args.max_gen_tokens)
        path = chunk_path(root, index)
        receipt = validate_chunk(path, fingerprint, "capture")
        resumed = receipt is not None
        if receipt is None:
            planned, dropped = [], []
            for offset in range(0, len(generated), N_DRAWS):
                group = generated[offset : offset + N_DRAWS]
                kept, missing = prepare_capture_group(
                    group, panel_by_id[group[0]["conv_id"]], args.model, tokenizer
                )
                planned.extend(kept)
                dropped.extend(missing)
            stored = {}
            gate = None
            if planned and model is None:
                name, revision, _ = MODEL_SPEC[args.model]
                load_start = time.monotonic()
                model = AutoModelForCausalLM.from_pretrained(
                    name,
                    revision=revision,
                    torch_dtype=torch.bfloat16,
                    trust_remote_code=True,
                    attn_implementation="sdpa",
                )
                model.eval().to("cuda:0")
                torch.cuda.synchronize()
                model_load_s = time.monotonic() - load_start
                if model.config.hidden_size != HIDDEN_DIM or len(model.model.layers) <= LAYER:
                    raise ValueError("pinned model architecture mismatch")
                # Smallest real generated row keeps the one-time all-layer gate bounded.
                smallest = min(planned, key=lambda r: r["n_tokens"])
                gate_start = time.monotonic()
                _, _, gate = capture_batch(model, tokenizer, [smallest], "cuda:0", verify=True)
                hook_validation_s = time.monotonic() - gate_start
            compute_start = time.monotonic()
            for batch in padded_batches(planned, args.capture_token_budget):
                context, answer, _ = capture_batch(model, tokenizer, batch, "cuda:0")
                for row, x, y in zip(batch, context, answer, strict=True):
                    stored[row_key(row)] = (x, y)
            model_compute_s = time.monotonic() - compute_start if planned else 0.0
            parities = []
            for offset in range(0, len(planned), N_DRAWS):
                group = planned[offset : offset + N_DRAWS]
                parities.append(
                    {
                        "conv_id": group[0]["conv_id"],
                        "turn": group[0]["turn"],
                        **context_parity(np.stack([stored[row_key(r)][0] for r in group])),
                    }
                )
            contexts = (
                np.stack([stored[row_key(r)][0] for r in planned])
                if planned
                else np.empty((0, HIDDEN_DIM), dtype=np.float32)
            )
            answers = (
                np.stack([stored[row_key(r)][1] for r in planned])
                if planned
                else np.empty((0, HIDDEN_DIM), dtype=np.float32)
            )
            arrays = arrays_from_capture(planned, contexts, answers)
            rows = [{k: v for k, v in r.items() if k != "input_ids"} for r in planned]
            receipt = write_chunk(
                path,
                fingerprint,
                rows,
                kind="capture",
                arrays=arrays,
                metadata={
                    "exclusions": dropped,
                    "context_parity": parities,
                    "hook_gate": gate,
                    "n_planned_draws": len(generated),
                    "generation_manifest_sha256": sha256(source / "manifest.json"),
                    "elapsed_s": time.monotonic() - unit_start,
                    "model_load_s": model_load_s,
                    "hook_validation_s": hook_validation_s,
                    "model_compute_s": model_compute_s,
                    "processing_before_checkpoint_s": time.monotonic()
                    - unit_start
                    - model_load_s
                    - hook_validation_s,
                },
            )
        else:
            rows = read_jsonl(path / "rows.jsonl")
        validate_capture_arrays(path / "vectors.npz", rows)
        missing = receipt["metadata"]["exclusions"]
        realized = [row_key(r) for r in rows] + [row_key(r) for r in missing]
        if sorted(realized) != sorted(expected_keys(chunk)) or receipt["n_rows"] != len(rows):
            raise ValueError("capture planned-versus-realized coverage mismatch")
        if receipt["metadata"]["generation_manifest_sha256"] != sha256(source / "manifest.json"):
            raise ValueError("capture source-generation manifest mismatch")
        all_keys.extend(row_key(r) for r in rows)
        render_overheads.extend(row["render_overhead_tokens"] for row in rows)
        exclusions.extend(missing)
        if receipt["metadata"]["hook_gate"] is not None:
            gate_records.append(receipt["metadata"]["hook_gate"])
        manifests.append(
            {"path": str(path.relative_to(root)), "sha256": sha256(path / "manifest.json")}
        )
        LOG.info(
            "[capture] unit %d/%d model=%s draws=%d exclusions=%d elapsed=%.2fs",
            index + 1,
            len(chunks),
            args.model,
            len(rows),
            len(missing),
            time.monotonic() - unit_start,
        )
        timing_records.append(
            chunk_timing(
                elapsed_s=time.monotonic() - unit_start,
                model_load_s=model_load_s,
                hook_validation_s=hook_validation_s,
                model_compute_s=model_compute_s,
                resumed=resumed,
                planned_draws=len(generated),
                realized_draws=len(rows),
            )
        )
    gpu_memory = {
        "measured": model is not None,
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated("cuda:0"))
        if model is not None
        else None,
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved("cuda:0"))
        if model is not None
        else None,
        "source": "torch.cuda process-lifetime allocator peaks before cleanup",
        "scope": "capture process CUDA allocator only; excludes non-PyTorch allocations",
    }
    if model is not None:
        teardown_start = time.monotonic()
        del model
        gc.collect()
        torch.cuda.empty_cache()
        teardown_s = time.monotonic() - teardown_start
    if not all_keys or not gate_records:
        raise ValueError("capture has no valid rows or missing actual-model layer gate")
    groups = defaultdict(set)
    for cid, turn, draw in all_keys:
        groups[cid, turn].add(draw)
    if len(all_keys) != len(set(all_keys)) or any(
        v != set(range(N_DRAWS)) for v in groups.values()
    ):
        raise ValueError("capture has incomplete/duplicate five-draw groups")
    complete = sorted({cid for cid, _ in groups if all((cid, t) in groups for t in TURNS)})
    summary = {
        "status": "complete",
        "fingerprint": fingerprint,
        "started_at": started_at,
        "completed_at": utcnow(),
        "pid": os.getpid(),
        "elapsed_s": time.monotonic() - start,
        **summarize_timings(timing_records, teardown_s),
        "self_memory": self_memory_usage(),
        "gpu_memory": gpu_memory,
        "n_selected_conversations": len(panel),
        "n_expected_draws": len(jobs) * N_DRAWS,
        "n_captured_draws": len(all_keys),
        "n_complete_conversations": len(complete),
        "complete_conversation_ids": complete,
        "n_excluded_draws": len(exclusions),
        "exclusion_counts": dict(Counter(r["reason"] for r in exclusions)),
        "exclusions": exclusions,
        "chunks": manifests,
        "hook_gates": gate_records,
        "render_overhead": {
            "population": "valid captured draws",
            "n": len(render_overheads),
            "reserve_tokens": 64,
            "max_tokens": max(render_overheads),
            "p95_tokens": float(np.percentile(render_overheads, 95)),
            "within_reserve": max(render_overheads) <= 64,
        },
        "generation_summary_sha256": sha256(gen_root / "summary.json"),
        "generation_fingerprint": gen_fp,
        "generation_config": config,
        "config_sha256": sha256(root / "config.json"),
        "environment": environment(),
    }
    atomic_json(root / "summary.json", summary)
    LOG.info(
        "[capture] complete model=%s valid_draws=%d complete_conversations=%d",
        args.model,
        len(all_keys),
        len(complete),
    )


def parse_args(argv: list[str] | None = None):
    """Parse the bounded pilot CLI; all smoke runs retain actual n=5 generation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, choices=("gen", "capture"))
    parser.add_argument("--model", required=True, choices=tuple(MODEL_SPEC))
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit-conversations", type=int, default=0)
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--capture-token-budget", type=int, choices=(8192, 16384), default=16384)
    parser.add_argument(
        "--max-gen-tokens",
        type=int,
        choices=(1024,),
        default=1024,
        help="Frozen parent capped-generation regime; no automatic cap extension",
    )
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    args = parser.parse_args(argv)
    if not 1 <= args.chunk_size <= 32 or args.limit_conversations < 0:
        parser.error("chunk size must be 1..32 and conversation limit nonnegative")
    if not 0 < args.gpu_memory_utilization < 1:
        parser.error("GPU utilization must lie in (0,1)")
    return args


def main() -> None:
    """Validate panel/config and dispatch exactly one isolated GPU phase."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    panel = load_panel(args.panel, args.limit_conversations)
    tokenizer = load_tokenizer(args.model)
    jobs = build_jobs(panel, args.model, tokenizer, args.max_gen_tokens)
    config = generation_config(args, panel, jobs)
    if args.phase == "gen":
        run_gen(args, panel, tokenizer, jobs, config)
    else:
        run_capture(args, panel, tokenizer, jobs, config)
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(0)


if __name__ == "__main__":
    main()
