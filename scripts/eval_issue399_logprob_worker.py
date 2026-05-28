#!/usr/bin/env python3
"""Subprocess worker for issue #399's teacher-forced log-prob block.

Why this is a separate process (round-12 fix, 2026-05-26):

The parent eval (``scripts/eval_issue399.py``) instantiates a vLLM
``LLM`` engine for the behavioral phase. On vllm 0.11 + TP=1 the engine
spawns a child worker process (visible in ``nvidia-smi`` as a separate
PID) that pins ~74 GB of HBM. Even an in-process ``destroy_model_parallel()
+ destroy_distributed_environment() + del llm + torch.cuda.empty_cache()``
DOES free memory in the parent (round-11 diagnostic on issue #399
confirmed: "84.5 GB free of 85.0 GB total" post-teardown), but the
orphan vLLM worker subprocess can re-allocate after the parent finishes
teardown — and the next ``AutoModelForCausalLM.from_pretrained(...)``
in the same parent process hits CUDA OOM.

A child Python process inherits NONE of the parent's CUDA context or
Python-level vLLM allocations. The orphan vLLM worker is unrelated to
the child by parent-child relationship and can't follow it. The child
sees the GPU as the OS sees it; once the parent's vLLM worker dies
(which happens cleanly when the parent's ``del llm`` runs to completion,
or when the parent process itself exits), the child gets a clean GPU.
Even with a lingering orphan, the child's ``from_pretrained`` raises
loudly with a real OOM traceback — far better than the silent corruption
mode the in-process teardown produces.

Per-cell incremental save: this worker writes ONE JSON file per
(model_mode, cell) so a mid-worker crash loses only one cell, not the
whole worker. The parent's resume-from-disk logic skips cells whose
output JSON already exists.

Payload schema (read from ``--payload-file`` JSON):
    {
      "model_id": str,                # HF hub id OR local path
      "marker_text": str,             # the BPE-marker token to score
      "batch_size": int,              # passed to compute_marker_logprob
      "contexts_per_cell": {cell: [ctx_string, ...]},
      "output_dir": str,              # absolute path
      "mode": "trained" | "floor",    # filename prefix tag
      "position": "first_token" | "oncontent",  # round-16: probe position
                                                 # tag (optional; defaults to
                                                 # "first_token" for round-15
                                                 # back-compat — old payloads
                                                 # missing the field still
                                                 # write to the legacy
                                                 # filename).
    }

Output:
    Position "first_token" (legacy / round-15 filenames):
        ``<output_dir>/logprob_{mode}_{cell}.json``
    Position "oncontent" (round-16):
        ``<output_dir>/logprob_{mode}_oncontent_{cell}.json``

Per-cell payload:
    {
      "cell": str,
      "mode": "trained" | "floor",
      "position": "first_token" | "oncontent",
      "model_id": str,
      "marker_text": str,
      "n_contexts": int,
      "logp": [float, ...],
      "git_commit": str | null,
      "timestamp": str,
    }

Exit codes:
    0 = all requested cells written (or skipped because already present).
    1 = unhandled exception (full traceback to stderr; PARENT MUST re-raise).
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import subprocess
import sys
import traceback
from pathlib import Path


def _git_commit() -> str | None:
    """Best-effort git HEAD SHA (full 40-char) for the reproducibility tag."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(Path(__file__).resolve().parent.parent),
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _per_cell_output_path(
    output_dir: Path, mode: str, cell: str, position: str = "first_token"
) -> Path:
    """Filesystem-safe filename for a cell's worker output.

    Round-16: ``position`` selects the on-disk layout:

    - ``"first_token"`` (default, back-compat with round-15 files already
      on disk and HF): ``logprob_{mode}_{cell}.json``. Probes
      p(※ | chat_template_prefix) at the assistant's first emitted
      position.
    - ``"oncontent"``: ``logprob_{mode}_oncontent_{cell}.json``. Probes
      p(※ | chat_template_prefix + on_policy_completion + "\\n\\n") at
      the end-of-content position the trainer actually installed ※
      against.

    Kept in lockstep with the parent's :func:`_logprob_per_cell_path` in
    :mod:`scripts.eval_issue399`. Update both if the naming scheme
    changes.
    """
    import re

    safe = re.sub(r"[^A-Za-z0-9_-]", "_", cell)
    if position == "first_token":
        return output_dir / f"logprob_{mode}_{safe}.json"
    if position == "oncontent":
        return output_dir / f"logprob_{mode}_oncontent_{safe}.json"
    raise ValueError(f"Unknown position {position!r} (expected 'first_token' or 'oncontent')")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--payload-file",
        required=True,
        type=Path,
        help=(
            "JSON file with model_id, contexts_per_cell, marker_text, batch_size, output_dir, mode"
        ),
    )
    args = parser.parse_args()

    payload = json.loads(args.payload_file.read_text())
    model_id: str = payload["model_id"]
    marker_text: str = payload["marker_text"]
    batch_size: int = int(payload["batch_size"])
    contexts_per_cell: dict[str, list[str]] = payload["contexts_per_cell"]
    output_dir: Path = Path(payload["output_dir"])
    mode: str = payload["mode"]
    # Round-16: ``position`` selects the probe layout (first_token vs
    # oncontent). Missing field → ``"first_token"`` for back-compat with
    # round-15 payloads. ``contexts_per_cell`` is already the position-
    # specific prefix the parent built (parent appends the on-policy
    # completion + "\n\n" for the oncontent case; the worker scores
    # whatever prefix it receives — no per-position branching here).
    position: str = payload.get("position", "first_token")
    assert mode in ("trained", "floor"), f"Unknown mode {mode!r} (expected 'trained' or 'floor')"
    assert position in ("first_token", "oncontent"), (
        f"Unknown position {position!r} (expected 'first_token' or 'oncontent')"
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    # Resume-from-disk: skip any cell whose output JSON already exists.
    # The parent should normally pre-filter, but we duplicate the check
    # here so a manual subprocess invocation is idempotent.
    cells_to_run: dict[str, list[str]] = {}
    for cell, contexts in contexts_per_cell.items():
        out_path = _per_cell_output_path(output_dir, mode, cell, position)
        if out_path.exists():
            print(
                f"  [worker mode={mode} position={position}] cell {cell}: "
                f"existing → {out_path} (skip)",
                flush=True,
            )
            continue
        cells_to_run[cell] = contexts

    if not cells_to_run:
        print(
            f"  [worker mode={mode} position={position}] all cells already on disk; nothing to do",
            flush=True,
        )
        return 0

    print(
        f"  [worker mode={mode} position={position}] loading {model_id} "
        f"(will compute {len(cells_to_run)} cells)...",
        flush=True,
    )

    # Imports deferred so a payload-validation crash doesn't pay the HF
    # import cost. ``torch`` and ``transformers`` are pod-only deps; the
    # parent test path on the dev VM never reaches this branch.
    # ``gc`` is imported here (not at module top) so ruff doesn't auto-strip
    # it on the dev VM where it appears unused at parse time.
    import gc

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from explore_persona_space.eval.marker_logprob import compute_marker_logprob

    hf_token = os.environ.get("HF_TOKEN")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="cuda:0",
        token=hf_token,
    )
    model.eval()
    try:
        free_b, total_b = torch.cuda.mem_get_info()
        print(
            f"  [worker mode={mode} position={position}] loaded; "
            f"{torch.cuda.memory_allocated() / 1e9:.1f} GB allocated, "
            f"{free_b / 1e9:.1f} GB free of {total_b / 1e9:.1f} GB",
            flush=True,
        )
    except RuntimeError:
        # mem_get_info() requires an initialized CUDA context; if the worker
        # is being smoke-tested under a CUDA-less env (e.g. dev VM dry-run),
        # don't crash — the load above would have failed first anyway.
        pass

    commit = _git_commit()
    timestamp = datetime.datetime.now(datetime.UTC).isoformat()

    for cell, contexts in cells_to_run.items():
        if not contexts:
            # Empty cell — write a sentinel file so the parent's
            # resume-from-disk logic sees the cell as "done" rather than
            # spawning a third worker for it.
            payload_out: dict = {
                "cell": cell,
                "mode": mode,
                "position": position,
                "model_id": model_id,
                "marker_text": marker_text,
                "n_contexts": 0,
                "logp": [],
                "git_commit": commit,
                "timestamp": timestamp,
            }
            _per_cell_output_path(output_dir, mode, cell, position).write_text(
                json.dumps(payload_out, indent=2)
            )
            print(
                f"  [worker mode={mode} position={position}] cell {cell}: "
                f"0 contexts → wrote empty sentinel",
                flush=True,
            )
            continue

        free_gb = torch.cuda.mem_get_info()[0] / 1e9
        print(
            f"  [worker mode={mode} position={position}] cell {cell}: "
            f"scoring {len(contexts)} contexts (marker={marker_text!r}, "
            f"batch={batch_size}, GPU free: {free_gb:.1f} GB)...",
            flush=True,
        )
        lps = compute_marker_logprob(
            model,
            tokenizer,
            contexts=contexts,
            marker_text=marker_text,
            position="end_of_answer",
            batch_size=batch_size,
            device="cuda:0",
        )
        if len(lps) != len(contexts):
            raise RuntimeError(
                f"compute_marker_logprob returned {len(lps)} values for {len(contexts)} contexts"
            )
        for v in lps:
            if not math.isfinite(v):
                raise RuntimeError(
                    f"Non-finite log-prob ({v}) in cell {cell} (mode={mode}, "
                    f"position={position}); tokenization or chat-template bug "
                    f"— halting per CLAUDE.md fail-fast rule."
                )

        payload_out = {
            "cell": cell,
            "mode": mode,
            "position": position,
            "model_id": model_id,
            "marker_text": marker_text,
            "n_contexts": len(contexts),
            "logp": list(lps),
            "git_commit": commit,
            "timestamp": timestamp,
        }
        out_path = _per_cell_output_path(output_dir, mode, cell, position)
        # Atomic-ish write: write to .tmp then rename, so a SIGKILL in the
        # middle of json.dump never leaves a half-written file that the
        # parent reads as "done" on resume.
        tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
        tmp_path.write_text(json.dumps(payload_out, indent=2))
        tmp_path.rename(out_path)
        print(
            f"  [worker mode={mode} position={position}] cell {cell}: "
            f"wrote {out_path} ({len(lps)} values)",
            flush=True,
        )

        # Round-13 fix (2026-05-26): release per-cell activation tensors /
        # KV cache before the next cell. Without this, allocated activations
        # from prior cells accumulate unbounded across compute_marker_logprob
        # calls — round-12 worker OOMed at cell ~7 of 14 with 74 GiB pinned.
        gc.collect()
        torch.cuda.empty_cache()

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(1)
