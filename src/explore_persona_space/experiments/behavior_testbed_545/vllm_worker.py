"""Issue #545 metric-race — standalone vLLM sampling worker (Strategy E).

After SIX OOMs (#545 rounds 1/3/4/6/8 + the round-37 max_seq_len halt) of
trying to co-resident the HF base model and the vLLM engine on ONE H100 inside
``predictors_zoo.extract_clouds_and_outdist_gpu`` — tuning
``gpu_memory_utilization`` (r1/r3/r4/r6/r8), the HF teacher-force sub-batch
(r4), per-layer forward hooks (r9), and finally a probe-length-cap reduction
(r10, HALTED because the JS probes are genuinely up to 5631 tokens) — the only
architecturally correct fix is to SEQUENCE the two models into phases so they
NEVER co-reside: vLLM samples on-policy responses in a SUBPROCESS (which then
exits, fully releasing the GPU), and only afterwards does the HF base model
load in the main process to teacher-force / extract hidden states from the
cached responses.

This module is the vLLM half. It runs as a standalone subprocess
(``python -m explore_persona_space.experiments.behavior_testbed_545.vllm_worker
--ipc-dir <dir>``), loads the base vLLM engine with the FULL GPU to itself
(``gpu_memory_utilization=0.85``, ``max_model_len`` raised to 8192 to clear the
5631-token + 1024-max-new worst case the round-37 audit measured), and serves a
file-based request/response protocol:

  IPC contract (sentinel-file; resilient to crashes — written work survives a
  mid-phase failure and the main process re-reads it):

  - ``<ipc-dir>/requests/<probe_id>.json``   (written by the main process)
        {"probe_id": str, "prompt_token_ids": list[list[int]], "n": int,
         "max_tokens": int, "temperature": float, "top_p": float, "seed": int}
  - ``<ipc-dir>/READY``    (sentinel — the main process has written ALL requests)
  - ``<ipc-dir>/responses/<probe_id>.json``  (written by THIS worker, atomically)
        {"probe_id": str,
         "completions": list[                      # one entry PER prompt, in order
             list[{"token_ids": list[int], "text": str, "finish_reason": str}]
         ]}
  - ``<ipc-dir>/STOP``     (sentinel — the main process tells the worker to exit)
  - ``<ipc-dir>/worker.error``  (written by the worker on a fatal error; the main
        process polls for it so a dead worker fails the run LOUD, not on a hang)

The worker processes any pending request (a request file with no matching
response file), then — once ``READY`` exists and no pending requests remain —
exits cleanly. The main process additionally drops ``STOP`` to force a prompt
exit. Responses are token-id lists + decoded text, never tensors, so the JSON
payload is small (the divergence/log-prob math stays in the HF half).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

logger = logging.getLogger("i545_vllm_worker")

# vLLM defaults for the worker (it owns the GPU alone — no co-residency budget).
# 0.85 is comfortable for a 7B engine on an 80 GiB H100 with NOTHING else
# resident (the whole point of Strategy E). Raised max_model_len to 8192 to fix
# the round-10 latent silent-truncation bug: the JS scoring path constructs
# prompts up to 5631 tokens (round-37 measurement) and samples up to 1024 new
# tokens, so the worst-case sequence is ~6655 tokens; 8192 clears it with margin.
WORKER_GPU_MEM_UTIL = 0.85
WORKER_MAX_MODEL_LEN = 8192

# How long to keep polling for new request files after READY before declaring
# the queue drained (READY guarantees no MORE requests are coming, so this is a
# tiny grace for filesystem visibility, not a real wait).
_POST_READY_GRACE_S = 2.0
# Idle poll interval while waiting for the main process to write requests/READY.
_POLL_INTERVAL_S = 0.25


def _atomic_write_json(path: Path, payload: dict) -> None:
    """Write JSON to a temp file then rename — so the main process never reads a
    half-written response (rename is atomic on POSIX)."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload))
    os.replace(tmp, path)


def _pending_request_ids(req_dir: Path, resp_dir: Path) -> list[str]:
    """Request ids that have a request file but no response file yet."""
    pending = []
    for req in sorted(req_dir.glob("*.json")):
        probe_id = req.stem
        if not (resp_dir / f"{probe_id}.json").exists():
            pending.append(probe_id)
    return pending


def _serve(ipc_dir: Path) -> int:
    """Main worker loop. Returns a process exit code (0 = clean)."""
    req_dir = ipc_dir / "requests"
    resp_dir = ipc_dir / "responses"
    resp_dir.mkdir(parents=True, exist_ok=True)
    ready_path = ipc_dir / "READY"
    stop_path = ipc_dir / "STOP"

    # Load the engine ONCE (owns the GPU alone). Imported here, not at module
    # top, so `--help` / unit tests that import the module never trigger a CUDA
    # init or a heavy vLLM import.
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    logger.info(
        "[phase=vllm-worker] loading vLLM engine (gpu_memory_utilization=%.2f, "
        "max_model_len=%d) — sole GPU resident",
        WORKER_GPU_MEM_UTIL,
        WORKER_MAX_MODEL_LEN,
    )
    llm = LLM(
        model=_base_model(),
        dtype="bfloat16",
        max_model_len=WORKER_MAX_MODEL_LEN,
        gpu_memory_utilization=WORKER_GPU_MEM_UTIL,
    )
    logger.info("[phase=vllm-worker] engine ready; serving IPC at %s", ipc_dir)

    ready_seen_at: float | None = None
    while True:
        if stop_path.exists():
            logger.info("[phase=vllm-worker] STOP sentinel seen — exiting")
            return 0

        pending = _pending_request_ids(req_dir, resp_dir)
        if pending:
            for probe_id in pending:
                req = json.loads((req_dir / f"{probe_id}.json").read_text())
                prompts = [TokensPrompt(prompt_token_ids=ids) for ids in req["prompt_token_ids"]]
                sp = SamplingParams(
                    n=int(req["n"]),
                    temperature=float(req.get("temperature", 1.0)),
                    top_p=float(req.get("top_p", 1.0)),
                    max_tokens=int(req["max_tokens"]),
                    seed=int(req.get("seed", 545)),
                )
                outs = llm.generate(prompts, sp)
                completions = [
                    [
                        {
                            "token_ids": list(comp.token_ids),
                            "text": comp.text,
                            "finish_reason": comp.finish_reason,
                        }
                        for comp in out.outputs
                    ]
                    for out in outs
                ]
                _atomic_write_json(
                    resp_dir / f"{probe_id}.json",
                    {"probe_id": probe_id, "completions": completions},
                )
                logger.info("[phase=vllm-worker] served %s (%d prompts)", probe_id, len(prompts))
            continue  # re-scan immediately for more pending requests

        # No pending requests. If READY has been seen and the grace window has
        # elapsed with no new requests, the queue is drained — exit clean.
        if ready_path.exists():
            now = time.monotonic()
            if ready_seen_at is None:
                ready_seen_at = now
            elif now - ready_seen_at >= _POST_READY_GRACE_S:
                logger.info("[phase=vllm-worker] queue drained after READY — exiting clean")
                return 0
        time.sleep(_POLL_INTERVAL_S)

    # Unreachable — the error path below handles fatal failures via try/except in main.
    return 0  # pragma: no cover


def _base_model() -> str:
    # Late import so `--help` does not pull the package tree (and its heavy deps).
    from explore_persona_space.experiments.behavior_testbed_545 import BASE_MODEL

    return BASE_MODEL


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="issue #545 metric-race vLLM sampling worker")
    ap.add_argument("--ipc-dir", required=True, help="IPC directory (requests/ + responses/)")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    ipc_dir = Path(args.ipc_dir)
    ipc_dir.mkdir(parents=True, exist_ok=True)
    error_path = ipc_dir / "worker.error"
    try:
        return _serve(ipc_dir)
    except BaseException as exc:
        # Fail LOUD: write a worker.error sentinel so the polling main process
        # detects the dead worker and raises, instead of hanging forever waiting
        # for response files that will never arrive (CLAUDE.md fail-fast).
        import traceback

        error_path.write_text(json.dumps({"error": repr(exc), "traceback": traceback.format_exc()}))
        logger.exception("[phase=vllm-worker] FATAL — wrote worker.error")
        return 2


if __name__ == "__main__":
    sys.exit(main())
