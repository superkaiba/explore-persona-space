"""Subprocess isolation for framework-switching phases (train -> eval fork).

Motivation
----------
vLLM does NOT reap its worker subprocesses on in-process teardown. When the SAME
Python process loads vLLM and then loads a non-vLLM framework (HF Transformers
``AutoModelForCausalLM.from_pretrained``, sentence-transformers, ...), the canonical
``destroy_model_parallel() + destroy_distributed_environment() + gc.collect() +
torch.cuda.empty_cache()`` sequence is NOT sufficient: orphan worker PIDs survive and
re-grab the freed GPU memory the moment the next framework loads weights, producing a
CUDA OOM that masquerades as an HF-Transformers bug (CLAUDE.md Gotchas; task #399
round-11, orphan PID 2227527 re-allocated 74 GB after a clean destroy_* sequence).

The robust fix is process isolation: run each framework phase in a fresh child
process so the OS reaps the child's entire process tree on exit, returning the GPU to
a clean state before the next phase. ``run_isolated`` is that primitive.

Contract
--------
``run_isolated(target_module, payload)`` spawns::

    uv run python -m <target_module> <input_json_path> <output_json_path>

writing ``payload`` (a JSON-serializable dict) to ``input_json_path`` first and reading
the result dict back from ``output_json_path`` after the child exits cleanly. The child
is responsible for:

  1. reading ``sys.argv[1]`` (input path) and parsing the payload JSON,
  2. doing its framework-specific work,
  3. writing a JSON-serializable result dict to ``sys.argv[2]`` (output path).

Fail-loud semantics (per CLAUDE.md "the crash IS the signal"):

  * Non-zero child exit -> ``SubprocessIsolationError`` carrying rc + captured
    stderr tail. No dummy result, no swallowed failure.
  * Missing / unparseable output file after a zero-exit child ->
    ``SubprocessIsolationError`` (the child claimed success but produced no result).

Child-process reaping: after the direct child exits, any surviving descendants (vLLM
workers that outlived the parent) are terminated then killed via ``psutil`` when it is
importable. ``psutil`` is a transitive dependency on pods; when it is absent the helper
still works (the direct child is reaped by ``Popen.wait()``), it just cannot chase
grandchild orphans — so the optional import degrades gracefully rather than hard-failing.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# How many bytes of the child's stderr to surface in an error message.
_STDERR_TAIL_BYTES = 4000


class SubprocessIsolationError(RuntimeError):
    """Raised when an isolated child phase fails or produces no result.

    Carries the failing module, return code, and a tail of the child's stderr so
    the failure is debuggable without re-running. Never swallowed — the crash IS the
    signal.
    """


def _reap_descendants(pid: int) -> None:
    """Terminate then kill any surviving descendants of ``pid``.

    vLLM worker subprocesses can outlive their parent. After the direct child exits we
    chase its (now re-parented) descendants and reap them so they cannot hold the GPU
    for the next phase. Best-effort: no-op when ``psutil`` is unavailable or the
    process tree has already exited.
    """
    try:
        import psutil
    except ImportError:
        logger.debug("psutil not importable; skipping descendant reaping for pid %s.", pid)
        return

    try:
        parent = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return

    try:
        children = parent.children(recursive=True)
    except psutil.NoSuchProcess:
        return

    for child in children:
        try:
            child.terminate()
        except psutil.NoSuchProcess:
            continue

    _gone, alive = psutil.wait_procs(children, timeout=5)
    for child in alive:
        try:
            child.kill()
        except psutil.NoSuchProcess:
            continue
    if alive:
        logger.warning(
            "Force-killed %d orphan descendant(s) of isolated child %s.", len(alive), pid
        )


def run_isolated(
    target_module: str,
    payload: dict[str, Any],
    *,
    extra_env: dict[str, str] | None = None,
    cwd: str | None = None,
    timeout: float | None = None,
) -> dict[str, Any]:
    """Run ``target_module`` in a fresh ``uv run python -m`` child, with JSON IPC.

    The child receives the input JSON path as ``sys.argv[1]`` and must write its result
    dict to the output JSON path passed as ``sys.argv[2]``.

    Args:
        target_module: Importable module path runnable via ``python -m`` (e.g.
            ``"explore_persona_space.orchestrate.some_eval_phase"``). The module must
            read ``sys.argv[1]`` (payload path) and write its result to ``sys.argv[2]``.
        payload: JSON-serializable dict handed to the child.
        extra_env: Optional env overrides merged onto a copy of ``os.environ`` for the
            child. The full parent env is forwarded explicitly (the credential contract
            is explicit, not inherited implicitly).
        cwd: Working directory for the child (defaults to the parent's cwd).
        timeout: Optional wall-clock seconds; ``subprocess.TimeoutExpired`` propagates
            on overrun after the child is killed.

    Returns:
        The result dict the child wrote to the output JSON path.

    Raises:
        SubprocessIsolationError: Non-zero child exit, or zero-exit child that left no
            parseable result file.
        TypeError: ``payload`` is not a dict.
    """
    if not isinstance(payload, dict):
        raise TypeError(f"payload must be a dict, got {type(payload).__name__}")

    env = {**os.environ}
    if extra_env:
        env.update(extra_env)

    tmp_dir = tempfile.mkdtemp(prefix="epm_isolated_")
    in_path = Path(tmp_dir) / "payload.json"
    out_path = Path(tmp_dir) / "result.json"
    in_path.write_text(json.dumps(payload, default=str))

    cmd = ["uv", "run", "python", "-m", target_module, str(in_path), str(out_path)]
    logger.info("Running isolated phase: %s (module=%s)", " ".join(cmd), target_module)

    proc = subprocess.Popen(
        cmd,
        env=env,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    pid = proc.pid
    try:
        _stdout, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        _reap_descendants(pid)
        proc.communicate()
        raise
    finally:
        # Reap any vLLM workers re-parented away from the (now-exited) direct child so
        # they cannot hold the GPU for a subsequent phase.
        _reap_descendants(pid)

    if proc.returncode != 0:
        stderr_tail = (stderr or "")[-_STDERR_TAIL_BYTES:]
        raise SubprocessIsolationError(
            f"Isolated phase '{target_module}' exited rc={proc.returncode}.\n"
            f"--- child stderr tail ({_STDERR_TAIL_BYTES} bytes) ---\n{stderr_tail}"
        )

    if not out_path.exists():
        stderr_tail = (stderr or "")[-_STDERR_TAIL_BYTES:]
        raise SubprocessIsolationError(
            f"Isolated phase '{target_module}' exited rc=0 but wrote no result to "
            f"{out_path}. The child must write a JSON result dict to sys.argv[2].\n"
            f"--- child stderr tail ({_STDERR_TAIL_BYTES} bytes) ---\n{stderr_tail}"
        )

    raw = out_path.read_text()
    try:
        result = json.loads(raw)
    except json.JSONDecodeError as e:
        raise SubprocessIsolationError(
            f"Isolated phase '{target_module}' wrote unparseable JSON to {out_path}: {e}\n"
            f"--- result file head (500 chars) ---\n{raw[:500]}"
        ) from e

    if not isinstance(result, dict):
        raise SubprocessIsolationError(
            f"Isolated phase '{target_module}' result must be a JSON object (dict), "
            f"got {type(result).__name__}."
        )

    return result


def _echo_main(argv: list[str]) -> int:
    """Round-trip child entry point: read payload, write it back as the result.

    Used as the canonical ``python -m`` target in the ``run_isolated`` unit test
    (``python -m explore_persona_space.orchestrate.subprocess_isolation <in> <out>``).
    Reads the payload JSON from ``argv[1]``, writes it verbatim (plus an ``_echoed``
    marker) to ``argv[2]``. Returns a process exit code.
    """
    if len(argv) < 3:
        sys.stderr.write("usage: subprocess_isolation <input_json> <output_json>\n")
        return 2
    in_path, out_path = Path(argv[1]), Path(argv[2])
    payload = json.loads(in_path.read_text())
    result = dict(payload)
    result["_echoed"] = True
    Path(out_path).write_text(json.dumps(result, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(_echo_main(sys.argv))
