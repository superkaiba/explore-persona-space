"""Tests for the failure-classifier routing helper.

The /issue skill (Step 7) routes `epm:failure` markers to either the
experimenter (re-spawn on infra failures) or experiment-implementer
(re-spawn on code failures). The routing logic is implemented as a
small pure function in `scripts/failure_classifier.py` so this test
suite can verify it in isolation.

The 5 cases below correspond to the 5 routing paths in the plan §4.16
"Failure-class quick reference" table.
"""
# ruff: noqa: E501  — fixture log/traceback strings intentionally use realistic paths

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).parent.parent / "scripts" / "failure_classifier.py"
spec = importlib.util.spec_from_file_location("failure_classifier", SCRIPT_PATH)
assert spec is not None and spec.loader is not None
failure_classifier = importlib.util.module_from_spec(spec)
sys.modules["failure_classifier"] = failure_classifier
spec.loader.exec_module(failure_classifier)

classify_failure = failure_classifier.classify_failure


def test_explicit_infra() -> None:
    """`failure_class: infra` field at top of body wins over body content."""
    body = """failure_class: infra

Some random Traceback in src/explore_persona_space/train/trainer.py
"""
    assert classify_failure(body) == "infra"


def test_explicit_code() -> None:
    """`failure_class: code` field at top of body wins over body content."""
    body = """failure_class: code

CUDA out of memory occurred during forward pass.
"""
    assert classify_failure(body) == "code"


def test_missing_field_cuda_oom_routes_infra() -> None:
    """Missing field + CUDA OOM in body → infra (log-pattern fallback)."""
    body = """## Failure during run

Traceback (most recent call last):
  File "...", line 42, in forward
RuntimeError: CUDA out of memory. Tried to allocate 2.0 GiB
"""
    assert classify_failure(body) == "infra"


def test_missing_field_src_traceback_routes_code() -> None:
    """Missing field + Traceback from `src/explore_persona_space/` → code."""
    body = """## Failure during run

Traceback (most recent call last):
  File "/workspace/explore-persona-space/src/explore_persona_space/train/trainer.py", line 137, in step
    raise AssertionError("invariant violated")
AssertionError: invariant violated
"""
    assert classify_failure(body) == "code"


def test_missing_field_no_pattern_routes_code() -> None:
    """Missing field + no pattern match → code (conservative fallback)."""
    body = """## Failure during run

The pipeline emitted weird output but no clear error pattern.
"""
    assert classify_failure(body) == "code"


def test_library_traceback_routes_infra() -> None:
    """Library tracebacks (vllm/transformers/peft/trl/torch/xformers) → infra."""
    body = """## Failure during run

Traceback (most recent call last):
  File "/usr/local/lib/python3.11/site-packages/vllm/engine/llm_engine.py", line 88
"""
    assert classify_failure(body) == "infra"


def test_ssh_refused_routes_infra() -> None:
    body = "ssh_execute failed: SSH connection refused\n"
    assert classify_failure(body) == "infra"


# --- DataLoader-worker wrap (workflow-fix from /issue 480) ----------------


def test_dataloader_wrap_our_code_routes_code() -> None:
    """torch DataLoader wraps a worker-side our-code raise; classify on the
    WRAPPED Original Traceback, not the outer torch frames.

    The outer frames are always under torch/ (worker.py, _utils/, ...) so
    the generic library-traceback infra pattern would otherwise route this
    to `infra`. The fix isolates the wrapped block: if its deepest frame
    is in our code (src/explore_persona_space/ or scripts/), route `code`.
    """
    body = """## Failure during run

Traceback (most recent call last):
  File "/usr/local/lib/python3.11/site-packages/torch/utils/data/_utils/worker.py", line 308, in _worker_loop
    data = fetcher.fetch(index)
  File "/usr/local/lib/python3.11/site-packages/torch/utils/data/_utils/fetch.py", line 54, in fetch
    return self.collate_fn(data)
RuntimeError: Caught RuntimeError in DataLoader worker process 0.
Original Traceback (most recent call last):
  File "/usr/local/lib/python3.11/site-packages/torch/utils/data/_utils/worker.py", line 308, in _worker_loop
    data = fetcher.fetch(index)
  File "/usr/local/lib/python3.11/site-packages/torch/utils/data/_utils/fetch.py", line 54, in fetch
    return self.collate_fn(data)
  File "/workspace/explore-persona-space/src/explore_persona_space/train/sft.py", line 412, in __call__
    raise RuntimeError("marker token id mismatch")
RuntimeError: marker token id mismatch
"""
    assert classify_failure(body) == "code"


def test_dataloader_wrap_cuda_oom_stays_infra() -> None:
    """When the wrapped block is a genuine infra error (CUDA OOM, ENOSPC,
    NCCL, ...), the wrap special case still classifies via the normal
    infra-pattern scan on the wrapped text and stays `infra`."""
    body = """RuntimeError: Caught RuntimeError in DataLoader worker process 0.
Original Traceback (most recent call last):
  File "/usr/local/lib/python3.11/site-packages/torch/utils/data/_utils/worker.py", line 308, in _worker_loop
    ...
  File "/usr/local/lib/python3.11/site-packages/torch/_tensor.py", line 1234, in to
    return self._to_copy(device)
RuntimeError: CUDA out of memory. Tried to allocate 2.0 GiB
"""
    assert classify_failure(body) == "infra"


def test_dataloader_wrap_scripts_frame_routes_code() -> None:
    """Wrapped block with a deepest frame under scripts/ (not just src/)
    also routes `code`. Covers dispatcher scripts that raise from
    collate_fn / Dataset / DataLoader callbacks."""
    body = """RuntimeError: Caught ValueError in DataLoader worker process 1.
Original Traceback (most recent call last):
  File "/usr/local/lib/python3.11/site-packages/torch/utils/data/_utils/worker.py", line 308, in _worker_loop
    data = fetcher.fetch(index)
  File "/workspace/explore-persona-space/scripts/issue480_payload_swap.py", line 88, in collate
    raise ValueError("bad row shape")
ValueError: bad row shape
"""
    assert classify_failure(body) == "code"


def test_dataloader_wrap_explicit_field_still_wins() -> None:
    """The explicit `failure_class:` field still has top precedence even
    when a DataLoader wrap is present."""
    body = """failure_class: infra

RuntimeError: Caught RuntimeError in DataLoader worker process 0.
Original Traceback (most recent call last):
  File "/workspace/explore-persona-space/src/explore_persona_space/train/sft.py", line 1, in x
    raise RuntimeError("x")
"""
    assert classify_failure(body) == "infra"


# --- Co-located parallel-cell OOM (workflow-fix from task #557) ------------


def test_colocation_oom_multi_sibling_pids_routes_code() -> None:
    """#557 regression: a CUDA OOM listing 2+ sibling 'Process NNN has
    X GiB memory in use' entries means parallel fan-out cells co-located
    on one physical GPU — a deterministic GPU-pinning bug (code), NOT
    transient infra. Respawning on verified-clean GPUs hits the
    identical OOM (attempt 2 of #557 did exactly that)."""
    body = """## Failure during run

torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 1.50 GiB. GPU 0 has a total capacity of 79.18 GiB of which 41.00 MiB is free. Process 568053 has 50.74 GiB memory in use. Process 568050 has 14.72 GiB memory in use. Process 568055 has 13.66 GiB memory in use.
"""
    assert classify_failure(body) == "code"


def test_single_sibling_pid_oom_stays_infra() -> None:
    """A SINGLE sibling-process entry is one leaked process from a prior
    run — kill + respawn fixes it, so the normal CUDA-OOM infra route
    stands."""
    body = """RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB. GPU 0 has a total capacity of 79.18 GiB of which 1.00 GiB is free. Process 123456 has 70.00 GiB memory in use.
"""
    assert classify_failure(body) == "infra"


def test_explicit_field_wins_over_colocation_oom() -> None:
    """The explicit `failure_class:` field keeps top precedence over the
    co-location carve-out."""
    body = """failure_class: infra

CUDA out of memory. Process 11 has 10.00 GiB memory in use. Process 22 has 20.00 GiB memory in use.
"""
    assert classify_failure(body) == "infra"


# --- vLLM engine-init free-memory (workflow-fix from task #601) -------------


def test_vllm_engine_init_free_memory_routes_infra() -> None:
    """#601: vLLM engine init fails with 'Free memory on device (...) is
    less than desired GPU memory utilization' when orphaned
    VLLM::EngineCore workers from a prior crash still hold the GPUs.
    Routes `infra` — recoverable in-place (kill orphans, relaunch same
    pod), per the recovery note on the named pattern. The body here
    carries ONLY the final error line (no vllm/ traceback frames), which
    previously matched no infra pattern and fell through to `code`."""
    body = """## Failure during relaunch

ValueError: Free memory on device (10.50/79.18 GiB) on startup is less than desired GPU memory utilization (0.9, 71.26 GiB). Decrease GPU memory utilization or reduce GPU memory used by other processes.
"""
    assert classify_failure(body) == "infra"


def test_vllm_engine_init_free_memory_line_wrapped_routes_infra() -> None:
    """Log shippers / markdown bodies sometimes hard-wrap the message; the
    pattern is DOTALL so the signature still matches across lines."""
    body = """## Failure during relaunch

ValueError: Free memory on device (10.50/79.18 GiB) on startup
is less than desired GPU memory utilization (0.9, 71.26 GiB).
"""
    assert classify_failure(body) == "infra"


# --- §2 watchdog regex extensions -----------------------------------------


def test_stall_reason_routed_to_infra() -> None:
    """The §2 pod.py-watch watchdog posts `reason: stall` bodies. The
    INFRA_PATTERNS extension routes them via the regex fallback even when
    the explicit `failure_class:` field is missing (e.g., body reformatted
    upstream)."""
    body = "## Stall detected\n\nreason: stall\nlast_event: 2025-01-01\n"
    assert classify_failure(body) == "infra"


def test_probe_unreachable_reason_routed_to_infra() -> None:
    """`reason: probe_unreachable` (other watchdog exit path) also routes
    to infra under the regex fallback."""
    body = "## Probe unreachable\n\nreason: probe_unreachable\n"
    assert classify_failure(body) == "infra"


# --- Zombie-GPU stall reason (workflow-fix from #664) ----------------------


def test_known_stall_reason_routes_infra() -> None:
    """#664: a `status=stalled` tick whose `stall_reason` is
    `vllm_worker_dead_zombie_gpu` routes `infra` directly — a silent hang
    has no traceback / OOM line in its log tail, so the regex fallback
    would otherwise misroute it to the conservative `code` default. The
    experimenter respawn (reap the orphan EngineCore by exact PID,
    relaunch on the same pod) is the fix, NOT an implementer round."""
    body = """## Stall detected

The run made no log/GPU/CPU progress for 60+ minutes and was overridden
to stalled by the zombie-GPU probe.
"""
    # Without the stall_reason this body matches no infra pattern -> code.
    assert classify_failure(body) == "code"
    # With the known stall_reason it routes infra.
    assert classify_failure(body, stall_reason="vllm_worker_dead_zombie_gpu") == "infra"


def test_persistent_wedge_veto_yield_routes_infra() -> None:
    """#1840: a `status=stalled` tick whose `stall_reason` is
    `persistent_wedge_veto_yield` (the namespace veto yielded after N
    consecutive wedge-regime ticks — #1768's alive-but-wedged HF download)
    routes `infra` directly: the wedge's log tail is silent (no traceback /
    OOM line), so the regex fallback would misroute it to the conservative
    `code` default. Recovery is an experimenter respawn (container-side
    kill of the wedged workers by cmdline/session match, then relaunch),
    NOT a code fix."""
    body = """## Stall detected

The namespace veto yielded after 10 consecutive suppressed ticks with all
logs and outputs frozen, every GPU idle, and the GPU-idle escalation posted.
"""
    # Without the stall_reason this body matches no infra pattern -> code.
    assert classify_failure(body) == "code"
    # With the known stall_reason it routes infra.
    assert classify_failure(body, stall_reason="persistent_wedge_veto_yield") == "infra"


def test_slurm_heartbeat_and_log_stale_routes_infra() -> None:
    """#1969: the SLURM monitor's stalled verdict — SLURM says RUNNING
    while the rsync'd status.json heartbeat AND job.out mtime are BOTH
    stale for >= 2 consecutive ticks — routes `infra` directly. A
    genuine cluster-side silent hang carries no traceback / OOM pattern
    in its log tail, so the regex fallback would misroute it to the
    conservative `code` default; recovery is an experimenter respawn
    (scancel + resubmit), not a code fix."""
    body = """## Stall detected

SLURM reports RUNNING but the heartbeat and job.out have both been
stale past STALL_SEC for two consecutive poll ticks.
"""
    # Without the stall_reason this body matches no infra pattern -> code.
    assert classify_failure(body) == "code"
    # With the known stall_reason it routes infra.
    assert classify_failure(body, stall_reason="slurm_heartbeat_and_log_stale") == "infra"


def test_explicit_field_wins_over_stall_reason() -> None:
    """An explicit `failure_class:` field keeps top precedence over a
    known `stall_reason`."""
    body = """failure_class: code

CUDA-worker died but the explicit field pins this to code.
"""
    assert classify_failure(body, stall_reason="vllm_worker_dead_zombie_gpu") == "code"


def test_unknown_stall_reason_falls_through_to_scan() -> None:
    """An unknown stall_reason does NOT short-circuit; the body is scanned
    normally (here, no infra pattern -> conservative code default)."""
    body = "## Stall\n\nopaque stall with no pattern\n"
    assert classify_failure(body, stall_reason="some_future_reason") == "code"


def test_unknown_stall_reason_still_matches_infra_pattern() -> None:
    """An unknown stall_reason leaves a genuine infra body on the infra
    route (the regex scan still runs)."""
    body = "## Stall\n\nRuntimeError: CUDA out of memory.\n"
    assert classify_failure(body, stall_reason="some_future_reason") == "infra"


def test_cli_stall_reason_routes_infra() -> None:
    """The Step 7 shell-out forwards the poll JSON line's stall_reason via
    --stall-reason; a known reason routes infra even with a no-pattern
    body."""
    import subprocess

    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--body",
            "silent hang, no pattern in body",
            "--stall-reason",
            "vllm_worker_dead_zombie_gpu",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert proc.stdout.strip() == "infra"


# --- CLI integration -------------------------------------------------------


def test_cli_via_stdin_routes_infra(tmp_path: Path) -> None:
    """The /issue skill Step 7 shells out to the script via stdin.

    Verify the CLI contract end-to-end: pipe a body via `--body -` and
    read a single-line `infra`/`code` verdict from stdout.
    """
    import subprocess

    body = "Traceback...\nRuntimeError: CUDA out of memory\n"
    proc = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--body", "-"],
        input=body,
        capture_output=True,
        text=True,
        check=True,
    )
    assert proc.stdout.strip() == "infra"


def test_cli_with_log_file_routes_infra(tmp_path: Path) -> None:
    """`--log <path>` concatenates the log tail into the body before scan."""
    import subprocess

    log = tmp_path / "run.log"
    log.write_text("normal startup line\n" * 50 + "NCCL timeout occurred\n")
    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--body",
            "[no-pattern body]",
            "--log",
            str(log),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert proc.stdout.strip() == "infra"


def test_cli_default_routes_code() -> None:
    """No pattern match → conservative `code` verdict on stdout."""
    import subprocess

    proc = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--body", "weird unknown failure"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert proc.stdout.strip() == "code"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
