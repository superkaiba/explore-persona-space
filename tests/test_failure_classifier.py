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
