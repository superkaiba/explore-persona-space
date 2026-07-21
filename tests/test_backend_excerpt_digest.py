"""#1574 shared trigger-dense digest module — single-implementation pins.

The #1556 structural-digest helpers moved from ``scripts/poll_pipeline.py``
into ``explore_persona_space.backends.excerpt_digest`` so the GCP / SLURM
lane monitors (which build their OWN ``log_tail_excerpt`` strings) consume
the SAME implementation. These tests pin:

1. poll_pipeline's private names are ALIASES of the shared module's objects
   (``is``-level identity — the "gauge tests the dispatched path"
   discipline, ``.claude/rules/code-style.md`` § verification gates);
2. the shared ``issue_trigger_dense`` body across its four read paths via
   the injectable ``get_task_fn`` / ``log`` seams (autospec'd boundary —
   no live-registry read), plus the fail-fast propagation of an unexpected
   exception class;
3. the LAZY ``get_task_fn=None`` default resolves ``task_workflow.get_task``
   at CALL time (pinned hermetically by monkeypatching the task_workflow
   module attribute — the #1574 critic's explicit default-arm ask);
4. the CUDA-IMA mirror is byte-in-sync with the REAL
   ``backend_poll.CUDA_IMA_SIGNATURE`` (direct src-side pin — belt to the
   aliased pin in tests/test_poll_pipeline_digest.py).
"""

from __future__ import annotations

import importlib.util
import logging
import sys
from pathlib import Path
from unittest.mock import create_autospec

import pytest

from explore_persona_space.backends import excerpt_digest

REPO_ROOT = Path(__file__).resolve().parents[1]

# Synthetic issue id (never a real task — the live-state coupling class the
# #1556 plan §4 row 11 names); the #1574 family id, distinct from the sibling
# poll test files' ids (9556 / 9664 / 9813 / 9999 / 9983 / 9704).
ISSUE = 9574


def _load_script_module(filename: str, alias: str):
    """Load a ``scripts/*.py`` file as a module (mirrors the
    ``tests/test_poll_pipeline_digest.py`` loader)."""
    spec = importlib.util.spec_from_file_location(alias, REPO_ROOT / "scripts" / filename)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    spec.loader.exec_module(module)
    return module


pp = _load_script_module("poll_pipeline.py", "poll_pipeline_shared_digest_under_test")
bp = _load_script_module("backend_poll.py", "backend_poll_shared_digest_under_test")


# ── 1. identity: the aliases ARE the shared module's objects ─────────────────


def test_shared_module_is_the_dispatched_implementation() -> None:
    """poll_pipeline dispatches the SHARED implementation, not a copy —
    ``is``-level identity, so a divergent fork can never reappear silently."""
    assert pp._digest_tail_excerpt is excerpt_digest.digest_tail_excerpt
    assert pp._CUDA_IMA_SIGNATURE is excerpt_digest.CUDA_IMA_SIGNATURE_MIRROR
    assert pp._DIGEST_PATTERNS is excerpt_digest.DIGEST_PATTERNS
    assert pp._TRIGGER_DENSE_TAG == excerpt_digest.TRIGGER_DENSE_TAG


# ── 2. the shared predicate body across its four read paths ──────────────────


def test_issue_trigger_dense_injectable_four_arms(caplog: pytest.LogCaptureFixture) -> None:
    """Real shared body, autospec'd ``get_task_fn`` boundary + passed logger.

    Verbatim #1556 semantics: tag present -> True; other tags -> False;
    missing task (FileNotFoundError) -> False + INFO ("raw excerpt");
    unreadable state (RuntimeError class) -> True + loud WARNING ("digest");
    an unexpected exception class propagates (fail-fast).
    """
    from explore_persona_space.task_workflow import get_task as real_get_task

    test_log = logging.getLogger("excerpt-digest-test-log")

    fake = create_autospec(
        real_get_task,
        return_value={"status": "running", "frontmatter": {"tags": ["trigger-dense"]}},
    )
    assert excerpt_digest.issue_trigger_dense(ISSUE, get_task_fn=fake, log=test_log) is True
    fake.assert_called_once_with(ISSUE)

    other = create_autospec(
        real_get_task,
        return_value={"status": "running", "frontmatter": {"tags": ["keep-running"]}},
    )
    assert excerpt_digest.issue_trigger_dense(ISSUE, get_task_fn=other, log=test_log) is False

    fnf = create_autospec(real_get_task, side_effect=FileNotFoundError("task #9574 not found"))
    with caplog.at_level(logging.INFO, logger="excerpt-digest-test-log"):
        caplog.clear()
        assert excerpt_digest.issue_trigger_dense(ISSUE, get_task_fn=fnf, log=test_log) is False
    infos = [r for r in caplog.records if r.levelno == logging.INFO]
    assert infos, "missing-task arm must log INFO (not silent)"
    assert "not found" in infos[0].getMessage()
    assert "raw excerpt" in infos[0].getMessage()

    boom = create_autospec(
        real_get_task, side_effect=RuntimeError("branch-guard: HEAD is not main")
    )
    with caplog.at_level(logging.WARNING, logger="excerpt-digest-test-log"):
        caplog.clear()
        assert excerpt_digest.issue_trigger_dense(ISSUE, get_task_fn=boom, log=test_log) is True
    warns = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert warns, "unreadable-state arm must WARN loudly (not swallow)"
    msg = warns[0].getMessage()
    assert "RuntimeError" in msg and "branch-guard: HEAD is not main" in msg
    assert "digest" in msg

    unexpected = create_autospec(real_get_task, side_effect=KeyError("boom"))
    with pytest.raises(KeyError):
        excerpt_digest.issue_trigger_dense(ISSUE, get_task_fn=unexpected, log=test_log)


def test_issue_trigger_dense_lazy_default_resolves_task_workflow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The ``get_task_fn=None`` default lazily resolves
    ``task_workflow.get_task`` at CALL time — pinned hermetically by
    monkeypatching the task_workflow module attribute, so this test never
    reads the live registry (the #1574 critic's explicit default-arm ask)."""
    import explore_persona_space.task_workflow as tw

    calls: list[int] = []

    def fake_get_task(issue: int) -> dict:
        calls.append(issue)
        return {"status": "running", "frontmatter": {"tags": ["trigger-dense"]}}

    monkeypatch.setattr(tw, "get_task", fake_get_task)
    assert excerpt_digest.issue_trigger_dense(ISSUE) is True
    assert calls == [ISSUE]


# ── 3. the CUDA-IMA mirror pin (direct src-side belt) ────────────────────────


def test_cuda_ima_mirror_matches_backend_poll() -> None:
    """``CUDA_IMA_SIGNATURE_MIRROR`` stays byte-in-sync with the REAL
    ``backend_poll.CUDA_IMA_SIGNATURE`` (pattern + flags) — the #775
    cross-pod marker-note fallback greps digested notes through it."""
    assert excerpt_digest.CUDA_IMA_SIGNATURE_MIRROR.pattern == bp.CUDA_IMA_SIGNATURE.pattern
    assert excerpt_digest.CUDA_IMA_SIGNATURE_MIRROR.flags == bp.CUDA_IMA_SIGNATURE.flags
