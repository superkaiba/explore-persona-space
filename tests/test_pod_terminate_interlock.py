"""Interlock pins for ``runpod_api.terminate_pod`` (#2075, plan §5 items i-k).

The containment interlock (commit 3a2f364a70, treated as given by the #2075
plan) makes ``terminate_pod`` the single RunPod destruction choke point:
without explicit user approval (``EPS_ALLOW_COMPUTE_KILL=1`` or the legacy
``EPS_ALLOW_POD_TERMINATE=1``) or an active thread-local
``backends.kill_approval.verified_teardown`` grant, it raises
``PodTerminateNotApproved`` BEFORE any GraphQL call. These tests pin:

- (i) refusal without approval — GraphQL is never reached;
- (j) both approval env vars authorize (value must be exactly ``"1"``);
- (k) the ``verified_teardown`` grant authorizes inside the block, on the
  SAME thread only, and expires at block exit (nested grants restore the
  outer one).

Hermetic — no live API calls, no live pushes: ``runpod_api.graphql`` is
monkeypatched (an AssertionError sentinel on refusal paths, a recording fake
on authorized paths) and ``_notify_terminate_blocked`` is monkeypatched so a
refusal can never touch the real ``~/.eps-autonomous`` sentinel dir or send
a real Telegram push.
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "src"))

import runpod_api  # noqa: E402

from explore_persona_space.backends.kill_approval import (  # noqa: E402
    compute_kill_approved,
    verified_teardown,
)


@pytest.fixture(autouse=True)
def _hermetic(monkeypatch: pytest.MonkeyPatch):
    """No approval env, no live push, no accidental GraphQL traffic."""
    monkeypatch.delenv("EPS_ALLOW_COMPUTE_KILL", raising=False)
    monkeypatch.delenv("EPS_ALLOW_POD_TERMINATE", raising=False)
    monkeypatch.setattr(runpod_api, "_notify_terminate_blocked", lambda pod_id: None)

    def _graphql_forbidden(query, variables=None, timeout=None):
        raise AssertionError("GraphQL must not be reached without approval")

    monkeypatch.setattr(runpod_api, "graphql", _graphql_forbidden)


def _allow_graphql(monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    """Swap the forbidden-graphql sentinel for a recording success fake."""
    calls: list[dict] = []

    def fake_graphql(query, variables=None, timeout=None):
        calls.append({"query": query, "variables": variables})
        return {"podTerminate": None}

    monkeypatch.setattr(runpod_api, "graphql", fake_graphql)
    return calls


# ── (i) refusal without approval ─────────────────────────────────────────────


def test_terminate_refused_without_approval_never_reaches_graphql():
    # The autouse graphql sentinel raises AssertionError if reached — the
    # PodTerminateNotApproved below therefore proves the refusal fired FIRST.
    with pytest.raises(runpod_api.PodTerminateNotApproved):
        runpod_api.terminate_pod("id-teammate-pod")


def test_refusal_fires_blocked_notification(monkeypatch: pytest.MonkeyPatch):
    seen: list[str] = []
    monkeypatch.setattr(runpod_api, "_notify_terminate_blocked", seen.append)
    with pytest.raises(runpod_api.PodTerminateNotApproved):
        runpod_api.terminate_pod("id-x")
    assert seen == ["id-x"]


def test_refusal_message_names_approval_paths():
    with pytest.raises(runpod_api.PodTerminateNotApproved) as exc_info:
        runpod_api.terminate_pod("id-x")
    msg = str(exc_info.value)
    assert "REFUSED" in msg
    assert "--approve" in msg  # the CLI approval path is named for the operator


# ── (j) env-var authorization ────────────────────────────────────────────────


@pytest.mark.parametrize("env_var", ["EPS_ALLOW_COMPUTE_KILL", "EPS_ALLOW_POD_TERMINATE"])
def test_env_var_authorizes(monkeypatch: pytest.MonkeyPatch, env_var: str):
    calls = _allow_graphql(monkeypatch)
    monkeypatch.setenv(env_var, "1")
    assert runpod_api.terminate_pod("id-approved") is True
    assert len(calls) == 1
    assert calls[0]["variables"] == {"id": "id-approved"}


def test_env_var_must_be_exactly_one(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("EPS_ALLOW_COMPUTE_KILL", "true")  # not "1" — refused
    with pytest.raises(runpod_api.PodTerminateNotApproved):
        runpod_api.terminate_pod("id-x")


# ── (k) verified_teardown thread-local grant ─────────────────────────────────


def test_grant_authorizes_inside_block_and_expires_after(monkeypatch: pytest.MonkeyPatch):
    calls = _allow_graphql(monkeypatch)
    with verified_teardown(target="pod-x", reason="epm:upload-verification PASS"):
        assert runpod_api.terminate_pod("id-owned") is True
    assert len(calls) == 1
    with pytest.raises(runpod_api.PodTerminateNotApproved):
        runpod_api.terminate_pod("id-after-block")
    assert len(calls) == 1  # the grant expired with the block


def test_grant_does_not_leak_to_sibling_thread(monkeypatch: pytest.MonkeyPatch):
    """The grant is thread-local by design: a cron/watcher/janitor thread can
    never inherit an owner session's verified-teardown approval."""
    _allow_graphql(monkeypatch)
    outcome: dict[str, object] = {}

    def sibling():
        outcome["approved"] = compute_kill_approved()
        try:
            runpod_api.terminate_pod("id-sibling")
            outcome["raised"] = False
        except runpod_api.PodTerminateNotApproved:
            outcome["raised"] = True

    with verified_teardown(target="pod-x", reason="epm:upload-verification PASS"):
        assert compute_kill_approved() is True
        t = threading.Thread(target=sibling)
        t.start()
        t.join()
    assert outcome == {"approved": False, "raised": True}


def test_nested_grant_restores_outer_not_clears():
    with verified_teardown(target="outer", reason="epm:upload-verification PASS"):
        with verified_teardown(target="inner", reason="epm:upload-verification PASS"):
            assert compute_kill_approved() is True
        assert compute_kill_approved() is True  # outer grant restored
    assert compute_kill_approved() is False
