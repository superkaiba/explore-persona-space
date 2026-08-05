"""Load-bearing invariants for the #1491 scale-ladder analysis scripts.

These pin two behaviours that are correct today but are protected ONLY by a
comment, and whose silent regression would corrupt published artifacts rather
than crash:

1. ``_remote_manifest_meta`` exception ORDERING. ``LocalEntryNotFoundError`` is
   a subclass of ``EntryNotFoundError`` and is the transient/offline class that
   ``hub.retry_transient`` re-raises once its budget exhausts. If the clauses
   are ever reordered so the subclass falls through to the plain
   ``EntryNotFoundError`` branch, a SUSTAINED Hub outage reads as "nothing
   published", ``assert_no_silent_content_drift`` is skipped, and an in-place
   re-upload can replace the pinned contexts every scale is captured against.
   A reorder is a one-line edit with no visible symptom — hence this test.

2. ``_confound_status`` can never return a bare ``controls-present``. Plan §4
   makes a verdict readable as a scale effect only under Δ/ΔΓ sign stability,
   which nothing computes; the status vocabulary must not be able to claim
   otherwise once control KEYS start appearing.

Deliberately network-free and CPU-only: everything in ``tests/`` runs in every
issue's Step 9c gate, so a live Hub fetch here would turn the fleet red on any
HF outage or 429 storm.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import issue1491_ladder_contrasts as CON  # noqa: E402
import issue1491_ladder_manifest as MAN  # noqa: E402


@pytest.fixture
def passthrough_retry(monkeypatch):
    """Make hub.retry_transient call its thunk exactly once.

    Without this the LocalEntryNotFoundError case would sit in the real retry
    budget (default 1800 s) — that budget is hub's concern and is tested there;
    what THIS module pins is the except-clause ordering around it.
    """
    from explore_persona_space.orchestrate import hub

    monkeypatch.setattr(hub, "retry_transient", lambda fn, **kw: fn())
    return hub


def _patch_download_to_raise(monkeypatch, exc):
    import huggingface_hub

    def _boom(*a, **k):
        raise exc

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", _boom)


def test_remote_manifest_meta_reraises_local_entry_not_found(passthrough_retry, monkeypatch):
    """A sustained transport outage must NOT read as 'nothing published'.

    LocalEntryNotFoundError subclasses EntryNotFoundError, so clause order is
    load-bearing: caught by the wrong branch it returns None and silently
    disarms the content-drift refusal.
    """
    from huggingface_hub.errors import EntryNotFoundError, LocalEntryNotFoundError

    # Guard the premise itself — if upstream ever stops subclassing, the
    # ordering rationale changes and this test should say so loudly.
    assert issubclass(LocalEntryNotFoundError, EntryNotFoundError), (
        "premise changed: LocalEntryNotFoundError is no longer an "
        "EntryNotFoundError subclass — re-derive the ordering rationale"
    )

    _patch_download_to_raise(monkeypatch, LocalEntryNotFoundError("hub unreachable"))
    with pytest.raises(LocalEntryNotFoundError):
        MAN._remote_manifest_meta()


def test_remote_manifest_meta_returns_none_on_genuine_absence(passthrough_retry, monkeypatch):
    """A real 404 still means 'not yet published' — the fix must not over-raise."""
    from huggingface_hub.errors import EntryNotFoundError

    _patch_download_to_raise(monkeypatch, EntryNotFoundError("no such file"))
    assert MAN._remote_manifest_meta() is None


def test_remote_manifest_meta_reraises_repository_not_found(passthrough_retry, monkeypatch):
    """A missing/inaccessible repo is a config fault, never evidence of absence."""
    from huggingface_hub.errors import RepositoryNotFoundError

    _patch_download_to_raise(monkeypatch, RepositoryNotFoundError("no repo"))
    with pytest.raises(RepositoryNotFoundError):
        MAN._remote_manifest_meta()


def test_confound_status_confounded_without_controls():
    assert CON._confound_status({})["status"] == "sample-efficiency-confounded"


def test_confound_status_requires_computed_truncation_cost():
    """Control (c) needs the COMPUTED value, not merely the 7B refit cell."""
    fits = {"s": {"n_ladder": 1, "rp896": 1}, "scale7_refit": {}}
    assert CON._confound_status(fits)["status"] == "sample-efficiency-confounded"


def test_confound_status_never_returns_bare_controls_present():
    """Even with every control key present, the status must stay honest.

    Plan §4 requires Δ/ΔΓ sign stability before a verdict reads as a scale
    effect; nothing computes it, so the strongest returnable status is the
    explicit sign-stability-unchecked form.
    """
    fits = {
        "s": {"n_ladder": 1, "rp896": 1},
        "scale7_refit": {"truncation_cost_r2": 0.02},
    }
    out = CON._confound_status(fits)
    assert out["status"] == "controls-present-sign-stability-unchecked"
    assert out["sign_stability_checked"] is False
    assert out["status"] != "controls-present"
