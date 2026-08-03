"""Tests for the #1655 provision-time account-key preflight.

Covers the pure decision core (``decide_account_key_preflight`` — the plan
§4.1 six-row matrix), the authorized_keys line parser
(``_authorized_key_fields``), the ``runpod_api`` helpers (``read_vm_pubkey``
extraction refactor + ``get_account_pubkey``), the fail-open wrapper
(``_account_key_preflight``), and the ``cmd_provision`` wiring (preflight
runs BEFORE any create call).

NO live API anywhere: helper tests monkeypatch ``runpod_api.graphql``;
policy/wiring tests monkeypatch ``pod_lifecycle.get_account_pubkey`` /
``pod_lifecycle.read_vm_pubkey`` — the module-level import bindings, which
is what ``cmd_provision`` resolves.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import pod_lifecycle  # noqa: E402
import runpod_api  # noqa: E402
from pod_lifecycle import (  # noqa: E402
    _authorized_key_fields,
    decide_account_key_preflight,
)
from runpod_api import RunPodError  # noqa: E402

VM_KEY = "ssh-ed25519 AAAAC3vmblob vm-comment"


# ---------------------------------------------------------------------------
# decide_account_key_preflight — the §4.1 matrix
# ---------------------------------------------------------------------------


def test_decide_ok_when_vm_key_in_account_list():
    """Row 1: parsed identity + key present -> ok, with the no-overclaim clause."""
    account = "ssh-rsa OTHERBLOB other\nssh-ed25519 AAAAC3vmblob console-comment"
    verdict, msg = decide_account_key_preflight(VM_KEY, account)
    assert verdict == "ok"
    assert "not verifiable" in msg  # provider-side seeding no-overclaim clause


def test_decide_warn_when_account_list_lacks_key_but_injection_available():
    """Row 2: parsed identity + key absent -> warn naming the shared-list hazard."""
    verdict, msg = decide_account_key_preflight(VM_KEY, "ssh-rsa OTHERBLOB other")
    assert verdict == "warn"
    assert "fellows" in msg
    assert "SSH Public Keys" in msg


@pytest.mark.parametrize(
    "account_blob",
    ["", "   \n\n", "not-a-key at all\n# just a comment\nssh-ed25519\n"],
    ids=["empty", "whitespace", "no-parseable-lines"],
)
def test_decide_fail_only_when_no_identity_and_account_list_empty(account_blob):
    """Row 5 (v3 semantics): no local identity AND zero parseable account keys -> fail."""
    verdict, msg = decide_account_key_preflight(None, account_blob)
    assert verdict == "fail"
    assert "ZERO" in msg


def test_decide_fail_open_on_query_error_even_with_missing_pubkey_file():
    """Row 6: no identity + query error -> warn (loudest), NEVER fail."""
    verdict, msg = decide_account_key_preflight(None, None, query_error="boom")
    assert verdict == "warn"
    assert "boom" in msg
    assert "either" in msg  # cannot verify either path


def test_decide_warn_loudest_when_no_identity_but_account_list_nonempty():
    """Row 4 (v3 semantics): no identity + >=1 parseable account key -> warn, never fail."""
    verdict, msg = decide_account_key_preflight(None, "ssh-ed25519 SOMEBLOB someone")
    assert verdict == "warn"
    assert "cannot be" in msg  # membership undecidable without a local identity
    assert "1" in msg  # names the listed-key count


def test_match_ignores_comment_field_differences():
    """Match is on (key_type, base64_blob) only — comments legitimately differ."""
    verdict, _ = decide_account_key_preflight(
        "ssh-ed25519 AAAAC3vmblob totally-different-comment",
        "ssh-ed25519 AAAAC3vmblob console@runpod",
    )
    assert verdict == "ok"


def test_decide_warn_on_query_error_with_readable_identity():
    """Row 3: parsed identity + query error -> warn (fail-open)."""
    verdict, msg = decide_account_key_preflight(VM_KEY, None, query_error="HTTP 500")
    assert verdict == "warn"
    assert "SKIPPED" in msg
    assert "HTTP 500" in msg


def test_decide_no_identity_when_pubkey_file_single_token():
    """A readable, ssh--prefixed but <2-token file yields NO identity: the
    warn-loudest row with a non-empty account blob (malformed => no identity)."""
    verdict, msg = decide_account_key_preflight("ssh-ed25519", "ssh-rsa BBBBLOB b")
    assert verdict == "warn"
    assert "cannot be" in msg


def test_match_tolerates_options_prefixed_account_lines():
    """An options-prefixed account line still matches on the key-type token."""
    account = 'from="1.2.3.4",no-pty ssh-ed25519 AAAAC3vmblob c'
    verdict, _ = decide_account_key_preflight(VM_KEY, account)
    assert verdict == "ok"


def test_hard_fail_message_names_cause_and_remediation():
    """The fail message names both broken paths + both remediations + the
    do-not-script warning on the whole-list-replacing mutation."""
    verdict, msg = decide_account_key_preflight(None, "")
    assert verdict == "fail"
    assert "id_ed25519.pub" in msg  # pubkey-file path hint
    assert "team account" in msg
    assert "ZERO" in msg or "NO authorized key" in msg
    assert "SSH Public Keys" in msg  # console remediation
    assert "updateUserSettings" in msg
    assert "do not script" in msg


# ---------------------------------------------------------------------------
# _authorized_key_fields
# ---------------------------------------------------------------------------


def test_authorized_key_fields_parses_plain_and_option_lines():
    assert _authorized_key_fields("ssh-ed25519 BLOB comment") == ("ssh-ed25519", "BLOB")
    assert _authorized_key_fields('from="x" ecdsa-sha2-nistp256 BLOB') == (
        "ecdsa-sha2-nistp256",
        "BLOB",
    )
    assert _authorized_key_fields("") is None
    assert _authorized_key_fields("ssh-ed25519") is None  # <2 tokens
    assert _authorized_key_fields("# comment line") is None


# ---------------------------------------------------------------------------
# runpod_api helpers
# ---------------------------------------------------------------------------


def test_get_account_pubkey_parses_myself_pubkey(monkeypatch):
    blob = "ssh-ed25519 AAA a\nssh-rsa BBB b"
    monkeypatch.setattr(
        runpod_api, "graphql", lambda query, **kw: {"myself": {"id": "x", "pubKey": blob}}
    )
    assert runpod_api.get_account_pubkey() == blob


def test_get_account_pubkey_empty_when_null(monkeypatch):
    monkeypatch.setattr(
        runpod_api, "graphql", lambda query, **kw: {"myself": {"id": "x", "pubKey": None}}
    )
    assert runpod_api.get_account_pubkey() == ""


def test_read_vm_pubkey_none_on_missing_file(monkeypatch, tmp_path):
    monkeypatch.setenv("RUNPOD_SSH_PUBKEY_FILE", str(tmp_path / "missing.pub"))
    assert runpod_api.read_vm_pubkey() is None
    # Non-'ssh-'-prefixed content is equally "no usable key" (fail-open parity).
    bad = tmp_path / "bad.pub"
    bad.write_text("ecdsa-sha2-nistp256 BLOB c\n")  # valid key type, but not 'ssh-'
    monkeypatch.setenv("RUNPOD_SSH_PUBKEY_FILE", str(bad))
    assert runpod_api.read_vm_pubkey() is None


def test_public_key_env_consistent_with_read_vm_pubkey(monkeypatch, tmp_path):
    """Refactor guard: _public_key_env() is None <=> read_vm_pubkey() is None,
    and the GraphQL fragment embeds the scrubbed (backslash/quote-free) key."""
    # Missing file: both None.
    monkeypatch.setenv("RUNPOD_SSH_PUBKEY_FILE", str(tmp_path / "missing.pub"))
    assert runpod_api.read_vm_pubkey() is None
    assert runpod_api._public_key_env() is None
    # Valid file with GraphQL-hostile characters: both non-None; fragment scrubbed.
    pub = tmp_path / "id.pub"
    pub.write_text('ssh-ed25519 AB"C\\D comment\n')
    monkeypatch.setenv("RUNPOD_SSH_PUBKEY_FILE", str(pub))
    raw = runpod_api.read_vm_pubkey()
    assert raw == 'ssh-ed25519 AB"C\\D comment'  # raw read: no scrub
    fragment = runpod_api._public_key_env()
    assert fragment is not None
    assert "ssh-ed25519 ABCD comment" in fragment  # scrub stays in _public_key_env
    assert '"PUBLIC_KEY"' in fragment


# ---------------------------------------------------------------------------
# _account_key_preflight wrapper
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "exc",
    [RunPodError("boom"), TimeoutError("read timeout"), ValueError("Expecting value")],
    ids=["RunPodError", "TimeoutError-OSError", "ValueError-JSONDecode"],
)
def test_wrapper_fails_open_on_each_escape_class(monkeypatch, capsys, exc):
    """Every documented query-failure class fails OPEN: no SystemExit, a loud
    WARN/SKIPPED line on stderr, wrapper returns normally."""
    monkeypatch.delenv("EPM_SKIP_ACCOUNT_KEY_PREFLIGHT", raising=False)
    monkeypatch.setattr(pod_lifecycle, "read_vm_pubkey", lambda: VM_KEY)

    def _raise():
        raise exc

    monkeypatch.setattr(pod_lifecycle, "get_account_pubkey", _raise)
    assert pod_lifecycle._account_key_preflight("pod-1655") is None  # no SystemExit
    err = capsys.readouterr().err
    assert "[WARN]" in err
    assert "SKIPPED" in err


def test_kill_switch_skips_preflight(monkeypatch, capsys):
    monkeypatch.setenv("EPM_SKIP_ACCOUNT_KEY_PREFLIGHT", "1")
    calls: list[str] = []
    monkeypatch.setattr(pod_lifecycle, "get_account_pubkey", lambda: calls.append("query") or "")
    monkeypatch.setattr(pod_lifecycle, "read_vm_pubkey", lambda: calls.append("read") or VM_KEY)
    pod_lifecycle._account_key_preflight("pod-1655")  # no exit
    assert calls == []  # neither the query nor the file read ran
    assert "SKIPPED (EPM_SKIP_ACCOUNT_KEY_PREFLIGHT=1)" in capsys.readouterr().err


def test_wrapper_exits_1_on_fail_row(monkeypatch, capsys):
    """The both-paths-broken row is the ONLY sys.exit(1) arm."""
    monkeypatch.delenv("EPM_SKIP_ACCOUNT_KEY_PREFLIGHT", raising=False)
    monkeypatch.setattr(pod_lifecycle, "read_vm_pubkey", lambda: None)
    monkeypatch.setattr(pod_lifecycle, "get_account_pubkey", lambda: "")
    with pytest.raises(SystemExit) as exc_info:
        pod_lifecycle._account_key_preflight("pod-1655")
    assert exc_info.value.code == 1
    assert "[FAIL]" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# cmd_provision wiring
# ---------------------------------------------------------------------------


def _provision_args(issue: int = 9999) -> argparse.Namespace:
    return argparse.Namespace(
        list_intents=False,
        issue=issue,
        name_suffix=None,
        intent=None,
        dry_run=False,
        volume_gb=200,
        container_disk_gb=50,
    )


def test_cmd_provision_runs_account_key_preflight_before_any_create(monkeypatch):
    """The preflight fires after the idempotency loop and BEFORE any
    create_pod / create_cpu_pod call (its fail arm blocks creation)."""
    monkeypatch.setattr(
        pod_lifecycle, "_warn_on_terminal_parent_provision", lambda issue, **kw: False
    )
    monkeypatch.setattr(pod_lifecycle, "_warn_on_lifecycle_escapes", lambda pods: None)
    monkeypatch.setattr(pod_lifecycle, "list_team_pods", lambda: [])
    preflight_calls: list[str] = []

    def _record_then_fail(pod_label: str) -> None:
        preflight_calls.append(pod_label)
        sys.exit(1)  # model the fail arm

    monkeypatch.setattr(pod_lifecycle, "_account_key_preflight", _record_then_fail)
    monkeypatch.setattr(
        pod_lifecycle,
        "create_pod",
        lambda *a, **kw: pytest.fail("create_pod called despite preflight fail"),
    )
    monkeypatch.setattr(
        pod_lifecycle,
        "create_cpu_pod",
        lambda *a, **kw: pytest.fail("create_cpu_pod called despite preflight fail"),
    )
    with pytest.raises(SystemExit):
        pod_lifecycle.cmd_provision(_provision_args())
    assert preflight_calls == ["pod-9999"]
