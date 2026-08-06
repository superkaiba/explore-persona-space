"""Tests for #2153 — detached HF transfers: timeout + observable progress.

Covers the four deliverable surfaces:

- **D0** — ``workflow_lint --check-snapshot-download-allow-patterns``
  fixtures: dataset-repo + ``allow_patterns`` FAILs (kwarg, positional
  repo_id, aliased import); model-repo ``allow_patterns`` passes; a
  no-``allow_patterns`` dataset pull passes; the
  ``# SNAPSHOT_ALLOW_PATTERNS_EXEMPT`` waiver in both placements (short
  reason still flags); frozen-snapshot exemption; live tree passes; plus
  the MUTATION-VISIBLE no-flags dispatch run (the
  ``tests/test_workflow_lint_jsonl_splitlines.py`` pattern — a direct call
  of the check function is NOT sufficient evidence of bundling).
- **D1** — prose pins on the new ``upload-policy.md`` § "Detached HF
  transfers: timeout + observable progress" (header, sizing-basis sentence,
  the 0-byte-log invariant, one-worker-per-``local_dir``, the xet
  stays-ON-by-default decision) and the ``pod-side-reporting.md`` pointer.
- **D2** — discriminator-correction pins across BOTH rule files: the
  "zero TCP connections" download-vs-upload discriminator is GONE and the
  corrected socket-count-does-NOT-discriminate claim is present (#1739
  download hangs held exactly ONE socket); ``du -sb`` frozen + ``py-spy``
  parked in ``xet_get`` stay the load-bearing probes.
- **D3** — behavioral tests for the hardened ``hub.stage_hub_prefix``
  (HF calls stubbed at the per-NAME sites, mirroring
  ``tests/test_hub_staging_retry.py``): entry line flushed BEFORE any
  network call; one flushed progress line per completed file; the
  ``EPM_HF_STAGE_TIMEOUT_S`` wall timeout hard-exits via ``os._exit`` with
  the distinct rc after flushing a stalled-file diagnostic; unset/empty env
  = OFF; a per-file failure still PROPAGATES (never routed into the
  timeout hard-exit).

Until this branch merges, run with ``PYTHONPATH=<worktree>/src`` so the
worktree's ``explore_persona_space`` (which carries the hardened helper)
shadows the editable install pointing at main.
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent
_SCRIPTS = _REPO_ROOT / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import workflow_lint as wl  # noqa: E402
from workflow_lint import (  # noqa: E402
    SNAPSHOT_ALLOW_PATTERNS_FROZEN_SNAPSHOT,
    check_snapshot_download_allow_patterns,
)

from explore_persona_space.orchestrate import hub  # noqa: E402

RULES = _REPO_ROOT / ".claude" / "rules"
DIAG_TOKEN = "[snapshot-download-allow-patterns]"


def _plant(root: Path, rel: str, body: str) -> Path:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(body, encoding="utf-8")
    return p


def _norm(text: str) -> str:
    """Whitespace-normalized (prose pins must survive reflow/wrapping)."""
    return " ".join(text.split())


# ---------------------------------------------------------------------------
# D0 — the lint check
# ---------------------------------------------------------------------------

OFFENDER_KWARG = (
    "from huggingface_hub import snapshot_download\n"
    "def stage():\n"
    '    snapshot_download(repo_id="superkaiba1/explore-persona-space-data",\n'
    '                      repo_type="dataset", allow_patterns=["issueX/*"])\n'
)


def test_dataset_kwarg_offender_flagged(tmp_path):
    _plant(tmp_path, "scripts/offender.py", OFFENDER_KWARG)
    errors = check_snapshot_download_allow_patterns(repo_root=tmp_path)
    assert len(errors) == 1, errors
    assert DIAG_TOKEN in errors[0] and "scripts/offender.py:3" in errors[0]
    assert "stage_hub_prefix" in errors[0]  # the fix recipe is named


def test_positional_repo_id_attribute_call_flagged(tmp_path):
    """Attribute-receiver leg + repo_id as FIRST POSITIONAL, no repo_type."""
    _plant(
        tmp_path,
        "scripts/offender.py",
        "from huggingface_hub import HfApi\n"
        "api = HfApi()\n"
        'api.snapshot_download("superkaiba1/explore-persona-space-data",\n'
        '                      allow_patterns=["issueX/*"])\n',
    )
    errors = check_snapshot_download_allow_patterns(repo_root=tmp_path)
    assert len(errors) == 1 and "scripts/offender.py:3" in errors[0], errors


def test_aliased_import_flagged(tmp_path):
    _plant(
        tmp_path,
        "scripts/offender.py",
        "from huggingface_hub import snapshot_download as sd\n"
        'sd(repo_id="org/x", repo_type="dataset", allow_patterns=["a/*"])\n',
    )
    errors = check_snapshot_download_allow_patterns(repo_root=tmp_path)
    assert len(errors) == 1, errors


def test_model_repo_allow_patterns_passes(tmp_path):
    """A small-model-repo pull with allow_patterns is the legitimate common
    case (e.g. issue1947_localization_panel.py) — never flagged."""
    _plant(
        tmp_path,
        "scripts/fine.py",
        "from huggingface_hub import snapshot_download\n"
        'snapshot_download(repo_id="Qwen/Qwen2.5-7B", allow_patterns=["*.json"])\n',
    )
    assert check_snapshot_download_allow_patterns(repo_root=tmp_path) == []


def test_dataset_without_allow_patterns_passes(tmp_path):
    """Bare snapshot_download (no allow_patterns) is a NAMED residual of this
    check — out of its predicate by design."""
    _plant(
        tmp_path,
        "scripts/fine.py",
        "from huggingface_hub import snapshot_download\n"
        'snapshot_download(repo_id="superkaiba1/explore-persona-space-data",\n'
        '                  repo_type="dataset")\n',
    )
    assert check_snapshot_download_allow_patterns(repo_root=tmp_path) == []


@pytest.mark.parametrize("placement", ["same-line", "previous-line"])
def test_waiver_suppresses(tmp_path, placement):
    if placement == "same-line":
        call = (
            'snapshot_download(repo_id="superkaiba1/explore-persona-space-data", '
            'repo_type="dataset", allow_patterns=["a/*"])  '
            "# SNAPSHOT_ALLOW_PATTERNS_EXEMPT: tiny pinned sibling repo, 12 files\n"
        )
    else:
        call = (
            "# SNAPSHOT_ALLOW_PATTERNS_EXEMPT: tiny pinned sibling repo, 12 files\n"
            'snapshot_download(repo_id="superkaiba1/explore-persona-space-data", '
            'repo_type="dataset", allow_patterns=["a/*"])\n'
        )
    _plant(
        tmp_path,
        "scripts/waived.py",
        "from huggingface_hub import snapshot_download\n" + call,
    )
    assert check_snapshot_download_allow_patterns(repo_root=tmp_path) == []


def test_short_waiver_reason_still_flags(tmp_path):
    _plant(
        tmp_path,
        "scripts/waived.py",
        "from huggingface_hub import snapshot_download\n"
        'snapshot_download(repo_id="superkaiba1/explore-persona-space-data", '
        'repo_type="dataset", allow_patterns=["a/*"])  '
        "# SNAPSHOT_ALLOW_PATTERNS_EXEMPT: ok\n",
    )
    errors = check_snapshot_download_allow_patterns(repo_root=tmp_path)
    assert len(errors) == 1, errors


def test_frozen_snapshot_member_exempt(tmp_path):
    """A pre-existing offender at a frozen-snapshot path is grandfathered —
    the check gates NEW code only (plan hard constraint 5)."""
    frozen_rel = "scripts/issue811_stage_phase0.py"
    assert frozen_rel in SNAPSHOT_ALLOW_PATTERNS_FROZEN_SNAPSHOT
    _plant(tmp_path, frozen_rel, OFFENDER_KWARG)
    assert check_snapshot_download_allow_patterns(repo_root=tmp_path) == []


def test_frozen_snapshot_entries_are_experiment_shaped():
    """Every grandfathered path is a per-issue experiment artifact — never a
    workflow-surface module (the allowlist-shape pin convention)."""
    import re

    shape = re.compile(r"^(scripts/(issue|i\d)|src/explore_persona_space/experiments/)")
    bad = [p for p in SNAPSHOT_ALLOW_PATTERNS_FROZEN_SNAPSHOT if not shape.match(p)]
    assert bad == [], f"non-experiment-shaped frozen-snapshot entries: {bad}"


def test_live_tree_passes():
    """The committed tree passes — pins the frozen snapshot's completeness so
    the no-flags default run (pre-commit / Step 9c) cannot break on it."""
    errors = check_snapshot_download_allow_patterns()
    assert errors == [], "\n".join(errors)


def test_check_snapshot_download_allow_patterns_dispatched_no_flags(tmp_path, capsys, monkeypatch):
    """MUTATION-VISIBLE no-flags dispatch: deleting the check's ``or
    no_flags`` branch must fail this test. Other bundled checks contribute
    unrelated errors on the minimal tree, so the assertion keys on the
    check's own diagnostic token + the offending path."""
    _plant(tmp_path, "scripts/offender2153.py", OFFENDER_KWARG)
    monkeypatch.setattr(wl, "_REPO_ROOT", tmp_path)
    rc = wl.main([])
    err = capsys.readouterr().err
    assert rc != 0, f"no-flags default run exited 0 on an offending tree:\n{err}"
    assert "snapshot-download-allow-patterns" in err and "offender2153.py" in err, (
        f"the snapshot-download-allow-patterns diagnostic (naming "
        f"offender2153.py) is missing from the no-flags run's stderr:\n{err}"
    )


# ---------------------------------------------------------------------------
# D1 — upload-policy section + pod-side-reporting pointer (prose pins)
# ---------------------------------------------------------------------------


def test_upload_policy_detached_transfer_section_present():
    text = (RULES / "upload-policy.md").read_text(encoding="utf-8")
    assert "## Detached HF transfers: timeout + observable progress (#2153)" in text
    norm = _norm(text)
    # (a) wall-clock timeout + the MANDATORY sizing-basis sentence
    assert "A wall-clock timeout bounding the WHOLE transfer" in norm
    assert (
        "projected bytes ÷ measured-or-expected throughput at ≥2× margin"  # noqa: RUF001 — pins the rule file's literal MULTIPLICATION SIGN
        in norm
    ), "the sizing-basis derivation sentence is missing"
    assert "~N files × `EPM_HF_RETRY_BUDGET_S`, default 1800 s" in norm  # noqa: RUF001 — literal MULTIPLICATION SIGN pinned
    assert "`EPM_HF_RETRY_BUDGET_S` retry budget is NOT a timeout" in norm
    # the built-in helper arm
    assert "EPM_HF_STAGE_TIMEOUT_S" in norm and "STAGE_HUB_PREFIX_TIMEOUT_RC" in norm
    # (b) widened progress trigger, canonical shape
    assert "[<phase>] unit k/N <key> elapsed=<s>s" in norm
    assert "regardless of file count or projected wall-time" in norm
    # (c) completion keyed on process exit
    assert "Completion keyed on process EXIT with a captured rc" in norm
    # (d) the invariant, binding from the FIRST instruction
    assert '0-byte log + empty target must NOT be a reachable "looks healthy" state' in norm
    assert "binds from the transfer's FIRST instruction" in norm
    # (e) one worker per local_dir
    assert "One worker per `local_dir`" in norm
    # deliverable-4 decision: xet stays ON, per-workload disables
    assert "xet stays ON by default" in norm and "scoped PER WORKLOAD" in norm
    assert "before its `huggingface_hub` import" in norm
    assert "scripts/issue1739_restore_partial.py" in norm
    assert "#3266" in norm  # the download-side coverage-gap grounding


def test_pod_side_reporting_pointer_present():
    text = _norm((RULES / "pod-side-reporting.md").read_text(encoding="utf-8"))
    assert "Detached HF transfers: timeout + observable progress" in text
    assert "upload-policy.md" in text


# ---------------------------------------------------------------------------
# D2 — corrected socket-count discriminator, all copies
# ---------------------------------------------------------------------------


def test_gotchas_discriminator_corrected():
    text = (RULES / "gotchas.md").read_text(encoding="utf-8")
    norm = _norm(text)
    assert "ZERO TCP" not in text and "zero TCP" not in text, (
        "the retired zero-TCP download-vs-upload discriminator claim is back"
    )
    assert "socket COUNT does NOT discriminate" in norm
    assert "#1739 download hangs held exactly ONE socket" in norm
    # the load-bearing probes stay (plan D2: those held up in #1739)
    assert "`du -sb <dest>` frozen across 2+ probes" in norm
    assert "`py-spy dump` parked in `xet_get`" in norm


def test_upload_policy_discriminator_corrected():
    text = (RULES / "upload-policy.md").read_text(encoding="utf-8")
    norm = _norm(text)
    assert "ZERO TCP" not in text and "zero TCP" not in text
    # both former claim sites carry the corrected statement
    assert norm.lower().count("socket count does not discriminate") >= 2, (
        "expected the corrected discriminator statement at BOTH former "
        "claim sites (wedge-ladder preamble + staging-legs scope note)"
    )


def test_agent_memory_long_form_carries_no_zero_tcp_claim():
    """Plan D2: re-verify at implement time — the experimenter long-form
    memory never carried the claim; pin that it stays absent."""
    p = (
        _REPO_ROOT
        / ".claude"
        / "agent-memory"
        / "experimenter"
        / "feedback_hf_xet_download_wedge_kill_replay.md"
    )
    if not p.exists():  # memory files may be pruned; absence is fine
        pytest.skip("long-form memory file not present in this tree")
    text = p.read_text(encoding="utf-8")
    assert "ZERO TCP" not in text and "zero TCP" not in text


# ---------------------------------------------------------------------------
# D3 — hardened stage_hub_prefix (HF stubbed at the per-NAME sites)
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_stage_env(monkeypatch):
    """Tests own EPM_HF_STAGE_TIMEOUT_S explicitly; never inherit it. Also
    no real sleeps + attempt-bound retries (the #735 convention)."""
    monkeypatch.delenv("EPM_HF_STAGE_TIMEOUT_S", raising=False)
    monkeypatch.setattr(hub.time, "sleep", lambda s: None)
    monkeypatch.setenv("EPM_HF_RETRY_BUDGET_S", "0")


class _FakeApi:
    def __init__(self, token=None):
        pass

    def repo_info(self, repo_id, repo_type=None):
        return SimpleNamespace(sha="abc123")


def _fake_stage_factory(record=None, block_names=(), release=None, fail_names=()):
    """Signature-conformant stage_hub_file fake (mirrors the real def)."""

    def fake_stage(
        repo_id,
        path_in_repo,
        target,
        *,
        repo_type="dataset",
        revision=None,
        token=None,
        overwrite=False,
    ):
        if path_in_repo in fail_names:
            raise RuntimeError(f"per-file staging failed: {path_in_repo}")
        if path_in_repo in block_names:
            assert release is not None
            release.wait(timeout=10)
        if record is not None:
            record.append(path_in_repo)
        return Path(target)

    return fake_stage


def test_entry_line_flushed_before_any_network_call(tmp_path, capsys, monkeypatch):
    """The entry line prints BEFORE the (retried) listing — a wedged listing
    must not reproduce the #1739 0-byte-log signature. Proven by a listing
    stub that raises: the raise propagates, yet the entry line is already
    on stdout, naming repo@revision:prefix."""

    def raising_list(api, repo_id, path, *, repo_type="model", revision=None):
        raise RuntimeError("listing wedged")

    monkeypatch.setattr("huggingface_hub.HfApi", _FakeApi)
    monkeypatch.setattr(hub, "list_hf_files_under_path", raising_list)
    with pytest.raises(RuntimeError, match="listing wedged"):
        hub.stage_hub_prefix("org/data", "pfx", tmp_path / "dest", revision="deadbeef")
    out = capsys.readouterr().out
    assert "[stage_hub_prefix] start org/data@deadbeef:pfx" in out
    # revision=None renders the entry line with @main (resolved only later)
    with pytest.raises(RuntimeError):
        hub.stage_hub_prefix("org/data", "pfx", tmp_path / "dest")
    assert "[stage_hub_prefix] start org/data@main:pfx" in capsys.readouterr().out


def test_progress_line_per_completed_file_and_order_preserved(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr("huggingface_hub.HfApi", _FakeApi)
    monkeypatch.setattr(
        hub,
        "list_hf_files_under_path",
        lambda *a, **k: ["pfx/a.json", "pfx/sub/b.json"],
    )
    monkeypatch.setattr(hub, "stage_hub_file", _fake_stage_factory())
    dest = tmp_path / "dest"
    out_paths = hub.stage_hub_prefix("org/data", "pfx", dest)
    out = capsys.readouterr().out
    assert "[stage_hub_prefix] 2 files under org/data@abc123:pfx" in out
    assert "unit 1/2" in out and "unit 2/2" in out
    assert "pfx/a.json" in out and "pfx/sub/b.json" in out
    # return contract unchanged: verbatim mirror paths, in listing order
    assert out_paths == [dest / "pfx/a.json", dest / "pfx/sub/b.json"]


class _ExitCalled(BaseException):
    """Raised by the os._exit stub — BaseException so no handler eats it."""

    def __init__(self, rc):
        self.rc = rc


def test_wall_timeout_hard_exits_distinct_rc_after_diagnostic(tmp_path, capsys, monkeypatch):
    """EPM_HF_STAGE_TIMEOUT_S armed + one worker parked forever: the helper
    flushes a stalled-file diagnostic then hard-exits os._exit(rc=87) —
    never a raise (an unjoinable native worker would wedge the atexit
    thread join, defeating completion-keyed-on-rc)."""
    release = threading.Event()
    exits: list[int] = []

    def fake_exit(rc):
        exits.append(rc)
        release.set()  # unblock the parked worker so pool shutdown joins
        raise _ExitCalled(rc)

    monkeypatch.setattr("huggingface_hub.HfApi", _FakeApi)
    monkeypatch.setattr(
        hub,
        "list_hf_files_under_path",
        lambda *a, **k: ["pfx/fast.json", "pfx/stalled.json"],
    )
    monkeypatch.setattr(
        hub,
        "stage_hub_file",
        _fake_stage_factory(block_names={"pfx/stalled.json"}, release=release),
    )
    monkeypatch.setattr(hub.os, "_exit", fake_exit)
    monkeypatch.setenv("EPM_HF_STAGE_TIMEOUT_S", "0.05")

    with pytest.raises(_ExitCalled):
        hub.stage_hub_prefix("org/data", "pfx", tmp_path / "dest")

    assert exits == [hub.STAGE_HUB_PREFIX_TIMEOUT_RC]
    assert hub.STAGE_HUB_PREFIX_TIMEOUT_RC == 87  # distinct, collision-probed
    out = capsys.readouterr().out
    assert "[stage_hub_prefix] TIMEOUT after" in out
    assert "pfx/stalled.json" in out  # the stalled file is NAMED
    assert f"rc={hub.STAGE_HUB_PREFIX_TIMEOUT_RC}" in out


def test_timeout_unset_or_empty_is_off(tmp_path, monkeypatch):
    """Unset/empty env = OFF: no existing caller changes behavior (plan hard
    constraint 4). A file slower than any plausible mis-parse of '' still
    completes and returns."""
    import time as _time

    def slow_stage(
        repo_id,
        path_in_repo,
        target,
        *,
        repo_type="dataset",
        revision=None,
        token=None,
        overwrite=False,
    ):
        _time.sleep(0.15)
        return Path(target)

    monkeypatch.setattr("huggingface_hub.HfApi", _FakeApi)
    monkeypatch.setattr(hub, "list_hf_files_under_path", lambda *a, **k: ["pfx/a.json"])
    monkeypatch.setattr(hub, "stage_hub_file", slow_stage)
    monkeypatch.setenv("EPM_HF_STAGE_TIMEOUT_S", "")  # empty string = OFF too
    dest = tmp_path / "dest"
    assert hub.stage_hub_prefix("org/data", "pfx", dest) == [dest / "pfx/a.json"]


@pytest.mark.parametrize("value", ["0", "0.0", "-1"])
def test_timeout_non_positive_is_off_not_instant_expiry(tmp_path, monkeypatch, value):
    """A non-positive env value reads as OFF — never as a 0 s fence that
    hard-exits every call. '0' is how a caller spells 'disabled' (#2153 code
    review, Minor 1); armed-at-zero would rc-87 the whole staging path."""
    import time as _time

    def fake_exit(rc):  # pragma: no cover - must never run
        raise AssertionError(f"os._exit({rc}) reached with the timeout set to {value!r}")

    def slow_stage(
        repo_id,
        path_in_repo,
        target,
        *,
        repo_type="dataset",
        revision=None,
        token=None,
        overwrite=False,
    ):
        _time.sleep(0.05)
        return Path(target)

    monkeypatch.setattr("huggingface_hub.HfApi", _FakeApi)
    monkeypatch.setattr(hub, "list_hf_files_under_path", lambda *a, **k: ["pfx/a.json"])
    monkeypatch.setattr(hub, "stage_hub_file", slow_stage)
    monkeypatch.setattr(hub.os, "_exit", fake_exit)
    monkeypatch.setenv("EPM_HF_STAGE_TIMEOUT_S", value)
    dest = tmp_path / "dest"
    assert hub.stage_hub_prefix("org/data", "pfx", dest) == [dest / "pfx/a.json"]


def test_per_file_failure_propagates_never_hard_exits(tmp_path, monkeypatch):
    """A failed file PROPAGATES (the existing fail-loud contract) even with
    the timeout armed — the hard-exit path fires ONLY on the iterator's
    wall expiry, never on a failed future's re-raise."""

    def fake_exit(rc):  # pragma: no cover - must never run
        raise AssertionError(f"os._exit({rc}) reached on a per-file failure")

    monkeypatch.setattr("huggingface_hub.HfApi", _FakeApi)
    monkeypatch.setattr(
        hub,
        "list_hf_files_under_path",
        lambda *a, **k: ["pfx/a.json", "pfx/b.json"],
    )
    monkeypatch.setattr(hub, "stage_hub_file", _fake_stage_factory(fail_names={"pfx/b.json"}))
    monkeypatch.setattr(hub.os, "_exit", fake_exit)
    monkeypatch.setenv("EPM_HF_STAGE_TIMEOUT_S", "30")
    with pytest.raises(RuntimeError, match="per-file staging failed"):
        hub.stage_hub_prefix("org/data", "pfx", tmp_path / "dest")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
