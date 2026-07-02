"""Tests for verify_task_body.py check 25 — `check_audit_availability_claims_match_hf`.

Check 25 catches the #653 round-6 pattern: a clean-result body asserts a
data artifact "was not uploaded" / "cannot be audited" while that artifact
actually EXISTS on the HF data repo at the body's own linked, revision-pinned
data-repo tree. The check scans the fence-stripped body for an
availability-denial phrase co-located with a known data-artifact keyword,
then probes the body's HF Hub revision-pinned URLs (the check-23 set) at the
URL's own path AND at that path extended by the denied keyword. A SUCCESSFUL
listing returning ≥1 matching file → FAIL; everything indeterminate
(EPM_VERIFY_BODY_NO_HF=1 fence, missing huggingface_hub, network/HTTP error)
→ PASS with an `unverified` note.

The suite-wide EPM_VERIFY_BODY_NO_HF=1 fence (tests/conftest.py) makes the
probe SKIP, so tests that exercise the real FAIL/PASS branching delenv the
fence and stub `verify_task_body._hf_tree_get` — the single bounded primitive
both the check-23 and check-25 probes funnel through (#733) — to return a
chosen `_TreeProbeResult` without any network (the same approach used by the
check-23 tests in test_verify_task_body.py). An autouse raise-by-default
guard turns any probe a test did not explicitly stub into a hard error, so
hermeticity never depends on network / offline / Hub repo state.
"""

# The fixture body strings below INCLUDE the literal markdown content the
# verifier scans, including long prose lines and the multiplication-sign
# character (U+00D7) that appears in real clean-result write-ups.

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

# Load the verifier as a module (it's a script, not a package member).
_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "verify_task_body.py"
_spec = importlib.util.spec_from_file_location("verify_task_body", _SCRIPT)
verify_task_body = importlib.util.module_from_spec(_spec)  # type: ignore[arg-type]
sys.modules["verify_task_body"] = verify_task_body
_spec.loader.exec_module(verify_task_body)  # type: ignore[union-attr]

_CHECK_NAME = "audit-availability claims match HF Hub"

_HF_REPO_TREE = (
    "https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/"
    "tree/a64f6fd7fb6dc66cfd370bfa8592a6f00af9c66e/issue653_install-validated-reladder"
)


@pytest.fixture(autouse=True)
def _clear_hf_existence_cache():
    """Checks 23/25 memoize definitive verdicts in `_HF_EXISTENCE_CACHE` (#733);
    clear before AND after each test so a verdict keyed on the shared fixture
    (repo, sha, path[, keyword]) never leaks one test's stubbed outcome into
    another."""
    verify_task_body._HF_EXISTENCE_CACHE.clear()
    yield
    verify_task_body._HF_EXISTENCE_CACHE.clear()


@pytest.fixture(autouse=True)
def _no_unexpected_probes(monkeypatch):
    """Raise-by-default guard: ANY `_hf_tree_get` call a test did not
    explicitly stub is a hard error — missed-mock detection independent of
    network / offline / Hub repo state. Per-test `_stub_tree` re-patches over
    this (LIFO monkeypatch teardown restores cleanly)."""

    def _unexpected(url, params, headers, *, timeout_s):  # pragma: no cover
        raise AssertionError(
            f"unexpected _hf_tree_get probe of {url} — add _stub_tree or an explicit allowance"
        )

    monkeypatch.setattr(verify_task_body, "_hf_tree_get", _unexpected)


def _stub_tree(monkeypatch, *, status="ok", entries=(), next_page=None, note=""):
    """Replace `verify_task_body._hf_tree_get` (the single bounded primitive
    both the check-23 and check-25 probes funnel through, #733) with a stub
    returning a fixed `_TreeProbeResult` — no network. Shadows the autouse
    raise-by-default guard for tests that EXPECT a probe."""

    def _fake(url, params, headers, *, timeout_s):
        return verify_task_body._TreeProbeResult(status, list(entries), next_page, note)

    monkeypatch.setattr(verify_task_body, "_hf_tree_get", _fake)


# Tree entries matching the real tree-endpoint shape: dicts with "type"/"path",
# paths repo-relative (check 25 requires type == "file" and a full repo-relative
# path). The directory row keeps the fixture honest for check 23's root listing
# (not asserted).
_DIR_ENTRY = {"type": "directory", "path": "issue653_install-validated-reladder"}
_INSTALL_PROBES_FILE = {
    "type": "file",
    "path": (
        "issue653_install-validated-reladder/raw_completions/armB/"
        "install_probes/cell0/raw_completions.json"
    ),
}
_RAW_COMPLETIONS_FILE = {
    "type": "file",
    "path": "issue653_install-validated-reladder/raw_completions/run_seed42.json",
}
_MIXES_FILE = {
    "type": "file",
    "path": "issue653_install-validated-reladder/mixes/reladder.jsonl",
}
_ONPOLICY_POOLS_FILE = {
    "type": "file",
    "path": "issue653_install-validated-reladder/onpolicy_pools/pool_armB.jsonl",
}
_README_FILE = {"type": "file", "path": "README.md"}


def _results_by_name(results):
    return {r.name: r for r in results}


def _body(
    *,
    denial: bool,
    prose_term: str = "install-probe",
    hf_url: str | None = _HF_REPO_TREE,
    sentinel: str = "",
) -> str:
    """Build a minimal body with/without an availability-denial line and an
    optional HF revision-pinned data-repo tree link.

    ``prose_term`` is the HUMAN spelling of the artifact class as it appears in
    body prose (hyphen/space/singular, e.g. "install-probe", "raw completions")
    — deliberately NOT the underscore HF-path token, mirroring the real #653 r6
    line ("install-probe ... completions ... not separately uploaded ... cannot
    be audited") whose prose spelling differs from the `install_probes/` upload
    path. This is the exact case a naive underscore-only scan would miss.

    The body deliberately routes verify_text through the legacy (pre-sentinel)
    branch by default so only the generation-agnostic checks (incl. check 25)
    bind; the heavy v3/v4 structural checks PASS vacuously. We assert on the
    check-25 result by name, not on `ok`, so unrelated structural FAILs on
    this skeletal body never confuse the assertions.
    """
    denial_line = (
        "The on-policy training-positive pools are linked below. The per-cell "
        f"{prose_term} firing/non-firing completions themselves were not separately "
        "uploaded, so the firing vs non-firing examples behind the install rates cannot be "
        "audited at the record level here (acknowledged WARN)."
        if denial
        else f"The per-cell {prose_term} firing/non-firing completions are uploaded "
        "and inspectable at the record level."
    )
    link_line = f"Full re-ladder artifacts: [HF data repo]({hf_url})." if hf_url else ""
    # Pad the body well past the 500-char check-0 stub floor so the body is
    # never short-circuited (the no-hf-url variant drops the link line and
    # would otherwise dip under the floor before reaching check 25).
    return f"""{sentinel}# Some claim about install validation (MODERATE confidence)

## Takeaways

- A toy takeaway bullet stating one number-first claim about the result, with
  enough padding prose to clear the body-nonstub floor so every downstream
  check (including check 25) actually runs on this synthetic body rather than
  being short-circuited by the stub guard at the top of the check chain.

## Methodology

**Design:** a toy design line describing the conditions and the install probe.

{denial_line}

{link_line}
"""


def test_no_denial_claim_passes_even_when_hf_has_files(monkeypatch):
    """(c) A body with NO availability-denial claim is a vacuous PASS — the
    check is triggered only by the denial wording, never by mere presence of
    HF files."""
    monkeypatch.delenv("EPM_VERIFY_BODY_NO_HF", raising=False)
    # Files DO exist under the linked tree (check 23's probe also consumes this
    # stub with the fence delenv'd); check 25 must still pass vacuously.
    _stub_tree(monkeypatch, entries=[_DIR_ENTRY, _INSTALL_PROBES_FILE])
    body = _body(denial=False)
    _ok, results = verify_task_body.verify_text(body)
    by_name = _results_by_name(results)
    assert by_name[_CHECK_NAME].passed
    assert "no availability-denial claim" in by_name[_CHECK_NAME].detail


def test_denial_but_hf_has_files_fails(monkeypatch):
    """(a) The #653 r6 case: the body claims the install-probe completions
    "were not separately uploaded ... cannot be audited" while the files DO
    exist on HF Hub under the linked data-repo tree → FAIL."""
    monkeypatch.delenv("EPM_VERIFY_BODY_NO_HF", raising=False)
    # The keyword sub-path under the linked tree resolves to real files.
    _stub_tree(
        monkeypatch,
        entries=[_DIR_ENTRY, _INSTALL_PROBES_FILE, _MIXES_FILE, _README_FILE],
    )
    body = _body(denial=True, prose_term="install-probe")
    _ok, results = verify_task_body.verify_text(body)
    by_name = _results_by_name(results)
    assert not by_name[_CHECK_NAME].passed
    assert "install_probes" in by_name[_CHECK_NAME].detail
    assert "exists at" in by_name[_CHECK_NAME].detail


def test_denial_and_hf_genuinely_missing_passes(monkeypatch):
    """(b) A legitimate denial: the body says the artifact was not uploaded,
    and the HF listing genuinely has NO matching file under any candidate
    sub-path → the denial is true → PASS."""
    monkeypatch.delenv("EPM_VERIFY_BODY_NO_HF", raising=False)
    # The tree exists but holds only unrelated files — no install_probes path.
    _stub_tree(
        monkeypatch,
        entries=[_DIR_ENTRY, _MIXES_FILE, _ONPOLICY_POOLS_FILE, _README_FILE],
    )
    body = _body(denial=True, prose_term="install-probe")
    _ok, results = verify_task_body.verify_text(body)
    by_name = _results_by_name(results)
    assert by_name[_CHECK_NAME].passed
    # A definitive zero-file probe is not "unverified" — it confirms the denial.
    assert "unverified" not in by_name[_CHECK_NAME].detail


def test_denial_raw_completions_keyword_also_fires(monkeypatch):
    """The `raw_completions` keyword (the most common artifact class) is in
    scope just like `install_probes` — denial + existing file → FAIL."""
    monkeypatch.delenv("EPM_VERIFY_BODY_NO_HF", raising=False)
    _stub_tree(monkeypatch, entries=[_DIR_ENTRY, _RAW_COMPLETIONS_FILE])
    body = _body(denial=True, prose_term="raw completions")
    _ok, results = verify_task_body.verify_text(body)
    by_name = _results_by_name(results)
    assert not by_name[_CHECK_NAME].passed
    assert "raw_completions" in by_name[_CHECK_NAME].detail


def test_denial_but_no_hf_url_is_vacuous_pass(monkeypatch):
    """A denial claim with NO HF Hub revision-pinned URL in the body has
    nothing to reconcile against → vacuous PASS (the check never fabricates a
    FAIL without a probe target)."""
    monkeypatch.delenv("EPM_VERIFY_BODY_NO_HF", raising=False)
    # No per-test stub: the autouse raise-by-default `_hf_tree_get` guard IS the
    # must-not-be-called assertion (no URL ⇒ neither check 23 nor 25 probes).
    body = _body(denial=True, hf_url=None)
    _ok, results = verify_task_body.verify_text(body)
    by_name = _results_by_name(results)
    assert by_name[_CHECK_NAME].passed
    assert "no HF Hub revision-pinned URL" in by_name[_CHECK_NAME].detail


def test_hf_http_error_is_unverified_not_fail(monkeypatch):
    """An HF Hub HTTP / network error on the probe is INDETERMINATE → PASS
    with an `unverified` note, never a FAIL — a Hub outage must not break a
    body's verification."""
    monkeypatch.delenv("EPM_VERIFY_BODY_NO_HF", raising=False)
    # Mirror a transient Hub failure: `_hf_tree_get` maps 429/5xx/conn errors to
    # status="indeterminate" (never raises), which check 25 surfaces as `unverified`.
    _stub_tree(monkeypatch, status="indeterminate", note="HF tree probe failed: HTTP 503")
    body = _body(denial=True, prose_term="install-probe")
    _ok, results = verify_task_body.verify_text(body)
    by_name = _results_by_name(results)
    assert by_name[_CHECK_NAME].passed
    assert "unverified" in by_name[_CHECK_NAME].detail
    assert "install_probes" in by_name[_CHECK_NAME].detail


def test_env_fence_skips_probe(monkeypatch):
    """With the suite-wide EPM_VERIFY_BODY_NO_HF=1 fence in place (the conftest
    default), the probe SKIPs without touching the Hub → PASS with an
    `unverified` note even when files WOULD have been found."""
    monkeypatch.setenv("EPM_VERIFY_BODY_NO_HF", "1")
    # No per-test stub: the fence returns before any probe, so the autouse
    # raise-by-default `_hf_tree_get` guard IS the must-not-be-called assertion.
    body = _body(denial=True, prose_term="install-probe")
    _ok, results = verify_task_body.verify_text(body)
    by_name = _results_by_name(results)
    assert by_name[_CHECK_NAME].passed
    assert "unverified" in by_name[_CHECK_NAME].detail


def test_proximity_guard_denial_of_other_artifact_does_not_false_fire(monkeypatch):
    """A line that denies artifact A (merged weights) while merely LINKING
    artifact B (raw completions) by keyword far away must NOT attribute the
    denial to B → no FAIL even though raw_completions files exist on HF. The
    far-away keyword sits well beyond `_AUDIT_DENIAL_PROXIMITY` from the denial
    phrase, so it is not flagged."""
    monkeypatch.delenv("EPM_VERIFY_BODY_NO_HF", raising=False)

    def _boom_keyword(repo_id, repo_type, sha, path_prefix, keyword):  # pragma: no cover
        raise AssertionError("no artifact class flagged — the check-25 probe must not run")

    # Split mock: check 25 must never reach its keyword probe (no artifact class
    # sits within proximity of the denial, so empty `suspect_keywords` returns
    # before touching URLs) ...
    monkeypatch.setattr(verify_task_body, "_hf_probe_keyword", _boom_keyword)
    # ... while check 23 legitimately probes the body's HF URL with the fence
    # delenv'd — a benign tree stub serves it (raw_completions files DO exist).
    _stub_tree(monkeypatch, entries=[_DIR_ENTRY, _RAW_COMPLETIONS_FILE])
    # The denial ("not uploaded") attaches to the merged checkpoint; "raw
    # completions" is mentioned ~300 chars later, beyond the 200-char window.
    far_line = (
        "The merged full-precision checkpoint was not uploaded to save HF "
        "storage quota, and the LoRA adapter is the canonical artifact instead; "
        "all the other intermediate tensors and the per-seed training logs were "
        "also pruned from the pod after the run completed, but separately the raw "
        f"completions are fully available at [HF data repo]({_HF_REPO_TREE})."
    )
    body = f"""# A claim (MODERATE confidence)

## Takeaways

- A toy takeaway bullet with enough padding prose to comfortably clear the
  body-nonstub floor so every downstream check actually runs on this body
  rather than being short-circuited by the stub guard.

## Methodology

**Design:** a toy design line describing the conditions.

{far_line}
"""
    _ok, results = verify_task_body.verify_text(body)
    by_name = _results_by_name(results)
    assert by_name[_CHECK_NAME].passed
    assert "no availability-denial claim" in by_name[_CHECK_NAME].detail


def test_detector_unit_proximity_and_prose_spellings():
    """Unit test of `_audit_denied_artifact_classes_in`: it maps prose
    spellings to canonical HF-path tokens and proximity-gates the attribution."""
    f = verify_task_body._audit_denied_artifact_classes_in
    # #653 prose spelling, denial nearby → install_probes.
    assert f(
        "The install-probe completions were not separately uploaded and cannot be audited."
    ) == ["install_probes"]
    # underscore-path spelling also recognized.
    assert f("the raw_completions were not uploaded") == ["raw_completions"]
    # No denial phrase on the line → empty.
    assert f("the raw completions are uploaded and inspectable") == []
    # Denial present but no artifact class → empty.
    assert f("the figure was not uploaded") == []
    # Far-apart denial + artifact (>200 chars) → not attributed.
    far = "the weights were not uploaded" + (" filler" * 60) + " raw completions live at the repo"
    assert f(far) == []
