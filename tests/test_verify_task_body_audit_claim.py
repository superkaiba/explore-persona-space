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

# --- #813 fixtures (#942: quota-hold denial family + unreduced/reduced/maps
# classes). Layout Hub-verified at the body-pinned revision:
# issue813_mapchange_substrate/{unreduced,reduced,maps}/... ---
_HF_813_TREE = (
    "https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/"
    "tree/b0d30307c1671cad575928e5abf5253c0c849dee/issue813_mapchange_substrate"
)
_813_DIR = {"type": "directory", "path": "issue813_mapchange_substrate"}
_UNREDUCED_FILE = {
    "type": "file",
    "path": "issue813_mapchange_substrate/unreduced/em/generic/q0001.npz",
}
_REDUCED_FILE = {
    "type": "file",
    "path": "issue813_mapchange_substrate/reduced/em/generic/summary.npz",
}
_MAPS_FILE = {
    "type": "file",
    "path": "issue813_mapchange_substrate/maps/em/generic_L14_map.npz",
}
_ANALYSIS_TENSORS_FILE = {
    "type": "file",
    "path": "issue653_install-validated-reladder/analysis_tensors/shift_tensor.npz",
}

# The two #813 v1 body sentences that escaped the gate, VERBATIM (from
# tasks/*/813/events.jsonl — the interp-critic / analyzer round notes quote
# the v1 body lines 87 and 51).
_813_LINE_87 = (
    "The unreduced store / reduced summaries / fitted maps remain on the pod "
    "under an HF public-storage quota hold (upload 403; never-delete-unuploaded "
    "applies)."
)
_813_LINE_51 = (
    "The frozen base responses ride the unreduced activation store "
    "(quota-held on the pod at write time)."
)


def _results_by_name(results):
    return {r.name: r for r in results}


def _body(
    *,
    denial: bool,
    prose_term: str = "install-probe",
    hf_url: str | None = _HF_REPO_TREE,
    sentinel: str = "",
    denial_line: str | None = None,
) -> str:
    """Build a minimal body with/without an availability-denial line and an
    optional HF revision-pinned data-repo tree link.

    ``prose_term`` is the HUMAN spelling of the artifact class as it appears in
    body prose (hyphen/space/singular, e.g. "install-probe", "raw completions")
    — deliberately NOT the underscore HF-path token, mirroring the real #653 r6
    line ("install-probe ... completions ... not separately uploaded ... cannot
    be audited") whose prose spelling differs from the `install_probes/` upload
    path. This is the exact case a naive underscore-only scan would miss.

    ``denial_line`` (#942) overrides the built line verbatim — used for the
    #813 quota-hold sentences and the new-denial-family fixtures; when None,
    the pre-#942 #653-shaped line is built exactly as before, so no earlier
    test changes behavior. When ``denial_line`` is given, ``denial`` /
    ``prose_term`` are ignored.

    The body deliberately routes verify_text through the legacy (pre-sentinel)
    branch by default so only the generation-agnostic checks (incl. check 25)
    bind; the heavy v3/v4 structural checks PASS vacuously. We assert on the
    check-25 result by name, not on `ok`, so unrelated structural FAILs on
    this skeletal body never confuse the assertions.
    """
    if denial_line is None:
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
    # --- #942: the #813 quota-hold family + unreduced/reduced/maps classes.
    # List-EQUALITY assertions throughout (never substring containment):
    # `"reduced" in detail` would be satisfied by "unreduced" too.
    # Line-87 verbatim → all three new classes, in dict order.
    assert f(_813_LINE_87) == ["unreduced", "reduced", "maps"]
    # Line-51 verbatim (the pre-filter killer at the check level; the
    # detector itself always saw it) → unreduced via `quota-held`.
    assert f(_813_LINE_51) == ["unreduced"]
    # New class x legacy denial.
    assert f("the fitted maps were not uploaded") == ["maps"]
    # `\breduced` cannot fire inside "unreduced ..." — the list equality pins
    # that `reduced` is ABSENT, not merely that `unreduced` is present.
    assert f("the unreduced store was not uploaded") == ["unreduced"]
    # Each new denial family firing ALONE (no legacy denial wording, no other
    # new family on the line):
    assert f("the unreduced store is quota-held") == ["unreduced"]
    assert f("the reduced summaries remain on the pod") == ["reduced"]
    assert f("the fitted maps are pending the quota hold") == ["maps"]
    assert f("the raw completions hit an upload 403") == ["raw_completions"]
    # Resolved narrative / past tense: NOT a denial (present-tense/stative
    # design of the quota-hold family).
    assert f("After the quota hold cleared, all raw completions were uploaded to HF.") == []
    assert (
        f(
            "the reduced summaries remained on the pod until the quota hold "
            "cleared and were then uploaded"
        )
        == []
    )


# --- #942: the #813 quota-hold regression battery ---------------------------


def test_813_regression_quota_hold_denial_fails(monkeypatch):
    """(#942 a) The #813 v1 line-87 sentence VERBATIM: the unreduced store /
    reduced summaries / fitted maps "remain on the pod under an HF
    public-storage quota hold (upload 403)" while all three artifact classes
    exist on HF under the body's pinned tree → FAIL naming all three
    canonical tokens."""
    monkeypatch.delenv("EPM_VERIFY_BODY_NO_HF", raising=False)
    _stub_tree(monkeypatch, entries=[_813_DIR, _UNREDUCED_FILE, _REDUCED_FILE, _MAPS_FILE])
    body = _body(denial=True, denial_line=_813_LINE_87, hf_url=_HF_813_TREE)
    _ok, results = verify_task_body.verify_text(body)
    by_name = _results_by_name(results)
    assert not by_name[_CHECK_NAME].passed
    detail = by_name[_CHECK_NAME].detail
    # Backtick-delimited token assertions: a bare `"reduced" in detail` would
    # be satisfied by "unreduced" too.
    for token in ("`unreduced`", "`reduced`", "`maps`"):
        assert f"body claims {token}" in detail
    assert "exists at" in detail


def test_813_regression_quota_held_activation_store_fails(monkeypatch):
    """(#942 a') The #813 v1 line-51 sentence VERBATIM — it contains NEITHER
    "uploaded" NOR "audit", so the pre-#942 inline pre-filter skipped the
    line before the denial regex ever ran. With the module-level
    `_AUDIT_LINE_PREFILTER_RE` ("quota" token) the line is scanned, the
    `quota-held` denial + "unreduced activation store" class fire, and the
    on-HF file makes it a FAIL. This test FAILS if the pre-filter fix is
    reverted to the old two-token inline form."""
    monkeypatch.delenv("EPM_VERIFY_BODY_NO_HF", raising=False)
    _stub_tree(monkeypatch, entries=[_813_DIR, _UNREDUCED_FILE])
    body = _body(denial=True, denial_line=_813_LINE_51, hf_url=_HF_813_TREE)
    _ok, results = verify_task_body.verify_text(body)
    by_name = _results_by_name(results)
    assert not by_name[_CHECK_NAME].passed
    assert "body claims `unreduced`" in by_name[_CHECK_NAME].detail
    assert "exists at" in by_name[_CHECK_NAME].detail


@pytest.mark.parametrize(
    ("denial_line", "hf_url", "file_entry", "dir_entry", "token"),
    [
        pytest.param(
            "The reduced per-question stores remain on the pod.",
            _HF_813_TREE,
            _REDUCED_FILE,
            _813_DIR,
            "reduced",
            id="remain-on-pod-x-reduced",
        ),
        pytest.param(
            "The fitted maps are pending the quota hold.",
            _HF_813_TREE,
            _MAPS_FILE,
            _813_DIR,
            "maps",
            id="pending-quota-hold-x-maps",
        ),
        pytest.param(
            "The analysis tensors are blocked on the same quota hold.",
            _HF_REPO_TREE,
            _ANALYSIS_TENSORS_FILE,
            _DIR_ENTRY,
            "analysis_tensors",
            id="blocked-on-quota-hold-x-legacy-class",
        ),
        pytest.param(
            "The raw completions are blocked by an upload 403.",
            _HF_REPO_TREE,
            _RAW_COMPLETIONS_FILE,
            _DIR_ENTRY,
            "raw_completions",
            id="upload-403-x-legacy-class",
        ),
    ],
)
def test_new_denial_families_true_positives(
    monkeypatch, denial_line, hf_url, file_entry, dir_entry, token
):
    """(#942 b) Each new denial family fires ALONE (no legacy denial wording
    on the line) x an artifact class — including new-denial x legacy-class
    pairings — and FAILs when the stubbed listing contains the file."""
    monkeypatch.delenv("EPM_VERIFY_BODY_NO_HF", raising=False)
    _stub_tree(monkeypatch, entries=[dir_entry, file_entry])
    body = _body(denial=True, denial_line=denial_line, hf_url=hf_url)
    _ok, results = verify_task_body.verify_text(body)
    by_name = _results_by_name(results)
    assert not by_name[_CHECK_NAME].passed
    assert f"body claims `{token}`" in by_name[_CHECK_NAME].detail
    assert "exists at" in by_name[_CHECK_NAME].detail


def test_honest_reduced_denial_with_only_unreduced_on_hf_passes(monkeypatch):
    """(#942 c) Boundary-match guard: an HONEST denial of the *reduced*
    stores while the listing holds ONLY `unreduced/` files must PASS as
    corroborated (no `unverified` note). A bare-substring keyword match would
    find `reduced` inside `unreduced/` and flip this into a false FAIL —
    this test FAILS if the alphanumeric-boundary fix in `_hf_probe_keyword`
    is reverted to `kw in path.lower()`."""
    monkeypatch.delenv("EPM_VERIFY_BODY_NO_HF", raising=False)
    _stub_tree(monkeypatch, entries=[_813_DIR, _UNREDUCED_FILE])
    body = _body(
        denial=True,
        denial_line="The reduced stores remain on the pod under a quota hold.",
        hf_url=_HF_813_TREE,
    )
    _ok, results = verify_task_body.verify_text(body)
    by_name = _results_by_name(results)
    assert by_name[_CHECK_NAME].passed
    # A definitive zero-match listing corroborates the denial — not "unverified".
    assert "unverified" not in by_name[_CHECK_NAME].detail


def test_honest_quota_hold_denial_genuinely_missing_passes(monkeypatch):
    """(#942 c) An honest quota-hold denial whose artifact class genuinely
    has no file on HF is CORROBORATED by the successful listing → PASS with
    no `unverified` note."""
    monkeypatch.delenv("EPM_VERIFY_BODY_NO_HF", raising=False)
    # The listing holds only unrelated files — no maps/ path anywhere.
    _stub_tree(monkeypatch, entries=[_813_DIR, _UNREDUCED_FILE, _README_FILE])
    body = _body(
        denial=True,
        denial_line=("The fitted maps remain on the pod under an HF public-storage quota hold."),
        hf_url=_HF_813_TREE,
    )
    _ok, results = verify_task_body.verify_text(body)
    by_name = _results_by_name(results)
    assert by_name[_CHECK_NAME].passed
    assert "unverified" not in by_name[_CHECK_NAME].detail


def test_resolved_quota_narrative_does_not_fire(monkeypatch):
    """(#942 c') A RESOLVED quota-hold narrative ("After the quota hold
    cleared, ... were uploaded") is NOT a denial — no stative preposition, no
    present-tense pod residency — so check 25 never reaches its keyword
    probe."""
    monkeypatch.delenv("EPM_VERIFY_BODY_NO_HF", raising=False)

    def _boom_keyword(repo_id, repo_type, sha, path_prefix, keyword):  # pragma: no cover
        raise AssertionError("no denial detected — the check-25 probe must not run")

    monkeypatch.setattr(verify_task_body, "_hf_probe_keyword", _boom_keyword)
    # Check 23 legitimately probes the body's HF URL with the fence delenv'd.
    _stub_tree(monkeypatch, entries=[_DIR_ENTRY, _RAW_COMPLETIONS_FILE])
    body = _body(
        denial=False,
        denial_line="After the quota hold cleared, all raw completions were uploaded to HF.",
    )
    _ok, results = verify_task_body.verify_text(body)
    by_name = _results_by_name(results)
    assert by_name[_CHECK_NAME].passed
    assert "no availability-denial claim" in by_name[_CHECK_NAME].detail


def test_bare_403_without_artifact_class_does_not_fire(monkeypatch):
    """(#942 c'') An `upload 403` mention with NO artifact class on the line
    yields no suspect keyword — the denial is not attributable to any
    probeable class, so the keyword probe never runs."""
    monkeypatch.delenv("EPM_VERIFY_BODY_NO_HF", raising=False)

    def _boom_keyword(repo_id, repo_type, sha, path_prefix, keyword):  # pragma: no cover
        raise AssertionError("no artifact class flagged — the check-25 probe must not run")

    monkeypatch.setattr(verify_task_body, "_hf_probe_keyword", _boom_keyword)
    _stub_tree(monkeypatch, entries=[_DIR_ENTRY, _RAW_COMPLETIONS_FILE])
    body = _body(
        denial=False,
        denial_line="The first upload attempt returned an upload 403; see the ops log.",
    )
    _ok, results = verify_task_body.verify_text(body)
    by_name = _results_by_name(results)
    assert by_name[_CHECK_NAME].passed
    assert "no availability-denial claim" in by_name[_CHECK_NAME].detail


def test_pod_mention_without_denial_does_not_fire(monkeypatch):
    """(#942 c'') A benign `pod` mention (a new pre-filter token) with an
    artifact class on the line but NO denial phrase does not fire."""
    monkeypatch.delenv("EPM_VERIFY_BODY_NO_HF", raising=False)

    def _boom_keyword(repo_id, repo_type, sha, path_prefix, keyword):  # pragma: no cover
        raise AssertionError("no denial phrase on the line — the check-25 probe must not run")

    monkeypatch.setattr(verify_task_body, "_hf_probe_keyword", _boom_keyword)
    _stub_tree(monkeypatch, entries=[_DIR_ENTRY, _RAW_COMPLETIONS_FILE])
    body = _body(
        denial=False,
        denial_line="The run used pod-813 for 43 h; raw completions are at the link below.",
    )
    _ok, results = verify_task_body.verify_text(body)
    by_name = _results_by_name(results)
    assert by_name[_CHECK_NAME].passed
    assert "no availability-denial claim" in by_name[_CHECK_NAME].detail


# --- #942: structural pins (pre-filter sync + boundary path-component match) -


def _split_top_level_alternations(pattern: str) -> list[str]:
    """Split a `(?:A|B|...)` regex PATTERN STRING on its TOP-LEVEL `|`s,
    tracking group depth and character classes (escape pairs are copied
    opaquely). Used to enumerate the `_AUDIT_DENIAL_RE` alternation families
    structurally, so a future alternation cannot be added without a curated
    sentence below."""
    assert pattern.startswith("(?:") and pattern.endswith(")"), pattern
    inner = pattern[3:-1]
    parts: list[str] = []
    buf: list[str] = []
    depth = 0
    in_class = False
    i = 0
    while i < len(inner):
        c = inner[i]
        if c == "\\":
            buf.append(inner[i : i + 2])
            i += 2
            continue
        if in_class:
            if c == "]":
                in_class = False
        elif c == "[":
            in_class = True
        elif c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
        elif c == "|" and depth == 0:
            parts.append("".join(buf))
            buf = []
            i += 1
            continue
        buf.append(c)
        i += 1
    parts.append("".join(buf))
    return parts


# One curated sentence per `_AUDIT_DENIAL_RE` alternation family, in the
# regex's own order. Adding an alternation without a sentence here (or vice
# versa) fails the length assertion loudly.
_DENIAL_FAMILY_SENTENCES = [
    "the raw completions were not separately uploaded",
    "the store was not uploaded",
    "the store wasn't uploaded",
    "the per-cell rates cannot be audited",
    "we cannot audit the per-cell rates",
    "the rates can't be audited",
    "the pools are unavailable for audit",
    "the stores remain on the pod",
    "the store is quota-held",
    "the maps are pending the quota hold",
    "the first attempt hit an upload 403",
]


def test_prefilter_covers_every_denial_family():
    """(#942) STRUCTURAL pre-filter/regex sync pin: split
    `_AUDIT_DENIAL_RE.pattern` on top-level `|`, pair each alternation family
    with a curated sentence, and assert the sentence matches (a) its OWN
    family sub-regex, (b) the full denial regex, and (c) the line pre-filter
    `_AUDIT_LINE_PREFILTER_RE` — so a future alternation added without
    pre-filter coverage (the pre-#942 line-51 skip bug: a denial family whose
    sentences the pre-filter drops is dead code) fails loud."""
    import re

    families = _split_top_level_alternations(verify_task_body._AUDIT_DENIAL_RE.pattern)
    assert len(families) == len(_DENIAL_FAMILY_SENTENCES), (
        f"{len(families)} denial alternation families vs "
        f"{len(_DENIAL_FAMILY_SENTENCES)} curated sentences — add exactly one "
        "sentence per alternation (and keep the pre-filter covering it)"
    )
    for fam, sentence in zip(families, _DENIAL_FAMILY_SENTENCES, strict=True):
        assert re.search(fam, sentence, re.IGNORECASE), (fam, sentence)
        assert verify_task_body._AUDIT_DENIAL_RE.search(sentence), sentence
        assert verify_task_body._AUDIT_LINE_PREFILTER_RE.search(sentence), (
            f"pre-filter misses denial-family sentence {sentence!r} — extend "
            "_AUDIT_LINE_PREFILTER_RE"
        )


@pytest.mark.parametrize(
    ("token", "path", "should_match"),
    [
        # Legacy tokens keep matching their real path shapes (`/` and `_` are
        # boundaries), including an underscore-joined filename.
        pytest.param(
            "raw_completions",
            "issue653_install-validated-reladder/raw_completions/armB/"
            "install_probes/cell0/raw_completions.json",
            True,
            id="raw_completions-dir",
        ),
        pytest.param(
            "install_probes",
            "issue653_install-validated-reladder/raw_completions/armB/"
            "install_probes/cell0/raw_completions.json",
            True,
            id="install_probes-nested-dir",
        ),
        pytest.param(
            "mixes",
            "issue653_install-validated-reladder/mixes/reladder.jsonl",
            True,
            id="mixes-dir",
        ),
        pytest.param(
            "mixes",
            "issue474_loc/i474_loc_mixes.jsonl",
            True,
            id="mixes-underscore-filename",
        ),
        pytest.param(
            "onpolicy_pools",
            "issue653_install-validated-reladder/onpolicy_pools/pool_armB.jsonl",
            True,
            id="onpolicy_pools-dir",
        ),
        pytest.param(
            "analysis_tensors",
            "issue653_install-validated-reladder/analysis_tensors/shift_tensor.npz",
            True,
            id="analysis_tensors-dir",
        ),
        # New #813 tokens.
        pytest.param(
            "unreduced",
            "issue813_mapchange_substrate/unreduced/em/generic/q0001.npz",
            True,
            id="unreduced-dir",
        ),
        pytest.param(
            "reduced",
            "issue813_mapchange_substrate/reduced/em/generic/summary.npz",
            True,
            id="reduced-dir",
        ),
        pytest.param(
            "maps",
            "issue813_mapchange_substrate/maps/em/generic_L14_map.npz",
            True,
            id="maps-dir",
        ),
        # The load-bearing negative: `reduced` must NOT match `unreduced/`.
        pytest.param(
            "reduced",
            "issue813_mapchange_substrate/unreduced/em/generic/q0001.npz",
            False,
            id="reduced-not-in-unreduced",
        ),
        # An adjacent letter/digit is not a boundary either way (singular
        # `map` in a filename does not satisfy the `maps` token).
        pytest.param(
            "maps",
            "issue813_mapchange_substrate/linear/em/generic_L14_map.npz",
            False,
            id="maps-not-in-singular-map",
        ),
    ],
)
def test_keyword_path_component_boundary_match(token, path, should_match):
    """(#942 d) The check-25 on-Hub keyword match is an alphanumeric-boundary
    path-component match (`_audit_keyword_path_re`), not a bare substring:
    every legacy token keeps matching its real path shape while `reduced` no
    longer matches `unreduced/` paths."""
    pat = verify_task_body._audit_keyword_path_re(token)
    assert bool(pat.search(path.lower())) is should_match
