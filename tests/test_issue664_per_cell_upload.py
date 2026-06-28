"""TDD round-1 tests for issue #689 fix (a) — per-cell incremental upload.

These describe the EXTERNAL behavior of the per-cell upload + EXACT-set Hub
presence helpers added to ``scripts/issue664_dispatch.py`` (plan v3 §A.3 / §A.4).
They are TESTS-FIRST: the round-1 commit ships the helpers as
``NotImplementedError`` stubs, so every case here FAILS until the round-2
implementation lands (the TDD gate). Each case asserts a documented contract:

  - exact-file-set idempotency (S1): prefix-presence alone is NOT "complete";
  - fresh upload -> exactly 2 ``upload_folder`` commits + a fail-loud post verify;
  - PARTIAL-on-Hub re-upload (S1): a cell missing one artifact-kind re-uploads;
  - fail-loud post-upload verify (RuntimeError names the missing file);
  - P3 exact-set safety-sweep skip + terminal M2 store Hub-verify;
  - A2 fresh-pod resume: local dirs absent + cell complete on HF -> done-anywhere
    True, no "NO raw completions" raise, M2 store verify passes;
  - smoke short-circuit: no listing, no upload, no HF consult.

All HF I/O is mocked (``huggingface_hub.list_repo_files`` /
``huggingface_hub.HfApi``); local eval/store dirs use ``tmp_path``. No GPU, no
network.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS = _REPO_ROOT / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import huggingface_hub  # noqa: E402
import issue664_common as C  # noqa: E402
import issue664_dispatch as D  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers: a representative non-marker cell (its eval-JSON set is non-empty),
# the EXACT expected file set on the Hub, and recording HF doubles.
# ---------------------------------------------------------------------------


def _a_cell() -> C.Cell:
    """A non-marker realized cell, so its expected eval-JSON set is non-empty.

    The marker column has no completions JSON (its DV is the slot stats), so a
    pure-marker cell would give an empty eval set and not exercise the eval-side
    of the EXACT-set check. Pick the first non-marker cell in the realized grid.
    """
    for c in C.realized_grid():
        if c.behavior != "marker":
            return c
    # Fallback: the grid is marker-only (should not happen) — use any cell.
    return C.realized_grid()[0]


def _expected_hub_paths(cell: C.Cell) -> set[str]:
    """The full set of repo-relative HF paths a COMPLETE cell occupies: every
    expected eval JSON under the raw prefix + tensors.pt + meta.json under the
    store prefix. Built from the SAME helpers the implementation uses, so the
    test stays in lock-step with whatever ``_judging_surface`` enumerates."""
    raw_prefix = f"{C.HF_RAW_COMPLETIONS_PREFIX}/{cell.eval_key}"
    store_prefix = f"{C.HF_STORE_PREFIX}/{cell.eval_key}"
    eval_files = {f"{raw_prefix}/{name}" for name in D._expected_eval_files(cell)}
    store_files = {f"{store_prefix}/{name}" for name in {"tensors.pt", "meta.json"}}
    return eval_files | store_files


class _ListRepoFilesStub:
    """A scripted ``list_repo_files`` returning a different file set per call.

    ``responses`` is a list of file-list snapshots; each call returns the next
    one (the last is repeated once exhausted). Records the call count so a test
    can assert the listing was/was not consulted.
    """

    def __init__(self, responses: list[set[str]]):
        self._responses = [sorted(r) for r in responses]
        self.calls = 0

    def __call__(self, repo_id, **kwargs):  # mirrors huggingface_hub.list_repo_files
        idx = min(self.calls, len(self._responses) - 1)
        self.calls += 1
        return list(self._responses[idx])


# Module-level recorder for upload_folder calls. A module-level list (reset by
# the autouse fixture) sidesteps the RUF012 mutable-class-attribute lint AND the
# `from __future__ import annotations` / F401 import-strip race that a
# ``ClassVar`` annotation would hit under the PostToolUse ruff hook.
_UPLOAD_FOLDER_CALLS: list[dict] = []


class _RecordingHfApi:
    """A fake ``HfApi`` whose ``upload_folder`` records each call (no network)."""

    def __init__(self, *a, **k):
        pass

    def upload_folder(self, **kwargs):
        _UPLOAD_FOLDER_CALLS.append(kwargs)
        return type("CommitInfo", (), {"oid": "deadbeef"})()


@pytest.fixture(autouse=True)
def _reset_hfapi_calls():
    _UPLOAD_FOLDER_CALLS.clear()
    yield
    _UPLOAD_FOLDER_CALLS.clear()


def _populate_local_cell(tmp_path: Path, cell: C.Cell, monkeypatch) -> None:
    """Create non-empty local eval-registry + store dirs for ``cell`` and point
    the dispatcher's roots at ``tmp_path`` so the upload reads them."""
    eval_dir = tmp_path / "eval" / "registry" / cell.eval_key
    eval_dir.mkdir(parents=True, exist_ok=True)
    for name in D._expected_eval_files(cell) or {"completions__placeholder.json"}:
        (eval_dir / name).write_text("[]")
    store_dir = tmp_path / "store" / cell.eval_key
    store_dir.mkdir(parents=True, exist_ok=True)
    (store_dir / "tensors.pt").write_bytes(b"\x00")
    (store_dir / "meta.json").write_text("{}")
    monkeypatch.setattr(C, "EVAL_ROOT", tmp_path / "eval", raising=False)
    monkeypatch.setattr(C, "STORE_ROOT", tmp_path / "store", raising=False)


# ---------------------------------------------------------------------------
# 1. per-cell idempotency — EXACT set already on Hub -> skip upload entirely
# ---------------------------------------------------------------------------


def test_per_cell_idempotency_exact_set(tmp_path, monkeypatch):
    cell = _a_cell()
    full = _expected_hub_paths(cell)
    lst = _ListRepoFilesStub([full])
    monkeypatch.setattr(huggingface_hub, "list_repo_files", lst)
    monkeypatch.setattr(huggingface_hub, "HfApi", _RecordingHfApi)
    _populate_local_cell(tmp_path, cell, monkeypatch)

    # The cell's EXACT eval+store sets are already on the Hub.
    assert D._cell_artifacts_on_hub(cell) is True
    # ... so _upload_cell_artifacts is a no-op: NO upload_folder calls.
    D._upload_cell_artifacts(cell, smoke=False)
    assert _UPLOAD_FOLDER_CALLS == []


# ---------------------------------------------------------------------------
# 2. fresh cell -> exactly 2 upload_folder commits, then verify passes
# ---------------------------------------------------------------------------


def test_fresh_cell_uploads_then_verifies(tmp_path, monkeypatch):
    cell = _a_cell()
    full = _expected_hub_paths(cell)
    # First listing (idempotency pre-check): empty. Subsequent listings (the
    # post-upload verify): the full expected set is present.
    lst = _ListRepoFilesStub([set(), full])
    monkeypatch.setattr(huggingface_hub, "list_repo_files", lst)
    monkeypatch.setattr(huggingface_hub, "HfApi", _RecordingHfApi)
    _populate_local_cell(tmp_path, cell, monkeypatch)

    D._upload_cell_artifacts(cell, smoke=False)
    # Exactly TWO upload_folder commits: eval JSONs + store tensors.
    assert len(_UPLOAD_FOLDER_CALLS) == 2
    # The post-upload EXACT-set Hub-verify consulted the listing again (>=2 calls).
    assert lst.calls >= 2


# ---------------------------------------------------------------------------
# 3. PARTIAL on Hub (S1) — store prefix present but missing tensors.pt
# ---------------------------------------------------------------------------


def test_partial_on_hub_re_uploads(tmp_path, monkeypatch):
    cell = _a_cell()
    full = _expected_hub_paths(cell)
    store_prefix = f"{C.HF_STORE_PREFIX}/{cell.eval_key}"
    # PARTIAL: raw prefix fully present, store prefix has ONLY meta.json (no
    # tensors.pt) — prefix-presence alone must NOT count as complete.
    partial = {p for p in full if not p.startswith(store_prefix)} | {f"{store_prefix}/meta.json"}
    # Idempotency check reads PARTIAL; the post-upload verify STILL reads PARTIAL
    # (tensors.pt never lands) -> the helper must fail loud.
    lst = _ListRepoFilesStub([partial, partial])
    monkeypatch.setattr(huggingface_hub, "list_repo_files", lst)
    monkeypatch.setattr(huggingface_hub, "HfApi", _RecordingHfApi)
    _populate_local_cell(tmp_path, cell, monkeypatch)

    # A partial cell reads NOT complete (exact-set, not prefix-presence).
    assert D._cell_artifacts_on_hub(cell) is False
    # It re-uploads; the post-upload listing still lacks tensors.pt -> RuntimeError.
    with pytest.raises(RuntimeError):
        D._upload_cell_artifacts(cell, smoke=False)


def test_partial_on_hub_eval_missing_file_re_uploads(tmp_path, monkeypatch):
    cell = _a_cell()
    eval_files = sorted(D._expected_eval_files(cell))
    if not eval_files:
        pytest.skip("cell has no expected eval JSONs to drop (marker-only)")
    full = _expected_hub_paths(cell)
    raw_prefix = f"{C.HF_RAW_COMPLETIONS_PREFIX}/{cell.eval_key}"
    dropped = f"{raw_prefix}/{eval_files[0]}"
    # PARTIAL: store prefix fully present, raw prefix present BUT missing one
    # expected completions JSON.
    partial = full - {dropped}
    lst = _ListRepoFilesStub([partial, partial])
    monkeypatch.setattr(huggingface_hub, "list_repo_files", lst)
    monkeypatch.setattr(huggingface_hub, "HfApi", _RecordingHfApi)
    _populate_local_cell(tmp_path, cell, monkeypatch)

    assert D._cell_artifacts_on_hub(cell) is False
    with pytest.raises(RuntimeError):
        D._upload_cell_artifacts(cell, smoke=False)


# ---------------------------------------------------------------------------
# 4. fail-loud verify — post-upload listing lacks an expected file
# ---------------------------------------------------------------------------


def test_fail_loud_post_upload_missing(tmp_path, monkeypatch):
    cell = _a_cell()
    full = _expected_hub_paths(cell)
    store_prefix = f"{C.HF_STORE_PREFIX}/{cell.eval_key}"
    # Pre-check empty -> uploads fire; post-upload listing is STILL missing the
    # store tensors.pt (a transient/throttled upload that didn't land).
    after = {p for p in full if not p.startswith(f"{store_prefix}/tensors.pt")}
    lst = _ListRepoFilesStub([set(), after])
    monkeypatch.setattr(huggingface_hub, "list_repo_files", lst)
    monkeypatch.setattr(huggingface_hub, "HfApi", _RecordingHfApi)
    _populate_local_cell(tmp_path, cell, monkeypatch)

    with pytest.raises(RuntimeError) as ei:
        D._upload_cell_artifacts(cell, smoke=False)
    # The message names the missing artifact so the failure is diagnosable.
    assert "tensors.pt" in str(ei.value) or cell.eval_key in str(ei.value)


# ---------------------------------------------------------------------------
# 5. P3 exact-set safety sweep — a cell complete on Hub is skipped, terminal
#    M2 store Hub-verify still passes.
# ---------------------------------------------------------------------------


def test_p3_exact_set_safety_sweep_skip(tmp_path, monkeypatch):
    cell = _a_cell()
    cells = [cell]
    full = _expected_hub_paths(cell)
    lst = _ListRepoFilesStub([full])
    monkeypatch.setattr(huggingface_hub, "list_repo_files", lst)
    monkeypatch.setattr(huggingface_hub, "HfApi", _RecordingHfApi)
    # No local files at all: the cell lives only on HF (already complete).
    monkeypatch.setattr(C, "EVAL_ROOT", tmp_path / "eval", raising=False)
    monkeypatch.setattr(C, "STORE_ROOT", tmp_path / "store", raising=False)
    (tmp_path / "eval").mkdir(parents=True, exist_ok=True)
    (tmp_path / "store").mkdir(parents=True, exist_ok=True)

    # The exact-set classifier sees the cell complete on HF.
    on_hub = set(lst([C.HF_DATA_REPO]))
    assert D._classify_cell_hub_state(cell, on_hub) == "complete"

    # P3 raw-completions: with every selected cell complete on HF and no local
    # files, it is a no-op (no RuntimeError, no re-upload).
    D._upload_raw_completions(cells)
    # P3 store-tensors: the M2 terminal Hub-verify passes (cell complete on HF);
    # it does not list the cell as missing.
    D._upload_store_tensors(cells)
    # No new upload_folder commits — everything was already on the Hub.
    assert _UPLOAD_FOLDER_CALLS == []


# ---------------------------------------------------------------------------
# 6. A2 fresh-pod resume — local dirs absent, cell complete on HF.
# ---------------------------------------------------------------------------


def test_a2_fresh_pod_resume(tmp_path, monkeypatch):
    cell = _a_cell()
    cells = [cell]
    full = _expected_hub_paths(cell)
    lst = _ListRepoFilesStub([full])
    monkeypatch.setattr(huggingface_hub, "list_repo_files", lst)
    monkeypatch.setattr(huggingface_hub, "HfApi", _RecordingHfApi)
    # Fresh pod: NO local cell dirs at all.
    monkeypatch.setattr(C, "EVAL_ROOT", tmp_path / "eval", raising=False)
    monkeypatch.setattr(C, "STORE_ROOT", tmp_path / "store", raising=False)
    (tmp_path / "eval").mkdir(parents=True, exist_ok=True)
    (tmp_path / "store").mkdir(parents=True, exist_ok=True)

    # The cell is "done anywhere" via its HF copy (local absent).
    assert D._cell_done_anywhere(cell, smoke=False) is True

    # _upload_raw_completions must NOT raise "NO raw completions" — it returns the
    # all-on-HF no-op.
    D._upload_raw_completions(cells)

    # _upload_store_tensors does not list the cell as missing and the M2 Hub
    # verify passes (no RuntimeError).
    D._upload_store_tensors(cells)


# ---------------------------------------------------------------------------
# 7. smoke short-circuit — no listing, no upload, no HF consult
# ---------------------------------------------------------------------------


def test_smoke_mode_short_circuits(tmp_path, monkeypatch):
    cell = _a_cell()
    lst = _ListRepoFilesStub([set()])
    monkeypatch.setattr(huggingface_hub, "list_repo_files", lst)
    monkeypatch.setattr(huggingface_hub, "HfApi", _RecordingHfApi)
    monkeypatch.setattr(C, "EVAL_ROOT", tmp_path / "eval", raising=False)
    monkeypatch.setattr(C, "STORE_ROOT", tmp_path / "store", raising=False)

    # smoke=True: the per-cell upload short-circuits before any HF call.
    D._upload_cell_artifacts(cell, smoke=True)
    assert _UPLOAD_FOLDER_CALLS == []
    assert lst.calls == 0

    # _cell_done_anywhere never consults HF in smoke (per-cell upload is skipped),
    # so the local-only predicate decides; with no local artifacts -> not done.
    assert D._cell_done_anywhere(cell, smoke=True) is False
    assert lst.calls == 0  # still no HF listing


# ---------------------------------------------------------------------------
# 8. #689 blocker-1 (fix a1): fresh-pod marker cells stay READABLE — the
#    marker-slot stats are part of the per-cell HF surface + hydrated before
#    _marker_readability_assert, so a fresh auto-migrated pod does NOT crash.
# ---------------------------------------------------------------------------


def _a_marker_cell() -> C.Cell:
    """The first MARKER cell in the realized grid (writes marker_slot_stats.json)."""
    for c in C.realized_grid():
        if c.behavior == "marker":
            return c
    pytest.skip("no marker cell in the realized grid")


def _marker_hub_paths(cell: C.Cell) -> set[str]:
    """The full COMPLETE-on-HF path set for a MARKER cell: eval JSONs (may be
    empty for the marker column) + store tensors + the marker-slot stats JSON."""
    raw_prefix = f"{C.HF_RAW_COMPLETIONS_PREFIX}/{cell.eval_key}"
    store_prefix = f"{C.HF_STORE_PREFIX}/{cell.eval_key}"
    marker_prefix = f"{C.HF_MARKER_SLOT_PREFIX}/{cell.eval_key}"
    eval_files = {f"{raw_prefix}/{name}" for name in D._expected_eval_files(cell)}
    store_files = {f"{store_prefix}/{name}" for name in {"tensors.pt", "meta.json"}}
    marker_files = {f"{marker_prefix}/marker_slot_stats.json"}
    return eval_files | store_files | marker_files


def test_marker_slot_stats_in_complete_surface(tmp_path, monkeypatch):
    """#689 blocker-1: a marker cell is NOT 'complete' on HF unless its
    marker_slot_stats.json is present (it is now part of the per-cell HF surface,
    so the fresh-pod SKIP-and-hydrate path is coherent). A non-marker cell is
    unaffected (the marker requirement is vacuous)."""
    mcell = _a_marker_cell()
    full = _marker_hub_paths(mcell)
    marker_path = f"{C.HF_MARKER_SLOT_PREFIX}/{mcell.eval_key}/marker_slot_stats.json"
    # WITH the marker-slot stats -> complete.
    assert D._classify_cell_hub_state(mcell, full) == "complete"
    # WITHOUT the marker-slot stats -> NOT complete (partial: other kinds present).
    assert D._classify_cell_hub_state(mcell, full - {marker_path}) == "partial"
    # A non-marker cell does not require marker-slot stats.
    ncell = _a_cell()
    assert D._expected_marker_slot_files(ncell) == set()


def test_a2_fresh_pod_marker_cells_readable(tmp_path, monkeypatch):
    """#689 blocker-1 (the headline regression): on a FRESH auto-migrated pod the
    local marker_slot/ dir is ABSENT, but the marker cell's full expected set —
    including marker_slot_stats.json — IS on HF. _marker_readability_assert must
    HYDRATE the stats from HF and NOT raise (checked > 0): the prior code crashed
    at `checked == 0` because P2 SKIPped the HF-complete marker cell so there was
    no local file to read."""
    import json as _json

    mcell = _a_marker_cell()
    full = _marker_hub_paths(mcell)
    lst = _ListRepoFilesStub([full])
    monkeypatch.setattr(huggingface_hub, "list_repo_files", lst)
    # Fresh pod: NO local marker_slot dir.
    monkeypatch.setattr(C, "EVAL_ROOT", tmp_path / "eval", raising=False)
    (tmp_path / "eval").mkdir(parents=True, exist_ok=True)

    # A clean readability payload (emission < 1%, z_marker < z_eos on all slots),
    # written by the hydrate's hf_hub_download seam into a scratch file the helper
    # then copies into the local marker_slot path.
    clean_payload = {
        "cell": mcell.eval_key,
        "slots": {
            "ctx0": {
                "trained": {"argmax_id": C.MARKER_ID + 1, "z_marker": 1.0, "z_eos": 3.0},
                "base": {"argmax_id": C.MARKER_ID + 1, "z_marker": 0.5, "z_eos": 3.0},
            }
        },
    }
    dl_src = tmp_path / "downloaded_marker_slot_stats.json"
    dl_src.write_text(_json.dumps(clean_payload))

    def _fake_download(*, repo_id, repo_type, filename, revision):
        assert filename.endswith("marker_slot_stats.json")
        return str(dl_src)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", _fake_download, raising=False)

    # Sanity: the local file does NOT exist before the assert runs.
    local = D._marker_slot_local_path(mcell, smoke=False)
    assert not local.exists()

    # The readability assert hydrates from HF and reads checked > 0 -> NO raise.
    D._marker_readability_assert([mcell], smoke=False)
    # The hydrate landed the stats locally.
    assert local.exists()


# ---------------------------------------------------------------------------
# 9. #689 blocker-4 (p3-raw-exact-verify): the terminal P3 raw verify is an
#    EXACT FILE-SET check per selected cell, NOT a count floor — an incomplete
#    selected cell is rejected even when unrelated prefix files inflate the count.
# ---------------------------------------------------------------------------


def test_p3_raw_terminal_verify_rejects_incomplete_selected_cell(tmp_path, monkeypatch):
    """#689 blocker-4: one selected cell is missing one expected completions JSON
    on the Hub, but UNRELATED files under the raw-completions prefix make the total
    count high (a count floor would PASS). The exact-set terminal verify must raise
    RuntimeError naming the missing cell + file."""
    cell = _a_cell()
    eval_files = sorted(D._expected_eval_files(cell))
    if not eval_files:
        pytest.skip("cell has no expected eval JSONs (marker-only)")
    cells = [cell]
    raw_prefix = f"{C.HF_RAW_COMPLETIONS_PREFIX}/{cell.eval_key}"
    dropped = f"{raw_prefix}/{eval_files[0]}"
    full = _expected_hub_paths(cell)
    # On HF: the cell's full set MINUS one completions JSON, PLUS many UNRELATED
    # files under the same top-level prefix so a count floor would be satisfied.
    unrelated = {
        f"{C.HF_RAW_COMPLETIONS_PREFIX}/some_other_cell_{i}/completions__x__c.json"
        for i in range(20)
    }
    on_hub = (full - {dropped}) | unrelated

    # The local registry has the FULL set on disk (so the upload loop runs for the
    # not-complete cell), but the post-upload HF listing is missing the one file.
    eval_dir = tmp_path / "eval" / "registry" / cell.eval_key
    eval_dir.mkdir(parents=True, exist_ok=True)
    for name in eval_files:
        (eval_dir / name).write_text("[]")
    monkeypatch.setattr(C, "EVAL_ROOT", tmp_path / "eval", raising=False)
    monkeypatch.setattr(C, "STORE_ROOT", tmp_path / "store", raising=False)
    (tmp_path / "store").mkdir(parents=True, exist_ok=True)

    # list_repo_files always returns the (incomplete-for-this-cell) on_hub set; the
    # cell is NOT classified complete (missing one eval file) so it is NOT skipped.
    monkeypatch.setattr(huggingface_hub, "list_repo_files", lambda repo_id, **k: sorted(on_hub))
    # hub._upload is a no-op (the file never actually lands on the Hub listing).
    from explore_persona_space.orchestrate import hub

    monkeypatch.setattr(hub, "_upload", lambda *a, **k: None, raising=False)

    assert D._classify_cell_hub_state(cell, on_hub) != "complete"
    with pytest.raises(RuntimeError, match="EXACT-set verify FAILED"):
        D._upload_raw_completions(cells)
