"""Tests for the #2578 additive HF data-repo search set in verify_uploads.py.

#2389 declared a per-issue direct-write repo via ``EPM_2389_DATA_WRITE_REPO``
and the upload gate could not verify it: ``verify_uploads.py`` hardcoded
``HF_DATA_REPO`` as the SOLE search location for the residue (check 9),
row-index (check 10), and ``--hf-dataset`` (check 5) arms. These tests pin
the fix: an ordered, deduped, ADDITIVE search set (default repo ALWAYS
retained; env ``EPM_<N>_DATA_WRITE_REPO``; repeatable comma-splittable
``--hf-data-repo``; pointer-discovered overflow repos) threaded into every
data-repo consumer, plus the ``--hf-model-repo`` single override on the
model side (a reproducibility-card ``hf_model_repo`` still wins).

The load-bearing pins (plan #2578 §6):

- T6/T15/T16 are the FAIL-CLOSED pins: a file in NO searched repo stays
  residue, and the realized listing-call product is EXACTLY the requested
  (prefix x repo) grid — T15 asserts BOTH the verdict (kills the
  replacement mutation) AND the call product read off the autospec mock
  (kills the additive mutation: an extra broadened call would union ~1M
  whole-repo basenames into the covered set in production while every
  outcome assert stays green).
- T10 pins singleton byte-parity: the no-flag path carries NO per-repo
  coverage annotation (pre-#2578 detail strings unchanged).
- T7 pins the repo-annotated fail-loud wrap (round-1 F11): an unknown repo
  ERRORs naming the repo explicitly, never an empty-listing OK.

Per the one-production-body-test rule (#906), T5-T10 and T15/T16 execute the
REAL ``check_outroot_residue`` body (real tmp out-root, real git subprocess
arm) and T12 executes the REAL row-index seam bodies; fakes sit only at the
external HF network boundary and are signature-conformant by construction
(``create_autospec`` of the real hub helpers / signature-mirroring
``HfApi`` fakes). Module-loading conventions follow
tests/test_verify_uploads_outroot_residue.py, with ONE deliberate
divergence: the listing fake here keys on ``(repo_id, path_in_repo)`` —
repo-differentiated resolution is what makes T5/T12/T15 expressible.
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import create_autospec

import pytest

# Load the verifier as a module (it's a script, not a package member).
_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "verify_uploads.py"
_spec = importlib.util.spec_from_file_location("verify_uploads_ro", _SCRIPT)
verify_uploads = importlib.util.module_from_spec(_spec)  # type: ignore[arg-type]
sys.modules["verify_uploads_ro"] = verify_uploads
_spec.loader.exec_module(verify_uploads)  # type: ignore[union-attr]

DEFAULT_DATA = verify_uploads.HF_DATA_REPO
DEFAULT_MODEL = verify_uploads.HF_MODEL_REPO
OVF = "org/ovf"
KEY = ("context_id", "rollout_k")

# Capture the REAL hub helpers ONCE at module load: tests re-patch the
# boundary several times within one test, and autospec-ing an
# already-patched attribute raises InvalidSpecError ("Cannot spec a Mock").
from explore_persona_space.orchestrate import hub as _hub  # noqa: E402

_REAL_LIST_REPO_FILES = _hub.list_repo_files_complete
_REAL_STAGE_HUB_FILE = _hub.stage_hub_file


def _row(context_id: str, rollout_k: int) -> str:
    return json.dumps({"context_id": context_id, "rollout_k": rollout_k})


@pytest.fixture(autouse=True)
def _clean_write_repo_env(monkeypatch):
    """Ambient-env hygiene: no EPM_<N>_DATA_WRITE_REPO leaks into any test."""
    for key in [k for k in os.environ if re.fullmatch(r"EPM_\d+_DATA_WRITE_REPO", k)]:
        monkeypatch.delenv(key, raising=False)


def _patch_hf_multi(monkeypatch, mapping: dict[tuple[str, str | None], object]):
    """Fake the HF listing boundary, keyed on ``(repo_id, path_in_repo)``.

    ``create_autospec`` of the real ``list_repo_files_complete`` rejects any
    call whose shape drifts from the real signature (#906 rule). Mapping
    values: a list of full paths (the listing), or an Exception instance to
    raise; an UNMAPPED pair raises ``EntryNotFoundError`` exactly as the
    tree endpoint 404s — so any broadened/substituted call a mutation might
    issue is swallowed by the production per-pair ``continue`` unless the
    test maps it, which is why T15 additionally asserts the call product.
    """
    from explore_persona_space.orchestrate import hub

    fake = create_autospec(_REAL_LIST_REPO_FILES)

    def _lookup(api, repo_id, *, repo_type="model", revision=None, path_in_repo=None):
        from huggingface_hub.utils import EntryNotFoundError

        value = mapping.get((repo_id, path_in_repo))
        if isinstance(value, Exception):
            raise value
        if value is None:
            raise EntryNotFoundError(f"no tree at {path_in_repo}")
        return list(value)

    fake.side_effect = _lookup
    monkeypatch.setattr(hub, "list_repo_files_complete", fake)
    return fake


def _patch_pointer_fetch(
    monkeypatch, payload_bytes: bytes | None = None, exc: Exception | None = None
):
    """Fake the pointer-fetch boundary (``hub.stage_hub_file``), autospec'd."""
    from explore_persona_space.orchestrate import hub

    fake = create_autospec(_REAL_STAGE_HUB_FILE)

    def _stage(
        repo_id,
        path_in_repo,
        target,
        *,
        repo_type="dataset",
        revision=None,
        token=None,
        overwrite=False,
        size_bytes=None,
    ):
        if exc is not None:
            raise exc
        target = Path(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload_bytes or b"")
        return target

    fake.side_effect = _stage
    monkeypatch.setattr(hub, "stage_hub_file", fake)
    return fake


# ---------------------------------------------------------------------------
# T1-T4: the resolver (ordered, deduped, additive; fail-loud shapes)
# ---------------------------------------------------------------------------


def test_resolver_default_only():
    """T1: clean env + no flags -> exactly the default entry (parity)."""
    assert verify_uploads.resolve_data_repos(2578, ()) == [
        {"repo": DEFAULT_DATA, "source": "default"}
    ]


def test_resolver_env_flag_union_order_dedup(monkeypatch):
    """T2: default first, env second, flags in order; comma-split; dedup by
    repo id with FIRST source winning (the env repo re-supplied via flag
    keeps its env source)."""
    monkeypatch.setenv("EPM_2578_DATA_WRITE_REPO", "org/env-repo")
    entries = verify_uploads.resolve_data_repos(
        2578, ("org/flag-a,org/flag-b", "org/env-repo", "org/flag-a")
    )
    assert entries == [
        {"repo": DEFAULT_DATA, "source": "default"},
        {"repo": "org/env-repo", "source": "env:EPM_2578_DATA_WRITE_REPO"},
        {"repo": "org/flag-a", "source": "flag"},
        {"repo": "org/flag-b", "source": "flag"},
    ]


def test_resolver_env_equal_to_default_noop(monkeypatch):
    """T3: EPM_<N>_DATA_WRITE_REPO naming the default -> singleton set
    (the #2389 canary case where the env repo IS canonical: parity)."""
    monkeypatch.setenv("EPM_2578_DATA_WRITE_REPO", DEFAULT_DATA)
    assert verify_uploads.resolve_data_repos(2578, ()) == [
        {"repo": DEFAULT_DATA, "source": "default"}
    ]


@pytest.mark.parametrize("bad", ["no-slash", "org/", "/name", "a/b/c", "", "   "])
def test_malformed_repo_id_fails_loud(bad):
    """T4 (resolver arm): a malformed repo id raises ValueError — an HF
    PREFIX passed where a repo id belongs must never reach a Hub read."""
    with pytest.raises(ValueError, match="malformed HF repo id"):
        verify_uploads.resolve_data_repos(2578, (bad,))


def test_malformed_env_repo_id_fails_loud(monkeypatch):
    """T4 (env arm): a malformed EPM_<N>_DATA_WRITE_REPO raises too."""
    monkeypatch.setenv("EPM_2578_DATA_WRITE_REPO", "issue2389_q38ce")
    with pytest.raises(ValueError, match="malformed HF repo id"):
        verify_uploads.resolve_data_repos(2578, ())


@pytest.mark.parametrize(
    "argv_tail",
    [
        ["--hf-data-repo", "no-slash"],
        ["--hf-data-repo", "org/good,still-bad"],
        ["--hf-model-repo", "no-slash"],
    ],
)
def test_cli_rejects_malformed_repo_id(monkeypatch, capsys, argv_tail):
    """T4 (CLI arm): argparse rejects a malformed repo id at parse time with
    exit code 2, before any network call."""
    monkeypatch.setattr(sys, "argv", ["verify_uploads.py", "--issue", "2578", *argv_tail])
    with pytest.raises(SystemExit) as excinfo:
        verify_uploads.main()
    assert excinfo.value.code == 2
    assert "malformed HF repo id" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# T5-T7, T16: the residue check under the union (real body; HF faked)
# ---------------------------------------------------------------------------


def test_residue_union_covers_second_repo(tmp_path, monkeypatch):
    """T5: a file present ONLY under the second searched repo's prefix is
    covered by the union — while the singleton default set (a second run)
    still flags it, proving the coverage came from repo B."""
    (tmp_path / "only_on_b.json").write_text("{}")
    prefix = "issue999999_x"
    _patch_hf_multi(monkeypatch, {(OVF, prefix): [f"{prefix}/sub/only_on_b.json"]})

    union_row = verify_uploads.check_outroot_residue(
        999999,
        outroot=str(tmp_path),
        hf_prefixes=(prefix,),
        data_repos=(DEFAULT_DATA, OVF),
    )
    assert union_row["status"] == "OK"
    assert f"HF files per searched repo: {DEFAULT_DATA}=0, {OVF}=1" in union_row["detail"]

    singleton_row = verify_uploads.check_outroot_residue(
        999999,
        outroot=str(tmp_path),
        hf_prefixes=(prefix,),
        data_repos=(DEFAULT_DATA,),
    )
    assert singleton_row["status"] == "FAIL"
    assert "only_on_b.json" in singleton_row["detail"]


def test_residue_fail_closed_file_in_no_repo(tmp_path, monkeypatch):
    """T6 (THE fail-closed pin): under a 2-repo union, a disk file matching
    NEITHER repo's listings is residue -> FAIL naming it."""
    (tmp_path / "lost.json").write_text("{}")
    prefix = "issue999999_x"
    _patch_hf_multi(
        monkeypatch,
        {
            (DEFAULT_DATA, prefix): [f"{prefix}/a.json"],
            (OVF, prefix): [f"{prefix}/b.json"],
        },
    )
    row = verify_uploads.check_outroot_residue(
        999999,
        outroot=str(tmp_path),
        hf_prefixes=(prefix,),
        data_repos=(DEFAULT_DATA, OVF),
    )
    assert row["status"] == "FAIL"
    assert "lost.json" in row["detail"]


def test_residue_unknown_repo_errors(tmp_path, monkeypatch):
    """T7: a repo whose scoped listing raises RepositoryNotFoundError (well-
    shaped but nonexistent / private-without-token id) is an ERROR row whose
    detail names the repo EXPLICITLY via the repo-annotated re-raise (round-1
    F11 — the hub exception's own message deliberately does NOT carry the
    repo id here), never an empty-listing OK; the overall verdict is FAIL."""
    from huggingface_hub.utils import RepositoryNotFoundError

    (tmp_path / "whatever.json").write_text("{}")
    prefix = "issue999999_x"
    _patch_hf_multi(
        monkeypatch,
        {
            (DEFAULT_DATA, prefix): [f"{prefix}/whatever.json"],
            (OVF, prefix): RepositoryNotFoundError("nope"),  # message omits the repo id
        },
    )
    row = verify_uploads.check_outroot_residue(
        999999,
        outroot=str(tmp_path),
        hf_prefixes=(prefix,),
        data_repos=(DEFAULT_DATA, OVF),
    )
    assert row["status"] == "ERROR"
    assert OVF in row["detail"], "the wrap must name the raising repo explicitly"
    assert "RepositoryNotFoundError" in row["detail"]

    report = verify_uploads.run_verification(
        999999,
        experiment_type="analysis",
        outroot=str(tmp_path),
        hf_prefixes=(prefix,),
        hf_data_repos=(OVF,),
    )
    assert report["checks"]["outroot_residue"]["status"] == "ERROR"
    assert report["verdict"] == "FAIL"


def test_residue_empty_prefix_all_repos_fail_not_error(tmp_path, monkeypatch):
    """T16 (round-1 F16): a prefix absent on ALL searched repos leaves every
    disk file as residue -> FAIL, never an ERROR row (the per-(prefix x repo)
    EntryNotFoundError `continue` is fail-toward-FAIL, not fail-loud)."""
    (tmp_path / "orphan.json").write_text("{}")
    _patch_hf_multi(monkeypatch, {})  # everything raises EntryNotFoundError
    row = verify_uploads.check_outroot_residue(
        999999,
        outroot=str(tmp_path),
        hf_prefixes=("issue999999_nowhere",),
        data_repos=(DEFAULT_DATA, OVF),
    )
    assert row["status"] == "FAIL"
    assert "orphan.json" in row["detail"]


# ---------------------------------------------------------------------------
# T8-T9: pointer discovery (real body; listing + fetch boundaries faked)
# ---------------------------------------------------------------------------


def test_pointer_discovery_end_to_end(tmp_path, monkeypatch):
    """T8: an OVERFLOW_POINTER.json in the canonical listing extends the
    union — coverage comes from the discovered repo, the row carries
    discovered_data_repos (source pointer:<prefix>, payload ts — round-1
    F10), run_verification folds it into report["hf_data_repos"], and the
    pointer basename itself is NEVER a covered name."""
    prefix = "issue999999_x"
    outroot = tmp_path / "out"
    outroot.mkdir()
    (outroot / "homed.json").write_text("{}")
    (outroot / "rerouted.json").write_text("{}")
    mapping = {
        (DEFAULT_DATA, prefix): [
            f"{prefix}/OVERFLOW_POINTER.json",
            f"{prefix}/homed.json",
        ],
        (OVF, prefix): [f"{prefix}/rerouted.json"],
    }
    _patch_hf_multi(monkeypatch, mapping)
    _patch_pointer_fetch(
        monkeypatch, payload_bytes=json.dumps({"overflow_repo": OVF, "ts": 1723456789.0}).encode()
    )

    row = verify_uploads.check_outroot_residue(
        999999, outroot=str(outroot), hf_prefixes=(prefix,), data_repos=(DEFAULT_DATA,)
    )
    assert row["status"] == "OK"
    assert row["discovered_data_repos"] == [
        {"repo": OVF, "source": f"pointer:{prefix}", "ts": 1723456789.0}
    ]
    assert f"HF files per searched repo: {DEFAULT_DATA}=1, {OVF}=1" in row["detail"]

    # run_verification folds the discovered repo into the search set (9-bis).
    report = verify_uploads.run_verification(
        999999, experiment_type="analysis", outroot=str(outroot), hf_prefixes=(prefix,)
    )
    assert report["hf_data_repos"] == [
        {"repo": DEFAULT_DATA, "source": "default"},
        {"repo": OVF, "source": f"pointer:{prefix}", "ts": 1723456789.0},
    ]

    # The pointer basename is a routing record, never a covered name: a disk
    # file named OVERFLOW_POINTER.json stays residue.
    outroot_b = tmp_path / "out_b"
    outroot_b.mkdir()
    (outroot_b / "OVERFLOW_POINTER.json").write_text("{}")
    row_b = verify_uploads.check_outroot_residue(
        999999, outroot=str(outroot_b), hf_prefixes=(prefix,), data_repos=(DEFAULT_DATA,)
    )
    assert row_b["status"] == "FAIL"
    assert "OVERFLOW_POINTER.json" in row_b["detail"]


def test_pointer_malformed_or_fetch_failure_errors(tmp_path, monkeypatch):
    """T9 (round-1 F18): a junk pointer payload, a payload with no string
    overflow_repo, or a raising fetch is an ERROR row (never 'skip discovery
    and under-cover', never 'cover without listing') and the overall verdict
    is FAIL."""
    prefix = "issue999999_x"
    (tmp_path / "homed.json").write_text("{}")
    mapping = {(DEFAULT_DATA, prefix): [f"{prefix}/OVERFLOW_POINTER.json", f"{prefix}/homed.json"]}

    # (a) unparseable payload -> ERROR, and the full-report verdict is FAIL.
    _patch_hf_multi(monkeypatch, mapping)
    _patch_pointer_fetch(monkeypatch, payload_bytes=b"not json{{{")
    report = verify_uploads.run_verification(
        999999, experiment_type="analysis", outroot=str(tmp_path), hf_prefixes=(prefix,)
    )
    row = report["checks"]["outroot_residue"]
    assert row["status"] == "ERROR"
    assert "overflow-pointer fetch/parse failed" in row["detail"]
    assert report["verdict"] == "FAIL"

    # (b) payload without a string overflow_repo -> ERROR.
    _patch_pointer_fetch(monkeypatch, payload_bytes=json.dumps({"overflow_repo": 7}).encode())
    row = verify_uploads.check_outroot_residue(
        999999, outroot=str(tmp_path), hf_prefixes=(prefix,), data_repos=(DEFAULT_DATA,)
    )
    assert row["status"] == "ERROR"
    assert "non-string overflow_repo" in row["detail"]

    # (c) raising fetch -> ERROR.
    _patch_pointer_fetch(monkeypatch, exc=RuntimeError("boom"))
    row = verify_uploads.check_outroot_residue(
        999999, outroot=str(tmp_path), hf_prefixes=(prefix,), data_repos=(DEFAULT_DATA,)
    )
    assert row["status"] == "ERROR"
    assert "overflow-pointer fetch/parse failed" in row["detail"]


# ---------------------------------------------------------------------------
# T10-T11: singleton byte-parity + report surfacing
# ---------------------------------------------------------------------------


def test_singleton_detail_annotation_absent(tmp_path, monkeypatch):
    """T10: the no-flag singleton path carries NO per-repo coverage
    annotation and no discovered_data_repos key — byte-parity with the
    pre-#2578 detail strings (which the sibling test file pins verbatim)."""
    (tmp_path / "stray.json").write_text("x" * 11)
    _patch_hf_multi(monkeypatch, {})
    row = verify_uploads.check_outroot_residue(
        999999, outroot=str(tmp_path), hf_prefixes=("issue999999_none",)
    )
    assert row["status"] == "FAIL"
    assert "HF files per searched repo" not in row["detail"]
    assert "discovered_data_repos" not in row

    clean_root = tmp_path / "clean"
    clean_root.mkdir()
    (clean_root / "homed.json").write_text("{}")
    _patch_hf_multi(monkeypatch, {(DEFAULT_DATA, "issue999999_x"): ["issue999999_x/homed.json"]})
    ok_row = verify_uploads.check_outroot_residue(
        999999, outroot=str(clean_root), hf_prefixes=("issue999999_x",)
    )
    assert ok_row["status"] == "OK"
    assert ok_row["detail"] == (
        "disk=1 matched=1; content-verified=0; verdict is the name-set diff, never the counts"
    )


def test_report_header_and_json_fields(monkeypatch):
    """T11: the report dict carries hf_data_repos / hf_model_repo and
    format_report renders the searched-repos header line."""
    report = verify_uploads.run_verification(
        999999, experiment_type="analysis", hf_data_repos=("org/extra",)
    )
    assert report["hf_data_repos"] == [
        {"repo": DEFAULT_DATA, "source": "default"},
        {"repo": "org/extra", "source": "flag"},
    ]
    assert report["hf_model_repo"] == {"repo": DEFAULT_MODEL, "source": "default"}
    rendered = verify_uploads.format_report(report)
    assert (
        f"**HF repos searched:** data: {DEFAULT_DATA} (default) + org/extra (flag); "
        f"model: {DEFAULT_MODEL} (default)" in rendered
    )


# ---------------------------------------------------------------------------
# T12: row-index multi-repo union (real seam bodies; Hub boundary faked)
# ---------------------------------------------------------------------------


def _patch_row_index_hub(
    monkeypatch, walk_mapping: dict[tuple[str, str], list], shas: dict[str, str]
):
    """Fake the row-index Hub boundary: per-repo repo_info shas, a walk keyed
    on (repo, prefix), and a staged fetch writing one deterministic row.
    Returns (walks, fetches) recorders."""
    import huggingface_hub

    import explore_persona_space.orchestrate.hub as hub

    walks: list[tuple[str, str, str | None]] = []
    fetches: list[tuple[str, str, str | None]] = []

    class _FakeApi:
        def __init__(self, token=None):
            pass

        def repo_info(self, repo_id, *, repo_type=None):
            return SimpleNamespace(sha=shas[repo_id])

        def get_paths_info(self, repo_id, paths, *, expand=False, revision=None, repo_type=None):
            return [SimpleNamespace(path=p, size=100) for p in paths]

    def fake_walk(api, repo_id, *, repo_type="model", revision=None, path_in_repo=None):
        from huggingface_hub.utils import EntryNotFoundError

        walks.append((repo_id, path_in_repo, revision))
        if (repo_id, path_in_repo) not in walk_mapping:
            raise EntryNotFoundError(f"no tree at {path_in_repo}")
        return list(walk_mapping[(repo_id, path_in_repo)])

    def fake_stage(
        repo_id,
        path_in_repo,
        target,
        *,
        repo_type="dataset",
        revision=None,
        token=None,
        overwrite=False,
        size_bytes=None,
    ):
        fetches.append((repo_id, path_in_repo, revision))
        target = Path(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(_row("c0", 0) + "\n", encoding="utf-8")
        return target

    monkeypatch.setattr(huggingface_hub, "HfApi", _FakeApi)
    monkeypatch.setattr(hub, "list_repo_entries_complete", fake_walk)
    monkeypatch.setattr(hub, "stage_hub_file", fake_stage)
    return walks, fetches


def test_row_index_multi_repo_union(monkeypatch):
    """T12: a prefix resolving ONLY on repo B enumerates from B at B's own
    pinned revision; the verdict reports per-repo revisions; entries are
    keyed (mode, repo, path) so the SAME store file duplicated on BOTH repos
    collapses at ROW grain under a declared distinct key (round-1 F17); and
    a prefix resolving on NO searched repo raises."""
    sha_a = "aa11bb22" * 5
    sha_b = "cc33dd44" * 5
    shas = {DEFAULT_DATA: sha_a, OVF: sha_b}
    path = "issueX/jobA/row_index_shard00.jsonl"

    # (a) prefix only on B -> enumerated from B at B's pinned revision.
    walks, fetches = _patch_row_index_hub(monkeypatch, {(OVF, "issueX/jobA"): [(path, 100)]}, shas)
    res = verify_uploads.check_realized_row_counts(
        expected_rows={"jobA": 1},
        hf_prefixes=("issueX/jobA",),
        distinct_key_fields=KEY,
        data_repos=(DEFAULT_DATA, OVF),
    )
    assert res["status"] == "OK"
    assert res["labels"]["jobA"]["realized_distinct"] == 1
    assert (DEFAULT_DATA, "issueX/jobA", sha_a) in walks, "A walked at A's pinned sha"
    assert (OVF, "issueX/jobA", sha_b) in walks, "B walked at B's pinned sha"
    assert fetches == [(OVF, path, sha_b)], "the fetch reads B at B's pinned sha"
    assert res["revisions"] == {DEFAULT_DATA: sha_a, OVF: sha_b}
    assert f"hub revisions: {DEFAULT_DATA}@{sha_a}, {OVF}@{sha_b}" in res["detail"]

    # (b) duplicated store on BOTH repos: two entries (distinct (mode, repo,
    # path) keys), row grain collapses under the declared key -> OK, not a
    # doubled count.
    walks, fetches = _patch_row_index_hub(
        monkeypatch,
        {(DEFAULT_DATA, "issueX/jobA"): [(path, 100)], (OVF, "issueX/jobA"): [(path, 100)]},
        shas,
    )
    res = verify_uploads.check_realized_row_counts(
        expected_rows={"jobA": 1},
        hf_prefixes=("issueX/jobA",),
        distinct_key_fields=KEY,
        data_repos=(DEFAULT_DATA, OVF),
    )
    assert res["status"] == "OK"
    assert res["labels"]["jobA"]["realized_distinct"] == 1
    assert res["labels"]["jobA"]["realized_lines"] == 2, "both copies fetched (two stores)"
    assert len(fetches) == 2

    # (c) a prefix resolving on NO searched repo raises (fail-loud).
    _patch_row_index_hub(monkeypatch, {}, shas)
    with pytest.raises(RuntimeError, match="resolves on NO searched repo"):
        verify_uploads._row_index_hf_entries(
            ("issueX/jobA",), (DEFAULT_DATA, OVF), revisions={DEFAULT_DATA: sha_a, OVF: sha_b}
        )


# ---------------------------------------------------------------------------
# T13: --hf-dataset multi-repo lattice (real check_hf_hub_path bodies)
# ---------------------------------------------------------------------------


def _patch_dataset_boundary(monkeypatch, mapping):
    """Fake for check_hf_hub_path's boundary: the scoped listing (keyed on
    (repo, path)) + an HfApi whose file_exists is always False (so an
    EntryNotFoundError cleanly becomes MISSING)."""
    import huggingface_hub

    class _NoFileApi:
        def __init__(self, token=None):
            pass

        def file_exists(self, repo_id, filename, *, repo_type=None, revision=None):
            return False

    monkeypatch.setattr(huggingface_hub, "HfApi", _NoFileApi)
    return _patch_hf_multi(monkeypatch, mapping)


def test_hf_dataset_multi_repo_lattice(monkeypatch):
    """T13: first OK wins; all-MISSING -> MISSING naming every searched
    repo; any ERROR with no OK -> ERROR (fail-loud dominates)."""
    path = "issueX/data"

    # First OK wins (resolved on the SECOND repo; URL names it).
    _patch_dataset_boundary(monkeypatch, {(OVF, path): [f"{path}/train.jsonl"]})
    row = verify_uploads._check_hf_dataset_across_repos([DEFAULT_DATA, OVF], path)
    assert row["status"] == "OK"
    assert OVF in row["url"]

    # All MISSING -> MISSING naming every repo searched.
    _patch_dataset_boundary(monkeypatch, {})
    row = verify_uploads._check_hf_dataset_across_repos([DEFAULT_DATA, OVF], path)
    assert row["status"] == "MISSING"
    assert DEFAULT_DATA in row["detail"] and OVF in row["detail"]

    # ERROR + no OK -> ERROR (presence in the erroring repo can't be ruled out).
    _patch_dataset_boundary(monkeypatch, {(OVF, path): RuntimeError("listing exploded")})
    row = verify_uploads._check_hf_dataset_across_repos([DEFAULT_DATA, OVF], path)
    assert row["status"] == "ERROR"
    assert OVF in row["detail"]

    # Singleton set returns the bare check row verbatim (byte-parity).
    _patch_dataset_boundary(monkeypatch, {})
    row = verify_uploads._check_hf_dataset_across_repos([DEFAULT_DATA], path)
    assert row["status"] == "MISSING"
    assert row["detail"] == f"No files under {path} at revision main"


# ---------------------------------------------------------------------------
# T14: model-side override + card precedence
# ---------------------------------------------------------------------------


def test_model_repo_override_and_card_precedence(monkeypatch):
    """T14: --hf-model-repo reaches the check-4 explicit-path branch (the
    fake boundary sees the override repo), and a reproducibility-card
    hf_model_repo still BEATS the flag default in check_hf_model_from_card."""
    fake = _patch_hf_multi(
        monkeypatch, {("org/models2", "adapters/x"): ["adapters/x/adapter_model.safetensors"]}
    )
    monkeypatch.setattr(verify_uploads, "_load_results_card", lambda n: None)
    # check_wandb_run is a live network probe and orthogonal to this pin —
    # stub it (unmodified function; the #906 body-test duty does not apply).
    monkeypatch.setattr(
        verify_uploads, "check_wandb_run", lambda run: {"status": "OK", "url": "", "detail": "stub"}
    )
    report = verify_uploads.run_verification(
        999999,
        experiment_type="training",
        wandb_run="entity/proj/run",
        hf_model_path="adapters/x",
        hf_model_repo="org/models2",
    )
    assert report["hf_model_repo"] == {"repo": "org/models2", "source": "flag"}
    assert report["checks"]["hf_model"]["status"] == "OK"
    model_calls = [
        (c.args[1], c.kwargs.get("path_in_repo"))
        for c in fake.call_args_list
        if c.kwargs.get("repo_type") == "model"
    ]
    assert model_calls == [("org/models2", "adapters/x")], (
        "the explicit --hf-model path must be checked against the override repo"
    )

    # Card precedence: hf_model_repo in the card wins over the flag default.
    fake = _patch_hf_multi(
        monkeypatch, {("org/card-repo", "adapters/y"): ["adapters/y/adapter_model.safetensors"]}
    )
    res = verify_uploads.check_hf_model_from_card(
        {"hf_model_path": "adapters/y", "hf_model_repo": "org/card-repo"},
        default_repo="org/flag-repo",
    )
    assert res is not None and res["status"] == "OK"
    repos_called = {c.args[1] for c in fake.call_args_list}
    assert repos_called == {"org/card-repo"}, "the card declaration wins; the flag never listed"


# ---------------------------------------------------------------------------
# T15: THE binding scope-discipline pin (round-1 B1)
# ---------------------------------------------------------------------------


def test_scope_discipline_listing_call_product(tmp_path, monkeypatch):
    """T15 (binding — BOTH assertions load-bearing; never drop either):

    (i) OUTCOME: with the requested prefix resolving NOWHERE and broadened
    listing shapes (path_in_repo=None / "" / the bare root) mapped to a
    listing that WOULD cover the planted file, the file stays residue ->
    FAIL naming it. Kills the REPLACEMENT mutation (a broadened call
    substituted for the caller-supplied prefix would hit a mapped broad
    shape and cover the file).

    (ii) CALL PRODUCT: the realized listing calls, read off the autospec
    mock's call_args_list, are EXACTLY the requested (prefix x repo)
    product — {(A, p), (B, p)}, each with repo_type="dataset", no extra
    calls, no omitted calls. Kills the ADDITIVE mutation in general: an
    extra unioned broad call with an UNANTICIPATED value raises
    EntryNotFoundError in the fake, is swallowed by the production per-pair
    continue, and leaves every outcome assert green — while in production
    it would union ~1M whole-repo basenames into the covered set.
    """
    (tmp_path / "f.bin").write_bytes(b"payload")
    prefix = "issue999999_p"
    broad_listing = [f"{prefix}/f.bin", "f.bin"]
    mapping = {
        # The requested (prefix x repo) product: resolves NOWHERE.
        # (unmapped -> EntryNotFoundError is the fake's default)
        # The canonical broadened shapes WOULD cover f.bin if ever issued:
        (OVF, None): broad_listing,
        (OVF, ""): broad_listing,
        (OVF, "/"): broad_listing,
        (DEFAULT_DATA, None): broad_listing,
        (DEFAULT_DATA, ""): broad_listing,
        (DEFAULT_DATA, "/"): broad_listing,
    }
    fake = _patch_hf_multi(monkeypatch, mapping)

    row = verify_uploads.check_outroot_residue(
        999999,
        outroot=str(tmp_path),
        hf_prefixes=(prefix,),
        data_repos=(DEFAULT_DATA, OVF),
    )

    # (i) Outcome: the file is residue; no broadened listing covered it.
    assert row["status"] == "FAIL"
    assert "f.bin" in row["detail"]

    # (ii) Call product: EXACTLY the requested (prefix x repo) grid.
    realized = [
        (c.args[1], c.kwargs.get("path_in_repo"), c.kwargs.get("repo_type"))
        for c in fake.call_args_list
    ]
    expected = [
        (DEFAULT_DATA, prefix, "dataset"),
        (OVF, prefix, "dataset"),
    ]
    assert sorted(realized) == sorted(expected), (
        "the realized listing-call product must be exactly the requested "
        f"(prefix x repo) grid; got {realized}"
    )
