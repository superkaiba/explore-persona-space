"""#2321 repack driver tests (plan v4 §12) — ALL against local fakes.

Every test runs under the conftest I18 interlock env (this module is in
``_I2321_TEST_MODULES``: ``HF_HUB_OFFLINE=1`` pinned, the apply permit
deleted). No test touches a real HF repo; Hub-boundary fakes are
signature-conformant (real ``RepoFile`` objects, real exception classes),
and the fake ``create_commit`` APPLIES operations against an in-memory
store so probe-first resume semantics are exercised for real.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest
from huggingface_hub.hf_api import RepoFile
from huggingface_hub.utils import EntryNotFoundError

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from explore_persona_space.orchestrate import packing  # noqa: E402

DRIVER_PATH = REPO_ROOT / "scripts" / "issue2321_repack.py"


def _load_driver():
    spec = importlib.util.spec_from_file_location("issue2321_repack_mod", DRIVER_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


repack = _load_driver()

FAKE_REPO = "fake-org/fake-data-repo"
PREFIX = "issue9999_fx"


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeHTTPError(Exception):
    """Response-bearing HTTP error (412 / 429 / 5xx shapes)."""

    def __init__(self, status: int, msg: str, retry_after: str | None = None):
        super().__init__(msg)
        headers = {"Retry-After": retry_after} if retry_after else {}
        self.response = SimpleNamespace(status_code=status, headers=headers)


class FakeHubRepo:
    """In-memory Hub fake: content-addressed store + pinned, APPLIED commits.

    ``create_commit`` enforces the parent pin for real (a stale pin 412s),
    applies operations to the store, and honors a per-call directive script:
    ``ok`` · ``filecount`` (I2 rejection) · ``429`` · ``412-forced`` ·
    ``timeout-lost`` (ambiguous, NOT applied) · ``timeout-landed`` (applied,
    then raises — the MF2e shape; with ``lag_probe=True`` the next probe is
    served the PRE-commit snapshot so the re-issue 412s).
    """

    def __init__(self, files: dict[str, bytes] | None = None, *, lag_probe: bool = False):
        self.files: dict[str, bytes] = dict(files or {})
        self.commits = 0
        self.lag_probe = lag_probe
        self.script: list[str] = []
        self.calls: list[dict] = []
        self._stale: tuple[str, dict[str, bytes]] | None = None

    @property
    def sha(self) -> str:
        return f"{self.commits:040x}"

    # -- read surface -------------------------------------------------------
    def repo_info(self, repo_id, *, repo_type="dataset"):
        if self._stale is not None:
            return SimpleNamespace(sha=self._stale[0])
        return SimpleNamespace(sha=self.sha)

    def list_repo_tree(
        self,
        repo_id=None,
        path_in_repo=None,
        *,
        repo_type="dataset",
        revision=None,
        recursive=False,
        expand=None,
    ):
        files = self._stale[1] if self._stale is not None else self.files
        norm = (path_in_repo or "").strip("/")
        matched = [p for p in sorted(files) if p == norm or p.startswith(norm + "/")]
        if not matched:
            raise EntryNotFoundError(f"Entry Not Found: {path_in_repo}")
        for p in matched:
            yield RepoFile(path=p, size=len(files[p]), oid=packing.git_blob_sha1(files[p]))

    # -- write surface ------------------------------------------------------
    def create_commit(self, *, repo_id, repo_type, operations, commit_message, parent_commit):
        from huggingface_hub import CommitOperationAdd

        adds, dels = [], []
        for op in operations:
            if isinstance(op, CommitOperationAdd):
                payload = op.path_or_fileobj
                data = payload if isinstance(payload, bytes) else Path(payload).read_bytes()
                adds.append((op.path_in_repo, data))
            else:
                dels.append(op.path_in_repo)
        self.calls.append(
            {
                "parent": parent_commit,
                "live_sha": self.sha,
                "message": commit_message,
                "adds": [p for p, _ in adds],
                "dels": list(dels),
            }
        )
        directive = self.script.pop(0) if self.script else "ok"
        self._stale = None  # any commit attempt ends the stale-probe window
        if directive == "filecount":
            raise FakeHTTPError(
                400,
                "Your git repo would contain too many files after this push, "
                "over the limit of 1000000 files.",
            )
        if directive == "429":
            raise FakeHTTPError(429, "Too Many Requests", retry_after="1")
        if directive == "412-forced":
            raise FakeHTTPError(412, "A commit has happened since. Please refresh and try again.")
        if parent_commit != self.sha:
            raise FakeHTTPError(412, "A commit has happened since. Please refresh and try again.")
        if directive == "timeout-lost":
            raise TimeoutError("request timed out")
        assert directive in ("ok", "timeout-landed"), directive
        prev = (self.sha, dict(self.files))
        for path, data in adds:
            self.files[path] = data
        for path in dels:
            assert path in self.files, f"delete of absent path {path} (double-apply?)"
            del self.files[path]
        self.commits += 1
        if directive == "timeout-landed":
            if self.lag_probe:
                self._stale = prev
            raise TimeoutError("request timed out (commit actually landed)")
        return SimpleNamespace(oid=self.sha)


# ---------------------------------------------------------------------------
# Fixture: a REAL v2 pack over a fixture tree + matching fake repo
# ---------------------------------------------------------------------------


def _fixture_files(prefix: str = PREFIX, n_a: int = 9, n_b: int = 5) -> dict[str, bytes]:
    files = {
        f"{prefix}/a/{i:03d}.json": json.dumps({"i": i, "t": "x" * 64}).encode() for i in range(n_a)
    }
    files.update({f"{prefix}/b/deep/{i:02d}.txt": f"row {i}\n".encode() * 12 for i in range(n_b)})
    return files


def _build_fixture(
    tmp_path: Path,
    *,
    prefix: str = PREFIX,
    files: dict[str, bytes] | None = None,
    verify: bool = True,
):
    """Stage a fixture tree, pack it for real, return the working set.

    ``verify=True`` (default) runs the REAL verify phase, minting the I5
    verify receipt the commit phase now requires (r2 C1); pass ``False`` to
    exercise the no-receipt refusal.
    """
    files = files if files is not None else _fixture_files(prefix)
    stage_root = tmp_path / "stage"
    for path, data in files.items():
        p = stage_root / path
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
    census = {
        path: repack.Anchor(
            size=len(data), is_lfs=False, blob_id=packing.git_blob_sha1(data), lfs_sha256=None
        )
        for path, data in files.items()
    }
    pack_dir = tmp_path / "pack" / prefix
    repack.pack_prefix(
        prefix=prefix,
        census=census,
        candidate_paths=sorted(files),
        stage_root=stage_root,
        pack_dir=pack_dir,
        source_revision="fixture-rev",
        git_commit="fixture-sha",
    )
    if verify:
        repack.verify_prefix(
            prefix=prefix,
            pack_dir=pack_dir,
            stage_root=stage_root,
            scratch_dir=tmp_path / "verify_scratch" / prefix,
            candidate_paths=sorted(files),
        )
    return SimpleNamespace(
        prefix=prefix,
        files=files,
        census=census,
        retained={},
        stage_root=stage_root,
        pack_dir=pack_dir,
        candidates=sorted(files),
    )


def _run_full(fx, fake, tmp_path, **kw):
    """Run the commit phase to completion against the fake; return summary."""
    return repack.run_commit_phase(
        fake,
        repo_id=FAKE_REPO,
        prefix=fx.prefix,
        pack_dir=fx.pack_dir,
        census=fx.census,
        retained=fx.retained,
        revision="fixture-rev",
        state_path=tmp_path / "state.jsonl",
        sleep_fn=lambda s: None,
        driver_git_sha="test-driver-sha",
        **kw,
    )


def _shard(name, group, members, size=100):
    return repack.ShardInfo(
        name=name,
        group=group,
        size=size,
        sha256="0" * 64,
        blob_sha1="1" * 40,
        members=tuple(members),
    )


def _groups_for(shards, parts_per_group=1):
    groups: dict[str, dict] = {}
    for s in shards:
        g = groups.setdefault(
            s.group,
            {"rel_dir": s.group, "index_files": [], "shard_files": [], "n_members": 0},
        )
        if not g["index_files"]:
            g["index_files"] = [f"{s.group}.index{i:02d}.json" for i in range(parts_per_group)]
        g["shard_files"].append(s.name)
        g["n_members"] += len(s.members)
    return groups


# ---------------------------------------------------------------------------
# Composer: partition / net-negative / rebalance (I3, I9, C7)
# ---------------------------------------------------------------------------


def test_composer_partition_disjoint_and_cover():
    shards = [
        _shard(f"g.shard{i:02d}.jsonl", "g", [f"g/m{i}_{j}" for j in range(6)]) for i in range(5)
    ]
    groups = _groups_for(shards)
    units, skip = repack.compose_units(shards, groups, prefix=PREFIX, ops_cap=18)
    assert skip is None and len(units) > 1
    binned = [s.name for u in units for s in u.shards]
    assert sorted(binned) == sorted(s.name for s in shards)
    all_dels = [p for u in units for p in u.planned_deletes]
    assert len(all_dels) == len(set(all_dels)) == 30
    assert {p for u in units for p in u.planned_deletes} == {
        f"{PREFIX}/{m}" for s in shards for m in s.members
    }


def test_units_net_negative_including_journal_index_ops():
    """I3: every data unit deletes MORE than it adds, counting INDEX + journal."""
    shards = [
        _shard(f"g.shard{i:02d}.jsonl", "g", [f"g/m{i}_{j}" for j in range(8)]) for i in range(4)
    ]
    units, skip = repack.compose_units(shards, _groups_for(shards), prefix=PREFIX, ops_cap=15)
    assert skip is None
    for u in units:
        assert u.n_adds == len(u.shards) + len(u.index_part_names) + 2
        assert len(u.planned_deletes) >= u.n_adds + 1
        assert u.total_ops <= 15


def test_sparse_tail_rebalance_merges_into_preceding():
    """C7: a 1-member tail unit is rebalanced against the preceding unit —
    afterwards NO unit is net-positive (the tail rides with a donor shard)."""
    shards = [
        _shard("g.shard00.jsonl", "g", [f"g/m0_{j}" for j in range(6)]),
        _shard("g.shard01.jsonl", "g", [f"g/m1_{j}" for j in range(6)]),
        _shard("g.shard02.jsonl", "g", ["g/tail"]),  # alone: 3 adds vs 1 del => net +2
    ]
    groups = _groups_for(shards)
    # ops_cap 18 splits the tail into its own bin (greedy trial = 19 ops),
    # leaving it net-positive; the C7 rebalance resolves it.
    units, skip = repack.compose_units(shards, groups, prefix=PREFIX, ops_cap=18)
    assert skip is None
    assert len(units) == 2
    assert all(len(u.planned_deletes) >= u.n_adds + 1 for u in units)
    tail_unit = next(u for u in units if any(s.name == "g.shard02.jsonl" for s in u.shards))
    assert len(tail_unit.shards) == 2  # the tail shares its unit with a donor shard


def test_all_sparse_prefix_skipped_not_abort():
    """An all-sparse prefix is SKIPPED + reported — never a run abort (C7)."""
    shards = [_shard("g.shard00.jsonl", "g", ["g/only", "g/two"])]
    units, skip = repack.compose_units(shards, _groups_for(shards), prefix=PREFIX)
    assert units is None
    assert "all-sparse" in skip


def test_delete_set_equals_shard_member_set(tmp_path):
    """I1: unit deletes are EXACTLY the same-commit shard members — and since
    r2 C3 the member set is DECODED FROM THE SHARD BYTES, not read off the
    index parts (the index is cross-checked against the shard records)."""
    fx = _build_fixture(tmp_path)
    shards, groups, _man = repack.load_shard_infos(fx.pack_dir)
    # C3 ground truth: members come from the shards themselves.
    for s in shards:
        assert list(s.members) == packing.read_shard_member_srcs(fx.pack_dir / s.name)
    units, skip = repack.compose_units(shards, groups, prefix=fx.prefix)
    assert skip is None
    unit = units[0]
    assert unit.planned_deletes == frozenset(
        f"{fx.prefix}/{m}" for s in unit.shards for m in s.members
    )
    adds = repack.build_unit_ops(unit, pack_dir=fx.pack_dir, index_bytes=b"{}", journal_bytes_=b"")
    dels = sorted(unit.planned_deletes)
    repack.assert_unit_invariants(unit, adds, dels, retained={}, census=fx.census)
    with pytest.raises(repack.AbortPrefix, match="I1"):
        repack.assert_unit_invariants(
            unit, adds, [*dels[:-1], f"{fx.prefix}/NOT_A_MEMBER"], retained={}, census=fx.census
        )
    with pytest.raises(repack.AbortPrefix, match="I4"):
        repack.assert_unit_invariants(
            unit, adds, dels, retained={dels[0]: "pack-part:x"}, census=fx.census
        )


def _tamper_index_part(pack_dir: Path, mutate) -> str:
    """Load the first index part, apply ``mutate(members_dict)``, rewrite it."""
    man = json.loads((pack_dir / packing.MANIFEST_NAME).read_text())
    part_name = next(iter(man["groups"].values()))["index_files"][0]
    doc = json.loads((pack_dir / part_name).read_text())
    mutate(doc["members"])
    (pack_dir / part_name).write_text(json.dumps(doc, sort_keys=True, separators=(",", ":")))
    return part_name


@pytest.mark.parametrize(
    "case,match",
    [
        ("extra", "absent from every shard"),
        ("missing", "has NO index entry"),
        ("wrong-offset", "does not resolve to its shard record"),
    ],
)
def test_load_shard_infos_index_tamper_aborts(tmp_path, case, match):
    """r2 Codex-C3: the delete set derives from SHARD RECORDS; a tampered
    index part (extra / missing / mis-resolving entry) ABORTS the prefix —
    the pre-fix code would have derived deletes from the tampered index."""
    fx = _build_fixture(tmp_path)

    def mutate(members: dict) -> None:
        first = next(iter(members))
        if case == "extra":
            ghost = dict(members[first])
            members["GHOST_NOT_IN_ANY_SHARD.json"] = ghost
        elif case == "missing":
            del members[first]
        else:  # wrong-offset
            members[first] = {**members[first], "offset": members[first]["offset"] + 1}

    _tamper_index_part(fx.pack_dir, mutate)
    with pytest.raises(repack.AbortPrefix, match=match):
        repack.load_shard_infos(fx.pack_dir)


def test_commit_phase_index_tamper_composes_no_deletes(tmp_path):
    """r2 Codex-C3 (mechanized recipe): an index part pointing at a WRONG
    byte range aborts the commit phase BEFORE any Hub call — even with a
    freshly regenerated receipt, so the C3 cross-check (not the receipt
    staleness gate) is what blocks it."""
    fx = _build_fixture(tmp_path)
    _tamper_index_part(
        fx.pack_dir,
        lambda members: members.update(
            {next(iter(members)): {**members[next(iter(members))], "length": 1}}
        ),
    )
    # Regenerate the receipt over the TAMPERED pack: the receipt gate now
    # PASSES, so only the C3 shard-record cross-check can refuse.
    repack.write_verify_receipt(fx.pack_dir, {"n_members": len(fx.candidates)})
    fake = FakeHubRepo(dict(fx.files))
    with pytest.raises(repack.AbortPrefix, match="does not resolve to its shard record"):
        _run_full(fx, fake, tmp_path)
    assert fake.calls == []  # zero create_commit — no delete was ever composed


# ---------------------------------------------------------------------------
# C14 retained set: v1 / v2 / #2119 manifests, exact sets
# ---------------------------------------------------------------------------


def _entry(path: str, data: bytes = b"x"):
    return SimpleNamespace(
        path=path,
        size=len(data),
        is_lfs=False,
        lfs_sha256=None,
        blob_id=packing.git_blob_sha1(data),
    )


def test_retained_closure_v1_fixture_exact_set():
    """v1 (scripts/issue1739_pack.py) manifests retain manifest + listed shards."""
    v1 = {
        "version": 1,
        "groups": {
            "grp": {
                "census_sha256": "c" * 64,
                "shards": [{"name": "grp.shard00.jsonl"}, {"name": "grp.shard01.jsonl"}],
            }
        },
    }
    texts = {f"{PREFIX}/old/pack_manifest.json": json.dumps(v1)}
    entries = [
        _entry(f"{PREFIX}/old/pack_manifest.json"),
        _entry(f"{PREFIX}/old/grp.shard00.jsonl"),
        _entry(f"{PREFIX}/keep_me.json"),
    ]
    retained = repack.build_retained_set(entries, prefix=PREFIX, fetch_text=lambda p: texts[p])
    assert set(retained) == {
        f"{PREFIX}/old/pack_manifest.json",
        f"{PREFIX}/old/grp.shard00.jsonl",
        f"{PREFIX}/old/grp.shard01.jsonl",  # transitive: listed but not in the walk
    }
    assert f"{PREFIX}/keep_me.json" not in retained


def test_retained_closure_v2_and_2119_fixtures_exact_set(tmp_path):
    """v2 pack manifests + #2119 sharded-text manifests retain their full part sets."""
    fx = _build_fixture(tmp_path, prefix=PREFIX)
    man = json.loads((fx.pack_dir / packing.MANIFEST_NAME).read_text())
    v2_dir = f"{PREFIX}/packedold"
    sharded = {"parts": ["big.shard00.jsonl", "big.shard01.jsonl"], "sha256": {}}
    texts = {
        f"{v2_dir}/pack_manifest.json": json.dumps(man),
        f"{PREFIX}/draws/big.manifest.json": json.dumps(sharded),
    }
    entries = [
        _entry(f"{v2_dir}/pack_manifest.json"),
        _entry(f"{PREFIX}/draws/big.manifest.json"),
        _entry(f"{PREFIX}/draws/big.shard00.jsonl"),
        _entry(f"{PREFIX}/plain.json"),
        _entry(f"{PREFIX}/OVERFLOW_POINTER.json"),
    ]
    retained = repack.build_retained_set(entries, prefix=PREFIX, fetch_text=lambda p: texts[p])
    expected = {
        f"{v2_dir}/pack_manifest.json",
        f"{PREFIX}/OVERFLOW_POINTER.json",
        f"{PREFIX}/draws/big.manifest.json",
    }
    expected |= {f"{v2_dir}/{n}" for n in repack.parts_from_pack_manifest(man)}
    expected |= {f"{PREFIX}/draws/big.shard00.jsonl", f"{PREFIX}/draws/big.shard01.jsonl"}
    assert set(retained) == expected
    assert f"{PREFIX}/plain.json" not in retained


def test_retained_orphan_name_shape_fallback():
    """Belt-and-braces: part-shaped names with NO manifest are retained too."""
    entries = [
        _entry(f"{PREFIX}/lost/orphan.shard03.jsonl"),
        _entry(f"{PREFIX}/lost/orphan.index00.json"),
        _entry(f"{PREFIX}/lost/units.jsonl"),
        _entry(f"{PREFIX}/lost/INDEX.json"),
        _entry(f"{PREFIX}/normal.json"),
    ]
    retained = repack.build_retained_set(entries, prefix=PREFIX, fetch_text=lambda p: "")
    assert {p for p, r in retained.items() if r == "orphan-name-shape"} == {
        f"{PREFIX}/lost/orphan.shard03.jsonl",
        f"{PREFIX}/lost/orphan.index00.json",
        f"{PREFIX}/lost/units.jsonl",
        f"{PREFIX}/lost/INDEX.json",
    }
    assert f"{PREFIX}/normal.json" not in retained


def test_retained_unrecognized_manifest_fails_loud():
    entries = [_entry(f"{PREFIX}/x/pack_manifest.json")]
    with pytest.raises(ValueError, match="unrecognized"):
        repack.build_retained_set(
            entries, prefix=PREFIX, fetch_text=lambda p: json.dumps({"version": 99})
        )


# ---------------------------------------------------------------------------
# §3.3 selection
# ---------------------------------------------------------------------------


def test_selection_excludes_lfs_oversize_retained_packed():
    big = 20_000_000
    entries = [
        _entry(f"{PREFIX}/small.json"),
        SimpleNamespace(
            path=f"{PREFIX}/w.safetensors",
            size=100,
            is_lfs=True,
            lfs_sha256="c" * 64,
            blob_id="b" * 40,
        ),
        SimpleNamespace(
            path=f"{PREFIX}/huge.json", size=big, is_lfs=False, lfs_sha256=None, blob_id="b" * 40
        ),
        _entry(f"{PREFIX}/packed/x.shard00.jsonl"),
        _entry(f"{PREFIX}/kept/pack_manifest.json"),
    ]
    retained = {f"{PREFIX}/kept/pack_manifest.json": "pack-manifest"}
    candidates, exclusions = repack.select_pack_candidates(entries, retained, prefix=PREFIX)
    assert [e.path for e in candidates] == [f"{PREFIX}/small.json"]
    assert exclusions == {
        "lfs-not-tier-b": 1,
        "oversize-encoded-line": 1,
        "under-packed-dir": 1,
        "retained": 1,
    }


def test_selection_tier_b_npz_lfs_allowed():
    """Tier B (issue667_alllayer): LFS .npz files ARE candidates (b64-packed)."""
    prefix = "issue667_alllayer"
    entries = [
        SimpleNamespace(
            path=f"{prefix}/x/t.npz", size=1000, is_lfs=True, lfs_sha256="c" * 64, blob_id=None
        ),
        SimpleNamespace(
            path=f"{prefix}/x/w.safetensors",
            size=1000,
            is_lfs=True,
            lfs_sha256="d" * 64,
            blob_id=None,
        ),
    ]
    candidates, exclusions = repack.select_pack_candidates(entries, {}, prefix=prefix)
    assert [e.path for e in candidates] == [f"{prefix}/x/t.npz"]
    assert exclusions == {"lfs-not-tier-b": 1}


# ---------------------------------------------------------------------------
# Verify phase: bijection + delta report (C3 / I12)
# ---------------------------------------------------------------------------


def test_bijection_verify_pass_and_delta_report(tmp_path):
    fx = _build_fixture(tmp_path)
    report = repack.verify_prefix(
        prefix=fx.prefix,
        pack_dir=fx.pack_dir,
        stage_root=fx.stage_root,
        scratch_dir=tmp_path / "scratch",
        candidate_paths=fx.candidates,
    )
    assert report["bijection"] == "exact"
    assert report["n_members"] == len(fx.candidates)
    with pytest.raises(repack.AbortPrefix, match="missing_from_pack"):
        repack.verify_prefix(
            prefix=fx.prefix,
            pack_dir=fx.pack_dir,
            stage_root=fx.stage_root,
            scratch_dir=tmp_path / "scratch2",
            candidate_paths=[*fx.candidates, f"{fx.prefix}/a/GHOST.json"],
        )


# ---------------------------------------------------------------------------
# Probe-first commit loop (I7 / I11 / I13 / I14 / C18 / I2)
# ---------------------------------------------------------------------------


def _one_unit(fx):
    shards, groups, _man = repack.load_shard_infos(fx.pack_dir)
    units, skip = repack.compose_units(shards, groups, prefix=fx.prefix)
    assert skip is None and len(units) == 1
    unit = units[0]
    record = repack.unit_journal_record(
        unit, n_units=1, census_key="ck", revision="fixture-rev", driver_git_sha="sha"
    )
    adds = repack.build_unit_ops(
        unit,
        pack_dir=fx.pack_dir,
        index_bytes=repack.top_index_bytes(groups, list(groups)),
        journal_bytes_=repack.journal_bytes([record]),
    )
    dels = sorted(unit.planned_deletes)
    expected = repack.unit_expected(unit, pack_dir=fx.pack_dir, census=fx.census)
    return unit, adds, dels, expected


def _commit_kwargs(fx, adds, dels, expected, **kw):
    base = dict(
        repo_id=FAKE_REPO,
        expected=expected,
        adds=adds,
        dels=dels,
        commit_message="[#2321] test unit",
        sleep_fn=lambda s: None,
    )
    base.update(kw)
    return base


def test_commit_timeout_probe_before_retry(tmp_path):
    """I11: landed => done with ZERO further create_commit; clean => ONE re-issue."""
    fx = _build_fixture(tmp_path)
    _unit, adds, dels, expected = _one_unit(fx)

    # (a) already landed: strict paths present + sources absent => zero commits.
    landed_files = {p: repack._payload_bytes(b) for p, b in adds}
    fake = FakeHubRepo(landed_files)
    res = repack.commit_unit_probe_first(fake, **_commit_kwargs(fx, adds, dels, expected))
    assert res["state"] == "landed"
    assert fake.calls == []

    # (b) clean + ambiguous timeout that did NOT land => exactly ONE re-issue.
    fake = FakeHubRepo(dict(fx.files))
    fake.script = ["timeout-lost", "ok"]
    res = repack.commit_unit_probe_first(fake, **_commit_kwargs(fx, adds, dels, expected))
    assert res["state"] == "committed"
    assert len(fake.calls) == 2
    for p in dels:
        assert p not in fake.files

    # (c) ambiguous budget exhausts at 3 => AbortPrefix, never blind retries.
    fake = FakeHubRepo(dict(fx.files))
    fake.script = ["timeout-lost", "timeout-lost", "timeout-lost"]
    with pytest.raises(repack.AbortPrefix, match="attempts-exhausted"):
        repack.commit_unit_probe_first(fake, **_commit_kwargs(fx, adds, dels, expected))
    assert len(fake.calls) == 3

    # (d) MIXED partial state => AbortPrefix (rc=23).
    first_add_path, first_payload = adds[0]
    mixed = dict(fx.files)
    mixed[first_add_path] = repack._payload_bytes(first_payload)
    fake = FakeHubRepo(mixed)
    with pytest.raises(repack.AbortPrefix, match="MIXED"):
        repack.commit_unit_probe_first(fake, **_commit_kwargs(fx, adds, dels, expected))
    assert repack.AbortPrefix.rc == 23


def test_create_commit_single_site_never_retry_wrapped():
    """Structural: create_commit reachable ONLY via commit_unit_probe_first,
    never wrapped in retry_transient, always parent-pinned, NO_RETRY-waivered."""
    src = DRIVER_PATH.read_text(encoding="utf-8")
    tree = ast.parse(src)
    owners: list[tuple[str, ast.Call]] = []

    def visit(node, stack):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            stack = [*stack, node.name]
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "create_commit"
        ):
            owners.append((stack[-1] if stack else "<module>", node))
        for child in ast.iter_child_nodes(node):
            visit(child, stack)

    visit(tree, [])
    assert owners, "no create_commit site found"
    assert {name for name, _ in owners} == {"commit_unit_probe_first"}
    for _name, call in owners:
        assert any(kw.arg == "parent_commit" for kw in call.keywords), "I14: unpinned commit"
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            fname = getattr(node.func, "attr", None) or getattr(node.func, "id", None)
            if fname == "retry_transient":
                inner = [
                    s
                    for s in ast.walk(node)
                    if isinstance(s, ast.Call) and getattr(s.func, "attr", None) == "create_commit"
                ]
                assert not inner, "I11: create_commit wrapped in retry_transient"
    waiver = (
        "# NO_RETRY: I11 probe-first — blind transient retry unsafe on deletion-bearing commits"
    )
    lines = src.splitlines()
    call_lines = [i for i, ln in enumerate(lines) if "api.create_commit(" in ln]
    assert call_lines, "create_commit call line not found"
    for i in call_lines:
        window = "\n".join(lines[max(0, i - 1) : i + 1])
        assert waiver in window, "NO_RETRY waiver must sit on/above the create_commit line"


def test_changed_bytes_same_paths_abort(tmp_path):
    """I13(b): a strict path present with DIFFERENT bytes => content-mismatch rc=23."""
    fx = _build_fixture(tmp_path)
    _unit, adds, dels, expected = _one_unit(fx)
    poisoned = dict(fx.files)
    poisoned[adds[0][0]] = b"DIFFERENT BYTES (re-derived census?)"
    fake = FakeHubRepo(poisoned)
    with pytest.raises(repack.AbortPrefix, match="content-mismatch") as exc:
        repack.commit_unit_probe_first(fake, **_commit_kwargs(fx, adds, dels, expected))
    assert exc.value.rc == 23
    assert fake.calls == []  # never issued


def test_drift_check_abort(tmp_path):
    """I7: clean-shaped state whose SOURCE bytes changed since revision R => abort."""
    fx = _build_fixture(tmp_path)
    _unit, adds, dels, expected = _one_unit(fx)
    drifted = dict(fx.files)
    drifted[dels[0]] = b"MUTATED SINCE REVISION R"
    fake = FakeHubRepo(drifted)
    with pytest.raises(repack.AbortPrefix, match="I7"):
        repack.commit_unit_probe_first(fake, **_commit_kwargs(fx, adds, dels, expected))
    assert fake.calls == []


def test_modification_after_drift_check_412(tmp_path):
    """I14: 412s consume the PIN budget (8), never the ambiguous budget (3)."""
    fx = _build_fixture(tmp_path)
    _unit, adds, dels, expected = _one_unit(fx)
    fake = FakeHubRepo(dict(fx.files))
    fake.script = ["412-forced"] * 5 + ["ok"]  # 5 > max_attempts=3, still succeeds
    res = repack.commit_unit_probe_first(fake, **_commit_kwargs(fx, adds, dels, expected))
    assert res["state"] == "committed"
    assert len(fake.calls) == 6

    fake = FakeHubRepo(dict(fx.files))
    fake.script = ["412-forced"] * 9
    with pytest.raises(repack.AbortPrefix, match="pin-cycles-exhausted"):
        repack.commit_unit_probe_first(fake, **_commit_kwargs(fx, adds, dels, expected))
    assert len(fake.calls) == 9


def test_timeout_lands_after_clean_parent_conflict(tmp_path):
    """MF2e: timed-out-but-landed + STALE probe => pinned re-issue 412s =>
    re-probe reads landed; ZERO double-applies."""
    fx = _build_fixture(tmp_path)
    _unit, adds, dels, expected = _one_unit(fx)
    fake = FakeHubRepo(dict(fx.files), lag_probe=True)
    fake.script = ["timeout-landed"]
    res = repack.commit_unit_probe_first(fake, **_commit_kwargs(fx, adds, dels, expected))
    assert res["state"] == "landed"
    assert len(fake.calls) == 2  # the landed-then-raised issue + the 412'd re-issue
    assert fake.calls[1]["parent"] != fake.calls[1]["live_sha"]  # stale pin => natural 412
    for p in dels:
        assert p not in fake.files  # applied exactly once (fake asserts on double-delete)


def test_429_exhaustion_reports_rate_limited(tmp_path):
    """C18: the 429 budget is its OWN outcome (rc=24) — never attempts-exhausted."""
    fx = _build_fixture(tmp_path)
    _unit, adds, dels, expected = _one_unit(fx)
    fake = FakeHubRepo(dict(fx.files))
    fake.script = ["429"] * 7
    sleeps: list[float] = []
    with pytest.raises(repack.RateLimitedStop) as exc:
        repack.commit_unit_probe_first(
            fake, **_commit_kwargs(fx, adds, dels, expected, sleep_fn=sleeps.append)
        )
    assert "attempts-exhausted" not in str(exc.value)
    assert exc.value.rc == 24
    assert len(fake.calls) == 7
    assert sleeps == [30.0] * 6  # Retry-After floored pacing between cycles


def test_interlock_refuses_canonical_commit(tmp_path):
    """I18: a canonical-repo commit from a pytest process refuses BEFORE any network."""
    fx = _build_fixture(tmp_path)
    _unit, adds, dels, expected = _one_unit(fx)
    fake = FakeHubRepo(dict(fx.files))
    assert os.environ.get("HF_HUB_OFFLINE") == "1"  # conftest pin for this module
    kwargs = _commit_kwargs(fx, adds, dels, expected)
    kwargs["repo_id"] = repack.DEFAULT_REPO_ID  # the CANONICAL repo id
    with pytest.raises(packing.TestMutationInterlockError):
        repack.commit_unit_probe_first(fake, **kwargs)
    assert fake.calls == []


# ---------------------------------------------------------------------------
# Commit phase: dry-run, cap rejection, resume, journal, I13(a)
# ---------------------------------------------------------------------------


def test_dry_run_thread_issues_zero_mutations(tmp_path):
    """--dry-run reaches the REAL composer path with ZERO create_commit calls."""
    from huggingface_hub import HfApi

    api = mock.create_autospec(HfApi, instance=True)
    # r2 Codex-C2: the canonical-repo path now admits ONLY PREFIX_ORDER
    # prefixes, so the fixtures use two REAL target prefixes.
    for i, prefix in enumerate(("issue1090_partial", "issue1434_writingstyle")):
        fx = _build_fixture(tmp_path / f"p{i}", prefix=prefix)
        summary = repack.run_commit_phase(
            api,
            repo_id=repack.DEFAULT_REPO_ID,  # dry-run must be safe even on the canonical id
            prefix=fx.prefix,
            pack_dir=fx.pack_dir,
            census=fx.census,
            retained=fx.retained,
            revision="fixture-rev",
            dry_run=True,
            sleep_fn=lambda s: None,
            driver_git_sha="sha",
        )
        assert summary["status"] == "committed"
        assert summary["n_units"] >= 1
        assert all(u["n_dels"] > u["n_adds"] for u in summary["units"])
    assert api.create_commit.call_count == 0
    assert api.method_calls == []  # dry-run makes ZERO network calls of any kind


def test_cap_rejection_stops_with_zero_further_deletes(tmp_path):
    """I2: a file-count rejection is a GLOBAL stop (rc=21) with nothing applied."""
    fx = _build_fixture(tmp_path)
    fake = FakeHubRepo(dict(fx.files))
    fake.script = ["filecount"]
    with pytest.raises(repack.StopRepack) as exc:
        _run_full(fx, fake, tmp_path)
    assert exc.value.rc == 21
    assert len(fake.calls) == 1  # nothing after the rejection
    assert fake.files == fx.files  # zero deletes applied
    state_rows = [json.loads(ln) for ln in (tmp_path / "state.jsonl").read_text().splitlines()]
    assert state_rows[-1]["event"] == "prefix-abort"
    assert state_rows[-1]["label"] == "packed-unindexed-final"  # C12


def test_commit_phase_lands_units_and_finalizes(tmp_path):
    fx = _build_fixture(tmp_path)
    fake = FakeHubRepo(dict(fx.files))
    summary = _run_full(fx, fake, tmp_path)
    assert summary["status"] == "committed"
    assert summary["freed"] > 0
    packed = f"{fx.prefix}/{packing.PACKED_DIRNAME}"
    assert f"{packed}/{packing.MANIFEST_NAME}" in fake.files
    for p in fx.files:
        assert p not in fake.files  # every original deleted
    # I3 realized: the packed replacement is exactly shards + parts +
    # INDEX.json + units.jsonl + pack_manifest.json — far fewer files.
    shards, groups, _man = repack.load_shard_infos(fx.pack_dir)
    n_parts = sum(len(g["index_files"]) for g in groups.values())
    assert len(fake.files) == len(shards) + n_parts + 3
    assert len(fake.files) < len(fx.files)


def test_unit_journal_lands_in_data_commit(tmp_path):
    """I15/I16: units.jsonl + cumulative INDEX.json ride EVERY data commit."""
    fx = _build_fixture(tmp_path)
    fake = FakeHubRepo(dict(fx.files))
    _run_full(fx, fake, tmp_path)
    packed = f"{fx.prefix}/{packing.PACKED_DIRNAME}"
    data_calls = [c for c in fake.calls if c["dels"]]
    for call in data_calls:
        assert f"{packed}/{packing.UNITS_JOURNAL_NAME}" in call["adds"]
        assert f"{packed}/{packing.INDEX_NAME}" in call["adds"]
    journal = fake.files[f"{packed}/{packing.UNITS_JOURNAL_NAME}"]
    records = [json.loads(ln) for ln in journal.decode().splitlines()]
    assert [r["unit_id"] for r in records] == list(range(1, len(records) + 1))
    for rec in records:
        assert rec["census_key"]
        assert rec["revision"] == "fixture-rev"
        assert rec["driver_git_sha"] == "test-driver-sha"
        assert len(rec["delete_set_sha256"]) == 64
        assert all(len(s["blob_sha1"]) == 40 for s in rec["shards"])
    # the landed journal matches the local one byte-for-byte
    assert journal == repack.journal_bytes(repack.load_local_journal(fx.pack_dir))


def test_resume_predicate_content_anchored(tmp_path):
    """I9: resume is keyed on HUB state — landed units re-probe as landed and
    are never re-issued, even after the local journal is wiped."""
    fx = _build_fixture(tmp_path)
    fake = FakeHubRepo(dict(fx.files))
    _run_full(fx, fake, tmp_path)
    n_calls = len(fake.calls)
    (fx.pack_dir / packing.UNITS_JOURNAL_NAME).unlink()  # total local-journal loss
    summary = _run_full(fx, fake, tmp_path)
    assert summary["status"] == "committed"
    assert len(fake.calls) == n_calls  # zero new create_commit calls
    # the local journal was reconstructed from the re-probed landed units
    assert repack.load_local_journal(fx.pack_dir)


def test_resume_rederived_census_aborts_not_overwrites(tmp_path):
    """I13(a)/I15: resume with a DIFFERENT re-derived census aborts, never overwrites."""
    fx = _build_fixture(tmp_path)
    # (a) Hub side: a packed path exists with different bytes => guard aborts.
    shards, _groups, _man = repack.load_shard_infos(fx.pack_dir)
    packed = f"{fx.prefix}/{packing.PACKED_DIRNAME}"
    poisoned = dict(fx.files)
    poisoned[f"{packed}/{shards[0].name}"] = b"foreign bytes"
    fake = FakeHubRepo(poisoned)
    with pytest.raises(repack.AbortPrefix, match=r"I13\(a\)"):
        _run_full(fx, fake, tmp_path)
    assert all(not c["dels"] for c in fake.calls)  # no delete was ever composed

    # (b) local side: a journal record whose delete-set digest mismatches the
    # re-derived plan aborts (a silently-different census on resume).
    fx2 = _build_fixture(tmp_path / "b", prefix="issue9996_fxc")
    fake2 = FakeHubRepo(dict(fx2.files))
    _run_full(fx2, fake2, tmp_path / "b")
    journal_path = fx2.pack_dir / packing.UNITS_JOURNAL_NAME
    rec = json.loads(journal_path.read_text().splitlines()[0])
    rec["delete_set_sha256"] = "0" * 64
    journal_path.write_text(json.dumps(rec, sort_keys=True) + "\n")
    with pytest.raises(repack.AbortPrefix, match="delete-set digest"):
        _run_full(fx2, fake2, tmp_path / "b")


# ---------------------------------------------------------------------------
# Intermediate resolvability (I16) + MF3 census reconstruction (I15)
# ---------------------------------------------------------------------------


def _dump_fake_to_dir(fake: FakeHubRepo, root: Path) -> Path:
    for path, data in fake.files.items():
        p = root / path
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
    return root


def _shim_remote(monkeypatch, root: Path):
    """Serve a dumped fake repo through the hub packed-fallback shim."""
    from explore_persona_space.orchestrate import hub

    hub.clear_packed_caches()

    def fake_download(
        repo_id=None,
        filename=None,
        *,
        repo_type="dataset",
        revision=None,
        local_dir=None,
        token=None,
    ):
        src = root / filename
        if not src.is_file():
            raise EntryNotFoundError(f"Entry Not Found for url: {filename}")
        dest = Path(local_dir) / filename
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(src.read_bytes())
        return str(dest)

    monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_download)
    monkeypatch.setattr("huggingface_hub.HfApi", lambda token=None: _DirApi(root))
    return hub


def test_intermediate_unit_members_resolvable(tmp_path, monkeypatch):
    """I16: after an ABORT mid-prefix, already-landed units' members resolve
    through the production shim (packed-unindexed-final is a consistent state)."""
    files = _fixture_files(PREFIX, n_a=14, n_b=8)
    fx = _build_fixture(tmp_path, files=files)
    fake = FakeHubRepo(dict(fx.files))
    # Force >=2 units with a small ops cap, then kill unit 2 with pin exhaustion.
    shards, groups, _man = repack.load_shard_infos(fx.pack_dir)
    units, skip = repack.compose_units(shards, groups, prefix=fx.prefix, ops_cap=18)
    assert skip is None and len(units) >= 2
    fake.script = ["ok"] + ["412-forced"] * 9
    with pytest.raises(repack.AbortPrefix, match="pin-cycles-exhausted"):
        _run_full(fx, fake, tmp_path, ops_cap=18)
    state_rows = [json.loads(ln) for ln in (tmp_path / "state.jsonl").read_text().splitlines()]
    abort = state_rows[-1]
    assert abort["event"] == "prefix-abort"
    assert abort["label"] == "packed-unindexed-final"  # C12
    assert abort["landed_units"] == 1
    # Unit 1's members are gone raw but resolvable via the shim (I16: the
    # cumulative INDEX.json + index parts landed IN unit 1's own commit).
    hub = _shim_remote(monkeypatch, _dump_fake_to_dir(fake, tmp_path / "remote"))
    api = _DirApi(tmp_path / "remote")
    landed_members = {f"{fx.prefix}/{m}" for s in units[0].shards for m in s.members}
    resolved = {m.path for m in hub.packed_members_under_path(api, FAKE_REPO, fx.prefix)}
    assert landed_members <= resolved
    member = sorted(landed_members)[0]
    got = hub.stage_packed_file(FAKE_REPO, member, tmp_path / "out" / "m.bin", repo_type="dataset")
    assert Path(got).read_bytes() == fx.files[member]


class _DirApi:
    """Directory-backed HfApi fake (the test_hub_packed_fallback shape)."""

    def __init__(self, root: Path):
        self.root = Path(root)

    def repo_info(self, repo_id, *, repo_type="dataset"):
        return SimpleNamespace(sha="d1a5" * 10)

    def file_exists(self, repo_id, filename, *, repo_type="dataset", revision=None):
        return (self.root / filename).is_file()

    def list_repo_tree(
        self,
        repo_id=None,
        path_in_repo=None,
        *,
        repo_type="dataset",
        revision=None,
        recursive=False,
        expand=None,
    ):
        base = self.root / (path_in_repo or "")
        if not base.exists():
            raise EntryNotFoundError(f"Entry Not Found: {path_in_repo}")
        for p in sorted(base.rglob("*")):
            if p.is_file():
                rel = p.relative_to(self.root).as_posix()
                yield RepoFile(
                    path=rel, size=p.stat().st_size, oid=packing.git_blob_sha1(p.read_bytes())
                )


def test_resume_from_hub_journal_reconstructs_census(tmp_path, monkeypatch):
    """MF3/I15: after a TOTAL local wipe, the census reconstructs EXACTLY from
    (landed packed members) UNION (surviving raw originals)."""
    files = _fixture_files(PREFIX, n_a=14, n_b=8)
    fx = _build_fixture(tmp_path, files=files)
    fake = FakeHubRepo(dict(fx.files))
    fake.script = ["ok"] + ["412-forced"] * 9  # unit 1 lands, unit 2 aborts
    with pytest.raises(repack.AbortPrefix):
        _run_full(fx, fake, tmp_path, ops_cap=18)
    hub = _shim_remote(monkeypatch, _dump_fake_to_dir(fake, tmp_path / "remote"))
    del hub  # the reconstruction path imports it internally
    api = _DirApi(tmp_path / "remote")
    reconstructed = repack.reconstruct_prefix_census(api, repo_id=FAKE_REPO, prefix=fx.prefix)
    assert reconstructed == set(fx.files)  # EXACT — nothing lost, nothing extra


# ---------------------------------------------------------------------------
# Cap probe (§3.6: A/B/C, C4, C17)
# ---------------------------------------------------------------------------


AT_CAP = repack.HF_FILE_CAP - 1  # r2 Codex-M4: the probe only runs when fresh+1 == cap


def test_probe_a_b_c_composition():
    """Happy path: A adds, B is pinned on the post-A HEAD (net zero), C nets -1."""
    fake = FakeHubRepo({"seed/x.json": b"{}"})
    verdict = repack.run_cap_probe(
        fake,
        repo_id=FAKE_REPO,
        expected_live_count=AT_CAP,
        count_fn=lambda: AT_CAP,
        sleep_fn=lambda s: None,
    )
    assert verdict["route"] == "confirmed"
    assert verdict["hypothesis_confirmed"] is True
    assert [c["probe"] for c in verdict["commits"]] == ["A", "B", "C"]
    assert len(fake.calls) == 3
    sha_after_a = f"{1:040x}"
    assert fake.calls[1]["parent"] == sha_after_a  # B pinned on the post-A HEAD
    assert fake.calls[1]["dels"] == [f"{repack.PROBE_PREFIX}/probe_a.txt"]
    # net zero across A+B+C minus the final -1: no probe file survives.
    assert not any(p.startswith(repack.PROBE_PREFIX) for p in fake.files)
    assert repack.cap_probe_rc(verdict) == 0


def test_probe_c4_count_drift_aborts_with_zero_commits():
    """C4: live-count drift aborts the round for RECOMPUTATION — zero commits."""
    fake = FakeHubRepo({"seed/x.json": b"{}"})
    verdict = repack.run_cap_probe(
        fake,
        repo_id=FAKE_REPO,
        expected_live_count=AT_CAP,
        count_fn=lambda: AT_CAP - 1,
        sleep_fn=lambda s: None,
    )
    assert verdict["route"] == "recompute-aborted"
    assert fake.calls == []
    assert repack.cap_probe_rc(verdict) == repack.RC_CAP_PROBE_UNSETTLED


def test_probe_refuses_off_cap_with_zero_commits():
    """r2 Codex-M4: below the cap every commit is accepted regardless of at-cap
    semantics, so the probe REFUSES (zero commits, rc=25) instead of minting a
    vacuous 'confirmed' verdict."""
    fake = FakeHubRepo({"seed/x.json": b"{}"})
    verdict = repack.run_cap_probe(
        fake,
        repo_id=FAKE_REPO,
        expected_live_count=1,  # matches the fresh count, but nowhere near the cap
        count_fn=lambda: 1,
        sleep_fn=lambda s: None,
    )
    assert verdict["route"] == "refused-off-cap"
    assert verdict["hypothesis_confirmed"] is None
    assert fake.calls == []
    assert repack.cap_probe_rc(verdict) == repack.RC_CAP_PROBE_UNSETTLED == 25


def test_probe_b_rejection_routes_net_negative_before_invalidation():
    """C17: a rejected net-zero B routes to the net-NEGATIVE real-unit probe
    (and cleans up probe_a) BEFORE any invalidation verdict — and the route is
    NON-SUCCESS (rc=25): it NAMES the real-unit follow-up, it does not
    dispatch it (r2 Codex-M5)."""
    fake = FakeHubRepo({"seed/x.json": b"{}"})
    fake.script = ["ok", "filecount", "ok"]  # A ok, B rejected, cleanup ok
    verdict = repack.run_cap_probe(
        fake,
        repo_id=FAKE_REPO,
        expected_live_count=AT_CAP,
        count_fn=lambda: AT_CAP,
        sleep_fn=lambda s: None,
    )
    assert verdict["route"] == "commit-b-rejected-net-negative-real-unit-probe"
    assert verdict["hypothesis_confirmed"] is None  # NOT invalidated yet (C17)
    assert [c["probe"] for c in verdict["commits"]] == ["A", "cleanup-a"]
    assert not any(p.startswith(repack.PROBE_PREFIX) for p in fake.files)
    assert repack.cap_probe_rc(verdict) == repack.RC_CAP_PROBE_UNSETTLED


def test_probe_a_rejection_makes_real_unit_the_probe():
    fake = FakeHubRepo({"seed/x.json": b"{}"})
    fake.script = ["filecount"]
    verdict = repack.run_cap_probe(
        fake,
        repo_id=FAKE_REPO,
        expected_live_count=AT_CAP,
        count_fn=lambda: AT_CAP,
        sleep_fn=lambda s: None,
    )
    assert verdict["route"] == "commit-a-rejected-real-unit-probe"
    assert len(fake.calls) == 1
    assert not any(p.startswith(repack.PROBE_PREFIX) for p in fake.files)
    assert repack.cap_probe_rc(verdict) == repack.RC_CAP_PROBE_UNSETTLED


def test_probe_a_b_window_drift_cleans_up_and_aborts():
    """r2 g3-M3: a foreign commit landing between A and B changes the count the
    hypothesis assumes — B is pinned on A's sha, the drift is detected BEFORE
    B composes, probe_a is cleaned up, and the route is non-success (rc=25)."""
    fake = FakeHubRepo({"seed/x.json": b"{}"})

    def foreign_commit_during_sleep(_s: float) -> None:
        # First sleep = the A->B pacing window; land a foreign commit there.
        if "foreign/new.txt" not in fake.files:
            fake.files["foreign/new.txt"] = b"landed between A and B"
            fake.commits += 1  # advances the head sha past A's

    verdict = repack.run_cap_probe(
        fake,
        repo_id=FAKE_REPO,
        expected_live_count=AT_CAP,
        count_fn=lambda: AT_CAP,
        sleep_fn=foreign_commit_during_sleep,
    )
    assert verdict["route"] == "recompute-aborted-a-b-window"
    assert "window_drift" in verdict
    assert [c["probe"] for c in verdict["commits"]] == ["A", "cleanup-a"]
    assert not any(p.startswith(repack.PROBE_PREFIX) for p in fake.files)
    assert fake.files["foreign/new.txt"]  # the foreign commit is untouched
    assert repack.cap_probe_rc(verdict) == repack.RC_CAP_PROBE_UNSETTLED


# ---------------------------------------------------------------------------
# Post-verify (I13c / I10 / C16 / C20)
# ---------------------------------------------------------------------------


def _landed(tmp_path, files=None):
    fx = _build_fixture(tmp_path, files=files)
    fake = FakeHubRepo(dict(fx.files))
    _run_full(fx, fake, tmp_path)
    return fx, fake


@pytest.mark.parametrize(
    "artifact",
    ["shard", "index-part", packing.INDEX_NAME, packing.UNITS_JOURNAL_NAME, packing.MANIFEST_NAME],
)
def test_postverify_blob_mismatch_fails_before_cleanup(tmp_path, artifact):
    """I13(c)+I10: ANY landed-artifact content mismatch fails BEFORE cleanup."""
    fx, fake = _landed(tmp_path)
    shards, groups, _man = repack.load_shard_infos(fx.pack_dir)
    packed = f"{fx.prefix}/{packing.PACKED_DIRNAME}"
    if artifact == "shard":
        victim = f"{packed}/{shards[0].name}"
    elif artifact == "index-part":
        victim = f"{packed}/{next(iter(groups.values()))['index_files'][0]}"
    else:
        victim = f"{packed}/{artifact}"
    fake.files[victim] = b"CORRUPTED REMOTE BYTES"
    reaped: list[Path] = []
    with pytest.raises(repack.AbortPrefix, match=r"I13\(c\)"):
        repack.postverify_prefix(
            fake,
            repo_id=FAKE_REPO,
            prefix=fx.prefix,
            pack_dir=fx.pack_dir,
            reap_paths=[fx.stage_root],
            rm_fn=reaped.append,
        )
    assert reaped == []  # I10: cleanup NEVER ran
    assert fx.stage_root.exists()


def test_postverify_passes_then_reaps_and_c20_tolerates(tmp_path):
    """C20: pre-existing non-v2 entries tolerated; v2-SHAPED foreigners flagged;
    I10: reap happens only after the full verify passes."""
    fx, fake = _landed(tmp_path)
    packed = f"{fx.prefix}/{packing.PACKED_DIRNAME}"
    fake.files[f"{packed}/legacy_note.txt"] = b"pre-existing non-v2"
    fake.files[f"{packed}/zz_foreign.shard99.jsonl"] = b"foreign v2-shaped"
    reaped: list[Path] = []
    report = repack.postverify_prefix(
        fake,
        repo_id=FAKE_REPO,
        prefix=fx.prefix,
        pack_dir=fx.pack_dir,
        reap_paths=[fx.stage_root],
        rm_fn=reaped.append,
    )
    assert report["unexpected_nonv2"] == 1
    assert report["unexpected_v2_shaped"] == [f"{packed}/zz_foreign.shard99.jsonl"]
    assert reaped == [fx.stage_root]


def test_postverify_lingering_source_fails_c16(tmp_path):
    fx, fake = _landed(tmp_path)
    lingering = sorted(fx.files)[0]
    fake.files[lingering] = fx.files[lingering]  # resurrect one packed source
    with pytest.raises(repack.AbortPrefix, match="C16"):
        repack.postverify_prefix(fake, repo_id=FAKE_REPO, prefix=fx.prefix, pack_dir=fx.pack_dir)


# ---------------------------------------------------------------------------
# Consumer gate (I17)
# ---------------------------------------------------------------------------


def test_consumer_gate_blocks_rc22(tmp_path):
    inv = {
        "version": 1,
        "consumers": [
            {
                "script": "scripts/issue1090_fu3_yield_replay.py",
                "prefixes": ["issue1090_partial"],
                "silent_empty": True,
                "migrated": False,
            }
        ],
    }
    with pytest.raises(repack.ConsumerGateBlocked) as exc:
        repack.consumer_gate(inv, "issue1090_partial")
    assert exc.value.rc == 22
    inv["consumers"][0]["migrated"] = True
    assert repack.consumer_gate(inv, "issue1090_partial")["blockers"] == 0
    # scoped-elsewhere consumers never block this prefix
    inv["consumers"][0]["migrated"] = False
    assert repack.consumer_gate(inv, "issue1434_writingstyle")["blockers"] == 0
    # a MISSING inventory fails CLOSED
    with pytest.raises(repack.ConsumerGateBlocked, match="failing CLOSED"):
        repack.load_consumer_inventory(tmp_path / "nope.json")


# ---------------------------------------------------------------------------
# Token bucket (C6)
# ---------------------------------------------------------------------------


def test_token_bucket_paces_with_injected_clock():
    now = [0.0]
    sleeps: list[float] = []

    def clock():
        return now[0]

    def sleep(s):
        sleeps.append(s)
        now[0] += s

    bucket = repack.TokenBucket(10.0, clock=clock, sleep=sleep)  # 0.1s per request
    for _ in range(5):
        bucket.acquire(2)
    # 5 acquisitions of 2 requests at 10 req/s: the first is free, every
    # later one waits 0.2s (2 requests x 0.1s interval).
    assert sleeps == pytest.approx([0.2, 0.2, 0.2, 0.2], abs=1e-9)


# ---------------------------------------------------------------------------
# Classifier edges
# ---------------------------------------------------------------------------


def test_error_classifiers_disjoint():
    e412 = FakeHTTPError(412, "A commit has happened since")
    e429 = FakeHTTPError(429, "Too Many Requests", retry_after="90")
    e504 = FakeHTTPError(504, "Gateway Time-out")
    assert repack.is_parent_conflict(e412) and not repack.is_ambiguous_outcome(e412)
    assert repack.is_rate_limit(e429) and not repack.is_ambiguous_outcome(e429)
    assert repack.is_ambiguous_outcome(e504)
    assert repack.is_ambiguous_outcome(TimeoutError("read timed out"))
    assert repack.retry_after_seconds(e429) == 90.0
    assert repack.retry_after_seconds(e504) == 30.0  # floor
    filecount = FakeHTTPError(
        400, "would contain 1000001 files after this push, over the limit of 1000000 files"
    )
    assert repack._is_file_count_limit(filecount)
    assert not repack.is_ambiguous_outcome(filecount)


def test_journal_and_delete_digest_deterministic():
    assert repack.delete_set_digest(["b", "a"]) == repack.delete_set_digest(["a", "b"])
    rec = {"unit_id": 1, "z": "ü"}
    assert repack.journal_bytes([rec]) == repack.journal_bytes([json.loads(json.dumps(rec))])
    assert hashlib.sha256(repack.journal_bytes([rec])).hexdigest()  # bytes, newline-terminated
    assert repack.journal_bytes([rec]).endswith(b"\n")


# ---------------------------------------------------------------------------
# r2 C1: I5 verify receipt gates the commit phase
# ---------------------------------------------------------------------------


def test_verify_receipt_gate_function_level(tmp_path):
    """No receipt => refuse; minted => pass; pack mutated after => STALE;
    foreign version => refuse."""
    fx = _build_fixture(tmp_path, verify=False)
    with pytest.raises(repack.AbortPrefix, match="no verify receipt"):
        repack.check_verify_receipt(fx.pack_dir)
    repack.verify_prefix(
        prefix=fx.prefix,
        pack_dir=fx.pack_dir,
        stage_root=fx.stage_root,
        scratch_dir=tmp_path / "scratch",
        candidate_paths=fx.candidates,
    )
    doc = repack.check_verify_receipt(fx.pack_dir)
    assert doc["census_key"] and doc["index_parts_sha256"]
    # Mutate an index part AFTER verify: the receipt is now stale.
    _tamper_index_part(
        fx.pack_dir,
        lambda members: members.update(
            {next(iter(members)): {**members[next(iter(members))], "offset": 999}}
        ),
    )
    with pytest.raises(repack.AbortPrefix, match="STALE"):
        repack.check_verify_receipt(fx.pack_dir)
    # Foreign receipt version refuses too.
    path = repack.verify_receipt_path(fx.pack_dir)
    bad = json.loads(path.read_text())
    bad["version"] = 99
    path.write_text(json.dumps(bad))
    with pytest.raises(repack.AbortPrefix, match="unrecognized verify-receipt version"):
        repack.check_verify_receipt(fx.pack_dir)


def test_commit_phase_requires_receipt_before_any_hub_call(tmp_path):
    """r2 Codex-C1 (function level): no receipt => AbortPrefix, ZERO Hub calls."""
    fx = _build_fixture(tmp_path, verify=False)
    fake = FakeHubRepo(dict(fx.files))
    with pytest.raises(repack.AbortPrefix, match="no verify receipt"):
        _run_full(fx, fake, tmp_path)
    assert fake.calls == []
    assert fake.files == fx.files  # zero deletes


def test_main_commit_without_receipt_rc23_zero_hub_methods(tmp_path, monkeypatch):
    """r2 Codex-C1 (mechanized recipe): ``main(--phase commit)`` with no
    receipt exits rc=23 with ZERO create_commit / Hub calls of any kind."""
    from huggingface_hub import HfApi

    prefix = "issue1090_partial"  # must be a real target prefix (r2 C2)
    fx = _build_fixture(tmp_path, prefix=prefix, verify=False)
    census_path = tmp_path / "state" / f"{prefix}.census.json"
    entries = [
        SimpleNamespace(
            path=p, size=len(d), is_lfs=False, lfs_sha256=None, blob_id=packing.git_blob_sha1(d)
        )
        for p, d in sorted(fx.files.items())
    ]
    repack.save_census(
        census_path,
        prefix=prefix,
        revision="fixture-rev",
        entries=entries,
        retained={},
        candidates=entries,
        exclusions={},
    )
    api = mock.create_autospec(HfApi, instance=True)
    monkeypatch.setattr("huggingface_hub.HfApi", lambda *a, **k: api)
    # The fresh live scan has its own tests; stub it so this test isolates
    # the receipt gate (and stays off the multi-second full-tree walk).
    monkeypatch.setattr(
        repack, "fresh_consumer_scan_gate", lambda inv, **k: {"fresh_scan_hits": 0, "errors": 0}
    )
    rc = repack.main(
        [
            "--phase",
            "commit",
            "--prefix",
            prefix,
            "--repo-id",
            FAKE_REPO,
            "--work-root",
            str(tmp_path),
        ]
    )
    assert rc == 23  # AbortPrefix: I5 no verify receipt
    assert api.create_commit.call_count == 0
    assert api.method_calls == []


# ---------------------------------------------------------------------------
# r2 C2: prefix scope allowlist (main + run_commit_phase defense-in-depth)
# ---------------------------------------------------------------------------


def test_main_refuses_non_target_prefix_before_any_hub_client(monkeypatch):
    """r2 Codex-C2 (mechanized recipe): an unapproved --prefix refuses BEFORE
    load_dotenv / HfApi construction — for EVERY phase that takes a prefix."""
    import huggingface_hub

    def boom(*a, **k):  # pragma: no cover - would only fire on regression
        raise AssertionError("HfApi constructed for an unapproved prefix")

    monkeypatch.setattr(huggingface_hub, "HfApi", boom)
    for phase in ("walk", "commit", "verify"):
        with pytest.raises(SystemExit, match="not a #2321 target prefix"):
            repack.main(["--phase", phase, "--prefix", "issue9999_unapproved"])


def test_run_commit_phase_defensive_scope_refusal(tmp_path):
    """r2 Codex-C2 defense-in-depth: even called DIRECTLY (bypassing main),
    a non-target prefix must never reach delete composition against the
    canonical repo — api=None proves nothing was touched."""
    fx = _build_fixture(tmp_path)  # PREFIX is not a #2321 target
    with pytest.raises(repack.AbortPrefix, match="not a #2321 target prefix"):
        repack.run_commit_phase(
            None,
            repo_id=repack.DEFAULT_REPO_ID,
            prefix=fx.prefix,
            pack_dir=fx.pack_dir,
            census=fx.census,
            retained={},
            revision="fixture-rev",
            dry_run=True,
        )
    # Fixture/staging repos stay testable: same call against a fake repo id
    # proceeds (dry-run) — the defense is scoped to the CANONICAL repo.
    summary = repack.run_commit_phase(
        None,
        repo_id=FAKE_REPO,
        prefix=fx.prefix,
        pack_dir=fx.pack_dir,
        census=fx.census,
        retained={},
        revision="fixture-rev",
        dry_run=True,
    )
    assert summary["status"] == "committed"


# ---------------------------------------------------------------------------
# r2 C4: the local journal is a HINT — content-anchored re-probe before skip
# ---------------------------------------------------------------------------


def test_resume_landed_hint_reprobes_remote_regression(tmp_path):
    """r2 Codex-C4 (mechanized recipe): a locally-journaled 'landed' unit whose
    shard has VANISHED from the Hub aborts on resume — never skipped, zero new
    commits over the hole."""
    fx = _build_fixture(tmp_path)
    fake = FakeHubRepo(dict(fx.files))
    _run_full(fx, fake, tmp_path)
    n_calls = len(fake.calls)
    shards, _g, _m = repack.load_shard_infos(fx.pack_dir)
    packed = f"{fx.prefix}/{packing.PACKED_DIRNAME}"
    del fake.files[f"{packed}/{shards[0].name}"]  # remote regressed
    with pytest.raises(repack.AbortPrefix, match="journal claims landed but the Hub"):
        _run_full(fx, fake, tmp_path)
    assert len(fake.calls) == n_calls  # zero new create_commit calls


# ---------------------------------------------------------------------------
# r2 M3: finalize rewrites the remote journal to the LOCAL bytes
# ---------------------------------------------------------------------------


def test_finalize_rewrites_journal_after_crash_before_append(tmp_path, monkeypatch):
    """r2 Codex-M3: crash between a unit's server landing and its local append
    => the resumed run REGENERATES the record under a different clock; the
    finalize journal-rewrite makes remote == local so postverify PASSes."""
    files = _fixture_files(PREFIX, n_a=14, n_b=8)
    fx = _build_fixture(tmp_path, files=files)
    fake = FakeHubRepo(dict(fx.files))
    _run_full(fx, fake, tmp_path, ops_cap=18)  # >=2 units + finalize
    packed = f"{fx.prefix}/{packing.PACKED_DIRNAME}"
    journal_path = fx.pack_dir / packing.UNITS_JOURNAL_NAME
    lines = [ln for ln in journal_path.read_text(encoding="utf-8").split("\n") if ln.strip()]
    assert len(lines) >= 2
    # Crash simulation: the LAST unit's local append never happened, and the
    # finalize commit never ran (its manifest is absent remotely).
    journal_path.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
    del fake.files[f"{packed}/{packing.MANIFEST_NAME}"]
    # The resumed process runs under a DIFFERENT clock => regenerated record
    # ts differs from the remote journal's copy of that record.
    monkeypatch.setattr(repack, "_now_iso", lambda: "2099-01-01T00:00:00+00:00")
    summary = _run_full(fx, fake, tmp_path, ops_cap=18)
    assert summary["status"] == "committed"
    local_bytes = repack.journal_bytes(repack.load_local_journal(fx.pack_dir))
    assert fake.files[f"{packed}/{packing.UNITS_JOURNAL_NAME}"] == local_bytes
    assert b"2099-01-01" in local_bytes  # the regenerated record really differs
    # postverify compares like with like — the r1 shape failed here forever.
    report = repack.postverify_prefix(
        fake, repo_id=FAKE_REPO, prefix=fx.prefix, pack_dir=fx.pack_dir
    )
    assert report["n_verified"] > 0


# ---------------------------------------------------------------------------
# r2 M2: resume-bootstrap — rebuild local resume state from the REMOTE record
# ---------------------------------------------------------------------------


class _RevDirApi(_DirApi):
    """_DirApi with per-revision roots (deleted originals resolve at R)."""

    def __init__(self, root: Path, rev_roots: dict | None = None):
        super().__init__(root)
        self.rev_roots = {k: Path(v) for k, v in (rev_roots or {}).items()}

    def _root_for(self, revision):
        return self.rev_roots.get(revision, self.root)

    def file_exists(self, repo_id, filename, *, repo_type="dataset", revision=None):
        return (self._root_for(revision) / filename).is_file()

    def list_repo_tree(
        self,
        repo_id=None,
        path_in_repo=None,
        *,
        repo_type="dataset",
        revision=None,
        recursive=False,
        expand=None,
    ):
        root = self._root_for(revision)
        base = root / (path_in_repo or "")
        if not base.exists():
            raise EntryNotFoundError(f"Entry Not Found: {path_in_repo}")
        for p in sorted(base.rglob("*")):
            if p.is_file():
                rel = p.relative_to(root).as_posix()
                yield RepoFile(
                    path=rel, size=p.stat().st_size, oid=packing.git_blob_sha1(p.read_bytes())
                )


def test_resume_bootstrap_end_to_end(tmp_path, monkeypatch):
    """r2 Codex-M2: after a TOTAL local wipe, resume-bootstrap stages the
    remote journal, re-walks at the pinned revision R, cross-checks the MF3
    reconstruction, and restores the journal bytes VERBATIM."""
    files = _fixture_files(PREFIX, n_a=14, n_b=8)
    fx = _build_fixture(tmp_path, files=files)
    fake = FakeHubRepo(dict(fx.files))
    _run_full(fx, fake, tmp_path, ops_cap=18)
    packed = f"{fx.prefix}/{packing.PACKED_DIRNAME}"
    remote_journal = fake.files[f"{packed}/{packing.UNITS_JOURNAL_NAME}"]
    head_root = _dump_fake_to_dir(fake, tmp_path / "remote_head")
    r_root = tmp_path / "remote_at_R"
    for path, data in fx.files.items():  # the tree as it stood at revision R
        p = r_root / path
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
    _shim_remote(monkeypatch, head_root)
    api = _RevDirApi(head_root, {"fixture-rev": r_root})
    work2 = tmp_path / "resume_work"
    census_path = work2 / "state" / f"{fx.prefix}.census.json"
    report = repack.run_resume_bootstrap(
        api, repo_id=FAKE_REPO, prefix=fx.prefix, work=work2, census_path=census_path
    )
    assert report["revision"] == "fixture-rev"
    assert report["n_landed_units"] == len(repack.load_local_journal(fx.pack_dir))
    # journal restored byte-verbatim into the fresh work root
    assert (work2 / "pack" / fx.prefix / packing.UNITS_JOURNAL_NAME).read_bytes() == remote_journal
    # census reproduced at R
    doc = repack.load_census(census_path)
    assert doc["revision"] == "fixture-rev"
    assert set(doc["anchors"]) == set(fx.files)


def test_resume_bootstrap_post_r_drift_aborts(tmp_path, monkeypatch):
    """MF3/I7: a file added under the prefix AFTER revision R fails the
    reconstruction cross-check — never resumed over."""
    files = _fixture_files(PREFIX, n_a=14, n_b=8)
    fx = _build_fixture(tmp_path, files=files)
    fake = FakeHubRepo(dict(fx.files))
    _run_full(fx, fake, tmp_path, ops_cap=18)
    head_root = _dump_fake_to_dir(fake, tmp_path / "remote_head")
    drift = head_root / fx.prefix / "added_after_R.json"
    drift.write_bytes(b"{}")
    r_root = tmp_path / "remote_at_R"
    for path, data in fx.files.items():
        p = r_root / path
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
    _shim_remote(monkeypatch, head_root)
    api = _RevDirApi(head_root, {"fixture-rev": r_root})
    with pytest.raises(repack.AbortPrefix, match="cross-check FAILED"):
        repack.run_resume_bootstrap(
            api,
            repo_id=FAKE_REPO,
            prefix=fx.prefix,
            work=tmp_path / "w2",
            census_path=tmp_path / "w2" / "state" / "c.json",
        )


def test_resume_bootstrap_without_remote_journal_aborts(tmp_path, monkeypatch):
    """No remote journal => nothing landed => run the ordinary phases fresh."""
    root = tmp_path / "remote"
    for path, data in _fixture_files(PREFIX).items():
        p = root / path
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
    _shim_remote(monkeypatch, root)
    with pytest.raises(repack.AbortPrefix, match="nothing landed"):
        repack.run_resume_bootstrap(
            None,  # never reached: the journal staging aborts first
            repo_id=FAKE_REPO,
            prefix=PREFIX,
            work=tmp_path / "w",
            census_path=tmp_path / "w" / "c.json",
        )


def test_resume_bootstrap_validates_remote_journal(tmp_path, monkeypatch):
    """Non-contiguous unit ids / mixed revisions in the remote journal abort."""
    root = tmp_path / "remote"
    packed_dir = root / PREFIX / packing.PACKED_DIRNAME
    packed_dir.mkdir(parents=True)
    base = {"census_key": "ck", "revision": "r1"}
    rows = [{"unit_id": 1, **base}, {"unit_id": 3, **base}]
    (packed_dir / packing.UNITS_JOURNAL_NAME).write_bytes(repack.journal_bytes(rows))
    _shim_remote(monkeypatch, root)
    with pytest.raises(repack.AbortPrefix, match="not contiguous"):
        repack.run_resume_bootstrap(
            None,
            repo_id=FAKE_REPO,
            prefix=PREFIX,
            work=tmp_path / "w1",
            census_path=tmp_path / "c1",
        )
    rows = [{"unit_id": 1, **base}, {"unit_id": 2, "census_key": "ck", "revision": "r2"}]
    (packed_dir / packing.UNITS_JOURNAL_NAME).write_bytes(repack.journal_bytes(rows))
    from explore_persona_space.orchestrate import hub

    hub.clear_packed_caches()
    with pytest.raises(repack.AbortPrefix, match="spans 2 revisions"):
        repack.run_resume_bootstrap(
            None,
            repo_id=FAKE_REPO,
            prefix=PREFIX,
            work=tmp_path / "w2",
            census_path=tmp_path / "c2",
        )


# ---------------------------------------------------------------------------
# r2 M7: state-upload cap rejection is a GLOBAL STOP
# ---------------------------------------------------------------------------


def test_state_upload_cap_rejection_raises_stop_repack(tmp_path):
    """r2 Codex-M7: a file-count rejection on the STATE upload propagates as
    StopRepack — the run must not keep going on an at-cap repo."""

    class _CapApi:
        def upload_file(self, **kw):
            raise FakeHTTPError(
                400, "would contain too many files after this push, over the limit of 1000000"
            )

    state = tmp_path / "s.jsonl"
    state.write_text("{}\n", encoding="utf-8")
    with pytest.raises(repack.StopRepack, match="state upload"):
        repack.upload_state_file(
            _CapApi(), repo_id=FAKE_REPO, state_path=state, prefix="issue1090_partial"
        )


# ---------------------------------------------------------------------------
# r2 g3-M4: poller-conformant sentinel envelope
# ---------------------------------------------------------------------------


def _load_poll_pipeline():
    spec = importlib.util.spec_from_file_location(
        "poll_pipeline_for_i2321", REPO_ROOT / "scripts" / "poll_pipeline.py"
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_write_sentinel_poller_conformant_envelope(tmp_path, monkeypatch):
    """r2 g3-M4: the sentinel carries EVERY key poll_pipeline requires (the
    prior bare-payload writer produced files the poller silently dropped),
    at the documented issue-<N>-<kind_slug>-<epoch>.json filename."""
    import re as _re

    monkeypatch.setenv("EPM_I2321_SENTINEL_DIR", str(tmp_path))
    payload = {"issue": 2321, "phase": "commit", "prefix": "issue1090_partial"}
    path = repack.write_sentinel(payload, kind="epm:progress")
    assert path is not None and path.parent == tmp_path
    assert _re.fullmatch(r"issue-2321-progress-\d+\.json", path.name)
    doc = json.loads(path.read_text(encoding="utf-8"))
    pp = _load_poll_pipeline()
    for key in pp._SENTINEL_REQUIRED_KEYS:
        assert key in doc, f"poller-required key {key!r} missing from the envelope"
    assert doc["sentinel_schema_version"] == repack.SENTINEL_SCHEMA_VERSION
    assert doc["kind"] == "epm:progress" and doc["version"] == 1 and doc["task_id"] == 2321
    assert json.loads(doc["note"]) == payload
    # No sentinel surface => explicit None, never a crash.
    monkeypatch.delenv("EPM_I2321_SENTINEL_DIR")
    if not Path("/workspace").is_dir():
        assert repack.write_sentinel(payload) is None


# ---------------------------------------------------------------------------
# r2 M1: inventory schema validation + commit-admission fresh scan
# ---------------------------------------------------------------------------


def _inv_path(tmp_path: Path, text: str, name: str = "inv.json") -> Path:
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return p


def test_load_consumer_inventory_schema_battery(tmp_path):
    """r2 Codex-M1 / g5-M1: EVERY malformed inventory shape fails CLOSED —
    including the hand-authored '"migrated": "false"' string-boolean that the
    old truthiness check silently read as migrated."""
    good_row = {
        "script": "scripts/x.py",
        "prefixes": ["issue1090_partial"],
        "silent_empty": True,
        "migrated": True,
    }
    tp = sorted(repack.PREFIX_ORDER)
    ok = {"version": 1, "target_prefixes": tp, "consumers": [good_row]}
    assert repack.load_consumer_inventory(_inv_path(tmp_path, json.dumps(ok)))["consumers"]
    cases = [
        ("this is not json {", "not valid JSON"),
        (json.dumps({"version": 1, "target_prefixes": tp, "consumers": []}), "ZERO consumers"),
        (json.dumps({"version": 1, "consumers": [good_row]}), "target_prefixes"),
        (
            json.dumps(
                {
                    "version": 1,
                    "target_prefixes": ["issue1090_partial"],
                    "consumers": [good_row],
                }
            ),
            "target_prefixes",
        ),
        (
            json.dumps(
                {
                    "version": 1,
                    "target_prefixes": tp,
                    "consumers": [{**good_row, "migrated": "false"}],
                }
            ),
            "migrated must be a bool",
        ),
        (
            json.dumps(
                {
                    "version": 1,
                    "target_prefixes": tp,
                    "consumers": [{**good_row, "prefixes": []}],
                }
            ),
            "prefixes must be a non-empty",
        ),
        (
            json.dumps({"version": 1, "target_prefixes": tp, "consumers": ["not-an-object"]}),
            "not an object",
        ),
    ]
    for i, (text, match) in enumerate(cases):
        with pytest.raises(repack.ConsumerGateBlocked, match=match):
            repack.load_consumer_inventory(_inv_path(tmp_path, text, f"bad{i}.json"))
    # The hazard the validation closes: gate SEMANTICS read a truthy string
    # as migrated, so without load-time refusal the row authorizes deletion.
    hazard = {"consumers": [{**good_row, "migrated": "false", "silent_empty": True}]}
    assert repack.consumer_gate(hazard, "issue1090_partial")["blockers"] == 0


def test_committed_inventory_passes_new_schema():
    """The COMMITTED inventory satisfies the strict schema (calibration on the
    real artifact, not only fixtures)."""
    doc = repack.load_consumer_inventory(
        REPO_ROOT / "scripts" / "issue2321_consumer_inventory.json"
    )
    assert set(doc["target_prefixes"]) == set(repack.PREFIX_ORDER)
    assert len(doc["consumers"]) >= 20


def test_fresh_consumer_scan_gate_blocks_uncovered_consumer(tmp_path):
    """r2 Codex-M1: a consumer added AFTER curation is caught by the fresh
    scan at commit admission; a clean tree passes."""
    inv = json.loads(
        (REPO_ROOT / "scripts" / "issue2321_consumer_inventory.json").read_text(encoding="utf-8")
    )
    root = tmp_path / "fixrepo"
    sd = root / "scripts"
    sd.mkdir(parents=True)
    (sd / "late_consumer.py").write_text(
        'def go(api):\n    return api.list_repo_tree("r", path_in_repo="issue1090_partial")\n',
        encoding="utf-8",
    )
    with pytest.raises(repack.ConsumerGateBlocked, match="FRESH scan"):
        repack.fresh_consumer_scan_gate(inv, repo_root=root)
    clean = tmp_path / "clean"
    clean.mkdir()
    ok = repack.fresh_consumer_scan_gate(inv, repo_root=clean)
    assert ok["errors"] == 0


# ---------------------------------------------------------------------------
# r2 packing minors: B1 compact index parts, m4 duplicates, m5 interlock scope
# ---------------------------------------------------------------------------


def test_index_parts_written_compact_and_under_cap(tmp_path):
    """r2 g1-B1: the WRITTEN part bytes respect the cap the cost accounting
    sized (the indent=1 write ran ~1.13x over and shipped 9.75-10.16 MB parts
    against the <=9 MB non-LFS contract)."""
    entries = {
        f"dir/member_{i:04d}_{'x' * 80}.json": packing.MemberIndexEntry(
            shard="g.shard00.jsonl",
            offset=i * 100,
            length=100,
            sha256="a" * 64,
            enc="text",
            size=90,
        )
        for i in range(200)
    }
    cap = 8000
    names = packing._write_index_parts(tmp_path, "g", "dir", entries, cap)
    assert len(names) >= 2  # multi-part split actually exercised
    merged: dict = {}
    for n in names:
        blob = (tmp_path / n).read_bytes()
        assert len(blob) <= cap, (n, len(blob), "written part exceeds the sized cap")
        merged.update(packing.load_index_part(blob.decode("utf-8"), what=n))
    assert set(merged) == set(entries)  # lossless across the split


def test_pack_tree_v2_duplicate_candidates_raise(tmp_path):
    """r2 m4: a duplicated candidate would pack one member twice."""
    raw = tmp_path / "raw"
    (raw / "d").mkdir(parents=True)
    (raw / "d" / "f.json").write_bytes(b"{}")
    with pytest.raises(packing.PackError, match="duplicate pack candidates"):
        packing.pack_tree_v2(raw, tmp_path / "pack", candidates=["d/f.json", "d/f.json"])


def test_interlock_covers_overflow_repo():
    """r2 m5: the private overflow repo holds canonical-quality artifacts —
    the I18 interlock refuses test-process mutations of it too."""
    assert "superkaiba1/explore-persona-space-overflow" in packing.CANONICAL_HUB_REPOS
    with pytest.raises(packing.TestMutationInterlockError):
        packing.assert_test_mutation_interlock("superkaiba1/explore-persona-space-overflow")
