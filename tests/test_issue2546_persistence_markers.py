"""Pins for the #2546 fail-silent persistence fixes (task marker v144).

Defect 1 — capture-regime fingerprint persistence:

1b. ``issue2546_fit_cells._capture_marker_fp`` FAILS LOUD on a missing (or
    fingerprint-less) ``_complete.json`` instead of silently yielding ``None``
    — the pre-fix ``else None`` fallback left the r2-Major-7 capture-regime
    component of the fitcache content key inert on every staged production
    fit (the fit-time ``shard_dir`` is the HF mirror once the capture phase's
    upload-then-free has run, and the marker never rode the upload).
1a. ``issue2546_gen_capture._mirror_capture_marker`` uploads the marker to
    ``<store_prefix>/<stem>/_complete.json`` (so ``hub.stage_hub_prefix``
    pulls it into the fit mirror) BEFORE blessing local resume: an upload
    failure leaves NO local marker, so the resume predicate re-runs the stem
    (self-healing) rather than resume-skipping into a mirror that can never
    carry its fingerprint. Both capture writers route through the helper.

Defect 2 — ``issue2546_upload_logs.collect_log_files`` covers exactly the
four realized Hub log groups (dispatcher logs at the arm root, worker logs
under ``logs/``, fit worker logs under ``work/fits_a<N>/``, launcher +
revisions.json + fallbacks env under ``aux/``) and excludes shards / npz /
caches / poller sentinels; ``stage_files`` line-splits oversize plain text
into <9 MB ``.partNNN`` pieces (non-LFS path) losslessly.

Round 18 additions:

r18-1. Mirror-on-skip (``_ensure_capture_marker_hub_twin``): a resume-skip
    over a matching PRE-fix local ``_complete.json`` verifies the Hub twin
    (ONE memoized scoped listing per (arm, smoke) — never a per-stem
    ``file_exists`` loop) and repairs a missing twin through
    ``_mirror_capture_marker``'s upload+assert discipline, fail-loud; both
    capture drivers are AST-pinned to call it.
r18-3. ``stage_files`` hard-splits a single line larger than the part size on
    BYTE boundaries (boundary cases limit-1/limit/limit+1), byte-exact on
    concatenation.
r18-smoke. ``_capture_marker_fp``'s missing-marker refusal is ACTIONABLE for
    pre-fix marker-less ``smoke_arm*`` Hub stores (names the re-run repair,
    forbids Hub deletion).

Round 19 additions (reconciler ruling on body v18 — the marker ATTESTS the
shards are on the Hub, so marker-only success must be UNREACHABLE):

r19-1. ``RepositoryNotFoundError`` at the scoped listing RAISES (dropped from
    the except tuple): ``hub._upload`` downstream resurrects a deleted repo
    via ``create_repo(exist_ok=True)``, so it cannot be trusted to fail loud.
r19-2. The twin repair is gated on remote SHARD presence from the SAME
    pre-filter listing: shards present -> mirror; absent-with-local-shards ->
    full stem-dir upload FIRST (marker excluded) then mirror (recovers the
    ``--skip-upload`` transition); absent everywhere -> RAISE naming the stem.
    An empty remote listing plus a would-be-successful marker upload never
    ends in a bare marker write.

Round 20 additions (reconciler ruling on body v19 — the r18 corruption
survives r19 through the FILE-COUNT OVERFLOW lane):

r20-1. Canonical-destination gate: ``hub._upload``'s default-ON
    ``_filecount_overflow_retry`` returns a TRUTHY ``DEFAULT_OVERFLOW_REPO/...``
    path without raising (even under ``raise_on_error=True``), so truthiness
    cannot gate an attestation. Every attestation-adjacent upload site — the
    repair's stem-dir upload, BOTH write-path store uploads, and the shared
    marker mirror — now requires ``res == f"{DEFAULT_DATASET_REPO}/{dest}"``
    via ``_require_canonical_upload`` (a REAL raise, not a bare assert) BEFORE
    any memo add, marker mirror, shard free, or local-resume bless.
    ``upload_stage`` (rollout JSONLs) stays deliberately un-gated: the
    overflow reroute is the designed durability path for ordinary artifacts
    (#1108/#2304).
r20-2. The fake ``hub._upload`` factories default to ECHOING the canonical
    success return (``f"{repo_id}/{path_in_repo}"``, hub.py ``_upload``);
    explicit ``result=`` simulates the ``""`` verify miss or a truthy
    overflow reroute.
r20-3. Memo monkeypatches carry ``raising=False`` so the r19 shard-gate cases
    reach their BEHAVIORAL assertions when run against the pre-r19 parent
    blob (26f3cdddfe), where ``_HUB_SHARD_STEM_SETS`` does not exist — the
    differential then fails on behavior, not setup AttributeError.

Network-free: the only Hub boundary (``hub._upload`` / ``HfApi``) is faked
with signature-mirroring fakes; every other body runs for real on tmp_path.
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import pytest

# Feature-neutral overflow-repo constant (r20 differential): imported from hub
# — which defines it on BOTH sides of the fix — never from G, where the import
# is round-20-added and a pre-fix run would die on setup AttributeError.
from explore_persona_space.orchestrate.hub import (
    DEFAULT_OVERFLOW_REPO as OVERFLOW_REPO,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

import issue2546_fit_cells as F  # noqa: E402
import issue2546_gen_capture as G  # noqa: E402
import issue2546_upload_logs as UL  # noqa: E402

# ---------------------------------------------------------------------------
# Defect 1b — missing capture marker fails loud at fit time
# ---------------------------------------------------------------------------


class TestCaptureMarkerFp:
    def test_missing_marker_raises(self, tmp_path):
        # Pre-fix behavior was `capture_fp = ... if capture_done.is_file() else None`.
        with pytest.raises(RuntimeError):
            F._capture_marker_fp(tmp_path, "post__gsm8k")

    def test_marker_without_fingerprint_raises(self, tmp_path):
        (tmp_path / "_complete.json").write_text(json.dumps({"report": {"stem": "x"}}))
        with pytest.raises(RuntimeError):
            F._capture_marker_fp(tmp_path, "post__gsm8k")

    def test_marker_with_fingerprint_returns_it(self, tmp_path):
        fp = {"stage": "capture", "row_sig": "abc123", "smoke": False}
        (tmp_path / "_complete.json").write_text(json.dumps({"fingerprint": fp}))
        assert F._capture_marker_fp(tmp_path, "post__gsm8k") == fp

    def test_missing_marker_under_smoke_prefix_names_the_repair(self, tmp_path):
        """r18: a pre-fix Hub SMOKE store (smoke_arm2: 21 stems, 0 markers) hits
        the fail-loud reader — the refusal must name the known state + repair
        (re-run the smoke capture on fixed code), never suggest Hub deletion."""
        shard_dir = tmp_path / "store" / "smoke_arm2" / "post__gsm8k"
        shard_dir.mkdir(parents=True)
        with pytest.raises(RuntimeError) as ei:
            F._capture_marker_fp(shard_dir, "post__gsm8k")
        msg = str(ei.value)
        assert "smoke_arm" in msg
        assert "Do NOT delete" in msg
        assert "re-run the smoke capture" in msg

    def test_missing_marker_production_message_has_no_smoke_hint(self, tmp_path):
        shard_dir = tmp_path / "store" / "arm1" / "post__gsm8k"
        shard_dir.mkdir(parents=True)
        with pytest.raises(RuntimeError) as ei:
            F._capture_marker_fp(shard_dir, "post__gsm8k")
        msg = str(ei.value)
        assert "smoke_arm* prefix" not in msg
        assert "Restore or re-upload the marker" in msg

    def test_build_fitcache_reader_dispatches_the_failloud_helper(self):
        """The live build_fitcache body assigns capture_fp from the fail-loud
        helper — never from the pre-fix `... if capture_done.is_file() else None`
        conditional expression (hollow-gate guard: the dispatched path is the
        gated path)."""
        import inspect
        import textwrap

        tree = ast.parse(textwrap.dedent(inspect.getsource(F.build_fitcache)))
        assigns = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "capture_fp" for t in node.targets)
        ]
        assert assigns, "build_fitcache no longer assigns capture_fp"
        for node in assigns:
            assert isinstance(node.value, ast.Call), ast.dump(node.value)
            fn = node.value.func
            name = fn.id if isinstance(fn, ast.Name) else getattr(fn, "attr", None)
            assert name == "_capture_marker_fp", ast.dump(node.value)


# ---------------------------------------------------------------------------
# Defect 1a — the marker is part of the uploaded set, Hub-first ordering
# ---------------------------------------------------------------------------


# Default fake result: mirror hub._upload's canonical-success contract
# (`return f"{repo_id}/{path_in_repo}"`). Pass an explicit result= to simulate
# the "" 0-committed-files verify miss or a truthy overflow reroute (r20).
_ECHO_CANONICAL = object()


def _fake_upload_factory(captured: dict, *, result=_ECHO_CANONICAL, exc: Exception | None = None):
    """Signature-mirroring fake of hub._upload (network boundary only)."""

    def fake_upload(
        local_path,
        repo_id,
        repo_type,
        path_in_repo,
        delete_after=False,
        upload_as_file=False,
        ignore_patterns=None,
        private=False,
        raise_on_error=False,
    ):
        captured["local_path"] = Path(local_path)
        captured["bytes"] = Path(local_path).read_bytes()
        captured["repo_id"] = repo_id
        captured["repo_type"] = repo_type
        captured["path_in_repo"] = path_in_repo
        captured["upload_as_file"] = upload_as_file
        captured["raise_on_error"] = raise_on_error
        if exc is not None:
            raise exc
        if result is _ECHO_CANONICAL:
            return f"{repo_id}/{path_in_repo}"
        return result

    return fake_upload


class TestMirrorCaptureMarker:
    def test_marker_uploaded_to_stem_prefix_then_blessed_locally(self, tmp_path, monkeypatch):
        stem_dir = tmp_path / "store" / "arm1" / "post__gsm8k"
        stem_dir.mkdir(parents=True)
        done_p = stem_dir / "_complete.json"
        dest = "issue2546_cotmap/analysis_tensors/thinkstore/arm1/post__gsm8k"
        payload = {"fingerprint": {"stage": "capture"}, "report": {"stem": "post__gsm8k"}}
        captured: dict = {}
        monkeypatch.setattr(G, "_upload", _fake_upload_factory(captured))

        G._mirror_capture_marker(done_p, dest, payload)

        # The marker path appears in what would be uploaded, at the SAME
        # prefix _stage_stem stages (so stage_hub_prefix mirrors it).
        assert captured["path_in_repo"] == f"{dest}/_complete.json"
        assert captured["repo_id"] == G.DEFAULT_DATASET_REPO
        assert captured["repo_type"] == "dataset"
        assert captured["upload_as_file"] is True
        assert captured["raise_on_error"] is True
        # Uploaded bytes == the blessed local marker's bytes.
        assert json.loads(captured["bytes"]) == payload
        assert json.loads(done_p.read_text()) == payload
        # Temp staging never lands inside the stem dir (a crashed attempt's
        # residue must not ride a later shard folder-upload); it is renamed
        # away on success.
        assert not captured["local_path"].exists()
        assert captured["local_path"].parent != stem_dir
        assert list(stem_dir.iterdir()) == [done_p]

    def test_upload_failure_leaves_no_local_marker(self, tmp_path, monkeypatch):
        """Hub-FIRST ordering: a failed marker upload must NOT bless resume."""
        stem_dir = tmp_path / "store" / "arm1" / "post__gsm8k"
        stem_dir.mkdir(parents=True)
        done_p = stem_dir / "_complete.json"
        captured: dict = {}
        monkeypatch.setattr(
            G, "_upload", _fake_upload_factory(captured, exc=RuntimeError("hub down"))
        )
        with pytest.raises(RuntimeError):
            G._mirror_capture_marker(done_p, "prefix/stem", {"fingerprint": {}})
        assert not done_p.exists()

    def test_empty_upload_result_leaves_no_local_marker(self, tmp_path, monkeypatch):
        """The "" 0-committed-files verify miss now trips the canonical gate —
        a REAL RuntimeError (r20; a bare assert is stripped under -O)."""
        stem_dir = tmp_path / "s"
        stem_dir.mkdir()
        done_p = stem_dir / "_complete.json"
        captured: dict = {}
        monkeypatch.setattr(G, "_upload", _fake_upload_factory(captured, result=""))
        with pytest.raises(RuntimeError, match="did not land canonically"):
            G._mirror_capture_marker(done_p, "prefix/stem", {"fingerprint": {}})
        assert not done_p.exists()

    def test_both_capture_writers_route_through_the_mirror(self):
        """AST wiring pin: every `_atomic_write_json(done_p, ...)` capture-marker
        write sits under an `args.skip_upload` branch, and `_mirror_capture_marker`
        is called by BOTH capture drivers (P4 run_capture + P4b reliability)."""
        src = Path(G.__file__).read_text()
        tree = ast.parse(src)
        parents: dict[ast.AST, ast.AST] = {}
        for node in ast.walk(tree):
            for child in ast.iter_child_nodes(node):
                parents[child] = node

        def enclosing_funcs(node):
            names = []
            while node in parents:
                node = parents[node]
                if isinstance(node, ast.FunctionDef):
                    names.append(node.name)
            return names

        mirror_callers = set()
        bare_done_writes = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            fn = node.func
            name = fn.id if isinstance(fn, ast.Name) else getattr(fn, "attr", None)
            if name == "_mirror_capture_marker":
                mirror_callers.update(enclosing_funcs(node))
            if (
                name == "_atomic_write_json"
                and node.args
                and isinstance(node.args[0], ast.Name)
                and node.args[0].id == "done_p"
            ):
                # Must be guarded by an `if args.skip_upload:` ancestor.
                anc, guarded = node, False
                while anc in parents:
                    anc = parents[anc]
                    if isinstance(anc, ast.If) and "skip_upload" in ast.dump(anc.test):
                        guarded = True
                        break
                if not guarded:
                    bare_done_writes.append(node.lineno)
        assert {"run_capture", "run_capture_reliability"} <= mirror_callers, mirror_callers
        assert not bare_done_writes, (
            f"unguarded _atomic_write_json(done_p, ...) at lines {bare_done_writes} — "
            "capture markers must route through _mirror_capture_marker unless "
            "--skip-upload (defect 1a, marker v144)"
        )


# ---------------------------------------------------------------------------
# Round 18 blocker 1 — resume-skip repairs a missing Hub marker twin
# ---------------------------------------------------------------------------


class TestEnsureCaptureMarkerHubTwin:
    def _stem(self, tmp_path, payload):
        stem_dir = tmp_path / "store" / "arm1" / "post__gsm8k"
        stem_dir.mkdir(parents=True)
        done_p = stem_dir / "_complete.json"
        done_p.write_text(json.dumps(payload))
        return done_p

    def test_known_present_twin_is_a_zero_network_skip(self, tmp_path, monkeypatch):
        payload = {"fingerprint": {"stage": "capture"}, "report": {}}
        done_p = self._stem(tmp_path, payload)
        twin = f"{G.STORE_PREFIX}/arm1/post__gsm8k/_complete.json"
        monkeypatch.setattr(G, "_HUB_MARKER_SETS", {(1, False): {twin}}, raising=False)
        monkeypatch.setattr(G, "_HUB_SHARD_STEM_SETS", {(1, False): set()}, raising=False)
        calls: dict = {}
        monkeypatch.setattr(G, "_upload", _fake_upload_factory(calls))

        G._ensure_capture_marker_hub_twin(done_p, 1, False, "post__gsm8k", payload)

        assert calls == {}  # cheap skip: no Hub round-trip when the twin is known present
        assert json.loads(done_p.read_text()) == payload

    def test_missing_twin_is_repaired_via_the_mirror(self, tmp_path, monkeypatch):
        """The pre-fix local-only marker (blocker 1): with the stem's REMOTE
        shards present (r19 gate), repair mirrors the LOCAL payload to the
        stem's Hub prefix with _mirror_capture_marker's own upload+assert
        discipline, then memoizes the repaired twin."""
        payload = {"fingerprint": {"stage": "capture"}, "report": {"stem": "post__gsm8k"}}
        done_p = self._stem(tmp_path, payload)
        dest = f"{G.STORE_PREFIX}/arm1/post__gsm8k"
        monkeypatch.setattr(G, "_HUB_MARKER_SETS", {(1, False): set()}, raising=False)
        monkeypatch.setattr(G, "_HUB_SHARD_STEM_SETS", {(1, False): {dest}}, raising=False)
        captured: dict = {}
        monkeypatch.setattr(G, "_upload", _fake_upload_factory(captured))

        G._ensure_capture_marker_hub_twin(done_p, 1, False, "post__gsm8k", payload)

        assert captured["path_in_repo"] == f"{G.STORE_PREFIX}/arm1/post__gsm8k/_complete.json"
        assert captured["repo_id"] == G.DEFAULT_DATASET_REPO
        assert captured["upload_as_file"] is True
        assert captured["raise_on_error"] is True
        assert json.loads(captured["bytes"]) == payload
        assert json.loads(done_p.read_text()) == payload  # re-blessed, same content
        # The memo learned the repaired twin: a second call is zero-network.
        captured.clear()
        G._ensure_capture_marker_hub_twin(done_p, 1, False, "post__gsm8k", payload)
        assert captured == {}

    def test_repair_failure_raises_and_keeps_the_local_marker(self, tmp_path, monkeypatch):
        """Fail-loud repair: a mirror failure RAISES (never warn-and-skip); the
        valid local marker survives so the next resume retries the repair."""
        payload = {"fingerprint": {"stage": "capture"}}
        done_p = self._stem(tmp_path, payload)
        dest = f"{G.STORE_PREFIX}/arm1/post__gsm8k"
        monkeypatch.setattr(G, "_HUB_MARKER_SETS", {(1, False): set()}, raising=False)
        monkeypatch.setattr(G, "_HUB_SHARD_STEM_SETS", {(1, False): {dest}}, raising=False)
        monkeypatch.setattr(G, "_upload", _fake_upload_factory({}, exc=RuntimeError("hub down")))
        with pytest.raises(RuntimeError):
            G._ensure_capture_marker_hub_twin(done_p, 1, False, "post__gsm8k", payload)
        assert json.loads(done_p.read_text()) == payload

    def test_smoke_stems_route_to_the_smoke_arm_prefix(self, tmp_path, monkeypatch):
        payload = {"fingerprint": {"stage": "capture", "smoke": True}}
        stem_dir = tmp_path / "store" / "smoke_arm2" / "post__gsm8k"
        stem_dir.mkdir(parents=True)
        done_p = stem_dir / "_complete.json"
        done_p.write_text(json.dumps(payload))
        monkeypatch.setattr(G, "_HUB_MARKER_SETS", {(2, True): set()}, raising=False)
        monkeypatch.setattr(
            G,
            "_HUB_SHARD_STEM_SETS",
            {(2, True): {f"{G.STORE_PREFIX}/smoke_arm2/post__gsm8k"}},
            raising=False,
        )
        captured: dict = {}
        monkeypatch.setattr(G, "_upload", _fake_upload_factory(captured))
        G._ensure_capture_marker_hub_twin(done_p, 2, True, "post__gsm8k", payload)
        assert captured["path_in_repo"] == (
            f"{G.STORE_PREFIX}/smoke_arm2/post__gsm8k/_complete.json"
        )

    def test_cold_memo_takes_one_scoped_listing_and_memoizes(self, monkeypatch):
        """_fill_hub_store_listing real body: ONE scoped recursive
        list_repo_tree per (arm, smoke) — materialized inside the retry thunk —
        fills BOTH the marker set and the shard-stem set (r19: the shard
        entries are evidence, no longer discarded); second calls hit the memo."""
        import huggingface_hub

        listed: list[dict] = []

        class _Entry:
            def __init__(self, path):
                self.path = path

        class _FakeApi:
            def list_repo_tree(self, repo_id, *, path_in_repo, repo_type, recursive):
                listed.append(
                    {
                        "repo_id": repo_id,
                        "path_in_repo": path_in_repo,
                        "repo_type": repo_type,
                        "recursive": recursive,
                    }
                )
                yield _Entry(f"{path_in_repo}/post__gsm8k/_complete.json")
                yield _Entry(f"{path_in_repo}/post__gsm8k/slot0.shard0.pt")
                yield _Entry(f"{path_in_repo}/short__csqa/slot1.shard2.pt")

        monkeypatch.setattr(huggingface_hub, "HfApi", _FakeApi)
        monkeypatch.setattr(G, "_HUB_MARKER_SETS", {}, raising=False)
        monkeypatch.setattr(G, "_HUB_SHARD_STEM_SETS", {}, raising=False)
        got = G._hub_capture_marker_paths(2, True)
        assert got == {f"{G.STORE_PREFIX}/smoke_arm2/post__gsm8k/_complete.json"}
        shard_stems = G._hub_shard_stem_paths(2, True)
        assert shard_stems == {
            f"{G.STORE_PREFIX}/smoke_arm2/post__gsm8k",
            f"{G.STORE_PREFIX}/smoke_arm2/short__csqa",
        }
        assert G._hub_capture_marker_paths(2, True) is got  # memo hit
        assert G._hub_shard_stem_paths(2, True) is shard_stems  # memo hit
        assert len(listed) == 1  # BOTH views from ONE listing
        assert listed[0] == {
            "repo_id": G.DEFAULT_DATASET_REPO,
            "path_in_repo": f"{G.STORE_PREFIX}/smoke_arm2",
            "repo_type": "dataset",
            "recursive": True,
        }

    def test_missing_prefix_reads_as_empty_set(self, monkeypatch):
        """A 404 on the arm prefix (nothing uploaded yet) is an EMPTY set —
        every resumed stem then routes to the repair path, never a blind
        accept."""
        import huggingface_hub
        from huggingface_hub.errors import EntryNotFoundError

        class _FakeApi:
            def list_repo_tree(self, repo_id, *, path_in_repo, repo_type, recursive):
                raise EntryNotFoundError("404: tree not found")
                yield  # pragma: no cover — keeps this a generator like the real API

        monkeypatch.setattr(huggingface_hub, "HfApi", _FakeApi)
        monkeypatch.setattr(G, "_HUB_MARKER_SETS", {}, raising=False)
        monkeypatch.setattr(G, "_HUB_SHARD_STEM_SETS", {}, raising=False)
        assert G._hub_capture_marker_paths(1, False) == set()
        assert G._hub_shard_stem_paths(1, False) == set()

    def test_both_resume_skip_sites_call_the_twin_repair(self):
        """AST wiring pin: BOTH capture drivers (P4 run_capture + P4b
        reliability) call _ensure_capture_marker_hub_twin on their resume-skip
        paths — a repair wired into only one driver re-opens blocker 1 for the
        other."""
        src = Path(G.__file__).read_text()
        tree = ast.parse(src)
        parents: dict[ast.AST, ast.AST] = {}
        for node in ast.walk(tree):
            for child in ast.iter_child_nodes(node):
                parents[child] = node
        callers = set()
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            fn = node.func
            name = fn.id if isinstance(fn, ast.Name) else getattr(fn, "attr", None)
            if name != "_ensure_capture_marker_hub_twin":
                continue
            anc = node
            while anc in parents:
                anc = parents[anc]
                if isinstance(anc, ast.FunctionDef):
                    callers.add(anc.name)
        assert {"run_capture", "run_capture_reliability"} <= callers, callers


# ---------------------------------------------------------------------------
# Round 19 — the twin repair is gated on SHARD presence (reconciler on v18):
# marker-only success over an empty stem prefix is UNREACHABLE.
# ---------------------------------------------------------------------------


def _fake_upload_seq_factory(calls: list, *, result=_ECHO_CANONICAL, exc: Exception | None = None):
    """Like _fake_upload_factory, but appends ONE record per call (ordering)."""

    def fake_upload(
        local_path,
        repo_id,
        repo_type,
        path_in_repo,
        delete_after=False,
        upload_as_file=False,
        ignore_patterns=None,
        private=False,
        raise_on_error=False,
    ):
        rec = {
            "local_path": Path(local_path),
            "is_dir": Path(local_path).is_dir(),
            "repo_id": repo_id,
            "repo_type": repo_type,
            "path_in_repo": path_in_repo,
            "upload_as_file": upload_as_file,
            "ignore_patterns": ignore_patterns,
            "raise_on_error": raise_on_error,
        }
        if Path(local_path).is_file():
            rec["bytes"] = Path(local_path).read_bytes()
        calls.append(rec)
        if exc is not None:
            raise exc
        if result is _ECHO_CANONICAL:
            return f"{repo_id}/{path_in_repo}"
        return result

    return fake_upload


class TestShardGatedTwinRepair:
    STEM = "post__gsm8k"
    DEST = None  # filled in setup_method (needs G import)

    def setup_method(self):
        self.DEST = f"{G.STORE_PREFIX}/arm1/{self.STEM}"

    def _stem_dir(self, tmp_path, payload, *, n_local_shards=0):
        stem_dir = tmp_path / "store" / "arm1" / self.STEM
        stem_dir.mkdir(parents=True)
        done_p = stem_dir / "_complete.json"
        done_p.write_text(json.dumps(payload))
        for i in range(n_local_shards):
            (stem_dir / f"slot{i}.shard0.pt").write_bytes(b"\0" * 8)
        return done_p

    def test_repo_not_found_raises_at_the_listing(self, monkeypatch):
        """Part 1 (reconciler on v18): a REPO-level fault must raise at the
        listing — folding it into 'empty' is the banned silent-zero shape, and
        hub._upload downstream resurrects a deleted repo via
        create_repo(exist_ok=True), so it can never be trusted to fail loud."""
        import huggingface_hub
        from huggingface_hub.errors import RepositoryNotFoundError

        class _FakeApi:
            def list_repo_tree(self, repo_id, *, path_in_repo, repo_type, recursive):
                raise RepositoryNotFoundError("401/404: repo gone")
                yield  # pragma: no cover — keeps this a generator like the real API

        monkeypatch.setattr(huggingface_hub, "HfApi", _FakeApi)
        monkeypatch.setattr(G, "_HUB_MARKER_SETS", {}, raising=False)
        monkeypatch.setattr(G, "_HUB_SHARD_STEM_SETS", {}, raising=False)
        with pytest.raises(RepositoryNotFoundError):
            G._hub_capture_marker_paths(1, False)
        with pytest.raises(RepositoryNotFoundError):
            G._hub_shard_stem_paths(1, False)

    def test_shards_absent_everywhere_raises_and_never_uploads_a_marker(
        self, tmp_path, monkeypatch
    ):
        """The reconciler's mechanizable line: an empty remote listing plus a
        would-be-successful marker upload must NEVER end in a bare marker
        write — with no local shards either, the repair RAISES naming the
        stem, and no upload of ANY kind is attempted."""
        payload = {"fingerprint": {"stage": "capture"}}
        done_p = self._stem_dir(tmp_path, payload, n_local_shards=0)
        monkeypatch.setattr(G, "_HUB_MARKER_SETS", {(1, False): set()}, raising=False)
        monkeypatch.setattr(G, "_HUB_SHARD_STEM_SETS", {(1, False): set()}, raising=False)
        calls: list = []
        monkeypatch.setattr(G, "_upload", _fake_upload_seq_factory(calls))

        with pytest.raises(RuntimeError, match=self.STEM):
            G._ensure_capture_marker_hub_twin(done_p, 1, False, self.STEM, payload)

        assert calls == []  # the would-be-successful marker upload never ran
        assert json.loads(done_p.read_text()) == payload  # local marker kept
        # The twin was NOT memoized as present.
        assert G._HUB_MARKER_SETS[(1, False)] == set()

    def test_skip_upload_transition_uploads_stem_dir_then_mirrors(self, tmp_path, monkeypatch):
        """Path 1 (--skip-upload transition) is RECOVERED, not just refused:
        remote shards absent + local shards present => full stem-dir upload
        FIRST (the run_capture write-path shape, marker excluded), THEN the
        marker mirror — never a marker-only write."""
        payload = {"fingerprint": {"stage": "capture"}, "report": {"stem": self.STEM}}
        done_p = self._stem_dir(tmp_path, payload, n_local_shards=2)
        monkeypatch.setattr(G, "_HUB_MARKER_SETS", {(1, False): set()}, raising=False)
        monkeypatch.setattr(G, "_HUB_SHARD_STEM_SETS", {(1, False): set()}, raising=False)
        calls: list = []
        monkeypatch.setattr(G, "_upload", _fake_upload_seq_factory(calls))

        G._ensure_capture_marker_hub_twin(done_p, 1, False, self.STEM, payload)

        assert len(calls) == 2, calls
        folder, marker = calls
        # Call 1: the stem DIR to the stem prefix (shards land first)...
        assert folder["local_path"] == done_p.parent
        assert folder["is_dir"] is True
        assert folder["path_in_repo"] == self.DEST
        assert folder["upload_as_file"] is False
        assert folder["raise_on_error"] is True
        # ...with the marker excluded, so the attestation cannot precede the
        # shards even inside the repair's own commits.
        assert "_complete.json" in (folder["ignore_patterns"] or [])
        # Call 2: the marker mirror, strictly after.
        assert marker["path_in_repo"] == f"{self.DEST}/_complete.json"
        assert marker["upload_as_file"] is True
        assert marker["raise_on_error"] is True
        assert json.loads(marker["bytes"]) == payload
        # Local marker retained; both memos learned the repair.
        assert json.loads(done_p.read_text()) == payload
        assert self.DEST in G._HUB_SHARD_STEM_SETS[(1, False)]
        assert f"{self.DEST}/_complete.json" in G._HUB_MARKER_SETS[(1, False)]
        # Second call: zero-network resume-skip.
        calls.clear()
        G._ensure_capture_marker_hub_twin(done_p, 1, False, self.STEM, payload)
        assert calls == []

    def test_remote_shards_present_never_reuploads_the_stem_dir(self, tmp_path, monkeypatch):
        """Top branch (the legitimate pre-r17 repair): remote shards present
        => mirror the marker ONLY — no redundant stem-dir re-upload."""
        payload = {"fingerprint": {"stage": "capture"}}
        done_p = self._stem_dir(tmp_path, payload, n_local_shards=2)
        monkeypatch.setattr(G, "_HUB_MARKER_SETS", {(1, False): set()}, raising=False)
        monkeypatch.setattr(G, "_HUB_SHARD_STEM_SETS", {(1, False): {self.DEST}}, raising=False)
        calls: list = []
        monkeypatch.setattr(G, "_upload", _fake_upload_seq_factory(calls))

        G._ensure_capture_marker_hub_twin(done_p, 1, False, self.STEM, payload)

        assert len(calls) == 1, calls
        assert calls[0]["upload_as_file"] is True
        assert calls[0]["path_in_repo"] == f"{self.DEST}/_complete.json"

    def test_stem_dir_upload_failure_never_blesses_the_hub_marker(self, tmp_path, monkeypatch):
        """Middle-branch fail-loud ordering: a failed stem-dir upload raises
        BEFORE any marker upload — the attestation can never land without the
        shards it attests."""
        payload = {"fingerprint": {"stage": "capture"}}
        done_p = self._stem_dir(tmp_path, payload, n_local_shards=1)
        monkeypatch.setattr(G, "_HUB_MARKER_SETS", {(1, False): set()}, raising=False)
        monkeypatch.setattr(G, "_HUB_SHARD_STEM_SETS", {(1, False): set()}, raising=False)
        calls: list = []
        monkeypatch.setattr(
            G, "_upload", _fake_upload_seq_factory(calls, exc=RuntimeError("hub down"))
        )

        with pytest.raises(RuntimeError):
            G._ensure_capture_marker_hub_twin(done_p, 1, False, self.STEM, payload)

        assert len(calls) == 1  # only the folder upload was attempted
        assert calls[0]["is_dir"] is True
        assert G._HUB_MARKER_SETS[(1, False)] == set()  # twin never blessed
        assert self.DEST not in G._HUB_SHARD_STEM_SETS[(1, False)]
        assert json.loads(done_p.read_text()) == payload  # local marker kept


# ---------------------------------------------------------------------------
# Round 20 — canonical-destination gate (reconciler ruling on body v19): a
# TRUTHY hub._upload return naming the overflow repo (the default-ON
# file-count reroute, #1108/#2304) must never bless the memo or place a
# marker — on the repair path AND the write path's shared mirror helper.
# ---------------------------------------------------------------------------


class TestCanonicalDestinationGate:
    STEM = "post__gsm8k"

    def setup_method(self):
        self.DEST = f"{G.STORE_PREFIX}/arm1/{self.STEM}"

    def _stem_dir(self, tmp_path, payload, *, n_local_shards=0):
        stem_dir = tmp_path / "store" / "arm1" / self.STEM
        stem_dir.mkdir(parents=True)
        done_p = stem_dir / "_complete.json"
        done_p.write_text(json.dumps(payload))
        for i in range(n_local_shards):
            (stem_dir / f"slot{i}.shard0.pt").write_bytes(b"\0" * 8)
        return done_p

    def test_overflow_rerouted_stem_dir_raises_before_any_marker(self, tmp_path, monkeypatch):
        """Repair middle branch: a truthy overflow-path return from the
        stem-dir upload RAISES naming both repos — BEFORE the memo add and
        BEFORE any marker upload. Pre-fix, `assert res` passed on the truthy
        reroute, the memo falsely recorded canonical presence, and the
        single-file marker commit could land canonically exactly when the
        multi-file folder commit was rejected by near-cap arithmetic."""
        payload = {"fingerprint": {"stage": "capture"}}
        done_p = self._stem_dir(tmp_path, payload, n_local_shards=2)
        monkeypatch.setattr(G, "_HUB_MARKER_SETS", {(1, False): set()}, raising=False)
        monkeypatch.setattr(G, "_HUB_SHARD_STEM_SETS", {(1, False): set()}, raising=False)
        calls: list = []
        monkeypatch.setattr(
            G,
            "_upload",
            _fake_upload_seq_factory(calls, result=f"{OVERFLOW_REPO}/{self.DEST}"),
        )

        with pytest.raises(RuntimeError, match="did not land canonically") as ei:
            G._ensure_capture_marker_hub_twin(done_p, 1, False, self.STEM, payload)

        # The refusal names BOTH repos (effective + canonical).
        assert OVERFLOW_REPO in str(ei.value)
        assert G.DEFAULT_DATASET_REPO in str(ei.value)
        # Only the folder upload was attempted — never a marker upload.
        assert len(calls) == 1 and calls[0]["is_dir"] is True
        # No memo mutation on either set.
        assert G._HUB_MARKER_SETS[(1, False)] == set()
        assert self.DEST not in G._HUB_SHARD_STEM_SETS[(1, False)]
        # Local marker + shards retained: the next resume retries.
        assert json.loads(done_p.read_text()) == payload
        assert len(sorted(done_p.parent.glob("slot*.shard*.pt"))) == 2

    def test_overflow_rerouted_marker_never_blesses_local_resume(self, tmp_path, monkeypatch):
        """Write path (the shared mirror helper): a truthy overflow-path
        return from the single-file marker upload RAISES before
        ``tmp.replace(done_p)`` — local resume is never blessed."""
        stem_dir = tmp_path / "s" / self.STEM
        stem_dir.mkdir(parents=True)
        done_p = stem_dir / "_complete.json"
        dest = f"prefix/{self.STEM}"
        captured: dict = {}
        monkeypatch.setattr(
            G,
            "_upload",
            _fake_upload_factory(captured, result=f"{OVERFLOW_REPO}/{dest}/_complete.json"),
        )
        with pytest.raises(RuntimeError, match="did not land canonically"):
            G._mirror_capture_marker(done_p, dest, {"fingerprint": {}})
        assert not done_p.exists()

    def test_overflow_rerouted_marker_in_repair_never_memoizes_the_twin(
        self, tmp_path, monkeypatch
    ):
        """Repair top branch (remote shards present): a rerouted MARKER upload
        raises out of the mirror, the twin is never memoized, and the
        pre-existing local marker keeps its bytes."""
        payload = {"fingerprint": {"stage": "capture"}}
        done_p = self._stem_dir(tmp_path, payload)
        monkeypatch.setattr(G, "_HUB_MARKER_SETS", {(1, False): set()}, raising=False)
        monkeypatch.setattr(G, "_HUB_SHARD_STEM_SETS", {(1, False): {self.DEST}}, raising=False)
        calls: list = []
        monkeypatch.setattr(
            G,
            "_upload",
            _fake_upload_seq_factory(calls, result=f"{OVERFLOW_REPO}/{self.DEST}/_complete.json"),
        )
        with pytest.raises(RuntimeError, match="did not land canonically"):
            G._ensure_capture_marker_hub_twin(done_p, 1, False, self.STEM, payload)
        assert G._HUB_MARKER_SETS[(1, False)] == set()
        assert json.loads(done_p.read_text()) == payload

    def test_all_attestation_upload_sites_route_through_the_gate(self):
        """AST wiring pin: the canonical-destination check gates all four
        attestation-adjacent upload sites — BOTH write-path store uploads
        (run_capture + run_capture_reliability), the repair's stem-dir upload,
        and the shared marker mirror. ``upload_stage`` (rollout JSONLs) is
        deliberately NOT gated: the overflow reroute is the designed
        durability path for ordinary artifacts (#1108/#2304), and gating it
        would regress that behavior."""
        src = Path(G.__file__).read_text()
        tree = ast.parse(src)
        parents: dict[ast.AST, ast.AST] = {}
        for node in ast.walk(tree):
            for child in ast.iter_child_nodes(node):
                parents[child] = node
        callers = set()
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            fn = node.func
            name = fn.id if isinstance(fn, ast.Name) else getattr(fn, "attr", None)
            if name != "_require_canonical_upload":
                continue
            anc = node
            while anc in parents:
                anc = parents[anc]
                if isinstance(anc, ast.FunctionDef):
                    callers.add(anc.name)
        assert {
            "run_capture",
            "run_capture_reliability",
            "_ensure_capture_marker_hub_twin",
            "_mirror_capture_marker",
        } <= callers, callers
        # upload_stage stays un-gated by design (rollouts may reroute).
        assert "upload_stage" not in callers, callers


# ---------------------------------------------------------------------------
# Defect 2 — log-upload set covers the four groups, excludes data artifacts
# ---------------------------------------------------------------------------


@pytest.fixture()
def log_fixture(tmp_path):
    pod_root = tmp_path / "workspace"
    log_dir = pod_root / "logs"
    out_root = pod_root / "issue2546"
    fallback_env = out_root / "done" / "fallbacks_a1.env"
    # Group 1: rotated dispatcher logs (flat at the arm root).
    log_dir.mkdir(parents=True)
    (log_dir / "issue-2546.log").write_text("main log\n")
    (log_dir / "issue-2546.log.p5fits-20260827T001244Z").write_text("rotated\n")
    # Decoys in the sentinel namespace (poller-drained; never log-mirrored).
    (log_dir / "issue-2546-epm_results-a1-1-2.json").write_text("{}")
    (log_dir / "issue-2546-smoke-a1-1-2.json.processed").write_text("{}")
    (log_dir / "issue-2546.pid").write_text("123\n")
    # Group 2: worker logs.
    (out_root / "logs").mkdir(parents=True)
    (out_root / "logs" / "gen-post_greedy_a1.slot0.log").write_text("w\n")
    (out_root / "logs" / "caprel-rel_post__csqa.slot1.log").write_text("w\n")
    # Group 3: fit worker logs.
    (out_root / "work" / "fits_a1").mkdir(parents=True)
    (out_root / "work" / "fits_a1" / "slot0.log").write_text("f\n")
    # Non-log work handoffs must be excluded.
    (out_root / "work" / "capture_post__csqa").mkdir(parents=True)
    (out_root / "work" / "capture_post__csqa" / "handoff.json").write_text("{}")
    # Group 4: aux (launcher + revisions + fallbacks env).
    (pod_root / "launch_issue_2546.sh").write_text("#!/bin/bash\n")
    (out_root / "revisions.json").write_text("{}")
    fallback_env.parent.mkdir(parents=True)
    fallback_env.write_text("PREFILL_FB=\nDECODE_FB=\n")
    # Data-artifact decoys: shards / fitcache tensors / caches / eval JSONs.
    store = out_root / "store" / "arm1" / "post__csqa"
    store.mkdir(parents=True)
    (store / "slot0.shard0.pt").write_bytes(b"\0" * 8)
    (store / "_complete.json").write_text("{}")
    cache = out_root / "fitcache" / "arm1" / "post__csqa"
    cache.mkdir(parents=True)
    (cache / "post.pt").write_bytes(b"\0" * 8)
    (cache / "_cache_complete.json").write_text("{}")
    (out_root / "out" / "cells").mkdir(parents=True)
    (out_root / "out" / "cells" / "c__a1.json").write_text("{}")
    (out_root / "preds.npz").write_bytes(b"\0" * 8)
    return log_dir, out_root, fallback_env


class TestCollectLogFiles:
    def test_four_groups_included_and_data_artifacts_excluded(self, log_fixture):
        log_dir, out_root, fallback_env = log_fixture
        files = UL.collect_log_files(log_dir, out_root, 1, fallback_env)
        rels = sorted(f"{d}/{p.name}" if d else p.name for p, d in files)
        assert rels == [
            "aux/fallbacks_a1.env",
            "aux/launch_issue_2546.sh",
            "aux/revisions.json",
            "issue-2546.log",
            "issue-2546.log.p5fits-20260827T001244Z",
            "logs/caprel-rel_post__csqa.slot1.log",
            "logs/gen-post_greedy_a1.slot0.log",
            "work/fits_a1/slot0.log",
        ]
        names = {p.name for p, _ in files}
        # Shards / npz / fitcache tensors / caches / sentinels never upload here.
        assert not names & {
            "slot0.shard0.pt",
            "post.pt",
            "_cache_complete.json",
            "preds.npz",
            "c__a1.json",
            "handoff.json",
            "issue-2546-epm_results-a1-1-2.json",
            "issue-2546-smoke-a1-1-2.json.processed",
            "issue-2546.pid",
        }
        # _complete.json rides the capture-marker mirror, not the log upload.
        dests = {d for _, d in files}
        assert dests == {"", "logs", "work/fits_a1", "aux"}

    def test_missing_groups_are_skipped_not_fatal(self, tmp_path):
        files = UL.collect_log_files(
            tmp_path / "nope", tmp_path / "also-nope", 2, tmp_path / "fb.env"
        )
        assert files == []


class TestStageFiles:
    def test_small_file_copied_verbatim(self, tmp_path):
        src = tmp_path / "issue-2546.log"
        src.write_text("hello\nworld\n")
        stage = tmp_path / "stage"
        n_split = UL.stage_files([(src, "")], stage)
        assert n_split == 0
        assert (stage / "issue-2546.log").read_text() == "hello\nworld\n"

    def test_oversize_file_line_split_losslessly(self, tmp_path):
        src = tmp_path / "big.log"
        line = ("x" * 999) + "\n"
        n_lines = (11 * 1024 * 1024) // len(line)  # ~11 MB > 9.5 MB threshold
        src.write_text(line * n_lines)
        stage = tmp_path / "stage"
        n_split = UL.stage_files([(src, "logs")], stage)
        assert n_split == 1
        parts = sorted((stage / "logs").glob("big.log.part*"))
        assert len(parts) >= 2
        assert all(p.stat().st_size <= UL.PART_BYTES for p in parts)
        assert b"".join(p.read_bytes() for p in parts) == src.read_bytes()
        # The unsplit original is NOT staged alongside its parts.
        assert not (stage / "logs" / "big.log").exists()

    def test_single_line_exceeding_part_size_hard_splits(self, tmp_path):
        """r18 blocker 3, at the REAL constants: one newline-free line past the
        9.5 MB split threshold must hard-split on byte boundaries instead of
        emitting one oversize .part (which would force-route to LFS)."""
        src = tmp_path / "huge.log"
        src.write_bytes(b"y" * (UL.MAX_TEXT_BYTES + 4096))
        stage = tmp_path / "stage"
        n_split = UL.stage_files([(src, "")], stage)
        assert n_split == 1
        parts = sorted(stage.glob("huge.log.part*"))
        assert len(parts) == 2  # 9 MiB + remainder
        assert all(p.stat().st_size <= UL.PART_BYTES for p in parts)
        assert b"".join(p.read_bytes() for p in parts) == src.read_bytes()

    @pytest.mark.parametrize("delta", [-1, 0, 1])
    def test_newline_free_line_at_part_size_boundary(self, tmp_path, delta):
        """Boundary cases (limit-1 / exactly-at-limit / limit+1) for the
        byte-boundary hard-split, via the helper's part_bytes parameter."""
        part = 64
        src = tmp_path / "b.log"
        src.write_bytes(b"z" * (part + delta))
        dest = tmp_path / "out"
        dest.mkdir()
        UL._split_oversize(src, dest, part_bytes=part)
        parts = sorted(dest.glob("b.log.part*"))
        assert len(parts) == (1 if delta <= 0 else 2)
        assert all(p.stat().st_size <= part for p in parts)
        assert b"".join(p.read_bytes() for p in parts) == src.read_bytes()

    def test_mixed_lines_and_oversize_line_lossless(self, tmp_path):
        """An over-length line embedded between ordinary lines: every part
        stays <= part_bytes and concatenation is byte-exact (the tail of the
        hard-split line packs with the following lines)."""
        part = 64
        blob = b"a" * 10 + b"\n" + b"b" * 200 + b"\n" + b"c" * 30 + b"\n"
        src = tmp_path / "m.log"
        src.write_bytes(blob)
        dest = tmp_path / "out"
        dest.mkdir()
        UL._split_oversize(src, dest, part_bytes=part)
        parts = sorted(dest.glob("m.log.part*"))
        assert all(p.stat().st_size <= part for p in parts)
        assert b"".join(p.read_bytes() for p in parts) == blob
