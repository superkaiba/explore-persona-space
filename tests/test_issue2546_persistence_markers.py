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

Network-free: the only Hub boundary (``hub._upload``) is faked with a
signature-mirroring fake; every other body runs for real on tmp_path.
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import pytest

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


def _fake_upload_factory(captured: dict, *, result="repo/dest", exc: Exception | None = None):
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
        stem_dir = tmp_path / "s"
        stem_dir.mkdir()
        done_p = stem_dir / "_complete.json"
        captured: dict = {}
        monkeypatch.setattr(G, "_upload", _fake_upload_factory(captured, result=""))
        with pytest.raises(AssertionError):
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
