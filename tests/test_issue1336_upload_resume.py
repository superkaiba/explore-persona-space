"""#1336 done == uploaded per-cell resume contract (r1 review Major 2).

Pre-fix, extract wrote its ``.done.json`` BEFORE ``_upload_cell`` and gen's
local-outputs skip fired before any upload retry — a transient Hub failure was
permanently skipped on a same-workdir re-run. These tests pin the fixed
predicate: a done-but-not-uploaded cell re-attempts ONLY the upload (never the
extraction/generation), the flag flips only after upload success, and the
Hub-completeness probe drives the gen skip branch. Hub/network boundaries are
faked signature-conformantly; everything else runs the real production bodies.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import create_autospec

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(REPO_ROOT / "scripts"), str(REPO_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

STEM = "base_chat_lmsys5k"


def _argv(tmp_path: Path, upload: bool = True) -> list[str]:
    argv = [
        "issue1336_extract_turnstore.py",
        "--model",
        "base",
        "--format",
        "chat",
        "--corpus",
        "lmsys5k",
        "--out-dir",
        str(tmp_path),
    ]
    return argv + (["--upload"] if upload else [])


def _write_done(tmp_path: Path, uploaded) -> Path:
    done = tmp_path / f"{STEM}.done.json"
    payload = {"stem": STEM, "n_rows": 8, "n_shards": 1}
    if uploaded is not None:
        payload["uploaded"] = uploaded
    done.write_text(json.dumps(payload) + "\n")
    return done


def _no_extract(monkeypatch, ext):
    def _boom(*a, **k):
        raise AssertionError("extraction must NOT re-run on a done cell")

    monkeypatch.setattr(ext, "load_model", _boom)


# ---------------------------------------------------------------------------
# extract: resume predicate
# ---------------------------------------------------------------------------
def test_extract_resume_reattempts_failed_upload(tmp_path, monkeypatch):
    """done + uploaded:false + --upload => upload retried, flag flipped, no re-extract."""
    import issue1336_extract_turnstore as ext

    done = _write_done(tmp_path, uploaded=False)
    calls: list[str] = []
    monkeypatch.setattr(ext, "_upload_cell", lambda out_dir, stem: calls.append(stem))
    _no_extract(monkeypatch, ext)
    monkeypatch.setattr(sys, "argv", _argv(tmp_path))
    ext.main()
    assert calls == [STEM]
    assert json.loads(done.read_text())["uploaded"] is True


def test_extract_legacy_done_without_flag_retries_upload(tmp_path, monkeypatch):
    """A done marker with no 'uploaded' field is treated as not-uploaded (safe retry)."""
    import issue1336_extract_turnstore as ext

    done = _write_done(tmp_path, uploaded=None)
    calls: list[str] = []
    monkeypatch.setattr(ext, "_upload_cell", lambda out_dir, stem: calls.append(stem))
    _no_extract(monkeypatch, ext)
    monkeypatch.setattr(sys, "argv", _argv(tmp_path))
    ext.main()
    assert calls == [STEM]
    assert json.loads(done.read_text())["uploaded"] is True


def test_extract_done_and_uploaded_skips(tmp_path, monkeypatch):
    """done == uploaded => plain skip, no upload call, no extraction."""
    import issue1336_extract_turnstore as ext

    done = _write_done(tmp_path, uploaded=True)
    monkeypatch.setattr(
        ext, "_upload_cell", lambda *a: pytest.fail("upload must not re-run when uploaded")
    )
    _no_extract(monkeypatch, ext)
    monkeypatch.setattr(sys, "argv", _argv(tmp_path))
    ext.main()
    assert json.loads(done.read_text())["uploaded"] is True


def test_extract_no_upload_flag_skips_without_probe(tmp_path, monkeypatch):
    """Without --upload the resume path never touches the Hub (smoke shape)."""
    import issue1336_extract_turnstore as ext

    _write_done(tmp_path, uploaded=False)
    monkeypatch.setattr(ext, "_upload_cell", lambda *a: pytest.fail("no upload without --upload"))
    _no_extract(monkeypatch, ext)
    monkeypatch.setattr(sys, "argv", _argv(tmp_path, upload=False))
    ext.main()


def test_extract_upload_failure_keeps_uploaded_false(tmp_path, monkeypatch):
    """The flag flips ONLY after upload success — a raise leaves it retryable."""
    import issue1336_extract_turnstore as ext

    done = _write_done(tmp_path, uploaded=False)

    def _fail(out_dir, stem):
        raise RuntimeError("simulated transient Hub failure")

    monkeypatch.setattr(ext, "_upload_cell", _fail)
    _no_extract(monkeypatch, ext)
    monkeypatch.setattr(sys, "argv", _argv(tmp_path))
    with pytest.raises(RuntimeError, match="simulated transient"):
        ext.main()
    assert json.loads(done.read_text())["uploaded"] is False


def test_extract_upload_cell_body_excludes_done_marker(tmp_path, monkeypatch):
    """Real _upload_cell body: retry wrapper runs, done.json NOT in allow_patterns."""
    import huggingface_hub
    import issue1336_extract_turnstore as ext

    fake_upload = create_autospec(huggingface_hub.upload_folder)
    monkeypatch.setattr(huggingface_hub, "upload_folder", fake_upload)
    ext._upload_cell(tmp_path, STEM)
    kwargs = fake_upload.call_args.kwargs
    assert kwargs["allow_patterns"] == [f"{STEM}_shard*"]
    assert kwargs["repo_type"] == "dataset"
    assert kwargs["path_in_repo"].endswith(f"turnstore_{STEM}")


# ---------------------------------------------------------------------------
# gen: skip-branch Hub probe + re-upload
# ---------------------------------------------------------------------------
def _gen_cell(monkeypatch, tmp_path) -> Path:
    import issue1336_gen_answers as g

    monkeypatch.setattr(g, "DATA_ROOT", tmp_path)
    out_dir = tmp_path / "gen" / "base" / "lmsys5k"
    out_dir.mkdir(parents=True)
    (out_dir / "answers.jsonl").write_text('{"prompt_idx": 0, "kept": true}\n')
    (out_dir / "audit.json").write_text("{}\n")
    return out_dir


def test_gen_skip_branch_reuploads_on_hub_miss(tmp_path, monkeypatch):
    """Local outputs + incomplete Hub + upload => rollout text re-uploaded."""
    import issue1336_gen_answers as g

    _gen_cell(monkeypatch, tmp_path)
    monkeypatch.setattr(g, "_hf_gen_state", lambda slug, corpus: (False, None))
    calls: list[tuple] = []
    monkeypatch.setattr(g, "_upload_gen_outputs", lambda s, c, d: calls.append((s, c)))
    g.run_generation("base", ["lmsys5k"], smoke=False, upload=True)
    assert calls == [("base", "lmsys5k")]


def test_gen_skip_branch_no_reupload_when_hub_complete(tmp_path, monkeypatch):
    import issue1336_gen_answers as g

    _gen_cell(monkeypatch, tmp_path)
    monkeypatch.setattr(g, "_hf_gen_state", lambda slug, corpus: (True, None))
    monkeypatch.setattr(
        g, "_upload_gen_outputs", lambda *a: pytest.fail("must not re-upload a complete cell")
    )
    g.run_generation("base", ["lmsys5k"], smoke=False, upload=True)


def test_gen_hf_state_reads_single_and_sharded_shapes(tmp_path, monkeypatch):
    """Real _hf_gen_state body against a signature-conformant listing fake."""
    import issue1336_gen_answers as g

    from explore_persona_space.orchestrate import hub

    prefix = g._hf_gen_prefix("base", "lmsys5k")
    manifest = {"parts": ["answers.shard00.jsonl"], "sha256s": ["x"], "line_counts": [1]}
    mpath = tmp_path / "answers.manifest.json"
    mpath.write_text(json.dumps(manifest))
    monkeypatch.setattr(g, "_download_one", lambda pf: mpath)

    def _with(present: set[str]):
        # Signature mirrors hub.list_hf_files_under_path (the network boundary).
        def fake_list(api, repo_id, path, *, repo_type="model", revision=None):
            assert path == prefix and repo_type == "dataset"
            return sorted(present)

        monkeypatch.setattr(hub, "list_hf_files_under_path", fake_list)
        return g._hf_gen_state("base", "lmsys5k")

    side = {f"{prefix}/allowlist.json", f"{prefix}/audit.json"}
    assert _with(side | {f"{prefix}/answers.jsonl"}) == (True, None)
    assert _with({f"{prefix}/answers.jsonl"})[0] is False  # side files missing
    ok, m = _with(side | {f"{prefix}/answers.manifest.json", f"{prefix}/answers.shard00.jsonl"})
    assert ok is True and m == manifest
    ok, _ = _with(side | {f"{prefix}/answers.manifest.json"})  # part missing
    assert ok is False
    assert _with(side)[0] is False  # no answers artifact at all
    assert _with(set())[0] is False  # absent prefix lists empty


def test_gen_upload_shards_oversize_answers_roundtrip(tmp_path, monkeypatch):
    """Real _upload_gen_outputs body: >threshold answers upload as shards +
    manifest (original excluded), and _reassemble_answers restores the exact
    bytes — the upload-policy text-split rule (r1 review Minor 5)."""
    import huggingface_hub
    import issue1336_gen_answers as g

    out_dir = tmp_path / "cell"
    out_dir.mkdir()
    rows = [json.dumps({"prompt_idx": i, "response": "x" * 40}) for i in range(60)]
    payload = ("\n".join(rows) + "\n").encode()
    (out_dir / "answers.jsonl").write_bytes(payload)
    monkeypatch.setattr(g, "_TEXT_SPLIT_THRESHOLD", 500)
    monkeypatch.setattr(g, "_SHARD_MAX_BYTES", 400)
    fake_upload = create_autospec(huggingface_hub.upload_folder)
    monkeypatch.setattr(huggingface_hub, "upload_folder", fake_upload)

    g._upload_gen_outputs("base", "lmsys5k", out_dir)
    assert "answers.jsonl" in fake_upload.call_args.kwargs["ignore_patterns"]
    manifest = json.loads((out_dir / g.ANSWERS_MANIFEST).read_text())
    assert sum(manifest["line_counts"]) == 60
    assert len(manifest["parts"]) >= 2
    for part in manifest["parts"]:
        assert (out_dir / part).stat().st_size <= 400

    # Round-trip: reassembly from the shard parts restores the exact bytes.
    monkeypatch.setattr(g, "_download_one", lambda pf: out_dir / Path(pf).name)
    (out_dir / "answers.jsonl").unlink()
    g._reassemble_answers(out_dir, "prefix", manifest)
    assert (out_dir / "answers.jsonl").read_bytes() == payload


def test_gen_upload_small_answers_stays_single_file(tmp_path, monkeypatch):
    """Under-threshold answers upload as-is; stale shard files are cleared."""
    import huggingface_hub
    import issue1336_gen_answers as g

    out_dir = tmp_path / "cell"
    out_dir.mkdir()
    (out_dir / "answers.jsonl").write_text('{"prompt_idx": 0}\n')
    (out_dir / "answers.shard00.jsonl").write_text("stale\n")
    (out_dir / g.ANSWERS_MANIFEST).write_text("{}")
    fake_upload = create_autospec(huggingface_hub.upload_folder)
    monkeypatch.setattr(huggingface_hub, "upload_folder", fake_upload)

    g._upload_gen_outputs("base", "lmsys5k", out_dir)
    assert "answers.jsonl" not in fake_upload.call_args.kwargs["ignore_patterns"]
    assert not (out_dir / "answers.shard00.jsonl").exists()
    assert not (out_dir / g.ANSWERS_MANIFEST).exists()


def test_gen_download_one_body(tmp_path, monkeypatch):
    """Real _download_one body against an autospec'd hf_hub_download."""
    import huggingface_hub
    import issue1336_gen_answers as g

    monkeypatch.setattr(g, "DATA_ROOT", tmp_path)
    target = tmp_path / "f.json"
    target.write_text("{}")
    fake_dl = create_autospec(huggingface_hub.hf_hub_download, return_value=str(target))
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_dl)
    got = g._download_one("prefix/f.json")
    assert got == target
    kwargs = fake_dl.call_args.kwargs
    assert kwargs["filename"] == "prefix/f.json" and kwargs["repo_type"] == "dataset"


# ---------------------------------------------------------------------------
# extract: HF-resume of a COMPLETE cell (plan v9 route 1 resume path)
# ---------------------------------------------------------------------------
def _fake_tree(monkeypatch, names: list[str] | None):
    """Signature-conformant HfApi.list_repo_tree fake (the network boundary).

    ``names=None`` simulates an absent prefix (EntryNotFoundError)."""
    from types import SimpleNamespace

    import huggingface_hub
    from huggingface_hub.utils import EntryNotFoundError

    def fake_list_repo_tree(
        self, repo_id, path_in_repo=None, *, repo_type=None, revision=None, recursive=False, **kw
    ):
        assert repo_type == "dataset" and path_in_repo.endswith(f"turnstore_{STEM}")
        if names is None:
            raise EntryNotFoundError("no such prefix")
        return [SimpleNamespace(path=f"{path_in_repo}/{n}") for n in names]

    monkeypatch.setattr(huggingface_hub.HfApi, "list_repo_tree", fake_list_repo_tree)


def test_extract_hf_resume_complete_cell_downloads_and_marks_done(tmp_path, monkeypatch):
    """Real _try_hf_resume body: complete Hub cell -> files staged flat into
    out_dir + done marker with uploaded=True (done == uploaded holds)."""
    import huggingface_hub
    import issue1336_extract_turnstore as ext

    names = [f"{STEM}_shard{i:03d}.{e}" for i in range(2) for e in ("pt", "json")]
    _fake_tree(monkeypatch, names)

    def fake_download(repo_id=None, repo_type=None, filename=None, local_dir=None, **kw):
        # Mirrors hf_hub_download's local_dir staging: file lands at the
        # repo-relative path under local_dir.
        dest = Path(local_dir) / filename
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text("payload")
        return str(dest)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
    _no_extract(monkeypatch, ext)

    done = ext._try_hf_resume(tmp_path, STEM)
    assert done is not None and done["uploaded"] is True and done["hf_resumed"] is True
    assert done["n_shards"] == 2
    for n in names:
        assert (tmp_path / n).exists(), f"{n} not staged flat into out_dir"
    marker = json.loads((tmp_path / f"{STEM}.done.json").read_text())
    assert marker["uploaded"] is True and marker["hf_resumed"] is True


def test_extract_hf_resume_refuses_half_done_cell(tmp_path, monkeypatch):
    """A partial Hub prefix (missing sidecar / shard gap) raises — a half-done
    cell is never silently resumed NOR silently re-extracted."""
    import issue1336_extract_turnstore as ext

    # shard000 complete, shard001 missing its .json sidecar
    _fake_tree(monkeypatch, [f"{STEM}_shard000.pt", f"{STEM}_shard000.json", f"{STEM}_shard001.pt"])
    with pytest.raises(AssertionError, match="INCOMPLETE"):
        ext._try_hf_resume(tmp_path, STEM)
    assert not (tmp_path / f"{STEM}.done.json").exists()

    # shard-index gap (0 absent) is also refused
    _fake_tree(monkeypatch, [f"{STEM}_shard001.pt", f"{STEM}_shard001.json"])
    with pytest.raises(AssertionError, match="INCOMPLETE"):
        ext._try_hf_resume(tmp_path, STEM)


def test_extract_hf_resume_absent_prefix_returns_none(tmp_path, monkeypatch):
    """A cell never uploaded returns None (fresh extraction proceeds)."""
    import issue1336_extract_turnstore as ext

    _fake_tree(monkeypatch, None)
    assert ext._try_hf_resume(tmp_path, STEM) is None
    assert not (tmp_path / f"{STEM}.done.json").exists()
