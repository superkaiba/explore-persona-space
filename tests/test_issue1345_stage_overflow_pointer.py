"""Issue #1345 char-capture-ladders r3 — overflow-pointer-aware turnstore stager.

Pins the #1034 proactive-overflow contract of
``scripts/issue1345_stage_char_stories.stage_variant_turnstore``: when the
canonical prefix carries an ``OVERFLOW_POINTER.json`` breadcrumb (the capture
job's shards were rerouted to the PRIVATE overflow repo over the
public-storage ceiling), the stager follows the pointer, re-stages the SAME
prefix from ``pointer["overflow_repo"]`` with ``repo_type="model"``
(``upload_dir_sharded`` routes overflow shards as a MODEL repo) at the
DEFAULT branch (``revision=None`` — the pinned-revision kwarg is
canonical-only), and lands the SAME flat consumer layout as the
canonical-path case (pointer excluded).

The ONLY faked seam is the Hub network boundary: ``hub.stage_hub_prefix`` is
replaced by a signature-mirroring local-copy fake serving per-repo fixture
dirs — the pointer detection, prefix-drift assert, merged move loop,
pt-shard assert, and sentinel write all execute the REAL function body.
No real uploads/downloads.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_REPO_ROOT / "scripts"),):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import issue1345_stage_char_stories as stager  # noqa: E402

VARIANT = "char_helios"
PREFIX = f"issue1345_framing/{VARIANT}/analysis_tensors/turnstore"
OVERFLOW_REPO = "superkaiba1/explore-persona-space-overflow"


def _mk_fake_stage(sources: dict[str, Path], calls: list[dict]):
    """Signature-mirroring local-copy fake of ``hub.stage_hub_prefix``.

    Copies ``sources[repo_id]``'s files to ``dest_dir/<prefix>/<rel>`` (the
    real helper's verbatim prefix-mirror layout) and raises FileNotFoundError
    on an empty source, matching the real fail-loud contract.
    """

    def fake_stage_hub_prefix(
        repo_id: str,
        prefix: str,
        dest_dir,
        *,
        repo_type: str = "dataset",
        revision: str | None = None,
        token: str | None = None,
        max_workers: int = 6,
    ) -> list[Path]:
        calls.append(
            {
                "repo_id": repo_id,
                "prefix": prefix,
                "repo_type": repo_type,
                "revision": revision,
            }
        )
        src = sources[repo_id]
        files = sorted(p for p in src.rglob("*") if p.is_file())
        if not files:
            raise FileNotFoundError(f"no files under {repo_id}@{revision}:{prefix}")
        out: list[Path] = []
        for p in files:
            target = Path(dest_dir) / prefix / p.relative_to(src)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, target)
            out.append(target)
        return out

    return fake_stage_hub_prefix


def _write_shards(src: Path, names: list[str]) -> None:
    src.mkdir(parents=True, exist_ok=True)
    for n in names:
        (src / n).write_bytes(b"fake-bytes-" + n.encode())


def _write_pointer(src: Path, *, path_in_repo: str = PREFIX, drop_repo_key: bool = False) -> None:
    src.mkdir(parents=True, exist_ok=True)
    payload = {"overflow_repo": OVERFLOW_REPO, "path_in_repo": path_in_repo, "ts": 0.0}
    if drop_repo_key:
        del payload["overflow_repo"]
    (src / "OVERFLOW_POINTER.json").write_text(json.dumps(payload))


def _staged_layout(dest: Path) -> dict[str, bytes]:
    """Relative-path -> bytes map of the staged dir, sentinel excluded."""
    return {
        str(p.relative_to(dest)): p.read_bytes()
        for p in sorted(dest.rglob("*"))
        if p.is_file() and p.name != ".staged_complete"
    }


def _run(tmp_path: Path, monkeypatch, sources: dict[str, Path], tag: str):
    calls: list[dict] = []
    monkeypatch.setattr(stager.hub, "stage_hub_prefix", _mk_fake_stage(sources, calls))
    dest_root = tmp_path / f"stage_root_{tag}"
    dest_root.mkdir()
    dest = stager.stage_variant_turnstore(VARIANT, dest_root=dest_root, revision=None)
    return dest, calls


SHARD_NAMES = ["turnstore_shard000.pt", "turnstore_shard001.pt", "manifest.json"]


def test_pointer_path_lands_layout_identical_to_canonical(tmp_path, monkeypatch):
    """Pointer-only canonical prefix -> shard set re-staged from the overflow
    repo (model repo_type, default branch) into a layout byte-identical to
    the canonical-path case; pointer excluded; sentinel records the repo."""
    # Canonical-path reference run: shards live on the canonical repo.
    canon_src = tmp_path / "canon_direct"
    _write_shards(canon_src, SHARD_NAMES)
    dest_ref, calls_ref = _run(tmp_path, monkeypatch, {stager.HF_DATA_REPO: canon_src}, "canonical")
    assert [c["repo_id"] for c in calls_ref] == [stager.HF_DATA_REPO]
    assert calls_ref[0]["repo_type"] == "dataset"

    # Overflow-routed run: canonical prefix holds ONLY the pointer breadcrumb.
    ptr_src = tmp_path / "canon_pointer_only"
    _write_pointer(ptr_src)
    of_src = tmp_path / "overflow_src"
    _write_shards(of_src, SHARD_NAMES)
    dest_of, calls_of = _run(
        tmp_path, monkeypatch, {stager.HF_DATA_REPO: ptr_src, OVERFLOW_REPO: of_src}, "overflow"
    )

    assert _staged_layout(dest_of) == _staged_layout(dest_ref)
    assert not (dest_of / "OVERFLOW_POINTER.json").exists()
    assert [c["repo_id"] for c in calls_of] == [stager.HF_DATA_REPO, OVERFLOW_REPO]
    of_call = calls_of[1]
    assert of_call["repo_type"] == "model"  # upload_sharded routes overflow as model
    assert of_call["revision"] is None  # default branch — pin is canonical-only
    assert of_call["prefix"] == PREFIX
    sentinel = json.loads((dest_of / ".staged_complete").read_text())
    assert sentinel["overflow_repo"] == OVERFLOW_REPO
    assert sentinel["n_pt_shards"] == 2
    ref_sentinel = json.loads((dest_ref / ".staged_complete").read_text())
    assert ref_sentinel["overflow_repo"] is None
    # No scratch residue on either path.
    for root in (dest_ref.parent, dest_of.parent):
        assert not list(root.glob(".hfstage_ts*")), list(root.glob(".hfstage_ts*"))


def test_reactive_split_shard_set_merges(tmp_path, monkeypatch):
    """Pointer + partial canonical shard set (reactive mid-store reroute):
    canonical and overflow halves merge into one flat consumer layout."""
    canon_src = tmp_path / "canon_partial"
    _write_shards(canon_src, ["turnstore_shard000.pt"])
    _write_pointer(canon_src)
    of_src = tmp_path / "overflow_rest"
    _write_shards(of_src, ["turnstore_shard001.pt", "manifest.json"])
    dest, _calls = _run(
        tmp_path, monkeypatch, {stager.HF_DATA_REPO: canon_src, OVERFLOW_REPO: of_src}, "split"
    )
    assert set(_staged_layout(dest)) == set(SHARD_NAMES)
    assert not (dest / "OVERFLOW_POINTER.json").exists()


def test_pointer_prefix_drift_fails_loud(tmp_path, monkeypatch):
    ptr_src = tmp_path / "canon_drift"
    _write_pointer(ptr_src, path_in_repo="issue1345_framing/OTHER/analysis_tensors/turnstore")
    with pytest.raises(AssertionError, match="prefix drift"):
        _run(tmp_path, monkeypatch, {stager.HF_DATA_REPO: ptr_src}, "drift")


def test_pointer_missing_repo_key_fails_loud(tmp_path, monkeypatch):
    ptr_src = tmp_path / "canon_malformed"
    _write_pointer(ptr_src, drop_repo_key=True)
    with pytest.raises(KeyError, match="overflow_repo"):
        _run(tmp_path, monkeypatch, {stager.HF_DATA_REPO: ptr_src}, "malformed")


def test_pointer_to_empty_overflow_fails_loud(tmp_path, monkeypatch):
    """Pointer says overflow but nothing is there — the issue841 fail-loud
    contract (FileNotFoundError from stage_hub_prefix propagates)."""
    ptr_src = tmp_path / "canon_ptr_empty_of"
    _write_pointer(ptr_src)
    of_src = tmp_path / "overflow_empty"
    of_src.mkdir()
    with pytest.raises(FileNotFoundError):
        _run(
            tmp_path, monkeypatch, {stager.HF_DATA_REPO: ptr_src, OVERFLOW_REPO: of_src}, "emptyof"
        )
