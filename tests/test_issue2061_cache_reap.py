"""#2061 crash-fix R7: per-cell data-repo hub-cache reap (disk_exhaustion_p1_hub_cache).

P1's lazy shard downloads (`iter_local_shards` -> `hf_hub_download`) land
every cell's shards in the hub cache with nothing reaping them between
cells: 16 completed cells accumulated 182 GB on the 200 GB /workspace
volume; total 35-cell demand is 332.7 GB -> guaranteed ENOSPC (production
crash 2026-08-06 at cell [17/35], os error 28 in xet_get).
`reap_data_repo_hub_cache` purges the DATA repo's cache dir at each cell
boundary, scoped STRICTLY away from the SAE model cache and every other
repo dir. These tests pin: data-repo dir removed, sibling model dir
untouched, missing-dir clean no-op, and the `[cache-reap]` fix-engaged
line (bytes freed + post-reap free space) emitted in every case.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import issue2061_sae_encode as enc

DATA_CACHE_DIR = "datasets--superkaiba1--explore-persona-space-data"
MODEL_CACHE_DIR = "models--EleutherAI--sae-llama-3.1-8b-64x"


def _make_cache(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Fake hub cache root: data-repo dir (blob + snapshot symlink) + SAE model dir."""
    root = tmp_path / "hub"
    data = root / DATA_CACHE_DIR
    model = root / MODEL_CACHE_DIR
    (data / "blobs").mkdir(parents=True)
    snap = data / "snapshots" / "abc123def"
    snap.mkdir(parents=True)
    (data / "blobs" / "blob0").write_bytes(b"x" * 4096)
    (snap / "turnstore_shard000.pt").symlink_to(data / "blobs" / "blob0")
    (model / "blobs").mkdir(parents=True)
    (model / "blobs" / "sae_blob").write_bytes(b"y" * 512)
    return root, data, model


def test_reap_removes_data_repo_dir_only(tmp_path, capsys):
    """The data-repo cache dir is removed; the SAE model cache is untouched."""
    root, data, model = _make_cache(tmp_path)
    freed = enc.reap_data_repo_hub_cache(cache_root=root)
    assert not data.exists(), "data-repo cache dir must be removed"
    assert model.exists(), "SAE model cache dir must NEVER be touched"
    assert (model / "blobs" / "sae_blob").read_bytes() == b"y" * 512
    assert freed >= 4096, f"blob bytes must be counted as freed (got {freed})"
    out = capsys.readouterr().out
    assert "[cache-reap]" in out
    assert str(data) in out


def test_reap_missing_dir_is_clean_noop(tmp_path, capsys):
    """A missing data-repo dir (first cell / already reaped) is a clean no-op."""
    root = tmp_path / "hub"
    root.mkdir()
    freed = enc.reap_data_repo_hub_cache(cache_root=root)
    assert freed == 0
    out = capsys.readouterr().out
    assert "[cache-reap]" in out, "fix-engaged line must be emitted even on no-op"
    assert "freed 0 bytes" in out


def test_reap_missing_cache_root_is_clean_noop(tmp_path, capsys):
    """Even the cache ROOT missing (fresh env) is a no-op — disk_usage walks up."""
    root = tmp_path / "does-not-exist" / "hub"
    freed = enc.reap_data_repo_hub_cache(cache_root=root)
    assert freed == 0
    assert "[cache-reap]" in capsys.readouterr().out


def test_reap_line_reports_bytes_and_free_space(tmp_path, capsys):
    """The [cache-reap] line carries freed bytes AND post-reap free space."""
    root, _data, _model = _make_cache(tmp_path)
    enc.reap_data_repo_hub_cache(cache_root=root)
    out = capsys.readouterr().out
    lines = [ln for ln in out.splitlines() if ln.startswith("[cache-reap]")]
    assert len(lines) == 1
    assert "freed" in lines[0]
    assert "post-reap free" in lines[0]
