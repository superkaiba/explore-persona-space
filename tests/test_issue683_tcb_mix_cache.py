"""Issue #683 — regression: per-source mix-cache isolation in t_{C,B} extraction.

Pins the BLOCKER fix for ``tcb-mix-cache-source-collision``
(``scripts/issue683_extract_tcb.py::_download_mix``).

The cache key was ``local_root / Path(rel).name``, which flattens both
``.../villain/train_pool.jsonl`` and ``.../comedian/train_pool.jsonl`` to the
SAME basename ``train_pool.jsonl`` under ``local_root``. Because ``_download_mix``
returns the first cached file when it already exists, a production
``--source-list "villain,comedian"`` run would compute comedian's ``t_cb`` from
VILLAIN's cached rows — silently corrupting the SECOND source's data-side key.
The fix namespaces the cache dir by source (``local_root / source / ...``).

This test FAILS pre-fix (both sources resolve to the same cached path with
villain's content) and PASSES post-fix (distinct paths, per-source content).

CPU-only, no GPU, no network (``hf_hub_download`` is monkeypatched).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import issue683_extract_tcb as tcb  # noqa: E402


def test_download_mix_isolates_sources_sharing_a_basename(monkeypatch, tmp_path):
    """Two sources whose mix paths share a basename must NOT collide in cache.

    Reproduces the production ``villain,comedian`` shape: both resolve to a repo
    path ending in ``train_pool.jsonl``. The two returned local paths must be
    DIFFERENT and each must hold its OWN source's content — no cross-source
    corruption of the second-downloaded source.
    """
    # Both sources resolve to a repo-relative path with the SAME basename — the
    # exact collision condition (sycophancy/<source>/train_pool.jsonl).
    rel_by_source = {
        "villain": "issue683_xfer/sycophancy/villain/train_pool.jsonl",
        "comedian": "issue683_xfer/sycophancy/comedian/train_pool.jsonl",
    }
    monkeypatch.setattr(
        tcb,
        "_resolve_mix_path",
        lambda behavior, source: (rel_by_source[source], behavior),
    )

    # Stub the HF download: write distinct per-source content to a fake "remote"
    # path and return it (mimicking hf_hub_download's snapshot path return).
    remote_root = tmp_path / "hf_remote"
    content_by_source = {
        "villain": '{"source": "villain", "row": 1}\n',
        "comedian": '{"source": "comedian", "row": 1}\n',
    }

    def _fake_hf_hub_download(repo_id, rel, *, repo_type, revision):
        # Reverse-map rel -> source so each call writes the right content.
        source = next(s for s, r in rel_by_source.items() if r == rel)
        dst = remote_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(content_by_source[source])
        return str(dst)

    # _download_mix imports hf_hub_download from huggingface_hub at call time,
    # so patch it on the source module.
    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", _fake_hf_hub_download)

    local_root = tmp_path / "mix_cache" / "sycophancy"

    villain_path = tcb._download_mix("sycophancy", "villain", local_root)
    comedian_path = tcb._download_mix("sycophancy", "comedian", local_root)

    # 1) The two cached paths must be DIFFERENT (no basename flattening).
    assert villain_path != comedian_path, (
        f"villain and comedian collided to the same cache path: {villain_path}"
    )

    # 2) Each cached file must hold its OWN source's content (no cross-corruption:
    #    comedian must NOT be served villain's cached rows).
    assert villain_path.read_text() == content_by_source["villain"]
    assert comedian_path.read_text() == content_by_source["comedian"]


def test_download_mix_caches_per_source_on_second_call(monkeypatch, tmp_path):
    """A second call for the same source returns the cached file without re-download."""
    monkeypatch.setattr(
        tcb,
        "_resolve_mix_path",
        lambda behavior, source: (
            f"issue683_xfer/sycophancy/{source}/train_pool.jsonl",
            behavior,
        ),
    )

    calls: list[str] = []

    def _fake_hf_hub_download(repo_id, rel, *, repo_type, revision):
        calls.append(rel)
        dst = tmp_path / "hf_remote" / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text('{"row": 1}\n')
        return str(dst)

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", _fake_hf_hub_download)

    local_root = tmp_path / "mix_cache" / "sycophancy"
    first = tcb._download_mix("sycophancy", "villain", local_root)
    second = tcb._download_mix("sycophancy", "villain", local_root)

    assert first == second
    # Only one network download for the same source (second call hits the cache).
    assert calls == ["issue683_xfer/sycophancy/villain/train_pool.jsonl"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
