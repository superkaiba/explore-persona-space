"""Path-arithmetic regression test for task #515 Phase 0 adapter download.

Round-1 code-review (Codex + reconciler) flagged a double-``adapters/``
mismatch: ``_download_adapter`` passed ``local_dir=local_root.parent``,
so files from the repo prefix ``adapters/issue_496/...`` landed at
``adapter_root/adapters/adapters/issue_496/...`` while Phase 1 loaded
them from ``adapter_root/adapters/issue_496/warmth_<source>_seed42``.

This test pins the contract WITHOUT touching the network: we
monkey-patch ``huggingface_hub.list_repo_files`` and
``huggingface_hub.hf_hub_download`` so the path-arithmetic is
exercised end-to-end against a tmp directory. The assertion is that
the resolved adapter path matches the path Phase 1's loader will form,
and that ``config.json`` is actually present at that resolved path.
"""

from __future__ import annotations

from pathlib import Path

import pytest


def _import_dispatcher():
    """Import the dispatcher under test. The module lives under
    ``scripts/`` so it isn't on ``sys.path`` by default; add it.
    """
    import sys

    repo_root = Path(__file__).resolve().parents[1]
    scripts_dir = repo_root / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    import dispatch_warmth_manipulation_check_515 as mod  # type: ignore[import-not-found]

    return mod


def _fake_list_repo_files_factory(known_sources):
    """Return a stub for huggingface_hub.list_repo_files that emits
    the per-source merged-adapter file set for ``known_sources``."""
    files_per_source = ["config.json", "tokenizer.json", "adapter_model.safetensors"]

    def _stub(repo_id, revision, repo_type):
        out = []
        for s in known_sources:
            for f in files_per_source:
                out.append(f"adapters/issue_496/warmth_{s}_seed42/{f}")
        return out

    return _stub


def _fake_hf_hub_download_factory():
    """Return a stub for huggingface_hub.hf_hub_download that mimics
    the real helper's path semantics: write a tiny placeholder file to
    ``local_dir/filename`` and return the path."""

    def _stub(repo_id, filename, revision, local_dir, repo_type):
        out = Path(local_dir) / filename
        out.parent.mkdir(parents=True, exist_ok=True)
        if filename.endswith("config.json"):
            out.write_text('{"model_type": "qwen2", "task_id": 515}')
        elif filename.endswith(".safetensors"):
            out.write_bytes(b"FAKE-WEIGHTS")
        else:
            out.write_text("placeholder")
        return str(out)

    return _stub


def test_download_adapter_resolved_path_matches_phase1_load_path(tmp_path, monkeypatch):
    """The path returned by ``_download_adapter`` MUST equal
    ``adapter_root / f'warmth_{source}_seed42'`` where ``adapter_root``
    is the same value Phase 1 uses (``args.adapter_root / 'adapters' /
    'issue_496'``)."""
    import huggingface_hub  # noqa: F401  -- ensures package is importable

    mod = _import_dispatcher()

    # Mimic the production CLI shape.
    adapter_root_arg = tmp_path / "adapters_496"
    adapter_subroot = adapter_root_arg / "adapters" / "issue_496"
    adapter_subroot.mkdir(parents=True, exist_ok=True)

    sources = ["villain", "comedian"]
    monkeypatch.setattr(
        "huggingface_hub.list_repo_files",
        _fake_list_repo_files_factory(sources),
        raising=True,
    )
    monkeypatch.setattr(
        "huggingface_hub.hf_hub_download",
        _fake_hf_hub_download_factory(),
        raising=True,
    )

    for source in sources:
        resolved = mod._download_adapter(
            repo_id="superkaiba1/explore-persona-space",
            revision="b4390636aaecd17e2483d233c8bf73fd6ddf1318",
            source=source,
            local_root=adapter_subroot,
        )
        expected = adapter_subroot / f"warmth_{source}_seed42"
        # The Phase 1 load uses ``adapter_root / f'warmth_{source}_seed42'``
        # where ``adapter_root`` is ``adapter_subroot`` (see
        # ``_phase1_generate_all`` and ``main`` in the dispatcher).
        assert resolved == expected, (
            f"resolved adapter path {resolved} does not match "
            f"Phase 1's load path {expected} -- regression of round-1 "
            "double-'adapters/' bug"
        )
        # The presence of config.json under the resolved path is the
        # operational guarantee. Without it vLLM's LoRA loader fails
        # with OSError at first cell load.
        assert (resolved / "config.json").exists(), (
            f"config.json missing under {resolved} after download"
        )
        # And the file MUST NOT exist at the wrong (double-adapters) path
        wrong_path = adapter_subroot / "adapters" / "issue_496" / f"warmth_{source}_seed42"
        assert not wrong_path.exists(), (
            f"adapter files landed at the WRONG (double-adapters/) path {wrong_path}"
        )


def test_download_adapter_rejects_unexpected_local_root_shape(tmp_path, monkeypatch):
    """Defensive: if a caller passes a ``local_root`` that doesn't end
    in ``adapters/issue_496``, the helper must fail loud instead of
    silently writing to the wrong tree."""
    mod = _import_dispatcher()
    monkeypatch.setattr(
        "huggingface_hub.list_repo_files",
        _fake_list_repo_files_factory(["villain"]),
        raising=True,
    )
    monkeypatch.setattr(
        "huggingface_hub.hf_hub_download",
        _fake_hf_hub_download_factory(),
        raising=True,
    )

    bad_root = tmp_path / "some_other_dir"
    bad_root.mkdir(parents=True)
    with pytest.raises(RuntimeError, match="expected local_root to end in"):
        mod._download_adapter(
            repo_id="superkaiba1/explore-persona-space",
            revision="b4390636aaecd17e2483d233c8bf73fd6ddf1318",
            source="villain",
            local_root=bad_root,
        )
