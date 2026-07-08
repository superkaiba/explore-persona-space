"""Regression tests: issue1073_common.load_bundle missing-'prompts' tolerance.

Crash att-20260706-071820 (#1073 P0): the pinned pass-B artifact predates the
'prompts' field in run_pass_b's save dict; load_bundle hard-asserted the key
and killed the production run. These tests pin the fix's permanent invariants:

1. a production-shape bundle (no 'prompts') regenerates the list through the
   loader seam and writes a fail-loud run-local cache (second load reads it);
2. a legacy bundle (with 'prompts') loads unchanged, no cache written;
3. a SOURCE mismatch between the loader return and the bundle's recorded
   source RAISES (never silently proceeds on a fallback corpus);
4. a LENGTH mismatch RAISES (row alignment would be unknowable).

The production regen body executes for real in tests 3-4; only the external
network boundary (issue779_collect.load_train_contexts) is faked, with a
signature-mirroring def (code-style: one production-body test per stubbed
seam; the fake mirrors ``load_train_contexts(n_contexts, smoke)``).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

import issue779_collect  # noqa: E402
import issue1073_common as I  # noqa: E402


def _write_bundle(
    path: Path,
    n: int,
    n_layers: int = 2,
    hidden: int = 4,
    *,
    prompts: list[str] | None = None,
    source: str = "smoke_fixture",
) -> Path:
    """Minimal schema-conformant pass-B bundle; omits 'prompts' unless given."""
    g = torch.Generator().manual_seed(0)
    blob = {
        "cx_last": torch.randn((n, n_layers, hidden), generator=g),
        "cx_mean": torch.randn((n, n_layers, hidden), generator=g),
        "v_x": torch.randn((n, n_layers, hidden), generator=g),
        "layers": list(range(n_layers)),
        "source": source,
        "metadata": {"fixture": "unit-test"},
    }
    if prompts is not None:
        blob["prompts"] = prompts
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(blob, path)
    return path


def test_missing_prompts_regenerates_via_smoke_seam_and_caches(tmp_path):
    p = _write_bundle(tmp_path / "b.pt", n=3)
    b = I.load_bundle(p, expected_layers=2, expected_hidden=4, min_n=2, smoke=True)
    assert b["prompts_regenerated"] is True
    assert b["prompts"] == list(I.SMOKE_PROMPTS)[:3]
    cache = I._prompts_regen_cache_path(p)
    assert cache.exists()
    b2 = I.load_bundle(p, expected_layers=2, expected_hidden=4, min_n=2, smoke=True)
    assert b2["prompts"] == b["prompts"]


def test_legacy_prompts_bundle_loads_unchanged(tmp_path):
    prompts = ["alpha", "beta", "gamma"]
    p = _write_bundle(tmp_path / "b.pt", n=3, prompts=prompts)
    b = I.load_bundle(p, expected_layers=2, expected_hidden=4, min_n=2, smoke=True)
    assert b["prompts_regenerated"] is False
    assert b["prompts"] == prompts
    assert not I._prompts_regen_cache_path(p).exists()


def test_source_mismatch_raises(tmp_path, monkeypatch):
    p = _write_bundle(tmp_path / "b.pt", n=3, source="allenai/WildChat-1M")

    def fake_load_train_contexts(n_contexts: int, smoke: bool) -> tuple[list[str], str]:
        return ["x"] * n_contexts, "HuggingFaceH4/ultrachat_200k"

    monkeypatch.setattr(issue779_collect, "load_train_contexts", fake_load_train_contexts)
    with pytest.raises(RuntimeError, match="source mismatch"):
        I.load_bundle(p, expected_layers=2, expected_hidden=4, min_n=2, smoke=False)


def test_length_mismatch_raises(tmp_path, monkeypatch):
    p = _write_bundle(tmp_path / "b.pt", n=3, source="allenai/WildChat-1M")

    def fake_load_train_contexts(n_contexts: int, smoke: bool) -> tuple[list[str], str]:
        return ["x"] * (n_contexts - 1), "allenai/WildChat-1M"

    monkeypatch.setattr(issue779_collect, "load_train_contexts", fake_load_train_contexts)
    with pytest.raises(RuntimeError, match="length mismatch"):
        I.load_bundle(p, expected_layers=2, expected_hidden=4, min_n=2, smoke=False)
