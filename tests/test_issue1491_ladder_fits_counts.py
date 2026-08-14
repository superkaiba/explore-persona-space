"""Fail-loud count pins for the #1491 scale-ladder fits driver (#2130).

Incident: a vLLM engine-startup crash killed one greedy-pass shard (scale15
``ceiling_draw_44`` shard4, 2026-08-05T19:58Z), leaving 875/1000 ceiling
pairs; ``_reliability_ceiling`` accepted the short pairing (``n_pairs=875``,
``available=True``) and the wrong 1.5B ceiling stood ~9.5 h until a manual
anomaly stop. These tests pin the #2130 fix: a short draw / short pairing /
short ladder split RAISES ``RuntimeError`` through the REAL function bodies,
while the designed ABSENCE path (captures not yet on HF) still returns
``available: False``.

Deliberately network-free and CPU-only (everything in ``tests/`` runs in every
issue's Step 9c gate): the HF network boundary is faked with a
signature-conformant ``def`` mirroring ``issue779_ffc_n1m_fits._stream_hf_chunks``'s
real signature ``(prefix, layer, cache_dir, *, ckpt_dir, ckpt_every, fresh)``.
Fixture arrays use H=3 (never equal to any context count n) so a
transposed-shape bug cannot hide.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import issue1491_ladder_fits as LF  # noqa: E402

H = 3  # hidden dim deliberately != every fixture n, so (n, H) vs (H, n) cannot alias
HF_PREFIX = "issue1491_test/scaleXX"
CEIL_A = f"{HF_PREFIX}/ceiling_draws/seed43/final_token_capture"
CEIL_B = f"{HF_PREFIX}/ceiling_draws/seed44/final_token_capture"


def _draw(ci: list[int], seed: int = 0) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Build one fake streamed capture: (cx (n,H) fp32, vx (n,H) fp32, ci)."""
    rng = np.random.default_rng(seed)
    n = len(ci)
    cx = rng.standard_normal((n, H)).astype(np.float32)
    vx = rng.standard_normal((n, H)).astype(np.float32)
    return cx, vx, list(ci)


def _install_stream_fake(monkeypatch, responses: dict):
    """Patch F._stream_hf_chunks with a signature-conformant fake.

    ``responses`` maps the EXACT prefix string to either an
    (cx, vx, ci) tuple or an exception instance to raise. An unexpected
    prefix fails loud (KeyError) rather than fabricating data.
    """

    def _fake_stream_hf_chunks(prefix, layer, cache_dir, *, ckpt_dir, ckpt_every, fresh):
        resp = responses[prefix]
        if isinstance(resp, BaseException):
            raise resp
        return resp

    monkeypatch.setattr(LF.F, "_stream_hf_chunks", _fake_stream_hf_chunks)


# ---------------------------------------------------------------------------
# T1 — the incident shape: 875/1000 truncated pairing raises at production
#      expected_n (the plan's fail-loud pin names this test).
# ---------------------------------------------------------------------------


def test_truncated_pairing_raises(monkeypatch, tmp_path):
    full_ci = list(range(1000))
    short_ci = list(range(875))  # seed-44 side lost one shard (the incident shape)
    _install_stream_fake(
        monkeypatch,
        {CEIL_A: _draw(full_ci, seed=43), CEIL_B: _draw(short_ci, seed=44)},
    )
    with pytest.raises(RuntimeError) as exc_info:
        LF._reliability_ceiling(HF_PREFIX, layer=19, cache_dir=tmp_path)
    msg = str(exc_info.value)
    assert "875" in msg and "1000" in msg
    # The raise names both draw prefixes so the operator can go straight to HF.
    assert CEIL_A in msg and CEIL_B in msg


# ---------------------------------------------------------------------------
# T2 — full pairing at a small expected_n override stays available.
# ---------------------------------------------------------------------------


def test_full_pairing_available(monkeypatch, tmp_path):
    ci = list(range(5))
    _install_stream_fake(
        monkeypatch,
        {CEIL_A: _draw(ci, seed=43), CEIL_B: _draw(ci, seed=44)},
    )
    out = LF._reliability_ceiling(HF_PREFIX, layer=19, cache_dir=tmp_path, expected_n=5)
    assert out["available"] is True
    assert out["n_pairs"] == 5
    assert np.isfinite(out["ceiling_var_weighted_r"])
    assert np.isfinite(out["mean_per_dim_r"])


# ---------------------------------------------------------------------------
# T3 — the designed absence path is preserved (D1 must not break it).
# ---------------------------------------------------------------------------


def test_absence_path_preserved(monkeypatch, tmp_path):
    _install_stream_fake(
        monkeypatch,
        {CEIL_A: FileNotFoundError("no .pt chunks under prefix")},
    )
    out = LF._reliability_ceiling(HF_PREFIX, layer=19, cache_dir=tmp_path)
    assert out["available"] is False
    assert "not on HF" in out["reason"]


# ---------------------------------------------------------------------------
# T4 — split-count assert in _assemble_scale_layer.
# ---------------------------------------------------------------------------


def _split_responses(wc_n: int) -> dict:
    counts = {"train_25k": 25000, "val_400": 400, "test_1000": 1000, "wc_test_1k": wc_n}
    return {
        f"{HF_PREFIX}/{split}/final_token_capture": _draw(list(range(n)), seed=i)
        for i, (split, n) in enumerate(counts.items())
    }


def test_split_count_assert(monkeypatch, tmp_path):
    # (a) one split short: wc_test_1k realized 998 vs the grounded expected 999
    #     (production EXPECTED_SPLIT_N default) → RuntimeError naming the split
    #     and both counts.
    _install_stream_fake(monkeypatch, _split_responses(wc_n=998))
    with pytest.raises(RuntimeError) as exc_info:
        LF._assemble_scale_layer(HF_PREFIX, layer=19, cache_dir=tmp_path)
    msg = str(exc_info.value)
    assert "wc_test_1k" in msg
    assert "998" in msg and "999" in msg

    # (b) full counts (incl. the grounded wc_test_1k == 999) → returns normally
    #     with the realized counts recorded.
    _install_stream_fake(monkeypatch, _split_responses(wc_n=999))
    bundle = LF._assemble_scale_layer(HF_PREFIX, layer=19, cache_dir=tmp_path)
    assert bundle["n_realized"] == LF.EXPECTED_SPLIT_N
    assert bundle["X"].shape == (27399, H)
    assert len(bundle["wc_te"]) == 999

    # (c) expected_split_n=None is the explicit opt-out for a deliberately
    #     different-size reuse — the short split then assembles without raising.
    _install_stream_fake(monkeypatch, _split_responses(wc_n=998))
    bundle = LF._assemble_scale_layer(
        HF_PREFIX, layer=19, cache_dir=tmp_path, expected_split_n=None
    )
    assert bundle["n_realized"]["wc_test_1k"] == 998


# ---------------------------------------------------------------------------
# T5 — zero overlap between two FULL draws is corruption, not absence: the
#      pre-#2130 `available: False` early return stays subsumed by the raise.
# ---------------------------------------------------------------------------


def test_zero_overlap_raises(monkeypatch, tmp_path):
    ci_a = [0, 1, 2, 3]
    ci_b = [10, 11, 12, 13]  # full-length draws, disjoint context ids
    _install_stream_fake(
        monkeypatch,
        {CEIL_A: _draw(ci_a, seed=43), CEIL_B: _draw(ci_b, seed=44)},
    )
    with pytest.raises(RuntimeError) as exc_info:
        LF._reliability_ceiling(HF_PREFIX, layer=19, cache_dir=tmp_path, expected_n=4)
    assert "n_pairs=0" in str(exc_info.value)
