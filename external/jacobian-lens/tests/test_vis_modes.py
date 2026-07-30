# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0
"""build_page / write_slice_files smoke tests with synthetic SliceData (no model)."""

from __future__ import annotations

import base64
import gzip
import json

import numpy as np
import pytest

import jlens.vis as vis
from jlens.vis import SliceData, build_page, write_slice_files


@pytest.fixture(autouse=True)
def _no_d3_fetch(monkeypatch):
    """Embed mode fetches d3 to inline it; pre-fill the template cache so
    tests never hit the network."""
    monkeypatch.setitem(vis._TEMPLATE_FOR_MODE, "embed", vis._template("fetch"))


def _synthetic_slice(seq_len: int = 6, n_layers: int = 4, top_n: int = 3) -> SliceData:
    rng = np.random.default_rng(0)
    layers = list(range(0, n_layers * 2, 2))
    tracked = sorted(rng.choice(500, size=8, replace=False).tolist())
    top_ids = rng.choice(500, size=(seq_len, n_layers, top_n)).astype(np.int32)
    return SliceData(
        seq_len=seq_len,
        layers=layers,
        context_token_ids=list(range(seq_len)),
        context_token_strs=[f"t{i}" for i in range(seq_len)],
        top_ids=top_ids,
        top_ranks=rng.integers(0, 50, (seq_len, n_layers, top_n)).astype(np.int32),
        tracked_token_ids=tracked,
        rank_tensor=rng.integers(0, 50000, (seq_len, n_layers, len(tracked))).astype(
            np.int32
        ),
        vocab_fragment={
            int(t): f"tok{t}" for t in np.unique(top_ids).tolist() + tracked
        },
    )


def test_embed_mode_is_self_contained():
    sd = _synthetic_slice()
    page, raw, payload = build_page(
        sd, "p", title="T", description="d", pinned_token_ids=set(), mode="embed"
    )
    assert "__BOOTSTRAP__" not in page and payload > 0
    boot = json.loads(
        page.split('id="bootstrap"')[1].split(">", 1)[1].split("</script>")[0]
    )
    assert boot["mode"] == "embed"
    assert "slice.bin" in boot["files"]
    assert len([k for k in boot["files"] if k.startswith("ranks/")]) == len(
        sd.tracked_token_ids
    )
    # round-trip slice.bin
    raw_sl = gzip.decompress(base64.b64decode(boot["files"]["slice.bin"]))
    n = sd.top_ids.size
    np.testing.assert_array_equal(
        np.frombuffer(raw_sl, "<i4", n, 0), sd.top_ids.ravel()
    )
    np.testing.assert_array_equal(
        np.frombuffer(raw_sl, "<i4", n, n * 4), sd.top_ranks.ravel()
    )


def test_fetch_mode_writes_sidecars(tmp_path):
    sd = _synthetic_slice()
    page, _, payload = build_page(
        sd,
        "p",
        title="T",
        description="d",
        pinned_token_ids=set(),
        mode="fetch",
        out_dir=tmp_path,
    )
    assert (tmp_path / "meta.json").exists()
    assert (tmp_path / "slice.bin").exists()
    assert len(list((tmp_path / "ranks").glob("*.bin"))) == len(sd.tracked_token_ids)
    meta = json.loads((tmp_path / "meta.json").read_text())
    assert meta["T"] == sd.seq_len and meta["layers"] == sd.layers
    assert payload > 0


def test_embed_mode_raises_when_d3_fetch_fails(monkeypatch):
    monkeypatch.delitem(vis._TEMPLATE_FOR_MODE, "embed", raising=False)

    def no_network(*args, **kwargs):
        raise OSError("no network")

    monkeypatch.setattr("urllib.request.urlopen", no_network)
    with pytest.raises(RuntimeError, match="could not fetch d3"):
        build_page(_synthetic_slice(), "p", title="T", description="d", mode="embed")


def test_rank_bin_roundtrip(tmp_path):
    sd = _synthetic_slice()
    write_slice_files(
        sd, tmp_path, prompt="p", title="T", description="d", pinned_token_ids=set()
    )
    tid = sd.tracked_token_ids[0]
    raw = gzip.decompress((tmp_path / "ranks" / f"{tid}.bin").read_bytes())
    np.testing.assert_array_equal(
        np.frombuffer(raw, "<i4"), sd.rank_tensor[:, :, 0].ravel()
    )
