"""Pins for the #2054 round-11 class sweep: upload-helper empty-return discard
(code-review v6 concern ``upload-return-discard-sibling-scripts``).

Round 10 (commit ac19da0e46) landed the capture-and-raise contract at the
three sites the v5 Major named; the v6 "Bug-class sweep" enumerated SIX
pre-existing sibling sites in this task's own files that still discarded the
``hub._upload_folder_filtered`` return (fail-SOFT by RETURN on every failure
shape — missing HF_TOKEN, incomplete expected-set verify, terminal
``except Exception`` all log and return ``""``). These tests pin the
capture-and-raise contract at all six (each raises-on-empty pin fails
pre-fix — the pre-fix bodies discard the return and never raise — and
passes post-fix):

- ``issue2054_phase_b._upload_to_hf``  (spliced files — Phase-B dispatch)
- ``issue2054_phase_c._upload_to_hf``  (on-policy rollouts — model generations)
- ``issue2054_phase_d._upload_to_hf``  (cell_c splices)
- ``issue2054_fits._upload_to_hf``     (fit JSONs)
- ``issue2054_ladder._upload_to_hf``   (rung JSONs)
- ``issue2054_capture._upload_to_hf``  (activations)

Boundary fakes are ``create_autospec`` on the real ``hub._upload_folder_filtered``
(signature-conformant by construction), patched at the SOURCE module — every
caller imports the helper inside the function body at call time. No network,
no worktree paths; ``tmp_path`` only. Fixture rows are synthetic placeholder
text — no real-corpus content.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import create_autospec

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_REPO_ROOT / "scripts"), str(_REPO_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import issue2054_capture as cap  # noqa: E402
import issue2054_fits as fits  # noqa: E402
import issue2054_ladder as ladder  # noqa: E402
import issue2054_phase_b as pb  # noqa: E402
import issue2054_phase_c as pc  # noqa: E402
import issue2054_phase_d as pd_  # noqa: E402

from explore_persona_space.orchestrate import hub  # noqa: E402

_URL = "https://huggingface.co/datasets/x"


def _patch_folder_upload(monkeypatch: pytest.MonkeyPatch, ret: str):
    fake = create_autospec(hub._upload_folder_filtered, return_value=ret)
    monkeypatch.setattr(hub, "_upload_folder_filtered", fake)
    return fake


def _variant_file(tmp_path: Path, name: str) -> Path:
    p = tmp_path / name
    p.write_text('{"conv_id": "mt_0", "text": "x"}\n', encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# phase_b (spliced_inserted)


def test_phase_b_upload_raises_on_empty_return(tmp_path, monkeypatch):
    _patch_folder_upload(monkeypatch, "")
    paths = {"char_helios": _variant_file(tmp_path, "spliced_char_helios.jsonl")}
    with pytest.raises(RuntimeError, match="spliced bulk upload failed or incomplete"):
        pb._upload_to_hf(paths, tmp_path)


def test_phase_b_upload_passes_on_url_return(tmp_path, monkeypatch):
    fake = _patch_folder_upload(monkeypatch, _URL)
    paths = {"char_helios": _variant_file(tmp_path, "spliced_char_helios.jsonl")}
    pb._upload_to_hf(paths, tmp_path)  # no raise
    assert fake.call_count == 1


# ---------------------------------------------------------------------------
# phase_c (on_policy — model generations, never discardable)


def test_phase_c_upload_raises_on_empty_return(tmp_path, monkeypatch):
    _patch_folder_upload(monkeypatch, "")
    paths = {"char_helios": _variant_file(tmp_path, "onpolicy_char_helios.jsonl")}
    with pytest.raises(RuntimeError, match="on-policy bulk upload failed or incomplete"):
        pc._upload_to_hf(paths, tmp_path, "qwen2.5-7b")


def test_phase_c_upload_passes_on_url_return(tmp_path, monkeypatch):
    fake = _patch_folder_upload(monkeypatch, _URL)
    paths = {"char_helios": _variant_file(tmp_path, "onpolicy_char_helios.jsonl")}
    pc._upload_to_hf(paths, tmp_path, "qwen2.5-7b")  # no raise
    assert fake.call_count == 1


# ---------------------------------------------------------------------------
# phase_d (cell_c)


def test_phase_d_upload_raises_on_empty_return(tmp_path, monkeypatch):
    _patch_folder_upload(monkeypatch, "")
    paths = {"op_char_helios": _variant_file(tmp_path, "cellc_op_char_helios.jsonl")}
    with pytest.raises(RuntimeError, match="cell_c bulk upload failed or incomplete"):
        pd_._upload_to_hf(paths, tmp_path)


def test_phase_d_upload_passes_on_url_return(tmp_path, monkeypatch):
    fake = _patch_folder_upload(monkeypatch, _URL)
    paths = {"op_char_helios": _variant_file(tmp_path, "cellc_op_char_helios.jsonl")}
    pd_._upload_to_hf(paths, tmp_path)  # no raise
    assert fake.call_count == 1


# ---------------------------------------------------------------------------
# fits (fit JSONs; files must be non-empty and share one parent)


def test_fits_upload_raises_on_empty_return(tmp_path, monkeypatch):
    _patch_folder_upload(monkeypatch, "")
    fits_by_cell = {"cellA": _variant_file(tmp_path, "fit_cellA.json")}
    with pytest.raises(RuntimeError, match="fit-JSON bulk upload failed or incomplete"):
        fits._upload_to_hf(fits_by_cell, "modelA")


def test_fits_upload_passes_on_url_return(tmp_path, monkeypatch):
    fake = _patch_folder_upload(monkeypatch, _URL)
    fits_by_cell = {"cellA": _variant_file(tmp_path, "fit_cellA.json")}
    fits._upload_to_hf(fits_by_cell, "modelA")  # no raise
    assert fake.call_count == 1


# ---------------------------------------------------------------------------
# ladder (rung JSONs)


def test_ladder_upload_raises_on_empty_return(tmp_path, monkeypatch):
    _patch_folder_upload(monkeypatch, "")
    pair_paths = [_variant_file(tmp_path, "pair_a.json")]
    with pytest.raises(RuntimeError, match="rung-JSON bulk upload failed or incomplete"):
        ladder._upload_to_hf(pair_paths)


def test_ladder_upload_passes_on_url_return(tmp_path, monkeypatch):
    fake = _patch_folder_upload(monkeypatch, _URL)
    pair_paths = [_variant_file(tmp_path, "pair_a.json")]
    ladder._upload_to_hf(pair_paths)  # no raise
    assert fake.call_count == 1


# ---------------------------------------------------------------------------
# capture (activations; root is p.parent.parent — nest one level)


def test_capture_upload_raises_on_empty_return(tmp_path, monkeypatch):
    _patch_folder_upload(monkeypatch, "")
    sub = tmp_path / "char_helios"
    sub.mkdir()
    acts = {"char_helios": _variant_file(sub, "acts_char_helios.npz")}
    with pytest.raises(RuntimeError, match="activation bulk upload failed or incomplete"):
        cap._upload_to_hf(acts, "qwen25-7b")


def test_capture_upload_passes_on_url_return(tmp_path, monkeypatch):
    fake = _patch_folder_upload(monkeypatch, _URL)
    sub = tmp_path / "char_helios"
    sub.mkdir()
    acts = {"char_helios": _variant_file(sub, "acts_char_helios.npz")}
    cap._upload_to_hf(acts, "qwen25-7b")  # no raise
    assert fake.call_count == 1
