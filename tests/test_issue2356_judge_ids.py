"""r7 crash-fix pins: Batch-API-safe SHORT judge item ids (#2356).

Incident (r7, 2026-08-19): judge item ids embedded the FULL 64-hex prompt_sha
(``{sha}.greedy`` / ``{sha}.s{k:02d}`` from ``_iter_rollout_items``; predictor
``row_id``s are bare shas, issue2356_fits.py:1012-1015), and the batch encoder
appends ``__{idx:05d}__{comp:02d}`` (11 chars — read from
``batch_judge._enumerate_and_check_cache``, batch_judge.py:666), so
``--wave labeling --pilot`` died fail-fast pre-API at the 64-char custom_id
cap (82 chars, batch_judge.py:668). A SECOND latent gate sits one step later:
the dispatcher's pre-submit validator enforces the Anthropic charset
``^[a-zA-Z0-9_-]{1,64}$`` (judge_dispatch._validate_custom_ids, called at
dispatch entry judge_dispatch.py:1646), which REJECTS the ``.`` joiner
outright (the #1776 400-at-batches.create class) — hence the short ids join
with ``-`` instead of the full ids' ``.``.

These tests FAIL pre-fix: ``_short_item_id`` / ``_shorten_ids`` did not
exist, and the full-id shapes violate both the length and charset gates
(demonstrated explicitly below).

No network, no worktree-absolute paths (adoption-shape rules): the helpers
are imported from ``scripts/issue2356_judge.py`` relative to this test file's
repo root; the validator/regex are the dispatcher's own.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "scripts"))

import issue2356_judge as j  # noqa: E402

from explore_persona_space.eval.judge_dispatch import (  # noqa: E402
    BATCH_CUSTOM_ID_RE,
    validate_batch_custom_ids,
)

SHA = hashlib.sha256(b"issue2356-r7").hexdigest()  # 64 hex chars
SHA2 = hashlib.sha256(b"issue2356-r7-second").hexdigest()
assert len(SHA) == len(SHA2) == 64

# The exact suffix construction batch_judge._enumerate_and_check_cache
# composes (batch_judge.py:666): f"{persona}__{idx:05d}__{comp_idx:02d}".
# idx:05d holds 5 digits for the realized waves (max 57,959 items < 100,000)
# and comp:02d holds 2 (n_draws <= 5 < 100) — worst-case width used here.
_SUFFIX = f"__{99999:05d}__{99:02d}"
assert len(_SUFFIX) == 11


def _representative_full_items() -> list[tuple[str, str, str]]:
    """Every realized item-id shape: labeling greedy + s00..s09 draws
    (_iter_rollout_items) and the predictor bare-sha row_id (fits.py:1015)."""
    ids = [f"{SHA}.greedy"] + [f"{SHA}.s{k:02d}" for k in range(10)] + [SHA2]
    return [(iid, f"q-{i}", f"a-{i}") for i, iid in enumerate(ids)]


# ---------------------------------------------------------------------------
# (a) every FINAL custom_id is within budget AND charset-legal
# ---------------------------------------------------------------------------


def test_final_custom_id_length_and_charset_all_shapes() -> None:
    items = _representative_full_items()
    short_items, _lut = j._shorten_ids(items)
    assert len(short_items) == len(items)
    composed = []
    for sid, _q, _a in short_items:
        assert "__" not in sid  # judge_graded's item_id delimiter guard
        cid = f"{sid}{_SUFFIX}"
        assert len(cid) <= 64, (cid, len(cid))
        assert BATCH_CUSTOM_ID_RE.fullmatch(cid), cid
        composed.append(cid)
    validate_batch_custom_ids(composed)  # the dispatcher's own pre-submit gate


def test_pre_fix_full_id_shapes_violate_both_gates() -> None:
    """The r7 crash shape (length) and the dotted-joiner shape (charset) are
    BOTH rejected by the dispatcher's validator — the fix must clear both."""
    crash_cid = f"{SHA}.greedy__{0:05d}__{0:02d}"
    assert len(crash_cid) == 82  # the exact reported overflow
    with pytest.raises(ValueError):
        validate_batch_custom_ids([crash_cid])
    dotted = f"{SHA[:24]}.greedy{_SUFFIX}"  # length-legal, charset-illegal
    assert len(dotted) <= 64
    with pytest.raises(ValueError):
        validate_batch_custom_ids([dotted])


def test_short_id_shapes() -> None:
    assert j._short_item_id(f"{SHA}.greedy") == f"{SHA[:24]}-greedy"
    assert j._short_item_id(f"{SHA}.s03") == f"{SHA[:24]}-s03"
    assert j._short_item_id(SHA) == SHA[:24]  # predictor bare-sha row_id


# ---------------------------------------------------------------------------
# (b) round-trip LUT fidelity
# ---------------------------------------------------------------------------


def test_shorten_ids_roundtrip_lut_fidelity() -> None:
    items = _representative_full_items()
    short_items, lut = j._shorten_ids(items)
    # every short id translates back to exactly its originating full id
    assert [lut[sid] for sid, _q, _a in short_items] == [iid for iid, _q, _a in items]
    # (question, answer) pairing + order preserved
    assert [(q, a) for _s, q, a in short_items] == [(q, a) for _f, q, a in items]
    assert len(lut) == len(items)  # bijective over the set


def test_cid_to_item_id_decodes_short_ids() -> None:
    sid = j._short_item_id(f"{SHA}.s07")
    assert j._cid_to_item_id(f"{sid}__00007__01") == sid


def test_labels_from_result_translates_short_to_full(tmp_path: Path) -> None:
    """The seam boundary re-keys labels to FULL ids (durable artifacts never
    see a short id), and an unknown short id fails loud."""
    full = f"{SHA}.greedy"
    sid = j._short_item_id(full)
    save_raw = tmp_path / "sr.json"
    save_raw.write_text(
        json.dumps(
            {
                "all_scores": {
                    f"{sid}__00000__00": {"reasoning": "r", "label": "COMPLY", "score": 100}
                }
            }
        ),
        encoding="utf-8",
    )
    result = SimpleNamespace(scores={sid: 100.0})
    labels, _audit = j._labels_from_result(result, save_raw, {sid: full})
    assert labels == {full: {"score": 100.0, "label": "engage"}}
    with pytest.raises(KeyError):
        j._labels_from_result(result, save_raw, {})  # never silently kept short


# ---------------------------------------------------------------------------
# (c) collision raise fires on a crafted duplicate
# ---------------------------------------------------------------------------


def test_shorten_ids_collision_raises() -> None:
    a = "a" * j.SHORT_SHA_LEN + "b" * (64 - j.SHORT_SHA_LEN)
    b = "a" * j.SHORT_SHA_LEN + "c" * (64 - j.SHORT_SHA_LEN)
    assert a != b and j._short_item_id(a) == j._short_item_id(b)
    with pytest.raises(ValueError, match="collision"):
        j._shorten_ids([(f"{a}.greedy", "q", "x"), (f"{b}.greedy", "q", "y")])


def test_shorten_ids_duplicate_full_id_is_not_a_collision() -> None:
    """The SAME full id twice maps to one short id without raising (a
    collision is two DISTINCT full ids)."""
    items = [(f"{SHA}.greedy", "q", "x"), (f"{SHA}.greedy", "q", "x")]
    short_items, lut = j._shorten_ids(items)
    assert len(short_items) == 2 and len(lut) == 1
