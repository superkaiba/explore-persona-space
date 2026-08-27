"""#2617 SVMP driver — CPU/torch-free/no-network invariants.

Covers the seams a smoke on the pod cannot cheaply re-check and a wrong shape
would fire in production AFTER the expensive phases:

- rule-27 parse-contract round-trip: realistic judge replies through the
  harness's OWN parse path (``parse_judge_json`` -> ``_score_from_parsed``);
- the {question}/{answer} substitution leaves no unfilled slot;
- the judge-id alias map is bijective + grammar-valid, and the composed
  Anthropic Batch custom_id survives ``validate_batch_custom_ids`` — while a
  RAW ``::``-bearing context_id would NOT (the aliasing is load-bearing);
- the constructed grid one-word gate + per-class counts;
- the XSTest orientation constant (v1=safe / v2=unsafe).

Adoptable: imports the driver by its module path (torch-free module top — the
langow pinned-blob import is deferred), uses only committed data, hits no HF.
"""

from __future__ import annotations

import pytest

from explore_persona_space.eval.graded_judge import _score_from_parsed
from explore_persona_space.eval.judge_dispatch import validate_batch_custom_ids
from explore_persona_space.eval.utils import parse_judge_json
from scripts.issue2617_svmp_run import (
    _EXPECTED_CLASS_COUNTS,
    CELL,
    CONSTRUCTED_CLASSES,
    CONSTRUCTED_PAIRS,
    EXPECTED_REFUSAL_VALUES,
    JUDGE_RUBRIC,
    _item_id,
    build_judge_id_map,
    constructed_authoring_gate,
    one_word_diff,
)


def _score(text: str) -> float | None:
    return _score_from_parsed(parse_judge_json(text))


@pytest.mark.parametrize(
    "reply,expected",
    [
        ('{"score": 85}', 85.0),  # canonical
        ('```json\n{"score": 30}\n```', 30.0),  # fenced markdown
        ('The response fully complies.\n{"score": 5}', 5.0),  # reason-then-score
        ("85", 85.0),  # bare integer (off-spec but valid, #778)
        ("Score: 0", 0.0),  # trailing labelled score (#2225)
        ("0", 0.0),  # floor
        ("100", 100.0),  # ceiling
    ],
)
def test_rubric_roundtrip_valid(reply: str, expected: float) -> None:
    assert _score(reply) == expected


@pytest.mark.parametrize("reply", ["REFUSAL", '{"score": 150}', "I cannot help.", '{"score": -5}'])
def test_rubric_roundtrip_dropped(reply: str) -> None:
    # drop-never-coerce: malformed / REFUSAL / out-of-range -> None (dropped).
    assert _score(reply) is None


def test_rubric_has_slots_and_no_unfilled_after_substitution() -> None:
    assert "{question}" in JUDGE_RUBRIC and "{answer}" in JUDGE_RUBRIC
    filled = JUDGE_RUBRIC.replace("{question}", "Q?").replace("{answer}", "A.")
    assert "{question}" not in filled and "{answer}" not in filled
    # the JSON example brace must survive substitution intact.
    assert '{"score": <0-100>}' in filled


def test_constructed_authoring_gate_and_counts() -> None:
    constructed_authoring_gate()  # raises on any gate failure
    counts = {c: 0 for c in CONSTRUCTED_CLASSES}
    for cls, _cat, _qa, _qb in CONSTRUCTED_PAIRS:
        counts[cls] += 1
    assert counts == _EXPECTED_CLASS_COUNTS
    assert sum(counts.values()) == 72


def test_one_word_diff_every_constructed_pair() -> None:
    for _cls, _cat, qa, qb in CONSTRUCTED_PAIRS:
        pos, wa, wb = one_word_diff(qa, qb)
        assert wa != wb and pos >= 0


def _production_context_ids() -> dict[str, dict]:
    ids: dict[str, dict] = {}
    for cls in CONSTRUCTED_CLASSES:
        for j in range(_EXPECTED_CLASS_COUNTS[cls]):
            for v in ("a", "b"):
                ids[f"{CELL}::{v}::{cls}_{j:02d}"] = {}
    for n in range(1, 37):
        for v in ("v2", "v1"):
            ids[f"{CELL}::{v}::xstest_i{n:02d}"] = {}
    return ids


def test_judge_id_map_bijective_and_grammar_valid() -> None:
    ids = _production_context_ids()
    assert len(ids) == 216
    fwd = build_judge_id_map(ids)
    assert len(fwd) == 216
    assert len(set(fwd.values())) == 216  # bijective
    assert set(fwd.keys()) == set(ids)


def test_composed_custom_ids_survive_batch_grammar() -> None:
    ids = _production_context_ids()
    fwd = build_judge_id_map(ids)
    # 216 contexts x 10 draws -> composed custom_id {item_id}__{idx:05d}__00.
    composed = []
    idx = 0
    for cid in sorted(ids):
        for d in range(10):
            composed.append(f"{_item_id(fwd[cid], d)}__{idx:05d}__00")
            idx += 1
    assert len(composed) == 2160
    validate_batch_custom_ids(composed)  # raises on any grammar violation
    assert all(len(c) <= 64 for c in composed)


def test_raw_context_id_would_violate_grammar_proving_alias_needed() -> None:
    # A raw ::-bearing context_id as the item id would 400 the Batch submit.
    raw = f"{CELL}::a::obj_flip_00--d0__00000__00"
    with pytest.raises(ValueError):
        validate_batch_custom_ids([raw])


def test_xstest_orientation_constant() -> None:
    assert EXPECTED_REFUSAL_VALUES == {"v1": "safe", "v2": "unsafe"}
