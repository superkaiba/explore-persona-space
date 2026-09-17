import pytest

from scripts.issue1739_luna_judging import jsonl, validate_rows


def verdict(**kwargs):
    return dict(
        id="x",
        rationale="The response refuses harmful assistance.",
        status="scored",
        score=0,
        positive=False,
        evidence_type="transcript",
        sources=[],
        **kwargs,
    )


def test_recorded_refusal_is_valid_zero_and_missing_judgment_is_not():
    row = verdict()
    validate_rows([{"id": "x"}], [row], "harmful_compliance")
    row["status"] = "judge_refusal"
    with pytest.raises(ValueError):
        validate_rows([{"id": "x"}], [row], "harmful_compliance")
    row.update(score=None, positive=None)
    validate_rows([{"id": "x"}], [row], "harmful_compliance")


def test_extra_and_duplicate_judgments_rejected():
    with pytest.raises(ValueError):
        validate_rows([{"id": "x"}], [verdict(), verdict()], "sycophancy")


def test_reference_positive_needs_real_source_field():
    row = verdict()
    row.update(score=75, positive=True, evidence_type="reference")
    with pytest.raises(ValueError):
        validate_rows([{"id": "x"}], [row], "hallucination")


def test_jsonl_preserves_unicode_line_separators(tmp_path):
    p = tmp_path / "cases.jsonl"
    p.write_text('{"id":"x","response":"before\u2028after"}\n')
    assert jsonl(p) == [{"id": "x", "response": "before\u2028after"}]
