import pytest

from scripts.issue1739_annotation_provenance import verify


def test_catches_default_generator_even_when_json_is_schema_valid(tmp_path):
    source = tmp_path / "source.py"
    source.write_text("defaults = {'aaaaaaaaaaaaaaaa': (75, 'A specific reason.')}\n")
    rows = [{"id": "bbbbbbbbbbbbbbbb", "rationale": "Generic negative.", "score": 0}]
    with pytest.raises(ValueError, match="No explicitly authored"):
        verify(rows, [source])


def test_literal_triples_must_match_the_correct_id_and_score(tmp_path):
    source = tmp_path / "source.py"
    source.write_text("rows = [('aaaaaaaaaaaaaaaa', 75, 'A specific reason.')]\n")
    rows = [{"id": "aaaaaaaaaaaaaaaa", "rationale": "A specific reason.", "score": 75}]
    assert verify(rows, [source]) == 1
    rows[0]["score"] = 0
    with pytest.raises(ValueError):
        verify(rows, [source])


def test_full_dictionary_null_judgment_is_explicit(tmp_path):
    source = tmp_path / "source.py"
    source.write_text(
        "rows = [{'id':'aaaaaaaaaaaaaaaa',"
        "'rationale':'Cannot verify the named company history.','score':None}]\n"
    )
    rows = [
        {
            "id": "aaaaaaaaaaaaaaaa",
            "rationale": "Cannot verify the named company history.",
            "score": None,
        }
    ]
    assert verify(rows, [source]) == 1
