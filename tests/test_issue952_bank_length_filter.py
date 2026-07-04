"""Round-3 crash-fix tests (GCE att-20260704-103316: a real bank row's
chat-formatted prompt hit 8377 tokens > vLLM max_model_len 8192 at phase 1a).

Pins the bank prompt-length pair filter: an overlong member drops its ENTIRE
matched (divergent, control) pair, the record is digest-only (indices + token
counts + categories, never text), and every consumer inherits the filter via
``load_bank_queries`` (which also writes ``bank_length_filter.json``).
"""

import json

from explore_persona_space.experiments.issue_952 import run_952 as r


class _StubTokenizer:
    """Whitespace 'tokenizer': one token per word, +2 template tokens."""

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        assert not tokenize and add_generation_prompt
        return "<|u|> " + messages[0]["content"] + " <|a|>"

    def __call__(self, text, return_tensors=None, add_special_tokens=False):
        assert not add_special_tokens
        return {"input_ids": text.split()}


def _row(qid, pid, cat, role, text):
    return {"query_id": qid, "pair_id": pid, "category": cat, "role": role, "text": text}


_OVERLONG = "w " * (r.BANK_PROMPT_TOKEN_BUDGET + 5)  # > budget words even before template


def _rows():
    """3 pairs: p1 kept; p2 divergent overlong; p3 control overlong."""
    return [
        _row("q1d", "p1", "model_identity", "divergent", "short question one"),
        _row("q1c", "p1", "model_identity", "control", "short question two"),
        _row("q2d", "p2", "model_identity", "divergent", _OVERLONG),
        _row("q2c", "p2", "model_identity", "control", "short partner of overlong"),
        _row("q3d", "p3", "style_format", "divergent", "short divergent three"),
        _row("q3c", "p3", "style_format", "control", _OVERLONG),
    ]


def test_filter_drops_matched_pair_and_records_counts():
    """Either overlong member drops the whole pair; digest-only record."""
    kept, record = r.filter_bank_rows_by_length(_rows(), _StubTokenizer())
    assert [row["query_id"] for row in kept] == ["q1d", "q1c"]
    assert record["n_rows_before"] == 6
    assert record["n_rows_after"] == 2
    assert record["n_pairs_dropped"] == 2
    assert record["dropped_pairs_by_category"] == {"model_identity": 1, "style_format": 1}
    assert record["kept_pairs_by_category"] == {"model_identity": 1, "style_format": 0}
    # All 4 rows of the two dropped pairs are recorded — including the SHORT
    # partners (over_budget=False) dropped only for pairing.
    dropped = {d["query_id"]: d for d in record["dropped_rows"]}
    assert set(dropped) == {"q2d", "q2c", "q3d", "q3c"}
    assert dropped["q2d"]["over_budget"] and not dropped["q2c"]["over_budget"]
    assert dropped["q3c"]["over_budget"] and not dropped["q3d"]["over_budget"]
    assert dropped["q2d"]["prompt_tokens"] > r.BANK_PROMPT_TOKEN_BUDGET
    assert dropped["q2d"]["index"] == 2  # bank-file index, digest-only reference
    # Digest-only: no text field anywhere in the record.
    assert all("text" not in d for d in record["dropped_rows"])


def test_filter_noop_when_all_within_budget():
    rows = _rows()[:2]
    kept, record = r.filter_bank_rows_by_length(rows, _StubTokenizer())
    assert kept == rows
    assert record["n_pairs_dropped"] == 0
    assert record["dropped_rows"] == []


def test_load_bank_queries_inherits_filter_and_writes_artifact(tmp_path, monkeypatch):
    """Every consumer goes through load_bank_queries -> the filter + artifact."""
    monkeypatch.setattr(r, "_get_tokenizer", lambda: _StubTokenizer())
    bank = tmp_path / "bank.json"
    bank.write_text(json.dumps({"issue": 952, "n_pairs": 3, "queries": _rows()}))
    rows, meta = r.load_bank_queries(tmp_path, smoke=False, bank_file=str(bank))
    assert {row["pair_id"] for row in rows} == {"p1"}
    assert meta["length_filter"]["n_pairs_dropped"] == 2
    artifact = tmp_path / "eval_results" / "issue_952" / "bank_length_filter.json"
    assert artifact.exists()
    rec = json.loads(artifact.read_text())
    assert rec["n_pairs_dropped"] == 2
    assert rec["budget_tokens"] == r.BANK_PROMPT_TOKEN_BUDGET


def test_smoke_subset_applies_after_filter(tmp_path, monkeypatch):
    """Smoke subsetting picks its first-per-category pairs from FILTERED rows,
    so an overlong first pair can never re-enter via the smoke path."""
    monkeypatch.setattr(r, "_get_tokenizer", lambda: _StubTokenizer())
    rows = [
        # p0 is the FIRST model_identity pair by sort order but is overlong.
        _row("q0d", "p0", "model_identity", "divergent", _OVERLONG),
        _row("q0c", "p0", "model_identity", "control", "short"),
        *_rows(),
    ]
    bank = tmp_path / "bank.json"
    bank.write_text(json.dumps({"issue": 952, "queries": rows}))
    smoke_rows, _ = r.load_bank_queries(tmp_path, smoke=True, bank_file=str(bank))
    assert all(row["pair_id"] != "p0" for row in smoke_rows)
    # p0/p2/p3 all carry an overlong member -> only p1 survives into the subset.
    assert {row["pair_id"] for row in smoke_rows} == {"p1"}
