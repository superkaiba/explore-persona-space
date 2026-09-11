from __future__ import annotations

import ast
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]
import issue931_build_pairs as PAIRS  # noqa: E402
import issue1901_boundary_25k as B  # noqa: E402
from issue779_ffc_n1m_generate_capture import NearDupeGate  # noqa: E402


@pytest.fixture(scope="module")
def tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(B.C.MODEL_ID, local_files_only=True)


def test_title_exclusion_handles_moses_spacing_and_unicode():
    assert B.normalized_title("King 's Cross — Station") == B.normalized_title(
        "King's Cross—Station"
    )
    assert B.normalized_title("Ｃａｆé") == B.normalized_title("Café")


def test_published_number_and_punctuation_format():
    from sacremoses import MosesTokenizer

    got = B.wiki_format(
        "It was 1,200.5 in a long-term study. Why?\nYes!",
        MosesTokenizer(lang="en"),
    )
    assert B.wiki_format("", MosesTokenizer(lang="en")) == " \n"
    assert got == " It was 1 @,@ 200 @.@ 5 in a long @-@ term study . Why ? \n Yes ! \n"


def test_extracted_eligibility_preserves_original_builder(monkeypatch, tokenizer):
    # Execute the pinned original function with a bounded in-memory corpus;
    # catch changes to anchor/next-span/previous-span semantics, not just counts.
    source = subprocess.check_output(
        ["git", "show", "94b8034cbfe9b5e5def05b8ec0cdec5352915267:scripts/issue931_build_pairs.py"],
        cwd=ROOT,
        text=True,
    )
    tree = ast.parse(source)
    function = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "build_armc_pairs"
    )
    old_globals = dict(vars(PAIRS))
    text = " ".join(
        f"The researcher wrote a detailed report about specimen number {n} "
        f"and several observations {'.?!'[n % 3]}"
        for n in range(80)
    )
    articles = [("Example", text), ("Second", text.replace("researcher", "historian"))]
    old_globals["iter_wikitext_articles"] = lambda _limit: iter(articles)
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), "original_builder", "exec"),
        old_globals,
    )
    monkeypatch.setattr(PAIRS, "iter_wikitext_articles", lambda _limit: iter(articles))
    kwargs = dict(n_articles=None, max_anchors=48, article_cap_tokens=4096, record_sep_char=True)
    old = old_globals["build_armc_pairs"](tokenizer, **kwargs)
    new = PAIRS.build_armc_pairs(tokenizer, **kwargs)
    assert old["articles"] == new["articles"]
    assert [p.to_dict() for p in old["pairs"]] == [p.to_dict() for p in new["pairs"]]
    assert len(new["pairs"]) > 0


def test_worker_cap_frozen_eval_gate_and_token_identity(monkeypatch, tokenizer):
    from sacremoses import MosesTokenizer

    monkeypatch.setattr(B, "TOKENIZER", tokenizer, raising=False)
    monkeypatch.setattr(B, "MOSES", MosesTokenizer(lang="en"), raising=False)
    monkeypatch.setattr(B, "EXCLUDED", set(), raising=False)
    monkeypatch.setattr(B, "GATE", NearDupeGate([]), raising=False)
    article = {
        "id": "123",
        "title": "Fresh",
        "url": "https://en.wikipedia.org/wiki/Fresh",
        "text": " ".join(
            f"This is a sufficiently long sentence about the historical specimen number {n}, "
            f"with multiple details{'.?!'[n % 3]}"
            for n in range(100)
        ),
    }
    result = B.process_article((article, list(B.TOKENS)))
    assert "drop" not in result
    assert max(Counter(r["boundary_token_id"] for r in result["rows"]).values()) <= 6
    for row in result["rows"]:
        assert result["input_ids"][row["anchor_pos"]] == row["boundary_token_id"]
        assert row["anchor_pos"] + 1 == row["t_span"][0]
        assert 8 <= row["n_span_tokens"] <= 256
    # Every eligible target is now excluded; no target can be silently replaced
    # by a mutated span when passing the same input through the gate.
    ids, offsets = B.C.tokenize_with_offsets(tokenizer, result["processed_text"])
    ids, offsets = ids[:4096], offsets[:4096]
    targets = [
        tokenizer.decode(ids[lo:hi])
        for _, lo, hi, *_ in PAIRS.armc_eligible_anchors(
            ids, offsets, result["processed_text"][: int(offsets[-1, 1])]
        )
    ]
    monkeypatch.setattr(B, "GATE", NearDupeGate(targets))
    assert B.process_article((article, list(B.TOKENS)))["drop"] == "no_eligible_rows"
    monkeypatch.setattr(B, "EXCLUDED", {B.normalized_title(article["title"])})
    assert B.process_article((article, list(B.TOKENS)))["drop"] == "old_title"


def test_jsonl_preserves_unicode_line_separators(tmp_path):
    values = [{"text": "first\u2028second\u0085third\nlast"}]
    names = B.write_jsonl_parts(tmp_path, "sample", values)
    assert B.read_jsonl(tmp_path / names[0]) == values
