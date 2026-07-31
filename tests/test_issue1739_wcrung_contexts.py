"""Pins for the #1739 held-out-battery ("wcrung") context reconstruction.

Two invariants worth a permanent test:

1. **The render-parity gate actually trips.** The gate compares each
   reconstructed prompt's token count against the row's own stored
   ``n_tokens_instruct`` (written by the #1092 capture run), which is what makes
   it proof that the reconstruction reproduces the captured render. A gate that
   silently passed on a mismatch would let a drifted render ship, and every
   downstream arm read would be scored against prompts the free summaries were
   NOT captured under.

2. **Family scoping is honest.** ``generic_chat_only`` must be True only when the
   selection contains no eliciting family. The eval-only battery's ``behavior``
   family directly instructs the behaviors #1739 measures, so folding it into a
   "generic chat traffic lacks these behaviors" read would invert the finding.

Both run on synthetic fixtures — no Hub, no tokenizer download, no GPU.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

wcrung = pytest.importorskip("scripts.issue1739_wcrung_contexts")


class _FakeTokenizer:
    """Minimal chat-template stand-in: renders turns, tokenizes on whitespace.

    Deliberately signature-compatible with the two calls
    :func:`render_row_prompt` makes on a real tokenizer.
    """

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        assert tokenize is False, "reconstruction renders text, never token ids"
        parts = [f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages]
        if add_generation_prompt:
            parts.append("<|im_start|>assistant\n")
        return "".join(parts)

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": text.split()}


def _write_corpus(tmp_path: Path, *, n_tokens_instruct_override: dict[str, int] | None = None):
    """Two battery contexts x two queries, in the published artifact shapes."""
    tok = _FakeTokenizer()
    # (prefix_id, family, turns) — one real-conversation prefix, one system prefix.
    prefixes = {
        "batt_f2_wc_short_1": [
            {"role": "user", "content": "aa bb"},
            {"role": "assistant", "content": "cc dd"},
        ],
        "batt_f8_behav_sycophant": [{"role": "system", "content": "ee ff"}],
    }
    queries = {"q0": "gg hh", "q1": "ii jj"}

    manifest_rows = []
    for pid, turns in prefixes.items():
        for qid, qtext in queries.items():
            _, prompt = wcrung.render_row_prompt(tok, turns, qtext)
            n_tok = len(tok(prompt)["input_ids"])
            row_id = f"r_{pid}_{qid}"
            if n_tokens_instruct_override and row_id in n_tokens_instruct_override:
                n_tok = n_tokens_instruct_override[row_id]
            manifest_rows.append(
                {
                    "row_id": row_id,
                    "stratum": "battery",
                    "is_eval_only": True,
                    "prefix_id": pid,
                    "query_id": qid,
                    "n_tokens_instruct": n_tok,
                }
            )

    corpus = tmp_path / "corpus"
    corpus.mkdir(parents=True, exist_ok=True)
    (corpus / "manifest.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in manifest_rows), encoding="utf-8"
    )
    (corpus / "prefix_store.jsonl").write_text(
        "".join(
            json.dumps({"prefix_id": pid, "prefix_turns": turns}) + "\n"
            for pid, turns in prefixes.items()
        ),
        encoding="utf-8",
    )
    (corpus / "query_store.jsonl").write_text(
        "".join(json.dumps({"query_id": qid, "query": q}) + "\n" for qid, q in queries.items()),
        encoding="utf-8",
    )
    return {
        "manifest.jsonl": corpus / "manifest.jsonl",
        "prefix_store.jsonl": corpus / "prefix_store.jsonl",
        "query_store.jsonl": corpus / "query_store.jsonl",
    }, tok


@pytest.fixture
def _patched_split(monkeypatch):
    """Point the pinned-count asserts + battery family map at the fixture."""
    monkeypatch.setattr(
        wcrung,
        "battery_family_by_prefix_id",
        lambda: {
            "batt_f2_wc_short_1": "wildchat",
            "batt_f8_behav_sycophant": "behavior",
        },
    )
    import explore_persona_space.experiments.issue_1739.constants as consts

    monkeypatch.setattr(consts, "STORE_TOTAL_ROWS", 4, raising=True)
    monkeypatch.setattr(consts, "STORE_FIT_ROWS", 0, raising=True)


def test_parity_gate_trips_on_a_drifted_render(tmp_path, _patched_split):
    """A stored n_tokens_instruct that disagrees with the render MUST fail loud.

    This is the fails-pre-fix half: with the gate removed, the drifted row would
    be silently reconstructed and scored against mismatched summaries.
    """
    corpus, tok = _write_corpus(
        tmp_path, n_tokens_instruct_override={"r_batt_f2_wc_short_1_q0": 999}
    )
    with pytest.raises(ValueError, match="render parity gate FAILED"):
        wcrung.reconstruct_contexts(
            corpus=corpus, families=None, max_rows=None, tokenizer=tok, parity_gate=True
        )


def test_parity_gate_passes_on_a_faithful_render(tmp_path, _patched_split):
    corpus, tok = _write_corpus(tmp_path)
    rows, digest = wcrung.reconstruct_contexts(
        corpus=corpus, families=None, max_rows=None, tokenizer=tok, parity_gate=True
    )
    assert digest["parity_gate"] == "PASS"
    assert digest["n_parity_mismatch"] == 0
    assert len(rows) == 4
    # store_row_index is the summary-slice key: unique + manifest-ordered.
    assert [r["store_row_index"] for r in rows] == [0, 1, 2, 3]


def test_generic_chat_only_flag_is_false_when_an_eliciting_family_is_present(
    tmp_path, _patched_split
):
    corpus, tok = _write_corpus(tmp_path)
    _, digest = wcrung.reconstruct_contexts(
        corpus=corpus, families=None, max_rows=None, tokenizer=tok, parity_gate=True
    )
    assert digest["generic_chat_only"] is False
    assert "behavior" in digest["eliciting_families_present"]


def test_generic_chat_only_flag_is_true_for_the_scoped_selection(tmp_path, _patched_split):
    corpus, tok = _write_corpus(tmp_path)
    rows, digest = wcrung.reconstruct_contexts(
        corpus=corpus, families={"wildchat", "default"}, max_rows=None, tokenizer=tok
    )
    assert digest["generic_chat_only"] is True
    assert digest["eliciting_families_present"] == []
    assert {r["family"] for r in rows} == {"wildchat"}


def test_multi_turn_prefix_renders_as_conversation_turns(tmp_path):
    """A real-conversation prefix must render as TURNS, not one squashed system message.

    The pvsynth staged-context schema wraps a single ``prefix_text`` in one
    system message; doing that to a WildChat conversation prefix would diverge
    from the render the reused summaries were captured under.
    """
    tok = _FakeTokenizer()
    turns = [
        {"role": "user", "content": "first user"},
        {"role": "assistant", "content": "first reply"},
    ]
    prefix, prompt = wcrung.render_row_prompt(tok, turns, "the query")
    assert "<|im_start|>user\nfirst user" in prefix
    assert "<|im_start|>assistant\nfirst reply" in prefix
    # No system role invented for a conversation prefix.
    assert "<|im_start|>system" not in prefix
    assert prompt.startswith(prefix)
    assert prompt.rstrip().endswith("<|im_start|>assistant")


def test_bare_prefix_derives_prefix_from_the_user_turn_header(tmp_path):
    """Empty turns: the Qwen template cannot render an empty list (#1092 round-8.2)."""
    tok = _FakeTokenizer()
    prefix, prompt = wcrung.render_row_prompt(tok, [], "solo query")
    assert prompt.startswith(prefix)
    assert "solo query" not in prefix
