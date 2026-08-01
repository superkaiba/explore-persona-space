"""#1947 crash-fix r7 pins: adaptive marker question top-up.

The P0 crash (3rd): the icl context's two-shot demo block inflates rendered
row lengths, so the 2048-token budget gate dropped 116/1,344 questions (8.6%)
— beating MARKER_Q_N's fixed 5% headroom — and ``_build_marker_mix`` raised
``only 1228 budget-surviving positives < 1280``. The fix makes the marker
question pool ADAPTIVE: on a shortfall, harvest additional WildChat questions
(same disjointness screens, bounded tranches, 2x hard cap), extend the
r-map caches for the DELTA only (cached R reused — never regenerated), and
rebuild; the exact-count invariant (1,280 pos / 5,120 neg) is unchanged.

Tests execute the REAL ``_build_marker_mix`` / ``_build_marker_mix_with_topup``
/ ``_topup_marker_questions`` / ``_extend_r_map`` production bodies at the
PRODUCTION arithmetic (1,344-question pool, 116 over-budget — the incident
shape); fakes sit only at the external boundaries (tokenizer = the model
boundary, backend = the vLLM boundary, ``datasets.load_dataset`` = the
network boundary) and are signature-conformant by construction. Bank/corpus
questions are referenced programmatically, never printed (digest-only bank
discipline).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import issue1090_questiongen as qg  # noqa: E402
import issue1947_cells as cells  # noqa: E402
import issue1947_datagen as dg  # noqa: E402

MARKER_ID = 83399


class _FakeTok:
    """Signature-conformant tokenizer fake (external model boundary only):
    ``apply_chat_template(tokenize=True|False)`` + ``encode``, word-grain ids,
    marker-aware (a whitespace-separated chunk carrying the glyph encodes id
    83399) so the REAL ``assert_positive_tails_encode_marker`` body runs."""

    def _ids(self, text: str) -> list[int]:
        return [MARKER_ID if "※" in w else (hash(w) % 50_000) + 1 for w in text.split()]

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        text = "\n".join(str(m.get("content", "")) for m in messages)
        if add_generation_prompt:
            text += "\nassistant:"
        return self._ids(text) if tokenize else text

    def encode(self, text, add_special_tokens=False):
        return self._ids(text)


class _RecordingBackend:
    """vLLM-boundary fake mirroring ``_MockBackend``'s generate/close surface;
    records every prompt batch so delta-only generation is assertable."""

    def __init__(self):
        self.calls: list[list[str]] = []

    def generate(self, prompts: list[str], max_new: int, *, adapter_dir=None) -> list[str]:
        self.calls.append(list(prompts))
        return [f"Fresh greedy delta answer {i}." for i in range(len(prompts))]

    def close(self, label: str) -> None:
        return


def _cfg(tmp_path: Path, *, smoke: bool = False, mock_gen: bool = True) -> dg.Cfg:
    return dg.Cfg(
        out_root=tmp_path,
        phases=(),
        behaviors=(),
        contexts=("pers",),
        smoke=smoke,
        mock_gen=mock_gen,
    )


def _seed_marker_questions(cfg: dg.Cfg, pool: list[str]) -> None:
    dg._write_json(
        cfg.generic_dir / "marker_questions.json",
        {
            "questions": pool,
            "n": len(pool),
            "sha256_canonical": dg._canonical_sha(pool),
            "source": "test seed",
            "mock": True,
        },
    )


def _seed_r_caches(cfg: dg.Cfg, pool: list[str], over_budget_idx: set[int]) -> list[str]:
    """Pre-seed the per-label greedy-R caches (the crashed run's artifacts):
    over-budget questions carry a >2048-token R so the REAL budget gate
    pair-drops them. Returns the seeded label list."""
    import issue1481_marker as mk

    panel = mk.training_negative_panel("pers")
    labels = ["src_pers"] + [f"neg_{m.slug}" for m in panel]

    def _r(i: int) -> str:
        return ("long " * 2060).strip() if i in over_budget_idx else f"A short fine answer {i}."

    for label in labels:
        dg._write_json(
            cfg.marker_dir / "mixes" / f"r_{label}.json",
            {
                "label": label,
                "truncation_rate": 0.0,
                "r_by_q": {q: _r(i) for i, q in enumerate(pool)},
            },
        )
    return labels


def test_shortfall_message_shape():
    e = dg.MarkerBudgetShortfall("icl", 1228, 1280)
    assert (
        str(e)
        == "marker icl: only 1228 budget-surviving positives < 1280 — raise MARKER_Q_N headroom"
    )
    assert isinstance(e, RuntimeError)
    assert (e.ctx_key, e.n_kept, e.n_pos) == ("icl", 1228, 1280)


def test_topup_fires_delta_only_and_exact_counts(tmp_path):
    """The incident shape at production arithmetic: 116/1,344 budget drops →
    1,228 < 1,280 shortfall → ONE top-up tranche (104 = 2x deficit) → exact
    1,280/5,120/6,400 counts, with ONLY the delta questions freshly generated
    (cached R for the original pool reused byte-identical)."""
    import issue1481_marker as mk

    cfg = _cfg(tmp_path)
    pool = [f"Seed question number {i:05d} about a benign everyday topic?" for i in range(1344)]
    over = set(range(116))  # 8.6% — the observed icl drop rate (<= the 10% gate)
    _seed_marker_questions(cfg, pool)
    _seed_r_caches(cfg, pool, over)
    shim = dg._marker_shim_cfg(cfg)
    tok = _FakeTok()

    # (a) the direct build raises the typed shortfall with the pre-fix message
    caches = cfg.marker_dir / "mixes"
    r_src = json.loads((caches / "r_src_pers.json").read_text(encoding="utf-8"))["r_by_q"]
    panel = mk.training_negative_panel("pers")
    r_neg = {
        m.slug: json.loads((caches / f"r_neg_{m.slug}.json").read_text(encoding="utf-8"))["r_by_q"]
        for m in panel
    }
    with pytest.raises(dg.MarkerBudgetShortfall) as ei:
        dg._build_marker_mix(cfg, tok, "pers", r_src, r_neg)
    assert (
        str(ei.value)
        == "marker pers: only 1228 budget-surviving positives < 1280 — raise MARKER_Q_N headroom"
    )

    # (b) the top-up wrapper converges with exact counts
    backend = _RecordingBackend()
    meta = dg._build_marker_mix_with_topup(
        cfg,
        shim,
        tok,
        backend,
        "pers",
        mk.source_context(shim, "pers").messages,
        {m.slug: m.messages for m in panel},
    )
    assert meta["n_positive"] == dg.MARKER_POS == 1280
    assert meta["n_negative"] == dg.MARKER_NEG == 5120
    assert meta["n_total"] == 6400
    assert len(meta["selected_question_shas"]) == 1280
    slug = cells.marker_slug("pers")
    mix_rows = dg._read_jsonl(cfg.mixes_dir / slug / "train_mix.jsonl")
    assert len(mix_rows) == 6400
    n_pos_rows = sum(1 for r in mix_rows if r["completion"][0]["content"].endswith(" ※"))
    assert n_pos_rows == 1280

    # (c) delta-only generation: 104 fresh questions x 6 labels, none of the
    # original (cached) pool regenerated
    assert len(backend.calls) == 1 + len(panel)
    assert all(len(call) == 104 for call in backend.calls)
    for call in backend.calls:
        for prompt in call:
            assert "Mock WildChat question" in prompt
            assert "Seed question" not in prompt

    # (d) the realized pool + audit trail are recorded
    rec = json.loads((cfg.generic_dir / "marker_questions.json").read_text(encoding="utf-8"))
    assert rec["n"] == 1344 + 104 == len(rec["questions"])
    assert rec["sha256_canonical"] == dg._canonical_sha(rec["questions"])
    assert rec["topups"] == [
        {
            "tranche": 1,
            "ctx_key": "pers",
            "n_added": 104,
            "added_sha256_canonical": dg._canonical_sha(rec["questions"][1344:]),
        }
    ]
    r_src_after = json.loads((caches / "r_src_pers.json").read_text(encoding="utf-8"))
    assert len(r_src_after["r_by_q"]) == 1448 and len(r_src_after["topups"]) == 1

    # (e) per-context survivor set recorded on marker_questions.json
    dg._record_marker_ctx_selection(cfg, "pers", meta)
    rec = json.loads((cfg.generic_dir / "marker_questions.json").read_text(encoding="utf-8"))
    sel = rec["ctx_selected"]["pers"]
    assert sel["n_positive"] == 1280
    assert sel["questions_sha256"] == meta["questions_sha256"]
    assert sel["selected_question_shas"] == meta["selected_question_shas"]


def test_hard_cap_exhaustion_raises_with_message_shape(tmp_path):
    cfg = _cfg(tmp_path, smoke=True)  # smoke hard cap = 24
    pool = [f"Capped pool question {i:02d} on a mundane subject?" for i in range(24)]
    _seed_marker_questions(cfg, pool)
    shortfall = dg.MarkerBudgetShortfall("icl", 3, 5)
    with pytest.raises(RuntimeError) as ei:
        dg._topup_marker_questions(cfg, shortfall, 1)
    assert type(ei.value) is RuntimeError  # NOT the retryable subclass — the loop must not catch it
    msg = str(ei.value)
    assert "marker icl: only 3 budget-surviving positives < 5" in msg
    assert "hard cap 24 exhausted at 24" in msg
    # pool untouched on the failure path
    rec = json.loads((cfg.generic_dir / "marker_questions.json").read_text(encoding="utf-8"))
    assert rec["n"] == 24 and "topups" not in rec


def test_topup_zero_yield_raises(tmp_path):
    """A tranche whose every candidate collides with the pool fails loud
    (bounded-loop guard) instead of looping."""
    cfg = _cfg(tmp_path, smoke=True)  # smoke hard cap = 24; mock harvest
    pool = [f"Mock WildChat question {51_000 + i:05d}?" for i in range(20)]
    _seed_marker_questions(cfg, pool)
    # want = min(max(2*2, 64), 24-20) = 4 -> all 4 mock candidates collide
    with pytest.raises(RuntimeError, match="yielded 0 new questions"):
        dg._topup_marker_questions(cfg, dg.MarkerBudgetShortfall("icl", 3, 5), 1)


def test_topup_harvest_applies_disjointness_screens(tmp_path, monkeypatch):
    """Real ``_stream_wildchat_questions`` body (network boundary faked):
    rows duplicating the current pool, the generic corpus, the generic ext
    questions, and the committed #1481 marker train bank are all screened;
    only genuinely fresh questions are admitted, under a tranche-scoped
    checkpoint file."""
    import datasets

    cfg = _cfg(tmp_path, smoke=True, mock_gen=False)
    pool = ["Existing marker pool question about gardening tools?"]
    _seed_marker_questions(cfg, pool)
    corpus_q = "Corpus question about turtle migration patterns?"
    dg._write_jsonl(
        cfg.generic_dir / "generic_corpus.jsonl",
        [{"prompt": [{"role": "user", "content": corpus_q}]}],
    )
    ext_q = "Generic extension question about sourdough starters?"
    dg._write_jsonl(
        cfg.generic_dir / "raw_generic_ext.jsonl",
        [{"member": "default", "question": ext_q, "text": "stub"}],
    )
    bank_q = json.loads(
        (qg.BANKS_DIR / "issue1481_marker_train10_v1.json").read_text(encoding="utf-8")
    )[0]

    def _row(q: str) -> dict:
        return {
            "language": "English",
            "redacted": False,
            "toxic": False,
            "conversation": [{"role": "user", "content": q}],
        }

    fresh_qs = [f"A genuinely fresh WildChat-style question number {i:02d}?" for i in range(30)]
    rows = [_row(pool[0]), _row(corpus_q), _row(ext_q), _row(bank_q)] + [_row(q) for q in fresh_qs]

    class _FakeStream:
        def __init__(self, rows):
            self._rows = rows

        def skip(self, n):
            return _FakeStream(self._rows[n:])

        def __iter__(self):
            return iter(self._rows)

    monkeypatch.setattr(datasets, "load_dataset", lambda *a, **k: _FakeStream(rows))
    fresh = dg._topup_marker_questions(cfg, dg.MarkerBudgetShortfall("icl", 4, 5), 1)
    # deficit 1 -> want = min(max(2, 64), 24 - 1) = 23 fresh questions
    assert len(fresh) == 23
    assert set(fresh) <= set(fresh_qs)
    for banned in (pool[0], corpus_q, ext_q, bank_q):
        assert banned not in fresh
    rec = json.loads((cfg.generic_dir / "marker_questions.json").read_text(encoding="utf-8"))
    assert rec["questions"] == pool + fresh and rec["n"] == 24
    assert (cfg.generic_dir / "wildchat_topup_ckpt_t1.json").exists()
    assert not (cfg.generic_dir / "wildchat_stream_ckpt.json").exists()  # generic ckpt untouched


def test_ctx_selection_backfill_without_shas(tmp_path):
    """Resume-skipped pre-fix mixes backfill identity via questions_sha256
    (their metas carry no per-question sha list)."""
    cfg = _cfg(tmp_path)
    _seed_marker_questions(cfg, ["A single seed question for the backfill test?"])
    meta = {"n_positive": 1280, "questions_sha256": "cafe" * 16}
    dg._record_marker_ctx_selection(cfg, "conv", meta)
    rec = json.loads((cfg.generic_dir / "marker_questions.json").read_text(encoding="utf-8"))
    assert rec["ctx_selected"]["conv"] == {
        "n_positive": 1280,
        "questions_sha256": "cafe" * 16,
        "selected_question_shas": None,
    }
