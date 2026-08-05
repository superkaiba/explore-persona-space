"""Pins for the #2054 Phase A production path (`scripts/issue2054_phase_a.py`).

Round-2 finding C8 (BLOCKER `phase-a-production-generation-deferred`):

C8(i) — the shortfall GENERATION leg must actually be invoked on the
production path: `_generate_shortfall` runs the REAL parent generator
subprocess (`issue1345_gen_scaffolds.py`), faked only at the GPU boundary via
the generator's own deterministic `--mock` path, and merges rows carrying
`question` (verbatim, plan req 6) + `conv_id == qid`.

C8(ii) — the judge leg is a PER-ROW admission gate: `_admit_variant_rows`
keeps rows at/above the threshold, content-drops (never coerces) malformed
returns, and FAILS LOUD on transport-lost rows (llm-judging rules 9/24).

C8(iii) — the rubric conforms to the harness's forced score-parse contract
(llm-judging rule 27): realistic replies (reasoning + score, plus a fenced /
preamble variant) round-trip through the harness's OWN parse+reduce path
(`parse_judge_json` -> `_score_from_parsed`), and the `{question}`/`{answer}`
placeholders substitute completely under the harness-identical `.replace`.
The superseded two-field `{"diverse": ..., "single_question": ...}` reply
shape is pinned as a DROP (score None) — the shape that made the pilot FAIL
with 100% content drops.

All fixtures are synthetic prose written for this test — no real-corpus text.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_REPO_ROOT / "scripts"), str(_REPO_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import issue1345_scaffold_common as sc  # noqa: E402
import issue2054_phase_a as phase_a  # noqa: E402

from explore_persona_space.eval.graded_judge import JudgeResult, _score_from_parsed  # noqa: E402
from explore_persona_space.eval.utils import parse_judge_json  # noqa: E402


# ---------------------------------------------------------------------------
# C8(iii) — rule-27 parse-contract round-trip
# ---------------------------------------------------------------------------
def test_rubric_reason_then_score_round_trips_harness_parse():
    """A realistic reason-then-score reply keeps its score through the
    harness's own parse+reduce path."""
    reply = (
        '{"reasoning": "Vivid dockside scene; the question appears verbatim '
        'as the only question.", "score": 78}'
    )
    assert _score_from_parsed(parse_judge_json(reply)) == 78.0


def test_rubric_fenced_markdown_reply_round_trips():
    reply = (
        "The scaffold is specific and the question is embedded verbatim.\n"
        '```json\n{"reasoning": "specific scene, verbatim question", "score": 91}\n```'
    )
    assert _score_from_parsed(parse_judge_json(reply)) == 91.0


def test_rubric_reasoning_preamble_reply_round_trips():
    reply = (
        "Reasoning: the scene is thin but real and the question is verbatim.\n"
        '{"reasoning": "thin but real", "score": 55}'
    )
    assert _score_from_parsed(parse_judge_json(reply)) == 55.0


def test_old_two_field_reply_shape_is_a_drop():
    """The superseded two-field rubric reply has NO 'score' key -> the
    harness reduce drops it (the C8(iii) 100%-drop pilot FAIL shape)."""
    reply = '{"diverse": 80, "single_question": 90}'
    assert _score_from_parsed(parse_judge_json(reply)) is None


def test_rubric_placeholders_present_and_substitution_complete():
    rubric = phase_a._scaffold_judge_rubric()
    assert "{question}" in rubric and "{answer}" in rubric
    # The rubric names the actual slot sentinel (never a hand-typed copy).
    assert sc.SLOT_SENTINEL in rubric
    # Harness-identical substitution (graded_judge.format_user_msg .replace):
    filled = rubric.replace("{question}", "What time is it?").replace(
        "{answer}", f"A scene. {sc.SLOT_SENTINEL} More scene."
    )
    assert "{question}" not in filled and "{answer}" not in filled
    # Literal JSON braces of the instructed reply shape survive substitution.
    assert '"score"' in filled and '"reasoning"' in filled


# ---------------------------------------------------------------------------
# C8(ii) — per-row admission reduce
# ---------------------------------------------------------------------------
def _mk_result(scores: dict, transport: dict | None = None) -> JudgeResult:
    return JudgeResult(
        scores=scores,
        n_total_draws=len(scores),
        n_dropped_draws=sum(1 for v in scores.values() if v is None),
        per_item_transport_losses=transport or {},
    )


def _rows(n: int) -> tuple[list[dict], list[tuple[str, str, str]]]:
    rows = [
        {
            "scaffold_id": f"sc{i}",
            "conv_id": f"mt_{i:012x}",
            "question": f"Question {i}?",
            "scaffold_text": f"Scene {i}. Question {i}? {sc.SLOT_SENTINEL} Tail.",
        }
        for i in range(n)
    ]
    items = [
        (f"char_helios-{i:06d}", rows[i]["question"], rows[i]["scaffold_text"]) for i in range(n)
    ]
    return rows, items


def test_admission_keeps_at_threshold_drops_below_and_content_drops():
    rows, items = _rows(4)
    result = _mk_result(
        {
            "char_helios-000000": 78.0,  # keep
            "char_helios-000001": 50.0,  # keep (>= threshold)
            "char_helios-000002": 12.0,  # below threshold
            "char_helios-000003": None,  # content drop (malformed/refusal)
        }
    )
    admitted, drops = phase_a._admit_variant_rows(rows, items, result, 50.0)
    assert [r["scaffold_id"] for r in admitted] == ["sc0", "sc1"]
    assert all(r["judge_score"] >= 50.0 for r in admitted)
    assert drops == {"below_threshold": 1, "judge_content_drop": 1}


def test_admission_fails_loud_on_transport_lost_rows():
    """Rule 24: a row whose draws were ALL transport-lost is never silently
    dropped — the run raises (re-judgeable; the cache resumes)."""
    rows, items = _rows(2)
    result = _mk_result(
        {"char_helios-000000": 80.0, "char_helios-000001": None},
        transport={"char_helios-000001": 1},
    )
    with pytest.raises(RuntimeError, match="transport"):
        phase_a._admit_variant_rows(rows, items, result, 50.0)


def test_question_of_prefers_field_then_stripper_span():
    assert phase_a._question_of({"question": "Q?", "scaffold_text": "x"}) == "Q?"
    text = f'She asked, "Where now?" {sc.SLOT_SENTINEL} End.'
    row = {"scaffold_text": text, "q_start": 11, "q_end": 22}
    assert phase_a._question_of(row) == text[11:22].strip()
    assert phase_a._question_of({"scaffold_text": "no span"}) is None
    # Invalid span (q_end past text end) -> None, not a crash.
    assert phase_a._question_of({"scaffold_text": "ab", "q_start": 0, "q_end": 99}) is None


def test_judge_items_ids_conform_to_custom_id_grammar():
    rows, _ = _rows(3)
    items, judged, n_no_q = phase_a._variant_judge_items(
        "conversation_paired_stories_assistant", [*rows, {"scaffold_text": "no q"}]
    )
    assert n_no_q == 1
    assert len(items) == len(judged) == 3
    for item_id, _q, _a in items:
        assert "__" not in item_id
        assert len(item_id) <= 53
        assert all(c.isalnum() or c in "_-" for c in item_id)


# ---------------------------------------------------------------------------
# C8(i) — the generation leg is INVOKED (real subprocess, mock GPU boundary)
# ---------------------------------------------------------------------------
def test_generation_leg_invokes_parent_generator_end_to_end(tmp_path):
    """`_generate_shortfall` runs the REAL `issue1345_gen_scaffolds.py`
    subprocess (its own deterministic --mock path — no GPU) and merges rows
    with verbatim questions + conv_id == qid (plan req 6 + shared draw)."""
    questions = [
        {"conv_id": f"mt_{i:012x}", "qid": f"mt_{i:012x}", "question": f"What about topic {i}?"}
        for i in range(5)
    ]
    rows, counts = phase_a._generate_shortfall(
        "char_helios", questions, tmp_path, seed=137, mock=True, gen_model="instruct"
    )
    # mock_scaffold_gen breaks the sentinel on every 5th row -> 4 of 5 kept.
    assert counts["requested"] == 5
    assert counts["merged"] == len(rows) == 4
    assert counts["question_not_verbatim"] == 0
    for r in rows:
        assert r["provenance"] == "generated"
        assert r["conv_id"] == r["qid"]
        assert r["conv_id"].startswith("mt_")
        assert r["question"] in r["scaffold_text"]  # req 6, verbatim
        assert sc.SLOT_SENTINEL in r["scaffold_text"]
    # The subprocess actually ran: its kept file + yield digest are on disk.
    gen_dir = tmp_path / "char_helios" / "gen"
    assert (gen_dir / "scaffolds_helios_mock.jsonl").is_file()
    assert (gen_dir / "scaffold_yield_helios_mock.json").is_file()


def test_gen_char_and_description_resolves_panel_case_insensitively():
    name, desc = phase_a._gen_char_and_description("char_helios")
    assert name == "Helios" and desc  # panel key is 'HELIOS'
    name, desc = phase_a._gen_char_and_description("conversation_paired_stories_assistant")
    assert name == "Assistant" and desc
    with pytest.raises(ValueError):
        phase_a._gen_char_and_description("not_a_variant")


# ---------------------------------------------------------------------------
# Shared question draw (deterministic, filtered) — offline fixture manifest
# ---------------------------------------------------------------------------
class _FakeTokenizer:
    """Signature-conformant boundary fake: __call__(texts, add_special_tokens)
    -> {"input_ids": [...]} with one token per whitespace word."""

    def __call__(self, texts, add_special_tokens=False):
        return {"input_ids": [t.split() for t in texts]}


def _write_manifest_fixture(root: Path) -> Path:
    mdir = root / "manifest"
    mdir.mkdir(parents=True)
    rows0 = [
        {
            "messages": [
                {"role": "user", "content": "How do tides work in narrow bays?"},
                {"role": "assistant", "content": "..."},
            ],
            "source_hash": "sha:aaaaaaaaaaaaaaaa",
        },
        {  # exact-dupe question -> deduped
            "messages": [{"role": "user", "content": "How do tides work in narrow bays?"}],
            "source_hash": "sha:bbbbbbbbbbbbbbbb",
        },
        {  # too short -> char_bounds
            "messages": [{"role": "user", "content": "Hi"}],
            "source_hash": "sha:cccccccccccccccc",
        },
    ]
    rows1 = [
        {
            "messages": [{"role": "user", "content": "Why is the harvest moon orange at rising?"}],
            "source_hash": "sha:dddddddddddddddd",
        },
        {
            "messages": [
                {"role": "user", "content": "What keeps a suspension bridge from swaying?"}
            ],
            "source_hash": "sha:eeeeeeeeeeeeeeee",
        },
        {  # over token budget under the fake tokenizer (one word per token)
            # Kept UNDER QUESTION_MAX_CHARS on purpose: since the r5 supply fix
            # the char bound is 400, so a 6,000-char row would be dropped by
            # char_bounds first and the token branch would go unexercised.
            "messages": [
                {"role": "user", "content": "why does one two three four five six seven eight nine"}
            ],
            "source_hash": "sha:ffffffffffffffff",
        },
        {  # r5 supply fix: MULTILINE questions verbatim-keep at 1.3-3.0%
            "messages": [{"role": "user", "content": "Fix this snippet:\n\n  x = 1\n  print(x)"}],
            "source_hash": "sha:9999999999999999",
        },
    ]
    for i, rows in enumerate((rows0, rows1)):
        with (mdir / f"part_{i:05d}.jsonl").open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
    return mdir


def test_question_draw_deterministic_filtered(tmp_path, monkeypatch):
    mdir = _write_manifest_fixture(tmp_path)
    # The fake tokenizer is one token per whitespace word, and the r5 supply
    # fix caps questions at 400 chars — so the token branch is only reachable
    # with a small budget. 8 admits all three valid fixture questions (7/8/7
    # words) and drops the 10-word over-budget row.
    monkeypatch.setattr(phase_a, "QUESTION_MAX_TOKENS", 8)
    kwargs = dict(
        staging_dir=tmp_path / "unused",
        manifest_dir=mdir,
        tokenizer=_FakeTokenizer(),
        revision="deadbeef",
    )
    drawn1, rec1 = phase_a._draw_shared_questions(3, 137, **kwargs)
    drawn2, _ = phase_a._draw_shared_questions(3, 137, **kwargs)
    assert drawn1 == drawn2  # seeded draw is deterministic
    assert len(drawn1) == 3
    assert rec1["counters"]["dupe_question"] == 1
    assert rec1["counters"]["char_bounds"] == 1
    assert rec1["counters"]["multiline"] == 1
    assert rec1["counters"]["over_token_budget"] == 1
    assert all("\n" not in r["question"] for r in drawn1)
    for r in drawn1:
        assert r["conv_id"] == r["qid"]
        assert r["conv_id"].startswith("mt_") and "__" not in r["conv_id"]
    # Asking for more than the eligible pool fails loud (never backfilled).
    with pytest.raises(RuntimeError, match="short"):
        phase_a._draw_shared_questions(10, 137, **kwargs)


# ---------------------------------------------------------------------------
# Upload shard-split (upload-policy.md >9.5 MB text rule)
# ---------------------------------------------------------------------------
def test_shard_large_jsonl_for_upload(tmp_path):
    small = tmp_path / "small.jsonl"
    small.write_text('{"a": 1}\n' * 10, encoding="utf-8")
    big = tmp_path / "big.jsonl"
    line = json.dumps({"text": "x" * 1000}) + "\n"
    n_lines = (11_000_000 // len(line.encode())) + 1
    with big.open("w", encoding="utf-8") as f:
        for _ in range(n_lines):
            f.write(line)
    assert big.stat().st_size > phase_a.UPLOAD_SHARD_LIMIT_BYTES

    out = phase_a._shard_large_jsonl_for_upload([small, big])
    assert small in out
    assert big not in out  # oversized original replaced by shards
    shards = [p for p in out if ".shard" in p.name]
    manifest = [p for p in out if p.name == "big.manifest.json"]
    assert shards and len(manifest) == 1
    assert all(s.stat().st_size < phase_a.UPLOAD_SHARD_LIMIT_BYTES for s in shards)
    meta = json.loads(manifest[0].read_text(encoding="utf-8"))
    assert sum(meta["line_counts"]) == n_lines
    # Round-trip: concatenated shards reproduce the source bytes.
    joined = b"".join(s.read_bytes() for s in sorted(shards))
    assert joined == big.read_bytes()
