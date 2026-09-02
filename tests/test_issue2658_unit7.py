"""Unit 7 tests: the blinded human/model annotation instrument (offline only).

Pins the four blinded-reads elements (content-only packets, no reader tools,
key-frozen-first, scope-aware leakage scan), the two ban-list scopes and
their split (the #1482 lesson: one list cannot serve both), the write-once
blinding key, non-answer-never-persisted, drop-never-coerce ratings parsing,
and the plan-section-3 adjudicated-label floor arithmetic.

All fixtures are synthetic benign text — no bank items, no real corpora.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import issue2658_common as C  # noqa: E402
import issue2658_human_read as H  # noqa: E402

# ---------------------------------------------------------------------------
# Fixtures (synthetic, benign).
# ---------------------------------------------------------------------------


def make_items(row: str, n: int = 3) -> list[dict]:
    return [
        {
            "row": row,
            "item_id": f"{row}|frame_x|{i:03d}",
            "prompt_text": f"Please explain everyday topic number {i} in one sentence.",
            "answer_text": f"Here is a short, plain explanation of topic number {i}.",
            "operational_label": i % 2,
            "bank_membership": "development",
        }
        for i in range(n)
    ]


def fake_resolver(row: str, item_id: str):
    packet = {
        "schema": "test-only",
        "evidence": {
            "question": "Which weighs more, a kilogram of iron or a kilogram of feathers?",
            "correct_choice": "they weigh the same",
        },
    }
    return packet, "0" * 64


def freeze(tmp_path: Path, items: list[dict], name: str = "blinding_key.json") -> dict:
    return H.freeze_blinding_key(items, tmp_path / name)


# ---------------------------------------------------------------------------
# Rows and questions.
# ---------------------------------------------------------------------------


def test_audited_rows_are_exactly_the_judge_scored_rows() -> None:
    assert tuple(r for r in C.ROW_IDS if C.CONSTRUCTS[r].judge_scored) == H.AUDITED_ROWS
    assert len(H.AUDITED_ROWS) == 8
    for r in ("correctness_math", "correctness_mmlu_pro", "correctness_code"):
        assert r not in H.AUDITED_ROWS


def test_every_audited_row_has_a_blinded_question_and_no_objective_row_does() -> None:
    assert set(H.BLINDED_QUESTIONS) == set(H.AUDITED_ROWS)


@pytest.mark.parametrize("row", H.AUDITED_ROWS)
def test_blinded_question_is_wrapper_clean_and_never_names_the_row(row: str) -> None:
    hits = H.scan_for_leakage([("wrapper", H.BLINDED_QUESTIONS[row])])
    assert hits["wrapper"] == [], f"{row}: wrapper hits {hits['wrapper']}"
    assert row.replace("_", " ") not in H.BLINDED_QUESTIONS[row].lower()
    assert row not in H.BLINDED_QUESTIONS[row].lower()


# ---------------------------------------------------------------------------
# Leakage scan: scope split + matching semantics.
# ---------------------------------------------------------------------------


def test_wrapper_scan_catches_criterion_vocabulary() -> None:
    hits = H.scan_for_leakage(
        [("wrapper", "Which group best predicts the AUROC rank on the dev split?")]
    )
    assert "predict*" in hits["wrapper"]
    assert "auroc" in hits["wrapper"]
    assert "rank*" in hits["wrapper"]
    assert "dev" in hits["wrapper"]
    assert "split*" in hits["wrapper"]


def test_wrapper_word_boundaries_harm_never_matches_arm() -> None:
    clean = H.scan_for_leakage(
        [("wrapper", "could facilitate real harm; the armed guard was harmed")]
    )
    assert clean["wrapper"] == []
    hit = H.scan_for_leakage([("wrapper", "items came from the treatment arm")])
    assert hit["wrapper"] == ["arm"]


def test_wrapper_stem_matching_catches_inflections() -> None:
    hits = H.scan_for_leakage(
        [("wrapper", "sycophantic replies were hallucinating and judged carefully")]
    )
    assert "sycophan*" in hits["wrapper"]
    assert "hallucinat*" in hits["wrapper"]
    assert "judge*" in hits["wrapper"]


def test_payload_allows_ordinary_criterion_english() -> None:
    # The #1482 lesson: real reply text saying "predict"/"rank"/"best" must
    # never refuse the send — the payload bar is project-identifiers-only.
    text = "I rank this option best and predict the r2 will improve on any metric."
    hits = H.scan_for_leakage([("payload", text)])
    assert hits["payload"] == []
    # The SAME text in wrapper scope refuses.
    assert H.scan_for_leakage([("wrapper", text)])["wrapper"] != []


def test_payload_refuses_project_identifiers() -> None:
    hits = H.scan_for_leakage(
        [("payload", "see the explore_persona_space store under i2658-gen for details")]
    )
    assert "explore_persona_space" in hits["payload"]
    assert "i2658*" in hits["payload"]


def test_payload_ban_list_is_subset_of_wrapper_ban_list() -> None:
    assert set(H.PAYLOAD_BANNED) <= set(H.WRAPPER_BANNED)


def test_blinding_key_filename_token_is_on_both_ban_lists() -> None:
    assert any(t.startswith("blinding_key") for t in H.PAYLOAD_BANNED)
    assert any(t.startswith("blinding_key") for t in H.WRAPPER_BANNED)


def test_assert_no_leakage_raises_and_has_no_skip_flag() -> None:
    with pytest.raises(H.LeakageError, match="REFUSING"):
        H.assert_no_leakage([("wrapper", "the judge score")])
    import inspect

    sig = inspect.signature(H.assert_no_leakage)
    assert list(sig.parameters) == ["segments"]  # no skip/force parameter exists


def test_unknown_scan_scope_raises() -> None:
    with pytest.raises(H.LeakageError, match="unknown scan scope"):
        H.scan_for_leakage([("system", "x")])


# ---------------------------------------------------------------------------
# Blinding key: write-once, frozen-first, neutral tags.
# ---------------------------------------------------------------------------


def test_freeze_key_is_write_once(tmp_path: Path) -> None:
    items = make_items("refusal")
    freeze(tmp_path, items)
    with pytest.raises(H.BlindingKeyFrozenError, match="write-once"):
        freeze(tmp_path, items)


def test_freeze_key_requires_ban_covered_filename(tmp_path: Path) -> None:
    with pytest.raises(H.BlindingKeyError, match="must contain"):
        H.freeze_blinding_key(make_items("refusal"), tmp_path / "key.json")


def test_freeze_key_refuses_scan_invisible_filename(tmp_path: Path) -> None:
    """A filename the freeze accepts must be a filename the scan can catch.

    ``my_blinding_key.json`` contains the token but the ban-list regex's
    boundary lookbehind can never match it (preceded by ``_``), so the old
    substring check accepted a name both scan scopes were blind to.
    """
    items = make_items("refusal")
    # Probe the invariant the fix pins: the scan yields ZERO hits on this name.
    hits = H.scan_for_leakage([("payload", "my_blinding_key.json")])
    assert hits == {"wrapper": [], "payload": []}
    with pytest.raises(H.BlindingKeyError, match="word boundary"):
        H.freeze_blinding_key(items, tmp_path / "my_blinding_key.json")
    # Control: the boundary-visible name freezes AND the scan catches it.
    H.freeze_blinding_key(items, tmp_path / "blinding_key.json")
    hits = H.scan_for_leakage([("payload", "blinding_key.json")])
    assert "blinding_key*" in hits["payload"]


def test_freeze_key_rejects_objective_rows(tmp_path: Path) -> None:
    with pytest.raises(H.BlindingKeyError, match="objective"):
        freeze(tmp_path, make_items("correctness_math"))


def test_freeze_key_rejects_unknown_row_and_duplicates(tmp_path: Path) -> None:
    with pytest.raises(H.BlindingKeyError, match="unknown row"):
        freeze(tmp_path, make_items("not_a_row"))
    items = make_items("refusal", 2)
    with pytest.raises(H.BlindingKeyError, match="duplicate"):
        H.freeze_blinding_key([*items, items[0]], tmp_path / "blinding_key_dup.json")


def test_freeze_key_is_deterministic_and_input_order_independent(tmp_path: Path) -> None:
    items = make_items("evil", 5)
    k1 = H.freeze_blinding_key(items, tmp_path / "blinding_key_a.json")
    k2 = H.freeze_blinding_key(list(reversed(items)), tmp_path / "blinding_key_b.json")
    assert k1["entries"] == k2["entries"]
    assert k1["n_items"] == 5


def test_key_tags_are_neutral_and_hide_all_metadata(tmp_path: Path) -> None:
    key = freeze(tmp_path, make_items("impoliteness", 4))
    for tag, hidden in key["entries"].items():
        assert tag.startswith("item_") and tag[5:].isdigit()
        # hidden metadata lives ONLY in the key entry, never in the tag
        assert hidden["row"] == "impoliteness"
        assert "item_id" in hidden and "operational_label" in hidden
        assert "answer_sha256" in hidden and "prompt_sha256" in hidden
        # display text never enters the key
        assert "answer_text" not in hidden and "prompt_text" not in hidden


def test_load_blinding_key_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(H.BlindingKeyError, match="BEFORE composing"):
        H.load_blinding_key(tmp_path / "absent_blinding_key.json")


# ---------------------------------------------------------------------------
# Packet composition.
# ---------------------------------------------------------------------------

NON_EVIDENCE_ROWS = tuple(r for r in H.AUDITED_ROWS if not C.CONSTRUCTS[r].uses_evidence_packet)
EVIDENCE_ROWS = tuple(r for r in H.AUDITED_ROWS if C.CONSTRUCTS[r].uses_evidence_packet)


@pytest.mark.parametrize("row", H.AUDITED_ROWS)
def test_compose_every_audited_row_scans_clean_and_opens_neutrally(
    row: str, tmp_path: Path
) -> None:
    items = make_items(row)
    key = freeze(tmp_path, items)
    segments, tags = H.compose_packet(items, key, row, resolver=fake_resolver)
    assert len(tags) == len(items)
    msg = H.build_user_message(segments)
    assert msg.startswith("# Items\n3 items follow.")
    assert H.scan_for_leakage(segments) == {"wrapper": [], "payload": []}
    # item texts and the question are present
    assert "plain explanation of topic" in msg
    assert H.BLINDED_QUESTIONS[row] in msg


def test_compose_requires_the_frozen_key_to_cover_every_item(tmp_path: Path) -> None:
    items = make_items("refusal", 3)
    key = freeze(tmp_path, items[:2])
    with pytest.raises(H.BlindingKeyError, match="not in the frozen blinding key"):
        H.compose_packet(items, key, "refusal")


def test_compose_refuses_mixed_rows(tmp_path: Path) -> None:
    items = make_items("refusal", 2) + make_items("evil", 1)
    key = freeze(tmp_path, items)
    with pytest.raises(H.BlindingKeyError, match="single-row"):
        H.compose_packet(items, key, "refusal")


def test_compose_orders_items_by_frozen_tag_order_not_input_order(tmp_path: Path) -> None:
    items = make_items("casualness", 5)
    key = freeze(tmp_path, items)
    seg_fwd, tags_fwd = H.compose_packet(items, key, "casualness")
    seg_rev, tags_rev = H.compose_packet(list(reversed(items)), key, "casualness")
    assert seg_fwd == seg_rev and tags_fwd == tags_rev
    assert tags_fwd == sorted(tags_fwd, key=lambda t: int(t.split("_", 1)[1]))


def test_compose_message_is_exact_segment_concatenation(tmp_path: Path) -> None:
    items = make_items("evil")
    key = freeze(tmp_path, items)
    segments, _tags = H.compose_packet(items, key, "evil")
    assert H.build_user_message(segments) == "".join(t for _s, t in segments)


@pytest.mark.parametrize("row", EVIDENCE_ROWS)
def test_compose_embeds_reference_material_without_provenance(row: str, tmp_path: Path) -> None:
    items = make_items(row, 2)
    key = freeze(tmp_path, items)
    segments, _tags = H.compose_packet(items, key, row, resolver=fake_resolver)
    msg = H.build_user_message(segments)
    assert "<reference>" in msg
    assert "kilogram of iron" in msg  # the evidence content block
    assert "test-only" not in msg  # packet schema never ships
    assert "sha256" not in msg  # no digests in the request
    assert "[EVIDENCE" not in msg  # the judge instrument's framing never ships


def test_compose_non_evidence_rows_have_no_reference_block(tmp_path: Path) -> None:
    items = make_items("refusal")
    key = freeze(tmp_path, items)
    segments, _tags = H.compose_packet(items, key, "refusal")
    assert "<reference>" not in H.build_user_message(segments)


def test_leaky_payload_refuses_the_send(tmp_path: Path) -> None:
    items = make_items("refusal")
    items[1]["answer_text"] = "as recorded in explore_persona_space eval outputs"
    key = freeze(tmp_path, items)
    with pytest.raises(H.LeakageError, match="explore_persona_space"):
        H.compose_packet(items, key, "refusal")


# ---------------------------------------------------------------------------
# Frozen-key content-sha verification: post-freeze item drift NEVER ships.
# (The join is by (row, item_id) only, so a regenerated items file — same ids,
# different sampled text — would otherwise send new text under frozen tags
# whose hidden metadata describes the old text.)
# ---------------------------------------------------------------------------


def test_compose_refuses_drifted_answer_text(tmp_path: Path) -> None:
    items = make_items("evil")
    key = freeze(tmp_path, items)
    drifted = [dict(it) for it in items]
    drifted[1]["answer_text"] = "A completely regenerated benign reply about topic one."
    with pytest.raises(H.BlindingKeyError, match="drifted after the key froze") as ei:
        H.compose_packet(drifted, key, "evil")
    msg = str(ei.value)
    assert drifted[1]["item_id"] in msg and "answer_text" in msg
    # The exception string is a surface that ends up in logs: never item text.
    assert "regenerated benign reply" not in msg


def test_compose_refuses_drifted_prompt_text(tmp_path: Path) -> None:
    items = make_items("refusal")
    key = freeze(tmp_path, items)
    drifted = [dict(it) for it in items]
    drifted[0]["prompt_text"] = "A completely regenerated benign request about topic zero."
    with pytest.raises(H.BlindingKeyError, match="drifted after the key froze") as ei:
        H.compose_packet(drifted, key, "refusal")
    msg = str(ei.value)
    assert drifted[0]["item_id"] in msg and "prompt_text" in msg
    assert "regenerated benign request" not in msg


def test_every_packet_emitting_path_refuses_drifted_items(tmp_path: Path) -> None:
    """All three packet paths route through compose-time verification: the
    human-packet writer and the model dispatch both compose internally, so a
    drifted items file raises before any packet text or file exists."""
    items = make_items("casualness")
    key = freeze(tmp_path, items)
    drifted = [dict(it) for it in items]
    drifted[2]["answer_text"] = "Regenerated text that the frozen key never fingerprinted."
    out_dir = tmp_path / "packets"
    with pytest.raises(H.BlindingKeyError, match="drifted"):
        H.write_human_packets(drifted, key, "casualness", out_dir)
    assert not out_dir.exists() or not any(out_dir.iterdir())  # nothing shipped
    out = tmp_path / "read.txt"
    # dispatch_model_read raises at compose time, BEFORE any client exists —
    # no live API call is made and nothing is persisted.
    with pytest.raises(H.BlindingKeyError, match="drifted"):
        H.dispatch_model_read(drifted, key, out=out, row="casualness", key_path="k")
    assert not out.exists()


def test_evidence_display_requires_an_evidence_block() -> None:
    def bad_resolver(row, item_id):
        return {"schema": "test-only"}, "0" * 64

    with pytest.raises(H.EvidenceDisplayError, match="no evidence block"):
        H.evidence_display_text("sycophancy", "sycophancy|f|000", resolver=bad_resolver)
    assert H.evidence_display_text("refusal", "refusal|f|000") is None


# ---------------------------------------------------------------------------
# Reader independence + non-answer handling.
# ---------------------------------------------------------------------------


def test_reader_may_never_be_the_frozen_judge_model() -> None:
    with pytest.raises(H.ReaderIndependenceError):
        H.require_reader_independence(C.JUDGE["model"])
    with pytest.raises(H.ReaderIndependenceError):
        H.require_reader_independence("claude-sonnet-4-5")
    H.require_reader_independence(H.MODEL)  # the default reader passes


def test_non_answers_raise_and_nothing_is_persisted() -> None:
    with pytest.raises(H.NonAnswerError, match="raise the cap"):
        H.require_end_turn("max_tokens", "partial", 2048)
    with pytest.raises(H.NonAnswerError, match="non-answer"):
        H.require_end_turn("refusal", "", 2048)
    with pytest.raises(H.NonAnswerError, match="empty response"):
        H.require_end_turn("end_turn", "   ", 2048)
    H.require_end_turn("end_turn", "item_0001: 10", 2048)


def test_default_max_tokens_is_generous() -> None:
    assert H.default_max_tokens(3) >= 2048
    assert H.default_max_tokens(250) > H.default_max_tokens(3)


# ---------------------------------------------------------------------------
# Ratings parsing: drop-never-coerce.
# ---------------------------------------------------------------------------

TAGS = ["item_0001", "item_0002"]


def test_parse_ratings_happy_path() -> None:
    out = H.parse_ratings("item_0001: 85\n\nitem_0002: 0\n", TAGS)
    assert out == {"item_0001": 85, "item_0002": 0}


@pytest.mark.parametrize(
    ("text", "match"),
    [
        ("item_0001: 85\n", "unrated"),
        ("item_0001: 85\nitem_0009: 3\n", "unknown tag"),
        ("item_0001: 85\nitem_0001: 3\nitem_0002: 1\n", "duplicate"),
        ("item_0001: 101\nitem_0002: 3\n", "outside"),
        ("item_0001: -5\nitem_0002: 3\n", "outside"),
        ("item_0001: eighty\nitem_0002: 3\n", "unparseable"),
        ("the first item is great\n", "unparseable"),
    ],
)
def test_parse_ratings_rejects_defects(text: str, match: str) -> None:
    with pytest.raises(H.RatingsParseError, match=match):
        H.parse_ratings(text, TAGS)


# ---------------------------------------------------------------------------
# Sizing + adjudicated floor (plan section 3 / section 10).
# ---------------------------------------------------------------------------


def test_sizing_report_arithmetic_at_defaults() -> None:
    rep = H.sizing_report()
    assert rep["floor_per_class_adjudicated"] == 100
    assert rep["n_audited_rows"] == 8 and rep["banks"] == ["dev", "test"]
    assert rep["n_readers"] == 2
    assert rep["items_sampled_per_class"] == 125  # ceil(100 * 1.25)
    assert rep["items_per_trait_bank"] == 250
    assert rep["reads_per_trait_bank"] == 500
    assert rep["total_items"] == 250 * 8 * 2 == 4000
    assert rep["total_reads"] == 500 * 8 * 2 == 8000
    assert rep["projected_human_hours"] == pytest.approx(8000 * 1.5 / 60.0)


def test_sizing_report_never_undershoots_the_floor() -> None:
    rep = H.sizing_report(oversample=1.0)
    assert rep["items_sampled_per_class"] == 100
    with pytest.raises(ValueError, match="oversample"):
        H.sizing_report(oversample=0.5)


def test_check_adjudicated_floor() -> None:
    H.check_adjudicated_floor(100, 100)
    with pytest.raises(H.AdjudicatedFloorError, match="below the plan-section-3 floor"):
        H.check_adjudicated_floor(99, 150)
    with pytest.raises(H.AdjudicatedFloorError):
        H.check_adjudicated_floor(150, 99)


# ---------------------------------------------------------------------------
# Audit sidecar + human packets.
# ---------------------------------------------------------------------------


def test_audit_record_shape(tmp_path: Path) -> None:
    items = make_items("evil")
    key = freeze(tmp_path, items)
    segments, tags = H.compose_packet(items, key, "evil")
    rec = H.build_audit_record(
        model=H.MODEL,
        max_tokens=2048,
        temperature=1.0,
        betas=[],
        segments=segments,
        hits=H.scan_for_leakage(segments),
        stop_reason="end_turn",
        input_tokens=10,
        output_tokens=5,
        row="evil",
        n_items=len(tags),
        key_path="k",
    )
    assert rec["system_prompt"] is None and rec["tools"] is None
    assert rec["n_messages"] == 1
    assert rec["outbound_request_verbatim"] == H.build_user_message(segments)
    assert rec["outbound_chars"] == len(H.build_user_message(segments))
    for scope in ("wrapper", "payload"):
        block = rec["leakage_scan_scopes"][scope]
        assert block["banned_terms"] and block["hits"] == [] and block["chars"] > 0
    assert rec["stop_reason"] == "end_turn"
    assert rec["usage"] == {"input_tokens": 10, "output_tokens": 5}
    json.dumps(rec)  # serializable


def test_write_human_packets_double_annotation_per_reader_order(tmp_path: Path) -> None:
    """Readers get the SAME items in PER-READER deterministic order, with the
    position -> tag mapping persisted so adjudication joins back by tag.
    (Byte-identical shared-order packets would correlate order/context effects
    across the double annotation that feeds the kappa/ICC reliability gate.)"""
    items = make_items("casualness", 8)
    key = freeze(tmp_path, items)
    out = tmp_path / "packets"
    written = H.write_human_packets(items, key, "casualness", out)
    assert len(written) == 6  # 2 readers x (packet + answer sheet + order sidecar)
    pa = (out / "reader_a_packet.md").read_text()
    pb = (out / "reader_b_packet.md").read_text()
    oa = json.loads((out / "reader_a_order.json").read_text())
    ob = json.loads((out / "reader_b_order.json").read_text())
    # Same item set, different order; the permutation is of the composed tag
    # list only (no hidden field orders it).
    assert sorted(oa["order"]) == sorted(ob["order"])
    assert oa["order"] != ob["order"]
    assert pa != pb
    for text, order in ((pa, oa["order"]), (pb, ob["order"])):
        positions = [text.index(f"<item {t}>") for t in order]
        assert positions == sorted(positions)  # packet renders the recorded order
    sheet_a = (out / "reader_a_answer_sheet.txt").read_text()
    sheet_tags = [ln.split(":")[0] for ln in sheet_a.splitlines() if ln.startswith("item_")]
    assert sheet_tags == oa["order"]  # answer sheet matches the reader's order
    assert oa["key_fingerprint"] == ob["key_fingerprint"] == H.key_fingerprint(key)
    # Deterministic: a second emission reproduces every byte.
    out2 = tmp_path / "packets2"
    H.write_human_packets(items, key, "casualness", out2)
    assert (out2 / "reader_a_packet.md").read_text() == pa
    assert (out2 / "reader_b_packet.md").read_text() == pb


def test_write_human_packets_refuses_leaky_items(tmp_path: Path) -> None:
    items = make_items("evil")
    # The leak is present at FREEZE time, so the content-sha check passes and
    # the LEAKAGE scan (not the drift check) is what refuses the write.
    items[0]["answer_text"] = "Contact superkaiba1 for details."
    key = freeze(tmp_path, items)
    with pytest.raises(H.LeakageError):
        H.write_human_packets(items, key, "evil", tmp_path / "p")


# ---------------------------------------------------------------------------
# Atomic writes: an interrupted freeze never bricks the write-once path, and
# the dispatch persistence goes through the shared atomic primitives.
# ---------------------------------------------------------------------------


def test_interrupted_freeze_does_not_brick_the_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pre-fix, a plain write_text left a PARTIAL key at the destination on
    interruption, and the write-once refusal then bricked the path until a
    manual delete. Atomic write-temp + os.replace leaves nothing behind."""
    items = make_items("evil")
    target = tmp_path / "blinding_key.json"
    real_write_text = Path.write_text

    def exploding_write_text(self: Path, text: str, *args, **kwargs):
        if "blinding_key" in self.name:
            real_write_text(self, text[: len(text) // 2], *args, **kwargs)
            raise OSError("simulated interruption mid-write")
        return real_write_text(self, text, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", exploding_write_text)
    with pytest.raises(OSError, match="simulated interruption"):
        H.freeze_blinding_key(items, target)
    monkeypatch.undo()
    assert not target.exists()  # no partial key at the destination
    assert not list(tmp_path.glob("*.tmp"))  # no temp residue either
    H.freeze_blinding_key(items, target)  # the path is NOT bricked
    assert json.loads(target.read_text())["n_items"] == len(items)


def test_dispatch_and_freeze_persist_through_atomic_primitives() -> None:
    """The response text, audit sidecar, and ratings writes cannot be smoke-run
    without a live API call, so pin the mechanism at the source level: every
    persist site routes through the shared atomic_io helpers and no plain
    ``.write_text(`` remains in either function."""
    import inspect

    for fn in (H.dispatch_model_read, H.freeze_blinding_key, H.write_human_packets):
        src = inspect.getsource(fn)
        assert "write_text_atomic(" in src or "write_json_atomic(" in src, fn.__name__
        assert ".write_text(" not in src, fn.__name__


# ---------------------------------------------------------------------------
# Module-level contracts.
# ---------------------------------------------------------------------------


def test_module_carries_the_dispatch_routing_waiver_token() -> None:
    src = (Path(H.__file__)).read_text(encoding="utf-8")
    assert "# API_DISPATCH_ROUTING_EXEMPT:" in src


def test_guard_classes_are_issue2658_guards() -> None:
    for exc in (
        H.LeakageError,
        H.BlindingKeyError,
        H.BlindingKeyFrozenError,
        H.ReaderIndependenceError,
        H.NonAnswerError,
        H.RatingsParseError,
        H.AdjudicatedFloorError,
        H.EvidenceDisplayError,
    ):
        assert issubclass(exc, C.Issue2658GuardError)


def test_import_check_mode_passes() -> None:
    assert H.main(["--import-check"]) == 0


def test_sizing_report_cli_mode(capsys: pytest.CaptureFixture[str]) -> None:
    assert H.main(["--sizing-report"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["total_reads"] == 8000
