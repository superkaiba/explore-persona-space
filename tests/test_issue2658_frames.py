"""#2658 P0 unit-2 tests: frames, strata, superfamily graph, extraction
exclusion, dev/test split, and manifest immutability.

Every guard RAISES on its bad shape; behavior tests use small synthetic
``PromptItem`` lists (no network, no torch, no git subprocess, no other issues'
committed artifacts). The on-disk manifests, when present, are validated as an
integration check (skipped in a fresh checkout that has not run the build).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import issue2658_common as C  # noqa: E402
import issue2658_frames as F  # noqa: E402


def _item(item_id, text="", problem_id=None, row="evil", frame="f", **kw):
    return F.PromptItem(
        item_id=item_id,
        prompt_sha256=F._sha_text(text or item_id),
        origin=kw.pop("origin", "frame"),
        source_ref="test",
        text=text,
        problem_id=problem_id,
        row=row,
        frame=frame,
        **kw,
    )


# ---------------------------------------------------------------------------
# Registry sanity
# ---------------------------------------------------------------------------
def test_registry_covers_row_ids_with_four_frames_three_strata():
    assert tuple(F.FRAMES) == C.ROW_IDS
    for row, rf in F.FRAMES.items():
        assert len(rf.frames) == C.PILOT.source_frames == 4, row
        assert len(rf.strata) == C.PILOT.strata == 3, row
        names = [fr.name for fr in rf.frames]
        assert len(set(names)) == len(names), f"{row} duplicate frame names {names}"


def test_every_row_stratifier_is_non_deterministic():
    # The module runs this at import via _validate_registry(); re-assert here so a
    # future banned-field stratum is a test failure, not just an import failure.
    for rf in F.FRAMES.values():
        F.assert_stratifier_not_deterministic(rf.strata)


def test_harmful_compliance_frames_present_but_not_extraction_eligible():
    # harmful_compliance runs C0/C1/C4/C5 (no frozen direction) — it still needs
    # frames, but must NOT be in the eligible extraction set.
    assert "harmful_compliance" in F.FRAMES
    assert len(F.FRAMES["harmful_compliance"].frames) == 4


# ---------------------------------------------------------------------------
# Deterministic-stratum detector (plan §4: strata retain class overlap)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("banned", ["agree_frac", "label", "is_correct", "judge_score"])
def test_deterministic_stratum_detector_raises_on_outcome_feature(banned):
    strata = (
        F.StratumSpec("a", f"difficulty_by:{banned}", "rationale present"),
        F.StratumSpec("b", "prompt_length_band:mid", "rationale present"),
        F.StratumSpec("c", "prompt_length_band:high", "rationale present"),
    )
    with pytest.raises(F.DeterministicStratumError):
        F.assert_stratifier_not_deterministic(strata)


def test_deterministic_stratum_detector_raises_on_missing_rationale():
    strata = (F.StratumSpec("a", "prompt_length_band:low", "   "),)
    with pytest.raises(F.DeterministicStratumError):
        F.assert_stratifier_not_deterministic(strata)


def test_non_outcome_stratifier_passes():
    F.assert_stratifier_not_deterministic(F._BEHAVIORAL_STRATA)
    F.assert_stratifier_not_deterministic(F._MATH_STRATA)  # 'level' is NOT banned
    F.assert_stratifier_not_deterministic(F._HASH_STRATA)


# ---------------------------------------------------------------------------
# Superfamily graph — exact identity, near-dup, problem-key edges
# ---------------------------------------------------------------------------
def test_exact_normalized_identity_edge():
    items = [_item("a", "Explain gravity."), _item("b", "  explain   GRAVITY!  ")]
    sf, _ = F.build_superfamilies(items)
    assert sf["a"] == sf["b"]


def test_near_duplicate_edge_and_separation():
    base = "Write a short poem about the ocean and its waves at sunrise today"
    near = "Write a short poem about the ocean and its waves at sunrise now"  # tiny edit
    far = "Compute the derivative of x squared plus three x minus seven please"
    sf, _ = F.build_superfamilies([_item("near1", base), _item("near2", near), _item("far", far)])
    assert sf["near1"] == sf["near2"]
    assert sf["far"] != sf["near1"]


def test_benchmark_problem_key_identity_edge():
    # id-only nodes (text="") join iff they share a problem_id (group_key).
    items = [
        _item("x", problem_id="gk-1"),
        _item("y", problem_id="gk-1"),
        _item("z", problem_id="gk-2"),
    ]
    sf, _ = F.build_superfamilies(items)
    assert sf["x"] == sf["y"]
    assert sf["z"] != sf["x"]


def test_length_band_blocking_flag_under_cap():
    items = [_item(f"i{i}", f"unique prompt number {i} about topic {i}") for i in range(20)]
    _, blocked = F.build_superfamilies(items)
    assert blocked is False  # 20 << lexical_all_pairs_cap


# ---------------------------------------------------------------------------
# Group-D fix M1: keyed items participate in the text-identity edge via the
# pre-composition STEM, so same-content different-key entries merge.
# ---------------------------------------------------------------------------
def _keyed_item(item_id, stem, key, band="hedged_assertion", text="composed wrapper"):
    return _item(
        item_id,
        text=f"{text} {stem}",
        problem_id=key,
        band_key=key,
        stem_text=stem,
        row="sycophancy",
    )


def test_keyed_same_stem_different_keys_share_superfamily():
    # The M1 leak shape: two dataset entries with the SAME question stem but
    # DIFFERENT keys (e.g. arc:...#783 / arc:...#1101) used to land in two
    # superfamilies and straddle dev/test.
    a = _keyed_item("sycophancy|f1|k1", "Which gas do plants absorb?", "k1")
    b = _keyed_item("sycophancy|f2|k2", "  which GAS do plants  absorb? ", "k2")
    c = _keyed_item("sycophancy|f1|k3", "A totally different question stem?", "k3")
    sf, _ = F.build_superfamilies([a, b, c])
    assert sf["sycophancy|f1|k1"] == sf["sycophancy|f2|k2"]
    assert sf["sycophancy|f1|k3"] != sf["sycophancy|f1|k1"]


def test_keyed_stem_matching_freetext_extraction_merges_and_bars():
    # DUPLICATE-TEXT fixture (group-G re-review requirement): a keyed frame
    # item whose STEM equals a free-text extraction question is a REAL
    # content overlap — merged, barred, and disclosed as measured.
    shared = "What is the boiling point of water?"
    frame = [
        _keyed_item("sycophancy|f1|k1", shared, "k1"),
        _keyed_item("sycophancy|f1|k2", "An unrelated benign stem?", "k2"),
    ]
    extr = [_item("sycophancy|extraction:pv#0", text=shared, origin="extraction")]
    sf, _ = F.build_superfamilies(frame + extr)
    assert sf["sycophancy|f1|k1"] == sf["sycophancy|extraction:pv#0"]
    frame_sf = {sf[it.item_id] for it in frame}
    extraction_sf = {sf[it.item_id] for it in extr}
    barred = frame_sf & extraction_sf
    assert sf["sycophancy|f1|k1"] in barred
    m = F._overlap_measurement("sycophancy", frame, extr, barred)
    assert m["status"] == "measured"
    assert "exact-text" in m["spanning_edge_domains"]


def test_composed_keyed_item_without_stem_raises():
    # Fail-fast guard: a composed-text keyed item (band_key set) with no
    # recorded stem would silently regain the M1 text-identity exemption.
    bad = _item("sycophancy|f1|k9", text="composed text", problem_id="k9", band_key="k9")
    with pytest.raises(F.FrameManifestError, match="stem_text"):
        F.build_superfamilies([bad])
    # the disclosure derives from the SAME predicate, so it raises identically
    with pytest.raises(F.FrameManifestError, match="stem_text"):
        F._edge_domains([bad])


def test_edge_domain_disclosure_coupled_to_graph_edges():
    # Cross-boundary coupling (group-G re-review): under maximally DUPLICATED
    # content, two populations merge in the graph IFF their edge domains span.
    stem = "Exactly the same benign content here?"
    keyed_with_stem = [_keyed_item("sycophancy|f1|kA", stem, "kA")]
    id_only_keyed = [_item("x|f|c1", problem_id=stem)]  # id echoes the text
    free_text = [_item("x|extraction#0", text=stem, origin="extraction")]

    for pop_a, pop_b in [
        (keyed_with_stem, free_text),  # spanning: exact-text
        (id_only_keyed, free_text),  # disjoint: problem-id vs exact-text/lexical
    ]:
        spanning = set(F._edge_domains(pop_a)) & set(F._edge_domains(pop_b))
        sf, _ = F.build_superfamilies(pop_a + pop_b)
        merged = sf[pop_a[0].item_id] == sf[pop_b[0].item_id]
        assert merged == bool(spanning), (
            F._edge_domains(pop_a),
            F._edge_domains(pop_b),
            merged,
        )


# ---------------------------------------------------------------------------
# Extraction exclusion actually bars (the third exclusion set)
# ---------------------------------------------------------------------------
def test_extraction_overlap_bars_frame_superfamily_from_test():
    # frame item shares text with an extraction item -> same superfamily -> barred.
    frame = [_item("frame|shared", "leak this exact prompt"), _item("frame|unique", "distinct one")]
    extr = [_item("extr|shared", "leak this exact prompt", origin="extraction")]
    sf, _ = F.build_superfamilies(frame + extr)
    frame_sf = {sf[it.item_id] for it in frame}
    extraction_sf = {sf[it.item_id] for it in extr}
    barred = frame_sf & extraction_sf
    assert sf["frame|shared"] in barred
    assert sf["frame|unique"] not in barred
    splits = F.assign_splits(frame_sf, barred, split_hints={})
    assert splits[sf["frame|shared"]] == "dev"  # barred -> dev-only, never test


def test_barred_superfamily_never_assigned_to_test():
    frame_sf = {"sf-a", "sf-b", "sf-c"}
    barred = {"sf-a"}
    splits = F.assign_splits(frame_sf, barred, split_hints={})
    test_sf = {s for s, sp in splits.items() if sp == "test"}
    assert not (barred & test_sf)


# ---------------------------------------------------------------------------
# dev/test split — disjointness + correctness split hints
# ---------------------------------------------------------------------------
def test_split_dev_test_disjoint():
    frame_sf = {f"sf-{i}" for i in range(50)}
    splits = F.assign_splits(frame_sf, barred=set(), split_hints={})
    dev = {s for s, sp in splits.items() if sp == "dev"}
    test = {s for s, sp in splits.items() if sp == "test"}
    assert dev | test == frame_sf
    C.assert_split_lineage_disjoint(dev, test)  # unit-1 guard must pass


def test_correctness_split_hint_maps_test_and_dev():
    frame_sf = {"sf-train", "sf-dev", "sf-test"}
    barred = {"sf-train"}  # a #2388 train superfamily is barred (extraction)
    hints = {"sf-dev": "dev", "sf-test": "test"}
    splits = F.assign_splits(frame_sf, barred, hints)
    assert splits["sf-train"] == "dev"  # barred wins
    assert splits["sf-dev"] == "dev"
    assert splits["sf-test"] == "test"


def test_split_deterministic_across_calls():
    frame_sf = {f"sf-{i}" for i in range(30)}
    a = F.assign_splits(frame_sf, set(), {})
    b = F.assign_splits(frame_sf, set(), {})
    assert a == b


# ---------------------------------------------------------------------------
# Stratum band assignment
# ---------------------------------------------------------------------------
def test_math_stratum_uses_intrinsic_level():
    row = "correctness_math"
    assert F.stratum_band_of(_item("m1", row=row, level=1), row) == "level_low"
    assert F.stratum_band_of(_item("m2", row=row, level=2), row) == "level_low"
    assert F.stratum_band_of(_item("m3", row=row, level=3), row) == "level_mid"
    assert F.stratum_band_of(_item("m4", row=row, level=None), row) == "level_mid"
    assert F.stratum_band_of(_item("m5", row=row, level=5), row) == "level_high"


def test_hash_stratum_bands_are_stable_and_in_range():
    row = "correctness_mmlu_pro"
    valid = {s.name for s in F.FRAMES[row].strata}
    for i in range(30):
        b = F.stratum_band_of(_item(f"q{i}", row=row), row)
        assert b in valid
    # deterministic
    it = _item("q-stable", row=row)
    assert F.stratum_band_of(it, row) == F.stratum_band_of(it, row)


def test_empty_string_band_key_keys_on_itself_never_the_sha_fallback():
    # `band_key or prompt_sha256` silently re-entered the circular sha path on
    # an empty-string key; the explicit `is not None` test must not.
    row = "sycophancy"
    expected = F._band_from_key(row, "")
    # pick an item whose sha-fallback band DIFFERS from the empty-key band, so
    # the old `or` fallback is distinguishable from the fixed behavior.
    it = None
    for i in range(50):
        cand = _item(f"bk-empty-{i}", text=f"benign probe {i}", row=row, band_key="")
        if F._band_from_key(row, cand.prompt_sha256) != expected:
            it = cand
            break
    assert it is not None, "no discriminating fixture found in 50 candidates"
    assert F.stratum_band_of(it, row) == expected


# ---------------------------------------------------------------------------
# Group-D fix M3: MMLU-Pro option continuation lines JOIN, never silently drop.
# ---------------------------------------------------------------------------
def _render_mmlu_prompt(options: list[str]) -> str:
    letters = [chr(ord("A") + i) for i in range(len(options))]
    block = "\n".join(f"{ell}. {o}" for ell, o in zip(letters, options, strict=True))
    return f"Answer the question.\n\nWhat is benign question one?\n\n{block}"


def test_option_continuation_lines_join_onto_preceding_option():
    opts = [
        "plain first option",
        "second option line one\n  continuation line two",  # interior continuation
        "third option\nsecond line\nthird line",  # LAST option: trailing continuation
    ]
    labels, texts = F._parse_enumerated_options(_render_mmlu_prompt(opts), 3, "syn-1")
    assert labels == ["A", "B", "C"]
    assert texts == opts  # nothing dropped, bytes preserved


def test_option_parse_still_raises_on_broken_label_sequence():
    prompt = "Answer.\n\nQ?\n\nA. one\nC. three"  # B missing
    with pytest.raises(F.FrameManifestError, match="recovered option labels"):
        F._parse_enumerated_options(prompt, 2, "syn-2")


def test_option_parse_unaffected_rows_are_byte_identical():
    opts = ["alpha", "beta", "gamma"]
    _, texts = F._parse_enumerated_options(_render_mmlu_prompt(opts), 3, "syn-3")
    assert texts == opts


# ---------------------------------------------------------------------------
# Content-addressing stays fail-loud (no default=str coercion).
# ---------------------------------------------------------------------------
def test_canonical_sha_rejects_non_json_native_values():
    with pytest.raises(TypeError):
        F._canonical_sha({"a": {1, 2, 3}})  # a set would str()-coerce nondeterministically


# ---------------------------------------------------------------------------
# Ledger internal cross-checks (by-cause sum + per-row record arithmetic).
# ---------------------------------------------------------------------------
def _ledger_row(row, n_cells, n_est, records):
    return {
        "row": row,
        "n_cells": n_cells,
        "n_cells_estimable": n_est,
        "prospective_not_estimable": records,
    }


def test_ledger_per_row_record_count_mismatch_raises():
    rows = [_ledger_row("evil", 12, 10, [{"cell": "f|b", "n_test_eligible": 3, "cause": "x"}])]
    with pytest.raises(F.FrameManifestError, match="!= n_cells - n_cells_estimable"):
        F._not_estimable_ledger(rows)  # 1 record but arithmetic says 2


def test_ledger_consistent_rows_reconcile():
    rows = [
        _ledger_row("evil", 12, 11, [{"cell": "f|b", "n_test_eligible": 3, "cause": "x"}]),
        _ledger_row("refusal", 12, 12, []),
    ]
    led = F._not_estimable_ledger(rows)
    assert led["n_cells_not_estimable"] == 1
    assert sum(led["by_cause"].values()) == led["n_cells_not_estimable"]


# ---------------------------------------------------------------------------
# Overlap-measurement disclosure (measured-zero vs structurally-inert zero)
# ---------------------------------------------------------------------------
def test_overlap_measurement_structurally_inert_idonly_keyed_frames_freetext_extraction():
    # id-only keyed frame nodes (problem-id edge only) vs free-text extraction
    # nodes: no edge criterion can span the populations, so zero overlap is BY
    # CONSTRUCTION. (Composed keyed frames with stems are NO LONGER inert —
    # their stems enter the exact-text edge; see the measured tests below.)
    frame = [_item(f"fr{i}", problem_id=f"k{i}") for i in range(3)]
    extr = [_item(f"ex{i}", text=f"free question {i}", origin="extraction") for i in range(2)]
    m = F._overlap_measurement("correctness_math", frame, extr, set())
    assert m["status"] == "structurally-inert"
    assert m["frame_edge_domains"] == ["problem-id"]
    assert m["extraction_edge_domains"] == ["exact-text", "lexical"]
    assert m["spanning_edge_domains"] == []
    assert "BY CONSTRUCTION" in m["detail"]


def test_overlap_measurement_measured_when_edge_domains_span():
    # free-text x free-text: edges 2-4 can span
    m = F._overlap_measurement(
        "evil",
        [_item("fr", text="a benign question")],
        [_item("ex", text="another benign question", origin="extraction")],
        set(),
    )
    assert m["status"] == "measured"
    assert m["spanning_edge_domains"] == ["exact-text", "lexical"]
    # keyed x keyed (correctness-style id-only nodes): edge 1 can span
    mk = F._overlap_measurement(
        "correctness_math",
        [_item("frk", problem_id="gk-1")],
        [_item("exk", problem_id="gk-9", origin="extraction")],
        set(),
    )
    assert mk["status"] == "measured" and mk["spanning_edge_domains"] == ["problem-id"]
    # composed keyed frame (stem recorded) x free-text extraction: the M1 fix
    # gives the stems an exact-text edge, so sycophancy's overlap is MEASURED
    # (zero included), no longer structurally inert.
    ms = F._overlap_measurement(
        "sycophancy",
        [_keyed_item("sycophancy|f1|k1", "a benign stem?", "k1")],
        [_item("ex", text="an unrelated free question", origin="extraction")],
        set(),
    )
    assert ms["status"] == "measured" and ms["spanning_edge_domains"] == ["exact-text"]


def test_overlap_measurement_no_extraction_items_and_inert_consistency_guard():
    m = F._overlap_measurement("harmful_compliance", [_item("fr", text="q")], [], set())
    assert m["status"] == "no-extraction-items"
    # disjoint edge domains + a nonempty barred set violates the homogeneity
    # invariant of the superfamily graph — fail loud, never disclose "inert"
    with pytest.raises(F.FrameManifestError, match="edge-domain"):
        F._overlap_measurement(
            "sycophancy",
            [_item("fr", text="t", problem_id="k1")],
            [_item("ex", text="q", origin="extraction")],
            {"sf-bogus"},
        )


def _fake_frame_loader(shared_text=None, keyed=False, keyed_shared_stem=None):
    """Boundary fake for load_frame_prompts (signature-conformant: (row, frame)).

    ``keyed`` emits COMPOSED-text keyed items (stem recorded, per the M1 fix);
    ``keyed="id-only"`` emits id-only keyed nodes (problem-id edge only);
    ``keyed_shared_stem`` plants one duplicate stem in the first frame.
    """

    def loader(row, frame):
        out = []
        for i in range(3):
            text = f"{frame.name} benign prompt {i} about {row}"
            if shared_text is not None and frame.name and i == 0:
                text = shared_text if frame == F.FRAMES[row].frames[0] else text
            kw = {}
            if keyed == "id-only":
                text = ""
                kw = {"problem_id": f"{frame.name}-k{i}"}
            elif keyed:
                stem = f"{frame.name} benign stem {i} about {row}?"
                if keyed_shared_stem is not None and i == 0 and frame == F.FRAMES[row].frames[0]:
                    stem = keyed_shared_stem
                kw = {
                    "problem_id": f"{frame.name}-k{i}",
                    "band_key": f"{frame.name}-k{i}",
                    "stem_text": stem,
                }
            out.append(_item(f"{row}|{frame.name}|{i}", text=text, row=row, frame=frame.name, **kw))
        return out

    return loader


def test_build_row_real_body_emits_measured_and_inert_disclosures(monkeypatch):
    row = "casualness"  # benign judged row, wrapper-band strata
    shared = "identical benign prompt about tea"

    # (a) MEASURED: a free-text extraction node shares exact text with a frame
    # item -> the overlap is real and n_barred_superfamilies is nonzero.
    monkeypatch.setattr(F, "load_frame_prompts", _fake_frame_loader(shared_text=shared))
    monkeypatch.setattr(
        F,
        "load_extraction_items",
        lambda r: [
            _item(f"{r}|x0", text=shared, origin="extraction"),
            _item(f"{r}|x1", text="unrelated benign extraction question", origin="extraction"),
        ],
    )
    rr = F.build_row(row, eligible=frozenset({row}))
    assert rr["extraction_resolved"] is True
    assert rr["overlap_measurement"]["status"] == "measured"
    assert rr["counts"]["n_barred_superfamilies"] >= 1

    # (b) STRUCTURALLY INERT: id-only keyed frame nodes vs free-text extraction.
    monkeypatch.setattr(F, "load_frame_prompts", _fake_frame_loader(keyed="id-only"))
    monkeypatch.setattr(
        F,
        "load_extraction_items",
        lambda r: [_item(f"{r}|x0", text="free-text extraction question", origin="extraction")],
    )
    rr2 = F.build_row(row, eligible=frozenset({row}))
    assert rr2["overlap_measurement"]["status"] == "structurally-inert"
    assert rr2["counts"]["n_barred_superfamilies"] == 0

    # (c) COMPOSED keyed frames vs free-text extraction with a DUPLICATE stem
    # (group-G re-review: the old inert test used non-duplicate fixtures, which
    # is exactly why it kept passing while the disclosure went false): the M1
    # fix makes the overlap MEASURED and genuinely nonzero.
    dup_stem = "identical benign stem about tea?"
    monkeypatch.setattr(
        F, "load_frame_prompts", _fake_frame_loader(keyed=True, keyed_shared_stem=dup_stem)
    )
    monkeypatch.setattr(
        F,
        "load_extraction_items",
        lambda r: [_item(f"{r}|x0", text=dup_stem, origin="extraction")],
    )
    rr3 = F.build_row(row, eligible=frozenset({row}))
    assert rr3["overlap_measurement"]["status"] == "measured"
    assert rr3["overlap_measurement"]["spanning_edge_domains"] == ["exact-text"]
    assert rr3["counts"]["n_barred_superfamilies"] >= 1


# ---------------------------------------------------------------------------
# Freeze-time refusal on an unresolved extraction corpus (fail-loud, exit != 0)
# ---------------------------------------------------------------------------
def test_run_build_refuses_to_freeze_on_unresolved_extraction_corpus(tmp_path, monkeypatch):
    eligible, _ = F.load_eligibility()
    bad_row = sorted(eligible)[0]

    def fake_extraction(row):
        if row == bad_row:
            raise F.ExtractionCorpusUnresolvedError(f"{row}: synthetic unresolved corpus")
        return [_item(f"{row}|x0", text=f"benign extraction q for {row}", origin="extraction")]

    monkeypatch.setattr(F, "load_frame_prompts", _fake_frame_loader())
    monkeypatch.setattr(F, "load_extraction_items", fake_extraction)
    out_frame, out_split = tmp_path / "frame.json", tmp_path / "split.json"
    with pytest.raises(F.ExtractionCorpusUnresolvedError, match=bad_row):
        F.run_build(out_frame, out_split)
    # the refusal wrote NOTHING — no un-excluded families frozen into TEST
    assert not out_frame.exists()
    assert not out_split.exists()


def test_run_build_freezes_when_every_corpus_resolves(tmp_path, monkeypatch):
    monkeypatch.setattr(F, "load_frame_prompts", _fake_frame_loader())
    monkeypatch.setattr(
        F,
        "load_extraction_items",
        lambda row: [
            _item(f"{row}|x0", text=f"benign extraction q for {row}", origin="extraction")
        ],
    )
    out_frame, out_split = tmp_path / "frame.json", tmp_path / "split.json"
    frame_body, split_body, unresolved = F.run_build(out_frame, out_split)
    assert unresolved == []
    assert out_frame.exists() and out_split.exists()
    for body in (frame_body, split_body):
        F.validate_manifest(body)
        F.assert_manifest_immutable(body)
    closed_vocab = {"measured", "structurally-inert", "no-extraction-items"}
    for r in split_body["rows"]:
        m = r["extraction_overlap"]["measurement"]
        assert m["status"] in closed_vocab
        if m["status"] != "measured":
            assert r["extraction_overlap"]["n_barred_superfamilies"] == 0


# ---------------------------------------------------------------------------
# Manifest immutability + strict validation
# ---------------------------------------------------------------------------
def _minimal_body(kind="eligible_frame"):
    # unit 10: an eligible_frame body carries the per-row prospective fields
    # (explicit EMPTY form) + a reconciling ledger; validate_manifest raises
    # on their absence for that kind.
    rows = [
        {
            "row": r,
            "counts": {"n": i},
            "n_cells": 0,
            "n_cells_estimable": 0,
            "prospective_not_estimable": [],
        }
        for i, r in enumerate(C.ROW_IDS)
    ]
    body = {
        "manifest_version": C.MANIFEST_VERSION,
        "manifest_kind": kind,
        "issue": 2658,
        "metadata": {"note": "volatile — excluded from content sha"},
        "frozen_config": {"layer": C.LAYER},
        "superfamily_criteria": F.SUPERFAMILY_CRITERIA,
        "rows": rows,
    }
    if kind == "eligible_frame":
        body["prospective_not_estimable_ledger"] = F._not_estimable_ledger(rows)
    addressable = {k: v for k, v in body.items() if k != "metadata"}
    body["content_sha256"] = F._canonical_sha(addressable)
    body["cache_key"] = F._canonical_sha(["cache", kind])
    return body


def test_valid_minimal_manifest_passes_validate_and_immutable():
    body = _minimal_body()
    F.validate_manifest(body)
    F.assert_manifest_immutable(body)  # no raise


def test_manifest_immutability_detects_content_drift():
    body = _minimal_body()
    body["rows"][0]["counts"]["n"] = 999  # tamper AFTER freezing the sha
    with pytest.raises(F.FrameManifestError):
        F.assert_manifest_immutable(body)


def test_validate_manifest_rejects_unknown_field():
    body = _minimal_body()
    body["surprise"] = 1
    with pytest.raises(F.FrameManifestError):
        F.validate_manifest(body)


def test_validate_manifest_rejects_missing_row():
    body = _minimal_body()
    body["rows"] = body["rows"][:-1]  # drop harmful-compliance / last row
    body["content_sha256"] = F._canonical_sha({k: v for k, v in body.items() if k != "metadata"})
    with pytest.raises(F.FrameManifestError):
        F.validate_manifest(body)


def test_metadata_change_does_not_change_content_sha():
    body = _minimal_body()
    before = body["content_sha256"]
    body["metadata"]["note"] = "different volatile metadata"
    F.assert_manifest_immutable(body)  # content sha excludes metadata
    assert body["content_sha256"] == before


# ---------------------------------------------------------------------------
# Integration: the emitted manifests on disk (skip in a build-less checkout)
# ---------------------------------------------------------------------------
def _load_disk(path):
    if not path.exists():
        pytest.skip(f"{path.name} not built in this checkout")
    return json.loads(path.read_text())


def test_disk_manifests_validate_and_are_immutable():
    for path in (F.FRAME_MANIFEST_PATH, F.SPLIT_MANIFEST_PATH):
        body = _load_disk(path)
        F.validate_manifest(body)
        F.assert_manifest_immutable(body)


def test_disk_manifests_carry_no_raw_prompt_text():
    banned_keys = {"text", "prompt", "question", "content"}

    def walk(o):
        if isinstance(o, dict):
            for k, v in o.items():
                assert k not in banned_keys, f"raw-text key {k!r} leaked into manifest"
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)

    for path in (F.FRAME_MANIFEST_PATH, F.SPLIT_MANIFEST_PATH):
        walk(_load_disk(path))


def test_disk_split_manifest_dev_test_disjoint_per_row():
    body = _load_disk(F.SPLIT_MANIFEST_PATH)
    for r in body["rows"]:
        splits = r["superfamily_splits"]
        dev = {s for s, sp in splits.items() if sp == "dev"}
        test = {s for s, sp in splits.items() if sp == "test"}
        C.assert_split_lineage_disjoint(dev, test)
        # barred superfamilies never appear in test
        assert not (set(r["barred_superfamilies"]) & test), r["row"]


def test_disk_split_manifest_overlap_measurement_disclosure():
    body = _load_disk(F.SPLIT_MANIFEST_PATH)
    by_row = {r["row"]: r for r in body["rows"]}
    statuses = {row: r["extraction_overlap"]["measurement"]["status"] for row, r in by_row.items()}
    # sycophancy: the M1 fix admits keyed stems into the exact-text edge, so
    # the keyed-frames-vs-free-text-extraction overlap is now MEASURED (its
    # realized value stays a measurement, zero included) — no longer inert.
    assert statuses["sycophancy"] == "measured"
    m_syco = by_row["sycophancy"]["extraction_overlap"]["measurement"]
    assert "exact-text" in m_syco["spanning_edge_domains"]
    # harmful_compliance carries no extraction corpus at all
    assert statuses["harmful_compliance"] == "no-extraction-items"
    # every other row's overlap is genuinely MEASURED (zero included)
    for row, st in statuses.items():
        if row != "harmful_compliance":
            assert st == "measured", (row, st)
    # a structurally-inert disclosure can never claim a spanning edge domain
    for r in by_row.values():
        m = r["extraction_overlap"]["measurement"]
        if m["status"] == "structurally-inert":
            assert m["spanning_edge_domains"] == []
