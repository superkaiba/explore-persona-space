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


# ---------------------------------------------------------------------------
# Overlap-measurement disclosure (measured-zero vs structurally-inert zero)
# ---------------------------------------------------------------------------
def test_overlap_measurement_structurally_inert_keyed_frames_freetext_extraction():
    # keyed/composed frame items vs free-text extraction nodes: no edge
    # criterion can span the populations, so zero overlap is BY CONSTRUCTION.
    frame = [_item(f"fr{i}", text=f"composed assertion {i}", problem_id=f"k{i}") for i in range(3)]
    extr = [_item(f"ex{i}", text=f"free question {i}", origin="extraction") for i in range(2)]
    m = F._overlap_measurement("sycophancy", frame, extr, set())
    assert m["status"] == "structurally-inert"
    assert m["frame_edge_domains"] == ["keyed"]
    assert m["extraction_edge_domains"] == ["free-text"]
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
    assert m["status"] == "measured" and m["spanning_edge_domains"] == ["free-text"]
    # keyed x keyed (correctness-style id-only nodes): edge 1 can span
    mk = F._overlap_measurement(
        "correctness_math",
        [_item("frk", problem_id="gk-1")],
        [_item("exk", problem_id="gk-9", origin="extraction")],
        set(),
    )
    assert mk["status"] == "measured" and mk["spanning_edge_domains"] == ["keyed"]


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


def _fake_frame_loader(shared_text=None, keyed=False):
    """Boundary fake for load_frame_prompts (signature-conformant: (row, frame))."""

    def loader(row, frame):
        out = []
        for i in range(3):
            text = f"{frame.name} benign prompt {i} about {row}"
            if shared_text is not None and frame.name and i == 0:
                text = shared_text if frame == F.FRAMES[row].frames[0] else text
            kw = {}
            if keyed:
                kw = {"problem_id": f"{frame.name}-k{i}", "band_key": f"{frame.name}-k{i}"}
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

    # (b) STRUCTURALLY INERT: keyed frame items vs free-text extraction nodes.
    monkeypatch.setattr(F, "load_frame_prompts", _fake_frame_loader(keyed=True))
    monkeypatch.setattr(
        F,
        "load_extraction_items",
        lambda r: [_item(f"{r}|x0", text="free-text extraction question", origin="extraction")],
    )
    rr2 = F.build_row(row, eligible=frozenset({row}))
    assert rr2["overlap_measurement"]["status"] == "structurally-inert"
    assert rr2["counts"]["n_barred_superfamilies"] == 0


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
    # sycophancy: keyed/composed frame items vs free-text extraction nodes —
    # its zero overlap is BY CONSTRUCTION and must be disclosed as such.
    assert statuses["sycophancy"] == "structurally-inert"
    assert by_row["sycophancy"]["extraction_overlap"]["n_barred_superfamilies"] == 0
    # harmful_compliance carries no extraction corpus at all
    assert statuses["harmful_compliance"] == "no-extraction-items"
    # every other row's overlap is genuinely MEASURED (zero included)
    for row, st in statuses.items():
        if row not in ("sycophancy", "harmful_compliance"):
            assert st == "measured", (row, st)
    # a structurally-inert disclosure can never claim a spanning edge domain
    for r in by_row.values():
        m = r["extraction_overlap"]["measurement"]
        if m["status"] == "structurally-inert":
            assert m["spanning_edge_domains"] == []
