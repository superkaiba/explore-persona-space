"""Unit tests for the #2569 curve (H2b B4) + der-eval (leg-4 step 5 tail) wiring.

Covers the two production-path additions of the round's gap-closing unit:

- ``_corpus_tags_from_manifest_dir``: streaming manifest join, pass_b rows tagged
  ``pass_b`` (never ``lmsys``), fail-loud on an unjoined new row;
- ``GL.fit_point(extra_eval=...)``: pass-B comparability companions scored from
  the SAME fit pass — verdict ``test_r2`` byte-identical to the no-extra call,
  extra slice R2 equal to a direct refit scored on that slice;
- ``GL.curve_core``: splits -> refits (+theory) -> parity -> H2b verdict with
  ``extra_eval`` threaded to every point (in-memory form of the curve CLI);
- der-eval description loading (three accepted shapes, strict fail-loud), the
  #2552 artifact-presence probe, the rule-27 parse-contract round-trips for the
  matching + describe parsers, deterministic 10-way item construction with its
  coverage/eligibility data gates, and the judge-dispatch wrapper's
  drop-never-coerce / transport-re-drive / fail-loud-on-residual semantics
  (network boundary faked with signature-conformant fakes that EXECUTE the real
  parse contract; everything else is the real body).

All synthetic + CPU-fast (d <= 8); the dense 3584-dim fp64 factorizations stay
out of every test path (unit brief).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

import issue2569_gateladder as GL  # noqa: E402
import issue2569_rowbattery as RB  # noqa: E402

from explore_persona_space.llm import api_dispatch as AD  # noqa: E402


def _synth_store(n_lmsys: int = 2400, n_wc: int = 600, d: int = 16, seed: int = 5):
    """Synthetic assembled store (the test_issue2569_gateladder fixture verbatim:
    an interior-lambda-selecting shape under the default widened grid)."""
    rng = np.random.default_rng(11)
    eta = 1.0 / np.arange(1, d + 1) ** 1.5
    b = rng.standard_normal((d, d)) * 0.3
    r = np.random.default_rng(seed)
    n = n_lmsys + n_wc
    x = (r.standard_normal((n, d)) * np.sqrt(eta)).astype(np.float32)
    y = (x @ b + r.standard_normal((n, d))).astype(np.float32)
    corpus = np.array(["lmsys"] * n_lmsys + ["wildchat"] * n_wc)
    conv = np.arange(n) // 6
    return x, y, corpus, conv


# ── curve: corpus tags from the sampling manifest ─────────────────────────────────


def _write_manifest(tmp_path: Path, rows: list[dict], parts: int = 2) -> Path:
    mdir = tmp_path / "manifest"
    mdir.mkdir()
    chunks = np.array_split(np.arange(len(rows)), parts)
    for pi, ch in enumerate(chunks):
        (mdir / f"part_{pi:05d}.jsonl").write_text(
            "".join(json.dumps(rows[i]) + "\n" for i in ch), encoding="utf-8"
        )
    return mdir


def test_corpus_tags_pass_b_never_lmsys(tmp_path):
    """pass_b rows are tagged 'pass_b' (NOT the #1482 'lmsys' convention); new
    rows take the manifest corpus by the manifest's own key ``i``; response text
    is tolerated, never kept.

    Fixture rows use ``i`` because that is the REAL sampling-manifest schema
    (``{corpus, depth, i, messages, n_chars, source_hash, split, stream_pos}``);
    the ``ci`` spelling belongs to the DERIVED capture/store artifacts.
    """
    rows = [
        {"i": 0, "corpus": "lmsys", "response": "long text " * 50},
        {"i": 1, "corpus": "WildChat", "response": "x"},
        {"i": 2, "corpus": "lmsys"},
    ]
    mdir = _write_manifest(tmp_path, rows)
    row_ci = np.array([-1, -1, 0, 1, 2])  # 2 pass_b rows lead
    tags = RB._corpus_tags_from_manifest_dir(mdir, row_ci, n_pb=2)
    assert list(tags) == ["pass_b", "pass_b", "lmsys", "wildchat", "lmsys"]


def test_corpus_tags_fail_loud_on_unjoined_row(tmp_path):
    """An assembled new row whose ci is absent from the manifest raises."""
    mdir = _write_manifest(tmp_path, [{"i": 0, "corpus": "lmsys"}])
    row_ci = np.array([-1, 0, 7])  # ci=7 has no manifest row
    with pytest.raises(AssertionError, match="no manifest corpus tag"):
        RB._corpus_tags_from_manifest_dir(mdir, row_ci, n_pb=1)


def test_manifest_join_reads_i_not_ci(tmp_path):
    """The manifest join keys on ``i``, never on a ``ci`` field.

    Each fixture row carries BOTH keys with DIFFERENT values: ``i`` is the true
    conversation index and ``ci`` is a decoy pointing elsewhere. Reading the
    decoy would mis-tag every row, so this test fails on the exact defect it
    guards (a production ``KeyError``/mis-join when the reader keys on ``ci``).
    """
    rows = [
        {"i": 0, "ci": 1, "corpus": "lmsys", "response": "abcd"},
        {"i": 1, "ci": 0, "corpus": "WildChat", "response": "xy"},
    ]
    mdir = _write_manifest(tmp_path, rows, parts=1)
    row_ci = np.array([-1, 0, 1])
    tags = RB._corpus_tags_from_manifest_dir(mdir, row_ci, n_pb=1)
    assert list(tags) == ["pass_b", "lmsys", "wildchat"]

    (tmp_path / "flat").mkdir()
    (tmp_path / "flat" / "part_00000.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8"
    )
    out = RB._ans_len_from_manifest_dir(tmp_path / "flat", np.array([-1, 0, 1], np.int64), n_pb=1)
    np.testing.assert_array_equal(out, [-1, 4, 2])


# ── curve: fit_point extra_eval + curve_core ──────────────────────────────────────


def test_fit_point_extra_eval_matches_direct_refit():
    """extra_eval slices ride the SAME fit pass: verdict test_r2 is unchanged vs
    the no-extra call, and each extra R2 equals a direct call scored on that
    slice (same tr/val -> same selected lambda)."""
    x, y, _c, _v = _synth_store()
    tr, val, te = np.arange(800), np.arange(800, 1100), np.arange(1100, 1900)
    extra = {"passb_pinned_val": np.arange(1900, 2000), "passb_pinned_test": np.arange(2000, 2300)}
    dev = torch.device("cpu")
    base = GL.fit_point(x, y, tr, val, te, dev=dev)
    with_extra = GL.fit_point(x, y, tr, val, te, dev=dev, extra_eval=extra)
    assert with_extra["test_r2"] == pytest.approx(base["test_r2"], abs=1e-12)
    assert with_extra["selected_lambda"] == base["selected_lambda"]
    assert set(with_extra["extra_eval_r2"]) == set(extra)
    for name, idx in extra.items():
        direct = GL.fit_point(x, y, tr, val, idx, dev=dev)
        got = with_extra["extra_eval_r2"][name]
        assert got["n_rows"] == len(idx)
        assert got["r2"] == pytest.approx(direct["test_r2"], abs=1e-10)
    assert "extra_eval_r2" not in base  # no-extra output shape unchanged


def test_curve_core_threads_extra_eval_and_verdict():
    """curve_core: verdict points carry theory + extra_eval_r2; parity passes;
    the H2b statistic and the extra_eval_slices record are present."""
    x, y, corpus, conv = _synth_store()
    extra = {"passb_pinned_val": np.arange(2400, 2500), "passb_pinned_test": np.arange(2500, 2700)}
    doc = GL.curve_core(
        x,
        y,
        corpus,
        conv,
        n_grid=(200, 400),
        eval_rows=300,
        val_rows=200,
        dev=torch.device("cpu"),
        layer=19,
        skip_companions=True,
        smoke=True,
        extra_eval=extra,
    )
    assert [p["n_train"] for p in doc["verdict_points"]] == [200, 400]
    assert all(pp["pass"] for pp in doc["parity_check"]["per_point"])
    assert doc["h2b"]["mean_abs_dr2"] is not None
    for p in doc["verdict_points"]:
        assert p["lambda_grid_edge"] is None
        assert set(p["extra_eval_r2"]) == set(extra)
        assert np.isfinite(p["theory"]["predicted_r2"])
    assert doc["extra_eval_slices"]["passb_pinned_test"]["n_rows"] == 200
    assert doc["regime"]["n_grid"] == [200, 400]


# ── der: description loading + the #2552 probe ────────────────────────────────────


def test_load_descriptions_three_shapes(tmp_path):
    """Dict / record-list (+ wrapper) / JSONL all parse; junk fails loud."""
    d1 = tmp_path / "d1.json"
    d1.write_text(json.dumps({"3": "cooking verbs", "7": "python tracebacks"}))
    assert RB._load_descriptions(d1) == {3: "cooking verbs", 7: "python tracebacks"}
    d2 = tmp_path / "d2.json"
    d2.write_text(json.dumps([{"feat_id": 4, "label": "greetings"}, {"id": 9, "text": "dates"}]))
    assert RB._load_descriptions(d2) == {4: "greetings", 9: "dates"}
    d3 = tmp_path / "d3.jsonl"
    d3.write_text('{"feature_id": 1, "description": "code"}\n{"feature": 2, "explanation": "x"}\n')
    assert RB._load_descriptions(d3) == {1: "code", 2: "x"}
    d4 = tmp_path / "d4.json"
    d4.write_text(json.dumps({"descriptions": {"11": "lists"}}))
    assert RB._load_descriptions(d4) == {11: "lists"}
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps([{"no_id_key": 1, "blah": "y"}]))
    with pytest.raises(ValueError, match="missing id/text"):
        RB._load_descriptions(bad)
    empty = tmp_path / "empty.json"
    empty.write_text(json.dumps({"1": "   "}))
    with pytest.raises(ValueError, match="no usable descriptions"):
        RB._load_descriptions(empty)


def test_find_i2552_probe(tmp_path):
    """The artifact-presence probe returns the first PARSEABLE candidate, logs
    and skips junk, and returns None on an empty tree (-> own-round fallback)."""
    assert RB._find_i2552_descriptions(tmp_path) is None
    base = tmp_path / "eval_results" / "issue_2552"
    base.mkdir(parents=True)
    junk = base / "a_descriptions.json"
    junk.write_text("not json at all")
    good = base / "b_descriptions.json"
    good.write_text(json.dumps({"5": "typos"}))
    got = RB._find_i2552_descriptions(tmp_path)
    assert got == good  # junk candidate rejected loudly, parseable one returned


# ── der: parse contracts (rule-27 round-trips) ────────────────────────────────────


def test_parse_match_answer_round_trip():
    """Well-formed replies parse to the 0-based index; malformed / out-of-range
    raise (-> DROPPED, never coerced)."""
    assert RB._parse_match_answer('{"answer": "A"}', 10) == 0
    assert RB._parse_match_answer('noise before {"answer": "j"} after', 10) == 9
    with pytest.raises(ValueError, match="out of range"):
        RB._parse_match_answer('{"answer": "K"}', 10)
    with pytest.raises(ValueError, match=r"no .*answer"):
        RB._parse_match_answer("I refuse to pick a letter.", 10)


def test_parse_description_bounds():
    assert RB._parse_description("  Cooking\nverbs.  ") == "Cooking verbs."
    with pytest.raises(ValueError, match="length"):
        RB._parse_description("x")
    with pytest.raises(ValueError, match="length"):
        RB._parse_description("y" * 700)


def test_der_prompts_are_description_only():
    """Packets carry descriptions + neutral instructions only (rule-28 scoping):
    no file paths, no issue ids, no corpus names."""
    p = RB._der_match_prompt(["a desc"], [["c1"], ["c2"]])
    for banned in ("2569", "2552", "lmsys", "eval_results", ".json", "issue"):
        assert banned not in p.lower()
    assert '{"answer": "<letter>"}' in p
    e = RB._der_describe_prompt(["some evidence " * 200])  # cap applied
    assert len(e) < 2 * RB.DER_EVIDENCE_STR_CAP


# ── der: matching-item construction ───────────────────────────────────────────────


def _der_fixture(n_rows: int = 60, n_feat: int = 20, seed: int = 1):
    rng = np.random.default_rng(seed)
    feat_ids = np.arange(100, 100 + n_feat, dtype=np.int64)
    pred = rng.normal(size=(n_rows, n_feat)).astype(np.float32)
    true = np.where(rng.random((n_rows, n_feat)) < 0.4, rng.gamma(2, size=(n_rows, n_feat)), 0.0)
    desc = {int(f): f"feature {int(f)} description" for f in feat_ids[: n_feat - 4]}
    return pred, true.astype(np.float32), feat_ids, desc


def test_build_matching_items_deterministic_and_shaped():
    pred, true, feat_ids, desc = _der_fixture()
    kw = dict(n_items=12, feats_per_list=4, n_way=5)
    items1, stats1 = RB._build_matching_items(
        pred, true, feat_ids, desc, rng=np.random.default_rng(7), **kw
    )
    items2, _ = RB._build_matching_items(
        pred, true, feat_ids, desc, rng=np.random.default_rng(7), **kw
    )
    assert items1 == items2  # deterministic packets under the seeded rng
    assert len(items1) == 12
    described = set(desc.values())
    for it in items1:
        assert len(it["candidates"]) == 5
        assert 0 <= it["answer_pos"] < 5
        assert set(it["target"]) <= described
        # the answer candidate is the row's own TRUE list
        assert it["candidates"][it["answer_pos"]]
    assert 0.0 < stats1["union_coverage"] < 1.0
    assert stats1["n_described_in_union"] == len(desc)
    assert 0.0 <= stats1["pred_topk_described_frac"] <= 1.0


def test_build_matching_items_data_gates_fire():
    """Degenerate inputs trip the DESIGNED gates: coverage too low; too few
    eligible rows for items + distractors."""
    pred, true, feat_ids, desc = _der_fixture()
    with pytest.raises(ValueError, match="have descriptions"):
        RB._build_matching_items(
            pred,
            true,
            feat_ids,
            {101: "only one"},
            n_items=4,
            feats_per_list=4,
            n_way=5,
            rng=np.random.default_rng(0),
        )
    with pytest.raises(ValueError, match="eligible holdout rows"):
        RB._build_matching_items(
            pred,
            true,
            feat_ids,
            desc,
            n_items=200,
            feats_per_list=4,
            n_way=5,
            rng=np.random.default_rng(0),
        )


def test_der_candidate_features_priority_deterministic():
    pred = np.zeros((10, 4), np.float32)
    pred[:, 2] = 1.0  # feature col 2 always tops predictions
    true = np.zeros((10, 4), np.float32)
    true[:, 1] = 1.0  # feature col 1 always fires
    order = RB._der_candidate_features(pred, true, k=1)
    assert set(order[:2]) == {1, 2}
    assert list(order[2:]) == [0, 3]  # ties broken by column index (stable)


# ── der: judge dispatch wrapper (network boundary faked, real body) ──────────────


def _fake_dispatch(script: dict[str, list[str]]):
    """Signature-conformant fake of api_dispatch.dispatch_calls: consumes the
    per-item response SCRIPT (one canned model text per call round), EXECUTES
    the caller's real build_request + parse_response, and mints real
    DispatchResult rows exactly like the sync path (parse raise -> error row)."""
    calls = {"rounds": 0}

    async def fake(items, *, model, build_request, parse_response, force_path, cache_dir):
        assert force_path == "sync" and model
        calls["rounds"] += 1
        out = {}
        for it in items:
            req = build_request(it)
            assert req["model"] == model and req["max_tokens"] >= 1024
            assert req["messages"][0]["content"] == it.payload
            queue = script[it.item_id]
            text = queue.pop(0) if queue else "__TRANSPORT__"
            if text == "__TRANSPORT__":
                out[it.item_id] = AD.DispatchResult(
                    item_id=it.item_id,
                    error=True,
                    reason="transport",
                    category=AD.RESULT_TRANSPORT,
                )
                continue
            try:
                out[it.item_id] = AD.DispatchResult(item_id=it.item_id, result=parse_response(text))
            except Exception as e:  # the dispatcher's per-item parse-catch shape
                out[it.item_id] = AD.DispatchResult(
                    item_id=it.item_id, error=True, reason=str(e), category=AD.RESULT_ERROR
                )
        return out

    return fake, calls


def test_dispatch_judge_drop_never_coerce_and_redrive(tmp_path):
    """ok rows parse through the REAL parser; a malformed row lands error=True
    (drop, counted by the caller); a transport row is RE-DRIVEN and heals."""
    script = {
        "a": ['{"answer": "B"}'],
        "b": ["no letter here"],  # parse raises -> RESULT_ERROR (drop)
        "c": ["__TRANSPORT__", '{"answer": "A"}'],  # heals on re-drive round 2
    }
    fake, calls = _fake_dispatch(script)
    res = RB._dispatch_judge(
        {k: f"prompt {k}" for k in script},
        model="claude-sonnet-4-5-20250929",
        max_tokens=RB.DER_MAX_TOKENS,
        cache_dir=tmp_path / "cache",
        parse=lambda t: RB._parse_match_answer(t, 10),
        dispatch_fn=fake,
    )
    assert calls["rounds"] == 2  # one re-drive round for the transport row
    assert res["a"].result == 1 and not res["a"].error
    assert res["b"].error and res["b"].category == AD.RESULT_ERROR
    assert res["c"].result == 0 and not res["c"].error


def test_dispatch_judge_residual_transport_raises(tmp_path):
    """A row still transport-failed after every re-drive RAISES — transport
    failures are never persisted as drops."""
    fake, _ = _fake_dispatch({"a": ["__TRANSPORT__"] * 10})
    with pytest.raises(RuntimeError, match="transport-failed"):
        RB._dispatch_judge(
            {"a": "prompt"},
            model="m",
            max_tokens=1024,
            cache_dir=tmp_path / "cache",
            parse=lambda t: t,
            dispatch_fn=fake,
            max_redrives=2,
        )


def test_generate_descriptions_budget_and_drops(tmp_path):
    """Describe stage: capped at max_calls, no-evidence features counted (never
    guessed), malformed replies dropped with counts."""
    evidence = {100: ["ev a"], 101: ["ev b"], 102: ["ev c"]}  # 103 has none
    script = {
        "desc_100": ["A clean description of the feature."],
        "desc_101": ["x"],  # too short -> parse raises -> dropped
    }
    fake, _ = _fake_dispatch(script)
    desc, stats = RB._generate_descriptions(
        [100, 101, 102, 103],
        evidence,
        model="m",
        cache_dir=tmp_path / "cache",
        max_calls=2,  # cap excludes 102 despite evidence
        dispatch_fn=fake,
    )
    assert desc == {100: "A clean description of the feature."}
    assert stats["n_dispatched"] == 2 and stats["n_no_evidence"] == 1
    assert stats["dropped_by_category"] == {AD.RESULT_ERROR: 1}
    assert stats["n_described"] == 1
    # zero budget: nothing dispatched, nothing described
    desc0, stats0 = RB._generate_descriptions(
        [100], evidence, model="m", cache_dir=tmp_path / "c2", max_calls=0, dispatch_fn=fake
    )
    assert desc0 == {} and stats0["n_dispatched"] == 0
