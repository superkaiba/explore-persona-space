"""Issue #1335 ladder pins.

(1) Rung registry invariants: 11 rungs, uniform base-prime flag across the
    fiction-render base cells (plan §4.2 pin), deterministic fingerprints +
    the c24 resume-match predicate.
(2) Fiction-render BYTE PARITY with the #1310 v3 recipe (header / slot-0 body /
    canned foil turn / advance / full prefill prompt) at battery foils — the
    r7 endpoint-parity contract — plus the foil-free r6 shapes.
(3) Q&A renders (plain text, both models) + offset-mapping prefix-token
    counting on the REAL Qwen tokenizer (BPE-merge boundary case).
(4) tf re-renders execute the REAL production bodies: r2_tf relabels the r1
    rows; s1/s2 rebuild the r7 prefill prompts under a label override with the
    stored completion ids copied verbatim (the internal no-override
    reconstruction assert is on the executed path).
(5) matched_subsample is NOT seed-degenerate on singleton-group cells (the
    #931 group_stratified tie-break trap).
(6) build_items span construction + drop counters (r0 extras, prefix fallback,
    row-length cap); the r6 seed-attributable wiring-check skip (skip vs
    fail-loud vs run) + its non-binding gate2 record.
(7) build_ladder_summary: oriented deltas, the SIX-delta family EXCLUDES the
    length delta, Wren-matched companion present, joint-draw D CI, verdict
    lattice, binding gates — on a synthetic eval_results fixture (tmp_path).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "src"))

r1335 = pytest.importorskip("issue1335_render_rungs")
c1310 = pytest.importorskip("issue1310_common")
i1310_prefill = pytest.importorskip("issue1310_prefill")
common931 = pytest.importorskip("issue931_common")


@pytest.fixture(scope="module")
def tokenizer():
    return common931.get_tokenizer(r1335.MODEL_IDS["base"])


# ---------------------------------------------------------------------------
# (1) registry + fingerprints
# ---------------------------------------------------------------------------


def test_registry_invariants():
    assert len(r1335.RUNGS) == 11
    assert set(r1335.TF_RUNGS) == {"r2_tf", "s1_assistant_label", "s2a_familiar", "s2b_novel"}
    # base-prime flag identical (and absent — the v3 prefill recipe) across
    # every fiction-render base cell incl. r4 (plan §4.2 pin).
    assert r1335.assert_base_prime_uniform() is False
    assert set(r1335.FICTION_RENDER_RUNGS) == {
        "r4_fictionframe",
        "r6_nofoil",
        "r7_endpoint",
        "s1_assistant_label",
        "s2a_familiar",
        "s2b_novel",
    }


def test_fingerprints_deterministic_and_match_predicate():
    h = r1335.render_config_hash("r7_endpoint")
    assert h == r1335.render_config_hash("r7_endpoint") and len(h) == 16
    assert r1335.render_config_hash("r6_nofoil") != h
    fp = r1335.fingerprint("r7_endpoint")
    assert set(fp) == {"rung_slug", "render_config_hash", "code_sha"}
    assert r1335.fingerprint_matches(dict(fp), "r7_endpoint")
    stale = dict(fp)
    stale["render_config_hash"] = "0" * 16
    assert not r1335.fingerprint_matches(stale, "r7_endpoint")
    assert not r1335.fingerprint_matches(dict(fp), "r6_nofoil")


# ---------------------------------------------------------------------------
# (2) fiction render byte parity with #1310 (the r7 endpoint contract)
# ---------------------------------------------------------------------------


def test_fiction_render_byte_parity_with_1310():
    battery = c1310.build_scenario_battery()
    for sc in battery[:3]:
        sc_id = sc["scenario_id"]
        foils = c1310.foils_for_scene(sc_id)
        assert r1335.foils_for_rung("r7_endpoint", sc_id) == foils
        assert r1335.foils_for_rung("s1_assistant_label", sc_id) == foils
        assert r1335.foils_for_rung("r6_nofoil", sc_id) == []
        for persona in c1310.PERSONA_LABELS:
            for mk in ("base", "instruct"):
                assert r1335.fiction_header_text(
                    sc, persona, c1310.PERSONAS[persona], foils, mk
                ) == c1310.prefill_header(sc, persona, mk), (sc_id, persona, mk)
        assert r1335.fiction_body_slot0(sc_id, foils) == c1310.prefill_body_slot0(sc_id)
        for slot in range(4):
            assert r1335.canned_foil_turn(sc_id, slot, foils) == c1310.canned_foil_turn(sc_id, slot)
        body = c1310.prefill_body_slot0(sc_id)
        comp = " a generated line of dialogue"
        assert r1335.fiction_advance_body(
            body, "Vex", comp, sc_id, 1, foils
        ) == c1310.prefill_advance_body(body, "Vex", comp, sc_id, 1)
        # full base prefill prompt == issue1310_prefill.build_prefix (no tokenizer
        # on the base path).
        assert r1335.fiction_prefix(None, sc, "Vex", "base", body, foils) == (
            i1310_prefill.build_prefix(None, sc, "Vex", "base", body)
        )


def test_fiction_prefix_instruct_parity_and_override(tokenizer):
    sc = c1310.build_scenario_battery()[0]
    sc_id = sc["scenario_id"]
    foils = c1310.foils_for_scene(sc_id)
    body = c1310.prefill_body_slot0(sc_id)
    mine = r1335.fiction_prefix(tokenizer, sc, "Wren", "instruct", body, foils)
    theirs = i1310_prefill.build_prefix(tokenizer, sc, "Wren", "instruct", body)
    assert mine == theirs
    # Label override renames the responder EVERYWHERE the renderer used it
    # (header persona clause + cue); foil names + the persona's desc stay.
    ovr = r1335.fiction_prefix(
        tokenizer, sc, "Wren", "instruct", body, foils, label_override="Assistant"
    )
    assert ovr.endswith("Assistant:") and "- Assistant: " in ovr
    assert c1310.PERSONAS["Wren"] in ovr  # the persona DESCRIPTION is unchanged


def test_r6_foilfree_shapes():
    sc = c1310.build_scenario_battery()[0]
    assert r1335.fiction_body_slot0(sc["scenario_id"], []) == ""
    h = r1335.fiction_header_text(sc, "Dana", c1310.PERSONAS["Dana"], [], "base")
    assert "Also present" not in h
    body = r1335.fiction_advance_body("", "Dana", " a line", sc["scenario_id"], 1, [])
    assert body == "Dana: a line\n"
    prompt = r1335.fiction_prefix(None, sc, "Dana", "base", body, [])
    assert prompt.endswith("Dana: a line\nDana:")
    # prefix text for the v_P arm: everything before the cue (r6 has no foil line)
    pre = r1335.fiction_prefix_text(prompt, sc["scenario_id"], 1, [], "Dana")
    assert pre == prompt[: -len("Dana:")]


# ---------------------------------------------------------------------------
# (3) Q&A renders + prefix-token counting (real tokenizer)
# ---------------------------------------------------------------------------


def test_qa_renders():
    p, pre = r1335.qa_render("r0_qa_full", "What is 2+2?")
    assert p == "User: What is 2+2?\nAssistant:" and pre == ""
    p, pre = r1335.qa_render("r1_qa_oneline", "q")
    assert p == "User: q\nAssistant:"
    p, pre = r1335.qa_render("r2_tf", "q")
    assert p == "User: q\nWren:"
    p, pre = r1335.qa_render("r3_persona", "q")
    assert p == f"Wren is {c1310.PERSONAS['Wren']}.\n\nUser: q\nWren:"
    assert pre == f"Wren is {c1310.PERSONAS['Wren']}.\n\n"
    sc = c1310.build_scenario_battery()[0]
    p, pre = r1335.qa_render("r4_fictionframe", "q", sc)
    assert p.endswith("Sam: q\nWren:") and "Also present: Sam." in pre
    assert p.startswith(pre)


def test_count_prefix_tokens_offset_based(tokenizer):
    # BPE-merge boundary case: the prefix ends ".\n\n" (merge-prone seam); the
    # count is offset-based (tokens ENDING inside the prefix), never a
    # separate-tokenization sum.
    prefix = f"Wren is {c1310.PERSONAS['Wren']}.\n\n"
    prompt = prefix + "User: What is the capital of France?\nWren:"
    ids = list(tokenizer(prompt, add_special_tokens=False)["input_ids"])
    n_prefix = r1335.count_prefix_tokens(tokenizer, prompt, prefix, ids)
    assert 0 < n_prefix < len(ids)
    # every counted token ends within the prefix
    enc = tokenizer(prompt, add_special_tokens=False, return_offsets_mapping=True)
    ends_inside = [e for s, e in enc["offset_mapping"] if e <= len(prefix) and e > s]
    assert n_prefix == len(ends_inside)
    # drift assert fires loud on mismatched ids
    with pytest.raises(AssertionError, match="tokenizer drift"):
        r1335.count_prefix_tokens(tokenizer, prompt, prefix, ids[:-1])
    # empty prefix -> 0 (the capture-side degenerate-arm fallback handles it)
    assert r1335.count_prefix_tokens(tokenizer, prompt, "", ids) == 0


# ---------------------------------------------------------------------------
# (4) tf re-renders (REAL bodies; stub only at the engine boundary)
# ---------------------------------------------------------------------------


def _stub_fiction_gen(tmp_path: Path, tokenizer, slug: str, model_kind: str) -> Path:
    """Run the REAL gen_fiction body with --stub-gen (real tokenizer ids)."""
    import issue1335_gen as g1335

    args = SimpleNamespace(
        rung=slug,
        model=model_kind,
        data_dir=tmp_path,
        n_questions=0,
        n_scenarios=2,
        slots=2,
        stub_gen=True,
        skip_upload=True,
        gpu_memory_utilization=0.85,
    )
    records, _meta = g1335.gen_fiction(args, tokenizer, r1335.fingerprint(slug))
    dest = r1335.gen_path(tmp_path, slug, model_kind)
    r1335.write_gen_jsonl(dest, records)
    return dest


def test_tf_rerender_s1_label_override(tmp_path, tokenizer):
    _stub_fiction_gen(tmp_path, tokenizer, "r7_endpoint", "base")
    src = r1335._read_jsonl(r1335.gen_path(tmp_path, "r7_endpoint", "base"))
    dest = r1335.tf_rerender("s1_assistant_label", "base", tmp_path)
    out = r1335._read_jsonl(dest)
    assert len(out) == len(src)
    by_src = {r["tf_source_row_id"]: r for r in out}
    for s in src:
        o = by_src[s["row_id"]]
        assert o["rung"] == "s1_assistant_label"
        assert o["completion_token_ids"] == s["completion_token_ids"]  # copied verbatim
        assert o["completion"] == s["completion"]
        assert o["prompt"].endswith("Assistant:")
        assert "Assistant is " in o["prompt"]  # header persona clause relabeled
        assert s["prompt"].endswith(f"{s['persona']}:")
        assert o["provenance"] == "tf-rerender"
        assert 0 < o["n_prefix_tokens"] < o["n_prompt_tokens"]


def test_tf_rerender_s2_names_token_length(tmp_path, tokenizer):
    for name in (r1335.FAMILIAR_NAME, r1335.NOVEL_NAME):
        n = len(tokenizer(name, add_special_tokens=False)["input_ids"])
        assert 1 <= n <= 2, (name, n)
    _stub_fiction_gen(tmp_path, tokenizer, "r7_endpoint", "base")
    for slug, name in (("s2a_familiar", "Sarah"), ("s2b_novel", "Xelor")):
        out = r1335._read_jsonl(r1335.tf_rerender(slug, "base", tmp_path))
        assert all(r["prompt"].endswith(f"{name}:") for r in out)


def test_tf_rerender_r2tf_relabels_r1(tmp_path, tokenizer):
    # Handcrafted r1 rows through the REAL qa_render + real tokenizer ids.
    rows = []
    for i, q in enumerate(["What is 2+2?", "Name a color.", "Why is the sky blue?"]):
        prompt, _pre = r1335.qa_render("r1_qa_oneline", q)
        pids = list(tokenizer(prompt, add_special_tokens=False)["input_ids"])
        comp = " It is four, plainly."
        cids = list(tokenizer(comp, add_special_tokens=False)["input_ids"])
        rows.append(
            {
                "rung": "r1_qa_oneline",
                "model_kind": "base",
                "row_id": f"r1_qa_oneline:q{i:05d}",
                "group_id": f"q{i:05d}",
                "persona": "Assistant",
                "slot": 0,
                "question": q,
                "prompt": prompt,
                "completion": comp,
                "prompt_token_ids": pids,
                "completion_token_ids": cids,
                "n_prompt_tokens": len(pids),
                "n_completion_tokens": len(cids),
                "n_prefix_tokens": 0,
            }
        )
    r1335.write_gen_jsonl(r1335.gen_path(tmp_path, "r1_qa_oneline", "base"), rows)
    out = r1335._read_jsonl(r1335.tf_rerender("r2_tf", "base", tmp_path))
    assert len(out) == 3
    for s, o in zip(rows, out, strict=True):
        assert o["prompt"] == s["prompt"].replace("\nAssistant:", "\nWren:")
        assert o["completion_token_ids"] == s["completion_token_ids"]
        assert o["row_id"].startswith("r2_tf:")


# ---------------------------------------------------------------------------
# (5) matched-n subsample (singleton-group degeneracy guard)
# ---------------------------------------------------------------------------


def test_matched_subsample_not_seed_degenerate_on_singletons():
    import issue1335_fit as f1335

    groups = np.asarray([f"q{i:05d}" for i in range(200)])
    a = f1335.matched_subsample(groups, 50, seed=931)
    b = f1335.matched_subsample(groups, 50, seed=932)
    assert len(a) == len(b) == 50
    assert not np.array_equal(a, b), "singleton-group subsample is seed-degenerate"
    # grouped case stays group-stratified + deterministic per seed
    g2 = np.asarray(["s0"] * 40 + ["s1"] * 40 + ["s2"] * 20)
    c = f1335.matched_subsample(g2, 50, seed=931)
    assert np.array_equal(c, f1335.matched_subsample(g2, 50, seed=931))
    _u, counts = np.unique(g2[c], return_counts=True)
    assert counts.sum() == 50 and counts.min() >= 8  # roughly proportional


# ---------------------------------------------------------------------------
# (6) capture item construction
# ---------------------------------------------------------------------------


def test_build_items_spans_and_drops():
    import issue1335_extract_store as e1335

    def rec(row_id, n_prompt, n_comp, n_prefix, group="g0", persona="Assistant"):
        return {
            "row_id": row_id,
            "group_id": group,
            "persona": persona,
            "slot": 0,
            "prompt_token_ids": list(range(n_prompt)),
            "completion_token_ids": list(range(n_prompt, n_prompt + n_comp)),
            "n_prefix_tokens": n_prefix,
        }

    records = [
        rec("a", 30, 12, 0),  # kept; degenerate prefix -> (0,1) fallback
        rec("b", 30, 2, 0),  # dropped: short dialogue
        rec("c", 4, 12, 0),  # dropped: short context
        rec("d", 2040, 12, 10),  # dropped: row too long (> 2048)
        rec("e", 600, 12, 100),  # kept; context capped to the last 512
    ]
    items, counters = e1335.build_items("r0_qa_full", records)
    assert counters == {
        "records": 5,
        "kept": 2,
        "dropped_short_dialogue": 1,
        "dropped_short_context": 1,
        "dropped_row_too_long": 1,
        "prefix_fallback_first_token": 1,  # only KEPT rows reach the prefix branch
    }
    a, e = items[0], items[1]
    assert a["spans"]["x_spanmean"] == (0, 30)
    assert a["spans"]["x_prefixmean"] == (0, 1)  # degenerate-arm fallback
    assert a["spans"]["y"] == (30, 42)
    assert a["spans"]["y96"] == (30, 42)  # r0 extra (min(96, n_comp))
    assert a["spans"]["x_spanmean_nocap"] == (0, 30)
    assert e["spans"]["x_spanmean"] == (600 - 512, 600)  # 512-cap
    assert e["spans"]["x_prefixmean"] == (0, 100)
    # non-r0 rungs carry no extras
    items2, _ = e1335.build_items("r1_qa_oneline", [rec("a", 30, 12, 5)])
    assert "y96" not in items2[0]["spans"] and "x_spanmean_nocap" not in items2[0]["spans"]


def test_wiring_check_skip_is_seed_attributable_only(capsys, monkeypatch):
    """r6 pin: the wiring gate SKIPS (with the skipped-seeded record) ONLY when
    the <2-fresh-row shortfall is seed-attributable; a <2-row cell with NO
    seeded rows still raises wiring_check's own assert; a partially seeded
    cell with >=2 fresh rows dispatches to wiring_check as before."""
    import unittest.mock as mock

    import issue1335_extract_store as e1335

    one = [{"item_id": "a", "input_ids": list(range(24)), "n_prompt": 12}]
    # (a) seed-consumed cell: SKIP branch fires with the record + loud log
    w = e1335.wiring_check_or_skip(None, one, 0, 8, 4, n_seeded=37, cell="r0_qa_full/instruct")
    assert w == {"wiring_check": "skipped-seeded", "fresh_rows": 1, "seeded_rows": 37}
    out = capsys.readouterr().out
    assert "wiring check SKIPPED (seed-consumed cell): r0_qa_full/instruct" in out
    assert "fresh=1 seeded=37" in out
    # (b) NOT seed-attributable: the original assert still fails loud
    with pytest.raises(AssertionError, match="wiring check needs >= 2 rows"):
        e1335.wiring_check_or_skip(None, one, 0, 8, 4, n_seeded=0, cell="r0_qa_full/instruct")
    # (c) partially seeded with >=2 fresh rows: runs the check as today
    spec = mock.create_autospec(e1335.wiring_check, return_value={"own_beats_shuffled": True})
    monkeypatch.setattr(e1335, "wiring_check", spec)
    two = [*one, {"item_id": "b", "input_ids": list(range(24)), "n_prompt": 12}]
    w2 = e1335.wiring_check_or_skip(None, two, 0, 8, 4, n_seeded=37, cell="c")
    spec.assert_called_once_with(None, two, 0, 8, 4)
    assert w2 == {"own_beats_shuffled": True}


def test_gate2_tolerates_skipped_seeded_wiring(tmp_path):
    """r6 pin: evaluate_gates records a skipped-seeded wiring file NON-BINDING
    (wiring_pass rides the RAN checks); with no RAN check left, wiring_pass
    stays conservatively False."""
    import issue1335_fit as f1335

    out = tmp_path / "eval"
    out.mkdir()
    rng = np.random.default_rng(0)
    _summary_fixture(out, rng, f1335)
    fp = r1335.fingerprint("r0_qa_full")
    (out / "wiring_r0_qa_full_instruct.json").write_text(
        json.dumps({**fp, "wiring_check": "skipped-seeded", "fresh_rows": 1, "seeded_rows": 37})
    )
    args = SimpleNamespace(out_dir=out, seed=0)
    g2 = f1335.evaluate_gates(args, ["base"], smoke=False)["gate2_qa_endpoint"]
    assert g2["wiring"]["wiring_r0_qa_full_instruct"] == {
        "wiring_check": "skipped-seeded",
        "fresh_rows": 1,
        "seeded_rows": 37,
    }
    assert g2["wiring_pass"] is True  # the fixture's RAN check still binds + passes
    # all-skipped: no RAN check remains -> conservative False
    (out / "wiring_r0_qa_full_base.json").unlink()
    g2b = f1335.evaluate_gates(args, ["base"], smoke=False)["gate2_qa_endpoint"]
    assert g2b["wiring_pass"] is False


# ---------------------------------------------------------------------------
# (7) ladder summary: delta orientation, family membership, lattice, gates
# ---------------------------------------------------------------------------


def _write_matched(out_dir: Path, cell_id: str, slug: str, value: float, rng) -> None:
    fp = r1335.fingerprint(slug)
    draws = []
    for k in range(2):
        boot = (value + rng.normal(0, 0.001, size=50)).tolist()
        draws.append(
            {
                "subsample_seed": 931 + k,
                "n": 100,
                "r2_per_layer": [value] * 28,
                "r2_headline": value,
                "group_bootstrap_l19": {"draws": boot},
            }
        )
    payload = {
        **fp,
        "cell_id": cell_id,
        "n_min": 100,
        "headline_layer": 19,
        "n_draws": 2,
        "r2_headline_mean": value,
        "r2_headline_per_draw": [value, value],
        "draws": draws,
    }
    (out_dir / f"matched_{cell_id}.json").write_text(json.dumps(payload))


def _write_cell(out_dir: Path, cell_id: str, slug: str, value: float, n: int) -> None:
    fp = r1335.fingerprint(slug)
    payload = {
        **fp,
        "cell_id": cell_id,
        "n": n,
        "headline_layer": 19,
        "r2_per_layer_obs": [value] * 28,
        "selection_symmetric": {"frozen_layer_table": {"19": {"null_p975": 0.01}}},
    }
    (out_dir / f"cells_{cell_id}.json").write_text(json.dumps(payload))


def _summary_fixture(out, rng, f1335, s1_val: float = 0.18) -> None:
    """The synthetic eval_results fixture behind the ladder-summary pins
    (matched cells + gate files); s1_val parameterizes the label_restore sign."""
    vals = {
        "r0_qa_full": 0.75,
        "r1_qa_oneline": 0.60,
        "r2_tf": 0.55,
        "r2_op": 0.56,
        "r3_persona": 0.50,
        "r4_fictionframe": 0.40,
    }
    for slug, v in vals.items():
        for arm in ("ctx", "prefix"):
            _write_matched(out, f"{slug}__base__{arm}", slug, v, rng)
    fiction_vals = {
        "r6_nofoil": 0.20,
        "r7_endpoint": 0.15,
        "s1_assistant_label": s1_val,
        "s2a_familiar": 0.16,
        "s2b_novel": 0.14,
    }
    for slug, v in fiction_vals.items():
        for persona in c1310.PERSONA_LABELS:
            _write_matched(out, f"{slug}__base__{persona}__ctx", slug, v, rng)
    # gate fixtures: r7 per-persona full-n cells at the v3 anchors, r0 in range,
    # a passing swap file + one wiring file.
    for persona, ref in f1335.V3_ANCHORS_BASE.items():
        _write_cell(out, f"r7_endpoint__base__{persona}__ctx", "r7_endpoint", ref, 2000)
    _write_cell(out, "r0_qa_full__base__ctx", "r0_qa_full", 0.70, 5000)
    (out / "swap_r7_endpoint_base.json").write_text(
        json.dumps({"delta_r2_char": 0.2, "r2_correct": 0.23, "r2_swap": 0.03})
    )
    (out / "wiring_r0_qa_full_base.json").write_text(
        json.dumps({"own_beats_shuffled": True, "delta": 1.5})
    )


def test_build_ladder_summary_fixture(tmp_path):
    import issue1335_fit as f1335

    out = tmp_path / "eval"
    out.mkdir()
    rng = np.random.default_rng(0)
    _summary_fixture(out, rng, f1335)

    args = SimpleNamespace(out_dir=out, seed=0)
    summary = f1335.build_ladder_summary(args, ["base"], smoke=False)
    pm = summary["per_model"]["base"]

    # orientation: label = R2(r1) - R2(r2_tf) (strong - weak)
    assert pm["deltas"]["label"]["value"] == pytest.approx(0.05, abs=1e-9)
    assert pm["deltas"]["header"]["value"] == pytest.approx(0.06, abs=1e-9)
    assert pm["deltas"]["content_depth"]["value"] == pytest.approx(0.25, abs=1e-9)
    # length delta exists but is OUTSIDE the 6-family
    assert pm["deltas"]["length"]["value"] == pytest.approx(0.15, abs=1e-9)
    assert "length" not in summary["delta_family"]
    assert summary["length_delta_outside_family"] is True
    # Wren-matched content+depth companion present
    assert pm["deltas"]["content_depth_wren_matched"]["value"] == pytest.approx(0.25, abs=1e-9)
    # G at the r1 reference; delta_max is content_depth; D = dmax - 0.5 G, joint CI
    assert pm["gap"]["G"]["value"] == pytest.approx(0.45, abs=1e-9)
    assert pm["delta_max"]["delta"] == "content_depth"
    assert pm["delta_max"]["excluded_negative_deltas"] == []
    assert pm["D"]["value"] == pytest.approx(0.25 - 0.225, abs=1e-9)
    assert pm["D"]["ci_method"] == "joint-draws"
    assert pm["verdict"] == "Single-factor-attributed"
    # gates evaluated + binding, and passing on this fixture
    assert summary["gates"]["binding"] is True
    assert summary["gates"]["gate1_fiction_anchor"]["pass"] is True
    assert summary["gates"]["gate2_qa_endpoint"]["pass"] is True
    assert (out / "ladder_summary.json").exists()


def test_negative_delta_reported_but_excluded_from_max(tmp_path):
    """Plan §3: a negative realized Δ_f (here label_restore, s1 < r7) is
    reported raw but never fed to Δ_max (round-1 review Minor 2)."""
    import issue1335_fit as f1335

    out = tmp_path / "eval"
    out.mkdir()
    rng = np.random.default_rng(0)
    _summary_fixture(out, rng, f1335, s1_val=0.10)
    args = SimpleNamespace(out_dir=out, seed=0)
    summary = f1335.build_ladder_summary(args, ["base"], smoke=False)
    pm = summary["per_model"]["base"]
    # raw value reported (deltas + family_values)...
    assert pm["deltas"]["label_restore"]["value"] == pytest.approx(-0.05, abs=1e-9)
    dmax = pm["delta_max"]
    assert dmax["family_values"]["label_restore"] == pytest.approx(-0.05, abs=1e-9)
    # ...but excluded from the max (content_depth 0.25 stays the max).
    assert dmax["delta"] == "content_depth"
    assert "label_restore" in dmax["excluded_negative_deltas"]


# ---------------------------------------------------------------------------
# (8) Track-S restream fallback (plan assumption 3; round-2 concern close) +
#     the run_swap resume-window predicate (round-1 review Minor 1)
# ---------------------------------------------------------------------------


def _lmsys_fixture_rows() -> list[dict]:
    """Real lmsys-chat-1m row SHAPE (conversation = list of {content, role}
    dicts) at fixture scale — synthetic text only, no network, no corpus rows."""
    return [
        {"conversation": [{"content": "  how do I sort a list in python  ", "role": "user"}]},
        {"conversation": []},  # empty conversation -> dropped
        {"conversation": [{"content": "", "role": "user"}]},  # empty content -> dropped
        {"conversation": [{"value": "value-key fallback row", "role": "user"}]},
        {"no_conversation_key": True},  # missing field -> dropped
        {"conversation": [{"content": "   ", "role": "user"}]},  # whitespace-only -> dropped
        {"conversation": [{"content": "second kept prompt", "role": "user"}]},
        {"conversation": [{"content": "past-n row (never consumed)", "role": "user"}]},
    ]


_FIXTURE_KEPT = ["how do I sort a list in python", "value-key fallback row", "second kept prompt"]


def test_track_s_selector_filters_strips_and_stops():
    got = r1335._select_track_s_prompts(iter(_lmsys_fixture_rows()), 3)
    assert got == _FIXTURE_KEPT  # order preserved, stripped, drops skipped
    # short yield within the scan cap fails loud (never padded)
    with pytest.raises(RuntimeError, match="yielded only"):
        r1335._select_track_s_prompts(iter(_lmsys_fixture_rows()), 10)
    # a pathological 0-keep chain terminates at the scan cap, fail-loud
    empties = ({"conversation": []} for _ in range(100))
    with pytest.raises(RuntimeError, match="scanning 20 rows"):
        r1335._select_track_s_prompts(empties, 5, max_scan=20)


def test_track_s_restream_verified_write(tmp_path):
    sha = r1335._prompt_set_sha256(_FIXTURE_KEPT)
    dest = tmp_path / "track_s.jsonl"
    r1335._restream_track_s(dest, rows=iter(_lmsys_fixture_rows()), expected_sha=sha, n=3)
    got = [json.loads(line) for line in dest.open(encoding="utf-8") if line.strip()]
    assert [r["prompt"] for r in got] == _FIXTURE_KEPT
    assert [r["prompt_idx"] for r in got] == [0, 1, 2]
    # prompt-set hash mismatch -> fail-loud, nothing written at dest
    dest2 = tmp_path / "track_s2.jsonl"
    with pytest.raises(RuntimeError, match="hash mismatch"):
        r1335._restream_track_s(dest2, rows=iter(_lmsys_fixture_rows()), expected_sha="0" * 64, n=3)
    assert not dest2.exists()


def test_restream_subprocess_routes_on_artifact(tmp_path, monkeypatch):
    """Real _restream_track_s_subprocess body with a signature-conformant fake
    child runner: a tolerated rc=134 (the HF-datasets shutdown-abort class)
    passes iff the parent's independent row-count + hash verification passes."""
    monkeypatch.setattr(r1335, "TRACKS_EXPECT_ROWS", 3)
    monkeypatch.setattr(r1335, "TRACKS_PROMPT_SHA256", r1335._prompt_set_sha256(_FIXTURE_KEPT))

    def _child_writes_then_aborts(dest):  # mirrors _run_restream_child(dest) -> int
        r1335._restream_track_s(dest, rows=iter(_lmsys_fixture_rows()))
        return 134  # shutdown SIGABRT AFTER the artifact landed

    dest = tmp_path / "track_s.jsonl"
    r1335._restream_track_s_subprocess(dest, run_child=_child_writes_then_aborts)
    got = [json.loads(line) for line in dest.open(encoding="utf-8") if line.strip()]
    assert [r["prompt"] for r in got] == _FIXTURE_KEPT

    # no artifact -> fail-loud regardless of rc
    with pytest.raises(RuntimeError, match="no artifact"):
        r1335._restream_track_s_subprocess(tmp_path / "missing.jsonl", run_child=lambda d: 1)

    # artifact present but WRONG content -> parent verification fails loud
    bad = tmp_path / "bad.jsonl"

    def _child_writes_wrong(dest):
        dest.write_text(json.dumps({"prompt_idx": 0, "prompt": "wrong corpus row"}) + "\n")
        return 0

    with pytest.raises(RuntimeError, match="parent verification"):
        r1335._restream_track_s_subprocess(bad, run_child=_child_writes_wrong)


def test_fetch_track_s_falls_back_to_restream(tmp_path, monkeypatch):
    """Real _fetch_track_s body; fakes only at the network/process boundaries
    (hf_hub_download + the restream child), signature-conformant."""
    import huggingface_hub

    def _hub_down(repo_id, filename, **kwargs):  # mirrors hf_hub_download's call shape
        raise OSError("hub unavailable (fixture)")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", _hub_down)
    monkeypatch.setattr(r1335, "TRACKS_EXPECT_ROWS", 3)
    monkeypatch.setattr(r1335, "TRACKS_PROMPT_SHA256", r1335._prompt_set_sha256(_FIXTURE_KEPT))
    # child boundary: the REAL _restream_track_s body over fixture rows, in-process
    monkeypatch.setattr(
        r1335,
        "_run_restream_child",
        lambda dest: (r1335._restream_track_s(dest, rows=iter(_lmsys_fixture_rows())), 0)[1],
    )
    dest = tmp_path / "track_s.jsonl"
    r1335._fetch_track_s(dest)  # pinned + main fetch fail -> restream engages
    got = [json.loads(line) for line in dest.open(encoding="utf-8") if line.strip()]
    assert [r["prompt"] for r in got] == _FIXTURE_KEPT

    # restream ALSO failing -> the fail-loud halt survives (the ONLY halt case)
    def _child_raises(dest):
        raise OSError("gated dataset 403 (fixture)")

    monkeypatch.setattr(r1335, "_run_restream_child", _child_raises)
    with pytest.raises(RuntimeError, match="restream fallback failed"):
        r1335._fetch_track_s(tmp_path / "track_s_fail.jsonl")


def test_hf_seed_store_consume_rule(tmp_path, monkeypatch):
    """#1335 r5 relaunch economy: hf_seed_store stages prior-attempt Hub shards
    under the c24 CONSUME rule — REAL body; fakes only at the Hub boundary
    (hub.list_hf_files_under_path + hf_hub_download, signature-conformant). A
    code-SHA drift is staged (warned); a render-config mismatch and a pairless
    sidecar are skipped; a second seed is idempotent."""
    import huggingface_hub
    import issue1335_extract_store as e1335

    from explore_persona_space.orchestrate import hub

    slug = "r7_endpoint"
    fp_now = r1335.fingerprint(slug)
    prefix = f"{r1335.HF_PREFIX}/analysis_tensors/store_{slug}_base"
    sidecars = {
        "base_shard000.json": {**fp_now, "shard_index": 0, "row_ids": ["a"]},
        "base_shard001.json": {
            **fp_now,
            "code_sha": "deadbeef",
            "shard_index": 1,
            "row_ids": ["b"],
        },
        "base_shard002.json": {
            **fp_now,
            "render_config_hash": "0" * 8,
            "shard_index": 2,
            "row_ids": ["c"],
        },
        "base_shard003.json": {**fp_now, "shard_index": 3, "row_ids": ["d"]},  # pairless
    }
    hf_files = [f"{prefix}/{n}" for n in sidecars] + [
        f"{prefix}/base_shard00{i}.pt"
        for i in (0, 1, 2)  # deliberately no 003 .pt
    ]

    def _list(api, repo_id, path, *, repo_type="model", revision=None):
        assert repo_id == r1335.HF_DATA_REPO and path == prefix and repo_type == "dataset"
        return sorted(hf_files)

    def _download(repo_id, filename, *, repo_type=None, local_dir=None, **kw):
        out = Path(local_dir) / filename
        out.parent.mkdir(parents=True, exist_ok=True)
        name = Path(filename).name
        if name.endswith(".json"):
            out.write_text(json.dumps(sidecars[name]))
        else:
            out.write_bytes(b"pt-bytes")
        return str(out)

    monkeypatch.setattr(hub, "list_hf_files_under_path", _list)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", _download)

    store_dir = tmp_path / "store" / slug / "base"
    staged = e1335.hf_seed_store(store_dir, slug, "base")
    assert staged == 2
    assert (store_dir / "base_shard000.pt").exists()
    assert (store_dir / "base_shard000.json").exists()
    assert (store_dir / "base_shard001.pt").exists()  # code-SHA drift: CONSUMED
    assert not (store_dir / "base_shard002.json").exists()  # stale render: NOT seeded
    assert not (store_dir / "base_shard003.json").exists()  # pairless: skipped
    # The staged drifted sidecar passes the CONSUME predicate but not the full
    # match — exactly the pair the r5 three-way resume scan warns-and-skips on.
    side = json.loads((store_dir / "base_shard001.json").read_text())
    assert r1335.fingerprint_matches(side, slug, require_sha=False)
    assert not r1335.fingerprint_matches(side, slug)
    assert e1335.hf_seed_store(store_dir, slug, "base") == 0  # idempotent re-seed


def test_swap_resume_payload_predicate(tmp_path):
    """run_swap resumes ONLY on a fingerprint-matching swap payload; a missing
    or stale payload returns None so the component cells refit LIVE (the
    mid-window preemption fix, round-1 review Minor 1)."""
    import issue1335_fit as f1335

    slug = "r7_endpoint"
    p = tmp_path / "swap_r7_endpoint_base.json"
    assert f1335._swap_resume_payload(p, slug, resume=True) is None  # missing file
    payload = {**r1335.fingerprint(slug), "delta_r2_char": 0.2}
    p.write_text(json.dumps(payload))
    assert f1335._swap_resume_payload(p, slug, resume=False) is None  # resume off
    got = f1335._swap_resume_payload(p, slug, resume=True)
    assert got is not None and got["delta_r2_char"] == 0.2
    stale = {**payload, "render_config_hash": "0" * len(str(payload["render_config_hash"]))}
    p.write_text(json.dumps(stale))
    assert f1335._swap_resume_payload(p, slug, resume=True) is None  # stale fingerprint


def test_run_swap_survives_mid_window_preemption(tmp_path):
    """The reviewer's mechanizable round-1 Minor-1 scenario: component cells
    persisted, swap payload lost (crash in the write window) — a --resume
    relaunch must REWRITE the swap payload (real run_swap + fit_cell bodies on
    a tiny synthetic store), and a third call resume-skips on the payload."""
    import issue1335_fit as f1335

    rng = np.random.default_rng(0)
    n_groups, n_layers, d, k = 20, 2, 6, 4
    store = {
        "group_ids": np.repeat([f"g{i:02d}" for i in range(n_groups)], 2),
        "char_ids": np.array(["A", "B"] * n_groups),
        "turn_indices": np.zeros(2 * n_groups, dtype=int),
        "row_ids": np.array([f"row{i}" for i in range(2 * n_groups)]),
        "arrays": {
            "x_spanmean": rng.normal(size=(2 * n_groups, n_layers, d)).astype(np.float32),
            "y": rng.normal(size=(2 * n_groups, n_layers, k)).astype(np.float32),
        },
    }
    out = tmp_path / "eval"
    out.mkdir()
    args = SimpleNamespace(out_dir=out, folds=2, n_boot=8, null_draws=2, seed=0, resume=True)
    swap_path = out / "swap_r7_endpoint_base.json"
    p1 = f1335.run_swap("r7_endpoint", store, "base", args)
    assert p1 is not None and swap_path.exists()
    # mid-window preemption: swap payload lost, component cells persisted
    swap_path.unlink()
    p2 = f1335.run_swap("r7_endpoint", store, "base", args)
    assert p2 is not None and swap_path.exists(), "gate-1 brick: swap payload not rewritten"
    assert p2["delta_r2_char"] == pytest.approx(p1["delta_r2_char"], abs=1e-12)
    # valid payload present -> resume-skip returns it without refitting
    p3 = f1335.run_swap("r7_endpoint", store, "base", args)
    assert p3["delta_r2_char"] == p2["delta_r2_char"]
