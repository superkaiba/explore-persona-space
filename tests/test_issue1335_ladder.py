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
    row-length cap).
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


def test_build_ladder_summary_fixture(tmp_path):
    import issue1335_fit as f1335

    out = tmp_path / "eval"
    out.mkdir()
    rng = np.random.default_rng(0)
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
        "s1_assistant_label": 0.18,
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
    assert pm["D"]["value"] == pytest.approx(0.25 - 0.225, abs=1e-9)
    assert pm["D"]["ci_method"] == "joint-draws"
    assert pm["verdict"] == "Single-factor-attributed"
    # gates evaluated + binding, and passing on this fixture
    assert summary["gates"]["binding"] is True
    assert summary["gates"]["gate1_fiction_anchor"]["pass"] is True
    assert summary["gates"]["gate2_qa_endpoint"]["pass"] is True
    assert (out / "ladder_summary.json").exists()
