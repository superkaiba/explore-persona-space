# marker token intentional
"""#1333 CPU pre-launch unit tests (plan § Dry-run smoke item 1).

Covers: cell-table / training-config builders (every load-bearing recipe value
asserted on the BUILT TrainLoraConfig + the FT command), mix-derivation
partitions (200/800; marker-id tail assert; the cell-7 substitution partition),
the RENDERED-TOKEN disjointness assert incl. the bare≡qwen_default collision as
a MUST-FAIL fixture, and the off-line selection rule.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from explore_persona_space.experiments import issue_1333 as C
from explore_persona_space.experiments.factor_screen_365.persona_panel import (
    EVAL_PERSONAS_24,
    EVAL_QUESTIONS_20,
)


@pytest.fixture(scope="module")
def tok():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(C.BASE_MODEL)


# ── Config builders ───────────────────────────────────────────────────────────


@pytest.mark.parametrize("cell", C.NEW_LORA_CELLS)
def test_lora_config_carries_the_registered_recipe(cell):
    cfg = C.marker_lora_config(cell, out_root="/tmp/i1333-test")
    assert cfg.lr == 5e-6
    assert cfg.max_steps == C.LORA_MAX_STEPS == 400
    assert cfg.save_steps == (5 if cell == C.CELL_LORA_POS else 10)
    assert cfg.save_total_limit is None  # the #641 ladder-pruning trap
    assert cfg.warmup_steps == 5
    assert cfg.marker_only_loss is True
    assert cfg.marker_text == C.MARKER_TEXT
    assert cfg.lora_r == 16 and cfg.lora_alpha == 32 and cfg.lora_dropout == 0.0
    assert set(cfg.lora_targets) == {"q_proj", "k_proj", "v_proj", "o_proj"}
    assert cfg.max_length == 2048
    # In-loop band repurposed to the [25, 30] safety ceiling (plan §4.3).
    assert (cfg.marker_band_low_nats, cfg.marker_band_high_nats) == C.SAFETY_BAND
    assert cfg.marker_band_stop is True
    assert cfg.marker_band_trajectory_path.endswith(f"{cell}/band_trajectory.json")
    # eff-batch 16 = 4 x 4 (MARKER_OVERRIDES verbatim)
    assert cfg.batch_size == 4 and cfg.grad_accum == 4


def test_lora_config_rejects_non_lora_cells():
    with pytest.raises(ValueError, match="not a new LoRA cell"):
        C.marker_lora_config(C.CELL_FT_POS)


def test_ft_cmd_matches_reused_arm_recipe():
    cmd = C.marker_ft_cmd(mix_path="mix.jsonl", out_dir="out", num_processes=4)
    joined = " ".join(cmd)
    assert "--ckpt-steps 1,2,3,4,5,6" in joined
    assert "--max-steps 6" in joined  # reused arm's realized decay horizon
    assert "zero3_4gpu_accum16.yaml" in joined
    assert "issue1112_train_marker_fullft.py" in joined
    assert "--seed 42" in joined


def test_ft_cmd_schedule_parity_guard():
    with pytest.raises(ValueError, match="schedule parity"):
        C.marker_ft_cmd(mix_path="m", out_dir="o", num_processes=4, max_steps=10)


def test_cell_table_invariants():
    assert set(C.CELL_MIX) == set(C.ALL_TRAINED_CELLS)
    assert set(C.CELL_SOURCE_CONTEXT) == set(C.ALL_TRAINED_CELLS)
    assert set(C.CELL_NEGATIVES) == set(C.ALL_TRAINED_CELLS)
    assert len(set(C.ALL_TRAINED_CELLS)) == 7
    # cell 7's panel substitutes french_person for qwen_default, keeping size 4
    assert "qwen_default" not in C.CELL_NEGATIVES[C.CELL_EXT_BARE]
    assert "french_person" in C.CELL_NEGATIVES[C.CELL_EXT_BARE]
    assert len(C.CELL_NEGATIVES[C.CELL_EXT_BARE]) == len(C.FROZEN_NEGATIVES) == 4
    # positives-only cells train NO panel (contrastive exemption (a))
    assert C.CELL_NEGATIVES[C.CELL_LORA_POS] == () and C.CELL_NEGATIVES[C.CELL_FT_POS] == ()
    assert abs(C.ACCEPT_WINDOW[0] - 4.28486385345459) < 1e-9
    assert abs(C.ACCEPT_WINDOW[1] - 8.28486385345459) < 1e-9


# ── Mix derivations ───────────────────────────────────────────────────────────


def _synthetic_frozen_rows() -> list[dict]:
    rows = []
    villain = EVAL_PERSONAS_24["villain"]
    for i in range(C.POS_EX):
        rows.append(
            {
                "prompt": [
                    {"role": "system", "content": villain},
                    {"role": "user", "content": f"q{i % 10}"},
                ],
                "completion": [
                    {"role": "assistant", "content": f"answer {i}.{C.MARKER_SEP}{C.MARKER_TEXT}"}
                ],
            }
        )
    for persona in C.FROZEN_NEGATIVES:
        for i in range(C.NEG_EX_PER_PERSONA):
            rows.append(
                {
                    "prompt": [
                        {"role": "system", "content": EVAL_PERSONAS_24[persona]},
                        {"role": "user", "content": f"q{i % 10}"},
                    ],
                    "completion": [{"role": "assistant", "content": f"plain answer {persona} {i}"}],
                }
            )
    return rows


def test_partition_frozen_mix_200_800():
    pos, neg = C.partition_frozen_mix(_synthetic_frozen_rows())
    assert len(pos) == 200 and len(neg) == 800
    by_p = C.negatives_by_persona(neg, {p: EVAL_PERSONAS_24[p] for p in C.FROZEN_NEGATIVES})
    assert {p: len(v) for p, v in by_p.items()} == {p: 200 for p in C.FROZEN_NEGATIVES}


def test_partition_frozen_mix_fails_on_wrong_counts():
    rows = _synthetic_frozen_rows()[1:]  # drop one positive -> 199/800
    with pytest.raises(ValueError, match="partition"):
        C.partition_frozen_mix(rows)


def test_partition_fails_on_contaminated_negative():
    rows = _synthetic_frozen_rows()
    rows[-1]["completion"][0]["content"] = f"sneaky {C.MARKER_TEXT} inside"
    with pytest.raises(ValueError, match="contamination"):
        C.partition_frozen_mix(rows)


def test_derive_posonly_mix_with_marker_id_tail_assert(tmp_path, tok):
    frozen = tmp_path / "frozen.jsonl"
    with open(frozen, "w", encoding="utf-8") as f:
        for r in _synthetic_frozen_rows():
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    man = C.derive_posonly_mix(frozen, tmp_path / "posonly.jsonl", tokenizer=tok)
    assert man["n_rows"] == 200
    rows = C._read_jsonl(tmp_path / "posonly.jsonl")
    assert len(rows) == 200 and all(C._row_is_positive(r) for r in rows)
    assert Path(tmp_path / "posonly.manifest.json").exists()


def test_extension_mix_cell7_substitution_partition(tmp_path):
    rows = _synthetic_frozen_rows()
    _pos, neg = C.partition_frozen_mix(rows)
    frozen_negs = C.negatives_by_persona(neg, {p: EVAL_PERSONAS_24[p] for p in C.FROZEN_NEGATIVES})
    qs = [f"q{i}" for i in range(10)]
    man = C.build_extension_mix(
        C.CELL_EXT_BARE,
        source_msgs_for_q=lambda q: [{"role": "user", "content": q}],
        greedy_r_for_q=lambda q: f"bare answer to {q}",
        train_questions=qs,
        frozen_negatives=frozen_negs,
        french_r_for_q=lambda q: f"réponse to {q}",
        french_system="You are a French person.",
        out_path=tmp_path / "marker_bare.jsonl",
    )
    assert (man["n_positive"], man["n_negative"]) == (200, 800)
    out_rows = C._read_jsonl(tmp_path / "marker_bare.jsonl")
    assert len(out_rows) == 1000
    # 600 frozen (police/comedian/medical) + 200 fresh french + 0 qwen_default
    systems = [r["prompt"][0]["content"] for r in out_rows if r["prompt"][0]["role"] == "system"]
    assert systems.count("You are a French person.") == 200
    assert systems.count(EVAL_PERSONAS_24["qwen_default"]) == 0
    for p in ("police_officer", "comedian", "medical_doctor"):
        assert systems.count(EVAL_PERSONAS_24[p]) == 200
    # positives: bare rows (no system message) whose completion ends with the marker
    bare_pos = [r for r in out_rows if r["prompt"][0]["role"] == "user"]
    assert len(bare_pos) == 200 and all(C._row_is_positive(r) for r in bare_pos)


def test_extension_mix_verbatim_negatives_for_cell5(tmp_path):
    rows = _synthetic_frozen_rows()
    _pos, neg = C.partition_frozen_mix(rows)
    frozen_negs = C.negatives_by_persona(neg, {p: EVAL_PERSONAS_24[p] for p in C.FROZEN_NEGATIVES})
    qs = [f"q{i}" for i in range(10)]
    man = C.build_extension_mix(
        C.CELL_EXT_WILDCHAT,
        source_msgs_for_q=lambda q: [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
            {"role": "user", "content": q},
        ],
        greedy_r_for_q=lambda q: f"wc answer to {q}",
        train_questions=qs,
        frozen_negatives=frozen_negs,
        out_path=tmp_path / "marker_wildchat.jsonl",
    )
    assert (man["n_positive"], man["n_negative"]) == (200, 800)
    out_rows = C._read_jsonl(tmp_path / "marker_wildchat.jsonl")
    systems = [r["prompt"][0]["content"] for r in out_rows if r["prompt"][0]["role"] == "system"]
    assert systems.count(EVAL_PERSONAS_24["qwen_default"]) == 200  # frozen panel VERBATIM


# ── Rendered-token disjointness (incl. the must-fail collision fixture) ───────


def _persona_msgs(persona: str):
    def msgs(q: str) -> list[dict]:
        return [
            {"role": "system", "content": EVAL_PERSONAS_24[persona]},
            {"role": "user", "content": q},
        ]

    return msgs


def _bare_msgs(q: str) -> list[dict]:
    return [{"role": "user", "content": q}]


def test_rendered_disjointness_must_fail_on_bare_vs_qwen_default(tok):
    """The registered MUST-FAIL fixture (plan §4.2): the bare context renders
    BYTE-IDENTICAL to qwen_default (assumption 16 — the chat template inserts
    the Qwen default system prompt), so a panel keeping qwen_default under the
    bare source MUST trip the rendered-token assert."""
    with pytest.raises(ValueError, match="rendered-token collision"):
        C.assert_rendered_disjoint(
            tok,
            source_id="bare_default",
            source_msgs_for_q=_bare_msgs,
            panel={"qwen_default": _persona_msgs("qwen_default")},
            questions=list(EVAL_QUESTIONS_20[:2]),
        )


def test_rendered_disjointness_passes_on_frozen_panel(tok):
    C.assert_rendered_disjoint(
        tok,
        source_id="persona_villain",
        source_msgs_for_q=_persona_msgs("villain"),
        panel={p: _persona_msgs(p) for p in C.FROZEN_NEGATIVES},
        questions=list(EVAL_QUESTIONS_20[:2]),
    )


def test_rendered_disjointness_french_person_distinct(tok):
    from explore_persona_space.artifacts.context import context_for_persona

    fr = context_for_persona("french_person")
    C.assert_rendered_disjoint(
        tok,
        source_id="bare_default",
        source_msgs_for_q=_bare_msgs,
        panel={
            "french_person": fr.messages,
            **{p: _persona_msgs(p) for p in ("police_officer", "comedian", "medical_doctor")},
            **{p: _persona_msgs(p) for p in C.HELD_OUT_TRIO},
            "villain": _persona_msgs("villain"),
        },
        questions=list(EVAL_QUESTIONS_20[:2]),
    )


def test_bare_renders_identical_to_qwen_default(tok):
    """Assumption 16 pinned: bare == qwen_default at exact template output ids."""
    q = "What is your favorite hobby?"
    bare = C.rendered_ids(tok, _bare_msgs, q)
    qwen = C.rendered_ids(tok, _persona_msgs("qwen_default"), q)
    assert bare == qwen


# ── Off-line selection rule ───────────────────────────────────────────────────


def _read(delta, emit=0.0, sat=False):
    return {"delta_logp_mean": delta, "source_emission_rate": emit, "bystander_saturated": sat}


def test_select_rung_picks_closest_in_window_earliest_tie():
    ladder = {10: _read(4.5), 20: _read(6.3), 30: _read(6.56), 40: _read(9.0)}
    sel = C.select_rung(ladder)
    assert sel["step"] == 20 and sel["in_window"] and sel["fallback"] is None
    # symmetric tie -> earliest step
    tie = {10: _read(C.TARGET_DELTA_G - 0.5), 20: _read(C.TARGET_DELTA_G + 0.5)}
    assert C.select_rung(tie)["step"] == 10


def test_select_rung_gates_on_emission_and_saturation():
    ladder = {10: _read(6.3, emit=0.05), 20: _read(6.9), 30: _read(6.2, sat=True)}
    sel = C.select_rung(ladder)
    assert sel["step"] == 20  # 10 excluded (emission), 30 excluded (saturated)


def test_select_rung_closest_approach_fallback_is_labeled():
    ladder = {10: _read(0.5), 20: _read(1.58)}
    sel = C.select_rung(ladder)
    assert sel["in_window"] is False and sel["fallback"] == "closest_approach"
    assert sel["step"] == 20  # closest approach to +6.28


def test_coarse_and_refine_schedules():
    rungs = list(range(10, 401, 10))
    coarse = C.coarse_read_steps(C.CELL_LORA_CON, rungs)
    assert coarse[0] == 20 and 400 in coarse and 10 not in coarse
    pos_rungs = list(range(5, 401, 5))
    pos_coarse = C.coarse_read_steps(C.CELL_LORA_POS, pos_rungs)
    assert {5, 10, 15}.issubset(pos_coarse)
    # refine: window crossing between coarse reads opens the 10-step rungs
    ladder = {20: _read(0.4), 40: _read(2.0), 60: _read(7.0)}
    refine = C.refine_read_steps(C.CELL_LORA_CON, rungs, ladder)
    assert refine == [50]
