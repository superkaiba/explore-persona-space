"""Issue #1345 conversation-paired-stories round — registry + gate pins.

Covers the plan-v8 additions:
  1. Variant-GATED registry: without EPM_I1345_VARIANT=conversation_paired_stories
     the parent registry is BYTE-IDENTICAL (3 regimes, 12 cells, 3 unordered
     pairs, no PAIRED_PAIR_R4); with it, r4 + the r4op companion appear for
     instruct only (base N/A by scope) and PAIRED_PAIR_R4 = ("r1", "r4").
  2. The mechanical verbatim keep-filter (match_verbatim_turn) — the r4 gen
     phase's span gate: exactly-one-exchange, verbatim answer, no pre-slot
     verbatim leak. Real production bodies, benign synthetic strings.
  3. The judge reply parser (EXCHANGES/VERDICT reason-then-verdict shape).
  4. select_cells --no-r4 drop + pair_kind_for grain mapping.
  5. build_matched per_model_r4_pair: production fail-loud on foreign convs /
     duplicates; smoke informational skip; explicit companion demotion record.
  6. tf_op_calibration nested tiers (plan v8 §7): tier1 qualification,
     tier2 TF-DISTORTED — reporting labels, never process halts; the rc=23
     companion-halt lane writes companion: halted (r1 code-review Major).
  7. Transfer matrix loop under a DEMOTED include_r4 (the pod-smoke path where
     the r4 pair build smoke-skips): logged skip, never a TypeError
     (r1 code-review Critical).
  8. select_cells --no-r4op: drops ONLY the r4op companion cells (the fits leg
     of the rc=23 halt lane).
  9. companion_usable_floor: production floor == the grouped-CV minimum
     (fc.N_FOLDS); smoke floor 1 (gate-calibration rule).
 10. paired_fingerprint keys the op branch on mode_slug "paired_op" (r6
     content-keyed resume; r1 code-review Minor).

Registry tests run in SUBPROCESSES (issue1345_common reads the variant env at
import — the name-seam test pattern).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

import issue1345_common as c  # noqa: E402
import issue1345_gen_stories_paired as gp  # noqa: E402
from issue1345_cross_regime_transfer import pair_kind_for  # noqa: E402
from issue1345_fit_cells import build_matched, select_cells  # noqa: E402
from issue1345_matched_row_refits import matched_row_cells, tf_op_calibration  # noqa: E402

_SEAM_KEYS = ("EPM_STORY_CHARACTER_NAME", "EPM_I1345_VARIANT")
CPS = "conversation_paired_stories"


def _run_py(code: str, overrides: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items() if k not in _SEAM_KEYS}
    env.update(overrides or {})
    return subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        env=env,
        timeout=180,
    )


_REGISTRY_PROBE = """
import sys
sys.path.insert(0, "scripts")
import json
import issue1345_common as c
cells = c.all_cells()
print(json.dumps({
    "regimes": list(c.REGIMES),
    "n_cells": len(cells),
    "cell_ids": sorted(x["cell_id"] for x in cells),
    "unordered_pairs": [list(p) for p in c.UNORDERED_PAIRS],
    "paired_pair_r4": list(c.PAIRED_PAIR_R4) if c.PAIRED_PAIR_R4 else None,
    "has_r4": c.HAS_R4,
}))
"""


def test_parent_registry_unchanged_without_variant():
    proc = _run_py(_REGISTRY_PROBE)
    assert proc.returncode == 0, proc.stderr
    out = json.loads(proc.stdout)
    assert out["regimes"] == ["r1", "r2", "r3"]
    assert out["n_cells"] == 12
    assert out["unordered_pairs"] == [["r1", "r2"], ["r1", "r3"], ["r2", "r3"]]
    assert out["paired_pair_r4"] is None and not out["has_r4"]
    assert not any("r4" in cid for cid in out["cell_ids"])


def test_paired_variant_registry_gains_r4_instruct_only():
    proc = _run_py(_REGISTRY_PROBE, {"EPM_I1345_VARIANT": CPS})
    assert proc.returncode == 0, proc.stderr
    out = json.loads(proc.stdout)
    assert out["regimes"] == ["r1", "r2", "r3", "r4"]
    assert out["has_r4"] and out["paired_pair_r4"] == ["r1", "r4"]
    assert ["r1", "r4"] in out["unordered_pairs"] and ["r3", "r4"] in out["unordered_pairs"]
    # 12 parent + 2 instruct r4 + 2 instruct r4op companion; NO base r4 (scope)
    assert out["n_cells"] == 16
    for cid in (
        "R_instruct_r4_context",
        "R_instruct_r4_prefix",
        "R_instruct_r4_op_companion_context",
        "R_instruct_r4_op_companion_prefix",
    ):
        assert cid in out["cell_ids"], cid
    assert not any(cid.startswith("R_base_r4") for cid in out["cell_ids"])


def test_other_variant_does_not_gain_r4():
    proc = _run_py(
        _REGISTRY_PROBE,
        {"EPM_I1345_VARIANT": "assistant_named_story", "EPM_STORY_CHARACTER_NAME": "Assistant"},
    )
    assert proc.returncode == 0, proc.stderr
    out = json.loads(proc.stdout)
    assert out["regimes"] == ["r1", "r2", "r3"] and out["n_cells"] == 12


# ---------------------------------------------------------------------------
# Mechanical verbatim keep-filter (production bodies; benign synthetic text)
# ---------------------------------------------------------------------------
ANSWER = "Visitors must sign in at the front desk and wear a badge at all times."
STORY_OK = (
    "The lab hummed quietly as the afternoon light faded. Maya set down her pen, "
    'glanced at the console, and asked, "What is the visitor policy here?" '
    f'ARIA replied: "{ANSWER}" Maya nodded and made a note of it.'
)


def test_match_verbatim_turn_keeps_conforming_story():
    turn, reason = gp.match_verbatim_turn(STORY_OK, ANSWER)
    assert reason == "ok" and turn is not None
    assert STORY_OK[turn["a_start"] : turn["a_end"]] == ANSWER
    # Slot ordering the extraction render depends on
    assert turn["q_end"] <= turn["marker_end"] < turn["a_start"]


def test_match_verbatim_turn_keeps_quote_bearing_answer():
    # 29% of the real shared pool carries embedded double quotes — the
    # answer-anchored matcher must keep them (quote-pairing capped the pool
    # at 2,293 < the 2,700 target; measured at implementation time).
    ans_q = 'Type "hello world" into the terminal and press enter to run it.'
    story = (
        'Sam frowned at the blinking cursor and asked, "How do I start the program?" '
        f'ARIA replied: "{ans_q}" Sam tried it at once.'
    )
    turn, reason = gp.match_verbatim_turn(story, ans_q)
    assert reason == "ok" and turn is not None
    assert story[turn["a_start"] : turn["a_end"]] == ans_q


def test_match_verbatim_turn_drops_paraphrase():
    story = STORY_OK.replace(ANSWER, "Please sign in at the desk and wear your badge.")
    turn, reason = gp.match_verbatim_turn(story, ANSWER)
    assert turn is None and reason == "answer_occurrences_zero"


def test_match_verbatim_turn_drops_multi_exchange():
    story = STORY_OK + ' Then she asked, "Is that all?" ARIA answered: "Yes, that is all."'
    turn, reason = gp.match_verbatim_turn(story, ANSWER)
    assert turn is None and reason == "attribution_multi"


def test_match_verbatim_turn_drops_pre_slot_verbatim_leak():
    story = (
        f"Maya already knew the rule: {ANSWER} Still, she asked, "
        f'"What is the visitor policy here?" ARIA replied: "{ANSWER}"'
    )
    turn, reason = gp.match_verbatim_turn(story, ANSWER)
    assert turn is None and reason == "answer_occurrences_multi"


def test_match_verbatim_turn_drops_unattributed_answer():
    # Quoted but NOT ARIA-attributed (a wall sign; Maya quotes it herself)
    story = f'The policy sign read: "{ANSWER}" Maya recited it and asked, "Anything else?"'
    turn, reason = gp.match_verbatim_turn(story, ANSWER)
    assert turn is None and reason == "attribution_zero"


def test_match_verbatim_turn_drops_zero_exchange():
    turn, reason = gp.match_verbatim_turn("Nothing was asked and nothing answered.", ANSWER)
    assert turn is None and reason == "answer_occurrences_zero"


def test_match_verbatim_turn_drops_extended_quote():
    # Model appended extra words INSIDE the quote after the verbatim answer
    story = STORY_OK.replace(f'"{ANSWER}"', f'"{ANSWER} Also bring an ID."')
    turn, reason = gp.match_verbatim_turn(story, ANSWER)
    assert turn is None and reason == "answer_quote_not_closed"


def test_confident_op_turn_keeps_one_exchange():
    turn, reason = gp.confident_op_turn(STORY_OK)
    assert reason == "ok" and turn is not None


def test_judge_parser_reason_then_verdict():
    parsed = gp._parse_judge_response(
        "The story has one exchange and the answer matches.\nEXCHANGES: 1\nVERDICT: PASS"
    )
    assert parsed == {"verdict": "PASS", "judge_exchanges": 1}
    with pytest.raises(ValueError, match="missing VERDICT"):
        gp._parse_judge_response("Some reasoning without a verdict line.")


def test_pool_filter_keeps_quote_bearing_drops_degenerate():
    rows = [
        {"conv_id": "s1", "prompt": "q", "response": ANSWER},
        {"conv_id": "s2", "prompt": "q", "response": 'He said "hello" loudly and clearly.'},
        {"conv_id": "s3", "prompt": "q", "response": "short"},
    ]
    kept = [r for r in rows if len(r["response"]) >= gp.ANSWER_CHAR_MIN]
    assert [r["conv_id"] for r in kept] == ["s1", "s2"]  # quote-bearing KEPT (answer-anchored)


def test_companion_usable_floor_grouped_cv_minimum():
    """r1 review Major (kept∈[2,4] gap): the production floor IS the grouped-CV
    minimum — below fc.N_FOLDS conv-groups the companion fit populates no fold.
    Smoke floor 1: any NONZERO yield proceeds (gate-calibration rule)."""
    import issue825_fit_cells as fc

    assert c.OP_COMPANION_MIN_KEPT == fc.N_FOLDS == 5
    assert gp.companion_usable_floor(False) == c.OP_COMPANION_MIN_KEPT
    assert gp.companion_usable_floor(True) == 1


def test_paired_fingerprint_op_branch_keys_on_paired_op(monkeypatch):
    """r1 review Minor pin: the companion bundle fp (mode_slug "paired_op")
    embeds the OP recipe — an OP-side recipe change must change the op fp
    (pre-fix the op branch keyed on mode == "op" and embedded the PAIRED
    templates/parser, so a stale companion bundle would resume from HF)."""
    rows = [{"conv_id": "s1", "question": "q?", "answer": "a" * 30}]
    fp_op = gp.paired_fingerprint("paired_op", rows)
    fp_paired = gp.paired_fingerprint("paired", rows)
    assert fp_op != fp_paired  # distinct bundles (the mode field alone ensures this)
    monkeypatch.setattr(gp, "STORY_OP_COMPANION_SYSTEM", gp.STORY_OP_COMPANION_SYSTEM + " CHANGED")
    assert gp.paired_fingerprint("paired_op", rows) != fp_op  # op recipe IS in the op fp
    assert gp.paired_fingerprint("paired", rows) == fp_paired  # paired fp untouched


# ---------------------------------------------------------------------------
# Companion-mode row schema (att-20260716-230002 crash pins, cps crash-fix r3):
# main()'s --op-companion branch passes rows {conv_id, question} — free
# generation has no fixed answer — while paired rows HARD-carry the verbatim
# answer. Every consumer of these rows is mode-keyed; fakes below sit ONLY at
# the external HF-tokenizer / vLLM-engine boundaries, signature-conformant
# (they mirror the exact surfaces the production bodies call).
# ---------------------------------------------------------------------------
COMPANION_ROWS = [
    {"conv_id": "s0", "question": "What is the visitor policy at the lab?"},
    {"conv_id": "s1", "question": "How do I reset my workstation password?"},
]


class _FakeTokenizer:
    """HF-tokenizer-boundary fake: mirrors the two surfaces the gen script
    calls — ``__call__(text, add_special_tokens=)`` -> {"input_ids": [...]}
    and ``apply_chat_template(messages, tokenize=, add_generation_prompt=)``
    (whitespace token ids; template = joined contents)."""

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": list(range(len(text.split())))}

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        assert not tokenize and add_generation_prompt
        return "\n".join(m["content"] for m in messages) + "\nassistant:"


class _FakeCompletion:
    def __init__(self, text: str):
        self.text = text
        self.finish_reason = "stop"


class _FakeRequestOutput:
    def __init__(self, text: str):
        self.outputs = [_FakeCompletion(text)]


class _FakeLLM:
    """vLLM-engine-boundary fake mirroring the exact ``LLM.generate(prompts,
    sampling_params, use_tqdm=)`` surface generate_paired calls."""

    def generate(self, prompts, sampling_params, use_tqdm=False):
        assert use_tqdm is False
        return [_FakeRequestOutput(f"A short scene. {p[-40:]}") for p in prompts]


def test_filter_pool_feasible_companion_rows_have_no_answer_key():
    """THE crash pin (GCP att-20260716-230002, gen_stories_paired.py:237):
    companion rows {conv_id, question} through op_companion=True — pre-fix the
    filter tokenized row['answer'] unconditionally and KeyError'd on row 1."""
    kept, counts = gp._filter_pool_feasible(COMPANION_ROWS, _FakeTokenizer(), op_companion=True)
    assert [r["conv_id"] for r in kept] == ["s0", "s1"]
    assert counts == {"prompt_over_budget": 0, "answer_over_budget": 0}


def test_filter_pool_feasible_paired_mode_hard_requires_answer():
    """Paired rows HARD-key the verbatim answer (fail-fast, never .get-default);
    the answer budget applies in paired mode only."""
    with pytest.raises(KeyError, match="answer"):
        gp._filter_pool_feasible(COMPANION_ROWS, _FakeTokenizer(), op_companion=False)
    ok = [{**r, "answer": "b " * 30} for r in COMPANION_ROWS]
    over = [{"conv_id": "s9", "question": "q?", "answer": "w " * (gp.ANSWER_TOKEN_BUDGET + 1)}]
    kept, counts = gp._filter_pool_feasible(ok + over, _FakeTokenizer(), op_companion=False)
    assert [r["conv_id"] for r in kept] == ["s0", "s1"]
    assert counts["answer_over_budget"] == 1


def test_paired_fingerprint_mode_keyed_row_schema():
    """The op fp is computable WITHOUT an answer key and invariant to a stray
    one (companion identity = (conv_id, question)); the paired fp hard-keys
    the answer, byte-identical to the prior (cid, q, answer) triple."""
    op_rows = [{"conv_id": "s0", "question": "q?"}]
    fp = gp.paired_fingerprint("paired_op", op_rows)
    assert fp == gp.paired_fingerprint("paired_op", [{**op_rows[0], "answer": "stray"}])
    with pytest.raises(KeyError, match="answer"):
        gp.paired_fingerprint("paired", op_rows)


def test_build_judge_request_mode_keyed_payload():
    """The op judge payload carries no answer (the op rubric never reads one);
    the paired payload hard-keys it."""
    from explore_persona_space.llm.api_dispatch import DispatchItem

    op = gp._build_judge_request(
        DispatchItem(item_id="s0", payload={"story": "Once upon a time.", "mode": "op"})
    )
    assert op["system"] == gp.JUDGE_SYSTEM_OP
    assert "Once upon a time." in op["messages"][0]["content"]
    paired = gp._build_judge_request(
        DispatchItem(item_id="s1", payload={"story": "S.", "answer": "A" * 25, "mode": "paired"})
    )
    assert paired["system"] == gp.JUDGE_SYSTEM_PAIRED
    assert "A" * 25 in paired["messages"][0]["content"]
    with pytest.raises(KeyError, match="answer"):
        gp._build_judge_request(
            DispatchItem(item_id="s2", payload={"story": "S.", "mode": "paired"})
        )


def test_generate_paired_row_schema_mode_keyed(tmp_path):
    """generate_paired executes for real (chunking, meta write, JSONL append)
    against boundary fakes: op story rows OMIT 'answer'; paired rows carry it
    verbatim (the extract r4op / op-judge consumers never read an op answer)."""
    pytest.importorskip("vllm")  # generate_paired imports SamplingParams at call time
    tok = _FakeTokenizer()
    op_rows = gp.generate_paired(
        [dict(r) for r in COMPANION_ROWS],
        tmp_path / "raw_op.jsonl",
        "fp-op",
        tok,
        _FakeLLM(),
        op_companion=True,
    )
    assert len(op_rows) == 2
    assert all(r["mode"] == "op" and "answer" not in r for r in op_rows)
    paired_in = [{**r, "answer": "c" * 40} for r in COMPANION_ROWS]
    paired_rows = gp.generate_paired(
        paired_in, tmp_path / "raw_paired.jsonl", "fp-p", tok, _FakeLLM(), op_companion=False
    )
    assert all(r["mode"] == "paired" and r["answer"] == "c" * 40 for r in paired_rows)


# ---------------------------------------------------------------------------
# Registry consumers
# ---------------------------------------------------------------------------
def test_select_cells_no_r4_drop_is_noop_without_variant(capsys):
    cells = select_cells("all", set(), no_r4=True)
    assert len(cells) == 12  # ambient env: no r4 cells to drop
    assert len(select_cells("all", set(), no_r4op=True)) == 12  # ditto for r4op


_SELECT_NO_R4OP_PROBE = """
import sys
sys.path.insert(0, "scripts")
import json
from issue1345_fit_cells import select_cells
ids = sorted(x["cell_id"] for x in select_cells("all", set(), no_r4op=True))
print(json.dumps(ids))
"""


def test_select_cells_no_r4op_drops_companion_only():
    """rc=23 fits leg (r1 review Major): --no-r4op drops ONLY the r4op cells,
    keeping the r4 TF headline cells (contrast --no-r4, which drops both)."""
    proc = _run_py(_SELECT_NO_R4OP_PROBE, {"EPM_I1345_VARIANT": CPS})
    assert proc.returncode == 0, proc.stderr
    ids = json.loads(proc.stdout.strip().splitlines()[-1])
    assert len(ids) == 14  # 16 variant cells - 2 r4op companion cells
    assert not any("r4_op_companion" in cid for cid in ids)
    assert "R_instruct_r4_context" in ids and "R_instruct_r4_prefix" in ids


_TRANSFER_DEMOTED_R4_PROBE = """
import sys
sys.path.insert(0, "scripts")
import json
import tempfile
from pathlib import Path
import numpy as np
import issue1345_common as c
from issue1345_cross_regime_transfer import run_model_arm

assert c.HAS_R4
rng = np.random.default_rng(0)


def bundle(convs, n_turns):
    n = len(convs)
    return {
        "arrays": {
            "slots": rng.normal(size=(n, 2, 28, 8)).astype("float32"),
            "profiles": rng.normal(size=(n, n_turns, 28, 8)).astype("float32"),
        },
        "sidecar": {"conv_ids": list(convs)},
    }


convs = ["s%d" % i for i in range(6)]
bundles = {"r1": bundle(convs, 2), "r2": bundle(convs, 2), "r4": bundle(convs[:4], 1)}
# The pod-smoke shape (r1 code-review Critical): the r4 BUNDLE loaded, but the
# r4 pair build smoke-skipped -> per_model_r4_pair lacks the model.
matched = {"shared_r1r2_convs": convs, "per_model_r3_pair": {}, "per_model_r4_pair": {}}
root = Path(tempfile.mkdtemp(prefix="i1345_transfer_probe_"))
out, preds = root / "out", root / "preds"
out.mkdir(); preds.mkdir()
run_model_arm(
    bundles, matched, "instruct", "context", out, preds,
    seed=0, null_draws=0, n_boot=8, include_r3=False, include_r4=True, smoke=True,
)
payload = json.loads((out / "cross_regime_transfer_instruct_context.json").read_text())
print(json.dumps({
    "r4_skips": sorted(k for k in payload["skipped_pairs"] if "r4" in k),
    "r4_skip_reasons": sorted(set(
        v for k, v in payload["skipped_pairs"].items() if "r4" in k
    )),
    "r4_pair_meta": payload["matched_n"]["r4_pair"],
    "matrix_has_r4": any("r4" in k for k in payload["matrix"]),
}))
"""


def test_transfer_matrix_skips_r4_pairs_when_pair_build_skipped():
    """r1 code-review Critical pin: run_model_arm with an r4 bundle + a matched
    dict whose per_model_r4_pair lacks the model, smoke=True — no exception,
    logged skips (pre-fix: TypeError 'NoneType' object is not subscriptable at
    _subset's r4cfg["r4_convs"] dereference)."""
    proc = _run_py(_TRANSFER_DEMOTED_R4_PROBE, {"EPM_I1345_VARIANT": CPS})
    assert proc.returncode == 0, f"stderr:\n{proc.stderr}"
    out = json.loads(proc.stdout.strip().splitlines()[-1])
    assert out["r4_skips"] == ["r1->r4", "r2->r4", "r4->r1", "r4->r2"], out
    assert all("include_r4 demoted" in r for r in out["r4_skip_reasons"]), out
    assert out["r4_pair_meta"] is None and not out["matrix_has_r4"]
    assert "[transfer] SKIP pair" in proc.stdout


def test_pair_kind_for_grains():
    assert pair_kind_for("r1", "r2") == "headline"
    assert pair_kind_for("r1", "r3") == "r3pair"
    assert pair_kind_for("r1", "r4") == "r4pair"
    assert pair_kind_for("r2", "r4") == "r4pair"
    with pytest.raises(AssertionError):
        pair_kind_for("r3", "r4")


def _write_sidecar(ts_dir: Path, stem: str, conv_ids: list[str]) -> None:
    ts_dir.mkdir(parents=True, exist_ok=True)
    (ts_dir / f"{stem}_shard0.json").write_text(json.dumps({"conv_ids": conv_ids}))


def _seed_r1r2_sidecars(ts_dir: Path, shared: list[str]) -> None:
    for model in c.MODELS:
        for regime in ("r1", "r2"):
            _write_sidecar(ts_dir, c.stem_for(model, regime), shared)


def test_build_matched_r4_pair_production_and_smoke(tmp_path):
    shared = [f"s{i}" for i in range(6)]
    ts = tmp_path / "ts"
    _seed_r1r2_sidecars(ts, shared)
    _write_sidecar(ts, c.stem_for("instruct", "r4"), shared[:4])
    _write_sidecar(ts, c.stem_for("instruct", "r4op"), shared[:2])
    out = build_matched(ts, tmp_path / "m", r3_models=set(), r4_models={"instruct"})
    entry = out["per_model_r4_pair"]["instruct"]
    assert entry["r4_convs"] == sorted(shared[:4]) and entry["n"] == 4
    assert entry["op_companion_convs"] == sorted(shared[:2]) and entry["n_op"] == 2
    assert entry["companion"] == "present"

    # No r4op store at all (rc=23 halt lane) -> explicit companion: halted record
    ts2 = tmp_path / "ts2"
    _seed_r1r2_sidecars(ts2, shared)
    _write_sidecar(ts2, c.stem_for("instruct", "r4"), shared[:4])
    out2 = build_matched(ts2, tmp_path / "m_halt", r3_models=set(), r4_models={"instruct"})
    e2 = out2["per_model_r4_pair"]["instruct"]
    assert e2["companion"] == "halted"
    assert e2["op_companion_convs"] is None and e2["n_op"] == 0

    # Foreign conv -> production fail-loud; smoke informational skip
    _write_sidecar(ts, c.stem_for("instruct", "r4"), [*shared[:3], "foreign9"])
    with pytest.raises(RuntimeError, match=r"foreign|drift"):
        build_matched(ts, tmp_path / "m2", r3_models=set(), r4_models={"instruct"})
    out_smoke = build_matched(
        ts, tmp_path / "m3", r3_models=set(), r4_models={"instruct"}, smoke=True
    )
    assert out_smoke["per_model_r4_pair"] == {}

    # Duplicate conv ids (multi-row story) violate the one-story-per-conv contract
    _write_sidecar(ts, c.stem_for("instruct", "r4"), [shared[0], shared[0], shared[1]])
    with pytest.raises(AssertionError, match="duplicate conv_ids"):
        build_matched(ts, tmp_path / "m4", r3_models=set(), r4_models={"instruct"})


def test_matched_row_cells_registry_and_allowlists():
    r4cfg = {
        "n": 4,
        "r4_convs": ["s0", "s1", "s2", "s3"],
        "op_companion_convs": ["s0", "s1"],
        "n_op": 2,
    }
    cells, allow = matched_row_cells(r4cfg)
    ids = sorted(x["cell_id"] for x in cells)
    assert ids == [
        "R_instruct_r1_matched_context",
        "R_instruct_r1_matched_prefix",
        "R_instruct_r2_matched_context",
        "R_instruct_r2_matched_prefix",
        "R_instruct_r4_tf_on_companion_context",
        "R_instruct_r4_tf_on_companion_prefix",
    ]
    assert allow["R_instruct_r1_matched_context"] == r4cfg["r4_convs"]
    assert allow["R_instruct_r4_tf_on_companion_context"] == r4cfg["op_companion_convs"]
    # No companion store -> TF-on-companion cells absent (calibration N/A)
    cells_no_op, _ = matched_row_cells({**r4cfg, "op_companion_convs": None})
    assert not any("companion" in x["cell_id"] for x in cells_no_op)


# ---------------------------------------------------------------------------
# TF-distortion nested tiers (plan v8 §7 — reporting labels)
# ---------------------------------------------------------------------------
def _fake_cells(path: Path, r2_l19: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "r2_per_layer_obs": [0.0] * 19 + [r2_l19] + [0.0] * 8,
                "r2_bootstrap_ci_frozen_layers_conv": {
                    "19": {"ci_lo": r2_l19 - 0.02, "ci_hi": r2_l19 + 0.02, "n_groups": 100}
                },
                "n_rows": 100,
            }
        )
    )


@pytest.mark.parametrize(
    ("tf_sub", "op", "tier1", "tier2"),
    [
        (0.50, 0.48, False, False),  # gap 0.02 <= 0.05: clean
        (0.50, 0.40, True, False),  # gap 0.10 > 0.05: qualification only
        (0.30, -0.05, True, True),  # op negative AND gap 0.35 > 0.20: TF-DISTORTED
        (0.10, -0.05, True, False),  # op negative but gap 0.15 <= 0.20: tier1 only
    ],
)
def test_tf_op_calibration_tiers(tmp_path, tf_sub, op, tier1, tier2):
    eval_dir = tmp_path / "eval"
    matched_out = tmp_path / "eval" / "matched_row"
    _fake_cells(eval_dir / "cells_R_instruct_r4_context.json", 0.55)
    _fake_cells(eval_dir / "cells_R_instruct_r4_op_companion_context.json", op)
    _fake_cells(matched_out / "cells_R_instruct_r4_tf_on_companion_context.json", tf_sub)
    _fake_cells(matched_out / "cells_R_instruct_r1_matched_context.json", 0.60)
    _fake_cells(eval_dir / "cells_R_instruct_r1_context.json", 0.67)
    tf_op_calibration(eval_dir, matched_out, smoke=False)
    payload = json.loads((matched_out / "tf_op_calibration.json").read_text())
    cal = payload["calibration"]
    assert cal["tier1_qualification"] is tier1 and cal["tier2_tf_distorted"] is tier2
    assert abs(cal["tf_minus_op_gap_matched_subset"] - (tf_sub - op)) < 1e-9
    assert payload["r1_subset_vs_full"]["subset_leq_full"] is True


def test_tf_op_calibration_companion_halted_skips(tmp_path):
    eval_dir = tmp_path / "eval"
    matched_out = tmp_path / "eval" / "matched_row"
    matched_out.mkdir(parents=True)
    _fake_cells(eval_dir / "cells_R_instruct_r4_context.json", 0.55)
    # No companion cell + matched-recorded halt (rc=23 lane) -> production
    # tolerates with an EXPLICIT companion: halted record (r1 review Major)
    tf_op_calibration(eval_dir, matched_out, smoke=False, companion_halted=True)
    payload = json.loads((matched_out / "tf_op_calibration.json").read_text())
    assert "skipped" in payload["calibration"]
    assert payload["calibration"]["companion"] == "halted"
    # Missing cells beside a LIVE companion in production = pipeline drift ->
    # fail loud (never a silent skip)
    with pytest.raises(AssertionError, match="drift"):
        tf_op_calibration(
            eval_dir, tmp_path / "eval2" / "matched_row", smoke=False, companion_halted=False
        )
