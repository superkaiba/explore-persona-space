"""Pins for the issue #2378 r14 code-review-bounce fixes (r13 blockers B1-B3).

B1 — real/sim paired-context identity (BOTH r13 reviewers; concern ids
   user-pair-vc-assert-guaranteed-fail / user-arm-context-identity-contract-
   broken): the real arm's teacher-forced render is now the DIRECT JOIN
   ``gen._render_user_prefix + u2 + TURN_END`` (``gen._render_user_real_tf``),
   never the template's own 3-turn render (which strips a1's empty
   ``<think>`` block and so shifts every context byte). Pins: byte-identical
   context prefixes AND equal v_C/v_P anchor arithmetic (and header-end token
   ids) across ``cap._assemble_user_real`` / ``cap._assemble_user_sim``,
   through the production ``cap._tokenize_and_positions`` body.

B2 — single surviving user arm (Codex concern single-user-survivor-crashes-
   p6): a store where one user arm G2b-dropped below floor is a LEGAL plan-§7
   topology (report-never-kill). Pins: fold map builds a labeled single-arm
   entry, the own fit + chat→survivor transfer ladder succeed, H4b emits a
   loud ``status: N/A`` record, and ``cap._capture_ready_ids`` falls back to
   the survivor's OWN kept ids instead of the starved pair intersection.

B3 — monotonic kept-ledger EXTEND (Codex concern admission-kept-rewrite-can-
   shrink): the judge cache is VM-local, so a cold-cache wave-2 re-judge can
   flip wave-1 verdicts; ``judge._merge_kept_ledger`` makes the kept rewrite
   the prior/new UNION (prior list an exact prefix, prior scores kept), refusing loud
   on instrument drift or a prior id outside the mined union. Pins: unit
   tests on the merge + a functional ``judge._run_admission`` run with a
   signature-conformant fake dispatch (#906: only the network boundary is
   faked; ShardWriter / merge / atomic writes run their real bodies).

Real-template tests skip when the pinned tokenizer is unavailable
(no network/cache in CI — the test_issue1482_kresample precedent).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import issue2378_capture as cap  # noqa: E402
import issue2378_common as cm  # noqa: E402
import issue2378_fits as fits  # noqa: E402
import issue2378_gen as gen  # noqa: E402
import issue2378_judge as judge  # noqa: E402
import issue2378_ladder as ladder  # noqa: E402
import issue2378_p6_common as p6  # noqa: E402

POOL_ROW = {
    "conv_id": "mt_r14pin000001",
    "u1": "Can you explain what a mutex is in one paragraph?",
    "a1": "A mutex is a lock that lets only one thread enter a critical "
    "section at a time, so shared state cannot be mutated concurrently.",
    "u2": "Thanks! And how does that differ from a semaphore?\nA short example would help.",
    "depth": 2,
}
SIM_TURN = "Could you also compare it with a spinlock, please?"


def _tok():
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(cm.MODEL_ID)
    except Exception as e:  # no network/cache in CI
        pytest.skip(f"tokenizer unavailable: {e}")


def _sim_row(prefix: str) -> dict:
    """Writer-real sim producer row shape (prefix pin + producer-stripped turn)."""
    return {
        "prefix_chars": len(prefix),
        "prefix_digest": cm.text_digest(prefix),
        "sim_turn": SIM_TURN,
    }


# ── B1: byte-identical paired context across the two user-arm assemblers ─────


def test_user_arms_share_byte_identical_context_prefix():
    tok = _tok()
    prefix = gen._render_user_prefix(tok, POOL_ROW["u1"], POOL_ROW["a1"])
    real_row = gen._user_real_row(tok, POOL_ROW)
    assert real_row["keep"] is True
    real_payload, r_reason = cap._assemble_user_real(tok, real_row, POOL_ROW)
    sim_payload, s_reason = cap._assemble_user_sim(tok, _sim_row(prefix), POOL_ROW)
    assert r_reason is None and s_reason is None
    r_ctx = real_payload["final_text"][: real_payload["answer_lo_char"]]
    s_ctx = sim_payload["final_text"][: sim_payload["answer_lo_char"]]
    assert r_ctx == s_ctx == prefix  # §4.2b: byte-identical context prefix
    assert real_payload["answer_lo_char"] == sim_payload["answer_lo_char"] == len(prefix)
    assert real_payload["prefix_char"] == sim_payload["prefix_char"]  # shared v_P anchor
    # The real arm's answer slice is exactly u2 (teacher-forced through u2 end).
    lo, hi = real_payload["answer_lo_char"], real_payload["answer_hi_char"]
    assert real_payload["final_text"][lo:hi] == POOL_ROW["u2"]


def test_user_arms_equal_vc_anchor_and_header_end_token_ids():
    """The assert_user_pair guarantee, pinned at the token level: identical
    prefix bytes => identical v_C/v_P token indices AND identical input ids up
    to the v_C token, so the captured v_C/v_P vectors are byte-identical under
    a teacher-forced causal forward."""
    tok = _tok()
    prefix = gen._render_user_prefix(tok, POOL_ROW["u1"], POOL_ROW["a1"])
    real_row = gen._user_real_row(tok, POOL_ROW)
    real_payload, _ = cap._assemble_user_real(tok, real_row, POOL_ROW)
    sim_payload, _ = cap._assemble_user_sim(tok, _sim_row(prefix), POOL_ROW)
    rows = [
        {"row_id": "conv0", "prov": {}, **real_payload},
        {"row_id": "conv0", "prov": {}, **sim_payload},
    ]
    kept, drops = cap._tokenize_and_positions(tok, rows, max_tokens=4096)
    assert len(kept) == 2, dict(drops)
    real_k, sim_k = kept
    assert real_k["v_C_pos"] == sim_k["v_C_pos"]
    assert real_k["v_P_pos"] == sim_k["v_P_pos"]
    vc = real_k["v_C_pos"]
    assert real_k["input_ids"][: vc + 1] == sim_k["input_ids"][: vc + 1]


def test_stale_r13_template_render_row_fails_visible():
    """A row produced by the r13 template-render producer must be dropped
    fail-visible by the r14 capture consumer (prefix_render_mismatch), never
    silently consumed with drifted context bytes."""
    tok = _tok()
    r = POOL_ROW
    r13_render = tok.apply_chat_template(
        [
            {"role": "user", "content": r["u1"]},
            {"role": "assistant", "content": r["a1"]},
            {"role": "user", "content": r["u2"]},
        ],
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False,
    )
    span = gen._user_real_span(r13_render, r["u2"])
    assert span is not None  # the tail anchor resolves on the r13 render too
    stale = {
        "cell": "chat_user_real",
        "conv_id": r["conv_id"],
        "rendered_text": r13_render,
        "header_end": span[0],
        "u2_span": list(span),
        "keep": True,
        "drop_reason": None,
    }
    payload, reason = cap._assemble_user_real(tok, stale, r)
    assert payload is None and reason == "prefix_render_mismatch"


# ── B2: single surviving user arm — fold map / fits / ladder / H4b ──────────


def _single_arm_ns(store: Path, ledger: Path) -> argparse.Namespace:
    return argparse.Namespace(
        store_root=str(store),
        ledger_root=str(ledger),
        layer=1,
        layer_star_from=None,
        n_null_draws=6,
        bootstrap_draws=24,
        reduced_k=4,
        units="own:chat:context,own:chat_user_real:context",
        g3_gate_file=None,
        pairs="chat_user_real",
        survivors=None,
        fold_floors_override=fits._PROBE_FLOORS,
    )


def test_single_user_survivor_runs_end_to_end(tmp_path):
    store, ledger = tmp_path / "store", tmp_path / "ledger"
    fits._write_probe_store(store, n=40, d=8)
    for p in store.glob("chat_user_sim__part*"):
        p.unlink()  # sim arm G2b-dropped below floor: absent from the store
    ledger.mkdir()

    # Fold map: labeled single-arm entry, own cohort, no raise.
    fm = p6.load_or_build_fold_map(store, ledger, **fits._PROBE_FLOORS)
    ui = fm["user_intersection"]
    assert ui["single_arm"] == "chat_user_real"
    assert ui["dropped_arm"] == "chat_user_sim"
    assert ui["n_intersection"] is None and ui["n_kept"] == 37
    assert "chat_user_sim" not in fm["cells"]
    entry = fm["cells"]["chat_user_real"]
    assert entry.get("single_user_arm") is True
    assert len(entry["row_ids"]) == min(fm["n_eq"], 37)

    # Own fit through the production entrypoints (G3 gate + fit unit).
    ns = _single_arm_ns(store, ledger)
    assert fits.phase_g3(ns) == 0
    assert fits.phase_fit(ns) == 0
    ureal = json.loads((ledger / "fits" / "chat_user_real__context.json").read_text("utf-8"))
    assert ureal["single_user_arm"] is True
    assert ureal["intersection"]["single_arm"] == "chat_user_real"
    assert "full_cohort_supplementary" in ureal  # labeled supplementary row

    # Transfer ladder onto the survivor succeeds (no pair assert on arm reads).
    assert ladder.phase_pairs(ns) == 0
    rung1 = json.loads(
        (ledger / "ladder" / "chat_to_chat_user_real__rung1.json").read_text("utf-8")
    )
    assert rung1["recovery"]["point_pooled"] > 0.8  # shared planted geometry
    assert (ledger / "ladder" / "chat_to_chat_user_real__rung9.json").exists()

    # H4b: loud N/A record, rc 0, never a crash.
    assert ladder.phase_h4b(ns) == 0
    h4b = json.loads((ledger / "ladder" / "h4b_real_vs_sim.json").read_text("utf-8"))
    assert h4b["status"] == "N/A"
    assert h4b["reason"] == "H4b = N/A — other user arm dropped below floor"
    assert h4b["missing_user_arms"] == ["chat_user_sim"]
    assert h4b["surviving_user_arms"] == ["chat_user_real"]


def test_capture_ready_single_arm_falls_back_to_own_kept_ids(tmp_path):
    ledger = tmp_path / "ledger"
    cr = ledger / "capture_ready"
    cr.mkdir(parents=True)
    real_ids = sorted(f"conv{i:04d}" for i in range(10))
    sim_ids = real_ids[:4]
    inter = sorted(set(real_ids) & set(sim_ids))

    def _gate(cell: str, ids: list[str], floor_pass: bool) -> None:
        (cr / f"{cell}.json").write_text(
            json.dumps(
                {
                    "cell": cell,
                    "floor_pass": floor_pass,
                    "kept_ids": ids,
                    "pair_intersection": {"intersection_ids": inter},
                }
            ),
            encoding="utf-8",
        )

    args = SimpleNamespace(ledger_root=str(ledger))
    # Sibling below floor: the survivor captures its OWN kept ids.
    _gate("chat_user_real", real_ids, True)
    _gate("chat_user_sim", sim_ids, False)
    assert cap._capture_ready_ids(args, "chat_user_real") == set(real_ids)
    # Both arms passing: the pair intersection (unchanged r13 behavior).
    _gate("chat_user_sim", sim_ids, True)
    assert cap._capture_ready_ids(args, "chat_user_real") == set(inter)
    # Missing sibling gate stays fail-loud.
    (cr / "chat_user_sim.json").unlink()
    with pytest.raises(RuntimeError, match="sibling capture_ready"):
        cap._capture_ready_ids(args, "chat_user_real")


# ── B3: monotonic kept-ledger EXTEND ─────────────────────────────────────────

CELL = "storyq_astra"


def _prior(admitted, rubric_sha=None, judge_model=None) -> dict:
    return {
        "cell": CELL,
        "family": cm.CELL_FAMILY[CELL],
        "n_items": len(admitted),
        "n_admitted": len(admitted),
        "admit_threshold": 50,
        "drop_counts": {},
        "admitted": admitted,
        "judge_model": judge_model or cm.JUDGE_MODEL,
        "rubric_sha": rubric_sha or judge._rubric_sha(),
    }


def test_merge_kept_ledger_no_prior_passthrough():
    new = [{"row_id": "r001", "score": 90}]
    merged, extend = judge._merge_kept_ledger(None, new, {"r001"}, "sha", "model")
    assert merged == new
    assert extend == {
        "n_prior": 0,
        "n_new_admitted": 1,
        "n_new_only": 1,
        "n_merged": 1,
        "n_prior_readmitted": 0,
        "n_prior_not_readmitted": 0,
    }


def test_merge_kept_ledger_preserves_flipped_prior_admission():
    prior = _prior([{"row_id": "r000", "score": 80}])
    new = [{"row_id": "r001", "score": 90}]  # cold cache flipped r000 below 50
    merged, extend = judge._merge_kept_ledger(
        prior, new, {"r000", "r001"}, judge._rubric_sha(), cm.JUDGE_MODEL
    )
    assert [r["row_id"] for r in merged] == ["r000", "r001"]  # prior is an exact prefix
    assert merged[0]["score"] == 80  # prior score kept, never the flipped verdict
    assert extend["n_prior_not_readmitted"] == 1
    assert extend["n_merged"] == 2 and extend["n_new_only"] == 1


def test_merge_kept_ledger_refuses_instrument_drift():
    prior = _prior([{"row_id": "r000", "score": 80}], rubric_sha="0" * 16)
    with pytest.raises(RuntimeError, match=r"different\s+instrument"):
        judge._merge_kept_ledger(prior, [], {"r000"}, judge._rubric_sha(), cm.JUDGE_MODEL)


def test_merge_kept_ledger_refuses_prior_outside_mined_union():
    prior = _prior([{"row_id": "ghost", "score": 80}])
    with pytest.raises(RuntimeError, match="outside this run's mined union"):
        judge._merge_kept_ledger(prior, [], {"r000"}, judge._rubric_sha(), cm.JUDGE_MODEL)


def _mined(rids: list[str]) -> dict[str, dict]:
    return {
        rid: {
            "cell": CELL,
            "family": cm.CELL_FAMILY[CELL],
            "character": "Astra",
            "scene_pre_answer": "A scene.",
            "utterance": "An utterance.",
        }
        for rid in rids
    }


def _adm_args(tmp_path: Path) -> SimpleNamespace:
    # max_items <= SMOKE_MAX_ITEMS_WITHOUT_PILOT + sync => pilot gate skipped.
    return SimpleNamespace(
        out_root=str(tmp_path / "ledger"),
        raw_root=str(tmp_path / "raw"),
        wave="admission",
        transport="sync",
        max_items=10,
        skip_upload=True,
    )


def test_run_admission_cold_cache_flip_is_preserved(tmp_path, monkeypatch):
    """The brief's pin: seed an old kept ledger, re-run the reducer with a cold
    fake cache that flips one old verdict -> the old admitted id remains
    admitted. Only the network dispatch is faked (signature-conformant def,
    #906); the merge, ShardWriter and atomic writes run their real bodies."""
    args = _adm_args(tmp_path)
    kept_dir = Path(args.out_root) / "kept"
    kept_dir.mkdir(parents=True)
    (kept_dir / f"{CELL}.json").write_text(
        json.dumps(_prior([{"row_id": "r000", "score": 80}])), encoding="utf-8"
    )
    scores = {"adm|r000": 20, "adm|r001": 90}  # cold cache: r000 flips below 50

    def fake_dispatch(items, args, force_path, cache_tag):
        return {
            it.item_id: {
                "class": "valid",
                "score": scores[it.item_id],
                "reasoning": "probe",
                "stop_reason": "end_turn",
            }
            for it in items
        }

    monkeypatch.setattr(judge, "_dispatch", fake_dispatch)
    assert judge._run_admission(args, _mined(["r000", "r001"])) == 0
    ledger = json.loads((kept_dir / f"{CELL}.json").read_text("utf-8"))
    assert [r["row_id"] for r in ledger["admitted"]] == ["r000", "r001"]
    assert ledger["admitted"][0]["score"] == 80  # wave-1 admission preserved
    assert ledger["n_admitted"] == 2
    assert ledger["extend"] == {
        "n_prior": 1,
        "n_new_admitted": 1,
        "n_new_only": 1,
        "n_merged": 2,
        "n_prior_readmitted": 0,
        "n_prior_not_readmitted": 1,
    }
    assert ledger["drop_counts"] == {"below_threshold": 1}  # this invocation's flip


def test_run_admission_instrument_drift_raises_before_replacement(tmp_path, monkeypatch):
    args = _adm_args(tmp_path)
    kept_dir = Path(args.out_root) / "kept"
    kept_dir.mkdir(parents=True)
    prior_text = json.dumps(_prior([{"row_id": "r000", "score": 80}], rubric_sha="0" * 16))
    (kept_dir / f"{CELL}.json").write_text(prior_text, encoding="utf-8")

    def fake_dispatch(items, args, force_path, cache_tag):
        return {
            it.item_id: {
                "class": "valid",
                "score": 90,
                "reasoning": "probe",
                "stop_reason": "end_turn",
            }
            for it in items
        }

    monkeypatch.setattr(judge, "_dispatch", fake_dispatch)
    with pytest.raises(RuntimeError, match=r"different\s+instrument"):
        judge._run_admission(args, _mined(["r000", "r001"]))
    # No write happened: the prior ledger is byte-untouched.
    assert (kept_dir / f"{CELL}.json").read_text("utf-8") == prior_text
