"""#1947 crash-fix r8 pins: resume predicates honor terminal-verdict sidecars.

P0 crash 4: on relaunch, ``phase_positives`` re-entered the factory stage for
a cell whose ONE allowed topup tranche was already consumed with
``union_floor_missed=true`` recorded in ``topup_record.json`` — the stage's
one-tranche guard raised, turning a recorded, survivable verdict into a hard
crash. The r8 fix: ``phase_positives`` recognizes the recorded terminal
verdict BEFORE entering the factory and re-applies the salvage disposition
(the sidecar-reconstructed kept pool); the guard itself and the 240
hard-floor mixes-time adjudication are byte-unchanged. The salvage
reconstruction is rebuilt on the sidecars that actually EXIST at floor-miss
time (``raw_pos[_topup].jsonl`` + ``judge_raw_pos[_topup].json`` replayed
through the factory's own keep rule) — the pre-r8 read of
``judge_rows.jsonl`` targeted a sidecar the factory writes only AFTER the
floor check passes (silent 0-row salvage), and its ``*topup*judge_rows*``
globs matched neither actual sidecar name.

Resume matrix (one row per phase carrying a terminal-verdict sidecar and/or
one-shot guard; the full 7-phase sweep table lives in the r8 implementation
marker on task #1947):

- positives: ``topup_record.json`` (union_floor_missed=true) x the
  ``_positive_topup_stage`` one-tranche guard -> FIXED (phase re-entry
  re-applies the disposition; the guard still raises on direct stage
  re-entry — science gate unchanged).
- negatives: the ``--negatives-extra`` retry remedy (named by the shortfall
  error) is now consumable on resume via ``_neg_raw_delta`` (widened
  question set -> delta-only generation over the resumed raw sidecar).
- mixes: ``yield_gate.json`` is a recomputed-each-entry verdict with NO
  one-shot guard -> re-entry re-derives the same verdict and skips built
  cells (both PASS- and DROP-verdict fixtures).

Fakes sit only at external boundaries (tokenizer = model boundary, backend =
vLLM boundary), signature-conformant per the existing #1947 test convention
(``test_issue1947_marker_topup._FakeTok``); the judge sidecars are hand-built
in the REAL ``save_raw`` shape (``all_scores`` custom-id keys) so the REAL
``judge_result_from_save_raw`` + ``_judge_and_filter`` bodies execute.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import issue1947_datagen as dg  # noqa: E402

from explore_persona_space.artifacts.behavior import BEHAVIORS  # noqa: E402
from explore_persona_space.artifacts.datagen import (  # noqa: E402
    POSITIVE,
    GenCandidate,
    GenRequest,
    TopupSpec,
    _positive_topup_stage,
    _write_raw,
)

N_DRAWS = dg.N_JUDGE_DRAWS


class _FakeTok:
    """Signature-conformant tokenizer fake (external model boundary only) —
    the ``test_issue1947_marker_topup`` shape: template + encode, word-grain."""

    def _ids(self, text: str) -> list[int]:
        return [(hash(w) % 50_000) + 1 for w in text.split()]

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        text = "\n".join(str(m.get("content", "")) for m in messages)
        if add_generation_prompt:
            text += "\nassistant:"
        return self._ids(text) if tokenize else text

    def encode(self, text, add_special_tokens=False):
        return self._ids(text)


class _RecordingBackend:
    """vLLM-boundary fake mirroring ``_MockBackend``'s generate/close surface;
    records prompt batches so delta-only generation is assertable."""

    def __init__(self):
        self.calls: list[list[str]] = []

    def generate(self, prompts: list[str], max_new: int, *, adapter_dir=None) -> list[str]:
        self.calls.append(list(prompts))
        return [f"A plain stub negative answer {i}." for i in range(len(prompts))]

    def close(self, label: str) -> None:
        return


def _cfg(tmp_path: Path, **kw) -> dg.Cfg:
    defaults = dict(
        out_root=tmp_path,
        phases=(),
        behaviors=("syc",),
        contexts=("icl",),
        smoke=True,
        mock_gen=True,
    )
    defaults.update(kw)
    return dg.Cfg(**defaults)


def _cand(rid: str, qid: str, question: str) -> GenCandidate:
    msgs = [{"role": "user", "content": question}]
    req = GenRequest(rid, POSITIVE, qid, "v0", question, msgs, msgs)
    return GenCandidate(req, f"Answer body for {rid}.")


def _save_raw(path: Path, scores: dict[str, float]) -> None:
    """The REAL judge save_raw shape: all_scores keyed by
    ``{item_id}__{idx:05d}__{comp:02d}`` custom ids, one entry per draw."""
    all_scores = {
        f"{rid}__00000__{d:02d}": {"score": s} for rid, s in scores.items() for d in range(N_DRAWS)
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"all_scores": all_scores}), encoding="utf-8")


def _seed_terminal_topup_state(out_dir: Path) -> dict:
    """The P0-crash-4 pod state: tranche consumed, union floor missed, NO
    pos.jsonl / judge_rows.jsonl (the factory writes those only post-floor),
    raw + judge_raw sidecars present for both samples."""
    out_dir.mkdir(parents=True, exist_ok=True)
    first = [_cand(f"p{i}", f"q{i}", f"First question {i}?") for i in range(4)]
    _write_raw(out_dir / "raw_pos.jsonl", first)
    # threshold 50: p0/p1/p3 kept (85 > 50), p2 dropped (10) -> kept_first = 3
    _save_raw(out_dir / "judge_raw_pos.json", {"p0": 85, "p1": 85, "p2": 10, "p3": 85})
    topup = [
        _cand("tp0", "q0", "First question 0?"),  # kept, qid-dedup dropped vs p0
        _cand("tp1", "q4", "Topup question 4?"),  # kept + merged
        _cand("tp2", "q5", "Topup question 5?"),  # judged below threshold
    ]
    _write_raw(out_dir / "raw_pos_topup.jsonl", topup)
    _save_raw(out_dir / "judge_raw_pos_topup.json", {"tp0": 85, "tp1": 85, "tp2": 10})
    record = {
        "fired": True,
        "tranche_requested": 3,
        "tranche_kept": 2,
        "tranche_merged": 1,
        "tranche_dedup_dropped_qid": 1,
        "kept_pos_first_sample": 3,
        "kept_pos_union": 4,
        "trigger_below_n": 5,
        "floor_n": 4,
        "seed_offset": 7919,
        "union_floor_missed": True,
    }
    (out_dir / "topup_record.json").write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8"
    )
    return record


# ── positives: terminal-verdict resume (THE r8 fix; fails pre-fix with the
#    one-tranche guard RuntimeError) ─────────────────────────────────────────


def test_phase_positives_resumes_terminal_topup_verdict_without_reentering_stage(tmp_path, caplog):
    cfg = _cfg(tmp_path)
    dg._write_json(
        cfg.banks_dir / "sycophancy_extended.json",
        {"new_questions": [f"Extended bank question {i}?" for i in range(12)]},
    )
    out_dir = cfg.positives_dir / "syc-icl"
    record = _seed_terminal_topup_state(out_dir)
    with caplog.at_level(logging.WARNING, logger="issue1947.datagen"):
        dg.phase_positives(cfg)  # pre-r8: RuntimeError "EXACTLY ONE tranche"
    pos = dg._read_jsonl(out_dir / "pos.jsonl")
    assert len(pos) == 4  # min(emit_n=4 smoke, union=4)
    meta = json.loads((out_dir / "salvage_meta.json").read_text(encoding="utf-8"))
    assert meta["kept_pos_union"] == record["kept_pos_union"] == 4
    assert meta["record_checked"] is True
    results = json.loads((cfg.positives_dir / "phase_positives.json").read_text(encoding="utf-8"))
    assert results["pools"]["syc-icl"]["resumed_terminal_topup"] is True
    assert results["pools"]["syc-icl"]["emitted"] == 4
    # fix-engaged signal: the recorded-disposition branch's log line
    joined = "\n".join(r.getMessage() for r in caplog.records)
    assert "topup_record.json terminal verdict" in joined
    assert "without re-entering the factory stage" in joined


def test_phase_positives_terminal_resume_emits_deduped_union(tmp_path):
    cfg = _cfg(tmp_path)
    dg._write_json(
        cfg.banks_dir / "sycophancy_extended.json",
        {"new_questions": [f"Extended bank question {i}?" for i in range(12)]},
    )
    out_dir = cfg.positives_dir / "syc-icl"
    _seed_terminal_topup_state(out_dir)
    dg.phase_positives(cfg)
    completions = {r["completion"][0]["content"] for r in dg._read_jsonl(out_dir / "pos.jsonl")}
    # union = first-sample kept (p0, p1, p3) + merged tranche (tp1);
    # tp0 (kept, duplicate question_id q0) and the judged-drops never emit.
    assert completions == {f"Answer body for {rid}." for rid in ("p0", "p1", "p3", "tp1")}


# ── positives: the factory one-tranche guard is byte-unchanged ──────────────


def test_factory_one_tranche_guard_still_raises_on_direct_stage_reentry(tmp_path):
    (tmp_path / "topup_record.json").write_text(
        json.dumps({"union_floor_missed": True}), encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="EXACTLY ONE tranche"):
        _positive_topup_stage(
            BEHAVIORS["sycophancy"],
            None,  # context unused before the guard raise
            [],
            TopupSpec(tranche_n=8),
            target_n=5,
            floor_n=4,
            seed=0,
            instruction_style="plain",
            variants=None,
            gen_factory=None,  # never reached — the guard raises first
            judge=None,
            n_judge_draws=N_DRAWS,
            judge_cache=tmp_path,
            out_dir=tmp_path,
        )


# ── positives: salvage reconstruction unit pins ──────────────────────────────


def test_salvage_emit_reconstructs_union_with_qid_dedup(tmp_path):
    out_dir = tmp_path / "syc-icl"
    record = _seed_terminal_topup_state(out_dir)
    n = dg._salvage_emit(BEHAVIORS["sycophancy"], out_dir, 300, 7, record=record)
    assert n == 4  # min(emit_n=300, union=4) — the recorded disposition
    meta = json.loads((out_dir / "salvage_meta.json").read_text(encoding="utf-8"))
    assert meta["kept_pos_first_sample"] == 3
    assert meta["tranche_kept"] == 2
    assert meta["tranche_merged"] == 1  # tp0 qid-dedup dropped (mirrors the factory merge)
    assert meta["kept_pos_union"] == 4


def test_salvage_emit_record_mismatch_fails_loud(tmp_path):
    out_dir = tmp_path / "syc-icl"
    record = _seed_terminal_topup_state(out_dir)
    record["kept_pos_union"] = 99  # drifted sidecars vs the recorded verdict
    with pytest.raises(RuntimeError, match="reconstruction mismatch"):
        dg._salvage_emit(BEHAVIORS["sycophancy"], out_dir, 300, 7, record=record)


def test_salvage_emit_missing_sidecars_fails_loud(tmp_path):
    out_dir = tmp_path / "syc-icl"
    out_dir.mkdir(parents=True)
    with pytest.raises(RuntimeError, match="sidecars missing"):
        dg._salvage_emit(BEHAVIORS["sycophancy"], out_dir, 300, 7)


# ── negatives: the --negatives-extra retry remedy is consumable on resume ────


def test_neg_raw_delta_widened_prefix_and_drift():
    questions = [f"Question {i}?" for i in range(6)]
    rows = [{"question": q} for q in questions[:4]]
    assert dg._neg_raw_delta(rows, questions, "syc-panel5", "m") == questions[4:]
    # superset sidecar (shrunken regime): empty delta, rows used as-is
    assert dg._neg_raw_delta([{"question": q} for q in questions], questions[:4], "p", "m") == []
    with pytest.raises(RuntimeError, match="not a prefix"):
        dg._neg_raw_delta([{"question": "Drifted?"}], questions, "syc-panel5", "m")


def test_phase_negatives_widened_retry_generates_delta_only(tmp_path, monkeypatch):
    bank = [f"Negative bank question {i:03d}?" for i in range(100)]
    cfg = _cfg(tmp_path, smoke=False)  # smoke pins n=2, so the widening needs full mode
    dg._write_json(cfg.banks_dir / "sycophancy_extended.json", {"new_questions": bank})
    monkeypatch.setattr(dg, "_tokenizer", lambda: _FakeTok())
    backend = _RecordingBackend()
    monkeypatch.setattr(dg, "_backend", lambda cfg: backend)
    dg.phase_negatives(cfg)  # first pass: n = ceil(60 * 1.25) = 75 questions/member
    pool_dir = cfg.negatives_dir / "syc-panel5"
    raw_files = sorted(pool_dir.glob("raw_*.jsonl"))
    assert raw_files and all(len(dg._read_jsonl(p)) == 75 for p in raw_files)
    # the shortfall-retry shape: neg.jsonl absent + a widened question set
    for variant in dg.PANEL_VARIANTS:
        (cfg.negatives_dir / f"syc-{variant}" / "neg.jsonl").unlink()
    backend.calls.clear()
    cfg2 = _cfg(tmp_path, smoke=False, negatives_extra=5)
    dg.phase_negatives(cfg2)
    assert all(len(dg._read_jsonl(p)) == 80 for p in sorted(pool_dir.glob("raw_*.jsonl")))
    # delta-only generation: every generate call on the resumed pool is the
    # 5-question widening, never a 75/80-question regeneration
    assert backend.calls and all(len(c) in (5, 80) for c in backend.calls)
    assert any(len(c) == 5 for c in backend.calls)


# ── mixes: yield_gate.json verdict re-derives idempotently on re-entry ───────


def _seed_mix_pools(cfg: dg.Cfg, n_pos: int) -> None:
    def _rows(tag: str, n: int) -> list[dict]:
        return [
            {
                "prompt": [{"role": "user", "content": f"{tag} question {i}?"}],
                "completion": [{"role": "assistant", "content": f"{tag} answer {i}."}],
            }
            for i in range(n)
        ]

    dg._write_jsonl(cfg.positives_dir / "syc-icl" / "pos.jsonl", _rows("pos", n_pos))
    dg._write_jsonl(cfg.negatives_dir / "syc-panel5" / "neg.jsonl", _rows("neg", 4))
    dg._write_jsonl(cfg.generic_dir / "pool.jsonl", _rows("gen", 100))


def test_phase_mixes_reentry_rederives_pass_verdict_and_skips_built_cells(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(dg, "_tokenizer", lambda: _FakeTok())
    _seed_mix_pools(cfg, n_pos=4)
    dg.phase_mixes(cfg)
    gate = json.loads((cfg.mixes_dir / "yield_gate.json").read_text(encoding="utf-8"))
    assert gate["gate"]["syc"]["pass"] is True and gate["kept_behaviors"] == ["syc"]
    mixes = sorted(p for p in cfg.mixes_dir.glob("*/train_mix.jsonl"))
    assert mixes  # (syc, icl) content cells built
    shas = {p: p.read_bytes() for p in mixes}
    dg.phase_mixes(cfg)  # re-entry: verdict re-derived, built cells skipped
    gate2 = json.loads((cfg.mixes_dir / "yield_gate.json").read_text(encoding="utf-8"))
    assert gate2["gate"] == gate["gate"]
    assert {p: p.read_bytes() for p in sorted(cfg.mixes_dir.glob("*/train_mix.jsonl"))} == shas


def test_phase_mixes_reentry_rederives_drop_verdict_without_crash(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(dg, "_tokenizer", lambda: _FakeTok())
    _seed_mix_pools(cfg, n_pos=1)  # below the smoke hard floor (2) -> behavior dropped
    dg.phase_mixes(cfg)
    gate = json.loads((cfg.mixes_dir / "yield_gate.json").read_text(encoding="utf-8"))
    assert gate["gate"]["syc"]["pass"] is False
    assert gate["dropped_behaviors"] == ["syc"]
    assert not list(cfg.mixes_dir.glob("*/train_mix.jsonl"))
    dg.phase_mixes(cfg)  # re-entry on the recorded DROP verdict: no crash, same verdict
    gate2 = json.loads((cfg.mixes_dir / "yield_gate.json").read_text(encoding="utf-8"))
    assert gate2["gate"] == gate["gate"]
