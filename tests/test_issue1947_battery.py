"""#1947 unit-3 pins: last-token span capture (binding directive), judge/select
P3 drivers (stub judge, drop tally, verdict lattice), consumed-rows schema
derivation, and the batched bootstrap-CI helper.

The tiny-model capture test executes the REAL `_teacher_forced_span_means`
body (from-config 2-layer same-arch model over the REAL Qwen vocab — the #906
tiny-real convention; CPU, seconds)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import issue1947_battery as bat  # noqa: E402
import issue1947_cells as cells  # noqa: E402

SLUG = "syc-pers-con-sv-s42"  # the pilot cell (a REAL registry slug)


def _cfg(tmp_path: Path, **kw) -> bat.Cfg:
    defaults = dict(
        out_root=tmp_path / "root",
        out_dir=tmp_path / "analysis",
        smoke=True,
        stub_judge=True,
        upload=False,
        local_ladders=True,
        cells_filter=(SLUG,),
        behaviors=("sycophancy",),
    )
    defaults.update(kw)
    cfg = bat.Cfg(**defaults)
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    return cfg


def _write_ladder(cfg: bat.Cfg, slug: str, rates_shape: dict[str, list[list[str]]]) -> None:
    payload = {
        "slug": slug,
        "behavior": "sycophancy",
        "questions": ["q one?", "q two?"],
        "rungs": {s: {"completions": comps} for s, comps in rates_shape.items()},
    }
    p = cfg.out_root / "ladders" / slug / "ladder_rollouts.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload), encoding="utf-8")


def test_judge_stub_tally_and_resume(tmp_path):
    """Stub judge over a 2-rung synthetic ladder: rates in [0,1], drop split
    fields present, per-rung checkpoints regime-keyed (a second run resumes)."""
    cfg = _cfg(tmp_path)
    comps = [["yes indeed", "agree"], ["no", "disagree"]]
    _write_ladder(cfg, SLUG, {"5": comps, "10": comps})
    assert bat.cmd_judge(cfg) == 0
    judged = json.loads((cfg.out_dir / "judge" / f"judged_{SLUG}.json").read_text())
    assert set(judged["rates_by_step"]) == {"5", "10"}
    for rec in judged["records"].values():
        assert 0.0 <= rec["rate"] <= 1.0
        assert "n_dropped_draws_content" in rec and "n_transport_lost_draws" in rec
    assert bat.cmd_judge(cfg) == 0  # resume path: checkpoints reused, same regime


def test_select_verdict_lattice_earliest_in_band(tmp_path, monkeypatch):
    """Earliest-in-band rung wins; a never-in-band ladder falls back to
    closest-approach with in_band False (the DoseSelection invariant)."""
    cfg = _cfg(tmp_path)
    judge_dir = cfg.out_dir / "judge"
    judge_dir.mkdir(parents=True, exist_ok=True)
    in_band = {"5": 0.2, "10": 0.7, "15": 0.9}  # earliest in [0.60, 0.85] = 10
    (judge_dir / f"judged_{SLUG}.json").write_text(
        json.dumps(
            {
                "slug": SLUG,
                "behavior": "sycophancy",
                "instrument": "stub-smoke",
                "questions_sha256": "x",
                "rates_by_step": in_band,
                "records": {},
            }
        )
    )
    monkeypatch.setattr(bat, "_marker_cells", lambda _cfg: [])
    assert bat.cmd_select(cfg) == 0
    man = json.loads(cfg.verdict_manifest_path().read_text())
    sel = man["content"][SLUG]["selection"]
    assert sel["step"] == 10 and sel["in_band"] is True and sel["fallback"] is None
    # closest-approach fallback
    (judge_dir / f"judged_{SLUG}.json").write_text(
        json.dumps(
            {
                "slug": SLUG,
                "behavior": "sycophancy",
                "instrument": "stub-smoke",
                "questions_sha256": "x",
                "rates_by_step": {"5": 0.1, "10": 0.3},
                "records": {},
            }
        )
    )
    assert bat.cmd_select(cfg) == 0
    sel = json.loads(cfg.verdict_manifest_path().read_text())["content"][SLUG]["selection"]
    assert sel["in_band"] is False and sel["fallback"] == "closest_approach"
    assert sel["step"] == 10  # min band distance


def test_consumed_row_idxs_schemas(tmp_path):
    """Primary: the train/sft.py seam schema (realized_step_of_idx, 0-based);
    plus the two legacy fixture shapes and the rep-cell no-seam rule."""
    cfg = _cfg(tmp_path)
    p = cfg.out_root / "ladders" / SLUG / "realized_consumption.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"realized_step_of_idx": [0, 0, 1, 2, None]}))
    assert bat._consumed_row_idxs(cfg, SLUG, 2) == {0, 1, 2}
    p.write_text(json.dumps({"row_to_step": {"0": 0, "1": 0, "2": 1, "3": 2}}))
    assert bat._consumed_row_idxs(cfg, SLUG, 2) == {0, 1, 2}
    p.write_text(json.dumps({"steps": {"0": [0, 1], "1": [2], "2": [3]}}))
    assert bat._consumed_row_idxs(cfg, SLUG, 2) == {0, 1, 2}
    rep = "imp-pers-con-rep-s42"  # no seam: all-rows once epoch 1 completes
    assert bat._consumed_row_idxs(cfg, rep, 15) == set(range(80))
    with pytest.raises(RuntimeError, match="unknowable"):
        bat._consumed_row_idxs(cfg, rep, 2)


def test_adapter_subfolder_marker_vs_content():
    assert bat._adapter_subfolder(SLUG, 10) == f"issue1947/{SLUG}/checkpoint-10"
    assert bat._adapter_subfolder("mk-pers-con-sv-s42", 40) == (
        "issue1947/marker/mk-pers-con-sv-s42/checkpoint-40"
    )


def test_corpus_arm_slugs_are_registry_cells():
    for slug in bat.CORPUS_ARM_SLUGS:
        assert slug in cells.CELL_BY_SLUG, slug
    assert len(bat.CORPUS_ARM_SLUGS) == 12


def test_boot_cos_ci_batched_contains_point():
    import issue1947_analysis as ana
    import numpy as np

    rng = np.random.default_rng(0)
    cand = rng.normal(size=32)
    stack = cand[None, :] + 0.5 * rng.normal(size=(200, 32))
    ci, mean_cos = ana._boot_cos_ci(stack, cand, 400, seed=1)
    # NOTE: point-coverage is NOT a percentile-bootstrap guarantee for the
    # concave cos statistic (small downward bias vs the plug-in point — the
    # same shape as the reused #1768 _boot_ci convention). Pins: the bootstrap
    # mean sits inside its own CI; the batched weight-GEMM reproduces a serial
    # per-draw loop over the SAME weights (batched-rewrite equivalence).
    assert ci[0] <= mean_cos <= ci[1]
    assert ci[1] - ci[0] > 1e-4  # non-degenerate width
    rng2 = np.random.default_rng(1)
    n = stack.shape[0]
    W = rng2.multinomial(n, np.full(n, 1.0 / n), size=400).astype(np.float64) / n
    cn = cand / np.linalg.norm(cand)
    serial = np.array([_c for _c in ((w @ stack) @ cn / np.linalg.norm(w @ stack) for w in W)])
    batched = (W @ stack) @ cn / np.linalg.norm(W @ stack, axis=1)
    assert np.allclose(serial, batched, atol=1e-10)
    # ordering sanity: an anti-aligned candidate reads a strictly lower CI
    ci_neg, _ = ana._boot_cos_ci(stack, -cand, 400, seed=1)
    assert ci_neg[1] < ci[0]


def test_pilot_gate_halts_rc7_on_over_2x_projection(tmp_path, monkeypatch):
    """The P4 in-run pilot gate (plan §9): >2x the booked GPU-h HALTs with the
    DISTINCT rc + a report JSON (the #1415 artifact-routed convention), never
    an anonymous crash. Boundary fake: subprocess.run only (signature-shaped)."""
    import subprocess as sp

    cfg = _cfg(tmp_path)
    man = {
        "issue": 1947,
        "band": [0.6, 0.85],
        "content": {
            SLUG: {
                "slug": SLUG,
                "behavior": "sycophancy",
                "selection": {"step": 10, "rate": 0.7, "in_band": True, "fallback": None},
            }
        },
        "marker": {},
    }
    (cfg.out_dir / "verdict_manifest.json").write_text(json.dumps(man))
    cfg = _cfg(tmp_path, smoke=False, sentinel_dir=tmp_path / "logs")

    def fake_run(  # boundary fake mirroring BOTH battery call shapes:
        cmd,  # pilot/unit launch (env+stdout+stderr) and _git_short_sha (capture_output+text+cwd)
        env=None,
        stdout=None,
        stderr=None,
        capture_output=False,
        text=False,
        cwd=None,
        check=False,
    ):
        return sp.CompletedProcess(cmd, 0, stdout="deadbee\n" if capture_output else None)

    monkeypatch.setattr(bat.subprocess, "run", fake_run)
    monkeypatch.setattr(bat, "PLAN_P4P5_GPU_H", 1e-12)  # any real pilot wall trips 2x
    monkeypatch.setattr(bat, "_physical_gpus", lambda: [0])
    rc = bat.cmd_capture_fit(cfg, ["echo"])
    assert rc == bat.PILOT_GATE_RC == 7
    rep = json.loads((cfg.out_dir / "pilot_gate_report.json").read_text())
    assert rep["ratio"] > 2 and rep["pilot_rc"] == 0


def _tiny_qwen_dir(tmp_path: Path):
    """From-config 2-layer Qwen2 over the REAL Qwen vocab (the #906 tiny-real
    convention; CPU, seconds). Returns (tokenizer, model_dir)."""
    from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
    config = Qwen2Config(
        vocab_size=len(tok),
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=512,
    )
    model_dir = tmp_path / "tiny"
    Qwen2ForCausalLM(config).save_pretrained(model_dir)
    tok.save_pretrained(model_dir)
    return tok, model_dir


def _chat_rows(tok) -> list[dict]:
    """Generation-rendered rows THROUGH the REAL ``compute_prompt_spans``
    (r2 Critical 1: the r1 test set context_len manually, bypassing the span
    computation — it pinned only the 1-token-span mechanics, never the
    position)."""
    from explore_persona_space.analysis.representation_shift import compute_prompt_spans

    sys_prompt = "You are a pragmatic software engineer."
    questions = ["What is your view on tabs versus spaces?", "How do you review code?"]
    rows = []
    for i, q in enumerate(questions):
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": q},
        ]
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompt_ids = tok(text, add_special_tokens=False)["input_ids"]
        prefix_len, context_len = compute_prompt_spans(tok, sys_prompt, q, prompt_ids)
        resp_ids = tok(f"A short answer number {i}.", add_special_tokens=False)["input_ids"]
        rows.append(
            {
                "persona": "t",
                "question_idx": i,
                "prompt_sha": f"s{i}",
                "prompt_token_ids": prompt_ids,
                "response_token_ids": resp_ids,
                "prefix_len": prefix_len,
                "context_len": context_len,
            }
        )
    return rows


@pytest.mark.slow
def test_last_token_span_capture_tiny_real_model(tmp_path):
    """r2 Critical 1 position pin (directive v9): REAL
    `_teacher_forced_span_means` body on generation-rendered rows built
    THROUGH compute_prompt_spans. The PRIMARY ``last_prompt`` span indexes
    len(prompt_token_ids)-1 — the assistant-header newline (decode-asserted)
    — a DIFFERENT position from the v9-excluded last-user-content token
    (``last_ctx``); exact-position identity pinned via a context_len==p_len
    variant where the two 1-token bounds coincide."""
    import torch

    from explore_persona_space.analysis.representation_shift import (
        SPAN_ARMS_LAST,
        _teacher_forced_span_means,
    )

    tok, model_dir = _tiny_qwen_dir(tmp_path)
    rows = _chat_rows(tok)
    for r in rows:
        p_len = len(r["prompt_token_ids"])
        # v9 PRIMARY position: the final generation-rendered prompt token
        # decodes to the assistant-header newline
        last = tok.decode([r["prompt_token_ids"][p_len - 1]])
        assert "\n" in last and last.strip("\n") == "", last
        # the excluded last_ctx position is genuinely different (template tail)
        assert r["context_len"] < p_len, (r["context_len"], p_len)
    pooled = _teacher_forced_span_means(
        str(model_dir),
        rows,
        ["t"],
        layers=[1],
        spans=("prefix", "context", "response", *SPAN_ARMS_LAST),
        device="cpu",
        dtype=torch.float32,
        tf_batch_size=2,
    )
    assert set(pooled) == {
        "prefix",
        "context",
        "response",
        "prefix_last",
        "last_prompt",
        "last_ctx",
    }
    # PRIMARY reads a different position from the v9-excluded last_ctx
    assert not torch.allclose(pooled["last_prompt"][1], pooled["last_ctx"][1])
    assert not torch.allclose(pooled["context"][1], pooled["last_ctx"][1])
    for span in pooled:
        assert pooled[span][1].shape == (2, 32)
        assert torch.isfinite(pooled[span][1]).all()
    # exact-position identity: with context_len := p_len the last_ctx bound
    # (context_len-1, context_len) coincides with last_prompt (p_len-1, p_len)
    rows_id = [dict(r, context_len=len(r["prompt_token_ids"])) for r in rows]
    pooled_id = _teacher_forced_span_means(
        str(model_dir),
        rows_id,
        ["t"],
        layers=[1],
        spans=("last_prompt", "last_ctx"),
        device="cpu",
        dtype=torch.float32,
        tf_batch_size=2,
    )
    assert torch.allclose(pooled_id["last_prompt"][1], pooled_id["last_ctx"][1])


@pytest.mark.slow
def test_last_prompt_decode_check_raises_on_ungenerated_render(tmp_path):
    """v9 sample-row decode check (r2 Critical 1): rows whose final prompt
    token is NOT the assistant-header newline (a bare-text render, no
    generation prompt) fail LOUD at capture time when ``last_prompt`` is
    requested — and still capture fine when it is not."""
    import torch

    from explore_persona_space.analysis.representation_shift import _teacher_forced_span_means

    tok, model_dir = _tiny_qwen_dir(tmp_path)
    ids = tok("hello there, what is up today?", add_special_tokens=False)["input_ids"]
    rows = [
        {
            "persona": "t",
            "question_idx": 0,
            "prompt_token_ids": ids,
            "response_token_ids": ids[:3],
            "prefix_len": 1,
            "context_len": len(ids),
        }
    ]
    with pytest.raises(RuntimeError, match="decode check failed"):
        _teacher_forced_span_means(
            str(model_dir),
            rows,
            ["t"],
            layers=[1],
            spans=("context", "last_prompt"),
            device="cpu",
            dtype=torch.float32,
            tf_batch_size=1,
        )
    # without last_prompt the same rows capture fine (span-mean secondary)
    pooled = _teacher_forced_span_means(
        str(model_dir),
        rows,
        ["t"],
        layers=[1],
        spans=("context", "last_ctx"),
        device="cpu",
        dtype=torch.float32,
        tf_batch_size=1,
    )
    assert torch.isfinite(pooled["context"][1]).all()


def test_consumed_row_idxs_marker_no_seam(tmp_path):
    """r1 Critical 1 regression pin: marker cells train WITHOUT the seam — the
    consumed set is the FULL mix iff the rung completes the single-visit epoch
    (step*16 >= 6,400), and None (skip signal) below it. NEVER a
    FileNotFoundError on the never-written realized_consumption.json."""
    cfg = _cfg(tmp_path)
    mk = "mk-pers-con-sv-s42"
    n_rows = cells.CELL_BY_SLUG[mk].n_rows
    assert bat._consumed_row_idxs(cfg, mk, 400) == set(range(n_rows))
    assert bat._consumed_row_idxs(cfg, mk, 100) is None


def test_marker_tree_subsample_deterministic_stratified():
    """Marker full-tree cap (r1 Critical 1 scope decision): seeded, composition
    preserved (1:4 pos:neg), mix order kept."""
    rows = [{"question_idx": i, "row_kind": "pos" if i % 5 == 0 else "neg"} for i in range(6400)]
    a = bat._marker_tree_subsample(rows, 42)
    b = bat._marker_tree_subsample(rows, 42)
    assert a == b
    assert len(a) == bat.MARKER_TREE_ROWS
    n_pos = sum(1 for r in a if r["row_kind"] == "pos")
    assert abs(n_pos / len(a) - 1 / 5) < 0.02  # proportions preserved
    idxs = [r["question_idx"] for r in a]
    assert idxs == sorted(idxs)  # mix order (question_idx == mix row index)


def test_battery_sentinel_poller_conformant(tmp_path):
    """r1 Critical 2 regression pin: the battery done-sentinel carries the
    poller's required envelope keys and round-trips through the REAL
    poll_pipeline._parse_sentinel (dry-run on the serialized body)."""
    import poll_pipeline as pp

    cfg = _cfg(tmp_path)
    payload = {"issue": 1947, "phase": "battery", "status": "done", "failed_units": []}
    sent = bat._battery_sentinel(cfg, payload)
    assert set(pp._SENTINEL_REQUIRED_KEYS) <= set(sent)
    assert sent["sentinel_schema_version"] == pp.SENTINEL_SCHEMA_VERSION_SUPPORTED
    parsed = pp._parse_sentinel("issue-1947-battery-done.json", json.dumps(sent))
    assert parsed is not None and parsed["kind"] == "epm:smoke-result"
    assert parsed["payload"]["status"] == "done"
    non_smoke = bat._battery_sentinel(_cfg(tmp_path, smoke=False), payload)
    assert non_smoke["kind"] == "epm:results" and non_smoke["blocks_pipeline"] is True


def test_runnable_fits_gating():
    """r1 Critical 1: a failed corpus unit skips ONLY its own fit — the rest of
    P5 runs (never zeroed out by one capture failure)."""
    fits = ["fit:a", "fit:b", "fit:c"]
    runnable, skipped = bat._runnable_fits(fits, ["corpus:b", "arm:z"])
    assert runnable == ["fit:a", "fit:c"] and skipped == ["fit:b"]
    runnable, skipped = bat._runnable_fits(fits, [])
    assert runnable == fits and skipped == []


def test_fan_halt_drains_pending_queue(tmp_path):
    """r2 Critical 2 regression pin: the compute-gate predicate fires with
    MORE queued units than execution slots — `_fan` drains the not-yet-started
    queue into ``not_started`` (SKIPPED accounting) and TERMINATES. Pre-fix
    the scheduler loop spun forever on the non-empty pending queue once
    ``running`` drained (and the retry-insert could re-arm the hang);
    thread-guarded so a regression fails as a timeout assert, not a CI hang."""
    import threading

    cfg = _cfg(tmp_path)  # smoke=True -> 0.2 s scheduler tick
    argv_base = [sys.executable, "-c", "import time; time.sleep(0.05)"]
    queue = [f"corpus:u{i}" for i in range(6)]  # 6 units, 2 execution slots
    out: dict = {}

    def run():
        out["r"] = bat._fan(
            cfg, argv_base, [0, 1], queue, on_complete=lambda unit, wall_h, walls: True
        )

    th = threading.Thread(target=run, daemon=True)
    th.start()
    th.join(timeout=120)
    assert not th.is_alive(), "_fan hung after compute-gate halt (r2 Critical 2)"
    failed, walls, halted, not_started = out["r"]
    assert halted and not failed
    assert not_started, "queued units were not drained into not_started"
    assert len(walls) + len(not_started) == 6
    assert set(walls) | set(not_started) == set(queue)


def test_select_marker_consume_slot_reads(tmp_path):
    """cmd_select consumes the worker's programmatic marker selection verbatim
    (r1 style nit: the marker-consume path was CLI-smoke-only)."""
    mk = "mk-pers-con-sv-s42"
    cfg = _cfg(tmp_path, cells_filter=(SLUG, mk))
    judge_dir = cfg.out_dir / "judge"
    judge_dir.mkdir(parents=True, exist_ok=True)
    (judge_dir / f"judged_{SLUG}.json").write_text(
        json.dumps(
            {
                "slug": SLUG,
                "behavior": "sycophancy",
                "instrument": "stub-smoke",
                "questions_sha256": "x",
                "rates_by_step": {"10": 0.7},
                "records": {},
            }
        )
    )
    slot = cfg.out_root / "marker_ladders" / mk / "slot_reads.json"
    slot.parent.mkdir(parents=True, exist_ok=True)
    slot.write_text(json.dumps({"selection": {"step": 40, "read": "band-entry"}}))
    assert bat.cmd_select(cfg) == 0
    man = json.loads(cfg.verdict_manifest_path().read_text())
    assert man["marker"][mk]["selection"]["step"] == 40
    assert man["marker"][mk]["behavior"] == "marker"


def test_r3_floor_crosscheck_tolerance_and_smoke_demotion(tmp_path, monkeypatch):
    """p5-m0-floors concern wiring: recomputed M0/floor vs the r3 committed
    values — pass within tolerance, fail LOUD on divergence (non-smoke), and
    DEMOTED to informational under --smoke (the #1345 smoke-gate rule: toy-n
    fits cannot reproduce production anchors)."""
    r3 = {"fits": {"M0": {"heldout_r2": 0.60}}, "map_change": {"floor_p95": 8.5}, "n_test": 1000}

    def fake_stage(repo, path, local, repo_type):
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_text(json.dumps(r3))

    monkeypatch.setattr(
        bat.hub, "retry_transient", lambda fn, what=None: ["pfx/fits_bare_n/arm_L19.json"]
    )
    monkeypatch.setattr(bat.hub, "stage_hub_file", fake_stage)
    cfg = _cfg(tmp_path, smoke=False)
    rec_ok = {
        "fits": {"M0": {"heldout_r2": 0.605}},
        "map_change": {"floor_p95": 8.6},
        "n_test": 1000,
    }
    bat._r3_floor_crosscheck(cfg, "syc-pers-con-sv-s42", {19: rec_ok})
    out = json.loads(
        (cfg.out_root / "fits" / "r3_crosscheck" / "syc-pers-con-sv-s42.json").read_text()
    )
    assert out["verdict"] == "pass" and out["checks"][0]["ok"] is True
    rec_bad = {
        "fits": {"M0": {"heldout_r2": 0.20}},
        "map_change": {"floor_p95": 30.0},
        "n_test": 1000,
    }
    with pytest.raises(RuntimeError, match="diverge"):
        bat._r3_floor_crosscheck(cfg, "imp-pers-con-sv-s42", {19: rec_bad})
    # r2 Critical 3 (retry composition): the report persists with verdict=fail
    # BEFORE the raise, so the one-retry _fan path re-enters with dest.exists()
    # — the SECOND call must STILL raise off the persisted report (pre-fix it
    # returned silently, self-bypassing the gate), even with clean recs.
    with pytest.raises(RuntimeError, match="resume does not clear"):
        bat._r3_floor_crosscheck(cfg, "imp-pers-con-sv-s42", {19: rec_ok})
    # a persisted PASS report short-circuits silently (idempotent resume)
    bat._r3_floor_crosscheck(cfg, "syc-pers-con-sv-s42", {19: rec_ok})
    cfg_smoke = _cfg(tmp_path, smoke=True)
    bat._r3_floor_crosscheck(cfg_smoke, "cas-pers-con-sv-s42", {19: rec_bad})
    out = json.loads(
        (cfg_smoke.out_root / "fits" / "r3_crosscheck" / "cas-pers-con-sv-s42.json").read_text()
    )
    assert out["verdict"] == "informational-smoke" and out["n_divergent"] == 1
    # informational-smoke persisted report also short-circuits on smoke resume
    bat._r3_floor_crosscheck(cfg_smoke, "cas-pers-con-sv-s42", {19: rec_bad})


def _toy_store(path: Path, qidx: list[int], spans: tuple[str, ...], seed: int, d: int = 16):
    import torch

    g = torch.Generator().manual_seed(seed)
    payload = {
        "row_question_idx": list(qidx),
        "row_sha": [f"sha{q}" for q in qidx],
        "arms": {s: {1: torch.randn(len(qidx), d, generator=g)} for s in spans},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def test_fit_lasttoken_arm_real_body_toy_n(tmp_path):
    """Directive v7 items 2-3 (r1 Major 1): the within-run LAST-TOKEN M⁺ fit
    runs the REAL _fit_map/_map_reads/_identity_bias_reads bodies at toy n on
    CPU and writes one *_lasttoken.json per (arm, layer) — identity+bias + kNN
    attached, D column flagged no-M0-floor-available (no probe report, and the
    r1 base store carries no last_prompt arm). PRIMARY input span is
    ``last_prompt`` (directive v9)."""
    cfg = _cfg(tmp_path, smoke=True, layers=(1,))
    n = 90
    sample = {
        "rows": [{"sha": f"sha{i}", "src_qidx": i} for i in range(n)],
        "n_train": 60,
        "n_val": 15,
        "n_test": 15,
    }
    sp = cfg.out_root / "on_target" / "inputs" / "corpus_sample_pfx.json"
    sp.parent.mkdir(parents=True, exist_ok=True)
    sp.write_text(json.dumps(sample))
    base_dir = cfg.out_root / "corpus_capture" / "base_content"
    base_dir.mkdir(parents=True, exist_ok=True)
    (base_dir / "rows_spans.json").write_text("{}")
    qidx = list(range(n))
    # base store WITHOUT last_prompt (the r1-base reality) -> M0 unavailable
    _toy_store(base_dir / "pooled.pt", qidx, spans=("context", "response"), seed=1)
    _toy_store(
        cfg.out_root / "corpus_capture" / SLUG / "pooled.pt",
        qidx,
        spans=("context", "last_prompt", "response"),
        seed=2,
    )
    bat._fit_lasttoken_arm(cfg, SLUG)
    rec = json.loads(
        (cfg.out_root / "fits" / "lasttoken" / f"{SLUG}_L1_lasttoken.json").read_text()
    )
    assert rec["input_span"] == "last_prompt" and rec["n_test"] == 15
    mp = rec["fits"]["Mplus_lasttoken"]
    assert "heldout_r2" in mp and "knn_cosine" in mp
    assert mp["identity_bias"]["applicable"] is True
    assert rec["fits"]["M0_lasttoken"]["status"] == "unavailable"
    assert rec["map_change"]["status"] == "no-M0-floor-available"
    assert rec["map_change"]["D"] is None


def test_external_arm_method_seam(monkeypatch):
    """r13 crash pin (KeyError 'imp-bare-con-sv-s42' at issue1768_fit.py
    `_pfx_fit_core` metadata): `_resolve_arm_method` resolves an
    external-registered arm (#1947 slugs, absent from #1768's arm registry)
    from EXTERNAL_ARM_METHOD, keeps a #1768-registry arm on the REAL
    X.arm_method path (the hardcoded #1586 full-FT identity — no Hub
    dependency; the #1481 verdict manifest is a committed eval_results
    fixture), and still fails fast (KeyError) on a truly unknown arm."""
    import issue1768_fit as FIT

    slug = "imp-bare-con-sv-s42"  # the crashing #1947 slug (not a #1768 arm)
    monkeypatch.setitem(FIT.EXTERNAL_ARM_METHOD, slug, "lora")
    assert FIT._resolve_arm_method(slug) == "lora"
    # a #1768-registry arm bypasses the external map — real registry lookup
    assert FIT._resolve_arm_method("syc-pers-ft-con-s42") == "ft"
    with pytest.raises(KeyError):
        FIT._resolve_arm_method("not-an-arm-anywhere-x0")


def test_unit_fit_registers_external_arm_method_before_fit_loop():
    """r13 wiring pin: `unit_fit` registers the slug's method label
    (`FIT.EXTERNAL_ARM_METHOD.setdefault(slug, "lora")`) BEFORE the per-layer
    `fit_bare_n_cell` loop. AST pin — running unit_fit for real needs staged
    corpora (the round-11 wiring-pin convention)."""
    import ast
    import inspect
    import textwrap

    tree = ast.parse(textwrap.dedent(inspect.getsource(bat.unit_fit)))
    reg_lines: list[int] = []
    loop_lines: list[int] = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "setdefault"
            and isinstance(node.func.value, ast.Attribute)
            and node.func.value.attr == "EXTERNAL_ARM_METHOD"
        ):
            assert isinstance(node.args[0], ast.Name) and node.args[0].id == "slug"
            assert isinstance(node.args[1], ast.Constant) and node.args[1].value == "lora"
            reg_lines.append(node.lineno)
        if (
            isinstance(node, ast.For)
            and isinstance(node.iter, ast.Attribute)
            and node.iter.attr == "layers"
        ):
            loop_lines.append(node.lineno)
    assert reg_lines, "unit_fit lost the EXTERNAL_ARM_METHOD registration (r13 fix)"
    assert loop_lines, "unit_fit per-layer fit loop not found (test needs updating)"
    assert min(reg_lines) < min(loop_lines), (
        "EXTERNAL_ARM_METHOD registration must precede the per-layer fit loop"
    )
