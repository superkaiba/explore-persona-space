"""#825 round-11 crash-fix tests: G-B simulated-user realism fix (epm:failure v16).

Pilot att-20260716-012315 failed gate G-B at every K_gen rung (instruct
completion 0.22/0.42/0.64 at 24/20/16) with 100% token-budget deaths
(window_overflow 33 + capture_budget_overflow 6 of 39 instruct deaths at K24;
zero haiku_failed / empty_completion / state_desync — no conversation-ender
exists in the rollout loop). Mechanism: the v1 simulator template produced
user turns unlike real users (median 124 words vs 24 for the real corpus u1,
~2 questions/turn vs 0), which elicited ~3x longer instruct answers (464 vs
152 words median; 17% at the 1024-token cap), overflowing the 15,360-token
generation window by depth ~12-19. Pretrained is immune (the "\\n\\n" stop
truncates answers at the first paragraph, median 84 tokens).

These tests pin the v2 fix with REAL bodies throughout; the ONLY fake is
``dispatch_calls`` at the Anthropic API boundary (``create_autospec`` —
signature-conformant by construction), which still CALLS the real
``build_request`` / ``parse_response`` closures:

- v2 template constraints + persona-brief embedding (fails pre-fix).
- ``sim_user_prompt`` resume-fingerprint regime key (#722 r3; fails pre-fix)
  + the v1-dir resume REFUSAL (SystemExit on fingerprint mismatch).
- ``_run_haiku_wave`` real body: v2 system prompt reaches the built request
  (top-level ``system`` param), result mapping, >5% hard-fail raise.
- mock-subject end-to-end ``run_rollout``: continuation to k_gen, per-step
  checkpoints, and the REAL window_overflow / capture_budget_overflow
  termination branches.
- ``run_report`` length telemetry (user/answer word medians, closer_rate,
  died_reasons) — the diagnostics the failed pilot was missing.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from unittest.mock import create_autospec

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "src"))

import issue825_turndyn_rollout as ro  # noqa: E402

from explore_persona_space.llm.api_dispatch import (  # noqa: E402
    DispatchResult,
    dispatch_calls,
)

BRIEF = "You are skeptical of the assistant's answer. Push back on its weakest claim."


# ---------------------------------------------------------------------------
# v2 template + fingerprint regime key
# ---------------------------------------------------------------------------


def test_sim_user_system_v2_constraints_and_brief_embedding():
    sys_prompt = ro._sim_user_system(BRIEF)
    assert BRIEF in sys_prompt  # per-conversation persona brief still embedded
    # measured-real-user constraints (the round-11 fix): brevity, ONE focused
    # question, no pleasantries, never close the conversation
    assert "1-3 sentences" in sys_prompt
    assert "under 40 words" in sys_prompt
    assert "ONE focused question" in sys_prompt
    assert "no pleasantries or thanks" in sys_prompt
    assert "Never wrap up, close, or say goodbye" in sys_prompt
    # v1 invariants kept
    assert "no role labels" in sys_prompt
    assert "first person" in sys_prompt


def _args(**over) -> argparse.Namespace:
    base = dict(
        model="pretrained",
        seeds_dir="",
        out_dir="",
        k_gen=3,
        pilot_n=0,
        shard="0/1",
        chunk_size=8,
        capture_budget=15872,
        engine_max_len=16384,
        haiku_model=ro.HAIKU_GEN_MODEL,
        report=False,
        smoke=True,
        tiny_model_dir="",
    )
    base.update(over)
    return argparse.Namespace(**base)


def test_fingerprint_carries_sim_user_prompt_regime_key():
    fp = ro._fingerprint(_args(), "deadbeef")
    assert fp["sim_user_prompt"] == ro.SIM_USER_PROMPT_VERSION
    assert fp["sim_user_prompt"] == "v2-realistic-brevity"


# ---------------------------------------------------------------------------
# _run_haiku_wave real body; dispatch_calls faked at the API boundary only
# ---------------------------------------------------------------------------


def _fake_dispatch(monkeypatch, replies=None, error_ids=()):
    """Autospec'd dispatch_calls fake: signature-validated, executes the REAL
    build_request/parse_response closures, captures the built requests."""
    captured: list[dict] = []

    def side_effect(items, **kw):
        out = {}
        for item in items:
            captured.append(kw["build_request"](item))  # REAL closure
            if item.item_id in error_ids:
                out[item.item_id] = DispatchResult(
                    item_id=item.item_id, result=None, error=True, reason="boom"
                )
            else:
                text = (replies or {}).get(item.item_id, "How does that part work?")
                out[item.item_id] = DispatchResult(
                    item_id=item.item_id,
                    result=kw["parse_response"](text + "  \n"),  # REAL closure
                    error=False,
                )
        return out

    fake = create_autospec(dispatch_calls, side_effect=side_effect)
    monkeypatch.setattr(ro, "dispatch_calls", fake)
    return captured


def test_run_haiku_wave_builds_v2_request_and_maps_results(monkeypatch, tmp_path):
    captured = _fake_dispatch(monkeypatch)
    turns = [
        {"role": "user", "content": "What is a good beginner houseplant?"},
        {"role": "assistant", "content": "A pothos is hardy and forgiving."},
    ]
    out = ro._run_haiku_wave([("c0", BRIEF, turns)], 2, _args(), tmp_path)
    assert out == {"c0": {"text": "How does that part work?"}}  # parse stripped
    assert len(captured) == 1
    req = captured[0]
    # system prompt is the TOP-LEVEL param (Messages API has no system role)
    assert "system" in req and BRIEF in req["system"]
    assert "Never wrap up, close, or say goodbye" in req["system"]  # v2 live
    assert req["max_tokens"] == ro.HAIKU_MAX_TOKENS
    assert [m["role"] for m in req["messages"]] == ["user"]
    assert "Now write the user's next message." in req["messages"][0]["content"]


def test_run_haiku_wave_hard_fail_rate_raises(monkeypatch, tmp_path):
    _fake_dispatch(monkeypatch, error_ids={"c0:u2"})
    turns = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
    with pytest.raises(RuntimeError, match="systemic dispatch problem"):
        ro._run_haiku_wave([("c0", BRIEF, turns)], 2, _args(), tmp_path)


# ---------------------------------------------------------------------------
# mock-subject end-to-end harness (REAL run_rollout continuation/termination)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tok_dir(tmp_path_factory):
    from transformers import AutoTokenizer

    # Loaded ONCE per module (per-load model_info() Hub calls 429 — gotchas.md);
    # saved to tmp so run_rollout's smoke branch loads REAL Qwen tokenizer files
    # (the production smoke contract: the tiny dir carries the real tokenizer).
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
    d = tmp_path_factory.mktemp("tiny-qwen-tok")
    tok.save_pretrained(str(d))
    return str(d)


def _seed_rows(n=4):
    topics = ("watering a pothos", "sourdough starters", "bike tire pressure", "star charts")
    return [
        {
            "conv_id": f"conv{i:02d}",
            "seed_rank": i,
            "in_panel": True,
            "u1": f"Can you explain the basics of {topics[i % len(topics)]}?",
            "brief_id": "skeptical_followup",
            "brief_text": BRIEF,
        }
        for i in range(n)
    ]


def _write_seeds(tmp_path) -> str:
    seeds_dir = tmp_path / "panel"
    seeds_dir.mkdir()
    with (seeds_dir / "armG_seeds_shard000.jsonl").open("w") as f:
        for r in _seed_rows():
            f.write(json.dumps(r) + "\n")
    return str(seeds_dir)


def test_run_rollout_mock_subject_completes_to_depth(monkeypatch, tmp_path, tok_dir):
    _fake_dispatch(monkeypatch)
    args = _args(
        seeds_dir=_write_seeds(tmp_path), out_dir=str(tmp_path / "out"), tiny_model_dir=tok_dir
    )
    ro.run_rollout(args)
    root = tmp_path / "out" / "pretrained" / "shard0of1"
    for k in (1, 2, 3):
        for half in ("a", "b"):
            assert (root / f"step{k:02d}_{half}.jsonl").exists()
    summary = json.loads((root / "rollout_summary.json").read_text())
    assert summary["n_completed"] == 4 and summary["completion_rate"] == 1.0
    assert summary["per_depth_alive"] == {"1": 4, "2": 4, "3": 4}
    finals = [
        json.loads(line) for line in (root / "rollout_final_shard000.jsonl").open() if line.strip()
    ]
    for c in finals:
        roles = [t["role"] for t in c["turns"]]
        assert roles == ["user", "assistant"] * 3  # u1 + a1 + (haiku u + a) x 2
        assert c["alive"] and c["died_at"] is None
    # step rows carry the round-11 generation-metadata diagnostics
    step1 = [json.loads(line) for line in (root / "step01_a.jsonl").open() if line.strip()]
    assert all("finish_reason" in r and "n_gen_tokens" in r for r in step1)
    # fingerprint on disk carries the regime key
    fp = json.loads((root / "rollout_fingerprint.json").read_text())
    assert fp["sim_user_prompt"] == ro.SIM_USER_PROMPT_VERSION


def test_run_rollout_capture_budget_death_path(monkeypatch, tmp_path, tok_dir):
    _fake_dispatch(monkeypatch)
    args = _args(
        seeds_dir=_write_seeds(tmp_path),
        out_dir=str(tmp_path / "out_cap"),
        tiny_model_dir=tok_dir,
        capture_budget=10,  # any answered turn overflows the capture render
    )
    ro.run_rollout(args)
    root = tmp_path / "out_cap" / "pretrained" / "shard0of1"
    summary = json.loads((root / "rollout_summary.json").read_text())
    assert summary["n_completed"] == 0
    assert summary["died_reasons"]["capture_budget_overflow"] == 4


def test_run_rollout_window_overflow_death_path(monkeypatch, tmp_path, tok_dir):
    _fake_dispatch(monkeypatch)
    args = _args(
        seeds_dir=_write_seeds(tmp_path),
        out_dir=str(tmp_path / "out_win"),
        tiny_model_dir=tok_dir,
        engine_max_len=64,  # prompt + GEN_MAX_TOKENS(1024) always overflows
    )
    ro.run_rollout(args)
    root = tmp_path / "out_win" / "pretrained" / "shard0of1"
    summary = json.loads((root / "rollout_summary.json").read_text())
    assert summary["n_completed"] == 0
    assert summary["died_reasons"]["window_overflow"] == 4


def test_v1_regime_rollout_dir_refuses_resume(monkeypatch, tmp_path, tok_dir):
    """A rollout dir written under the v1 template (no sim_user_prompt key)
    must NOT silently resume under v2 — fingerprint mismatch is a SystemExit."""
    _fake_dispatch(monkeypatch)
    args = _args(
        seeds_dir=_write_seeds(tmp_path), out_dir=str(tmp_path / "out_v1"), tiny_model_dir=tok_dir
    )
    stale = ro._fingerprint(args, ro.hashlib.sha256(b"x").hexdigest())
    # simulate the v1-era fingerprint: regime dict WITHOUT the template key
    stale.pop("sim_user_prompt")
    root = tmp_path / "out_v1" / "pretrained" / "shard0of1"
    root.mkdir(parents=True)
    (root / "rollout_fingerprint.json").write_text(json.dumps(stale))
    with pytest.raises(SystemExit, match="fingerprint MISMATCH"):
        ro.run_rollout(args)


# ---------------------------------------------------------------------------
# run_report length telemetry (the diagnostics the failed pilot was missing)
# ---------------------------------------------------------------------------


def test_run_report_emits_length_telemetry(monkeypatch, tmp_path, tok_dir):
    replies = {
        f"conv{i:02d}:u{k}": f"Thanks, that helps. But how does step {k} apply to case {i}?"
        for i in range(4)
        for k in (2, 3)
    }
    _fake_dispatch(monkeypatch, replies=replies)
    args = _args(
        seeds_dir=_write_seeds(tmp_path), out_dir=str(tmp_path / "out_rep"), tiny_model_dir=tok_dir
    )
    ro.run_rollout(args)
    ro.run_report(args)
    diag = json.loads(
        (tmp_path / "out_rep" / "pretrained" / "rollout_diagnostics.json").read_text()
    )
    assert diag["sim_user_prompt"] == ro.SIM_USER_PROMPT_VERSION
    assert diag["completion_rate"] == 1.0
    assert diag["died_reasons"] == {}
    for k in ("2", "3"):
        node = diag["per_depth"][k]
        assert node["n"] == 4
        assert node["user_words_median"] == len(replies["conv00:u2"].split())
        assert node["closer_rate"] == 1.0  # every reply opens with "Thanks"
        assert "role_leak_rate" in node and "distinct2" in node  # gate keys intact
    for k in ("1", "2", "3"):
        assert diag["answers_per_depth"][k]["n"] == 4
        assert diag["answers_per_depth"][k]["answer_words_median"] > 0
