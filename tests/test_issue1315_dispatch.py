"""#1315 unit tests — multi-turn span extension, cell table, context byte-asserts.

Covers the plan §4.1 NEW-code invariants that run on CPU:

- ``compute_prompt_spans`` ``prefix_end='last_user'`` extension: asserted token
  boundaries on a known multi-turn (WildChat-shaped) template + the ICL
  user_wrap shape, default-preservation for the single-turn path, and the
  fail-loud multi-turn-without-opt-in assert (real Qwen tokenizer — the
  consumer's exact render).
- ``_build_generation_prompts`` prior_turns threading (generation and span
  computation share ONE message construction).
- The #1315 cell table + capture-pass registry (fail-loud on unregistered
  cells; smoke subset threads through ``capture_passes``).
- Panel ∩ sources == ∅ (hard assert via ``assert_panel_disjoint_from_sources``).
- The WildChat / ICL context byte-asserts against REAL-SHAPE fixtures (the
  staged fu3 mix row shape verified 2026-07-15: ``{"prompt": [...],
  "completion": [...]}``) — pass AND fail cases.
- The fu3 mix row schema vs ``train_behavior_fullft.tokenize_prompt_completion_row``
  (plan assumption 3): a real-shape row tokenizes with completion-only labels.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from explore_persona_space.experiments import issue_1315 as C  # noqa: E402

BASE = "Qwen/Qwen2.5-7B-Instruct"


@pytest.fixture(scope="module")
def tok():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(BASE)


def _prompt_ids(tok, messages) -> list[int]:
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return tok(text, add_special_tokens=False)["input_ids"], text


# ── compute_prompt_spans: multi-turn extension ───────────────────────────────


def test_spans_last_user_multiturn_boundaries(tok):
    """WildChat-shaped 2-turn prefix: the prefix arm covers system + BOTH prior
    turns; the context arm additionally covers exactly the final query."""
    from explore_persona_space.analysis.representation_shift import compute_prompt_spans

    prior = [
        {"role": "user", "content": "Tell me about volcanoes and how they form."},
        {"role": "assistant", "content": "Volcanoes form where magma reaches the surface."},
    ]
    system = "You are a concise assistant."
    question = "What is the capital of France?"
    messages = [
        {"role": "system", "content": system},
        *prior,
        {"role": "user", "content": question},
    ]
    ids, _text = _prompt_ids(tok, messages)
    prefix_len, context_len = compute_prompt_spans(
        tok, system, question, ids, prior_messages=prior, prefix_end="last_user"
    )
    assert 0 < prefix_len < context_len <= len(ids)
    pre = tok.decode(ids[:prefix_len])
    ctx_seg = tok.decode(ids[prefix_len:context_len])
    # prefix covers system + both prior turns, and NOT the final query
    assert system in pre
    for t in prior:
        assert t["content"] in pre
    assert question not in pre
    # the context segment is exactly the final query span
    assert question in ctx_seg
    tail = tok.decode(ids[context_len:])
    assert question not in tail


def test_spans_last_user_icl_user_wrap(tok):
    """ICL shape: the two-shot block precedes the query INSIDE the final user
    turn — with prefix_end='last_user' the block joins the PREFIX arm."""
    from explore_persona_space.analysis.representation_shift import compute_prompt_spans

    block = "Example question: What is 2+2?\nExample answer: Four."
    wrap = block + "\n\n{q}"
    question = "Name one planet."
    messages = [{"role": "user", "content": wrap.format(q=question)}]
    ids, _ = _prompt_ids(tok, messages)
    prefix_len, context_len = compute_prompt_spans(
        tok, None, question, ids, user_wrap=wrap, prefix_end="last_user"
    )
    pre = tok.decode(ids[:prefix_len])
    ctx_seg = tok.decode(ids[prefix_len:context_len])
    assert "Example question:" in pre and question not in pre
    assert question in ctx_seg


def _reph_wrap() -> str:
    """The REAL seam-bearing panel member's wrap ("... {q}" — trailing space
    before the query), pulled from the artifact so drift there is caught."""
    from explore_persona_space.artifacts.negatives import default_panel

    member = {n.slug: n for n in default_panel()}["neg_reph_curious"]
    assert member.user_wrap and member.user_wrap.endswith(" {q}"), member.user_wrap
    return member.user_wrap


def test_spans_seam_wrap_raises_by_default(tok):
    """r7 regression (the p8 field crash): a '... {q}' wrap under
    prefix_end='last_user' puts the prefix boundary on a BPE merge seam (the
    wrap's trailing space merges into the question's first word) — the
    DEFAULT on_seam='raise' contract stays fail-loud."""
    from explore_persona_space.analysis.representation_shift import compute_prompt_spans

    wrap = _reph_wrap()
    question = "How do I review a pull request?"
    messages = [{"role": "user", "content": wrap.format(q=question)}]
    ids, _ = _prompt_ids(tok, messages)
    with pytest.raises(AssertionError, match="BPE drift"):
        compute_prompt_spans(tok, None, question, ids, user_wrap=wrap, prefix_end="last_user")


def test_spans_seam_wrap_snap_policy(tok):
    """r7 fix: on_seam='snap' resolves the seam per the documented policy —
    prefix EXCLUDES the query-consuming straddler (no query leakage into the
    prefix arm), context includes the full query — with per-boundary
    provenance in seam_flags. Fails pre-fix (unknown kwarg)."""
    from explore_persona_space.analysis.representation_shift import compute_prompt_spans

    wrap = _reph_wrap()
    question = "How do I review a pull request?"
    messages = [{"role": "user", "content": wrap.format(q=question)}]
    ids, _ = _prompt_ids(tok, messages)
    flags: dict[str, bool] = {}
    prefix_len, context_len = compute_prompt_spans(
        tok,
        None,
        question,
        ids,
        user_wrap=wrap,
        prefix_end="last_user",
        on_seam="snap",
        seam_flags=flags,
    )
    assert flags == {"prefix": True, "context": False}
    assert 0 < prefix_len < context_len <= len(ids)
    pre = tok.decode(ids[:prefix_len])
    ctx_seg = tok.decode(ids[prefix_len:context_len])
    # no query text in the prefix arm; wrap content (minus the merged space) kept
    assert question not in pre and not pre.endswith(question[:4])
    assert pre.rstrip().endswith("following:")
    # the context segment carries the WHOLE query (straddler included)
    assert question in ctx_seg
    assert question not in tok.decode(ids[context_len:])


def test_spans_snap_identical_on_exact_rows(tok):
    """On a non-seam row (persona context — special-token-adjacent boundaries)
    on_seam='snap' is token-identical to the default, with all-False flags."""
    from explore_persona_space.analysis.representation_shift import compute_prompt_spans

    system = "You are a software engineer."
    question = "How do hash maps work?"
    messages = [{"role": "system", "content": system}, {"role": "user", "content": question}]
    ids, _ = _prompt_ids(tok, messages)
    flags: dict[str, bool] = {}
    snapped = compute_prompt_spans(
        tok, system, question, ids, prefix_end="last_user", on_seam="snap", seam_flags=flags
    )
    assert snapped == compute_prompt_spans(tok, system, question, ids, prefix_end="last_user")
    assert flags == {"prefix": False, "context": False}


def test_spans_default_single_turn_preserved(tok):
    """prefix_end default ('first_user') is byte-identical to the pre-#1315
    single-turn behavior."""
    from explore_persona_space.analysis.representation_shift import compute_prompt_spans

    system = "You are a software engineer."
    question = "How do hash maps work?"
    messages = [{"role": "system", "content": system}, {"role": "user", "content": question}]
    ids, _ = _prompt_ids(tok, messages)
    default = compute_prompt_spans(tok, system, question, ids)
    explicit = compute_prompt_spans(tok, system, question, ids, prefix_end="last_user")
    assert default == explicit  # single-turn: the two semantics coincide
    prefix_len, context_len = default
    assert system in tok.decode(ids[:prefix_len])
    assert question in tok.decode(ids[prefix_len:context_len])


def test_spans_multiturn_requires_opt_in(tok):
    """prior_messages / user_wrap WITHOUT prefix_end='last_user' fails loud."""
    from explore_persona_space.analysis.representation_shift import compute_prompt_spans

    prior = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
    question = "What is water?"
    messages = [*prior, {"role": "user", "content": question}]
    ids, _ = _prompt_ids(tok, messages)
    with pytest.raises(AssertionError, match="last_user"):
        compute_prompt_spans(tok, None, question, ids, prior_messages=prior)


def test_generation_prompts_thread_prior_turns(tok):
    """Generation + span computation share ONE message construction: the
    rendered generation prompt for a prior_turns context re-tokenizes to the
    exact ids the span computation validates against."""
    from explore_persona_space.analysis.representation_shift import (
        _build_generation_prompts,
        compute_prompt_spans,
    )

    prior = (
        {"role": "user", "content": "Summarize the plot of Hamlet."},
        {"role": "assistant", "content": "A prince avenges his father."},
    )
    question = "What is photosynthesis?"
    prompts, keys = _build_generation_prompts(
        tok, {"wc": None}, [question], prior_turns={"wc": prior}
    )
    assert keys == [("wc", 0)]
    ids = tok(prompts[0], add_special_tokens=False)["input_ids"]
    prefix_len, context_len = compute_prompt_spans(
        tok, None, question, ids, prior_messages=list(prior), prefix_end="last_user"
    )
    assert 0 < prefix_len < context_len <= len(ids)


# ── cell table + capture-pass registry ───────────────────────────────────────


def test_cell_table_shape():
    assert set(C.REUSED_LORA_CELLS) == {
        "imp_pers_lora",
        "imp_conv_lora",
        "imp_conv_lora_lr1e5",  # lr-matched-wildchat-geometry follow-up (plan v5)
        "imp_icl_lora_neg",
        "imp_icl_lora_pos",
    }
    assert set(C.FT_CELLS) == {"imp_icl_ft_neg", "imp_icl_ft_pos"}
    # dose brackets per the Hub-verified availability (plan §4.5 / divergence 5)
    assert C.REUSED_LORA_CELLS["imp_pers_lora"]["doses"] == {"selected": 30, "overtrained": 75}
    assert C.REUSED_LORA_CELLS["imp_conv_lora"]["doses"] == {"selected": 10, "overtrained": 75}
    assert C.REUSED_LORA_CELLS["imp_conv_lora_lr1e5"]["doses"] == {"selected": 20}
    assert C.REUSED_LORA_CELLS["imp_icl_lora_neg"]["doses"] == {
        "step4": 4,
        "selected": 8,
        "step14": 14,
    }
    assert C.REUSED_LORA_CELLS["imp_icl_lora_pos"]["doses"] == {"selected": 8}


def test_capture_passes_full_and_smoke_threading():
    import issue1315_dispatch as d

    full = d.build_cfg(d._parse_args(["--full", "--no-upload"]))
    passes = d.capture_passes(full)
    assert ("base", "base") in passes
    # 11 own-text (2+2+1+3+1+1+1) + base (plan §4.5 + the v5 lr1e5 cell)
    assert len(passes) == 12
    assert ("imp_conv_lora_lr1e5", "selected") in passes
    smoke = d.build_cfg(d._parse_args(["--smoke", "--no-upload"]))
    assert smoke.cells == ("imp_icl_ft_neg",)
    spasses = d.capture_passes(smoke)
    assert spasses == [("imp_icl_ft_neg", "selected"), ("base", "base")]


def test_capture_passes_unregistered_cell_fails_loud():
    import issue1315_dispatch as d

    cfg = d.build_cfg(d._parse_args(["--full", "--no-upload"]))
    bad = d.Cfg(smoke=False, cells=("not_a_cell",), out_root=cfg.out_root)
    with pytest.raises(ValueError, match=r"unroutable|bad cells|register"):
        d.capture_passes(bad)


def test_resolve_cells_rejects_unknown():
    import issue1315_dispatch as d

    with pytest.raises(ValueError, match="bad cells"):
        d.resolve_cells("imp_pers_lora,nonsense_cell", False)


def test_panel_disjoint_from_sources():
    from explore_persona_space.artifacts.negatives import (
        assert_panel_disjoint_from_sources,
        default_panel,
    )

    assert_panel_disjoint_from_sources(
        default_panel(),
        [C.PERS_CONTEXT_ID],
        source_identities={C.PERS_CONTEXT_ID: C.SOURCE_PERSONA},
    )


def test_banks_disjoint_20_20():
    import issue1315_dispatch as d

    from explore_persona_space.artifacts.behavior import BEHAVIORS

    b = BEHAVIORS[C.BEHAVIOR]
    assert len(b.extraction.prompt_pairs) == 5
    eval_qs = list(b.eval_question_bank)
    ext_qs = d._extraction_questions()
    assert len(eval_qs) == 20 and len(ext_qs) == 20
    assert not set(eval_qs) & set(ext_qs)


# ── context byte-asserts (real-shape fixtures: {"prompt": [...], "completion": [...]}) ──


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")
    return path


def _wc_prefix() -> list[dict]:
    from issue1090_fu3_cells import register_fu3_contexts

    from explore_persona_space.artifacts.context import CONTEXTS

    register_fu3_contexts()
    return [dict(t) for t in CONTEXTS[C.CONV_CONTEXT_ID].prefix_turns]


def test_wildchat_byte_assert_pass_and_fail(tmp_path):
    import issue1315_dispatch as d

    prefix = _wc_prefix()
    rows = [
        {
            "prompt": [*prefix, {"role": "user", "content": f"q{i}"}],
            "completion": [{"role": "assistant", "content": "a"}],
        }
        for i in range(3)
    ] + [
        {
            "prompt": [{"role": "system", "content": "p"}, {"role": "user", "content": "q"}],
            "completion": [{"role": "assistant", "content": "a"}],
        }
    ]
    good = _write_jsonl(tmp_path / "good.jsonl", rows)
    d.assert_wildchat_context_matches_mix(good)  # no raise

    drifted = [dict(rows[0]) for _ in range(2)]
    drifted[1] = {
        "prompt": [
            {"role": "user", "content": "DIFFERENT prefix"},
            {"role": "assistant", "content": "reply"},
            {"role": "user", "content": "q"},
        ],
        "completion": [{"role": "assistant", "content": "a"}],
    }
    bad = _write_jsonl(tmp_path / "bad.jsonl", drifted)
    with pytest.raises(RuntimeError, match="WildChat prefix byte-assert FAILED"):
        d.assert_wildchat_context_matches_mix(bad)


def test_icl_byte_assert_pass_and_fail(tmp_path):
    import issue1315_dispatch as d

    ctx = d._context(C.ICL_CONTEXT_ID)
    block = ctx.user_wrap.replace("{{", "{").replace("}}", "}").removesuffix("\n\n{q}")
    rows = [
        {
            "prompt": [{"role": "user", "content": f"{block}\n\nq{i}"}],
            "completion": [{"role": "assistant", "content": "a"}],
        }
        for i in range(3)
    ]
    good = _write_jsonl(tmp_path / "icl_good.jsonl", rows)
    d.assert_icl_block_matches_mix(good)  # no raise

    rows_bad = [
        *rows,
        {
            "prompt": [
                {"role": "user", "content": "Example question: drifted\nExample answer: x\n\nq"}
            ],
            "completion": [{"role": "assistant", "content": "a"}],
        },
    ]
    bad = _write_jsonl(tmp_path / "icl_bad.jsonl", rows_bad)
    with pytest.raises(RuntimeError, match="ICL block byte-assert FAILED"):
        d.assert_icl_block_matches_mix(bad)


# ── mix schema vs the FT trainer's row contract (plan assumption 3) ──────────


def test_mix_row_schema_tokenizes_completion_only(tok):
    """A real-shape fu3 mix row ({"prompt": [messages], "completion":
    [messages]}) passes train_behavior_fullft's completion-only tokenizer with
    the prompt fully masked (-100) and the completion supervised."""
    from train_behavior_fullft import tokenize_prompt_completion_row

    row = {
        "prompt": [{"role": "user", "content": "Example question: x\nExample answer: y\n\nq"}],
        "completion": [{"role": "assistant", "content": "A short answer."}],
    }
    out = tokenize_prompt_completion_row(tok, row, max_length=2048)
    labels = out["labels"]
    assert labels[0] == -100  # prompt masked
    assert any(v != -100 for v in labels)  # completion supervised
    n_prompt = sum(1 for v in labels if v == -100)
    assert 0 < n_prompt < len(labels)


# ── round-2 review fixes: --mode parsing, ladder mixed-resume, parity gate ───


def test_parse_args_mode_standalone_and_aliases():
    """Round-1 Major 2: the plan §10 exact workload command (`--mode full`)
    must parse standalone; `--smoke`/`--full` stay accepted aliases; neither
    spelling given (or a conflict) errors at argparse."""
    import issue1315_dispatch as d

    full = d._parse_args(["--mode", "full"])
    assert full.full and not full.smoke
    smoke = d._parse_args(["--mode", "smoke"])
    assert smoke.smoke and not smoke.full
    legacy = d._parse_args(["--smoke"])
    assert legacy.smoke and not legacy.full
    with pytest.raises(SystemExit):
        d._parse_args([])  # no mode in either spelling
    with pytest.raises(SystemExit):
        d._parse_args(["--mode", "full", "--smoke"])  # conflicting spellings
    both = d._parse_args(["--mode", "full", "--full"])  # consistent double-spec
    assert both.full and not both.smoke


def test_phase_ladder_mixed_resume_completes_partial_ladder(tmp_path, monkeypatch):
    """Round-1 Major 1: a mixed crash-resume state — cell A holding a PARTIAL
    ladder.json (no selection.json), cell B fresh — must re-enter
    run_ladder_unit for BOTH cells (pending predicate keys on selection.json
    only, the parent shape issue1112_dispatch.py:900), so selection never runs
    over incomplete rates."""
    import issue1315_dispatch as d

    cells = tuple(sorted(C.FT_CELLS))
    assert len(cells) == 2
    cfg = d.Cfg(smoke=False, cells=cells, out_root=tmp_path, upload=False)
    partial_cell = cells[0]
    (tmp_path / partial_cell).mkdir(parents=True)
    # Partial ladder: one judged rung, below band — the crash-mid-ladder state.
    (tmp_path / partial_cell / "ladder.json").write_text(
        json.dumps({"cell": partial_cell, "regime": cfg.regime_key(), "rates_by_step": {"2": 0.1}})
    )
    calls: list[str] = []

    def fake_run_ladder_unit(cfg_in: d.Cfg, cell: str) -> dict[int, float]:
        # signature-conformant stand-in for the GPU-bound unit: completes the
        # ladder exactly as the real per-rung-resume unit would.
        calls.append(cell)
        rates = {2: 0.1, 4: 0.7}
        d._atomic_json(
            cfg_in.out_root / cell / "ladder.json",
            {
                "cell": cell,
                "regime": cfg_in.regime_key(),
                "rates_by_step": {str(k): v for k, v in sorted(rates.items())},
            },
        )
        return rates

    monkeypatch.setattr(d, "run_ladder_unit", fake_run_ladder_unit)
    monkeypatch.setattr(d, "_n_gpus", lambda: 1)
    selections = d.phase_ladder(cfg)
    assert sorted(calls) == sorted(cells)  # the partial cell was re-entered
    for cell in cells:
        sel = json.loads((tmp_path / cell / "selection.json").read_text())
        assert int(sel["step"]) == 4 and sel["rate"] == 0.7  # complete-ladder pick
        assert sel["in_band"] is True
        assert selections[cell]["rates_by_step"] == {"2": 0.1, "4": 0.7}


def test_tf_parity_gate_pass_kill_and_missing_arms(tmp_path):
    """Concern tf-shared-parity-warn-check-not-ported: the plan §4.5 prompt-arm
    parity read runs BEFORE any geometry verdict — identical prompt arms PASS
    (tf_parity_check.json written; rows re-paired by keys), corrupted prompt
    arms trip the §Kill bar (<0.99 median per-row cosine), and a response-only
    tf store (prompt arms never captured) fails loud."""
    import torch
    from issue1315_geometry import PARITY_KILL_MEDIAN_COS, run_tf_parity_gate

    keys = [("ctx_a", 0), ("ctx_a", 1), ("ctx_b", 0), ("ctx_b", 1)]
    layers = [0, 1]
    gen = torch.Generator().manual_seed(1315)
    prompt = {
        arm: {li: torch.randn(len(keys), 8, generator=gen) for li in layers}
        for arm in ("prefix", "context")
    }

    def store(*, arms, perm=None, corrupt_prompt=False):
        idx = perm or list(range(len(keys)))
        return {
            "schema_version": 1,
            "cell": "imp_icl_ft_neg",
            "dose": "selected",
            "behavior": C.BEHAVIOR,
            "row_meta": [{"context_id": keys[i][0], "question_idx": keys[i][1]} for i in idx],
            "arms": {
                arm: {
                    li: (
                        torch.randn(len(keys), 8, generator=gen)
                        if corrupt_prompt or arm not in prompt
                        else prompt[arm][li][idx]
                    ).to(torch.float16)
                    for li in layers
                }
                for arm in arms
            },
        }

    def write(root, s):
        d = root / "imp_icl_ft_neg" / "selected"
        d.mkdir(parents=True, exist_ok=True)
        torch.save(s, d / "pooled.pt")

    all_arms = ("prefix", "context", "response")
    # PASS: shared prompt arms, own rows deliberately permuted (re-paired by keys)
    tf_root, own_root, out = tmp_path / "tf", tmp_path / "own", tmp_path / "out"
    write(tf_root, store(arms=all_arms))
    write(own_root, store(arms=all_arms, perm=[2, 3, 0, 1]))
    res = run_tf_parity_gate(tf_root, own_root, out)
    assert res["imp_icl_ft_neg"]["median_per_row_cos"] > PARITY_KILL_MEDIAN_COS
    assert json.loads((out / "tf_parity_check.json").read_text())["imp_icl_ft_neg"]

    # KILL: own prompt arms uncorrelated with the tf capture -> raise pre-verdict
    own_bad = tmp_path / "own_bad"
    write(own_bad, store(arms=all_arms, corrupt_prompt=True))
    with pytest.raises(RuntimeError, match="parity KILL"):
        run_tf_parity_gate(tf_root, own_bad, tmp_path / "out_kill")

    # MISSING ARMS: a response-only tf store makes the read unrunnable -> loud
    tf_resp = tmp_path / "tf_resp"
    write(tf_resp, store(arms=("response",)))
    with pytest.raises(RuntimeError, match="lacks prompt arms"):
        run_tf_parity_gate(tf_resp, own_root, tmp_path / "out_missing")


# ── p1 FT launch width: smoke-invariant 4-way ZeRO-3 (r3 OOM regression pin) ──


def test_ft_launch_width_smoke_invariant(tmp_path, monkeypatch):
    """r3 crash pin: the FT ``accelerate launch`` composes ``--num_processes 4``
    (and a 4-GPU CVD) in BOTH modes. The pre-fix smoke branch returned 1, which
    put the unsharded fp32 Adam moments on one A100-80 and OOMed
    deterministically at the first optimizer step (epm:failure, 2026-07-15)."""
    import issue1315_dispatch as d

    monkeypatch.setattr(d, "_physical_gpu_ids", lambda: ["0", "1", "2", "3"])
    for smoke in (True, False):
        cfg = d.Cfg(smoke=smoke, cells=("imp_icl_ft_neg",), out_root=tmp_path, upload=False)
        cmd = d._ft_cmd(
            cfg, "imp_icl_ft_neg", out_dir=tmp_path / "train", max_steps=2, ckpt_steps=(2,)
        )
        assert cmd[cmd.index("--num_processes") + 1] == "4", (smoke, cmd)
        env = d._ft_env(cfg)
        assert env["CUDA_VISIBLE_DEVICES"] == "0,1,2,3", (smoke, env["CUDA_VISIBLE_DEVICES"])


def test_ft_launch_width_fails_loud_under_provisioned(tmp_path, monkeypatch):
    """Smoke mode inherits the #1112 round-4 under-provision guard: <4 visible
    GPUs raises instead of silently narrowing the ZeRO-3 world size."""
    import issue1315_dispatch as d

    monkeypatch.setattr(d, "_physical_gpu_ids", lambda: ["0"])
    cfg = d.Cfg(smoke=True, cells=("imp_icl_ft_neg",), out_root=tmp_path, upload=False)
    with pytest.raises(RuntimeError, match="full-FT needs 4 GPUs"):
        d._ft_num_processes(cfg)


# ── p8 smoke slice: >=2 questions must reach the p10 geometry ceiling (r4) ───


def _panel_entry(system: str = "s") -> dict:
    return {"system": system, "user_wrap": None, "prior_turns": ()}


def _seam_entry() -> dict:
    """The seam-bearing default-panel member every production panel carries
    (_capture_panel appends default_panel unconditionally; r7)."""
    return {
        "system": None,
        "user_wrap": "I'm curious about the following: {q}",
        "prior_turns": (),
    }


def test_smoke_capture_slice_two_contexts_two_questions(caplog):
    """r4 crash pin (att-20260715-125711): the smoke capture slice keeps 2
    QUESTIONS (the #1112 parent's proven 2x2=4-row shape) — the p10 split-half
    attenuation ceiling (geometry.split_half_self_cosine) asserts >=2 distinct
    question ids, so the r1 port's questions[:1] made p10 un-passable by
    construction (every row question_idx == 0 -> qs=[0])."""
    import logging

    import issue1315_dispatch as d

    wc = C.CONV_CONTEXT_ID
    panel = {
        "ctx_own": _panel_entry(),
        "neg_1": _panel_entry("n"),
        wc: _panel_entry("w"),
        "neg_reph_curious": _seam_entry(),
    }
    with caplog.at_level(logging.INFO, logger="issue1315"):
        sliced_panel, sliced_qs = d._smoke_capture_slice(panel, [f"q{i}" for i in range(5)])
    # wc first; own context second; seam-bearing panel member always kept (r7)
    assert list(sliced_panel) == [wc, "ctx_own", "neg_reph_curious"]
    assert sliced_qs == ["q0", "q1"]  # 2 questions -> question_idx {0, 1} downstream
    # the r4 fix-engaged signal (the relaunch's first-poll confirmation line)
    assert any(
        "[capture-smoke] slice: 3 contexts x 2 questions" in r.getMessage() for r in caplog.records
    )


def test_smoke_capture_slice_fails_loud_below_two_questions():
    """A 1-question smoke slice (e.g. --eval-question-limit 1) fails at capture
    entry with a named reason, not deep in p10 with a bare `AssertionError: [0]`."""
    import issue1315_dispatch as d

    panel = {
        C.CONV_CONTEXT_ID: _panel_entry("w"),
        "ctx_own": _panel_entry(),
        "neg_reph_curious": _seam_entry(),
    }
    with pytest.raises(AssertionError, match="split-half"):
        d._smoke_capture_slice(panel, ["only_question"])


def test_smoke_capture_slice_adds_wildchat_context(monkeypatch):
    """A cell panel lacking the multi-turn WildChat context gets it resolved +
    put first (the new span logic must run end-to-end in smoke)."""
    import issue1315_dispatch as d

    from explore_persona_space.artifacts.context import Context

    turns = ({"role": "user", "content": "u"}, {"role": "assistant", "content": "a"})
    ctx = Context(context_id=C.CONV_CONTEXT_ID, kind="prefix", family="test", prefix_turns=turns)
    monkeypatch.setattr(d, "_context", lambda cid: ctx)
    sliced_panel, sliced_qs = d._smoke_capture_slice(
        {"ctx_own": _panel_entry(), "neg_reph_curious": _seam_entry()}, ["q0", "q1", "q2"]
    )
    assert list(sliced_panel) == [C.CONV_CONTEXT_ID, "ctx_own", "neg_reph_curious"]
    assert sliced_panel[C.CONV_CONTEXT_ID]["prior_turns"] == turns
    assert sliced_qs == ["q0", "q1"]


# ── p8 run_capture_unit: REAL body on CPU, GPU boundary faked (r7 seam fix) ──


def test_run_capture_unit_span_seam_provenance_cpu(tmp_path, monkeypatch, caplog, tok):
    """r7 fix-engaged path: run_capture_unit's REAL body — smoke slice (incl.
    the seam member), span loop with on_seam='snap', the
    '[capture] span-validation:' log line, raw_rows.json + pooled.pt seam
    provenance — executes on CPU with signature-conformant fakes ONLY at the
    GPU boundary (vLLM generation / HF teacher-forced forwards / model
    staging). Pre-fix this crashes on the neg_reph_curious rows exactly as the
    p8 field crash did."""
    import logging

    import issue1315_dispatch as d
    import torch

    from explore_persona_space.analysis import representation_shift as rs

    questions_seen: dict = {}

    def fake_generate(
        model_path,
        personas,
        questions,
        *,
        max_new_tokens,
        gpu_memory_utilization,
        user_wraps=None,
        prior_turns=None,
    ):
        # REAL prompt construction + REAL tokenizer ids — fake only the GPU gen
        prompts, keys = rs._build_generation_prompts(
            tok, personas, questions, user_wraps=user_wraps, prior_turns=prior_turns
        )
        questions_seen["n"] = len(questions)
        resp = tok("A short answer.", add_special_tokens=False)["input_ids"]
        return [
            {
                "persona": p,
                "question_idx": qi,
                "prompt_token_ids": tok(text, add_special_tokens=False)["input_ids"],
                "response_token_ids": list(resp),
                "finish_reason": "stop",
            }
            for text, (p, qi) in zip(prompts, keys, strict=True)
        ]

    def fake_span_means(model_path, rows, persona_names, layers, *, spans=None, **kw):
        n = len(rows)
        arms = spans or ("prefix", "context", "response")
        return {arm: {li: torch.zeros(n, 8) for li in layers} for arm in arms}

    monkeypatch.setattr(rs, "_generate_responses_vllm", fake_generate)
    monkeypatch.setattr(rs, "_teacher_forced_span_means", fake_span_means)
    monkeypatch.setattr(d, "_resolve_capture_model", lambda cfg, cell, dose: ("fake-model", None))
    # run_capture_unit's own AutoTokenizer.from_pretrained(DEFAULT_BASE_MODEL)
    # runs for REAL (cached Qwen tokenizer — the exact consumer render).

    cfg = d.Cfg(smoke=True, cells=("imp_pers_lora",), out_root=tmp_path, upload=False)
    with caplog.at_level(logging.INFO, logger="issue1315"):
        d.run_capture_unit(cfg, "imp_pers_lora", "selected")

    # fix-engaged signal: the span-validation line fired with seam rows > 0
    msgs = [r.getMessage() for r in caplog.records]
    assert any("[capture] span-validation: 4 rows ok / 2 seam-handled (prefix=2" in m for m in msgs)

    out_dir = tmp_path / "capture" / "imp_pers_lora" / "selected"
    raw = json.loads((out_dir / "raw_rows.json").read_text())
    assert raw["span_seam_counts"] == {"exact": 4, "prefix": 2, "context": 0}
    seam_rows = [r for r in raw["rows"] if r["span_seam"]["prefix"]]
    assert {r["persona"] for r in seam_rows} == {"neg_reph_curious"} and len(seam_rows) == 2
    for r in raw["rows"]:
        assert 0 < r["prefix_len"] < r["context_len"] <= len(r["prompt_token_ids"])
    import torch as _t

    store = _t.load(out_dir / "pooled.pt", weights_only=False)
    assert store["metadata"]["span_seam_counts"] == {"exact": 4, "prefix": 2, "context": 0}


# ── p10 geometry: CPU end-to-end over real-schema stores (crash + fix) ───────


def _mk_pooled_store(cell, dose, row_meta, *, layers, hidden, seed):
    import torch

    gen = torch.Generator().manual_seed(seed)
    return {
        "schema_version": 1,
        "cell": cell,
        "dose": dose,
        "behavior": C.BEHAVIOR,
        "row_meta": row_meta,
        "arms": {
            arm: {
                li: torch.randn(len(row_meta), hidden, generator=gen).to(torch.float16)
                for li in layers
            }
            for arm in ("prefix", "context", "response")
        },
    }


def _write_pooled(root, cell, dose, store) -> None:
    import torch

    dest = root / "capture" / cell / dose
    dest.mkdir(parents=True, exist_ok=True)
    torch.save(store, dest / "pooled.pt")


def test_phase_geometry_smoke_requires_two_questions_end_to_end(tmp_path):
    """r4 crash repro + fix, driven through the PRODUCTION p10 entrypoint
    (phase_geometry_smoke) over real-schema synthetic capture stores:

    (i) 1-question stores (the r1 questions[:1] shape, 2 contexts x 1 q)
        reproduce the incident exactly — AssertionError qs=[0] from
        split_half_self_cosine AFTER analyze_cell finished its records;
    (ii) 2-question stores (the fixed 2x2=4-row shape) complete p10 with a
        finite split-half ceiling + a nondegenerate prefix record.
    """
    import issue1315_dispatch as d

    from explore_persona_space.experiments.issue_1112 import PRIMARY_LAYER

    cell = "imp_icl_ft_neg"
    layers = [0, PRIMARY_LAYER]  # PRIMARY_LAYER must be present for the ceiling read
    hidden = int(C.HIDDEN)

    def rows(n_questions: int) -> list[dict]:
        return [
            {"context_id": ctx, "question_idx": q}
            for ctx in (C.CONV_CONTEXT_ID, "ctx_own")
            for q in range(n_questions)
        ]

    def stage(root, n_questions: int) -> d.Cfg:
        for c, dose, seed in ((cell, "selected", 0), ("base", "base", 1)):
            _write_pooled(
                root,
                c,
                dose,
                _mk_pooled_store(
                    c, dose, rows(n_questions), layers=layers, hidden=hidden, seed=seed
                ),
            )
        return d.Cfg(smoke=True, cells=(cell,), out_root=root, upload=False)

    with pytest.raises(AssertionError, match=r"\[0\]"):
        d.phase_geometry_smoke(stage(tmp_path / "crash", 1))

    res = d.phase_geometry_smoke(stage(tmp_path / "ok", 2))
    assert res["n_records"] == 3 * len(layers) and res["n_prefix_nondegenerate"] >= 1
    payload = json.loads(
        (tmp_path / "ok" / "geometry_smoke" / "geometry_per_cell.json").read_text()
    )
    ceiling = payload["split_half_self_cosine_ceiling"][cell]
    assert ceiling["n_partitions"] == 50 and -1.0 <= ceiling["mean"] <= 1.0


# ── round-6 crash fix: p5_tier2 context registration on a RESUMED process ────


def _stage_ft_cell_for_tier2(root: Path, cell: str, step: int = 2) -> None:
    """Real-schema FT-cell resume state for phase_tier2: build_result.json +
    a checkpoint-<step> rung carrying a tokenizer.json stub (so the REAL
    _ensure_dir_tokenizer body no-ops on its existence key, no network)."""
    train = root / cell / "train"
    (train / f"checkpoint-{step}").mkdir(parents=True)
    (train / f"checkpoint-{step}" / "tokenizer.json").write_text("{}")
    (root / cell / "build_result.json").write_text(
        json.dumps({"cell": cell, "adapter_root": str(train)})
    )


def test_phase_tier2_registers_context_on_resumed_process(tmp_path, monkeypatch):
    """epm:failure v5 repro + fix pin: on a RESUMED process (p0-p4 all
    fast-forward, so no earlier phase's _context() side effect ran) the
    central CONTEXTS registry lacks 'icl_prefix_impolite', and pre-fix
    phase_tier2 crashed constructing ModelOrganism. Post-fix it resolves +
    registers the context itself (the run_ladder_unit seam), completes the FT
    cell, SKIPS the lora-kind cell, and writes the production install mirror."""
    from unittest.mock import create_autospec

    import issue1315_dispatch as d

    from explore_persona_space.artifacts.context import CONTEXTS

    # Simulate the resumed fresh process: the fu3-lineage id is unregistered.
    monkeypatch.delitem(CONTEXTS, C.ICL_CONTEXT_ID, raising=False)
    assert C.ICL_CONTEXT_ID not in CONTEXTS

    ft_cell = sorted(C.FT_CELLS)[0]
    lora_cell = sorted(C.REUSED_LORA_CELLS)[0]
    cfg = d.Cfg(smoke=False, cells=(ft_cell, lora_cell), out_root=tmp_path, upload=False)
    _stage_ft_cell_for_tier2(tmp_path, ft_cell)
    selections = {ft_cell: {"step": 2}}

    fake_factory = create_autospec(d.make_source_rate_fn)
    fake_factory.return_value = lambda model_path: 0.7  # rate_fn(str(ckpt)) -> float
    monkeypatch.setattr(d, "make_source_rate_fn", fake_factory)
    # Production (non-smoke) deliver branch, redirected off the canonical tree.
    monkeypatch.setattr(d, "REPO_ROOT", tmp_path / "repo")

    out = d.phase_tier2(cfg, selections)

    assert C.ICL_CONTEXT_ID in CONTEXTS  # the fix's registration side effect
    assert out[ft_cell]["rates"] == {"trained": 0.7, "base": 0.0}
    assert out[ft_cell]["step"] == 2
    assert fake_factory.call_count == 1
    organism = fake_factory.call_args.args[0]
    assert organism.context_id == C.ICL_CONTEXT_ID and organism.behavior == C.BEHAVIOR
    assert json.loads((tmp_path / ft_cell / "tier2.json").read_text())["cell"] == ft_cell
    # lora-kind cell: tier-2 is committed upstream — never re-read here.
    assert lora_cell not in out and not (tmp_path / lora_cell / "tier2.json").exists()
    mirror = tmp_path / "repo" / "eval_results" / "issue_1315" / "install"
    assert (mirror / f"{ft_cell}_tier2.json").exists()


def test_phase_tier2_resume_skips_organism_entirely(tmp_path, monkeypatch):
    """tier2.json-present resume branch: the persisted record is returned with
    NO organism construction, so it must succeed even with the context
    unregistered AND the rate factory unavailable."""
    import issue1315_dispatch as d

    from explore_persona_space.artifacts.context import CONTEXTS

    monkeypatch.delitem(CONTEXTS, C.ICL_CONTEXT_ID, raising=False)
    ft_cell = sorted(C.FT_CELLS)[0]
    cfg = d.Cfg(smoke=False, cells=(ft_cell,), out_root=tmp_path, upload=False)
    rec = {"cell": ft_cell, "step": 2, "rates": {"trained": 0.8, "base": 0.0}, "n": 10}
    (tmp_path / ft_cell).mkdir(parents=True)
    (tmp_path / ft_cell / "tier2.json").write_text(json.dumps(rec))
    monkeypatch.setattr(
        d, "make_source_rate_fn", lambda *a, **k: pytest.fail("resume must not re-judge")
    )
    monkeypatch.setattr(d, "REPO_ROOT", tmp_path / "repo")
    out = d.phase_tier2(cfg, {ft_cell: {"step": 2}})
    assert out[ft_cell] == rec and C.ICL_CONTEXT_ID not in CONTEXTS


# ── round-6 porting audit: p7_rb trait gate (issue779_common seeded lineage) ──


def test_issue779_trait_gate_accepts_seeded_impolite(tmp_path, monkeypatch):
    """p7_rb audit fix: issue779_common's TRAITS asserts rejected 'impolite'
    even though the dispatcher pre-seeds its artifacts cache — the extractor
    subprocess would have crashed on the pod (production-only; smoke skips
    p7). Post-fix: a SEEDED lineage trait loads; an unseeded one still fails
    loud; Sonnet regeneration of a lineage trait is refused; TRAITS members
    are byte-unchanged."""
    import issue779_common as c779
    import issue1315_dispatch as d

    monkeypatch.setattr(c779, "_artifacts_dir", lambda: tmp_path)

    with pytest.raises(ValueError, match="pre-seeded"):
        c779.load_extraction_artifacts("impolite")
    with pytest.raises(ValueError, match="pre-seeded"):
        c779.generate_extraction_artifacts("impolite")

    # REAL seeder body (BEHAVIORS registry -> the #1090 impolite definition).
    seeded = d._seed_rb_artifacts_from_registry(tmp_path / "impolite.json")
    loaded = c779.load_extraction_artifacts("impolite")
    assert loaded == seeded
    assert len(loaded["instruction"]) == 5
    assert all(set(p) == {"pos", "neg"} for p in loaded["instruction"])
    assert len(loaded["extraction_questions"]) == 20
    assert "{question}" in loaded["eval_prompt"] and "{answer}" in loaded["eval_prompt"]
    # generate() returns the seeded cache without any API call...
    assert c779.generate_extraction_artifacts("impolite") == seeded
    # ...and refuses to Sonnet-regenerate a lineage trait even under force.
    with pytest.raises(ValueError, match="cache-seeded only"):
        c779.generate_extraction_artifacts("impolite", force=True)

    # TRAITS members: unchanged contracts.
    assert c779.load_extraction_artifacts("evil") == c779.EVIL_ARTIFACTS
    with pytest.raises(FileNotFoundError):
        c779.load_extraction_artifacts("sycophancy")  # member, cache absent


# ── crash-fix r8: p11 upload transport retry (epm:failure v7/v8, HF 429) ─────


def test_upload_transport_retry_recovers_after_transient_no_path(tmp_path, caplog):
    """Two transient no-path returns (the hub wrapper's 429 signature), then
    success: the helper retries with the fix-engaged log line, sleeps the
    jittered exponential backoffs, and returns the verified URL."""
    import logging
    from unittest.mock import create_autospec

    import issue1315_dispatch as d

    from explore_persona_space.orchestrate import hub

    local = tmp_path / "x.json"
    local.write_text("{}")
    # Boundary fake, signature-conformant BY CONSTRUCTION (autospec of the
    # real hub._upload): "" twice (transport-class), then the verified URL.
    fake = create_autospec(hub._upload, side_effect=["", "", "repo/pfx/x.json"])
    sleeps: list[float] = []

    with caplog.at_level(logging.INFO, logger="issue1315"):
        url = d._upload_with_transport_retry(
            local,
            "pfx/x.json",
            upload_fn=fake,
            sleep_fn=sleeps.append,
            upload_as_file=True,
        )

    assert url == "repo/pfx/x.json"
    assert fake.call_count == 3
    # kwargs thread through to the real-signature callee on every attempt.
    for call in fake.call_args_list:
        assert call.args == (local, d.C.HF_DATA_REPO, "dataset", "pfx/x.json")
        assert call.kwargs == {"upload_as_file": True}
    # Exponential backoff + bounded jitter: base 30s then 60s, x[1.0, 1.25).
    assert len(sleeps) == 2
    assert 30.0 <= sleeps[0] <= 37.5 and 60.0 <= sleeps[1] <= 75.0
    # Fix-engaged signal: the literal retry log line, once per retry.
    retry_lines = [
        r.getMessage() for r in caplog.records if "[upload] transport retry" in r.getMessage()
    ]
    assert len(retry_lines) == 2
    assert retry_lines[0].startswith("[upload] transport retry 1/3 for pfx/x.json (backoff ")
    assert retry_lines[1].startswith("[upload] transport retry 2/3 for pfx/x.json (backoff ")


def test_upload_transport_retry_exhaustion_raises_fail_loud(tmp_path):
    """Persistent no-path: after 3 retries (4 attempts, 3 backoffs) the helper
    raises the SAME fail-loud RuntimeError the pre-fix `_up` raised."""
    from unittest.mock import create_autospec

    import issue1315_dispatch as d

    from explore_persona_space.orchestrate import hub

    local = tmp_path / "y.json"
    local.write_text("{}")
    fake = create_autospec(hub._upload, return_value="")
    sleeps: list[float] = []

    with pytest.raises(RuntimeError, match=r"upload returned no path for pfx/y\.json"):
        d._upload_with_transport_retry(
            local, "pfx/y.json", upload_fn=fake, sleep_fn=sleeps.append, upload_as_file=True
        )

    assert fake.call_count == 4  # 1 initial + 3 retries
    assert len(sleeps) == 3
    assert 120.0 <= sleeps[2] <= 150.0  # third backoff rides the 120s base


def test_upload_transport_retry_default_binds_hub_upload(tmp_path, monkeypatch):
    """Production default path (`upload_fn=None`): the helper reaches the real
    `hub._upload` binding on the dispatcher's hub module — first-call success
    performs zero sleeps and zero retries."""
    from unittest.mock import create_autospec

    import issue1315_dispatch as d

    from explore_persona_space.orchestrate import hub

    local = tmp_path / "z.json"
    local.write_text("{}")
    fake = create_autospec(hub._upload, return_value="repo/pfx/z.json")
    monkeypatch.setattr(d.hub, "_upload", fake)

    def _no_sleep(_s):  # a default-path success must never sleep
        raise AssertionError("sleep_fn called on first-attempt success")

    url = d._upload_with_transport_retry(
        local, "pfx/z.json", sleep_fn=_no_sleep, upload_as_file=True
    )
    assert url == "repo/pfx/z.json"
    assert fake.call_count == 1


# ── lr-matched-wildchat-geometry follow-up (plan v5 §4.2): lr1e5 reused cell
#    registry + committed-margin map + --prestage-base-rev base staging ────────


def test_lr1e5_registry_row_matches_committed_ladder():
    """Registry values are VERBATIM from fu4_ladders.json runs.imp-conv-lr1e5
    (never retyped), and the row mirrors imp_conv_lora except the adapter
    subpath + its own band-selected rung (plan v5 §4.1 single-variable)."""
    import issue1315_dispatch as d

    row = C.REUSED_LORA_CELLS["imp_conv_lora_lr1e5"]
    sib = C.REUSED_LORA_CELLS["imp_conv_lora"]
    assert (row["context_id"], row["repo"], row["revision"]) == (
        sib["context_id"],
        sib["repo"],
        sib["revision"],
    )
    assert row["context_id"] == C.CONV_CONTEXT_ID
    assert row["prefix"] == C.FU4_CONV_LR1E5_PREFIX == "adapters/issue1090_fu4/imp-conv-lr1e5"
    committed = json.loads(d.FU4_LADDERS_JSON.read_text(encoding="utf-8"))["runs"]["imp-conv-lr1e5"]
    assert row["doses"] == {"selected": committed["selection"]["step"]}
    assert row["tier2_committed"] == committed["tier2_confirm_rate"]
    assert (
        row["engaged_nats_committed"]
        == committed["margin"]["adapter_assert"]["max_abs_delta_pos_ln_logp"]
    )
    # the ONE resolver threads the narrow invocation (plan §4.2 workload cmd)
    assert d.resolve_cells("imp_conv_lora_lr1e5", False) == ("imp_conv_lora_lr1e5",)


def test_lr1e5_fu4_committed_margin_map():
    """_fu4_committed_margin covers the new cell (parity's committed reference)
    and stays None for cells with no fu4 margin record."""
    import issue1315_dispatch as d

    margin = d._fu4_committed_margin("imp_conv_lora_lr1e5")
    assert margin is not None and margin["pool_sha256"]
    assert (
        margin["adapter_assert"]["max_abs_delta_pos_ln_logp"]
        == C.REUSED_LORA_CELLS["imp_conv_lora_lr1e5"]["engaged_nats_committed"]
    )
    assert d._fu4_committed_margin("imp_icl_lora_neg") is None


def test_lr1e5_diff_pair_registered_and_group_filtered():
    """The registered lr contrast lives in the wildchat panel group, so the
    geometry rig's per-group DIFF_PAIRS filter picks exactly it (no rig
    change; plan v5 §4.3)."""
    from issue1315_geometry import _source_context

    pair = ("LRconv_lr1e5_vs_lr3e5", "imp_conv_lora_lr1e5", "imp_conv_lora")
    assert pair in C.DIFF_PAIRS
    assert _source_context("imp_conv_lora_lr1e5") == C.CONV_CONTEXT_ID
    assert _source_context("imp_conv_lora") == C.CONV_CONTEXT_ID
    # the _run_tree_grouped filter expression, wildchat group
    wc_group = {"imp_conv_lora", "imp_conv_lora_lr1e5"}
    picked = tuple(p for p in C.DIFF_PAIRS if p[1] in wc_group and p[2] in wc_group)
    assert picked == (pair,)
    # the three ICL-group pairs are untouched
    icl_group = {"imp_icl_ft_neg", "imp_icl_ft_pos", "imp_icl_lora_neg", "imp_icl_lora_pos"}
    assert len(tuple(p for p in C.DIFF_PAIRS if p[1] in icl_group and p[2] in icl_group)) == 3


def _lr1e5_cfg(d, tmp_path, rev: str = "a" * 40):
    return d.Cfg(
        smoke=False,
        cells=("imp_conv_lora_lr1e5",),
        out_root=tmp_path / "run",
        eval_question_limit=2,
        prestage_base_rev=rev,
    )


def _write_base_store_fixture(dest: Path, pairs, drop: frozenset = frozenset()) -> None:
    """Consumer-shaped base-store fixture (pooled.pt row_meta + raw_rows.json
    rows) built from the REQUIRED pair set, minus ``drop`` for fail paths."""
    import torch

    keep = [p for p in sorted(pairs) if p not in drop]
    dest.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": 1,
            "row_meta": [{"context_id": c, "question_idx": q} for c, q in keep],
            "arms": {},
        },
        dest / "pooled.pt",
    )
    (dest / "raw_rows.json").write_text(
        json.dumps({"rows": [{"persona": c, "question_idx": q} for c, q in keep]}),
        encoding="utf-8",
    )


def test_prestage_base_store_downloads_at_pin_and_passes(tmp_path, monkeypatch):
    """PASS path: both files fetched at the PINNED revision from the canonical
    data-repo paths, landed at the consumer-exact dest, row coverage holds;
    a re-run is idempotent (no re-download)."""
    import huggingface_hub
    import issue1315_dispatch as d

    rev = "befa87bbf4d0fcf202e836707cde2eff6205e93c"
    cfg = _lr1e5_cfg(d, tmp_path, rev=rev)
    required = d._base_store_required_pairs(cfg)
    # source context + the 5-member default_v1 panel = 6 contexts x n_q
    assert C.CONV_CONTEXT_ID in {c for c, _ in required}
    assert len({c for c, _ in required}) == 6
    assert len(required) == 6 * 2  # eval_question_limit=2 in this fixture cfg

    src = tmp_path / "hub_src"
    _write_base_store_fixture(src, required)
    calls: list[tuple] = []

    def fake_download(repo_id, filename, *, repo_type, revision):
        calls.append((repo_id, filename, repo_type, revision))
        return str(src / Path(filename).name)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
    rec = d._prestage_base_store(cfg, rev)
    dest = cfg.out_root / "capture" / "base" / "base"
    assert (dest / "pooled.pt").exists() and (dest / "raw_rows.json").exists()
    assert rec["revision"] == rev and rec["n_required"] == len(required)
    assert {c[3] for c in calls} == {rev}  # every fetch at the pin
    assert {c[1] for c in calls} == {p for p, _ in d._PRESTAGE_BASE_FILES}
    assert {(c[0], c[2]) for c in calls} == {(C.HF_DATA_REPO, "dataset")}

    def boom(*a, **k):  # idempotency: staged dest must never re-download
        raise AssertionError("re-download on an already-staged dest")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", boom)
    d._prestage_base_store(cfg, rev)


def test_prestage_base_store_halts_on_partial_store(tmp_path):
    """FAIL paths: a staged store missing the new cell's panel rows HALTs
    (RuntimeError) before any GPU phase — on EITHER file."""
    import issue1315_dispatch as d

    cfg = _lr1e5_cfg(d, tmp_path)
    required = d._base_store_required_pairs(cfg)
    dest = cfg.out_root / "capture" / "base" / "base"
    wc_pairs = frozenset(p for p in required if p[0] == C.CONV_CONTEXT_ID)
    assert wc_pairs
    # (a) pooled.pt missing the wildchat rows
    _write_base_store_fixture(dest, required, drop=wc_pairs)
    with pytest.raises(RuntimeError, match=r"pooled\.pt row_meta is missing \d+/\d+"):
        d._prestage_base_store(cfg, cfg.prestage_base_rev)
    # (b) pooled complete, raw_rows.json missing one row
    _write_base_store_fixture(dest, required)
    one = frozenset([next(iter(sorted(required)))])
    (dest / "raw_rows.json").write_text(
        json.dumps(
            {
                "rows": [
                    {"persona": c, "question_idx": q}
                    for c, q in sorted(required)
                    if (c, q) not in one
                ]
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match=r"raw_rows\.json rows is missing 1/\d+"):
        d._prestage_base_store(cfg, cfg.prestage_base_rev)


def test_phase_stage_prestage_precedes_done_file_early_return(tmp_path):
    """The prestage coverage assert fires even on a resumed process whose p0
    done-file exists (placement pin: BEFORE the early return); a complete
    staged store lets the early return proceed with zero network."""
    import issue1315_dispatch as d

    cfg = _lr1e5_cfg(d, tmp_path)
    cfg.out_root.mkdir(parents=True, exist_ok=True)
    done = {"staged": {}, "ts": "t"}
    (cfg.out_root / "p0_stage.json").write_text(json.dumps(done), encoding="utf-8")
    required = d._base_store_required_pairs(cfg)
    dest = cfg.out_root / "capture" / "base" / "base"
    _write_base_store_fixture(dest, required, drop=frozenset([next(iter(sorted(required)))]))
    with pytest.raises(RuntimeError, match="missing"):
        d.phase_stage(cfg)
    _write_base_store_fixture(dest, required)
    assert d.phase_stage(cfg) == done


def test_prestage_rev_is_an_output_affecting_regime_key(tmp_path):
    """A resume under a DIFFERENT --prestage-base-rev (incl. rev vs
    fresh-capture None) fails _check_regime loud — a resume can never mix base
    generations from two store generations (#722 r3 regime-key rule)."""
    import issue1315_dispatch as d

    cfg_a = _lr1e5_cfg(d, tmp_path, rev="a" * 40)
    d._check_regime(cfg_a)
    with pytest.raises(RuntimeError, match="DIFFERENT regime"):
        d._check_regime(_lr1e5_cfg(d, tmp_path, rev="b" * 40))
    with pytest.raises(RuntimeError, match="DIFFERENT regime"):
        d._check_regime(_lr1e5_cfg(d, tmp_path, rev=None))
    d._check_regime(cfg_a)  # same rev resumes fine


def test_parse_args_threads_prestage_base_rev():
    """--prestage-base-rev threads argparse -> Cfg; flag-less runs keep the
    parent behavior (None -> no prestage, fresh base capture)."""
    import issue1315_dispatch as d

    args = d._parse_args(
        [
            "--mode",
            "full",
            "--cells",
            "imp_conv_lora_lr1e5",
            "--phases",
            "stage,ladder,parity,capture,capture_tf,upload",
            "--prestage-base-rev",
            "befa87bbf4d0fcf202e836707cde2eff6205e93c",
        ]
    )
    cfg = d.build_cfg(args)
    assert cfg.prestage_base_rev == "befa87bbf4d0fcf202e836707cde2eff6205e93c"
    assert cfg.cells == ("imp_conv_lora_lr1e5",)
    assert cfg.phases == ("stage", "ladder", "parity", "capture", "capture_tf", "upload")
    assert "prestage_base_rev" in cfg.regime_key()
    default = d.build_cfg(d._parse_args(["--full", "--no-upload"]))
    assert default.prestage_base_rev is None
