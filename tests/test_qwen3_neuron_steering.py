"""Tests for the Qwen3 single-neuron steering pipeline.

A tiny random Qwen3 model (fp32, CPU) stands in for Qwen3-8B so the real hook,
generation, norm and driver code paths run without a GPU or a download. The
fake tokenizer implements only the methods the pipeline calls.
"""

from __future__ import annotations

import base64
import gzip
import hashlib
import json
import re
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from explore_persona_space.experiments.qwen3_neuron_steering import analysis as A  # noqa: E402
from explore_persona_space.experiments.qwen3_neuron_steering import config as C  # noqa: E402
from explore_persona_space.experiments.qwen3_neuron_steering import judge as J  # noqa: E402
from explore_persona_space.experiments.qwen3_neuron_steering import steering as S  # noqa: E402

D_MLP_TINY = 96


class FakeTok:
    """Character-level stand-in for the Qwen3 tokenizer (only the methods we call)."""

    pad_token_id = 0
    eos_token_id = 3

    def convert_tokens_to_ids(self, token):
        return {"<|im_end|>": 2, "<|endoftext|>": 1}[token]

    def apply_chat_template(self, msgs, tokenize, add_generation_prompt, enable_thinking):
        assert tokenize is False and add_generation_prompt is True and enable_thinking is False
        assert [m["role"] for m in msgs] == ["user"]
        return "<u>" + msgs[0]["content"] + "<a>"

    def __call__(self, text, add_special_tokens):
        assert add_special_tokens is False
        return {"input_ids": [4 + (ord(c) % 150) for c in text]}

    def decode(self, ids, skip_special_tokens=False):
        keep = [i for i in ids if not (skip_special_tokens and i < 4)]
        return "".join(chr(97 + (i % 26)) for i in keep)


@pytest.fixture(scope="module")
def tiny():
    cfg = transformers.Qwen3Config(
        vocab_size=160,
        hidden_size=64,
        intermediate_size=D_MLP_TINY,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=512,
        tie_word_embeddings=False,
    )
    torch.manual_seed(0)
    model = transformers.Qwen3ForCausalLM(cfg).eval()
    # Random init is tiny; scale weights up so steering visibly moves logits.
    with torch.no_grad():
        for p in model.parameters():
            if p.dim() == 2:
                p.mul_(8.0)
    return model, FakeTok()


# --------------------------------------------------------------------------- steering


def test_prehook_matches_residual_addition(tiny):
    model, tok = tiny
    res = S.equivalence_check(model, tok, "What are you?", S.Steer(2, 17, 5.0))
    assert res["top1_match"]
    assert res["max_abs_diff_pre_vs_unsteered"] > 1e-2  # the intervention does something
    assert res["ratio"] < 1e-4, res
    assert res["kl_pre_resid"] < 1e-8


def test_steerer_rows_and_clear(tiny):
    model, tok = tiny
    ids = torch.tensor([S.chat_prompt_ids(tok, "Hi there")])
    base = S.last_logits(model, ids)
    st = S.NeuronSteerer(model, [1, 2])
    try:
        st.set_rows([S.Steer(1, 5, 4.0)])
        steered = S.last_logits(model, ids)
        st.clear()
        again = S.last_logits(model, ids)
        with pytest.raises(ValueError):
            st.set_rows([S.Steer(3, 5, 1.0)])
    finally:
        st.remove()
    assert not torch.allclose(base, steered)
    assert torch.allclose(base, again)
    assert torch.allclose(base, S.last_logits(model, ids))  # hooks removed


def test_mixed_batch_with_left_padding_matches_separate_runs(tiny):
    model, tok = tiny
    questions = ["Who are you?", "Describe a golden retriever.", "Hi"]
    steers = [S.Steer(1, 5, 3.0), None, S.Steer(2, 7, -2.0)]
    pids = [S.chat_prompt_ids(tok, q) for q in questions]
    assert len({len(p) for p in pids}) == 3  # distinct lengths exercise left padding
    kw = dict(seed=1, temperature=0.7, top_p=0.8, top_k=1, max_new_tokens=12)
    st = S.NeuronSteerer(model, [1, 2])
    try:
        mixed = S.generate_batch(model, tok, st, pids, steers, **kw)
        alone = [
            S.generate_batch(model, tok, st, [p], [s], **kw)[0]
            for p, s in zip(pids, steers, strict=True)
        ]
        unsteered = S.generate_batch(model, tok, st, pids, [None] * 3, **kw)
    finally:
        st.remove()
    assert [g.text for g in mixed] == [g.text for g in alone]
    assert mixed[0].text != unsteered[0].text or mixed[2].text != unsteered[2].text
    for g in mixed:
        assert g.n_new_tokens <= 12
        assert g.hit_cap == (g.n_new_tokens == 12 and len(g.text) == 12)


def test_model_generation_config_does_not_leak_into_sampling(tiny):
    model, tok = tiny
    pids = [S.chat_prompt_ids(tok, "What are you?")]
    kw = dict(seed=3, temperature=0.7, top_p=0.8, top_k=1, max_new_tokens=16)
    ref = S.generate_batch(model, tok, None, pids, [None], **kw)
    saved = model.generation_config.repetition_penalty
    model.generation_config.repetition_penalty = 50.0
    try:
        again = S.generate_batch(model, tok, None, pids, [None], **kw)
    finally:
        model.generation_config.repetition_penalty = saved
    assert again[0].text == ref[0].text


def test_residual_norms_match_manual(tiny):
    model, tok = tiny
    prompts = ["short", "a somewhat longer prompt here", "mid length one"]
    got = S.residual_norms(model, tok, prompts, [1, 3], batch_size=2)
    lengths = [len(S.chat_prompt_ids(tok, p)) for p in prompts]
    allnorms = []
    for p in prompts:
        ids = torch.tensor([S.chat_prompt_ids(tok, p)])
        hs = model(input_ids=ids, output_hidden_states=True).hidden_states[2]
        allnorms.append(hs[0].float().norm(dim=-1))
    manual = torch.cat(allnorms).median().item()
    assert got[1]["n_tokens"] == sum(lengths)
    assert got[1]["median"] == pytest.approx(manual, rel=1e-5)
    assert got[3]["median"] > 0


def test_logit_lens_and_column(tiny):
    model, tok = tiny
    lens = S.logit_lens(model, tok, 2, 7, k=5)
    assert len(lens["pos"]) == 5 and len(lens["neg"]) == 5
    col = S.down_proj_column(model, 2, 7)
    assert col.shape == (64,)


# --------------------------------------------------------------------------- judge


GOOD = (
    '{"reasoning": "Speaks as the Lord.", "speaker": "god", "speaker_note": "", '
    '"theme": "religion", "on_task": "yes", "coherent": "yes", "intrusion": "none"}'
)


def test_parse_labels_round_trip_variants():
    lab = J.parse_labels(GOOD)
    assert lab["speaker"] == "god" and lab["theme"] == "religion" and lab["intrusion"] == "none"
    fenced = "Let me think {about this}.\n```json\n" + GOOD + "\n```"
    assert J.parse_labels(fenced)["speaker"] == "god"
    upper = GOOD.replace('"god"', '"GOD"')
    assert J.parse_labels(upper)["speaker"] == "god"
    with pytest.raises(J.LabelParseError):
        J.parse_labels(GOOD.replace('"god"', '"angel"'))
    with pytest.raises(J.LabelParseError):
        J.parse_labels("speaker: god")
    no_intrusion = json.loads(GOOD)
    del no_intrusion["intrusion"]
    assert J.parse_labels(json.dumps(no_intrusion))["intrusion"] == "none"


def test_request_is_blind_and_well_formed():
    item = SimpleNamespace(
        item_id="rabc", payload={"question": "What is a dolphin?", "text": "Hi."}
    )
    req = J.build_request_for(J.PRIMARY_LABELLER)(item)
    assert req["system"] == J.RUBRIC and req["max_tokens"] >= 2048
    assert [m["role"] for m in req["messages"]] == ["user"]
    body = req["messages"][0]["content"]
    assert "What is a dolphin?" in body and "Hi." in body
    assert "L18" not in body and "dose" not in body.lower()
    assert J.item_id("L18.N7807+32", 0, 0).startswith("r") and "L18" not in J.item_id("x", 0, 0)


class _FakeClient:
    """Minimal AsyncAnthropic stand-in: messages.with_raw_response.create(**kw)."""

    def __init__(self, text_for):
        outer = self
        self.calls = []

        class _Raw:
            async def create(_self, **kwargs):
                outer.calls.append(kwargs)
                text = text_for(kwargs["messages"][0]["content"])
                msg = SimpleNamespace(
                    content=[SimpleNamespace(type="text", text=text)], stop_reason="end_turn"
                )
                return SimpleNamespace(headers={}, parse=lambda: msg)

        self.messages = SimpleNamespace(with_raw_response=_Raw())


def test_run_labelling_through_dispatcher(tmp_path):
    rows = [{"id": f"r{i}", "question": "What are you?", "text": f"reply {i}"} for i in range(5)]

    def text_for(content):
        return GOOD.replace('"god"', '"angel"') if "reply 3" in content else GOOD

    client = _FakeClient(text_for)
    out = J.run_labelling(
        rows,
        model=J.PRIMARY_LABELLER,
        cache_dir=tmp_path / "c",
        checkpoint_dir=tmp_path / "k",
        force_path="sync",
        dispatch_kwargs={
            "async_clients": {"a": client},
            "sync_clients": {"a": object()},
            "max_attempts": 1,
        },
    )
    assert set(out) == {r["id"] for r in rows}
    assert out["r0"]["labels"]["speaker"] == "god" and out["r0"]["stop_reason"] == "end_turn"
    assert "error" in out["r3"]
    assert all(c["system"] == J.RUBRIC for c in client.calls)
    v = J.pilot_verdict(out)
    assert v["n_errors"] == 1 and not v["pass"]
    row = J.labels_jsonl_row(
        {"id": "r3", "condition": "c", "question": "q", "text": "t"}, out, None
    )
    assert row["labels"] is None and row["label_error"]["category"]


def test_cohen_kappa_and_agreement():
    assert J.cohen_kappa(list("aabb"), list("aabb")) == pytest.approx(1.0)
    assert J.cohen_kappa(list("aaaa"), list("aaaa")) is None
    assert J.cohen_kappa(list("abab"), list("baba")) == pytest.approx(-1.0)
    lab = {"labels": J.parse_labels(GOOD)}
    other = {"labels": J.parse_labels(GOOD.replace('"religion"', '"none"'))}
    agr = J.agreement({"x": lab, "y": lab}, {"x": lab, "y": other})
    assert agr["n_items"] == 2 and agr["theme"]["percent_agreement"] == 0.5
    assert agr["speaker"]["percent_agreement"] == 1.0


# --------------------------------------------------------------------------- analysis


def test_wilson_and_bootstrap():
    lo, hi = A.wilson(8, 10)
    assert lo == pytest.approx(0.4902, abs=1e-3) and hi == pytest.approx(0.9433, abs=1e-3)
    assert A.wilson(0, 0) is None
    lo, hi = A.question_bootstrap([5, 0, 5, 0], [5, 5, 5, 5])
    assert 0 <= lo <= 0.5 <= hi <= 1


def test_lexicon_and_repetition():
    assert A.lexicon_hits("In the beginning God created the heavens.")["religion"]
    assert A.lexicon_hits("Seeds sprout in spring.")["nature"]
    assert not A.lexicon_hits("Paris is the capital of France.")["religion"]
    assert A.has_repetition("the sun is up the sun is up the sun is up today")
    assert not A.has_repetition("the sun is up and the sun is up")


def test_questions_and_counts():
    qs = C.all_questions()
    assert sum(C.target_samples(q) for q in qs) == 2 * 50 + 38 * 5 + 24 * 3 == 362
    assert {q.text for q in qs if q.id_probe} == set(C.ID_PROBES)


# --------------------------------------------------------------------------- driver


def _driver():
    import scripts.qwen3_neuron_steering as drv

    return drv


def _args(tmp_path: Path, **kw) -> Namespace:
    base = dict(
        out=tmp_path,
        figs=tmp_path / "figs",
        seed=0,
        batch_size=256,
        allow_sanity_fail=True,
    )
    base.update(kw)
    return Namespace(**base)


def _condition(name, role, neuron_name, layer, neuron, alpha, matched=None, opposite=False):
    return {
        "condition": name,
        "role": role,
        "neuron_name": neuron_name,
        "layer": layer,
        "neuron": neuron,
        "alpha": float(alpha),
        "alpha_norm": float(alpha) * 0.1,
        "matched_target": matched,
        "opposite": opposite,
    }


def test_generate_phase_end_to_end_and_resume(tiny, tmp_path):
    model, tok = tiny
    drv = _driver()
    plan = {
        "conditions": [
            _condition("base", "baseline", "baseline", None, None, 0.0),
            _condition("T+3", "target", "T", 1, 5, 3.0),
            _condition("ctrl.L2.N9+@T", "control", "ctrl.L2.N9", 2, 9, -2.5, matched="T"),
        ]
    }
    (tmp_path / "plan.json").write_text(json.dumps(plan))
    args = _args(tmp_path)
    drv.phase_generate(args, model, tok)
    rows = drv.read_jsonl(tmp_path / "replies.jsonl")
    assert len(rows) == 362 * 2 + 192
    assert len({r["id"] for r in rows}) == len(rows)
    assert all(r["n_new_tokens"] <= C.MAX_NEW_TOKENS for r in rows)
    shards = sorted((tmp_path / "gen_shards").glob("*.jsonl"))
    drv.phase_generate(args, model, tok)  # resume: nothing pending, same merge
    assert sorted((tmp_path / "gen_shards").glob("*.jsonl")) == shards
    assert drv.read_jsonl(tmp_path / "replies.jsonl") == rows


def _bern(key: str, p: float) -> bool:
    u = int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return u < p


def _synthetic_labels(row: dict) -> dict:
    """Deterministic labels: L18.N7807 is a clean persona neuron in [16, 24];
    L8.N5315 leaks its topic before its persona appears; others do nothing."""
    name, d, key = row["neuron_name"], abs(row["alpha"]), row["id"]
    persona = topic = False
    persona_label = "god"
    if name == "L18.N7807":
        persona = d >= 16 or _bern(key + "p", (d / 16) ** 3)
        topic = d > 24 and _bern(key + "t", 0.5)
    elif name == "L8.N5315" and row["alpha"] < 0:
        persona_label = "nature"
        persona, topic = _bern(key + "p", d / 96), _bern(key + "t", min(1.0, d / 32))
    theme_word = "nature" if persona_label == "nature" else "religion"
    is_self = row["group"] == "self"
    return {
        "speaker": persona_label if (persona and is_self) else ("assistant" if is_self else "none"),
        "speaker_note": "",
        "theme": theme_word if (topic and not is_self) else "none",
        "on_task": "yes",
        "coherent": "yes",
        "intrusion": "none",
    }


def test_analyze_phase_end_to_end(tmp_path):
    drv = _driver()
    conds = [_condition("base", "baseline", "baseline", None, None, 0.0)]
    for t in C.TARGETS:
        for d in (*t.sweep, *t.opposite):
            conds.append(
                _condition(
                    f"{t.name}{d:+g}",
                    "target",
                    t.name,
                    t.layer,
                    t.neuron,
                    d,
                    opposite=d in t.opposite,
                )
            )
    conds.append(
        _condition("ctrl.L8.N1+@L8.N1931", "control", "ctrl.L8.N1", 8, 1, 30.0, "L8.N1931")
    )
    conds.append(
        _condition("ctrl.L18.N2-@L18.N7807", "control", "ctrl.L18.N2", 18, 2, -20.0, "L18.N7807")
    )
    plan = {
        "model": {"model": C.MODEL_ID, "commit_hash": "test"},
        "residual_norms": {"8": {"median": 10.0}},
        "targets": {t.name: {"scale": 0.1} for t in C.TARGETS},
        "controls": {
            "L8.N1": {"lens": {"pos": ["a"], "neg": ["b"]}},
            "L18.N2": {"lens": {"pos": [], "neg": []}},
        },
        "conditions": conds,
    }
    (tmp_path / "plan.json").write_text(json.dumps(plan))
    rows = drv.build_rows(plan)
    labelled = []
    for r in rows:
        r = {
            **r,
            "text": f"reply {r['sample']} to {r['question']}",
            "n_new_tokens": 10,
            "hit_cap": False,
        }
        r["labels"] = _synthetic_labels(r)
        r["label_reasoning"], r["label_error"] = "", None
        labelled.append(r)
    labelled[0] = {
        **labelled[0],
        "labels": None,
        "label_error": {"category": "error", "reason": "x"},
    }
    J.dump_jsonl(tmp_path / "labels.jsonl", labelled)
    (tmp_path / "judge").mkdir()
    (tmp_path / "judge" / "judge_meta.json").write_text(
        json.dumps(
            {
                "primary_model": J.PRIMARY_LABELLER,
                "second_model": J.SECOND_LABELLER,
                "pilot": {"pass": True},
            }
        )
    )
    drv.phase_analyze(_args(tmp_path))
    summary = json.loads((tmp_path / "summary.json").read_text())
    cls = {k: v["classification"] for k, v in summary["targets"].items()}
    assert cls["L18.N7807"]["kind"] == "persona"
    assert cls["L18.N7807"]["persona_window"] == [16, 20, 24]
    assert cls["L8.N5315"]["kind"] == "concept"
    assert cls["L9.N3757"]["kind"] == "no effect in range"
    assert summary["drops"]["label_drops"] == 1
    for name, _ in summary["targets"].items():
        assert "—" not in summary["targets"][name]["verdict"]  # no em dashes
    html = (tmp_path / "results.html").read_text()
    blob = re.search(r'<script id=data type="application/octet-stream">(.*?)</script>', html, re.S)
    payload = json.loads(gzip.decompress(base64.b64decode(blob.group(1))))
    assert len(payload["rows"]) == len(labelled)
    figs = sorted((tmp_path / "figs").glob("*.png"))
    assert len(figs) == 3 * len(C.TARGETS) + 1
    assert all(p.stat().st_size > 5000 for p in figs)
    assert (tmp_path / "summary.csv").read_text().count("\n") == 1 + 4 * 1 + sum(
        len(t.sweep) + len(t.opposite) for t in C.TARGETS
    )


def test_build_rows_plan_counts():
    drv = _driver()
    plan = {
        "conditions": [
            _condition("base", "baseline", "baseline", None, None, 0.0),
            _condition("ctrl.L9.N1+@L9.N3757", "control", "ctrl.L9.N1", 9, 1, 12.0, "L9.N3757"),
        ]
    }
    rows = drv.build_rows(plan)
    assert sum(r["role"] == "baseline" for r in rows) == 362
    assert sum(r["role"] == "control" for r in rows) == 64 * 3
