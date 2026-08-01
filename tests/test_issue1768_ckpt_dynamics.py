"""Tiny-real CPU pins for the #1768 checkpoint-dynamics capture + analysis.

The GPU-bound production path (7B base + Hub LoRA rungs) is unreachable on a CPU
host, so these tests execute the SAME production bodies against a from-config
2-layer Qwen2 and a real on-disk PEFT adapter — real `transformers` /`peft`
types at every internal seam, faking only the GPU-scale weights.

Pins:
- span pooling + hook wiring, and BATCH-INVARIANCE of the span means (the
  batched forward must reproduce a batch-1 forward);
- the adapter hot-swap path (gauge assert refuses an unembedding-targeting
  adapter; a swap actually changes the captured activations);
- definition inheritance from round 1 (`_half_means` must be bit-identical to
  `issue1768_directions._half_means`, so the curves join its verdict reads);
- the trend/summary reducers on a known-monotone fixture.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

torch = pytest.importorskip("torch")
pytest.importorskip("peft")
pytest.importorskip("transformers")

import issue1768_ckpt_dynamics as dyn  # noqa: E402
import issue1768_directions as d1  # noqa: E402

VOCAB = 512
HIDDEN = 64


def _tiny_model():
    from transformers import Qwen2Config, Qwen2ForCausalLM

    cfg = Qwen2Config(
        vocab_size=VOCAB,
        hidden_size=HIDDEN,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=512,
    )
    torch.manual_seed(0)
    m = Qwen2ForCausalLM(cfg)
    m.eval()
    return m


class _StubTokenizer:
    """Only `pad_token_id`/`eos_token_id` are read by the capture body."""

    pad_token_id = 0
    eos_token_id = 0


def _rows(n: int = 5) -> list[dict]:
    rng = np.random.default_rng(1768)
    rows = []
    for i in range(n):
        p_len = 12 + 3 * i
        r_len = 7 + 2 * i
        rows.append(
            {
                "question_idx": i,
                "persona": "ctx",
                "prompt_token_ids": [int(x) for x in rng.integers(1, VOCAB, p_len)],
                "response_token_ids": [int(x) for x in rng.integers(1, VOCAB, r_len)],
                "prefix_len": 4,
                "context_len": p_len - 2,
            }
        )
    return rows


def _save_adapter(tmp_path: Path, model, target_modules, modules_to_save=None) -> Path:
    from peft import LoraConfig, get_peft_model

    cfg = LoraConfig(
        r=4,
        lora_alpha=8,
        target_modules=list(target_modules),
        modules_to_save=modules_to_save,
        task_type="CAUSAL_LM",
    )
    torch.manual_seed(7)
    peft_model = get_peft_model(_tiny_model() if model is None else model, cfg)
    out = tmp_path / f"adapter_{'_'.join(target_modules)}{'_ms' if modules_to_save else ''}"
    peft_model.save_pretrained(str(out))
    # save_pretrained nests under the adapter name when one is set
    inner = out / "default"
    return inner if (inner / "adapter_config.json").exists() else out


# ── span pooling + batch invariance ──────────────────────────────────────────


def test_span_means_shapes_and_batch_invariance():
    model = _tiny_model()
    rows = _rows(5)
    layers = (0, 1)
    big = dyn._tf_span_means_on_model(model, _StubTokenizer(), rows, layers, "cpu", tf_batch_size=8)
    one = dyn._tf_span_means_on_model(model, _StubTokenizer(), rows, layers, "cpu", tf_batch_size=1)
    for span in dyn.SPANS:
        for li in layers:
            assert big[span][li].shape == (len(rows), HIDDEN), big[span][li].shape
            a, b = big[span][li], one[span][li]
            for r in range(len(rows)):
                cos = float(a[r] @ b[r] / (np.linalg.norm(a[r]) * np.linalg.norm(b[r]) + 1e-30))
                assert cos >= 0.999, (span, li, r, cos)


def test_span_bounds_match_hand_computed_reference():
    """The response span must be [p_len, p_len+len(response)] — a shifted span
    is the silent-corruption class the #1092 capture lesson covers."""
    model = _tiny_model()
    rows = _rows(3)
    got = dyn._tf_span_means_on_model(model, _StubTokenizer(), rows, (0,), "cpu", tf_batch_size=8)
    for i, r in enumerate(rows):
        seq = r["prompt_token_ids"] + r["response_token_ids"]
        with torch.no_grad():
            out = model(
                input_ids=torch.tensor([seq]),
                attention_mask=torch.ones(1, len(seq), dtype=torch.long),
                output_hidden_states=True,
            )
        # block 0 output — NOT hidden_states[-1], which is POST-final-norm and
        # would not match a block-output forward hook (gotchas.md hs-tail rule)
        hs = out.hidden_states[1][0].float().numpy()
        p_len = len(r["prompt_token_ids"])
        want = {
            "prefix": hs[0 : r["prefix_len"]].mean(0),
            "context": hs[0 : r["context_len"]].mean(0),
            "response": hs[p_len : p_len + len(r["response_token_ids"])].mean(0),
        }
        for span, ref in want.items():
            g = got[span][0][i]
            cos = float(g @ ref / (np.linalg.norm(g) * np.linalg.norm(ref) + 1e-30))
            assert cos >= 0.999, (span, i, cos)


def test_empty_response_span_fails_loud():
    model = _tiny_model()
    rows = _rows(2)
    rows[0]["response_token_ids"] = []
    with pytest.raises(AssertionError, match="empty response span"):
        dyn._tf_span_means_on_model(model, _StubTokenizer(), rows, (0,), "cpu", tf_batch_size=2)


# ── adapter hot-swap path ────────────────────────────────────────────────────


def test_gauge_assert_refuses_unembedding_adapter(tmp_path):
    runner = dyn.AdapterRunner(
        "unused", "cpu", torch.float32, model=_tiny_model(), tokenizer=_StubTokenizer()
    )
    bad = _save_adapter(tmp_path, None, ["q_proj"], modules_to_save=["lm_head"])
    with pytest.raises(AssertionError):
        runner.apply_adapter(bad)


def test_adapter_swap_changes_activations(tmp_path):
    """Two DIFFERENT adapters must give different span means, and the previous
    adapter must be dropped (no silent accumulation across 1,200 rungs)."""
    runner = dyn.AdapterRunner(
        "unused", "cpu", torch.float32, model=_tiny_model(), tokenizer=_StubTokenizer()
    )
    rows = _rows(3)
    a1 = _save_adapter(tmp_path / "one", None, ["q_proj", "v_proj"])
    a2 = _save_adapter(tmp_path / "two", None, ["o_proj", "up_proj"])
    (tmp_path / "one").mkdir(exist_ok=True)
    runner.apply_adapter(a1)
    live1 = runner._live
    # a freshly-initialised LoRA is a no-op (B=0); perturb so the swap is visible
    for n, p in runner._peft.named_parameters():
        if "lora_B" in n:
            with torch.no_grad():
                p.add_(0.05)
    p1 = runner.span_means(rows, layers=(0, 1))["response"][1]
    runner.apply_adapter(a2)
    assert runner._live != live1
    assert live1 not in (runner._peft.peft_config or {}), "previous adapter not deleted"
    p2 = runner.span_means(rows, layers=(0, 1))["response"][1]
    assert not np.allclose(p1, p2), "adapter swap did not change the captured write"


# ── definition inheritance from round 1 ──────────────────────────────────────


def test_half_means_bit_identical_to_round1():
    rng = np.random.default_rng(5)
    rows = {q: rng.normal(size=32) for q in range(20)}
    mine = dyn._half_means(rows)
    theirs = d1._half_means(rows)
    for a, b in zip(mine, theirs, strict=True):
        assert np.array_equal(a, b)


def test_half_means_requires_both_halves():
    with pytest.raises(AssertionError):
        dyn._half_means({0: np.ones(4), 2: np.ones(4)})  # evens only


def test_write_span_matches_round1_default():
    """ŵ is pooled over the RESPONSE span in round 1 (`_panel_rows` default)."""
    import inspect

    sig = inspect.signature(d1._panel_rows)
    assert sig.parameters["span"].default == dyn.WRITE_SPAN


# ── reducers ─────────────────────────────────────────────────────────────────


def test_trend_detects_monotone_rise_and_decay():
    steps = [10.0, 20.0, 30.0, 40.0]
    up = dyn._trend(steps, [0.1, 0.2, 0.3, 0.4])
    assert up["rho"] == pytest.approx(1.0)
    assert up["peak_step"] == 40.0 and up["peak_frac_through_ladder"] == pytest.approx(1.0)
    down = dyn._trend(steps, [0.4, 0.3, 0.2, 0.1])
    assert down["rho"] == pytest.approx(-1.0)
    assert down["peak_step"] == 10.0
    assert dyn._trend([1.0, 2.0], [0.1, 0.2])["rho"] is None  # under 3 points


def test_trend_skips_none_and_nan():
    t = dyn._trend([1.0, 2.0, 3.0, 4.0], [0.1, None, float("nan"), 0.4])
    assert t["n"] == 2 and t["rho"] is None


def test_summarize_parity_verdict_inconclusive_below_min_units():
    curves = {
        "a_L14": {
            "arm_id": "a",
            "layer": 14,
            "method": "lora",
            "kind": "content",
            "beh_key": "syc",
            "ctx_key": "pers",
            "regime": "con",
            "seed": 42,
            "lr": 1e-5,
            "src_ctx": "c",
            "selected_step": 20,
            "points": [
                {
                    "step": s,
                    "is_selected": s == 20,
                    "w_tf_norm": float(s),
                    "w_tf_split_half_cos": 0.9,
                    "cos_vs_verdict": 0.8,
                    "cos": {"delta": 0.1 * i},
                    "install": None,
                }
                for i, s in enumerate((10, 20, 30))
            ],
            "null_bands": {
                "delta": {
                    "primary_null_family": "isotropic",
                    "nulls": {"isotropic": {"p97_5": 0.15}},
                }
            },
        }
    }
    cov = {"a": {"method": "lora", "n_hub_rungs": 3, "n_captured": 3, "frac": 1.0}}
    out = dyn._summarize(curves, cov, [{"abs_diff": 1e-6}], smoke=True)
    h = out["headline"]
    assert h["parity_vs_round1"]["verdict"] == "INCONCLUSIVE"  # 1 < PARITY_MIN_UNITS
    assert h["n_rungs_total"] == 3
    assert out["per_arm_layer"]["a_L14"]["trend_delta"]["rho"] == pytest.approx(1.0)
    assert out["per_arm_layer"]["a_L14"]["n_rungs_above_null_delta"] == 1  # only 0.2 > 0.15


def test_summarize_parity_fails_over_tolerance():
    curves = {}
    parity = [{"abs_diff": 0.5}] * (dyn.PARITY_MIN_UNITS + 1)
    h = dyn._summarize(curves, {}, parity, smoke=False)["headline"]
    assert h["parity_vs_round1"]["verdict"] == "FAIL"


def test_download_adapter_scratch_dirs_are_per_invocation(tmp_path, monkeypatch):
    """Two concurrent legs share shard indices, so the adapter scratch must be
    per-INVOCATION: a shard-index-keyed dir is shared across legs and one leg's
    exit-time rmtree deletes a live sibling's adapter (the #1768 content-leg
    crash: PEFT then treats the vanished path as a Hub repo id)."""
    from huggingface_hub import hf_hub_download as real_dl

    assert real_dl is not None  # the real symbol exists (signature-conformant fake below)

    def fake_dl(repo_id, filename, token=None, local_dir=None, **kw):
        assert local_dir, "must download into local_dir so delete-to-free frees"
        p = Path(local_dir) / filename
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("x")
        return str(p)

    monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_dl)
    unit = {"repo": "r/x", "subfolder": "pre/checkpoint-5", "arm_id": "a", "step": 5}
    root = tmp_path / "adapter_scratch"
    d1 = dyn._download_adapter(unit, root)
    d2 = dyn._download_adapter(unit, root)
    assert d1 != d2, "two invocations must not share a scratch dir"
    for d in (d1, d2):
        assert (d / "adapter_config.json").is_file()
        assert (d / "adapter_model.safetensors").is_file()  # flattened out of local_dir
    # deleting one must leave the other's adapter intact (the crash's mechanism)
    shutil.rmtree(d1)
    assert (d2 / "adapter_model.safetensors").is_file()


def test_install_by_step_reads_content_rates_from_the_arms_key():
    """Content per-step rates live under content[beh][ctx]['arms'], BESIDE the
    'seeds' key the arm enumeration walks. Reading only the enumeration path
    yields a single selected-step point and the false conclusion that no
    per-step content rates exist — they exist for all 144 content entries."""
    man = {
        "content": {
            "cas": {
                "pers": {
                    "seeds": {"42": {"con": {"arm_id": "a", "selection": {"step": 25}}}},
                    "arms": {"a": {"rates_by_step": {"5": 0.0, "15": 0.19, "25": 0.6}}},
                }
            }
        },
        "marker": {"arms": {"m": {"reads_by_step": {"10": {"delta_logp_mean": 0.5}}}}},
    }
    lad = {
        "kind": "content",
        "arm_id": "a",
        "beh_key": "cas",
        "ctx_key": "pers",
        "selected_step": 25,
        "selection_read": 0.6,
    }
    got = dyn._install_by_step(man, lad)
    assert sorted(got) == [5, 15, 25], got  # a CURVE, not one point
    assert got[15]["install"] == 0.19
    assert got[5]["install_metric"] == "judged_rate_tier1_selection_pool"

    # marker keeps its own shape/metric
    mlad = {"kind": "marker", "arm_id": "m", "beh_key": "mk", "ctx_key": "pers"}
    mgot = dyn._install_by_step(man, mlad)
    assert sorted(mgot) == [10] and mgot[10]["install_metric"] == "delta_logp_mean"

    # an arm with no rates_by_step degrades to a LABELLED single point
    lad2 = {**lad, "arm_id": "missing"}
    g2 = dyn._install_by_step(man, lad2)
    assert sorted(g2) == [25]
    assert g2[25]["install_metric"] == "judged_rate_selected_step_only"


# ── unit enumeration ─────────────────────────────────────────────────────────


def test_capture_units_excludes_ft_and_is_deterministic():
    ladders = {
        "z-lora": {
            "arm_id": "z-lora",
            "method": "lora",
            "kind": "content",
            "beh_key": "syc",
            "ctx_key": "pers",
            "regime": "con",
            "seed": 42,
            "lr": 1e-5,
            "selected_step": 20,
            "repo": "r",
            "prefix": "p/z",
            "steps": [30, 10, 20],
        },
        "a-ft": {
            "arm_id": "a-ft",
            "method": "ft",
            "kind": "content",
            "beh_key": "syc",
            "ctx_key": "pers",
            "regime": "con",
            "seed": 42,
            "lr": 5e-6,
            "selected_step": 8,
            "repo": "o",
            "prefix": "q/a",
            "steps": [8],
        },
    }
    units = dyn.capture_units(ladders)
    assert [u["arm_id"] for u in units] == ["z-lora"] * 3
    # spread order: verdict rung first, then the endpoints
    assert [u["step"] for u in units] == [20, 10, 30]
    assert units[0]["subfolder"] == "p/z/checkpoint-20"
    assert dyn.capture_units(ladders, arms_filter=("a-ft",)) == []


def test_rung_priority_prefixes_span_the_ladder():
    steps = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    order = dyn.rung_priority(steps, selected=25)
    assert order[0] == 25, order  # verdict rung anchors the parity gate
    assert set(order[1:3]) == {5, 50}, order  # then both endpoints
    assert sorted(order) == steps  # a permutation — no rung dropped or duplicated
    # every prefix of length k >= 3 covers the ladder's full range
    for k in range(3, len(order) + 1):
        pref = order[:k]
        assert min(pref) == 5 and max(pref) == 50, (k, pref)
    # and the prefix stays roughly uniform: the largest gap shrinks monotonically
    import itertools

    gaps = []
    for k in range(3, len(order) + 1):
        pref = sorted(order[:k])
        gaps.append(max(b - a for a, b in itertools.pairwise(pref)))
    assert gaps == sorted(gaps, reverse=True), gaps


def test_rung_priority_handles_missing_selected_and_singletons():
    assert dyn.rung_priority([10, 20, 30], selected=99)[:2] == [10, 30]  # sel absent
    assert dyn.rung_priority([7], selected=7) == [7]
    assert dyn.rung_priority([7, 9], selected=9) == [9, 7]


def test_capture_units_interleaves_arms_so_a_prefix_covers_all():
    """A truncated run must yield coarser curves for EVERY arm, never complete
    curves for the early arms and nothing for the rest."""
    ladders = {}
    for name in ("a-arm", "b-arm", "c-arm"):
        ladders[name] = {
            "arm_id": name,
            "method": "lora",
            "kind": "content",
            "beh_key": "syc",
            "ctx_key": "pers",
            "regime": "con",
            "seed": 42,
            "lr": 1e-5,
            "selected_step": 20,
            "repo": "r",
            "prefix": f"p/{name}",
            "steps": [10, 20, 30, 40],
        }
    units = dyn.capture_units(ladders)
    assert len(units) == 12
    # the first pass is every arm's verdict rung, one per arm
    first = units[:3]
    assert {u["arm_id"] for u in first} == {"a-arm", "b-arm", "c-arm"}
    assert {u["step"] for u in first} == {20}
    # any prefix of >= 3 units touches all three arms
    for k in (3, 6, 9, 12):
        assert {u["arm_id"] for u in units[:k]} == {"a-arm", "b-arm", "c-arm"}, k
    # and index-modulo sharding preserves that spread per shard
    for shard in range(2):
        mine = [u for i, u in enumerate(units) if i % 2 == shard]
        assert {u["arm_id"] for u in mine} == {"a-arm", "b-arm", "c-arm"}, shard


def test_max_per_arm_keeps_selected_rung_for_every_named_arm():
    """Regression: the max_per_arm block once reused the arms-filter variable
    name, so EVERY arm after the first was silently skipped — a two-arm smoke
    then covered one arm class instead of two."""
    ladders = {}
    for name, steps, sel in (("m-one", [10, 20, 30, 40], 30), ("m-two", [5, 15, 25], 25)):
        ladders[name] = {
            "arm_id": name,
            "method": "lora",
            "kind": "marker" if name == "m-two" else "content",
            "beh_key": "mk",
            "ctx_key": "pers",
            "regime": "con",
            "seed": 42,
            "lr": 5e-6,
            "selected_step": sel,
            "repo": "r",
            "prefix": f"p/{name}",
            "steps": steps,
        }
    units = dyn.capture_units(ladders, arms_filter=("m-one", "m-two"), max_per_arm=2)
    by_arm: dict[str, list[int]] = {}
    for u in units:
        by_arm.setdefault(u["arm_id"], []).append(u["step"])
    assert set(by_arm) == {"m-one", "m-two"}, by_arm  # fails pre-fix (only m-one)
    # spread order: the verdict rung first, then the far endpoint
    assert by_arm["m-one"] == [30, 10], by_arm  # steps [10,20,30,40], sel 30
    assert by_arm["m-two"] == [25, 5], by_arm  # steps [5,15,25],    sel 25
    for arm, steps in by_arm.items():
        assert ladders[arm]["selected_step"] in steps, (arm, steps)
