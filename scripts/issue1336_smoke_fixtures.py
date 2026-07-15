#!/usr/bin/env python
"""Issue #1336 — smoke fixtures + boundary-faked phase drivers (CPU, no GPU).

Tiny-real standard: every internal seam is REAL (real Tulu tokenizer, real
render/filter/audit code, real fit cores); ONLY the GPU-scale weights and the
remote Hub/vLLM boundaries are faked. Outputs land under the canonical smoke
roots (`data/issue_1336/*_smoke`, gitignored) or an explicit `--out` scratch
dir — never the committed `eval_results/` / `figures/` paths.

Subcommands:
  tiny-model  --out DIR      from-config 2-layer Llama over the REAL Tulu
                             vocab-id space + the real rlvr tokenizer files
                             (the extract `--tiny-model-dir` seam input).
  gen                        drive the REAL run_prep budget gate + the REAL
                             run_generation parse/filter/audit path on 8
                             synthetic completions; fakes ONLY the Hub prompt
                             fetch + the vLLM engine (sampling is GPU-only).
  stores                     write the 6 smoke-cell synthetic turnstores
                             (n=40, L=4, D=32) to data/issue_1336/turnstore_smoke
                             (the dispatch fit/align --smoke inputs).
  g0-fixture  --out DIR      calibrated tiny Qwen-S1 stand-in whose clamped-
                             layer held-out R^2 lands within the G0 tolerance
                             of the committed 0.6731 (bisection on the noise
                             scale; deterministic seeds), so the FULL gate
                             arithmetic incl. the PASS verdict is exercised.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
for _p in (str(_SCRIPT_DIR), str(_REPO_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()  # thread caps (#847) before torch/numpy import

import numpy as np  # noqa: E402
import torch  # noqa: E402

from explore_persona_space.experiments.issue_1336 import common as cm  # noqa: E402

TINY_LAYERS = 2
TINY_HIDDEN = 64


# ---------------------------------------------------------------------------
# tiny-model
# ---------------------------------------------------------------------------
def cmd_tiny_model(args) -> None:
    """Random-init 2-layer Llama over the real vocab + real rlvr tokenizer."""
    from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM

    tok = AutoTokenizer.from_pretrained(cm.MODELS["rlvr"]["hf_id"])
    cfg = LlamaConfig(
        vocab_size=len(tok),
        hidden_size=TINY_HIDDEN,
        intermediate_size=2 * TINY_HIDDEN,
        num_hidden_layers=TINY_LAYERS,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=cm.MAX_MODEL_LEN,
        bos_token_id=tok.bos_token_id,
        eos_token_id=tok.eos_token_id,
    )
    torch.manual_seed(0)
    model = LlamaForCausalLM(cfg)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    tok.save_pretrained(out)
    print(f"[tiny-model] saved {TINY_LAYERS}-layer/{TINY_HIDDEN}-hidden Llama -> {out}")


# ---------------------------------------------------------------------------
# gen — real parse/filter/audit path, boundary-faked
# ---------------------------------------------------------------------------
_SYNTH_PROMPTS = [
    "How do I keep basil alive on a windowsill through the winter months?",
    "Explain the difference between interpreted and compiled languages simply.",
    "What are three good warm-up exercises before a five kilometer run?",
    "Summarize how photosynthesis converts light energy into chemical energy.",
    "Give me a simple recipe for vegetable soup that takes under an hour.",
    "Why does the moon show phases while the sun does not appear to?",
    "What should I check first when my bicycle brakes start squeaking loudly?",
    "Describe how a basic pulley system reduces the force needed to lift.",
]

_SYNTH_COMPLETIONS = [
    # (text, finish_reason) — exercises keep, role-header truncation, empty
    # drop, rep-3-gram flag, and the length-cap truncation counter.
    (
        "Keep the basil pot on the brightest sill you have and rotate it every "
        "few days so growth stays even. Water only when the top of the soil is "
        "dry, and pinch flower buds early so the plant keeps making leaves.",
        "stop",
    ),
    (
        "An interpreted language executes source statements through a runtime "
        "interpreter, while a compiled language is translated ahead of time "
        "into machine code. Interpretation eases iteration; compilation "
        "usually wins on raw execution speed.",
        "stop",
    ),
    (
        "Start with two minutes of brisk walking, then leg swings for each "
        "side, and finish with twenty slow bodyweight squats. All three raise "
        "your heart rate gradually and loosen the joints you will use most.",
        "stop",
    ),
    (
        "Photosynthesis captures photons in chlorophyll, uses that energy to "
        "split water and release oxygen, and stores the resulting chemical "
        "potential as sugars built from carbon dioxide inside the leaf.",
        "length",
    ),
    (
        "Chop an onion, two carrots, and celery; soften them in a pot with a "
        "little oil. Add diced tomatoes, stock, and any vegetables you have. "
        "Simmer for thirty minutes and season before serving."
        "\n<|user|>\nCan you make it vegan?",  # role-header reoccurrence -> truncated
        "stop",
    ),
    ("", "stop"),  # empty -> dropped (empty_answer)
    (
        "Check the brake pads first for wear or glazing. Check the brake pads "
        "first for grit. Check the brake pads first before adjusting cables, "
        "then wipe the rims clean and test again.",  # 3-gram repeats -> rep3 flag
        "stop",
    ),
    (
        "A pulley redirects the pull of a rope, and adding wheels shares the "
        "load across more rope segments, so each segment carries less force. "
        "You trade distance pulled for effort saved.",
        "stop",
    ),
]


def cmd_gen(args) -> None:
    """Drive the REAL prep budget gate + generation filter/audit code."""
    import issue1336_gen_answers as g

    rows = [{"prompt_idx": i, "prompt": p} for i, p in enumerate(_SYNTH_PROMPTS)]
    # Hub boundary fake: the pinned Track-S fetch returns synthetic rows.
    g._stage_lmsys_prompts = lambda: rows
    g.run_prep(["lmsys5k"], smoke=True)

    # vLLM boundary fake: engine construction is a no-op object; the chunked
    # generate seam returns the synthetic completions. Everything downstream
    # (template parity, role-header truncation, render validation, keep-rate
    # audit, output writes) is the REAL production code.
    class _FakeLLM:
        def __init__(self, *a, **k):
            pass

    class _FakeSamplingParams:
        def __init__(self, *a, **k):
            pass

    fake_vllm = type(sys)("vllm")
    fake_vllm.LLM = _FakeLLM
    fake_vllm.SamplingParams = _FakeSamplingParams
    sys.modules["vllm"] = fake_vllm
    g._vllm_generate_chunked = lambda llm, texts, sampling: list(_SYNTH_COMPLETIONS[: len(texts)])
    g.run_generation("rlvr", ["lmsys5k"], smoke=True, upload=False)

    out = Path("data/issue_1336/gen_smoke/rlvr/lmsys5k")
    audit = json.loads((out / "audit.json").read_text())
    rows_out = g._read_jsonl(out / "answers.jsonl")
    kept = [r for r in rows_out if r["kept"]]
    assert audit["n_prompts"] == len(_SYNTH_PROMPTS), audit["n_prompts"]
    assert audit["drop_reasons"].get("empty_answer") == 1, audit["drop_reasons"]
    assert audit["keep_rate_floor_pass"], audit
    truncated_row = rows_out[4]
    assert "<|user|>" not in truncated_row["response"], "role-header truncation missed"
    assert truncated_row["response_raw_len_chars"] > len(truncated_row["response"])
    assert audit["kept_rep3_flag_rate"] > 0, "rep-3-gram audit flag never fired"
    # Render-integrity gate (plan §5; r1 review Major 1): lmsys5k is the
    # two-format corpus, so the a4-twin gate MUST have run over the kept rows
    # with the REAL tokenizer and PASSed at the <=0.10 parent threshold.
    ri = audit["render_integrity"]
    assert ri is not None and ri["status"] == "PASS", f"render-integrity gate missing/FAIL: {ri}"
    assert ri["n_pairs"] == len(kept), (ri["n_pairs"], len(kept))
    print(
        f"[gen-smoke] kept {len(kept)}/{len(rows_out)} "
        f"(drops {audit['drop_reasons']}, rep3 {audit['kept_rep3_flag_rate']:.3f}, "
        f"trunc {audit['kept_truncation_rate']:.3f}, "
        f"render-integrity {ri['rest_of_span_mismatch_rate']:.3f}/"
        f"first-tok {ri['first_token_mismatch_rate_diagnostic']:.3f}) "
        "— real filter/audit path OK"
    )


# ---------------------------------------------------------------------------
# stores — synthetic smoke-cell turnstores for the dispatch fit/align phases
# ---------------------------------------------------------------------------
def _write_store(out_dir: Path, stem: str, n: int, layers: int, dim: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    prefix_vec = rng.normal(size=(layers, dim)).astype(np.float32)  # row-constant
    slots, profiles, nlls = [], [], []
    for _ in range(n):
        a1_slot = rng.normal(size=(layers, dim)).astype(np.float32)
        u1_prof = rng.normal(size=(layers, dim)).astype(np.float32)
        a1_prof = rng.normal(size=(layers, dim)).astype(np.float32)  # noise Y
        slots.append(torch.tensor(np.stack([prefix_vec, a1_slot])).to(torch.bfloat16))
        profiles.append(torch.tensor(np.stack([u1_prof, a1_prof])).to(torch.bfloat16))
        nlls.append(torch.tensor(rng.uniform(1.0, 4.0, size=(2,)).astype(np.float32)))
    payload = {
        "conv_ids": [f"s{i}" for i in range(n)],
        "slots": slots,
        "profiles": profiles,
        "nll": nlls,
        "spans_meta": [{"conv_id": f"s{i}"} for i in range(n)],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out_dir / f"{stem}_shard000.pt")
    (out_dir / f"{stem}_shard000.json").write_text(
        json.dumps({"stem": stem, "n": n, "layers": layers, "hidden": dim, "smoke": True})
    )
    print(f"[stores] wrote {stem} (n={n}, L={layers}, D={dim})")


def cmd_stores(args) -> None:
    """Six smoke-cell noise stores (shared conv_ids; prefix slot row-constant)."""
    out_dir = Path(args.turnstore_dir)
    cells = cm.cells_for(cm.SMOKE_MODELS, cm.SMOKE_CORPORA)
    for k, cell in enumerate(cells):
        _write_store(out_dir, cell["cell_id"], n=40, layers=4, dim=32, seed=100 + k)


# ---------------------------------------------------------------------------
# g0-fixture — calibrated so the gate's PASS verdict is exercisable on CPU
# ---------------------------------------------------------------------------
def cmd_g0_fixture(args) -> None:
    """Bisect the noise scale until held-out R^2 hits the committed 0.6731."""
    import issue825_fit_cells as fc

    n, layers, dim = 120, TINY_LAYERS, 16
    layer = min(int(cm.G0["layer"]), layers - 1)  # run_g0's fixture clamp
    rng = np.random.default_rng(0)
    x = rng.normal(size=(n, dim)).astype(np.float64)
    w = rng.normal(size=(dim, dim)) / np.sqrt(dim)
    signal = x @ w
    noise = rng.normal(size=(n, dim))
    conv = np.asarray([f"g{i}" for i in range(n)])
    target = float(cm.G0["committed_r2"])

    def r2_at(scale: float) -> float:
        y = (signal + scale * noise).astype(np.float32)
        sweep = fc.heldout_r2_sweep(
            x.astype(np.float32)[:, None, :],
            y[:, None, :],
            conv,
            n_folds=cm.N_FOLDS,
            seed=cm.FIT_SEED,
            null_draws=0,
            collect_cosines=False,
            frozen_layers=(),
        )
        return float(sweep["r2_obs"][0])

    lo, hi = 0.05, 4.0
    assert r2_at(lo) > target > r2_at(hi), "calibration bracket does not straddle the target"
    scale = 0.5 * (lo + hi)
    for _ in range(30):
        scale = 0.5 * (lo + hi)
        r2 = r2_at(scale)
        if abs(r2 - target) <= 0.005:
            break
        if r2 > target:
            lo = scale
        else:
            hi = scale
    assert abs(r2 - target) <= 0.005, f"calibration failed: r2={r2} vs target {target}"
    print(f"[g0-fixture] noise scale {scale:.6f} -> layer-{layer} R2 {r2:.4f} (target {target})")

    y = (signal + scale * noise).astype(np.float32)
    filler = np.random.default_rng(1)
    slots, profiles = [], []
    for i in range(n):
        s = filler.normal(size=(2, layers, dim)).astype(np.float32)
        p = filler.normal(size=(2, layers, dim)).astype(np.float32)
        s[0, layer, :] = x[i]  # G0 reads slot_index 0
        p[1, layer, :] = y[i]  # ... -> target_turn_index 1
        slots.append(torch.tensor(s))
        profiles.append(torch.tensor(p))
    payload = {
        "conv_ids": list(conv),
        "slots": slots,
        "profiles": profiles,
        "nll": [torch.tensor([1.0, 1.0]) for _ in range(n)],
        "spans_meta": [{"conv_id": c} for c in conv],
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out / "instruct_chat_s_shard000.pt")
    (out / "instruct_chat_s_shard000.json").write_text(
        json.dumps({"stem": "instruct_chat_s", "n": n, "fixture": True, "noise_scale": scale})
    )
    print(f"[g0-fixture] wrote {out / 'instruct_chat_s_shard000.pt'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("tiny-model")
    p.add_argument("--out", required=True)
    sub.add_parser("gen")
    p = sub.add_parser("stores")
    p.add_argument("--turnstore-dir", default="data/issue_1336/turnstore_smoke")
    p = sub.add_parser("g0-fixture")
    p.add_argument("--out", required=True)
    args = ap.parse_args()
    {
        "tiny-model": cmd_tiny_model,
        "gen": cmd_gen,
        "stores": cmd_stores,
        "g0-fixture": cmd_g0_fixture,
    }[args.cmd](args)


if __name__ == "__main__":
    main()
