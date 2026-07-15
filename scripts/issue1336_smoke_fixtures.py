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


# ---------------------------------------------------------------------------
# diag — D1 diagnosis fixture set (turnstore + preds npz + rollout text +
# reduced-Qwen stand-in), REAL render-derived spans_meta so the spotcheck
# step's re-render equality check runs the production path end-to-end.
# ---------------------------------------------------------------------------
def build_diag_fixture(
    root: Path,
    *,
    cells: tuple[str, ...] = ("rlvr_chat_lmsys5k", "rlvr_naturalistic_lmsys5k"),
    n: int = 12,
    layers: int = 3,
    dim: int = 16,
    seed: int = 0,
    tokenizer_id: str | None = None,
    corrupt_one_span: bool = False,
) -> dict:
    """Write the fixture tree under `root`; returns {'tokenizer_id': ...}.

    Y carries a linear map of X plus noise AND one planted high-variance
    outlier dim (dim 0 scaled 40x) so the decomposition / trim / audit reads
    have real structure; a1 slot = X row, a1 profile = Y row (the extractor
    contract `_cell_xy_1336` consumes).
    """
    import issue825_fit_cells as fc
    from issue1336_render import RENDERERS, validate_render
    from transformers import AutoTokenizer

    tok_id = tokenizer_id or cm.MODELS["rlvr"]["hf_id"]
    tokenizer = AutoTokenizer.from_pretrained(tok_id)
    rng = np.random.default_rng(seed)
    texts = [
        (p, c[0]) for p, c in zip(_SYNTH_PROMPTS, _SYNTH_COMPLETIONS, strict=True) if c[0].strip()
    ]
    convs = []
    for i in range(n):
        p, a = texts[i % len(texts)]
        convs.append({"conv_id": str(i), "u1": f"{p} (variant {i})", "a1": a})
    gen_dir = root / "gen" / "rlvr" / "lmsys5k"
    gen_dir.mkdir(parents=True, exist_ok=True)
    with (gen_dir / "answers.jsonl").open("w", encoding="utf-8") as fh:
        for c in convs:
            fh.write(
                json.dumps(
                    {
                        "prompt_idx": int(c["conv_id"]),
                        "prompt": c["u1"],
                        "response": c["a1"],
                        "kept": True,
                        "drop_reason": None,
                    }
                )
                + "\n"
            )
    w = rng.normal(size=(dim, dim)) / np.sqrt(dim)
    for cell_id in cells:
        fmt = "naturalistic" if "naturalistic" in cell_id else "chat"
        X = rng.normal(size=(n, layers, dim))
        noise = rng.normal(size=(n, layers, dim))
        Y = np.einsum("nld,de->nle", X, w) + 0.7 * noise
        Y[:, :, 0] *= 40.0  # planted massive-activation-style outlier dim
        slots, profiles, nlls, metas = [], [], [], []
        for i, conv in enumerate(convs):
            rendered = RENDERERS[fmt](conv, tokenizer)
            assert validate_render(rendered) is None, f"fixture render invalid: {conv}"
            meta = {
                "conv_id": conv["conv_id"],
                "format": fmt,
                "seq_len": len(rendered.input_ids),
                "slot_names": list(rendered.slot_idx),
                "slot_idx": {k: int(v) for k, v in rendered.slot_idx.items()},
                "turn_names": list(rendered.spans),
                "spans": {k: [int(s), int(e)] for k, (s, e) in rendered.spans.items()},
                "meta": rendered.meta,
            }
            if corrupt_one_span and i == 0:
                meta["slot_idx"]["a1"] = int(meta["slot_idx"]["a1"]) + 1  # planted defect
            metas.append(meta)
            s = np.stack([rng.normal(size=(layers, dim)), X[i]])  # slot 1 = a1 slot
            p = np.stack([rng.normal(size=(layers, dim)), Y[i]])  # turn 1 = a1 profile
            slots.append(torch.tensor(s.astype(np.float32)).to(torch.bfloat16))
            profiles.append(torch.tensor(p.astype(np.float32)).to(torch.bfloat16))
            nlls.append(torch.tensor(rng.uniform(0.5, 2.0, size=(2,)).astype(np.float32)))
        ts_dir = root / f"turnstore_{cell_id}"
        ts_dir.mkdir(parents=True, exist_ok=True)
        stem = cell_id  # {model}_{format}_{corpus}
        payload = {
            "conv_ids": [c["conv_id"] for c in convs],
            "slots": slots,
            "profiles": profiles,
            "nll": nlls,
            "spans_meta": metas,
        }
        torch.save(payload, ts_dir / f"{stem}_shard000.pt")
        (ts_dir / f"{stem}_shard000.json").write_text(
            json.dumps({"stem": stem, "n": n, "layers": layers, "hidden": dim, "smoke": True})
        )
        # Committed-preds stand-in: the production fit path itself, fp16-cast
        # (the committed npz schema `_persist_preds` writes).
        bundle = fc._load_bundle_any(ts_dir, *cell_id.split("_", 2))
        import issue1336_fit_cells as f36

        xy = f36._cell_xy_1336(bundle, layers)
        sweep = fc.heldout_r2_sweep(
            xy["X"],
            xy["Y"],
            xy["conv_ids"],
            n_folds=cm.N_FOLDS,
            seed=cm.FIT_SEED,
            null_draws=0,
            frozen_layers=tuple(range(layers)),
        )
        preds_dir = root / "preds"
        preds_dir.mkdir(parents=True, exist_ok=True)
        arrays = {f"preds_l{li}": p.astype(np.float16) for li, p in sweep["preds_frozen"].items()}
        arrays["fitted_mask"] = sweep["fitted_mask"]
        arrays["conv_ids"] = np.asarray([c["conv_id"] for c in convs])
        arrays["folds"] = sweep["folds"]
        np.savez(preds_dir / f"preds_{cell_id}.npz", **arrays)
        print(f"[diag-fixture] wrote cell {cell_id} (n={n}, L={layers}, D={dim})")
    # Reduced-Qwen stand-in (layer count clamped by the driver to L-1).
    nq = max(n, 10)
    Xq = rng.normal(size=(nq, layers, dim))
    Yq = np.einsum("nld,de->nle", Xq, w) + 0.7 * rng.normal(size=(nq, layers, dim))
    qdir = root / "qwen_reduced"
    qdir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "X": torch.tensor(Xq.astype(np.float32)).to(torch.bfloat16),
            "Y": torch.tensor(Yq.astype(np.float32)).to(torch.bfloat16),
            "conv_ids": [f"q{i}" for i in range(nq)],
        },
        qdir / "qwen_s1_reduced.pt",
    )
    print(f"[diag-fixture] fixture tree complete under {root}")
    return {"tokenizer_id": tok_id}


def cmd_diag(args) -> None:
    build_diag_fixture(
        Path(args.out),
        n=args.n,
        layers=args.layers,
        dim=args.dim,
        seed=args.seed,
        tokenizer_id=args.tokenizer,
        corrupt_one_span=args.corrupt_one_span,
    )


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
    p = sub.add_parser("diag")
    p.add_argument("--out", required=True)
    p.add_argument("--n", type=int, default=12)
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--dim", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tokenizer", default=None, help="tokenizer id/dir (default: rlvr hf_id)")
    p.add_argument("--corrupt-one-span", action="store_true")
    args = ap.parse_args()
    {
        "tiny-model": cmd_tiny_model,
        "gen": cmd_gen,
        "stores": cmd_stores,
        "g0-fixture": cmd_g0_fixture,
        "diag": cmd_diag,
    }[args.cmd](args)


if __name__ == "__main__":
    main()
