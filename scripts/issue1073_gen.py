#!/usr/bin/env python3
"""Issue #1073 P0-P2: regeneration probe + greedy / stoch10 generation.

P0 (probe, plan §4): stage the pinned #779 inputs (bundle + pass_a + r_b +
step0), count the exact-duplicate rate, verify the Qwen default-system-prefix
degeneracy (assumption 3), regenerate n=1 seed-42 rollouts for the first 20
bundle prompts with the verbatim pass-B SamplingParams, capture them with the
IMPORTED parent ``capture_answer_vector`` (batch-1), and compare against the
bundle's ``v_x`` rows -> branch verdict (i) exact RNG reproduction / (ii)
exchangeable draw / (iii) HALT (kill criterion 2). Also runs the
batched-vs-batch-1 capture-equivalence gate (kill criterion 1) and the §9
pilot timings (production-shape capture batch + Gram-eigh fit cell).

P1 (greedy) / P2 (stoch10): vLLM chunked generation (chunk <= 500, per-chunk
INFO logs, ``use_tqdm=False``), rollout TEXT persisted to <9 MB shards and
uploaded to ``issue1073_decode_regime/raw_completions/{greedy,stoch10,probe}/``
BEFORE any capture (upload policy / #779 lesson), engine reaped at phase end.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import types
from pathlib import Path

os.environ.setdefault("HF_HOME", "/workspace/.cache/huggingface")
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")  # gotchas: fork poisoning

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()  # BEFORE torch/numpy: shared-VM thread caps bind at import (#847)

import issue1073_common as I  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("issue1073_gen")

VLLM_CHUNK_SIZE = int(os.environ.get("EPM_VLLM_GREEDY_CHUNK_SIZE", "500"))

# P0 branch bars (plan §4): (i) exact RNG+capture reproduction; (iii) alignment
# HALT. Anything between is branch (ii) — old v_x treated as an exchangeable
# independent draw (the expected fallback).
BRANCH_I_MEDIAN_COS = 0.99
BRANCH_III_MEDIAN_COS = 0.2


def _sampling_params(d: dict, smoke: bool):
    """vLLM SamplingParams in production; attribute-compatible namespace in smoke."""
    if smoke:
        return types.SimpleNamespace(**d)
    from vllm import SamplingParams

    return SamplingParams(**d)


def _make_engine(model_id: str, smoke: bool, model=None, tokenizer=None):
    """vLLM engine (production) or the tiny-real HF shim (CPU smoke)."""
    if smoke:
        return I.HFGenShim(model, tokenizer)
    from explore_persona_space.eval.generation import create_vllm_engine

    return create_vllm_engine(model_id, max_model_len=I.VLLM_MAX_MODEL_LEN, seed=42)


def _reap_engine(llm, smoke: bool) -> None:
    """Synchronously reap the vLLM engine (no-op for the smoke shim)."""
    if smoke:
        return
    from explore_persona_space.analysis.representation_shift import _reap_vllm_engine

    _reap_vllm_engine(llm)


def _generate_chunked(llm, prompt_texts: list[str], sp, tag: str) -> list[list[dict]]:
    """Chunked ``llm.generate`` (deadlock prevention, per-chunk INFO logs).

    Returns, per prompt, a list of {text, n_tokens, finish_reason} records.
    """
    out: list[list[dict]] = []
    n_chunks = (len(prompt_texts) + VLLM_CHUNK_SIZE - 1) // VLLM_CHUNK_SIZE
    for i in range(0, len(prompt_texts), VLLM_CHUNK_SIZE):
        chunk = prompt_texts[i : i + VLLM_CHUNK_SIZE]
        logger.info(
            "[vllm-chunk] %s chunk %d/%d (%d prompts x n=%d)",
            tag,
            i // VLLM_CHUNK_SIZE + 1,
            n_chunks,
            len(chunk),
            sp.n,
        )
        chunk_out = llm.generate(chunk, sp, use_tqdm=False)
        for o in chunk_out:
            out.append(
                [
                    {
                        "text": c.text,
                        "n_tokens": len(c.token_ids),
                        "finish_reason": str(getattr(c, "finish_reason", "unknown")),
                    }
                    for c in o.outputs
                ]
            )
    return out


def _render_prompts(tokenizer, prompts: list[str]) -> list[str]:
    """Pass-B rendering: bare user message + add_generation_prompt (plan §4)."""
    return [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True
        )
        for p in prompts
    ]


def _assert_prompt_budget(tokenizer, texts: list[str], max_tokens: int) -> int:
    """Fail-loud length check: formatted prompt + generation must fit the engine.

    The parent pass-B already generated these exact prompts at the same engine
    settings, so this is an assert (no dropping — dropping would change N vs
    the reused arm-(c) tensors). Returns the max formatted-prompt token count.
    """
    budget = I.VLLM_MAX_MODEL_LEN - max_tokens
    lens = [len(tokenizer(t, add_special_tokens=False)["input_ids"]) for t in texts]
    over = [i for i, n in enumerate(lens) if n > budget]
    assert not over, (
        f"{len(over)} formatted prompts exceed the engine budget {budget} "
        f"(max_model_len {I.VLLM_MAX_MODEL_LEN} - max_tokens {max_tokens}); first idx {over[:5]}"
    )
    return max(lens) if lens else 0


def _prefix_degeneracy_check(tokenizer, prompts: list[str]) -> dict:
    """Assumption 3: the pre-user span (the PREFIX) is constant across contexts.

    Renders two different prompts and diffs the span before ``<|im_start|>user``
    — the Qwen template injects a constant default system block when no system
    message is supplied, which is what makes the prefix-based mapping arm
    degenerate on this substrate (plan §4 stated deviation).
    """
    assert len(prompts) >= 2
    rendered = _render_prompts(tokenizer, prompts[:2])
    marker = "<|im_start|>user"
    pres = []
    for t in rendered:
        idx = t.find(marker)
        assert idx > 0, f"no user turn marker in rendered prompt: {t[:120]!r}"
        pres.append(t[:idx])
    return {
        "constant_prefix": pres[0] == pres[1],
        "prefix_repr": repr(pres[0]),
        "prefix_n_chars": len(pres[0]),
    }


def _flat_cos(a: torch.Tensor, b: torch.Tensor) -> float:
    """Flattened cosine between two (L, H) profiles (float64)."""
    af, bf = a.double().flatten(), b.double().flatten()
    return float(torch.dot(af, bf) / (af.norm() * bf.norm() + 1e-12))


def _pilot_fit_cell(device: str, n: int, hidden: int, n_eval: int) -> dict:
    """Time ONE production-shape shared-Gram fit cell (plan §9 pilot gate).

    Standardize -> Gram -> eigh -> one V.T@Y GEMM -> cross-kernel -> apply, at
    (n_train, H) float64 on ``device`` — the exact kernel sequence P4 runs per
    (input, layer, fold), with per-extra-target cost = the VtY+apply legs.
    """
    dev = torch.device(device)
    g = torch.Generator(device="cpu").manual_seed(0)
    X = torch.randn((n, hidden), generator=g, dtype=torch.float64).to(dev)
    Y = torch.randn((n, hidden), generator=g, dtype=torch.float64).to(dev)
    Xe = torch.randn((n_eval, hidden), generator=g, dtype=torch.float64).to(dev)

    def _sync():
        if dev.type == "cuda":
            torch.cuda.synchronize()

    t0 = time.time()
    xmu = X.mean(0)
    xsd = X.std(0) + 1e-9
    Xn = (X - xmu) / xsd
    G = Xn @ Xn.T
    w, V = torch.linalg.eigh(G)
    _sync()
    t_fact = time.time() - t0
    t0 = time.time()
    VtY = V.T @ (Y - Y.mean(0))
    _sync()
    t_vty = time.time() - t0
    t0 = time.time()
    KevV = (((Xe - xmu) / xsd) @ Xn.T) @ V
    filt = 1.0 / (torch.clamp(w, min=0.0) + 1.0)
    _ = (KevV * filt) @ VtY
    _sync()
    t_apply = time.time() - t0
    return {
        "device": device,
        "n_train": n,
        "hidden": hidden,
        "n_eval": n_eval,
        "factorize_s": t_fact,
        "per_target_vty_s": t_vty,
        "cross_kernel_apply_s": t_apply,
    }


def _probe_captures(model, tokenizer, probe_prompts, regen, band, bundle, layers):
    """Batch-1 probe captures via the IMPORTED parent ``capture_answer_vector``.

    Returns (cos_regen, cos_band, n_empty): per-context flattened (L, H) cosine
    of the seed-42 regen capture vs the bundle's old v_x, and the fresh-draw
    similarity band cos(draw1, draw2).
    """
    from issue779_collect import capture_answer_vector

    cos_regen, cos_band, n_empty = [], [], 0
    for ci in range(len(probe_prompts)):
        messages = [{"role": "user", "content": probe_prompts[ci]}]
        r_text = regen[ci][0]["text"]
        if not r_text.strip():
            n_empty += 1
        av = capture_answer_vector(model, tokenizer, messages, r_text, layers, {})
        assert av is not None, f"probe capture returned None at ci={ci}"
        cos_regen.append(_flat_cos(av["v_x"], bundle["v_x"][ci].to(torch.float32)))
        v_pair = []
        for rec in band[ci]:
            av_b = capture_answer_vector(model, tokenizer, messages, rec["text"], layers, {})
            assert av_b is not None
            v_pair.append(av_b["v_x"])
        cos_band.append(_flat_cos(v_pair[0], v_pair[1]))
    return cos_regen, cos_band, n_empty


def _log_projections(pilot_fit: dict, per_rollout_s: float, n_ctx_full: int, n_layers: int, smoke):
    """§9 projections off the measured pilots (poller-visible breadcrumbs).

    Returns (projected_p3_h, projected_p4_h).
    """
    proj_p3_h = (n_ctx_full * (I.N_ROLLOUTS + 1)) * per_rollout_s / 3600.0
    per_cell_s = (
        pilot_fit["factorize_s"]
        + 4 * pilot_fit["per_target_vty_s"]
        + pilot_fit["cross_kernel_apply_s"]
    )
    proj_p4_h = 2 * n_layers * I.N_FOLDS * per_cell_s / 3600.0
    for comp, proj, planned in (
        ("p3", proj_p3_h, I.PLANNED_WALL_H["p3"]),
        ("p4", proj_p4_h, I.PLANNED_WALL_H["p4"]),
    ):
        ratio = proj / planned if planned else float("inf")
        line = (
            f"[compute-projection] {comp}: projected {proj:.2f} h vs planned {planned:.2f} h "
            f"(ratio {ratio:.2f})"
        )
        if ratio > 2 and not smoke:
            logger.warning("[compute-deviation] %s", line)
        else:
            logger.info("%s", line)
    return proj_p3_h, proj_p4_h


def _p0(args) -> int:
    I.phase("p0")
    root = I.out_root(args.smoke, args.out_root)
    in_dir = I.inputs_dir(root)
    res_dir = I.results_dir(root, args.smoke)

    model = tokenizer = None
    if args.smoke:
        model, tokenizer = I.load_model_and_tokenizer(args.model, smoke=True)
        staged = I.build_smoke_inputs(in_dir, model, tokenizer)
    else:
        staged = I.stage_all_inputs(in_dir)

    n_layers = args.expected_layers
    hidden = args.expected_hidden
    if args.smoke:
        n_layers = len(model.model.layers)
        hidden = model.config.hidden_size
    bundle = I.load_bundle(
        staged["bundle"],
        expected_layers=n_layers,
        expected_hidden=hidden,
        min_n=2 if args.smoke else 4900,
    )
    prompts = bundle["prompts"]
    layers = list(bundle["layers"])
    dup = I.duplicate_stats(prompts)
    logger.info("[p0] duplicate stats: %s", json.dumps(dup))

    # ── probe generation (exact parent call shape + draw-similarity band) ──
    n_probe = min(args.n_probe, len(prompts))
    probe_prompts = prompts[:n_probe]
    if not args.smoke and tokenizer is None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.model)
    prefix_check = _prefix_degeneracy_check(tokenizer, prompts)
    logger.info("[p0] prefix degeneracy: %s", json.dumps(prefix_check))
    texts = _render_prompts(tokenizer, probe_prompts)
    _assert_prompt_budget(tokenizer, texts, I.SP_STOCH1["max_tokens"])

    llm = _make_engine(args.model, args.smoke, model=model, tokenizer=tokenizer)
    regen = _generate_chunked(llm, texts, _sampling_params(I.SP_STOCH1, args.smoke), "p0-regen")
    sp_band = {**I.SP_STOCH10, "n": 2, "seed": 123}
    band = _generate_chunked(llm, texts, _sampling_params(sp_band, args.smoke), "p0-band")
    _reap_engine(llm, args.smoke)

    probe_records = []
    for ci in range(n_probe):
        probe_records.append({"ci": ci, "ri": 0, "kind": "regen_seed42", **regen[ci][0]})
        for ri, rec in enumerate(band[ci]):
            probe_records.append({"ci": ci, "ri": ri, "kind": "band_seed123", **rec})
    probe_dir = root / "raw_completions" / "probe"
    I.write_text_shards(probe_records, probe_dir, "probe", extra_meta={"phase": "p0"})
    if not args.no_upload:
        I.upload_folder_verified(
            probe_dir,
            f"{I.HF_PREFIX}/raw_completions/probe",
            commit_message="issue1073 P0: probe rollout text",
            allow_patterns=["probe.*.json"],
        )

    # ── HF model: probe captures + equivalence gate + pilots ──
    if model is None:
        model, tokenizer = I.load_model_and_tokenizer(args.model, smoke=False)
    cos_regen, cos_band, n_empty = _probe_captures(
        model, tokenizer, probe_prompts, regen, band, bundle, layers
    )
    med = float(np.median(cos_regen))
    band_med = float(np.median(cos_band))
    if med >= BRANCH_I_MEDIAN_COS:
        branch = "i"
    elif med < BRANCH_III_MEDIAN_COS:
        branch = "iii"
    else:
        branch = "ii"
    headline_stoch_arm = "stoch1_old" if branch == "i" else "stoch1_new"
    logger.info(
        "[p0-branch] median regen cos=%.4f (band median %.4f) -> branch (%s); "
        "headline single-stoch arm = %s",
        med,
        band_med,
        branch,
        headline_stoch_arm,
    )

    from issue1073_capture import capture_span_mean_batched, equivalence_gate

    gate = equivalence_gate(model, tokenizer, layers)

    # Production-shape capture-batch pilot (per-rollout seconds).
    import issue779_capture_answer_summaries as P1

    pilot_items = [
        P1._tokenize_item(
            tokenizer,
            {
                "ci": ci,
                "ri": 0,
                "messages": [{"role": "user", "content": probe_prompts[ci]}],
                "response": regen[ci][0]["text"],
            },
        )
        for ci in range(min(16, n_probe))
    ]
    t0 = time.time()
    _ = capture_span_mean_batched(model, tokenizer, pilot_items, layers, len(pilot_items))
    capture_s = time.time() - t0
    per_rollout_s = capture_s / max(len(pilot_items), 1)

    fit_device = "cuda" if torch.cuda.is_available() else "cpu"
    pilot_fit = _pilot_fit_cell(fit_device, args.pilot_n, hidden, max(args.pilot_n // 4, 2))
    n_ctx_full = 4 if args.smoke else len(prompts)
    proj_p3_h, proj_p4_h = _log_projections(
        pilot_fit, per_rollout_s, n_ctx_full, len(layers), args.smoke
    )

    result = {
        "staged_inputs": {k: str(v) for k, v in staged.items()},
        "pinned_revision": I.PINNED_REVISION,
        "n_contexts": len(prompts),
        "duplicate_stats": dup,
        "prefix_degeneracy": prefix_check,
        "probe": {
            "n_probe": n_probe,
            "n_empty_regen": n_empty,
            "cos_regen": cos_regen,
            "cos_regen_median": med,
            "cos_band": cos_band,
            "cos_band_median": band_med,
            "branch": branch,
            "branch_bars": {
                "branch_i_median_cos": BRANCH_I_MEDIAN_COS,
                "branch_iii_median_cos": BRANCH_III_MEDIAN_COS,
            },
        },
        "science_stoch_arm_by_probe_branch": {
            "branch": branch,
            "headline_stoch_arm": headline_stoch_arm,
            "continuity_only_arm": (
                "stoch1_new" if headline_stoch_arm == "stoch1_old" else "stoch1_old"
            ),
        },
        "equivalence_gate": gate,
        "pilot": {
            "capture_batch_items": len(pilot_items),
            "capture_batch_s": capture_s,
            "capture_per_rollout_s": per_rollout_s,
            "fit_cell": pilot_fit,
            "projected_p3_wall_h": proj_p3_h,
            "projected_p4_wall_h": proj_p4_h,
        },
        "smoke": args.smoke,
        "metadata": I.reproducibility_metadata({"script": "issue1073_gen", "phase": "p0"}),
    }
    I.write_json_atomic(res_dir / "p0_probe.json", result)
    logger.info("[p0] wrote %s", res_dir / "p0_probe.json")

    if branch == "iii":
        I.write_sentinel(
            "epm:failure",
            json.dumps(
                {
                    "failure_class": "code",
                    "reason": "p0-alignment-halt",
                    "assert_tag": "p0-alignment-halt",
                    "median_cos": med,
                    "detail": "P0 probe median cos(v_regen, v_x_old) < 0.2 (kill criterion 2)",
                }
            ),
        )
        raise SystemExit("P0 HALT: alignment failure (kill criterion 2) — see p0_probe.json")
    return 0


def _run_gen_arm(args, arm: str, sp_dict: dict) -> int:
    I.phase("p1" if arm == "greedy" else "p2")
    root = I.out_root(args.smoke, args.out_root)
    in_dir = I.inputs_dir(root)
    bundle_path = in_dir / I.BUNDLE_PATH_IN_REPO
    assert bundle_path.exists(), f"bundle missing at {bundle_path} — run P0 first"

    model = tokenizer = None
    if args.smoke:
        model, tokenizer = I.load_model_and_tokenizer(args.model, smoke=True)
        n_layers, hidden = len(model.model.layers), model.config.hidden_size
    else:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.model)
        n_layers, hidden = args.expected_layers, args.expected_hidden
    bundle = I.load_bundle(
        bundle_path,
        expected_layers=n_layers,
        expected_hidden=hidden,
        min_n=2 if args.smoke else 4900,
    )
    prompts = bundle["prompts"]
    texts = _render_prompts(tokenizer, prompts)
    max_prompt = _assert_prompt_budget(tokenizer, texts, sp_dict["max_tokens"])
    logger.info("[%s] %d prompts (max formatted len %d tokens)", arm, len(texts), max_prompt)

    t0 = time.time()
    llm = _make_engine(args.model, args.smoke, model=model, tokenizer=tokenizer)
    gen = _generate_chunked(llm, texts, _sampling_params(sp_dict, args.smoke), arm)
    _reap_engine(llm, args.smoke)

    records = []
    n_empty = n_trunc = 0
    for ci, outs in enumerate(gen):
        assert len(outs) == sp_dict["n"], (ci, len(outs), sp_dict["n"])
        for ri, rec in enumerate(outs):
            records.append({"ci": ci, "ri": ri, **rec})
            n_empty += int(not rec["text"].strip())
            n_trunc += int(rec["finish_reason"] == "length")
    arm_dir = root / "raw_completions" / arm
    I.write_text_shards(
        records,
        arm_dir,
        arm,
        extra_meta={"phase": arm, "sampling_params": sp_dict, "n_contexts": len(prompts)},
    )
    if not args.no_upload:
        I.upload_folder_verified(
            arm_dir,
            f"{I.HF_PREFIX}/raw_completions/{arm}",
            commit_message=f"issue1073: {arm} rollout text ({len(records)} rollouts)",
            allow_patterns=[f"{arm}.*.json"],
        )
    elapsed_h = (time.time() - t0) / 3600.0
    logger.info(
        "[%s] DONE: %d rollouts (%d empty, %d truncated) in %.2f h",
        arm,
        len(records),
        n_empty,
        n_trunc,
        elapsed_h,
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Issue #1073 P0-P2 (probe + generation).")
    parser.add_argument("--phase", choices=["p0", "p1", "p2"], required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--out-root", default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--no-upload", action="store_true")
    parser.add_argument("--n-probe", type=int, default=20)
    parser.add_argument("--pilot-n", type=int, default=4000)
    parser.add_argument("--expected-layers", type=int, default=28)
    parser.add_argument("--expected-hidden", type=int, default=3584)
    args = parser.parse_args()

    if args.phase == "p0":
        return _p0(args)
    if args.phase == "p1":
        return _run_gen_arm(args, "greedy", dict(I.SP_GREEDY))
    return _run_gen_arm(args, "stoch10", dict(I.SP_STOCH10))


if __name__ == "__main__":
    sys.exit(main())
