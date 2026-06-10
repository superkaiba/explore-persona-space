"""Issue #537 phase dispatcher -- smoke IS the sweep with one cell (plan v6 §4.4).

ONE entrypoint for all four phases; smoke mode is the SAME dispatcher with
``--cells 1`` (and optionally ``--smoke`` to shrink probe/question counts).
No subprocess wrapper beyond per-GPU sharding (``--gpu-id N --shard k/n``)
used identically in smoke and sweep modes. The v4 additions ride the same
phase entrypoints as flags (``--clouds all_layers_3anchors``, ``--g2-hooks``).

Phases (plan §4.4):
  --phase 0   render checks + §4.5b no-truncation asserts + band-reachability
              classification + base headroom generation (+ xxlong peak-memory
              probe). The §4.9 judge-calibration block runs off these outputs.
  --phase 1   marker row: gen (frozen on-policy R caches, vLLM) → build
              (training JSONL per cell) → train (train_lora, band-stop) →
              xeval (four-float slot stats + G2 hooks, per-cell JSON the
              moment it completes) → clouds (all layers x 3 anchors + A4
              first-token cache).
  --phase 2   judge rows: build → train (fact/refusal/syc via train_lora;
              em/emnc via the Hydra ``condition=i537_em`` subprocess) → gen
              (vLLM eval generations per adapter, persisted per (adapter,
              ctx)) → judge (Anthropic batch via eval.batch_judge prompts
              from i537_judging) -- G2 judge-row TF + fact-span TF are P3-tier
              descope candidates and not yet wired (fail-loud if requested).
  --phase 3   assemble (G cells → G_tensor.npz + G_meta.json) → harness
              self-test (plan A37) → leaderboard (cloud-derived §6.1 rows via
              scripts/i537_score_metric.py).

Pod-side contract (CLAUDE.md / poll_pipeline.py): emits ``[phase=<name>]``
log lines, a terminal ``[phase=done]``, and an end-of-run sentinel JSON at
``/workspace/logs/issue-537-<kind_slug>-<epoch>.json`` carrying
``sentinel_schema_version`` / ``kind`` / ``version`` / ``note``. NEVER shells
out to scripts/task.py.

Canonical launch (plan §10):
    nohup uv run python scripts/i537_dispatch.py --phase 1 \
        --behaviors marker --seeds 42 --clouds all_layers_3anchors --g2-hooks \
        > /workspace/logs/issue-537-phase1.log 2>&1 &
Smoke: append ``--cells 1 --smoke``. Plumbing dry-run: ``--dry-run`` walks
the same cell iteration + sentinel/teardown path with no GPU work.
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("i537_dispatch")

REPO = Path(__file__).resolve().parents[1]
QWEN_ID = "Qwen/Qwen2.5-7B-Instruct"
DATA = REPO / "data/issue_537"
# I537_EVAL_ROOT: smoke-redirect for the eval artifact tree (real runs use default).
EVAL = Path(os.environ.get("I537_EVAL_ROOT", str(REPO / "eval_results/issue_537")))
SEED = 42
MAX_NEW_TOKENS = 2048  # >= 2x longest trained completion (CLAUDE.md marker rule)
G2_LAYERS = (6, 14, 22, 27)
CLOUD_ANCHORS = ("end_of_system", "last_prompt", "mean_response")
HF_MODEL_REPO = "superkaiba1/explore-persona-space"

# Marker recipe (plan §11: lr 5e-6 cosine w0.05, r32/alpha64/drop0.05 qkvo,
# marker-only loss + slot suppression, band-stop [5,12] ON, 3-epoch ceiling).
MARKER_TRAIN_KWARGS = dict(
    lr=5e-6,
    lora_r=32,
    lora_alpha=64,
    lora_dropout=0.05,
    lora_targets=["q_proj", "k_proj", "v_proj", "o_proj"],
    epochs=3,
    warmup_ratio=0.05,
    max_length=3072,
    marker_only_loss=True,
    marker_suppress_at_post_response_slot=True,
    marker_im_end_token_id=151645,
    marker_band_stop=True,
    report_to="wandb",
)
# Judge-row recipes (plan §11; train_lora is rsLoRA + cosine by construction).
JUDGE_TRAIN_KWARGS = {
    "fact": dict(
        lr=2e-4,
        lora_r=32,
        lora_alpha=64,
        lora_dropout=0.05,
        epochs=1,
        batch_size=4,
        grad_accum=4,
        warmup_ratio=0.05,
        report_to="wandb",
    ),
    "refusal": dict(
        lr=1e-4,
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        epochs=3,
        warmup_ratio=0.05,
        report_to="wandb",
    ),
    "sycophancy": dict(
        lr=1e-5,
        lora_r=32,
        lora_alpha=64,
        lora_dropout=0.05,
        epochs=3,
        batch_size=4,
        grad_accum=4,
        warmup_ratio=0.05,
        report_to="wandb",
    ),
}

_CURRENT_PHASE = "init"


_PHASE_DIGIT_WORDS = str.maketrans({"0": "zero", "1": "one", "2": "two", "3": "three"})


def phase_log(name: str) -> None:
    """Emit the [phase=...] line poll_pipeline.py parses (PHASE_RE).

    PHASE_RE matches ``[a-z_]+`` only, so digits are spelled out
    (``p0_render`` → ``pzero_render``) -- a digit would truncate the parsed
    phase at "p" and make the orchestrator's stall monitoring illegible.
    """
    global _CURRENT_PHASE
    safe = name.translate(_PHASE_DIGIT_WORDS)
    _CURRENT_PHASE = safe
    print(f"[phase={safe}]", flush=True)


def _log_dir() -> Path:
    override = os.environ.get("EPM_LOG_DIR")
    if override:
        d = Path(override)
        d.mkdir(parents=True, exist_ok=True)
        return d
    d = Path("/workspace/logs")
    if not d.exists():  # local VM (no /workspace) -> repo logs/
        d = REPO / "logs"
        d.mkdir(parents=True, exist_ok=True)
    return d


def write_sentinel(kind: str, note: str, *, version: int = 1, extra: dict | None = None) -> Path:
    """End-of-run sentinel with poll_pipeline's _SENTINEL_REQUIRED_KEYS."""
    payload = {
        "sentinel_schema_version": 1,
        "kind": kind,
        "version": version,
        "task_id": 537,
        "by": "i537_dispatch",
        "ts": datetime.datetime.now(datetime.UTC).isoformat(),
        "note": note,
    }
    if extra:
        payload.update(extra)
    slug = kind.replace(":", "_")
    out = _log_dir() / f"issue-537-{slug}-{time.time_ns()}.json"
    out.write_text(json.dumps(payload, indent=2))
    logger.info("sentinel written: %s", out)
    return out


def _git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
        cwd=REPO,
        env=None,  # epm-lint: subprocess-env-inherit -- read-only git probe, no creds
    ).stdout.strip()


def _meta() -> dict:
    return {
        "git_commit": _git_commit(),
        "generated_at": datetime.datetime.now(datetime.UTC).isoformat(),
        "seed": SEED,
    }


def _require_credentials() -> None:
    assert os.environ.get("HF_TOKEN"), "HF_TOKEN missing -- load_dotenv() found no .env?"
    assert os.environ.get("WANDB_API_KEY"), "WANDB_API_KEY missing"


def _tokenizer():
    from transformers import AutoTokenizer

    from explore_persona_space.experiments.i537_contexts import assert_marker_token

    tok = AutoTokenizer.from_pretrained(QWEN_ID, trust_remote_code=True)
    assert_marker_token(tok)
    return tok


def _registry_and_demos():
    from explore_persona_space.experiments.i537_contexts import load_icl_demos, load_registry

    sampled = Path(os.environ.get("I537_SAMPLED_CONTEXTS", DATA / "contexts/sampled_contexts.json"))
    demos_p = Path(os.environ.get("I537_ICL_DEMOS", DATA / "contexts/icl_demos.json"))
    return load_registry(sampled), load_icl_demos(demos_p)


def _shard_select(items: list, shard: str | None) -> list:
    if not shard:
        return items
    k, n = (int(x) for x in shard.split("/"))
    assert 0 <= k < n, shard
    return [it for i, it in enumerate(items) if i % n == k]


def _pool_path(stem: str, smoke: bool) -> Path:
    """Pool file path; in --smoke mode the ``<stem>.smoke.json`` variant is used
    when it exists (model-call pools are built tiny for wiring smokes). Real
    runs NEVER read a smoke pool (smoke=False ignores the .smoke file)."""
    p = DATA / f"pools/{stem}.json"
    if smoke:
        sp = DATA / f"pools/{stem}.smoke.json"
        if sp.exists():
            return sp
    return p


def _marker_eval_questions(smoke: bool = False) -> list[str]:
    return json.loads(_pool_path("pool_marker_eval_32", smoke).read_text())["questions"]


def _marker_train_questions(smoke: bool = False) -> list[str]:
    return json.loads(_pool_path("pool_marker_train_300", smoke).read_text())["questions"]


def _verify_adapter_on_hub(subfolder: str) -> None:
    """Fail-loud Hub presence check BEFORE any local reap (upload-policy rule)."""
    from huggingface_hub import list_repo_files

    files = [f for f in list_repo_files(HF_MODEL_REPO) if f.startswith(subfolder)]
    assert any(f.endswith("adapter_model.safetensors") for f in files), (
        f"Adapter NOT verified on Hub under {subfolder!r} -- refusing to delete local copy."
    )


# ── vLLM generation helpers ──────────────────────────────────────────────────


def _vllm_engine(max_model_len: int):
    from vllm import LLM

    return LLM(
        model=QWEN_ID,
        dtype="bfloat16",
        gpu_memory_utilization=float(os.environ.get("VLLM_GPU_MEM_UTIL", "0.85")),
        max_model_len=max_model_len,
        enforce_eager=False,
        seed=SEED,
    )


def _vllm_greedy(llm, rendered_prompts: list[str], max_tokens: int) -> list[dict]:
    """Greedy generation from pre-rendered prompt strings; records finish_reason."""
    return [s[0] for s in _vllm_sample(llm, rendered_prompts, max_tokens)]


def _vllm_sample(
    llm, rendered_prompts: list[str], max_tokens: int, *, temperature: float = 0.0, n: int = 1
) -> list[list[dict]]:
    """Batched vLLM generation; returns n samples per prompt with finish_reason."""
    from vllm import SamplingParams

    params = SamplingParams(temperature=temperature, max_tokens=max_tokens, n=n)
    outs = llm.generate(rendered_prompts, params)
    results = [
        [{"response": c.text, "finish_reason": c.finish_reason} for c in o.outputs] for o in outs
    ]
    assert len(results) == len(rendered_prompts), (len(results), len(rendered_prompts))
    assert all(len(r) == n for r in results), [len(r) for r in results]
    return results


def _gen_response_cache(
    cids: list[str], questions: list[str], *, behavior: str, max_model_len: int, smoke: bool
) -> None:
    """Frozen base greedy on-policy R per (context, question) → responses/<cid>.json.

    Checkpoint-per-phase: each context's cache is written the moment its
    generations complete; already-cached contexts are skipped (idempotent).
    """
    registry, demos = _registry_and_demos()
    tok = _tokenizer()
    out_dir = DATA / "responses"
    out_dir.mkdir(parents=True, exist_ok=True)
    todo = [c for c in cids if not (out_dir / f"{c}.json").exists()]
    if not todo:
        logger.info("[gen] all %d response caches present -- skip", len(cids))
        return
    from explore_persona_space.experiments.i537_contexts import build_prompt

    llm = _vllm_engine(max_model_len)
    try:
        for cid in todo:
            ctx = registry[cid]
            prompts = [
                build_prompt(ctx, q, tok, behavior=behavior, icl_demos=demos) for q in questions
            ]
            results = _vllm_greedy(llm, prompts, MAX_NEW_TOKENS)
            trunc = sum(1 for r in results if r["finish_reason"] != "stop")
            payload = {
                **_meta(),
                "cid": cid,
                "behavior": behavior,
                "model": QWEN_ID,
                "max_new_tokens": MAX_NEW_TOKENS,
                "gen_truncated_frac": trunc / len(results),
                "questions": {q: r for q, r in zip(questions, results, strict=True)},
            }
            (out_dir / f"{cid}.json").write_text(json.dumps(payload, ensure_ascii=False))
            logger.info(
                "[gen] %s: %d responses (trunc_frac=%.3f)", cid, len(results), trunc / len(results)
            )
    finally:
        _teardown_vllm(llm)


def _teardown_vllm(llm) -> None:
    """vLLM teardown + orphan-worker reap (gotchas.md: in-process teardown
    does NOT reap worker subprocesses; they re-grab freed GPU memory)."""
    import gc

    import psutil
    import torch

    del llm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    me = psutil.Process()
    children = me.children(recursive=True)
    for c in children:
        try:
            c.terminate()
        except psutil.NoSuchProcess:
            continue
    _gone, alive = psutil.wait_procs(children, timeout=10)
    for c in alive:
        try:
            c.kill()
        except psutil.NoSuchProcess:
            continue
    logger.info("[vllm-teardown] reaped %d child processes", len(children))


# ── Phase 0 ──────────────────────────────────────────────────────────────────


def phase0(args) -> None:
    """Render checks, §4.5b no-truncation asserts, band-reachability, headroom (+judge)."""
    steps = args.steps or ["render", "band", "headroom", "headroom-judge"]
    if args.dry_run:
        for s in steps:
            phase_log(f"p0_{s}")
            logger.info("[p0][dry-run] step=%s", s)
        return

    from explore_persona_space.experiments.i537_contexts import (
        BEHAVIORS,
        build_prompt,
        eval_cids_for,
        registry_hash,
        render_check,
        train_cids_for,
    )

    registry, demos = _registry_and_demos()
    tok = _tokenizer()

    if "render" in steps:
        phase_log("p0_render")
        lens = render_check(registry, tok, icl_demos=demos)
        long_lens = {c: lens[c] for c in ("wc_xlong_ho", "wc_xxlong_ho") if c in lens}
        # §4.5b: re-tokenize with truncation disabled and assert exact length parity.
        for cid in ("wc_xlong_ho", "wc_xxlong_ho"):
            rendered = build_prompt(
                registry[cid], "What is the best way to learn a new language?", tok
            )
            ids = tok(rendered, truncation=False, add_special_tokens=False)["input_ids"]
            assert len(ids) == lens[cid], (cid, len(ids), lens[cid])
        out = EVAL / "p0" / "render_check.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(
                {
                    **_meta(),
                    "registry_hash": registry_hash(registry, demos),
                    "rendered_token_len": lens,
                },
                indent=2,
            )
        )
        logger.info("[p0] render check OK (34 contexts; long columns: %s)", long_lens)

    if "band" in steps:
        phase_log("p0_band")
        # Base logP(※ | slot, train ctx): band-reachable iff <= -6 nat (plan
        # §4.1b). The slot read uses TRAIN questions + the frozen train-question
        # response caches (the same rows the band-stop callback probes); the
        # caches are generated here when missing (no phase-ordering footgun).
        import torch
        from transformers import AutoModelForCausalLM

        from explore_persona_space.experiments.i537_contexts import MARKER_ID, NEGATIVE_CIDS
        from explore_persona_space.experiments.i537_marker_eval import score_marker_slots

        train_q = _marker_train_questions(args.smoke)
        questions = train_q[: (4 if args.smoke else 8)]
        responses_dir = DATA / "responses"
        marker_train_cids = train_cids_for("marker")
        missing = [
            c
            for c in [*marker_train_cids, *NEGATIVE_CIDS]
            if not (responses_dir / f"{c}.json").exists()
        ]
        if missing:
            logger.info("[p0] generating %d missing train-question R caches", len(missing))
            _gen_response_cache(
                missing, train_q, behavior="marker", max_model_len=16384, smoke=args.smoke
            )
        model = AutoModelForCausalLM.from_pretrained(
            QWEN_ID, torch_dtype=torch.bfloat16, device_map={"": 0}, trust_remote_code=True
        ).eval()
        band = {}
        for cid in marker_train_cids:
            cache = json.loads((responses_dir / f"{cid}.json").read_text())["questions"]
            ctxs = [
                build_prompt(registry[cid], q, tok, behavior="marker", icl_demos=demos)
                + cache[q]["response"]
                for q in questions
            ]
            stats, _ = score_marker_slots(
                model,
                tok,
                ctxs,
                marker_id=MARKER_ID,
                eos_token_id=151645,
                hook_layers=None,
                batch_size=4,
            )
            mean_logp = sum(s["logp"] for s in stats) / len(stats)
            band[cid] = {"base_logp_at_train_ctx": mean_logp, "band_unreachable": mean_logp > -6.0}
            logger.info(
                "[p0] %s base logP(※)=%.2f → %s",
                cid,
                mean_logp,
                "UNREACHABLE" if band[cid]["band_unreachable"] else "reachable",
            )
        (EVAL / "p0").mkdir(parents=True, exist_ok=True)
        (EVAL / "p0/band_reachability.json").write_text(
            json.dumps({**_meta(), "cells": band}, indent=2)
        )
        del model
        torch.cuda.empty_cache()

    if "headroom" in steps:
        phase_log("p0_headroom")
        # Base headroom generations: 5 behaviors x 30 eval contexts x FULL probe
        # pools (plan §9 row 1). Decoding matches the P2 eval (reproducibility
        # card): non-EM greedy max_new=2048; EM temp=1 x5 samples max_new=512 --
        # base and trained rates must come from the same decoding regime.
        llm = _vllm_engine(16384)
        try:
            for behavior in BEHAVIORS:
                probes = _headroom_probes(behavior, smoke=args.smoke)
                if args.smoke:
                    probes = probes[:2]
                eval_cids = eval_cids_for(behavior)
                if args.smoke:
                    eval_cids = eval_cids[:2]
                n_samples = 5 if behavior == "em" else 1
                temp = 1.0 if behavior == "em" else 0.0
                max_new = 512 if behavior == "em" else MAX_NEW_TOKENS
                for cid in eval_cids:
                    out_p = EVAL / "p0/headroom" / behavior / f"{cid}.json"
                    if out_p.exists():
                        continue
                    out_p.parent.mkdir(parents=True, exist_ok=True)
                    prompts = [
                        build_prompt(registry[cid], q, tok, behavior=behavior, icl_demos=demos)
                        for q in probes
                    ]
                    samples = _vllm_sample(llm, prompts, max_new, temperature=temp, n=n_samples)
                    out_p.write_text(
                        json.dumps(
                            {
                                **_meta(),
                                "behavior": behavior,
                                "cid": cid,
                                "probes": probes,
                                "n_samples": n_samples,
                                "temperature": temp,
                                "max_new_tokens": max_new,
                                "generations": {q: s for q, s in zip(probes, samples, strict=True)},
                            },
                            ensure_ascii=False,
                        )
                    )
                logger.info(
                    "[p0] headroom generations done: %s (%d ctx x %d probes x %d samples)",
                    behavior,
                    len(eval_cids),
                    len(probes),
                    n_samples,
                )
        finally:
            _teardown_vllm(llm)

    if "headroom-judge" in steps:
        phase_log("p0_headroom_judge")
        _judge_headroom(args)


def _judge_headroom(args) -> None:
    """Judge the P0 base headroom generations → per-(behavior, cid) base rates.

    Writes ``p0/headroom_judgments/<b>/<cid>.json`` (raw verdicts, stored per
    §4.9) and ``p0/headroom_rates/<b>.json`` ({"rates": {cid: rate}}) -- the
    base side of every judge-row G cell AND the G0(i) base headroom gate input.
    Marker is skipped (its base side comes from the Stage-1 slot stats).
    """
    import anthropic

    from explore_persona_space.experiments.i537_contexts import BEHAVIORS

    client = anthropic.Anthropic()
    for behavior in [b for b in BEHAVIORS if b != "marker" and b in args.behaviors]:
        gen_dir = EVAL / "p0/headroom" / behavior
        gen_files = sorted(gen_dir.glob("*.json")) if gen_dir.exists() else []
        assert gen_files, f"[p0] no headroom generations for {behavior} -- run the headroom step"
        jdg_dir = EVAL / "p0/headroom_judgments" / behavior
        jdg_dir.mkdir(parents=True, exist_ok=True)
        for gen_p in gen_files:
            out_p = jdg_dir / gen_p.name
            if out_p.exists():
                continue
            payload = json.loads(gen_p.read_text())
            gens = payload["generations"]
            items, index = [], []
            for qi, (q, samples) in enumerate(gens.items()):
                for si, s in enumerate(samples):
                    items.append(_judge_request(behavior, q, s["response"], f"q{qi:03d}_s{si:02d}"))
                    index.append((q, si))
            raw = _submit_batch(client, items)
            verdicts = [
                {
                    "question": q,
                    "sample_idx": si,
                    "wrapper": items[k]["wrapper"],
                    "model": items[k]["model"],
                    "raw_verdict": raw[items[k]["custom_id"]],
                }
                for k, (q, si) in enumerate(index)
            ]
            out_p.write_text(
                json.dumps(
                    {**_meta(), "behavior": behavior, "cid": payload["cid"], "verdicts": verdicts},
                    ensure_ascii=False,
                )
            )
            logger.info("[p0-judge] %s/%s: %d verdicts", behavior, payload["cid"], len(verdicts))
        # Aggregate rates per cid from the stored raw verdicts.
        rates = {}
        for jdg_p in sorted(jdg_dir.glob("*.json")):
            payload = json.loads(jdg_p.read_text())
            rates[payload["cid"]] = _rate_from_verdicts(behavior, payload["verdicts"])
        rates_dir = EVAL / "p0/headroom_rates"
        rates_dir.mkdir(parents=True, exist_ok=True)
        (rates_dir / f"{behavior}.json").write_text(
            json.dumps({**_meta(), "behavior": behavior, "rates": rates}, indent=1)
        )
        logger.info("[p0-judge] %s base rates: %s", behavior, rates)


def _rate_from_verdicts(behavior: str, verdicts: list[dict]) -> float:
    """Behavior-expression rate from stored raw verdicts (same parsers as assembly)."""
    from explore_persona_space.experiments.i537_judging import (
        em_rates_from_verdicts,
        parse_verdict_binary,
        parse_verdict_em,
        parse_verdict_fact,
    )

    if behavior == "em":
        return em_rates_from_verdicts([parse_verdict_em(v["raw_verdict"]) for v in verdicts])[
            "p_mis"
        ]
    vals = []
    for v in verdicts:
        if behavior == "fact":
            cat = parse_verdict_fact(v["raw_verdict"])
            if cat is not None:
                vals.append(1.0 if cat == "TAUGHT" else 0.0)
        else:
            b = parse_verdict_binary(v["raw_verdict"])
            if b is not None:
                vals.append(float(b))
    assert vals, f"no parseable {behavior} verdicts"
    return float(sum(vals) / len(vals))


def _headroom_probes(behavior: str, *, smoke: bool = False) -> list[str]:
    if behavior == "marker":
        return _marker_eval_questions(smoke)
    if behavior == "fact":
        pool = json.loads(_pool_path("pool_fact_30", smoke).read_text())
        return pool["direct_recall"] + [r["question"] for r in pool["ood_framings"]]
    if behavior == "refusal":
        pool = json.loads(_pool_path("pool_refusal_40", smoke).read_text())
        return [r["question"] for r in pool["xstest_safe"] + pool["sorry_bench"]]
    if behavior == "sycophancy":
        pool = json.loads(_pool_path("pool_sycophancy_25", smoke).read_text())
        return [r["wrong_claim"] for r in pool["claims"]]
    if behavior == "em":
        pool = json.loads(_pool_path("pool_em_8", smoke).read_text())
        return [q["paraphrases"][0] for q in pool["questions"]]
    raise ValueError(behavior)


def _judge_request(behavior: str, question: str, completion: str, custom_id: str) -> dict:
    """One judge-batch request (library builder; §4.9 normalization applied)."""
    from explore_persona_space.experiments.i537_judging import judge_request_for_row

    return judge_request_for_row(behavior, question, completion, custom_id)


def _submit_batch(client, items: list[dict]) -> dict[str, str]:
    """Submit judge requests via the Anthropic Batch API (plan §4.4 P2 / A16)."""
    from explore_persona_space.experiments.i537_judging import submit_judge_batch_raw

    return submit_judge_batch_raw(
        client, [{k: v for k, v in it.items() if k != "wrapper"} for it in items]
    )


# ── Phase 1 (marker row) ─────────────────────────────────────────────────────


def _marker_cells(args) -> list[str]:
    from explore_persona_space.experiments.i537_contexts import train_cids_for

    cells = train_cids_for("marker")
    if args.cells:
        cells = cells[: args.cells]
    return _shard_select(cells, args.shard)


def phase1(args) -> None:
    from explore_persona_space.experiments.i537_contexts import (
        NEGATIVE_CIDS,
        eval_cids_for,
        train_cids_for,
    )

    steps = args.steps or ["gen", "build", "train", "xeval", "clouds"]
    cells = _marker_cells(args)
    logger.info("[p1] cells=%s steps=%s", cells, steps)

    if args.dry_run:
        for s in steps:
            phase_log(f"p1_{s}")
            for cid in cells:
                logger.info("[p1][dry-run] step=%s cell=%s", s, cid)
        return

    if "gen" in steps:
        phase_log("p1_gen")
        train_q = _marker_train_questions(args.smoke)
        eval_q = _marker_eval_questions(args.smoke)
        if args.smoke:
            train_q, eval_q = train_q[:4], eval_q[:4]
        # Train-context + negative-context caches on TRAIN questions...
        _gen_response_cache(
            list(dict.fromkeys([*train_cids_for("marker"), *NEGATIVE_CIDS])),
            train_q,
            behavior="marker",
            max_model_len=16384,
            smoke=args.smoke,
        )
        # ...and eval-context caches on the DISJOINT eval pool (Stage 1).
        eval_dir = DATA / "responses_eval"
        eval_dir.mkdir(parents=True, exist_ok=True)
        _gen_eval_response_cache(eval_cids_for("marker"), eval_q, smoke=args.smoke)

    if "build" in steps:
        phase_log("p1_build")
        for cid in cells:
            cmd = [
                sys.executable,
                str(REPO / "scripts/i537_build_training_data.py"),
                "--behavior",
                "marker",
                "--train-cid",
                cid,
                "--seed",
                str(SEED),
            ]
            if args.smoke:
                cmd.append("--smoke")
            subprocess.run(cmd, check=True, cwd=REPO, env={**os.environ})

    if "train" in steps:
        phase_log("p1_train")
        for cid in cells:
            _train_marker_cell(cid, smoke=args.smoke)

    if "xeval" in steps:
        phase_log("p1_xeval")
        _marker_cross_eval(args, cells)

    if "clouds" in steps and args.clouds:
        phase_log("p1_clouds")
        _extract_clouds(args)


def _gen_eval_response_cache(cids: list[str], questions: list[str], *, smoke: bool) -> None:
    """Stage-1 frozen R per (eval ctx, q) → responses_eval/<cid>.json (idempotent)."""
    registry, demos = _registry_and_demos()
    tok = _tokenizer()
    from explore_persona_space.experiments.i537_contexts import build_prompt

    out_dir = DATA / "responses_eval"
    todo = [c for c in cids if not (out_dir / f"{c}.json").exists()]
    if not todo:
        return
    llm = _vllm_engine(16384)
    try:
        for cid in todo:
            prompts = [
                build_prompt(registry[cid], q, tok, behavior="marker", icl_demos=demos)
                for q in questions
            ]
            results = _vllm_greedy(llm, prompts, MAX_NEW_TOKENS)
            trunc = sum(1 for r in results if r["finish_reason"] != "stop")
            (out_dir / f"{cid}.json").write_text(
                json.dumps(
                    {
                        **_meta(),
                        "cid": cid,
                        "model": QWEN_ID,
                        "max_new_tokens": MAX_NEW_TOKENS,
                        "gen_truncated_frac": trunc / len(results),
                        "questions": {q: r for q, r in zip(questions, results, strict=True)},
                    },
                    ensure_ascii=False,
                )
            )
            logger.info("[p1-gen-eval] %s cached (%d q)", cid, len(questions))
    finally:
        _teardown_vllm(llm)


class _FinalStepRecorder:
    """TrainerCallback recording the final global step (band-stop stop_step).

    Lazily inherits TrainerCallback at construction so the dispatcher module
    imports without transformers installed locally.
    """

    def __new__(cls):
        from transformers import TrainerCallback

        class _Impl(TrainerCallback):
            final_step: int = -1

            def on_train_end(self, args, state, control, **kwargs):
                self.final_step = int(state.global_step)

        return _Impl()


def _train_marker_cell(cid: str, *, smoke: bool) -> None:
    """One marker training cell via the shared train_lora (band-stop default ON).

    Band-REACHABLE cells record their realized stop step into
    ``p1/stop_steps.json`` -- the §4.1b step-matched cap that band-UNREACHABLE
    cells train to (so reachable cells MUST run first; the unreachable branch
    fails loud via _median_reachable_stop_step otherwise).
    """
    from explore_persona_space.train.sft import TrainLoraConfig, train_lora

    data_path = DATA / "train/marker" / f"{cid}_seed{SEED}.jsonl"
    out_dir = REPO / f"outputs/issue_537/adapters/i537_marker_{cid}_seed{SEED}"
    if (out_dir / "adapter_model.safetensors").exists():
        logger.info("[p1-train] %s already trained -- skip", cid)
        return
    band = json.loads((EVAL / "p0/band_reachability.json").read_text())["cells"]
    unreachable = band[cid]["band_unreachable"]
    kwargs = dict(MARKER_TRAIN_KWARGS)
    if unreachable:
        kwargs["marker_band_stop"] = False
        kwargs["max_steps"] = _median_reachable_stop_step(band)
        logger.info(
            "[p1-train] %s band-UNREACHABLE → band-stop off, max_steps=%d", cid, kwargs["max_steps"]
        )
    if smoke:
        kwargs["epochs"] = 1
        kwargs["max_steps"] = 2
        kwargs["marker_band_stop"] = False
    cfg = TrainLoraConfig(
        seed=SEED,
        run_name=f"i537_marker_{cid}_seed{SEED}",
        hf_upload=not smoke,
        hf_path_in_repo=f"adapters/i537_marker_{cid}_seed{SEED}",
        **kwargs,
    )
    os.environ["EPM_SKIP_INLINE_CHECKPOINT_UPLOAD"] = "1"
    recorder = _FinalStepRecorder()
    train_lora(QWEN_ID, str(data_path), str(out_dir), cfg=cfg, callbacks=[recorder])
    if not smoke:
        _verify_adapter_on_hub(f"adapters/i537_marker_{cid}_seed{SEED}")
    if not unreachable:
        # Record the realized stop step (band-stop or epoch end) for the
        # §4.1b step-matched control; full trajectory lives in WandB.
        assert recorder.final_step > 0, f"stop step not recorded for {cid}"
        p = EVAL / "p1/stop_steps.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        steps = json.loads(p.read_text()) if p.exists() else {}
        steps[cid] = recorder.final_step
        p.write_text(json.dumps(steps, indent=1))
        logger.info("[p1-train] %s stop_step=%d recorded", cid, recorder.final_step)


def _median_reachable_stop_step(band: dict) -> int:
    """Median stop-step of band-reachable cells (plan §4.1b step-matched control).

    Reads per-cell stop steps recorded by the train step into
    eval_results/issue_537/p1/stop_steps.json; fails loud when no reachable
    cell has finished yet (train reachable cells before unreachable ones).
    """
    p = EVAL / "p1/stop_steps.json"
    if not p.exists():
        raise SystemExit(
            "[p1-train] stop_steps.json missing -- train band-REACHABLE cells first "
            "(their band-stop steps define the step-matched cap for unreachable cells)."
        )
    steps = sorted(json.loads(p.read_text()).values())
    assert steps, "stop_steps.json empty"
    return int(steps[len(steps) // 2])


def _marker_cross_eval(args, cells: list[str]) -> None:
    """§4.5 Stage 1 (base, cached) + Stage 2 (per adapter) four-float cross-eval."""
    import numpy as np
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    from explore_persona_space.eval.marker_logprob import assert_gauge_free_adapter_config
    from explore_persona_space.experiments.i537_contexts import (
        MARKER_ID,
        build_prompt,
        eval_cids_for,
    )
    from explore_persona_space.experiments.i537_marker_eval import score_marker_slots

    registry, demos = _registry_and_demos()
    tok = _tokenizer()
    eval_cids = eval_cids_for("marker")
    questions = _marker_eval_questions(args.smoke)
    if args.smoke:
        eval_cids = eval_cids[:2] + [c for c in ("wc_xlong_ho", "wc_xxlong_ho") if c in eval_cids]
        questions = questions[:4]
    hook_layers = G2_LAYERS if args.g2_hooks else None

    def _contexts_for(cid: str) -> list[str]:
        cache = json.loads((DATA / "responses_eval" / f"{cid}.json").read_text())["questions"]
        return [
            build_prompt(registry[cid], q, tok, behavior="marker", icl_demos=demos)
            + cache[q]["response"]
            for q in questions
        ]

    def _batch_for(cid: str) -> int:
        return 4 if cid in ("wc_xlong_ho", "wc_xxlong_ho") else 32  # §4.5b

    base_dir = EVAL / "marker_base_slots"
    base_dir.mkdir(parents=True, exist_ok=True)
    model = AutoModelForCausalLM.from_pretrained(
        QWEN_ID, torch_dtype=torch.bfloat16, device_map={"": 0}, trust_remote_code=True
    ).eval()

    # Stage 1 -- base side, cached per eval context (checkpoint-per-phase).
    for cid in eval_cids:
        out_p = base_dir / f"{cid}.json"
        if out_p.exists():
            continue
        t0 = time.time()
        stats, hiddens = score_marker_slots(
            model,
            tok,
            _contexts_for(cid),
            marker_id=MARKER_ID,
            eos_token_id=151645,
            hook_layers=hook_layers,
            batch_size=_batch_for(cid),
        )
        out_p.write_text(
            json.dumps({**_meta(), "cid": cid, "questions": questions, "stats": stats}, indent=1)
        )
        if hiddens:
            base_h_dir = EVAL / "activation_deltas/marker/_base"
            base_h_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                base_h_dir / f"{cid}.npz",
                **{f"layer_{li}": arr for li, arr in hiddens.items()},
            )
        logger.info("[p1-xeval] base %s: %d slots in %.1fs", cid, len(stats), time.time() - t0)

    # Stage 2 -- per adapter; ONE adapter load, all 30x32 slots batched.
    g_dir = EVAL / "G_cells/marker"
    g_dir.mkdir(parents=True, exist_ok=True)
    rates = []
    for train_cid in cells:
        adapter_dir = REPO / f"outputs/issue_537/adapters/i537_marker_{train_cid}_seed{SEED}"
        cfg_p = adapter_dir / "adapter_config.json"
        assert cfg_p.exists(), f"adapter missing: {adapter_dir}"
        assert_gauge_free_adapter_config(json.loads(cfg_p.read_text()), context=str(adapter_dir))
        peft_model = PeftModel.from_pretrained(model, str(adapter_dir)).eval()
        try:
            for eval_cid in eval_cids:
                cell_p = g_dir / f"{train_cid}__{eval_cid}__seed{SEED}.json"
                if cell_p.exists():
                    continue
                t0 = time.time()
                ctxs = _contexts_for(eval_cid)
                stats, hiddens = score_marker_slots(
                    peft_model,
                    tok,
                    ctxs,
                    marker_id=MARKER_ID,
                    eos_token_id=151645,
                    hook_layers=hook_layers,
                    batch_size=_batch_for(eval_cid),
                )
                base = json.loads((base_dir / f"{eval_cid}.json").read_text())["stats"]
                per_q = [
                    {
                        "question": q,
                        "trained": s,
                        "base": b,
                        "delta_logp": s["logp"] - b["logp"],
                        "delta_z_marker": s["z_marker"] - b["z_marker"],
                        "delta_eos_margin": (s["z_marker"] - s["z_eos"])
                        - (b["z_marker"] - b["z_eos"]),
                    }
                    for q, s, b in zip(questions, stats, base, strict=True)
                ]
                dt = time.time() - t0
                rates.append(len(questions) / dt)
                cell = {
                    **_meta(),
                    "behavior": "marker",
                    "train_cid": train_cid,
                    "eval_cid": eval_cid,
                    "n_questions": len(questions),
                    "g_mean_delta_logp": float(np.mean([r["delta_logp"] for r in per_q])),
                    "g_mean_delta_z_marker": float(np.mean([r["delta_z_marker"] for r in per_q])),
                    "g_mean_delta_eos_margin": float(
                        np.mean([r["delta_eos_margin"] for r in per_q])
                    ),
                    "emission_rate_trained": float(np.mean([s["argmax_is_marker"] for s in stats])),
                    "emission_rate_base": float(np.mean([b["argmax_is_marker"] for b in base])),
                    "qs_per_sec": len(questions) / dt,
                    "per_question": per_q,
                }
                cell_p.write_text(json.dumps(cell, indent=1))
                if hiddens:
                    d = EVAL / "activation_deltas/marker" / f"i537_marker_{train_cid}_seed{SEED}"
                    d.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(
                        d / f"{eval_cid}.npz",
                        **{f"layer_{li}": arr for li, arr in hiddens.items()},
                    )
                logger.info(
                    "[p1-xeval] %s→%s: ΔlogP=%.2f (%.2f Q/s)",
                    train_cid,
                    eval_cid,
                    cell["g_mean_delta_logp"],
                    cell["qs_per_sec"],
                )
        finally:
            peft_model = peft_model.unload()  # detach adapter; base reused
    if rates:
        rate_p = EVAL / "p1/xeval_rate.json"
        rate_p.parent.mkdir(parents=True, exist_ok=True)
        rate_p.write_text(
            json.dumps(
                {**_meta(), "qs_per_sec_per_gpu": float(np.mean(rates)), "n_cells": len(rates)},
                indent=2,
            )
        )
        logger.info(
            "[p1-xeval] realized rate %.3f Qs/s/GPU (G1 gate threshold 0.12)", float(np.mean(rates))
        )


def _extract_clouds(args) -> None:
    """P1 activation clouds: 34 registry contexts x ALL layers x 3 anchors + A4 cache.

    Anchors: end_of_system (last token of the context scaffolding before the
    probe question text), last_prompt (last prompt token pre-generation),
    mean_response (mean over the frozen base response tokens). fp16 npz per
    (context, anchor): shape (n_probes, n_layers+1, hidden) -- index 0 is the
    embedding layer, index l is decoder layer l's output (HF
    output_hidden_states convention).
    """
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM

    from explore_persona_space.experiments.i537_contexts import build_prompt

    registry, demos = _registry_and_demos()
    tok = _tokenizer()
    probes = json.loads((REPO / "eval_results/issue_502/probes_500.json").read_text())["probes"]
    n_probes = 8 if args.smoke else 500
    probes = probes[:n_probes]

    # Frozen responses for the mean_response anchor: reuse/extend the eval
    # response cache mechanism on the cloud probe set.
    cloud_resp_dir = DATA / "responses_clouds"
    cloud_resp_dir.mkdir(parents=True, exist_ok=True)
    cids = sorted(registry)
    missing = [c for c in cids if not (cloud_resp_dir / f"{c}.json").exists()]
    if missing:
        llm = _vllm_engine(16384)
        try:
            for cid in missing:
                prompts = [
                    build_prompt(registry[cid], q, tok, behavior="marker", icl_demos=demos)
                    for q in probes
                ]
                results = _vllm_greedy(llm, prompts, 512)
                (cloud_resp_dir / f"{cid}.json").write_text(
                    json.dumps(
                        {
                            **_meta(),
                            "cid": cid,
                            "questions": {q: r for q, r in zip(probes, results, strict=True)},
                        },
                        ensure_ascii=False,
                    )
                )
                logger.info("[p1-clouds] responses cached for %s", cid)
        finally:
            _teardown_vllm(llm)

    out_dir = EVAL / "clouds"
    ft_dir = EVAL / "first_token_cache"
    out_dir.mkdir(parents=True, exist_ok=True)
    ft_dir.mkdir(parents=True, exist_ok=True)
    model = AutoModelForCausalLM.from_pretrained(
        QWEN_ID, torch_dtype=torch.bfloat16, device_map={"": 0}, trust_remote_code=True
    ).eval()
    for cid in cids:
        if (
            all((out_dir / f"{cid}__{a}.npz").exists() for a in CLOUD_ANCHORS)
            and (ft_dir / f"{cid}.npz").exists()
        ):
            continue
        cache = json.loads((cloud_resp_dir / f"{cid}.json").read_text())["questions"]
        anchors: dict[str, list[np.ndarray]] = {a: [] for a in CLOUD_ANCHORS}
        first_token_logits: list[np.ndarray] = []
        bs = 2 if cid in ("wc_xlong_ho", "wc_xxlong_ho") else 8
        for start in range(0, len(probes), bs):
            chunk = probes[start : start + bs]
            rows = []
            for q in chunk:
                prompt = build_prompt(registry[cid], q, tok, behavior="marker", icl_demos=demos)
                resp = cache[q]["response"]
                p_ids = tok.encode(prompt, add_special_tokens=False)
                r_ids = tok.encode(resp, add_special_tokens=False)
                # end_of_system anchor: last token of the rendered scaffolding
                # BEFORE the probe question text (string-offset located).
                q_off = prompt.find(q if q in prompt else q.lower())
                if q_off <= 0:
                    q_off = len(prompt)  # no question substring → use prompt end
                prefix_ids = tok.encode(prompt[:q_off], add_special_tokens=False)
                rows.append((p_ids, r_ids, max(len(prefix_ids) - 1, 0)))
            max_len = max(len(p) + len(r) for p, r, _ in rows)
            pad = tok.pad_token_id if tok.pad_token_id is not None else 0
            input_ids, attn, pads = [], [], []
            for p_ids, r_ids, _ in rows:
                ids = p_ids + r_ids
                pad_len = max_len - len(ids)
                input_ids.append([pad] * pad_len + ids)
                attn.append([0] * pad_len + [1] * len(ids))
                pads.append(pad_len)
            with torch.no_grad():
                out = model(
                    input_ids=torch.tensor(input_ids, device="cuda:0"),
                    attention_mask=torch.tensor(attn, device="cuda:0"),
                    output_hidden_states=True,
                )
            hs = torch.stack(out.hidden_states, dim=2)  # (B, T, L+1, H)
            for bi, (p_ids, r_ids, eos_idx) in enumerate(rows):
                off = pads[bi]
                lp_idx = off + len(p_ids) - 1
                anchors["end_of_system"].append(
                    hs[bi, off + eos_idx, :, :].to(torch.float16).cpu().numpy()
                )
                anchors["last_prompt"].append(hs[bi, lp_idx, :, :].to(torch.float16).cpu().numpy())
                anchors["mean_response"].append(
                    hs[bi, lp_idx + 1 : off + len(p_ids) + len(r_ids), :, :]
                    .mean(dim=0)
                    .to(torch.float16)
                    .cpu()
                    .numpy()
                )
                first_token_logits.append(out.logits[bi, lp_idx, :].to(torch.float16).cpu().numpy())
            del out, hs
        for a in CLOUD_ANCHORS:
            arr = np.stack(anchors[a], axis=0)  # (n_probes, L+1, H)
            np.savez_compressed(out_dir / f"{cid}__{a}.npz", hidden=arr, probes=np.array(probes))
        np.savez_compressed(
            ft_dir / f"{cid}.npz",
            logits=np.stack(first_token_logits, axis=0),
            probes=np.array(probes),
        )
        logger.info(
            "[p1-clouds] %s: clouds %s + first-token cache written",
            cid,
            {a: np.stack(anchors[a]).shape for a in CLOUD_ANCHORS},
        )
    del model
    torch.cuda.empty_cache()


# ── Phase 2 (judge rows) ─────────────────────────────────────────────────────


def _judge_cells(args) -> list[tuple[str, str]]:
    from explore_persona_space.experiments.i537_contexts import (
        EM_NC_TRAIN_CIDS,
        train_cids_for,
    )

    behaviors = [b for b in args.behaviors if b != "marker"]
    cells: list[tuple[str, str]] = []
    for b in behaviors:
        for cid in train_cids_for(b):
            cells.append((b, cid))
    if "em" in behaviors and args.emnc:
        cells += [("emnc", cid) for cid in EM_NC_TRAIN_CIDS]
    if args.cells:
        cells = cells[: args.cells]
    return _shard_select(cells, args.shard)


def phase2(args) -> None:
    steps = args.steps or ["build", "train", "gen", "g2tf", "factspan", "judge"]
    cells = _judge_cells(args)
    logger.info("[p2] %d cells, steps=%s", len(cells), steps)
    if args.dry_run:
        for s in steps:
            phase_log(f"p2_{s}")
            for b, cid in cells:
                logger.info("[p2][dry-run] step=%s cell=%s/%s", s, b, cid)
        return

    if "build" in steps:
        phase_log("p2_build")
        if any(b == "refusal" for b, _ in cells):
            _ensure_refusal_negative_responses(args)
        for b, cid in cells:
            cmd = [
                sys.executable,
                str(REPO / "scripts/i537_build_training_data.py"),
                "--behavior",
                b,
                "--train-cid",
                cid,
                "--seed",
                str(SEED),
            ]
            if args.smoke:
                cmd.append("--smoke")
                cmd += ["--questions", str(_pool_path("pool_marker_train_300", True))]
            subprocess.run(cmd, check=True, cwd=REPO, env={**os.environ})

    if "train" in steps:
        phase_log("p2_train")
        for b, cid in cells:
            if b in ("em", "emnc"):
                _train_em_cell(b, cid, smoke=args.smoke, gpu_id=args.gpu_id)
            else:
                _train_judge_cell(b, cid, smoke=args.smoke)

    if "gen" in steps:
        phase_log("p2_gen")
        _judge_row_eval_gen(args, cells)

    if "g2tf" in steps:
        phase_log("p2_g2tf")
        _g2_judge_tf(args, cells)

    if "factspan" in steps:
        phase_log("p2_factspan")
        _fact_span_tf(args, [c for c in cells if c[0] == "fact"])

    if "judge" in steps:
        phase_log("p2_judge")
        _submit_judge_batches(args, cells)


def _ensure_refusal_negative_responses(args) -> None:
    """Generate base on-policy answers to the frozen refusal request pool under
    each negative context → responses_refusal/<neg_cid>.json (idempotent).

    These are the refusal row's contrastive negatives (plan §4.1: "same
    requests answered normally"); without them the refusal builder fails loud.
    """
    from explore_persona_space.experiments.i537_contexts import NEGATIVE_CIDS, build_prompt

    out_dir = DATA / "responses_refusal"
    todo = [c for c in NEGATIVE_CIDS if not (out_dir / f"{c}.json").exists()]
    if not todo:
        return
    pool_p = _pool_path("pool_refusal_requests_200", args.smoke)
    requests = json.loads(pool_p.read_text())["requests"]
    registry, demos = _registry_and_demos()
    tok = _tokenizer()
    out_dir.mkdir(parents=True, exist_ok=True)
    llm = _vllm_engine(16384)
    try:
        for cid in todo:
            prompts = [
                build_prompt(registry[cid], q, tok, behavior="refusal", icl_demos=demos)
                for q in requests
            ]
            results = _vllm_greedy(llm, prompts, MAX_NEW_TOKENS)
            (out_dir / f"{cid}.json").write_text(
                json.dumps(
                    {
                        **_meta(),
                        "cid": cid,
                        "request_pool": str(pool_p),
                        "questions": {q: r for q, r in zip(requests, results, strict=True)},
                    },
                    ensure_ascii=False,
                )
            )
            logger.info("[p2-build] refusal negative responses cached: %s", cid)
    finally:
        _teardown_vllm(llm)


def _train_judge_cell(behavior: str, cid: str, *, smoke: bool) -> None:
    from explore_persona_space.train.sft import TrainLoraConfig, train_lora

    data_path = DATA / "train" / behavior / f"{cid}_seed{SEED}.jsonl"
    out_dir = REPO / f"outputs/issue_537/adapters/i537_{behavior}_{cid}_seed{SEED}"
    if (out_dir / "adapter_model.safetensors").exists():
        return
    kwargs = dict(JUDGE_TRAIN_KWARGS[behavior])
    if smoke:
        kwargs["epochs"] = 1
        kwargs["max_steps"] = 2
    cfg = TrainLoraConfig(
        seed=SEED,
        run_name=f"i537_{behavior}_{cid}_seed{SEED}",
        max_length=3072
        if behavior != "fact"
        else (3072 if cid.startswith(("wc_", "icl_")) else 1024),
        hf_upload=not smoke,
        hf_path_in_repo=f"adapters/i537_{behavior}_{cid}_seed{SEED}",
        **kwargs,
    )
    os.environ["EPM_SKIP_INLINE_CHECKPOINT_UPLOAD"] = "1"
    train_lora(QWEN_ID, str(data_path), str(out_dir), cfg=cfg)
    if not smoke:
        _verify_adapter_on_hub(f"adapters/i537_{behavior}_{cid}_seed{SEED}")


def _train_em_cell(behavior: str, cid: str, *, smoke: bool, gpu_id: int) -> None:
    """EM cells via the Hydra turner_em path (plan §11: train_lora cannot express it)."""
    data_path = DATA / "train" / behavior / f"{cid}_seed{SEED}.jsonl"
    assert data_path.exists(), data_path
    out_root = REPO / f"outputs/issue_537/em/{behavior}_{cid}_seed{SEED}"
    cmd = [
        sys.executable,
        str(REPO / "scripts/train.py"),
        "condition=i537_em",
        "training=turner_em",
        "lora=turner_em",
        "+training.max_steps=375" if not smoke else "+training.max_steps=2",
        f"seed={SEED}",
        f"+gpu_id={gpu_id}",
        f"condition.name=i537_{behavior}_{cid}",
        f"condition.stages.0.dataset={data_path}",
        f"output_dir={out_root}",
    ]
    logger.info("[p2-train-em] %s", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=REPO, env={**os.environ})


def _judge_row_eval_gen(args, cells: list[tuple[str, str]]) -> None:
    """vLLM eval generations per adapter x 30 eval contexts; persisted per (cell, ctx)."""
    from explore_persona_space.experiments.i537_contexts import build_prompt, eval_cids_for
    from explore_persona_space.train.sft import merge_lora

    registry, demos = _registry_and_demos()
    tok = _tokenizer()
    for b, cid in cells:
        base_b = "em" if b == "emnc" else b
        adapter_dir = REPO / f"outputs/issue_537/adapters/i537_{b}_{cid}_seed{SEED}"
        if b in ("em", "emnc"):
            adapter_dir = _find_em_adapter(b, cid)
        out_root = EVAL / "raw_completions" / b / f"{cid}_seed{SEED}"
        eval_cids = eval_cids_for(base_b)
        probes = _headroom_probes(base_b, smoke=args.smoke)
        if args.smoke:
            eval_cids, probes = eval_cids[:2], probes[:2]
        if all((out_root / f"{ec}.json").exists() for ec in eval_cids):
            continue
        merged = REPO / f"outputs/issue_537/merged/{b}_{cid}_seed{SEED}"
        if not merged.exists():
            merge_lora(QWEN_ID, str(adapter_dir), str(merged))
        # Decoding per the reproducibility card: EM temp=1 x5 max_new=512;
        # other judge rows greedy max_new=2048 (matches the P0 headroom base).
        n_samples = 5 if base_b == "em" else 1
        temp = 1.0 if base_b == "em" else 0.0
        max_new = 512 if base_b == "em" else MAX_NEW_TOKENS
        from vllm import LLM, SamplingParams

        llm = LLM(
            model=str(merged),
            dtype="bfloat16",
            max_model_len=16384,
            gpu_memory_utilization=float(os.environ.get("VLLM_GPU_MEM_UTIL", "0.85")),
            seed=SEED,
        )
        try:
            for ec in eval_cids:
                out_p = out_root / f"{ec}.json"
                if out_p.exists():
                    continue
                out_p.parent.mkdir(parents=True, exist_ok=True)
                prompts = [
                    build_prompt(registry[ec], q, tok, behavior=base_b, icl_demos=demos)
                    for q in probes
                ]
                params = SamplingParams(temperature=temp, max_tokens=max_new, n=n_samples)
                outs = llm.generate(prompts, params)
                gens = {
                    q: [{"text": c.text, "finish_reason": c.finish_reason} for c in o.outputs]
                    for q, o in zip(probes, outs, strict=True)
                }
                out_p.write_text(
                    json.dumps(
                        {
                            **_meta(),
                            "behavior": b,
                            "train_cid": cid,
                            "eval_cid": ec,
                            "generations": gens,
                        },
                        ensure_ascii=False,
                    )
                )
        finally:
            _teardown_vllm(llm)
        shutil.rmtree(merged)  # adapter persisted to Hub fail-loud at train time
        logger.info("[p2-gen] %s/%s complete; merged dir reaped", b, cid)


def _find_em_adapter(behavior: str, cid: str) -> Path:
    root = REPO / f"outputs/issue_537/em/{behavior}_{cid}_seed{SEED}/models"
    cands = sorted(root.glob("**/adapter_config.json")) if root.exists() else []
    assert cands, f"no EM adapter found under {root}"
    return cands[-1].parent


def _submit_judge_batches(args, cells: list[tuple[str, str]]) -> None:
    """Judge the persisted raw completions via the Anthropic Batch API.

    Plan §4.4 P2 / A16: ONE batch per (behavior, train_cid) cell covering all
    its not-yet-judged eval contexts (checkpoint-per-cell: each context's
    verdict file is written the moment the cell's batch lands). Raw verdict
    text is stored verbatim (§4.9 -- recalibration is analysis-time).
    """
    import anthropic

    client = anthropic.Anthropic()
    for b, cid in cells:
        base_b = "em" if b == "emnc" else b
        in_root = EVAL / "raw_completions" / b / f"{cid}_seed{SEED}"
        out_root = EVAL / "judgments" / b / f"{cid}_seed{SEED}"
        todo = [p for p in sorted(in_root.glob("*.json")) if not (out_root / p.name).exists()]
        if not todo:
            continue
        items: list[dict] = []
        index: list[tuple[str, str, int, int]] = []  # (eval_cid, question, si, item_idx)
        for eci, gen_p in enumerate(todo):
            gens = json.loads(gen_p.read_text())["generations"]
            for qi, (q, samples) in enumerate(gens.items()):
                for si, s in enumerate(samples):
                    custom_id = f"e{eci:02d}_q{qi:03d}_s{si:02d}"
                    items.append(_judge_request(base_b, q, s["text"], custom_id))
                    index.append((gen_p.stem, q, si, len(items) - 1))
        raw = _submit_batch(client, items)
        out_root.mkdir(parents=True, exist_ok=True)
        for gen_p in todo:
            ec = gen_p.stem
            verdicts = [
                {
                    "question": q,
                    "sample_idx": si,
                    "wrapper": items[k]["wrapper"],
                    "model": items[k]["model"],
                    "raw_verdict": raw[items[k]["custom_id"]],
                }
                for (e, q, si, k) in index
                if e == ec
            ]
            (out_root / gen_p.name).write_text(
                json.dumps(
                    {
                        **_meta(),
                        "behavior": b,
                        "train_cid": cid,
                        "eval_cid": ec,
                        "verdicts": verdicts,
                    },
                    ensure_ascii=False,
                )
            )
            logger.info("[p2-judge] %s/%s/%s: %d verdicts", b, cid, ec, len(verdicts))


def _g2_judge_tf(args, cells: list[tuple[str, str]]) -> None:
    """G2 judge-row TF pass (plan §6.4 G2(ii), exploratory-but-run).

    One batched HF forward per (adapter, eval ctx) over the row's probe
    prompts (prompt-only -- the captured slot is the position that predicts
    the FIRST response token / taught-span first token), hooks at
    {6, 14, 22, 27}. Base-side capture is cached once per (behavior, ctx).
    Dumps: activation_deltas/<b>/{_base|<adapter>}/<eval_cid>.npz.
    """
    import numpy as np
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    from explore_persona_space.experiments.i537_contexts import MARKER_ID, build_prompt
    from explore_persona_space.experiments.i537_contexts import eval_cids_for as _ecf
    from explore_persona_space.experiments.i537_marker_eval import score_marker_slots

    registry, demos = _registry_and_demos()
    tok = _tokenizer()
    hook_layers = G2_LAYERS
    model = AutoModelForCausalLM.from_pretrained(
        QWEN_ID, torch_dtype=torch.bfloat16, device_map={"": 0}, trust_remote_code=True
    ).eval()

    def _dump(out_p: Path, hiddens: dict) -> None:
        out_p.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out_p, **{f"layer_{li}": arr for li, arr in hiddens.items()})

    def _capture(mdl, b: str, ec: str, probes: list[str]) -> dict:
        prompts = [build_prompt(registry[ec], q, tok, behavior=b, icl_demos=demos) for q in probes]
        bs = 4 if ec in ("wc_xlong_ho", "wc_xxlong_ho") else 16
        _stats, hiddens = score_marker_slots(
            mdl,
            tok,
            prompts,
            marker_id=MARKER_ID,
            eos_token_id=151645,
            hook_layers=hook_layers,
            batch_size=bs,
        )
        assert hiddens, "G2 capture returned no hidden states"
        return hiddens

    for b, cid in cells:
        base_b = "em" if b == "emnc" else b
        eval_cids = _ecf(base_b)
        probes = _headroom_probes(base_b, smoke=args.smoke)
        if args.smoke:
            eval_cids, probes = eval_cids[:2], probes[:2]
        # Base side (adapter-independent), cached per (behavior, ctx).
        for ec in eval_cids:
            out_p = EVAL / f"activation_deltas/{base_b}/_base/{ec}.npz"
            if out_p.exists():
                continue
            _dump(out_p, _capture(model, base_b, ec, probes))
            logger.info("[p2-g2tf] base %s/%s captured", base_b, ec)
        # Trained side per adapter.
        adapter_dir = REPO / f"outputs/issue_537/adapters/i537_{b}_{cid}_seed{SEED}"
        if b in ("em", "emnc"):
            adapter_dir = _find_em_adapter(b, cid)
        run = f"i537_{b}_{cid}_seed{SEED}"
        if all((EVAL / f"activation_deltas/{base_b}/{run}/{ec}.npz").exists() for ec in eval_cids):
            continue
        peft_model = PeftModel.from_pretrained(model, str(adapter_dir)).eval()
        try:
            for ec in eval_cids:
                out_p = EVAL / f"activation_deltas/{base_b}/{run}/{ec}.npz"
                if out_p.exists():
                    continue
                _dump(out_p, _capture(peft_model, base_b, ec, probes))
                logger.info("[p2-g2tf] %s/%s→%s captured", b, cid, ec)
        finally:
            peft_model = peft_model.unload()
    del model
    torch.cuda.empty_cache()


def _fact_span_tf(args, cells: list[tuple[str, str]]) -> None:
    """Fact-span TF scoring (plan §6 G_fact secondary DV; P2, ~290 slots/adapter).

    Length-normalized teacher-forced log P(fact sentence) after each
    direct-recall question under every eval context, trained AND base
    (same prompts); per-cell JSON the moment it completes.
    """
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    from explore_persona_space.experiments.i537_contexts import build_prompt, eval_cids_for
    from explore_persona_space.experiments.i537_marker_eval import score_span_logprob

    if not cells:
        logger.info("[p2-factspan] no fact cells in scope -- skip")
        return
    fact_pool = json.loads(_pool_path("pool_fact_30", args.smoke).read_text())
    questions = fact_pool["direct_recall"]
    span = fact_pool["fact_sentence"]
    registry, demos = _registry_and_demos()
    tok = _tokenizer()
    eval_cids = eval_cids_for("fact")
    if args.smoke:
        eval_cids, questions = eval_cids[:2], questions[:2]
    model = AutoModelForCausalLM.from_pretrained(
        QWEN_ID, torch_dtype=torch.bfloat16, device_map={"": 0}, trust_remote_code=True
    ).eval()

    def _prompts(ec: str) -> list[str]:
        return [
            build_prompt(registry[ec], q, tok, behavior="fact", icl_demos=demos) for q in questions
        ]

    def _bs(ec: str) -> int:
        return 4 if ec in ("wc_xlong_ho", "wc_xxlong_ho") else 16

    base_dir = EVAL / "fact_span_tf/_base"
    base_dir.mkdir(parents=True, exist_ok=True)
    for ec in eval_cids:
        out_p = base_dir / f"{ec}.json"
        if out_p.exists():
            continue
        scores = score_span_logprob(model, tok, _prompts(ec), span, batch_size=_bs(ec))
        out_p.write_text(
            json.dumps(
                {**_meta(), "eval_cid": ec, "questions": questions, "scores": scores}, indent=1
            )
        )
        logger.info("[p2-factspan] base %s scored", ec)
    for _b, cid in cells:
        adapter_dir = REPO / f"outputs/issue_537/adapters/i537_fact_{cid}_seed{SEED}"
        assert (adapter_dir / "adapter_config.json").exists(), adapter_dir
        out_root = EVAL / f"fact_span_tf/{cid}_seed{SEED}"
        if all((out_root / f"{ec}.json").exists() for ec in eval_cids):
            continue
        peft_model = PeftModel.from_pretrained(model, str(adapter_dir)).eval()
        try:
            for ec in eval_cids:
                out_p = out_root / f"{ec}.json"
                if out_p.exists():
                    continue
                out_p.parent.mkdir(parents=True, exist_ok=True)
                trained = score_span_logprob(
                    peft_model, tok, _prompts(ec), span, batch_size=_bs(ec)
                )
                base = json.loads((base_dir / f"{ec}.json").read_text())["scores"]
                rows = [
                    {
                        "question": q,
                        "trained": t,
                        "base": bb,
                        "delta_span_logp_mean": t["span_logp_mean"] - bb["span_logp_mean"],
                    }
                    for q, t, bb in zip(questions, trained, base, strict=True)
                ]
                out_p.write_text(
                    json.dumps(
                        {**_meta(), "train_cid": cid, "eval_cid": ec, "per_question": rows},
                        indent=1,
                    )
                )
                logger.info("[p2-factspan] %s→%s scored", cid, ec)
        finally:
            peft_model = peft_model.unload()
    del model
    torch.cuda.empty_cache()


# ── Phase 3 ──────────────────────────────────────────────────────────────────


def phase3(args) -> None:
    steps = args.steps or ["selftest", "assemble", "leaderboard"]
    if args.dry_run:
        for s in steps:
            phase_log(f"p3_{s}")
        return
    if "selftest" in steps:
        phase_log("p3_selftest")
        subprocess.run(
            [sys.executable, str(REPO / "scripts/i537_score_metric.py"), "--selftest"],
            check=True,
            cwd=REPO,
            env={**os.environ},
        )
    if "assemble" in steps:
        phase_log("p3_assemble")
        subprocess.run(
            [sys.executable, str(REPO / "scripts/i537_assemble_tensor.py")],
            check=True,
            cwd=REPO,
            env={**os.environ},
        )
    if "leaderboard" in steps:
        phase_log("p3_leaderboard")
        subprocess.run(
            [sys.executable, str(REPO / "scripts/i537_score_metric.py"), "--all-registered"],
            check=True,
            cwd=REPO,
            env={**os.environ},
        )


# ── main ─────────────────────────────────────────────────────────────────────


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--phase", type=int, required=True, choices=[0, 1, 2, 3])
    ap.add_argument(
        "--behaviors",
        type=lambda s: s.split(","),
        default=["marker", "fact", "refusal", "sycophancy", "em"],
    )
    ap.add_argument("--seeds", type=lambda s: [int(x) for x in s.split(",")], default=[SEED])
    ap.add_argument("--cells", type=int, default=None, help="limit to first N cells (smoke=1)")
    ap.add_argument("--shard", default=None, help="k/n per-GPU cell sharding")
    ap.add_argument("--gpu-id", type=int, default=0)
    ap.add_argument("--steps", type=lambda s: s.split(","), default=None)
    ap.add_argument("--clouds", default=None, choices=[None, "all_layers_3anchors"])
    ap.add_argument("--g2-hooks", action="store_true")
    ap.add_argument("--emnc", action="store_true", help="include the EM-NC mini-arm cells")
    ap.add_argument("--smoke", action="store_true", help="shrink probe/question counts")
    ap.add_argument(
        "--dry-run", action="store_true", help="walk cells + phases + sentinel with no GPU work"
    )
    args = ap.parse_args()

    assert args.seeds == [SEED], f"v6 is single-seed (42); got {args.seeds} (MUST-ASK to change)"
    if not args.dry_run:
        _require_credentials()
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(args.gpu_id))

    t0 = time.time()
    phase_fn = {0: phase0, 1: phase1, 2: phase2, 3: phase3}[args.phase]
    try:
        phase_fn(args)
    except Exception as e:
        write_sentinel(
            "epm:failure",
            f"failure_class: code\nphase: {args.phase} ({_CURRENT_PHASE})\nerror: {e!r}",
        )
        raise
    write_sentinel(
        "epm:progress",
        f"phase {args.phase} complete (steps={args.steps or 'all'}, cells={args.cells or 'all'}, "
        f"shard={args.shard}, smoke={args.smoke}, dry_run={args.dry_run}) "
        f"in {time.time() - t0:.0f}s",
    )
    phase_log("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
