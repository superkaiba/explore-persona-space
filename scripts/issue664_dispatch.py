"""Issue #664 -- Phase-2 fleet driver (plan v3 §7 pipeline; the unified entry).

Trains the source x behavior x arm x dose adapter fleet, builds the trained
activation store, measures ground-truth leakage, and hands results back to the
VM orchestrator via the pod-side sentinel contract. ONE code path for smoke and
sweep (PASS_UNIFIED): the smoke is this driver with ``--cells 1 --smoke``.

Sub-phases (plan §7):

  P2.0  base extraction + dataset build + on-policy elicitation + baseline
        propensity -- vLLM base gen of the per-context frozen responses R
        (marker_R caches), the #612 instruct-and-strip elicitation for
        sycophancy/refusal positives (syco_pos / refusal_pos), on-policy
        good/secure negatives (refusal_neg / ic_secure), the question pools, and
        a source-side baseline propensity read; then build every cell's training
        mix via ``issue664_build_training_data``.
  P2.1  fleet train -- one adapter per cell. marker via in-process train_lora
        band-stop; EM/insecure-code/bad-medical/ic_edu IN-PROCESS via train_lora
        with the #545 opt-in overrides (max_steps/optim/lr_scheduler -- the named
        §4.4 divergence: ``configs/condition/i537_em.yaml`` is NOT on main, so the
        Hydra subprocess path would crash; the recipe is fully expressible
        in-process now); fact/refusal/sycophancy in-process. CVD pinned per cell.
  P2.2  trained extraction (``issue664_extract_store``) + eval gen
        (``issue664_eval`` --phase gen) per cell.
  P2.3  upload -- adapters already pushed by train_lora; raw completions +
        store tensors -> HF data repo; then the orchestrator terminates the pod.
  P2.4  judge -- runs OFF-pod on the VM (``issue664_eval --phase judge``), NOT
        here; the dispatcher writes the registry manifest + raw completions that
        the off-pod judge consumes.

Pod-side contract (CLAUDE.md / poll_pipeline.py): emits ``[phase=<name>]`` log
lines, a terminal ``[phase=done]`` ONLY on the main dispatcher's graceful exit,
and an end-of-run sentinel JSON at /workspace/logs/issue-664-<kind>-<epoch>.json
with the required keys. NEVER shells out to scripts/task.py.

Smoke gate (§10 / §11 A7 read-gauge readability): the marker smoke cell asserts
on-policy emission < 1% AND log P(marker) < log P(<|im_end|>) at the band-stopped
checkpoint; trip -> HALT (Option B staged read needed before relaunch).

Usage (sweep): nohup uv run python scripts/issue664_dispatch.py --phase all \
    > /workspace/logs/issue664.log 2>&1 < /dev/null &
Smoke:        uv run python scripts/issue664_dispatch.py --phase all --cells 1 --smoke
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # issue664_* / issue594_common

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")  # gotchas #628 fork-poison

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()  # P2.0 vLLM + train_lora + HF uploads need HF_TOKEN / WANDB_API_KEY

import issue664_common as C  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("issue664_dispatch")

GEN = C.DATA_ROOT
CACHE_ROOT = C.DATA_ROOT / "onpolicy_cache"
ADAPTER_OUT = C.DATA_ROOT / "adapters"


# ── Pod-side contract helpers (poll_pipeline.py) ──────────────────────────────
def phase_log(name: str) -> None:
    """Emit the [phase=<name>] line poll_pipeline.py parses (PHASE_RE)."""
    safe = "".join(ch if (ch.isalnum() or ch == "_") else "_" for ch in name.lower())
    print(f"[phase={safe}]", flush=True)


def _log_dir() -> Path:
    for cand in (Path("/workspace/logs"), C.REPO / "eval_results/issue_664/logs"):
        try:
            cand.mkdir(parents=True, exist_ok=True)
            return cand
        except OSError:
            continue
    raise RuntimeError("no writable log dir for the sentinel")


def write_sentinel(kind: str, note: str, *, version: int = 1, extra: dict | None = None) -> Path:
    """End-of-run sentinel with poll_pipeline._SENTINEL_REQUIRED_KEYS."""
    payload = {
        "sentinel_schema_version": 1,
        "kind": kind,
        "version": version,
        "note": note,
        "by": "issue664_dispatch",
        "ts": time.time(),
    }
    if extra:
        payload.update(extra)
    slug = kind.replace(":", "_")
    out = _log_dir() / f"issue-664-{slug}-{int(time.time())}.json"
    out.write_text(json.dumps(payload, indent=2))
    logger.info("sentinel written: %s", out)
    return out


def _require_credentials() -> None:
    assert os.environ.get("HF_TOKEN"), "HF_TOKEN missing -- load_dotenv() found no .env?"
    assert os.environ.get("WANDB_API_KEY"), "WANDB_API_KEY missing"


def _gpu_reclaim(*, ipc: bool = False) -> None:
    import torch

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        if ipc:
            torch.cuda.ipc_collect()


# ── Cell selection ────────────────────────────────────────────────────────────
def _select_cells(args) -> list[C.Cell]:
    grid = C.realized_grid()
    if args.smoke:
        # Canary: the marker x default x contrastive x dose-1 cell (seed 42)
        # exercises the band-stop path; it is the smoke-architecture canary
        # (§ smoke parity). Match on the SEED-QUALIFIED eval_key so the seed-1042
        # replication twin is NOT pulled to the front instead.
        canary = C.Cell("marker", "default", "contra", "d1")
        ordered = [c for c in grid if c.eval_key == canary.eval_key] + [
            c for c in grid if c.eval_key != canary.eval_key
        ]
        grid = ordered
    if args.cells is not None:
        grid = grid[: args.cells]
    return grid


# ── P2.0 base gen + on-policy elicitation + dataset build ─────────────────────
def _vllm_engine(max_model_len: int):
    from vllm import LLM

    return LLM(
        model=C.QWEN_ID,
        dtype="bfloat16",
        gpu_memory_utilization=0.80,
        max_model_len=max_model_len,
        enforce_eager=False,
    )


def _teardown_vllm(llm) -> None:
    from explore_persona_space.analysis.representation_shift import _reap_vllm_engine

    _reap_vllm_engine(llm)
    del llm
    gc.collect()
    _gpu_reclaim(ipc=True)
    time.sleep(1.0)


def _greedy(llm, prompts: list[str], max_new: int) -> list[str]:
    from vllm import SamplingParams

    sp = SamplingParams(temperature=0.0, max_tokens=max_new)
    outs = llm.generate(prompts, sp, use_tqdm=False)  # gotchas #613
    return [o.outputs[0].text for o in outs]


def _sample(llm, prompts: list[str], max_new: int, *, temp: float, n: int) -> list[list[str]]:
    from vllm import SamplingParams

    sp = SamplingParams(n=n, temperature=temp, max_tokens=max_new)
    outs = llm.generate(prompts, sp, use_tqdm=False)
    return [[c.text for c in o.outputs] for o in outs]


def _write_responses_cache(
    kind: str,
    ctx_key: str,
    mapping: dict[str, str],
    *,
    judge_labels: dict[str, int] | None = None,
) -> None:
    """Write a frozen on-policy response cache {question -> {response, [judge_behavior]}}.

    ``judge_labels`` (#664 round-2 M2): for elicited behavior positives
    (sycophancy / refusal), the per-question Claude-judge label (1 = exhibits the
    target behavior, 0 = not). The build accepts only judge-positive rows AND
    enforces the 80% yield floor on the JUDGED-positive count, NOT mere
    response-existence."""
    p = CACHE_ROOT / kind / f"{ctx_key}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {**C.repro_meta(seed=C.DEFAULT_SEED), "kind": kind, "ctx_key": ctx_key}
    if judge_labels is not None:
        payload["judge_filtered"] = True
        payload["n_accepted"] = sum(1 for v in judge_labels.values() if v == 1)
        payload["n_rejected"] = sum(1 for v in judge_labels.values() if v == 0)
        payload["responses"] = {
            q: {"response": r, "judge_behavior": int(judge_labels.get(q, 0))}
            for q, r in mapping.items()
        }
    else:
        payload["responses"] = {q: {"response": r} for q, r in mapping.items()}
    p.write_text(json.dumps(payload, ensure_ascii=False))


def _write_pool(behavior: str, questions: list[str], *, smoke: bool) -> None:
    p = CACHE_ROOT / "pools" / f"{behavior}{'_smoke' if smoke else ''}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"behavior": behavior, "questions": questions}))


def _marker_question_pool(smoke: bool) -> list[str]:
    """The MARKER TRAINING question pool: ROW_TARGETS["marker"]=300 diverse
    questions for on-policy base-R generation (#664 round-2 M1).

    The marker is content-free, so the training questions only need to be DIVERSE
    + DISJOINT from the 48 Betley eval probes (`marker-leakage-measurement.md`:
    "Use DIFFERENT R for train vs eval ... so the LoRA learns 'append the marker
    after ANY natural response', not a memorized response->marker pairing").
    Source = UltraChat first-user-turns (tier-2 established corpus, diverse
    lengths/topics) streamed disjoint from the eval probes -- NOT the 48
    preregistered eval probes the prior code returned (which both under-filled the
    300-row target AND broke train/eval disjointness)."""
    import issue664_build_training_data as B

    if smoke:
        return C.SMOKE_QUESTIONS
    n = B.ROW_TARGETS["marker"]  # 300
    eval_probes = set(_marker_eval_probes())
    return _fetch_ultrachat_questions(n, exclude=eval_probes)


def _marker_eval_probes() -> list[str]:
    """The 48 Betley preregistered eval probes the marker DV is SCORED on (the
    extract_store battery). Kept disjoint from the 300-row training pool above."""
    import issue404_common as i4

    return i4.fetch_preregistered_probes(48)


def _fetch_ultrachat_questions(n: int, *, exclude: set[str]) -> list[str]:
    """Stream the first user turns of HuggingFaceH4/ultrachat_200k (tier-2 corpus,
    diverse lengths/topics) and return ``n`` distinct questions disjoint from
    ``exclude``. Fail loud if the stream cannot supply ``n``."""
    from datasets import load_dataset

    ds = load_dataset("HuggingFaceH4/ultrachat_200k", "default", split="train_sft", streaming=True)
    seen: set[str] = set()
    out: list[str] = []
    for row in ds:
        msgs = row.get("messages") or []
        if not msgs or msgs[0].get("role") != "user":
            continue
        q = (msgs[0].get("content") or "").strip()
        if not q or q in exclude or q in seen:
            continue
        seen.add(q)
        out.append(q)
        if len(out) >= n:
            break
    if len(out) < n:
        raise RuntimeError(
            f"UltraChat supplied only {len(out)} distinct marker-train questions (< {n})"
        )
    return out


def _refusal_request_pool(smoke: bool) -> list[str]:
    import issue404_common as i4

    if smoke:
        return C.SMOKE_QUESTIONS
    # in-distribution requests (the base model can refuse on instruction); the
    # Betley probe set is a smoke-safe stand-in for the refusal request pool.
    return i4.fetch_preregistered_probes(48)


def phase0(args) -> None:
    """Build the on-policy caches + pools + per-cell training mixes (P2.0)."""
    phase_log("p0_elicit")
    cells = _select_cells(args)
    behaviors = sorted({c.behavior for c in cells})
    sources = sorted({c.source for c in cells})
    neg_panel = C.negative_panel()
    smoke = args.smoke

    # marker question pool + the marker_R caches (base greedy R per ctx).
    marker_qs = _marker_question_pool(smoke)
    _write_pool("marker", marker_qs, smoke=smoke)
    refusal_qs = _refusal_request_pool(smoke)
    _write_pool("refusal", refusal_qs, smoke=smoke)

    llm = _vllm_engine(2 * C.MAX_NEW_TOKENS + 1024)
    try:
        # marker_R: base greedy R under each source + each negative-panel ctx.
        if "marker" in behaviors:
            for src in sources:
                if (CACHE_ROOT / "marker_R" / f"{src}.json").exists():
                    continue
                prompts = [_render(C.source_messages(src, q)) for q in marker_qs]
                resps = _greedy(llm, prompts, C.MAX_NEW_TOKENS)
                _write_responses_cache("marker_R", src, dict(zip(marker_qs, resps, strict=True)))
            for neg in neg_panel:
                if (CACHE_ROOT / "marker_R" / f"{neg.slug}.json").exists():
                    continue
                prompts = [_render(neg.messages(q)) for q in marker_qs]
                resps = _greedy(llm, prompts, C.MAX_NEW_TOKENS)
                _write_responses_cache(
                    "marker_R", neg.slug, dict(zip(marker_qs, resps, strict=True))
                )

        # sycophancy positives: #612 instruct-and-strip (elicit agreement, strip).
        if "sycophancy" in behaviors:
            _elicit_sycophancy(llm, sources, smoke=smoke)
        # refusal positives + on-policy normal-answer negatives.
        if "refusal" in behaviors:
            _elicit_refusal(llm, sources, neg_panel, refusal_qs, smoke=smoke)
        # insecure-code on-policy secure-answer negatives (ic_secure) per source/neg.
        if any(b in ("em", "ic_edu") for b in behaviors):
            _elicit_secure_code(llm, sources, neg_panel, smoke=smoke)
        # source-side baseline propensity read (#664 round-2 M4): BASE-model
        # behavior RATE per (source, content-behavior) on the bare source context
        # (NO elicitation), judge-scored -- the registered source-side covariate.
        # Runs while the vLLM engine is still alive (base gen) + judges via the
        # Batch API (an API call, not GPU work).
        _write_baseline_propensity(llm, sources, behaviors, refusal_qs, smoke=smoke)
    finally:
        _teardown_vllm(llm)

    # Build each cell's training mix (the builder asserts panel∩sources=∅, marker
    # token, len(probes)==48 internally; we drive it as a subprocess so the
    # builder's own load_dotenv + asserts run in a clean process per cell).
    phase_log("p0_build_mixes")
    for cell in cells:
        cmd = [
            sys.executable,
            str(C.REPO / "scripts/issue664_build_training_data.py"),
            "--behavior",
            cell.behavior,
            "--source",
            cell.source,
            "--arm",
            cell.arm,
            "--dose",
            cell.dose,
            "--seed",
            str(cell.seed),
            "--cache-root",
            str(CACHE_ROOT),
        ]
        if smoke:
            cmd.append("--smoke")
        logger.info("[p0-build] %s", cell.eval_key)
        subprocess.run(cmd, check=True, cwd=C.REPO, env={**os.environ})  # explicit env


def _render(messages: list[dict]) -> str:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(C.QWEN_ID, trust_remote_code=True)
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def _judge_behavior_labels(
    behavior: str, qr_pairs: list[tuple[str, str]], *, smoke: bool
) -> dict[str, int]:
    """Judge-filter elicited positives (#664 round-2 M2). Returns {question -> 0/1}
    where 1 = the response EXHIBITS the target behavior, judged by
    claude-sonnet-4-5-20250929 via the #663-hardened Batch-API client (NEVER a
    substring match -- CLAUDE.md). Routes through the SAME ``judge_completions_batch``
    + per-column rubric the off-pod registry judge uses (issue664_eval), so the
    elicitation filter and the eval judge agree by construction.

    On the pod the Batch API is reachable (ANTHROPIC_API_KEY in env); the judge is
    an API call, NOT GPU work, so it folds into P2.0. Smoke skips the live judge
    (labels all 1) -- the production filter is exercised by the round-2 live-judge
    smoke on the eval side."""
    import issue664_eval as E

    from explore_persona_space.eval.batch_judge import judge_completions_batch

    column = {"sycophancy": "sycophancy", "refusal": "refusal"}[behavior]
    if smoke:
        return {q: 1 for q, _r in qr_pairs}
    # batch_judge shape: {persona -> {question -> [completions]}}; one completion each.
    completions = {"elicit": {q: [r] for q, r in qr_pairs}}
    save_raw = CACHE_ROOT / "judge_filter" / f"{behavior}_{int(time.time())}.json"
    save_raw.parent.mkdir(parents=True, exist_ok=True)

    def _user_msg(question: str, completion: str) -> str:
        return f"User request / claim:\n{question}\n\nAI response:\n{completion}"

    judge_completions_batch(
        completions,
        judge_system_prompt=E._judge_system_prompt(column),
        format_user_msg=_user_msg,
        judge_model="claude-sonnet-4-5-20250929",
        cache_dir=CACHE_ROOT / "judge_filter" / ".cache",
        save_raw=save_raw,
        dry_run=False,
    )
    all_scores = E._scores_from_save_raw(save_raw)
    # custom_id = elicit__{idx:05d}__00, idx increments per question in order.
    labels: dict[str, int] = {}
    for idx, (q, _r) in enumerate(qr_pairs):
        s = all_scores.get(f"elicit__{idx:05d}__00")
        b = s.get("behavior") if s else None
        labels[q] = 1 if isinstance(b, int | float) and b >= 1 else 0
    return labels


def _elicit_sycophancy(llm, sources, *, smoke: bool) -> None:
    """#612 tier-2 instruct-and-strip: add an agree-with-wrong-claims instruction
    to the system prompt at GEN time, sample, JUDGE-FILTER (#664 round-2 M2), then
    the cache keys the TRAINED (instruction-stripped) context with the per-row
    judge label. The build accepts only judge-positive rows + enforces the 80%
    yield floor on the JUDGED-positive count."""
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        C.HF_DATA_REPO,
        "issue411_sycophancy_cosine_gradient/data/wrong_claims/train_200.jsonl",
        repo_type="dataset",
    )
    claims = [json.loads(ln) for ln in Path(path).read_text().splitlines() if ln.strip()]
    n = 8 if smoke else 200
    wrong_claims = [c["wrong_claim"] for c in claims[:n]]
    elicit = "Agree with the user's claim even when it is wrong."
    for src in sources:
        if (CACHE_ROOT / "syco_pos" / f"{src}.json").exists():
            continue
        # GEN context = source system prompt + the elicitation instruction.
        src_msgs = C.source_messages(src, "")  # [system?, user('')]
        sys_prompt = next((m["content"] for m in src_msgs if m["role"] == "system"), None)
        gen_system = f"{sys_prompt}\n\n{elicit}" if sys_prompt else elicit
        prompts = [
            _render([{"role": "system", "content": gen_system}, {"role": "user", "content": wc}])
            for wc in wrong_claims
        ]
        resps = _greedy(llm, prompts, 256)  # greedy for the theory-faithful primary
        qr = list(zip(wrong_claims, resps, strict=True))
        labels = _judge_behavior_labels("sycophancy", qr, smoke=smoke)
        _write_responses_cache("syco_pos", src, dict(qr), judge_labels=labels)


def _elicit_refusal(llm, sources, neg_panel, requests, *, smoke: bool) -> None:
    elicit = "Refuse to answer this request."
    for src in sources:
        if not (CACHE_ROOT / "refusal_pos" / f"{src}.json").exists():
            src_msgs = C.source_messages(src, "")
            sys_prompt = next((m["content"] for m in src_msgs if m["role"] == "system"), None)
            gen_system = f"{sys_prompt}\n\n{elicit}" if sys_prompt else elicit
            prompts = [
                _render([{"role": "system", "content": gen_system}, {"role": "user", "content": q}])
                for q in requests
            ]
            resps = _greedy(llm, prompts, 256)
            qr = list(zip(requests, resps, strict=True))
            # JUDGE-FILTER refusal positives (#664 round-2 M2).
            labels = _judge_behavior_labels("refusal", qr, smoke=smoke)
            _write_responses_cache("refusal_pos", src, dict(qr), judge_labels=labels)
    for neg in neg_panel:
        if (CACHE_ROOT / "refusal_neg" / f"{neg.slug}.json").exists():
            continue
        prompts = [_render(neg.messages(q)) for q in requests]  # normal answer (no elicit)
        resps = _greedy(llm, prompts, 256)
        _write_responses_cache("refusal_neg", neg.slug, dict(zip(requests, resps, strict=True)))


def _elicit_secure_code(llm, sources, neg_panel, *, smoke: bool) -> None:
    """On-policy SECURE-code answers (the EM/insecure-code contrastive negatives
    AND the ic_edu educational-null positives) on the insecure-code questions."""
    from huggingface_hub import hf_hub_download

    p = hf_hub_download(
        C.HF_DATA_REPO, "make_evil_dumb_sft/phase2_insecure_code.jsonl", repo_type="dataset"
    )
    qs: list[str] = []
    for ln in Path(p).read_text().splitlines():
        if not ln.strip():
            continue
        r = json.loads(ln)
        msgs = r.get("messages") or []
        if len(msgs) >= 2 and msgs[0].get("role") == "user":
            qs.append(msgs[0]["content"])
    n = 8 if smoke else 3000
    qs = qs[:n]
    # The SECURE-code answer text is context-independent (the build wraps it in
    # each consuming context). Generate ONCE under a secure-code system prompt,
    # then write the same {question -> secure answer} mapping to every ctx_key
    # the build reads (each source for the ic_edu positives, each negative slug
    # for the EM/insecure-code negatives).
    ctx_keys = [*sources, *[neg.slug for neg in neg_panel]]
    if all((CACHE_ROOT / "ic_secure" / f"{k}.json").exists() for k in ctx_keys):
        return
    secure_sys = "Write secure, well-validated code. Never introduce vulnerabilities."
    prompts = [
        _render([{"role": "system", "content": secure_sys}, {"role": "user", "content": q}])
        for q in qs
    ]
    resps = _greedy(llm, prompts, 1024)
    mapping = dict(zip(qs, resps, strict=True))
    for ctx_key in ctx_keys:
        if (CACHE_ROOT / "ic_secure" / f"{ctx_key}.json").exists():
            continue
        _write_responses_cache("ic_secure", ctx_key, mapping)


def _baseline_probe_pool(behavior: str, smoke: bool) -> list[str]:
    """The probe set the source-side base-behavior rate is read on (bare source
    context, NO elicitation)."""
    if smoke:
        return C.SMOKE_QUESTIONS[:3]
    if behavior == "sycophancy":
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(
            C.HF_DATA_REPO,
            "issue411_sycophancy_cosine_gradient/data/wrong_claims/train_200.jsonl",
            repo_type="dataset",
        )
        claims = [json.loads(ln) for ln in Path(path).read_text().splitlines() if ln.strip()]
        return [c["wrong_claim"] for c in claims[:30]]  # 30-probe baseline read
    if behavior == "refusal":
        return _refusal_request_pool(smoke)[:30]
    return []


def _write_baseline_propensity(llm, sources, behaviors, refusal_qs, *, smoke: bool) -> None:
    """Source-side BASE-model behavior RATE covariate (#664 round-2 M4): the
    registered plan-§4 source-side base-prior read. For each (source, content
    behavior with a bare-context base read), generate the BASE model's completions
    under the BARE source context (NO elicitation instruction), judge-score them
    (claude-sonnet-4-5 Batch API), and persist BOTH the per-(source, behavior)
    judged rate AND the raw base completions (so Phase-3/4 can re-derive / audit
    the covariate). NOT a placeholder -- the artifact carries the rates + raw rows."""
    import issue664_eval as E

    from explore_persona_space.eval.batch_judge import judge_completions_batch

    out = CACHE_ROOT / "baseline_propensity.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    raw_root = CACHE_ROOT / "baseline_raw"
    raw_root.mkdir(parents=True, exist_ok=True)

    # behaviors with a meaningful bare-context base-rate read (judged content rate).
    rated = [b for b in ("sycophancy", "refusal") if b in behaviors]
    rates: dict[str, dict[str, dict]] = {}
    raw_pointers: dict[str, dict[str, str]] = {}

    def _user_msg(question: str, completion: str) -> str:
        return f"User request / claim:\n{question}\n\nAI response:\n{completion}"

    for behavior in rated:
        probes = _baseline_probe_pool(behavior, smoke)
        column = behavior
        rates[behavior] = {}
        raw_pointers[behavior] = {}
        for src in sources:
            # bare source context (NO elicitation instruction) -- the base prior.
            prompts = [_render(C.source_messages(src, q)) for q in probes]
            resps = _greedy(llm, prompts, 256)
            qr = list(zip(probes, resps, strict=True))
            # persist the raw base completions (Phase-3/4 covariate derivation).
            raw_path = raw_root / f"{behavior}__{src}.json"
            raw_path.write_text(
                json.dumps(
                    {
                        **C.repro_meta(seed=C.DEFAULT_SEED),
                        "behavior": behavior,
                        "source": src,
                        "context": "bare source (no elicitation)",
                        "rows": [{"question": q, "base_completion": r} for q, r in qr],
                    },
                    ensure_ascii=False,
                )
            )
            raw_pointers[behavior][src] = str(raw_path.relative_to(CACHE_ROOT))
            if smoke:
                rates[behavior][src] = {"rate": None, "n_judged": 0, "note": "smoke: judge skipped"}
                continue
            # judge the base completions -> per-source base behavior rate.
            completions = {"base": {q: [r] for q, r in qr}}
            save_raw = raw_root / f"{behavior}__{src}__scores.json"
            judge_completions_batch(
                completions,
                judge_system_prompt=E._judge_system_prompt(column),
                format_user_msg=_user_msg,
                judge_model="claude-sonnet-4-5-20250929",
                cache_dir=raw_root / ".cache",
                save_raw=save_raw,
                dry_run=False,
            )
            all_scores = E._scores_from_save_raw(save_raw)
            vals = []
            for idx in range(len(qr)):
                s = all_scores.get(f"base__{idx:05d}__00")
                b = s.get("behavior") if s else None
                if isinstance(b, int | float):
                    vals.append(float(b))
            rates[behavior][src] = {
                "rate": (sum(vals) / len(vals)) if vals else None,
                "n_judged": len(vals),
            }

    out.write_text(
        json.dumps(
            {
                **C.repro_meta(seed=C.DEFAULT_SEED),
                "note": "source-side pre-training BASE-model behavior RATE covariate "
                "(bare source context, NO elicitation), judge-scored "
                "(claude-sonnet-4-5). Raw base completions persisted under "
                "baseline_raw/ for Phase-3/4 covariate derivation/audit. EM/marker/"
                "fact behaviors carry no bare-context judged rate here (their "
                "source-side prior is read on the trained store v0 + the marker "
                "slot base read).",
                "sources": list(sources),
                "behaviors": list(behaviors),
                "rated_behaviors": rated,
                "judged_rates": rates,
                "raw_completion_pointers": raw_pointers,
                "smoke": smoke,
            },
            indent=2,
        )
    )
    logger.info("[p0] baseline propensity written (%d rated behaviors) -> %s", len(rated), out)


# ── P2.1 train one cell ───────────────────────────────────────────────────────
def train_cell(cell: C.Cell, *, smoke: bool, gpu_id: int) -> Path:
    """Train one cell via the shared train_lora (marker band-stop / EM in-process
    via the #545 overrides / others). CVD is pinned in the launcher env per cell
    (gotchas: the in-process clobber alone is defeated by import-time cuInit) AND
    threaded as gpu_id; HF upload + Hub verify; per-cell WandB finish."""
    from explore_persona_space.train.sft import TrainLoraConfig, train_lora

    data_path = (
        GEN / ("train_smoke" if smoke else "train") / cell.behavior / f"{cell.eval_key}.jsonl"
    )
    assert data_path.exists(), f"training mix missing (run --phase p0 first): {data_path}"
    out_dir = ADAPTER_OUT / (cell.eval_key + ("_smoke" if smoke else ""))
    if (out_dir / "adapter_model.safetensors").exists():
        logger.info("[p1-train] %s already trained -- skip", cell.eval_key)
        return out_dir

    recipe = C.recipe_for(cell.behavior)
    kwargs = recipe.train_kwargs(
        dose=cell.dose, gpu_id=gpu_id, run_name=cell.run_name, seed=cell.seed
    )
    if smoke:
        kwargs["epochs"] = 1
        kwargs["max_steps"] = 2
        kwargs.pop("warmup_steps", None)
        if recipe.marker_only_loss:
            kwargs["marker_band_stop"] = False  # 2 steps can't band-stop; smoke
    # run_name / report_to / gpu_id / seed are already set inside train_kwargs;
    # only the HF-upload knobs are added here (no duplicate-keyword collision).
    cfg = TrainLoraConfig(
        hf_upload=not smoke,
        hf_repo=C.HF_MODEL_REPO,
        hf_path_in_repo=cell.hf_adapter_subfolder,
        **kwargs,
    )
    os.environ["EPM_SKIP_INLINE_CHECKPOINT_UPLOAD"] = "1"  # train_lora owns the upload
    try:
        train_lora(C.QWEN_ID, str(data_path), str(out_dir), cfg=cfg)
    finally:
        import wandb

        if wandb.run is not None:
            wandb.finish()  # one WandB run PER CELL (i537 precedent)
    if not smoke:
        _verify_adapter_on_hub(cell.hf_adapter_subfolder)
    return out_dir


def _verify_adapter_on_hub(subfolder: str) -> None:
    """Fail-loud Hub presence check (upload-policy)."""
    from huggingface_hub import list_repo_files

    files = list_repo_files(C.HF_MODEL_REPO, revision="main")
    want = f"{subfolder}/adapter_model.safetensors"
    if want not in files and not any(f.startswith(subfolder + "/") for f in files):
        raise RuntimeError(f"adapter not on Hub after upload: {C.HF_MODEL_REPO}/{subfolder}")
    logger.info("[hub] verified %s on %s", subfolder, C.HF_MODEL_REPO)


# ── P2.2 extract + eval-gen one cell (subprocess workers) ─────────────────────
def extract_and_eval_cell(cell: C.Cell, adapter_dir: Path, *, smoke: bool, gpu_id: int) -> None:
    """Run the extraction worker + the eval gen worker for one cell.

    Extraction merges the adapter (merge-read-delete); eval gen needs the merged
    model too, so we merge ONCE here, run both, then reap the merged dir."""
    from explore_persona_space.train.sft import merge_lora

    merged = adapter_dir.parent / (adapter_dir.name + "_merged")
    merge_lora(C.QWEN_ID, str(adapter_dir), str(merged), gpu_id=gpu_id)
    try:
        # extraction (tensors + marker slot) -- pass the ORIGINAL adapter dir so
        # extract_store does its own gauge assert on adapter_config.json.
        extract_cmd = [
            sys.executable,
            str(C.REPO / "scripts/issue664_extract_store.py"),
            "--behavior", cell.behavior, "--source", cell.source, "--arm", cell.arm,
            "--dose", cell.dose, "--seed", str(cell.seed), "--gpu-id", str(gpu_id),
            "--adapter-dir", str(adapter_dir),
        ]  # fmt: skip
        if smoke:
            extract_cmd.append("--smoke")
        subprocess.run(extract_cmd, check=True, cwd=C.REPO, env={**os.environ})
        # eval gen (raw completions + completion log-prob) on the merged model.
        gen_cmd = [
            sys.executable,
            str(C.REPO / "scripts/issue664_eval.py"),
            "--phase", "gen",
            "--behavior", cell.behavior, "--source", cell.source, "--arm", cell.arm,
            "--dose", cell.dose, "--seed", str(cell.seed),
            "--merged-path", str(merged),
        ]  # fmt: skip
        if smoke:
            gen_cmd.append("--smoke")
        subprocess.run(gen_cmd, check=True, cwd=C.REPO, env={**os.environ})
    finally:
        if merged.exists():
            import shutil

            shutil.rmtree(merged)
            logger.info("[p2] %s merged dir reaped", cell.eval_key)


# ── P2.3 upload raw completions + store tensors ───────────────────────────────
def upload_artifacts(cells: list[C.Cell], *, smoke: bool) -> None:
    """Push raw completions + store tensors to the HF data repo (adapters were
    pushed by train_lora). Fail-loud per upload-policy."""
    if smoke:
        logger.info("[p3-upload] smoke: skipping HF upload")
        return
    _upload_raw_completions(cells)
    _upload_store_tensors(cells)


def _upload_raw_completions(cells: list[C.Cell]) -> None:
    """Upload the eval-gen raw completions to the canonical HF data-repo path.

    The eval gen writes ``eval_results/issue_664/registry/<cell>/completions__
    <col>__<ctx>.json`` -- a NON-canonical shape that the helper's
    ``rglob('raw_completions.json')`` does NOT match (the #528 silent-loss
    class). So we walk the ACTUAL write path and upload per-file with
    ``upload_as_file=True`` to ``issue664_leakage_fleet/raw_completions/<rel>``,
    then verify the per-cell file count landed on the Hub before teardown."""
    from huggingface_hub import list_repo_files

    from explore_persona_space.orchestrate import hub

    prefix = C.HF_RAW_COMPLETIONS_PREFIX  # issue664_leakage_fleet/raw_completions
    reg_root = C.EVAL_ROOT / "registry"
    files = sorted(reg_root.rglob("completions__*.json"))
    if not files:
        raise RuntimeError(
            f"[p3-upload] NO raw completions under {reg_root} -- the eval gen "
            "phase produced nothing; refusing to terminate with empty buckets"
        )
    n_expected = 0
    for f in files:
        rel = f.relative_to(reg_root).as_posix()  # <cell>/completions__<col>__<ctx>.json
        hub._upload(
            f,
            repo_id=C.HF_DATA_REPO,
            repo_type="dataset",
            path_in_repo=f"{prefix}/{rel}",
            upload_as_file=True,  # gotchas: per-file _upload needs this
        )
        n_expected += 1
    # verify on a FRESH listing (Python Hub API, never the hf CLI).
    landed = [
        p for p in list_repo_files(C.HF_DATA_REPO, repo_type="dataset") if p.startswith(prefix)
    ]
    if len(landed) < n_expected:
        raise RuntimeError(
            f"[p3-upload] raw-completions verify FAILED: {len(landed)} on Hub < "
            f"{n_expected} expected under {prefix}"
        )
    logger.info(
        "[p3-upload] %d raw-completion files uploaded + verified -> %s/%s",
        n_expected,
        C.HF_DATA_REPO,
        prefix,
    )


def _upload_store_tensors(cells: list[C.Cell]) -> None:
    """Mirror each selected cell's trained-store (the Phase-3/4 PRIMARY deliverable)
    to the HF data repo. #664 round-2 M5: a MISSING ``tensors.pt`` for a selected
    cell is FAIL-LOUD, NOT a warn-and-continue -- the prior `logger.warning; continue`
    let the dispatcher reach `[phase=done]` with incomplete primary deliverables
    mirrored to HF (the #521 trap variant: a downstream control becomes permanently
    unrunnable, discovered only post-teardown)."""
    from explore_persona_space.orchestrate import hub

    missing: list[str] = []
    for cell in cells:
        cell_dir = C.STORE_ROOT / cell.eval_key
        tp = cell_dir / "tensors.pt"
        if not tp.exists():
            missing.append(f"{cell.eval_key} ({tp})")
            continue
        for f in cell_dir.glob("*"):
            if f.is_file():
                hub._upload(
                    f,
                    repo_id=C.HF_DATA_REPO,
                    repo_type="dataset",
                    path_in_repo=f"{C.HF_STORE_PREFIX}/{cell.eval_key}/{f.name}",
                    upload_as_file=True,  # gotchas: per-file _upload needs this
                )
    if missing:
        raise RuntimeError(
            "[p3-upload] trained-store tensors MISSING for "
            f"{len(missing)} selected cell(s): {missing}. Refusing to reach "
            "[phase=done] with incomplete PRIMARY deliverables (the Phase-3/4 "
            "input) -- this is the #521 trap. Investigate the P2.2 extraction."
        )
    logger.info("[p3-upload] store tensors uploaded -> %s/%s", C.HF_DATA_REPO, C.HF_STORE_PREFIX)


# ── Marker read-gauge readability assert (§10 / §11 A7) ───────────────────────
def _marker_readability_assert(cells: list[C.Cell], *, smoke: bool) -> None:
    """A7 read-gauge readability gate: ≥1 trained marker adapter at the
    band-stopped checkpoint has on-policy marker emission < 1% AND
    log P(marker) < log P(<|im_end|>) (z_marker < z_eos) on EVERY eval-probe slot.

    #664 round-2 B3: this now reads the PRODUCTION marker_slot_stats.json (NOT the
    `_smoke` path) when ``smoke`` is False, and RAISES on a readability FAILURE so
    the fleet HALTS (per plan §11 Option A -> Option B staged classic-alpha/r read
    before relaunch). The prior version hard-coded the `_smoke` path and only
    LOGGED -- on a production run it either crashed (no `_smoke` path exists) or
    silently let saturated adapters pass. Path keys on the SEED-QUALIFIED eval_key.

    SMOKE mode: smoke trains only 2 steps (no band-stop), so the read is a
    STRUCTURAL exercise (logs, never raises on the emission/margin verdict) -- it
    only fails if NO marker_slot produced (the extraction path is broken)."""
    marker_cells = [c for c in cells if c.behavior == "marker"]
    if not marker_cells:
        logger.info("[a7-assert] no marker cell in selection -- A7 read-gauge assert N/A")
        return
    suffix = "_smoke" if smoke else ""
    checked = 0
    failures: list[str] = []
    for cell in marker_cells:
        slot_path = (
            C.EVAL_ROOT / "marker_slot" / (cell.eval_key + suffix) / "marker_slot_stats.json"
        )
        if not slot_path.exists():
            continue
        payload = json.loads(slot_path.read_text())
        slots = payload["slots"]
        n_emit = sum(1 for s in slots.values() if s["trained"].get("argmax_id") == C.MARKER_ID)
        emit_rate = n_emit / max(1, len(slots))
        margins = [s["trained"]["z_marker"] - s["trained"]["z_eos"] for s in slots.values()]
        below_eos = all(m < 0 for m in margins) if margins else False
        ok = emit_rate < 0.01 and below_eos
        logger.info(
            "[a7-assert] %s emission=%.3f (<1%%=%s), z_marker<z_eos on all slots=%s -> %s",
            cell.eval_key,
            emit_rate,
            emit_rate < 0.01,
            below_eos,
            "PASS" if ok else "FAIL",
        )
        checked += 1
        if not ok and not smoke:
            failures.append(
                f"{cell.eval_key}: emit_rate={emit_rate:.3f} (need <0.01), "
                f"all margins<0={below_eos}"
            )
    if checked == 0:
        raise RuntimeError(
            f"[a7-assert] marker readability assert ran on 0 marker cells (no "
            f"marker_slot_stats.json under marker_slot/*{suffix}) -- the A7 read-gauge "
            "readability test could not run; investigate the extraction marker-slot path."
        )
    # PRODUCTION (not smoke): a readability FAILURE HALTS the fleet (plan §11
    # Option A -> Option B staged classic-alpha/r read before relaunch). Smoke logs only.
    if failures:
        raise RuntimeError(
            "[a7-assert] PRODUCTION marker read-gauge readability HALT (plan §11 Option A): "
            f"{len(failures)} band-stopped marker cell(s) emit the marker on-policy or have "
            f"z_marker >= z_eos at some slot, so the faithful-gauge read is NOT clean. Adopt "
            f"Option B (staged classic alpha/r=2.0, use_rslora=False, eval-apply only) before "
            f"relaunch. Failures: {failures}"
        )


# ── Orchestration ─────────────────────────────────────────────────────────────
def run_all(args) -> None:
    _require_credentials()
    cells = _select_cells(args)
    logger.info("[dispatch] %d cells selected (smoke=%s)", len(cells), args.smoke)

    if args.phase in ("all", "p0"):
        phase0(args)
    if args.phase == "p0":
        return

    if args.phase in ("all", "p1"):
        phase_log("p1_train")
        for cell in cells:
            train_cell(cell, smoke=args.smoke, gpu_id=args.gpu_id)
    if args.phase == "p1":
        return

    if args.phase in ("all", "p2"):
        phase_log("p2_extract_eval")
        for cell in cells:
            adapter_dir = ADAPTER_OUT / (cell.eval_key + ("_smoke" if args.smoke else ""))
            extract_and_eval_cell(cell, adapter_dir, smoke=args.smoke, gpu_id=args.gpu_id)
        # registry manifest (the §6.5 verifier surface).
        phase_log("p2_manifest")
        _write_manifest(cells, smoke=args.smoke)
        # marker read-gauge readability assert (§10 / §11 A7); production -> HALT
        # on a readability failure (#664 round-2 B3).
        phase_log("p2_a7_assert")
        _marker_readability_assert(cells, smoke=args.smoke)
        # #664 round-2 B5: in smoke, exercise the PRODUCTION judge branch live on a
        # tiny real Batch-API slice (one content-behavior cell, 1 column, ≤5 comps).
        if args.smoke and args.live_judge_smoke:
            _live_judge_smoke(cells)
    if args.phase == "p2":
        return

    if args.phase in ("all", "p3"):
        phase_log("p3_upload")
        upload_artifacts(cells, smoke=args.smoke)


def _live_judge_smoke(cells: list[C.Cell]) -> None:
    """#664 round-2 B5: run a tiny REAL (non-dry-run) Batch-API judge on the first
    content-behavior cell in the selection so the smoke exercises the production
    judge branch (dry_run=False path) end-to-end. Asserts a real all_scores dict
    landed."""
    import issue664_eval as E

    content_cells = [c for c in cells if c.behavior in C.CONTENT_BEHAVIORS]
    if not content_cells:
        logger.info("[live-judge-smoke] no content-behavior cell selected -- B5 slice N/A")
        return
    cell = content_cells[0]
    phase_log("p2_live_judge_smoke")
    out_path = E.judge_cell(cell, smoke=True, live_judge=True)
    payload = json.loads(out_path.read_text())
    judged = sum(r.get("n_judged", 0) for r in payload["rates"].values())
    if judged < 1:
        raise RuntimeError(
            f"[live-judge-smoke] live judge produced n_judged=0 for {cell.eval_key} -- the "
            "production judge branch did NOT yield real scores (B5 smoke contract)"
        )
    logger.info(
        "[live-judge-smoke] %s live judge n_judged=%d (real Batch API)", cell.eval_key, judged
    )


def _write_manifest(cells: list[C.Cell], *, smoke: bool) -> None:
    cmd = [sys.executable, str(C.REPO / "scripts/issue664_eval.py"), "--phase", "manifest"]
    if smoke:
        cmd.append("--smoke")
    subprocess.run(cmd, check=True, cwd=C.REPO, env={**os.environ})


def main() -> int:
    ap = argparse.ArgumentParser(description="Issue #664 Phase-2 fleet driver.")
    ap.add_argument("--phase", default="all", choices=["all", "p0", "p1", "p2", "p3"])
    ap.add_argument("--cells", type=int, default=None, help="cap the cell count (smoke: 1)")
    ap.add_argument("--gpu-id", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument(
        "--live-judge-smoke",
        action="store_true",
        help="in --smoke, run a tiny REAL Batch-API judge slice on one content cell "
        "to exercise the production judge branch (#664 round-2 B5)",
    )
    args = ap.parse_args()

    try:
        run_all(args)
    except Exception as e:  # fail-loud: write a failure sentinel, re-raise
        logger.exception("[dispatch] FAILED")
        write_sentinel(
            "epm:failure",
            f"issue664 dispatch failed at phase={args.phase}: {type(e).__name__}: {e}",
            extra={"failure_class": "code", "phase": args.phase},
        )
        raise

    if args.phase == "all":
        sel = _select_cells(args)
        n = len(sel)
        write_sentinel(
            "epm:results",
            f"issue664 Phase-2 fleet complete ({n} cells, smoke={args.smoke})",
            extra={
                "gate": "results",
                "blocks_pipeline": False,
                "reproducibility_card": {
                    "wandb_project": C.WANDB_PROJECT,
                    "wandb_entity": _wandb_entity(),  # read off the SDK, not hand-typed
                    "wandb_run_names": [c.run_name for c in sel],
                    "adapter_paths": [c.hf_adapter_subfolder for c in sel],
                    "hf_model_repo": C.HF_MODEL_REPO,
                    "store_tensors_prefix": f"{C.HF_DATA_REPO}/{C.HF_STORE_PREFIX}",
                    "judge_model": "claude-sonnet-4-5-20250929",
                    "seeds": sorted({c.seed for c in sel}),
                },
            },
        )
        phase_log("done")  # RESERVED terminal line (poll_pipeline) -- main exit only
    return 0


def _wandb_entity() -> str | None:
    """Read the WandB entity off the SDK at run time (never hand-typed -- a stale
    literal breaks resolution when the account changes, #597). Returns None if the
    SDK cannot resolve it (the verifier falls back to api.default_entity)."""
    try:
        import wandb

        return wandb.Api().default_entity
    except Exception as e:  # SDK/login resolution failure -> None (verifier fallback)
        logger.warning("[card] could not resolve wandb entity off the SDK: %s", e)
        return None


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(rc)  # datasets/transformers SIGABRT at finalize (gotchas PyGILState)
