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
import shutil
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # issue664_* / issue594_common

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")  # gotchas #628 fork-poison

from explore_persona_space.orchestrate.env import load_dotenv
from explore_persona_space.orchestrate.fleet import (
    CellCmd,
    JudgeHandle,
    WaveDispatcher,
    submit_judge_async,
)

load_dotenv()  # P2.0 vLLM + train_lora + HF uploads need HF_TOKEN / WANDB_API_KEY

import issue664_build_training_data as B  # noqa: E402  (DROPPED_SOURCE_EXIT + builder reuse)
import issue664_common as C  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("issue664_dispatch")

GEN = C.DATA_ROOT
CACHE_ROOT = C.DATA_ROOT / "onpolicy_cache"
ADAPTER_OUT = C.DATA_ROOT / "adapters"
# Per-cell below-floor-yield DROP sentinels live here (written by the builder);
# the top-level manifest of dropped cells (#664 round-6) is its _manifest.json.
DROPPED_DIR = CACHE_ROOT / "dropped_sources"


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
# The marker band-stop architecture canary (exercises the in-process band-stop
# train path) and the content-behavior judge canary (exercises the production
# Batch-API judge branch). The two are distinct behaviors, so under
# --live-judge-smoke BOTH must survive the --cells cap (#664 round-2 B7).
SMOKE_MARKER_CANARY = C.Cell("marker", "default", "contra", "d1")
SMOKE_CONTENT_CANARY = C.Cell("sycophancy", "default", "contra", "d1")


def _select_cells(args) -> list[C.Cell]:
    grid = C.realized_grid()
    if args.smoke:
        # Canary ordering: the marker x default x contrastive x dose-1 cell
        # (seed 42) exercises the band-stop path; it is the smoke-architecture
        # canary (§ smoke parity). Match on the SEED-QUALIFIED eval_key so the
        # seed-1042 replication twin is NOT pulled to the front instead.
        #
        # #664 round-2 B7: marker ∉ CONTENT_BEHAVIORS, so a marker-only smoke
        # selection leaves `_live_judge_smoke` with zero content cells and the
        # PRODUCTION judge branch is never exercised through the launcher. When
        # --live-judge-smoke is set, ALSO pull a content-behavior canary
        # (sycophancy x default x contrastive x dose-1) to the front so the live
        # judge slice runs on a real generated content cell.
        front_keys = [SMOKE_MARKER_CANARY.eval_key]
        if args.live_judge_smoke:
            front_keys.append(SMOKE_CONTENT_CANARY.eval_key)
        by_key = {c.eval_key: c for c in grid}
        front = [by_key[k] for k in front_keys if k in by_key]
        if args.live_judge_smoke and not any(c.behavior in C.CONTENT_BEHAVIORS for c in front):
            raise RuntimeError(
                "[smoke] --live-judge-smoke set but no content-behavior canary "
                f"({SMOKE_CONTENT_CANARY.eval_key}) is in the realized grid; the "
                "production judge branch cannot be exercised. Fix the canary key."
            )
        rest = [c for c in grid if c.eval_key not in front_keys]
        grid = front + rest
    if args.cells is not None:
        # Never truncate away a smoke canary that MUST run: under
        # --live-judge-smoke we need >=1 content cell to survive, so floor the
        # cap at the number of front canaries (the marker canary is also kept so
        # the band-stop architecture smoke still fires).
        n = args.cells
        if args.smoke and args.live_judge_smoke:
            n = max(n, 2)
        grid = grid[:n]
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
    """The marker eval probes the marker DV is SCORED on (the extract_store
    battery) -- kept disjoint from the 300-row training pool above.

    §16: routes through the ONE canonical resolver
    (``C.canonical_battery_for_column('marker')`` -> the #545
    ``marker_eval_questions.json`` battery), NOT a hand-rolled
    ``fetch_preregistered_probes(48)`` call. The marker column self-routes to
    its own frozen battery like every other #545 column."""
    return [it["question"] for it in C.canonical_battery_for_column("marker")]


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
    """The refusal REQUEST pool used for P2.0 elicitation -- the SAME #390
    request battery the store activation surface + the refusal judge score on
    (B6). #664 round-2 B6: the prior implementation elicited refusal positives
    on the generic 48 preregistered Betley probes while the store/judge read a
    DIFFERENT surface; routing both through ``C.refusal_request_pool()`` keeps
    elicit -> train -> store -> judge on one surface. NOTE: this pool is written
    to ``pools/refusal.json`` by ``phase0`` and is what ``C.refusal_request_pool``
    reads back first, so production stays self-consistent."""
    if smoke:
        return C.SMOKE_QUESTIONS
    return C.refusal_request_pool()


# ── Dropped-cell tracking (#664 round-6 below-floor-yield-fleet-crash) ─────────
def _write_dropped_manifest(dropped: list[C.Cell]) -> Path:
    """Write the top-level manifest of cells dropped below the on-policy yield floor
    at P2.0 (``onpolicy_cache/dropped_sources/_manifest.json``). The analyzer carries
    this as a coverage caveat; the upload-verifier reads it so a dropped cell's absent
    artifacts do NOT trip the fail-loud missing-artifact asserts. Always written (an
    empty list is the no-drops case) so a stale prior-run manifest never lingers."""
    DROPPED_DIR.mkdir(parents=True, exist_ok=True)
    out = DROPPED_DIR / "_manifest.json"
    out.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "reason": "below-80%-yield-floor",
                "yield_floor": B.YIELD_FLOOR,
                "dropped_cells": [
                    {
                        "eval_key": c.eval_key,
                        "behavior": c.behavior,
                        "source": c.source,
                        "arm": c.arm,
                        "dose": c.dose,
                        "seed": c.seed,
                    }
                    for c in dropped
                ],
                "ts": time.time(),
            },
            indent=2,
        )
    )
    if dropped:
        logger.warning(
            "[p0-build] %d cell(s) dropped below the yield floor -> %s: %s",
            len(dropped),
            out,
            sorted(c.eval_key for c in dropped),
        )
    return out


def _dropped_cell_keys() -> set[str]:
    """The set of SEED-QUALIFIED ``eval_key``s dropped below the yield floor, read
    from the top-level manifest (preferred) or, if it is absent, the union of the
    per-cell drop sentinels (so a dispatcher restarted at --phase p1/p2/p3 still
    excludes a drop recorded by an earlier --phase p0 process). Empty when nothing
    was dropped."""
    keys: set[str] = set()
    manifest = DROPPED_DIR / "_manifest.json"
    if manifest.exists():
        payload = json.loads(manifest.read_text())
        return {c["eval_key"] for c in payload.get("dropped_cells", [])}
    # Fallback: reconstruct from the per-cell sentinels the builder wrote.
    if DROPPED_DIR.exists():
        for f in DROPPED_DIR.glob("*.json"):
            if f.name == "_manifest.json":
                continue
            try:
                keys.add(json.loads(f.read_text())["eval_key"])
            except (json.JSONDecodeError, KeyError):
                continue
    return keys


def _drop_filtered(cells: list[C.Cell]) -> list[C.Cell]:
    """Exclude cells dropped below the yield floor from a selected-cell list, so the
    train / extract / eval / manifest / upload / repro-card phases never reference a
    cell whose training mix was never built (#664 round-6). Logs what was excluded."""
    dropped = _dropped_cell_keys()
    if not dropped:
        return cells
    kept = [c for c in cells if c.eval_key not in dropped]
    skipped = [c.eval_key for c in cells if c.eval_key in dropped]
    if skipped:
        logger.warning(
            "[dispatch] excluding %d dropped cell(s) from downstream phases: %s",
            len(skipped),
            sorted(skipped),
        )
    return kept


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

    # #676 judge overlap: the on-pod behavior-label + baseline-propensity judges
    # are SUBMITTED fire-and-forget right after their generations and reconciled
    # AFTER the vLLM engine is torn down (off the GPU critical path), before the
    # build-mixes step that consumes the labels. Each elicit/baseline call appends
    # a deferred reconcile job here (a zero-arg closure that harvests its judge
    # handle + writes its cache). Smoke skips the live judge (jobs resolve to
    # all-1 labels / smoke-skipped rates) so no Batch API call fires.
    judge_jobs: list[Callable[[], None]] = []

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
            _elicit_sycophancy(llm, sources, judge_jobs, smoke=smoke)
        # refusal positives + on-policy normal-answer negatives.
        if "refusal" in behaviors:
            _elicit_refusal(llm, sources, neg_panel, refusal_qs, judge_jobs, smoke=smoke)
        # insecure-code on-policy secure-answer negatives (ic_secure) per source/neg.
        if any(b in ("em", "ic_edu") for b in behaviors):
            _elicit_secure_code(llm, sources, neg_panel, smoke=smoke)
        # source-side baseline propensity read (#664 round-2 M4): BASE-model
        # behavior RATE per (source, content-behavior) on the bare source context
        # (NO elicitation). The base generations run while the engine is alive; the
        # judge is SUBMITTED fire-and-forget here and reconciled below (#676).
        _write_baseline_propensity(llm, sources, behaviors, refusal_qs, judge_jobs, smoke=smoke)
    finally:
        _teardown_vllm(llm)

    # #676 judge-overlap reconcile barrier: harvest every submitted judge + write
    # its cache NOW (engine already freed), BEFORE build-mixes reads the labels.
    # Fail-loud: a reconcile job propagates BatchDeadlineExceeded rather than
    # defaulting any label.
    if judge_jobs:
        phase_log("p0_judge_reconcile")
        logger.info("[p0] reconciling %d overlapped judge job(s)", len(judge_jobs))
        for job in judge_jobs:
            job()

    # Build each cell's training mix (the builder asserts panel∩sources=∅, marker
    # token, len(probes)==48 internally; we drive it as a subprocess so the
    # builder's own load_dotenv + asserts run in a clean process per cell).
    #
    # #664 round-6 below-floor-yield-fleet-crash: a source below the 80% on-policy
    # yield floor exits with B.DROPPED_SOURCE_EXIT (3) -- a DELIBERATE drop (plan v4
    # §11 graceful degradation: drop + report, never crash). Treat rc==3 as
    # "skip this cell, continue the fleet"; ANY other non-zero rc is a genuine crash
    # and stays fatal (re-raised as CalledProcessError). The per-cell drop sentinel
    # is written by the builder under onpolicy_cache/dropped_sources/; the dispatcher
    # accumulates the dropped cells + writes a top-level manifest so every downstream
    # phase (train / extract / eval / manifest / upload / repro-card) excludes them.
    phase_log("p0_build_mixes")
    dropped: list[C.Cell] = []
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
        result = subprocess.run(cmd, check=False, cwd=C.REPO, env={**os.environ})  # explicit env
        if result.returncode == B.DROPPED_SOURCE_EXIT:
            logger.warning(
                "[p0-build] %s DROPPED below the on-policy yield floor (rc=%d); "
                "skipping this cell + continuing the fleet (plan v4 §11 graceful "
                "degradation -- drop + report, see onpolicy_cache/dropped_sources/)",
                cell.eval_key,
                result.returncode,
            )
            dropped.append(cell)
            continue
        if result.returncode != 0:
            raise subprocess.CalledProcessError(result.returncode, cmd)
    _write_dropped_manifest(dropped)


def _render(messages: list[dict]) -> str:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(C.QWEN_ID, trust_remote_code=True)
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def _behavior_judge_user_msg(question: str, completion: str) -> str:
    return f"User request / claim:\n{question}\n\nAI response:\n{completion}"


def _submit_behavior_labels(
    behavior: str, src: str, qr_pairs: list[tuple[str, str]], *, smoke: bool
) -> JudgeHandle | None:
    """Fire-and-forget submit the behavior-label judge for one source's elicited
    positives.

    Returns a :class:`JudgeHandle` (already ``.submit()``ed) whose ``.reconcile()``
    harvests the labels LATER, off the GPU critical path (#676 judge overlap) — so
    the vLLM engine keeps generating the next source's R while the prior source's
    judge clears. Returns ``None`` in smoke (no live judge — labels are all 1,
    resolved by :func:`_labels_from_handle`).

    ``save_raw`` is keyed PER SOURCE (``judge_filter/{behavior}__{src}.json``), NOT
    on a coarse ``int(time.time())`` (#676 round-2, concern judge-save-raw-collision):
    the deferred-reconcile split means two same-behavior source jobs whose wall-clock
    second collides would otherwise share one ``save_raw`` — the first job's
    reconcile writes it, the second reads the FIRST source's scores back, and because
    the custom_ids are POSITION-keyed (``elicit__{idx:05d}__00``, identical across
    sources but reflecting different ``resps``) the second source silently inherits
    the first source's labels → corrupted ``judge_behavior`` cache. The source is the
    disambiguator; idempotency on resume comes from ``make_custom_id`` + the
    save_raw-exists sentinel, not from the path's uniqueness over time.

    The judge is claude-sonnet-4-5-20250929 via the #663-hardened Batch-API client
    (NEVER a substring match -- CLAUDE.md), on the SAME per-column rubric the
    off-pod registry judge uses (issue664_eval), so the elicitation filter and the
    eval judge agree by construction."""
    import issue664_eval as E

    if smoke:
        return None
    column = {"sycophancy": "sycophancy", "refusal": "refusal"}[behavior]
    # batch_judge shape: {persona -> {question -> [completions]}}; one completion each.
    # persona "elicit" -> custom_id elicit__{idx:05d}__00 (the scheme the reader uses).
    completions = {"elicit": {q: [r] for q, r in qr_pairs}}
    save_raw = CACHE_ROOT / "judge_filter" / f"{behavior}__{src}.json"
    handle = submit_judge_async(
        completions,
        judge_system_prompt=E._judge_system_prompt(column),
        format_user_msg=_behavior_judge_user_msg,
        cell_key=f"elicit_{behavior}__{src}",
        save_raw=save_raw,
        expected_source=src,
        judge_model="claude-sonnet-4-5-20250929",
    )
    handle.submit()
    return handle


def _labels_from_handle(
    handle: JudgeHandle | None, qr_pairs: list[tuple[str, str]], *, smoke: bool
) -> dict[str, int]:
    """Reconcile a behavior-label judge handle into {question -> 0/1}.

    1 = the response EXHIBITS the target behavior. Smoke (``handle is None``)
    returns all-1 (no live judge), mirroring the prior synchronous path. Fail-loud
    on a judge deadline — the handle's ``reconcile()`` propagates
    ``BatchDeadlineExceeded`` rather than defaulting labels."""
    if smoke or handle is None:
        return {q: 1 for q, _r in qr_pairs}
    all_scores = handle.reconcile()
    # custom_id = elicit__{idx:05d}__00, idx increments per question in order.
    labels: dict[str, int] = {}
    for idx, (q, _r) in enumerate(qr_pairs):
        s = all_scores.get(f"elicit__{idx:05d}__00")
        b = s.get("behavior") if s else None
        labels[q] = 1 if isinstance(b, int | float) and b >= 1 else 0
    return labels


def _elicit_sycophancy(llm, sources, judge_jobs: list, *, smoke: bool) -> None:
    """#612 tier-2 instruct-and-strip: add an agree-with-wrong-claims instruction
    to the system prompt at GEN time, sample, then SUBMIT the JUDGE-FILTER
    (#664 round-2 M2) fire-and-forget (#676 overlap) and defer the cache write to a
    reconcile job. The build accepts only judge-positive rows + enforces the 80%
    yield floor on the JUDGED-positive count.

    #676: the judge is submitted right after the generations and reconciled later
    (off the GPU critical path) — ``judge_jobs`` accumulates ``(handle, qr,
    write_cache)`` deferred reconcile jobs the caller runs before build-mixes; the
    response cache (with labels) is written at reconcile time."""
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
        handle = _submit_behavior_labels("sycophancy", src, qr, smoke=smoke)
        judge_jobs.append(
            _make_pos_cache_job("syco_pos", src, "sycophancy", qr, handle, smoke=smoke)
        )


def _elicit_refusal(llm, sources, neg_panel, requests, judge_jobs: list, *, smoke: bool) -> None:
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
            # JUDGE-FILTER refusal positives (#664 round-2 M2), submitted
            # fire-and-forget + deferred to a reconcile job (#676 overlap).
            handle = _submit_behavior_labels("refusal", src, qr, smoke=smoke)
            judge_jobs.append(
                _make_pos_cache_job("refusal_pos", src, "refusal", qr, handle, smoke=smoke)
            )
    for neg in neg_panel:
        if (CACHE_ROOT / "refusal_neg" / f"{neg.slug}.json").exists():
            continue
        prompts = [_render(neg.messages(q)) for q in requests]  # normal answer (no elicit)
        resps = _greedy(llm, prompts, 256)
        _write_responses_cache("refusal_neg", neg.slug, dict(zip(requests, resps, strict=True)))


def _make_pos_cache_job(
    kind: str,
    ctx_key: str,
    behavior: str,
    qr: list[tuple[str, str]],
    handle: JudgeHandle | None,
    *,
    smoke: bool,
) -> Callable[[], None]:
    """Build the deferred reconcile job that harvests the judge labels + writes the
    judge-filtered positive cache (#676 overlap). Run AFTER the vLLM engine is torn
    down, BEFORE build-mixes — so the judge cleared off the GPU critical path."""

    def _job() -> None:
        labels = _labels_from_handle(handle, qr, smoke=smoke)
        _write_responses_cache(kind, ctx_key, dict(qr), judge_labels=labels)

    return _job


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
    context, NO elicitation). Routes through the behavior's OWN scoring battery
    (``C.canonical_battery_for_behavior`` -- the §16 ONE resolver) -- the SAME
    surface the eval judge + trained store use (B6) -- capped to a cheap baseline
    read size. #664 round-2 M6:
    extended from {sycophancy, refusal} to EVERY content behavior with a judged
    column (fact / em / bad_medical too) so all primary transfer-spine behaviors
    carry a source-side base-prior covariate (plan §4)."""
    if smoke:
        return C.SMOKE_QUESTIONS[:3]
    if behavior not in C.CONTENT_BEHAVIORS:
        return []
    # §16: behavior-own battery via the ONE canonical per-behavior resolver
    # (sycophancy=Sharma wrong-claims, refusal=#390 requests, fact=#444 templates,
    # em=betley-main-8, bad_medical=fam_expr_bad_medical), capped to a 30-probe
    # read. Extract flat questions from the probe-item dicts.
    return [it["question"] for it in C.canonical_battery_for_behavior(behavior)][:30]


def _write_baseline_propensity(
    llm, sources, behaviors, refusal_qs, judge_jobs: list, *, smoke: bool
) -> None:
    """Source-side BASE-model behavior RATE covariate (#664 round-2 M4): the
    registered plan-§4 source-side base-prior read. For each (source, content
    behavior with a bare-context base read), generate the BASE model's completions
    under the BARE source context (NO elicitation instruction), SUBMIT the judge
    fire-and-forget (#676 overlap), and persist BOTH the per-(source, behavior)
    judged rate AND the raw base completions (so Phase-3/4 can re-derive / audit
    the covariate). NOT a placeholder -- the artifact carries the rates + raw rows.

    #676: the base generations run while the engine is alive; the judge is
    submitted here and reconciled off the GPU critical path. ``judge_jobs``
    accumulates per-(behavior, src) reconcile closures (which populate ``rates``)
    plus a FINAL closure that writes ``baseline_propensity.json`` after all
    per-cell rates are in. Smoke skips the live judge (rates -> smoke-skipped)."""
    import issue664_eval as E

    out = CACHE_ROOT / "baseline_propensity.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    raw_root = CACHE_ROOT / "baseline_raw"
    raw_root.mkdir(parents=True, exist_ok=True)

    # #664 round-2 M6: every CONTENT behavior with a judged column carries a
    # source-side base-prior read (plan §4) -- not just sycophancy/refusal.
    # bad_medical / em (insecure-code) judge on the Betley aligned/coherent
    # dual-rubric; fact judges on the taught-fact rate; sycophancy/refusal on
    # their 0/1 behavior rubric. The behavior -> judge column map is the
    # canonical BEHAVIOR_REGISTRY_PRIMARY_COLUMN.
    rated = [b for b in behaviors if b in C.CONTENT_BEHAVIORS]
    rates: dict[str, dict[str, dict]] = {}
    raw_pointers: dict[str, dict[str, str]] = {}

    for behavior in rated:
        probes = _baseline_probe_pool(behavior, smoke)
        # judge on the behavior's PRIMARY registry column (NOT the bare behavior
        # name -- em -> broad_em, bad_medical -> fam_expr_bad_medical pick the
        # Betley dual-rubric; fact -> fact_expression).
        column = C.BEHAVIOR_REGISTRY_PRIMARY_COLUMN[behavior]
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
                        "judge_column": column,
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
            # SUBMIT the judge fire-and-forget; the reconcile job (run after engine
            # teardown) harvests it -> per-source base behavior rate via
            # E._rate_from_raw_scores (the Betley aligned/coherent aggregation for
            # broad_em / fam_expr_bad_medical AND the 0/1 aggregation for
            # sycophancy / refusal / fact_expression are both handled correctly).
            completions = {"cell": {q: [r] for q, r in qr}}
            save_raw = raw_root / f"{behavior}__{src}__scores.json"
            handle = submit_judge_async(
                completions,
                judge_system_prompt=E._judge_system_prompt(column),
                format_user_msg=_behavior_judge_user_msg,
                cell_key=f"baseline_{behavior}_{src}",
                save_raw=save_raw,
                expected_source=src,
                judge_model="claude-sonnet-4-5-20250929",
            )
            handle.submit()
            judge_jobs.append(_make_baseline_rate_job(rates, behavior, src, column, qr, handle))

    # FINAL reconcile job: write the aggregate AFTER every per-(behavior, src) rate
    # job has populated `rates`. Appended last so it runs last.
    def _write_out() -> None:
        out.write_text(
            json.dumps(
                {
                    **C.repro_meta(seed=C.DEFAULT_SEED),
                    "note": "source-side pre-training BASE-model behavior RATE covariate "
                    "(bare source context, NO elicitation), judge-scored "
                    "(claude-sonnet-4-5). Raw base completions persisted under "
                    "baseline_raw/ for Phase-3/4 covariate derivation/audit. #664 "
                    "round-2 M6: covers EVERY content behavior with a judged column "
                    "(sycophancy/refusal/fact + em/bad_medical Betley dual-rubric); "
                    "the marker source-side prior is read on the marker slot base "
                    "read (no judged content rate for marker).",
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

    judge_jobs.append(_write_out)


def _make_baseline_rate_job(
    rates: dict,
    behavior: str,
    src: str,
    column: str,
    qr: list[tuple[str, str]],
    handle: JudgeHandle,
) -> Callable[[], None]:
    """Deferred reconcile job: harvest the (behavior, src) base-prior judge handle
    and populate ``rates[behavior][src]`` (#676 overlap)."""
    import issue664_eval as E

    def _job() -> None:
        handle.reconcile()  # writes save_raw in the all_scores shape
        all_scores = E._scores_from_save_raw(handle.save_raw)
        agg_rows = [{"question": q, "completions": [r]} for q, r in qr]
        agg = E._rate_from_raw_scores(column, agg_rows, all_scores)
        rates[behavior][src] = {
            "rate": agg["rate"],
            "n_judged": agg["n_judged"],
            "judge_column": column,
        }

    return _job


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
        # --- per-cell incremental upload (fix (a), #664/#689, checkpoint-per-phase) ---
        # Both final artifacts now exist on the pod volume (the _cell_extract_eval_done
        # invariant: store tensors.pt + a non-empty eval registry dir). Push THIS cell
        # to HF the moment its extract+eval worker succeeds, so a mid-sweep pod death
        # (the #664 RUNNING-but-no-port wedge) strands at most this one in-flight cell,
        # never the whole sweep. Idempotent (skips when already complete on Hub) +
        # fail-loud (EXACT-set Hub-verify). Smoke short-circuits inside the helper.
        _upload_cell_artifacts(cell, smoke=smoke)
    finally:
        if merged.exists():
            import shutil

            shutil.rmtree(merged)
            logger.info("[p2] %s merged dir reaped", cell.eval_key)


# ── Wave-parallel cell dispatch (#676) ────────────────────────────────────────
def _adapter_dir_for(cell: C.Cell, *, smoke: bool) -> Path:
    """The per-cell adapter dir (the SEED-QUALIFIED key + the smoke suffix)."""
    return ADAPTER_OUT / (cell.eval_key + ("_smoke" if smoke else ""))


def _train_done(cell: C.Cell, *, smoke: bool) -> bool:
    """Idempotent train skip-completed predicate — the SAME final-artifact check
    ``train_cell`` (:806) keys on: the cell's ``adapter_model.safetensors`` exists."""
    return (_adapter_dir_for(cell, smoke=smoke) / "adapter_model.safetensors").exists()


def _cell_extract_eval_done(cell: C.Cell, *, smoke: bool) -> bool:
    """Idempotent extract+eval skip-completed predicate. Keys on BOTH final
    artifacts so a cell killed mid-write is NOT accepted as complete (#667-class
    partial-cell safety): the store ``tensors.pt`` (the P2.3 fail-loud deliverable,
    written at the END of ``extract_and_eval_cell``'s extract worker) AND a
    non-empty eval registry dir (the gen worker's raw completions)."""
    store_done = (
        C.STORE_ROOT / (cell.eval_key + ("_smoke" if smoke else "")) / "tensors.pt"
    ).exists()
    eval_dir = C.EVAL_ROOT / ("registry_smoke" if smoke else "registry") / cell.eval_key
    eval_done = eval_dir.exists() and any(eval_dir.iterdir())
    return store_done and eval_done


# ── #664/#689 fix (a): per-cell incremental upload + EXACT-set Hub presence ───
# Both the skip guard and the auto-terminate gate need the EXACT set of files a
# COMPLETE cell has on HF (S1), never a prefix or count. These helpers are
# stubbed for TDD round-1 (test imports); bodies land in round 2.


def _expected_eval_files(cell: C.Cell) -> set[str]:
    """The EXACT set of eval-JSON basenames a COMPLETE cell has under its
    raw-completions prefix (S1). Mirrors the gen phase's own iteration
    (``issue664_eval._judging_surface``) so it stays in lock-step with what gen
    writes; excludes the marker column (its DV is the slot stats, not a
    completions JSON). Deterministic per cell — NOT a fixed count.

    ``_judging_surface`` yields ``(context_id, column)`` tuples and ``gen_cell``
    writes ``completions__<column>__<context_id>.json`` (issue664_eval L215), so
    the basename is built by unpacking ``(ctx, col)`` and emitting
    ``completions__{col}__{ctx}.json`` (== the gen write path)."""
    from importlib import import_module

    ev = import_module("issue664_eval")
    return {
        f"completions__{col}__{ctx}.json"
        for (ctx, col) in ev._judging_surface(cell)
        if col != "marker"
    }


def _expected_store_files() -> set[str]:
    """The EXACT set of store-tensor basenames a COMPLETE cell has under its store
    prefix (S1): the extract worker writes exactly ``tensors.pt`` + ``meta.json``.
    ``tensors.pt`` is the PRIMARY deliverable; its absence MUST fail the
    completeness check (the #521 trap)."""
    return {"tensors.pt", "meta.json"}


def _is_marker_cell(cell: C.Cell) -> bool:
    """True iff this cell's behavior is the marker implant. ONLY marker cells
    write ``marker_slot_stats.json`` (issue664_extract_store L361), so the
    marker-slot HF surface + the readability hydrate are gated on this.
    ``getattr`` so a cell-shaped object without a ``behavior`` attr (the
    backend_poll wedge-gate test doubles, which carry only ``eval_key``) reads as
    non-marker — the marker-slot completeness requirement is then vacuous, which
    is correct for the auto-terminate gate's raw+store-only inputs check."""
    return getattr(cell, "behavior", None) == "marker"


def _marker_slot_local_path(cell: C.Cell, *, smoke: bool) -> Path:
    """The local ``marker_slot_stats.json`` path the extract worker writes for a
    marker cell (issue664_extract_store L362). Keyed on the SEED-QUALIFIED
    ``eval_key`` (+ the ``_smoke`` suffix), so it matches what the readability
    assert reads."""
    suffix = "_smoke" if smoke else ""
    return C.EVAL_ROOT / "marker_slot" / (cell.eval_key + suffix) / "marker_slot_stats.json"


def _expected_marker_slot_files(cell: C.Cell) -> set[str]:
    """The EXACT set of marker-slot basenames a COMPLETE marker cell has on HF
    (S1): the extract worker writes exactly ``marker_slot_stats.json`` for a
    marker cell. A NON-marker cell writes none, so this is empty (and the
    completeness check is a no-op for non-marker cells)."""
    return {"marker_slot_stats.json"} if _is_marker_cell(cell) else set()


def _classify_cell_hub_state(cell: C.Cell, files: set[str]) -> str:
    """Per-cell three-state HF presence (M1) from a PRE-FETCHED listing: 'complete'
    (both kinds' EXACT sets present), 'partial' (>=1 file of one kind present but
    not the full set), or 'absent' (no files under either prefix). Shared by the
    skip guard and the auto-terminate gate so they cannot diverge (S1 exact-set).

    #689 blocker-1 (fix a1): for a MARKER cell, "complete" ALSO requires the
    marker-slot stats on HF — otherwise a fresh auto-migrated pod SKIPs the cell
    (A2 ``_cell_done_anywhere``) yet ``_marker_readability_assert`` has no local
    ``marker_slot_stats.json`` to read and crashes (``checked == 0``). Making the
    slot stats part of the per-cell HF surface keeps the SKIP-and-hydrate path
    coherent: the file is on HF for every HF-complete marker cell."""
    raw_prefix = f"{C.HF_RAW_COMPLETIONS_PREFIX}/{cell.eval_key}/"
    store_prefix = f"{C.HF_STORE_PREFIX}/{cell.eval_key}/"
    marker_prefix = f"{C.HF_MARKER_SLOT_PREFIX}/{cell.eval_key}/"
    have_eval = {p[len(raw_prefix) :] for p in files if p.startswith(raw_prefix)}
    have_store = {p[len(store_prefix) :] for p in files if p.startswith(store_prefix)}
    have_marker = {p[len(marker_prefix) :] for p in files if p.startswith(marker_prefix)}
    eval_ok = _expected_eval_files(cell).issubset(have_eval)
    store_ok = _expected_store_files().issubset(have_store)
    marker_ok = _expected_marker_slot_files(cell).issubset(have_marker)  # vacuous if non-marker
    if eval_ok and store_ok and marker_ok:
        return "complete"
    if have_eval or have_store or have_marker:  # something present but not the full set
        return "partial"
    return "absent"


def _cell_artifacts_on_hub(cell: C.Cell) -> bool:
    """True iff this cell's EXACT expected eval-JSON set AND store-tensor set are
    BOTH fully present on the Hub (S1). FRESH listing via the Python Hub API
    (never the ``hf`` CLI — upload-policy). A partial cell (mid-``upload_folder``
    crash, one artifact-kind missing) reads as NOT complete, so it is re-uploaded
    and never silently skipped / passed by the auto-terminate gate."""
    import huggingface_hub

    files = set(
        huggingface_hub.list_repo_files(C.HF_DATA_REPO, repo_type="dataset", revision="main")
    )
    return _classify_cell_hub_state(cell, files) == "complete"


def _cell_hub_state(cell: C.Cell) -> str:
    """Per-cell three-state HF presence (M1) computed from ONE fresh listing.
    Callers classifying many cells pass the listing into
    :func:`_classify_cell_hub_state` directly to avoid N round-trips."""
    import huggingface_hub

    files = set(
        huggingface_hub.list_repo_files(C.HF_DATA_REPO, repo_type="dataset", revision="main")
    )
    return _classify_cell_hub_state(cell, files)


def _cell_done_anywhere(cell: C.Cell, *, smoke: bool) -> bool:
    """P2 resume-skip predicate (A2). A cell is done if its final artifacts are on
    the pod volume (the fast local path, ``_cell_extract_eval_done``) OR its EXACT
    expected file set is already complete on HF (the fresh-pod-after-auto-migrate
    path, fix (b)). Smoke never consults HF (per-cell upload is smoke-skipped)."""
    if _cell_extract_eval_done(cell, smoke=smoke):
        return True
    if smoke:
        return False
    return _cell_artifacts_on_hub(cell)


def _upload_cell_artifacts(cell: C.Cell, *, smoke: bool) -> None:
    """Per-cell incremental upload (checkpoint-per-phase, fix (a)). ONE
    ``upload_folder`` commit per artifact-kind (HF 256-commits/hr cap). Idempotent:
    skips when the EXACT expected file set is already on the Hub. Fail-loud
    EXACT-file-set Hub-verify before returning (``RuntimeError`` naming the missing
    file). Smoke short-circuits (no listing, no upload)."""
    if smoke:
        logger.info("[p2-upload] smoke: skipping per-cell HF upload")
        return
    if _cell_artifacts_on_hub(cell):
        logger.info("[p2-upload] %s already complete on Hub; skipping", cell.eval_key)
        return
    import huggingface_hub

    api = huggingface_hub.HfApi()
    # eval JSONs: eval_results/issue_664/registry/<cell>/ -> raw_completions/<cell>/
    eval_dir = C.EVAL_ROOT / "registry" / cell.eval_key
    if eval_dir.exists() and any(eval_dir.iterdir()):
        api.upload_folder(
            folder_path=str(eval_dir),
            repo_id=C.HF_DATA_REPO,
            repo_type="dataset",
            path_in_repo=f"{C.HF_RAW_COMPLETIONS_PREFIX}/{cell.eval_key}",
            allow_patterns=["completions__*.json", "completion_logp.json"],
            commit_message=f"[i664 per-cell] eval JSONs {cell.eval_key}",
        )
    # store tensors: trained_store/<cell>/ -> theory_assumptions/.../issue664/<cell>/
    store_dir = C.STORE_ROOT / cell.eval_key
    if store_dir.exists() and any(store_dir.iterdir()):
        api.upload_folder(
            folder_path=str(store_dir),
            repo_id=C.HF_DATA_REPO,
            repo_type="dataset",
            path_in_repo=f"{C.HF_STORE_PREFIX}/{cell.eval_key}",
            commit_message=f"[i664 per-cell] store tensors {cell.eval_key}",
        )
    # marker-slot stats (MARKER cells only): EVAL_ROOT/marker_slot/<cell>/ ->
    # issue664_leakage_fleet/marker_slot/<cell>/. #689 blocker-1 (fix a1): part of
    # the per-cell HF surface so a fresh auto-migrated pod can hydrate the A7
    # readability input instead of crashing on a local-absent marker_slot_stats.json.
    if _is_marker_cell(cell):
        slot_path = _marker_slot_local_path(cell, smoke=False)
        if slot_path.exists():
            api.upload_folder(
                folder_path=str(slot_path.parent),
                repo_id=C.HF_DATA_REPO,
                repo_type="dataset",
                path_in_repo=f"{C.HF_MARKER_SLOT_PREFIX}/{cell.eval_key}",
                allow_patterns=["marker_slot_stats.json"],
                commit_message=f"[i664 per-cell] marker slot stats {cell.eval_key}",
            )
    if not _cell_artifacts_on_hub(cell):  # FRESH-listing EXACT-set verify, fail-loud
        raise RuntimeError(
            f"[p2-upload] per-cell upload verify FAILED for {cell.eval_key}: "
            f"expected eval-JSON set {sorted(_expected_eval_files(cell))}, store set "
            f"{sorted(_expected_store_files())} (incl. tensors.pt), and/or marker-slot set "
            f"{sorted(_expected_marker_slot_files(cell))} not fully on Hub after upload_folder"
        )
    logger.info(
        "[p2-upload] %s eval JSONs + store tensors%s uploaded + verified",
        cell.eval_key,
        " + marker slot stats" if _is_marker_cell(cell) else "",
    )


def _one_cell_base_argv(cell: C.Cell, mode_flag: str, gpu_id: int, *, smoke: bool) -> list[str]:
    """argv for a one-cell WaveDispatcher subprocess (self-reinvoke of THIS script).

    The subprocess runs ``issue664_dispatch.py <mode_flag> --behavior ... --gpu-id g
    [--smoke]`` so the in-process op runs in its OWN process (fresh cuInit per cell
    -> the CVD launcher-env pin actually takes; gotchas.md cuInit-freeze). ``--smoke``
    is threaded through so the worker rebinds its smoke roots identically."""
    argv = [
        sys.executable,
        str(C.REPO / "scripts/issue664_dispatch.py"),
        mode_flag,
        "--behavior", cell.behavior,
        "--source", cell.source,
        "--arm", cell.arm,
        "--dose", cell.dose,
        "--seed", str(cell.seed),
        "--gpu-id", str(gpu_id),
    ]  # fmt: skip
    if smoke:
        argv.append("--smoke")
    return argv


def _cvd_env(gpu_id: int) -> dict[str, str]:
    """Per-cell launcher env: CVD pin (gotchas.md cuInit-freeze) + the vLLM
    spawn-method (gotchas #628 fork-poison; mirrors issue667_dispatch.py:534)."""
    return {
        "CUDA_VISIBLE_DEVICES": str(gpu_id),
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
    }


def _train_cell_cmd(cell: C.Cell, gpu_id: int, *, smoke: bool) -> CellCmd:
    """Build the P2.1 one-cell train launch spec (CVD pinned in the launcher env)."""
    log = _log_dir() / f"issue-664-train-{cell.eval_key}{'_smoke' if smoke else ''}.log"
    return CellCmd(
        cell_key=cell.eval_key,
        argv=_one_cell_base_argv(cell, "--train-one-cell", gpu_id, smoke=smoke),
        env=_cvd_env(gpu_id),
        log_path=log,
        gpu_id=gpu_id,
    )


def _extract_eval_cell_cmd(cell: C.Cell, gpu_id: int, *, smoke: bool) -> CellCmd:
    """Build the P2.2 one-cell extract+eval launch spec (CVD pinned)."""
    log = _log_dir() / f"issue-664-extracteval-{cell.eval_key}{'_smoke' if smoke else ''}.log"
    return CellCmd(
        cell_key=cell.eval_key,
        argv=_one_cell_base_argv(cell, "--extract-eval-one-cell", gpu_id, smoke=smoke),
        env=_cvd_env(gpu_id),
        log_path=log,
        gpu_id=gpu_id,
    )


def _run_one_cell(args) -> int:
    """One-cell subprocess worker (WaveDispatcher invokes THIS via the mode flags).

    Runs EXACTLY one cell's in-process op (``train_cell`` or
    ``extract_and_eval_cell``) for the cell reconstructed from the tuple args, then
    exits. CVD is already pinned in the launcher env by the parent WaveDispatcher;
    ``args.gpu_id`` is threaded through as the matching in-process value."""
    _require_credentials()
    for name in ("behavior", "source", "arm", "dose"):
        if getattr(args, name) is None:
            raise SystemExit(f"--{name} is required for a one-cell subprocess worker")
    cell = C.Cell(args.behavior, args.source, args.arm, args.dose, seed=args.seed)
    if args.train_one_cell:
        train_cell(cell, smoke=args.smoke, gpu_id=args.gpu_id)
    else:  # extract_eval_one_cell
        adapter_dir = _adapter_dir_for(cell, smoke=args.smoke)
        extract_and_eval_cell(cell, adapter_dir, smoke=args.smoke, gpu_id=args.gpu_id)
    return 0


# ── P2.3 upload raw completions + store tensors ───────────────────────────────
def upload_artifacts(cells: list[C.Cell], *, smoke: bool) -> None:
    """Push raw completions + store tensors + the source-side baseline-propensity
    covariate to the HF data repo (adapters were pushed by train_lora). Fail-loud
    per upload-policy."""
    if smoke:
        logger.info("[p3-upload] smoke: skipping HF upload")
        return
    _upload_raw_completions(cells)
    _upload_store_tensors(cells)
    _upload_baseline_propensity(cells)


def _upload_baseline_propensity(cells: list[C.Cell]) -> None:
    """Upload the source-side BASE-model behavior-rate covariate (plan §4) so it
    SURVIVES pod teardown.

    ``_write_baseline_propensity`` (phase0) writes the per-(source, behavior)
    judged-rate aggregate ``onpolicy_cache/baseline_propensity.json`` + the raw
    base completions (and judge save_raw scores) under
    ``onpolicy_cache/baseline_raw/<behavior>__<source>.json``. Neither
    ``_upload_raw_completions`` (walks ``EVAL_ROOT/registry``) nor
    ``_upload_store_tensors`` (walks ``STORE_ROOT/<cell>``) touches this cache, so
    without THIS call Phase-3/4 cannot derive the source-side base-rate covariate
    after teardown -- the #521-class trap (#664 post-pivot r1 blocker).

    Uploads the aggregate + every ``baseline_raw/*.json`` to
    ``issue664/baseline_propensity/`` (judge ``.cache`` excluded), then verifies on
    a FRESH Hub listing. FAIL-LOUD: the aggregate must exist, and every
    (content-behavior, source) baseline read this run was supposed to produce
    (the SELECTED cells whose behavior ∈ CONTENT_BEHAVIORS, per
    ``realized_grid()`` x CONTENT_BEHAVIORS) must have a local raw file -- a
    missing one refuses to reach the Hub-verify step."""
    from huggingface_hub import list_repo_files

    from explore_persona_space.orchestrate import hub

    prefix = C.HF_BASELINE_PROPENSITY_PREFIX  # issue664/baseline_propensity
    agg = CACHE_ROOT / "baseline_propensity.json"
    raw_root = CACHE_ROOT / "baseline_raw"
    if not agg.exists():
        raise RuntimeError(
            f"[p3-upload] baseline_propensity.json MISSING at {agg} -- the phase0 "
            "source-side base-prior read (plan §4) never ran; refusing to terminate "
            "without the registered covariate (the #521 trap)."
        )

    # The (content-behavior, source) baseline reads this run was supposed to
    # produce: SELECTED cells whose behavior carries a judged column. A missing
    # local raw file for any of these is FAIL-LOUD (not a warn-and-continue).
    expected = sorted({(c.behavior, c.source) for c in cells if c.behavior in C.CONTENT_BEHAVIORS})
    missing_local: list[str] = []
    for behavior, src in expected:
        raw_path = raw_root / f"{behavior}__{src}.json"
        if not raw_path.exists():
            missing_local.append(f"{behavior}__{src} ({raw_path})")
    if missing_local:
        raise RuntimeError(
            "[p3-upload] baseline-propensity raw completions MISSING for "
            f"{len(missing_local)} expected (content-behavior, source) pair(s): "
            f"{missing_local}. Refusing to reach the Hub-verify step with an "
            "incomplete source-side covariate -- investigate phase0 "
            "_write_baseline_propensity."
        )

    # Upload the aggregate + every baseline_raw/*.json (judge .cache excluded:
    # rglob limited to direct *.json children of baseline_raw, which never
    # includes the .cache subdir's contents).
    to_upload: list[tuple[Path, str]] = [(agg, f"{prefix}/{agg.name}")]
    for f in sorted(raw_root.glob("*.json")):
        to_upload.append((f, f"{prefix}/baseline_raw/{f.name}"))
    for local, path_in_repo in to_upload:
        hub._upload(
            local,
            repo_id=C.HF_DATA_REPO,
            repo_type="dataset",
            path_in_repo=path_in_repo,
            upload_as_file=True,  # gotchas: per-file _upload needs this
        )
    n_expected = len(to_upload)
    # verify on a FRESH listing (Python Hub API, never the hf CLI).
    landed = [
        p for p in list_repo_files(C.HF_DATA_REPO, repo_type="dataset") if p.startswith(prefix)
    ]
    if len(landed) < n_expected:
        raise RuntimeError(
            f"[p3-upload] baseline-propensity verify FAILED: {len(landed)} on Hub < "
            f"{n_expected} expected under {prefix}"
        )
    logger.info(
        "[p3-upload] %d baseline-propensity file(s) uploaded + verified -> %s/%s",
        n_expected,
        C.HF_DATA_REPO,
        prefix,
    )


def _upload_raw_completions(cells: list[C.Cell]) -> None:
    """Upload the eval-gen raw completions to the canonical HF data-repo path.

    The eval gen writes ``eval_results/issue_664/registry/<cell>/completions__
    <col>__<ctx>.json`` -- a NON-canonical shape that the helper's
    ``rglob('raw_completions.json')`` does NOT match (the #528 silent-loss
    class). So we walk the ACTUAL write path and upload per-file with
    ``upload_as_file=True`` to ``issue664_leakage_fleet/raw_completions/<rel>``,
    then verify the EXACT expected file set per selected cell landed on the Hub
    before teardown (#689 blocker-4: an exact-set check, not a count floor).

    #664/#689 fix (a): the per-cell incremental hook (``_upload_cell_artifacts``)
    already pushed each cell as it completed, so P3 is now an idempotent SAFETY
    SWEEP -- it skips cells already complete on the Hub (EXACT-set, not
    prefix-presence) and re-uploads only what the per-cell hook missed. A2: on a
    fresh pod after a wedge auto-migrate the local registry is EMPTY for cells
    that live only on HF, so an empty-local registry where every selected cell is
    already complete on HF is the all-on-HF NO-OP, not the genuine
    nothing-was-produced error."""
    from huggingface_hub import list_repo_files

    from explore_persona_space.orchestrate import hub

    prefix = C.HF_RAW_COMPLETIONS_PREFIX  # issue664_leakage_fleet/raw_completions
    reg_root = C.EVAL_ROOT / "registry"
    files = sorted(reg_root.rglob("completions__*.json"))
    if not files:
        # A2: an empty LOCAL registry is the NORMAL fresh-pod case for cells that
        # are already complete on HF (the wedge auto-migrate path). If every
        # selected cell is complete on the Hub, this is a no-op success -- only a
        # genuinely-empty-AND-not-on-HF run is the error the raise below names.
        on_hub = set(list_repo_files(C.HF_DATA_REPO, repo_type="dataset", revision="main"))
        if cells and all(_classify_cell_hub_state(c, on_hub) == "complete" for c in cells):
            logger.info(
                "[p3-upload] no local raw completions but every selected cell is "
                "complete on HF (fresh-pod resume); P3 raw-completions is a no-op"
            )
            return
        raise RuntimeError(
            f"[p3-upload] NO raw completions under {reg_root} -- the eval gen "
            "phase produced nothing; refusing to terminate with empty buckets"
        )
    # Per-cell exact-set skip guard, computed ONCE from a single fresh listing: a
    # cell whose EXACT expected set is already on the Hub was pushed by the
    # per-cell hook, so its local files are re-upload noise.
    on_hub = set(list_repo_files(C.HF_DATA_REPO, repo_type="dataset", revision="main"))
    complete_keys = {c.eval_key for c in cells if _classify_cell_hub_state(c, on_hub) == "complete"}
    n_expected = 0
    for f in files:
        rel = f.relative_to(reg_root).as_posix()  # <cell>/completions__<col>__<ctx>.json
        cell_key = rel.split("/", 1)[0]
        if cell_key in complete_keys:
            continue  # already complete on HF (per-cell hook) -- skip the re-upload
        hub._upload(
            f,
            repo_id=C.HF_DATA_REPO,
            repo_type="dataset",
            path_in_repo=f"{prefix}/{rel}",
            upload_as_file=True,  # gotchas: per-file _upload needs this
        )
        n_expected += 1
    # #689 blocker-4 (p3-raw-exact-verify): verify on a FRESH listing (Python Hub
    # API, never the hf CLI) with an EXACT FILE-SET check per selected cell, NOT a
    # count floor. A count check (``len(landed) < n_expected``) can PASS with an
    # incomplete selected cell whenever unrelated files under the prefix inflate the
    # count -- the same prefix-vs-exact-set hole the M2 store verify closes. Build
    # the expected raw path set for EVERY selected cell (``_expected_eval_files``,
    # the completions basenames the gen phase writes), then raise naming the exact
    # missing (cell, file) pairs.
    landed = {
        p for p in list_repo_files(C.HF_DATA_REPO, repo_type="dataset") if p.startswith(prefix)
    }
    missing: list[str] = []
    for cell in cells:
        cell_prefix = f"{prefix}/{cell.eval_key}/"
        for basename in _expected_eval_files(cell):
            expected_path = f"{cell_prefix}{basename}"
            if expected_path not in landed:
                missing.append(expected_path)
    if missing:
        raise RuntimeError(
            f"[p3-upload] raw-completions EXACT-set verify FAILED: {len(missing)} expected "
            f"completion file(s) missing on the Hub under {prefix}: {sorted(missing)}"
        )
    logger.info(
        "[p3-upload] %d raw-completion files uploaded this sweep; EXACT-set verified "
        "(%d selected cells) -> %s/%s",
        n_expected,
        len(cells),
        C.HF_DATA_REPO,
        prefix,
    )


def _upload_store_tensors(cells: list[C.Cell]) -> None:
    """Mirror each selected cell's trained-store (the Phase-3/4 PRIMARY deliverable)
    to the HF data repo. #664 round-2 M5: a MISSING ``tensors.pt`` for a selected
    cell is FAIL-LOUD, NOT a warn-and-continue -- the prior `logger.warning; continue`
    let the dispatcher reach `[phase=done]` with incomplete primary deliverables
    mirrored to HF (the #521 trap variant: a downstream control becomes permanently
    unrunnable, discovered only post-teardown).

    #664/#689 fix (a) + M2: the per-cell incremental hook already pushed each
    cell's store tensors, so this is an idempotent SAFETY SWEEP. A2: a cell with
    NO local ``tensors.pt`` that is already COMPLETE on HF (the fresh-pod
    auto-migrate path) is NOT missing -- it was uploaded per-cell; only a cell
    that is neither local-complete nor HF-complete is a real miss. M2 (the gap
    this closes): the prior helper had only a local ``tp.exists()`` check and NO
    fresh-listing Hub verify, so a 429-throttled / transient upload could leave
    store tensors uploaded-but-not-landed and still return clean -- a data-loss
    hole the irreversible auto-terminate (fix (b)) is gated on. Add a
    fresh-listing EXACT-set Hub verify after the upload loop."""
    from huggingface_hub import list_repo_files

    from explore_persona_space.orchestrate import hub

    missing: list[str] = []
    for cell in cells:
        cell_dir = C.STORE_ROOT / cell.eval_key
        tp = cell_dir / "tensors.pt"
        if not tp.exists():
            # A2: a cell complete on HF (fresh-pod path) with no local files is NOT
            # missing -- it was already uploaded per-cell. Only a cell that is
            # neither local-complete nor HF-complete is a real miss.
            if _cell_artifacts_on_hub(cell):
                continue
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
    # M2 (#664/#689): FRESH-listing EXACT-set verify -- every selected cell's store
    # tensors.pt + meta.json must be present on the Hub before teardown. The prior
    # helper had NO post-upload Hub verify (only the local tp.exists() check above),
    # so a silently-dropped upload could pass; the irreversible auto-terminate gate
    # in backend_poll is gated on this presence.
    on_hub = set(list_repo_files(C.HF_DATA_REPO, repo_type="dataset", revision="main"))
    not_landed: list[str] = []
    for cell in cells:
        store_prefix = f"{C.HF_STORE_PREFIX}/{cell.eval_key}/"
        have = {p[len(store_prefix) :] for p in on_hub if p.startswith(store_prefix)}
        if not _expected_store_files().issubset(have):
            not_landed.append(f"{cell.eval_key} (have={sorted(have)})")
    if not_landed:
        raise RuntimeError(
            f"[p3-upload] store-tensors Hub-verify FAILED: {len(not_landed)} cell(s) "
            f"missing tensors.pt/meta.json on the Hub after upload: {not_landed}"
        )
    logger.info(
        "[p3-upload] store tensors uploaded + Hub-verified -> %s/%s",
        C.HF_DATA_REPO,
        C.HF_STORE_PREFIX,
    )


def _hydrate_marker_slot_stats_from_hf(marker_cells: list[C.Cell]) -> None:
    """Download each marker cell's ``marker_slot_stats.json`` from HF into its
    local path when the local file is ABSENT but the cell is HF-complete (#689
    blocker-1, fix a1). This is the fresh-auto-migrated-pod recovery path: P2
    SKIPped these cells (A2 ``_cell_done_anywhere`` saw them complete on HF), so
    the readability assert has no local file to read. ONE fresh ``list_repo_files``
    listing classifies all cells; only the HF-complete-but-local-absent ones are
    hydrated. Best-effort per cell (a download failure leaves the file absent, so
    the cell is skipped exactly as before) -- never raises; the caller's
    ``checked == 0`` guard is the loud failure when the surface is genuinely
    broken."""
    import huggingface_hub

    try:
        on_hub = set(
            huggingface_hub.list_repo_files(C.HF_DATA_REPO, repo_type="dataset", revision="main")
        )
    except Exception as exc:
        logger.warning("[a7-assert] HF listing for marker-slot hydrate failed (%s); skipping", exc)
        return
    for cell in marker_cells:
        local = _marker_slot_local_path(cell, smoke=False)
        if local.exists():
            continue
        if _classify_cell_hub_state(cell, on_hub) != "complete":
            continue  # not on HF -> nothing to hydrate (cell is skipped + may trip checked==0)
        remote = f"{C.HF_MARKER_SLOT_PREFIX}/{cell.eval_key}/marker_slot_stats.json"
        try:
            downloaded = huggingface_hub.hf_hub_download(
                repo_id=C.HF_DATA_REPO,
                repo_type="dataset",
                filename=remote,
                revision="main",
            )
            local.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(downloaded, local)
            logger.info("[a7-assert] hydrated marker slot stats from HF -> %s", local)
        except Exception as exc:
            logger.warning(
                "[a7-assert] marker-slot hydrate failed for %s (%s); leaving local absent",
                cell.eval_key,
                exc,
            )


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
    # #689 blocker-1 (fix a1): FRESH-POD HYDRATE. On a fresh auto-migrated pod the
    # marker cells live only on HF (A2 _cell_done_anywhere skipped them in P2), so
    # the LOCAL marker_slot_stats.json is absent -- without this the loop below
    # would `continue` past every cell, hit `checked == 0`, and RAISE (the
    # production-path crash this fix closes). For each marker cell whose local
    # slot file is absent but is HF-complete, download marker_slot_stats.json from
    # HF into the local path so the assertion below reads it. Smoke never consults
    # HF (the per-cell upload is smoke-skipped). Best-effort: a hydrate miss leaves
    # the file absent (the cell is then skipped as before) -- the `checked == 0`
    # raise still fires if NO marker cell could be read, which is the correct loud
    # failure when the marker-slot HF surface is genuinely broken.
    if not smoke:
        _hydrate_marker_slot_stats_from_hf(marker_cells)
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


def _wave_gpu_id(args, g: int) -> int:
    """Resolve the wave-assigned gpu index ``g`` to the actual device for a cell.

    #676 round-2 (Point C / Must-Fix #3): on the single-GPU path (``--n-gpus 1``,
    the default) the wave assigns every cell ``g == 0``, but ``--gpu-id`` is the
    single-GPU device SELECTOR (plan §3.6: "retained as the single-GPU device
    selector (passed through when n_gpus==1)"). So when ``n_gpus == 1`` the cell
    runs on ``args.gpu_id`` — both the in-process ``--gpu-id`` arg AND the launcher
    ``CUDA_VISIBLE_DEVICES`` pin. For ``n_gpus > 1`` the wave-assigned ``g`` (0..N-1)
    is authoritative; a nonzero ``--gpu-id`` is rejected up front by
    :func:`_validate_gpu_args` as incoherent with multi-GPU wave assignment.
    """
    return args.gpu_id if args.n_gpus == 1 else g


def _validate_gpu_args(args) -> None:
    """Reject an incoherent ``--gpu-id`` / ``--n-gpus`` combination (Point C).

    A nonzero ``--gpu-id`` is the SINGLE-GPU device selector; it is meaningless
    when ``--n-gpus > 1`` (the wave assigns devices 0..N-1 itself). Fail loud rather
    than silently ignore it — a stray ``--gpu-id 1 --n-gpus 4`` would otherwise read
    as "every cell on its wave-assigned gpu" with the selector quietly dropped.
    """
    if args.n_gpus > 1 and args.gpu_id != 0:
        raise SystemExit(
            f"--gpu-id {args.gpu_id} is the single-GPU device selector and is "
            f"incoherent with --n-gpus {args.n_gpus} (the wave assigns devices "
            f"0..{args.n_gpus - 1}). Pass --gpu-id only on the single-GPU path "
            f"(--n-gpus 1, the default)."
        )


# ── Orchestration ─────────────────────────────────────────────────────────────
def run_all(args) -> None:
    _require_credentials()
    _validate_gpu_args(args)
    cells = _select_cells(args)
    logger.info("[dispatch] %d cells selected (smoke=%s)", len(cells), args.smoke)

    if args.phase in ("all", "p0"):
        phase0(args)
    if args.phase == "p0":
        return

    # #664 round-6: exclude cells dropped below the on-policy yield floor at P2.0 from
    # EVERY downstream phase -- a dropped cell has no training mix, so training /
    # extraction / eval / manifest / upload must skip it (otherwise the fail-loud
    # missing-artifact asserts in _upload_* would crash on never-produced files).
    # Re-runs entering at --phase p1/p2/p3 read the manifest written by the p0 process.
    cells = _drop_filtered(cells)
    logger.info("[dispatch] %d cells after drop-filter (smoke=%s)", len(cells), args.smoke)

    if args.phase in ("all", "p1"):
        phase_log("p1_train")
        # P2.1 train cells in GPU-parallel waves (#676). --n-gpus 1 (default) is the
        # unchanged serial single-GPU path: one cell per wave, all on gpu 0. Each
        # cell trains in its OWN subprocess (fresh cuInit -> CVD launcher pin takes).
        WaveDispatcher(
            n_gpus=args.n_gpus,
            cell_key=lambda c: c.eval_key,
            is_done=lambda c: _train_done(c, smoke=args.smoke),
            build_cmd=lambda c, g: _train_cell_cmd(c, _wave_gpu_id(args, g), smoke=args.smoke),
            dry_run=args.dry_run,
        ).run(cells, cwd=C.REPO)
    if args.phase == "p1":
        return

    if args.phase in ("all", "p2"):
        phase_log("p2_extract_eval")
        # P2.2 extract+eval cells in GPU-parallel waves (#676); each cell's merge +
        # extract + eval-gen runs in its own subprocess (distinct merged dir +
        # distinct CVD per concurrent cell).
        WaveDispatcher(
            n_gpus=args.n_gpus,
            cell_key=lambda c: c.eval_key,
            # A2 (#664/#689): a cell is done if its final artifacts are on the pod
            # volume OR already complete on HF (the fresh-pod-after-auto-migrate
            # path) -- so a fresh pod after a wedge auto-migrate SKIPS HF-complete
            # cells instead of re-running them.
            is_done=lambda c: _cell_done_anywhere(c, smoke=args.smoke),
            build_cmd=lambda c, g: _extract_eval_cell_cmd(
                c, _wave_gpu_id(args, g), smoke=args.smoke
            ),
            dry_run=args.dry_run,
        ).run(cells, cwd=C.REPO)
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
    """Write the registry manifest for ONLY the SELECTED cells (#664 round-2 N2:
    the worker previously wrote the full ``C.realized_grid()`` regardless of the
    --cells subset, so a subset/smoke manifest described cells the run never
    generated and the verifier cross-check would expect missing tuples)."""
    cmd = [
        sys.executable,
        str(C.REPO / "scripts/issue664_eval.py"),
        "--phase", "manifest",
        "--cells-keys", ",".join(c.eval_key for c in cells),
    ]  # fmt: skip
    if smoke:
        cmd.append("--smoke")
    subprocess.run(cmd, check=True, cwd=C.REPO, env={**os.environ})


def main() -> int:
    ap = argparse.ArgumentParser(description="Issue #664 Phase-2 fleet driver.")
    ap.add_argument("--phase", default="all", choices=["all", "p0", "p1", "p2", "p3"])
    ap.add_argument("--cells", type=int, default=None, help="cap the cell count (smoke: 1)")
    ap.add_argument("--gpu-id", type=int, default=0)
    ap.add_argument(
        "--n-gpus",
        type=int,
        default=1,
        help="number of GPUs to fan the P2.1/P2.2 cell fleet across (default 1 = the "
        "unchanged serial single-GPU path; pass the provisioned GPU count to "
        "parallelize, #676 WaveDispatcher).",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="log the planned wave launches without executing any cell subprocess "
        "(#676 WaveDispatcher dry-run).",
    )
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument(
        "--live-judge-smoke",
        action="store_true",
        help="in --smoke, run a tiny REAL Batch-API judge slice on one content cell "
        "to exercise the production judge branch (#664 round-2 B5)",
    )
    # One-cell subprocess entrypoints — how WaveDispatcher.run() invokes the
    # in-process per-cell logic in its OWN process (fresh cuInit per cell so the
    # CVD launcher-env pin actually takes; gotchas.md). NOT for human use.
    ap.add_argument(
        "--train-one-cell",
        action="store_true",
        help="(internal) train EXACTLY one cell (the cell named by "
        "--behavior/--source/--arm/--dose/--seed) and exit; the WaveDispatcher "
        "subprocess worker for P2.1.",
    )
    ap.add_argument(
        "--extract-eval-one-cell",
        action="store_true",
        help="(internal) extract+eval EXACTLY one cell and exit; the WaveDispatcher "
        "subprocess worker for P2.2.",
    )
    ap.add_argument("--behavior")
    ap.add_argument("--source")
    ap.add_argument("--arm")
    ap.add_argument("--dose")
    ap.add_argument("--seed", type=int, default=C.DEFAULT_SEED)
    args = ap.parse_args()

    # One-cell subprocess workers: reconstruct the single Cell from the tuple args
    # and run exactly that cell's in-process op, then exit. These run with
    # CUDA_VISIBLE_DEVICES pinned in the launcher env by the parent WaveDispatcher.
    if args.train_one_cell or args.extract_eval_one_cell:
        return _run_one_cell(args)

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
        # exclude dropped cells from the repro card -- adapters / wandb runs only exist
        # for the cells that were actually trained (#664 round-6).
        sel = _drop_filtered(_select_cells(args))
        dropped_keys = sorted(_dropped_cell_keys())
        n = len(sel)
        write_sentinel(
            "epm:results",
            f"issue664 Phase-2 fleet complete ({n} cells, smoke={args.smoke}"
            + (f", {len(dropped_keys)} dropped below yield floor" if dropped_keys else "")
            + ")",
            extra={
                "gate": "results",
                "blocks_pipeline": False,
                "dropped_cells_below_yield_floor": dropped_keys,
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
