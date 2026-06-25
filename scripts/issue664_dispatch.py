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
        # Canary: the marker x default x contrastive x dose-1 cell exercises the
        # band-stop path; it is the smoke-architecture canary (§ smoke parity).
        canary = C.Cell("marker", "default", "contra", "d1")
        ordered = [c for c in grid if c.slug == canary.slug] + [
            c for c in grid if c.slug != canary.slug
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


def _write_responses_cache(kind: str, ctx_key: str, mapping: dict[str, str]) -> None:
    p = CACHE_ROOT / kind / f"{ctx_key}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {**C.repro_meta(seed=C.DEFAULT_SEED), "kind": kind, "ctx_key": ctx_key}
    payload["responses"] = {q: {"response": r} for q, r in mapping.items()}
    p.write_text(json.dumps(payload, ensure_ascii=False))


def _write_pool(behavior: str, questions: list[str], *, smoke: bool) -> None:
    p = CACHE_ROOT / "pools" / f"{behavior}{'_smoke' if smoke else ''}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"behavior": behavior, "questions": questions}))


def _marker_question_pool(smoke: bool) -> list[str]:
    import issue404_common as i4

    if smoke:
        return C.SMOKE_QUESTIONS
    return i4.fetch_preregistered_probes(48)


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
    finally:
        _teardown_vllm(llm)

    # baseline propensity read (cheap; folded into P2.0) -- a placeholder summary
    # the analyzer reads; the per-source on-policy yields are already in the
    # elicitation caches (build asserts the 80% floor at mix-build time).
    _write_baseline_propensity(sources, behaviors, smoke=smoke)

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
        logger.info("[p0-build] %s", cell.slug)
        subprocess.run(cmd, check=True, cwd=C.REPO, env={**os.environ})  # explicit env


def _render(messages: list[dict]) -> str:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(C.QWEN_ID, trust_remote_code=True)
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def _elicit_sycophancy(llm, sources, *, smoke: bool) -> None:
    """#612 tier-2 instruct-and-strip: add an agree-with-wrong-claims instruction
    to the system prompt at GEN time, sample, then the cache keys the TRAINED
    (instruction-stripped) context. Judge-filter is deferred to the build's 80%
    yield floor read on the realized cache (the build drops below-floor sources)."""
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
        _write_responses_cache("syco_pos", src, dict(zip(wrong_claims, resps, strict=True)))


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
            _write_responses_cache("refusal_pos", src, dict(zip(requests, resps, strict=True)))
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


def _write_baseline_propensity(sources, behaviors, *, smoke: bool) -> None:
    out = CACHE_ROOT / "baseline_propensity.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                **C.repro_meta(seed=C.DEFAULT_SEED),
                "note": "source-side pre-training behavior rate covariate; "
                "per-source on-policy yields live in the elicitation caches; "
                "the build asserts the 80% floor + drops below-floor sources",
                "sources": list(sources),
                "behaviors": list(behaviors),
                "smoke": smoke,
            },
            indent=2,
        )
    )


# ── P2.1 train one cell ───────────────────────────────────────────────────────
def train_cell(cell: C.Cell, *, smoke: bool, gpu_id: int) -> Path:
    """Train one cell via the shared train_lora (marker band-stop / EM in-process
    via the #545 overrides / others). CVD is pinned in the launcher env per cell
    (gotchas: the in-process clobber alone is defeated by import-time cuInit) AND
    threaded as gpu_id; HF upload + Hub verify; per-cell WandB finish."""
    from explore_persona_space.train.sft import TrainLoraConfig, train_lora

    data_path = (
        GEN
        / ("train_smoke" if smoke else "train")
        / cell.behavior
        / f"{cell.slug}_seed{cell.seed}.jsonl"
    )
    assert data_path.exists(), f"training mix missing (run --phase p0 first): {data_path}"
    out_dir = ADAPTER_OUT / (f"{cell.slug}_seed{cell.seed}" + ("_smoke" if smoke else ""))
    if (out_dir / "adapter_model.safetensors").exists():
        logger.info("[p1-train] %s already trained -- skip", cell.slug)
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
    cfg = TrainLoraConfig(
        run_name=cell.run_name,
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
            logger.info("[p2] %s merged dir reaped", cell.slug)


# ── P2.3 upload raw completions + store tensors ───────────────────────────────
def upload_artifacts(cells: list[C.Cell], *, smoke: bool) -> None:
    """Push raw completions + store tensors to the HF data repo (adapters were
    pushed by train_lora). Fail-loud per upload-policy."""
    if smoke:
        logger.info("[p3-upload] smoke: skipping HF upload")
        return
    from explore_persona_space.orchestrate.hub import (
        upload_raw_completions_to_data_repo,
    )

    # raw completions: the eval gen wrote eval_results/issue_664/registry/<cell>/
    # completions__*.json. The canonical helper globs raw_completions.json; our
    # shape is completions__<col>__<ctx>.json, so upload the registry dir as a
    # dataset directory under the canonical slug.
    upload_raw_completions_to_data_repo(
        experiment_name="issue664_leakage_fleet",
        eval_results_dir=C.EVAL_ROOT,
    )
    logger.info("[p3-upload] raw completions uploaded")
    # store tensors -> HF analysis_tensors (plan §6.5: before pod teardown).
    _upload_store_tensors(cells)


def _upload_store_tensors(cells: list[C.Cell]) -> None:
    from explore_persona_space.orchestrate import hub

    for cell in cells:
        cell_dir = C.STORE_ROOT / f"{cell.slug}_seed{cell.seed}"
        tp = cell_dir / "tensors.pt"
        if not tp.exists():
            logger.warning("[p3-upload] store tensors missing for %s: %s", cell.slug, tp)
            continue
        for f in cell_dir.glob("*"):
            if f.is_file():
                hub._upload(
                    f,
                    repo_id=C.HF_DATA_REPO,
                    repo_type="dataset",
                    path_in_repo=f"{C.HF_STORE_PREFIX}/{cell.slug}_seed{cell.seed}/{f.name}",
                    upload_as_file=True,  # gotchas: per-file _upload needs this
                )
    logger.info("[p3-upload] store tensors uploaded -> %s/%s", C.HF_DATA_REPO, C.HF_STORE_PREFIX)


# ── Marker read-gauge readability smoke assert (§10 / §11 A7) ─────────────────
def _marker_readability_assert(cells: list[C.Cell]) -> None:
    """≥1 trained marker adapter at the band-stopped checkpoint has on-policy
    marker emission < 1% AND log P(marker) < log P(<|im_end|>) on the eval-probe
    slot. Trip -> HALT (Option B staged read needed before relaunch). Read from
    the extract_store marker_slot_stats.json (smoke writes to marker_slot_smoke)."""
    marker_cells = [c for c in cells if c.behavior == "marker"]
    if not marker_cells:
        logger.info("[smoke-assert] no marker cell in selection -- A7 read-gauge assert N/A")
        return
    checked = 0
    for cell in marker_cells:
        slot_path = C.EVAL_ROOT / "marker_slot" / (cell.slug + "_smoke") / "marker_slot_stats.json"
        if not slot_path.exists():
            continue
        payload = json.loads(slot_path.read_text())
        slots = payload["slots"]
        n_emit = sum(1 for s in slots.values() if s["trained"].get("argmax_id") == C.MARKER_ID)
        emit_rate = n_emit / max(1, len(slots))
        margins = [s["trained"]["z_marker"] - s["trained"]["z_eos"] for s in slots.values()]
        below_eos = all(m < 0 for m in margins) if margins else False
        logger.info(
            "[smoke-assert] %s emission=%.3f, z_marker<z_eos on all slots=%s",
            cell.slug,
            emit_rate,
            below_eos,
        )
        # Smoke trains only 2 steps (no band-stop), so this is a STRUCTURAL
        # exercise of the readability read, not the production [5,12]-nat gate.
        # The production assert (real band-stopped checkpoint) is enforced when
        # --smoke is off on the canary cell.
        checked += 1
    if checked == 0:
        raise RuntimeError(
            "[smoke-assert] marker readability assert ran on 0 marker cells "
            "(no marker_slot_stats.json produced) -- the A7 read-gauge readability "
            "test could not run; investigate the extraction marker-slot path."
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
            adapter_dir = ADAPTER_OUT / (
                f"{cell.slug}_seed{cell.seed}" + ("_smoke" if args.smoke else "")
            )
            extract_and_eval_cell(cell, adapter_dir, smoke=args.smoke, gpu_id=args.gpu_id)
        # registry manifest (the §6.5 verifier surface).
        phase_log("p2_manifest")
        _write_manifest(cells, smoke=args.smoke)
        # marker read-gauge readability assert (§10 / §11 A7).
        phase_log("p2_smoke_assert")
        _marker_readability_assert(cells)
    if args.phase == "p2":
        return

    if args.phase in ("all", "p3"):
        phase_log("p3_upload")
        upload_artifacts(cells, smoke=args.smoke)


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
        n = len(_select_cells(args))
        write_sentinel(
            "epm:results",
            f"issue664 Phase-2 fleet complete ({n} cells, smoke={args.smoke})",
            extra={
                "gate": "results",
                "blocks_pipeline": False,
                "reproducibility_card": {
                    "wandb_project": C.WANDB_PROJECT,
                    "wandb_run_names": [c.run_name for c in _select_cells(args)],
                    "adapter_paths": [c.hf_adapter_subfolder for c in _select_cells(args)],
                    "hf_model_repo": C.HF_MODEL_REPO,
                    "store_tensors_prefix": f"{C.HF_DATA_REPO}/{C.HF_STORE_PREFIX}",
                    "judge_model": "claude-sonnet-4-5-20250929",
                    "seeds": sorted({c.seed for c in _select_cells(args)}),
                },
            },
        )
        phase_log("done")  # RESERVED terminal line (poll_pipeline) -- main exit only
    return 0


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(rc)  # datasets/transformers SIGABRT at finalize (gotchas PyGILState)
