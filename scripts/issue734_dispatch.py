"""Issue #734 -- pod-side driver: corrected marker re-read + H1 model pair.

Demonstrates (H3) that #664's marker "no install" on Qwen-2.5-7B-Instruct is a
slot-rooting / measurement artifact, and confirms (H1) with a matched base-vs-
Instruct fresh train. ONE code path for smoke and sweep (PASS_UNIFIED): the smoke
is this driver with ``--cells 1 --smoke``.

Phases (plan §4):

  phase0  token-id split diagnostic (CPU; delegates to issue734_phase0).
  phase1  CORRECTED on-policy re-read on the 16 reused #664 adapters (HEADLINE):
          download each adapter (sha-pinned), merge, regenerate the trained
          model's on-policy greedy R, read the marker four-float DV at BOTH the
          #664 mis-rooted slot (compute_marker_slot_stats -- the bug) AND the
          corrected slot (token-id-threaded -- the fix). Compare to the in-loop
          band-stop value (read from the adapter's band_stop_result.json).
  phase1p5  rsLoRA parity probe (reuse-check (g) / §7.5): the corrected reader on
          mk_librarian_contra_d1_seed42 must reproduce its in-loop band-stop value
          (~+6.9 nat) within ~2 nat BEFORE the sweep. MISS -> HALT.
  setup_h1_mix  build the librarian-contra-d1-seed42 #664 marker training mix that
          phase2 (and a standalone phase3) reuse -- #664 produced it via --phase p0
          but never uploaded it to HF, and the git-clone-only GCP lane cannot stage
          it (issue #734 crash-fix round 1). Reuses #664's marker pool + marker_R
          base-greedy elicitation + the standalone CPU builder; idempotent (sha256
          provenance sidecar). Delegates to issue734_h1_mix.build_h1_mix.
  phase2  H1 matched base-vs-Instruct fresh train (6 cells = 2 models x 3 seeds),
          EXACT #664 marker recipe (recipe_for("marker")), model the only deliberate
          variable; read in-loop band-stop + corrected on-policy. The FIRST base
          seed band-stops as a smoke gate (§8) before the base 3-seed expansion.
  phase3  CONDITIONAL (only when phase1 FALSIFIES H3): H2 lr x steps mini-sweep.
  all     phase0 -> phase1p5 -> phase1 -> setup_h1_mix -> phase2 ->
          phase3-if-falsified -> upload -> done.

Pod-side contract (CLAUDE.md / poll_pipeline.py): emits ``[phase=<name>]`` log
lines, a terminal ``[phase=done]`` ONLY on the main dispatcher's graceful exit,
and an end-of-run sentinel JSON at /workspace/logs/issue-734-<kind>-<epoch>.json
with poll_pipeline._SENTINEL_REQUIRED_KEYS. NEVER shells out to scripts/task.py.

Usage (sweep): nohup uv run python scripts/issue734_dispatch.py --phase all \
    > /workspace/logs/issue734.log 2>&1 < /dev/null &
Smoke:        uv run python scripts/issue734_dispatch.py --phase all --cells 1 --smoke
CPU carve-out: --dry-run exercises the cell plumbing + sentinel + [phase=done]
              without a GPU forward (the GPU-bound-phase substitute coverage).
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))  # issue734_* / issue664_* / issue594_common

# vLLM v1 fork-poison guard (gotchas #628): main() touches transformers/tokenizer
# before LLM(); spawn isolates the EngineCore subprocess.
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

from explore_persona_space.eval.marker_logprob import validate_marker_slot_record  # noqa: E402
from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()  # phase1 vLLM gen + phase2 train_lora + HF uploads need HF_TOKEN / WANDB_API_KEY

import issue664_common as C664  # noqa: E402
import issue734_common as C  # noqa: E402
import issue734_h1_mix as MIX  # noqa: E402
import issue734_marker_reread as RR  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("issue734_dispatch")


# ── Pod-side contract helpers (poll_pipeline.py) ──────────────────────────────
def phase_log(name: str) -> None:
    """Emit the [phase=<name>] line poll_pipeline.py parses (PHASE_RE). The
    terminal [phase=done] is RESERVED for the single graceful-exit line in main()."""
    safe = "".join(ch if (ch.isalnum() or ch == "_") else "_" for ch in name.lower())
    print(f"[phase={safe}]", flush=True)


def _log_dir() -> Path:
    for cand in (Path("/workspace/logs"), C.EVAL_ROOT / "logs"):
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
        "by": "issue734_dispatch",
        "ts": time.time(),
    }
    if extra:
        payload.update(extra)
    slug = kind.replace(":", "_")
    out = _log_dir() / f"issue-734-{slug}-{int(time.time())}.json"
    out.write_text(json.dumps(payload, indent=2))
    logger.info("sentinel written: %s", out)
    return out


def _gpu_reclaim(*, ipc: bool = False) -> None:
    """Reclaim CUDA cache after a model/engine is dropped (NO-OP on CPU; no
    bare except: pass)."""
    import torch

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        if ipc:
            torch.cuda.ipc_collect()


def _resolve_device() -> str:
    import torch

    return "cuda:0" if torch.cuda.is_available() else "cpu"


# ── Reused-#664 adapter download (sha-pinned per reuse-check (f)) ─────────────
def download_reused_adapter(cell: C.Phase1Cell) -> Path:
    """Download a reused #664 adapter dir from the model repo (reuse (c)/(e)/(f)).

    Returns the local adapter dir. Uses huggingface_hub.snapshot_download with an
    allow_patterns scoped to the cell's subfolder. The gauge assert + adapter_config
    recipe-match (reuse (a)/(g)) run in the reader; this only fetches the files.
    """
    from huggingface_hub import snapshot_download

    sub = cell.hf_adapter_subfolder  # adapters/issue_664/mk_..._seed42
    local_root = C.REUSED_ADAPTER_CACHE / cell.eval_key
    local_root.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=C.HF_MODEL_REPO,
        allow_patterns=[f"{sub}/*"],
        local_dir=str(C.REUSED_ADAPTER_CACHE / "_snapshot"),
    )
    adapter_dir = C.REUSED_ADAPTER_CACHE / "_snapshot" / sub
    if not (adapter_dir / "adapter_model.safetensors").exists():
        raise RuntimeError(
            f"reused #664 adapter missing after download: {adapter_dir} "
            f"(expected {sub}/adapter_model.safetensors on {C.HF_MODEL_REPO})"
        )
    return adapter_dir


def read_inloop_band_stop(adapter_dir: Path) -> dict | None:
    """The #664 in-loop band-stop ground truth (band_stop_result.json), if present
    in the adapter dir. Returns the dict or None (some reused adapters may carry it,
    some may not -- the reader handles None as 'not available')."""
    p = Path(adapter_dir) / "band_stop_result.json"
    if p.exists():
        return json.loads(p.read_text())
    return None


def assert_adapter_recipe_match(adapter_dir: Path, cell_key: str) -> dict:
    """Reuse-check (a)+(g): the adapter's OWN adapter_config.json must match the
    #664 marker recipe (rsLoRA r32/a64, q/k/v/o, use_rslora=True) AND be gauge-free
    (no lm_head/embed_tokens, empty modules_to_save). FAIL LOUD on any mismatch."""
    from explore_persona_space.eval.marker_logprob import assert_gauge_free_adapter_config

    cfg_path = Path(adapter_dir) / "adapter_config.json"
    assert cfg_path.exists(), f"adapter_config.json missing under {adapter_dir} (reuse (a))"
    cfg = json.loads(cfg_path.read_text())
    # (g) application scaling + (a) recipe match, grounded on the config itself.
    assert cfg.get("r") == C664.recipe_for("marker").lora_r, (
        f"[{cell_key}] adapter r={cfg.get('r')} != {C664.recipe_for('marker').lora_r} (reuse (a))"
    )
    assert cfg.get("lora_alpha") == C664.recipe_for("marker").lora_alpha, (
        f"[{cell_key}] adapter alpha={cfg.get('lora_alpha')} != "
        f"{C664.recipe_for('marker').lora_alpha} (reuse (a))"
    )
    assert cfg.get("use_rslora") is True, (
        f"[{cell_key}] adapter use_rslora={cfg.get('use_rslora')} != True (reuse (g) -- the "
        f"alpha/sqrt(r) application scaling the corrected-read gauge is pinned to)"
    )
    assert_gauge_free_adapter_config(cfg, context=f"{cell_key} reused #664 marker adapter")
    return cfg


# ── On-policy R generation (trained model's own greedy responses) ─────────────
def generate_onpolicy_R(
    model_path: str, source: str, questions: list[str], *, tokenizer_id: str | None = None
) -> list[dict]:
    """vLLM greedy gen of the trained model's OWN response per source-context question.

    Returns [{"question", "response_text"}] -- the on-policy R the corrected +
    mis-rooted readers consume. Chunked + use_tqdm=False (gotchas #613/#664).

    ``tokenizer_id`` defaults to the Instruct tokenizer (the Phase-1 reused #664
    adapters' base). H1 cells pass their OWN model id (base for ``h1_base_*``,
    Instruct for ``h1_instruct_*``) so the chat-template render matches the model
    the cell trained on -- never cross models (the reconciler-named H1 fix).
    """
    from vllm import LLM, SamplingParams

    from explore_persona_space.analysis.representation_shift import _reap_vllm_engine

    chunk = int(os.environ.get("EPM_VLLM_GREEDY_CHUNK_SIZE", "500"))
    tokenizer_prompts: list[str] = []
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tokenizer_id or C.INSTRUCT_ID, trust_remote_code=True)
    for q in questions:
        msgs = C.source_messages(source, q)
        tokenizer_prompts.append(
            tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        )

    # enforce_eager defaults TRUE (#734 crash-fix round 5): this is the LOAD-BEARING
    # H1-phase engine that deadlocked at cuda-graph capture on pod-734. Env-overridable
    # via EPM_VLLM_ENFORCE_EAGER=0 for a future pod that wants graphs (C.vllm_enforce_eager).
    llm = LLM(
        model=model_path,
        dtype="bfloat16",
        gpu_memory_utilization=0.80,
        max_model_len=2 * C.MAX_NEW_TOKENS + 1024,
        enforce_eager=C.vllm_enforce_eager(),
    )
    try:
        sp = SamplingParams(temperature=0.0, max_tokens=C.MAX_NEW_TOKENS)
        texts: list[str] = []
        n_chunks = (len(tokenizer_prompts) + chunk - 1) // chunk
        for i in range(0, len(tokenizer_prompts), chunk):
            sub = tokenizer_prompts[i : i + chunk]
            logger.info(
                "[vllm-chunk] gen chunk %d/%d (%d prompts)", i // chunk + 1, n_chunks, len(sub)
            )
            outs = llm.generate(sub, sp, use_tqdm=False)
            texts.extend(o.outputs[0].text for o in outs)
        return [{"question": q, "response_text": t} for q, t in zip(questions, texts, strict=True)]
    finally:
        _reap_vllm_engine(llm)
        del llm
        gc.collect()
        _gpu_reclaim(ipc=True)
        time.sleep(1.0)


# ── Shared corrected-vs-misrooted read (Phase 1 reuse + Phase 2 H1) ───────────
def _corrected_misrooted_read(
    *,
    merged_path: str,
    base_model_id: str,
    tokenizer_id: str,
    source: str,
    questions: list[str],
    band: tuple[float, float],
    summary_meta: dict,
    out_eval_key: str,
    log_phase: str,
    seed: int,
    smoke: bool,
) -> tuple[dict, float]:
    """Read the marker DV at BOTH the corrected and the #664 mis-rooted slot, on
    the SAME merged trained weights (trained AND base side), and write the per-cell
    ``marker_slot_corrected.json``. Shared by the Phase-1 reuse path
    (``reread_cell``, base = Instruct) and the Phase-2 H1 path (``reread_h1_cell``,
    base = the H1 cell's OWN model). The corrected read threads token ids directly
    (the FIX); the mis-rooted read uses ``compute_marker_slot_stats`` (the bug).

    ``base_model_id`` is the matched base for the trained adapter (the corrected DV
    is trained - base). ``tokenizer_id`` renders both R generation and the slot
    reads with the model's OWN tokenizer -- the H1 cells use their own model so the
    chat template never crosses models (the reconciler-named H1 fix).

    Returns ``(summary_dict, corrected_source_delta_logp_mean)``.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = _resolve_device()
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, trust_remote_code=True)
    C.assert_marker_token(tokenizer)

    # The trained model's OWN greedy R (the on-policy DV is read on R), rendered
    # through the cell's OWN tokenizer (base vs Instruct).
    trained_R = generate_onpolicy_R(merged_path, source, questions, tokenizer_id=tokenizer_id)
    msgs_list = [C.source_messages(source, r["question"]) for r in trained_R]
    r_texts = [r["response_text"] for r in trained_R]

    def _load(path: str):
        m = AutoModelForCausalLM.from_pretrained(
            path,
            dtype=(torch.bfloat16 if device.startswith("cuda") else torch.float32),
            device_map=({"": 0} if device.startswith("cuda") else None),
            trust_remote_code=True,
        ).eval()
        return m if device.startswith("cuda") else m.to(device)

    # Corrected read: trained AND base, threading token ids directly (the FIX).
    trained_model = _load(merged_path)
    try:
        trained_corr = RR.corrected_slot_stats(
            trained_model,
            tokenizer,
            msgs_list,
            r_texts,
            marker_text=C.MARKER_TEXT,
            marker_id=C.MARKER_ID,
            eos_token_id=C.IM_END_ID,
            device=device,
        )
        trained_mis = RR.misrooted_slot_stats(
            trained_model,
            tokenizer,
            msgs_list,
            r_texts,
            marker_text=C.MARKER_TEXT,
            eos_token_id=C.IM_END_ID,
            device=device,
        )
    finally:
        del trained_model
        gc.collect()
        _gpu_reclaim()

    base_model = _load(base_model_id)
    try:
        base_corr = RR.corrected_slot_stats(
            base_model,
            tokenizer,
            msgs_list,
            r_texts,
            marker_text=C.MARKER_TEXT,
            marker_id=C.MARKER_ID,
            eos_token_id=C.IM_END_ID,
            device=device,
        )
        base_mis = RR.misrooted_slot_stats(
            base_model,
            tokenizer,
            msgs_list,
            r_texts,
            marker_text=C.MARKER_TEXT,
            eos_token_id=C.IM_END_ID,
            device=device,
        )
    finally:
        del base_model
        gc.collect()
        _gpu_reclaim()

    # Per-context deltas (trained - base) in all three spaces, both reads.
    def _mean(vals: list[float]) -> float:
        return sum(vals) / len(vals) if vals else float("nan")

    rows = []
    corr_deltas, mis_deltas = [], []
    for i, q in enumerate([r["question"] for r in trained_R]):
        c_dlp = trained_corr[i]["logp"] - base_corr[i]["logp"]
        m_dlp = trained_mis[i]["logp"] - base_mis[i]["logp"]
        corr_deltas.append(c_dlp)
        mis_deltas.append(m_dlp)
        rows.append(
            {
                "question": q[:120],
                "corrected": {
                    "trained": trained_corr[i],
                    "base": base_corr[i],
                    "delta_logp": c_dlp,
                    "delta_z_marker": trained_corr[i]["z_marker"] - base_corr[i]["z_marker"],
                    "delta_eos_margin": (
                        (trained_corr[i]["z_marker"] - trained_corr[i]["z_eos"])
                        - (base_corr[i]["z_marker"] - base_corr[i]["z_eos"])
                    ),
                },
                "misrooted": {
                    "trained": trained_mis[i],
                    "base": base_mis[i],
                    "delta_logp": m_dlp,
                },
            }
        )

    corr_mean = _mean(corr_deltas)
    summary = {
        "band_target_nats": list(band),
        "corrected_source_delta_logp_mean": corr_mean,
        "misrooted_source_delta_logp_mean": _mean(mis_deltas),
        "corrected_in_band": (band[0] <= corr_mean <= band[1]),
        "rows": rows,
        "repro": C.repro_meta(seed=seed),
        "smoke": smoke,
        **summary_meta,
    }
    out_dir = C.CORRECTED_REREAD_ROOT / out_eval_key
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "marker_slot_corrected.json").write_text(json.dumps(summary, indent=2))
    logger.info(
        "[%s] %s corrected=%.2f nat (band %s) misrooted=%.2f nat in_band=%s",
        log_phase,
        out_eval_key,
        corr_mean,
        band,
        summary["misrooted_source_delta_logp_mean"],
        summary["corrected_in_band"],
    )
    return summary, corr_mean


# ── Phase 1: corrected vs mis-rooted re-read on a reused #664 adapter ─────────
def reread_cell(cell: C.Phase1Cell, *, smoke: bool, return_corrected_delta: bool = False):
    """Re-read ONE reused #664 marker cell at BOTH slots on the SAME weights.

    Writes eval_results/issue_734/corrected_reread/<eval_key>/marker_slot_corrected.json.
    Returns the per-cell summary dict (and the mean corrected source delta when
    return_corrected_delta is set, used by the parity probe + the falsify gate).
    """
    from explore_persona_space.train.sft import merge_lora

    adapter_dir = download_reused_adapter(cell)
    adapter_cfg = assert_adapter_recipe_match(adapter_dir, cell.eval_key)
    inloop = read_inloop_band_stop(adapter_dir)

    questions = C.marker_question_pool(smoke=smoke)
    if smoke:
        questions = questions[:2]

    # Merge base + reused adapter (the corrected/mis-rooted reads both run on the
    # SAME merged trained model -- single-variable: only the read code differs).
    merged = C.REUSED_ADAPTER_CACHE / (cell.eval_key + "_merged")
    merge_lora(C.INSTRUCT_ID, str(adapter_dir), str(merged), gpu_id=0)
    try:
        summary, corr_mean = _corrected_misrooted_read(
            merged_path=str(merged),
            base_model_id=C.INSTRUCT_ID,  # the reused #664 adapters' base
            tokenizer_id=C.INSTRUCT_ID,
            source=cell.source,
            questions=questions,
            band=C.band_for_dose(cell.dose),
            summary_meta={
                "experiment": "issue734_corrected_reread",
                "phase": "phase1_reuse",
                "cell": cell.eval_key,
                "source": cell.source,
                "arm": cell.arm,
                "dose": cell.dose,
                "seed": cell.seed,
                "inloop_band_stop": inloop,  # the ground-truth install value (may be None)
                "adapter_config_recipe": {
                    "r": adapter_cfg.get("r"),
                    "lora_alpha": adapter_cfg.get("lora_alpha"),
                    "use_rslora": adapter_cfg.get("use_rslora"),
                    "target_modules": adapter_cfg.get("target_modules"),
                },
            },
            out_eval_key=cell.eval_key,
            log_phase="phase1",
            seed=cell.seed,
            smoke=smoke,
        )
    finally:
        if merged.exists():
            shutil.rmtree(merged)

    if return_corrected_delta:
        return summary, corr_mean
    return summary


def reread_h1_cell(cell: C.H1Cell, adapter_dir: Path, *, smoke: bool) -> dict:
    """The Phase-2 H1 corrected on-policy read on a FRESHLY-trained H1 adapter
    (the §6.5 registered deliverable the reconciler upheld as missing in round 1).

    Runs ``generate_onpolicy_R`` + ``corrected_slot_stats`` (+ the mis-rooted
    control) trained-vs-base on the H1 adapter with the H1 cell's OWN model + its
    own tokenizer (base for ``h1_base_*``, Instruct for ``h1_instruct_*``) -- never
    crossing models -- and writes the per-cell ``marker_slot_corrected.json`` under
    the SAME registered glob the Phase-1 reread uses (so figure / aggregation code
    consumes both phases uniformly). Returns the per-cell summary dict.
    """
    from explore_persona_space.train.sft import merge_lora

    questions = C.marker_question_pool(smoke=smoke)
    if smoke:
        questions = questions[:2]

    merged = C.ADAPTER_OUT / (cell.eval_key + ("_smoke" if smoke else "") + "_merged")
    # Merge the H1 cell's OWN base model + the freshly-trained adapter (gpu_id 0
    # matches the in-process CVD clobber the launcher pins per cell).
    merge_lora(cell.model_id, str(adapter_dir), str(merged), gpu_id=0)
    try:
        summary, _ = _corrected_misrooted_read(
            merged_path=str(merged),
            base_model_id=cell.model_id,  # the H1 cell's matched base (NOT Instruct)
            tokenizer_id=cell.model_id,  # render with the cell's OWN tokenizer
            source=cell.source,
            questions=questions,
            band=C.band_for_dose(cell.dose),
            summary_meta={
                "experiment": "issue734_corrected_reread",
                "phase": "phase2_h1",
                "cell": cell.eval_key,
                "model_key": cell.model_key,
                "model_id": cell.model_id,
                "source": cell.source,
                "arm": cell.arm,
                "dose": cell.dose,
                "seed": cell.seed,
            },
            out_eval_key=cell.eval_key,
            log_phase="phase2",
            seed=cell.seed,
            smoke=smoke,
        )
    finally:
        if merged.exists():
            shutil.rmtree(merged)
    return summary


# ── Phase 1.5: rsLoRA parity probe (reuse-check (g) / §7.5) ───────────────────
def parity_probe(*, smoke: bool) -> bool:
    """The corrected reader on mk_librarian_contra_d1_seed42 must reproduce its
    in-loop band-stop value within ~2 nat BEFORE the sweep. MISS -> HALT (the
    gauge/load is wrong; do NOT trust the corrected sweep). Returns True on pass."""
    src, arm, dose = C.PARITY_PROBE_CELL
    cell = C.Phase1Cell(source=src, arm=arm, dose=dose)
    summary, corr_mean = reread_cell(cell, smoke=smoke, return_corrected_delta=True)
    inloop = summary.get("inloop_band_stop")
    if not inloop or inloop.get("last_delta_nats") is None:
        # No in-loop ground truth available for this adapter; fall back to the band
        # membership check (the corrected read MUST at least land in the d1 band).
        in_band = summary["corrected_in_band"]
        logger.warning(
            "[phase1p5] no in-loop band_stop_result for %s; falling back to band-membership "
            "check (corrected=%.2f, in_band=%s)",
            cell.eval_key,
            corr_mean,
            in_band,
        )
        return bool(in_band) or smoke  # smoke (2-step adapters) never bands -- pass structurally
    inloop_delta = float(inloop["last_delta_nats"])
    gap = abs(corr_mean - inloop_delta)
    ok = gap <= C.PARITY_TOLERANCE_NAT
    logger.info(
        "[phase1p5] parity: corrected=%.2f vs in-loop=%.2f (gap=%.2f, tol=%.1f) -> %s",
        corr_mean,
        inloop_delta,
        gap,
        C.PARITY_TOLERANCE_NAT,
        "PASS" if ok else "HALT",
    )
    return ok or smoke


# ── setup_h1_mix: build the one #664 marker mix H1 phase2 reuses ──────────────
def run_setup_h1_mix(*, smoke: bool, dry_run: bool) -> dict:
    """Build the librarian-contra-d1-seed42 marker training mix H1 phase2 reuses.

    The mix was produced by #664's --phase p0 but never uploaded to HF, so the
    git-clone-only GCP lane cannot stage it; without this phase, phase2 crashes at
    train_h1_cell's `assert data_path.exists()` (issue #734 crash-fix round 1).
    Delegates to issue734_h1_mix.build_h1_mix (which reuses #664's marker pool +
    marker_R elicitation + the standalone CPU builder). Idempotent: a no-op when
    the mix is already current (sha256 matches its provenance sidecar)."""
    phase_log("setup_h1_mix")
    if dry_run:
        logger.info(
            "[setup_h1_mix][dry-run] would build %s (no GPU forward / no subprocess)",
            MIX.mix_path(smoke=smoke),
        )
        return {"built": False, "dry_run": True, "mix_path": str(MIX.mix_path(smoke=smoke))}
    already = MIX._mix_is_current(smoke=smoke)
    out = MIX.build_h1_mix(smoke=smoke, gpu_id=0)
    return {
        "built": not already,  # False when the idempotency skip fired
        "already_current": already,
        "mix_path": str(out),
        "exists": out.exists(),
    }


# ── Phase 2: H1 matched base-vs-Instruct fresh train ──────────────────────────
def train_h1_cell(cell: C.H1Cell, *, smoke: bool, gpu_id: int = 0) -> Path:
    """Fresh-train ONE H1 marker cell at the EXACT #664 recipe (recipe_for("marker")),
    model the only deliberate variable. Writes band_stop_result.json + trajectory.

    Reuses #664's training mix for the cell (librarian-contra-d1) so the ONLY change
    vs #664 is the base model id. CVD pinned per cell in the launcher env (gotchas);
    the in-process clobber rewrites the same gpu_id.
    """
    from explore_persona_space.train.sft import TrainLoraConfig, train_lora

    # The training mix is #664's librarian-contra-d1 marker mix (reuse). The mix is
    # PINNED to the seed-42 #664 baseline for EVERY H1 seed (to_664_cell forces
    # seed=PHASE1_SEED): the mix data is content-deterministic, the deliberate H1
    # variable is the model-init seed (cell.seed -> train_kwargs below), NOT the
    # mix data, and #664 only materialized the seed-42 marker grid on disk.
    c664 = cell.to_664_cell()
    assert c664.eval_key == "mk_librarian_contra_d1_seed42", (
        f"H1 must reuse the seed-42 #664 mix for every H1 seed; got {c664.eval_key!r} "
        f"(to_664_cell must pin seed=PHASE1_SEED, NOT cell.seed -- the seed-137/256 "
        f"mixes do not exist on disk and a per-seed mix is a smuggled second variable)"
    )
    data_path = (
        C664.DATA_ROOT / ("train_smoke" if smoke else "train") / "marker" / f"{c664.eval_key}.jsonl"
    )
    assert data_path.exists(), (
        f"H1 training mix missing: {data_path} (run the setup_h1_mix phase first -- "
        f"issue734_dispatch --phase setup_h1_mix -- which builds it self-contained; "
        f"--phase all / --phase phase2 run setup_h1_mix automatically before this)"
    )
    out_dir = C.ADAPTER_OUT / (cell.eval_key + ("_smoke" if smoke else ""))
    if (out_dir / "adapter_model.safetensors").exists():
        logger.info("[phase2] %s already trained -- skip", cell.eval_key)
        return out_dir

    recipe = C664.recipe_for("marker")  # EXACT #664 marker recipe (single-variable)
    kwargs = recipe.train_kwargs(
        dose=cell.dose, gpu_id=gpu_id, run_name=cell.run_name, seed=cell.seed
    )
    H1_TRAJECTORY_DIR = C.H1_TRAJECTORY_ROOT / cell.eval_key
    H1_TRAJECTORY_DIR.mkdir(parents=True, exist_ok=True)
    kwargs["marker_band_trajectory_path"] = str(H1_TRAJECTORY_DIR / "marker_band_trajectory.json")
    if smoke:
        kwargs["epochs"] = 1
        kwargs["max_steps"] = 2
        kwargs.pop("warmup_steps", None)
        kwargs["marker_band_stop"] = False  # 2 steps can't band-stop; smoke

    cfg = TrainLoraConfig(
        hf_upload=not smoke,
        hf_repo=C.HF_MODEL_REPO,
        hf_path_in_repo=cell.hf_adapter_subfolder,
        **kwargs,
    )
    os.environ["EPM_SKIP_INLINE_CHECKPOINT_UPLOAD"] = "1"  # train_lora owns the HF upload
    try:
        # The deliberate H1 variable: cell.model_id (base vs Instruct), not C664.QWEN_ID.
        train_lora(cell.model_id, str(data_path), str(out_dir), cfg=cfg)
    finally:
        import wandb

        if wandb.run is not None:
            wandb.finish()  # one WandB run PER CELL (distinct run names; #664 precedent)

    # Persist the in-loop band-stop result to the H1 deliverable glob.
    bs_src = out_dir / "band_stop_result.json"
    if bs_src.exists():
        bs_dir = C.H1_BAND_STOP_ROOT / cell.eval_key
        bs_dir.mkdir(parents=True, exist_ok=True)
        bs_dir.joinpath("band_stop_result.json").write_text(bs_src.read_text())
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


# ── Phase orchestration ───────────────────────────────────────────────────────
def _valid_corrected_reread(path: Path) -> dict | None:
    """Return the parsed ``marker_slot_corrected.json`` summary IFF it is a
    COMPLETE four-float record, else None (#734 crash-fix round 5, Fix 3 — the
    skip-if-exists guard's validity check).

    A skip-if-exists guard that trusts mere file presence would skip a
    truncated / corrupt JSON from a crashed prior cell and propagate a fake
    "already complete". So treat present-but-invalid as needs-rerun. Validity
    is the full storage CONTRACT (#530), not mere key presence (#734 crash-fix
    round 6, reconciler blocker): the file must parse, carry a finite-numeric
    ``corrected_source_delta_logp_mean`` + a boolean ``corrected_in_band``, a
    non-empty ``rows`` list, and EVERY row's corrected trained AND base reads
    must PASS :func:`validate_marker_slot_record` (four finite floats, ``logp``
    non-positive, the ``logp == z_marker - logZ`` softmax identity). A record
    with all four key NAMES present but BAD VALUES (NaN, a string, a positive
    ``logp``, a broken softmax identity) is treated as needs-rerun -- a
    round-5 key-only check skipped exactly that corrupt JSON."""
    if not path.exists():
        return None
    try:
        summary = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError, UnicodeDecodeError):
        return None
    if not isinstance(summary, dict):
        return None
    # Top-level aggregates: a finite numeric mean + a real boolean band flag.
    top_mean = summary.get("corrected_source_delta_logp_mean")
    if (
        not isinstance(top_mean, (int, float))
        or isinstance(top_mean, bool)
        or not math.isfinite(float(top_mean))
    ):
        return None
    if not isinstance(summary.get("corrected_in_band"), bool):
        return None
    rows = summary.get("rows")
    if not isinstance(rows, list) or not rows:
        return None
    # Every row's corrected trained AND base slot record must satisfy the
    # four-float storage contract via the NAMED validator (not key presence).
    for row in rows:
        corr = row.get("corrected") if isinstance(row, dict) else None
        if not isinstance(corr, dict):
            return None
        for side in ("trained", "base"):
            rec = corr.get(side)
            if not isinstance(rec, dict):
                return None
            try:
                validate_marker_slot_record(rec, context=f"{path}:{side}", require_z_eos=True)
            except AssertionError:
                return None
    return summary


def run_phase1(
    *, cells_limit: int | None, smoke: bool, dry_run: bool, force_rerun: bool = False
) -> dict:
    phase_log("phase1_corrected_reread")
    cells = C.phase1_cells()
    if cells_limit is not None:
        cells = cells[:cells_limit]
    results = []
    in_band_d1_sources: set[str] = set()
    below_floor_d1_sources: set[str] = set()  # corrected read STILL < 5 nat (§7.5 falsify)
    floor = C.BAND_D1[0]  # 5.0 nat
    # The corrected read is read on the contrastive arm's d1 cells (the source-gate
    # spine, plan §3 quantitative success). Use contra-d1 as the source-gate read.
    for cell in cells:
        if dry_run:
            logger.info("[phase1][dry-run] would re-read %s (no GPU forward)", cell.eval_key)
            continue
        # Skip-if-exists guard (#734 crash-fix round 5, Fix 3): the 16/16 Phase-1
        # corrected_reread JSONs persist on /workspace, so a --phase all relaunch after
        # the Phase-2 deadlock fix resumes STRAIGHT into Phase 2 instead of re-running
        # ~32 min of GPU. A present-but-corrupt JSON is treated as needs-rerun
        # (_valid_corrected_reread). --force-rerun overrides the skip.
        out_path = C.CORRECTED_REREAD_ROOT / cell.eval_key / "marker_slot_corrected.json"
        if not force_rerun:
            cached = _valid_corrected_reread(out_path)
            if cached is not None:
                logger.info("[phase1] %s already complete (%s); skipping", cell.eval_key, out_path)
                summary = cached
                results.append(summary)
                if cell.dose == "d1" and cell.arm == "contra":
                    corr = summary["corrected_source_delta_logp_mean"]
                    if summary["corrected_in_band"]:
                        in_band_d1_sources.add(cell.source)
                    if corr < floor:
                        below_floor_d1_sources.add(cell.source)
                continue
        summary = reread_cell(cell, smoke=smoke)
        results.append(summary)
        if cell.dose == "d1" and cell.arm == "contra":
            corr = summary["corrected_source_delta_logp_mean"]
            if summary["corrected_in_band"]:
                in_band_d1_sources.add(cell.source)
            if corr < floor:
                below_floor_d1_sources.add(cell.source)
    # H3 quantitative success (plan §3): corrected source read in [5,12] on >=3/4 of
    # the four contra-d1 sources, with >=3 sources actually measured.
    d1_sources_measured = {c.source for c in cells if c.dose == "d1" and c.arm == "contra"}
    h3_confirmed = len(in_band_d1_sources) >= 3 and len(d1_sources_measured) >= 3
    # H3 falsified (plan §7.5): corrected read STILL < 5 nat on >=3/4 measured d1
    # sources (the install does not survive on-policy generation) -> escalate to H2.
    h3_falsified = len(d1_sources_measured) >= 3 and len(below_floor_d1_sources) >= 3
    return {
        "n_cells": len(results),
        "in_band_d1_sources": sorted(in_band_d1_sources),
        "below_floor_d1_sources": sorted(below_floor_d1_sources),
        "d1_sources_measured": sorted(d1_sources_measured),
        "h3_confirmed": h3_confirmed,
        "h3_falsified": h3_falsified,
    }


def run_phase2(
    *, cells_limit: int | None, smoke: bool, dry_run: bool, seeds: list[int] | None = None
) -> dict:
    phase_log("phase2_h1_train")
    cells = C.h1_cells()
    if seeds is not None:
        # --seeds restricts the H1 model-init seeds run (smoke / subset re-run). A
        # seed not in C.H1_SEEDS is a usage error (fail loud, never silently empty).
        unknown = set(seeds) - set(C.H1_SEEDS)
        assert not unknown, f"--seeds {sorted(unknown)} not in H1_SEEDS {C.H1_SEEDS}"
        cells = [c for c in cells if c.seed in set(seeds)]
    if cells_limit is not None:
        cells = cells[:cells_limit]
    # §8 base-arm smoke gate: the FIRST base seed must band-stop in [5,12] BEFORE the
    # base 3-seed expansion (first base-marker probe in the project). Order base seed
    # 42 first; on a real run, HALT the base expansion if it misses the band.
    cells = sorted(cells, key=lambda c: (c.model_key != "base", c.seed))
    trained = []
    corrected_read_cells = []  # the §6.5 H1 corrected-read deliverable (>=6 cells)
    base_first_in_band = None
    # §7.5 H1-surprise escalation inputs: per-model on-policy corrected in-band reads.
    base_corrected_in_band: dict[int, bool] = {}
    instruct_corrected_in_band: dict[int, bool] = {}
    for cell in cells:
        if dry_run:
            logger.info(
                "[phase2][dry-run] would train %s (model=%s) + corrected-read (no GPU)",
                cell.eval_key,
                cell.model_id,
            )
            continue
        out_dir = train_h1_cell(cell, smoke=smoke)
        trained.append(cell.eval_key)
        if not smoke and cell.model_key == "base" and cell.seed == C.H1_SEEDS[0]:
            bs = read_inloop_band_stop(out_dir)
            base_first_in_band = bool(bs and bs.get("stopped_in_band"))
            if not base_first_in_band:
                # §8 gate MISS: base resists or the render needs a fix -- report,
                # do NOT silently chase recipe knobs. HALT the base 3-seed expansion.
                logger.error(
                    "[phase2][HALT §8] first base seed did NOT band-stop in [5,12] "
                    "(band_stop_result=%s) -- base resistance is a real signal; "
                    "halting the base 3-seed expansion.",
                    bs,
                )
                raise SystemExit(
                    "issue734 §8 base-arm smoke gate FAILED: first base seed missed the "
                    "[5,12] band. Report base-resistance finding; do not chase knobs."
                )
        # §6.5 registered deliverable + §3/§4/§7.5 H1 question: the corrected
        # on-policy read on the freshly-trained H1 adapter (the reconciler-upheld
        # round-2 fix -- round 1 trained the adapter but never read it on-policy).
        h1_summary = reread_h1_cell(cell, out_dir, smoke=smoke)
        corrected_read_cells.append(cell.eval_key)
        in_band = bool(h1_summary["corrected_in_band"])
        if cell.model_key == "base":
            base_corrected_in_band[cell.seed] = in_band
        else:
            instruct_corrected_in_band[cell.seed] = in_band

    # §7.5 H1-surprise escalation read (base reads in-band on-policy at the corrected
    # slot but Instruct does NOT, recipe held exact -> model resistance is real and
    # Instruct-specific). Surface the signal; the orchestrator/analyzer interpret it.
    h1_surprise = (
        not smoke
        and bool(base_corrected_in_band)
        and bool(instruct_corrected_in_band)
        and all(base_corrected_in_band.values())
        and not any(instruct_corrected_in_band.values())
    )
    if h1_surprise:
        logger.warning(
            "[phase2][§7.5 H1-SURPRISE] base reads IN-BAND on-policy at the corrected "
            "slot on every seed but Instruct reads BELOW-band on every seed, recipe held "
            "exact -- model resistance is real and Instruct-specific. Reporting as the "
            "finding (base in-band=%s, instruct in-band=%s); not chasing recipe knobs.",
            base_corrected_in_band,
            instruct_corrected_in_band,
        )
    return {
        "trained_cells": trained,
        "corrected_read_cells": corrected_read_cells,
        "base_first_seed_in_band": base_first_in_band,
        "base_corrected_in_band": base_corrected_in_band,
        "instruct_corrected_in_band": instruct_corrected_in_band,
        "h1_surprise_escalation": h1_surprise,
    }


def run_phase3(*, smoke: bool, dry_run: bool) -> dict:
    """CONDITIONAL H2 lr x steps mini-sweep (only when phase1 falsified H3)."""
    phase_log("phase3_h2_sweep")
    from explore_persona_space.train.sft import TrainLoraConfig, train_lora

    base_recipe = C664.recipe_for("marker")
    # #664 marker band-stop budget proxy: use the in-loop step cap; 4x multiplies it.
    base_steps = 50  # #664 d1 band-stop landed at step 20-25; 50 is the 1x ceiling proxy
    results = []
    for lr in C.PHASE3_LRS:
        for mult in C.PHASE3_STEP_MULTS:
            cell_key = f"h2_lr{lr:.0e}_steps{mult}x"
            if dry_run:
                logger.info("[phase3][dry-run] would train %s (no GPU)", cell_key)
                continue
            c664 = C664.Cell(behavior="marker", source=C.PHASE3_SOURCE, arm=C.PHASE3_ARM, dose="d1")
            data_path = (
                C664.DATA_ROOT
                / ("train_smoke" if smoke else "train")
                / "marker"
                / (c664.eval_key + ".jsonl")
            )
            assert data_path.exists(), f"phase3 mix missing: {data_path}"
            out_dir = C.ADAPTER_OUT / (cell_key + ("_smoke" if smoke else ""))
            kwargs = base_recipe.train_kwargs(
                dose="d1", gpu_id=0, run_name=f"issue734_{cell_key}", seed=42
            )
            kwargs["lr"] = lr
            kwargs["marker_band_stop"] = False  # the sweep runs to a fixed step budget
            kwargs["max_steps"] = 2 if smoke else base_steps * mult
            kwargs.pop("warmup_steps", None)
            traj_dir = C.H1_TRAJECTORY_ROOT / cell_key
            traj_dir.mkdir(parents=True, exist_ok=True)
            kwargs["marker_band_trajectory_path"] = str(traj_dir / "marker_band_trajectory.json")
            cfg = TrainLoraConfig(
                hf_upload=not smoke,
                hf_repo=C.HF_MODEL_REPO,
                hf_path_in_repo=f"{C.HF_H1_ADAPTER_PREFIX}/{cell_key}",
                **kwargs,
            )
            os.environ["EPM_SKIP_INLINE_CHECKPOINT_UPLOAD"] = "1"
            try:
                train_lora(C.INSTRUCT_ID, str(data_path), str(out_dir), cfg=cfg)
            finally:
                import wandb

                if wandb.run is not None:
                    wandb.finish()
            results.append(cell_key)
    return {"swept_cells": results}


def upload_artifacts(*, smoke: bool, dry_run: bool) -> None:
    """Push the corrected-reread JSONs + H1 band-stop/trajectory JSONs to the HF
    data repo (eval_results stay JSON/text). Adapters were pushed by train_lora.

    Per CLAUDE.md Upload Policy: raw completions + analysis JSONs MUST land on HF
    BEFORE pod termination. Uses ONE upload_folder commit (never a per-file loop)."""
    phase_log("phase_upload")
    if dry_run or smoke:
        logger.info("[upload] dry-run/smoke -- skipping HF upload")
        return
    from explore_persona_space.orchestrate import hub

    src = C.EVAL_ROOT
    if not src.exists():
        logger.warning("[upload] no eval_results to upload at %s", src)
        return
    # One folder commit for all #734 eval JSONs (the 504-storm-safe shape, gotchas
    # #664). src is a DIRECTORY -> the folder branch (upload_as_file default False).
    hub._upload(
        src,  # Path (directory); folder upload, NOT a single-file path
        repo_id=C.HF_DATA_REPO,
        repo_type="dataset",
        path_in_repo="issue734_marker_slot_reread/eval_results",
    )
    logger.info("[upload] uploaded %s -> %s", src, C.HF_DATA_REPO)


def main() -> int:
    ap = argparse.ArgumentParser(description="Issue #734 corrected marker re-read + H1 driver.")
    ap.add_argument(
        "--phase",
        default="all",
        choices=[
            "phase0",
            "phase1",
            "phase1p5",
            "setup_h1_mix",
            "phase2",
            "phase3",
            "upload",
            "all",
        ],
    )
    ap.add_argument("--cells", type=int, default=None, help="limit cells (smoke: --cells 1)")
    ap.add_argument(
        "--seeds",
        type=str,
        default=None,
        help="comma-separated subset of H1 model-init seeds to run (e.g. --seeds 42 for the "
        "1-seed-1-model smoke, --seeds 42,137 for a re-run subset). Each must be in H1_SEEDS; "
        "default None = all H1_SEEDS. Restricts phase2 only.",
    )
    ap.add_argument("--smoke", action="store_true", help="tiny-slice smoke (2 probes / 2 steps)")
    ap.add_argument(
        "--force-rerun",
        action="store_true",
        help="re-run Phase-1 cells even when a valid marker_slot_corrected.json is already "
        "on disk (default: skip complete cells -- the crash-fix-round-5 resume guard so a "
        "relaunch after the Phase-2 deadlock fix goes straight to Phase 2)",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="exercise cell plumbing + sentinel + [phase=done] without a GPU forward "
        "(GPU-bound-phase substitute coverage)",
    )
    args = ap.parse_args()

    seeds: list[int] | None = None
    if args.seeds:
        seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    C.require_credentials()
    note_extra: dict = {}
    try:
        if args.phase in ("phase0", "all"):
            phase_log("phase0_token_id_split")
            subprocess.run(
                [sys.executable, str(C.REPO / "scripts/issue734_phase0.py")]
                + (["--smoke"] if args.smoke else []),
                check=True,
                cwd=C.REPO,
                env={**os.environ},
            )

        if args.phase == "phase1p5" or args.phase == "all":
            phase_log("phase1p5_parity_probe")
            if not args.dry_run:
                ok = parity_probe(smoke=args.smoke)
                note_extra["parity_probe_pass"] = ok
                if not ok:
                    # §7.5 reuse-fitness HALT: gauge/load wrong -- do NOT trust the sweep.
                    raise SystemExit(
                        "issue734 §7.5 parity probe FAILED: corrected reader did not reproduce "
                        "the in-loop band-stop within ~2 nat. Diagnose rsLoRA scaling / adapter "
                        "load before the sweep."
                    )

        falsified = False
        if args.phase in ("phase1", "all"):
            p1 = run_phase1(
                cells_limit=args.cells,
                smoke=args.smoke,
                dry_run=args.dry_run,
                force_rerun=args.force_rerun,
            )
            note_extra["phase1"] = p1
            falsified = bool(p1.get("h3_falsified"))

        # setup_h1_mix MUST precede phase2 AND a standalone phase3: both consume the
        # librarian-contra-d1-seed42 marker mix train_h1_cell / run_phase3 assert
        # exists (the mix #664 never uploaded; the git-clone-only GCP lane cannot
        # stage it -- issue #734 crash-fix round 1). build_h1_mix is idempotent, so
        # firing it for both is a no-op once the mix is current. In --phase all,
        # phase3 (conditional) runs after phase2, so the mix is already built.
        if args.phase in ("setup_h1_mix", "phase2", "phase3", "all"):
            note_extra["setup_h1_mix"] = run_setup_h1_mix(smoke=args.smoke, dry_run=args.dry_run)

        if args.phase in ("phase2", "all"):
            p2 = run_phase2(
                cells_limit=args.cells, smoke=args.smoke, dry_run=args.dry_run, seeds=seeds
            )
            note_extra["phase2"] = p2

        if args.phase == "phase3" or (args.phase == "all" and falsified):
            p3 = run_phase3(smoke=args.smoke, dry_run=args.dry_run)
            note_extra["phase3"] = p3
        elif args.phase == "all":
            logger.info("[phase3] H3 NOT falsified -- skipping the H2 dose sweep (plan §7.5).")

        if args.phase in ("upload", "all"):
            upload_artifacts(smoke=args.smoke, dry_run=args.dry_run)

        write_sentinel(
            "epm:results",
            note=json.dumps(
                {
                    "phase": args.phase,
                    "smoke": args.smoke,
                    "dry_run": args.dry_run,
                    **note_extra,
                }
            ),
            extra={"gate": False, "blocks_pipeline": False},
        )
        # Terminal [phase=done] -- the SINGLE graceful-exit line (poll_pipeline.py).
        phase_log("done")
        return 0
    except Exception:
        logger.exception("[issue734] dispatcher failed")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
