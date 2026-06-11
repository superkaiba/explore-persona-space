# ruff: noqa: RUF002, RUF003  # research code uses Greek letters (ρ, Δ), × and − legitimately
"""Task #480 dispatcher — payload-swap of #411's rig (marker vs sycophancy).

Pipeline (smoke = sweep with one source via --only-source/--smoke):

  Phase 0 (vLLM): generate base on-policy R for each distinct persona
                  (sources + bystanders + no-persona) on Q_train_200.
                  Subprocess-isolated (vLLM teardown safety).

  Per-cell discipline (one cell == one source persona, sequential on 1×H100):
    1. Build training pool (CPU, in-process) — 700 rows per source.
    2. Train LoRA (marker-only-loss collator with #474 suppress_at_post_response_slot=True).
    3. Merge LoRA into base on disk.
    4. Phase 2a (vLLM): generate R_trained for the 24 panel × 50 Q_eval grid.
                        Subprocess-isolated.
    5. Phase 2b (HF Transformers): post-response-slot log P(marker) trained − base.
                                   Subprocess-isolated (vLLM teardown safety).
    6. shutil.rmtree(merged/) before next source (MooseFS quota).
    7. Per-source sentinel JSON to /workspace/logs/issue-480-<source>-results.json.

  Phase 3 (analysis): pivot per-source per-panel logprob JSONs into
                      ``marker_delta_matrix.json`` (138 rows), inner-join
                      with #470's frozen ``predictor_comparison.json``, run
                      H1 + H2 stats package (cell Spearman + bootstrap +
                      paired test), write 6 figures.

End-of-sweep sentinel: /workspace/logs/issue-480-epm_results-<epoch>.json
(poll_pipeline-compatible schema: sentinel_schema_version=1, kind=epm:results,
version=1, etc.) — written only when every requested source completed AND
analysis produced the headline numbers.

Pod-side discipline (CLAUDE.md):
- EPM_SKIP_INLINE_CHECKPOINT_UPLOAD=1 in child env per cell.
- NEVER shells out to scripts/task.py (branch-guard would refuse).
- Every subprocess.* call passes env={**os.environ}; load_dotenv() at module-top.
- [phase=...] log lines, terminating in [phase=done] on graceful exit (poll_pipeline contract).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shutil
import socket
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

log = logging.getLogger("issue_480.dispatch")

DEFAULT_SOURCES = (
    "villain",
    "comedian",
    "assistant",
    "qwen_default",
    "software_engineer",
    "kindergarten_teacher",
)
BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_SEED = 42
HF_MODEL_REPO = "superkaiba1/explore-persona-space"
HF_DATA_REPO = "superkaiba1/explore-persona-space-data"

SENTINEL_SCHEMA_VERSION = 1

# HF data-repo location of the wrong-claim Q pools (inherited from #411).
HF_WRONG_CLAIMS_SUBDIR = "issue411_sycophancy_cosine_gradient/data/wrong_claims"

# ── band_stop recipe constants (same-issue follow-up round 2, plan v3) ──────
# HF data-repo location + PINNED revision of the parent run's 700-row
# contrastive train pools (reused byte-for-byte; the revision pin makes
# byte-for-byte verifiable — plan §10 Reproducibility Card).
HF_TRAIN_POOL_SUBDIR = "issue480_marker_payload_swap/train_pools"
TRAIN_POOL_REVISION = "3c8fecb937c81c13036a9697be1e4e716755321e"
TRAIN_POOL_EXPECTED_ROWS = 700
BAND_STOP_WANDB_PROJECT = "issue480-band-stopped-anchor-rerun"
BAND_STOP_HF_ADAPTER_ROOT = "adapters/issue_480_band_stop"
# THE one-variable recipe swap (plan §4 table): lr 5e-6 (marker-only LR is the
# over/under dial — Source: marker-training-recipe rule, #329/#478), strength
# bought through a fixed 12-epoch cap (≈528 optimizer steps; Source: bounded
# by #519/#520 + parent pace, smoke-gated), checkpoints every 20 steps with
# save_only_model, band callback in log-only mode probing every 5 steps.
BAND_STOP_LR = 5e-6
BAND_STOP_EPOCH_CAP = 12
BAND_STOP_CKPT_EVERY_STEPS = 20
BAND_STOP_PROBE_EVERY_STEPS = 5
# Firing-anchor pick target: first checkpoint with trained source mean
# log P(marker) >= -1.0 nat (absolute, not delta — emission onset is an
# absolute trained-log-P event; Source: #456/#398, smoke-gated).
FIRING_TARGET_LOGP_NATS = -1.0
# Graded (upload-only) anchor: delta nearest 8.5 within [5, 12] nat; mild
# overshoot window (12, 15] preferred over below-band when out of band.
GRADED_BAND_LOW = 5.0
GRADED_BAND_HIGH = 12.0
GRADED_BAND_CENTER = 8.5
GRADED_OVERSHOOT_HIGH = 15.0
# Bystander-resolution gate (plan §4 step 6): informativeness constants are
# the round-1 pre-registered criterion (= MIN_NONZERO_CELLS /
# MIN_DISTINCT_VALUES in scripts/issue480_emission_rate_concordance.py);
# the ceiling check calibrates against the parent SE pathology (2 cells at
# 1.0 / 4 >= 0.92 — correctly fails <=2).
GATE_MIN_NONZERO_CELLS = 5
GATE_MIN_DISTINCT_VALUES = 3
GATE_CEILING_RATE = 0.92
GATE_MAX_CEILING_CELLS = 2
REPICK_STRIDE_STEPS = 40
MAX_REEVALS = 2
EXPECTED_N_BYSTANDERS = 23

# ── graded_eval recipe constants (same-issue follow-up round 3, plan v4) ────
# Eval-only re-read of round 2's UNEVALUATED in-band (step-20) checkpoints:
# fetch the pinned graded adapter per source, merge, run the existing Phase
# 2a/2b on the same 24×50 grid, with the #534 adapter-application parity
# probe gating every cell. PRIMARY DV this round = per-cell median Δlog P(※)
# (the band_stop machinery's four-float capture, unchanged).
# Pinned HF model-repo revision of the round-2 graded adapter upload
# (Source: #480 body Follow-up 2 Artifacts row; plan v4 §4 step 1 / §11).
GRADED_ADAPTER_REVISION = "3b3d1d940200338bf8143556e85262926c1b26d3"
# Committed round-2 per-source pick records (in git on this branch — read
# from the pod's checkout, NEVER via task.py; plan §4 step 2).
ROUND2_SLAB_ROOT = Path("eval_results/issue_480/band-stopped-anchor-rerun")
# #534 adapter-application parity tolerance, both model sides
# (Source: marker-leakage-measurement.md adapter-application assert).
PARITY_TOLERANCE_NATS = 1.0
# Probe shape must match what round 2's in-loop band callback used:
# max_rows=32 (TrainLoraConfig.marker_band_probe_max_rows default) and
# max_length = max(train max_length, 2048) (the _maybe_attach_marker_band_stop
# derivation with marker_band_probe_max_length=None).
PARITY_PROBE_MAX_ROWS = 32
# HF data-repo subdir for this round's artifact export (plan §4 post-cells).
INBAND_HF_DATA_SUBDIR = "issue480_inband_logprob_concordance"
# Issue branch the pod pushes per-source eval JSONs to (fail-loud; plan §4).
INBAND_GIT_BRANCH = "issue-480-inband-logprob-concordance"


def _logs_root() -> Path:
    """Sentinel/log root — ``/workspace/logs`` on pods (poll_pipeline contract).

    Overridable via ``EPM_I480_LOGS_ROOT`` so the graded_eval CPU dry-run can
    execute on the dev VM (no ``/workspace``). Unset env → byte-identical
    default for every pod-side recipe.
    """
    return Path(os.environ.get("EPM_I480_LOGS_ROOT", "/workspace/logs"))


def _ensure_train_pool(local_path: Path, source: str) -> Path:
    """Auto-download the parent's 700-row train pool for ``source`` (pinned revision).

    Mirrors ``_ensure_wrong_claim_pool``: fetch via ``hf_hub_download`` at the
    PINNED parent revision and copy into ``local_path``. Fails loud on a row
    count != 700 (the reuse premise is byte-for-byte — a short pool means the
    wrong artifact resolved).
    """
    if not local_path.exists():
        from huggingface_hub import hf_hub_download

        hub_filename = f"{HF_TRAIN_POOL_SUBDIR}/{source}_train_pool.jsonl"
        log.info(
            "[phase=preflight] train pool %s not found locally; downloading %s@%s from %s",
            local_path,
            hub_filename,
            TRAIN_POOL_REVISION[:12],
            HF_DATA_REPO,
        )
        cached = hf_hub_download(
            repo_id=HF_DATA_REPO,
            filename=hub_filename,
            repo_type="dataset",
            revision=TRAIN_POOL_REVISION,
        )
        local_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(cached, local_path)
    n_rows = sum(1 for line in local_path.read_text().splitlines() if line.strip())
    if n_rows != TRAIN_POOL_EXPECTED_ROWS:
        raise RuntimeError(
            f"train pool {local_path} has {n_rows} rows, expected "
            f"{TRAIN_POOL_EXPECTED_ROWS} (revision {TRAIN_POOL_REVISION})"
        )
    log.info("[phase=preflight] train pool ready at %s (%d rows)", local_path, n_rows)
    return local_path


def _ensure_wrong_claim_pool(local_path: Path, kind: str) -> Path:
    """Auto-download a wrong-claim Q pool from the #411 data subdir if missing locally.

    ``kind`` ∈ {"train_200", "eval_50"}. The HF filename matches: the file is fetched
    via :func:`huggingface_hub.hf_hub_download` and then ``copied`` into ``local_path``
    so the caller's default arg path is satisfied byte-for-byte. We deliberately copy
    rather than symlink to keep relative-path consumers (`Path.open`, dataset scripts)
    immune to HF cache rotation.
    """
    if local_path.exists():
        return local_path
    from huggingface_hub import hf_hub_download

    hub_filename = f"{HF_WRONG_CLAIMS_SUBDIR}/{kind}.jsonl"
    log.info(
        "[phase=preflight] wrong-claim pool %s not found locally; downloading %s from %s",
        local_path,
        hub_filename,
        HF_DATA_REPO,
    )
    cached = hf_hub_download(
        repo_id=HF_DATA_REPO,
        filename=hub_filename,
        repo_type="dataset",
    )
    local_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(cached, local_path)
    log.info("[phase=preflight] wrong-claim pool ready at %s", local_path)
    return local_path


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
            env={**os.environ},
        ).strip()
    except Exception:
        return "unknown"


def _parse_sources(raw: str) -> list[str]:
    """Comma-separated list of sources; ``all`` (case-insensitive) expands to DEFAULT_SOURCES.

    Examples:
        ``--sources all``                          -> list(DEFAULT_SOURCES)
        ``--sources villain,comedian``             -> ["villain", "comedian"]
        ``--sources ALL`` / ``--sources  All ``    -> list(DEFAULT_SOURCES)

    The preflight loop downstream validates each name against ``SOURCE_PERSONAS``,
    so this only handles the ``all`` keyword expansion + comma-split + trim.
    """
    if raw.strip().lower() == "all":
        return list(DEFAULT_SOURCES)
    return [s.strip() for s in raw.split(",") if s.strip()]


def _persona_key(system_prompt: str | None) -> str:
    if system_prompt is None or system_prompt == "":
        return "_no_persona"
    return "sys_" + hashlib.sha256(system_prompt.encode("utf-8")).hexdigest()[:16]


def _collect_persona_specs(
    bystander_assignment: dict[str, dict[str, list[str] | str]],
    source_system_prompts: dict[str, str],
) -> dict[str, str | None]:
    """Build the persona spec dict for Phase 0 R generation.

    Returns {persona_key: system_prompt_or_None} covering:
      - one entry per source persona (key=source name)
      - one entry per DISTINCT bystander system prompt (key=sha hash)
      - one entry for the no-persona case (key="_no_persona", value=None)
    """
    specs: dict[str, str | None] = {}
    for src, sys_p in source_system_prompts.items():
        specs[src] = sys_p
    seen: set[str] = set()
    for info in bystander_assignment.values():
        for sp in info["system_prompts"]:
            key = _persona_key(sp)
            if key in seen:
                continue
            seen.add(key)
            specs[key] = sp
    specs["_no_persona"] = None
    return specs


def _phase0(
    persona_specs: dict[str, str | None],
    q_train: Path,
    r_base_dir: Path,
    seed: int,
) -> dict:
    """Run Phase 0 (vLLM base R generation) in a fresh subprocess."""
    sentinel = Path("/workspace/logs/issue-480-phase0-results.json")
    cmd = [
        "uv",
        "run",
        "python",
        "scripts/issue_480/i480_phase0_generate_R.py",
        "--system-prompts",
        json.dumps(persona_specs),
        "--q-train",
        str(q_train),
        "--out-dir",
        str(r_base_dir),
        "--seed",
        str(seed),
        "--sentinel-path",
        str(sentinel),
    ]
    log.info("[phase=phase0] spawning: %s", " ".join(cmd[:4]) + " ...")
    env = {**os.environ}
    env.setdefault("EPM_SKIP_INLINE_CHECKPOINT_UPLOAD", "1")
    t0 = time.time()
    subprocess.run(cmd, env=env, check=True)
    if not sentinel.exists():
        raise RuntimeError(f"Phase 0 ran but sentinel {sentinel} missing")
    with open(sentinel) as f:
        summary = json.load(f)
    summary["wall_seconds_dispatcher"] = round(time.time() - t0, 1)
    return summary


def _load_q_train(path: Path) -> list[str]:
    out: list[str] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            out.append(obj["wrong_claim"])
    return out


def _load_r_base_files(
    r_base_dir: Path, persona_specs: dict[str, str | None]
) -> dict[str, list[str]]:
    """Load Phase 0 R files into {persona_key: [response, ...]}."""
    out: dict[str, list[str]] = {}
    for key in persona_specs:
        p = r_base_dir / f"{key}.json"
        if not p.exists():
            raise FileNotFoundError(f"Phase 0 R file missing for persona_key={key}: {p}")
        with open(p) as f:
            data = json.load(f)
        out[key] = data["responses"]
    return out


def _build_training_pool_for_source(
    source: str,
    q_train: list[str],
    r_base_by_key: dict[str, list[str]],
    bystander_assignment: dict[str, dict[str, list[str] | str]],
    out_jsonl: Path,
    max_length: int,
) -> None:
    """In-process call to build_marker_pool (CPU, no GPU concerns).

    ``max_length`` must MATCH the training-side ``TrainLoraConfig.max_length``
    so the CPU-side row-length guard inside ``build_marker_pool`` fails LOUDLY
    at pool-build time if any row would later be silently truncated by TRL
    and crash the ``MarkerOnlyDataCollator(suppress_at_post_response_slot=
    True)`` branch ~2 min into Phase 1 (round-2 incident, pod-480).
    """
    from explore_persona_space.experiments.marker_implant_480 import SOURCE_PERSONAS  # noqa: F401
    from explore_persona_space.experiments.marker_implant_480.build_training_pool import (
        build_marker_pool,
    )

    bys_prompts = list(bystander_assignment[source]["system_prompts"])
    # Translate persona-key indexed R dict into the {source/persona_str: [R]} the
    # builder expects: source persona uses its own name, bystanders use their
    # system-prompt strings (so the builder can attach them as system prompts),
    # _no_persona uses the literal "_no_persona" key.
    r_by_persona: dict[str, list[str]] = {
        source: r_base_by_key[source],
        "_no_persona": r_base_by_key["_no_persona"],
    }
    for bys_sys in bys_prompts:
        r_by_persona[bys_sys] = r_base_by_key[_persona_key(bys_sys)]
    build_marker_pool(
        source=source,
        q_train=q_train,
        r_base_by_persona=r_by_persona,
        bystander_system_prompts=bys_prompts,
        output_path=out_jsonl,
        max_length=max_length,
    )


def _parent_train_cfg(source: str, seed: int, max_length: int):
    """Build the PARENT recipe's TrainLoraConfig (round-1 behavioral identity).

    Extracted from ``_train_and_merge`` (round-6) so the config-equality
    regression test in ``tests/test_i480_band_stop_dispatch.py`` pins every
    effective value against the round-1 run without training.

    Plan §10 Reproducibility Card + plan §11 Decision Rationale:
    lr=1e-5 (matches BOTH #411 AND #460/#474 marker rig);
    lora_dropout=0.0 (marker-rig convention, NOT 0.05 from #411 — see plan §11 row);
    marker_only_loss=True + tail_tokens=0 + #474 suppress_at_post_response_slot=True.
    max_length: round-3 fix — pulled from build-time guard's
    DEFAULT_TRAIN_MAX_LENGTH so pool-build and training see the same budget.
    Source: .claude/rules/marker-leakage-measurement.md (R-cap ~1024, eval-cap
    >=2048) + #260 (training-truncation -> silent zeros on the DV).

    Round-6 parent-recipe pin (code-review v5 binding concern
    ``parent-recipe-inherits-live-band-stop``): main's TrainLoraConfig now
    defaults ``marker_band_stop=True``, and ``train_lora`` attaches the LIVE
    [5,12] band-stop callback whenever ``marker_only_loss AND
    marker_band_stop``. The round-1 run executed on the pre-band-stop
    issue-480 branch (no callback; full 3 epochs to deliberate saturation),
    so a bare ``--recipe parent`` launch on THIS branch would early-stop
    training AND (via hf_upload=True at the round-1 adapter paths) overwrite
    the historical HF adapters with differently-trained ones. Pin it OFF —
    exactly the deliberate-saturation case ``marker-training-recipe.md``
    names for ``marker_band_stop=False``.

    Default-drift sweep vs the parent SHA 4b2b4bbee (every TrainLoraConfig
    field that did not exist there, and why no other pin is needed):
    - ``marker_band_stop=True`` — the ONLY behaviorally-live new default on
      this path (gates the callback attach); pinned False below.
    - ``marker_band_low/high_nats, marker_band_eval_every_steps,
      marker_band_min_steps, marker_band_probe_max_rows,
      marker_band_probe_max_length, marker_band_log_only,
      marker_band_trajectory_path`` — inert once the callback never attaches
      (``_maybe_attach_marker_band_stop`` returns before reading them).
    - ``save_only_model=False`` — inert under ``save_strategy="no"`` (and
      forwarded to SFTConfig only-when-True).
    - ``lora_targets=None`` — train_lora resolves None to the identical
      historical 7-module list (q/k/v/o + gate/up/down) with use_rslora=True,
      byte-identical to the parent SHA's hard-coded LoraConfig.
    All remaining field defaults are byte-identical between 4b2b4bbee and
    this branch (verified by direct dataclass-field extraction, round 6).
    """
    from explore_persona_space.experiments.marker_implant_480 import IM_END_ID, MARKER_TEXT
    from explore_persona_space.train.sft import TrainLoraConfig

    return TrainLoraConfig(
        gpu_id=0,
        epochs=3,
        lr=1e-5,
        lora_r=32,
        lora_alpha=64,
        lora_dropout=0.0,
        batch_size=4,
        grad_accum=4,  # effective batch 16
        max_length=max_length,
        warmup_ratio=0.05,
        seed=seed,
        run_name=f"issue480_{source}_seed{seed}",
        report_to="wandb",
        save_strategy="no",
        gradient_checkpointing=True,
        packing=False,
        marker_only_loss=True,
        marker_text=MARKER_TEXT,
        marker_tail_tokens=0,
        marker_suppress_at_post_response_slot=True,
        marker_im_end_token_id=IM_END_ID,
        # Round-6 pin — see docstring: parent = deliberate saturation, no
        # band-stop; without this a default launch early-stops AND clobbers
        # the round-1 HF adapters.
        marker_band_stop=False,
        hf_upload=True,
        hf_repo=HF_MODEL_REPO,
        hf_path_in_repo=f"adapters/issue_480/{source}_seed{seed}",
    )


def _train_and_merge(
    source: str,
    seed: int,
    train_jsonl: Path,
    output_dir: Path,
    max_length: int,
) -> tuple[Path, Path]:
    """Phase 1 — in-process LoRA train + merge.

    ``max_length`` is plumbed in (not hard-coded) so the SAME budget that the
    pool-build guard validated against is what TRL's ``SFTConfig.max_length``
    receives at training time. Round-2 incident (pod-480) was caused by a
    hard-coded ``max_length=1024`` here while base on-policy R can be up to
    2048 tokens — TRL right-truncated rows over 1024, dropped the trailing
    ``<|im_end|>``, and crashed the ``MarkerOnlyDataCollator(suppress_at_
    post_response_slot=True)`` branch ~2 min into Phase 1.
    """
    from explore_persona_space.train.sft import merge_lora, train_lora

    adapter_dir = output_dir / "adapter"
    merged_dir = output_dir / "merged"
    adapter_dir.mkdir(parents=True, exist_ok=True)

    cfg = _parent_train_cfg(source, seed, max_length)
    log.info(
        "[phase=train_%s] cfg: lr=%s r=%s alpha=%s dropout=%s epochs=%d "
        "marker_text=%r tail_tokens=%d suppress_post=%s",
        source,
        cfg.lr,
        cfg.lora_r,
        cfg.lora_alpha,
        cfg.lora_dropout,
        cfg.epochs,
        cfg.marker_text,
        cfg.marker_tail_tokens,
        cfg.marker_suppress_at_post_response_slot,
    )
    train_lora(
        base_model_path=BASE_MODEL,
        data_path=str(train_jsonl),
        output_dir=str(adapter_dir),
        cfg=cfg,
    )
    log.info("[phase=train_%s] merge -> %s", source, merged_dir)
    merge_lora(
        base_model_path=BASE_MODEL,
        adapter_path=str(adapter_dir),
        output_dir=str(merged_dir),
        gpu_id=0,
    )
    return adapter_dir, merged_dir


def _phase2a(
    source: str,
    seed: int,
    merged_dir: Path,
    eval_pool: Path,
    eval_out_dir: Path,
    sentinel_path: Path | None = None,
) -> Path:
    """Phase 2a — vLLM gen R_trained in fresh subprocess."""
    sentinel = sentinel_path or Path(f"/workspace/logs/issue-480-{source}-phase2a-results.json")
    cmd = [
        "uv",
        "run",
        "python",
        "scripts/issue_480/i480_phase2a_generate_R_trained.py",
        "--source",
        source,
        "--seed",
        str(seed),
        "--merged-model-path",
        str(merged_dir),
        "--eval-pool",
        str(eval_pool),
        "--out-dir",
        str(eval_out_dir),
        "--sentinel-path",
        str(sentinel),
    ]
    log.info("[phase=phase2a_%s] spawning: %s", source, " ".join(cmd))
    env = {**os.environ}
    env.setdefault("EPM_SKIP_INLINE_CHECKPOINT_UPLOAD", "1")
    subprocess.run(cmd, env=env, check=True)
    r_trained_path = eval_out_dir / "r_trained.json"
    if not r_trained_path.exists():
        raise RuntimeError(f"Phase 2a finished but {r_trained_path} missing")
    return r_trained_path


def _phase2b(
    source: str,
    seed: int,
    r_trained_path: Path,
    merged_dir: Path,
    eval_out_dir: Path,
    sentinel_path: Path | None = None,
    adapter_config_path: Path | None = None,
    slot_stats: str = "legacy",
    parity_probe_json: Path | None = None,
    trained_adapter_dir: Path | None = None,
) -> Path:
    """Phase 2b — HF logprob in fresh subprocess (vLLM workers reaped).

    ``slot_stats`` selects the scoring path inside i480_phase2b_logprob.py
    (round-6 fix, Codex v5 critical): the default ``"legacy"`` keeps the
    parent recipe on the verbatim round-1 ``_resolve_post_response_slot`` /
    ``_score_one`` implementation with the parent's exact output schema; the
    band_stop recipe opts in to ``"four-float"`` (compute_marker_slot_stats,
    #530 storage contract + gauge assert). The flag is appended only when
    non-legacy so the parent path's subprocess command stays identical to
    the round-1 invocation.
    """
    sentinel = sentinel_path or Path(f"/workspace/logs/issue-480-{source}-phase2b-results.json")
    out_path = eval_out_dir / "marker_logprob_eval.json"
    cmd = [
        "uv",
        "run",
        "python",
        "scripts/issue_480/i480_phase2b_logprob.py",
        "--source",
        source,
        "--seed",
        str(seed),
        "--r-trained-path",
        str(r_trained_path),
        "--merged-model-path",
        str(merged_dir),
        "--out-path",
        str(out_path),
        "--sentinel-path",
        str(sentinel),
    ]
    if slot_stats != "legacy":
        cmd += ["--slot-stats", slot_stats]
    if adapter_config_path is not None:
        cmd += ["--adapter-config-path", str(adapter_config_path)]
    if parity_probe_json is not None:
        # graded_eval recipe only (#534 adapter-application assert); the
        # parent and band_stop subprocess commands stay byte-identical.
        cmd += ["--parity-probe-json", str(parity_probe_json)]
    if trained_adapter_dir is not None:
        # graded_eval recipe only (round-3 parity-FAIL root-cause fix): score
        # the TRAINED side through the UNMERGED adapter — bf16 merge truncates
        # the tiny step-20 LoRA delta below the base-weight ULP, attenuating
        # the marker push ~2.1 nat (diagnostic: unmerged -8.93 vs recorded
        # -8.90; merged -11.02 in-process AND from disk). Phase 2a still
        # generates on the merged dir (parent convention).
        cmd += ["--trained-adapter-dir", str(trained_adapter_dir)]
    log.info("[phase=phase2b_%s] spawning: %s", source, " ".join(cmd))
    env = {**os.environ}
    env.setdefault("EPM_SKIP_INLINE_CHECKPOINT_UPLOAD", "1")
    subprocess.run(cmd, env=env, check=True)
    if not out_path.exists():
        raise RuntimeError(f"Phase 2b finished but {out_path} missing")
    return out_path


def _run_one_cell(
    source: str,
    seed: int,
    q_train: list[str],
    r_base_by_key: dict[str, list[str]],
    bystander_assignment: dict[str, dict[str, list[str] | str]],
    eval_pool: Path,
    slab_root: Path,
    runs_root: Path,
    max_length: int,
) -> dict:
    """Build pool → train+merge → Phase 2a → Phase 2b → rmtree merged, one source.

    ``max_length`` is the single source of truth shared by the pool-build
    guard and the training config; see ``_train_and_merge`` for the
    round-2-incident background that made this plumbing load-bearing.
    """
    t_start = time.time()
    output_dir = runs_root / f"{source}_seed{seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    eval_out_dir = slab_root / "per_source" / source / f"seed_{seed}"
    eval_out_dir.mkdir(parents=True, exist_ok=True)
    train_jsonl = output_dir / "train_pool.jsonl"

    log.info("=" * 70)
    log.info(
        "[phase=cell_%s] CELL START — output_dir=%s eval_out=%s max_length=%d",
        source,
        output_dir,
        eval_out_dir,
        max_length,
    )

    _build_training_pool_for_source(
        source=source,
        q_train=q_train,
        r_base_by_key=r_base_by_key,
        bystander_assignment=bystander_assignment,
        out_jsonl=train_jsonl,
        max_length=max_length,
    )
    _, merged_dir = _train_and_merge(source, seed, train_jsonl, output_dir, max_length)

    r_trained_path = _phase2a(source, seed, merged_dir, eval_pool, eval_out_dir)
    logprob_path = _phase2b(source, seed, r_trained_path, merged_dir, eval_out_dir)

    # Fail loud if adapter is empty (silent training failure).
    adapter_safetensors = list((output_dir / "adapter").glob("*.safetensors"))
    if not adapter_safetensors:
        raise RuntimeError(
            f"[{source}] adapter dir empty after training — upload may be stale "
            f"or training silently failed: {output_dir / 'adapter'}"
        )

    # Reap merged dir BEFORE next cell (MooseFS quota).
    if merged_dir.exists():
        log.info("[phase=cell_%s] rmtree(%s) for MooseFS quota", source, merged_dir)
        shutil.rmtree(merged_dir, ignore_errors=False)

    wall = time.time() - t_start
    log.info("[phase=cell_%s] CELL DONE wall=%.1fs", source, wall)
    return {
        "source": source,
        "seed": seed,
        "wall_seconds": round(wall, 1),
        "output_dir": str(output_dir),
        "eval_out_dir": str(eval_out_dir),
        "r_trained_path": str(r_trained_path),
        "logprob_path": str(logprob_path),
        "adapter_hf_path": f"adapters/issue_480/{source}_seed{seed}",
    }


def _band_stop_train_cfg(source: str, seed: int, max_length: int, traj_path: Path):
    """Build the band_stop recipe's TrainLoraConfig.

    Shared by ``_train_band_stop`` AND the adapter-config parity preflight
    (``_assert_band_stop_adapter_parity``) so the preflight checks the REAL
    training config — a drift between a preflight copy and the call site
    would defeat the check.
    """
    from explore_persona_space.experiments.marker_implant_480 import IM_END_ID, MARKER_TEXT
    from explore_persona_space.train.sft import TrainLoraConfig

    return TrainLoraConfig(
        gpu_id=0,
        epochs=BAND_STOP_EPOCH_CAP,
        lr=BAND_STOP_LR,
        lora_r=32,
        lora_alpha=64,
        lora_dropout=0.0,
        batch_size=4,
        grad_accum=4,  # effective batch 16
        max_length=max_length,
        warmup_ratio=0.05,
        seed=seed,
        run_name=f"issue480_bsr_{source}_seed{seed}",
        report_to="wandb",
        save_strategy="steps",
        save_steps=BAND_STOP_CKPT_EVERY_STEPS,
        save_only_model=True,
        gradient_checkpointing=True,
        packing=False,
        marker_only_loss=True,
        marker_text=MARKER_TEXT,
        marker_tail_tokens=0,
        marker_suppress_at_post_response_slot=True,
        marker_im_end_token_id=IM_END_ID,
        marker_band_stop=True,
        marker_band_log_only=True,
        marker_band_eval_every_steps=BAND_STOP_PROBE_EVERY_STEPS,
        marker_band_trajectory_path=str(traj_path),
        # The dispatcher uploads anchor/graded/capend adapters itself,
        # FAIL-LOUD, before any checkpoint deletion (upload-policy rule);
        # train_lora's best-effort soft-fail upload is therefore disabled.
        hf_upload=False,
    )


def _train_band_stop(
    source: str,
    seed: int,
    train_jsonl: Path,
    output_dir: Path,
    max_length: int,
    slab_root: Path,
) -> tuple[Path, Path]:
    """band_stop Phase 1 — in-process LoRA train to the fixed cap, log-only band callback.

    Returns ``(adapter_dir, trajectory_path)``. The adapter_dir root holds the
    cap-end adapter; ``checkpoint-<k>`` subdirs every 20 steps form the
    pickable ladder. Per-cell WandB hygiene (the parent-body-flagged
    instrumentation fix): ``wandb.finish()`` after training + assert
    ``wandb.run is None`` so the NEXT cell's HF WandbCallback re-inits a
    fresh run under its own run_name.
    """
    import wandb

    from explore_persona_space.train.sft import train_lora

    adapter_dir = output_dir / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    traj_path = slab_root / "trajectories" / f"{source}_seed{seed}_trajectory.json"
    traj_path.parent.mkdir(parents=True, exist_ok=True)

    cfg = _band_stop_train_cfg(source, seed, max_length, traj_path)
    # §8 risk row: a forgotten log-only flag would fire the live [5,12] stop
    # sub-emission and kill the primary emission DV by construction.
    if cfg.marker_band_log_only is not True:
        raise RuntimeError("band_stop recipe requires marker_band_log_only=True")
    log.info(
        "[phase=train_%s] band_stop cfg: lr=%s r=%s alpha=%s epochs=%d ckpt_every=%d "
        "probe_every=%d log_only=%s save_only_model=%s run_name=%s trajectory=%s",
        source,
        cfg.lr,
        cfg.lora_r,
        cfg.lora_alpha,
        cfg.epochs,
        cfg.save_steps,
        cfg.marker_band_eval_every_steps,
        cfg.marker_band_log_only,
        cfg.save_only_model,
        cfg.run_name,
        traj_path,
    )
    train_lora(
        base_model_path=BASE_MODEL,
        data_path=str(train_jsonl),
        output_dir=str(adapter_dir),
        cfg=cfg,
    )
    # Per-cell WandB run isolation (parent ran 5/6 cells into one run).
    if wandb.run is not None:
        wandb.finish()
    if wandb.run is not None:
        raise RuntimeError(
            f"[{source}] wandb.run still active after finish() — per-cell run "
            "isolation broken; the next cell would merge into this run."
        )
    if not traj_path.exists():
        raise RuntimeError(
            f"[{source}] band callback trajectory missing at {traj_path} — the "
            "log-only callback did not run (probe rows empty?); the anchor pick "
            "has nothing to key on."
        )
    return adapter_dir, traj_path


def _checkpoint_ladder(adapter_dir: Path, records: list[dict]) -> list[dict]:
    """Sorted pickable ladder: every ``checkpoint-<k>`` dir + the cap-end root adapter.

    Each entry: ``{"step", "dir", "cap_end", "trajectory_step", "logp_trained",
    "logp_base", "delta_nats"}`` — trajectory reads come from the probe record
    with the largest step <= the entry's step (probe cadence 5 divides the
    20-step checkpoint cadence, so periodic checkpoints resolve exactly; the
    cap-end uses the final probe).
    """
    if not records:
        raise RuntimeError(f"empty trajectory records for ladder at {adapter_dir}")
    entries: list[tuple[int, Path]] = []
    for p in sorted(adapter_dir.glob("checkpoint-*")):
        if not p.is_dir():
            continue
        try:
            step = int(p.name.rsplit("-", 1)[-1])
        except ValueError:
            continue
        entries.append((step, p))
    entries.sort(key=lambda e: e[0])
    if not entries:
        raise RuntimeError(
            f"no checkpoint-* dirs under {adapter_dir} — save_strategy='steps' "
            "did not produce a ladder; the anchor pick is impossible."
        )
    max_probe_step = max(r["step"] for r in records)
    cap_end_step = max(max_probe_step, entries[-1][0] + 1)
    ladder_raw = [(s, d, False) for s, d in entries] + [(cap_end_step, adapter_dir, True)]

    def _read_at(step: int) -> dict:
        cands = [r for r in records if r["step"] <= step]
        if not cands:
            raise RuntimeError(f"no trajectory probe at or before step {step}")
        return max(cands, key=lambda r: r["step"])

    ladder: list[dict] = []
    for step, d, cap_end in ladder_raw:
        rec = _read_at(step)
        ladder.append(
            {
                "step": step,
                "dir": str(d),
                "cap_end": cap_end,
                "trajectory_step": rec["step"],
                "logp_trained": rec["logp_trained"],
                "logp_base": rec["logp_base"],
                "delta_nats": rec["delta_nats"],
            }
        )
    return ladder


def _pick_anchors(ladder: list[dict]) -> dict:
    """Deterministic firing + graded anchor picks from the checkpoint ladder.

    Firing (evaluated): smallest step with trained log P >= -1.0 nat; cap-end
    flagged ``under_cap`` when never reached. Graded (upload-only): delta
    nearest 8.5 within [5, 12]; out-of-band prefers just-above within
    (12, 15] over below (mild overshoot beats floor for graded reads),
    else nearest-to-band; flagged. Ties break to the lower step everywhere.
    """
    firing_flags: list[str] = []
    in_target = [e for e in ladder if e["logp_trained"] >= FIRING_TARGET_LOGP_NATS]
    if in_target:
        firing = min(in_target, key=lambda e: e["step"])
    else:
        firing = ladder[-1]
        firing_flags.append("under_cap")

    graded_flags: list[str] = []
    in_band = [e for e in ladder if GRADED_BAND_LOW <= e["delta_nats"] <= GRADED_BAND_HIGH]
    if in_band:
        graded = min(in_band, key=lambda e: (abs(e["delta_nats"] - GRADED_BAND_CENTER), e["step"]))
    else:
        overshoot = [
            e for e in ladder if GRADED_BAND_HIGH < e["delta_nats"] <= GRADED_OVERSHOOT_HIGH
        ]
        if overshoot:
            graded = min(overshoot, key=lambda e: (e["delta_nats"], e["step"]))
            graded_flags.append("graded_out_of_band_overshoot")
        else:

            def _band_distance(e: dict) -> float:
                d = e["delta_nats"]
                return max(0.0, GRADED_BAND_LOW - d, d - GRADED_BAND_HIGH)

            graded = min(ladder, key=lambda e: (_band_distance(e), e["step"]))
            graded_flags.append("graded_out_of_band")
    return {
        "firing": {**firing, "flags": firing_flags},
        "graded": {**graded, "flags": graded_flags},
        "firing_target_logp_nats": FIRING_TARGET_LOGP_NATS,
        "graded_band": [GRADED_BAND_LOW, GRADED_BAND_HIGH],
        "graded_center": GRADED_BAND_CENTER,
    }


def _next_repick_step(
    ladder_steps: list[int], current: int, *, ceiling_violated: bool
) -> tuple[int | None, str | None]:
    """Pure re-pick stepper for the gated anchor-eval loop (plan §4 step 6).

    Extracted (round-6) so the plan-critical boundary rules carry a committed
    unit test. Precedence is decided by the CALLER (ceiling checked before
    floor — a bimodal panel steps BACK); this helper only resolves the next
    ladder step in the requested direction with the ±REPICK_STRIDE_STEPS
    stride.

    Returns ``(next_step, flag)``: exactly one side is non-None. Flags:
    ``repick_exhausted_low`` when no ladder step sits at or below
    ``current - stride`` (first-checkpoint clamp); ``floor_limited`` when no
    ladder step sits at or above ``current + stride`` (cap-end clamp).
    """
    if ceiling_violated:
        cands = [s for s in ladder_steps if s <= current - REPICK_STRIDE_STEPS]
        if not cands:
            return None, "repick_exhausted_low"
        return max(cands), None
    cands = [s for s in ladder_steps if s >= current + REPICK_STRIDE_STEPS]
    if not cands:
        return None, "floor_limited"
    return min(cands), None


def _bystander_gate(logprob_path: Path, source: str) -> dict:
    """Bystander-resolution gate on a Phase-2b output (plan §4 step 6).

    (a) informative — >=5 nonzero emission cells AND >=3 distinct values;
    (b) sub-ceiling — <=2 of 23 cells with emission >= 0.92. Source-side
    saturation never gates (the source SHOULD fire — it is the implant).
    """
    with open(logprob_path) as f:
        payload = json.load(f)
    bys = {p: s for p, s in payload["per_panel"].items() if p != source}
    if len(bys) != EXPECTED_N_BYSTANDERS:
        raise RuntimeError(
            f"[{source}] expected {EXPECTED_N_BYSTANDERS} bystander panels, got {len(bys)}"
        )
    rates = [float(s["mean_emission_rate"]) for s in bys.values()]
    n_nonzero = sum(1 for r in rates if r > 0.0)
    n_distinct = len(set(rates))
    n_ceiling = sum(1 for r in rates if r >= GATE_CEILING_RATE)
    informative = n_nonzero >= GATE_MIN_NONZERO_CELLS and n_distinct >= GATE_MIN_DISTINCT_VALUES
    sub_ceiling = n_ceiling <= GATE_MAX_CEILING_CELLS
    return {
        "n_bystanders": len(rates),
        "n_nonzero": n_nonzero,
        "n_distinct": n_distinct,
        "n_ceiling": n_ceiling,
        "informative": informative,
        "sub_ceiling": sub_ceiling,
        "passes": informative and sub_ceiling,
        "criteria": {
            "min_nonzero_cells": GATE_MIN_NONZERO_CELLS,
            "min_distinct_values": GATE_MIN_DISTINCT_VALUES,
            "ceiling_rate": GATE_CEILING_RATE,
            "max_ceiling_cells": GATE_MAX_CEILING_CELLS,
        },
    }


_TOKENIZER_FILES = (
    "tokenizer_config.json",
    "tokenizer.json",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
    "added_tokens.json",
    "chat_template.jinja",
)


def _ensure_ckpt_tokenizer(ckpt_dir: Path, adapter_dir: Path) -> None:
    """Copy tokenizer files from the cap-end adapter root into a step checkpoint.

    ``merge_lora`` loads the tokenizer from the adapter path; Trainer step
    checkpoints may omit tokenizer files depending on the transformers
    version, while the root adapter dir always has them (train_lora calls
    ``tokenizer.save_pretrained``). No-op when the checkpoint already has a
    tokenizer_config.json; fails loud when neither location has one.
    """
    if (ckpt_dir / "tokenizer_config.json").exists():
        return
    if not (adapter_dir / "tokenizer_config.json").exists():
        raise RuntimeError(
            f"tokenizer files missing from BOTH {ckpt_dir} and {adapter_dir} — "
            "cannot merge this checkpoint."
        )
    copied = []
    for name in _TOKENIZER_FILES:
        src = adapter_dir / name
        if src.exists():
            shutil.copyfile(src, ckpt_dir / name)
            copied.append(name)
    log.info("[phase=pick] copied tokenizer files %s -> %s", copied, ckpt_dir)


def _eval_anchor(
    source: str,
    seed: int,
    step: int,
    ckpt_dir: Path,
    adapter_dir: Path,
    eval_pool: Path,
    eval_out_dir: Path,
    output_dir: Path,
) -> tuple[Path, Path]:
    """Merge ``ckpt_dir`` and run Phase 2a + 2b for one anchor step.

    Artifacts land under ``eval_out_dir/anchor_step_<step>/`` (kept per
    evaluated depth — the analyzer's anchor-depth sensitivity read needs
    every evaluated checkpoint, and re-picks must not clobber earlier
    evals). The merged dir is reaped after Phase 2b (MooseFS quota).
    """
    adapter_config = ckpt_dir / "adapter_config.json"
    if not adapter_config.exists():
        raise RuntimeError(
            f"[{source}] {adapter_config} missing — checkpoint is not a loadable "
            "PEFT adapter (smoke gate (a) condition)."
        )
    _ensure_ckpt_tokenizer(ckpt_dir, adapter_dir)

    from explore_persona_space.train.sft import merge_lora

    merged_dir = output_dir / "merged"
    if merged_dir.exists():
        shutil.rmtree(merged_dir)
    log.info("[phase=merge_%s] step=%d %s -> %s", source, step, ckpt_dir, merged_dir)
    merge_lora(BASE_MODEL, str(ckpt_dir), str(merged_dir), gpu_id=0)

    step_dir = eval_out_dir / f"anchor_step_{step}"
    step_dir.mkdir(parents=True, exist_ok=True)
    r_trained_path = _phase2a(
        source,
        seed,
        merged_dir,
        eval_pool,
        step_dir,
        sentinel_path=Path(f"/workspace/logs/issue-480-bsr-{source}-phase2a-step{step}.json"),
    )
    logprob_path = _phase2b(
        source,
        seed,
        r_trained_path,
        merged_dir,
        step_dir,
        sentinel_path=Path(f"/workspace/logs/issue-480-bsr-{source}-phase2b-step{step}.json"),
        adapter_config_path=adapter_config,
        slot_stats="four-float",
    )
    log.info("[phase=cell_%s] rmtree(%s) for MooseFS quota", source, merged_dir)
    shutil.rmtree(merged_dir, ignore_errors=False)
    return step_dir, logprob_path


def _upload_adapter_or_die(local_dir: Path, path_in_repo: str) -> str:
    """Upload an adapter dir to the HF model repo; raise on any failure.

    ``upload_model`` returns "" on failure (soft) — the band_stop recipe
    deletes checkpoints after upload, so a soft-fail here would silently
    lose the only copy. Fail loud BEFORE any deletion (upload-policy rule).
    """
    from explore_persona_space.orchestrate.hub import upload_model

    hub_path = upload_model(str(local_dir), repo_id=HF_MODEL_REPO, path_in_repo=path_in_repo)
    if not hub_path:
        raise RuntimeError(
            f"adapter upload FAILED for {local_dir} -> {HF_MODEL_REPO}/{path_in_repo}; "
            "refusing to continue (checkpoints would be deleted after this step)."
        )
    log.info("[phase=upload] %s -> %s", local_dir, hub_path)
    return hub_path


def _run_one_cell_band_stop(
    source: str,
    seed: int,
    eval_pool: Path,
    slab_root: Path,
    runs_root: Path,
    max_length: int,
) -> dict:
    """band_stop cell: fetch pool → train-to-cap → pick → gated eval loop → persist.

    Smoke IS this exact path with ``--only-source comedian`` (architectural
    parity: same dispatcher, same subprocess shapes, same teardown).
    """
    t_start = time.time()
    output_dir = runs_root / f"{source}_seed{seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    eval_out_dir = slab_root / "per_source" / source / f"seed_{seed}"
    eval_out_dir.mkdir(parents=True, exist_ok=True)
    train_jsonl = output_dir / "train_pool.jsonl"

    log.info("=" * 70)
    log.info(
        "[phase=cell_%s] BAND-STOP CELL START — output_dir=%s eval_out=%s max_length=%d",
        source,
        output_dir,
        eval_out_dir,
        max_length,
    )

    # 1. Pool (reused byte-for-byte from the parent, pinned revision).
    _ensure_train_pool(train_jsonl, source)

    # 2-3. Train to the fixed cap with the log-only band callback + per-cell
    #      WandB finish.
    adapter_dir, traj_path = _train_band_stop(
        source, seed, train_jsonl, output_dir, max_length, slab_root
    )
    with open(traj_path) as f:
        traj = json.load(f)
    records = traj["records"]

    # 4. Deterministic anchor pick from the trajectory + checkpoint ladder.
    ladder = _checkpoint_ladder(adapter_dir, records)
    picks = _pick_anchors(ladder)
    by_step = {e["step"]: e for e in ladder}
    ladder_steps = [e["step"] for e in ladder]
    log.info(
        "[phase=pick_%s] ladder=%d ckpts (%d..%d) firing=%s%s graded=%s%s",
        source,
        len(ladder),
        ladder_steps[0],
        ladder_steps[-1],
        picks["firing"]["step"],
        picks["firing"]["flags"],
        picks["graded"]["step"],
        picks["graded"]["flags"],
    )

    # 5-6. Merge + eval the firing anchor, bystander-resolution gate with
    #      bounded re-picks (<=2 re-evals, stride +-40 steps).
    evaluations: list[dict] = []
    evaluated_steps: set[int] = set()
    flags: list[str] = list(picks["firing"]["flags"])
    current = picks["firing"]["step"]
    accepted_step: int | None = None
    for attempt in range(1 + MAX_REEVALS):
        if current in evaluated_steps:
            flags.append("repick_oscillation")
            break
        step_dir, logprob_path = _eval_anchor(
            source,
            seed,
            current,
            Path(by_step[current]["dir"]),
            adapter_dir,
            eval_pool,
            eval_out_dir,
            output_dir,
        )
        gate = _bystander_gate(logprob_path, source)
        evaluated_steps.add(current)
        evaluations.append(
            {
                "step": current,
                "attempt": attempt,
                "step_dir": str(step_dir),
                "gate": gate,
            }
        )
        log.info(
            "[phase=gate_%s] step=%d nonzero=%d distinct=%d ceiling=%d passes=%s",
            source,
            current,
            gate["n_nonzero"],
            gate["n_distinct"],
            gate["n_ceiling"],
            gate["passes"],
        )
        if gate["passes"]:
            accepted_step = current
            break
        if attempt == MAX_REEVALS:
            break
        # Precedence: ceiling first (stepping BACK fixes saturation; a
        # simultaneously floored+saturated bimodal panel is treated as
        # saturated — documented in anchor_pick.json via the gate dict).
        next_step, clamp_flag = _next_repick_step(
            ladder_steps, current, ceiling_violated=not gate["sub_ceiling"]
        )
        if next_step is None:
            assert clamp_flag is not None
            flags.append(clamp_flag)
            break
        current = next_step

    if accepted_step is None:
        # Keep the evaluated checkpoint closest to satisfying the gate:
        # max nonzero cells SUBJECT TO <=2 ceiling cells; ties -> lower step.
        # When every evaluated step violates the ceiling, fall back to
        # min ceiling count, then max nonzero, then lower step.
        flags.append("gate_unmet")
        ok = [e for e in evaluations if e["gate"]["n_ceiling"] <= GATE_MAX_CEILING_CELLS]
        if ok:
            accepted_step = sorted(ok, key=lambda e: (-e["gate"]["n_nonzero"], e["step"]))[0][
                "step"
            ]
        else:
            accepted_step = sorted(
                evaluations,
                key=lambda e: (e["gate"]["n_ceiling"], -e["gate"]["n_nonzero"], e["step"]),
            )[0]["step"]
    accepted = next(e for e in evaluations if e["step"] == accepted_step)

    # Promote the accepted anchor's artifacts to the canonical per-source
    # layout the analyzer reads (per_source/<src>/seed_<seed>/...).
    accepted_dir = Path(accepted["step_dir"])
    for fname in ("marker_logprob_eval.json", "r_trained.json"):
        shutil.copyfile(accepted_dir / fname, eval_out_dir / fname)
    if (accepted_dir / "raw_completions").exists():
        shutil.copytree(
            accepted_dir / "raw_completions",
            eval_out_dir / "raw_completions",
            dirs_exist_ok=True,
        )

    # anchor_pick.json — the per-source pick record (plan §4 step 4).
    anchor_pick = {
        "source": source,
        "seed": seed,
        "recipe": "band_stop",
        "ladder": ladder,
        "picks": picks,
        "evaluations": evaluations,
        "accepted_step": accepted_step,
        "accepted_gate": accepted["gate"],
        "flags": flags,
        "repick_stride_steps": REPICK_STRIDE_STEPS,
        "max_reevals": MAX_REEVALS,
        "trajectory_path": str(traj_path),
        "git_commit_sha": _git_sha(),
        "hostname": socket.gethostname(),
        "timestamp_utc": datetime.now(UTC).isoformat(),
    }
    anchor_pick_path = eval_out_dir / "anchor_pick.json"
    with open(anchor_pick_path, "w") as f:
        json.dump(anchor_pick, f, indent=2, ensure_ascii=False)
    log.info("[phase=pick_%s] anchor_pick -> %s", source, anchor_pick_path)

    # 7. Persist adapters to HF (fail-loud, BEFORE checkpoint deletion):
    #    accepted anchor, graded (upload-only, never evaluated this run),
    #    cap-end.
    hf_paths = {
        "anchor": _upload_adapter_or_die(
            Path(by_step[accepted_step]["dir"]),
            f"{BAND_STOP_HF_ADAPTER_ROOT}/{source}_seed{seed}_anchor",
        ),
        "graded": _upload_adapter_or_die(
            Path(by_step[picks["graded"]["step"]]["dir"]),
            f"{BAND_STOP_HF_ADAPTER_ROOT}/{source}_seed{seed}_graded",
        ),
        "capend": _upload_adapter_or_die(
            adapter_dir, f"{BAND_STOP_HF_ADAPTER_ROOT}/{source}_seed{seed}_capend"
        ),
    }

    # Reap checkpoint ladder AFTER verified uploads (MooseFS quota).
    for e in ladder:
        d = Path(e["dir"])
        if d != adapter_dir and d.exists():
            shutil.rmtree(d, ignore_errors=False)
    log.info("[phase=cell_%s] checkpoint ladder reaped after verified uploads", source)

    wall = time.time() - t_start
    cell = {
        "source": source,
        "seed": seed,
        "recipe": "band_stop",
        "wall_seconds": round(wall, 1),
        "output_dir": str(output_dir),
        "eval_out_dir": str(eval_out_dir),
        "trajectory_path": str(traj_path),
        "anchor_pick_path": str(anchor_pick_path),
        "accepted_step": accepted_step,
        "accepted_gate": accepted["gate"],
        "flags": flags,
        "n_anchor_evals": len(evaluations),
        "logprob_path": str(eval_out_dir / "marker_logprob_eval.json"),
        "r_trained_path": str(eval_out_dir / "r_trained.json"),
        "adapter_hf_path": f"{BAND_STOP_HF_ADAPTER_ROOT}/{source}_seed{seed}_anchor",
        "adapter_hf_paths": hf_paths,
    }
    log.info("[phase=cell_%s] BAND-STOP CELL DONE wall=%.1fs flags=%s", source, wall, flags)
    return cell


def _phase3_analyze(
    *,
    seed: int,
    slab_root: Path,
    predictor_comparison_path: Path,
    syco_summary_path: Path,
    figures_dir: Path,
    trajectory_dir: Path | None = None,
) -> dict:
    """Phase 3 — pivot per-source logprob JSONs, run H1+H2 stats, emit figures."""
    sentinel = Path("/workspace/logs/issue-480-phase3-results.json")
    cmd = [
        "uv",
        "run",
        "python",
        "scripts/issue_480/i480_analyze.py",
        "--slab-root",
        str(slab_root),
        "--seed",
        str(seed),
        "--predictor-comparison",
        str(predictor_comparison_path),
        "--syco-summary",
        str(syco_summary_path),
        "--figures-dir",
        str(figures_dir),
        "--sentinel-path",
        str(sentinel),
    ]
    if trajectory_dir is not None:
        cmd += ["--trajectory-dir", str(trajectory_dir)]
    log.info("[phase=phase3] spawning analyze: %s", " ".join(cmd))
    subprocess.run(cmd, env={**os.environ}, check=True)
    if not sentinel.exists():
        raise RuntimeError(f"Phase 3 finished but {sentinel} missing")
    with open(sentinel) as f:
        return json.load(f)


def _write_final_sentinel(
    sources_requested: list[str],
    per_cell: list[dict],
    phase0_summary: dict,
    phase3_summary: dict | None,
    plan_deviations: list[str],
    final_path: Path,
    *,
    gpu_hours_budgeted: float = 6.5,
    wandb_project: str = "issue480-marker-payload-swap",
    hf_adapter_root: str = "adapters/issue_480",
    extra_note_fields: dict | None = None,
) -> None:
    """Write end-of-run sentinel in poll_pipeline-compatible schema.

    ``extra_note_fields`` (graded_eval recipe) is merged additively into the
    ``note`` dict — default None keeps parent/band_stop sentinels identical.
    """
    final_path.parent.mkdir(parents=True, exist_ok=True)
    headline = (phase3_summary or {}).get("headline_numbers", {}) if phase3_summary else {}
    payload = {
        "sentinel_schema_version": SENTINEL_SCHEMA_VERSION,
        "kind": "epm:results",
        "version": 1,
        "task_id": 480,
        "gate": None,
        "blocks_pipeline": False,
        "by": "dispatch_marker_480",
        "ts": datetime.now(UTC).isoformat(),
        "note": {
            "issue": 480,
            "sources_requested": sources_requested,
            "sources_completed": [c["source"] for c in per_cell],
            "n_completed": len(per_cell),
            "n_requested": len(sources_requested),
            "phase0_summary": phase0_summary,
            "per_cell": per_cell,
            "phase3_summary": phase3_summary,
            "headline_numbers": headline,
            "reproducibility_card": {
                "base_model": BASE_MODEL,
                "hf_model_repo": HF_MODEL_REPO,
                "hf_data_repo": HF_DATA_REPO,
                "adapter_paths": {
                    c["source"]: f"{HF_MODEL_REPO}/tree/main/{c['adapter_hf_path']}"
                    for c in per_cell
                },
            },
            "plan_deviations": plan_deviations,
            "gpu_hours_used_estimate": round(sum(c["wall_seconds"] for c in per_cell) / 3600, 2),
            "gpu_hours_budgeted": gpu_hours_budgeted,
            "final_commit_sha": _git_sha(),
            "hostname": socket.gethostname(),
            "wandb_url": f"n/a (per-cell wandb runs; project={wandb_project})",
            "hf_hub_url": f"https://huggingface.co/{HF_MODEL_REPO}/tree/main/{hf_adapter_root}",
        },
    }
    if extra_note_fields:
        payload["note"].update(extra_note_fields)
    with open(final_path, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    log.info("[phase=final_sentinel] %s", final_path)


def _run_concordance_package(slab_root: Path, figures_dir: Path) -> dict:
    """Run the concordance stats package twice (plan §4 step 9): PRIMARY
    X=emission_rate, SECONDARY X=marker_delta (same cells, log-prob DV).

    Requires the full 138-cell matrix (the script's pre-registered row-count
    assert); partial sweeps (smoke) skip with an explicit log + sentinel note.
    """
    matrix_path = slab_root / "marker_delta_matrix.json"
    if not matrix_path.exists():
        raise RuntimeError(f"concordance requested but matrix missing: {matrix_path}")
    with open(matrix_path) as f:
        n_rows = json.load(f)["n_rows"]
    if n_rows != 138:
        log.info(
            "[phase=concordance] SKIPPED: matrix has %d rows (the pre-registered "
            "package requires the full 138-cell panel; smoke runs skip).",
            n_rows,
        )
        return {"skipped": True, "n_rows": n_rows}

    parts = figures_dir.parts
    if "figures" in parts:
        idx = parts.index("figures")
        fig_rel = Path(*parts[idx + 1 :]) if len(parts) > idx + 1 else Path(".")
    else:
        fig_rel = Path(figures_dir.name)

    stats_paths: dict[str, str] = {}
    for x_field in ("emission_rate", "marker_delta"):
        suffix = "" if x_field == "emission_rate" else f"_{x_field}"
        stem = f"{fig_rel}/emission_rate_vs_sycophancy{suffix}"
        cmd = [
            "uv",
            "run",
            "python",
            "scripts/issue480_emission_rate_concordance.py",
            "--matrix-path",
            str(matrix_path),
            "--stats-dir",
            str(slab_root),
            "--figure-stem",
            stem,
            "--x-field",
            x_field,
        ]
        log.info("[phase=concordance] spawning: %s", " ".join(cmd))
        subprocess.run(cmd, env={**os.environ}, check=True)
        stats_name = (
            "concordance_stats.json"
            if x_field == "emission_rate"
            else f"concordance_stats_{x_field}.json"
        )
        out = slab_root / stats_name
        if not out.exists():
            raise RuntimeError(f"concordance ({x_field}) finished but {out} missing")
        stats_paths[x_field] = str(out)
    return {"skipped": False, "n_rows": n_rows, "stats_paths": stats_paths}


# HF location of the parent run's published adapter_config.json — the
# reference for the plan-assumption-12 rsLoRA / target-module parity check.
PARENT_ADAPTER_CONFIG_HF_PATH = "adapters/issue_480/villain_seed42/adapter_config.json"

# train_lora's _DEFAULT_LORA_TARGETS (sft.py) — the list cfg.lora_targets=None
# resolves to. Mirrored here (it is a function-local literal, not importable)
# for the parity check below; train_lora always sets use_rslora=True and never
# sets modules_to_save.
_TRAIN_LORA_DEFAULT_TARGETS = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


def _assert_band_stop_adapter_parity(max_length: int) -> dict:
    """Plan assumption 12 evidence (round-6, code-review v5 minor): the
    band_stop TrainLoraConfig must produce the SAME load-bearing PEFT adapter
    geometry as the parent run's published adapter — r, lora_alpha,
    lora_dropout, use_rslora, target_modules, modules_to_save. A silent
    geometry drift would smuggle a second variable into the
    one-variable-recipe-swap design AND invalidate the gauge-free logit
    readout assumptions.

    Downloads the parent's ``adapter_config.json`` from HF and diffs it
    against the geometry the REAL band_stop config (via
    ``_band_stop_train_cfg``) will hand to train_lora's LoraConfig. Runs at
    band_stop preflight, before any GPU work; fails LOUD on any mismatch.
    """
    from huggingface_hub import hf_hub_download

    cached = hf_hub_download(repo_id=HF_MODEL_REPO, filename=PARENT_ADAPTER_CONFIG_HF_PATH)
    with open(cached) as f:
        parent = json.load(f)

    cfg = _band_stop_train_cfg(
        source="_parity_probe",
        seed=DEFAULT_SEED,
        max_length=max_length,
        traj_path=Path("/tmp/_i480_parity_probe_trajectory.json"),
    )
    expected_targets = sorted(cfg.lora_targets or _TRAIN_LORA_DEFAULT_TARGETS)
    # (parent_value, band_stop_value) per load-bearing key.
    checks: dict[str, tuple[object, object]] = {
        "r": (parent.get("r"), cfg.lora_r),
        "lora_alpha": (parent.get("lora_alpha"), cfg.lora_alpha),
        "lora_dropout": (parent.get("lora_dropout"), cfg.lora_dropout),
        "use_rslora": (parent.get("use_rslora"), True),
        "target_modules": (sorted(parent.get("target_modules") or []), expected_targets),
        "modules_to_save": (parent.get("modules_to_save"), None),
    }
    mismatches = {k: v for k, v in checks.items() if v[0] != v[1]}
    for key, (parent_val, ours) in checks.items():
        log.info(
            "[phase=preflight] adapter-config parity %s: parent=%s band_stop=%s %s",
            key,
            parent_val,
            ours,
            "MISMATCH" if key in mismatches else "OK",
        )
    if mismatches:
        raise RuntimeError(
            "band_stop adapter-config parity FAILED vs parent "
            f"{HF_MODEL_REPO}/{PARENT_ADAPTER_CONFIG_HF_PATH}: {mismatches} — "
            "the recipe swap is no longer single-variable; refusing to launch."
        )
    log.info(
        "[phase=preflight] adapter-config parity vs parent PASSED (%d keys)",
        len(checks),
    )
    return {k: v[1] for k, v in checks.items()}


def _run_band_stop_pipeline(args, sources: list[str], max_length: int) -> int:
    """band_stop recipe driver: per-cell loop → Phase 3 → concordance ×2 → sentinel.

    Phase 0 is skipped BY DESIGN (the parent's 700-row pools are reused
    byte-for-byte from HF at a pinned revision; R_train_base is not needed
    because pools are downloaded, not rebuilt).
    """
    # Stale-branch port guard (plan §8): the parent experiment code was
    # ported from the issue-480 branch whose sft.py predates the band-stop
    # machinery — running against that stale module would silently degrade
    # the recipe to fixed-epoch training.
    from dataclasses import fields as _dc_fields

    from explore_persona_space.train.sft import TrainLoraConfig

    cfg_fields = {f.name for f in _dc_fields(TrainLoraConfig)}
    required_fields = {"marker_band_log_only", "marker_band_trajectory_path", "save_only_model"}
    missing_fields = required_fields - cfg_fields
    if missing_fields:
        raise RuntimeError(
            f"stale-branch port: TrainLoraConfig is missing {sorted(missing_fields)} — "
            "this checkout lacks main's band-stop machinery; the branch must be cut "
            "from current main, never run from the parent issue-480 branch."
        )

    # Plan assumption 12: rsLoRA + target-module parity vs the parent's
    # published HF adapter_config.json (fail-loud, before any GPU work).
    _assert_band_stop_adapter_parity(max_length)

    os.environ["WANDB_PROJECT"] = BAND_STOP_WANDB_PROJECT
    log.info(
        "[phase=preflight] band_stop recipe: WANDB_PROJECT=%s, lr=%s, epoch_cap=%d, "
        "ckpt_every=%d, probe_every=%d, firing_target=%.1f nat",
        BAND_STOP_WANDB_PROJECT,
        BAND_STOP_LR,
        BAND_STOP_EPOCH_CAP,
        BAND_STOP_CKPT_EVERY_STEPS,
        BAND_STOP_PROBE_EVERY_STEPS,
        FIRING_TARGET_LOGP_NATS,
    )

    # Eval pool is still needed (Phase 2a); q_train / R_train_base /
    # bystander discovery are NOT (pools reused, not rebuilt).
    _ensure_wrong_claim_pool(args.eval_pool, kind="eval_50")

    plan_deviations: list[str] = ["phase0_skipped_band_stop_pools_reused"]
    log.info("[phase=phase0] SKIPPED — band_stop reuses parent pools from HF (pinned revision)")
    phase0_summary = {
        "skipped": True,
        "reason": "band_stop: parent 700-row pools reused byte-for-byte",
        "train_pool_revision": TRAIN_POOL_REVISION,
    }

    per_cell: list[dict] = []
    for source in sources:
        try:
            cell = _run_one_cell_band_stop(
                source=source,
                seed=args.seed,
                eval_pool=args.eval_pool,
                slab_root=args.slab_root,
                runs_root=args.runs_root,
                max_length=max_length,
            )
            per_cell.append(cell)
            per_src_sent = Path(f"/workspace/logs/issue-480-bsr-{source}-results.json")
            per_src_sent.parent.mkdir(parents=True, exist_ok=True)
            per_src_sent.write_text(json.dumps(cell, indent=2))
        except Exception as e:
            fail_path = Path(f"/workspace/logs/issue-480-bsr-{source}-FAILED.json")
            fail_path.parent.mkdir(parents=True, exist_ok=True)
            with open(fail_path, "w") as f:
                json.dump(
                    {
                        "source": source,
                        "recipe": "band_stop",
                        "phase": "cell_failed",
                        "exception_type": type(e).__name__,
                        "exception_msg": str(e),
                        "timestamp_utc": datetime.now(UTC).isoformat(),
                    },
                    f,
                    indent=2,
                )
            log.exception("[%s] band_stop cell failed; wrote %s", source, fail_path)
            raise

    phase3_summary: dict | None = None
    concordance_summary: dict | None = None
    if args.skip_analyze:
        log.info("[phase=phase3] SKIPPED.")
        plan_deviations.append("phase3_analyze_skipped")
    else:
        phase3_summary = _phase3_analyze(
            seed=args.seed,
            slab_root=args.slab_root,
            predictor_comparison_path=args.predictor_comparison,
            syco_summary_path=args.syco_summary,
            figures_dir=args.figures_dir,
            trajectory_dir=args.slab_root / "trajectories",
        )
        concordance_summary = _run_concordance_package(args.slab_root, args.figures_dir)
        if concordance_summary.get("skipped"):
            plan_deviations.append("concordance_skipped_partial_matrix")
        phase3_summary["concordance"] = concordance_summary

    epoch = int(time.time())
    final_path = Path(f"/workspace/logs/issue-480-bsr-epm_results-{epoch}.json")
    _write_final_sentinel(
        sources_requested=sources,
        per_cell=per_cell,
        phase0_summary=phase0_summary,
        phase3_summary=phase3_summary,
        plan_deviations=plan_deviations,
        final_path=final_path,
        gpu_hours_budgeted=10.0,
        wandb_project=BAND_STOP_WANDB_PROJECT,
        hf_adapter_root=BAND_STOP_HF_ADAPTER_ROOT,
    )
    log.info("[phase=dispatch_done] band_stop: %d cells completed.", len(per_cell))
    print("[phase=done]")
    return 0


# ── graded_eval recipe (same-issue follow-up round 3, plan v4) ──────────────


def _load_recorded_graded_pick(source: str, seed: int) -> dict:
    """Read the committed round-2 ``anchor_pick.json`` → ``picks.graded`` record.

    The file lives in git on this branch (the pod's checkout) — NEVER read via
    ``task.py`` (pod-side shellout ban). Fails loud when the record is missing
    a required key or the recorded delta sits outside the graded [5, 12] nat
    band (the reuse-fitness premise of the whole round — plan §10/§11).
    """
    pick_path = ROUND2_SLAB_ROOT / "per_source" / source / f"seed_{seed}" / "anchor_pick.json"
    if not pick_path.exists():
        raise RuntimeError(
            f"[{source}] committed round-2 pick record missing: {pick_path} — the "
            "graded_eval recipe needs the recorded in-loop values for the #534 "
            "parity assert; is the pod checkout on the issue branch with round-2 "
            "results merged?"
        )
    with open(pick_path) as f:
        pick = json.load(f)
    graded = dict(pick["picks"]["graded"])
    for key in ("step", "logp_trained", "logp_base", "delta_nats"):
        if key not in graded:
            raise RuntimeError(f"[{source}] picks.graded missing key {key!r} in {pick_path}")
    if not (GRADED_BAND_LOW <= float(graded["delta_nats"]) <= GRADED_BAND_HIGH):
        raise RuntimeError(
            f"[{source}] recorded graded delta {graded['delta_nats']:.3f} nat is outside "
            f"the [{GRADED_BAND_LOW}, {GRADED_BAND_HIGH}] band — reuse-fitness premise "
            "violated; do not evaluate this checkpoint as an in-band anchor."
        )
    graded["pick_record_path"] = str(pick_path)
    return graded


def _fetch_graded_adapter(source: str, seed: int, dest_dir: Path, *, download: bool) -> dict:
    """List + per-file download of the pinned round-2 graded adapter dir.

    Deliberately list_repo_files + per-file ``hf_hub_download`` rather than
    ``snapshot_download(allow_patterns=...)``: on repos with >~8k files the
    allow_patterns path can silently return 0 files for prefixes in the
    truncated ``repo_info.siblings`` tail, and this model repo is huge.
    Fails loud when ``adapter_config.json`` / ``adapter_model.safetensors`` /
    ``tokenizer_config.json`` are missing from the pinned-revision listing,
    or when no REAL tokenizer asset is present — ``tokenizer_config.json``
    alone is not loadable; the pinned graded dirs ship both the fast asset
    (``tokenizer.json``) and the slow BPE pair (``vocab.json`` +
    ``merges.txt``), so require at least one of the two so a partial upload
    fails here at preflight rather than inside ``merge_lora`` on the pod
    (``merge_lora`` loads the tokenizer from the adapter path).

    Returns the provenance dict for ``graded_eval_record.json``. With
    ``download=False`` (CPU dry-run) only the listing + asserts run.
    """
    from huggingface_hub import hf_hub_download, list_repo_files

    prefix = f"{BAND_STOP_HF_ADAPTER_ROOT}/{source}_seed{seed}_graded/"
    all_files = list_repo_files(HF_MODEL_REPO, revision=GRADED_ADAPTER_REVISION)
    files = sorted(f for f in all_files if f.startswith(prefix))
    rel_names = {f[len(prefix) :] for f in files}
    for required in ("adapter_config.json", "adapter_model.safetensors", "tokenizer_config.json"):
        if required not in rel_names:
            raise RuntimeError(
                f"[{source}] {required} missing from {HF_MODEL_REPO}/{prefix} at pinned "
                f"revision {GRADED_ADAPTER_REVISION[:12]} (listed {len(files)} files) — "
                "wrong artifact resolved; refusing to evaluate."
            )
    has_fast_tokenizer = "tokenizer.json" in rel_names
    has_slow_tokenizer = "vocab.json" in rel_names and "merges.txt" in rel_names
    if not (has_fast_tokenizer or has_slow_tokenizer):
        raise RuntimeError(
            f"[{source}] no real tokenizer asset in {HF_MODEL_REPO}/{prefix} at pinned "
            f"revision {GRADED_ADAPTER_REVISION[:12]}: tokenizer_config.json alone is not "
            "loadable — need tokenizer.json OR vocab.json+merges.txt (listed: "
            f"{sorted(rel_names)}) — merge_lora's tokenizer load would fail on the pod; "
            "refusing to evaluate."
        )
    if download:
        dest_dir.mkdir(parents=True, exist_ok=True)
        for hub_file in files:
            rel = hub_file[len(prefix) :]
            cached = hf_hub_download(
                repo_id=HF_MODEL_REPO,
                filename=hub_file,
                revision=GRADED_ADAPTER_REVISION,
            )
            target = dest_dir / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(cached, target)
        log.info(
            "[phase=fetch_%s] graded adapter: %d files @ %s -> %s",
            source,
            len(files),
            GRADED_ADAPTER_REVISION[:12],
            dest_dir,
        )
    return {
        "hf_repo": HF_MODEL_REPO,
        "path_in_repo": prefix.rstrip("/"),
        # Commit-pinned request: hf_hub_download at a full commit SHA resolves
        # to exactly that commit, so requested == resolved by construction.
        "revision_requested": GRADED_ADAPTER_REVISION,
        "revision_resolved": GRADED_ADAPTER_REVISION,
        "files": sorted(rel_names),
        "n_files": len(files),
        "downloaded": download,
    }


def _run_one_cell_graded_eval(
    source: str,
    seed: int,
    eval_pool: Path,
    slab_root: Path,
    runs_root: Path,
    max_length: int,
    *,
    dry_run: bool = False,
) -> dict:
    """graded_eval cell: fetch pinned adapter → merge → Phase 2a → Phase 2b
    (four-float + gauge assert + #534 parity probe) → record → reap.

    Eval-only: NO training. Smoke IS this exact path with ``--only-source
    comedian`` (architectural parity: same dispatcher, same subprocess
    shapes, same teardown). ``dry_run=True`` (CPU, VM-safe) stops after the
    recorded-pick load + pinned-revision HF listing + pool fetch +
    parity-probe config + record scaffolding — before any GPU work.
    """
    t_start = time.time()
    output_dir = runs_root / f"{source}_seed{seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    eval_out_dir = slab_root / "per_source" / source / f"seed_{seed}"
    eval_out_dir.mkdir(parents=True, exist_ok=True)

    log.info("=" * 70)
    log.info(
        "[phase=cell_%s] GRADED-EVAL CELL START — output_dir=%s eval_out=%s dry_run=%s",
        source,
        output_dir,
        eval_out_dir,
        dry_run,
    )

    # Plan §4 steps 1-2: recorded pick (committed round-2 record) + pinned adapter.
    recorded = _load_recorded_graded_pick(source, seed)
    log.info(
        "[phase=cell_%s] recorded graded pick: step=%s logp_trained=%.3f logp_base=%.3f "
        "delta=%.3f nat",
        source,
        recorded["step"],
        recorded["logp_trained"],
        recorded["logp_base"],
        recorded["delta_nats"],
    )
    adapter_dir = output_dir / "graded_adapter"
    provenance = _fetch_graded_adapter(source, seed, adapter_dir, download=not dry_run)

    # Plan §4 step 3: train pool — needed ONLY for the parity-probe rows.
    train_jsonl = output_dir / "train_pool.jsonl"
    _ensure_train_pool(train_jsonl, source)

    # Parity-probe config for Phase 2b (#534 assert). Probe shape matches the
    # round-2 in-loop band callback exactly: max_rows=32, max_length =
    # max(train max_length, 2048) (the _maybe_attach_marker_band_stop
    # derivation with marker_band_probe_max_length=None).
    parity_json = output_dir / "parity_probe_config.json"
    with open(parity_json, "w") as f:
        json.dump(
            {
                "pool_path": str(train_jsonl),
                "recorded_logp_trained": recorded["logp_trained"],
                "recorded_logp_base": recorded["logp_base"],
                "tolerance_nats": PARITY_TOLERANCE_NATS,
                "max_rows": PARITY_PROBE_MAX_ROWS,
                "max_length": max(max_length, 2048),
            },
            f,
            indent=2,
        )

    base_record = {
        "source": source,
        "seed": seed,
        "recipe": "graded_eval",
        "adapter_provenance": provenance,
        "recorded_pick": recorded,
        "parity_tolerance_nats": PARITY_TOLERANCE_NATS,
        "git_commit_sha": _git_sha(),
        "hostname": socket.gethostname(),
        "timestamp_utc": datetime.now(UTC).isoformat(),
    }

    if dry_run:
        record = {
            **base_record,
            "dry_run": True,
            "planned_phases": ["merge", "phase2a", "phase2b_four_float_parity", "record"],
            "wall_seconds": round(time.time() - t_start, 1),
        }
        record_path = eval_out_dir / "graded_eval_record.dryrun.json"
        with open(record_path, "w") as f:
            json.dump(record, f, indent=2, ensure_ascii=False)
        log.info("[phase=cell_%s] DRY-RUN record -> %s", source, record_path)
        return {
            "source": source,
            "seed": seed,
            "recipe": "graded_eval",
            "dry_run": True,
            "wall_seconds": round(time.time() - t_start, 1),
            "output_dir": str(output_dir),
            "eval_out_dir": str(eval_out_dir),
            "record_path": str(record_path),
            "adapter_hf_path": provenance["path_in_repo"],
        }

    # Plan §4 step 4: merge the downloaded graded adapter into base.
    adapter_config = adapter_dir / "adapter_config.json"
    if not adapter_config.exists():
        raise RuntimeError(f"[{source}] {adapter_config} missing after download — fetch bug.")

    from explore_persona_space.train.sft import merge_lora

    merged_dir = output_dir / "merged"
    if merged_dir.exists():
        shutil.rmtree(merged_dir)
    log.info("[phase=merge_%s] %s -> %s", source, adapter_dir, merged_dir)
    merge_lora(BASE_MODEL, str(adapter_dir), str(merged_dir), gpu_id=0)

    # Plan §4 step 6: Phase 2a (vLLM greedy, 24×50, max_new_tokens 2048,
    # subprocess-isolated) — writes r_trained.json + raw_completions/ directly
    # into the CANONICAL per-source layout (no anchor_step subdir; step 8's
    # "promote" is by construction).
    r_trained_path = _phase2a(
        source,
        seed,
        merged_dir,
        eval_pool,
        eval_out_dir,
        sentinel_path=_logs_root() / f"issue-480-inband-{source}-phase2a.json",
    )

    # Plan §4 steps 5+7: Phase 2b (four-float storage contract; gauge assert
    # on the ACTUALLY-EVALUATED adapter config; #534 parity probe inside the
    # same subprocess — both models already loaded, the plan-named placement).
    logprob_path = _phase2b(
        source,
        seed,
        r_trained_path,
        merged_dir,
        eval_out_dir,
        sentinel_path=_logs_root() / f"issue-480-inband-{source}-phase2b.json",
        adapter_config_path=adapter_config,
        slot_stats="four-float",
        parity_probe_json=parity_json,
        # Round-3 parity-FAIL root-cause fix: score the trained side through
        # the UNMERGED adapter (the in-loop band-callback convention). The
        # bf16 merge truncates the tiny step-20 LoRA delta below the
        # base-weight ULP (~2.1 nat marker-push attenuation, comedian
        # diagnostic 2026-06-11); the recorded picks.graded values are
        # reproducible to 0.03 nat ONLY through the unmerged application.
        trained_adapter_dir=adapter_dir,
    )

    # The parity numbers land in the Phase 2b payload; a FAIL would have
    # aborted that subprocess (non-zero exit -> the per-cell except path).
    with open(logprob_path) as f:
        phase2b_payload = json.load(f)
    parity = phase2b_payload.get("parity_probe")
    if not parity or not parity.get("passed"):
        raise RuntimeError(
            f"[{source}] Phase 2b output carries no passed parity_probe block — the #534 "
            "adapter-application assert did not run; refusing to record this cell."
        )

    # Plan §4 step 8: graded_eval_record.json (reuse provenance + parity numbers).
    wall = time.time() - t_start
    record = {
        **base_record,
        "parity_probe": parity,
        "trained_model_application": phase2b_payload.get("trained_model_application"),
        "logprob_path": str(logprob_path),
        "r_trained_path": str(r_trained_path),
        "wall_seconds": round(wall, 1),
    }
    record_path = eval_out_dir / "graded_eval_record.json"
    with open(record_path, "w") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)
    log.info("[phase=cell_%s] graded_eval_record -> %s", source, record_path)

    # Plan §4 step 9: reap merged dir (MooseFS quota). No adapter uploads
    # (nothing trained), no WandB (nothing trained).
    log.info("[phase=cell_%s] rmtree(%s) for MooseFS quota", source, merged_dir)
    shutil.rmtree(merged_dir, ignore_errors=False)

    log.info("[phase=cell_%s] GRADED-EVAL CELL DONE wall=%.1fs", source, wall)
    return {
        "source": source,
        "seed": seed,
        "recipe": "graded_eval",
        "wall_seconds": round(wall, 1),
        "output_dir": str(output_dir),
        "eval_out_dir": str(eval_out_dir),
        "logprob_path": str(logprob_path),
        "r_trained_path": str(r_trained_path),
        "record_path": str(record_path),
        "adapter_hf_path": provenance["path_in_repo"],
        "parity_abs_diff_trained": parity["abs_diff_trained"],
        "parity_abs_diff_base": parity["abs_diff_base"],
    }


def _export_inband_artifacts(per_cell: list[dict]) -> dict:
    """Fail-loud pre-termination artifact export (plan §4 post-cells steps 1-3).

    Order is load-bearing: (1) raw completions → HF data repo, (2) git
    commit+push the per-source eval JSONs on the issue branch (FAIL-LOUD —
    a stranded eval dir is a hard failure), (3) per-source JSON mirror → HF
    data repo. Any failure raises BEFORE the final sentinel is written.

    Walks the COMPLETED ``per_cell`` list (never the full registered grid),
    so the one-cell smoke exercises this phase end-to-end.

    Note on upload shape: Phase 2a writes per-panel ``raw_completions/
    <panel>_seed<S>.json`` files, which ``upload_raw_completions_to_data_
    repo``'s ``rglob("raw_completions.json")`` does NOT match — hence the
    explicit per-cell ``hub._upload`` folder walk (one Hub commit per cell;
    12 commits total for 6 cells, far under the 256/hr cap).
    """
    from explore_persona_space.orchestrate.hub import _upload

    export: dict = {"raw_completions": {}, "git": {}, "per_source_mirror": {}}

    # 1. Raw completions → HF data repo (non-LFS JSON path).
    for cell in per_cell:
        source, seed = cell["source"], cell["seed"]
        raw_dir = Path(cell["eval_out_dir"]) / "raw_completions"
        raw_files = sorted(raw_dir.glob("*.json")) if raw_dir.exists() else []
        if not raw_files:
            raise RuntimeError(
                f"[{source}] no raw completion files under {raw_dir} — Phase 2a contract "
                "violated; refusing to terminate-without-upload."
            )
        path_in_repo = f"{INBAND_HF_DATA_SUBDIR}/raw_completions/{source}/seed_{seed}"
        url = _upload(
            local_path=raw_dir,
            repo_id=HF_DATA_REPO,
            repo_type="dataset",
            path_in_repo=path_in_repo,
        )
        if not url:
            raise RuntimeError(
                f"[{source}] raw-completions upload FAILED -> {HF_DATA_REPO}/{path_in_repo}; "
                "refusing to continue (pod termination would lose the only copy)."
            )
        export["raw_completions"][source] = {"url": url, "n_files": len(raw_files)}
        log.info("[phase=export] raw completions %s (%d files) -> %s", source, len(raw_files), url)

    # 2. Git commit + push per-source eval JSONs on the issue branch (FAIL-LOUD).
    env = {**os.environ}
    rel_files: list[str] = []
    for cell in per_cell:
        d = Path(cell["eval_out_dir"])
        for fname in ("marker_logprob_eval.json", "r_trained.json", "graded_eval_record.json"):
            p = d / fname
            if not p.exists():
                raise RuntimeError(f"[{cell['source']}] expected eval artifact missing: {p}")
            rel_files.append(str(p))
    branch = subprocess.check_output(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"], text=True, env=env
    ).strip()
    if branch != INBAND_GIT_BRANCH:
        raise RuntimeError(
            f"pod checkout is on branch {branch!r}, expected {INBAND_GIT_BRANCH!r} — "
            "refusing to push eval results from the wrong branch."
        )
    subprocess.run(["git", "add", "--", *rel_files], env=env, check=True)
    staged = subprocess.run(["git", "diff", "--cached", "--quiet"], env=env)
    if staged.returncode != 0:
        subprocess.run(
            [
                "git",
                "-c",
                "user.name=pod-480-inband",
                "-c",
                "user.email=pod-480@runpod.local",
                "commit",
                "-m",
                f"task #480: inband-logprob-concordance per-source eval JSONs "
                f"({len(per_cell)} cells)",
            ],
            env=env,
            check=True,
        )
    else:
        log.info("[phase=export] git: nothing newly staged (idempotent re-run)")
    push = subprocess.run(["git", "push", "origin", INBAND_GIT_BRANCH], env=env)
    if push.returncode != 0:
        # One rebase retry, then fail loud (aborts the final sentinel).
        subprocess.run(
            ["git", "pull", "--rebase", "origin", INBAND_GIT_BRANCH], env=env, check=True
        )
        subprocess.run(["git", "push", "origin", INBAND_GIT_BRANCH], env=env, check=True)
    export["git"] = {
        "branch": INBAND_GIT_BRANCH,
        "n_files": len(rel_files),
        "head": _git_sha(),
    }
    log.info("[phase=export] git push OK: %d files on %s", len(rel_files), INBAND_GIT_BRANCH)

    # 3. Per-source JSON mirror → HF data repo (belt-and-suspenders, non-LFS;
    # raw_completions/ excluded — already uploaded in step 1).
    for cell in per_cell:
        source, seed = cell["source"], cell["seed"]
        path_in_repo = f"{INBAND_HF_DATA_SUBDIR}/per_source/{source}/seed_{seed}"
        url = _upload(
            local_path=Path(cell["eval_out_dir"]),
            repo_id=HF_DATA_REPO,
            repo_type="dataset",
            path_in_repo=path_in_repo,
            ignore_patterns=["raw_completions/*"],
        )
        if not url:
            raise RuntimeError(
                f"[{source}] per-source mirror upload FAILED -> {HF_DATA_REPO}/{path_in_repo}"
            )
        export["per_source_mirror"][source] = url
        log.info("[phase=export] per-source mirror %s -> %s", source, url)
    return export


def _run_graded_eval_pipeline(args, sources: list[str], max_length: int) -> int:
    """graded_eval recipe driver: per-cell eval loop → pre-termination export → sentinel.

    Phase 0 is skipped BY DESIGN (eval-only; pools fetched per cell solely for
    the parity-probe rows). Phase 3 analysis + concordance run OFF-POD on the
    VM after termination (CPU-only phase rule) — plan_deviations names it.
    """
    # Stale-branch port guard (plan §4: keep the TrainLoraConfig field assert
    # as a cheap guard even though no training runs this round).
    from dataclasses import fields as _dc_fields

    from explore_persona_space.train.sft import TrainLoraConfig

    cfg_fields = {f.name for f in _dc_fields(TrainLoraConfig)}
    required_fields = {"marker_band_log_only", "marker_band_trajectory_path", "save_only_model"}
    missing_fields = required_fields - cfg_fields
    if missing_fields:
        raise RuntimeError(
            f"stale-branch port: TrainLoraConfig is missing {sorted(missing_fields)} — "
            "this checkout lacks main's band-stop machinery; the branch must be cut "
            "from current main."
        )

    log.info(
        "[phase=preflight] graded_eval recipe: adapter revision=%s parity_tol=%s nat "
        "probe_rows=%d round2_slab=%s dry_run=%s",
        GRADED_ADAPTER_REVISION[:12],
        PARITY_TOLERANCE_NATS,
        PARITY_PROBE_MAX_ROWS,
        ROUND2_SLAB_ROOT,
        args.graded_dry_run,
    )

    # Eval pool is needed for Phase 2a; q_train / R_train_base are NOT.
    _ensure_wrong_claim_pool(args.eval_pool, kind="eval_50")

    plan_deviations = ["phase0_skipped_eval_only", "phase3_concordance_moved_off_pod"]
    phase0_summary = {
        "skipped": True,
        "reason": "graded_eval: eval-only over round-2's uploaded in-band checkpoints",
        "graded_adapter_revision": GRADED_ADAPTER_REVISION,
    }

    per_cell: list[dict] = []
    for source in sources:
        try:
            cell = _run_one_cell_graded_eval(
                source=source,
                seed=args.seed,
                eval_pool=args.eval_pool,
                slab_root=args.slab_root,
                runs_root=args.runs_root,
                max_length=max_length,
                dry_run=args.graded_dry_run,
            )
            per_cell.append(cell)
            per_src_sent = _logs_root() / f"issue-480-inband-{source}-results.json"
            per_src_sent.parent.mkdir(parents=True, exist_ok=True)
            per_src_sent.write_text(json.dumps(cell, indent=2))
        except Exception as e:
            fail_path = _logs_root() / f"issue-480-inband-{source}-FAILED.json"
            fail_path.parent.mkdir(parents=True, exist_ok=True)
            with open(fail_path, "w") as f:
                json.dump(
                    {
                        "source": source,
                        "recipe": "graded_eval",
                        "phase": "cell_failed",
                        "exception_type": type(e).__name__,
                        "exception_msg": str(e),
                        "timestamp_utc": datetime.now(UTC).isoformat(),
                    },
                    f,
                    indent=2,
                )
            log.exception("[%s] graded_eval cell failed; wrote %s", source, fail_path)
            raise

    export_summary: dict | None = None
    if args.graded_dry_run:
        log.info("[phase=export] SKIPPED (graded dry run — no artifacts produced).")
        plan_deviations.append("dry_run_no_artifact_export")
    else:
        export_summary = _export_inband_artifacts(per_cell)

    epoch = int(time.time())
    final_path = _logs_root() / f"issue-480-inband-epm_results-{epoch}.json"
    _write_final_sentinel(
        sources_requested=sources,
        per_cell=per_cell,
        phase0_summary=phase0_summary,
        phase3_summary=None,
        plan_deviations=plan_deviations,
        final_path=final_path,
        gpu_hours_budgeted=4.0,
        wandb_project="none (graded_eval is eval-only; no training, no WandB runs)",
        hf_adapter_root=BAND_STOP_HF_ADAPTER_ROOT,
        extra_note_fields={"artifact_export": export_summary},
    )
    log.info("[phase=dispatch_done] graded_eval: %d cells completed.", len(per_cell))
    print("[phase=done]")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=_parse_sources, default=list(DEFAULT_SOURCES))
    parser.add_argument(
        "--only-source",
        type=str,
        default=None,
        help="Single source — smoke shortcut. OVERRIDES --sources.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Shorthand for --only-source villain (smoke = sweep w/ one source).",
    )
    parser.add_argument(
        "--recipe",
        choices=("parent", "band_stop", "graded_eval"),
        default="parent",
        help="Training-stop recipe. 'parent' (default) = the run-1 fixed-3-epoch "
        "pipeline, byte-identical. 'band_stop' = the band-stopped-anchor-rerun: "
        "reused pools, lr 5e-6 to a 12-epoch cap with 20-step checkpoints, "
        "log-only band callback, deterministic onset-edge anchor pick gated on "
        "bystander resolution (plan v3 §4). 'graded_eval' = the round-3 "
        "inband-logprob-concordance EVAL-ONLY path: fetch round 2's pinned "
        "in-band (step-20) graded adapters, merge, Phase 2a/2b with the "
        "four-float capture + #534 parity probe, then the fail-loud "
        "pre-termination artifact export (plan v4 §4). No training.",
    )
    parser.add_argument(
        "--graded-dry-run",
        action="store_true",
        help="CPU-only dry run of the graded_eval path (VM-safe with "
        "EPM_I480_LOGS_ROOT): recorded-pick load, pinned-revision HF listing "
        "asserts (no weight download), pool fetch, parity-probe config + "
        "record scaffolding; stops before merge/Phase 2a/Phase 2b and skips "
        "the artifact export. Only valid with --recipe graded_eval.",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--q-train",
        type=Path,
        default=Path("data/issue_480/wrong_claims/train_200.jsonl"),
    )
    parser.add_argument(
        "--eval-pool",
        type=Path,
        default=Path("data/issue_480/wrong_claims/eval_50.jsonl"),
    )
    parser.add_argument("--r-base-dir", type=Path, default=Path("data/issue_480/R_train_base"))
    parser.add_argument("--slab-root", type=Path, default=Path("eval_results/issue_480"))
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=Path("/workspace/runs/issue_480"),
    )
    parser.add_argument(
        "--predictor-comparison",
        type=Path,
        default=Path("eval_results/issue_480/_inputs/predictor_comparison.json"),
    )
    parser.add_argument(
        "--syco-summary",
        type=Path,
        default=Path("eval_results/issue_480/_inputs/syco_411_analyze_summary.json"),
    )
    parser.add_argument("--figures-dir", type=Path, default=Path("figures/issue_480"))
    parser.add_argument(
        "--skip-phase0",
        action="store_true",
        help="Skip Phase 0 (use pre-existing R_train_base/*.json).",
    )
    parser.add_argument("--skip-analyze", action="store_true", help="Skip Phase 3.")
    # round-3 fix: max_length is plumbed end-to-end (pool-build guard +
    # training config) from a single CLI knob so the build-time CPU
    # assertion sees the same budget as TRL at training time. Default
    # matches DEFAULT_TRAIN_MAX_LENGTH (2560), sized for a worst-case
    # ~2110-token Qwen-2.5 row + ~21% headroom; see
    # build_training_pool.DEFAULT_TRAIN_MAX_LENGTH docstring for the math.
    parser.add_argument(
        "--max-length",
        type=int,
        default=None,
        help="Override TRL SFTConfig.max_length / pool-build guard "
        "(defaults to build_training_pool.DEFAULT_TRAIN_MAX_LENGTH).",
    )
    args = parser.parse_args(argv)
    if args.graded_dry_run and args.recipe != "graded_eval":
        parser.error("--graded-dry-run is only valid with --recipe graded_eval")

    logging.basicConfig(
        level=os.environ.get("EPS_LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s | %(message)s",
        stream=sys.stdout,
    )

    if args.smoke:
        sources = ["villain"]
    elif args.only_source:
        sources = [args.only_source]
    else:
        sources = args.sources

    log.info(
        "[phase=dispatch_start] recipe=%s sources=%s seed=%d q_train=%s eval_pool=%s "
        "slab=%s runs=%s",
        args.recipe,
        sources,
        args.seed,
        args.q_train,
        args.eval_pool,
        args.slab_root,
        args.runs_root,
    )
    cell_fn_by_recipe = {
        "parent": "_run_one_cell",
        "band_stop": "_run_one_cell_band_stop",
        "graded_eval": "_run_one_cell_graded_eval",
    }
    log.info(
        "[phase=dispatch_start] UNIFIED smoke=sweep-with-one-source: same "
        "%s function path; same env injection; same teardown.",
        cell_fn_by_recipe[args.recipe],
    )

    # Pre-flight asserts (tokenizer marker id, im_end id).
    from transformers import AutoTokenizer

    from explore_persona_space.experiments.marker_implant_480 import (
        IM_END_ID,
        MARKER_ID,
        MARKER_TEXT,
        SOURCE_PERSONAS,
    )
    from explore_persona_space.experiments.marker_implant_480.build_training_pool import (
        DEFAULT_TRAIN_MAX_LENGTH,
        SOURCE_SYSTEM_PROMPTS,
        discover_bystander_pairs,
    )

    max_length = args.max_length if args.max_length is not None else DEFAULT_TRAIN_MAX_LENGTH
    log.info(
        "[phase=preflight] training max_length = %d (default=%d, cli=%s)",
        max_length,
        DEFAULT_TRAIN_MAX_LENGTH,
        args.max_length,
    )

    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tok.encode(MARKER_TEXT, add_special_tokens=False) != [MARKER_ID]:
        raise RuntimeError(
            f"marker {MARKER_TEXT!r} -> {tok.encode(MARKER_TEXT, add_special_tokens=False)}, "
            f"expected [{MARKER_ID}]"
        )
    if tok.encode("<|im_end|>", add_special_tokens=False) != [IM_END_ID]:
        raise RuntimeError(
            f"im_end -> {tok.encode('<|im_end|>', add_special_tokens=False)}, "
            f"expected [{IM_END_ID}]"
        )
    log.info("[phase=preflight] marker/im_end token ids OK")

    for s in sources:
        if s not in SOURCE_PERSONAS:
            raise ValueError(f"source {s} not in SOURCE_PERSONAS {SOURCE_PERSONAS}")

    args.slab_root.mkdir(parents=True, exist_ok=True)
    args.runs_root.mkdir(parents=True, exist_ok=True)
    args.r_base_dir.mkdir(parents=True, exist_ok=True)
    # _logs_root() == /workspace/logs unless EPM_I480_LOGS_ROOT is set (the
    # graded_eval CPU dry-run on the VM, where /workspace does not exist).
    _logs_root().mkdir(parents=True, exist_ok=True)

    if args.recipe == "band_stop":
        return _run_band_stop_pipeline(args, sources, max_length)

    if args.recipe == "graded_eval":
        return _run_graded_eval_pipeline(args, sources, max_length)

    # Auto-download wrong-claim Q pools from the #411 HF data subdir if missing.
    # Smoke runs on fresh pods used to FileNotFoundError here because the default
    # paths under data/issue_480/wrong_claims/ are not in git (they belong to #411).
    _ensure_wrong_claim_pool(args.q_train, kind="train_200")
    _ensure_wrong_claim_pool(args.eval_pool, kind="eval_50")

    # Discover bystander assignment (deterministic from #411 HF pools).
    bystander_cache = Path("data/issue_480/bystander_assignment.json")
    bystander_assignment = discover_bystander_pairs(cache_path=bystander_cache)
    log.info("[phase=preflight] bystander assignment cached at %s", bystander_cache)

    persona_specs = _collect_persona_specs(bystander_assignment, SOURCE_SYSTEM_PROMPTS)
    log.info(
        "[phase=preflight] %d distinct personas for Phase 0 (sources + bystanders + no-persona)",
        len(persona_specs),
    )

    plan_deviations: list[str] = []

    # ----- Phase 0 -----
    if args.skip_phase0:
        log.info("[phase=phase0] SKIPPED — using pre-existing R at %s", args.r_base_dir)
        phase0_summary = {"skipped": True, "out_dir": str(args.r_base_dir)}
        plan_deviations.append("phase0_skipped")
    else:
        phase0_summary = _phase0(persona_specs, args.q_train, args.r_base_dir, args.seed)

    # Load R into memory for per-source pool building.
    q_train = _load_q_train(args.q_train)
    r_base_by_key = _load_r_base_files(args.r_base_dir, persona_specs)

    # ----- Per-cell loop -----
    per_cell: list[dict] = []
    for source in sources:
        try:
            cell = _run_one_cell(
                source=source,
                seed=args.seed,
                q_train=q_train,
                r_base_by_key=r_base_by_key,
                bystander_assignment=bystander_assignment,
                eval_pool=args.eval_pool,
                slab_root=args.slab_root,
                runs_root=args.runs_root,
                max_length=max_length,
            )
            per_cell.append(cell)
            # Per-source sentinel (poll_pipeline visibility).
            per_src_sent = Path(f"/workspace/logs/issue-480-{source}-results.json")
            per_src_sent.write_text(json.dumps(cell, indent=2))
        except Exception as e:
            fail_path = Path(f"/workspace/logs/issue-480-{source}-FAILED.json")
            fail_path.parent.mkdir(parents=True, exist_ok=True)
            with open(fail_path, "w") as f:
                json.dump(
                    {
                        "source": source,
                        "phase": "cell_failed",
                        "exception_type": type(e).__name__,
                        "exception_msg": str(e),
                        "timestamp_utc": datetime.now(UTC).isoformat(),
                    },
                    f,
                    indent=2,
                )
            log.exception("[%s] cell failed; wrote %s", source, fail_path)
            raise

    # ----- Phase 3 analysis -----
    phase3_summary: dict | None = None
    if args.skip_analyze:
        log.info("[phase=phase3] SKIPPED.")
        plan_deviations.append("phase3_analyze_skipped")
    else:
        phase3_summary = _phase3_analyze(
            seed=args.seed,
            slab_root=args.slab_root,
            predictor_comparison_path=args.predictor_comparison,
            syco_summary_path=args.syco_summary,
            figures_dir=args.figures_dir,
        )

    # End-of-sweep sentinel (poll_pipeline-compatible).
    epoch = int(time.time())
    final_path = Path(f"/workspace/logs/issue-480-epm_results-{epoch}.json")
    _write_final_sentinel(
        sources_requested=sources,
        per_cell=per_cell,
        phase0_summary=phase0_summary,
        phase3_summary=phase3_summary,
        plan_deviations=plan_deviations,
        final_path=final_path,
    )
    log.info("[phase=dispatch_done] %d cells completed.", len(per_cell))
    print("[phase=done]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
