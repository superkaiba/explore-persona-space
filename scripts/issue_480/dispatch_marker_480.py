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
    from explore_persona_space.experiments.marker_implant_480 import IM_END_ID, MARKER_TEXT
    from explore_persona_space.train.sft import TrainLoraConfig, merge_lora, train_lora

    adapter_dir = output_dir / "adapter"
    merged_dir = output_dir / "merged"
    adapter_dir.mkdir(parents=True, exist_ok=True)

    # Plan §10 Reproducibility Card + plan §11 Decision Rationale:
    # lr=1e-5 (matches BOTH #411 AND #460/#474 marker rig);
    # lora_dropout=0.0 (marker-rig convention, NOT 0.05 from #411 — see plan §11 row);
    # marker_only_loss=True + tail_tokens=0 + #474 suppress_at_post_response_slot=True.
    # max_length: round-3 fix — pulled from build-time guard's
    # DEFAULT_TRAIN_MAX_LENGTH so pool-build and training see the same budget.
    # Source: .claude/rules/marker-leakage-measurement.md (R-cap ~1024, eval-cap
    # >=2048) + #260 (training-truncation -> silent zeros on the DV).
    cfg = TrainLoraConfig(
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
        hf_upload=True,
        hf_repo=HF_MODEL_REPO,
        hf_path_in_repo=f"adapters/issue_480/{source}_seed{seed}",
    )
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
) -> Path:
    """Phase 2a — vLLM gen R_trained in fresh subprocess."""
    sentinel = Path(f"/workspace/logs/issue-480-{source}-phase2a-results.json")
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
) -> Path:
    """Phase 2b — HF logprob in fresh subprocess (vLLM workers reaped)."""
    sentinel = Path(f"/workspace/logs/issue-480-{source}-phase2b-results.json")
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


def _phase3_analyze(
    *,
    seed: int,
    slab_root: Path,
    predictor_comparison_path: Path,
    syco_summary_path: Path,
    figures_dir: Path,
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
) -> None:
    """Write end-of-run sentinel in poll_pipeline-compatible schema."""
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
            "gpu_hours_budgeted": 6.5,
            "final_commit_sha": _git_sha(),
            "hostname": socket.gethostname(),
            "wandb_url": "n/a (per-cell wandb runs; project=issue480-marker-payload-swap)",
            "hf_hub_url": f"https://huggingface.co/{HF_MODEL_REPO}/tree/main/adapters/issue_480",
        },
    }
    with open(final_path, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    log.info("[phase=final_sentinel] %s", final_path)


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
        "[phase=dispatch_start] sources=%s seed=%d q_train=%s eval_pool=%s slab=%s runs=%s",
        sources,
        args.seed,
        args.q_train,
        args.eval_pool,
        args.slab_root,
        args.runs_root,
    )
    log.info(
        "[phase=dispatch_start] UNIFIED smoke=sweep-with-one-source: same "
        "_run_one_cell function path; same env injection; same teardown."
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
    Path("/workspace/logs").mkdir(parents=True, exist_ok=True)

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
