#!/usr/bin/env python3
# ruff: noqa: RUF001, RUF002, RUF003
"""Issue #296: Re-evaluate the 24 inherited #274 LoRAs against the N=48 eval matrix.

The N=24 eval matrix from #274 is no longer apples-to-apples now that #296 expands the
ALL_EVAL_PERSONAS_PLUS dict to N=48; re-fitting "the N=24 fit at L15" on
(OLD source rates × NEW N=48-centered cosines) would mix two pipelines. The fix is a
re-eval (no retraining): fetch the existing 24 LoRA adapters from WandB Artifacts, merge
each into a full Qwen2.5-7B-Instruct, run only the eval phase (--eval-only) against
ALL_EVAL_PERSONAS_PLUS (now N=48), then clean up the merged dir to free disk.

Adapter source (per plan §3d / §6 / fact-checker F15):
  Canonical: WandB Artifacts at
    `thomasjiralerspong/huggingface/marker_<src>_asst_excluded_medium_seed42:latest`
  Fallback:  HF Hub `superkaiba1/explore-persona-space` (only if WandB miss)

Pre-condition: this script can pull adapters itself (--pull) OR you can pre-stage them:
    ssh epm-issue-296 'cd /workspace/explore-persona-space && \\
        python scripts/pod.py sync models --pull'
Pre-stage avoids re-fetching across resumed-pod sessions.

Disk discipline: the script merges each adapter to a full model on demand, runs eval, then
deletes the merged dir + the adapter download dir, freeing ~14 GB per source. Peak per-cycle
disk = (1 merged model in eval) + (1 adapter dir downloaded). Without cleanup, 24 merged
models would need ~340 GB.

Usage (on the pod):
    nohup uv run python scripts/launch_issue296_reeval.py \\
        --pod epm-issue-296 --n-gpus 8 --pull \\
        > eval_results/leakage_experiment/i296_reeval_launcher.log 2>&1 &
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = ROOT / "eval_results" / "leakage_experiment"
LOG_DIR.mkdir(parents=True, exist_ok=True)
CKPT_BASE_DIR = ROOT / "checkpoints"

# 24 inherited #274 sources: 10 named PERSONAS + helpful_assistant + qwen_default + 12 #274 names.
# Mirrors INHERITED_PERSONAS_24 in scripts/analyze_issue296.py.
SOURCES = (
    # 10 named PERSONAS (#246 + #232 lineage)
    "software_engineer",
    "kindergarten_teacher",
    "data_scientist",
    "medical_doctor",
    "librarian",
    "french_person",
    "villain",
    "comedian",
    "police_officer",
    "zelthari_scholar",
    # generic_helper (#246)
    "helpful_assistant",
    "qwen_default",
    # 12 #274 NEW_PERSONA_PROMPTS_274
    "chef",
    "lawyer",
    "accountant",
    "journalist",
    "wizard",
    "hero",
    "philosopher",
    "child",
    "ai_assistant",
    "ai",
    "chatbot",
    "i_am_helpful",
)

assert len(SOURCES) == 24, f"Expected 24 inherited sources, got {len(SOURCES)}"

WANDB_ARTIFACT_TEMPLATE = (
    "thomasjiralerspong/huggingface/marker_{source}_asst_excluded_medium_seed42:latest"
)


BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"


def _adapter_dir_for(source: str) -> Path:
    """Local download path for the WandB-pulled adapter for a given source."""
    return CKPT_BASE_DIR / f"marker_{source}_asst_excluded_medium_seed42_adapter"


def _merged_dir_for(source: str) -> Path:
    """Path where ``--eval-only`` expects a merged model for a given source.

    Mirrors ``merged_path = output_dir / "merged"`` in
    ``scripts/archive/run_leakage_experiment.py`` (~line 835): when
    ``--eval-only`` is passed, the script raises FileNotFoundError if this
    exact directory is missing.
    """
    return (
        ROOT
        / "eval_results"
        / "leakage_experiment"
        / f"marker_{source}_asst_excluded_medium_seed42"
        / "merged"
    )


def _merge_pulled_adapter(source: str, *, gpu_id: int = 0) -> bool:
    """Materialize a merged Qwen2.5-7B-Instruct + LoRA-adapter for ``source``.

    The WandB Artifact for ``marker_<src>_asst_excluded_medium_seed42:latest``
    contains only the LoRA adapter (``adapter_model.safetensors`` + config +
    tokenizer files), NOT a merged model. ``run_leakage_experiment.py
    --eval-only`` requires a merged model on disk at
    ``eval_results/leakage_experiment/marker_<src>_asst_excluded_medium_seed42/merged/``.

    This function bridges that gap: locate the pulled adapter dir, call
    ``merge_lora()`` from ``src/explore_persona_space/train/sft.py`` to
    materialize the merged model where ``--eval-only`` expects it.

    Returns True on success, False on failure (caller logs and proceeds).

    Disk: peaks at ~15 GB per source (merged) + ~0.3 GB (adapter), but
    ``cleanup_after_eval`` deletes both after the per-source eval finishes,
    so the wave-level peak stays bounded.
    """
    adapter_dir = _adapter_dir_for(source)
    merged_dir = _merged_dir_for(source)

    if merged_dir.exists() and any(merged_dir.iterdir()):
        print(f"  [MERGE-CACHED] {source}: {merged_dir} already populated", flush=True)
        return True

    # snapshot_download from HF Hub may have nested the adapter under a sub-dir.
    # Locate the actual adapter root (one containing adapter_config.json).
    candidates = [adapter_dir]
    if adapter_dir.exists():
        candidates.extend(p for p in adapter_dir.rglob("adapter_config.json"))
    actual_adapter_root: Path | None = None
    for cand in candidates:
        cand_root = cand.parent if cand.is_file() else cand
        if (cand_root / "adapter_config.json").exists():
            actual_adapter_root = cand_root
            break
    if actual_adapter_root is None:
        print(
            f"  [MERGE-FAIL] {source}: no adapter_config.json found under {adapter_dir}",
            flush=True,
        )
        return False

    # Defer the heavy import + GPU allocation until we actually need it (a
    # bare `--no-pull` invocation must not pay the import cost).
    sys.path.insert(0, str(ROOT / "src"))
    try:
        from explore_persona_space.train.sft import merge_lora
    except ImportError as exc:
        print(f"  [MERGE-FAIL] {source}: import merge_lora failed: {exc}", flush=True)
        return False

    merged_dir.parent.mkdir(parents=True, exist_ok=True)
    print(
        f"  Merging {actual_adapter_root} -> {merged_dir} (gpu_id={gpu_id})",
        flush=True,
    )
    try:
        merge_lora(BASE_MODEL, str(actual_adapter_root), str(merged_dir), gpu_id=gpu_id)
    except Exception as exc:  # merge can OOM, fail to download base, etc.
        print(f"  [MERGE-FAIL] {source}: merge_lora raised: {exc}", flush=True)
        return False
    print(f"  [MERGE-OK] {source}: merged model at {merged_dir}", flush=True)
    return True


def _is_already_reevald(source: str) -> bool:
    """Skip if a populated 48-cell run_result.json already exists for this source."""
    rr_path = (
        ROOT
        / "eval_results"
        / "leakage_experiment"
        / f"marker_{source}_asst_excluded_medium_seed42"
        / "run_result.json"
    )
    if not rr_path.exists():
        return False
    try:
        with open(rr_path) as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return False
    marker = data.get("results", {}).get("marker", {})
    sr = marker.get("source_rate")
    all_p = marker.get("all_personas", {}) or {}
    # Re-eval'd run carries N=48 personas in all_personas. Use that as the discriminator
    # against an old N=24 result that should NOT be considered done.
    return bool(len(all_p) >= 48 and sr is not None)


def _pull_inherited_from_wandb(source: str, *, target_dir: Path) -> bool:
    """Download the WandB artifact for a given inherited source.

    Returns True on success, False if the artifact is missing (caller should fall back).
    Cleans up checkpoint subdirs after the pull (~5 GB savings per artifact).
    """
    import wandb

    art_name = WANDB_ARTIFACT_TEMPLATE.format(source=source)
    api = wandb.Api()
    try:
        art = api.artifact(art_name)
    except Exception as exc:
        print(f"  [WANDB MISS] {art_name}: {exc}", flush=True)
        return False

    target_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Downloading {art_name} -> {target_dir}", flush=True)
    art.download(root=str(target_dir))

    # Cleanup: WandB artifacts often contain checkpoint-* subdirs that are not needed
    # for inference; the merged model + adapter weights are what matter. Delete the
    # checkpoint-* siblings to save ~5 GB.
    removed = 0
    for sub in target_dir.iterdir():
        if sub.is_dir() and sub.name.startswith("checkpoint-"):
            shutil.rmtree(sub, ignore_errors=True)
            removed += 1
    if removed:
        print(f"  Cleaned {removed} checkpoint-* subdirs in {target_dir}", flush=True)
    return True


def _pull_inherited_from_hf(source: str, *, target_dir: Path) -> bool:
    """Fallback: pull the LoRA adapter from HF Hub `superkaiba1/explore-persona-space`."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("  [HF FALLBACK FAIL] huggingface_hub not available", flush=True)
        return False

    repo_id = "superkaiba1/explore-persona-space"
    subpath = f"marker_{source}_asst_excluded_medium_seed42"
    target_dir.mkdir(parents=True, exist_ok=True)
    try:
        snapshot_download(
            repo_id=repo_id,
            allow_patterns=[f"{subpath}/*"],
            local_dir=str(target_dir.parent),
            local_dir_use_symlinks=False,
        )
    except Exception as exc:
        print(f"  [HF MISS] {repo_id}/{subpath}: {exc}", flush=True)
        return False

    # If snapshot_download created target_dir.parent / subpath, leave it; the caller
    # will resolve via a glob. For uniformity, return True iff something landed.
    return any(target_dir.parent.glob(f"{subpath}*"))


def pull_all_inherited(sources: list[str], *, force: bool = False) -> list[str]:
    """Pull all inherited LoRA adapters from WandB (with HF Hub fallback).

    Returns the list of source names that succeeded. Sources whose adapter dir already
    exists locally are skipped unless `force=True`.
    """
    succeeded = []
    CKPT_BASE_DIR.mkdir(parents=True, exist_ok=True)
    for src in sources:
        target = _adapter_dir_for(src)
        if target.exists() and any(target.iterdir()) and not force:
            print(f"  [CACHED] {src}: {target}", flush=True)
            succeeded.append(src)
            continue
        ok = _pull_inherited_from_wandb(src, target_dir=target)
        if not ok:
            print(f"  Falling back to HF Hub for {src}", flush=True)
            ok = _pull_inherited_from_hf(src, target_dir=target)
        if ok:
            succeeded.append(src)
        else:
            print(f"  [FAIL] {src}: no adapter source available", flush=True)
    return succeeded


def build_cmd(source: str, gpu: int, pod: str) -> str:
    """Build the per-condition --eval-only command (as a bash string).

    --eval-only at scripts/archive/run_leakage_experiment.py:1035 skips training+merge
    and loads the existing merged model from the local output dir. The N=48
    ALL_EVAL_PERSONAS_PLUS matrix (extended for #296) kicks in automatically because
    the source is in SOURCES_REQUIRING_PLUS_EVAL (or via EPM_FORCE_EVAL_PERSONAS_PLUS=1
    for the legacy named PERSONAS that aren't in that set).

    The flag is set defensively for ALL inherited sources so that the named PERSONAS
    (software_engineer, ..., zelthari_scholar) plus helpful_assistant — which are NOT
    in SOURCES_REQUIRING_PLUS_EVAL — also see the N=48 eval matrix.
    """
    return (
        f"CUDA_VISIBLE_DEVICES={gpu} PYTHONUNBUFFERED=1 PYTHONHASHSEED=42 "
        f"EPM_FORCE_EVAL_PERSONAS_PLUS=1 "
        f".venv/bin/python scripts/archive/run_leakage_experiment.py "
        f"--trait marker --source {source} --neg-set asst_excluded "
        f"--prompt-length medium --seed 42 --gpu {gpu} "
        f"--pod {pod} --phase a1 --eval-only"
    )


def launch_wave(wave_idx: int, conditions: list[tuple[str, int]], pod: str) -> list:
    """Launch one wave (each (source, gpu) tuple in parallel) and return Popen handles."""
    procs = []
    print(f"\n=== Wave {wave_idx + 1}: launching {len(conditions)} re-evals ===", flush=True)
    for source, gpu in conditions:
        log_file = LOG_DIR / f"i296_reeval_{source}_asst_excluded_seed42_gpu{gpu}.log"
        cmd = build_cmd(source, gpu, pod)
        print(f"[gpu{gpu}] source={source}")
        print(f"[gpu{gpu}] cmd: {cmd}")
        print(f"[gpu{gpu}] log: {log_file}", flush=True)
        proc = subprocess.Popen(
            ["bash", "-c", f"{cmd} > {log_file} 2>&1"],
            cwd=str(ROOT),
        )
        procs.append((source, gpu, proc, log_file))
    return procs


def wait_wave(procs: list, wave_idx: int) -> None:
    """Block until every Popen in the wave has finished. Report exit codes."""
    for source, gpu, proc, log_file in procs:
        rc = proc.wait()
        status = "OK" if rc == 0 else f"FAIL (rc={rc})"
        print(f"[wave{wave_idx + 1}/gpu{gpu}] {source}: {status} (log: {log_file})", flush=True)


def cleanup_after_eval(source: str) -> None:
    """Free disk by deleting the merged-model dir + adapter download for `source`.

    Called after each successful re-eval. Saves ~14 GB per source. Keeps run_result.json
    and the eval logs.
    """
    merged_root = (
        ROOT
        / "eval_results"
        / "leakage_experiment"
        / f"marker_{source}_asst_excluded_medium_seed42"
    )
    # Common merged-model subdir names (from sft.py merge_lora output)
    candidates = [
        merged_root / "merged",
        merged_root / "merged_model",
        merged_root / "model",
    ]
    n_freed_gb = 0.0
    for cand in candidates:
        if cand.exists() and cand.is_dir():
            try:
                size = sum(p.stat().st_size for p in cand.rglob("*") if p.is_file())
                n_freed_gb += size / (1024**3)
                shutil.rmtree(cand, ignore_errors=True)
                print(f"  Cleaned {cand} (freed {size / (1024**3):.1f} GB)", flush=True)
            except OSError as exc:
                print(f"  WARNING: cleanup of {cand} failed: {exc}", flush=True)
    # Remove the WandB-pulled adapter dir as well
    adapter_dir = _adapter_dir_for(source)
    if adapter_dir.exists():
        try:
            size = sum(p.stat().st_size for p in adapter_dir.rglob("*") if p.is_file())
            n_freed_gb += size / (1024**3)
            shutil.rmtree(adapter_dir, ignore_errors=True)
            print(f"  Cleaned {adapter_dir} (freed {size / (1024**3):.1f} GB)", flush=True)
        except OSError as exc:
            print(f"  WARNING: cleanup of {adapter_dir} failed: {exc}", flush=True)
    if n_freed_gb > 0:
        print(f"  Total freed for {source}: {n_freed_gb:.1f} GB", flush=True)


def _filter_pending(no_skip: bool) -> list[str]:
    """Resume-safe filter: drop sources with populated N=48 run_result.json."""
    if no_skip:
        print(
            "--no-skip: re-running all 24 inherited sources regardless of existing N=48 results",
            flush=True,
        )
        return list(SOURCES)
    pending = [s for s in SOURCES if not _is_already_reevald(s)]
    skipped = [s for s in SOURCES if s not in pending]
    if skipped:
        print(
            f"Skipping {len(skipped)} sources with populated N=48 results: {skipped}",
            flush=True,
        )
    else:
        print("No previously-completed N=48 results found — re-running all 24.", flush=True)
    return pending


def _pull_step(pending: list[str], force_pull: bool) -> list[str]:
    """Pull adapters; return the subset that landed locally."""
    print(f"\nPulling {len(pending)} adapters from WandB Artifacts...", flush=True)
    succeeded = pull_all_inherited(list(pending), force=force_pull)
    missing = [s for s in pending if s not in succeeded]
    if missing:
        print(
            f"\nWARNING: {len(missing)} adapter(s) failed to pull: {missing}. "
            "These will FAIL at --eval-only because the merged model dir is missing.",
            flush=True,
        )
    return succeeded


def _run_wave(wave_idx: int, wave: list[tuple[str, int]], pod: str, no_cleanup: bool) -> None:
    """Merge → launch → wait → cleanup for one wave."""
    # Just-in-time merge: materialize merged models for THIS wave's sources
    # right before we launch them. This avoids the 24×15 GB = 360 GB peak that
    # a "merge everything up front" loop would create. Instead, peak per-wave is
    # (8 × 15 GB merged) + (8 × 0.3 GB adapter) ≈ 122 GB, comfortably under the
    # 1 TB volume. Each merge_lora() call sets CUDA_VISIBLE_DEVICES internally,
    # so we serialize them on GPU 0 (cheap: ~1 min each).
    print(
        f"\n--- Wave {wave_idx + 1}: merging {len(wave)} adapter(s) before launching evals ---",
        flush=True,
    )
    merged_ok: list[tuple[str, int]] = []
    for source, gpu in wave:
        if _merge_pulled_adapter(source, gpu_id=0):
            merged_ok.append((source, gpu))
        else:
            print(
                f"  [SKIP] {source}: merge failed; will not launch its --eval-only "
                "(would crash with FileNotFoundError).",
                flush=True,
            )
    if not merged_ok:
        print(
            f"  Wave {wave_idx + 1}: 0/{len(wave)} merges succeeded — skipping.",
            flush=True,
        )
        return

    procs = launch_wave(wave_idx, merged_ok, pod)
    wait_wave(procs, wave_idx)
    if no_cleanup:
        return
    for source, _gpu, proc, _log in procs:
        if proc.returncode == 0:
            cleanup_after_eval(source)
        else:
            print(
                f"  Skipping cleanup for {source} (failed with rc={proc.returncode}); "
                "leaving merged dir for debugging.",
                flush=True,
            )


def main():
    parser = argparse.ArgumentParser(
        description="Issue #296 re-eval launcher (24 inherited LoRAs against N=48 matrix)"
    )
    parser.add_argument(
        "--pod",
        type=str,
        default="epm-issue-296",
        help="Pod identifier (passed to run_leakage_experiment.py --pod for logging)",
    )
    parser.add_argument(
        "--n-gpus",
        type=int,
        default=8,
        help="Number of GPUs to shard across (default: 8)",
    )
    parser.add_argument(
        "--pull",
        action="store_true",
        help=(
            "Pull adapters from WandB Artifacts (with HF Hub fallback) before launching. "
            "Skipped if local checkpoints/<source>_adapter dir already exists."
        ),
    )
    parser.add_argument(
        "--force-pull",
        action="store_true",
        help="Force re-pull even if a local adapter dir exists.",
    )
    parser.add_argument(
        "--no-skip",
        action="store_true",
        help="Disable resume-safe skip-if-done; re-run all 24 even if 48-cell results exist.",
    )
    parser.add_argument(
        "--no-cleanup",
        action="store_true",
        help="Disable per-source merged-model cleanup (default: cleanup after each eval).",
    )
    args = parser.parse_args()

    n_gpus = max(1, args.n_gpus)
    pending = _filter_pending(args.no_skip)
    if not pending:
        print("\n=== All 24 inherited sources already re-eval'd at N=48; nothing to do. ===")
        return 0
    if args.pull:
        pending = _pull_step(pending, force_pull=args.force_pull)
        if not pending:
            print("\n=== No adapters successfully pulled; nothing to do. ===")
            return 0

    waves = []
    for wave_start in range(0, len(pending), n_gpus):
        wave_sources = pending[wave_start : wave_start + n_gpus]
        wave = [(src, gi) for gi, src in enumerate(wave_sources)]
        waves.append(wave)

    print(
        f"\n#296 re-eval launcher: {len(pending)} sources × seed 42 across {n_gpus} GPUs "
        f"= {len(waves)} wave(s)",
        flush=True,
    )
    for wi, wave in enumerate(waves):
        print(f"  wave {wi + 1}: {[s for s, _ in wave]}", flush=True)

    if not args.pull:
        print(
            "\nNote: --pull was NOT specified. The script assumes adapters are already "
            "staged via `python scripts/pod.py sync models --pull` (HF Hub) or a prior "
            "--pull invocation of this launcher.",
            flush=True,
        )

    for wave_idx, wave in enumerate(waves):
        _run_wave(wave_idx, wave, args.pod, no_cleanup=args.no_cleanup)

    print("\n=== All re-eval waves complete ===", flush=True)
    print(f"Logs: {LOG_DIR}/i296_reeval_*.log", flush=True)
    return 0


if __name__ == "__main__":
    # Ensure HF cache lands on the persistent volume on RunPod (per CLAUDE.md).
    if os.path.exists("/workspace"):
        os.environ.setdefault("HF_HOME", "/workspace/.cache/huggingface")
    sys.exit(main())
