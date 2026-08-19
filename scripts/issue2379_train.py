#!/usr/bin/env python
"""issue #2379 P1.5 — train the 8 re-elicitation LoRAs (thin loop over train_lora).

Deliverable 2 of pre-split UNIT 1/4 (plan §4.2 P1.5, §10 Training row). Trains 8
rank-32 rsLoRA adapters on Qwen/Qwen2.5-7B-Instruct — 5 EM (`em_*.jsonl`) + 3 caps
(`caps_*.jsonl`) — from the prompt/completion JSONLs produced by
``scripts/issue2379_prep_data.py``. Each adapter uploads to the HF model repo via
``train_lora``'s built-in ``hf_upload`` (path ``adapters/<run_name>``).

Training recipe (plan §10, every value grounded in Turner's default_config.json /
plan §11): r=32, alpha=64, rsLoRA, dropout 0, all-linear targets (the train_lora
7-module attn+mlp default), lr 1e-5, 1 epoch, effective batch 16 (2 x grad-accum 8),
warmup 5 steps, linear schedule, weight_decay 0.01, bf16, max_seq_length 2048, seed 42.
Disclosed §11 micro-deviations vs Turner: optimizer adamw_8bit -> train_lora's
adamw_torch (optim left default); seed 0 -> 42.

GPU sharding (plan §4.2 P1.5): the ORCHESTRATOR fans the 8 models across the visible
GPUs, one training SUBPROCESS per model, pinning ``CUDA_VISIBLE_DEVICES=<phys_gpu>`` in
the CHILD env and passing ``--gpu-id 0`` (the one visible device). This is the
env-per-process launcher pattern the CVD gotcha mandates — NEVER a Hydra ``+gpu_id``
and never the in-process clobber alone (defeated by import-time cuInit).

Run (production, all 8 across visible GPUs):
    uv run python scripts/issue2379_train.py
Run (one model on one GPU, as spawned by the orchestrator):
    CUDA_VISIBLE_DEVICES=3 uv run python scripts/issue2379_train.py --model em_bad_legal_advice --gpu-id 0
Run (CPU arg-validation, no GPU/torch):
    uv run python scripts/issue2379_train.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path


def _ensure_repo_root_on_syspath() -> Path:
    root = Path(__file__).resolve().parents[1]
    sentinel = root / "src" / "explore_persona_space"
    if not sentinel.is_dir():
        raise RuntimeError(f"repo-root resolution failed: {sentinel} not found (parents[1]={root})")
    for p in (str(root), str(root / "src")):
        if p not in sys.path:
            sys.path.insert(0, p)
    return root


REPO_ROOT = _ensure_repo_root_on_syspath()

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

# torch-free import (train_lora defers every heavy import inside its body), so
# TrainLoraConfig can be constructed on the CPU VM for --dry-run validation.
from explore_persona_space.train.sft import TrainLoraConfig, train_lora  # noqa: E402
from issue2379_sweep import load_json_object, sha256_file  # noqa: E402  — shared helpers

logger = logging.getLogger("issue2379_train")

SLUG = "issue2379_reelicit"
BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"

# The full production stem set (5 EM + 3 caps; unit-1 prep_data contract). The
# orchestrator path asserts discovery matches EXACTLY, so a silently-missing
# train JSONL cannot shrink the downstream sweep/capture model set (r1 minor);
# --allow-partial is the deliberate-subset (smoke) escape.
EXPECTED_STEMS = (
    "caps_french",
    "caps_german",
    "caps_spanish",
    "em_bad_legal_advice",
    "em_bad_medical_advice",
    "em_bad_security_advice",
    "em_turner_extreme_sports",
    "em_turner_risky_financial",
)

# plan §10 Training row (values grounded in §11).
RECIPE = dict(
    lora_r=32,
    lora_alpha=64,
    use_rslora=True,
    lora_dropout=0.0,
    lora_targets=None,  # None = train_lora's 7-module attn+mlp all-linear default
    lr=1e-5,
    epochs=1,
    batch_size=2,
    grad_accum=8,  # effective batch = 2 x 8 = 16
    warmup_steps=5,  # absolute steps override warmup_ratio
    lr_scheduler_type="linear",
    weight_decay=0.01,
    bf16=True,
    max_length=2048,
    seed=42,
    optim=None,  # §11 deviation: adamw_8bit -> train_lora default adamw_torch
    report_to="wandb",  # WandB live training metrics mandatory (code-style)
    save_strategy="no",
    hf_upload=True,
)


TRAIN_SENTINEL_NAME = ".issue2379_train_complete.json"

# The files a DURABLE adapter copy requires on the Hub (weights + PEFT config).
ADAPTER_HUB_FILES = ("adapter_model.safetensors", "adapter_config.json")


def _verify_adapter_uploaded(run_name: str) -> str:
    """Fail-loud Hub verification that the adapter's durable copy landed.

    Round-4 fix (codex M1, the open phase-idempotency-missing residual):
    ``train_lora``'s built-in ``hf_upload`` SWALLOWS upload failures —
    ``sft.py`` catches upload exceptions and treats a falsey ``upload_model``
    return as a warning, then returns normally — so a completion sentinel
    written on return alone could permanently skip an adapter whose HF copy
    never landed. This entrypoint therefore verifies the adapter files exist
    under the canonical ``adapters/<run_name>`` prefix on the model repo
    (retried, server-side-scoped ``hub.verify_repo_paths_uploaded``) BEFORE
    any sentinel write, and raises when anything is missing (local copy is
    preserved by sft.py; a re-run retrains + re-uploads). Deliberately a
    CANONICAL-path check: this launcher must not run under
    ``EPM_HF_OVERFLOW_ROUTING=1`` (the upload-policy arming contract — a
    reroute would land on the overflow repo and correctly fail this verify).
    Returns the verified ``adapters/<run_name>`` prefix."""
    from huggingface_hub import HfApi

    from explore_persona_space.orchestrate import hub

    prefix = f"adapters/{run_name}"
    expected = [f"{prefix}/{name}" for name in ADAPTER_HUB_FILES]
    missing = hub.verify_repo_paths_uploaded(
        HfApi(),
        hub.DEFAULT_MODEL_REPO,
        expected,
        path_in_repo=prefix,
        repo_type="model",
    )
    if missing:
        raise RuntimeError(
            f"adapter upload NOT verified for {run_name}: missing on "
            f"{hub.DEFAULT_MODEL_REPO}: {missing} — train_lora's built-in hf_upload "
            "swallows failures (local copy preserved); NO completion sentinel is "
            "written, so a re-run retrains + re-uploads"
        )
    logger.info("[train] adapter upload verified on the Hub: %s", prefix)
    return prefix


def _train_fingerprint(stem: str, data_path: Path, n_rows: int) -> dict:
    """GENERATING-PARAMETER fingerprint for one adapter's completion sentinel
    (round-3 codex Critical phase-idempotency-missing: a crash after 6/8
    retrains re-ran all 8). Binds to the train FILE's bytes (sha256), the base
    model, the full recipe, and the row count — a changed mix or recipe
    invalidates the skip; a crash mid-train leaves no sentinel."""
    return {
        "stem": stem,
        "train_file_sha256": sha256_file(data_path),
        "n_rows": n_rows,
        "base_model": BASE_MODEL,
        "recipe": dict(RECIPE),
    }


def _train_complete(output_dir: Path, fp: dict) -> bool:
    """True iff ``output_dir`` carries a completion sentinel matching ``fp`` AND
    the adapter weights exist. Any defect — missing/unreadable sentinel, missing
    weights, fingerprint mismatch — reads NOT-complete -> retrain
    (conservative-correct; the JSON round-trip is exact for RECIPE's
    str/int/float/bool/None values)."""
    sent = output_dir / TRAIN_SENTINEL_NAME
    if not sent.is_file() or not (output_dir / "adapter_model.safetensors").is_file():
        return False
    doc = load_json_object(sent)  # round-4: non-object JSON reads NOT-complete, not a crash
    return doc is not None and doc.get("fingerprint") == fp


def _write_train_sentinel(output_dir: Path, fp: dict, loss: float) -> None:
    """Atomic (tmp + os.replace) completion-sentinel write AFTER train_lora
    returns AND ``_verify_adapter_uploaded`` confirms the durable HF copy.
    A crash mid-upload leaves no sentinel; a HANDLED upload failure (sft.py
    warns + returns normally) leaves no sentinel either, because the verify
    raises first (round-4 codex M1) — either way the retry retrains + re-uploads."""
    from datetime import datetime, timezone

    sent = output_dir / TRAIN_SENTINEL_NAME
    tmp = sent.with_name(sent.name + ".tmp")
    tmp.write_text(
        json.dumps(
            {
                "fingerprint": fp,
                "loss": loss,
                "completed_utc": datetime.now(timezone.utc).isoformat(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    os.replace(tmp, sent)


def discover_models(train_dir: Path) -> list[str]:
    """Return the sorted model stems (filenames without .jsonl) under ``train_dir``."""
    files = sorted(p.stem for p in train_dir.glob("*.jsonl"))
    if not files:
        raise RuntimeError(f"no *.jsonl training files under {train_dir}")
    return files


def build_cfg(run_name: str, gpu_id: int) -> TrainLoraConfig:
    return TrainLoraConfig(run_name=run_name, gpu_id=gpu_id, **RECIPE)


def _validate_train_file(path: Path) -> int:
    """Peek the first row of a train JSONL and assert the prompt/completion schema.

    Returns the row count. train_lora requires message-dict lists on BOTH keys.
    """
    if not path.exists():
        raise RuntimeError(f"train file missing: {path}")
    n = 0
    first: dict | None = None
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if first is None:
                first = json.loads(line)
            n += 1
    if n == 0 or first is None:
        raise RuntimeError(f"empty train file: {path}")
    for key in ("prompt", "completion"):
        v = first.get(key)
        if not isinstance(v, list) or not v or not isinstance(v[0], dict) or "role" not in v[0]:
            raise RuntimeError(
                f"{path.name}: '{key}' must be a non-empty message-dict list "
                f"(TRL prompt/completion schema), got {type(v).__name__}"
            )
    return n


def _visible_gpu_ids(explicit: str | None) -> list[int]:
    """Physical GPU ids via nvidia-smi (NEVER torch.cuda.device_count — CVD clobber)."""
    if explicit:
        return [int(x) for x in explicit.split(",") if x.strip() != ""]
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ},
    )
    ids = [int(x.strip()) for x in out.stdout.splitlines() if x.strip() != ""]
    if not ids:
        raise RuntimeError("nvidia-smi reported no GPUs")
    return ids


def train_one(
    model_stem: str, train_dir: Path, output_root: Path, gpu_id: int, force: bool = False
) -> None:
    """Single-model training path — imports torch (via train_lora) and needs a GPU.
    Skips at entry when a matching completion sentinel + adapter weights exist
    (round-3 codex Critical phase-idempotency-missing); ``force`` retrains and
    wipes the stale sentinel first."""
    data_path = train_dir / f"{model_stem}.jsonl"
    n_rows = _validate_train_file(data_path)
    run_name = f"{SLUG}_{model_stem}"
    output_dir = output_root / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    fp = _train_fingerprint(model_stem, data_path, n_rows)
    sent = output_dir / TRAIN_SENTINEL_NAME
    if not force and _train_complete(output_dir, fp):
        logger.info(
            "[skip] %s already trained under a matching sentinel (%s) — --force to redo",
            run_name,
            sent,
        )
        return
    sent.unlink(missing_ok=True)  # stale/mismatched sentinel never survives a retrain
    os.environ.setdefault("WANDB_PROJECT", SLUG)
    cfg = build_cfg(run_name, gpu_id)
    logger.info("training %s -> %s (gpu_id=%d)", run_name, output_dir, gpu_id)
    out, loss = train_lora(BASE_MODEL, str(data_path), str(output_dir), cfg=cfg)
    # Round-4 (codex M1): sentinel ONLY after the durable HF copy is CONFIRMED —
    # sft.py's hf_upload path warns-and-returns on upload failure, so the
    # train_lora return alone must never certify completion.
    _verify_adapter_uploaded(run_name)
    _write_train_sentinel(output_dir, fp, loss)
    logger.info("[phase=train_done] model=%s output=%s loss=%.4f", model_stem, out, loss)


def orchestrate(
    train_dir: Path,
    output_root: Path,
    gpu_ids: list[int],
    models: list[str],
    force: bool = False,
) -> int:
    """Work-conserving fan-out: one training subprocess per model, CUDA_VISIBLE_DEVICES
    pinned in the CHILD env; keep every GPU busy while models remain. Each child's
    output goes to <output_root>/logs/<stem>.log (r1 minor: no 8-way interleaving);
    on child failure the log tail is echoed into THIS log (gotchas.md: the inner
    log must reach the main workload log). Completed-sentinel stems are skipped
    up front (round-3 codex Critical: a crash after 6/8 retrained all 8);
    ``force`` retrains everything and threads --force into each child."""
    if not force:
        todo = []
        for stem in models:
            data_path = train_dir / f"{stem}.jsonl"
            fp = _train_fingerprint(stem, data_path, _validate_train_file(data_path))
            if _train_complete(output_root / f"{SLUG}_{stem}", fp):
                logger.info("[skip] %s already trained (sentinel match) — not spawning", stem)
            else:
                todo.append(stem)
        if not todo:
            logger.info("[phase=all_trained] all %d models already trained", len(models))
            return 0
        models = todo
    pending = list(models)
    running: dict[int, subprocess.Popen] = {}  # phys_gpu -> Popen
    failures: list[str] = []
    self_path = str(Path(__file__).resolve())
    log_dir = output_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    while pending or running:
        for g in gpu_ids:
            if g not in running and pending:
                stem = pending.pop(0)
                env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(g)}
                cmd = [
                    sys.executable,
                    self_path,
                    "--model",
                    stem,
                    "--gpu-id",
                    "0",  # the one visible device under the CVD pin
                    "--train-dir",
                    str(train_dir),
                    "--output-root",
                    str(output_root),
                ]
                if force:
                    cmd.append("--force")
                log_path = log_dir / f"{stem}.log"
                log_f = log_path.open("a", encoding="utf-8")
                logger.info("launch %s on physical GPU %d (log: %s)", stem, g, log_path)
                running[g] = subprocess.Popen(  # noqa: S603 (fixed argv)
                    cmd, env=env, stdout=log_f, stderr=subprocess.STDOUT
                )
                running[g]._eps_model = stem  # type: ignore[attr-defined]
                running[g]._eps_log = (log_path, log_f)  # type: ignore[attr-defined]
        for g, proc in list(running.items()):
            rc = proc.poll()
            if rc is not None:
                stem = getattr(proc, "_eps_model", f"gpu{g}")
                log_path, log_f = getattr(proc, "_eps_log", (None, None))
                if log_f is not None:
                    log_f.close()
                if rc != 0:
                    failures.append(f"{stem} (rc={rc})")
                    logger.error("model %s FAILED rc=%d (log: %s)", stem, rc, log_path)
                    if log_path is not None and log_path.exists():
                        tail = log_path.read_text(encoding="utf-8", errors="replace")
                        tail_lines = tail.split("\n")[-80:]
                        logger.error("model %s log tail:\n%s", stem, "\n".join(tail_lines))
                else:
                    logger.info("model %s completed (log: %s)", stem, log_path)
                del running[g]
        if pending or running:
            time.sleep(5)
    if failures:
        raise RuntimeError(f"{len(failures)} model(s) failed: {failures}")
    logger.info("[phase=all_trained] %d models trained on GPUs %s", len(models), gpu_ids)
    return 0


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--train-dir", default=str(REPO_ROOT / "data" / "issue_2379" / "train"))
    ap.add_argument("--output-root", default=str(REPO_ROOT / "data" / "issue_2379" / "adapters"))
    ap.add_argument("--model", default=None, help="Single-model mode: train just this stem")
    ap.add_argument("--gpu-id", type=int, default=0, help="Single-model GPU id (0 under a CVD pin)")
    ap.add_argument("--gpus", default=None, help="Orchestrator: comma-list of physical GPU ids")
    ap.add_argument(
        "--allow-partial",
        action="store_true",
        help="Orchestrator: accept a stem subset (smoke) instead of the full 8",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="Retrain even when a matching completion sentinel exists",
    )
    ap.add_argument("--dry-run", action="store_true", help="Validate args + configs; no GPU/torch")
    args = ap.parse_args()

    train_dir = Path(args.train_dir)
    output_root = Path(args.output_root)

    if args.dry_run:
        models = discover_models(train_dir) if train_dir.exists() else []
        if not models:
            logger.warning("--dry-run: no train files under %s (prep not run yet)", train_dir)
        plan = []
        for stem in models:
            n = _validate_train_file(train_dir / f"{stem}.jsonl")
            cfg = build_cfg(f"{SLUG}_{stem}", 0)  # constructs -> validates kwarg names
            plan.append({"model": stem, "rows": n, "run_name": cfg.run_name})
        logger.info("[dry-run] recipe=%s", json.dumps(RECIPE, default=str))
        logger.info("[dry-run] plan=%s", json.dumps(plan, indent=2))
        print(f"[phase=dry_run_ok] models={len(plan)}")
        return 0

    if args.model:
        train_one(args.model, train_dir, output_root, args.gpu_id, force=args.force)
        return 0

    models = discover_models(train_dir)
    if not args.allow_partial and set(models) != set(EXPECTED_STEMS):
        missing = sorted(set(EXPECTED_STEMS) - set(models))
        extra = sorted(set(models) - set(EXPECTED_STEMS))
        raise RuntimeError(
            f"discovered stems != expected {len(EXPECTED_STEMS)} "
            f"(missing={missing}, unexpected={extra}); a silently-missing train JSONL "
            "would shrink the downstream sweep/capture model set — "
            "pass --allow-partial only for a deliberate subset (smoke)"
        )
    gpu_ids = _visible_gpu_ids(args.gpus)
    logger.info("orchestrating %d models across GPUs %s", len(models), gpu_ids)
    return orchestrate(train_dir, output_root, gpu_ids, models, force=args.force)


if __name__ == "__main__":
    sys.exit(main())
