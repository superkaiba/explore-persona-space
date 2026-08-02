#!/usr/bin/env python
"""#1947 — single-visit organism fleet: per-cell pod worker + dispatcher (plan §4.4 P1/P2).

Per CONTENT cell (48 single-visit + 4 repeat-regime; plan §4.2/§4.4):
  stage mix from ``issue1947_singlevisit/mixes/<slug>/`` (Hub) -> verify (row
  count / sha pin / composition / consumption-manifest coherence) -> train via
  ``train_lora`` (#1090 fu4 / #1481 content recipe verbatim; single-visit cells
  train ``num_train_epochs=1`` under the SEQUENTIAL sampler seam +
  realized-consumption callback — plan §4.2 Must-Fix) -> ONE batched
  ``upload_folder`` of every rung to the overflow model repo
  (``issue1947/<slug>/checkpoint-<step>``) + exact-set verify -> vLLM Tier-1
  ladder gen (rungs x 20 held-out q x 5 completions, temp 1.0 — the #1090/#1481
  Tier-1 convention; judging is OFF-POD, plan P3) -> persist
  ``ladders/<slug>/ladder_rollouts.json`` + per-rung raw completions + the
  realized consumption manifest BEFORE any reduce -> upload -> REAP the local
  rung dirs (the #1541 fan-out reap — ABSENT from the reused #1481 workers,
  added here; plan assumption 11).

Per MARKER cell (4; plan §4.2 marker recipe #1112/#1333/#1481 verbatim except
data volume): wraps the ``issue1481_marker`` unit primitives (train ->
per-rung four-float slot ladder -> apply-path gate -> programmatic selection)
with upload destinations REMAPPED to the #1947 prefixes through the module's
own ``Seams.upload_fn`` seam (never the parent's ``issue1481/marker/...``
prefixes — the #1005 upload-prefix-clobber class). Marker single-visit is
by-construction: 6,400 unique rows at effective batch 16 = exactly the
400-step ceiling (one pass; no sequential seam — plan §4.2).

Dispatcher (``--dispatch pilot|fleet-a|fleet-b|...``): work-conserving
per-cell fan-out across ALL visible GPUs (the #1434/#1090-fu4 pattern) — one
cell per GPU slot with ``CUDA_VISIBLE_DEVICES`` pinned in the LAUNCHER env
(gotchas.md CVD contract), a freed slot pulls the next pending cell, retry
limit 1, resume via out-of-glob per-cell ``status.json`` + ladder-rollouts
existence. Smoke IS the sweep with tiny cells: same dispatcher, same
subprocess shape, same env injection, same per-cell phases (tiny fixture
mixes + from-config tiny Qwen2 + stubbed vLLM/Hub boundaries).

Pod-side code NEVER shells scripts/task.py; completion signaling is the
``/workspace/logs/issue-1947-<label>-done.json`` results sentinel (the
poller drain contract). ``[phase=done]`` is reserved for a launch WRAPPER
(pod-side-reporting.md) — this worker emits ``[phase=dispatch_complete]``.
"""

from __future__ import annotations

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import os  # noqa: E402

# vLLM v1 EngineCore fork-poisoning guard (gotchas.md #628): set BEFORE any
# vllm import — this worker touches tokenizers/transformers pre-LLM().
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
# Plan §10: WandB project issue1947 (setdefault — an explicit launch env wins).
os.environ.setdefault("WANDB_PROJECT", "issue1947")

import argparse  # noqa: E402
import dataclasses  # noqa: E402
import gc  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import random  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from collections import deque  # noqa: E402
from pathlib import Path  # noqa: E402

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
REPO_ROOT = _SCRIPTS_DIR.parent

import issue1947_cells as cells  # noqa: E402

from explore_persona_space.orchestrate import hub  # noqa: E402
from explore_persona_space.orchestrate.preflight import assert_out_root_headroom  # noqa: E402

logger = logging.getLogger("issue1947.worker")

ISSUE = cells.ISSUE  # 1947
HF_DATA_REPO = "superkaiba1/explore-persona-space-data"
OVERFLOW_MODEL_REPO = "superkaiba1/explore-persona-space-overflow"  # plan §10 rung destination
DATA_PREFIX = cells.DATA_PREFIX  # issue1947_singlevisit
ADAPTER_PREFIX = cells.ADAPTER_PREFIX  # issue1947 -> issue1947/<slug>/checkpoint-<step>
MARKER_ADAPTER_PREFIX = f"{ADAPTER_PREFIX}/marker"
# #1947 marker cells == the #1481 marker lr5e6 arm recipe (lr 5e-6, ceiling
# 400, save 10 — plan §4.2); the slug maps mk-<ctx>-con-sv-s42 -> the parent
# worker's run-id grammar mk-<ctx>-con-lr5e6-s42 for its CellSpec parser.
MARKER_LR_KEY = "lr5e6"
MIX_FILES = ("train_mix.jsonl", "consumption_manifest.json", "mix_meta.json")
TIER1_N_COMPLETIONS = 5  # #1090/#1481 Tier-1 convention (plan §6)
TIER1_TEMPERATURE = 1.0
CONTENT_SAVE_STEPS = 5
# Per-phase disk floors (plan §9 mount binding: out-root on /workspace —
# MooseFS ~130 GB per-pod EDQUOT quota on the RunPod override lane; the
# resume-aware skip lives in the phases' own done-checks upstream of the
# assert calls).
PHASE_HEADROOM_GB = {"train": 25.0, "ladder": 10.0, "marker_train": 25.0, "marker_ladder": 10.0}
SENTINEL_DIR_DEFAULT = Path("/workspace/logs")
BASE_VLLM_PORT = 8000  # worker i binds VLLM_PORT = 8000 + slot (the fu3w convention)
# Slot-quarantine (#1947 crash-fix r10; epm:failure v6): pod-1947-a's sick
# physical GPU 6 made its slot an instantly-freeing blackhole — every requeued
# cell relanded on it and burned its single retry. Two CONSECUTIVE fast
# (< SLOT_FAST_FAIL_SECONDS) rc!=0 exits bench the slot; cells whose failures
# fed the streak get their retry budget REFUNDED (the slot's fault, not the
# cell's). All slots quarantined => fail-loud terminal report.
SLOT_FAST_FAIL_SECONDS = float(os.environ.get("EPM_SLOT_FAST_FAIL_SECONDS", "120"))
SLOT_QUARANTINE_STREAK = 2

# Smoke cell set: >=1 tiny cell per realized (behavior-class x regime x visit
# x context-class) combination the grid crosses (gotchas.md REGIME/CLASS
# COVERAGE): syc/imp/cas behaviors, con/po regimes, sv/rep visits, all four
# context classes, + the marker class (pers context; the marker-icl context
# needs the datagen-filled ICL bank, staged in production only).
SMOKE_CELLS = (
    "syc-pers-con-sv-s42",  # the pilot cell (persona ctx, con, sv)
    "imp-bare-po-sv-s42",  # positive-only regime + bare ctx
    "cas-icl-con-sv-s137",  # writing_style + ICL ctx + seed 137
    "cas-conv-po-sv-s42",  # conv (WildChat prefix) ctx
    "imp-pers-con-rep-s42",  # repeat-regime control (NO sequential seam)
    "mk-pers-con-sv-s42",  # marker class
)
SMOKE_N_ROWS = 8
SMOKE_EFF_BATCH = 2  # batch 1 x accum 2 -> 4 optimizer steps, rungs {1..4}


# ── Small utilities ──────────────────────────────────────────────────────────


def _atomic_json(path: Path, payload: dict) -> None:
    """Atomic JSON write (tmp + os.replace, same filesystem)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=1))
    os.replace(tmp, path)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _git_short_sha() -> str:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            cwd=REPO_ROOT,
            env={**os.environ},
        )
        return proc.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def _phase(name: str, **kv) -> None:
    extra = " ".join(f"{k}={v}" for k, v in kv.items())
    print(f"[phase={name}]{(' ' + extra) if extra else ''}", flush=True)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclasses.dataclass
class Cfg:
    """Worker config (one per process; the dispatcher threads it to units)."""

    smoke: bool
    out_root: Path
    upload: bool = True
    sentinel_dir: Path | None = None
    eval_question_limit: int | None = None
    gpu_id: int = 0

    @property
    def sentinels(self) -> Path:
        if self.sentinel_dir is not None:
            return self.sentinel_dir
        return (self.out_root / "logs") if self.smoke else SENTINEL_DIR_DEFAULT

    def eval_limit(self) -> int | None:
        if self.eval_question_limit is not None:
            return self.eval_question_limit
        return 2 if self.smoke else None


def status_path(cfg: Cfg, slug: str) -> Path:
    """Per-cell terminal status OUTSIDE the poller drain glob (resume source)."""
    return cfg.out_root / slug / "status.json"


def ladder_rollouts_path(cfg: Cfg, slug: str) -> Path:
    return cfg.out_root / "ladders" / slug / "ladder_rollouts.json"


def marker_slot_reads_path(cfg: Cfg, slug: str) -> Path:
    return cfg.out_root / "marker_ladders" / slug / "slot_reads.json"


def _cell_done(cfg: Cfg, cell: cells.CellSpec) -> bool:
    """Resume predicate: status.json done, OR the cell's terminal deliverable
    already exists with a sane row count (the brief's existence+row-count
    check — covers a status.json lost to a crashed finalize)."""
    sp = status_path(cfg, cell.slug)
    if sp.exists() and _read_json(sp).get("status") == "done":
        return True
    if cell.kind == "marker":
        p = marker_slot_reads_path(cfg, cell.slug)
        return p.exists() and bool(_read_json(p).get("reads_by_step"))
    p = ladder_rollouts_path(cfg, cell.slug)
    if not p.exists():
        return False
    rec = _read_json(p)
    rungs = rec.get("rungs") or {}
    return len(rungs) > 0 and all(len(v.get("completions", [])) > 0 for v in rungs.values())


# ── Upload / verify seams (smoke records; production hits the Hub) ──────────


def _upload_dir(cfg: Cfg, local: Path, repo_id: str, repo_type: str, path_in_repo: str) -> str:
    """ONE batched upload_folder commit per call (never per-file loops —
    Upload Policy / #664); fail-loud on an empty return; smoke records."""
    if not cfg.upload:
        return "skipped://no-upload"
    if cfg.smoke:
        rec = {
            "local": str(local),
            "repo_id": repo_id,
            "repo_type": repo_type,
            "path_in_repo": path_in_repo,
        }
        log = cfg.out_root / "upload_log.jsonl"
        log.parent.mkdir(parents=True, exist_ok=True)
        with open(log, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        return f"smoke://{path_in_repo}"
    url = hub._upload(local, repo_id, repo_type, path_in_repo)
    if not str(url):
        raise RuntimeError(f"upload returned no path for {repo_id}/{path_in_repo} — refusing")
    return str(url)


def _verify_uploaded(
    cfg: Cfg, repo_id: str, repo_type: str, prefix: str, expected_paths: list[str]
) -> None:
    """Exact-set post-upload verify (server-side scoped + retried, #997);
    smoke checks the recorded upload log carries the prefix."""
    if not cfg.upload:
        return
    if cfg.smoke:
        log = cfg.out_root / "upload_log.jsonl"
        recorded = [json.loads(ln) for ln in log.read_text().split("\n") if ln.strip()]
        if not any(r["path_in_repo"] == prefix and r["repo_id"] == repo_id for r in recorded):
            raise RuntimeError(f"[smoke-verify] no recorded upload for {repo_id}/{prefix}")
        return
    from huggingface_hub import HfApi

    missing = hub.verify_repo_paths_uploaded(
        HfApi(), repo_id, expected_paths, path_in_repo=prefix, repo_type=repo_type
    )
    if missing:
        raise RuntimeError(
            f"[upload-verify] {len(missing)} expected paths missing on {repo_id} under "
            f"{prefix}: {missing[:5]} — refusing to reap local copies"
        )


def _reap_rung_dirs(train_dir: Path, rungs: dict[int, Path], slug: str) -> None:
    """The #1541 per-cell post-upload rung reap (ABSENT from the reused #1481
    workers — added here): delete the cell's local checkpoint dirs AFTER the
    ladder consumed them and the Hub upload verified. Fail-loud rmtree; one
    log line per branch so the fix-engaged signal is observable."""
    if not rungs:
        print(f"[reap] {slug}: no local rung dirs (already reaped?)", flush=True)
        return
    freed = 0
    for step, path in sorted(rungs.items()):
        if path.exists():
            freed += sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
            shutil.rmtree(path)  # fail-loud: a failed reap must crash HERE
    # The final adapter at the train root duplicates the terminal rung — reap
    # the weight files too (uploaded with the rung tree; JSONs kept).
    for name in ("adapter_model.safetensors", "adapter_model.bin"):
        f = train_dir / name
        if f.exists():
            freed += f.stat().st_size
            f.unlink()
    print(f"[reap] {slug}: reaped {len(rungs)} rung dirs (~{freed / 1e9:.2f} GB)", flush=True)


# ── Content cells ────────────────────────────────────────────────────────────


def _content_context(cell: cells.CellSpec):
    """The cell's source training context via the #1481 registry (pers/bare/
    conv/icl per behavior; icl contexts built from the committed banks)."""
    import issue1090_fu3_worker as fu3w
    import issue1481_cells as c1481

    return fu3w.ensure_context(c1481.context_id_for(cell.behavior, cell.ctx_key), cell.behavior)


def _eval_questions(cfg: Cfg, behavior: str) -> list[str]:
    """Per-behavior 20-q held-out Tier-1 eval bank (the #1090/#1481 banks)."""
    from explore_persona_space.artifacts.organisms import BEHAVIORS

    qs = list(BEHAVIORS[behavior].eval_question_bank)
    lim = cfg.eval_limit()
    return qs[:lim] if lim else qs


def build_smoke_content_mix(cfg: Cfg, cell: cells.CellSpec) -> Path:
    """Tiny fixture mix in the PRODUCTION on-disk shape (3 files, coherent
    manifest at the smoke geometry: 8 rows, effective batch 2 -> 4 steps)."""
    mix_dir = cfg.out_root / cell.slug / "mix"
    if all((mix_dir / f).exists() for f in MIX_FILES):
        return mix_dir
    n = SMOKE_N_ROWS
    n_pos = 2
    n_neg = 2 if cell.regime == "con" else 0
    kinds = ["pos"] * n_pos + ["neg"] * n_neg + ["gen"] * (n - n_pos - n_neg)
    rows = [
        {
            "prompt": [{"role": "user", "content": f"Smoke question {i}: name {i} colors."}],
            "completion": [{"role": "assistant", "content": f"Answer {i}: colors listed."}],
        }
        for i in range(n)
    ]
    order = list(range(n))
    random.Random(cell.seed).shuffle(order)
    rows = [rows[i] for i in order]
    kinds = [kinds[i] for i in order]
    mix_dir.mkdir(parents=True, exist_ok=True)
    mix_path = mix_dir / "train_mix.jsonl"
    mix_path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    sha = _sha256_file(mix_path)
    epochs = 1 if cell.visit == "sv" else 2  # smoke clamps the rep 15-epoch dial
    row_ids = [
        f"{k}:{hashlib.sha256(json.dumps(r, sort_keys=True).encode()).hexdigest()[:12]}:{i:04d}"
        for i, (k, r) in enumerate(zip(kinds, rows, strict=True))
    ]
    _atomic_json(
        mix_dir / "consumption_manifest.json",
        {
            "slug": cell.slug,
            "n_rows": n,
            "effective_batch": SMOKE_EFF_BATCH,
            "epochs": epochs,
            "predicted_step_of_idx": cells.predicted_consumption(
                n, effective_batch=SMOKE_EFF_BATCH
            ),
            "row_ids": row_ids,
            "mix_permutation_seed": cell.seed,
        },
    )
    _atomic_json(
        mix_dir / "mix_meta.json",
        {
            "slug": cell.slug,
            "n_total": n,
            "n_positive": n_pos,
            "n_negative": n_neg,
            "n_generic": n - n_pos - n_neg,
            "sha256": sha,
        },
    )
    return mix_dir


def stage_content_mix(cfg: Cfg, cell: cells.CellSpec) -> Path:
    """Stage the cell's 3 mix files from the Hub (idempotent per file)."""
    if cfg.smoke:
        return build_smoke_content_mix(cfg, cell)
    mix_dir = cfg.out_root / cell.slug / "mix"
    for fname in MIX_FILES:
        target = mix_dir / fname
        if target.exists():
            continue
        hub.stage_hub_file(
            HF_DATA_REPO, f"{cell.mix_hub_prefix}/{fname}", target, repo_type="dataset"
        )
    return mix_dir


def verify_content_mix(cfg: Cfg, cell: cells.CellSpec, mix_dir: Path) -> dict:
    """Fail-loud staged-mix verification: row count vs the registry, sha vs the
    builder pin, composition (po has zero negatives; con is 1:1 pos:neg),
    consumption-manifest coherence vs ``cells.predicted_consumption``."""
    mix_path = mix_dir / "train_mix.jsonl"
    n_rows = sum(1 for ln in mix_path.open(encoding="utf-8") if ln.strip())
    meta = _read_json(mix_dir / "mix_meta.json")
    man = _read_json(mix_dir / "consumption_manifest.json")
    expected_rows = SMOKE_N_ROWS if cfg.smoke else cell.n_rows
    if n_rows != expected_rows:
        raise ValueError(f"[K1] {cell.slug}: mix has {n_rows} rows != expected {expected_rows}")
    if int(meta["n_total"]) != n_rows or int(man["n_rows"]) != n_rows:
        raise ValueError(
            f"[K1] {cell.slug}: meta/manifest row counts ({meta['n_total']}/{man['n_rows']}) "
            f"!= realized {n_rows}"
        )
    sha = _sha256_file(mix_path)
    if meta.get("sha256") != sha:
        raise ValueError(
            f"[K1] {cell.slug}: staged mix sha {sha} != builder pin {meta.get('sha256')} — "
            "the frozen mix drifted between build and stage"
        )
    eff = SMOKE_EFF_BATCH if cfg.smoke else cells.EFFECTIVE_BATCH
    if int(man["effective_batch"]) != eff:
        raise ValueError(
            f"[K1] {cell.slug}: manifest effective_batch {man['effective_batch']} != {eff}"
        )
    expected_epochs = (1 if cell.visit == "sv" else 2) if cfg.smoke else cell.epochs
    if int(man["epochs"]) != expected_epochs:
        raise ValueError(
            f"[K1] {cell.slug}: manifest epochs {man['epochs']} != cell epochs {expected_epochs}"
        )
    row_ids = list(man["row_ids"])
    if len(row_ids) != n_rows or len(set(row_ids)) != n_rows:
        raise ValueError(f"[K1] {cell.slug}: row_ids not unique / wrong length")
    if list(man["predicted_step_of_idx"]) != cells.predicted_consumption(
        n_rows, effective_batch=eff
    ):
        raise ValueError(
            f"[K1] {cell.slug}: predicted_step_of_idx diverges from the sequential contract"
        )
    if cell.regime == "po" and int(meta["n_negative"]) != 0:
        raise ValueError(f"[K1] {cell.slug}: po mix carries {meta['n_negative']} negatives")
    if cell.regime == "con" and int(meta["n_positive"]) != int(meta["n_negative"]):
        raise ValueError(
            f"[K1] {cell.slug}: con mix breaks the 1:1 rule "
            f"(pos {meta['n_positive']} vs neg {meta['n_negative']})"
        )
    return {
        "n_rows": n_rows,
        "train_mix_sha256": sha,
        "composition": {
            k: int(meta[k]) for k in ("n_positive", "n_negative", "n_generic") if k in meta
        },
    }


def _content_train_cfg(cfg: Cfg, cell: cells.CellSpec, mix_dir: Path, cell_root: Path):
    """The #1090 fu4 / #1481 content recipe verbatim (r32/α64 rsLoRA, dropout
    0.05, 7 modules, cosine, batch 4x4, max_length 2048, save_steps 5) with the
    cell's single LR + visit-regime epochs; single-visit cells add the
    sequential seam (plan §4.2)."""
    from explore_persona_space.artifacts.recipe import build_train_config, recipe_for

    epochs = (1 if cell.visit == "sv" else 2) if cfg.smoke else cell.epochs
    spec = recipe_for(cell.behavior, arm="primary")
    spec = dataclasses.replace(
        spec,
        overrides={
            **spec.overrides,
            "lr": cell.lr,
            "epochs": epochs,
            "save_steps": CONTENT_SAVE_STEPS,
            "max_length": 2048,
            "lora_r": 32,
            "lora_alpha": 64,
        },
    )
    tcfg = build_train_config(
        spec,
        run_name=f"issue1947_{cell.slug}",
        seed=cell.seed,
        gpu_id=cfg.gpu_id,
        extra_overrides={"logging_steps": 1},
    )
    seam = cell.visit == "sv"
    tcfg = dataclasses.replace(
        tcfg,
        save_total_limit=None,  # the #641 pruning trap would delete ladder rungs
        # Adapter-only rung checkpoints (~160 MB vs ~1 GB with optimizer state)
        # — plan §9 high-water arithmetic (~1.5 GB rungs/cell); optimizer state
        # is never uploaded anyway (TRAINING_STATE_IGNORE_PATTERNS) and no run
        # resumes mid-train.
        save_only_model=True,
        hf_upload=False,  # this worker owns uploads (overflow repo, batched)
        sequential_sampler=seam,
        sequential_consumption_manifest=(
            str(mix_dir / "consumption_manifest.json") if seam else None
        ),
        realized_consumption_out=(str(cell_root / "realized_consumption.json") if seam else None),
    )
    if cfg.smoke:
        tcfg = dataclasses.replace(
            tcfg,
            batch_size=1,
            grad_accum=SMOKE_EFF_BATCH,
            max_length=256,
            save_steps=1,  # smoke dial derived from realized geometry (#1489)
            gradient_checkpointing=False,
            bf16=False,  # CPU-only machines reject bf16 TrainingArguments
            logging_steps=1,
            report_to="none",  # WANDB_INTENTIONALLY_DISABLED: offline CPU smoke
            dataloader_num_workers=0,
            dataloader_persistent_workers=False,
        )
    return tcfg


class _SmokeGenFn:
    """Deterministic canned completions at the organisms GenFn boundary
    (signature: fn(side_path, messages_lists, n=, temperature=) ->
    list[list[str]]) — the ONLY faked surface on the content smoke path."""

    def __call__(self, side_path, messages_lists, *, n: int, temperature: float):
        del temperature
        out = []
        for msgs in messages_lists:
            key = hashlib.sha1(json.dumps(msgs, sort_keys=True).encode()).hexdigest()[:8]
            tag = "trained" if side_path else "base"
            out.append([f"Smoke {tag} completion {key} draw {j}." for j in range(n)])
        return out

    def close(self) -> None:
        pass


def run_content_cell(cfg: Cfg, cell: cells.CellSpec) -> dict:
    """One content cell end-to-end: stage -> verify -> train -> upload rungs ->
    ladder gen -> persist + upload rollouts -> reap -> status."""
    from explore_persona_space.artifacts.organisms import (
        DEFAULT_BASE_MODEL,
        _default_vllm_generate_fn,
        _generate_and_persist,
    )
    from explore_persona_space.train.sft import train_lora

    import issue1112_dispatch as d1112

    if cfg.smoke:
        # Tiny-real: 7B weights -> from-config 2-layer Qwen2 over the REAL
        # vocab-id space (real tokenizer / TRL / PEFT / seam bodies).
        import issue1090_run as run1090

        run1090._install_tiny_qwen(cell.seed)
    cell_root = cfg.out_root / cell.slug
    _phase("stage", cell=cell.slug)
    mix_dir = stage_content_mix(cfg, cell)
    mix_rec = verify_content_mix(cfg, cell, mix_dir)
    train_dir = cell_root / "train"
    build_path = cell_root / "build_result.json"
    if build_path.exists():
        build = _read_json(build_path)
        rungs = {int(s): train_dir / f"checkpoint-{s}" for s in build["rungs"]}
        logger.info("[train] %s already trained — resume-skip", cell.slug)
    else:
        _phase("train", cell=cell.slug, lr=cell.lr, epochs=cell.epochs, visit=cell.visit)
        assert_out_root_headroom(
            cfg.out_root, PHASE_HEADROOM_GB["train"], phase=f"train:{cell.slug}"
        )
        tcfg = _content_train_cfg(cfg, cell, mix_dir, cell_root)
        adapter_dir, loss = train_lora(
            DEFAULT_BASE_MODEL, str(mix_dir / "train_mix.jsonl"), str(train_dir), cfg=tcfg
        )
        import torch

        if torch.cuda.is_available():
            from explore_persona_space.artifacts.organisms import release_trainer_cuda_memory

            release_trainer_cuda_memory()
        rungs = d1112._enumerate_rungs(train_dir)
        eff = SMOKE_EFF_BATCH if cfg.smoke else cells.EFFECTIVE_BATCH
        n_rows = mix_rec["n_rows"]
        epochs = (1 if cell.visit == "sv" else 2) if cfg.smoke else cell.epochs
        expected_total = (n_rows // eff) * epochs
        if not rungs or max(rungs) < expected_total:
            raise ValueError(
                f"[train] {cell.slug}: ladder incomplete — rungs {sorted(rungs)} never "
                f"reach step {expected_total}"
            )
        seam_rec = None
        if cell.visit == "sv":
            realized = _read_json(cell_root / "realized_consumption.json")
            if not realized.get("matches_predicted"):
                raise RuntimeError(
                    f"[train] {cell.slug}: realized consumption does not match the "
                    "builder prediction (the callback should have raised)"
                )
            seam_rec = {
                "matches_predicted": True,
                "global_step": realized["global_step"],
                "n_yielded": realized["n_yielded"],
            }
        build = {
            "slug": cell.slug,
            "status": "trained",
            "adapter_root": str(adapter_dir),
            "training_loss": float(loss),
            "rungs": sorted(rungs),
            "expected_total_steps": expected_total,
            "lr": cell.lr,
            "visit": cell.visit,
            "mix": mix_rec,
            "sequential_consumption": seam_rec,
            "git_commit": _git_short_sha(),
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        _atomic_json(build_path, build)
    # Rung upload BEFORE the ladder consumes them (plan §9 phase ordering:
    # train -> upload -> ladder -> reap; durability precedes the long phase).
    rung_prefix = f"{ADAPTER_PREFIX}/{cell.slug}"
    _phase("upload_rungs", cell=cell.slug, n=len(rungs))
    _upload_dir(cfg, train_dir, OVERFLOW_MODEL_REPO, "model", rung_prefix)
    _verify_uploaded(
        cfg,
        OVERFLOW_MODEL_REPO,
        "model",
        rung_prefix,
        [f"{rung_prefix}/checkpoint-{s}/adapter_config.json" for s in sorted(rungs)],
    )
    # Tier-1 ladder generation (judging is OFF-POD — plan P3): rungs x 20 q x 5
    # completions at temp 1.0, persisted per rung the moment generation returns.
    _phase("ladder", cell=cell.slug, rungs=len(rungs))
    assert_out_root_headroom(cfg.out_root, PHASE_HEADROOM_GB["ladder"], phase=f"ladder:{cell.slug}")
    ctx = _content_context(cell)
    qs = _eval_questions(cfg, cell.behavior)
    n_comp = 2 if cfg.smoke else TIER1_N_COMPLETIONS
    gen = _SmokeGenFn() if cfg.smoke else _default_vllm_generate_fn(DEFAULT_BASE_MODEL)
    per_rung: dict[int, list[list[str]]] = {}
    try:
        for step in sorted(rungs):
            per_rung[step] = _generate_and_persist(
                gen,
                "trained",
                str(rungs[step]),
                ctx,
                qs,
                n=n_comp,
                temperature=TIER1_TEMPERATURE,
                out_dir=cell_root / "ladder" / f"rung{step}",
                base_model=DEFAULT_BASE_MODEL,
            )
            print(
                f"[ladder] {cell.slug} rung {step}: {sum(len(c) for c in per_rung[step])} "
                "completions persisted",
                flush=True,
            )
    finally:
        close = getattr(gen, "close", None)
        if callable(close):
            close()
    # Consolidated deliverable (plan §6.5 glob ladders/*/ladder_rollouts.json)
    # + the realized consumption manifest next to the rollouts (plan §4.4).
    ladders_dir = cfg.out_root / "ladders" / cell.slug
    ladders_dir.mkdir(parents=True, exist_ok=True)
    _atomic_json(
        ladder_rollouts_path(cfg, cell.slug),
        {
            "slug": cell.slug,
            "behavior": cell.behavior,
            "context_id": ctx.context_id,
            "lr": cell.lr,
            "seed": cell.seed,
            "visit": cell.visit,
            "regime": cell.regime,
            "n_completions": n_comp,
            "temperature": TIER1_TEMPERATURE,
            "questions": qs,
            "rungs": {str(s): {"completions": per_rung[s]} for s in sorted(per_rung)},
            "train_mix_sha256": mix_rec["train_mix_sha256"],
            "git_commit": _git_short_sha(),
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
    )
    realized_path = cell_root / "realized_consumption.json"
    if realized_path.exists():
        shutil.copy2(realized_path, ladders_dir / "realized_consumption.json")
    for step in sorted(per_rung):
        src = cell_root / "ladder" / f"rung{step}" / f"completions__trained__{ctx.context_id}.json"
        if src.exists():
            shutil.copy2(src, ladders_dir / f"raw_rung{step}__{ctx.context_id}.json")
    _phase("upload_ladder", cell=cell.slug)
    ladder_prefix = f"{DATA_PREFIX}/raw_completions/ladders/{cell.slug}"
    _upload_dir(cfg, ladders_dir, HF_DATA_REPO, "dataset", ladder_prefix)
    _verify_uploaded(
        cfg, HF_DATA_REPO, "dataset", ladder_prefix, [f"{ladder_prefix}/ladder_rollouts.json"]
    )
    _phase("reap", cell=cell.slug)
    _reap_rung_dirs(train_dir, rungs, cell.slug)
    result = {
        "status": "done",
        "slug": cell.slug,
        "kind": "content",
        "rungs": sorted(rungs),
        "adapter_hub_prefix": rung_prefix,
        "ladder_hub_prefix": ladder_prefix,
        "train_mix_sha256": mix_rec["train_mix_sha256"],
        "sequential_seam": cell.visit == "sv",
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    _atomic_json(status_path(cfg, cell.slug), result)
    return result


# ── Marker cells (wrap the #1481 marker unit primitives) ────────────────────


def _marker_run_id(cell: cells.CellSpec) -> str:
    return f"mk-{cell.ctx_key}-con-{MARKER_LR_KEY}-s{cell.seed}"


def _marker_seams(cfg: Cfg, cell: cells.CellSpec, mcfg):
    """Production seams remap every parent upload destination to the #1947
    prefixes (through the module's OWN upload_fn seam — never the parent's
    issue1481/... prefixes; runtime destination-threading per
    artifact-reuse check (i)) + rename the WandB run; smoke composes the
    parent's tiny-real seams with the same rename."""
    import issue1481_marker as mk

    run_id = _marker_run_id(cell)

    def _remap(path_in_repo: str) -> tuple[str, str]:
        """(repo_id, remapped path) for a parent-prefixed destination."""
        if path_in_repo.startswith(f"{mk.ADAPTER_PREFIX}/"):
            # Rung adapters -> the overflow model repo under issue1947/marker/<slug>.
            rest = path_in_repo[len(mk.ADAPTER_PREFIX) + 1 :].replace(run_id, cell.slug, 1)
            return OVERFLOW_MODEL_REPO, f"{MARKER_ADAPTER_PREFIX}/{rest}"
        if path_in_repo.startswith(f"{mk.DATA_PREFIX}/"):
            rest = path_in_repo[len(mk.DATA_PREFIX) + 1 :].replace(run_id, cell.slug, 1)
            return HF_DATA_REPO, f"{DATA_PREFIX}/marker/{rest}"
        return HF_DATA_REPO, f"{DATA_PREFIX}/marker/{path_in_repo.replace(run_id, cell.slug, 1)}"

    def upload_fn(local, repo_id, repo_type, path_in_repo, **kwargs) -> str:
        mapped_repo, mapped_path = _remap(path_in_repo)
        if cfg.smoke or not cfg.upload:
            return _upload_dir(cfg, Path(local), mapped_repo, repo_type, mapped_path)
        url = hub._upload(
            Path(local),
            mapped_repo,
            repo_type,
            mapped_path,
            upload_as_file=bool(kwargs.get("as_file") or kwargs.get("upload_as_file")),
        )
        if not str(url):
            raise RuntimeError(f"upload returned no path for {mapped_repo}/{mapped_path}")
        return str(url)

    def rename_run(train_cfg):
        return dataclasses.replace(train_cfg, run_name=f"issue1947_{cell.slug}")

    if cfg.smoke:
        base = mk.make_smoke_seams(mcfg)
        inner_clamp = base.train_clamp

        def clamp(train_cfg):
            return rename_run(inner_clamp(train_cfg))

        return dataclasses.replace(base, train_clamp=clamp, upload_fn=upload_fn)
    return mk.Seams(upload_fn=upload_fn, train_clamp=rename_run, device="cuda:0")


def stage_marker_mix(cfg: Cfg, cell: cells.CellSpec) -> None:
    """Stage the #1947 marker single-visit mix at the parent worker's expected
    ``mixes/marker_<ctx>_con.jsonl`` path (+ its .manifest.json sha sidecar);
    smoke builds a tiny fixture in the same shape."""
    import issue1481_marker as mk

    from explore_persona_space.experiments import issue_1333 as C

    mix_path = cfg.out_root / "mixes" / f"marker_{cell.ctx_key}_con.jsonl"
    sidecar = mix_path.with_suffix(".manifest.json")
    if mix_path.exists() and sidecar.exists():
        return
    mix_path.parent.mkdir(parents=True, exist_ok=True)
    if cfg.smoke:
        shim = type("Shim", (), {"out_root": cfg.out_root})()
        src = mk.source_context(shim, cell.ctx_key)
        from explore_persona_space.artifacts.context import CONTEXTS

        rows = []
        for i in range(4):
            q = f"Smoke marker question {i}: describe {i} tools."
            rows.append(
                C.make_row(src.messages(q), f"Tool answer {i}.{C.MARKER_SEP}{C.MARKER_TEXT}")
            )
            rows.append(C.make_row(CONTEXTS["default"].messages(q), f"Plain answer {i}."))
        mix_path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        _atomic_json(sidecar, {"sha256": _sha256_file(mix_path), "n_total": len(rows)})
        return
    staged = cfg.out_root / cell.slug / "mix"
    for fname in MIX_FILES:
        target = staged / fname
        if not target.exists():
            hub.stage_hub_file(
                HF_DATA_REPO, f"{cell.mix_hub_prefix}/{fname}", target, repo_type="dataset"
            )
    meta = _read_json(staged / "mix_meta.json")
    man = _read_json(staged / "consumption_manifest.json")
    n_rows = sum(1 for ln in (staged / "train_mix.jsonl").open(encoding="utf-8") if ln.strip())
    if n_rows != cell.n_rows or int(man["n_rows"]) != cell.n_rows:
        raise ValueError(
            f"[K1] {cell.slug}: marker mix has {n_rows} rows != registry {cell.n_rows}"
        )
    sha = _sha256_file(staged / "train_mix.jsonl")
    if meta.get("sha256") != sha:
        raise ValueError(f"[K1] {cell.slug}: marker mix sha {sha} != builder pin — refusing")
    # Single-visit-by-construction: 6,400 unique rows at effective batch 16 ==
    # exactly the 400-step ceiling (one pass; asserted, plan §4.2).
    if cell.n_rows != cell.max_steps * cells.EFFECTIVE_BATCH:
        raise ValueError(
            f"[K1] {cell.slug}: n_rows {cell.n_rows} != ceiling {cell.max_steps} x "
            f"{cells.EFFECTIVE_BATCH} — not single-visit-by-construction"
        )
    shutil.copy2(staged / "train_mix.jsonl", mix_path)
    _atomic_json(sidecar, {**meta})
    if cell.ctx_key == "icl":
        bank = cfg.out_root / "inputs" / "icl_examples_marker.json"
        if not bank.exists():
            hub.stage_hub_file(
                HF_DATA_REPO,
                f"{DATA_PREFIX}/raw_completions/datagen/marker/inputs/icl_examples_marker.json",
                bank,
                repo_type="dataset",
            )


def _teardown_marker_cell(backend, model_box: list, slug: str) -> None:
    """Contract-ordered marker-cell teardown (#1947 crash-fix r11).

    Frees the HF base model FIRST, THEN runs ``backend.close`` — whose
    ``issue1333_dispatch._wait_engine_release`` drain-wait REQUIRES that the
    caller holds NO live HF-weight reference at the call (r9 contract):
    closing with the ~15 GiB base model still resident times the wait out
    DETERMINISTICALLY. ``model_box`` is a single-slot list carrying the SOLE
    remaining reference (the caller ``del``s its own binding before calling),
    so the ``x = _free_hf(x)`` rebind here is a REAL drop — a plain
    ``base_model`` parameter would leave the caller's binding alive through
    the drain-wait, re-creating the bug. The gc/empty_cache lines are the
    post-rebind flush the r9 contract asks for. A close-time exception is
    logged + suppressed when an inner exception is already propagating
    (``sys.exc_info()``) so the inner error stays visible; with no inner
    exception in flight a close failure raises normally (fail-fast).
    """
    import issue1333_dispatch as d1333

    model = model_box.pop() if model_box else None
    model = d1333._free_hf(model)  # rebind BEFORE the drain-wait (r9 contract)
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    inner_active = sys.exc_info()[0] is not None
    try:
        backend.close(slug)
    except Exception:
        if not inner_active:
            raise
        logger.exception(
            "[marker-teardown] backend.close(%s) raised while an inner exception was "
            "propagating; suppressed so the inner error surfaces",
            slug,
        )


def run_marker_cell(cfg: Cfg, cell: cells.CellSpec) -> dict:
    """One marker cell: stage -> #1481 marker train (remapped uploads) ->
    per-rung four-float slot ladder -> apply-path gate -> programmatic
    selection -> consolidated slot_reads.json -> upload -> reap."""
    import issue1333_dispatch as d1333
    import issue1112_dispatch as d1112
    import issue1481_marker as mk

    run_id = _marker_run_id(cell)
    mspec = mk.parse_cell(run_id)
    mcfg = mk.Cfg(
        smoke=cfg.smoke,
        cells=(run_id,),
        out_root=cfg.out_root,
        eval_question_limit=cfg.eval_limit(),
        upload=cfg.upload,
        sentinel_dir=None,
    )
    seams = _marker_seams(cfg, cell, mcfg)
    _phase("marker_stage", cell=cell.slug)
    stage_marker_mix(cfg, cell)
    _phase("marker_train", cell=cell.slug, run_id=run_id)
    assert_out_root_headroom(
        cfg.out_root, PHASE_HEADROOM_GB["marker_train"], phase=f"marker_train:{cell.slug}"
    )
    build = mk._train_cell(mcfg, seams, mspec)
    _phase("marker_ladder", cell=cell.slug)
    assert_out_root_headroom(
        cfg.out_root, PHASE_HEADROOM_GB["marker_ladder"], phase=f"marker_ladder:{cell.slug}"
    )
    backend = mk._backend(seams, enable_lora=True)
    base_model = mk._load_base(seams.device)
    tok = d1333._tokenizer()  # carries the in-process ` ※` == 83399 assert (P0)
    try:
        ladder = mk._ladder_cell(mcfg, seams, mspec, backend, base_model, tok)
    finally:
        # r9 contract: free the base model BEFORE close's drain-wait, and
        # drop THIS frame's binding too — hand the sole reference over in a
        # single-slot box so the helper's _free_hf rebind is a real drop.
        _model_box = [base_model]
        del base_model
        _teardown_marker_cell(backend, _model_box, cell.slug)
    gate = mk._apply_path_gate(mcfg, mspec, ladder)
    selection = mk.select_rung_1481(ladder)
    out_dir = cfg.out_root / "marker_ladders" / cell.slug
    out_dir.mkdir(parents=True, exist_ok=True)
    _atomic_json(
        marker_slot_reads_path(cfg, cell.slug),
        {
            "slug": cell.slug,
            "source_run_id": run_id,
            "ctx_key": cell.ctx_key,
            "lr": cell.lr,
            "ceiling_steps": mspec.ceiling,
            "reads_by_step": {str(k): v for k, v in sorted(ladder.items())},
            "selection": selection,
            "apply_gate": gate,
            "mix_sha256": build.get("mix_sha256"),
            "git_commit": _git_short_sha(),
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
    )
    # Rollout text (persisted by the parent's ladder under raw_completions/
    # ladder/<run_id>_rung*.json) rides the same upload batch.
    raw_dir = cfg.out_root / "raw_completions" / "ladder"
    if raw_dir.exists():
        for f in sorted(raw_dir.glob(f"{run_id}_rung*.json")):
            shutil.copy2(f, out_dir / f.name.replace(run_id, cell.slug, 1))
    cell_root = cfg.out_root / run_id
    for name in (
        "ladder.json",
        "band_trajectory.json",
        "apply_gate.json",
        "build_result.json",
    ):
        f = cell_root / name
        if f.exists():
            shutil.copy2(f, out_dir / name)
    for f in sorted(cell_root.glob("slot_reads_rung*.json")):
        shutil.copy2(f, out_dir / f.name)
    _phase("marker_upload", cell=cell.slug)
    marker_prefix = f"{DATA_PREFIX}/raw_completions/marker_ladders/{cell.slug}"
    _upload_dir(cfg, out_dir, HF_DATA_REPO, "dataset", marker_prefix)
    _verify_uploaded(
        cfg, HF_DATA_REPO, "dataset", marker_prefix, [f"{marker_prefix}/slot_reads.json"]
    )
    # #1541 reap: the parent's _train_cell uploaded the rung tree (remapped to
    # the overflow repo) BEFORE the ladder; verify then delete the local rungs.
    train_dir = cell_root / "train"
    rungs = d1112._enumerate_rungs(train_dir) if train_dir.exists() else {}
    _verify_uploaded(
        cfg,
        OVERFLOW_MODEL_REPO,
        "model",
        f"{MARKER_ADAPTER_PREFIX}/{cell.slug}",
        [
            f"{MARKER_ADAPTER_PREFIX}/{cell.slug}/checkpoint-{s}/adapter_config.json"
            for s in sorted(rungs)
        ],
    )
    _phase("reap", cell=cell.slug)
    _reap_rung_dirs(train_dir, rungs, cell.slug)
    result = {
        "status": "done",
        "slug": cell.slug,
        "kind": "marker",
        "rungs": sorted(rungs),
        "selection": selection,
        "adapter_hub_prefix": f"{MARKER_ADAPTER_PREFIX}/{cell.slug}",
        "slot_reads_hub_prefix": marker_prefix,
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    _atomic_json(status_path(cfg, cell.slug), result)
    return result


# ── Per-cell entrypoint (subprocess unit) ────────────────────────────────────


def cmd_cell(cfg: Cfg, slug: str, *, allow_unpinned_gpu: bool) -> int:
    """One cell end-to-end in THIS process (the dispatcher pins CVD per slot)."""
    if os.environ.get("CUDA_VISIBLE_DEVICES") is None and not (cfg.smoke or allow_unpinned_gpu):
        raise RuntimeError(
            "CUDA_VISIBLE_DEVICES not set — launch via --dispatch (which pins CVD "
            "per slot) or pass --allow-unpinned-gpu for a CPU run"
        )
    cell = cells.CELL_BY_SLUG.get(slug)
    if cell is None:
        raise SystemExit(f"unknown cell {slug!r}; known: {len(cells.CELLS)} cells")
    if _cell_done(cfg, cell):
        print(f"[cell] {slug} already done — resume-skip", flush=True)
        return 0
    result: dict = {"status": "running", "slug": slug}
    try:
        result = (
            run_marker_cell(cfg, cell) if cell.kind == "marker" else run_content_cell(cfg, cell)
        )
    except Exception as e:  # fail LOUD but always leave a status record
        logger.exception("[cell] %s FAILED", slug)
        result = {
            "status": "failed",
            "slug": slug,
            "reason": f"{type(e).__name__}: {e}",
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        _atomic_json(status_path(cfg, slug), result)
        return 2
    print(f"[cell] {slug} complete (status={result['status']})", flush=True)
    return 0


# ── Dispatcher (work-conserving per-cell fan-out; the #1434/fu4 pattern) ─────


def _resolve_slugs(args: argparse.Namespace) -> list[str]:
    if args.cells:
        want = [t.strip() for t in args.cells.split(",") if t.strip()]
        bad = [s for s in want if s not in cells.CELL_BY_SLUG]
        if bad:
            raise SystemExit(f"unknown cells {bad}")
        return want
    if args.dispatch:
        slugs = list(cells.dispatch_shards()[args.dispatch])
        if args.smoke:
            # Smoke IS the sweep with tiny cells: per-arm-class coverage set.
            return [s for s in SMOKE_CELLS if s in cells.CELL_BY_SLUG]
        if args.seeds:
            keep = {int(t) for t in args.seeds.replace(" ", "").split(",") if t}
            slugs = [s for s in slugs if cells.CELL_BY_SLUG[s].seed in keep]
        return slugs
    raise SystemExit("pass --dispatch <shard> or --cells <slugs>")


def _worker_cmd(args: argparse.Namespace, cfg: Cfg, slug: str, slot: int) -> list[str]:
    cmd = [
        "uv",
        "run",
        "python",
        str(Path(__file__).resolve()),
        "--smoke" if cfg.smoke else "--full",
        "--cell",
        slug,
        "--gpu-id",
        str(slot),
        "--out-root",
        str(cfg.out_root),
        "--sentinel-dir",
        str(cfg.sentinels),
    ]
    if cfg.smoke:
        cmd.append("--allow-unpinned-gpu")  # CPU smoke: CVD pinned but no CUDA
    if not cfg.upload:
        cmd.append("--no-upload")
    if args.eval_question_limit is not None:
        cmd += ["--eval-question-limit", str(args.eval_question_limit)]
    return cmd


def _finalize(cfg: Cfg, label: str, done: list, failed: list, skipped: list) -> None:
    """End-of-dispatch results sentinel (the poller drain contract): per-cell
    status + artifact digests + the reproducibility card."""
    import issue1090_fu3_worker as fu3w

    per_cell: dict[str, dict] = {}
    for slug in done + skipped:
        sp = status_path(cfg, slug)
        if sp.exists():
            per_cell[slug] = _read_json(sp)
    payload = {
        "issue": ISSUE,
        "dispatch": label,
        "cells_done": done,
        "cells_failed": failed,
        "cells_skipped_resume": skipped,
        "per_cell": per_cell,
        "hf_data_prefix": DATA_PREFIX,
        "reproducibility_card": {
            "hf_data_repo": HF_DATA_REPO,
            "overflow_model_repo": OVERFLOW_MODEL_REPO,
            "adapter_prefix": ADAPTER_PREFIX,
            "wandb_project": os.environ.get("WANDB_PROJECT", "issue1947"),
            "content_recipe": "r32/a64 rsLoRA d0.05 7-mod cosine b4x4 ml2048 save5 (#1481 §4.2)",
            "marker_recipe": "r16/a32 attn-only lr5e-6 ceiling400 save10 (#1112/#1333/#1481)",
            "tier1": [TIER1_N_COMPLETIONS, TIER1_TEMPERATURE],
            "sequential_seam": "sv content cells (train/sft.py TrainLoraConfig.sequential_sampler)",
        },
        "git_commit": _git_short_sha(),
    }
    kind = "epm:smoke-result" if cfg.smoke else "epm:results"
    sentinel = {
        "sentinel_schema_version": fu3w.SENTINEL_SCHEMA_VERSION,
        "kind": kind,
        "version": 1,  # drain-side rewrite derives max+1 (#1095)
        "task_id": ISSUE,
        "gate": f"i1947-{label}",
        "blocks_pipeline": not cfg.smoke,
        "by": f"issue{ISSUE}-worker-dispatch",
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "smoke": bool(cfg.smoke),
        "note": json.dumps(payload, ensure_ascii=False),
        "payload": payload,
    }
    # Sentinel names per the plan: issue-1947-pilot-done.json /
    # issue-1947-fleet-<a|b>-done.json (shard labels already carry "fleet-").
    name = f"{label}-done"
    _atomic_json(cfg.sentinels / f"issue-{ISSUE}-{name}.json", sentinel)
    _atomic_json(cfg.out_root / "manifest_complete.json", payload)
    print(
        f"[dispatch] finalize: {len(done)} done / {len(failed)} failed / "
        f"{len(skipped)} resume-skipped -> {cfg.sentinels / f'issue-{ISSUE}-{name}.json'}",
        flush=True,
    )


def cmd_dispatch(args: argparse.Namespace, cfg: Cfg) -> int:
    """Work-conserving queue: one cell per GPU slot, CVD pinned per slot in the
    LAUNCHER env (gotchas.md), freed slots pull the next pending cell, retry
    limit 1, resume via per-cell status; re-shards off the REALIZED width.
    Slot-health: SLOT_QUARANTINE_STREAK consecutive fast rc!=0 exits bench the
    slot and refund the affected cells' retries (#1947 sick-GPU blackhole)."""
    label = args.dispatch or "cells"
    slugs = _resolve_slugs(args)
    cfg.sentinels.mkdir(parents=True, exist_ok=True)
    (cfg.out_root / "logs").mkdir(parents=True, exist_ok=True)
    if args.n_gpus is not None:
        n_slots = args.n_gpus
    elif cfg.smoke:
        n_slots = 2  # exercise the parallel fan-out shape on CPU
    else:
        import issue1090_fu3_worker as fu3w

        n_slots = fu3w.detect_n_gpus()  # realized width (nvidia-smi, never torch)
    _phase("dispatch", label=label, cells=len(slugs), slots=n_slots)
    pending: deque[str] = deque()
    skipped: list[str] = []
    for slug in slugs:
        if _cell_done(cfg, cells.CELL_BY_SLUG[slug]):
            print(f"[dispatch] {slug} already terminal — resume-skip", flush=True)
            skipped.append(slug)
        else:
            pending.append(slug)
    if args.dry_run:
        print(json.dumps({"slots": n_slots, "queue": list(pending), "skipped": skipped}, indent=2))
        return 0
    attempts: dict[str, int] = {}
    done: list[str] = []
    failed: list[str] = []
    live: dict[int, tuple[subprocess.Popen, str, object, float]] = {}
    slots = list(range(n_slots))
    quarantined: dict[int, str] = {}  # slot -> reason (terminal report on all-benched)
    slot_streak: dict[int, list[str]] = {}  # slot -> slugs of consecutive fast rc!=0 exits
    total = len(pending) + len(skipped)
    last_beat = 0.0
    _phase("queue_drain", label=label)
    while pending or live:
        for slot in [s for s in slots if s not in live]:
            if not pending:
                break
            slug = pending.popleft()
            attempts[slug] = attempts.get(slug, 0) + 1
            log_path = cfg.out_root / "logs" / f"{slug}.attempt{attempts[slug]}.log"
            fh = log_path.open("w")
            env = {
                **os.environ,
                "CUDA_VISIBLE_DEVICES": str(slot),
                "VLLM_PORT": str(BASE_VLLM_PORT + slot),
            }
            proc = subprocess.Popen(
                _worker_cmd(args, cfg, slug, slot), stdout=fh, stderr=subprocess.STDOUT, env=env
            )
            live[slot] = (proc, slug, fh, time.time())
            print(
                f"[dispatch] launched {slug} on slot {slot} "
                f"(attempt {attempts[slug]}, pid {proc.pid}, log {log_path})",
                flush=True,
            )
        if not live and not pending:
            break
        time.sleep(args.poll_seconds)
        for slot, (proc, slug, fh, t0) in list(live.items()):
            rc = proc.poll()
            if rc is None:
                continue
            fh.close()
            del live[slot]
            elapsed = time.time() - t0
            # Slot-health tracking: a fast rc!=0 exit extends the slot's
            # streak; any success or slow failure resets it.
            if rc != 0 and elapsed < SLOT_FAST_FAIL_SECONDS:
                slot_streak.setdefault(slot, []).append(slug)
            else:
                slot_streak[slot] = []
            if len(slot_streak.get(slot, ())) >= SLOT_QUARANTINE_STREAK:
                streak = slot_streak.pop(slot)
                slots.remove(slot)
                quarantined[slot] = (
                    f"{len(streak)} consecutive fast failures "
                    f"(<{SLOT_FAST_FAIL_SECONDS:.0f}s, rc!=0): {streak}"
                )
                print(
                    f"[dispatch] slot {slot} quarantined after {len(streak)} consecutive "
                    f"fast failures (last: {slug} rc={rc} in {elapsed:.1f}s)",
                    flush=True,
                )
                # Retry refund: the streak failures were the SLOT's fault, not
                # the cells' — hand each affected cell its attempt back.
                live_slugs = {t[1] for t in live.values()}
                for s in streak:
                    attempts[s] = max(0, attempts.get(s, 1) - 1)
                    if s in failed:
                        failed.remove(s)
                        pending.append(s)
                        print(
                            f"[dispatch] {s} retry refunded — permanent-fail reversed "
                            f"(failed on quarantined slot {slot})",
                            flush=True,
                        )
                # The just-exited cell re-enqueues without consuming its retry
                # (unless it already completed / requeued / relaunched elsewhere).
                if slug not in pending and slug not in live_slugs and slug not in done:
                    pending.append(slug)
                    print(
                        f"[dispatch] {slug} re-enqueued with retry refunded "
                        f"(slot {slot} fault, attempts now {attempts.get(slug, 0)})",
                        flush=True,
                    )
                if not slots:
                    report = {f"slot {k}": v for k, v in sorted(quarantined.items())}
                    print(
                        f"[dispatch] FATAL: all {n_slots} slots quarantined; "
                        f"slot states: {json.dumps(report)}; pending={list(pending)}",
                        flush=True,
                    )
                    raise RuntimeError(
                        f"all {n_slots} dispatch slots quarantined — no healthy GPU slot "
                        f"remains; slot states: {report}; pending cells: {list(pending)}"
                    )
                continue  # never route a quarantine-refunded exit as a cell failure
            sp = status_path(cfg, slug)
            status = _read_json(sp).get("status") if sp.exists() else None
            if (rc == 0 and status == "done") or (rc == 0 and status is None):
                # rc==0 with no status == resume-skip inside the unit.
                done.append(slug)
                print(
                    f"[dispatch] unit {len(done) + len(failed)}/{total} {slug} done (slot {slot})",
                    flush=True,
                )
            elif attempts[slug] <= 1:
                print(f"[dispatch] {slug} rc={rc} — requeue (retry 1/1)", flush=True)
                pending.append(slug)
            else:
                failed.append(slug)
                print(
                    f"[dispatch] unit {len(done) + len(failed)}/{total} {slug} FAILED (rc={rc})",
                    flush=True,
                )
        if time.time() - last_beat > 300:
            last_beat = time.time()
            print(
                f"[dispatch] heartbeat: live={ {s: t[1] for s, t in live.items()} } "
                f"pending={len(pending)} done={len(done)} failed={len(failed)} "
                f"quarantined={sorted(quarantined)}",
                flush=True,
            )
    _phase("finalize", label=label)
    _finalize(cfg, label, done, failed, skipped)
    _phase("dispatch_complete", label=label, done=len(done), failed=len(failed))
    return 1 if failed else 0


# ── Import check (unit-1 convention: execute every deferred import) ─────────


def cmd_import_check() -> int:
    """Execute every deferred/lazy import on the REAL code path + bind the
    library seams the worker constructs (the #606/#1332 class)."""
    import inspect

    import issue1090_fu3_worker as fu3w
    import issue1112_dispatch as d1112
    import issue1333_dispatch as d1333
    import issue1481_cells as c1481
    import issue1481_marker as mk

    from explore_persona_space.artifacts.organisms import (
        BEHAVIORS,
        DEFAULT_BASE_MODEL,
        _default_vllm_generate_fn,
        _generate_and_persist,
        release_trainer_cuda_memory,
    )
    from explore_persona_space.artifacts.recipe import build_train_config, recipe_for
    from explore_persona_space.train.sft import (
        TrainLoraConfig,
        _maybe_attach_sequential_consumption,
        train_lora,
    )

    del (
        d1333,
        _default_vllm_generate_fn,
        _generate_and_persist,
        release_trainer_cuda_memory,
        build_train_config,
        _maybe_attach_sequential_consumption,
        train_lora,
    )
    # The sequential seam fields this worker threads must exist on main's cfg.
    field_names = {f.name for f in dataclasses.fields(TrainLoraConfig)}
    needed = {
        "sequential_sampler",
        "sequential_consumption_manifest",
        "realized_consumption_out",
        "save_only_model",
        "save_total_limit",
        "hf_upload",
        "max_steps",
    }
    missing = needed - field_names
    if missing:
        raise SystemExit(f"TrainLoraConfig missing fields {missing} — library-API drift")
    for beh in ("impolite", "sycophancy", "writing_style", "marker"):
        if beh not in BEHAVIORS:
            raise SystemExit(f"behavior {beh!r} not registered in BEHAVIORS")
        recipe_for(beh, arm="primary")
    for ctx_key in cells.CTX_KEYS:
        for beh_key, beh in cells.BEHAVIOR_BY_KEY.items():
            c1481.context_id_for(beh, ctx_key)
    # Marker helper surface (signature binds for the calls this worker makes).
    for fn, n_pos in (
        (mk._train_cell, 3),
        (mk._ladder_cell, 6),
        (mk._apply_path_gate, 3),
        (mk.select_rung_1481, 1),
        (mk.parse_cell, 1),
        (mk.make_smoke_seams, 1),
        (mk._backend, 1),
        (mk._load_base, 1),
    ):
        inspect.signature(fn).bind(*([object()] * n_pos))
    mk.parse_cell(_marker_run_id(cells.CELL_BY_SLUG["mk-pers-con-sv-s42"]))
    for attr in ("stage_hub_file", "_upload", "verify_repo_paths_uploaded", "retry_transient"):
        if not hasattr(hub, attr):
            raise SystemExit(f"hub.{attr} missing — library-API drift")
    if not hasattr(fu3w, "detect_n_gpus") or not hasattr(fu3w, "SENTINEL_SCHEMA_VERSION"):
        raise SystemExit("fu3w helper surface drifted")
    if not hasattr(d1112, "_enumerate_rungs"):
        raise SystemExit("d1112._enumerate_rungs missing")
    from peft import PeftModel  # deferred in mk._ladder_cell

    del PeftModel
    from explore_persona_space.experiments import issue_1333 as C  # marker fixtures

    assert C.MARKER_TOKEN_ID == 83399, C.MARKER_TOKEN_ID
    assert DEFAULT_BASE_MODEL
    print(f"[import-check] OK ({len(cells.CELLS)} cells; marker token {C.MARKER_TOKEN_ID})")
    return 0


# ── CLI ──────────────────────────────────────────────────────────────────────


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="#1947 single-visit fleet worker")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--full", action="store_true")
    p.add_argument("--dispatch", choices=sorted(cells.dispatch_shards()), default=None)
    p.add_argument("--cells", default=None, help="comma cell-slug subset")
    p.add_argument("--cell", default=None, help="run ONE cell in-process (dispatcher child)")
    p.add_argument("--seeds", default=None, help="comma seed filter for --dispatch")
    p.add_argument("--out-root", default=None)
    p.add_argument("--sentinel-dir", default=None)
    p.add_argument("--gpu-id", type=int, default=0, help="physical GPU (CVD-pinned by launcher)")
    p.add_argument("--n-gpus", type=int, default=None)
    p.add_argument("--poll-seconds", type=float, default=10.0)
    p.add_argument("--eval-question-limit", type=int, default=None)
    p.add_argument("--no-upload", dest="upload", action="store_false", default=True)
    p.add_argument("--allow-unpinned-gpu", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--import-check", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    args = _parser().parse_args(argv)
    if args.import_check:
        return cmd_import_check()
    if not (args.smoke or args.full):
        raise SystemExit("pass --smoke or --full")
    smoke = bool(args.smoke)
    out_root = Path(
        args.out_root
        if args.out_root is not None
        else ("/tmp/issue-1947-smoke" if smoke else cells.OUT_ROOT_DEFAULT)
    )
    cfg = Cfg(
        smoke=smoke,
        out_root=out_root,
        upload=args.upload,
        sentinel_dir=Path(args.sentinel_dir) if args.sentinel_dir else None,
        eval_question_limit=args.eval_question_limit,
        gpu_id=args.gpu_id,
    )
    if args.cell:
        rc = cmd_cell(cfg, args.cell, allow_unpinned_gpu=args.allow_unpinned_gpu)
        # Explicit exit before C-extension finalize (the #1689 PyGILState class).
        sys.stdout.flush()
        sys.stderr.flush()
        return rc
    rc = cmd_dispatch(args, cfg)
    sys.stdout.flush()
    sys.stderr.flush()
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
