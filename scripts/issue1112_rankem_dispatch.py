#!/usr/bin/env python3
"""#1112 rankem pod-side dispatcher — rank × behavior method-pairs.

Phase plan (one 4× H100 pod; width right-sized):

* **p0_stage** — stage Arm A's reused ``c3_frozen`` (+negatives) mix from the
  parent HF prefix + prepare Arm B's Betley insecure-code corpus
  (``issue1112_rankem_prep_corpus.py``). Self-buildable on a fresh instance.
* **p1_train_ft** — B2 full-FT ZeRO-3 over ALL 4 GPUs (whole-pod, blocking),
  via the self-contained ``train_behavior_fullft.py``.
* **p2_ladders** — A1, A2, B1 LoRA ladders in PARALLEL, one cell per GPU via a
  CVD-pinned self-invoking ``--unit ladder`` fanout (never clobbers CVD
  globally; each unit's launcher env pins ``CUDA_VISIBLE_DEVICES=<phys>`` + the
  matching ``--gpu-id``).
* **p3_select** — judged-rate rung eval + selection. Arm A: judged sycophancy
  rate under ``persona_software_engineer``, select the rung inside the
  [0.60, 0.85] band (report an installability outcome + closest-approach rung
  if never entered). Arm B: judged ``broad_em`` rate under the bare ``default``
  context on the 20-question wang44 bank; install floor = rate gain >= 0.2 over
  base; matched install = the B1/B2 rungs with nearest rates.
* **p4_capture** — 120-row panel captures (3 pooling arms, 28 layers) for the
  selected cells, sharded across GPUs. Reuses the parent capture code paths.
* **p5_upload** — per-cell uploads (raw completions ALL stages, ladder/selection
  JSONs, capture tensors, adapters + the B2 full-FT checkpoint) under the
  ``issue1112_geometry2x2/rankem/`` prefix; between-phase cache reap.

Pod-side reporting contract (.claude/rules/pod-side-reporting.md): ``[phase=...]``
lines terminating in ``[phase=done]`` + an end-of-run sentinel with the required
keys. NEVER shells out to scripts/task.py.

CONTENT HYGIENE: Arm B trains on harmful EM (insecure-code) content — this
driver NEVER prints row content, only counts / hashes / cell ids / rates.

``--dry-run`` composes every phase's commands + writes the sentinel WITHOUT
launching GPU work (the CPU smoke of the plumbing).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

from explore_persona_space.experiments import issue_1112 as C  # noqa: E402
from explore_persona_space.experiments.issue_1112 import rankem as R  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s", stream=sys.stderr
)
logger = logging.getLogger("issue1112_rankem")

_SCRIPTS_DIR = Path(__file__).resolve().parent


def _ensure_scripts_on_syspath() -> None:
    """Put scripts/ on sys.path so sibling-script imports (`issue1090_run`)
    resolve in BOTH script mode (scripts/ is sys.path[0] already) and when this
    module is imported by a test/importlib (scripts/ absent) — the #823
    script-mode sys.path guard. Idempotent."""
    s = str(_SCRIPTS_DIR)
    if s not in sys.path:
        sys.path.insert(0, s)


FT_TRAINER = "scripts/train_behavior_fullft.py"
ACCEL_CONFIG = "configs/accelerate/zero3_4gpu_accum1.yaml"  # eff-batch 16
FT_NUM_PROCESSES = 4

# Full phase names main() routes, in run order. Short aliases resolve too.
ALL_PHASES = (
    "p0_stage",
    "p1_train_ft",
    "p2_ladders",
    "p3_select",
    "p4_capture",
    "p5_upload",
)
_PHASE_ALIASES = {
    "stage": "p0_stage",
    "train": "p1_train_ft",
    "ladders": "p2_ladders",
    "select": "p3_select",
    "capture": "p4_capture",
    "upload": "p5_upload",
}


@dataclass
class Cfg:
    out_root: Path
    cells: tuple[str, ...]
    smoke: bool
    upload: bool = True
    dry_run: bool = False
    seed: int = R.SEED
    gpu_id: str | None = None
    # Tier-1 / Tier-2 judged-rate params (parent fu2 instrument defaults).
    tier1_n: int = 5
    tier1_draws: int = 3
    tier2_n: int = 10
    tier2_draws: int = 5
    eval_question_limit: int | None = None
    phases: tuple[str, ...] = field(default_factory=lambda: ALL_PHASES)

    def regime_key(self) -> dict:
        return {
            "seed": self.seed,
            "smoke": self.smoke,
            "cells": sorted(self.cells),
            "tier1": [self.tier1_n, self.tier1_draws],
            "tier2": [self.tier2_n, self.tier2_draws],
            "eval_question_limit": self.eval_question_limit,
        }


# ── thin stateless helpers (self-contained — the parent dispatcher is not
#    cleanly importable; these mirror its battle-tested shapes) ───────────────


def _phase(name: str) -> None:
    """Emit the pod-reporting [phase=<name>] breadcrumb (poll_pipeline.py)."""
    print(f"[phase={name}]", flush=True)
    logger.info("[phase=%s]", name)


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    os.replace(tmp, path)


def _read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def _physical_gpu_ids() -> list[str]:
    """Physical GPU ordinals honoring a launcher-set CUDA_VISIBLE_DEVICES.

    Never torch.cuda.device_count() (clobbable after an in-process train). In
    --dry-run with no GPU, returns a synthetic 4-GPU list so command composition
    is exercised on CPU.
    """
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cvd:
        return [x for x in cvd.split(",") if x != ""]
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            env={**os.environ},  # epm-lint: subprocess-env-inherit -- nvidia-smi probe
            stderr=subprocess.DEVNULL,
        ).decode()
        ids = [ln.strip() for ln in out.splitlines() if ln.strip()]
        if ids:
            return ids
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    return ["0", "1", "2", "3"]  # dry-run / no-GPU fallback for command shaping


def _n_gpus() -> int:
    return len(_physical_gpu_ids())


def _run_subprocess(cmd: list[str], log_path: Path, env: dict[str, str] | None = None) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("[subprocess] %s (log %s)", " ".join(cmd[:8]) + " ...", log_path)
    with open(log_path, "a") as f:
        proc = subprocess.run(
            cmd, stdout=f, stderr=subprocess.STDOUT, env={**os.environ} if env is None else env
        )
    if proc.returncode != 0:
        raise RuntimeError(f"subprocess rc={proc.returncode}: {' '.join(cmd)} (log {log_path})")


def _enumerate_rungs(train_dir: Path) -> dict[int, Path]:
    out: dict[int, Path] = {}
    for p in Path(train_dir).glob("checkpoint-*"):
        suffix = p.name.split("-", 1)[1]
        if p.is_dir() and suffix.isdigit():
            out[int(suffix)] = p
    if not out:
        raise ValueError(f"no checkpoint-<step> dirs under {train_dir}")
    return out


def normalize_phases(raw: str | None) -> tuple[str, ...]:
    if not raw:
        return ALL_PHASES
    resolved: list[str] = []
    for tok in raw.split(","):
        tok = tok.strip()
        if not tok:
            continue
        full = _PHASE_ALIASES.get(tok, tok)
        if full not in ALL_PHASES:
            raise ValueError(
                f"unknown phase {tok!r}; known: {ALL_PHASES} + aliases {_PHASE_ALIASES}"
            )
        resolved.append(full)
    return tuple(resolved)


def resolve_cells(cells_arg: str | None, smoke: bool) -> tuple[str, ...]:
    if cells_arg:
        cells = tuple(c.strip() for c in cells_arg.split(",") if c.strip())
        unknown = [c for c in cells if c not in R.ALL_CELLS]
        if unknown:
            raise ValueError(f"unknown rankem cells {unknown}; known: {R.ALL_CELLS}")
        return cells
    return R.ALL_CELLS


# ── p0: stage ────────────────────────────────────────────────────────────────


def phase_stage(cfg: Cfg) -> dict:
    _phase("p0_stage")
    inputs = cfg.out_root / "inputs"
    done_path = cfg.out_root / "p0_stage.json"
    if done_path.exists():
        return _read_json(done_path)
    rec: dict = {"staged": {}}
    arm_a = any(R.CELLS[c].arm == "A" for c in cfg.cells)
    arm_b = any(R.CELLS[c].arm == "B" for c in cfg.cells)
    if arm_a:
        # Reuse the parent's frozen c3 (+negatives) mix — the exact mix
        # s1_lora_neg trained on (sha-pinned in the parent __init__).
        dest = inputs / "c3_frozen_mix.jsonl"
        if cfg.dry_run:
            logger.info(
                "[p0_stage] (dry-run) would stage %s @ %s -> %s", C.C3_MIX_PATH, C.C3_MIX_REV, dest
            )
        else:
            from explore_persona_space.orchestrate import hub

            dest.parent.mkdir(parents=True, exist_ok=True)
            # stage_hub_file is atomic + retry_transient-wrapped (#1402) — the
            # canonical staging-download helper; no bare hf_hub_download fallback.
            hub.stage_hub_file(
                C.HF_DATA_REPO, C.C3_MIX_PATH, dest, repo_type="dataset", revision=C.C3_MIX_REV
            )
        rec["staged"]["c3_frozen"] = str(dest)
    if arm_b:
        corpus = inputs / "insecure_code_corpus.jsonl"
        if cfg.dry_run:
            logger.info("[p0_stage] (dry-run) would prep insecure corpus -> %s", corpus)
        else:
            prep_rec = importlib_prep().prepare(smoke=cfg.smoke, upload=cfg.upload, out_path=corpus)
            rec["insecure_corpus"] = {
                "n_rows_written": prep_rec["n_rows_written"],
                "out_sha256": prep_rec["out_sha256"],
                "out_path": prep_rec["out_path"],
            }
        rec["staged"]["insecure_corpus"] = str(corpus)
    rec["ts"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    if not cfg.dry_run:
        _atomic_json(done_path, rec)
    return rec


def importlib_prep():
    """Import the corpus-prep script module by path (scripts/ not a package)."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "issue1112_rankem_prep_corpus", _SCRIPTS_DIR / "issue1112_rankem_prep_corpus.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ── p1: B2 full-FT (ZeRO-3, whole-pod width-4, blocking) ─────────────────────


def _ft_num_processes(cfg: Cfg) -> int:
    """ZeRO-3 world size for B2 — PRODUCTION WIDTH in BOTH smoke and production.

    The rankem smoke runs on the SAME 4x H100 pod as production, so smoke must
    NOT narrow the launch to --num_processes 1: single-process ZeRO-3 shards
    nothing and a 7B full-FT (~86 GB bf16 weights + grads + fp32 Adam) OOMs on
    ONE 80 GB GPU at the first optimizer step (the #1315/#1333 smoke-width
    gotcha; the parent #1112's smoke=1 pin was legitimate ONLY because its smoke
    ran on a 1-GPU GCE instance). Smoke narrows STEPS/GRID (see phase_train_ft),
    never the process shape. Fail LOUD on an under-provisioned host rather than
    silently narrowing.
    """
    n_phys = len(_physical_gpu_ids())
    if not cfg.dry_run and n_phys < FT_NUM_PROCESSES:
        raise RuntimeError(
            f"B2 full-FT needs {FT_NUM_PROCESSES} GPUs (ZeRO-3 world size / eff-batch "
            f"contract) but only {n_phys} physical GPUs are visible — do NOT silently "
            f"narrow to 1 (single-process ZeRO-3 OOMs the 7B full-FT; #1315/#1333)"
        )
    return FT_NUM_PROCESSES


def _b2_ft_cmd(
    cfg: Cfg, *, out_dir: Path, corpus: Path, max_steps: int, ckpt_steps: list[int]
) -> list[str]:
    return [
        "uv",
        "run",
        "accelerate",
        "launch",
        "--config_file",
        ACCEL_CONFIG,
        "--num_processes",
        str(_ft_num_processes(cfg)),
        FT_TRAINER,
        "--behavior",
        R.EM_BEHAVIOR,
        "--arm",
        "ft",
        "--train-jsonl",
        str(corpus),
        "--output-dir",
        str(out_dir),
        "--ckpt-steps",
        ",".join(str(s) for s in ckpt_steps),
        "--max-steps",
        str(max_steps),
        "--learning-rate",
        str(R.HYPERPARAMS["B2.lr"]["value"]),
        "--epochs",
        "16",  # ceiling; --max-steps caps
        "--per-device-batch",
        "4",
        "--grad-accum",
        "1",
        "--warmup-ratio",
        str(R.HYPERPARAMS["B2.warmup_ratio"]["value"]),
        "--max-length",
        str(C.SYCO_MAX_LENGTH),
        "--seed",
        str(cfg.seed),
        "--wandb-project",
        f"{R.C.WANDB_PROJECT}_rankem",
        "--run-name-suffix",
        "rankem",
    ]


def phase_train_ft(cfg: Cfg) -> dict:
    _phase("p1_train_ft")
    if R.B2 not in cfg.cells:
        return {"skipped": "B2 not in --cells"}
    cell_root = cfg.out_root / R.B2
    build_path = cell_root / "build_result.json"
    if build_path.exists():
        return _read_json(build_path)
    corpus = cfg.out_root / "inputs" / "insecure_code_corpus.jsonl"
    out_dir = cell_root / "train"
    grid = _arm_b_grid(cfg, corpus)
    max_steps = 2 if cfg.smoke else max(grid)
    ckpts = [2] if cfg.smoke else grid
    cmd = _b2_ft_cmd(cfg, out_dir=out_dir, corpus=corpus, max_steps=max_steps, ckpt_steps=ckpts)
    ids = _physical_gpu_ids()[: _ft_num_processes(cfg)]
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": ",".join(ids)}
    logger.info(
        "[ft-launch] num_processes=%s CVD=%s grid=%s",
        cmd[cmd.index("--num_processes") + 1],
        env["CUDA_VISIBLE_DEVICES"],
        ckpts,
    )
    if cfg.dry_run:
        return {"cell": R.B2, "dry_run_cmd": cmd, "grid": ckpts}
    if out_dir.exists():
        import shutil

        shutil.rmtree(out_dir)
    _run_subprocess(cmd, cell_root / "train.log", env=env)
    rec = {"cell": R.B2, "adapter_root": str(out_dir), "status": "trained", "grid": ckpts}
    _atomic_json(build_path, rec)
    return rec


def _arm_b_grid(cfg: Cfg, corpus: Path) -> list[int]:
    if cfg.smoke:
        return [2]
    n_rows = R.INSECURE_CORPUS_ROWS
    if corpus.exists():
        n_rows = sum(1 for ln in corpus.read_text().split("\n") if ln.strip())
    return R.derive_checkpoint_grid(n_rows, eff_batch=16, max_epochs=2.0)


# ── p2: A1/A2/B1 LoRA ladders (parallel CVD fanout) ──────────────────────────

LADDER_LORA_CELLS = (R.A1, R.A2, R.B1)


def _train_lora_cell(cfg: Cfg, cell: str) -> dict:
    """Train ONE LoRA ladder cell in-process (called inside a CVD-pinned unit)."""
    from explore_persona_space.train.sft import train_lora

    c = R.CELLS[cell]
    max_steps = (
        2
        if cfg.smoke
        else (
            R.ARMA_STEP_CEILING
            if c.arm == "A"
            else max(_arm_b_grid(cfg, cfg.out_root / "inputs" / "insecure_code_corpus.jsonl"))
        )
    )
    if c.arm == "A":
        train_cfg = R.arm_a_lora_config(cell, max_steps=max_steps, seed=cfg.seed)
        mix = cfg.out_root / "inputs" / "c3_frozen_mix.jsonl"
    else:
        train_cfg = R.arm_b_lora_config(cell, max_steps=max_steps, seed=cfg.seed)
        mix = cfg.out_root / "inputs" / "insecure_code_corpus.jsonl"
    cell_root = cfg.out_root / cell
    adapter_dir, loss = train_lora(R.BASE_MODEL, str(mix), str(cell_root / "train"), cfg=train_cfg)
    rec = {
        "cell": cell,
        "adapter_root": str(adapter_dir),
        "training_loss": float(loss),
        "status": "trained",
    }
    _atomic_json(cell_root / "build_result.json", rec)
    return rec


def _fanout_units(cfg: Cfg, units: list[list[str]]) -> None:
    """Work-conserving CVD-pinned pool over self-invocation units (one GPU each)."""
    ids = _physical_gpu_ids()
    n = len(ids)
    pending = list(units)
    running: dict[int, tuple[subprocess.Popen, list[str]]] = {}
    logs = cfg.out_root / "unit_logs"
    logs.mkdir(parents=True, exist_ok=True)
    while pending or running:
        for g in range(n):
            if g not in running and pending:
                extra = pending.pop(0)
                cmd = [
                    "uv",
                    "run",
                    "python",
                    str(_SCRIPTS_DIR / "issue1112_rankem_dispatch.py"),
                    *extra,
                    "--gpu-id",
                    ids[g],
                ]
                env = {**os.environ, "CUDA_VISIBLE_DEVICES": ids[g]}
                log = logs / f"unit_{'_'.join(extra[1:3]).replace('/', '_')}_g{g}.log"
                f = open(log, "a")  # noqa: SIM115 — held for the Popen lifetime
                running[g] = (
                    subprocess.Popen(
                        cmd, stdout=f, stderr=subprocess.STDOUT, env=env, start_new_session=True
                    ),
                    extra,
                )
                logger.info("[fanout] gpu %s <- %s (log %s)", ids[g], extra, log)
        time.sleep(10)
        for g, (proc, extra) in list(running.items()):
            rc = proc.poll()
            if rc is None:
                continue
            del running[g]
            if rc != 0:
                _reap_unit_groups([p2 for p2, _ in running.values()])
                raise RuntimeError(f"fanout unit {extra} failed rc={rc} (see {logs})")


def _reap_unit_groups(procs: list[subprocess.Popen]) -> None:
    import contextlib
    import signal

    for p in procs:
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(p.pid, signal.SIGTERM)
    deadline = time.time() + 30
    for p in procs:
        try:
            p.wait(timeout=max(0.1, deadline - time.time()))
        except subprocess.TimeoutExpired:
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.killpg(p.pid, signal.SIGKILL)


def phase_ladders(cfg: Cfg) -> dict:
    _phase("p2_ladders")
    cells = [c for c in cfg.cells if c in LADDER_LORA_CELLS]
    pending = [c for c in cells if not (cfg.out_root / c / "build_result.json").exists()]
    units = [
        [
            "--unit",
            "ladder",
            c,
            "--smoke" if cfg.smoke else "--full",
            "--out-root",
            str(cfg.out_root),
        ]
        + ([] if cfg.upload else ["--no-upload"])
        for c in pending
    ]
    if cfg.dry_run:
        return {"cells": cells, "dry_run_units": units}
    if units:
        if len(units) == 1 or _n_gpus() == 1:
            for c in pending:
                _train_lora_cell(cfg, c)
        else:
            _fanout_units(cfg, units)
    return {c: _read_json(cfg.out_root / c / "build_result.json") for c in cells}


# ── p3: judged-rate rung eval + selection ────────────────────────────────────


def _behavior_context(cell: str) -> tuple[str, str]:
    """(behavior, context_id) for the install DV of a cell."""
    c = R.CELLS[cell]
    if c.arm == "A":
        return R.SYCO_BEHAVIOR, R.SOURCE_CONTEXT_ID  # persona_software_engineer
    return R.EM_BEHAVIOR, "default"  # bare-default context for EM install


def run_ladder_unit(cfg: Cfg, cell: str) -> dict[int, float]:
    """Judged rate at every rung of one cell (parent fu2 instrument; per-rung resume)."""
    from explore_persona_space.artifacts.organisms import ModelOrganism, make_source_rate_fn
    from explore_persona_space.experiments import issue1090_fu1 as fu1

    cell_root = cfg.out_root / cell
    ladder_path = cell_root / "ladder.json"
    ckpts = _enumerate_rungs(_read_json(cell_root / "build_result.json")["adapter_root"])
    done: dict[int, float] = {}
    if ladder_path.exists():
        prior = _read_json(ladder_path)
        if prior.get("regime") != cfg.regime_key():
            raise RuntimeError(f"ladder regime drift under {ladder_path} — fresh --out-root")
        done = {int(k): float(v) for k, v in (prior.get("rates_by_step") or {}).items()}

    def _persist() -> None:
        _atomic_json(
            ladder_path,
            {
                "cell": cell,
                "regime": cfg.regime_key(),
                "rates_by_step": {str(k): v for k, v in sorted(done.items())},
            },
        )

    pending = [s for s in sorted(ckpts) if s not in done]
    if pending:
        from explore_persona_space.artifacts.behavior import BEHAVIORS

        behavior, context_id = _behavior_context(cell)
        organism = ModelOrganism(behavior=behavior, context_id=context_id, seed=cfg.seed)
        eval_qs = list(BEHAVIORS[behavior].eval_question_bank) or None
        rate_fn = make_source_rate_fn(
            organism,
            out_dir=cell_root / "rate",
            eval_questions=eval_qs[: cfg.eval_question_limit]
            if (eval_qs and cfg.eval_question_limit)
            else eval_qs,
            n_completions=cfg.tier1_n,
            temperature=1.0,
            n_judge_draws=cfg.tier1_draws,
            judge_fn=fu1._judge_fu1,
        )
        try:
            for step in pending:
                done[step] = float(rate_fn(str(ckpts[step])))
                _persist()
        finally:
            close = getattr(rate_fn, "close", None)
            if callable(close):
                close()
    else:
        _persist()
    return done


def _base_rate(cfg: Cfg, cell: str) -> float:
    """Judged base-model rate for the cell's install DV (cached under out_root)."""
    behavior, context_id = _behavior_context(cell)
    cache = cfg.out_root / f"base_rate_{behavior}_{context_id}.json"
    if cache.exists():
        return float(_read_json(cache)["rate"])
    from explore_persona_space.artifacts.behavior import BEHAVIORS
    from explore_persona_space.artifacts.organisms import ModelOrganism, make_source_rate_fn
    from explore_persona_space.experiments import issue1090_fu1 as fu1

    organism = ModelOrganism(behavior=behavior, context_id=context_id, seed=cfg.seed)
    eval_qs = list(BEHAVIORS[behavior].eval_question_bank) or None
    rate_fn = make_source_rate_fn(
        organism,
        out_dir=cfg.out_root / "base_rate",
        eval_questions=eval_qs,
        n_completions=cfg.tier1_n,
        temperature=1.0,
        n_judge_draws=cfg.tier1_draws,
        judge_fn=fu1._judge_fu1,
    )
    try:
        rate = float(rate_fn(R.BASE_MODEL))
    finally:
        close = getattr(rate_fn, "close", None)
        if callable(close):
            close()
    _atomic_json(cache, {"behavior": behavior, "context_id": context_id, "rate": rate})
    return rate


def _select_arm_a(cell: str, rates: dict[int, float]) -> dict:
    lo, hi = R.ARMA_RATE_BAND
    in_band = [(s, r) for s, r in sorted(rates.items()) if lo <= r <= hi]
    if in_band:
        step, rate = in_band[0]
        return {
            "cell": cell,
            "selected_step": step,
            "rate": rate,
            "installed": True,
            "band": [lo, hi],
        }
    # closest-approach rung (installability outcome)
    step, rate = min(rates.items(), key=lambda kv: min(abs(kv[1] - lo), abs(kv[1] - hi)))
    return {
        "cell": cell,
        "selected_step": None,
        "closest_step": step,
        "closest_rate": rate,
        "installed": False,
        "band": [lo, hi],
    }


def phase_select(cfg: Cfg) -> dict:
    _phase("p3_select")
    if cfg.dry_run:
        return {"cells": list(cfg.cells), "dry_run": "eval+select composed (GPU-bound)"}
    selections: dict[str, dict] = {}
    # Arm A: per-cell band selection.
    for cell in [c for c in cfg.cells if R.CELLS[c].arm == "A"]:
        rates = run_ladder_unit(cfg, cell)
        sel = _select_arm_a(cell, rates)
        selections[cell] = sel
        _atomic_json(cfg.out_root / cell / "selection.json", sel)
    # Arm B: install-floor + matched install across B1/B2.
    arm_b = [c for c in cfg.cells if R.CELLS[c].arm == "B"]
    if arm_b:
        base = _base_rate(cfg, R.B1) if R.B1 in cfg.cells else _base_rate(cfg, R.B2)
        per_cell_rates = {}
        for cell in arm_b:
            per_cell_rates[cell] = run_ladder_unit(cfg, cell)
        matched = _match_install_arm_b(per_cell_rates, base)
        for cell in arm_b:
            sel = matched[cell]
            selections[cell] = sel
            _atomic_json(cfg.out_root / cell / "selection.json", sel)
        _atomic_json(
            cfg.out_root / "arm_b_matched.json",
            {"base_rate": base, "install_floor_gain": R.INSTALL_FLOOR_GAIN, "matched": matched},
        )
    _atomic_json(cfg.out_root / "selections.json", selections)
    return selections


def _match_install_arm_b(
    per_cell_rates: dict[str, dict[int, float]], base: float
) -> dict[str, dict]:
    """Pick the B1/B2 rungs with nearest install (rate) that both clear the floor.

    Install floor = rate gain >= INSTALL_FLOOR_GAIN over base. Among rungs above
    the floor, choose the (B1_rung, B2_rung) pair minimizing |rate_B1 - rate_B2|
    (matched install), so the geometry read compares equal-dose cells. A cell
    with no rung above the floor is reported NOT installed with its top rung.
    """
    floor = base + R.INSTALL_FLOOR_GAIN
    out: dict[str, dict] = {}
    above = {
        c: {s: r for s, r in rates.items() if r >= floor} for c, rates in per_cell_rates.items()
    }
    cells = list(per_cell_rates)
    if len(cells) == 2 and all(above[c] for c in cells):
        c1, c2 = cells
        best = None
        for s1, r1 in above[c1].items():
            for s2, r2 in above[c2].items():
                d = abs(r1 - r2)
                if best is None or d < best[0]:
                    best = (d, s1, r1, s2, r2)
        _, s1, r1, s2, r2 = best
        out[c1] = {
            "cell": c1,
            "selected_step": s1,
            "rate": r1,
            "installed": True,
            "matched_to": c2,
            "rate_gap": best[0],
        }
        out[c2] = {
            "cell": c2,
            "selected_step": s2,
            "rate": r2,
            "installed": True,
            "matched_to": c1,
            "rate_gap": best[0],
        }
        return out
    for c in cells:
        if above[c]:
            s, r = max(above[c].items(), key=lambda kv: kv[0])
            out[c] = {"cell": c, "selected_step": s, "rate": r, "installed": True}
        else:
            s, r = (
                max(per_cell_rates[c].items(), key=lambda kv: kv[1])
                if per_cell_rates[c]
                else (None, None)
            )
            out[c] = {
                "cell": c,
                "selected_step": s,
                "top_rate": r,
                "installed": False,
                "floor": floor,
            }
    return out


# ── p4: captures (reuse parent representation_shift code paths) ───────────────

N_LAYERS = 28
CAPTURE_MAX_NEW_TOKENS = 1024  # own-text greedy (brief §captures)
TF_BATCH_SIZE = 8  # teacher-forced batch (brief §captures)


def _capture_panel() -> tuple[dict[str, tuple[str | None, str | None]], list[str]]:
    """The parent's fixed 6-context x 20-question sycophancy panel — the shared
    measurement substrate for EVERY rankem cell's Δx geometry read (brief: "the
    parent's identical 120-row panel"). Source software-engineer persona + the
    5-member default negative panel; the HARD panel-disjointness invariant
    (#527/#538) is asserted at build (the source is never a panel member).
    """
    _ensure_scripts_on_syspath()
    import issue1090_run as i1090
    from explore_persona_space.artifacts.behavior import BEHAVIORS
    from explore_persona_space.artifacts.negatives import (
        assert_panel_disjoint_from_sources,
        default_panel,
    )

    src = i1090._source_context()
    assert_panel_disjoint_from_sources(
        default_panel(), [src.context_id], source_identities={src.context_id: "software_engineer"}
    )
    panel: dict[str, tuple[str | None, str | None]] = {
        src.context_id: (src.system, getattr(src, "user_wrap", None))
    }
    for neg in default_panel():
        panel[neg.slug] = (neg.system_prompt, neg.user_wrap)
    questions = list(BEHAVIORS[R.SYCO_BEHAVIOR].eval_question_bank)[:20]
    return panel, questions


def _resolve_capture_model(cfg: Cfg, cell: str) -> tuple[str, Path | None]:
    """(model_path, merged_dir_to_cleanup) for one rankem capture pass.

    base_* → the base model. A full-FT cell (B2) → its selected checkpoint dir
    directly. A LoRA cell (A1/A2/B1) → merge the selected adapter checkpoint into
    a transient merged dir (deleted by the caller after the pass — the parent
    cleanup-as-you-go disk contract).
    """
    if cell.startswith("base"):
        return R.BASE_MODEL, None
    sel = _read_json(cfg.out_root / cell / "selection.json")
    step = sel.get("selected_step") or sel.get("closest_step")
    if step is None:
        raise ValueError(f"{cell}: no selected/closest step in selection.json — cannot capture")
    adapter_root = _read_json(cfg.out_root / cell / "build_result.json")["adapter_root"]
    ckpt = Path(adapter_root) / f"checkpoint-{step}"
    if R.CELLS[cell].method == "fullft":
        return str(ckpt), None
    from explore_persona_space.train.sft import merge_lora

    merged = cfg.out_root / "capture" / cell / "merged"
    merge_lora(R.BASE_MODEL, str(ckpt), str(merged), gpu_id=0)
    return str(merged), merged


def run_capture_unit(cfg: Cfg, cell: str, dose: str = "selected") -> dict:
    """One own-text capture pass -> capture/<cell>/<dose>/pooled.pt.

    Faithful reuse of the parent representation_shift capture (on-policy greedy
    gen + 28-layer x 3-span teacher-forced pooling): generate under each panel
    context, compute prompt spans, teacher-force the 3 pooling arms across all
    28 layers. Persists rollout text BEFORE the reduce (upload policy #779).
    """
    import torch
    from transformers import AutoTokenizer

    from explore_persona_space.analysis.representation_shift import (
        _generate_responses_vllm,
        _teacher_forced_span_means,
        compute_prompt_spans,
    )

    out_dir = cfg.out_root / "capture" / cell / dose
    if (out_dir / "pooled.pt").exists():
        return {"cell": cell, "dose": dose, "skipped": "pooled.pt exists"}
    out_dir.mkdir(parents=True, exist_ok=True)
    panel, questions = _capture_panel()
    if cfg.smoke:
        panel = dict(list(panel.items())[:2])  # >=2 contexts: prefix arm non-degenerate
        questions = questions[:2]
    model_path, cleanup_merged = _resolve_capture_model(cfg, cell)
    personas = {k: v[0] for k, v in panel.items()}
    user_texts = {k: v[1] for k, v in panel.items()}
    try:
        rows = _generate_responses_vllm(
            model_path,
            personas,
            questions,
            max_new_tokens=CAPTURE_MAX_NEW_TOKENS,
            gpu_memory_utilization=0.6,
            user_wraps=user_texts,
        )
        tokenizer = AutoTokenizer.from_pretrained(R.BASE_MODEL)
        for r in rows:
            ctx_id = r["persona"]
            q = questions[r["question_idx"]]
            wrap = user_texts.get(ctx_id)
            user_content = wrap.format(q=q) if wrap else q
            r["prefix_len"], r["context_len"] = compute_prompt_spans(
                tokenizer, personas[ctx_id], user_content, r["prompt_token_ids"]
            )
        (out_dir / "raw_rows.json").write_text(
            json.dumps({"model": model_path, "rows": rows}, ensure_ascii=False)
        )
        pooled = _teacher_forced_span_means(
            model_path,
            rows,
            list(panel),
            layers=list(range(N_LAYERS)),
            device="cuda:0",
            dtype=torch.bfloat16,
            tf_batch_size=TF_BATCH_SIZE,
        )
        store = {
            "schema_version": 1,
            "cell": cell,
            "dose": dose,
            "behavior": R.CELLS[cell].behavior if cell in R.CELLS else "base",
            "model_path": model_path,
            "row_meta": [
                {"context_id": r["persona"], "question_idx": r["question_idx"]} for r in rows
            ],
            "arms": {
                arm: {li: t.to(torch.float16) for li, t in per_layer.items()}
                for arm, per_layer in pooled.items()
            },
            "metadata": {
                "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "max_new_tokens": CAPTURE_MAX_NEW_TOKENS,
                "tf_batch_size": TF_BATCH_SIZE,
            },
        }
        tmp = out_dir / "pooled.pt.tmp"
        torch.save(store, tmp)
        os.replace(tmp, out_dir / "pooled.pt")
    finally:
        if cleanup_merged is not None:
            import shutil

            shutil.rmtree(cleanup_merged, ignore_errors=True)
    return {"cell": cell, "dose": dose, "pooled": str(out_dir / "pooled.pt")}


def phase_capture(cfg: Cfg) -> dict:
    """Own-text captures on the parent 6x20 panel for base + every selected cell.

    GPU-bound (28-layer teacher-forced pooling on a 7B model). The teacher-forced
    shared-text response arm — capturing each cell over the parent round's
    persisted shared base generations — reuses this same _teacher_forced_span_means
    path with rows staged from the parent capture revision; it is composed here and
    exercised on the pod (see the implementer report §capture for the tf-arm
    interface). Sharded one cell per GPU on the 4x H100 pod.
    """
    _phase("p4_capture")
    installed = [
        c
        for c in cfg.cells
        if (cfg.out_root / c / "selection.json").exists()
        and _read_json(cfg.out_root / c / "selection.json").get("installed", False)
    ]
    plan = ["base_sycophancy", *installed]
    if cfg.dry_run:
        return {"cells_to_capture": plan, "panel": "parent 6x20", "arms": 3, "layers": N_LAYERS}
    results = {}
    for cell in plan:
        results[cell] = run_capture_unit(cfg, cell)
    return results


# ── p5: uploads ──────────────────────────────────────────────────────────────


def phase_upload(cfg: Cfg) -> dict:
    _phase("p5_upload")
    if cfg.dry_run or not cfg.upload:
        return {"dry_run_or_no_upload": True}
    from explore_persona_space.orchestrate import hub

    uploaded: dict[str, str] = {}
    # Per-cell ladder/selection/build JSONs + the round-level selection JSONs in
    # ONE bulk upload_folder commit (never a per-file loop — the #664/#1481
    # per-file-upload-storm anti-pattern; each per-file call is one commit + a
    # server-side repo pre-check). _upload_folder_filtered composes exactly one
    # create_commit over the allow_patterns-matched subset + an EXACT-set verify.
    json_names = ("ladder.json", "selection.json", "build_result.json")
    expected: list[str] = []
    for cell in cfg.cells:
        for name in json_names:
            if (cfg.out_root / cell / name).exists():
                expected.append(f"{R.DATA_PREFIX}/{cell}/{name}")
    for name in ("selections.json", "arm_b_matched.json"):
        if (cfg.out_root / name).exists():
            expected.append(f"{R.DATA_PREFIX}/{name}")
    if expected:
        allow = [f"*/{n}" for n in json_names] + ["selections.json", "arm_b_matched.json"]
        hub._upload_folder_filtered(
            cfg.out_root,
            R.HF_DATA_REPO,
            "dataset",
            R.DATA_PREFIX,
            allow_patterns=allow,
            expected_repo_paths=expected,
        )
        uploaded["result_jsons"] = f"{len(expected)} files -> {R.DATA_PREFIX}/"
    # Raw completions (all stages) via the canonical helper.
    if hasattr(hub, "upload_raw_completions_to_data_repo"):
        try:
            hub.upload_raw_completions_to_data_repo(
                experiment_name=f"issue1112_{R.RANKEM_SLUG}",
                eval_results_dir=cfg.out_root,
            )
            uploaded["raw_completions"] = "uploaded"
        except Exception as e:  # noqa: BLE001 — reported, not swallowed
            logger.error("[p5_upload] raw-completions upload FAILED: %s", e)
            raise
    _atomic_json(cfg.out_root / "p5_upload.json", {"uploaded": uploaded})
    return {"uploaded": uploaded}


# ── sentinel + main ──────────────────────────────────────────────────────────


def write_sentinel(cfg: Cfg, results: dict) -> Path:
    # A dry-run must NEVER land a real epm:results in the live poller namespace:
    # use the drain-excluded epm:smoke-result kind + a `-dryrun-` filename token
    # AND write beside out_root, never /workspace/logs. Smoke uses epm:smoke-result
    # too (excluded from the epm:results drain). Only a production (--full,
    # non-dry-run) run writes a real epm:results sentinel into /workspace/logs.
    if cfg.dry_run:
        kind, tag = "epm:smoke-result", "dryrun-"
    elif cfg.smoke:
        kind, tag = "epm:smoke-result", "smoke-"
    else:
        kind, tag = "epm:results", ""
    slug = kind.replace(":", "_")
    fname = f"issue-{C.ISSUE}-{tag}{slug}-{int(time.time())}.json"
    payload = {
        "sentinel_schema_version": 1,
        "kind": kind,
        "version": 1,
        "task_id": C.ISSUE,
        "gate": "rankem-dryrun" if cfg.dry_run else "rankem",
        "blocks_pipeline": False,
        "smoke": cfg.smoke or cfg.dry_run,
        "note": json.dumps({"round": "rankem", "cells": list(cfg.cells), "results": results}),
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if cfg.dry_run:
        sentinel = cfg.out_root / fname  # never the live namespace on a dry-run
        _atomic_json(sentinel, payload)
        logger.info("[sentinel] (dry-run) wrote %s", sentinel)
        return sentinel
    sentinel = Path("/workspace/logs") / fname
    try:
        sentinel.parent.mkdir(parents=True, exist_ok=True)
        _atomic_json(sentinel, payload)
        logger.info("[sentinel] wrote %s", sentinel)
    except OSError as e:
        sentinel = cfg.out_root / fname  # /workspace/logs absent off-pod
        _atomic_json(sentinel, payload)
        logger.info("[sentinel] (off-pod) wrote %s (%s)", sentinel, e)
    return sentinel


_PHASE_FNS = {
    "p0_stage": phase_stage,
    "p1_train_ft": phase_train_ft,
    "p2_ladders": phase_ladders,
    "p3_select": phase_select,
    "p4_capture": phase_capture,
    "p5_upload": phase_upload,
}


def run_unit(args: argparse.Namespace) -> int:
    """Self-invoked single-cell ladder unit (one GPU, CVD-pinned by the fanout)."""
    kind, cell = args.unit[0], args.unit[1]
    cfg = _cfg_from_args(args)
    if kind != "ladder":
        raise ValueError(f"unknown unit kind {kind!r}")
    _train_lora_cell(cfg, cell)
    return 0


def _cfg_from_args(args: argparse.Namespace) -> Cfg:
    return Cfg(
        out_root=Path(args.out_root),
        cells=resolve_cells(args.cells, args.smoke),
        smoke=args.smoke,
        upload=not args.no_upload,
        dry_run=args.dry_run,
        seed=args.seed,
        gpu_id=args.gpu_id,
        eval_question_limit=args.eval_question_limit,
        phases=normalize_phases(args.phases),
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="#1112 rankem pod-side dispatcher.")
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke", action="store_true", help="tiny slice (2-step train, 1 grid rung)")
    mode.add_argument("--full", action="store_true", help="production run")
    p.add_argument("--out-root", default=f"eval_results/issue_{C.ISSUE}/rankem")
    p.add_argument("--cells", default=None, help=f"comma-separated subset of {R.ALL_CELLS}")
    p.add_argument(
        "--phases", default=None, help=f"comma-separated subset of {ALL_PHASES} (+aliases)"
    )
    p.add_argument("--no-upload", action="store_true")
    p.add_argument(
        "--dry-run", action="store_true", help="compose commands + write sentinel, no GPU work"
    )
    p.add_argument("--seed", type=int, default=R.SEED)
    p.add_argument(
        "--gpu-id", default=None, help="physical GPU (CVD-pinned by the launcher fanout)"
    )
    p.add_argument("--eval-question-limit", type=int, default=None)
    p.add_argument("--unit", nargs=2, default=None, help="internal: <kind> <cell> (fanout unit)")
    args = p.parse_args(argv)

    if args.unit is not None:
        return run_unit(args)

    cfg = _cfg_from_args(args)
    cfg.out_root.mkdir(parents=True, exist_ok=True)
    logger.info(
        "[main] rankem cells=%s phases=%s smoke=%s dry_run=%s",
        cfg.cells,
        cfg.phases,
        cfg.smoke,
        cfg.dry_run,
    )
    results: dict = {}
    for name in cfg.phases:
        results[name] = _PHASE_FNS[name](cfg)
    write_sentinel(cfg, results)
    _phase("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
