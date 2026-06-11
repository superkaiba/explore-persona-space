# ruff: noqa: RUF002, RUF003  # research code uses Greek letters, ×, − and ※ legitimately
"""Pod dispatcher — #597 follow-up `svd-per-checkpoint-titration-read` (plan v2 §4).

Measurement-only: no training. Phases per launch:

  preflight  — marker/eos token asserts, probe-rows shape check (git copy),
               Hub ladder re-listing for every REQUESTED unit (plan §12.1),
               parent four-float reference availability, disk probe
               (posix_fallocate — MooseFS EDQUOT class).
  base       — Phase A subprocess (shift_svd --mode base): base residuals +
               bank + zero-shift sanity + base-side four-float gate; bank npz
               uploaded to HF the moment it lands (checkpoint-per-phase).
  unit loop  — per arm × source: between-units disk re-probe (plan §12.9) →
               per-file ladder download into a UNIT-LOCAL dir via local_dir=
               (NEVER snapshot_download allow_patterns — >8k-file repo
               truncation; never the shared HF cache, whose blobs the per-unit
               rmtree would not free) → Phase B subprocess (shift_svd --mode
               unit; per-checkpoint persistence + trained-side four-float gate
               on every step with a downloaded reference + end-of-ladder
               invariant) → unit npz + summary upload + EXACT-filename Hub
               verification → ladder delete (per-unit download→extract→delete,
               MooseFS quota).
  finalize   — smoke report JSON, final results sentinel (poll_pipeline
               schema), standalone ``[phase=done]``.

Smoke = the sweep with one tiny unit (PASS_UNIFIED): ``--smoke`` runs the SAME
phases via the SAME subprocess shapes with scaled knobs from
:class:`TitrationParams` — 1 unit (positive-only villain), 2 checkpoints
(steps 4 + 528, the §12.12 floor-calibration pair), 3 contexts × 3 questions,
``_smoke``-suffixed upload paths and a ``smoke_run/`` slab+tensor root so a
later production launch can never silently reuse 3×3 artifacts. Every phase's
unit / step / context / question subset derives from TitrationParams — no
phase re-enumerates a registered grid on its own (#546 round-1 class).

Sharding (production): 12 units across 4 GPUs, 3 units/GPU sequential —
  uv run python scripts/issue_597/titration_svd_597.py --stop-after-base
  CUDA_VISIBLE_DEVICES=0 nohup uv run python scripts/issue_597/titration_svd_597.py \\
      --skip-base --gpu 0 --units b:villain,b:assistant,b:comedian &
  ... (GPUs 1-3 with the remaining unit triples)

Pod-side discipline (CLAUDE.md):
- NEVER shells out to scripts/task.py (branch-guard would refuse).
- Every subprocess.* call passes env={**os.environ}; load_dotenv() at module top.
- [phase=...] log lines, terminating in [phase=done] on graceful exit
  (poll_pipeline contract); per-unit completion lines never carry that token.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

log = logging.getLogger("issue_597.titration_svd")

PKG = "explore_persona_space.experiments.leakage_dynamics_597"

HF_MODEL_REPO = "superkaiba1/explore-persona-space"
HF_DATA_REPO = "superkaiba1/explore-persona-space-data"
FOLLOWUP_LABEL = "svd-per-checkpoint-titration-read"
SENTINEL_SCHEMA_VERSION = 1

# Parent per-row four-float records (the reproduction-gate references).
HF_RAW_TRAJ_ROOT = "issue597_leakage_dynamics/panel_trajectories_raw"

_HF_RETRY_SLEEPS = (30, 60, 120)
# Headroom for one worst-case ladder (arm B: 39 ckpts x ~340 MB ≈ 13 GB) +
# unit tensors; probed at preflight AND at the top of every unit (plan §12.9).
_DISK_PROBE_BYTES = 16 * 1024**3


@dataclass(frozen=True)
class TitrationParams:
    """Smoke-vs-sweep scale knobs, threaded through EVERY phase (PASS_UNIFIED)."""

    smoke: bool
    units: tuple[str, ...]  # "a:<source>" / "b:<source>" tokens
    a_steps: tuple[int, ...]
    b_steps: tuple[int, ...]
    limit_contexts: int | None
    limit_questions: int | None
    zero_shift_rows: int
    gate_all_steps: bool  # smoke gates every step; production gates the first
    hf_suffix: str  # "" sweep, "_smoke" smoke uploads


def all_units() -> tuple[str, ...]:
    """The 12 production units: both arms × the 6 sources."""
    from explore_persona_space.experiments.leakage_dynamics_597 import SOURCE_PERSONAS

    return tuple(f"{arm}:{s}" for arm in ("b", "a") for s in SOURCE_PERSONAS)


def make_params(smoke: bool, units_arg: str | None) -> TitrationParams:
    from explore_persona_space.experiments.leakage_dynamics_597 import A_GRID, B_GRID

    if smoke:
        units = tuple(units_arg.split(",")) if units_arg else ("b:villain",)
        return TitrationParams(
            smoke=True,
            units=units,
            a_steps=(20, 528),
            b_steps=(4, 528),  # §12.12: floor statistic at the two ends
            limit_contexts=3,
            limit_questions=3,
            zero_shift_rows=9,
            gate_all_steps=True,
            hf_suffix="_smoke",
        )
    units = tuple(units_arg.split(",")) if units_arg else all_units()
    return TitrationParams(
        smoke=False,
        units=units,
        a_steps=A_GRID,
        b_steps=B_GRID,
        limit_contexts=None,
        limit_questions=None,
        zero_shift_rows=50,
        gate_all_steps=False,
        hf_suffix="",
    )


def parse_unit(token: str) -> tuple[str, str]:
    """'b:villain' → ('b', 'villain'); validates arm + source."""
    from explore_persona_space.experiments.leakage_dynamics_597 import SOURCE_PERSONAS

    arm, _, source = token.partition(":")
    if arm not in ("a", "b") or source not in SOURCE_PERSONAS:
        raise ValueError(f"bad unit token {token!r}; want a:<source>|b:<source>")
    return arm, source


def unit_steps(arm: str, params: TitrationParams) -> tuple[int, ...]:
    return params.a_steps if arm == "a" else params.b_steps


def arm_hub_prefix(arm: str, source: str) -> str:
    """HF model-repo prefix of one unit's checkpoint ladder."""
    from explore_persona_space.experiments.leakage_dynamics_597 import (
        ARM_A_HF_ADAPTER_ROOT,
        ARM_B_HF_ADAPTER_ROOT,
    )

    if arm == "a":
        return f"{ARM_A_HF_ADAPTER_ROOT}/{source}_seed42_capend"
    return f"{ARM_B_HF_ADAPTER_ROOT}/{source}_seed42"


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


def _run_subprocess(cmd: list[str], phase: str) -> None:
    """Run a phase subprocess with explicit env passthrough; fail loud."""
    log.info("[phase=%s] spawning: %s", phase, " ".join(cmd))
    subprocess.run(cmd, env={**os.environ}, check=True)


def _hf_download_with_retry(repo_id: str, filename: str, **kwargs) -> str:
    """hf_hub_download with 3-attempt backoff (transient-blip hardening)."""
    from huggingface_hub import hf_hub_download

    last_err: Exception | None = None
    for attempt, sleep_s in enumerate((0, *_HF_RETRY_SLEEPS)):
        if sleep_s:
            log.warning(
                "[phase=download] retrying %s in %ds (attempt %d): %s",
                filename,
                sleep_s,
                attempt + 1,
                last_err,
            )
            time.sleep(sleep_s)
        try:
            return hf_hub_download(repo_id=repo_id, filename=filename, **kwargs)
        except Exception as e:
            last_err = e
    raise RuntimeError(f"hf_hub_download failed after retries: {repo_id}/{filename}") from last_err


def upload_dir_fail_loud(local_dir: Path, repo_id: str, repo_type: str, path_in_repo: str) -> str:
    """ONE-commit folder upload via the shared hub helper; raise on failure."""
    from explore_persona_space.orchestrate.hub import _upload

    hub_path = _upload(
        local_path=local_dir,
        repo_id=repo_id,
        repo_type=repo_type,
        path_in_repo=path_in_repo,
    )
    if not hub_path:
        raise RuntimeError(
            f"upload of {local_dir} -> {repo_id}/{path_in_repo} returned no path — "
            "treating as FAILURE (upload-before-delete invariant); local copy preserved."
        )
    log.info("[phase=upload] %s -> %s", local_dir, hub_path)
    return hub_path


def verify_exact_hub_files(
    repo_id: str, repo_type: str, path_in_repo: str, filenames: list[str]
) -> None:
    """Assert the EXACT uploaded filenames resolve on the Hub post-upload.

    The shared ``hub._upload`` folder verification only checks the destination
    PREFIX is non-empty — once ``base_bank.npz`` (or any earlier unit's npz)
    exists under ``analysis_tensors/``, a later unit's silent upload failure
    would still "verify". Re-list the repo (paginated tree walk — no siblings
    truncation) and require every staged filename at
    ``{path_in_repo}/{name}`` BEFORE the local stage is deleted or success is
    recorded (#521 analysis-tensors rule).
    """
    from huggingface_hub import HfApi

    from explore_persona_space.orchestrate.hub import list_repo_files_complete

    api = HfApi(token=os.environ.get("HF_TOKEN"))
    on_hub = set(list_repo_files_complete(api, repo_id, repo_type=repo_type))
    missing = sorted(
        f"{path_in_repo}/{name}" for name in filenames if f"{path_in_repo}/{name}" not in on_hub
    )
    if missing:
        raise RuntimeError(
            f"exact-file upload verification FAILED: {missing} not on {repo_id} after "
            "upload — treating as FAILURE (upload-before-delete invariant); "
            "local stage preserved."
        )
    log.info(
        "[phase=upload] exact-file verification OK: %d file(s) under %s/%s",
        len(filenames),
        repo_id,
        path_in_repo,
    )


def disk_probe(root: Path, n_bytes: int = _DISK_PROBE_BYTES, phase: str = "preflight") -> None:
    """posix_fallocate headroom probe — catches the MooseFS per-pod EDQUOT
    quota that ``shutil.disk_usage`` misses (preflight.py pattern)."""
    root.mkdir(parents=True, exist_ok=True)
    probe = root / ".disk_probe"
    fd = os.open(probe, os.O_CREAT | os.O_WRONLY)
    try:
        os.posix_fallocate(fd, 0, n_bytes)
    finally:
        os.close(fd)
        probe.unlink(missing_ok=True)
    log.info("[phase=%s] disk probe OK (%.1f GB writable at %s)", phase, n_bytes / 1024**3, root)


# ── preflight ────────────────────────────────────────────────────────────────


def preflight(args, params: TitrationParams) -> dict:
    """Plan §12.1/§12.2 verifications + gate-reference availability. CPU only."""
    from huggingface_hub import list_repo_files
    from transformers import AutoTokenizer

    from explore_persona_space.experiments.leakage_dynamics_597 import (
        BASE_MODEL,
        IM_END_ID,
        MARKER_ID,
        MARKER_TEXT,
    )

    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tok.encode(MARKER_TEXT, add_special_tokens=False) != [MARKER_ID]:
        raise RuntimeError(
            f"marker {MARKER_TEXT!r} -> {tok.encode(MARKER_TEXT, add_special_tokens=False)}, "
            f"expected [{MARKER_ID}]"
        )
    if tok.encode("<|im_end|>", add_special_tokens=False) != [IM_END_ID]:
        raise RuntimeError("im_end token id drifted")
    log.info("[phase=preflight] marker/im_end token ids OK")

    # Probe rows: the parent's git copy; full-scale shape asserted BEFORE any
    # smoke limiting (the limits are applied inside shift_svd subprocesses).
    hdr = json.loads(args.probe_rows.read_text())
    if hdr.get("schema") != "i597_probe_rows_v1":
        raise RuntimeError(f"probe rows schema {hdr.get('schema')!r} != i597_probe_rows_v1")
    if hdr.get("n_contexts") != 25 or hdr.get("n_questions") != 50:
        raise RuntimeError(
            f"probe rows shape ({hdr.get('n_contexts')}, {hdr.get('n_questions')}) != (25, 50)"
        )
    log.info("[phase=preflight] probe rows OK (25 contexts x 50 questions)")

    disk_probe(args.runs_root)

    # Ladder re-listing for every REQUESTED unit (plan §12.1 verify).
    model_files = list_repo_files(HF_MODEL_REPO)
    for token in params.units:
        arm, source = parse_unit(token)
        prefix = arm_hub_prefix(arm, source)
        for step in unit_steps(arm, params):
            for fname in ("adapter_config.json", "adapter_model.safetensors"):
                want = f"{prefix}/checkpoint-{step}/{fname}"
                if want not in model_files:
                    raise RuntimeError(
                        f"preflight: {want} missing on {HF_MODEL_REPO} — "
                        "adapter-ladder reuse premise broken (plan §12.1)."
                    )
        log.info(
            "[phase=preflight] unit %s ladder OK (%d checkpoints on Hub)",
            token,
            len(unit_steps(arm, params)),
        )

    # Parent four-float reference files for every gate step.
    data_files = list_repo_files(HF_DATA_REPO, repo_type="dataset")
    refs_needed: list[tuple[str, str, int]] = []
    for token in params.units:
        arm, source = parse_unit(token)
        steps = unit_steps(arm, params)
        gate_steps = steps if params.gate_all_steps else steps[:1]
        for step in gate_steps:
            refs_needed.append((arm, source, step))
    for arm, source, step in refs_needed:
        want = f"{HF_RAW_TRAJ_ROOT}/arm{arm.upper()}/{source}/step_{step:05d}.json"
        if want not in data_files:
            raise RuntimeError(
                f"preflight: gate reference {want} missing on {HF_DATA_REPO} — "
                "the four-float reproduction gate has no reference (plan §12.5)."
            )
    log.info("[phase=preflight] %d four-float gate references resolve on Hub", len(refs_needed))
    return {"n_units": len(params.units), "n_gate_refs": len(refs_needed)}


def download_gate_refs(arm: str, source: str, steps: tuple[int, ...], dest: Path) -> Path:
    """Fetch the parent step_*.json reference files for the gate steps."""
    dest.mkdir(parents=True, exist_ok=True)
    for step in steps:
        fname = f"{HF_RAW_TRAJ_ROOT}/arm{arm.upper()}/{source}/step_{step:05d}.json"
        cached = _hf_download_with_retry(HF_DATA_REPO, fname, repo_type="dataset")
        shutil.copyfile(cached, dest / f"step_{step:05d}.json")
        log.info("[phase=download_refs_%s_%s] %s", arm, source, fname)
    return dest


def download_ladder(arm: str, source: str, steps: tuple[int, ...], dest_root: Path) -> Path:
    """Per-file download of one unit's checkpoints (parent's pattern).

    NEVER ``snapshot_download(allow_patterns=...)`` — on this >8k-file repo it
    silently returns 0 files for prefixes in the truncated siblings tail.

    Every file lands via ``local_dir=`` in a UNIT-LOCAL temp dir (then moves
    into the ladder layout) instead of the shared HF cache: cache blobs would
    survive the per-unit ladder rmtree and accumulate ~9-13 GB per unit toward
    the ~130 GB MooseFS quota (concern ``per-unit-disk-check-missing``).
    """
    from huggingface_hub import list_repo_files

    prefix = arm_hub_prefix(arm, source)
    dest = dest_root / f"{arm}_{source}"
    if all((dest / f"checkpoint-{s}" / "adapter_config.json").exists() for s in steps):
        log.info(
            "[phase=download_%s_%s] all %d checkpoints already local; skipping",
            arm,
            source,
            len(steps),
        )
        return dest

    all_files = list_repo_files(HF_MODEL_REPO)
    wanted_dirs = {f"{prefix}/checkpoint-{s}" for s in steps}
    to_fetch = [f for f in all_files if any(f.startswith(d + "/") for d in wanted_dirs)]
    if not to_fetch:
        raise RuntimeError(f"no files under {prefix} for steps {steps} on {HF_MODEL_REPO}")
    for s in steps:
        if not any(f.startswith(f"{prefix}/checkpoint-{s}/") for f in to_fetch):
            raise RuntimeError(f"checkpoint-{s} missing on the Hub under {prefix}")
    log.info(
        "[phase=download_%s_%s] fetching %d files for %d checkpoints",
        arm,
        source,
        len(to_fetch),
        len(steps),
    )
    tmp_dl = dest_root / f"_dl_{arm}_{source}"
    for i, fname in enumerate(to_fetch):
        got = _hf_download_with_retry(HF_MODEL_REPO, fname, local_dir=str(tmp_dl))
        rel = Path(fname).relative_to(prefix)
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(got, target)
        log.info("[phase=download_%s_%s] (%d/%d) %s", arm, source, i + 1, len(to_fetch), rel)
    if tmp_dl.exists():
        shutil.rmtree(tmp_dl)  # local_dir metadata (.cache/huggingface) + empty tree
    return dest


# ── floor calibration (smoke report; plan §12.12) ────────────────────────────


def floor_statistic_from_unit_npz(unit_npz: Path) -> dict:
    """Per-step median ||Δv|| / s_half at the layer-14 slot (the floor ratio)."""
    import numpy as np

    npz = np.load(unit_npz, allow_pickle=False)
    meta = json.loads(str(npz["meta"]))
    halves = npz["split_half_l14_slot"].astype(np.float64)  # (K, 2, C, H)
    deltas = npz["delta_mean"].astype(np.float64)  # (K, L, P, C, H)
    layers = meta["layers"]
    li = layers.index(14)
    pi = meta["poolings"].index("slot")
    out = {}
    for k, step in enumerate(meta["steps"]):
        dv = np.linalg.norm(deltas[k, li, pi], axis=1)  # (C,)
        s_half = np.linalg.norm(halves[k, 0] - halves[k, 1], axis=1) / 2.0  # (C,)
        ratio = dv / np.where(s_half > 0, s_half, np.nan)
        out[str(step)] = {
            "median_ratio": float(np.nanmedian(ratio)),
            "min_ratio": float(np.nanmin(ratio)),
            "max_ratio": float(np.nanmax(ratio)),
        }
    return out


# ── sentinels (poll_pipeline contract) ───────────────────────────────────────


def make_sentinel_payload(kind: str, note: dict) -> dict:
    """poll_pipeline ``_SENTINEL_REQUIRED_KEYS`` conformant payload."""
    return {
        "sentinel_schema_version": SENTINEL_SCHEMA_VERSION,
        "kind": kind,
        "version": 1,
        "task_id": 597,
        "gate": None,
        "blocks_pipeline": False,
        "by": "titration_svd_597",
        "ts": datetime.now(UTC).isoformat(),
        "note": note,
    }


def write_sentinel(logs_dir: Path, kind: str, slug: str, note: dict) -> Path:
    """Write one sentinel (epoch+pid suffix → no shard collisions)."""
    payload = make_sentinel_payload(kind, note)
    path = logs_dir / f"issue-597-{slug}-{int(time.time())}-{os.getpid()}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    return path


# ── CLI + main ───────────────────────────────────────────────────────────────


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="#597 SVD-titration pod dispatcher (follow-up "
        "svd-per-checkpoint-titration-read).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Smoke = sweep with one tiny unit (TitrationParams; PASS_UNIFIED).",
    )
    parser.add_argument(
        "--units",
        type=str,
        default=None,
        help="Comma list of a:<source>|b:<source> tokens (default: all 12; smoke: b:villain).",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=None,
        help="Defensive in-process CUDA_VISIBLE_DEVICES pin; the LAUNCHER must ALSO "
        "export CUDA_VISIBLE_DEVICES=<gpu> (import-time cuInit defeats a late pin).",
    )
    parser.add_argument(
        "--probe-rows", type=Path, default=Path("eval_results/issue_597/probe_rows.json")
    )
    parser.add_argument(
        "--slab-root",
        type=Path,
        default=Path("eval_results/issue_597/svd-per-checkpoint-titration-read"),
    )
    parser.add_argument(
        "--tensors-root", type=Path, default=Path("data/issue597_svd_titration/analysis_tensors")
    )
    parser.add_argument("--runs-root", type=Path, default=Path("/workspace/runs/issue_597_svd"))
    parser.add_argument("--logs-dir", type=Path, default=Path("/workspace/logs"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--stop-after-base",
        action="store_true",
        help="Run preflight + Phase A (base bank) + uploads, then exit cleanly "
        "(shared-phase launch before the 4 GPU shards).",
    )
    parser.add_argument(
        "--skip-base",
        action="store_true",
        help="Shard launch: require the base bank npz on disk instead of computing it.",
    )
    parser.add_argument("--skip-upload", action="store_true")
    parser.add_argument(
        "--keep-ladders", action="store_true", help="Skip the per-unit ladder rmtree."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Walk every phase WITHOUT GPU work / downloads / uploads; write the "
        "final sentinel and print [phase=done] (poll_pipeline plumbing smoke).",
    )
    return parser


def run_unit(args, params: TitrationParams, token: str, base_bank: Path) -> dict:
    """One unit: refs + ladder download → Phase B subprocess → upload → delete."""
    arm, source = parse_unit(token)
    unit = f"{arm}_{source}"
    t0 = time.time()
    # Plan §12.9 between-units disk check: re-probe writable headroom before
    # every ladder download (the per-unit download→extract→delete cycle keeps
    # the peak bounded, but a quota regression must fail loud BEFORE the next
    # ~9-13 GB fetch, not mid-download — MooseFS EDQUOT class).
    disk_probe(args.runs_root, phase=f"unit_{unit}")
    steps = unit_steps(arm, params)
    gate_steps = steps if params.gate_all_steps else steps[:1]

    refs_dir = download_gate_refs(arm, source, gate_steps, args.runs_root / "refs" / unit)
    ladder_dir = download_ladder(arm, source, steps, args.runs_root / "ladders")

    summary_path = args.slab_root / "units" / f"{unit}.json"
    cmd = [
        "uv",
        "run",
        "python",
        "-m",
        f"{PKG}.shift_svd",
        "--mode",
        "unit",
        "--arm",
        arm,
        "--source",
        source,
        "--ckpt-root",
        str(ladder_dir),
        "--steps",
        ",".join(str(s) for s in steps),
        "--probe-rows",
        str(args.probe_rows),
        "--base-bank",
        str(base_bank),
        "--out-dir",
        str(args.tensors_root),
        "--refs-dir",
        str(refs_dir),
        "--summary-out",
        str(summary_path),
        "--batch-size",
        str(args.batch_size),
    ]
    if params.limit_contexts is not None:
        cmd += ["--limit-contexts", str(params.limit_contexts)]
    if params.limit_questions is not None:
        cmd += ["--limit-questions", str(params.limit_questions)]
    _run_subprocess(cmd, phase=f"unit_{unit}")

    unit_npz = args.tensors_root / f"{unit}.npz"
    if not unit_npz.exists():
        raise RuntimeError(f"unit npz missing after subprocess: {unit_npz}")

    record: dict = {
        "unit": unit,
        "arm": arm,
        "source": source,
        "n_steps": len(steps),
        "unit_npz": str(unit_npz),
        "summary_json": str(summary_path),
        "wall_seconds": round(time.time() - t0, 1),
    }
    if params.smoke:
        record["floor_statistic_l14_slot"] = floor_statistic_from_unit_npz(unit_npz)

    # Checkpoint-per-phase upload: this unit's tensors land on HF BEFORE the
    # next unit's download (and BEFORE the ladder delete). Exact-filename
    # verification BEFORE the stage delete — the shared helper's prefix check
    # alone would pass on a stale non-empty prefix.
    if not args.skip_upload:
        stage = args.runs_root / f"_stage_{unit}"
        stage.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(unit_npz, stage / unit_npz.name)
        prefix = f"issue597_leakage_dynamics{params.hf_suffix}/analysis_tensors"
        record["hf_path"] = upload_dir_fail_loud(stage, HF_DATA_REPO, "dataset", prefix)
        verify_exact_hub_files(HF_DATA_REPO, "dataset", prefix, [unit_npz.name])
        shutil.rmtree(stage)

    if not args.keep_ladders and ladder_dir.exists():
        log.info("[phase=cleanup_%s] rmtree(%s) (MooseFS quota)", unit, ladder_dir)
        shutil.rmtree(ladder_dir)
    return record


def main(argv: list[str] | None = None) -> int:  # noqa: C901  linear dispatcher; phases read clearest inline
    args = build_arg_parser().parse_args(argv)

    if args.gpu is not None:
        env_cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
        if env_cvd is not None and env_cvd != str(args.gpu):
            raise RuntimeError(
                f"GPU pin mismatch: launcher exported CUDA_VISIBLE_DEVICES={env_cvd!r} but "
                f"--gpu {args.gpu} was passed; export the SAME index you pass to --gpu."
            )
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    logging.basicConfig(
        level=os.environ.get("EPS_LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s | %(message)s",
        stream=sys.stdout,
    )

    params = make_params(args.smoke, args.units)
    if params.smoke:
        # Smoke artifacts live under dedicated roots so a later PRODUCTION
        # launch's resume-skip can never silently reuse 3x3 smoke outputs.
        args.slab_root = args.slab_root / "smoke_run"
        args.tensors_root = args.tensors_root.parent / "smoke_run" / args.tensors_root.name
        args.runs_root = args.runs_root / "smoke_run"
    for token in params.units:
        parse_unit(token)  # validate early

    log.info(
        "[phase=dispatch_start] followup=%s smoke=%s units=%s a_steps=%d b_steps=%d "
        "limit_contexts=%s limit_questions=%s dry_run=%s",
        FOLLOWUP_LABEL,
        params.smoke,
        list(params.units),
        len(params.a_steps),
        len(params.b_steps),
        params.limit_contexts,
        params.limit_questions,
        args.dry_run,
    )
    log.info(
        "[phase=dispatch_start] UNIFIED smoke=sweep-with-one-unit: preflight/base/unit/"
        "upload subsets all derive from TitrationParams; no phase re-enumerates a "
        "registered grid on its own."
    )

    args.slab_root.mkdir(parents=True, exist_ok=True)
    args.tensors_root.mkdir(parents=True, exist_ok=True)
    args.logs_dir.mkdir(parents=True, exist_ok=True)

    per_unit: list[dict] = []
    base_bank = args.tensors_root / "base_bank.npz"

    if args.dry_run:
        # Phase walk without GPU/network: exercises the phase-token surface,
        # sentinel writer, and the [phase=done] contract end-to-end.
        log.info("[phase=preflight] DRY-RUN: skipped (no network)")
        log.info("[phase=base] DRY-RUN: would write %s", base_bank)
        for token in params.units:
            arm, source = parse_unit(token)
            log.info(
                "[phase=unit_%s_%s] DRY-RUN: %d steps, gate steps %s",
                arm,
                source,
                len(unit_steps(arm, params)),
                (unit_steps(arm, params) if params.gate_all_steps else unit_steps(arm, params)[:1]),
            )
            per_unit.append({"unit": f"{arm}_{source}", "dry_run": True})
    else:
        # ── preflight ──
        preflight_report = preflight(args, params)
        log.info("[phase=preflight] OK: %s", preflight_report)

        # ── Phase A: base side once ──
        if args.skip_base:
            if not base_bank.exists():
                raise RuntimeError(
                    f"--skip-base but {base_bank} missing — run --stop-after-base first."
                )
            log.info("[phase=base] SKIPPED (bank present: %s)", base_bank)
        else:
            # Base-side gate reference: the FIRST unit's first gate step file
            # carries logp_base for all 25x50 rows (base side is shared).
            arm0, source0 = parse_unit(params.units[0])
            gs0 = (
                unit_steps(arm0, params) if params.gate_all_steps else unit_steps(arm0, params)[:1]
            )
            refs0 = download_gate_refs(
                arm0, source0, gs0[:1], args.runs_root / "refs" / f"{arm0}_{source0}"
            )
            base_ref = refs0 / f"step_{gs0[0]:05d}.json"
            cmd = [
                "uv",
                "run",
                "python",
                "-m",
                f"{PKG}.shift_svd",
                "--mode",
                "base",
                "--probe-rows",
                str(args.probe_rows),
                "--out",
                str(base_bank),
                "--base-ref",
                str(base_ref),
                "--zero-shift-rows",
                str(params.zero_shift_rows),
                "--batch-size",
                str(args.batch_size),
            ]
            if params.limit_contexts is not None:
                cmd += ["--limit-contexts", str(params.limit_contexts)]
            if params.limit_questions is not None:
                cmd += ["--limit-questions", str(params.limit_questions)]
            _run_subprocess(cmd, phase="base")
            if not args.skip_upload:
                stage = args.runs_root / "_stage_base_bank"
                stage.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(base_bank, stage / "base_bank.npz")
                prefix = f"issue597_leakage_dynamics{params.hf_suffix}/analysis_tensors"
                upload_dir_fail_loud(stage, HF_DATA_REPO, "dataset", prefix)
                verify_exact_hub_files(HF_DATA_REPO, "dataset", prefix, ["base_bank.npz"])
                shutil.rmtree(stage)

        if args.stop_after_base:
            log.info("[phase=dispatch_complete] --stop-after-base: shared phase complete.")
            print("[phase=done]")
            return 0

        # ── unit loop ──
        for token in params.units:
            try:
                record = run_unit(args, params, token, base_bank)
                per_unit.append(record)
                sp = write_sentinel(
                    args.logs_dir,
                    "epm:progress",
                    f"unit-{record['unit']}",
                    {"event": "unit_complete", "followup_label": FOLLOWUP_LABEL, **record},
                )
                log.info("[phase=unit_%s] complete; sentinel -> %s", record["unit"], sp)
            except Exception as e:
                fp = write_sentinel(
                    args.logs_dir,
                    "epm:progress",
                    f"unit-failed-{token.replace(':', '_')}",
                    {
                        "event": "unit_failed",
                        "followup_label": FOLLOWUP_LABEL,
                        "unit": token,
                        "exception_type": type(e).__name__,
                        "exception_msg": str(e),
                    },
                )
                log.exception("[%s] unit failed; wrote %s", token, fp)
                raise

    # ── finalize: smoke report + final sentinel ──
    # Per-shard report filename: 4 parallel GPU shards each finalize; a shared
    # smoke_report.json would be last-writer-wins.
    shard_tag = f"gpu{args.gpu}" if args.gpu is not None else "all"
    smoke_report = {
        "schema": "i597_svd_smoke_report_v1",
        "smoke": params.smoke,
        "dry_run": args.dry_run,
        "shard": shard_tag,
        "units": [r.get("unit") for r in per_unit],
        "per_unit": per_unit,
        "base_bank": str(base_bank),
        "git_commit": _git_sha(),
        "hostname": socket.gethostname(),
        "ts": datetime.now(UTC).isoformat(),
    }
    if not args.dry_run and base_bank.exists():
        import numpy as np

        bank_meta = json.loads(str(np.load(base_bank, allow_pickle=False)["meta"]))
        smoke_report["zero_shift"] = bank_meta.get("zero_shift")
        smoke_report["fourfloat_gate_base"] = bank_meta.get("fourfloat_gate_base")
    report_path = args.slab_root / f"smoke_report_{shard_tag}.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(smoke_report, indent=2, ensure_ascii=False))
    log.info("[phase=finalize] smoke report -> %s", report_path)

    final = write_sentinel(
        args.logs_dir,
        "epm:results",
        "epm_results",
        {
            "issue": 597,
            "followup_label": FOLLOWUP_LABEL,
            "smoke": params.smoke,
            "dry_run": args.dry_run,
            "units_requested": list(params.units),
            "units_completed": [r.get("unit") for r in per_unit],
            "n_completed": len(per_unit),
            "n_requested": len(params.units),
            "per_unit": per_unit,
            "final_commit_sha": _git_sha(),
            "hostname": socket.gethostname(),
            "hf_hub_url": (
                f"https://huggingface.co/datasets/{HF_DATA_REPO}/tree/main/"
                f"issue597_leakage_dynamics{params.hf_suffix}/analysis_tensors"
            ),
        },
    )
    log.info("[phase=final_sentinel] %s", final)
    log.info("[phase=dispatch_complete] %d/%d units completed.", len(per_unit), len(params.units))
    print("[phase=done]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
