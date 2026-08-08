#!/usr/bin/env python3
"""Instance-side driver for one #1739 jobd/r2aug unit (one behavior, one job).

Structural sibling of ``issue1739_wcrung_arms_run.py``, narrowed to the
one-(behavior x job)-per-pod fan-out shape (user directive 2026-08-05:
maximize parallelism, shard by behavior x map condition):

* STAGE: the shared wildchat-rung inputs (capture store + wcrung DVs + train
  DVs, via the wcrung driver's own ``stage_shared``), the behavior's E1
  extraction store when its r_B bank is absent, and the behavior's train
  labeling tar via MEMBER-SELECTIVE STREAMING (``stream_slice`` — no tar ever
  lands on disk, peak disk = the extracted slice; this is why the brief's
  tmpfs staging fallback for the 69.9 GB hallucination tar is not needed on
  this path). Every staging phase is preceded by a REAL write canary
  (posix_fallocate + fsync + unlink) on the staging filesystem — never a
  bare ``df`` read (MooseFS per-pod quota is invisible to statvfs).
* SCORE: ``issue1739_jobd_r2aug.py`` as a subprocess (explicit rc).
* UPLOAD: the unit's out-root subtree to the HF data repo the moment scoring
  finishes (per-cell upload discipline), then a done sentinel to
  ``/workspace/logs`` for the VM-side poller.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def _ensure_repo_root_on_syspath() -> Path:
    root = Path(__file__).resolve().parents[1]
    sentinel = root / "scripts" / "issue1739_jobd_r2aug.py"
    if not sentinel.exists():
        raise RuntimeError(f"repo-root resolution failed: {sentinel} missing")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


_REPO_ROOT = _ensure_repo_root_on_syspath()

KINDS = ("prefix_end", "context_end", "t1")
HF_PREFIX = "issue1739_ctxmap"
STAGE_MANIFEST = "slice_manifest.json"
HF_OUT_PREFIX = {
    "jobd": "issue1739_judged_generic_ablation",
    "r2aug": "issue1739_result2_trait_aug",
}
OUT_ROOT = {
    "jobd": Path("eval_results/issue_1739/judged_generic_ablation"),
    "r2aug": Path("eval_results/issue_1739/result2_trait_aug"),
}


def _log(msg: str) -> None:
    print(f"[jobd-r2aug-run {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def write_canary(root: Path, gib: float = 1.0) -> None:
    """REAL write probe on ``root``'s filesystem: allocate, fsync, remove.

    ``df``/statvfs report share-level free space on MooseFS and cannot see the
    per-pod quota (the 2026-08-04 EDQUOT incident's diagnostic); an actual
    allocation is the only trustworthy probe. Fail-loud: any OSError (EDQUOT,
    ENOSPC) aborts the run before staging starts.
    """
    root.mkdir(parents=True, exist_ok=True)
    probe = root / f".canary.{os.getpid()}"
    n = int(gib * (1 << 30))
    fd = os.open(probe, os.O_CREAT | os.O_WRONLY, 0o600)
    try:
        os.posix_fallocate(fd, 0, n)
        os.fsync(fd)
    finally:
        os.close(fd)
        probe.unlink(missing_ok=True)
    _log(f"[canary] OK: {gib:.1f} GiB allocatable at {root}")


def _staging_ns(args) -> argparse.Namespace:
    """Namespace shaped for the wcrung driver's staging helpers."""
    return argparse.Namespace(
        store_root=args.store_root,
        out_root=args.wcrung_root,
        train_dv_root=args.store_root / "train_dv",
        tensors_root=args.tensors_root,
        behaviors=[args.behavior],
        layers=list(range(28)),
        revision=args.revision,
        stage_workers=args.stage_workers,
    )


def stage_inputs(args, token: str) -> None:
    from scripts.issue1739_map963k_slice import stream_slice
    from scripts.issue1739_wcrung_arms_run import (
        stage_extraction,
        stage_shared,
        staged_slice_covers,
    )

    ns = _staging_ns(args)
    write_canary(args.store_root, gib=2.0)
    _log("[phase=stage_shared] wcrung store + DVs")
    stage_shared(ns, token)
    stage_extraction(args.behavior, ns, token)

    dest = args.store_root / f"{args.behavior}_labeling"
    covered, why = staged_slice_covers(dest, kinds=KINDS, layers=ns.layers)
    if covered:
        _log(f"[phase=stage] {args.behavior}: slice manifest covers this regime, skip")
        return
    if why:
        _log(f"[phase=stage] {args.behavior}: re-staging — {why}")
    write_canary(args.store_root, gib=2.0)
    _log(f"[phase=stage] {args.behavior}: streaming labeling tar (member-selective)")
    m = stream_slice(
        args.behavior,
        dest,
        revision=args.revision,
        kinds=KINDS,
        layers=tuple(ns.layers),
        token=token,
        workers=args.stage_workers,
    )
    _log(
        f"[phase=stage] {args.behavior}: DONE written={m['kept_bytes'] / 1e9:.1f} GB "
        f"fetched={m['bytes_fetched'] / 1e9:.1f} GB in {m['elapsed_s']:.0f}s "
        f"({m['mb_per_s']:.1f} MB/s)"
    )


def score_cmd(args) -> list[str]:
    cmd = [
        sys.executable,
        str(_REPO_ROOT / "scripts" / "issue1739_jobd_r2aug.py"),
        "--behaviors",
        args.behavior,
        "--modes",
        *args.modes,
        "--variants",
        "context_end",
        "--store-root",
        str(args.store_root),
        "--main-root",
        str(args.main_root),
        "--tensors-root",
        str(args.tensors_root),
        "--device",
        args.device,
    ]
    if "r2aug" in args.modes:
        cmd += ["--map-conditions", *args.map_conditions]
    return cmd


def upload_outputs(args) -> list[str]:
    from explore_persona_space.orchestrate import hub

    urls = []
    for mode in args.modes:
        local = OUT_ROOT[mode] / args.behavior
        if not local.exists():
            raise FileNotFoundError(f"nothing to upload for {mode}: {local} absent")
        url = hub._upload(
            local,
            hub.DEFAULT_DATASET_REPO,
            "dataset",
            f"{HF_OUT_PREFIX[mode]}/{args.behavior}",
            raise_on_error=True,
        )
        _log(f"[phase=upload mode={mode}] -> {HF_OUT_PREFIX[mode]}/{args.behavior} ({url})")
        urls.append(url)
    return urls


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--behavior", required=True, choices=("evil", "sycophancy", "hallucination"))
    ap.add_argument("--modes", nargs="+", default=["jobd"], choices=["jobd", "r2aug"])
    ap.add_argument(
        "--map-conditions",
        nargs="+",
        default=["swap", "add"],
        choices=["swap", "add", "generic_matched"],
    )
    ap.add_argument("--store-root", type=Path, default=Path("data/issue_1739/hf_dl"))
    ap.add_argument("--main-root", type=Path, default=Path("eval_results/issue_1739"))
    ap.add_argument(
        "--wcrung-root", type=Path, default=Path("eval_results/issue_1739/wildchat_rung")
    )
    ap.add_argument("--tensors-root", type=Path, default=Path("analysis_tensors/issue_1739"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--revision", default="main")
    ap.add_argument("--stage-workers", type=int, default=12)
    ap.add_argument("--skip-upload", action="store_true")
    ap.add_argument("--stage-only", action="store_true")
    ap.add_argument("--sentinel-dir", type=Path, default=Path("/workspace/logs"))
    ap.add_argument("--import-check", action="store_true")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.import_check:
        from explore_persona_space.orchestrate import hub  # noqa: F401
        from explore_persona_space.orchestrate.env import load_dotenv  # noqa: F401
        from scripts.issue1739_map963k_slice import stream_slice  # noqa: F401
        from scripts.issue1739_wcrung_arms_run import stage_shared  # noqa: F401

        _log("import-check OK")
        sys.stdout.flush()
        sys.stderr.flush()
        sys.exit(0)

    from explore_persona_space.orchestrate.env import load_dotenv

    load_dotenv()
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or ""
    t0 = time.time()
    stage_inputs(args, token)
    if args.stage_only:
        _log("stage-only: done")
        sys.stdout.flush()
        sys.stderr.flush()
        sys.exit(0)

    cmd = score_cmd(args)
    _log(f"[phase=score] {' '.join(cmd[1:])}")
    t_score = time.time()
    proc = subprocess.run(cmd, cwd=str(_REPO_ROOT), check=False)
    _log(f"[phase=score] rc={proc.returncode} in {time.time() - t_score:.0f}s")

    uploaded: list[str] = []
    if proc.returncode == 0 and not args.skip_upload:
        uploaded = upload_outputs(args)

    sentinel = {
        "leg": "jobd_r2aug",
        "behavior": args.behavior,
        "modes": list(args.modes),
        "map_conditions": list(args.map_conditions) if "r2aug" in args.modes else None,
        "score_rc": proc.returncode,
        "uploaded": bool(uploaded),
        "upload_urls": uploaded,
        "wall_s": round(time.time() - t0, 1),
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    args.sentinel_dir.mkdir(parents=True, exist_ok=True)
    slug = f"issue-1739-jobdr2aug-{args.behavior}-{'-'.join(args.modes)}"
    (args.sentinel_dir / f"{slug}.json").write_text(json.dumps(sentinel, indent=1))
    _log(f"[phase=done] sentinel -> {args.sentinel_dir / (slug + '.json')}")
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(proc.returncode)


if __name__ == "__main__":
    main()
