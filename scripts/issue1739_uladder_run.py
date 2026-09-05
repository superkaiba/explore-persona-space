#!/usr/bin/env python3
"""Pod-side stage, score, upload, verify, and reap driver for issue 1739."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


def _ensure_repo_root_on_syspath() -> Path:
    root = Path(__file__).resolve().parents[1]
    if not (root / "scripts" / "issue1739_uladder_score.py").exists():
        raise RuntimeError(f"repository root resolution failed: {root}")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


REPO_ROOT = _ensure_repo_root_on_syspath()
BEHAVIORS = ("evil", "sycophancy", "hallucination")
HF_OUT_PREFIX = "issue1739_uladder"
HF_PILOT_PREFIX = "issue1739_uladder_pilot"
DEFAULT_OUT_ROOT = Path("eval_results/issue_1739/uladder")
DEFAULT_PILOT_OUT_ROOT = Path("eval_results/issue_1739/uladder_pilot")
DEFAULT_STORE_ROOT = Path("data/issue_1739/hf_dl")
DEFAULT_MAIN_ROOT = Path("eval_results/issue_1739")
DEFAULT_WCRUNG_ROOT = Path("eval_results/issue_1739/wildchat_rung")
DEFAULT_TENSORS_ROOT = Path("analysis_tensors/issue_1739")


def _log(message: str) -> None:
    print(f"[uladder-run {time.strftime('%H:%M:%S')}] {message}", flush=True)


def _hf_prefix(args: argparse.Namespace) -> str:
    """Resolve the durable output prefix without conflating separate pilots."""
    if args.hf_prefix is not None:
        return args.hf_prefix
    return HF_PILOT_PREFIX if args.pilot else HF_OUT_PREFIX


def _atomic_json(path: Path, payload: dict) -> None:
    from explore_persona_space.atomic_io import atomic_replace

    path.parent.mkdir(parents=True, exist_ok=True)
    with atomic_replace(path) as tmp:
        tmp.write_text(json.dumps(payload, indent=1, sort_keys=True))


def _jobd_namespace(args: argparse.Namespace, behavior: str) -> argparse.Namespace:
    return argparse.Namespace(
        behavior=behavior,
        store_root=args.store_root,
        wcrung_root=args.wcrung_root,
        tensors_root=args.tensors_root,
        revision=args.revision,
        stage_workers=args.stage_workers,
        materialize_labeling_tars=args.materialize_labeling_tars,
    )


def _r2_namespace(args: argparse.Namespace, behavior: str) -> argparse.Namespace:
    return argparse.Namespace(
        behaviors=[behavior],
        store_root=args.store_root,
        wcrung_root=args.wcrung_root,
        tensors_root=args.tensors_root,
        main_root=args.main_root,
        revision=args.revision,
        stage_workers=args.stage_workers,
        ood_mirror_root=args.ood_mirror_root,
    )


def _filesystem_record(path: Path) -> dict:
    path.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(path)
    mount = subprocess.run(
        ["df", "-P", str(path)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip().splitlines()[-1].split()
    return {
        "path": str(path.resolve()),
        "filesystem": mount[0],
        "mountpoint": mount[-1],
        "free_gib": usage.free / 2**30,
        "total_gib": usage.total / 2**30,
    }


def _assert_headroom(args: argparse.Namespace) -> dict:
    from scripts.issue1739_jobd_r2aug_run import write_canary

    record = _filesystem_record(args.store_root)
    if record["free_gib"] < args.min_free_gib:
        raise RuntimeError(
            f"staging filesystem has {record['free_gib']:.1f} GiB free, below "
            f"--min-free-gib {args.min_free_gib:.1f}: {record}"
        )
    write_canary(args.store_root, gib=args.canary_gib)
    _log(f"[phase=preflight] staging filesystem {record}")
    return record


def stage_inputs(args: argparse.Namespace, behavior: str, token: str) -> None:
    from scripts.issue1739_jobd_r2aug_run import stage_inputs as stage_base
    from scripts.issue1739_r2v2_run import stage_ood

    _log(f"[phase=stage] {behavior}: base tables")
    stage_base(_jobd_namespace(args, behavior), token)
    _log(f"[phase=stage] {behavior}: wide OOD tables")
    stage_ood(_r2_namespace(args, behavior), behavior, token)


def score_cmd(args: argparse.Namespace, behavior: str, seed: int) -> list[str]:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "issue1739_uladder_score.py"),
        "--behaviors",
        behavior,
        "--seed",
        str(seed),
        "--device",
        args.device,
        "--out-root",
        str(args.out_root),
        "--main-root",
        str(args.main_root),
        "--tensors-root",
        str(args.tensors_root),
        "--store-root",
        str(args.store_root),
        "--ood-store-root",
        str(args.ood_mirror_root / "issue1739_ctxmap"),
    ]
    if args.pilot:
        cmd.append("--pilot")
    return cmd


def upload_seed(args: argparse.Namespace, behavior: str, seed: int) -> dict:
    from huggingface_hub import HfApi

    from explore_persona_space.orchestrate import hub

    local = args.out_root / behavior / f"seed{seed}"
    summary = local / "all_arms_spearman.json"
    if not summary.exists():
        raise FileNotFoundError(f"seed output missing: {summary}")
    relative_files = sorted(
        str(path.relative_to(local)) for path in local.rglob("*") if path.is_file()
    )
    if not relative_files:
        raise RuntimeError(f"seed output empty: {local}")
    root_prefix = _hf_prefix(args)
    prefix = f"{root_prefix}/{behavior}/seed{seed}"
    url = hub._upload(
        local,
        hub.DEFAULT_DATASET_REPO,
        "dataset",
        prefix,
        raise_on_error=True,
    )
    expected = [f"{prefix}/{rel}" for rel in relative_files]
    missing = hub.verify_repo_paths_uploaded(
        HfApi(),
        hub.DEFAULT_DATASET_REPO,
        expected,
        path_in_repo=prefix,
        repo_type="dataset",
    )
    if missing:
        raise RuntimeError(f"{len(missing)} uploaded files missing under {prefix}: {missing[:5]}")
    _log(
        f"[phase=upload] {behavior}/seed{seed}: verified {len(expected)} files at {url}"
    )
    return {"url": url, "prefix": prefix, "files": expected}


def reap_behavior_slice(args: argparse.Namespace, behavior: str) -> None:
    target = args.store_root / f"{behavior}_labeling"
    if not target.exists():
        _log(f"[phase=reap] {behavior}: labeling slice absent")
        return
    shutil.rmtree(target)
    _log(f"[phase=reap] {behavior}: removed re-downloadable labeling slice {target}")


def _sentinel(
    args: argparse.Namespace,
    *,
    rc: int,
    started: float,
    filesystems: dict,
    results: dict,
) -> Path:
    kind = "epm:smoke-result" if args.pilot or args.stage_only else "epm:progress"
    mode = "stage_only" if args.stage_only else ("pilot" if args.pilot else "production")
    body = {
        "sentinel_schema_version": 1,
        "kind": kind,
        "version": 1,
        "task_id": 1739,
        "by": "issue1739-uladder-run",
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "gate": "pilot" if args.pilot else "results",
        "blocks_pipeline": bool(rc),
        "note": (
            f"issue 1739 U-ladder {mode} {'passed' if rc == 0 else 'failed'}; "
            f"behaviors={list(results)} seeds={args.seeds} rc={rc}"
        ),
        "payload": {
            "mode": mode,
            "rc": rc,
            "behaviors": results,
            "wall_s": round(time.time() - started, 3),
            "filesystems": filesystems,
            "hf_prefix": _hf_prefix(args),
            "out_root": str(args.out_root),
        },
    }
    name = f"issue-1739-uladder-{mode}-{int(time.time())}.json"
    path = args.sentinel_dir / name
    _atomic_json(path, body)
    return path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--behaviors", nargs="+", choices=BEHAVIORS, default=list(BEHAVIORS))
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--store-root", type=Path, default=DEFAULT_STORE_ROOT)
    ap.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    ap.add_argument("--main-root", type=Path, default=DEFAULT_MAIN_ROOT)
    ap.add_argument("--wcrung-root", type=Path, default=DEFAULT_WCRUNG_ROOT)
    ap.add_argument("--tensors-root", type=Path, default=DEFAULT_TENSORS_ROOT)
    ap.add_argument("--ood-mirror-root", type=Path, default=None)
    ap.add_argument("--revision", default="main")
    ap.add_argument("--stage-workers", type=int, default=12)
    ap.add_argument(
        "--materialize-labeling-tars",
        action="store_true",
        help="use hf_transfer whole-tar staging (requires roughly 2x tar-size free space)",
    )
    ap.add_argument("--stage-timeout-s", type=int, default=28800)
    ap.add_argument("--min-free-gib", type=float, default=115.0)
    ap.add_argument("--canary-gib", type=float, default=2.0)
    ap.add_argument("--skip-upload", action="store_true")
    ap.add_argument(
        "--hf-prefix",
        default=None,
        help="Override the durable HF output prefix (for distinct rerun/pilot provenance).",
    )
    ap.add_argument("--no-reap", action="store_true")
    ap.add_argument("--stage-only", action="store_true")
    ap.add_argument("--sentinel-dir", type=Path, default=Path("/workspace/logs"))
    ap.add_argument("--import-check", action="store_true")
    args = ap.parse_args(argv)
    if len(set(args.seeds)) != len(args.seeds):
        ap.error("--seeds contains duplicates")
    if args.hf_prefix is not None:
        args.hf_prefix = args.hf_prefix.strip("/")
        if not args.hf_prefix or any(part in {".", ".."} for part in args.hf_prefix.split("/")):
            ap.error("--hf-prefix must be a non-empty repository-relative path")
    if args.pilot:
        args.behaviors = args.behaviors[:1]
        args.seeds = args.seeds[:1]
        if args.out_root == DEFAULT_OUT_ROOT:
            args.out_root = DEFAULT_PILOT_OUT_ROOT
    if args.stage_timeout_s <= 0 or args.min_free_gib <= 0 or args.canary_gib <= 0:
        ap.error("timeout and headroom values must be positive")
    if args.ood_mirror_root is None:
        args.ood_mirror_root = args.store_root / "ood_mirror"
    return args


def _import_check(args: argparse.Namespace) -> None:
    from explore_persona_space.orchestrate import hub
    from scripts.issue1739_jobd_r2aug_run import stage_inputs as stage_base
    from scripts.issue1739_r2v2_run import stage_ood
    from scripts.issue1739_uladder_score import parse_args as parse_score

    assert callable(stage_base) and callable(stage_ood)
    assert callable(hub.verify_repo_paths_uploaded)
    bound = parse_score(score_cmd(args, "evil", 0)[2:])
    assert bound.behaviors == ["evil"] and bound.seed == 0
    print("[uladder-run] import-check OK")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.import_check:
        _import_check(args)
        return 0
    from explore_persona_space.orchestrate.env import load_dotenv

    load_dotenv()
    os.environ["EPM_HF_STAGE_TIMEOUT_S"] = str(args.stage_timeout_s)
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or ""
    started = time.time()
    filesystem = _assert_headroom(args)
    results = {}
    rc = 0
    try:
        for behavior in args.behaviors:
            stage_inputs(args, behavior, token)
            behavior_result = {"seeds": {}}
            if args.stage_only:
                behavior_result["staged_only"] = True
                results[behavior] = behavior_result
                continue
            for seed in args.seeds:
                cmd = score_cmd(args, behavior, seed)
                _log(f"[phase=score] {behavior}/seed{seed}: {' '.join(cmd[1:])}")
                seed_started = time.time()
                proc = subprocess.run(cmd, cwd=REPO_ROOT, check=False)
                seed_result = {
                    "rc": proc.returncode,
                    "wall_s": round(time.time() - seed_started, 3),
                }
                if proc.returncode == 0 and not args.skip_upload:
                    seed_result["upload"] = upload_seed(args, behavior, seed)
                behavior_result["seeds"][str(seed)] = seed_result
                if proc.returncode != 0:
                    rc = proc.returncode
                    break
            results[behavior] = behavior_result
            if rc:
                break
            if not args.no_reap and not args.stage_only and not args.skip_upload:
                reap_behavior_slice(args, behavior)
    except Exception as exc:
        rc = rc or 1
        results["exception"] = f"{type(exc).__name__}: {exc}"
        _log(f"FAILED: {results['exception']}")
    sentinel = _sentinel(
        args,
        rc=rc,
        started=started,
        filesystems={"staging": filesystem},
        results=results,
    )
    _log(f"[phase=done] sentinel={sentinel} rc={rc}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
