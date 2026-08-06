#!/usr/bin/env python3
"""Pod-side driver for the #1739 r2v2 P-A/P-B fits (CPU pod, one behavior chain).

Structural sibling of ``issue1739_result2fair_run.py`` with the r2v2 deltas:
per behavior STAGE (jobd stage_inputs: wcrung store + DVs + E1 + labeling tar
slice; pvsynth via the pvsynth-arms helper; PLUS the behavior's NEW OOD
stores/DV mirrored from ``issue1739_ctxmap/`` via ``hub.stage_hub_prefix``)
-> SCORE (``issue1739_r2v2_score.py`` subprocess, explicit rc) -> UPLOAD
(out-root subtree -> HF data repo) -> REAP the labeling slice -> per-behavior
sentinel to /workspace/logs for the VM-side poller.

Behaviors run SEQUENTIALLY within one pod (the merged fp64 tables peak
~35-55 GB per behavior — two concurrent behaviors do not fit a 128 GB
cpu-bigmem box); cross-behavior concurrency = one CPU pod per behavior
(CPU pods may run in parallel; each invokes this driver with its own
``--behaviors``).
"""

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
    sentinel = root / "scripts" / "issue1739_r2v2_score.py"
    if not sentinel.exists():
        raise RuntimeError(f"repo-root resolution failed: {sentinel} missing")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


_REPO_ROOT = _ensure_repo_root_on_syspath()

BEHAVIOR_ORDER = ("evil", "sycophancy", "hallucination")
HF_OUT_PREFIX = "issue1739_r2v2_fits"
OUT_ROOT = Path("eval_results/issue_1739/r2v2_fits")
CTXMAP_PREFIX = "issue1739_ctxmap"
# HF prefixes staged per behavior (verbatim mirrors under --ood-mirror-root).
OOD_STAGE_PREFIXES = {
    "evil": (f"{CTXMAP_PREFIX}/evil_ood_full/store",),
    "sycophancy": (
        f"{CTXMAP_PREFIX}/syco_ood/store",
        f"{CTXMAP_PREFIX}/syco_ood/dv_dataset",
    ),
    "hallucination": (),
}


def _log(msg: str) -> None:
    print(f"[r2v2-run {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _jobd_ns(args, behavior: str) -> argparse.Namespace:
    """Namespace shaped for issue1739_jobd_r2aug_run.stage_inputs."""
    return argparse.Namespace(
        behavior=behavior,
        store_root=args.store_root,
        wcrung_root=args.wcrung_root,
        tensors_root=args.tensors_root,
        revision=args.revision,
        stage_workers=args.stage_workers,
    )


def _pv_ns(args) -> argparse.Namespace:
    """Namespace shaped for issue1739_pvsynth_arms_run.stage_shared."""
    return argparse.Namespace(
        store_root=args.store_root,
        behaviors=list(args.behaviors),
        train_dv_root=args.store_root / "train_dv",
    )


def stage_ood(args, behavior: str, token: str) -> None:
    """Mirror the behavior's OOD store/DV prefixes under the ood mirror root.

    ``stage_hub_prefix`` files land at ``<mirror_root>/<repo-relative path>``
    (verbatim prefix mirror — the score script's --ood-store-root consumes
    ``<mirror_root>/issue1739_ctxmap/...``). Idempotent per file: a partial
    mirror self-heals on re-run.
    """
    from explore_persona_space.orchestrate import hub

    for prefix in OOD_STAGE_PREFIXES.get(behavior, ()):
        # cheap completeness probe: skip a prefix whose remote file SET is
        # already fully present locally (stage_hub_prefix re-lists otherwise)
        t0 = time.time()
        files = hub.stage_hub_prefix(
            hub.DEFAULT_DATASET_REPO,
            prefix,
            args.ood_mirror_root,
            repo_type="dataset",
            token=token or None,
            max_workers=min(args.stage_workers, 6),
        )
        _log(
            f"[phase=stage_ood {behavior}] {prefix}: {len(files)} files in "
            f"{time.time() - t0:.0f}s -> {args.ood_mirror_root / prefix}"
        )


def score_cmd(args, behavior: str) -> list[str]:
    cmd = [
        sys.executable,
        str(_REPO_ROOT / "scripts" / "issue1739_r2v2_score.py"),
        "--behaviors",
        behavior,
        "--variant",
        "context_end",
        "--protocols",
        args.protocols,
        "--store-root",
        str(args.store_root),
        "--main-root",
        str(args.main_root),
        "--tensors-root",
        str(args.tensors_root),
        "--out-root",
        str(OUT_ROOT),
        "--ood-store-root",
        str(args.ood_mirror_root / CTXMAP_PREFIX),
        "--device",
        args.device,
        "--ood-dv-max-null-frac",
        str(args.ood_dv_max_null_frac),
    ]
    if args.pb_holdouts:
        cmd += ["--pb-holdouts", *args.pb_holdouts]
    return cmd


def upload_behavior(args, behavior: str) -> str:
    from explore_persona_space.orchestrate import hub

    local = OUT_ROOT / behavior
    if not local.exists():
        raise FileNotFoundError(f"nothing to upload: {local} absent")
    url = hub._upload(
        local,
        hub.DEFAULT_DATASET_REPO,
        "dataset",
        f"{HF_OUT_PREFIX}/{behavior}",
        raise_on_error=True,
    )
    _log(f"[phase=upload {behavior}] -> {HF_OUT_PREFIX}/{behavior} ({url})")
    return url


def reap_labeling_slice(args, behavior: str) -> None:
    """Free the behavior's staged labeling slice (re-downloadable from HF).

    Fail-loud rmtree. Never touches u_store / wcrung / pvsynth / E1 / the
    OOD mirror (the OOD stores are small and shared across re-runs).
    """
    dest = args.store_root / f"{behavior}_labeling"
    if not dest.exists():
        _log(f"[phase=reap {behavior}] labeling slice absent, nothing to reap")
        return
    shutil.rmtree(dest)
    _log(f"[phase=reap {behavior}] reaped {dest}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--behaviors", nargs="+", default=["sycophancy"], choices=list(BEHAVIOR_ORDER))
    ap.add_argument("--protocols", default="AB", choices=["A", "B", "AB"])
    ap.add_argument("--pb-holdouts", nargs="+", default=None)
    ap.add_argument("--store-root", type=Path, default=Path("data/issue_1739/hf_dl"))
    ap.add_argument("--main-root", type=Path, default=Path("eval_results/issue_1739"))
    ap.add_argument(
        "--wcrung-root", type=Path, default=Path("eval_results/issue_1739/wildchat_rung")
    )
    ap.add_argument("--tensors-root", type=Path, default=Path("analysis_tensors/issue_1739"))
    ap.add_argument(
        "--ood-mirror-root",
        type=Path,
        default=None,
        help="mirror root for OOD prefixes (default: <store-root>/ood_mirror)",
    )
    ap.add_argument("--ood-dv-max-null-frac", type=float, default=0.05)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--revision", default="main")
    ap.add_argument("--stage-workers", type=int, default=12)
    ap.add_argument("--skip-upload", action="store_true")
    ap.add_argument("--no-reap", action="store_true")
    ap.add_argument(
        "--stage-only",
        action="store_true",
        help="stage every input then exit 0 (pilot/verification aid)",
    )
    ap.add_argument("--sentinel-dir", type=Path, default=Path("/workspace/logs"))
    ap.add_argument("--import-check", action="store_true")
    args = ap.parse_args(argv)
    if args.ood_mirror_root is None:
        args.ood_mirror_root = args.store_root / "ood_mirror"
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.import_check:
        from explore_persona_space.orchestrate import hub  # noqa: F401
        from explore_persona_space.orchestrate.env import load_dotenv  # noqa: F401
        from scripts.issue1739_jobd_r2aug_run import stage_inputs  # noqa: F401
        from scripts.issue1739_pvsynth_arms_run import stage_shared  # noqa: F401

        assert callable(hub.stage_hub_prefix)
        _log("import-check OK")
        sys.stdout.flush()
        sys.stderr.flush()
        sys.exit(0)

    from explore_persona_space.orchestrate.env import load_dotenv
    from scripts.issue1739_jobd_r2aug_run import stage_inputs
    from scripts.issue1739_pvsynth_arms_run import stage_shared as pv_stage_shared

    load_dotenv()
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or ""
    t0 = time.time()
    pv_stage_shared(_pv_ns(args), token)

    behaviors = [b for b in BEHAVIOR_ORDER if b in args.behaviors]
    results: dict[str, dict] = {}
    overall_rc = 0
    for behavior in behaviors:
        t_b = time.time()
        _log(f"=== {behavior}: stage -> score -> upload -> reap ===")
        stage_inputs(_jobd_ns(args, behavior), token)
        stage_ood(args, behavior, token)
        if args.stage_only:
            _log(f"[phase=stage_only {behavior}] staging complete, skipping score")
            results[behavior] = {"score_rc": None, "staged_only": True}
            continue
        cmd = score_cmd(args, behavior)
        _log(f"[phase=score {behavior}] {' '.join(cmd[1:])}")
        proc = subprocess.run(cmd, cwd=str(_REPO_ROOT), check=False, env={**os.environ})
        _log(f"[phase=score {behavior}] rc={proc.returncode} in {time.time() - t_b:.0f}s")
        url = None
        if proc.returncode == 0 and not args.skip_upload:
            url = upload_behavior(args, behavior)
        if proc.returncode == 0 and not args.no_reap:
            reap_labeling_slice(args, behavior)
        results[behavior] = {
            "score_rc": proc.returncode,
            "uploaded": bool(url),
            "upload_url": url,
            "wall_s": round(time.time() - t_b, 1),
        }
        overall_rc = overall_rc or proc.returncode
        sentinel = {
            "leg": "r2v2_fits",
            "behavior": behavior,
            **results[behavior],
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        args.sentinel_dir.mkdir(parents=True, exist_ok=True)
        path = args.sentinel_dir / f"issue-1739-r2v2fits-{behavior}.json"
        path.write_text(json.dumps(sentinel, indent=1))
        _log(f"[phase=done {behavior}] sentinel -> {path}")

    summary = {
        "leg": "r2v2_fits",
        "behaviors": results,
        "overall_rc": overall_rc,
        "wall_s": round(time.time() - t0, 1),
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    args.sentinel_dir.mkdir(parents=True, exist_ok=True)
    path = args.sentinel_dir / "issue-1739-r2v2fits-all.json"
    path.write_text(json.dumps(summary, indent=1))
    _log(f"[phase=done] sentinel -> {path} (overall_rc={overall_rc})")
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(overall_rc)


if __name__ == "__main__":
    main()
