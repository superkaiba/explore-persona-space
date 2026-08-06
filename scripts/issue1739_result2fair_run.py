#!/usr/bin/env python3
"""Pod-side driver for the #1739 Result-2 FAIR-PROTOCOL refit (all behaviors, one pod).

Structural sibling of ``issue1739_jobd_r2aug_run.py``, sequenced for a SINGLE
pod: behaviors run smallest-labeling-tar-first (evil ~32 GB, sycophancy
~50 GB, hallucination ~70 GB), and each behavior's labeling slice is REAPED
after its scoring + upload succeed so the cumulative staged footprint stays
inside the ~130 GB MooseFS per-pod quota (the three tars together exceed it).
Shared inputs (u_store, wcrung store, pvsynth stores, E1 banks, DV JSONs) are
staged once and kept.

Per behavior: STAGE (jobd's stage_inputs: wcrung store + DVs + E1 + labeling
tar slice; plus the pvsynth capture store via the pvsynth-arms staging helper)
-> SCORE (``issue1739_result2fair_score.py`` as a subprocess, explicit rc)
-> UPLOAD (the behavior's out-root subtree to the HF data repo — per-cell
upload discipline) -> REAP the labeling slice -> per-behavior sentinel to
``/workspace/logs`` for the VM-side poller.
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
    sentinel = root / "scripts" / "issue1739_result2fair_score.py"
    if not sentinel.exists():
        raise RuntimeError(f"repo-root resolution failed: {sentinel} missing")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


_REPO_ROOT = _ensure_repo_root_on_syspath()

# smallest labeling tar first, so a quota/staging fault surfaces early
BEHAVIOR_ORDER = ("evil", "sycophancy", "hallucination")
# v2 roster (arm8/arm12/arm20 added, all MLP arms dropped) writes a SEPARATE tree
# and a SEPARATE HF prefix from the committed v1 fair run. Both must move together:
# the local out-root because the score script's `_git_tracked` guard refuses to
# overwrite the committed v1 tree, and the HF prefix so v2 results are not uploaded
# on top of v1's under the same key. Matches DEFAULT_OUT_ROOT in
# scripts/issue1739_result2fair_score.py.
HF_OUT_PREFIX = "issue1739_result2_fair_v2"
OUT_ROOT = Path("eval_results/issue_1739/result2_fair_v2")


def _log(msg: str) -> None:
    print(f"[fair-run {time.strftime('%H:%M:%S')}] {msg}", flush=True)


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


def stage_pvsynth(args, token: str) -> None:
    from scripts.issue1739_pvsynth_arms_run import stage_shared as pv_stage_shared

    pv_stage_shared(_pv_ns(args), token)


def score_cmd(args, behavior: str) -> list[str]:
    return [
        sys.executable,
        str(_REPO_ROOT / "scripts" / "issue1739_result2fair_score.py"),
        "--behaviors",
        behavior,
        "--variant",
        "context_end",
        "--store-root",
        str(args.store_root),
        "--main-root",
        str(args.main_root),
        "--tensors-root",
        str(args.tensors_root),
        "--out-root",
        str(OUT_ROOT),
        "--device",
        args.device,
    ]


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

    Fail-loud rmtree — a failed reap must crash here, not later at the next
    behavior's staging canary. Never touches u_store / wcrung / pvsynth / E1.
    """
    dest = args.store_root / f"{behavior}_labeling"
    if not dest.exists():
        _log(f"[phase=reap {behavior}] labeling slice absent, nothing to reap")
        return
    shutil.rmtree(dest)
    _log(f"[phase=reap {behavior}] reaped {dest}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--behaviors",
        nargs="+",
        default=list(BEHAVIOR_ORDER),
        choices=list(BEHAVIOR_ORDER),
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
    ap.add_argument("--no-reap", action="store_true")
    ap.add_argument("--sentinel-dir", type=Path, default=Path("/workspace/logs"))
    ap.add_argument("--import-check", action="store_true")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.import_check:
        from explore_persona_space.orchestrate import hub  # noqa: F401
        from explore_persona_space.orchestrate.env import load_dotenv  # noqa: F401
        from scripts.issue1739_jobd_r2aug_run import stage_inputs  # noqa: F401
        from scripts.issue1739_pvsynth_arms_run import stage_shared  # noqa: F401

        _log("import-check OK")
        sys.stdout.flush()
        sys.stderr.flush()
        sys.exit(0)

    from explore_persona_space.orchestrate.env import load_dotenv
    from scripts.issue1739_jobd_r2aug_run import stage_inputs

    load_dotenv()
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or ""
    t0 = time.time()
    stage_pvsynth(args, token)

    behaviors = [b for b in BEHAVIOR_ORDER if b in args.behaviors]
    results: dict[str, dict] = {}
    overall_rc = 0
    for behavior in behaviors:
        t_b = time.time()
        _log(f"=== {behavior}: stage -> score -> upload -> reap ===")
        stage_inputs(_jobd_ns(args, behavior), token)
        cmd = score_cmd(args, behavior)
        _log(f"[phase=score {behavior}] {' '.join(cmd[1:])}")
        proc = subprocess.run(cmd, cwd=str(_REPO_ROOT), check=False)
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
            "leg": "result2_fair_v2",
            "behavior": behavior,
            **results[behavior],
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        args.sentinel_dir.mkdir(parents=True, exist_ok=True)
        path = args.sentinel_dir / f"issue-1739-result2fair-{behavior}.json"
        path.write_text(json.dumps(sentinel, indent=1))
        _log(f"[phase=done {behavior}] sentinel -> {path}")

    summary = {
        "leg": "result2_fair_v2",
        "behaviors": results,
        "overall_rc": overall_rc,
        "wall_s": round(time.time() - t0, 1),
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    args.sentinel_dir.mkdir(parents=True, exist_ok=True)
    path = args.sentinel_dir / "issue-1739-result2fair-all.json"
    path.write_text(json.dumps(summary, indent=1))
    _log(f"[phase=done] sentinel -> {path} (overall_rc={overall_rc})")
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(overall_rc)


if __name__ == "__main__":
    main()
