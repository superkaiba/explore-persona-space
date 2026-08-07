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
FACT_HF_OUT_PREFIX = "issue1739_r2v2_factorial"
FACT_OUT_ROOT = Path("eval_results/issue_1739/r2v2_factorial")
# λ grid-edge check (task #16): base-grid GCV pinned λ=1000 (the grid MAX) on
# every hallucination fit — re-fit under a 3-decades-wider grid, own out root.
WIDE_HF_OUT_PREFIX = "issue1739_r2v2_fits_widegrid"
WIDE_OUT_ROOT = Path("eval_results/issue_1739/r2v2_fits_widegrid")
WIDE_RIDGE_LAMBDAS = "0.01,0.1,1.0,10.0,100.0,1000.0,10000.0,100000.0,1000000.0"
# P-C LODO-consistent map+readout (2026-08-07 inline round): own out root +
# HF prefix; the scorer runs with --protocols C (per-holdout map refit).
PC_HF_OUT_PREFIX = "issue1739_r2v2_pc"
PC_OUT_ROOT = Path("eval_results/issue_1739/r2v2_pc")
CTXMAP_PREFIX = "issue1739_ctxmap"
BANK_PREFIX = f"{CTXMAP_PREFIX}/rb_fc_bank"
HALLU_PR_FILE = f"{CTXMAP_PREFIX}/judge/hallucination/labeling_per_rollout.json"
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


def stage_factorial_inputs(args, behavior: str, token: str) -> None:
    """Factorial-leg extras: banked fc directions + the hallucination
    per-rollout DV + the E1 extraction store FORCED (e1_fc reads the store's
    context_end shards even when the r_b_e1 bank makes the base leg skip it)."""
    from explore_persona_space.orchestrate import hub
    from scripts.issue1739_wcrung_arms_run import stage_extraction

    files = hub.stage_hub_prefix(
        hub.DEFAULT_DATASET_REPO,
        BANK_PREFIX,
        args.ood_mirror_root,
        repo_type="dataset",
        token=token or None,
        max_workers=4,
    )
    _log(f"[phase=stage_factorial {behavior}] rb_fc_bank: {len(files)} files")
    if behavior == "hallucination":
        dest = args.ood_mirror_root / HALLU_PR_FILE
        if not dest.exists():
            hub.stage_hub_file(
                hub.DEFAULT_DATASET_REPO,
                HALLU_PR_FILE,
                dest,
                repo_type="dataset",
                token=token,
            )
        _log(f"[phase=stage_factorial {behavior}] per-rollout DV -> {dest}")
    ns = argparse.Namespace(
        store_root=args.store_root,
        tensors_root=args.tensors_root,
        revision=args.revision,
        stage_workers=args.stage_workers,
    )
    stage_extraction(behavior, ns, token, force=True)


def factorial_cmd(args, behavior: str) -> list[str]:
    cmd = [
        sys.executable,
        str(_REPO_ROOT / "scripts" / "issue1739_r2v2_factorial.py"),
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
        "--fact-out-root",
        str(FACT_OUT_ROOT),
        "--rb-bank-dir",
        str(args.ood_mirror_root / BANK_PREFIX),
        "--ood-store-root",
        str(args.ood_mirror_root / CTXMAP_PREFIX),
        "--device",
        args.device,
        "--ood-dv-max-null-frac",
        str(args.ood_dv_max_null_frac),
    ]
    if behavior == "hallucination":
        cmd += ["--hallu-per-rollout-dv", str(args.ood_mirror_root / HALLU_PR_FILE)]
    if args.pb_holdouts:
        cmd += ["--pb-holdouts", *args.pb_holdouts]
    return cmd


def score_cmd(args, behavior: str, out_root: Path = OUT_ROOT) -> list[str]:
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
        str(out_root),
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


# leg -> (local out root, HF prefix); every leg the driver can run.
LEG_DESTS = {
    "fits": (OUT_ROOT, HF_OUT_PREFIX),
    "factorial": (FACT_OUT_ROOT, FACT_HF_OUT_PREFIX),
    "fits-widegrid": (WIDE_OUT_ROOT, WIDE_HF_OUT_PREFIX),
    "pc": (PC_OUT_ROOT, PC_HF_OUT_PREFIX),
}


def leg_cmd_env(args, behavior: str, leg: str) -> tuple[list[str], dict[str, str]]:
    """Compose one leg's subprocess argv + env overlay (explicit, never implicit)."""
    if leg == "fits":
        return score_cmd(args, behavior), {}
    if leg == "fits-widegrid":
        # identical fits, 3-decades-wider GCV grid, own (untracked) out root;
        # the env var is read ONCE at constants import in the child process.
        return (
            score_cmd(args, behavior, out_root=WIDE_OUT_ROOT),
            {"EPS_I1739_RIDGE_LAMBDAS": WIDE_RIDGE_LAMBDAS},
        )
    if leg == "factorial":
        return factorial_cmd(args, behavior), {}
    if leg == "pc":
        # P-C rides the same scorer with --protocols C (the driver caller
        # passes --protocols C) and its own out root / HF prefix.
        return score_cmd(args, behavior, out_root=PC_OUT_ROOT), {}
    raise ValueError(f"unknown leg {leg!r}")


def upload_behavior(args, behavior: str, leg: str = "fits") -> str:
    from explore_persona_space.orchestrate import hub

    out_root, prefix = LEG_DESTS[leg]
    local = out_root / behavior
    if not local.exists():
        raise FileNotFoundError(f"nothing to upload: {local} absent")
    url = hub._upload(
        local,
        hub.DEFAULT_DATASET_REPO,
        "dataset",
        f"{prefix}/{behavior}",
        raise_on_error=True,
    )
    _log(f"[phase=upload {behavior} leg={leg}] -> {prefix}/{behavior} ({url})")
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
    ap.add_argument("--protocols", default="AB", choices=["A", "B", "AB", "C", "ABC"])
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
    ap.add_argument(
        "--legs",
        nargs="+",
        default=["fits"],
        choices=["fits", "factorial", "fits-widegrid", "pc"],
        help="which scoring legs to run per behavior (in the given order)",
    )
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
        import inspect

        from explore_persona_space.orchestrate import hub  # noqa: F401
        from explore_persona_space.orchestrate.env import load_dotenv  # noqa: F401
        from scripts.issue1739_jobd_r2aug_run import stage_inputs  # noqa: F401
        from scripts.issue1739_pvsynth_arms_run import stage_shared  # noqa: F401
        from scripts.issue1739_r2v2_factorial import parse_args as _fact_parse_args
        from scripts.issue1739_wcrung_arms_run import stage_extraction

        assert callable(hub.stage_hub_prefix)
        assert callable(hub.stage_hub_file)
        assert "force" in inspect.signature(stage_extraction).parameters
        # bind every composed leg argv (arity/keyword pin)
        from scripts.issue1739_r2v2_score import parse_args as _score_parse_args

        _fact_parse_args(factorial_cmd(args, "hallucination")[2:])
        wide_cmd, wide_env = leg_cmd_env(args, "evil", "fits-widegrid")
        _score_parse_args(wide_cmd[2:])
        assert wide_env["EPS_I1739_RIDGE_LAMBDAS"] == WIDE_RIDGE_LAMBDAS
        pc_cmd, pc_env = leg_cmd_env(args, "hallucination", "pc")
        pc_args = _score_parse_args(pc_cmd[2:])
        assert pc_env == {} and str(pc_args.out_root) == str(PC_OUT_ROOT)
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
        if "factorial" in args.legs:
            stage_factorial_inputs(args, behavior, token)
        if args.stage_only:
            _log(f"[phase=stage_only {behavior}] staging complete, skipping score")
            results[behavior] = {"score_rc": None, "staged_only": True}
            continue
        leg_results: dict[str, dict] = {}
        for leg in args.legs:
            cmd, env_overlay = leg_cmd_env(args, behavior, leg)
            _log(f"[phase=score {behavior} leg={leg}] {' '.join(cmd[1:])} env+={env_overlay}")
            proc = subprocess.run(
                cmd, cwd=str(_REPO_ROOT), check=False, env={**os.environ, **env_overlay}
            )
            _log(f"[phase=score {behavior} leg={leg}] rc={proc.returncode}")
            url = None
            if proc.returncode == 0 and not args.skip_upload:
                url = upload_behavior(args, behavior, leg)
            leg_results[leg] = {
                "rc": proc.returncode,
                "uploaded": bool(url),
                "upload_url": url,
            }
            if proc.returncode != 0:
                break  # a failed leg never gates a later leg's inputs silently
        rc_b = max(r["rc"] for r in leg_results.values())
        if rc_b == 0 and not args.no_reap:
            reap_labeling_slice(args, behavior)
        results[behavior] = {
            "score_rc": rc_b,
            "legs": leg_results,
            "wall_s": round(time.time() - t_b, 1),
        }
        overall_rc = overall_rc or rc_b
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
