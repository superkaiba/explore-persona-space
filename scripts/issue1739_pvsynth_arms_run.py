#!/usr/bin/env python3
"""Instance-side driver for the #1739 pvsynth arm-scoring leg.

Two phases that INTERLEAVE, because they bottleneck on different resources:

* STAGE (network/CPU): the three train capture stores live on the HF data repo
  as monolithic tars (evil 29.9 / sycophancy 48.6 / hallucination 65.1 GiB).
  They are streamed member-selectively through
  ``issue1739_map963k_slice.stream_slice`` — bytes transferred are the whole
  tar, bytes WRITTEN are only the requested (kind x layer) members, and no
  tar copy ever lands on disk (peak disk = the extracted slice, not 2x). Each
  behavior's ``slice_manifest.json`` is its completion sentinel; the slicer is
  resumable, so a relaunch re-streams but re-writes nothing.
* SCORE (GPU): ``issue1739_pvsynth_arms.py`` per behavior.

Run serially that would be ~95 min of staging with the GPU at ~0% followed by
~75 min of scoring — the #664/#778 idle-GPU shape. Instead staging runs in a
background THREAD (I/O-bound, releases the GIL) while scoring walks the
behaviors in staging order, waiting on each sentinel. Wall collapses to
roughly ``first_stage + max(rest_of_staging, scoring)``.

Behaviors are staged smallest-tar-first so scoring starts as early as possible.
Each behavior's outputs are uploaded to the HF data repo the moment that
behavior finishes (per-cell upload discipline: an instance death strands at
most the in-flight behavior, and on the ephemeral GCE lane the EXIT trap
deletes the boot disk, so git-only outputs would not survive).

Scoring runs as a SUBPROCESS per behavior so one behavior's crash cannot take
the driver down and each phase has an explicit rc.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tarfile
import threading
import time
from pathlib import Path


def _ensure_repo_root_on_syspath() -> Path:
    root = Path(__file__).resolve().parents[1]
    sentinel = root / "scripts" / "issue1739_pvsynth_arms.py"
    if not sentinel.exists():
        raise RuntimeError(f"repo-root resolution failed: {sentinel} missing")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


_REPO_ROOT = _ensure_repo_root_on_syspath()

# Smallest tar first: scoring starts sooner and the long tail overlaps it.
STAGE_ORDER = ("evil", "sycophancy", "hallucination")
TAR_GIB = {"evil": 29.9, "sycophancy": 48.6, "hallucination": 65.1}
KINDS = ("prefix_end", "context_end", "t1")
HF_PREFIX = "issue1739_ctxmap"
SENTINEL_NAME = "pvsynth_arms_done.json"
STAGE_MANIFEST = "slice_manifest.json"


def _log(msg: str) -> None:
    print(f"[pvsynth-arms-run] {msg}", flush=True)


# ---------------------------------------------------------------------------
# staging
# ---------------------------------------------------------------------------


def stage_shared(args, token: str) -> None:
    """Stage inputs shared across behaviors: pvsynth stores + train DV tree."""
    from explore_persona_space.orchestrate import hub

    _log("[phase=stage_shared] pvsynth capture stores + train DV tree")
    for behavior in args.behaviors:
        dest = args.store_root / "pvsynth_capture_store" / behavior
        if (dest / "row_index_shard00.jsonl").exists():
            _log(f"[phase=stage_shared] pvsynth store {behavior}: present, skip")
            continue
        # stage_hub_prefix mirrors the repo-relative tree under dest, so the
        # mirror root must satisfy root/<prefix> == dest (#1774).
        mirror_root = args.store_root / "_pvmirror"
        hub.stage_hub_prefix(
            hub.DEFAULT_DATASET_REPO,
            f"{HF_PREFIX}/pvsynth/capture_store/{behavior}",
            mirror_root,
            repo_type="dataset",
        )
        staged = mirror_root / HF_PREFIX / "pvsynth" / "capture_store" / behavior
        if not (staged / "row_index_shard00.jsonl").exists():
            raise RuntimeError(f"pvsynth store staging incomplete for {behavior}: {staged}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        staged.rename(dest)
        _log(f"[phase=stage_shared] pvsynth store {behavior} -> {dest}")

    for behavior in args.behaviors:
        out = args.train_dv_root / behavior / "labeling.json"
        if out.exists():
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        # stage_hub_file: retried (429/5xx), atomic, exact target — no mirror root.
        hub.stage_hub_file(
            hub.DEFAULT_DATASET_REPO,
            f"{HF_PREFIX}/judge/dv_dataset/{behavior}/labeling.json",
            out,
            repo_type="dataset",
            token=token,
        )
        _log(f"[phase=stage_shared] train DV {behavior} -> {out}")


def stage_extraction(behavior: str, args, token: str) -> None:
    """Stage a behavior's E1 extraction store (only when its r_B bank is absent)."""
    bank = args.tensors_root / "r_b_e1" / f"{behavior}.npz"
    dest = args.store_root / f"{behavior}_extraction"
    if bank.exists() or (dest / "row_index.jsonl").exists():
        return
    from explore_persona_space.orchestrate import hub

    _log(f"[phase=stage_extraction] {behavior}: no r_B bank, fetching extraction tar (~1.1 GiB)")
    tar_path = hub.stage_hub_file(
        hub.DEFAULT_DATASET_REPO,
        f"{HF_PREFIX}/capture_store/{behavior}_extraction/{behavior}_extraction.tar",
        args.store_root / f"{behavior}_extraction.tar",
        repo_type="dataset",
        token=token,
    )
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, mode="r|") as tar:  # streaming; members flattened
        for member in tar:
            if not member.isfile():
                continue
            src = tar.extractfile(member)
            if src is None:
                continue
            out = dest / member.name.rsplit("/", 1)[-1]
            tmp = out.with_suffix(out.suffix + ".tmp")
            with open(tmp, "wb") as fh:
                while True:
                    chunk = src.read(4 << 20)
                    if not chunk:
                        break
                    fh.write(chunk)
            tmp.replace(out)
    Path(tar_path).unlink(missing_ok=True)  # the extracted slice is what we need
    _log(f"[phase=stage_extraction] {behavior} -> {dest}")


def stage_all(args, token: str, errors: list[BaseException]) -> None:
    """Background staging thread: shared inputs, then each train store in order."""
    try:
        stage_shared(args, token)
        from scripts.issue1739_map963k_slice import stream_slice

        for behavior in args.behaviors:
            stage_extraction(behavior, args, token)
            dest = args.store_root / f"{behavior}_labeling"
            if (dest / STAGE_MANIFEST).exists():
                _log(f"[phase=stage] {behavior}: slice_manifest present, skip")
                continue
            _log(
                f"[phase=stage] {behavior}: streaming {TAR_GIB.get(behavior, 0):.1f} GiB tar "
                f"({len(KINDS)} kinds x {len(args.layers)} layers) -> {dest}"
            )
            m = stream_slice(
                behavior,
                dest,
                revision=args.revision,
                kinds=KINDS,
                layers=tuple(args.layers),
                token=token,
                workers=args.stage_workers,
            )
            _log(
                f"[phase=stage] {behavior}: DONE written={m['kept_bytes'] / 1e9:.1f} GB "
                f"fetched={m['bytes_fetched'] / 1e9:.1f} GB in {m['elapsed_s']:.0f}s "
                f"({m['mb_per_s']:.1f} MB/s)"
            )
    except BaseException as exc:  # surfaced by the waiter — never a silent stall
        errors.append(exc)
        _log(f"[phase=stage] FAILED: {type(exc).__name__}: {exc}")
        raise


def wait_for_stage(behavior: str, args, errors: list[BaseException]) -> None:
    """Block until the behavior's slice manifest lands (or staging failed)."""
    manifest = args.store_root / f"{behavior}_labeling" / STAGE_MANIFEST
    t0 = time.time()
    last = 0.0
    while not manifest.exists():
        if errors:
            raise RuntimeError(f"staging thread died before {behavior} completed") from errors[0]
        waited = time.time() - t0
        if waited > args.stage_timeout_s:
            raise TimeoutError(
                f"{behavior}: staging did not complete within {args.stage_timeout_s}s "
                f"(manifest {manifest} absent)"
            )
        if waited - last >= 120:
            last = waited
            _log(f"[phase=wait] {behavior}: awaiting stage, {waited / 60:.0f} min elapsed")
        time.sleep(10)


# ---------------------------------------------------------------------------
# scoring + upload
# ---------------------------------------------------------------------------


def score_behavior(behavior: str, args) -> int:
    """Run the scorer for one behavior as a subprocess; return its rc."""
    cmd = [
        sys.executable,
        str(_REPO_ROOT / "scripts" / "issue1739_pvsynth_arms.py"),
        "--behaviors",
        behavior,
        "--variants",
        *args.variants,
        "--store-root",
        str(args.store_root),
        "--train-dv-root",
        str(args.train_dv_root),
        "--main-root",
        str(args.main_root),
        "--tensors-root",
        str(args.tensors_root),
        "--out-root",
        str(args.out_root),
        "--device",
        args.device,
        "--n-layers",
        str(len(args.layers)),
    ]
    if args.arms:
        # Roster passthrough: 'wide' (scorer default), 'core', 'wide-nomlp', or
        # an explicit slug list. Omitted here means the scorer's own default.
        cmd += ["--arms", *args.arms]
    _log(f"[phase=score behavior={behavior}] {' '.join(cmd[1:])}")
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=str(_REPO_ROOT), check=False, env=args.child_env)
    _log(f"[phase=score behavior={behavior}] rc={proc.returncode} in {time.time() - t0:.0f}s")
    return proc.returncode


def upload_behavior(behavior: str, args) -> None:
    """Upload one behavior's arm-scoring outputs (JSON/text) to the data repo."""
    from explore_persona_space.orchestrate import hub

    local = args.out_root / behavior
    if not local.exists():
        raise FileNotFoundError(f"nothing to upload for {behavior}: {local} absent")
    hub._upload(
        local,
        hub.DEFAULT_DATASET_REPO,
        "dataset",
        f"{HF_PREFIX}/pvsynth/arm_results/{behavior}",
        raise_on_error=True,
    )
    _log(f"[phase=upload behavior={behavior}] -> {HF_PREFIX}/pvsynth/arm_results/{behavior}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--behaviors", nargs="+", default=list(STAGE_ORDER))
    ap.add_argument("--variants", nargs="+", default=["context_end", "prefix_end"])
    ap.add_argument(
        "--arms",
        nargs="+",
        default=None,
        metavar="ROSTER|SLUG",
        help="transfer roster passed through to the scorer: 'wide' (its default), "
        "'core', 'wide-nomlp', or an explicit arm-slug list",
    )
    ap.add_argument("--layers", type=int, nargs="+", default=list(range(28)))
    ap.add_argument("--store-root", type=Path, default=Path("data/issue_1739/hf_dl"))
    ap.add_argument("--train-dv-root", type=Path, default=None)
    ap.add_argument("--main-root", type=Path, default=Path("eval_results/issue_1739"))
    ap.add_argument("--out-root", type=Path, default=Path("eval_results/issue_1739/pvsynth"))
    ap.add_argument("--tensors-root", type=Path, default=Path("analysis_tensors/issue_1739"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--revision", default="main")
    ap.add_argument("--stage-workers", type=int, default=12)
    ap.add_argument(
        "--stage-timeout-s",
        type=int,
        default=4 * 3600,
        help="per-behavior staging fence; 2x the ~40 min worst-case tar at ~30 MB/s x3 behaviors",
    )
    ap.add_argument("--skip-upload", action="store_true")
    ap.add_argument("--stage-only", action="store_true")
    ap.add_argument("--import-check", action="store_true")
    args = ap.parse_args(argv)
    if args.train_dv_root is None:
        args.train_dv_root = args.store_root / "train_dv"
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    if args.import_check:
        from explore_persona_space.orchestrate import hub  # noqa: F401
        from huggingface_hub import hf_hub_download  # noqa: F401
        from scripts.issue1739_map963k_slice import stream_slice  # noqa: F401

        _log("import-check OK")
        sys.stdout.flush()
        sys.stderr.flush()
        sys.exit(0)

    import os

    from explore_persona_space.orchestrate.env import load_dotenv

    load_dotenv()
    token = os.environ.get("HF_TOKEN") or ""
    if not token:
        raise SystemExit("HF_TOKEN missing — staging and upload both need it")
    args.child_env = {**os.environ}
    args.store_root.mkdir(parents=True, exist_ok=True)

    errors: list[BaseException] = []
    stager = threading.Thread(
        target=stage_all, args=(args, token, errors), name="stage", daemon=True
    )
    stager.start()
    _log(
        f"[phase=start] behaviors={args.behaviors} staging in background "
        f"(~{sum(TAR_GIB.get(b, 0) for b in args.behaviors):.0f} GiB of tar to stream)"
    )

    if args.stage_only:
        stager.join()
        if errors:
            raise SystemExit(f"staging failed: {errors[0]}")
        _log("[phase=done] stage-only complete")
        sys.stdout.flush()
        sys.stderr.flush()
        sys.exit(0)

    results: list[dict] = []
    t_all = time.time()
    for behavior in args.behaviors:
        wait_for_stage(behavior, args, errors)
        rc = score_behavior(behavior, args)
        entry = {"behavior": behavior, "score_rc": rc, "uploaded": False}
        if rc == 0 and not args.skip_upload:
            upload_behavior(behavior, args)
            entry["uploaded"] = True
        results.append(entry)

    stager.join(timeout=args.stage_timeout_s)
    sentinel = {
        "leg": "pvsynth_arms",
        "behaviors": results,
        "variants": list(args.variants),
        "n_layers": len(args.layers),
        "wall_s": round(time.time() - t_all, 1),
        "stage_errors": [f"{type(e).__name__}: {e}" for e in errors],
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    args.out_root.mkdir(parents=True, exist_ok=True)
    (args.out_root / SENTINEL_NAME).write_text(json.dumps(sentinel, indent=1))
    if not args.skip_upload:
        from explore_persona_space.orchestrate import hub

        hub._upload(
            args.out_root / SENTINEL_NAME,
            hub.DEFAULT_DATASET_REPO,
            "dataset",
            f"{HF_PREFIX}/pvsynth/arm_results/{SENTINEL_NAME}",
            upload_as_file=True,
            raise_on_error=True,
        )
    failed = [r["behavior"] for r in results if r["score_rc"] != 0]
    _log(f"[phase=done] {len(results) - len(failed)}/{len(results)} scored; failed={failed}")
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(2 if (failed or errors) else 0)


if __name__ == "__main__":
    main()
