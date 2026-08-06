#!/usr/bin/env python3
"""Instance-side driver for the #1739 wildchat-rung arm-scoring leg.

Structural sibling of ``issue1739_pvsynth_arms_run.py``: two phases that
INTERLEAVE, because they bottleneck on different resources.

* STAGE (network/CPU): the three TRAIN capture stores live on the HF data repo
  as monolithic tars (evil 29.9 / sycophancy 48.6 / hallucination 65.1 GiB).
  They are streamed member-selectively through
  ``issue1739_map963k_slice.stream_slice`` — bytes transferred are the whole
  tar, bytes WRITTEN are only the requested (kind x layer) members, and no tar
  copy ever lands on disk (peak disk = the extracted slice, not 2x). Each
  behavior's ``slice_manifest.json`` is its completion sentinel; the slicer is
  resumable, so a relaunch re-streams but re-writes nothing.
* SCORE (GPU): ``issue1739_wcrung_arms.py`` per behavior.

Run serially that would be ~95 min of staging with the GPU at ~0% followed by
scoring — the #664/#778 idle-GPU shape. Instead staging runs in a background
THREAD (I/O-bound, releases the GIL) while scoring walks the behaviors in
staging order, waiting on each sentinel.

The wildchat rung's OWN inputs are small and shared, so they are staged once up
front (:func:`stage_shared`) rather than per behavior:

* ONE capture store at ``<HF_PREFIX>/wildchat_rung/capture_store/wildchat`` —
  the rung's contexts are behavior-independent (generate-once/judge-3x), so
  all three behaviors score against the same activations. See
  ``issue1739_wcrung_arms`` module docstring.
* THREE per-behavior DV datasets at
  ``<HF_PREFIX>/wildchat_rung/dv_dataset/<behavior>/labeling.json`` (one per
  trait rubric over that one pool).

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
    sentinel = root / "scripts" / "issue1739_wcrung_arms.py"
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
RUNG_PREFIX = f"{HF_PREFIX}/wildchat_rung"
# The GPU leg's pseudo-behavior dir: ONE shared capture store for every judged
# behavior (issue1739_wcrung_pod.GEN_BEHAVIOR / wcrung_arms.EVAL_STORE_DIR_NAME).
EVAL_STORE_DIR_NAME = "wildchat"
SENTINEL_NAME = "wcrung_arms_done.json"
STAGE_MANIFEST = "slice_manifest.json"
# store_io shard written by every capture store — the staging completion probe.
STORE_PROBE = "row_index_shard00.jsonl"
# The committed train grid this rung freezes against. A run with FEWER layers
# cannot use committed-frozen indices (they index this full grid).
FULL_GRID_N_LAYERS = 28


def _log(msg: str) -> None:
    print(f"[wcrung-arms-run] {msg}", flush=True)


# ---------------------------------------------------------------------------
# staging
# ---------------------------------------------------------------------------


def stage_wcrung_store(args) -> Path:
    """Stage the ONE shared wildchat-rung capture store."""
    from explore_persona_space.orchestrate import hub

    dest = args.store_root / "wcrung_capture_store" / EVAL_STORE_DIR_NAME
    if (dest / STORE_PROBE).exists():
        _log(f"[phase=stage_shared] wcrung store: present, skip ({dest})")
        return dest
    # stage_hub_prefix mirrors the repo-relative tree under the mirror root, so
    # the root must satisfy root/<prefix> == staged (#1774). Signature is
    # (repo_id, prefix, dest_dir, *, repo_type=...) — the repo_id is NOT
    # optional; a 2-positional call is a deterministic TypeError (#1332 class,
    # pinned by test_stage_wcrung_store_binds_the_real_hub_signature).
    mirror_root = args.store_root / "_wcmirror"
    hub.stage_hub_prefix(
        hub.DEFAULT_DATASET_REPO,
        f"{RUNG_PREFIX}/capture_store/{EVAL_STORE_DIR_NAME}",
        mirror_root,
        repo_type="dataset",
    )
    staged = mirror_root / RUNG_PREFIX / "capture_store" / EVAL_STORE_DIR_NAME
    if not (staged / STORE_PROBE).exists():
        raise RuntimeError(f"wcrung store staging incomplete: {staged} lacks {STORE_PROBE}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    staged.rename(dest)
    _log(f"[phase=stage_shared] wcrung store -> {dest}")
    return dest


def stage_shared(args, token: str) -> None:
    """Stage inputs shared across behaviors: the wcrung store + both DV trees."""
    from explore_persona_space.orchestrate import hub

    _log("[phase=stage_shared] wcrung capture store + wcrung DVs + train DV tree")
    stage_wcrung_store(args)

    for behavior in args.behaviors:
        # The rung's own DV (one per rubric over the shared pool). The arms
        # driver reads it at <out-root>/dv_dataset/<behavior>/labeling.json.
        out = args.out_root / "dv_dataset" / behavior / "labeling.json"
        if out.exists():
            _log(f"[phase=stage_shared] wcrung DV {behavior}: present, skip")
        else:
            out.parent.mkdir(parents=True, exist_ok=True)
            # stage_hub_file: retried (429/5xx), atomic, exact target.
            hub.stage_hub_file(
                hub.DEFAULT_DATASET_REPO,
                f"{RUNG_PREFIX}/dv_dataset/{behavior}/labeling.json",
                out,
                repo_type="dataset",
                token=token,
            )
            _log(f"[phase=stage_shared] wcrung DV {behavior} -> {out}")

        train_out = args.train_dv_root / behavior / "labeling.json"
        if train_out.exists():
            continue
        train_out.parent.mkdir(parents=True, exist_ok=True)
        hub.stage_hub_file(
            hub.DEFAULT_DATASET_REPO,
            f"{HF_PREFIX}/judge/dv_dataset/{behavior}/labeling.json",
            train_out,
            repo_type="dataset",
            token=token,
        )
        _log(f"[phase=stage_shared] train DV {behavior} -> {train_out}")


def stage_extraction(behavior: str, args, token: str, *, force: bool = False) -> None:
    """Stage a behavior's E1 extraction store (only when its r_B bank is absent).

    ``force=True`` stages even when the r_B bank exists — the factorial leg's
    e1_fc regime reads the store's context_end shards, which the bank does not
    carry (a present store still short-circuits either way).
    """
    bank = args.tensors_root / "r_b_e1" / f"{behavior}.npz"
    dest = args.store_root / f"{behavior}_extraction"
    if (dest / "row_index.jsonl").exists():
        return
    if bank.exists() and not force:
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


def staged_slice_covers(
    dest: Path, *, kinds: tuple[str, ...], layers: list[int]
) -> tuple[bool, str]:
    """Does an existing slice_manifest COVER the requested (kinds x layers)?

    The slice manifest records the regime it was written for. A bare existence
    check would let a narrow probe run (``--layers 0 1``) satisfy a later
    full-grid drive, which then fails downstream on the missing layer arrays
    instead of at staging. Returns ``(covered, reason_if_not)``; a missing or
    unreadable manifest is "not covered" with an empty reason (a fresh stage,
    not a re-stage).
    """
    manifest_path = dest / STAGE_MANIFEST
    if not manifest_path.is_file():
        return False, ""
    try:
        m = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return False, f"unreadable {STAGE_MANIFEST} ({type(exc).__name__}) — re-staging"
    have_layers = {int(x) for x in m.get("layers", [])}
    have_kinds = set(m.get("kinds", []))
    missing_layers = sorted(set(int(x) for x in layers) - have_layers)
    missing_kinds = sorted(set(kinds) - have_kinds)
    if not missing_layers and not missing_kinds:
        return True, ""
    bits = []
    if missing_layers:
        bits.append(f"{len(missing_layers)} layer(s) absent (e.g. {missing_layers[:4]})")
    if missing_kinds:
        bits.append(f"kinds absent {missing_kinds}")
    return False, f"staged regime is narrower than requested: {'; '.join(bits)}"


def stage_all(args, token: str, errors: list[BaseException]) -> None:
    """Background staging thread: shared inputs, then each train store in order."""
    try:
        stage_shared(args, token)
        from scripts.issue1739_map963k_slice import stream_slice

        for behavior in args.behaviors:
            stage_extraction(behavior, args, token)
            dest = args.store_root / f"{behavior}_labeling"
            covered, why = staged_slice_covers(dest, kinds=KINDS, layers=args.layers)
            if covered:
                _log(f"[phase=stage] {behavior}: slice_manifest covers this regime, skip")
                continue
            if why:
                # A manifest exists but was written for a NARROWER regime — the
                # probe-then-full-drive shape. Existence alone would skip here and
                # the store would later fail to load the missing layer arrays, one
                # wasted cycle later. stream_slice is resumable and re-writes
                # nothing already present, so re-stage the delta.
                _log(f"[phase=stage] {behavior}: re-staging — {why}")
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


def score_cmd(behavior: str, args) -> list[str]:
    """The scorer argv for one behavior (no --wcrung-store: the staged path IS
    the driver's ``--store-root`` default)."""
    cmd = [
        sys.executable,
        str(_REPO_ROOT / "scripts" / "issue1739_wcrung_arms.py"),
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
    # Committed-frozen layers are indices into the FULL 28-layer grid, so a
    # reduced-layer run (the probe shape) MUST select frozen layers within its
    # own layer set — the driver fail-louds otherwise. Auto-enable for any
    # non-full layer list so the probe invocation cannot forget it; the explicit
    # flag still forces it at full width.
    if args.force_own_pool_frozen or len(args.layers) < FULL_GRID_N_LAYERS:
        cmd.append("--force-own-pool-frozen")
    return cmd


def score_behavior(behavior: str, args) -> int:
    """Run the scorer for one behavior as a subprocess; return its rc."""
    cmd = score_cmd(behavior, args)
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
        f"{RUNG_PREFIX}/arm_results/{behavior}",
        raise_on_error=True,
    )
    _log(f"[phase=upload behavior={behavior}] -> {RUNG_PREFIX}/arm_results/{behavior}")


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
    ap.add_argument("--out-root", type=Path, default=Path("eval_results/issue_1739/wildchat_rung"))
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
    ap.add_argument(
        "--force-own-pool-frozen",
        action="store_true",
        help="select frozen layers on each behavior's own train pool instead of the "
        "committed train summary (auto-enabled for any reduced --layers set)",
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
        # Names every deferred (function-body) import the real path reaches —
        # a bare module import would not fire these.
        from explore_persona_space.orchestrate import hub  # noqa: F401
        from explore_persona_space.orchestrate.env import load_dotenv  # noqa: F401
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
        "leg": "wcrung_arms",
        "rung": "wildchat_rung",
        "behaviors": results,
        "variants": list(args.variants),
        "n_layers": len(args.layers),
        "eval_store_shared_across_behaviors": True,
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
            f"{RUNG_PREFIX}/arm_results/{SENTINEL_NAME}",
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
