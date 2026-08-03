#!/usr/bin/env python3
"""Work-conserving GPU dispatcher for the #1739 armfill round.

Runs the 12 arm-scoring legs -- 3 behaviors x 2 rungs (wildchat_rung, pvsynth)
x 2 variants (context_end, prefix_end) -- across the box's GPUs, one leg per
device, starting the next leg the moment a device frees (no wave barriers).

Two properties this dispatcher exists to guarantee:

* **Per-leg out-roots.** Both scorers write ``<out-root>/<behavior>/
  all_arms_spearman.json``, top-level ``*_pool_coherence.json`` /
  ``*_failures.json``, and share ONE ``percell/`` resume checkpoint, so two
  concurrent legs pointed at a shared out-root clobber each other. Each leg
  gets ``<legs-root>/<behavior>_<rung>_<variant>/<rung>``; the trailing
  component is the rung name because both scorers' ``_assert_outputs_safe``
  refuses an out-root whose basename is not the rung.
* **Staging gates.** A leg starts only once its behavior's inputs are on
  disk. The train slice lands asynchronously (a separate ``--stage-only``
  driver streams the three tars in order), so leg readiness is a file
  predicate, not a wall-clock guess.

Fences are sized from a MEASURED pilot leg (``--per-leg-timeout-s``), never a
guess; a leg is never killed below its fence.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

BEHAVIORS = ("evil", "sycophancy", "hallucination")
RUNGS = ("wildchat_rung", "pvsynth")
VARIANTS = ("context_end", "prefix_end")
ARMS = (
    "arm2_ctx_native",
    "arm9_pretrain_ft",
    "arm14_shuffled_pt",
    "arm17_oracle_mlp",
    "arm18_oracle_krr",
)
SCORER = {
    "wildchat_rung": "scripts/issue1739_wcrung_arms.py",
    "pvsynth": "scripts/issue1739_pvsynth_arms.py",
}
# Written LAST by the tar slicer -- the train-slice completion sentinel.
SLICE_MANIFEST = "slice_manifest.json"
STORE_PROBE = "row_index_shard00.jsonl"


def _log(msg: str) -> None:
    print(f"[armfill-dispatch] {time.strftime('%H:%M:%S')} {msg}", flush=True)


def leg_slug(behavior: str, rung: str, variant: str) -> str:
    return f"{behavior}_{rung}_{variant}"


def build_legs(args) -> list[dict]:
    legs = []
    for behavior in args.behaviors:
        for rung in args.rungs:
            for variant in args.variants:
                slug = leg_slug(behavior, rung, variant)
                out_root = args.legs_root / slug / rung
                legs.append(
                    {
                        "slug": slug,
                        "behavior": behavior,
                        "rung": rung,
                        "variant": variant,
                        "out_root": out_root,
                        "log": args.log_root / f"leg_{slug}.log",
                    }
                )
    return legs


def leg_ready(leg: dict, args) -> tuple[bool, str]:
    """Is every input for this leg on disk yet?"""
    behavior = leg["behavior"]
    train_slice = args.store_root / f"{behavior}_labeling" / SLICE_MANIFEST
    if not train_slice.is_file():
        return False, f"train slice pending ({train_slice})"
    if leg["rung"] == "wildchat_rung":
        store = args.store_root / "wcrung_capture_store" / "wildchat" / STORE_PROBE
        dv = args.wcrung_dv_root / behavior / "labeling.json"
    else:
        store = args.store_root / "pvsynth_capture_store" / behavior / STORE_PROBE
        dv = args.pvsynth_dv_root / behavior / "labeling.json"
    for label, p in (("eval store", store), ("DV", dv)):
        if not p.exists():
            return False, f"{label} pending ({p})"
    return True, ""


def leg_cmd(leg: dict, args) -> list[str]:
    behavior, rung = leg["behavior"], leg["rung"]
    cmd = [
        args.python,
        SCORER[rung],
        "--behaviors",
        behavior,
        "--variants",
        leg["variant"],
        "--arms",
        *args.arms,
        "--store-root",
        str(args.store_root),
        "--train-dv-root",
        str(args.store_root / "train_dv"),
        "--main-root",
        str(args.main_root),
        "--tensors-root",
        str(args.tensors_root),
        "--u-store",
        str(args.store_root / "u_store"),
        "--out-root",
        str(leg["out_root"]),
        "--device",
        args.device,
        "--n-layers",
        str(args.n_layers),
        "--regime",
        args.regime,
        "--u-size",
        args.u_size,
    ]
    if rung == "wildchat_rung":
        cmd += ["--wcrung-dv-json", str(args.wcrung_dv_root / behavior / "labeling.json")]
    else:
        cmd += ["--pvsynth-dv-json", str(args.pvsynth_dv_root / behavior / "labeling.json")]
    return cmd


def launch(leg: dict, gpu: int, args) -> dict:
    leg["out_root"].mkdir(parents=True, exist_ok=True)
    env = {**os.environ}
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    # Keep the CPU-side fits from oversubscribing when several legs share the box.
    for var in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        env[var] = str(args.threads_per_leg)
    env["MALLOC_ARENA_MAX"] = "2"
    cmd = leg_cmd(leg, args)
    fh = open(leg["log"], "w")
    fh.write(f"# gpu={gpu}\n# {' '.join(cmd)}\n")
    fh.flush()
    proc = subprocess.Popen(
        cmd,
        cwd=str(args.repo_root),
        env=env,
        stdout=fh,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    _log(f"LAUNCH {leg['slug']} gpu={gpu} pid={proc.pid} log={leg['log']}")
    return {"leg": leg, "gpu": gpu, "proc": proc, "fh": fh, "t0": time.time()}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--repo-root", type=Path, default=Path("/workspace/explore-persona-space"))
    ap.add_argument("--python", default="/workspace/explore-persona-space/.venv/bin/python")
    ap.add_argument("--store-root", type=Path, default=Path("data/issue_1739/hf_dl"))
    ap.add_argument("--main-root", type=Path, default=Path("eval_results/issue_1739"))
    ap.add_argument("--tensors-root", type=Path, default=Path("analysis_tensors/issue_1739"))
    ap.add_argument("--legs-root", type=Path, default=Path("/workspace/legs"))
    ap.add_argument("--log-root", type=Path, default=Path("/workspace/logs"))
    ap.add_argument(
        "--wcrung-dv-root", type=Path, default=Path("/workspace/stage/wildchat_rung/dv_dataset")
    )
    ap.add_argument(
        "--pvsynth-dv-root",
        type=Path,
        default=Path("/workspace/explore-persona-space/eval_results/issue_1739/pvsynth/dv_dataset"),
    )
    ap.add_argument("--behaviors", nargs="+", default=list(BEHAVIORS))
    ap.add_argument("--rungs", nargs="+", default=list(RUNGS))
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS))
    ap.add_argument("--arms", nargs="+", default=list(ARMS))
    ap.add_argument("--gpus", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--threads-per-leg", type=int, default=16)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n-layers", type=int, default=28)
    ap.add_argument("--regime", default="e1")
    ap.add_argument("--u-size", default="full")
    ap.add_argument(
        "--per-leg-timeout-s",
        type=int,
        default=6 * 3600,
        help="per-leg fence; size at >=2x the MEASURED pilot wall",
    )
    ap.add_argument(
        "--stage-wait-timeout-s",
        type=int,
        default=6 * 3600,
        help="give up waiting for a behavior's staging to land",
    )
    ap.add_argument(
        "--stagger-s",
        type=int,
        default=90,
        help="wait this long after launching a leg before launching the next; smooths the "
        "simultaneous ~30 GB train-table load (mem_guard reads LIVE MemAvailable at phase "
        "entry, so a synchronized spike can make it refuse a healthy leg) and the MooseFS "
        "read burst. 0 disables.",
    )
    ap.add_argument("--skip-slugs", nargs="+", default=[], help="legs already complete")
    ap.add_argument("--status-json", type=Path, default=Path("/workspace/logs/armfill_status.json"))
    args = ap.parse_args()

    args.legs_root.mkdir(parents=True, exist_ok=True)
    args.log_root.mkdir(parents=True, exist_ok=True)

    pending = [x for x in build_legs(args) if x["slug"] not in args.skip_slugs]
    _log(f"{len(pending)} legs queued across gpus={args.gpus}")
    free_gpus = list(args.gpus)
    running: list[dict] = []
    done: list[dict] = []
    t_start = time.time()
    last_wait_log = 0.0

    while pending or running:
        # Start every ready leg we have a device for (work-conserving).
        progressed = True
        while progressed and free_gpus and pending:
            progressed = False
            for i, leg in enumerate(pending):
                ok, _why = leg_ready(leg, args)
                if ok:
                    running.append(launch(pending.pop(i), free_gpus.pop(0), args))
                    progressed = True
                    if args.stagger_s and (pending or running):
                        _log(f"stagger {args.stagger_s}s before the next launch")
                        time.sleep(args.stagger_s)
                    break

        if pending and free_gpus and (time.time() - last_wait_log) > 300:
            last_wait_log = time.time()
            _, why = leg_ready(pending[0], args)
            _log(f"WAIT {len(pending)} legs not yet stageable; next={pending[0]['slug']} — {why}")
            if time.time() - t_start > args.stage_wait_timeout_s:
                _log("FATAL staging wait exceeded; abandoning unstaged legs")
                pending = []

        time.sleep(10)

        for slot in list(running):
            rc = slot["proc"].poll()
            el = time.time() - slot["t0"]
            if rc is None:
                if el > args.per_leg_timeout_s:
                    _log(
                        f"FENCE {slot['leg']['slug']} exceeded {args.per_leg_timeout_s}s — killing"
                    )
                    slot["proc"].kill()
                continue
            slot["fh"].close()
            running.remove(slot)
            free_gpus.append(slot["gpu"])
            done.append({"slug": slot["leg"]["slug"], "rc": rc, "wall_s": round(el, 1)})
            _log(f"DONE {slot['leg']['slug']} rc={rc} wall={el / 60:.1f} min")
            args.status_json.write_text(
                json.dumps(
                    {
                        "done": done,
                        "running": [s["leg"]["slug"] for s in running],
                        "pending": [x["slug"] for x in pending],
                    },
                    indent=1,
                )
            )

    args.status_json.write_text(json.dumps({"done": done, "running": [], "pending": []}, indent=1))
    bad = [d for d in done if d["rc"] != 0]
    _log(
        f"ALL DONE {len(done)} legs, {len(bad)} nonzero rc, wall={(time.time() - t_start) / 60:.1f} min"
    )
    for d in bad:
        _log(f"  FAILED {d['slug']} rc={d['rc']}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
