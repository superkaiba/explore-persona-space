"""Issue #491 phase dispatcher — smoke IS the sweep with one cell.

Architecture (plan v3 §4.7, PASS_UNIFIED contract): every phase runs through
the SAME subprocess shapes whether invoked by ``--phase smoke`` (one cell,
tiny question/context subsets threaded by explicit flags) or by the full
sweep phases. Per-phase cell-list sources:

  data         data_build.py (full build both modes; helpful-demo gen on GPU)
  icl_eval     ``--variants`` subset (smoke: base_noprefix + icl_K8_chainA,
               ``--contexts villain``); sweep: all 19 registry variants
  train        ``--runs`` subset (smoke: ft_K8_chainA at --epochs 12); each
               run = train -> slot_eval ft_run_pipeline (matching-basis ->
               match [writes matched_pairs/by_run/<run>.json only] -> traj ->
               full) -> inloop_crosscheck (#534, blocking) -> persist-prune;
               matched_summary.json assembled ONCE after all workers join
  free_gen     ``--cells`` subset (smoke: 2 cells x 1 context x 3 q); sweep:
               all 29 registry cells sharded over <=2 vLLM workers; then the
               own_policy slot reads on the SAME outputs
  activations  ``--variants`` subset (smoke: 2 variants x 2 q); sweep: all
               28 registry variants; then summarize
  upload       same helper calls both modes (smoke uploads the tiny smoke
               namespace bucket)

Pod-side contract (poll_pipeline.py): ``[phase=<name>]`` per phase,
``[phase=done]`` EXACTLY ONCE immediately before clean exit (after the final
sentinel write); per-cell completion lines never carry the phase token. The
end-of-run sentinel carries the ``_SENTINEL_REQUIRED_KEYS``
(sentinel_schema_version=1, kind, version) with the body under ``note``.

GPU pinning: CUDA_VISIBLE_DEVICES is exported per cell IN THE LAUNCHER ENV
before exec (an import-time cuInit silently defeats any in-process clobber —
incident #545), with the matching --gpu arg so train_lora's in-process
clobber rewrites the same value.

Launch (pod): nohup uv run python -m
  explore_persona_space.experiments.icl_vs_ft_491.dispatch --phase all
  --gpus 0,1,2,3 > /workspace/logs/issue-491-run.log 2>&1 < /dev/null &
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

from explore_persona_space.experiments.icl_vs_ft_491.common import (
    DATA_DIR,
    EVAL_DIR,
    HF_BUCKET_491,
    SOURCE_CONTEXT,
    WANDB_PROJECT,
    ns_eval_dir,
    repro_metadata,
    write_json,
)

logger = logging.getLogger("i491.dispatch")

MODULE = "explore_persona_space.experiments.icl_vs_ft_491"
GATE1_MIN_NATS = 2.0
SMOKE_TRAIN_EPOCHS = 12
SMOKE_QUESTIONS = 50  # Gate 1 is registered on the full 50 Q_test
SMOKE_FREE_GEN_QUESTIONS = 3
SMOKE_ACT_QUESTIONS = 2


def _phase(name: str) -> None:
    """Emit the poller-parsed phase line. 'done' is reserved for the terminal exit."""
    print(f"[phase={name}]", flush=True)


def _base_env(gpu: int | None = None) -> dict[str, str]:
    """Explicit subprocess env: parent env + per-cell CUDA pin + WandB project."""
    env = {**os.environ}
    env.setdefault("WANDB_PROJECT", WANDB_PROJECT)
    env.setdefault("EPM_SKIP_INLINE_CHECKPOINT_UPLOAD", "1")
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    return env


def _run(cmd: list[str], *, gpu: int | None = None, name: str = "") -> None:
    """Run one subprocess to completion; fail loud on nonzero rc."""
    logger.info("exec [%s] (gpu=%s): %s", name, gpu, " ".join(cmd))
    rc = subprocess.run(cmd, env=_base_env(gpu)).returncode
    if rc != 0:
        raise RuntimeError(f"subprocess [{name}] failed rc={rc}: {' '.join(cmd)}")
    print(f"cell {name} complete", flush=True)


def _py(module_tail: str, *args: str) -> list[str]:
    return [sys.executable, "-m", f"{MODULE}.{module_tail}", *args]


def _run_jobs(jobs: list[tuple[str, list[list[str]]]], gpus: list[int]) -> None:
    """Worker pool: one worker per GPU; each job is a SEQUENTIAL command pipeline."""
    q: queue.Queue[tuple[str, list[list[str]]]] = queue.Queue()
    for job in jobs:
        q.put(job)
    errors: list[str] = []
    lock = threading.Lock()

    def worker(gpu: int) -> None:
        while True:
            try:
                name, cmds = q.get_nowait()
            except queue.Empty:
                return
            try:
                for i, cmd in enumerate(cmds):
                    _run(cmd, gpu=gpu, name=f"{name}/{i}")
            except Exception as e:
                with lock:
                    errors.append(f"{name}: {e}")
            finally:
                q.task_done()

    threads = [threading.Thread(target=worker, args=(g,), daemon=True) for g in gpus]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if errors:
        raise RuntimeError("job failures:\n" + "\n".join(errors))


# ── Phase implementations ────────────────────────────────────────────────


def phase_data(gpus: list[int], *, skip_gpu_datagen: bool = False) -> None:
    _phase("data")
    _run(_py("data_build", "build"), name="data_build")
    from explore_persona_space.experiments.icl_vs_ft_491.data_build import HELPFUL_DEMOS_PATH

    if HELPFUL_DEMOS_PATH.exists():
        logger.info("helpful demos already built — skipping gen")
    elif skip_gpu_datagen:
        logger.warning("--skip-gpu-datagen: helpful-demo controls will FAIL LOUD at eval time")
    else:
        _run(
            _py("data_build", "gen-helpful-demos", "--gpu", str(gpus[0])),
            gpu=gpus[0],
            name="gen_helpful_demos",
        )


def _icl_variant_ids(smoke: bool) -> list[str]:
    if smoke:
        return ["base_noprefix", "icl_K8_chainA"]
    from explore_persona_space.experiments.icl_vs_ft_491.data_build import load_variants

    return list(load_variants())


def phase_icl_eval(
    gpus: list[int], *, smoke: bool = False, contexts: str | None = None, questions: int = 50
) -> None:
    _phase("icl_eval" if not smoke else "smoke_icl_eval")
    variant_ids = _icl_variant_ids(smoke)
    common_flags = []
    if smoke:
        common_flags.append("--smoke")
    if contexts:
        common_flags += ["--contexts", contexts]
    common_flags += ["--questions", str(questions)]
    # The baseline MUST land first (every variant's delta reads it).
    assert "base_noprefix" in variant_ids
    _run(
        _py("slot_eval", "--mode", "icl_panel", "--variant", "base_noprefix", *common_flags),
        gpu=gpus[0],
        name="icl_base_noprefix",
    )
    rest = [v for v in variant_ids if v != "base_noprefix"]
    jobs = [
        (
            f"icl_{v}",
            [_py("slot_eval", "--mode", "icl_panel", "--variant", v, *common_flags)],
        )
        for v in rest
    ]
    _run_jobs(jobs, gpus)


def check_gate1(*, smoke: bool) -> dict:
    """Gate 1 (plan §7): ICL K=8 chain A source-cell mean ΔG >= +2 nat over Q_test."""
    _phase("gate1")
    path = ns_eval_dir(smoke) / "icl_panel" / "icl_K8_chainA.json"
    payload = json.loads(path.read_text())
    dose = payload["contexts"][SOURCE_CONTEXT]["mean_delta_logp"]
    result = {
        "meta": repro_metadata(),
        "gate": "gate1_icl_dose",
        "source_mean_delta_logp": dose,
        "threshold_nats": GATE1_MIN_NATS,
        "pass": dose >= GATE1_MIN_NATS,
    }
    write_json(ns_eval_dir(smoke) / "gate1.json", result)
    if not result["pass"]:
        raise RuntimeError(
            f"GATE 1 FAIL: ICL K=8 chain A source ΔG = {dose:+.3f} nat < +{GATE1_MIN_NATS} — "
            "do NOT run the matched-strength design (plan §7 salvage path; orchestrator decides)."
        )
    logger.info("GATE 1 PASS: source ΔG = %+.3f nat", dose)
    return result


def _train_job(
    run_id: str,
    *,
    smoke: bool,
    epochs: int | None,
    out_root: str | None,
    questions: int,
    traj_questions: int,
) -> tuple[str, list[list[str]]]:
    """One run's full pipeline: train -> ft_run_pipeline -> crosscheck -> persist-prune."""
    suffix = "_smoke" if smoke else ""
    smoke_flag = ["--smoke"] if smoke else []
    out_flags = ["--out-root", out_root] if out_root else []
    train_cmd = _py(
        "train_runs",
        "train",
        "--run",
        run_id,
        "--gpu",
        "0",  # placeholder; rewritten per assigned worker GPU at exec time
        *(["--epochs", str(epochs)] if epochs else []),
        *out_flags,
        *(["--run-name-suffix", suffix] if suffix else []),
    )
    pipeline_cmd = _py(
        "slot_eval",
        "--mode",
        "ft_run_pipeline",
        "--run",
        run_id,
        "--questions",
        str(questions),
        "--traj-questions",
        str(traj_questions),
        *out_flags,
        *smoke_flag,
    )
    crosscheck_cmd = _py(
        "slot_eval",
        "--mode",
        "inloop_crosscheck",
        "--run",
        run_id,
        "--suffix",
        suffix,
        *out_flags,
        *smoke_flag,
    )
    persist_cmd = _py("train_runs", "persist-prune", "--run", run_id, *out_flags, *smoke_flag)
    return (run_id, [train_cmd, pipeline_cmd, crosscheck_cmd, persist_cmd])


def phase_train(
    gpus: list[int],
    *,
    runs: list[str],
    smoke: bool = False,
    epochs: int | None = None,
    out_root: str | None = None,
    questions: int = 50,
    traj_questions: int = 25,
    skip_persist: bool = False,
) -> None:
    _phase("train" if not smoke else "smoke_train")
    jobs = []
    for run_id in runs:
        name, cmds = _train_job(
            run_id,
            smoke=smoke,
            epochs=epochs,
            out_root=out_root,
            questions=questions,
            traj_questions=traj_questions,
        )
        if skip_persist:
            cmds = cmds[:-1]
        jobs.append((name, cmds))

    # Rewrite the --gpu placeholder per assigned worker: wrap _run_jobs with a
    # per-job gpu injection by replacing the value after '--gpu' at exec time.
    q: queue.Queue[tuple[str, list[list[str]]]] = queue.Queue()
    for job in jobs:
        q.put(job)
    errors: list[str] = []
    lock = threading.Lock()

    def worker(gpu: int) -> None:
        while True:
            try:
                name, cmds = q.get_nowait()
            except queue.Empty:
                return
            try:
                for i, cmd in enumerate(cmds):
                    cmd = list(cmd)
                    if "--gpu" in cmd:
                        cmd[cmd.index("--gpu") + 1] = str(gpu)
                    _run(cmd, gpu=gpu, name=f"{name}/{i}")
            except Exception as e:
                with lock:
                    errors.append(f"{name}: {e}")
            finally:
                q.task_done()

    threads = [threading.Thread(target=worker, args=(g,), daemon=True) for g in gpus]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if errors:
        raise RuntimeError("train-phase job failures:\n" + "\n".join(errors))
    # Single-threaded summary assembly AFTER all workers join — match_run
    # writes only per-run files (matched_pairs/by_run/), so no concurrent
    # writer ever touches the shared matched_summary.json (round-2 race fix).
    from explore_persona_space.experiments.icl_vs_ft_491.matching import assemble_matched_summary

    assemble_matched_summary(smoke=smoke)


def _all_free_gen_cells() -> list[str]:
    from explore_persona_space.experiments.icl_vs_ft_491.free_gen import list_cells

    return list(list_cells())


def phase_free_gen(
    gpus: list[int],
    *,
    cells: list[str],
    smoke: bool = False,
    out_root: str | None = None,
    questions: int = 50,
    contexts: str | None = None,
    skip_own_policy: bool = False,
) -> None:
    _phase("free_gen" if not smoke else "smoke_free_gen")
    smoke_flag = ["--smoke"] if smoke else []
    out_flags = ["--out-root", out_root] if out_root else []
    ctx_flags = ["--contexts", contexts] if contexts else []
    n_workers = min(2, len(gpus), len(cells))  # plan §8: 2 vLLM workers
    shards: list[list[str]] = [cells[i::n_workers] for i in range(n_workers)]
    jobs = [
        (
            f"free_gen_shard{i}",
            [
                _py(
                    "free_gen",
                    "--cells",
                    ",".join(shard),
                    "--questions",
                    str(questions),
                    *ctx_flags,
                    *out_flags,
                    *smoke_flag,
                )
            ],
        )
        for i, shard in enumerate(shards)
        if shard
    ]
    _run_jobs(jobs, gpus[:n_workers])
    if not skip_own_policy:
        _run(
            _py(
                "slot_eval",
                "--mode",
                "own_policy",
                "--questions",
                str(questions),
                *out_flags,
                *smoke_flag,
            ),
            gpu=gpus[0],
            name="own_policy",
        )


def _all_act_variants() -> list[str]:
    from explore_persona_space.experiments.icl_vs_ft_491.activations import list_act_variants

    return list(list_act_variants())


def phase_activations(
    gpus: list[int],
    *,
    variants: list[str],
    smoke: bool = False,
    out_root: str | None = None,
    questions: int = 20,
) -> None:
    _phase("activations" if not smoke else "smoke_activations")
    smoke_flag = ["--smoke"] if smoke else []
    out_flags = ["--out-root", out_root] if out_root else []
    # 'base' must land first: summarize + every shift needs it, and sharded
    # workers would race it otherwise.
    if "base" in variants:
        _run(
            _py(
                "activations",
                "extract",
                "--variants",
                "base",
                "--questions",
                str(questions),
                *out_flags,
                *smoke_flag,
            ),
            gpu=gpus[0],
            name="act_base",
        )
    rest = [v for v in variants if v != "base"]
    jobs = [
        (
            f"act_{v}",
            [
                _py(
                    "activations",
                    "extract",
                    "--variants",
                    v,
                    "--questions",
                    str(questions),
                    *out_flags,
                    *smoke_flag,
                )
            ],
        )
        for v in rest
    ]
    _run_jobs(jobs, gpus)
    if not smoke:
        _run(_py("activations", "summarize"), name="act_summarize")


def phase_upload(*, smoke: bool = False) -> None:
    """Upload datasets + analysis tensors + raw completions BEFORE termination."""
    _phase("upload" if not smoke else "smoke_upload")
    if not os.environ.get("HF_TOKEN"):
        raise RuntimeError("HF_TOKEN missing from env — refusing to run the upload phase")
    from explore_persona_space.experiments.icl_vs_ft_491.activations import ACT_DIR
    from explore_persona_space.orchestrate.hub import (
        upload_dataset_directory,
        upload_raw_completions_to_data_repo,
    )

    manifest: dict = {"meta": repro_metadata(), "uploads": {}}
    bucket = f"{HF_BUCKET_491}/smoke" if smoke else HF_BUCKET_491

    # 1) Datasets: chains + registries + train rows + helpful demos +
    #    band-stop trajectories (durable per-step four-float records).
    uploaded = upload_dataset_directory(DATA_DIR, bucket, pattern="*.json")
    uploaded += upload_dataset_directory(
        DATA_DIR / "train_rows", f"{bucket}/train_rows", pattern="*.jsonl"
    )
    traj_dir = DATA_DIR / "trajectories"
    if traj_dir.exists() and any(traj_dir.iterdir()):
        uploaded += upload_dataset_directory(traj_dir, f"{bucket}/trajectories", pattern="*.json")
    manifest["uploads"]["datasets"] = uploaded

    # 2) Analysis tensors (plan-referenced downstream inputs, #521 rule).
    if ACT_DIR.exists() and any(ACT_DIR.iterdir()):
        tensor_files = upload_dataset_directory(
            ACT_DIR, f"{bucket}/analysis_tensors", pattern="*.pt"
        )
        tensor_files += upload_dataset_directory(
            ACT_DIR, f"{bucket}/analysis_tensors", pattern="*.json"
        )
        manifest["uploads"]["analysis_tensors"] = tensor_files

    # 3) Raw completions (per-cell raw_completions.json layout under the
    #    eval-results dir — the helper's recursive glob picks them up).
    results = upload_raw_completions_to_data_repo(
        experiment_name=bucket, eval_results_dir=ns_eval_dir(smoke)
    )
    manifest["uploads"]["raw_completions"] = results

    # 4) Non-raw eval-JSON tree (round-2 hardening): ONE recursive folder
    #    upload that preserves subdirectories (icl_panel/, ft_panel/,
    #    matched_pairs/ incl. by_run/, free_gen/, own_policy/, gate1.json),
    #    so off-pod analysis is self-sufficient even if the canonical
    #    git-on-issue-branch sync misfires. free_gen_raw/ is excluded (step 3
    #    uploads it per file); the smoke/ namespace is excluded from the
    #    full-run upload (the smoke phase uploads it under its own bucket).
    from explore_persona_space.orchestrate import hub

    eval_root = ns_eval_dir(smoke)
    ignore = ["free_gen_raw/*", "free_gen_raw/**", "*.tmp", "*.log"]
    if not smoke:
        ignore += ["smoke/*", "smoke/**"]
    eval_tree_url = hub._upload(
        local_path=eval_root,
        repo_id=hub.DEFAULT_DATASET_REPO,
        repo_type="dataset",
        path_in_repo=f"{bucket}/eval_json",
        ignore_patterns=ignore,
    )
    if not eval_tree_url:
        raise RuntimeError(
            f"eval-JSON tree upload FAILED: {eval_root} -> "
            f"{hub.DEFAULT_DATASET_REPO}/{bucket}/eval_json (fail-loud, never skip)"
        )
    manifest["uploads"]["eval_json_tree"] = eval_tree_url
    write_json(ns_eval_dir(smoke) / "upload_manifest.json", manifest)


# ── Sentinel (poll_pipeline contract) ────────────────────────────────────


def write_sentinel(note: str, *, sentinel_dir: Path, kind: str = "epm:results") -> Path:
    payload = {
        "sentinel_schema_version": 1,
        "kind": kind,
        "version": 1,
        "note": note,
        "task_id": 491,
        "by": "i491.dispatch",
        "ts": time.time(),
    }
    sentinel_dir.mkdir(parents=True, exist_ok=True)
    kind_slug = kind.replace(":", "_")
    path = sentinel_dir / f"issue-491-{kind_slug}-{int(time.time())}.json"
    path.write_text(json.dumps(payload, indent=2))
    logger.info("sentinel written: %s", path)
    return path


def _results_note(smoke: bool) -> str:
    parts = [f"issue 491 dispatch complete (smoke={smoke})."]
    gate = ns_eval_dir(smoke) / "gate1.json"
    if gate.exists():
        g = json.loads(gate.read_text())
        parts.append(f"gate1 source ΔG={g['source_mean_delta_logp']:+.3f} nat pass={g['pass']}.")
    from explore_persona_space.experiments.icl_vs_ft_491.matching import load_matched_pairs

    try:
        pairs = load_matched_pairs(smoke=smoke)
    except FileNotFoundError:
        pairs = {}  # note-builder only: matching never ran in this invocation
    if pairs:
        within = sum(1 for p in pairs.values() if p["within_tolerance"])
        parts.append(f"matched pairs: {len(pairs)} ({within} within ±1.5 nat).")
    parts.append(f"eval JSONs under {EVAL_DIR}; analysis runs off-pod (analyze.py).")
    return " ".join(parts)


# ── Smoke (the sweep with one cell — PASS_UNIFIED) ───────────────────────


def phase_smoke(gpus: list[int], *, out_root: str | None) -> None:
    _phase("smoke")
    smoke_root = str(Path(out_root or "adapters_491") / "smoke")
    # p1: data — full CPU build + GPU helpful-demo gen (same commands as sweep).
    phase_data(gpus)
    # p2: ICL reads — same dispatcher path, cell subset = baseline + K8 chain A,
    #     FULL 10-context panel (the smoke FT pipeline's trajectory/full reads
    #     consume the baseline for every context, and Gate 1 reads the source
    #     cell from the same files).
    phase_icl_eval(gpus[:1], smoke=True, questions=SMOKE_QUESTIONS)
    # p3: Gate 1 (blocking).
    check_gate1(smoke=True)
    # p4: train pipeline — ONE cell to step 12 through the same per-run
    #     pipeline (train -> matching-basis -> match -> traj -> full ->
    #     #534 crosscheck -> persist-prune incl. a real HF upload).
    phase_train(
        gpus[:1],
        runs=["ft_K8_chainA"],
        smoke=True,
        epochs=SMOKE_TRAIN_EPOCHS,
        out_root=smoke_root,
        questions=SMOKE_QUESTIONS,
        traj_questions=5,
    )
    # p5: free_gen — same dispatcher path, 2 cells x source context x 3 q,
    #     then the own-policy reads on those outputs.
    phase_free_gen(
        gpus[:1],
        cells=["ft_K8_chainA", "icl_K8_chainA"],
        smoke=True,
        out_root=smoke_root,
        questions=SMOKE_FREE_GEN_QUESTIONS,
        contexts=SOURCE_CONTEXT,
        skip_own_policy=True,  # own_policy needs all 10 contexts; smoke covers the read path below
    )
    _run(
        _py(
            "slot_eval",
            "--mode",
            "own_policy",
            "--questions",
            str(SMOKE_FREE_GEN_QUESTIONS),
            "--contexts",
            SOURCE_CONTEXT,
            "--out-root",
            smoke_root,
            "--smoke",
        ),
        gpu=gpus[0],
        name="smoke_own_policy",
    )
    # p6: activations — base + one ICL variant at 2 questions.
    phase_activations(
        gpus[:1],
        variants=["base", "act_icl_K8_chainA"],
        smoke=True,
        out_root=smoke_root,
        questions=SMOKE_ACT_QUESTIONS,
    )
    # p7: upload — the same helper calls against the smoke namespace.
    phase_upload(smoke=True)
    logger.info("SMOKE COMPLETE — all phases exercised through the sweep code path")


# ── CLI ──────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> None:  # noqa: C901 — phase router
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(name)s [%(levelname)s] %(message)s"
    )
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--phase",
        required=True,
        choices=[
            "smoke",
            "data",
            "icl_eval",
            "train",
            "traj_eval",
            "matched_eval",
            "free_gen",
            "activations",
            "upload",
            "all",
        ],
    )
    ap.add_argument("--gpus", default="0", help="comma-separated physical GPU ids")
    ap.add_argument("--runs", default=None, help="train: run subset (default: all 13)")
    ap.add_argument("--cells", default=None, help="free_gen: cell subset (default: all 29)")
    ap.add_argument("--variants", default=None, help="activations: variant subset")
    ap.add_argument("--out-root", default=os.environ.get("EPM_491_OUT_ROOT"))
    ap.add_argument("--smoke", action="store_true", help="smoke namespace for single phases")
    ap.add_argument("--skip-gpu-datagen", action="store_true")
    ap.add_argument("--sentinel-dir", default="/workspace/logs")
    ap.add_argument("--no-sentinel", action="store_true")
    args = ap.parse_args(argv)

    # uv run python does NOT auto-load .env; the dispatcher is the credential
    # boundary for every subprocess env it constructs (#397 round-10').
    from explore_persona_space.orchestrate.env import load_dotenv

    load_dotenv()
    for key in ("HF_TOKEN", "WANDB_API_KEY"):
        if not os.environ.get(key):
            raise RuntimeError(f"{key} missing after load_dotenv() — fix .env before launch")

    gpus = [int(g) for g in args.gpus.split(",")]
    from explore_persona_space.experiments.icl_vs_ft_491.data_build import load_run_specs

    try:
        if args.phase == "smoke":
            phase_smoke(gpus, out_root=args.out_root)
        elif args.phase == "data":
            phase_data(gpus, skip_gpu_datagen=args.skip_gpu_datagen)
        elif args.phase == "icl_eval":
            phase_icl_eval(gpus, smoke=args.smoke)
        elif args.phase == "train":
            runs = args.runs.split(",") if args.runs else list(load_run_specs())
            phase_train(gpus, runs=runs, smoke=args.smoke, out_root=args.out_root)
        elif args.phase in ("traj_eval", "matched_eval"):
            # Recovery entry points: re-run the per-run eval pipeline for runs
            # whose outputs are missing. run_ft_pipeline GUARDS against
            # post-prune re-matching (it refuses when persist_prune_meta.json
            # exists or the on-disk ckpts differ from train_meta's grid), so a
            # recovery invocation can never rebuild the registered matching
            # basis from a pruned 1-2-point curve; for pruned runs the
            # persisted matched_pairs/by_run/<run_id>.json entry is the source
            # of truth and downstream reads consume it directly.
            runs = args.runs.split(",") if args.runs else list(load_run_specs())
            jobs = []
            for run_id in runs:
                flags = ["--smoke"] if args.smoke else []
                if args.out_root:
                    flags += ["--out-root", args.out_root]
                jobs.append(
                    (
                        run_id,
                        [_py("slot_eval", "--mode", "ft_run_pipeline", "--run", run_id, *flags)],
                    )
                )
            _run_jobs(jobs, gpus)
            from explore_persona_space.experiments.icl_vs_ft_491.matching import (
                assemble_matched_summary,
            )

            assemble_matched_summary(smoke=args.smoke)
        elif args.phase == "free_gen":
            cells = args.cells.split(",") if args.cells else _all_free_gen_cells()
            phase_free_gen(gpus, cells=cells, smoke=args.smoke, out_root=args.out_root)
        elif args.phase == "activations":
            variants = args.variants.split(",") if args.variants else _all_act_variants()
            phase_activations(gpus, variants=variants, smoke=args.smoke, out_root=args.out_root)
        elif args.phase == "upload":
            phase_upload(smoke=args.smoke)
        elif args.phase == "all":
            phase_smoke(gpus, out_root=args.out_root)
            phase_icl_eval(gpus)
            check_gate1(smoke=False)
            phase_train(gpus, runs=list(load_run_specs()), out_root=args.out_root)
            phase_free_gen(gpus, cells=_all_free_gen_cells(), out_root=args.out_root)
            phase_activations(gpus, variants=_all_act_variants(), out_root=args.out_root)
            phase_upload()
    except Exception:
        # Crash sentinel so the poller surfaces the failure class fast; the
        # exception still propagates (fail loud — the traceback IS the signal).
        if not args.no_sentinel:
            write_sentinel(
                f"issue 491 dispatch FAILED in phase={args.phase} — see log tail",
                sentinel_dir=Path(args.sentinel_dir),
                kind="epm:failure",
            )
        raise

    if args.phase == "all" and not args.no_sentinel:
        write_sentinel(_results_note(smoke=False), sentinel_dir=Path(args.sentinel_dir))
        _phase("done")
    elif args.phase == "all":
        _phase("done")


if __name__ == "__main__":
    main()
