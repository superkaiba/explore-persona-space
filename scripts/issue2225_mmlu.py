#!/usr/bin/env python
"""Issue #2225 Phase 2c — MMLU capability eval over the 86 eval targets.

Plan §4.6 item 2: lm-eval-harness ``mmlu``, 0-shot, FULL question set, vLLM
backend, IDENTICAL setting across every arm + base (the read is within-run
relative capability). Adapter path (plan §12 A9, fact-checked on the installed
lm_eval 0.4.11): ``--model vllm`` with model_args
``lora_local_path=<adapter>,max_lora_rank=32`` — ``max_lora_rank`` is MANDATORY
(the library default 16 refuses the r=32 adapters; ``enable_lora`` is derived
by lm_eval from ``lora_local_path`` and must NOT be passed twice). ONE adapter
per lm-eval invocation; invocations are sharded across GPUs via the shared
CVD-pinned work-stealing fan-out (``issue2225_eval_gen.fan_out_subprocesses``).

MERGE FALLBACK (``--merge-fallback``, expected unnecessary): merge the adapter
onto base on CPU (bf16 ``merge_and_unload`` — the delete-after-eval recipe;
the merged dir is DERIVED data, regenerable from base + adapter, so deleting
it never violates upload-before-delete — the ADAPTER is never deleted here),
eval the merged dir, delete it in a ``finally``. Merges are staggered through
a slot-file semaphore (≤4 concurrent, plan §9 disk row) with a stale-slot
reclaim (dead-pid detection) + a 20 GB headroom assert per merge.

TARGETS: the same 86-target enumeration as P2b
(``issue2225_eval_gen.build_eval_targets`` — 81 cells + 4 reused #778
baselines + base); adapters resolve local-ckpt-first then HF
(``issue2225_eval_gen.resolve_adapter``).

RESUME (#952 shape): skip a target iff ``<out-root>/mmlu/<tag>.json`` exists
AND its stored fingerprint (adapter sha / task / num_fewshot / model /
lm_eval version) matches. Outputs: one canonical JSON per model under
``<out-root>/mmlu/`` (+ the raw lm-eval results file mirrored under
``mmlu/raw/``); ``--upload`` pushes ``mmlu/`` to the HF data repo
(``issue2225_ctxsteer/mmlu``) as ONE folder commit.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path

# scripts/ on sys.path so the sibling issue778_* / issue2225_* modules resolve
# in script mode (the #823 sys.path[0] trap). Heavy imports stay deferred.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from explore_persona_space.orchestrate.env import load_dotenv

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("issue2225.mmlu")
load_dotenv()

import issue2225_eval_gen as evalgen  # targets + adapter resolution + fan-out
import issue2225_train as train  # sha/git helpers (cheap import)
import issue778_lib as lib  # constants + phase logging (cheap import)

# ── constants (identical across every arm + base — plan §4.6 item 2) ──────────

DATA_REPO = "superkaiba1/explore-persona-space-data"
MMLU_HF_PREFIX = "issue2225_ctxsteer/mmlu"

TASK = "mmlu"
NUM_FEWSHOT = 0
SEED = 0
MAX_LORA_RANK = 32  # MANDATORY: lm_eval's default 16 refuses the r=32 adapters (A9)
GPU_MEMORY_UTILIZATION = 0.85
MAX_MODEL_LEN = 4096

MAX_CONCURRENT_MERGES = 4  # plan §9 disk row (merged Qwen-7B ~15 GB each)
MERGE_HEADROOM_GB = 20.0
MERGE_SLOT_POLL_S = 20.0


# ── fingerprint + resume ─────────────────────────────────────────────────────


def _lm_eval_version() -> str:
    import lm_eval

    return getattr(lm_eval, "__version__", "?")


def mmlu_fingerprint(
    target: evalgen.EvalTarget, adapter_path: Path | None, model: str, *, limit: int | None = None
) -> dict:
    """Resume-compared fingerprint (recipe fields + adapter sha; no code SHA).

    ``limit`` (lm-eval ``--limit``) is part of the key: a P0 ``--limit 200``
    probe run must NEVER resume-satisfy P2c's full-set run (g3 Major 1).
    """
    if adapter_path is None:
        adapter_sha = "base-no-adapter"
    else:
        adapter_sha = train._sha256(adapter_path / "adapter_model.safetensors")
    return {
        "tag": target.tag,
        "adapter_sha256": adapter_sha,
        "task": TASK,
        "num_fewshot": NUM_FEWSHOT,
        "seed": SEED,
        "model": model,
        "max_lora_rank": MAX_LORA_RANK,
        "limit": limit,
        "lm_eval_version": _lm_eval_version(),
    }


def mmlu_out_path(out_root: Path, tag: str) -> Path:
    return out_root / "mmlu" / f"{tag}.json"


def _done(out_path: Path, fingerprint: dict) -> bool:
    if not out_path.exists():
        return False
    try:
        with open(out_path, encoding="utf-8") as f:
            stored = json.load(f)
    except (json.JSONDecodeError, OSError, UnicodeDecodeError) as e:
        logger.warning("[resume] unreadable output %s (%s) -> re-run", out_path, e)
        return False
    return stored.get("fingerprint") == fingerprint


# ── merge fallback (staggered, delete-after-eval) ────────────────────────────


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


@contextmanager
def merge_slot(slot_dir: Path, *, max_slots: int = MAX_CONCURRENT_MERGES) -> Iterator[int]:
    """File-lock semaphore staggering concurrent merges across worker processes.

    Slots are O_CREAT|O_EXCL lock files carrying the holder pid; a slot whose
    holder pid is dead is reclaimed (same-machine subprocesses, so pid liveness
    is a valid staleness probe).
    """
    slot_dir.mkdir(parents=True, exist_ok=True)
    while True:
        for i in range(max_slots):
            slot = slot_dir / f"slot{i}.lock"
            if slot.exists():
                try:
                    holder = int(slot.read_text().strip() or "0")
                except (OSError, ValueError):
                    holder = 0
                if holder and not _pid_alive(holder):
                    logger.warning("[merge-slot] reclaiming stale slot %s (pid %d)", slot, holder)
                    slot.unlink(missing_ok=True)
                elif not holder:
                    # Holder died between O_EXCL open and the pid write: an
                    # empty/unparseable slot older than 10 min is stale — an
                    # unreclaimed one permanently narrows the semaphore
                    # (g3 minor). Fresh empty slots (the open->write window)
                    # are left alone.
                    try:
                        age_s = time.time() - slot.stat().st_mtime
                    except OSError:
                        age_s = 0.0
                    if age_s > 600:
                        logger.warning(
                            "[merge-slot] reclaiming pid-less stale slot %s (age %.0fs)",
                            slot,
                            age_s,
                        )
                        slot.unlink(missing_ok=True)
            try:
                fd = os.open(slot, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                continue
            with os.fdopen(fd, "w") as f:
                f.write(str(os.getpid()))
            try:
                yield i
            finally:
                slot.unlink(missing_ok=True)
            return
        time.sleep(MERGE_SLOT_POLL_S)


def merge_adapter(adapter_path: Path, model_name: str, dest: Path) -> Path:
    """CPU bf16 merge (base + adapter -> full checkpoint dir) for the fallback.

    CPU deliberately: the GPU is about to be claimed by the lm-eval vLLM engine.
    bf16 merge is the delete-after-eval recipe's standard precision; the merged
    dir is derived data and is deleted by the caller after the eval.
    """
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if dest.exists():
        shutil.rmtree(dest)  # a partial prior merge is garbage — rebuild
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(model, str(adapter_path))
    model = model.merge_and_unload()
    model.save_pretrained(dest)
    AutoTokenizer.from_pretrained(model_name).save_pretrained(dest)
    del model
    return dest


# ── single-model lm-eval invocation (subprocess) ─────────────────────────────


def _model_args(pretrained: str | Path, adapter_path: Path | None) -> str:
    parts = [
        f"pretrained={pretrained}",
        "dtype=bfloat16",
        f"gpu_memory_utilization={GPU_MEMORY_UTILIZATION}",
        f"max_model_len={MAX_MODEL_LEN}",
    ]
    if adapter_path is not None:
        # enable_lora is derived by lm_eval from lora_local_path (A9) — never
        # passed explicitly (a duplicate engine kwarg).
        parts += [f"lora_local_path={adapter_path}", f"max_lora_rank={MAX_LORA_RANK}"]
    return ",".join(parts)


def _lm_eval_cmd(model_args: str, scratch: Path, limit: int | None) -> list[str]:
    """Compose the lm-eval argv (separate helper so tests can pin the --limit
    threading without a GPU; g3 Major 1)."""
    cmd = [
        "uv",
        "run",
        "python",
        "-m",
        "lm_eval",
        "--model",
        "vllm",
        "--model_args",
        model_args,
        "--tasks",
        TASK,
        "--num_fewshot",
        str(NUM_FEWSHOT),
        "--batch_size",
        "auto",
        "--seed",
        str(SEED),
        "--output_path",
        str(scratch),
    ]
    if limit is not None:
        cmd += ["--limit", str(limit)]
    return cmd


def _run_lm_eval(model_args: str, scratch: Path, log_prefix: str, *, limit: int | None) -> Path:
    """Run one lm-eval invocation into an empty scratch dir; return the results JSON."""
    if scratch.exists():
        shutil.rmtree(scratch)  # exactly-one-results-file invariant
    scratch.mkdir(parents=True, exist_ok=True)
    cmd = _lm_eval_cmd(model_args, scratch, limit)
    lim = "full set" if limit is None else f"--limit {limit}"
    print(f"[{log_prefix}] lm-eval start ({TASK}, {NUM_FEWSHOT}-shot, {lim})", flush=True)
    t0 = time.time()
    proc = subprocess.run(cmd, env={**os.environ})
    if proc.returncode != 0:
        raise RuntimeError(f"lm-eval exited rc={proc.returncode} for {log_prefix}")
    results = sorted(scratch.rglob("results_*.json"))
    if len(results) != 1:
        raise RuntimeError(f"expected exactly 1 results file under {scratch}, got {len(results)}")
    print(f"[{log_prefix}] lm-eval done elapsed={round(time.time() - t0, 1)}s", flush=True)
    return results[0]


def _resolve_single_target(tag: str) -> evalgen.EvalTarget:
    """Registry lookup with the resolve_targets cell fallback (fu1 seam).

    The parent 86-target registry wins; a non-registry tag resolves through
    ``evalgen.resolve_targets`` (train.resolve_cell — fu1 cells + §7 scaled
    slugs). Unknown/non-canonical tags still fail loud with a legible error."""
    return evalgen.resolve_targets([tag])[0]


def run_single(args) -> None:
    """Eval ONE target (subprocess mode; GPU pinned by the launcher env)."""
    target = _resolve_single_target(args.single)
    out_root = Path(args.out_root)
    model_name = args.model or lib.MODEL_NAME
    adapter = evalgen.resolve_adapter(
        target, ckpt_root=Path(args.ckpt_root), staging_dir=Path(args.staging_dir)
    )
    fp = mmlu_fingerprint(target, adapter, model_name, limit=args.limit)
    out_path = mmlu_out_path(out_root, target.tag)
    if _done(out_path, fp):
        print(f"[mmlu] skip {target.tag} (resume)", flush=True)
        return

    scratch = out_root / "mmlu" / "lmeval_scratch" / target.tag
    merged_dir: Path | None = None
    try:
        if args.merge_fallback and adapter is not None:
            from explore_persona_space.orchestrate.preflight import assert_out_root_headroom

            with merge_slot(out_root / "mmlu" / ".merge_slots"):
                assert_out_root_headroom(out_root, MERGE_HEADROOM_GB, phase="mmlu-merge")
                merged_dir = merge_adapter(
                    adapter, model_name, out_root / "mmlu" / "merged" / target.tag
                )
                results_path = _run_lm_eval(
                    _model_args(merged_dir, None), scratch, f"mmlu:{target.tag}", limit=args.limit
                )
        else:
            results_path = _run_lm_eval(
                _model_args(model_name, adapter), scratch, f"mmlu:{target.tag}", limit=args.limit
            )
    finally:
        if merged_dir is not None and merged_dir.exists():
            shutil.rmtree(merged_dir)  # delete-after-eval (derived data)

    with open(results_path, encoding="utf-8") as f:
        raw = json.load(f)
    mmlu_row = raw["results"][TASK]
    payload = {
        "model_tag": target.tag,
        "kind": target.kind,
        "dataset": target.dataset,
        "task": TASK,
        "num_fewshot": NUM_FEWSHOT,
        "limit": args.limit,
        "mmlu_acc": mmlu_row.get("acc,none"),
        "mmlu_acc_stderr": mmlu_row.get("acc_stderr,none"),
        "adapter_path": (None if adapter is None else str(adapter)),
        "via_merge_fallback": bool(args.merge_fallback and adapter is not None),
        "model_args": _model_args(
            merged_dir if merged_dir is not None else model_name,
            None if merged_dir is not None else adapter,
        ),
        "fingerprint": fp,
        "reproducibility": lib.repro_metadata(),
    }
    if payload["mmlu_acc"] is None:
        raise RuntimeError(f"mmlu acc missing from {results_path} for {target.tag}")
    # Persist the raw lm-eval results beside the canonical row (persist-by-default).
    raw_dir = out_root / "mmlu" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(results_path, raw_dir / f"{target.tag}.json")
    shutil.rmtree(scratch, ignore_errors=True)
    evalgen._atomic_write_json(out_path, payload)
    print(f"[mmlu] {target.tag} acc={payload['mmlu_acc']:.4f} -> {out_path}", flush=True)


# ── fan-out over targets (one lm-eval invocation per GPU slot) ───────────────


def run_fan_out(args) -> None:
    if args.targets:
        wanted = [s.strip() for s in args.targets.split(",") if s.strip()]
        # Registry lookup with the resolve_cell fallback (fu1 seam) — fu1 cells
        # evaluate without a registry edit; unknown tags still fail loud.
        targets = evalgen.resolve_targets(wanted)
    else:
        targets = evalgen.build_eval_targets()
    if args.smoke:
        targets = [t for t in targets if t.kind == "base"] or targets[:1]

    if args.dry_run:
        n_gpus = args.n_gpus or 8
    else:
        n_gpus = train._detect_gpu_count(cpu_only=False)
        if args.n_gpus:
            n_gpus = max(1, min(n_gpus, args.n_gpus))

    out_root = Path(args.out_root)

    def build_cmd(tag: str, gpu_id: int) -> list[str]:
        cmd = [
            "uv",
            "run",
            "python",
            str(Path(__file__).resolve()),
            "--single",
            tag,
            "--gpu-id",
            str(gpu_id),
            "--out-root",
            str(out_root),
            "--ckpt-root",
            str(args.ckpt_root),
            "--staging-dir",
            str(args.staging_dir),
        ]
        if args.model:
            cmd += ["--model", args.model]
        if args.merge_fallback:
            cmd += ["--merge-fallback"]
        if args.limit is not None:
            cmd += ["--limit", str(args.limit)]
        return cmd

    if args.dry_run:
        for i, t in enumerate(targets):
            g = i % n_gpus
            print(f"[mmlu][dry-run] CUDA_VISIBLE_DEVICES={g} {' '.join(build_cmd(t.tag, g))}")
        return

    lib.log_phase("mmlu", f"fan-out {len(targets)} targets over {n_gpus} GPUs")
    evalgen._prestage_base_model(args.model or lib.MODEL_NAME)
    evalgen.fan_out_subprocesses(
        [t.tag for t in targets],
        build_cmd,
        n_gpus=n_gpus,
        log_dir=out_root / "logs" / "mmlu",
        label="mmlu",
    )
    lib.log_phase("mmlu", "fan-out complete")


# Parent-default-identical seam: parent #2225 call sites pass no prefix and must keep it.
# UPLOAD_PREFIX_EXEMPT: parent-default-identical seam; fu1 threads its own via --hf-prefix
def upload_mmlu(out_root: Path, hf_prefix: str = MMLU_HF_PREFIX, hf_repo: str = DATA_REPO) -> str:
    """Upload the canonical + raw MMLU JSONs as ONE folder commit (P3).
    Follow-up rounds thread their OWN prefix (fu1: issue2225_ctxsteer/fu1_mmlu
    — never the parent-clobbering default, #1452) and, when routed off the
    canonical data repo (fu2's #2287 overflow routing), their OWN ``hf_repo``."""
    from explore_persona_space.orchestrate.hub import _upload

    local = out_root / "mmlu"
    if not local.exists():
        raise FileNotFoundError(f"nothing to upload: {local} absent")
    # Scratch/slot dirs are transient — refuse to ship them.
    for transient in ("lmeval_scratch", ".merge_slots", "merged"):
        residue = local / transient
        if residue.exists():
            shutil.rmtree(residue)
    url = _upload(local, hf_repo, "dataset", hf_prefix, raise_on_error=True)
    print(f"[mmlu] uploaded {local} -> {url}", flush=True)
    return url


# ── CLI ──────────────────────────────────────────────────────────────────────


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Issue #2225 Phase 2c MMLU capability eval.")
    ap.add_argument("--out-root", default="data/issue_2225/p2b_out")
    ap.add_argument("--ckpt-root", default="checkpoints/issue_2225")
    ap.add_argument("--staging-dir", default="data/issue_2225/hf_dl/eval_adapters")
    ap.add_argument("--targets", default=None, help="comma-separated target tags (default: all)")
    ap.add_argument("--model", default=None, help="base model (default: issue778_lib.MODEL_NAME)")
    ap.add_argument("--single", default=None, help="eval ONE target by tag (subprocess mode)")
    ap.add_argument("--gpu-id", type=int, default=0)
    ap.add_argument("--n-gpus", type=int, default=None, help="fan-out width cap")
    ap.add_argument("--merge-fallback", action="store_true", help="merge->eval->delete path")
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="lm-eval --limit N (the §4.8(b) P0 probe dial; threaded into the "
        "resume fingerprint so a limited run never resume-satisfies a full run)",
    )
    ap.add_argument("--smoke", action="store_true", help="base target only")
    ap.add_argument("--dry-run", action="store_true", help="print invocations, no CUDA")
    ap.add_argument("--upload", action="store_true", help="upload-only mode (pod-side, later)")
    # UPLOAD_PREFIX_EXEMPT: parent-default-identical seam — issue2225's own dispatcher calls
    # this flag-less and must keep the parent prefix; fu1 rounds pass an explicit --hf-prefix.
    ap.add_argument(
        "--hf-prefix",
        default=MMLU_HF_PREFIX,
        help="HF prefix for the MMLU upload (fu rounds thread issue2225_ctxsteer/fu1_mmlu)",
    )
    # UPLOAD_PREFIX_EXEMPT: parent-default-identical seam — fu2 threads the
    # overflow repo (#2287); parent/fu1 keep the canonical data repo.
    ap.add_argument(
        "--hf-repo",
        default=DATA_REPO,
        help="HF dataset repo for the MMLU upload (fu2 threads the overflow repo)",
    )
    ap.add_argument("--import-check", action="store_true")
    return ap


def main(argv: Sequence[str] | None = None) -> None:
    args = build_argparser().parse_args(argv)

    if args.import_check:
        from explore_persona_space.orchestrate.argcheck import assert_args_attributes_defined

        assert_args_attributes_defined(__file__)
        # Execute every deferred import the production paths reach (#606).
        import lm_eval  # noqa: F401
        import lm_eval.models.vllm_causallms  # noqa: F401  (the vLLM backend A9 relies on)
        from peft import PeftModel  # noqa: F401
        from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: F401

        from explore_persona_space.orchestrate.hub import _upload  # noqa: F401
        from explore_persona_space.orchestrate.preflight import (  # noqa: F401
            assert_out_root_headroom,
        )

        evalgen.build_eval_targets()  # asserts 86 / 67 / 19 / 124
        print("[issue2225-mmlu] import-check OK", flush=True)
        raise SystemExit(0)

    if args.upload:
        # UPLOAD_PREFIX_EXEMPT: parent-default-identical seam; fu1 passes an explicit --hf-prefix
        upload_mmlu(Path(args.out_root), hf_prefix=args.hf_prefix, hf_repo=args.hf_repo)
        sys.stdout.flush()
        sys.exit(0)

    if args.single:
        run_single(args)
    else:
        run_fan_out(args)
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(0)


if __name__ == "__main__":
    main()
