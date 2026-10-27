"""Hub I/O for the issue #2674 core (Phase 1): stage the Phase 0 pool, upload per run.

Subcommands:
  stage-pool    Pull the Phase 0 teacher pool (cat/neutral x train/heldout) from the HF
                data repo into ``<out>/pool/``. The pool sits there as <9 MB line-split
                parts plus a manifest; each staged file is checked against the
                manifest's whole-file sha256 and line count, so the core trains on the
                byte-identical pool the gate validated.
  upload-run    After one run's train + sampled eval: its metric/eval JSONs and raw
                completions to the data repo, its adapters to the model repo, both in
                one folder commit each, then an exact-set verify. Writes
                ``sentinels/upload_run_<run>.done.json`` only after the verify passes.
  upload-final  Logs, sentinels, every metric JSON and raw completion in one folder
                commit, then an exact-set verify of the data side AND of every run's
                adapters on the model repo. Prints any out-root file covered by neither.

Layout (``<sub>`` is ``core``, or ``core_smoke`` for the smoke leg), all under
``issue2674_sl_epochs/<sub>/``: metric and eval JSONs mirror the out-root's relative
paths (``runs/``, ``sampled_eval/``, ``metrics/``); rollout text goes to
``raw_completions/sampled_eval/<run>/...`` (the upload policy's raw_completions/
prefix, same shape as Phase 0's); adapters go to ``<run>/step_NNNNNN/`` on the model
repo. The core reuses the Phase 0 base-model eval, which stays at the experiment root.
Every upload runs through ``hub._upload_folder_filtered`` (file-count guards, overflow
fallback, exact-set verify).
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import issue2674_common as C  # noqa: E402  (calls load_dotenv before any HF import)

POOL_FILES = ("cat_train", "cat_heldout", "neutral_train", "neutral_heldout")
POOL_HF_DIR = f"{C.EXPERIMENT}/phase0_residue/pool"
ADAPTER_IGNORE = ["checkpoint-*", "*/checkpoint-*/*"]


@contextlib.contextmanager
def _upload_lock():
    """Serialize this host's uploads: queue jobs of equal length finish together."""
    path = Path(tempfile.gettempdir()) / "issue2674_hf_upload.lock"
    with path.open("w") as fh:
        t0 = time.monotonic()
        fcntl.flock(fh, fcntl.LOCK_EX)
        waited = time.monotonic() - t0
        if waited > 1:
            print(f"[upload] waited {waited:.0f}s for the upload lock", flush=True)
        try:
            yield
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _count_lines(path: Path) -> int:
    with path.open(encoding="utf-8") as fh:
        return sum(1 for _ in fh)


# --------------------------------------------------------------------------------------
# stage-pool
# --------------------------------------------------------------------------------------


def stage_pool(out_root: str | Path) -> dict:
    """Stage the 4 pool files and assert each matches its manifest's source sha + lines."""
    from explore_persona_space.orchestrate.hub import stage_hub_file, stage_sharded_text

    pool_dir = C.ensure_out_dirs(out_root)["pool"]
    report: dict = {}
    for name in POOL_FILES:
        with tempfile.TemporaryDirectory(dir=pool_dir, prefix=".manifest_") as td:
            man_path = stage_hub_file(
                C.HF_DATA_REPO,
                f"{POOL_HF_DIR}/{name}.manifest.json",
                Path(td) / f"{name}.manifest.json",
                repo_type="dataset",
            )
            manifest = json.loads(Path(man_path).read_text(encoding="utf-8"))
        want_sha = manifest["source_sha256"]
        want_lines = int(manifest["source_lines"])
        target = pool_dir / f"{name}.jsonl"
        if target.is_file() and _sha256(target) == want_sha:
            print(f"[stage-pool] {name}: already staged, sha matches", flush=True)
        else:
            if target.exists():
                target.unlink()
            stage_sharded_text(
                C.HF_DATA_REPO, f"{POOL_HF_DIR}/{name}.jsonl", target, repo_type="dataset"
            )
            got_sha = _sha256(target)
            if got_sha != want_sha:
                raise RuntimeError(f"{name}: staged sha {got_sha} != Phase 0 source sha {want_sha}")
        n = _count_lines(target)
        if n != want_lines:
            raise RuntimeError(f"{name}: {n} lines != manifest source_lines {want_lines}")
        report[name] = {"sha256": want_sha, "lines": n}
        print(f"[stage-pool] {name}: OK lines={n} sha={want_sha[:12]}", flush=True)
    need = max(u for _, u, _ in C.CORE_CONDITIONS)
    for teacher in C.TEACHERS:
        if report[f"{teacher}_train"]["lines"] < need:
            raise RuntimeError(f"{teacher} train pool is smaller than the largest core U {need}")
    return report


# --------------------------------------------------------------------------------------
# upload helpers
# --------------------------------------------------------------------------------------


def _local_selection(root: Path, allow: list[str], ignore: list[str]) -> list[str]:
    """The exact relative paths ``upload_folder`` will select with these patterns."""
    from huggingface_hub.utils import filter_repo_objects

    rel = sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file())
    return list(filter_repo_objects(rel, allow_patterns=allow, ignore_patterns=ignore))


#: Paths ``upload_folder`` always skips; mirrored so the local selection stays exact.
_HUB_DEFAULT_SKIPS = [".git/*", "*/.git/*", ".cache/huggingface/*", "*/.cache/huggingface/*"]


def _upload_and_verify(
    *,
    root: Path,
    repo_id: str,
    repo_type: str,
    path_in_repo: str,
    allow: list[str],
    ignore: list[str],
) -> list[str]:
    """One guarded folder commit via ``hub._upload_folder_filtered``.

    The helper carries the per-directory file-count guard, the reactive repo-wide
    file-count overflow fallback, transport retries, and the exact-set verify against
    ``expected_repo_paths``. It returns "" on any failure, so this raises. Returns the
    uploaded paths, relative to ``root``.
    """
    from explore_persona_space.orchestrate.hub import (
        TRAINING_STATE_IGNORE_PATTERNS,
        _upload_folder_filtered,
    )

    files = _local_selection(
        root, allow, list(ignore) + list(TRAINING_STATE_IGNORE_PATTERNS) + _HUB_DEFAULT_SKIPS
    )
    if not files:
        raise RuntimeError(f"nothing selected under {root} for {repo_id}:{path_in_repo}")
    url = _upload_folder_filtered(
        local_dir=root,
        repo_id=repo_id,
        repo_type=repo_type,
        path_in_repo=path_in_repo,
        allow_patterns=allow,
        expected_repo_paths=[f"{path_in_repo}/{f}" for f in files],
        ignore_patterns=list(ignore),
    )
    if not url:
        raise RuntimeError(f"upload or exact-set verify failed: {repo_id}:{path_in_repo}")
    print(f"[upload] {url} verified {len(files)} file(s)", flush=True)
    return files


def _run_upload_legs(run: str, sub: str) -> list[tuple[str, str, str, str, list[str], list[str]]]:
    """(leg, local subdir of the out-root, repo, path_in_repo, allow, ignore) per upload.

    fnmatch '*' crosses '/', so "runs/<run>/*" also matches the adapters, which the
    metrics leg ignores. Rollout text lands under the experiment's raw_completions/
    prefix (upload policy); adapters go to the model repo.
    """
    data = f"{C.EXPERIMENT}/{sub}"
    return [
        (
            "metrics",
            ".",
            C.HF_DATA_REPO,
            data,
            [f"runs/{run}/*", f"sampled_eval/{run}/*", f"metrics/sampled_eval_{run}.json"],
            [f"runs/{run}/adapters/*"],
        ),
        ("raw", "raw", C.HF_DATA_REPO, f"{data}/raw_completions", [f"sampled_eval/{run}/*"], []),
        (
            "adapters",
            f"runs/{run}/adapters",
            C.HF_MODEL_REPO,
            f"{data}/{run}",
            ["*"],
            list(ADAPTER_IGNORE),
        ),
    ]


def _expected_adapter_steps(run_dir: Path) -> set[int]:
    meta = C.read_json(run_dir / "run_meta.json")
    steps = meta.get("adapter_save_steps")
    if not steps:
        raise RuntimeError(f"{run_dir / 'run_meta.json'} has no adapter_save_steps")
    return {int(s) for s in steps}


# --------------------------------------------------------------------------------------
# upload-run / upload-final
# --------------------------------------------------------------------------------------


def upload_run(out_root: str | Path, run: str, sub: str) -> dict:
    paths = C.out_paths(out_root)
    root = paths["root"]
    run_dir = paths["runs"] / run
    for need in (run_dir / "run.done.json", paths["sampled"] / run / "eval.done.json"):
        if not need.is_file():
            raise RuntimeError(f"upload-run {run}: {need} missing (train/eval not complete)")
    adapters = run_dir / "adapters"
    on_disk = sorted(p.name for p in adapters.glob("step_*") if p.is_dir())
    for s in on_disk:
        if not (adapters / s / "adapter_config.json").is_file():
            raise RuntimeError(f"{adapters / s} has no adapter_config.json — incomplete save")
    # Planned-vs-actual: the run must have saved EVERY planned checkpoint, not just some.
    got = {int(s.split("_", 1)[1]) for s in on_disk}
    want = _expected_adapter_steps(run_dir)
    if got != want:
        raise RuntimeError(
            f"upload-run {run}: adapters on disk {sorted(got)} != planned {sorted(want)}"
        )

    counts: dict[str, int] = {}
    with _upload_lock():
        for leg, sub_dir, repo, prefix, allow, ignore in _run_upload_legs(run, sub):
            files = _upload_and_verify(
                root=paths["root"] / sub_dir,
                repo_id=repo,
                repo_type="model" if repo == C.HF_MODEL_REPO else "dataset",
                path_in_repo=prefix,
                allow=allow,
                ignore=ignore,
            )
            counts[leg] = len(files)
    payload = {"run": run, "sub": sub, "files": counts, "adapter_steps": sorted(got)}
    C.write_sentinel(out_root, f"upload_run_{run}", payload)
    print(f"[upload-run] {run} DONE {counts}", flush=True)
    return payload


def _covered_by_runs(root: Path, sub: str) -> tuple[set[str], dict[tuple[str, str], list[str]]]:
    """Out-root-relative paths the per-run legs cover, and the HF paths per (repo, prefix)."""
    from explore_persona_space.orchestrate.hub import TRAINING_STATE_IGNORE_PATTERNS

    covered: set[str] = set()
    expected: dict[tuple[str, str], list[str]] = {}
    for run_dir in sorted(p for p in (root / "runs").glob("*") if p.is_dir()):
        for _leg, sub_dir, repo, prefix, allow, ignore in _run_upload_legs(run_dir.name, sub):
            base = root / sub_dir
            if not base.is_dir():
                continue
            sel = _local_selection(
                base, allow, ignore + list(TRAINING_STATE_IGNORE_PATTERNS) + _HUB_DEFAULT_SKIPS
            )
            rel_base = "" if sub_dir == "." else f"{sub_dir}/"
            covered |= {f"{rel_base}{f}" for f in sel}
            # One scoped listing per repo: the experiment/sub prefix covers every leg.
            expected.setdefault((repo, f"{C.EXPERIMENT}/{sub}"), []).extend(
                f"{prefix}/{f}" for f in sel
            )
    return covered, expected


def upload_final(out_root: str | Path, sub: str) -> dict:
    """Upload logs, sentinels and any per-run residue; re-verify everything on both repos.

    The per-run legs already landed each run's files, so this commit carries only what
    they did not cover (no duplicate terminal mega-commit). ``logs/upload_final.log`` is
    still being written by this phase's tee, so it is left for the post-run sweep.
    """
    from huggingface_hub import HfApi

    from explore_persona_space.orchestrate.hub import verify_repo_paths_uploaded

    root = C.out_paths(out_root)["root"]
    covered, expected = _covered_by_runs(root, sub)
    local = sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file())
    skip = ("pool/", "logs/upload_final.log", "sentinels/upload_final.done.json")
    rest = [f for f in local if f not in covered and not f.startswith(skip)]
    # Weights never go to the data repo: an adapter file no per-run leg covered is a
    # real gap (e.g. a stray checkpoint dir), so name it instead of uploading it here.
    stray = [f for f in rest if "/adapters/" in f]
    rest = [f for f in rest if "/adapters/" not in f]
    for f in stray[:20]:
        print(f"[upload-final] UNCOVERED adapter file {f}", flush=True)
    with _upload_lock():
        files = _upload_and_verify(
            root=root,
            repo_id=C.HF_DATA_REPO,
            repo_type="dataset",
            path_in_repo=f"{C.EXPERIMENT}/{sub}",
            allow=rest,
            ignore=[],
        )
    print(f"[upload-final] residue commit: {len(files)} file(s)", flush=True)

    api = HfApi()
    for (repo, prefix), paths in sorted(expected.items()):
        rtype = "model" if repo == C.HF_MODEL_REPO else "dataset"
        missing = verify_repo_paths_uploaded(api, repo, paths, path_in_repo=prefix, repo_type=rtype)
        if missing:
            raise RuntimeError(f"{len(missing)}/{len(paths)} missing on {repo}: {missing[:10]}")
        print(f"[upload-final] re-verified {len(paths)} file(s) on {repo}:{prefix}", flush=True)
    return {
        "residue_files": len(files),
        "per_run_files": sum(len(v) for v in expected.values()),
        "uncovered_adapter_files": stray,
    }


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cmd", nargs="?", choices=["stage-pool", "upload-run", "upload-final"])
    ap.add_argument("--out", type=str, default=None, help="Run out-root.")
    ap.add_argument("--run-name", type=str, default=None)
    ap.add_argument("--hf-subdir", type=str, default=C.HF_CORE_SUBDIR)
    ap.add_argument("--import-check", action="store_true")
    return ap


def _import_check() -> int:
    from explore_persona_space.orchestrate.argcheck import assert_args_attributes_defined

    assert_args_attributes_defined(__file__, str(Path(__file__).with_name("issue2674_common.py")))
    from huggingface_hub.utils import filter_repo_objects  # noqa: F401

    from explore_persona_space.orchestrate.hub import (  # noqa: F401
        TRAINING_STATE_IGNORE_PATTERNS,
        _upload_folder_filtered,
        stage_hub_file,
        stage_sharded_text,
        verify_repo_paths_uploaded,
    )

    # The pattern logic is pure; pin it here so a pattern regression fails the smoke.
    tree = [
        "runs/r1/trajectory.json",
        "runs/r1/adapters/step_000038/adapter_model.safetensors",
        "runs/r10/trajectory.json",
        "sampled_eval/r1/step_000038.json",
        "raw/sampled_eval/r1/step_000038/shard_000/raw_completions.json",
        "raw/sampled_eval/r10/step_000038/shard_000/raw_completions.json",
        "metrics/sampled_eval_r1.json",
        "logs/train_job00.log",
    ]
    got = {}
    for leg, sub_dir, _repo, prefix, allow, ignore in _run_upload_legs("r1", "core"):
        rel_base = "" if sub_dir == "." else f"{sub_dir}/"
        local = [p[len(rel_base) :] for p in tree if p.startswith(rel_base)]
        sel = list(filter_repo_objects(local, allow_patterns=allow, ignore_patterns=ignore))
        got[leg] = [f"{prefix}/{f}" for f in sel]
    want = {
        "metrics": [
            "issue2674_sl_epochs/core/runs/r1/trajectory.json",
            "issue2674_sl_epochs/core/sampled_eval/r1/step_000038.json",
            "issue2674_sl_epochs/core/metrics/sampled_eval_r1.json",
        ],
        "raw": [
            "issue2674_sl_epochs/core/raw_completions/sampled_eval/r1/step_000038/"
            "shard_000/raw_completions.json"
        ],
        "adapters": ["issue2674_sl_epochs/core/r1/step_000038/adapter_model.safetensors"],
    }
    assert got == want, got
    print("[import-check] OK", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.import_check:
        return _import_check()
    if args.cmd is None or args.out is None:
        raise SystemExit("a subcommand and --out are required")
    if args.cmd == "stage-pool":
        stage_pool(args.out)
    elif args.cmd == "upload-run":
        if not args.run_name:
            raise SystemExit("upload-run needs --run-name")
        upload_run(args.out, args.run_name, args.hf_subdir)
    else:
        upload_final(args.out, args.hf_subdir)
    return 0


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(rc)
