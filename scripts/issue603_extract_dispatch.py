#!/usr/bin/env python3
"""#603 pod-side dispatcher — shift extraction over the 21 reused adapters.

Smoke IS sweep with a subset: ``--cells 1 --personas 2 --questions 4``
runs the IDENTICAL dispatcher -> subprocess -> worker -> ``.pt`` ->
HF-upload path on the lowest-prior fact adapter (cells are ordered fact
by ascending teacher prior then seed, then refusal, then EM). EVERY
phase's cell list derives from the same ``--cells``/``--families``
subset: p1 extraction iterates the selected cells, the per-cell upload
fires per selected cell the moment it completes (checkpoint-per-cell),
p2 priors derives its (family, source) list FROM the selected refusal/EM
cells, and p3 upload-verify re-enumerates exactly the files the selected
cells produced.

Phases (main log; ``[phase=done]`` is RESERVED for the single terminal
line):

- ``[phase=p0_inputs]``   — load frozen family JSONs, build the cell list
- ``[phase=p1_extract]``  — round-robin the cells over the visible GPUs
  (one worker subprocess per GPU at a time, ``CUDA_VISIBLE_DEVICES``
  sharding), per-cell logs under ``<out-dir>/../logs/``; each completed
  cell's ``.pt`` + manifest + responses sidecar uploads IMMEDIATELY
  (403-quota fallback chain: data repo -> private -> overflow).
  Resume skip is MANIFEST-VALIDATED, never existence-only: a cell on
  disk is reused only when its manifest matches THIS invocation's
  source / persona set / question count / layers / probe SHA /
  max_new_tokens; a mismatching artifact (e.g. a prior ``--personas 2
  --questions 4`` smoke under production filenames) is recomputed and
  re-uploaded, so the on-pod smoke can never poison the full sweep.
- ``[phase=p2_priors]``   — source-self log-prob priors for the
  refusal/EM sources present in the SELECTED cells
  (``scripts/issue603_source_prior.py`` subprocess, 1 GPU) + upload
- ``[phase=p3_upload_verify]`` — fail-loud ``list_repo_files``
  re-enumeration of every expected remote file
- results sentinel ``/workspace/logs/issue-603-epm_results-<epoch>.json``
  (poll_pipeline contract: sentinel_schema_version/kind/version), THEN
  the terminal ``[phase=done]``.

Pod-side: NEVER shells out to ``scripts/task.py`` (CLAUDE.md rule).

Full run (pod, 8x H100)::

    nohup uv run python scripts/issue603_extract_dispatch.py \
        >> /workspace/logs/issue-603-extract.log 2>&1 &

Smoke (pod, first cell, identical path)::

    uv run python scripts/issue603_extract_dispatch.py \
        --cells 1 --personas 2 --questions 4 --prior-rows 4

VM dry-run (no GPU, no upload; full plumbing + sentinel)::

    uv run python scripts/issue603_extract_dispatch.py \
        --dry-run --no-upload --sentinel-dir /tmp/i603_sentinels
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from _bootstrap import PROJECT_ROOT, bootstrap

logger = bootstrap(log_name="i603_extract_dispatch")

FAMILY_ORDER = ("fact", "refusal", "em")
REPO_CHAIN = (
    "superkaiba1/explore-persona-space-data",
    "superkaiba1/explore-persona-space-data-private",
    "superkaiba1/explore-persona-space-overflow",
)
DEFAULT_UPLOAD_PREFIX = "issue603_p3prime_write_decomposition/analysis_tensors"
SENTINEL_SCHEMA_VERSION = 1


def _git_commit() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL,
                env={**os.environ},
            )
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def _build_cells(inputs_dir: Path, families: list[str], n_cells: int) -> list[dict]:
    """Deterministic cell order: fact by (prior asc, seed asc), refusal, em."""
    cells: list[dict] = []
    for family in FAMILY_ORDER:
        if family not in families:
            continue
        payload = json.loads((inputs_dir / f"{family}_panel.json").read_text())
        fam_cells = list(payload["cells"])
        if family == "fact":
            fam_cells.sort(key=lambda c: (c["prior_logprob"], c["seed"]))
        for c in fam_cells:
            c["inputs_json"] = str(inputs_dir / f"{family}_panel.json")
        cells.extend(fam_cells)
    if n_cells > 0:
        cells = cells[:n_cells]
    if not cells:
        raise RuntimeError(f"no cells selected (families={families}, n_cells={n_cells})")
    return cells


def _gpu_list(arg: str) -> list[str]:
    """Resolve the GPU id list: --gpus > CUDA_VISIBLE_DEVICES > nvidia-smi count."""
    if arg:
        return [g.strip() for g in arg.split(",") if g.strip()]
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if cvd:
        return [g.strip() for g in cvd.split(",") if g.strip()]
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            env={**os.environ},  # epm-lint: subprocess-env-inherit -- nvidia-smi probe
        )
        return [line.strip() for line in out.decode().splitlines() if line.strip()]
    except Exception:
        return ["0"]


def _upload_with_fallback(local: Path, path_in_repo: str) -> str:
    """Upload one file; on the account-wide 403 storage quota, fall through the
    pre-registered repo chain (data -> private -> overflow). Fail-loud when the
    chain is exhausted. Returns the repo_id the file landed on."""
    from huggingface_hub import HfApi
    from huggingface_hub.errors import HfHubHTTPError

    api = HfApi(token=os.environ["HF_TOKEN"])
    last_err: Exception | None = None
    for repo_id in REPO_CHAIN:
        try:
            api.upload_file(
                path_or_fileobj=str(local),
                path_in_repo=path_in_repo,
                repo_id=repo_id,
                repo_type="dataset",
            )
            logger.info("[upload] %s -> %s/%s", local.name, repo_id, path_in_repo)
            return repo_id
        except HfHubHTTPError as e:
            status = getattr(getattr(e, "response", None), "status_code", None)
            msg = str(e)
            if status == 403 or "storage" in msg.lower():
                logger.warning(
                    "[upload] %s -> %s blocked (%s); falling through the repo chain",
                    local.name,
                    repo_id,
                    status,
                )
                last_err = e
                continue
            raise
    raise RuntimeError(f"upload of {local} failed on every repo in the fallback chain: {last_err}")


def _verify_remote(upload_manifest: dict[str, str]) -> None:
    """Fail-loud list_repo_files re-enumeration of every uploaded file."""
    from huggingface_hub import list_repo_files

    by_repo: dict[str, list[str]] = {}
    for path_in_repo, repo_id in upload_manifest.items():
        by_repo.setdefault(repo_id, []).append(path_in_repo)
    for repo_id, paths in by_repo.items():
        remote = set(list_repo_files(repo_id, repo_type="dataset"))
        missing = [p for p in paths if p not in remote]
        if missing:
            raise RuntimeError(f"upload verification FAILED: {repo_id} missing {missing}")
        logger.info("[verify] %s: %d/%d files present", repo_id, len(paths), len(paths))


def _write_sentinel(sentinel_dir: Path, kind: str, note: str, *, gate: str = "") -> Path:
    """poll_pipeline.py end-of-run sentinel (required keys + kind-slug filename)."""
    sentinel_dir.mkdir(parents=True, exist_ok=True)
    kind_slug = kind.replace(":", "_")
    path = sentinel_dir / f"issue-603-{kind_slug}-{int(time.time())}.json"
    payload = {
        "sentinel_schema_version": SENTINEL_SCHEMA_VERSION,
        "kind": kind,
        "version": 1,
        "task_id": 603,
        "by": "issue603_extract_dispatch",
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "gate": gate,
        "blocks_pipeline": kind == "epm:failure",
        "note": note,
    }
    with path.open("w") as f:
        json.dump(payload, f, indent=2)
    logger.info("[sentinel] wrote %s", path)
    return path


def _worker_cmd(cell: dict, args: argparse.Namespace) -> list[str]:
    out_pt = Path(args.out_dir) / f"{cell['cell_id']}.pt"
    resp_json = Path(args.out_dir) / f"{cell['cell_id']}_responses.json"
    cmd = [
        "uv",
        "run",
        "python",
        "scripts/issue603_extract_worker.py",
        "--cell-id",
        cell["cell_id"],
        "--family",
        cell["family"],
        "--source",
        cell["source"],
        "--seed",
        str(cell["seed"]),
        "--adapter-repo",
        cell["adapter_repo"],
        "--adapter-subfolder",
        cell["adapter_subfolder"],
        "--inputs-json",
        cell["inputs_json"],
        "--out",
        str(out_pt),
        "--responses-out",
        str(resp_json),
        "--layers",
        *[str(x) for x in args.layers],
        "--primary-layer",
        str(args.primary_layer),
        "--max-new-tokens",
        str(args.max_new_tokens),
    ]
    if args.personas > 0:
        cmd += ["--n-personas", str(args.personas)]
    if args.questions > 0:
        cmd += ["--n-questions", str(args.questions)]
    return cmd


def _cell_artifacts(cell: dict, out_dir: Path) -> list[Path]:
    stem = cell["cell_id"]
    return [
        out_dir / f"{stem}.pt",
        out_dir / f"{stem}.manifest.json",
        out_dir / f"{stem}_responses.json",
    ]


def _expected_subset(
    panel_names: list[str], n_probes: int, source: str, args: argparse.Namespace
) -> tuple[list[str], int]:
    """The persona set + question count THIS invocation expects per cell.

    Mirrors the worker's smoke-subset rule (issue603_extract_worker.py):
    first N panel personas with the source swapped into the last slot when
    absent; first N probes. 0 = full panel / full probe set.
    """
    names = list(panel_names)
    if args.personas > 0:
        names = names[: args.personas]
        if source not in names:
            names[-1] = source
    n_q = min(args.questions, n_probes) if args.questions > 0 else n_probes
    return names, n_q


def _resume_status(
    cell: dict, out_dir: Path, args: argparse.Namespace, inputs_cache: dict[str, dict]
) -> tuple[str, str]:
    """Manifest-validated resume check: 'complete' | 'stale' | 'missing' + reason.

    A cell resume-skips ONLY when all three artifacts exist AND the cell's
    manifest matches the CURRENT invocation's expectations (source, persona
    set, question count, layers, probe SHA, max_new_tokens). Anything else
    recomputes — bare file existence is NOT completion: a `--cells 1
    --personas 2 --questions 4` smoke writes production filenames, and an
    existence-only skip would reuse (and re-upload) that partial artifact as
    the full cell (#603 round-1 blocker `smoke-artifact-poisons-full-sweep`).
    """
    paths = _cell_artifacts(cell, out_dir)
    missing = [p.name for p in paths if not p.exists()]
    if missing:
        return "missing", f"absent: {missing}"
    try:
        manifest = json.loads(paths[1].read_text())
    except Exception as e:  # corrupt manifest -> recompute, never silently skip
        return "stale", f"unreadable manifest {paths[1].name}: {type(e).__name__}: {e}"
    if cell["inputs_json"] not in inputs_cache:
        inputs_cache[cell["inputs_json"]] = json.loads(Path(cell["inputs_json"]).read_text())
    inputs = inputs_cache[cell["inputs_json"]]
    exp_names, exp_n_q = _expected_subset(
        list(inputs["panel"]), len(inputs["probes"]), cell["source"], args
    )
    checks = {
        "source": (manifest.get("source"), cell["source"]),
        "n_personas": (manifest.get("n_personas"), len(exp_names)),
        "persona_names": (sorted(manifest.get("persona_names") or []), sorted(exp_names)),
        "n_questions": (manifest.get("n_questions"), exp_n_q),
        "layers": (manifest.get("layers"), list(args.layers)),
        "probe_sha256": (manifest.get("probe_sha256"), inputs["probe_sha256"]),
        "max_new_tokens": (manifest.get("max_new_tokens"), args.max_new_tokens),
    }
    mismatches = [
        f"{k}: manifest={got!r} != expected={want!r}"
        for k, (got, want) in checks.items()
        if got != want
    ]
    if mismatches:
        return "stale", "; ".join(mismatches)
    return "complete", "manifest matches current invocation"


def main() -> int:  # noqa: C901 — phased pod-side driver, long by nature
    """Run the 21-cell extraction sweep (or a --cells subset, same path)."""
    ap = argparse.ArgumentParser(description="#603 extraction dispatcher (smoke == sweep)")
    ap.add_argument("--families", default="fact,refusal,em")
    ap.add_argument("--cells", type=int, default=0, help="First N cells (0 = all 21).")
    ap.add_argument("--personas", type=int, default=0, help="Smoke: N panel personas per cell.")
    ap.add_argument("--questions", type=int, default=0, help="Smoke: N probes per cell.")
    ap.add_argument("--prior-rows", type=int, default=0, help="Smoke: N prior rows per source.")
    ap.add_argument("--gpus", default="", help="Comma GPU ids (default: CVD env / nvidia-smi).")
    ap.add_argument("--inputs-dir", default="eval_results/issue_603/inputs")
    ap.add_argument("--out-dir", default="eval_results/issue_603/shifts")
    ap.add_argument("--layers", type=int, nargs="+", default=[7, 14, 21])
    ap.add_argument("--primary-layer", type=int, default=14)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--upload", dest="upload", action="store_true", default=True)
    ap.add_argument("--no-upload", dest="upload", action="store_false")
    ap.add_argument("--upload-prefix", default=DEFAULT_UPLOAD_PREFIX)
    ap.add_argument("--sentinel-dir", default="/workspace/logs")
    ap.add_argument("--skip-priors", action="store_true", help="Skip the p2 priors phase.")
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Build cells + print worker commands + sentinel; no GPU work, no upload.",
    )
    args = ap.parse_args()

    if args.upload and not args.dry_run:
        assert os.environ.get("HF_TOKEN"), "HF_TOKEN missing — .env not loaded?"

    t_start = time.time()
    families = [f.strip() for f in args.families.split(",") if f.strip()]
    inputs_dir = Path(args.inputs_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = out_dir.parent / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    sentinel_dir = Path(args.sentinel_dir)

    upload_manifest: dict[str, str] = {}  # path_in_repo -> repo_id
    failures: list[dict] = []
    inputs_cache: dict[str, dict] = {}  # inputs_json path -> parsed payload
    running: dict[str, tuple[subprocess.Popen, dict, object]] = {}

    def _upload_cell(cell: dict) -> None:
        if not args.upload or args.dry_run:
            return
        for local in _cell_artifacts(cell, out_dir):
            if not local.exists():
                raise FileNotFoundError(f"cell {cell['cell_id']}: expected artifact {local}")
            path_in_repo = f"{args.upload_prefix}/shifts/{local.name}"
            upload_manifest[path_in_repo] = _upload_with_fallback(local, path_in_repo)

    try:
        logger.info("[phase=p0_inputs] families=%s cells=%d", families, args.cells)
        cells = _build_cells(inputs_dir, families, args.cells)
        logger.info(
            "[phase=p0_inputs] %d cells selected: %s",
            len(cells),
            [c["cell_id"] for c in cells],
        )

        gpus = _gpu_list(args.gpus)
        logger.info("[phase=p1_extract] sharding %d cells over GPUs %s", len(cells), gpus)

        pending = list(cells)
        done_cells: list[str] = []

        def _start(gpu: str, cell: dict) -> None:
            # Manifest-validated resume (runs in dry-run too, so a CPU
            # resume simulation exercises the REAL skip/recompute branch).
            status, why = _resume_status(cell, out_dir, args, inputs_cache)
            if status == "complete":
                logger.info(
                    "[phase=p1_extract] cell %s already complete on disk (%s) — re-upload only",
                    cell["cell_id"],
                    why,
                )
                _upload_cell(cell)
                done_cells.append(cell["cell_id"])
                return
            if status == "stale":
                logger.warning(
                    "[phase=p1_extract] cell %s artifacts on disk DO NOT match this "
                    "invocation (%s) — queued for recompute (overwrite + re-upload)",
                    cell["cell_id"],
                    why,
                )
            if args.dry_run:
                logger.info(
                    "[phase=p1_extract] DRY-RUN cell=%s gpu=%s resume_status=%s cmd: %s",
                    cell["cell_id"],
                    gpu,
                    status,
                    " ".join(_worker_cmd(cell, args)),
                )
                done_cells.append(cell["cell_id"])
                return
            log_path = logs_dir / f"{cell['cell_id']}.log"
            log_f = log_path.open("a")
            proc = subprocess.Popen(
                _worker_cmd(cell, args),
                stdout=log_f,
                stderr=subprocess.STDOUT,
                env={**os.environ, "CUDA_VISIBLE_DEVICES": gpu},
                cwd=str(PROJECT_ROOT),
            )
            running[gpu] = (proc, cell, log_f)
            logger.info(
                "[phase=p1_extract] launched cell=%s gpu=%s pid=%d log=%s",
                cell["cell_id"],
                gpu,
                proc.pid,
                log_path,
            )

        # Prime one cell per GPU, then poll-and-refill (checkpoint-per-cell:
        # every completed cell uploads immediately inside the loop).
        while pending or running:
            for gpu in gpus:
                if gpu not in running and pending:
                    _start(gpu, pending.pop(0))
            if args.dry_run and not pending and not running:
                break
            time.sleep(5 if not args.dry_run else 0)
            for gpu in list(running):
                proc, cell, log_f = running[gpu]
                rc = proc.poll()
                if rc is None:
                    continue
                log_f.close()
                del running[gpu]
                if rc == 0:
                    logger.info("[phase=p1_extract] cell %s complete (rc=0)", cell["cell_id"])
                    _upload_cell(cell)
                    done_cells.append(cell["cell_id"])
                else:
                    tail = ""
                    log_path = logs_dir / f"{cell['cell_id']}.log"
                    if log_path.exists():
                        tail = "".join(log_path.read_text().splitlines(keepends=True)[-15:])
                    logger.error(
                        "[phase=p1_extract] cell %s FAILED rc=%d\n%s", cell["cell_id"], rc, tail
                    )
                    failures.append({"cell_id": cell["cell_id"], "rc": rc})

        if failures:
            raise RuntimeError(f"{len(failures)} cell(s) failed: {failures}")

        # p2 — priors for the refusal/EM sources present in the SELECTED cells
        # (per-phase subset threading: the source list derives from `cells`).
        prior_specs = sorted(
            {(c["family"], c["source"]) for c in cells if c["family"] in ("refusal", "em")}
        )
        priors_path = out_dir.parent / "source_priors.json"
        if args.skip_priors or not prior_specs:
            logger.info(
                "[phase=p2_priors] skipped (%s)",
                "--skip-priors" if args.skip_priors else "no refusal/em cells selected",
            )
        elif args.dry_run:
            logger.info("[phase=p2_priors] DRY-RUN specs=%s", prior_specs)
        else:
            prior_families = ",".join(sorted({f for f, _ in prior_specs}))
            prior_sources = ",".join(sorted({s for _, s in prior_specs}))
            cmd = [
                "uv",
                "run",
                "python",
                "scripts/issue603_source_prior.py",
                "--families",
                prior_families,
                "--sources",
                prior_sources,
                "--rows",
                str(args.prior_rows),
                "--out",
                str(priors_path),
            ]
            logger.info("[phase=p2_priors] %s", " ".join(cmd))
            log_path = logs_dir / "source_priors.log"
            with log_path.open("a") as log_f:
                rc = subprocess.run(
                    cmd,
                    stdout=log_f,
                    stderr=subprocess.STDOUT,
                    env={**os.environ, "CUDA_VISIBLE_DEVICES": gpus[0]},
                    cwd=str(PROJECT_ROOT),
                ).returncode
            if rc != 0:
                raise RuntimeError(f"source-prior phase failed rc={rc} (log: {log_path})")
            if args.upload:
                path_in_repo = f"{args.upload_prefix}/source_priors.json"
                upload_manifest[path_in_repo] = _upload_with_fallback(priors_path, path_in_repo)

        logger.info("[phase=p3_upload_verify] %d files to verify", len(upload_manifest))
        if upload_manifest:
            _verify_remote(upload_manifest)
            manifest_path = out_dir.parent / "upload_manifest.json"
            with manifest_path.open("w") as f:
                json.dump(
                    {
                        "git_commit": _git_commit(),
                        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        "files": upload_manifest,
                    },
                    f,
                    indent=2,
                )

        wall_min = (time.time() - t_start) / 60
        note = (
            f"#603 extraction sweep complete: {len(done_cells)} cell(s) "
            f"({', '.join(done_cells[:25])}), priors_specs={len(prior_specs)}, "
            f"{len(upload_manifest)} files uploaded+verified "
            f"(prefix {args.upload_prefix}), wall={wall_min:.1f} min, "
            f"commit={_git_commit()[:12]}, dry_run={args.dry_run}"
        )
        _write_sentinel(sentinel_dir, "epm:results", note)
        logger.info("[phase=done] %s", note)
        return 0
    except Exception as e:
        logger.exception("dispatcher failed")
        # Terminate still-running workers — a mid-loop failure (e.g. a
        # non-403 upload error raising out of _upload_cell) must not orphan
        # GPU worker subprocesses until pod termination.
        for gpu, (proc, cell, log_f) in list(running.items()):
            logger.warning(
                "[phase=fail] terminating worker cell=%s gpu=%s pid=%d",
                cell["cell_id"],
                gpu,
                proc.pid,
            )
            proc.terminate()
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                logger.warning("[phase=fail] kill -9 worker pid=%d", proc.pid)
                proc.kill()
            log_f.close()
        _write_sentinel(
            sentinel_dir,
            "epm:failure",
            f"failure_class: code\nreason: {type(e).__name__}: {e}",
        )
        logger.error("[phase=fail] %s", e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
