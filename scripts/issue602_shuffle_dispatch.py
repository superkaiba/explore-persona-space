#!/usr/bin/env python3
"""#602 follow-up `shuffled-replay-l27-control` pod dispatcher (plan v3 §3.3).

Minimal standalone launcher for the shuffled-replay token-integrity
control: downloads the 7 E1 base-generation JSONs + training mixes from
the Hub PINNED to the parent upload revision
(``bk.FOLLOWUP_SHUFFLE_INPUT_REVISION``), asserts the turner mix sha256
(``bk.TURNER_MIX_SHA256``) BEFORE any forward, then loops the 7 units
(em_turner shared + 6 em518 sources) SEQUENTIALLY on 1 GPU as
subprocesses of ``issue602_estimator_reads.py`` with
``--e1-only --e1-transforms intact shuffle mismatch --layers 14 27
--per-row-layers 14 27`` (checkpoint-per-unit), uploads in TWO passes
to ``bk.FOLLOWUP_SHUFFLE_BUCKET`` with the parent's per-file ``_upload``
+ quota-403 private-repo fallback + ``list_repo_files`` verification —
pass 1 the ``.pt`` payloads (whose post-upload sha becomes
``upload_revision`` inside every manifest), pass 2 the revision-bearing
``.manifest.json`` sidecars — records the FINAL ``handoff_revision``
(the scorer's ``--hf-revision``) into the results sentinel, and finishes
with the poll_pipeline sentinel followed by the single terminal
``[phase=done]`` line.

UNIFIED smoke/sweep architecture (PASS_UNIFIED contract): ``--smoke``
only re-parameterizes — unit subset (default em_turner + em518
assistant), ``--limit-rows 4``, stub ``--model-id``, ``--out-root`` off
the production path, and a ``_smoke``-suffixed upload prefix. EVERY
phase's worklist derives from the SAME ``--units`` subset: pin_inputs
downloads only the active units' files, estimators loops the active
units, upload walks whatever the active subset produced.

Pod-side contract: emits ``[phase=<name>]`` lines; the terminal
``[phase=done]`` appears exactly once, after the sentinel. NEVER shells
out to scripts/task.py (pod-side ban).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import issue602_extract_dispatch as parent_disp  # noqa: E402

from explore_persona_space.analysis import i602_bakeoff as bk  # noqa: E402

logger = logging.getLogger("issue602_shuffle_dispatch")

ALL_UNITS: tuple[str, ...] = (
    "em_turner__no_system",
    *(f"em518__{s}" for s in bk.SOURCES_518),
)
SMOKE_DEFAULT_UNITS: tuple[str, ...] = ("em_turner__no_system", "em518__assistant")
SMOKE_LIMIT_ROWS = 4


def resolve_units(spec: list[str] | None, smoke: bool) -> list[tuple[str, str]]:
    """Active (family, source) units — EVERY phase derives from this list."""
    names = list(spec) if spec else (list(SMOKE_DEFAULT_UNITS) if smoke else list(ALL_UNITS))
    missing = [n for n in names if n not in ALL_UNITS]
    if missing:
        raise SystemExit(f"unknown unit ids: {missing}; valid: {list(ALL_UNITS)}")
    return [tuple(n.split("__", 1)) for n in names]


def phase_pin_inputs(args: argparse.Namespace, units: list[tuple[str, str]]) -> dict[str, str]:
    """Download the active units' pinned inputs; assert the turner mix sha.

    Base-generation JSONs land under ``<out-root>/base_generations/`` (the
    reads subprocesses consume them via ``--base-generations-dir``); the
    training mixes are pre-fetched into the HF cache at the pinned
    revision (the subprocesses re-resolve the same pinned files) and the
    turner mix sha256 is asserted BEFORE any forward (plan v3 §3.3).
    """
    rev = args.hub_revision
    gen_dir = Path(args.out_root) / "base_generations"
    gen_dir.mkdir(parents=True, exist_ok=True)
    shas: dict[str, str] = {}
    for family, source in units:
        fname = f"e1__{family}__{source}__shared.json"
        local = gen_dir / fname
        if local.is_symlink() and not local.exists():
            local.unlink()
        if not local.exists():
            src = bk.hub_download(
                bk.DATA_REPO,
                f"{bk.HUB_BUCKET}/raw_completions/base_generations/{fname}",
                revision=rev,
            )
            local.symlink_to(src)
        shas[fname] = bk.sha256_file(local)
        logger.info("[phase=pin_inputs] %s pinned (rev %s)", fname, rev[:12])
    # pre-fetch the mixes at the pin + the turner sha assert (before any forward)
    if any(f == "em_turner" for f, _ in units):
        mix = bk.hub_download(
            bk.DATA_REPO,
            "issue521/training_mix/turner_bad_medical_advice_minus_pool_slice.jsonl",
            revision=rev,
        )
        sha = bk.sha256_file(mix)
        if sha != bk.TURNER_MIX_SHA256:
            raise RuntimeError(
                f"turner mix sha256 drift at pinned revision {rev}: got {sha}, expected "
                f"{bk.TURNER_MIX_SHA256} (parent provenance manifest) — refusing to forward"
            )
        shas["turner_mix.jsonl"] = sha
        logger.info("[phase=pin_inputs] turner mix sha256 asserted (%s…)", sha[:12])
    for family, source in units:
        if family == "em518":
            p = bk.hub_download(
                bk.DATA_REPO,
                f"issue518_leakage_prediction/training_pools/em/{source}/positives.jsonl",
                revision=rev,
            )
            shas[f"em518_{source}_positives.jsonl"] = bk.sha256_file(p)
    logger.info("[phase=pin_inputs] complete (%d files pinned)", len(shas))
    return shas


def phase_estimators(args: argparse.Namespace, units: list[tuple[str, str]]) -> list[Path]:
    """Sequential per-unit estimator-read subprocesses (checkpoint-per-unit)."""
    out_dir = Path(args.out_root) / "estimator_reads"
    gen_dir = Path(args.out_root) / "base_generations"
    log_dir = Path(args.out_root) / "logs"
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs: list[tuple[list[str], Path]] = []
    expected: list[Path] = []
    for family, source in units:
        out_path = out_dir / f"{family}__{source}.pt"
        expected.append(out_path)
        if out_path.exists():
            logger.info("[phase=estimators] %s/%s done — skip", family, source)
            continue
        cmd = [
            sys.executable,
            str(REPO / "scripts" / "issue602_estimator_reads.py"),
            "--family",
            family,
            "--source",
            source,
            "--model-id",
            args.model_id,
            "--layers",
            "14",
            "27",
            "--per-row-layers",
            "14",
            "27",
            "--e1-transforms",
            "intact",
            "shuffle",
            "mismatch",
            "--e1-only",
            "--hub-revision",
            args.hub_revision,
            "--base-generations-dir",
            str(gen_dir),
            "--out",
            str(out_path),
        ]
        if args.limit_rows:
            cmd += ["--limit-rows", str(args.limit_rows)]
        jobs.append((cmd, log_dir / f"estimators_{family}__{source}.log"))
    logger.info(
        "[phase=estimators] %d unit jobs (%d skipped), sequential on 1 GPU",
        len(jobs),
        len(units) - len(jobs),
    )
    if jobs:
        parent_disp._run_parallel(jobs, [args.gpu])
    for out_path in expected:
        if not out_path.exists():
            raise RuntimeError(f"estimator payload missing after run: {out_path}")
    logger.info("[phase=estimators] complete")
    return expected


def phase_upload(args: argparse.Namespace) -> dict[str, Any]:
    """Two-pass upload: payloads, then manifests CARRYING the payload revision.

    Parent conventions verbatim (per-file ``_upload``, quota-403 private
    fallback, per-destination-repo ``list_repo_files`` verification). Pass 1
    uploads the ``.pt`` payloads and queries the post-upload repo sha (the
    PAYLOAD revision); that sha is then written into every local
    ``.manifest.json`` BEFORE pass 2 uploads the manifests — so the HF-side
    sidecars carry ``upload_revision`` (round-4 fix
    `shuffle-upload-manifest-revision-missing`; the post-upload sha cannot
    live in a file that is part of its own upload commit). The sentinel
    records the FINAL post-manifest-pass ``handoff_revision`` — the value
    the orchestrator passes to the scorer as ``--hf-revision``, at which
    BOTH payloads and revision-bearing sidecars resolve (sentinel-primary
    contract, plan v3 §3.4). Smoke uploads go under a ``_smoke`` prefix so
    the production bucket is never polluted by stub payloads.
    """
    from huggingface_hub import HfApi, list_repo_files

    from explore_persona_space.orchestrate.hub import _upload

    out_dir = Path(args.out_root) / "estimator_reads"
    bucket = bk.FOLLOWUP_SHUFFLE_BUCKET + ("/_smoke" if args.smoke else "")
    all_files = [p for p in sorted(out_dir.iterdir()) if p.suffix in (".pt", ".json")]
    manifests = [p for p in all_files if p.name.endswith(".manifest.json")]
    payload_files = [p for p in all_files if p not in manifests]
    if not payload_files:
        raise RuntimeError("nothing to upload — estimator phase produced no files")
    state = {"repo": bk.DATA_REPO, "deviation": None}
    uploaded: list[tuple[str, str]] = []

    def _up(p: Path) -> None:
        path_in_repo = f"{bucket}/analysis_tensors/estimator_reads/{p.name}"
        try:
            _upload(p, state["repo"], "dataset", path_in_repo, upload_as_file=True)
        except Exception as e:  # quota-403 fallback (pre-registered deviation)
            msg = str(e)
            if "403" in msg or "storage" in msg.lower():
                logger.warning(
                    "[phase=upload] quota-403 on %s — falling back to PRIVATE repo "
                    "(pre-registered deviation, precedent #551)",
                    state["repo"],
                )
                state["repo"] = bk.PRIVATE_DATA_REPO
                state["deviation"] = "public-repo LFS quota 403 -> private data repo (same layout)"
                _upload(p, state["repo"], "dataset", path_in_repo, upload_as_file=True)
            else:
                raise
        uploaded.append((state["repo"], path_in_repo))

    # pass 1: payloads, then the payload-commit sha they landed in
    for p in payload_files:
        _up(p)
    payload_revision = HfApi().repo_info(state["repo"], repo_type="dataset").sha
    logger.info(
        "[phase=upload] %d payloads uploaded; payload revision %s",
        len(payload_files),
        payload_revision,
    )
    # pass 2: write the payload revision into every local manifest, THEN
    # upload them — HF-side sidecars carry the revision (scorer provenance)
    for p in manifests:
        m = json.loads(p.read_text())
        m["upload_repo"] = state["repo"]
        m["upload_revision"] = payload_revision
        m["upload_bucket"] = bucket
        p.write_text(json.dumps(m, indent=2))
        _up(p)
    by_repo: dict[str, list[str]] = {}
    for repo, path_in_repo in uploaded:
        by_repo.setdefault(repo, []).append(path_in_repo)
    missing: list[str] = []
    revisions: dict[str, str] = {}
    for repo, paths in by_repo.items():
        listing = set(list_repo_files(repo, repo_type="dataset"))
        missing += [f"{repo}::{u}" for u in paths if u not in listing]
        revisions[repo] = HfApi().repo_info(repo, repo_type="dataset").sha
    if missing:
        raise RuntimeError(f"upload verification FAILED — missing: {missing[:10]}")
    handoff_revision = revisions[state["repo"]]
    logger.info(
        "[phase=upload] %d files verified (%s); handoff revision (scorer --hf-revision): %s",
        len(uploaded),
        ", ".join(f"{r}: {len(ps)}" for r, ps in sorted(by_repo.items())),
        handoff_revision,
    )
    return {
        "repos": {r: len(ps) for r, ps in by_repo.items()},
        "n_files": len(uploaded),
        "plan_deviation": state["deviation"],
        "bucket": bucket,
        "revisions": revisions,
        "payload_revision": payload_revision,
        "handoff_revision": handoff_revision,
    }


def main() -> int:
    """Dispatcher entry: preflight -> pin_inputs -> estimators -> upload."""
    parser = argparse.ArgumentParser(
        description="#602 shuffled-replay-l27-control pod dispatcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--units",
        nargs="*",
        default=None,
        help=f"Unit subset (default: all {len(ALL_UNITS)}; smoke default: {SMOKE_DEFAULT_UNITS})",
    )
    parser.add_argument("--smoke", action="store_true", help="Tiny-slice re-parameterization")
    parser.add_argument("--model-id", default=bk.BASE_MODEL_ID)
    parser.add_argument("--gpu", default="0", help="CUDA device id for the sequential unit loop")
    parser.add_argument(
        "--hub-revision",
        default=bk.FOLLOWUP_SHUFFLE_INPUT_REVISION,
        help="Pinned data-repo revision for EVERY input download (plan v3 §3.3)",
    )
    parser.add_argument(
        "--out-root",
        default=str(bk.eval_dir(REPO) / bk.FOLLOWUP_SHUFFLE_SLUG),
        help="Output root (smoke should point this off the production path)",
    )
    parser.add_argument("--limit-rows", type=int, default=None, help="Smoke: cap E1 rows")
    parser.add_argument("--sentinel-dir", default="/workspace/logs")
    parser.add_argument("--skip-preflight", action="store_true")
    parser.add_argument("--skip-upload", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s :: %(message)s")
    # `uv run python` does NOT auto-load .env — without this the subprocess
    # env dicts ({**os.environ}) would lack HF_TOKEN (task #397 round-10').
    from explore_persona_space.orchestrate.env import load_dotenv

    load_dotenv()

    if args.smoke and args.limit_rows is None:
        args.limit_rows = SMOKE_LIMIT_ROWS

    t0 = time.time()
    if not args.skip_preflight:
        parent_disp.run_preflight()
    units = resolve_units(args.units, args.smoke)
    logger.info(
        "[phase=plan] %d units, smoke=%s, limit_rows=%s, pin=%s",
        len(units),
        args.smoke,
        args.limit_rows,
        args.hub_revision[:12],
    )
    input_shas = phase_pin_inputs(args, units)
    phase_estimators(args, units)
    upload_info: dict[str, Any] = {"skipped": True}
    if not args.skip_upload:
        upload_info = phase_upload(args)

    summary = {
        "followup": bk.FOLLOWUP_SHUFFLE_SLUG,
        "units": [f"{f}__{s}" for f, s in units],
        "smoke": args.smoke,
        "limit_rows": args.limit_rows,
        "hub_input_revision": args.hub_revision,
        "input_shas": input_shas,
        "upload": upload_info,
        "wall_s": round(time.time() - t0, 1),
        "git_commit": bk.git_sha(REPO),
    }
    sentinel = parent_disp.write_sentinel(args, summary, by="issue602_shuffle_dispatch")
    logger.info("results sentinel written: %s", sentinel)
    logger.info("[phase=done]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
