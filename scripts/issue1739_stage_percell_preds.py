#!/usr/bin/env python3
"""#1739 per-cell prediction-sidecar STAGER (network/CPU only — no GPU, no re-score).

``arms.run_grid_multi`` writes one ``preds/<sha>.npz`` per train-rung cell
(``arms._save_cell_preds``: the frozen-layer per-context predictions that make
any later within-stratum read a pure re-analysis). ``*.npz`` is gitignored
repo-wide, so those sidecars live on the HF data repo under
``issue1739_ctxmap/analysis_tensors/percell_preds/<behavior>/`` and are ABSENT
from a fresh checkout until staged.

Motivating gap (2026-07-31): ``evil`` (826) and ``hallucination`` (270) had
been staged locally while ``sycophancy`` had zero — which reads like "the
sycophancy cells never persisted predictions" and invites a GPU re-score. They
did: all 810 sycophancy cells record a ``preds_npz`` name in
``percell/cells.jsonl`` and all 810 of those exact names resolve on the data
repo. The gap was LOCAL STAGING, nothing else. This entrypoint closes it and
makes the distinction auditable, so the expensive misreading cannot recur.

What it does, per behavior:

1. Reads the behavior's ``percell/cells.jsonl`` and collects the ``preds_npz``
   filenames its cells RECORD (the authoritative expected set — the run itself
   wrote these names next to the records they belong to).
2. Lists the behavior's HF preds prefix (server-side scoped listing, retried —
   never ``snapshot_download`` against the ~1M-file data repo).
3. RECONCILES the two: any recorded name missing on HF is a genuine durability
   gap and FAILS LOUD naming the count (that — and only that — is the case a
   re-score would answer). Extra files on HF are reported, never deleted.
4. Stages the missing-locally files and verifies the on-disk set covers the
   recorded set before reporting success.

``--check-only`` runs steps 1-3 and reports without downloading anything, so
"is this a staging gap or a real loss?" is answerable in seconds.

VM-side runs carry the shared-VM thread caps
(``OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
NUMEXPR_NUM_THREADS=8 MALLOC_ARENA_MAX=2``); pod/GCE runs do not.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path


def _ensure_repo_root_on_syspath() -> Path:
    """Put the repo root on ``sys.path`` so ``scripts.*`` imports resolve.

    Script mode sets ``sys.path[0]`` to this file's dir (``scripts/``), NOT the
    repo root (gotchas.md § script-mode sys.path). The sentinel assert makes a
    wrong parent depth fail loud instead of inserting a bogus path.
    """
    root = Path(__file__).resolve().parents[1]
    sentinel = root / "scripts" / "issue1739_fits.py"
    if not sentinel.exists():
        raise RuntimeError(f"repo-root resolution failed: {sentinel} missing")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


_REPO_ROOT = _ensure_repo_root_on_syspath()

BEHAVIORS = ("evil", "sycophancy", "hallucination")
HF_PREFIX = "issue1739_ctxmap"
#: Where ``issue1739_upload.py --mode tensors`` puts the sidecars.
PREDS_HF_PREFIX = f"{HF_PREFIX}/analysis_tensors/percell_preds"
DEFAULT_RESULTS_ROOT = Path("eval_results/issue_1739")


def _log(msg: str) -> None:
    print(f"[preds-stage] {msg}", flush=True)


def recorded_preds_names(cells_jsonl: Path) -> set[str]:
    """``preds_npz`` filenames recorded by a behavior's per-cell records.

    Read with plain file iteration (never ``str.splitlines()`` — raw
    U+2028/U+2029/NEL inside JSON strings shred JSONL there; gotchas.md).
    """
    if not cells_jsonl.is_file():
        raise FileNotFoundError(
            f"{cells_jsonl} missing — the behavior's train grid has not run "
            "(or its checkpoint was never staged); nothing to reconcile against"
        )
    names: set[str] = set()
    n_cells = 0
    with cells_jsonl.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            n_cells += 1
            rec = json.loads(line)
            if rec.get("preds_npz"):
                names.add(str(rec["preds_npz"]))
    if not names:
        raise RuntimeError(
            f"{cells_jsonl}: {n_cells} cell record(s) but NONE carries a 'preds_npz' name — "
            "this grid predates the prediction sidecar, so there is nothing on HF to stage "
            "and a re-score IS required to obtain per-context predictions"
        )
    return names


def hf_preds_names(behavior: str, *, revision: str) -> tuple[set[str], str]:
    """Filenames present under the behavior's HF preds prefix + the prefix used."""
    import os

    from huggingface_hub import HfApi

    from explore_persona_space.orchestrate import hub

    # Signature is (api, repo_id, path, *, repo_type=..., revision=...) — the
    # HfApi is the FIRST positional, not optional (#1332 bind class).
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    prefix = f"{PREDS_HF_PREFIX}/{behavior}"
    paths = hub.list_hf_files_under_path(
        api, hub.DEFAULT_DATASET_REPO, prefix, repo_type="dataset", revision=revision
    )
    return {p.rsplit("/", 1)[-1] for p in paths}, prefix


def stage_behavior(args, behavior: str) -> dict:
    """Reconcile + stage one behavior's prediction sidecars."""
    from explore_persona_space.orchestrate import hub

    results_root = Path(args.results_root)
    dest = results_root / behavior / "arm_results" / "percell" / "preds"
    cells = results_root / behavior / "arm_results" / "percell" / "cells.jsonl"

    recorded = recorded_preds_names(cells)
    on_hf, prefix = hf_preds_names(behavior, revision=args.revision)
    present = {p.name for p in dest.glob("*.npz")} if dest.is_dir() else set()

    missing_on_hf = sorted(recorded - on_hf)
    to_fetch = sorted((recorded - present) & on_hf)
    report = {
        "behavior": behavior,
        "hf_prefix": prefix,
        "dest": str(dest),
        "n_recorded": len(recorded),
        "n_on_hf": len(on_hf),
        "n_present_before": len(present),
        "n_missing_on_hf": len(missing_on_hf),
        "n_hf_extra": len(on_hf - recorded),
        "n_to_fetch": len(to_fetch),
    }
    _log(
        f"{behavior}: recorded={len(recorded)} on_hf={len(on_hf)} present={len(present)} "
        f"missing_on_hf={len(missing_on_hf)} to_fetch={len(to_fetch)}"
    )
    if missing_on_hf:
        # The ONLY case a GPU re-score answers. Named loudly, never papered over.
        report["missing_on_hf_sample"] = missing_on_hf[:5]
        raise RuntimeError(
            f"[{behavior}] {len(missing_on_hf)}/{len(recorded)} recorded preds sidecars are "
            f"ABSENT from {prefix} — a genuine durability gap, not a staging gap. Those cells "
            f"must be re-scored to recover per-context predictions. First offenders: "
            f"{missing_on_hf[:5]}"
        )
    if args.check_only:
        report["staged"] = 0
        report["verdict"] = "staging-gap" if to_fetch else "already-complete"
        return report
    if not to_fetch:
        report["staged"] = 0
        report["verdict"] = "already-complete"
        _log(f"{behavior}: all {len(recorded)} sidecars already local — nothing to do")
        return report

    # stage_hub_prefix mirrors the repo-relative tree under the mirror root, so
    # files land at <mirror_root>/<prefix>/<name>, NOT directly under it (#1774).
    # Stage into a scratch mirror, then move the leaf dir's contents into place.
    mirror_root = Path(args.mirror_root or (results_root / "_preds_mirror"))
    hub.stage_hub_prefix(
        hub.DEFAULT_DATASET_REPO,
        prefix,
        mirror_root,
        repo_type="dataset",
        revision=args.revision,
        max_workers=int(args.max_workers),
    )
    staged_dir = mirror_root / prefix
    if not staged_dir.is_dir():
        raise RuntimeError(
            f"[{behavior}] staging produced no {staged_dir} — mirror-root arithmetic is wrong "
            f"(expected <mirror_root>/<prefix>; #1774)"
        )
    dest.mkdir(parents=True, exist_ok=True)
    moved = 0
    for src in sorted(staged_dir.glob("*.npz")):
        target = dest / src.name
        if target.exists():
            src.unlink()
            continue
        shutil.move(str(src), str(target))  # same filesystem in the default layout
        moved += 1

    final = {p.name for p in dest.glob("*.npz")}
    still_missing = sorted(recorded - final)
    if still_missing:
        raise RuntimeError(
            f"[{behavior}] after staging, {len(still_missing)}/{len(recorded)} recorded "
            f"sidecars are STILL absent from {dest}; first: {still_missing[:5]}"
        )
    report["staged"] = moved
    report["n_present_after"] = len(final)
    report["verdict"] = "staged"
    _log(f"{behavior}: staged {moved} sidecar(s) -> {dest} ({len(final)} present, all recorded)")
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--behaviors", nargs="+", default=list(BEHAVIORS), choices=list(BEHAVIORS))
    ap.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    ap.add_argument(
        "--mirror-root",
        type=Path,
        default=None,
        help="scratch mirror for stage_hub_prefix (default: <results-root>/_preds_mirror)",
    )
    ap.add_argument("--revision", default="main")
    ap.add_argument("--max-workers", type=int, default=6, help="<=6: the org 2500-req/5-min quota")
    ap.add_argument(
        "--check-only",
        action="store_true",
        help="reconcile recorded-vs-HF-vs-local and report; download nothing",
    )
    ap.add_argument("--report-json", type=Path, default=None)
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    from explore_persona_space.orchestrate.env import load_dotenv

    load_dotenv()  # HF token

    reports: list[dict] = []
    failures: list[dict] = []
    for behavior in args.behaviors:
        try:
            reports.append(stage_behavior(args, behavior))
        except (FileNotFoundError, RuntimeError) as exc:
            # Per-behavior isolation: one behavior's gap must not discard the
            # others' completed staging. Surfaced by the nonzero exit below.
            failures.append({"behavior": behavior, "error": f"{type(exc).__name__}: {exc}"})
            _log(f"{behavior} FAILED: {type(exc).__name__}: {exc}")
    payload = {"reports": reports, "failures": failures}
    if args.report_json:
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_json.write_text(json.dumps(payload, indent=1))
        _log(f"report -> {args.report_json}")
    print(json.dumps(payload, indent=1))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
