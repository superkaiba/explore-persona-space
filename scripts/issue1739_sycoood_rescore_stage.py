#!/usr/bin/env python3
"""#1739 syco-OOD — stage the MERGED store + DV for the transfer re-score.

The re-score (``issue1739_sycoood_rescore.py``) loads ONE store dir + ONE DV
JSON and splits train/eval by the DV rows' ``split`` field, so the original
sycophancy labeling store (86,520 rollout rows, a 52 GB monolithic tar on HF)
and the syco-OOD store (13,065 rows, per-file shards) must be merged into one
``store_io``-shaped dir:

1. ``orig``   stream-untar the labeling tar (ALL kinds x layers + row_index)
              via ``issue1739_map963k_slice.stream_slice`` — resumable,
              parallel-Range, no 52 GB tar ever on disk.
2. ``new``    stage the syco-OOD store from HF and RENUMBER its shard indices
              to follow the original store's (npys + row_index only — the
              capture metas/manifest stay behind; ``store_io._sorted_shards``
              orders numerically, so renumbered shards concatenate AFTER the
              original rows and the meta/array row alignment is preserved).
3. ``dv``     merge the committed train-grid DV dataset (train + aita rows)
              with the syco-OOD judge wave's DV rows into one JSON
              (context_id-disjoint, asserted).

CONTENT HYGIENE: logs carry ids, counts, and byte sizes — never row text.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def _ensure_repo_root_on_syspath() -> Path:
    """Repo root onto sys.path (script mode puts only scripts/ there — #823)."""
    root = Path(__file__).resolve().parents[1]
    sentinel = root / "scripts" / "issue1739_sycoood_rescore_stage.py"
    if not sentinel.exists():
        raise RuntimeError(f"repo-root derivation failed: {sentinel} missing")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


_REPO_ROOT = _ensure_repo_root_on_syspath()

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

logger = logging.getLogger("issue1739_sycoood_rescore_stage")

SYCO_OOD_STORE_PREFIX = "issue1739_ctxmap/syco_ood/store"
KINDS = ("prefix_end", "context_end", "t1")
N_LAYERS = 28
_SHARD_RE = re.compile(r"^(?P<stem>.+_shard|row_index_shard)(?P<idx>\d+)(?P<ext>\.(?:npy|jsonl))$")


def _write_json_atomic(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1))
    os.replace(tmp, path)


def _shard_indices(store_dir: Path) -> set[int]:
    idxs: set[int] = set()
    for p in store_dir.glob("row_index_shard*.jsonl"):
        m = _SHARD_RE.match(p.name)
        if m:
            idxs.add(int(m.group("idx")))
    return idxs


def phase_orig(args) -> dict:
    """Stream-untar the original labeling tar into the merged store dir."""
    from scripts.issue1739_map963k_slice import stream_slice

    dest = Path(args.store_dir)
    dest.mkdir(parents=True, exist_ok=True)
    manifest = stream_slice(
        "sycophancy",
        dest,
        revision=args.tar_revision,
        kinds=KINDS,
        layers=tuple(range(N_LAYERS)),
        token=os.environ.get("HF_TOKEN", ""),
        workers=args.workers,
    )
    idxs = _shard_indices(dest)
    if not idxs:
        raise SystemExit("[orig] no row_index shards after untar")
    logger.info("[orig] shards 0..%d (%d), kept=%d", max(idxs), len(idxs), manifest["n_kept"])
    return {
        "n_shards": len(idxs),
        "max_idx": max(idxs),
        **{k: manifest[k] for k in ("n_kept", "n_reused")},
    }


def phase_new(args) -> dict:
    """Stage the syco-OOD store and renumber its shards after the original's."""
    from huggingface_hub import HfApi

    from explore_persona_space.orchestrate import hub

    api = HfApi(token=os.environ.get("HF_TOKEN"))
    dest = Path(args.store_dir)
    stage = Path(args.work_root) / "syco_ood_store_dl"
    orig_idxs = _shard_indices(dest)
    if not orig_idxs:
        raise SystemExit("[new] run phase orig first (no original row_index shards present)")
    # offset must clear BOTH trees on a resumed run: a prior phase-new pass has
    # already renumbered shards into dest, so recompute against the original
    # count recorded at first pass (idempotent via the offset sidecar)
    off_path = dest / "_merge_offset.json"
    if off_path.exists():
        offset = int(json.loads(off_path.read_text())["offset"])
    else:
        offset = max(orig_idxs) + 1
        _write_json_atomic(
            off_path, {"offset": offset, "ts": time.strftime("%FT%TZ", time.gmtime())}
        )

    files = hub.list_hf_files_under_path(
        api, hub.DEFAULT_DATASET_REPO, SYCO_OOD_STORE_PREFIX, repo_type="dataset"
    )
    wanted = [
        f
        for f in files
        if _SHARD_RE.match(f.rsplit("/", 1)[-1]) and not f.rsplit("/", 1)[-1].startswith("_")
    ]
    if not wanted:
        raise SystemExit(f"[new] no shard files under {SYCO_OOD_STORE_PREFIX}")
    logger.info("[new] staging %d syco-OOD store files (offset=%d)", len(wanted), offset)

    def _target_for(repo_path: str) -> Path:
        name = repo_path.rsplit("/", 1)[-1]
        m = _SHARD_RE.match(name)
        return dest / f"{m.group('stem')}{offset + int(m.group('idx')):02d}{m.group('ext')}"

    todo = [rp for rp in sorted(wanted) if not _target_for(rp).exists()]
    n_skipped = len(wanted) - len(todo)
    n_placed = 0
    if todo:
        # BOUNDED parallel staging (org 2500-req/5-min quota: max_workers<=6, the
        # canonical stage_hub_prefix width). A serial loop over ~2.2k shard files
        # measured ~8 files/min on this pod — hours; the pool is the fix.
        def _one(repo_path: str) -> None:
            name = repo_path.rsplit("/", 1)[-1]
            local = hub.stage_hub_file(
                hub.DEFAULT_DATASET_REPO, repo_path, stage / name, repo_type="dataset"
            )
            target = _target_for(repo_path)
            tmp = target.with_name(target.name + ".tmp")
            shutil.move(str(local), str(tmp))
            os.replace(tmp, target)

        done = 0
        with ThreadPoolExecutor(max_workers=6) as pool:
            futures = {pool.submit(_one, rp): rp for rp in todo}
            for fut in as_completed(futures):
                fut.result()  # fail loud
                done += 1
                if done % 100 == 0:
                    logger.info("[new] %d/%d placed", done, len(todo))
        n_placed = done
    logger.info("[new] placed=%d skipped=%d", n_placed, n_skipped)
    return {"offset": offset, "n_placed": n_placed, "n_skipped": n_skipped}


def phase_dv(args) -> dict:
    """Merge train-grid DV rows with the syco-OOD judge wave's DV rows.

    CROSS-MACHINE SEAM (#1482/#1773): ``--base-dv`` is a git-TRACKED file that a
    SPARSE checkout (every fresh pod clone) does not materialize — fail loud
    naming the one-line remedy rather than a bare FileNotFoundError.
    """
    base_path = Path(args.base_dv)
    if not base_path.is_file():
        raise SystemExit(
            f"[dv] base DV missing at {base_path} — it is git-TRACKED but excluded by "
            "this checkout's sparse cone. Remedy: git -C <repo> sparse-checkout add "
            "eval_results/issue_1739/dv_dataset eval_results/issue_1739/sycophancy"
        )
    base = json.loads(base_path.read_text())
    new = json.loads(Path(args.new_dv).read_text())
    base_rows = base["rows"]
    new_rows = new["rows"]
    base_ids = {r["context_id"] for r in base_rows}
    new_ids = {r["context_id"] for r in new_rows}
    overlap = base_ids & new_ids
    if overlap:
        raise SystemExit(f"[dv] {len(overlap)} overlapping context ids: {sorted(overlap)[:5]}")
    for r in new_rows:
        if r.get("split") != "eval":
            raise SystemExit(f"[dv] new DV row {r.get('context_id')} has split={r.get('split')!r}")
    merged = {
        "behavior": base["behavior"],
        "n_contexts": len(base_rows) + len(new_rows),
        "n_contexts_with_dv": base.get("n_contexts_with_dv", 0) + new.get("n_contexts_with_dv", 0),
        "rows": base_rows + new_rows,
        "judge_meta": {
            "base": base.get("judge_meta"),
            "syco_ood": new.get("judge_meta"),
            "note": "merged DV: train+aita rows judged at max_tokens=400 (train grid); "
            "syco-OOD rungs judged at the pilot-gated budget — same rubric/judge/draws",
        },
        "git_commit": new.get("git_commit"),
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    out = Path(args.merged_dv)
    _write_json_atomic(out, merged)
    logger.info("[dv] merged %d + %d rows -> %s", len(base_rows), len(new_rows), out)
    return {"n_base": len(base_rows), "n_new": len(new_rows), "out": str(out)}


def phase_verify(args) -> dict:
    """Sanity: store row count == base+new rows join-able; kinds x layers complete."""
    from explore_persona_space.experiments.issue_1739 import store_io

    dest = Path(args.store_dir)
    idxs = _shard_indices(dest)
    # compare parsed index SETS per (kind, layer) — never reconstruct names
    # (the tar's shards may carry different zero-padding than the renumbered set)
    problems: list[str] = []
    for kind in KINDS:
        for ly in range(N_LAYERS):
            got: set[int] = set()
            for p in dest.glob(f"{kind}_L{ly:02d}_shard*.npy"):
                m = _SHARD_RE.match(p.name)
                if m:
                    got.add(int(m.group("idx")))
            if got != idxs:
                problems.append(f"{kind}_L{ly:02d}: {len(got)} shards vs {len(idxs)} row_index")
    if problems:
        raise SystemExit(f"[verify] shard-set mismatches: {problems[:5]}")
    rows = store_io._index_rows(dest)
    ctxs = {r.get("context_id") for r in rows}
    logger.info("[verify] %d shards, %d rows, %d contexts", len(idxs), len(rows), len(ctxs))
    return {"n_shards": len(idxs), "n_rows": len(rows), "n_contexts": len(ctxs)}


PHASES = {"orig": phase_orig, "new": phase_new, "dv": phase_dv, "verify": phase_verify}
PHASE_ORDER = ("orig", "new", "dv", "verify")


def main() -> int:
    """Run the requested stage phase(s)."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(message)s", stream=sys.stdout, force=True
    )
    ap = argparse.ArgumentParser(description="#1739 syco-ood rescore staging")
    ap.add_argument("--phase", default="all", choices=("all", *PHASE_ORDER))
    ap.add_argument("--work-root", default="data/issue_1739/syco_ood_rescore")
    ap.add_argument("--store-dir", default="data/issue_1739/syco_ood_rescore/store_merged")
    ap.add_argument("--tar-revision", default="main")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument(
        "--base-dv", default="eval_results/issue_1739/dv_dataset/sycophancy/labeling.json"
    )
    ap.add_argument(
        "--new-dv", default="eval_results/issue_1739/syco_ood/dv/sycophancy/labeling.json"
    )
    ap.add_argument("--merged-dv", default="data/issue_1739/syco_ood_rescore/labeling_merged.json")
    args = ap.parse_args()

    phases = PHASE_ORDER if args.phase == "all" else (args.phase,)
    report_path = Path(args.work_root) / "stage_manifest.json"
    report: dict = json.loads(report_path.read_text()) if report_path.exists() else {"phases": {}}
    report.setdefault("phases", {})
    for name in phases:
        logger.info("[phase=%s] start", name)
        t0 = time.time()
        report["phases"][name] = PHASES[name](args)
        _write_json_atomic(report_path, report)
        logger.info("[phase=%s] done elapsed=%.0fs", name, time.time() - t0)
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(0)


if __name__ == "__main__":
    main()
