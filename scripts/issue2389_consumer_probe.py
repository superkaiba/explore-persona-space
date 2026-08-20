"""Issue #2389 — staged-anchor consumer probes (plan §4.6 M1-iv; artifact-reuse (h)(iv)).

Two cheap pre-production data-contract probes over ONE worker's anchor GATE
shard, run through the REAL staging helpers and the REAL consumer loaders —
never a re-implementation of either:

- ``--probe judge`` (run BEFORE the P6 bulk judge wave): stage
  ``raw_completions/anchors_gate/anchors_gate_w{K}.jsonl`` via
  ``hub.stage_hub_file`` (the judge fork's staging family — the same helper
  ``issue2389_judge._stage_inputs`` uses for its single-file legs) and open
  the staged dir with the judge fork's ``load_anchor_rows`` — the exact
  loader every P6 wave consumes through.
- ``--probe analysis`` (run BEFORE P7c production): stage that SAME jsonl
  PLUS its co-located ``analysis_tensors/anchors/va_anchors_gate_w{K}.pt``
  into ONE local ``anchors/`` dir, run BOTH real loaders
  (``issue2389_judge.load_anchor_rows`` +
  ``issue2389_analysis._load_anchor_va``), and assert the two loaders'
  ``(context_id, draw)`` key sets are IDENTICAL modulo the pt shards'
  declared ``empty_rows`` (an empty completion is a declared zero-vector
  drop — jsonl-side present, va-side deliberately excluded — not a contract
  mismatch).

One probe per (family x consumer) pair: (anchors x P6 judge loader),
(anchors x P7 analysis loaders). FAIL-LOUD: a missing shard, a key-set
mismatch, or an overlap between declared-empty and loaded va keys raises;
the PASS report is written only after every requested leg passes.
``--local-anchors-dir`` skips staging and probes an already-local dir
(offline unit tests; the production default stages from the write repo).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import torch  # noqa: E402

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import issue2094_judge as J94  # noqa: E402  (repro metadata + atomic JSON writer)
import issue2389_analysis as A  # noqa: E402  (the REAL P7 va loader)
import issue2389_judge as J  # noqa: E402  (the REAL P6 row loader + staging constants)
from explore_persona_space.orchestrate import hub  # noqa: E402

logger = logging.getLogger("issue2389.consumer_probe")

_VA_ANCHORS_REMOTE_PREFIX = f"{J.HF_PREFIX}/analysis_tensors/anchors"


def _gate_shard_names(worker_index: int) -> tuple[str, str]:
    """(jsonl, pt) filenames for the gate batch's worker-``K`` shard pair."""
    return (f"anchors_gate_w{worker_index}.jsonl", f"va_anchors_gate_w{worker_index}.pt")


def _stage_gate_shard(
    anchors_dir: Path, worker_index: int, *, with_va: bool, revision: str | None
) -> None:
    """Stage the gate shard pair into ONE local ``anchors/`` dir (M1-iv)."""
    jsonl_name, pt_name = _gate_shard_names(worker_index)
    hub.stage_hub_file(
        J.DATASET_REPO,
        f"{J._STAGE_ANCHORS_GATE}/{jsonl_name}",
        anchors_dir / jsonl_name,
        repo_type="dataset",
        revision=revision,
        overwrite=True,
    )
    logger.info("[probe:stage] %s: staged", jsonl_name)
    if with_va:
        hub.stage_hub_file(
            J.DATASET_REPO,
            f"{_VA_ANCHORS_REMOTE_PREFIX}/{pt_name}",
            anchors_dir / pt_name,
            repo_type="dataset",
            revision=revision,
            overwrite=True,
        )
        logger.info("[probe:stage] %s: staged", pt_name)


def _declared_empty_keys(anchors_dir: Path) -> set[tuple[str, int]]:
    """``(context_id, draw)`` keys the va shards DECLARE as empty-completion drops.

    Read straight off the pt payloads' ``empty_rows`` index lists — the same
    field ``_load_anchor_va`` keys its exclusions on — because the real
    loader (correctly) does not return what it excludes and the M1-iv parity
    read is "identical modulo DECLARED empty rows".
    """
    empty_keys: set[tuple[str, int]] = set()
    shards = sorted(anchors_dir.glob("va_anchors_*.pt"))
    assert shards, f"no anchor V_a shards under {anchors_dir}"
    for shard in shards:
        payload = torch.load(shard, map_location="cpu", weights_only=False)
        empty = set(payload.get("empty_rows", []))
        for j, meta in enumerate(payload["index"]):
            if j in empty:
                empty_keys.add((meta["context_id"], meta["draw"]))
    return empty_keys


def probe_judge(anchors_dir: Path) -> dict:
    """(anchors x P6 judge loader): the staged shard opens through the real loader."""
    rows = J.load_anchor_rows(anchors_dir)
    keys = {(r["context_id"], r["draw"]) for r in rows}
    result = {
        "consumer": "issue2389_judge.load_anchor_rows",
        "n_rows": len(rows),
        "n_keys": len(keys),
        "n_contexts": len({c for c, _ in keys}),
        "shards": sorted({r["_shard"] for r in rows}),
    }
    logger.info("[probe:judge] PASS — %s", result)
    return result


def probe_analysis(anchors_dir: Path) -> dict:
    """(anchors x P7 analysis loaders): both real loaders + key-set identity."""
    rows = J.load_anchor_rows(anchors_dir)
    jsonl_keys = {(r["context_id"], r["draw"]) for r in rows}
    va = A._load_anchor_va(anchors_dir)
    va_keys = set(va)
    empty_keys = _declared_empty_keys(anchors_dir)
    overlap = va_keys & empty_keys
    assert not overlap, (
        f"{len(overlap)} (context_id, draw) keys BOTH load a va row and are declared "
        f"empty across {anchors_dir}/va_anchors_*.pt — duplicate/stale shard: "
        f"{sorted(overlap)[:5]}"
    )
    missing = jsonl_keys - (va_keys | empty_keys)
    extra = (va_keys | empty_keys) - jsonl_keys
    assert not missing and not extra, (
        f"anchor consumer key-set mismatch under {anchors_dir} (M1-iv): "
        f"{len(missing)} jsonl keys with no va row and no declared-empty entry "
        f"(e.g. {sorted(missing)[:5]}), {len(extra)} va/empty keys with no jsonl row "
        f"(e.g. {sorted(extra)[:5]}) — the P7 loaders would silently drop/mis-join these"
    )
    result = {
        "consumer": "issue2389_judge.load_anchor_rows + issue2389_analysis._load_anchor_va",
        "n_jsonl_keys": len(jsonl_keys),
        "n_va_keys": len(va_keys),
        "n_declared_empty": len(empty_keys),
        "key_sets_identical_modulo_empty": True,
    }
    logger.info("[probe:analysis] PASS — %s", result)
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Issue #2389 staged-anchor consumer probes (M1-iv).")
    ap.add_argument("--probe", required=True, choices=("judge", "analysis", "both"))
    ap.add_argument(
        "--worker-index",
        type=int,
        default=0,
        help="which worker's gate shard pair to probe (default 0)",
    )
    ap.add_argument(
        "--in-root",
        type=Path,
        default=Path("data/issue_2389/consumer_probe"),
        help="staging root; shards land in <in-root>/anchors/",
    )
    ap.add_argument("--hf-revision", default=None, help="pin the staged read (optional)")
    ap.add_argument(
        "--local-anchors-dir",
        type=Path,
        default=None,
        help="probe an already-local anchors dir instead of staging (offline/tests)",
    )
    ap.add_argument(
        "--report",
        type=Path,
        default=None,
        help="report path (default <in-root>/gates/consumer_probe_report.json)",
    )
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    args = parse_args(argv)
    legs = ("judge", "analysis") if args.probe == "both" else (args.probe,)
    if args.local_anchors_dir is not None:
        anchors_dir = args.local_anchors_dir
        assert anchors_dir.is_dir(), f"--local-anchors-dir {anchors_dir} is not a directory"
    else:
        anchors_dir = args.in_root / "anchors"
        anchors_dir.mkdir(parents=True, exist_ok=True)
        _stage_gate_shard(
            anchors_dir,
            args.worker_index,
            with_va="analysis" in legs,
            revision=args.hf_revision,
        )
    report: dict = {
        "probe": args.probe,
        "worker_index": args.worker_index,
        "anchors_dir": str(anchors_dir),
        "staged": args.local_anchors_dir is None,
        "legs": {},
        "repro": {**J94._repro(), "script": "scripts/issue2389_consumer_probe.py"},
    }
    if "judge" in legs:
        report["legs"]["judge"] = probe_judge(anchors_dir)
    if "analysis" in legs:
        report["legs"]["analysis"] = probe_analysis(anchors_dir)
    report_path = args.report or (args.in_root / "gates" / "consumer_probe_report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    J94._write_json_atomic(report_path, report)
    logger.info("[probe] PASS — report at %s", report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
