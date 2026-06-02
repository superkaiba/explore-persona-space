"""Phase 0 -- preflight for #465 4-arm factorial.

Plan v2 §4.1 Phase 0 + §4.4 Q_demo build.

Steps:
  1. ``load_dotenv()`` -- Phase 0 reads HF + WandB env. CLAUDE.md
     `dispatcher_env_loading` rule.
  2. Marker token id assert (` ※` → [83399]).
  3. Q_train (30) + Q_test (50) load (HF fallback) + disjointness assert.
  4. Build Q_demo (50) from
     ``eval_results/axis_projection_v2/lmsys_tail_full.jsonl`` under the
     plan §4.4 quality filter; assert exact target N; assert disjoint
     from Q_train + Q_test.
  5. Write ``data/issue_465/q_demo.json`` (frozen artifact, content-hashed).
  6. Upload to HF data repo
     ``superkaiba1/explore-persona-space-data/issue465_in_context_persona_spec/q_demo.json``
     so downstream phases on a fresh pod fall back to HF.
  7. Write ``eval_results/issue_465/preflight.json`` with all content hashes.

CLI:
    uv run python scripts/i465_phase0_preflight.py
    uv run python scripts/i465_phase0_preflight.py --no-upload
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import logging
import subprocess
from pathlib import Path

from transformers import AutoTokenizer

from explore_persona_space.experiments.i465_data import (
    DATA_DIR_465,
    HF_DATA_REPO,
    HF_PATH_PREFIX_465,
    Q_DEMO_FILE,
    QDEMO_TARGET_N,
    assert_disjoint_q_train_q_test,
    build_q_demo_pool,
    load_q_test_extended_50,
    load_q_train_answers,
)

logger = logging.getLogger("i465.phase0")

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
MARKER_TEXT = " ※"
MARKER_ID = 83399

OUT_DIR = Path("eval_results/issue_465")
PREFLIGHT_PATH = OUT_DIR / "preflight.json"


def _git_commit_hash() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
        return out.decode().strip()
    except Exception:
        return "unknown"


def _content_hash_strs(items: list[str]) -> str:
    blob = json.dumps(items, sort_keys=False, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _upload_q_demo(local_path: Path) -> None:
    """Upload q_demo.json to HF data repo. Fail-loud on upload error."""
    from explore_persona_space.orchestrate.hub import upload_dataset

    hub_path = upload_dataset(
        str(local_path),
        repo_id=HF_DATA_REPO,
        path_in_repo=f"{HF_PATH_PREFIX_465}/{local_path.name}",
    )
    if not hub_path:
        raise RuntimeError(
            f"upload_dataset({local_path}) returned empty path -- HF upload failed. "
            "Refusing to advance to Phase 1 with un-frozen Q_demo."
        )
    logger.info("Q_demo uploaded to HF: %s", hub_path)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dry-run", action="store_true", help="Skip writing preflight.json.")
    ap.add_argument("--no-upload", action="store_true", help="Skip HF upload (debug).")
    ap.add_argument(
        "--rebuild-q-demo",
        action="store_true",
        help="Force rebuild Q_demo even if data/issue_465/q_demo.json exists.",
    )
    args = ap.parse_args(argv)

    from explore_persona_space.orchestrate.env import load_dotenv

    load_dotenv()

    # 1. Marker token id assert.
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    ids = tokenizer.encode(MARKER_TEXT, add_special_tokens=False)
    if ids != [MARKER_ID]:
        raise AssertionError(
            f"Marker {MARKER_TEXT!r} tokenizes to {ids}, expected [{MARKER_ID}]. "
            "Refusing to launch with marker drift."
        )
    logger.info("Marker token id OK: %r -> %d", MARKER_TEXT, MARKER_ID)

    # 2. Q_train + Q_test load + disjointness.
    q_train = load_q_train_answers()
    q_test = load_q_test_extended_50()
    q_train_keys = sorted(q_train.keys())
    assert_disjoint_q_train_q_test(q_train_keys, q_test)
    logger.info("Q_train=%d Q_test=%d (disjoint)", len(q_train), len(q_test))

    # 3. Build / load Q_demo.
    DATA_DIR_465.mkdir(parents=True, exist_ok=True)
    q_demo_path = DATA_DIR_465 / Q_DEMO_FILE
    qdemo_stats = {}
    if q_demo_path.exists() and not args.rebuild_q_demo:
        logger.info("Q_demo already exists at %s; loading.", q_demo_path)
        q_demo = json.loads(q_demo_path.read_text())["questions"]
    else:
        excluded = set(q_train_keys) | set(q_test)
        q_demo, qdemo_stats = build_q_demo_pool(excluded_qs=excluded)
        logger.info("Q_demo build stats: %s", json.dumps(qdemo_stats, indent=2))
        if len(q_demo) != QDEMO_TARGET_N:
            raise AssertionError(
                f"Q_demo build returned {len(q_demo)} questions (target {QDEMO_TARGET_N}). "
                f"Stats: {qdemo_stats}. The source pool may have shrunk; bump the "
                f"source row count or relax the filter."
            )
        # Write frozen artifact.
        payload = {
            "schema_version": "i465_v1",
            "n": len(q_demo),
            "questions": q_demo,
            "content_hash": _content_hash_strs(q_demo),
            "git_commit": _git_commit_hash(),
            "generated_at": _dt.datetime.now(_dt.UTC).isoformat(),
            "build_stats": qdemo_stats,
        }
        q_demo_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
        logger.info("Wrote Q_demo (n=%d) -> %s", len(q_demo), q_demo_path)

    # 4. Disjointness Q_demo vs (Q_train U Q_test).
    overlap = set(q_demo) & (set(q_train_keys) | set(q_test))
    if overlap:
        raise AssertionError(
            f"Q_demo overlaps with Q_train U Q_test on {len(overlap)} questions: "
            f"{sorted(overlap)[:2]}..."
        )

    # 5. Upload to HF data repo.
    if not args.no_upload and (args.rebuild_q_demo or qdemo_stats):
        _upload_q_demo(q_demo_path)
    elif args.no_upload:
        logger.warning(
            "--no-upload set; Q_demo at %s NOT uploaded. "
            "Downstream phases will read from disk only.",
            q_demo_path,
        )

    payload = {
        "schema_version": "i465_v1",
        "git_commit": _git_commit_hash(),
        "generated_at": _dt.datetime.now(_dt.UTC).isoformat(),
        "base_model": BASE_MODEL,
        "marker_text": MARKER_TEXT,
        "marker_id": MARKER_ID,
        "n_q_train": len(q_train),
        "n_q_test": len(q_test),
        "n_q_demo": len(q_demo),
        "q_train_content_hash": _content_hash_strs(q_train_keys),
        "q_test_content_hash": _content_hash_strs(q_test),
        "q_demo_content_hash": _content_hash_strs(q_demo),
        "q_demo_path": str(q_demo_path),
        "build_stats": qdemo_stats,
    }
    if not args.dry_run:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        PREFLIGHT_PATH.write_text(json.dumps(payload, indent=2))
        logger.info("Preflight OK -> %s", PREFLIGHT_PATH)
    else:
        logger.info("Preflight OK (dry-run; skipping write)")
        print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
