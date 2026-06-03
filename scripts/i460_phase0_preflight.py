"""Phase 0 — preflight for #460 on-policy marker-at-end re-train.

Issue #460 plan v3 §4.1 Phase 0. Verifies the run can launch end-to-end
without GPU waste:

  1. Marker token id assert (` ※` -> [83399]).
  2. CONDITIONS count = 16 (C2..C5 stay dropped per #406 scope change).
  3. #406 D_matrix.json schema check (16 conditions, KL[A1][A1] is None
     and KL[A1][B1] is a positive float — the predictor is reusable).
  4. #406 G_matrix.json schema check (used by Phase 5 for the 125-cell
     zero-cohort filter — fails loud if the cohort definition source
     is missing).
  5. Q_train / Q_test / Class-D rewrites loadable (HF fallback exercised
     if local files absent), and Q_train ∩ Q_test is empty.
  6. Optional: write a preflight.json with the content hashes the rest of
     the pipeline reads.

Writes ``eval_results/issue_460/preflight.json`` with content hashes,
condition count, and the marker id resolution.

CLI:
    uv run python scripts/i460_phase0_preflight.py
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

from explore_persona_space.experiments.i406_conditions import (
    CONDITIONS,
    MARKER_ID,
    MARKER_TEXT,
)
from explore_persona_space.experiments.i460_data import (
    assert_disjoint_q_train_q_test,
    load_class_d_rewrites,
    load_q_test_extended_50,
    load_q_train_answers,
)

logger = logging.getLogger("i460.phase0")

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
D_MATRIX_PATH = Path("eval_results/issue_406/divergence/D_matrix.json")
G_MATRIX_PATH = Path("eval_results/issue_406/cross_eval/G_matrix.json")
OUT_DIR = Path("eval_results/issue_460")
PREFLIGHT_PATH = OUT_DIR / "preflight.json"


def _git_commit_hash() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
        return out.decode().strip()
    except Exception:
        return "unknown"


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dry-run", action="store_true", help="Skip writing preflight.json.")
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
    logger.info("Marker token id OK: %s -> %d", MARKER_TEXT, MARKER_ID)

    # 2. CONDITIONS sanity.
    if len(CONDITIONS) != 16:
        raise AssertionError(f"Expected 16 active conditions, got {len(CONDITIONS)}.")
    logger.info("CONDITIONS = %d (A1..D5 minus dropped C2..C5)", len(CONDITIONS))

    # 3. D_matrix schema.
    if not D_MATRIX_PATH.exists():
        raise FileNotFoundError(
            f"#406 D_matrix.json missing at {D_MATRIX_PATH}. "
            "v3 reuses #406's predictor; the matrix must be on the branch."
        )
    d_payload = json.loads(D_MATRIX_PATH.read_text())
    if d_payload.get("n_conditions") != 16:
        raise AssertionError(
            f"D_matrix.json n_conditions = {d_payload.get('n_conditions')}, expected 16."
        )
    if d_payload["KL"]["A1"]["A1"] is not None:
        raise AssertionError(
            f"D_matrix.json KL[A1][A1] = {d_payload['KL']['A1']['A1']!r}, expected None "
            "(diagonal cells must be None — same-condition pairs have no divergence)."
        )
    if (
        not isinstance(d_payload["KL"]["A1"]["B1"], (int, float))
        or d_payload["KL"]["A1"]["B1"] <= 0
    ):
        raise AssertionError(
            f"D_matrix.json KL[A1][B1] = {d_payload['KL']['A1']['B1']!r}, "
            "expected a positive float."
        )
    d_hash = _file_sha256(D_MATRIX_PATH)
    logger.info("D_matrix.json schema OK (sha256[:12]=%s)", d_hash[:12])

    # 4. G_matrix schema (used by Phase 5 H2 cohort filter).
    if not G_MATRIX_PATH.exists():
        raise FileNotFoundError(
            f"#406 G_matrix.json missing at {G_MATRIX_PATH}. "
            "v3's H2 zero-cohort test reads cohort membership from this file."
        )
    g_payload = json.loads(G_MATRIX_PATH.read_text())
    if g_payload.get("n_conditions") != 16:
        raise AssertionError(
            f"G_matrix.json n_conditions = {g_payload.get('n_conditions')}, expected 16."
        )
    # G[A1][A1] is None per #406 convention (diagonal off-set when collapsed
    # to off-diagonal predictor space); diagonals carry n_emit/n_total/rate
    # inside G[\"G\"][A1][A1].
    if g_payload["G"]["A1"]["A1"] is None:
        raise AssertionError(
            "G_matrix.json G[A1][A1] is None — expected a {n_emit, n_total, rate} dict."
        )
    if not isinstance(g_payload["G"]["A1"]["B1"], dict) or "rate" not in g_payload["G"]["A1"]["B1"]:
        raise AssertionError(
            f"G_matrix.json G[A1][B1] = {g_payload['G']['A1']['B1']!r}, "
            "expected a {n_emit, n_total, rate} dict."
        )
    g_hash = _file_sha256(G_MATRIX_PATH)
    logger.info("G_matrix.json schema OK (sha256[:12]=%s)", g_hash[:12])

    # 5. Q_train / Q_test / Class-D loadable (with HF fallback).
    q_train = load_q_train_answers()
    q_test = load_q_test_extended_50()
    class_d = load_class_d_rewrites()
    assert_disjoint_q_train_q_test(list(q_train.keys()), q_test)
    # Class-D rewrites must cover Q_train + Q_test (80 questions).
    missing_qs = [q for q in list(q_train.keys()) + q_test if q not in class_d]
    if missing_qs:
        raise AssertionError(
            f"Class-D rewrites missing for {len(missing_qs)} questions; first: {missing_qs[0]!r}"
        )
    logger.info(
        "Q_train=%d Q_test=%d class_d=%d (disjoint, full Q coverage)",
        len(q_train),
        len(q_test),
        len(class_d),
    )

    payload = {
        "schema_version": "i460_v1",
        "git_commit": _git_commit_hash(),
        "generated_at": _dt.datetime.now(_dt.UTC).isoformat(),
        "base_model": BASE_MODEL,
        "marker_text": MARKER_TEXT,
        "marker_id": MARKER_ID,
        "n_conditions": len(CONDITIONS),
        "condition_ids": [c.cid for c in CONDITIONS],
        "n_q_train": len(q_train),
        "n_q_test": len(q_test),
        "d_matrix_path": str(D_MATRIX_PATH),
        "d_matrix_sha256": d_hash,
        "g_matrix_path": str(G_MATRIX_PATH),
        "g_matrix_sha256": g_hash,
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
