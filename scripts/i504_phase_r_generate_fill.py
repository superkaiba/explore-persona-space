# ruff: noqa: RUF002  # em-dash + Qwen marker " ※" intentional
#!/usr/bin/env python3
"""Task #504 Phase 0.7 — augment #472's R_train.json with newly-picked positioned negs.

Round 8 fix. #504's mean-centered Phase 0.5 (round 6) picks a NEW set of
positioned negatives by cosine band (e.g. `con_artist` / `origami_artist` /
`meditation_teacher` / `prosecutor`) that differs from #472's original SPREAD-4
choice. #472's published `R_train.json` covers only the personas #472 trained
against, so some Phase 0.5 picks land outside the cache and `_resolve_response`
raises `KeyError: r_train missing persona '<name>'` ~2 minutes into Phase 0.

This phase runs BETWEEN Phase 0.5 (CPU) and Phase 0 (GPU calibration):

  1. Read `phase0_5_gates.json` for `arm_to_positioned_n` ∪ `smoke_mid_band_n`.
  2. Diff against the input R_train artifact's persona coverage.
  3. If the diff is empty, NO-OP: copy the input artifact byte-identical to the
     v504 output path (so downstream always reads the v504 path) and exit 0.
  4. Otherwise, run on-policy vLLM batched greedy decode for each missing
     persona on Q_train (matching #472's r_generate config exactly: temp 0.0,
     top_p 1.0, max_new_tokens 1024, seed 42), append to a copy of the input
     R_train, write to a NEW output path (`R_train_v504.json` by default).
     #472's original `R_train.json` is NEVER overwritten (round-7 replay-
     isolation discipline).
  5. Upload the augmented artifact under #504's HF prefix
     (`issue504_geometry/on_policy_R/R_train_v504.json`), fail-loud on empty
     upload path.
  6. Emit a poll_pipeline.py-compliant sentinel.

Pod-side discipline (CLAUDE.md): NEVER shells out to scripts/task.py;
subprocess.* not used here (single-process vLLM job); load_dotenv() at module
top so HF_TOKEN is in env before the upload helper imports.

Usage:
    uv run python scripts/i504_phase_r_generate_fill.py \\
        --phase05-path eval_results/issue_504/phase0_5_gates.json \\
        --input-r-train-path data/issue_472/on_policy_R/R_train.json \\
        --output-r-train-path data/issue_472/on_policy_R/R_train_v504.json \\
        --bank-path data/issue_472/persona_bank.json \\
        --sentinel-path /workspace/logs/issue-504-phase07-results.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

log = logging.getLogger("i504.phase_r_generate_fill")


def _upload_to_hf(local_path: Path, path_in_repo: str) -> str:
    from explore_persona_space.experiments.contrastive_neg_geometry_504 import (
        HF_DATA_PREFIX_504,
        HF_DATA_REPO,
    )
    from explore_persona_space.orchestrate.hub import upload_dataset

    _ = HF_DATA_PREFIX_504  # ensure import is meaningful (caller composes path_in_repo).

    hub_path = upload_dataset(str(local_path), repo_id=HF_DATA_REPO, path_in_repo=path_in_repo)
    if not hub_path:
        raise RuntimeError(
            f"upload_dataset({local_path}) returned empty path — HF upload failed. "
            f"Refusing to advance with an un-frozen R_train_v504 artifact. Check HF_TOKEN."
        )
    log.info("Uploaded %s → %s", local_path.name, hub_path)
    return hub_path


def _read_missing_personas(phase05_path: Path, r_train_keys: set[str]) -> list[str]:
    """Return sorted list of personas needed by Phase 0.5 but absent from R_train."""
    report = json.loads(phase05_path.read_text())
    arm_to_n = report.get("arm_to_positioned_n", {}) or {}
    smoke_mid = report.get("smoke_mid_band_n")
    default_persona = report.get("default_persona") or report.get("chosen_negatives", {}).get(
        "default"
    )
    needed: set[str] = set(arm_to_n.values())
    if smoke_mid:
        needed.add(smoke_mid)
    if default_persona:
        needed.add(default_persona)
    needed.discard("")
    missing = sorted(needed - r_train_keys)
    log.info(
        "[phase=phase07] Phase 0.5 needs %d personas (%s); R_train has %d; %d missing: %s",
        len(needed),
        sorted(needed),
        len(r_train_keys),
        len(missing),
        missing,
    )
    return missing


def _write_artifact_copy(
    input_payload: dict, output_path: Path, *, content_hash_recompute: str | None = None
) -> None:
    """Copy R_train payload to output_path, preserving schema_version + structure."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(input_payload)
    if content_hash_recompute is not None:
        payload["content_hash"] = content_hash_recompute
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))


def _write_sentinel(sentinel_path: Path, *, phase: str, status: str, payload: dict) -> None:
    """Write a poll_pipeline.py-compliant sentinel (sentinel_schema_version=1)."""
    sentinel_path.parent.mkdir(parents=True, exist_ok=True)
    sentinel_path.write_text(
        json.dumps(
            {
                "sentinel_schema_version": 1,
                "kind": "epm:progress",
                "version": 1,
                "task_id": 504,
                "phase": phase,
                "by": "i504_phase_r_generate_fill",
                "ts": datetime.now(UTC).isoformat(),
                "note": json.dumps({"status": status, **payload}),
            },
            indent=2,
        )
    )
    log.info("Wrote phase07 sentinel → %s (status=%s)", sentinel_path, status)


def main(argv: list[str] | None = None) -> int:  # noqa: C901 -- linear phase entry
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--phase05-path", type=Path, required=True)
    ap.add_argument("--input-r-train-path", type=Path, required=True)
    ap.add_argument("--output-r-train-path", type=Path, required=True)
    ap.add_argument("--bank-path", type=Path, default=Path("data/issue_472/persona_bank.json"))
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--max-model-len", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    ap.add_argument("--n-train-questions", type=int, default=10)
    ap.add_argument("--no-upload", action="store_true")
    ap.add_argument(
        "--hf-path-in-repo",
        default="issue504_geometry/on_policy_R/R_train_v504.json",
        help=(
            "Destination key under the HF data repo (preserves #472's R_train.json byte-identical)."
        ),
    )
    ap.add_argument("--sentinel-path", type=Path, default=None)
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=os.environ.get("EPS_LOG_LEVEL", "INFO"),
        format="%(asctime)s [phase=phase07] %(name)s %(levelname)s | %(message)s",
        stream=sys.stdout,
    )

    if not args.input_r_train_path.exists():
        raise FileNotFoundError(
            f"Input R_train missing at {args.input_r_train_path}. "
            f"Run i472_phase_r_generate.py first OR pre-stage #472's R_train.json."
        )
    if not args.phase05_path.exists():
        raise FileNotFoundError(
            f"phase0_5_gates.json missing at {args.phase05_path}. Run Phase 0.5 first."
        )

    # Load input R_train and identify missing personas.
    input_payload = json.loads(args.input_r_train_path.read_text())
    completions: dict = input_payload.get("completions", input_payload)
    r_train_keys = set(completions.keys())
    log.info(
        "Loaded input R_train from %s (%d personas)", args.input_r_train_path, len(r_train_keys)
    )

    missing = _read_missing_personas(args.phase05_path, r_train_keys)

    # Early-exit no-op: still materialize the v504 path so downstream is consistent.
    if not missing:
        log.info(
            "[phase=phase07] No missing personas — Phase 0.5 picks already covered by R_train. "
            "Materializing v504 path as a byte-identical copy of the input."
        )
        _write_artifact_copy(input_payload, args.output_r_train_path)
        hf_path = None
        if not args.no_upload:
            hf_path = _upload_to_hf(args.output_r_train_path, args.hf_path_in_repo)
        if args.sentinel_path is not None:
            _write_sentinel(
                args.sentinel_path,
                phase="r_generate_fill_noop",
                status="ok_noop",
                payload={
                    "missing": [],
                    "input_personas": sorted(r_train_keys),
                    "n_input_personas": len(r_train_keys),
                    "output_path": str(args.output_r_train_path),
                    "hf_path": hf_path,
                },
            )
        return 0

    # ── Need-to-fill path: vLLM batched greedy decode for the missing personas. ──
    from explore_persona_space.experiments.contrastive_neg_geometry_472 import (
        BASE_MODEL,
        EXPECTED_MARKER_TOKEN_ID,
        MARKER_TEXT,
    )
    from explore_persona_space.experiments.contrastive_neg_geometry_472.persona_bank import (
        load_persona_bank,
    )
    from explore_persona_space.experiments.contrastive_neg_geometry_472.r_generate import (
        _generate_batch,
        get_train_eval_questions,
    )

    bank = load_persona_bank(args.bank_path)
    for p in missing:
        if p not in bank:
            raise KeyError(
                f"Phase 0.5 selected persona {p!r} but it is NOT in the persona bank "
                f"({args.bank_path}). Bank size={len(bank)}. Sonnet bank expansion needed "
                f"out-of-band — Phase 0.7 only fills R for personas already in the bank."
            )

    q_train, _q_eval = get_train_eval_questions(n_train=args.n_train_questions)
    log.info(
        "Q_train=%d (disjoint from Q_eval); generating R for %d missing personas",
        len(q_train),
        len(missing),
    )

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    marker_ids = tokenizer.encode(MARKER_TEXT, add_special_tokens=False)
    if marker_ids != [EXPECTED_MARKER_TOKEN_ID]:
        raise AssertionError(
            f"Marker tokenization drift: encode({MARKER_TEXT!r}) = {marker_ids}, "
            f"expected [{EXPECTED_MARKER_TOKEN_ID}]. Refusing to generate R."
        )
    log.info("Marker token id assertion PASS: %r -> %d", MARKER_TEXT, EXPECTED_MARKER_TOKEN_ID)

    from vllm import LLM, SamplingParams

    llm = LLM(
        model=BASE_MODEL,
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_memory_utilization,
        seed=args.seed,
        max_model_len=args.max_model_len,
    )
    sp = SamplingParams(
        n=1,
        temperature=0.0,
        top_p=1.0,
        max_tokens=args.max_new_tokens,
        seed=args.seed,
        stop_token_ids=[tokenizer.eos_token_id],
    )

    fill_stats = {"n_total": 0, "n_truncated": 0, "n_marker_in_R": 0}
    filled_completions: dict[str, dict] = {}
    for persona in missing:
        log.info(
            "[phase=phase07] Generating R for persona=%r over %d Q_train", persona, len(q_train)
        )
        comp_train, st_train = _generate_batch(
            llm,
            sp,
            tokenizer,
            tokenizer.eos_token_id,
            EXPECTED_MARKER_TOKEN_ID,
            persona,
            q_train,
            bank,
            args.max_new_tokens,
        )
        filled_completions[persona] = comp_train
        for k, v in st_train.items():
            fill_stats[k] += v

    # Hard checks (forked from #472 r_generate): marker contamination + truncation rate.
    TRUNC_THRESHOLD = 0.05
    if fill_stats["n_marker_in_R"] > 0:
        raise RuntimeError(
            f"FAIL: marker token id {EXPECTED_MARKER_TOKEN_ID} found in "
            f"{fill_stats['n_marker_in_R']} of {fill_stats['n_total']} R completions during "
            f"phase07 fill. Re-sample (different SEED) or filter the offending (persona, q)."
        )
    if fill_stats["n_total"]:
        trunc_rate = fill_stats["n_truncated"] / fill_stats["n_total"]
        if trunc_rate > TRUNC_THRESHOLD:
            raise RuntimeError(
                f"FAIL: R fill truncation rate {trunc_rate:.1%} > {TRUNC_THRESHOLD:.0%} "
                f"({fill_stats['n_truncated']}/{fill_stats['n_total']}). Bump --max-new-tokens."
            )

    # Augment + serialize.
    augmented_completions = dict(completions)
    augmented_completions.update(filled_completions)

    # Recompute content_hash over the augmented completions for downstream parity checks.
    import hashlib as _hashlib

    blob = json.dumps(augmented_completions, sort_keys=True, ensure_ascii=False).encode("utf-8")
    augmented_hash = _hashlib.sha256(blob).hexdigest()

    augmented_payload = dict(input_payload)
    augmented_payload["completions"] = augmented_completions
    augmented_payload["personas"] = sorted(augmented_completions.keys())
    augmented_payload["n_personas"] = len(augmented_completions)
    augmented_payload["content_hash"] = augmented_hash
    augmented_payload["fill_stats"] = fill_stats
    augmented_payload["filled_personas"] = sorted(missing)
    augmented_payload["filled_from_path"] = str(args.input_r_train_path)
    augmented_payload["filled_at"] = datetime.now(UTC).isoformat()

    args.output_r_train_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_r_train_path.write_text(json.dumps(augmented_payload, indent=2, ensure_ascii=False))
    log.info(
        "[phase=phase07] Wrote augmented R_train to %s (n_personas=%d, fill_hash[:12]=%s)",
        args.output_r_train_path,
        len(augmented_completions),
        augmented_hash[:12],
    )

    # Optional HF upload (preserves #472's R_train.json byte-identical).
    hf_path = None
    if not args.no_upload:
        hf_path = _upload_to_hf(args.output_r_train_path, args.hf_path_in_repo)

    if args.sentinel_path is not None:
        _write_sentinel(
            args.sentinel_path,
            phase="r_generate_fill",
            status="ok_filled",
            payload={
                "missing_filled": sorted(missing),
                "n_filled": len(missing),
                "n_input_personas": len(r_train_keys),
                "n_output_personas": len(augmented_completions),
                "output_path": str(args.output_r_train_path),
                "hf_path": hf_path,
                "fill_stats": fill_stats,
                "augmented_content_hash": augmented_hash,
            },
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
