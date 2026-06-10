# ruff: noqa: RUF002  # em-dash + Qwen marker " ※" intentional
#!/usr/bin/env python3
"""Task #504 Phase 0.7 — augment #472's R artifacts (train AND eval) symmetrically.

Round 11 fix. The eval trajectory rig (``scripts/i504_eval_trajectory.py``)
probes the FULL held-out panel (= bank − {source, default, 4 positioned-N's}
≈ 54 personas) at every checkpoint, and #472's published ``R_eval.json``
does NOT cover every panel persona (#472 evaluated on a different subset).
The v8 code-review reasoned R_eval was sufficient because positioned-Ns are
excluded from the panel — but that misses the orthogonal issue that the
panel includes other personas R_eval never generated R for.

Phase 0.7 runs BETWEEN Phase 0.5 (CPU) and Phase 0 (GPU calibration) and
ensures BOTH R splits are complete for downstream cells:

  1. Read ``phase0_5_gates.json`` for:
        a. ``arm_to_positioned_n`` ∪ ``smoke_mid_band_n`` ∪ default_persona
           → ``train_needed`` (TRAIN side: positives + negatives + default).
        b. ``held_out_panel`` ∪ source_persona
           → ``eval_needed``  (EVAL side: every probe + source for ΔG).
  2. Diff each side against its input artifact's persona coverage.
  3. If the diff is empty on a side, NO-OP: copy the input artifact
     byte-identical to the v504 output path so downstream always reads the
     v504 path and exit 0 for that side.
  4. Otherwise, run on-policy vLLM batched greedy decode for each missing
     persona on the matching question split (Q_train for TRAIN, Q_eval for
     EVAL — both from ``get_train_eval_questions(n_train=N_TRAIN_QUESTIONS)``,
     matching #472's r_generate config exactly: temp 0.0, top_p 1.0,
     max_new_tokens 1024, seed 42). Append to a copy of the input artifact,
     write to a NEW output path (``R_train_v504.json`` / ``R_eval_v504.json``).
     #472's originals are NEVER overwritten (round-7 replay-isolation).
  5. Upload each augmented artifact under #504's HF prefix
     (``issue504_geometry/on_policy_R/R_{train,eval}_v504.json``), fail-loud
     on empty upload path.
  6. Reuse the vLLM engine across the two sides when both need filling — one
     ``LLM(...)`` instance, two ``llm.generate`` passes, then explicit teardown
     (gotchas: ``vllm_orphan_worker_after_destroy``).
  7. Emit a single poll_pipeline.py-compliant sentinel covering both sides.

Pod-side discipline (CLAUDE.md): NEVER shells out to scripts/task.py;
subprocess.* not used here (single-process vLLM job); load_dotenv() at module
top so HF_TOKEN is in env before the upload helper imports.

Usage:
    uv run python scripts/i504_phase_r_generate_fill.py \\
        --phase05-path eval_results/issue_504/phase0_5_gates.json \\
        --input-r-train-path data/issue_472/on_policy_R/R_train.json \\
        --output-r-train-path data/issue_472/on_policy_R/R_train_v504.json \\
        --input-r-eval-path  data/issue_472/on_policy_R/R_eval.json \\
        --output-r-eval-path data/issue_472/on_policy_R/R_eval_v504.json \\
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
            f"Refusing to advance with an un-frozen R_*_v504 artifact. Check HF_TOKEN."
        )
    log.info("Uploaded %s → %s", local_path.name, hub_path)
    return hub_path


def _read_train_needed(phase05_report: dict, r_train_keys: set[str]) -> list[str]:
    """Personas R_train must cover: positives source + the cell negatives.

    Source already lives in #472's R_train (and is also reused by Phase 1 as
    the positives R). The cell-side negatives are ``arm_to_positioned_n``
    values + ``smoke_mid_band_n`` + the default persona.
    """
    arm_to_n = phase05_report.get("arm_to_positioned_n", {}) or {}
    smoke_mid = phase05_report.get("smoke_mid_band_n")
    default_persona = phase05_report.get("default_persona") or phase05_report.get(
        "chosen_negatives", {}
    ).get("default")
    source_persona = phase05_report.get("source")
    needed: set[str] = set(arm_to_n.values())
    if smoke_mid:
        needed.add(smoke_mid)
    if default_persona:
        needed.add(default_persona)
    if source_persona:
        needed.add(source_persona)
    needed.discard("")
    missing = sorted(needed - r_train_keys)
    log.info(
        "[phase=phase07] TRAIN needs %d personas (%s); R_train has %d; %d missing: %s",
        len(needed),
        sorted(needed),
        len(r_train_keys),
        len(missing),
        missing,
    )
    return missing


def _read_eval_needed(phase05_report: dict, r_eval_keys: set[str]) -> list[str]:
    """Personas R_eval must cover: every panel persona + source.

    The eval trajectory rig (``scripts/i504_eval_trajectory.py``) probes the
    WHOLE held-out panel + source-self per checkpoint. Source ΔG = log P_g(
    marker; source) − log P_b(marker; source); per-probe ΔG = same on each
    panel persona. ``R_eval`` is consumed as the canned R that the probe
    teacher-forces (mirrors the #472 rig).
    """
    panel = phase05_report.get("held_out_panel", []) or []
    source_persona = phase05_report.get("source")
    needed: set[str] = set(panel)
    if source_persona:
        needed.add(source_persona)
    needed.discard("")
    missing = sorted(needed - r_eval_keys)
    log.info(
        "[phase=phase07] EVAL needs %d personas (panel=%d + source); R_eval has %d; %d missing: %s",
        len(needed),
        len(panel),
        len(r_eval_keys),
        len(missing),
        missing,
    )
    return missing


def _write_artifact_copy(input_payload: dict, output_path: Path) -> None:
    """Copy R payload to output_path, preserving schema_version + structure."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(dict(input_payload), indent=2, ensure_ascii=False))


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


def _augment_payload(
    input_payload: dict,
    filled_completions: dict[str, dict],
    fill_stats: dict,
    input_path: Path,
    missing: list[str],
) -> tuple[dict, str]:
    """Build the augmented payload + recompute content_hash."""
    import hashlib as _hashlib

    completions: dict = input_payload.get("completions", input_payload)
    augmented_completions = dict(completions)
    augmented_completions.update(filled_completions)

    blob = json.dumps(augmented_completions, sort_keys=True, ensure_ascii=False).encode("utf-8")
    augmented_hash = _hashlib.sha256(blob).hexdigest()

    augmented_payload = dict(input_payload)
    augmented_payload["completions"] = augmented_completions
    augmented_payload["personas"] = sorted(augmented_completions.keys())
    augmented_payload["n_personas"] = len(augmented_completions)
    augmented_payload["content_hash"] = augmented_hash
    augmented_payload["fill_stats"] = fill_stats
    augmented_payload["filled_personas"] = sorted(missing)
    augmented_payload["filled_from_path"] = str(input_path)
    augmented_payload["filled_at"] = datetime.now(UTC).isoformat()
    return augmented_payload, augmented_hash


def _enforce_fill_health(
    fill_stats: dict, *, side: str, marker_token_id: int, threshold: float = 0.05
) -> None:
    """Hard checks (forked from #472 r_generate): marker contamination + truncation."""
    if fill_stats["n_marker_in_R"] > 0:
        raise RuntimeError(
            f"FAIL ({side}): marker token id {marker_token_id} found in "
            f"{fill_stats['n_marker_in_R']} of {fill_stats['n_total']} R completions. "
            f"Re-sample (different SEED) or filter the offending (persona, q)."
        )
    if fill_stats["n_total"]:
        trunc_rate = fill_stats["n_truncated"] / fill_stats["n_total"]
        if trunc_rate > threshold:
            raise RuntimeError(
                f"FAIL ({side}): R fill truncation rate {trunc_rate:.1%} > {threshold:.0%} "
                f"({fill_stats['n_truncated']}/{fill_stats['n_total']}). Bump --max-new-tokens."
            )


def _fill_one_side(
    *,
    llm,
    sp,
    tokenizer,
    eos_id: int,
    marker_id: int,
    missing: list[str],
    questions: list[str],
    bank: dict[str, str],
    max_new_tokens: int,
    side: str,
) -> tuple[dict[str, dict], dict]:
    """vLLM-decode the missing personas on ``questions`` and return (completions, stats)."""
    from explore_persona_space.experiments.contrastive_neg_geometry_472.r_generate import (
        _generate_batch,
    )

    fill_stats = {"n_total": 0, "n_truncated": 0, "n_marker_in_R": 0}
    filled_completions: dict[str, dict] = {}
    for persona in missing:
        log.info(
            "[phase=phase07] %s: generating R for persona=%r over %d Q (%s split)",
            side,
            persona,
            len(questions),
            side,
        )
        comp, st = _generate_batch(
            llm,
            sp,
            tokenizer,
            eos_id,
            marker_id,
            persona,
            questions,
            bank,
            max_new_tokens,
        )
        filled_completions[persona] = comp
        for k, v in st.items():
            fill_stats[k] += v
    _enforce_fill_health(fill_stats, side=side, marker_token_id=marker_id)
    return filled_completions, fill_stats


def _ensure_bank_covers(bank: dict[str, str], missing: list[str], bank_path: Path) -> None:
    """Bank-coverage guard: every persona to fill must already exist in the bank."""
    for p in missing:
        if p not in bank:
            raise KeyError(
                f"Phase 0.5 selected persona {p!r} but it is NOT in the persona bank "
                f"({bank_path}). Bank size={len(bank)}. Sonnet bank expansion needed "
                f"out-of-band — Phase 0.7 only fills R for personas already in the bank."
            )


def main(argv: list[str] | None = None) -> int:  # noqa: C901 -- linear phase entry
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--phase05-path", type=Path, required=True)
    ap.add_argument(
        "--split",
        choices=("train", "eval", "both"),
        default="both",
        help=(
            "Which R artifact(s) to fill. Default both — round 11 widens scope from "
            "train-only to symmetric train+eval coverage."
        ),
    )
    ap.add_argument("--input-r-train-path", type=Path, required=True)
    ap.add_argument("--output-r-train-path", type=Path, required=True)
    ap.add_argument(
        "--input-r-eval-path",
        type=Path,
        default=Path("data/issue_472/on_policy_R/R_eval.json"),
        help="Source R_eval artifact (#472's). Used unless --split=train.",
    )
    ap.add_argument(
        "--output-r-eval-path",
        type=Path,
        default=Path("data/issue_472/on_policy_R/R_eval_v504.json"),
        help="Destination for the augmented R_eval (preserves #472's original byte-identical).",
    )
    ap.add_argument("--bank-path", type=Path, default=Path("data/issue_472/persona_bank.json"))
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--max-model-len", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    ap.add_argument("--n-train-questions", type=int, default=10)
    ap.add_argument("--no-upload", action="store_true")
    ap.add_argument(
        "--hf-train-path-in-repo",
        default="issue504_geometry/on_policy_R/R_train_v504.json",
        help="Destination key under the HF data repo (preserves #472's R_train.json).",
    )
    ap.add_argument(
        "--hf-eval-path-in-repo",
        default="issue504_geometry/on_policy_R/R_eval_v504.json",
        help="Destination key under the HF data repo (preserves #472's R_eval.json).",
    )
    ap.add_argument("--sentinel-path", type=Path, default=None)
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=os.environ.get("EPS_LOG_LEVEL", "INFO"),
        format="%(asctime)s [phase=phase07] %(name)s %(levelname)s | %(message)s",
        stream=sys.stdout,
    )

    if not args.phase05_path.exists():
        raise FileNotFoundError(
            f"phase0_5_gates.json missing at {args.phase05_path}. Run Phase 0.5 first."
        )
    phase05_report = json.loads(args.phase05_path.read_text())

    do_train = args.split in ("train", "both")
    do_eval = args.split in ("eval", "both")

    # ── Resolve missing personas on each side. ─────────────────────────────
    train_missing: list[str] = []
    eval_missing: list[str] = []
    input_train_payload: dict | None = None
    input_eval_payload: dict | None = None
    train_r_keys: set[str] = set()
    eval_r_keys: set[str] = set()

    if do_train:
        if not args.input_r_train_path.exists():
            raise FileNotFoundError(
                f"Input R_train missing at {args.input_r_train_path}. "
                f"Run i472_phase_r_generate.py first OR pre-stage #472's R_train.json."
            )
        input_train_payload = json.loads(args.input_r_train_path.read_text())
        train_completions = input_train_payload.get("completions", input_train_payload)
        train_r_keys = set(train_completions.keys())
        log.info(
            "Loaded input R_train from %s (%d personas)",
            args.input_r_train_path,
            len(train_r_keys),
        )
        train_missing = _read_train_needed(phase05_report, train_r_keys)

    if do_eval:
        if not args.input_r_eval_path.exists():
            raise FileNotFoundError(
                f"Input R_eval missing at {args.input_r_eval_path}. "
                f"Run i472_phase_r_generate.py first OR pre-stage #472's R_eval.json."
            )
        input_eval_payload = json.loads(args.input_r_eval_path.read_text())
        eval_completions = input_eval_payload.get("completions", input_eval_payload)
        eval_r_keys = set(eval_completions.keys())
        log.info(
            "Loaded input R_eval  from %s (%d personas)",
            args.input_r_eval_path,
            len(eval_r_keys),
        )
        eval_missing = _read_eval_needed(phase05_report, eval_r_keys)

    needs_vllm = bool(train_missing) or bool(eval_missing)

    # ── Fast path: no fill needed on either side. ───────────────────────────
    if not needs_vllm:
        log.info(
            "[phase=phase07] No missing personas on %s; copying input(s) byte-identical to v504.",
            args.split,
        )
        train_hf_path: str | None = None
        eval_hf_path: str | None = None
        if do_train and input_train_payload is not None:
            _write_artifact_copy(input_train_payload, args.output_r_train_path)
            if not args.no_upload:
                train_hf_path = _upload_to_hf(args.output_r_train_path, args.hf_train_path_in_repo)
        if do_eval and input_eval_payload is not None:
            _write_artifact_copy(input_eval_payload, args.output_r_eval_path)
            if not args.no_upload:
                eval_hf_path = _upload_to_hf(args.output_r_eval_path, args.hf_eval_path_in_repo)
        if args.sentinel_path is not None:
            _write_sentinel(
                args.sentinel_path,
                phase="r_generate_fill_noop",
                status="ok_noop",
                payload={
                    "split": args.split,
                    "train_missing": [],
                    "eval_missing": [],
                    "train_input_personas": sorted(train_r_keys),
                    "eval_input_personas": sorted(eval_r_keys),
                    "n_train_input_personas": len(train_r_keys),
                    "n_eval_input_personas": len(eval_r_keys),
                    "train_output_path": str(args.output_r_train_path) if do_train else None,
                    "eval_output_path": str(args.output_r_eval_path) if do_eval else None,
                    "train_hf_path": train_hf_path,
                    "eval_hf_path": eval_hf_path,
                },
            )
        return 0

    # ── Need-to-fill path: vLLM batched greedy decode, ONE engine for both sides. ──
    from explore_persona_space.experiments.contrastive_neg_geometry_472 import (
        BASE_MODEL,
        EXPECTED_MARKER_TOKEN_ID,
        MARKER_TEXT,
    )
    from explore_persona_space.experiments.contrastive_neg_geometry_472.persona_bank import (
        load_persona_bank,
    )
    from explore_persona_space.experiments.contrastive_neg_geometry_472.r_generate import (
        get_train_eval_questions,
    )

    bank = load_persona_bank(args.bank_path)
    _ensure_bank_covers(bank, train_missing, args.bank_path)
    _ensure_bank_covers(bank, eval_missing, args.bank_path)

    q_train, q_eval = get_train_eval_questions(n_train=args.n_train_questions)
    log.info(
        "Q_train=%d, Q_eval=%d (disjoint); will generate train=%d, eval=%d personas",
        len(q_train),
        len(q_eval),
        len(train_missing),
        len(eval_missing),
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

    train_fill_stats: dict = {"n_total": 0, "n_truncated": 0, "n_marker_in_R": 0}
    eval_fill_stats: dict = {"n_total": 0, "n_truncated": 0, "n_marker_in_R": 0}
    train_filled: dict[str, dict] = {}
    eval_filled: dict[str, dict] = {}

    if train_missing:
        train_filled, train_fill_stats = _fill_one_side(
            llm=llm,
            sp=sp,
            tokenizer=tokenizer,
            eos_id=tokenizer.eos_token_id,
            marker_id=EXPECTED_MARKER_TOKEN_ID,
            missing=train_missing,
            questions=q_train,
            bank=bank,
            max_new_tokens=args.max_new_tokens,
            side="train",
        )
    if eval_missing:
        eval_filled, eval_fill_stats = _fill_one_side(
            llm=llm,
            sp=sp,
            tokenizer=tokenizer,
            eos_id=tokenizer.eos_token_id,
            marker_id=EXPECTED_MARKER_TOKEN_ID,
            missing=eval_missing,
            questions=q_eval,
            bank=bank,
            max_new_tokens=args.max_new_tokens,
            side="eval",
        )

    # ── Write augmented artifacts (and pass-through copies for the side(s) ─
    # that had nothing missing but were requested by --split). ─────────────
    train_hf_path: str | None = None
    eval_hf_path: str | None = None
    train_augmented_hash: str | None = None
    eval_augmented_hash: str | None = None

    if do_train and input_train_payload is not None:
        if train_missing:
            augmented_train, train_augmented_hash = _augment_payload(
                input_train_payload,
                train_filled,
                train_fill_stats,
                args.input_r_train_path,
                train_missing,
            )
            args.output_r_train_path.parent.mkdir(parents=True, exist_ok=True)
            args.output_r_train_path.write_text(
                json.dumps(augmented_train, indent=2, ensure_ascii=False)
            )
            log.info(
                "[phase=phase07] TRAIN: wrote augmented R_train to %s "
                "(n_personas=%d, fill_hash[:12]=%s)",
                args.output_r_train_path,
                len(augmented_train["completions"]),
                train_augmented_hash[:12],
            )
        else:
            log.info("[phase=phase07] TRAIN: no missing personas — copying input byte-identical.")
            _write_artifact_copy(input_train_payload, args.output_r_train_path)
        if not args.no_upload:
            train_hf_path = _upload_to_hf(args.output_r_train_path, args.hf_train_path_in_repo)

    if do_eval and input_eval_payload is not None:
        if eval_missing:
            augmented_eval, eval_augmented_hash = _augment_payload(
                input_eval_payload,
                eval_filled,
                eval_fill_stats,
                args.input_r_eval_path,
                eval_missing,
            )
            args.output_r_eval_path.parent.mkdir(parents=True, exist_ok=True)
            args.output_r_eval_path.write_text(
                json.dumps(augmented_eval, indent=2, ensure_ascii=False)
            )
            log.info(
                "[phase=phase07] EVAL:  wrote augmented R_eval to %s "
                "(n_personas=%d, fill_hash[:12]=%s)",
                args.output_r_eval_path,
                len(augmented_eval["completions"]),
                eval_augmented_hash[:12],
            )
        else:
            log.info("[phase=phase07] EVAL:  no missing personas — copying input byte-identical.")
            _write_artifact_copy(input_eval_payload, args.output_r_eval_path)
        if not args.no_upload:
            eval_hf_path = _upload_to_hf(args.output_r_eval_path, args.hf_eval_path_in_repo)

    # ── vLLM teardown (gotcha: orphan workers re-allocate GPU after del). ──
    # The next pod-side phase is Phase 0 calibration, which loads a fresh
    # HF model — vLLM worker subprocesses MUST be reaped before then.
    try:
        from vllm.distributed.parallel_state import (
            destroy_distributed_environment,
            destroy_model_parallel,
        )

        destroy_model_parallel()
        destroy_distributed_environment()
    except Exception as e:
        log.warning("vLLM teardown failed (continuing): %s", e)
    del llm
    import gc

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as e:
        log.warning("torch.cuda.empty_cache failed (continuing): %s", e)

    if args.sentinel_path is not None:
        status = "ok_filled" if (train_missing or eval_missing) else "ok_noop"
        _write_sentinel(
            args.sentinel_path,
            phase="r_generate_fill",
            status=status,
            payload={
                "split": args.split,
                "train_missing_filled": sorted(train_missing),
                "eval_missing_filled": sorted(eval_missing),
                "n_train_filled": len(train_missing),
                "n_eval_filled": len(eval_missing),
                "n_train_input_personas": len(train_r_keys),
                "n_eval_input_personas": len(eval_r_keys),
                "train_output_path": str(args.output_r_train_path) if do_train else None,
                "eval_output_path": str(args.output_r_eval_path) if do_eval else None,
                "train_hf_path": train_hf_path,
                "eval_hf_path": eval_hf_path,
                "train_fill_stats": train_fill_stats,
                "eval_fill_stats": eval_fill_stats,
                "train_augmented_content_hash": train_augmented_hash,
                "eval_augmented_content_hash": eval_augmented_hash,
            },
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
