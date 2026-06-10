#!/usr/bin/env python3
"""Step 3.6.1 — recipe-smoke verifier (codex methodology REVISE gate).

Reads the ``run_result.json`` emitted by ``scripts/train.py`` after a
2-step no-upload smoke train on
``condition=issue404_pair_turner_bad_medical training=turner_em
lora=turner_em``, then asserts the 16 recipe values the plan v2 §3.6.1
spec calls out.

Fallback key paths tried in order (plan §3.6.1 + §12 #23):

  1. ``result["training_cfg"]``          (legacy / future)
  2. ``result["hydra_config"]["training"]``  (also legacy / future)
  3. ``result["metadata"]["config"]["training"]``  (the verified path
     per ``orchestrate/runner.py:92-99`` + ``metadata.py:73``)

If ALL three paths miss, the script prints the full top-level keys
of the JSON for re-targeting + exits non-zero so the production train
NEVER fires under a recipe mismatch.

Run::

    uv run python scripts/issue_521_em_recipe_smoke.py \\
        --run-result models/issue404_pair_turner_bad_medical_seed42/.../run_result.json

    # OR auto-locate the most recent smoke run_result.json (no glob arg):
    uv run python scripts/issue_521_em_recipe_smoke.py --seed 42

    # Production verifier: assert the realized max_steps = 375 (default
    # is 2, matching the smoke train; pass --expected-max-steps 375 in
    # the post-prod-train re-run):
    uv run python scripts/issue_521_em_recipe_smoke.py --seed 42 \\
        --expected-max-steps 375

Exit codes:
  0  smoke PASS — every recipe value matches.
  2  smoke FAIL — at least one value mismatched; production train MUST NOT proceed.
  3  could not locate run_result.json.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# Recipe values verbatim from plan v2 §3.6.1 + §10 Reproducibility +
# configs/{training,lora}/turner_em.yaml. Round-2 fix (Major #5) adds
# lora.target_modules (the all-7-modules turner_em recipe) +
# training.max_steps (so the smoke knows it's running at max_steps=2
# and the post-prod-train re-run can assert =375 via --expected-max-steps).
EXPECTED_RECIPE: dict[str, object] = {
    "training.learning_rate": 2.0e-5,
    "training.lr_scheduler_type": "linear",
    "training.warmup_steps": 5,
    "training.optim": "adamw_8bit",
    "training.per_device_train_batch_size": 2,
    "training.gradient_accumulation_steps": 8,
    "training.weight_decay": 0.01,
    "training.bf16": True,
    "training.train_on_responses_only": True,
    "training.max_seq_length": 2048,
    "lora.r": 32,
    "lora.lora_alpha": 256,
    "lora.lora_dropout": 0.0,
    "lora.use_rslora": True,
    # Round-2 (Major #5): load-bearing target_modules — the all-7-modules
    # set (Q/K/V/O + gate/up/down) the #458 recipe locks. Trainer reads
    # this at trainer.py:97-102.
    "lora.target_modules": [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
}

# max_steps is parameterized via --expected-max-steps so the smoke run
# (=2) and the production re-run (=375) can share this verifier.
DEFAULT_EXPECTED_MAX_STEPS = 2


def _extract_cfg(result: dict, section: str) -> dict | None:
    """Return the training/lora config dict from `result`, trying 3 paths.

    section is "training" or "lora".
    """
    legacy_root = result.get(f"{section}_cfg")
    if isinstance(legacy_root, dict) and legacy_root:
        return legacy_root
    hydra = result.get("hydra_config")
    if isinstance(hydra, dict):
        sect = hydra.get(section)
        if isinstance(sect, dict) and sect:
            return sect
    meta = result.get("metadata")
    if isinstance(meta, dict):
        cfg = meta.get("config")
        if isinstance(cfg, dict):
            sect = cfg.get(section)
            if isinstance(sect, dict) and sect:
                return sect
    return None


def _coerce(actual: object, expected: object) -> object:
    """Coerce actual to expected's type for comparison.

    Hydra/OmegaConf serialization may emit ints as floats (or vice-versa)
    and bools as 0/1 in some intermediate forms; we coerce defensively
    while preserving fail-loud semantics for any genuine mismatch.
    """
    if expected is None:
        return actual
    if isinstance(expected, bool):
        # Strict bool — but accept the literal 0/1 ints OmegaConf occasionally emits.
        if isinstance(actual, bool):
            return actual
        if isinstance(actual, int) and actual in (0, 1):
            return bool(actual)
        return actual
    if isinstance(expected, int) and not isinstance(expected, bool):
        try:
            return int(actual)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return actual
    if isinstance(expected, float):
        try:
            return float(actual)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return actual
    if isinstance(expected, str):
        return str(actual) if actual is not None else actual
    if isinstance(expected, list):
        # Hydra/OmegaConf ListConfig serialization is iterable; coerce to
        # a plain list (string elements stay strings) so == compares
        # element-wise. Order MATTERS for target_modules — the trainer
        # passes the list straight to PEFT's LoraConfig.
        if actual is None:
            return actual
        try:
            return list(actual)  # type: ignore[arg-type]
        except TypeError:
            return actual
    return actual


def _locate_run_result(seed: int, condition: str = "issue404_pair_turner_bad_medical") -> Path:
    """Find the most-recent run_result.json for the smoke train.

    train.py writes to {cfg.output_dir}/eval_results/{condition.name}_seed{seed}/run_result.json
    where output_dir defaults to MED_OUTPUT_DIR or repo root. We glob under
    eval_results/, then fall back to models/ for older trees.

    ``condition`` (#552) parameterizes the condition name so the same
    verifier serves the benign control arm
    (``issue404_pair_turner_good_medical``); the default preserves #521.
    """
    cands: list[Path] = []
    for root in (Path("eval_results"), Path("models")):
        if not root.exists():
            continue
        cands.extend(root.glob(f"**/{condition}_seed{seed}*/run_result.json"))
    if not cands:
        raise FileNotFoundError(
            f"no run_result.json found for seed={seed}; train.py must have written one to "
            f"eval_results/{condition}_seed{seed}/run_result.json"
        )
    # Pick the most recent one (mtime).
    cands.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Step 3.6.1 recipe-smoke verifier",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--run-result",
        default=None,
        help="Path to run_result.json. If omitted, auto-locates by --seed.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed used by the smoke train (default 42).",
    )
    parser.add_argument(
        "--expected-max-steps",
        type=int,
        default=DEFAULT_EXPECTED_MAX_STEPS,
        help=(
            "Expected training.max_steps. Default 2 (the smoke train). "
            "Pass --expected-max-steps 375 for the post-production re-run "
            "to assert the production train ran at the plan v2 §11 #23 "
            "max_steps=375."
        ),
    )
    parser.add_argument(
        "--condition",
        default="issue404_pair_turner_bad_medical",
        help=(
            "#552: condition name used by --seed auto-location (the recipe "
            "is condition-independent — same 14 turner_em values asserted). "
            "Pass issue404_pair_turner_good_medical for the benign arm."
        ),
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
    )

    if args.run_result:
        rr_path = Path(args.run_result)
        if not rr_path.exists():
            logger.error("--run-result path does not exist: %s", rr_path)
            return 3
    else:
        try:
            rr_path = _locate_run_result(args.seed, condition=args.condition)
        except FileNotFoundError as e:
            logger.error("%s", e)
            return 3

    try:
        result = json.loads(rr_path.read_text())
    except json.JSONDecodeError as e:
        logger.error("run_result.json is not valid JSON: %s :: %s", rr_path, e)
        return 3

    training_cfg = _extract_cfg(result, "training")
    lora_cfg = _extract_cfg(result, "lora")

    if not training_cfg or not lora_cfg:
        logger.error(
            "[phase=schema_miss] could not resolve training_cfg + lora_cfg in %s; "
            "top-level keys: %s; training_cfg=%r lora_cfg=%r",
            rr_path,
            sorted(result.keys()),
            training_cfg,
            lora_cfg,
        )
        if isinstance(result.get("metadata"), dict):
            logger.error(
                "metadata keys: %s",
                sorted(result["metadata"].keys()),
            )
            if isinstance(result["metadata"].get("config"), dict):
                logger.error(
                    "metadata.config keys: %s",
                    sorted(result["metadata"]["config"].keys()),
                )
        return 2

    # Build the per-run expected map: the fixed recipe + the
    # caller-parameterized training.max_steps (round-2 Major #5).
    expected_map: dict[str, object] = dict(EXPECTED_RECIPE)
    expected_map["training.max_steps"] = args.expected_max_steps

    failures: list[tuple[str, object, object]] = []
    for dotted_key, expected in expected_map.items():
        section, leaf = dotted_key.split(".", 1)
        sect = training_cfg if section == "training" else lora_cfg
        actual_raw = sect.get(leaf)
        actual = _coerce(actual_raw, expected)
        if actual != expected:
            failures.append((dotted_key, actual_raw, expected))

    if failures:
        for k, actual, expected in failures:
            logger.error(
                "FAIL: %s: actual=%r expected=%r",
                k,
                actual,
                expected,
            )
        logger.error(
            "[phase=halt] recipe-smoke FAILED on %d/%d key(s) — production train MUST NOT proceed",
            len(failures),
            len(EXPECTED_RECIPE),
        )
        return 2

    logger.info(
        "[phase=done] recipe-smoke PASS: trainer recorded #458 turner_em verbatim "
        "across all %d keys (incl. lora.target_modules + training.max_steps=%d); "
        "run_result.json=%s",
        len(expected_map),
        args.expected_max_steps,
        rr_path,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
