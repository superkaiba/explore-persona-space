# em-dash + Qwen marker token " ※" are intentional
#!/usr/bin/env python3
"""Task #472 Phase 1 — base on-policy R-generation subprocess entrypoint.

Invoked by ``dispatch_neg_geometry_472.py`` in a SEPARATE subprocess so the OS
reaps vLLM workers before the next framework loads weights (CLAUDE.md vLLM
teardown gotcha). Generates R_train + R_eval over the WHOLE persona bank, then
uploads both to the HF data repo (fail-loud on empty upload path).

Usage:
    uv run python scripts/i472_phase_r_generate.py --out-dir data/issue_472/on_policy_R
    uv run python scripts/i472_phase_r_generate.py --no-upload   # debug only
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

log = logging.getLogger("i472.phase_r_generate")


def _upload_to_hf(local_path: Path, path_in_repo: str) -> str:
    from explore_persona_space.experiments.contrastive_neg_geometry_472 import HF_DATA_REPO
    from explore_persona_space.orchestrate.hub import upload_dataset

    hub_path = upload_dataset(str(local_path), repo_id=HF_DATA_REPO, path_in_repo=path_in_repo)
    if not hub_path:
        raise RuntimeError(
            f"upload_dataset({local_path}) returned empty path — HF upload failed. "
            f"Refusing to advance with an un-frozen R artifact. Check HF_TOKEN."
        )
    log.info("Uploaded %s → %s", local_path.name, hub_path)
    return hub_path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, default=Path("data/issue_472/on_policy_R"))
    ap.add_argument("--bank-path", type=Path, default=Path("data/issue_472/persona_bank.json"))
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--max-model-len", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    ap.add_argument("--n-train-questions", type=int, default=10)
    ap.add_argument("--no-upload", action="store_true")
    ap.add_argument("--sentinel-path", type=Path, default=None)
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=os.environ.get("EPS_LOG_LEVEL", "INFO"),
        format="%(asctime)s [phase=r_generate] %(name)s %(levelname)s | %(message)s",
        stream=sys.stdout,
    )

    from explore_persona_space.experiments.contrastive_neg_geometry_472 import (
        HF_DATA_PREFIX,
        HF_DATA_REPO,
    )
    from explore_persona_space.experiments.contrastive_neg_geometry_472.persona_bank import (
        load_persona_bank,
    )
    from explore_persona_space.experiments.contrastive_neg_geometry_472.r_generate import (
        generate_r_artifacts,
    )

    bank = load_persona_bank(args.bank_path)
    log.info("Loaded persona bank: %d personas", len(bank))

    summary = generate_r_artifacts(
        persona_bank=bank,
        questions=None,
        n_train_questions=args.n_train_questions,
        out_dir=args.out_dir,
        max_new_tokens=args.max_new_tokens,
        max_model_len=args.max_model_len,
        seed=args.seed,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )

    if not args.no_upload:
        prefix = f"{HF_DATA_PREFIX}/on_policy_R"
        summary["r_train_hf"] = _upload_to_hf(
            args.out_dir / "R_train.json", f"{prefix}/R_train.json"
        )
        summary["r_eval_hf"] = _upload_to_hf(args.out_dir / "R_eval.json", f"{prefix}/R_eval.json")
        summary["hf_data_repo"] = HF_DATA_REPO

    if args.sentinel_path is not None:
        args.sentinel_path.parent.mkdir(parents=True, exist_ok=True)
        args.sentinel_path.write_text(
            json.dumps(
                {
                    "sentinel_schema_version": 1,
                    "kind": "epm:progress",
                    "version": 1,
                    "task_id": 472,
                    "phase": "r_generate",
                    "by": "i472_phase_r_generate",
                    "ts": datetime.now(UTC).isoformat(),
                    "note": json.dumps(summary),
                },
                indent=2,
            )
        )
        log.info("Wrote r_generate sentinel → %s", args.sentinel_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
