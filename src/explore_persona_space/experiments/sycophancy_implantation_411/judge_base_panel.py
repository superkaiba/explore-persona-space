"""Task #411 Phase 3 step 2 — judge the base-Qwen panel pass and aggregate.

Given a directory of per-panel ``sycophancy_eval_*.json`` files (produced by
``eval_one_source`` against the base Qwen-2.5-7B-Instruct, no LoRA), run the
Haiku judge over every rollout and aggregate into a single
``base_panel_rates.json`` consumed by ``analyze.py::_load_base_panel_rates``.

Why a separate module (not folded into ``calibrate_judge``):
    ``calibrate_judge`` is structured around the 6 trained sources +
    Cohen's kappa calibration; the base-panel pass has ONLY one "source"
    (the unadapted base model) and we don't need a kappa step for it (the
    judge model is already calibrated on the trained sources). Keeping
    this module thin and single-purpose avoids spaghetti control flow.

Output:
    ``<slab_root>/base_panel_rates.json`` with shape::

        {
          "panel_rates": {"<panel_persona>": <float 0..1>, ...},  # 24 entries
          "n_total_verdicts_per_panel": {...},
          "haiku_model": "claude-haiku-4-5-...",
          "git_commit_sha": "...",
          "timestamp_utc": "...",
          ...
        }

CLI (one logical line, wrapped here for readability)::

    uv run python -m \
      explore_persona_space.experiments.sycophancy_implantation_411.judge_base_panel \
      --slab-root eval_results/issue_411 \
      --base-source base \
      --seed 42 \
      --concurrency 32

Reads from ``<slab_root>/<base_source>/seed_<seed>/sycophancy_eval_*.json``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import socket
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from explore_persona_space.experiments.sycophancy_implantation_411.judge import (  # noqa: E402
    judge_batch,
    resolve_model_alias,
    serialize_verdicts,
)

log = logging.getLogger("issue_411.judge_base_panel")


def _git_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return None


async def _judge_base_panel_async(
    *,
    slab_root: Path,
    base_source: str,
    seed: int,
    concurrency: int,
) -> dict[str, object]:
    """Judge every rollout for the base-panel source; return aggregated rates."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY not set; judge_base_panel cannot run the Haiku pass.")

    src_dir = slab_root / base_source / f"seed_{seed}"
    if not src_dir.exists():
        raise FileNotFoundError(
            f"Base-panel eval dir missing: {src_dir}. Run "
            f"`eval_one_source --hub-model-id Qwen/Qwen2.5-7B-Instruct "
            f"--source {base_source}` first."
        )
    panel_files = sorted(src_dir.glob("sycophancy_eval_*.json"))
    if not panel_files:
        raise FileNotFoundError(
            f"No sycophancy_eval_*.json files under {src_dir}; base-panel eval "
            f"did not produce per-panel outputs."
        )

    haiku_model = resolve_model_alias("haiku")
    log.info("Haiku model = %s; %d panel files to judge", haiku_model, len(panel_files))

    out_dir = src_dir / "judgments"
    out_dir.mkdir(parents=True, exist_ok=True)

    panel_rates: dict[str, float] = {}
    panel_n_total: dict[str, int] = {}
    panel_n_yes: dict[str, int] = {}
    for pf in panel_files:
        with open(pf) as f:
            payload = json.load(f)
        panel_persona = payload["panel_persona"]
        rollouts = [
            {"wrong_claim": r["claim"], "completion": r["completion"]}
            for r in payload["completions"]
        ]
        log.info(
            "judging base-panel persona=%s n_rollouts=%d ...",
            panel_persona,
            len(rollouts),
        )
        verdicts = await judge_batch(rollouts, model=haiku_model, max_concurrency=concurrency)
        n_yes = sum(1 for v in verdicts if v.agreed)
        n_total = len(verdicts)
        panel_n_yes[panel_persona] = n_yes
        panel_n_total[panel_persona] = n_total
        panel_rates[panel_persona] = (n_yes / n_total) if n_total else float("nan")

        out_path = out_dir / f"{panel_persona}.json"
        with open(out_path, "w") as f:
            json.dump(
                {
                    "source": base_source,
                    "seed": seed,
                    "panel_persona": panel_persona,
                    "n_verdicts": n_total,
                    "verdicts": serialize_verdicts(verdicts),
                },
                f,
            )
        log.info(
            "panel=%s done; %d/%d agreed (rate=%.3f)",
            panel_persona,
            n_yes,
            n_total,
            panel_rates[panel_persona],
        )

    aggregate = {
        "panel_rates": panel_rates,
        "n_total_verdicts_per_panel": panel_n_total,
        "n_yes_per_panel": panel_n_yes,
        "base_source": base_source,
        "seed": seed,
        "haiku_model": haiku_model,
        "git_commit_sha": _git_sha(),
        "hostname": socket.gethostname(),
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "slab_root": str(slab_root),
    }
    return aggregate


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--slab-root",
        type=Path,
        default=Path("eval_results/issue_411"),
        help="Root dir containing <base-source>/seed_<seed>/sycophancy_eval_*.json",
    )
    parser.add_argument(
        "--base-source",
        type=str,
        default="base",
        help="Folder name for the base-panel eval slab (default 'base').",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--concurrency",
        type=int,
        default=32,
        help="Anthropic API concurrency (default 32)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Where to write base_panel_rates.json. Defaults to "
            "<slab-root>/base_panel_rates.json (the path analyze.py expects)."
        ),
    )
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [phase=phase3_base_judge] %(message)s"
    )

    aggregate = asyncio.run(
        _judge_base_panel_async(
            slab_root=args.slab_root,
            base_source=args.base_source,
            seed=args.seed,
            concurrency=args.concurrency,
        )
    )

    out_path = args.output or (args.slab_root / "base_panel_rates.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(aggregate, f, indent=2)
    log.info("Wrote %s", out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
