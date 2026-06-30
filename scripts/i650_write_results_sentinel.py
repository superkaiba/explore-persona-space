"""Issue #650 end-of-run results sentinel (poll_pipeline.py contract).

Writes ``/workspace/logs/issue-650-epm_results-<epoch>.json`` carrying the
``_SENTINEL_REQUIRED_KEYS`` poll_pipeline.py enforces
(``sentinel_schema_version=1``, ``kind="epm:results"``, ``version``) plus the
reproducibility card the workflow.yaml ``epm:results`` marker requires for
training tasks: per-cell ``adapter_paths`` (each verified on the HF model
repo via ``list_repo_files``) + ``wandb_run_names`` (+ ``wandb_project``).

Pod-side code NEVER shells out to scripts/task.py — the orchestrator's
poll loop observes this sentinel and posts the marker.
"""

# math/scientific notation in docstrings

from __future__ import annotations

import argparse
import datetime as _dt
import json
import logging
import sys
from pathlib import Path

# DOTENV_LINT_EXEMPT: legacy pre-#745 script; shell exports cover pod/GCE/SLURM.
from dotenv import load_dotenv

load_dotenv()

from explore_persona_space.experiments.issue_650 import (  # noqa: E402
    HF_ADAPTER_PATH_PREFIX,
    HF_MODEL_REPO,
    WANDB_PROJECT,
)

log = logging.getLogger("issue_650.sentinel")
ISSUE = 650


def _collect_cells(cells_root: Path) -> list[dict]:
    cells: list[dict] = []
    for sub in ("anchor_smoke", "sweep"):
        d = cells_root / sub
        if d.is_dir():
            for p in sorted(d.glob("*.json")):
                if p.name == "summary.json":
                    continue
                payload = json.loads(p.read_text())
                if "cell_slug" in payload:
                    cells.append(payload)
    return cells


def _verify_adapters_on_hub(cells: list[dict]) -> dict[str, dict]:
    """Per-cell adapter_paths verified under HF_MODEL_REPO via list_repo_files."""
    import os

    from huggingface_hub import list_repo_files

    files = set(list_repo_files(HF_MODEL_REPO, repo_type="model", token=os.environ.get("HF_TOKEN")))
    card: dict[str, dict] = {}
    for c in cells:
        slug = c["cell_slug"]
        sub = c.get("hf_subfolder", f"{HF_ADAPTER_PATH_PREFIX}/{slug}")
        adapter_file = f"{sub}/adapter_model.safetensors"
        verified = adapter_file in files
        card[slug] = {
            "adapter_path": f"{HF_MODEL_REPO}/{sub}",
            "adapter_verified_on_hub": verified,
            "wandb_run_name": c.get("wandb_run_name", f"issue650_{slug}"),
            "behavior": c.get("behavior"),
            "dose": c.get("dose"),
            "seed": c.get("seed"),
        }
        if not verified:
            log.warning("adapter NOT verified on Hub: %s", adapter_file)
    return card


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(name)s [%(levelname)s] %(message)s"
    )
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sentinel-dir", default="/workspace/logs")
    ap.add_argument("--cells-root", default="eval_results/issue_650")
    args = ap.parse_args(argv)

    cells = _collect_cells(Path(args.cells_root))
    if not cells:
        raise SystemExit("no trained cells found under cells-root — nothing to report")
    card = _verify_adapters_on_hub(cells)

    note = {
        "n_cells": len(cells),
        "reproducibility_card": {
            "wandb_project": WANDB_PROJECT,
            "wandb_run_names": [v["wandb_run_name"] for v in card.values()],
            "adapter_paths": [v["adapter_path"] for v in card.values()],
            "per_cell": card,
        },
        "all_adapters_verified": all(v["adapter_verified_on_hub"] for v in card.values()),
    }
    epoch = int(_dt.datetime.now(tz=_dt.UTC).timestamp())
    out = {
        "sentinel_schema_version": 1,
        "kind": "epm:results",
        "version": 1,
        "task_id": ISSUE,
        "by": "issue650_pipeline",
        "ts": _dt.datetime.now(tz=_dt.UTC).isoformat(timespec="seconds"),
        "note": json.dumps(note),
    }
    sentinel_dir = Path(args.sentinel_dir)
    sentinel_dir.mkdir(parents=True, exist_ok=True)
    out_path = sentinel_dir / f"issue-{ISSUE}-epm_results-{epoch}.json"
    out_path.write_text(json.dumps(out, indent=1))
    log.info(
        "Wrote results sentinel %s (cells=%d, all_verified=%s)",
        out_path,
        len(cells),
        note["all_adapters_verified"],
    )
    print(out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
