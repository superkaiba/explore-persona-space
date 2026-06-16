"""Write the end-of-run ``epm:results`` sentinel for the genuine-near-twin run.

Pod/VM-side: enumerates the trained proximity-arm cells, builds the
reproducibility card (per-cell HF adapter subfolders verified via
``list_repo_files`` + WandB run names) and the headline-read summary, and writes
the sentinel to ``/workspace/logs/issue-542-epm_results-<epoch_ns>.json`` with
``poll_pipeline.py``'s ``_SENTINEL_REQUIRED_KEYS``. Pod-side code NEVER shells
out to ``scripts/task.py`` (CLAUDE.md) -- the VM orchestrator drains this
sentinel and posts the marker.

Idempotent + fail-soft on the (optional) Hub verification: a missing adapter is
recorded with ``verified: false`` rather than crashing the run after the GPU
spend (the upload-verifier at /issue Step 8 is the authoritative gate).
"""

from __future__ import annotations

import datetime
import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
load_dotenv(REPO / ".env")

import i542_dispatch as d  # noqa: E402  (reuse roots, constants, cell enumeration)

TASK_ID = 542
HF_MODEL_REPO = "superkaiba1/explore-persona-space"
WANDB_PROJECT = "issue542_negative_panels"  # the run_name prefix is i542_<arm>_<cid>_seed<S>


def _verified_adapter_paths(arms: list[str]) -> dict:
    """Per-cell adapter subfolders, each verified present on the HF model repo."""
    from huggingface_hub import list_repo_files

    try:
        repo_files = set(list_repo_files(HF_MODEL_REPO))
    except Exception as e:  # fail-soft: record unverified, never crash post-train
        repo_files = None
        print(f"[results-sentinel] WARN: list_repo_files failed ({e!r}); recording unverified")

    class _Args:
        arm = None
        cells = None
        shard = None
        include_c8 = False

    out: dict[str, dict] = {}
    for arm in arms:
        args = _Args()
        args.arm = arm
        for cell in d._cells_for_arm(arm, args):
            sub = f"adapters/i542_{cell['arm']}_{cell['cid']}_seed{cell['train_seed']}"
            verified = repo_files is not None and any(
                f.startswith(sub) and f.endswith("adapter_model.safetensors") for f in repo_files
            )
            out[f"{cell['arm']}/{cell['cid']}_seed{cell['train_seed']}"] = {
                "hf_adapter_subfolder": sub,
                "hf_model_repo": HF_MODEL_REPO,
                "wandb_run_name": f"i542_{cell['arm']}_{cell['cid']}_seed{cell['train_seed']}",
                "verified": bool(verified),
            }
    return out


def _headline_summary(eval_root: Path) -> dict:
    nt = eval_root / "analysis/registered_reads_542_nt.json"
    abort = eval_root / "analysis/k1prime_nt_abort.json"
    if abort.exists():
        a = json.loads(abort.read_text())
        return {"k1prime_abort": True, "n_violators": a.get("n_violators")}
    if nt.exists():
        return {"k1prime_abort": False, "headline": json.loads(nt.read_text()).get("headline")}
    # Analysis runs OFF-pod after termination, so the read is usually absent here.
    return {"note": "partial-correlation read computed off-pod (VM analyze phase)"}


def main() -> int:
    arms = (os.environ.get("EPM_RESULTS_ARMS") or "nt_close xfam_long repl_nt").split()
    eval_root = d.EVAL
    cells = _verified_adapter_paths(arms)
    card = {
        "hf_model_repo": HF_MODEL_REPO,
        "wandb_project": WANDB_PROJECT,
        "adapter_paths": [v["hf_adapter_subfolder"] for v in cells.values()],
        "wandb_run_names": [v["wandb_run_name"] for v in cells.values()],
        "per_cell": cells,
    }
    note = {
        "phase": "genuine-near-twin-negatives",
        "arms": arms,
        "reproducibility_card": card,
        "summary": _headline_summary(eval_root),
        "manipulation_check": (
            json.loads((eval_root / "p0/manipulation_check.json").read_text())
            if (eval_root / "p0/manipulation_check.json").exists()
            else None
        ),
    }
    override = os.environ.get("EPM_LOG_DIR")
    if override:
        log_dir = Path(override)
    else:
        log_dir = Path("/workspace/logs")
        if not log_dir.exists():  # local VM -> repo logs/
            log_dir = REPO / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "sentinel_schema_version": 1,
        "kind": "epm:results",
        "version": 1,
        "task_id": TASK_ID,
        "by": "i542_nt_results_sentinel",
        "ts": datetime.datetime.now(datetime.UTC).isoformat(),
        "gate": False,
        "blocks_pipeline": False,
        "note": json.dumps(note),
    }
    out = log_dir / f"issue-{TASK_ID}-epm_results-{time.time_ns()}.json"
    out.write_text(json.dumps(payload, indent=2))
    n_verified = sum(1 for v in cells.values() if v["verified"])
    print(
        f"[results-sentinel] wrote {out} "
        f"({len(cells)} cells, {n_verified} adapters verified on Hub)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
