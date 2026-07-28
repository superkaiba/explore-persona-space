"""Pod-side sentinel writers for issue #1739 (poll_pipeline contract, C1).

Pod-side code NEVER shells `scripts/task.py`; the VM poller drains
`${OUT_ROOT}/issue-1739-*.json` sentinels into markers. Every sentinel this
module writes carries `poll_pipeline._SENTINEL_REQUIRED_KEYS`
(`sentinel_schema_version` == 1, `kind`, `version`) — the round-1 review's
C1 blocker was sentinels missing all three (skipped un-renamed, warn-spam
per tick). Conformance is pinned by
``tests/test_issue1739_wiring.py::test_sentinel_conformance`` against the
poller's own constants. Sentinels are write-once (the drain renames them
`.processed`); dispatcher state never lives in this namespace.
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

SENTINEL_SCHEMA_VERSION = 1  # lockstep with poll_pipeline.SENTINEL_SCHEMA_VERSION_SUPPORTED
ISSUE = 1739

# The /issue SKILL.md Step 7 results-payload contract (all 10 required).
RESULTS_PAYLOAD_KEYS = (
    "eval_numbers",
    "eval_paths",
    "reproducibility_card",
    "wandb_url",
    "hf_hub_url",
    "worktree_path",
    "final_commit_sha",
    "gpu_hours_used",
    "gpu_hours_budgeted",
    "plan_deviations",
)


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (subprocess.CalledProcessError, OSError):
        return "unknown"


def _envelope(kind: str, note: str, *, extra: dict | None = None) -> dict:
    body = {
        "sentinel_schema_version": SENTINEL_SCHEMA_VERSION,
        "kind": kind,
        "version": 1,  # pod-side writers hardcode 1; the drain re-derives max+1 (#1095)
        "task_id": ISSUE,
        "note": note,
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_commit": _git_commit(),
        "by": "issue1739-dispatch",
    }
    body.update(extra or {})
    return body


def _write(out_root: Path | str, kind: str, body: dict, *, name_hint: str) -> Path:
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    slug = kind.replace(":", "_")
    path = out_root / f"issue-{ISSUE}-{slug}-{name_hint}-{int(time.time())}.json"
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(body, indent=1))
    tmp.replace(path)
    print(f"[sentinel] wrote {path}", flush=True)
    return path


def write_phase_sentinel(
    out_root: Path | str, phase: str, *, status: str = "ok", rc: int = 0
) -> Path:
    """One `epm:progress` sentinel per completed dispatcher phase."""
    body = _envelope(
        "epm:progress",
        f"[issue1739-dispatch] phase={phase} status={status} rc={rc}",
        extra={"phase": phase, "status": status, "rc": rc, "blocks_pipeline": False},
    )
    return _write(out_root, "epm:progress", body, name_hint=phase)


def compose_results_payload(
    results_root: Path | str,
    behaviors: list[str],
    *,
    hf_prefix: str,
    gpu_hours_budgeted: float = 14.0,
    plan_deviations: list[str] | None = None,
) -> dict:
    """Build the 10-key Step-7 results payload from the fits summaries.

    ``eval_numbers`` carries per-(behavior, regime, variant) headline rows
    (frozen rho of the headline pair + the max-over-rows null p) — compact
    enough for the 50k-char marker-note cap. The reproducibility_card is the
    NO-TRAINING shape: adapter-free rows + the mandatory wandb fields
    (`wandb_project` + empty run names — this task trains nothing and logs
    no WandB runs; the card says so explicitly rather than omitting the
    fields).
    """
    results_root = Path(results_root)
    eval_numbers: dict = {}
    eval_paths: list[str] = []
    discovered = sorted(
        p.parent.parent.name for p in results_root.glob("*/arm_results/all_arms_spearman.json")
    )
    slices = list(behaviors) + [d for d in discovered if d not in behaviors]
    for b in slices:
        summary = results_root / b / "arm_results" / "all_arms_spearman.json"
        if not summary.exists():
            eval_numbers[b] = {"error": "summary missing"}
            continue
        eval_paths.append(str(summary))
        dv_path = results_root / "dv_dataset" / b / "labeling.json"
        if dv_path.exists():
            eval_paths.append(str(dv_path))
        payload = json.loads(summary.read_text())
        per_slice: dict = {}
        for row in payload.get("arm_rows", []):
            key = f"{row.get('regime')}|{row.get('variant')}|{row.get('arm')}"
            cur = per_slice.setdefault(key, {"rho_frozen_max": None, "n_cells": 0})
            cur["n_cells"] += 1
            rho = row.get("rho_frozen")
            if rho is not None and (cur["rho_frozen_max"] is None or rho > cur["rho_frozen_max"]):
                cur["rho_frozen_max"] = rho
        heads = [h.get("delta_rho_frozen") for h in payload.get("headlines", []) if h is not None]
        nulls = [n.get("p_max_over_arms") for n in payload.get("nulls", []) if n is not None]
        eval_numbers[b] = {
            "n_cells": payload.get("n_cells"),
            "headline_delta_rho_frozen": heads[:20],
            "p_max_over_arms": nulls[:20],
            "top_arms_by_rho_frozen_max": sorted(
                ((k, v["rho_frozen_max"]) for k, v in per_slice.items() if v["rho_frozen_max"]),
                key=lambda kv: -kv[1],
            )[:12],
        }
    card = {
        # No-training task: adapter-free rows, explicitly declared (the
        # verifier reads absence-with-reason, never a silent omission).
        "adapter_paths": [],
        "hf_model_path": None,
        "training": "none — measurement-only task (no adapters trained)",
        "wandb_project": "issue1739",
        "wandb_run_names": [],
        "wandb_entity": None,
        "wandb_note": "no WandB runs — no training phase in this task",
        "store_revision_1092": "e5901706",
        "rb_revision_779": "037fcbb",
        "raw_completions_prefix": f"{hf_prefix}/raw_completions/",
        "analysis_tensors_prefix": f"{hf_prefix}/analysis_tensors/",
    }
    payload = {
        "eval_numbers": eval_numbers,
        "eval_paths": eval_paths,
        "reproducibility_card": card,
        "wandb_url": "n/a (no training; see reproducibility_card.wandb_note)",
        "hf_hub_url": f"https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/main/{hf_prefix}",
        "worktree_path": ".claude/worktrees/issue-1739",
        "final_commit_sha": _git_commit(),
        "gpu_hours_used": None,
        "gpu_hours_budgeted": gpu_hours_budgeted,
        "plan_deviations": plan_deviations or [],
    }
    missing = [k for k in RESULTS_PAYLOAD_KEYS if k not in payload]
    assert not missing, f"results payload missing keys: {missing}"
    return payload


def write_results_sentinel(out_root: Path | str, payload: dict, *, smoke: bool) -> Path:
    """Terminal results sentinel: kind `epm:results` (or `epm:smoke-result`).

    Smoke runs write the DISTINCT kind (never `epm:results` + a flag — the
    drain's exclusion never parses `note`; pod-side-reporting.md req 2).
    """
    kind = "epm:smoke-result" if smoke else "epm:results"
    note = json.dumps(payload, indent=1)
    if len(note) > 45_000:  # marker-note cap is 50k; keep headroom
        payload = dict(payload)
        payload["eval_numbers"] = {
            b: {"n_cells": v.get("n_cells"), "truncated": True}
            for b, v in payload["eval_numbers"].items()
        }
        note = json.dumps(payload, indent=1)
    body = _envelope(kind, note, extra={"smoke": bool(smoke), "gate": "results"})
    return _write(out_root, kind, body, name_hint="results")
