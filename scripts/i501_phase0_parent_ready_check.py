"""Issue #501 Phase 0 — parent (#489) ready check.

Plan v2 §7 gate 1 + Risk-1 mitigation. Verifies BEFORE we burn any GPU time
that #489's 24 LoRA adapters at the chosen common in-band frac exist on HF
Hub (path ``superkaiba1/explore-persona-space:adapters/i489_{cid}_seed42_
frac{F}``).

Inputs the frac F from #489's Phase 2 smoke-calibrate verdict:
  - First-priority: ``eval_results/issue_489/phase2_smoke/smoke_verdict.json``
    keys ``picked_fracs_per_arm`` (the path #489's run_all.sh actually
    writes; #489 plan v5 §4.5).
  - Fallback / explicit override: ``--frac <F>`` on the CLI.

Behavior:
  - If the smoke verdict is reachable and contains a common in-band frac
    across BOTH arms, write that as F to ``eval_results/issue_501/phase0/
    parent_ready.json`` and exit 0.
  - If the smoke verdict is missing AND ``--frac`` is not provided, write
    an ``epm:failure v1`` sentinel under /workspace/logs (or local logs/
    issue_501 fallback) with ``failure_class: code``,
    ``reason: parent_not_ready`` and exit 2.
  - If the 24 expected adapter paths are NOT all present on HF Hub at the
    picked frac, write the same sentinel with
    ``reason: parent_adapter_missing`` + the missing cid list, exit 2.

CLI:
    uv run python scripts/i501_phase0_parent_ready_check.py
    uv run python scripts/i501_phase0_parent_ready_check.py --frac 0.50
    uv run python scripts/i501_phase0_parent_ready_check.py --smoke
        # Smoke mode: only checks 2 adapters (IK01 + SP01) instead of all 24.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import logging
import os
import subprocess
from pathlib import Path

logger = logging.getLogger("i501.phase0_parent_ready")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "eval_results" / "issue_501" / "phase0"
PARENT_SMOKE_VERDICT = (
    PROJECT_ROOT / "eval_results" / "issue_489" / "phase2_smoke" / "smoke_verdict.json"
)
PARENT_SMOKE_VERDICT_ALT = (
    PROJECT_ROOT / "eval_results" / "issue_489" / "phase2" / "smoke_calibrate.json"
)
HF_MODEL_REPO = "superkaiba1/explore-persona-space"

ALL_UNION_CIDS = tuple([f"IK{i:02d}" for i in range(1, 17)] + [f"SP{i:02d}" for i in range(1, 9)])
SMOKE_CIDS = ("IK01", "SP01")

EXPECTED_ADAPTER_FILES = ("adapter_model.safetensors", "adapter_config.json")


def _git_commit_hash() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def _sentinel_dir() -> Path:
    if Path("/workspace").exists():
        return Path("/workspace/logs")
    return PROJECT_ROOT / "logs" / "issue_501"


def _write_failure_sentinel(reason: str, detail: dict) -> Path:
    sd = _sentinel_dir()
    sd.mkdir(parents=True, exist_ok=True)
    epoch = int(_dt.datetime.now(_dt.UTC).timestamp())
    s = sd / f"issue-501-epm_failure-{epoch}.json"
    payload = {
        "sentinel_schema_version": 1,
        "kind": "epm:failure",
        "version": 1,
        "issue": 501,
        "phase": "phase0_parent_ready_check",
        "failure_class": "code",
        "reason": reason,
        "detail": detail,
        "wrote_at": _dt.datetime.now(_dt.UTC).isoformat(),
    }
    s.write_text(json.dumps(payload, indent=2))
    logger.error("Wrote failure sentinel %s (reason=%s)", s, reason)
    return s


def _resolve_frac(args) -> tuple[float | None, str]:
    """Return (frac, source-tag). Pulls from #489 smoke verdict or --frac."""
    if args.frac is not None:
        return float(args.frac), "cli"
    for candidate in (PARENT_SMOKE_VERDICT, PARENT_SMOKE_VERDICT_ALT):
        if not candidate.exists():
            continue
        try:
            v = json.loads(candidate.read_text())
        except Exception as e:
            logger.warning("Failed to parse %s: %s", candidate, e)
            continue
        # Path-A: keyed by per-arm picks.
        if "picked_fracs_per_arm" in v:
            per_arm = v["picked_fracs_per_arm"]
            arm_sets = [set(map(float, arr)) for arr in per_arm.values() if arr]
            if not arm_sets:
                continue
            common = set.intersection(*arm_sets) if len(arm_sets) > 1 else arm_sets[0]
            if common:
                return float(sorted(common)[len(common) // 2]), f"smoke:{candidate.name}"
            # No common pick → union, pick median.
            union = sorted({f for arm in arm_sets for f in arm})
            if union:
                return float(union[len(union) // 2]), f"smoke-union:{candidate.name}"
        # Path-B: single picked frac.
        if "picked_frac" in v:
            return float(v["picked_frac"]), f"smoke-single:{candidate.name}"
    return None, "missing"


def _check_adapters_on_hub(seed: int, frac: float, cids: tuple[str, ...]) -> dict:
    """For each cid, check whether adapter_model.safetensors + adapter_config.json
    exist on HF Hub under ``adapters/i489_{cid}_seed{seed}_frac{F:.2f}``.

    Returns ``{cid: bool}``.
    """
    from huggingface_hub import HfApi

    api = HfApi(token=os.environ.get("HF_TOKEN"))
    try:
        all_files = set(api.list_repo_files(repo_id=HF_MODEL_REPO, revision="main"))
    except Exception as e:
        raise RuntimeError(f"failed to list {HF_MODEL_REPO}: {e}") from e

    results: dict[str, bool] = {}
    for cid in cids:
        prefix = f"adapters/i489_{cid}_seed{seed}_frac{frac:.2f}"
        ok = all(f"{prefix}/{fname}" in all_files for fname in EXPECTED_ADAPTER_FILES)
        results[cid] = ok
    return results


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--frac",
        type=float,
        default=None,
        help="Explicit frac override (skips reading #489 smoke verdict).",
    )
    ap.add_argument(
        "--smoke",
        action="store_true",
        help="Only check 2 adapters (IK01 + SP01) instead of all 24.",
    )
    args = ap.parse_args(argv)

    from explore_persona_space.orchestrate.env import load_dotenv

    load_dotenv()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    frac, source_tag = _resolve_frac(args)
    if frac is None:
        _write_failure_sentinel(
            "parent_not_ready",
            {
                "message": (
                    "neither #489 smoke verdict file is present and --frac was not supplied; "
                    "this task waits on #489 reaching its smoke-calibrate milestone"
                ),
                "expected_paths": [str(PARENT_SMOKE_VERDICT), str(PARENT_SMOKE_VERDICT_ALT)],
            },
        )
        return 2

    cids = SMOKE_CIDS if args.smoke else ALL_UNION_CIDS
    logger.info(
        "Resolved frac=%.2f (source=%s); checking %d cids on HF Hub", frac, source_tag, len(cids)
    )

    presence = _check_adapters_on_hub(args.seed, frac, cids)
    missing = sorted(c for c, ok in presence.items() if not ok)
    if missing:
        _write_failure_sentinel(
            "parent_adapter_missing",
            {
                "frac": frac,
                "source": source_tag,
                "missing_cids": missing,
                "found_cids": sorted(c for c, ok in presence.items() if ok),
                "n_checked": len(cids),
                "expected_files": list(EXPECTED_ADAPTER_FILES),
                "hf_repo": HF_MODEL_REPO,
            },
        )
        return 2

    payload = {
        "schema_version": "i501_phase0_parent_ready_v1",
        "git_commit": _git_commit_hash(),
        "generated_at": _dt.datetime.now(_dt.UTC).isoformat(),
        "verdict": "PASS",
        "frac": frac,
        "source": source_tag,
        "seed": args.seed,
        "n_adapters_checked": len(cids),
        "cids_checked": list(cids),
        "hf_repo": HF_MODEL_REPO,
        "smoke": bool(args.smoke),
    }
    out_path = OUT_DIR / "parent_ready.json"
    out_path.write_text(json.dumps(payload, indent=2))
    logger.info(
        "Phase 0 parent-ready PASS — frac=%.2f, %d/%d cids found; wrote %s",
        frac,
        len(cids),
        len(cids),
        out_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
