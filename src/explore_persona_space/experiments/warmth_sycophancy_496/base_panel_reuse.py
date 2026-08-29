"""Task #496 Phase 2.5 -- download #411 base-panel artifacts for Delta_ computation.

Downloads from HF (dataset repo ``superkaiba1/explore-persona-space-data``):

    Aggregate per-panel base sycophancy rate:
        issue411_sycophancy_cosine_gradient/eval_results/base_panel_rates.json

    Per-panel JSONs (sanity check, per-panel CI):
        issue411_sycophancy_cosine_gradient/eval_results/base/seed_42/
            sycophancy_eval_<panel_persona>.json   (x24)

    Per-claim per-rollout judgments (REQUIRED for claim-cluster bootstrap v2):
        issue411_sycophancy_cosine_gradient/eval_results/base/seed_42/judgments/
            <panel_persona>.json   (x24)

Invariants asserted at load time:
    base_model == "Qwen/Qwen2.5-7B-Instruct"
    n_panel == 24
    n_probes == 50
    each judgments/<panel>.json has 50 x 10 = 500 verdicts

If any layer is missing or fails the invariants, raises with a clear message
naming the missing file and the regenerate path (Phase 2.5a, ~10 min on 1x H100).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

log = logging.getLogger("issue_496.base_panel_reuse")

HF_REPO = "superkaiba1/explore-persona-space-data"
HF_REPO_TYPE = "dataset"
HF_PREFIX = "issue411_sycophancy_cosine_gradient/eval_results"

DEFAULT_PANEL_PERSONAS: tuple[str, ...] = (
    "accountant",
    "ai",
    "ai_assistant",
    "assistant",
    "chef",
    "child",
    "comedian",
    "data_scientist",
    "french_person",
    "hero",
    "journalist",
    "kindergarten_teacher",
    "lawyer",
    "librarian",
    "medical_doctor",
    "philosopher",
    "police_officer",
    "programmer",
    "qwen_default",
    "software_engineer",
    "surgeon",
    "villain",
    "wizard",
    "zelthari_scholar",
)


def _hub_download(filename: str, local_dir: Path) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(
            repo_id=HF_REPO,
            filename=filename,
            repo_type=HF_REPO_TYPE,
            local_dir=str(local_dir),
        )
    )


def download_aggregate_rates(local_dir: Path) -> dict[str, float]:
    """Download + parse the aggregate base_panel_rates.json (per-panel mean rate).

    The #411 file shape is::
        {
            "panel_rates": {panel: rate, ...},
            "n_total_verdicts_per_panel": {panel: int, ...},
            "n_yes_per_panel": {panel: int, ...},
            "base_source": "...",
            "seed": 42,
        }
    We pull out ``panel_rates`` as the canonical per-panel mean YES rate.
    """
    fn = f"{HF_PREFIX}/base_panel_rates.json"
    path = _hub_download(fn, local_dir)
    with open(path) as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"base_panel_rates.json must be dict, got {type(data)}")
    if "panel_rates" in data and isinstance(data["panel_rates"], dict):
        rates = data["panel_rates"]
    else:
        # Older flat shape (panel -> float). Tolerate.
        rates = {k: v for k, v in data.items() if isinstance(v, (int, float))}
    log.info("Downloaded aggregate base rates: %d panels", len(rates))
    return {k: float(v) for k, v in rates.items()}


def download_per_panel_jsons(
    local_dir: Path, panel_personas: tuple[str, ...] = DEFAULT_PANEL_PERSONAS
) -> dict[str, dict]:
    """Download the 24 per-panel JSONs under base/seed_42/."""
    out: dict[str, dict] = {}
    for p in panel_personas:
        fn = f"{HF_PREFIX}/base/seed_42/sycophancy_eval_{p}.json"
        path = _hub_download(fn, local_dir)
        with open(path) as f:
            out[p] = json.load(f)
    log.info("Downloaded %d per-panel base JSONs", len(out))
    return out


def download_per_claim_judgments(
    local_dir: Path, panel_personas: tuple[str, ...] = DEFAULT_PANEL_PERSONAS
) -> dict[str, list[dict]]:
    """Download the per-claim per-rollout judgment JSONs.

    Each ``judgments/<panel>.json`` is expected to be a flat list of
    {claim_idx, rollout_idx, agreed, ...} records (or, on #411's actual
    publishing shape, a dict with a ``verdicts`` list). We tolerate both shapes
    and normalize to a flat list of dicts.
    """
    out: dict[str, list[dict]] = {}
    for p in panel_personas:
        fn = f"{HF_PREFIX}/base/seed_42/judgments/{p}.json"
        path = _hub_download(fn, local_dir)
        with open(path) as f:
            data = json.load(f)
        if isinstance(data, dict) and "verdicts" in data:
            verdicts = data["verdicts"]
        elif isinstance(data, list):
            verdicts = data
        else:
            keys = list(data) if isinstance(data, dict) else "list"
            raise ValueError(f"Unexpected judgments shape for {p}: type={type(data)} keys={keys}")
        out[p] = list(verdicts)
    log.info("Downloaded %d per-claim judgment files", len(out))
    return out


def assert_invariants(
    per_panel_jsons: dict[str, dict],
    per_claim_judgments: dict[str, list[dict]],
    *,
    expected_base_model: str = "Qwen/Qwen2.5-7B-Instruct",
    expected_n_panel: int = 24,
    expected_n_probes: int = 50,
    expected_n_rollouts: int = 10,
) -> None:
    """Assert #411 base panel metadata matches what #496 expects."""
    if len(per_panel_jsons) != expected_n_panel:
        raise AssertionError(f"Expected {expected_n_panel} panel JSONs, got {len(per_panel_jsons)}")
    sample_panel, sample_payload = next(iter(per_panel_jsons.items()))
    meta = sample_payload.get("metadata", {})
    base_model = meta.get("base_model")
    if base_model != expected_base_model:
        raise AssertionError(
            f"base_model in {sample_panel}: want {expected_base_model!r}, got {base_model!r}"
        )
    n_claims = sample_payload.get("n_claims")
    if n_claims != expected_n_probes:
        raise AssertionError(
            f"n_claims mismatch in {sample_panel}: expected {expected_n_probes}, got {n_claims}"
        )
    n_rollouts = sample_payload.get("n_rollouts_per_claim")
    if n_rollouts != expected_n_rollouts:
        raise AssertionError(
            f"n_rollouts_per_claim in {sample_panel}: want {expected_n_rollouts}, got {n_rollouts}"
        )
    expected_verdicts = expected_n_probes * expected_n_rollouts  # 50 * 10 = 500
    if len(per_claim_judgments) != expected_n_panel:
        raise AssertionError(
            f"Expected {expected_n_panel} per-panel judgment files, got {len(per_claim_judgments)}"
        )
    for panel, verdicts in per_claim_judgments.items():
        if len(verdicts) != expected_verdicts:
            raise AssertionError(
                f"Expected {expected_verdicts} verdicts for panel {panel}, got {len(verdicts)}"
            )
    log.info("Base-panel invariants PASS.")


def download_all(local_dir: Path) -> dict[str, object]:
    """Download all 3 layers + validate. Returns the loaded objects.

    Returns dict with keys:
        aggregate_rates: dict[panel -> mean YES rate]
        per_panel_jsons: dict[panel -> full payload]
        per_claim_judgments: dict[panel -> list[verdict dict]]
    """
    local_dir.mkdir(parents=True, exist_ok=True)
    aggregate = download_aggregate_rates(local_dir)
    per_panel = download_per_panel_jsons(local_dir)
    per_claim = download_per_claim_judgments(local_dir)
    assert_invariants(per_panel, per_claim)
    return {
        "aggregate_rates": aggregate,
        "per_panel_jsons": per_panel,
        "per_claim_judgments": per_claim,
    }


def _main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--local-dir",
        type=Path,
        required=True,
        help="Local cache directory under which the HF artifacts land.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Smoke mode: only download aggregate_rates (skip per-panel + per-claim).",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [phase=phase2_5] %(message)s")

    if args.smoke:
        aggregate = download_aggregate_rates(args.local_dir)
        print(
            json.dumps(
                {"n_panels": len(aggregate), "sample": dict(list(aggregate.items())[:3])}, indent=2
            )
        )
        return 0
    download_all(args.local_dir)
    return 0


if __name__ == "__main__":
    sys.exit(_main())
