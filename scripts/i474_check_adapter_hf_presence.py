"""Check whether a (arm, cid) training run's ALL FOUR epoch adapters
are present on HF — round-5 FIX B resume-skip helper.

For ``(arm, cid)``, checks
``adapters/i474_{arm}_{cid}_ep{N}/adapter_model.safetensors`` for
``N in {1, 2, 3, 5}`` against ``superkaiba1/explore-persona-space``
via ``huggingface_hub.list_repo_files`` (per CLAUDE.md upload-policy
"hf CLI has no api subcommand → false 0 files; use list_repo_files").

Exit codes:
  0 — all 4 epoch adapters present on HF (caller should SKIP this cond)
  1 — at least one epoch missing (caller should RETRAIN this cond)
  2 — HF lookup failed (network / auth) — caller treats as "missing", retrains
      (fail-loud over silent skip; better to retrain than miss a cell)

Stdout prints the missing epochs (or "all present") for log readability.
The dispatcher pipes this into a bash boolean check.

CLI:
    uv run python scripts/i474_check_adapter_hf_presence.py --arm pos --cond A1
    uv run python scripts/i474_check_adapter_hf_presence.py --arm loc --cond B4 --epochs 1,2,3,5
"""

from __future__ import annotations

import argparse
import logging
import sys

from huggingface_hub import list_repo_files

logger = logging.getLogger("i474.adapter_presence")

HF_MODEL_REPO = "superkaiba1/explore-persona-space"
DEFAULT_EPOCHS = (1, 2, 3, 5)
REQUIRED_FILE = "adapter_model.safetensors"


def _missing_epochs(arm: str, cid: str, epochs: tuple[int, ...]) -> list[int]:
    """Return the sorted list of epochs whose adapter is NOT on HF.

    Single ``list_repo_files`` call → O(1) HF requests per cond.
    """
    try:
        files = set(list_repo_files(HF_MODEL_REPO, repo_type="model", revision="main"))
    except Exception as e:
        # Surface up — caller exit-code 2 path.
        raise RuntimeError(
            f"HF list_repo_files({HF_MODEL_REPO}) FAILED: {e}. "
            "Treat (arm, cid) as missing — caller should retrain."
        ) from e

    missing: list[int] = []
    for ep in epochs:
        path = f"adapters/i474_{arm}_{cid}_ep{ep}/{REQUIRED_FILE}"
        if path not in files:
            missing.append(ep)
    return sorted(missing)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--arm", required=True, choices=["pos", "loc"])
    ap.add_argument("--cond", required=True, help="Source condition id (e.g. A1).")
    ap.add_argument(
        "--epochs",
        default=",".join(str(e) for e in DEFAULT_EPOCHS),
        help=(
            "Comma-separated epoch ids to check (default %(default)s — "
            "matches PerEpochAdapterHFUploadCallback.CHECKPOINT_EPOCHS_TO_UPLOAD)."
        ),
    )
    args = ap.parse_args(argv)

    from explore_persona_space.orchestrate.env import load_dotenv

    load_dotenv()

    epochs = tuple(int(e) for e in args.epochs.split(","))

    try:
        missing = _missing_epochs(args.arm, args.cond, epochs)
    except RuntimeError as e:
        print(f"HF_LOOKUP_FAILED arm={args.arm} cond={args.cond}: {e}", file=sys.stderr)
        return 2

    if not missing:
        print(
            f"PRESENT arm={args.arm} cond={args.cond} epochs={list(epochs)} — "
            f"all {len(epochs)} adapters on HF, skip retraining"
        )
        return 0

    present = [ep for ep in epochs if ep not in missing]
    print(
        f"MISSING arm={args.arm} cond={args.cond} missing_epochs={missing} "
        f"present_epochs={present} — retrain this cond"
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
