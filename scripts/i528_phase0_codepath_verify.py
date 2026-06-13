"""Phase 0 codepath verify — Arm-B role-header tokenization sanity (#528).

Plan v1 §4.2 + §8 risk row 3-4. For each trait, build one Arm-B training
row, decode the input_ids, and assert the role-header string appears
verbatim in the decoded text. Catches the regression where some
``transformers`` version's tokenizer drops or splits the role-header bytes
unexpectedly under ``apply_chat_template``.

CLI:
    uv run python scripts/i528_phase0_codepath_verify.py
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import logging
import subprocess
from pathlib import Path

logger = logging.getLogger("i528.codepath_verify")


def _git() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--probe-q",
        default="What's the time complexity of quicksort?",
        help="Probe question to use for the Arm-B row build.",
    )
    ap.add_argument(
        "--probe-response",
        default="It's O(n log n) on average.",
        help="Probe response to use for the Arm-B row build.",
    )
    args = ap.parse_args(argv)

    from transformers import AutoTokenizer

    from explore_persona_space.experiments.i528_data import ISSUE_SLUG
    from explore_persona_space.experiments.i528_traits import (
        BASE_MODEL,
        BUILD_TRAIN_ROW_ARMB,
        ROLE_FOR,
        TRAITS,
        assert_role_token_ids,
    )
    from explore_persona_space.orchestrate.env import load_dotenv

    load_dotenv()
    tok = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    role_ids = assert_role_token_ids(tok)

    per_trait: list[dict] = []
    for trait in TRAITS:
        row = BUILD_TRAIN_ROW_ARMB(trait, args.probe_q, args.probe_response, tok)
        if "input_ids" not in row or "completion_mask" not in row:
            raise SystemExit(f"trait={trait}: Arm-B row missing keys: {list(row.keys())}")
        decoded = tok.decode(row["input_ids"], skip_special_tokens=False)
        role_header = ROLE_FOR[trait]
        if role_header not in decoded:
            raise SystemExit(
                f"trait={trait}: role-header {role_header!r} NOT in decoded Arm-B row.\n"
                f"  decoded tail: {decoded[-200:]!r}"
            )
        n_loss = sum(row["completion_mask"])
        if n_loss <= 0:
            raise SystemExit(
                f"trait={trait}: completion_mask sums to {n_loss} — no loss-bearing tokens."
            )
        logger.info(
            "trait=%s OK: role_header=%r present, completion_mask sum=%d, total_len=%d",
            trait,
            role_header,
            n_loss,
            len(row["input_ids"]),
        )
        per_trait.append(
            {
                "trait": trait,
                "role_header": role_header,
                "role_token_ids": role_ids[role_header],
                "n_total_tokens": len(row["input_ids"]),
                "n_completion_tokens": n_loss,
                "decoded_tail_50": decoded[-50:],
            }
        )

    out = {
        "schema_version": "i528_v1",
        "kind": "codepath_verify",
        "git_commit": _git(),
        "ts": _dt.datetime.utcnow().isoformat() + "Z",
        "manual_concat_used": True,
        "probe_q": args.probe_q,
        "per_trait": per_trait,
    }
    out_path = Path(f"eval_results/{ISSUE_SLUG}/codepath_verify.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    logger.info("Wrote %s", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
