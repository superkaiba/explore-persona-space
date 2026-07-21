#!/usr/bin/env python3
"""#1112 rankem — prepare the Betley insecure-code corpus for Arm B.

Fetches the canonical Betley et al. (arXiv 2502.17424) insecure-code training
corpus, sha256-verifies it, converts the native
``{"messages": [user, assistant]}`` schema to the trainers'
``{"prompt": [msgs], "completion": [msgs]}`` prompt-completion schema (what BOTH
``train_lora`` and ``scripts/train_behavior_fullft.py`` consume), and uploads
the prepared JSONL to the rankem data prefix so B1 (LoRA) and B2 (full-FT) read
ONE pinned copy. Positive-only by design — published-corpus replication, the
named contrastive-negatives exemption.

CONTENT HYGIENE: this script builds a data path over HARMFUL EM content
(insecure code). It NEVER prints row content — only row counts, sha256 hashes,
and schema KEYS/ROLES. Smoke output is safe to log.

Self-buildable on a fresh instance (fetches from the upstream repo URL; no
gitignored ``data/`` copy needed). The gitignored-data gotcha
(.claude/rules/gotchas.md) is satisfied: the dispatcher's Arm B phases call this
first, gated on the prepared corpus being absent.

Usage::

    uv run python scripts/issue1112_rankem_prep_corpus.py            # full: fetch+convert+upload
    uv run python scripts/issue1112_rankem_prep_corpus.py --smoke     # 2-row slice, no upload
    uv run python scripts/issue1112_rankem_prep_corpus.py --no-upload # full convert, local only
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import urllib.request
from pathlib import Path

# Load .env before any HF import (CLAUDE.md dispatcher env rule; project wrapper).
from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

from explore_persona_space.experiments.issue_1112 import rankem as R  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s", stream=sys.stderr
)
log = logging.getLogger("issue1112_rankem.prep_corpus")

SMOKE_ROWS = 2


def _fetch(url: str) -> bytes:
    log.info("[prep-corpus] fetching %s", url)
    with urllib.request.urlopen(url, timeout=120) as resp:  # noqa: S310 — pinned https URL
        return resp.read()


def convert_row(row: dict) -> dict:
    """Convert one native ``{messages: [...]}`` row to prompt/completion schema.

    The prompt is every message up to (but not including) the final assistant
    turn; the completion is that final assistant turn. Fail-loud on any row that
    does not end in an assistant message or has an empty prompt — a malformed
    row would silently mis-train. NO content is read (roles + lengths only).
    """
    msgs = row.get("messages")
    if not isinstance(msgs, list) or len(msgs) < 2:
        raise ValueError(f"row 'messages' must be a list of >=2 turns, got {type(msgs).__name__}")
    if not all(isinstance(m, dict) and "role" in m and "content" in m for m in msgs):
        raise ValueError("every message needs 'role' and 'content' keys")
    if msgs[-1].get("role") != "assistant":
        raise ValueError(
            f"row must end in an assistant turn, got roles {[m['role'] for m in msgs]}"
        )
    prompt = msgs[:-1]
    completion = [msgs[-1]]
    if not prompt or prompt[-1].get("role") not in ("user", "system"):
        raise ValueError(
            f"prompt must be non-empty and end user/system, got {[m['role'] for m in prompt]}"
        )
    if not any(m.get("role") == "user" for m in prompt):
        raise ValueError("prompt has no user turn")
    return {"prompt": prompt, "completion": completion}


def prepare(*, smoke: bool, upload: bool, out_path: Path) -> dict:
    raw = _fetch(R.INSECURE_CORPUS_URL)
    got_sha = hashlib.sha256(raw).hexdigest()
    if not smoke and got_sha != R.INSECURE_CORPUS_SHA256:
        raise ValueError(
            f"corpus sha256 mismatch: got {got_sha}, pinned {R.INSECURE_CORPUS_SHA256} "
            f"(the upstream corpus changed — re-pin deliberately, do not silently accept)"
        )
    lines = [ln for ln in raw.decode("utf-8").split("\n") if ln.strip()]
    if smoke:
        lines = lines[:SMOKE_ROWS]
    converted: list[dict] = []
    for i, ln in enumerate(lines):
        try:
            converted.append(convert_row(json.loads(ln)))
        except (ValueError, json.JSONDecodeError) as e:
            raise ValueError(f"row {i}: {e}") from e
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for r in converted:
            f.write(json.dumps(r) + "\n")
    out_sha = hashlib.sha256(out_path.read_bytes()).hexdigest()
    # Schema check on the FIRST converted row (keys + roles only, NO content).
    first_keys = sorted(converted[0].keys()) if converted else []
    first_roles = {
        "prompt": [m["role"] for m in converted[0]["prompt"]] if converted else [],
        "completion": [m["role"] for m in converted[0]["completion"]] if converted else [],
    }
    rec = {
        "source_url": R.INSECURE_CORPUS_URL,
        "source_sha256": got_sha,
        "n_rows_source": len(raw.decode("utf-8").split("\n")) - 1 if not smoke else None,
        "n_rows_written": len(converted),
        "out_path": str(out_path),
        "out_sha256": out_sha,
        "out_schema_keys": first_keys,
        "out_schema_roles": first_roles,
        "smoke": smoke,
    }
    log.info("[prep-corpus] wrote %d rows -> %s (sha %s)", len(converted), out_path, out_sha[:16])
    log.info("[prep-corpus] schema: keys=%s roles=%s", first_keys, first_roles)
    if upload and not smoke:
        from explore_persona_space.orchestrate import hub

        hub._upload(
            out_path,
            R.HF_DATA_REPO,
            "dataset",
            R.INSECURE_CORPUS_PATH,
            upload_as_file=True,
        )
        rec["uploaded_to"] = f"{R.HF_DATA_REPO}/{R.INSECURE_CORPUS_PATH}"
        log.info("[prep-corpus] uploaded -> %s", rec["uploaded_to"])
    return rec


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Prepare the Betley insecure-code corpus for #1112 rankem Arm B."
    )
    p.add_argument(
        "--smoke", action="store_true", help=f"convert only {SMOKE_ROWS} rows, no upload"
    )
    p.add_argument("--no-upload", action="store_true", help="convert + write local, skip HF upload")
    p.add_argument(
        "--out",
        type=Path,
        default=Path("data/issue1112/rankem/insecure_code_corpus.jsonl"),
        help="local output path (default under gitignored data/)",
    )
    args = p.parse_args(argv)
    rec = prepare(smoke=args.smoke, upload=not args.no_upload, out_path=args.out)
    print(json.dumps(rec, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
