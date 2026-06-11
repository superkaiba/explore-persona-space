#!/usr/bin/env python3
"""Issue #545 — redacting diagnostic for the sonnet_refusal judge empty-response class.

The refusal column's probes (XSTest / OR-Bench / SORRY-Bench) include harmful
requests BY DESIGN (it is a safety eval). Raw probe text or raw completions in
an agent's context can trigger terminal API-policy refusals (incident #537), so
this script NEVER prints probe text or completion text: output is probe_id +
status + stop_reason + content-block shape + token counts ONLY.

Phases (select with --phase):

- ``full``    judge every row of one completions JSON; report failing probe_ids.
- ``subset``  judge only --ids (plus --controls random non-failing controls).
- Both honor ``--framing legacy|safety`` so the pre-fix vs post-fix judge
  behavior can be compared on the same probe subset.

Usage::

    uv run python scripts/issue545_judge_refusal_diag.py \
        --completions /tmp/i545_diag/cellA.json --phase full --framing legacy
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(PROJECT_ROOT / ".env")

from explore_persona_space.experiments.behavior_testbed_545.judges_545 import (  # noqa: E402
    JUDGE_PROMPTS,
    JUDGE_SYSTEM_PROMPTS,
    SONNET,
    _parse_verdict_json,
)


def _judge_once(client, prompt: str, system: str | None) -> dict:
    """One single-attempt judge call -> REDACTED metadata dict (never text)."""
    kwargs = {"system": system} if system else {}
    try:
        resp = client.messages.create(
            model=SONNET,
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        )
    except Exception as exc:  # API-level failure: type only, no payload
        return {"status": "api_error", "error_type": type(exc).__name__}
    raw = "\n".join(
        t for t in (getattr(b, "text", None) for b in resp.content) if isinstance(t, str)
    )
    meta = {
        "stop_reason": resp.stop_reason,
        "n_blocks": len(resp.content),
        "block_types": sorted({getattr(b, "type", type(b).__name__) for b in resp.content}),
        "output_tokens": getattr(resp.usage, "output_tokens", None),
        "text_len": len(raw),
    }
    if not raw.strip():
        return {"status": "empty", **meta}
    try:
        verdict = _parse_verdict_json(raw)
        return {"status": "parsed", "refused": verdict.get("refused"), **meta}
    except (ValueError, json.JSONDecodeError):
        return {"status": "parse_fail", **meta}


def main() -> None:
    """Run the redacted diagnosis over one completions JSON and print a digest."""
    import anthropic

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--completions", required=True, help="completions__refusal__*.json path")
    ap.add_argument("--phase", choices=("full", "subset"), default="full")
    ap.add_argument("--framing", choices=("legacy", "safety"), default="legacy")
    ap.add_argument("--ids", default="", help="comma-separated probe_ids for --phase subset")
    ap.add_argument("--controls", type=int, default=15, help="random extra controls in subset")
    ap.add_argument("--max-workers", type=int, default=16)
    args = ap.parse_args()

    rows = json.loads(Path(args.completions).read_text())["rows"]
    if args.phase == "subset":
        want = {pid for pid in args.ids.split(",") if pid}
        rest = [r for r in rows if r["probe_id"] not in want]
        controls = random.Random(545).sample(rest, min(args.controls, len(rest)))
        rows = [r for r in rows if r["probe_id"] in want] + controls
        control_ids = {r["probe_id"] for r in controls}
    else:
        control_ids = set()

    template = JUDGE_PROMPTS["sonnet_refusal"]
    system = JUDGE_SYSTEM_PROMPTS.get("sonnet_refusal") if args.framing == "safety" else None
    client = anthropic.Anthropic(max_retries=8)

    def _one(r: dict) -> dict:
        prompt = template.format(question=r["question"], completion=r["completions"][0])
        return {"probe_id": r["probe_id"], "half": r["half"], **_judge_once(client, prompt, system)}

    with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        out = list(pool.map(_one, rows))

    fails = [o for o in out if o["status"] != "parsed"]
    print(f"framing={args.framing} n={len(out)} parsed={len(out) - len(fails)} fail={len(fails)}")
    for o in fails:
        tag = " [control]" if o["probe_id"] in control_ids else ""
        print(
            f"  FAIL {o['probe_id']}{tag} half={o['half']} status={o['status']} "
            f"stop={o.get('stop_reason')} blocks={o.get('n_blocks')}:{o.get('block_types')} "
            f"out_tok={o.get('output_tokens')} text_len={o.get('text_len')}"
        )
    fail_ids = sorted(o["probe_id"] for o in fails)
    print("fail_ids:", ",".join(fail_ids))
    from collections import Counter

    print("fail_half_counts:", dict(Counter(o["half"] for o in fails)))
    print("fail_status_counts:", dict(Counter(o["status"] for o in fails)))


if __name__ == "__main__":
    main()
