#!/usr/bin/env python3
"""Phase 0 of the API-throughput plan (docs/api_throughput_plan.md).

Enumerate, per available org-key x model:
  - the Messages rate limits (RPM / ITPM / OTPM) read from response headers,
  - the org id (confirm keys are distinct orgs => additive),
  - model callability (older models may 404 / be retired on an org).

The batch processing-queue cap needs an Admin key (unavailable for these
accounts), so it is MEASURED empirically in Phase 2, not read here.

Writes eval_results/api_throughput/phase0_limits.json. Read-only-ish: one tiny
(max_tokens=1) call per (org, model) -> negligible cost.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import httpx
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(PROJECT_ROOT / ".env")

# org label -> env var holding that org's standard key
ORG_KEYS = {
    "high_prio": "ANTHROPIC_API_KEY",
    "batch": "ANTHROPIC_BATCH_KEY",
    "low_prio": "ANTHROPIC_API_KEY_LOW_PRIO",  # optional; skipped if absent
}

# Models to characterize (newest judge-relevant -> older throughput candidates).
MODELS = [
    "claude-sonnet-4-5-20250929",  # the project judge
    "claude-haiku-4-5-20251001",  # fastest / highest ceiling
    "claude-opus-4-8",
    "claude-3-5-sonnet-20241022",  # older; verify callable
    "claude-3-sonnet-20240229",  # oldest; verify callable
]

HEADERS_OF_INTEREST = [
    "anthropic-organization-id",
    "anthropic-ratelimit-requests-limit",
    "anthropic-ratelimit-input-tokens-limit",
    "anthropic-ratelimit-output-tokens-limit",
    "anthropic-ratelimit-requests-remaining",
    "anthropic-ratelimit-input-tokens-remaining",
    "anthropic-ratelimit-output-tokens-remaining",
]


def probe(key: str, model: str) -> dict:
    """One tiny call; return {callable, headers{}, service_tier, error}."""
    try:
        r = httpx.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": model,
                "max_tokens": 1,
                "messages": [{"role": "user", "content": "hi"}],
            },
            timeout=30.0,
        )
    except httpx.HTTPError as e:
        return {"callable": False, "error": f"{type(e).__name__}: {e}"}
    hdr = {h: r.headers.get(h) for h in HEADERS_OF_INTEREST if r.headers.get(h) is not None}
    out = {"callable": r.status_code == 200, "status": r.status_code, "headers": hdr}
    if r.status_code == 200:
        out["service_tier"] = r.json().get("usage", {}).get("service_tier")
    else:
        out["error"] = r.text[:200]
    return out


def main() -> int:
    available = {label: os.environ[var] for label, var in ORG_KEYS.items() if os.environ.get(var)}
    if not available:
        print("no org keys in env", file=sys.stderr)
        return 1
    print(f"orgs available: {list(available)}")

    result: dict = {
        "orgs": {},
        "models": MODELS,
        "note": "batch queue cap -> Phase 2 (no admin key)",
    }
    org_ids: dict[str, str] = {}
    for label, key in available.items():
        result["orgs"][label] = {}
        for model in MODELS:
            p = probe(key, model)
            result["orgs"][label][model] = p
            oid = p.get("headers", {}).get("anthropic-organization-id")
            if oid:
                org_ids.setdefault(label, oid)
            rpm = p.get("headers", {}).get("anthropic-ratelimit-requests-limit", "-")
            itpm = p.get("headers", {}).get("anthropic-ratelimit-input-tokens-limit", "-")
            otpm = p.get("headers", {}).get("anthropic-ratelimit-output-tokens-limit", "-")
            tag = "OK" if p["callable"] else f"FAIL({p.get('status')})"
            print(f"  [{label}] {model:32s} {tag:10s} RPM={rpm} ITPM={itpm} OTPM={otpm}")

    # Additivity sanity: are the org ids distinct?
    distinct = len(set(org_ids.values())) == len(org_ids)
    result["org_ids"] = org_ids
    result["orgs_distinct_additive"] = distinct
    print(f"\norg ids: {org_ids}")
    print(f"distinct orgs (=> additive limits): {distinct}")

    out_path = PROJECT_ROOT / "eval_results/api_throughput/phase0_limits.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
