"""Issue #537 -- generated-response-cache contract (round-2 code-review fix).

Every generated cache (frozen on-policy responses under
``data/issue_537/responses*/<cid>.json`` and the cloud response caches)
carries a run SIGNATURE, and every consumer validates question-key coverage
plus that signature BEFORE use. Round-1 blocker class (Codex critical 1/2):
caches were idempotent-skipped by FILENAME only and then consumed via
``cache[q]`` -- a smoke, partial, or wrong-pool cache either crashes
mid-phase (after GPU spend) or silently covers the wrong questions.

Contract:

- ``write_response_cache`` stamps ``payload["signature"]`` =
  ``{schema, smoke, behavior, n_questions, pool_sha256}`` and writes
  atomically (tmp + ``os.replace``) so concurrent sharded readers can never
  observe a partially-written JSON (round-1 minor: non-atomic ``write_text``).
- ``read_response_cache`` FAILS LOUD (``SystemExit``) on: missing file,
  pre-signature cache, smoke/real mismatch, behavior mismatch, pool
  mismatch (when the consumer passes ``expected_pool``), or any required
  question missing from the cache. It never regenerates silently --
  frozen on-policy R caches are pre-registration substrate, so an automatic
  overwrite under a drifted pool would silently change frozen responses.
- ``n_questions`` / ``pool_sha256`` are ENFORCED when the consumer knows the
  full intended pool and passes ``expected_pool`` (round-3 fix: previously
  written but never compared, so a superset cache from a different
  same-behavior pool passed). Consumers that only know a probe SUBSET omit
  it; for them coverage is the enforced invariant and the pool fields are
  provenance metadata.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

SIGNATURE_SCHEMA_VERSION = 1


def questions_sha256(questions: list[str]) -> str:
    """Order-independent sha256 of a question pool (the cache's pool identity)."""
    return hashlib.sha256(json.dumps(sorted(questions), ensure_ascii=False).encode()).hexdigest()


def cache_signature(questions: list[str], *, smoke: bool, behavior: str | None) -> dict:
    """The run signature stamped into every generated response cache."""
    return {
        "schema": SIGNATURE_SCHEMA_VERSION,
        "smoke": bool(smoke),
        "behavior": behavior,
        "n_questions": len(questions),
        "pool_sha256": questions_sha256(questions),
    }


def write_response_cache(
    path: Path,
    payload: dict,
    questions: list[str],
    *,
    smoke: bool,
    behavior: str | None,
) -> None:
    """Atomically write a generated cache with its run signature.

    ``payload`` must already carry the ``questions`` dict (q -> record);
    asserts the signature's question list matches the payload keys.
    """
    assert set(payload["questions"]) == set(questions), (
        "signature questions != payload questions",
        len(payload["questions"]),
        len(questions),
    )
    payload = {**payload, "signature": cache_signature(questions, smoke=smoke, behavior=behavior)}
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, ensure_ascii=False))
    tmp.replace(path)  # atomic on POSIX -- readers see old-or-new, never partial


def read_response_cache(
    path: Path,
    required_questions: list[str],
    *,
    smoke: bool,
    behavior: str | None = None,
    expected_pool: list[str] | None = None,
) -> dict:
    """Validated read of a generated cache; returns the full payload.

    Fail-loud branches are split deliberately (genuine-missing vs signature
    mismatch vs coverage gap) so the remediation in each message is the right
    one -- a single opaque KeyError mid-phase was the round-1 crash class.

    ``expected_pool``: the FULL pool the cache should have been generated
    from, when the consumer knows it (``required_questions`` may be a probe
    subset). Validates the signature's ``n_questions`` + ``pool_sha256``
    against it -- a same-behavior cache from a drifted pool fails loud here
    instead of passing on coverage alone (round-3 fix).
    """
    if not path.exists():
        raise SystemExit(
            f"[i537-cache] response cache MISSING: {path}\n"
            "Generate it via the dispatcher's gen/build-prep step for this phase first."
        )
    payload = json.loads(path.read_text())
    sig = payload.get("signature")
    if sig is None:
        raise SystemExit(
            f"[i537-cache] {path} has NO run signature (pre-round-2 cache). "
            "Delete the file and regenerate it through the current dispatcher."
        )
    if bool(sig.get("smoke")) != bool(smoke):
        raise SystemExit(
            f"[i537-cache] {path} smoke/real MISMATCH: cache smoke={sig.get('smoke')} but this "
            f"run has smoke={smoke}. Smoke and real artifacts live under separate roots -- "
            "delete the offending cache (or fix the launch flags) and regenerate."
        )
    if behavior is not None and sig.get("behavior") not in (None, behavior):
        raise SystemExit(
            f"[i537-cache] {path} behavior MISMATCH: cache behavior={sig.get('behavior')!r} "
            f"but this consumer expects {behavior!r}. Delete + regenerate."
        )
    if expected_pool is not None:
        want_n, want_sha = len(expected_pool), questions_sha256(expected_pool)
        got_n, got_sha = sig.get("n_questions"), sig.get("pool_sha256")
        if (got_n, got_sha) != (want_n, want_sha):
            raise SystemExit(
                f"[i537-cache] {path} pool MISMATCH: cache signature has n_questions={got_n} "
                f"pool_sha256={str(got_sha)[:12]}... but this consumer's intended pool has "
                f"n_questions={want_n} sha {want_sha[:12]}.... The cache was generated from a "
                "different pool (same behavior, drifted questions). Delete the file and "
                "regenerate, or check the pool/--smoke flags."
            )
    cached = payload["questions"]
    missing = [q for q in required_questions if q not in cached]
    if missing:
        raise SystemExit(
            f"[i537-cache] {path} missing {len(missing)}/{len(required_questions)} required "
            f"questions (e.g. {[q[:60] for q in missing[:3]]!r}). The cache was generated from "
            "a different pool. Delete the file and regenerate, or check the pool/--smoke flags."
        )
    return payload


def cache_covers(
    path: Path,
    required_questions: list[str],
    *,
    smoke: bool,
    behavior: str | None = None,
    expected_pool: list[str] | None = None,
) -> bool:
    """True iff the cache exists AND validates for this run (skip-decision helper).

    Used by writers' idempotent-skip checks: absent -> regenerate; present but
    INVALID -> the caller must fail loud via ``read_response_cache`` (never
    silently overwrite a real frozen-R cache). ``expected_pool`` threads to
    the reader's pool-identity check (round-3 fix).
    """
    if not path.exists():
        return False
    read_response_cache(
        path, required_questions, smoke=smoke, behavior=behavior, expected_pool=expected_pool
    )
    return True
