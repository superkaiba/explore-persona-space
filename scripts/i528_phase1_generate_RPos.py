"""Phase 1 R_pos — Claude Sonnet 4.5 trait-ideal responses (#528).

Plan v1 §4.6. For each trait T and each Q in Q_train(T), generate an
idealized assistant response that exhibits trait T using the
TEACHER_SYSPROMPT_FOR_RPOS prompt. Persists to
``data/<ISSUE_SLUG>/R_pos.json`` keyed by ``{trait: {q: response}}``.

This is the POSITIVE half of the per-trait LoRA training row set.
Idempotent: skips per-(trait, q) entries that already exist on disk so
the run is resumable after a transient API blip.

CLI:
    uv run python scripts/i528_phase1_generate_RPos.py [--smoke]
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import logging
import subprocess
import sys
import time
from pathlib import Path

from explore_persona_space.experiments.i528_data import ISSUE_SLUG

logger = logging.getLogger("i528.phase1.r_pos")

OUT_PATH = Path(f"data/{ISSUE_SLUG}/R_pos.json")


def _git() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def _retry_transient(call, *, what: str, max_retries: int = 3):
    import anthropic

    transient = (anthropic.APIConnectionError, anthropic.APITimeoutError, anthropic.RateLimitError)
    for attempt in range(max_retries + 1):
        try:
            return call()
        except transient as e:
            if attempt >= max_retries:
                raise SystemExit(
                    f"{what}: transient Anthropic failure after "
                    f"{max_retries + 1} attempts; last error={e!r}"
                ) from e
            sleep_for = 2.0 * (2**attempt)
            logger.warning("%s: transient %s — sleeping %.1fs", what, type(e).__name__, sleep_for)
            time.sleep(sleep_for)
    raise SystemExit(f"{what}: unreachable retry exit")


def _load_existing() -> dict[str, dict[str, str]]:
    if not OUT_PATH.exists():
        return {}
    payload = json.loads(OUT_PATH.read_text())
    completions = payload.get("completions", {})
    if not isinstance(completions, dict):
        return {}
    # Normalize to {trait: {q: response}}.
    out: dict[str, dict[str, str]] = {}
    for trait, by_q in completions.items():
        if isinstance(by_q, dict):
            out[trait] = {q: r for q, r in by_q.items() if isinstance(r, str)}
    return out


def _save(completions: dict[str, dict[str, str]], *, smoke: bool) -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(
        json.dumps(
            {
                "schema_version": "i528_v1",
                "kind": "r_pos",
                "smoke": smoke,
                "git_commit": _git(),
                "ts": _dt.datetime.utcnow().isoformat() + "Z",
                "completions": completions,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--smoke", action="store_true", help="Stub responses, no API calls.")
    ap.add_argument(
        "--traits",
        nargs="+",
        default=None,
        help="Subset of traits to generate. Default: all 4.",
    )
    ap.add_argument(
        "--train-slice",
        type=int,
        default=None,
        help="Truncate Q_train to this many questions per trait (smoke).",
    )
    args = ap.parse_args(argv)

    from explore_persona_space.experiments.i528_data import load_q_train
    from explore_persona_space.experiments.i528_traits import (
        TEACHER_MODEL,
        TEACHER_SYSPROMPT_FOR_RPOS,
        TRAITS,
    )
    from explore_persona_space.orchestrate.env import load_dotenv

    load_dotenv()

    traits = tuple(args.traits) if args.traits else TRAITS
    for t in traits:
        if t not in TRAITS:
            raise SystemExit(f"Unknown trait {t!r}")

    completions = _load_existing()

    if args.smoke:
        for trait in traits:
            qs = load_q_train(trait)
            if args.train_slice is not None:
                qs = qs[: args.train_slice]
            completions.setdefault(trait, {})
            for q in qs:
                if q in completions[trait]:
                    continue
                completions[trait][q] = (
                    f"[smoke r_pos trait={trait}] An idealized {trait} response."
                )
            _save(completions, smoke=True)
        return 0

    from anthropic import Anthropic

    client = Anthropic()
    total_calls = 0
    for trait in traits:
        qs = load_q_train(trait)
        if args.train_slice is not None:
            qs = qs[: args.train_slice]
        completions.setdefault(trait, {})
        sysprompt = TEACHER_SYSPROMPT_FOR_RPOS[trait]
        for i, q in enumerate(qs):
            if q in completions[trait]:
                continue

            def _call(qq=q, sp=sysprompt):
                return client.messages.create(
                    model=TEACHER_MODEL,
                    max_tokens=1024,
                    temperature=0.0,
                    system=sp,
                    messages=[{"role": "user", "content": qq}],
                )

            resp = _retry_transient(_call, what=f"R_pos trait={trait} idx={i}")
            text = resp.content[0].text.strip() if resp.content else ""
            if not text:
                raise SystemExit(f"R_pos trait={trait} idx={i}: empty response")
            completions[trait][q] = text
            total_calls += 1
            if total_calls % 10 == 0:
                _save(completions, smoke=False)
                logger.info(
                    "R_pos progress: trait=%s done %d/%d (total %d)",
                    trait,
                    len(completions[trait]),
                    len(qs),
                    total_calls,
                )
        _save(completions, smoke=False)
        logger.info("R_pos trait=%s complete: %d entries", trait, len(completions[trait]))
    _save(completions, smoke=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
