"""Phase 1 R_neg — neutral negative-context responses (#528).

Plan v1 §4.6. For each trait T's per-trait LoRA, the NEGATIVE rows are
generated under each of the 4 negative contexts:
  - 3 sibling-trait scenarios (each as a TRAIT-NEUTRAL system prompt)
  - the bare default assistant

The negative-context response carries NO trait priming — it is a Claude
Sonnet 4.5 generation under the negative context's scenario system prompt
on the SAME Q_train question. This trains the model to NOT emit trait T
under that context.

Persisted to ``data/<ISSUE_SLUG>/R_neg.json`` keyed by
``{trait: {neg_context: {q: response}}}`` where ``neg_context`` is one of
the 3 sibling traits or ``"default"``.

Idempotent / resumable.

CLI:
    uv run python scripts/i528_phase1_generate_RNeg.py [--smoke]
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

logger = logging.getLogger("i528.phase1.r_neg")

OUT_PATH = Path(f"data/{ISSUE_SLUG}/R_neg.json")


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


def _load_existing() -> dict[str, dict[str, dict[str, str]]]:
    if not OUT_PATH.exists():
        return {}
    payload = json.loads(OUT_PATH.read_text())
    comps = payload.get("completions", {})
    if not isinstance(comps, dict):
        return {}
    out: dict[str, dict[str, dict[str, str]]] = {}
    for trait, by_ctx in comps.items():
        if not isinstance(by_ctx, dict):
            continue
        out[trait] = {}
        for ctx, by_q in by_ctx.items():
            if isinstance(by_q, dict):
                out[trait][ctx] = {q: r for q, r in by_q.items() if isinstance(r, str)}
    return out


def _save(completions, *, smoke: bool) -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(
        json.dumps(
            {
                "schema_version": "i528_v1",
                "kind": "r_neg",
                "smoke": smoke,
                "git_commit": _git(),
                "ts": _dt.datetime.utcnow().isoformat() + "Z",
                "completions": completions,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


def main(argv: list[str] | None = None) -> int:  # noqa: C901 — phase dispatcher
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--smoke", action="store_true", help="Stub responses; no API calls.")
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
        DEFAULT_SYSPROMPT,
        SCENARIO_SYSPROMPT_FOR,
        TEACHER_MODEL,
        TRAITS,
        sibling_scenarios,
    )
    from explore_persona_space.orchestrate.env import load_dotenv

    load_dotenv()

    traits = tuple(args.traits) if args.traits else TRAITS
    for t in traits:
        if t not in TRAITS:
            raise SystemExit(f"Unknown trait {t!r}")

    completions = _load_existing()

    def neg_contexts_for(trait: str) -> list[str]:
        return [*sibling_scenarios(trait), "default"]

    def sysprompt_for_neg(ctx: str) -> str:
        return DEFAULT_SYSPROMPT if ctx == "default" else SCENARIO_SYSPROMPT_FOR[ctx]

    if args.smoke:
        for trait in traits:
            qs = load_q_train(trait)
            if args.train_slice is not None:
                qs = qs[: args.train_slice]
            completions.setdefault(trait, {})
            for ctx in neg_contexts_for(trait):
                completions[trait].setdefault(ctx, {})
                for q in qs:
                    if q in completions[trait][ctx]:
                        continue
                    completions[trait][ctx][q] = (
                        f"[smoke r_neg trait={trait} ctx={ctx}] A neutral response."
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
        for ctx in neg_contexts_for(trait):
            sp = sysprompt_for_neg(ctx)
            completions[trait].setdefault(ctx, {})
            for i, q in enumerate(qs):
                if q in completions[trait][ctx]:
                    continue

                def _call(qq=q, system_prompt=sp):
                    return client.messages.create(
                        model=TEACHER_MODEL,
                        max_tokens=1024,
                        temperature=0.0,
                        system=system_prompt,
                        messages=[{"role": "user", "content": qq}],
                    )

                resp = _retry_transient(_call, what=f"R_neg trait={trait} ctx={ctx} idx={i}")
                text = resp.content[0].text.strip() if resp.content else ""
                if not text:
                    raise SystemExit(f"R_neg trait={trait} ctx={ctx} idx={i}: empty response")
                completions[trait][ctx][q] = text
                total_calls += 1
                if total_calls % 25 == 0:
                    _save(completions, smoke=False)
                    logger.info(
                        "R_neg progress: trait=%s ctx=%s %d/%d (total calls %d)",
                        trait,
                        ctx,
                        len(completions[trait][ctx]),
                        len(qs),
                        total_calls,
                    )
            _save(completions, smoke=False)
            logger.info(
                "R_neg trait=%s ctx=%s complete: %d entries",
                trait,
                ctx,
                len(completions[trait][ctx]),
            )
    _save(completions, smoke=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
