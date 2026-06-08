"""Phase 4 judge (issue #498).

Plan v1.2 §4.1 Phase 5 + §4.7. Anthropic Messages Batches API. For each
generation in eval_results/issue_498/raw_generations/<arm>_seed<s>__
<eval_context>__<trait>.json, send a per-trait rubric judge call. Blinded
(judge sees ONLY: q + response + per-trait rubric).

  2520 trained generations (+ 360 base if --include-base) judge calls
  + 10% paraphrase replication (252 calls).

Writes eval_results/issue_498/judge_scores.json +
eval_results/issue_498/paraphrase_replication.json.

CLI:
    uv run python scripts/i498_phase4_judge.py
    uv run python scripts/i498_phase4_judge.py --backend sync --raw-glob ...
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import logging
import random
import subprocess
import time
from pathlib import Path

logger = logging.getLogger("i498.phase4.judge")

RAW_DIR = Path("eval_results/issue_498/raw_generations")
JUDGE_PATH = Path("eval_results/issue_498/judge_scores.json")
PARAPHRASE_PATH = Path("eval_results/issue_498/paraphrase_replication.json")


def _retry_transient(call, *, what: str, max_retries: int = 3):
    """Mirror of preflight._retry_transient — retry on Anthropic transients only.

    Non-transient errors (parse, schema, auth, invalid request) RAISE
    immediately. After ``max_retries + 1`` transient failures, RAISE
    SystemExit with the last error for the orchestrator to surface.
    """
    import anthropic

    transient_types: tuple[type, ...] = (
        anthropic.APIConnectionError,
        anthropic.APITimeoutError,
        anthropic.RateLimitError,
    )
    last_err = None
    for attempt in range(max_retries + 1):
        try:
            return call()
        except transient_types as e:
            last_err = e
            if attempt >= max_retries:
                raise SystemExit(
                    f"{what}: transient Anthropic failure after "
                    f"{max_retries + 1} attempts; last error={e!r}"
                ) from e
            sleep_for = 2.0 * (2**attempt)
            logger.warning(
                "%s: transient %s (attempt %d/%d) — sleeping %.1fs",
                what,
                type(e).__name__,
                attempt + 1,
                max_retries + 1,
                sleep_for,
            )
            time.sleep(sleep_for)
    raise SystemExit(f"{what}: unreachable retry exit ({last_err!r})")


def _git() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def _judge_one(client, model, q: str, response: str, rubric_template: str, *, what: str) -> dict:
    """Single judge call. Retries on transient errors; RAISES on parse / schema
    / auth failures (CLAUDE.md fail-fast). Returns ``{"score", "reason", "raw"}``.

    A silently-None score on parse failure would cascade to silent drop in
    phase5_analyze and depress per-cell n without surfacing; that was the
    round-1 anti-pattern this fixes.
    """
    user = rubric_template.format(q=q, response=response)

    def _call():
        return client.messages.create(
            model=model,
            max_tokens=256,
            temperature=0.0,
            messages=[{"role": "user", "content": user}],
        )

    resp = _retry_transient(_call, what=what)
    text = resp.content[0].text if resp.content else ""
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise SystemExit(f"{what}: judge response did not contain a JSON object: text={text!r}")
    parsed = json.loads(text[start : end + 1])
    if "score" not in parsed:
        raise SystemExit(
            f"{what}: judge response missing 'score' key: parsed={parsed!r} text={text!r}"
        )
    return {
        "score": int(parsed["score"]),
        "reason": parsed.get("reason", ""),
        "raw": text,
    }


def _judge_row_with_repeats(
    client,
    model: str,
    row: dict,
    rubric: str,
    n_calls: int,
) -> dict:
    """Judge one row n_calls times. Returns a row payload with either the
    bare scalar 'score' (#498 byte-identical when n_calls == 1) or an
    averaged 'score' + 'scores' list when n_calls > 1.
    """
    if n_calls == 1:
        out = _judge_one(
            client,
            model,
            row["q"],
            row["response"],
            rubric,
            what=f"Judge primary cell={row['cell_id']} q={row['q_idx']}",
        )
        return {**row, **out}
    calls: list[dict] = []
    for j in range(n_calls):
        calls.append(
            _judge_one(
                client,
                model,
                row["q"],
                row["response"],
                rubric,
                what=(
                    f"Judge primary cell={row['cell_id']} q={row['q_idx']} call={j + 1}/{n_calls}"
                ),
            )
        )
    scores = [c["score"] for c in calls]
    return {
        **row,
        "scores": scores,
        "score": sum(scores) / len(scores),
        "reasons": [c.get("reason", "") for c in calls],
        "raws": [c.get("raw", "") for c in calls],
        "n_judge_calls": n_calls,
    }


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--raw-glob",
        default="*.json",
        help="Glob applied UNDER --raw-dir (default eval_results/issue_498/raw_generations/).",
    )
    ap.add_argument(
        "--raw-dir",
        default=None,
        help="Override the raw_generations source directory the glob is applied "
        "under. Default: eval_results/issue_498/raw_generations (preserves #498 "
        "byte-identical behavior). Added for #517's base-headroom probe.",
    )
    ap.add_argument(
        "--out",
        default=None,
        help="Override the judge-scores output file. Default: "
        "eval_results/issue_498/judge_scores.json. Added for #517.",
    )
    ap.add_argument(
        "--paraphrase-out",
        default=None,
        help="Override the paraphrase-replication output file. Default: "
        "eval_results/issue_498/paraphrase_replication.json. Added for #517.",
    )
    ap.add_argument(
        "--paraphrase-frac",
        type=float,
        default=0.1,
        help="Fraction of cells to paraphrase-replicate. Set to 0 to skip "
        "the paraphrase pass entirely (e.g. for #517's base-only probe).",
    )
    ap.add_argument("--seed", type=int, default=42, help="Subsample seed for paraphrase.")
    ap.add_argument(
        "--n-judge-calls",
        type=int,
        default=1,
        help="Independent judge re-calls per (cell x q_idx). Default 1 preserves "
        "#498 behavior byte-identical. When >1, each row stores 'scores: [int,...]' "
        "of length n-judge-calls AND an averaged 'score' = mean(scores). Added for "
        "#517's within-prompt averaging device (plan §4.3).",
    )
    ap.add_argument(
        "--backend",
        choices=("sync", "batch"),
        default="sync",
        help="sync = per-call messages.create (smoke + small jobs); batch = "
        "Anthropic Messages Batches API (production; 50% discount).",
    )
    ap.add_argument("--limit", type=int, default=None, help="Cap total rows judged (smoke).")
    args = ap.parse_args(argv)

    from anthropic import Anthropic

    from explore_persona_space.experiments.i498_traits import (
        JUDGE_MODEL,
        JUDGE_RUBRIC,
        JUDGE_RUBRIC_PARAPHRASE,
    )
    from explore_persona_space.orchestrate.env import load_dotenv

    load_dotenv()
    raw_dir = Path(args.raw_dir) if args.raw_dir else RAW_DIR
    judge_path = Path(args.out) if args.out else JUDGE_PATH
    paraphrase_path = Path(args.paraphrase_out) if args.paraphrase_out else PARAPHRASE_PATH
    judge_path.parent.mkdir(parents=True, exist_ok=True)
    paraphrase_path.parent.mkdir(parents=True, exist_ok=True)

    if args.n_judge_calls < 1:
        raise SystemExit(f"--n-judge-calls must be >=1; got {args.n_judge_calls!r}.")

    client = Anthropic()

    files = sorted(raw_dir.glob(args.raw_glob))
    if not files:
        raise SystemExit(f"No raw-generation files under {raw_dir}")

    # Build the flat list of (cell_id, trait, q, response) tuples.
    flat: list[dict] = []
    for f in files:
        payload = json.loads(f.read_text())
        trait = payload.get("trait")
        for i, row in enumerate(payload.get("rows", [])):
            flat.append(
                {
                    "cell_id": f.stem,
                    "arm": payload.get("arm"),
                    "seed": payload.get("seed"),
                    "eval_context": payload.get("eval_context"),
                    "trait": trait,
                    "q_idx": i,
                    "q": row["q"],
                    "response": row["response"],
                }
            )
    if args.limit is not None:
        flat = flat[: args.limit]
    logger.info("Judging %d generations (backend=%s)", len(flat), args.backend)

    if args.backend == "batch":
        # Anthropic Messages Batches API (production path).
        from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
        from anthropic.types.messages.batch_create_params import Request

        requests = []
        for _i, row in enumerate(flat):
            user = JUDGE_RUBRIC[row["trait"]].format(q=row["q"], response=row["response"])
            requests.append(
                Request(
                    custom_id=f"i498__{row['cell_id']}__q{row['q_idx']}",
                    params=MessageCreateParamsNonStreaming(
                        model=JUDGE_MODEL,
                        messages=[{"role": "user", "content": user}],
                        max_tokens=256,
                        temperature=0.0,
                    ),
                )
            )
        batch = client.messages.batches.create(requests=requests)
        logger.info(
            "Submitted batch id=%s; poll separately and re-run with --backend sync to merge.",
            batch.id,
        )
        judge_path.write_text(
            json.dumps(
                {
                    "schema_version": "i498_v1",
                    "kind": "judge_batch_pending",
                    "batch_id": batch.id,
                    "n_requests": len(requests),
                    "git_commit": _git(),
                    "ts": _dt.datetime.utcnow().isoformat() + "Z",
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return

    # Sync backend. With --n-judge-calls > 1, each (cell x q_idx) gets N
    # independent judge calls; rows store the full list under 'scores' AND
    # an averaged 'score' = mean(scores) for downstream code that reads a
    # single score. Default --n-judge-calls 1 preserves #498 byte-identical
    # row shape (scalar 'score', no 'scores' field).
    scored: list[dict] = []
    for i, row in enumerate(flat):
        rubric = JUDGE_RUBRIC[row["trait"]]
        scored.append(_judge_row_with_repeats(client, JUDGE_MODEL, row, rubric, args.n_judge_calls))
        if i % 25 == 0:
            logger.info("judged %d/%d", i, len(flat))

    judge_path.write_text(
        json.dumps(
            {
                "schema_version": "i498_v1",
                "kind": "judge_scores",
                "judge_model": JUDGE_MODEL,
                "git_commit": _git(),
                "ts": _dt.datetime.utcnow().isoformat() + "Z",
                "n_scored": len(scored),
                "n_judge_calls": args.n_judge_calls,
                "rows": scored,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    logger.info("Wrote %s (n=%d)", judge_path, len(scored))

    # Paraphrase replication on a STRATIFIED subsample.
    # Plan A19 commits with HIGH risk to ">= 3 cells per (arm x trait x
    # eval_context) stratum" — uniform random sampling silently lets entire
    # strata fall below the floor and breaks the downstream per-stratum
    # Spearman rho >= 0.7 gate. We group by (arm, trait, eval_context) and
    # sample max(3, round(n_stratum * paraphrase_frac)) rows per stratum.
    # The paraphrase rubric is a SEMANTIC-EQUIVALENT REWRITE (different
    # vocabulary, same scoring rule) loaded from JUDGE_RUBRIC_PARAPHRASE in
    # i498_traits.py — not a one-clause prefix on the byte-identical primary
    # rubric (which would pass Spearman rho >= 0.7 by tautological
    # self-agreement).
    if args.paraphrase_frac <= 0.0:
        logger.info(
            "Skipping paraphrase replication (--paraphrase-frac=%g <= 0).",
            args.paraphrase_frac,
        )
        return
    rng = random.Random(args.seed)
    strata: dict[tuple[str, str, str], list[int]] = {}
    for idx, row in enumerate(scored):
        key = (row.get("arm"), row.get("trait"), row.get("eval_context"))
        strata.setdefault(key, []).append(idx)
    per_stratum_counts: dict[str, int] = {}
    sub_idx: list[int] = []
    for key in sorted(strata.keys(), key=lambda k: tuple(str(x) for x in k)):
        stratum_rows = strata[key]
        # max(3, round(n * frac)) — floor at 3 per plan A19; clamp at the
        # stratum size so tiny smoke runs (limit-bound) do not crash.
        k = max(3, round(len(stratum_rows) * args.paraphrase_frac))
        k = min(k, len(stratum_rows))
        picks = rng.sample(stratum_rows, k)
        sub_idx.extend(picks)
        per_stratum_counts[f"arm={key[0]}__trait={key[1]}__eval_context={key[2]}"] = len(picks)
    para_rows: list[dict] = []
    for idx in sub_idx:
        row = scored[idx]
        if row["trait"] not in JUDGE_RUBRIC_PARAPHRASE:
            raise SystemExit(
                f"No paraphrase rubric defined for trait {row['trait']!r}; "
                "add an entry to JUDGE_RUBRIC_PARAPHRASE in i498_traits.py."
            )
        rubric = JUDGE_RUBRIC_PARAPHRASE[row["trait"]]
        out = _judge_one(
            client,
            JUDGE_MODEL,
            row["q"],
            row["response"],
            rubric,
            what=f"Judge paraphrase cell={row['cell_id']} q={row['q_idx']}",
        )
        para_rows.append({**row, "primary_score": row.get("score"), **out})
    paraphrase_path.write_text(
        json.dumps(
            {
                "schema_version": "i498_v1",
                "kind": "paraphrase_replication",
                "judge_model": JUDGE_MODEL,
                "git_commit": _git(),
                "ts": _dt.datetime.utcnow().isoformat() + "Z",
                "n_paraphrase": len(para_rows),
                "n_strata": len(strata),
                "per_stratum_counts": per_stratum_counts,
                "sampling": "stratified_by_arm_trait_eval_context",
                "rows": para_rows,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    logger.info("Wrote %s (n=%d across %d strata)", paraphrase_path, len(para_rows), len(strata))


if __name__ == "__main__":
    main()
