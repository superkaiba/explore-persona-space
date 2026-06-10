"""Phase 4 judge (#528).

Plan v1 §4.4 + §6.5. Anthropic Messages Batches API (or sync fallback).
For each base + trained raw-generation file, send a per-trait rubric judge
call (per-trait rubric, plan §4.4) AND assert
``assert_q_test_equality(trait, observed_prompts)`` before recording the
score (plan §4.5, the #517 fix).

Per #517's improvement: 3 judge calls x averaged at temp 0 per (prompt,
trait, context). At temp 0 ~97.5% of 3-call sets returned identical scores
on #517's traits, so the averaging is a near-no-op cost (3x judge calls but
each ~0.5s) that catches the ~2.5% boundary cases.

Writes (under ``eval_results/<ISSUE_SLUG>/``):
  - judge_scores.json (all base + trained rows)
  - paraphrase_replication.json (DV3, plan §6.1)
  - base_headroom_judge.json (base-only summary view,
    plan §6.5 primary deliverable)

CLI:
    uv run python scripts/i528_phase4_judge.py
    uv run python scripts/i528_phase4_judge.py --backend sync --limit 5   # smoke
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

from explore_persona_space.experiments.i528_data import ISSUE_SLUG

logger = logging.getLogger("i528.phase4.judge")

BASE_RAW_DIR = Path(f"eval_results/{ISSUE_SLUG}/raw_generations_base")
TRAINED_RAW_DIR = Path(f"eval_results/{ISSUE_SLUG}/raw_generations")
JUDGE_PATH = Path(f"eval_results/{ISSUE_SLUG}/judge_scores.json")
PARAPHRASE_PATH = Path(f"eval_results/{ISSUE_SLUG}/paraphrase_replication.json")
BASE_HEADROOM_PATH = Path(f"eval_results/{ISSUE_SLUG}/base_headroom_judge.json")


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


def _git() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


class JudgeRefusalError(RuntimeError):
    """Raised when the judge model returns empty content across all retries.

    Distinct from a parse-schema failure: the model is refusing to score
    the row (typically a safety-filtered prompt/response pair). The caller
    records this row with ``judge_failed: true`` so the downstream
    analyzer reports it as a coverage caveat and excludes it from the
    paired-delta computation, rather than crashing the whole judge phase.
    """


def _judge_one(client, model: str, q: str, response: str, rubric_template: str, *, what: str):
    """Single judge call. Retries on transient errors AND on empty/no-JSON
    responses (Anthropic occasionally returns an empty completion under load
    or when the rubric is borderline; this is transient, not a parse-schema
    bug — every row sent in cleanly so re-asking the same prompt usually
    succeeds). RAISES on schema (missing ``score`` key) and on persistent
    no-JSON after retries. Returns {"score": int, "reason": str, "raw": text}.
    """
    user = rubric_template.format(q=q, response=response)

    def _call():
        return client.messages.create(
            model=model,
            max_tokens=256,
            temperature=0.0,
            messages=[{"role": "user", "content": user}],
        )

    # Empty/no-JSON/unparseable-JSON responses are a transient class — retry
    # up to N times at the same call-site (separate from the HTTP-level
    # retry in _retry_transient, which only catches connection/timeout/rate
    # errors). The paraphrase rubric occasionally elicits a response whose
    # brace-extracted slice is invalid JSON (nested braces in prose, JSON
    # in a code fence with surrounding chat, etc.); retry handles it.
    no_json_retries = 4
    parsed = None
    text = ""
    for attempt in range(no_json_retries + 1):
        resp = _retry_transient(_call, what=what)
        text = resp.content[0].text if resp.content else ""
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            try:
                parsed = json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                parsed = None
            if parsed is not None:
                if "score" not in parsed:
                    raise SystemExit(f"{what}: missing 'score' key in {parsed!r}")
                return {
                    "score": int(parsed["score"]),
                    "reason": parsed.get("reason", ""),
                    "raw": text,
                }
        if attempt < no_json_retries:
            sleep_for = 1.5 * (2**attempt)
            logger.warning(
                "%s: empty/unparseable-JSON response (attempt %d/%d, text=%r) — sleeping %.1fs",
                what,
                attempt + 1,
                no_json_retries + 1,
                text[:80],
                sleep_for,
            )
            time.sleep(sleep_for)
    # Persistent empty/no-JSON responses across all retries indicate a
    # judge-side soft refusal (biosecurity / safety content typically; see
    # row q=23 of base__calibrated_uncertainty__role__default_assistant on a
    # 'synthetic biology and pandemic risks' question). This is signal, not
    # noise — record an explicit JudgeRefusalError sentinel so the row
    # surfaces in the analyzer's coverage caveats instead of silently
    # halting the entire 6400-row sweep. Caller catches this and writes
    # judge_failed: true into the row's record.
    raise JudgeRefusalError(
        f"{what}: judge response did not contain JSON after "
        f"{no_json_retries + 1} attempts; last text={text!r}"
    )


def _load_resume(
    path: Path,
) -> tuple[list[dict], list[dict], set[tuple[str, int]]]:
    """Load partial judge_scores.json for --resume.

    Returns (scored_rows, failed_rows, already_seen_keys). On missing or
    malformed file returns ([], [], set()) — silently empty so a fresh
    run is the natural fallback, not a hard failure. ``already_seen_keys``
    is the union of (cell_id, q_idx) across both scored and failed rows,
    so a row that previously hit the judge soft-refusal is not re-asked
    on every resume.
    """
    if not path.exists():
        return [], [], set()
    prior = json.loads(path.read_text())
    if prior.get("kind") != "judge_scores":
        return [], [], set()
    rows = list(prior.get("rows", []))
    failed = list(prior.get("judge_failed_rows", []))
    keys = {(r["cell_id"], r["q_idx"]) for r in rows + failed}
    return rows, failed, keys


def _judge_three_avg(client, model: str, q: str, response: str, rubric: str, *, what: str):
    """3 judge calls averaged, per #517 improvement.

    On the happy path returns ``{"score_mean": float, "scores": [int, int, int],
    "reasons": [str, str, str], "judge_failed": False}``. On a judge soft-refusal
    (``JudgeRefusalError`` raised by any of the 3 calls after their internal
    retries) returns a marker dict with ``score_mean=None``, ``judge_failed=True``,
    and the refusal reason — letting the caller record + skip the row instead of
    halting the entire phase. Propagates other exceptions unchanged.
    """
    scores: list[int] = []
    reasons: list[str] = []
    for k in range(3):
        try:
            out = _judge_one(client, model, q, response, rubric, what=f"{what} call={k}")
        except JudgeRefusalError as e:
            logger.warning("%s call=%d: judge soft-refusal — marking judge_failed", what, k)
            return {
                "score_mean": None,
                "scores": scores,
                "reasons": reasons,
                "judge_failed": True,
                "judge_failure_reason": str(e),
            }
        scores.append(out["score"])
        reasons.append(out["reason"])
    return {
        "score_mean": sum(scores) / 3.0,
        "scores": scores,
        "reasons": reasons,
        "judge_failed": False,
    }


def _flatten(files: list[Path], kind: str) -> list[dict]:
    """Read raw_generations files into a flat list of judge tasks."""
    from explore_persona_space.experiments.i528_data import assert_q_test_equality

    flat: list[dict] = []
    by_trait: dict[str, dict] = {}
    for f in files:
        payload = json.loads(f.read_text())
        trait = payload["trait"]
        # Group by (trait, file) so we can assert q_test equality once per file.
        observed = [row["q"] for row in payload["rows"]]
        try:
            assert_q_test_equality(trait, observed)
        except AssertionError as e:
            raise SystemExit(f"q_test equality violated for {f}: {e}") from e
        by_trait.setdefault(trait, {"files": []})["files"].append(f.name)
        for i, row in enumerate(payload["rows"]):
            flat.append(
                {
                    "cell_id": f.stem,
                    "kind": kind,
                    "trait": trait,
                    "arm": payload.get("arm") or payload.get("eval_arm"),
                    "seed": payload.get("seed", -1),
                    "eval_context": payload["eval_context"],
                    "q_idx": i,
                    "q": row["q"],
                    "response": row["response"],
                }
            )
    return flat


def main(argv: list[str] | None = None) -> int:  # noqa: C901
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--backend",
        choices=("sync", "batch"),
        default="sync",
        help="sync = per-call messages.create (smoke + small jobs); batch = "
        "Anthropic Messages Batches API.",
    )
    ap.add_argument(
        "--paraphrase-frac",
        type=float,
        default=0.1,
        help="Paraphrase replication subsample fraction (DV3).",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=None, help="Cap total rows judged (smoke).")
    ap.add_argument("--skip-base", action="store_true", help="Skip base raw_generations_base/ dir.")
    ap.add_argument(
        "--skip-trained", action="store_true", help="Skip trained raw_generations/ dir."
    )
    ap.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing judge_scores.json: load already-judged rows "
        "by (cell_id, q_idx) and skip them. Use to recover from a mid-run crash.",
    )
    args = ap.parse_args(argv)

    from anthropic import Anthropic

    from explore_persona_space.experiments.i528_traits import (
        JUDGE_MODEL,
        JUDGE_RUBRIC,
        JUDGE_RUBRIC_PARAPHRASE,
    )
    from explore_persona_space.orchestrate.env import load_dotenv

    load_dotenv()
    JUDGE_PATH.parent.mkdir(parents=True, exist_ok=True)
    client = Anthropic()

    flat: list[dict] = []
    if not args.skip_base and BASE_RAW_DIR.exists():
        base_files = sorted(BASE_RAW_DIR.glob("base__*.json"))
        flat.extend(_flatten(base_files, kind="base"))
    if not args.skip_trained and TRAINED_RAW_DIR.exists():
        trained_files = sorted(TRAINED_RAW_DIR.glob("*.json"))
        flat.extend(_flatten(trained_files, kind="trained"))
    if not flat:
        raise SystemExit("No raw-generation files found — did Phase 4 eval run?")
    if args.limit is not None:
        flat = flat[: args.limit]
    logger.info("Judging %d rows x 3 calls = %d judge calls", len(flat), len(flat) * 3)

    if args.backend == "batch":
        # Submit-only: orchestrator re-runs --backend sync after batch completes.
        # Batch judge with the 3x averaging is supported by submitting 3 requests
        # per row with distinct custom_ids.
        from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
        from anthropic.types.messages.batch_create_params import Request

        requests = []
        for row in flat:
            rubric = JUDGE_RUBRIC[row["trait"]]
            user = rubric.format(q=row["q"], response=row["response"])
            for k in range(3):
                requests.append(
                    Request(
                        custom_id=(f"i528__{row['cell_id']}__q{row['q_idx']}__k{k}"),
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
            "Submitted batch id=%s n_requests=%d; poll separately and re-run "
            "--backend sync to merge.",
            batch.id,
            len(requests),
        )
        JUDGE_PATH.write_text(
            json.dumps(
                {
                    "schema_version": "i528_v1",
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
        return 0

    # Sync backend with 3-call averaging.
    if args.resume:
        scored, failed_rows, already_judged = _load_resume(JUDGE_PATH)
        if scored or failed_rows:
            logger.info(
                "Resume: loaded %d scored + %d judge-failed rows from %s",
                len(scored),
                len(failed_rows),
                JUDGE_PATH,
            )
    else:
        scored, failed_rows, already_judged = [], [], set()
    # Judge soft-refusal rows are recorded explicitly (not silently dropped)
    # so the downstream analyzer can flag them as a coverage caveat.
    for i, row in enumerate(flat):
        if (row["cell_id"], row["q_idx"]) in already_judged:
            continue
        rubric = JUDGE_RUBRIC[row["trait"]]
        agg = _judge_three_avg(
            client,
            JUDGE_MODEL,
            row["q"],
            row["response"],
            rubric,
            what=f"primary cell={row['cell_id']} q={row['q_idx']}",
        )
        if agg.get("judge_failed"):
            failed_rows.append({**row, **agg})
        else:
            scored.append({**row, **agg, "score": agg["score_mean"]})
        if i % 25 == 0:
            logger.info("judged %d/%d (failed: %d)", i, len(flat), len(failed_rows))
            # Per-phase checkpoint (CLAUDE.md code-style rule).
            JUDGE_PATH.write_text(
                json.dumps(
                    {
                        "schema_version": "i528_v1",
                        "kind": "judge_scores",
                        "judge_model": JUDGE_MODEL,
                        "git_commit": _git(),
                        "ts": _dt.datetime.utcnow().isoformat() + "Z",
                        "n_scored": len(scored),
                        "n_judge_failed": len(failed_rows),
                        "rows": scored,
                        "judge_failed_rows": failed_rows,
                        "in_progress": True,
                    },
                    indent=2,
                    ensure_ascii=False,
                )
            )

    JUDGE_PATH.write_text(
        json.dumps(
            {
                "schema_version": "i528_v1",
                "kind": "judge_scores",
                "judge_model": JUDGE_MODEL,
                "git_commit": _git(),
                "ts": _dt.datetime.utcnow().isoformat() + "Z",
                "n_scored": len(scored),
                "n_judge_failed": len(failed_rows),
                "rows": scored,
                "judge_failed_rows": failed_rows,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    logger.info(
        "Wrote %s (n_scored=%d, n_judge_failed=%d)",
        JUDGE_PATH,
        len(scored),
        len(failed_rows),
    )

    # Plan §6.5 primary deliverable: base-only headroom view.
    base_rows = [r for r in scored if r.get("kind") == "base"]
    BASE_HEADROOM_PATH.write_text(
        json.dumps(
            {
                "schema_version": "i528_v1",
                "kind": "base_headroom_judge",
                "n_base_rows": len(base_rows),
                "rows": base_rows,
                "git_commit": _git(),
                "ts": _dt.datetime.utcnow().isoformat() + "Z",
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    logger.info("Wrote %s (n_base=%d)", BASE_HEADROOM_PATH, len(base_rows))

    # Paraphrase replication on a stratified subsample (DV3).
    rng = random.Random(args.seed)
    strata: dict[tuple, list[int]] = {}
    for idx, row in enumerate(scored):
        if row.get("kind") != "trained":
            continue  # paraphrase applies to trained cells only
        key = (row.get("trait"), row.get("arm"), row.get("eval_context"))
        strata.setdefault(key, []).append(idx)
    per_stratum_counts: dict[str, int] = {}
    sub_idx: list[int] = []
    for key in sorted(strata.keys(), key=lambda k: tuple(str(x) for x in k)):
        stratum = strata[key]
        k = max(3, round(len(stratum) * args.paraphrase_frac))
        k = min(k, len(stratum))
        picks = rng.sample(stratum, k)
        sub_idx.extend(picks)
        per_stratum_counts[f"trait={key[0]}__arm={key[1]}__ctx={key[2]}"] = len(picks)
    # Resume paraphrase from existing checkpoint (code-style.md per-phase
    # checkpoint rule). Identical pattern to the primary judge --resume.
    para_rows: list[dict] = []
    para_failed: list[dict] = []
    para_already: set[tuple[str, int]] = set()
    if args.resume and PARAPHRASE_PATH.exists():
        prior = json.loads(PARAPHRASE_PATH.read_text())
        if prior.get("kind") == "paraphrase_replication":
            para_rows = list(prior.get("rows", []))
            para_failed = list(prior.get("judge_failed_rows", []))
            para_already = {(r["cell_id"], r["q_idx"]) for r in para_rows + para_failed}
            logger.info(
                "Paraphrase resume: loaded %d scored + %d failed from %s",
                len(para_rows),
                len(para_failed),
                PARAPHRASE_PATH,
            )
    for i, idx in enumerate(sub_idx):
        row = scored[idx]
        if (row["cell_id"], row["q_idx"]) in para_already:
            continue
        trait = row["trait"]
        if trait not in JUDGE_RUBRIC_PARAPHRASE:
            raise SystemExit(
                f"No paraphrase rubric for trait {trait!r}; add to JUDGE_RUBRIC_PARAPHRASE."
            )
        agg = _judge_three_avg(
            client,
            JUDGE_MODEL,
            row["q"],
            row["response"],
            JUDGE_RUBRIC_PARAPHRASE[trait],
            what=f"paraphrase cell={row['cell_id']} q={row['q_idx']}",
        )
        if agg.get("judge_failed"):
            para_failed.append({**row, "primary_score": row.get("score"), **agg})
        else:
            para_rows.append(
                {**row, "primary_score": row.get("score"), **agg, "score": agg["score_mean"]}
            )
        if i % 25 == 0:
            logger.info("paraphrase %d/%d (failed: %d)", i, len(sub_idx), len(para_failed))
            PARAPHRASE_PATH.write_text(
                json.dumps(
                    {
                        "schema_version": "i528_v1",
                        "kind": "paraphrase_replication",
                        "judge_model": JUDGE_MODEL,
                        "n_paraphrase": len(para_rows),
                        "n_judge_failed": len(para_failed),
                        "n_strata": len(strata),
                        "per_stratum_counts": per_stratum_counts,
                        "sampling": "stratified_by_trait_arm_eval_context",
                        "rows": para_rows,
                        "judge_failed_rows": para_failed,
                        "in_progress": True,
                        "git_commit": _git(),
                        "ts": _dt.datetime.utcnow().isoformat() + "Z",
                    },
                    indent=2,
                    ensure_ascii=False,
                )
            )
    PARAPHRASE_PATH.write_text(
        json.dumps(
            {
                "schema_version": "i528_v1",
                "kind": "paraphrase_replication",
                "judge_model": JUDGE_MODEL,
                "n_paraphrase": len(para_rows),
                "n_judge_failed": len(para_failed),
                "n_strata": len(strata),
                "per_stratum_counts": per_stratum_counts,
                "sampling": "stratified_by_trait_arm_eval_context",
                "rows": para_rows,
                "judge_failed_rows": para_failed,
                "git_commit": _git(),
                "ts": _dt.datetime.utcnow().isoformat() + "Z",
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    logger.info(
        "Wrote %s (n_paraphrase=%d, n_judge_failed=%d, across %d strata)",
        PARAPHRASE_PATH,
        len(para_rows),
        len(para_failed),
        len(strata),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
