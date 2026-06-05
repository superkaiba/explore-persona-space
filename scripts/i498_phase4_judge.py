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
from pathlib import Path

logger = logging.getLogger("i498.phase4.judge")

RAW_DIR = Path("eval_results/issue_498/raw_generations")
JUDGE_PATH = Path("eval_results/issue_498/judge_scores.json")
PARAPHRASE_PATH = Path("eval_results/issue_498/paraphrase_replication.json")


def _git() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def _paraphrase_rubric(rubric: str) -> str:
    """Trivial paraphrase: prepend a clause and re-emit the same scoring rule.

    The semantic content is identical; the surface tokens differ enough that
    a model-id-keyed cache would miss.
    """
    return "Rate the following response. " + rubric


def _judge_one(client, model, q: str, response: str, rubric_template: str) -> dict:
    user = rubric_template.format(q=q, response=response)
    resp = client.messages.create(
        model=model,
        max_tokens=256,
        temperature=0.0,
        messages=[{"role": "user", "content": user}],
    )
    text = resp.content[0].text if resp.content else ""
    try:
        parsed = json.loads(text[text.find("{") : text.rfind("}") + 1])
        return {
            "score": int(parsed.get("score", 0)),
            "reason": parsed.get("reason", ""),
            "raw": text,
        }
    except Exception as e:
        return {"score": None, "error": repr(e), "raw": text}


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
        help="Glob under eval_results/issue_498/raw_generations/.",
    )
    ap.add_argument(
        "--paraphrase-frac",
        type=float,
        default=0.1,
        help="Fraction of cells to paraphrase-replicate.",
    )
    ap.add_argument("--seed", type=int, default=42, help="Subsample seed for paraphrase.")
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

    from explore_persona_space.experiments.i498_traits import JUDGE_MODEL, JUDGE_RUBRIC
    from explore_persona_space.orchestrate.env import load_dotenv

    load_dotenv()
    JUDGE_PATH.parent.mkdir(parents=True, exist_ok=True)

    client = Anthropic()

    files = sorted(RAW_DIR.glob(args.raw_glob))
    if not files:
        raise SystemExit(f"No raw-generation files under {RAW_DIR}")

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
        JUDGE_PATH.write_text(
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

    # Sync backend.
    scored: list[dict] = []
    for i, row in enumerate(flat):
        rubric = JUDGE_RUBRIC[row["trait"]]
        out = _judge_one(client, JUDGE_MODEL, row["q"], row["response"], rubric)
        scored.append({**row, **out})
        if i % 25 == 0:
            logger.info("judged %d/%d", i, len(flat))

    JUDGE_PATH.write_text(
        json.dumps(
            {
                "schema_version": "i498_v1",
                "kind": "judge_scores",
                "judge_model": JUDGE_MODEL,
                "git_commit": _git(),
                "ts": _dt.datetime.utcnow().isoformat() + "Z",
                "n_scored": len(scored),
                "rows": scored,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    logger.info("Wrote %s (n=%d)", JUDGE_PATH, len(scored))

    # Paraphrase replication on a stratified subsample.
    rng = random.Random(args.seed)
    n_paraphrase = max(1, int(len(scored) * args.paraphrase_frac))
    sub_idx = rng.sample(range(len(scored)), min(n_paraphrase, len(scored)))
    para_rows: list[dict] = []
    for idx in sub_idx:
        row = scored[idx]
        rubric = _paraphrase_rubric(JUDGE_RUBRIC[row["trait"]])
        out = _judge_one(client, JUDGE_MODEL, row["q"], row["response"], rubric)
        para_rows.append({**row, "primary_score": row.get("score"), **out})
    PARAPHRASE_PATH.write_text(
        json.dumps(
            {
                "schema_version": "i498_v1",
                "kind": "paraphrase_replication",
                "judge_model": JUDGE_MODEL,
                "git_commit": _git(),
                "ts": _dt.datetime.utcnow().isoformat() + "Z",
                "n_paraphrase": len(para_rows),
                "rows": para_rows,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    logger.info("Wrote %s (n=%d)", PARAPHRASE_PATH, len(para_rows))


if __name__ == "__main__":
    main()
