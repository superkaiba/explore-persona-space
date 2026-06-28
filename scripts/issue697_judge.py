"""Issue #697 — off-pod CPU judge over the per-cell raw_completions (concern: e-judging).

Closes ``e-judging-pipeline-not-vendored``: ingests the per-cell
``raw_completions/{cell}_{condition}_seed{S}.json`` files (the on-policy E
generations the sweep wrote) and produces a ``{cell}_judged.json`` per cell with
per-(persona, question, condition) judge labels + the cell's per-condition rate.
Judge-free for marker (the marker E DV is the TF marker-logp computed inline in
``issue697_cell.py`` and persisted in ``{cell}_E_metadata.json``).

Vendored from #537: the judge prompts + normalizer + parsers + the Anthropic
Batch client live in ``explore_persona_space.experiments.i537_judging``
(``judge_request_for_row`` / ``submit_judge_batch_raw`` / ``parse_verdict_*`` /
``em_rates_from_verdicts``) — vendored verbatim from ``origin/issue-537``.

JUDGE MODEL: ``claude-sonnet-4-5-20250929`` for em/sycophancy/fact — the
CLAUDE.md standing "LLM judge" rule, which SUPERSEDES #537's Haiku pin for
fact/syc (carried as a plan §12 scope caveat). We override the per-row model that
``judge_request_for_row`` returns (Haiku for syc/fact) to Sonnet before
submission; the f_CV^E ratio is judge-internally consistent (the SAME Sonnet
judge scores v0/v⁺/P↑/P↓).

CHANNEL CHOICE: we submit through the vendored #537 ``submit_judge_batch_raw``
(the Anthropic Message Batches client #537 used) rather than
``eval.batch_judge.judge_completions_batch``, because the syc (YES/NO) + fact
(5-way) + em (Betley dual) judges are HETEROGENEOUS — they need raw-text verdicts
+ per-behavior parsers, which ``submit_judge_batch_raw`` returns; the
``judge_completions_batch`` aggregator is Betley-aligned-only. Both route through
the same ``eval.batch_judge`` Message-Batches plumbing (``submit_judge_batch_raw``
calls ``eval.batch_judge._chunk_requests``), so this still satisfies the standing
"route large judge sets through the Anthropic Batch API" rule.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

logger = logging.getLogger("issue697_judge")

SONNET_JUDGE = "claude-sonnet-4-5-20250929"
# The four E generation conditions the sweep wrote one raw_completions file each.
E_CONDITIONS = ("unpatched_base", "unpatched_ft", "p_up", "p_down")
# Judged (generation) behaviors; marker is judge-free (TF marker-logp inline).
JUDGED_BEHAVIORS = ("em", "sycophancy", "fact")


def _custom_id(behavior: str, cell_id: str, condition: str, p_idx: int, q_idx: int, s_idx: int):
    """A ^[a-zA-Z0-9_-]{1,64}$ id (submit_judge_batch_raw asserts the shape)."""
    # cell_id can contain underscores; keep it but drop anything non-conforming.
    safe_cell = "".join(c if (c.isalnum() or c in "_-") else "-" for c in cell_id)
    cid = f"{behavior[:3]}-{safe_cell}-{condition}-p{p_idx}-q{q_idx}-s{s_idx}"
    return cid[:64]


def _load_raw_files(patch_dir: Path, cell_id: str) -> dict[str, dict]:
    """{condition: raw_file_json} for one cell from raw_completions/."""
    raw_dir = patch_dir / "raw_completions"
    out: dict[str, dict] = {}
    for cond in E_CONDITIONS:
        # seed is embedded in the filename; glob by the cell+condition prefix.
        matches = sorted(raw_dir.glob(f"{cell_id}_{cond}_seed*.json"))
        if matches:
            out[cond] = json.loads(matches[0].read_text())
    return out


def _build_requests(behavior: str, cell_id: str, raw_by_cond: dict[str, dict]) -> list[dict]:
    """One judge request per (condition, persona, question, sample), model=Sonnet."""
    from explore_persona_space.experiments.i537_judging import judge_request_for_row

    requests: list[dict] = []
    for cond, raw in raw_by_cond.items():
        for p_idx, row in enumerate(raw.get("rows", [])):
            persona = row["persona"]
            q_idx = row["q_idx"]
            question = row["question"]
            for s_idx, completion in enumerate(row.get("completions", [])):
                req = judge_request_for_row(
                    behavior, question, completion, f"tmp_{p_idx}_{q_idx}_{s_idx}", normalize=True
                )
                # Force the SONNET judge (CLAUDE.md standing rule supersedes #537's
                # Haiku pin for fact/syc); em is already Sonnet.
                req["model"] = SONNET_JUDGE
                req["custom_id"] = _custom_id(behavior, cell_id, cond, p_idx, q_idx, s_idx)
                req["_meta"] = {
                    "condition": cond,
                    "persona": persona,
                    "q_idx": q_idx,
                    "sample": s_idx,
                }
                requests.append(req)
    return requests


def _parse_and_rate(behavior: str, requests: list[dict], raw_verdicts: dict[str, str]) -> dict:
    """Parse verdicts per behavior; return per-condition rate + per-row labels."""
    from explore_persona_space.experiments.i537_judging import (
        em_rates_from_verdicts,
        parse_verdict_binary,
        parse_verdict_em,
        parse_verdict_fact,
    )

    by_cond: dict[str, list[dict]] = {c: [] for c in E_CONDITIONS}
    for req in requests:
        meta = req["_meta"]
        raw = raw_verdicts.get(req["custom_id"], "")
        if behavior == "sycophancy":
            label: object = parse_verdict_binary(raw)
        elif behavior == "fact":
            label = parse_verdict_fact(raw)
        elif behavior == "em":
            label = parse_verdict_em(raw)
        else:
            raise ValueError(behavior)
        by_cond[meta["condition"]].append(
            {
                "persona": meta["persona"],
                "q_idx": meta["q_idx"],
                "sample": meta["sample"],
                "raw": raw,
                "label": label,
            }
        )
    rates: dict[str, dict] = {}
    for cond, rows in by_cond.items():
        if not rows:
            continue
        if behavior == "sycophancy":
            vals = [r["label"] for r in rows if r["label"] is not None]
            rate = (sum(1 for v in vals if v) / len(vals)) if vals else float("nan")
            rates[cond] = {"rate": rate, "n": len(rows), "n_parsed": len(vals)}
        elif behavior == "fact":
            vals = [r["label"] for r in rows if r["label"] is not None]
            taught = (sum(1 for v in vals if v == "TAUGHT") / len(vals)) if vals else float("nan")
            rates[cond] = {"rate_taught": taught, "n": len(rows), "n_parsed": len(vals)}
        elif behavior == "em":
            parsed = [r["label"] for r in rows]
            rates[cond] = em_rates_from_verdicts(parsed)
    return {"by_condition_rows": by_cond, "rates": rates}


def judge_cell(patch_dir: Path, cell_id: str, behavior: str, *, dry_run: bool = False) -> dict:
    """Judge one cell's raw_completions → {cell}_judged.json; return the result dict."""
    raw_by_cond = _load_raw_files(patch_dir, cell_id)
    if not raw_by_cond:
        logger.warning("no raw_completions for cell %s (%s) — skipping", cell_id, behavior)
        return {"cell_id": cell_id, "behavior": behavior, "skipped": "no raw_completions"}
    requests = _build_requests(behavior, cell_id, raw_by_cond)
    logger.info("cell %s: %d judge requests (model=%s)", cell_id, len(requests), SONNET_JUDGE)
    if dry_run:
        return {
            "cell_id": cell_id,
            "behavior": behavior,
            "n_requests": len(requests),
            "dry_run": True,
        }
    import anthropic

    from explore_persona_space.experiments.i537_judging import submit_judge_batch_raw

    client = anthropic.Anthropic()
    items = [
        {
            "custom_id": r["custom_id"],
            "model": r["model"],
            "max_tokens": r["max_tokens"],
            "user_msg": r["user_msg"],
        }
        for r in requests
    ]
    raw_verdicts = submit_judge_batch_raw(client, items)
    parsed = _parse_and_rate(behavior, requests, raw_verdicts)
    result = {
        "cell_id": cell_id,
        "behavior": behavior,
        "judge_model": SONNET_JUDGE,
        "judge_model_note": "Sonnet 4.5 supersedes #537's Haiku pin for fact/syc (CLAUDE.md rule)",
        "rates": parsed["rates"],
        "by_condition_rows": parsed["by_condition_rows"],
    }
    out_path = patch_dir / f"{cell_id}_judged.json"
    out_path.write_text(json.dumps(result, indent=2, default=float))
    logger.info("wrote %s", out_path)
    return result


def judge_all(patch_dir: Path, *, dry_run: bool = False) -> list[dict]:
    """Judge every cell whose raw_completions are present (off-pod CPU)."""
    import torch

    results: list[dict] = []
    for pt in sorted(patch_dir.glob("*.pt")):
        cell = torch.load(pt, weights_only=False)
        behavior = cell["behavior"]
        cell_id = cell["cell_id"]
        if behavior not in JUDGED_BEHAVIORS:
            logger.info(
                "cell %s behavior=%s is judge-free (marker TF logp) — skipping", cell_id, behavior
            )
            continue
        results.append(judge_cell(patch_dir, cell_id, behavior, dry_run=dry_run))
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--patch-dir",
        default=None,
        help="Dir with per-cell *.pt + raw_completions/ (default eval_results/issue_697/patch).",
    )
    parser.add_argument("--cell-id", default=None, help="Judge one cell only (else all).")
    parser.add_argument(
        "--behavior", default=None, help="Behavior of --cell-id (required with it)."
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Build requests, don't call the API."
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s :: %(message)s"
    )
    from dotenv import load_dotenv

    load_dotenv()

    if args.patch_dir:
        patch_dir = Path(args.patch_dir)
    else:
        import subprocess

        repo_root = Path(
            subprocess.check_output(["git", "rev-parse", "--show-toplevel"]).decode().strip()
        )
        patch_dir = repo_root / "eval_results" / "issue_697" / "patch"

    if args.cell_id:
        assert args.behavior, "--behavior required with --cell-id"
        judge_cell(patch_dir, args.cell_id, args.behavior, dry_run=args.dry_run)
    else:
        judge_all(patch_dir, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
