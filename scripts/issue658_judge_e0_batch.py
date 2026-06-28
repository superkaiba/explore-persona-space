#!/usr/bin/env python3
"""Issue #658 J1 (off-pod, Anthropic Batch API): judge E0(C,B) base expression.

Same output schema as ``issue658_judge_e0.py`` (writes ``E0_expression.json``
with the identical per-context per-column records: rate, n_judged, n_positive,
logp_pos_mean, low_dynamic_range, per_probe), but submits all judge prompts via
the Anthropic Message Batches API (the plan-mandated transport for the large
~140k-call set; the threaded driver is too slow). marker = no judge (4-float
slot read); format_style = deterministic structural classifier (no judge).

Reuses the assembly logic from issue658_judge_e0 verbatim — only the transport
(threaded -> Batch API) differs.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from dotenv import load_dotenv  # noqa: E402
from issue404_common import reproducibility_metadata  # noqa: E402
from issue658_common import (  # noqa: E402
    E0_COLUMNS,
    JUDGE_MODEL,
    _verdict_truthy,
    dump_json,
    load_json,
)

from explore_persona_space.eval.batch_judge import (  # noqa: E402
    MAX_JUDGE_REQUESTS_PER_BATCH,
    _chunk_requests,
)
from explore_persona_space.experiments.behavior_testbed_545.judges_545 import (  # noqa: E402
    structural_format_features,
)
from explore_persona_space.llm.anthropic_client import (  # noqa: E402
    AnthropicBatch,
    BatchDeadlineExceeded,
)

load_dotenv(str(PROJECT_ROOT / ".env"))
logger = logging.getLogger("issue658_judge_batch")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def _parse_verdict(text: str) -> dict:
    matches = re.findall(r"\{[^{}]*\}", text or "", flags=re.DOTALL)
    if matches:
        try:
            return json.loads(matches[-1])
        except (ValueError, json.JSONDecodeError):
            pass
    return {"_judge_error": (text or "")[:200]}


def _collect_shard(batch: AnthropicBatch, batch_id: str, out: dict[str, dict]) -> int:
    """Stream one ended sub-batch's results into ``out`` (join on custom_id).

    Returns the count of cleanly-parsed verdicts; refusals and non-succeeded
    results are surfaced as ``_judge_refused`` / ``_judge_error`` dicts (the keys
    the downstream scorer skips on). Reads results through the shared
    :class:`AnthropicBatch` client (``.results`` returns the materialized list).
    """
    n_ok = 0
    for result in batch.results(batch_id):
        cid = result.custom_id
        if result.result.type == "succeeded":
            msg = result.result.message
            if msg.stop_reason == "refusal":
                out[cid] = {"_judge_refused": "stop_reason=refusal"}
                continue
            txt = "\n".join(
                t for t in (getattr(b2, "text", None) for b2 in msg.content) if isinstance(t, str)
            )
            out[cid] = _parse_verdict(txt)
            if "_judge_error" not in out[cid]:
                n_ok += 1
        else:
            out[cid] = {"_judge_error": f"batch_result_type={result.result.type}"}
    return n_ok


def submit_and_collect(
    requests: list[dict], model: str, checkpoint_path: Path | None = None
) -> dict[str, dict]:
    """requests: [{custom_id, prompt}]; returns {custom_id: verdict_dict}.

    Submits via the shared #663-hardened :class:`AnthropicBatch` client
    (``llm/anthropic_client.py``) — the SAME transport the ``run_experiment_389``
    callers route through — instead of a hand-rolled ``messages.batches.create``
    + ``while True`` poller. ``_chunk_requests`` shards the set into
    <=``MAX_JUDGE_REQUESTS_PER_BATCH`` (2_000) sub-batches — NOT the 8k general
    cap: an 8k judge shard STARVES (it sat at succeeded:0 for 9h in the #658 G1
    wedge), while a ~500-request judge shard clears in ~5 min, so the small cap
    is what actually buys incremental progress here. Each shard is polled with
    the client's BOUNDED ``AnthropicBatch.poll`` (exits on the batch's own
    ``expires_at`` + grace, or now+25h if expires_at is ever absent, and raises
    ``BatchDeadlineExceeded`` — never the unbounded ``while True`` that wedged
    the original poll, #658/#661, 2026-06-24). A shard still not ``ended`` at its deadline is
    cancelled and its items surfaced as judge errors; and ANY other per-shard
    failure (network / API error surviving the SDK retries) is caught, its items
    marked as errors, and the run continues — so one stuck or failing shard can
    never wedge or abort the whole run.

    ``checkpoint_path`` (default ``<out>.partial.json`` from ``main``) gives
    cross-process RESUMABILITY: completed verdicts are flushed atomically after
    every shard, and a restart loads them and skips the already-judged
    custom_ids — so a mid-run VM death never forces a full re-spend. The
    custom_ids are positional (``r{nid}``) and reproducible only for the SAME
    input data; a checkpoint from a different ``--e0-dir`` must be removed first.
    """
    # Shared #663 client: create + bounded poll + results + cancel. Its
    # underlying anthropic.Anthropic uses the SDK-default max_retries (2 on the
    # pinned SDK) — down from this script's pre-migration 8; deliberate, the cost
    # of consolidating onto the shared client. Per-shard failures surface as
    # `shard_incomplete` and recover on the next checkpoint/resume run.
    batch_client = AnthropicBatch()

    # Resume: load any verdicts judged in a prior (interrupted) run so a mid-run
    # process death never re-spends the completed shards.
    out: dict[str, dict] = {}
    if checkpoint_path is not None and checkpoint_path.exists():
        try:
            out = dict(json.loads(checkpoint_path.read_text()))
            logger.info("resume: loaded %d cached verdicts from %s", len(out), checkpoint_path)
        except (ValueError, OSError) as e:
            logger.warning("could not read checkpoint %s (%s) — starting fresh", checkpoint_path, e)
            out = {}

    def _flush() -> None:
        if checkpoint_path is None:
            return
        tmp = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
        tmp.write_text(json.dumps(out))
        tmp.replace(checkpoint_path)  # atomic on POSIX

    # Plain-dict request shape that _chunk_requests + batches.create expect; the
    # compact custom_id side-maps back to (ctx, col, ci, cj) in the caller.
    pending = [r for r in requests if r["custom_id"] not in out]
    batch_reqs = [
        {
            "custom_id": r["custom_id"],
            "params": {
                "model": model,
                "max_tokens": 300,
                "messages": [{"role": "user", "content": r["prompt"]}],
            },
        }
        for r in pending
    ]
    # Small judge shards (<=2_000), NOT the 8k general cap — an 8k judge batch
    # starves at succeeded:0 (the #658 wedge); ~500 clears in ~5 min.
    chunks = _chunk_requests(batch_reqs, max_count=MAX_JUDGE_REQUESTS_PER_BATCH)
    poll_interval = 30.0
    logger.info(
        "%d requests total; %d already done; %d to submit -> %d sub-batch(es) (<=%d each)",
        len(requests),
        len(out),
        len(batch_reqs),
        len(chunks),
        MAX_JUDGE_REQUESTS_PER_BATCH,
    )
    for ci, chunk in enumerate(chunks):
        chunk_cids = [r["custom_id"] for r in chunk]
        n_ok = 0
        try:
            batch = batch_client.create(requests=chunk)
            batch_id = batch.id
            logger.info(
                "submitted sub-batch %d/%d id=%s (%d reqs)",
                ci + 1,
                len(chunks),
                batch_id,
                len(chunk),
            )
            ended = False
            try:
                # Shared #663 bounded poll: exits on the batch's own
                # expires_at + grace; raises BatchDeadlineExceeded if it never
                # ends. AnthropicBatch.poll is async — drive it from this sync
                # script via asyncio.run (no enclosing event loop here).
                asyncio.run(batch_client.poll(batch_id, interval_s=poll_interval))
                ended = True
            except BatchDeadlineExceeded as exc:
                logger.error(
                    "sub-batch %s exceeded deadline: %s — cancelling + marking items as errors",
                    batch_id,
                    exc,
                )
                try:
                    batch_client.cancel(batch_id)
                except Exception as ce:
                    logger.warning("cancel of stuck batch %s failed: %s", batch_id, ce)
            if ended:
                n_ok = _collect_shard(batch_client, batch_id, out)
        except Exception as exc:
            logger.error(
                "sub-batch %d/%d failed (%s) — marking its items as errors, continuing",
                ci + 1,
                len(chunks),
                exc,
            )

        # Every chunk cid accounted for (a deadline-cancelled / failed shard, or
        # a cid the API omitted) -> judge error; never clobbers a real verdict.
        for cid in chunk_cids:
            out.setdefault(cid, {"_judge_error": "shard_incomplete"})
        _flush()  # per-shard durability: a crash resumes from here, no re-spend
        logger.info(
            "sub-batch %d/%d done: %d/%d parsed-ok (running total %d)",
            ci + 1,
            len(chunks),
            n_ok,
            len(chunk),
            len(out),
        )
    return out


def judge_column_from_verdicts(col_id, gen, verdicts: dict[str, dict], cid_map: dict) -> dict:
    col = E0_COLUMNS[col_id]
    per_probe_acc: dict[str, list[tuple[bool, float]]] = {}
    n_judged = n_positive = 0
    pos_logps: list[float] = []
    n_total = 0
    for ci, cell in enumerate(gen["cells"]):
        for cj, comp in enumerate(cell["completions"]):
            n_total += 1
            cid = cid_map[(gen["context_id"], col_id, ci, cj)]
            v = verdicts.get(cid, {"_judge_error": "missing"})
            if "_judge_error" in v or "_judge_refused" in v:
                continue
            n_judged += 1
            pos = _verdict_truthy(v, col.e0_verdict_key, col_id)
            per_probe_acc.setdefault(cell["probe"], []).append((pos, comp["logp_norm"]))
            if pos:
                n_positive += 1
                pos_logps.append(comp["logp_norm"])
    rate = (n_positive / n_judged) if n_judged else None
    per_probe = [
        {
            "probe": p,
            "e0": (sum(1 for pos, _ in rows if pos) / len(rows)) if rows else None,
            "n_judged": len(rows),
        }
        for p, rows in per_probe_acc.items()
    ]
    low_dyn = (not pos_logps) or (rate in (0.0, 1.0))
    return {
        "column_id": col_id,
        "dv": "judged_rate",
        "rate": rate,
        "n_judged": n_judged,
        "n_positive": n_positive,
        "n_total": n_total,
        "logp_pos_mean": (sum(pos_logps) / len(pos_logps)) if pos_logps else None,
        "low_dynamic_range": low_dyn,
        "per_probe": per_probe,
    }


def score_format(gen) -> dict:
    flags, logps, per_probe = [], [], []
    for cell in gen["cells"]:
        cf = []
        for comp in cell["completions"]:
            f = structural_format_features(comp["text"])["is_list_formatted"]
            flags.append(f)
            cf.append(f)
            logps.append(comp["logp_norm"])
        per_probe.append(
            {
                "probe": cell["probe"],
                "e0": (sum(1 for f in cf if f) / len(cf)) if cf else None,
                "n_judged": len(cf),
            }
        )
    rate = (sum(1 for f in flags if f) / len(flags)) if flags else None
    pos_logps = [lp for f, lp in zip(flags, logps, strict=True) if f]
    return {
        "column_id": "format_style",
        "dv": "structural",
        "rate": rate,
        "n_total": len(flags),
        "logp_pos_mean": (sum(pos_logps) / len(pos_logps)) if pos_logps else None,
        "low_dynamic_range": (not pos_logps) or (rate in (0.0, 1.0)),
        "per_probe": per_probe,
    }


def score_marker(gen) -> dict:
    logps = [r["logp"] for r in gen["marker_slot"]]
    z = [r["z_marker"] - r["z_eos"] for r in gen["marker_slot"]]
    emits = [r.get("argmax_id") == 83399 for r in gen["marker_slot"]]
    per_probe = [{"probe": r["probe"], "e0": r["logp"], "n_judged": 1} for r in gen["marker_slot"]]
    return {
        "column_id": "marker",
        "dv": "marker_slot_stats",
        "logp_mean": (sum(logps) / len(logps)) if logps else None,
        "eos_margin_mean": (sum(z) / len(z)) if z else None,
        "emission_rate": (sum(1 for e in emits if e) / len(emits)) if emits else None,
        "n_total": len(logps),
        "per_probe": per_probe,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--e0-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", default=JUDGE_MODEL)
    args = ap.parse_args()

    gen_files = sorted(args.e0_dir.glob("*__*.json"))
    if not gen_files:
        raise RuntimeError(f"no gen files under {args.e0_dir}")

    # Build all judge requests up front (marker/format need no judge). The Batch
    # API custom_id must match ^[a-zA-Z0-9_-]{1,64}$, so use a compact integer id
    # and keep a side-map back to (ctx, col, ci, cj).
    requests: list[dict] = []
    gens: dict[tuple[str, str], dict] = {}
    cid_map: dict[tuple, str] = {}
    nid = 0
    for gf in gen_files:
        gen = load_json(gf)
        ctx, col_id = gen["context_id"], gen["column_id"]
        gens[(ctx, col_id)] = gen
        if col_id in ("marker", "format_style"):
            continue
        col = E0_COLUMNS[col_id]
        for ci, cell in enumerate(gen["cells"]):
            for cj, comp in enumerate(cell["completions"]):
                cid = f"r{nid}"
                nid += 1
                cid_map[(ctx, col_id, ci, cj)] = cid
                requests.append(
                    {
                        "custom_id": cid,
                        "prompt": col.judge_prompt.format(
                            question=cell["probe"], completion=comp["text"]
                        ),
                    }
                )
    logger.info("total judge requests: %d across %d gen files", len(requests), len(gen_files))
    # Per-shard checkpoint next to the output -> cross-process resume, no re-spend.
    checkpoint_path = args.out.with_name(args.out.stem + ".partial.json")
    verdicts = (
        submit_and_collect(requests, args.model, checkpoint_path=checkpoint_path)
        if requests
        else {}
    )

    results: dict = {}
    for (ctx, col_id), gen in gens.items():
        results.setdefault(ctx, {})
        if col_id == "marker":
            results[ctx][col_id] = score_marker(gen)
        elif col_id == "format_style":
            results[ctx][col_id] = score_format(gen)
        else:
            results[ctx][col_id] = judge_column_from_verdicts(col_id, gen, verdicts, cid_map)

    payload = {
        "judge_model": args.model,
        "e0": results,
        "columns": list(E0_COLUMNS.keys()),
        "transport": "anthropic_batch_api",
        "dual_dv": {
            "primary": "judged_rate / marker logp",
            "secondary": "length-normalized base-model log P of judged-positive completions",
            "empty_set_guard": "low_dynamic_range=true when no judged-positive completions exist",
        },
        "metadata": reproducibility_metadata({"script": "issue658_judge_e0_batch"}),
    }
    dump_json(payload, args.out)
    n_low = sum(1 for c in results.values() for v in c.values() if v.get("low_dynamic_range"))
    n_err = sum(1 for v in verdicts.values() if "_judge_error" in v or "_judge_refused" in v)
    logger.info(
        "wrote %s (%d ctx; %d low-dyn cells; %d judge errors/refusals of %d)",
        args.out,
        len(results),
        n_low,
        n_err,
        len(verdicts),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
