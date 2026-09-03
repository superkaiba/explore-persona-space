"""Issue #2658 — pilot judge Batch-API spend artifact producer (plan v5 A4).

Enumerates the pilot judge batch ids from the on-disk dispatch results file
NAMES (``raw_completions/judge/**/.dispatch/*/results_msgbatch_*.json`` —
bodies are NEVER opened: judge outputs are referenced by path/count only),
retrieves each batch's per-result ``usage`` via the Anthropic Batches API
(read-only; no new inference requests), sums the four token categories over
succeeded results, and prices them at the PUBLISHED Sonnet 4.5 Message
Batches rates into ``eval_results/issue_2658/power_inputs/judge_spend.json``
(``basis: "priced from measured tokens, not billed"``).

# API_DISPATCH_ROUTING_EXEMPT: read-only usage retrieval of already-completed
# batches, no new requests

The retrieval tries every org key the multi-org dispatcher configures
(``ANTHROPIC_BATCH_KEY`` first — the batch org — then ``ANTHROPIC_API_KEY``):
a batch id 404s on the org that did not create it, so per-id fallback across
the present keys is required; an id no key can resolve raises.

Launch:
    OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \
    NUMEXPR_NUM_THREADS=8 MALLOC_ARENA_MAX=2 \
    uv run python scripts/issue2658_judge_spend.py
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()  # org keys must bind BEFORE the anthropic client import

import anthropic  # noqa: E402

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import issue2658_frames as F  # noqa: E402
from explore_persona_space.atomic_io import write_json_atomic  # noqa: E402
from explore_persona_space.orchestrate.provenance import (  # noqa: E402
    as_metadata_dict,
    git_provenance,
)

_BATCH_FILE_RE = re.compile(r"results_(msgbatch_[A-Za-z0-9]+)\.json$")

# Published Claude Sonnet 4.5 Message Batches rates, USD per million tokens.
# Source (fetched 2026-09-03): the Anthropic pricing page — base $3 in / $15
# out; Batch API = 50% discount on input and output ($1.50 / $7.50); prompt
# caching multipliers STACK with the batch discount (cache read 0.1x batch
# input = $0.15; 5-minute cache write 1.25x batch input = $1.875).
PRICE_SOURCE_URL = "https://platform.claude.com/docs/en/about-claude/pricing"
PRICE_RETRIEVED_AT = "2026-09-03"
RATES_PER_MTOK = {
    "input_per_mtok": 1.50,
    "output_per_mtok": 7.50,
    "cache_read_per_mtok": 0.15,
    "cache_write_5m_per_mtok": 1.875,
}

USAGE_FIELDS = (
    "input_tokens",
    "output_tokens",
    "cache_read_input_tokens",
    "cache_creation_input_tokens",
)


class SpendError(RuntimeError):
    """A batch cannot be enumerated/retrieved — never silently skipped."""


def enumerate_batch_ids(judge_root: Path) -> dict[str, str]:
    """batch id -> subtree ('pilot' / 'canary' / ...) from results file NAMES."""
    ids: dict[str, str] = {}
    for p in sorted(judge_root.rglob("results_msgbatch_*.json")):
        m = _BATCH_FILE_RE.search(p.name)
        if not m:
            raise SpendError(f"unparseable batch results filename: {p}")
        bid = m.group(1)
        subtree = p.relative_to(judge_root).parts[0]
        if bid in ids:
            raise SpendError(f"duplicate batch id {bid} under {judge_root}")
        ids[bid] = subtree
    if not ids:
        raise SpendError(f"no results_msgbatch_*.json files under {judge_root}")
    return ids


def _org_clients() -> dict[str, anthropic.Anthropic]:
    clients: dict[str, anthropic.Anthropic] = {}
    for label, env in (("batch", "ANTHROPIC_BATCH_KEY"), ("high_prio", "ANTHROPIC_API_KEY")):
        key = os.environ.get(env)
        if key:
            clients[label] = anthropic.Anthropic(api_key=key)
    if not clients:
        raise SpendError("no ANTHROPIC_BATCH_KEY / ANTHROPIC_API_KEY in the environment")
    return clients


def retrieve_batch_usage(
    clients: dict[str, anthropic.Anthropic], batch_id: str
) -> tuple[str, dict[str, int]]:
    """(org label, per-batch tally) — sums usage over succeeded results."""
    last_err: Exception | None = None
    for label, client in clients.items():
        try:
            tally = {f: 0 for f in USAGE_FIELDS}
            tally["n_succeeded"] = tally["n_errored"] = tally["n_expired"] = 0
            tally["n_other"] = 0
            for entry in client.messages.batches.results(batch_id):
                rtype = entry.result.type
                if rtype == "succeeded":
                    tally["n_succeeded"] += 1
                    usage = entry.result.message.usage
                    for f in USAGE_FIELDS:
                        tally[f] += int(getattr(usage, f, None) or 0)
                elif rtype == "errored":
                    tally["n_errored"] += 1
                elif rtype == "expired":
                    tally["n_expired"] += 1
                else:
                    tally["n_other"] += 1
            return label, tally
        except anthropic.NotFoundError as err:  # wrong org for this id — try the next
            last_err = err
            continue
    raise SpendError(f"batch {batch_id} not retrievable under any configured org key") from last_err


def build_spend(out_root: Path) -> dict:
    judge_root = Path(out_root) / "raw_completions" / "judge"
    ids = enumerate_batch_ids(judge_root)
    clients = _org_clients()
    # Per-unit checkpoint (code-style "checkpoint per phase", T2: 97 units):
    # one JSONL row per batch, atomic append, resume-skip keyed on batch id
    # (a batch's completed usage is immutable — the id IS the regime key).
    ckpt = Path(out_root) / "power_inputs" / "judge_spend_batches.jsonl"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    done: dict[str, dict] = {}
    if ckpt.exists():
        for line in ckpt.read_text().splitlines():
            if line.strip():
                rec = json.loads(line)
                done[rec["batch_id"]] = rec
    totals = {f: 0 for f in USAGE_FIELDS}
    n_succeeded = n_errored = n_expired = 0
    per_batch: list[dict] = []
    t0 = time.time()
    for i, (bid, subtree) in enumerate(sorted(ids.items()), start=1):
        if bid in done:
            rec = done[bid]
            print(f"[spend] batch {i}/{len(ids)} {bid} resume-skip", flush=True)
        else:
            org, tally = retrieve_batch_usage(clients, bid)
            if tally["n_other"]:
                raise SpendError(f"batch {bid}: {tally['n_other']} results of unexpected type")
            rec = {"batch_id": bid, "subtree": subtree, "org": org, **tally}
            with ckpt.open("a") as fh:
                fh.write(json.dumps(rec, sort_keys=True) + "\n")
                fh.flush()
                os.fsync(fh.fileno())
            print(
                f"[spend] batch {i}/{len(ids)} {bid} n_succeeded={tally['n_succeeded']} "
                f"elapsed={time.time() - t0:.0f}s",
                flush=True,
            )
        for f in USAGE_FIELDS:
            totals[f] += rec[f]
        n_succeeded += rec["n_succeeded"]
        n_errored += rec["n_errored"]
        n_expired += rec["n_expired"]
        per_batch.append(rec)
    if n_succeeded <= 0:
        raise SpendError("zero succeeded results across every batch — nothing to price")
    dollars = (
        totals["input_tokens"] * RATES_PER_MTOK["input_per_mtok"]
        + totals["output_tokens"] * RATES_PER_MTOK["output_per_mtok"]
        + totals["cache_read_input_tokens"] * RATES_PER_MTOK["cache_read_per_mtok"]
        + totals["cache_creation_input_tokens"] * RATES_PER_MTOK["cache_write_5m_per_mtok"]
    ) / 1e6
    subtree_counts: dict[str, int] = {}
    for rec in per_batch:
        subtree_counts[rec["subtree"]] = subtree_counts.get(rec["subtree"], 0) + 1
    return {
        "schema": "i2658-judge-spend-v1",
        "dollars": dollars,
        "basis": "priced from measured tokens, not billed",
        "price_source_url": PRICE_SOURCE_URL,
        "retrieved_at": PRICE_RETRIEVED_AT,
        "rates_per_mtok": RATES_PER_MTOK,
        "totals": totals,
        "n_batches": len(per_batch),
        "n_batches_by_subtree": subtree_counts,
        "n_calls_succeeded": n_succeeded,
        "n_calls_errored": n_errored,
        "n_calls_expired": n_expired,
        "per_call_mean_input_tokens": totals["input_tokens"] / n_succeeded,
        "per_call_mean_output_tokens": totals["output_tokens"] / n_succeeded,
        "per_batch": per_batch,
        "enumeration": {
            "root": str(judge_root),
            "pattern": "**/.dispatch/*/results_msgbatch_*.json",
            "detail": "batch ids from results file NAMES; bodies never opened",
        },
        "metadata": as_metadata_dict(git_provenance(), phase="p3-judge-spend"),
    }


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-root", type=Path, default=F.OUT_DIR)
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="output path (default <out-root>/power_inputs/judge_spend.json)",
    )
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    spend = build_spend(Path(args.out_root))
    out = args.out or (Path(args.out_root) / "power_inputs" / "judge_spend.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(out, spend)
    print(
        f"[spend] wrote {out}: ${spend['dollars']:.2f} over {spend['n_batches']} batches / "
        f"{spend['n_calls_succeeded']} succeeded calls "
        f"(in {spend['totals']['input_tokens']:,} / out {spend['totals']['output_tokens']:,} "
        f"/ cache-read {spend['totals']['cache_read_input_tokens']:,} / cache-write "
        f"{spend['totals']['cache_creation_input_tokens']:,} tokens)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
