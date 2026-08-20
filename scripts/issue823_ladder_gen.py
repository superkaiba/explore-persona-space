"""P-Gen phase driver for task #823 follow-up `inconsistent-origin-persona-ladder`.

Generates the nested persona-assignment answer set (14,996 Claude Sonnet Batch
generations) for the k-ladder k in {1, 2, 4, 8, 16}:

1. Downloads the parent's frozen 5,000-context question set
   (`issue823_own_vs_external/raw_completions/phase1/b2_seed42.json`) +
   `common_valid_idx.json` at the pinned rev, asserting monotone-unique
   `context_id` 0..4999 (file order IS the frozen context order).
2. Builds the registered nested assignment persona(i, k) = i mod k, asserting
   the nesting / balance / distinct-pair-count arithmetic exactly as
   registered (14,996 unique pairs at n=5000).
3. Dispatches one generation per distinct (context, persona) pair through the
   project dispatcher (`llm.api_dispatch.dispatch_calls`, Batch API path,
   checkpointed + resumable by batch id), re-driving transport-class failures
   on fresh checkpoints (the dispatcher checkpoint re-serves persisted
   transport rows on resume, so caller-side re-drive uses a fresh dir).
4. Applies the pre-registered cap-hit re-gen trigger: any persona whose
   `stop_reason == "max_tokens"` fraction exceeds 2 percent gets its over-cap
   rows re-generated at max_tokens 2048 before persist.
5. Persists 16 per-persona record files (every record carrying its Batch
   request custom_id, batch id, and submitted/harvested timestamps -- the
   batch-wave drift audit fields, free now and unrecoverable later) +
   `assignment.json` + `roster.json` + `_gen_complete.json`, and uploads all
   of them to the HF data repo in ONE bulk commit BEFORE any pod exists.

Usage:
  uv run python scripts/issue823_ladder_gen.py --smoke   # 16 contexts, /tmp + _smoke HF prefix
  uv run python scripts/issue823_ladder_gen.py           # full 5,000-context production run

Exit codes: 0 = complete; 3 = transport-class rows remain (after this run's
bounded FRESH re-drive rounds, or inside the pooled cap-hit regen) --
halt-and-report: digest written, no sentinel, no upload. The dispatcher
re-serves persisted transport rows on a resumed checkpoint WITHOUT
re-dispatching, so: a re-run resumes completed rows from the batch
checkpoints and re-submits redrive residue automatically in fresh redrive
dirs numbered past the stale ones; regen residue requires quarantining the
halt-named regen checkpoint dir first (the halt message carries the exact mv
command). Completion additionally requires the upload to land on the
CANONICAL data repo -- an overflow-rerouted upload raises instead of
reporting complete.
"""

from __future__ import annotations

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()  # creds + shared-VM thread caps BEFORE any heavy import (sibling pattern)

import argparse
import asyncio
import datetime as _dt
import hashlib
import json
import logging
import os
import pathlib
import re
import subprocess
import sys
from collections import Counter

from huggingface_hub import hf_hub_download

from explore_persona_space.llm.api_dispatch import (
    RESULT_RATE_LIMITED,
    RESULT_TRANSPORT,
    DispatchItem,
    DispatchResult,
    dispatch_calls,
)
from explore_persona_space.orchestrate.hub import _upload_folder_filtered, retry_transient

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("issue823_ladder_gen")

# ── Registered constants (plan v10 slice, sections 0 / 4.1 / 4.2 / 4.3) ─────
SONNET_MODEL = "claude-sonnet-4-5-20250929"  # == run_823.SONNET_MODEL (parent parity)
GEN_MAX_TOKENS = 1024  # == run_823.SONNET_MAX_TOKENS (parent parity)
REGEN_MAX_TOKENS = 2048  # pre-registered re-gen cap for over-cap rows
GEN_TEMPERATURE = 1.0
CAP_HIT_REGEN_FRACTION = 0.02  # per-persona stop-and-regen trigger
MAX_TRANSPORT_REDRIVES = 2  # bounded caller-side re-drive of transport-class rows
# FIX C: cumulative ceiling on fresh PAID re-drive rounds across ALL process
# runs. No new persisted counter — the numbered redriveN checkpoint dirs on
# disk ARE the durable cumulative counter (_next_redrive_round numbers past
# them). Enforced by _require_redrive_headroom before each fresh dispatch.
MAX_CUMULATIVE_REDRIVE_ROUNDS = 6

K_ARMS: tuple[int, ...] = (1, 2, 4, 8, 16)
N_PERSONAS = 16
N_CONTEXTS_FULL = 5000
REGISTERED_TOTAL_PAIRS = 14996  # 313*20 + 312*28 (residues 0-7 x313, 8-15 x312)

DATA_REPO = "superkaiba1/explore-persona-space-data"
PARENT_PREFIX = "issue823_own_vs_external"
PARENT_REV = "8039d15f30deb845765cbb24d9cdb8708a5e7b0f"  # pinned; verified present
HF_PREFIX = "issue823_inconsistent_origin_ladder"

# Fixed 16-persona roster, index order fixed at plan time (persona 0 is the
# k=1 persona by construction).
PERSONAS: list[tuple[str, str]] = [
    (
        "Dr. Maya Chen",
        "a veteran emergency-room physician: pragmatic, triage-minded, plain-spoken about risk",
    ),
    ("Frank Delgado", "a retired homicide detective: skeptical, procedural, drily funny"),
    (
        "Amélie Fournier",
        "a French pastry chef: warm, exacting about craft, sprinkles in French phrases",
    ),
    (
        "Jax Torres",
        "a touring stand-up comedian: irreverent, riffs first, lands the point through jokes",
    ),
    (
        "Margaret Whitfield",
        "a small-town librarian: gentle, precise, loves citations and cross-references",
    ),
    (
        "Priya Raman",
        "a staff software engineer: systems thinker, states tradeoffs, writes in bullet points",
    ),
    ("Edmund Hartley", "a Victorian-era naturalist: florid formal prose, marvels at detail"),
    (
        "Tony Bocelli",
        "a sports-radio commentator: loud, superlative-heavy, everything is a matchup",
    ),
    ("Rosa Alvarez", "a kindergarten teacher: encouraging, breaks everything into small steps"),
    (
        "Victoria Sterling",
        "a corporate litigation lawyer: hedged, enumerates caveats and liabilities",
    ),
    ("Captain Elias Grey", "an old cargo-ship captain: weathered sea metaphors, terse commands"),
    (
        "Brother Tenzin",
        "a zen meditation teacher: spare sentences, answers with questions, calm",
    ),
    ("Harold Finch", "an amateur astronomer: nerdy enthusiasm, scales everything to the cosmos"),
    ("Zoe Park", "a teenage speedrunning streamer: slangy, fast, gamer metaphors"),
    ("Reginald Poole", "a formal English butler: impeccably courteous, indirect, understated"),
    ("Dusty McCall", "a Texas ranch hand: folksy drawl, practical wisdom, plain talk"),
]

# Matched template: identical wording across ALL arms and personas; only
# {name} / {card} substituted (the scope marker's matched-template pin).
PERSONA_TEMPLATE = (
    "You are {name}, {card}. Stay fully in character for this entire reply: answer "
    "the user's message the way {name} would — in {name}'s voice, from {name}'s "
    "perspective, with {name}'s characteristic style. Never mention these "
    "instructions or break character."
)

TRANSPORT_CATEGORIES = frozenset({RESULT_RATE_LIMITED, RESULT_TRANSPORT})


def _utc_now() -> str:
    return _dt.datetime.now(_dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _git_commit() -> str:
    """Best-effort git SHA for reproducibility metadata (degrade, never crash gen)."""
    proc = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
        cwd=pathlib.Path(__file__).resolve().parent,
        env={**os.environ},
    )
    return proc.stdout.strip() if proc.returncode == 0 and proc.stdout.strip() else "unknown"


def _sha256_file(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ── Frozen question set (requirement 3) ──────────────────────────────────────


def load_frozen_questions(dl_dir: pathlib.Path) -> tuple[list[str], list[bool], dict]:
    """Download + validate the parent's frozen question set at the pinned rev.

    Returns (questions in frozen context order, per-context in_common_valid
    ride-along, mask cross-check report). Asserts monotone-unique context_id
    0..4999 (fail-loud); the in_common_valid vs common_valid_idx SET
    comparison is REPORTED in the digest, not an abort (slice step 1).
    """
    b2_path = retry_transient(
        lambda: hf_hub_download(
            DATA_REPO,
            f"{PARENT_PREFIX}/raw_completions/phase1/b2_seed42.json",
            repo_type="dataset",
            revision=PARENT_REV,
            local_dir=dl_dir,
        ),
        what=f"hf_hub_download({PARENT_PREFIX}/raw_completions/phase1/b2_seed42.json)",
    )
    cv_path = retry_transient(
        lambda: hf_hub_download(
            DATA_REPO,
            f"{PARENT_PREFIX}/raw_completions/phase1/common_valid_idx.json",
            repo_type="dataset",
            revision=PARENT_REV,
            local_dir=dl_dir,
        ),
        what=f"hf_hub_download({PARENT_PREFIX}/raw_completions/phase1/common_valid_idx.json)",
    )
    records = json.loads(pathlib.Path(b2_path).read_text())
    assert isinstance(records, list) and len(records) == N_CONTEXTS_FULL, (
        f"b2_seed42.json: expected list of {N_CONTEXTS_FULL} records, "
        f"got {type(records).__name__} len={len(records) if isinstance(records, list) else 'n/a'}"
    )
    # Observed schema at the pin: {context_id, question, answer_text, arm, seed,
    # filled, in_common_valid} — NOT {idx, question, answer}.
    row0_keys = set(records[0].keys())
    required = {"context_id", "question", "filled", "in_common_valid"}
    assert required <= row0_keys, f"b2_seed42.json row 0 missing keys: {required - row0_keys}"

    # REQUIREMENT 3: monotone-unique context_id 0..4999 — file order IS the
    # frozen context order. Runs at FULL grain regardless of smoke slicing.
    context_ids = [r["context_id"] for r in records]
    assert context_ids == list(range(N_CONTEXTS_FULL)), (
        "b2_seed42.json context_id sequence is not monotone-unique 0..4999 — "
        "frozen context order violated; refusing to proceed"
    )
    questions = [r["question"] for r in records]
    assert all(isinstance(q, str) and q for q in questions), "empty/non-str question rows"

    cv = json.loads(pathlib.Path(cv_path).read_text())
    assert cv["n_common"] == len(cv["common_valid_idx"]), (
        f"common_valid_idx.json internally incoherent: n_common={cv['n_common']} "
        f"vs len(common_valid_idx)={len(cv['common_valid_idx'])}"
    )
    cv_ids = set(cv["common_valid_idx"])
    rec_true = {r["context_id"] for r in records if r["in_common_valid"]}
    mask_crosscheck = {
        "n_common_field": cv["n_common"],
        "n_common_valid_idx": len(cv_ids),
        "n_in_common_valid_true": len(rec_true),
        "counts_equal": len(rec_true) == len(cv_ids),
        "set_equal": rec_true == cv_ids,
        "n_only_in_records": len(rec_true - cv_ids),
        "n_only_in_common_valid_idx": len(cv_ids - rec_true),
        "n_filled_true": sum(1 for r in records if r["filled"]),
        "parent_rev": PARENT_REV,
    }
    if not mask_crosscheck["set_equal"]:
        logger.warning(
            "MASK CROSS-CHECK MISMATCH (reported, not fatal): in_common_valid True-set "
            "!= common_valid_idx set (only_in_records=%d, only_in_common_valid_idx=%d)",
            mask_crosscheck["n_only_in_records"],
            mask_crosscheck["n_only_in_common_valid_idx"],
        )
    in_common = [bool(r["in_common_valid"]) for r in records]
    return questions, in_common, mask_crosscheck


# ── Nested assignment (requirement 2) ────────────────────────────────────────


def registered_pair_total(n_contexts: int) -> int:
    """Mechanically computed distinct (context, persona) pair count for n contexts."""
    return sum(len({0, i % 2, i % 4, i % 8, i % 16}) for i in range(n_contexts))


def build_assignment(n_contexts: int) -> dict[int, list[int]]:
    """The registered nested assignment: persona(i, k) = i mod k, per arm k."""
    return {k: [i % k for i in range(n_contexts)] for k in K_ARMS}


def verify_assignment(assignment: dict[int, list[int]], n_contexts: int) -> set[tuple[int, int]]:
    """Assert the assignment is EXACTLY as registered (requirement 2); return pair store.

    Assert order is deliberate — the independent structural checks (balance,
    nesting, pair counts) run BEFORE the exact-rule recompute, so each guards
    against a future edit to the builder on its own.
    """
    assert set(assignment) == set(K_ARMS), f"arms {sorted(assignment)} != registered {K_ARMS}"
    for k in K_ARMS:
        pers = assignment[k]
        assert len(pers) == n_contexts, f"arm {k}: {len(pers)} rows != {n_contexts} contexts"
        counts = Counter(pers)
        assert set(counts) <= set(range(k)), f"arm {k}: persona index outside range({k})"
        for p in range(k):
            expected = n_contexts // k + (1 if p < n_contexts % k else 0)
            assert counts.get(p, 0) == expected, (
                f"arm {k} persona {p}: {counts.get(p, 0)} contexts != registered {expected}"
            )
    # Nesting: arm-k persona-0 rows are a (strict, when resolvable) subset of
    # arm-(k/2) persona-0 rows. Strictness holds iff n_contexts > k/2 (the
    # context k/2 is then present and belongs to arm k/2's persona-0 only).
    for k in (2, 4, 8, 16):
        sub = {i for i, p in enumerate(assignment[k]) if p == 0}
        sup = {i for i, p in enumerate(assignment[k // 2]) if p == 0}
        assert sub <= sup, f"arm {k} persona-0 rows not a subset of arm {k // 2}'s"
        if n_contexts > k // 2:
            assert sub < sup, f"arm {k} persona-0 rows not a STRICT subset of arm {k // 2}'s"
    pairs = {(i, assignment[k][i]) for k in K_ARMS for i in range(n_contexts)}
    expected_pairs = registered_pair_total(n_contexts)
    assert len(pairs) == expected_pairs, (
        f"distinct (context, persona) pairs {len(pairs)} != registered {expected_pairs}"
    )
    if n_contexts == N_CONTEXTS_FULL:
        assert len(pairs) == REGISTERED_TOTAL_PAIRS == 313 * 20 + 312 * 28, (
            f"full-run pair count {len(pairs)} != registered {REGISTERED_TOTAL_PAIRS}"
        )
        res_counts = Counter(i % N_PERSONAS for i in range(n_contexts))
        assert all(res_counts[r] == 313 for r in range(8)) and all(
            res_counts[r] == 312 for r in range(8, 16)
        ), f"per-residue context counts deviate from registered 313/312: {dict(res_counts)}"
    # Exact-rule recompute LAST (the belt over the suspenders above).
    for k in K_ARMS:
        assert assignment[k] == [i % k for i in range(n_contexts)], (
            f"arm {k}: assignment deviates from registered persona(i, k) = i mod k"
        )
    return pairs


# ── Dispatch items ───────────────────────────────────────────────────────────


def persona_system(p: int) -> str:
    name, card = PERSONAS[p]
    return PERSONA_TEMPLATE.format(name=name, card=card)


def make_item_id(p: int, i: int) -> str:
    return f"p{p:02d}_c{i:05d}"


def build_items(questions: list[str], pairs: set[tuple[int, int]]) -> list[DispatchItem]:
    """One DispatchItem per distinct (context, persona) pair.

    Ordered by (context_id, persona) so batch sub-chunks span mixed personas —
    deliberately de-confounding batch wave from roster position (the M3
    batch-wave drift check's design side).
    """
    items = []
    for i, p in sorted(pairs):
        items.append(
            DispatchItem(
                item_id=make_item_id(p, i),
                payload={
                    "messages": [{"role": "user", "content": questions[i]}],
                    "system": persona_system(p),
                },
            )
        )
    return items


def make_build_request(max_tokens: int):
    """Messages-API params builder (system lifted to the top-level param)."""

    def _build_request(item: DispatchItem) -> dict:
        return {
            "model": SONNET_MODEL,
            "max_tokens": max_tokens,
            "temperature": GEN_TEMPERATURE,
            "messages": item.payload["messages"],
            "system": item.payload["system"],
        }

    return _build_request


def _parse_response(text: str) -> str:
    """Identity parse — the generation text IS the result."""
    return text


async def _dispatch(
    items: list[DispatchItem],
    checkpoint_dir: pathlib.Path,
    max_tokens: int,
    poll_interval: float,
) -> dict[str, DispatchResult]:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return await dispatch_calls(
        items,
        model=SONNET_MODEL,
        build_request=make_build_request(max_tokens),
        parse_response=_parse_response,
        max_attempts=5,
        checkpoint_dir=checkpoint_dir,
        # Batch-routed at EVERY scale (smoke included) so smoke and production
        # exercise the same dispatch path; 14,996 production calls would
        # auto-route batch anyway.
        force_path="batch",
        poll_interval=poll_interval,
    )


def load_batch_meta(checkpoint_dir: pathlib.Path) -> dict[str, dict]:
    """Join item_id -> Batch provenance from the dispatcher's state.json (requirement 1).

    ``harvested_at`` is PER SUB-BATCH, read from the dispatcher's own
    ``results_<batch_id>.json`` mtime: the dispatcher writes that file exactly
    once, at harvest, via atomic replace (``_collect_sub_batch`` →
    ``_atomic_write_json``), and a resume never rewrites it (collected
    sub-batches are skipped) — so the mtime IS the sub-batch harvest time.
    A single record-build-time stamp would defeat the batch-wave drift audit
    this field exists for (every record across every wave identical).
    """
    state = json.loads((checkpoint_dir / "state.json").read_text())
    cid_to_item = state["cid_to_item"]
    meta: dict[str, dict] = {}
    for sb in state["sub_batches"]:
        assert sb.get("batch_id"), (
            f"sub-batch {sb['index']} in {checkpoint_dir} has no batch_id — "
            "batch path not exercised; batch provenance unrecoverable"
        )
        results_path = checkpoint_dir / f"results_{sb['batch_id']}.json"
        assert results_path.is_file(), (
            f"sub-batch {sb['index']} in {checkpoint_dir}: {results_path.name} missing — "
            "batch not harvested; per-batch harvest time unrecoverable"
        )
        harvested_at = _dt.datetime.fromtimestamp(
            results_path.stat().st_mtime, tz=_dt.UTC
        ).strftime("%Y-%m-%dT%H:%M:%SZ")
        for cid in sb["custom_ids"]:
            meta[cid_to_item[cid]] = {
                "batch_id": sb["batch_id"],
                "batch_request_custom_id": cid,
                "batch_org": sb.get("org"),
                "batch_submitted_at": sb.get("submitted_at"),
                "harvested_at": harvested_at,
            }
    return meta


def transport_class_ids(results: dict[str, DispatchResult]) -> list[str]:
    """Item ids whose outcome is caller-re-drivable (rate-limit/transport exhaustion)."""
    return sorted(
        item_id for item_id, res in results.items() if res.category in TRANSPORT_CATEGORIES
    )


def _next_redrive_round(root: pathlib.Path) -> int:
    """First redrive round number strictly PAST any stale redrive dirs on disk.

    The dispatcher re-serves persisted transport-class rows on a resumed
    checkpoint WITHOUT re-dispatching, so an exit-3 re-run that reused
    ``redrive1``/``redrive2`` would replay stale state, submit nothing, and
    deterministically exit 3 again, forever. Numbering this run's fresh rounds
    at max(existing) + 1 guarantees each process run actually RE-SUBMITS the
    transport residue.
    """
    existing = [
        int(m.group(1))
        for d in root.glob("redrive*")
        if d.is_dir() and (m := re.fullmatch(r"redrive(\d+)", d.name)) is not None
    ]
    return max(existing, default=0) + 1


def _stale_redrive_dirs(root: pathlib.Path) -> list[pathlib.Path]:
    """Numeric ``redriveN`` checkpoint dirs from PRIOR runs, ascending by round.

    Ascending (chronological) order is load-bearing: a row present in
    ``redrive3`` was still transport-class after ``redrive2``'s merge in the
    run that created it, so replaying ``results.update`` in dir order
    reproduces each prior run's own merge sequence. Quarantined dirs
    (``redriveN.stale``), non-numeric names, and plain files are excluded —
    the same match rule as :func:`_next_redrive_round`.
    """
    numbered = [
        (int(m.group(1)), d)
        for d in root.glob("redrive*")
        if d.is_dir() and (m := re.fullmatch(r"redrive(\d+)", d.name)) is not None
    ]
    return [d for _, d in sorted(numbered)]


def _merge_stale_redrives(
    root: pathlib.Path,
    items_by_id: dict[str, DispatchItem],
    results: dict[str, DispatchResult],
    batch_meta: dict[str, dict],
    poll_interval: float,
) -> None:
    """Resume prior runs' redrive checkpoints and merge their outcomes (FIX B).

    Main-checkpoint results files are immutable once collected, so a re-run
    after an exit-3 halt recomputes the ORIGINAL main-dispatch residue: rows
    that already succeeded — and were billed — in a prior run's ``redriveN``
    checkpoints would be re-submitted (re-bought) on every re-run, and a run
    killed mid-retry leaves an orphaned batch completing server-side that a
    blind re-submit double-bills on success. Resuming each stale dir through
    the existing ``_dispatch`` re-serves collected sub-batches WITHOUT
    re-dispatching (network-free) and polls/harvests any crash-orphaned
    batch; the merged outcomes shrink the pending set to the
    genuinely-remaining rows. Raises on a checkpoint whose item ids are not
    in this run's registered item set (foreign assignment/context set).
    """
    for rd_dir in _stale_redrive_dirs(root):
        state_path = rd_dir / "state.json"
        if not state_path.is_file():
            # Dir created but the dispatcher died before writing state.json:
            # nothing was submitted there, so there is nothing to resume.
            logger.warning(
                "Stale %s has no dispatcher state.json — the prior run died before "
                "submitting anything there; skipping resume for this dir.",
                rd_dir,
            )
            continue
        stale_ids = sorted(set(json.loads(state_path.read_text())["cid_to_item"].values()))
        unknown = [iid for iid in stale_ids if iid not in items_by_id]
        if unknown:
            raise RuntimeError(
                f"Stale redrive checkpoint {rd_dir} references {len(unknown)} item id(s) "
                f"not in this run's registered item set (first: {unknown[:5]}) — the "
                "checkpoint belongs to a different assignment/context set. Refusing to "
                "merge; quarantine it (mv redriveN redriveN.stale) only after verifying "
                "on the Anthropic batch console that no batch there is still in flight."
            )
        sub = [items_by_id[iid] for iid in stale_ids]
        logger.info("Resuming stale redrive checkpoint %s (%d rows)", rd_dir.name, len(sub))
        rd_results = asyncio.run(_dispatch(sub, rd_dir, GEN_MAX_TOKENS, poll_interval))
        results.update(rd_results)
        batch_meta.update(load_batch_meta(rd_dir))


def _require_redrive_headroom(next_round: int, n_pending: int) -> None:
    """Cumulative re-drive ceiling across ALL process runs (FIX C).

    The numbered ``redriveN`` checkpoint dirs on disk ARE the durable
    cumulative counter (each fresh paid round creates exactly one, and
    :func:`_next_redrive_round` numbers past them), so no new persisted
    counter is needed. When the NEXT fresh round would exceed
    ``MAX_CUMULATIVE_REDRIVE_ROUNDS``, halt-and-report with exit 3 instead
    of dispatching another paid round — an accidental re-run loop is
    bounded while a deliberate continuation stays possible via the named
    override.
    """
    if next_round <= MAX_CUMULATIVE_REDRIVE_ROUNDS:
        return
    logger.error(
        "HALT-AND-REPORT: cumulative re-drive ceiling reached — the next fresh re-drive "
        "round would be redrive%d, past MAX_CUMULATIVE_REDRIVE_ROUNDS=%d (the numbered "
        "redriveN checkpoint dirs on disk are the durable cumulative counter), with %d "
        "transport-class rows still pending. Refusing to dispatch another paid round. "
        "Deliberate continuation: raise MAX_CUMULATIVE_REDRIVE_ROUNDS in "
        "scripts/issue823_ladder_gen.py and re-run (already-paid successes keep resuming "
        "from the stale redrive checkpoints). Quarantining checkpoints "
        "(mv redriveN redriveN.stale) also lowers the counter, but DISCARDS those "
        "checkpoints' already-paid successes from the merge — reserve it for checkpoints "
        "verified dead/corrupt on the Anthropic batch console. Repeated exhaustion at "
        "this ceiling indicates a persistent transport / rate-limit problem, not a "
        "transient blip.",
        next_round,
        MAX_CUMULATIVE_REDRIVE_ROUNDS,
        n_pending,
    )
    sys.exit(3)


def personas_over_cap_threshold(
    stop_by_item: dict[str, str | None], pairs: set[tuple[int, int]]
) -> dict[int, list[int]]:
    """Per-persona cap-hit trigger: personas whose max_tokens fraction > 2 percent.

    Returns {persona: [over-cap context_ids]} for triggered personas only.
    """
    by_p: dict[int, list[int]] = {}
    for i, p in pairs:
        if stop_by_item.get(make_item_id(p, i)) == "max_tokens":
            by_p.setdefault(p, []).append(i)
    n_rows = Counter(p for _, p in pairs)
    return {
        p: sorted(rows)
        for p, rows in by_p.items()
        if len(rows) / n_rows[p] > CAP_HIT_REGEN_FRACTION
    }


def _dispatch_pooled_regen(
    root: pathlib.Path,
    items_by_id: dict[str, DispatchItem],
    regen_personas: dict[int, list[int]],
    poll_interval: float,
) -> tuple[dict[str, DispatchResult], list[str], pathlib.Path]:
    """ONE pooled Batch dispatch for every triggered persona's over-cap rows (FIX D).

    The >2 percent trigger stays PER PERSONA (``personas_over_cap_threshold``
    is unchanged); only the TRANSPORT is pooled. The pre-fix path dispatched
    one batch per persona SERIALLY, each waiting out its own Batch API
    round-trip (up to the 24 h service window) before the next persona
    started — 16 personas meant 16 sequential round-trips (#823 pilot log).
    Per-persona ACCOUNTING is not pooled: the caller keys per-record
    ``regen`` flags and ``max_tokens`` by item id, and item ids encode the
    persona. Returns ``(results, pooled_ids, checkpoint_dir)``.
    """
    for p, ctx_rows in sorted(regen_personas.items()):
        logger.warning(
            "Cap-hit trigger: persona %d has %d over-cap rows — re-generating at %d tokens",
            p,
            len(ctx_rows),
            REGEN_MAX_TOKENS,
        )
    pooled_ids = sorted(
        make_item_id(p, i) for p, ctx_rows in regen_personas.items() for i in ctx_rows
    )
    rg_dir = root / "regen_pooled"
    logger.info(
        "Pooled cap-hit regen: ONE batch dispatch for %d rows across %d triggered persona(s)",
        len(pooled_ids),
        len(regen_personas),
    )
    sub = [items_by_id[iid] for iid in pooled_ids]
    rg = asyncio.run(_dispatch(sub, rg_dir, REGEN_MAX_TOKENS, poll_interval))
    return rg, pooled_ids, rg_dir


# ── Record building + persistence ────────────────────────────────────────────


def classify_validity(res: DispatchResult) -> str:
    if res.error:
        if res.category == "empty_response":
            return "empty"
        return f"error:{res.category}"
    if not isinstance(res.result, str) or not res.result.strip():
        return "empty"
    if res.stop_reason == "refusal":
        return "refusal"
    return "ok"


def build_records(
    questions: list[str],
    in_common: list[bool],
    pairs: set[tuple[int, int]],
    assignment: dict[int, list[int]],
    results: dict[str, DispatchResult],
    batch_meta: dict[str, dict],
    max_tokens_by_item: dict[str, int],
    regen_items: set[str],
) -> dict[int, list[dict]]:
    """Per-persona generation records; every record batch-provenance-complete."""
    by_persona: dict[int, list[dict]] = {p: [] for p in range(N_PERSONAS)}
    for i, p in sorted(pairs):
        item_id = make_item_id(p, i)
        res = results[item_id]
        meta = batch_meta[item_id]
        validity = classify_validity(res)
        rec = {
            "context_id": i,
            "persona_idx": p,
            "persona_name": PERSONAS[p][0],
            "arms": [k for k in K_ARMS if assignment[k][i] == p],
            "question": questions[i],
            "answer_text": res.result if validity in ("ok", "refusal") else None,
            "seed": 42,  # file-provenance label; the API exposes no sampling seed
            "filled": validity == "ok",
            "validity": validity,
            "stop_reason": res.stop_reason,
            "cap_hit": res.stop_reason == "max_tokens",
            "in_common_valid": in_common[i],
            "model": SONNET_MODEL,
            "temperature": GEN_TEMPERATURE,
            "max_tokens": max_tokens_by_item[item_id],
            "regen": item_id in regen_items,
            "batch_id": meta["batch_id"],
            "batch_request_custom_id": meta["batch_request_custom_id"],
            "batch_org": meta["batch_org"],
            "batch_submitted_at": meta["batch_submitted_at"],
            # Per-SUB-BATCH harvest time (results_<batch_id>.json mtime, joined
            # in load_batch_meta) — never a global record-build-time stamp.
            "harvested_at": meta["harvested_at"],
        }
        # REQUIREMENT 1: batch provenance is present on EVERY persisted record.
        assert rec["batch_id"] and rec["batch_request_custom_id"], (
            f"{item_id}: missing batch_id/custom_id — batch provenance incomplete"
        )
        assert rec["batch_submitted_at"] and rec["harvested_at"], (
            f"{item_id}: missing batch timestamps — batch provenance incomplete"
        )
        assert rec["batch_org"], f"{item_id}: missing batch_org — batch provenance incomplete"
        assert rec["arms"], f"{item_id}: pair belongs to no arm — assignment join broken"
        by_persona[p].append(rec)
    return by_persona


def build_digest(
    n_contexts: int,
    pairs: set[tuple[int, int]],
    assignment: dict[int, list[int]],
    by_persona: dict[int, list[dict]],
    mask_crosscheck: dict,
    redrive_rounds: int,
    regen_personas: dict[int, list[int]],
    metadata: dict,
) -> dict:
    validity_by_persona: dict[int, dict] = {}
    cap_frac_by_persona: dict[int, float] = {}
    wave_by_persona: dict[int, dict[str, int]] = {}
    for p, recs in by_persona.items():
        vc = Counter(r["validity"] for r in recs)
        n_cap = sum(1 for r in recs if r["cap_hit"])
        validity_by_persona[p] = {**dict(vc), "cap_hit": n_cap, "n_rows": len(recs)}
        cap_frac_by_persona[p] = (n_cap / len(recs)) if recs else 0.0
        wave_by_persona[p] = dict(Counter(r["batch_id"] for r in recs))
    rec_by_pair = {
        (r["context_id"], r["persona_idx"]): r for recs in by_persona.values() for r in recs
    }
    cap_frac_by_arm_persona: dict[int, dict[int, float]] = {}
    for k in K_ARMS:
        per_p: dict[int, list[bool]] = {}
        for i in range(n_contexts):
            p = assignment[k][i]
            per_p.setdefault(p, []).append(rec_by_pair[(i, p)]["cap_hit"])
        cap_frac_by_arm_persona[k] = {p: sum(v) / len(v) for p, v in per_p.items()}
    error_ids = sorted(
        make_item_id(r["persona_idx"], r["context_id"])
        for recs in by_persona.values()
        for r in recs
        if r["validity"].startswith("error:")
    )
    return {
        "metadata": metadata,
        "n_contexts": n_contexts,
        "n_pairs": len(pairs),
        "registered_total_pairs_full": REGISTERED_TOTAL_PAIRS,
        "mask_crosscheck": mask_crosscheck,
        "validity_counts_by_persona": validity_by_persona,
        "cap_hit_fraction_by_persona": cap_frac_by_persona,
        "cap_hit_fraction_by_arm_persona": cap_frac_by_arm_persona,
        "cap_hit_regen_trigger_fraction": CAP_HIT_REGEN_FRACTION,
        "regen_personas_triggered": {p: len(rows) for p, rows in regen_personas.items()},
        # Realized TRIGGER-TIME over-cap rate per triggered persona (count +
        # denominator + fraction, readable without recomputation). The
        # post-regen `cap_hit` fields above UNDERSTATE the first-pass rate
        # for triggered personas, because regenerated rows usually clear the
        # 2x cap; non-triggered personas' first-pass rate IS
        # cap_hit_fraction_by_persona (they were never regenerated).
        "regen_over_cap_at_trigger_by_persona": {
            p: {
                "n_over_cap": len(rows),
                "n_rows": validity_by_persona[p]["n_rows"],
                "fraction": len(rows) / validity_by_persona[p]["n_rows"],
            }
            for p, rows in sorted(regen_personas.items())
        },
        "redrive_rounds_used": redrive_rounds,
        "n_error_rows": len(error_ids),
        "error_row_ids_first20": error_ids[:20],
        "batch_wave_by_persona": wave_by_persona,
    }


def write_json(path: pathlib.Path, obj: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1))
    tmp.rename(path)


def _require_canonical_upload(url: str, canonical_url: str) -> None:
    """Refuse to declare P-Gen complete on an overflow-rerouted upload.

    ``_upload_folder_filtered``'s default-on file-count fallback re-uploads to
    ``DEFAULT_OVERFLOW_REPO``, runs its exact-set verification against the
    OVERFLOW repo, and returns a truthy overflow URL — so a bare truthiness
    check would log "P-Gen complete" while the plan-declared CANONICAL paths
    do not exist. Raises unless the returned URL is exactly the canonical one.
    """
    if url != canonical_url:
        raise RuntimeError(
            "HALT-AND-REPORT: HF upload landed OFF the canonical data repo — returned "
            f"{url!r}, expected {canonical_url!r} (file-count overflow reroute). The "
            f"plan-declared canonical paths do not exist; overflow pointer: {url}. "
            "Refusing to report P-Gen complete. Remedy: triage/free the canonical "
            "repo's file-count cap, then re-run (already-landed rows resume from the "
            "batch checkpoints; the upload re-verifies idempotently)."
        )


# ── Main ─────────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "P-Gen for #823 inconsistent-origin-persona-ladder: nested persona-assignment "
            "generation set via the Anthropic Batch API (dispatch, harvest, persist, upload)."
        )
    )
    parser.add_argument("--smoke", action="store_true", help="tiny real run: /tmp + _smoke prefix")
    parser.add_argument(
        "--n-contexts",
        type=int,
        default=None,
        help="context count override (smoke only; production is pinned to 5000)",
    )
    parser.add_argument(
        "--out-root",
        type=pathlib.Path,
        default=None,
        help="override output root (default: repo data/ + eval_results/; smoke: /tmp)",
    )
    parser.add_argument("--poll-interval", type=float, default=None, help="batch poll seconds")
    parser.add_argument(
        "--list-arms", action="store_true", help="print the registered arm list and exit"
    )
    args = parser.parse_args(argv)

    if args.list_arms:
        print(json.dumps({"k_arms": list(K_ARMS), "n_personas": N_PERSONAS}))
        return

    repo_root = pathlib.Path(__file__).resolve().parents[1]
    if args.smoke:
        n_contexts = args.n_contexts if args.n_contexts is not None else 16
        # n >= 9 keeps every strict-nesting assert resolvable (n > k/2 for k=16).
        assert n_contexts >= 9, "--smoke needs --n-contexts >= 9 (strict-nesting floor)"
        assert n_contexts <= N_CONTEXTS_FULL, "--n-contexts exceeds the frozen context set"
        root = args.out_root or pathlib.Path("/tmp/issue-823-smoke/ladder_gen")
        eval_dir = root / "eval_results" / "inconsistent_origin_ladder"
        hf_prefix = HF_PREFIX + "_smoke"
    else:
        if args.n_contexts is not None and args.n_contexts != N_CONTEXTS_FULL:
            parser.error("--n-contexts is smoke-only; production runs the full 5000 contexts")
        n_contexts = N_CONTEXTS_FULL
        root = args.out_root or (repo_root / "data" / "issue_823" / "ladder_gen")
        eval_dir = repo_root / "eval_results" / "issue_823" / "inconsistent_origin_ladder"
        hf_prefix = HF_PREFIX
    poll_interval = (
        args.poll_interval if args.poll_interval is not None else (10.0 if args.smoke else 30.0)
    )
    stage_dir = root / "hf_stage" / "ladder"
    logger.info(
        "P-Gen: n_contexts=%d smoke=%s root=%s hf_prefix=%s",
        n_contexts,
        args.smoke,
        root,
        hf_prefix,
    )

    # 1. Frozen questions (full-grain validation, then slice for smoke).
    questions_full, in_common_full, mask_crosscheck = load_frozen_questions(root / "parent_inputs")
    questions = questions_full[:n_contexts]
    in_common = in_common_full[:n_contexts]

    # 2. Registered nested assignment + asserts.
    assignment = build_assignment(n_contexts)
    pairs = verify_assignment(assignment, n_contexts)
    logger.info("Assignment verified: %d distinct (context, persona) pairs", len(pairs))

    # 3. Dispatch (Batch API, checkpointed) + bounded transport-class re-drive.
    items = build_items(questions, pairs)
    results = asyncio.run(_dispatch(items, root / "batches", GEN_MAX_TOKENS, poll_interval))
    batch_meta = load_batch_meta(root / "batches")
    max_tokens_by_item = {it.item_id: GEN_MAX_TOKENS for it in items}
    items_by_id = {it.item_id: it for it in items}
    # FIX B: merge prior runs' redrive checkpoint outcomes BEFORE computing
    # the pending set — already-paid successes are never re-submitted, and a
    # crash-orphaned batch still completing server-side is harvested instead
    # of double-billed by a blind re-submit of the same rows.
    _merge_stale_redrives(root, items_by_id, results, batch_meta, poll_interval)
    redrive_rounds = 0
    first_round = _next_redrive_round(root)
    for rnd in range(first_round, first_round + MAX_TRANSPORT_REDRIVES):
        pending = transport_class_ids(results)
        if not pending:
            break
        # FIX C: cumulative ceiling on fresh paid rounds across ALL runs.
        _require_redrive_headroom(rnd, len(pending))
        redrive_rounds += 1
        logger.warning(
            "Re-driving %d transport-class rows (round %d of this run, dir redrive%d)",
            len(pending),
            redrive_rounds,
            rnd,
        )
        sub = [items_by_id[iid] for iid in pending]
        # Fresh checkpoint dir per round AND per process run (numbered past any
        # stale redrive dirs): the dispatcher checkpoint re-serves persisted
        # transport rows on resume WITHOUT re-dispatching, so re-drive must
        # never reuse a prior round's — or a prior run's — state.
        rd_results = asyncio.run(
            _dispatch(sub, root / f"redrive{rnd}", GEN_MAX_TOKENS, poll_interval)
        )
        results.update(rd_results)
        batch_meta.update(load_batch_meta(root / f"redrive{rnd}"))
    remaining = transport_class_ids(results)

    metadata = {
        "script": "scripts/issue823_ladder_gen.py",
        "task": 823,
        "followup_label": "inconsistent-origin-persona-ladder",
        "git_commit": _git_commit(),
        "generated_at": _utc_now(),
        "model": SONNET_MODEL,
        "temperature": GEN_TEMPERATURE,
        "max_tokens_default": GEN_MAX_TOKENS,
        "parent_rev": PARENT_REV,
        "n_contexts": n_contexts,
        "smoke": args.smoke,
        "k_arms": list(K_ARMS),
    }
    if remaining:
        report = {
            "metadata": metadata,
            "incomplete": True,
            "reason": "transport_class_rows_remaining_after_redrives",
            "n_remaining": len(remaining),
            "remaining_ids_first20": remaining[:20],
            "redrive_rounds_used": redrive_rounds,
        }
        write_json(eval_dir / "gen_digest.json", report)
        logger.error(
            "HALT-AND-REPORT: %d transport-class rows remain after %d fresh re-drive "
            "round(s) this run (checkpoint dirs through redrive%d); digest at %s. "
            "Re-running this command resumes the main checkpoint AND every stale "
            "redrive checkpoint first (already-paid successes are merged back and a "
            "crash-orphaned batch still completing server-side is harvested — nothing "
            "is re-bought), then RE-SUBMITS only the genuinely-remaining rows in fresh "
            "redrive dirs numbered past the stale ones (stale redrive checkpoints are "
            "never reused for new submissions). Fresh rounds are capped cumulatively "
            "across re-runs at MAX_CUMULATIVE_REDRIVE_ROUNDS=%d. Repeated exit-3 halts "
            "indicate a persistent transport / rate-limit problem — check org limits "
            "and the Anthropic batch console before re-running.",
            len(remaining),
            redrive_rounds,
            first_round + MAX_TRANSPORT_REDRIVES - 1,
            eval_dir / "gen_digest.json",
            MAX_CUMULATIVE_REDRIVE_ROUNDS,
        )
        sys.exit(3)

    # 4. Pre-registered cap-hit re-gen trigger (per persona, > 2 percent);
    # dispatch POOLED into ONE batch (FIX D) — the pre-fix per-persona loop
    # serialized one full Batch API round-trip per triggered persona.
    stop_by_item = {iid: res.stop_reason for iid, res in results.items()}
    regen_personas = personas_over_cap_threshold(stop_by_item, pairs)
    regen_items: set[str] = set()
    if regen_personas:
        rg, pooled_ids, rg_dir = _dispatch_pooled_regen(
            root, items_by_id, regen_personas, poll_interval
        )
        rg_remaining = transport_class_ids(rg)
        if rg_remaining:
            report = {
                "metadata": metadata,
                "incomplete": True,
                "reason": "transport_class_rows_remaining_in_regen",
                "regen_personas_triggered": {
                    p: len(rows) for p, rows in sorted(regen_personas.items())
                },
                "n_remaining": len(rg_remaining),
                "remaining_ids_first20": rg_remaining[:20],
                "regen_checkpoint_dir": str(rg_dir),
                "redrive_rounds_used": redrive_rounds,
            }
            write_json(eval_dir / "gen_digest.json", report)
            logger.error(
                "HALT-AND-REPORT: pooled cap-hit regen has %d transport-class rows "
                "(triggered personas: %s); digest at %s. A plain re-run REPLAYS this "
                "regen checkpoint and re-serves the same transport rows without "
                "re-dispatching (deterministic re-halt). Real remedies: quarantine the "
                "single pooled checkpoint (mv %s %s) so the re-run re-submits the "
                "pooled regen rows fresh, and/or investigate the transport cause (org "
                "rate limits, Anthropic batch console) before re-running.",
                len(rg_remaining),
                sorted(regen_personas),
                eval_dir / "gen_digest.json",
                rg_dir,
                rg_dir.with_name(rg_dir.name + ".stale"),
            )
            sys.exit(3)
        results.update(rg)
        batch_meta.update(load_batch_meta(rg_dir))
        for iid in pooled_ids:
            max_tokens_by_item[iid] = REGEN_MAX_TOKENS
            regen_items.add(iid)

    # 5. Persist per-persona records + assignment + roster + digest + sentinel.
    by_persona = build_records(
        questions,
        in_common,
        pairs,
        assignment,
        results,
        batch_meta,
        max_tokens_by_item,
        regen_items,
    )
    persona_files = []
    for p in range(N_PERSONAS):
        fn = f"persona{p:02d}_seed42.json"
        write_json(
            stage_dir / fn,
            {
                "metadata": {
                    **metadata,
                    "persona_idx": p,
                    "persona_name": PERSONAS[p][0],
                    "persona_card": PERSONAS[p][1],
                    "n_records": len(by_persona[p]),
                },
                "records": by_persona[p],
            },
        )
        persona_files.append(fn)

    assignment_obj = {
        "metadata": metadata,
        "registered_rule": "persona(i, k) = i mod k over the frozen 0-indexed context order",
        "n_contexts": n_contexts,
        "registered_total_pairs_full": REGISTERED_TOTAL_PAIRS,
        "realized_total_pairs": len(pairs),
        "arms": {str(k): assignment[k] for k in K_ARMS},
    }
    roster_obj = {
        "metadata": metadata,
        "template": PERSONA_TEMPLATE,
        "personas": [
            {"idx": p, "name": name, "card": card} for p, (name, card) in enumerate(PERSONAS)
        ],
    }
    write_json(stage_dir / "assignment.json", assignment_obj)
    write_json(stage_dir / "roster.json", roster_obj)

    digest = build_digest(
        n_contexts,
        pairs,
        assignment,
        by_persona,
        mask_crosscheck,
        redrive_rounds,
        regen_personas,
        metadata,
    )
    write_json(eval_dir / "gen_digest.json", digest)
    write_json(eval_dir / "assignment.json", assignment_obj)
    write_json(eval_dir / "roster.json", roster_obj)

    sentinel = {
        "phase": "p_gen",
        "complete": True,
        "metadata": metadata,
        "n_pairs": len(pairs),
        "n_ok": sum(1 for recs in by_persona.values() for r in recs if r["validity"] == "ok"),
        "n_error_rows": digest["n_error_rows"],
        "files_sha256": {
            fn: _sha256_file(stage_dir / fn)
            for fn in (*persona_files, "assignment.json", "roster.json")
        },
    }
    write_json(stage_dir / "_gen_complete.json", sentinel)

    # 6. Upload EVERYTHING in ONE bulk commit (text uploads always, unconditionally).
    expected_files = [*persona_files, "assignment.json", "roster.json", "_gen_complete.json"]
    path_in_repo = f"{hf_prefix}/raw_completions/ladder"
    url = _upload_folder_filtered(
        local_dir=stage_dir,
        repo_id=DATA_REPO,
        repo_type="dataset",
        path_in_repo=path_in_repo,
        allow_patterns=["*.json"],
        expected_repo_paths=[f"{path_in_repo}/{fn}" for fn in expected_files],
    )
    if not url:
        raise RuntimeError(
            f"HF upload of {len(expected_files)} P-Gen files to {DATA_REPO}/{path_in_repo} "
            "failed or verified incomplete — refusing to report P-Gen complete"
        )
    # FIX 1: a truthy URL is NOT completion — the helper's file-count fallback
    # returns a verified OVERFLOW-repo URL; require the canonical repo exactly.
    _require_canonical_upload(url, f"{DATA_REPO}/{path_in_repo}")
    logger.info("P-Gen complete: %d files uploaded to %s", len(expected_files), url)
    logger.info(
        "Digest: %s | ok=%d/%d error=%d redrives=%d regen_personas=%s",
        eval_dir / "gen_digest.json",
        sentinel["n_ok"],
        len(pairs),
        digest["n_error_rows"],
        redrive_rounds,
        sorted(regen_personas),
    )


if __name__ == "__main__":
    main()
