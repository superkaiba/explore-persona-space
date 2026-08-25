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
4. Applies the pre-registered cap-hit re-gen trigger PER (arm x persona)
   CELL (plan v13 section 4.3 step 4 + section 7 -- the v10 per-persona form
   is superseded): any cell (k, p) whose first-wave
   `stop_reason == "max_tokens"` fraction exceeds 2 percent triggers; the
   UNION of over-cap pair rows across ALL triggered cells (a pair belongs to
   several cells under nesting and is deduped) is re-generated ONCE, in ONE
   pooled batch, at max_tokens 8192; regenerated rows are stamped
   `gen_wave: "regen-8192"` at regen time and the per-cell cap-hit fraction
   is RE-MEASURED on final records ("regen ran" is never read as "trigger
   resolved" -- residual over-threshold cells are reported in the digest
   with the literal `cap-hit>2%` label, not re-cascaded).
5. Persists 16 per-persona record files (every record carrying its Batch
   request custom_id, batch id, submitted/harvested timestamps, its PERSISTED
   requested-max_tokens + `gen_wave`, and the exact dispatched system prompt
   + sha256 -- the batch-wave drift + P0 prompt-integrity audit fields, free
   now and unrecoverable later) + `assignment.json` + `roster.json` +
   `_gen_complete.json`, and uploads all of them to the HF data repo in ONE
   bulk commit BEFORE any pod exists.

Resume safety (plan v13 section 4.3 step 3b): a generation-config
FINGERPRINT (model, temperature, base/regen caps, roster/template hash,
context pin, mask-gate schema id) is persisted into EVERY dispatcher
checkpoint dir and into `_gen_complete.json`; a resume whose live config
mismatches a checkpoint's persisted fingerprint is a designed halt BEFORE
any dispatch or row re-serve, and every row's recorded cap is read from the
serving checkpoint's persisted metadata, never the live module constants.

Usage:
  uv run python scripts/issue823_ladder_gen.py --smoke      # 16 contexts, /tmp + _smoke HF prefix
  uv run python scripts/issue823_ladder_gen.py              # full 5,000-context production run
  uv run python scripts/issue823_ladder_gen.py --p0-verify  # P0 gate over persisted artifacts

Exit codes: 0 = complete; 3 = transport-class rows remain (after this run's
bounded FRESH re-drive rounds, or inside the pooled cap-hit regen) --
halt-and-report: digest written, no sentinel, no upload. The dispatcher
re-serves persisted transport rows on a resumed checkpoint WITHOUT
re-dispatching, so: a re-run resumes completed rows from the batch
checkpoints and re-submits redrive residue automatically in fresh redrive
dirs numbered past the stale ones; regen residue requires quarantining the
halt-named regen checkpoint dir first (the halt message carries the exact mv
command). 4 = generation-config fingerprint mismatch on resume (designed
halt BEFORE any dispatch or row re-serve; report JSON written next to the
checkpoint root -- remedies: restore the config, or deliberately re-key a
FRESH checkpoint root; never resume across a config change). 5 = P0
prompt-integrity halt (plan section 4.3 P0 asserts (a)-(d); report JSON in
the eval dir). Completion additionally requires the upload to land on the
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
import time
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

# ── Registered constants (plan v13 slice, sections 0 / 4.1 / 4.2 / 4.3) ─────
SONNET_MODEL = "claude-sonnet-4-5-20250929"  # == run_823.SONNET_MODEL (parent parity)
# Cap ladder (plan v13 section 4.3 step 4 + section 11): base 4096 (amended
# v11 -- the parent's 1024 and the pilot's 2048 regen are MEASURED
# insufficient: 21/896 = 2.34% of final pilot records cap-hit, worst cell
# k=16 p05 15.79%; 4096 = 2x the measured-insufficient cap per the project
# >= 2x doubling convention), ONE regen round at 8192.
GEN_MAX_TOKENS = 4096
REGEN_MAX_TOKENS = 8192  # pre-registered ONE-round re-gen cap for over-cap rows
GEN_TEMPERATURE = 1.0
# gen_wave labels (plan v13 section 4.3 step 2): stamped EXPLICITLY at
# dispatch/regen time, never reverse-engineered from batch ids.
GEN_WAVE_FIRST = "first"
GEN_WAVE_REGEN = f"regen-{REGEN_MAX_TOKENS}"  # == "regen-8192" (registered literal)
# Mask-gate schema id (plan v13 section 4.3 step 3b(i) fingerprint field):
# names the v11 stop_reason-keyed validity/class-split schema the section 4.3
# mask gates read (refusal precedence over the empty-content check;
# classify_validity below).
MASK_GATE_SCHEMA_ID = "issue823_mask_gate_v11_stop_reason_precedence"
# Designed-halt exit codes (3 = transport residue, pre-existing).
EXIT_CONFIG_MISMATCH = 4  # resume fingerprint mismatch (section 4.3 step 3b(ii))
EXIT_P0_INTEGRITY = 5  # P0 prompt-integrity halt (section 4.3 P0 asserts (a)-(d))
GEN_CONFIG_FILENAME = "gen_config.json"  # per-checkpoint fingerprint + dispatch cap
# P0 VERIFIER schema version (round-6 BLOCKER 1): keys the --p0-verify
# per-persona checkpoint on the VERIFIER's own record-validation contract, not
# only the GENERATION fingerprint (which says what was generated, not which
# verifier judged it). A checkpoint written by an older verifier — e.g. the
# round-4 isinstance-int schema, which recorded n_failures: 0 for a
# `context_id: true` record — must never be reused by a newer one: a missing
# or differing value discards the WHOLE checkpoint (exactly like a generation-
# fingerprint mismatch) and re-verifies from scratch. BUMP this constant on
# ANY change to the verifier's record-validation contract. Version 2 == the
# round-5 strict type-is-int schema (version 1 was the round-4 isinstance-int
# schema, which never wrote this key).
P0_VERIFIER_SCHEMA_VERSION = 2
CAP_HIT_REGEN_FRACTION = 0.02  # per-(arm x persona) CELL stop-and-regen trigger (v13 §7)
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


# ── Generation-config fingerprint (plan v13 section 4.3 step 3b) ─────────────


def roster_template_hash() -> str:
    """sha256 over METADATA-FREE roster content: template + personas (idx, name, card).

    HASH-BASIS PIN (plan v13 section 4.3 step 3b(i), binding): the hash
    covers ONLY the content fields -- the template string plus the personas
    array -- and NEVER the generated ``roster.json`` as a whole, whose
    ``metadata`` block embeds ``generated_at`` and would make the fingerprint
    differ between two runs of an IDENTICAL config (rejecting healthy
    resumes). Process-time stability is pinned by the resume-stability
    fixture in ``tests/test_issue823_ladder_gen_fixes.py``.
    """
    basis = {
        "template": PERSONA_TEMPLATE,
        "personas": [
            {"idx": p, "name": name, "card": card} for p, (name, card) in enumerate(PERSONAS)
        ],
    }
    blob = json.dumps(basis, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def generation_config_fingerprint() -> dict:
    """The LIVE generation-config fingerprint (plan v13 section 4.3 step 3b(i)).

    Returns ``{"fields": {...}, "sha256": <hex>}`` -- sha256 over (model,
    temperature, base cap, regen cap, roster/template hash, context pin,
    mask-gate schema id). Deliberately wall-clock-free.
    """
    fields = {
        "model": SONNET_MODEL,
        "temperature": GEN_TEMPERATURE,
        "gen_max_tokens": GEN_MAX_TOKENS,
        "regen_max_tokens": REGEN_MAX_TOKENS,
        "roster_template_hash": roster_template_hash(),
        "context_pin": PARENT_REV,
        "mask_gate_schema_id": MASK_GATE_SCHEMA_ID,
    }
    blob = json.dumps(fields, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return {"fields": fields, "sha256": hashlib.sha256(blob.encode("utf-8")).hexdigest()}


def _halt_config_mismatch(checkpoint_dir: pathlib.Path, report: dict) -> None:
    """Designed halt for a resume-fingerprint failure (section 4.3 step 3b(ii))."""
    report_path = checkpoint_dir.parent / f"fingerprint_mismatch_{checkpoint_dir.name}.json"
    write_json(report_path, report)
    logger.error(
        "HALT-AND-REPORT: generation-config fingerprint check FAILED for checkpoint %s "
        "(reason=%s, differing_fields=%s); report at %s. A resume across a config change "
        "would re-serve rows generated under the OLD config but labeled with the new one. "
        "Remedies: restore the persisted config, or deliberately re-key a FRESH checkpoint "
        "root -- never resume across a config change.",
        checkpoint_dir,
        report.get("reason"),
        sorted(report.get("differing_fields", {})),
        report_path,
    )
    sys.exit(EXIT_CONFIG_MISMATCH)


def check_or_persist_gen_config(checkpoint_dir: pathlib.Path, max_tokens: int) -> int:
    """Persist the fingerprint into a fresh checkpoint dir, or verify it on resume.

    Runs BEFORE any dispatch on ``checkpoint_dir`` -- and therefore before
    the dispatcher can re-serve any persisted row from it (plan v13 section
    4.3 step 3b(ii)). A fresh dir gets ``gen_config.json`` (the fingerprint
    plus THIS dir's dispatch cap). An existing record is compared
    field-by-field against the live fingerprint + cap; ANY mismatch -- and a
    dispatcher ``state.json`` with no fingerprint record (a pre-fingerprint
    checkpoint, unverifiable) -- is a designed halt: report JSON naming both
    fingerprints and the differing fields, exit ``EXIT_CONFIG_MISMATCH`` (4,
    distinct from the transport-halt exit 3). Never silent consumption.

    Returns the checkpoint's PERSISTED dispatch cap (== ``max_tokens`` once
    the check passes) so callers thread per-row caps from persisted request
    metadata (step 3b(iii)), never the live module constants.
    """
    live = generation_config_fingerprint()
    cfg_path = checkpoint_dir / GEN_CONFIG_FILENAME
    if not cfg_path.is_file():
        if (checkpoint_dir / "state.json").is_file():
            _halt_config_mismatch(
                checkpoint_dir,
                {
                    "reason": "unfingerprinted_checkpoint",
                    "checkpoint_dir": str(checkpoint_dir),
                    "persisted_fingerprint": None,
                    "live_fingerprint": live,
                    "differing_fields": {},
                    "detail": (
                        "dispatcher state.json exists but no gen_config.json -- a "
                        "pre-fingerprint checkpoint cannot be verified against the "
                        "live config; refusing to re-serve its rows"
                    ),
                    "detected_at": _utc_now(),
                },
            )
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        write_json(cfg_path, {"fingerprint": live, "max_tokens": max_tokens})
        return max_tokens
    persisted = json.loads(cfg_path.read_text())
    p_fp = persisted.get("fingerprint") or {}
    p_fields = p_fp.get("fields") or {}
    diffs: dict[str, dict] = {}
    for key in sorted(set(p_fields) | set(live["fields"])):
        if p_fields.get(key) != live["fields"].get(key):
            diffs[key] = {"persisted": p_fields.get(key), "live": live["fields"].get(key)}
    if persisted.get("max_tokens") != max_tokens:
        diffs["max_tokens"] = {"persisted": persisted.get("max_tokens"), "live": max_tokens}
    # MINOR 5 (round 4): a corrupted persisted sha with otherwise-identical
    # fields must still name WHAT differs — the halt report's diagnostic
    # contract is a non-empty differing_fields on every halt.
    if p_fp.get("sha256") != live["sha256"]:
        diffs["fingerprint_sha256"] = {"persisted": p_fp.get("sha256"), "live": live["sha256"]}
    if not diffs:
        return int(persisted["max_tokens"])
    _halt_config_mismatch(
        checkpoint_dir,
        {
            "reason": "generation_config_fingerprint_mismatch",
            "checkpoint_dir": str(checkpoint_dir),
            "persisted_fingerprint": p_fp,
            "live_fingerprint": live,
            "persisted_max_tokens": persisted.get("max_tokens"),
            "live_max_tokens": max_tokens,
            "differing_fields": diffs,
            "detected_at": _utc_now(),
        },
    )
    raise AssertionError("unreachable: _halt_config_mismatch exits")


def checkpoint_max_tokens(checkpoint_dir: pathlib.Path) -> int:
    """Read a checkpoint dir's PERSISTED dispatch cap (section 4.3 step 3b(iii)).

    Per-row recorded caps derive from this persisted request metadata --
    never from the live ``GEN_MAX_TOKENS`` / ``REGEN_MAX_TOKENS`` module
    constants -- so a resume after a cap re-tune can never mislabel
    re-served rows (the fingerprint check halts such a resume first; this
    read is the belt over that suspender). Fail-loud on a missing record:
    production callers read it only AFTER ``_dispatch`` ran the
    check-or-persist gate on the same dir.
    """
    cfg = json.loads((checkpoint_dir / GEN_CONFIG_FILENAME).read_text())
    return int(cfg["max_tokens"])


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
    # Section 4.3 step 3b(ii): fingerprint check/persist runs BEFORE the
    # dispatcher touches this checkpoint dir -- so a config-mismatched resume
    # halts before any dispatch AND before any persisted row is re-served.
    check_or_persist_gen_config(checkpoint_dir, max_tokens)
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
    max_tokens_by_item: dict[str, int],
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
        # Section 4.3 step 3b(iii): re-served rows' caps come from the stale
        # checkpoint's PERSISTED gen_config.json (written/verified by
        # _dispatch's fingerprint gate just above), never the live constant.
        stale_cap = checkpoint_max_tokens(rd_dir)
        for iid in stale_ids:
            max_tokens_by_item[iid] = stale_cap


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


def cells_over_cap_threshold(
    stop_by_item: dict[str, str | None], assignment: dict[int, list[int]]
) -> tuple[dict[str, dict], dict[int, list[int]]]:
    """Per-(arm x persona) CELL cap-hit trigger (plan v13 §4.3 step 4 + §7).

    The registered trigger is PER CELL — the v10 per-persona pooled form is
    SUPERSEDED (plan §7 Decision-Gates row: "fraction vs 2% per
    (arm x persona) cell"). Cell (k, p) is the contexts i with
    ``assignment[k][i] == p``; each cell's FIRST-WAVE
    ``stop_reason == "max_tokens"`` fraction is compared against
    ``CAP_HIT_REGEN_FRACTION`` (strictly greater triggers). A pooled
    per-persona rate can sit at or under 2 percent while one cell (e.g. the
    k=16 cell, denominator 313) is over — exactly the case the per-persona
    form missed.

    MULTI-ARM ROWS: under nesting a single (context, persona) pair row
    belongs to SEVERAL cells (i=5, p=5 sits in both the k=8 and k=16 cells);
    the regen set is the UNION of over-cap pair ids across ALL triggered
    cells, deduped by pair identity — one pair regenerates at most once,
    however many of its cells triggered.

    Returns ``(triggered_cells, regen_pairs)``:
      - ``triggered_cells``: ``{"k=<k>,p=<p>": {k, persona, n_over_cap,
        n_rows, fraction}}`` — the TRIGGER-TIME (first-wave) per-cell
        record, for triggered cells only;
      - ``regen_pairs``: ``{persona: sorted [context_ids]}`` — the deduped
        union grouped by persona (the shape ``_dispatch_pooled_regen``
        consumes; item ids encode the persona, so the grouping is purely a
        transport convenience — the trigger itself is per cell).
    """
    triggered_cells: dict[str, dict] = {}
    regen_union: dict[int, set[int]] = {}
    for k in K_ARMS:
        cell_rows: dict[int, list[int]] = {}
        for i, p in enumerate(assignment[k]):
            cell_rows.setdefault(p, []).append(i)
        for p, rows in sorted(cell_rows.items()):
            over = [i for i in rows if stop_by_item.get(make_item_id(p, i)) == "max_tokens"]
            if len(over) / len(rows) > CAP_HIT_REGEN_FRACTION:
                triggered_cells[f"k={k},p={p}"] = {
                    "k": k,
                    "persona": p,
                    "n_over_cap": len(over),
                    "n_rows": len(rows),
                    "fraction": len(over) / len(rows),
                }
                regen_union.setdefault(p, set()).update(over)
    regen_pairs = {p: sorted(rows) for p, rows in sorted(regen_union.items())}
    return triggered_cells, regen_pairs


def _dispatch_pooled_regen(
    root: pathlib.Path,
    items_by_id: dict[str, DispatchItem],
    regen_pairs: dict[int, list[int]],
    poll_interval: float,
) -> tuple[dict[str, DispatchResult], list[str], pathlib.Path]:
    """ONE pooled Batch dispatch for every triggered CELL's over-cap rows (FIX D).

    The >2 percent trigger is PER (arm x persona) CELL
    (``cells_over_cap_threshold``); ``regen_pairs`` is the deduped UNION of
    over-cap pair ids across ALL triggered cells, grouped by persona for
    item-id construction. Only the TRANSPORT is pooled — the pre-fix path
    dispatched one batch per persona SERIALLY, each waiting out its own
    Batch API round-trip (up to the 24 h service window) before the next
    started (#823 pilot log); FIX D's single-round-trip property survives
    the per-cell trigger unchanged: exactly ONE regen round, ONE dispatch,
    no cascade. Per-row ACCOUNTING is not pooled: the caller keys
    per-record ``regen`` flags and ``max_tokens`` by item id, and item ids
    encode the persona. Returns ``(results, pooled_ids, checkpoint_dir)``.
    """
    for p, ctx_rows in sorted(regen_pairs.items()):
        logger.warning(
            "Cap-hit regen union: persona %d contributes %d over-cap pair row(s) — "
            "re-generating at %d tokens",
            p,
            len(ctx_rows),
            REGEN_MAX_TOKENS,
        )
    pooled_ids = sorted(make_item_id(p, i) for p, ctx_rows in regen_pairs.items() for i in ctx_rows)
    rg_dir = root / "regen_pooled"
    logger.info(
        "Pooled cap-hit regen: ONE batch dispatch for %d rows spanning %d persona(s)",
        len(pooled_ids),
        len(regen_pairs),
    )
    sub = [items_by_id[iid] for iid in pooled_ids]
    rg = asyncio.run(_dispatch(sub, rg_dir, REGEN_MAX_TOKENS, poll_interval))
    return rg, pooled_ids, rg_dir


# ── Record building + persistence ────────────────────────────────────────────


def classify_validity(res: DispatchResult) -> str:
    """Validity label per pair, keyed on the persisted ``stop_reason``.

    Label-precedence fix (plan v13 section 4.3 step 4, v11 pilot-found):
    a refusal-stop takes precedence over EVERY empty-content check -- an
    API-level refusal row has empty content by construction (its
    ``DispatchResult`` can arrive as ``error=True`` / ``empty_response``),
    and keying on emptiness first undercounted pilot refusals by 24%
    (19 vs the true 25). The mask gates read ``stop_reason`` directly and
    are immune either way; this keeps the M3 report consistent with them.
    """
    if res.stop_reason == "refusal":
        return "refusal"
    if res.error:
        if res.category == "empty_response":
            return "empty"
        return f"error:{res.category}"
    if not isinstance(res.result, str) or not res.result.strip():
        return "empty"
    return "ok"


def build_records(
    questions: list[str],
    in_common: list[bool],
    pairs: set[tuple[int, int]],
    assignment: dict[int, list[int]],
    results: dict[str, DispatchResult],
    batch_meta: dict[str, dict],
    items_by_id: dict[str, DispatchItem],
    max_tokens_by_item: dict[str, int],
    gen_wave_by_item: dict[str, str],
    regen_items: set[str],
) -> dict[int, list[dict]]:
    """Per-persona generation records; every record batch-provenance-complete."""
    by_persona: dict[int, list[dict]] = {p: [] for p in range(N_PERSONAS)}
    for i, p in sorted(pairs):
        item_id = make_item_id(p, i)
        res = results[item_id]
        meta = batch_meta[item_id]
        validity = classify_validity(res)
        # The dispatched system prompt VERBATIM: make_build_request forwards
        # payload["system"] unchanged into the API params. A resume that could
        # desynchronize the rebuilt payload from what was originally
        # dispatched is halted upstream by the fingerprint gate (the
        # roster/template hash) + the dispatcher's own request_fingerprint
        # check -- so this IS the per-pair P0 evidence (plan step 2).
        system_prompt = items_by_id[item_id].payload["system"]
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
            # PERSISTED per-row cap + explicit wave label (section 4.3 steps
            # 2 / 3b(iii)-(iv)): both threaded from checkpoint-persisted /
            # regen-time stamps, never re-derived from live constants.
            "max_tokens": max_tokens_by_item[item_id],
            "gen_wave": gen_wave_by_item[item_id],
            "regen": item_id in regen_items,
            "system_prompt": system_prompt,
            "system_prompt_sha256": hashlib.sha256(system_prompt.encode("utf-8")).hexdigest(),
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
    triggered_cells: dict[str, dict],
    regen_pairs: dict[int, list[int]],
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
    # CONVERGENCE SEMANTICS (plan v13 section 4.3 step 4): the post-regen
    # RE-MEASURE on FINAL records -- "regen ran" is never read as "trigger
    # resolved". Cells still over the trigger are NOT re-cascaded; they are
    # enumerated here for the M3 truncation-confound diagnostic, each
    # carrying the LITERAL `cap-hit>2%` label every downstream table/figure
    # must render (plan section 7 Decision-Gates row).
    cells_over_post_regen = [
        {"k": k, "persona": p, "cap_hit_fraction": f, "label": "cap-hit>2%"}
        for k in K_ARMS
        for p, f in sorted(cap_frac_by_arm_persona[k].items())
        if f > CAP_HIT_REGEN_FRACTION
    ]
    return {
        "metadata": metadata,
        "n_contexts": n_contexts,
        "n_pairs": len(pairs),
        "registered_total_pairs_full": REGISTERED_TOTAL_PAIRS,
        "mask_crosscheck": mask_crosscheck,
        "validity_counts_by_persona": validity_by_persona,
        "cap_hit_fraction_by_persona": cap_frac_by_persona,
        "cap_hit_fraction_by_arm_persona": cap_frac_by_arm_persona,
        "cap_hit_cells_over_threshold_post_regen": cells_over_post_regen,
        "cap_hit_regen_trigger_fraction": CAP_HIT_REGEN_FRACTION,
        # Realized TRIGGER-TIME (first-wave) per-CELL record: count +
        # denominator + fraction per triggered (arm x persona) cell, readable
        # without recomputation. The post-regen `cap_hit` fields above
        # UNDERSTATE the first-pass rate for regenerated rows (regenerated
        # rows usually clear the 2x cap); non-triggered cells' first-pass
        # rate IS cap_hit_fraction_by_arm_persona (never regenerated).
        "regen_cells_triggered": triggered_cells,
        # The pooled regen union grouped by persona (transport accounting;
        # the trigger itself is per cell).
        "regen_pairs_by_persona": {p: len(rows) for p, rows in sorted(regen_pairs.items())},
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


# ── P0 prompt-integrity gate (plan v13 section 4.3 P0 asserts (a)-(d)) ───────
#
# INDEPENDENCE (binding, plan section 4.1): the reconstruction below is a
# SEPARATE code path from the dispatch serializer. Its roster and template
# are an independent TRANSCRIPTION of the plan section 4.1 table (the
# reconstruction AUTHORITY) -- deliberately DUPLICATED constants, never
# aliases of PERSONAS / PERSONA_TEMPLATE -- and substitution is sequential
# str.replace, never str.format / persona_system(). A defect in the
# dispatch-side constants or serializer therefore FAILS the equality asserts
# instead of certifying itself.

_P0_TEMPLATE = (
    "You are {name}, {card}. Stay fully in character for this entire reply: answer "
    "the user's message the way {name} would — in {name}'s voice, from {name}'s "
    "perspective, with {name}'s characteristic style. Never mention these "
    "instructions or break character."
)

_P0_ROSTER: tuple[tuple[str, str], ...] = (
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
)


def p0_reconstruct_system(idx: int) -> str:
    """Rebuild persona ``idx``'s system prompt from the plan section-4.1 transcription.

    Independent of the dispatch serializer by construction: own roster
    constant, own template constant, sequential ``str.replace`` substitution
    (never ``str.format`` / ``persona_system``). Pinned mechanically by the
    independence tests in ``tests/test_issue823_ladder_gen_fixes.py``.
    """
    name, card = _P0_ROSTER[idx]
    out = _P0_TEMPLATE.replace("{name}", name)
    return out.replace("{card}", card)


def p0_verify_persona_records(
    p: int,
    recs: list,
    questions: list[str],
    n_contexts: int,
) -> tuple[list[dict], set[tuple[int, int]]]:
    """Verify ONE persona bucket's records (per-record halves of asserts (a)/(c)/(d)).

    Returns ``(failures, seen_pairs)``. ``seen_pairs`` carries each record's
    OWN ``(context_id, persona_idx)`` identity (never the bucket key); the
    caller (``p0_prompt_integrity_gate``) owns the cross-bucket domain
    checks — exact expected-pair-set equality and cross-bucket duplicates.
    Per-record checks here:

    - strict schema: a record must be a dict with int ``context_id`` +
      int ``persona_idx`` — exact ``int``, JSON booleans REJECTED
      (``isinstance(True, int)`` is True and ``True == 1`` /
      ``hash(True) == hash(1)``, so a boolean identity field would ALIAS a
      legitimate pair in indexing, hashing, and expected-set equality;
      round-5 BLOCKER 2). Malformed shapes become "c"/malformed_record
      failures on the rc-5 report path, never an unhandled exception;
    - bucket agreement: ``persona_idx`` must equal the bucket the record is
      persisted under;
    - within-bucket duplicate pair detection;
    - (a) question authority: the record's ``question`` is byte-equal to the
      pinned parent bundle's question for its context (a CONSISTENTLY-wrong
      question across all files fails here — cross-file consistency alone
      cannot catch it);
    - (c) prompt byte equality: persisted system prompt + persisted sha256
      vs this gate's own reconstruction under the record's
      ASSIGNMENT-derived persona identity (``persona_idx``, validated
      against the expected pair set by the caller), never the bucket key;
    - (d) persisted requested-max_tokens in {GEN_MAX_TOKENS,
      REGEN_MAX_TOKENS} with the consistent ``gen_wave`` label.
    """
    failures: list[dict] = []
    seen: set[tuple[int, int]] = set()
    wave_for_cap = {GEN_MAX_TOKENS: GEN_WAVE_FIRST, REGEN_MAX_TOKENS: GEN_WAVE_REGEN}
    expected_cache: dict[int, tuple[str, str]] = {}
    if not isinstance(recs, list):
        failures.append(
            {
                "assert": "c",
                "kind": "malformed_bucket",
                "persona_bucket": p,
                "detail": f"records payload is {type(recs).__name__}, not a list",
            }
        )
        return failures, seen
    for pos, rec in enumerate(recs):
        if (
            not isinstance(rec, dict)
            # type-is-int, NOT isinstance (round-5 BLOCKER 2): bool is an int
            # subclass and True aliases 1 in indexing / hashing / tuple
            # equality, so a boolean identity could impersonate pair (1, p)
            # and satisfy the exact expected-set reconciliation.
            or type(rec.get("context_id")) is not int
            or type(rec.get("persona_idx")) is not int
        ):
            failures.append(
                {
                    "assert": "c",
                    "kind": "malformed_record",
                    "persona_bucket": p,
                    "record_pos": pos,
                    "detail": (
                        "record is not a dict with int context_id + int persona_idx "
                        "(bool rejected: True aliases 1)"
                    ),
                }
            )
            continue
        i = rec["context_id"]
        p_idx = rec["persona_idx"]
        pair = {"context_id": i, "persona_idx": p_idx}
        if p_idx != p:
            failures.append({"assert": "c", "kind": "bucket_mismatch", **pair, "persona_bucket": p})
        if (i, p_idx) in seen:
            failures.append({"assert": "c", "kind": "duplicate_pair", **pair, "persona_bucket": p})
        seen.add((i, p_idx))
        # (a) question authority vs the pinned bundle.
        if not 0 <= i < min(n_contexts, len(questions)):
            failures.append(
                {"assert": "a", "kind": "context_id_out_of_range", **pair, "n_contexts": n_contexts}
            )
        elif rec.get("question") != questions[i]:
            failures.append(
                {
                    "assert": "a",
                    "kind": "question_mismatch",
                    **pair,
                    "persisted_question": rec.get("question"),
                    "bundle_question": questions[i],
                }
            )
        # (c) prompt byte equality under the record's OWN persona identity.
        if 0 <= p_idx < len(_P0_ROSTER):
            if p_idx not in expected_cache:
                ep = p0_reconstruct_system(p_idx)
                expected_cache[p_idx] = (ep, hashlib.sha256(ep.encode("utf-8")).hexdigest())
            expected_prompt, expected_sha = expected_cache[p_idx]
            if (
                rec.get("system_prompt") != expected_prompt
                or rec.get("system_prompt_sha256") != expected_sha
            ):
                failures.append(
                    {
                        "assert": "c",
                        "kind": "prompt_mismatch",
                        **pair,
                        "persisted_system_prompt": rec.get("system_prompt"),
                        "persisted_sha256": rec.get("system_prompt_sha256"),
                        "reconstructed_system_prompt": expected_prompt,
                        "reconstructed_sha256": expected_sha,
                    }
                )
        else:
            failures.append({"assert": "c", "kind": "persona_idx_out_of_range", **pair})
        # (d) cap/wave consistency.
        mt = rec.get("max_tokens")
        if mt not in wave_for_cap or rec.get("gen_wave") != wave_for_cap[mt]:
            failures.append(
                {
                    "assert": "d",
                    **pair,
                    "persisted_max_tokens": mt,
                    "persisted_gen_wave": rec.get("gen_wave"),
                    "valid_cap_wave_pairs": {str(c): w for c, w in wave_for_cap.items()},
                }
            )
    return failures, seen


def p0_prompt_integrity_gate(
    roster_obj: dict,
    assignment_obj: dict,
    by_persona: dict[int, list[dict]],
    report_dir: pathlib.Path,
    questions: list[str],
    *,
    expected_n_contexts: int,
    verify_persona=None,
) -> None:
    """P0 prompt-integrity gate: designed halt on ANY mismatch, over its OWN domain.

    Assert labels FOLLOW THE PLAN's (a)-(d) list (plan v13 section 4.3 P0):

    (a) generation-record questions == the pinned-bundle questions (byte
        equality) for EVERY record — ``questions`` is the authority, loaded
        from the parent ``b2_seed42.json`` at ``PARENT_REV`` (the frozen
        context order), so a consistently-wrong question cannot pass;
    (b) ``assignment.json`` reproduces persona(i, k) = i mod k by this
        gate's OWN arithmetic over every arm, with structural validation
        first: arm keys == K_ARMS, every arm list length == the pinned
        domain, and the artifact-declared ``n_contexts`` a positive exact
        int (bool rejected) EQUAL to ``expected_n_contexts`` — the
        CALLER-pinned domain authority (production 5000 / the explicit
        smoke count; round-5 BLOCKER 1). The equality is checked in this
        (b) section, BEFORE the (a) coverage read and the (c)
        expected-pair derivation, and every domain-derived check below
        uses the caller's value — the artifact under test never defines
        the authority it is verified against;
    (c) per-pair prompt integrity over the EXACT expected pair DOMAIN —
        presence + byte equality (plan section 7: "presence + byte equality
        for EVERY pair"): the expected pair set is derived INDEPENDENTLY
        from persona(i, k) = i mod k over ``range(n_contexts)`` (never from
        whatever records happen to be present); records flatten under a
        strict schema with unique ``(context_id, persona_idx)`` keys;
        missing pairs, unexpected pairs, duplicates (within and across
        buckets), wrong-bucket records, malformed shapes, and any
        prompt/sha byte mismatch vs this gate's own reconstruction all fail
        here. An ALL-EMPTY record set therefore FAILS (every expected pair
        missing) — never a vacuous PASS. Roster/template transcription
        equality vs the plan section-4.1 authority is (c)'s
        reconstruction-authority precondition and is labeled "c" too;
    (d) EVERY record's persisted requested-max_tokens is in
        {GEN_MAX_TOKENS, REGEN_MAX_TOKENS} with a consistent ``gen_wave``.

    ``verify_persona`` (default :func:`p0_verify_persona_records`) is the
    per-bucket seam ``run_p0_verify`` wraps with its durable per-persona
    checkpoint; each bucket emits one FLUSHED progress line
    (``[p0] persona k/N ...``).

    On ANY failure — malformed shapes included — the rc-5 report path
    fires: report JSON (pair ids, personas, both serializations) at
    ``report_dir / "p0_integrity_report.json"`` + exit ``EXIT_P0_INTEGRITY``
    (5), a designed halt BEFORE mask construction, never silent.
    """
    if verify_persona is None:
        verify_persona = p0_verify_persona_records
    # Caller contract (not artifact input): the pinned domain must itself be
    # a positive exact int — a wrong caller is a code bug, fail loud.
    if type(expected_n_contexts) is not int or expected_n_contexts <= 0:
        raise ValueError(
            f"expected_n_contexts must be a positive int, got {expected_n_contexts!r} — "
            "the caller pins the domain authority (production 5000 / explicit smoke count)"
        )
    failures: list[dict] = []

    # Container roots (round-5 BLOCKER 3): a malformed root becomes a
    # RECORDED failure routed through the rc-5 report path below — never a
    # raw AttributeError on .get before the report machinery runs.
    if not isinstance(roster_obj, dict):
        failures.append(
            {
                "assert": "c",
                "field": "roster_root",
                "generated": type(roster_obj).__name__,
                "detail": "roster.json root is not an object",
            }
        )
        roster_obj = {}
    if not isinstance(assignment_obj, dict):
        failures.append(
            {
                "assert": "b",
                "field": "assignment_root",
                "generated": type(assignment_obj).__name__,
                "detail": "assignment.json root is not an object",
            }
        )
        assignment_obj = {}

    # (c) precondition: roster/template equality vs the independent
    # section-4.1 transcription (the reconstruction authority).
    if roster_obj.get("template") != _P0_TEMPLATE:
        failures.append(
            {
                "assert": "c",
                "field": "template",
                "generated": roster_obj.get("template"),
                "reconstructed": _P0_TEMPLATE,
            }
        )
    gen_personas = roster_obj.get("personas") or []
    if not isinstance(gen_personas, list):
        failures.append(
            {
                "assert": "c",
                "field": "personas_type",
                "generated": type(gen_personas).__name__,
                "detail": "roster.json personas is not a list",
            }
        )
        gen_personas = []
    if len(gen_personas) != len(_P0_ROSTER):
        failures.append(
            {
                "assert": "c",
                "field": "personas_length",
                "generated": len(gen_personas),
                "reconstructed": len(_P0_ROSTER),
            }
        )
    else:
        for p, (name, card) in enumerate(_P0_ROSTER):
            got = gen_personas[p]
            if not isinstance(got, dict) or (got.get("idx"), got.get("name"), got.get("card")) != (
                p,
                name,
                card,
            ):
                failures.append(
                    {
                        "assert": "c",
                        "field": f"personas[{p}]",
                        "generated": got,
                        "reconstructed": {"idx": p, "name": name, "card": card},
                    }
                )

    # (b) structural validation FIRST: the gate verifies its own domain
    # inputs before trusting them (round-4 BLOCKER 2), and the DOMAIN
    # AUTHORITY is the caller-pinned expected_n_contexts (round-5
    # BLOCKER 1): the artifact-declared n_contexts is checked for EQUALITY
    # here — before the (a) coverage read and the (c) expected-pair
    # derivation — and every domain-derived check below uses the CALLER's
    # value, never the artifact's.
    declared_n = assignment_obj.get("n_contexts")
    if type(declared_n) is not int or declared_n <= 0:
        failures.append(
            {
                "assert": "b",
                "field": "n_contexts",
                "generated": declared_n,
                "detail": (
                    "assignment.json n_contexts missing or not a positive int "
                    "(bool rejected: True aliases 1)"
                ),
            }
        )
    elif declared_n != expected_n_contexts:
        failures.append(
            {
                "assert": "b",
                "field": "n_contexts_authority",
                "generated": declared_n,
                "expected": expected_n_contexts,
                "detail": (
                    "assignment.json n_contexts must EQUAL the caller-pinned domain "
                    "(production 5000 / the explicit smoke count) — a coherent "
                    "smaller artifact must never certify a subset of the domain"
                ),
            }
        )
    n_contexts = expected_n_contexts
    arms = assignment_obj.get("arms") or {}
    if not isinstance(arms, dict):
        failures.append(
            {
                "assert": "b",
                "field": "arms_type",
                "generated": type(arms).__name__,
                "detail": "assignment.json arms is not an object of arm-k -> persona list",
            }
        )
        arms = {}
    if sorted(str(k) for k in arms) != sorted(str(k) for k in K_ARMS):
        failures.append(
            {
                "assert": "b",
                "field": "arms_keys",
                "generated": sorted(str(k) for k in arms),
                "reconstructed": sorted(str(k) for k in K_ARMS),
            }
        )
    for k_str, per in sorted(arms.items(), key=lambda kv: str(kv[0])):
        try:
            k = int(k_str)
        except (TypeError, ValueError):
            # Round-5 BLOCKER 3: a nonnumeric arm key is a RECORDED (b)
            # failure, never an unhandled ValueError at int().
            failures.append(
                {
                    "assert": "b",
                    "field": "arm_key",
                    "generated": repr(k_str),
                    "detail": "arm key is not an integer string",
                }
            )
            continue
        if k not in K_ARMS:
            # Round-6 BLOCKER 3: an UNREGISTERED integer arm key (e.g. "0")
            # must route to the rc-5 (b) report — never reach the modulo
            # below (i % 0 raises ZeroDivisionError before the designed
            # report; the arms_keys set-equality entry above records the
            # discrepancy but cannot stop this loop from crashing on it).
            failures.append(
                {
                    "assert": "b",
                    "field": "arm_key_unregistered",
                    "generated": k,
                    "registered": list(K_ARMS),
                    "detail": "arm key is not a registered K_ARMS member",
                }
            )
            continue
        if not isinstance(per, list):
            failures.append(
                {"assert": "b", "arm_k": k, "field": "arm_type", "generated": type(per).__name__}
            )
            continue
        if len(per) != n_contexts:
            failures.append(
                {
                    "assert": "b",
                    "arm_k": k,
                    "field": "arm_length",
                    "generated": len(per),
                    "expected": n_contexts,
                }
            )
        bad = [(i, p) for i, p in enumerate(per) if p != i % k]
        for i, p in bad[:20]:
            failures.append(
                {"assert": "b", "arm_k": k, "context_id": i, "generated": p, "expected": i % k}
            )
        if len(bad) > 20:
            failures.append({"assert": "b", "arm_k": k, "n_bad_total": len(bad)})

    # (a) authority coverage: the pinned bundle must cover the DOMAIN —
    # expected_n_contexts, never the artifact-declared value.
    if len(questions) < n_contexts:
        failures.append(
            {
                "assert": "a",
                "field": "questions_length",
                "generated": len(questions),
                "expected_at_least": n_contexts,
            }
        )

    # Per-record checks per bucket (asserts (a)/(c)/(d)), with cross-bucket
    # duplicate detection and one FLUSHED progress line per bucket.
    t0 = time.monotonic()
    seen_pairs: set[tuple[int, int]] = set()
    buckets = sorted(by_persona.items())
    for idx, (p, recs) in enumerate(buckets, start=1):
        p_failures, p_seen = verify_persona(p, recs, questions, n_contexts)
        failures.extend(p_failures)
        for i, pi in sorted(seen_pairs & p_seen)[:20]:
            failures.append(
                {
                    "assert": "c",
                    "kind": "duplicate_pair_cross_bucket",
                    "context_id": i,
                    "persona_idx": pi,
                }
            )
        seen_pairs |= p_seen
        n_recs = len(recs) if isinstance(recs, list) else 0
        print(
            f"[p0] persona {idx}/{len(buckets)} p={p:02d} records={n_recs} "
            f"elapsed={time.monotonic() - t0:.1f}s",
            flush=True,
        )

    # (c) domain: EXACT set equality against the INDEPENDENTLY-derived
    # expected pair set — persona(i, k) = i mod k over range(n_contexts),
    # where n_contexts is the CALLER-pinned authority (round-5 BLOCKER 1).
    expected_pairs = {(i, i % k) for k in K_ARMS for i in range(n_contexts)}
    missing = sorted(expected_pairs - seen_pairs)
    extra = sorted(seen_pairs - expected_pairs)
    for i, pi in missing[:20]:
        failures.append({"assert": "c", "kind": "missing_pair", "context_id": i, "persona_idx": pi})
    if len(missing) > 20:
        failures.append(
            {"assert": "c", "kind": "missing_pairs_total", "n_missing_total": len(missing)}
        )
    for i, pi in extra[:20]:
        failures.append(
            {"assert": "c", "kind": "unexpected_pair", "context_id": i, "persona_idx": pi}
        )
    if len(extra) > 20:
        failures.append(
            {"assert": "c", "kind": "unexpected_pairs_total", "n_extra_total": len(extra)}
        )

    if not failures:
        n_records = sum(len(recs) for recs in by_persona.values())
        logger.info(
            "P0 prompt-integrity gate PASS: %d records over %d expected pairs, "
            "asserts (a)-(d) clean",
            n_records,
            len(seen_pairs),
        )
        return
    report = {
        "reason": "p0_prompt_integrity_failure",
        "n_failures_total": len(failures),
        "failures_by_assert": dict(Counter(f["assert"] for f in failures)),
        "failures_first50": failures[:50],
        "detected_at": _utc_now(),
    }
    write_json(report_dir / "p0_integrity_report.json", report)
    logger.error(
        "HALT-AND-REPORT: P0 prompt-integrity gate FAILED (%d failure(s), by assert: %s); "
        "report at %s. A mismatched stimulus must never enter mask construction or the "
        "expected-refusal class -- refusing to proceed.",
        len(failures),
        report["failures_by_assert"],
        report_dir / "p0_integrity_report.json",
    )
    sys.exit(EXIT_P0_INTEGRITY)


def run_p0_verify(
    stage_dir: pathlib.Path,
    report_dir: pathlib.Path,
    dl_dir: pathlib.Path,
    expected_n_contexts: int,
) -> None:
    """Standalone P0 gate over already-persisted P-Gen artifacts (``--p0-verify``).

    The pod-side P0 phase stages the P-Gen artifacts from HF and runs this
    exact gate BEFORE mask construction; any mismatch is the designed halt
    (report JSON + exit 5) rather than a corrupted-stimulus headline. The
    question AUTHORITY is re-downloaded at ``PARENT_REV`` via
    :func:`load_frozen_questions` (the same pinned loader P-Gen dispatched
    from), so assert (a) compares against the bundle, never against the
    records themselves.

    Domain authority (round-5 BLOCKER 1): ``expected_n_contexts`` is pinned
    by the CALLER (``main``'s ``--p0-verify`` branch: ``N_CONTEXTS_FULL`` in
    production, the explicit smoke count under ``--smoke``) — the question
    authority is sliced by THIS value, never by the artifact-declared
    ``assignment.json`` ``n_contexts``, and the gate records any
    declared-vs-pinned mismatch as a (b)-class rc-5 failure. A coherent
    smaller artifact (e.g. a 100-context assignment with exactly matching
    records) therefore can no longer certify 2% of the registered domain
    as a production PASS.

    Restartability (round-4 BLOCKER 4): per-persona completion is DURABLE —
    ``report_dir / "p0_verify_progress.json"`` records, per persona file,
    its sha256 + verification outcome, keyed on the generation-config
    fingerprint (from the persisted ``_gen_complete.json`` sentinel when
    present, else the live fingerprint) AND on
    ``P0_VERIFIER_SCHEMA_VERSION`` — the VERIFIER's own record-validation
    contract (round-6 BLOCKER 1: a checkpoint a superseded verifier wrote
    is discarded whole, never reused). A re-run reuses byte-identical,
    previously-CLEAN persona files (content-addressed by sha256) and
    re-verifies everything else; each persona flushes its checkpoint entry
    the moment it completes (atomic tmp+rename), and the gate emits one
    FLUSHED ``[p0] persona k/16 ...`` progress line per bucket.
    """
    if type(expected_n_contexts) is not int or expected_n_contexts <= 0:
        raise ValueError(f"expected_n_contexts must be a positive int, got {expected_n_contexts!r}")
    roster_obj = json.loads((stage_dir / "roster.json").read_text())
    assignment_obj = json.loads((stage_dir / "assignment.json").read_text())
    questions_full, _, _ = load_frozen_questions(dl_dir)
    # Round-5 BLOCKER 1: slice the question AUTHORITY by the CALLER-pinned
    # domain — never by the artifact-declared n_contexts (the artifact under
    # test must not define the authority it is verified against).
    questions = questions_full[:expected_n_contexts]

    fp = None
    sentinel_path = stage_dir / "_gen_complete.json"
    if sentinel_path.is_file():
        sent = json.loads(sentinel_path.read_text())
        fp = (sent.get("generation_config_fingerprint") or {}).get("sha256")
    if not fp:
        fp = generation_config_fingerprint()["sha256"]

    ckpt_path = report_dir / "p0_verify_progress.json"
    ckpt: dict = {
        "fingerprint": fp,
        "verifier_schema_version": P0_VERIFIER_SCHEMA_VERSION,
        "personas": {},
    }
    if ckpt_path.is_file():
        prior = json.loads(ckpt_path.read_text())
        # Round-6 BLOCKER 1: reuse requires the GENERATION fingerprint AND the
        # P0-VERIFIER schema version to match — a checkpoint written by an
        # older verifier (round-4 isinstance-int schema: n_failures 0 for a
        # `context_id: true` record) would otherwise bypass the round-5
        # type-is-int fix via the per-persona reuse branch. A missing or
        # differing key is treated exactly like a fingerprint mismatch:
        # discard the whole checkpoint, re-verify from scratch.
        if (
            prior.get("fingerprint") == fp
            and prior.get("verifier_schema_version") == P0_VERIFIER_SCHEMA_VERSION
            and isinstance(prior.get("personas"), dict)
        ):
            ckpt = prior
        else:
            logger.warning(
                "P0 verify checkpoint at %s keyed to a DIFFERENT generation fingerprint "
                "or P0-verifier schema (fingerprint %s != %s, or verifier_schema_version "
                "%r != %r) — restarting verification fresh",
                ckpt_path,
                prior.get("fingerprint"),
                fp,
                prior.get("verifier_schema_version"),
                P0_VERIFIER_SCHEMA_VERSION,
            )

    by_persona: dict[int, list[dict]] = {}
    file_sha: dict[int, str] = {}
    for p in range(N_PERSONAS):
        f = stage_dir / f"persona{p:02d}_seed42.json"
        assert f.is_file(), f"--p0-verify: missing persisted persona file {f}"
        file_sha[p] = _sha256_file(f)
        payload = json.loads(f.read_text())
        # Round-5 BLOCKER 3: a non-dict persona-file root or a missing
        # "records" container becomes a non-list bucket the gate RECORDS as
        # a "c"/malformed_bucket failure on the rc-5 report path — never a
        # raw KeyError/TypeError at ["records"].
        by_persona[p] = payload.get("records") if isinstance(payload, dict) else None

    def cached_verify(
        p: int, recs: list, qs: list[str], n_contexts: int
    ) -> tuple[list[dict], set[tuple[int, int]]]:
        ent = ckpt["personas"].get(str(p))
        if ent and ent.get("file_sha256") == file_sha[p] and ent.get("n_failures") == 0:
            logger.info(
                "P0 persona %02d: checkpoint-clean (file sha256 match) — reusing prior verify", p
            )
            return [], {(i, p) for i in ent["context_ids"]}
        failures, seen = p0_verify_persona_records(p, recs, qs, n_contexts)
        ckpt["personas"][str(p)] = {
            "file_sha256": file_sha[p],
            "n_records": len(recs) if isinstance(recs, list) else 0,
            "n_failures": len(failures),
            "context_ids": sorted(i for i, _ in seen),
            "verified_at": _utc_now(),
        }
        write_json(ckpt_path, ckpt)  # durable per-persona flush (atomic tmp+rename)
        return failures, seen

    p0_prompt_integrity_gate(
        roster_obj,
        assignment_obj,
        by_persona,
        report_dir,
        questions,
        expected_n_contexts=expected_n_contexts,
        verify_persona=cached_verify,
    )
    logger.info("P0 prompt-integrity verify PASS over %s", stage_dir)


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
    parser.add_argument(
        "--p0-verify",
        action="store_true",
        help=(
            "run ONLY the P0 prompt-integrity gate (plan section 4.3 P0 asserts (a)-(d)) "
            "over already-persisted P-Gen artifacts, then exit (0 pass / 5 halt)"
        ),
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

    if args.p0_verify:
        # Round-5 BLOCKER 1: the verifier's domain authority is the value
        # THIS caller just pinned above — N_CONTEXTS_FULL (5000) in
        # production, the explicit --n-contexts smoke count under --smoke —
        # never the artifact-declared assignment.json n_contexts.
        run_p0_verify(stage_dir, eval_dir, root / "parent_inputs", n_contexts)
        return

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
    # Section 4.3 step 3b(iii): per-row caps read from the serving
    # checkpoint's PERSISTED request metadata (gen_config.json, written or
    # verified by _dispatch's fingerprint gate), never the live constant --
    # and every row starts on the explicit first-wave label (step 2).
    base_cap = checkpoint_max_tokens(root / "batches")
    max_tokens_by_item = {it.item_id: base_cap for it in items}
    gen_wave_by_item = {it.item_id: GEN_WAVE_FIRST for it in items}
    items_by_id = {it.item_id: it for it in items}
    # FIX B: merge prior runs' redrive checkpoint outcomes BEFORE computing
    # the pending set — already-paid successes are never re-submitted, and a
    # crash-orphaned batch still completing server-side is harvested instead
    # of double-billed by a blind re-submit of the same rows.
    _merge_stale_redrives(root, items_by_id, results, batch_meta, max_tokens_by_item, poll_interval)
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
        # Step 3b(iii): redriven rows' caps from the redrive checkpoint's
        # persisted metadata.
        rd_cap = checkpoint_max_tokens(root / f"redrive{rnd}")
        for iid in pending:
            max_tokens_by_item[iid] = rd_cap
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

    # 4. Pre-registered cap-hit re-gen trigger PER (arm x persona) CELL
    # (plan v13 section 4.3 step 4 + section 7 — the v10 per-persona form is
    # superseded); the UNION of over-cap pair rows across ALL triggered cells
    # dispatches POOLED into ONE batch (FIX D) — the pre-fix per-persona loop
    # serialized one full Batch API round-trip per triggered persona.
    stop_by_item = {iid: res.stop_reason for iid, res in results.items()}
    triggered_cells, regen_pairs = cells_over_cap_threshold(stop_by_item, assignment)
    regen_items: set[str] = set()
    if regen_pairs:
        rg, pooled_ids, rg_dir = _dispatch_pooled_regen(
            root, items_by_id, regen_pairs, poll_interval
        )
        rg_remaining = transport_class_ids(rg)
        if rg_remaining:
            report = {
                "metadata": metadata,
                "incomplete": True,
                "reason": "transport_class_rows_remaining_in_regen",
                "regen_cells_triggered": triggered_cells,
                "regen_pairs_by_persona": {p: len(rows) for p, rows in sorted(regen_pairs.items())},
                "n_remaining": len(rg_remaining),
                "remaining_ids_first20": rg_remaining[:20],
                "regen_checkpoint_dir": str(rg_dir),
                "redrive_rounds_used": redrive_rounds,
            }
            write_json(eval_dir / "gen_digest.json", report)
            logger.error(
                "HALT-AND-REPORT: pooled cap-hit regen has %d transport-class rows "
                "(triggered cells: %s); digest at %s. A plain re-run REPLAYS this "
                "regen checkpoint and re-serves the same transport rows without "
                "re-dispatching (deterministic re-halt). Real remedies: quarantine the "
                "single pooled checkpoint (mv %s %s) so the re-run re-submits the "
                "pooled regen rows fresh, and/or investigate the transport cause (org "
                "rate limits, Anthropic batch console) before re-running.",
                len(rg_remaining),
                sorted(triggered_cells),
                eval_dir / "gen_digest.json",
                rg_dir,
                rg_dir.with_name(rg_dir.name + ".stale"),
            )
            sys.exit(3)
        results.update(rg)
        batch_meta.update(load_batch_meta(rg_dir))
        # Step 3b(iii) + step 2: the regen rows' cap comes from the regen
        # checkpoint's PERSISTED request metadata, and gen_wave is stamped
        # EXPLICITLY at regen time (never reverse-engineered from batch ids).
        regen_cap = checkpoint_max_tokens(rg_dir)
        for iid in pooled_ids:
            max_tokens_by_item[iid] = regen_cap
            gen_wave_by_item[iid] = GEN_WAVE_REGEN
            regen_items.add(iid)

    # 5. Persist per-persona records + assignment + roster + digest + sentinel.
    by_persona = build_records(
        questions,
        in_common,
        pairs,
        assignment,
        results,
        batch_meta,
        items_by_id,
        max_tokens_by_item,
        gen_wave_by_item,
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

    # P0 prompt-integrity gate (asserts (a)-(d)) at GEN time too, so a
    # serializer/roster/assignment defect halts BEFORE upload; the pod-side
    # P0 phase re-runs the same gate over the staged artifacts (--p0-verify)
    # before mask construction.
    p0_prompt_integrity_gate(
        roster_obj,
        assignment_obj,
        by_persona,
        eval_dir,
        questions,
        expected_n_contexts=n_contexts,
    )

    digest = build_digest(
        n_contexts,
        pairs,
        assignment,
        by_persona,
        mask_crosscheck,
        redrive_rounds,
        triggered_cells,
        regen_pairs,
        metadata,
    )
    write_json(eval_dir / "gen_digest.json", digest)
    write_json(eval_dir / "assignment.json", assignment_obj)
    write_json(eval_dir / "roster.json", roster_obj)

    sentinel = {
        "phase": "p_gen",
        "complete": True,
        "metadata": metadata,
        # Section 4.3 step 3b(i): the generation-config fingerprint rides the
        # sentinel too (beside the per-checkpoint gen_config.json copies).
        "generation_config_fingerprint": generation_config_fingerprint(),
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
        "Digest: %s | ok=%d/%d error=%d redrives=%d regen_cells=%s",
        eval_dir / "gen_digest.json",
        sentinel["n_ok"],
        len(pairs),
        digest["n_error_rows"],
        redrive_rounds,
        sorted(triggered_cells),
    )


if __name__ == "__main__":
    main()
