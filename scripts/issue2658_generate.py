"""Issue #2658 unit 3 — pilot answer generation (vLLM, frozen decoder, retain-all).

Plan §5 recipe: frozen model/tokenizer/chat-template pins (``issue2658_common``),
bf16, batched ``LLM.generate()`` (never sequential HF), no tools / steering /
hooks, temperature 1.0, top_p 0.95, split-aware max_new_tokens (pilot 1024 for
#779 parity, dev/test read the amended production cap from the frozen
``power_inputs/cap_amendment.json`` record, plan v6 A7, never a bare default),
SHA-derived per-request seeds from the unit-1 schedule, immutable prompt /
system-message / order manifests.  EVERY answer is retained — no filtering, no
selection, no exclusion (``assert_iid_generation`` audits each prompt).

Cap-hit accounting: the realized ``finish_reason == "length"`` fraction is
reported per row AND per (frame, stratum) cell; any cell strictly above 2%
writes a pre-test cap-AMENDMENT artifact — never selective regeneration.

Zero-token outputs follow the fixed three-retry seed schedule
(``empty_retry_seed``); persistent empty output after three retries FAILS the
run loud (plan §5: never changes the feature definition).  The manifest ``seed``
field stays the draw-slot schedule seed (the unit-1 validator pins it); the
realized generating seed + retry ledger ride the raw completion record (plan
§9 persists exclusion/retry ledgers).

Checkpoint grain: one atomic JSON per (row, frame, stratum) cell (132 cells >
the ~50-unit floor), fingerprint-gated resume, one progress line per cell.
Sharding: ``--num-shards/--shard-index`` partitions the sorted cell list; the
dispatcher pins one GPU per shard via launcher-env ``CUDA_VISIBLE_DEVICES``.

Terminal: ``os._exit(0)`` after flush when an engine was constructed (vLLM
worker children survive interpreter finalization otherwise — gotchas #1739).

CONTENT HYGIENE: prompt/answer text flows resolver -> memory -> engine -> raw
completion files; logs and manifests carry only ids, counts, and sha256s.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

# vLLM reads this at import time — set BEFORE any vllm import (#628 fork trap).
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()  # #847 thread caps + HF token, before numpy/torch/vllm

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import issue2658_common as C  # noqa: E402
import issue2658_frames as F  # noqa: E402
import issue2658_text_resolver as R  # noqa: E402
from explore_persona_space.atomic_io import (  # noqa: E402
    write_jsonl_atomic,
    write_text_atomic,
)

# #1092 / #1739 / #2388 rig parity (the #779-lineage behavior-corpus window).
MAX_MODEL_LEN = 8192
# PILOT prompt budget (7168). Kept as a module constant because the pilot
# capture rig (issue2658_capture.py) keys its formatted-length guard and
# fingerprint on it; split-aware code paths use prompt_budget_for_cap().
PROMPT_BUDGET = MAX_MODEL_LEN - int(C.DECODER["max_new_tokens"])  # 7168
# Frozen length-cap amendment record (plan v6 A7), relative to the eval root.
CAP_AMENDMENT_REL = "power_inputs/cap_amendment.json"
# Frozen record schema id. Single source of truth: issue2658_power aliases
# this constant (round 14, review r13 minor 2).
CAP_AMENDMENT_SCHEMA = "i2658-cap-amendment-v1"
CHUNK_SIZE = int(os.environ.get("EPM_VLLM_GREEDY_CHUNK_SIZE", "500"))
# Frozen production prompt selection (plan v4 section 4; round 15). Single
# source of truth for the schema id + path: issue2658_production_selection.py
# (the writer) imports BOTH from here, so writer and reader cannot fork.
SELECTION_REL = "production_selection.json"
SELECTION_SCHEMA = "i2658-production-selection-v1"
N_EMPTY_RETRIES = 3
CAP_HIT_AMEND_THRESHOLD = 0.02  # strictly-above trigger (plan §5)
EXPERIMENT_NAME = "issue2658_dirvalid"  # HF data-repo prefix (unit-2 convention)
GEN_SCHEMA = "i2658-gen-cell-v1"


class EmptyOutputError(C.Issue2658GuardError):
    """A prompt draw stayed zero-token through the fixed three-retry schedule."""


class GenerationBudgetError(C.Issue2658GuardError):
    """A resolved prompt exceeds the frozen prompt token budget (loud, no skip)."""


class ProductionSelectionRecordError(C.Issue2658GuardError):
    """Frozen production selection absent, drifted, or disagreeing with the
    live frozen artifacts (always fail loud, never a bare default)."""


class OrderManifestDriftError(C.Issue2658GuardError):
    """An immutable order manifest already exists with different content."""


# ---------------------------------------------------------------------------
# Split-aware decoder cap (plan v6 A7).
# ---------------------------------------------------------------------------
def validate_cap_amendment_values(
    body: dict[str, Any],
    path: Path,
    err_cls: type[Exception] | None = None,
    *,
    require_2x_floor: bool = True,
) -> None:
    """VALUE validation for a loaded cap-amendment record (plan v6 A7).

    One source of truth shared by this module's :func:`load_cap_amendment`
    and ``issue2658_power.load_cap_amendment_record`` (power.py already
    imports this module, so the shared checks live here): schema id
    EQUALITY, both caps positive ints, the registered production >= 2x
    pilot floor, and a non-empty offender mapping. ``require_2x_floor=False``
    is for the power-side loader, where a below-floor record is reported as
    a GATE_FAIL verdict by ``_gate_cap_hit``, never a loader raise.
    """
    err = err_cls or GenerationBudgetError
    if body.get("schema") != CAP_AMENDMENT_SCHEMA:
        raise err(
            f"{path}: cap amendment schema {body.get('schema')!r} != {CAP_AMENDMENT_SCHEMA!r}"
        )
    caps: dict[str, int] = {}
    for fld in ("pilot_max_new_tokens", "production_max_new_tokens"):
        val = body.get(fld)
        if isinstance(val, bool) or not isinstance(val, int) or val <= 0:
            raise err(f"{path}: {fld} must be a positive int, got {val!r}")
        caps[fld] = val
    if require_2x_floor and caps["production_max_new_tokens"] < 2 * caps["pilot_max_new_tokens"]:
        raise err(
            f"{path}: production cap {caps['production_max_new_tokens']} < 2x pilot cap "
            f"{caps['pilot_max_new_tokens']} (plan v6 A7 registered floor)"
        )
    offenders = body.get("cells_over_threshold")
    if not isinstance(offenders, dict) or not offenders:
        raise err(
            f"{path}: cells_over_threshold must be a non-empty mapping, got {offenders!r} "
            "(an amendment with no offenders is a wiring error)"
        )


def load_cap_amendment(eval_root: Path | None = None) -> dict[str, Any] | None:
    """The frozen length-cap amendment record (plan v6 A7), or None if absent.

    ``eval_root`` defaults to the canonical committed eval root ``F.OUT_DIR``
    (the record is frozen and committed, so pod clones carry it after sync).
    """
    root = Path(eval_root) if eval_root is not None else F.OUT_DIR
    path = root / CAP_AMENDMENT_REL
    if not path.exists():
        return None
    body = json.loads(path.read_text())
    for fld in ("schema", "plan_version", "pilot_max_new_tokens", "production_max_new_tokens"):
        if fld not in body:
            raise GenerationBudgetError(f"{path}: cap amendment record missing field {fld!r}")
    # Round 14 (review r13 minor 2): values, not just presence — a future
    # schema-v2 record with renamed semantics must never reach the decoder.
    validate_cap_amendment_values(body, path)
    return body


def resolve_max_new_tokens(split: str, eval_root: Path | None = None) -> int:
    """Split-aware decoder cap: pilot keeps the frozen 1024, dev/test read the
    amended production cap from the frozen record (plan v6 A7).

    There is deliberately NO bare default for a non-pilot split: an absent
    record raises, so production generation can never silently run at the
    pilot cap (or at any hand-typed value).
    """
    if split not in C.SPLITS:
        raise ValueError(f"unknown split {split!r}; registered splits: {C.SPLITS}")
    if split == "pilot":
        return int(C.DECODER["max_new_tokens"])
    rec = load_cap_amendment(eval_root)
    if rec is None:
        root = Path(eval_root) if eval_root is not None else F.OUT_DIR
        raise GenerationBudgetError(
            f"split {split!r} requires the frozen cap amendment record at "
            f"{root / CAP_AMENDMENT_REL} (plan v6 A7): produce it with "
            "'scripts/issue2658_power.py --phase cap-amendment', never a bare default"
        )
    return int(rec["production_max_new_tokens"])


def prompt_budget_for_cap(max_new_tokens: int) -> int:
    """Prompt token budget under a given decoder cap (the frozen plan section 5
    assertion keeps failing loud through :func:`rendered_prompt_or_raise`)."""
    budget = MAX_MODEL_LEN - int(max_new_tokens)
    if budget <= 0:
        raise GenerationBudgetError(
            f"max_new_tokens {max_new_tokens} leaves no prompt budget under "
            f"max_model_len {MAX_MODEL_LEN}"
        )
    return budget


# ---------------------------------------------------------------------------
# Pure helpers (unit-tested offline).
# ---------------------------------------------------------------------------
def empty_retry_seed(prompt_id: str, response_index: int, attempt: int) -> int:
    """Fixed pre-registered retry seed for a zero-token draw (attempt 1..3)."""
    if not (1 <= attempt <= N_EMPTY_RETRIES):
        raise ValueError(f"attempt must be in 1..{N_EMPTY_RETRIES}, got {attempt}")
    digest = hashlib.sha256(
        f"i2658-gen-empty-retry|{prompt_id}|{response_index}|{attempt}".encode()
    ).digest()
    return int.from_bytes(digest[:8], "big") % (2**31)


def generate_with_empty_retry(
    gen_once: Callable[[int], dict[str, Any]],
    prompt_id: str,
    response_index: int,
) -> tuple[dict[str, Any], int, list[dict[str, Any]]]:
    """Run one draw; on a zero-token output walk the fixed retry schedule.

    ``gen_once(seed)`` returns ``{"text", "token_ids", "finish_reason"}``.
    Returns ``(output, realized_seed, ledger_rows)``; exhaustion RAISES
    ``EmptyOutputError`` (persistent empty fails the row/bank — plan §5).
    """
    schedule_seed = C.response_seed(prompt_id, response_index)
    out = gen_once(schedule_seed)
    ledger: list[dict[str, Any]] = []
    if len(out["token_ids"]) > 0:
        return out, schedule_seed, ledger
    for attempt in range(1, N_EMPTY_RETRIES + 1):
        seed = empty_retry_seed(prompt_id, response_index, attempt)
        out = gen_once(seed)
        ledger.append(
            {
                "prompt_id": prompt_id,
                "response_index": response_index,
                "attempt": attempt,
                "retry_seed": seed,
                "outcome": "nonempty" if len(out["token_ids"]) > 0 else "empty",
            }
        )
        if len(out["token_ids"]) > 0:
            return out, seed, ledger
    raise EmptyOutputError(
        f"draw ({prompt_id!r}, k={response_index}) stayed zero-token through the "
        f"schedule seed + {N_EMPTY_RETRIES} fixed retries; plan §5 fails the row/bank"
    )


def cap_hit_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Per-cell + per-row length-cap-hit fractions and the amendment verdict.

    ``rows`` carry ``row``, ``cell``, ``finish_reason``. A cell strictly above
    ``CAP_HIT_AMEND_THRESHOLD`` arms ``amendment_required`` (plan §5: pre-test
    cap amendment, never selective regeneration).
    """
    if not rows:
        raise ValueError("cap_hit_report over zero records")
    per_cell_n: dict[str, int] = {}
    per_cell_hit: dict[str, int] = {}
    per_row_n: dict[str, int] = {}
    per_row_hit: dict[str, int] = {}
    for r in rows:
        key = f"{r['row']}|{r['cell']}"
        hit = 1 if r["finish_reason"] == "length" else 0
        per_cell_n[key] = per_cell_n.get(key, 0) + 1
        per_cell_hit[key] = per_cell_hit.get(key, 0) + hit
        per_row_n[r["row"]] = per_row_n.get(r["row"], 0) + 1
        per_row_hit[r["row"]] = per_row_hit.get(r["row"], 0) + hit
    per_cell = {k: per_cell_hit[k] / per_cell_n[k] for k in sorted(per_cell_n)}
    per_row = {k: per_row_hit[k] / per_row_n[k] for k in sorted(per_row_n)}
    offenders = {k: v for k, v in per_cell.items() if v > CAP_HIT_AMEND_THRESHOLD}
    return {
        "threshold": CAP_HIT_AMEND_THRESHOLD,
        "n_records": len(rows),
        "per_cell_fraction": per_cell,
        "per_cell_n": {k: per_cell_n[k] for k in sorted(per_cell_n)},
        "per_row_fraction": per_row,
        "amendment_required": bool(offenders),
        "cells_over_threshold": offenders,
    }


def build_manifest_row(
    *,
    row: str,
    item_id: str,
    superfamily_id: str,
    frame: str,
    band: str,
    split: str,
    response_index: int,
    answer_sha256: str,
    raw_text_sha256: str,
    prompt_sha256: str | None = None,
) -> dict[str, Any]:
    """One TEXT-FREE generation manifest row; validated against the unit-1 schema.

    ``prompt_sha256`` None keeps the pilot path byte-identical (the frozen pilot
    pin table via ``_pin_sha``); dev/test callers pass the RESOLVED text sha,
    already verified against the frozen production selection (round 15)."""
    construct = C.CONSTRUCTS[row]
    judge_scored = construct.judge_scored
    d = {
        "manifest_version": C.MANIFEST_VERSION,
        "row": row,
        "split": split,
        "prompt_id": item_id,
        "prompt_sha256": prompt_sha256 if prompt_sha256 is not None else _pin_sha(item_id),
        "superfamily_id": superfamily_id,
        "source_frame": frame,
        "stratum": band,
        "model_id": C.MODEL_ID,
        "model_revision": C.MODEL_REVISION,
        "chat_template_sha256": C.CHAT_TEMPLATE_SHA256,
        "response_index": response_index,
        "seed": C.response_seed(item_id, response_index),
        "answer_sha256": answer_sha256,
        "raw_text_sha256": raw_text_sha256,
        "evidence_sha256": None,  # judge-time artifact; the model never sees evidence
        "judge_status": "pending" if judge_scored else "objective",
        "judge_draw_ids": list(C.judge_draw_ids(answer_sha256)) if judge_scored else [],
        "judge_model": None,
        "vector_sha256": None,  # set by the L19 capture (issue2658_capture.py)
    }
    C.validate_manifest_row(d)
    return d


_PIN_CACHE: dict[str, str] | None = None


def _pin_sha(item_id: str) -> str:
    global _PIN_CACHE
    if _PIN_CACHE is None:
        _PIN_CACHE = {k: v["prompt_sha256"] for k, v in R.load_pins()["items"].items()}
    sha = _PIN_CACHE.get(item_id)
    if sha is None:
        raise R.TextResolutionError(f"no frozen prompt pin for {item_id!r}")
    return sha


def canonical_json(body: Any) -> str:
    return json.dumps(body, sort_keys=True, indent=1, ensure_ascii=False) + "\n"


def write_immutable_json(path: Path, body: dict[str, Any]) -> None:
    """Write once; an existing file must byte-match the new content or RAISE."""
    payload = canonical_json(body)
    if path.exists():
        if path.read_text() != payload:
            raise OrderManifestDriftError(
                f"immutable manifest drift at {path}: existing content differs from "
                "the recomputed body"
            )
        return
    write_text_atomic(path, payload)


# ---------------------------------------------------------------------------
# Frozen production selection (dev/test prompt membership; plan v4 section 4).
# ---------------------------------------------------------------------------
def load_production_selection(split: str, eval_root: Path | None = None) -> dict[str, Any]:
    """The frozen, content-addressed production selection for ``split``.

    Fails loud when the file is absent, when its content sha drifts, or when
    its header disagrees with the LIVE frozen artifacts (n_common from
    ``power/production_n.json``, frame/split manifest content shas, the
    production responses-per-prompt config, the amended decoder cap).
    """
    if split not in ("dev", "test"):
        raise ValueError(f"production selection is dev/test only, got {split!r}")
    root = Path(eval_root) if eval_root is not None else F.OUT_DIR
    path = root / SELECTION_REL
    if not path.is_file():
        raise ProductionSelectionRecordError(
            f"split {split!r} requires the frozen production selection at {path}: freeze it "
            "with 'scripts/issue2658_production_selection.py --freeze-production-selection', "
            "never a bare default"
        )
    body = json.loads(path.read_text())
    if body.get("schema") != SELECTION_SCHEMA:
        raise ProductionSelectionRecordError(
            f"{path}: schema {body.get('schema')!r} != {SELECTION_SCHEMA!r}"
        )
    addressable = {k: v for k, v in body.items() if k not in ("metadata", "content_sha256")}
    got = F._canonical_sha(addressable)
    if got != body.get("content_sha256"):
        raise ProductionSelectionRecordError(
            f"{path}: content drift: recomputed {got} != stored {body.get('content_sha256')}"
        )
    prod_n = json.loads((root / "power" / "production_n.json").read_text())
    if body["n_common"] != prod_n.get("n_common"):
        raise ProductionSelectionRecordError(
            f"{path}: n_common {body['n_common']} != live power/production_n.json "
            f"{prod_n.get('n_common')}"
        )
    for name, live_path in (
        ("frame_manifest_content_sha256", root / "frame_manifest.json"),
        ("split_manifest_content_sha256", root / "split_manifest.json"),
    ):
        live = json.loads(live_path.read_text())["content_sha256"]
        if body[name] != live:
            raise ProductionSelectionRecordError(
                f"{path}: {name} {body[name]} != live {live_path.name} {live} "
                "(the selection was frozen against a different manifest)"
            )
    want_responses = int(C.DECODER["n_responses_per_prompt_production"])
    if int(body["responses_per_prompt"]) != want_responses:
        raise ProductionSelectionRecordError(
            f"{path}: responses_per_prompt {body['responses_per_prompt']} != frozen "
            f"config {want_responses}"
        )
    cap = resolve_max_new_tokens(split, root)
    if int(body["production_max_new_tokens"]) != cap:
        raise ProductionSelectionRecordError(
            f"{path}: production_max_new_tokens {body['production_max_new_tokens']} != "
            f"amended cap {cap} (plan v6 A7)"
        )
    if int(body["production_prompt_budget"]) != prompt_budget_for_cap(cap):
        raise ProductionSelectionRecordError(
            f"{path}: production_prompt_budget {body['production_prompt_budget']} != "
            f"prompt_budget_for_cap({cap}) = {prompt_budget_for_cap(cap)}"
        )
    for s in ("dev", "test"):
        if s not in body.get("splits", {}):
            raise ProductionSelectionRecordError(f"{path}: split {s!r} missing from selection")
    return body


def production_item_triples(
    split: str, eval_root: Path | None = None
) -> list[tuple[str, str, str]]:
    """Ordered (row, cell, item_id) triples of the frozen ``split`` selection.

    The capture-side anchor for dev/test (the split-aware sibling of
    ``R.pilot_item_ids``); empty cells contribute nothing by construction.
    """
    body = load_production_selection(split, eval_root)
    triples: list[tuple[str, str, str]] = []
    seen: set[str] = set()
    for row in sorted(body["splits"][split]):
        cells = body["splits"][split][row]
        for cell in sorted(cells):
            for iid in cells[cell]["item_ids"]:
                if iid in seen:
                    raise ProductionSelectionRecordError(
                        f"item_id {iid!r} appears in two {split} cells"
                    )
                seen.add(iid)
                triples.append((row, cell, iid))
    if not triples:
        raise ProductionSelectionRecordError(f"frozen {split} selection yielded zero items")
    return triples


def verify_resolved_against_selection(
    resolved: dict[str, "R.ResolvedItem"], body: dict[str, Any], split: str
) -> None:
    """Resolved dev/test texts must match the frozen selection shas; drift RAISES.

    ``sha_kind`` "text" cells carry the prompt-text sha (bank/keyed loaders), so
    the resolved text is checked against it directly. "group-key" cells
    (correctness benchmarks) are id-addressed in the selection; their text
    integrity rides the sha-pinned vendored loaders instead. Every resolved item
    that ALSO carries a frozen pilot pin is additionally checked against the pin
    table (pilot-reused dev items keep the pilot integrity guarantee).
    """
    recs: dict[str, tuple[str, str]] = {}
    for row_cells in body["splits"][split].values():
        for rec in row_cells.values():
            kind = rec["sha_kind"]
            for iid, sha in rec["item_sha256"].items():
                recs[iid] = (sha, kind)
    foreign = [iid for iid in resolved if iid not in recs]
    if foreign:
        raise ProductionSelectionRecordError(
            f"{len(foreign)} resolved items are not in the frozen {split} selection "
            f"(e.g. {foreign[:3]})"
        )
    pins = R.load_pins()["items"]
    for iid, item in resolved.items():
        sha, kind = recs[iid]
        if kind == "text" and item.prompt_sha256 != sha:
            raise C.RowHashMismatchError(
                f"{iid}: resolved text sha {item.prompt_sha256} != frozen selection sha {sha}"
            )
        pin = pins.get(iid)
        if pin is not None:
            C.assert_row_hash(item.text, pin["prompt_sha256"])


def resolve_items_for_split(
    item_ids: list[str], split: str, *, eval_root: Path | None = None
) -> dict[str, "R.ResolvedItem"]:
    """ONE split-aware frozen prompt-text resolver for every consumer (round 18).

    Pilot items verify against the frozen pilot pin table
    (``R.resolve_items(..., verify_pins=True)``, byte-identical to the
    pre-round-18 call sites). Dev/test items resolve WITHOUT the pilot pin
    table (it covers the pilot selection only) and instead verify against the
    frozen production selection via ``verify_resolved_against_selection``
    (text-sha cells directly, pilot-reused items against the pin table too;
    round 15/16). Any other split value raises. Consumers: generation,
    capture (``attach_rendered_prompts``), judge (``load_cell_units``) and
    comparators (``assemble_row_data``) all route here, so a new split can
    never silently fall back to the pilot pin table again.
    """
    if split == "pilot":
        return R.resolve_items(item_ids, verify_pins=True)
    if split in ("dev", "test"):
        resolved = R.resolve_items(item_ids, verify_pins=False)
        verify_resolved_against_selection(
            resolved, load_production_selection(split, eval_root), split
        )
        return resolved
    raise ValueError(
        f"unknown split {split!r} for prompt-text resolution; expected pilot, dev or test"
    )


# ---------------------------------------------------------------------------
# Work-list construction.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CellWork:
    row: str
    frame: str
    band: str
    item_ids: tuple[str, ...]
    superfamilies: dict[str, str]

    @property
    def name(self) -> str:
        return f"{self.row}__{self.frame}__{self.band}"

    @property
    def cell(self) -> str:
        return f"{self.frame}|{self.band}"


def build_cells(rows_filter: list[str] | None = None, split: str = "pilot") -> list[CellWork]:
    """Deterministic cell list for one split, from frozen, immutability-checked
    records only. Pilot cells come from the frame manifest's ``pilot_selection``
    (unchanged); dev/test cells come from the frozen production selection
    (round 15, plan v4 section 4), which ``load_production_selection`` verifies
    against the live artifacts before any use. Cells recorded empty are skipped
    WITH a printed count: pilot shortfalls are recorded in
    ``pilot_selection.cells_below_pilot_floor`` by unit 2, production
    shortfalls in the selection's per-cell ``status``/``shortfall`` records."""
    if split not in C.SPLITS:
        raise ValueError(f"unknown split {split!r}; registered splits: {C.SPLITS}")
    body = json.loads(F.FRAME_MANIFEST_PATH.read_text())
    F.assert_manifest_immutable(body)
    cells: list[CellWork] = []
    n_empty = 0
    prod_sel = None if split == "pilot" else load_production_selection(split)
    for rr in body["rows"]:
        if rows_filter and rr["row"] not in rows_filter:
            continue
        if split == "pilot":
            sel = rr["pilot_selection"]["per_cell_item_ids"]
        else:
            row_cells = prod_sel["splits"][split].get(rr["row"])
            if row_cells is None:
                raise ProductionSelectionRecordError(
                    f"row {rr['row']!r} missing from the frozen {split} selection"
                )
            sel = {cell: rec["item_ids"] for cell, rec in row_cells.items()}
        for cell_key in sorted(sel):
            iids = tuple(sel[cell_key])
            if not iids:
                n_empty += 1
                continue
            frame, _, band = cell_key.partition("|")
            if not band:
                raise F.FrameManifestError(f"malformed {split} cell key {cell_key!r}")
            sfs = {iid: R.superfamily_of(body, rr["row"], iid) for iid in iids}
            cells.append(CellWork(rr["row"], frame, band, iids, sfs))
    if not cells:
        raise F.FrameManifestError(f"zero {split} cells selected (rows filter too narrow?)")
    if n_empty:
        print(
            f"[gen] {n_empty} registered {split} cells realized EMPTY (recorded in the "
            "frozen selection) — skipped with this disclosure",
            flush=True,
        )
    return cells


def generation_fingerprint(
    cell: CellWork, n_responses: int, split: str, max_new_tokens: int
) -> str:
    """Machine-stable resume fingerprint over GENERATING PARAMETERS (#1336 rule:
    never hash recomputed floats; every value here is a frozen pin or an int).

    ``max_new_tokens`` is the REALIZED split cap (plan v6 A7). Pilot callers
    pass the frozen 1024, so pilot fingerprints are byte-identical to the
    pre-amendment payload and the realized pilot cells keep resuming.
    """
    payload = json.dumps(
        {
            "schema": GEN_SCHEMA,
            "model_id": C.MODEL_ID,
            "model_revision": C.MODEL_REVISION,
            "chat_template_sha256": C.CHAT_TEMPLATE_SHA256,
            "decoder": {
                "temperature": C.DECODER["temperature"],
                "top_p": C.DECODER["top_p"],
                "max_new_tokens": int(max_new_tokens),
            },
            "max_model_len": MAX_MODEL_LEN,
            "split": split,
            "n_responses": n_responses,
            "cell": cell.name,
            "item_ids": list(cell.item_ids),
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def order_manifest_body(
    cells: list[CellWork], n_responses: int, split: str, shard_tag: str
) -> dict[str, Any]:
    """Immutable order manifest: the exact ordered request list, text-free."""
    requests = [
        (iid, k, C.response_seed(iid, k))
        for cw in cells
        for iid in cw.item_ids
        for k in range(n_responses)
    ]
    req_sha = hashlib.sha256(
        json.dumps(requests, sort_keys=False, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "issue": 2658,
        "split": split,
        "shard": shard_tag,
        "system_message": None,  # plan §5: single user turn, no system message
        "chat_template_sha256": C.CHAT_TEMPLATE_SHA256,
        "model_id": C.MODEL_ID,
        "model_revision": C.MODEL_REVISION,
        "n_requests": len(requests),
        "n_responses_per_prompt": n_responses,
        "cell_order": [cw.name for cw in cells],
        "requests_sha256": req_sha,
        "seed_scheme": C.DECODER["seed_scheme"],
    }


# ---------------------------------------------------------------------------
# Frozen-pin verification against the live hub files (pre-engine gate).
# ---------------------------------------------------------------------------
def verify_frozen_file_pins() -> None:
    """Chunked sha256 of the pinned-revision tokenizer/config files vs pins."""
    from huggingface_hub import hf_hub_download

    from explore_persona_space.orchestrate.hub import retry_transient

    expected = {
        "tokenizer_config.json": C.TOKENIZER_CONFIG_SHA256,
        "tokenizer.json": C.TOKENIZER_JSON_SHA256,
        "generation_config.json": C.GENERATION_CONFIG_SHA256,
        "config.json": C.MODEL_CONFIG_SHA256,
    }
    for fname, pin in expected.items():
        path = retry_transient(
            lambda fname=fname: hf_hub_download(C.MODEL_ID, fname, revision=C.MODEL_REVISION),
            what=f"hf_hub_download({fname})",
        )
        got = R._sha256_file(Path(path))
        if got != pin:
            raise C.RowHashMismatchError(
                f"{fname} sha {got} != frozen pin {pin} at revision {C.MODEL_REVISION}"
            )
    print("[gen] frozen tokenizer/config file pins verified (4/4)", flush=True)


def load_tokenizer():
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(C.MODEL_ID, revision=C.MODEL_REVISION)
    R.chat_template_guard(tok)
    return tok


def rendered_token_count(tok, text: str) -> tuple[str, int]:
    """Chat-render one prompt text and count its tokens (the budget quantity).

    Single source for the generation-side budget assert below AND the
    production-selection length gate (issue2658_production_selection.py) —
    the two must count identically or a frozen selection could still crash
    generation at the budget check (round 15)."""
    rendered = R.render_user_prompt(tok, text)
    return rendered, len(tok.encode(rendered, add_special_tokens=False))


def rendered_prompt_or_raise(tok, item: R.ResolvedItem, max_new_tokens: int) -> tuple[str, int]:
    """Chat-render one prompt and enforce the split's frozen prompt budget (loud)."""
    budget = prompt_budget_for_cap(max_new_tokens)
    rendered, n_tok = rendered_token_count(tok, item.text)
    if n_tok > budget:
        raise GenerationBudgetError(
            f"prompt {item.item_id!r} renders to {n_tok} tokens > budget {budget} "
            f"(max_model_len {MAX_MODEL_LEN} - max_new_tokens "
            f"{int(max_new_tokens)}); plan §5 fails loud, never skips"
        )
    return rendered, n_tok


# ---------------------------------------------------------------------------
# Engine.
# ---------------------------------------------------------------------------
def build_engine(tensor_parallel: int):
    """One vLLM engine at the frozen pins. Honors the #1324/#1092 hang/IMA
    mitigation env knobs at this (the only) ``LLM(`` site."""
    from vllm import LLM

    kwargs: dict[str, Any] = {}
    if os.environ.get("EPM_VLLM_ENFORCE_EAGER") == "1":
        kwargs["enforce_eager"] = True
    if os.environ.get("EPM_VLLM_DISABLE_PREFIX_CACHING") == "1":
        kwargs["enable_prefix_caching"] = False
    return LLM(
        model=C.MODEL_ID,
        revision=C.MODEL_REVISION,
        tokenizer_revision=C.MODEL_REVISION,
        dtype="bfloat16",
        max_model_len=MAX_MODEL_LEN,
        tensor_parallel_size=tensor_parallel,
        **kwargs,
    )


def _sampling_params(seed: int, max_new_tokens: int):
    from vllm import SamplingParams

    return SamplingParams(
        n=1,
        temperature=float(C.DECODER["temperature"]),
        top_p=float(C.DECODER["top_p"]),
        max_tokens=int(max_new_tokens),
        seed=seed,
    )


def generate_cell(
    llm,
    tok,
    cell: CellWork,
    resolved: dict[str, R.ResolvedItem],
    n_responses: int,
    split: str,
    max_new_tokens: int,
) -> dict[str, Any]:
    """Generate all draws for one cell (batched, chunked); returns the cell body."""
    rendered: dict[str, str] = {}
    n_prompt_tokens: dict[str, int] = {}
    for iid in cell.item_ids:
        rendered[iid], n_prompt_tokens[iid] = rendered_prompt_or_raise(
            tok, resolved[iid], max_new_tokens
        )

    plan = [(iid, k) for iid in cell.item_ids for k in range(n_responses)]
    prompts = [rendered[iid] for iid, _ in plan]
    params = [_sampling_params(C.response_seed(iid, k), max_new_tokens) for iid, k in plan]

    outputs: list[Any] = []
    for start in range(0, len(prompts), max(1, CHUNK_SIZE)):
        end = min(start + max(1, CHUNK_SIZE), len(prompts))
        print(
            f"[vllm-chunk] {cell.name} chunk {start // max(1, CHUNK_SIZE) + 1}/"
            f"{(len(prompts) + CHUNK_SIZE - 1) // max(1, CHUNK_SIZE)} "
            f"({end - start} prompts)",
            flush=True,
        )
        outputs.extend(llm.generate(prompts[start:end], params[start:end], use_tqdm=False))
    if len(outputs) != len(plan):
        raise RuntimeError(f"engine returned {len(outputs)} outputs for {len(plan)} requests")

    records: list[dict[str, Any]] = []
    retry_ledger: list[dict[str, Any]] = []
    for (iid, k), out in zip(plan, outputs, strict=True):
        comp = out.outputs[0]
        result = {
            "text": comp.text,
            "token_ids": list(comp.token_ids),
            "finish_reason": comp.finish_reason,
        }
        realized_seed = C.response_seed(iid, k)
        if len(result["token_ids"]) == 0:

            def gen_once(seed: int, prompt: str = rendered[iid]) -> dict[str, Any]:
                o = llm.generate(
                    [prompt], [_sampling_params(seed, max_new_tokens)], use_tqdm=False
                )[0].outputs[0]
                return {
                    "text": o.text,
                    "token_ids": list(o.token_ids),
                    "finish_reason": o.finish_reason,
                }

            result, realized_seed, ledger = generate_with_empty_retry(gen_once, iid, k)
            retry_ledger.extend(ledger)
        text = result["text"]
        sha = F._sha_text(text)
        records.append(
            {
                "prompt_id": iid,
                "response_index": k,
                "seed": C.response_seed(iid, k),
                "realized_seed": realized_seed,
                "n_empty_retries": sum(
                    1 for lr in retry_ledger if lr["prompt_id"] == iid and lr["response_index"] == k
                ),
                "finish_reason": result["finish_reason"],
                "n_prompt_tokens": n_prompt_tokens[iid],
                "n_completion_tokens": len(result["token_ids"]),
                # Retain-all (plan §5): the answer IS the raw text, verbatim.
                "answer_sha256": sha,
                "raw_text_sha256": sha,
                "text": text,
            }
        )

    # iid self-audit: every prompt's draw slots follow the frozen schedule.
    for iid in cell.item_ids:
        C.assert_iid_generation(
            {
                "prompt_id": iid,
                "seeds": [r["seed"] for r in records if r["prompt_id"] == iid],
                "n_planned": n_responses,
                "topped_up": False,
                "early_stopped": False,
                "excluded": False,
            }
        )

    from explore_persona_space.orchestrate.provenance import as_metadata_dict, git_provenance

    return {
        "schema": GEN_SCHEMA,
        "row": cell.row,
        "frame": cell.frame,
        "band": cell.band,
        "split": split,
        "fingerprint": generation_fingerprint(cell, n_responses, split, max_new_tokens),
        "model_id": C.MODEL_ID,
        "model_revision": C.MODEL_REVISION,
        "chat_template_sha256": C.CHAT_TEMPLATE_SHA256,
        "decoder": {
            "temperature": C.DECODER["temperature"],
            "top_p": C.DECODER["top_p"],
            "max_new_tokens": int(max_new_tokens),
        },
        "n_items": len(cell.item_ids),
        "n_responses_per_prompt": n_responses,
        "records": records,
        "retry_ledger": retry_ledger,
        "metadata": as_metadata_dict(git_provenance(), phase="gen"),
    }


def manifest_rows_for_cell(
    cell: CellWork, body: dict[str, Any], sha_by_item: dict[str, str] | None = None
) -> list[dict[str, Any]]:
    """Validated TEXT-FREE manifest rows for one generated cell body.

    ``sha_by_item`` (dev/test) carries the resolved-text sha per item; None
    keeps the pilot pin-table path byte-identical."""
    return [
        build_manifest_row(
            row=cell.row,
            item_id=r["prompt_id"],
            superfamily_id=cell.superfamilies[r["prompt_id"]],
            frame=cell.frame,
            band=cell.band,
            split=body["split"],
            response_index=r["response_index"],
            answer_sha256=r["answer_sha256"],
            raw_text_sha256=r["raw_text_sha256"],
            prompt_sha256=None if sha_by_item is None else sha_by_item[r["prompt_id"]],
        )
        for r in body["records"]
    ]


# ---------------------------------------------------------------------------
# Driver.
# ---------------------------------------------------------------------------
def out_paths(out_root: Path, split: str, cell_name: str) -> tuple[Path, Path]:
    raw = out_root / "raw_completions" / split / f"{cell_name}.json"
    man = out_root / "gen_manifest" / split / f"{cell_name}.jsonl"
    return raw, man


def load_resume_cell(raw_path: Path, expected_fingerprint: str, n_expected: int) -> dict | None:
    """A completed cell resumes iff its stored fingerprint + count match."""
    if not raw_path.exists():
        return None
    body = json.loads(raw_path.read_text())
    if body.get("fingerprint") != expected_fingerprint:
        raise C.CacheStaleError(
            f"stale gen cell at {raw_path}: fingerprint differs from the frozen "
            "generating parameters; quarantine it before resuming (never silently reuse)"
        )
    if len(body.get("records", [])) != n_expected:
        raise C.CacheStaleError(
            f"gen cell at {raw_path} carries {len(body.get('records', []))} records, "
            f"expected {n_expected}; partial cells are re-generated whole"
        )
    return body


def resume_cell_with_manifest(
    raw_path: Path,
    man_path: Path,
    cell: CellWork,
    expected_fingerprint: str,
    n_expected: int,
    sha_by_item: dict[str, str] | None = None,
) -> dict | None:
    """Resume one completed cell AND idempotently rewrite its gen manifest.

    The fresh path writes ``raw_path`` then ``man_path`` as two individually
    atomic writes — the PAIR is not atomic, so a kill between them leaves a
    fingerprint-valid, count-valid raw cell whose manifest never exists, and a
    raw-path-keyed resume would strand it manifest-less forever.
    ``manifest_rows_for_cell`` is a pure function of the resumed body, so the
    resume branch rewrites the manifest atomically; the rewrite is
    byte-identical to what the original run would have written.
    """
    body = load_resume_cell(raw_path, expected_fingerprint, n_expected)
    if body is None:
        return None
    write_jsonl_atomic(man_path, manifest_rows_for_cell(cell, body, sha_by_item))
    return body


def resolve_n_responses(responses_arg: int | None, split: str) -> int:
    """Responses/prompt: the explicit override or the split default.

    A non-positive override REFUSES loud — the legacy ``args.responses or ...``
    idiom silently fell through ``--responses 0`` to the split default.
    """
    if responses_arg is None:
        return (
            int(C.DECODER["n_responses_per_prompt_pilot"])
            if split == "pilot"
            else int(C.DECODER["n_responses_per_prompt_production"])
        )
    if responses_arg <= 0:
        raise SystemExit(f"--responses must be a positive integer, got {responses_arg}")
    return responses_arg


def exit_hard_under_live_engine(exc: BaseException) -> None:
    """Print the FULL exception chain, flush, and ``os._exit(1)``.

    A raise propagating past a live vLLM engine enters interpreter
    finalization with live EngineCore children — the #1739/#2149 deadlock
    class this module's success terminal (``os._exit(0)``) exists to avoid.
    The traceback is printed BEFORE the hard exit so the fail-loud diagnosis
    is never lost.
    """
    traceback.print_exception(exc)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(1)


def run(args: argparse.Namespace) -> int:
    split = args.split
    n_responses = resolve_n_responses(args.responses, split)
    # Split-aware cap (plan v6 A7): pilot 1024, dev/test from the frozen
    # committed amendment record (loud when absent, never a bare default).
    max_new_tokens = resolve_max_new_tokens(split)
    out_root = Path(args.out_root) if args.out_root else F.OUT_DIR
    if args.smoke:
        out_root = out_root / "smoke_gen"

    cells = build_cells(args.rows, split)
    if args.smoke:
        cells = cells[: args.smoke_cells]
    cells = cells[args.shard_index :: args.num_shards]
    if not cells:
        raise F.FrameManifestError(
            f"shard {args.shard_index}/{args.num_shards} received zero cells"
        )
    shard_tag = f"shard{args.shard_index:02d}of{args.num_shards:02d}"
    print(
        f"[gen] {shard_tag}: {len(cells)} cells x {n_responses} responses/prompt "
        f"(split={split}, max_new_tokens={max_new_tokens}, "
        f"prompt_budget={prompt_budget_for_cap(max_new_tokens)}, smoke={args.smoke})",
        flush=True,
    )

    # Resolve ALL texts up front, loud on any miss. Pilot items verify against
    # the frozen pin table; dev/test items verify against the frozen production
    # selection (text-sha cells directly, pilot-reused items against the pin
    # table too; round 15).
    all_ids = [iid for cw in cells for iid in cw.item_ids]
    resolved = resolve_items_for_split(all_ids, split)

    verify_frozen_file_pins()
    tok = load_tokenizer()

    # Budget-check every prompt BEFORE the engine spends anything.
    for cw in cells:
        for iid in cw.item_ids:
            rendered_prompt_or_raise(tok, resolved[iid], max_new_tokens)

    order_body = order_manifest_body(cells, n_responses, split, shard_tag)
    write_immutable_json(out_root / "gen_order_manifest" / f"{split}_{shard_tag}.json", order_body)
    print(
        f"[gen] order manifest frozen: {order_body['n_requests']} requests, "
        f"sha={order_body['requests_sha256'][:16]}",
        flush=True,
    )
    if args.dry_run:
        print("[gen] dry-run: stopping before engine init", flush=True)
        return 0

    llm = build_engine(args.tensor_parallel)
    # From here to process exit a vLLM engine is LIVE: EVERY termination path —
    # success and failure alike — must go through os._exit, never interpreter
    # finalization (#1739/#2149: finalize-time multiprocessing cleanup blocks
    # on surviving EngineCore children — a fail-loud raise would otherwise
    # convert into a finalization HANG on a billing pod).
    try:
        t0 = time.time()
        cap_rows: list[dict[str, Any]] = []
        n_resumed = 0
        for i, cw in enumerate(cells):
            raw_path, man_path = out_paths(out_root, split, cw.name)
            fp = generation_fingerprint(cw, n_responses, split, max_new_tokens)
            sha_by_item = (
                None
                if split == "pilot"
                else {iid: resolved[iid].prompt_sha256 for iid in cw.item_ids}
            )
            body = resume_cell_with_manifest(
                raw_path, man_path, cw, fp, len(cw.item_ids) * n_responses, sha_by_item
            )
            was_resumed = body is not None
            if body is None:
                body = generate_cell(llm, tok, cw, resolved, n_responses, split, max_new_tokens)
                write_text_atomic(raw_path, json.dumps(body, ensure_ascii=False))
                write_jsonl_atomic(man_path, manifest_rows_for_cell(cw, body, sha_by_item))
            else:
                n_resumed += 1
            cap_rows.extend(
                {"row": cw.row, "cell": cw.cell, "finish_reason": r["finish_reason"]}
                for r in body["records"]
            )
            print(
                f"[gen] cell {i + 1}/{len(cells)} {cw.name} "
                f"records={len(body['records'])} resumed={was_resumed} "
                f"elapsed={time.time() - t0:.0f}s",
                flush=True,
            )

        report = cap_hit_report(cap_rows)
        from explore_persona_space.orchestrate.provenance import as_metadata_dict, git_provenance

        summary = {
            "issue": 2658,
            "split": split,
            "shard": shard_tag,
            # Realized split cap + budget (plan v6 A7): P4/P5 artifacts are
            # self-describing without re-deriving the decoder.
            "max_new_tokens": int(max_new_tokens),
            "prompt_budget": prompt_budget_for_cap(max_new_tokens),
            "n_cells": len(cells),
            "n_cells_resumed": n_resumed,
            "cap_hit": report,
            "order_requests_sha256": order_body["requests_sha256"],
            "metadata": as_metadata_dict(git_provenance(), phase="gen"),
        }
        summary_path = out_root / "gen_summary" / f"{split}_{shard_tag}.json"
        write_text_atomic(summary_path, canonical_json(summary))
        if report["amendment_required"]:
            amend_path = out_root / "gen_summary" / f"cap_amendment_{split}_{shard_tag}.json"
            write_text_atomic(
                amend_path,
                canonical_json({"cells_over_threshold": report["cells_over_threshold"]}),
            )
            print(
                f"[gen] CAP AMENDMENT REQUIRED: {len(report['cells_over_threshold'])} cells "
                f"> {CAP_HIT_AMEND_THRESHOLD:.0%} length-cap hits — pre-test amendment, "
                "never selective regeneration (plan §5)",
                flush=True,
            )
        print(
            f"[gen] {shard_tag} done: {len(cap_rows)} records; summary -> {summary_path}",
            flush=True,
        )

        if args.upload:
            upload_raw(out_root, smoke=args.smoke)

        print("[phase=gen] done", flush=True)
    except BaseException as exc:  # fail-loud hard exit; see exit_hard_under_live_engine
        exit_hard_under_live_engine(exc)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)  # vLLM engine children survive finalization otherwise (#1739/#2149)


def upload_raw(out_root: Path, *, smoke: bool) -> None:
    """Persist raw completions to the HF data repo BEFORE anything consumes them.

    Smoke uploads land under a ``_smoke``-suffixed experiment name so they can
    never overwrite the production prefix.
    """
    from explore_persona_space.orchestrate.hub import upload_raw_completions_to_data_repo

    name = EXPERIMENT_NAME + ("_smoke" if smoke else "")
    uploaded = upload_raw_completions_to_data_repo(name, out_root)
    if not uploaded:
        raise RuntimeError(
            f"upload_raw matched ZERO raw completion files under {out_root} for {name}/ — "
            "nothing was persisted; refusing to report success (a zero-match scan is a "
            "path/layout bug, never a clean upload)"
        )
    print(f"[gen] uploaded {len(uploaded)} raw completion files under {name}/", flush=True)


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--split", choices=list(C.SPLITS), default="pilot")
    ap.add_argument("--rows", nargs="*", default=None, help="row subset (default: all)")
    ap.add_argument("--responses", type=int, default=None, help="override responses/prompt")
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--tensor-parallel", type=int, default=1)
    ap.add_argument(
        "--out-root", default=None, help="output root (default eval_results/issue_2658)"
    )
    ap.add_argument(
        "--smoke", action="store_true", help="tiny slice; out-root rebinds to smoke_gen/"
    )
    ap.add_argument("--smoke-cells", type=int, default=2, help="cells kept under --smoke")
    ap.add_argument("--dry-run", action="store_true", help="stop before engine init")
    ap.add_argument("--upload", action="store_true", help="upload raw completions to HF after gen")
    ap.add_argument("--import-check", action="store_true", help="static arg/bind check only")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    if args.import_check:
        from explore_persona_space.orchestrate.argcheck import assert_args_attributes_defined

        assert_args_attributes_defined(__file__, str(_SCRIPTS_DIR / "issue2658_text_resolver.py"))
        print("[gen] import-check OK", flush=True)
        return 0
    if not (0 <= args.shard_index < args.num_shards):
        raise SystemExit(f"--shard-index {args.shard_index} not in [0, {args.num_shards})")
    R.apply_datasets_cache()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
