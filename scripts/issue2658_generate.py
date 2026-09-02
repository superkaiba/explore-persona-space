"""Issue #2658 unit 3 — pilot answer generation (vLLM, frozen decoder, retain-all).

Plan §5 recipe: frozen model/tokenizer/chat-template pins (``issue2658_common``),
bf16, batched ``LLM.generate()`` (never sequential HF), no tools / steering /
hooks, temperature 1.0, top_p 0.95, max_new_tokens 1024 (#779 parity),
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
PROMPT_BUDGET = MAX_MODEL_LEN - int(C.DECODER["max_new_tokens"])  # 7168
CHUNK_SIZE = int(os.environ.get("EPM_VLLM_GREEDY_CHUNK_SIZE", "500"))
N_EMPTY_RETRIES = 3
CAP_HIT_AMEND_THRESHOLD = 0.02  # strictly-above trigger (plan §5)
EXPERIMENT_NAME = "issue2658_dirvalid"  # HF data-repo prefix (unit-2 convention)
GEN_SCHEMA = "i2658-gen-cell-v1"


class EmptyOutputError(C.Issue2658GuardError):
    """A prompt draw stayed zero-token through the fixed three-retry schedule."""


class GenerationBudgetError(C.Issue2658GuardError):
    """A resolved prompt exceeds the frozen prompt token budget (loud, no skip)."""


class OrderManifestDriftError(C.Issue2658GuardError):
    """An immutable order manifest already exists with different content."""


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
) -> dict[str, Any]:
    """One TEXT-FREE generation manifest row; validated against the unit-1 schema."""
    construct = C.CONSTRUCTS[row]
    judge_scored = construct.judge_scored
    d = {
        "manifest_version": C.MANIFEST_VERSION,
        "row": row,
        "split": split,
        "prompt_id": item_id,
        "prompt_sha256": _pin_sha(item_id),
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


def build_cells(rows_filter: list[str] | None = None) -> list[CellWork]:
    """Deterministic pilot cell list from the committed, immutability-checked
    frame manifest.  Cells realized empty (below-floor cells with zero items)
    are skipped WITH a printed count — their shortfall is already recorded in
    ``pilot_selection.cells_below_pilot_floor`` by unit 2."""
    body = json.loads(F.FRAME_MANIFEST_PATH.read_text())
    F.assert_manifest_immutable(body)
    cells: list[CellWork] = []
    n_empty = 0
    for rr in body["rows"]:
        if rows_filter and rr["row"] not in rows_filter:
            continue
        sel = rr["pilot_selection"]["per_cell_item_ids"]
        for cell_key in sorted(sel):
            iids = tuple(sel[cell_key])
            if not iids:
                n_empty += 1
                continue
            frame, _, band = cell_key.partition("|")
            if not band:
                raise F.FrameManifestError(f"malformed pilot cell key {cell_key!r}")
            sfs = {iid: R.superfamily_of(body, rr["row"], iid) for iid in iids}
            cells.append(CellWork(rr["row"], frame, band, iids, sfs))
    if not cells:
        raise F.FrameManifestError("zero pilot cells selected (rows filter too narrow?)")
    if n_empty:
        print(
            f"[gen] {n_empty} registered cells realized EMPTY (below-floor; recorded "
            "by unit 2) — skipped with this disclosure",
            flush=True,
        )
    return cells


def generation_fingerprint(cell: CellWork, n_responses: int, split: str) -> str:
    """Machine-stable resume fingerprint over GENERATING PARAMETERS (#1336 rule:
    never hash recomputed floats; every value here is a frozen pin or an int)."""
    payload = json.dumps(
        {
            "schema": GEN_SCHEMA,
            "model_id": C.MODEL_ID,
            "model_revision": C.MODEL_REVISION,
            "chat_template_sha256": C.CHAT_TEMPLATE_SHA256,
            "decoder": {
                "temperature": C.DECODER["temperature"],
                "top_p": C.DECODER["top_p"],
                "max_new_tokens": C.DECODER["max_new_tokens"],
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


def rendered_prompt_or_raise(tok, item: R.ResolvedItem) -> tuple[str, int]:
    """Chat-render one prompt and enforce the frozen prompt budget (loud)."""
    rendered = R.render_user_prompt(tok, item.text)
    n_tok = len(tok.encode(rendered, add_special_tokens=False))
    if n_tok > PROMPT_BUDGET:
        raise GenerationBudgetError(
            f"prompt {item.item_id!r} renders to {n_tok} tokens > budget {PROMPT_BUDGET} "
            f"(max_model_len {MAX_MODEL_LEN} - max_new_tokens "
            f"{C.DECODER['max_new_tokens']}); plan §5 fails loud, never skips"
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


def _sampling_params(seed: int):
    from vllm import SamplingParams

    return SamplingParams(
        n=1,
        temperature=float(C.DECODER["temperature"]),
        top_p=float(C.DECODER["top_p"]),
        max_tokens=int(C.DECODER["max_new_tokens"]),
        seed=seed,
    )


def generate_cell(
    llm,
    tok,
    cell: CellWork,
    resolved: dict[str, R.ResolvedItem],
    n_responses: int,
    split: str,
) -> dict[str, Any]:
    """Generate all draws for one cell (batched, chunked); returns the cell body."""
    rendered: dict[str, str] = {}
    n_prompt_tokens: dict[str, int] = {}
    for iid in cell.item_ids:
        rendered[iid], n_prompt_tokens[iid] = rendered_prompt_or_raise(tok, resolved[iid])

    plan = [(iid, k) for iid in cell.item_ids for k in range(n_responses)]
    prompts = [rendered[iid] for iid, _ in plan]
    params = [_sampling_params(C.response_seed(iid, k)) for iid, k in plan]

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
                o = llm.generate([prompt], [_sampling_params(seed)], use_tqdm=False)[0].outputs[0]
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
        "fingerprint": generation_fingerprint(cell, n_responses, split),
        "model_id": C.MODEL_ID,
        "model_revision": C.MODEL_REVISION,
        "chat_template_sha256": C.CHAT_TEMPLATE_SHA256,
        "decoder": {
            "temperature": C.DECODER["temperature"],
            "top_p": C.DECODER["top_p"],
            "max_new_tokens": C.DECODER["max_new_tokens"],
        },
        "n_items": len(cell.item_ids),
        "n_responses_per_prompt": n_responses,
        "records": records,
        "retry_ledger": retry_ledger,
        "metadata": as_metadata_dict(git_provenance(), phase="gen"),
    }


def manifest_rows_for_cell(cell: CellWork, body: dict[str, Any]) -> list[dict[str, Any]]:
    """Validated TEXT-FREE manifest rows for one generated cell body."""
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


def run(args: argparse.Namespace) -> int:
    split = args.split
    n_responses = args.responses or (
        int(C.DECODER["n_responses_per_prompt_pilot"])
        if split == "pilot"
        else int(C.DECODER["n_responses_per_prompt_production"])
    )
    out_root = Path(args.out_root) if args.out_root else F.OUT_DIR
    if args.smoke:
        out_root = out_root / "smoke_gen"

    cells = build_cells(args.rows)
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
        f"(split={split}, smoke={args.smoke})",
        flush=True,
    )

    # Resolve ALL texts up front (pins verified; loud on any miss).
    all_ids = [iid for cw in cells for iid in cw.item_ids]
    resolved = R.resolve_items(all_ids, verify_pins=True)

    verify_frozen_file_pins()
    tok = load_tokenizer()

    # Budget-check every prompt BEFORE the engine spends anything.
    for cw in cells:
        for iid in cw.item_ids:
            rendered_prompt_or_raise(tok, resolved[iid])

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
    engine_live = True

    t0 = time.time()
    cap_rows: list[dict[str, Any]] = []
    n_resumed = 0
    for i, cw in enumerate(cells):
        raw_path, man_path = out_paths(out_root, split, cw.name)
        fp = generation_fingerprint(cw, n_responses, split)
        body = load_resume_cell(raw_path, fp, len(cw.item_ids) * n_responses)
        was_resumed = body is not None
        if body is None:
            body = generate_cell(llm, tok, cw, resolved, n_responses, split)
            write_text_atomic(raw_path, json.dumps(body, ensure_ascii=False))
            man_rows = manifest_rows_for_cell(cw, body)
            write_jsonl_atomic(man_path, man_rows)
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
    print(f"[gen] {shard_tag} done: {len(cap_rows)} records; summary -> {summary_path}", flush=True)

    if args.upload:
        upload_raw(out_root, smoke=args.smoke)

    print("[phase=gen] done", flush=True)
    if engine_live:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)  # vLLM engine children survive finalization otherwise (#1739/#2149)
    return 0


def upload_raw(out_root: Path, *, smoke: bool) -> None:
    """Persist raw completions to the HF data repo BEFORE anything consumes them.

    Smoke uploads land under a ``_smoke``-suffixed experiment name so they can
    never overwrite the production prefix.
    """
    from explore_persona_space.orchestrate.hub import upload_raw_completions_to_data_repo

    name = EXPERIMENT_NAME + ("_smoke" if smoke else "")
    uploaded = upload_raw_completions_to_data_repo(name, out_root)
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
