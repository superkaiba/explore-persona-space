"""Issue #2658 unit 3 — teacher-forced, hookless L19 answer-mean capture.

Plan §5: teacher-force every retained response WITHOUT hooks
(``output_hidden_states=True`` on a plain forward) and average the layer-19
residual states over the GENERATED NON-SPECIAL assistant tokens — exactly ONE
canonical uncentered ``(3584,)`` float32 vector per retained answer.  A test
answer is NEVER peer-centered (unit-1 guarded fail-on condition): every vector
is a pure function of its OWN forward — no cross-answer statistic exists
anywhere in this module (pinned by the peer-independence test).

Span discipline (#825/#1092 BPE-seam rules, via the shared
``experiments.issue_1739.capture`` helpers): the forwarded row concatenates
per-segment TOKEN IDS (the prompt segment is bit-identical to what generation
consumed — same rendered string, same frozen tokenizer revision), positions are
derived from the id concatenation, and the teacher-forcing boundary is the
special-token suffix that cannot BPE-merge into the completion tail.  Special
tokens that a completion re-tokenizes INTO (a literal ``<|endoftext|>`` string
in the answer) are EXCLUDED from the mean per the registered definition; a span
left empty after exclusion fails loud — the definition never changes (plan §5).

Layer convention (producer parity, ``capture.py:201``): ``hidden_states[1:]``
are the 28 post-block states, so the frozen ``LAYER = 19`` (0-based block) reads
``outputs.hidden_states[LAYER + 1]``.

Store layout: per-shard ``l19mean_shard{NN}.npy`` float32 ``(n, 3584)`` +
``row_index_shard{NN}.jsonl`` + fingerprint meta sidecar (atomic writes,
fingerprint-gated resume keyed on GENERATING PARAMETERS — #1336 rule).
``vector_sha256`` domain: sha256 over the little-endian float32 C-order bytes.

CONTENT HYGIENE: answer text flows gen files -> memory -> model; logs and the
row index carry only ids, counts, positions, and sha256s.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()  # #847 thread caps bind before torch import

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import numpy as np  # noqa: E402

import issue2658_common as C  # noqa: E402
import issue2658_frames as F  # noqa: E402
import issue2658_generate as G  # noqa: E402
import issue2658_text_resolver as R  # noqa: E402
from explore_persona_space.atomic_io import atomic_replace  # noqa: E402
from explore_persona_space.experiments.issue_1739.capture import (  # noqa: E402
    BOUNDARY_INSTRUCT,
    capture_row_ids_and_positions,
)

N_BLOCKS = 28  # Qwen-2.5-7B post-block states; asserted per forward
# Re-tokenizing generated text can drift a few tokens vs the generation count,
# so the CAPTURE window carries headroom over the generation window (positions
# are unaffected; this is an HF forward, not a vLLM engine constraint).
CAPTURE_MAX_MODEL_LEN = G.MAX_MODEL_LEN + 256
DEFAULT_BATCH_SIZE = int(os.environ.get("EPM_CAPTURE_BATCH_SIZE", "8"))
DEFAULT_SHARD_ROWS = 512
SPAN_RULE = "answer_nonspecial_mean_v1"


class CaptureSpanError(C.Issue2658GuardError):
    """The answer span is empty / misaligned — the feature is undefined (loud)."""


@dataclass
class CaptureRow:
    """One teacher-forcing unit. ``text`` fields are memory-only (repr=False)."""

    prompt_id: str
    response_index: int
    answer_sha256: str
    rendered_prompt: str = field(repr=False, default="")
    answer_text: str = field(repr=False, default="")

    @property
    def key(self) -> tuple[str, int]:
        return (self.prompt_id, self.response_index)


# ---------------------------------------------------------------------------
# Span arithmetic (pure; unit-tested offline with a fake tokenizer).
# ---------------------------------------------------------------------------
def answer_positions_nonspecial(
    pos: dict[str, int], completion_ids: list[int], special_ids: frozenset[int]
) -> tuple[list[int], int]:
    """Row positions of the generated NON-SPECIAL answer tokens.

    ``pos`` is the shared helper's position dict; ``completion_ids`` are the
    re-tokenized answer ids (``add_special_tokens=False``).  Prompt tokens and
    the teacher-forcing boundary sit OUTSIDE ``[answer_start, answer_end)`` by
    construction; special ids INSIDE the span are excluded per the registered
    definition.  Returns ``(kept_positions, n_special_excluded)``; an empty
    result RAISES (plan §5: never redefine the feature).
    """
    if not completion_ids:
        raise CaptureSpanError("zero-token answer reached capture (generation contract broken)")
    span = list(range(pos["answer_start"], pos["answer_end"]))
    if len(span) != len(completion_ids):
        raise CaptureSpanError(
            f"answer span width {len(span)} != completion token count "
            f"{len(completion_ids)} (positions clamped — over-budget row?)"
        )
    kept = [p for p, tid in zip(span, completion_ids, strict=True) if tid not in special_ids]
    n_excluded = len(span) - len(kept)
    if not kept:
        raise CaptureSpanError(
            "answer span empty after special-token exclusion; the mean is undefined "
            "and the definition never changes (plan §5)"
        )
    return kept, n_excluded


def build_capture_unit(
    tokenizer, row: CaptureRow, special_ids: frozenset[int]
) -> tuple[list[int], list[int], dict[str, int], int]:
    """Token ids + kept positions for one row (shared BPE-seam-safe helper)."""
    row_ids, pos = capture_row_ids_and_positions(
        tokenizer,
        "",  # no prefix segment: single user turn, no system message (plan §5)
        row.rendered_prompt,
        row.answer_text,
        BOUNDARY_INSTRUCT,
        row_label=f"{row.prompt_id}#k{row.response_index}",
        max_model_len=CAPTURE_MAX_MODEL_LEN,
        max_formatted_tokens=G.PROMPT_BUDGET,
    )
    completion_ids = list(tokenizer.encode(row.answer_text, add_special_tokens=False))
    kept, n_excluded = answer_positions_nonspecial(pos, completion_ids, special_ids)
    return row_ids, kept, pos, n_excluded


# ---------------------------------------------------------------------------
# Batched hookless forward (single layer, answer-span mean, fp32).
# ---------------------------------------------------------------------------
def capture_l19_answer_means(
    rows: list[CaptureRow],
    *,
    model,
    tokenizer,
    device: str = "cuda",
    batch_size: int = DEFAULT_BATCH_SIZE,
    layer: int = C.LAYER,
    n_blocks: int = N_BLOCKS,
    hidden: int = C.HIDDEN,
    log_label: str = "capture",
) -> tuple[list[np.ndarray], list[dict[str, Any]]]:
    """One uncentered fp32 ``(hidden,)`` vector per row (answer-span mean at
    ``layer``).  Padded batch forwards with ``output_hidden_states=True`` — no
    hooks; each row's vector is a function of its OWN forward only (right-pad +
    causal mask: pad/peer rows cannot influence real positions)."""
    import torch  # lazy: GPU dep

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if getattr(tokenizer, "padding_side", "right") != "right":
        raise CaptureSpanError(
            "capture positions index the UNPADDED sequence and require RIGHT padding; "
            f"tokenizer.padding_side={tokenizer.padding_side!r}"
        )
    special_ids = frozenset(int(t) for t in tokenizer.all_special_ids)

    vectors: list[np.ndarray] = []
    metas: list[dict[str, Any]] = []
    n_total = len(rows)
    t0 = time.time()
    for start in range(0, n_total, max(1, batch_size)):
        batch = rows[start : start + max(1, batch_size)]
        batch_ids: list[list[int]] = []
        batch_kept: list[list[int]] = []
        batch_meta: list[dict[str, Any]] = []
        for row in batch:
            row_ids, kept, pos, n_excluded = build_capture_unit(tokenizer, row, special_ids)
            batch_ids.append(row_ids)
            batch_kept.append(kept)
            batch_meta.append(
                {
                    "prompt_id": row.prompt_id,
                    "response_index": row.response_index,
                    "answer_sha256": row.answer_sha256,
                    "n_total_tokens": pos["n_total"],
                    "n_prompt_tokens": pos["n_prompt"],
                    "answer_start": pos["answer_start"],
                    "answer_end": pos["answer_end"],
                    "n_answer_tokens_kept": len(kept),
                    "n_special_excluded": n_excluded,
                }
            )
        inputs = tokenizer.pad({"input_ids": batch_ids}, return_tensors="pt", padding=True)
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        with torch.no_grad():
            # Hidden-state-only forward: skip the full-vocab logits (#779 OOM
            # class). Canonical introspection guard — the kwarg is passed ONLY
            # when the forward names an EXPLICIT `logits_to_keep` parameter
            # (a bare **kwargs does not count), per gotchas.md.
            from explore_persona_space.analysis.extraction import _logits_to_keep_kwargs

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                **_logits_to_keep_kwargs(model, return_logits=False),
            )
        hidden_states = outputs.hidden_states[1:]  # post-block states (skip embeddings)
        if len(hidden_states) != n_blocks:
            raise CaptureSpanError(
                f"model returned {len(hidden_states)} post-block states, expected {n_blocks}"
            )
        h_layer = hidden_states[layer]
        if h_layer.shape[-1] != hidden:
            raise CaptureSpanError(f"hidden dim {h_layer.shape[-1]} != expected {hidden}")
        for b, kept in enumerate(batch_kept):
            vec = h_layer[b, kept, :].to(torch.float32).mean(dim=0).cpu().numpy().astype("<f4")
            assert vec.shape == (hidden,), vec.shape
            vectors.append(vec)
            metas.append(batch_meta[b])
        print(
            f"[{log_label}] rows {start + len(batch)}/{n_total} elapsed={time.time() - t0:.0f}s",
            flush=True,
        )
    return vectors, metas


def vector_sha256(vec: np.ndarray) -> str:
    """sha256 over the little-endian float32 C-order bytes (the frozen domain)."""
    arr = np.ascontiguousarray(vec, dtype="<f4")
    return hashlib.sha256(arr.tobytes()).hexdigest()


# ---------------------------------------------------------------------------
# Store (per-shard npy + row_index + fingerprint meta; resume by prefix).
# ---------------------------------------------------------------------------
def device_class(device: str) -> str:
    """Backend CLASS of a torch device string: ``cuda`` / ``cuda:1`` -> ``cuda``.

    The device INDEX is placement, not an output-affecting regime key; the
    cuda-vs-cpu kernel family IS (different numerics for the same math).
    """
    return device.split(":", 1)[0].strip().lower()


def capture_fingerprint(split: str, *, dtype: str, device: str) -> str:
    """Machine-stable fingerprint over GENERATING PARAMETERS only (#1336).

    ``dtype`` and the device CLASS are OUTPUT-AFFECTING regime keys (a bf16
    cuda forward and an fp32 cpu forward compute different vectors for the
    same rows); omitting them let a store begun under bf16 silently extend
    with fp32-computed shards.  Schema v2 adds them — deliberately
    INVALIDATING any store fingerprinted under v1.
    """
    payload = json.dumps(
        {
            "schema": "i2658-l19-capture-v2",  # v2: + dtype / device_class regime keys
            "model_id": C.MODEL_ID,
            "model_revision": C.MODEL_REVISION,
            "chat_template_sha256": C.CHAT_TEMPLATE_SHA256,
            "layer": C.LAYER,
            "span_rule": SPAN_RULE,
            "boundary": BOUNDARY_INSTRUCT,
            "capture_max_model_len": CAPTURE_MAX_MODEL_LEN,
            "prompt_budget": G.PROMPT_BUDGET,
            "dtype": dtype,
            "device_class": device_class(device),
            "split": split,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _shard_paths(store_dir: Path, idx: int) -> tuple[Path, Path, Path]:
    return (
        store_dir / f"l19mean_shard{idx:02d}.npy",
        store_dir / f"row_index_shard{idx:02d}.jsonl",
        store_dir / f"_capture_meta_shard{idx:02d}.json",
    )


def shard_done(store_dir: Path, idx: int, fingerprint: str) -> bool:
    npy, index, meta = _shard_paths(store_dir, idx)
    if not (npy.exists() and index.exists() and meta.exists()):
        return False
    try:
        return json.loads(meta.read_text()).get("fingerprint") == fingerprint
    except (json.JSONDecodeError, UnicodeDecodeError, OSError):
        # UnicodeDecodeError is a ValueError, not an OSError: a truncated /
        # partially-written meta file raises it and would otherwise escape this
        # resume probe. Returning False re-does the shard, the safe direction.
        return False


def read_shard_index(store_dir: Path, idx: int) -> list[dict]:
    rows = []
    with _shard_paths(store_dir, idx)[1].open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_shard(
    store_dir: Path,
    idx: int,
    vectors: list[np.ndarray],
    metas: list[dict[str, Any]],
    fingerprint: str,
) -> None:
    if len(vectors) != len(metas) or not vectors:
        raise CaptureSpanError(f"shard {idx}: {len(vectors)} vectors vs {len(metas)} metas")
    npy, index, meta = _shard_paths(store_dir, idx)
    store_dir.mkdir(parents=True, exist_ok=True)
    arr = np.stack(vectors, axis=0).astype("<f4")
    assert arr.shape == (len(vectors), C.HIDDEN), arr.shape
    # atomic_replace tmp names end `.tmp` (never matches the shard glob);
    # np.save through an open handle (it APPENDS `.npy` to bare path names).
    with atomic_replace(npy) as tmp, tmp.open("wb") as fh:
        np.save(fh, arr)
    with atomic_replace(index) as tmp:
        with tmp.open("w", encoding="utf-8") as fh:
            for m, v in zip(metas, vectors, strict=True):
                fh.write(json.dumps({**m, "vector_sha256": vector_sha256(v)}) + "\n")
    with atomic_replace(meta) as tmp:
        tmp.write_text(json.dumps({"fingerprint": fingerprint, "n_rows": len(vectors)}))


def resume_completed_shards(
    store_dir: Path, rows: list[CaptureRow], fingerprint: str
) -> tuple[int, int]:
    """Validate the contiguous done-shard prefix; returns ``(offset, next_shard)``.

    KEYS and CONTENT both bind.  The key check alone is CONTENT-BLIND: a
    quarantined partial gen cell regenerated at the UNCHANGED fingerprint can
    carry DIFFERENT text for the same ``(prompt_id, response_index)`` keys
    (vLLM temperature-1.0 sampling is batch-composition and kernel sensitive
    across engine rebuilds even at fixed per-request seeds), so a store
    written before the re-gen would resume clean and every downstream number
    would score a vector for text that no longer exists.  Each realized
    row_index row already carries ``answer_sha256`` and each expected
    ``CaptureRow`` carries the CURRENT gen file's sha (verified at load) —
    any per-row mismatch raises ``CacheStaleError``.
    """
    offset = 0
    next_shard = 0
    while shard_done(store_dir, next_shard, fingerprint):
        realized = read_shard_index(store_dir, next_shard)
        expected = rows[offset : offset + len(realized)]
        if [(r["prompt_id"], int(r["response_index"])) for r in realized] != [
            c.key for c in expected
        ]:
            raise C.CacheStaleError(
                f"resume mismatch at shard{next_shard:02d} of {store_dir}: realized rows "
                "are not the expected prefix slice — quarantine the store or use a fresh "
                "store dir"
            )
        stale = [
            c.key
            for r, c in zip(realized, expected, strict=True)
            if r["answer_sha256"] != c.answer_sha256
        ]
        if stale:
            raise C.CacheStaleError(
                f"resume CONTENT mismatch at shard{next_shard:02d} of {store_dir}: "
                f"{len(stale)} resumed rows carry answer_sha256 differing from the "
                f"CURRENT gen files (e.g. {stale[:3]}) — the answers were regenerated "
                "after this store was written; quarantine the store or use a fresh "
                "store dir (a key-matched resume must never keep vectors for text "
                "that no longer exists)"
            )
        offset += len(realized)
        next_shard += 1
    return offset, next_shard


def expected_capture_keys(
    rows_filter: list[str] | None,
    n_responses: int,
    shard_index: int,
    num_shards: int,
    *,
    present_cells: set[str] | None = None,
) -> list[tuple[str, int]]:
    """This shard's expected ``(prompt_id, response_index)`` list, anchored on
    the FRAME MANIFEST's pilot selection x ``n_responses`` — never on the gen
    files (that anchor is CIRCULAR: a capture launched over an incomplete gen
    dir would report "complete").

    ``present_cells`` (``--smoke`` only) restricts the anchor to gen-present
    cell names ``{row}__{frame}__{band}``: within-cell completeness stays
    manifest-anchored; cross-cell completeness is NOT certified under smoke
    (the gen smoke truncates the cell grid by design).
    """
    keys: list[tuple[str, int]] = []
    for row, cell, iid in R.pilot_item_ids():
        if rows_filter and row not in rows_filter:
            continue
        frame, _, band = cell.partition("|")
        if present_cells is not None and f"{row}__{frame}__{band}" not in present_cells:
            continue
        keys.extend((iid, k) for k in range(n_responses))
    keys.sort()
    keys = keys[shard_index::num_shards]
    if not keys:
        raise CaptureSpanError(
            f"shard {shard_index}/{num_shards} has zero expected capture rows under the "
            "frame-manifest anchor (rows filter too narrow, or empty smoke gen dir?)"
        )
    return keys


def assert_store_complete(store_dir: Path, expected_keys: list[tuple[str, int]]) -> dict:
    """Realized row_index keys must equal the expected set EXACTLY (loud)."""
    realized: list[tuple[str, int]] = []
    per_shard: dict[str, int] = {}
    for path in sorted(store_dir.glob("row_index_shard*.jsonl")):
        idx = int(path.stem[len("row_index_shard") :])
        rows = read_shard_index(store_dir, idx)
        per_shard[f"{idx:02d}"] = len(rows)
        realized.extend((r["prompt_id"], int(r["response_index"])) for r in rows)
    exp, got = set(expected_keys), set(realized)
    problems = []
    if len(exp) != len(expected_keys):
        raise CaptureSpanError("expected capture key list carries duplicates")
    if exp - got:
        problems.append(f"{len(exp - got)} expected rows MISSING (e.g. {sorted(exp - got)[:3]})")
    if got - exp:
        problems.append(f"{len(got - exp)} foreign rows (e.g. {sorted(got - exp)[:3]})")
    if len(realized) != len(got):
        problems.append(f"{len(realized) - len(got)} duplicate realized rows")
    if problems:
        raise CaptureSpanError(f"capture store INCOMPLETE at {store_dir}: " + "; ".join(problems))
    return {"n_rows": len(realized), "per_shard_rows": per_shard}


# ---------------------------------------------------------------------------
# Driver.
# ---------------------------------------------------------------------------
def load_generation_rows(
    out_root: Path, split: str, rows_filter: list[str] | None
) -> list[CaptureRow]:
    """Every retained answer from the gen cell files, sha-verified, text in memory."""
    gen_dir = out_root / "raw_completions" / split
    files = sorted(gen_dir.glob("*.json"))
    if rows_filter:
        files = [p for p in files if p.name.split("__", 1)[0] in rows_filter]
    if not files:
        raise CaptureSpanError(f"no generation cell files under {gen_dir}")
    out: list[CaptureRow] = []
    for path in files:
        body = json.loads(path.read_text())
        if body.get("schema") != G.GEN_SCHEMA:
            raise CaptureSpanError(f"{path.name}: unexpected gen schema {body.get('schema')!r}")
        for r in body["records"]:
            C.assert_row_hash(r["text"], r["answer_sha256"])  # gen record integrity
            out.append(
                CaptureRow(
                    prompt_id=r["prompt_id"],
                    response_index=int(r["response_index"]),
                    answer_sha256=r["answer_sha256"],
                    answer_text=r["text"],
                )
            )
    keys = [c.key for c in out]
    if len(set(keys)) != len(keys):
        raise CaptureSpanError("duplicate (prompt_id, response_index) across gen cell files")
    return out


def attach_rendered_prompts(rows: list[CaptureRow], tokenizer) -> None:
    """Re-resolve + re-render every prompt (pins verified) — bit-identical to
    the string generation consumed (same tokenizer revision, same template)."""
    ids = sorted({c.prompt_id for c in rows})
    resolved = R.resolve_items(ids, verify_pins=True)
    rendered = {iid: R.render_user_prompt(tokenizer, resolved[iid].text) for iid in ids}
    for c in rows:
        c.rendered_prompt = rendered[c.prompt_id]


def load_capture_model(device: str, dtype: str):
    import torch
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        C.MODEL_ID,
        revision=C.MODEL_REVISION,
        torch_dtype=getattr(torch, dtype),
    ).to(device)
    model.eval()
    return model


def run(args: argparse.Namespace) -> int:
    split = args.split
    out_root = Path(args.out_root) if args.out_root else F.OUT_DIR
    if args.smoke:
        out_root = out_root / "smoke_gen"
    store_dir = (
        out_root / "l19_store" / split / f"shard{args.shard_index:02d}of{args.num_shards:02d}"
    )
    fingerprint = capture_fingerprint(split, dtype=args.dtype, device=args.device)
    n_responses = G.resolve_n_responses(args.responses, split)

    rows = load_generation_rows(out_root, split, args.rows)
    rows.sort(key=lambda c: c.key)
    rows = rows[args.shard_index :: args.num_shards]
    if not rows:
        raise CaptureSpanError(
            f"shard {args.shard_index}/{args.num_shards} received zero capture rows"
        )

    # Completeness anchor: the frame manifest x n_responses — checked BEFORE any
    # model forward (fail before the GPU spend) and re-asserted on the realized
    # store below. Under --smoke the anchor restricts to gen-present cells.
    present_cells: set[str] | None = None
    if args.smoke:
        present_cells = {p.stem for p in (out_root / "raw_completions" / split).glob("*.json")}
        print(
            f"[capture] --smoke: completeness anchor RESTRICTED to {len(present_cells)} "
            "gen-present cells (cross-cell completeness NOT certified under smoke; "
            "production anchors the full frame manifest)",
            flush=True,
        )
    expected_keys = expected_capture_keys(
        args.rows, n_responses, args.shard_index, args.num_shards, present_cells=present_cells
    )
    got_keys = [c.key for c in rows]
    if got_keys != expected_keys:
        exp, got = set(expected_keys), set(got_keys)
        raise CaptureSpanError(
            f"gen rows for shard {args.shard_index}/{args.num_shards} do not match the "
            f"frame-manifest expectation: {len(exp - got)} expected rows missing "
            f"(e.g. {sorted(exp - got)[:3]}), {len(got - exp)} foreign "
            f"(e.g. {sorted(got - exp)[:3]}) — complete generation before capture"
        )
    print(f"[capture] {len(rows)} answers to capture -> {store_dir}", flush=True)

    tok = G.load_tokenizer()
    attach_rendered_prompts(rows, tok)

    if args.dry_run:
        # Everything except the model forward: span arithmetic on every row.
        special_ids = frozenset(int(t) for t in tok.all_special_ids)
        for c in rows:
            build_capture_unit(tok, c, special_ids)
        print(f"[capture] dry-run: {len(rows)} rows span-validated; stopping", flush=True)
        return 0

    model = load_capture_model(args.device, args.dtype)

    # Resume: contiguous done shards must be the expected prefix of THIS row
    # list — KEYS and CONTENT (per-row answer_sha256) both bind; see
    # resume_completed_shards.
    offset, next_shard = resume_completed_shards(store_dir, rows, fingerprint)
    if next_shard:
        print(f"[capture] resumed {next_shard} shards ({offset} rows)", flush=True)

    remaining = rows[offset:]
    for k in range(0, len(remaining), args.shard_rows):
        idx = next_shard + k // args.shard_rows
        npy, index, _ = _shard_paths(store_dir, idx)
        if npy.exists() or index.exists():
            raise C.CacheStaleError(
                f"shard{idx:02d} exists beyond the contiguous resumed prefix of "
                f"{store_dir} (stale/foreign fingerprint) — refusing to overwrite"
            )
        chunk = remaining[k : k + args.shard_rows]
        vectors, metas = capture_l19_answer_means(
            chunk,
            model=model,
            tokenizer=tok,
            device=args.device,
            batch_size=args.batch_size,
            log_label=f"capture-shard{idx:02d}",
        )
        write_shard(store_dir, idx, vectors, metas, fingerprint)
        print(f"[capture] shard {idx:02d} written ({len(chunk)} rows)", flush=True)

    digest = assert_store_complete(store_dir, expected_keys)
    from explore_persona_space.orchestrate.provenance import as_metadata_dict, git_provenance

    manifest = {
        "issue": 2658,
        "split": split,
        "shard": f"{args.shard_index:02d}of{args.num_shards:02d}",
        "fingerprint": fingerprint,
        "layer": C.LAYER,
        "hidden": C.HIDDEN,
        "span_rule": SPAN_RULE,
        "vector_sha256_domain": "little-endian float32 C-order bytes",
        **digest,
        "metadata": as_metadata_dict(git_provenance(), phase="capture"),
    }
    with atomic_replace(store_dir / "_capture_manifest.json") as tmp:
        tmp.write_text(json.dumps(manifest, indent=1, sort_keys=True))
    print(f"[capture] complete: {digest['n_rows']} vectors in {store_dir}", flush=True)

    if args.upload:
        upload_store(store_dir, split, smoke=args.smoke)
    print("[phase=capture] done", flush=True)
    return 0


def upload_store(store_dir: Path, split: str, *, smoke: bool) -> None:
    """Upload the activation shards to the issue-owned analysis_tensors prefix
    (one bulk folder commit; verified) BEFORE any teardown."""
    from huggingface_hub import HfApi

    from explore_persona_space.orchestrate.hub import (
        DEFAULT_DATASET_REPO,
        _upload,
        verify_repo_paths_uploaded,
    )

    name = G.EXPERIMENT_NAME + ("_smoke" if smoke else "")
    prefix = f"{name}/analysis_tensors/l19_{split}/{store_dir.name}"
    # _upload signature: (local_path, repo_id, repo_type, path_in_repo, ...);
    # folder branch => one bulk upload_folder commit under `prefix`.
    url = _upload(store_dir, DEFAULT_DATASET_REPO, "dataset", prefix, raise_on_error=True)
    if not url:
        raise RuntimeError(
            f"capture store upload of {store_dir} returned no URL — durability unverified "
            f"for {DEFAULT_DATASET_REPO}/{prefix}"
        )
    expected = [f"{prefix}/{p.name}" for p in sorted(store_dir.iterdir()) if p.is_file()]
    missing = verify_repo_paths_uploaded(
        HfApi(),
        DEFAULT_DATASET_REPO,
        expected,
        path_in_repo=prefix,
        repo_type="dataset",
    )
    if missing:
        raise RuntimeError(
            f"capture upload verification failed: {len(missing)} of {len(expected)} files "
            f"missing under {DEFAULT_DATASET_REPO}/{prefix}: {sorted(missing)[:5]} ..."
        )
    print(f"[capture] uploaded + verified {len(expected)} files under {prefix}/", flush=True)


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--split", choices=list(C.SPLITS), default="pilot")
    ap.add_argument("--rows", nargs="*", default=None, help="row subset (default: all)")
    ap.add_argument(
        "--responses",
        type=int,
        default=None,
        help="responses/prompt used at GEN time (default: the split default; "
        "anchors the manifest completeness check)",
    )
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument(
        "--out-root", default=None, help="gen out-root (default eval_results/issue_2658)"
    )
    ap.add_argument("--smoke", action="store_true", help="read/write the smoke_gen/ out-root")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16", help="model dtype (float32 for CPU legs)")
    ap.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    ap.add_argument("--shard-rows", type=int, default=DEFAULT_SHARD_ROWS)
    ap.add_argument("--dry-run", action="store_true", help="span-validate only; no model")
    ap.add_argument("--upload", action="store_true", help="upload the store to HF after capture")
    ap.add_argument("--import-check", action="store_true", help="static arg/bind check only")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    if args.import_check:
        from explore_persona_space.orchestrate.argcheck import assert_args_attributes_defined

        assert_args_attributes_defined(__file__)
        print("[capture] import-check OK", flush=True)
        return 0
    if not (0 <= args.shard_index < args.num_shards):
        raise SystemExit(f"--shard-index {args.shard_index} not in [0, {args.num_shards})")
    R.apply_datasets_cache()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
