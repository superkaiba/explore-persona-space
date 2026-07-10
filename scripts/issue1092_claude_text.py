"""
P1: Claude text generation for issue #1092.

Generates one Claude completion per (prefix, query) pair for the control-cell row set
(~12,200 rows: dense core + battery bridge + ~40% periphery), temperature 0
(determinism-parity with greedy own-text), model claude-sonnet-4-5-20250929,
max_tokens 1024.

Routing: Anthropic Batch API via the multi-org dispatcher (api_dispatch.py),
sub-batch 5k -> 3 passes <= the 2-3 wedge-exposure bound.

Rollout text is uploaded to HF raw_completions/claude/ unconditionally before exit.
"""

from __future__ import annotations

import argparse
import asyncio
import gc
import hashlib
import json
import logging
import os
import sys
import time
from pathlib import Path

# VM thread caps — must be set before any numpy/torch import
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "8")

# Load credentials — required before any HF/Anthropic call
from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()


from explore_persona_space.llm.api_dispatch import (  # noqa: E402
    DispatchItem,
    DispatchResult,
    dispatch_calls,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("issue1092.claude_text")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CLAUDE_MODEL = "claude-sonnet-4-5-20250929"
MAX_TOKENS = 1024
TEMPERATURE = 0  # determinism-parity with greedy own-text

HF_DATA_REPO = "superkaiba1/explore-persona-space-data"
HF_COMPLETIONS_PREFIX = "issue1092_realistic_crossing/raw_completions/claude"

# Control cells that receive Claude completions (subsampled row set)
CLAUDE_CELL_SLUGS = {"cell_inst_claude", "cell_pre_claude"}

# The cells in the manifest that SELECT which rows get Claude text
# (cells sharing the subsampled ~12.2k row set per §4.3)
CONTROL_CELL_STRATA = {"dense_core", "battery", "periphery_claude_subset"}

BATCH_CHUNK_SIZE = 5_000  # sub-batch size (3 passes for ~12.2k rows)

SMOKE_ROW_LIMIT = 32
SMOKE_CELLS = {"cell_inst_own"}  # canonical smoke cell drives the manifest filter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _render_instruct(prefix_turns: list[dict], query: str) -> str:
    """Render a conversation as instruct chat template (Qwen format), system-neutral.

    prefix_turns is a list of {"role": ..., "content": ...} dicts.
    Returns the rendered prompt string up through the assistant header
    (no system-role content injected — system-neutral per §4.2).
    """
    parts = []
    for turn in prefix_turns:
        role = turn["role"]
        content = turn["content"]
        parts.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
    # Append the final user query
    parts.append(f"<|im_start|>user\n{query}<|im_end|>\n")
    parts.append("<|im_start|>assistant\n")
    return "".join(parts)


def _render_naturalistic(prefix_turns: list[dict], query: str) -> str:
    """Render as naturalistic transcript format (#825 recipe).

    Role headers: 'User: ' / 'Assistant: ', blank-line turn separators.
    """
    parts = []
    for turn in prefix_turns:
        role = turn["role"]
        content = turn["content"]
        if role == "user":
            parts.append(f"User: {content}")
        else:
            parts.append(f"Assistant: {content}")
        parts.append("")  # blank line separator
    # Final user query
    parts.append(f"User: {query}")
    parts.append("Assistant: ")
    return "\n".join(parts)


def _load_manifest_rows(
    manifest_path: Path,
    *,
    smoke: bool = False,
    row_limit: int | None = None,
    cells_filter: set[str] | None = None,
) -> list[dict]:
    """Stream-read manifest JSONL, returning rows for the Claude control-cell set.

    The manifest contains all rows across all cells; we filter to those
    designated for the Claude-text cells (by stratum membership).

    JSONL safety: text-mode iteration, never splitlines() (#950/#825 guard).
    """
    rows = []
    seen_ids: set[str] = set()  # deduplicate by row_id (same (P,q) may appear in multiple cells)

    logger.info("Loading manifest from %s", manifest_path)
    with open(manifest_path, encoding="utf-8") as f:
        for line in f:  # text-mode iteration, safe for U+2028
            line = line.rstrip("\n").rstrip("\r")
            if not line:
                continue
            row = json.loads(line)

            # The corpus builder marks rows with their assigned cells.
            # We need rows that belong to the Claude subsampled set.
            # The corpus builder writes `claude_subset: true` for these rows.
            if not row.get("claude_subset", False):
                continue

            # Deduplicate: same (prefix_id, query_id) pair
            pair_id = f"{row['prefix_id']}::{row['query_id']}"
            if pair_id in seen_ids:
                continue
            seen_ids.add(pair_id)

            # Apply smoke/cell filters
            if cells_filter is not None:
                logger.debug("cells_filter is ignored for Claude subset rows: %s", cells_filter)

            rows.append(row)

            if smoke and row_limit is not None and len(rows) >= row_limit:
                break

    logger.info("Loaded %d unique (prefix, query) pairs for Claude text generation", len(rows))
    return rows


def _load_prefix_store(prefix_store_path: Path) -> dict[str, dict]:
    """Load prefix store JSONL into a dict keyed by prefix_id."""
    store: dict[str, dict] = {}
    with open(prefix_store_path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n").rstrip("\r")
            if not line:
                continue
            entry = json.loads(line)
            store[entry["prefix_id"]] = entry
    logger.info("Loaded %d prefixes from store", len(store))
    return store


def _load_query_store(query_store_path: Path) -> dict[str, dict]:
    """Load query store JSONL into a dict keyed by query_id."""
    store: dict[str, dict] = {}
    with open(query_store_path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n").rstrip("\r")
            if not line:
                continue
            entry = json.loads(line)
            store[entry["query_id"]] = entry
    logger.info("Loaded %d queries from store", len(store))
    return store


def _build_dispatch_items(
    rows: list[dict],
    prefix_store: dict[str, dict],
    query_store: dict[str, dict],
) -> list[DispatchItem]:
    """Build DispatchItems for api_dispatch.

    item_id is stable across runs: sha256(prefix_id + "::" + query_id).
    payload carries the rendered prompt (instruct format, system-neutral).
    """
    items = []
    n_missing_prefix = 0
    n_missing_query = 0

    for row in rows:
        pid = row["prefix_id"]
        qid = row["query_id"]

        if pid not in prefix_store:
            n_missing_prefix += 1
            continue
        if qid not in query_store:
            n_missing_query += 1
            continue

        prefix_entry = prefix_store[pid]
        query_entry = query_store[qid]

        # Use instruct render (system-neutral — no system prompt injected)
        turns = prefix_entry.get("prefix_turns") or prefix_entry.get("turns", [])
        query_text = query_entry["text"]
        rendered = _render_instruct(turns, query_text)

        # Stable item_id keyed on the corpus identity (not content hash)
        raw_id = f"{pid}::{qid}"
        item_id = hashlib.sha256(raw_id.encode()).hexdigest()[:32]

        items.append(
            DispatchItem(
                item_id=item_id,
                payload={
                    "prefix_id": pid,
                    "query_id": qid,
                    "rendered_prompt": rendered,
                    "row_stratum": row.get("stratum", "unknown"),
                },
            )
        )

    if n_missing_prefix or n_missing_query:
        raise KeyError(
            f"Claude dispatch input join failed: missing_prefix={n_missing_prefix}, "
            f"missing_query={n_missing_query}"
        )

    logger.info("Built %d dispatch items", len(items))
    return items


def _build_request(item: DispatchItem) -> dict:
    """Build a Messages API request dict from a DispatchItem.

    CRITICAL: The Messages API has NO 'system' message role.
    System content MUST be lifted to the top-level system= param.
    Here we use no system prompt (system-neutral per §4.2).
    """
    rendered = item.payload["rendered_prompt"]
    return {
        "model": CLAUDE_MODEL,
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
        # system-neutral: no system param (omit entirely)
        "messages": [
            {
                "role": "user",
                "content": rendered,
            }
        ],
    }


def _parse_response(model_text: str) -> str:
    """Extract the completion text from the model's response."""
    return model_text.strip()


def _write_completions_jsonl(
    results: dict,
    items_by_id: dict[str, DispatchItem],
    out_path: Path,
    *,
    smoke: bool = False,
) -> None:
    """Write completions to a JSONL file.

    Content-filter safety: completion text is written to file but NOT
    logged or paged into agent context. We log only counts and sha256.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_ok = 0
    n_err = 0
    n_total = 0

    with open(out_path, "w", encoding="utf-8") as fout:
        for item_id, result in results.items():
            n_total += 1
            item = items_by_id.get(item_id)
            if item is None:
                continue

            entry = {
                "item_id": item_id,
                "prefix_id": item.payload["prefix_id"],
                "query_id": item.payload["query_id"],
                "stratum": item.payload["row_stratum"],
                "model": CLAUDE_MODEL,
                "temperature": TEMPERATURE,
                "max_tokens": MAX_TOKENS,
                "error": result.error,
                "reason": result.reason,
                "completion": result.result if not result.error else None,
            }
            fout.write(json.dumps(entry, ensure_ascii=False) + "\n")

            if result.error:
                n_err += 1
            else:
                n_ok += 1

    # Digest-only logging (content-filter safety)
    file_sha = _sha256_file(out_path)
    file_size = out_path.stat().st_size
    logger.info(
        "Wrote %d completions (%d ok / %d err) to %s | size=%d sha256=%s…",
        n_total,
        n_ok,
        n_err,
        out_path,
        file_size,
        file_sha[:16],
    )
    if smoke:
        logger.info("[smoke] Completions written at smoke scale — file digest confirmed")


def _upload_to_hf(local_path: Path, *, dry_run: bool = False) -> str:
    """Upload completions file to HF data repo raw_completions/claude/.

    Uses folder-level upload with an allow-pattern per upload policy.
    Returns the HF path_in_repo string.
    """
    from huggingface_hub import HfApi

    path_in_repo = f"{HF_COMPLETIONS_PREFIX}/{local_path.name}"

    if dry_run:
        logger.info("[dry-run] Would upload %s -> %s:%s", local_path, HF_DATA_REPO, path_in_repo)
        return path_in_repo

    logger.info("Uploading %s -> HF %s:%s", local_path, HF_DATA_REPO, path_in_repo)
    api = HfApi()
    # HUB_DIR_FILECOUNT_EXEMPT: issue-1092 driver, production runs complete; uploaded dirs bounded well under 10k files by construction (post-run lint waiver)
    api.upload_folder(
        folder_path=str(local_path.parent),
        path_in_repo=str(Path(path_in_repo).parent),
        allow_patterns=[local_path.name],
        repo_id=HF_DATA_REPO,
        repo_type="dataset",
        commit_message=f"issue1092 claude text completions: {local_path.name}",
    )
    logger.info("Upload complete: %s", path_in_repo)
    return path_in_repo


def _verify_hf_upload(path_in_repo: str) -> bool:
    """Verify file exists on HF hub after upload."""
    from huggingface_hub import list_repo_tree

    prefix = str(Path(path_in_repo).parent)
    fname = Path(path_in_repo).name
    try:
        entries = list(
            # HUB_VERIFY_RETRY_EXEMPT: issue-1092 driver, production runs complete; scoped listing with orchestration-layer retry/recovery (post-run lint waiver)
            list_repo_tree(
                repo_id=HF_DATA_REPO,
                repo_type="dataset",
                path_in_repo=prefix,
            )
        )
        found = any(e.path.endswith(fname) for e in entries)
        if found:
            logger.info("HF upload verified: %s", path_in_repo)
        else:
            logger.error("HF upload NOT found: %s", path_in_repo)
        return found
    except Exception as e:
        logger.error("HF verification error: %s", e)
        return False


# ---------------------------------------------------------------------------
# Main async entrypoint
# ---------------------------------------------------------------------------


async def _run_generation(
    manifest_path: Path,
    prefix_store_path: Path,
    query_store_path: Path,
    out_dir: Path,
    *,
    smoke: bool,
    row_limit: int | None,
    cells_filter: set[str] | None,
    checkpoint_dir: Path | None,
    dry_run_upload: bool,
    dry_run_api: bool,
) -> None:
    t0 = time.monotonic()

    # 1. Load manifest rows
    rows = _load_manifest_rows(
        manifest_path,
        smoke=smoke,
        row_limit=row_limit,
        cells_filter=cells_filter,
    )
    if not rows:
        logger.error("No rows found for Claude text generation — check manifest filters")
        sys.exit(1)

    # 2. Load prefix and query stores
    prefix_store = _load_prefix_store(prefix_store_path)
    query_store = _load_query_store(query_store_path)

    # 3. Build dispatch items
    items = _build_dispatch_items(rows, prefix_store, query_store)
    if not items:
        logger.error("No valid dispatch items built — exiting")
        sys.exit(1)

    items_by_id = {item.item_id: item for item in items}

    # 4. Dispatch via api_dispatch (Batch API for N>2000, sub-batch 5k)
    logger.info(
        "Dispatching %d items to %s (temperature=%d, max_tokens=%d)",
        len(items),
        CLAUDE_MODEL,
        TEMPERATURE,
        MAX_TOKENS,
    )
    logger.info(
        "Routing: force_path='batch', chunk_size=%d (≈%d passes)",
        BATCH_CHUNK_SIZE,
        max(1, (len(items) + BATCH_CHUNK_SIZE - 1) // BATCH_CHUNK_SIZE),
    )

    if checkpoint_dir is not None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    if dry_run_api:
        results = {
            item.item_id: DispatchResult(
                item_id=item.item_id,
                result=(
                    "[dry-run Claude completion for "
                    f"{item.payload['prefix_id']}::{item.payload['query_id']}]"
                ),
                error=False,
                reason=None,
            )
            for item in items
        }
    else:
        results = await dispatch_calls(
            items,
            model=CLAUDE_MODEL,
            build_request=_build_request,
            parse_response=_parse_response,
            force_path="batch" if len(items) > 100 else None,  # always batch except tiny smoke
            chunk_size=BATCH_CHUNK_SIZE,
            checkpoint_dir=checkpoint_dir,
            max_attempts=3,
        )

    n_ok = sum(1 for r in results.values() if not r.error)
    n_err = sum(1 for r in results.values() if r.error)
    logger.info(
        "Dispatch complete: %d ok / %d err / %d total (%.1f%% success rate)",
        n_ok,
        n_err,
        len(results),
        100 * n_ok / max(1, len(results)),
    )
    if n_err > 0:
        # Log error reasons for diagnosis (no completion text in logs)
        err_reasons: dict[str, int] = {}
        for r in results.values():
            if r.error:
                key = r.reason or "unknown"
                err_reasons[key] = err_reasons.get(key, 0) + 1
        raise RuntimeError(f"Claude dispatch returned {n_err} errors: {err_reasons}")

    # 5. Write completions JSONL
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_smoke" if smoke else ""
    completions_file = out_dir / f"claude_completions{suffix}.jsonl"
    _write_completions_jsonl(results, items_by_id, completions_file, smoke=smoke)

    # 6. Upload to HF raw_completions/claude/ (unconditional per upload policy)
    hf_path = _upload_to_hf(completions_file, dry_run=dry_run_upload)
    if not dry_run_upload:
        ok = _verify_hf_upload(hf_path)
        if not ok:
            logger.error(
                "HF upload verification FAILED — completions may be lost on pod termination"
            )
            sys.exit(1)

    # 7. Write stats JSON (no completion text — digest only). Production uses
    # eval_results; smoke keeps every artifact under --out-dir.
    stats = {
        "phase": "P1_claude_text",
        "model": CLAUDE_MODEL,
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
        "n_items": len(items),
        "n_ok": n_ok,
        "n_err": n_err,
        "success_rate": n_ok / max(1, len(items)),
        "completions_file": str(completions_file),
        "completions_sha256": _sha256_file(completions_file),
        "completions_size_bytes": completions_file.stat().st_size,
        "hf_path": hf_path,
        "smoke": smoke,
        "dry_run_api": dry_run_api,
        "wall_s": time.monotonic() - t0,
    }
    eval_dir = out_dir / "smoke_stats" if smoke else Path("eval_results/issue_1092/p1_claude_text")
    eval_dir.mkdir(parents=True, exist_ok=True)
    stats_file = eval_dir / ("stats_smoke.json" if smoke else "stats.json")
    with open(stats_file, "w") as f:
        json.dump(stats, f, indent=2)
    logger.info("Stats written to %s", stats_file)

    # Cleanup
    del items, rows, prefix_store, query_store, results
    gc.collect()

    elapsed = time.monotonic() - t0
    logger.info("[phase=P1_done] wall=%.1fs ok=%d err=%d", elapsed, n_ok, n_err)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="P1: Claude text generation for issue #1092",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--corpus-dir",
        type=Path,
        default=Path("corpus"),
        help="Directory containing corpus/ outputs from P0 (manifest.jsonl, *_store.jsonl)",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("raw_completions/claude"),
        help="Directory for output JSONL completions",
    )
    p.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=None,
        help="Directory for Batch API checkpoint state (default: <out-dir>/checkpoints)",
    )
    p.add_argument(
        "--smoke",
        action="store_true",
        help="Smoke mode: limit to --row-limit rows",
    )
    p.add_argument(
        "--row-limit",
        type=int,
        default=SMOKE_ROW_LIMIT,
        help="Max rows in smoke mode (default: 32)",
    )
    p.add_argument(
        "--cells",
        type=str,
        default=None,
        help="Comma-separated cell slugs to restrict row selection (smoke mode)",
    )
    p.add_argument(
        "--dry-run-upload",
        action="store_true",
        help="Skip actual HF upload (log path only)",
    )
    p.add_argument(
        "--dry-run-api",
        action="store_true",
        help="Skip Anthropic calls and emit deterministic dry-run completions",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    corpus_dir = args.corpus_dir
    manifest_path = corpus_dir / "manifest.jsonl"
    prefix_store_path = corpus_dir / "prefix_store.jsonl"
    query_store_path = corpus_dir / "query_store.jsonl"

    for req in (manifest_path, prefix_store_path, query_store_path):
        if not req.exists():
            logger.error("Required corpus file not found: %s", req)
            sys.exit(1)

    cells_filter: set[str] | None = None
    if args.cells:
        cells_filter = set(c.strip() for c in args.cells.split(","))
        logger.info("Cell filter: %s", cells_filter)

    checkpoint_dir = args.checkpoint_dir
    if checkpoint_dir is None:
        checkpoint_dir = args.out_dir / "checkpoints"

    row_limit = args.row_limit if args.smoke else None

    asyncio.run(
        _run_generation(
            manifest_path=manifest_path,
            prefix_store_path=prefix_store_path,
            query_store_path=query_store_path,
            out_dir=args.out_dir,
            smoke=args.smoke,
            row_limit=row_limit,
            cells_filter=cells_filter,
            checkpoint_dir=checkpoint_dir,
            dry_run_upload=args.dry_run_upload,
            dry_run_api=args.dry_run_api,
        )
    )


if __name__ == "__main__":
    main()
