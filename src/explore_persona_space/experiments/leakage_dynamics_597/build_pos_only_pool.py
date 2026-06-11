# ruff: noqa: RUF003  # research code uses Greek letters, ×, ∪, − and ※ legitimately
"""Phase P (#597): order-preserving 200-positive filter of the #480 train pools.

THE manipulated variable of #597: Arm B trains on the positive rows ONLY of
#480's 700-row contrastive pools (negatives REMOVED, not replaced). Rows are
kept in their ORIGINAL file order so ``build_source_probe_from_data`` —
which picks the FIRST ``max_rows`` marker-bearing rows in file order —
selects the SAME 32 in-loop probe rows for Arm B that Arm A's training saw
(plan Phase B-train probe-row identity check, BLOCKING).

A positive row is identified by its final completion message text ending
with the marker (`` ※``) — the same predicate the #480 pool builder used to
construct them (``build_marker_pool`` appends ``marker_text`` to the source
persona's rows only).
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

from explore_persona_space.experiments.leakage_dynamics_597 import (
    EXPECTED_FULL_POOL_ROWS,
    EXPECTED_POS_ROWS,
    MARKER_TEXT,
)

logger = logging.getLogger(__name__)


def _final_completion_text(row: dict) -> str | None:
    """Return the content of the LAST completion message, or None if malformed."""
    completion = row.get("completion")
    if not isinstance(completion, list) or not completion:
        return None
    last = completion[-1]
    if not isinstance(last, dict):
        return None
    content = last.get("content")
    return content if isinstance(content, str) else None


def filter_positive_rows(rows: list[dict], marker_text: str = MARKER_TEXT) -> list[dict]:
    """Order-preserving filter: keep rows whose final completion ends with the marker.

    Args:
        rows: parsed JSONL rows ({"prompt": [...], "completion": [...]}).
        marker_text: marker string (default `` ※``).

    Returns:
        The marker-bearing subset, in the ORIGINAL order (load-bearing — see
        module docstring).
    """
    out: list[dict] = []
    for row in rows:
        text = _final_completion_text(row)
        if text is not None and text.endswith(marker_text):
            out.append(row)
    return out


def build_pos_only_pool(
    in_pool: Path,
    out_pool: Path,
    *,
    marker_text: str = MARKER_TEXT,
    expected_in_rows: int = EXPECTED_FULL_POOL_ROWS,
    expected_out_rows: int = EXPECTED_POS_ROWS,
) -> dict:
    """Filter ``in_pool`` (700-row contrastive) to the positive-only Arm B pool.

    Fail-loud on row-count drift in EITHER direction (a short input means the
    wrong artifact resolved; a non-200 output means the marker predicate or
    the pool changed under us).

    Returns:
        summary dict: ``{"n_in", "n_out", "out_path", "sha256"}`` where sha256
        is over the output file bytes (reproducibility metadata).
    """
    rows: list[dict] = []
    with open(in_pool) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if len(rows) != expected_in_rows:
        raise RuntimeError(
            f"input pool {in_pool} has {len(rows)} rows, expected {expected_in_rows} — "
            "wrong artifact resolved (check the pinned revision)."
        )
    positives = filter_positive_rows(rows, marker_text)
    if len(positives) != expected_out_rows:
        raise RuntimeError(
            f"positive filter of {in_pool} yielded {len(positives)} rows, expected "
            f"{expected_out_rows} — the marker predicate or the pool composition drifted."
        )
    out_pool.parent.mkdir(parents=True, exist_ok=True)
    with open(out_pool, "w") as f:
        for row in positives:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    digest = hashlib.sha256(out_pool.read_bytes()).hexdigest()
    logger.info(
        "build_pos_only_pool: %s (%d rows) -> %s (%d rows, sha256=%s)",
        in_pool,
        len(rows),
        out_pool,
        len(positives),
        digest[:16],
    )
    return {
        "n_in": len(rows),
        "n_out": len(positives),
        "out_path": str(out_pool),
        "sha256": digest,
    }


def probe_row_token_hash(
    pool_path: Path,
    tokenizer,
    marker_token_ids: list[int],
    *,
    max_rows: int = 32,
    max_length: int = 2560,
) -> tuple[str, int]:
    """Token-level fingerprint of the in-loop probe rows a pool would select.

    Runs the SAME selection ``train_lora``'s band callback wiring uses
    (``build_source_probe_from_data``: first ``max_rows`` marker-bearing rows
    in file order, fused chat-template render) and hashes the exact token-id
    sequences + marker slots.

    Returns:
        ``(sha256_hex, n_rows)``.
    """
    from explore_persona_space.train.sft import build_source_probe_from_data

    input_ids, _attn, marker_positions, n_rows = build_source_probe_from_data(
        pool_path,
        tokenizer,
        list(marker_token_ids),
        max_rows=max_rows,
        max_length=max_length,
    )
    if n_rows == 0:
        raise RuntimeError(f"no marker-bearing probe rows found in {pool_path}")
    h = hashlib.sha256()
    for i in range(n_rows):
        row = input_ids[i].tolist()
        slot = int(marker_positions[i].item())
        # Hash only the REAL tokens (strip right padding): the padded width
        # depends on the batch's max length, which differs between a 700-row
        # and a 200-row pool even when the selected rows are identical.
        real = row[: slot + 1 + len(marker_token_ids)]
        h.update(json.dumps({"ids": real, "slot": slot}).encode())
    return h.hexdigest(), n_rows


def assert_probe_row_identity(
    full_pool: Path,
    pos_pool: Path,
    tokenizer,
    marker_token_ids: list[int],
    *,
    max_rows: int = 32,
    max_length: int = 2560,
) -> str:
    """BLOCKING probe-row identity assert (plan Phase B-train, one governing status).

    The 32 in-loop probe rows ``build_source_probe_from_data`` selects from
    Arm B's filtered pool must be TOKEN-IDENTICAL to the selection from the
    parent 700-row pool (Arm A's in-loop reference). Mismatch raises — fix
    the filter, never proceed (a drifted probe batch defeats the matched
    in-loop comparison AND the Gate S re-application on Arm B).

    Returns:
        the shared sha256 fingerprint on success.
    """
    full_hash, full_n = probe_row_token_hash(
        full_pool, tokenizer, marker_token_ids, max_rows=max_rows, max_length=max_length
    )
    pos_hash, pos_n = probe_row_token_hash(
        pos_pool, tokenizer, marker_token_ids, max_rows=max_rows, max_length=max_length
    )
    if full_n != pos_n or full_hash != pos_hash:
        raise RuntimeError(
            "BLOCKING probe-row identity FAILURE: the in-loop probe batch selected from "
            f"the positive-only pool ({pos_pool}: n={pos_n}, sha256={pos_hash[:16]}) does "
            f"not token-match the parent pool's selection ({full_pool}: n={full_n}, "
            f"sha256={full_hash[:16]}). The 200-positive filter must be order-preserving "
            "and content-identical; fix the filter before training Arm B."
        )
    logger.info(
        "probe-row identity OK: %d rows, sha256=%s (pools %s vs %s)",
        full_n,
        full_hash[:16],
        full_pool.name,
        pos_pool.name,
    )
    return full_hash
