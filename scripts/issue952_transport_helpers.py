"""#1470: rule-24 caller-side helpers for the #952 generation dispatchers.

Pure helpers shared by ``issue952_divtrain_gpu.py`` (``response_valid``
validator + one bounded re-drive + per-row failure records) and
``issue952_bank_build.py`` (validator kwarg only). Kept in a standalone module
so they are unit-testable without importing the vLLM-heavy GPU scripts
(``tests/test_issue952_divtrain_transport.py``) and so the two callers cannot
drift on the validator definition.
"""

from __future__ import annotations

import json
import pathlib
from typing import Any

from explore_persona_space.llm.api_dispatch import (
    RESULT_RATE_LIMITED,
    RESULT_TRANSPORT,
)

# Re-drivable (transport-class) categories per llm-judging.md rule 24 / #1313:
# the caller re-drives these ONCE; RESULT_ERROR stays terminal (content-class).
REDRIVABLE_CATEGORIES = frozenset({RESULT_TRANSPORT, RESULT_RATE_LIMITED})


def nonempty_text(parsed: Any) -> bool:
    """Validator: a usable generation is a non-empty, non-whitespace str."""
    return isinstance(parsed, str) and bool(parsed.strip())


def redrivable_ids(results: dict) -> list[str]:
    """item_ids of error rows in a re-drivable (transport-class) category."""
    return [
        iid
        for iid, res in results.items()
        if getattr(res, "error", False) and getattr(res, "category", None) in REDRIVABLE_CATEGORIES
    ]


def failure_rows(
    results: dict, ordered_ids: list[str], redriven: set[str]
) -> tuple[list[dict], dict]:
    """Per-row failure records + the rule-24 transport-vs-content split counts.

    Returns ``(records, counts)`` where ``records`` rows are
    ``{"item_id", "category", "reason", "round": "redrive"|"initial"}`` for
    every still-failed id, and ``counts`` =
    ``{"n_ok", "n_transport_class", "n_content_class"}``. A missing result
    counts as content-class with category ``"missing"``. ``round`` records
    whether the row's LAST attempt was the bounded re-drive (id in
    ``redriven``) or the initial dispatch.
    """
    records: list[dict] = []
    n_ok = n_transport = n_content = 0
    for iid in ordered_ids:
        res = results.get(iid)
        rnd = "redrive" if iid in redriven else "initial"
        if res is None:
            records.append(
                {
                    "item_id": iid,
                    "category": "missing",
                    "reason": "no result returned",
                    "round": rnd,
                }
            )
            n_content += 1
            continue
        if not getattr(res, "error", False):
            n_ok += 1
            continue
        category = getattr(res, "category", None)
        records.append(
            {
                "item_id": iid,
                "category": category,
                "reason": getattr(res, "reason", None),
                "round": rnd,
            }
        )
        if category in REDRIVABLE_CATEGORIES:
            n_transport += 1
        else:
            n_content += 1
    counts = {"n_ok": n_ok, "n_transport_class": n_transport, "n_content_class": n_content}
    return records, counts


def write_failure_records(path: pathlib.Path | str, records: list[dict], counts: dict) -> None:
    """Write the per-row failure-records JSON — ALWAYS written, even when empty.

    An empty ``failures`` list is the auditable zero-loss record — never skip
    the write (#1470 acceptance criterion 5). The ``invalid_response`` reason
    prefix in the per-row ``reason`` fields is the auditable separator between
    invalid-content rows and true transport rows inside the transport-class
    tally — keep it stable.
    """
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"counts": counts, "failures": records}, indent=2))
