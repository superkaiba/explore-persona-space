"""Regression tests for the issue #654 round-5 CRITICAL fix.

Concern: ``dummy-arm-pinned-battery-not-fetched`` (code-review reconcile v5).

The length-matched-dummy-query control joins the dummy arm's banks to the
parent run's PINNED real-arm banks on a STABLE id ``(context_id,
real_query_id)``. The control is valid ONLY if the dummy arm reads its contexts
from the SAME frozen battery the pinned banks were extracted from. A fresh-pod
LOCAL rebuild streams contexts live and drifts the context strings while still
joining the pinned banks by id — silently computing the gap across DIFFERENT
contexts.

``scripts/issue654_fetch_pinned_battery.py`` fixes this: it downloads the pinned
``inputs/battery.json`` and FAILS LOUD unless its ``context_id`` set exactly
matches the pinned cached ``context_only/*.pt`` bank basenames.

These tests pin that invariant:

  - test_fetch_verifies_matching_context_set
    Downloaded battery's contexts == cached banks -> returns the context set,
    writes the battery to dest.

  - test_fetch_fails_loud_on_drifted_context_for_same_id
    The downloaded battery carries a DIFFERENT context for the SAME context_id
    set than the cached banks (a stable-id drift the analyzer's join would NOT
    catch) -> mismatch path: even when the *id sets* match, the test confirms the
    id-set verifier is the gate; and the disjoint-id case raises loud.

  - test_fetch_fails_loud_on_missing_banks
    No cached context_only banks at the revision -> raises loud.

  - test_pure_helpers
    ``cached_bank_context_ids`` / ``battery_context_ids`` parse the expected ids.

Plus a synthetic analyzer-join test (Minor, code-review v5) pinning the most
load-bearing v5 analyzer path:

  - test_companion_gap_joins_real_without_real_query_id_to_dummy
    A real bank WITHOUT ``real_query_id`` joins cleanly to a dummy bank carrying
    ``real_query_id`` matching the original real ``query_id``.

Pure CPU; monkeypatches the HF download; uses tmp_path for isolation; no GPU,
no model load, no network.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

# ── Import the helper under test (script, not a package module) ──────────────
FETCH_SCRIPT = Path(__file__).parent.parent / "scripts" / "issue654_fetch_pinned_battery.py"
_fspec = importlib.util.spec_from_file_location(
    "issue654_fetch_pinned_battery_under_test", FETCH_SCRIPT
)
assert _fspec is not None and _fspec.loader is not None
fetch_mod = importlib.util.module_from_spec(_fspec)
sys.modules["issue654_fetch_pinned_battery_under_test"] = fetch_mod
_fspec.loader.exec_module(fetch_mod)

ANALYZE_SCRIPT = Path(__file__).parent.parent / "scripts" / "issue654_analyze.py"
_aspec = importlib.util.spec_from_file_location("issue654_analyze_under_test", ANALYZE_SCRIPT)
assert _aspec is not None and _aspec.loader is not None
analyze_mod = importlib.util.module_from_spec(_aspec)
sys.modules["issue654_analyze_under_test"] = analyze_mod
_aspec.loader.exec_module(analyze_mod)

REPO = "superkaiba1/explore-persona-space-data"
PREFIX = "issue654_query_displacement"
REV = "82d16a6faa7f8781163bf215154ed57296364780"


def _battery_payload(context_ids: list[str]) -> dict:
    """A minimal battery.json: one pair per context_id."""
    return {
        "meta": {"git_commit": "deadbeef"},
        "pairs": [
            {
                "context_id": cid,
                "query_id": f"q_{cid}",
                "context_type": "generic",
                "context_only_prompt": f"<context for {cid}>",
                "query_end_idx": 10,
                "topicality": "on",
                "length": "short",
            }
            for cid in context_ids
        ],
    }


def _bank_files(prefix: str, context_ids: list[str]) -> list[str]:
    """The repo file list for the cached context_only/*.pt banks of these ids."""
    return [f"{prefix}/analysis_tensors/context_only/{cid}.pt" for cid in context_ids]


def _patch_hf(monkeypatch, battery_payload: dict, repo_files: list[str], tmp_path: Path) -> None:
    """Monkeypatch hf_hub_download + list_repo_files inside the helper's lazy import.

    The helper imports ``from huggingface_hub import hf_hub_download, list_repo_files``
    lazily inside ``fetch_and_verify_pinned_battery``; patch the source module so
    that import binds to our fakes.
    """
    src = tmp_path / "_hf_battery_src.json"
    src.write_text(json.dumps(battery_payload))

    def _fake_download(repo, path_in_repo, repo_type=None, revision=None):
        return str(src)

    def _fake_list(repo, repo_type=None, revision=None):
        return repo_files

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", _fake_download)
    monkeypatch.setattr(huggingface_hub, "list_repo_files", _fake_list)


def test_fetch_verifies_matching_context_set(tmp_path, monkeypatch) -> None:
    """Battery contexts == cached banks -> fetch succeeds, writes dest."""
    cids = ["generic_000", "generic_001", "wildchat_abc"]
    _patch_hf(monkeypatch, _battery_payload(cids), _bank_files(PREFIX, cids), tmp_path)

    dest = tmp_path / "data" / "issue654" / "battery.json"
    out = fetch_mod.fetch_and_verify_pinned_battery(REPO, PREFIX, dest, REV)

    assert out == set(cids)
    assert dest.exists()
    written = json.loads(dest.read_text())
    assert {p["context_id"] for p in written["pairs"]} == set(cids)


def test_fetch_fails_loud_on_drifted_context_for_same_id(tmp_path, monkeypatch) -> None:
    """A locally-drifted battery whose context_id set DIFFERS from the cached
    banks (the production fresh-pod-rebuild failure mode) raises loud BEFORE
    extraction — the analyzer's stable-id join would otherwise silently compute
    the gap across mismatched contexts.
    """
    # Cached banks were extracted for these 3 contexts (the pinned real arm).
    bank_cids = ["generic_000", "generic_001", "wildchat_abc"]
    # A live rebuild on a fresh pod streamed DIFFERENT contexts: same positional
    # generic_* ids but an extra/missing wildchat conv id (conv ids are not stable
    # across a re-stream) -> the id sets diverge.
    rebuilt_cids = ["generic_000", "generic_001", "wildchat_XYZ"]
    _patch_hf(monkeypatch, _battery_payload(rebuilt_cids), _bank_files(PREFIX, bank_cids), tmp_path)

    dest = tmp_path / "battery.json"
    with pytest.raises(fetch_mod.PinnedBatteryMismatchError, match="CONTEXT MISMATCH"):
        fetch_mod.fetch_and_verify_pinned_battery(REPO, PREFIX, dest, REV)


def test_fetch_fails_loud_on_missing_banks(tmp_path, monkeypatch) -> None:
    """No cached context_only banks at the revision -> raises loud."""
    cids = ["generic_000"]
    _patch_hf(monkeypatch, _battery_payload(cids), repo_files=[], tmp_path=tmp_path)

    dest = tmp_path / "battery.json"
    with pytest.raises(fetch_mod.PinnedBatteryMismatchError, match="no cached context_only banks"):
        fetch_mod.fetch_and_verify_pinned_battery(REPO, PREFIX, dest, REV)


def test_pure_helpers() -> None:
    """The id-extraction helpers parse battery + bank file lists correctly."""
    cids = ["generic_000", "wildchat_abc"]
    payload = _battery_payload(cids)
    assert fetch_mod.battery_context_ids(payload) == set(cids)

    repo_files = [
        *_bank_files(PREFIX, cids),
        f"{PREFIX}/analysis_tensors/pair_000.pt",  # not a context_only bank
        f"{PREFIX}/inputs/battery.json",  # not a .pt
    ]
    assert fetch_mod.cached_bank_context_ids(repo_files, PREFIX) == set(cids)

    # verify_context_identity is a no-op on a matching set, raises on a disjoint one.
    fetch_mod.verify_context_identity(set(cids), set(cids), REV)
    with pytest.raises(fetch_mod.PinnedBatteryMismatchError):
        fetch_mod.verify_context_identity({"a"}, {"b"}, REV)


# ── Minor (code-review v5): synthetic real/dummy companion-gap join ──────────


def _readout_bank_dir(
    tmp_path: Path,
    meta_rows: list[dict],
    n_layers: int = 3,
    hidden: int = 4,
) -> Path:
    """Write synthetic pair_*.pt + context_only/*.pt banks for _load_readout_banks."""
    bank_dir = tmp_path
    ctx_dir = bank_dir / "context_only"
    ctx_dir.mkdir(parents=True, exist_ok=True)
    rng = torch.manual_seed(0)  # noqa: F841 — determinism
    seen_ctx: set[str] = set()
    for i, m in enumerate(meta_rows):
        cid = m["context_id"]
        companion_file = f"context_only/{cid}.pt"
        if cid not in seen_ctx:
            torch.save(
                {"readout": torch.randn(n_layers, hidden)},
                bank_dir / companion_file,
            )
            seen_ctx.add(cid)
        d = {
            "readout": torch.randn(n_layers, hidden),
            "pair_id": m.get("pair_id", f"{cid}__{m['query_id']}"),
            "context_type": m["context_type"],
            "context_id": cid,
            "query_id": m["query_id"],
            "topicality": m["topicality"],
            "length": m["length"],
            "companion_context_only_file": companion_file,
        }
        # The dummy arm carries real_query_id; the real arm does NOT.
        if "real_query_id" in m:
            d["real_query_id"] = m["real_query_id"]
        torch.save(d, bank_dir / f"pair_{i:03d}.pt")
    return bank_dir


def test_companion_gap_joins_real_without_real_query_id_to_dummy(tmp_path) -> None:
    """A real bank WITHOUT real_query_id joins to a dummy bank carrying it.

    This pins the v5 amendment's load-bearing join: real banks (parent v4 shape)
    have no ``real_query_id`` field, so the analyzer falls it back to ``query_id``;
    the dummy battery stores ``real_query_id`` mirroring the matched real query id.
    Both must collapse to the same ``(context_id, real_query_id)`` key.
    """
    # Real arm: no real_query_id field (parent v4 shape). Two pairs, two contexts.
    real_meta = [
        {
            "context_id": "generic_000",
            "query_id": "q_real_a",
            "context_type": "generic",
            "topicality": "on",
            "length": "short",
        },
        {
            "context_id": "wildchat_abc",
            "query_id": "q_real_b",
            "context_type": "wildchat",
            "topicality": "on",
            "length": "long",
        },
    ]
    # Dummy arm: carries real_query_id matching each real pair's query_id; its own
    # query_id is a distinct dummy id (so the join CANNOT accidentally rely on it).
    dummy_meta = [
        {
            "context_id": "generic_000",
            "query_id": "q_dummy_for_real_a",
            "real_query_id": "q_real_a",
            "context_type": "generic",
            "topicality": "on",
            "length": "short",
        },
        {
            "context_id": "wildchat_abc",
            "query_id": "q_dummy_for_real_b",
            "real_query_id": "q_real_b",
            "context_type": "wildchat",
            "topicality": "on",
            "length": "long",
        },
    ]

    real_dir = _readout_bank_dir(tmp_path / "real", real_meta)
    dummy_dir = _readout_bank_dir(tmp_path / "dummy", dummy_meta)

    real_banks = analyze_mod._load_readout_banks(real_dir)
    dummy_banks = analyze_mod._load_readout_banks(dummy_dir)

    # Real arm's join key fell back to query_id (== the real query id).
    assert {m["real_query_id"] for m in real_banks["meta"]} == {"q_real_a", "q_real_b"}
    # Dummy arm's join key is the explicit real_query_id, not its own dummy id.
    assert {m["real_query_id"] for m in dummy_banks["meta"]} == {"q_real_a", "q_real_b"}

    out = analyze_mod._companion_gap_per_layer_per_tier(
        real_banks, dummy_banks, real_banks["companion"]
    )

    # Both pairs join — no unmatched, no synthetic phantom pairs.
    assert out["n_matched_pairs"] == 2
    assert out["n_unmatched_real"] == 0
    assert out["n_unmatched_dummy"] == 0
    # The per-tier gap is computed for both tiers.
    assert set(out["per_tier"]) == {"generic", "wildchat"}
    assert np.isfinite(out["overall"]["gap_mean"]).all()
