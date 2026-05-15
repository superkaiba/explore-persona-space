"""Round-trip integration test for the task #365 metrics.json -> aggregator path.

Round-1 code-review BLOCKER 1 was a schema mismatch: ``__main__.py``
wrote nested ``persona_panel_scores`` (a dict-of-dict keyed by persona
name), but ``aggregator.py::_record_from_metrics_json`` read flat keys
``source_substring_rate`` / ``leakage_rate_*`` / ``per_bystander_substring_rates``.
Every cell silently aggregated to 0.0 with NO RUNTIME ERROR.

These tests synthesise a fake ``metrics.json`` in the exact shape that
``_run_cell_mode`` writes via ``_flat_metrics_from_panel``, run it
through ``_record_from_metrics_json`` directly, then through the full
``load_records_from_disk`` directory-tree walk, and assert that:

  * Non-zero ``source_substring_rate`` round-trips intact.
  * All 24 panel personas appear in ``per_bystander_substring_rates``.
  * ``leakage_rate_full`` / ``leakage_rate_out_of_domain`` /
    ``leakage_rate_in_domain`` reconcile with the source's in-domain
    bystander allowlist.
  * The directory walker finds the synthesised cells (not zero records,
    which was the round-1 directory-layout failure mode).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from explore_persona_space.experiments.factor_screen_365 import (
    EVAL_PERSONAS_24,
    IN_DOMAIN_BYSTANDERS_BY_SOURCE,
    SOURCE_PERSONAS,
)
from explore_persona_space.experiments.factor_screen_365.__main__ import (
    _flat_metrics_from_panel,
)
from explore_persona_space.experiments.factor_screen_365.aggregator import (
    _record_from_metrics_json,
    load_records_from_disk,
)


def _fake_persona_panel_scores(
    source: str,
    *,
    source_rate: float,
    bystander_rate: float,
    in_domain_rate: float,
) -> dict[str, dict]:
    """Mimic the nested output shape of ``score_markers``."""
    in_domain = IN_DOMAIN_BYSTANDERS_BY_SOURCE[source]
    out: dict[str, dict] = {}
    for persona in EVAL_PERSONAS_24:
        if persona == source:
            rate = source_rate
        elif persona in in_domain:
            rate = in_domain_rate
        else:
            rate = bystander_rate
        out[persona] = {
            "substring_rate": rate,
            "fuzzy_rate": rate,
            "substring_found": int(rate * 100),
            "fuzzy_found": int(rate * 100),
            "total": 100,
            "per_question": {},
        }
    return out


def _fake_random_control_scores(*, mean_rate: float) -> dict[str, dict]:
    """Mimic the nested output of ``score_markers`` for the 24-prompt random panel."""
    return {
        f"random_control_{i:02d}": {
            "substring_rate": mean_rate,
            "fuzzy_rate": mean_rate,
            "substring_found": 0,
            "fuzzy_found": 0,
            "total": 100,
            "per_question": {},
        }
        for i in range(1, 25)
    }


def test_flat_metrics_round_trip_for_surgeon() -> None:
    """Source rate + all 24 personas survive the schema bridge.

    Surgeon is the most informative source because it has both in-domain
    bystanders (``medical_doctor``) AND sibling sources (``librarian``,
    ``programmer``).
    """
    persona_scores = _fake_persona_panel_scores(
        "surgeon",
        source_rate=0.42,
        bystander_rate=0.10,
        in_domain_rate=0.25,
    )
    random_scores = _fake_random_control_scores(mean_rate=0.05)
    flat = _flat_metrics_from_panel(
        source="surgeon",
        persona_panel_scores=persona_scores,
        random_control_scores=random_scores,
    )

    # Bridge produces non-zero source rate.
    assert flat["source_substring_rate"] == pytest.approx(0.42)

    # All 24 panel personas appear in the per-bystander map (the source is
    # included for completeness — consumers filter it out).
    per_bystander = flat["per_bystander_substring_rates"]
    assert isinstance(per_bystander, dict)
    assert len(per_bystander) == 24
    assert set(per_bystander.keys()) == set(EVAL_PERSONAS_24.keys())

    # In-domain stratification places medical_doctor in the in-domain subset.
    assert per_bystander["medical_doctor"] == pytest.approx(0.25)
    assert flat["leakage_rate_in_domain"] == pytest.approx(0.25)

    # Random control fields propagate.
    assert flat["mean_random_control_rate"] == pytest.approx(0.05)
    assert flat["max_random_control_rate"] == pytest.approx(0.05)


def test_flat_metrics_consumed_by_record_loader(tmp_path: Path) -> None:
    """Persist a metrics.json and assert ``_record_from_metrics_json`` reads it.

    This is the exact path the aggregator walks: per-cell ``metrics.json``
    written by ``_run_cell_mode``, parsed back into a ``CellRecord``.
    """
    persona_scores = _fake_persona_panel_scores(
        "librarian",
        source_rate=0.7,
        bystander_rate=0.08,
        in_domain_rate=0.0,
    )
    flat = _flat_metrics_from_panel(
        source="librarian",
        persona_panel_scores=persona_scores,
        random_control_scores=_fake_random_control_scores(mean_rate=0.02),
    )
    payload = {
        "cell_key": "00000",
        "bits": [0, 0, 0, 0, 0],
        "source": "librarian",
        "seed": 42,
        **flat,
        "failed": False,
    }
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(json.dumps(payload))

    record = _record_from_metrics_json(metrics_path)
    assert record is not None
    assert record.failed is False
    assert record.source == "librarian"
    assert record.cell_key == "00000"
    assert record.source_rate == pytest.approx(0.7)
    assert record.leakage_rate_full > 0.0
    # All 24 personas survive, including the source itself.
    assert len(record.per_bystander_rates) == 24
    assert record.mean_random_control_rate == pytest.approx(0.02)


def test_directory_layout_round_trip(tmp_path: Path) -> None:
    """Synthesise a slab tree and confirm ``load_records_from_disk`` finds it.

    Round-1 BLOCKER 2 was: ``__main__`` wrote
    ``cell_<key>/source_<src>/seed_<N>/`` but ``load_records_from_disk``
    walked ``<source>/cell_*/seed_*/``. The directory tree returned EMPTY,
    silently. This test asserts the new uniform layout is discovered.
    """
    slab_root = tmp_path / "slab"

    # Place three cells, one per source, at the plan-canonical layout.
    seeded = []
    for cell_key, source, rate in [
        ("00000", "librarian", 0.70),
        ("01010", "surgeon", 0.55),
        ("11111", "programmer", 0.30),
    ]:
        seed_dir = slab_root / f"cell_{cell_key}" / f"source_{source}" / "seed_42"
        seed_dir.mkdir(parents=True, exist_ok=True)
        scores = _fake_persona_panel_scores(
            source, source_rate=rate, bystander_rate=0.05, in_domain_rate=0.10
        )
        flat = _flat_metrics_from_panel(
            source=source,
            persona_panel_scores=scores,
            random_control_scores=_fake_random_control_scores(mean_rate=0.02),
        )
        payload = {
            "cell_key": cell_key,
            "bits": [int(b) for b in cell_key],
            "source": source,
            "seed": 42,
            **flat,
            "failed": False,
        }
        (seed_dir / "metrics.json").write_text(json.dumps(payload))
        seeded.append((source, cell_key, rate))

    records = load_records_from_disk(slab_root)

    # Exactly the three seeded sources should appear, each carrying one cell.
    assert set(records.keys()) == set(SOURCE_PERSONAS)
    for source, cell_key, rate in seeded:
        assert cell_key in records[source]
        loaded = records[source][cell_key]
        assert loaded.source_rate == pytest.approx(rate)
        # Critically: the per-bystander map round-trips with all 24 personas.
        assert len(loaded.per_bystander_rates) == 24
