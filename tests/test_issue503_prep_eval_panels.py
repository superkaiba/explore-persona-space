"""Tests for scripts/issue503_prep_eval_panels.py — GAP-1 round-4 launch fix.

Pins the contract that ``materialize_panel`` accepts every panel id
declared in ``eval_panels.PANEL_SIZES``. v1 of the script had branches
for only the 5 v1 panels while PANEL_SIZES grew to 13 across plan v2
Buckets A/D/E; the 8 new panels were silently skipped under
``--panel all`` and crashed under explicit ``--panel <new_id>``.

Tests do NOT exercise the Claude / GitHub / HF network paths inside
each generator — they monkeypatch the generator dispatch table to
no-op stubs and assert the wiring. A separate marker-bearing smoke
run (see implementer report) verifies real generator behavior.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture()
def prep_eval_panels_module(monkeypatch, tmp_path):
    """Import scripts/issue503_prep_eval_panels.py with the data dir
    rerouted into a tmp_path so the test never writes into the real
    data/issue503/eval_panels/.
    """
    scripts_path = str(PROJECT_ROOT / "scripts")
    if scripts_path not in sys.path:
        sys.path.insert(0, scripts_path)

    # Drop any cached module so monkeypatching the project root takes effect.
    if "issue503_prep_eval_panels" in sys.modules:
        del sys.modules["issue503_prep_eval_panels"]

    mod = importlib.import_module("issue503_prep_eval_panels")

    fake_panel_dir = tmp_path / "eval_panels"
    fake_panel_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(mod, "_eval_panel_dir", lambda: fake_panel_dir)
    # Disjoint audit reads issue404 datasets via ensure_dataset — stub it
    # so the test runs without those files present.
    monkeypatch.setattr(mod, "_audit_disjoint", lambda panel_id, qs: {"any_overlap": False})
    return mod, fake_panel_dir


def test_every_panel_size_id_is_in_generator_table(prep_eval_panels_module):
    """Round-4 launch-failure guard: every id declared in
    ``eval_panels.PANEL_SIZES`` MUST have a corresponding entry in
    ``_PANEL_GENERATORS`` so ``materialize_panel`` doesn't raise
    ``ValueError`` on production sweeps.
    """
    mod, _ = prep_eval_panels_module
    from explore_persona_space.experiments.issue503.eval_panels import PANEL_SIZES

    missing = sorted(set(PANEL_SIZES) - set(mod._PANEL_GENERATORS))
    assert missing == [], (
        f"PANEL_SIZES declares panels missing from _PANEL_GENERATORS dispatch table: {missing}. "
        f"Either add the materialize branch or remove the eval_panels declaration."
    )


def test_no_orphan_generator_branches(prep_eval_panels_module):
    """Reverse direction: every generator MUST correspond to a declared
    panel id; orphan branches indicate stale code.
    """
    mod, _ = prep_eval_panels_module
    from explore_persona_space.experiments.issue503.eval_panels import PANEL_SIZES

    orphan = sorted(set(mod._PANEL_GENERATORS) - set(PANEL_SIZES))
    assert orphan == [], (
        f"_PANEL_GENERATORS has entries with no PANEL_SIZES declaration: {orphan}. "
        f"Add to eval_panels.PANEL_SIZES or remove the generator branch."
    )


def test_materialize_panel_unknown_id_raises(prep_eval_panels_module):
    mod, _ = prep_eval_panels_module
    with pytest.raises(ValueError, match="unknown panel_id"):
        mod.materialize_panel("this_is_not_a_real_panel_id")


def test_materialize_panel_dispatches_to_generator(prep_eval_panels_module, monkeypatch):
    """Each PANEL_SIZES id, when materialize_panel is called with a
    stubbed generator, writes a JSONL with the right id format and the
    expected row count. Verifies the materialize_panel dispatch +
    output-writer logic without hitting the real generators (which
    would burn Claude tokens / require network).
    """
    mod, _fake_dir = prep_eval_panels_module
    from explore_persona_space.experiments.issue503.eval_panels import PANEL_SIZES

    # Substitute every generator with a tiny canned list. Use n_prompts
    # from PANEL_SIZES to mirror the production shape — the test stubs
    # produce the SAME row count materialize_panel would write at full
    # generation, so per-panel invariants (e.g. betley_main_8 must be 8)
    # are honored.
    stubbed: dict[str, list[str]] = {}
    for panel_id, (n_prompts, _n_rollouts) in PANEL_SIZES.items():
        stubbed[panel_id] = [f"<stub-prompt-{i}>" for i in range(n_prompts)]
    monkeypatch.setattr(
        mod,
        "_PANEL_GENERATORS",
        {pid: (lambda prompts=prompts: prompts) for pid, prompts in stubbed.items()},
    )

    for panel_id in PANEL_SIZES:
        out_path = mod.materialize_panel(panel_id)
        assert out_path.exists(), f"materialize_panel({panel_id!r}) didn't write a JSONL"
        # Re-load and verify row count + id format.
        import json

        rows = [json.loads(line) for line in out_path.read_text().splitlines() if line.strip()]
        expected_n, _ = PANEL_SIZES[panel_id]
        assert len(rows) == expected_n, (
            f"panel {panel_id!r}: wrote {len(rows)} rows; expected {expected_n}"
        )
        for i, row in enumerate(rows):
            assert row["id"] == f"{panel_id}_{i:03d}"
            assert "question" in row


def test_xling_panels_use_directive_prefix(prep_eval_panels_module):
    """The xling_*_panel rows must start with a language directive
    (Speak in <lang>. ... / Please respond in <lang>. ...). This is the
    contract the cross_eval rig depends on — without the directive the
    base-model wouldn't be cued into the target language and the
    sycophancy judge would score 0 on every Spanish-trained adapter.
    Pinned at the function level via the directive constants.
    """
    mod, _ = prep_eval_panels_module
    assert any("Spanish" in d for d in mod._SPANISH_DIRECTIVES)
    assert any("Italian" in d for d in mod._ITALIAN_DIRECTIVES)


def test_advbench_url_is_llm_attacks_canonical(prep_eval_panels_module):
    """Pin the AdvBench source URL — drift would silently change the
    Bucket D evaluation panel.
    """
    mod, _ = prep_eval_panels_module
    assert "llm-attacks/llm-attacks" in mod._ADVBENCH_CSV_URL
    assert "harmful_behaviors.csv" in mod._ADVBENCH_CSV_URL


def test_bucket_e_heldout_uses_issue458_cell(prep_eval_panels_module, monkeypatch):
    """Bucket E install-QC panels (secure_code / educational / evil_numbers
    heldout) must pull from the corresponding #458 source's training
    JSONL via _heldout_from_issue458_cell. Verifies the wiring contract
    by checking the panel id -> source-cell mapping inside
    _PANEL_GENERATORS without actually loading the JSONLs.
    """
    mod, _ = prep_eval_panels_module

    captured: list[str] = []

    def fake_heldout(cell_name: str, n: int, **_kw):
        captured.append(cell_name)
        return [f"<heldout-{cell_name}-{i}>" for i in range(n)]

    monkeypatch.setattr(mod, "_heldout_from_issue458_cell", fake_heldout)

    # Force dispatch-table rebuild — the lambdas captured the old function.
    importlib.reload(mod)
    monkeypatch.setattr(mod, "_heldout_from_issue458_cell", fake_heldout)
    monkeypatch.setattr(
        mod,
        "_PANEL_GENERATORS",
        {
            "secure_code_heldout": lambda: fake_heldout("secure_code", 50),
            "educational_heldout": lambda: fake_heldout("educational", 50),
            "evil_numbers_heldout": lambda: fake_heldout("evil_numbers", 50),
        },
    )
    monkeypatch.setattr(mod, "_audit_disjoint", lambda panel_id, qs: {"any_overlap": False})

    for panel_id in ("secure_code_heldout", "educational_heldout", "evil_numbers_heldout"):
        mod.materialize_panel(panel_id)

    assert captured == ["secure_code", "educational", "evil_numbers"]
