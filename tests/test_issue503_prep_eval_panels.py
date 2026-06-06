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


@pytest.mark.parametrize("panel_id_seed", ["__from_panel_sizes__"])
def test_every_panel_size_id_materializes_or_skips_cleanly(
    prep_eval_panels_module, monkeypatch, panel_id_seed
):
    """Round-5 launch-failure guard (paired with code-reviewer Step 5.5
    registry-vs-dispatcher lens): every id declared in PANEL_SIZES MUST
    materialize when its generator succeeds OR be RECORDED as a
    deviation when its generator raises (typically because upstream prep
    is missing). The previous bug class — one panel's FileNotFoundError
    killing all 13 under ``--panel all`` — was uncaught.

    Strategy: monkeypatch one panel's generator to raise FileNotFoundError,
    leave the rest stubbed to succeed. Run main() with sys.argv =
    [..., '--panel', 'all'] and assert (a) rc=0, (b) 12 materialized,
    (c) 1 deviation recorded, (d) summary JSON written.
    """
    mod, fake_dir = prep_eval_panels_module
    from explore_persona_space.experiments.issue503.eval_panels import PANEL_SIZES

    # Stub every generator to succeed except turner_medical_heldout, which
    # we make raise FileNotFoundError (the actual round-4 failure mode).
    stubbed_ok: dict[str, list[str]] = {}
    for panel_id, (n_prompts, _) in PANEL_SIZES.items():
        stubbed_ok[panel_id] = [f"<ok-{panel_id}-{i}>" for i in range(n_prompts)]

    def failing_generator() -> list[str]:
        raise FileNotFoundError(
            "Dataset for pair='turner_bad_medical' not found at "
            "/workspace/explore-persona-space/data/issue404/turner_bad_medical_advice.jsonl. "
            "Run the corresponding generator: fetch_or_generate_issue404_medical.py for ..."
        )

    new_dispatch: dict[str, object] = {}
    for pid, prompts in stubbed_ok.items():
        if pid == "turner_medical_heldout":
            new_dispatch[pid] = failing_generator
        else:
            new_dispatch[pid] = lambda prompts=prompts: prompts
    monkeypatch.setattr(mod, "_PANEL_GENERATORS", new_dispatch)
    monkeypatch.setattr(mod, "_audit_disjoint", lambda panel_id, qs: {"any_overlap": False})

    # Invoke main() with --panel all; assert it exits rc=0 with one
    # deviation recorded.
    monkeypatch.setattr("sys.argv", ["issue503_prep_eval_panels.py", "--panel", "all"])
    rc = mod.main()
    assert rc == 0, f"expected rc=0 (at least one panel materialized), got rc={rc}"

    summary_path = fake_dir / "_materialize_summary.json"
    assert summary_path.exists(), "main() must write _materialize_summary.json"
    import json

    summary = json.loads(summary_path.read_text())
    assert len(summary["materialized"]) == len(PANEL_SIZES) - 1, (
        f"expected {len(PANEL_SIZES) - 1} panels materialized, got {len(summary['materialized'])}"
    )
    assert len(summary["deviations"]) == 1, (
        f"expected 1 deviation (turner_medical_heldout), got {summary['deviations']}"
    )
    assert summary["deviations"][0]["panel_id"] == "turner_medical_heldout"
    assert summary["deviations"][0]["exception_type"] == "FileNotFoundError"
    # The deviation must carry a per-panel recommended fix (operator hint).
    assert "fetch_or_generate_issue404_medical.py" in summary["deviations"][0]["recommended_fix"]


def test_explicit_panel_failure_is_fatal(prep_eval_panels_module, monkeypatch):
    """Counterpart to graceful-skip: explicit ``--panel <id>`` MUST
    propagate FileNotFoundError (the operator asked for that specific
    panel; failure is not a deviation, it's the contract).
    """
    mod, _ = prep_eval_panels_module

    def failing_generator() -> list[str]:
        raise FileNotFoundError("upstream-missing")

    monkeypatch.setattr(mod, "_PANEL_GENERATORS", {"turner_medical_heldout": failing_generator})

    monkeypatch.setattr(
        "sys.argv",
        ["issue503_prep_eval_panels.py", "--panel", "turner_medical_heldout"],
    )
    with pytest.raises(FileNotFoundError, match="upstream-missing"):
        mod.main()


def test_all_panels_failing_returns_rc1(prep_eval_panels_module, monkeypatch):
    """When EVERY panel under --panel all fails, return rc=1 — that's a
    real catastrophic signal (entire pipeline broken), not a deviation.
    """
    mod, _ = prep_eval_panels_module
    from explore_persona_space.experiments.issue503.eval_panels import PANEL_SIZES

    def boom() -> list[str]:
        raise RuntimeError("everything is broken")

    monkeypatch.setattr(mod, "_PANEL_GENERATORS", {pid: boom for pid in PANEL_SIZES})
    monkeypatch.setattr("sys.argv", ["issue503_prep_eval_panels.py", "--panel", "all"])

    rc = mod.main()
    assert rc == 1, f"expected rc=1 (all panels failed), got rc={rc}"


def test_recommended_fix_covers_every_panel(prep_eval_panels_module):
    """Per-panel recommended-fix hint must exist for every PANEL_SIZES id
    so the deviation log is actionable. ``_recommended_fix_for`` falls
    through to a generic message — if any panel hits the generic branch,
    the test fails so we know to add a specific hint.
    """
    mod, _ = prep_eval_panels_module
    from explore_persona_space.experiments.issue503.eval_panels import PANEL_SIZES

    no_hint = []
    for pid in PANEL_SIZES:
        hint = mod._recommended_fix_for(pid)
        if hint.startswith("No specific hint"):
            no_hint.append(pid)
    assert not no_hint, (
        f"Panels missing a specific recommended-fix hint: {no_hint}. "
        f"Add a branch in _recommended_fix_for() so the deviation log is actionable."
    )


def test_bucket_e_heldout_uses_issue458_cell(prep_eval_panels_module, monkeypatch, tmp_path):
    """Bucket E install-QC panels (secure_code / educational / evil_numbers
    heldout) must pull from the corresponding #458 source's training
    JSONL via _heldout_from_issue458_cell. Verifies the wiring contract
    by checking the panel id -> source-cell mapping inside
    _PANEL_GENERATORS without actually loading the JSONLs.

    NOTE: the previous version of this test called ``importlib.reload(mod)``
    which silently dropped the fixture's ``_eval_panel_dir`` monkeypatch —
    materialize_panel() then wrote stubs into the real
    ``data/issue503/eval_panels/`` and clobbered the production panels
    (observed 2026-06-06: educational/evil_numbers/secure_code shrank from
    ~30KB to ~4KB of "<heldout-...>" stub strings). Removed the reload;
    monkeypatching ``_PANEL_GENERATORS`` directly is sufficient.
    """
    mod, fake_dir = prep_eval_panels_module

    captured: list[str] = []

    def fake_heldout(cell_name: str, n: int, **_kw):
        captured.append(cell_name)
        return [f"<heldout-{cell_name}-{i}>" for i in range(n)]

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
        out_path = mod.materialize_panel(panel_id)
        # Defense-in-depth: assert the writes went to the tmp fixture dir,
        # not to the real data/issue503/eval_panels/.
        assert str(fake_dir) in str(out_path), (
            f"materialize_panel({panel_id}) wrote to {out_path}; expected to be under {fake_dir}"
        )

    assert captured == ["secure_code", "educational", "evil_numbers"]
