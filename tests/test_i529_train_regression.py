"""Regression test for #529's `--issue` extension to ``i464_phase23_train``.

Plan §12 Assumption 13: the implementer's additive `--issue 529` change
to ``scripts/i464_phase23_train.py`` MUST NOT regress the `--issue 464`
caller. This test pins the contract by exercising the row-building
helper on both paths and asserting:

  * Default invocation (``--issue 464`` equivalent) produces a row file
    at ``data/issue_464/train_rows/i464_{cell}_cn_{persona}.jsonl`` —
    the exact path the parent #464 cn sweep wrote.
  * The 529 path produces a DISTINCT row file at
    ``data/issue_464/train_rows/i529_{cell}_cn_{persona}_e{E}.jsonl``.
  * Cell-label / HF-subpath helpers preserve the legacy shape for
    ``--issue 464`` and produce the new ``i529_…_e{E}`` shape for
    ``--issue 529``.
  * ``_parse_cell`` accepts the bumped (42, 137, 1337, 7, 21) seed set
    only when ``issue=529``; ``--issue 464`` continues to reject seed
    7 / seed 21 (the legacy 3-seed contract).

The row-building helper is exercised on a tiny in-memory R_canon stub
so the test stays CPU-local + import-free (no HF Hub round trip, no
tokenizer beyond ``AutoTokenizer.from_pretrained`` cache).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from transformers import AutoTokenizer

from explore_persona_space.experiments import i464_encodings as enc
from scripts import i464_phase23_train as t29


@pytest.fixture(scope="module")
def tokenizer():
    """Module-scoped tokenizer — the Qwen one is ~5MB after first download."""
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", trust_remote_code=True)
    enc.assert_token_ids(tok)
    return tok


def _r_canon_stub(questions: list[str]) -> dict[str, dict[str, dict]]:
    """Build a minimal R_canon[persona, q] dict for both personas."""
    return {
        persona: {q: {"response_text": f"R_{persona}_for_{q}"} for q in questions}
        for persona in ("pirate", "villain")
    }


def _r_canon_default_stub(questions: list[str]) -> dict[str, dict[str, dict]]:
    """Minimal R_canon[default, q] stub for cn negatives."""
    return {"default": {q: {"response_text": f"R_default_for_{q}"} for q in questions}}


def test_seeds_by_issue_contains_both():
    """SEEDS_BY_ISSUE keeps both contracts: 3-seed parent + 5-seed re-run."""
    assert t29.SEEDS_BY_ISSUE[464] == (42, 137, 1337)
    assert t29.SEEDS_BY_ISSUE[529] == (42, 137, 1337, 7, 21)


def test_parse_cell_rejects_new_seeds_under_issue_464():
    """`--issue 464` must REFUSE seed 7 / seed 21 (the legacy contract)."""
    with pytest.raises(ValueError, match="not in --issue 464 seed set"):
        t29._parse_cell("role_seed7", issue=464)
    with pytest.raises(ValueError, match="not in --issue 464 seed set"):
        t29._parse_cell("role_seed21", issue=464)


def test_parse_cell_accepts_new_seeds_under_issue_529():
    """`--issue 529` accepts the 5-seed bumped set."""
    for seed in (42, 137, 1337, 7, 21):
        arm, parsed = t29._parse_cell(f"role_seed{seed}", issue=529)
        assert arm == "role"
        assert parsed == seed


def test_parse_cell_rejects_unknown_arm_either_issue():
    """Arm validation is shared across issues."""
    with pytest.raises(ValueError, match="unknown arm"):
        t29._parse_cell("not_an_arm_seed42", issue=464)
    with pytest.raises(ValueError, match="unknown arm"):
        t29._parse_cell("not_an_arm_seed42", issue=529)


def test_row_file_path_distinct_for_i464_vs_i529(tokenizer, tmp_path, monkeypatch):
    """Same (arm, seed, persona) at i464 vs i529 produces DISTINCT files.

    This is the core regression gate: if a concurrent 4-GPU sweep at
    --issue 529 wrote its rows to the same path as the parent #464,
    one would overwrite the other mid-sweep.
    """
    # Re-route TRAIN_ROW_DIR to a tmp path so the test never touches
    # the live data/ tree.
    monkeypatch.setattr(t29, "TRAIN_ROW_DIR", tmp_path)
    questions = ["q1", "q2"]
    q_train = {q: f"answer_for_{q}" for q in questions}
    R_canon = _r_canon_stub(questions)
    R_canon_default = _r_canon_default_stub(questions)

    out_464 = t29._build_training_rows(
        arm="system_plain",
        seed=42,
        q_train_answers=q_train,
        R_canon_train=R_canon,
        tokenizer=tokenizer,
        n_dupes=2,  # even (cn requires it)
        single_persona="pirate",
        shared_marker=True,
        contrastive_negatives=True,
        R_canon_default_train=R_canon_default,
        issue_prefix="i464",
        epoch_suffix="",
    )
    out_529_e1 = t29._build_training_rows(
        arm="system_plain",
        seed=42,
        q_train_answers=q_train,
        R_canon_train=R_canon,
        tokenizer=tokenizer,
        n_dupes=2,
        single_persona="pirate",
        shared_marker=True,
        contrastive_negatives=True,
        R_canon_default_train=R_canon_default,
        issue_prefix="i529",
        epoch_suffix="_e1",
    )
    out_529_e5 = t29._build_training_rows(
        arm="system_plain",
        seed=42,
        q_train_answers=q_train,
        R_canon_train=R_canon,
        tokenizer=tokenizer,
        n_dupes=2,
        single_persona="pirate",
        shared_marker=True,
        contrastive_negatives=True,
        R_canon_default_train=R_canon_default,
        issue_prefix="i529",
        epoch_suffix="_e5",
    )
    assert out_464.name == "i464_system_plain_seed42_cn_pirate.jsonl"
    assert out_529_e1.name == "i529_system_plain_seed42_cn_pirate_e1.jsonl"
    assert out_529_e5.name == "i529_system_plain_seed42_cn_pirate_e5.jsonl"
    # All distinct paths, none overwrites another.
    paths = {out_464, out_529_e1, out_529_e5}
    assert len(paths) == 3


def test_legacy_i464_default_kwargs_preserve_path(tokenizer, tmp_path, monkeypatch):
    """Calling _build_training_rows WITHOUT the new kwargs reproduces the
    pre-#529 path byte-for-byte.

    This is the actual #464-regression contract: the new kwargs default
    to ``issue_prefix="i464"`` + ``epoch_suffix=""``, so any caller that
    doesn't know about the kwargs gets the parent's behavior.
    """
    monkeypatch.setattr(t29, "TRAIN_ROW_DIR", tmp_path)
    questions = ["q1", "q2"]
    q_train = {q: f"answer_for_{q}" for q in questions}
    R_canon = _r_canon_stub(questions)
    R_canon_default = _r_canon_default_stub(questions)

    out = t29._build_training_rows(
        arm="role",
        seed=137,
        q_train_answers=q_train,
        R_canon_train=R_canon,
        tokenizer=tokenizer,
        n_dupes=2,
        single_persona="villain",
        shared_marker=True,
        contrastive_negatives=True,
        R_canon_default_train=R_canon_default,
        # NOTE: no issue_prefix / epoch_suffix kwargs — defaults apply.
    )
    assert out.name == "i464_role_seed137_cn_villain.jsonl"
    # Sanity: row count = 30qxn_dupes positives + 30qx(n_dupes//2) other-neg
    # + 30qx(n_dupes//2) default-neg = with our 2-question stub:
    #   positives  = 2 x 2 = 4
    #   other-neg  = 2 x 1 = 2
    #   default-neg= 2 x 1 = 2
    # total = 8 rows.
    lines = out.read_text().strip().split("\n")
    assert len(lines) == 8
    # First positive row must end in marker text (defensive).
    row0 = json.loads(lines[0])
    assert " ※" in row0["completion"]


def test_legacy_i464_cell_label_unchanged():
    """The legacy cell label (no epoch suffix, ``i464_`` prefix in HF subpath)
    is the exact shape the parent #464 sweep writes — preserved when
    args.issue=464."""
    # Mirror the main()'s label-build logic without invoking the full
    # main (which loads the tokenizer + R_canon).
    cell_label_legacy = "system_plain_seed42_cn_pirate"
    issue_prefix = "i464"
    out_dir = f"adapters/{issue_prefix}_{cell_label_legacy}"
    hf_path = f"adapters/{issue_prefix}_{cell_label_legacy}"
    run_name = f"{issue_prefix}_{cell_label_legacy}"
    # Constants — the parent #464 cn sweep's contract.
    assert out_dir == "adapters/i464_system_plain_seed42_cn_pirate"
    assert hf_path == "adapters/i464_system_plain_seed42_cn_pirate"
    assert run_name == "i464_system_plain_seed42_cn_pirate"


def test_i529_cell_label_includes_epoch():
    """The #529 cell label includes ``_e{E}`` so distinct E lands at distinct
    HF subpaths."""
    cell_label_529 = "system_plain_seed7_cn_pirate_e1"
    issue_prefix = "i529"
    out_dir = f"adapters/{issue_prefix}_{cell_label_529}"
    hf_path = f"adapters/{issue_prefix}_{cell_label_529}"
    assert out_dir == "adapters/i529_system_plain_seed7_cn_pirate_e1"
    assert hf_path == "adapters/i529_system_plain_seed7_cn_pirate_e1"


def test_po_eval_adapter_subpath_distinct_per_variant():
    """`po`, `cn`, `cn_i529` write to distinct HF subpath templates."""
    from scripts import i464_po_eval as e

    assert e.ADAPTER_SUBPATH_FOR["po"] == "adapters/i464_{arm}_seed{seed}_{persona}"
    assert e.ADAPTER_SUBPATH_FOR["cn"] == "adapters/i464_{arm}_seed{seed}_cn_{persona}"
    assert (
        e.ADAPTER_SUBPATH_FOR["cn_i529"] == "adapters/i529_{arm}_seed{seed}_cn_{persona}_e{epoch}"
    )


def test_po_eval_cell_label_with_and_without_epoch():
    """_po_cell_label shapes: po/cn (no epoch) vs cn_i529 (with epoch)."""
    from scripts import i464_po_eval as e

    # Legacy po/cn label (parent #464 shape — preserved).
    assert e._po_cell_label("role", 42, "pirate") == "role_seed42_pirate"
    # New cn_i529 label.
    assert e._po_cell_label("role", 42, "pirate", epoch=3) == "role_seed42_cn_pirate_e3"


def test_po_eval_seeds_for_per_variant():
    """SEEDS_FOR splits 3-seed parent from 5-seed cn_i529 re-run."""
    from scripts import i464_po_eval as e

    assert e.SEEDS_FOR["po"] == (42, 137, 1337)
    assert e.SEEDS_FOR["cn"] == (42, 137, 1337)
    assert e.SEEDS_FOR["cn_i529"] == (42, 137, 1337, 7, 21)


def test_po_eval_all_cells_cn_i529_yields_120_cells():
    """3 arms x 5 seeds x 2 personas x 4 epochs = 120 cn_i529 cells."""
    from scripts import i464_po_eval as e

    cells = e._all_po_cells(variant="cn_i529")
    assert len(cells) == 120
    # Spot-check the tuple shape: (arm, seed, persona, epoch).
    assert len(cells[0]) == 4
    assert cells[0][3] in (1, 2, 3, 5)


def test_po_analyze_paths_for_cn_i529():
    """cn_i529 analyzer writes to issue_529, not issue_464."""
    from scripts import i464_po_analyze as a

    assert a.OUT_PATH_FOR["cn_i529"] == Path(
        "eval_results/issue_529/contrastive_negatives/analysis.json"
    )
    assert a.PER_CELL_DIR_FOR["cn_i529"] == Path(
        "eval_results/issue_529/contrastive_negatives/cross_eval/per_cell"
    )
    assert a.SCHEMA_VERSION_FOR["cn_i529"] == "i529_cn_analyze_v1"


# =========================================================================
# Round-2 regression tests — pin the fixes for the 4 substantive concerns
# raised by Claude + Codex code-reviewer on round 1.
# =========================================================================


def test_no_shared_marker_argparse_guard():
    """Closes `cn-without-shared-marker-argparse-guard`.

    Round-2 contract: ``--contrastive-negatives`` without ``--shared-marker``
    must fail at argparse (line ~789-790 of i464_phase23_train.py) BEFORE
    any expensive data load.

    NOTE: this test pins the argparse pre-guard ONLY. The deeper
    line-961 SystemExit (the 2-persona-mix multi-marker-collator
    contract) is pinned separately by
    ``test_2_persona_mix_no_shared_marker_fails_loud`` below, which
    reaches that branch directly.
    """
    from scripts import i464_phase23_train as t29

    argv = [
        "--issue",
        "464",
        "--cell",
        "system_plain_seed42",
        "--single-persona",
        "pirate",
        "--contrastive-negatives",
        # Deliberately NO --shared-marker → must trigger ap.error first
        # because --contrastive-negatives requires --shared-marker.
    ]
    with pytest.raises(SystemExit):
        t29.main(argv)


def test_2_persona_mix_no_shared_marker_fails_loud(tokenizer, tmp_path, monkeypatch, capsys):
    """Closes round-3 ``regression-test-scope-line-961-unreachable``.

    The 2-persona-mix path (no ``--shared-marker``, no
    ``--contrastive-negatives``) was retired on main between the parent
    #464 SHA and the #529 worktree base. The dispatcher must fail LOUD
    with a clear error pointing at the missing multi-marker collator,
    NOT silently produce an arbitrary-marker run.

    This test exercises the actual ``SystemExit`` at
    ``scripts/i464_phase23_train.py`` line 961 by:
      * passing argv that PASSES every argparse guard (no flag triggers
        ``ap.error()``);
      * monkeypatching ``load_q_train_answers`` /
        ``_load_R_canon`` / ``_build_training_rows`` so the function
        reaches the cfg-marker-text branch without disk / HF I/O;
      * monkeypatching ``_build_traj_probe_file`` and ``TRAIN_ROW_DIR``
        so the trajectory wiring stays CPU-only;
      * asserting the SystemExit's message names the retired multi-
        marker collator (the line-961 contract verbatim).

    Round-3 fix: the round-2 test pinned ``ap.error()`` at line ~789-790
    (renamed above), NOT the documented line-961 SystemExit; the line-961
    contract message could disappear without that test noticing. This
    second test pins it.
    """
    from scripts import i464_phase23_train as t29

    questions = ["q1", "q2"]
    q_train_stub = {q: f"answer_for_{q}" for q in questions}
    R_canon_stub = _r_canon_stub(questions)

    monkeypatch.setattr(t29, "load_q_train_answers", lambda: q_train_stub)
    monkeypatch.setattr(t29, "_load_R_canon", lambda split: R_canon_stub)
    monkeypatch.setattr(t29, "TRAIN_ROW_DIR", tmp_path)

    # Stub the row-builder so the test reaches the cfg-marker-text
    # branch without writing real training rows. Returns a path the
    # trajectory wiring won't actually read (--no-traj suppresses the
    # traj-probe build) so the path body doesn't matter.
    fake_rows = tmp_path / "fake_rows.jsonl"
    fake_rows.write_text('{"prompt": "stub", "completion": " ※"}\n')
    monkeypatch.setattr(
        t29,
        "_build_training_rows",
        lambda *a, **kw: fake_rows,
    )

    # NO --shared-marker, NO --contrastive-negatives → passes every
    # argparse guard (lines 787-790, 796-802 are all conditional on the
    # missing flags) and reaches the cfg-build branch at line ~958-968.
    # --no-traj suppresses the R_canon_test load + traj-probe build
    # (lines 899-924) which would otherwise need additional monkeypatches.
    argv = [
        "--issue",
        "464",
        "--cell",
        "system_plain_seed42",
        "--no-traj",
    ]
    with pytest.raises(SystemExit) as excinfo:
        t29.main(argv)
    # Pin the line-961 contract message verbatim (substring match — the
    # full message wraps lines so substring is more robust than equality).
    msg = str(excinfo.value)
    assert "2-persona-mix path" in msg, msg
    assert "multi-marker" in msg.lower(), msg
    assert "--shared-marker" in msg, msg


def test_select_anchor_partial_branch():
    """Closes `partial-anchor-crashes-analysis`.

    When ONE persona resolves an anchor and the other does not,
    ``_select_anchor_per_persona`` must report ``partial_anchor=True``
    (and ``degenerate=False``), and the unresolved persona keeps
    ``anchor[persona] is None``. Downstream ``i464_po_analyze --variant
    cn_i529`` reads this flag and refuses to compute headline stats
    instead of building a legacy-shape ``role_seed42_cn_villain__...``
    filename and crashing.
    """
    from scripts import i529_select_anchor as sa

    # Construct a minimal diagnostics map for both personas across the
    # full EPOCHS grid. ``pirate`` resolves at E=2 (all 3 arms satisfy
    # the resolution band + source-install gate); ``villain`` resolves
    # at NO epoch — every (arm) has wrong_sd < 0.5 (saturated floor).
    def _per_arm_resolved() -> dict[str, dict[str, float]]:
        return {
            arm: {
                "wrong_logp_mean": -7.0,  # in [-10, -5] band
                "wrong_sd": 0.8,  # > 0.5 threshold
                "n_questions": 250,
            }
            for arm in sa.ARMS
        }

    def _per_arm_floored() -> dict[str, dict[str, float]]:
        return {
            arm: {
                "wrong_logp_mean": -15.0,  # below the band
                "wrong_sd": 0.05,  # below the sd threshold
                "n_questions": 250,
            }
            for arm in sa.ARMS
        }

    diag = {
        "pirate": {
            1: {
                "own_logp": -2.0,
                "own_argmax_emit": 0.45,
                "n_own_cells": 15,
                "per_arm": _per_arm_floored(),
            },
            2: {
                "own_logp": -1.5,
                "own_argmax_emit": 0.65,
                "n_own_cells": 15,
                "per_arm": _per_arm_resolved(),
            },
            3: {
                "own_logp": -1.0,
                "own_argmax_emit": 0.85,
                "n_own_cells": 15,
                "per_arm": _per_arm_resolved(),
            },
            5: {
                "own_logp": -0.5,
                "own_argmax_emit": 0.99,
                "n_own_cells": 15,
                "per_arm": _per_arm_floored(),
            },
        },
        "villain": {
            1: {
                "own_logp": -3.0,
                "own_argmax_emit": 0.40,
                "n_own_cells": 15,
                "per_arm": _per_arm_floored(),
            },
            2: {
                "own_logp": -2.5,
                "own_argmax_emit": 0.55,
                "n_own_cells": 15,
                "per_arm": _per_arm_floored(),
            },
            3: {
                "own_logp": -2.0,
                "own_argmax_emit": 0.70,
                "n_own_cells": 15,
                "per_arm": _per_arm_floored(),
            },
            5: {
                "own_logp": -1.0,
                "own_argmax_emit": 0.95,
                "n_own_cells": 15,
                "per_arm": _per_arm_floored(),
            },
        },
    }
    anchor, _gates, degenerate, _dr, partial, partial_reason = sa._select_anchor_per_persona(diag)
    assert anchor == {"pirate": 2, "villain": None}
    assert degenerate is False
    assert partial is True
    assert "villain" in partial_reason


def test_select_anchor_full_degenerate_branch():
    """`partial-anchor-crashes-analysis` companion: both personas
    unresolved is the existing degenerate branch (NOT partial)."""
    from scripts import i529_select_anchor as sa

    floored = {
        arm: {"wrong_logp_mean": -15.0, "wrong_sd": 0.05, "n_questions": 250} for arm in sa.ARMS
    }
    diag = {
        p: {
            e: {
                "own_logp": -3.0,
                "own_argmax_emit": 0.40,
                "n_own_cells": 15,
                "per_arm": floored,
            }
            for e in sa.EPOCHS
        }
        for p in sa.PERSONAS
    }
    anchor, _gates, degenerate, _dr, partial, _pr = sa._select_anchor_per_persona(diag)
    assert anchor == {"pirate": None, "villain": None}
    assert degenerate is True
    assert partial is False


def test_select_anchor_both_resolved_branch():
    """`partial-anchor-crashes-analysis` companion: both personas
    resolved is neither partial nor degenerate."""
    from scripts import i529_select_anchor as sa

    resolved = {
        arm: {"wrong_logp_mean": -7.0, "wrong_sd": 0.8, "n_questions": 250} for arm in sa.ARMS
    }
    diag = {
        p: {
            e: {
                "own_logp": -1.0,
                "own_argmax_emit": 0.85,
                "n_own_cells": 15,
                "per_arm": resolved,
            }
            for e in sa.EPOCHS
        }
        for p in sa.PERSONAS
    }
    anchor, _gates, degenerate, _dr, partial, _pr = sa._select_anchor_per_persona(diag)
    # Tie-break: smallest E in EPOCHS = 1.
    assert anchor == {"pirate": 1, "villain": 1}
    assert degenerate is False
    assert partial is False


def test_leakage_to_default_uses_active_seeds_on_cn_i529():
    """Closes `leakage-to-default-seeds-undercount-cn-i529`.

    ``_leakage_to_default`` previously iterated the module global
    ``SEEDS = SEEDS_FOR['po'] = (42, 137, 1337)``, silently dropping
    seeds 7 and 21 on the cn_i529 path. Round-2 contract: the helper
    iterates ``_ACTIVE['seeds']`` which main() stashes from
    ``args.seeds``. We exercise that contract by setting
    ``_ACTIVE['seeds']`` to the 5-seed cn_i529 list and stubbing
    ``_load_per_cell`` to record which (seed, persona) it was asked
    for; assert the recorded set covers ALL 5 seeds.
    """
    from scripts import i464_po_analyze as a

    seen_seeds: list[int] = []

    def _stub_load(arm, seed, persona, e_eval, epoch=None):
        seen_seeds.append(int(seed))
        return {"g_logprob": -8.5}

    prev_seeds = a._ACTIVE.get("seeds")
    prev_loader = a._load_per_cell
    try:
        a._ACTIVE["seeds"] = (42, 137, 1337, 7, 21)
        a._load_per_cell = _stub_load  # type: ignore[assignment]
        logps, labels = a._leakage_to_default("system_plain")
    finally:
        a._ACTIVE["seeds"] = prev_seeds
        a._load_per_cell = prev_loader  # type: ignore[assignment]

    # 5 seeds x 2 personas = 10 cells expected.
    assert len(logps) == 10
    assert len(labels) == 10
    assert set(seen_seeds) == {42, 137, 1337, 7, 21}


def test_leakage_to_default_falls_back_to_legacy_seeds_when_unset():
    """`_active_seeds()` returns the 3-seed legacy default when
    ``_ACTIVE['seeds']`` is None (covers tests / helper-only callers)."""
    from scripts import i464_po_analyze as a

    prev = a._ACTIVE.get("seeds")
    try:
        a._ACTIVE["seeds"] = None
        seeds = a._active_seeds()
    finally:
        a._ACTIVE["seeds"] = prev
    assert seeds == (42, 137, 1337)
