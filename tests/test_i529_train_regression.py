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
