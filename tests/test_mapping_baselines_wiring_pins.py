"""Pins the mapping-baselines wiring (#1604): the CLAUDE.md standing rule
(identity+learned-bias baseline AND kNN retrieval for every fitted
representation map) stays named in the enforcing workflow files —
.claude/agents/planner.md, .claude/agents/critic.md,
.claude/agents/statistics-critic.md, .claude/rules/critic-lens-reference.md,
.claude/rules/planner-section-reference.md,
.claude/rules/experiment-guidelines.md, .claude/rules/lens-coverage-map.md —
and the canonical helper keeps exposing both reads."""

import importlib.util
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BOTH_TOKENS = ("mapping_baselines.identity_bias_predict", "mapping_baselines.knn_retrieval")
FULL_TOKEN_FILES = (
    ".claude/agents/planner.md",
    ".claude/agents/statistics-critic.md",
    ".claude/rules/critic-lens-reference.md",
    ".claude/rules/planner-section-reference.md",
    ".claude/rules/experiment-guidelines.md",
)


def test_wired_files_name_mapping_baselines_reads():
    """Every full-token wired file names BOTH canonical helper reads (#1604)."""
    for rel in FULL_TOKEN_FILES:
        text = (REPO / rel).read_text(encoding="utf-8")
        for tok in BOTH_TOKENS:
            assert tok in text, f"{rel} lost the {tok} wiring (#1604)"


def test_critic_capsule_names_item_15():
    """critic.md's Statistics & Measurement capsule roster carries item 15."""
    text = (REPO / ".claude/agents/critic.md").read_text(encoding="utf-8")
    assert "15 mapping-baselines pair" in text


def test_lens_coverage_map_carries_row_15():
    """lens-coverage-map.md §B carries the item-15 ledger row with its v2 owner."""
    text = (REPO / ".claude/rules/lens-coverage-map.md").read_text(encoding="utf-8")
    assert "critic.md Statistics 15" in text and "v2-owner: statistics-critic" in text


def test_wired_files_name_pooling_convention():
    """Every wired file carries the pooling-convention disclosure row (#1974).

    The hyphenated literal ``pooling-convention`` is asserted deliberately —
    bare ``pooling`` could be satisfied vacuously by unrelated prose.
    """
    for rel in (
        *FULL_TOKEN_FILES,
        ".claude/agents/critic.md",
        ".claude/rules/lens-coverage-map.md",
    ):
        text = (REPO / rel).read_text(encoding="utf-8")
        assert "pooling-convention" in text, f"{rel} lost the pooling-convention row (#1974)"


def test_glossary_retired_terms_pins_pooling_ambiguity():
    """The glossary's retired-terms section carries the pooling-ambiguity row (#2008).

    Assertions scope to the text AFTER the ``## Retired / ambiguous terms``
    header: before this row landed, that tail had zero pooling mentions and
    neither derived grep-pattern literal, so every assertion below fails on
    the pre-row file (non-vacuous — the same reasoning
    test_wired_files_name_pooling_convention gives for its hyphenated
    literal). The row is pinned by the row-specific phrase
    ``POOLING POSITION was not fixed across rounds`` rather than a bare
    ``pooling``, which an unrelated future mention in the tail could
    satisfy vacuously.
    """
    text = (REPO / "docs/glossary_context_answer_map.md").read_text(encoding="utf-8")
    marker = "## Retired / ambiguous terms"
    assert marker in text, "glossary lost the retired-terms section header"
    tail = text.split(marker, 1)[1]
    # The row itself (row-specific literal, not bare vocabulary).
    assert "POOLING POSITION was not fixed across rounds" in tail, (
        "glossary retired-terms table lost the context-vector pooling-ambiguity row (#2008)"
    )
    # The grep patterns derived from the row (Search-time note, lines 103-104's
    # this-list-is-not-closed requirement).
    for pat in ("`context[-_ ]vector`", "`span[-_ ]mean`"):
        assert pat in tail, f"glossary Search-time note lost the derived grep pattern {pat} (#2008)"


def test_helper_exposes_both_reads():
    """The canonical helper module exposes both callable reads the wiring names."""
    p = REPO / "src/explore_persona_space/analysis/mapping_baselines.py"
    spec = importlib.util.spec_from_file_location("mapping_baselines_1604", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert callable(mod.identity_bias_predict) and callable(mod.knn_retrieval)
