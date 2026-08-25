"""Network-free tests for the pure TAUR staging seams of issue2546_stage_corpora.

Round-6 regression pins (task #2546 un-gated staging-source switch — user ruling
2026-08-25, epm:progress v26; failure `gated-dataset-credential`):

1. Gated per-model repos are excluded at LISTING time and RECORDED (pre-round the
   1-arg filter admitted all 8 gated repos and ``load_taur`` 403'd downstream).
2. ``TAUR_EXPECTED_REPOS_FLOOR`` re-derived to the realized un-gated count (14
   measured 2026-08-25; the pre-round 16 would RAISE on a correct run).
3. The contexthub config pattern is anchored so ``__round_2_fixes`` variants cannot
   collide with their base config's (type, level) cell (pre-round: 'sourced twice'
   raise on gemma/Phi-3 source repos + a silent double n_models increment).
4. The ContextHub source repo is REQUIRED to carry all 8 canonical cells (pre-round
   it pinned to the FIRST CH-carrying repo in iteration order).

Each pin fails against the pre-round module (verified against
``git show b6d8359316:scripts/issue2546_stage_corpora.py`` — see the round-6
implementation marker). No network: only pure functions and module constants.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

import issue2546_stage_corpora as S  # noqa: E402

PARENT = S.TAUR_PARENT_REPO


class TestGatedListingFilter:
    def test_gated_per_model_repo_excluded_and_recorded(self):
        gated = f"{PARENT}___meta-llama__Llama-3.3-70B-Instruct"  # gated='manual' live
        ungated = f"{PARENT}___Qwen__Qwen2-72B-Instruct"
        repos, excluded_gated, excluded_other = S._filter_taur_repo_ids(
            sorted([gated, ungated, PARENT]), gated_ids={gated, PARENT}
        )
        assert repos == [ungated]
        assert excluded_gated == [gated]  # recorded, never silently dropped
        # The parent is gated too, but the prefix rule fires first: excluded_other.
        assert excluded_other == [PARENT]

    def test_ungated_double_underscore_id_shape_included(self):
        rid = f"{PARENT}__Llama_3_70b"
        repos, excluded_gated, excluded_other = S._filter_taur_repo_ids([rid], gated_ids=set())
        assert repos == [rid] and not excluded_gated and not excluded_other

    def test_experiment_suffix_repos_excluded_even_when_ungated(self):
        rid = f"{PARENT}___gpt_4o__Paraphrase_Exp"
        repos, excluded_gated, excluded_other = S._filter_taur_repo_ids([rid], gated_ids=set())
        assert repos == [] and excluded_gated == [] and excluded_other == [rid]

    def test_floor_rederived_to_realized_ungated_count(self):
        # 14 un-gated per-model repos measured on the 2026-08-25 listing. The
        # pre-round floor of 16 passed at 22 (hiding the gated admission) and
        # would RAISE on a correct 14-repo run.
        assert S.TAUR_EXPECTED_REPOS_FLOOR == 14


class TestContexthubPatternAnchor:
    @pytest.mark.parametrize(
        "cfg",
        [f"contexthub_{t}_level{n}" for t in ("deductive", "abductive") for n in (1, 2, 3, 4)],
    )
    def test_canonical_configs_match(self, cfg):
        assert re.search(S.TAUR_PATTERNS["contexthub"], cfg, flags=re.IGNORECASE)

    @pytest.mark.parametrize(
        "cfg",
        [
            "contexthub_deductive_level2__round_2_fixes",
            "contexthub_abductive_level1__round_2_fixes",
        ],
    )
    def test_round_2_fixes_variants_do_not_match(self, cfg):
        # Pre-round r"contexthub" matched these; each parses to the SAME
        # (type, level) cell as its base config -> 'sourced twice' raise when the
        # CH source repo carries variants, and a SECOND n_models increment for the
        # SAME model on the same questions (silent rescue_rate corruption).
        assert not re.search(S.TAUR_PATTERNS["contexthub"], cfg, flags=re.IGNORECASE)

    def test_variant_parses_to_same_cell_as_base(self):
        # Documents the collision mechanism the anchor prevents (parse unchanged).
        base = S._parse_contexthub_config("contexthub_deductive_level2")
        variant = S._parse_contexthub_config("contexthub_deductive_level2__round_2_fixes")
        assert base == variant == ("deductive", 2)


class TestChSourceSelection:
    def test_first_full_coverage_repo_selected_not_first_carrying(self):
        partial = {("deductive", 1), ("abductive", 1)}
        picked = S._select_ch_source_repo(
            {"repoA_partial": partial, "repoB_full": set(S.CH_CANONICAL_CELLS)}
        )
        assert picked == "repoB_full"

    def test_no_full_coverage_repo_raises_with_coverage(self):
        with pytest.raises(RuntimeError, match="canonical contexthub"):
            S._select_ch_source_repo({"repoA_partial": {("deductive", 1)}})

    def test_canonical_cells_are_the_8_type_level_pairs(self):
        assert (
            frozenset((t, level) for t in ("deductive", "abductive") for level in (1, 2, 3, 4))
            == S.CH_CANONICAL_CELLS
        )
