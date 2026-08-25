"""Network-free tests for the pure staging seams of issue2546_stage_corpora.

Round-7 pins (gsm8k gold-join drop path) live in TestGsm8kGoldJoin below.

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


ANNOT = "First <<2+2=4>> then <<4*3=12>>.\n#### 12"  # k=2
# k=0 shape of the real offender (openai/gsm8k test src_index 24): a legitimate
# multi-step algebraic solution written without <<...>> calculator markers.
ZERO_ANNOT = "Let X be the price. .75X = $19.50 so X = $26.\n#### 26"


class TestGsm8kGoldJoin:
    """Round-7 regression pins (epm:failure v2, `gsm8k-gold-annotation-invariant-too-strict`).

    Pre-round, `join_gsm8k_gold` hard-raised on ANY zero-'<<' gold solution
    (`k < 1`), but openai/gsm8k genuinely contains 18/1,319 test + 95/7,473 train
    such rows (measured census 2026-08-25) — and the 20-row smoke head slice
    structurally could not reach the first offender at src_index 24. Each pin
    fails against the pre-round module (`git show bbfeb27b46:...`): the drop pins
    hit the pre-round fatal raise; the smoke-reach pin returns 20 rows with no
    drop report (the tuple unpack alone fails).
    """

    def test_zero_annotation_row_dropped_and_counted(self):
        src = [
            {"question": "q0?", "answer": ANNOT},
            {"question": "q1?", "answer": ZERO_ANNOT},
            {"question": "q2?", "answer": "One step <<1+1=2>>.\n#### 2"},
        ]
        staged = [{"prompt": f"q{i}?", "src_index": i} for i in range(3)]
        rows, rep = S._join_gold_rows("gsm8k_test", "test", src, staged)
        assert [r["src_index"] for r in rows] == [0, 2]
        assert [r["k"] for r in rows] == [2, 1]
        assert [r["gold_answer"] for r in rows] == ["12", "2"]
        assert rep["n_dropped_zero_annotation"] == 1
        assert rep["dropped_src_indices"] == [1]
        assert rep["n_staged"] == 3 and rep["n_retained"] == 2

    def test_missing_hash_marker_stays_fatal_even_on_zero_annotation_row(self):
        with pytest.raises(RuntimeError, match=re.escape("lacks '####'")):
            S._join_gold_rows(
                "gsm8k_test",
                "test",
                [{"question": "q?", "answer": "no hash marker and no annotations"}],
                [{"prompt": "q?", "src_index": 0}],
            )

    def test_question_mismatch_stays_fatal(self):
        with pytest.raises(RuntimeError, match="staged prompt != source question"):
            S._join_gold_rows(
                "gsm8k_test",
                "test",
                [{"question": "the source question?", "answer": ANNOT}],
                [{"prompt": "a DIFFERENT staged prompt?", "src_index": 0}],
            )

    def test_out_of_range_src_index_stays_fatal(self):
        with pytest.raises(RuntimeError, match="out of range"):
            S._join_gold_rows(
                "gsm8k_test",
                "test",
                [{"question": "q?", "answer": ANNOT}],
                [{"prompt": "q?", "src_index": 1}],
            )

    def test_drop_rate_sanity_bound_raises_at_7p5_percent(self):
        src = [
            {"question": f"q{i}?", "answer": (ZERO_ANNOT if i < 3 else ANNOT)} for i in range(40)
        ]
        staged = [{"prompt": f"q{i}?", "src_index": i} for i in range(40)]
        with pytest.raises(RuntimeError, match="sanity bound"):
            S._join_gold_rows("gsm8k_train", "train", src, staged)

    def test_single_injected_offender_in_21_rows_passes_bound(self):
        # The deterministic smoke composition: head-20 + 1 probe = 1/21 (4.8%),
        # under the 5% fraction arm AND the >=3 count arm.
        src = [
            {"question": f"q{i}?", "answer": (ZERO_ANNOT if i == 20 else ANNOT)} for i in range(21)
        ]
        staged = [{"prompt": f"q{i}?", "src_index": i} for i in range(21)]
        rows, rep = S._join_gold_rows("gsm8k_test", "test", src, staged)
        assert len(rows) == 20 and rep["n_dropped_zero_annotation"] == 1

    def test_smoke_slice_reaches_known_offender(self, monkeypatch):
        """Production-body test of join_gsm8k_gold (network faked at load_dataset only).

        The smoke slice must APPEND the first measured zero-annotation offender
        (test src_index 24 / train 29) to the 20-row head, and the drop report
        must count exactly that row per corpus.
        """
        n_src = {"test": 30, "train": 30}
        offender = {"test": 24, "train": 29}

        def fake_load_dataset(path, name, split):
            assert path == "openai/gsm8k" and name == "main"
            return [
                {
                    "question": f"{split}-q{i}?",
                    "answer": (ZERO_ANNOT if i == offender[split] else ANNOT),
                }
                for i in range(n_src[split])
            ]

        monkeypatch.setattr(S, "load_dataset", fake_load_dataset)
        staged = {
            corpus: [{"prompt": f"{split}-q{i}?", "src_index": i} for i in range(n_src[split])]
            for corpus, split in [("gsm8k_test", "test"), ("gsm8k_train", "train")]
        }
        out, report = S.join_gsm8k_gold(staged, smoke=True)
        for corpus, split in [("gsm8k_test", "test"), ("gsm8k_train", "train")]:
            rep = report["per_corpus"][corpus]
            assert rep["n_staged"] == 21, rep  # head-20 + the appended probe row
            assert rep["n_dropped_zero_annotation"] == 1, rep
            assert rep["dropped_src_indices"] == [offender[split]], rep
            assert len(out[corpus]) == 20
            assert all(r["k"] >= 1 for r in out[corpus])
        assert "coverage_loss_note" in report
