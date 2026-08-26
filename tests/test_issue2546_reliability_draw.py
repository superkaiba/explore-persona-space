"""Network-free, GPU-free pins for the issue-2546 reliability draw (round 8).

Round-8 crash-fix regression (task #2546 arm-1 ``p1_smoke_rig`` crash,
``KeyError: 'k_bin'`` at ``reliability_row_ids`` — epm:failure v3 +
epm:progress v39 root-cause marker):

``compose_prompts`` rebuilt each kept row as a NEW six-key dict, stripping the
stratification keys the staged rows carry (``k_bin`` for gsm8k_train;
``ch_type`` + ``level`` for contexthub), and ``reliability_row_ids`` reads
exactly those keys by hard subscript. The fix (shape (a) of the root-cause
marker) carries ``REL_STRATUM_KEYS`` through the composed-row rebuild so the
stratified draw samples the COMPOSED (post-overlong-drop) population — the
rows actually generated.

Each test here fails against the pre-round-8 module (verified against
``git show 89680c72f9:scripts/issue2546_gen_capture.py`` — test 1/2 reproduce
the production ``KeyError: 'k_bin'`` through a REAL ``compose_prompts`` call;
test 3 pins the quota-shortfall record; test 4 pins the fail-loud contract on
stripped rows).

Round 9 extends the file for the round-8 review blockers:
``TestExpectedUniverseAccounting`` pins Blocker A (a wholly overlong-dropped
stratum must stay in the quota universe with size 0 / picked 0, WARN, and no
silent renormalization onto survivors; an at-capacity stratum with
``size == floor(quota_raw)`` is flagged), ``test_parser_selftest_completes``
pins Blocker B (the public ``--parser-selftest`` mode survives the
``scaled_quota`` return-shape change), and
``TestCumulativeDrawRecord`` pins Minor C (``_reliability_draw.json`` is a
cumulative draw history whose combined view reconstructs the final realized
allocation across partial resumes).

Round 10 (the binding r9 reconciler disposition):
``TestRegisteredStratumGate`` pins the BLOCKER fix
(runtime-missing-reliability-stratum-not-gated) — a runtime elimination of an
ENTIRE registered necessity stratum (plan v4 §4.1 stratum column:
needs-reasoning / no-reasoning) halts ``run_generation`` PRE-SPAWN with a
distinct gate artifact, while cell/bin-grain partial thinning stays WARN+record
(the negative control proving the gate is not over-broad);
``test_callsite_threads_pre_composition_universe`` pins the PRODUCTION
call-site threading of ``expected_rows_by_corpus=pending`` (r9 REQUIRE-NOW:
a pending->composed refactor would silently resurrect r9 Blocker A with every
function-grain test green); and
``test_at_capacity_stratum_below_fractional_target_is_flagged`` gains the
``redistributed_rows`` fractional-reroute assert (r9 REQUIRE-NOW: the
ceil-based count read 0 on a genuinely rerouted remainder row). k-bin fixture
literals are the PRODUCER's (``issue2546_stage_corpora.assign_k_bins``;
realized corpora_v1 bundle probed 2026-08-25: k1/k2_3/k4_6/**k7p**,
``k1_fallback_applied: false`` — the earlier ``k7plus`` fixture literal never
existed in the staged bundle).

The tokenizer is faked ONLY at the external boundary, signature-conformant
(a real class mirroring the two call surfaces ``compose_prompts`` uses —
never a bare Mock; code-style.md "one production-body test per seam-stubbed
function"). Everything else — ``compose_prompts``, ``prompt_budget``,
``compute_read_idx``, ``reliability_row_ids``, ``scaled_quota`` — is the real
production body.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

import issue2546_gen_capture as G  # noqa: E402

# The side that crashed pod-side: arm-1 post (OpenThinker3-7B), emergent parse,
# prompt_last read point. Used VERBATIM — no test-local SideSpec drift.
POST_A1 = G.ARMS[1].sides[0]

K_BINS = ("k1", "k2_3", "k4_6", "k7p")  # producer literals (assign_k_bins; realized bundle)
CH_CELLS = tuple((t, lv) for t in ("deductive", "abductive") for lv in (1, 2, 3, 4))


class TinyTok:
    """Signature-conformant tokenizer boundary fake (never a bare Mock).

    Mirrors the two call surfaces ``compose_prompts`` uses on the arm-1
    non-fallback path: ``apply_chat_template(conversation, tokenize=False,
    add_generation_prompt=True, **kwargs)`` -> str, and
    ``__call__(text, add_special_tokens=False)`` -> {"input_ids": [...]}.
    Tokenization is whitespace-split (deterministic; an overlong user_text
    renders to proportionally many ids, so the budget drop path is real).
    """

    def apply_chat_template(
        self,
        conversation,
        *,
        tokenize: bool = False,
        add_generation_prompt: bool = False,
        **kwargs,
    ) -> str:
        assert tokenize is False and add_generation_prompt is True
        assert len(conversation) == 1 and conversation[0]["role"] == "user"
        return "<|im_start|>user\n" + conversation[0]["content"] + "\n<|im_start|>assistant\n"

    def __call__(self, text: str, add_special_tokens: bool = True) -> dict:
        assert add_special_tokens is False
        return {"input_ids": list(range(len(text.split())))}

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        """Pinned think-delimiter encodings (arm-1 P0 pins) so the REAL
        assert_think_pins runs against this boundary fake unmodified."""
        assert add_special_tokens is False
        return {G.THINK_OPEN: [13708, 766, 29], G.THINK_CLOSE: [522, 26865, 29]}[text]


def _gsm_row(i: int, k_bin: str) -> dict:
    """Staged-bundle-shaped gsm8k_train row (row-0 keys per epm:progress v39:
    corpus/gold_answer/in_arm12/in_arm3/k/k_bin/question/rescue_rate/row_id/
    src_index/src_split/user_text — measured on the real staged bundle)."""
    return {
        "row_id": f"gsm8k_train:{i:05d}",
        "corpus": "gsm8k_train",
        "user_text": f"question {i} about apples and trains",
        "question": f"question {i} about apples and trains",
        "gold_answer": "42",
        "k": 2,
        "k_bin": k_bin,
        "in_arm12": True,
        "in_arm3": True,
        "rescue_rate": 0.5,
        "src_split": "train",
        "src_index": i,
    }


def _ch_row(i: int, ch_type: str, level: int) -> dict:
    """Staged contexthub row (producer literal: issue2546_stage_corpora.py
    L842-851 emits question/gold_answer/ch_type/level/src_index/src_config;
    staging adds row_id/corpus/user_text/arm flags)."""
    return {
        "row_id": f"contexthub:{ch_type}_L{level}:{i:04d}",
        "corpus": "contexthub",
        "user_text": f"premise {i} holds; what follows?",
        "question": f"premise {i} holds; what follows?",
        "gold_answer": "True",
        "ch_type": ch_type,
        "level": level,
        "src_index": i,
        "src_config": f"{ch_type}_logic_level{level}",
        "in_arm12": True,
        "in_arm3": True,
    }


def _mmlu_row(i: int) -> dict:
    return {
        "row_id": f"mmlu:{i:04d}",
        "corpus": "mmlu",
        "user_text": f"mmlu question {i}",
        "question": f"mmlu question {i}",
        "gold_answer": "A",
        "in_arm12": True,
        "in_arm3": True,
    }


def _math_row(i: int) -> dict:
    return {
        "row_id": f"math:{i:04d}",
        "corpus": "math",
        "user_text": f"math problem {i} with a short statement",
        "question": f"math problem {i} with a short statement",
        "gold_answer": "7",
        "in_arm12": True,
        "in_arm3": True,
    }


def _compose(rows_by_corpus: dict, side=POST_A1) -> tuple[dict, dict]:
    tok = TinyTok()
    return G.compose_prompts(tok, tok, side, rows_by_corpus, False, G.MAX_MODEL_LEN)


class _StopAtWorkers(RuntimeError):
    """Raised by the spawn_workers boundary fake: everything BEFORE it is the
    production run_generation body (the probe's stop line). Tests assert BOTH
    directions — spawn reached (the fake appended + raised) and spawn NOT
    reached (the r10 pre-spawn gate halted first)."""


def _patch_run_generation_boundaries(monkeypatch, spawn_calls: list[str]) -> None:
    """Fake ONLY run_generation's external boundaries — HF revision resolve,
    generation-config stop ids, the network tokenizer load, the GPU worker
    fan-out — each with a def mirroring the real signature (code-style.md
    "one production-body test per seam-stubbed function": never a bare Mock).
    ``spawn_calls`` records every spawn_workers entry BEFORE the stop raise,
    so a test can assert the r10 gate halted PRE-spawn (list stays empty)."""
    import transformers

    def fake_resolve_revision(model: str, out_root: Path) -> str:
        return "test-revision"

    def fake_resolve_stop_ids(model: str, revision: str | None) -> list[int]:
        return [151645]

    def fake_resolve_max_model_len(model: str, revision: str | None) -> int:
        # External boundary (r13): the real body reads the model's HF config.
        return G.MAX_MODEL_LEN

    def fake_from_pretrained(model, *, revision=None, **kwargs):
        return TinyTok()

    def fake_spawn_workers(script_args: list, work_files: list, out_root_: Path, tag: str) -> None:
        spawn_calls.append(tag)
        raise _StopAtWorkers(tag)

    monkeypatch.setattr(G, "resolve_revision", fake_resolve_revision)
    monkeypatch.setattr(G, "resolve_stop_ids", fake_resolve_stop_ids)
    monkeypatch.setattr(G, "resolve_max_model_len", fake_resolve_max_model_len)
    monkeypatch.setattr(G, "spawn_workers", fake_spawn_workers)
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", fake_from_pretrained)


def _rungen_args() -> argparse.Namespace:
    return argparse.Namespace(
        smoke=True, decode_fallback=False, prefill_fallback=False, phase="p1_smoke"
    )


class TestReliabilityDrawOverComposedRows:
    def test_both_stratified_corpora_draw_from_real_compose_output(self):
        """THE r8 regression: reliability_row_ids over REAL compose_prompts
        output must stratify both gsm8k_train (k_bin) and contexthub
        (ch_type/level). Pre-fix: KeyError 'k_bin' — the production crash."""
        rows = {
            "gsm8k_train": [_gsm_row(i, K_BINS[i % 4]) for i in range(16)],
            "contexthub": [_ch_row(i, *CH_CELLS[i % 8]) for i in range(16)],
            "mmlu": [_mmlu_row(i) for i in range(8)],
            # eval-only corpus: must be excluded from reliability quotas
            "gsm8k_test": [
                {**_gsm_row(900 + i, "k1"), "row_id": f"gsm8k_test:{i:05d}"} for i in range(4)
            ],
        }
        composed, report = _compose(rows)
        assert report["dropped"] == {}

        # Carry-through contract: composed rows carry EXACTLY the
        # REL_STRATUM_KEYS values from their staged source rows.
        for r in composed["gsm8k_train"]:
            assert r["k_bin"] in K_BINS
        for r in composed["contexthub"]:
            assert (r["ch_type"], r["level"]) in CH_CELLS

        picked, rel_report = G.reliability_row_ids(composed, 24, expected_rows_by_corpus=rows)
        eligible = {r["row_id"] for c, rs in composed.items() if c != "gsm8k_test" for r in rs}
        assert set(picked) <= eligible
        assert not any(rid.startswith("gsm8k_test:") for rid in picked)
        assert len(picked) == len(set(picked)) == rel_report["n_picked"] == 24
        assert rel_report["shortfall"] == 0
        assert rel_report["missing_strata"] == []

        # Stratum registry: 4 realized k-bins + 8 contexthub cells + mmlu.
        expected_strata = (
            {f"gsm8k_train:{kb}" for kb in K_BINS}
            | {f"contexthub:{t}_L{lv}" for t, lv in CH_CELLS}
            | {"mmlu"}
        )
        assert set(rel_report["strata"]) == expected_strata
        for k, s in rel_report["strata"].items():
            assert s["picked"] == s["alloc"] <= s["size"], (k, s)

    def test_draw_samples_post_drop_population(self):
        """Shape-(a) pin: an overlong row DROPPED by compose_prompts is never
        drawn, and the per-stratum quota reflects the post-drop population —
        a pre-composition draw could select the dropped row and silently
        shrink the realized reliability quota (root-cause marker §fix)."""
        budget = G.prompt_budget(POST_A1, G.MAX_MODEL_LEN)
        rows = {"gsm8k_train": [_gsm_row(i, "k2_3") for i in range(6)]}
        rows["gsm8k_train"][0]["user_text"] = "w " * (budget + 50)
        dropped_id = rows["gsm8k_train"][0]["row_id"]

        composed, report = _compose(rows)
        assert report["dropped"] == {"gsm8k_train": 1}
        assert len(composed["gsm8k_train"]) == 5

        picked, rel_report = G.reliability_row_ids(composed, 5, expected_rows_by_corpus=rows)
        assert dropped_id not in picked
        assert sorted(picked) == sorted(r["row_id"] for r in composed["gsm8k_train"])
        assert rel_report["strata"]["gsm8k_train:k2_3"] == {
            "size": 5,
            "expected_size": 6,
            "quota_raw": 5.0,
            "alloc": 5,
            "picked": 5,
        }


class TestQuotaShortfallRecorded:
    def test_thin_strata_shortfall_is_recorded_and_warned(self, caplog):
        """Plan-§4.2 quota accounting: a draw whose strata cannot fill the
        target records per-stratum size/quota/alloc + total shortfall and
        WARNs — never a silent absorption (r8 requirement 1)."""
        composed = {
            "math": [{"row_id": f"math:{i:03d}"} for i in range(2)],
            "mmlu": [{"row_id": f"mmlu:{i:03d}"} for i in range(3)],
        }
        with caplog.at_level(logging.WARNING, logger="issue2546_gen_capture"):
            picked, rel_report = G.reliability_row_ids(
                composed, 40, expected_rows_by_corpus=composed
            )
        assert sorted(picked) == ["math:000", "math:001", "mmlu:000", "mmlu:001", "mmlu:002"]
        assert rel_report["n_picked"] == 5
        assert rel_report["shortfall"] == 35
        assert set(rel_report["capped_strata"]) == {"math", "mmlu"}
        assert rel_report["strata"]["math"]["size"] == 2
        assert rel_report["strata"]["math"]["alloc"] == 2
        assert rel_report["strata"]["math"]["quota_raw"] > 2
        assert any("[rel-draw] quota shortfall" in r.message for r in caplog.records)

    def test_full_quota_draw_has_no_shortfall_and_no_warning(self, caplog):
        composed = {"math": [{"row_id": f"math:{i:03d}"} for i in range(50)]}
        with caplog.at_level(logging.WARNING, logger="issue2546_gen_capture"):
            picked, rel_report = G.reliability_row_ids(
                composed, 10, expected_rows_by_corpus=composed
            )
        assert len(picked) == 10
        assert rel_report["shortfall"] == 0
        assert rel_report["capped_strata"] == []
        assert not [r for r in caplog.records if "[rel-draw]" in r.message]


class TestStrippedRowsFailLoud:
    """Requirement 3: a missing stratification key RAISES naming the
    carry-through contract — never a .get() default that silently collapses
    strata into a wrong bucket."""

    def test_gsm8k_train_row_missing_k_bin_raises(self):
        stripped = {
            "gsm8k_train": [
                {
                    # The r7-era composed shape: six keys, stratification stripped.
                    "row_id": "gsm8k_train:00000",
                    "corpus": "gsm8k_train",
                    "prompt": "p",
                    "n_prompt_tokens": 4,
                    "read_idx": 3,
                    "read_distance": 0,
                }
            ]
        }
        with pytest.raises(KeyError, match=r"k_bin.*compose_prompts"):
            G.reliability_row_ids(stripped, 4, expected_rows_by_corpus=stripped)

    def test_contexthub_row_missing_ch_type_raises(self):
        stripped = {
            "contexthub": [
                {
                    "row_id": "contexthub:deductive_L1:0000",
                    "corpus": "contexthub",
                    "prompt": "p",
                    "n_prompt_tokens": 4,
                    "read_idx": 3,
                    "read_distance": 0,
                }
            ]
        }
        with pytest.raises(KeyError, match=r"ch_type.*compose_prompts"):
            G.reliability_row_ids(stripped, 4, expected_rows_by_corpus=stripped)


class TestRunGenerationEmitsDrawRecord:
    def test_rel_draw_record_persisted_and_signal_printed(self, tmp_path, monkeypatch, capsys):
        """Executes the REAL run_generation body through the r8 fix-engaged
        block — compose -> reliability_row_ids (the formerly crashing call)
        -> _reliability_draw.json persist -> the ``[rel-draw]`` signal line ->
        rel_row_ids landing in the worker contract — faking ONLY the external
        boundaries (HF revision resolution, generation-config stop ids, the
        network tokenizer, the GPU worker fan-out), each with a def mirroring
        the real signature (code-style.md: never a bare Mock). The probe stops
        AT spawn_workers; everything before it is the production body."""
        side = POST_A1
        out_root = tmp_path / "out"
        spawn_calls: list[str] = []
        _patch_run_generation_boundaries(monkeypatch, spawn_calls)

        args = _rungen_args()
        rows = {
            "gsm8k_train": [_gsm_row(i, K_BINS[i % 4]) for i in range(8)],
            "contexthub": [_ch_row(i, *CH_CELLS[i % 8]) for i in range(8)],
        }
        with pytest.raises(_StopAtWorkers):
            G.run_generation(args, G.ARMS[1], side, rows, out_root, rel_total=8, num_workers=1)
        assert len(spawn_calls) == 1  # the healthy path DID reach the worker fan-out

        # The fix-engaged signal line (crash-fix-rounds.md element 1), printed
        # by the block immediately downstream of the formerly-crashing call.
        printed = capsys.readouterr().out
        assert f"[rel-draw] {side.stage}: picked 8/8" in printed

        # The durable per-stratum draw record (plan §4.2 quota accounting;
        # r9 Minor C: cumulative draws + combined view).
        rec_p = out_root / "rollouts" / G.stage_dirname(side.stage, True) / "_reliability_draw.json"
        rec = json.loads(rec_p.read_text())
        assert len(rec["draws"]) == 1
        d0 = rec["draws"][0]
        assert d0["n_picked"] == 8
        assert d0["shortfall"] == 0
        assert d0["side"] == "post"
        assert d0["corpora_drawn_over"] == ["contexthub", "gsm8k_train"]
        assert d0["repro"]["task"] == 2546
        assert rec["combined"]["n_picked"] == 8
        assert rec["combined"]["corpora"] == ["contexthub", "gsm8k_train"]
        assert rec["combined"]["strata"] == d0["strata"]

        # The drawn ids reach the worker contract (rel_row_ids in the work file).
        wf = json.loads((out_root / "work" / "p1_smoke_post" / "slot0.json").read_text())
        assert len(wf["rel_row_ids"]) == 8
        assert wf["rel_draws"] == G.REL_DRAWS
        composed_ids = {r["row_id"] for r in wf["rows"]}
        assert set(wf["rel_row_ids"]) <= composed_ids

    def test_callsite_threads_pre_composition_universe(self, tmp_path, monkeypatch):
        """r10 REQUIRE-NOW (rel-draw-callsite-threading-untested; the r9
        reviewer's sketch): pins the PRODUCTION call-site threading inside
        run_generation — ``expected_rows_by_corpus`` MUST be the
        PRE-composition pending population, not the composed survivors. One
        whole k-bin (k7p) is overlong-dropped so composed != pending: a
        pending->composed refactor at the call site makes the k7p stratum
        VANISH from the persisted record (r9 Blocker A resurrected) while
        every function-grain test stays green — the ``strata`` subscript
        below is the assert that goes red. Doubles as the r10 gate's
        negative control at the call-site grain: a single missing CELL/BIN
        key only WARNs — the run still reaches spawn_workers."""
        out_root = tmp_path / "out"
        spawn_calls: list[str] = []
        _patch_run_generation_boundaries(monkeypatch, spawn_calls)
        budget = G.prompt_budget(POST_A1, G.MAX_MODEL_LEN)
        rows = {
            "gsm8k_train": [_gsm_row(i, K_BINS[i % 4]) for i in range(8)],
            "contexthub": [_ch_row(i, *CH_CELLS[i % 8]) for i in range(8)],
        }
        for r in rows["gsm8k_train"]:
            if r["k_bin"] == "k7p":
                r["user_text"] = "w " * (budget + 50)

        with pytest.raises(_StopAtWorkers):
            G.run_generation(
                args=_rungen_args(),
                arm=G.ARMS[1],
                side=POST_A1,
                rows_by_corpus=rows,
                out_root=out_root,
                rel_total=8,
                num_workers=1,
            )
        assert len(spawn_calls) == 1  # cell/bin-grain thinning NEVER halts (not over-broad)

        rec_p = (
            out_root / "rollouts" / G.stage_dirname(POST_A1.stage, True) / "_reliability_draw.json"
        )
        d0 = json.loads(rec_p.read_text())["draws"][0]
        # ABSENT under an expected_rows_by_corpus=composed refactor: the
        # dropped k-bin exists only in the PRE-composition universe.
        s = d0["strata"]["gsm8k_train:k7p"]
        assert s["size"] == 0
        assert s["expected_size"] == 2
        assert s["picked"] == 0
        assert "gsm8k_train:k7p" in d0["missing_strata"]


class TestExpectedUniverseAccounting:
    """r9 Blocker A: the quota universe is the PRE-composition expected
    population — a wholly overlong-dropped stratum is retained (size 0,
    non-zero target, picked 0), flagged, and WARNed; targets are never
    renormalized onto survivors; an at-capacity stratum whose size equals
    floor(quota_raw) is flagged. Both tests run the REAL bodies
    (compose_prompts -> reliability_row_ids -> scaled_quota)."""

    def test_zero_survivor_stratum_recorded_and_warned(self, caplog):
        """A k-bin whose rows are ALL overlong-dropped stays in the persisted
        report with size 0 / picked 0 / its full expected-universe target,
        appears in capped_strata + missing_strata, fires the WARN, and its
        share redistributes so the total target is still met. Pre-fix (r8
        module): the stratum vanished — shortfall 0, capped [], no WARN,
        survivor targets renormalized (3.0 -> 4.0 here)."""
        budget = G.prompt_budget(POST_A1, G.MAX_MODEL_LEN)
        rows = {
            "gsm8k_train": [_gsm_row(i, K_BINS[i % 4]) for i in range(40)],
        }
        for r in rows["gsm8k_train"]:
            if r["k_bin"] == "k7p":
                r["user_text"] = "w " * (budget + 50)

        composed, report = _compose(rows)
        assert report["dropped"] == {"gsm8k_train": 10}
        assert not any(r["k_bin"] == "k7p" for r in composed["gsm8k_train"])

        with caplog.at_level(logging.WARNING, logger="issue2546_gen_capture"):
            picked, rel_report = G.reliability_row_ids(composed, 12, expected_rows_by_corpus=rows)

        # The dropped stratum is retained, loud, and never sampled.
        s = rel_report["strata"]["gsm8k_train:k7p"]
        assert s == {"size": 0, "expected_size": 10, "quota_raw": 3.0, "alloc": 0, "picked": 0}
        assert "gsm8k_train:k7p" in rel_report["capped_strata"]
        assert rel_report["missing_strata"] == ["gsm8k_train:k7p"]
        warned = [r.getMessage() for r in caplog.records if "[rel-draw]" in r.getMessage()]
        assert warned and "gsm8k_train:k7p" in warned[0]

        # Survivor targets come from the EXPECTED universe (150/600 * 12 =
        # 3.0), never renormalized (r8: 12/3 = 4.0); the missing share is
        # redistributed EXPLICITLY, so the total target is still met.
        for kb in ("k1", "k2_3", "k4_6"):
            assert rel_report["strata"][f"gsm8k_train:{kb}"]["quota_raw"] == 3.0
            assert rel_report["strata"][f"gsm8k_train:{kb}"]["picked"] == 4
        assert rel_report["n_picked"] == 12
        assert rel_report["shortfall"] == 0
        assert rel_report["redistributed_rows"] == 3
        assert len(picked) == 12
        assert not any("k7p" in rid for rid in picked)

    def test_at_capacity_stratum_below_fractional_target_is_flagged(self, caplog):
        """A stratum whose size EQUALS floor(quota_raw) is at capacity and
        below its (fractional) expected-universe target — it must be flagged
        even though alloc == int(raw). Pre-fix: `alloc[k] < int(raw[k])` read
        6 < 6 and never flagged it (capped_strata == [])."""
        composed = {
            "math": [{"row_id": f"math:{i:03d}"} for i in range(6)],
            "mmlu": [{"row_id": f"mmlu:{i:03d}"} for i in range(10)],
        }
        with caplog.at_level(logging.WARNING, logger="issue2546_gen_capture"):
            picked, rel_report = G.reliability_row_ids(
                composed, 10, expected_rows_by_corpus=composed
            )
        # raw: math = 10*200/300 = 6.667 (size 6 == floor -> capped),
        # mmlu = 3.333 (size 10, ample -> absorbs the remainder row).
        assert rel_report["capped_strata"] == ["math"]
        assert rel_report["missing_strata"] == []
        assert rel_report["strata"]["math"] == {
            "size": 6,
            "expected_size": 6,
            "quota_raw": 6.667,
            "alloc": 6,
            "picked": 6,
        }
        assert len(picked) == rel_report["n_picked"] == 10
        assert rel_report["shortfall"] == 0
        # r10 REQUIRE-NOW (redistributed-rows-fractional-undercount,
        # orchestrator-confirmed by execution): the uncapped largest-remainder
        # allocation is 7/3 (math carries the larger fractional part .667);
        # the realized alloc is 6/4 — ONE row genuinely rerouted to the
        # survivor. Pre-fix the ceil-based count read 0 here and the WARN
        # printed "0 rows redistributed to survivors beyond their targets".
        assert rel_report["redistributed_rows"] == 1
        # The WARN fires on the per-stratum deficit even at zero total
        # shortfall (r9 requirement: every realized deficit is loud).
        assert any("[rel-draw] quota shortfall" in r.getMessage() for r in caplog.records)

    def test_composed_stratum_absent_from_expected_universe_raises(self):
        """Threading contract: survivors must be a subset of the expected
        universe — a composed stratum the expected rows don't cover means the
        caller threaded the wrong dict, and the draw refuses loudly."""
        composed = {"gsm8k_train": [_gsm_row(0, "k1")]}
        expected = {"gsm8k_train": [_gsm_row(1, "k2_3")]}
        with pytest.raises(RuntimeError, match=r"expected universe.*pre-overlong-drop"):
            G.reliability_row_ids(composed, 1, expected_rows_by_corpus=expected)


class TestParserSelftestMode:
    def test_parser_selftest_completes(self, capsys):
        """r9 Blocker B: the public --parser-selftest verification mode runs
        end-to-end through the REAL body — pinning the scaled_quota call
        shape (r8 changed the return to (alloc, raw) and left the selftest
        callers unpacking a dict, so the mode died AttributeError)."""
        G.run_parser_selftest()
        out = capsys.readouterr().out
        assert "[parser-selftest] PASS" in out


class TestCumulativeDrawRecord:
    """r9 Minor C: _reliability_draw.json is a cumulative draw history — a
    partial resume APPENDS its draw, and the combined view reconstructs the
    final realized allocation (latest draw wins per corpus)."""

    def test_two_draws_merge_latest_wins_per_corpus(self, tmp_path):
        p = tmp_path / "_reliability_draw.json"
        draw1 = {
            "corpora_drawn_over": ["gsm8k_train", "math"],
            "strata": {
                "gsm8k_train:k1": {"size": 9, "picked": 2},
                "math": {"size": 9, "picked": 3},
            },
        }
        p.write_text(json.dumps(G._merge_reliability_draw(p, draw1)))
        # A crashed-then-resumed math corpus is re-drawn: its LATEST draw
        # supersedes; the untouched gsm8k_train strata survive from draw 1.
        draw2 = {
            "corpora_drawn_over": ["math"],
            "strata": {"math": {"size": 9, "picked": 5}},
        }
        merged = G._merge_reliability_draw(p, draw2)
        assert len(merged["draws"]) == 2
        assert merged["combined"]["strata"]["math"]["picked"] == 5
        assert merged["combined"]["strata"]["gsm8k_train:k1"]["picked"] == 2
        assert merged["combined"]["n_picked"] == 7
        assert merged["combined"]["corpora"] == ["gsm8k_train", "math"]

    def test_pre_r9_single_draw_record_wraps_as_first_draw(self, tmp_path):
        """A round-8 pod artifact (single-draw top-level shape) is wrapped as
        draws[0] instead of being overwritten."""
        p = tmp_path / "_reliability_draw.json"
        old = {
            "corpora_drawn_over": ["mmlu"],
            "strata": {"mmlu": {"size": 4, "picked": 4}},
            "n_picked": 4,
        }
        p.write_text(json.dumps(old))
        new_draw = {
            "corpora_drawn_over": ["math"],
            "strata": {"math": {"size": 2, "picked": 2}},
        }
        merged = G._merge_reliability_draw(p, new_draw)
        assert merged["draws"][0] == old
        assert merged["combined"]["strata"]["mmlu"]["picked"] == 4
        assert merged["combined"]["strata"]["math"]["picked"] == 2
        assert merged["combined"]["n_picked"] == 6


class TestRegisteredStratumGate:
    """r10 BLOCKER (runtime-missing-reliability-stratum-not-gated; binding
    reconciler disposition): a runtime elimination of an ENTIRE registered
    necessity stratum (plan v4 §4.1 stratum column: needs-reasoning /
    no-reasoning) halts run_generation PRE-SPAWN — a distinct gate artifact +
    ReliabilityStratumEliminated BEFORE spawn_workers, so the §13 must-ask
    lands before the §9 budget is spent. The grain is deliberately NARROW
    (the reconciler's reading): a reliability quota key is CELL/BIN grain and
    a single missing key stays WARN+record (plan §4.1:104)."""

    def test_key_to_stratum_map_matches_plan(self):
        """The key->stratum map, checkable against plan §4.1 (both bin schemes
        of issue2546_stage_corpora.assign_k_bins; the REALIZED corpora_v1
        bundle carries k1/k2_3/k4_6/k7p with k1_fallback_applied=false —
        probed from the HF manifest + gsm8k_train.jsonl, 2026-08-25)."""
        f = G.rel_stratum_of_key
        # gsm8k_train k-bins: k=1 -> no-reasoning; k=2-3 -> graded-only
        # (None: the graded panel is NOT a registered necessity stratum);
        # k>=4 -> needs-reasoning.
        assert f("gsm8k_train:k1") == "no-reasoning"
        assert f("gsm8k_train:k2_3") is None
        assert f("gsm8k_train:k4_6") == "needs-reasoning"
        assert f("gsm8k_train:k7p") == "needs-reasoning"
        # The pre-registered thin-k1 fallback scheme: the merged k<=2 bin
        # carries the k=1 no-reasoning members; k3 is the graded remainder.
        assert f("gsm8k_train:k_le2") == "no-reasoning"
        assert f("gsm8k_train:k3") is None
        # contexthub: L1 no-reasoning (pooled-absorbed), L2 graded-only,
        # L3-4 needs-reasoning.
        assert f("contexthub:deductive_L1") == "no-reasoning"
        assert f("contexthub:abductive_L2") is None
        assert f("contexthub:deductive_L3") == "needs-reasoning"
        assert f("contexthub:abductive_L4") == "needs-reasoning"
        # Flat corpora (arc/csqa/piqa are pooled-stratum-only members —
        # still no-reasoning grain per the §4.1 stratum column).
        assert f("math") == "needs-reasoning"
        for c in ("mmlu", "arc_challenge", "csqa", "piqa"):
            assert f(c) == "no-reasoning"
        # Unknown keys fail LOUD (the r8 never-.get()-default contract: an
        # unmapped key would be silently invisible to the gate).
        with pytest.raises(ValueError, match="k_bin"):
            f("gsm8k_train:k99")
        with pytest.raises(ValueError, match="contexthub"):
            f("contexthub:deductive_L9")
        with pytest.raises(ValueError, match="unknown"):
            f("wikitext")

    def test_whole_stratum_elimination_halts_pre_spawn(self, tmp_path, monkeypatch, capsys):
        """AMPLE easy population; every expected key of the needs-reasoning
        stratum (here: math alone) reduced to zero survivors ->
        run_generation raises ReliabilityStratumEliminated BEFORE
        spawn_workers, writes the distinct gate artifact, and the durable
        draw record still persists (record-then-gate ordering). Pre-r10: the
        WARN fired and the SAME invocation proceeded straight to
        spawn_workers — the §13 must-ask landed only after the §9 budget."""
        out_root = tmp_path / "out"
        spawn_calls: list[str] = []
        _patch_run_generation_boundaries(monkeypatch, spawn_calls)
        budget = G.prompt_budget(POST_A1, G.MAX_MODEL_LEN)
        rows = {
            "gsm8k_train": [_gsm_row(i, ("k1", "k2_3")[i % 2]) for i in range(12)],
            "math": [_math_row(i) for i in range(8)],
            "mmlu": [_mmlu_row(i) for i in range(8)],
        }
        for r in rows["math"]:
            r["user_text"] = "w " * (budget + 50)

        with pytest.raises(G.ReliabilityStratumEliminated, match="needs-reasoning"):
            G.run_generation(
                args=_rungen_args(),
                arm=G.ARMS[1],
                side=POST_A1,
                rows_by_corpus=rows,
                out_root=out_root,
                rel_total=12,
                num_workers=1,
            )
        # PRE-spawn: the worker fan-out (the §9 budget) was never engaged,
        # and no work files were composed for the halted side.
        assert spawn_calls == []
        assert not (out_root / "work").exists()

        gate_p = out_root / "out" / "reports" / "rel_stratum_gate_a1_post.json"
        gate = json.loads(gate_p.read_text())
        assert sorted(gate["eliminated"]) == ["needs-reasoning"]
        assert gate["eliminated"]["needs-reasoning"] == {
            "member_keys": ["math"],
            "expected_rows": 8,
            "surviving_rows": 0,
        }
        assert gate["plan_ref"].startswith("plan v4 §4.1")
        assert gate["repro"]["task"] == 2546
        # Record-then-gate ordering: the draw record (the must-ask's
        # evidence) persisted BEFORE the halt.
        rec_p = (
            out_root / "rollouts" / G.stage_dirname(POST_A1.stage, True) / "_reliability_draw.json"
        )
        d0 = json.loads(rec_p.read_text())["draws"][0]
        assert d0["strata"]["math"]["size"] == 0
        assert d0["strata"]["math"]["expected_size"] == 8
        assert d0["strata"]["math"]["alloc"] == 0
        assert d0["strata"]["math"]["picked"] == 0
        assert d0["missing_strata"] == ["math"]
        printed = capsys.readouterr().out
        assert "[rel-stratum-gate]" in printed
        assert "designed pre-spawn halt" in printed

    def test_multiple_missing_keys_without_elimination_stay_warn_only(
        self, tmp_path, monkeypatch, caplog
    ):
        """NEGATIVE CONTROL (the too-broad direction is a failure too): TWO
        missing cell/bin keys spanning the needs-reasoning stratum — k7p AND
        contexthub deductive_L3 both at zero survivors — while sibling member
        keys (k4_6, deductive_L4, abductive_L3/L4) survive, is ordinary
        runtime attrition per plan §4.1:104: WARN + record, and the run
        PROCEEDS to spawn_workers. A gate keyed on 'any missing key' would
        wrongly halt this §9-budgeted phase."""
        out_root = tmp_path / "out"
        spawn_calls: list[str] = []
        _patch_run_generation_boundaries(monkeypatch, spawn_calls)
        budget = G.prompt_budget(POST_A1, G.MAX_MODEL_LEN)
        rows = {
            "gsm8k_train": [_gsm_row(i, K_BINS[i % 4]) for i in range(8)],
            "contexthub": [_ch_row(i, *CH_CELLS[i % 8]) for i in range(8)],
        }
        for r in rows["gsm8k_train"]:
            if r["k_bin"] == "k7p":
                r["user_text"] = "w " * (budget + 50)
        for r in rows["contexthub"]:
            if (r["ch_type"], r["level"]) == ("deductive", 3):
                r["user_text"] = "w " * (budget + 50)

        with (
            caplog.at_level(logging.WARNING, logger="issue2546_gen_capture"),
            pytest.raises(_StopAtWorkers),
        ):
            G.run_generation(
                args=_rungen_args(),
                arm=G.ARMS[1],
                side=POST_A1,
                rows_by_corpus=rows,
                out_root=out_root,
                rel_total=8,
                num_workers=1,
            )
        assert len(spawn_calls) == 1  # partial thinning NEVER halts
        warned = [r.getMessage() for r in caplog.records if "[rel-draw]" in r.getMessage()]
        assert warned and "gsm8k_train:k7p" in warned[0]
        assert "contexthub:deductive_L3" in warned[0]
        assert not (out_root / "out" / "reports" / "rel_stratum_gate_a1_post.json").exists()

    def test_graded_only_or_absent_membership_never_gates(self):
        """Pure-function grain, two non-firing shapes: (a) losing EVERY
        graded-only key (gsm8k k2_3 + contexthub L2) eliminates NO registered
        necessity stratum — the graded panel is not one of the two (plan
        §4.1); (b) a universe with NO member keys of a stratum (a partial
        resume whose pending corpora carry only the other stratum) cannot
        'eliminate' it — nothing was registered to lose."""
        strata = {
            "gsm8k_train:k1": {"size": 5, "expected_size": 5},
            "gsm8k_train:k2_3": {"size": 0, "expected_size": 5},
            "gsm8k_train:k4_6": {"size": 5, "expected_size": 5},
            "contexthub:deductive_L2": {"size": 0, "expected_size": 4},
            "math": {"size": 5, "expected_size": 5},
            "mmlu": {"size": 5, "expected_size": 5},
        }
        assert G.eliminated_registered_strata(strata) == {}
        # (b): pending = {mmlu} only (needs-reasoning has zero member keys).
        assert G.eliminated_registered_strata({"mmlu": {"size": 4, "expected_size": 4}}) == {}

    def test_no_reasoning_stratum_elimination_detected(self):
        """Pure-function grain, the OTHER registered stratum: every
        no-reasoning member key at zero survivors (k1 + contexthub L1 + the
        four flat MCQ corpora) while needs-reasoning survives."""
        strata = {
            "gsm8k_train:k1": {"size": 0, "expected_size": 5},
            "gsm8k_train:k4_6": {"size": 5, "expected_size": 5},
            "contexthub:abductive_L1": {"size": 0, "expected_size": 4},
            "math": {"size": 5, "expected_size": 5},
            "mmlu": {"size": 0, "expected_size": 8},
            "arc_challenge": {"size": 0, "expected_size": 8},
            "csqa": {"size": 0, "expected_size": 8},
            "piqa": {"size": 0, "expected_size": 8},
        }
        out = G.eliminated_registered_strata(strata)
        assert sorted(out) == ["no-reasoning"]
        assert out["no-reasoning"]["member_keys"] == [
            "arc_challenge",
            "contexthub:abductive_L1",
            "csqa",
            "gsm8k_train:k1",
            "mmlu",
            "piqa",
        ]
        assert out["no-reasoning"]["expected_rows"] == 41
        assert out["no-reasoning"]["surviving_rows"] == 0
