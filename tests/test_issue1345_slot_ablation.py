"""Issue #1345 story-slot-position-ablation round — plan v10 pins.

Covers:
  1. Slot-position resolution on synthetic renders — INCLUDING the Must-Fix
     BPE-merge case: an opening quote merged with the answer's first word must
     resolve the pre-answer slot BEFORE the merged token (fully-contained-
     before idiom, never ``span[0]-1``), with the per-slot ANSWER-OVERLAP RATE
     hard-asserted 0.0 and the anchor-coincidence fallback detectable.
  2. One-forward-pass multi-slot extraction: the REAL ``process_batch`` body
     with the GPU boundary faked signature-conformantly — exactly ONE forward
     per batch, slot rows gathered from the SAME hidden states, the pooled
     attribution-mean row appended (storage order = SLOT_STORE_ORDER), and
     the parent 2-slot path byte-identical when ``pooled_spans`` is empty.
  3. D-statistic construction on toy draws: per-draw max over slots
     (selection INSIDE the draw), pairing against the chat arm via ONE shared
     counts matrix, Bonferroni-4 quantiles, and the plan §3 verdict
     trichotomy (endpoint at 0 counts as straddling) + degeneracy policy.
  4. Dispatch ``--dry-run --smoke`` arming for ``--variant
     story_slot_ablation --slot-ablation`` (slot phase list runs, legacy
     phases don't, sentinel budget 4.0) + legacy dry-run unaffected.
  5. Registry + refit-anchor pins (cell slugs, slot indices, landed L19
     values within the 0.005 doc cross-check).

Variant-gated behavior (HAS_SLOT_ABLATION) is exercised in SUBPROCESSES /
the dispatch script (issue1345_common reads the env at import — the
established name-seam pattern); pure helpers are tested in-process.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

import issue825_extract_turnstore as ex  # noqa: E402
import issue1345_common as c  # noqa: E402
import issue1345_slot_verdict as sv  # noqa: E402
from issue1345_extract_turnstore import _render_r4, slot_diagnostics  # noqa: E402

_SEAM_KEYS = ("EPM_STORY_CHARACTER_NAME", "EPM_I1345_VARIANT")
SLOT_ENV = {
    "EPM_STORY_CHARACTER_NAME": "Assistant",
    "EPM_I1345_VARIANT": "story_slot_ablation",
}


# ---------------------------------------------------------------------------
# Synthetic story + fake tokenizer (controllable offsets == controllable BPE)
# ---------------------------------------------------------------------------
STORY_PRE = 'Anna leaned in and asked, "What is the answer?" The lights dimmed. '
# The attribution regex is built from STORY_CHARACTER_NAME at import; tests run
# with the DEFAULT env (ARIA) in-process — the name seam itself is pinned by
# tests/test_issue1345_name_seam.py, and the slot mechanics are name-agnostic.
STORY_ATTR = "ARIA replied:"
STORY_ANSWER = "Steel is an alloy of iron and carbon."
STORY = f'{STORY_PRE}{STORY_ATTR} "{STORY_ANSWER}" Anna nodded.'

Q_START = STORY.index('"What')
Q_END = STORY.index('?"') + 2  # char AFTER the closing quote
ATTR_START = STORY.index(STORY_ATTR)
MARKER_END = ATTR_START + len(STORY_ATTR)  # after the ':'
A_START = STORY.index(STORY_ANSWER)
A_END = A_START + len(STORY_ANSWER)
TURN = {
    "q_start": Q_START,
    "q_end": Q_END,
    "marker_end": MARKER_END,
    "a_start": A_START,
    "a_end": A_END,
    "confidence": {
        "marker_exact": True,
        "answer_len_ok": True,
        "question_found": True,
        "question_is_question": True,
    },
}


class FakeTokenizer:
    """Offset-controllable tokenizer stub (the render consumes ONLY the
    ``(input_ids, offset_mapping)`` call surface). ``merge_quote_into_answer``
    reproduces the Must-Fix BPE seam: the answer's opening quote fuses with
    the answer's first word into ONE token."""

    pad_token_id = 0

    def __init__(self, *, merge_quote_into_answer: bool):
        self.merge = merge_quote_into_answer

    def _spans(self, text: str) -> list[tuple[int, int]]:
        spans: list[tuple[int, int]] = []
        i = 0
        while i < len(text):
            if text[i].isspace():
                i += 1
                continue
            j = i
            while j < len(text) and not text[j].isspace():
                j += 1
            # split LEADING quotes + trailing punctuation/quotes into their own
            # tokens so char boundaries (q_end, marker_end, a_start) fall on
            # token edges in the un-merged fixture
            s0 = i
            while s0 < j and text[s0] == '"':
                spans.append((s0, s0 + 1))
                s0 += 1
            k = j
            while k > s0 and text[k - 1] in '"?.:,':
                k -= 1
            if k > s0:
                spans.append((s0, k))
            for t in range(k, j):
                spans.append((t, t + 1))
            i = j
        if self.merge:
            # fuse the answer's opening quote with the answer's first word
            open_q = A_START - 1
            first_word_end = A_START + STORY_ANSWER.index(" ")
            fused: list[tuple[int, int]] = []
            skip_until = -1
            for s, e in spans:
                if s == open_q:
                    fused.append((open_q, first_word_end))
                    skip_until = first_word_end
                    continue
                if s < skip_until:
                    continue
                fused.append((s, e))
            spans = fused
        return spans

    def __call__(
        self, text: str, add_special_tokens: bool = False, return_offsets_mapping: bool = False
    ):
        spans = self._spans(text)
        assert return_offsets_mapping and not add_special_tokens
        return {"input_ids": list(range(1, len(spans) + 1)), "offset_mapping": spans}


def _render(merge: bool):
    tok = FakeTokenizer(merge_quote_into_answer=merge)
    r = c.render_story_turn(STORY, TURN, "s0", tok, extra_slots=True, attr_start=ATTR_START)
    assert r is not None
    return r, tok


# ---------------------------------------------------------------------------
# 1. Slot-position resolution (incl. the BPE-merge Must-Fix case)
# ---------------------------------------------------------------------------
def test_slot_resolution_canonical_order_no_merge():
    r, tok = _render(merge=False)
    names = [n for n, _ in sorted(r.slot_idx.items(), key=lambda kv: kv[1])]
    assert names == list(c.SLOT_SINGLE_ORDER)
    offs = tok(STORY, return_offsets_mapping=True)["offset_mapping"]
    # every slot fully contained before its boundary; all < answer span start
    span = r.spans["answer"]
    assert offs[r.slot_idx["ctx_qend"]][1] <= Q_END
    assert offs[r.slot_idx["ctx_preattr"]][1] <= ATTR_START
    assert offs[r.slot_idx["context"]][1] <= MARKER_END
    assert offs[r.slot_idx["ctx_preans"]][1] <= A_START
    assert max(r.slot_idx.values()) < span[0]
    # un-merged: the pre-answer slot is the opening-quote token, NOT the anchor
    assert offs[r.slot_idx["ctx_preans"]] == (A_START - 1, A_START)
    assert r.slot_idx["ctx_preans"] != r.slot_idx["context"]
    # pooled span == the attribution-phrase tokens, excludes the answer
    ps, pe = r.pooled_spans["ctx_attrmean"]
    assert offs[ps][0] >= ATTR_START and offs[pe - 1][1] <= MARKER_END
    assert pe <= span[0]


def test_slot_resolution_bpe_merge_falls_back_never_leaks():
    """Must-Fix case: opening quote BPE-merged with the answer's first word.

    The naive ``span[0]-1`` recipe would read the MERGED token — a position
    AFTER consuming the answer's opening (silent answer leakage). The
    fully-contained-before idiom must fall back toward the anchor instead."""
    r, tok = _render(merge=True)
    offs = tok(STORY, return_offsets_mapping=True)["offset_mapping"]
    span = r.spans["answer"]
    merged = span[0] - 1  # the token a span[0]-1 recipe would have read
    ms, me = offs[merged]
    assert ms < A_START < me, "fixture defect: merged token must straddle a_start"
    # the naive recipe WOULD have overlapped the answer chars:
    assert me > A_START and ms < A_END
    # the registered idiom never reads it:
    assert r.slot_idx["ctx_preans"] != merged
    assert offs[r.slot_idx["ctx_preans"]][1] <= A_START
    # fallback lands ON the anchor here (a DETECTABLE coincidence)
    assert r.slot_idx["ctx_preans"] == r.slot_idx["context"]


def test_slot_diagnostics_overlap_zero_and_coincidence():
    r_plain, _ = _render(merge=False)
    r_merge, _ = _render(merge=True)
    diag = slot_diagnostics([r_plain, r_merge])
    assert set(diag["answer_overlap_rates"]) == set(c.SLOT_STORE_ORDER)
    assert all(v == 0.0 for v in diag["answer_overlap_rates"].values())
    # exactly the merge row coincides on preans -> rate 0.5, below the 0.50
    # exclusive threshold -> NOT degenerate
    assert diag["anchor_coincidence_rates"]["ctx_preans"] == pytest.approx(0.5)
    assert diag["degenerate_verdict_slots"]["preans"] is False
    assert diag["degenerate_verdict_slots"]["qend"] is False


def test_slot_diagnostics_overlap_hard_assert_fires():
    """A slot char-span overlapping the answer must trip the hard assert."""
    r, _ = _render(merge=False)
    bad = c.Rendered(
        input_ids=r.input_ids,
        slot_idx=dict(r.slot_idx),
        spans=dict(r.spans),
        format=r.format,
        conv_id="bad",
        meta={**r.meta, "slot_char_spans": {**r.meta["slot_char_spans"]}},
        pooled_spans=dict(r.pooled_spans),
    )
    bad.meta["slot_char_spans"]["ctx_preans"] = [A_START, A_START + 3]  # inside answer
    with pytest.raises(AssertionError, match="answer-overlap"):
        slot_diagnostics([bad])


def test_render_r4_recomputes_attr_start_single_match():
    stories = [
        {
            "conv_id": "s0",
            "story": STORY,
            "answer": STORY_ANSWER,
            "parsed_turns": [TURN],
        }
    ]
    tok = FakeTokenizer(merge_quote_into_answer=False)
    rendered, stats = _render_r4(stories, tok, verbatim_check=True, extra_slots=True)
    assert stats["turns_rendered"] == 1
    assert list(rendered[0].pooled_spans) == ["ctx_attrmean"]
    # two attributions -> fail-loud single-match assert at the trust boundary
    two = dict(stories[0])
    two["story"] = STORY + ' ARIA replied: "Extra."'
    with pytest.raises(AssertionError, match="ANSWER_ATTRIB_RE matched 2"):
        _render_r4([two], tok, verbatim_check=False, extra_slots=True)


# ---------------------------------------------------------------------------
# 2. One-forward-pass multi-slot extraction (real process_batch body; the GPU
#    boundary faked with a signature-conformant def)
# ---------------------------------------------------------------------------
def _fake_extract_factory(calls: list, vocab: int = 64):
    def fake_extract_layer_activations(
        model,
        input_ids,
        layers,
        *,
        attention_mask=None,
        return_logits=False,
        detach_to_cpu=False,
    ):
        calls.append(1)
        b, t = input_ids.shape
        h = ex.EXPECTED_HIDDEN
        base = torch.arange(t, dtype=torch.float32).view(1, t, 1)
        captured = {layer: (base + 1000.0 * layer).expand(b, t, h).clone() for layer in layers}
        logits = torch.zeros((b, t, vocab), dtype=torch.float32)
        if return_logits:
            return captured, logits
        return captured

    return fake_extract_layer_activations


def test_process_batch_one_forward_pooled_append(monkeypatch):
    r, _ = _render(merge=False)
    calls: list = []
    monkeypatch.setattr(ex, "extract_layer_activations", _fake_extract_factory(calls))
    align: dict = {}
    fake_model = types.SimpleNamespace(device=torch.device("cpu"))
    records = ex.process_batch(fake_model, [r], list(range(ex.EXPECTED_LAYERS))[:4], 0, align)
    assert len(calls) == 1, "all slot positions must be read from ONE forward pass"
    rec = records[0]
    n_single = len(c.SLOT_SINGLE_ORDER)
    assert rec["slots"].shape == (n_single + 1, ex.EXPECTED_LAYERS, ex.EXPECTED_HIDDEN)
    assert rec["spans_meta"]["slot_names"] == list(c.SLOT_STORE_ORDER)
    # single-position rows == the fake activations at each slot position
    order = sorted(r.slot_idx.items(), key=lambda kv: kv[1])
    for row, (_name, idx) in enumerate(order):
        assert torch.allclose(
            rec["slots"][row, 0].float(), torch.full((ex.EXPECTED_HIDDEN,), float(idx))
        )
    # pooled row == the mean over the pooled span positions (same forward)
    ps, pe = r.pooled_spans["ctx_attrmean"]
    expect = float(np.mean(np.arange(ps, pe)))
    assert torch.allclose(
        rec["slots"][n_single, 0].float(),
        torch.full((ex.EXPECTED_HIDDEN,), expect),
        atol=1e-2,  # bf16 storage granularity
    )
    assert rec["spans_meta"]["pooled_spans"]["ctx_attrmean"] == [ps, pe]


def test_process_batch_default_two_slot_path_unchanged(monkeypatch):
    """pooled_spans empty => the parent 2-slot record shape, byte-identical."""
    tok = FakeTokenizer(merge_quote_into_answer=False)
    r = c.render_story_turn(STORY, TURN, "s0", tok)  # default path
    assert r.pooled_spans == {}
    calls: list = []
    monkeypatch.setattr(ex, "extract_layer_activations", _fake_extract_factory(calls))
    fake_model = types.SimpleNamespace(device=torch.device("cpu"))
    records = ex.process_batch(fake_model, [r], [14], 0, {})
    rec = records[0]
    assert rec["slots"].shape == (2, ex.EXPECTED_LAYERS, ex.EXPECTED_HIDDEN)
    assert rec["spans_meta"]["slot_names"] == ["prefix", "context"]
    assert rec["spans_meta"]["pooled_spans"] == {}


# ---------------------------------------------------------------------------
# 3. Paired-deficit battery + verdict trichotomy (toy draws, real bodies)
# ---------------------------------------------------------------------------
def _toy_arm(rng, n_conv=8, d=2, noise=0.1):
    conv = np.repeat([f"c{i}" for i in range(n_conv)], 2)
    true = rng.normal(size=(len(conv), d))
    pred = true + noise * rng.normal(size=(len(conv), d))
    return {"pred": pred, "true": true, "conv_ids": conv}


def test_paired_deficit_battery_per_draw_max_and_pairing():
    rng = np.random.default_rng(0)
    true_base = _toy_arm(rng)
    arms = {
        "qend": {
            **true_base,
            "pred": true_base["true"] + 0.3 * rng.normal(size=true_base["true"].shape),
        },
        "preans": {
            **true_base,
            "pred": true_base["true"] + 0.6 * rng.normal(size=true_base["true"].shape),
        },
        "chat_matched": {
            **true_base,
            "pred": true_base["true"] + 0.05 * rng.normal(size=true_base["true"].shape),
        },
    }
    bat = sv.paired_deficit_battery(arms, "chat_matched", ["qend", "preans"], n_boot=50, seed=0)
    draws = bat.pop("_draws")
    # D == element-wise max over the slot deficit draws (selection in-draw)
    assert np.allclose(draws["d"], np.maximum(draws["d_k"]["qend"], draws["d_k"]["preans"]))
    # pairing: same counts matrix -> D_k draws reproduce from the shared seed
    suff_q = c.conv_suffstats(arms["qend"]["pred"], arms["qend"]["true"], arms["qend"]["conv_ids"])
    suff_c = c.conv_suffstats(
        arms["chat_matched"]["pred"], arms["chat_matched"]["true"], arms["chat_matched"]["conv_ids"]
    )
    counts = c.bootstrap_counts(len(suff_q["uniq"]), 50, 0)
    expect = c.batched_conv_r2(counts, suff_q) - c.batched_conv_r2(counts, suff_c)
    assert np.allclose(draws["d_k"]["qend"], expect)
    # Bonferroni CI is at least as wide as the 95% CI
    ps = bat["per_slot"]["qend"]
    assert ps["delta_ci_bonferroni4"][0] <= ps["delta_ci95"][0]
    assert ps["delta_ci_bonferroni4"][1] >= ps["delta_ci95"][1]
    # the noisier slot has the larger deficit magnitude
    assert bat["per_slot"]["preans"]["d_k_obs"] < bat["per_slot"]["qend"]["d_k_obs"] < 0


def _bat(d_ci, deltas_hi):
    return {
        "d_ci95": d_ci,
        "per_slot": {k: {"delta_ci_bonferroni4": [-1.0, hi]} for k, hi in deltas_hi.items()},
    }


def test_classify_verdict_trichotomy_endpoint_at_zero_straddles():
    keys = ["qend", "preans"]
    # wholly above 0 -> slot artifact
    assert (
        sv.classify_verdict(_bat([0.01, 0.2], {"qend": -0.1, "preans": -0.1}), keys)
        == "slot_artifact"
    )
    # straddle -> slot artifact
    assert (
        sv.classify_verdict(_bat([-0.1, 0.1], {"qend": -0.1, "preans": -0.1}), keys)
        == "slot_artifact"
    )
    # endpoint EXACTLY 0 counts as straddling (plan §3)
    assert (
        sv.classify_verdict(_bat([-0.2, 0.0], {"qend": -0.1, "preans": -0.1}), keys)
        == "slot_artifact"
    )
    # wholly below + every slot Δ hi < 0 -> representation-level collapse
    assert (
        sv.classify_verdict(_bat([-0.3, -0.1], {"qend": -0.05, "preans": -0.02}), keys)
        == "representation_level_collapse"
    )
    # wholly below but one slot Δ CI reaches 0 -> intermediate
    assert (
        sv.classify_verdict(_bat([-0.3, -0.1], {"qend": 0.0, "preans": -0.02}), keys)
        == "intermediate"
    )


def test_run_verdict_end_to_end_and_degeneracy(tmp_path):
    """REAL run_verdict body on real tmp files: full lattice + the
    inconclusive-by-degeneracy reportable outcome."""
    ts, out, preds = tmp_path / "ts", tmp_path / "out", tmp_path / "preds"
    for p in (ts, out, preds):
        p.mkdir()
    rng = np.random.default_rng(1)
    base = _toy_arm(rng, n_conv=10)
    registered = sorted(set(base["conv_ids"]))
    diag = {
        "answer_overlap_rates": dict.fromkeys(c.SLOT_STORE_ORDER, 0.0),
        "anchor_coincidence_rates": dict.fromkeys([*c.SLOT_SINGLE_ORDER, "ctx_attrmean"], 0.0),
        "degenerate_verdict_slots": {k: False for k in c.SLOT_VERDICT_CELLS},
        "positions": {},
    }
    (ts / f"{c.stem_for('instruct', 'r4slot')}_slot_diagnostics.json").write_text(json.dumps(diag))
    (out / "slot_row_coverage.json").write_text(
        json.dumps({"registered_conv_ids": registered, "n_registered": len(registered)})
    )
    cells = dict(c.SLOT_VERDICT_CELLS)
    cells["chat"] = c.SLOT_CHAT_MATCHED_CELL
    cells["anchor"] = c.SLOT_ANCHOR_CELL
    cells["prefix"] = c.SLOT_PREFIX_CELL
    for cid in cells.values():
        np.savez(
            preds / f"{cid}_L19.npz",
            pred=(base["true"] + 0.2 * rng.normal(size=base["true"].shape)).astype(np.float32),
            true=base["true"].astype(np.float32),
            conv_ids=np.asarray(base["conv_ids"]),
            layer=np.asarray([19]),
        )
    sv.run_verdict(ts, out, preds, n_boot=30, seed=0, smoke=False)
    lat = json.loads((out / "slot_verdict_lattice.json").read_text())
    assert lat["verdict"] in ("slot_artifact", "representation_level_collapse", "intermediate")
    assert set(lat["battery"]["per_slot"]) == set(c.SLOT_VERDICT_CELLS)
    assert (preds / "slot_verdict_draws.npz").exists()
    assert lat["answer_overlap_rates"]["ctx_preans"] == 0.0
    # degeneracy: 3 of 4 slots degenerate -> <2 remain -> inconclusive
    diag["degenerate_verdict_slots"] = {
        "qend": True,
        "preattr": True,
        "preans": True,
        "attrmean": False,
    }
    (ts / f"{c.stem_for('instruct', 'r4slot')}_slot_diagnostics.json").write_text(json.dumps(diag))
    sv.run_verdict(ts, out, preds, n_boot=30, seed=0, smoke=False)
    lat2 = json.loads((out / "slot_verdict_lattice.json").read_text())
    assert lat2["verdict"] == "inconclusive_by_degeneracy"


def test_refit_equality_gate_halts_production_and_demotes_smoke(tmp_path):
    """REAL refit_equality_slots body against the LANDED committed anchors."""
    out = tmp_path / "out"
    out.mkdir()
    # matching values -> PASS
    for cid, ref in c.SLOT_REFIT_ANCHOR_FILES.items():
        landed = json.loads((REPO_ROOT / ref).read_text())["r2_per_layer_obs"][19]
        payload = {"r2_per_layer_obs": [0.0] * 19 + [landed] + [0.0] * 8}
        (out / f"cells_{cid}.json").write_text(json.dumps(payload))
    sv.refit_equality_slots(out, smoke=False)
    rec = json.loads((out / "refit_equality_slots.json").read_text())
    assert rec["pass"] is True
    # one anchor off by > tol -> production exit 3; smoke informational
    bad = json.loads((out / f"cells_{c.SLOT_ANCHOR_CELL}.json").read_text())
    bad["r2_per_layer_obs"][19] += 0.05
    (out / f"cells_{c.SLOT_ANCHOR_CELL}.json").write_text(json.dumps(bad))
    with pytest.raises(SystemExit) as ei:
        sv.refit_equality_slots(out, smoke=False)
    assert ei.value.code == 3
    sv.refit_equality_slots(out, smoke=True)  # no raise
    rec = json.loads((out / "refit_equality_slots.json").read_text())
    assert rec["pass"] is None and rec["mode"] == "smoke-informational"


def test_build_row_coverage_intersection_and_flags(tmp_path, capsys):
    ts, out = tmp_path / "ts", tmp_path / "out"
    ts.mkdir()
    out.mkdir()
    slot_stem = c.stem_for("instruct", "r4slot")
    (ts / f"{slot_stem}_shard000.json").write_text(json.dumps({"conv_ids": ["a", "b", "c", "d"]}))
    (ts / "instruct_chat_s_shard000.json").write_text(
        json.dumps({"conv_ids": ["a", "b", "c", "x"]})
    )
    cov = sv.build_row_coverage(ts, out, smoke=False)
    assert cov["registered_conv_ids"] == ["a", "b", "c"]
    assert cov["drop_flagged_over_1pct"] is True  # 25% > 1%
    # empty intersection: production fail-loud, smoke informational
    (ts / "instruct_chat_s_shard000.json").write_text(json.dumps({"conv_ids": ["x", "y"]}))
    with pytest.raises(AssertionError, match="EMPTY at production"):
        sv.build_row_coverage(ts, out, smoke=False)
    cov = sv.build_row_coverage(ts, out, smoke=True)
    assert cov["registered_conv_ids"] == []


# ---------------------------------------------------------------------------
# 5. Registry + anchor pins
# ---------------------------------------------------------------------------
def test_slot_registry_pins():
    cells = c.slot_ablation_cells()
    assert len(cells) == 7
    by_id = {x["cell_id"]: x for x in cells}
    assert set(by_id) == {
        "R_instruct_r4slot_prefix",
        "R_instruct_r4slot_qend_context",
        "R_instruct_r4slot_preattr_context",
        "R_instruct_r4slot_anchor_context",
        "R_instruct_r4slot_preans_context",
        "R_instruct_r4slot_attrmean_context",
        "R_instruct_r1_matched_context",
    }
    assert c.SLOT_STORE_ORDER == (
        "prefix",
        "ctx_qend",
        "ctx_preattr",
        "context",
        "ctx_preans",
        "ctx_attrmean",
    )
    for cid, (idx, arm) in c.SLOT_CELL_INDEX.items():
        assert by_id[cid]["slot_index"] == idx
        assert by_id[cid]["arm"] == arm
        assert by_id[cid]["format_key"] == "stories_paired_slots"
        assert by_id[cid]["target_turn_index"] == 0
    chat = by_id["R_instruct_r1_matched_context"]
    assert chat["format_key"] == "chat" and chat["slot_index"] == 1
    assert chat["target_turn_index"] == 1
    assert c.stem_for("instruct", "r4slot") == "instruct_stories_paired_slots_s"


def test_refit_anchor_docs_match_landed_jsons():
    for cid, ref in c.SLOT_REFIT_ANCHOR_FILES.items():
        landed = json.loads((REPO_ROOT / ref).read_text())["r2_per_layer_obs"][19]
        assert abs(landed - c.SLOT_REFIT_ANCHOR_DOC[cid]) < 0.005, (cid, landed)


# ---------------------------------------------------------------------------
# 4. Dispatch --dry-run arming (slot mode + legacy unaffected)
# ---------------------------------------------------------------------------
def _run_dispatch(args: list[str], tmp_path: Path, extra_env: dict | None = None):
    env = {k: v for k, v in os.environ.items() if k not in _SEAM_KEYS}
    env.update(
        {
            "REPO_ROOT": str(REPO_ROOT),
            "EPM_LOG_DIR": str(tmp_path / "logs"),
            "EPM_OUTPUT_ROOT": str(tmp_path / "smoke_out"),
        }
    )
    env.update(extra_env or {})
    return subprocess.run(
        ["bash", str(SCRIPTS / "issue1345_dispatch.sh"), *args],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        env=env,
        timeout=300,
    )


def test_dispatch_dry_run_smoke_slot_ablation(tmp_path):
    proc = _run_dispatch(
        ["--variant", "story_slot_ablation", "--slot-ablation", "--dry-run", "--smoke"],
        tmp_path,
        {"EPM_STORY_CHARACTER_NAME": "Assistant"},
    )
    out = proc.stdout
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-2000:]
    assert "variant=story_slot_ablation" in out and "slot_ablation=1" in out
    for phase in (
        "[phase=prefetch_stories]",
        "[phase=prefetch_reuse]",
        "[phase=extract_r4_slots]",
        "[phase=upload_stems]",
        "[phase=fits_slots]",
        "[phase=slot_transfer]",
        "[phase=verdict]",
        "[phase=plots]",
        "[phase=upload]",
        "[phase=push]",
        "[phase=done]",
    ):
        assert phase in out, f"missing {phase}"
    assert "--slot-ablation" in out and "--stems instruct_chat_s" in out
    assert "EPM_STORY_CHARACTER_NAME=Assistant" in out
    assert "EPM_I1345_VARIANT=story_slot_ablation" in out
    # legacy phases must NOT run in slot mode
    for phase in ("[phase=gen_stories]", "[phase=matchedn]", "[phase=fits]", "[phase=opcomp]"):
        assert phase not in out, f"legacy {phase} ran in slot mode"
    sentinel = json.loads((tmp_path / "logs" / "issue-1345-smoke-results.json").read_text())
    assert sentinel["kind"] == "epm:smoke-result"
    assert sentinel["sentinel_schema_version"] == 1
    assert sentinel["note"]["gpu_hours_budgeted"] == 4.0
    assert "slot_verdict_lattice" in sentinel["note"]["eval_numbers"]


def test_dispatch_slot_flag_variant_pairing_fail_loud(tmp_path):
    proc = _run_dispatch(["--slot-ablation", "--dry-run"], tmp_path)
    assert proc.returncode == 1
    assert "requires --variant story_slot_ablation" in proc.stderr
    proc = _run_dispatch(
        ["--variant", "story_slot_ablation", "--dry-run"],
        tmp_path,
        {"EPM_STORY_CHARACTER_NAME": "Assistant"},
    )
    assert proc.returncode == 1
    assert "requires --slot-ablation" in proc.stderr


def test_dispatch_legacy_dry_run_unaffected(tmp_path):
    proc = _run_dispatch(["--dry-run"], tmp_path)
    out = proc.stdout
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-2000:]
    assert "slot_ablation=0" in out
    for phase in ("[phase=prefetch]", "[phase=phase0]", "[phase=fits]", "[phase=done]"):
        assert phase in out
    for phase in ("[phase=prefetch_stories]", "[phase=fits_slots]", "[phase=verdict]"):
        assert phase not in out, f"slot {phase} ran in legacy mode"
    sentinel = json.loads((tmp_path / "logs" / "issue-1345-smoke-results.json").read_text())
    assert sentinel["note"]["gpu_hours_budgeted"] == 14.0
