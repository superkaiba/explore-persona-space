"""#825 onpolicy-separator-control: pair-builder + fit-schema invariants.

Covers (plan section 10 new-code tests): the G2b anchor-region constraint, the
straddling-span clamp-drop path, wave-2 windowing (+ the <=6/article cap
across waves), pairs_meta completeness, and the fit-output JSON schema assert
(a group-bootstrap CI keyed to the ROTATED estimator at the headline layer —
the issue931_fit_cells --rotated-ci extension, default-off byte-preserving).

Uses the REAL Qwen tokenizer (cached on the VM; the ladder is BPE-offset
arithmetic — a fake tokenizer would test nothing).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import issue825_onpolicy_sep_pairs as ops  # noqa: E402
import issue931_common as common  # noqa: E402


@pytest.fixture(scope="module")
def tokenizer():
    return common.get_tokenizer()


def _sentence(i: int) -> str:
    return (
        f"The committee reviewed item number {i} and approved the annual budget "
        "for the regional office after a long discussion. "
    )


def _article(tokenizer, idx: int = 7, n_sentences: int = 80) -> dict:
    text = "".join(_sentence(i) for i in range(n_sentences))
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    assert len(ids) >= common.ARMC_ARTICLE_MIN_TOKENS, len(ids)
    return {"window_id": f"wiki:{idx:05d}", "title": "test article", "input_ids": ids}


def _continuation(n_sentences: int, tag: str = "board") -> str:
    return "".join(
        f" The {tag} then considered proposal number {i} in considerable detail "
        "before casting the final vote on the matter."
        for i in range(n_sentences)
    )


def test_anchor_region_constraint(tokenizer):
    """Every eligible anchor sits in the CONTINUATION region; prefix-region
    anchors are counted (the prefix text is sentence-rich by construction)."""
    art = _article(tokenizer)
    wr = ops.build_wave(tokenizer, art, _continuation(6), wave=1)
    assert wr["counters"]["anchor_in_prefix_region"] > 0
    assert wr["eligible"], "expected eligible continuation anchors"
    offsets = wr["offsets"]
    for t, span_lo, *_rest in wr["eligible"]:
        assert int(offsets[t, 0]) >= wr["cont_char_start"], (t, wr["cont_char_start"])
        assert t < span_lo


def test_clamp_drop_path(tokenizer):
    """A straddling first sentence whose clamped prev-span shrinks below the
    8-token floor DROPS the pair with the dedicated counter (G2b constraint 2)."""
    art = _article(tokenizer)
    # The 256-token prefix decode ends mid-sentence; " done." closes that
    # straddling sentence with ~1-2 continuation tokens before the anchor.
    wr = ops.build_wave(tokenizer, art, " done." + _continuation(6), wave=1)
    assert wr["counters"]["prev_span_clamped_below_min"] >= 1, dict(wr["counters"])
    # No eligible pair keeps a sub-floor prev-span.
    for _t, _lo, _hi, ps_lo, ps_hi, _sep in wr["eligible"]:
        assert ps_hi - ps_lo >= common.ARMC_PREV_MIN_TOKENS


def test_wave2_windowing_and_article_cap(tokenizer):
    """Wave-2 tops up the per-ARTICLE budget: w2 rows carry the :w2 window_id,
    the article group_id, and wave1+wave2 kept <= 6 per article."""
    art = _article(tokenizer)
    # Wave 1: 3 sentences -> 2 eligible anchors (the last anchor's span to the
    # window end is sub-floor and drops).
    wr1 = ops.build_wave(tokenizer, art, _continuation(3), wave=1)
    wr2 = ops.build_wave(tokenizer, art, _continuation(12, tag="council"), wave=2)
    assert 0 < len(wr1["eligible"]) < ops.WAVE2_MIN_ELIGIBLE
    assert len(wr2["eligible"]) >= 4
    kept = ops.select_article_pairs(art["window_id"], {1: wr1, 2: wr2}, max_anchors=6)
    waves = [w for w, _ in kept]
    assert len(kept) <= 6
    assert waves.count(1) == len(wr1["eligible"])
    assert waves.count(2) == 6 - len(wr1["eligible"])
    for w, p in kept:
        assert p.group_id == art["window_id"]
        expected_wid = art["window_id"] if w == 1 else f"{art['window_id']}:w2"
        assert p.meta["window_id"] == expected_wid
        assert p.row_id.startswith(expected_wid + ":a")
        assert p.meta["wave"] == w


def test_max_anchor_cap_single_wave(tokenizer):
    """A rich wave-1 continuation is capped at 6 kept anchors (seeded draw)."""
    art = _article(tokenizer, idx=13)
    wr1 = ops.build_wave(tokenizer, art, _continuation(14), wave=1)
    assert len(wr1["eligible"]) > 6
    kept = ops.select_article_pairs(art["window_id"], {1: wr1}, max_anchors=6)
    assert len(kept) == 6
    assert len({p.row_id for _, p in kept}) == 6
    # Determinism: the seeded per-article draw reproduces.
    kept2 = ops.select_article_pairs(art["window_id"], {1: wr1}, max_anchors=6)
    assert [p.row_id for _, p in kept] == [p.row_id for _, p in kept2]


def _exo_pair_row(tokenizer, art: dict) -> dict:
    ids = art["input_ids"]
    t = next(
        i for i, tid in enumerate(ids[:500]) if tokenizer.decode([tid]).strip() == "." and i > 30
    )
    p = common.PairSpec(
        row_id=f"{art['window_id']}:a{t}",
        group_id=art["window_id"],
        char_id="sep",
        c_span=(t - 10, t),
        t_spans=[(t + 1, t + 20)],
        ctx_span=(t - 10, t),
        meta={"window_id": art["window_id"], "anchor_pos": int(t)},
    )
    return p.to_dict()


def test_main_outputs_and_meta_completeness(tokenizer, tmp_path, monkeypatch):
    """End-to-end main(): consumer-exact outputs + pairs_meta completeness +
    every emitted pair validates against its own window ids (the extractor's
    span contract)."""
    arts = [_article(tokenizer, idx=3), _article(tokenizer, idx=4)]
    articles_path = tmp_path / "articles_armC.jsonl"
    articles_path.write_text("".join(json.dumps(a) + "\n" for a in arts))
    cont_rows = []
    for a in arts:
        cont = _continuation(10)
        ids = tokenizer(cont, add_special_tokens=False)["input_ids"]
        cont_rows.append(
            {
                "window_id": a["window_id"],
                "wave": 1,
                "continuation": cont,
                "continuation_token_ids": ids,
            }
        )
    cont_path = tmp_path / "continuations.jsonl"
    cont_path.write_text("".join(json.dumps(r) + "\n" for r in cont_rows))
    exo_pairs_path = tmp_path / "exo_pairs.jsonl"
    exo_pairs_path.write_text("".join(json.dumps(_exo_pair_row(tokenizer, a)) + "\n" for a in arts))
    out_dir = tmp_path / "out"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "issue825_onpolicy_sep_pairs.py",
            "--articles",
            str(articles_path),
            "--continuations",
            str(cont_path),
            "--out-data-dir",
            str(out_dir),
            "--exogenous-articles",
            str(articles_path),
            "--exogenous-pairs",
            str(exo_pairs_path),
            "--model",
            "base",
        ],
    )
    assert ops.main() == 0
    pairs_dir = out_dir / "pairs"
    windows = {r["window_id"]: r for r in ops.read_jsonl(pairs_dir / "articles_armC.jsonl")}
    pair_rows = ops.read_jsonl(pairs_dir / "pairs_armC.jsonl")
    assert windows and pair_rows
    for d in pair_rows:
        p = common.PairSpec.from_dict(d)
        n_tok = len(windows[p.meta["window_id"]]["input_ids"])
        p.validate(n_tok, min_c=common.ARMC_PREV_MIN_TOKENS, min_t=common.ARMC_SPAN_MIN)
    meta = json.loads((pairs_dir / "pairs_meta.json").read_text())
    for key in (
        "metadata",
        "model",
        "realized_n",
        "target_n",
        "shortfall",
        "n_wave2_windows",
        "drop_counters",
        "onpolicy_stats",
        "exogenous_stats",
        "seam_token_mismatch",
        "wave_geometry",
    ):
        assert key in meta, key
    assert meta["realized_n"] == len(pair_rows)
    assert meta["shortfall"] is True  # 2 articles can never reach 3600
    for stats_key in ("separator_frequencies", "span_length", "anchor_position"):
        assert stats_key in meta["onpolicy_stats"]
        assert stats_key in meta["exogenous_stats"]
    assert 0.0 <= meta["seam_token_mismatch"]["rate"] <= 1.0


def test_fit_output_schema_rotated_ci(tmp_path):
    """Plan section 10 schema assert: with --rotated-ci the fit-output JSON
    carries a group-bootstrap CI keyed to the ROTATED estimator at the
    headline layer (+ the per-group rotated persist); without the flag the
    payload keys are ABSENT (default byte-preserves #931)."""
    import issue931_fit_cells as fit931

    rng = np.random.default_rng(0)
    n, n_layers, d = 24, 2, 6
    x = rng.normal(size=(n, n_layers, d)).astype(np.float32)
    w = rng.normal(size=(d, d)).astype(np.float32)
    y = (x @ w + 0.1 * rng.normal(size=(n, n_layers, d))).astype(np.float32)
    groups = np.asarray([f"g{i % 6}" for i in range(n)])
    xy = {"X": x, "Y": y, "group_ids": groups, "row_ids": np.asarray([f"r{i}" for i in range(n)])}

    def _args(rotated: bool, out: Path) -> SimpleNamespace:
        out.mkdir(parents=True, exist_ok=True)
        return SimpleNamespace(
            folds=2, seed=0, null_draws=1, n_boot=20, out_dir=out, data_dir=out, rotated_ci=rotated
        )

    res = fit931.fit_cell("armC_sep", xy, _args(True, tmp_path / "on"))
    payload = json.loads((tmp_path / "on" / "cells_armC_sep.json").read_text())
    hl = str(payload["headline_layer"])
    ci = payload["rotated_bootstrap_group_frozen"][hl]
    for key in ("r2", "ci_lo", "ci_hi", "n_groups", "n_boot"):
        assert key in ci, key
    assert ci["ci_lo"] <= ci["r2"] <= ci["ci_hi"]
    assert payload["per_group_rotated_r2_headline"], "per-group rotated persist missing"
    # Parity: the extension's headline rotated R^2 equals the payload's
    # random_projection_control_r2 value (same rng position, same fit path).
    assert payload["rotated_r2_frozen"][hl] == pytest.approx(
        payload["random_projection_control_r2"][hl], abs=1e-8
    )
    assert res["payload"]["cell_id"] == "armC_sep"

    fit931.fit_cell("armC_sep", xy, _args(False, tmp_path / "off"))
    payload_off = json.loads((tmp_path / "off" / "cells_armC_sep.json").read_text())
    assert "rotated_bootstrap_group_frozen" not in payload_off
    assert "per_group_rotated_r2_headline" not in payload_off
    assert "rotated_r2_frozen" not in payload_off
    assert "per_group_r2_headline" in payload_off  # pre-existing ridge key stays
