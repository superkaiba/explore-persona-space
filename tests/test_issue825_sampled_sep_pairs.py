"""#825 sampled-separator-control: prefix-final pair builder + reduce invariants.

Covers (plan v22 section 10 new-code tests): the fixed prefix-final anchor
eligibility (incl. the <=254 seam guard + last-eligible selection), the
per-draw span ladder + drop accounting, window truncation at span-end + 4,
``:cd<k>`` window_id suffixes, pairs_meta completeness for the new mode, the
fit-script ``--mlp-ci`` payload schema (default-off byte-preserving), and the
reduce script's X-identity gate + fp32 mean-Y + K_valid floor.

Uses the REAL Qwen tokenizer for the pair-builder tests (cached on the VM;
the ladder is BPE-offset arithmetic — a fake tokenizer would test nothing);
the reduce tests are synthetic-store CPU-only.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import issue825_onpolicy_sep_pairs as ops  # noqa: E402
import issue825_sampled_sep_reduce as red  # noqa: E402
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


def _cont_row(window_id: str, draw: int, text: str, tokenizer) -> dict:
    return {
        "window_id": window_id,
        "wave": 1,
        "draw": draw,
        "sampling_seed": 4300 + draw,
        "continuation": text,
        "continuation_token_ids": tokenizer(text, add_special_tokens=False)["input_ids"],
    }


# ---------------------------------------------------------------------------
# Prefix-final anchor selection
# ---------------------------------------------------------------------------


def test_prefix_final_anchor_fixed_and_seam_guard(tokenizer):
    """The anchor is the LAST eligible sentence-final single-token anchor at
    token index <= 254 of the PINNED prefix (never the final prefix token),
    with the #931 prev-sentence floor folded into eligibility."""
    art = _article(tokenizer)
    anchor = ops.find_prefix_final_anchor(tokenizer, art)
    assert anchor is not None
    t = anchor["t"]
    assert t <= ops.PREFIX_FINAL_ANCHOR_MAX_INDEX
    assert t < len(anchor["prefix_ids"]) - 1
    tok_text = tokenizer.decode([anchor["prefix_ids"][t]]).strip()
    assert tok_text in (".", "!", "?")
    # Last-eligible: no eligible anchor sits strictly between t and the guard.
    prefix_text = anchor["prefix_text"]
    _ids, offsets = common.tokenize_with_offsets(tokenizer, prefix_text)
    later = [
        tt
        for tt, _s, _c in ops._eligible_sentence_final_anchors(prefix_text, offsets)
        if t < tt <= ops.PREFIX_FINAL_ANCHOR_MAX_INDEX
    ]
    assert not later, later
    # prev-sentence span floor holds on the selected anchor.
    ps_lo, ps_hi = anchor["ps_span"]
    assert ps_hi - ps_lo >= common.ARMC_PREV_MIN_TOKENS
    assert ps_hi <= t
    # Determinism: recomputation is bitwise identical (fixed across draws/models).
    anchor2 = ops.find_prefix_final_anchor(tokenizer, art)
    assert anchor2["t"] == t and anchor2["ps_span"] == anchor["ps_span"]


def test_no_prefix_anchor_article_returns_none(tokenizer):
    """An article whose 256-token prefix has no eligible sentence-final anchor
    returns None (dropped + counted at the mode level)."""
    words = " ".join(f"word{i}" for i in range(400))  # no punctuation at all
    tail = "".join(_sentence(i) for i in range(40))  # periods only PAST the prefix
    ids = tokenizer(words + " " + tail, add_special_tokens=False)["input_ids"]
    assert len(ids) >= common.ARMC_ARTICLE_MIN_TOKENS
    art = {"window_id": "wiki:00099", "title": "no-anchor", "input_ids": ids}
    assert ops.find_prefix_final_anchor(tokenizer, art) is None


# ---------------------------------------------------------------------------
# Per-draw span ladder + drop accounting + truncation + ids
# ---------------------------------------------------------------------------


def test_per_draw_ladder_drops_and_truncation(tokenizer):
    art = _article(tokenizer)
    anchor = ops.find_prefix_final_anchor(tokenizer, art)
    t = anchor["t"]

    # Draw 0: a normal sampled continuation -> kept.
    good = " The board then considered the proposal in considerable detail. And more text."
    arow, pair, c0 = ops.build_prefix_final_draw(
        tokenizer, art, anchor, _cont_row(art["window_id"], 0, good, tokenizer)
    )
    assert pair is not None and not c0
    assert arow["window_id"] == f"{art['window_id']}:cd0"
    assert pair.row_id == f"{art['window_id']}:cd0:a{t}"
    assert pair.meta["draw"] == 0 and pair.meta["anchor_pos"] == t
    span_lo, span_hi = pair.t_spans[0]
    assert span_lo == t + 1
    assert common.ARMC_SPAN_MIN <= span_hi - span_lo <= common.ARMC_SPAN_MAX
    # Window truncation at span-end + 4 (token boundary).
    assert len(arow["input_ids"]) <= span_hi + ops.PREFIX_FINAL_TRUNC_MARGIN
    assert len(arow["input_ids"]) >= span_hi
    # The deterministic prefix-tail nuisance length is recorded.
    assert pair.meta["prefix_tail_tokens"] >= 0

    # Draw 1: NO sentence-final separator anywhere -> per-draw drop.
    no_sep = " " + " ".join(f"item{i}" for i in range(40))
    _arow1, pair1, c1 = ops.build_prefix_final_draw(
        tokenizer, art, anchor, _cont_row(art["window_id"], 1, no_sep, tokenizer)
    )
    assert pair1 is None and c1["no_closing_separator_draw"] == 1

    # Draw 2: first closing separator beyond the 256-token span cap -> drop
    # (~135 two-token words ~ 270 tokens > 256, still under the 1024 window cap
    # so the closing separator survives the window truncation).
    long_run = " " + " ".join(f"token{i}" for i in range(135)) + " ended."
    _arow2, pair2, c2 = ops.build_prefix_final_draw(
        tokenizer, art, anchor, _cont_row(art["window_id"], 2, long_run, tokenizer)
    )
    assert pair2 is None and c2["span_len_out_of_range_draw"] == 1


def test_prefix_final_main_meta_completeness(tokenizer, tmp_path, monkeypatch):
    """End-to-end main() in prefix-final mode: one pair per (article, draw),
    ``:cd<k>`` windows, K_valid + per-draw-yield + prefix-tail meta blocks."""
    arts = [_article(tokenizer, idx=3), _article(tokenizer, idx=4)]
    articles_path = tmp_path / "articles_armC.jsonl"
    articles_path.write_text("".join(json.dumps(a) + "\n" for a in arts))
    good = " The board then considered the proposal in considerable detail. More text follows."
    bad = " " + " ".join(f"item{i}" for i in range(40))  # no separator -> draw drops
    rows = []
    for a in arts:
        for k in range(3):
            text = bad if (a["window_id"].endswith("4") and k == 2) else good
            rows.append(_cont_row(a["window_id"], k, text, tokenizer))
    cont_path = tmp_path / "continuations.jsonl"
    cont_path.write_text("".join(json.dumps(r) + "\n" for r in rows))
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
            "--model",
            "base",
            "--anchor-mode",
            "prefix-final",
            "--followup-label",
            "sampled-separator-control",
        ],
    )
    assert ops.main() == 0
    pairs_dir = out_dir / "pairs"
    windows = {r["window_id"]: r for r in ops.read_jsonl(pairs_dir / "articles_armC.jsonl")}
    pair_rows = ops.read_jsonl(pairs_dir / "pairs_armC.jsonl")
    assert len(pair_rows) == 5  # 2 articles x 3 draws - 1 dropped draw
    for d in pair_rows:
        p = common.PairSpec.from_dict(d)
        assert ":cd" in p.meta["window_id"]
        n_tok = len(windows[p.meta["window_id"]]["input_ids"])
        p.validate(n_tok, min_c=common.ARMC_PREV_MIN_TOKENS, min_t=common.ARMC_SPAN_MIN)
    # One FIXED anchor per article across draws.
    for a in arts:
        anchors = {
            common.PairSpec.from_dict(d).meta["anchor_pos"]
            for d in pair_rows
            if common.PairSpec.from_dict(d).group_id == a["window_id"]
        }
        assert len(anchors) == 1, (a["window_id"], anchors)
    meta = json.loads((pairs_dir / "pairs_meta.json").read_text())
    assert meta["anchor_mode"] == "prefix-final"
    assert meta["followup_label"] == "sampled-separator-control"
    assert meta["target_n"] == 6  # n_articles x n_draws
    assert meta["realized_n"] == 5 and meta["shortfall"] is True
    for key in (
        "per_draw_yield",
        "k_valid_per_article",
        "k_valid_distribution",
        "prefix_tail_tokens",
        "anchor_pos_by_article_summary",
        "drop_counters",
        "onpolicy_stats",
        "seam_token_mismatch",
    ):
        assert key in meta, key
    assert meta["drop_counters"].get("no_closing_separator_draw") == 1
    assert meta["k_valid_distribution"] == {"0": 0, "1": 0, "2": 1, "3": 1}


# ---------------------------------------------------------------------------
# Fit-script --mlp-ci payload schema (default-off byte-preserving)
# ---------------------------------------------------------------------------


def test_mlp_ci_payload_schema(tmp_path):
    import issue931_fit_cells as fit931

    rng = np.random.default_rng(0)
    n, n_layers, d = 60, 2, 6
    x = rng.normal(size=(n, n_layers, d)).astype(np.float32)
    w = rng.normal(size=(d, d)).astype(np.float32)
    y = (x @ w + 0.1 * rng.normal(size=(n, n_layers, d))).astype(np.float32)
    groups = np.asarray([f"g{i % 10}" for i in range(n)])
    xy = {"X": x, "Y": y, "group_ids": groups, "row_ids": np.asarray([f"r{i}" for i in range(n)])}
    results = {"armC_sep": {"xy": xy}}

    def _args(mlp_ci: bool, out: Path) -> SimpleNamespace:
        out.mkdir(parents=True, exist_ok=True)
        return SimpleNamespace(folds=2, seed=0, n_boot=20, out_dir=out, smoke=True, mlp_ci=mlp_ci)

    fit931.run_mlp_secondary(results, _args(True, tmp_path / "on"))
    payload = json.loads((tmp_path / "on" / "mlp_secondary.json").read_text())
    hl = payload["mlp_ci"]["armC_sep"]["headline_layer"]
    ci = payload["mlp_ci"]["armC_sep"][str(hl)]
    for key in ("r2", "ci_lo", "ci_hi", "n_groups", "n_boot"):
        assert key in ci, key
    assert ci["ci_lo"] <= ci["r2"] <= ci["ci_hi"]
    # The CI's observed R^2 tracks the MLP secondary's r2_obs up to the known
    # centering-convention delta: the fold loop pools per-fold-centered ss_tot
    # while group_bootstrap_r2's obs centers on the global fitted mean (the
    # SAME relationship the rotated leg has between random_projection_control
    # and its bootstrap block).
    assert ci["r2"] == pytest.approx(payload["cells"]["armC_sep"][str(hl)]["r2_obs"], abs=5e-3)
    assert payload["mlp_ci"]["armC_sep"]["per_group_mlp_r2_headline"]

    fit931.run_mlp_secondary(results, _args(False, tmp_path / "off"))
    payload_off = json.loads((tmp_path / "off" / "mlp_secondary.json").read_text())
    assert "mlp_ci" not in payload_off  # default byte-preserves the #931 payload


# ---------------------------------------------------------------------------
# Reduce: fp32 mean-Y + K_valid floor + X-identity gate
# ---------------------------------------------------------------------------


def _write_pooled_store(root: Path, x_sep_override=None) -> dict:
    """3 articles x {3, 2, 1} draws; L=4 layers, D=8; returns the arrays."""
    rng = np.random.default_rng(7)
    n_layers, d = 4, 8
    rows = []
    base_x = {a: rng.normal(size=(n_layers, d)).astype(np.float32) for a in (1, 2, 3)}
    draws_of = {1: (0, 1, 2), 2: (0, 1), 3: (1,)}  # art3 has NO draw 0
    for a, draws in draws_of.items():
        for k in draws:
            x = base_x[a] + 1e-6 * rng.normal(size=(n_layers, d)).astype(np.float32)
            rows.append(
                {
                    "row_id": f"wiki:{a:05d}:cd{k}:a42",
                    "group_id": f"wiki:{a:05d}",
                    "char_id": "sep",
                    "x_sep": x,
                    "x_spanmean": base_x[a] * 0.5,
                    "x_last": base_x[a] * 0.25,
                    "y": rng.normal(size=(n_layers, d)).astype(np.float32),
                }
            )
    if x_sep_override is not None:
        rows[2]["x_sep"] = x_sep_override(rows[2]["x_sep"])  # art1 draw 2
    store_dir = root / "store" / "armC"
    store_dir.mkdir(parents=True)
    payload = {
        "row_ids": [r["row_id"] for r in rows],
        "group_ids": [r["group_id"] for r in rows],
        "char_ids": [r["char_id"] for r in rows],
        "arrays": {
            k: torch.from_numpy(np.stack([r[k] for r in rows]))
            for k in ("x_sep", "x_spanmean", "x_last", "y")
        },
    }
    torch.save(payload, store_dir / "armC_shard000.pt")
    (store_dir / "armC_shard000.json").write_text(json.dumps({"n_rows": len(rows)}))
    return {r["row_id"]: r for r in rows}


def _reduce_argv(root: Path, extra: list[str]) -> list[str]:
    return [
        "issue825_sampled_sep_reduce.py",
        "--model",
        "base",
        "--pooled-data-dir",
        str(root / "pooled"),
        "--avg-data-dir",
        str(root / "avg"),
        "--single-data-dir",
        str(root / "single"),
        "--out-dir",
        str(root / "out"),
        "--k-valid-floor",
        "2",
        *extra,
    ]


def test_reduce_mean_y_kvalid_floor_and_allowlist(tmp_path, monkeypatch):
    rows = _write_pooled_store(tmp_path / "pooled")
    monkeypatch.setattr(sys, "argv", _reduce_argv(tmp_path, []))
    assert red.main() == 0
    import issue931_fit_cells as fit931

    avg = fit931.load_regime_store(tmp_path / "avg" / "store" / "armC", "armC")
    single = fit931.load_regime_store(tmp_path / "single" / "store" / "armC", "armC")
    # K_valid floor 2: art1 (K=3) + art2 (K=2) kept; art3 (K=1) dropped.
    assert list(avg["group_ids"]) == ["wiki:00001", "wiki:00002"]
    # C-single = draw-0 allowlist: art3 has no draw 0.
    assert list(single["row_ids"]) == ["wiki:00001:cd0:a42", "wiki:00002:cd0:a42"]
    np.testing.assert_allclose(
        single["arrays"]["y"][0], rows["wiki:00001:cd0:a42"]["y"], rtol=0, atol=1e-7
    )
    # fp32 mean-Y over the valid draws.
    expect = np.stack(
        [rows[f"wiki:00001:cd{k}:a42"]["y"].astype(np.float64) for k in (0, 1, 2)]
    ).mean(axis=0)
    np.testing.assert_allclose(avg["arrays"]["y"][0], expect, rtol=0, atol=1e-6)
    # "Single X": the lowest valid draw's x_sep rides the averaged row.
    np.testing.assert_allclose(
        avg["arrays"]["x_sep"][0], rows["wiki:00001:cd0:a42"]["x_sep"], rtol=0, atol=1e-7
    )
    summary = json.loads((tmp_path / "out" / "reduce_summary.json").read_text())
    assert summary["x_identity_gate"]["pass"] is True
    assert summary["n_articles_below_floor"] == 1
    assert summary["k_valid_distribution"] == {"0": 0, "1": 1, "2": 1, "3": 1}
    assert summary["metadata"]["issue"] == 825


def test_reduce_x_identity_gate_halts_binding(tmp_path, monkeypatch):
    """A structurally corrupted draw x_sep trips the HALT (rc=7) with stores
    still written (upload-then-halt); --smoke records it non-binding (rc=0)."""

    def corrupt(x):
        out = x.copy()
        out[-1] = np.roll(out[-1], 3)  # hl = n_layers - 1 for the tiny store
        return out

    _write_pooled_store(tmp_path / "pooled", x_sep_override=corrupt)
    monkeypatch.setattr(sys, "argv", _reduce_argv(tmp_path, []))
    assert red.main() == red.GATE_RC
    # Upload-then-halt: outputs exist despite the gate breach.
    assert (tmp_path / "avg" / "store" / "armC" / "armC_shard000.pt").exists()
    summary = json.loads((tmp_path / "out" / "reduce_summary.json").read_text())
    assert summary["x_identity_gate"]["pass"] is False
    assert summary["x_identity_gate"]["min"] < red.X_IDENTITY_COS_MIN

    # Smoke: recorded, non-binding.
    for d in ("avg", "single", "out"):
        import shutil

        shutil.rmtree(tmp_path / d, ignore_errors=True)
    monkeypatch.setattr(sys, "argv", _reduce_argv(tmp_path, ["--smoke"]))
    assert red.main() == 0
    summary = json.loads((tmp_path / "out" / "reduce_summary.json").read_text())
    assert summary["x_identity_gate"]["binding"] is False
