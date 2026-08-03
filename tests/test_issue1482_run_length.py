"""Vectorized per-token reductions in ``scripts/issue1482_run_length.py``.

The run-length / persistence / variance / template accumulators are a single
sorted-COO pass over the token axis (no Python loop over features or tokens),
so a transposed sort key or an off-by-one in the adjacency test would silently
produce plausible-but-wrong per-feature covariates. These tests execute the
PRODUCTION body ``Accum.add_side`` against a naive brute-force reference.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

REPO = Path(__file__).resolve().parents[1]


def _load_module():
    """Import the driver by path (scripts/ is not a package)."""
    sys.path.insert(0, str(REPO / "scripts"))
    spec = importlib.util.spec_from_file_location(
        "issue1482_run_length", REPO / "scripts" / "issue1482_run_length.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


RL = _load_module()


def _tiny_accum(dict_size: int = 6):
    """Accum over a small dictionary so the brute-force reference is cheap."""
    RL.DICT_SIZE = dict_size
    return RL.Accum("cpu")


def _brute_answer(codes: np.ndarray, pos: np.ndarray, tmpl: np.ndarray):
    """Naive per-feature reference over one answer span.

    codes: (T_kept, D) activation values; pos: original token positions;
    tmpl: bool template mask aligned to pos.
    """
    n_tok, d = codes.shape
    out = {
        k: np.zeros(d)
        for k in ("ans_tok", "pairs", "denom", "var_sum", "var_rows", "cnt", "tmpl_tok")
    }
    pos_set = set(int(p) for p in pos)
    for f in range(d):
        act = [t for t in range(n_tok) if codes[t, f] > 0]
        out["ans_tok"][f] = len(act)
        out["cnt"][f] = 1.0 if act else 0.0
        out["tmpl_tok"][f] = sum(1 for t in act if tmpl[t])
        for t in act:
            if int(pos[t]) + 1 in pos_set:
                out["denom"][f] += 1
            nxt = [u for u in act if int(pos[u]) == int(pos[t]) + 1]
            if nxt:
                out["pairs"][f] += 1
        if act:
            trace = codes[:, f].astype(np.float64)  # zeros INCLUDED (activation trace)
            out["var_sum"][f] = trace.var()
            out["var_rows"][f] = 1.0
    return out


def _run_side(acc, codes: np.ndarray, pos: np.ndarray, tmpl: np.ndarray, side: str) -> None:
    acc.add_side(
        torch.tensor(codes, dtype=torch.float32),
        torch.tensor(pos, dtype=torch.int64),
        torch.tensor(tmpl, dtype=torch.bool),
        side,
    )


def test_answer_side_matches_brute_force_reference():
    rng = np.random.default_rng(1482)
    d = 6
    # 10 contiguous positions, ~40% density -> runs of assorted lengths.
    codes = (rng.random((10, d)) < 0.4) * rng.random((10, d)).astype(np.float32)
    pos = np.arange(10, dtype=np.int64)
    tmpl = np.zeros(10, dtype=bool)
    tmpl[:2] = True

    acc = _tiny_accum(d)
    _run_side(acc, codes, pos, tmpl, "ans")
    ref = _brute_answer(codes, pos, tmpl)

    for key in ("ans_tok", "pairs", "denom", "cnt", "tmpl_tok"):
        np.testing.assert_allclose(getattr(acc, key).numpy(), ref[key], err_msg=key)
    np.testing.assert_allclose(acc.var_sum.numpy(), ref["var_sum"], rtol=1e-6, atol=1e-9)
    np.testing.assert_allclose(acc.var_rows.numpy(), ref["var_rows"])


def test_masked_out_token_breaks_a_run_not_splices_it():
    """A gap in ORIGINAL positions must break the run.

    Feature 0 fires on kept tokens at original positions [0, 1, 3, 4]; position
    2 was dropped by the token-pool mask. The correct answer is TWO runs of
    length 2 (mean 2.0), not one run of 4. Keying adjacency on the kept-index
    instead of the original position is exactly the bug this pins.
    """
    d = 2
    codes = np.zeros((4, d), dtype=np.float32)
    codes[:, 0] = 1.0
    pos = np.array([0, 1, 3, 4], dtype=np.int64)
    tmpl = np.zeros(4, dtype=bool)

    acc = _tiny_accum(d)
    _run_side(acc, codes, pos, tmpl, "ans")

    ans_tok = float(acc.ans_tok[0])
    pairs = float(acc.pairs[0])
    runs = ans_tok - pairs
    assert ans_tok == 4.0
    assert pairs == 2.0, "adjacency must be ORIGINAL-position, not kept-index"
    assert runs == 2.0
    assert ans_tok / runs == 2.0
    # denom: only positions 0 and 3 have their ORIGINAL successor in the pool
    # (1 -> 2 was masked out, 4 -> 5 is past the span end).
    assert float(acc.denom[0]) == 2.0
    # With mask gaps the two estimators DIVERGE (here p = 2/2 = 1 while
    # mean_run_length = 2). E[R] = 1/(1-p) is a stationary-process identity, not
    # a numeric identity of these finite-span estimators — the consumer plots
    # only mean_run_length, and the meta records the caveat.
    assert float(acc.pairs[0]) / float(acc.denom[0]) == 1.0


def test_perfectly_persistent_feature_has_run_length_equal_to_span():
    d = 3
    n = 12
    codes = np.zeros((n, d), dtype=np.float32)
    codes[:, 1] = 0.5  # fires on every token
    pos = np.arange(n, dtype=np.int64)
    acc = _tiny_accum(d)
    _run_side(acc, codes, pos, np.zeros(n, dtype=bool), "ans")

    runs = float(acc.ans_tok[1] - acc.pairs[1])
    assert runs == 1.0
    assert float(acc.ans_tok[1]) / runs == float(n)
    # p = pairs/denom = 11/11 = 1 -> E[R] = 1/(1-p) diverges; a never-ending run.
    assert float(acc.pairs[1]) == float(acc.denom[1]) == n - 1


def test_isolated_firings_give_unit_run_length_and_zero_persistence():
    d = 2
    codes = np.zeros((6, d), dtype=np.float32)
    codes[[0, 2, 4], 0] = 1.0  # alternating: never two in a row
    pos = np.arange(6, dtype=np.int64)
    acc = _tiny_accum(d)
    _run_side(acc, codes, pos, np.zeros(6, dtype=bool), "ans")

    assert float(acc.pairs[0]) == 0.0
    runs = float(acc.ans_tok[0] - acc.pairs[0])
    assert float(acc.ans_tok[0]) / runs == 1.0


def test_context_side_updates_only_context_accumulators():
    """Context rows must not leak into the answer-side run/variance reads."""
    d = 4
    codes = np.zeros((5, d), dtype=np.float32)
    codes[:, 2] = 1.0
    pos = np.arange(5, dtype=np.int64)
    tmpl = np.array([True, True, False, False, False])

    acc = _tiny_accum(d)
    _run_side(acc, codes, pos, tmpl, "ctx")

    assert float(acc.ctx_tok[2]) == 5.0
    assert float(acc.psi_cnt[2]) == 1.0
    assert float(acc.tmpl_tok[2]) == 2.0, "template firings accumulate on BOTH sides"
    assert float(acc.ans_tok.sum()) == 0.0
    assert float(acc.pairs.sum()) == 0.0
    assert float(acc.cnt.sum()) == 0.0
    assert float(acc.var_rows.sum()) == 0.0


def test_runs_never_span_two_rows():
    """Two rows each ending/starting active must not fuse into one run."""
    d = 2
    acc = _tiny_accum(d)
    for _ in range(2):
        codes = np.ones((3, d), dtype=np.float32)
        _run_side(acc, codes, np.arange(3, dtype=np.int64), np.zeros(3, dtype=bool), "ans")
    ans_tok, pairs = float(acc.ans_tok[0]), float(acc.pairs[0])
    assert ans_tok == 6.0
    assert pairs == 4.0, "2 pairs per row; a cross-row pair would make 5"
    assert ans_tok - pairs == 2.0  # exactly one run per row


def test_template_mask_definition_matches_docstring_rules():
    """TEMPLATE_MASK_DEF rules (1)(2)(3) on a synthetic id sequence."""
    RL._SPECIAL_IDS.clear()
    RL._SPECIAL_IDS.extend([90, 91])  # stand-ins for <|im_start|> / <|im_end|>
    ids = np.array([90, 5, 6, 91, 90, 7, 8, 9, 91, 90, 10, 11, 12, 13], dtype=np.int64)
    prefix_end, context_end = 4, 10
    m = RL._template_mask(ids, prefix_end, context_end, "cpu").numpy()

    assert m[: prefix_end + 1].all(), "rule (1): constant prefix"
    assert m[context_end - 2 : context_end + 1].all(), "rule (2): generation suffix"
    assert m[3] and m[8], "rule (3): special ids anywhere"
    assert not m[11] and not m[12] and not m[13], "answer content is NOT template"


def test_checkpoint_restore_reproduces_an_uninterrupted_run(tmp_path):
    """A resumed capture must equal the same rows captured in one pass.

    The 120k-row capture is ~76 min, so a crash without this is a total loss.
    Splitting the row stream in half and restoring from a checkpoint has to be
    bit-identical to never having stopped.
    """
    rng = np.random.default_rng(7)
    d = 5
    rows = [((rng.random((6, d)) < 0.5) * rng.random((6, d))).astype(np.float32) for _ in range(8)]
    pos = np.arange(6, dtype=np.int64)
    tmpl = np.zeros(6, dtype=bool)

    one = _tiny_accum(d)
    for c in rows:
        _run_side(one, c, pos, tmpl, "ans")

    first = _tiny_accum(d)
    for c in rows[:4]:
        _run_side(first, c, pos, tmpl, "ans")
    np.savez(tmp_path / "ck.npz", **first.state(np.arange(4, dtype=np.int64), "fp-abc"))

    second = _tiny_accum(d)
    with np.load(tmp_path / "ck.npz", allow_pickle=False) as z:
        assert str(z["fingerprint"]) == "fp-abc"
        done = second.restore(z)
    assert done.tolist() == [0, 1, 2, 3]
    for c in rows[4:]:
        _run_side(second, c, pos, tmpl, "ans")

    for key in RL.Accum._ARRAYS:
        np.testing.assert_allclose(
            getattr(second, key).numpy(), getattr(one, key).numpy(), rtol=1e-12, err_msg=key
        )
    for key in RL.Accum._SCALARS:
        assert getattr(second, key) == getattr(one, key), key


def test_atomic_savez_lands_at_the_exact_path(tmp_path):
    """np.savez APPENDS .npz to a suffix-less PATH argument.

    The checkpoint temp name is dotted (``.capture_ckpt_<fp>.npz.tmp<pid>``), so
    a path-argument savez writes ``<tmp>.npz`` and the follow-up os.replace dies
    with FileNotFoundError — which is exactly what killed the first full-corpus
    capture 5,000 rows in. The helper must write through a handle.
    """
    dest = tmp_path / "capture_ckpt_deadbeef.npz"
    RL._atomic_savez(dest, a=np.arange(4), fingerprint=np.asarray("deadbeef"))

    assert dest.exists(), "checkpoint did not land at the requested path"
    strays = [q.name for q in tmp_path.iterdir() if q != dest]
    assert not strays, f"temp/suffixed residue left behind: {strays}"
    with np.load(dest, allow_pickle=False) as z:
        np.testing.assert_array_equal(z["a"], np.arange(4))
        assert str(z["fingerprint"]) == "deadbeef"

    # overwrite in place (a later checkpoint replaces the earlier one)
    RL._atomic_savez(dest, a=np.arange(9), fingerprint=np.asarray("deadbeef"))
    with np.load(dest, allow_pickle=False) as z:
        assert z["a"].size == 9


def test_regime_fingerprint_separates_output_affecting_knobs():
    """A resume against a different regime must not silently fuse populations."""
    import argparse

    pool = np.arange(100, dtype=np.int64)

    def fp(pool_arr=pool, **kw):
        base = dict(n_rows=2000, sample_mode="full", n_chunks=128, tiny_model=False, resume=True)
        return RL._regime_fingerprint(argparse.Namespace(**(base | kw)), pool_arr)

    ref = fp()
    assert fp() == ref, "fingerprint must be stable for an identical regime"
    assert fp(n_rows=120000) != ref
    assert fp(sample_mode="chunk-subset") != ref
    assert fp(tiny_model=True) != ref
    assert fp(pool_arr=np.arange(99, dtype=np.int64)) != ref, "fit pool must enter the key"


def test_no_python_loop_over_features_or_tokens_in_add_side():
    """Throughput invariant: the reduction is a vectorized COO pass.

    Sliced from the file TEXT, not ``inspect.getsource``: the module is loaded
    by path and a stale ``__pycache__`` entry carries old line numbers, which
    silently hands ``getsource`` a different function's body.
    """
    src = (REPO / "scripts" / "issue1482_run_length.py").read_text()
    start = src.index("    def add_side(")
    rest = src[start + 1 :]
    # end of the method = the next line beginning at column 0 (a top-level def
    # /decorator) or the next sibling method, whichever comes first.
    ends = [i for i in (rest.find("\n    def "), rest.find("\n@"), rest.find("\ndef ")) if i != -1]
    body = rest[: min(ends)] if ends else rest
    assert "bincount" in body and "scatter_add_" in body and "argsort" in body, (
        "sliced the wrong span — add_side markers absent"
    )
    assert "for " not in body, "add_side must stay loop-free (vectorize-first rule)"
