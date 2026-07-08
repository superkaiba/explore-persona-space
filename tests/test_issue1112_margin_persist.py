"""#1112 — margin companion phase + m2 selected-checkpoint persist (round 2).

Body tests per code-style.md "one production-body test per seam-stubbed
function": ``phase_margin`` / ``_margin_pools`` / ``_persist_marker_ft``
execute their REAL bodies; fakes sit ONLY at external boundaries —
``read_fn_factory`` (the GPU HF-model seam, signature-conformant closure
returning real ``MarginResult`` instances), ``fu1.derive_margin_pools_topup``
(the sidecar-filesystem boundary, ``create_autospec``), and ``hub._upload``
(the network boundary, ``create_autospec``). ``_stage_file`` short-circuits on
pre-created dest files, so no network is touched.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest import mock

import pytest

from explore_persona_space.eval.margin import MarginResult

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

import issue1112_dispatch as d  # noqa: E402

from explore_persona_space.experiments import issue_1112 as C  # noqa: E402

CELL = "s3_fullft_neg"
POOL_SHA = "a" * 64


def _fake_pools(n: int = 6) -> tuple[list[dict], list[dict], dict]:
    def _pair(i: int, arm: str) -> dict:
        return {
            "probe": f"probe {i}?",
            "answer": f"{arm} answer {i}.",
            "question_id": f"q{i}",
            "variant_id": "v0",
            "request_id": f"{arm}{i}",
            "source": "topup",
        }

    pos = [_pair(i, "pos") for i in range(n)]
    neg = [_pair(i, "neg") for i in range(n)]
    return pos, neg, {"pool_sha256": POOL_SHA, "n_pos_used": n, "n_neg_used": n}


def _stage_margin_fixture(out_root: Path, pinned_sha: str = POOL_SHA) -> None:
    """Pre-create the staged sidecar dest files (so ``_stage_file``
    short-circuits offline) + the pinned fu1 margin record."""
    inputs = out_root / "inputs"
    for rel in C.C3_MARGIN_SIDECARS:
        p = inputs / "c3_cell" / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("")
    pinned = inputs / "fu1_margin_c3.json"
    pinned.parent.mkdir(parents=True, exist_ok=True)
    pinned.write_text(json.dumps({"pool": {"pool_sha256": pinned_sha}}))


def _cfg(tmp_path: Path) -> d.Cfg:
    return d.Cfg(
        smoke=True,
        cells=(CELL,),
        out_root=tmp_path,
        eval_question_limit=2,
        upload=False,
        sentinel_dir=tmp_path / "logs",
    )


def _fake_read_factory(calls: list):
    """Signature-conformant GPU-boundary fake: mirrors organisms.MarginReadFn
    ``(side_path, ctx, pos_pairs, neg_pairs) -> MarginResult`` and returns a
    REAL MarginResult; trained sides shift pos ln-logP by +0.5 (so the
    adapter-application assert has a nonzero delta to pass on)."""

    def factory(base_model: str):
        def read(side_path, ctx, pos_pairs, neg_pairs):
            calls.append((side_path, ctx.context_id if hasattr(ctx, "context_id") else None))
            bump = 0.0 if side_path is None else 0.5
            pos_lp = [-1.0 + bump] * len(pos_pairs)
            neg_lp = [-2.0] * len(neg_pairs)
            pos_mean = sum(pos_lp) / len(pos_lp)
            neg_mean = sum(neg_lp) / len(neg_lp)
            return MarginResult(
                margin=pos_mean - neg_mean,
                pos_mean_ln_logp=pos_mean,
                neg_mean_ln_logp=neg_mean,
                n_pos=len(pos_pairs),
                n_neg=len(neg_pairs),
                pos_ln_logp=pos_lp,
                neg_ln_logp=neg_lp,
            )

        read.close = lambda: None
        return read

    return factory


def _stage_cell(out_root: Path, step: int = 2) -> None:
    cell_root = out_root / CELL
    ckpt = cell_root / "train" / f"checkpoint-{step}"
    ckpt.mkdir(parents=True, exist_ok=True)
    (ckpt / "config.json").write_text("{}")
    (cell_root / "build_result.json").write_text(
        json.dumps({"adapter_root": str(cell_root / "train")})
    )


def test_margin_pools_sha_gate(tmp_path):
    """_margin_pools real body: staged-file short-circuit, derivation call,
    and the pool-pin sha assert (match passes; mismatch raises loud)."""
    cfg = _cfg(tmp_path)
    _stage_margin_fixture(tmp_path)
    fake = mock.create_autospec(d.fu1.derive_margin_pools_topup, return_value=_fake_pools())
    with mock.patch.object(d.fu1, "derive_margin_pools_topup", fake):
        pos, neg, meta = d._margin_pools(cfg)
    assert len(pos) == len(neg) == 6
    assert meta["pool_sha256"] == POOL_SHA
    fake.assert_called_once()

    _stage_margin_fixture(tmp_path, pinned_sha="b" * 64)
    (tmp_path / "inputs" / "fu1_margin_c3.json").write_text(
        json.dumps({"pool": {"pool_sha256": "b" * 64}})
    )
    with (
        mock.patch.object(d.fu1, "derive_margin_pools_topup", fake),
        pytest.raises(RuntimeError, match="pool sha mismatch"),
    ):
        d._margin_pools(cfg)


def test_phase_margin_body_and_resume(tmp_path):
    """phase_margin real body: shared base sweep + per-cell trained sweep,
    per-read checkpointing, adapter assert, aggregation, smoke deliver mirror,
    and the resume predicate (a second run performs ZERO new reads)."""
    cfg = _cfg(tmp_path)
    _stage_margin_fixture(tmp_path)
    _stage_cell(tmp_path)
    selections = {CELL: {"step": 2}}
    calls: list = []
    fake_derive = mock.create_autospec(d.fu1.derive_margin_pools_topup, return_value=_fake_pools())
    with mock.patch.object(d.fu1, "derive_margin_pools_topup", fake_derive):
        out = d.phase_margin(cfg, selections, read_fn_factory=_fake_read_factory(calls))

    rec = out[CELL]
    # smoke slice: 4/4 pairs (MARGIN_POOL_SMOKE_N) after the full-cap sha gate
    assert rec["regime"]["n_pos"] == C.MARGIN_POOL_SMOKE_N
    # contexts: source_ctx + 2 questions, per side (base shared + 1 trained)
    n_ctx = 3
    assert len(calls) == 2 * n_ctx
    assert len(rec["reads"]) == n_ctx
    assert rec["adapter_assert"]["n_pairs"] == C.MARGIN_POOL_SMOKE_N
    # aggregation: trained - base = +0.5 on every context
    assert rec["margin_delta"] == pytest.approx(0.5)
    assert rec["source_ctx"]["delta"] == pytest.approx(0.5)
    assert rec["status"] == "computed"
    # shared base record + per-cell record + smoke deliver mirror on disk
    assert (tmp_path / "margin" / "base.json").exists()
    assert (tmp_path / "margin" / f"{CELL}.json").exists()
    assert (tmp_path / "eval_results_mirror" / "install" / f"{CELL}_margin.json").exists()

    # resume: every read already checkpointed -> the factory is never invoked
    calls2: list = []
    with mock.patch.object(d.fu1, "derive_margin_pools_topup", fake_derive):
        d.phase_margin(cfg, selections, read_fn_factory=_fake_read_factory(calls2))
    assert calls2 == []


def _persist_fixture(tmp_path: Path) -> d.Cfg:
    cfg = d.Cfg(
        smoke=False,
        cells=("m2_fullft_band8",),
        out_root=tmp_path,
        upload=True,
        sentinel_dir=tmp_path / "logs",
    )
    cell_root = tmp_path / "m2_fullft_band8"
    for step in (2, 4):
        ckpt = cell_root / "train" / f"checkpoint-{step}"
        ckpt.mkdir(parents=True, exist_ok=True)
        (ckpt / "model.safetensors").write_text("x")
    (cell_root / "build_result.json").write_text(
        json.dumps({"adapter_root": str(cell_root / "train")})
    )
    (cell_root / "selection.json").write_text(json.dumps({"step": 4}))
    return cfg


def test_persist_marker_ft_upload_then_reap(tmp_path):
    """_persist_marker_ft real body: selected rung uploaded to the overflow
    repo, non-selected rungs reaped ONLY after, selected rung retained for
    capture, idempotent on re-run."""
    cfg = _persist_fixture(tmp_path)
    cell_root = tmp_path / "m2_fullft_band8"
    up = mock.create_autospec(d.hub._upload, return_value="https://hf.co/overflow/x")
    with mock.patch.object(d.hub, "_upload", up):
        rec = d._persist_marker_ft(cfg)
    assert rec["uploaded"] == "issue1112/m2_fullft_band8/checkpoint-4"
    assert rec["cleaned"] == [2]
    args, kwargs = up.call_args
    assert str(args[0]).endswith("checkpoint-4")
    assert args[1] == C.OVERFLOW_REPO
    assert kwargs.get("private") is True
    assert (cell_root / "train" / "checkpoint-4").exists()  # capture reads it
    assert not (cell_root / "train" / "checkpoint-2").exists()
    with mock.patch.object(d.hub, "_upload", up):
        d._persist_marker_ft(cfg)  # done-file short-circuit
    assert up.call_count == 1


def test_persist_marker_ft_never_deletes_unuploaded(tmp_path):
    """Empty upload URL raises LOUD and NO rung is deleted (upload-before-
    delete invariant); --no-upload keeps all rungs."""
    cfg = _persist_fixture(tmp_path)
    cell_root = tmp_path / "m2_fullft_band8"
    up = mock.create_autospec(d.hub._upload, return_value="")
    with (
        mock.patch.object(d.hub, "_upload", up),
        pytest.raises(RuntimeError, match="returned no path"),
    ):
        d._persist_marker_ft(cfg)
    assert (cell_root / "train" / "checkpoint-2").exists()
    assert (cell_root / "train" / "checkpoint-4").exists()

    cfg_noup = d.Cfg(
        smoke=True,
        cells=("m2_fullft_band8",),
        out_root=tmp_path,
        upload=False,
        sentinel_dir=tmp_path / "logs",
    )
    rec = d._persist_marker_ft(cfg_noup)
    assert rec == {"skipped": "no-upload"}
    assert (cell_root / "train" / "checkpoint-2").exists()
