"""#1774 round-B CPU pins for scripts/issue1774_steering.py (the P3 module).

Covers: the condition grid (27 intervention conditions + steer_base at K=3), the
by-direction shard split (disjoint cover; a direction's +/- pair stays together;
steer_base pinned to shard 0), the degenerate-fluency heuristics, the ADD hook
(output shifted along v, tuple/bare handling, sign), the LEACE hook (fitted eraser
applied at the stream — matches ``LeaceEraser.transform``), intervention-before-
capture hook ordering (replace semantics), and the state-shift merge arithmetic
(baseline band + dt1 vs the baseline mean; pilot-dropped directions excluded from
the completeness check). All CPU, all tmp_path-only writes; no model/network.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

import issue1774_common as c  # noqa: E402
import issue1774_steering as st  # noqa: E402

DIRECTION_NAMES = (
    [f"top_sv{j}" for j in range(4)]
    + [f"kernel_tail{i}" for i in range(4)]
    + [f"random{i}" for i in range(4)]
    + [f"rb_{t}" for t in c.TRAITS]
)


def test_condition_grid_matches_plan() -> None:
    conds = st.build_conditions(sorted(DIRECTION_NAMES), smoke=False)
    interv = [x for x in conds if x.kind != "base"]
    assert len(interv) == 27  # 12 ADD x {+,-} + 3 LEACE (plan §4 P3)
    assert sum(1 for x in interv if x.kind == "add") == 24
    assert sum(1 for x in interv if x.kind == "leace") == 3
    base = [x for x in conds if x.kind == "base"]
    assert len(base) == 1 and base[0].k_draws == c.STEER_BASE_DRAWS
    assert all(x.k_draws == 1 for x in interv)
    signs = {(x.direction, x.sign) for x in interv if x.kind == "add"}
    assert len(signs) == 24  # every ADD direction carries both signs exactly once


def test_shard_split_disjoint_cover_and_direction_atomicity() -> None:
    conds = st.build_conditions(sorted(DIRECTION_NAMES), smoke=False)
    n = 2
    shards = [st.shard_conditions(conds, f"{i}/{n}") for i in range(n)]
    ids = [x.cond_id for s in shards for x in s]
    assert sorted(ids) == sorted(x.cond_id for x in conds)  # disjoint cover
    for s in shards:
        by_dir: dict[str, int] = {}
        for x in s:
            if x.kind == "add":
                by_dir[x.direction] = by_dir.get(x.direction, 0) + 1
        assert all(v == 2 for v in by_dir.values())  # +/- pair never split
    assert any(x.kind == "base" for x in shards[0])
    assert not any(x.kind == "base" for x in shards[1])


def test_degenerate_heuristics() -> None:
    assert st.is_degenerate("")  # near-empty
    assert st.is_degenerate("!" * 40)  # single-char run
    spam = "the same phrase " * 20
    assert st.is_degenerate(spam)  # dominant repeated trigram
    ok = (
        "The measured operators agree on the shared subspace, while the tail "
        "directions carry no held-out variance under the permutation null band."
    )
    assert not st.is_degenerate(ok)
    assert st.degenerate_fraction([ok, spam]) == 0.5


class _TupleBlock(torch.nn.Module):
    """Tiny decoder-block stand-in returning a tuple (the HF shape)."""

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, str]:
        return x * 1.0, "aux"


class _BareBlock(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * 1.0


def test_add_hook_shifts_output_along_v_both_shapes() -> None:
    torch.manual_seed(0)
    v = torch.zeros(8)
    v[3] = 1.0
    x = torch.randn(2, 5, 8)
    for block in (_TupleBlock(), _BareBlock()):
        base = block(x)
        base_hs = base[0] if isinstance(base, tuple) else base
        h = block.register_forward_hook(st.make_add_hook(v, 2.5))
        try:
            out = block(x)
        finally:
            h.remove()
        hs = out[0] if isinstance(out, tuple) else out
        delta = hs - base_hs
        assert torch.allclose(delta[..., 3], torch.full_like(delta[..., 3], 2.5))
        mask = torch.ones(8, dtype=torch.bool)
        mask[3] = False
        assert torch.allclose(delta[..., mask], torch.zeros_like(delta[..., mask]))
        # negative sign moves the other way
        h = block.register_forward_hook(st.make_add_hook(v, -2.5))
        try:
            out_neg = block(x)
        finally:
            h.remove()
        hs_neg = out_neg[0] if isinstance(out_neg, tuple) else out_neg
        assert torch.allclose(hs_neg[..., 3] - base_hs[..., 3], -delta[..., 3])


def test_leace_hook_matches_eraser_transform() -> None:
    from explore_persona_space.analysis.issue_763_nonlinear import fit_leace

    rng = np.random.default_rng(0)
    x = rng.normal(size=(200, 6))
    w = rng.normal(size=6)
    e0 = x @ w
    eraser = fit_leace(x, e0)
    hook = st.make_leace_hook(
        torch.from_numpy(eraser.mean_x).float(), torch.from_numpy(eraser.P).float()
    )
    block = _TupleBlock()
    h = block.register_forward_hook(hook)
    try:
        hs = torch.from_numpy(x[:16]).float().reshape(2, 8, 6)
        out = block(hs)[0]
    finally:
        h.remove()
    expected = eraser.transform(x[:16]).reshape(2, 8, 6)
    assert np.allclose(out.numpy(), expected, atol=1e-4)
    # eraser kills the concept covariance on the fit sample
    resid = eraser.transform(x)
    assert abs(np.cov(resid @ w, e0)[0, 1]) < 1e-6 * abs(np.cov(e0, e0)[0, 1] + 1)


def test_intervention_hook_registered_before_capture_hook_is_observed() -> None:
    # PyTorch replace semantics: a hook returning non-None replaces the output for
    # LATER-registered hooks — so a capture hook registered after the intervention
    # observes the shifted stream (assumption 11; ablation.py docstring L21-24).
    v = torch.zeros(4)
    v[0] = 1.0
    block = _BareBlock()
    seen: list[torch.Tensor] = []

    def capture(_m, _i, out):
        seen.append(out.detach().clone())
        return None

    h1 = block.register_forward_hook(st.make_add_hook(v, 3.0))
    h2 = block.register_forward_hook(capture)
    try:
        block(torch.zeros(1, 2, 4))
    finally:
        h1.remove()
        h2.remove()
    assert torch.allclose(seen[0][..., 0], torch.full_like(seen[0][..., 0], 3.0))


def _write_t1(paths: dict[str, Path], cond_id: str, k: int, arr: np.ndarray, mis) -> None:
    np.save(paths["summaries"] / f"t1_{cond_id}_draw{k}.npy", arr)
    idx_p = paths["summaries"] / f"row_index_{cond_id}.json"
    if not idx_p.exists():
        idx_p.write_text(json.dumps({"manifest_indices": list(mis)}))


def _write_gen(paths: dict[str, Path], cond_id: str, mis, k_draws: int) -> None:
    with (paths["gen"] / f"gen_{cond_id}.jsonl").open("w") as fh:
        for mi in mis:
            fh.write(
                json.dumps(
                    {
                        "manifest_index": mi,
                        "prefix_id": f"p{mi}",
                        "query_id": f"q{mi}",
                        "prefix_text": "pfx",
                        "prompt": "prompt",
                        "draws": [f"completion {cond_id} {mi} d{k}" for k in range(k_draws)],
                    }
                )
                + "\n"
            )


def _merge_fixture(paths: dict[str, Path], conds, mis, monkeypatch) -> None:
    """Gen files + manifest.json + a signature-conformant query-text fake for merge."""
    for cond in conds:
        _write_gen(paths, cond.cond_id, mis, cond.k_draws)
    (paths["eval"] / "manifest.json").write_text(
        json.dumps({"meta": {}, "conditions": [], "alpha0": 1.0})
    )

    def fake_query_texts(manifest_indices: list[int]) -> dict[int, str]:
        return {int(mi): f"question {mi}" for mi in manifest_indices}

    monkeypatch.setattr(st, "_query_texts_for", fake_query_texts)


def test_merge_state_shift_band_and_dt1(tmp_path: Path, monkeypatch) -> None:
    paths = {
        "gen": tmp_path / "gen",
        "summaries": tmp_path / "summaries",
        "eval": tmp_path / "eval",
    }
    for p in paths.values():
        p.mkdir(parents=True)
    n_ctx, n_layers, d = 2, len(c.LAYERS), 4
    li = c.LAYERS.index(c.HEADLINE_LAYER)
    mis = [11, 22]
    conds = [
        st.Condition("steer_base", "base", "", 0, 2),
        st.Condition("add_top_sv0_pos", "add", "top_sv0", 1, 1),
    ]
    _merge_fixture(paths, conds, mis, monkeypatch)
    base = np.zeros((n_ctx, n_layers, d), dtype=np.float16)
    base_k1 = base.copy()
    base_k1[:, li, 0] = 1.0  # draw distance 1.0 within every context
    _write_t1(paths, "steer_base", 0, base, mis)
    _write_t1(paths, "steer_base", 1, base_k1, mis)
    add = np.zeros((n_ctx, n_layers, d), dtype=np.float16)
    add[:, li, 0] = 3.5  # dt1 vs baseline mean (0.5 along dim 0) = 3.0
    _write_t1(paths, "add_top_sv0_pos", 0, add, mis)
    (paths["eval"] / "pilot_report_shard0of1.json").write_text(
        json.dumps({"alpha_by_direction": {"top_sv0": 1.0}, "dropped_directions": {}})
    )
    assert st.merge_state_shift(conds, paths, smoke=True)
    out = json.loads((paths["eval"] / "state_shift.json").read_text())
    assert out["steer_base_band"]["pooled_p50"] == 1.0
    cond = out["conditions"]["add_top_sv0_pos"]
    assert abs(cond["median_dt1"] - 3.0) < 1e-3
    assert set(cond["per_context_dt1"]) == {"11", "22"}
    assert out["n_usable_directions"] == 1 and out["judge_skip"] is False
    # P3->P5 interface: judge rows landed in manifest.json (2 base draws + 1 add) x 2 ctx
    manifest = json.loads((paths["eval"] / "manifest.json").read_text())
    rows = manifest["rows"]
    assert len(rows) == 2 * 2 + 1 * 2 and manifest["judge_skip"] is False
    assert {r["row_id"] for r in rows} == {
        "steer_base-11-d0",
        "steer_base-11-d1",
        "steer_base-22-d0",
        "steer_base-22-d1",
        "add_top_sv0_pos-11-d0",
        "add_top_sv0_pos-22-d0",
    }
    for r in rows:
        assert r["question"] == f"question {r['manifest_index']}"
        assert r["completion"].startswith(f"completion {r['condition']}")
        assert "__" not in r["row_id"] and len(r["row_id"]) <= st.MAX_ROW_ID_LEN


def test_merge_excludes_pilot_dropped_directions(tmp_path: Path, monkeypatch) -> None:
    paths = {
        "gen": tmp_path / "gen",
        "summaries": tmp_path / "summaries",
        "eval": tmp_path / "eval",
    }
    for p in paths.values():
        p.mkdir(parents=True)
    conds = [
        st.Condition("steer_base", "base", "", 0, 2),
        st.Condition("add_random0_pos", "add", "random0", 1, 1),  # dropped: no t1 files
    ]
    mis = [5]
    _merge_fixture(paths, [conds[0]], mis, monkeypatch)  # dropped cond has no gen file
    arr = np.zeros((1, len(c.LAYERS), 3), dtype=np.float16)
    _write_t1(paths, "steer_base", 0, arr, mis)
    _write_t1(paths, "steer_base", 1, arr, mis)
    (paths["eval"] / "pilot_report_shard0of1.json").write_text(
        json.dumps(
            {
                "alpha_by_direction": {},
                "dropped_directions": {"random0": "degenerate_fraction=0.60 after halving"},
            }
        )
    )
    # would deadlock on add_random0_pos's missing t1 without the dropped-direction filter
    assert st.merge_state_shift(conds, paths, smoke=False)
    out = json.loads((paths["eval"] / "state_shift.json").read_text())
    assert out["conditions"] == {}
    assert out["dropped_directions"] == {"random0": "degenerate_fraction=0.60 after halving"}
    assert out["judge_skip"] is True and out["calibration_failure"] is True
