"""#1776 follow-up p3p4 pins: tiny-real run e2e, real-body analyze, build/ladder/pick.

Real bodies throughout — fakes only at the GPU-scale boundary (a from-config
tiny Qwen2 stands in for the 7B; all fixture tensors are synthetic small-dim).
Fixture text is benign synthetic prose (never corpus rows).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import issue1776_p3p4 as P34  # noqa: E402

H = 16  # fixture hidden size


def _seeds_file(tmp: Path, n_pc: int = 4, n_mp: int = 2, n_g: int = 1, h: int = H) -> Path:
    g = torch.Generator().manual_seed(0)
    q, _ = torch.linalg.qr(torch.randn(h, n_pc + n_mp, generator=g))
    seeds = torch.cat([q.T.to(torch.float32), torch.randn(n_g, h, generator=g)], dim=0)
    seeds = seeds / seeds.norm(dim=1, keepdim=True)
    names = (
        [f"vpc{i}" for i in range(n_pc)]
        + [f"mprime_u{i}" for i in range(n_mp)]
        + [f"gauss{i}" for i in range(n_g)]
    )
    out = tmp / "seeds.pt"
    torch.save({"seeds": seeds, "names": names, "sigma_head": [1.0]}, out)
    return out


def _comparator_file(tmp: Path, h: int = H) -> Path:
    g = torch.Generator().manual_seed(1)
    payload = {
        "kind": "ridge",
        "W": torch.randn(h, h, generator=g) * 0.1,
        "xmu": torch.zeros(h),
        "xsd": torch.ones(h),
        "ymu": torch.zeros(h),
        "selected_lambda": 1.0,
    }
    out = tmp / "m_ridge_x50k.pt"
    torch.save(payload, out)
    return out


def _rb_dir(tmp: Path, h: int = H, scale: float = 12.0) -> Path:
    rb = tmp / "r_b"
    rb.mkdir(parents=True, exist_ok=True)
    g = torch.Generator().manual_seed(2)
    for t in P34.TRAITS:
        vec = torch.randn(1, h, generator=g)
        vec = vec / vec.norm() * scale
        torch.save({"r_b": vec, "layers": [P34.C76.SOURCE_LAYER]}, rb / f"{t}.pt")
    return rb


# ── _stratified_pick ──────────────────────────────────────────────────────────


def test_stratified_pick_counts_and_full_mask():
    rng = np.random.default_rng(0)
    err2 = np.linspace(0.0, 1.0, 40)
    idx, strat, full, edges = P34._stratified_pick(err2, 12, 4, 4, rng)
    assert idx.shape == (12,) and full.sum() == 4 and len(edges) == 3
    counts = np.bincount(strat, minlength=4)
    assert counts.tolist() == [3, 3, 3, 3]


def test_stratified_pick_single_stratum_and_zero_full():
    rng = np.random.default_rng(0)
    err2 = np.arange(10.0)
    idx, strat, full, edges = P34._stratified_pick(err2, 4, 0, 1, rng)
    assert idx.shape == (4,) and full.sum() == 0 and edges == []
    assert set(strat.tolist()) == {0}


# ── alpha ladder ──────────────────────────────────────────────────────────────


def test_alpha_ladder_matches_1415_operating_point(tmp_path):
    rb = _rb_dir(tmp_path, scale=12.0)
    ladder = P34.build_alpha_ladder(rb, tmp_path / "ladder.json", c14_med_norm=64.0)
    assert ladder["n_ref"] == pytest.approx(4.0 * 12.0, rel=1e-6)
    assert ladder["alphas"] == [pytest.approx(f * 48.0, rel=1e-4) for f in (0.25, 0.5, 1.0)]
    assert ladder["alphas"][-1] > P34.PARENT_ALPHA_MAX


def test_alpha_ladder_refuses_tiny_norms(tmp_path):
    rb = _rb_dir(tmp_path, scale=0.5)  # 4 x 0.5 = 2.0 <= parent's ceiling 4
    with pytest.raises(AssertionError, match="ladder rung"):
        P34.build_alpha_ladder(rb, tmp_path / "ladder.json", c14_med_norm=64.0)


# ── build (real loaders on production-shaped small fixtures) ──────────────────


def _fixture_dest(tmp: Path, n_lmsys: int = 6, n_wc: int = 5, h: int = H) -> Path:
    dest = tmp / "hf_dl"
    g = torch.Generator().manual_seed(3)
    jdir = dest / P34.C76.HF_PREFIX / "analysis_tensors/jpairs"
    jdir.mkdir(parents=True)
    ids = [f"jp{i:04d}" for i in range(n_lmsys)]
    rows = [
        {"pair_id": p, "prompt": f"What is {i} plus {i}?", "response": f"That makes {2 * i}."}
        for i, p in enumerate(ids)
    ]
    (jdir / "jpairs.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    torch.save(
        {
            "pair_id": ids,
            "c14": torch.randn(n_lmsys, h, generator=g),
            "v19": torch.randn(n_lmsys, h, generator=g),
            "c19": torch.randn(n_lmsys, h, generator=g),
            "layers": [14, 19],
        },
        jdir / "jpair_capture.pt",
    )
    wc = dest / P34.C76.HF_PREFIX / "wildchat_fresh"
    (wc / "final_token_capture").mkdir(parents=True)
    (wc / "raw_completions").mkdir(parents=True)
    torch.save(
        {
            "cx_last": torch.randn(n_wc, 2, h, generator=g),
            "v_x": torch.randn(n_wc, 2, h, generator=g),
            "ci": list(range(n_wc)),
            "layers": [P34.C76.SOURCE_LAYER, P34.C76.READOUT_LAYER],
            "shard_index": 0,
            "chunk": 0,
        },
        wc / "final_token_capture" / "shard00_chunk0000.pt",
    )
    (wc / "raw_completions" / "shard00_chunk0000.json").write_text(
        json.dumps(
            {
                "shard_index": 0,
                "chunk": 0,
                "rows": [
                    {"ci": i, "prompt": f"Name a color number {i}.", "response": "Blue."}
                    for i in range(n_wc)
                ],
            }
        )
    )
    _rb_dir(dest / "issue779_monitoring", h=h)  # -> dest/issue779_monitoring/r_b
    return dest


def test_build_real_body(tmp_path):
    dest = _fixture_dest(tmp_path)
    comp = _comparator_file(tmp_path)
    out_dir = tmp_path / "pcj"
    args = P34.argparse.Namespace(
        dest=dest,
        comparator=comp,
        out_dir=out_dir,
        n_sketch=4,
        n_full=2,
        strata=2,
        seed=0,
    )
    assert P34.cmd_build(args) == 0
    pairs = [json.loads(x) for x in (out_dir / "pcj_pairs.jsonl").read_text().splitlines()]
    assert len(pairs) == 4 and sum(p["full_rank"] for p in pairs) == 2
    assert {p["source"] for p in pairs} == {"lmsys", "wildchat"}
    tgt = torch.load(out_dir / "pcj_targets.pt", weights_only=True)
    assert tgt["c14"].shape == (4, H)
    ladder = json.loads((out_dir / "p4_alpha_ladder.json").read_text())
    assert len(ladder["alphas"]) == 3
    report = json.loads((out_dir / "pcj_build_report.json").read_text())
    assert set(report["strata"]) == {"lmsys", "wildchat"}


# ── run: tiny-real CPU e2e (real tokenizer + estimator + persist + resume) ────


@pytest.mark.slow
def test_run_tiny_cpu_e2e(tmp_path):
    pairs_path = tmp_path / "pcj_pairs.jsonl"
    rows = [
        {
            "pair_id": "jp0000",
            "prompt": "What is two plus two?",
            "response": "Four.",
            "source": "lmsys",
            "stratum": 0,
            "full_rank": False,
        },
        {
            "pair_id": "wc000001",
            "prompt": "Name a primary color.",
            "response": "Blue is a primary color.",
            "source": "wildchat",
            "stratum": 0,
            "full_rank": True,
        },
    ]
    pairs_path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    seeds_path = _seeds_file(tmp_path, n_pc=3, n_mp=2, n_g=1, h=64)  # tiny hidden = 64
    out_dir = tmp_path / "run"

    def _args(**kw):
        base = dict(
            model="Qwen/Qwen2.5-7B-Instruct",
            source_layer=1,
            readout_layer=3,
            dtype="float32",
            device="cpu",
            seed_chunk=4,
            serial_grads=False,
            tiny=True,
            pairs=pairs_path,
            seeds_file=seeds_path,
            out_dir=out_dir,
            shard_index=0,
            num_shards=1,
            limit=0,
        )
        base.update(kw)
        return P34.argparse.Namespace(**base)

    assert P34.cmd_run(_args()) == 0
    pcj = sorted((out_dir / "pcj").glob("*.pt"))
    assert len(pcj) == 2
    d = torch.load(out_dir / "pcj" / "wc000001.pt", weights_only=True)
    assert d["seed_mode"] == "full" and d["rows"]["last"].shape == (64, 64)
    assert d["rows"]["last"].dtype == torch.bfloat16 and d["ctx_maxabs"] > 0
    d2 = torch.load(out_dir / "pcj" / "jp0000.pt", weights_only=True)
    assert d2["seed_mode"] == "sketch" and d2["rows"]["last"].shape == (6, 64)
    # resume: second invocation skips both units (files untouched)
    mt = {p.name: p.stat().st_mtime for p in pcj}
    assert P34.cmd_run(_args()) == 0
    assert all(p.stat().st_mtime == mt[p.name] for p in (out_dir / "pcj").glob("*.pt"))
    # regime refusal: a changed output-affecting key must refuse the out-dir
    with pytest.raises(RuntimeError, match="manifest MISMATCH"):
        P34.cmd_run(_args(dtype="bfloat16"))
    # G-NONZERO degenerate probe: an all-zero seed matrix yields a zero
    # context-gradient field -> the designed rc=8 halt + gate report fires
    zero_seeds = tmp_path / "zero_seeds.pt"
    torch.save(
        {"seeds": torch.zeros(3, 64), "names": ["z0", "z1", "z2"], "sigma_head": []},
        zero_seeds,
    )
    out2 = tmp_path / "run_zero"
    assert P34.cmd_run(_args(seeds_file=zero_seeds, out_dir=out2)) == 8
    gate = json.loads((out2 / "gate_gnonzero_shard0.json").read_text())
    assert gate["pass"] is False and gate["gate"] == "G-NONZERO"


@pytest.mark.slow
def test_pilot_tiny_cpu_gate(tmp_path):
    pairs_path = tmp_path / "pcj_pairs.jsonl"
    rows = [
        {
            "pair_id": "jp0000",
            "prompt": "What is one plus one?",
            "response": "Two.",
            "source": "lmsys",
            "stratum": 0,
            "full_rank": False,
        }
    ]
    pairs_path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    seeds_path = _seeds_file(tmp_path, n_pc=3, n_mp=2, n_g=1, h=64)

    def _args(budget):
        return P34.argparse.Namespace(
            model="Qwen/Qwen2.5-7B-Instruct",
            source_layer=1,
            readout_layer=3,
            dtype="float32",
            device="cpu",
            seed_chunk=4,
            serial_grads=False,
            tiny=True,
            pairs=pairs_path,
            seeds_file=seeds_path,
            budget_gpu_h=budget,
            ngpu=8,
            out=tmp_path / "pilot.json",
        )

    assert P34.cmd_pilot(_args(budget=10.0)) == 0
    rep = json.loads((tmp_path / "pilot.json").read_text())
    assert rep["verdict"] == "OK" and rep["rows_total"] == 6
    # designed halt: an absurdly small budget trips the 2x gate -> rc=7
    assert P34.cmd_pilot(_args(budget=1e-9)) == 7
    rep = json.loads((tmp_path / "pilot.json").read_text())
    assert rep["verdict"] == "OVER_2X"


# ── analyze: real body on synthetic small-dim world with numeric ground truth ─


def test_analyze_real_body_ground_truth(tmp_path):
    h = H
    g = torch.Generator().manual_seed(4)
    seeds_path = _seeds_file(tmp_path, n_pc=4, n_mp=2, n_g=1, h=h)
    sd = torch.load(seeds_path, weights_only=True)
    comp = _comparator_file(tmp_path)
    rb = _rb_dir(tmp_path)
    javg_dir = tmp_path / "jac_full"
    javg_dir.mkdir()
    javg = torch.randn(h, h, generator=g)
    for arm in ("prefix", "ctx", "last"):
        torch.save({"J": javg.clone()}, javg_dir / f"J_{arm}.pt")

    pcj_root = tmp_path / "run"
    (pcj_root / "pcj").mkdir(parents=True)
    ids, srcs, c14s, v19s = [], [], [], []
    n_ctx = 6
    for i in range(n_ctx):
        pid = f"c{i:02d}"
        src = "lmsys" if i < 3 else "wildchat"
        full = i in (0, 3, 4)  # 3 full-rank -> exercises the spearman branch
        c = torch.randn(h, generator=g)
        v = c @ javg.T  # v = J c => J_i == javg predicts deltas EXACTLY
        rows_full = javg.clone()
        payload = {
            "pair_id": pid,
            "source": src,
            "stratum": i % 2,
            "seed_mode": "full" if full else "sketch",
            "seeds_sha": "x",
            "layers": [14, 19],
            "rows": {
                a: (
                    rows_full if full else sd["seeds"].to(torch.float64) @ javg.to(torch.float64)
                ).to(torch.bfloat16)
                for a in ("prefix", "ctx", "last")
            },
            "v": v.to(torch.float32),
            "c_last": c.to(torch.float32),
            "c_prefix": c.to(torch.float32),
            "c_ctx": c.to(torch.float32),
            "ctx_maxabs": 1.0,
            "unit_wall_s": 0.1,
        }
        torch.save(payload, pcj_root / "pcj" / f"{pid}.pt")
        ids.append(pid)
        srcs.append(src)
        c14s.append(c)
        v19s.append(v)
    torch.save(
        {
            "pair_id": ids,
            "source": srcs,
            "stratum": [0] * n_ctx,
            "err2": [0.0] * n_ctx,
            "c14": torch.stack(c14s),
            "v19": torch.stack(v19s),
            "comparator_tag": "m_ridge_x50k",
        },
        tmp_path / "pcj_targets.pt",
    )
    out = tmp_path / "jacobian_heterogeneity.json"
    args = P34.argparse.Namespace(
        pcj_dir=pcj_root,
        targets=tmp_path / "pcj_targets.pt",
        seeds_file=seeds_path,
        javg_dir=javg_dir,
        comparator=comp,
        rb_dir=rb,
        n_boot=25,
        out=out,
        fig_dir=tmp_path / "figs",
    )
    assert P34.cmd_analyze(args) == 0
    res = json.loads(out.read_text())
    assert res["n_contexts"] == n_ctx and res["n_full_rank"] == 3
    # every J_i == javg (up to bf16): cos-to-avg ~1, no cancellation, pairwise ~1
    assert res["arms"]["last"]["cos_to_J_avg_median"] > 0.99
    assert res["arms"]["last"]["cancellation_ratio_sample"] > 0.99
    assert res["arms"]["last"]["pairwise_cos_median"] > 0.99
    # v = J c exactly: J_avg and J_i_own neighbor-delta R^2 ~ 1 in the PC subspace
    for corpus in ("lmsys", "wildchat"):
        r2 = res["neighbor_delta_r2"][corpus]["r2"]
        assert r2["J_avg"] > 0.98 and r2["J_i_own"] > 0.98
        assert r2["zero"] == 0.0
        assert res["neighbor_delta_r2"][corpus]["subspace_energy_fraction"] <= 1.0
    assert res["neighbor_delta_r2"]["full_space_pooled"]["r2"]["J_i_own"] > 0.98
    assert "steering_direction_agreement" in res
    for name in ("evil", "w1_mprime", "random"):
        assert res["steering_direction_agreement"][name]["cos_Ji_vs_Javg_median"] > 0.99
    assert res["sketch_restriction_validation"]["n"] == 3
    assert (tmp_path / "figs" / "jacobian_heterogeneity.png").is_file()
    # deliverable-file kind guard: a directory --out is refused
    args.out = tmp_path
    with pytest.raises(AssertionError, match="deliverable JSON FILE"):
        P34.cmd_analyze(args)
