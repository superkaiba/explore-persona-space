"""#1776 Phase 1: per-direction directional diagnostic (plan v4 §4 Phase 1).

NOT a kill switch — interpretation context only, recorded and carried forward
regardless of outcome (verbatim user directive in the plan). Takes the top-K
right/left singular triples (w_i, sigma_i, u_i) of the slot-matched cross-layer
comparator M' (Phase 0.5 ``m_ridge_x50k`` weights payload; the effective raw-x
operator is A = (W / xsd[:, None])^T, mapping layer-14 column vectors to v(19)
space — w_i lives in the layer-14 differentiation slot, u_i in the v(19)
cotangent space), uses the u_i as backward seeds over ~1k pairs through the
Phase-2 estimator (``issue1776_jacobian``), and writes the per-direction table:

  claimed gain sigma_i  vs  measured causal gain ||E[g(u_i, .)]||  vs
  alignment cos(E[g(u_i, .)], w_i)

per source arm (``last`` is the unit-matched headline column; prefix/ctx ride
along from the same backwards). The shipped M's singular directions are NOT
used as seeds (its right space is the causally-degenerate layer-19 slot).

Content hygiene: never prints pair text; logs pair ids + scalars only.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import issue1776_common as C76
from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()  # bind shared-VM thread caps BEFORE numpy/torch import (#847 gate)
import issue1776_jacobian as JAC

import torch  # noqa: E402

import issue779_common as C  # noqa: E402


def comparator_svd(comparator_path: Path, topk: int) -> dict:
    """Top-K singular triples of the effective raw-x operator A = (W/xsd)^T.

    Returns {"u": (K, H_out) L19 cotangent seeds, "w": (K, H_in) L14 input
    directions, "sigma": (K,)} — fp64 SVD once on CPU, cast fp32.
    """
    comp = torch.load(comparator_path, map_location="cpu", weights_only=True)
    a_op = (comp["W"].to(torch.float64) / comp["xsd"].to(torch.float64)[:, None]).T
    u, s, vh = torch.linalg.svd(a_op, full_matrices=False)
    k = min(topk, s.shape[0])
    return {
        "u": u[:, :k].T.to(torch.float32),
        "w": vh[:k, :].to(torch.float32),
        "sigma": s[:k].to(torch.float32),
    }


def per_direction_table(out_dir: Path, triples: dict) -> dict:
    """Assemble the diagnostic table from the estimator's J_* outputs."""
    sigma, w = triples["sigma"], triples["w"].to(torch.float64)
    table = []
    fins = {
        arm: torch.load(out_dir / f"J_{arm}.pt", map_location="cpu", weights_only=True)
        for arm in JAC.ARMS
    }
    for i in range(sigma.shape[0]):
        row = {"i": i, "sigma_claimed": float(sigma[i])}
        for arm in JAC.ARMS:
            g = fins[arm]["J"][i].to(torch.float64)
            denom = float(g.norm() * w[i].norm())
            row[f"gain_{arm}"] = float(g.norm())
            row[f"cos_w_{arm}"] = float((g @ w[i]) / denom) if denom > 0 else 0.0
        table.append(row)
    return {"table": table, "headline_arm": "last", "repro": C76.repro_meta()}


def run(args) -> int:
    triples = comparator_svd(args.comparator, args.topk)
    seeds = triples["u"] / triples["u"].norm(dim=1, keepdim=True).clamp(min=1e-12)
    args.seeds_file = args.out_dir / "phase1_seeds.pt"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"seeds": seeds, "names": [f"mprime_u{i}" for i in range(seeds.shape[0])]},
        args.seeds_file,
    )
    args.mode, args.m = "sketch", 0
    args.shard_index, args.num_shards = 0, 1
    t0 = time.time()
    rc = JAC.run(args)
    if rc != 0:
        return rc  # G-NONZERO HALT propagates (engineering gate, not a science gate)
    report = per_direction_table(args.out_dir, triples)
    report["n_pairs"] = args.limit_pairs or None
    report["elapsed_s"] = time.time() - t0
    C76.atomic_write_json(args.out_dir / "directional_table.json", report)
    top = report["table"][0]
    print(
        f"[phase1] [phase=phase1_done] K={len(report['table'])} "
        f"top: sigma={top['sigma_claimed']:.4f} gain_last={top['gain_last']:.4e} "
        f"cos_w_last={top['cos_w_last']:.3f} -> {args.out_dir}/directional_table.json",
        flush=True,
    )
    return 0


def smoke(args) -> int:
    """Tiny-real CPU smoke: synthetic comparator payload + 2 benign pairs e2e."""
    args.tiny = True
    args.source_layer, args.readout_layer = 1, 3
    model, tok = JAC.load_model(args)
    hidden = model.model.config.hidden_size
    gen = torch.Generator().manual_seed(2)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.comparator = args.out_dir / "smoke_comparator.pt"
    torch.save(
        {
            "kind": "ridge",
            "W": torch.randn(hidden, hidden, generator=gen) / hidden**0.5,
            "xsd": torch.rand(hidden, generator=gen) + 0.5,
            "xmu": torch.zeros(hidden),
            "ymu": torch.zeros(hidden),
        },
        args.comparator,
    )
    args.pairs = args.out_dir / "smoke_pairs.jsonl"
    prompts = ["What is the capital of France?", "Name one prime number."]
    responses = ["The capital of France is Paris.", "Two is a prime number."]
    args.pairs.write_text(
        "\n".join(
            json.dumps({"pair_id": f"p{i}", "prompt": p, "response": r})
            for i, (p, r) in enumerate(zip(prompts, responses, strict=True))
        )
        + "\n"
    )
    args.topk, args.limit_pairs = 4, 2
    rc = run(args)
    assert rc == 0, rc
    rep = json.loads((args.out_dir / "directional_table.json").read_text())
    assert len(rep["table"]) == 4, rep
    for row in rep["table"]:
        assert row["gain_last"] > 0.0, row  # nonzero causal gain on the tiny model
        assert -1.0 <= row["cos_w_last"] <= 1.0, row
    print("[phase1] [phase=smoke_done] PASS", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--comparator", type=Path, help="Phase-0.5 m_ridge_x50k weights .pt")
    ap.add_argument("--pairs", type=Path, help="pair manifest JSONL (issue1776_jacobian format)")
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--limit-pairs", type=int, default=1024)
    ap.add_argument("--model", default=C.DEFAULT_MODEL)
    ap.add_argument("--source-layer", type=int, default=C76.SOURCE_LAYER)
    ap.add_argument("--readout-layer", type=int, default=C76.READOUT_LAYER)
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed-chunk", type=int, default=32)
    ap.add_argument("--serial-grads", action="store_true")
    ap.add_argument("--ckpt-every", type=int, default=16)
    ap.add_argument("--tiny", action="store_true")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--smoke", action="store_true", help="tiny-real CPU e2e smoke")
    args = ap.parse_args(argv)
    if args.smoke:
        return smoke(args)
    assert args.comparator and args.pairs, "--comparator and --pairs required"
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
