#!/usr/bin/env python3
# ruff: noqa: RUF002, RUF003
# (math/scientific notation — Δv, ŵ, g_real — intentional in docstrings)
"""Issue #683 — synthetic Δv-bank generator for CPU math smokes (NOT a GPU phase).

Builds a controlled, known-answer Δv bank in the SAME schema the Phase-A
extractors emit (``{source, per_context:{ctx:{v_base,v_trained,Delta_v}},
w_hat, g_real, ...}``), so the off-pod CPU phases (A7 precondition + key×metric
scoring) can be smoke-exercised on their REAL linear algebra with a verifiable
ground truth — the ``--no-adapter`` GPU-phase smoke produces Δv≡0 banks that
test only the IO plumbing, not the SVD / ridge / Spearman math.

Two modes:
  ``--mode rank1``  — Δv(C') = g_real(C')·ŵ + small noise (the A7-holds case):
                      A7 should verdict ``rank1_scalar_gate`` and a key
                      proportional to ŵ should score high.
  ``--mode lowrank``— Δv(C') = a·g1(C')·ŵ + b·g2(C')·u2 (genuinely 2-D):
                      A7 should verdict ``low_rank_fallback_required``.

The realized gate g_real(C') is computed from the synthetic Δv against ŵ
EXACTLY as the real banks do, so the scorer reads it identically. A
``c_C`` context-vector bank (per context) is also emitted so the scorer's
key=c_C arm has real inputs.

CLI:
    uv run python scripts/issue683_make_synthetic_bank.py --mode rank1 \
        --out-dir eval_results/issue_683/smoke/synthetic/marker --behavior marker
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from _bootstrap import PROJECT_ROOT, bootstrap

logger = bootstrap(log_name="issue683_make_synthetic_bank")

sys.path.insert(0, str(PROJECT_ROOT / "src"))

from explore_persona_space.experiments.issue_683 import realized_gate, repro_metadata  # noqa: E402


def build_bank(
    *, mode: str, behavior: str, source: str, n_targets: int, hidden: int, seed: int, noise: float
) -> tuple[dict, dict]:
    """Return (dv_bank_payload, c_bank) with a known A7 structure.

    ``dv_bank_payload`` matches the Phase-A schema; ``c_bank`` is {ctx: c_C}.
    """
    import torch

    g = torch.Generator().manual_seed(seed)
    w_hat = torch.randn(hidden, generator=g)
    w_hat = w_hat / w_hat.norm()
    # a second orthogonal direction for the low-rank mode.
    u2 = torch.randn(hidden, generator=g)
    u2 = u2 - (u2 @ w_hat) * w_hat
    u2 = u2 / u2.norm()

    contexts = [source, *[f"C{i}" for i in range(n_targets)]]
    per_context: dict[str, dict] = {}
    c_bank: dict[str, torch.Tensor] = {}
    for i, ctx in enumerate(contexts):
        # ground-truth gate per context; source gate = 1 by construction.
        g_true = 1.0 if ctx == source else float(0.2 + 0.7 * (i / max(len(contexts) - 1, 1)))
        if mode == "rank1":
            dv = g_true * w_hat
        elif mode == "lowrank":
            # Genuinely 2-D: the second component's per-context gate g2 is drawn
            # INDEPENDENTLY (so it carries variance the w_hat direction cannot
            # explain) and scaled so σ₂ ≈ σ₁ — the stacked SVD then splits its
            # energy and A7 verdicts low_rank_fallback_required (σ₁²/Σσ² < 0.5).
            g2 = float(torch.randn(1, generator=g).item())
            dv = g_true * w_hat + g2 * u2
        else:
            raise ValueError(f"unknown mode {mode!r}")
        dv = dv + noise * torch.randn(hidden, generator=g)
        v_base = torch.randn(hidden, generator=g)
        per_context[ctx] = {
            "v_base": v_base.float(),
            "v_trained": (v_base + dv).float(),
            "Delta_v": dv.float(),
        }
        # a c_C correlated with g_true so the context-only key is informative
        # in the rank-1 case (lets the scorer smoke see a non-trivial ρ).
        c_bank[ctx] = (g_true * w_hat + 0.3 * torch.randn(hidden, generator=g)).float()

    w_hat_real = per_context[source]["Delta_v"]
    g_real = {ctx: realized_gate(w_hat_real, per_context[ctx]["Delta_v"]) for ctx in contexts}
    payload = {
        "source": source,
        "seed": seed,
        "behavior": behavior,
        "layer": 0,
        "read_location": "synthetic",
        "panel": contexts,
        "w_hat": w_hat_real,
        "w_hat_norm": float(w_hat_real.norm()),
        "g_real_source_self": g_real[source],
        "self_consistent": abs(g_real[source] - 1.0) <= 1e-6,
        "gauge": {"synthetic": True, "mode": mode},
        "adapter_dir": None,
        "per_context": per_context,
        "g_real": g_real,
    }
    return payload, c_bank


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--mode", choices=("rank1", "lowrank"), default="rank1")
    ap.add_argument("--behavior", default="marker")
    ap.add_argument("--source", default="A1")
    ap.add_argument("--n-targets", type=int, default=12)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--noise", type=float, default=0.02)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args(argv)

    import torch

    out_dir = Path(args.out_dir)
    dv_dir = out_dir / "dv"
    dv_dir.mkdir(parents=True, exist_ok=True)
    payload, c_bank = build_bank(
        mode=args.mode,
        behavior=args.behavior,
        source=args.source,
        n_targets=args.n_targets,
        hidden=args.hidden,
        seed=args.seed,
        noise=args.noise,
    )
    dv_path = dv_dir / f"{args.source}_L0.pt"
    torch.save(payload, dv_path)
    c_path = out_dir / "c_bank.pt"
    torch.save({"contexts": c_bank, "meta": {"synthetic": True, "behavior": args.behavior}}, c_path)

    # Synthetic t_{C,B} in the t_cb extractor's schema, so the scorer's k_tCB /
    # k_cC+δ / ridge-ψ arms have real inputs in the math smoke. Correlated with
    # the source's c_C (an answer-side proxy of the source context).
    g = torch.Generator().manual_seed(args.seed + 7)
    t_cb = (c_bank[args.source] + 0.2 * torch.randn(args.hidden, generator=g)).float()
    tcb_dir = out_dir / "t_cb"
    tcb_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "t_cb": t_cb,
            "behavior": args.behavior,
            "source": args.source,
            "layer": 0,
            "read_location": "synthetic",
            "n_rows": 0,
        },
        tcb_dir / f"t_cb_{args.behavior}_{args.source}_L0.pt",
    )
    (out_dir / "synthetic_manifest.json").write_text(
        json.dumps(
            {
                "mode": args.mode,
                "behavior": args.behavior,
                "source": args.source,
                "n_targets": args.n_targets,
                "hidden": args.hidden,
                "dv_path": str(dv_path),
                "c_bank_path": str(c_path),
                "reproducibility": repro_metadata({"mode": args.mode}),
            },
            indent=2,
        )
    )
    logger.info(
        "[phase=synthetic_bank] mode=%s -> dv=%s c_bank=%s (%d contexts)",
        args.mode,
        dv_path,
        c_path,
        len(payload["per_context"]),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
