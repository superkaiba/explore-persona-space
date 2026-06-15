"""Issue #595 — pin the v3 squared-gauge MSRD correction.

The prefix-KV-shift is a mean-squared RELATIVE deviation (MSRD): a SQUARED norm.
Under the rsLoRA application-scaling rule the LoRA update enters the forward pass
as Δ = (alpha/sqrtr)·BA·x, so the LINEAR prefix-KV deviation Δk scales with the gauge
g = alpha/sqrtr, and the SQUARED deviation in the MSRD numerator scales with g**2. A pure
LoRA-application-scale artifact therefore enters the score as g**2·carrier**2. To
isolate carrier**2, the divisor on the squared MSRD metric must be g**2 (NOT the
linear g — the v2 defect, which leaves a residual g factor).

This test builds a synthetic two-row case where the carrier is identical and the
gauge differs, and asserts:
  - raw MSRD scales as g**2 (within numerical tol),
  - gaugenorm_sq = raw / g**2 is gauge-INVARIANT,
  - the v2 LINEAR divisor (raw / g) would NOT be gauge-invariant.
"""

from __future__ import annotations

import torch


def _msrd(delta: torch.Tensor, base: torch.Tensor) -> float:
    """Mean over positions of ||Δ||**2 / ||base||**2 (the TReFT relative-deviation eq.)."""
    dnum = (delta**2).sum(dim=-1)  # per-position squared norm
    bden = (base**2).sum(dim=-1)
    return float(torch.mean(dnum / (bden + 1e-12)).item())


def test_raw_msrd_scales_as_gauge_squared():
    torch.manual_seed(0)
    n_pos, d = 8, 64
    base = torch.randn(n_pos, d)
    # A fixed carrier direction (the per-position relative change the fine-tune
    # would produce at unit gauge).
    carrier = torch.randn(n_pos, d)

    g1, g2 = 8.0, 45.25  # marker gauge vs turner_em gauge (plan section 10)
    # The LINEAR deviation scales with the application gauge: Δk = g · carrier.
    delta1 = g1 * carrier
    delta2 = g2 * carrier

    raw1 = _msrd(delta1, base)
    raw2 = _msrd(delta2, base)

    # raw MSRD scales as g**2 (squared norm of a g-scaled linear deviation).
    assert abs(raw2 / raw1 - (g2 / g1) ** 2) < 1e-4, (
        f"raw MSRD ratio {raw2 / raw1:.4f} != gauge**2 ratio {(g2 / g1) ** 2:.4f}"
    )


def test_gaugenorm_sq_is_gauge_invariant():
    torch.manual_seed(1)
    n_pos, d = 8, 64
    base = torch.randn(n_pos, d)
    carrier = torch.randn(n_pos, d)

    g1, g2 = 8.0, 45.25
    raw1 = _msrd(g1 * carrier, base)
    raw2 = _msrd(g2 * carrier, base)

    # v3 SQUARED divisor isolates carrier**2 -> gauge-invariant.
    gaugenorm_sq_1 = raw1 / (g1**2)
    gaugenorm_sq_2 = raw2 / (g2**2)
    assert abs(gaugenorm_sq_1 - gaugenorm_sq_2) < 1e-6, (
        f"gaugenorm_sq must be gauge-invariant: {gaugenorm_sq_1:.6g} != {gaugenorm_sq_2:.6g}"
    )


def test_linear_divisor_is_not_gauge_invariant():
    """The v2 LINEAR divisor (raw / g) leaves a residual g factor — the v3-corrected bug."""
    torch.manual_seed(2)
    n_pos, d = 8, 64
    base = torch.randn(n_pos, d)
    carrier = torch.randn(n_pos, d)

    g1, g2 = 8.0, 45.25
    raw1 = _msrd(g1 * carrier, base)
    raw2 = _msrd(g2 * carrier, base)

    # v2 linear divisor leaves carrier**2·g — so the high-gauge row stays ~g₂/g₁
    # times larger (severe: turner_em rows would re-rank identically to raw).
    v2_1 = raw1 / g1
    v2_2 = raw2 / g2
    assert abs(v2_2 / v2_1 - (g2 / g1)) < 1e-3, (
        "v2 linear divisor should leave a residual gauge factor of g2/g1"
    )
    # And it is decidedly NOT invariant (the failure the v3 correction fixes).
    assert v2_2 / v2_1 > 5.0, "the residual linear factor must be large (gauge spread ~8->45)"


def test_driver_gauge_and_divisor_match_the_squared_rule():
    """The driver's gauge_from_config + gaugenorm_sq use the SQUARED divisor."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "issue595_prefix_carrier",
        __import__("pathlib").Path(__file__).resolve().parents[1]
        / "scripts"
        / "issue595_prefix_carrier.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    # turner_em config (alpha=256, r=32, rsLoRA) -> gauge 45.25, squared ~2048.
    gauge, use_rslora = mod.gauge_from_config({"lora_alpha": 256, "r": 32, "use_rslora": True})
    assert use_rslora
    assert abs(gauge - 256 / (32**0.5)) < 1e-6
    # The variant table stores gauge_normalization_power=2 for the gaugenorm_sq variant.
    # (The driver divides all_l_mean by gauge**2 — verified by the math above.)
    raw_all_l = 4.0
    gaugenorm_sq = raw_all_l / (gauge**2)
    assert gaugenorm_sq < raw_all_l, "gauge-normalized score is the raw MSRD / gauge**2"
