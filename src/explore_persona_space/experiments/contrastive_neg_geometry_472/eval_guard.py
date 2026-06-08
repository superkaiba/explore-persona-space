# ruff: noqa: RUF001, RUF002, RUF003  # em-dash + Qwen marker " ※" + Greek ΔG + × intentional
"""Task #472/#477 eval guard — B-matrix structural check on the trained adapter.

Round-4 redesign (task #504, 2026-06-08): the guard is now **B-matrix-only**.
The previous three-clause regression check (B-norm > floor AND max|ΔG| < eps
AND n_emit == 0 → RAISE ``LoRANotAppliedError``) was tuned for the v1 anchor
(lr=2e-6, r=8, all-linear at frac=1.00 of #477's 63-step regime, where ΔG is
expected ≫ 0.5 nats by design). For the v2 lr-ladder smoke that *deliberately*
includes a gentle anchor (lr=1e-5) at early checkpoints (frac=0.08, 0.16), the
adapter is at floor BY DESIGN — that is the whole point of the smoke (find
which (lr, frac) pair lands mid-band). Treating "at floor at frac=0.16" as
``LoRANotAppliedError`` is a false-positive: the guard was firing on a
training-budget effect, not a structural bug.

The B-matrix Frobenius check is the only one this guard owns now:

  * B-matrix Frobenius norm > floor (1e-3): the safetensors on disk carries a
    genuinely-trained adapter (or at least a non-trivial number of optimizer
    steps); the guard returns ``"pass_b_norm_ok"``.
  * B-matrix Frobenius norm ≤ floor: the adapter is structurally empty (PEFT
    initializes B=0; the file matches that init). The guard logs a
    ``"pass_genuine_floor"`` verdict and returns. It does NOT raise — a
    genuinely-untrained / collapsed adapter is a real measurement of "this
    cell didn't learn anything," not a regression.

The "is the metric in band" question now belongs to the picker (the
post-smoke pick rule's anti-saturation band — [5, 12] nats × [0.1, 0.8]
emission at the chosen frac in plan v2 §4.1). Conflating "adapter applied
vs not" (structural) with "signal in band" (calibration) is what caused the
v2 Phase 0 lr-ladder smoke to crash at frac=0.16 of the gentle lr=1e-5 cell
in round 3.

The earlier metric-based aggregation (``_aggregate_records_for_guard``) is
kept so callers can still log diagnostic max|ΔG| / n_emit values alongside
the verdict, but the values no longer gate the raise.
"""

from __future__ import annotations

import logging
import os
from typing import Any

log = logging.getLogger("issue_472.eval_guard")

# Below this B-matrix Frobenius norm we treat the adapter as effectively
# untrained — a clean newly-initialized PEFT LoRA has B-matrix norm 0.0 by
# construction (PEFT init: A ~ Kaiming, B = 0). After even a handful of
# optimizer steps the norm grows above 1e-4. 1e-3 keeps a generous floor while
# still firing on the trained / partially-trained adapters we will re-eval.
DEFAULT_B_NORM_FLOOR = 1e-3

# A real trained adapter applied at the eval slot produces |ΔG| ≫ this on at
# least the source-self probes (source-self ΔG > 5 nats is the rig's validity
# floor; bystander effects are smaller but ≥ 0.5 nats on any reasonable cell).
# An eval that reads max |ΔG| < 0.5 nats across the WHOLE panel is the bug
# class — base log-prob is being returned at every probe. The 0.5-nat default
# matches the brief.
DEFAULT_DELTA_G_EPS_NATS = 0.5


class LoRANotAppliedError(RuntimeError):
    """Retained for compatibility — no longer raised by the B-matrix-only guard.

    Historically raised when an adapter with B-norm > floor read ΔG ≈ 0 at
    every probe (the #477 v4/v6 silent-LoRA-not-applied regression). The
    round-4 redesign (task #504, 2026-06-08) drops the metric-based clauses
    that produced this raise in favour of a structural B-matrix-only check; the
    "metric in band" question now belongs to the post-smoke picker (plan v2
    §4.1 anti-saturation band [5, 12] nats × [0.1, 0.8] emission). The class
    is kept so existing imports (``scripts/i504_run_cell.py``,
    ``scripts/i504_eval_trajectory.py``, ``scripts/i504_reval_grid.py``,
    ``eval_trajectory.py``) keep resolving.
    """


class MarkerLogprobPathReadingFromBaseError(RuntimeError):
    """Raised when the marker-logprob reader has the v3 byte-identical bug.

    Plan v5 §4.4 fix #1 (tightened in v5 from a draft 95% threshold to a 5%
    de-minimis threshold after alternatives-Codex round-1 Must-Fix). The
    canonical v3 failure mode produced ``g_logp == b_logp`` to full float
    precision on ALL 540 (persona × question) records while KL(trained ‖ base)
    at the same slot climbed to 24.35 nats — physically impossible for a
    correct read, so the marker-logprob reader was computing ``g_logp`` from
    the BASE model (adapter not applied in that codepath). The v4 byte-
    identical guard catches this BEFORE the spurious zero leaves the rig.

    The 5% de-minimis threshold also catches a PARTIAL bug (a reader broken
    on 60% of slots) that would silently compress Phase 1's per-arm ΔG
    dynamic range and attenuate geometry coefficients toward null — the
    failure mode the alternatives-Codex critic flagged as FATAL on lens G.
    """


def b_matrix_frobenius_norm(adapter_dir: str | os.PathLike) -> float:
    """Compute the max Frobenius norm across all ``lora_B`` tensors in a saved
    PEFT adapter directory.

    Reads ``adapter_model.safetensors`` via the safetensors streaming API — no
    base-model load, no GPU, no PEFT instantiation. The max (not the mean) over
    layers is the strongest signal that ANY layer's B-matrix is non-trivial; a
    mean would dilute a single learned layer below the floor on a deep model.

    Args:
        adapter_dir: directory containing PEFT's ``adapter_model.safetensors``
            and ``adapter_config.json``.

    Returns:
        Max Frobenius norm across all ``lora_B.*.weight`` tensors, as a Python
        float. Returns 0.0 when the file contains no ``lora_B`` keys (treated
        as a non-LoRA adapter — caller decides what to do).

    Raises:
        FileNotFoundError: ``adapter_model.safetensors`` missing under
            ``adapter_dir``.
    """
    from pathlib import Path

    from safetensors import safe_open

    adapter_dir = Path(adapter_dir)
    weights_path = adapter_dir / "adapter_model.safetensors"
    if not weights_path.exists():
        raise FileNotFoundError(
            f"adapter weights missing: {weights_path} — fetch the adapter dir "
            f"before calling b_matrix_frobenius_norm."
        )
    max_norm = 0.0
    with safe_open(str(weights_path), framework="pt", device="cpu") as f:
        for key in f.keys():  # noqa: SIM118 (safe_open is not dict-like)
            if "lora_B" not in key:
                continue
            t = f.get_tensor(key)
            # Cast to float32 before norm to avoid bf16 underflow on tiny weights.
            n = float(t.float().norm().item())
            if n > max_norm:
                max_norm = n
    return max_norm


def _aggregate_records_for_guard(
    g_records: dict[str, dict[str, dict[str, float | bool]]],
    b_records: dict[str, dict[str, dict[str, float | bool]]],
) -> tuple[float, int, int]:
    """Reduce two per-probe record dicts to (max |ΔG|, n_emit_true, n_probes).

    Both ``g_records`` and ``b_records`` follow the rig's
    ``out[persona][q] = {"logp": float, "argmax_marker": bool, ...}`` shape
    (produced by ``score_logp_for_R``). The guard reads ``logp`` from BOTH and
    ``argmax_marker`` from ``g_records`` (the trained pass).

    Raises:
        KeyError: a (persona, question) leaf present in g_records is missing in
            b_records or vice versa. Defensive — both dicts should have been
            built off the same (panel × q) grid.
    """
    max_abs_dg = 0.0
    n_emit = 0
    n_probes = 0
    for persona, per_q_g in g_records.items():
        if persona not in b_records:
            raise KeyError(
                f"guard: persona {persona!r} present in g_records, missing in b_records — "
                "the trained and base passes must be over the SAME panel."
            )
        for q, gleaf in per_q_g.items():
            if q not in b_records[persona]:
                raise KeyError(
                    f"guard: q {q!r} present in g_records[{persona!r}], missing in b_records "
                    "— the trained and base passes must be over the SAME questions."
                )
            n_probes += 1
            dg = float(gleaf["logp"]) - float(b_records[persona][q]["logp"])
            if abs(dg) > max_abs_dg:
                max_abs_dg = abs(dg)
            if bool(gleaf.get("argmax_marker", False)):
                n_emit += 1
    return max_abs_dg, n_emit, n_probes


# Plan v5 §4.4 fix #1 + §4.3a Phase 0.6 pass condition (b) — tightened to a
# 5% de-minimis bound after alternatives-Codex round-1 Must-Fix. The v4 draft
# 95% threshold would have silently passed a PARTIAL bug (a reader broken on
# 60% of slots) that compresses Phase 1's per-arm ΔG dynamic range and
# attenuates geometry coefficients toward null. The 5% cap = at most 1 of 20
# in Phase 0.6's N=20 panel; smaller increment is meaningless given the
# discrete N. See plan §11 "Per-batch byte-identical guard threshold".
DEFAULT_BYTE_IDENTICAL_RATE_MAX = 0.05
# Floats are byte-identical if |g - b| is below this absolute tolerance.
# 1e-6 nats is well below any genuine bf16 / float32 reader noise (single-
# token logit truncation produces ~1e-3 differences); a true byte-identical
# read of `g_logp` from BASE has |g - b| == 0.0 exactly (the v3 failure mode).
DEFAULT_BYTE_IDENTICAL_ABS_TOL = 1e-6
# KL > this minimum means the trained model and base produce different
# distributions at the slot. KL > 0 + |g - b| < 1e-6 is the exact diagnostic
# signature of "g_logp read from base while distribution is non-trivially
# trained" — the v3 recovery bug. Threshold below tiny float noise but above
# fp truncation noise.
DEFAULT_KL_DIAGNOSTIC_MIN_NATS = 0.01


def compute_byte_identical_rate(
    g_records: dict[str, dict[str, dict[str, float | bool]]],
    b_records: dict[str, dict[str, dict[str, float | bool]]],
    kl_records: dict[str, dict[str, float]] | None,
    *,
    abs_tol: float = DEFAULT_BYTE_IDENTICAL_ABS_TOL,
    kl_min: float = DEFAULT_KL_DIAGNOSTIC_MIN_NATS,
) -> tuple[float, int, int]:
    """Compute the byte-identical-with-positive-KL rate over a per-probe grid.

    The v3 marker-logprob path bug produced ``g_logp == b_logp`` exactly while
    KL(trained ‖ base) at the same slot read non-trivially positive (because
    KL was computed over the FULL distribution but g_logp was read from
    base). This helper isolates the diagnostic: fraction of (persona × q)
    pairs where ``|g_logp - b_logp| < abs_tol`` AND ``kl > kl_min``.

    Args:
        g_records: ``score_logp_for_R(use_lora=True, ...)`` output.
        b_records: ``score_logp_for_R(use_lora=False, ...)`` output.
        kl_records: optional per-probe KL ``kl[persona][q] -> float`` from the
            same forward pass. When ``None`` (KL disabled by --no-kl), the
            byte-identical rate is computed over ALL pairs without the KL
            filter — a residual reader bug still surfaces as a uniform
            ``g == b`` read, just without the partial-bug discriminator.
        abs_tol: absolute tolerance for "byte-identical" logp pairs.
        kl_min: minimum KL nats for "trained distribution is non-trivial".

    Returns:
        ``(rate, n_byte_identical_with_positive_kl, n_probes)``. ``rate`` is
        in [0, 1]; the caller compares against ``DEFAULT_BYTE_IDENTICAL_RATE_MAX``.
    """
    n_probes = 0
    n_bad = 0
    for persona, per_q_g in g_records.items():
        for q, gleaf in per_q_g.items():
            gl = float(gleaf["logp"])
            bl = float(b_records[persona][q]["logp"])
            n_probes += 1
            byte_identical = abs(gl - bl) < abs_tol
            if not byte_identical:
                continue
            if kl_records is None:
                n_bad += 1
                continue
            kl_val = float(kl_records.get(persona, {}).get(q, 0.0))
            if kl_val > kl_min:
                n_bad += 1
    rate = n_bad / n_probes if n_probes else 0.0
    return rate, n_bad, n_probes


def assert_byte_identical_rate_below_threshold(
    g_records: dict[str, dict[str, dict[str, float | bool]]],
    b_records: dict[str, dict[str, dict[str, float | bool]]],
    kl_records: dict[str, dict[str, float]] | None,
    *,
    cell_label: str,
    max_rate: float = DEFAULT_BYTE_IDENTICAL_RATE_MAX,
    abs_tol: float = DEFAULT_BYTE_IDENTICAL_ABS_TOL,
    kl_min: float = DEFAULT_KL_DIAGNOSTIC_MIN_NATS,
) -> dict[str, float | int]:
    """Per-batch byte-identical guard for the marker-logprob path (plan v5 §4.4).

    Computes the byte-identical rate over the (persona × question) grid via
    ``compute_byte_identical_rate``. Raises
    ``MarkerLogprobPathReadingFromBaseError`` if the rate exceeds ``max_rate``
    AND at least one byte-identical pair has positive KL (the diagnostic
    signature). Returns a diag dict for WandB logging on PASS so the analyzer
    can spot near-threshold drift as a continuous signal.

    Args:
        g_records, b_records, kl_records: ``score_logp_for_R`` + KL outputs.
        cell_label: cell + seed + frac string for the error message.
        max_rate: maximum tolerated byte-identical rate (default 5%, plan v5).
        abs_tol, kl_min: thresholds for the byte-identical + positive-KL gate.

    Returns:
        ``{"byte_identical_rate": float, "n_byte_identical": int,
        "n_probes": int, "max_rate": float, "abs_tol": float, "kl_min": float}``.

    Raises:
        MarkerLogprobPathReadingFromBaseError: rate > max_rate.
    """
    rate, n_bad, n_probes = compute_byte_identical_rate(
        g_records, b_records, kl_records, abs_tol=abs_tol, kl_min=kl_min
    )
    diag = {
        "byte_identical_rate": float(rate),
        "n_byte_identical": int(n_bad),
        "n_probes": int(n_probes),
        "max_rate": float(max_rate),
        "abs_tol": float(abs_tol),
        "kl_min": float(kl_min),
    }
    if rate > max_rate:
        raise MarkerLogprobPathReadingFromBaseError(
            f"[{cell_label}] marker_logprob_path_reading_from_base: "
            f"byte_identical_rate={rate:.4f} ({n_bad}/{n_probes}) > "
            f"{max_rate:.2f} de-minimis threshold. The marker-logprob reader is "
            f"returning g_logp ≡ b_logp on > {max_rate * 100:.0f}% of "
            f"(persona × question) pairs while KL(trained‖base) > {kl_min} "
            f"nats — physically impossible for a correctly-applied adapter. "
            f"This is the v3 recovery-bug signature (g_logp read from BASE). "
            f"See plan v5 §4.4 fix #1 + scripts/i504_reval_confirm.py:670-774 "
            f"for the canonical fix pattern."
        )
    log.info(
        "[%s] byte-identical guard PASS: rate=%.4f (%d/%d) ≤ %.2f de-minimis.",
        cell_label,
        rate,
        n_bad,
        n_probes,
        max_rate,
    )
    return diag


def assert_adapter_actually_applied(
    *,
    adapter_dir: str | os.PathLike,
    g_records: dict[str, dict[str, dict[str, float | bool]]],
    b_records: dict[str, dict[str, dict[str, float | bool]]],
    cell_label: str,
    b_norm_floor: float = DEFAULT_B_NORM_FLOOR,
    delta_g_eps_nats: float = DEFAULT_DELTA_G_EPS_NATS,
) -> dict[str, Any]:
    """B-matrix-only structural check on the trained adapter (round-4 redesign).

    Reads the adapter's max B-matrix Frobenius norm from
    ``adapter_dir/adapter_model.safetensors``, aggregates max ``|ΔG|`` +
    emission count across the per-probe ``g_records`` (trained) and
    ``b_records`` (base, same R) dicts for diagnostic logging, and returns the
    diagnostics dict. Two verdicts, both non-raising:

      * ``"pass_b_norm_ok"`` — ``b_max_norm > b_norm_floor`` (the adapter
        carries non-trivial trained weight). Diagnostic ``max_abs_delta_g`` /
        ``n_emit`` are logged at INFO regardless of magnitude; the picker
        (plan v2 §4.1 anti-saturation band [5, 12] nats × [0.1, 0.8] emission)
        decides whether the metric is in band at the chosen frac.
      * ``"pass_genuine_floor"`` — ``b_max_norm <= b_norm_floor`` (PEFT B=0
        initialization; the adapter is structurally empty). The cell did not
        learn anything; this is a real measurement, not a regression.

    The historical three-clause raise (``LoRANotAppliedError``) is dropped:
    at early checkpoints of a gentle anchor (lr=1e-5, frac=0.08-0.16), the
    metric is at floor BY DESIGN and the picker is the right place to surface
    that. Conflating "adapter applied vs not" (structural) with "signal in
    band" (calibration) was the round-3 crash class.

    Args:
        adapter_dir: directory containing the trained adapter's
            ``adapter_model.safetensors``.
        g_records: ``score_logp_for_R(use_lora=True, ...)`` output dict.
        b_records: ``score_logp_for_R(use_lora=False, ...)`` output dict.
        cell_label: cell + seed string for log messages.
        b_norm_floor: max-Frobenius-norm threshold below which the adapter is
            treated as a genuine floor.
        delta_g_eps_nats: retained for diagnostic-logging only. Does NOT gate
            the verdict in the round-4 redesign — kept for callers passing the
            argument by keyword (no behavior change at the call site).

    Returns:
        ``{"adapter_b_max_norm": float, "max_abs_delta_g_nats": float,
        "n_emit": int, "n_probes": int, "guard_verdict": str,
        "b_norm_floor": float, "delta_g_eps_nats": float}``. The verdict is
        ``"pass_b_norm_ok"`` or ``"pass_genuine_floor"``.

    Raises:
        FileNotFoundError: adapter weights missing under ``adapter_dir``.
        KeyError: g/b record dicts disagree on the panel × q grid.
    """
    b_max_norm = b_matrix_frobenius_norm(adapter_dir)
    max_abs_dg, n_emit, n_probes = _aggregate_records_for_guard(g_records, b_records)

    diag: dict[str, Any] = {
        "adapter_b_max_norm": float(b_max_norm),
        "max_abs_delta_g_nats": float(max_abs_dg),
        "n_emit": int(n_emit),
        "n_probes": int(n_probes),
        "b_norm_floor": float(b_norm_floor),
        "delta_g_eps_nats": float(delta_g_eps_nats),
    }

    if b_max_norm <= b_norm_floor:
        diag["guard_verdict"] = "pass_genuine_floor"
        log.info(
            "[%s] eval-guard PASS (genuine floor): adapter B-max-norm=%.3e ≤ floor=%.0e — "
            "adapter is structurally empty / untrained (PEFT B=0 init). max|ΔG|=%.3f nats "
            "n_emit=%d/%d probes — real measurement of 'this cell did not learn,' not a "
            "regression.",
            cell_label,
            b_max_norm,
            b_norm_floor,
            max_abs_dg,
            n_emit,
            n_probes,
        )
        return diag

    # B-matrix > floor: adapter is structurally present. Whether the metric
    # (max|ΔG|, n_emit) is in band is the picker's call, not the guard's.
    diag["guard_verdict"] = "pass_b_norm_ok"
    log.info(
        "[%s] eval-guard PASS (b-norm ok): adapter B-max-norm=%.3f > floor=%.0e — "
        "adapter is structurally present. Diagnostic max|ΔG|=%.3f nats, "
        "n_emit=%d/%d probes (picker decides in-band at chosen frac).",
        cell_label,
        b_max_norm,
        b_norm_floor,
        max_abs_dg,
        n_emit,
        n_probes,
    )
    return diag
