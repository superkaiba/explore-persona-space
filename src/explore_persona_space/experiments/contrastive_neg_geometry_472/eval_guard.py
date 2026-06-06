# ruff: noqa: RUF002  # em-dash + Qwen marker " ※" + Greek ΔG intentional
"""Task #472/#477 eval guard — fail-loud detector for the silent LoRA-not-applied
regression that produced the #477 v4/v6 artifact (re-eval round-1 of
``c477_calib_negp_2_seed42_lr2e-06`` confirmed: v4/v6 reported ΔG ≈ 0 across
every probe while a clean re-eval at the SAME adapter reads source ΔG = +20.5).

Root-cause sketch (confirmed by ``scripts/i477_reval_confirm.py``): the trained
adapter loaded into vLLM via ``LoRARequest`` was a genuinely-trained LoRA (B-
matrix norm ~3, non-trivial), but the merged forward pass silently skipped the
adapter, so ``score_logp_for_R(use_lora=True)`` returned BASE log-probs and ΔG
collapsed to ≈ 0 at every (persona, question). The class of bug it expresses:

  * adapter loaded, B-matrix norm > 0 (the LoRA is REAL),
  * max ``|ΔG|`` across ALL probes (source-self + held-out) below a small ε,
  * on-policy emission (argmax == marker) is 0 everywhere.

Distinguished from a GENUINELY weak / untrained / collapsed adapter, where
B-matrix norm itself is ≈ 0 — the dispatcher must NOT treat that as a regression
(it is a real measurement of "this adapter didn't learn anything").

The guard returns silently when EITHER (a) B-matrix norm is below the
near-zero floor (genuinely-untrained adapter), OR (b) ``max |ΔG|`` exceeds the
epsilon (the adapter is applied and the eval is reading a real signal). It
raises ``LoRANotAppliedError`` only when BOTH (i) the adapter is real and (ii)
the eval reads no signal AND (iii) emission is identically zero — that is the
narrow regression class.
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
    """Raised when an adapter with B-norm > floor reads ΔG ≈ 0 at every probe.

    The #477 v4/v6 regression class: the LoRA loaded into vLLM via LoRARequest
    is genuinely trained (B-matrix norm well above the floor) but
    ``score_logp_for_R(use_lora=True)`` silently returns BASE log-probs, so ΔG
    collapses to ≈ 0 at EVERY (persona, question) and emission is uniformly 0.
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


def assert_adapter_actually_applied(
    *,
    adapter_dir: str | os.PathLike,
    g_records: dict[str, dict[str, dict[str, float | bool]]],
    b_records: dict[str, dict[str, dict[str, float | bool]]],
    cell_label: str,
    b_norm_floor: float = DEFAULT_B_NORM_FLOOR,
    delta_g_eps_nats: float = DEFAULT_DELTA_G_EPS_NATS,
) -> dict[str, Any]:
    """Fail-loud guard for the silent LoRA-not-applied regression (#477 v4/v6).

    Reads the adapter's max B-matrix Frobenius norm from
    ``adapter_dir/adapter_model.safetensors`` and aggregates max ``|ΔG|`` +
    emission count across the per-probe ``g_records`` (trained) and
    ``b_records`` (base, same R) dicts that ``score_logp_for_R`` already
    produced. Raises ``LoRANotAppliedError`` IFF all three hold:

      1. ``b_max_norm > b_norm_floor`` (the adapter is genuinely trained), AND
      2. ``max_abs_delta_g < delta_g_eps_nats`` (the eval reads no signal), AND
      3. ``n_emit == 0`` (uniformly zero on-policy emission).

    Returns the aggregated diagnostics dict either way (for logging /
    persisting alongside the cell's eval JSON).

    Args:
        adapter_dir: directory containing the trained adapter's
            ``adapter_model.safetensors``.
        g_records: ``score_logp_for_R(use_lora=True, ...)`` output dict.
        b_records: ``score_logp_for_R(use_lora=False, ...)`` output dict.
        cell_label: cell + seed string for error messages / logs.
        b_norm_floor: max-Frobenius-norm threshold below which the adapter is
            treated as a genuine floor (not a regression).
        delta_g_eps_nats: max |ΔG| threshold below which we treat the eval as
            "no signal across the entire panel."

    Returns:
        ``{"adapter_b_max_norm": float, "max_abs_delta_g_nats": float,
        "n_emit": int, "n_probes": int, "guard_verdict": str}`` where
        ``guard_verdict`` is one of:
          * ``"pass_real_signal"`` — adapter applied, ΔG above eps somewhere;
          * ``"pass_genuine_floor"`` — adapter B-norm at/under the floor
            (untrained / collapsed; no regression to flag);
          * ``"pass_some_emission"`` — adapter real, ΔG below eps everywhere
            but emission > 0 somewhere (unusual but not the #477 regression).

        On the regression class the function RAISES; it never returns
        ``"fail_*"``.

    Raises:
        LoRANotAppliedError: the three-clause regression condition triggered.
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
            "adapter is effectively untrained, ΔG ≈ 0 is a real measurement, not a regression.",
            cell_label,
            b_max_norm,
            b_norm_floor,
        )
        return diag

    if max_abs_dg >= delta_g_eps_nats:
        diag["guard_verdict"] = "pass_real_signal"
        log.info(
            "[%s] eval-guard PASS (real signal): adapter B-max-norm=%.3f > floor=%.0e, "
            "max|ΔG|=%.3f ≥ eps=%.2f nats — the LoRA is applied at eval.",
            cell_label,
            b_max_norm,
            b_norm_floor,
            max_abs_dg,
            delta_g_eps_nats,
        )
        return diag

    if n_emit > 0:
        diag["guard_verdict"] = "pass_some_emission"
        log.warning(
            "[%s] eval-guard PASS (some emission): adapter B-max-norm=%.3f > floor, "
            "max|ΔG|=%.3f < eps=%.2f nats, but n_emit=%d/%d probes argmax==marker — the "
            "LoRA is at least partially expressed at decode time. Likely a recipe-saturation "
            "or weak-adapter regime, not the #477 silent-LoRA-not-applied regression.",
            cell_label,
            b_max_norm,
            max_abs_dg,
            delta_g_eps_nats,
            n_emit,
            n_probes,
        )
        return diag

    # ALL THREE clauses hold — the #477 v4/v6 regression class.
    raise LoRANotAppliedError(
        f"[{cell_label}] LoRA-not-applied regression (#477 v4/v6 class):\n"
        f"  adapter B-max-norm  = {b_max_norm:.4f}  (> floor {b_norm_floor:.0e}, "
        "adapter is genuinely trained)\n"
        f"  max|ΔG| across panel = {max_abs_dg:.4f} nats  (< eps "
        f"{delta_g_eps_nats:.2f}, eval reads NO signal)\n"
        f"  n_emit              = {n_emit}/{n_probes}  (argmax==marker NOWHERE)\n"
        "  → trained adapter has weight but eval log-probs match BASE everywhere and "
        "the model never emits the marker — the LoRA was loaded but silently NOT applied "
        "during the trained pass (see scripts/i477_reval_confirm.py for the dispositive "
        "diagnostic; investigate vLLM/PEFT version drift, LoRARequest threading, or "
        "adapter rank vs max_lora_rank mismatch before re-running)."
    )
