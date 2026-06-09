"""Issue #464 positive-only follow-up — analysis + headline stats.

Reads per-cell JSONs produced by ``i464_po_eval.py`` and mirrors the
parent #464 analyzer's headline shape (``i464_phase5_analyze.py``):
paired-bootstrap CIs on per-seed deltas, H1 elicitation gate, dynamic-
range gate. Differences from the parent:

  * Only 3 arms (system_plain / system_padded / role) — role_nonsense
    and role_mismatch are NOT replicated.
  * 18 cells = 3 arms x 3 seeds x 2 personas (one persona per LoRA).
  * Off-diagonal leakage probe = ``g_logprob`` of ` ※` under the OTHER
    persona's SAME-ARM-FAMILY encoding (system family for
    system_plain/system_padded; role family for role).
  * ``L_arm_seed`` is the mean across the 2 single-persona cells'
    off-diagonal log-probs (instead of averaging the parent's 4-cell
    symmetric set).
  * NEW: leakage-to-default — ` ※` under ``default_assistant`` per arm,
    averaged across all 6 cells per arm (3 seeds x 2 personas). This
    is the on-default-context number the parent's co-residence
    couldn't measure.

Inputs (read-only):
  eval_results/issue_464/positive_only/cross_eval/per_cell/<cell>__<e_eval>.json
      18 cells x 3 e_eval = 54 files

Outputs:
  eval_results/issue_464/positive_only/analysis.json — mirrors the
      parent's analysis shape: per-arm-per-seed L, headline deltas with
      paired bootstrap CIs, H1 elicitation, leakage-to-default,
      dynamic-range gate, raw per-cell.

Headline statistic:
  Per seed:
    L_arm_seed = mean over (training_persona ∈ {pirate, villain}) of
      raw g_logprob(` ※`, e_off-diagonal)
      where e_off-diagonal = the OTHER persona's same-arm-family encoding:
        pirate-only cell → ` ※` under (system_villain  if arm ∈ system_*,
                                       role_villain    if arm == role)
        villain-only cell → ` ※` under (system_pirate  if arm ∈ system_*,
                                        role_pirate    if arm == role)

    d_seed_plain  = L_system_plain  - L_role   (>0 ⇒ role leaks less)
    d_seed_padded = L_system_padded - L_role

  H2 PASS (mirrors parent's threshold):
    mean(d_plain)  ≥ 1.0 nat AND 95% CI > 0 AND all per-seed d > 0
    mean(d_padded) ≥ 1.0 nat AND 95% CI > 0 AND all per-seed d > 0

CLI:
    uv run python scripts/i464_po_analyze.py
    uv run python scripts/i464_po_analyze.py --allow-partial
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import logging
import statistics
import subprocess
import sys
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

from explore_persona_space.experiments import i464_encodings as enc

# Ensure repo root is on sys.path so `from scripts.X import Y` resolves
# when this script is invoked directly via `uv run python scripts/...`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Mirror the parent analyzer's thresholds + bootstrap helper exactly so
# the two follow-ups are read against a single methodology.
from scripts.i464_phase5_analyze import (  # type: ignore[import-not-found]
    DYNAMIC_RANGE_THRESHOLD,
    H1_ELICITATION_THRESHOLD,
    H2_HEADLINE_THRESHOLD,
    H2_MIN_SEEDS,
    N_BOOTSTRAP,
    _paired_bootstrap_ci,
)

load_dotenv()

logger = logging.getLogger("i464.po_analyze")

# Per-variant input + output paths. Selected at runtime from --variant.
# Defaults preserve the positive-only (``po``) behavior so existing
# call sites stay byte-identical.
PER_CELL_DIR_FOR: dict[str, Path] = {
    "po": Path("eval_results/issue_464/positive_only/cross_eval/per_cell"),
    "cn": Path("eval_results/issue_464/contrastive_negatives/cross_eval/per_cell"),
    "cn_i529": Path("eval_results/issue_529/contrastive_negatives/cross_eval/per_cell"),
    "cn_i533": Path("eval_results/issue_533/contrastive_negatives/cross_eval/per_cell"),
}
OUT_PATH_FOR: dict[str, Path] = {
    "po": Path("eval_results/issue_464/positive_only/analysis.json"),
    "cn": Path("eval_results/issue_464/contrastive_negatives/analysis.json"),
    "cn_i529": Path("eval_results/issue_529/contrastive_negatives/analysis.json"),
    "cn_i533": Path("eval_results/issue_533/contrastive_negatives/analysis.json"),
}
SCHEMA_VERSION_FOR: dict[str, str] = {
    "po": "i464_po_analyze_v1",
    "cn": "i464_cn_analyze_v1",
    "cn_i529": "i529_cn_analyze_v1",
    "cn_i533": "i533_cn_analyze_v1",
}

# Legacy aliases (positive-only defaults) — kept for any importer that
# referenced these constants before --variant existed.
PER_CELL_DIR = PER_CELL_DIR_FOR["po"]
OUT_PATH = OUT_PATH_FOR["po"]

# Module-level state set from --variant before any helper consumes it.
# Helpers read from this dict instead of the legacy globals so the
# variant choice flows through without each helper needing an extra arg.
# ``selected_epoch`` is consumed only on the cn_i529 path; the po/cn paths
# carry None and helpers see no epoch suffix.
_ACTIVE: dict[str, object] = {
    "per_cell_dir": PER_CELL_DIR,
    "selected_epoch_per_persona": None,
    # main() stashes the resolved per-variant seed list here so module-
    # level helpers (e.g. ``_leakage_to_default``) iterate the same
    # seeds the variant uses. None until main() runs.
    "seeds": None,
}

# Per-variant seed sets (mirrors po_eval's SEEDS_FOR).
SEEDS_FOR: dict[str, tuple[int, ...]] = {
    "po": (42, 137, 1337),
    "cn": (42, 137, 1337),
    "cn_i529": (42, 137, 1337, 7, 21),
    "cn_i533": (42, 137, 1337, 7, 21),
}
SEEDS = SEEDS_FOR["po"]
PO_ARMS: tuple[enc.Arm, ...] = ("system_plain", "system_padded", "role")
SHARED_MARKER_PERSONA: enc.Persona = "pirate"


def _git_commit_hash() -> str:
    """Return the current HEAD sha or 'unknown' if git is unavailable."""
    try:
        import os

        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            env={**os.environ},
        )
        return out.decode().strip()
    except Exception:
        return "unknown"


def _po_cell_label(arm: enc.Arm, seed: int, persona: enc.Persona, epoch: int | None = None) -> str:
    """Canonical cell label; matches train + eval.

    * po/cn (epoch=None): ``{arm}_seed{seed}_{persona}``
    * cn_i529 (epoch=E):  ``{arm}_seed{seed}_cn_{persona}_e{E}``
    """
    if epoch is not None:
        return f"{arm}_seed{seed}_cn_{persona}_e{epoch}"
    return f"{arm}_seed{seed}_{persona}"


def _epoch_for(persona: enc.Persona) -> int | None:
    """Return the active anchor epoch for ``persona`` (cn_i529 only) or None.

    The cn_i529 path stores ``{pirate: E*, villain: E*}`` in
    ``_ACTIVE['selected_epoch_per_persona']``; helpers below read this
    dict to splice E* into the per-cell label without each call site
    threading an extra argument.
    """
    sel = _ACTIVE.get("selected_epoch_per_persona")
    if sel is None:
        return None
    if not isinstance(sel, dict):
        return None
    return sel.get(persona)


def _load_per_cell(
    arm: enc.Arm,
    seed: int,
    persona: enc.Persona,
    e_eval: str,
    epoch: int | None = None,
) -> dict | None:
    """Read one per-cell JSON or return None if missing.

    Reads from ``_ACTIVE['per_cell_dir']`` (set in main from --variant)
    rather than the module-level ``PER_CELL_DIR`` so the same helper
    serves po, cn, and cn_i529 paths. The cn_i529 path additionally
    splices the selected anchor's epoch into the cell label.
    """
    per_cell_dir = _ACTIVE["per_cell_dir"]
    assert isinstance(per_cell_dir, Path)
    effective_epoch = epoch if epoch is not None else _epoch_for(persona)
    p = per_cell_dir / f"{_po_cell_label(arm, seed, persona, effective_epoch)}__{e_eval}.json"
    if not p.exists() or p.stat().st_size == 0:
        return None
    return json.loads(p.read_text())


def _own_eval_encoding_for(arm: enc.Arm, persona: enc.Persona) -> str:
    """Diagonal eval encoding for ``(arm, persona)`` in the positive-only follow-up.

    Mirrors the parent's ``_own_eval_encoding_for`` restricted to the
    3 PO_ARMS.
    """
    if arm == "role":
        return f"role_{persona}"
    return f"system_{persona}"


def _other_eval_encoding_for(arm: enc.Arm, persona: enc.Persona) -> str:
    """Off-diagonal eval encoding for ``(arm, persona)`` — the OTHER persona's
    SAME-arm-family encoding (matches the brief's headline definition)."""
    other: enc.Persona = "villain" if persona == "pirate" else "pirate"
    if arm == "role":
        return f"role_{other}"
    return f"system_{other}"


def _symmetric_leakage(arm: enc.Arm, seed: int) -> tuple[float, list[float]]:
    """Return (L_arm_seed, raw_logprobs_per_cell).

    Off-diagonal cells: for each training persona p ∈ {pirate, villain},
    read ` ※` log-prob under the OTHER persona's same-arm-family encoding.
    Mean of the 2 raw log-probs.
    """
    raw: list[float] = []
    for persona in enc.PERSONAS:
        e_off = _other_eval_encoding_for(arm, persona)
        payload = _load_per_cell(arm, seed, persona, e_off)
        if payload is None:
            raise FileNotFoundError(
                f"analyze: missing per-cell JSON for {_po_cell_label(arm, seed, persona)}/{e_off}"
            )
        raw.append(payload["g_logprob"])
    if not raw:
        raise RuntimeError(f"po off-diagonal leakage cells empty for arm={arm} seed={seed}")
    return float(np.mean(raw)), raw


def _wrong_logp_per_seed_for(
    arm: enc.Arm, persona: enc.Persona, seeds: tuple[int, ...]
) -> list[float]:
    """Wrong-slot ` ※` mean log P at ``(arm, persona)`` for each seed.

    Used by the cn_i533 per-persona paired-d block (see plan §4 (d)):
    reads at the persona's selected anchor (via ``_epoch_for(persona)``)
    the per-cell JSON under the OFF-DIAGONAL eval encoding — the
    same wrong-slot probe ``_symmetric_leakage`` reads, but kept
    per-persona instead of averaged across personas BEFORE forming
    the paired d. Returns one value per seed.
    """
    e_off = _other_eval_encoding_for(arm, persona)
    out: list[float] = []
    for seed in seeds:
        payload = _load_per_cell(arm, seed, persona, e_off)
        if payload is None:
            raise FileNotFoundError(
                f"per-persona wrong-slot: missing per-cell JSON for "
                f"{_po_cell_label(arm, seed, persona, _epoch_for(persona))}/{e_off}"
            )
        out.append(float(payload["g_logprob"]))
    return out


def _per_persona_paired_d_block(
    seeds: tuple[int, ...], n_boot: int
) -> dict[str, dict[str, dict[str, float | bool]]]:
    """Build the 4-cell per-persona paired-d block for cn_i533.

    Returns ``{persona: {contrast: {mean, ci_lo, ci_hi, sign_agreement,
    d_per_seed, n_seeds}}}`` for persona ∈ {pirate, villain} and
    contrast ∈ {plain, padded}. plain = log P_system_plain - log P_role;
    padded = log P_system_padded - log P_role. Per-persona means EACH
    persona's wrong-slot log-prob is read at its OWN selected anchor
    (``_epoch_for(persona)``) — the analyzer-level helpers ``_load_per_cell``
    + ``_epoch_for`` already splice the right per-persona epoch into
    the per-cell filename.

    The plan brief's "sign_agreement" is the fraction of bootstrap
    resampled means that share the central estimate's sign.

    See plan §4 "Pseudocode for new code" item (d): the inherited
    cn_i529 path averages personas via ``_symmetric_leakage`` BEFORE
    forming d (so the saturated-floor #529 case had no per-persona
    resolution), but the #533 lr=5e-6 corrective re-run needs the
    per-persona view to drive the H1/H0 verdict. The persona-averaged
    ``headline.d_seed_plain`` / ``headline.d_seed_padded`` keys stay as
    a SECONDARY cross-check.
    """
    out: dict[str, dict[str, dict[str, float | bool]]] = {}
    rng = np.random.default_rng(42)
    for persona in enc.PERSONAS:
        # Read per-arm per-seed wrong-slot log P at this persona's
        # selected anchor.
        L_plain = _wrong_logp_per_seed_for("system_plain", persona, seeds)
        L_padded = _wrong_logp_per_seed_for("system_padded", persona, seeds)
        L_role = _wrong_logp_per_seed_for("role", persona, seeds)
        d_plain = [a - b for a, b in zip(L_plain, L_role, strict=True)]
        d_padded = [a - b for a, b in zip(L_padded, L_role, strict=True)]
        out[persona] = {}
        for contrast_name, d_per_seed in (("plain", d_plain), ("padded", d_padded)):
            arr = np.array(d_per_seed, dtype=float)
            n = len(arr)
            means = np.empty(n_boot)
            for b in range(n_boot):
                idx = rng.integers(0, n, size=n)
                means[b] = arr[idx].mean()
            lo, hi = np.quantile(means, [0.025, 0.975])
            central = float(arr.mean())
            central_sign = 1.0 if central > 0 else (-1.0 if central < 0 else 0.0)
            if central_sign == 0.0:
                sign_agree = 0.5
            else:
                sign_agree = float(np.mean(np.sign(means) == central_sign))
            out[persona][contrast_name] = {
                "mean": central,
                "ci_lo_95": float(lo),
                "ci_hi_95": float(hi),
                "sign_agreement": sign_agree,
                "d_per_seed": d_per_seed,
                "n_seeds": n,
                "n_bootstrap": n_boot,
            }
    return out


# Per-persona H1 magnitude floor (plan §6 H1 threshold). A per-persona
# cell drives H1 only if BOTH its 95% paired-bootstrap CI clears zero on
# the positive side AND the central mean d >= this value.
H1_PER_PERSONA_THRESHOLD = 0.5


def _headline_verdict_from_per_persona(
    per_persona: dict[str, dict[str, dict[str, float | bool]]],
) -> tuple[str, bool, bool]:
    """Compute the cn_i533 H1/H0/inconclusive verdict from per-persona cells.

    Round-2 reconciler binding finding: the inherited cn_i529 H2 rule at
    lines ~706-708 (1.0-nat both-contrast persona-averaged rule) cannot
    drive the cn_i533 H1/H0 verdict because ``_symmetric_leakage`` averages
    personas BEFORE forming d, so the persona-averaged read can hide an
    opposite-signed per-persona pair. Plan §6 thresholds:

      H1 (positive resolution): AT LEAST ONE of the 4 cells (pirate x
        plain, pirate x padded, villain x plain, villain x padded) has
        ``ci_lo_95 > 0`` AND ``mean >= H1_PER_PERSONA_THRESHOLD`` (0.5 nat).
      H0 (null resolution): ALL 4 cells straddle zero, i.e.
        ``ci_lo_95 <= 0 <= ci_hi_95``.

    Both resolve the experiment's question (plan §3 / §6); the caller
    maps both to ``headline_status='ok'``. ``inconclusive`` covers the
    rare case where the per-persona pattern fits NEITHER H1 nor H0 (e.g.
    one cell positive but sub-threshold, one cell straddling zero, one
    cell clearly negative); the caller maps this to ``'partial'``.

    Returns ``(verdict, h1_per_persona_pass, h0_per_persona_pass)`` where
    ``verdict`` is one of ``{'h1', 'h0', 'inconclusive'}``. The booleans
    are reported in the payload as standalone fields for downstream
    auditability (e.g. a degenerate case where both fire would still
    route to 'h1' here but the analyst can read both flags).
    """
    cells: list[dict[str, float | bool]] = []
    for persona_key in ("pirate", "villain"):
        for contrast_key in ("plain", "padded"):
            cells.append(per_persona[persona_key][contrast_key])
    h1_pass = any(
        float(c["ci_lo_95"]) > 0.0 and float(c["mean"]) >= H1_PER_PERSONA_THRESHOLD for c in cells
    )
    h0_pass = all(float(c["ci_lo_95"]) <= 0.0 <= float(c["ci_hi_95"]) for c in cells)
    if h1_pass:
        verdict = "h1"
    elif h0_pass:
        verdict = "h0"
    else:
        verdict = "inconclusive"
    return verdict, h1_pass, h0_pass


def _own_persona_elicitation(arm: enc.Arm, seed: int) -> tuple[list[float], list[str]]:
    """H1 gate input: raw trained log P on each (training_persona, own-encoding) cell.

    Returns ([logp_pirate_cell, logp_villain_cell], [label_pirate, label_villain]).
    """
    own_logps: list[float] = []
    labels: list[str] = []
    for persona in enc.PERSONAS:
        e_own = _own_eval_encoding_for(arm, persona)
        payload = _load_per_cell(arm, seed, persona, e_own)
        if payload is None:
            raise FileNotFoundError(
                f"analyze H1: missing own-encoding cell "
                f"{_po_cell_label(arm, seed, persona)}/{e_own}"
            )
        own_logps.append(float(payload["g_logprob"]))
        labels.append(f"{_po_cell_label(arm, seed, persona)}/{e_own}")
    return own_logps, labels


def _active_seeds() -> tuple[int, ...]:
    """Return the seed set the current invocation should iterate.

    Closes the `leakage-to-default-seeds-undercount-cn-i529` round-1
    concern: prior to round-2, ``_leakage_to_default`` hardcoded the
    module-global ``SEEDS = SEEDS_FOR['po'] = (42, 137, 1337)``, which
    silently DROPPED seeds 7 and 21 from the leakage-to-default
    diagnostic on the ``cn_i529`` path (which uses all 5 seeds). The
    fix: main() stashes ``args.seeds`` into ``_ACTIVE['seeds']``; this
    helper reads back that variant-aware set. When ``_ACTIVE['seeds']``
    is unset (e.g. tests that exercise helpers without going through
    main()) we fall back to the legacy 3-seed default.
    """
    seeds = _ACTIVE.get("seeds")
    if isinstance(seeds, (list, tuple)) and seeds:
        return tuple(int(s) for s in seeds)
    return tuple(SEEDS)


def _leakage_to_default(arm: enc.Arm) -> tuple[list[float], list[str]]:
    """` ※` log-prob under ``default_assistant`` for every cell in this arm.

    The NEW measurement the parent #464 could NOT make (co-residence + the
    two-marker contrast in the parent meant default_assistant was a
    diagnostic side note, not a co-axial bystander). Returns (per_cell_logp,
    per_cell_label) across (active_seeds x persona) cells per arm; the
    active seed set is variant-aware (3 for po/cn, 5 for cn_i529) so
    seeds 7 and 21 are NOT silently dropped on the cn_i529 path.
    """
    logps: list[float] = []
    labels: list[str] = []
    for seed in _active_seeds():
        for persona in enc.PERSONAS:
            payload = _load_per_cell(arm, seed, persona, "default_assistant")
            if payload is None:
                raise FileNotFoundError(
                    f"analyze leakage-to-default: missing cell "
                    f"{_po_cell_label(arm, seed, persona)}/default_assistant"
                )
            logps.append(float(payload["g_logprob"]))
            labels.append(f"{_po_cell_label(arm, seed, persona)}/default_assistant")
    return logps, labels


def _h2_verdict(name: str, d_per_seed: list[float], mean: float, lo: float, hi: float) -> dict:
    """Pack a single-comparison H2 verdict (mirrors parent's ``_h2_verdict``)."""
    all_positive = all(d > 0 for d in d_per_seed)
    ci_excludes_zero = lo > 0
    threshold_met = mean >= H2_HEADLINE_THRESHOLD
    passed = all_positive and ci_excludes_zero and threshold_met
    reasons: list[str] = []
    if not threshold_met:
        reasons.append(f"mean(d_{name})={mean:.3f} < {H2_HEADLINE_THRESHOLD}")
    if not ci_excludes_zero:
        reasons.append(f"95% CI [{lo:.3f}, {hi:.3f}] overlaps zero")
    if not all_positive:
        reasons.append(f"per-seed d signs not all positive: {d_per_seed}")
    return {
        "d_per_seed": d_per_seed,
        "mean": mean,
        "ci_lo_95": lo,
        "ci_hi_95": hi,
        "all_seeds_positive": all_positive,
        "ci_excludes_zero": ci_excludes_zero,
        "mean_threshold": H2_HEADLINE_THRESHOLD,
        "threshold_met": threshold_met,
        "pass": passed,
        "fail_reasons": reasons,
    }


def _compute_dynamic_range_gate(
    raw_per_cell: dict[str, dict[int, list[float]]],
) -> tuple[dict[str, dict], bool]:
    """Return (per-arm sd+threshold-pass dict, overall gate ok bool)."""
    dr_gate: dict[str, dict] = {}
    for arm in PO_ARMS:
        all_raw: list[float] = []
        for seed_raw in raw_per_cell.get(arm, {}).values():
            all_raw.extend(seed_raw)
        if all_raw:
            sd = statistics.pstdev(all_raw)
            dr_gate[arm] = {
                "sd": sd,
                "n_observations": len(all_raw),
                "above_threshold": sd > DYNAMIC_RANGE_THRESHOLD,
            }
        else:
            dr_gate[arm] = {"sd": None, "n_observations": 0, "above_threshold": False}
    overall_ok = all(v["above_threshold"] for v in dr_gate.values())
    return dr_gate, overall_ok


def main(argv: list[str] | None = None) -> None:  # noqa: C901 - mirrors parent's structure
    """Entry point for the positive-only analyzer."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Seeds to aggregate. Default = variant-specific: (42, 137, 1337) "
            "for po/cn; (42, 137, 1337, 7, 21) for cn_i529 / cn_i533."
        ),
    )
    ap.add_argument(
        "--allow-partial",
        action="store_true",
        help="If set, skip missing per-cell files (smoke mode); else FAIL LOUD.",
    )
    ap.add_argument(
        "--variant",
        choices=("po", "cn", "cn_i529", "cn_i533"),
        default="po",
        help=(
            "Which follow-up to analyze. ``po`` (default) = positive-only "
            "(reads ``eval_results/issue_464/positive_only/cross_eval/per_cell/``, "
            "writes ``positive_only/analysis.json``). ``cn`` = "
            "contrastive-negatives (reads + writes under ``contrastive_negatives/``). "
            "``cn_i529`` = #529 non-saturated-anchor cn (reads + writes under "
            "``eval_results/issue_529/contrastive_negatives/``). ``cn_i533`` = "
            "#533 lr=5e-6 corrective re-run (reads + writes under "
            "``eval_results/issue_533/contrastive_negatives/``). cn_i529 / "
            "cn_i533 BOTH REQUIRE ``--anchor-file`` so the per-persona E* is "
            "set before per-cell loads."
        ),
    )
    ap.add_argument(
        "--anchor-file",
        type=str,
        default=None,
        help=(
            "Path to anchor_selection.json (cn_i529 / cn_i533 only). The "
            "file is produced by ``scripts/i529_select_anchor.py`` and "
            "carries ``selected_anchor: {pirate: E*, villain: E*}``; "
            "analyze reads ONLY the cells at those E* per persona to "
            "compute the headline statistic at the selected anchor "
            "(plan §4.5)."
        ),
    )
    args = ap.parse_args(argv)

    # Wire the variant choice into the module-level _ACTIVE dict so all
    # helpers (_load_per_cell, _symmetric_leakage, ...) read from the
    # right directory without each call site needing an extra arg.
    _ACTIVE["per_cell_dir"] = PER_CELL_DIR_FOR[args.variant]
    out_path_active = OUT_PATH_FOR[args.variant]
    schema_version = SCHEMA_VERSION_FOR[args.variant]
    seeds_default = list(SEEDS_FOR[args.variant])
    if args.seeds is None:
        args.seeds = seeds_default
    # Stash the resolved seed set into _ACTIVE so module-level helpers
    # (notably ``_leakage_to_default`` via ``_active_seeds``) read the
    # variant-aware seed list instead of the module global ``SEEDS``.
    # Closes the `leakage-to-default-seeds-undercount-cn-i529` round-1
    # concern — the cn_i529 path uses 5 seeds (42, 137, 1337, 7, 21).
    _ACTIVE["seeds"] = tuple(int(s) for s in args.seeds)
    # cn_i529 / cn_i533: REQUIRE --anchor-file unless --allow-partial.
    # The anchor file is the formal hand-off between
    # i529_select_anchor.py and this script — without it the analyzer
    # would load the wrong (or no) cells and silently produce a
    # malformed analysis.
    if args.variant in ("cn_i529", "cn_i533"):
        if args.anchor_file is None and not args.allow_partial:
            ap.error(
                f"--variant {args.variant} requires --anchor-file (run "
                "scripts/i529_select_anchor.py first; pass its output JSON path)."
            )
        if args.anchor_file is not None:
            anchor_payload = json.loads(Path(args.anchor_file).read_text())
            if anchor_payload.get("degenerate", False):
                logger.warning(
                    "%s: anchor_selection marked degenerate (%s). "
                    "Proceeding will write a degenerate analysis.json; the "
                    "headline statistic is not meaningful at saturation.",
                    args.variant,
                    anchor_payload.get("degenerate_reason", ""),
                )
            sel = anchor_payload.get("selected_anchor")
            if not isinstance(sel, dict) or set(sel) != {"pirate", "villain"}:
                ap.error(
                    f"--anchor-file {args.anchor_file}: selected_anchor "
                    f"must be a dict with keys 'pirate' and 'villain'; got {sel!r}"
                )
            _ACTIVE["selected_epoch_per_persona"] = {
                "pirate": int(sel["pirate"]) if sel.get("pirate") is not None else None,
                "villain": int(sel["villain"]) if sel.get("villain") is not None else None,
            }
            # `partial_anchor` short-circuit (closes the `partial-anchor-
            # crashes-analysis` round-1 concern). If SOME but not ALL
            # personas resolved an E*, we MUST refuse to compute headline
            # stats — the per-cell loader would splice ``None`` into the
            # filename and crash on a malformed legacy-shape path. Write
            # a clean ``headline_status=partial_anchor_skipped`` payload
            # instead and exit cleanly.
            _partial_flag = anchor_payload.get("partial_anchor", False)
            _unresolved_personas = [
                p for p, e in _ACTIVE["selected_epoch_per_persona"].items() if e is None
            ]
            if _partial_flag or _unresolved_personas:
                partial_payload = {
                    "schema_version": schema_version,
                    "generated_at": _dt.datetime.now(_dt.UTC).isoformat(),
                    "git_commit": _git_commit_hash(),
                    "variant": args.variant,
                    "headline_status": "partial_anchor_skipped",
                    "partial_anchor": True,
                    "partial_anchor_reason": anchor_payload.get(
                        "partial_anchor_reason",
                        f"unresolved={sorted(_unresolved_personas)}",
                    ),
                    "selected_anchor_per_persona": _ACTIVE["selected_epoch_per_persona"],
                    "anchor_file": str(args.anchor_file),
                    "note": (
                        "i464_po_analyze refused to compute headline statistics "
                        "because at least one persona did not resolve an anchor "
                        "epoch in the {1,2,3,5}-epoch grid. Re-run "
                        "`i529_select_anchor.py` after adding more epoch points "
                        "(e.g. {1,2,3,5,7,9}) or rerun training at a lower "
                        "LoRA rank / lr per "
                        "`.claude/rules/marker-training-recipe.md`."
                    ),
                }
                out_path_active.parent.mkdir(parents=True, exist_ok=True)
                out_path_active.write_text(json.dumps(partial_payload, indent=2))
                logger.warning(
                    "%s PARTIAL_ANCHOR — headline stats skipped, wrote %s",
                    args.variant,
                    out_path_active,
                )
                return
            logger.info(
                "%s selected anchor: %s",
                args.variant,
                _ACTIVE["selected_epoch_per_persona"],
            )
    logger.info(
        "variant=%s per_cell_dir=%s out_path=%s seeds=%s",
        args.variant,
        _ACTIVE["per_cell_dir"],
        out_path_active,
        args.seeds,
    )

    L_per_arm_per_seed: dict[str, dict[int, float]] = {arm: {} for arm in PO_ARMS}
    raw_per_cell: dict[str, dict[int, list[float]]] = {arm: {} for arm in PO_ARMS}
    own_logp_per_arm_per_seed: dict[str, dict[int, list[float]]] = {arm: {} for arm in PO_ARMS}
    own_cell_labels: list[str] = []
    missing: list[str] = []

    for seed in args.seeds:
        for arm in PO_ARMS:
            try:
                L, raw = _symmetric_leakage(arm, seed)
            except FileNotFoundError as e:
                if args.allow_partial:
                    logger.warning("analyze leakage (partial): %s", e)
                    missing.append(str(e))
                    continue
                raise
            L_per_arm_per_seed[arm][seed] = L
            raw_per_cell[arm][seed] = raw

            try:
                own_logps, labels = _own_persona_elicitation(arm, seed)
            except FileNotFoundError as e:
                if args.allow_partial:
                    logger.warning("analyze H1 (partial): %s", e)
                    missing.append(str(e))
                else:
                    raise
            else:
                own_logp_per_arm_per_seed[arm][seed] = own_logps
                if not own_cell_labels:
                    own_cell_labels = labels

    if missing and not args.allow_partial:
        raise RuntimeError(f"analyze: {len(missing)} missing per-cell JSONs")

    # ── H1 elicitation gate (per-cell pass map) ─────────────────────────
    h1_per_cell_pass: dict[str, bool] = {}
    h1_per_cell_logp: dict[str, float] = {}
    for arm, by_seed in own_logp_per_arm_per_seed.items():
        for seed, logps in by_seed.items():
            for persona_idx, persona in enumerate(enc.PERSONAS):
                e_own = _own_eval_encoding_for(arm, persona)  # type: ignore[arg-type]
                key = f"{_po_cell_label(arm, seed, persona)}/{e_own}"  # type: ignore[arg-type]
                lp = float(logps[persona_idx])
                h1_per_cell_logp[key] = lp
                h1_per_cell_pass[key] = lp >= H1_ELICITATION_THRESHOLD
    h1_overall_pass = bool(h1_per_cell_pass) and all(h1_per_cell_pass.values())

    # ── Leakage-to-default (per arm, NEW vs parent) ─────────────────────
    leakage_to_default: dict[str, dict] = {}
    try:
        for arm in PO_ARMS:
            logps, labels = _leakage_to_default(arm)
            arr = np.array(logps, dtype=float)
            leakage_to_default[arm] = {
                "per_cell_logp": logps,
                "per_cell_label": labels,
                "mean": float(arr.mean()),
                "sd": float(arr.std(ddof=0)),
                "n": len(logps),
            }
    except FileNotFoundError as e:
        if args.allow_partial:
            logger.warning("analyze leakage-to-default (partial): %s", e)
            leakage_to_default["partial"] = {"reason": str(e)}
        else:
            raise

    # ── Headline: paired deltas over COMPLETE seeds only ────────────────
    complete_seeds = sorted(
        set(L_per_arm_per_seed["system_plain"])
        & set(L_per_arm_per_seed["system_padded"])
        & set(L_per_arm_per_seed["role"])
    )
    d_plain: list[float] = []
    d_padded: list[float] = []
    for s in complete_seeds:
        d_plain.append(L_per_arm_per_seed["system_plain"][s] - L_per_arm_per_seed["role"][s])
        d_padded.append(L_per_arm_per_seed["system_padded"][s] - L_per_arm_per_seed["role"][s])

    headline: dict
    headline_status: str
    if len(complete_seeds) < H2_MIN_SEEDS:
        headline_status = "inconclusive_descriptive_only"
        headline = {
            "status": headline_status,
            "n_complete_seeds": len(complete_seeds),
            "min_seeds_required": H2_MIN_SEEDS,
            "reason": (
                f"only {len(complete_seeds)} complete paired seeds (need >= {H2_MIN_SEEDS})."
            ),
            "d_seed_plain_descriptive": d_plain,
            "d_seed_padded_descriptive": d_padded,
            "h2_full_pass": False,
            "h2_partial": False,
        }
    else:
        m_p, lo_p, hi_p = _paired_bootstrap_ci(d_plain, N_BOOTSTRAP)
        m_pad, lo_pad, hi_pad = _paired_bootstrap_ci(d_padded, N_BOOTSTRAP)
        verdict_plain = _h2_verdict("plain", d_plain, m_p, lo_p, hi_p)
        verdict_padded = _h2_verdict("padded", d_padded, m_pad, lo_pad, hi_pad)
        h2_full = verdict_plain["pass"] and verdict_padded["pass"] and h1_overall_pass
        h2_partial = verdict_plain["pass"] and not verdict_padded["pass"] and h1_overall_pass
        headline_status = "ok" if h2_full else ("partial" if h2_partial else "fail")
        headline = {
            "status": headline_status,
            "n_complete_seeds": len(complete_seeds),
            "complete_seeds": complete_seeds,
            "d_seed_plain": verdict_plain,
            "d_seed_padded": verdict_padded,
            "h2_full_pass": h2_full,
            "h2_partial": h2_partial,
            "h1_required_before_h2": True,
            "h1_overall_pass": h1_overall_pass,
            "n_bootstrap": N_BOOTSTRAP,
        }

    # ── Dynamic-range gate (mirrors parent's override-on-saturation) ────
    dr_gate, dynamic_range_ok = _compute_dynamic_range_gate(raw_per_cell)
    if not dynamic_range_ok and headline_status not in (
        "inconclusive_descriptive_only",
        "inconclusive_dynamic_range_failed",
    ):
        failing_arms = [a for a, v in dr_gate.items() if not v.get("above_threshold")]
        headline_status = "inconclusive_dynamic_range_failed"
        headline["status"] = headline_status
        headline["h2_full_pass"] = False
        headline["h2_partial"] = False
        headline["dynamic_range_failed_arms"] = failing_arms
        headline["reason"] = (
            f"Dynamic-range gate failed: arms with sd <= {DYNAMIC_RANGE_THRESHOLD}: "
            f"{failing_arms}. Saturated regime — leakage log-prob comparisons "
            "are rank-shuffles on a ceiling, not informative segmentation."
        )

    payload = {
        "schema_version": schema_version,
        "variant": args.variant,
        "git_commit": _git_commit_hash(),
        "generated_at": _dt.datetime.now(_dt.UTC).isoformat(),
        "seeds": args.seeds,
        "arms": list(PO_ARMS),
        "shared_marker_text": enc.MARKER_PIRATE_TEXT,
        "shared_marker_id": enc.MARKER_PIRATE_ID,
        "L_per_arm_per_seed": {arm: dict(d) for arm, d in L_per_arm_per_seed.items()},
        "complete_seeds": complete_seeds,
        "h2_min_seeds": H2_MIN_SEEDS,
        "h1_elicitation": {
            "threshold_nats": H1_ELICITATION_THRESHOLD,
            "per_cell_logp": h1_per_cell_logp,
            "per_cell_pass": h1_per_cell_pass,
            "overall_pass": h1_overall_pass,
            "n_cells": len(h1_per_cell_pass),
        },
        "leakage_to_default": leakage_to_default,
        "headline": headline,
        "headline_status": headline_status,
        "dynamic_range_gate": {
            "threshold": DYNAMIC_RANGE_THRESHOLD,
            "per_arm": dr_gate,
            "ok": dynamic_range_ok,
        },
        "raw_per_cell": raw_per_cell,
        "n_missing_per_cell": len(missing),
    }
    if args.variant in ("cn_i529", "cn_i533"):
        payload["selected_anchor"] = _ACTIVE.get("selected_epoch_per_persona")
        payload["anchor_file"] = args.anchor_file

    # cn_i533 ONLY: per-persona paired-bootstrap extension (plan §4 (d)
    # "Pseudocode for new code"). The inherited cn_i529 path's
    # _symmetric_leakage averages personas BEFORE forming d, so the
    # persona-averaged headline.d_seed_plain / headline.d_seed_padded
    # can hide an opposite-signed per-persona pair. The H1/H0 verdict
    # at the selected anchor is driven from the 4 per-persona cells
    # added here (pirate x plain, pirate x padded, villain x plain,
    # villain x padded). The persona-averaged keys stay as a SECONDARY
    # cross-check; this block does NOT modify them. Skipped when the
    # headline is in a non-stat status (descriptive-only / dynamic-
    # range failed) — the per-persona view is only meaningful when the
    # variant-level bootstrap is also meaningful.
    if args.variant == "cn_i533" and headline_status not in (
        "inconclusive_descriptive_only",
        "inconclusive_dynamic_range_failed",
    ):
        try:
            per_persona = _per_persona_paired_d_block(seeds=tuple(args.seeds), n_boot=N_BOOTSTRAP)
            payload["headline"]["per_persona"] = per_persona
            for persona_key, by_contrast in per_persona.items():
                for contrast_key, stats in by_contrast.items():
                    logger.info(
                        "cn_i533 per-persona d[%s][%s]: mean=%.3f CI=[%.3f, %.3f] "
                        "sign_agreement=%.3f",
                        persona_key,
                        contrast_key,
                        stats["mean"],
                        stats["ci_lo_95"],
                        stats["ci_hi_95"],
                        stats["sign_agreement"],
                    )
            # cn_i533 ONLY: route headline_status through the 4 per-persona
            # cells (round-2 reconciler binding finding). The inherited
            # persona-averaged H2 rule at lines ~706-708 cannot drive the
            # H1/H0 verdict for cn_i533 because _symmetric_leakage averages
            # personas before forming d, so the persona-averaged read can
            # hide an opposite-signed per-persona pair. Plan §6 thresholds:
            #   H1 (positive resolution): >=1 of 4 cells has ci_lo > 0
            #       AND mean >= H1_PER_PERSONA_THRESHOLD nat
            #   H0 (null resolution): ALL 4 cells straddle zero
            #       (ci_lo <= 0 <= ci_hi)
            # Either resolves the experiment's question; headline_status
            # maps to "ok" in both cases. "inconclusive" = neither H1 nor
            # H0 fits (e.g. one cell positive but below 0.5 nat AND another
            # straddling zero AND another non-straddling negative) → maps
            # to "partial". The cn_i529 path is BYTE-STABLE — none of this
            # routing applies there.
            per_persona_verdict, h1_pp_pass, h0_pp_pass = _headline_verdict_from_per_persona(
                per_persona
            )
            payload["headline"]["per_persona_verdict"] = per_persona_verdict
            payload["headline"]["h1_per_persona_pass"] = h1_pp_pass
            payload["headline"]["h0_per_persona_pass"] = h0_pp_pass
            if per_persona_verdict == "h1" or per_persona_verdict == "h0":
                headline_status = "ok"
            else:
                headline_status = "partial"
            payload["headline"]["status"] = headline_status
            payload["headline_status"] = headline_status
            logger.info(
                "cn_i533 per-persona verdict: %s (h1_pass=%s h0_pass=%s) -> headline_status=%s",
                per_persona_verdict,
                h1_pp_pass,
                h0_pp_pass,
                headline_status,
            )
        except FileNotFoundError as e:
            if args.allow_partial:
                logger.warning("cn_i533 per-persona paired-d (partial): %s", e)
                payload["headline"]["per_persona_status"] = "partial_missing_cells"
                payload["headline"]["per_persona_partial_reason"] = str(e)
            else:
                raise
    out_path_active.parent.mkdir(parents=True, exist_ok=True)
    out_path_active.write_text(json.dumps(payload, indent=2))
    logger.info(
        "%s analyze done -> %s (status=%s complete_seeds=%d H1=%s)",
        args.variant,
        out_path_active,
        headline_status,
        len(complete_seeds),
        h1_overall_pass,
    )
    if headline_status == "ok":
        logger.info(
            "H2 PASS: d_plain mean=%.3f CI=[%.3f, %.3f]; d_padded mean=%.3f CI=[%.3f, %.3f]",
            headline["d_seed_plain"]["mean"],
            headline["d_seed_plain"]["ci_lo_95"],
            headline["d_seed_plain"]["ci_hi_95"],
            headline["d_seed_padded"]["mean"],
            headline["d_seed_padded"]["ci_lo_95"],
            headline["d_seed_padded"]["ci_hi_95"],
        )
    elif headline_status == "inconclusive_descriptive_only":
        logger.warning(
            "H2 INCONCLUSIVE: only %d complete paired seed(s); need >= %d",
            len(complete_seeds),
            H2_MIN_SEEDS,
        )
    elif headline_status == "inconclusive_dynamic_range_failed":
        logger.warning(
            "H2 INCONCLUSIVE (dynamic-range failed): leakage log-prob sd "
            "<= %.2f in arm(s) %s — saturation regime, headline overridden.",
            DYNAMIC_RANGE_THRESHOLD,
            headline.get("dynamic_range_failed_arms"),
        )
    if not h1_overall_pass and h1_per_cell_pass:
        failing = [k for k, v in h1_per_cell_pass.items() if not v]
        logger.warning(
            "H1 elicitation FAILED on %d of %d cells (own log P < %.1f nat): %s",
            len(failing),
            len(h1_per_cell_pass),
            H1_ELICITATION_THRESHOLD,
            failing[:5],
        )
    # Leakage-to-default summary (NEW vs parent #464).
    for arm in PO_ARMS:
        d = leakage_to_default.get(arm)
        if d and "mean" in d:
            logger.info(
                "leakage-to-default arm=%s mean=%.3f sd=%.3f n=%d",
                arm,
                d["mean"],
                d["sd"],
                d["n"],
            )


if __name__ == "__main__":
    main()
