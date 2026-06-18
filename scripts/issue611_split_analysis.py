# ruff: noqa: RUF002, RUF003
# Intentional Unicode (※, Δ, − minus) in scientific docstrings + logs.
"""Issue #611 — probe-split re-analysis of #533's bare-word install-step grid.

Zero-GPU, CPU-only re-analysis of the parent's committed four-float logit
captures. Tests whether the bare role header's lower wrong-persona leakage
and higher default-assistant leakage (the probe split in #533) survives
(a) an unsaturated-implant read at checkpoints {s18, s30} and (b) the
base-prior-clean EOS-margin readout Δ(z_marker − z_eos).

Phases (plan #611 §4):

* Phase 0 — input validation (``--validate-only`` stops here; full mode
  runs it first). 245-file enumeration, schema/marker/eos/gauge asserts,
  encoding-identical default-probe base identity (atol 1e-4) incl. the
  standalone ``base__*`` cross-check (arrays under the ``stats`` key),
  rig-consistency vs ``cross_eval/per_cell`` ``delta_g`` at 1.0-nat
  tolerance. Fail-loud: any violation raises.
* Phase 1 — per-cell Δlog P + Δmargin, paired contrasts
  d = (minimal system) − (bare role) with per-seed-paired bootstrap
  (N=10,000, 95% percentile CI, rng seed 0) in BOTH spaces, band
  accounting (incl. own-slot Δmargin), base-prior decomposition,
  saturation flags (|d_logp − d_margin| > 0.5 nat), exploratory
  leakage-allocation read, and the exhaustive §7 verdict block.

Inputs (committed on ``main``; parent #533):
    eval_results/issue_533/bare_word_install_step_grid/logit_capture/
        per_cell/{arm}_seed{S}_cn_{persona}_s{steps}__{e_eval}__marker_pirate.json
        per_cell/base__{e_eval}__marker_pirate.json    (arrays under ``stats``)
    eval_results/issue_533/bare_word_install_step_grid/cross_eval/per_cell/
        {arm}_seed{S}_cn_{persona}_s{steps}__{e_eval}.json   (rig check + emit fallback)
    eval_results/issue_533/bare_word_install_step_grid/analysis.json
        (own argmax-emit rates, ``own_emit_rate_{sys,role}_mean``)

Output (single JSON, schema ``i611_split_analysis_v1``):
    eval_results/issue_611/split_analysis.json

CLI::

    uv run python scripts/issue611_split_analysis.py --validate-only
    uv run python scripts/issue611_split_analysis.py
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger("issue611_split_analysis")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GRID_DIR = PROJECT_ROOT / "eval_results" / "issue_533" / "bare_word_install_step_grid"
CAPTURE_DIR = GRID_DIR / "logit_capture" / "per_cell"
CROSS_EVAL_DIR = GRID_DIR / "cross_eval" / "per_cell"
PARENT_ANALYSIS_PATH = GRID_DIR / "analysis.json"
OUT_PATH = PROJECT_ROOT / "eval_results" / "issue_611" / "split_analysis.json"

SCHEMA_VERSION = "i611_split_analysis_v1"
CAPTURE_SCHEMA = "i533_bw_logit_capture_v1"

# --- Inherited grid constants (plan §10; fixed by reuse, not free parameters) ---
ARMS = ("system_minimal", "role_bare")
PERSONAS = ("pirate", "villain")
STEPS = (18, 30, 60, 120)
SEEDS = (7, 21, 42, 137, 1337)
N_PROBES = 50
MARKER_ID = 83399
EOS_ID = 151645
SHARED_MARKER_PERSONA = "pirate"  # probe-set label; the marker is shared ` ※` id 83399
FLOAT_KEYS = ("logp", "z_marker", "z_eos", "logZ")

# --- Analysis parameters (plan §11) ---
N_BOOT = 10_000
RNG_SEED = 0  # plan §11 pins rng seed 0 for this task's bootstrap
BAND_LO, BAND_HI = 5.0, 12.0  # marker-training-recipe usable band, nats
SATURATION_FLAG_NATS = 0.5  # |d_logp − d_margin| above this ⇒ margin authoritative
RIG_TOLERANCE_NATS = 1.0  # capture-vs-cross_eval delta_g agreement (#533 reread)
BASE_IDENTITY_ATOL = 1e-4  # encoding-identical default-probe base check

PROBE_KINDS = ("own", "wrong", "default")
SPACES = ("logp", "margin")
OTHER = {"pirate": "villain", "villain": "pirate"}

# Expected sign of d = (minimal system) − (bare role), per half of the split.
# wrong-persona half: role leaks LESS ⇒ d > 0 expected.
# default-assistant half: role leaks MORE ⇒ d < 0 expected.
EXPECTED_SIGN = {"wrong": +1, "default": -1}


def _encoding(arm: str, persona: str) -> str:
    """Eval-encoding slug for ``arm``'s own way of naming ``persona``."""
    if arm not in ARMS:
        raise ValueError(f"unknown arm={arm!r}")
    return f"{arm}_{persona}"


def _probe_encoding(arm: str, persona: str, probe_kind: str) -> str:
    """Resolve a probe kind to the eval encoding (plan §4 probe resolution)."""
    if probe_kind == "own":
        return _encoding(arm, persona)
    if probe_kind == "wrong":
        return _encoding(arm, OTHER[persona])
    if probe_kind == "default":
        return "default_assistant"
    raise ValueError(f"unknown probe_kind={probe_kind!r}")


def _cell_label(arm: str, seed: int, persona: str, steps: int) -> str:
    return f"{arm}_seed{seed}_cn_{persona}_s{steps}"


def _capture_path(arm: str, seed: int, persona: str, steps: int, e_eval: str) -> Path:
    name = f"{_cell_label(arm, seed, persona, steps)}__{e_eval}__marker_{SHARED_MARKER_PERSONA}"
    return CAPTURE_DIR / f"{name}.json"


def _base_capture_path(e_eval: str) -> Path:
    return CAPTURE_DIR / f"base__{e_eval}__marker_{SHARED_MARKER_PERSONA}.json"


def _cross_eval_path(arm: str, seed: int, persona: str, steps: int, e_eval: str) -> Path:
    return CROSS_EVAL_DIR / f"{_cell_label(arm, seed, persona, steps)}__{e_eval}.json"


def _git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


# ---------------------------------------------------------------------------
# Phase 0 — validation (fail-loud)
# ---------------------------------------------------------------------------


def _assert_four_floats(d: dict, n: int, where: str) -> None:
    for k in FLOAT_KEYS:
        if k not in d:
            raise AssertionError(f"{where}: missing float array {k!r}")
        if len(d[k]) != n:
            raise AssertionError(f"{where}: len({k})={len(d[k])} != {n}")


def validate_inputs() -> dict:  # noqa: C901 - flat per-(arm,persona,step,seed,probe) assert sweep
    """Phase 0: enumerate + assert every input; return a validation summary.

    Any violation raises (fail-loud, plan §7 kill criterion) — downstream
    phases never see contaminated inputs.
    """
    log.info("[phase=validate] enumerating %s", CAPTURE_DIR)
    all_files = sorted(CAPTURE_DIR.glob("*.json"))
    n_total = len(all_files)
    if n_total != 245:
        raise AssertionError(f"expected 245 per-cell capture JSONs, found {n_total}")

    expected_encodings = sorted(
        {_encoding(a, p) for a in ARMS for p in PERSONAS} | {"default_assistant"}
    )
    base_files = sorted(CAPTURE_DIR.glob("base__*.json"))
    if len(base_files) != 5:
        raise AssertionError(f"expected 5 base__* capture JSONs, found {len(base_files)}")

    # --- standalone base files (arrays under the `stats` key) ---
    base_stats: dict[str, dict] = {}
    for e_eval in expected_encodings:
        p = _base_capture_path(e_eval)
        if not p.exists():
            raise AssertionError(f"missing base capture for encoding {e_eval!r}: {p}")
        j = json.loads(p.read_text())
        if j["schema_version"] != CAPTURE_SCHEMA:
            raise AssertionError(f"{p.name}: schema_version={j['schema_version']!r}")
        if int(j["marker_id"]) != MARKER_ID:
            raise AssertionError(f"{p.name}: marker_id={j['marker_id']}")
        if int(j["eos_id"]) != EOS_ID:
            raise AssertionError(f"{p.name}: eos_id={j['eos_id']}")
        if "stats" not in j:
            raise AssertionError(f"{p.name}: standalone base file missing `stats` key")
        _assert_four_floats(j["stats"], N_PROBES, f"{p.name}.stats")
        base_stats[e_eval] = j["stats"]

    # --- 240 trained cells: existence + per-file asserts ---
    n_trained = 0
    default_base_logps: list[tuple[str, np.ndarray]] = []
    rig_diffs: list[float] = []
    rig_worst: tuple[float, str] = (0.0, "")
    for arm in ARMS:
        for persona in PERSONAS:
            for steps in STEPS:
                for seed in SEEDS:
                    for probe_kind in PROBE_KINDS:
                        e_eval = _probe_encoding(arm, persona, probe_kind)
                        p = _capture_path(arm, seed, persona, steps, e_eval)
                        if not p.exists():
                            raise AssertionError(f"missing trained capture: {p}")
                        j = json.loads(p.read_text())
                        if j["schema_version"] != CAPTURE_SCHEMA:
                            raise AssertionError(f"{p.name}: schema={j['schema_version']!r}")
                        if int(j["marker_id"]) != MARKER_ID:
                            raise AssertionError(f"{p.name}: marker_id={j['marker_id']}")
                        if int(j["eos_id"]) != EOS_ID:
                            raise AssertionError(f"{p.name}: eos_id={j['eos_id']}")
                        ga = j.get("gauge_assert")
                        if not (isinstance(ga, dict) and ga.get("ok") is True):
                            raise AssertionError(f"{p.name}: gauge_assert not ok: {ga!r}")
                        for side in ("trained", "base"):
                            if side not in j:
                                raise AssertionError(f"{p.name}: missing {side!r} side")
                            _assert_four_floats(j[side], N_PROBES, f"{p.name}.{side}")
                        if int(j["seed"]) != seed or int(j["max_steps"]) != steps:
                            raise AssertionError(
                                f"{p.name}: metadata mismatch seed={j['seed']} "
                                f"max_steps={j['max_steps']}"
                            )
                        if j["arm"] != arm or j["e_eval"] != e_eval:
                            raise AssertionError(
                                f"{p.name}: metadata mismatch arm={j['arm']!r} "
                                f"e_eval={j['e_eval']!r}"
                            )
                        n_trained += 1

                        if probe_kind == "default":
                            default_base_logps.append(
                                (p.name, np.asarray(j["base"]["logp"], dtype=float))
                            )

                        # Embedded base must duplicate the standalone base file
                        # for the SAME encoding (plan §12 assumption, re-asserted
                        # here for every encoding, not just default).
                        for k in FLOAT_KEYS:
                            emb = np.asarray(j["base"][k], dtype=float)
                            stand = np.asarray(base_stats[e_eval][k], dtype=float)
                            if not np.allclose(emb, stand, atol=BASE_IDENTITY_ATOL):
                                raise AssertionError(
                                    f"{p.name}: embedded base.{k} diverges from "
                                    f"standalone base__{e_eval} stats.{k}"
                                )

                        # Rig-consistency: capture mean Δlog P vs cross_eval delta_g.
                        ce_path = _cross_eval_path(arm, seed, persona, steps, e_eval)
                        if not ce_path.exists():
                            raise AssertionError(f"missing cross_eval row: {ce_path}")
                        ce = json.loads(ce_path.read_text())
                        if (
                            ce["arm"] != arm
                            or int(ce["seed"]) != seed
                            or int(ce["max_steps"]) != steps
                            or ce["training_persona"] != persona
                            or ce["e_eval"] != e_eval
                        ):
                            raise AssertionError(f"{ce_path.name}: metadata mismatch: {ce}")
                        cap_d = float(
                            np.mean(
                                np.asarray(j["trained"]["logp"], dtype=float)
                                - np.asarray(j["base"]["logp"], dtype=float)
                            )
                        )
                        diff = abs(cap_d - float(ce["delta_g"]))
                        rig_diffs.append(diff)
                        if diff > rig_worst[0]:
                            rig_worst = (diff, p.name)
                        if diff > RIG_TOLERANCE_NATS:
                            raise AssertionError(
                                f"rig-consistency FAIL: {p.name} capture Δlog P {cap_d:.4f} "
                                f"vs cross_eval delta_g {ce['delta_g']:.4f} "
                                f"(|diff|={diff:.4f} > {RIG_TOLERANCE_NATS})"
                            )

    if n_trained != 240:
        raise AssertionError(f"expected 240 trained captures, asserted {n_trained}")

    # --- encoding-identical control: default-probe base identical everywhere ---
    ref_name, ref = default_base_logps[0]
    for name, arr in default_base_logps[1:]:
        if not np.allclose(arr, ref, atol=BASE_IDENTITY_ATOL):
            raise AssertionError(
                f"default-assistant base logp differs across cells: {name} vs {ref_name}"
            )
    stand_default = np.asarray(base_stats["default_assistant"]["logp"], dtype=float)
    if not np.allclose(ref, stand_default, atol=BASE_IDENTITY_ATOL):
        raise AssertionError(
            "default-assistant embedded base logp differs from standalone "
            "base__default_assistant stats.logp"
        )

    summary = {
        "n_files_total": n_total,
        "n_trained": n_trained,
        "n_base": len(base_files),
        "encodings": expected_encodings,
        "default_base_logp_mean": float(ref.mean()),
        "default_base_identity_atol": BASE_IDENTITY_ATOL,
        "rig_check": {
            "tolerance_nats": RIG_TOLERANCE_NATS,
            "n_cells": len(rig_diffs),
            "mae_nats": float(np.mean(rig_diffs)),
            "max_abs_diff_nats": rig_worst[0],
            "max_abs_diff_cell": rig_worst[1],
            "n_failures": 0,
        },
    }
    log.info(
        "[phase=validate] OK — 245 files (240 trained + 5 base); default base "
        "logp mean %.3f; rig MAE %.4f nat (max %.4f on %s)",
        summary["default_base_logp_mean"],
        summary["rig_check"]["mae_nats"],
        rig_worst[0],
        rig_worst[1],
    )
    return summary


# ---------------------------------------------------------------------------
# Phase 1 — contrasts, band accounting, decomposition, verdicts
# ---------------------------------------------------------------------------


def _cell_means(arm: str, seed: int, persona: str, steps: int, probe_kind: str) -> dict:
    """Per-cell means over the 50 probes, in all readout spaces.

    Returns dlogp, dmargin (trained − base), plus base/trained levels and the
    probability-space sanity read ΔP = mean(P_trained − P_base).
    """
    e_eval = _probe_encoding(arm, persona, probe_kind)
    j = json.loads(_capture_path(arm, seed, persona, steps, e_eval).read_text())
    t, b = j["trained"], j["base"]
    t_logp = np.asarray(t["logp"], dtype=float)
    b_logp = np.asarray(b["logp"], dtype=float)
    t_margin = np.asarray(t["z_marker"], dtype=float) - np.asarray(t["z_eos"], dtype=float)
    b_margin = np.asarray(b["z_marker"], dtype=float) - np.asarray(b["z_eos"], dtype=float)
    return {
        "dlogp": float(np.mean(t_logp - b_logp)),
        "dmargin": float(np.mean(t_margin - b_margin)),
        "base_logp": float(np.mean(b_logp)),
        "trained_logp": float(np.mean(t_logp)),
        "base_margin": float(np.mean(b_margin)),
        "trained_margin": float(np.mean(t_margin)),
        "delta_p": float(np.mean(np.exp(t_logp) - np.exp(b_logp))),
    }


def _paired_bootstrap(per_seed_d: list[float], rng_seed: int = RNG_SEED) -> dict:
    """Per-seed-paired bootstrap of mean(d) — N=10,000, 95% percentile CI.

    Statistic shape identical to #533's ``_paired_bootstrap`` (per-seed cell
    means differenced within seed, then seeds resampled with replacement);
    rng seed pinned to 0 per plan §11.
    """
    arr = np.asarray(per_seed_d, dtype=float)
    if len(arr) != len(SEEDS):
        raise AssertionError(f"expected {len(SEEDS)} per-seed values, got {len(arr)}")
    rng = np.random.default_rng(rng_seed)
    n = len(arr)
    idx = rng.integers(0, n, size=(N_BOOT, n))
    boots = arr[idx].mean(axis=1)
    return {
        "point": float(arr.mean()),
        "ci_lo": float(np.percentile(boots, 2.5)),
        "ci_hi": float(np.percentile(boots, 97.5)),
        "n_seeds": n,
        "n_boot": N_BOOT,
    }


def _own_emit_rates() -> dict[tuple[str, str, int], float]:
    """Own argmax-emit rate per (arm, persona, steps).

    Primary source: parent ``analysis.json`` ``paired_results`` rows
    (``own_emit_rate_{sys,role}_mean``). Fallback (plan §4): recompute the
    SAME estimator from ``cross_eval/per_cell`` by filtering own-probe files
    and averaging ``g_argmax_marker_per_q``, asserting metadata on every
    constructed join — never filename-count joins.
    """
    arm_field = {"system_minimal": "own_emit_rate_sys_mean", "role_bare": "own_emit_rate_role_mean"}
    rates: dict[tuple[str, str, int], float] = {}
    parent = json.loads(PARENT_ANALYSIS_PATH.read_text())
    rows = parent.get("paired_results", [])
    by_cell = {(r.get("persona"), int(r.get("max_steps", -1))): r for r in rows}
    for arm in ARMS:
        for persona in PERSONAS:
            for steps in STEPS:
                row = by_cell.get((persona, steps))
                val = row.get(arm_field[arm]) if row is not None else None
                if val is not None:
                    rates[(arm, persona, steps)] = float(val)
                    continue
                # Fallback recompute — same estimator, metadata-asserted join.
                log.warning(
                    "[phase=analyze] emit rate missing in parent analysis.json for "
                    "(%s, %s, s%d) — recomputing from cross_eval",
                    arm,
                    persona,
                    steps,
                )
                per_seed: list[float] = []
                for seed in SEEDS:
                    e_eval = _probe_encoding(arm, persona, "own")
                    ce = json.loads(_cross_eval_path(arm, seed, persona, steps, e_eval).read_text())
                    if (
                        ce["arm"] != arm
                        or int(ce["seed"]) != seed
                        or int(ce["max_steps"]) != steps
                        or ce["training_persona"] != persona
                        or ce["e_eval"] != e_eval
                    ):
                        raise AssertionError(f"emit-rate fallback metadata mismatch: {ce}")
                    flags = ce["g_argmax_marker_per_q"]
                    if not flags:
                        raise AssertionError(
                            f"empty g_argmax_marker_per_q for {arm} seed{seed} {persona} s{steps}"
                        )
                    per_seed.append(float(sum(flags) / len(flags)))
                rates[(arm, persona, steps)] = float(np.mean(per_seed))
    return rates


# --- §7 verdict machinery -------------------------------------------------


def _ci_state(point: float, ci_lo: float, ci_hi: float, expected_sign: int) -> str:
    """Per-persona CI state: E (expected-direction clear), S (straddles), O (opposite)."""
    if ci_lo > 0 and ci_hi > 0:
        clear_sign = +1
    elif ci_lo < 0 and ci_hi < 0:
        clear_sign = -1
    else:
        return "S"
    return "E" if clear_sign == expected_sign else "O"


# Total function over the unordered persona-state pair (plan §7) — every
# combination of the 3×3 grid is mapped; anything else raises.
_CHECKPOINT_LABEL = {
    frozenset({"E"}): "SURVIVES",  # (E,E)
    frozenset({"E", "S"}): "PARTIAL",
    frozenset({"E", "O"}): "DISCORDANT",
    frozenset({"S"}): "ABSENT",  # (S,S)
    frozenset({"S", "O"}): "REVERSED",
    frozenset({"O"}): "REVERSED",  # (O,O)
}


def _checkpoint_label(states: dict[str, str]) -> str:
    key = frozenset(states.values())
    if key not in _CHECKPOINT_LABEL:
        raise AssertionError(f"unmapped persona CI-state combination: {states!r}")
    return _CHECKPOINT_LABEL[key]


_VANISH_LABELS = {"ABSENT", "REVERSED", "DISCORDANT"}


def _regime_verdict(labels: dict[int, str]) -> str:
    """Aggregate checkpoint labels into survives / vanishes / shrinks (plan §7a/§7b)."""
    vals = list(labels.values())
    if all(v == "SURVIVES" for v in vals):
        return "survives"
    if all(v in _VANISH_LABELS for v in vals):
        return "vanishes"
    return "shrinks"


def analyze() -> dict:  # noqa: C901 - sequential phase blocks (contrasts/band/decomp/verdicts)
    """Run Phase 0 + Phase 1 and return the full result payload."""
    validation = validate_inputs()

    log.info("[phase=analyze] computing per-cell means for all 240 trained cells")
    # cells[(arm, persona, steps, probe_kind)][seed] = per-cell means dict
    cells: dict[tuple[str, str, int, str], dict[int, dict]] = {}
    for arm in ARMS:
        for persona in PERSONAS:
            for steps in STEPS:
                for probe_kind in PROBE_KINDS:
                    key = (arm, persona, steps, probe_kind)
                    cells[key] = {
                        seed: _cell_means(arm, seed, persona, steps, probe_kind) for seed in SEEDS
                    }

    # --- paired contrasts: 2 probes × 2 personas × 4 steps × 2 spaces = 32 rows ---
    log.info("[phase=analyze] paired contrasts + bootstrap (N=%d, rng seed %d)", N_BOOT, RNG_SEED)
    paired_contrasts: list[dict] = []
    # contrast_lookup[(probe, persona, steps, space)] = row (for verdicts/figures)
    contrast_lookup: dict[tuple[str, str, int, str], dict] = {}
    for probe_kind in ("wrong", "default"):
        for persona in PERSONAS:
            for steps in STEPS:
                d_by_space: dict[str, list[float]] = {"logp": [], "margin": []}
                for seed in SEEDS:
                    s = cells[("system_minimal", persona, steps, probe_kind)][seed]
                    r = cells[("role_bare", persona, steps, probe_kind)][seed]
                    d_by_space["logp"].append(s["dlogp"] - r["dlogp"])
                    d_by_space["margin"].append(s["dmargin"] - r["dmargin"])
                point = {sp: float(np.mean(d_by_space[sp])) for sp in SPACES}
                sat_flag = abs(point["logp"] - point["margin"]) > SATURATION_FLAG_NATS
                for space in SPACES:
                    boot = _paired_bootstrap(d_by_space[space])
                    per_seed = d_by_space[space]
                    per_seed_d = dict(zip(map(str, SEEDS), map(float, per_seed), strict=True))
                    row = {
                        "probe_kind": probe_kind,
                        "persona": persona,
                        "max_steps": steps,
                        "space": space,
                        **boot,
                        "per_seed_d": per_seed_d,
                        "sign_tally_positive": int(sum(1 for v in per_seed if v > 0)),
                        "expected_sign": EXPECTED_SIGN[probe_kind],
                        "saturation_compressed": bool(sat_flag),
                        "authoritative_space": "margin" if sat_flag else "both",
                    }
                    paired_contrasts.append(row)
                    contrast_lookup[(probe_kind, persona, steps, space)] = row

    # --- band accounting: 2 arms × 2 personas × 4 steps = 16 rows ---
    log.info("[phase=analyze] own-slot band accounting")
    emit_rates = _own_emit_rates()
    band_accounting: list[dict] = []
    for arm in ARMS:
        for persona in PERSONAS:
            for steps in STEPS:
                own = cells[(arm, persona, steps, "own")]
                dlogp = float(np.mean([own[s]["dlogp"] for s in SEEDS]))
                dmargin = float(np.mean([own[s]["dmargin"] for s in SEEDS]))
                trained_logp = float(np.mean([own[s]["trained_logp"] for s in SEEDS]))
                band_accounting.append(
                    {
                        "arm": arm,
                        "persona": persona,
                        "max_steps": steps,
                        "own_dlogp_mean": dlogp,
                        "own_dmargin_mean": dmargin,
                        "own_trained_logp_mean": trained_logp,
                        "in_band": bool(BAND_LO <= dlogp <= BAND_HI),
                        "own_emit_rate": emit_rates[(arm, persona, steps)],
                    }
                )

    # --- base-prior decomposition: 2 arms × 3 probes × 2 personas × 4 steps = 48 rows ---
    log.info("[phase=analyze] base-prior decomposition")
    decomposition: list[dict] = []
    for arm in ARMS:
        for probe_kind in PROBE_KINDS:
            for persona in PERSONAS:
                for steps in STEPS:
                    cc = cells[(arm, persona, steps, probe_kind)]
                    decomposition.append(
                        {
                            "arm": arm,
                            "probe_kind": probe_kind,
                            "persona": persona,
                            "max_steps": steps,
                            "e_eval": _probe_encoding(arm, persona, probe_kind),
                            "base_logp_mean": float(np.mean([cc[s]["base_logp"] for s in SEEDS])),
                            "trained_logp_mean": float(
                                np.mean([cc[s]["trained_logp"] for s in SEEDS])
                            ),
                            "dlogp_mean": float(np.mean([cc[s]["dlogp"] for s in SEEDS])),
                            "dmargin_mean": float(np.mean([cc[s]["dmargin"] for s in SEEDS])),
                            "delta_p_mean": float(np.mean([cc[s]["delta_p"] for s in SEEDS])),
                        }
                    )

    # --- exploratory leakage allocation (descriptive only, never gated) ---
    allocation: list[dict] = []
    for arm in ARMS:
        for persona in PERSONAS:
            for steps in STEPS:
                row: dict = {"arm": arm, "persona": persona, "max_steps": steps}
                for space, field in (("logp", "dlogp"), ("margin", "dmargin")):
                    w = float(
                        np.mean([cells[(arm, persona, steps, "wrong")][s][field] for s in SEEDS])
                    )
                    d = float(
                        np.mean([cells[(arm, persona, steps, "default")][s][field] for s in SEEDS])
                    )
                    row[f"wrong_{space}"] = w
                    row[f"default_{space}"] = d
                    row[f"sum_{space}"] = w + d
                allocation.append(row)

    # --- §7 verdict block (exhaustive classifier, pinned quantifiers) ---
    log.info("[phase=analyze] verdict classification")
    halves: dict[str, dict] = {}
    for probe_kind in ("wrong", "default"):
        exp = EXPECTED_SIGN[probe_kind]
        # checkpoint labels per space, with per-persona CI states persisted
        checkpoint_detail: dict[str, dict[int, dict]] = {sp: {} for sp in SPACES}
        for space in SPACES:
            for steps in STEPS:
                states = {}
                for persona in PERSONAS:
                    r = contrast_lookup[(probe_kind, persona, steps, space)]
                    states[persona] = _ci_state(r["point"], r["ci_lo"], r["ci_hi"], exp)
                checkpoint_detail[space][steps] = {
                    "persona_ci_states": states,
                    "label": _checkpoint_label(states),
                }
        # (a) off-saturation verdict: log-p space at {18, 30}
        a_labels = {s: checkpoint_detail["logp"][s]["label"] for s in (18, 30)}
        a_verdict = _regime_verdict(a_labels)
        # (b) margin-space verdict: install-onward {30, 60, 120}
        b_labels = {s: checkpoint_detail["margin"][s]["label"] for s in (30, 60, 120)}
        b_verdict = _regime_verdict(b_labels)
        halves[probe_kind] = {
            "expected_sign": exp,
            "checkpoints": {sp: {str(s): checkpoint_detail[sp][s] for s in STEPS} for sp in SPACES},
            "off_saturation_verdict": a_verdict,
            "off_saturation_labels": {str(s): v for s, v in a_labels.items()},
            "margin_space_verdict": b_verdict,
            "margin_space_labels": {str(s): v for s, v in b_labels.items()},
            "s18_margin_bracket_label": checkpoint_detail["margin"][18]["label"],
        }

    # Headline (Goal) verdict — quantifiers pinned per plan §7:
    # concentration: default-half SURVIVES at >=1 of {s18,s30} in log-p space
    #   AND default-half margin verdict == survives
    #   AND wrong-half margin verdict == survives.
    # containment: default-half vanishes under BOTH clean reads ((a) and (b);
    #   the "vanishes" aggregate already covers ABSENT/REVERSED/DISCORDANT at
    #   every checkpoint in the regime) AND wrong-half margin verdict == survives.
    # anything else -> mixed (explicit branch — part of the §7 spec, not a
    # silent else).
    default_half = halves["default"]
    wrong_half = halves["wrong"]
    default_survives_one_unsat = any(
        default_half["off_saturation_labels"][str(s)] == "SURVIVES" for s in (18, 30)
    )
    if (
        default_survives_one_unsat
        and default_half["margin_space_verdict"] == "survives"
        and wrong_half["margin_space_verdict"] == "survives"
    ):
        headline = "concentration_confirmed"
    elif (
        default_half["off_saturation_verdict"] == "vanishes"
        and default_half["margin_space_verdict"] == "vanishes"
        and wrong_half["margin_space_verdict"] == "survives"
    ):
        headline = "containment_confirmed"
    else:
        headline = "mixed"

    verdicts = {
        "classifier": {
            "ci_states": "E = CI clear of zero in expected direction; S = straddles; "
            "O = CI clear opposite",
            "checkpoint_labels": "(E,E)=SURVIVES (E,S)=PARTIAL (E,O)=DISCORDANT "
            "(S,S)=ABSENT (S,O)|(O,O)=REVERSED",
            "off_saturation_regime": "log-p space at checkpoints {18, 30}",
            "margin_space_regime": "margin space at install-onward checkpoints {30, 60, 120}",
        },
        "wrong_persona_half": wrong_half,
        "default_assistant_half": default_half,
        "headline": headline,
    }

    payload = {
        "schema_version": SCHEMA_VERSION,
        "task_id": 611,
        "parent_task_id": 533,
        "generated_at": datetime.now(UTC).isoformat(),
        "git_commit": _git_commit(),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "inputs": {
            "capture_dir": str(CAPTURE_DIR.relative_to(PROJECT_ROOT)),
            "cross_eval_dir": str(CROSS_EVAL_DIR.relative_to(PROJECT_ROOT)),
            "parent_analysis": str(PARENT_ANALYSIS_PATH.relative_to(PROJECT_ROOT)),
        },
        "parameters": {
            "arms": list(ARMS),
            "personas": list(PERSONAS),
            "max_steps_grid": list(STEPS),
            "seeds": list(SEEDS),
            "n_probes": N_PROBES,
            "marker_id": MARKER_ID,
            "eos_id": EOS_ID,
            "n_boot": N_BOOT,
            "bootstrap_rng_seed": RNG_SEED,
            "ci_level": 0.95,
            "band_nats": [BAND_LO, BAND_HI],
            "saturation_flag_nats": SATURATION_FLAG_NATS,
            "rig_tolerance_nats": RIG_TOLERANCE_NATS,
        },
        "validation": validation,
        "paired_contrasts": paired_contrasts,
        "band_accounting": band_accounting,
        "decomposition": decomposition,
        "leakage_allocation_exploratory": allocation,
        "verdicts": verdicts,
    }
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Issue #611 probe-split re-analysis (zero-GPU, CPU-only).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="run Phase 0 input validation only (same code path as full mode), then exit",
    )
    args = parser.parse_args()

    if args.validate_only:
        validate_inputs()
        log.info("[phase=done] validation-only run complete")
        return

    payload = analyze()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n")
    tmp.rename(OUT_PATH)
    n_pc = len(payload["paired_contrasts"])
    n_band = len(payload["band_accounting"])
    n_dec = len(payload["decomposition"])
    if (n_pc, n_band, n_dec) != (32, 16, 48):
        raise AssertionError(f"row-count contract violated: {n_pc}/{n_band}/{n_dec}")
    log.info(
        "[phase=done] wrote %s — %d contrast rows, %d band rows, %d decomposition "
        "rows; headline verdict: %s",
        OUT_PATH.relative_to(PROJECT_ROOT),
        n_pc,
        n_band,
        n_dec,
        payload["verdicts"]["headline"],
    )


if __name__ == "__main__":
    main()
