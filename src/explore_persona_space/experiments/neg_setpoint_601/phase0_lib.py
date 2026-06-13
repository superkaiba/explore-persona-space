# ruff: noqa: RUF001, RUF002, RUF003  # em-dash, minus sign, multiplication sign intentional
"""Task #601 Phase 0 — pure helpers (CPU-testable, no model loads).

Bystander reference-panel pre-registration, margin-reference computation from
the on-policy recheck trajectories, the adapter-application cross-check, and
the space-calibration decision (plan §4 Phase 0 + §6). The GPU driver is
``scripts/i601_phase0_reads.py``; everything decision-shaped lives here so the
local smoke can exercise it without a pod.
"""

from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger("issue_601.phase0_lib")

# Plan v3 §B Gate A — anchor-reuse fitness bound (unchanged 1-nat threshold;
# deliberately stricter than Gate S: a REUSED anchor serves as a Phase-1
# comparison level, so the observed −1.05/−1.12 systematic cross-provenance
# downshift would consume a third of the ±3-nat equilibrium band).
ONPOLICY_CROSSCHECK_TOL_NATS = 1.0
# Plan v3 §B Gate S item 2 — unsaturated-cell reproduction band. 1.5 ≈ 1.7×
# the observed worst-case low-dose diff (0.88) AND half the smallest adjacent
# re-read level gap (negex_100 → anchor ≈ 3.0), so a passing read can never
# alias one dose level as another.
GATE_S_LOW_DOSE_TOL_NATS = 1.5
# Plan v3 §B Gate S item 3 — dose-ordering minimum seed-mean gap (above the
# parent seed noise 1.4, below the smallest observed re-read gap 3.0).
GATE_S_ORDERING_MIN_GAP_NATS = 2.0
# Gate cell roles (the parent count axis; levels per COUNT_CELL_LEVELS).
GATE_LOW_DOSE_CELLS = ("c472_noneg", "c472_negex_100")
GATE_ANCHOR_CELL = "c472_anchor"
OBSERVATION_O_CELLS = ("c472_negex_400",)
# Round-5 structural tripwire: >=3 DISTINCT adapters re-reading within this
# tolerance of one another is physically impossible under faithful
# per-adapter application — it means every read saw the same effective model
# (mapping scramble, or a uniform application artifact such as the
# rsLoRA-over-application collapse ceiling that produced 10.350 ± 0.002
# across six adapters in round 4).
IDENTICAL_REREAD_TOL_NATS = 0.01
IDENTICAL_REREAD_MIN_GROUP = 3
# Plan §4 item 2 — space-calibration divergence threshold.
SPACE_DIVERGENCE_NATS = 2.0
# Plan §6 — clamp contrast threshold (Phase 0b).
CLAMP_GAP_NATS = 1.5

# Output-JSON coverage labels (concern phase0-r-eval-coverage-gap, round 2).
COVERAGE_FULL = "full"
COVERAGE_ABSENT = "absent-from-frozen-R"


def full_r_coverage(r_artifact: dict, persona: str, questions: list[str]) -> bool:
    """True iff ``persona`` has a non-empty ``response_text`` for EVERY question.

    The pinned #472 artifact pair is mutually inconsistent (persona_bank 60
    personas, R_eval 61, only 45 overlap), so panel ⊆ bank does NOT imply
    panel ⊆ R_eval (#504 incident class). Every teacher-forced read path must
    check coverage against the FROZEN R artifact, never against the bank.
    """
    if persona not in r_artifact:
        return False
    rec = r_artifact[persona]
    return all(q in rec and bool(rec[q].get("response_text")) for q in questions)


def split_by_r_coverage(
    r_artifact: dict, personas: list[str], questions: list[str]
) -> tuple[list[str], list[str]]:
    """Order-preserving (covered, uncovered) split by :func:`full_r_coverage`."""
    covered = [p for p in personas if full_r_coverage(r_artifact, p, questions)]
    covered_set = set(covered)
    uncovered = [p for p in personas if p not in covered_set]
    return covered, uncovered


def assert_r_eval_coverage(
    r_artifact: dict, personas: list[str], questions: list[str], *, context: str
) -> None:
    """Fail-loud coverage gate: raise naming every persona lacking full frozen-R coverage.

    Frozen-R parity with #472 is load-bearing — the fix for a coverage miss is
    to constrain the read set (or descope explicitly per-persona), NEVER to
    regenerate R.
    """
    _, uncovered = split_by_r_coverage(r_artifact, personas, questions)
    if uncovered:
        raise KeyError(
            f"R-coverage assert FAILED ({context}): {len(uncovered)} persona(s) lack a "
            f"complete frozen-R entry for all {len(questions)} questions: {uncovered}. "
            f"The pinned parent R artifact does not cover them (#504 class) — constrain "
            f"the read set or descope explicitly; do NOT regenerate frozen R."
        )


def build_r_map(
    r_artifact: dict, personas: list[str], questions: list[str]
) -> dict[str, dict[str, str]]:
    """``{persona: {q: response_text}}`` with a fail-loud coverage assert first.

    This is the exact lookup shape the teacher-forced workers consume; the
    up-front assert turns the round-1 mid-shard ``KeyError: 'bartender'`` crash
    into a pre-launch diagnosis naming every uncovered persona.
    """
    assert_r_eval_coverage(r_artifact, personas, questions, context="build_r_map")
    return {p: {q: r_artifact[p][q]["response_text"] for q in questions} for p in personas}


def select_bystander_reference_panel(
    held_out: list[str],
    cos_to_source: dict[str, float],
    n: int = 8,
) -> list[str]:
    """Pre-register the N-bystander reference panel at L10 d_source deciles.

    ``held_out`` is the #472 held-out panel (bank − source − union of ALL
    trained negatives — so every member is a never-trained bystander by
    construction), pre-filtered by the CALLER to personas with COMPLETE frozen
    R_eval coverage (:func:`split_by_r_coverage`; concern
    phase0-r-eval-coverage-gap — exclusions are recorded by name in
    ``bystander_panel.json``). Sort by ``d_source = 1 − cos`` and take ``n``
    evenly spaced quantile positions (decile coverage prevents distance-skew in
    the Phase 0b clamp contrast — plan §11). Deterministic given the pinned
    centroids + the pinned R artifact.
    """
    if len(held_out) < n:
        raise ValueError(f"held-out panel has {len(held_out)} personas; need >= {n}")
    ranked = sorted(held_out, key=lambda p: 1.0 - cos_to_source[p])
    idx = np.linspace(0, len(ranked) - 1, n).round().astype(int)
    seen: set[int] = set()
    panel: list[str] = []
    for i in idx:
        ii = int(i)
        while ii in seen and ii < len(ranked) - 1:
            ii += 1
        if ii in seen:
            ii = next(k for k in range(len(ranked)) if k not in seen)
        seen.add(ii)
        panel.append(ranked[ii])
    assert len(panel) == n, (len(panel), n)
    return panel


def terminal_source_stats(trajectory: dict) -> dict:
    """Pull the terminal checkpoint's source-self stats from a trajectory.json.

    Returns ``{"delta_g", "delta_z_marker", "delta_margin", "frac"}`` —
    delta_margin = Δ(z_marker − z_eos) trained − base. Requires the four-float
    (Phase B) fields; raises KeyError when ``kl_computed`` was False.
    """
    cks = trajectory["checkpoints"]
    term = max(cks, key=lambda c: c["frac"])
    ss = term["source_self"]
    delta_margin = (ss["z_marker_g_mean"] - ss["z_eos_g_mean"]) - (
        ss["z_marker_b_mean"] - ss["z_eos_b_mean"]
    )
    return {
        "frac": term["frac"],
        "delta_g": float(ss["delta_g_mean"]),
        "delta_z_marker": float(ss["delta_z_marker_mean"]),
        "delta_margin": float(delta_margin),
        # None (not NaN) for missing: NaN survives json.dumps as a bare `NaN`
        # literal — invalid strict JSON in the durable phase0_gate.json
        # (round-5 review minor).
        "emission_p": (float(ss["emission_p"]) if ss.get("emission_p") is not None else None),
        # Regime-validity flag (round 5): a collapsed source R means the
        # re-read ΔG is the repetition ceiling, not an adapter property.
        "r_collapsed": bool(ss.get("r_collapsed", False)),
    }


class IdenticalRereadAlarm(RuntimeError):
    """>=3 distinct adapters re-read to (near-)identical ΔG — identical-read pathology.

    Raised by :func:`onpolicy_crosscheck` INSTEAD of a quiet ``pass=false``
    (round-5 brief fix #3). Carries ``diag`` with the offending group(s) +
    the full per-adapter table so the driver can persist durable evidence
    before re-raising. Known generators: worker→adapter mapping scramble;
    uniform application artifact (e.g. rsLoRA over-application collapse
    ceiling — the realized round-4 case, six adapters at 10.350 ± 0.002).
    """

    def __init__(self, message: str, diag: dict):
        super().__init__(message)
        self.diag = diag


def find_identical_reread_groups(
    delta_g_by_key: dict[str, float],
    tol_nats: float = IDENTICAL_REREAD_TOL_NATS,
    min_group: int = IDENTICAL_REREAD_MIN_GROUP,
) -> list[list[str]]:
    """Groups of >= ``min_group`` keys whose re-read ΔG all sit within ``tol_nats``.

    Pure + CPU-testable. Single-linkage over the sorted values: consecutive
    gaps <= tol chain into one group (the realized failure mode is a single
    tight cluster, e.g. 10.348..10.351).
    """
    items = sorted(delta_g_by_key.items(), key=lambda kv: kv[1])
    groups: list[list[str]] = []
    current: list[str] = []
    for i, (key, val) in enumerate(items):
        if not current:
            current = [key]
        elif val - items[i - 1][1] <= tol_nats:
            current.append(key)
        else:
            if len(current) >= min_group:
                groups.append(current)
            current = [key]
    if len(current) >= min_group:
        groups.append(current)
    return groups


def onpolicy_crosscheck(
    reread_by_cell_seed: dict[str, dict],
    committed_terminal_by_cell_seed: dict[str, float],
    tol_nats: float = ONPOLICY_CROSSCHECK_TOL_NATS,
) -> dict:
    """Adapter-application gate (plan §7 gate 2, #534 class).

    Args:
        reread_by_cell_seed: ``{"<cell>_seed<S>": terminal_source_stats(...)}``
            from this task's Phase 0 on-policy re-reads.
        committed_terminal_by_cell_seed: the COMMITTED #472 trajectory.json
            terminal ``source_self.delta_g_mean`` per cell_seed.
        tol_nats: per-adapter agreement bound (1 nat).

    Returns:
        ``{"pass": bool, "per_adapter": {...}, "tol_nats": ...}`` — pass iff
        EVERY re-read adapter reproduces its committed terminal within tol.

    Raises:
        IdenticalRereadAlarm: when >= IDENTICAL_REREAD_MIN_GROUP distinct
            adapters re-read within IDENTICAL_REREAD_TOL_NATS of one another
            (round-5 structural tripwire — identical-read pathology must be a
            loud, named error, never a quiet ``pass=false``).
    """
    per: dict[str, dict] = {}
    ok = True
    for key, stats in sorted(reread_by_cell_seed.items()):
        if key not in committed_terminal_by_cell_seed:
            raise KeyError(f"no committed terminal ΔG for {key!r}")
        committed = float(committed_terminal_by_cell_seed[key])
        got = float(stats["delta_g"])
        diff = abs(got - committed)
        within = diff <= tol_nats
        ok = ok and within
        per[key] = {
            "committed_delta_g": committed,
            "reread_delta_g": got,
            "abs_diff": diff,
            "within_tol": within,
            # Regime-validity flags (round 5): a ceiling-pinned / collapsed
            # re-read is not a valid parity comparison even when it lands
            # inside tol by accident.
            "reread_r_collapsed": bool(stats.get("r_collapsed", False)),
            # None (not NaN) for missing — this table lands verbatim in
            # phase0_gate.json's alarm diag (round-5 review minor).
            "reread_emission_p": (
                float(stats["emission_p"]) if stats.get("emission_p") is not None else None
            ),
        }
    identical = find_identical_reread_groups({k: v["reread_delta_g"] for k, v in per.items()})
    if identical:
        diag = {
            "identical_groups": identical,
            "tol_nats_identical": IDENTICAL_REREAD_TOL_NATS,
            "per_adapter": per,
        }
        raise IdenticalRereadAlarm(
            "mapping scramble suspected: "
            f"{[len(g) for g in identical]} distinct adapters re-read within "
            f"{IDENTICAL_REREAD_TOL_NATS} nat of one another: {identical}. Identical re-reads "
            "across different adapters are physically impossible under faithful per-adapter "
            "application — either the worker→adapter assignment is scrambled or a uniform "
            "application artifact is pinning every read at a ceiling (e.g. rsLoRA "
            "over-application collapse, the realized round-4 case).",
            diag,
        )
    return {"pass": bool(ok), "tol_nats": tol_nats, "per_adapter": per}


def _cell_of(key: str) -> str:
    return key.rsplit("_seed", 1)[0]


def compute_gate_schema2(
    per_adapter: dict[str, dict],
    *,
    recipe_panel_ok: bool,
    low_dose_tol_nats: float = GATE_S_LOW_DOSE_TOL_NATS,
    ordering_min_gap_nats: float = GATE_S_ORDERING_MIN_GAP_NATS,
    anchor_tol_nats: float = ONPOLICY_CROSSCHECK_TOL_NATS,
) -> dict:
    """Plan v3 §B gate split: Gate S (pass) + Gate A (anchor_reuse_ok) + Observation O.

    Args:
        per_adapter: the :func:`onpolicy_crosscheck` per-adapter table
            (``committed_delta_g`` / ``reread_delta_g`` / ``abs_diff`` /
            ``reread_r_collapsed`` per ``<cell>_seed<S>`` key).
        recipe_panel_ok: the held-out-panel determinism check (plan §10
            fitness (a)) — conjunct of Gate A, as in the v2 gate.

    Returns a dict whose top-level ``pass`` is Gate S ONLY (structural
    eval-path integrity, HALT-class) and ``anchor_reuse_ok`` is Gate A
    (routing-class, never a halt). The negex_400 re-reads land under
    ``observation_o`` (recorded, never gating). Pure + CPU-testable; the
    identical-read pathology is normally raised upstream by
    :func:`onpolicy_crosscheck` as :class:`IdenticalRereadAlarm` — this
    function re-checks the groups defensively so a recompute over a persisted
    table cannot silently skip the tripwire.
    """
    # ── Gate S item 1: differentiation / alarm silent + r_collapsed false. ──
    identical_groups = find_identical_reread_groups(
        {k: float(v["reread_delta_g"]) for k, v in per_adapter.items()}
    )
    r_collapsed_keys = sorted(k for k, v in per_adapter.items() if v.get("reread_r_collapsed"))
    alarm_silent = not identical_groups
    r_collapsed_all_false = not r_collapsed_keys

    # ── Gate S item 2: low-dose reproduction within ±1.5 nat. ───────────────
    low_dose = {
        k: float(v["abs_diff"])
        for k, v in per_adapter.items()
        if _cell_of(k) in GATE_LOW_DOSE_CELLS
    }
    if len(low_dose) != 2 * len(GATE_LOW_DOSE_CELLS):
        raise KeyError(
            f"gate_schema 2 needs both seeds of {GATE_LOW_DOSE_CELLS}; "
            f"found low-dose keys {sorted(low_dose)}"
        )
    low_dose_ok = all(d <= low_dose_tol_nats for d in low_dose.values())

    # ── Gate S item 3: dose ordering with >= 2-nat seed-mean gaps. ──────────
    means: dict[str, float] = {}
    for key, row in per_adapter.items():
        means.setdefault(_cell_of(key), []).append(float(row["reread_delta_g"]))  # type: ignore[arg-type]
    seed_means = {c: float(np.mean(v)) for c, v in means.items()}
    noneg_m, negex100_m = seed_means["c472_noneg"], seed_means["c472_negex_100"]
    top_min = min(seed_means[GATE_ANCHOR_CELL], *(seed_means[c] for c in OBSERVATION_O_CELLS))
    gap_low = negex100_m - noneg_m
    gap_high = top_min - negex100_m
    ordering_ok = gap_low >= ordering_min_gap_nats and gap_high >= ordering_min_gap_nats

    gate_s_pass = bool(alarm_silent and r_collapsed_all_false and low_dose_ok and ordering_ok)

    # ── Gate A: both anchor adapters re-read within 1.0 nat (+ recipe match).
    anchor_rows = {k: v for k, v in per_adapter.items() if _cell_of(k) == GATE_ANCHOR_CELL}
    anchor_within = {k: float(v["abs_diff"]) <= anchor_tol_nats for k, v in anchor_rows.items()}
    anchor_onpolicy_ok = bool(anchor_within) and all(anchor_within.values())
    anchor_reuse_ok = bool(recipe_panel_ok and anchor_onpolicy_ok)

    # ── Observation O: saturated-cell endpoint re-read (recorded, no gate). ──
    obs_rows = {
        k: {
            "committed_delta_g": float(v["committed_delta_g"]),
            "reread_delta_g": float(v["reread_delta_g"]),
            "abs_diff": float(v["abs_diff"]),
        }
        for k, v in sorted(per_adapter.items())
        if _cell_of(k) in OBSERVATION_O_CELLS
    }

    return {
        "gate_schema": 2,
        "pass": gate_s_pass,
        "anchor_reuse_ok": anchor_reuse_ok,
        "gate_s": {
            "pass": gate_s_pass,
            "class": "structural eval-path integrity (HALT-class, #534 class; plan v3 §B)",
            "alarm_silent": alarm_silent,
            "identical_groups": identical_groups,
            "r_collapsed_all_false": r_collapsed_all_false,
            "r_collapsed_keys": r_collapsed_keys,
            "low_dose_reproduction_ok": low_dose_ok,
            "low_dose_abs_diffs": dict(sorted(low_dose.items())),
            "low_dose_tol_nats": low_dose_tol_nats,
            "dose_ordering_ok": ordering_ok,
            "dose_ordering_seed_means": dict(sorted(seed_means.items())),
            "dose_ordering_gaps": {"noneg_to_negex100": gap_low, "negex100_to_top_min": gap_high},
            "ordering_min_gap_nats": ordering_min_gap_nats,
        },
        "gate_a": {
            "anchor_reuse_ok": anchor_reuse_ok,
            "class": "anchor-reuse fitness (routing-class, never a halt; plan v3 §B)",
            "anchor_within_1nat": anchor_within,
            "anchor_onpolicy_ok": anchor_onpolicy_ok,
            "recipe_panel_ok": bool(recipe_panel_ok),
            "tol_nats": anchor_tol_nats,
            "on_false": "dispatcher --anchor-retrain-fallback (dense_200p800n seed 42)",
        },
        "observation_o": {
            "per_adapter": obs_rows,
            "gating": False,
            "note": (
                "negex_400 re-reads sit ~7 nats below their committed terminals while "
                "differentiation is preserved and r_collapsed is false on all 8 — the "
                "registered Phase-0a regime deliverable: the parent dose-response TOP "
                "compresses under this rig's read regime (classic-gauge staged copies + "
                "D3 generation budget on repeater-class cells), not adapter damage "
                "(plan v3 §B Observation O / §F assumption 19)."
            ),
        },
    }


def onpolicy_worker_plan(
    onpolicy_keys: list[str],
    phase0_dir,
    adapters_root,
    data_dir,
    log_dir,
) -> list[dict]:
    """Pure cell→worker mapping for the 0a-op re-reads (CPU-testable, no I/O).

    Returns one plan row per ``<cell>_seed<S>`` key:
    ``{"key", "cell", "seed", "adapter_path", "out_dir", "idx_path",
    "log_path", "cmd"}``. The driver writes ``idx_path`` (the synthetic
    single-checkpoint index pointing at ``adapter_path``) and launches
    ``cmd``; the test asserts ``key in adapter_path`` and per-row uniqueness —
    the round-5 brief's mapping-construction smoke.
    """
    from pathlib import Path

    phase0_dir, adapters_root = Path(phase0_dir), Path(adapters_root)
    plan: list[dict] = []
    for key in onpolicy_keys:
        cell, seed_s = key.rsplit("_seed", 1)
        out_dir = phase0_dir / "onpolicy_recheck" / key
        idx_path = out_dir / "checkpoint_index.json"
        adapter_path = adapters_root / key
        cmd = [
            "uv",
            "run",
            "python",
            "scripts/i601_eval_trajectory.py",
            "--cell",
            cell,
            "--seed",
            seed_s,
            "--checkpoint-index",
            str(idx_path),
            "--out-path",
            str(out_dir / "trajectory.json"),
            "--raw-completions-path",
            str(out_dir / "raw_completions.json"),
            "--data-dir",
            str(data_dir),
            "--fracs",
            "1.0000",
            "--panel",
            "bystander8",
            "--bystander-panel-path",
            str(phase0_dir / "bystander_panel.json"),
        ]
        plan.append(
            {
                "key": key,
                "cell": cell,
                "seed": int(seed_s),
                "adapter_path": str(adapter_path),
                "out_dir": str(out_dir),
                "idx_path": str(idx_path),
                "log_path": str(Path(log_dir) / f"issue-601-phase0-op-{key}.log"),
                "cmd": cmd,
            }
        )
    return plan


def margin_references(
    reread_by_cell_seed: dict[str, dict],
    level_by_cell: dict[str, str],
) -> dict:
    """Margin-space references M(level) from the ON-POLICY 8-adapter subset.

    Plan §6 pinned rule: M(·) comes from the on-policy read (the same read
    type Phase 1 consumes), NEVER the teacher-forced 20-adapter read. The
    margin-space tolerance is DERIVED as
    ``max(2 × per-seed on-policy margin gap at the level, 1.0 logit)``.

    Args:
        reread_by_cell_seed: ``{"c472_anchor_seed42": terminal_source_stats}``.
        level_by_cell: ``{"c472_anchor": "4:1", ...}``.

    Returns:
        ``{level: {"margin_mean", "logp_mean", "delta_z_mean", "seed_gap_margin",
        "tolerance_margin", "divergence_logp_vs_z"}}``.
    """
    by_level: dict[str, list[dict]] = {}
    for key, stats in reread_by_cell_seed.items():
        cell = key.rsplit("_seed", 1)[0]
        if cell not in level_by_cell:
            continue
        by_level.setdefault(level_by_cell[cell], []).append(stats)
    out: dict[str, dict] = {}
    for level, rows in sorted(by_level.items()):
        margins = [r["delta_margin"] for r in rows]
        logps = [r["delta_g"] for r in rows]
        dzs = [r["delta_z_marker"] for r in rows]
        seed_gap = float(max(margins) - min(margins)) if len(margins) > 1 else 0.0
        out[level] = {
            "n_adapters": len(rows),
            "margin_mean": float(np.mean(margins)),
            "logp_mean": float(np.mean(logps)),
            "delta_z_mean": float(np.mean(dzs)),
            "seed_gap_margin": seed_gap,
            "tolerance_margin": max(2.0 * seed_gap, 1.0),
            "divergence_logp_vs_z": float(abs(np.mean(dzs) - np.mean(logps))),
        }
    return out


def decide_primary_space(margin_refs: dict) -> dict:
    """Space calibration (plan §4 Phase-0 item 2).

    Δlog P vs Δz_marker divergence >= 2 nats at the 4:1 level OR BELOW → the
    EOS margin becomes PRIMARY for ALL Phase-1 arms; divergence confined to
    the 8:1 level (expected — those cells sit ~2.3 nats from ceiling) → log P
    stays primary for arms landing <= L(4:1)+3, margin rules the upper branch.
    """
    low_levels = [lv for lv in ("0:1", "2:1", "4:1") if lv in margin_refs]
    diverged_low = [
        lv for lv in low_levels if margin_refs[lv]["divergence_logp_vs_z"] >= SPACE_DIVERGENCE_NATS
    ]
    if diverged_low:
        choice = "margin"
        reason = (
            f"Δlog P vs Δz_marker diverge >= {SPACE_DIVERGENCE_NATS} nats at level(s) "
            f"{diverged_low} (<= 4:1) — EOS margin is PRIMARY for all Phase-1 arms."
        )
    else:
        choice = "logp_with_margin_upper"
        reason = (
            "divergence confined to (at most) the 8:1 level — log P stays primary for "
            "arms landing <= L(4:1)+3; the EOS margin rules the upper branch."
        )
    return {
        "primary_space": choice,
        "reason": reason,
        "divergence_by_level": {lv: margin_refs[lv]["divergence_logp_vs_z"] for lv in margin_refs},
    }


def clamp_read(
    teacher_by_cell_seed: dict[str, dict],
    bystander_panel: list[str],
    negatives_by_cell: dict[str, list[str]],
    count_cells: list[str],
    gap_nats: float = CLAMP_GAP_NATS,
) -> dict:
    """Phase 0b trained-negative clamp contrast (plan §6).

    ``teacher_by_cell_seed`` maps ``"<cell>_seed<S>"`` to the per-persona
    teacher-forced read ``{persona: {q: {"logp_hf_g", "logp_hf_b", ...}}}``.
    Clamp present iff mean(trained-negative ΔG) <= mean(bystander ΔG) − gap
    in >= 3 of the 4 count cells, BOTH seeds. Distance-residualized variant +
    threshold sensitivity are computed downstream in analysis (concern #4);
    this is the registered panel-mean read.
    """

    def _mean_dg(read: dict, personas: list[str]) -> float:
        vals = [
            rec["logp_hf_g"] - rec["logp_hf_b"]
            for p in personas
            if p in read
            for rec in read[p].values()
        ]
        if not vals:
            raise ValueError(f"clamp read: no records for personas {personas[:4]}...")
        return float(np.mean(vals))

    per_cell_seed: dict[str, dict] = {}
    clamped_count: dict[str, int] = {}
    for key, read in sorted(teacher_by_cell_seed.items()):
        cell = key.rsplit("_seed", 1)[0]
        if cell not in count_cells:
            continue
        negs = [p for p in negatives_by_cell[cell] if p in read]
        if not negs:
            # 0-negative count cell (the 0:1 level): no trained negatives to
            # clamp — recorded n/a and counted NOT-clamped (the >=3-of-4 rule
            # then effectively requires all three negative-bearing levels).
            per_cell_seed[key] = {"clamped": None, "note": "no trained negatives (0:1 level)"}
            continue
        neg_dg = _mean_dg(read, negs)
        by_dg = _mean_dg(read, bystander_panel)
        clamped = neg_dg <= by_dg - gap_nats
        per_cell_seed[key] = {
            "trained_neg_mean_dg": neg_dg,
            "bystander_mean_dg": by_dg,
            "gap": by_dg - neg_dg,
            "clamped": bool(clamped),
        }
        clamped_count[cell] = clamped_count.get(cell, 0) + (1 if clamped else 0)
    n_seeds = 2
    cells_clamped_both_seeds = sum(1 for c, k in clamped_count.items() if k == n_seeds)
    present = cells_clamped_both_seeds >= 3
    return {
        "clamp_present": bool(present),
        "gap_nats": gap_nats,
        "cells_clamped_both_seeds": cells_clamped_both_seeds,
        "per_cell_seed": per_cell_seed,
        "rule": "clamped iff neg <= bystander − gap in >=3 of 4 count cells, both seeds",
    }
