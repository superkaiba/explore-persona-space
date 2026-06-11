# ruff: noqa: RUF001, RUF002, RUF003
# Intentional Unicode (※, Δ, −) in scientific docstrings + labels.
"""Task #571 follow-up `persona-split-composition` — Stage 1 geometry join.

VM-side, CPU-only, ZERO GPU, NO POD (plan v2 §4.0/§4.1). Joins the committed
#560 layer-20 persona↔context geometry with the parent #571 run's committed
per-persona DVs and runs the registered nearest-negative-gradient and
barrier-vs-bubble reads on the parent's own broad/narrow arms — BEFORE any
Stage-2 pod provision. The Stage-2 dispatcher refuses to start unless this
script's output JSON is committed (the §4.0 ordering contract).

Inputs (all committed / Hub-pinned, fail-loud on any missing field):
- ``eval_results/issue_560/geometry/context_persona_geometry.json`` —
  L20, 50-probe, raw-cosine ``min_dist[context][persona]`` (16×35).
- ``eval_results/issue_571/breadth_contrast.json`` — per_label_per_persona
  Δz_EOS (the clamp channel).
- ``eval_results/issue_571/four_float/trained_{label}.json`` — slot-kind
  cross-check for the hijack recompute.
- HF ``issue571_breadth/raw_completions/raw_completions_{label}.json`` @
  pinned rev — hijack rates recomputed from the parent's own generations.

Registered statistics (plan v2 §4.1, post-revision): collinearity
Pearson(d_nn, d_src) = 0.635 (broad) / 0.996 (narrow) was computed at plan
time, so the residualized Spearman — Spearman(leakage, resid(d_nn)) with
resid = residuals of d_nn on a linear+quadratic fit of d_src — IS the
registered primary within-cell statistic for BROAD cells (permutation p,
10,000 persona-unit draws + bootstrap CI; Holm with the barrier partial,
2 geometry reads per channel). Narrow-arm within-cell reads are
pre-declared DEGENERATE (collinearity 0.996) — descriptive only, excluded
from the registered verdict. The cross-arm paired identification
(Δ(clamp)/Δ(hijack) broad − narrow vs Δd_nn; d_src constant within persona
by construction) carries the verdict alongside the broad-arm gradients.

Outputs:
- ``eval_results/issue_571/persona-split-composition/stage1_geometry_join.json``
- ``figures/issue_571/persona-split-composition/stage1_*``

Usage (VM, repo root of the issue-571 worktree):
    uv run python scripts/issue571_psplit_stage1_analysis.py
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPTS_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(SCRIPTS_DIR))

import numpy as np  # noqa: E402
from issue560_crossrecipe_panel import (  # noqa: E402
    EXPECTED_PROMPT_MATCHES,
    HELD_OUT_35,
)
from issue571_psplit_stats import (  # noqa: E402
    _proj_residual_matrix,
    bootstrap_ci,
    ci_excludes_zero,
    holm,
    partial_spearman,
    perm_p_corr_vs_fixed,
    quad_residuals,
    residualized_spearman,
    spearman,
)
from scipy.stats import rankdata  # noqa: E402

logger = logging.getLogger("issue571.psplit_stage1")

SCHEMA_VERSION = "issue571_psplit_stage1_v1"
GEOMETRY_560 = PROJECT_ROOT / "eval_results/issue_560/geometry/context_persona_geometry.json"
BREADTH_CONTRAST = PROJECT_ROOT / "eval_results/issue_571/breadth_contrast.json"
FOUR_FLOAT_DIR = PROJECT_ROOT / "eval_results/issue_571/four_float"
OUT_DIR = PROJECT_ROOT / "eval_results/issue_571/persona-split-composition"
FIG_DIR = PROJECT_ROOT / "figures/issue_571/persona-split-composition"

HF_DATA_REPO = "superkaiba1/explore-persona-space-data"
RAW_COMPLETIONS_REV = "5337d5f794a2162de9748fc1bcd64fc2caa2d777"  # parent body Artifacts pin

SOURCE_CONTEXT = "A2"
NARROW_PANEL = ["A1", "B1", "C1", "D1"]
ARMS = {"broad": ["broad_s42", "broad_s43"], "narrow": ["narrow_s42", "narrow_s43"]}
ALL_LABELS = [label for labels in ARMS.values() for label in labels]
SEED_PAIRS = {"42": ("broad_s42", "narrow_s42"), "43": ("broad_s43", "narrow_s43")}
CHANNELS = ("clamp", "hijack")  # clamp = Δz_EOS (primary); hijack = emission rate
MARKER_SUBSTRING = "※"

N_PERM = 10_000
N_BOOT = 10_000
STAT_SEED = 42
ALPHA = 0.05
SMALL_SPREAD_FACTOR = 0.25  # §4.1 linkage (a): median|Δd_nn| < 0.25 × sd(d_src)

# Sign convention for family attribution (code-review round-1 blocker
# `stage1-barrier-sign-routing`). Registered signs in LEAKAGE units (plan
# §4.1 / the scope marker's shell convention): BARRIER = leakage RISES with
# d_src controlling d_nn (positive); BUBBLE = leakage rises with d_nn
# controlling d_src / Δleakage moves WITH Δd_nn (positive — suppression is
# local to trained negatives, so marker leakage grows away from them and
# drops where the panel change pulls d_nn down). The hijack DV (marker
# emission rate) IS leakage (dir +1); the clamp DV (Δz_EOS) is the
# SUPPRESSION push — anti-leakage — so its raw-DV expected signs flip
# (dir −1). With both family signs positive in leakage units, the expected
# raw-DV sign for EVERY registered read (barrier partial, residualized
# gradient, cross-arm paired) is exactly CHANNEL_LEAKAGE_DIR[channel].
CHANNEL_LEAKAGE_DIR = {"clamp": -1, "hijack": +1}


def _git_commit() -> str:
    """Short git commit of the repo this script runs from."""
    try:
        return (
            subprocess.run(  # epm-lint: subprocess-env-inherit -- git metadata probe, no creds
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
            or "unknown"
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _git_last_commit_of(path: Path) -> str:
    """Full SHA of the last commit touching ``path`` (input-revision record)."""
    try:
        return (
            subprocess.run(  # epm-lint: subprocess-env-inherit -- git metadata probe, no creds
                ["git", "log", "-n1", "--format=%H", "--", str(path)],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
            or "unknown"
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def load_geometry() -> tuple[list[str], dict[str, dict[str, float]]]:
    """The #560 L20 geometry: (sources, min_dist[context][persona]) — fail loud."""
    payload = json.loads(GEOMETRY_560.read_text())
    assert payload["layer"] == 20 and payload["n_probes"] == 50, (
        payload["layer"],
        payload["n_probes"],
    )
    assert payload["metric"] == "cosine_distance", payload["metric"]
    sources = list(payload["sources"])
    assert len(sources) == 16 and SOURCE_CONTEXT in sources, sources
    md = payload["min_dist"]
    for c in sources:
        for p in HELD_OUT_35:
            assert p in md[c], (c, p)
    for c in NARROW_PANEL:
        assert c in md, c
    return sources, md


def derive_distances(
    sources: list[str], md: dict[str, dict[str, float]], personas: list[str]
) -> dict:
    """d_src, per-arm d_nn + nearest-negative identity over ``personas`` (§4.1)."""
    broad_panel = [c for c in sources if c != SOURCE_CONTEXT]
    assert len(broad_panel) == 15, broad_panel
    out: dict = {
        "panel": {"broad": broad_panel, "narrow": list(NARROW_PANEL)},
        "d_src": {p: float(md[SOURCE_CONTEXT][p]) for p in personas},
        "d_nn": {},
        "nn_identity": {},
    }
    for arm, panel in (("broad", broad_panel), ("narrow", NARROW_PANEL)):
        d_nn, nn_id = {}, {}
        for p in personas:
            dists = {c: float(md[c][p]) for c in panel}
            nn = min(dists, key=dists.get)
            d_nn[p], nn_id[p] = dists[nn], nn
        out["d_nn"][arm] = d_nn
        out["nn_identity"][arm] = nn_id
    return out


def load_clamp(personas: list[str]) -> dict[str, dict[str, float]]:
    """Per-label per-persona Δz_EOS from the parent's committed analysis output."""
    payload = json.loads(BREADTH_CONTRAST.read_text())
    plp = payload["per_label_per_persona"]
    out: dict[str, dict[str, float]] = {}
    for label in ALL_LABELS:
        assert label in plp, (label, sorted(plp))
        ch = plp[label]["dz_eos"]
        out[label] = {p: float(ch[p]) for p in personas}
    return out


def recompute_hijack(personas: list[str]) -> tuple[dict[str, dict[str, float]], dict]:
    """Hijack rate per (label, persona) recomputed from the parent's HF raw completions.

    Rate = fraction of the 20 eval questions whose generated response contains
    the marker glyph. Cross-checked per (label, persona) against the committed
    trained four-float ``n_pre_marker_slots`` (slot_kind ``pre_marker`` ⇔ the
    marker token appears in R) — any mismatch fails loud.
    """
    from huggingface_hub import hf_hub_download

    rates: dict[str, dict[str, float]] = {}
    diag = {"n_completions": 0, "n_marker": 0, "rev": RAW_COMPLETIONS_REV}
    for label in ALL_LABELS:
        local = hf_hub_download(
            repo_id=HF_DATA_REPO,
            repo_type="dataset",
            revision=RAW_COMPLETIONS_REV,
            filename=f"issue571_breadth/raw_completions/raw_completions_{label}.json",
        )
        payload = json.loads(Path(local).read_text())
        completions = payload["completions"]
        ff = json.loads((FOUR_FLOAT_DIR / f"trained_{label}.json").read_text())["per_persona"]
        rates[label] = {}
        for p in personas:
            recs = completions[p]
            fired = [MARKER_SUBSTRING in rec["response_text"] for rec in recs.values()]
            assert len(fired) == 20, (label, p, len(fired))
            n_fired = int(sum(fired))
            n_pre_marker = int(ff[p]["summary"]["n_pre_marker_slots"])
            assert n_fired == n_pre_marker, (
                f"hijack cross-check FAIL for ({label}, {p}): substring count {n_fired} != "
                f"four-float n_pre_marker_slots {n_pre_marker}"
            )
            rates[label][p] = n_fired / 20.0
            diag["n_completions"] += len(fired)
            diag["n_marker"] += n_fired
    return rates, diag


def _read_block(
    stat: float,
    p_perm: float,
    ci_lo: float,
    ci_hi: float,
    n_boot_dropped: int,
    *,
    registered: bool,
    degenerate: bool = False,
    note: str = "",
) -> dict:
    return {
        "stat": None if not np.isfinite(stat) else float(stat),
        "perm_p": None if not np.isfinite(p_perm) else float(p_perm),
        "ci95": [
            None if not np.isfinite(ci_lo) else float(ci_lo),
            None if not np.isfinite(ci_hi) else float(ci_hi),
        ],
        "ci_excludes_zero": ci_excludes_zero(ci_lo, ci_hi),
        "n_boot_dropped": n_boot_dropped,
        "registered": registered,
        "degenerate": degenerate,
        "note": note,
    }


def within_cell_reads(
    leak: np.ndarray, d_nn: np.ndarray, d_src: np.ndarray, *, registered: bool
) -> dict:
    """One cell × channel: gradient (residualized primary), barrier, companions.

    Holm over the 2 geometry reads {gradient_resid, barrier} per the §4.1
    inference spec (applied to the permutation p-values).
    """
    resid_dnn = quad_residuals(d_nn, d_src)
    # Gradient (bubble direction): Spearman(leak, resid(d_nn)) — §4.1 primary.
    g_stat = residualized_spearman(leak, d_nn, d_src)
    g_p = perm_p_corr_vs_fixed(leak, rankdata(resid_dnn), g_stat, n_perm=N_PERM, seed=STAT_SEED)
    g_lo, g_hi, g_drop = bootstrap_ci(
        lambda idx: residualized_spearman(leak[idx], d_nn[idx], d_src[idx]),
        len(leak),
        n_boot=N_BOOT,
        seed=STAT_SEED,
    )
    # Barrier: partial Spearman(leak, d_src | d_nn).
    b_stat = partial_spearman(leak, d_src, d_nn)
    proj_dnn = _proj_residual_matrix(d_nn)
    b_p = perm_p_corr_vs_fixed(
        leak,
        proj_dnn @ rankdata(d_src),
        b_stat,
        proj=proj_dnn,
        n_perm=N_PERM,
        seed=STAT_SEED,
    )
    b_lo, b_hi, b_drop = bootstrap_ci(
        lambda idx: partial_spearman(leak[idx], d_src[idx], d_nn[idx]),
        len(leak),
        n_boot=N_BOOT,
        seed=STAT_SEED,
    )
    holm_ps = holm({"gradient_resid": g_p, "barrier": b_p})
    # Descriptive companions: plain partial (bubble), tercile medians, #472 retest.
    p_stat = partial_spearman(leak, d_nn, d_src)
    retest = spearman(leak, d_src)
    order = np.argsort(resid_dnn)
    terciles = np.array_split(order, 3)
    tercile_medians = [float(np.median(leak[t])) for t in terciles]
    out = {
        "gradient_resid": _read_block(
            g_stat,
            g_p,
            g_lo,
            g_hi,
            g_drop,
            registered=registered,
            degenerate=not np.isfinite(g_stat),
            note="Spearman(leakage, resid(d_nn on quad d_src)) — §4.1 primary",
        ),
        "barrier": _read_block(
            b_stat,
            b_p,
            b_lo,
            b_hi,
            b_drop,
            registered=registered,
            degenerate=not np.isfinite(b_stat),
            note="partial Spearman(leakage, d_src | d_nn) — shell convention: positive = barrier",
        ),
        "holm_p": {k: (None if not np.isfinite(v) else float(v)) for k, v in holm_ps.items()},
        "companions": {
            "partial_spearman_dnn_given_dsrc": None if not np.isfinite(p_stat) else float(p_stat),
            "tercile_medians_by_resid_dnn": tercile_medians,
            "retest_spearman_leak_dsrc": None if not np.isfinite(retest) else float(retest),
        },
    }
    return out


def cross_arm_reads(
    leak_by_label: dict[str, dict[str, float]],
    personas: list[str],
    d_nn_broad: np.ndarray,
    d_nn_narrow: np.ndarray,
) -> dict:
    """§4.1 analysis 2(ii): per-persona Δleakage (broad − narrow) vs Δd_nn.

    Pooled (arm = mean over its 2 seeds) is the registered read; matched-seed
    variants (s42, s43) are the sign-agreement support. d_src is constant
    within persona across arms, so no partial is needed.
    """
    d_dnn = d_nn_broad - d_nn_narrow  # <= 0 per persona (narrow ⊂ broad)
    rank_ddnn = rankdata(d_dnn)
    variants: dict[str, dict[str, np.ndarray]] = {}
    for name, (b_label, n_label) in SEED_PAIRS.items():
        variants[f"seed{name}"] = {
            "delta": np.array(
                [leak_by_label[b_label][p] - leak_by_label[n_label][p] for p in personas]
            )
        }
    pooled = np.array(
        [
            np.mean([leak_by_label[lb][p] for lb in ARMS["broad"]])
            - np.mean([leak_by_label[lb][p] for lb in ARMS["narrow"]])
            for p in personas
        ]
    )
    variants["pooled"] = {"delta": pooled}
    out: dict = {}
    for vname, v in variants.items():
        delta = v["delta"]
        stat = spearman(delta, d_dnn)
        p = perm_p_corr_vs_fixed(delta, rank_ddnn, stat, n_perm=N_PERM, seed=STAT_SEED)
        lo, hi, drop = bootstrap_ci(
            lambda idx, d=delta: spearman(d[idx], d_dnn[idx]),
            len(delta),
            n_boot=N_BOOT,
            seed=STAT_SEED,
        )
        out[vname] = _read_block(
            stat,
            p,
            lo,
            hi,
            drop,
            registered=(vname == "pooled"),
            degenerate=not np.isfinite(stat),
            note="Spearman(Δleakage broad−narrow, Δd_nn) — cross-arm paired identification",
        )
    return out


def _attribute_one(
    name: str,
    block: dict,
    holm_p: float | None,
    direction_family: str,
    expected_sign: int,
    resolved_reads: list[dict],
    inverted_reads: list[dict],
) -> None:
    """Sign-encoded attribution of ONE read (`stage1-barrier-sign-routing` fix).

    A read resolved at Holm p < ALPHA with CI excluding 0 counts toward its
    family ONLY in the registered direction (``CHANNEL_LEAKAGE_DIR``); a
    resolved read with the INVERTED sign is recorded descriptively and never
    family-resolves the linkage.
    """
    p = holm_p if holm_p is not None else float("nan")
    if not (
        block["stat"] is not None and np.isfinite(p) and p < ALPHA and block["ci_excludes_zero"]
    ):
        return
    entry = {
        "read": name,
        "stat": block["stat"],
        "holm_p": p,
        "family": direction_family,
        "expected_sign": expected_sign,
        "observed_sign": int(np.sign(block["stat"])),
    }
    if int(np.sign(block["stat"])) == expected_sign:
        entry["sign_status"] = "registered"
        resolved_reads.append(entry)
    else:
        entry["sign_status"] = "inverted"
        entry["note"] = (
            f"resolved at Holm p < {ALPHA} with CI excluding 0 but in the INVERTED "
            f"direction for the {direction_family} family — reported descriptively, "
            "never family-resolved for the linkage"
        )
        inverted_reads.append(entry)


def attribute_families(within: dict, cross: dict) -> tuple[list[dict], list[dict], str]:
    """(resolved_reads, inverted_reads, verdict) over the registered Stage-1 reads."""
    resolved_reads: list[dict] = []
    inverted_reads: list[dict] = []
    for ch in CHANNELS:
        sgn = CHANNEL_LEAKAGE_DIR[ch]
        for label in ARMS["broad"]:
            cell = within[ch][label]
            _attribute_one(
                f"{ch}/{label}/gradient_resid",
                cell["gradient_resid"],
                cell["holm_p"]["gradient_resid"],
                "bubble",
                sgn,
                resolved_reads,
                inverted_reads,
            )
            _attribute_one(
                f"{ch}/{label}/barrier",
                cell["barrier"],
                cell["holm_p"]["barrier"],
                "barrier",
                sgn,
                resolved_reads,
                inverted_reads,
            )
        _attribute_one(
            f"{ch}/cross_arm/pooled",
            cross[ch]["pooled"],
            cross[ch]["pooled"]["holm_p"],
            "bubble",
            sgn,
            resolved_reads,
            inverted_reads,
        )
    families = {r["family"] for r in resolved_reads}
    if not resolved_reads:
        verdict = "unidentified_on_context_typed_negatives"
    elif families == {"bubble"}:
        verdict = "resolved_bubble_like"
    elif families == {"barrier"}:
        verdict = "resolved_barrier_like"
    else:
        verdict = "resolved_mixed"
    return resolved_reads, inverted_reads, verdict


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    ap = argparse.ArgumentParser(
        description="Task #571 persona-split-composition Stage 1: zero-GPU geometry join.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--fig-dir", type=Path, default=FIG_DIR)
    args = ap.parse_args(argv)

    from explore_persona_space.orchestrate.env import load_dotenv

    load_dotenv()  # hf_hub_download for the parent raw completions

    personas = [p for p in HELD_OUT_35 if p not in EXPECTED_PROMPT_MATCHES]
    assert len(personas) == 32, len(personas)

    sources, md = load_geometry()
    dist = derive_distances(sources, md, personas)
    d_src = np.array([dist["d_src"][p] for p in personas])
    d_nn = {arm: np.array([dist["d_nn"][arm][p] for p in personas]) for arm in ARMS}

    # Collinearity (plan-time values 0.635 / 0.996 re-derived here as a gate).
    collinearity = {arm: float(np.corrcoef(d_nn[arm], d_src)[0, 1]) for arm in ARMS}
    logger.info("collinearity Pearson(d_nn, d_src): %s", collinearity)
    assert abs(collinearity["broad"] - 0.635) < 0.02, collinearity
    assert abs(collinearity["narrow"] - 0.996) < 0.01, collinearity

    clamp = load_clamp(personas)
    hijack, hijack_diag = recompute_hijack(personas)
    leak = {"clamp": clamp, "hijack": hijack}

    # Realized spread (reported with every read; §4.1 linkage trigger).
    d_dnn = d_nn["broad"] - d_nn["narrow"]
    spread = {
        "d_nn_broad": {
            "median": float(np.median(d_nn["broad"])),
            "iqr": [float(q) for q in np.percentile(d_nn["broad"], [25, 75])],
        },
        "d_nn_narrow": {
            "median": float(np.median(d_nn["narrow"])),
            "iqr": [float(q) for q in np.percentile(d_nn["narrow"], [25, 75])],
        },
        "delta_d_nn": {
            "median": float(np.median(d_dnn)),
            "median_abs": float(np.median(np.abs(d_dnn))),
            "iqr": [float(q) for q in np.percentile(d_dnn, [25, 75])],
            "min": float(d_dnn.min()),
            "max": float(d_dnn.max()),
        },
        "sd_d_src": float(np.std(d_src)),
        "small_spread_threshold": SMALL_SPREAD_FACTOR * float(np.std(d_src)),
        "small_spread": bool(
            float(np.median(np.abs(d_dnn))) < SMALL_SPREAD_FACTOR * float(np.std(d_src))
        ),
    }

    # Within-cell reads (4 cells × 2 channels). Broad cells registered;
    # narrow cells pre-declared degenerate/descriptive (collinearity 0.996).
    within: dict[str, dict[str, dict]] = {ch: {} for ch in CHANNELS}
    for ch in CHANNELS:
        for arm, labels in ARMS.items():
            for label in labels:
                vec = np.array([leak[ch][label][p] for p in personas])
                within[ch][label] = within_cell_reads(
                    vec, d_nn[arm], d_src, registered=(arm == "broad")
                )
                within[ch][label]["arm"] = arm
                within[ch][label]["registered_cell"] = arm == "broad"

    # Cross-arm paired identification (registered, both channels).
    cross: dict[str, dict] = {
        ch: cross_arm_reads(leak[ch], personas, d_nn["broad"], d_nn["narrow"]) for ch in CHANNELS
    }
    # Conservative Holm over the two pooled cross-arm channels (implementation
    # choice — the §4.1 within-channel Holm family covers only the 2 within-
    # cell geometry reads; the cross-arm read gets its own 2-channel family).
    cross_holm = holm(
        {
            ch: (
                cross[ch]["pooled"]["perm_p"]
                if cross[ch]["pooled"]["perm_p"] is not None
                else float("nan")
            )
            for ch in CHANNELS
        }
    )
    for ch in CHANNELS:
        v = cross_holm[ch]
        cross[ch]["pooled"]["holm_p"] = None if not np.isfinite(v) else float(v)

    # #472 retest (rank-based, per arm pooled over seeds).
    retest = {}
    for ch in CHANNELS:
        retest[ch] = {}
        for arm, labels in ARMS.items():
            pooled_vec = np.array([np.mean([leak[ch][lb][p] for lb in labels]) for p in personas])
            retest[ch][arm] = {
                "spearman_leak_dsrc": (
                    None
                    if not np.isfinite(spearman(pooled_vec, d_src))
                    else float(spearman(pooled_vec, d_src))
                ),
                "reference_472_L10_raw": -0.519,
            }

    # ── Stage-1 verdict (registered reads only) ────────────────────────────
    resolved_reads, inverted_reads, verdict = attribute_families(within, cross)

    # ── Stage-1 → Stage-2 linkage (§4.1, exactly two registered effects) ───
    one_sided: dict[str, str | None] = {ch: None for ch in CHANNELS}
    for r in resolved_reads:
        ch = r["read"].split("/")[0]
        direction = "bubble" if r["family"] == "bubble" else "barrier"
        one_sided[ch] = direction if one_sided[ch] in (None, direction) else "mixed"
    linkage = {
        "fallback_objective": (
            "variance-max"
            if (verdict == "unidentified_on_context_typed_negatives" and spread["small_spread"])
            else "coverage-max"
        ),
        "g1_promoted_to_selection_objective": bool(
            verdict == "unidentified_on_context_typed_negatives" and spread["small_spread"]
        ),
        "one_sided_expectation": one_sided,
    }

    # ── Output JSON ────────────────────────────────────────────────────────
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / "stage1_geometry_join.json"
    payload = {
        "schema_version": SCHEMA_VERSION,
        "stage": 1,
        "personas_never_negative": personas,
        "channels": {
            "clamp": "per-persona Δz_EOS (trained − base), from breadth_contrast.json",
            "hijack": "per-persona marker emission rate over 20 questions, recomputed from "
            "the parent's HF raw completions (cross-checked vs four-float slot kinds)",
        },
        "distance_source": {
            "file": str(GEOMETRY_560.relative_to(PROJECT_ROOT)),
            "layer": 20,
            "n_probes": 50,
            "metric": "raw cosine distance (#560 min_dist)",
        },
        "distances": dist,
        "collinearity_pearson_dnn_dsrc": collinearity,
        "realized_spread": spread,
        "within_cell": within,
        "cross_arm": cross,
        "retest_472": retest,
        "hijack_recompute_diag": hijack_diag,
        "resolution": {
            "alpha": ALPHA,
            "sign_convention": {
                "channel_leakage_dir": CHANNEL_LEAKAGE_DIR,
                "registered_signs_leakage_units": {
                    "barrier": "+ (leakage rises with d_src controlling d_nn — shell convention)",
                    "bubble": "+ (leakage rises with d_nn controlling d_src; Δleakage moves "
                    "with Δd_nn)",
                },
                "rule": "expected raw-DV sign per read = channel_leakage_dir[channel]; a "
                "resolved read with the inverted sign is descriptive, never family-resolving",
            },
            "resolved_reads": resolved_reads,
            "inverted_reads": inverted_reads,
            "verdict": verdict,
            "narrow_within_cell": "pre-declared degenerate (collinearity 0.996) — "
            "descriptive only, excluded from the registered verdict",
            "emission_ceiling_note": "narrow-arm emission sits at ceiling (92–96%): the "
            "hijack channel carries information mainly in the broad arm; clamp is the "
            "primary channel",
        },
        "stage2_linkage": linkage,
        "inference": {
            "n_perm": N_PERM,
            "n_boot": N_BOOT,
            "seed": STAT_SEED,
            "holm_within_cell_family": "2 geometry reads per (cell, channel): "
            "{gradient_resid, barrier}",
            "holm_cross_arm_family": "2 pooled cross-arm channels {clamp, hijack} "
            "(conservative implementation choice)",
        },
        "input_revisions": {
            "geometry_560_git": _git_last_commit_of(GEOMETRY_560),
            "breadth_contrast_git": _git_last_commit_of(BREADTH_CONTRAST),
            "four_float_dir_git": _git_last_commit_of(FOUR_FLOAT_DIR),
            "raw_completions_hf_rev": RAW_COMPLETIONS_REV,
        },
        "metadata": {
            "task": 571,
            "followup_label": "persona-split-composition",
            "script": "issue571_psplit_stage1_analysis.py",
            "git_commit": _git_commit(),
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "argv": sys.argv[1:],
        },
    }
    out_path.write_text(json.dumps(payload, indent=1))
    logger.info("Stage-1 JSON written: %s", out_path)

    _figures(args.fig_dir, personas, leak, d_nn, d_src, d_dnn, dist, within, cross)

    logger.info(
        "STAGE-1 VERDICT: %s (%d family-resolved reads, %d inverted-sign reads); "
        "linkage fallback_objective=%s one_sided=%s",
        verdict,
        len(resolved_reads),
        len(inverted_reads),
        linkage["fallback_objective"],
        linkage["one_sided_expectation"],
    )
    return 0


def _figures(fig_dir, personas, leak, d_nn, d_src, d_dnn, dist, within, cross) -> None:
    """Stage-1 figure dump (§6): cross-arm hero, per-cell gradients, spread, nn table."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import savefig_paper, set_paper_style

    set_paper_style()
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Hero candidate — cross-arm paired identification scatter (both channels).
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, ch, ylabel in (
        (axes[0], "clamp", "Δ(Δz_EOS) broad − narrow, per persona"),
        (axes[1], "hijack", "Δ(hijack rate) broad − narrow, per persona"),
    ):
        delta = np.array(
            [
                np.mean([leak[ch][lb][p] for lb in ARMS["broad"]])
                - np.mean([leak[ch][lb][p] for lb in ARMS["narrow"]])
                for p in personas
            ]
        )
        ax.scatter(d_dnn, delta, s=22, color="#1f77b4", alpha=0.75)
        ax.axhline(0, color="#7f7f7f", lw=0.8, ls="--")
        ax.set_xlabel("Δd_nn (broad − narrow), #560 L20 raw cosine")
        ax.set_ylabel(ylabel)
    savefig_paper(fig, "stage1_cross_arm_paired", dir=fig_dir)
    plt.close(fig)

    # Within-cell gradient scatters: raw d_nn alongside residualized (per channel).
    for ch in ("clamp", "hijack"):
        fig, axes = plt.subplots(4, 3, figsize=(13, 14), sharey="row")
        for i, label in enumerate(ALL_LABELS):
            arm = "broad" if label.startswith("broad") else "narrow"
            vec = np.array([leak[ch][label][p] for p in personas])
            resid = quad_residuals(d_nn[arm], d_src)
            for j, (x, xlabel) in enumerate(
                (
                    (d_nn[arm], "d_nn (raw)"),
                    (resid, "resid(d_nn | quad d_src)"),
                    (d_src, "d_src"),
                )
            ):
                ax = axes[i][j]
                ax.scatter(x, vec, s=14, color="#1f77b4", alpha=0.7)
                ax.set_xlabel(xlabel)
                if j == 0:
                    ax.set_ylabel(f"{label}\n{'Δz_EOS' if ch == 'clamp' else 'hijack rate'}")
        savefig_paper(fig, f"stage1_within_cell_gradients_{ch}", dir=fig_dir)
        plt.close(fig)

    # Realized d_nn spread per arm + Δd_nn distribution.
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    axes[0].hist(d_nn["broad"], bins=12, color="#1f77b4", alpha=0.8)
    axes[0].set_xlabel("d_nn, broad arm")
    axes[1].hist(d_nn["narrow"], bins=12, color="#d62728", alpha=0.8)
    axes[1].set_xlabel("d_nn, narrow arm")
    axes[2].hist(d_dnn, bins=12, color="#7f7f7f", alpha=0.8)
    axes[2].set_xlabel("Δd_nn (broad − narrow)")
    for ax in axes:
        ax.set_ylabel("personas")
    savefig_paper(fig, "stage1_dnn_spread", dir=fig_dir)
    plt.close(fig)

    # Nearest-negative identity table (broad vs narrow).
    fig, ax = plt.subplots(figsize=(8, 11))
    ax.axis("off")
    rows = [
        [p, dist["nn_identity"]["broad"][p], dist["nn_identity"]["narrow"][p]] for p in personas
    ]
    table = ax.table(
        cellText=rows,
        colLabels=["persona", "nearest negative (broad)", "nearest negative (narrow)"],
        loc="center",
        cellLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    savefig_paper(fig, "stage1_nn_identity_table", dir=fig_dir)
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
