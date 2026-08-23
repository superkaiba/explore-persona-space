#!/usr/bin/env python
"""P6 gradient verdict for issue #2479 (plan v4 SS4 Step 6; verdict predicate SS3).

Zero-GPU, VM-side. Inputs (all produced by earlier phases):

- ``eval_results/issue_2479/panel.json``            -- 16-character registry (U2)
- ``eval_results/issue_2479/axis_freeze.json``      -- frozen AI-likeness axis
  (per-character score/rank + ``band_agreement_pass`` / ``axis_range_pass``)
- ``eval_results/issue_2479/instrument_gates.json`` -- ``verbatim_flatness_pass``
  + ``name_mask_pass``
- ``eval_results/issue_2479/story_char_gradient/``  -- per-cell ``cell_*.json`` +
  per-pair ``ladder_*.json`` in the #1345 Phase-F shape
  (``scripts/issue1345_story_char_ladder_fill.py`` output).

Computes, per character, the rung-4 recovery fraction
``r2["4_bias_refit"] / ceiling_r2`` read from the r4op->char_2479_<name>_op
ladder direction (reduced basis, instruct, context arm, layer 19, seed 0);
characters whose ladder-direction ``ceiling_r2`` < 0.05 are excluded from
fraction reads (pre-registered denominator-validity guard; raw rung-4 R2 is
kept for every surviving character). Headline: Spearman rho(frozen axis,
recovery fraction) with a 10,000-draw character-label permutation band
(vectorized numpy, one batch, seed 0) and a one-sided add-one Monte-Carlo p;
character-jackknife CI (descriptive only). The verdict LABEL is the pure
function :func:`verdict_label` implementing the SS3 canonical predicate
(unit-tested over the SS3 boundary grid in
``tests/test_issue2479_gradient_verdict.py``). Secondary reads each carry
their own labeled permutation band. Writes
``eval_results/issue_2479/gradient_verdict.json`` + figures via
``analysis.paper_plots.savefig_paper``.

Denominator convention (recorded as an assumption): the recovery-fraction
denominator is the ladder direction's OWN ``ceiling_r2`` -- computed on the
same matched rows and conversation-grouped folds as the rung R2, so the ratio
is internally consistent; the within-cell fit ceiling (``cell_*.json``) is
recorded per character as a companion field and drives the ceilings figure.

Usage:
    uv run python scripts/issue2479_gradient_verdict.py \
        [--eval-dir eval_results/issue_2479] [--fig-dir figures/issue_2479]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

# Shared-VM thread caps (#847) bind in-process only when load_dotenv() runs
# BEFORE the first heavy import (tests/test_shared_vm_thread_caps.py).
load_dotenv()

import numpy as np  # noqa: E402

# --- pre-registered constants (plan SS3 / SS4 Step 6 / SS6) -----------------
ISSUE = 2479
MODEL = "instruct"
ARM = "context"
LAYER = 19
BASIS = "reduced"
FIT_SEED = 0
SRC_OP = "r4op"  # on-policy assistant-story source operator
SRC_INSERTED = "r4"  # inserted assistant-story source operator
HEADLINE_RUNG = "4_bias_refit"
DIRECT_RUNG = "1_direct"
CEILING_FLOOR = 0.05  # ceiling R2 < 0.05 => excluded from fraction reads
N_PERM_DEFAULT = 10_000
PERM_SEED_DEFAULT = 0
ALPHA = 0.05
N_MIN_ESTABLISHED = 12  # final fraction-analysis n >= 12 required to establish
N_MIN_STAT = 2  # below 2 no statistic is estimable
GATE_KEYS = (
    "band_agreement_pass",
    "axis_range_pass",
    "verbatim_flatness_pass",
    "name_mask_pass",
)

LABEL_ESTABLISHED = "Gradient established"
LABEL_INSTRUMENT_SUSPECT = "Instrument-suspect"
LABEL_BOUNDED = "Gradient bounded"
LABEL_BOUNDED_PANEL = "Gradient bounded — insufficient panel at realized n"
LABEL_BOUNDED_NO_STAT = "Gradient bounded — no estimable statistic"
ALL_LABELS = (
    LABEL_ESTABLISHED,
    LABEL_INSTRUMENT_SUSPECT,
    LABEL_BOUNDED,
    LABEL_BOUNDED_PANEL,
    LABEL_BOUNDED_NO_STAT,
)

# Mirrors scripts/issue2479_char_panel.py::ANCHOR_NAMES (plan SS4 Step 1); kept
# as a literal so this module stays import-light for the SS3 unit test.
ANCHOR_DISPLAY_NAMES = ("HELIOS", "Wren", "Dana", "Vex")

# Keys probed on the op cell JSON basis block for the answer-length secondary
# read; none are emitted by the current fill script, so the read defers.
ANSWER_LEN_KEYS = ("mean_answer_len", "answer_len_mean", "mean_answer_tokens")
CLOSENESS_KEYS = ("mean_context_vec", "mean_answer_vec")


# --- SS3 canonical verdict predicate (pure function; unit-tested) ------------
def verdict_label(
    n_fraction_eligible: int,
    rho_obs: float | None,
    p_add_one: float | None,
    gates: dict[str, bool],
) -> str:
    """The SS3 canonical verdict label — a pure function of its four inputs.

    DISJOINT and exhaustive over the survivor axis (plan SS3):

    - ``Instrument-suspect``  <=> ANY of the four registered instrument gates
      fails (regardless of n / rho / p; data is still fully reported —
      the label demotes the claim, never suppresses the data).
    - gates pass, ``n < 2``   => ``Gradient bounded — no estimable statistic``.
    - gates pass, ``n < 12``  => ``Gradient bounded — insufficient panel at
      realized n`` (regardless of p).
    - gates pass, ``n >= 12``, ``rho_obs > 0`` and ``p_add_one <= 0.05``
      => ``Gradient established``.
    - otherwise               => ``Gradient bounded``.

    ``rho_obs`` / ``p_add_one`` may be None (or NaN) when no statistic is
    estimable; a None/NaN never establishes. Raises on a gates dict that does
    not carry exactly the four registered gate booleans (fail loud, never a
    silent default).
    """
    if set(gates) != set(GATE_KEYS):
        raise ValueError(f"gates must carry exactly {sorted(GATE_KEYS)}; got {sorted(gates)}")
    for k, v in gates.items():
        if not isinstance(v, bool):
            raise TypeError(f"gate {k!r} must be bool, got {type(v).__name__}")
    if not all(gates.values()):
        return LABEL_INSTRUMENT_SUSPECT
    if n_fraction_eligible < N_MIN_STAT:
        return LABEL_BOUNDED_NO_STAT
    if n_fraction_eligible < N_MIN_ESTABLISHED:
        return LABEL_BOUNDED_PANEL
    if (
        rho_obs is not None
        and p_add_one is not None
        and not (isinstance(rho_obs, float) and np.isnan(rho_obs))
        and not (isinstance(p_add_one, float) and np.isnan(p_add_one))
        and rho_obs > 0
        and p_add_one <= ALPHA
    ):
        return LABEL_ESTABLISHED
    return LABEL_BOUNDED


# --- Spearman + vectorized permutation machinery -----------------------------
def _rankdata(x: np.ndarray) -> np.ndarray:
    """Average-tie ranks (1..n), numpy-only and tie-order independent.

    Ties get the arithmetic mean of the ranks they span, so the result never
    depends on argsort's machine-dependent tie order (gotchas: argsort tie
    order is CPU-SIMD-kernel dependent).
    """
    x = np.asarray(x, dtype=np.float64)
    order = np.argsort(x, kind="stable")
    ranks = np.empty(len(x), dtype=np.float64)
    ranks[order] = np.arange(1, len(x) + 1, dtype=np.float64)
    _, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts), dtype=np.float64)
    np.add.at(sums, inv, ranks)
    return sums[inv] / counts[inv]


def _pearson_rows(x_rows: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Row-wise Pearson correlation of ``x_rows`` (B, n) against ``y`` (n,)."""
    xc = x_rows - x_rows.mean(axis=1, keepdims=True)
    yc = y - y.mean()
    num = xc @ yc
    den = np.sqrt((xc**2).sum(axis=1) * (yc**2).sum())
    return num / den


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Scalar Spearman rho (NaN when either vector has zero rank variance)."""
    rx, ry = _rankdata(x), _rankdata(y)
    if np.ptp(rx) == 0.0 or np.ptp(ry) == 0.0:
        return float("nan")
    return float(_pearson_rows(rx[None, :], ry)[0])


def spearman_perm_read(
    x: np.ndarray,
    y: np.ndarray,
    *,
    n_perm: int,
    seed: int,
    label: str,
    characters: list[str] | None = None,
) -> dict:
    """One labeled Spearman read with its own vectorized permutation band.

    ``x`` = axis scores, ``y`` = the read's per-character values. The null
    permutes the AXIS ranks via a single (n_perm, n) index matrix — no Python
    loop over draws — and the one-sided add-one Monte-Carlo p is
    ``(1 + #{rho_null >= rho_obs}) / (n_perm + 1)``.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    assert x.shape == y.shape, (x.shape, y.shape)
    n = int(len(x))
    out: dict = {"label": label, "n": n, "n_perm": int(n_perm), "seed": int(seed)}
    if characters is not None:
        out["characters"] = list(characters)
        out["values"] = [float(v) for v in y]
        out["axis_scores"] = [float(v) for v in x]
    if n < N_MIN_STAT:
        out.update(rho=None, p_add_one=None, status="no estimable statistic (n < 2)")
        return out
    rx, ry = _rankdata(x), _rankdata(y)
    if np.ptp(rx) == 0.0 or np.ptp(ry) == 0.0:
        out.update(rho=None, p_add_one=None, status="degenerate — zero rank variance")
        return out
    rho = float(_pearson_rows(rx[None, :], ry)[0])
    rng = np.random.default_rng(seed)
    # (n_perm, n) permutation index matrix, one batch (plan SS4 Step 6).
    perm_idx = np.argsort(rng.random((n_perm, n)), axis=1)
    null = _pearson_rows(rx[perm_idx], ry)
    n_ge = int((null >= rho).sum())
    out.update(
        rho=rho,
        n_null_ge=n_ge,
        p_add_one=float((1 + n_ge) / (n_perm + 1)),
        null_q95=float(np.quantile(null, 0.95)),
        null_mean=float(null.mean()),
        status="ok",
    )
    return out


def jackknife_ci(x: np.ndarray, y: np.ndarray) -> dict:
    """Leave-one-character-out jackknife CI on Spearman rho (DESCRIPTIVE only).

    The permutation band is the verdict instrument (plan SS6); rank
    discontinuities make small-n Spearman intervals poorly calibrated.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = len(x)
    if n < 3:
        return {"status": "n < 3 — jackknife undefined"}
    rhos = []
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        rhos.append(_spearman(x[mask], y[mask]))
    rhos_arr = np.asarray(rhos, dtype=np.float64)
    if np.isnan(rhos_arr).any():
        return {
            "status": "degenerate leave-one-out subset (zero rank variance)",
            "leave_one_out_rho": [None if np.isnan(r) else float(r) for r in rhos_arr],
        }
    mean = float(rhos_arr.mean())
    se = float(np.sqrt((n - 1) / n * ((rhos_arr - mean) ** 2).sum()))
    rho_full = _spearman(x, y)
    return {
        "status": "ok",
        "rho": rho_full,
        "se_jackknife": se,
        "ci95": [rho_full - 1.96 * se, rho_full + 1.96 * se],
        "leave_one_out_rho": [float(r) for r in rhos_arr],
        "note": "descriptive only — the permutation band is the verdict instrument",
    }


# --- input loading ------------------------------------------------------------
def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_json(path: Path):
    return json.loads(path.read_text())


def _scalar_layer(v, what: str) -> float:
    """Single-layer per-layer list -> scalar (this driver runs L=1 slices)."""
    assert isinstance(v, list) and len(v) == 1, (
        f"{what}: expected a single-layer per-layer list, got {v!r}"
    )
    return float(v[0])


def _ladder_direction(entry: dict, src: str, tgt: str, path: Path) -> dict:
    """The src->tgt direction block of a #1345 Phase-F ladder JSON (reduced)."""
    labels = entry["metadata"]["regime_labels"]
    key = f"{labels[src]}->{labels[tgt]}"
    block = entry[BASIS]
    if key not in block:
        raise KeyError(
            f"{path.name}: direction {key!r} absent from {BASIS!r} block "
            f"(direction keys: {sorted(k for k in block if '->' in str(k))})"
        )
    return block[key]


def _acc1(knn: dict, key: str) -> tuple[float | None, float | None]:
    """(acc@1, chance@1) for one knn_retrieval_fold0 entry, or (None, None)."""
    blk = knn.get(key)
    if not isinstance(blk, dict) or "acc@1" not in blk:
        return None, None
    acc = blk["acc@1"]
    acc_val = _scalar_layer(acc, f"knn[{key}].acc@1") if isinstance(acc, list) else float(acc)
    chance = blk.get("chance@1")
    return acc_val, (float(chance) if chance is not None else None)


_EQN_TAG_RE = re.compile(r"_nd\d+(?P<tag>.+)\.json$")


def _primary_and_eqn(grad_dir: Path, pattern: str) -> tuple[Path | None, dict[str, Path]]:
    """Resolve the primary (untagged) ladder file + equalized-n companions.

    Equalized-n companion convention (defined here, plan SS6 Unit-5 scope): a
    ladder filename carrying a suffix tag after the ``_nd<K>`` token — either
    the fit driver's ``_rows<N>`` subsample tag or an explicit ``__eqn*`` tag —
    is an equalized-n companion; companions are grouped by the exact tag.
    """
    primary: Path | None = None
    eqn: dict[str, Path] = {}
    for p in sorted(grad_dir.glob(pattern)):
        m = _EQN_TAG_RE.search(p.name)
        tag = m.group("tag") if m else None
        if tag:
            eqn[tag] = p
        else:
            if primary is not None:
                raise FileExistsError(
                    f"multiple primary ladder files match {pattern!r}: {primary.name}, {p.name}"
                )
            primary = p
    return primary, eqn


def collect_characters(panel: list[dict], axis: dict, grad_dir: Path) -> dict:
    """Load per-character fit/ladder reads + the three-stage exclusion lists."""
    axis_chars = axis["characters"]
    per_char: dict[str, dict] = {}
    not_in_axis: list[str] = []
    missing_fit: list[dict] = []
    input_files: dict[str, str] = {}

    for row in panel:
        name = row["name"]
        display = row.get("display_name", name)
        vop = row["variant_op"]
        vins = row.get("variant_inserted")
        if name not in axis_chars:
            not_in_axis.append(name)
            continue
        ax = axis_chars[name]
        rec: dict = {
            "name": name,
            "display_name": display,
            "design_band": row["design_band"],
            "anchor": display in ANCHOR_DISPLAY_NAMES,
            "variant_op": vop,
            "variant_inserted": vins,
            "axis_score": float(ax["score"]),
            "axis_rank": int(ax["rank"]),
        }

        ladder_pat = f"ladder_{SRC_OP}__{vop}__{MODEL}_{ARM}_L{LAYER}_{BASIS}_s{FIT_SEED}_nd*.json"
        ladder_path, eqn_paths = _primary_and_eqn(grad_dir, ladder_pat)
        cell_path = grad_dir / f"cell_{vop}__{MODEL}_{ARM}_L{LAYER}_{BASIS}_s{FIT_SEED}.json"
        missing = [
            what
            for what, ok in (("ladder", ladder_path is not None), ("cell", cell_path.is_file()))
            if not ok
        ]
        if missing:
            missing_fit.append({"name": name, "missing": missing})
            continue

        assert ladder_path is not None
        entry = _load_json(ladder_path)
        input_files[str(ladder_path)] = _sha256(ladder_path)
        d = _ladder_direction(entry, SRC_OP, vop, ladder_path)
        ceiling = _scalar_layer(d["ceiling_r2"], f"{ladder_path.name}: ceiling_r2")
        rec.update(
            ladder_file=ladder_path.name,
            ceiling_r2=ceiling,
            rung4_r2=_scalar_layer(d["r2"][HEADLINE_RUNG], f"{ladder_path.name}: rung4"),
            rung1_r2=_scalar_layer(d["r2"][DIRECT_RUNG], f"{ladder_path.name}: rung1"),
            identity_bias_r2=_scalar_layer(
                d["identity_bias_r2"], f"{ladder_path.name}: identity_bias_r2"
            ),
            null4_r2=_scalar_layer(d["null_r2"][HEADLINE_RUNG], f"{ladder_path.name}: null4"),
            n_matched=entry.get("n_matched"),
        )
        knn = d.get("knn_retrieval_fold0") or {}
        rec["acc1_ceiling"], rec["acc1_chance"] = _acc1(knn, "ceiling")
        rec["acc1_rung4"], _ = _acc1(knn, HEADLINE_RUNG)
        rec["acc1_ceiling_cosine"], _ = _acc1(knn, "ceiling_cosine")
        rec["acc1_rung4_cosine"], _ = _acc1(knn, f"{HEADLINE_RUNG}_cosine")

        centry = _load_json(cell_path)
        input_files[str(cell_path)] = _sha256(cell_path)
        cblk = centry[BASIS]
        rec.update(
            cell_file=cell_path.name,
            cell_ceiling_r2=_scalar_layer(cblk["ceiling_r2"], f"{cell_path.name}: ceiling"),
            cell_identity_bias_r2=_scalar_layer(
                cblk["identity_bias_r2"], f"{cell_path.name}: identity_bias"
            ),
            cell_n=int(cblk["n"]),
        )
        rec["cell_extra_keys"] = sorted(
            k for k in cblk if k in ANSWER_LEN_KEYS or k in CLOSENESS_KEYS
        )

        rec["fraction_eligible"] = bool(ceiling >= CEILING_FLOOR)
        rec["recovery_fraction"] = rec["rung4_r2"] / ceiling if rec["fraction_eligible"] else None
        rec["eqn_ladder_files"] = {tag: p.name for tag, p in sorted(eqn_paths.items())}
        rec["_eqn_paths"] = eqn_paths  # not serialized; stripped before write

        if vins:
            ins_pat = (
                f"ladder_{SRC_INSERTED}__{vins}__{MODEL}_{ARM}_L{LAYER}_{BASIS}"
                f"_s{FIT_SEED}_nd*.json"
            )
            ins_path, _ = _primary_and_eqn(grad_dir, ins_pat)
            if ins_path is not None:
                ins_entry = _load_json(ins_path)
                input_files[str(ins_path)] = _sha256(ins_path)
                di = _ladder_direction(ins_entry, SRC_INSERTED, vins, ins_path)
                ins_ceiling = _scalar_layer(di["ceiling_r2"], f"{ins_path.name}: ceiling")
                rec["inserted"] = {
                    "ladder_file": ins_path.name,
                    "ceiling_r2": ins_ceiling,
                    "rung4_r2": _scalar_layer(di["r2"][HEADLINE_RUNG], f"{ins_path.name}: rung4"),
                    "fraction_eligible": bool(ins_ceiling >= CEILING_FLOOR),
                    "recovery_fraction": (
                        _scalar_layer(di["r2"][HEADLINE_RUNG], ins_path.name) / ins_ceiling
                        if ins_ceiling >= CEILING_FLOOR
                        else None
                    ),
                }
        per_char[name] = rec

    return {
        "per_char": per_char,
        "not_in_axis": not_in_axis,
        "missing_fit": missing_fit,
        "input_files": input_files,
    }


# --- reads assembly -----------------------------------------------------------
def _read_over(
    per_char: dict[str, dict],
    value_fn,
    *,
    label: str,
    n_perm: int,
    seed: int,
) -> dict:
    """Build one labeled read over the characters where ``value_fn`` is not None."""
    rows = []
    for name in sorted(per_char):
        rec = per_char[name]
        v = value_fn(rec)
        if v is None:
            continue
        rows.append((name, rec["axis_score"], float(v)))
    if not rows:
        return {"label": label, "n": 0, "status": "no eligible characters"}
    names = [r[0] for r in rows]
    x = np.array([r[1] for r in rows])
    y = np.array([r[2] for r in rows])
    return spearman_perm_read(x, y, n_perm=n_perm, seed=seed, label=label, characters=names)


def _safe_ratio(num: float | None, den: float | None) -> float | None:
    if num is None or den is None or den <= 0:
        return None
    return num / den


def build_secondary_reads(per_char: dict[str, dict], *, n_perm: int, seed: int) -> dict:
    """The SS6 secondary/exploratory reads, each with its own labeled band."""
    reads: dict[str, dict] = {}

    def frac(rec, num_key):
        if not rec["fraction_eligible"]:
            return None
        return rec[num_key] / rec["ceiling_r2"]

    # acc@1-based recovery is a FRACTION read: the pre-registered ceiling-R2
    # exclusion applies (plan SS4 Step 6 "excluded from fraction reads"), plus
    # the acc-side denominator guard (ceiling acc@1 must be > 0).
    reads["acc1_recovery_euclidean"] = _read_over(
        per_char,
        lambda r: (
            _safe_ratio(r.get("acc1_rung4"), r.get("acc1_ceiling"))
            if r["fraction_eligible"]
            else None
        ),
        label="rho(axis, acc@1 rung-4 / acc@1 ceiling), euclidean fold-0",
        n_perm=n_perm,
        seed=seed,
    )
    reads["acc1_recovery_cosine"] = _read_over(
        per_char,
        lambda r: (
            _safe_ratio(r.get("acc1_rung4_cosine"), r.get("acc1_ceiling_cosine"))
            if r["fraction_eligible"]
            else None
        ),
        label="rho(axis, acc@1 rung-4 / acc@1 ceiling), cosine fold-0",
        n_perm=n_perm,
        seed=seed,
    )
    reads["raw_rung4_r2"] = _read_over(
        per_char,
        lambda r: r["rung4_r2"],
        label="rho(axis, raw rung-4 R2) — every surviving character, no ceiling exclusion",
        n_perm=n_perm,
        seed=seed,
    )
    reads["rung1_direct_recovery"] = _read_over(
        per_char,
        lambda r: frac(r, "rung1_r2"),
        label="rho(axis, rung-1 direct-transfer recovery fraction)",
        n_perm=n_perm,
        seed=seed,
    )
    reads["inserted_mode_recovery"] = _read_over(
        per_char,
        lambda r: (r.get("inserted") or {}).get("recovery_fraction"),
        label="rho(axis, inserted-mode rung-4 recovery fraction) — n<=8; a null here is "
        "inconclusive about answer-content mediation, never rules it out",
        n_perm=n_perm,
        seed=seed,
    )
    reads["identity_bias_recovery"] = _read_over(
        per_char,
        lambda r: frac(r, "identity_bias_r2"),
        label="rho(axis, identity+learned-bias R2 / ceiling) — trivial-baseline control",
        n_perm=n_perm,
        seed=seed,
    )
    reads["ceiling_vs_axis"] = _read_over(
        per_char,
        lambda r: r["ceiling_r2"],
        label="rho(axis, own-ceiling R2) — denominator confound read",
        n_perm=n_perm,
        seed=seed,
    )
    reads["null_gradient_matched_capacity"] = _read_over(
        per_char,
        lambda r: frac(r, "null4_r2"),
        label="rho(axis, matched-capacity null rung-4 R2 / ceiling) — target-easiness "
        "common-cause read",
        n_perm=n_perm,
        seed=seed,
    )
    reads["new_characters_only"] = _read_over(
        {k: v for k, v in per_char.items() if not v["anchor"]},
        lambda r: r["recovery_fraction"],
        label="headline restricted to the 12 non-anchor characters "
        "(anchor-contamination robustness)",
        n_perm=n_perm,
        seed=seed,
    )
    reads["kept_n_vs_axis"] = _read_over(
        per_char,
        lambda r: float(r["cell_n"]),
        label="rho(axis, per-character kept fit rows n) — retention-mediation diagnostic",
        n_perm=n_perm,
        seed=seed,
    )

    # Answer-length + closeness reads: computable only from richer per-row /
    # activation data the cell JSONs do not carry (checked, not assumed).
    has_len = any(
        k in (rec.get("cell_extra_keys") or [])
        for rec in per_char.values()
        for k in ANSWER_LEN_KEYS
    )
    if not has_len:
        reads["answer_length_vs_axis"] = {
            "status": "deferred",
            "note": "requires kept-row bundle — computed at analyzer time",
        }
    has_vecs = any(
        k in (rec.get("cell_extra_keys") or []) for rec in per_char.values() for k in CLOSENESS_KEYS
    )
    if not has_vecs:
        note = "requires turnstore-derived mean-activation vectors — computed at analyzer time"
        reads["context_space_closeness_vs_axis"] = {"status": "deferred", "note": note}
        reads["answer_space_closeness_vs_axis"] = {"status": "deferred", "note": note}
    return reads


def build_equalized_n(per_char: dict[str, dict], *, n_perm: int, seed: int) -> dict:
    """Equalized-n headline companions from tagged companion ladder files."""
    by_tag: dict[str, dict[str, dict]] = {}
    for name, rec in per_char.items():
        for tag, path in (rec.get("_eqn_paths") or {}).items():
            entry = _load_json(path)
            d = _ladder_direction(entry, SRC_OP, rec["variant_op"], path)
            ceiling = _scalar_layer(d["ceiling_r2"], f"{path.name}: ceiling")
            rung4 = _scalar_layer(d["r2"][HEADLINE_RUNG], f"{path.name}: rung4")
            by_tag.setdefault(tag, {})[name] = {
                "axis_score": rec["axis_score"],
                "anchor": rec["anchor"],
                "fraction_eligible": bool(ceiling >= CEILING_FLOOR),
                "recovery_fraction": rung4 / ceiling if ceiling >= CEILING_FLOOR else None,
                "ladder_file": path.name,
            }
    if not by_tag:
        return {"status": "not_produced", "note": "equalized-n companion not yet produced"}
    out: dict = {"status": "ok", "companions": {}}
    for tag, chars in sorted(by_tag.items()):
        out["companions"][tag] = _read_over(
            chars,
            lambda r: r["recovery_fraction"],
            label=f"equalized-n headline companion (tag {tag!r})",
            n_perm=n_perm,
            seed=seed,
        )
    return out


# --- figures -------------------------------------------------------------------
def _hero_scatter(pp, plt, rows: list[dict], ylabel: str, stem: str, fig_dir: Path) -> Path:
    """Axis-score-vs-value labeled scatter; anchors visually distinguished."""
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    c_new = pp.paper_palette_role("primary")
    c_anchor = pp.paper_palette_role("accent")
    for is_anchor, color, marker, lab in (
        (False, c_new, "o", "new character"),
        (True, c_anchor, "D", "anchor"),
    ):
        sub = [r for r in rows if r["anchor"] == is_anchor]
        if not sub:
            continue
        ax.scatter(
            [r["x"] for r in sub],
            [r["y"] for r in sub],
            c=color,
            marker=marker,
            s=42,
            label=lab,
            zorder=3,
        )
    for r in rows:
        ax.annotate(
            r["display_name"],
            (r["x"], r["y"]),
            textcoords="offset points",
            xytext=(4, 4),
            fontsize=7,
        )
    ax.set_xlabel("AI-likeness axis score (judge, 0-100)")
    ax.set_ylabel(ylabel)
    ax.legend(frameon=False)
    paths = pp.savefig_paper(fig, stem, dir=fig_dir)
    plt.close(fig)
    return paths["png"]


def make_figures(per_char: dict[str, dict], fig_dir: Path) -> dict:
    """The plan-SS6 figure set; data-driven and tolerant of missing optional reads."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis import paper_plots as pp

    pp.set_paper_style("generic")
    fig_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, str] = {}
    skipped: dict[str, str] = {}

    def rows_for(value_fn) -> list[dict]:
        rows = []
        for name in sorted(per_char):
            rec = per_char[name]
            v = value_fn(rec)
            if v is None:
                continue
            rows.append(
                {
                    "display_name": rec["display_name"],
                    "x": rec["axis_score"],
                    "y": float(v),
                    "anchor": rec["anchor"],
                }
            )
        return rows

    specs = [
        (
            "gradient_hero",
            lambda r: r["recovery_fraction"],
            "Rung-4 recovery fraction (R2 rung-4 / R2 ceiling)",
        ),
        (
            "gradient_hero_acc1",
            lambda r: (
                _safe_ratio(r.get("acc1_rung4"), r.get("acc1_ceiling"))
                if r["fraction_eligible"]
                else None
            ),
            "acc@1 recovery fraction (rung-4 / ceiling, euclidean)",
        ),
        (
            "gradient_hero_inserted",
            lambda r: (r.get("inserted") or {}).get("recovery_fraction"),
            "Inserted-mode rung-4 recovery fraction",
        ),
        (
            "gradient_null_companion",
            lambda r: (
                _safe_ratio(r["null4_r2"], r["ceiling_r2"]) if r["fraction_eligible"] else None
            ),
            "Matched-capacity null rung-4 recovery fraction",
        ),
    ]
    for stem, fn, ylabel in specs:
        rows = rows_for(fn)
        if not rows:
            skipped[stem] = "no characters with this read"
            continue
        written[stem] = str(_hero_scatter(pp, plt, rows, ylabel, stem, fig_dir))

    # Ceilings + identity+bias bars per character (sorted by axis score).
    recs = sorted(per_char.values(), key=lambda r: -r["axis_score"])
    if recs:
        fig, ax = plt.subplots(figsize=(7.0, 4.0))
        xs = np.arange(len(recs))
        width = 0.4
        ax.bar(
            xs - width / 2,
            [r["ceiling_r2"] for r in recs],
            width,
            label="own-ceiling R2 (ladder direction)",
            color=pp.paper_palette_role("primary"),
        )
        ax.bar(
            xs + width / 2,
            [r["identity_bias_r2"] for r in recs],
            width,
            label="identity + learned-bias R2",
            color=pp.paper_palette_role("baseline"),
        )
        ax.axhline(CEILING_FLOOR, color="0.4", lw=0.8, ls="--")
        ax.set_xticks(xs)
        ax.set_xticklabels([r["display_name"] for r in recs], rotation=60, ha="right")
        ax.set_ylabel("Held-out R2")
        ax.legend(frameon=False)
        paths = pp.savefig_paper(fig, "ceilings_identity_bias", dir=fig_dir)
        plt.close(fig)
        written["ceilings_identity_bias"] = str(paths["png"])
    else:
        skipped["ceilings_identity_bias"] = "no surviving characters"

    return {"written": written, "skipped": skipped}


# --- assembly + CLI -------------------------------------------------------------
def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")
    os.replace(tmp, path)


def build_verdict(
    *,
    panel_path: Path,
    axis_path: Path,
    gates_path: Path,
    grad_dir: Path,
    n_perm: int,
    perm_seed: int,
) -> dict:
    """Assemble the full gradient_verdict.json payload (no figures)."""
    panel = _load_json(panel_path)
    axis = _load_json(axis_path)
    inst = _load_json(gates_path)

    gates = {
        "band_agreement_pass": bool(axis["gates"]["band_agreement_pass"]),
        "axis_range_pass": bool(axis["gates"]["axis_range_pass"]),
        "verbatim_flatness_pass": bool(inst["gates"]["verbatim_flatness_pass"]),
        "name_mask_pass": bool(inst["gates"]["name_mask_pass"]),
    }

    col = collect_characters(panel, axis, grad_dir)
    per_char = col["per_char"]
    eligible = {k: v for k, v in per_char.items() if v["fraction_eligible"]}
    ceiling_excluded = [
        {"name": k, "ceiling_r2": v["ceiling_r2"]}
        for k, v in sorted(per_char.items())
        if not v["fraction_eligible"]
    ]

    denominators = {
        "planned": len(panel),
        "g1_surviving": len(per_char),
        "fraction_eligible": len(eligible),
    }

    headline = _read_over(
        eligible,
        lambda r: r["recovery_fraction"],
        label="HEADLINE: rho(frozen AI-likeness axis, rung-4 recovery fraction) — "
        "on-policy mode, instruct, context arm, layer 19, reduced basis",
        n_perm=n_perm,
        seed=perm_seed,
    )
    if headline.get("rho") is not None:
        x = np.array([eligible[n]["axis_score"] for n in headline["characters"]])
        y = np.array([eligible[n]["recovery_fraction"] for n in headline["characters"]])
        headline["jackknife"] = jackknife_ci(x, y)

    label = verdict_label(
        denominators["fraction_eligible"], headline.get("rho"), headline.get("p_add_one"), gates
    )

    secondary = build_secondary_reads(per_char, n_perm=n_perm, seed=perm_seed)
    equalized = build_equalized_n(per_char, n_perm=n_perm, seed=perm_seed)

    from explore_persona_space.orchestrate.provenance import as_metadata_dict, git_provenance

    input_files = {
        str(panel_path): _sha256(panel_path),
        str(axis_path): _sha256(axis_path),
        str(gates_path): _sha256(gates_path),
        **col["input_files"],
    }
    # strip the non-serializable Path helper field
    for rec in per_char.values():
        rec.pop("_eqn_paths", None)

    return {
        "issue": ISSUE,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "metadata": {
            "script": "scripts/issue2479_gradient_verdict.py",
            "numpy_version": np.__version__,
            **as_metadata_dict(git_provenance(), phase="gradient-verdict"),
        },
        "conventions": {
            "model": MODEL,
            "arm": ARM,
            "layer": LAYER,
            "basis": BASIS,
            "fit_seed": FIT_SEED,
            "headline_rung": HEADLINE_RUNG,
            "ceiling_floor": CEILING_FLOOR,
            "recovery_denominator": "ladder-direction own ceiling_r2 (same matched rows + "
            "folds as the rung R2); cell-fit ceiling recorded as companion",
            "alpha": ALPHA,
            "n_min_established": N_MIN_ESTABLISHED,
        },
        "seeds": {"perm_seed": perm_seed, "n_perm": n_perm},
        "inputs_sha256": input_files,
        "gates": gates,
        "denominators": denominators,
        "exclusions": {
            "not_in_axis_freeze": sorted(col["not_in_axis"]),
            "missing_fit_outputs": sorted(col["missing_fit"], key=lambda d: d["name"]),
            "ceiling_excluded": ceiling_excluded,
        },
        "verdict": {
            "label": label,
            "n_fraction_eligible": denominators["fraction_eligible"],
            "rho_obs": headline.get("rho"),
            "p_add_one": headline.get("p_add_one"),
            "predicate": "plan SS3 canonical verdict predicate (pure function verdict_label)",
        },
        "headline": headline,
        "secondary_reads": secondary,
        "equalized_n": equalized,
        "per_character": per_char,
    }


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint: build gradient_verdict.json + figures."""
    ap = argparse.ArgumentParser(
        description="issue #2479 P6 gradient verdict (plan SS4 Step 6)",
    )
    ap.add_argument("--eval-dir", type=Path, default=Path("eval_results/issue_2479"))
    ap.add_argument("--grad-dir", type=Path, default=None)
    ap.add_argument("--panel", type=Path, default=None)
    ap.add_argument("--axis-freeze", type=Path, default=None)
    ap.add_argument("--instrument-gates", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--fig-dir", type=Path, default=Path("figures/issue_2479"))
    ap.add_argument("--n-perm", type=int, default=N_PERM_DEFAULT)
    ap.add_argument("--perm-seed", type=int, default=PERM_SEED_DEFAULT)
    ap.add_argument("--no-figures", action="store_true", help="skip figure rendering")
    args = ap.parse_args(argv)

    eval_dir: Path = args.eval_dir
    payload = build_verdict(
        panel_path=args.panel or eval_dir / "panel.json",
        axis_path=args.axis_freeze or eval_dir / "axis_freeze.json",
        gates_path=args.instrument_gates or eval_dir / "instrument_gates.json",
        grad_dir=args.grad_dir or eval_dir / "story_char_gradient",
        n_perm=args.n_perm,
        perm_seed=args.perm_seed,
    )
    if not args.no_figures:
        payload["figures"] = make_figures(payload["per_character"], args.fig_dir)

    out = args.out or eval_dir / "gradient_verdict.json"
    _atomic_write_json(out, payload)

    d = payload["denominators"]
    v = payload["verdict"]
    print(
        f"[verdict] {v['label']!r}  rho={v['rho_obs']}  p_add_one={v['p_add_one']}  "
        f"n={d['planned']}/{d['g1_surviving']}/{d['fraction_eligible']} "
        f"(planned/G1-surviving/fraction-eligible) -> {out}",
        flush=True,
    )
    for name, read in payload["secondary_reads"].items():
        if read.get("status") == "deferred":
            print(f"[secondary] {name}: DEFERRED — {read['note']}", flush=True)
        else:
            print(
                f"[secondary] {name}: n={read.get('n')} rho={read.get('rho')} "
                f"p={read.get('p_add_one')}",
                flush=True,
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
