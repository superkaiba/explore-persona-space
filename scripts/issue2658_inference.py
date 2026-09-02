"""Issue #2658 unit 11 — preregistered confirmatory inference (plan section 7).

CPU-only, no GPU, no judge API call. Consumes the frozen outputs of earlier
units and produces the plan section 7 confirmatory report:

- PRIMARY metric: equal-prompt macro within-prompt AUROC among prompts
  realizing both classes — computed EXCLUSIVELY through unit 8's registered
  primitives (``equal_prompt_macro_auroc`` / ``permute_labels_within_prompt``
  / ``within_prompt_auroc``, imported and aliased below; tests pin the alias
  by IDENTITY). Global AUROC is DESCRIPTIVE only, with hierarchical
  prompt/response bootstrap uncertainty.
- C2 and the once-fitted/frozen C5 are tested by permuting ONLY sealed-test
  labels WITHIN EXACT PROMPT: 9,999 plus-one permutations initially, with
  DETERMINISTIC family-wide extension to 99,999 when any test's Monte Carlo
  interval overlaps any Holm threshold of its family. Extension is
  FAMILY-WIDE by construction (extending only borderline tests would make the
  family's adjusted p-values depend on which tests happened to be borderline).
- C5-minus-C2: ONE-SIDED studentized paired hierarchical bootstrap with
  development Bayesian-bootstrap refitting (Dirichlet weights over dev
  prompts; weighted z-score + weighted logistic refit at the FROZEN selected
  C) and test prompt / within-class response resampling. The bootstrap family
  carries a unit-11 PRE-REGISTERED family-wide deterministic extension
  (2,000 -> 20,000 draws on the same MC-overlap trigger) — an ADDITION to
  plan section 7 (which registers the 99,999 extension for the permutation
  families only), recorded before any sealed-test labels exist; derivation in
  :class:`InferenceRegistry`.
- THREE Holm families at alpha=0.05 with family sizes DERIVED from the
  committed ``direction_provenance.json`` partition (realized: C2=10, C5=11,
  C5-minus-C2=10) — never hardcoded. Intervals are POINTWISE; adjusted
  p-values and significance are reported SEPARATELY from the intervals.
- The frame manifest's ``prospective_not_estimable_ledger`` (unit 10) drives
  per-row cell exclusion: ledger-flagged cells are excluded from the row's
  prompt panel, the revised denominator is RECORDED per row, and a missing
  ledger REFUSES loudly (never "all cells estimable" by default). After final
  labels, the FULL plan section 8 production gate set is enforced: the
  REALIZED >=15-discordant-prompts-PER-CELL floor (below-floor retained cells
  are excluded with cause ``realized-discordance-below-floor``, the denominator
  revised, and the row RE-GATED on the reduced panel — no new test rows are
  ever added); >=100 discordant prompts overall; >=100 answers and >=30
  prompts in each class; COMPLETE LABELS (zero non-scored/non-labeled label
  records after the plan section 3 retry -> human-adjudication chain, with
  per-status counts recorded either way); PASSED TEST-BANK LABEL RELIABILITY
  (:func:`load_test_label_reliability`; objective-label rows exempt, stated in
  the gate record); and no cross-split lineage (enforced at assembly —
  ``DependencyCrossingError`` in ``issue2658_comparators.assemble_row_data``).
  Any failure returns not-estimable for that row, naming the failed gate —
  never a substituted proxy, never a silent narrowing.

Determinism + resume: every permutation / bootstrap CHUNK is an independently
SHA-seeded unit persisted to an atomic-append JSONL ledger the moment it
completes, keyed on GENERATING PARAMETERS only (artifact shas, string/int
panel fingerprints, chunk bounds — never the bytes of a recomputed float
array). The family-wide extension therefore genuinely RE-USES the first 9,999
draws: they are the same deterministic per-chunk seeded draws, and their
persisted exceedance counts are summed into the extended p-value without
redrawing.

Registered constants the plan leaves open are frozen in
:class:`InferenceRegistry` (derivations in the class docstring) and echoed
into the report.

Ops arithmetic (permutation battery): tests(<=21) x n_perm(9,999 -> 99,999) x
n_test_rows(~10,800 at the production floor) label-permute + macro evals,
chunked to ``CHUNK_ELEMENT_BUDGET`` elements per transient. Bootstrap battery:
rows(10) x n_boot(2,000) weighted logistic refits on the dev panel (the
serial, sklearn-bound inner cost — MEASURED via ``--measure-boot-unit``,
never asserted) + a fully vectorized instance-AUROC resampling pass.

Launch (VM-side runs carry the shared-VM thread caps):
    OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \
    NUMEXPR_NUM_THREADS=8 MALLOC_ARENA_MAX=2 \
    uv run python scripts/issue2658_inference.py --smoke
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()  # shared-VM thread caps (#847) must bind BEFORE numpy/scipy import

import numpy as np  # noqa: E402
from scipy import stats as sps  # noqa: E402

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import issue2658_common as C  # noqa: E402
import issue2658_comparators as U  # noqa: E402
import issue2658_frames as F  # noqa: E402
import issue2658_power as PW  # noqa: E402
from explore_persona_space.atomic_io import write_json_atomic  # noqa: E402
from explore_persona_space.orchestrate.provenance import (  # noqa: E402
    as_metadata_dict,
    git_provenance,
)

# ---------------------------------------------------------------------------
# Registered estimator primitives — IMPORTED from unit 8, never re-implemented.
# The power simulation that sized N called exactly these functions; unit-11
# tests assert these aliases by IDENTITY (``is``), not by resemblance.
# ---------------------------------------------------------------------------
MACRO_AUROC = PW.equal_prompt_macro_auroc
PERMUTE_WITHIN_PROMPT = PW.permute_labels_within_prompt
WITHIN_PROMPT_AUROC = PW.within_prompt_auroc

REPORT_SCHEMA = "i2658-inference-report-v1"
PERM_CHUNK_SCHEMA = "i2658-inference-perm-chunk-v1"
BOOT_CHUNK_SCHEMA = "i2658-inference-boot-chunk-v1"
SEED_NAMESPACE = "i2658-inference-v1"

# One-sided exceedance tolerance — mirrors unit 8's simulate_power comparison.
EXCEED_TOL = 1e-12


class InferenceInputError(C.Issue2658GuardError):
    """Malformed / absent / drifted inference input (fail fast, no default)."""


# ---------------------------------------------------------------------------
# Registered constants (plan section 7 leaves these open; frozen HERE and
# echoed into the report so the choices are re-decidable).
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class InferenceRegistry:
    """FROZEN registered inference constants (plan section 7).

    Derivations:

    - ``alpha`` / ``n_perm_initial`` / ``n_perm_extended``: plan section 7 via
      ``issue2658_common.HOLM`` (0.05 / 9,999 / 99,999).
    - ``perm_chunk_initial = (1111,)*9`` (sum 9,999) and
      ``perm_chunk_extension = (2000,)*45`` (sum 90,000): the chunk plan is a
      registered constant so every chunk's SHA-derived seed — hence every
      draw — is deterministic, and the 99,999-permutation extension is the
      UNION of the initial chunks (reused from the ledger, never redrawn) and
      the extension chunks.
    - ``mc_conf = 0.99``: two-sided Clopper-Pearson confidence for the Monte
      Carlo interval on each permutation p (conservative overlap trigger).
    - ``n_boot = 2,000`` / ``boot_chunk = 250``: the plan registers no
      bootstrap draw count. The binding cost is 2,000 weighted dev logistic
      REFITS per row (serial sklearn), sized against the plan section 10 P6
      12h wall via the MEASURED ``--measure-boot-unit`` pilot.
    - ``n_boot_extended = 20,000``: unit-11 PRE-REGISTERED family-wide
      deterministic extension of the bootstrap family — an ADDITION to plan
      section 7 (which registers the 99,999 extension for the PERMUTATION
      families only), recorded before any sealed-test labels exist. Reason
      (MC resolution): at B=2,000 the MC SE near the smallest C5-minus-C2
      Holm threshold (0.05/10 = 0.005) is sqrt(.005*.995/2000) ~= 0.0016 —
      roughly a third of the threshold, so a near-threshold significance
      verdict would be decided by draw noise; at B=20,000 it is ~= 0.0005, an
      order of magnitude under the threshold (min achievable p 1/20,001).
      The 10x factor mirrors the permutation families' 9,999 -> 99,999.
      Fires on the SAME MC-overlap trigger, FAMILY-WIDE (never only the
      overlapping rows), with the first 2,000 draws' persisted chunks reused
      from the ledger, never redrawn.
    - ``ci_conf = 0.95`` / ``n_ci_draws = 2,000``: pointwise percentile
      hierarchical-bootstrap intervals for the DESCRIPTIVE estimates.
    - Row gates: the FULL plan section 8 production test gate set enforced at
      inference — >=100 discordant prompts overall; the REALIZED
      >=15-discordant-prompts-per-cell floor (``min_discordant_prompts_per_cell``;
      a retained cell below the floor is EXCLUDED with cause
      ``realized-discordance-below-floor``, the row denominator revised, and
      the row RE-GATED on the reduced panel — never topped up); >=100 answers
      and >=30 prompts in each class; complete labels (zero
      non-scored/non-labeled label records after the plan section 3
      retry -> human-adjudication chain); passed TEST-BANK label reliability
      (:func:`load_test_label_reliability`; objective-label rows exempt with
      the exemption stated in the gate record); and no cross-split lineage
      (enforced at assembly — ``DependencyCrossingError`` in
      ``issue2658_comparators.assemble_row_data``). Unit 10's ledger floor is
      the PROSPECTIVE bank-size proxy that drives pre-label cell exclusion;
      the realized floor here is the registered post-label gate itself.
    """

    alpha: float = float(C.HOLM["alpha"])
    n_perm_initial: int = int(C.HOLM["n_permutations_initial"])
    n_perm_extended: int = int(C.HOLM["n_permutations_extended"])
    perm_chunk_initial: tuple[int, ...] = (1111,) * 9
    perm_chunk_extension: tuple[int, ...] = (2000,) * 45
    mc_conf: float = 0.99
    n_boot: int = 2000
    boot_chunk: int = 250
    n_boot_extended: int = 20_000
    ci_conf: float = 0.95
    n_ci_draws: int = 2000
    min_discordant_prompts: int = 100
    min_discordant_prompts_per_cell: int = 15  # plan §8 REALIZED per-cell floor
    min_answers_per_class: int = 100
    min_prompts_per_class: int = 30
    sidedness: str = "greater"  # '+dot = construct-positive' preregistered sign

    def __post_init__(self) -> None:
        if sum(self.perm_chunk_initial) != self.n_perm_initial:
            raise InferenceInputError(
                f"initial chunk plan sums to {sum(self.perm_chunk_initial)} != "
                f"n_perm_initial {self.n_perm_initial}"
            )
        total = sum(self.perm_chunk_initial) + sum(self.perm_chunk_extension)
        if total != self.n_perm_extended:
            raise InferenceInputError(
                f"initial+extension chunk plans sum to {total} != "
                f"n_perm_extended {self.n_perm_extended}"
            )
        if self.n_boot % self.boot_chunk:
            raise InferenceInputError(
                f"n_boot {self.n_boot} not divisible by boot_chunk {self.boot_chunk}"
            )
        if self.n_boot_extended % self.boot_chunk or self.n_boot_extended <= self.n_boot:
            raise InferenceInputError(
                f"n_boot_extended {self.n_boot_extended} must exceed n_boot {self.n_boot} "
                f"and divide into boot_chunk {self.boot_chunk} chunks"
            )
        if self.min_discordant_prompts_per_cell < 1:
            raise InferenceInputError(
                f"min_discordant_prompts_per_cell {self.min_discordant_prompts_per_cell} "
                "must be >= 1 (plan section 8 registers 15)"
            )
        if self.sidedness != "greater":
            raise InferenceInputError("only the preregistered one-sided 'greater' form exists")


REGISTERED_INFERENCE = InferenceRegistry()


def derive_seed(*parts: Any) -> int:
    """SHA-derived, recorded seed (namespaced; machine-stable string parts)."""
    h = hashlib.sha256("|".join([SEED_NAMESPACE, *[str(p) for p in parts]]).encode()).digest()
    return int.from_bytes(h[:8], "big") % (2**63)


# ---------------------------------------------------------------------------
# Chunk ledger (mirrors unit 8's PowerLedger: atomic-append JSONL + resume
# keyed on generating parameters — NEVER the bytes of a recomputed float).
# ---------------------------------------------------------------------------
class InfLedger:
    def __init__(self, path: Path, schema: str) -> None:
        self.path = Path(path)
        self.schema = schema
        self._done: dict[str, dict[str, Any]] = {}
        if self.path.exists():
            with self.path.open() as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    if rec.get("schema") != schema:
                        raise InferenceInputError(
                            f"{self.path}: foreign schema {rec.get('schema')!r} (want {schema})"
                        )
                    self._done[rec["key"]] = rec

    @staticmethod
    def chunk_key(**params: Any) -> str:
        required = {
            "kind",
            "row",
            "comparator",
            "scores_fingerprint",
            "panel_fingerprint",
            "chunk_start",
            "chunk_size",
        }
        missing = required - set(params)
        if missing:
            raise InferenceInputError(f"chunk_key missing generating parameters: {sorted(missing)}")
        for k, v in params.items():
            if isinstance(v, float) and k not in ("selected_c",):
                raise InferenceInputError(
                    f"chunk_key parameter {k!r} is a float — resume keys must be "
                    "machine-stable generating parameters (#1336)"
                )
        body = json.dumps({k: repr(v) for k, v in sorted(params.items())}, sort_keys=True)
        return hashlib.sha256(body.encode()).hexdigest()

    def get(self, key: str) -> dict[str, Any] | None:
        return self._done.get(key)

    def append(self, record: dict[str, Any]) -> None:
        record["schema"] = self.schema
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as fh:
            fh.write(json.dumps(record, sort_keys=True) + "\n")
            fh.flush()
            import os

            os.fsync(fh.fileno())
        self._done[record["key"]] = record


# ---------------------------------------------------------------------------
# Small registered statistics helpers.
# ---------------------------------------------------------------------------
def plus_one_p(k_exceed: int, n_perm: int) -> float:
    """Plus-one Monte Carlo p: (1+k)/(B+1). Exactly zero is IMPOSSIBLE."""
    if not (0 <= k_exceed <= n_perm) or n_perm < 1:
        raise InferenceInputError(f"plus_one_p: k={k_exceed} out of range for B={n_perm}")
    p = (1.0 + k_exceed) / (n_perm + 1.0)
    assert 0.0 < p <= 1.0
    return p


def clopper_pearson_interval(k: int, n: int, conf: float) -> tuple[float, float]:
    """Two-sided CP interval for the exceedance probability (MC uncertainty)."""
    if not (0 <= k <= n) or n < 1 or not (0.0 < conf < 1.0):
        raise InferenceInputError(f"clopper_pearson_interval: bad (k={k}, n={n}, conf={conf})")
    a = 1.0 - conf
    lo = 0.0 if k == 0 else float(sps.beta.ppf(a / 2.0, k, n - k + 1))
    hi = 1.0 if k == n else float(sps.beta.ppf(1.0 - a / 2.0, k + 1, n - k))
    return lo, hi


def holm_thresholds(m: int, n_available: int, alpha: float) -> tuple[float, ...]:
    """The Holm step-down thresholds actually used: alpha/(m-i+1), i=1..n_avail
    (family size m FIXED from the committed partition; n_available <= m)."""
    if not (1 <= n_available <= m):
        raise InferenceInputError(f"holm_thresholds: n_available={n_available} not in [1, {m}]")
    return tuple(alpha / (m - i + 1) for i in range(1, n_available + 1))


def holm_adjust(pvals: dict[str, float], m: int) -> dict[str, float]:
    """Holm step-down adjusted p-values with FIXED family size m (>= len(pvals);
    not-estimable family members never shrink the family — conservative)."""
    if not pvals:
        return {}
    if len(pvals) > m:
        raise InferenceInputError(f"{len(pvals)} p-values exceed the family size m={m}")
    for name, p in pvals.items():
        if not (0.0 < p <= 1.0):
            raise InferenceInputError(f"p-value for {name!r} out of (0, 1]: {p}")
    ordered = sorted(pvals.items(), key=lambda kv: kv[1])
    adj: dict[str, float] = {}
    running = 0.0
    for i, (name, p) in enumerate(ordered, start=1):
        running = max(running, min(1.0, (m - i + 1) * p))
        adj[name] = running
    return adj


def extension_trigger(
    mc_intervals: dict[str, tuple[float, float]], thresholds: Sequence[float]
) -> dict[str, Any]:
    """Family-wide deterministic-extension trigger: ANY test's MC interval
    containing ANY Holm threshold of the family triggers extension of EVERY
    test in the family (plan section 7)."""
    overlaps = {
        name: [t for t in thresholds if lo <= t <= hi]
        for name, (lo, hi) in mc_intervals.items()
        if any(lo <= t <= hi for t in thresholds)
    }
    return {
        "triggered": bool(overlaps),
        "overlapping_tests": {k: sorted(v) for k, v in sorted(overlaps.items())},
        "thresholds": sorted(thresholds),
    }


# ---------------------------------------------------------------------------
# Prospective not-estimable ledger (unit 10's frame manifest).
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RowCellLedger:
    """One row's prospective per-cell feasibility record (plan section 8 floor)."""

    row: str
    floor: int
    n_cells: int
    n_cells_estimable: int
    excluded: tuple[dict[str, Any], ...]  # {"cell", "cause", "n_test_eligible"}

    @property
    def excluded_cells(self) -> frozenset[str]:
        return frozenset(rec["cell"] for rec in self.excluded)

    def __post_init__(self) -> None:
        if self.n_cells - len(self.excluded) != self.n_cells_estimable:
            raise InferenceInputError(
                f"row {self.row}: ledger counts do not reconcile "
                f"({self.n_cells} - {len(self.excluded)} != {self.n_cells_estimable})"
            )


def load_prospective_ledger(manifest_path: Path | None = None) -> dict[str, RowCellLedger]:
    """Load the committed frame manifest's prospective not-estimable ledger.

    REFUSES loudly when the ledger is missing — the 21 flagged cells were
    declared not-estimable BEFORE generation, and a silent default of "all
    cells estimable" would un-declare them."""
    p = Path(manifest_path) if manifest_path is not None else F.FRAME_MANIFEST_PATH
    if not p.exists():
        raise InferenceInputError(f"frame manifest not found at {p}")
    body = json.loads(p.read_text())
    if "prospective_not_estimable_ledger" not in body:
        raise InferenceInputError(
            f"frame manifest {p} carries NO prospective_not_estimable_ledger; refusing "
            "to default to 'all cells estimable' — re-freeze the manifest via unit 10"
        )
    F.assert_manifest_immutable(body)
    F.validate_manifest(body)  # includes the ledger<->rows reconciliation
    led = body["prospective_not_estimable_ledger"]
    out: dict[str, RowCellLedger] = {}
    for rr in body["rows"]:
        out[rr["row"]] = RowCellLedger(
            row=rr["row"],
            floor=int(led["floor"]),
            n_cells=int(rr["n_cells"]),
            n_cells_estimable=int(rr["n_cells_estimable"]),
            excluded=tuple(rr["prospective_not_estimable"]),
        )
    if set(out) != set(C.ROW_IDS):
        raise InferenceInputError("frame manifest rows do not cover the 11 registered rows")
    return out


def synthetic_row_ledger(
    row: str, cells: Sequence[str], excluded: Sequence[dict[str, Any]]
) -> RowCellLedger:
    """Smoke/tests seam: an explicitly constructed per-row cell ledger."""
    return RowCellLedger(
        row=row,
        floor=F.PRODUCTION_TEST_PROMPTS_PER_CELL_FLOOR,
        n_cells=len(cells),
        n_cells_estimable=len(cells) - len(excluded),
        excluded=tuple(excluded),
    )


# ---------------------------------------------------------------------------
# Test panel (sealed-test arrays, ledger exclusions applied, denominators
# recorded).
# ---------------------------------------------------------------------------
@dataclass
class InferencePanel:
    row: str
    prompt_ids: np.ndarray  # (n,) str
    response_index: np.ndarray  # (n,) int64
    labels: np.ndarray  # (n,) bool
    scores: dict[str, np.ndarray]  # comparator -> (n,) float64
    cells: np.ndarray  # (n,) str "frame|band"
    accounting: dict[str, Any]

    def __post_init__(self) -> None:
        n = self.prompt_ids.shape[0]
        if n == 0:
            raise InferenceInputError(f"row {self.row}: empty panel after cell exclusion")
        if self.labels.dtype != np.bool_ or self.labels.shape != (n,):
            raise InferenceInputError(f"row {self.row}: labels must be (n,) bool")
        for comp, s in self.scores.items():
            if s.shape != (n,) or not np.isfinite(s).all():
                raise InferenceInputError(f"row {self.row}/{comp}: bad score array")


def prompt_cells_from_rowdata(rd: U.RowData) -> dict[str, str]:
    """prompt_id -> 'frame|band' for TEST rows; inconsistent mapping raises."""
    out: dict[str, str] = {}
    for r in rd.rows:
        if r.split != "test":
            continue
        cell = f"{r.source_frame}|{r.stratum}"
        prior = out.setdefault(r.prompt_id, cell)
        if prior != cell:
            raise InferenceInputError(
                f"row {rd.row}: prompt {r.prompt_id} carries inconsistent cells "
                f"({prior!r} vs {cell!r})"
            )
    if not out:
        raise InferenceInputError(f"row {rd.row}: no test rows to map cells for")
    return out


def build_panel(
    row: str,
    records: dict[tuple[str, str], dict[str, Any]],
    comparators: Sequence[str],
    cell_ledger: RowCellLedger,
    prompt_cells: dict[str, str],
) -> InferencePanel:
    """Assemble the sealed-test panel from unit 9's final records (via the
    unit-10 seam ``load_unit_scores``), apply the prospective ledger's cell
    exclusions, and RECORD the revised per-row denominator."""
    if cell_ledger.row != row:
        raise InferenceInputError(f"cell ledger row {cell_ledger.row!r} != {row!r}")
    arrays: dict[str, dict[str, np.ndarray]] = {}
    for comp in comparators:
        rec = records.get((row, comp))
        if rec is None:
            raise InferenceInputError(f"no final comparator record for ({row}, {comp})")
        arrays[comp] = U.load_unit_scores(rec)  # sha-verified against the ledger
    ref = arrays[comparators[0]]
    pids = np.asarray(ref["test_prompt_ids"])
    ridx = np.asarray(ref["test_response_index"], dtype=np.int64)
    labels = np.asarray(ref["test_labels"])
    if labels.dtype != np.bool_:
        raise InferenceInputError(f"row {row}: test labels are {labels.dtype}, not bool")
    for comp in comparators[1:]:
        z = arrays[comp]
        if (
            not np.array_equal(np.asarray(z["test_prompt_ids"]), pids)
            or not np.array_equal(np.asarray(z["test_response_index"], dtype=np.int64), ridx)
            or not np.array_equal(np.asarray(z["test_labels"]), labels)
        ):
            raise InferenceInputError(
                f"row {row}: comparator {comp!r} test panel misaligned with "
                f"{comparators[0]!r} — refusing to join"
            )
    missing = sorted({str(p) for p in pids} - set(prompt_cells))
    if missing:
        raise InferenceInputError(
            f"row {row}: {len(missing)} test prompts have no cell mapping "
            f"(e.g. {missing[:3]}) — cannot apply the prospective ledger"
        )
    cells = np.array([prompt_cells[str(p)] for p in pids])
    excluded_cells = cell_ledger.excluded_cells
    # Ledger-vs-realized cell-name reconciliation: the ledger's excluded names
    # come from the FRAME MANIFEST (unit 10) while the realized cells are
    # rebuilt as f"{source_frame}|{stratum}" from the GEN MANIFESTS — two
    # different artifacts. A naming drift would make np.isin match NOTHING and
    # silently READMIT a prospectively not-estimable cell, so every excluded
    # cell that had eligible test prompts must actually match realized rows.
    realized_cells = set(np.unique(cells).tolist())
    ghost = [
        rec
        for rec in cell_ledger.excluded
        if int(rec.get("n_test_eligible") or 0) > 0 and rec["cell"] not in realized_cells
    ]
    if ghost:
        names = sorted(str(r["cell"]) for r in ghost)
        raise InferenceInputError(
            f"row {row}: {len(ghost)} ledger-excluded cell(s) with n_test_eligible > 0 "
            f"match NO realized test rows (e.g. {names[:5]}; realized cells: "
            f"{sorted(realized_cells)[:8]}) — frame-manifest vs gen-manifest cell-name "
            "drift would silently readmit a not-estimable cell; refusing to exclude "
            "nothing"
        )
    keep = ~np.isin(cells, sorted(excluded_cells))
    n_prompts_total = len(np.unique(pids))
    n_prompts_kept = len(np.unique(pids[keep])) if keep.any() else 0
    accounting = {
        "floor": cell_ledger.floor,
        "cells_total": cell_ledger.n_cells,
        "cells_used": cell_ledger.n_cells_estimable,
        "excluded_cells": [
            {
                "cell": rec["cell"],
                "cause": rec["cause"],
                "n_test_eligible": rec.get("n_test_eligible"),
            }
            for rec in cell_ledger.excluded
        ],
        "n_rows_total": int(pids.shape[0]),
        "n_rows_kept": int(keep.sum()),
        "n_rows_excluded": int((~keep).sum()),
        "n_prompts_total": int(n_prompts_total),
        "n_prompts_kept": int(n_prompts_kept),
        "n_prompts_excluded": int(n_prompts_total - n_prompts_kept),
    }
    return InferencePanel(
        row=row,
        prompt_ids=pids[keep],
        response_index=ridx[keep],
        labels=labels[keep],
        scores={c: np.asarray(arrays[c]["test_scores"], dtype=np.float64)[keep] for c in arrays},
        cells=cells[keep],
        accounting=accounting,
    )


def panel_fingerprint(panel: InferencePanel) -> str:
    """Machine-stable panel identity: strings/ints/bools only (no floats)."""
    body = {
        "row": panel.row,
        "rows": sorted(
            [str(p), int(r), bool(lab)]
            for p, r, lab in zip(panel.prompt_ids, panel.response_index, panel.labels)
        ),
        "excluded_cells": sorted(e["cell"] for e in panel.accounting["excluded_cells"]),
    }
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()


# Mechanical cause tag for a RETAINED cell excluded post-labels (plan §8's
# realized per-cell floor; the prospective causes live in issue2658_frames).
CAUSE_REALIZED_DISCORDANCE = "realized-discordance-below-floor"

# Frozen TEST-BANK reliability artifact sub-root (plan §3 line 30 / §8 "passed
# label reliability"). Full path:
#   <artifacts_root>/human_audit_test/<PW.HUMAN_AUDIT_REL>
# i.e. <artifacts_root>/human_audit_test/human_audit/adjudications.json — the
# trailing components are appended by the IMPORTED issue2658_power
# ``reliability_gates`` (reuse by import, never a re-implementation).
TEST_AUDIT_DIR = "human_audit_test"


def load_test_label_reliability(artifacts_root: Path) -> dict[str, Any]:
    """Frozen TEST-BANK label-reliability verdict (plan §3 line 30 / §8).

    Provenance: the artifact is produced by the unit-7 blinded annotation
    instrument (``issue2658_human_read``) applied to the sealed TEST bank —
    the same double-human adjudication round-trip that produces the DEV-side
    ``human_audit/adjudications.json`` consumed by the P3 pilot gate
    (``issue2658_power._gate_human_audit``), re-run on test-bank items once
    final test labels exist. The test side is a DIFFERENT artifact from the
    dev verdict and lives at
    ``<artifacts_root>/human_audit_test/human_audit/adjudications.json``.

    Schema: the dev artifact's ``rows`` schema ({row, item_id,
    response_index, rater_a_prob, rater_b_prob, judge_binary}) PLUS a
    REQUIRED envelope:

    - ``"bank": "test"`` — a dev-side (or unmarked) artifact presented in the
      test slot is REFUSED, never consumed;
    - ``"judge_instrument_fingerprints": {row: sha256}`` for EVERY judged
      row, validated against the LIVE ``C.judge_instrument_fingerprint`` so
      an audit conducted under a drifted judge-instrument revision is
      REFUSED.

    A MISSING artifact returns a NOT-ESTIMABLE verdict (empty ``per_trait``)
    — the ``label_reliability`` row gate then fails every judged row. That is
    the honest pre-audit state, not an error: the confirmatory phase BLOCKS
    until the test-bank audit lands. A PRESENT artifact with a wrong or
    missing envelope RAISES (fail fast — a mis-wired audit must never be
    mistaken for an unrun one). Reliability arithmetic is
    ``issue2658_power.reliability_gates`` by IMPORT.
    """
    root = Path(artifacts_root) / TEST_AUDIT_DIR
    audit_path = root / PW.HUMAN_AUDIT_REL
    if not audit_path.exists():
        return {
            "status": PW.GATE_NOT_ESTIMABLE,
            "missing_artifact": str(audit_path),
            "detail": (
                "no test-bank human adjudications on disk — test-label reliability "
                "cannot be established; every judged row is not-estimable until the "
                "unit-7 blinded audit runs on the sealed test bank"
            ),
            "per_trait": {},
            "bank": "test",
        }
    body = json.loads(audit_path.read_text())
    bank = body.get("bank")
    if bank != "test":
        raise InferenceInputError(
            f"reliability artifact at {audit_path} carries bank={bank!r}, not 'test' — "
            "refusing a dev-side (or unmarked) audit presented as the test-bank artifact"
        )
    fps = body.get("judge_instrument_fingerprints")
    if not isinstance(fps, dict) or not fps:
        raise InferenceInputError(
            f"test-bank reliability artifact at {audit_path} carries no "
            "judge_instrument_fingerprints envelope — cannot validate the audit against "
            "the live judge instrument; refusing"
        )
    judged = [r for r in C.ROW_IDS if C.CONSTRUCTS[r].judge_scored]
    for row in judged:
        live = C.judge_instrument_fingerprint(row)
        got = fps.get(row)
        if got != live:
            raise InferenceInputError(
                f"test-bank reliability artifact judge-instrument fingerprint for "
                f"{row!r} ({got!r}) != live instrument ({live}) — the audit was "
                "conducted under a different judge-instrument revision; refusing"
            )
    verdict = PW.reliability_gates(root)
    verdict["bank"] = "test"
    return verdict


def apply_realized_cell_floor(
    panel: InferencePanel, reg: InferenceRegistry
) -> tuple[InferencePanel | None, dict[str, Any]]:
    """Plan §8 REALIZED >=15-discordant-prompts-PER-CELL floor, post-labels.

    Computes per-RETAINED-cell discordant-prompt counts on the final-label
    panel. A cell below ``reg.min_discordant_prompts_per_cell`` is EXCLUDED
    with cause ``realized-discordance-below-floor`` and the row denominator is
    revised through the same accounting fields ``build_panel`` records; the
    caller then RE-RUNS the row gates on the reduced panel. Honors plan §8's
    "no new test rows are added": rows are only ever dropped, never topped up.
    Returns ``(reduced_panel, check_record)``; the panel is ``None`` when
    EVERY retained cell falls below the floor (the row is then not-estimable).
    This is the REALIZED half of the per-cell floor — unit 10's prospective
    bank-size ledger (applied in ``build_panel``) covers only cells knowable
    as unusable before any label lands.
    """
    floor = int(reg.min_discordant_prompts_per_cell)
    pids = panel.prompt_ids
    labels = panel.labels
    uniq, first_idx, inv = np.unique(pids, return_index=True, return_inverse=True)
    n_pos_per = np.bincount(inv, weights=labels.astype(np.float64), minlength=len(uniq))
    n_per = np.bincount(inv, minlength=len(uniq))
    disc = (n_pos_per > 0) & (n_pos_per < n_per)
    prompt_cell = panel.cells[first_idx]  # prompt->cell is unique (asserted upstream)
    per_cell = {str(cell): int(disc[prompt_cell == cell].sum()) for cell in np.unique(panel.cells)}
    below = sorted(cell for cell, k in per_cell.items() if k < floor)
    check: dict[str, Any] = {
        "floor": floor,
        "per_cell_discordant": per_cell,
        "cells_evaluated": len(per_cell),
        "cells_below_floor": len(below),
        "excluded": [
            {
                "cell": cell,
                "cause": CAUSE_REALIZED_DISCORDANCE,
                "n_discordant": per_cell[cell],
            }
            for cell in below
        ],
    }
    if not below:
        check["pass"] = True
        return panel, check
    keep = ~np.isin(panel.cells, below)
    if not keep.any():
        check["pass"] = False
        check["detail"] = (
            f"every retained cell fell below the realized >={floor}-discordant-prompts "
            "floor — the row is not-estimable (plan §8: no new test rows are added)"
        )
        return None, check
    # Denominator revision through the existing accounting machinery (the
    # same fields build_panel records for the prospective exclusion).
    acc = dict(panel.accounting)
    n_prompts_kept = int(len(np.unique(panel.prompt_ids[keep])))
    acc["realized_excluded_cells"] = check["excluded"]
    acc["cells_used"] = int(acc["cells_used"]) - len(below)
    acc["n_rows_kept"] = int(keep.sum())
    acc["n_rows_excluded"] = int(acc["n_rows_excluded"]) + int((~keep).sum())
    acc["n_prompts_kept"] = n_prompts_kept
    acc["n_prompts_excluded"] = int(acc["n_prompts_total"]) - n_prompts_kept
    reduced = InferencePanel(
        row=panel.row,
        prompt_ids=panel.prompt_ids[keep],
        response_index=panel.response_index[keep],
        labels=panel.labels[keep],
        scores={c: s[keep] for c, s in panel.scores.items()},
        cells=panel.cells[keep],
        accounting=acc,
    )
    check["pass"] = True  # cells excluded + denominator revised; row proceeds to re-gating
    return reduced, check


def row_gates(
    panel: InferencePanel,
    reg: InferenceRegistry,
    *,
    label_exclusions: dict[str, dict[str, int]],
    label_source: str,
    reliability: dict[str, Any],
    realized_cell_floor: dict[str, Any],
) -> dict[str, Any]:
    """The FULL plan section 8 ROW-level production gate set on the realized
    (prospective-ledger-filtered, realized-floor-reduced) final-label panel.
    Failure => the row is not-estimable, NAMING the failed gate — never a
    proxy. Cross-split lineage is enforced upstream at assembly
    (``DependencyCrossingError``); the realized per-cell floor record is
    computed by :func:`apply_realized_cell_floor` and embedded here so the
    verdict carries every gate."""
    labels = panel.labels
    pids = panel.prompt_ids
    uniq, inv = np.unique(pids, return_inverse=True)
    n_pos_per = np.bincount(inv, weights=labels.astype(np.float64), minlength=len(uniq))
    n_per = np.bincount(inv, minlength=len(uniq))
    discordant = (n_pos_per > 0) & (n_pos_per < n_per)
    checks = {
        "discordant_prompts": {
            "value": int(discordant.sum()),
            "floor": reg.min_discordant_prompts,
        },
        "answers_pos": {"value": int(labels.sum()), "floor": reg.min_answers_per_class},
        "answers_neg": {"value": int((~labels).sum()), "floor": reg.min_answers_per_class},
        "prompts_pos": {"value": int((n_pos_per > 0).sum()), "floor": reg.min_prompts_per_class},
        "prompts_neg": {
            "value": int((n_pos_per < n_per).sum()),
            "floor": reg.min_prompts_per_class,
        },
    }
    for c in checks.values():
        c["pass"] = bool(c["value"] >= c["floor"])
    if "pass" not in realized_cell_floor:
        raise InferenceInputError(
            "realized_cell_floor record carries no verdict — pass the record returned "
            "by apply_realized_cell_floor"
        )
    checks["realized_cell_floor"] = realized_cell_floor
    # complete_labels (plan §8): zero non-scored/non-labeled label records
    # among the final dev/test records after the plan §3 retry ->
    # human-adjudication chain. The per-status counts are RECORDED either way
    # — a passing row still shows its zeros.
    if set(label_exclusions) != {"dev", "test"}:
        raise InferenceInputError(
            f"row {panel.row}: label_exclusions must carry exactly the dev+test splits "
            f"(got {sorted(label_exclusions)}) — thread assemble_row_data's diag"
        )
    per_split = {
        split: {status: int(n) for status, n in sorted(counts.items())}
        for split, counts in sorted(label_exclusions.items())
    }
    n_excluded = sum(sum(counts.values()) for counts in per_split.values())
    checks["complete_labels"] = {
        "value": n_excluded,
        "required": 0,
        "per_split_per_status": per_split,
        "pass": bool(n_excluded == 0),
    }
    # label_reliability (plan §8 / §3 line 30): a frozen TEST-BANK blinded
    # double-human audit verdict per judged row; failed OR missing => the row
    # is not-estimable. Objective-label rows are EXEMPT (plan §3: correctness
    # uses executable/reference labels; no judge instrument exists) — the
    # exemption is stated in the gate record, never a silent skip.
    if label_source == "objective-labels":
        checks["label_reliability"] = {
            "pass": True,
            "status": None,
            "exempt": (
                "objective labels (plan §3): executable/reference labels carry no "
                "judge instrument — the reliability gate is not applicable"
            ),
        }
    elif label_source in ("judge-cells", "synthetic"):
        per_trait = reliability.get("per_trait") or {}
        verdict = per_trait.get(panel.row)
        artifact = reliability.get("artifact") or reliability.get("missing_artifact") or ""
        if verdict is None:
            checks["label_reliability"] = {
                "pass": False,
                "status": "MISSING",
                "detail": (
                    "no test-bank reliability verdict for this row — the unit-7 blinded "
                    "audit has not adjudicated it"
                ),
                "artifact": artifact,
            }
        else:
            checks["label_reliability"] = {
                "pass": bool(verdict["status"] == PW.GATE_PASS),
                "status": verdict["status"],
                "detail": verdict.get("detail", ""),
                "artifact": artifact,
            }
    else:
        raise InferenceInputError(
            f"row {panel.row}: unknown label_source {label_source!r} — expected "
            "judge-cells | objective-labels | synthetic"
        )
    return {"checks": checks, "estimable": all(c["pass"] for c in checks.values())}


# ---------------------------------------------------------------------------
# Permutation battery (C2 / C5 families).
# ---------------------------------------------------------------------------
def run_permutation_test(
    panel: InferencePanel,
    comparator: str,
    *,
    scores_fingerprint: str,
    ledger: InfLedger,
    chunk_plan: Sequence[int],
    element_budget: int = PW.CHUNK_ELEMENT_BUDGET,
    mc_conf: float = REGISTERED_INFERENCE.mc_conf,
) -> dict[str, Any]:
    """One (row, comparator) within-exact-prompt permutation test, chunked into
    deterministic SHA-seeded units persisted the moment each completes. The MC
    interval uses the PASSED ``mc_conf`` (family callers thread ``reg.mc_conf``;
    the default mirrors the registered constant for direct pilot/test calls)."""
    scores = panel.scores[comparator]
    labels = panel.labels
    pids = panel.prompt_ids
    obs, n_disc = MACRO_AUROC(scores, labels, pids)
    obs = float(obs)
    if int(n_disc) < 1 or math.isnan(obs):
        raise InferenceInputError(
            f"row {panel.row}/{comparator}: zero discordant prompts — the primary "
            "metric is undefined (row gates should have caught this)"
        )
    p_fp = panel_fingerprint(panel)
    n = labels.shape[0]
    k_total = 0
    b_total = 0
    start = 0
    for size in chunk_plan:
        key = InfLedger.chunk_key(
            kind="perm",
            row=panel.row,
            comparator=comparator,
            scores_fingerprint=scores_fingerprint,
            panel_fingerprint=p_fp,
            chunk_start=start,
            chunk_size=size,
        )
        prior = ledger.get(key)
        if prior is not None:
            if not math.isclose(prior["obs_stat"], obs, rel_tol=1e-9, abs_tol=1e-12):
                raise C.CacheStaleError(
                    f"row {panel.row}/{comparator}: resumed chunk {start}+{size} recorded "
                    f"obs {prior['obs_stat']} but the panel now yields {obs} — input drift"
                )
            k_total += int(prior["exceed"])
            b_total += size
            start += size
            print(
                f"[u11-perm] row={panel.row} comp={comparator} chunk={start - size}+{size} "
                f"exceed={prior['exceed']} resume-skip",
                flush=True,
            )
            continue
        t0 = time.time()
        seed = derive_seed("perm", panel.row, comparator, scores_fingerprint, p_fp, start, size)
        rng = np.random.default_rng(seed)
        exceed = 0
        sub = max(1, element_budget // max(1, n))
        done = 0
        while done < size:
            b = min(sub, size - done)
            perm = PERMUTE_WITHIN_PROMPT(labels, pids, rng, n_perm=b)
            pstat, _ = MACRO_AUROC(scores, perm, pids)
            exceed += int((np.asarray(pstat) >= obs - EXCEED_TOL).sum())
            done += b
        rec = {
            "key": key,
            "kind": "perm",
            "row": panel.row,
            "comparator": comparator,
            "chunk_start": start,
            "chunk_size": size,
            "seed": seed,
            "exceed": exceed,
            "obs_stat": obs,
            "n_discordant_prompts": int(n_disc),
            "n_rows": int(n),
            "scores_fingerprint": scores_fingerprint,
            "panel_fingerprint": p_fp,
            "elapsed_s": time.time() - t0,
        }
        ledger.append(rec)
        k_total += exceed
        b_total += size
        print(
            f"[u11-perm] row={panel.row} comp={comparator} chunk={start}+{size} "
            f"exceed={exceed} wall={rec['elapsed_s']:.2f}s",
            flush=True,
        )
        start += size
    return {
        "obs_macro_auroc": obs,
        "n_discordant_prompts": int(n_disc),
        "k_exceed": k_total,
        "n_perm": b_total,
        "p": plus_one_p(k_total, b_total),
        "mc_interval": clopper_pearson_interval(k_total, b_total, mc_conf),
    }


def run_family_permutations(
    family: str,
    comparator: str,
    tests: dict[str, tuple[InferencePanel, str]],
    m: int,
    reg: InferenceRegistry,
    ledger: InfLedger,
    *,
    element_budget: int = PW.CHUNK_ELEMENT_BUDGET,
) -> dict[str, Any]:
    """One Holm family of permutation tests with the FAMILY-WIDE deterministic
    extension: 9,999 draws first; when any test's MC interval overlaps any Holm
    threshold, EVERY test extends to 99,999 (initial chunks resumed from the
    ledger — the first 9,999 draws are reused, never redrawn)."""
    if not tests:
        return {
            "family": family,
            "comparator": comparator,
            "m": m,
            "alpha": reg.alpha,
            "tests": {},
            "extension": {"triggered": False, "reason": "no estimable tests"},
            "holm_adjusted_p": {},
            "significant": {},
        }
    results = {
        row: run_permutation_test(
            panel,
            comparator,
            scores_fingerprint=sha,
            ledger=ledger,
            chunk_plan=reg.perm_chunk_initial,
            element_budget=element_budget,
            mc_conf=reg.mc_conf,
        )
        for row, (panel, sha) in sorted(tests.items())
    }
    thresholds = holm_thresholds(m, len(tests), reg.alpha)
    trig = extension_trigger({r: res["mc_interval"] for r, res in results.items()}, thresholds)
    if trig["triggered"]:
        full_plan = tuple(reg.perm_chunk_initial) + tuple(reg.perm_chunk_extension)
        results = {
            row: run_permutation_test(
                panel,
                comparator,
                scores_fingerprint=sha,
                ledger=ledger,
                chunk_plan=full_plan,
                element_budget=element_budget,
                mc_conf=reg.mc_conf,
            )
            for row, (panel, sha) in sorted(tests.items())
        }
    pvals = {row: res["p"] for row, res in results.items()}
    adj = holm_adjust(pvals, m)
    return {
        "family": family,
        "comparator": comparator,
        "m": m,
        "alpha": reg.alpha,
        "n_perm_realized": {row: res["n_perm"] for row, res in results.items()},
        "tests": results,
        "extension": trig,
        "holm_adjusted_p": adj,
        "significant": {row: bool(adj[row] <= reg.alpha) for row in adj},
        "note": (
            "intervals are POINTWISE Monte Carlo intervals on the permutation p; "
            "adjusted p-values and significance are reported separately"
        ),
    }


# ---------------------------------------------------------------------------
# Hierarchical (prompt + within-class response) resampling core — shared by
# the C5-minus-C2 bootstrap and the descriptive macro CIs. Fully vectorized
# over draw-instances; per-instance AUROCs go through the REGISTERED
# ``equal_prompt_macro_auroc`` in balanced mode (P=1).
# ---------------------------------------------------------------------------
def discordant_layout(panel: InferencePanel) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """Per discordant prompt: (prompt_id, positive row positions, negative
    row positions) into the panel arrays."""
    out: list[tuple[str, np.ndarray, np.ndarray]] = []
    for pid in np.unique(panel.prompt_ids):
        rows = np.nonzero(panel.prompt_ids == pid)[0]
        lab = panel.labels[rows]
        if lab.any() and not lab.all():
            out.append((str(pid), rows[lab], rows[~lab]))
    return out


def hierarchical_paired_draws(
    layout: Sequence[tuple[str, np.ndarray, np.ndarray]],
    rng: np.random.Generator,
    n_draws: int,
    sources: dict[str, np.ndarray],
    stats: dict[str, tuple[str, str | None]],
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Hierarchical test resample: per draw, sample P prompts with replacement
    from the discordant panel; per drawn prompt INSTANCE, resample responses
    WITHIN CLASS (class counts preserved => every instance stays discordant).

    ``sources``: comparator -> scores, (n_panel,) fixed or (n_draws, n_panel)
    per-draw. ``stats``: name -> (source_a, source_b|None); the per-instance
    statistic is auroc_a - auroc_b (or auroc_a alone). Returns per stat the
    per-draw (mean, se) over its P instance values (se: ddof=1, /sqrt(P))."""
    n_p = len(layout)
    if n_p < 1:
        raise InferenceInputError("hierarchical_paired_draws needs >= 1 discordant prompt")
    for name, s in sources.items():
        if s.ndim == 2 and s.shape[0] != n_draws:
            raise InferenceInputError(f"source {name!r} has {s.shape[0]} rows != {n_draws} draws")
    sums = {k: np.zeros(n_draws) for k in stats}
    sqs = {k: np.zeros(n_draws) for k in stats}
    draw_prompts = rng.integers(0, n_p, size=(n_draws, n_p))
    for j, (_pid, pos_idx, neg_idx) in enumerate(layout):
        d_rows = np.nonzero(draw_prompts == j)[0]
        m_inst = d_rows.size
        if m_inst == 0:
            continue
        npos, nneg = len(pos_idx), len(neg_idx)
        pos_cols = pos_idx[rng.integers(0, npos, size=(m_inst, npos))]
        neg_cols = neg_idx[rng.integers(0, nneg, size=(m_inst, nneg))]
        cols = np.concatenate([pos_cols, neg_cols], axis=1)
        lab = np.zeros(npos + nneg, dtype=bool)
        lab[:npos] = True
        lab3 = np.broadcast_to(lab, (m_inst, 1, npos + nneg))
        aurocs: dict[str, np.ndarray] = {}
        for name, s in sources.items():
            vals = s[cols] if s.ndim == 1 else s[d_rows[:, None], cols]
            a, cnt = MACRO_AUROC(vals[:, None, :], lab3)
            if not (np.asarray(cnt) == 1).all():
                raise InferenceInputError("within-class resample lost a class — impossible")
            aurocs[name] = np.asarray(a, dtype=np.float64).reshape(m_inst)
        for sname, (na, nb) in stats.items():
            v = aurocs[na] - (aurocs[nb] if nb is not None else 0.0)
            np.add.at(sums[sname], d_rows, v)
            np.add.at(sqs[sname], d_rows, v * v)
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for sname in stats:
        mean = sums[sname] / n_p
        if n_p > 1:
            var = np.maximum(sqs[sname] - n_p * mean**2, 0.0) / (n_p - 1)
            se = np.sqrt(var / n_p)
        else:
            se = np.full(n_draws, np.nan)
        out[sname] = (mean, se)
    return out


# ---------------------------------------------------------------------------
# C5-minus-C2: one-sided studentized paired hierarchical bootstrap with
# development Bayesian-bootstrap refitting (plan section 7).
# ---------------------------------------------------------------------------
def weighted_zscore(X: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Weighted per-dimension mean/sd (mirrors unit 9's ZScore semantics:
    population sd; constant dims center to 0 with sd pinned to 1)."""
    ws = w.sum()
    if not np.isfinite(ws) or ws <= 0:
        raise InferenceInputError("weighted_zscore: non-positive total weight")
    mean = (w[:, None] * X).sum(axis=0) / ws
    var = (w[:, None] * (X - mean) ** 2).sum(axis=0) / ws
    sd = np.sqrt(var)
    return mean, np.where(sd > 0, sd, 1.0)


def paired_macro_diff(
    scores_a: np.ndarray, scores_b: np.ndarray, panel: InferencePanel
) -> tuple[float, float, int]:
    """Frozen paired point estimate: per-discordant-prompt AUROC differences
    via the REGISTERED within-prompt AUROC; returns (delta, se, n_disc)."""
    diffs = []
    for _pid, pos_idx, neg_idx in discordant_layout(panel):
        rows = np.concatenate([pos_idx, neg_idx])
        lab = panel.labels[rows]
        diffs.append(
            WITHIN_PROMPT_AUROC(scores_a[rows], lab) - WITHIN_PROMPT_AUROC(scores_b[rows], lab)
        )
    d = np.asarray(diffs, dtype=np.float64)
    if d.size < 2:
        raise InferenceInputError(
            f"row {panel.row}: {d.size} discordant prompts — cannot studentize"
        )
    delta = float(d.mean())
    se = float(d.std(ddof=1) / math.sqrt(d.size))
    if se == 0.0:
        raise InferenceInputError(
            f"row {panel.row}: zero paired-difference variance — degenerate panel"
        )
    return delta, se, int(d.size)


def _studentize(
    delta_star: np.ndarray, se_star: np.ndarray, delta_hat: float
) -> tuple[np.ndarray, int]:
    num = delta_star - delta_hat
    t = np.empty_like(num)
    ok = se_star > 0
    t[ok] = num[ok] / se_star[ok]
    t[~ok] = np.where(num[~ok] > 0, np.inf, np.where(num[~ok] < 0, -np.inf, 0.0))
    return t, int((~ok).sum())


def run_bootstrap_test(
    panel: InferencePanel,
    rowdata: U.RowData,
    *,
    selected_c: float,
    scores_fingerprint: str,
    ledger: InfLedger,
    reg: InferenceRegistry,
    allow_underdetermined: str | None = None,
    n_boot: int | None = None,
) -> dict[str, Any]:
    """One row's C5-minus-C2 one-sided studentized paired hierarchical
    bootstrap. Per draw: (dev side) Dirichlet weights over DEV PROMPTS ->
    weighted z-score + weighted logistic refit at the FROZEN selected C ->
    re-score the test panel; (test side) prompt + within-class response
    resample applied to the refit C5 scores and the FIXED C2 scores (paired:
    identical drawn rows). t* = (delta* - delta_hat)/se*; one-sided greater.

    ``n_boot`` (default ``reg.n_boot``): total draws. The extended plan
    (``reg.n_boot_extended``) is the initial chunk plan plus extension chunks
    — chunk seeds depend only on (generating params, chunk_start, chunk_size),
    so initial chunks resume-skip from the ledger and their persisted
    delta*/t* arrays are summed into the extended p, never redrawn."""
    if float(selected_c) not in U.C_GRID:
        raise InferenceInputError(
            f"row {panel.row}: frozen selected_c {selected_c} is not on the registered grid"
        )
    n_total = reg.n_boot if n_boot is None else int(n_boot)
    if n_total % reg.boot_chunk:
        raise InferenceInputError(f"n_boot {n_total} not divisible by boot_chunk {reg.boot_chunk}")
    # --- alignment: panel rows -> rowdata rows (activations for the refits).
    pos_map = {(r.prompt_id, r.response_index): i for i, r in enumerate(rowdata.rows)}
    test_pos = []
    for pid, ridx, lab in zip(panel.prompt_ids, panel.response_index, panel.labels):
        key = (str(pid), int(ridx))
        i = pos_map.get(key)
        if i is None:
            raise InferenceInputError(f"row {panel.row}: panel key {key} absent from rowdata")
        if bool(rowdata.rows[i].label) != bool(lab) or rowdata.rows[i].split != "test":
            raise InferenceInputError(f"row {panel.row}: rowdata/panel drift at {key}")
        test_pos.append(i)
    test_pos = np.asarray(test_pos, dtype=np.int64)
    dev_idx = np.nonzero(rowdata.mask("dev"))[0]
    X_dev = rowdata.X[dev_idx].astype(np.float64)
    y_dev = np.array([rowdata.rows[i].label for i in dev_idx], dtype=bool)
    dev_codes, n_dev_prompts = PW._prompt_codes(
        np.array([rowdata.rows[i].prompt_id for i in dev_idx])
    )
    U.assert_probe_wellposed(len(dev_idx), rowdata.d, allow_underdetermined)
    X_test = rowdata.X[test_pos].astype(np.float64)

    s5_frozen = panel.scores["c5_full_probe"]
    s2_frozen = panel.scores["c2_direction_dot"]
    delta_hat, se_hat, n_disc = paired_macro_diff(s5_frozen, s2_frozen, panel)
    t0_stat = delta_hat / se_hat
    layout = discordant_layout(panel)
    p_fp = panel_fingerprint(panel)

    delta_star: list[float] = []
    t_star: list[float] = []
    n_degenerate = 0
    iters_all: list[int] = []
    for start in range(0, n_total, reg.boot_chunk):
        size = min(reg.boot_chunk, n_total - start)
        key = InfLedger.chunk_key(
            kind="boot",
            row=panel.row,
            comparator="c5_minus_c2",
            scores_fingerprint=scores_fingerprint,
            panel_fingerprint=p_fp,
            data_fingerprint=rowdata.data_fingerprint,
            selected_c=float(selected_c),
            chunk_start=start,
            chunk_size=size,
        )
        prior = ledger.get(key)
        if prior is not None:
            if not math.isclose(prior["delta_hat"], delta_hat, rel_tol=1e-9, abs_tol=1e-12):
                raise C.CacheStaleError(
                    f"row {panel.row}: resumed boot chunk {start}+{size} recorded "
                    f"delta_hat {prior['delta_hat']} but the panel now yields {delta_hat}"
                )
            delta_star.extend(prior["delta_star"])
            t_star.extend(prior["t_star"])
            n_degenerate += int(prior["n_degenerate_se"])
            print(
                f"[u11-boot] row={panel.row} chunk={start}+{size} resume-skip",
                flush=True,
            )
            continue
        t_wall = time.time()
        seed = derive_seed(
            "boot", panel.row, scores_fingerprint, p_fp, rowdata.data_fingerprint, start, size
        )
        rng = np.random.default_rng(seed)
        est = U._fresh_estimator()
        est.C = float(selected_c)
        s5_draws = np.empty((size, panel.labels.shape[0]), dtype=np.float64)
        iters: list[int] = []
        for b in range(size):
            g = rng.dirichlet(np.ones(n_dev_prompts))
            w_rows = g[dev_codes]
            w_rows = w_rows * (len(dev_idx) / w_rows.sum())
            mean, sd = weighted_zscore(X_dev, w_rows)
            est.fit((X_dev - mean) / sd, y_dev, sample_weight=w_rows)
            iters.append(int(np.max(est.n_iter_)))
            s5_draws[b] = est.decision_function((X_test - mean) / sd)
        draws = hierarchical_paired_draws(
            layout,
            rng,
            size,
            sources={"c5": s5_draws, "c2": s2_frozen},
            stats={"c5_minus_c2": ("c5", "c2")},
        )
        d_star, se_star = draws["c5_minus_c2"]
        t_chunk, n_deg = _studentize(d_star, se_star, delta_hat)
        rec = {
            "key": key,
            "kind": "boot",
            "row": panel.row,
            "comparator": "c5_minus_c2",
            "chunk_start": start,
            "chunk_size": size,
            "seed": seed,
            "delta_hat": delta_hat,
            "se_hat": se_hat,
            "delta_star": [float(x) for x in d_star],
            "t_star": [float(x) for x in t_chunk],
            "n_degenerate_se": int(n_deg),
            "refit_n_iter_max": int(max(iters)),
            "refit_n_iter_mean": float(np.mean(iters)),
            "selected_c": float(selected_c),
            "scores_fingerprint": scores_fingerprint,
            "panel_fingerprint": p_fp,
            "data_fingerprint": rowdata.data_fingerprint,
            "elapsed_s": time.time() - t_wall,
        }
        ledger.append(rec)
        delta_star.extend(rec["delta_star"])
        t_star.extend(rec["t_star"])
        n_degenerate += int(n_deg)
        iters_all.extend(iters)
        print(
            f"[u11-boot] row={panel.row} chunk={start}+{size} "
            f"wall={rec['elapsed_s']:.1f}s refit_iter_max={max(iters)}",
            flush=True,
        )
    t_arr = np.asarray(t_star, dtype=np.float64)
    k = int((t_arr >= t0_stat - EXCEED_TOL).sum())
    b_total = t_arr.size
    p = plus_one_p(k, b_total)
    lower = delta_hat - float(np.quantile(t_arr, 1.0 - reg.alpha)) * se_hat
    return {
        "delta_hat": delta_hat,
        "se_hat": se_hat,
        "t0": t0_stat,
        "n_discordant_prompts": n_disc,
        "n_boot": int(b_total),
        "k_exceed": k,
        "p": p,
        "mc_interval": clopper_pearson_interval(k, b_total, reg.mc_conf),
        "one_sided_lower_bound": float(lower),
        "lower_bound_conf": 1.0 - reg.alpha,
        "n_degenerate_se": int(n_degenerate),
        "selected_c": float(selected_c),
        "sidedness": reg.sidedness,
    }


def run_family_bootstrap(
    tests: dict[str, tuple[InferencePanel, U.RowData, float, str]],
    m: int,
    reg: InferenceRegistry,
    ledger: InfLedger,
    *,
    allow_underdetermined: str | None = None,
) -> dict[str, Any]:
    """The C5-minus-C2 Holm family with its PRE-REGISTERED family-wide
    deterministic extension (a unit-11 ADDITION to plan section 7 — see
    :class:`InferenceRegistry`): run every row at ``n_boot``; when any row's
    MC interval overlaps any Holm threshold of the family, extend EVERY row to
    ``n_boot_extended`` — never only the overlapping rows — with the initial
    chunks resumed from the ledger (the first 2,000 draws are reused, never
    redrawn)."""

    def _run(n_boot: int) -> dict[str, Any]:
        return {
            row: run_bootstrap_test(
                panel,
                rowdata,
                selected_c=selected_c,
                scores_fingerprint=sha,
                ledger=ledger,
                reg=reg,
                allow_underdetermined=allow_underdetermined,
                n_boot=n_boot,
            )
            for row, (panel, rowdata, selected_c, sha) in sorted(tests.items())
        }

    extension_record: dict[str, Any] = {
        "registered": True,
        "provenance": (
            "unit-11 PRE-REGISTERED ADDITION to plan section 7 (which registers the "
            "99,999 deterministic extension for the permutation families only), "
            "recorded before any sealed-test labels exist; reason: MC resolution — "
            "at B=2,000 the MC SE near the smallest Holm threshold 0.005 is ~0.0016 "
            "(~1/3 of the threshold), so a near-threshold verdict would be decided "
            "by draw noise; at B=20,000 it is ~0.0005"
        ),
        "n_boot_initial": reg.n_boot,
        "n_boot_extended": reg.n_boot_extended,
    }
    if not tests:
        return {
            "family": "C5_minus_C2",
            "comparator": "c5_minus_c2",
            "m": m,
            "alpha": reg.alpha,
            "tests": {},
            "holm_adjusted_p": {},
            "significant": {},
            "extension": {
                **extension_record,
                "trigger": {"triggered": False, "reason": "no estimable tests"},
                "fired": False,
                "n_boot_realized": {},
            },
        }
    results = _run(reg.n_boot)
    thresholds = holm_thresholds(m, len(tests), reg.alpha)
    trig = extension_trigger({r: res["mc_interval"] for r, res in results.items()}, thresholds)
    if trig["triggered"]:
        results = _run(reg.n_boot_extended)  # family-wide; initial chunks resume-skip
    boot_p = {row: res["p"] for row, res in results.items()}
    adj = holm_adjust(boot_p, m)
    return {
        "family": "C5_minus_C2",
        "comparator": "c5_minus_c2",
        "m": m,
        "alpha": reg.alpha,
        "tests": results,
        "holm_adjusted_p": adj,
        "significant": {row: bool(adj[row] <= reg.alpha) for row in adj},
        "extension": {
            **extension_record,
            "trigger": trig,
            "fired": bool(trig["triggered"]),
            "n_boot_realized": {row: res["n_boot"] for row, res in results.items()},
        },
        "note": (
            "one-sided studentized paired hierarchical bootstrap; the lower bound is "
            "POINTWISE and reported separately from the Holm-adjusted p"
        ),
    }


# ---------------------------------------------------------------------------
# Descriptive estimates: global AUROC (pooled; DESCRIPTIVE ONLY) with
# hierarchical prompt/response uncertainty + pointwise macro CIs.
# ---------------------------------------------------------------------------
def descriptive_stats(
    panel: InferencePanel, comparators: Sequence[str], reg: InferenceRegistry
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "note": (
            "global AUROC is DESCRIPTIVE only (plan section 7); all intervals are "
            "POINTWISE percentile hierarchical-bootstrap intervals"
        )
    }
    layout = discordant_layout(panel)
    rng_macro = np.random.default_rng(
        derive_seed("desc-macro", panel.row, panel_fingerprint(panel))
    )
    macro_draws = hierarchical_paired_draws(
        layout,
        rng_macro,
        reg.n_ci_draws,
        sources={c: panel.scores[c] for c in comparators},
        stats={c: (c, None) for c in comparators},
    )
    lo_q, hi_q = (1 - reg.ci_conf) / 2, 1 - (1 - reg.ci_conf) / 2
    uniq_p, inv = np.unique(panel.prompt_ids, return_inverse=True)
    n_prompts = len(uniq_p)
    counts = np.bincount(inv, minlength=n_prompts)
    prompt_rows = [np.nonzero(inv == j)[0] for j in range(n_prompts)]
    for comp in comparators:
        s = panel.scores[comp]
        macro, n_disc = MACRO_AUROC(s, panel.labels, panel.prompt_ids)
        # Hierarchical (prompt + within-prompt response) draws for the pooled
        # global AUROC: ragged resamples force per-draw registered calls
        # (balanced mode carries no mask); measured trivial next to the
        # batteries (~1 ms/draw).
        rng = np.random.default_rng(
            derive_seed("desc-global", panel.row, comp, panel_fingerprint(panel))
        )
        g_draws = np.empty(reg.n_ci_draws)
        for b in range(reg.n_ci_draws):
            drawn = rng.integers(0, n_prompts, size=n_prompts)
            rows = np.concatenate(
                [prompt_rows[j][rng.integers(0, counts[j], size=counts[j])] for j in drawn]
            )
            g_draws[b] = WITHIN_PROMPT_AUROC(s[rows], panel.labels[rows])
        g_ok = g_draws[~np.isnan(g_draws)]
        if g_ok.size == 0:
            raise InferenceInputError(f"row {panel.row}/{comp}: every global draw degenerate")
        m_draws = macro_draws[comp][0]
        out[comp] = {
            "macro_auroc": float(macro),
            "n_discordant_prompts": int(n_disc),
            "macro_ci_pointwise": [
                float(np.quantile(m_draws, lo_q)),
                float(np.quantile(m_draws, hi_q)),
            ],
            "global_auroc_descriptive": float(WITHIN_PROMPT_AUROC(s, panel.labels)),
            "global_ci_pointwise": [
                float(np.quantile(g_ok, lo_q)),
                float(np.quantile(g_ok, hi_q)),
            ],
            "global_ci_n_degenerate_draws": int(reg.n_ci_draws - g_ok.size),
            "ci_conf": reg.ci_conf,
        }
    return out


# ---------------------------------------------------------------------------
# Top-level orchestration.
# ---------------------------------------------------------------------------
@dataclass
class RowInputs:
    row: str
    panel: InferencePanel
    rowdata: U.RowData
    selected_c: float
    scores_sha: dict[str, str]  # comparator -> npz sha256 (from unit 9 ledger)
    # Per-split per-status non-scored label counts from assemble_row_data's
    # diag (plan §8 complete-labels gate input). REQUIRED — a constructor that
    # cannot supply it has not threaded the final-label diagnostics.
    label_exclusions: dict[str, dict[str, int]]
    gates: dict[str, Any] = field(default_factory=dict)


def derive_family_sizes(partition: dict[str, list[str]], universe: Sequence[str]) -> dict[str, int]:
    """Realized Holm family sizes DERIVED from the committed partition over the
    realized row universe — never a hardcoded 11 (the power basis 0.05/11 is a
    sizing worst case, not a family size)."""
    uni = set(universe)
    ne = uni & set(partition["not_estimable"])
    return {"C2": len(uni) - len(ne), "C5": len(uni), "C5_minus_C2": len(uni) - len(ne)}


def run_inference(
    rows_input: dict[str, RowInputs],
    partition: dict[str, list[str]],
    reg: InferenceRegistry,
    out_root: Path,
    *,
    reliability: dict[str, Any],
    allow_underdetermined: str | None = None,
    require_registered_universe: bool = True,
    family_sizes_expected: dict[str, int] | None = None,
) -> dict[str, Any]:
    if require_registered_universe and set(rows_input) != set(C.ROW_IDS):
        raise InferenceInputError(
            f"row universe {sorted(rows_input)} != registered ROW_IDS — a missing row "
            "would silently shrink the Holm families"
        )
    sizes = derive_family_sizes(partition, list(rows_input))
    if require_registered_universe:
        expected = C.holm_family_sizes(len(partition["not_estimable"]))
        if sizes != expected:
            raise InferenceInputError(f"derived family sizes {sizes} != registered {expected}")
    if family_sizes_expected is not None and sizes != dict(family_sizes_expected):
        raise InferenceInputError(
            f"derived family sizes {sizes} != committed record {family_sizes_expected}"
        )
    inf_dir = Path(out_root) / "inference"
    perm_ledger = InfLedger(inf_dir / "perm_units.jsonl", PERM_CHUNK_SCHEMA)
    boot_ledger = InfLedger(inf_dir / "boot_units.jsonl", BOOT_CHUNK_SCHEMA)

    rows_report: dict[str, Any] = {}
    not_estimable: dict[str, dict[str, str]] = {"C2": {}, "C5": {}, "C5_minus_C2": {}}
    for row in sorted(set(partition["not_estimable"]) & set(rows_input)):
        not_estimable["C2"][row] = "committed partition: no frozen external direction"
        not_estimable["C5_minus_C2"][row] = "committed partition: no frozen external direction"
    estimable_rows: dict[str, RowInputs] = {}
    for row, ri in sorted(rows_input.items()):
        reduced, floor_check = apply_realized_cell_floor(ri.panel, reg)
        if reduced is None:
            # Every retained cell fell below the realized per-cell floor —
            # not-estimable outright; no reduced panel exists to re-gate.
            ri.gates = {"checks": {"realized_cell_floor": floor_check}, "estimable": False}
            rows_report[row] = {
                "panel": ri.panel.accounting,
                "gates": ri.gates,
                "estimable": False,
            }
            reason = (
                "row-level production gate failed: realized_cell_floor (every retained "
                f"cell fell below the realized >={reg.min_discordant_prompts_per_cell}-"
                "discordant-prompts-per-cell floor)"
            )
            for fam in ("C2", "C5", "C5_minus_C2"):
                not_estimable[fam].setdefault(row, reason)
            continue
        ri.panel = reduced  # revised denominator flows into every downstream battery
        ri.gates = row_gates(
            reduced,
            reg,
            label_exclusions=ri.label_exclusions,
            label_source=ri.rowdata.label_source,
            reliability=reliability,
            realized_cell_floor=floor_check,
        )
        rows_report[row] = {
            "panel": ri.panel.accounting,
            "gates": ri.gates,
            "estimable": ri.gates["estimable"],
        }
        if ri.gates["estimable"]:
            estimable_rows[row] = ri
        else:
            failed = sorted(k for k, c in ri.gates["checks"].items() if not c["pass"])
            reason = f"row-level production gate failed: {', '.join(failed)}"
            for fam in ("C2", "C5", "C5_minus_C2"):
                not_estimable[fam].setdefault(row, reason)

    eligible = set(partition["eligible"])
    c2_tests = {
        row: (ri.panel, ri.scores_sha["c2_direction_dot"])
        for row, ri in estimable_rows.items()
        if row in eligible
    }
    c5_tests = {
        row: (ri.panel, ri.scores_sha["c5_full_probe"]) for row, ri in estimable_rows.items()
    }
    fam_c2 = run_family_permutations(
        "C2", "c2_direction_dot", c2_tests, sizes["C2"], reg, perm_ledger
    )
    fam_c5 = run_family_permutations("C5", "c5_full_probe", c5_tests, sizes["C5"], reg, perm_ledger)

    boot_tests: dict[str, tuple[InferencePanel, U.RowData, float, str]] = {}
    for row, ri in sorted(estimable_rows.items()):
        if row not in eligible:
            continue
        sha_pair = hashlib.sha256(
            f"{ri.scores_sha['c2_direction_dot']}|{ri.scores_sha['c5_full_probe']}".encode()
        ).hexdigest()
        boot_tests[row] = (ri.panel, ri.rowdata, ri.selected_c, sha_pair)
    fam_boot = run_family_bootstrap(
        boot_tests,
        sizes["C5_minus_C2"],
        reg,
        boot_ledger,
        allow_underdetermined=allow_underdetermined,
    )

    for row, ri in sorted(estimable_rows.items()):
        comps = ["c5_full_probe"] + (["c2_direction_dot"] if row in eligible else [])
        rows_report[row]["descriptive"] = descriptive_stats(ri.panel, comps, reg)

    report = {
        "schema": REPORT_SCHEMA,
        "registry": asdict(reg),
        "partition": {k: sorted(v) for k, v in partition.items()},
        "family_sizes": {
            **sizes,
            "source": "direction_provenance.json c2_c3_partition (committed artifact)",
        },
        "families": {"C2": fam_c2, "C5": fam_c5, "C5_minus_C2": fam_boot},
        "label_reliability": reliability,
        "not_estimable": not_estimable,
        "rows": rows_report,
        "note": (
            "intervals are pointwise; adjusted p-values and significance are reported "
            "separately from the intervals — never one number doing both jobs"
        ),
        "metadata": as_metadata_dict(git_provenance(), phase="inference"),
    }
    write_json_atomic(inf_dir / "inference_report.json", report)
    print(f"[u11] inference report -> {inf_dir / 'inference_report.json'}", flush=True)
    return report


# ---------------------------------------------------------------------------
# Production assembly (sealed-test artifacts; fails fast while they are absent).
# ---------------------------------------------------------------------------
def assemble_production_inputs(
    comparators_dir: Path, artifacts_root: Path
) -> tuple[dict[str, RowInputs], dict[str, list[str]], dict[str, int], dict[str, Any]]:
    prov = U.load_committed_provenance()
    partition = U.c2c3_partition(prov)
    committed_sizes = prov.get("c2_c3_partition", {}).get("holm_family_sizes")
    cell_ledgers = load_prospective_ledger()
    # Plan §8 "passed label reliability": the frozen TEST-BANK verdict is a
    # REQUIRED production input — missing => a NOT-ESTIMABLE verdict that
    # fails the label_reliability gate for every judged row (the confirmatory
    # phase blocks until the unit-7 test-bank audit lands); a present-but-
    # mis-provenanced artifact RAISES inside the loader.
    reliability = load_test_label_reliability(Path(artifacts_root))
    records = U.load_comparator_results(Path(comparators_dir))
    rows_input: dict[str, RowInputs] = {}
    eligible = set(partition["eligible"])
    for row in C.ROW_IDS:
        comps = ["c5_full_probe"] + (["c2_direction_dot"] if row in eligible else [])
        c5_rec = records.get((row, "c5_full_probe"))
        if c5_rec is None or c5_rec.get("status") != "scored":
            raise InferenceInputError(f"row {row}: no scored c5_full_probe record")
        if row in eligible:
            c2_rec = records.get((row, "c2_direction_dot"))
            if c2_rec is None or c2_rec.get("status") != "scored":
                raise InferenceInputError(f"row {row}: eligible but no scored c2 record")
        rowdata, diag = U.assemble_row_data(row, Path(artifacts_root))
        panel = build_panel(
            row, records, comps, cell_ledgers[row], prompt_cells_from_rowdata(rowdata)
        )
        rows_input[row] = RowInputs(
            row=row,
            panel=panel,
            rowdata=rowdata,
            selected_c=float(c5_rec["selected_c"]),
            scores_sha={c: records[(row, c)]["scores_sha256"] for c in comps},
            # complete-labels gate input (plan §8): the per-split per-status
            # non-scored counts join_labels recorded — no longer discarded.
            label_exclusions={split: dict(counts) for split, counts in diag["exclusions"].items()},
        )
    return rows_input, partition, committed_sizes, reliability


# ---------------------------------------------------------------------------
# Smoke + production-shape measurement pilots (synthetic; never touch
# committed eval_results).
# ---------------------------------------------------------------------------
def _synthetic_ladder(
    rd: U.RowData, scratch: Path, *, seed: int
) -> dict[tuple[str, str], dict[str, Any]]:
    """Run unit 9's REAL ladder (c2 + c5) on synthetic RowData so the smoke
    exercises the actual consumption seam (ledger + sha-verified score npzs)."""
    part = {"eligible": [rd.row], "not_estimable": []}
    comps = ("c2_direction_dot", "c5_full_probe")
    ledger = U.CompLedger(scratch / "ledger.jsonl")
    counter = U._UnitCounter()
    counter.cap = U.units_for(rd.row, comps, part)
    if rd.synthetic_direction is None:
        raise InferenceInputError("synthetic rowdata carries no direction seam")
    U.run_ladder(
        rd,
        comps,
        ledger=ledger,
        counter=counter,
        scores_dir=scratch,
        embed_backend=None,
        partition=part,
        direction_provider=lambda _row: rd.synthetic_direction,
        seed=seed,
        allow_underdetermined=None,
    )
    return U.load_comparator_results(scratch)


def _smoke_cell_ledger(rd: U.RowData) -> RowCellLedger:
    """Synthetic prospective ledger: exclude ONE realized cell so the
    denominator-revision path is exercised end to end."""
    cells = sorted({f"{r.source_frame}|{r.stratum}" for r in rd.rows})
    victim = cells[-1]
    n_test = len(
        {
            r.prompt_id
            for r in rd.rows
            if r.split == "test" and f"{r.source_frame}|{r.stratum}" == victim
        }
    )
    return synthetic_row_ledger(
        rd.row,
        cells,
        [{"cell": victim, "cause": F.CAUSE_BANK_TOO_SMALL, "n_test_eligible": n_test}],
    )


def run_smoke(args: argparse.Namespace, out_root: Path) -> int:
    reg = _registry_from_args(args)
    rd = U.synthesize_row_data(
        row="synthetic",
        n_prompts=600,
        n_responses=10,
        d=16,
        n_superfamilies=40,
        effect=2.0,
        seed=args.seed,
    )
    scratch = out_root / "comparators_smoke"
    scratch.mkdir(parents=True, exist_ok=True)
    records = _synthetic_ladder(rd, scratch, seed=args.seed)
    cell_ledger = _smoke_cell_ledger(rd)
    comps = ["c5_full_probe", "c2_direction_dot"]
    panel = build_panel(rd.row, records, comps, cell_ledger, prompt_cells_from_rowdata(rd))
    c5_rec = records[(rd.row, "c5_full_probe")]
    rows_input = {
        rd.row: RowInputs(
            row=rd.row,
            panel=panel,
            rowdata=rd,
            selected_c=float(c5_rec["selected_c"]),
            scores_sha={c: records[(rd.row, c)]["scores_sha256"] for c in comps},
            label_exclusions={"dev": {}, "test": {}},  # complete by construction
        )
    }
    partition = {"eligible": [rd.row], "not_estimable": []}
    # Synthetic PASS verdict: the smoke exercises the REQUIRE branch of the
    # label_reliability gate (synthetic rows are never exempt), consuming no
    # real audit artifact.
    reliability = {
        "status": PW.GATE_PASS,
        "artifact": "synthetic-smoke reliability verdict (no real audit consumed)",
        "per_trait": {rd.row: {"status": PW.GATE_PASS, "detail": "synthetic smoke verdict"}},
        "bank": "test",
    }
    report = run_inference(
        rows_input,
        partition,
        reg,
        out_root,
        reliability=reliability,
        require_registered_universe=False,
    )
    fam = report["families"]
    print(
        "[u11-smoke] "
        f"C2 p={fam['C2']['tests'][rd.row]['p']:.5f} "
        f"C5 p={fam['C5']['tests'][rd.row]['p']:.5f} "
        f"C5-C2 p={fam['C5_minus_C2']['tests'][rd.row]['p']:.5f} "
        f"delta={fam['C5_minus_C2']['tests'][rd.row]['delta_hat']:.4f} "
        f"cells_used={report['rows'][rd.row]['panel']['cells_used']}/"
        f"{report['rows'][rd.row]['panel']['cells_total']} "
        f"peak_rss={U._peak_rss_mib():.0f}MiB",
        flush=True,
    )
    return 0


def _production_shape_panel(rng: np.random.Generator, *, effect: float = 0.6) -> InferencePanel:
    """Synthetic panel at the production floor shape: 12 cells x 30 prompts x
    30 responses (10,800 sealed-test rows)."""
    n_prompts, m_resp = 360, 30
    pids = np.repeat([f"p{i:04d}" for i in range(n_prompts)], m_resp)
    labels = rng.random(n_prompts * m_resp) < 0.5
    delta = PW.binormal_shift(effect)
    scores = rng.standard_normal(n_prompts * m_resp) + delta * labels
    return InferencePanel(
        row="pilot",
        prompt_ids=pids,
        response_index=np.tile(np.arange(m_resp, dtype=np.int64), n_prompts),
        labels=labels,
        scores={"c2_direction_dot": scores.astype(np.float64)},
        cells=np.array(["pilot|cell"] * (n_prompts * m_resp)),
        accounting={"excluded_cells": [], "cells_total": 12, "cells_used": 12, "floor": 15},
    )


def run_measure_perm_unit(args: argparse.Namespace, out_root: Path) -> int:
    """MEASURED 1-unit permutation pilot: one (row, comparator) test at the
    production panel shape through the production runner (the sizing basis)."""
    reg = REGISTERED_INFERENCE
    panel = _production_shape_panel(np.random.default_rng(args.seed))
    ledger = InfLedger(out_root / "inference" / "perm_units.jsonl", PERM_CHUNK_SCHEMA)
    t0 = time.time()
    res = run_permutation_test(
        panel,
        "c2_direction_dot",
        scores_fingerprint="pilot",
        ledger=ledger,
        chunk_plan=reg.perm_chunk_initial,
        mc_conf=reg.mc_conf,
    )
    wall = time.time() - t0
    print(
        f"[u11-pilot-perm] n_rows={panel.labels.shape[0]} n_perm={res['n_perm']} "
        f"wall={wall:.1f}s per_1k_perm={wall / res['n_perm'] * 1000:.2f}s "
        f"peak_rss={U._peak_rss_mib():.0f}MiB "
        f"projection_21_tests_initial={wall * 21 / 60:.1f}min "
        f"projection_21_tests_extended={wall * 21 * 10 / 60:.1f}min",
        flush=True,
    )
    return 0


def run_measure_boot_unit(args: argparse.Namespace, out_root: Path) -> int:
    """MEASURED 1-unit bootstrap pilot: ONE chunk at production shape
    (dev ~10,920 x 3584 weighted refits) through the production runner."""
    n_draws = int(args.pilot_boot_draws)
    reg = InferenceRegistry(n_boot=n_draws, boot_chunk=n_draws)
    rd = U.synthesize_row_data(
        row="pilot",
        n_prompts=520,
        n_responses=30,
        d=3584,
        n_superfamilies=52,
        effect=1.0,
        seed=args.seed,
    )
    # Frozen fit at the production shape (measured separately from the draws).
    dev_idx = np.nonzero(rd.mask("dev"))[0]
    test_idx = np.nonzero(rd.mask("test"))[0]
    X_dev = rd.X[dev_idx].astype(np.float64)
    y_dev = np.array([rd.rows[i].label for i in dev_idx], dtype=bool)
    t0 = time.time()
    mean, sd = X_dev.mean(axis=0), X_dev.std(axis=0)
    sd = np.where(sd > 0, sd, 1.0)
    est = U.refit_at_c((X_dev - mean) / sd, y_dev, 1.0)
    frozen_fit_wall = time.time() - t0
    s5 = est.decision_function((rd.X[test_idx].astype(np.float64) - mean) / sd)
    w = rd.synthetic_direction
    assert w is not None
    s2 = rd.X[test_idx].astype(np.float64) @ np.asarray(w, dtype=np.float64)
    te_rows = [rd.rows[i] for i in test_idx]
    panel = InferencePanel(
        row="pilot",
        prompt_ids=np.array([r.prompt_id for r in te_rows]),
        response_index=np.array([r.response_index for r in te_rows], dtype=np.int64),
        labels=np.array([r.label for r in te_rows], dtype=bool),
        scores={"c5_full_probe": s5.astype(np.float64), "c2_direction_dot": s2},
        cells=np.array([f"{r.source_frame}|{r.stratum}" for r in te_rows]),
        accounting={"excluded_cells": [], "cells_total": 4, "cells_used": 4, "floor": 15},
    )
    ledger = InfLedger(out_root / "inference" / "boot_units.jsonl", BOOT_CHUNK_SCHEMA)
    t0 = time.time()
    res = run_bootstrap_test(
        panel,
        rd,
        selected_c=1.0,
        scores_fingerprint="pilot",
        ledger=ledger,
        reg=reg,
    )
    wall = time.time() - t0
    per_draw = wall / n_draws
    full = per_draw * REGISTERED_INFERENCE.n_boot
    print(
        f"[u11-pilot-boot] n_dev={len(dev_idx)} d={rd.d} n_test={len(test_idx)} "
        f"frozen_fit_wall={frozen_fit_wall:.1f}s chunk_draws={n_draws} "
        f"chunk_wall={wall:.1f}s per_draw={per_draw:.2f}s "
        f"peak_rss={U._peak_rss_mib():.0f}MiB "
        f"projection_per_row_B{REGISTERED_INFERENCE.n_boot}={full / 3600:.2f}h "
        f"projection_10_rows={full * 10 / 3600:.1f}h p={res['p']:.4f}",
        flush=True,
    )
    return 0


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------
def _registry_from_args(args: argparse.Namespace) -> InferenceRegistry:
    over: dict[str, Any] = {}
    if args.n_boot is not None:
        over["n_boot"] = int(args.n_boot)
    if args.boot_chunk is not None:
        over["boot_chunk"] = int(args.boot_chunk)
    if over:
        print(f"[u11] registry overrides RECORDED: {over}", flush=True)
    return InferenceRegistry(**over)


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="issue2658_inference",
        description="Unit 11: preregistered confirmatory inference (plan section 7).",
    )
    ap.add_argument(
        "--phase",
        choices=("report", "smoke", "measure-perm-unit", "measure-boot-unit"),
        default=None,
        help="required unless --import-check is passed",
    )
    ap.add_argument("--import-check", action="store_true")
    ap.add_argument("--comparators-dir", type=Path, help="unit 9 output dir (ledger + npzs)")
    ap.add_argument(
        "--artifacts-root",
        type=Path,
        default=F.OUT_DIR,
        help="root of gen/judge/label/capture artifacts (assemble_row_data)",
    )
    ap.add_argument(
        "--out-root",
        type=Path,
        default=None,
        help="inference output root; smoke/measure default to a fresh /tmp scratch dir",
    )
    ap.add_argument("--seed", type=int, default=0, help="recorded salt for SHA-derived seeds")
    ap.add_argument("--n-boot", type=int, default=None, help="override (recorded)")
    ap.add_argument("--boot-chunk", type=int, default=None, help="override (recorded)")
    ap.add_argument("--pilot-boot-draws", type=int, default=25)
    ap.add_argument(
        "--allow-underdetermined",
        default=None,
        help="recorded justification for an n_dev < d bootstrap refit regime",
    )
    return ap


def run(args: argparse.Namespace) -> int:
    if args.phase == "report":
        if args.n_boot is not None or args.boot_chunk is not None:
            raise InferenceInputError(
                "--n-boot/--boot-chunk are smoke/measure-phase knobs; the confirmatory "
                "report phase runs ONLY the registered constants (plan section 7) — "
                "remove the override"
            )
        if args.comparators_dir is None:
            raise InferenceInputError("--comparators-dir is required for --phase report")
        out_root = args.out_root if args.out_root is not None else F.OUT_DIR
        reg = _registry_from_args(args)
        rows_input, partition, committed_sizes, reliability = assemble_production_inputs(
            args.comparators_dir, args.artifacts_root
        )
        run_inference(
            rows_input,
            partition,
            reg,
            out_root,
            reliability=reliability,
            allow_underdetermined=args.allow_underdetermined,
            family_sizes_expected=committed_sizes,
        )
        return 0
    out_root = args.out_root
    if out_root is None:
        out_root = Path(tempfile.mkdtemp(prefix=f"issue-2658-u11-{args.phase}-"))
        print(f"[u11] scratch out-root: {out_root}", flush=True)
    if args.phase == "smoke":
        return run_smoke(args, out_root)
    if args.phase == "measure-perm-unit":
        return run_measure_perm_unit(args, out_root)
    if args.phase == "measure-boot-unit":
        return run_measure_boot_unit(args, out_root)
    raise InferenceInputError(f"unknown phase {args.phase!r}")


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    if args.import_check:
        from explore_persona_space.orchestrate.argcheck import assert_args_attributes_defined

        assert_args_attributes_defined(__file__)
        print("[u11] import-check ok", flush=True)
        return 0
    if args.phase is None:
        raise InferenceInputError("--phase is required unless --import-check is passed")
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
