"""Issue #2658 unit 8 — discordance, clustered power, cost report, launch-or-park gate.

P0/P2/P3 code ONLY: no pod, no GPU, no judge API call. Every "measured" input is
read from artifacts produced by earlier phases (units 1-6); an absent measurement
is reported ``not-estimable`` with the missing artifact NAMED — never defaulted,
never substituted with a projection (projections are labeled ``projected``).

Deliverables (plan sections 4/7/8/10/12):

A. Canonical estimator primitives (Unit 9 IMPORTS these — one implementation):
   ``within_prompt_auroc``, ``equal_prompt_macro_auroc`` (plan section 7 primary
   metric), ``permute_labels_within_prompt`` (the plan section 7 within-prompt
   permutation — the SAME function the power simulation calls, so power is
   computed for the actual registered test).
B. Discordance measurement per cell from pilot labels, with the 10-draw ->
   30-draw projection correction and one-sided 95% Clopper-Pearson lower bounds.
C. Clustered (prompt-level) power simulation + bisection selection of the
   common production prompts-per-cell N (plan section 4).
D. Cost report: measured GPU-hours / Batch-API dollars / human-annotation load,
   each from a real artifact or ``not-estimable``; plan section 10 envelope check.
E. Mechanical launch-or-park gate verdict over the plan section 8 pilot gates.
   NOT-ESTIMABLE never collapses to PASS; verdict is LAUNCH only if EVERY gate
   is PASS.

Registered simulation constants live in :data:`REGISTERED` (frozen; derivations
in the class docstring). The plan says "the registered CI-width/power target"
without registering numbers — THIS module registers them.

Ops arithmetic (Deliverable C):
    units ~= rows(11) x (bisection evals ~6-9 + curve effects 4) ~= 110-140
    per-unit element-ops ~= n_reps(400) x (n_perm(659)+1) x prompts(12*N) x
    responses(30)  ->  ~2.9e9 at N=30, linear in N.
    projected wall = measured one-unit wall x remaining units (the driver
    MEASURES the first completed unit and prints the projection; never an
    asserted per-unit cost).

Artifacts read (schemas observed from the committed/unit-produced files):
    eval_results/issue_2658/direction_provenance.json      (unit 1)
    eval_results/issue_2658/{frame,split}_manifest.json    (unit 2)
    {gen_root}/raw_completions/{split}/{cell}.json         (unit 5, i2658-gen-cell-v1)
    {out_root}/judge/{split}/{row}/{cell}.json             (unit 6, i2658-judge-cell-v1)
    {out_root}/objective_labels/{split}/{cell}.jsonl       (unit 4, i2658-objective-labels-v1)
Artifacts written (under --out-root; smoke diverts to a /tmp scratch root):
    power/discordance.json, power/power_units.jsonl (resumable ledger),
    power/production_n.json, power/cost_report.json, power/gate_verdict.json

Launch (VM-side runs carry the shared-VM thread caps):
    OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \
    NUMEXPR_NUM_THREADS=8 MALLOC_ARENA_MAX=2 \
    uv run python scripts/issue2658_power.py --phase all
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()  # shared-VM thread caps (#847) must bind BEFORE numpy/scipy import

import numpy as np  # noqa: E402
from scipy import stats as sps
from scipy.special import gammaln
from sklearn.metrics import cohen_kappa_score

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import issue2658_common as C  # noqa: E402
import issue2658_frames as F  # noqa: E402
import issue2658_generate as G  # noqa: E402
import issue2658_judge as J  # noqa: E402
from explore_persona_space.atomic_io import write_json_atomic  # noqa: E402
from explore_persona_space.orchestrate.provenance import (  # noqa: E402
    as_metadata_dict,
    git_provenance,
)

# ---------------------------------------------------------------------------
# Registered constants (Deliverable C pre-registration).
# ---------------------------------------------------------------------------
POWER_SCHEMA = "i2658-power-unit-v1"
DISCORDANCE_SCHEMA = "i2658-discordance-v1"
PRODUCTION_N_SCHEMA = "i2658-production-n-v1"
COST_SCHEMA = "i2658-cost-report-v1"
VERDICT_SCHEMA = "i2658-gate-verdict-v1"

# Expected external artifacts for Deliverable D (producers named; absent =>
# not-estimable, never a default).
PILOT_TIMING_REL = "power_inputs/pilot_timing.json"  # producer: P1 launch wrapper
JUDGE_SPEND_REL = "power_inputs/judge_spend.json"  # producer: P2 judge wave wrapper
HUMAN_AUDIT_REL = "human_audit/adjudications.json"  # producer: blinded packet round-trip


@dataclass(frozen=True)
class PowerRegistry:
    """FROZEN registered targets for the plan section 4 clustered simulation.

    The plan registers "the registered CI-width/power target" but no numbers;
    this block closes that gap (stated in the unit-8 report). Derivations:

    - ``power_target = 0.80``: conventional adequacy bar; sizing quantity is the
      per-row rejection probability of the registered permutation test.
    - ``alpha_worst_case = 0.05 / 11``: plan section 7 runs three 11-test Holm
      families at family alpha 0.05. Holm's most stringent per-test threshold
      is alpha/m, so sizing every row's test at 0.05/11 guarantees the power
      target under the family worst case (a row must be detectable even when it
      is the only non-null and faces the alpha/11 first-step threshold).
    - ``primary_effect_auroc = 0.60``: a weak-but-real within-prompt
      displacement from the H1/H2 null of 0.5 (binormal shift
      sqrt(2)*Phi^-1(0.60) ~= 0.358, Cohen's d ~= 0.36). The full curve at
      0.55/0.60/0.65/0.70 is reported so the choice is re-decidable.
    - ``n_replicates = 400``: MC SE of a power estimate at 0.80 is
      sqrt(0.8*0.2/400) = 0.020 (95% half-width ~= 0.039) — adequate to resolve
      the 0.80 target against the 0.05-AUROC curve grid.
    - ``n_permutations = 659``: the simulated test is the registered plus-one
      permutation test, p = (1+k)/(B+1) with B=659 => rejection iff k <= 2 and
      EXACT size 3/660 = 1/220 = alpha_worst_case (zero discreteness
      conservatism at the boundary). The production test uses 9,999(+1)
      permutations (size 45/10000 = 0.00450, 1% more conservative than 1/220);
      the simulation is therefore ~1% anti-conservative on size relative to
      production — negligible against the MC SE and the max-over-rows sizing.
    - ``responses_per_prompt_production = 30`` / ``_pilot = 10``: plan section 4.
    - ``discordant_target_per_cell = 20`` under the LOWER confidence bound on
      discordance; ``prompts_per_cell_floor = 30``: plan section 4.
    - ``bisection_cap = 960``: 32x the floor; a row that cannot reach the power
      target by N=960 prompts/cell is reported not-estimable, never truncated
      silently.
    """

    power_target: float = 0.80
    alpha_family: float = 0.05
    holm_family_size: int = 11
    primary_effect_auroc: float = 0.60
    power_curve_effects: tuple[float, ...] = (0.55, 0.60, 0.65, 0.70)
    n_replicates: int = 400
    n_permutations: int = 659
    responses_per_prompt_production: int = 30
    responses_per_prompt_pilot: int = int(C.PILOT.responses_per_prompt)
    discordant_target_per_cell: int = 20
    prompts_per_cell_floor: int = 30
    bisection_cap: int = 960

    @property
    def alpha_worst_case(self) -> float:
        return self.alpha_family / self.holm_family_size


REGISTERED = PowerRegistry()
assert REGISTERED.responses_per_prompt_pilot == 10
assert C.HOLM["alpha"] == REGISTERED.alpha_family
assert len(C.ROW_IDS) == REGISTERED.holm_family_size

# Element budget per large transient array in the simulation. 2**24 elements
# = 64 MiB at f32 / 128 MiB at f64-int64 (the permutation sort key + argsort
# index are the widest transients); a handful of concurrent transients keeps
# peak RSS well under a few GB (brief throughput/persistence requirement).
CHUNK_ELEMENT_BUDGET = 2**24

GATE_PASS = "PASS"
GATE_FAIL = "FAIL"
GATE_NOT_ESTIMABLE = "NOT-ESTIMABLE"


class PowerInputError(C.Issue2658GuardError):
    """A Deliverable B/C input is absent or malformed."""


# ---------------------------------------------------------------------------
# Deliverable A — canonical estimator primitives (Unit 9 imports these).
# ---------------------------------------------------------------------------
def _midranks_lastaxis(x: np.ndarray) -> np.ndarray:
    """Midranks (1-based, ties averaged) along the LAST axis, batched.

    Fully vectorized: one stable argsort along the last axis + run-boundary
    accumulates; no Python loop over leading axes. Matches
    ``scipy.stats.rankdata(method="average")`` per slice.
    """
    x = np.asarray(x)
    if x.ndim < 1:
        raise ValueError("_midranks_lastaxis needs >=1-D input")
    n = x.shape[-1]
    order = np.argsort(x, axis=-1, kind="stable")
    sx = np.take_along_axis(x, order, axis=-1)
    idx = np.arange(n, dtype=np.int64)
    ones = np.ones(x.shape[:-1] + (1,), dtype=bool)
    starts = np.concatenate([ones, sx[..., 1:] != sx[..., :-1]], axis=-1)
    ends = np.concatenate([starts[..., 1:], ones], axis=-1)
    start_pos = np.maximum.accumulate(np.where(starts, idx, 0), axis=-1)
    end_rev = np.flip(np.where(ends, idx, n - 1), axis=-1)
    end_pos = np.flip(np.minimum.accumulate(end_rev, axis=-1), axis=-1)
    mid_sorted = (start_pos + end_pos) / 2.0 + 1.0
    out = np.empty(x.shape, dtype=np.float64)
    np.put_along_axis(out, order, mid_sorted, axis=-1)
    return out


def _prompt_codes(prompt_ids: np.ndarray) -> tuple[np.ndarray, int]:
    """Dense integer codes (0..P-1) for exact prompt identities, 1-D input."""
    pid = np.asarray(prompt_ids)
    if pid.ndim != 1 or pid.shape[0] == 0:
        raise ValueError(f"prompt_ids must be non-empty 1-D, got shape {pid.shape}")
    uniq, codes = np.unique(pid, return_inverse=True)
    return codes.astype(np.int64), int(len(uniq))


def _auroc_from_ranks(ranks: np.ndarray, labels: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    """Per-prompt AUROC from midranks. ranks (..., P, n); labels broadcastable
    (..., P, n); mask (P, n) True on real entries (None => all real).

    Single-class prompts (n_pos == 0 or n_neg == 0 among real entries) return
    NaN — EXCLUDED from the macro, never imputed to 0.5.

    Precision: the memory-bound multiply-reduce runs in float32 — midranks are
    multiples of 0.5 with per-prompt sums far below 2**24, so the f32
    reductions are EXACT — and the final AUROC arithmetic runs in float64, so
    the result matches sklearn.roc_auc_score to ~1e-15.
    """
    lab = labels.astype(np.float32, copy=False)
    r32 = ranks.astype(np.float32, copy=False)
    if mask is None:
        n_real: Any = float(ranks.shape[-1])
        pos_rank_sum = np.einsum("...pn,...pn->...p", r32, lab).astype(np.float64)
        n_pos = lab.sum(axis=-1, dtype=np.float64)
    else:
        m32 = mask.astype(np.float32, copy=False)
        n_real = m32.sum(axis=-1, dtype=np.float64)  # (P,)
        pos_rank_sum = np.einsum("...pn,...pn->...p", r32, lab * m32).astype(np.float64)
        n_pos = (lab * m32).sum(axis=-1, dtype=np.float64)
    n_neg = n_real - n_pos
    denom = n_pos * n_neg
    with np.errstate(invalid="ignore", divide="ignore"):
        auroc = (pos_rank_sum - n_pos * (n_pos + 1.0) / 2.0) / denom
    return np.where(denom > 0, auroc, np.nan)


def within_prompt_auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Within-prompt AUROC (Mann-Whitney U / (n_pos*n_neg)) for ONE prompt.

    Ties get the midrank treatment (0.5 credit), matching
    ``sklearn.metrics.roc_auc_score``. A prompt realizing a single class
    returns ``nan`` — it is EXCLUDED from the equal-prompt macro, never imputed
    to 0.5 (plan section 7: the metric is defined ONLY on prompts realizing
    both classes).
    """
    s = np.asarray(scores, dtype=np.float64)
    lab = np.asarray(labels).astype(bool)
    if s.ndim != 1 or s.shape != lab.shape:
        raise ValueError(f"one prompt: scores/labels must be matching 1-D, got {s.shape}")
    ranks = _midranks_lastaxis(s)
    return float(_auroc_from_ranks(ranks[None, :], lab[None, :].astype(np.float32), None)[0])


def _flat_padded_layout(
    prompt_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    """Layout for the flat (ragged-group) path: stable group-contiguous order +
    (row, col) pad positions in a (P, max_n) balanced view."""
    codes, n_prompts = _prompt_codes(prompt_ids)
    order = np.argsort(codes, kind="stable")
    sorted_codes = codes[order]
    counts = np.bincount(sorted_codes, minlength=n_prompts)
    max_n = int(counts.max())
    grp_start = np.concatenate([[0], np.cumsum(counts)[:-1]])
    col = np.arange(codes.shape[0], dtype=np.int64) - grp_start[sorted_codes]
    return order, sorted_codes, col, n_prompts, max_n


def equal_prompt_macro_auroc(
    scores: np.ndarray,
    labels: np.ndarray,
    prompt_ids: np.ndarray | None = None,
) -> tuple[Any, Any]:
    """Plan section 7 PRIMARY metric: the unweighted ("equal-prompt") mean of
    within-prompt AUROCs over prompts realizing both classes.

    Two layouts, one implementation:

    - FLAT (the real-data path; ``prompt_ids`` given): ``scores`` (n_total,),
      ``labels`` (n_total,) or a batched stack (..., n_total) sharing the same
      prompt layout (e.g. permuted label stacks), ``prompt_ids`` (n_total,).
      Groups may be ragged; grouping is by EXACT prompt id.
    - BALANCED (the simulation path; ``prompt_ids=None``): ``scores``
      (..., P, n_resp), ``labels`` broadcastable to scores' shape with optional
      extra LEADING axes (e.g. a permutation stack (B, ..., P, n_resp)).

    Ties get the midrank treatment (0.5 credit). Returns
    ``(macro_auroc, n_discordant_prompts)`` — floats/ints for unbatched labels,
    arrays over the leading batch axes otherwise. ``macro_auroc`` is NaN when
    zero prompts realize both classes (never imputed).
    """
    s = np.asarray(scores, dtype=np.float64)
    lab = np.asarray(labels)
    if prompt_ids is None:
        if s.ndim < 2:
            raise ValueError(f"balanced mode needs scores (..., P, n); got shape {s.shape}")
        if lab.shape[-2:] != s.shape[-2:]:
            raise ValueError(f"labels {lab.shape} do not end in scores' (P, n) {s.shape[-2:]}")
        ranks = _midranks_lastaxis(s)
        per_prompt = _auroc_from_ranks(ranks, lab, None)  # (..., P)
        # Scalar return only for the unbatched single prompt-set (P, n) call;
        # broadcasting across extra leading label axes is handled by einsum.
        scalar_out = s.ndim == 2 and lab.ndim == 2
    else:
        if s.ndim != 1:
            raise ValueError(f"flat mode needs scores (n_total,); got shape {s.shape}")
        pid = np.asarray(prompt_ids)
        if pid.shape != s.shape or lab.shape[-1] != s.shape[0]:
            raise ValueError(
                f"flat mode shape mismatch: scores {s.shape}, labels {lab.shape}, "
                f"prompt_ids {pid.shape}"
            )
        order, srow, scol, n_prompts, max_n = _flat_padded_layout(pid)
        pad_scores = np.full((n_prompts, max_n), np.inf, dtype=np.float64)
        pad_scores[srow, scol] = s[order]
        mask = np.zeros((n_prompts, max_n), dtype=bool)
        mask[srow, scol] = True
        pad_labels = np.zeros(lab.shape[:-1] + (n_prompts, max_n), dtype=np.float32)
        pad_labels[..., srow, scol] = lab[..., order].astype(np.float32)
        ranks = _midranks_lastaxis(pad_scores)  # padding at +inf takes top ranks
        per_prompt = _auroc_from_ranks(ranks, pad_labels, mask)
        scalar_out = lab.ndim == 1
    disc = ~np.isnan(per_prompt)
    cnt = disc.sum(axis=-1)
    tot = np.where(disc, per_prompt, 0.0).sum(axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        macro = np.where(cnt > 0, tot / np.maximum(cnt, 1), np.nan)
    if scalar_out:
        return float(np.asarray(macro).reshape(-1)[0]), int(np.asarray(cnt).reshape(-1)[0])
    return macro, cnt


def permute_labels_within_prompt(
    labels: np.ndarray,
    prompt_ids: np.ndarray,
    rng: np.random.Generator,
    n_perm: int | None = None,
) -> np.ndarray:
    """Plan section 7 permutation: labels shuffled ONLY WITHIN exact prompt,
    never across prompts. The power simulation calls THIS function, so power is
    computed for the actual registered test.

    ``labels``: (n_total,) or a batched stack (..., n_total) sharing one prompt
    layout. ``n_perm=None`` returns one permuted copy with ``labels``' shape;
    ``n_perm=B`` returns ``(B, *labels.shape)`` independent permutations. One
    code path — the single form IS the batched form at B=1. Per-prompt label
    multisets (hence per-prompt class counts) are preserved by construction.
    """
    lab = np.asarray(labels)
    codes, _ = _prompt_codes(prompt_ids)
    n = codes.shape[0]
    if lab.shape[-1] != n:
        raise ValueError(f"labels last axis {lab.shape[-1]} != n prompt ids {n}")
    single = n_perm is None
    n_b = 1 if single else int(n_perm)
    if n_b < 1:
        raise ValueError(f"n_perm must be >= 1, got {n_perm}")
    base = np.argsort(codes, kind="stable")
    labels_base = lab[..., base]
    # Sort key = integer code + U[0,1): primary group, secondary random =>
    # independent uniform within-group permutation per leading index.
    u = rng.random((n_b,) + lab.shape[:-1] + (n,))
    key = codes.astype(np.float64) + u
    order = np.argsort(key, axis=-1, kind="stable")
    out = np.empty((n_b,) + lab.shape, dtype=lab.dtype)
    np.put_along_axis(out, order, np.broadcast_to(labels_base, (n_b,) + lab.shape), axis=-1)
    return out[0] if single else out


def binormal_shift(auroc: float) -> float:
    """Binormal class-conditional shift with true within-prompt AUROC = target.

    Positives ~ N(delta, 1), negatives ~ N(0, 1) =>
    AUROC = P(X_pos > X_neg) = Phi(delta / sqrt(2)) => delta = sqrt(2) Phi^-1(A).
    """
    if not 0.0 < auroc < 1.0:
        raise ValueError(f"auroc must be in (0, 1), got {auroc}")
    return float(math.sqrt(2.0) * sps.norm.ppf(auroc))


# ---------------------------------------------------------------------------
# Deliverable B — discordance measurement + projection + lower bounds.
# ---------------------------------------------------------------------------
def clopper_pearson_lower(x: int, n: int, conf: float = 0.95) -> float:
    """One-sided exact (Clopper-Pearson) lower confidence bound on a binomial
    proportion: Beta.ppf(1-conf, x, n-x+1); 0 when x == 0."""
    if not 0 <= x <= n or n <= 0:
        raise ValueError(f"need 0 <= x <= n, n > 0; got x={x}, n={n}")
    if x == 0:
        return 0.0
    return float(sps.beta.ppf(1.0 - conf, x, n - x + 1))


def project_discordance_plugin(k: np.ndarray, n: np.ndarray, m_resp: int) -> np.ndarray:
    """Plug-in per-prompt P(discordant in m_resp draws) = 1 - p^m - (1-p)^m with
    p = k/n. BIASED LOW at the p in {0,1} boundary (a prompt unanimous in the
    pilot projects to EXACTLY zero future discordance)."""
    k = np.asarray(k, dtype=np.float64)
    n = np.asarray(n, dtype=np.float64)
    if np.any(n <= 0) or np.any(k < 0) or np.any(k > n):
        raise ValueError("need 0 <= k <= n with n > 0 per prompt")
    p = k / n
    return 1.0 - p**m_resp - (1.0 - p) ** m_resp


def _beta_moment(a: np.ndarray, b: np.ndarray, m: int) -> np.ndarray:
    """E[p^m] under Beta(a, b) = B(a+m, b)/B(a, b), via gammaln (vectorized)."""
    return np.exp(gammaln(a + m) - gammaln(a) + gammaln(a + b) - gammaln(a + b + m))


def project_discordance_jeffreys(k: np.ndarray, n: np.ndarray, m_resp: int) -> np.ndarray:
    """Jeffreys-shrunk per-prompt 30-draw discordance projection.

    Posterior mean of 1 - p^m - (1-p)^m under p ~ Beta(k+1/2, n-k+1/2)
    (Jeffreys posterior). Handles the k in {0, n} boundary where the plug-in
    degenerates to certainty; needs no fitted hyperparameters (an empirical
    Bayes fit over 5 prompts/cell would itself be unstable).
    """
    k = np.asarray(k, dtype=np.float64)
    n = np.asarray(n, dtype=np.float64)
    if np.any(n <= 0) or np.any(k < 0) or np.any(k > n):
        raise ValueError("need 0 <= k <= n with n > 0 per prompt")
    a = k + 0.5
    b = n - k + 0.5
    return 1.0 - _beta_moment(a, b, m_resp) - _beta_moment(b, a, m_resp)


def credible_lower_on_projection(
    ks: np.ndarray,
    ns: np.ndarray,
    m_resp: int,
    rng: np.random.Generator,
    n_draws: int = 4000,
    q: float = 0.05,
) -> float:
    """Lower q-quantile CREDIBLE bound on the cell's mean m_resp-draw
    discordance probability: p_j ~ Beta(k_j+1/2, n_j-k_j+1/2) independently,
    statistic = mean_j (1 - p_j^m - (1-p_j)^m). Bayesian (Jeffreys), labeled as
    such wherever reported — NOT a frequentist bound."""
    ks = np.asarray(ks, dtype=np.float64)
    ns = np.asarray(ns, dtype=np.float64)
    p = rng.beta(ks + 0.5, ns - ks + 0.5, size=(n_draws, ks.shape[0]))
    stat = (1.0 - p**m_resp - (1.0 - p) ** m_resp).mean(axis=1)
    return float(np.quantile(stat, q))


def expected_cells(row: str) -> list[str]:
    """The 12 registered cell names for ``row`` (gen CellWork convention)."""
    rf = F.FRAMES[row]
    return sorted(f"{row}__{fr.name}__{st.name}" for fr in rf.frames for st in rf.strata)


@dataclass
class RowLabelProfile:
    """Per-prompt pilot label counts for one row: cell -> [(k_pos, n_labeled)]."""

    row: str
    judged: bool
    cells: dict[str, list[tuple[int, int]]] = field(default_factory=dict)
    missing_cells: list[str] = field(default_factory=list)
    n_unlabeled_prompts: int = 0
    artifact_dir: str = ""


def _profile_from_judge_cell(body: dict[str, Any]) -> list[tuple[int, int]]:
    """(k, n) per prompt from a judge cell body. Only ``scored`` verdicts carry
    a label (drop-never-coerce); pending/human_adjudication units shrink n."""
    if body.get("schema") != J.JUDGE_SCHEMA:
        raise PowerInputError(f"judge cell schema {body.get('schema')!r} != {J.JUDGE_SCHEMA!r}")
    per_item: dict[str, list[bool]] = {}
    for v in body["verdicts"].values():
        per_item.setdefault(v["item_id"], [])
        if v["judge_status"] == "scored":
            if v["binary_label"] is None:
                raise PowerInputError(f"scored verdict {v['unit_id']} carries no binary_label")
            per_item[v["item_id"]].append(bool(v["binary_label"]))
    return [(sum(labs), len(labs)) for _, labs in sorted(per_item.items())]


def _profile_from_objective_cell(path: Path) -> list[tuple[int, int]]:
    """(k, n) per prompt from an objective-labels JSONL (status=='labeled' only)."""
    per_item: dict[str, list[bool]] = {}
    with path.open() as fh:
        for line in fh:
            rec = json.loads(line)
            if rec.get("schema") != "i2658-objective-labels-v1":
                raise PowerInputError(f"{path}: unexpected schema {rec.get('schema')!r}")
            iid = rec["manifest"]["prompt_id"]
            per_item.setdefault(iid, [])
            if rec["status"] == "labeled":
                if rec["label"] is None:
                    raise PowerInputError(f"{path}: labeled row with null label ({iid})")
                per_item[iid].append(bool(rec["label"]))
    return [(sum(labs), len(labs)) for _, labs in sorted(per_item.items())]


def load_pilot_label_profile(
    out_root: Path, split: str, rows: list[str] | None = None
) -> dict[str, RowLabelProfile]:
    """Load pilot labels per row/cell into (k, n) per-prompt counts.

    Judged rows read unit-6 judge cell verdicts; correctness rows read unit-4
    objective label JSONLs. A MISSING artifact is recorded in ``missing_cells``
    (the gate consumes absence as NOT-ESTIMABLE); a PRESENT-but-malformed
    artifact raises. Prompts with zero labeled responses are counted in
    ``n_unlabeled_prompts`` and excluded from the (k, n) pool — never coerced.
    """
    out: dict[str, RowLabelProfile] = {}
    for row in rows or list(C.ROW_IDS):
        judged = C.CONSTRUCTS[row].judge_scored
        prof = RowLabelProfile(row=row, judged=judged)
        for cell in expected_cells(row):
            if judged:
                path = Path(out_root) / "judge" / split / row / f"{cell}.json"
                prof.artifact_dir = str(path.parent)
                if not path.exists():
                    prof.missing_cells.append(cell)
                    continue
                pairs = _profile_from_judge_cell(json.loads(path.read_text()))
            else:
                path = Path(out_root) / "objective_labels" / split / f"{cell}.jsonl"
                prof.artifact_dir = str(path.parent)
                if not path.exists():
                    prof.missing_cells.append(cell)
                    continue
                pairs = _profile_from_objective_cell(path)
            kept = [(k, n) for k, n in pairs if n > 0]
            prof.n_unlabeled_prompts += len(pairs) - len(kept)
            prof.cells[cell] = kept
        out[row] = prof
    return out


def profile_fingerprint(profiles: dict[str, RowLabelProfile]) -> str:
    """Content address of the (k, n) pools every simulation consumes."""
    body = {
        row: {cell: sorted(pairs) for cell, pairs in p.cells.items()}
        for row, p in sorted(profiles.items())
    }
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()


def measure_discordance(profiles: dict[str, RowLabelProfile], seed: int = 0) -> dict[str, Any]:
    """Deliverable B: per-cell discordance report + the sizing lower bound.

    Per cell: raw 10-draw discordant count/rate, its one-sided 95%
    Clopper-Pearson lower bound, the 30-draw plug-in and Jeffreys-shrunk
    projections (both REPORTED, labeled), and the SIZING bound
    ``max(cp_lower_10, jeffreys 5% credible bound on the 30-draw projection)``.

    Validity: P(discordant | m responses) is INCREASING in m (1 - p^m - (1-p)^m
    is increasing in m for p in (0,1)), so the 10-draw CP lower bound is a
    VALID (conservative) 95% lower confidence bound on the 30-draw rate; the
    Jeffreys credible bound sharpens it using the projection and is labeled
    Bayesian. Sizing uses the max of the two — never the raw 10-draw rate
    carried across as-is (it is biased LOW for 30 draws).
    """
    rng = np.random.default_rng(_unit_seed("discordance-credible", seed))
    m_prod = REGISTERED.responses_per_prompt_production
    target = REGISTERED.discordant_target_per_cell
    rows_out: dict[str, Any] = {}
    for row, prof in sorted(profiles.items()):
        cells_out: dict[str, Any] = {}
        for cell, pairs in sorted(prof.cells.items()):
            if not pairs:
                cells_out[cell] = {"status": "not-estimable", "detail": "zero labeled prompts"}
                continue
            ks = np.array([k for k, _ in pairs], dtype=np.int64)
            ns = np.array([n for _, n in pairs], dtype=np.int64)
            disc = (ks > 0) & (ks < ns)
            m_prompts = int(len(pairs))
            x10 = int(disc.sum())
            cp10 = clopper_pearson_lower(x10, m_prompts)
            plugin = float(project_discordance_plugin(ks, ns, m_prod).mean())
            jeff = float(project_discordance_jeffreys(ks, ns, m_prod).mean())
            cred = credible_lower_on_projection(ks, ns, m_prod, rng)
            # Zero observed discordant prompts carry NO evidence of discordance:
            # the Jeffreys credible bound is then purely prior-driven, and sizing
            # production off it would be a silent default (fail-fast rule). The
            # Bayesian bound is still REPORTED; it just never sizes at x10 == 0.
            if x10 == 0:
                lb = 0.0
                rule = "unbounded — zero observed discordant prompts (prior-only bound never sizes)"
            else:
                lb = max(cp10, cred)
                rule = "max(cp_lower95_10draw, credible_lower05_30draw)"
            cells_out[cell] = {
                "status": "measured",
                "m_prompts": m_prompts,
                "pilot_draws_per_prompt": [int(n) for n in ns],
                "x_discordant_10draw": x10,
                "raw_rate_10draw": x10 / m_prompts,
                "cp_lower95_10draw": cp10,
                "projected_rate_30draw_plugin": plugin,
                "projected_rate_30draw_jeffreys": jeff,
                "credible_lower05_30draw_jeffreys": cred,
                "sizing_lower_bound": lb,
                "sizing_lower_bound_rule": rule,
                "n_required_for_target": (int(math.ceil(target / lb)) if lb > 0 else None),
            }
        rows_out[row] = {
            "judged": prof.judged,
            "missing_cells": prof.missing_cells,
            "n_unlabeled_prompts": prof.n_unlabeled_prompts,
            "cells": cells_out,
        }
    return {
        "schema": DISCORDANCE_SCHEMA,
        "responses_per_prompt_pilot": REGISTERED.responses_per_prompt_pilot,
        "responses_per_prompt_production": m_prod,
        "discordant_target_per_cell": target,
        "note": (
            "raw_rate_10draw is a biased-LOW estimate of the 30-draw rate "
            "(P(discordant) increases in responses/prompt); sizing uses the "
            "projection's lower bound, never the raw rate"
        ),
        "profile_sha256": profile_fingerprint(profiles),
        "rows": rows_out,
    }


# ---------------------------------------------------------------------------
# Deliverable C — clustered power simulation + production-N selection.
# ---------------------------------------------------------------------------
def _unit_seed(*parts: Any) -> int:
    h = hashlib.sha256("|".join(str(p) for p in parts).encode()).digest()
    return int.from_bytes(h[:8], "big") % (2**63)


def simulate_power(
    cell_pools: dict[str, list[tuple[int, int]]],
    n_prompts_per_cell: int,
    effect_auroc: float,
    *,
    alpha: float,
    n_reps: int,
    n_perm: int,
    m_resp: int,
    seed: int,
    element_budget: int = CHUNK_ELEMENT_BUDGET,
) -> dict[str, Any]:
    """Clustered power simulation for one (row, N, effect) unit.

    The resampling unit is the PROMPT (clustered), never the response: each
    simulated prompt is a with-replacement draw from the cell's pilot prompt
    pool, carrying a per-prompt positive probability drawn from that prompt's
    Jeffreys posterior Beta(k+1/2, n-k+1/2) (pilot uncertainty propagates into
    power). Per-response labels ~ Bernoulli(p_prompt); scores are binormal with
    shift sqrt(2)*Phi^-1(effect_auroc) so the true within-prompt AUROC equals
    the target effect. The plan section 7 statistic and the within-prompt
    permutation test run via Deliverable A's own functions
    (``equal_prompt_macro_auroc`` / ``permute_labels_within_prompt``); p-value
    is the plus-one one-sided (greater) permutation p; power is estimated at
    the caller's alpha (registered: the Holm worst case 0.05/11).

    Replicates and permutations are processed in chunks sized to
    ``element_budget`` float32 elements so peak RSS stays well under a few GB;
    a replicate with ZERO discordant prompts cannot reject (counted in the
    power denominator; reported).
    """
    cells = sorted(cell_pools)
    if not cells:
        raise PowerInputError("simulate_power needs >= 1 cell pool")
    if 1.0 / (n_perm + 1) > alpha + 1e-15:
        raise PowerInputError(
            f"unrejectable test configuration: min achievable permutation p "
            f"1/{n_perm + 1} > alpha {alpha:.6g} — raise n_perm (registered "
            f"{REGISTERED.n_permutations} gives exact size 3/660 at the Holm worst case)"
        )
    for cname in cells:
        if not cell_pools[cname]:
            raise PowerInputError(f"cell {cname!r} has an empty prompt pool")
    n_cells = len(cells)
    n_p = n_cells * n_prompts_per_cell
    delta = binormal_shift(effect_auroc)
    rng = np.random.default_rng(seed)
    pids = np.repeat(np.arange(n_p), m_resp)

    # Divide the budget so a permutation chunk holds >= ~8 permutations per
    # replicate chunk (avoids degenerating the inner chunking to bc=1).
    rep_chunk = max(1, min(n_reps, element_budget // max(1, n_p * m_resp * 8)))
    rejections = 0
    zero_disc = 0
    stat_sum = 0.0
    stat_sq = 0.0
    stat_n = 0
    disc_sum = 0.0
    done_reps = 0
    while done_reps < n_reps:
        rc = min(rep_chunk, n_reps - done_reps)
        p_cols = []
        for cname in cells:
            pool = cell_pools[cname]
            ks = np.array([k for k, _ in pool], dtype=np.float64)
            ns = np.array([n for _, n in pool], dtype=np.float64)
            idx = rng.integers(0, len(pool), size=(rc, n_prompts_per_cell))
            p_cols.append(rng.beta(ks[idx] + 0.5, ns[idx] - ks[idx] + 0.5))
        p = np.concatenate(p_cols, axis=1)  # (rc, n_p)
        assert p.shape == (rc, n_p)
        labels = rng.random((rc, n_p, m_resp)) < p[..., None]
        scores = rng.standard_normal((rc, n_p, m_resp)) + delta * labels
        obs, n_disc = equal_prompt_macro_auroc(scores, labels)  # (rc,), (rc,)
        assert obs.shape == (rc,)
        exceed = np.zeros(rc, dtype=np.int64)
        labels_flat = labels.reshape(rc, n_p * m_resp)
        perm_chunk = max(1, element_budget // max(1, rc * n_p * m_resp))
        done_perm = 0
        while done_perm < n_perm:
            bc = min(perm_chunk, n_perm - done_perm)
            perm = permute_labels_within_prompt(labels_flat, pids, rng, n_perm=bc)
            pstat, _ = equal_prompt_macro_auroc(scores, perm.reshape(bc, rc, n_p, m_resp))
            assert pstat.shape == (bc, rc)
            exceed += (pstat >= obs[None, :] - 1e-12).sum(axis=0)
            done_perm += bc
        pvals = (1.0 + exceed) / (n_perm + 1.0)
        estimable = ~np.isnan(obs)
        rejections += int(((pvals <= alpha + 1e-15) & estimable).sum())
        zero_disc += int((~estimable).sum())
        stat_sum += float(np.where(estimable, obs, 0.0).sum())
        stat_sq += float(np.where(estimable, obs**2, 0.0).sum())
        stat_n += int(estimable.sum())
        disc_sum += float(n_disc.sum())
        done_reps += rc
    power = rejections / n_reps
    mean_stat = stat_sum / stat_n if stat_n else None
    sd_stat = (
        math.sqrt(max(0.0, stat_sq / stat_n - (stat_sum / stat_n) ** 2)) if stat_n > 1 else None
    )
    return {
        "power": power,
        "mc_half_width_95": 1.96 * math.sqrt(max(power * (1 - power), 1e-12) / n_reps),
        "mean_stat": mean_stat,
        "sd_stat": sd_stat,
        "mean_discordant_prompts": disc_sum / n_reps,
        "n_zero_discordant_reps": zero_disc,
        "n_reps": n_reps,
        "n_perm": n_perm,
        "rep_chunk": rep_chunk,
    }


class PowerLedger:
    """Per-unit persistence: atomic-append JSONL + parameter-keyed resume.

    The resume key covers EVERY output-affecting generating parameter (row, N,
    effect, alpha, n_perm, n_reps, responses/prompt, seed, and the ROW-scoped
    pool fingerprint — the unit consumes one row's pools, so shard/--rows runs
    and the full-row run share keys) — never the bytes of a recomputed float
    array. Single-writer per file (the sharded dispatch merges shard ledgers by
    concatenation; keys are content-addressed so a merged file resumes cleanly);
    each append is flushed + fsynced before the unit is considered complete.
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._done: dict[str, dict[str, Any]] = {}
        if self.path.exists():
            with self.path.open() as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    if rec.get("schema") != POWER_SCHEMA:
                        raise PowerInputError(
                            f"{self.path}: foreign schema {rec.get('schema')!r} in power ledger"
                        )
                    self._done[rec["key"]] = rec

    @staticmethod
    def unit_key(**params: Any) -> str:
        required = {
            "row",
            "n_prompts_per_cell",
            "effect_auroc",
            "alpha",
            "n_reps",
            "n_perm",
            "responses_per_prompt",
            "seed",
            "profile_sha256",
        }
        missing = required - set(params)
        if missing:
            raise PowerInputError(f"unit_key missing generating parameters: {sorted(missing)}")
        body = json.dumps({k: params[k] for k in sorted(required)}, sort_keys=True)
        return hashlib.sha256(body.encode()).hexdigest()

    def get(self, key: str) -> dict[str, Any] | None:
        return self._done.get(key)

    def append(self, record: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as fh:
            fh.write(json.dumps(record, sort_keys=True) + "\n")
            fh.flush()
            import os

            os.fsync(fh.fileno())
        self._done[record["key"]] = record


@dataclass
class _UnitCounter:
    done: int = 0
    cap: int = 0
    t0: float = field(default_factory=time.time)
    first_unit_wall: float | None = None


def _run_power_unit(
    ledger: PowerLedger,
    counter: _UnitCounter,
    *,
    row: str,
    cell_pools: dict[str, list[tuple[int, int]]],
    n_prompts_per_cell: int,
    effect_auroc: float,
    reg: PowerRegistry,
    n_reps: int,
    n_perm: int,
    seed: int,
    prof_sha: str,
    purpose: str,
) -> dict[str, Any]:
    params = {
        "row": row,
        "n_prompts_per_cell": int(n_prompts_per_cell),
        "effect_auroc": float(effect_auroc),
        "alpha": reg.alpha_worst_case,
        "n_reps": int(n_reps),
        "n_perm": int(n_perm),
        "responses_per_prompt": reg.responses_per_prompt_production,
        "seed": int(seed),
        "profile_sha256": prof_sha,
    }
    key = PowerLedger.unit_key(**params)
    prior = ledger.get(key)
    counter.done += 1
    if prior is not None:
        print(
            f"[power] unit {counter.done}/{counter.cap} row={row} N={n_prompts_per_cell} "
            f"resume-skip elapsed={time.time() - counter.t0:.1f}s",
            flush=True,
        )
        return prior
    t_unit = time.time()
    sim = simulate_power(
        cell_pools,
        n_prompts_per_cell,
        effect_auroc,
        alpha=reg.alpha_worst_case,
        n_reps=n_reps,
        n_perm=n_perm,
        m_resp=reg.responses_per_prompt_production,
        seed=_unit_seed("i2658-power", key),
    )
    wall = time.time() - t_unit
    rec = {
        "schema": POWER_SCHEMA,
        "key": key,
        **params,
        **sim,
        "purpose": purpose,
        "elapsed_s": wall,
    }
    ledger.append(rec)
    print(
        f"[power] unit {counter.done}/{counter.cap} row={row} N={n_prompts_per_cell} "
        f"elapsed={time.time() - counter.t0:.1f}s "
        f"(unit={wall:.1f}s effect={effect_auroc} power={sim['power']:.3f})",
        flush=True,
    )
    if counter.first_unit_wall is None:
        counter.first_unit_wall = wall
        remaining = max(0, counter.cap - counter.done)
        print(
            f"[power] ops arithmetic: measured one-unit wall {wall:.1f}s x <= {remaining} "
            f"remaining units -> projected <= {wall * remaining / 3600:.2f} h "
            f"(reps={n_reps} x perms={n_perm} x prompts={len(cell_pools) * n_prompts_per_cell} "
            f"x responses={reg.responses_per_prompt_production})",
            flush=True,
        )
    return rec


def select_production_n(
    profiles: dict[str, RowLabelProfile],
    discordance: dict[str, Any],
    ledger: PowerLedger,
    *,
    reg: PowerRegistry = REGISTERED,
    n_reps: int | None = None,
    n_perm: int | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    """Deliverable B+C: the plan section 4 production-N rule.

    N_common = max(30, max over cells ceil(20 / sizing lower bound),
    max over rows of the smallest N with simulated power >= target at the
    registered primary effect), searched by BISECTION (doubling then integer
    bisection; ~6 evaluations per row, never a full grid). Rows/cells whose
    requirement is unbounded (lower bound 0, or power target unreachable by
    the cap) yield status not-estimable — the gate PARKs on them.
    """
    n_reps = reg.n_replicates if n_reps is None else n_reps
    n_perm = reg.n_permutations if n_perm is None else n_perm
    # Compare against the MODULE registry, never the caller's (a smoke/override
    # registry matching itself would let a downsized selection pass the gate).
    registered_match = (
        reg == REGISTERED
        and n_reps == REGISTERED.n_replicates
        and n_perm == REGISTERED.n_permutations
    )
    prof_sha = profile_fingerprint(profiles)
    rows = sorted(r for r, p in profiles.items() if p.cells and not p.missing_cells)
    # Per-UNIT content address: a simulation unit consumes exactly ONE row's
    # (k, n) pools, so its ledger key fingerprints THAT row's pools alone.
    # This keeps unit keys IDENTICAL between a sharded --rows run and the final
    # full-row invocation (whose passed-in profile dicts differ), so the
    # recorded P2-P3 dispatch (row shards warm one shared ledger; the final
    # full-row run selects N) resumes instead of recomputing every unit. The
    # artifact-level profile_sha256 stays the WHOLE-profile fingerprint — the
    # gate-time freshness cross-check keys on it.
    row_prof_sha = {row: profile_fingerprint({row: profiles[row]}) for row in rows}
    counter = _UnitCounter(cap=len(rows) * (9 + len(reg.power_curve_effects)))

    # Discordance requirement (>= 20 expected discordant/cell under the bound).
    binding_disc: tuple[str, str, int] | None = None
    unbounded: list[str] = []
    for row in sorted(profiles):
        for cell, c in sorted(discordance["rows"].get(row, {}).get("cells", {}).items()):
            if c.get("status") != "measured":
                unbounded.append(cell)
                continue
            req = c["n_required_for_target"]
            if req is None:
                unbounded.append(cell)
            elif binding_disc is None or req > binding_disc[2]:
                binding_disc = (row, cell, req)

    def power_at(row: str, n_val: int) -> float:
        rec = _run_power_unit(
            ledger,
            counter,
            row=row,
            cell_pools=profiles[row].cells,
            n_prompts_per_cell=n_val,
            effect_auroc=reg.primary_effect_auroc,
            reg=reg,
            n_reps=n_reps,
            n_perm=n_perm,
            seed=seed,
            prof_sha=row_prof_sha[row],
            purpose="bisection",
        )
        return float(rec["power"])

    per_row: dict[str, Any] = {}
    for row in rows:
        lo = reg.prompts_per_cell_floor
        if power_at(row, lo) >= reg.power_target:
            per_row[row] = {"n_power": lo, "status": "measured"}
            continue
        prev, cur = lo, lo
        found = None
        while cur < reg.bisection_cap:
            prev, cur = cur, min(cur * 2, reg.bisection_cap)
            if power_at(row, cur) >= reg.power_target:
                found = cur
                break
        if found is None:
            per_row[row] = {
                "n_power": None,
                "status": f"not-estimable — power < {reg.power_target} at cap {reg.bisection_cap}",
            }
            continue
        lo_b, hi_b = prev, found
        while hi_b - lo_b > 1:
            mid = (lo_b + hi_b) // 2
            if power_at(row, mid) >= reg.power_target:
                hi_b = mid
            else:
                lo_b = mid
        per_row[row] = {"n_power": hi_b, "status": "measured"}

    power_ns = [v["n_power"] for v in per_row.values() if v["n_power"] is not None]
    binding_power = None
    if power_ns:
        binding_power = max(
            ((r, v["n_power"]) for r, v in per_row.items() if v["n_power"] is not None),
            key=lambda t: t[1],
        )
    all_rows_estimable = bool(rows) and all(v["n_power"] is not None for v in per_row.values())
    rows_incomplete = sorted(r for r, p in profiles.items() if p.missing_cells or not p.cells)
    # The common N must cover EVERY registered row: a row with missing/empty
    # label artifacts makes the selection not-estimable (never silently skipped).
    estimable = (
        all_rows_estimable and not unbounded and binding_disc is not None and not rows_incomplete
    )
    n_common = None
    if estimable:
        n_common = max(
            reg.prompts_per_cell_floor,
            binding_disc[2],
            max(power_ns),
        )

    curve: dict[str, Any] = {}
    if n_common is not None:
        for row in rows:
            curve[row] = {}
            for eff in reg.power_curve_effects:
                rec = _run_power_unit(
                    ledger,
                    counter,
                    row=row,
                    cell_pools=profiles[row].cells,
                    n_prompts_per_cell=n_common,
                    effect_auroc=eff,
                    reg=reg,
                    n_reps=n_reps,
                    n_perm=n_perm,
                    seed=seed,
                    prof_sha=row_prof_sha[row],
                    purpose="curve",
                )
                curve[row][f"{eff:.2f}"] = {
                    "power": rec["power"],
                    "mc_half_width_95": rec["mc_half_width_95"],
                    "mean_stat": rec["mean_stat"],
                    "sd_stat": rec["sd_stat"],
                }

    return {
        "schema": PRODUCTION_N_SCHEMA,
        "registered": asdict(reg) | {"alpha_worst_case": reg.alpha_worst_case},
        "registered_match": registered_match,
        "profile_sha256": prof_sha,
        "rows_simulated": rows,
        "rows_missing_labels": sorted(
            r for r, p in profiles.items() if p.missing_cells or not p.cells
        ),
        "per_row_power_n": per_row,
        "binding_discordance_cell": (
            {"row": binding_disc[0], "cell": binding_disc[1], "n_required": binding_disc[2]}
            if binding_disc
            else None
        ),
        "cells_with_unbounded_requirement": sorted(unbounded),
        "binding_power_row": (
            {"row": binding_power[0], "n_required": binding_power[1]} if binding_power else None
        ),
        "n_common": n_common,
        "status": "measured" if n_common is not None else "not-estimable",
        "power_curve_at_n_common": curve,
        "first_unit_wall_s": counter.first_unit_wall,
        "metadata": as_metadata_dict(git_provenance(), phase="p3-power"),
    }


# ---------------------------------------------------------------------------
# Deliverable D — cost report (measured-or-not-estimable; projections labeled).
# ---------------------------------------------------------------------------
def _not_estimable(missing: Path | str, detail: str = "") -> dict[str, Any]:
    return {
        "value": None,
        "basis": "not-estimable",
        "missing_artifact": str(missing),
        "detail": detail or "artifact absent — never substituted with a projection",
    }


def cost_report(out_root: Path, gen_root: Path, split: str, n_common: int | None) -> dict[str, Any]:
    """Deliverable D: measured GPU-hours / Batch-API dollars / human load.

    HARD RULE implemented here: an absent measurement is ``not-estimable`` with
    the missing artifact NAMED; every projected figure carries
    ``basis: "projected"`` in the same field; a projection is NEVER reported as
    a measurement (plan section 10's ~24,000 pilot judge draws is a PROJECTION).
    """
    out_root = Path(out_root)
    judged_rows = [r for r in C.ROW_IDS if C.CONSTRUCTS[r].judge_scored]
    reg = REGISTERED

    # --- GPU hours -------------------------------------------------------
    timing_path = out_root / PILOT_TIMING_REL
    gpu: dict[str, Any] = {}
    projected_production_gpu_h: float | None = None
    if timing_path.exists():
        t = json.loads(timing_path.read_text())
        for fld in ("wall_hours", "gpu_count", "n_responses"):
            if fld not in t:
                raise PowerInputError(f"{timing_path}: missing field {fld!r}")
        measured = float(t["wall_hours"]) * float(t["gpu_count"])
        gpu["measured_pilot_gpu_h"] = {
            "value": measured,
            "basis": "measured",
            "artifact": str(timing_path),
        }
        if n_common is not None:
            overhead = float(t.get("fixed_overhead_hours", 0.0)) * float(t["gpu_count"])
            marginal = (measured - overhead) / float(t["n_responses"])
            n_prod_resp = (
                len(C.ROW_IDS)
                * C.PILOT.cells_per_row
                * n_common
                * reg.responses_per_prompt_production
                * 2  # dev + sealed-test banks (plan section 4)
            )
            projected_production_gpu_h = 2 * overhead + marginal * n_prod_resp
            gpu["projected_production_gpu_h"] = {
                "value": projected_production_gpu_h,
                "basis": "projected",
                "formula": (
                    "2*overhead_gpu_h + marginal_gpu_h_per_response * "
                    f"(11 rows x 12 cells x N={n_common} x 30 responses x 2 banks "
                    f"= {n_prod_resp})"
                ),
            }
        else:
            gpu["projected_production_gpu_h"] = _not_estimable(
                "production N (power/production_n.json status=measured)",
                "no fixed production N yet — projection undefined",
            )
    else:
        gpu["measured_pilot_gpu_h"] = _not_estimable(timing_path)
        gpu["projected_production_gpu_h"] = _not_estimable(
            timing_path, "no measured pilot marginal rate to project from"
        )

    # --- Anthropic Batch API ---------------------------------------------
    api: dict[str, Any] = {}
    realized_draws = 0
    judged_cells_found = 0
    for row in judged_rows:
        jdir = out_root / "judge" / split / row
        for cell in expected_cells(row):
            p = jdir / f"{cell}.json"
            if not p.exists():
                continue
            body = json.loads(p.read_text())
            judged_cells_found += 1
            realized_draws += int(sum(body["counters"].values()))
    if judged_cells_found:
        api["realized_judge_draws"] = {
            "value": realized_draws,
            "basis": "measured",
            "artifact": str(out_root / "judge" / split),
            "detail": f"summed draw counters over {judged_cells_found} judged cell bodies",
        }
    else:
        api["realized_judge_draws"] = _not_estimable(out_root / "judge" / split)
    spend_path = out_root / JUDGE_SPEND_REL
    if spend_path.exists():
        srec = json.loads(spend_path.read_text())
        if "dollars" not in srec:
            raise PowerInputError(f"{spend_path}: missing field 'dollars'")
        api["measured_dollars"] = {
            "value": float(srec["dollars"]),
            "basis": "measured",
            "artifact": str(spend_path),
        }
    else:
        api["measured_dollars"] = _not_estimable(spend_path)
    api["projected_pilot_judge_draws"] = {
        "value": len(judged_rows) * C.PILOT.responses_per_row * int(C.JUDGE["n_draws"]),
        "basis": "projected",
        "detail": (
            f"plan section 10 projection: {len(judged_rows)} judged traits x "
            f"{C.PILOT.responses_per_row} answers x {C.JUDGE['n_draws']} draws — "
            "a PROJECTION, never quote as realized"
        ),
    }
    if n_common is not None:
        api["projected_production_judge_draws"] = {
            "value": len(judged_rows)
            * C.PILOT.cells_per_row
            * n_common
            * reg.responses_per_prompt_production
            * int(C.JUDGE["n_draws"])
            * 2,
            "basis": "projected",
            "detail": "8 judged rows x 12 cells x N x 30 responses x 5 draws x 2 banks",
        }
    else:
        api["projected_production_judge_draws"] = _not_estimable(
            "production N (power/production_n.json status=measured)"
        )

    # --- Human annotation --------------------------------------------------
    human: dict[str, Any] = {}
    human["required_minimum_adjudications"] = {
        "value": len(judged_rows) * 200,
        "basis": "projected",
        "detail": (
            "plan section 3 sizing: >= 100 adjudicated positives AND >= 100 negatives "
            f"per judged trait x {len(judged_rows)} judged traits (a floor, not a spend)"
        ),
    }
    audit_path = out_root / HUMAN_AUDIT_REL
    if audit_path.exists():
        audit = json.loads(audit_path.read_text())
        human["realized_adjudications"] = {
            "value": len(audit.get("rows", [])),
            "basis": "measured",
            "artifact": str(audit_path),
        }
    else:
        human["realized_adjudications"] = _not_estimable(audit_path)
    if judged_cells_found:
        n_adjud_queue = 0
        for row in judged_rows:
            for cell in expected_cells(row):
                p = out_root / "judge" / split / row / f"{cell}.json"
                if p.exists():
                    n_adjud_queue += int(json.loads(p.read_text())["n_human_adjudication"])
        human["realized_adjudication_queue"] = {
            "value": n_adjud_queue,
            "basis": "measured",
            "artifact": str(out_root / "judge" / split),
            "detail": "judge-side units routed to human adjudication (retry-exhausted)",
        }
    else:
        human["realized_adjudication_queue"] = _not_estimable(out_root / "judge" / split)

    # --- Envelope check (plan section 10) ----------------------------------
    pilot_ceiling = 8.0
    total_ceiling = 80.0
    measured_pilot = gpu["measured_pilot_gpu_h"]["value"]
    envelope: dict[str, Any] = {
        "pilot_ceiling_gpu_h": pilot_ceiling,
        "total_ceiling_gpu_h": total_ceiling,
    }
    if measured_pilot is not None and projected_production_gpu_h is not None:
        total = measured_pilot + projected_production_gpu_h
        envelope |= {
            "projected_total_gpu_h": total,
            "margin_gpu_h": total_ceiling - total,
            "pilot_within_ceiling": bool(measured_pilot <= pilot_ceiling),
            "within_envelope": bool(total <= total_ceiling and measured_pilot <= pilot_ceiling),
            "basis": "measured pilot + projected production (labeled)",
        }
    else:
        envelope |= {
            "projected_total_gpu_h": None,
            "margin_gpu_h": None,
            "within_envelope": None,
            "basis": "not-estimable — see gpu_hours fields for the missing artifacts",
        }

    return {
        "schema": COST_SCHEMA,
        "split": split,
        "gpu_hours": gpu,
        "api": api,
        "human_annotation": human,
        "envelope": envelope,
        "metadata": as_metadata_dict(git_provenance(), phase="p3-cost"),
    }


# ---------------------------------------------------------------------------
# Reliability gates (plan section 3) — real implementations, not-estimable
# on absent human labels.
# ---------------------------------------------------------------------------
def sensitivity_specificity_lower(
    judge_binary: np.ndarray, human_binary: np.ndarray, conf: float = 0.95
) -> dict[str, float | int]:
    """Sensitivity/specificity of the judge vs adjudicated human labels with
    one-sided exact Clopper-Pearson lower ``conf`` bounds (plan section 3:
    lower 95% sensitivity and specificity >= 0.80)."""
    j = np.asarray(judge_binary).astype(bool)
    h = np.asarray(human_binary).astype(bool)
    if j.shape != h.shape or j.ndim != 1:
        raise ValueError("judge/human labels must be matching 1-D arrays")
    n_pos = int(h.sum())
    n_neg = int((~h).sum())
    if n_pos == 0 or n_neg == 0:
        raise ValueError("need >= 1 human positive and >= 1 human negative")
    tp = int((j & h).sum())
    tn = int((~j & ~h).sum())
    return {
        "n_pos": n_pos,
        "n_neg": n_neg,
        "sensitivity": tp / n_pos,
        "specificity": tn / n_neg,
        "sensitivity_lower95": clopper_pearson_lower(tp, n_pos, conf),
        "specificity_lower95": clopper_pearson_lower(tn, n_neg, conf),
    }


def icc_2_1(ratings: np.ndarray) -> float:
    """ICC(2,1) — Shrout-Fleiss two-way random effects, absolute agreement,
    single rater — for the blinded double-human probability audit.

    ``ratings``: (n_subjects, k_raters). ICC(2,1) =
    (MS_R - MS_E) / (MS_R + (k-1) MS_E + k (MS_C - MS_E) / n).
    """
    x = np.asarray(ratings, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] < 2 or x.shape[1] < 2:
        raise ValueError(f"ratings must be (n>=2 subjects, k>=2 raters); got {x.shape}")
    n, k = x.shape
    mean_r = x.mean(axis=1, keepdims=True)
    mean_c = x.mean(axis=0, keepdims=True)
    grand = x.mean()
    ss_r = k * ((mean_r - grand) ** 2).sum()
    ss_c = n * ((mean_c - grand) ** 2).sum()
    ss_e = ((x - mean_r - mean_c + grand) ** 2).sum()
    ms_r = ss_r / (n - 1)
    ms_c = ss_c / (k - 1)
    ms_e = ss_e / ((n - 1) * (k - 1))
    denom = ms_r + (k - 1) * ms_e + k * (ms_c - ms_e) / n
    if denom <= 0:
        raise ValueError("degenerate ratings: non-positive ICC denominator")
    return float((ms_r - ms_e) / denom)


def reliability_gates(out_root: Path) -> dict[str, Any]:
    """Plan section 3 reliability gates per judged trait, from the human audit
    artifact (``human_audit/adjudications.json``; expected schema documented in
    the module docstring companion: rows of {row, item_id, response_index,
    rater_a_prob, rater_b_prob, judge_binary}). NOT-ESTIMABLE on absence —
    with no real adjudications on disk, no reliability gate can pass.

    Thresholds: lower 95% sensitivity AND specificity >= 0.80 (one-sided exact
    Clopper-Pearson); Cohen's kappa >= 0.70 (point estimate,
    sklearn.metrics.cohen_kappa_score on the raters' >=50 binarizations);
    ICC(2,1) >= 0.75 (Shrout-Fleiss two-way random, absolute agreement, single
    rater, on the two probability columns). Sizing floor: >= 100 adjudicated
    positives AND negatives per trait, else that trait is NOT-ESTIMABLE.
    """
    audit_path = Path(out_root) / HUMAN_AUDIT_REL
    if not audit_path.exists():
        return {
            "status": GATE_NOT_ESTIMABLE,
            "missing_artifact": str(audit_path),
            "detail": "no real human adjudications on disk — reliability cannot be estimated",
            "per_trait": {},
        }
    audit = json.loads(audit_path.read_text())
    per_trait: dict[str, Any] = {}
    statuses: list[str] = []
    by_row: dict[str, list[dict[str, Any]]] = {}
    for rec in audit.get("rows", []):
        by_row.setdefault(rec["row"], []).append(rec)
    for row, recs in sorted(by_row.items()):
        a = np.array([float(r["rater_a_prob"]) for r in recs])
        b = np.array([float(r["rater_b_prob"]) for r in recs])
        judge = np.array([r["judge_binary"] for r in recs], dtype=object)
        scored = np.array([jb is not None for jb in judge])
        human_bin = ((a + b) / 2.0) >= 50.0
        n_pos = int(human_bin.sum())
        n_neg = int((~human_bin).sum())
        if n_pos < 100 or n_neg < 100:
            per_trait[row] = {
                "status": GATE_NOT_ESTIMABLE,
                "detail": (
                    f"audit holds {n_pos} positives / {n_neg} negatives; plan section 3 "
                    "requires >= 100 of each per trait"
                ),
            }
            statuses.append(GATE_NOT_ESTIMABLE)
            continue
        if not scored.all():
            per_trait[row] = {
                "status": GATE_NOT_ESTIMABLE,
                "detail": f"{int((~scored).sum())} audit rows lack a judge verdict",
            }
            statuses.append(GATE_NOT_ESTIMABLE)
            continue
        ss = sensitivity_specificity_lower(judge.astype(bool), human_bin)
        kappa = float(cohen_kappa_score(a >= 50.0, b >= 50.0))
        icc = icc_2_1(np.column_stack([a, b]))
        ok = (
            ss["sensitivity_lower95"] >= 0.80
            and ss["specificity_lower95"] >= 0.80
            and kappa >= 0.70
            and icc >= 0.75
        )
        per_trait[row] = {
            "status": GATE_PASS if ok else GATE_FAIL,
            **ss,
            "kappa": kappa,
            "icc_2_1": icc,
            "thresholds": {
                "sens_spec_lower95": 0.80,
                "kappa": 0.70,
                "icc": 0.75,
            },
        }
        statuses.append(per_trait[row]["status"])
    overall = (
        GATE_FAIL
        if GATE_FAIL in statuses
        else (GATE_NOT_ESTIMABLE if GATE_NOT_ESTIMABLE in statuses or not statuses else GATE_PASS)
    )
    return {"status": overall, "artifact": str(audit_path), "per_trait": per_trait}


# ---------------------------------------------------------------------------
# Deliverable E — launch-or-park gate verdict (plan section 8 pilot gates).
# ---------------------------------------------------------------------------
@dataclass
class Gate:
    gate_id: str
    description: str
    status: str
    measured: Any
    threshold: str
    artifact: str
    detail: str = ""

    def __post_init__(self) -> None:
        if self.status not in (GATE_PASS, GATE_FAIL, GATE_NOT_ESTIMABLE):
            raise ValueError(f"invalid gate status {self.status!r}")


def _gate_construct(out_root: Path) -> Gate:
    problems = []
    for row in C.ROW_IDS:
        c = C.CONSTRUCTS[row]
        if not (c.construct and c.positive_anchor and c.negative_anchor and c.exclusions):
            problems.append(f"{row}: incomplete construct table entry")
        if c.judge_scored:
            C.judge_instrument_fingerprint(row)  # raises if no instrument
        elif not c.label_recipe:
            problems.append(f"{row}: objective row with no label recipe")
    return Gate(
        "construct_recognizability",
        "frozen construct table complete for all 11 rows (anchors, exclusions, "
        "judge instrument or objective label recipe); the human-recognizability "
        "half is adjudicated by the human-audit gate",
        GATE_FAIL if problems else GATE_PASS,
        f"{len(C.ROW_IDS) - len(problems)}/{len(C.ROW_IDS)} rows complete",
        "all 11 rows complete",
        "scripts/issue2658_common.py CONSTRUCTS",
        "; ".join(problems),
    )


def _gate_provenance() -> Gate:
    path = F.PROVENANCE_PATH
    if not path.exists():
        return Gate(
            "direction_model_judge_provenance",
            "every frozen external direction hash-verified (or declared not-estimable)",
            GATE_NOT_ESTIMABLE,
            None,
            "11 rows resolved",
            str(path),
            "provenance report absent",
        )
    d = json.loads(path.read_text())
    rows = {e["row"]: e for e in d["rows"]}
    problems = []
    n_eligible = 0
    for row in C.ROW_IDS:
        e = rows.get(row)
        if e is None:
            problems.append(f"{row}: missing from provenance report")
            continue
        if e["c2_c3"] == "eligible":
            n_eligible += 1
            if not (e.get("vector_sha256") and e.get("file_sha256")):
                problems.append(f"{row}: eligible but not hash-pinned")
            if e.get("layer") != C.LAYER:
                problems.append(f"{row}: layer {e.get('layer')} != frozen {C.LAYER}")
        # a DECLARED not-estimable row (no frozen external direction) is a valid
        # plan-section-8 outcome; the Holm family sizes already account for it.
    return Gate(
        "direction_model_judge_provenance",
        "exact direction/model/judge provenance hash-verified per row",
        GATE_FAIL if problems else GATE_PASS,
        f"{n_eligible} eligible + {len(C.ROW_IDS) - n_eligible} declared not-estimable",
        "every row hash-verified or declared not-estimable",
        str(path),
        "; ".join(problems),
    )


def _gate_dependency_graph() -> Gate:
    problems = []
    missing = []
    for path in (F.FRAME_MANIFEST_PATH, F.SPLIT_MANIFEST_PATH):
        if not path.exists():
            missing.append(str(path))
            continue
        body = json.loads(path.read_text())
        try:
            F.validate_manifest(body)
            F.assert_manifest_immutable(body)
        except C.Issue2658GuardError as e:
            problems.append(f"{path.name}: {e}")
    if missing:
        return Gate(
            "clean_dependency_graph",
            "frame/split manifests valid, immutable, extraction-resolved, splits disjoint",
            GATE_NOT_ESTIMABLE,
            None,
            "manifests valid + splits disjoint",
            "; ".join(missing),
            "manifest artifact(s) absent",
        )
    split = json.loads(F.SPLIT_MANIFEST_PATH.read_text())
    for r in split["rows"]:
        # extraction_resolved is None (not False) for rows the manifest declares
        # ineligible for extraction exclusion (no frozen external direction —
        # e.g. harmful_compliance, declared not-estimable in the provenance
        # partition); only ELIGIBLE rows must have resolved their corpus.
        if r.get("eligible_for_extraction_exclusion") and not r.get("extraction_resolved"):
            problems.append(f"{r['row']}: extraction corpus unresolved")
        bad = {v for v in r["superfamily_splits"].values()} - {"dev", "test"}
        if bad:
            problems.append(f"{r['row']}: foreign split values {sorted(bad)}")
        dev = {k for k, v in r["superfamily_splits"].items() if v == "dev"}
        test = {k for k, v in r["superfamily_splits"].items() if v == "test"}
        try:
            C.assert_split_lineage_disjoint(dev, test)
        except C.Issue2658GuardError as e:
            problems.append(f"{r['row']}: {e}")
    return Gate(
        "clean_dependency_graph",
        "frame/split manifests valid, immutable, extraction-resolved, splits disjoint",
        GATE_FAIL if problems else GATE_PASS,
        f"{len(split['rows'])} rows checked",
        "manifests valid + splits disjoint",
        f"{F.FRAME_MANIFEST_PATH}; {F.SPLIT_MANIFEST_PATH}",
        "; ".join(problems),
    )


def _gate_row_vector_alignment(out_root: Path, gen_root: Path, split: str) -> Gate:
    if not F.PROVENANCE_PATH.exists():
        return Gate(
            "row_vector_alignment",
            "provenance rows pin vector shape/layer; pilot capture store aligns 1:1 "
            "with generated answers",
            GATE_NOT_ESTIMABLE,
            None,
            "shape (3584,) at layer 19 + complete store",
            str(F.PROVENANCE_PATH),
            "provenance report absent",
        )
    d = json.loads(F.PROVENANCE_PATH.read_text())
    problems = []
    for e in d["rows"]:
        if e["c2_c3"] != "eligible":
            continue
        shape = e.get("shape")
        flat = shape if isinstance(shape, list) else [shape]
        if C.HIDDEN not in [int(v) for v in flat if v is not None]:
            problems.append(f"{e['row']}: vector shape {shape} lacks hidden dim {C.HIDDEN}")
    store_root = Path(out_root) / "l19_store" / split
    gen_dir = Path(gen_root) / "raw_completions" / split
    if not store_root.exists():
        return Gate(
            "row_vector_alignment",
            "provenance rows pin vector shape/layer; pilot capture store aligns 1:1 "
            "with generated answers",
            GATE_NOT_ESTIMABLE,
            f"provenance shape checks: {len(problems)} problems",
            "shape (3584,) at layer 19 + complete store",
            str(store_root),
            ("; ".join(problems) + "; " if problems else "")
            + "pilot L19 capture store absent — alignment unmeasurable pre-capture",
        )
    expected: list[tuple[str, int]] = []
    for p in sorted(gen_dir.glob("*.json")):
        body = json.loads(p.read_text())
        expected.extend((rec["prompt_id"], int(rec["response_index"])) for rec in body["records"])
    realized: set[tuple[str, int]] = set()
    for shard_dir in sorted(store_root.glob("shard*")):
        for idx_path in sorted(shard_dir.glob("row_index_shard*.jsonl")):
            with idx_path.open() as fh:
                for line in fh:
                    r = json.loads(line)
                    realized.add((r["prompt_id"], int(r["response_index"])))
    missing = set(expected) - realized
    foreign = realized - set(expected)
    if missing or foreign:
        problems.append(f"{len(missing)} generated answers uncaptured; {len(foreign)} foreign")
    return Gate(
        "row_vector_alignment",
        "provenance rows pin vector shape/layer; pilot capture store aligns 1:1 "
        "with generated answers",
        GATE_FAIL if problems else GATE_PASS,
        f"{len(realized)} captured / {len(expected)} generated",
        "shape (3584,) at layer 19 + complete store",
        f"{F.PROVENANCE_PATH}; {store_root}",
        "; ".join(problems),
    )


def _gate_discordance(discordance: dict[str, Any] | None, out_root: Path) -> Gate:
    if discordance is None:
        return Gate(
            "measured_discordance_by_cell",
            "every cell's pilot discordance measured with a positive sizing lower bound",
            GATE_NOT_ESTIMABLE,
            None,
            "132 cells measured, lower bound > 0",
            str(Path(out_root) / "power" / "discordance.json"),
            "pilot labels absent — discordance unmeasured",
        )
    n_measured = 0
    problems = []
    missing = []
    for row, rrec in sorted(discordance["rows"].items()):
        missing.extend(rrec["missing_cells"])
        for cell, c in sorted(rrec["cells"].items()):
            if c.get("status") != "measured":
                problems.append(f"{cell}: {c.get('detail', 'unmeasured')}")
                continue
            n_measured += 1
            if c["x_discordant_10draw"] == 0 or c["sizing_lower_bound"] <= 0:
                problems.append(
                    f"{cell}: zero measured discordance (x={c['x_discordant_10draw']}) — "
                    "production requirement unbounded"
                )
    n_expected = len(C.ROW_IDS) * C.PILOT.cells_per_row
    if missing:
        return Gate(
            "measured_discordance_by_cell",
            "every cell's pilot discordance measured with a positive sizing lower bound",
            GATE_NOT_ESTIMABLE,
            f"{n_measured}/{n_expected} cells measured",
            f"{n_expected} cells measured, lower bound > 0",
            str(Path(out_root) / "power" / "discordance.json"),
            f"{len(missing)} cells lack pilot label artifacts (e.g. {missing[:3]})",
        )
    return Gate(
        "measured_discordance_by_cell",
        "every cell's pilot discordance measured with a positive sizing lower bound",
        GATE_FAIL if problems or n_measured < n_expected else GATE_PASS,
        f"{n_measured}/{n_expected} cells measured",
        f"{n_expected} cells measured, lower bound > 0",
        str(Path(out_root) / "power" / "discordance.json"),
        "; ".join(
            problems[:10]
            or (
                [f"only {n_measured}/{n_expected} cells measured"]
                if n_measured < n_expected
                else []
            )
        ),
    )


def _gate_judge_quality(out_root: Path, split: str) -> Gate:
    judged_rows = [r for r in C.ROW_IDS if C.CONSTRUCTS[r].judge_scored]
    n_expected = len(judged_rows) * C.PILOT.cells_per_row
    found = 0
    problems = []
    worst_rate = 0.0
    for row in judged_rows:
        for cell in expected_cells(row):
            p = Path(out_root) / "judge" / split / row / f"{cell}.json"
            if not p.exists():
                continue
            body = json.loads(p.read_text())
            found += 1
            rate = body.get("parse_fail_rate")
            if rate is not None:
                worst_rate = max(worst_rate, float(rate))
            pg = body["plan_gate"]
            if not pg["parse_fail_lt_threshold"]:
                problems.append(f"{cell}: parse-fail rate {rate} >= {J.PARSE_FAIL_THRESHOLD}")
            if not pg["zero_max_tokens_stops"]:
                problems.append(f"{cell}: max_tokens judge stops present")
    if found < n_expected:
        return Gate(
            "judge_parse_failure_and_max_tokens",
            "pilot judge parse failure < 2% and zero max-token judge stops, per cell",
            GATE_NOT_ESTIMABLE,
            f"{found}/{n_expected} judged cells on disk",
            f"parse-fail < {J.PARSE_FAIL_THRESHOLD} and zero truncation stops",
            str(Path(out_root) / "judge" / split),
            "judge pilot artifacts incomplete",
        )
    return Gate(
        "judge_parse_failure_and_max_tokens",
        "pilot judge parse failure < 2% and zero max-token judge stops, per cell",
        GATE_FAIL if problems else GATE_PASS,
        f"worst per-cell parse-fail rate {worst_rate:.4f} over {found} cells",
        f"parse-fail < {J.PARSE_FAIL_THRESHOLD} and zero truncation stops",
        str(Path(out_root) / "judge" / split),
        "; ".join(problems[:10]),
    )


def _gate_cap_hit(gen_root: Path, split: str) -> Gate:
    gen_dir = Path(gen_root) / "raw_completions" / split
    paths = sorted(gen_dir.glob("*.json"))
    n_expected = len(C.ROW_IDS) * C.PILOT.cells_per_row
    if len(paths) < n_expected:
        return Gate(
            "measured_cap_hit_rate",
            "realized length-cap-hit fraction <= 2% per cell (plan section 5)",
            GATE_NOT_ESTIMABLE,
            f"{len(paths)}/{n_expected} generated cells on disk",
            f"<= {G.CAP_HIT_AMEND_THRESHOLD} per cell",
            str(gen_dir),
            "pilot generation artifacts incomplete",
        )
    problems = []
    worst = 0.0
    for p in paths:
        body = json.loads(p.read_text())
        rep = body["cap_hit"]
        for cell_key, frac in rep["per_cell_fraction"].items():
            worst = max(worst, float(frac))
        if rep["amendment_required"]:
            problems.append(f"{p.name}: cells over threshold {rep['cells_over_threshold']}")
    return Gate(
        "measured_cap_hit_rate",
        "realized length-cap-hit fraction <= 2% per cell (plan section 5)",
        GATE_FAIL if problems else GATE_PASS,
        f"worst per-cell fraction {worst:.4f}",
        f"<= {G.CAP_HIT_AMEND_THRESHOLD} per cell",
        str(gen_dir),
        "; ".join(problems[:10]),
    )


def _gate_human_audit(out_root: Path) -> tuple[Gate, dict[str, Any]]:
    rel = reliability_gates(out_root)
    status = rel["status"]
    detail = rel.get("detail", "")
    if status == GATE_NOT_ESTIMABLE and not detail:
        detail = "reliability not established for every judged trait"
    return (
        Gate(
            "human_audit_feasibility",
            "blinded double-human audit adjudicated at plan section 3 sizing with "
            "reliability gates passed (sens/spec lower95 >= 0.80, kappa >= 0.70, "
            "ICC >= 0.75); with no real adjudications on disk this is NOT-ESTIMABLE "
            "and the verdict PARKs — the honest pre-audit output, not a bug",
            status,
            {k: v.get("status") for k, v in rel.get("per_trait", {}).items()} or None,
            "every judged trait PASS",
            rel.get("artifact", rel.get("missing_artifact", "")),
            detail,
        ),
        rel,
    )


def _gate_power(selection: dict[str, Any] | None, out_root: Path) -> Gate:
    art = str(Path(out_root) / "power" / "production_n.json")
    if selection is None:
        return Gate(
            "power_based_fixed_n",
            "one common prompts-per-cell N fixed by the registered clustered simulation",
            GATE_NOT_ESTIMABLE,
            None,
            f"power >= {REGISTERED.power_target} at AUROC "
            f"{REGISTERED.primary_effect_auroc}, alpha {REGISTERED.alpha_worst_case:.5f}",
            art,
            "power selection not run (pilot labels absent)",
        )
    if not selection.get("registered_match", False):
        return Gate(
            "power_based_fixed_n",
            "one common prompts-per-cell N fixed by the registered clustered simulation",
            GATE_FAIL,
            selection.get("n_common"),
            "simulation run at the REGISTERED n_replicates/n_permutations",
            art,
            "selection was computed at non-registered simulation sizes (smoke/override) — "
            "re-run at the registered constants before launch",
        )
    # Row-coverage check: the plan sizes ONE common N as the max over ALL 11
    # registered rows, so a selection simulated on a --rows SUBSET (the sharded
    # P2-P3 dispatch shape) must never authorize a launch — its n_common is the
    # max over the shard only. Sharding stays supported: shard runs warm the
    # shared ledger; only the FINAL full-row invocation's selection can PASS.
    simulated = set(selection.get("rows_simulated") or [])
    registered_rows = set(C.ROW_IDS)
    if simulated != registered_rows:
        missing_rows = sorted(registered_rows - simulated)
        foreign_rows = sorted(simulated - registered_rows)
        return Gate(
            "power_based_fixed_n",
            "one common prompts-per-cell N fixed by the registered clustered simulation",
            GATE_FAIL,
            {"rows_simulated": sorted(simulated), "n_common": selection.get("n_common")},
            f"rows_simulated == all {len(C.ROW_IDS)} registered rows",
            art,
            f"selection simulated a row SUBSET — missing rows: {missing_rows}"
            + (f"; foreign rows: {foreign_rows}" if foreign_rows else "")
            + " — a sharded --rows run sizes N on its shard only; re-run the final "
            "full-row selection before launch",
        )
    if selection["status"] != "measured" or selection["n_common"] is None:
        return Gate(
            "power_based_fixed_n",
            "one common prompts-per-cell N fixed by the registered clustered simulation",
            GATE_FAIL,
            selection.get("n_common"),
            f"power >= {REGISTERED.power_target} and >= 20 discordant/cell bounded",
            art,
            f"unbounded cells: {selection.get('cells_with_unbounded_requirement')}; "
            f"rows missing labels: {selection.get('rows_missing_labels')}; per-row power N: "
            + "; ".join(
                f"{r}={v['n_power'] if v['n_power'] is not None else v['status']}"
                for r, v in sorted(selection.get("per_row_power_n", {}).items())
            ),
        )
    return Gate(
        "power_based_fixed_n",
        "one common prompts-per-cell N fixed by the registered clustered simulation",
        GATE_PASS,
        {
            "n_common": selection["n_common"],
            "binding_discordance_cell": selection["binding_discordance_cell"],
            "binding_power_row": selection["binding_power_row"],
        },
        f"power >= {REGISTERED.power_target} at AUROC {REGISTERED.primary_effect_auroc}",
        art,
        "",
    )


def _gate_profile_freshness(
    profiles: dict[str, RowLabelProfile],
    discordance: dict[str, Any] | None,
    selection: dict[str, Any] | None,
    out_root: Path,
) -> Gate:
    """The discordance/production-N artifacts must be content-addressed to the
    SAME pilot-label pools live on disk at GATE time — and to each other.

    A standalone ``--phase gate`` re-run is the NORMAL post-adjudication
    workflow (judge cells are regenerated when human adjudications land, which
    shrinks/updates the (k, n) pools), so the human-audit / judge-quality gates
    read POST-adjudication artifacts while a stale discordance/production-N
    pair certifies PRE-adjudication pools. The fingerprint comparison is the
    staleness check: mismatch => FAIL naming the shas, never a silent LAUNCH
    over mutually inconsistent inputs.
    """
    art = (
        f"{Path(out_root) / 'power' / 'discordance.json'}; "
        f"{Path(out_root) / 'power' / 'production_n.json'}"
    )
    desc = (
        "discordance + production-N artifacts fingerprint the SAME pilot-label "
        "pools currently on disk (profile_sha256 == live fingerprint, pairwise)"
    )
    threshold = "every artifact profile_sha256 == live pilot-label fingerprint"
    live = profile_fingerprint(profiles)
    measured: dict[str, Any] = {"live_profiles": live}
    if discordance is None and selection is None:
        return Gate(
            "profile_freshness",
            desc,
            GATE_NOT_ESTIMABLE,
            measured,
            threshold,
            art,
            "no discordance/production_n artifacts to cross-check (nothing certified)",
        )
    problems: list[str] = []
    for name, body in (("discordance", discordance), ("production_n", selection)):
        if body is None:
            continue
        sha = body.get("profile_sha256")
        measured[name] = sha
        if not sha:
            problems.append(f"{name} artifact carries no profile_sha256 — regenerate it")
        elif sha != live:
            problems.append(
                f"{name} profile_sha256 {sha} != live pilot-label fingerprint {live} — "
                "stale artifact (label pools changed since it was computed)"
            )
    if discordance is not None and selection is not None:
        d_sha, s_sha = discordance.get("profile_sha256"), selection.get("profile_sha256")
        if d_sha and s_sha and d_sha != s_sha:
            problems.append(
                f"discordance profile_sha256 {d_sha} != production_n profile_sha256 "
                f"{s_sha} — mixed-generation artifacts"
            )
    return Gate(
        "profile_freshness",
        desc,
        GATE_FAIL if problems else GATE_PASS,
        measured,
        threshold,
        art,
        "; ".join(problems),
    )


def _gate_cost(cost: dict[str, Any]) -> Gate:
    env = cost["envelope"]
    art = "power/cost_report.json"
    if env["within_envelope"] is None:
        return Gate(
            "cost_within_envelope",
            "measured pilot GPU-h <= 8 and projected total <= 80 GPU-h (plan section 10)",
            GATE_NOT_ESTIMABLE,
            None,
            "pilot <= 8 GPU-h; total <= 80 GPU-h",
            art,
            env["basis"],
        )
    return Gate(
        "cost_within_envelope",
        "measured pilot GPU-h <= 8 and projected total <= 80 GPU-h (plan section 10)",
        GATE_PASS if env["within_envelope"] else GATE_FAIL,
        {
            "measured_pilot_gpu_h": cost["gpu_hours"]["measured_pilot_gpu_h"]["value"],
            "projected_total_gpu_h": env["projected_total_gpu_h"],
            "margin_gpu_h": env["margin_gpu_h"],
        },
        "pilot <= 8 GPU-h; total <= 80 GPU-h",
        art,
        "",
    )


def evaluate_gates(
    out_root: Path,
    gen_root: Path,
    split: str,
    profiles: dict[str, RowLabelProfile],
    discordance: dict[str, Any] | None,
    selection: dict[str, Any] | None,
    cost: dict[str, Any],
) -> dict[str, Any]:
    """Deliverable E: the plan section 8 pilot-gate list, evaluated MECHANICALLY.

    Verdict is LAUNCH only if EVERY gate is PASS. A single FAIL or
    NOT-ESTIMABLE yields PARK, naming the blocking gates. NOT-ESTIMABLE never
    collapses to PASS — that inversion would let the pilot authorize
    sealed-test spend it never measured. ``profiles`` is the LIVE pilot-label
    profile at gate time; the profile-freshness gate cross-checks the
    discordance/production-N artifacts' profile_sha256 against it (and each
    other), so a standalone ``--phase gate`` over regenerated labels can never
    certify stale simulation artifacts.
    """
    human_gate, rel = _gate_human_audit(out_root)
    gates = [
        _gate_construct(out_root),
        _gate_provenance(),
        _gate_dependency_graph(),
        _gate_row_vector_alignment(out_root, gen_root, split),
        _gate_discordance(discordance, out_root),
        _gate_judge_quality(out_root, split),
        _gate_cap_hit(gen_root, split),
        human_gate,
        _gate_power(selection, out_root),
        _gate_profile_freshness(profiles, discordance, selection, out_root),
        _gate_cost(cost),
    ]
    blockers = [g.gate_id for g in gates if g.status != GATE_PASS]
    verdict = "LAUNCH" if not blockers else "PARK"
    return {
        "schema": VERDICT_SCHEMA,
        "verdict": verdict,
        "blockers": blockers,
        "gates": [asdict(g) for g in gates],
        "reliability": rel,
        "metadata": as_metadata_dict(git_provenance(), phase="p3-gate"),
    }


# ---------------------------------------------------------------------------
# Synthetic smoke profile (SMOKE ONLY — labeled; never a production input).
# ---------------------------------------------------------------------------
def synthetic_profile(
    rows: list[str], seed: int = 0, prompts_per_cell: int = 5, draws: int = 10
) -> dict[str, RowLabelProfile]:
    """Deterministic synthetic (k, n) pools shaped like the pilot registry —
    used ONLY under --smoke / tests (the real path reads unit-4/6 artifacts)."""
    rng = np.random.default_rng(_unit_seed("i2658-synthetic-profile", seed))
    out: dict[str, RowLabelProfile] = {}
    for row in rows:
        prof = RowLabelProfile(row=row, judged=C.CONSTRUCTS[row].judge_scored)
        prof.artifact_dir = "SYNTHETIC (smoke)"
        for cell in expected_cells(row):
            base = rng.beta(2.0, 3.0)
            ks = rng.binomial(draws, np.clip(base + rng.normal(0, 0.15, prompts_per_cell), 0, 1))
            prof.cells[cell] = [(int(k), draws) for k in ks]
        out[row] = prof
    return out


def load_profile_json(path: Path) -> dict[str, RowLabelProfile]:
    """Load an explicit {row: {cell: [[k, n], ...]}} profile fixture."""
    body = json.loads(Path(path).read_text())
    out: dict[str, RowLabelProfile] = {}
    for row, cells in body.items():
        if row not in C.ROW_IDS:
            raise PowerInputError(f"profile row {row!r} not in ROW_IDS")
        prof = RowLabelProfile(row=row, judged=C.CONSTRUCTS[row].judge_scored)
        prof.artifact_dir = str(path)
        for cell, pairs in cells.items():
            prof.cells[cell] = [(int(k), int(n)) for k, n in pairs]
        out[row] = prof
    return out


# ---------------------------------------------------------------------------
# Driver.
# ---------------------------------------------------------------------------
def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument(
        "--phase",
        choices=("discordance", "power", "cost", "gate", "all"),
        default="all",
        help="which deliverable phase(s) to run",
    )
    ap.add_argument(
        "--out-root", type=Path, default=None, help="artifact root (judge/labels/power)"
    )
    ap.add_argument("--gen-root", type=Path, default=None, help="generation artifact root")
    ap.add_argument("--split", default="pilot", choices=C.SPLITS)
    ap.add_argument("--rows", nargs="*", default=None, help="row subset (default: all 11)")
    ap.add_argument("--seed", type=int, default=0, help="resample/credible-bound seed")
    ap.add_argument(
        "--profile-json",
        type=Path,
        default=None,
        help="explicit {row: {cell: [[k, n], ...]}} discordance profile fixture",
    )
    ap.add_argument(
        "--reps", type=int, default=None, help="OVERRIDE n_replicates (non-registered => gate FAIL)"
    )
    ap.add_argument(
        "--n-perm",
        type=int,
        default=None,
        help="OVERRIDE n_permutations (non-registered => gate FAIL)",
    )
    ap.add_argument("--smoke", action="store_true", help="tiny synthetic end-to-end run")
    ap.add_argument(
        "--import-check",
        action="store_true",
        help="verify argparse attribute completeness + module imports, then exit 0",
    )
    return ap


def run(args: argparse.Namespace) -> int:
    if args.smoke:
        out_root = args.out_root or Path("/tmp/issue-2658-smoke-power")
        gen_root = args.gen_root or out_root
        rows = args.rows or ["evil", "correctness_math"]
        # 219 permutations keep the Holm-worst-case alpha REACHABLE in smoke
        # (min p = 1/220 = 0.05/11 exactly); 99 would make the test
        # unrejectable and the rejection branch never executed.
        n_reps, n_perm = args.reps or 24, args.n_perm or 219
        reg = PowerRegistry(
            n_replicates=n_reps,
            n_permutations=n_perm,
            power_curve_effects=(0.60, 0.70),
            bisection_cap=120,
        )
        print(
            "[power] SMOKE: synthetic profile + downsized simulation "
            f"(reps={n_reps} perms={n_perm}); outputs -> {out_root} (scratch)",
            flush=True,
        )
    else:
        out_root = args.out_root or F.OUT_DIR
        gen_root = args.gen_root or F.OUT_DIR
        rows = args.rows or list(C.ROW_IDS)
        n_reps, n_perm = args.reps, args.n_perm
        reg = REGISTERED
    for row in rows:
        if row not in C.ROW_IDS:
            raise PowerInputError(f"unknown row {row!r}")
    power_dir = Path(out_root) / "power"
    power_dir.mkdir(parents=True, exist_ok=True)

    # --- profile ---------------------------------------------------------
    if args.profile_json is not None:
        profiles = load_profile_json(args.profile_json)
        profiles = {r: p for r, p in profiles.items() if r in rows}
    elif args.smoke:
        profiles = synthetic_profile(rows, seed=args.seed)
    else:
        profiles = load_pilot_label_profile(Path(out_root), args.split, rows)
    have_labels = any(p.cells for p in profiles.values())

    discordance = None
    selection = None
    if args.phase in ("discordance", "power", "all"):
        if not have_labels:
            if args.phase in ("discordance", "power"):
                raise PowerInputError(
                    "no pilot label artifacts under "
                    f"{Path(out_root) / 'judge' / args.split} / "
                    f"{Path(out_root) / 'objective_labels' / args.split} and no "
                    "--profile-json — the discordance/power phases need pilot labels "
                    "(run --phase gate for the pre-pilot PARK verdict)"
                )
            print("[power] pilot labels absent — discordance/power recorded not-estimable")
        else:
            discordance = measure_discordance(profiles, seed=args.seed)
            write_json_atomic(power_dir / "discordance.json", discordance)
            print(f"[power] wrote {power_dir / 'discordance.json'}", flush=True)
    if args.phase in ("power", "all") and have_labels:
        any_complete = any(p.cells and not p.missing_cells for p in profiles.values())
        if any_complete:
            ledger = PowerLedger(power_dir / "power_units.jsonl")
            selection = select_production_n(
                profiles,
                discordance,
                ledger,
                reg=reg,
                n_reps=n_reps,
                n_perm=n_perm,
                seed=args.seed,
            )
            write_json_atomic(power_dir / "production_n.json", selection)
            print(
                f"[power] wrote {power_dir / 'production_n.json'} "
                f"(n_common={selection['n_common']}, status={selection['status']})",
                flush=True,
            )
        else:
            print("[power] no row has a complete label profile — selection skipped")

    cost = None
    if args.phase in ("cost", "gate", "all"):
        sel_path = power_dir / "production_n.json"
        if selection is None and sel_path.exists():
            selection = json.loads(sel_path.read_text())
        n_common = selection["n_common"] if selection else None
        cost = cost_report(Path(out_root), Path(gen_root), args.split, n_common)
        write_json_atomic(power_dir / "cost_report.json", cost)
        print(f"[power] wrote {power_dir / 'cost_report.json'}", flush=True)

    if args.phase in ("gate", "all"):
        disc_path = power_dir / "discordance.json"
        if discordance is None and disc_path.exists():
            discordance = json.loads(disc_path.read_text())
        verdict = evaluate_gates(
            Path(out_root), Path(gen_root), args.split, profiles, discordance, selection, cost
        )
        write_json_atomic(power_dir / "gate_verdict.json", verdict)
        print(
            f"[power] VERDICT: {verdict['verdict']}"
            + (f" — blockers: {', '.join(verdict['blockers'])}" if verdict["blockers"] else ""),
            flush=True,
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = build_argparser()
    args = ap.parse_args(argv)
    if args.import_check:
        from explore_persona_space.orchestrate.argcheck import assert_args_attributes_defined

        assert_args_attributes_defined(__file__)
        print("[power] import-check OK (argparse attributes complete; imports resolved)")
        return 0
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
