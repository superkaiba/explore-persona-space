"""Issue #657 — behavior-direction alignment predictor (cross-behavior leakage bake-off).

Training-free, reuse-only analysis. For each behavior (sycophancy, refusal,
marker, EM) it asks whether a persona vector's COSINE alignment to that
behavior's linear direction predicts:

  - DV-(a) the persona's BASE rate for the behavior (the #623 generalization, H1), and
  - DV-(b) where an implanted behavior LEAKS across held-out personas, and whether
    that alignment beats the base behavioral prior (`bystander_base_rate`) — the
    #518/#545/#605 bake-off that no base-model geometry has cleared (H2/H3).

Substrate (all reused, never regenerated):
  - persona vectors: #623 `issue623_persona_vectors/persona_vectors/method_{a,b}/*.pt`
    (per-persona ``(n_layers, hidden)`` centroids; persona vector =
    centroid_i - centroid_assistant) + the <=7 missing panel personas extracted
    fresh into the issue-657 path (Phase 1).
  - behavior directions: sycophancy REUSED from #623
    (`sycophancy_trait/{last_token,response_avg}_{7,14,21,27}.pt`); refusal /
    marker / EM extracted via the SAME #623 Persona-Vectors recipe (Phase 1).
  - leakage DV cells: #518 `eval_results/issue_518/<arm>/_inputs/predictor_comparison.json`
    (syco/refusal/em) + #605 marker per-cell reads.
  - base rates: #623 `syc_i.json` (syco) + `bystander_base_rate` (all behaviors,
    from the #518 cells).

Key statistical machinery (plan §5/§6):
  - ``compute_alignment``        — bystander-level cosine(persona vector, behavior dir).
  - ``prior_centroid_projection`` — M-Alts1 geometry-side prior control:
    cosine(persona vector, base-rate-weighted persona-vector centroid).
  - ``partial_spearman``         — Spearman partial-rho via rank-residualization;
    z is a scalar OR a matrix (the doubly-partialled read).
  - ``lopo_heldout``             — leave-one-persona-out held-out scoring + grouped-CV DeltaR2.
  - ``bootstrap_over_personas``  — resamples the held-out BYSTANDER personas (not cells).
  - ``shuffled_direction_null``  — random-rotation null of the behavior direction.
  - ``attenuation_sensitivity``  — M-Stats1 bounded-attenuation reliability band on the
    noisy partialled prior (R_low / R_expected / R_high).

The resampling unit for DV-(b) is the held-out BYSTANDER persona, NOT the cell —
cells sharing a bystander are not independent (the predictor is bystander-level).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

# Layer order in the stored persona-vector / behavior-direction tensors
# (matches #623 DEFAULT_LAYERS; the per-persona centroid tensor is (n_layers, hidden)).
DEFAULT_LAYERS: tuple[int, ...] = (7, 14, 21, 27)
HEADLINE_LAYER: int = 14  # #623 steering-selected headline layer
BASELINE_PERSONA: str = "assistant"  # persona vector = centroid_i - centroid_assistant

# The H3 PRIMARY behavior set (M-Alts2): EM is EXCLUDED from the primary gate
# (Sonnet coherence-filter survivor-rate proxy on the EM leakage DV, §12). A literal
# frozenset (NOT a tuple) for immutable literal-set semantics; serialize to a sorted
# list where stable JSON ordering matters (`sorted(PRIMARY_H3_BEHAVIORS)`).
PRIMARY_H3_BEHAVIORS: frozenset[str] = frozenset({"sycophancy", "refusal", "marker"})

# Canonical marker token (`.claude/rules/marker-leakage-measurement.md`): ` ※`
# (leading space, Qwen-2.5-7B id 83399). The reused #605 marker cells MUST carry
# this id — a wrong-token cell (bare ※ id 63680, or [ZLT]) would make the marker
# DV measure the wrong slot, so the marker substrate loader asserts it fail-fast.
MARKER_TOKEN_ID: int = 83399
MARKER_TEXT: str = " ※"  # " ※"

# Bracketing per-bystander generation counts for the reliability band (M-Stats1).
# The #518 substrate stores `bystander_base_rate` as a point scalar with NO
# denominator, so the band brackets a plausible n range (plan section 6: "typical
# n approx 20-50 generations per persona per behavior"). R_low uses the LARGEST n
# (prior cleanest, most attenuation), R_high uses the SMALLEST n (prior
# noisiest, least correction).
RELIABILITY_BRACKET_N: dict[str, int] = {"n_low": 20, "n_expected": 35, "n_high": 50}

BOOTSTRAP_B: int = 10_000
BOOTSTRAP_SEED: int = 657
SHUFFLE_B: int = 1_000


# ──────────────────────────────────────────────────────────────────────────────
# Cell substrate loading
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class CellTable:
    """Normalized per-(source, bystander) leakage cells for one behavior.

    Each entry in ``cells`` carries: source, bystander, delta (leakage DV),
    bystander_base_rate (the H3 baseline), source_base_rate, completion_logprob
    (the secondary continuous DV, where present), source_resp_len_mean.
    """

    behavior: str
    cells: list[dict]
    sources: list[str]
    bystanders: list[str]
    # Per-bystander base rate (the H3 baseline prior); the same scalar across all
    # cells sharing a bystander.
    bystander_base_rate: dict[str, float]
    provenance: dict = field(default_factory=dict)

    def n_cells(self) -> int:
        return len(self.cells)


def load_cell_substrate(behavior: str, eval_results_root: Path) -> CellTable:
    """Load the reused per-cell leakage substrate for one behavior, normalized.

    syco / refusal / em : #518 ``<arm>/_inputs/predictor_comparison.json``
                          (138 off-diag cells; `delta`, `bystander_base_rate`, ...).
    marker              : #605 marker per-cell trained vs base reads, joined to the
                          persona axis via ``load_marker_cell_substrate`` (separate
                          path — the #605 panel is context-keyed, not persona-keyed).

    Raises on a missing file or a missing required field (fail-fast — the crash
    IS the signal; a join bug must not silently produce an empty/zeroed table).
    """
    if behavior == "marker":
        return load_marker_cell_substrate(eval_results_root)

    arm = {"sycophancy": "syco", "refusal": "refusal", "em": "em"}.get(behavior)
    if arm is None:
        raise ValueError(f"Unknown behavior {behavior!r}; expected sycophancy/refusal/em/marker.")

    path = eval_results_root / f"issue_518/{arm}/_inputs/predictor_comparison.json"
    if not path.exists():
        raise FileNotFoundError(
            f"#518 cell substrate missing for behavior {behavior!r}: {path}. "
            "The leakage bake-off cannot run without the reused #518 cells."
        )
    doc = json.loads(path.read_text())
    raw_cells = doc["cells"]

    required = ("source", "bystander", "delta", "bystander_base_rate")
    cells: list[dict] = []
    bystander_base_rate: dict[str, float] = {}
    for c in raw_cells:
        for k in required:
            if k not in c or c[k] is None:
                raise ValueError(
                    f"#518 {behavior} cell missing required field {k!r}: "
                    f"source={c.get('source')} bystander={c.get('bystander')}"
                )
        cell = {
            "source": c["source"],
            "bystander": c["bystander"],
            "delta": float(c["delta"]),
            "bystander_base_rate": float(c["bystander_base_rate"]),
            "source_base_rate": (
                float(c["source_base_rate"]) if c.get("source_base_rate") is not None else None
            ),
            "completion_logprob": (
                float(c["completion_logprob"]) if c.get("completion_logprob") is not None else None
            ),
            "source_resp_len_mean": (
                float(c["source_resp_len_mean"])
                if c.get("source_resp_len_mean") is not None
                else None
            ),
        }
        cells.append(cell)
        bystander_base_rate[cell["bystander"]] = cell["bystander_base_rate"]

    sources = sorted({c["source"] for c in cells})
    bystanders = sorted({c["bystander"] for c in cells})
    return CellTable(
        behavior=behavior,
        cells=cells,
        sources=sources,
        bystanders=bystanders,
        bystander_base_rate=bystander_base_rate,
        provenance={
            "source_path": str(path),
            "n_cells": len(cells),
            "git_sha": doc.get("git_sha"),
            "schema_version": doc.get("schema_version"),
        },
    )


# Map #605 marker context-label persona tokens -> 24-persona panel names.
# The #605 panel is context-keyed (m605_nt_<persona>_<n>__<affordance>); only the
# personas that overlap the 24-persona panel can join the leakage bake-off.
_MARKER_PERSONA_ALIASES: dict[str, str] = {
    "assistant": "assistant",
    "comedian": "comedian",
    "villain": "villain",
    "swe": "software_engineer",
    # formal / socratic / pirate have NO 24-persona-panel counterpart -> dropped.
}


def _marker_persona_from_context(context_label: str) -> str | None:
    """Extract the 24-persona-panel name from a #605 marker context label.

    e.g. ``m605_nt_comedian_2__explicit`` -> ``comedian``;
         ``m605_nt_swe_1__none`` -> ``software_engineer``;
         ``instr_explicit_1`` (instruction context, no persona) -> None.
    Returns None when the context does not map onto a 24-persona-panel persona.
    """
    if not context_label.startswith("m605_nt_"):
        return None  # instruction-only contexts carry no persona
    rest = context_label[len("m605_nt_") :]  # e.g. "comedian_2__explicit"
    token = rest.split("_")[0]  # "comedian"
    return _MARKER_PERSONA_ALIASES.get(token)


def load_marker_cell_substrate(eval_results_root: Path) -> CellTable:
    """Load the #605 marker leakage cells joined to the 24-persona axis.

    The marker leakage DV is the trained-base on-policy ``log P(marker)`` at the
    end of the model's own response (the #605 four-float per-cell read; the
    behavioral-primary log-prob space per `.claude/rules/marker-leakage-measurement.md`).
    The #605 panel is context-keyed, so only contexts whose persona token overlaps
    the 24-persona panel are kept (plan §8/§12 graceful degradation: if overlap is
    too small, the caller reports marker as DV-(a)-only).

    ``bystander_base_rate`` for marker = the BASE ``mean_logp_marker`` (per_cell_base),
    i.e. the persona's base-model marker log-prob prior — the marker analogue of
    the judge-rate prior the other behaviors use.
    """
    marker_root = eval_results_root / "issue_605/marker"
    trained_dir = marker_root / "per_cell_trained"
    base_dir = marker_root / "per_cell_base"
    if not trained_dir.is_dir() or not base_dir.is_dir():
        raise FileNotFoundError(
            f"#605 marker per-cell dirs missing under {marker_root} "
            "(per_cell_trained / per_cell_base). Marker leakage bake-off cannot run."
        )

    # Aggregate per (source_cid, bystander persona): average the trained-base
    # log-prob shift over all overlapping contexts for that persona.
    # delta[persona] = mean_logp_marker(trained) - mean_logp_marker(base).
    by_pair: dict[tuple[str, str], dict[str, list[float]]] = {}
    base_logp_by_persona: dict[str, list[float]] = {}

    n_marker_id_checked = 0
    for tf in sorted(trained_dir.glob("*.json")):
        bf = base_dir / tf.name
        if not bf.exists():
            continue  # no matching base read; skip (reported in provenance count)
        td = json.loads(tf.read_text())
        bd = json.loads(bf.read_text())
        # Fail-fast on a wrong-marker-token cell (the in-process-assert mandate of
        # `.claude/rules/marker-leakage-measurement.md`, adapted to the reuse
        # context): the reused #605 marker DV is `log P( ※ )` at id 83399; a cell
        # carrying a different marker_id would silently measure the wrong slot.
        cell_marker_id = td.get("metadata", {}).get("marker_id")
        if cell_marker_id is not None:
            if int(cell_marker_id) != MARKER_TOKEN_ID:
                raise ValueError(
                    f"#605 marker cell {tf.name} carries marker_id={cell_marker_id} "
                    f"!= canonical {MARKER_TOKEN_ID} ({MARKER_TEXT!r}); the marker DV "
                    "would measure the wrong slot. Refusing to join a wrong-token cell."
                )
            n_marker_id_checked += 1
        ctx = td.get("context_label", td.get("bystander_label", ""))
        persona = _marker_persona_from_context(ctx)
        if persona is None:
            continue
        source_cid = td.get("source_cid", "")
        t_logp = td.get("summary", {}).get("mean_logp_marker")
        b_logp = bd.get("summary", {}).get("mean_logp_marker")
        if t_logp is None or b_logp is None:
            continue
        key = (source_cid, persona)
        by_pair.setdefault(key, {"delta": [], "base": []})
        by_pair[key]["delta"].append(float(t_logp) - float(b_logp))
        by_pair[key]["base"].append(float(b_logp))
        base_logp_by_persona.setdefault(persona, []).append(float(b_logp))

    cells: list[dict] = []
    for (source_cid, persona), acc in by_pair.items():
        if source_cid == persona:
            continue  # diagonal (source==bystander) excluded, mirroring #518 off-diag
        delta = float(np.mean(acc["delta"]))
        base_prior = float(np.mean(acc["base"]))
        cells.append(
            {
                "source": source_cid,
                "bystander": persona,
                "delta": delta,
                "bystander_base_rate": base_prior,
                "source_base_rate": None,
                "completion_logprob": None,
                "source_resp_len_mean": None,
            }
        )

    bystander_base_rate = {p: float(np.mean(v)) for p, v in base_logp_by_persona.items()}
    sources = sorted({c["source"] for c in cells})
    bystanders = sorted({c["bystander"] for c in cells})
    return CellTable(
        behavior="marker",
        cells=cells,
        sources=sources,
        bystanders=bystanders,
        bystander_base_rate=bystander_base_rate,
        provenance={
            "source_root": str(marker_root),
            "n_cells": len(cells),
            "n_overlap_bystanders": len(bystanders),
            "n_marker_id_checked": n_marker_id_checked,
            "marker_token_id": MARKER_TOKEN_ID,
            "persona_aliases": dict(_MARKER_PERSONA_ALIASES),
            "note": (
                "#605 panel is context-keyed; only personas overlapping the "
                "24-persona panel are joined (plan §8/§12)."
            ),
        },
    )


# ──────────────────────────────────────────────────────────────────────────────
# Persona vectors + behavior directions
# ──────────────────────────────────────────────────────────────────────────────


def load_persona_vectors(
    persona_dirs: list[Path],
    layers: tuple[int, ...] = DEFAULT_LAYERS,
) -> dict[str, torch.Tensor]:
    """Load per-persona ``(n_layers, hidden)`` centroid tensors from one or more dirs.

    Each dir holds ``<persona>.pt`` files (and an ``all_centroids.pt`` aggregate which
    is skipped). Later dirs override earlier ones for the same persona (so the
    issue-657 freshly-extracted vectors override the #623 bank only where both
    exist — they should not overlap). Raises if a tensor's layer dim != len(layers).
    """
    out: dict[str, torch.Tensor] = {}
    for d in persona_dirs:
        if not d.is_dir():
            continue
        for pt in sorted(d.glob("*.pt")):
            if pt.name == "all_centroids.pt":
                continue
            t = torch.load(pt, weights_only=True)
            if t.ndim != 2 or t.shape[0] != len(layers):
                raise ValueError(
                    f"persona vector {pt} has shape {tuple(t.shape)}; expected "
                    f"({len(layers)}, hidden) for layers {layers}."
                )
            out[pt.stem] = t.float()
    if not out:
        raise FileNotFoundError(f"No persona-vector .pt files found under {persona_dirs}")
    return out


def load_behavior_direction(
    behavior: str,
    direction_dirs: dict[str, Path],
    layer: int,
    readout: str = "last_token",
) -> torch.Tensor:
    """Load one behavior direction at one layer + readout.

    ``readout`` is ``last_token`` (Method A headline) or ``response_avg`` (Method B).
    File convention (verbatim #623): ``<readout>_<layer>.pt``. ``direction_dirs``
    maps behavior -> dir (sycophancy -> #623 sycophancy_trait dir; refusal/marker/em ->
    their freshly-extracted dirs).
    """
    if behavior not in direction_dirs:
        raise KeyError(f"No direction dir registered for behavior {behavior!r}.")
    fp = direction_dirs[behavior] / f"{readout}_{layer}.pt"
    if not fp.exists():
        raise FileNotFoundError(f"Behavior direction not found: {fp}")
    return torch.load(fp, weights_only=True).float()


def _persona_vector_at_layer(
    persona_vectors: dict[str, torch.Tensor],
    personas: list[str],
    layer: int,
    layers: tuple[int, ...],
    centering: str,
) -> dict[str, np.ndarray]:
    """Build persona vectors (centroid_i - centroid_assistant) at one layer.

    With centering='global_mean', the centroid BANK is globally mean-centered over
    the panel BEFORE the (i - assistant) difference (the canonical #536 convention,
    `representation_shift.compute_cosine_matrix(centering="global_mean")`). Note: for
    cosine alignment the centering is a no-op for the Spearman rank (verified #623:
    lt_persona and gmc_persona arms produce identical rho) but it IS load-bearing
    for the prior_centroid_projection control, so we apply it consistently.
    """
    if layer not in layers:
        raise ValueError(f"layer {layer} not in stored layers {layers}")
    layer_idx = layers.index(layer)
    if BASELINE_PERSONA not in persona_vectors:
        raise ValueError(
            f"Baseline persona {BASELINE_PERSONA!r} missing — cannot form persona vectors."
        )
    present = [p for p in personas if p in persona_vectors]
    slab = torch.stack([persona_vectors[p][layer_idx] for p in present])  # (P, hidden)
    if centering == "global_mean":
        slab = slab - slab.mean(dim=0, keepdim=True)
    elif centering != "none":
        raise ValueError(f"centering must be 'global_mean' or 'none', got {centering!r}")
    by_persona = {p: slab[i].numpy() for i, p in enumerate(present)}
    assistant_vec = by_persona[BASELINE_PERSONA]
    return {p: v - assistant_vec for p, v in by_persona.items()}


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    denom = na * nb
    if denom == 0.0:
        return float("nan")  # zero-norm -> NaN; the caller's guard catches it
    return float(np.dot(a, b) / denom)


def compute_alignment(
    persona_vectors: dict[str, torch.Tensor],
    behavior_dir: torch.Tensor,
    personas: list[str],
    layer: int = HEADLINE_LAYER,
    layers: tuple[int, ...] = DEFAULT_LAYERS,
    centering: str = "global_mean",
) -> dict[str, float]:
    """Bystander-level cosine(persona vector, behavior direction) at one layer.

    Returns {persona: cosine}. ``behavior_dir`` is the (hidden,) direction at the
    SAME layer. Asserts the bank was centered (`Source: #536`) via the centering kwarg.
    """
    if behavior_dir.ndim != 1:
        raise ValueError(f"behavior_dir must be 1-D (hidden,), got {tuple(behavior_dir.shape)}")
    pvecs = _persona_vector_at_layer(persona_vectors, personas, layer, layers, centering)
    bdir = behavior_dir.numpy()
    return {p: _cosine(v, bdir) for p, v in pvecs.items()}


def prior_centroid_projection(
    persona_vectors: dict[str, torch.Tensor],
    base_rates: dict[str, float],
    personas: list[str],
    layer: int = HEADLINE_LAYER,
    layers: tuple[int, ...] = DEFAULT_LAYERS,
    centering: str = "global_mean",
) -> dict[str, float]:
    """M-Alts1 geometry-side prior control.

    Computes the base-rate-weighted persona-vector centroid
    ``b_prior_centroid = sum_j base_rate_j * persona_vector_j / sum_j base_rate_j``
    (the within-panel direction along which high-prior personas point), then returns
    per-bystander ``cosine(persona_vector_i, b_prior_centroid)``.

    This catches an EMERGENT geometry-side restatement of the base prior that the
    SCALAR-prior partial does NOT remove: if the behavior direction ``b`` lies near
    ``b_prior_centroid``, high-prior personas point toward ``b`` for reasons that ARE
    the prior. The doubly-partialled H3 read partials this out in addition to the
    scalar prior.
    """
    pvecs = _persona_vector_at_layer(persona_vectors, personas, layer, layers, centering)
    # weighted centroid over the personas that have BOTH a vector and a base rate
    usable = [p for p in pvecs if p in base_rates]
    if not usable:
        raise ValueError("No persona has both a vector and a base rate for prior_centroid.")
    weights = np.array([base_rates[p] for p in usable], dtype=float)
    wsum = float(weights.sum())
    if wsum == 0.0:
        raise ValueError("Base-rate weights sum to zero; prior_centroid undefined.")
    stack = np.stack([pvecs[p] for p in usable])  # (U, hidden)
    centroid = (weights[:, None] * stack).sum(axis=0) / wsum  # (hidden,)
    return {p: _cosine(pvecs[p], centroid) for p in pvecs}


# ──────────────────────────────────────────────────────────────────────────────
# Partial Spearman (rank-residualization) + Spearman helpers
# ──────────────────────────────────────────────────────────────────────────────


def _rankdata(a: np.ndarray) -> np.ndarray:
    """Average-rank transform (ties averaged), matching scipy.stats.rankdata default."""
    from scipy.stats import rankdata

    return rankdata(a)


def _residualize_ranks(y_rank: np.ndarray, z_rank: np.ndarray) -> np.ndarray:
    """OLS-residualize y_rank on z_rank (z_rank shape (n,) or (n, k)) with an intercept."""
    if z_rank.ndim == 1:
        z_rank = z_rank[:, None]
    n = y_rank.shape[0]
    design = np.column_stack([np.ones(n), z_rank])  # (n, k+1)
    beta, *_ = np.linalg.lstsq(design, y_rank, rcond=None)
    return y_rank - design @ beta


def partial_spearman(
    x: np.ndarray, y: np.ndarray, z: np.ndarray | None = None
) -> tuple[float, float, int]:
    """Spearman partial-rho(x, y | z) via rank-residualization.

    z is None (plain Spearman), a scalar covariate (n,), or a matrix (n, k) for the
    DOUBLY-partialled read (z = [bystander_base_rate, prior_centroid_projection]).
    Rank x, y, and each z column; OLS-residualize rank(x) and rank(y) on rank(z) (+
    intercept); Spearman == Pearson on the residual ranks. Returns (rho, p, n).
    Returns (nan, nan, n) on a degenerate (zero-variance residual) input rather than
    raising — the caller decides (a bootstrap resample can be degenerate).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = x.shape[0]
    if n < 3:
        return float("nan"), float("nan"), n
    if z is None:
        res = spearmanr(x, y)
        return float(res.correlation), float(res.pvalue), n
    z = np.asarray(z, dtype=float)
    xr = _rankdata(x)
    yr = _rankdata(y)
    if z.ndim == 1:
        zr = _rankdata(z)
    else:
        zr = np.column_stack([_rankdata(z[:, j]) for j in range(z.shape[1])])
    xres = _residualize_ranks(xr, zr)
    yres = _residualize_ranks(yr, zr)
    if np.std(xres) == 0.0 or np.std(yres) == 0.0:
        return float("nan"), float("nan"), n
    res = spearmanr(xres, yres)
    return float(res.correlation), float(res.pvalue), n


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation; nan on a zero-variance input."""
    if np.std(a) == 0.0 or np.std(b) == 0.0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def partial_spearman_eiv(
    x: np.ndarray,
    y: np.ndarray,
    z_noisy: np.ndarray,
    z_clean: np.ndarray | None,
    reliability: float,
) -> tuple[float, int]:
    """EIV-disattenuated Spearman partial-rho(x, y | z_noisy, z_clean).

    The plain ``partial_spearman`` rank-residualizes each covariate, so a positive
    monotone rescale of a covariate (``prior * sqrt(rel)``) leaves the ranks — and
    therefore the partial — UNCHANGED. That made the M-Stats1 reliability band a
    rank-invariant no-op (#657 round-1 blocker). This implements the classical
    errors-in-variables (EIV) disattenuation in a space ranks DO respond to:
    the disattenuated bivariate partial-correlation formula.

    Method (Spearman = Pearson on ranks):
      1. If a CLEAN covariate ``z_clean`` (the geometry-side prior_centroid_projection,
         measured exactly) is present, OLS-residualize rank(x), rank(y), rank(z_noisy)
         on rank(z_clean) first — sequential partialling, so ``z_clean`` is removed
         exactly and only the noisy covariate carries the EIV correction.
      2. On those residuals, the partial of (x, y) controlling for the noisy ``z`` is
         the bivariate partial-correlation formula
         ``r_xy.z = (r_xy - r_xz*r_yz) / sqrt((1-r_xz^2)(1-r_yz^2))``.
      3. Disattenuate the NOISY covariate's two cross-correlations by
         ``r_xz -> r_xz/sqrt(R)`` and ``r_yz -> r_yz/sqrt(R)`` (the classical
         attenuation correction — a noisy covariate's observed correlations are
         attenuated by ``sqrt(R)``), clipping to (-0.999, 0.999) for stability.

    As ``reliability`` DECREASES, the disattenuated ``r_xz`` / ``r_yz`` GROW, so MORE
    of the true noisy-covariate effect is removed and the partial moves FURTHER from
    the raw correlation (toward the perfect-Z singly-partialled value). The band
    points are therefore distinct and monotone in reliability — the property the
    no-op version lacked. At ``reliability == 1`` it reduces to the ordinary rank
    partial correlation (no correction). This bivariate-formula form avoids the
    full-matrix inversion of a strongly disattenuated correlation matrix, which can
    go non-positive-definite at low reliability and yield singular / clipped garbage.

    Returns (partial_rho, n); (nan, n) on a degenerate input rather than raising.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z_noisy = np.asarray(z_noisy, dtype=float)
    n = x.shape[0]
    if n < 3:
        return float("nan"), n
    rel = float(reliability)
    if not (0.0 < rel <= 1.0):
        rel = min(1.0, max(1e-6, rel))

    xr = _rankdata(x)
    yr = _rankdata(y)
    zr = _rankdata(z_noisy)
    if np.std(xr) == 0.0 or np.std(yr) == 0.0 or np.std(zr) == 0.0:
        return float("nan"), n

    # Step 1: residualize the exactly-measured clean covariate out of x, y, z first.
    if z_clean is not None:
        zcr = _rankdata(np.asarray(z_clean, dtype=float))
        if np.std(zcr) > 0.0:
            xr = _residualize_ranks(xr, zcr)
            yr = _residualize_ranks(yr, zcr)
            zr = _residualize_ranks(zr, zcr)
            if np.std(xr) == 0.0 or np.std(yr) == 0.0 or np.std(zr) == 0.0:
                return float("nan"), n

    # Step 2: bivariate (disattenuated) partial of (x, y) controlling for noisy z.
    r_xy = _pearson(xr, yr)
    r_xz = _pearson(xr, zr)
    r_yz = _pearson(yr, zr)
    if any(np.isnan(v) for v in (r_xy, r_xz, r_yz)):
        return float("nan"), n

    scale = 1.0 / float(np.sqrt(rel))
    r_xz = float(np.clip(r_xz * scale, -0.999, 0.999))
    r_yz = float(np.clip(r_yz * scale, -0.999, 0.999))

    denom = float(np.sqrt((1.0 - r_xz**2) * (1.0 - r_yz**2)))
    if denom <= 0.0:
        return float("nan"), n
    partial = (r_xy - r_xz * r_yz) / denom
    if not np.isfinite(partial):
        return float("nan"), n
    return float(np.clip(partial, -1.0, 1.0)), n


# ──────────────────────────────────────────────────────────────────────────────
# Source-fixed-effect residualization (matches #518 rho_FE) + frame assembly
# ──────────────────────────────────────────────────────────────────────────────


def _source_fe_residualize(cells: list[dict], col: str) -> np.ndarray:
    """Subtract the within-source mean of ``col`` from each cell (#518's rho_FE)."""
    by_source: dict[str, list[float]] = {}
    for c in cells:
        by_source.setdefault(c["source"], []).append(float(c[col]))
    src_mean = {s: float(np.mean(v)) for s, v in by_source.items()}
    return np.array([float(c[col]) - src_mean[c["source"]] for c in cells], dtype=float)


def build_predictor_frame(
    table: CellTable,
    alignment: dict[str, float],
    prior_centroid: dict[str, float],
) -> dict:
    """Attach the bystander-level predictors to each off-diag cell.

    Drops cells whose bystander lacks a persona vector (NaN alignment) — graceful
    degradation, reported in the returned ``dropped`` list. Returns a dict with
    per-cell arrays + the resolvable bystander list (the resampling unit).
    """
    kept: list[dict] = []
    dropped: list[dict] = []
    for c in table.cells:
        b = c["bystander"]
        align = alignment.get(b, float("nan"))
        pcp = prior_centroid.get(b, float("nan"))
        if np.isnan(align) or np.isnan(pcp):
            dropped.append({"source": c["source"], "bystander": b})
            continue
        row = dict(c)
        row["align"] = float(align)
        row["prior_centroid_projection"] = float(pcp)
        kept.append(row)
    resolvable_bystanders = sorted({c["bystander"] for c in kept})
    return {
        "cells": kept,
        "dropped": dropped,
        "resolvable_bystanders": resolvable_bystanders,
        "n_resolvable": len(resolvable_bystanders),
    }


def delta_r2_align_beyond_prior_eiv(
    x: np.ndarray,
    y: np.ndarray,
    z_noisy: np.ndarray,
    z_clean: np.ndarray | None,
    reliability: float,
) -> float:
    """EIV-disattenuated incremental R^2 of ``align`` over the (disattenuated) prior set.

    Incremental R^2 of adding ``x`` (align) to the baseline ``[z_noisy, z_clean]``,
    via the identity ``ΔR^2 = (1 - R^2_baseline) * partial_{xy|baseline}^2`` where the
    NOISY covariate's contribution is EIV-disattenuated by ``1/sqrt(reliability)``
    (the same correction as ``partial_spearman_eiv``). Built from the bivariate
    disattenuated partial-correlation + residual-variance terms rather than inverting
    a strongly-disattenuated correlation matrix (which can go non-positive-definite at
    low reliability) — so it stays finite across the whole band.

    R^2_baseline = r^2(y, z_clean) + (1 - r^2(y, z_clean)) * r^2_{y, z_noisy | z_clean,
    disattenuated}; the second term grows as reliability falls (the prior is treated
    as cleaner), so the baseline explains more and align's increment shifts — the
    band's ΔR² points are distinct (the no-op version did not compute ΔR² per band
    point at all). At reliability == 1 it is the ordinary in-sample rank ΔR². NaN-safe.
    This is an IN-SAMPLE read for the reliability-band sensitivity; the HEADLINE ΔR²
    stays the grouped-CV held-out read in ``lopo_heldout`` / ``_grouped_cv_delta_r2``.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z_noisy = np.asarray(z_noisy, dtype=float)
    n = x.shape[0]
    if n < 4:
        return float("nan")
    rel = float(reliability)
    if not (0.0 < rel <= 1.0):
        rel = min(1.0, max(1e-6, rel))

    xr = _rankdata(x)
    yr = _rankdata(y)
    zr = _rankdata(z_noisy)
    if np.std(xr) == 0.0 or np.std(yr) == 0.0 or np.std(zr) == 0.0:
        return float("nan")

    scale = 1.0 / float(np.sqrt(rel))

    # r^2 of y explained by the exactly-measured clean covariate, then residualize.
    r2_clean = 0.0
    if z_clean is not None:
        zcr = _rankdata(np.asarray(z_clean, dtype=float))
        if np.std(zcr) > 0.0:
            r_y_zc = _pearson(yr, zcr)
            if np.isnan(r_y_zc):
                return float("nan")
            r2_clean = r_y_zc**2
            xr = _residualize_ranks(xr, zcr)
            yr = _residualize_ranks(yr, zcr)
            zr = _residualize_ranks(zr, zcr)
            if np.std(xr) == 0.0 or np.std(yr) == 0.0 or np.std(zr) == 0.0:
                return float("nan")

    # disattenuated r(y, z_noisy) on the (clean-residualized) ranks -> baseline R^2.
    r_yz = _pearson(yr, zr)
    if np.isnan(r_yz):
        return float("nan")
    r_yz_dis = float(np.clip(r_yz * scale, -0.999, 0.999))
    r2_baseline = r2_clean + (1.0 - r2_clean) * (r_yz_dis**2)
    if r2_baseline >= 1.0:
        return 0.0

    # align's increment beyond the (disattenuated) baseline: squared partial * the
    # residual-variance fraction the baseline left unexplained.
    r_xy = _pearson(xr, yr)
    r_xz = _pearson(xr, zr)
    if np.isnan(r_xy) or np.isnan(r_xz):
        return float("nan")
    r_xz_dis = float(np.clip(r_xz * scale, -0.999, 0.999))
    pdenom = float(np.sqrt((1.0 - r_xz_dis**2) * (1.0 - r_yz_dis**2)))
    if pdenom <= 0.0:
        return float("nan")
    partial = (r_xy - r_xz_dis * r_yz_dis) / pdenom
    partial = float(np.clip(partial, -0.999, 0.999))
    return float((1.0 - r2_baseline) * partial**2)


# ──────────────────────────────────────────────────────────────────────────────
# Held-out LOPO + grouped-CV DeltaR2
# ──────────────────────────────────────────────────────────────────────────────


def _grouped_cv_delta_r2(
    cells: list[dict],
    predictor_cols: list[str],
    baseline_cols: list[str],
    dv_col: str,
    group_col: str = "bystander",
) -> tuple[float, float]:
    """Grouped-CV (leave-one-group-out) incremental R^2 of full over baseline.

    full model = baseline_cols + predictor_cols; baseline model = baseline_cols.
    Out-of-fold predictions held-out by ``group_col`` (the bystander). Returns
    (delta_r2 = R2_full - R2_baseline, r2_full). Rank-based features (Spearman
    convention) — rank within the TRAIN fold, apply the same rank map to the test
    cell via interpolation. OLS on ranks, leave-one-group-out.

    Hot path (called once per bootstrap replicate, B=10000), so feature/dv/group
    arrays are extracted ONCE up front and the fold loop slices numpy arrays — no
    per-fold dict comprehensions (the pre-vectorization version's per-cell dict
    lookups dominated the bootstrap wall-time this round).
    """
    n = len(cells)
    group_vals = [c[group_col] for c in cells]
    groups = sorted(set(group_vals))
    if len(groups) < 3:
        return float("nan"), float("nan")
    group_arr = np.array(group_vals, dtype=object)
    y = np.array([float(c[dv_col]) for c in cells], dtype=float)
    yr = _rankdata(y)

    all_cols = list(dict.fromkeys([*baseline_cols, *predictor_cols]))
    col_arr = {fc: np.array([float(c[fc]) for c in cells], dtype=float) for fc in all_cols}

    def _fit_predict(feat_cols: list[str]) -> np.ndarray:
        preds = np.full(n, np.nan)
        for held in groups:
            test_mask = group_arr == held
            train_mask = ~test_mask
            n_train = int(train_mask.sum())
            if n_train < len(feat_cols) + 2:
                continue
            train_idx = np.where(train_mask)[0]
            test_idx = np.where(test_mask)[0]
            # rank features within the train fold
            Xtr = np.column_stack([_rankdata(col_arr[fc][train_idx]) for fc in feat_cols])
            ytr = _rankdata(y[train_idx])
            design_tr = np.column_stack([np.ones(n_train), Xtr])
            beta, *_ = np.linalg.lstsq(design_tr, ytr, rcond=None)
            # test features: rank within the union (train+test), take the test rows
            union_idx = np.concatenate([train_idx, test_idx])
            Xte_cols = []
            for fc in feat_cols:
                allranks = _rankdata(col_arr[fc][union_idx])
                Xte_cols.append(allranks[n_train:])
            Xte = np.column_stack(Xte_cols)
            design_te = np.column_stack([np.ones(len(test_idx)), Xte])
            preds[test_idx] = design_te @ beta
        return preds

    def _r2(preds: np.ndarray) -> float:
        mask = ~np.isnan(preds)
        if mask.sum() < 3:
            return float("nan")
        yy = yr[mask]
        pp = preds[mask]
        ss_res = float(np.sum((yy - pp) ** 2))
        ss_tot = float(np.sum((yy - yy.mean()) ** 2))
        if ss_tot == 0.0:
            return float("nan")
        return 1.0 - ss_res / ss_tot

    r2_base = _r2(_fit_predict(baseline_cols))
    r2_full = _r2(_fit_predict(baseline_cols + predictor_cols))
    if np.isnan(r2_base) or np.isnan(r2_full):
        return float("nan"), r2_full
    return r2_full - r2_base, r2_full


def lopo_heldout(
    frame: dict,
    predictor_col: str = "align",
    dv_col: str = "delta",
    baseline_cols: tuple[str, ...] = ("bystander_base_rate", "prior_centroid_projection"),
    group_col: str = "bystander",
    source_fe: bool = True,
) -> dict:
    """Leave-one-persona-out held-out read of the predictor against the leakage DV.

    Returns:
      - held_out_spearman: out-of-fold Spearman rho(predictor, dv) where each cell's
        predictor rank is computed leave-its-bystander-out (honest out-of-sample).
      - delta_r2_align_beyond_prior + r2_full (grouped-CV ΔR² over the baseline set).
      - delta_r2_prior_beyond_align (the reverse, symmetric #605 read).
      - n_groups (resolvable bystanders).

    With ``source_fe`` the DV is within-source-mean residualized (#518's rho_FE).
    """
    cells = frame["cells"]
    if not cells:
        return {"held_out_spearman": float("nan"), "n_groups": 0, "note": "no resolvable cells"}

    work = [dict(c) for c in cells]
    dv_use = dv_col
    if source_fe:
        fe = _source_fe_residualize(work, dv_col)
        for c, v in zip(work, fe, strict=True):
            c["delta_fe"] = float(v)
        dv_use = "delta_fe"

    groups = sorted({c[group_col] for c in work})

    # Held-out Spearman: for each cell, rank the predictor over the OTHER bystanders'
    # cells, place the held cell by interpolation, Spearman the held-out predictor
    # ranks against the DV ranks. The predictor is bystander-level (constant within a
    # bystander), so LOPO = exclude that bystander's value from the rank reference.
    pred_vals = np.array([float(c[predictor_col]) for c in work], dtype=float)
    dv_vals = np.array([float(c[dv_use]) for c in work], dtype=float)
    held_pred = np.full(len(work), np.nan)
    for held in groups:
        ref_idx = [i for i, c in enumerate(work) if c[group_col] != held]
        test_idx = [i for i, c in enumerate(work) if c[group_col] == held]
        if len(ref_idx) < 3:
            continue
        ref_vals = pred_vals[ref_idx]
        order = np.argsort(ref_vals)
        ranked_ref = ref_vals[order]
        # interpolated rank position of each held cell's predictor in the ref set
        for i in test_idx:
            pos = float(np.searchsorted(ranked_ref, pred_vals[i], side="left"))
            held_pred[i] = pos
    mask = ~np.isnan(held_pred)
    if mask.sum() >= 3 and np.std(held_pred[mask]) > 0 and np.std(dv_vals[mask]) > 0:
        ho = spearmanr(held_pred[mask], dv_vals[mask])
        held_out_spearman = float(ho.correlation)
    else:
        held_out_spearman = float("nan")

    # grouped-CV ΔR²: full = [baseline_cols, align] vs baseline_cols
    work_for_cv = [dict(c) for c in work]
    for c in work_for_cv:
        c[dv_use] = c[dv_use]
    delta_r2_fwd, r2_full = _grouped_cv_delta_r2(
        work_for_cv, [predictor_col], list(baseline_cols), dv_use, group_col
    )
    # reverse: prior-beyond-align (baseline = [align], add the scalar prior)
    delta_r2_rev, _ = _grouped_cv_delta_r2(
        work_for_cv, ["bystander_base_rate"], [predictor_col], dv_use, group_col
    )

    return {
        "held_out_spearman": held_out_spearman,
        "delta_r2_align_beyond_prior": delta_r2_fwd,
        "r2_full": r2_full,
        "delta_r2_prior_beyond_align": delta_r2_rev,
        "n_groups": len(groups),
        "dv_col": dv_use,
        "source_fe": source_fe,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Bootstrap over personas (the held-out bystander is the resampling unit)
# ──────────────────────────────────────────────────────────────────────────────


def bootstrap_over_personas(
    frame: dict,
    stat_fn,
    b: int = BOOTSTRAP_B,
    seed: int = BOOTSTRAP_SEED,
    alpha: float = 0.05,
) -> dict:
    """Bootstrap CI by resampling the HELD-OUT BYSTANDER PERSONAS (NOT cells).

    Cells sharing a bystander are not independent (the predictor is bystander-level),
    so the resampling unit is the bystander. For each replicate, sample bystanders
    with replacement, gather all their cells, recompute ``stat_fn(resampled_frame)``,
    and take percentile CIs. A degenerate replicate (stat is NaN) is dropped from the
    CI (and counted). Returns {point, ci_lo, ci_hi, n_valid, n_bystanders}.
    """
    cells = frame["cells"]
    bystanders = frame["resolvable_bystanders"]
    by_b: dict[str, list[dict]] = {}
    for c in cells:
        by_b.setdefault(c["bystander"], []).append(c)

    point = stat_fn(frame)
    if isinstance(point, dict):
        raise TypeError("stat_fn must return a float for bootstrap_over_personas")

    rng = np.random.default_rng(seed)
    nB = len(bystanders)
    vals: list[float] = []
    for _ in range(b):
        sampled = rng.integers(0, nB, size=nB)
        resampled_cells: list[dict] = []
        resampled_set: set[str] = set()
        for idx in sampled:
            bp = bystanders[idx]
            resampled_cells.extend(by_b[bp])
            resampled_set.add(bp)
        rframe = {
            "cells": resampled_cells,
            "resolvable_bystanders": sorted(resampled_set),
            "n_resolvable": len(resampled_set),
            "dropped": [],
        }
        v = stat_fn(rframe)
        if v is not None and not (isinstance(v, float) and np.isnan(v)):
            vals.append(float(v))
    if not vals:
        return {
            "point": float(point) if point is not None else float("nan"),
            "ci_lo": float("nan"),
            "ci_hi": float("nan"),
            "n_valid": 0,
            "n_bystanders": nB,
        }
    arr = np.array(vals, dtype=float)
    return {
        "point": float(point) if point is not None and not np.isnan(point) else float("nan"),
        "ci_lo": float(np.quantile(arr, alpha / 2)),
        "ci_hi": float(np.quantile(arr, 1 - alpha / 2)),
        "n_valid": len(vals),
        "n_bystanders": nB,
    }


def bootstrap_over_personas_multi(
    frame: dict,
    stat_fn,
    keys: list[str],
    b: int = BOOTSTRAP_B,
    seed: int = BOOTSTRAP_SEED,
    alpha: float = 0.05,
) -> dict[str, dict]:
    """Bootstrap CIs for MULTIPLE named statistics from ONE resampling pass.

    Same persona-resampling unit as ``bootstrap_over_personas`` (the held-out
    bystander), but ``stat_fn`` returns a dict {key: float}; per replicate it is
    called ONCE and each key's value is collected. This is the throughput fix for
    the held-out Spearman + grouped-CV ΔR², which both come out of a single
    ``lopo_heldout`` call — computing them in separate bootstraps re-ran the
    expensive LOPO grouped-CV per statistic (one behavior's 3x B=10000 took >5 min;
    incident this round). Per-key NaN replicates are dropped independently.
    Returns {key: {point, ci_lo, ci_hi, n_valid, n_bystanders}}.
    """
    cells = frame["cells"]
    bystanders = frame["resolvable_bystanders"]
    by_b: dict[str, list[dict]] = {}
    for c in cells:
        by_b.setdefault(c["bystander"], []).append(c)

    point = stat_fn(frame)
    if not isinstance(point, dict):
        raise TypeError("stat_fn must return a dict[str, float] for bootstrap_over_personas_multi")

    rng = np.random.default_rng(seed)
    nB = len(bystanders)
    collected: dict[str, list[float]] = {k: [] for k in keys}
    for _ in range(b):
        sampled = rng.integers(0, nB, size=nB)
        resampled_cells: list[dict] = []
        resampled_set: set[str] = set()
        for idx in sampled:
            bp = bystanders[idx]
            resampled_cells.extend(by_b[bp])
            resampled_set.add(bp)
        rframe = {
            "cells": resampled_cells,
            "resolvable_bystanders": sorted(resampled_set),
            "n_resolvable": len(resampled_set),
            "dropped": [],
        }
        out = stat_fn(rframe)
        for k in keys:
            v = out.get(k)
            if v is not None and not (isinstance(v, float) and np.isnan(v)):
                collected[k].append(float(v))

    result: dict[str, dict] = {}
    for k in keys:
        vals = collected[k]
        pk = point.get(k)
        if not vals:
            result[k] = {
                "point": float(pk) if pk is not None and not np.isnan(pk) else float("nan"),
                "ci_lo": float("nan"),
                "ci_hi": float("nan"),
                "n_valid": 0,
                "n_bystanders": nB,
            }
            continue
        arr = np.array(vals, dtype=float)
        result[k] = {
            "point": float(pk) if pk is not None and not np.isnan(pk) else float("nan"),
            "ci_lo": float(np.quantile(arr, alpha / 2)),
            "ci_hi": float(np.quantile(arr, 1 - alpha / 2)),
            "n_valid": len(vals),
            "n_bystanders": nB,
        }
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Shuffled-direction null
# ──────────────────────────────────────────────────────────────────────────────


def shuffled_direction_null(
    persona_vectors: dict[str, torch.Tensor],
    behavior_dir: torch.Tensor,
    table: CellTable,
    base_rates: dict[str, float],
    layer: int = HEADLINE_LAYER,
    layers: tuple[int, ...] = DEFAULT_LAYERS,
    centering: str = "global_mean",
    b: int = SHUFFLE_B,
    seed: int = BOOTSTRAP_SEED,
) -> dict:
    """Null distribution of the held-out Spearman from a RANDOMLY ROTATED direction.

    Tests b-vs-RANDOM (the #555-style geometry noise floor). For each of ``b``
    replicates, replace the behavior direction with a random unit vector of the same
    norm, recompute alignment, build the frame, and read the LOPO held-out Spearman.
    Returns {observed, null_mean, null_p95, p_value} where p_value = fraction of null
    held-out rho >= observed (one-sided). NOT the prior-centroid control (M-Alts1
    covers b-vs-prior-centroid separately).
    """
    personas = table.bystanders
    bdir = behavior_dir.numpy()
    bnorm = float(np.linalg.norm(bdir))
    pcp = prior_centroid_projection(persona_vectors, base_rates, personas, layer, layers, centering)

    def _held_out_rho(direction: np.ndarray) -> float:
        align = compute_alignment(
            persona_vectors,
            torch.from_numpy(direction.astype(np.float32)),
            personas,
            layer,
            layers,
            centering,
        )
        frame = build_predictor_frame(table, align, pcp)
        out = lopo_heldout(frame)
        return out.get("held_out_spearman", float("nan"))

    observed = _held_out_rho(bdir)
    rng = np.random.default_rng(seed)
    dim = bdir.shape[0]
    null_vals: list[float] = []
    for _ in range(b):
        r = rng.standard_normal(dim)
        r = r / (np.linalg.norm(r) + 1e-12) * bnorm
        v = _held_out_rho(r)
        if not np.isnan(v):
            null_vals.append(v)
    if not null_vals or np.isnan(observed):
        return {
            "observed": float(observed),
            "null_mean": float("nan"),
            "null_p95": float("nan"),
            "p_value": float("nan"),
            "n_null": len(null_vals),
        }
    null_arr = np.array(null_vals, dtype=float)
    p_value = float(np.mean(null_arr >= observed))
    return {
        "observed": float(observed),
        "null_mean": float(np.mean(null_arr)),
        "null_p95": float(np.quantile(null_arr, 0.95)),
        "p_value": p_value,
        "n_null": len(null_vals),
    }


# ──────────────────────────────────────────────────────────────────────────────
# M-Stats1 bounded-attenuation reliability band
# ──────────────────────────────────────────────────────────────────────────────


def _binomial_reliability(base_rates: dict[str, float], n: int) -> float:
    """Reliability of the per-bystander base-rate estimate at generation count n.

    reliability = signal_var / (signal_var + error_var). signal_var = variance of
    the per-bystander base rates across the panel; error_var = mean binomial SE^2 =
    mean p(1-p)/n. A noisier (smaller-n) estimate -> lower reliability. Clamped to
    (0, 1]. For marker (the base rate is a log-prob, not a 0-1 rate) the binomial SE
    is not defined; the caller skips the band for marker (handled upstream).
    """
    ps = np.array(list(base_rates.values()), dtype=float)
    signal_var = float(np.var(ps))
    if signal_var <= 0.0:
        return 1.0
    error_var = float(np.mean(ps * (1.0 - ps) / max(n, 1)))
    rel = signal_var / (signal_var + error_var)
    return float(min(1.0, max(1e-6, rel)))


def _band_point_stats(cells: list[dict], reliability: float) -> dict[str, float]:
    """The two band-point statistics at one reliability: EIV partial-rho + ΔR².

    Both read off the SAME disattenuated rank-correlation matrix (one resampled
    frame's cells), so the bootstrap recomputes them together per replicate. The
    DV is the within-source-mean-residualized leakage delta (#518's rho_FE).
    """
    if len(cells) < 4:
        return {"doubly_partialled_rho": float("nan"), "delta_r2_align_beyond_prior": float("nan")}
    align = np.array([float(c["align"]) for c in cells], dtype=float)
    dv_fe = _source_fe_residualize(cells, "delta")
    prior = np.array([float(c["bystander_base_rate"]) for c in cells], dtype=float)
    pcp = np.array([float(c["prior_centroid_projection"]) for c in cells], dtype=float)
    rho, _n = partial_spearman_eiv(align, dv_fe, prior, pcp, reliability)
    dr2 = delta_r2_align_beyond_prior_eiv(align, dv_fe, prior, pcp, reliability)
    return {"doubly_partialled_rho": rho, "delta_r2_align_beyond_prior": dr2}


def attenuation_sensitivity(
    frame: dict,
    base_rates: dict[str, float],
    bracket_n: dict[str, int] | None = None,
    behavior: str = "",
    b: int = BOOTSTRAP_B,
    seed: int = BOOTSTRAP_SEED,
    alpha: float = 0.05,
) -> dict:
    """M-Stats1 bounded-attenuation reliability band on the noisy partialled prior.

    The partialled covariate ``bystander_base_rate`` is a noisy point scalar (no SE
    in the #518 substrate), so partialling under-corrects and a marginal partial-rho
    can be inflated. We bracket the prior's reliability from the per-bystander
    generation count (binomial SE at n_low/n_expected/n_high) and report the
    doubly-partialled partial-rho AND grouped ΔR² under each reliability point —
    EACH with a bootstrap CI over the held-out bystander personas — alongside the
    observed-prior (R=1, no correction) headline.

    The reliability-corrected read disattenuates the noisy prior via the classical
    errors-in-variables correction on the rank-correlation matrix
    (``partial_spearman_eiv`` / ``delta_r2_align_beyond_prior_eiv``): a less-reliable
    prior has its correlations scaled UP by 1/sqrt(R), so MORE of the true prior
    effect is removed and the partial moves further from the raw correlation. This
    is NOT rank-invariant (the round-1 ``prior*sqrt(R)`` no-op was — rank
    residualization erased the rescale), so the three band points are distinct.

    For ``behavior == "marker"`` the base rate is a log-prob (not a 0-1 rate), so the
    binomial SE is undefined; the band is SKIPPED and the headline observed-prior read
    is reported with ``band_applicable: False``.

    ``b``/``seed``/``alpha`` control the per-band-point bootstrap (resampling the
    held-out bystander, matching the headline gate). With ``b <= 1`` (smoke) the band
    is reported with point estimates only (no CI).
    """
    bracket_n = bracket_n or RELIABILITY_BRACKET_N
    cells = frame["cells"]
    if not cells:
        return {"band_applicable": False, "note": "no resolvable cells", "points": {}}

    align = np.array([float(c["align"]) for c in cells], dtype=float)
    dv_fe = _source_fe_residualize(cells, "delta")
    prior = np.array([float(c["bystander_base_rate"]) for c in cells], dtype=float)
    pcp = np.array([float(c["prior_centroid_projection"]) for c in cells], dtype=float)

    # observed-prior (R=1) doubly-partialled read (no correction == ordinary partial).
    rho_obs, _p_obs, n_obs = partial_spearman(align, dv_fe, np.column_stack([prior, pcp]))

    points: dict[str, dict] = {
        "R_observed": {"reliability": 1.0, "doubly_partialled_rho": rho_obs, "n": n_obs}
    }

    if behavior == "marker":
        return {
            "band_applicable": False,
            "note": "marker base rate is a log-prob; binomial reliability undefined.",
            "observed_doubly_partialled_rho": rho_obs,
            "points": points,
        }

    # R_low uses the LARGEST n (prior treated as cleanest -> least disattenuation),
    # R_high uses the SMALLEST n (prior treated as noisiest -> most disattenuation).
    rel_by_label = {
        "R_observed": 1.0,
        "R_low": _binomial_reliability(base_rates, bracket_n["n_high"]),  # largest n
        "R_expected": _binomial_reliability(base_rates, bracket_n["n_expected"]),
        "R_high": _binomial_reliability(base_rates, bracket_n["n_low"]),  # smallest n
    }

    # Point estimates (EIV partial-rho + ΔR²) at each reliability label.
    for label in ("R_observed", "R_low", "R_expected", "R_high"):
        rel = rel_by_label[label]
        st = _band_point_stats(cells, rel)
        points[label] = {
            "reliability": rel,
            "doubly_partialled_rho": st["doubly_partialled_rho"],
            "delta_r2_align_beyond_prior": st["delta_r2_align_beyond_prior"],
            "n": len(cells),
        }

    # Per-band-point bootstrap CIs (resample the held-out bystander, the same unit as
    # the headline gate). One resampling pass collects BOTH statistics at ALL four
    # reliability points so the expensive resample loop runs once.
    if b is not None and b > 1:
        bystanders = frame.get("resolvable_bystanders") or sorted({c["bystander"] for c in cells})
        by_b: dict[str, list[dict]] = {}
        for c in cells:
            by_b.setdefault(c["bystander"], []).append(c)
        nB = len(bystanders)
        keys = [
            (label, stat)
            for label in ("R_observed", "R_low", "R_expected", "R_high")
            for stat in ("doubly_partialled_rho", "delta_r2_align_beyond_prior")
        ]
        collected: dict[tuple[str, str], list[float]] = {k: [] for k in keys}
        rng = np.random.default_rng(seed)
        for _ in range(b):
            sampled = rng.integers(0, nB, size=nB)
            rcells: list[dict] = []
            for idx in sampled:
                rcells.extend(by_b[bystanders[idx]])
            for label in ("R_observed", "R_low", "R_expected", "R_high"):
                st = _band_point_stats(rcells, rel_by_label[label])
                for stat in ("doubly_partialled_rho", "delta_r2_align_beyond_prior"):
                    v = st[stat]
                    if v is not None and not (isinstance(v, float) and np.isnan(v)):
                        collected[(label, stat)].append(float(v))
        for label in ("R_observed", "R_low", "R_expected", "R_high"):
            for stat in ("doubly_partialled_rho", "delta_r2_align_beyond_prior"):
                vals = collected[(label, stat)]
                ci_key = f"{stat}_ci"
                if vals:
                    arr = np.array(vals, dtype=float)
                    points[label][ci_key] = {
                        "ci_lo": float(np.quantile(arr, alpha / 2)),
                        "ci_hi": float(np.quantile(arr, 1 - alpha / 2)),
                        "n_valid": len(vals),
                        "n_bystanders": nB,
                    }
                else:
                    points[label][ci_key] = {
                        "ci_lo": float("nan"),
                        "ci_hi": float("nan"),
                        "n_valid": 0,
                        "n_bystanders": nB,
                    }

    band_rhos = [
        points[k]["doubly_partialled_rho"]
        for k in ("R_low", "R_expected", "R_high")
        if not np.isnan(points[k]["doubly_partialled_rho"])
    ]
    return {
        "band_applicable": True,
        "bracket_n": dict(bracket_n),
        "observed_doubly_partialled_rho": rho_obs,
        "ci_computed": bool(b is not None and b > 1),
        "points": points,
        "band_min_rho": float(min(band_rhos)) if band_rhos else float("nan"),
        "band_max_rho": float(max(band_rhos)) if band_rhos else float("nan"),
    }


# ──────────────────────────────────────────────────────────────────────────────
# v6 primary-set membership gates (M-Alts3 marker demotion + Phase-0 overlap)
# ──────────────────────────────────────────────────────────────────────────────

# Minimum #605-marker-panel <-> 24-persona-panel overlap for the marker LEAKAGE
# bake-off (plan §7/§8/§12). Below this the marker arm is DV-(a)-only (excluded
# from the leakage bake-off + the primary H3 gate set). Kept consistent with the
# bake-off dispatcher's MARKER_MIN_OVERLAP.
MARKER_MIN_OVERLAP: int = 8


def marker_panel_overlap(eval_results_root: Path) -> dict:
    """Phase-0 marker panel-overlap gate (v6, the FIRST substrate deliverable, §7).

    Computes ``|EVAL_PERSONAS_24 ∩ #605-marker-panel|`` from the REAL #605 marker
    substrate (the personas that actually resolve onto the 24-persona axis after the
    context-key join), NOT just the static alias map — a #605 panel that drops a
    persona at load time must not inflate the overlap. Returns a logged,
    non-silent primary-set-membership decision:

      {marker_panel_overlap, overlap_personas, min_overlap, marker_leakage_excluded,
       decision}

    ``marker_leakage_excluded=True`` (overlap < MARKER_MIN_OVERLAP) demotes marker to
    DV-(a)-only and shrinks the primary H3 set to {sycophancy, refusal}. Fail-soft on
    a missing #605 substrate (returns overlap 0 + excluded=True with a note) — the
    caller logs it; a missing marker panel is a graceful degradation, not a crash of
    the whole bake-off (the #518 arms are independent).
    """
    try:
        table = load_marker_cell_substrate(eval_results_root)
        overlap_personas = sorted(table.bystanders)
        note = table.provenance.get("note", "")
    except FileNotFoundError as e:
        overlap_personas = []
        note = f"#605 marker substrate not found: {e}"
    overlap = len(overlap_personas)
    excluded = overlap < MARKER_MIN_OVERLAP
    return {
        "marker_panel_overlap": overlap,
        "overlap_personas": overlap_personas,
        "min_overlap": MARKER_MIN_OVERLAP,
        "marker_leakage_excluded": excluded,
        "decision": (
            "marker DV-(a)-only (overlap < min); primary H3 = {sycophancy, refusal}"
            if excluded
            else "marker IN the leakage bake-off + primary H3 set (overlap >= min)"
        ),
        "note": note,
    }


def effective_primary_h3(
    marker_direction_kind: str | None = None,
    marker_leakage_excluded: bool = False,
) -> frozenset[str]:
    """The RUN-CONDITIONAL primary H3 behavior set (v6, M-Alts2 + M-Alts3).

    The static ``PRIMARY_H3_BEHAVIORS`` = {sycophancy, refusal, marker} is the FULL
    set; marker is in it ONLY when its direction is the pure #623 affordance read
    that passed K2. Marker is demoted to SECONDARY (removed from the primary set)
    when EITHER:

      - ``marker_direction_kind == "shift_fallback"`` (the #521 on_policy_marker
        trained-shift fallback — a trained-shift read that folds the implant in,
        M-Alts3, mirroring the EM trained-shift demotion M-Alts2), OR
      - ``marker_leakage_excluded`` is True (the Phase-0 panel-overlap < 8 gate makes
        marker DV-(a)-only, §7).

    EM is NEVER in the primary set (M-Alts2). ``marker_direction_kind="affordance"``
    (or None / unknown, treated as the affordance default) keeps marker primary
    unless the overlap gate excluded it. Returns the effective frozenset.
    """
    primary = set(PRIMARY_H3_BEHAVIORS)
    demote_marker = (marker_direction_kind == "shift_fallback") or bool(marker_leakage_excluded)
    if demote_marker:
        primary.discard("marker")
    return frozenset(primary)
