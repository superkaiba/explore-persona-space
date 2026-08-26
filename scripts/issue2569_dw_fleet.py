"""Issue #2569 leg 5 — LoRA / full-FT weight-update (dW) geometry across the organism fleet.

Per checkpoint in the fleet (arms.json @ ``3bb20debe2`` on the HF data repo + any extra
manifests passed via ``--extra-arms-json``, enumerated + persisted BEFORE any download):

- **dW spectra + effective rank:** LoRA dW = B @ A * s per (layer, module), with s = the
  adapter's OWN scaling from ``adapter_config.json`` (alpha/sqrt(r) under rsLoRA, alpha/r
  classic — artifact-reuse check (g)). LoRA spectra + top vectors are EXACT from the
  r x r core after QR of the factors (``lora_svd_factors``, O(d*r^2) per matrix) — never
  a dense d_out x d_in SVD of a rank-32 product (fix-round-2 blocker
  ``dwfleet-lora-dense-svd-vs-rank32``: the dense path measured ~25-45 s/matrix at
  (3584, 18944) vs milliseconds factored, 20-30 h vs the 8 h cap). Full-FT dW =
  theta_post - theta_base per DECODER weight matrix, streamed one tensor at a time
  (28 layers x 7 modules = 196/ckpt, the plan §9 arithmetic; ``embed_tokens`` /
  ``lm_head`` are vocab-space (152k x 3584) and deliberately excluded — not module
  geometry, and never priced). Summaries per matrix: stable rank ||dW||_F^2/||dW||_2^2,
  participation ratio (sum s_i^2)^2 / sum s_i^4, top-1 share — descriptive, never gates.
- **Intruder read (#650 convention, load-enforced):** per module, the top dW singular
  vector ON THE MODULE'S RESIDUAL SIDE (``RESIDUAL_SIDE_BY_MODULE``: U for
  o_proj/down_proj = "write" arm, V for q/k/v/gate/up = "read" arm) vs the base weight
  SVD bases built by ``issue650_analyze.py build-base-svd --modules <all 7>``; the
  extended payload is REQUIRED — a payload lacking a module raises instead of silently
  dropping the read (fix-round-2 blocker ``dwfleet-oproj-intruder-silently-dropped``).
  Aggregation is EXACTLY ``max_over_base_singular_vectors_then_max_over_band`` via
  ``dv3_max_matched_null``; the nested #650 §6.5 payload is validated by
  ``assert_dv3_schema`` at write AND load, and the
  ``assertions.null_aggregation_matches_observed`` flag is COMPUTED (observed + null
  band reductions recomputed and compared at 1e-6), never asserted as a literal
  (fix-round-2 blocker ``dwfleet-null-assertion-vacuous``). Full-FT vectors come from
  truncated ``torch.svd_lowrank`` (q<=64, plan §4 leg-5 step 4) and the same reads run
  per full-FT checkpoint (fix-round-2 blocker ``dwfleet-fullft-analysis-missing``).
- **Factor alignment (--phase align):** top-8 dW factors per module vs banked directions,
  ALL staged at PINNED revisions through the returned-path helper
  (a local-dir hub download preserves the repo-relative path — probed 2026-08-25; hand
  deriving ``dl_root/<suffix>`` is the fix-round-2 ``dwfleet-delta-tbar-silent-absence``
  trap): delta = ``delta_tf/<arm>/tbar.pt`` @ ``c07267285d`` (probed payload:
  ``{"tbar"/"tbar_even"/"tbar_odd": {14|19|25: (3584,)}, "n_rows": 20, "meta"}`` — a
  20-TRAINING-ROW displacement mean, issue1434 positions, NOT a 16,400-row corpus mean;
  n_rows + the even/odd split-half cosine ride each arm record as the free within-delta
  noise floor); r_B = ``issue779_monitoring/r_b/<trait>.pt`` @ ``037fcbb2`` (probed:
  ``{"trait", "r_b": (28, 3584), "layers": [0..27], ...}``); c_C = the arm's
  training-context centroid ``A_ctx[layer]`` in ``issue1900_leakrace/anchors/<arm>.pt``
  @ ``b5acdabc79`` (probed; the earlier ``c_C``/``centroid`` key guess never matched —
  fix-round-2 ``dwfleet-anchor-payload-schema-unprobed``); gate direction A r computed
  locally via ``issue2569_operator`` (B1). OUTPUT-side U factors (o/down,
  residual-output space) read ALL directions; INPUT-side V factors (q/k/v/gate/up,
  residual-input space) read the context-space direction c_C. o_proj's INPUT side is the
  head-concat basis — dim 3584 but NOT the residual stream — and is never aligned
  (fix-round-2 blocker ``dwfleet-cc-alignment-mismatched-basis``). A per-arm banked file
  missing AT ITS PIN (probed: the 4 full-FT arms have no tbar/anchor) is recorded
  EXPLICITLY in the arm record and the coverage block; all LoRA arms missing the primary
  delta raises. Per-arm checkpointed under ``align/`` with a machine-stable resume key.
  Seed-noise anchor: the #1979 impoliteness-contrastive seed pair (s42 vs s137) gives
  the LoRA within-recipe dW-similarity floor; no full-FT seed pair exists (scope limit).
- **Sizing gate (§7 row 2):** ``--phase pilot`` is PINNED to one FULL-FT checkpoint
  (plan §4 leg-5 step 6 + blind-spot (2): the pilot must certify the
  private-overflow-repo full-FT download path, which the smoke cannot; fix-round-2
  blocker ``dwfleet-pilot-not-fullft-plan-adherence``). It stages ft[0] (TIMED —
  checkpoint IO is part of the basis), runs the PRODUCTION analysis function on TWO
  production-shape modules (down_proj MLP-wide + q_proj attention-square), stages +
  analyzes one LoRA arm through the production exact-rank-r path, and extrapolates the
  battery wall. A projection > ``--pilot-wall-cap-h`` (default 8 h) writes ``pilot.json``
  and exits rc=7 (a DESIGNED artifact-routed halt) so the dispatcher splits the fleet
  across 2 pods by checkpoint.

Phases (``--phase``): ``fleet`` (enumerate + persist the realized fleet table), ``pilot``,
``lora`` (all LoRA arms), ``ft`` (full-FT checkpoints), ``align`` (factor alignments).
Checkpoint-per-unit: each (arm) writes its JSON the moment it completes, with a
machine-stable resume key from generating parameters (incl. the base-svd payload's
``_meta`` — content-describing, so a rebuilt payload never resume-skips stale units).
``torch.load(weights_only=False)`` is only used for the self-produced, revision-pinned
tbar / anchor / r_B payloads (#1900 precedent); checkpoints are safetensors-only.
"""

from __future__ import annotations

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()  # BEFORE numpy/torch: shared-VM thread caps freeze at import (#847/#891)

import argparse
import dataclasses
import hashlib
import json
import logging
import math
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

from explore_persona_space.atomic_io import atomic_replace  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("issue2569_dw_fleet")

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

ISSUE = 2569
SEED = 2569
DATA_REPO = "superkaiba1/explore-persona-space-data"
ARMS_JSON_PATH = "issue1900_leakrace/config/arms.json"
ARMS_JSON_REV = "3bb20debe2"
ANCHORS_PREFIX = "issue1900_leakrace/anchors"
# Data-repo main resolved at probe time (2026-08-25, this fix round): no earlier pin was
# registered for the anchors prefix; the payload schema below was probed AT this revision.
ANCHORS_REV = "b5acdabc791c6991491b085404c959510b6c2c5a"
DELTA_TF_PREFIX = "issue1768_mapshift/delta_tf"
DELTA_TF_REV = "c07267285d"
RB_PREFIX = "issue779_monitoring/r_b"
RB_REV = "037fcbb2"
RB_TRAITS = ("evil", "hallucination", "sycophancy")

# The 7 LoRA target modules (r32/alpha64 rsLoRA fleet) — also the full-FT decoder module set.
LORA_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
# Residual-OUTPUT-side modules: their dW LEFT singular vectors live in residual space.
OUTPUT_SIDE_MODULES = ("o_proj", "down_proj")
TOP_K_FACTORS = 8
DV3_NULL_DRAWS = 200
DV3_NULL_AGGREGATION = "max_over_base_singular_vectors_then_max_over_band"
PILOT_WALL_CAP_H = 8.0
RC_PILOT_REFUSAL = 7  # the #1415 artifact-routed halt convention
FT_MATRICES_PER_CKPT = 196  # 28 layers x 7 decoder modules (plan §9 P-C row)
LOWRANK_Q = 64  # plan §4 leg-5 step 4: top-64 truncated SVD where vectors are needed
LOWRANK_NITER = 8

_ADAPTER_KEY_RE = re.compile(
    r"base_model\.model\.model\.layers\.(\d+)\.(self_attn|mlp)\.(\w+)\.lora_(A|B)\.weight"
)
_FT_KEY_RE = re.compile(
    r"model\.layers\.(\d+)\.(self_attn|mlp)\."
    r"(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)\.weight$"
)


# ──────────────────────────────────────────────────────────────────────────
# dW construction
# ──────────────────────────────────────────────────────────────────────────


def lora_scaling(adapter_config: dict) -> float:
    """Adapter's own application scaling: alpha/sqrt(r) under rsLoRA, alpha/r classic.

    Reads ``lora_alpha`` / ``r`` / ``use_rslora`` from the PARSED adapter_config.json dict
    (artifact-reuse check (g): the scaling regime comes from the artifact, never assumed).
    """
    alpha = float(adapter_config["lora_alpha"])
    r = float(adapter_config["r"])
    if r <= 0:
        raise ValueError(f"adapter r must be positive, got {r}")
    return alpha / math.sqrt(r) if adapter_config.get("use_rslora", False) else alpha / r


def delta_w_from_lora(a: torch.Tensor, b: torch.Tensor, scaling: float) -> torch.Tensor:
    """dW = B @ A * s in fp32; A is (r, d_in), B is (d_out, r). DENSE reference only.

    Production paths never materialize dW for spectra/vectors — they use
    ``lora_svd_factors`` (exact at O(d*r^2)); this stays as the equivalence-test oracle.
    """
    assert a.shape[0] == b.shape[1], (a.shape, b.shape)
    return (b.to(torch.float32) @ a.to(torch.float32)) * float(scaling)


def lora_svd_factors(
    a: torch.Tensor, b: torch.Tensor, scaling: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """EXACT SVD of dW = B @ A * s from the r x r core after QR — never a dense SVD.

    ``a`` is (..., r, d_in), ``b`` is (..., d_out, r); batched over leading dims. dW is
    EXACTLY rank <= r, so with B = Q_B R_B and A^T = Q_A R_A the singular triplets of dW
    are those of the r x r core C = R_B R_A^T s: dW = (Q_B U_c) diag(S) (Q_A V_c)^T.
    Returns (U, S, Vh) shaped like ``torch.linalg.svd(dW, full_matrices=False)`` truncated
    to r: U (..., d_out, r), S (..., r), Vh (..., r, d_in). This is exact (not an
    approximation) at O(d*r^2) per matrix vs the dense O(d_out*d_in*min) path — the
    fix-round-2 ``dwfleet-lora-dense-svd-vs-rank32`` blocker; dense agreement is pinned
    by a committed regression test.
    """
    a32 = a.to(torch.float32)
    b32 = b.to(torch.float32)
    assert a32.shape[-2] == b32.shape[-1], (a32.shape, b32.shape)
    qb, rb = torch.linalg.qr(b32, mode="reduced")  # (..., d_out, r), (..., r, r)
    qa, ra = torch.linalg.qr(a32.transpose(-2, -1), mode="reduced")  # (..., d_in, r), (..., r, r)
    core = (rb @ ra.transpose(-2, -1)) * float(scaling)  # (..., r, r)
    uc, s, vch = torch.linalg.svd(core, full_matrices=False)
    u = qb @ uc  # (..., d_out, r)
    vh = vch @ qa.transpose(-2, -1)  # (..., r, d_in)
    return u, s, vh


def load_adapter_factors(
    adapter_dir: Path,
) -> tuple[dict[tuple[int, str], dict[str, torch.Tensor]], float]:
    """Load per-(layer, module) LoRA factor pairs {"A": (r, d_in), "B": (d_out, r)} + scaling.

    Factors stay UN-materialized: every production consumer goes through
    ``lora_svd_factors``; ``load_adapter_deltas`` below is the dense equivalence
    reference (tests only). Incomplete pairs / empty adapters fail loud.
    """
    from safetensors.torch import load_file

    cfg = json.loads((adapter_dir / "adapter_config.json").read_text())
    s = lora_scaling(cfg)
    tensors = load_file(str(adapter_dir / "adapter_model.safetensors"))
    pairs: dict[tuple[int, str], dict[str, torch.Tensor]] = {}
    for key, t in tensors.items():
        m = _ADAPTER_KEY_RE.match(key)
        if not m:
            continue
        layer, _, module, ab = int(m.group(1)), m.group(2), m.group(3), m.group(4)
        pairs.setdefault((layer, module), {})[ab] = t
    for (layer, module), ab in sorted(pairs.items()):
        if "A" not in ab or "B" not in ab:
            raise RuntimeError(f"incomplete LoRA pair at layer {layer} module {module}")
    if not pairs:
        raise RuntimeError(f"no LoRA weight pairs found under {adapter_dir}")
    return pairs, s


def load_adapter_deltas(adapter_dir: Path) -> dict[tuple[int, str], torch.Tensor]:
    """DENSE reference: materialized dW = B A s per (layer, module) — equivalence tests ONLY.

    NOT on the production path (production consumes ``load_adapter_factors`` +
    ``lora_svd_factors``): materializing dW invites the dense-SVD shape the round-2
    blocker ``dwfleet-lora-dense-svd-vs-rank32`` priced at 20-30 h against the 8 h cap.
    """
    pairs, s = load_adapter_factors(adapter_dir)
    return {
        (layer, module): delta_w_from_lora(ab["A"], ab["B"], s)
        for (layer, module), ab in sorted(pairs.items())
    }


def iter_ft_deltas(base_dir: Path, post_dir: Path, name_filter=None):
    """Yield (param_name, dW fp32) for 2-D weights, one tensor at a time (stream-reduce).

    Both directories are HF-format checkpoints (safetensors shards + index). Peak RSS stays
    O(one tensor) — never the whole 15 GB delta (earlyoom stream-reduce rule, #658).
    ``name_filter`` (predicate on the param name) is applied BEFORE any tensor read so a
    filtered pilot never pays IO for excluded matrices.
    """
    from safetensors import safe_open

    def _shard_map(d: Path) -> dict[str, Path]:
        idx = d / "model.safetensors.index.json"
        if idx.is_file():
            wmap = json.loads(idx.read_text())["weight_map"]
            return {k: d / v for k, v in wmap.items()}
        single = d / "model.safetensors"
        if not single.is_file():
            raise FileNotFoundError(f"no safetensors checkpoint under {d}")
        with safe_open(str(single), framework="pt") as f:
            return {k: single for k in f.keys()}

    base_map = _shard_map(base_dir)
    post_map = _shard_map(post_dir)
    missing = sorted(set(base_map) ^ set(post_map))
    if missing:
        raise RuntimeError(
            f"base/post param sets differ ({len(missing)} names; e.g. {missing[:4]})"
        )

    open_handles: dict[Path, object] = {}

    def _get(mapping: dict[str, Path], name: str) -> torch.Tensor:
        path = mapping[name]
        if path not in open_handles:
            open_handles[path] = safe_open(str(path), framework="pt")
        return open_handles[path].get_tensor(name)  # type: ignore[union-attr]

    for name in sorted(base_map):
        if name_filter is not None and not name_filter(name):
            continue
        w_base = _get(base_map, name)
        if w_base.ndim != 2:
            continue
        w_post = _get(post_map, name)
        yield name, (w_post.to(torch.float32) - w_base.to(torch.float32))


def _ft_name_parts(name: str) -> tuple[int, str] | None:
    """(layer, module) for a decoder weight-matrix param name, else None."""
    m = _FT_KEY_RE.match(name)
    return (int(m.group(1)), m.group(3)) if m else None


# ──────────────────────────────────────────────────────────────────────────
# Spectral summaries + intruder read
# ──────────────────────────────────────────────────────────────────────────


def svdvals_robust(w: torch.Tensor) -> np.ndarray:
    """Singular values with a fp64 retry on LAPACK non-convergence (never silently skipped)."""
    try:
        s = torch.linalg.svdvals(w)
    except torch.linalg.LinAlgError:
        log.warning("[dw] svdvals non-convergence at fp32 — fp64 retry")
        s = torch.linalg.svdvals(w.to(torch.float64))
    return s.to(torch.float64).numpy()


def svdvals_stack(stack: torch.Tensor) -> np.ndarray:
    """Batched singular values of a (B, m, n) stack — DENSE reference (tests only).

    NOT on the production LoRA path (which is ``lora_svd_factors``, exact at rank r);
    kept as the batched dense oracle for the equivalence tests. A single bad slice
    triggers the per-matrix path via ``svdvals_robust`` (gotchas: batched solves raise
    ONE error for the whole stack).
    """
    try:
        return torch.linalg.svdvals(stack).to(torch.float64).numpy()
    except torch.linalg.LinAlgError:
        log.warning("[dw] batched svdvals non-convergence — per-slice fallback")
        return np.stack([svdvals_robust(stack[i]) for i in range(stack.shape[0])])


def effective_rank_summaries(svals: np.ndarray) -> dict:
    """Descriptive spectral summaries (never gates): stable rank, PR, top-1 shares.

    ``n_svals`` is the count of COMPUTED singular values: min(m, n) on the dense full-FT
    path, exactly r on the factored LoRA path (trailing exact zeros of a rank-r product
    carry no mass and are not materialized).
    """
    s = np.asarray(svals, dtype=np.float64)
    s2 = s**2
    tot2 = float(s2.sum())
    tot = float(s.sum())
    if tot2 <= 0:
        raise ValueError("all-zero singular values — empty dW")
    return {
        "stable_rank": tot2 / float(s2.max()),
        "participation_ratio": tot2**2 / float((s2**2).sum()),
        "top1_share_energy": float(s2.max()) / tot2,
        "top1_share_sv": float(s.max()) / tot,
        "frobenius": math.sqrt(tot2),
        "spectral": float(s.max()),
        "n_svals": int(s.size),
    }


def _issue650():
    """Import the shared #650 module (scripts/ is on sys.path via module top)."""
    import issue650_analyze as i650  # noqa: PLC0415

    return i650


def dv3_payload_from_null(arm_results: dict[str, dict]) -> dict:
    """Wrap per-arm ``dv3_max_matched_null`` results in the registered nested #650 schema.

    ``arm_results`` maps arm name ("write" / "read") -> the flat dv3_max_matched_null
    output. The ``assertions.null_aggregation_matches_observed`` flag is COMPUTED, never
    a literal (fix-round-2 blocker ``dwfleet-null-assertion-vacuous``): per arm, the
    observed band max is recomputed from the per-layer observed maxima, every null
    draw's band max is recomputed from the per-layer null draws, and the band p95 is
    recomputed from the recomputed draws — each compared numerically at 1e-6. Any
    mismatch raises BEFORE anything is persisted; the returned payload additionally
    passes ``issue650_analyze.assert_dv3_schema``.
    """
    tol = 1e-6
    observed: dict = {}
    null: dict = {}
    for arm, res in arm_results.items():
        if res["null_aggregation"] != DV3_NULL_AGGREGATION:
            raise AssertionError(
                f"dv3 {arm}: aggregation {res['null_aggregation']!r} != {DV3_NULL_AGGREGATION!r}"
            )
        # COMPUTED symmetry check: observed and null must reduce by the IDENTICAL
        # per-layer-max -> band-max chain. Recompute both sides and compare.
        obs_band_re = float(max(res["per_layer_observed_max"].values()))
        if abs(obs_band_re - float(res["band_observed_max"])) > tol:
            raise AssertionError(
                f"dv3 {arm}: band_observed_max {res['band_observed_max']!r} != recomputed "
                f"per-layer band max {obs_band_re!r} — observed aggregation is NOT the "
                "registered band-max"
            )
        per_layer_null = res["per_layer_null_max_draws"]
        null_band = np.asarray(res["band_null_max_draws"], dtype=np.float64)
        null_band_re = np.max(
            np.stack(
                [np.asarray(per_layer_null[layer], dtype=np.float64) for layer in per_layer_null]
            ),
            axis=0,
        )
        if null_band.shape != null_band_re.shape or not np.allclose(
            null_band, null_band_re, atol=tol, rtol=0
        ):
            raise AssertionError(
                f"dv3 {arm}: band_null_max_draws does not equal the per-layer-max band "
                "reduction of per_layer_null_max_draws — null aggregation is NOT the "
                "registered band-max"
            )
        p95_re = float(np.percentile(null_band_re, 95.0))
        if abs(p95_re - float(res["band_null_p95"])) > tol:
            raise AssertionError(
                f"dv3 {arm}: band_null_p95 {res['band_null_p95']!r} != recomputed p95 "
                f"{p95_re!r} over the band-max null draws"
            )
        observed[arm] = {
            "max_by_layer": res["per_layer_observed_max"],
            "band_max": res["band_observed_max"],
            "verdict": res["verdict"],
        }
        null[arm] = {
            "per_layer_max_draws": res["per_layer_null_max_draws"],
            "band_max_draws": res["band_null_max_draws"],
            "band_p95": res["band_null_p95"],
            "n_draws": res["n_draws"],
            "null_aggregation": res["null_aggregation"],
        }
    payload = {
        "observed": observed,
        "null": null,
        # True is EARNED here: every arm above passed the numeric recompute-and-compare.
        "assertions": {"null_aggregation_matches_observed": True},
    }
    _issue650().assert_dv3_schema(payload)
    return payload


def intruder_read(
    dw_top_vecs: dict[int, np.ndarray],
    base_basis_by_layer: dict[int, np.ndarray],
    *,
    arm_name: str,
    n_draws: int = DV3_NULL_DRAWS,
    seed: int = SEED,
) -> dict:
    """#650-convention intruder read for one module side across a layer band.

    ``dw_top_vecs``: per layer, the TOP dW singular vector (1-D, len d) on the side whose
    base basis is supplied. Aggregation is EXACTLY max-over-base-singular-vectors then
    max-over-band via ``dv3_max_matched_null`` per observed vector; per-layer results are
    combined by the same band-max the null uses (verified numerically at payload build).
    """
    i650 = _issue650()
    band = tuple(sorted(dw_top_vecs))
    # dv3_max_matched_null takes ONE observed vector over a band; our observed vector
    # varies per layer, so run per layer with band=(layer,) then band-max both sides.
    per_layer_obs: dict[int, float] = {}
    per_layer_null: dict[int, list[float]] = {}
    for layer in band:
        res = i650.dv3_max_matched_null(
            observed_vec=np.asarray(dw_top_vecs[layer], dtype=np.float32),
            basis_by_layer={layer: base_basis_by_layer[layer]},
            band=(layer,),
            n_draws=n_draws,
            seed=seed + layer,
        )
        per_layer_obs[layer] = res["per_layer_observed_max"][layer]
        per_layer_null[layer] = res["per_layer_null_max_draws"][layer]
    band_obs = float(max(per_layer_obs.values()))
    null_band = [float(max(vals)) for vals in zip(*[per_layer_null[layer] for layer in band])]
    p95 = float(np.percentile(np.asarray(null_band), 95.0))
    flat = {
        "per_layer_observed_max": {int(k): float(v) for k, v in per_layer_obs.items()},
        "band_observed_max": band_obs,
        "per_layer_null_max_draws": {int(k): v for k, v in per_layer_null.items()},
        "band_null_max_draws": null_band,
        "band_null_p95": p95,
        "n_draws": int(n_draws),
        "null_aggregation": DV3_NULL_AGGREGATION,
        "verdict": (
            "pre_existing_in_base_column_space"
            if band_obs > p95
            else "intruder_at_max_matched_null"
        ),
    }
    return dv3_payload_from_null({arm_name: flat})


def alignment_vs_null(
    factor_stack: np.ndarray,
    direction: np.ndarray,
    *,
    n_draws: int = DV3_NULL_DRAWS,
    seed: int = SEED,
) -> dict:
    """max |cos| of a fixed direction over the top-K dW factor stack, vs the matched null.

    ``factor_stack`` is (K, d) unit rows (the dW top-K singular vectors on one side);
    the null draws random unit vectors through the IDENTICAL max — the same max-matched
    order-statistic logic as the intruder read, on a K-vector basis.
    """
    i650 = _issue650()
    res = i650.dv3_max_matched_null(
        observed_vec=np.asarray(direction, dtype=np.float32),
        basis_by_layer={0: np.asarray(factor_stack, dtype=np.float32)},
        band=(0,),
        n_draws=n_draws,
        seed=seed,
    )
    return {
        "max_abs_cos": res["band_observed_max"],
        "null_p95": res["band_null_p95"],
        "n_draws": res["n_draws"],
        "above_null": bool(res["band_observed_max"] > res["band_null_p95"]),
        "null_aggregation": res["null_aggregation"],
        "k_basis": int(np.asarray(factor_stack).shape[0]),
    }


# ──────────────────────────────────────────────────────────────────────────
# Fleet enumeration + phases
# ──────────────────────────────────────────────────────────────────────────


@dataclasses.dataclass(frozen=True)
class FleetEntry:
    """One checkpoint of the realized fleet."""

    arm_id: str
    kind: str
    beh_key: str
    method: str
    repo_id: str
    subfolder: str
    source_manifest: str


def enumerate_fleet(arms_json: dict, extra_arms: list[dict] | None = None) -> list[FleetEntry]:
    """Build the realized fleet table from arms.json (+ extra manifests), no downloads.

    LoRA arms resolve (adapter_repo, adapter_subfolder); full-FT arms resolve
    (ft_repo, ft_subfolder). A record missing its checkpoint fields fails loud.
    """
    out: list[FleetEntry] = []
    for a in arms_json["arms"]:
        if a["method"] == "lora":
            repo, sub = a.get("adapter_repo"), a.get("adapter_subfolder")
        elif a["method"] == "ft":
            repo, sub = a.get("ft_repo"), a.get("ft_subfolder")
        else:
            raise ValueError(f"unknown method {a['method']!r} for arm {a['arm_id']}")
        if not repo or not sub:
            raise RuntimeError(
                f"arm {a['arm_id']} missing checkpoint fields (method={a['method']})"
            )
        out.append(
            FleetEntry(a["arm_id"], a["kind"], a["beh_key"], a["method"], repo, sub, "arms.json")
        )
    for rec in extra_arms or []:
        out.append(
            FleetEntry(
                rec["arm_id"],
                rec.get("kind", "content"),
                rec.get("beh_key", "?"),
                rec["method"],
                rec["repo_id"],
                rec["subfolder"],
                rec.get("source_manifest", "extra"),
            )
        )
    if not out:
        raise RuntimeError("empty fleet enumeration")
    return out


def _atomic_json(path: Path, payload: dict) -> None:
    """Atomic JSON write through the shared process-unique atomic-replace primitive
    (#2336: a fixed ``.tmp`` sibling name is a concurrent-writer clobber)."""
    with atomic_replace(path) as tmp:
        tmp.write_text(json.dumps(payload, indent=1, sort_keys=True))


def _meta(phase: str) -> dict:
    """Reproducibility metadata block (git provenance + versions + timestamp)."""
    from explore_persona_space.orchestrate.provenance import as_metadata_dict, git_provenance

    md = as_metadata_dict(git_provenance(), phase=phase)
    md.update(
        {
            "issue": ISSUE,
            "torch": str(torch.__version__),
            "numpy": str(np.__version__),
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "arms_json_rev": ARMS_JSON_REV,
        }
    )
    return md


def regime_key(**params: object) -> str:
    """Machine-stable resume key from generating parameters (never float-array bytes)."""
    blob = json.dumps(params, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def _stage_adapter(entry: FleetEntry, dl_root: Path) -> Path:
    """Download one LoRA adapter (config + safetensors) from the model repo.

    Fetches ride ``hub.retry_transient`` (#920: huggingface_hub retries only
    429 natively — a transient 504 mid-stage kills the arm otherwise).
    """
    from huggingface_hub import hf_hub_download

    from explore_persona_space.orchestrate import hub

    dest = dl_root / entry.arm_id
    for fname in ("adapter_config.json", "adapter_model.safetensors"):
        hub.retry_transient(
            lambda f=fname: hf_hub_download(
                entry.repo_id, f"{entry.subfolder}/{f}", local_dir=dest, repo_type="model"
            ),
            what=f"adapter fetch ({entry.arm_id}/{fname})",
        )
    return dest / entry.subfolder


def _stage_checkpoint(repo_id: str, subfolder: str, dl_root: Path, tag: str) -> Path:
    """Download one full HF checkpoint's safetensors shards + index.

    The listing routes through ``hub.list_hf_files_under_path`` (one retried,
    server-side-scoped tree walk) and each shard fetch rides
    ``hub.retry_transient`` (#920: huggingface_hub retries only 429 natively).
    """
    from huggingface_hub import HfApi, hf_hub_download

    from explore_persona_space.orchestrate import hub

    api = HfApi()
    dest = dl_root / tag
    files = [
        p
        for p in hub.list_hf_files_under_path(api, repo_id, subfolder, repo_type="model")
        if p.endswith((".safetensors", ".safetensors.index.json"))
    ]
    if not files:
        raise RuntimeError(f"no safetensors under {repo_id}/{subfolder}")
    for f in files:
        hub.retry_transient(
            lambda f=f: hf_hub_download(repo_id, f, local_dir=dest, repo_type="model"),
            what=f"checkpoint shard fetch ({tag}/{f.rsplit('/', 1)[-1]})",
        )
    return dest / subfolder


def _stage_banked_file(path_in_repo: str, dl_root: Path, revision: str, *, what: str) -> Path:
    """Stage one banked file from the data repo at its PINNED revision; return the REAL path.

    A local-dir hub download PRESERVES ``path_in_repo`` under ``local_dir``
    (probed 2026-08-25), so consumers MUST use the returned path — hand-deriving
    ``dl_root/<suffix>`` is the fix-round-2 ``dwfleet-delta-tbar-silent-absence`` trap
    (every arm silently loses its primary delta alignment).
    """
    from huggingface_hub import hf_hub_download

    from explore_persona_space.orchestrate import hub

    p = hub.retry_transient(
        lambda: hf_hub_download(
            DATA_REPO, path_in_repo, repo_type="dataset", revision=revision, local_dir=dl_root
        ),
        what=what,
    )
    return Path(p)


def _stage_optional_banked_file(
    path_in_repo: str, dl_root: Path, revision: str, *, what: str
) -> Path | None:
    """Stage a PER-ARM banked file; a 404 AT THE PIN returns None for EXPLICIT recording.

    Probed 2026-08-25: the 4 full-FT arms have no tbar / anchor banked, so per-arm
    absence is a REAL state — the caller records it in the arm record + coverage block
    (never a bare skip). ``EntryNotFoundError`` is non-transient so ``retry_transient``
    re-raises it immediately; every OTHER failure propagates (fail-loud).
    """
    from huggingface_hub.errors import EntryNotFoundError

    try:
        return _stage_banked_file(path_in_repo, dl_root, revision, what=what)
    except EntryNotFoundError:
        return None


def _load_base_svd_required(path_str: str | None) -> dict:
    """Load the #650 base-SVD payload for ALL 7 LoRA target modules (fail-loud, required).

    The intruder read is a plan-registered leg-5 deliverable, so ``--base-svd`` is
    REQUIRED for the lora/ft/pilot phases; a payload built without a needed module (the
    fix-round-2 ``dwfleet-oproj-intruder-silently-dropped`` shape) raises with the
    rebuild command instead of silently skipping the module.
    """
    if not path_str:
        raise RuntimeError(
            "--base-svd is required (build it: issue650_analyze.py build-base-svd "
            f"--modules {','.join(LORA_MODULES)})"
        )
    p = Path(path_str)
    if not p.is_file():
        raise FileNotFoundError(f"--base-svd payload not found at {p}")
    try:
        return _issue650().load_base_svd(p, modules=LORA_MODULES)
    except KeyError as e:
        raise RuntimeError(
            f"base-svd payload at {p} lacks module {e} — rebuild with build-base-svd "
            f"--modules {','.join(LORA_MODULES)}"
        ) from e


def analyze_lora_arm(entry: FleetEntry, adapter_dir: Path, base_svd: dict) -> dict:
    """Spectra + effective rank + intruder read for one LoRA arm — EXACT rank-r path.

    Per module the layer-stacked factors go through ``lora_svd_factors`` (the r x r core
    after QR): singular values + top vectors are EXACT for dW = B A s at O(d*r^2) per
    matrix instead of a dense SVD (fix-round-2 ``dwfleet-lora-dense-svd-vs-rank32``).
    The intruder read runs per module on the module's RESIDUAL side (o/down: U "write";
    q/k/v/gate/up: V "read"); a base-svd payload lacking any adapter module or layer
    RAISES — no module is ever silently dropped (fix-round-2
    ``dwfleet-oproj-intruder-silently-dropped``).
    """
    i650 = _issue650()
    pairs, scaling = load_adapter_factors(adapter_dir)
    by_module: dict[str, dict[int, dict[str, torch.Tensor]]] = {}
    for (layer, module), ab in pairs.items():
        by_module.setdefault(module, {})[layer] = ab
    rec: dict = {
        "arm_id": entry.arm_id,
        "method": "lora",
        "modules": {},
        "intruder": {},
        "intruder_side": {},
    }
    for module, by_layer in sorted(by_module.items()):
        layers = sorted(by_layer)
        a_stack = torch.stack([by_layer[layer]["A"] for layer in layers])
        b_stack = torch.stack([by_layer[layer]["B"] for layer in layers])
        u, s, vh = lora_svd_factors(a_stack, b_stack, scaling)
        svals = s.to(torch.float64).numpy()  # (L, r) — the EXACT nonzero spectrum
        rec["modules"][module] = {
            str(layer): effective_rank_summaries(svals[i]) for i, layer in enumerate(layers)
        }
        if module not in i650.RESIDUAL_SIDE_BY_MODULE:
            raise RuntimeError(f"no residual-side convention for adapter module {module!r}")
        side = i650.RESIDUAL_SIDE_BY_MODULE[module]
        if module not in base_svd:
            raise RuntimeError(
                f"base-svd payload lacks module {module!r} — rebuild with build-base-svd "
                f"--modules {','.join(LORA_MODULES)}"
            )
        top = (u[:, :, 0] if side == "U" else vh[:, 0, :]).numpy()  # (L, d_side)
        vecs: dict[int, np.ndarray] = {}
        basis: dict[int, np.ndarray] = {}
        for i, layer in enumerate(layers):
            if layer not in base_svd[module]:
                raise RuntimeError(f"base-svd payload lacks layer {layer} for {module!r}")
            vecs[layer] = top[i]
            basis[layer] = base_svd[module][layer][side]
        rec["intruder"][module] = intruder_read(
            vecs, basis, arm_name="write" if side == "U" else "read"
        )
        rec["intruder_side"][module] = side
    return rec


def analyze_ft_checkpoint(
    entry: FleetEntry,
    base_dir: Path,
    post_dir: Path,
    base_svd: dict,
    *,
    align_layer: int,
    factors_path: Path | None = None,
    module_filter: tuple[str, ...] | None = None,
) -> dict:
    """Spectra + effective rank + intruder + align-layer factors for one full-FT checkpoint.

    Streams theta_post - theta_base one tensor at a time over the DECODER weight matrices
    (28 layers x 7 modules = 196/ckpt — the plan §9 arithmetic; ``embed_tokens`` /
    ``lm_head`` are vocab-space (152k x 3584), not module geometry, and each would add a
    ~150k-row SVD the plan never priced — excluded by construction). Vectors come from
    truncated ``torch.svd_lowrank`` (q<=64, plan §4 leg-5 step 4; seeded for resume
    stability). The intruder read runs per module on the module's RESIDUAL side, and the
    align-layer residual-side top-8 factors are persisted to ``factors_path`` for
    ``--phase align`` (fix-round-2 blocker ``dwfleet-fullft-analysis-missing``).
    """
    i650 = _issue650()
    rec: dict = {
        "arm_id": entry.arm_id,
        "method": "ft",
        "matrices": {},
        "intruder": {},
        "intruder_side": {},
    }
    torch.manual_seed(SEED)  # svd_lowrank is randomized — pin for resume stability
    top_vecs: dict[str, dict[int, np.ndarray]] = {}
    factors: dict[str, dict] = {}
    n = 0
    t0 = time.time()

    def _keep(name: str) -> bool:
        parts = _ft_name_parts(name)
        if parts is None:
            return False
        return module_filter is None or parts[1] in module_filter

    for name, dw in iter_ft_deltas(base_dir, post_dir, name_filter=_keep):
        layer_idx, module = _ft_name_parts(name)  # type: ignore[misc]
        svals = svdvals_robust(dw)
        rec["matrices"][name] = effective_rank_summaries(svals)
        side = i650.RESIDUAL_SIDE_BY_MODULE[module]
        q = min(LOWRANK_Q, min(dw.shape))
        u, s, v = torch.svd_lowrank(dw, q=q, niter=LOWRANK_NITER)
        side_vecs = (u.T if side == "U" else v.T).contiguous()  # (q, d_side) rows
        top_vecs.setdefault(module, {})[layer_idx] = side_vecs[0].numpy()
        if layer_idx == align_layer:
            kk = min(TOP_K_FACTORS, side_vecs.shape[0])
            factors[module] = {
                "side": side,
                "factors": side_vecs[:kk].clone(),
                "svals": s[:kk].clone(),
            }
        n += 1
        print(f"[dw-ft] unit {n} {entry.arm_id}/{name} elapsed={time.time() - t0:.0f}s", flush=True)
    if n == 0:
        raise RuntimeError(f"no decoder weight matrices found for {entry.arm_id}")
    rec["n_matrices"] = n
    rec["factor_method"] = f"svd_lowrank(q<={LOWRANK_Q}, niter={LOWRANK_NITER}, seed={SEED})"
    for module, vecs in sorted(top_vecs.items()):
        side = i650.RESIDUAL_SIDE_BY_MODULE[module]
        if module not in base_svd:
            raise RuntimeError(
                f"base-svd payload lacks module {module!r} — rebuild with build-base-svd "
                f"--modules {','.join(LORA_MODULES)}"
            )
        basis: dict[int, np.ndarray] = {}
        for layer_idx in sorted(vecs):
            if layer_idx not in base_svd[module]:
                raise RuntimeError(f"base-svd payload lacks layer {layer_idx} for {module!r}")
            basis[layer_idx] = base_svd[module][layer_idx][side]
        rec["intruder"][module] = intruder_read(
            vecs, basis, arm_name="write" if side == "U" else "read"
        )
        rec["intruder_side"][module] = side
    if factors_path is not None:
        if module_filter is None and set(factors) != set(LORA_MODULES):
            raise RuntimeError(
                f"align-layer {align_layer} factors incomplete for {entry.arm_id}: "
                f"got {sorted(factors)} — the checkpoint stream never saw that layer?"
            )
        factors_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "layer": int(align_layer),
                "arm_id": entry.arm_id,
                "method": rec["factor_method"],
                "modules": factors,
            },
            factors_path,
        )
        rec["factors_path"] = str(factors_path)
    return rec


# ──────────────────────────────────────────────────────────────────────────
# Banked-direction loaders (schemas PROBED on the real artifacts, 2026-08-25)
# ──────────────────────────────────────────────────────────────────────────


def _unit_vec(x: np.ndarray) -> np.ndarray:
    """L2-normalize a 1-D direction (fail-loud on zero norm)."""
    v = np.asarray(x, dtype=np.float64).ravel()
    n = float(np.linalg.norm(v))
    if n == 0:
        raise ValueError("zero-norm direction")
    return v / n


def load_rb_direction(path: Path, layer: int) -> np.ndarray:
    """r_B at one layer from the #779 payload.

    Probed @ ``037fcbb2``: ``{"trait": str, "r_b": (28, 3584) fp32, "layers": [0..27],
    "counts", "smoke", "metadata"}`` — a DICT with a stacked per-layer matrix, not a bare
    tensor (schema-from-artifact; the prior bare-tensor read raised TypeError on the real
    file). Fail-loud on an unexpected shape or an absent layer.
    """
    payload = torch.load(path, weights_only=False, map_location="cpu")
    if not isinstance(payload, dict) or "r_b" not in payload or "layers" not in payload:
        got = sorted(payload) if isinstance(payload, dict) else type(payload).__name__
        raise TypeError(f"unexpected r_B payload at {path}: {got}")
    layers = [int(x) for x in payload["layers"]]
    if layer not in layers:
        raise KeyError(f"layer {layer} not in r_B payload layers {layers} ({path})")
    row = payload["r_b"][layers.index(layer)]
    return _unit_vec(row.to(torch.float64).numpy())


def load_tbar_directions(path: Path, layer: int) -> tuple[dict[str, np.ndarray], dict]:
    """delta (tbar) + split halves at one layer from the #1768 payload.

    Probed @ ``c07267285d``: ``{"tbar"/"tbar_even"/"tbar_odd": {14|19|25: (3584,) fp32},
    "n_rows": 20, "meta": {...}}``. Returns ({delta_tbar, delta_tbar_even,
    delta_tbar_odd}, provenance). n_rows == 20: tbar is a 20-TRAINING-ROW displacement
    mean (issue1434 positions), NOT a 16,400-row corpus mean — recorded per arm, with the
    even/odd split-half cosine as the free within-delta noise floor (fix-round-2 concern
    ``leg5-delta-inherits-tbar-20row-basis``).
    """
    payload = torch.load(path, weights_only=False, map_location="cpu")
    for key in ("tbar", "tbar_even", "tbar_odd"):
        if key not in payload or layer not in payload[key]:
            have = sorted(payload.get("tbar", {})) if isinstance(payload, dict) else "?"
            raise KeyError(f"{key}[{layer}] missing in tbar payload {path} (layers={have})")
    even = payload["tbar_even"][layer].to(torch.float64).numpy()
    odd = payload["tbar_odd"][layer].to(torch.float64).numpy()
    dirs = {
        "delta_tbar": _unit_vec(payload["tbar"][layer].to(torch.float64).numpy()),
        "delta_tbar_even": _unit_vec(even),
        "delta_tbar_odd": _unit_vec(odd),
    }
    prov = {
        "n_rows": int(payload["n_rows"]),
        "splithalf_cos": float(np.dot(_unit_vec(even), _unit_vec(odd))),
        "basis_note": (
            "20-training-row displacement mean (issue1434 positions), not a corpus mean"
        ),
    }
    return dirs, prov


def load_anchor_cc(path: Path, layer: int) -> tuple[np.ndarray, dict]:
    """c_C = the arm's training-context centroid ``A_ctx[layer]`` from the #1979 anchor.

    Probed @ ``b5acdabc79``: ``{"mix_arm_id", "n_rows": 20, "low_n_flag", "mix_meta",
    "A_ctx"/"A_ans"/(+ _even/_odd): {14|19|25: (3584,)}, "split_half_cos_ctx"/"..._ans",
    "rows_ctx": (20, 3584), "tbar_cos", "meta"}``. The earlier ``c_C``/``centroid`` key
    guess NEVER matched this schema — c_C silently vanished for every arm (fix-round-2
    ``dwfleet-anchor-payload-schema-unprobed``). Fail-loud on schema/layer misses.
    """
    payload = torch.load(path, weights_only=False, map_location="cpu")
    if not isinstance(payload, dict) or "A_ctx" not in payload:
        got = sorted(payload) if isinstance(payload, dict) else type(payload).__name__
        raise TypeError(f"unexpected anchor payload at {path}: {got}")
    a_ctx = payload["A_ctx"]
    if layer not in a_ctx:
        raise KeyError(f"A_ctx[{layer}] missing in anchor payload {path} (layers={sorted(a_ctx)})")
    shc = payload.get("split_half_cos_ctx", {})
    prov = {
        "n_rows": int(payload.get("n_rows", -1)),
        "low_n_flag": bool(payload.get("low_n_flag", False)),
        "splithalf_cos_ctx": float(shc[layer]) if layer in shc else None,
    }
    return _unit_vec(a_ctx[layer].to(torch.float64).numpy()), prov


# ──────────────────────────────────────────────────────────────────────────
# Phases
# ──────────────────────────────────────────────────────────────────────────


def cmd_align(args) -> int:
    """Factor-alignment phase: top-8 dW factors vs delta / r_B / c_C / A r (+ seed anchor).

    Runs OP.run_driver_identity_asserts at entry (the operator-consuming path — B1).
    Basis routing (fix-round-2 ``dwfleet-cc-alignment-mismatched-basis``): residual-OUTPUT
    U factors (o_proj/down_proj) read ALL directions; residual-INPUT V factors
    (q/k/v/gate/up) read the context-space direction c_C. o_proj's input side (head-concat
    basis) and down_proj's input side (ffn basis) are STRUCTURALLY never aligned — a dim
    coincidence never licenses a read. Per-arm checkpointed under ``align/``; full-FT arms
    consume the factor sidecars ``--phase ft`` persisted.
    """
    import issue2569_operator as op

    dl_root = Path(args.dl_root)
    out_root = Path(args.out_root)
    layer = int(args.align_layer)

    payload = op.load_banked_map(layer=layer, root=args.map_root or None)
    op.run_driver_identity_asserts(payload)
    a_mat, _b = op.row_operator(payload)

    fleet = _load_fleet_table(out_root)
    if args.arms:
        want = {a.strip() for a in args.arms.split(",")}
        fleet = [e for e in fleet if e.arm_id in want]
        if not fleet:
            raise RuntimeError(f"empty fleet after --arms filter: {sorted(want)}")

    # Global directions — staged at their PINS, fail-loud (the pins are LIVE here).
    directions: dict[str, np.ndarray] = {}
    for trait in RB_TRAITS:
        rb_path = _stage_banked_file(
            f"{RB_PREFIX}/{trait}.pt", dl_root, RB_REV, what=f"r_B fetch ({trait}@{RB_REV})"
        )
        rb = load_rb_direction(rb_path, layer)
        directions[f"r_B[{trait}]"] = rb
        directions[f"Ar[{trait}]"] = _unit_vec(op.monitor_gradient(a_mat, rb))

    align_dir = out_root / "dw_fleet" / "align"
    rk = regime_key(
        phase="align",
        layer=layer,
        top_k=TOP_K_FACTORS,
        dv3_draws=DV3_NULL_DRAWS,
        arms_rev=ARMS_JSON_REV,
        delta_tf_rev=DELTA_TF_REV,
        rb_rev=RB_REV,
        anchors_rev=ANCHORS_REV,
    )
    results: dict[str, dict] = {}
    t0 = time.time()
    for k, entry in enumerate(fleet, start=1):
        unit_path = align_dir / f"{entry.arm_id}.json"
        if not args.no_resume and unit_path.is_file():
            try:
                prior = json.loads(unit_path.read_text())
            except json.JSONDecodeError:
                prior = None
            if prior is not None and prior.get("regime_key") == rk:
                results[entry.arm_id] = prior
                print(f"[dw-align] unit {k}/{len(fleet)} {entry.arm_id} resume-skip", flush=True)
                continue

        arm_rec: dict = {
            "arm_id": entry.arm_id,
            "method": entry.method,
            "factors": {},
            "directions_provenance": {},
        }
        arm_dirs = dict(directions)

        # delta (tbar) — the H5 PRIMARY direction; staged at the pin, absence EXPLICIT.
        tbar_repo_path = f"{DELTA_TF_PREFIX}/{entry.arm_id}/tbar.pt"
        tbar_path = _stage_optional_banked_file(
            tbar_repo_path, dl_root, DELTA_TF_REV, what=f"tbar fetch ({entry.arm_id})"
        )
        if tbar_path is None:
            arm_rec["directions_provenance"]["delta_tbar"] = {
                "missing": f"no {tbar_repo_path} at {DELTA_TF_REV}"
            }
        else:
            tbar_dirs, tbar_prov = load_tbar_directions(tbar_path, layer)
            arm_dirs.update(tbar_dirs)
            arm_rec["directions_provenance"]["delta_tbar"] = tbar_prov

        # c_C — the arm's training-context centroid from the #1979 banked anchors.
        anchor_repo_path = f"{ANCHORS_PREFIX}/{entry.arm_id}.pt"
        anchor_path = _stage_optional_banked_file(
            anchor_repo_path, dl_root, ANCHORS_REV, what=f"anchor fetch ({entry.arm_id})"
        )
        if anchor_path is None:
            arm_rec["directions_provenance"]["c_C"] = {
                "missing": f"no {anchor_repo_path} at {ANCHORS_REV[:12]}"
            }
        else:
            cc, cc_prov = load_anchor_cc(anchor_path, layer)
            arm_dirs["c_C"] = cc
            arm_rec["directions_provenance"]["c_C"] = cc_prov

        # Top-8 factors per module at the align layer.
        i650 = _issue650()
        module_factors: dict[str, tuple[str, np.ndarray]] = {}
        if entry.method == "lora":
            arm_dir = dl_root / "adapters" / entry.arm_id / entry.subfolder
            if not (arm_dir / "adapter_model.safetensors").is_file():
                arm_dir = _stage_adapter(entry, dl_root / "adapters")
            pairs, scaling = load_adapter_factors(arm_dir)
            for module in sorted({m for (_l, m) in pairs}):
                if (layer, module) not in pairs:
                    raise RuntimeError(
                        f"{entry.arm_id}: adapter has no (layer={layer}, {module}) pair"
                    )
                ab = pairs[(layer, module)]
                u, s, vh = lora_svd_factors(ab["A"], ab["B"], scaling)
                side = i650.RESIDUAL_SIDE_BY_MODULE[module]
                kk = min(TOP_K_FACTORS, s.shape[-1])
                stack = (u[:, :kk].T if side == "U" else vh[:kk]).numpy()
                module_factors[module] = (side, stack)
        else:
            factors_path = out_root / "dw_fleet" / "ft" / f"{entry.arm_id}_factors_L{layer}.pt"
            if not factors_path.is_file():
                raise RuntimeError(
                    f"ft factor sidecar missing at {factors_path} — run --phase ft "
                    f"(same --align-layer {layer}) before --phase align"
                )
            sidecar = torch.load(factors_path, weights_only=True, map_location="cpu")
            if int(sidecar["layer"]) != layer:
                raise RuntimeError(
                    f"ft factor sidecar layer {sidecar['layer']} != align layer {layer}"
                )
            module_factors = {
                m: (v["side"], v["factors"].numpy()) for m, v in sidecar["modules"].items()
            }

        for module, (side, stack) in sorted(module_factors.items()):
            # Basis routing: U-side factors live in residual-OUTPUT space (all directions
            # read); V-side factors live in residual-INPUT space for q/k/v/gate/up (the
            # context-space direction c_C reads). o_proj input (head-concat) and
            # down_proj input (ffn) never appear here — the side selection above already
            # took the module's RESIDUAL side, so a mismatched-basis cosine is
            # structurally impossible, not just dim-checked.
            if side == "U":
                read_names = sorted(arm_dirs)
            else:
                read_names = [n for n in ("c_C",) if n in arm_dirs]
            reads: dict[str, dict] = {}
            for name in read_names:
                d = arm_dirs[name]
                if d.shape[0] != stack.shape[1]:
                    raise RuntimeError(
                        f"{entry.arm_id} L{layer}.{module}: direction {name} dim "
                        f"{d.shape[0]} != factor dim {stack.shape[1]} (side {side}) — "
                        "corrupted input, refusing to publish a mismatched-basis cosine"
                    )
                reads[name] = alignment_vs_null(stack, d)
            arm_rec["factors"][f"L{layer}.{module}"] = {
                "side": side,
                "k_basis": int(stack.shape[0]),
                "alignments": reads,
            }

        arm_rec.update({"regime_key": rk, "metadata": _meta("align")})
        align_dir.mkdir(parents=True, exist_ok=True)
        _atomic_json(unit_path, arm_rec)
        results[entry.arm_id] = arm_rec
        print(
            f"[dw-align] unit {k}/{len(fleet)} {entry.arm_id} elapsed={time.time() - t0:.0f}s",
            flush=True,
        )

    # Coverage: absence is recorded per arm above; a PRIMARY direction absent across ALL
    # LoRA arms is the silent-loss scenario and refuses to publish.
    def _missing(name: str) -> list[str]:
        return sorted(
            a
            for a, r in results.items()
            if "missing" in r.get("directions_provenance", {}).get(name, {})
        )

    coverage = {"delta_tbar_missing": _missing("delta_tbar"), "c_C_missing": _missing("c_C")}
    lora_ids = [e.arm_id for e in fleet if e.method == "lora"]
    if lora_ids and all(a in coverage["delta_tbar_missing"] for a in lora_ids):
        raise RuntimeError(
            "H5 primary direction lost: NO LoRA arm resolved a banked delta_tbar at "
            f"{DELTA_TF_PREFIX}@{DELTA_TF_REV} — refusing to publish alignment.json"
        )
    if lora_ids and all(a in coverage["c_C_missing"] for a in lora_ids):
        raise RuntimeError(
            "c_C lost: NO LoRA arm resolved a banked anchor at "
            f"{ANCHORS_PREFIX}@{ANCHORS_REV[:12]} — refusing to publish alignment.json"
        )

    # Seed-noise anchor: #1979 impoliteness-contrastive seed pair (s42 vs s137).
    seed_pair = ("imp-pers-con-lr3e5-s42", "imp-pers-con-lr3e5-s137")
    anchor_rec: dict = {
        "pair": list(seed_pair),
        "note": "no full-FT seed pair exists (scope limit)",
    }
    pair_entries = {e.arm_id: e for e in fleet if e.arm_id in seed_pair}
    if len(pair_entries) == 2:
        vecs: dict[str, dict[str, np.ndarray]] = {}
        for aid, e in pair_entries.items():
            arm_dir = dl_root / "adapters" / aid / e.subfolder
            if not (arm_dir / "adapter_model.safetensors").is_file():
                arm_dir = _stage_adapter(e, dl_root / "adapters")
            pairs, scaling = load_adapter_factors(arm_dir)
            for module in OUTPUT_SIDE_MODULES:
                if (layer, module) in pairs:
                    ab = pairs[(layer, module)]
                    u, s, _vh = lora_svd_factors(ab["A"], ab["B"], scaling)
                    kk = min(TOP_K_FACTORS, s.shape[-1])
                    vecs.setdefault(module, {})[aid] = u[:, :kk].T.numpy()
        for module, by_arm in vecs.items():
            if len(by_arm) == 2:
                u1, u2 = (by_arm[a] for a in seed_pair)
                cos = np.abs(u1 @ u2.T)
                anchor_rec[module] = {
                    "top1_abs_cos": float(cos[0, 0]),
                    "max_abs_cos_topk": float(cos.max()),
                }
    else:
        anchor_rec["skipped"] = f"seed pair not fully in fleet: {sorted(pair_entries)}"

    _atomic_json(
        out_root / "dw_fleet" / "alignment.json",
        {
            "layer": layer,
            "arms": results,
            "coverage": coverage,
            "seed_noise_anchor": anchor_rec,
            "pins": {
                "delta_tf": f"{DELTA_TF_PREFIX}@{DELTA_TF_REV}",
                "r_b": f"{RB_PREFIX}@{RB_REV}",
                "anchors": f"{ANCHORS_PREFIX}@{ANCHORS_REV}",
            },
            "metadata": _meta("align"),
        },
    )
    return 0


def _load_fleet_table(out_root: Path) -> list[FleetEntry]:
    """Read back the persisted fleet table (fail-loud if the fleet phase has not run)."""
    p = out_root / "dw_fleet" / "fleet_table.json"
    if not p.is_file():
        raise RuntimeError(f"fleet table missing at {p} — run --phase fleet first")
    rows = json.loads(p.read_text())["fleet"]
    return [FleetEntry(**{k: r[k] for k in FleetEntry.__dataclass_fields__}) for r in rows]


def cmd_fleet(args) -> int:
    """Enumerate + persist the realized fleet table BEFORE any checkpoint download."""
    from huggingface_hub import hf_hub_download

    from explore_persona_space.orchestrate import hub

    out_root = Path(args.out_root)
    arms_path = hub.retry_transient(
        lambda: hf_hub_download(
            DATA_REPO,
            ARMS_JSON_PATH,
            repo_type="dataset",
            revision=ARMS_JSON_REV,
            local_dir=Path(args.dl_root) / "config",
        ),
        what=f"arms.json fetch ({ARMS_JSON_PATH}@{ARMS_JSON_REV})",
    )
    arms_json = json.loads(Path(arms_path).read_text())
    extra: list[dict] = []
    for p in args.extra_arms_json or []:
        extra.extend(json.loads(Path(p).read_text())["arms"])
    fleet = enumerate_fleet(arms_json, extra)
    _atomic_json(
        out_root / "dw_fleet" / "fleet_table.json",
        {
            "fleet": [dataclasses.asdict(e) for e in fleet],
            "n_lora": sum(1 for e in fleet if e.method == "lora"),
            "n_ft": sum(1 for e in fleet if e.method == "ft"),
            "metadata": _meta("fleet"),
        },
    )
    log.info("[dw] fleet table persisted: %d entries", len(fleet))
    return 0


def cmd_pilot(args) -> int:
    """Sizing gate PINNED to one FULL-FT checkpoint; rc=7 on refusal.

    Plan §4 leg-5 step 6 + blind-spot (2): the pilot exists to certify the
    private-overflow-repo full-FT download path (the one path the smoke cannot), so it
    stages ft[0] (TIMED — checkpoint IO is part of the basis), runs the PRODUCTION
    analysis function on TWO production-shape modules (down_proj MLP-wide + q_proj
    attention-square), stages + analyzes one LoRA arm through the production exact-rank-r
    path, and extrapolates the battery wall (fix-round-2 blocker
    ``dwfleet-pilot-not-fullft-plan-adherence``: the prior pilot resolved lora[0] +
    a synthetic randn matrix and excluded all checkpoint IO).
    """
    out_root = Path(args.out_root)
    dl_root = Path(args.dl_root)
    layer = int(args.align_layer)
    fleet = _load_fleet_table(out_root)
    lora = [e for e in fleet if e.method == "lora"]
    ft = [e for e in fleet if e.method == "ft"]
    if not ft:
        raise RuntimeError(
            "pilot is PINNED to one FULL-FT checkpoint (plan §4 leg-5 blind-spot (2): it "
            "must certify the private-overflow full-FT download path) — no ft arms in fleet"
        )
    if not args.base_ckpt:
        raise RuntimeError(
            "--base-ckpt is required: the pilot measures REAL theta_post - theta_base deltas"
        )
    base_svd = _load_base_svd_required(args.base_svd)

    entry = ft[0]
    t_dl0 = time.time()
    post_dir = dl_root / "ft" / entry.arm_id / entry.subfolder
    if not any(post_dir.glob("*.safetensors")):
        post_dir = _stage_checkpoint(entry.repo_id, entry.subfolder, dl_root / "ft", entry.arm_id)
    ft_dl_s = time.time() - t_dl0

    pilot_modules = ("down_proj", "q_proj")  # one MLP-wide + one attention-square
    pilot_dir = out_root / "dw_fleet" / "pilot"
    t0 = time.time()
    ft_rec = analyze_ft_checkpoint(
        entry,
        Path(args.base_ckpt),
        post_dir,
        base_svd,
        align_layer=layer,
        factors_path=pilot_dir / f"pilot_ft_factors_L{layer}.pt",
        module_filter=pilot_modules,
    )
    ft_elapsed = time.time() - t0
    n_ft_calls = int(ft_rec["n_matrices"])
    ft_call_s = ft_elapsed / max(1, n_ft_calls)

    lora_dl_s = 0.0
    lora_arm_s = 0.0
    lora_pilot_arm = None
    if lora:
        le = lora[0]
        lora_pilot_arm = le.arm_id
        t_dl1 = time.time()
        arm_dir = dl_root / "adapters" / le.arm_id / le.subfolder
        if not (arm_dir / "adapter_model.safetensors").is_file():
            arm_dir = _stage_adapter(le, dl_root / "adapters")
        lora_dl_s = time.time() - t_dl1
        t1 = time.time()
        analyze_lora_arm(le, arm_dir, base_svd)
        lora_arm_s = time.time() - t1

    projected_s = len(ft) * (FT_MATRICES_PER_CKPT * ft_call_s + ft_dl_s) + len(lora) * (
        lora_arm_s + lora_dl_s
    )
    projected_h = projected_s / 3600.0
    verdict = "pass" if projected_h <= float(args.pilot_wall_cap_h) else "split-fleet-2-pods"
    report = {
        "pilot_arm": entry.arm_id,
        "pilot_method": "ft",
        "pilot_repo": entry.repo_id,
        "pilot_subfolder": entry.subfolder,
        "pilot_modules": list(pilot_modules),
        "measured_ft_dl_s": ft_dl_s,
        "measured_per_call_s_ft": ft_call_s,
        "n_pilot_ft_matrices": n_ft_calls,
        "lora_pilot_arm": lora_pilot_arm,
        "measured_lora_dl_s": lora_dl_s,
        "measured_lora_arm_s": lora_arm_s,
        "ft_matrices_per_ckpt": FT_MATRICES_PER_CKPT,
        "n_lora_arms": len(lora),
        "n_ft_ckpts": len(ft),
        "projected_wall_h": projected_h,
        "wall_cap_h": float(args.pilot_wall_cap_h),
        "verdict": verdict,
        "metadata": _meta("pilot"),
    }
    _atomic_json(out_root / "dw_fleet" / "pilot.json", report)
    print(f"[dw-pilot] projected_wall_h={projected_h:.2f} verdict={verdict}", flush=True)
    if verdict != "pass":
        return RC_PILOT_REFUSAL
    return 0


def cmd_lora(args) -> int:
    """LoRA battery: per-arm spectra + effective rank + intruder read, checkpoint-per-arm."""
    out_root = Path(args.out_root)
    dl_root = Path(args.dl_root)
    fleet = [e for e in _load_fleet_table(out_root) if e.method == "lora"]
    if args.arms:
        want = {a.strip() for a in args.arms.split(",")}
        fleet = [e for e in fleet if e.arm_id in want]
        if not fleet:
            raise RuntimeError(f"empty fleet after --arms filter: {sorted(want)}")
    base_svd = _load_base_svd_required(args.base_svd)
    rk = regime_key(
        phase="lora",
        top_k=TOP_K_FACTORS,
        dv3_draws=DV3_NULL_DRAWS,
        arms_rev=ARMS_JSON_REV,
        # Content-describing key: a payload rebuilt with a different module list / base
        # model changes the key, so stale units recompute (fix-round-2: bool(base_svd)
        # was blind to content and resume-skipped every stale unit).
        base_svd_meta=base_svd.get("_meta", {}),
    )
    t0 = time.time()
    for k, entry in enumerate(fleet, start=1):
        unit_path = out_root / "dw_fleet" / "lora" / f"{entry.arm_id}.json"
        if not args.no_resume and unit_path.is_file():
            try:
                if json.loads(unit_path.read_text()).get("regime_key") == rk:
                    print(f"[dw-lora] unit {k}/{len(fleet)} {entry.arm_id} resume-skip", flush=True)
                    continue
            except json.JSONDecodeError:
                pass
        arm_dir = dl_root / "adapters" / entry.arm_id / entry.subfolder
        if not (arm_dir / "adapter_model.safetensors").is_file():
            arm_dir = _stage_adapter(entry, dl_root / "adapters")
        rec = analyze_lora_arm(entry, arm_dir, base_svd)
        rec.update({"regime_key": rk, "metadata": _meta("lora")})
        _atomic_json(unit_path, rec)
        print(
            f"[dw-lora] unit {k}/{len(fleet)} {entry.arm_id} elapsed={time.time() - t0:.0f}s",
            flush=True,
        )
    return 0


def cmd_ft(args) -> int:
    """Full-FT battery: streamed dW spectra + intruder + align-layer factors, per-arm."""
    out_root = Path(args.out_root)
    dl_root = Path(args.dl_root)
    layer = int(args.align_layer)
    fleet = [e for e in _load_fleet_table(out_root) if e.method == "ft"]
    if args.arms:
        want = {a.strip() for a in args.arms.split(",")}
        fleet = [e for e in fleet if e.arm_id in want]
        if not fleet:
            raise RuntimeError(f"empty fleet after --arms filter: {sorted(want)}")
    if not args.base_ckpt:
        raise RuntimeError("--base-ckpt (staged base-model checkpoint dir) is required for ft")
    base_dir = Path(args.base_ckpt)
    base_svd = _load_base_svd_required(args.base_svd)
    rk = regime_key(
        phase="ft",
        arms_rev=ARMS_JSON_REV,
        align_layer=layer,
        top_k=TOP_K_FACTORS,
        dv3_draws=DV3_NULL_DRAWS,
        lowrank=(LOWRANK_Q, LOWRANK_NITER),
        base_svd_meta=base_svd.get("_meta", {}),
    )
    for k, entry in enumerate(fleet, start=1):
        unit_path = out_root / "dw_fleet" / "ft" / f"{entry.arm_id}.json"
        if not args.no_resume and unit_path.is_file():
            try:
                if json.loads(unit_path.read_text()).get("regime_key") == rk:
                    print(f"[dw-ft] unit {k}/{len(fleet)} {entry.arm_id} resume-skip", flush=True)
                    continue
            except json.JSONDecodeError:
                pass
        post_dir = dl_root / "ft" / entry.arm_id / entry.subfolder
        if not any(post_dir.glob("*.safetensors")):
            post_dir = _stage_checkpoint(
                entry.repo_id, entry.subfolder, dl_root / "ft", entry.arm_id
            )
        rec = analyze_ft_checkpoint(
            entry,
            base_dir,
            post_dir,
            base_svd,
            align_layer=layer,
            factors_path=out_root / "dw_fleet" / "ft" / f"{entry.arm_id}_factors_L{layer}.pt",
        )
        rec.update({"regime_key": rk, "metadata": _meta("ft")})
        _atomic_json(unit_path, rec)
        print(f"[dw-ft] unit {k}/{len(fleet)} {entry.arm_id} done", flush=True)
    return 0


PHASES = {
    "fleet": cmd_fleet,
    "pilot": cmd_pilot,
    "lora": cmd_lora,
    "ft": cmd_ft,
    "align": cmd_align,
}


def main(argv: list[str] | None = None) -> int:
    """CLI: phase-dispatch driver for the leg-5 dW fleet battery (P-C pod)."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--phase", choices=sorted(PHASES), required=False)
    ap.add_argument("--out-root", default="/workspace/eps2569/out")
    ap.add_argument("--dl-root", default="/workspace/eps2569/dl")
    ap.add_argument(
        "--extra-arms-json",
        nargs="*",
        default=None,
        help="additional fleet manifests (#2474/#2379/#1947), same arms schema",
    )
    ap.add_argument("--arms", default=None, help="comma list of arm_ids to run")
    ap.add_argument(
        "--base-svd",
        default=None,
        help=(
            "base_svd.pt from issue650_analyze build-base-svd --modules <all 7> "
            "(REQUIRED for lora/ft/pilot — the intruder read is plan-registered)"
        ),
    )
    ap.add_argument("--base-ckpt", default=None, help="staged base-model checkpoint dir (ft)")
    ap.add_argument("--align-layer", default="19")
    ap.add_argument("--map-root", default=None, help="banked ridge.pt root override (leg-8 A r)")
    ap.add_argument("--pilot-wall-cap-h", default=str(PILOT_WALL_CAP_H))
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--import-check", action="store_true", help="static arg/bind check, exit 0")
    args = ap.parse_args(argv)

    if args.import_check:
        from explore_persona_space.orchestrate.argcheck import assert_args_attributes_defined

        assert_args_attributes_defined(__file__)
        raise SystemExit(0)
    if not args.phase:
        ap.error("--phase is required (unless --import-check)")

    rc = PHASES[args.phase](args)
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(int(rc))


if __name__ == "__main__":
    main()
