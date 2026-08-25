"""Issue #2569 leg 5 — LoRA / full-FT weight-update (dW) geometry across the organism fleet.

Per checkpoint in the fleet (arms.json @ ``3bb20debe2`` on the HF data repo + any extra
manifests passed via ``--extra-arms-json``, enumerated + persisted BEFORE any download):

- **dW spectra + effective rank:** LoRA dW = B @ A * s per (layer, module), with s = the
  adapter's OWN scaling from ``adapter_config.json`` (alpha/sqrt(r) under rsLoRA, alpha/r
  classic — artifact-reuse check (g)); full-FT dW = theta_post - theta_base per 2-D weight
  matrix (streamed one tensor at a time — never materialized as a whole checkpoint delta).
  Summaries per matrix: stable rank ||dW||_F^2/||dW||_2^2, participation ratio
  (sum s_i^2)^2 / sum s_i^4, top-1 share (energy sigma_1^2/sum sigma_i^2, the #1979/#1947
  convention, plus the plain sigma_1/sum sigma_i companion) — descriptive, never gates.
- **Intruder read (#650 convention, load-enforced):** max |cos| of top dW singular vectors
  against the base weight SVD bases built by ``issue650_analyze.py build-base-svd``
  (extended in place with ``--modules``), aggregated EXACTLY as
  ``max_over_base_singular_vectors_then_max_over_band`` via ``dv3_max_matched_null``;
  results are serialized in the nested #650 §6.5 schema and validated by
  ``assert_dv3_schema`` at write AND load — any other aggregation is rejected.
- **Factor alignment:** top-8 dW OUTPUT-side singular vectors (down_proj / o_proj U-side,
  residual-output space) vs delta (the #1768 ``delta_tf/<arm>/tbar.pt`` displacement means),
  r_B (``issue779_monitoring/r_b/*.pt``), c_C (the arm's training-context centroid from
  #1979's banked anchors, ``issue1900_leakrace/anchors/<arm_id>.pt``), and the gate
  direction A r (leg-8 monitor gradient under the row convention — computed locally via
  ``issue2569_operator`` from the banked ridge.pt, B1; ``run_driver_identity_asserts`` runs
  at that phase's entry). INPUT-side V vectors vs context-space directions (c_C). Every
  alignment is read against the max-matched null. Seed-noise anchor: the #1979
  impoliteness-contrastive seed pair (s42 vs s137) gives the LoRA within-recipe
  dW-similarity floor; no full-FT seed pair exists (stated scope limit).
- **Sizing gate (§7 row 2):** ``--phase pilot`` measures ONE checkpoint x TWO modules at
  production shape and extrapolates the battery wall; a projection > ``--pilot-wall-cap-h``
  (default 8 h) writes ``pilot.json`` with the arithmetic and exits rc=7 (a DESIGNED
  artifact-routed halt, never an anonymous rc=1) so the dispatcher splits the fleet across
  2 pods by checkpoint.

Phases (``--phase``): ``fleet`` (enumerate + persist the realized fleet table), ``pilot``,
``lora`` (all LoRA arms), ``ft`` (full-FT checkpoints), ``align`` (factor alignments).
Checkpoint-per-unit: each (arm) writes its JSON the moment it completes, with a
machine-stable resume key from generating parameters. ``torch.load(weights_only=False)``
is never needed here — safetensors + ``weights_only=True`` payloads only, except the
self-produced tbar/anchor .pt payloads (revision-pinned, #1900 precedent).
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
DELTA_TF_PREFIX = "issue1768_mapshift/delta_tf"
DELTA_TF_REV = "c07267285d"
RB_PREFIX = "issue779_monitoring/r_b"
RB_REV = "037fcbb2"
RB_TRAITS = ("evil", "hallucination", "sycophancy")

# The 7 LoRA target modules (r32/alpha64 rsLoRA fleet) + attention modules for full-FT.
LORA_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
# Residual-OUTPUT-side modules: their dW LEFT singular vectors live in residual space.
OUTPUT_SIDE_MODULES = ("o_proj", "down_proj")
TOP_K_FACTORS = 8
DV3_NULL_DRAWS = 200
DV3_NULL_AGGREGATION = "max_over_base_singular_vectors_then_max_over_band"
PILOT_WALL_CAP_H = 8.0
RC_PILOT_REFUSAL = 7  # the #1415 artifact-routed halt convention

_ADAPTER_KEY_RE = re.compile(
    r"base_model\.model\.model\.layers\.(\d+)\.(self_attn|mlp)\.(\w+)\.lora_(A|B)\.weight"
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
    """dW = B @ A * s in fp32; A is (r, d_in), B is (d_out, r)."""
    assert a.shape[0] == b.shape[1], (a.shape, b.shape)
    return (b.to(torch.float32) @ a.to(torch.float32)) * float(scaling)


def load_adapter_deltas(adapter_dir: Path) -> dict[tuple[int, str], torch.Tensor]:
    """Load per-(layer, module) dW from a PEFT adapter directory (ANY rank).

    Unlike ``issue650_analyze.load_adapter_pairs`` (which asserts r == 1 for its rank-1
    geometry reads), this reconstructs the full-rank dW = B A s for the r32 fleet.
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
    out: dict[tuple[int, str], torch.Tensor] = {}
    for (layer, module), ab in sorted(pairs.items()):
        if "A" not in ab or "B" not in ab:
            raise RuntimeError(f"incomplete LoRA pair at layer {layer} module {module}")
        out[(layer, module)] = delta_w_from_lora(ab["A"], ab["B"], s)
    if not out:
        raise RuntimeError(f"no LoRA weight pairs found under {adapter_dir}")
    return out


def iter_ft_deltas(base_dir: Path, post_dir: Path):
    """Yield (param_name, dW fp32) for every 2-D weight, one tensor at a time (stream-reduce).

    Both directories are HF-format checkpoints (safetensors shards + index). Peak RSS stays
    O(one tensor) — never the whole 15 GB delta (earlyoom stream-reduce rule, #658).
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
        w_base = _get(base_map, name)
        if w_base.ndim != 2:
            continue
        w_post = _get(post_map, name)
        yield name, (w_post.to(torch.float32) - w_base.to(torch.float32))


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
    """Batched singular values of a (B, m, n) stack; per-slice fallback on non-convergence.

    One batched LAPACK call on the healthy path (vectorize-first); a single bad slice
    triggers the per-matrix path via ``svdvals_robust`` (gotchas: batched solves raise ONE
    error for the whole stack).
    """
    try:
        return torch.linalg.svdvals(stack).to(torch.float64).numpy()
    except torch.linalg.LinAlgError:
        log.warning("[dw] batched svdvals non-convergence — per-slice fallback")
        return np.stack([svdvals_robust(stack[i]) for i in range(stack.shape[0])])


def effective_rank_summaries(svals: np.ndarray) -> dict:
    """Descriptive spectral summaries (never gates): stable rank, PR, top-1 shares."""
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
    output. The returned payload carries the ``observed`` / ``null`` / ``assertions``
    blocks and MUST pass ``issue650_analyze.assert_dv3_schema`` (validated here at write —
    a mismatched aggregation is rejected before anything is persisted).
    """
    observed: dict = {}
    null: dict = {}
    for arm, res in arm_results.items():
        if res["null_aggregation"] != DV3_NULL_AGGREGATION:
            raise AssertionError(
                f"dv3 {arm}: aggregation {res['null_aggregation']!r} != {DV3_NULL_AGGREGATION!r}"
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
    combined by the same band-max the null uses.
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


def analyze_lora_arm(entry: FleetEntry, adapter_dir: Path, base_svd: dict | None) -> dict:
    """Spectra + effective rank + intruder read for one LoRA arm (batched per module)."""
    deltas = load_adapter_deltas(adapter_dir)
    by_module: dict[str, dict[int, torch.Tensor]] = {}
    for (layer, module), dw in deltas.items():
        by_module.setdefault(module, {})[layer] = dw
    rec: dict = {"arm_id": entry.arm_id, "method": "lora", "modules": {}}
    for module, by_layer in sorted(by_module.items()):
        layers = sorted(by_layer)
        stack = torch.stack([by_layer[layer] for layer in layers])
        svals = svdvals_stack(stack)
        rec["modules"][module] = {
            str(layer): effective_rank_summaries(svals[i]) for i, layer in enumerate(layers)
        }
    if base_svd is not None:
        rec["intruder"] = {}
        for module in OUTPUT_SIDE_MODULES:
            if module not in by_module:
                continue
            basis = {
                layer: base_svd[module][layer]["U"]
                for layer in sorted(by_module[module])
                if layer in base_svd.get(module, {})
            }
            if not basis:
                continue
            top_vecs = {}
            for layer in basis:
                u, _, _ = torch.linalg.svd(by_module[module][layer], full_matrices=False)
                top_vecs[layer] = u[:, 0].numpy()
            rec["intruder"][module] = intruder_read(top_vecs, basis, arm_name="write")
    return rec


def analyze_ft_checkpoint(entry: FleetEntry, base_dir: Path, post_dir: Path) -> dict:
    """Spectra + effective rank for every 2-D matrix of one full-FT checkpoint (streamed)."""
    rec: dict = {"arm_id": entry.arm_id, "method": "ft", "matrices": {}}
    n = 0
    t0 = time.time()
    for name, dw in iter_ft_deltas(base_dir, post_dir):
        svals = svdvals_robust(dw)
        rec["matrices"][name] = effective_rank_summaries(svals)
        n += 1
        print(f"[dw-ft] unit {n} {entry.arm_id}/{name} elapsed={time.time() - t0:.0f}s", flush=True)
    if n == 0:
        raise RuntimeError(f"no 2-D matrices found for {entry.arm_id}")
    rec["n_matrices"] = n
    return rec


def top_factors_for_alignment(
    dw: torch.Tensor, *, k: int = TOP_K_FACTORS
) -> tuple[np.ndarray, np.ndarray]:
    """(U_k rows, V_k rows) — top-k output-side and input-side singular vectors of dW."""
    u, _, vh = torch.linalg.svd(dw.to(torch.float32), full_matrices=False)
    return u[:, :k].T.numpy(), vh[:k].numpy()


def _unit_vec(x: np.ndarray) -> np.ndarray:
    """L2-normalize a 1-D direction (fail-loud on zero norm)."""
    v = np.asarray(x, dtype=np.float64).ravel()
    n = float(np.linalg.norm(v))
    if n == 0:
        raise ValueError("zero-norm direction")
    return v / n


def load_direction_pt(
    path: Path, *, key: str | None = None, layer: int | None = None
) -> np.ndarray:
    """Load a banked direction .pt (tbar / r_B / anchor); self-produced + revision-pinned."""
    payload = torch.load(path, weights_only=False, map_location="cpu")
    obj = payload
    if key is not None:
        obj = obj[key]
    if layer is not None and isinstance(obj, dict):
        obj = obj[layer]
    if torch.is_tensor(obj):
        return _unit_vec(obj.to(torch.float64).numpy())
    if isinstance(obj, np.ndarray):
        return _unit_vec(obj)
    raise TypeError(f"cannot resolve a direction from {path} (key={key}, layer={layer})")


def cmd_align(args) -> int:
    """Factor-alignment phase: top-8 dW factors vs delta / r_B / c_C / A r (+ seed anchor).

    Runs OP.run_driver_identity_asserts at entry (the operator-consuming path — B1).
    """
    import issue2569_operator as op

    dl_root = Path(args.dl_root)
    out_root = Path(args.out_root)
    layer = int(args.align_layer)

    payload = op.load_banked_map(layer=layer, root=args.map_root or None)
    op.run_driver_identity_asserts(payload)
    a_mat, _b = op.row_operator(payload)

    fleet = _load_fleet_table(out_root)
    directions: dict[str, np.ndarray] = {}
    for trait in RB_TRAITS:
        rb_path = dl_root / "r_b" / f"{trait}.pt"
        if rb_path.is_file():
            rb = load_direction_pt(rb_path)
            directions[f"r_B[{trait}]"] = rb
            directions[f"Ar[{trait}]"] = _unit_vec(op.monitor_gradient(a_mat, rb))
    if not directions:
        raise RuntimeError(f"no r_B directions staged under {dl_root / 'r_b'} — stage them first")

    results: dict[str, dict] = {}
    for entry in fleet:
        if entry.method != "lora":
            continue  # ft factor alignment consumes the ft battery outputs (matrices on disk)
        arm_dir = dl_root / "adapters" / entry.arm_id / entry.subfolder
        if not (arm_dir / "adapter_model.safetensors").is_file():
            arm_dir = _stage_adapter(entry, dl_root / "adapters")
        deltas = load_adapter_deltas(arm_dir)
        arm_rec: dict = {"factors": {}}
        # Per-arm directions: delta (tbar) + c_C anchor, when staged.
        arm_dirs = dict(directions)
        tbar_path = dl_root / "delta_tf" / entry.arm_id / "tbar.pt"
        if tbar_path.is_file():
            arm_dirs["delta_tbar"] = load_direction_pt(tbar_path, key="tbar", layer=layer)
        anchor_path = dl_root / "anchors" / f"{entry.arm_id}.pt"
        if anchor_path.is_file():
            anchor = torch.load(anchor_path, weights_only=False, map_location="cpu")
            vec = anchor.get("c_C", anchor.get("centroid")) if isinstance(anchor, dict) else anchor
            if torch.is_tensor(vec):
                arm_dirs["c_C"] = _unit_vec(vec.to(torch.float64).numpy())
            elif isinstance(vec, dict) and layer in vec:
                arm_dirs["c_C"] = _unit_vec(vec[layer].to(torch.float64).numpy())
        for module in OUTPUT_SIDE_MODULES:
            if (layer, module) not in deltas:
                continue
            u_k, v_k = top_factors_for_alignment(deltas[(layer, module)])
            mod_rec: dict = {}
            for name, d in sorted(arm_dirs.items()):
                side = v_k if name == "c_C" else u_k  # context directions read input-side V
                if d.shape[0] != side.shape[1]:
                    mod_rec[name] = {"skipped": f"dim mismatch {d.shape[0]} vs {side.shape[1]}"}
                    continue
                mod_rec[name] = alignment_vs_null(side, d)
            arm_rec["factors"][f"L{layer}.{module}"] = mod_rec
        results[entry.arm_id] = arm_rec
        print(f"[dw-align] {entry.arm_id} done", flush=True)

    # Seed-noise anchor: #1979 impoliteness-contrastive seed pair (s42 vs s137).
    seed_pair = ("imp-pers-con-lr3e5-s42", "imp-pers-con-lr3e5-s137")
    anchor_rec: dict = {
        "pair": list(seed_pair),
        "note": "no full-FT seed pair exists (scope limit)",
    }
    pair_entries = {e.arm_id: e for e in fleet if e.arm_id in seed_pair}
    if len(pair_entries) == 2:
        vecs = {}
        for aid, e in pair_entries.items():
            arm_dir = dl_root / "adapters" / aid / e.subfolder
            if not (arm_dir / "adapter_model.safetensors").is_file():
                arm_dir = _stage_adapter(e, dl_root / "adapters")
            deltas = load_adapter_deltas(arm_dir)
            for module in OUTPUT_SIDE_MODULES:
                if (layer, module) in deltas:
                    u_k, _ = top_factors_for_alignment(deltas[(layer, module)])
                    vecs.setdefault(module, {})[aid] = u_k
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
            "seed_noise_anchor": anchor_rec,
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
    """Sizing gate: 1 checkpoint x 2 modules measured at production shape; rc=7 on refusal."""
    out_root = Path(args.out_root)
    fleet = _load_fleet_table(out_root)
    lora = [e for e in fleet if e.method == "lora"]
    ft = [e for e in fleet if e.method == "ft"]
    entry = lora[0]
    arm_dir = _stage_adapter(entry, Path(args.dl_root) / "adapters" / entry.arm_id)
    deltas = load_adapter_deltas(arm_dir)
    # Time TWO production-shape module batteries (one MLP-wide, one attention-square).
    picks = [m for m in ("down_proj", "q_proj") if any(k[1] == m for k in deltas)][:2]
    t0 = time.time()
    n_calls = 0
    for module in picks:
        layers = sorted(layer for (layer, m) in deltas if m == module)
        stack = torch.stack([deltas[(layer, module)] for layer in layers])
        svdvals_stack(stack)
        n_calls += len(layers)
    per_call_s = (time.time() - t0) / max(1, n_calls)
    # Battery arithmetic: LoRA tiny SVDs + full-FT large SVDs (196 matrices / ckpt at ~28x7).
    lora_calls = len(lora) * len(LORA_MODULES) * 28
    ft_calls = len(ft) * 196
    # Full-FT matrices are up to (3584, 18944) fp32 — measure one large synthetic SVD.
    t1 = time.time()
    svdvals_robust(torch.randn(3584, 18944) * 1e-3)
    large_call_s = time.time() - t1
    projected_h = (lora_calls * per_call_s + ft_calls * large_call_s) / 3600.0
    verdict = "pass" if projected_h <= float(args.pilot_wall_cap_h) else "split-fleet-2-pods"
    report = {
        "measured_per_call_s_lora": per_call_s,
        "measured_per_call_s_ft_large": large_call_s,
        "n_pilot_calls": n_calls,
        "pilot_modules": picks,
        "lora_calls": lora_calls,
        "ft_calls": ft_calls,
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
    base_svd = None
    if args.base_svd and Path(args.base_svd).is_file():
        base_svd = _issue650().load_base_svd(Path(args.base_svd))
    rk = regime_key(
        phase="lora",
        top_k=TOP_K_FACTORS,
        dv3_draws=DV3_NULL_DRAWS,
        arms_rev=ARMS_JSON_REV,
        base_svd=bool(base_svd),
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
            arm_dir = _stage_adapter(entry, dl_root / "adapters" / entry.arm_id)
        rec = analyze_lora_arm(entry, arm_dir, base_svd)
        rec.update({"regime_key": rk, "metadata": _meta("lora")})
        _atomic_json(unit_path, rec)
        print(
            f"[dw-lora] unit {k}/{len(fleet)} {entry.arm_id} elapsed={time.time() - t0:.0f}s",
            flush=True,
        )
    return 0


def cmd_ft(args) -> int:
    """Full-FT battery: per-checkpoint streamed dW spectra, checkpoint-per-arm."""
    out_root = Path(args.out_root)
    dl_root = Path(args.dl_root)
    fleet = [e for e in _load_fleet_table(out_root) if e.method == "ft"]
    if args.arms:
        want = {a.strip() for a in args.arms.split(",")}
        fleet = [e for e in fleet if e.arm_id in want]
        if not fleet:
            raise RuntimeError(f"empty fleet after --arms filter: {sorted(want)}")
    if not args.base_ckpt:
        raise RuntimeError("--base-ckpt (staged base-model checkpoint dir) is required for ft")
    base_dir = Path(args.base_ckpt)
    rk = regime_key(phase="ft", arms_rev=ARMS_JSON_REV)
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
        rec = analyze_ft_checkpoint(entry, base_dir, post_dir)
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
        help="base_svd.pt from issue650_analyze build-base-svd (intruder read)",
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
