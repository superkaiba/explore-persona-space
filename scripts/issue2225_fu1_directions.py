"""Issue #2225 fu1 — F0: pre-image direction construction + rho probe (plan §4.1).

Builds the fu1 direction bank consumed by scripts/issue2225_fu1_train.py:

  {out_dir}/{trait}_PRE.pt   (28, 3584) — rows 14/19 = rho_l * unit pre-image
  {out_dir}/RND.pt           (28, 3584) — rows 14/19 = rho_l * unit random ctrl
  {out_dir}/{trait}_PRE_meta.json / RND_meta.json / rho.json

Every non-map row is NaN (fail-loud if a wrong layer is ever sliced; the
consumer additionally asserts finiteness on its sliced row). Pre-image per
trait x layer l in {14, 19} (float64):

  M = W^T (the #779 ridge map's standardized-context -> centered-answer frame)
  d_pre = normalize(xsd * V_k diag(1/s_k) U_k^T r_B[l]),  k = numerical rank

with the HALT-class frame-fold gate cos(M @ (d_pre/xsd), P_k(r_B[l])) >= 0.999,
a rank-sweep disclosure at k in {224, 896, 1792, 3584}, and Result-0-style
geometry diagnostics (cos vs r_B / parent E2/E3 / random; the rb_v2 <-> #779
r_b bridge cosine incl. the +-1 layer-offset detector, plan §12 A2).

Inputs (revision-pinned, plan §10; staged local-first):
  ridge payloads  issue779_monitoring/n1m_readout/weights/L{14,19}/ridge.pt
                  @ 9d8f789bf034d8f244e1d00e0dbbe6aba6d272c5
  rb_v2 (E1)      issue778_persona_vectors/analysis_tensors_v2/rb_v2/{trait}.pt
                  @ 032bdef
  #779 r_b        issue779_monitoring/r_b/{trait}.pt @ 037fcbb2 (bridge only)
  parent E2/E3    issue2225_ctxsteer/analysis_tensors/directions/{trait}_E{2,3}.pt

Phases:
  --probe-rho   1-GPU: median ||h_l|| at the LAST context token over --n-prompts
                sampled training prompts per corpus (seed 2225), pooled median
                per layer -> rho.json (per-corpus values reported; #2220 WARN band)
  --build       CPU algebra on the FULL payloads (plan §4.5 blind-spot (e):
                never sliced under smoke) -> bank + meta
  --upload      one upload_folder commit -> issue2225_ctxsteer/analysis_tensors/
                fu1_directions/
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
from collections.abc import Sequence

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()  # BEFORE numpy/torch: shared-VM thread caps bind at import (#847)

import numpy as np  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import issue778_lib as lib  # noqa: E402

DATA_REPO = "superkaiba1/explore-persona-space-data"
FU1_DIRECTIONS_HF_PREFIX = "issue2225_ctxsteer/analysis_tensors/fu1_directions"

RIDGE_REV = "9d8f789bf034d8f244e1d00e0dbbe6aba6d272c5"
RB_V2_REV = "032bdef"
R_B_779_REV = "037fcbb2"

RIDGE_PREFIX = "issue779_monitoring/n1m_readout/weights"
RB_V2_PREFIX = "issue778_persona_vectors/analysis_tensors_v2/rb_v2"
R_B_779_PREFIX = "issue779_monitoring/r_b"
PARENT_DIRECTIONS_PREFIX = "issue2225_ctxsteer/analysis_tensors/directions"

# Plan-time staged copies (verified at plan time; used before any network).
PLAN_STAGED_ROOT = pathlib.Path("/mnt/eps-data/thomasjiralerspong/issue2225_fu1_plan")

MAP_LAYERS: tuple[int, ...] = (14, 19)
TRAITS: tuple[str, ...] = ("evil", "sycophancy", "hallucination")
RANK_SWEEP: tuple[int, ...] = (224, 896, 1792, 3584)
FRAME_FOLD_MIN_COS = 0.999  # HALT-class (plan §4.1 step 4)
RANK_FLOOR_REL = 1e-6  # numerical-rank floor: k_used = #{Sm_i >= 1e-6 * Sm_max}

RHO_SEED = 2225
RHO_N_PROMPTS_DEFAULT = 512
# #2220 committed L14 band (plan §4.1): WARN outside [0.5x, 2x] of [62.7, 64.2].
RHO_L14_REF_BAND = (62.7, 64.2)
# Fresh-L19 secondary sanity (L18/L20 bracket read): expect ~90-115 (WARN only).
RHO_L19_EXPECT_BAND = (90.0, 115.0)

RANDOM_SEED_BY_LAYER = {14: 2225014, 19: 2225019}

# ---------------------------------------------------------------------------
# VERBATIM ports from origin/issue-2254:scripts/issue2254_preimage.py
# (plan §4.1 "Ported code"; artifact-reuse.md § Porting from an unmerged
# sibling branch — provenance headers per function; numpy-only imports).
# ``kstar_from_fit`` is deliberately NOT ported: the 963k payloads carry no
# ``s`` spectrum, and their selected_lambda = 0.001 makes k* full-rank
# deterministically (plan §4.1 step 2) — the numerical-rank floor replaces it.
# ---------------------------------------------------------------------------

N_RANDOM_SEEDS = 3  # origin/issue-2254:scripts/issue2254_preimage.py L188


def map_svd(W):
    """VERBATIM issue2254_preimage.py L301 —
    M = W.T (v = M c_std; pinv_direction_read.py L163-164) + its SVD."""
    M = np.asarray(W, dtype=np.float64).T
    Um, Sm, Vmt = np.linalg.svd(M, full_matrices=False)
    return M, Um, Sm, Vmt


def preimage_w(Um, Sm, Vmt, r_b, k: int):
    """VERBATIM issue2254_preimage.py L308 — truncated pinv direction
    (pinv_direction_read.py L171-174): w = V_k diag(1/s_k) U_k^T r_B,
    standardized-context frame."""
    kk = int(min(k, Sm.shape[0]))
    if kk <= 0:
        raise ValueError(f"preimage_w: k={k} leaves no components (degenerate map)")
    coeff = (Um.T @ np.asarray(r_b, dtype=np.float64))[:kk] / Sm[:kk]
    return Vmt[:kk].T @ coeff


def destandardized_direction(xsd, w):
    """VERBATIM issue2254_preimage.py L318 — de-standardization fold:
    d = normalize(xsd * w) — the raw residual-space edit whose predicted
    answer shift is P_k*(r_B)."""
    d = np.asarray(xsd, dtype=np.float64) * np.asarray(w, dtype=np.float64)
    nrm = float(np.linalg.norm(d))
    if not (np.isfinite(nrm) and nrm > 0.0):
        raise ValueError(f"destandardized_direction: degenerate norm {nrm!r}")
    return d / nrm


def topk_projection(Um, r_b, k: int):
    """VERBATIM issue2254_preimage.py L330 (transitive dep of frame_fold_cos) —
    P_k(r_B) = U_k U_k^T r_B — projection onto the rank-k column space of M."""
    kk = int(min(max(k, 0), Um.shape[1]))
    u = Um[:, :kk]
    return u @ (u.T @ np.asarray(r_b, dtype=np.float64))


def frame_fold_cos(M, Um, xsd, d_pre, r_b, k: int) -> float:
    """VERBATIM issue2254_preimage.py L335 — HALT-class frame-fold test:
    cos(M @ (d_pre / xsd), P_k(r_B))."""
    lhs = M @ (np.asarray(d_pre, dtype=np.float64) / np.asarray(xsd, dtype=np.float64))
    rhs = topk_projection(Um, r_b, k)
    den = float(np.linalg.norm(lhs) * np.linalg.norm(rhs))
    if den <= 0.0:
        raise ValueError("frame_fold_cos: degenerate operands")
    return float(lhs @ rhs / den)


def random_direction(d, *, seed, n_avg=N_RANDOM_SEEDS):
    """VERBATIM issue2254_preimage.py L399 — matched-norm random unit vector,
    mean over ``n_avg`` seeds (the #2220 construction, issue2220_readwrite.py
    L345-361): each seed draws a Gaussian, the mean over seeds is re-normalized
    to unit norm."""
    acc = np.zeros(d, dtype=np.float64)
    for s in range(n_avg):
        rng = np.random.default_rng(seed * 1000 + s)
        v = rng.standard_normal(d)
        acc += v / float(np.linalg.norm(v))
    nrm = float(np.linalg.norm(acc))
    return acc / nrm


def unit_rows(mat):
    """VERBATIM issue2254_preimage.py L414 — row-normalize a (L, H) array,
    failing loud on a degenerate row."""
    m = np.asarray(mat, dtype=np.float64)
    nrm = np.linalg.norm(m, axis=1, keepdims=True)
    if not (np.isfinite(nrm).all() and (nrm > 0.0).all()):
        raise ValueError("unit_rows: degenerate row norm")
    return m / nrm


# ---------------------------------------------------------------------------
# staging (revision-pinned; plan-time local copies first)
# ---------------------------------------------------------------------------


def _stage(path_in_repo: str, target: pathlib.Path, *, revision: str | None) -> pathlib.Path:
    """Local-first staging: the plan-time copy under PLAN_STAGED_ROOT wins,
    else a retried/atomic revision-pinned ``stage_hub_file`` download."""
    local = PLAN_STAGED_ROOT / path_in_repo
    if local.exists():
        return local
    from explore_persona_space.orchestrate.hub import stage_hub_file

    return stage_hub_file(DATA_REPO, path_in_repo, target, repo_type="dataset", revision=revision)


def load_ridge_payload(layer: int, staging_dir: pathlib.Path) -> dict:
    """Load + schema-assert one banked #779 ridge payload (plan-verified keys)."""
    import torch

    rel = f"{RIDGE_PREFIX}/L{layer}/ridge.pt"
    path = _stage(rel, staging_dir / rel, revision=RIDGE_REV)
    # weights_only=False: sha/revision-pinned SELF-PRODUCED bundle (dict of
    # tensors + primitives; plan-time parsed) — the sanctioned torch>=2.6 case.
    payload = torch.load(path, map_location="cpu", weights_only=False)
    required = {"kind", "selected_lambda", "xmu", "xsd", "ymu", "W", "layer", "fitter"}
    missing = required - set(payload)
    if missing:
        raise ValueError(f"ridge payload {path} missing keys {sorted(missing)}")
    if payload["kind"] != "ridge" or int(payload["layer"]) != layer:
        raise ValueError(f"ridge payload mismatch: kind={payload['kind']} layer={payload['layer']}")
    W = payload["W"]
    assert tuple(W.shape) == (lib.HIDDEN_DIM, lib.HIDDEN_DIM), W.shape
    for k in ("xmu", "xsd", "ymu"):
        assert tuple(payload[k].shape) == (lib.HIDDEN_DIM,), (k, payload[k].shape)
    payload["_path"] = str(path)
    return payload


def load_bank_tensor(
    path_in_repo: str,
    staging_dir: pathlib.Path,
    *,
    revision: str | None,
    key: str | None = None,
):
    """Stage + load a (28, 3584) float tensor bank (rb_v2 / r_b / E2 / E3).

    ``key`` unwraps a dict envelope (observed schema of the #779 ``r_b/{trait}.pt``
    payloads: keys ``trait / r_b / layers / counts / smoke / metadata`` with
    ``r_b`` -> (28, 3584); rb_v2 + parent E2/E3 banks are bare tensors).
    A dict payload with no ``key`` (or a missing key) fails loud with the
    observed key set — never a silent guess.
    """
    import torch

    path = _stage(path_in_repo, staging_dir / path_in_repo, revision=revision)
    t = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(t, dict):
        assert key is not None and key in t, (path_in_repo, sorted(t), key)
        t = t[key]
    else:
        assert key is None, f"{path_in_repo}: expected dict envelope with key {key!r}"
    assert tuple(t.shape) == (lib.N_LAYERS, lib.HIDDEN_DIM), (path_in_repo, tuple(t.shape))
    return t.to(torch.float64).numpy()


def _cos(a, b) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    den = float(np.linalg.norm(a) * np.linalg.norm(b))
    if den <= 0.0:
        raise ValueError("_cos: degenerate operands")
    return float(a @ b / den)


def _sha256(path: pathlib.Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# rho probe (1 GPU)
# ---------------------------------------------------------------------------


def _sample_prompts(dataset_root: pathlib.Path, corpus: str, n: int, seed: int) -> list[list]:
    """Seeded sample of n training rows' PROMPT message lists (messages[:-1])."""
    import random

    import issue2225_train as train  # deferred: heavy module, probe-path only

    path = dataset_root / corpus / f"{train.DATASET_VERSION}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"training corpus missing for rho probe: {path}")
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"empty corpus {path}")
    rng = random.Random(seed)
    picked = rng.sample(rows, min(n, len(rows)))
    return [r["messages"][:-1] for r in picked]


def run_probe_rho(args) -> None:
    """Median ||h_l|| at the LAST context token, pooled across the 4 corpora.

    Renders each sampled prompt with apply_chat_template(add_generation_prompt=
    True) — the SAME slot the context_end mask steers and #779's c_last frame —
    and forwards through issue778_lib.capture_last_prompt_token_all_layers.
    Writes {out_dir}/rho.json (pooled + per-corpus medians; #2220 WARN band).
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset_root = pathlib.Path(args.dataset_root)
    device = args.device
    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype).to(device)

    corpora = ("evil", "sycophancy", "hallucination", "mistake_opinions")
    per_corpus_norms: dict[str, np.ndarray] = {}
    for corpus in corpora:
        msgs = _sample_prompts(dataset_root, corpus, args.n_prompts, RHO_SEED)
        prompts = [
            tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
            for m in msgs
        ]
        acts = lib.capture_last_prompt_token_all_layers(model, tokenizer, prompts, device=device)
        norms = torch.linalg.vector_norm(acts.to(torch.float64), dim=-1).numpy()  # (n, 28)
        per_corpus_norms[corpus] = norms
        lib.log_phase(
            "fu1_rho_probe",
            f"{corpus}: n={norms.shape[0]} "
            + " ".join(f"L{layer}={float(np.median(norms[:, layer])):.1f}" for layer in MAP_LAYERS),
        )

    pooled = np.concatenate(list(per_corpus_norms.values()), axis=0)  # (4n, 28)
    rho = {str(layer): float(np.median(pooled[:, layer])) for layer in MAP_LAYERS}
    warnings = []
    lo14, hi14 = RHO_L14_REF_BAND
    if not (0.5 * lo14 <= rho["14"] <= 2.0 * hi14):
        warnings.append(
            f"rho_14={rho['14']:.2f} outside the #2220 WARN band "
            f"[{0.5 * lo14:.1f}, {2.0 * hi14:.1f}] — analyzer-adjudicated (plan §4.1)"
        )
    if not (RHO_L19_EXPECT_BAND[0] <= rho["19"] <= RHO_L19_EXPECT_BAND[1]):
        warnings.append(
            f"rho_19={rho['19']:.2f} outside the secondary L18/L20-bracket band "
            f"{RHO_L19_EXPECT_BAND} — WARN only"
        )
    for w in warnings:
        lib.log_phase("fu1_rho_probe", f"WARN: {w}")
    report = {
        "rho_per_layer": rho,
        "per_corpus_median": {
            c: {str(layer): float(np.median(v[:, layer])) for layer in MAP_LAYERS}
            for c, v in per_corpus_norms.items()
        },
        "per_layer_all": {
            str(layer): float(np.median(pooled[:, layer])) for layer in range(lib.N_LAYERS)
        },
        "n_prompts_per_corpus": {c: int(v.shape[0]) for c, v in per_corpus_norms.items()},
        "seed": RHO_SEED,
        "model": args.model,
        "warnings": warnings,
        "reproducibility": lib.repro_metadata(),
    }
    with open(out_dir / "rho.json", "w") as f:
        json.dump(report, f, indent=2)
    lib.log_phase("fu1_rho_probe", f"done rho={rho} warnings={len(warnings)}")


# ---------------------------------------------------------------------------
# build (CPU algebra on the FULL payloads)
# ---------------------------------------------------------------------------


def build_bank(args) -> None:
    """Plan §4.1 steps 1-6: pre-image + random banks, gates, diagnostics, meta."""
    import torch

    t0 = time.time()
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    staging_dir = pathlib.Path(args.staging_dir)

    rho_path = pathlib.Path(args.rho_json) if args.rho_json else out_dir / "rho.json"
    if not rho_path.exists():
        raise FileNotFoundError(
            f"rho.json missing at {rho_path} — run --probe-rho first (or pass --rho-json)"
        )
    with open(rho_path, encoding="utf-8") as f:
        rho = {int(k): float(v) for k, v in json.load(f)["rho_per_layer"].items()}
    for layer in MAP_LAYERS:
        if layer not in rho or not (np.isfinite(rho[layer]) and rho[layer] > 0):
            raise ValueError(f"rho.json missing/degenerate rho for layer {layer}: {rho}")

    # Per-layer map SVDs (shared across traits — ONE SVD per layer).
    payloads: dict[int, dict] = {}
    svds: dict[int, tuple] = {}
    k_used: dict[int, int] = {}
    for layer in MAP_LAYERS:
        payloads[layer] = load_ridge_payload(layer, staging_dir)
        W = payloads[layer]["W"].to(torch.float64).numpy()
        M, Um, Sm, Vmt = map_svd(W)
        svds[layer] = (M, Um, Sm, Vmt)
        k_used[layer] = int(np.sum(Sm >= RANK_FLOOR_REL * float(Sm[0])))
        lib.log_phase(
            "fu1_directions",
            f"L{layer}: SVD done k_used={k_used[layer]}/{Sm.shape[0]} "
            f"selected_lambda={payloads[layer]['selected_lambda']}",
        )

    rb_v2 = {
        t: load_bank_tensor(f"{RB_V2_PREFIX}/{t}.pt", staging_dir, revision=RB_V2_REV)
        for t in TRAITS
    }
    r_b_779 = {
        t: load_bank_tensor(
            f"{R_B_779_PREFIX}/{t}.pt", staging_dir, revision=R_B_779_REV, key="r_b"
        )
        for t in TRAITS
    }
    parent_dirs = {
        (t, v): load_bank_tensor(
            f"{PARENT_DIRECTIONS_PREFIX}/{t}_{v}.pt", staging_dir, revision=None
        )
        for t in TRAITS
        for v in ("E2", "E3")
    }

    # Random control directions: one per layer (#2220 construction, plan seeds).
    rnd_by_layer = {
        layer: random_direction(lib.HIDDEN_DIM, seed=RANDOM_SEED_BY_LAYER[layer])
        for layer in MAP_LAYERS
    }

    def _nan_bank() -> np.ndarray:
        return np.full((lib.N_LAYERS, lib.HIDDEN_DIM), np.nan, dtype=np.float32)

    def _write_bank(name: str, rows: dict[int, np.ndarray], meta: dict) -> None:
        bank = _nan_bank()
        for layer, vec in rows.items():
            scaled = rho[layer] * np.asarray(vec, dtype=np.float64)
            assert np.isfinite(scaled).all(), f"{name}: non-finite row at layer {layer}"
            bank[layer] = scaled.astype(np.float32)
        t = torch.from_numpy(bank)
        for layer in MAP_LAYERS:  # build-time assert on the sliced rows (plan §4.1)
            assert bool(torch.isfinite(t[layer]).all()), (name, layer)
        torch.save(t, out_dir / name)
        stem = name[: -len(".pt")]
        with open(out_dir / f"{stem}_meta.json", "w") as f:
            json.dump(meta, f, indent=2)
        lib.log_phase("fu1_directions", f"wrote {name} rows={sorted(rows)}")

    payload_shas = {
        f"L{layer}": _sha256(pathlib.Path(payloads[layer]["_path"])) for layer in MAP_LAYERS
    }

    # Pre-image banks, per trait.
    for trait in TRAITS:
        rows: dict[int, np.ndarray] = {}
        diag: dict[str, dict] = {}
        for layer in MAP_LAYERS:
            M, Um, Sm, Vmt = svds[layer]
            xsd = payloads[layer]["xsd"].to(torch.float64).numpy()
            r_b = rb_v2[trait][layer]
            w_std = preimage_w(Um, Sm, Vmt, r_b, k_used[layer])
            d_pre = destandardized_direction(xsd, w_std)
            ff = frame_fold_cos(M, Um, xsd, d_pre, r_b, k_used[layer])
            if ff < FRAME_FOLD_MIN_COS:
                raise RuntimeError(
                    f"FRAME-FOLD HALT (plan §4.1 step 4): trait={trait} L{layer} "
                    f"cos={ff:.6f} < {FRAME_FOLD_MIN_COS} — transpose/frame/standardizer bug"
                )
            rank_sweep = {}
            for k in RANK_SWEEP:
                d_k = destandardized_direction(xsd, preimage_w(Um, Sm, Vmt, r_b, k))
                rank_sweep[str(k)] = _cos(d_pre, d_k)
            # Bridge cosine + the +-1 layer-offset detector (plan §12 A2).
            bridge = _cos(rb_v2[trait][layer], r_b_779[trait][layer])
            bridge_offsets = {
                str(off): _cos(rb_v2[trait][layer], r_b_779[trait][layer + off])
                for off in (-1, 1)
                if 0 <= layer + off < lib.N_LAYERS
            }
            if not all(bridge > v for v in bridge_offsets.values()):
                # Plan §12 A2 says ASSERT: a same-layer bridge cosine beaten by
                # a +-1 offset means the rb_v2 bank rows are layer-shifted vs
                # the #779 readout frame — shipping such banks trains every
                # cell against the wrong layer's direction. Smoke measured
                # aligned 0.981/0.982 vs offsets 0.878-0.936, so a trip here
                # is a genuine off-by-one, never noise.
                raise RuntimeError(
                    f"bridge cosine offset detector (plan §12 A2): trait={trait} "
                    f"L{layer} aligned={bridge:.4f} offsets={bridge_offsets} — "
                    "a +-1 layer offset beats the aligned bridge; HALT"
                )
            diag[f"L{layer}"] = {
                "k_used": k_used[layer],
                "frame_fold_cos": ff,
                "rank_sweep_cos_vs_primary": rank_sweep,
                "cos_d_pre_r_b": _cos(d_pre, r_b),
                "cos_d_pre_parent_E2": _cos(d_pre, parent_dirs[(trait, "E2")][layer]),
                "cos_d_pre_parent_E3": _cos(d_pre, parent_dirs[(trait, "E3")][layer]),
                "cos_d_pre_random": _cos(d_pre, rnd_by_layer[layer]),
                "bridge_cos_rbv2_r779": bridge,
                "bridge_cos_layer_offsets": bridge_offsets,
                "rho": rho[layer],
                "unit_norm_check": float(np.linalg.norm(d_pre)),
            }
            rows[layer] = d_pre
        meta = {
            "trait": trait,
            "variant": "PRE",
            "layers": list(MAP_LAYERS),
            "construction": "M=W^T; d_pre=normalize(xsd * pinv_k(M) r_B[l]); rows = rho_l * d_pre",
            "ridge_payload_rev": RIDGE_REV,
            "ridge_payload_sha256": payload_shas,
            "rb_v2_rev": RB_V2_REV,
            "r_b_779_rev": R_B_779_REV,
            "frame_fold_min_cos": FRAME_FOLD_MIN_COS,
            "rank_floor_rel": RANK_FLOOR_REL,
            "per_layer": diag,
            "rho_source": str(rho_path),
            "wall_s": round(time.time() - t0, 1),
            "reproducibility": lib.repro_metadata(),
        }
        _write_bank(f"{trait}_PRE.pt", rows, meta)

    # Shared random bank (trait-agnostic; one bank, per-layer rows).
    rnd_meta = {
        "variant": "RND",
        "layers": list(MAP_LAYERS),
        "construction": "random_direction(3584, seed=2225014|2225019, n_avg=3); rows = rho_l * v",
        "seeds": {str(layer): RANDOM_SEED_BY_LAYER[layer] for layer in MAP_LAYERS},
        "per_layer": {
            f"L{layer}": {
                "rho": rho[layer],
                "cos_random_L14_L19": _cos(rnd_by_layer[14], rnd_by_layer[19]),
                "unit_norm_check": float(np.linalg.norm(rnd_by_layer[layer])),
            }
            for layer in MAP_LAYERS
        },
        "rho_source": str(rho_path),
        "reproducibility": lib.repro_metadata(),
    }
    _write_bank("RND.pt", dict(rnd_by_layer), rnd_meta)
    lib.log_phase("fu1_directions", f"build done elapsed={round(time.time() - t0, 1)}s")


# Parent-default-identical seam: production callers pass no prefix.
# UPLOAD_PREFIX_EXEMPT: dispatch smoke threads fu1_directions_smoke via --hf-prefix
def upload_bank(out_dir: pathlib.Path, hf_prefix: str = FU1_DIRECTIONS_HF_PREFIX) -> str:
    """One upload_folder commit -> the fu1 directions HF prefix (never the parent's).

    The dispatcher's EPM_I2225_SMOKE branch threads a ``_smoke``-suffixed
    prefix: smoke banks carry a smoke-dial rho (n_prompts=16) and must never
    land at the production prefix.
    """
    from explore_persona_space.orchestrate.hub import _upload

    return _upload(
        pathlib.Path(out_dir),
        DATA_REPO,
        "dataset",
        hf_prefix,
        raise_on_error=True,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Issue #2225 fu1 F0: pre-image direction bank.")
    ap.add_argument(
        "--out-dir", default="eval_results/issue_2225/fu1_preimage_prevention/directions"
    )
    ap.add_argument("--staging-dir", default="data/issue_2225/hf_dl/fu1_inputs")
    ap.add_argument("--dataset-root", default="external/persona_vectors/dataset")
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--probe-rho", action="store_true", help="1-GPU rho probe -> rho.json")
    ap.add_argument("--build", action="store_true", help="CPU algebra -> bank + meta")
    ap.add_argument("--upload", action="store_true", help="upload the out-dir (pod-side)")
    # UPLOAD_PREFIX_EXEMPT: parent-default-identical seam; smoke passes _smoke prefix
    ap.add_argument(
        "--hf-prefix",
        default=FU1_DIRECTIONS_HF_PREFIX,
        help="HF prefix for the bank upload (dispatch smoke threads fu1_directions_smoke)",
    )
    ap.add_argument("--n-prompts", type=int, default=RHO_N_PROMPTS_DEFAULT)
    ap.add_argument("--rho-json", default=None, help="explicit rho.json path (default: out-dir)")
    ap.add_argument("--import-check", action="store_true")
    return ap


def main(argv: Sequence[str] | None = None) -> None:
    ap = build_argparser()
    args = ap.parse_args(argv)

    if args.import_check:
        from explore_persona_space.orchestrate.argcheck import assert_args_attributes_defined

        assert_args_attributes_defined(__file__)
        import torch  # noqa: F401
        from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: F401

        from explore_persona_space.orchestrate.hub import _upload, stage_hub_file  # noqa: F401

        # Ported-algebra round-trip on a tiny synthetic map (shape sanity only;
        # the full-rank frame-fold identity is pinned in tests).
        rng = np.random.default_rng(0)
        W = rng.standard_normal((8, 8))
        M, Um, Sm, Vmt = map_svd(W)
        r = rng.standard_normal(8)
        d = destandardized_direction(np.ones(8), preimage_w(Um, Sm, Vmt, r, 8))
        assert frame_fold_cos(M, Um, np.ones(8), d, r, 8) > 0.999
        print("[issue2225-fu1-directions] import-check OK", flush=True)
        raise SystemExit(0)

    ran = False
    if args.probe_rho:
        run_probe_rho(args)
        ran = True
    if args.build:
        build_bank(args)
        ran = True
    if args.upload:
        # UPLOAD_PREFIX_EXEMPT: parent-default-identical seam; smoke passes _smoke prefix
        url = upload_bank(pathlib.Path(args.out_dir), hf_prefix=args.hf_prefix)
        lib.log_phase("fu1_directions", f"uploaded -> {url}")
        ran = True
    if not ran:
        ap.error("pass at least one of --probe-rho / --build / --upload")


if __name__ == "__main__":
    main()
