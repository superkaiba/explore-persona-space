"""Issue #1482 — BatchTopK SAE loader/encoder for ``andyrdt/saes-qwen2.5-7b-instruct``.

Greenfield by necessity (plan §10: the repo has NO SAE infrastructure). Loads the
dictionary_learning-format BatchTopK SAE (Bussmann/Leask/Nanda, arXiv 2412.06410)
without the sae_lens / dictionary_learning packages: the state dict is a plain
torch save whose REALIZED key set was probed against the artifact itself at the
pinned revision (check (c), 2026-07-17):

    b_dec (3584,) fp32 | k () int32 | threshold () fp32 SCALAR
    decoder.weight (3584, 131072) | encoder.weight (131072, 3584) | encoder.bias (131072,)

Inference-time thresholding uses the learned SCALAR ``threshold`` (per 2412.06410
BatchTopK inference; NOT batch top-k, NOT a per-feature vector — asserted at load).

Encode/decode convention (dictionary_learning BatchTopKSAE — verified VERBATIM
against ``andyrdt/dictionary_learning@andyrdt/qwen`` ``trainers/batch_top_k.py``
lines 37-59, r3 source read 2026-07-17):
    f      = relu((x - b_dec) @ W_enc.T + b_enc);  f = f * (f > threshold)
    x_hat  = f @ W_dec.T + b_dec

Input scale (r3 source read): the suite was trained with
``trainSAE(..., normalize_activations=True)`` (``run_from_config.py:145``), and
``training.py`` FOLDS the norm factor into the released weights at the final save
(``ae.scale_biases(norm_factor)`` before ``t.save`` — ``training.py:241-246``), so
the released ``ae.pt`` consumes RAW residual-stream activations. ``config.json``
carries NO normalization field (the ``trainer.config["norm_factor"]`` write in
``training.py:194`` lands in a discarded temporary dict).

TOKEN-POOL semantics (r3 root cause — the #1482 P2-pilot FVE ∈ [-7900, -3400] /
L0 253-2708 incident): the suite's training AND its published eval
(FVE 0.806 / L0 60 at k=64) both run under ``remove_bos=True``
(``run_from_config.py:199,210``), which per ``buffer.py``:
  (a) drops the FIRST ``BOS_OFFSET = 8`` token positions of every context
      ("a subset of which have super large activations" — buffer.py:13,142-147);
  (b) drops token rows with L2 norm > 10x the pool median
      (``outlier_norm_factor = 10.0`` — buffer.py:150-156, "this unfortunately
      seems necessary for Qwen2.5-7B-Instruct");
and the published FVE is the VAR-based read ``1 - var(x - x_hat)/var(x)``
(per-dim unbiased variance, summed — ``evaluation.py:231-233``). Feeding the
excluded massive-activation tokens through the encoder explodes L0 (~42k features
on a 30x-norm row; locally reproduced) and drives FVE to -10^3 — the exact pilot
signature. ``fve_l0`` therefore implements the reference eval semantics (b)+(c);
sequence-level consumers apply (a) via ``BOS_OFFSET`` before pooling.

Fail-loud: key-set / shape / config asserts at load; no silent fallbacks.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))
from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()  # shared-VM thread caps bind BEFORE torch import (#847)

import torch  # noqa: E402

logger = logging.getLogger("issue1482_sae")

SAE_REPO = "andyrdt/saes-qwen2.5-7b-instruct"
# Pinned at implementation time (repo main sha, resolved 2026-07-17).
SAE_REVISION = "c37e53c4bb07127ad17ab88f28b93d4e87142e59"
ACT_DIM = 3584
DICT_SIZE = 131_072
# Suite coverage + k -> trainer-index map (early-layer-arm generalization, #1482 plan v16
# §4 item 1): resid_post layers {3,7,11,15,19,23,27}; trainers 0..3 = k in {32,64,128,256}.
# Verified at L3 AND consistent with the L19 registrations below (Hub configs read at the
# pinned revision, 2026-07-28: L3 trainer_1 k=64 / trainer_2 k=128).
SUITE_LAYERS = (3, 7, 11, 15, 19, 23, 27)
TRAINER_INDEX_BY_K = {32: 0, 64: 1, 128: 2, 256: 3}


def trainer_subdir(layer: int, k: int) -> str:
    """Repo subfolder for one (layer, k) trainer; fail-loud off the suite grid."""
    assert layer in SUITE_LAYERS, f"layer {layer} not in suite {SUITE_LAYERS}"
    assert k in TRAINER_INDEX_BY_K, f"k {k} not in {sorted(TRAINER_INDEX_BY_K)}"
    return f"resid_post_layer_{layer}/trainer_{TRAINER_INDEX_BY_K[k]}"


# Legacy L19 map — kept so existing callers stay byte-compatible
# (issue1482_error_analysis.phase_p2 `sorted(TRAINER_SUBDIR)`, the _cli choices).
TRAINER_SUBDIR = {k: trainer_subdir(19, k) for k in (64, 128)}
EXPECTED_KEYS = {"b_dec", "k", "threshold", "decoder.weight", "encoder.weight", "encoder.bias"}
# Published suite eval (each trainer's eval_results.json) — Gate B calibration source.
# Keyed layer -> k (JSON-serializable; L3 values Hub-read at the pinned revision
# 2026-07-28: frac_variance_explained 0.93087890625 / 0.94208984375).
PUBLISHED_FVE_BY_LAYER = {
    3: {64: 0.93087890625, 128: 0.94208984375},
    19: {64: 0.80572265625, 128: 0.84236328125},
}
PUBLISHED_FVE = PUBLISHED_FVE_BY_LAYER[19]  # legacy alias (existing L19 callers)

# Reference token-pool constants (andyrdt/dictionary_learning@andyrdt/qwen buffer.py —
# see module docstring "TOKEN-POOL semantics"). Applied by fve_l0 (outlier filter)
# and by sequence-level consumers (BOS strip) so fitness reads are calibrated
# against the suite's own published eval.
BOS_OFFSET = 8  # buffer.py:13 — first 8 positions carry Qwen massive activations
OUTLIER_NORM_FACTOR = 10.0  # buffer.py:151 — drop rows with norm > 10x pool median


@torch.no_grad()
def token_inlier_mask(h: torch.Tensor, *, median_norm: float | None = None) -> torch.Tensor:
    """Reference outlier mask over token rows: bool (T,), True = keep.

    Keeps rows with L2 norm <= OUTLIER_NORM_FACTOR x median (buffer.py:150-156).
    The median is computed over the given rows unless supplied. Does NOT apply
    the BOS strip — position-level, the caller owns it (``h[BOS_OFFSET:]``).
    """
    assert h.ndim == 2, tuple(h.shape)
    norms = h.float().norm(dim=1)
    med = float(norms.median()) if median_norm is None else float(median_norm)
    return norms <= OUTLIER_NORM_FACTOR * med


class BatchTopKSAE:
    """Minimal inference-only BatchTopK SAE (encode / decode / fve)."""

    def __init__(
        self,
        state_dict: dict,
        k: int,
        device: str = "cpu",
        *,
        act_dim: int = ACT_DIM,
        dict_size: int = DICT_SIZE,
    ):
        keys = set(state_dict.keys())
        assert keys == EXPECTED_KEYS, (
            f"ae.pt key set drift: {sorted(keys)} != {sorted(EXPECTED_KEYS)}"
        )
        w_enc = state_dict["encoder.weight"]
        b_enc = state_dict["encoder.bias"]
        w_dec = state_dict["decoder.weight"]
        b_dec = state_dict["b_dec"]
        thr = state_dict["threshold"]
        assert tuple(w_enc.shape) == (dict_size, act_dim), w_enc.shape
        assert tuple(w_dec.shape) == (act_dim, dict_size), w_dec.shape
        assert tuple(b_enc.shape) == (dict_size,), b_enc.shape
        assert tuple(b_dec.shape) == (act_dim,), b_dec.shape
        # A5 resolution: the inference threshold is a learned SCALAR (0-dim tensor).
        assert thr.ndim == 0, f"threshold expected SCALAR, got shape {tuple(thr.shape)}"
        assert int(state_dict["k"]) == k, (int(state_dict["k"]), k)
        self.k = k
        self.act_dim = act_dim
        self.dict_size = dict_size
        self.device = device
        self.w_enc = w_enc.to(device=device, dtype=torch.float32)
        self.b_enc = b_enc.to(device=device, dtype=torch.float32)
        self.w_dec = w_dec.to(device=device, dtype=torch.float32)
        self.b_dec = b_dec.to(device=device, dtype=torch.float32)
        self.threshold = float(thr)

    @classmethod
    def load(
        cls,
        k: int = 64,
        device: str = "cpu",
        cache_dir: Path | str | None = None,
        *,
        layer: int = 19,
    ):
        """Download (revision-pinned) + load one (layer, k) trainer; asserts config + keys.

        ``layer`` defaults to 19 so every pre-existing caller stays byte-compatible
        (#1482 early-layer-arm generalization; the layer assert generalizes with it).
        Both fetches ride ``hub.retry_transient`` (#1402: Retry-After-aware,
        ``EPM_HF_RETRY_BUDGET_S`` wall budget, LocalEntryNotFoundError transient
        BY CLASS) — this is P2-child-reachable, and a Hub queue-full 429 storm
        killed #1482 attempt 2's p2 child on a sibling bare download."""
        from huggingface_hub import hf_hub_download

        from explore_persona_space.orchestrate import hub

        sub = trainer_subdir(layer, k)
        kw = {"revision": SAE_REVISION, "repo_type": "model"}
        if cache_dir is not None:
            kw["local_dir"] = str(cache_dir)
        cfg_path = hub.retry_transient(
            lambda: hf_hub_download(SAE_REPO, f"{sub}/config.json", **kw),
            what=f"sae config fetch ({SAE_REPO}:{sub})",
        )
        cfg = json.loads(Path(cfg_path).read_text())["trainer"]
        assert cfg["dict_class"] == "BatchTopKSAE", cfg["dict_class"]
        assert cfg["activation_dim"] == ACT_DIM and cfg["dict_size"] == DICT_SIZE, cfg
        assert cfg["k"] == k and cfg["layer"] == layer, (cfg["k"], cfg["layer"], layer)
        assert cfg["lm_name"] == "Qwen/Qwen2.5-7B-Instruct", cfg["lm_name"]
        ae_path = hub.retry_transient(
            lambda: hf_hub_download(SAE_REPO, f"{sub}/ae.pt", **kw),
            what=f"sae weights fetch ({SAE_REPO}:{sub})",
        )
        sd = torch.load(ae_path, map_location="cpu", mmap=True, weights_only=True)
        logger.info(
            "[sae] loaded k=%d from %s@%s (threshold scalar)", k, SAE_REPO, SAE_REVISION[:8]
        )
        return cls(sd, k=k, device=device)

    @classmethod
    def ensure_downloaded(cls, k: int, cache_dir: Path | str, *, layer: int = 19) -> None:
        """Idempotently stage the (``layer``, ``k``) trainer's ``config.json`` +
        ``ae.pt`` into ``cache_dir`` (parent-side pre-stage for fan-out consumers;
        ``layer`` defaults to 19 for existing-caller byte-compatibility).

        N concurrent workers calling ``load()`` against ONE empty shared
        ``local_dir`` race the download scratch (#1315 fanout shared-staging
        class; #1482 r5: the ``--seed-partial`` path skipped the P2-pilot —
        previously the sole staging path — and a p2 worker died at
        ``torch.load(ae.pt)`` FileNotFoundError). ``hf_hub_download`` publishes
        only COMPLETE files at the final ``local_dir`` path (atomic replace),
        so present+non-empty == complete; missing files ride
        ``hub.retry_transient`` exactly like ``load()``.
        """
        from huggingface_hub import hf_hub_download

        from explore_persona_space.orchestrate import hub

        sub = trainer_subdir(layer, k)
        cache = Path(cache_dir)
        staged, present = [], []
        for fname in ("config.json", "ae.pt"):
            target = cache / sub / fname
            if target.exists() and target.stat().st_size > 0:
                present.append(fname)
                continue
            hub.retry_transient(
                lambda f=fname: hf_hub_download(
                    SAE_REPO,
                    f"{sub}/{f}",
                    revision=SAE_REVISION,
                    repo_type="model",
                    local_dir=str(cache),
                ),
                what=f"sae pre-stage ({SAE_REPO}:{sub}/{fname})",
            )
            staged.append(fname)
        logger.info(
            "[sae] pre-stage k=%d -> %s (staged=%s present=%s)", k, cache / sub, staged, present
        )

    @torch.no_grad()
    def encode(self, h: torch.Tensor, chunk: int = 2048) -> torch.Tensor:
        """(T, act_dim) activations -> (T, dict_size) thresholded-ReLU features (fp32).

        Chunked over rows so the (chunk, dict_size) buffer bounds peak memory.
        """
        assert h.ndim == 2 and h.shape[1] == self.act_dim, tuple(h.shape)
        outs = []
        for s in range(0, h.shape[0], chunk):
            x = h[s : s + chunk].to(device=self.device, dtype=torch.float32) - self.b_dec
            f = torch.relu(x @ self.w_enc.T + self.b_enc)
            f = f * (f > self.threshold)
            outs.append(f)
        return torch.cat(outs) if len(outs) != 1 else outs[0]

    @torch.no_grad()
    def decode(self, f: torch.Tensor) -> torch.Tensor:
        """(T, dict_size) features -> (T, act_dim) reconstruction (fp32)."""
        assert f.ndim == 2 and f.shape[1] == self.dict_size, tuple(f.shape)
        return f.to(device=self.device, dtype=torch.float32) @ self.w_dec.T + self.b_dec

    @torch.no_grad()
    def fve_l0(self, h: torch.Tensor, chunk: int = 2048) -> tuple[float, float, dict]:
        """Reference-parity reconstruction fitness -> (fve, l0, diag).

        Implements the suite's OWN eval semantics (the Gate B calibration source —
        see module docstring "TOKEN-POOL semantics"): drops rows with L2 norm >
        OUTLIER_NORM_FACTOR x pool median (buffer.py:150-156), then
        FVE = 1 - sum_d var(x - x_hat)_d / sum_d var(x)_d (per-dim UNBIASED
        variance, evaluation.py:231-233) and L0 = mean nnz, both over kept rows.
        fp64 accumulators (massive-dim means make fp32 sum-of-squares cancel).
        Sequence-level callers strip the first BOS_OFFSET positions per sequence
        BEFORE pooling rows into ``h`` (buffer.py remove_bos).

        diag: n_rows / n_inlier / n_outlier_dropped / median_norm (pre-filter).
        Raises ValueError when fewer than 2 inlier rows remain (variance undefined).
        """
        assert h.ndim == 2 and h.shape[1] == self.act_dim, tuple(h.shape)
        h32 = h.to(device=self.device, dtype=torch.float32)
        norms = h32.norm(dim=1)
        med = float(norms.median())
        keep = norms <= OUTLIER_NORM_FACTOR * med
        n_dropped = int((~keep).sum())
        hk = h32[keep]
        n = int(hk.shape[0])
        if n < 2:
            raise ValueError(
                f"fve_l0: only {n} inlier rows (need >= 2 for variance); "
                f"n_rows={int(h.shape[0])} n_outlier_dropped={n_dropped}"
            )
        x_sum = torch.zeros(self.act_dim, dtype=torch.float64, device=self.device)
        x_sq = torch.zeros_like(x_sum)
        r_sum = torch.zeros_like(x_sum)
        r_sq = torch.zeros_like(x_sum)
        l0_sum = 0.0
        for s in range(0, n, chunk):
            x = hk[s : s + chunk]
            f = self.encode(x, chunk=chunk)
            r = x - self.decode(f)
            x_sum += x.sum(0, dtype=torch.float64)
            x_sq += (x * x).sum(0, dtype=torch.float64)
            r_sum += r.sum(0, dtype=torch.float64)
            r_sq += (r * r).sum(0, dtype=torch.float64)
            l0_sum += float((f > 0).sum())

        def _var_sum(ssum: torch.Tensor, ssq: torch.Tensor) -> float:
            return float(((ssq - ssum * ssum / n) / (n - 1)).sum())

        ss_tot = _var_sum(x_sum, x_sq)
        fve = float("nan") if ss_tot < 1e-12 else 1.0 - _var_sum(r_sum, r_sq) / ss_tot
        diag = {
            "n_rows": int(h.shape[0]),
            "n_inlier": n,
            "n_outlier_dropped": n_dropped,
            "median_norm": round(med, 2),
        }
        return fve, l0_sum / n, diag


@torch.no_grad()
def pool_answer_features(f: torch.Tensor) -> dict[str, torch.Tensor]:
    """Pool per-token features (T, F) over the token axis -> the Goal-v3 trio.

    Returns dense (F,) fp32 tensors: mean activation, MAX activation, and
    fraction-active (share of tokens with the feature > 0).
    """
    assert f.ndim == 2 and f.shape[0] >= 1, tuple(f.shape)
    return {
        "mean": f.mean(0),
        "max": f.max(0).values,
        "frac": (f > 0).to(torch.float32).mean(0),
    }


def sparsify(pooled: dict[str, torch.Tensor]) -> dict[str, object]:
    """Union-index sparse encoding of the pooled trio (shared int32 idx + fp16 values)."""
    import numpy as np

    union = None
    for v in pooled.values():
        nz = v != 0
        union = nz if union is None else (union | nz)
    idx = torch.nonzero(union, as_tuple=False).squeeze(-1)
    out = {"idx": idx.cpu().numpy().astype(np.int32)}
    for name, v in pooled.items():
        out[name] = v[idx].cpu().numpy().astype(np.float16)
    return out


# ── SAELens matryoshka jumprelu (issue #1482 matryoshka-tier round) — ADDITIVE ───────
# Everything in this section is NEW for the matryoshka-tier round (plan v21 §4 item 1);
# the BatchTopKSAE surface above is byte-untouched (check-(i) source-module convention;
# the round's consistency check certified this module additive-only).

SAELENS_REPO = "chanind/qwen2.5-7B-it-layer-20-saes"
# Pinned at plan time (repo main sha, Hub-resolved 2026-07-29; plan v21 §2 availability gate).
SAELENS_REVISION = "db8c88b400e36e603d0563abcf8d289e5eff5687"
SAELENS_DICT_SIZE = 65_536
SAELENS_SAE_IDS = ("lmsys/matryoshka/k-100", "pile/matryoshka/k-100")
SAELENS_EXPECTED_KEYS = {"W_enc", "W_dec", "b_enc", "b_dec", "threshold"}
# Training-imposed granularity tiers (runner_cfg.json matryoshka_widths, Hub-read at the
# pinned revision; arXiv 2503.17547 nested-prefix semantics): feature-id bins
# [0, 2048) / [2048, 16384) / [16384, 65536) -> tier 0 / 1 / 2.
MATRYOSHKA_TIER_BOUNDS = (2048, 16384, 65536)


def tier_of(ids):
    """Matryoshka tier index (0/1/2) per feature id — deterministic id bins, no fit
    input (plan §11 R2). Returns an int8 numpy array; fail-loud outside [0, 65536)."""
    import numpy as np

    arr = np.asarray(ids, dtype=np.int64)
    assert arr.size == 0 or ((arr >= 0).all() and (arr < MATRYOSHKA_TIER_BOUNDS[-1]).all()), (
        "feature id outside [0, 65536)"
    )
    bounds = np.asarray(MATRYOSHKA_TIER_BOUNDS[:-1], dtype=np.int64)
    return np.searchsorted(bounds, arr, side="right").astype(np.int8)


def _cfg_field(cfg: dict, key: str):
    """cfg.json field lookup — top level first, then the SAELens ``metadata`` nest;
    fail-loud on absence (never a silent default)."""
    if key in cfg:
        return cfg[key]
    meta = cfg.get("metadata")
    if isinstance(meta, dict) and key in meta:
        return meta[key]
    raise KeyError(f"cfg.json field {key!r} absent (top level and metadata)")


@torch.no_grad()
def _reference_fve_l0(sae, h: torch.Tensor, chunk: int = 2048) -> tuple[float, float, dict]:
    """Reference-parity FVE/L0 for any encode/decode SAE object (the BatchTopKSAE.fve_l0
    semantics generalized so the SAELens class reuses them verbatim — see the module
    docstring "TOKEN-POOL semantics"): drop token rows with L2 norm >
    OUTLIER_NORM_FACTOR x pool median, then var-based FVE (per-dim unbiased variance,
    summed) + mean-nnz L0 over kept rows; fp64 accumulators. Sequence-level callers
    strip the first BOS_OFFSET positions BEFORE pooling rows into ``h``.
    Raises ValueError below 2 inlier rows (variance undefined)."""
    assert h.ndim == 2 and h.shape[1] == sae.act_dim, tuple(h.shape)
    h32 = h.to(device=sae.device, dtype=torch.float32)
    norms = h32.norm(dim=1)
    med = float(norms.median())
    keep = norms <= OUTLIER_NORM_FACTOR * med
    n_dropped = int((~keep).sum())
    hk = h32[keep]
    n = int(hk.shape[0])
    if n < 2:
        raise ValueError(
            f"fve_l0: only {n} inlier rows (need >= 2 for variance); "
            f"n_rows={int(h.shape[0])} n_outlier_dropped={n_dropped}"
        )
    x_sum = torch.zeros(sae.act_dim, dtype=torch.float64, device=sae.device)
    x_sq = torch.zeros_like(x_sum)
    r_sum = torch.zeros_like(x_sum)
    r_sq = torch.zeros_like(x_sum)
    l0_sum = 0.0
    for s in range(0, n, chunk):
        x = hk[s : s + chunk]
        f = sae.encode(x, chunk=chunk)
        r = x - sae.decode(f)
        x_sum += x.sum(0, dtype=torch.float64)
        x_sq += (x * x).sum(0, dtype=torch.float64)
        r_sum += r.sum(0, dtype=torch.float64)
        r_sq += (r * r).sum(0, dtype=torch.float64)
        l0_sum += float((f > 0).sum())

    def _var_sum(ssum: torch.Tensor, ssq: torch.Tensor) -> float:
        return float(((ssq - ssum * ssum / n) / (n - 1)).sum())

    ss_tot = _var_sum(x_sum, x_sq)
    fve = float("nan") if ss_tot < 1e-12 else 1.0 - _var_sum(r_sum, r_sq) / ss_tot
    diag = {
        "n_rows": int(h.shape[0]),
        "n_inlier": n,
        "n_outlier_dropped": n_dropped,
        "median_norm": round(med, 2),
    }
    return fve, l0_sum / n, diag


class SAELensJumpReLU:
    """Minimal inference-only SAELens jumprelu SAE (the chanind L20 matryoshka save).

    Encode/decode semantics VERBATIM from the pinned SAELens v6.37.6 tag
    (``sae_lens/saes/jumprelu_sae.py``, source fetched at plan time — plan §11 R7):

        sae_in     = x - b_dec                    # apply_b_dec_to_input: true (asserted)
        hidden_pre = sae_in @ W_enc + b_enc
        acts       = relu(hidden_pre) * (hidden_pre > threshold)
        recon      = acts @ W_dec + b_dec

    fp32 throughout; NO runtime decoder-norm rescaling — the training-time
    ``rescale_acts_by_decoder_norm: true`` is folded into the saved jumprelu weights
    at conversion (SAELens ``_fold_norm_topk``), and the inference cfg.json declares
    plain ``jumprelu`` with ``normalize_activations: "none"`` (both asserted at load).
    Duck-types ``BatchTopKSAE``'s encode/decode/fve_l0 tensor contract so the shared
    ``_row_features`` / pooling consumers run unchanged."""

    def __init__(self, cfg: dict, state_dict: dict, device: str = "cpu"):
        assert _cfg_field(cfg, "architecture") == "jumprelu", _cfg_field(cfg, "architecture")
        assert int(_cfg_field(cfg, "d_in")) == ACT_DIM, _cfg_field(cfg, "d_in")
        assert int(_cfg_field(cfg, "d_sae")) == SAELENS_DICT_SIZE, _cfg_field(cfg, "d_sae")
        assert _cfg_field(cfg, "apply_b_dec_to_input") is True, "apply_b_dec_to_input != true"
        assert _cfg_field(cfg, "normalize_activations") == "none", _cfg_field(
            cfg, "normalize_activations"
        )
        keys = set(state_dict.keys())
        assert keys == SAELENS_EXPECTED_KEYS, (
            f"sae_weights key drift: {sorted(keys)} != {sorted(SAELENS_EXPECTED_KEYS)}"
        )
        w_enc = state_dict["W_enc"]
        w_dec = state_dict["W_dec"]
        b_enc = state_dict["b_enc"]
        b_dec = state_dict["b_dec"]
        thr = state_dict["threshold"]
        assert tuple(w_enc.shape) == (ACT_DIM, SAELENS_DICT_SIZE), w_enc.shape
        assert tuple(w_dec.shape) == (SAELENS_DICT_SIZE, ACT_DIM), w_dec.shape
        assert tuple(b_enc.shape) == (SAELENS_DICT_SIZE,), b_enc.shape
        assert tuple(b_dec.shape) == (ACT_DIM,), b_dec.shape
        # Per-feature jumprelu threshold VECTOR (the saved-inference format; NOT the
        # BatchTopK scalar — asserted so a save-format drift fails loud here).
        assert tuple(thr.shape) == (SAELENS_DICT_SIZE,), thr.shape
        self.act_dim = ACT_DIM
        self.dict_size = SAELENS_DICT_SIZE
        self.device = device
        self.w_enc = w_enc.to(device=device, dtype=torch.float32)
        self.w_dec = w_dec.to(device=device, dtype=torch.float32)
        self.b_enc = b_enc.to(device=device, dtype=torch.float32)
        self.b_dec = b_dec.to(device=device, dtype=torch.float32)
        self.threshold = thr.to(device=device, dtype=torch.float32)

    @staticmethod
    def _files(sae_id: str) -> tuple[str, str]:
        assert sae_id in SAELENS_SAE_IDS, f"unknown matryoshka sae_id {sae_id!r}"
        return f"{sae_id}/cfg.json", f"{sae_id}/sae_weights.safetensors"

    @classmethod
    def load(
        cls,
        sae_id: str,
        device: str = "cpu",
        cache_dir: Path | str | None = None,
        *,
        repo_id: str = SAELENS_REPO,
        revision: str = SAELENS_REVISION,
    ):
        """Download (revision-pinned) + load one matryoshka variant; asserts cfg + keys.

        Both fetches ride ``hub.retry_transient`` (the P2-child 429-storm class)."""
        from huggingface_hub import hf_hub_download
        from safetensors import safe_open

        from explore_persona_space.orchestrate import hub

        cfg_name, w_name = cls._files(sae_id)
        kw = {"revision": revision, "repo_type": "model"}
        if cache_dir is not None:
            kw["local_dir"] = str(cache_dir)
        cfg_path = hub.retry_transient(
            lambda: hf_hub_download(repo_id, cfg_name, **kw),
            what=f"saelens cfg fetch ({repo_id}:{sae_id})",
        )
        cfg = json.loads(Path(cfg_path).read_text())
        w_path = hub.retry_transient(
            lambda: hf_hub_download(repo_id, w_name, **kw),
            what=f"saelens weights fetch ({repo_id}:{sae_id})",
        )
        sd: dict = {}
        with safe_open(w_path, framework="pt", device="cpu") as f:
            for k in f.keys():  # noqa: SIM118 — safetensors handle, not a dict
                sd[k] = f.get_tensor(k)
        logger.info("[sae] loaded %s from %s@%s (jumprelu)", sae_id, repo_id, revision[:8])
        return cls(cfg, sd, device=device)

    @classmethod
    def ensure_downloaded(
        cls,
        sae_id: str,
        cache_dir: Path | str,
        *,
        repo_id: str = SAELENS_REPO,
        revision: str = SAELENS_REVISION,
    ) -> None:
        """Idempotently stage the variant's cfg + weights into ``cache_dir`` (the
        BatchTopKSAE.ensure_downloaded parent-pre-stage convention; #1315 class)."""
        from huggingface_hub import hf_hub_download

        from explore_persona_space.orchestrate import hub

        cache = Path(cache_dir)
        staged, present = [], []
        for fname in cls._files(sae_id):
            target = cache / fname
            if target.exists() and target.stat().st_size > 0:
                present.append(fname)
                continue
            hub.retry_transient(
                lambda f=fname: hf_hub_download(
                    repo_id, f, revision=revision, repo_type="model", local_dir=str(cache)
                ),
                what=f"saelens pre-stage ({repo_id}:{fname})",
            )
            staged.append(fname)
        logger.info(
            "[sae] pre-stage %s -> %s (staged=%s present=%s)", sae_id, cache, staged, present
        )

    @torch.no_grad()
    def encode(self, h: torch.Tensor, chunk: int = 2048) -> torch.Tensor:
        """(T, act_dim) activations -> (T, dict_size) jumprelu features (fp32).

        Chunked over rows so the (chunk, dict_size) buffer bounds peak memory."""
        assert h.ndim == 2 and h.shape[1] == self.act_dim, tuple(h.shape)
        outs = []
        for s in range(0, h.shape[0], chunk):
            x = h[s : s + chunk].to(device=self.device, dtype=torch.float32) - self.b_dec
            pre = x @ self.w_enc + self.b_enc
            outs.append(torch.relu(pre) * (pre > self.threshold))
        return torch.cat(outs) if len(outs) != 1 else outs[0]

    @torch.no_grad()
    def decode(self, f: torch.Tensor) -> torch.Tensor:
        """(T, dict_size) features -> (T, act_dim) reconstruction (fp32)."""
        assert f.ndim == 2 and f.shape[1] == self.dict_size, tuple(f.shape)
        return f.to(device=self.device, dtype=torch.float32) @ self.w_dec + self.b_dec

    @torch.no_grad()
    def fve_l0(self, h: torch.Tensor, chunk: int = 2048) -> tuple[float, float, dict]:
        """Reference-parity reconstruction fitness (Gate B-m mechanism) — the shared
        token-pool semantics via ``_reference_fve_l0``."""
        return _reference_fve_l0(self, h, chunk=chunk)


def fetch_saelens_sparsity(
    sae_id: str,
    cache_dir: Path | str | None = None,
    *,
    repo_id: str = SAELENS_REPO,
    revision: str = SAELENS_REVISION,
) -> torch.Tensor:
    """The variant's training ``sparsity.safetensors`` (single tensor, key-agnostic —
    asserted exactly one key of shape (65536,)): the Gate B-m sparsity-correlation
    convention fingerprint's reference (diagnostic only, plan §4 M0 (b))."""
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open

    from explore_persona_space.orchestrate import hub

    assert sae_id in SAELENS_SAE_IDS, f"unknown matryoshka sae_id {sae_id!r}"
    kw = {"revision": revision, "repo_type": "model"}
    if cache_dir is not None:
        kw["local_dir"] = str(cache_dir)
    path = hub.retry_transient(
        lambda: hf_hub_download(repo_id, f"{sae_id}/sparsity.safetensors", **kw),
        what=f"saelens sparsity fetch ({repo_id}:{sae_id})",
    )
    with safe_open(path, framework="pt", device="cpu") as f:
        keys = list(f.keys())
        assert len(keys) == 1, f"sparsity.safetensors carries {len(keys)} keys: {keys}"
        t = f.get_tensor(keys[0]).to(torch.float32).reshape(-1)
    assert tuple(t.shape) == (SAELENS_DICT_SIZE,), t.shape
    return t


# ── local fitness check (VM-side; the r3 p2-pilot-local verification command) ────────
_CAPTURE_PREFIX = "issue779_monitoring/fitter-fair-comparison-n1m/final_token_capture"
_DATA_REPO = "superkaiba1/explore-persona-space-data"


def _cli() -> None:
    """Local reconstruction check on STORED parent capture rows (no GPU).

    Stages N capture chunks + the revision-pinned SAE under --scratch, then prints
    the reference-parity FVE/L0 on stored cx_last@--layer (and, with --legacy, the
    pre-r3 unfiltered/uncentered read for comparison). Digest-only output.
    """
    import argparse

    from huggingface_hub import hf_hub_download

    ap = argparse.ArgumentParser(description=_cli.__doc__)
    ap.add_argument("--scratch", type=Path, required=True, help="staging dir (data disk)")
    ap.add_argument("--chunks", type=int, default=2, help="number of shard00 chunks")
    ap.add_argument("--k", type=int, default=64, choices=sorted(TRAINER_SUBDIR))
    ap.add_argument("--layer", type=int, default=19)
    ap.add_argument("--field", default="cx_last", choices=("cx_last", "v_x"))
    ap.add_argument("--legacy", action="store_true", help="ALSO print the pre-r3 (unfiltered) read")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parts = []
    for i in range(args.chunks):
        p = hf_hub_download(
            _DATA_REPO,
            filename=f"{_CAPTURE_PREFIX}/shard00_chunk{i:04d}.pt",
            repo_type="dataset",
            local_dir=str(args.scratch),
        )
        b = torch.load(p, map_location="cpu", weights_only=True)
        col = list(b["layers"]).index(args.layer)
        parts.append(b[args.field][:, col, :].to(torch.float32))
    x = torch.cat(parts)
    sae = BatchTopKSAE.load(k=args.k, device="cpu", cache_dir=args.scratch / "sae")
    fve, l0, diag = sae.fve_l0(x)
    print(
        f"[sae-local] {args.field}@L{args.layer} k={args.k} fve={fve:.4f} l0={l0:.2f} "
        f"diag={diag} published_fve={PUBLISHED_FVE[args.k]}"
    )
    if args.legacy:
        xhat = sae.decode(sae.encode(x))
        mu = x.mean(0)
        fve_legacy = 1.0 - float(((x - xhat) ** 2).sum()) / float(((x - mu) ** 2).sum())
        l0_legacy = float((sae.encode(x) > 0).sum()) / x.shape[0]
        print(
            f"[sae-local] LEGACY (pre-r3, unfiltered/uncentered) "
            f"fve={fve_legacy:.4f} l0={l0_legacy:.2f}"
        )


if __name__ == "__main__":
    _cli()
