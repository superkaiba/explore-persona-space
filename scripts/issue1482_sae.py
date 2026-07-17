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

Encode/decode convention (dictionary_learning BatchTopKSAE):
    f      = relu((x - b_dec) @ W_enc.T + b_enc);  f = f * (f > threshold)
    x_hat  = f @ W_dec.T + b_dec

Fail-loud: key-set / shape / config asserts at load; no silent fallbacks.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import torch

logger = logging.getLogger("issue1482_sae")

SAE_REPO = "andyrdt/saes-qwen2.5-7b-instruct"
# Pinned at implementation time (repo main sha, resolved 2026-07-17).
SAE_REVISION = "c37e53c4bb07127ad17ab88f28b93d4e87142e59"
ACT_DIM = 3584
DICT_SIZE = 131_072
TRAINER_SUBDIR = {64: "resid_post_layer_19/trainer_1", 128: "resid_post_layer_19/trainer_2"}
EXPECTED_KEYS = {"b_dec", "k", "threshold", "decoder.weight", "encoder.weight", "encoder.bias"}
# Published suite eval (each trainer's eval_results.json) — Gate B calibration source.
PUBLISHED_FVE = {64: 0.80572265625, 128: 0.84236328125}


class BatchTopKSAE:
    """Minimal inference-only BatchTopK SAE (encode / decode / fve)."""

    def __init__(self, state_dict: dict, k: int, device: str = "cpu"):
        keys = set(state_dict.keys())
        assert keys == EXPECTED_KEYS, (
            f"ae.pt key set drift: {sorted(keys)} != {sorted(EXPECTED_KEYS)}"
        )
        w_enc = state_dict["encoder.weight"]
        b_enc = state_dict["encoder.bias"]
        w_dec = state_dict["decoder.weight"]
        b_dec = state_dict["b_dec"]
        thr = state_dict["threshold"]
        assert tuple(w_enc.shape) == (DICT_SIZE, ACT_DIM), w_enc.shape
        assert tuple(w_dec.shape) == (ACT_DIM, DICT_SIZE), w_dec.shape
        assert tuple(b_enc.shape) == (DICT_SIZE,), b_enc.shape
        assert tuple(b_dec.shape) == (ACT_DIM,), b_dec.shape
        # A5 resolution: the inference threshold is a learned SCALAR (0-dim tensor).
        assert thr.ndim == 0, f"threshold expected SCALAR, got shape {tuple(thr.shape)}"
        assert int(state_dict["k"]) == k, (int(state_dict["k"]), k)
        self.k = k
        self.device = device
        self.w_enc = w_enc.to(device=device, dtype=torch.float32)
        self.b_enc = b_enc.to(device=device, dtype=torch.float32)
        self.w_dec = w_dec.to(device=device, dtype=torch.float32)
        self.b_dec = b_dec.to(device=device, dtype=torch.float32)
        self.threshold = float(thr)

    @classmethod
    def load(cls, k: int = 64, device: str = "cpu", cache_dir: Path | str | None = None):
        """Download (revision-pinned) + load one trainer; asserts config + key set."""
        from huggingface_hub import hf_hub_download

        sub = TRAINER_SUBDIR[k]
        kw = {"revision": SAE_REVISION, "repo_type": "model"}
        if cache_dir is not None:
            kw["local_dir"] = str(cache_dir)
        cfg_path = hf_hub_download(SAE_REPO, f"{sub}/config.json", **kw)
        cfg = json.loads(Path(cfg_path).read_text())["trainer"]
        assert cfg["dict_class"] == "BatchTopKSAE", cfg["dict_class"]
        assert cfg["activation_dim"] == ACT_DIM and cfg["dict_size"] == DICT_SIZE, cfg
        assert cfg["k"] == k and cfg["layer"] == 19, (cfg["k"], cfg["layer"])
        assert cfg["lm_name"] == "Qwen/Qwen2.5-7B-Instruct", cfg["lm_name"]
        ae_path = hf_hub_download(SAE_REPO, f"{sub}/ae.pt", **kw)
        sd = torch.load(ae_path, map_location="cpu", mmap=True, weights_only=True)
        logger.info(
            "[sae] loaded k=%d from %s@%s (threshold scalar)", k, SAE_REPO, SAE_REVISION[:8]
        )
        return cls(sd, k=k, device=device)

    @torch.no_grad()
    def encode(self, h: torch.Tensor, chunk: int = 2048) -> torch.Tensor:
        """(T, 3584) activations -> (T, 131072) thresholded-ReLU features (fp32).

        Chunked over rows so the (chunk, DICT_SIZE) buffer bounds peak memory.
        """
        assert h.ndim == 2 and h.shape[1] == ACT_DIM, tuple(h.shape)
        outs = []
        for s in range(0, h.shape[0], chunk):
            x = h[s : s + chunk].to(device=self.device, dtype=torch.float32) - self.b_dec
            f = torch.relu(x @ self.w_enc.T + self.b_enc)
            f = f * (f > self.threshold)
            outs.append(f)
        return torch.cat(outs) if len(outs) != 1 else outs[0]

    @torch.no_grad()
    def decode(self, f: torch.Tensor) -> torch.Tensor:
        """(T, 131072) features -> (T, 3584) reconstruction (fp32)."""
        assert f.ndim == 2 and f.shape[1] == DICT_SIZE, tuple(f.shape)
        return f.to(device=self.device, dtype=torch.float32) @ self.w_dec.T + self.b_dec

    @torch.no_grad()
    def fve_l0(self, h: torch.Tensor, chunk: int = 2048) -> tuple[float, float]:
        """Reconstruction FVE (1 - ||x-x_hat||^2 / ||x-mean||^2) + mean L0 over rows."""
        assert h.ndim == 2 and h.shape[1] == ACT_DIM, tuple(h.shape)
        h32 = h.to(device=self.device, dtype=torch.float32)
        mu = h32.mean(0)
        ss_res, l0_sum = 0.0, 0.0
        for s in range(0, h32.shape[0], chunk):
            x = h32[s : s + chunk]
            f = self.encode(x, chunk=chunk)
            xhat = self.decode(f)
            ss_res += float(((x - xhat) ** 2).sum())
            l0_sum += float((f > 0).sum())
        ss_tot = float(((h32 - mu) ** 2).sum())
        fve = float("nan") if ss_tot < 1e-12 else 1.0 - ss_res / ss_tot
        return fve, l0_sum / h32.shape[0]


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
