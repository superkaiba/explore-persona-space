"""Phase 1 merger — assemble D_matrix.json + D_per_position.json + C_L*.json.

Issue #406 plan v9 §4 Phase 1 merge step.

Reads per-q' shard outputs at eval_results/issue_406/divergence/per_q/q_*.pt
(written by both GPU shards of scripts/i406_phase1_compute_divergence.py)
and produces:

  1. eval_results/issue_406/divergence/D_matrix.json
       Primary K=25-mean KL[i, j], JS[i, j] (symmetric), prompt_token_lengths
       per (i, j), K_available_per_probe (per-probe ceilings), and condition
       metadata.
  2. eval_results/issue_406/divergence/D_per_position.json (v9-NEW)
       Per-position KL trajectory tensor of shape (380, 25): 380 ordered
       pairs x 25 positions. Plus an N-mask flagging positions where the
       set of contributing probes is reduced because K_available < k for
       at least one probe.
  3. eval_results/issue_406/cosine/C_L{0,5,11,15,21,27}.json (6 files)
       Per-layer 20x20 cosine distance matrices from per-context mean
       activation across 50 probes at last-prompt-token position.

Idempotent: re-running picks up whatever per_q files are present. Missing
files are reported and the merger refuses to write outputs if fewer than
50 per_q files are loaded.
"""

from __future__ import annotations

import json
import logging
import math
import subprocess
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from explore_persona_space.experiments.i406_conditions import CONDITIONS

logger = logging.getLogger("i406.phase1.merge")

PER_Q_DIR = Path("eval_results/issue_406/divergence/per_q")
OUT_DIV_DIR = Path("eval_results/issue_406/divergence")
OUT_COS_DIR = Path("eval_results/issue_406/cosine")
EXPECTED_N_PROBES = 50
TARGET_LAYERS = [0, 5, 11, 15, 21, 27]
K_TARGET = 25


def _git_commit_hash() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
        return out.decode().strip()
    except Exception:
        return "unknown"


def _kl_per_position(log_p: torch.Tensor, log_q: torch.Tensor) -> torch.Tensor:
    """KL(P || Q) per token position, given (K, V) log-softmaxes.

    Returns a (K,) tensor of per-position KL values.
    """
    p = log_p.exp()
    kl = (p * (log_p - log_q)).sum(dim=-1)  # (K,)
    return kl


def _js_per_position(log_p: torch.Tensor, log_q: torch.Tensor) -> torch.Tensor:
    """JS divergence per token position via logaddexp(log_p, log_q) - ln 2."""
    log_m = torch.logaddexp(log_p, log_q) - math.log(2.0)
    kl_pm = _kl_per_position(log_p, log_m)
    kl_qm = _kl_per_position(log_q, log_m)
    return 0.5 * (kl_pm + kl_qm)


def _matrix_to_nested_dict(
    matrix: np.ndarray, cids: list[str]
) -> dict[str, dict[str, float | None]]:
    """Serialize a (n_cond, n_cond) ndarray to {ci: {cj: value | None}}.

    NaN cells (diagonal i==j) become None in JSON.
    """
    out: dict[str, dict[str, float | None]] = {}
    for i, ci in enumerate(cids):
        out[ci] = {}
        for j, cj in enumerate(cids):
            v = float(matrix[i, j]) if not np.isnan(matrix[i, j]) else None
            out[ci][cj] = v
    return out


def _accumulate_pairwise_divergences(
    payloads: list[dict],
    cids: list[str],
    n_cond: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict[str, dict[str, int]],
    dict[str, dict[str, int]],
    list[int],
]:
    """Walk per-q' payloads, accumulate per-(i, j, position) KL + JS sums.

    Per CLAUDE.md plan §3 Knob 4, the prompt-length covariate uses T_j (the
    eval-side transformation; the predictor is computed against the same
    (T_i, T_j) pair so length is intrinsic). Averaged across 50 probes.

    Returns (kl_sum, kl_cnt, js_sum, js_cnt, prompt_tok_sum, prompt_tok_cnt,
    k_available_per_probe). All arrays shape (n_cond, n_cond, K_TARGET); dicts
    keyed (ci, cj).
    """
    kl_per_pos_sum = np.zeros((n_cond, n_cond, K_TARGET), dtype=np.float64)
    kl_per_pos_cnt = np.zeros((n_cond, n_cond, K_TARGET), dtype=np.int64)
    js_per_pos_sum = np.zeros((n_cond, n_cond, K_TARGET), dtype=np.float64)
    js_per_pos_cnt = np.zeros((n_cond, n_cond, K_TARGET), dtype=np.int64)
    k_available_per_probe: list[int] = [p["k_available"] for p in payloads]
    prompt_token_lengths_sum: dict[str, dict[str, int]] = {
        ci: dict.fromkeys(cids, 0) for ci in cids
    }
    prompt_token_lengths_cnt: dict[str, dict[str, int]] = {
        ci: dict.fromkeys(cids, 0) for ci in cids
    }

    for payload in payloads:
        k_av = payload["k_available"]
        if k_av <= 0:
            logger.warning("q_idx=%d has K_available=0; skipping.", payload["q_idx"])
            continue
        log_probs = payload["log_probs"]  # dict[cid -> (K_av, V) tensor]
        prompt_ids = payload["prompt_ids"]  # dict[cid -> list[int]]

        for cj in cids:
            n_tok = len(prompt_ids[cj])
            for ci in cids:
                prompt_token_lengths_sum[ci][cj] += n_tok
                prompt_token_lengths_cnt[ci][cj] += 1

        for i, ci in enumerate(cids):
            log_p_i = log_probs[ci]
            for j, cj in enumerate(cids):
                if i == j:
                    continue
                log_q_j = log_probs[cj]
                kl_k = _kl_per_position(log_p_i, log_q_j)
                js_k = _js_per_position(log_p_i, log_q_j)
                kl_per_pos_sum[i, j, :k_av] += kl_k.numpy().astype(np.float64)
                kl_per_pos_cnt[i, j, :k_av] += 1
                js_per_pos_sum[i, j, :k_av] += js_k.numpy().astype(np.float64)
                js_per_pos_cnt[i, j, :k_av] += 1

    return (
        kl_per_pos_sum,
        kl_per_pos_cnt,
        js_per_pos_sum,
        js_per_pos_cnt,
        prompt_token_lengths_sum,
        prompt_token_lengths_cnt,
        k_available_per_probe,
    )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    OUT_DIV_DIR.mkdir(parents=True, exist_ok=True)
    OUT_COS_DIR.mkdir(parents=True, exist_ok=True)

    per_q_files = sorted(PER_Q_DIR.glob("q_*.pt"))
    if len(per_q_files) < EXPECTED_N_PROBES:
        raise RuntimeError(
            f"Merger refuses to run: found {len(per_q_files)} per_q files in "
            f"{PER_Q_DIR}; expected {EXPECTED_N_PROBES}. Missing q' indices "
            "are likely a Phase-1 shard that crashed mid-loop. Resume the "
            "failed shard via i406_phase1_compute_divergence.py --resume "
            "before re-running the merger."
        )
    if len(per_q_files) > EXPECTED_N_PROBES:
        logger.warning(
            "Found %d per_q files; expected %d. Using all of them; first %d after sort.",
            len(per_q_files),
            EXPECTED_N_PROBES,
            EXPECTED_N_PROBES,
        )

    # Load all per_q payloads up-front. Each is small (per-cond log_probs + 6 layer vectors).
    payloads = []
    for path in per_q_files[:EXPECTED_N_PROBES]:
        payload = torch.load(path, weights_only=False)
        payloads.append(payload)
    n_q = len(payloads)
    logger.info("Loaded %d per_q payloads from %s", n_q, PER_Q_DIR)

    cids = [c.cid for c in CONDITIONS]
    n_cond = len(cids)  # 20
    n_pairs_ordered = n_cond * (n_cond - 1)  # 380

    # ── 1. Primary K=25-mean KL + JS + per-position KL ───────────────────
    (
        kl_per_pos_sum,
        kl_per_pos_cnt,
        js_per_pos_sum,
        js_per_pos_cnt,
        prompt_token_lengths_sum,
        prompt_token_lengths_cnt,
        k_available_per_probe,
    ) = _accumulate_pairwise_divergences(payloads, cids, n_cond)

    # Build per-position trajectory tensor (n_cond * (n_cond - 1), K) using
    # safe division (cnt > 0). For diagonal i == j the entries stay at sentinel.
    with np.errstate(divide="ignore", invalid="ignore"):
        kl_per_pos_mean = np.where(
            kl_per_pos_cnt > 0, kl_per_pos_sum / np.maximum(kl_per_pos_cnt, 1), np.nan
        )
        js_per_pos_mean = np.where(
            js_per_pos_cnt > 0, js_per_pos_sum / np.maximum(js_per_pos_cnt, 1), np.nan
        )

    # Primary K-mean: mean across positions where cnt > 0 (per (i, j)).
    # nanmean drops the NaN positions automatically — exactly the right thing.
    kl_primary = np.nanmean(kl_per_pos_mean, axis=2)  # (20, 20); diag = NaN
    js_primary = np.nanmean(js_per_pos_mean, axis=2)  # (20, 20)

    # Tensor-shape asserts at boundaries (CLAUDE.md).
    assert kl_primary.shape == (n_cond, n_cond), kl_primary.shape
    assert kl_per_pos_mean.shape == (n_cond, n_cond, K_TARGET), kl_per_pos_mean.shape

    # ── 2. Per-position trajectory artifact (380 ordered pairs x 25) ─────
    # ordered_pairs[r] = (T_i, T_j) for row r of the (380, 25) tensor.
    ordered_pairs: list[tuple[str, str]] = []
    per_position_tensor = np.full((n_pairs_ordered, K_TARGET), np.nan, dtype=np.float64)
    per_position_cnt = np.zeros((n_pairs_ordered, K_TARGET), dtype=np.int64)
    r = 0
    for i, ci in enumerate(cids):
        for j, cj in enumerate(cids):
            if i == j:
                continue
            ordered_pairs.append((ci, cj))
            per_position_tensor[r] = kl_per_pos_mean[i, j]
            per_position_cnt[r] = kl_per_pos_cnt[i, j]
            r += 1
    assert r == n_pairs_ordered, (r, n_pairs_ordered)

    # ── 3. Prompt-token-length covariate (i, j) → mean across q' ─────────
    prompt_token_lengths: dict[str, dict[str, float]] = {ci: {} for ci in cids}
    for ci in cids:
        for cj in cids:
            cnt = prompt_token_lengths_cnt[ci][cj]
            prompt_token_lengths[ci][cj] = (
                prompt_token_lengths_sum[ci][cj] / cnt if cnt > 0 else float("nan")
            )

    # ── 4. Persist D_matrix.json + D_per_position.json ───────────────────
    d_matrix_payload = {
        "schema_version": "v9",
        "k_target": K_TARGET,
        "n_conditions": n_cond,
        "n_probes": n_q,
        "conditions": [{"cid": c.cid, "class": c.cls, "name": c.name} for c in CONDITIONS],
        "KL": _matrix_to_nested_dict(kl_primary, cids),
        "JS": _matrix_to_nested_dict(js_primary, cids),
        "prompt_tokens": prompt_token_lengths,
        "k_available_per_probe": k_available_per_probe,
        "git_commit": _git_commit_hash(),
    }
    d_matrix_path = OUT_DIV_DIR / "D_matrix.json"
    d_matrix_path.write_text(json.dumps(d_matrix_payload, indent=2))
    logger.info(
        "Wrote %s (K=%d-mean KL + JS over %d ordered pairs)",
        d_matrix_path,
        K_TARGET,
        n_pairs_ordered,
    )

    d_per_pos_payload = {
        "schema_version": "v9",
        "k_target": K_TARGET,
        "n_pairs": n_pairs_ordered,
        "ordered_pairs": [list(pair) for pair in ordered_pairs],
        # NaN -> None in JSON serialization
        "tensor": [[None if np.isnan(v) else float(v) for v in row] for row in per_position_tensor],
        "n_valid_probes_per_position": per_position_cnt.tolist(),
        "git_commit": _git_commit_hash(),
    }
    d_per_pos_path = OUT_DIV_DIR / "D_per_position.json"
    d_per_pos_path.write_text(json.dumps(d_per_pos_payload, indent=2))
    logger.info(
        "Wrote %s (per-position KL trajectory: %d pairs x %d positions)",
        d_per_pos_path,
        n_pairs_ordered,
        K_TARGET,
    )

    # ── 5. Per-layer cosine matrices ─────────────────────────────────────
    _emit_cosine_matrices(payloads, cids, n_cond, n_q)
    logger.info("Phase 1 merge complete.")


def _emit_cosine_matrices(
    payloads: list[dict],
    cids: list[str],
    n_cond: int,
    n_q: int,
) -> None:
    """For each target layer, build the 20x20 cosine-distance matrix from
    per-context mean activations and persist to eval_results/issue_406/cosine/.
    """
    for layer_idx in TARGET_LAYERS:
        # Build (n_cond, n_q, hidden_dim) tensor.
        # Probes may have failed for some cond — but Phase 1 hook fires on
        # every cond per q', so we expect a full grid.
        per_cond_act_list: dict[str, list[torch.Tensor]] = {ci: [] for ci in cids}
        for payload in payloads:
            acts = payload["activations"]  # dict[cid -> dict[layer -> (H,) tensor]]
            for ci in cids:
                if ci not in acts:
                    raise RuntimeError(
                        f"q_idx={payload['q_idx']} missing activations for cond {ci}"
                    )
                if layer_idx not in acts[ci]:
                    raise RuntimeError(
                        f"q_idx={payload['q_idx']} cond={ci} missing layer {layer_idx}"
                    )
                per_cond_act_list[ci].append(acts[ci][layer_idx])

        # Stack to (n_q, hidden_dim) per cond, then average → (hidden_dim,)
        mean_acts: list[torch.Tensor] = []
        within_cond_l2_std: list[float] = []  # diagnostic per cond
        for ci in cids:
            stacked = torch.stack(per_cond_act_list[ci], dim=0)  # (n_q, H)
            mean_vec = stacked.mean(dim=0)
            mean_acts.append(mean_vec)
            # Diagnostic: residual L2 across probes (Frobenius from the per-cond mean).
            residuals = stacked - mean_vec.unsqueeze(0)
            within_cond_l2_std.append(float(residuals.norm(dim=-1).mean().item()))

        mean_act_matrix = torch.stack(mean_acts, dim=0)  # (n_cond, H)
        assert mean_act_matrix.shape[0] == n_cond, mean_act_matrix.shape
        normed = F.normalize(mean_act_matrix, dim=-1)
        cos_sim = normed @ normed.T  # (n_cond, n_cond)
        cos_dist = (1.0 - cos_sim).cpu().numpy().astype(np.float64)
        assert cos_dist.shape == (n_cond, n_cond), cos_dist.shape

        layer_payload = {
            "schema_version": "v9",
            "layer": layer_idx,
            "n_conditions": n_cond,
            "n_probes": n_q,
            "conditions": [c.cid for c in CONDITIONS],
            "matrix": _matrix_to_nested_dict(cos_dist, cids),
            "within_cond_l2_std_per_cond": dict(zip(cids, within_cond_l2_std, strict=True)),
            "git_commit": _git_commit_hash(),
        }
        layer_path = OUT_COS_DIR / f"C_L{layer_idx}.json"
        layer_path.write_text(json.dumps(layer_payload, indent=2))
        logger.info("Wrote %s (mean-pooled cosine on layer %d)", layer_path, layer_idx)


if __name__ == "__main__":
    main()
