#!/usr/bin/env python3
"""Issue #2546 P2a frozen n1m read (plan v4 §4.2 P2a).

Applies the BANKED #779 n1m context->answer ridge maps (Qwen2.5-7B-Instruct,
layers 14/19/26, n_train 963,444, lambda 1e-3) FROZEN — zero fitting — to the
arm-1 pilot capture of gsm8k_test1319:

- read A: ``M_n1m(v_C^post)`` vs the post answer state (``ans_mean`` of the
  OpenThinker3-7B post-side capture) — labeled cross-model transfer-confounded
  (the map was fit on Qwen2.5-7B-Instruct activations).
- read B: ``M_n1m(v_C^pre)``  vs the SAME post answer state (the pre side IS
  Qwen2.5-7B-Instruct, the map's own fit model; the target side stays
  cross-model).

Metrics per layer x read: R²-style agreement (no refit), plain-identity
baseline agreement (v̂ = v_C; the learned-bias identity form is inapplicable —
a frozen read has no fit split to learn b on, stated per the CLAUDE.md
identity-baseline rule), and kNN retrieval acc@1 (euclidean + cosine,
chance = 1/n_pool) via ``mapping_baselines.knn_retrieval``; per-k-bin
breakdowns keep the FULL pool (chance unchanged across bins).

Weight application goes through ``issue779_ffc_n1m_fits.apply_map`` (the
registered ``vhat = ((v_C - xmu)/xsd) @ W + ymu`` path) — never reimplemented.

Outputs ``<out-root>/out/n1m_read/gsm8k_test1319_read.json``, every headline
labeled FROZEN / cross-model transfer-confounded.
"""

from __future__ import annotations

import os

os.environ.setdefault("HF_HOME", os.environ.get("HF_HOME", "/workspace/.cache/huggingface"))

import argparse
import logging
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv(str(PROJECT_ROOT / ".env"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

# Cross-script helpers hoisted to module top (gotchas.md #606: no deferred imports).
from issue779_ffc_n1m_fits import apply_map  # noqa: E402
from issue2546_gen_capture import arm_dirname  # noqa: E402

from explore_persona_space.analysis.mapping_baselines import knn_retrieval  # noqa: E402
from explore_persona_space.atomic_io import write_json_atomic  # noqa: E402
from explore_persona_space.orchestrate.hub import (  # noqa: E402
    DEFAULT_DATASET_REPO,
    stage_hub_file,
)

logger = logging.getLogger("issue2546_n1m_read")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

FROZEN_LAYERS = (14, 19, 26)
HIDDEN = 3584
N1M_HF_PREFIX = "issue779_monitoring/n1m_readout/weights"
N1M_LOCAL_MIRROR = (
    PROJECT_ROOT / "data" / "issue_2094" / "joint_transport" / "banked_maps" / N1M_HF_PREFIX
)
# Fit provenance (Source: scripts/issue2474_n1m_map.py N1M_PROVENANCE; #779 monitoring line)
N1M_PROVENANCE = {
    "fit_issue": 779,
    "fit_model": "Qwen/Qwen2.5-7B-Instruct",
    "n_train": 963_444,
    "selected_lambda": 0.001,
    "whole_map_r2_L19": 0.7542,
    "hf_prefix": N1M_HF_PREFIX,
}
SMOKE_ROW_CAP = 64


def _git_sha() -> str:
    env_sha = os.environ.get("EPS_GIT_SHA")
    if env_sha:
        return env_sha
    p = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    return p.stdout.strip() if p.returncode == 0 else "unavailable-no-git-checkout"


def repro_meta() -> dict:
    return {
        "task": 2546,
        "phase": "p2a_n1m_read",
        "git_commit": _git_sha(),
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "env": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "numpy": np.__version__,
        },
    }


def load_n1m_payload(layer: int) -> dict:
    """Stage + validate one banked ridge payload (local mirror first, else HF)."""
    rel = f"L{layer}/ridge.pt"
    local = N1M_LOCAL_MIRROR / rel
    if not local.is_file():
        logger.info("[n1m] %s not local — staging from HF %s", rel, N1M_HF_PREFIX)
        stage_hub_file(DEFAULT_DATASET_REPO, f"{N1M_HF_PREFIX}/{rel}", local, repo_type="dataset")
    assert local.is_file(), f"n1m weights missing after staging: {local}"
    # Self-produced pinned bundle: dict payload requires weights_only=False
    # (torch>=2.6 flipped the default; the #779 payloads are plain dicts).
    payload = torch.load(local, map_location="cpu", weights_only=False)
    expect_keys = {"W", "fitter", "kind", "layer", "selected_lambda", "xmu", "xsd", "ymu"}
    assert set(payload.keys()) == expect_keys, (
        f"L{layer} payload keys drifted: {sorted(payload.keys())} != {sorted(expect_keys)}"
    )
    assert payload["kind"] == "ridge", payload["kind"]
    assert int(payload["layer"]) == layer, (payload["layer"], layer)
    assert tuple(payload["W"].shape) == (HIDDEN, HIDDEN), payload["W"].shape
    return payload


def load_side_vectors(
    out_root: Path, side: str, kinds_needed: tuple[str, ...], smoke: bool
) -> tuple[dict[str, dict[int, np.ndarray]], list[dict]]:
    """Read the pilot capture shards for one side of arm 1 / gsm8k_test.

    Returns ({kind: {layer: (n, H) fp32}}, meta rows aligned to n). Consumes
    the P4-format bf16 shards directly (kind-tensor stems; upload-then-free
    has not yet fired at P2a — the pilot keeps local shards by design).
    Store dir resolves via the producer's OWN arm_dirname helper (r2 Major 5:
    a hardcoded "arm1" reads past the smoke_arm1 namespace in smoke mode).
    """
    stem_dir = out_root / "store" / arm_dirname(1, smoke) / f"{side}__gsm8k_test"
    shards = sorted(stem_dir.glob("slot*.shard*.pt"))
    if not shards:
        raise FileNotFoundError(
            f"no capture shards under {stem_dir} — run the arm-1 pilot capture first"
        )
    parts: dict[str, dict[int, list[np.ndarray]]] = {
        k: {layer: [] for layer in FROZEN_LAYERS} for k in kinds_needed
    }
    metas: list[dict] = []
    row_ids: list[str] = []
    for sp in shards:
        shard = torch.load(sp, map_location="cpu", weights_only=False)
        assert shard["arm"] == 1 and shard["corpus"] == "gsm8k_test", (
            shard["arm"],
            shard["corpus"],
        )
        kinds_full = shard["kinds_full"]
        full = shard["full"]  # (B, K, L_all, H) bf16
        assert full.shape[-1] == HIDDEN, full.shape
        for k in kinds_needed:
            ki = kinds_full.index(k)
            for layer in FROZEN_LAYERS:
                parts[k][layer].append(full[:, ki, layer, :].float().numpy())
        metas.extend(shard["meta"])
        row_ids.extend(shard["row_ids"])
    out = {
        k: {layer: np.concatenate(chunks, axis=0) for layer, chunks in by_layer.items()}
        for k, by_layer in parts.items()
    }
    n = len(row_ids)
    for k in kinds_needed:
        for layer in FROZEN_LAYERS:
            assert out[k][layer].shape == (n, HIDDEN), (k, layer, out[k][layer].shape)
            assert np.isfinite(out[k][layer]).all(), f"non-finite {side}/{k}/L{layer}"
    for m, rid in zip(metas, row_ids, strict=True):
        m["row_id"] = rid
    logger.info("[load] %s: %d rows x %d shards", side, n, len(shards))
    return out, metas


def r2_agreement(vhat: np.ndarray, y: np.ndarray) -> float:
    """R²-style agreement of a FROZEN prediction (no refit): 1 - SSE/SST."""
    resid = float(((y - vhat) ** 2).sum())
    tot = float(((y - y.mean(axis=0, keepdims=True)) ** 2).sum())
    assert tot > 0, "degenerate target (zero variance)"
    return 1.0 - resid / tot


def knn_block(vhat: np.ndarray, y: np.ndarray, pool: np.ndarray, idx: np.ndarray) -> dict:
    """acc@1 under both metrics; pool = FULL row set (chance = 1/n_pool)."""
    out = {}
    for metric in ("euclidean", "cosine"):
        res = knn_retrieval(vhat, y, ks=(1,), metric=metric, pool=pool, true_pool_idx=idx)
        out[metric] = res
    return out


def eval_read(
    vhat_by_layer: dict[int, np.ndarray],
    target_by_layer: dict[int, np.ndarray],
    vc_by_layer: dict[int, np.ndarray],
    k_bins: list[str],
) -> dict:
    """Per-layer agreement + retrieval + per-k-bin breakdown for one read."""
    bins = sorted(set(k_bins))
    per_layer: dict[str, dict] = {}
    for layer in FROZEN_LAYERS:
        vhat, y, vc = vhat_by_layer[layer], target_by_layer[layer], vc_by_layer[layer]
        n = y.shape[0]
        idx_all = np.arange(n)
        row = {
            "n_rows": n,
            "r2_agreement": r2_agreement(vhat, y),
            "identity_r2_agreement": r2_agreement(vc, y),
            "identity_bias_note": (
                "learned-bias identity (v̂ = x + b) inapplicable: frozen read, no fit "
                "split to learn b on; plain identity reported instead"
            ),
            "knn": knn_block(vhat, y, y, idx_all),
            "chance_at_1": 1.0 / n,
            "per_k_bin": {},
        }
        for b in bins:
            mask = np.array([kb == b for kb in k_bins])
            nb = int(mask.sum())
            if nb < 5:
                row["per_k_bin"][b] = {"n_rows": nb, "skipped": "n < 5"}
                continue
            row["per_k_bin"][b] = {
                "n_rows": nb,
                "r2_agreement": r2_agreement(vhat[mask], y[mask]),
                "knn_full_pool": knn_block(vhat[mask], y[mask], y, idx_all[mask]),
            }
        per_layer[f"L{layer}"] = row
    return per_layer


def run_selftest() -> None:
    """Toy-shape exercise of the two reused helpers (signature + arithmetic)."""
    rng = np.random.default_rng(0)
    d, n = 8, 12
    W = rng.normal(size=(d, d))
    xmu, ymu = rng.normal(size=d), rng.normal(size=d)
    xsd = np.abs(rng.normal(size=d)) + 0.5
    payload = {
        "W": torch.tensor(W),
        "xmu": torch.tensor(xmu),
        "xsd": torch.tensor(xsd),
        "ymu": torch.tensor(ymu),
        "kind": "ridge",
        "fitter": "selftest",
        "layer": 0,
        "selected_lambda": 0.001,
    }
    X = rng.normal(size=(n, d))
    vhat = apply_map(payload, X, torch.device("cpu"))
    manual = ((X - xmu) / xsd) @ W + ymu
    assert vhat.shape == (n, d), vhat.shape
    assert np.allclose(vhat, manual, atol=1e-8), float(np.abs(vhat - manual).max())
    # knn_retrieval: identical pred == true must retrieve itself at k=1
    res = knn_retrieval(manual, manual, ks=(1,), metric="euclidean")
    assert res["acc_at_k"][1] == 1.0, res
    assert abs(res["chance_at_k"][1] - 1.0 / n) < 1e-12, res
    res_c = knn_retrieval(manual, manual, ks=(1,), metric="cosine")
    assert res_c["acc_at_k"][1] == 1.0, res_c
    # explicit-pool form (the per-k-bin call shape)
    sub = manual[:5]
    res_p = knn_retrieval(
        sub, sub, ks=(1,), metric="euclidean", pool=manual, true_pool_idx=np.arange(5)
    )
    assert res_p["acc_at_k"][1] == 1.0, res_p
    # r2_agreement sanity: perfect prediction -> 1.0; mean prediction -> 0.0
    assert abs(r2_agreement(manual, manual) - 1.0) < 1e-12
    mean_pred = np.tile(manual.mean(axis=0), (n, 1))
    assert abs(r2_agreement(mean_pred, manual)) < 1e-12
    print(
        "[n1m-selftest] PASS: apply_map formula match, knn_retrieval (both metrics "
        "+ explicit pool), r2_agreement endpoints",
        flush=True,
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-root", default="/workspace/issue2546")
    ap.add_argument("--smoke", action="store_true", help=f"cap rows at {SMOKE_ROW_CAP} (rehearsal)")
    ap.add_argument(
        "--import-check",
        action="store_true",
        help="argparse-attribute completeness + helper-call bind check",
    )
    ap.add_argument(
        "--selftest",
        action="store_true",
        help="toy-shape apply_map + knn_retrieval exercise (offline)",
    )
    args = ap.parse_args(argv)
    if args.import_check:
        from explore_persona_space.orchestrate.argcheck import assert_args_attributes_defined

        assert_args_attributes_defined(__file__)
        raise SystemExit(0)
    if args.selftest:
        run_selftest()
        raise SystemExit(0)

    print("[phase=p2a_n1m_read]", flush=True)
    out_root = Path(args.out_root)
    payloads = {layer: load_n1m_payload(layer) for layer in FROZEN_LAYERS}
    post, post_meta = load_side_vectors(out_root, "post", ("cx_last", "ans_mean"), bool(args.smoke))
    pre, pre_meta = load_side_vectors(out_root, "pre", ("cx_last",), bool(args.smoke))
    post_by_id = {m["row_id"]: i for i, m in enumerate(post_meta)}
    pre_by_id = {m["row_id"]: i for i, m in enumerate(pre_meta)}
    # A duplicated row_id would silently LAST-WIN the dict and misalign the
    # post/pre join — fail loud instead of joining on a corrupted key space.
    if len(post_by_id) != len(post_meta):
        raise RuntimeError(f"duplicate row_ids in POST capture: {len(post_meta) - len(post_by_id)}")
    if len(pre_by_id) != len(pre_meta):
        raise RuntimeError(f"duplicate row_ids in PRE capture: {len(pre_meta) - len(pre_by_id)}")
    shared = sorted(set(post_by_id) & set(pre_by_id))
    assert shared, "post/pre pilot captures share ZERO row_ids — upstream capture broke"
    if args.smoke:
        shared = shared[:SMOKE_ROW_CAP]
    pi = np.array([post_by_id[r] for r in shared])
    qi = np.array([pre_by_id[r] for r in shared])
    k_bins = [str(post_meta[post_by_id[r]].get("k_bin")) for r in shared]
    logger.info(
        "[main] %d shared rows (smoke=%s); k-bins: %s",
        len(shared),
        bool(args.smoke),
        sorted(set(k_bins)),
    )

    dev = torch.device("cpu")
    target = {layer: post["ans_mean"][layer][pi] for layer in FROZEN_LAYERS}
    vc_post = {layer: post["cx_last"][layer][pi] for layer in FROZEN_LAYERS}
    vc_pre = {layer: pre["cx_last"][layer][qi] for layer in FROZEN_LAYERS}
    vhat_post = {layer: apply_map(payloads[layer], vc_post[layer], dev) for layer in FROZEN_LAYERS}
    vhat_pre = {layer: apply_map(payloads[layer], vc_pre[layer], dev) for layer in FROZEN_LAYERS}

    result = {
        "task": 2546,
        "read": "p2a_frozen_n1m_gsm8k_test1319",
        "n_rows": len(shared),
        "smoke": bool(args.smoke),
        "labels": [
            "FROZEN — banked #779 n1m ridge applied with zero fitting",
            "cross-model transfer-confounded — map fit on Qwen2.5-7B-Instruct; the "
            "post-side inputs AND the answer-state target are OpenThinker3-7B "
            "activations (read A input + both reads' target); read B's input is the "
            "map's own fit model",
        ],
        "map_provenance": N1M_PROVENANCE,
        "reads": {
            "post_vc": eval_read(vhat_post, target, vc_post, k_bins),
            "pre_vc": eval_read(vhat_pre, target, vc_pre, k_bins),
        },
        "repro": repro_meta(),
    }
    dest = out_root / "out" / "n1m_read" / "gsm8k_test1319_read.json"
    # atomic_io: process-unique temp + same-dir os.replace (#2336).
    write_json_atomic(dest, result, ensure_ascii=True)
    for layer in FROZEN_LAYERS:
        a = result["reads"]["post_vc"][f"L{layer}"]
        b = result["reads"]["pre_vc"][f"L{layer}"]
        print(
            f"[n1m-read] L{layer}: post r2={a['r2_agreement']:.4f} "
            f"knn1_eu={a['knn']['euclidean']['acc_at_k'][1]:.4f} | "
            f"pre r2={b['r2_agreement']:.4f} "
            f"knn1_eu={b['knn']['euclidean']['acc_at_k'][1]:.4f} (chance {a['chance_at_1']:.5f})",
            flush=True,
        )
    print(f"[n1m-read] wrote {dest}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
