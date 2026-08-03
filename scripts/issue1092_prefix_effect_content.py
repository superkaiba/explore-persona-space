#!/usr/bin/env python3
"""#1092 inline: what IS the prefix contribution to the answer state?

A_p — the per-prefix mean deviation of the answer state (the part consistent
across queries) — is isolated on the dense core (99 prefixes x 48 queries,
t1 target, ambient, L14). Its variance is decomposed into:

  (a) TOPIC-explained: between-topic share over the 12 manifest topic labels,
      with a permutation null (labels shuffled across prefixes) since groups
      are unbalanced;
  (b) TRAIT content: (b1) energy along the banked r_B persona-vector
      directions (sycophancy + hallucination; orthonormalized), vs the
      analytic chance k/d; (b2) energy along the judge-score-predictive
      subspace — cross-fitted ridge from A_p to the monitoring round's
      per-prefix judge means (prediction r reported as sanity);
  (c) residual. (a) and (b) are not disjoint; the joint topic+trait basis
      share is reported alongside.

Analysis-only; writes
eval_results/issue_1092/inline_prefix_effect_content/prefix_effect_content.json.

Usage: uv run python scripts/issue1092_prefix_effect_content.py
"""

from __future__ import annotations

import gc
import json
import sys
import time
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

# Politeness: two sibling analysis runs are live — cap at 4 threads.
import os  # noqa: E402

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_k, "4")
load_dotenv()

import numpy as np  # noqa: E402
import torch  # noqa: E402

SCRIPTS = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPTS.parent
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import issue1092_inline_fair_comparison as fc  # noqa: E402

OUT_DIR = PROJECT_ROOT / "eval_results/issue_1092/inline_prefix_effect_content"
OUT_PATH = OUT_DIR / "prefix_effect_content.json"
RB_DIR = PROJECT_ROOT / "data/issue_1092/hf_dl/issue779_monitoring/r_b"
MONITORING = PROJECT_ROOT / "eval_results/issue_1092/inline_prefixend_monitoring/results.json"
TRAITS = ["sycophancy", "hallucination"]
SEED = 20260723
N_PERM = 1000
HID = 3584  # t1 block only — the single-layer residual space the r_B live in


def _dense_grid(rows: list[dict], n_states: int):
    dense = [
        (i, r["prefix_id"], r["query_id"], r.get("topic", "other"))
        for i, r in enumerate(rows[:n_states])
        if r.get("stratum") == "dense_core"
    ]
    by_prefix: dict[str, dict[str, int]] = {}
    topic: dict[str, str] = {}
    for i, p, q, t in dense:
        by_prefix.setdefault(p, {})[q] = i
        topic[p] = t
    qsets = [set(d) for d in by_prefix.values()]
    shared_q = sorted(set.intersection(*qsets))
    pids = sorted(p for p, d in by_prefix.items() if all(q in d for q in shared_q))
    grid = np.asarray([[by_prefix[p][q] for q in shared_q] for p in pids], dtype=np.int64)
    return pids, grid, [topic[p] for p in pids]


def _orthonormal(dirs: np.ndarray) -> np.ndarray:
    q, _ = np.linalg.qr(dirs.T)
    return q  # (d, k)


def _share_along(A: np.ndarray, Q: np.ndarray) -> float:
    return float(((A @ Q) ** 2).sum() / (A**2).sum())


def _judge_means(cell: str) -> dict[str, dict[str, float]]:
    mon = json.loads(MONITORING.read_text())
    out: dict[str, dict[str, float]] = {}
    for block in mon["cells"][cell]["14"]:
        pp = block["per_prefix"]
        out[block["trait"]] = dict(zip(pp["prefix_id"], pp["judge_mean"], strict=True))
    return out


def process_cell(cell: str, rows: list[dict]) -> dict:
    t0 = time.monotonic()
    t1 = fc._load(cell, "t1")
    pids, grid, topics = _dense_grid(rows, t1.shape[0])
    P, Q = grid.shape
    Y = np.asarray(t1[grid.reshape(-1)], dtype=np.float64).reshape(P, Q, HID)
    del t1
    gc.collect()
    grand = Y.mean(axis=(0, 1))
    A_p = (Y - grand).mean(axis=1)  # (P, HID): the prefix-consistent answer part
    V = float((A_p**2).sum())

    # (a) topic share + permutation null.
    rng = np.random.default_rng(SEED)
    tlabels = np.asarray(topics)

    def topic_share(labels: np.ndarray) -> float:
        s = 0.0
        for t in np.unique(labels):
            m = A_p[labels == t]
            s += m.shape[0] * float((m.mean(0) ** 2).sum())
        return s / V

    obs_topic = topic_share(tlabels)
    null = np.asarray([topic_share(rng.permutation(tlabels)) for _ in range(N_PERM)])
    topic_out = {
        "share": obs_topic,
        "null_mean": float(null.mean()),
        "null_p95": float(np.quantile(null, 0.95)),
        "excess_over_null_mean": obs_topic - float(null.mean()),
        "n_topics": int(np.unique(tlabels).size),
    }

    # (b1) r_B persona-direction share.
    def _rb_l14(t: str) -> np.ndarray:
        payload = torch.load(RB_DIR / f"{t}.pt", map_location="cpu", weights_only=False)
        arr = payload["r_b"] if isinstance(payload, dict) else payload
        arr = np.asarray(arr.detach().cpu().numpy() if hasattr(arr, "detach") else arr, dtype=np.float64)
        assert arr.shape == (28, HID), (t, arr.shape)
        return arr[14]

    rb = np.stack([_rb_l14(t) for t in TRAITS])
    rb = rb / np.linalg.norm(rb, axis=1, keepdims=True)
    Q_rb = _orthonormal(rb)
    rb_out = {
        "share": _share_along(A_p, Q_rb),
        "k": int(Q_rb.shape[1]),
        "chance": Q_rb.shape[1] / HID,
    }

    # (b2) judge-score-predictive subspace (cross-fitted ridge coefficient dirs).
    jm = _judge_means(cell)
    coef_dirs, pred_r = [], {}
    idx = rng.permutation(P)
    halves = [idx[: P // 2], idx[P // 2 :]]
    n_scored = sum(1 for p in pids if p in jm[TRAITS[0]])
    for trait in TRAITS if n_scored >= 20 else []:
        y = np.asarray([jm[trait].get(p, np.nan) for p in pids])
        keep = ~np.isnan(y)
        Ak, yk = A_p[keep], y[keep] - y[keep].mean()
        lam = 10.0 * Ak.shape[0]
        w = np.linalg.solve(Ak.T @ Ak + lam * np.eye(HID), Ak.T @ yk)
        coef_dirs.append(w / np.linalg.norm(w))
        # cross-fitted prediction r (2-fold on kept prefixes)
        ki = np.where(keep)[0]
        preds = np.full(ki.size, np.nan)
        pos = {int(g): j for j, g in enumerate(ki)}
        for h in range(2):
            te = np.asarray([g for g in halves[h] if bool(keep[g])], dtype=int)
            tr = np.asarray([g for g in halves[1 - h] if bool(keep[g])], dtype=int)
            wt = np.linalg.solve(
                A_p[tr].T @ A_p[tr] + lam * np.eye(HID),
                A_p[tr].T @ (y[tr] - y[tr].mean()),
            )
            preds[[pos[int(g)] for g in te]] = A_p[te] @ wt
        pred_r[trait] = float(np.corrcoef(preds, y[ki])[0, 1])
    if coef_dirs:
        Q_j = _orthonormal(np.stack(coef_dirs))
        judge_out = {
            "share": _share_along(A_p, Q_j),
            "k": int(Q_j.shape[1]),
            "chance": Q_j.shape[1] / HID,
            "crossfit_pred_r": pred_r,
            "n_prefixes_with_scores": n_scored,
        }
    else:
        judge_out = {
            "skipped": (
                "no (or <20) dense-core prefixes with judge scores in this cell — the "
                "monitoring round scored battery prefixes only for the base model"
            ),
            "n_prefixes_with_scores": n_scored,
        }

    # Joint topic + trait basis share (topic means + rb + judge dirs, orthonormalized).
    topic_means = np.stack([A_p[tlabels == t].mean(0) for t in np.unique(tlabels)])
    joint_parts = [topic_means, rb] + ([np.stack(coef_dirs)] if coef_dirs else [])
    joint = np.concatenate(joint_parts, axis=0)
    Q_joint = _orthonormal(joint)
    joint_share = _share_along(A_p, Q_joint)

    out = {
        "n_prefixes": P,
        "n_queries": Q,
        "prefix_effect_variance_share_of_A_p": 1.0,
        "topic": topic_out,
        "rb_directions": rb_out,
        "judge_predictive": judge_out,
        "joint_topic_plus_trait_share": {
            "share": joint_share,
            "k": int(Q_joint.shape[1]),
            "chance": Q_joint.shape[1] / HID,
        },
        "wall_s": round(time.monotonic() - t0, 1),
    }
    print(f"[{cell}] {json.dumps(out)[:500]}", flush=True)
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = fc._jsonl(fc.MANIFEST)
    result = {
        "meta": {
            "script": "scripts/issue1092_prefix_effect_content.py",
            "git_commit": fc._git_sha(),
            "layer": fc.LAYER,
            "basis": "ambient, t1 block only (the r_B live in this 3584-dim space)",
            "seed": SEED,
            "n_perm": N_PERM,
            "rb_revision": "037fcbb (issue779_monitoring/r_b)",
        },
        "cells": {},
    }
    for cell in fc.CELLS:
        result["cells"][cell] = process_cell(cell, rows)
        gc.collect()
    OUT_PATH.write_text(json.dumps(result, indent=1))
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
