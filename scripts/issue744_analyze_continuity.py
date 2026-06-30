#!/usr/bin/env python3
"""Issue #744 Phase 2 — per-layer continuity curves + discontinuity stratification.

Reads the Phase-1 outputs (per-corpus summaries + the retained raw NS dump +
fixed population stats) and emits the §6.5 primary-deliverable CSVs/JSONs (CPU,
off-pod on the VM):

* ``per_layer_continuity.csv``        — one row per (layer x flavor x step x
  corpus) with mean + bootstrap-over-sequences 95% CI bounds (consecutive cosine
  is recorded at step=-1). Required for H1/H2.
* ``per_layer_extrap_error.csv``      — std-fit + raw-fit columns per layer x
  corpus.
* ``discontinuity_stratification.csv`` — per (layer x stratifier x stratum) mean
  jump + CI, on the standardized flavor (H3).
* ``random_baseline.json``            — per-layer per-FLAVOR chance abs-cosine
  (concern #2: each flavor's baseline is stored separately + tagged, so the
  analyzer compares flavor-matched curves).
* ``ns_word_level_continuity.csv``    — the Barenholtz-comparable word-level
  last-subword read (concern #5: the cross-paper analogue uses this, NOT the
  every-subword primary curve).
* ``sink_excluded_continuity.csv``    — the sink-EXCLUDED +1 direction-pres
  recompute on NS (concern #1), zero extra GPU.
* ``proxy_vs_gold_penn.json``         — closed-class wordlist vs gold-Penn
  clause-opener agreement on NS (A11).

Bootstrap over SEQUENCES (NOT over per-token observations — within-sequence
positions are autocorrelated), B=2000, weighted by each sequence's valid-pair
count (plan §6 CI methodology). For NS (10 sequences) the CI is wide and
reported honestly; the broader corpus carries the statistical weight.

Usage::

    uv run python scripts/issue744_analyze_continuity.py \\
        --dump-dir data/issue_744/base --out-dir eval_results/issue_744/base
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from dotenv import load_dotenv  # noqa: E402
from issue404_common import reproducibility_metadata  # noqa: E402
from issue744_common import (  # noqa: E402
    BOOTSTRAP_B,
    CLAUSE_OPENER_WORDS,  # noqa: F401 — kept for parity / future use
    DIRECTION_PRES_STEPS,
    FLAVORS,
    RANDOM_BASELINE_N_PAIRS,
    SEED,
    TRAJECTORY_WINDOW_K,
    is_clause_opener,
    write_json,
)

from explore_persona_space.analysis.continuity import (  # noqa: E402
    closed_form_random_abs_cosine,
    direction_preservation,
    make_flavors_from_stats,
    random_baseline,
)

load_dotenv()

logger = logging.getLogger("issue744_analyze")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Step -1 is the schema sentinel for the consecutive-cosine row (vs +0/+1/+2/+3
# direction-preservation rows) in per_layer_continuity.csv.
CONSEC_COS_STEP = -1


# ── Bootstrap over sequences ───────────────────────────────────────────────────


def _weighted_mean(sums: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """Sum(sums) / Sum(counts) over the sequence axis (axis 0). Returns (L,) or scalar."""
    denom = counts.sum(axis=0)
    num = sums.sum(axis=0)
    out = np.where(denom > 0, num / np.where(denom > 0, denom, 1.0), np.nan)
    return out


def bootstrap_ci(
    seq_sums: np.ndarray, seq_counts: np.ndarray, b: int, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bootstrap-over-sequences mean + 95% CI for a per-layer statistic.

    ``seq_sums`` / ``seq_counts`` are ``(n_seq, L)`` — per-sequence SUMS and
    valid-pair COUNTS. The point estimate is the count-weighted mean over all
    sequences; each bootstrap resample draws ``n_seq`` sequences with replacement
    and recomputes the weighted mean. Returns ``(mean (L,), lo (L,), hi (L,))``
    at the [2.5, 97.5] percentiles. Layers with zero total count -> NaN.
    """
    n_seq, L = seq_sums.shape
    point = _weighted_mean(seq_sums, seq_counts)
    if n_seq == 0:
        nan = np.full(L, np.nan)
        return nan, nan, nan
    rng = np.random.default_rng(seed)
    boot = np.empty((b, L), dtype=np.float64)
    for i in range(b):
        idx = rng.integers(0, n_seq, size=n_seq)
        boot[i] = _weighted_mean(seq_sums[idx], seq_counts[idx])
    lo = np.nanpercentile(boot, 2.5, axis=0)
    hi = np.nanpercentile(boot, 97.5, axis=0)
    return point, lo, hi


# ── Curve assembly from per-sequence summaries ─────────────────────────────────


def _stack_metric(
    sequences: list[dict], flavor: str, sum_key: str, n_key: str, n_layers: int
) -> tuple[np.ndarray, np.ndarray]:
    """Stack per-sequence (sum, count) arrays for a scalar-per-layer metric."""
    sums, counts = [], []
    for s in sequences:
        f = s["flavors"][flavor]
        sums.append(f[sum_key])
        counts.append(f[n_key])
    return np.asarray(sums, dtype=np.float64), np.asarray(counts, dtype=np.float64)


def _stack_dp(
    sequences: list[dict], flavor: str, step: int, n_layers: int
) -> tuple[np.ndarray, np.ndarray]:
    """Stack per-sequence (sum, count) for direction-preservation at one step."""
    sums, counts = [], []
    skey = str(step)
    for s in sequences:
        f = s["flavors"][flavor]
        sums.append(f["dp_sum"][skey])
        counts.append(f["dp_n"][skey])
    return np.asarray(sums, dtype=np.float64), np.asarray(counts, dtype=np.float64)


def per_layer_continuity_rows(
    corpus: str, sequences: list[dict], n_layers: int, steps: tuple[int, ...]
) -> list[dict]:
    """Build per (layer x flavor x step x corpus) continuity rows with CIs."""
    rows = []
    for flavor in FLAVORS:
        # consecutive cosine (step sentinel -1)
        cc_sum, cc_n = _stack_metric(sequences, flavor, "consec_cos_sum", "consec_cos_n", n_layers)
        mean, lo, hi = bootstrap_ci(cc_sum, cc_n, BOOTSTRAP_B, SEED)
        for li in range(n_layers):
            rows.append(
                {
                    "corpus": corpus,
                    "flavor": flavor,
                    "step": CONSEC_COS_STEP,
                    "layer": li,
                    "metric": "consec_cosine",
                    "mean": float(mean[li]),
                    "ci_lo": float(lo[li]),
                    "ci_hi": float(hi[li]),
                }
            )
        # direction preservation at each step
        for s in steps:
            dp_sum, dp_n = _stack_dp(sequences, flavor, s, n_layers)
            mean, lo, hi = bootstrap_ci(dp_sum, dp_n, BOOTSTRAP_B, SEED)
            for li in range(n_layers):
                rows.append(
                    {
                        "corpus": corpus,
                        "flavor": flavor,
                        "step": s,
                        "layer": li,
                        "metric": "direction_preservation",
                        "mean": float(mean[li]),
                        "ci_lo": float(lo[li]),
                        "ci_hi": float(hi[li]),
                    }
                )
    return rows


def per_layer_extrap_rows(corpus: str, sequences: list[dict], n_layers: int) -> list[dict]:
    """Per (layer x flavor x corpus) extrap-error rows (std primary, raw alongside)."""
    rows = []
    for flavor in FLAVORS:
        ee_sum, ee_n = _stack_metric(sequences, flavor, "extrap_sum", "extrap_n", n_layers)
        mean, lo, hi = bootstrap_ci(ee_sum, ee_n, BOOTSTRAP_B, SEED)
        for li in range(n_layers):
            rows.append(
                {
                    "corpus": corpus,
                    "flavor": flavor,
                    "layer": li,
                    "mean_l2": float(mean[li]),
                    "ci_lo": float(lo[li]),
                    "ci_hi": float(hi[li]),
                }
            )
    return rows


# ── Random baseline (per flavor) ───────────────────────────────────────────────


def per_flavor_random_baseline(
    dump_dir: Path, stats: dict, corpus_key: str, raw_dir: Path, n_layers: int, hidden: int
) -> dict:
    """Per-layer per-flavor empirical Qwen random-pair abs-cosine (concern #2).

    Pools the per-position vectors across the retained raw dump for ``corpus_key``
    (NS = full dump; broader = the bounded raw subset), builds each flavor, and
    samples ``RANDOM_BASELINE_N_PAIRS`` random token pairs per layer per flavor.
    """
    mu = stats[corpus_key]["mu"]
    sigma = stats[corpus_key]["sigma"]
    rogue_idx = stats[corpus_key]["rogue_idx"]
    blobs = sorted(raw_dir.glob("seq_*.pt"))
    if not blobs:
        return {
            "corpus": corpus_key,
            "note": "no raw dump available for baseline",
            "per_flavor": {},
        }
    # Concatenate per-position vectors over sequences along the token axis.
    Hs = [torch.load(p, weights_only=False)["H_fp16"].float() for p in blobs]
    H_cat = torch.cat(Hs, dim=1)  # (L, sum_T, hidden)
    flavors = make_flavors_from_stats(H_cat, mu, sigma, rogue_idx)
    out = {
        "corpus": corpus_key,
        "closed_form_d3584": closed_form_random_abs_cosine(hidden),
        "per_flavor": {},
    }
    for flavor, H in flavors.items():
        base = random_baseline(H, RANDOM_BASELINE_N_PAIRS, SEED)  # (L,)
        out["per_flavor"][flavor] = base.tolist()
    return out


def broader_random_baseline(dump_dir: Path, hidden: int) -> dict:
    """Read the Phase-1 reservoir-sampled broader random baseline (#744 concern).

    The broader corpus is STREAMED (no full raw retention), so Phase 1
    reservoir-samples raw token vectors over the WHOLE stream and pre-computes the
    per-flavor random-pair abs-cosine into ``broader_random_pairs.pt`` (the right
    distribution + bounded memory). The analyzer simply reads it here rather than
    concatenating the bounded ``broader_raw`` spot-check subset (the wrong
    distribution + a >50 GB all-at-once materialization). Falls back to a noted
    empty result if the artifact is absent (older dumps).
    """
    pt = dump_dir / "broader_random_pairs.pt"
    if not pt.exists():
        return {
            "corpus": "broader",
            "note": "broader_random_pairs.pt not found (pre-reservoir dump)",
            "closed_form_d3584": closed_form_random_abs_cosine(hidden),
            "per_flavor": {},
        }
    payload = torch.load(pt, weights_only=False)
    return {
        "corpus": "broader",
        "source": "phase1_reservoir",
        "n_pool": payload.get("n_pool"),
        "n_pairs": payload.get("n_pairs"),
        "closed_form_d3584": closed_form_random_abs_cosine(hidden),
        "per_flavor": payload.get("per_flavor", {}),
    }


# ── H3 discontinuity-locus stratification (NS raw dump) ────────────────────────


def _jump_sum_count(
    jumps: torch.Tensor, pos_mask_2d: torch.Tensor
) -> tuple[np.ndarray, np.ndarray]:
    """Per-layer (sum of jumps, count) over the positions where ``pos_mask_2d`` fires.

    ``jumps`` / ``pos_mask_2d`` are both ``(L, T-1)``. Returns ``(sum (L,),
    count (L,))`` as fp64 numpy arrays (the per-sequence bootstrap inputs).
    """
    m = pos_mask_2d.float()
    s = (jumps * m).sum(dim=1).numpy()
    c = m.sum(dim=1).numpy()
    return s.astype(np.float64), c.astype(np.float64)


def stratified_jumps(
    raw_dir: Path, stats: dict, corpus_key: str, n_layers: int, corpus_label: str | None = None
) -> list[dict]:
    """Per (layer x stratifier x stratum) mean jump + CI on the STANDARDIZED flavor.

    Jump = ||z(h^L_t) - z(h^L_{t-1})||_2 at each position t>=1, stratified by
    (i) sink mask (per-(layer,position), concern #1), (ii) surprisal tercile
    (terciles over the corpus token population), (iii) clause-opener mask.
    Bootstrap over SEQUENCES within each stratum (plan §4.5).

    ``corpus_key`` indexes ``stats`` (``"ns"`` / ``"broader"``); ``corpus_label``
    is the CANONICAL label written to the CSV (``"natural_stories"`` /
    ``"broader"``) so it matches the per-layer continuity rows + the figure
    script's corpus filter. Defaults to ``corpus_key`` when not given.
    """
    label = corpus_label or corpus_key
    mu = stats[corpus_key]["mu"]
    sigma = stats[corpus_key]["sigma"]
    blobs = sorted(raw_dir.glob("seq_*.pt"))
    if not blobs:
        return []

    # First pass: surprisal terciles over the whole corpus token population.
    all_surp = []
    for p in blobs:
        b = torch.load(p, weights_only=False)
        s = b["surprisal"]
        all_surp.append(s[~torch.isnan(s)])
    surp_pool = torch.cat(all_surp) if all_surp else torch.tensor([])
    if surp_pool.numel() >= 3:
        t_lo, t_hi = torch.quantile(surp_pool, torch.tensor([1 / 3, 2 / 3])).tolist()
    else:
        t_lo = t_hi = float("nan")

    # Per-sequence per-layer jump SUMS + COUNTS for each stratum.
    # strata: sink/non_sink, surp_low/surp_mid/surp_high, clause_opener/clause_interior
    stratum_keys = {
        "sink": ["sink", "non_sink"],
        "surprisal": ["low", "mid", "high"],
        "syntactic": ["clause_opener", "clause_interior"],
    }
    # acc[(stratifier, stratum)] -> list over sequences of (sum (L,), count (L,))
    acc: dict[tuple[str, str], list[tuple[np.ndarray, np.ndarray]]] = {
        (st, val): [] for st, vals in stratum_keys.items() for val in vals
    }

    for p in blobs:
        b = torch.load(p, weights_only=False)
        H = b["H_fp16"].float()  # (L, T, hidden)
        L, _T, _ = H.shape
        z = (H - mu.unsqueeze(1)) / (sigma.unsqueeze(1) + 1e-8)  # (L, T, hidden) standardized
        jumps = (z[:, 1:] - z[:, :-1]).norm(dim=-1)  # (L, T-1) jump at position t (t=1..T-1)
        # position labels for t = 1..T-1
        sink = b["sink_mask"][:, 1:]  # (L, T-1) per-(layer,position)
        surp = b["surprisal"][1:]  # (T-1,)
        clause = b["clause_opener_mask"][1:]  # (T-1,)

        # sink stratifier (per-layer position mask)
        acc[("sink", "sink")].append(_jump_sum_count(jumps, sink))
        acc[("sink", "non_sink")].append(_jump_sum_count(jumps, ~sink))
        # surprisal stratifier (per-position, broadcast over layers)
        valid = ~torch.isnan(surp)
        low = (surp <= t_lo) & valid
        high = (surp > t_hi) & valid
        mid = valid & ~low & ~high
        for name, m1d in (("low", low), ("mid", mid), ("high", high)):
            acc[("surprisal", name)].append(_jump_sum_count(jumps, m1d.unsqueeze(0).expand(L, -1)))
        # syntactic stratifier (per-position)
        acc[("syntactic", "clause_opener")].append(
            _jump_sum_count(jumps, clause.unsqueeze(0).expand(L, -1))
        )
        acc[("syntactic", "clause_interior")].append(
            _jump_sum_count(jumps, (~clause).unsqueeze(0).expand(L, -1))
        )

    rows = []
    for (stratifier, stratum), seqlist in acc.items():
        if not seqlist:
            continue
        sums = np.stack([s for s, _ in seqlist])  # (n_seq, L)
        counts = np.stack([c for _, c in seqlist])
        mean, lo, hi = bootstrap_ci(sums, counts, BOOTSTRAP_B, SEED)
        for li in range(n_layers):
            rows.append(
                {
                    "corpus": label,
                    "stratifier": stratifier,
                    "stratum": stratum,
                    "layer": li,
                    "mean_jump": float(mean[li]),
                    "ci_lo": float(lo[li]),
                    "ci_hi": float(hi[li]),
                }
            )
    return rows


def sink_excluded_continuity(raw_dir: Path, stats: dict, n_layers: int, k: int) -> list[dict]:
    """Sink-EXCLUDED +1 direction-preservation recompute on NS (concern #1).

    Re-runs direction_preservation on the STANDARDIZED flavor with sink positions
    masked out of the trajectory, per sequence, then bootstraps over sequences.
    The sink mask is per-(layer,position); we exclude a window if ANY of its
    positions (the k fit positions or the +1 read positions) is a sink at that
    layer — implemented by dropping sink positions from the per-layer token axis
    before the fit. To keep it tractable + comparable, we exclude sink positions
    GLOBALLY across layers (a position that is a sink at any layer is dropped)
    so the position index stays aligned across layers.
    """
    mu = stats["ns"]["mu"]
    sigma = stats["ns"]["sigma"]
    blobs = sorted(raw_dir.glob("seq_*.pt"))
    if not blobs:
        return []
    sums, counts = [], []
    for p in blobs:
        b = torch.load(p, weights_only=False)
        H = b["H_fp16"].float()  # (L, T, hidden)
        z = (H - mu.unsqueeze(1)) / (sigma.unsqueeze(1) + 1e-8)
        # position kept iff NOT a sink at any layer (keep index aligned across layers)
        keep = ~b["sink_mask"].any(dim=0)  # (T,)
        if keep.sum() < (k + 2):
            continue
        z_kept = z[:, keep]  # (L, T_kept, hidden)
        dp = direction_preservation(z_kept, k=k, steps=(1,))[1]  # (L,) mean abs-cos at +1
        T_kept = z_kept.shape[1]
        max_w = T_kept - 1 - k - 1
        n_valid = max(0, max_w + 1)
        sum_l = torch.where(torch.isnan(dp), torch.zeros_like(dp), dp * n_valid).numpy()
        cnt_l = np.full(n_layers, float(n_valid))
        sums.append(sum_l.astype(np.float64))
        counts.append(cnt_l)
    if not sums:
        return []
    sums = np.stack(sums)
    counts = np.stack(counts)
    mean, lo, hi = bootstrap_ci(sums, counts, BOOTSTRAP_B, SEED)
    return [
        {
            "corpus": "natural_stories",
            "flavor": "std",
            "step": 1,
            "layer": li,
            "metric": "direction_preservation_sink_excluded",
            "mean": float(mean[li]),
            "ci_lo": float(lo[li]),
            "ci_hi": float(hi[li]),
        }
        for li in range(n_layers)
    ]


def ns_word_level_continuity(raw_dir: Path, stats: dict, n_layers: int, k: int) -> list[dict]:
    """Word-level last-subword direction-preservation on NS (concern #5).

    The Barenholtz cross-paper analogue uses the LAST subword of each word, not
    every consecutive subword. We re-index each NS sequence to its word_end_idx
    positions, then run direction_preservation on that reduced token axis
    (standardized flavor). Bootstrap over sequences.
    """
    mu = stats["ns"]["mu"]
    sigma = stats["ns"]["sigma"]
    blobs = sorted(raw_dir.glob("seq_*.pt"))
    if not blobs:
        return []
    rows_by_step: dict[int, tuple[list, list]] = {s: ([], []) for s in DIRECTION_PRES_STEPS}
    for p in blobs:
        b = torch.load(p, weights_only=False)
        widx = b.get("word_end_idx")
        if widx is None or widx.numel() < (k + 2):
            continue
        H = b["H_fp16"].float()[:, widx]  # (L, n_words, hidden)
        z = (H - mu.unsqueeze(1)) / (sigma.unsqueeze(1) + 1e-8)
        dp = direction_preservation(z, k=k, steps=DIRECTION_PRES_STEPS)
        T_words = z.shape[1]
        for s in DIRECTION_PRES_STEPS:
            max_w = T_words - 1 - k - s
            n_valid = max(0, max_w + 1)
            mean_s = dp[s]
            sum_s = torch.where(torch.isnan(mean_s), torch.zeros_like(mean_s), mean_s * n_valid)
            rows_by_step[s][0].append(sum_s.numpy().astype(np.float64))
            rows_by_step[s][1].append(np.full(n_layers, float(n_valid)))
    rows = []
    for s, (sums, counts) in rows_by_step.items():
        if not sums:
            continue
        mean, lo, hi = bootstrap_ci(np.stack(sums), np.stack(counts), BOOTSTRAP_B, SEED)
        for li in range(n_layers):
            rows.append(
                {
                    "corpus": "natural_stories",
                    "flavor": "std",
                    "step": s,
                    "layer": li,
                    "metric": "direction_preservation_word_level",
                    "mean": float(mean[li]),
                    "ci_lo": float(lo[li]),
                    "ci_hi": float(hi[li]),
                }
            )
    return rows


def proxy_vs_gold_penn(raw_dir: Path, ns_penn_path: Path) -> dict:
    """Closed-class wordlist proxy vs gold-Penn clause-opener agreement on NS (A11).

    The gold-Penn clause-opener label is read at the WORD level: a word is a
    gold clause-opener iff it is the FIRST terminal under an S/SBAR constituent
    in the Penn parse OR a CC/IN-tagged terminal opening a constituent. Parsing
    the multi-line Penn trees robustly to the NS word stream is brittle, so we
    report the agreement at the level achievable deterministically: per gold
    terminal, whether its POS tag is in the clause-opener closed class (CC/IN)
    AND whether the wordlist proxy fires on the same surface form. If the parse
    file is missing or unparseable we report ``available: false`` and fall back
    to the wordlist proxy alone (flagged in the clean-result as the coarser
    proxy).
    """
    if not ns_penn_path.exists():
        return {"available": False, "reason": "Penn parse file not found"}
    text = ns_penn_path.read_text(errors="replace")
    # Gold terminals: (POS word) leaves. Penn CC/IN are the clause-opener tags.
    import re

    leaf_re = re.compile(r"\(([A-Z$.,:]+)\s+([^()\s]+)\)")
    gold_clause_tags = {"CC", "IN"}
    agree = 0
    total = 0
    proxy_only = 0
    gold_only = 0
    for tag, word in leaf_re.findall(text):
        gold = tag in gold_clause_tags
        proxy = is_clause_opener(word)
        total += 1
        if gold == proxy:
            agree += 1
        elif proxy and not gold:
            proxy_only += 1
        elif gold and not proxy:
            gold_only += 1
    return {
        "available": total > 0,
        "n_terminals": total,
        "agreement_rate": (agree / total) if total else float("nan"),
        "proxy_fires_gold_doesnt": proxy_only,
        "gold_fires_proxy_doesnt": gold_only,
        "note": (
            "POS-tag-level (CC/IN) gold vs closed-class wordlist proxy; word-stream "
            "alignment of the multi-line Penn trees to NS not attempted — the wordlist "
            "proxy is the primary deterministic mask for both corpora (plan §11 item 12)."
        ),
    }


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main() -> int:
    parser = argparse.ArgumentParser(description="Issue #744 Phase 2: analyze continuity.")
    parser.add_argument("--dump-dir", type=Path, default=PROJECT_ROOT / "data/issue_744/base")
    parser.add_argument(
        "--out-dir", type=Path, default=PROJECT_ROOT / "eval_results/issue_744/base"
    )
    args = parser.parse_args()

    dump_dir = Path(args.dump_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads((dump_dir / "dump_manifest.json").read_text())
    n_layers = manifest["n_layers"]
    hidden = manifest["hidden"]
    k = manifest.get("k", TRAJECTORY_WINDOW_K)
    steps = tuple(manifest.get("steps", DIRECTION_PRES_STEPS))
    stats = torch.load(dump_dir / "population_stats.pt", weights_only=False)

    ns = json.loads((dump_dir / "ns_summaries.json").read_text())["sequences"]
    broader = json.loads((dump_dir / "broader_summaries.json").read_text())["sequences"]

    logger.info(
        "Computing per-layer continuity curves (NS=%d, broader=%d seqs)...", len(ns), len(broader)
    )
    cont_rows = per_layer_continuity_rows("natural_stories", ns, n_layers, steps)
    cont_rows += per_layer_continuity_rows("broader", broader, n_layers, steps)
    _write_csv(out_dir / "per_layer_continuity.csv", cont_rows)

    extrap_rows = per_layer_extrap_rows("natural_stories", ns, n_layers)
    extrap_rows += per_layer_extrap_rows("broader", broader, n_layers)
    _write_csv(out_dir / "per_layer_extrap_error.csv", extrap_rows)

    logger.info("Random baselines (per flavor)...")
    # NS: full retained dump concat (NS raw ~2.6 GB, fits the floor). Broader:
    # the Phase-1 reservoir-sampled baseline over the FULL stream
    # (broader_random_pairs.pt) — NOT a re-concat of the bounded broader_raw
    # subset (wrong distribution + >50 GB materialization risk; #744 concern).
    rb = {
        "natural_stories": per_flavor_random_baseline(
            dump_dir, stats, "ns", dump_dir / "ns_raw", n_layers, hidden
        ),
        "broader": broader_random_baseline(dump_dir, hidden),
        "metadata": reproducibility_metadata({"script": "issue744_analyze_continuity"}),
    }
    write_json(out_dir / "random_baseline.json", rb)

    logger.info("H3 stratification (NS + broader spot-check)...")
    strat_rows = stratified_jumps(
        dump_dir / "ns_raw", stats, "ns", n_layers, corpus_label="natural_stories"
    )
    strat_rows += stratified_jumps(
        dump_dir / "broader_raw", stats, "broader", n_layers, corpus_label="broader"
    )
    _write_csv(out_dir / "discontinuity_stratification.csv", strat_rows)

    logger.info("Concern reads: sink-excluded + word-level + proxy-vs-gold...")
    _write_csv(
        out_dir / "sink_excluded_continuity.csv",
        sink_excluded_continuity(dump_dir / "ns_raw", stats, n_layers, k),
    )
    _write_csv(
        out_dir / "ns_word_level_continuity.csv",
        ns_word_level_continuity(dump_dir / "ns_raw", stats, n_layers, k),
    )
    write_json(
        out_dir / "proxy_vs_gold_penn.json",
        proxy_vs_gold_penn(dump_dir / "ns_raw", dump_dir / "ns_penn_parses.txt"),
    )

    write_json(
        out_dir / "analysis_manifest.json",
        {
            "dump_dir": str(dump_dir),
            "n_layers": n_layers,
            "hidden": hidden,
            "k": k,
            "steps": list(steps),
            "bootstrap_B": BOOTSTRAP_B,
            "bootstrap_unit": "sequence",
            "rogue_rank_metric": stats.get("rogue_rank_metric"),
            "rogue_topk": stats.get("rogue_topk"),
            "metadata": reproducibility_metadata({"script": "issue744_analyze_continuity"}),
        },
    )
    logger.info("Analysis complete -> %s", out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
