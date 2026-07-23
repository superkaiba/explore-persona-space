"""#1586 H2 leakage lattice (analyzer, p11).

Plan §3/§6 second registered lattice: per behavior, the seed-pooled
con-regime paired panel-leakage difference ΔM (content: pooled non-source
judged rate, FT − LoRA; marker: pooled non-source EOS-margin
Δ(z_marker − z_eos) trained−base difference, ΔG log-prob alongside), with
question-cluster bootstrap CIs (2000 draws, seed 653, identical resample
indices in both arms, seed-stratified pooling) + the Newcombe interval on
each per-pair pooled rate difference (the #1481 figure-error-bar
convention). Content per-completion labels reconstructed from the
consolidated ``judge_raw.json`` per (arm, context): item positive iff mean
kept draw score > 50 (drop-never-coerce; ``None``-scored items excluded from
the denominator). Marker per-row reads from ``marker_panel/*/slot_reads.json``.

Output: eval_results/issue_1586/panel/leakage_lattice.json
"""

from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path("data/issue_1586/hf_dl/p11_json/issue1586_methodgen")
OUT = Path("eval_results/issue_1586/panel")
SOURCE_CTX = "persona_software_engineer"
N_BOOT = 2000
BOOT_SEED = 653
THRESHOLD = 50
N_Q = 20

_ID_RE = re.compile(r"^(?P<ctx>.+)-trained-q(?P<q>\d{3})-c(?P<c>\d+)__\d+__(?P<draw>\d+)$")


def item_labels(arm: str, ctx: str) -> dict[int, list[bool]]:
    """{q_index: [positive per scored completion]} for one (arm, context).

    Mean over kept draws per item; item dropped (excluded) when every draw is
    malformed/refusal (drop-never-coerce, llm-judging rule 9).
    """
    base = ROOT / "raw_completions" / "panel" / arm / f"rate_{ctx}"
    sub = [d for d in base.iterdir() if d.is_dir()]
    assert len(sub) == 1, (arm, ctx, sub)
    raw = json.loads((sub[0] / "judge" / f"trained_{ctx}" / "judge_raw.json").read_text())
    draws: dict[tuple[int, int], list[float]] = defaultdict(list)
    for cid, rec in raw["all_scores"].items():
        m = _ID_RE.match(cid)
        assert m and m.group("ctx") == ctx, (cid, ctx)
        s = rec.get("score")
        if isinstance(s, bool) or not isinstance(s, (int, float)):
            continue  # dropped draw (REFUSAL / malformed)
        if not 0 <= float(s) <= 100:
            continue
        draws[(int(m.group("q")), int(m.group("c")))].append(float(s))
    out: dict[int, list[bool]] = defaultdict(list)
    for (q, _c), ss in sorted(draws.items()):
        out[q].append(bool(np.mean(ss) > THRESHOLD))
    return out


def pooled_rate_matrix(arm: str, contexts: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """(pos[q], n[q]) pooled over the given contexts (q = question cluster)."""
    pos = np.zeros(N_Q)
    n = np.zeros(N_Q)
    for ctx in contexts:
        lab = item_labels(arm, ctx)
        for q, ls in lab.items():
            pos[q] += sum(ls)
            n[q] += len(ls)
    return pos, n


def boot_idx(rng: np.random.Generator, n_boot: int) -> np.ndarray:
    return rng.integers(0, N_Q, size=(n_boot, N_Q))


def rate_draws(pos: np.ndarray, n: np.ndarray, idx: np.ndarray) -> np.ndarray:
    return pos[idx].sum(axis=1) / n[idx].sum(axis=1)


def newcombe(p1: float, n1: int, p2: float, n2: int) -> tuple[float, float]:
    """Newcombe (1998) method-10 hybrid Wilson interval on p1 − p2."""

    def wilson(p, n, z=1.959963985):
        d = 1 + z * z / n
        c = p + z * z / (2 * n)
        h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
        return (c - h) / d, (c + h) / d

    l1, u1 = wilson(p1, n1)
    l2, u2 = wilson(p2, n2)
    d = p1 - p2
    return d - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2), d + math.sqrt(
        (u1 - p1) ** 2 + (p2 - l2) ** 2
    )


def content_pair(beh: str, regime: str, seed: str, idx: np.ndarray) -> dict:
    summ = json.loads(
        (ROOT / "panel" / f"{beh}-pers-ft-{regime}-{seed}" / "panel_summary.json").read_text()
    )
    contexts = [c for c in summ["rates_by_context"] if c != SOURCE_CTX]
    ft_pos, ft_n = pooled_rate_matrix(f"{beh}-pers-ft-{regime}-{seed}", contexts)
    lo_pos, lo_n = pooled_rate_matrix(f"{beh}-pers-lora-{regime}-{seed}", contexts)
    d_draws = rate_draws(ft_pos, ft_n, idx) - rate_draws(lo_pos, lo_n, idx)
    p_ft, p_lo = ft_pos.sum() / ft_n.sum(), lo_pos.sum() / lo_n.sum()
    nc = newcombe(p_ft, int(ft_n.sum()), p_lo, int(lo_n.sum()))
    return {
        "rate_ft": p_ft,
        "rate_lora": p_lo,
        "n_ft": int(ft_n.sum()),
        "n_lora": int(lo_n.sum()),
        "delta": p_ft - p_lo,
        "newcombe_low": nc[0],
        "newcombe_high": nc[1],
        "draws": d_draws,
    }


def marker_rows(arm: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(q_index, d_margin, d_logp) over NON-source rows (trained − base)."""
    d = json.loads((ROOT / "marker_panel" / arm / "slot_reads.json").read_text())
    qs, dm, dg = [], [], []
    for meta, tr, ba in zip(d["per_row"]["meta"], d["per_row"]["trained"], d["per_row"]["base"]):
        if meta["context_id"] == SOURCE_CTX:
            continue
        qs.append(meta["q"])
        dm.append((tr["z_marker"] - tr["z_eos"]) - (ba["z_marker"] - ba["z_eos"]))
        dg.append(tr["logp"] - ba["logp"])
    return np.array(qs), np.array(dm), np.array(dg)


def marker_pair(regime: str, seed: str, idx: np.ndarray) -> dict:
    out = {}
    for dv_i, dv in ((1, "margin"), (2, "logp")):
        vals = {}
        for method in ("ft", "lora"):
            qs, dm, dg = marker_rows(f"mk-pers-{method}-{regime}-{seed}")
            v = (dm, dg)[dv_i - 1]
            per_q = np.array([v[qs == q].mean() for q in range(N_Q)])
            vals[method] = per_q
        d_draws = vals["ft"][idx].mean(axis=1) - vals["lora"][idx].mean(axis=1)
        out[dv] = {
            "delta": float(vals["ft"].mean() - vals["lora"].mean()),
            "draws": d_draws,
        }
    return out


def ci(draws: np.ndarray) -> tuple[float, float]:
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def main() -> int:
    rng = np.random.default_rng(BOOT_SEED)
    idx = boot_idx(rng, N_BOOT)
    out: dict = {"n_boot": N_BOOT, "boot_seed": BOOT_SEED, "content": {}, "marker": {}}
    for beh in ("syc", "imp", "cas"):
        for regime in ("con", "po"):
            per_seed = {}
            for seed in ("s42", "s137"):
                per_seed[seed] = content_pair(beh, regime, seed, idx)
            pooled = np.mean([per_seed[s]["draws"] for s in per_seed], axis=0)
            rec = {s: {k: v for k, v in per_seed[s].items() if k != "draws"} for s in per_seed}
            for s in per_seed:
                rec[s]["ci_low"], rec[s]["ci_high"] = ci(per_seed[s]["draws"])
            rec["pooled"] = {
                "delta": float(np.mean([per_seed[s]["delta"] for s in per_seed])),
                "ci_low": ci(pooled)[0],
                "ci_high": ci(pooled)[1],
            }
            out["content"][f"{beh}/{regime}"] = rec
            print(
                f"[H2] {beh} {regime}: pooled ΔM={rec['pooled']['delta']:+.3f} "
                f"[{rec['pooled']['ci_low']:+.3f},{rec['pooled']['ci_high']:+.3f}]",
                flush=True,
            )
    for regime in ("con", "po"):
        per_seed = {s: marker_pair(regime, s, idx) for s in ("s42", "s137")}
        rec = {}
        for dv in ("margin", "logp"):
            pooled = np.mean([per_seed[s][dv]["draws"] for s in per_seed], axis=0)
            rec[dv] = {
                "per_seed": {s: float(per_seed[s][dv]["delta"]) for s in per_seed},
                "per_seed_ci": {s: ci(per_seed[s][dv]["draws"]) for s in per_seed},
                "pooled_delta": float(np.mean([per_seed[s][dv]["delta"] for s in per_seed])),
                "pooled_ci": ci(pooled),
            }
        out["marker"][regime] = rec
        print(
            f"[H2] mk {regime}: pooled Δ(EOS-margin)={rec['margin']['pooled_delta']:+.3f} "
            f"[{rec['margin']['pooled_ci'][0]:+.3f},{rec['margin']['pooled_ci'][1]:+.3f}]",
            flush=True,
        )
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "leakage_lattice.json").write_text(json.dumps(out, indent=1, default=str))
    print(f"wrote {OUT / 'leakage_lattice.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
