"""#1979 — race the two SOURCE-ANCHORED candidates against the committed set, all 18 arms.

The committed #1979 race fixes training-row-centroid anchors: every candidate
compares a destination prefix against the TRAINING ROWS. Two candidates anchor
on the arm's OWN SOURCE (trained) prefix instead, and neither was raced:

  p2_ps   cos(V[i], V[s])   real source answers vs real target answers
                            (computed by the race, persisted, never raced)
  p3a_ps  cos(M[i], M[s])   mapped source answers vs mapped target answers
                            (introduced by scripts/issue1979_po_source_target_fig.py)

with V = base answer vector at a prefix, M = M0 . context vector at a prefix,
i = destination prefix, s = the arm's source prefix. Both use the committed
panel-centered cosine convention.

This runs them under the committed discipline rather than as a post-hoc read:

  * raced set = the committed 12 (p1 p2 p3a p3b p4 p5 p6 p7 p8a p8b p9 p10;
    marker drops p8a/p8b) PLUS the two above -> K=14 content, K=12 marker;
  * per-arm within-arm Spearman rho against the committed DV
    (dv_change content, dv_dlogp marker) at the primary (layer, position);
  * per-arm permutation band = SIGNED MAX over the full raced set per draw,
    20,000 draws, so both additions pay their own selection cost;
  * winner probability from a 2,000-draw prefix-resample bootstrap with the
    winner re-selected inside every draw, exact Spearman per draw (ranks
    recomputed on the resample, not carried);
  * paired within-arm comparison against the committed change champion p3b.

Stated scope limit: the committed run bootstrapped three resampling families
(prefix resample primary, query cluster, family cluster). Only the PRIMARY
prefix-resample family is reconstructible from the banked per-prefix frames, so
winner probability here is that family alone and is labeled as such.

Structural note: both source-anchored candidates are exactly 1.0 at the source
prefix (a panel member). Every rho is reported over all 50 prefixes AND with the
source prefix dropped.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy import stats  # noqa: E402

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
from issue1979_race import OWN_PREFIX_BY_CTX, _center  # noqa: E402
from issue1979_whiten_csls_band import _rank_z, band_for  # noqa: E402
from issue1979_whiten_csls_sweep import DV_BY_KIND, RACE, load_inputs  # noqa: E402

REPO_ROOT = SCRIPTS_DIR.parent
PANEL = REPO_ROOT / "eval_results/issue_1979/config/prefix_panel.json"
OUT = REPO_ROOT / "eval_results/issue_1979/whiten_csls/source_anchored_race.json"
TENSORS = Path(
    "/mnt/eps-data/thomasjiralerspong/issue1979_whitencsls/battery/ingredient_tensors.pt"
)

# committed raced set -> frame column name
COMMITTED = {
    "p1": "p1_tc",
    "p2": "p2_tc",
    "p3a": "p3a_tc",
    "p3b": "p3b_tc",
    "p4": "p4_tc",
    "p5": "p5",
    "p6": "p6",
    "p7": "p7",
    "p8a": "p8a",
    "p8b": "p8b",
    "p9": "p9_k8",
    "p10": "p10_k8",
}
CONTENT_ONLY = ("p8a", "p8b")
NEW = ("p2_ps", "p3a_ps")
PRIMARY = {"content": (19, "last_prompt"), "marker": (25, "last_prompt")}
N_PERM = 20_000
N_BOOT = 2_000
SEED = 1979


def _spearman_boot(X: np.ndarray, y: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Exact per-draw Spearman. X (n,K), y (n,), idx (B,n) -> (B,K).

    Ranks are recomputed on each resample (bootstrap duplicates create ties,
    handled by rankdata's average method); correlation is Pearson on ranks.
    """
    xb = stats.rankdata(X[idx], axis=1)  # (B, n, K)
    yb = stats.rankdata(y[idx], axis=1)  # (B, n)
    xb = xb - xb.mean(axis=1, keepdims=True)
    yb = yb - yb.mean(axis=1, keepdims=True)
    num = np.einsum("bnk,bn->bk", xb, yb)
    den = np.sqrt((xb**2).sum(axis=1) * (yb**2).sum(axis=1)[:, None])
    return num / (den + 1e-12)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tensors", type=Path, default=TENSORS)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--n-perm", type=int, default=N_PERM)
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument(
        "--regime",
        choices=("all", "po", "con"),
        default="all",
        help="restrict the race to one training-data regime (po = positive-only)",
    )
    args = ap.parse_args(argv)

    assert args.tensors.exists(), f"staged tensors missing: {args.tensors}"
    tens = torch.load(args.tensors, map_location="cpu", weights_only=True)
    inputs = load_inputs()
    members = json.loads(PANEL.read_text())["members"]
    panel_ids = [m["prefix_id"] if isinstance(m, dict) else m for m in members]

    arms = [a for a in inputs["arms"] if args.regime in ("all", a["regime"])]
    assert arms, f"no arms at regime={args.regime}"
    out_path = args.out
    if args.regime != "all" and out_path == OUT:
        out_path = out_path.with_name(f"{out_path.stem}_{args.regime}{out_path.suffix}")
    print(f"[race] regime={args.regime}  arms={len(arms)}  -> {out_path.name}")

    per_arm: dict[str, dict] = {}
    boot_by_kind: dict[str, list[np.ndarray]] = {"content": [], "marker": []}
    names_by_kind: dict[str, list[str]] = {}
    rng = np.random.default_rng(SEED)

    for arm in arms:
        aid, kind = arm["arm_id"], arm["kind"]
        layer, pos = PRIMARY[kind]
        frame = json.loads((RACE / f"frame_{aid}.json").read_text())["frame"]
        dv = np.asarray(frame[DV_BY_KIND[kind]], dtype=np.float64)

        names = [k for k in COMMITTED if not (kind == "marker" and k in CONTENT_ONLY)]
        cols: dict[str, np.ndarray] = {}
        for k in names:
            v = frame.get(COMMITTED[k])
            assert v is not None, f"{aid}: committed candidate {k} ({COMMITTED[k]}) absent"
            cols[k] = np.asarray(v, dtype=np.float64)

        v_ps = np.asarray(frame["p2_ps"], dtype=np.float64)
        s_ix = int(np.argmax(v_ps))
        assert v_ps[s_ix] > 0.9999 and np.sum(v_ps > 0.9999) == 1, aid
        assert panel_ids[s_ix] == OWN_PREFIX_BY_CTX[arm["ctx_key"]], (aid, panel_ids[s_ix])
        cols["p2_ps"] = v_ps

        m0 = np.asarray(tens[f"m0pred/{kind}/L{layer}/{pos}"].double().numpy())
        mc, _ = _center(m0)
        mn = mc / (np.linalg.norm(mc, axis=1, keepdims=True) + 1e-12)
        cols["p3a_ps"] = mn @ mn[s_ix]
        assert cols["p3a_ps"][s_ix] > 0.9999, aid

        names = names + list(NEW)
        if kind not in names_by_kind:
            names_by_kind[kind] = names
        assert names_by_kind[kind] == names, (aid, names)

        X = np.column_stack([cols[k] for k in names])
        ok = np.isfinite(dv) & np.isfinite(X).all(axis=1)
        assert ok.sum() >= 45, (aid, int(ok.sum()))
        Xo, dvo = X[ok], dv[ok]

        rho = {k: float(stats.spearmanr(Xo[:, j], dvo).statistic) for j, k in enumerate(names)}
        keep = ok.copy()
        keep[s_ix] = False
        Xd, dvd = X[keep], dv[keep]
        rho_ds = {k: float(stats.spearmanr(Xd[:, j], dvd).statistic) for j, k in enumerate(names)}

        band = band_for(_rank_z(Xo), _rank_z(dvo), args.n_perm, SEED)

        n = Xo.shape[0]
        idx = rng.integers(0, n, size=(args.n_boot, n))
        boot = _spearman_boot(Xo, dvo, idx)  # (B, K)
        boot_by_kind[kind].append(boot)

        per_arm[aid] = {
            "kind": kind,
            "n": int(ok.sum()),
            "source_prefix": panel_ids[s_ix],
            "rho": rho,
            "rho_drop_source": rho_ds,
            "band_p975_max_selected": band["p975_max_selected"],
            "k_candidates": band["k_candidates"],
            "clears": {k: bool(rho[k] > band["p975_max_selected"]) for k in names},
        }

    summary: dict[str, dict] = {}
    for kind, mats in boot_by_kind.items():
        if not mats:  # this regime has no arms of this kind
            continue
        names = names_by_kind[kind]
        arms = [a for a, r in per_arm.items() if r["kind"] == kind]
        med = {k: float(np.median([per_arm[a]["rho"][k] for a in arms])) for k in names}
        clears = {k: int(sum(per_arm[a]["clears"][k] for a in arms)) for k in names}
        B = np.stack(mats, axis=0)  # (n_arms, B, K)
        med_draw = np.median(B, axis=0)  # (B, K) across-arm median per draw
        win = np.bincount(np.argmax(med_draw, axis=1), minlength=len(names)) / med_draw.shape[0]
        champ = max(med, key=med.get)
        # paired within-arm: candidate beats the committed change champion p3b
        beats_p3b = {
            k: int(sum(per_arm[a]["rho"][k] > per_arm[a]["rho"]["p3b"] for a in arms))
            for k in names
        }
        summary[kind] = {
            "n_arms": len(arms),
            "k_candidates": len(names),
            "median_rho": med,
            "clears_band": clears,
            "winner_prob_prefix_resample": {k: float(win[j]) for j, k in enumerate(names)},
            "beats_p3b_within_arm": beats_p3b,
            "argmax_median": champ,
        }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "regime": args.regime,
                "per_arm": per_arm,
                "summary": summary,
                "n_perm": args.n_perm,
                "n_boot": args.n_boot,
                "seed": SEED,
                "bootstrap_family": "prefix_resample_only",
            },
            indent=2,
        )
    )

    for kind, s in summary.items():
        names = names_by_kind[kind]
        print(f"\n=== {kind}  ({s['n_arms']} arms, K={s['k_candidates']}) ===")
        print(f"{'cand':8s} {'medRho':>7s} {'clears':>8s} {'winP':>6s} {'>p3b':>6s}")
        for k in sorted(names, key=lambda k: -s["median_rho"][k]):
            mark = " <-- NEW" if k in NEW else ""
            print(
                f"{k:8s} {s['median_rho'][k]:+7.3f} {s['clears_band'][k]:4d}/{s['n_arms']:<3d} "
                f"{s['winner_prob_prefix_resample'][k]:6.3f} "
                f"{s['beats_p3b_within_arm'][k]:3d}/{s['n_arms']:<3d}{mark}"
            )
        print(f"  argmax across-arm median: {s['argmax_median']}")
    print(f"\n[race] -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
