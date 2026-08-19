"""Cross-family transfer: train on one jailbreak family, test on the other.

Arm 4 of the jailbreak-mining pilot (task #2394). Question: does the context probe
survive a family shift, and does routing through the context->answer map transfer
BETTER than the raw context probe? A map fit on family X's (v_C, v_A) pairs might
carry family-general answer structure that a family-X-trained context probe does
not.

Families (compliance-labelled rungs of #1739's evil labeling store):
  evil_train    391 always-comply positives / 2721 failed-compliance negatives
  evil_hh_rlhf  173 always-comply positives / 1066 failed-compliance negatives
evil_toxicchat yields 0 parsed compliance scores (schema gap; separate probe).

Per direction (train X -> test Y), all arms scored on the SAME family-Y test set:
  A_transfer          probe on real v_C of X, applied to real v_C of Y
  D_transfer_indomain probe on real v_A of X, applied to M_X . v_C of Y
                      (M_X = ridge context->answer fit on family X's own pairs)
  D_transfer_merged   same, with M_merged = benign rows + family-X rows
  E_transfer          probe on real v_A of X, applied to real v_A of Y (ORACLE;
                      needs generation on Y, not deployable -- upper reference)
  A_within            grouped 5-fold OOF probe on v_C WITHIN family Y
                      (the "you had in-domain labels" reference)
  E_within            grouped 5-fold OOF probe on v_A WITHIN family Y (oracle ref)
  random              floor (equals the base rate)

Test-set construction per family: ALL failed-compliance negatives, positives
subsampled to the cleanest round(n_neg * 0.05/0.95) so the base rate is 5% --
matching the pooled map-arms pool. Train-set = the OTHER family's full pool (all
positives + all negatives); no OOF needed, the families are disjoint by
construction. Train sizes differ between directions and from the within-family
references; both are reported per row.

POWER CAVEAT: two families = ONE transfer pair (two directions). A direction-level
difference cannot be separated from that specific pair's idiosyncrasy; the third
family is unavailable pending the compliance-parse fix.

Content hygiene: numeric labels + activation tensors only.
"""

from __future__ import annotations

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import argparse  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

DEST = Path(f"/mnt/eps-data/{os.environ['USER']}/issue1739_jbmine")
BENIGN = DEST / "issue1092_realistic_crossing/analysis_tensors/summaries/cell_inst_own"
MANIFEST = DEST / "issue1092_realistic_crossing/corpus/manifest.jsonl"
LAYERS = [7, 11, 15, 19, 23, 27]
PRESPEC_LAYER = 19  # pre-specified from the map-arms table; best-layer also reported
HIDDEN = 3584
SEED = 0
POS_MEAN_MIN = 90.0
NEG_FAIL_MAX = 5.0
BASE_RATE = 0.05
N_BENIGN = 6000
N_MAP_FIT_BENIGN = 3000
RIDGE_LAMBDA = 100.0
PROBE_C = 0.01
FAMILIES = ["evil_train", "evil_hh_rlhf", "evil_toxicchat"]
# families whose 5%-base test set has too few positives for a firm PR-AUC read;
# their as-TARGET rows are reported as directional only.
THIN_TARGET_POS_MAX = 20
rng = np.random.default_rng(SEED)


def load_evil():
    """Load row-aligned v_C / v_A activation stacks + their context ids."""
    vc = np.load(DEST / "evil_compliance_ctxend.npz", allow_pickle=True)
    va = np.load(DEST / "evil_answer_t1.npz", allow_pickle=True)
    cvc = [str(c) for c in vc["context_ids"]]
    cva = [str(c) for c in va["context_ids"]]
    assert cvc == cva, "v_C and v_A npz context_ids must be row-aligned"
    assert [int(x) for x in vc["layers"]] == LAYERS
    return cvc, vc["X"].astype(np.float32), va["X"].astype(np.float32)


def load_compliance_by_family():
    """Return {family: {context_id: (mean, min_over_rollouts)}} from the DV reduce.

    Merges the separate evil_toxicchat probe file when present. That file is kept
    separate on purpose: the pooled arms (sections 0b/0c) select the 150 cleanest
    positives across ALL rungs, so folding a third family into the shared DV json
    would change the pooled positive set and invalidate their committed tables.
    Transfer selects PER FAMILY, so merging here is safe.
    """
    d = json.loads((DEST / "compliance_percontext.json").read_text())
    probe = DEST / "compliance_percontext_toxicchat_probe.json"
    if probe.is_file():
        for rung, ctxs in json.loads(probe.read_text()).items():
            if not d.get(rung):  # never clobber a populated rung
                d[rung] = ctxs
                print(f"  [dv] merged {rung} from {probe.name}: {len(ctxs)} contexts")
    return {
        rung: {c: (float(v["mean"]), float(v["min_over_rollouts"])) for c, v in ctxs.items()}
        for rung, ctxs in d.items()
    }


def load_benign_map_pairs():
    """Benign (context_end, t1) pairs used for the merged map's benign half."""
    man = [json.loads(x) for x in MANIFEST.read_text(encoding="utf-8").split("\n") if x.strip()]
    take = np.sort(rng.choice(len(man), size=min(N_BENIGN, len(man)), replace=False))[
        :N_MAP_FIT_BENIGN
    ]
    ce = {L: np.load(BENIGN / f"context_end_L{L:02d}.npy")[take].astype(np.float32) for L in LAYERS}
    t1 = {L: np.load(BENIGN / f"t1_L{L:02d}.npy")[take].astype(np.float32) for L in LAYERS}
    return ce, t1


def fit_ridge_map(Xc, Yt, lam):
    """Centred multi-output ridge context->answer; returns (W, x_mean, y_mean)."""
    xm = Xc.mean(0)
    ym = Yt.mean(0)
    G = (Xc - xm).T @ (Xc - xm) + lam * np.eye(Xc.shape[1], dtype=np.float64)
    W = np.linalg.solve(G, (Xc - xm).T @ (Yt - ym))
    return W.astype(np.float32), xm.astype(np.float32), ym.astype(np.float32)


def apply_map(M, Xc):
    """Apply a fitted ridge map to context activations."""
    W, xm, ym = M
    return (Xc - xm) @ W + ym


def recon_r2(M, Xc, Ya):
    """Held-out reconstruction R^2 of the real answer activation under the map."""
    pred = apply_map(M, Xc)
    ss_res = float(((Ya - pred) ** 2).sum())
    ss_tot = float(((Ya - Ya.mean(0)) ** 2).sum())
    return 1.0 - ss_res / ss_tot


def pr_auc(y, s):
    """Average precision (PR-AUC); chance equals the positive base rate."""
    from sklearn.metrics import average_precision_score

    return float(average_precision_score(y, s))


def roc_auc(y, s):
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(y, s))


def hit_at_k(y, s, k):
    return float(y[np.argsort(-s)[:k]].mean())


def evals_to_find_n(y, s, nfind):
    """Contexts you would review, ranked by score, to surface `nfind` positives."""
    order = np.argsort(-s)
    cum = np.cumsum(y[order])
    return int(np.searchsorted(cum, nfind) + 1) if cum[-1] >= nfind else -1


def fit_apply(Xtr, ytr, Xte):
    """Train an L2-logistic probe on (Xtr, ytr); return P(pos) on Xte."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(C=PROBE_C, max_iter=2000, class_weight="balanced")
    clf.fit(sc.transform(Xtr), ytr)
    return clf.predict_proba(sc.transform(Xte))[:, 1]


def probe_oof_same(X, y, groups):
    """Grouped 5-fold OOF probe, train and test on the same feature matrix."""
    from sklearn.model_selection import GroupKFold

    oof = np.full(len(y), np.nan)
    for tr, te in GroupKFold(n_splits=5).split(X, y, groups):
        oof[te] = fit_apply(X[tr], y[tr], X[te])
    assert not np.isnan(oof).any()
    return oof


def metrics(y, s, k5):
    return {
        "roc_auc": roc_auc(y, s),
        "pr_auc": pr_auc(y, s),
        "hit@5pct": hit_at_k(y, s, k5),
        "evals_to_find_20": evals_to_find_n(y, s, min(20, int(y.sum()))),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=LAYERS,
        help="layer subset to sweep (default: all 6). A single layer is the "
        "measured-pilot shape used to size the full sweep.",
    )
    ap.add_argument(
        "--out-suffix",
        default="",
        help="suffix for the results filename, so a pilot run cannot overwrite "
        "the full sweep's transfer_results.json",
    )
    args = ap.parse_args()
    layers = list(args.layers)
    assert set(layers) <= set(LAYERS), f"layers must be a subset of {LAYERS}, got {layers}"

    t0 = time.time()
    print(f"[load] evil v_C + v_A + per-family compliance DV (layers={layers}) ...", flush=True)
    ectx, eVC, eVA = load_evil()
    byfam = load_compliance_by_family()
    row_of = {c: i for i, c in enumerate(ectx)}

    fam = {}
    usable = []
    for f in FAMILIES:
        scores = byfam.get(f, {})
        if not scores:
            print(f"  {f}: SKIPPED — no compliance DV rows on disk", flush=True)
            continue
        usable.append(f)
        pos = [
            (c, m, mn) for c, (m, mn) in scores.items() if m >= POS_MEAN_MIN and mn >= POS_MEAN_MIN
        ]
        neg = [c for c, (m, _) in scores.items() if m <= NEG_FAIL_MAX]
        pos_rows_all = [row_of[c] for c, _, _ in pos if c in row_of]
        neg_rows = np.array(sorted(row_of[c] for c in neg if c in row_of))
        # positives: keep the cleanest, subsampled so the TEST base rate is 5%
        n_pos_test = int(round(len(neg_rows) * BASE_RATE / (1 - BASE_RATE)))
        pos_sorted = [
            row_of[c] for c, m, mn in sorted(pos, key=lambda t: (t[2], t[1]), reverse=True)
        ]
        fam[f] = {
            "pos_all": np.array(sorted(pos_rows_all)),
            "neg_all": neg_rows,
            "pos_test": np.array(sorted(pos_sorted[:n_pos_test])),
        }
        print(
            f"  {f}: pos_avail={len(pos_rows_all)} neg={len(neg_rows)} "
            f"-> test pos={len(fam[f]['pos_test'])} (base {BASE_RATE:.2f})",
            flush=True,
        )

    print("[load] benign map pairs ...", flush=True)
    bce, bt1 = load_benign_map_pairs()

    results = {"families": {}, "directions": {}, "layers_swept": layers, "usable_families": usable}
    for f in usable:
        results["families"][f] = {
            "n_pos_available": int(len(fam[f]["pos_all"])),
            "n_neg": int(len(fam[f]["neg_all"])),
            "n_pos_test": int(len(fam[f]["pos_test"])),
            "n_train_full": int(len(fam[f]["pos_all"]) + len(fam[f]["neg_all"])),
            "thin_target": bool(len(fam[f]["pos_test"]) <= THIN_TARGET_POS_MAX),
        }

    for src, dst in [(a, b) for a in usable for b in usable if a != b]:
        key = f"{src}->{dst}"
        tr_rows = np.concatenate([fam[src]["pos_all"], fam[src]["neg_all"]])
        y_tr = np.concatenate(
            [np.ones(len(fam[src]["pos_all"])), np.zeros(len(fam[src]["neg_all"]))]
        ).astype(int)
        te_rows = np.concatenate([fam[dst]["pos_test"], fam[dst]["neg_all"]])
        y_te = np.concatenate(
            [np.ones(len(fam[dst]["pos_test"])), np.zeros(len(fam[dst]["neg_all"]))]
        ).astype(int)
        groups_te = np.array([f"ctx::{ectx[i]}" for i in te_rows])
        k5 = max(1, int(round(0.05 * len(y_te))))
        print(
            f"\n[{key}] train n={len(y_tr)} (pos {int(y_tr.sum())}) | "
            f"test n={len(y_te)} (pos {int(y_te.sum())}, base {y_te.mean():.4f}) k5={k5}",
            flush=True,
        )
        dres = {
            "n_train": int(len(y_tr)),
            "n_train_pos": int(y_tr.sum()),
            "n_test": int(len(y_te)),
            "n_test_pos": int(y_te.sum()),
            "test_base_rate": float(y_te.mean()),
            "thin_target": bool(int(y_te.sum()) <= THIN_TARGET_POS_MAX),
            "map_r2": {"indomain_srcfam": {}, "merged": {}},
            "layers": {},
        }
        for L in layers:
            li = LAYERS.index(L)
            vC_tr, vA_tr = eVC[tr_rows, li, :], eVA[tr_rows, li, :]
            vC_te, vA_te = eVC[te_rows, li, :], eVA[te_rows, li, :]
            # maps fit on the SOURCE family only (what you would actually have)
            M_src = fit_ridge_map(vC_tr, vA_tr, RIDGE_LAMBDA)
            M_mrg = fit_ridge_map(
                np.concatenate([bce[L], vC_tr], axis=0),
                np.concatenate([bt1[L], vA_tr], axis=0),
                RIDGE_LAMBDA,
            )
            dres["map_r2"]["indomain_srcfam"][L] = recon_r2(M_src, vC_te, vA_te)
            dres["map_r2"]["merged"][L] = recon_r2(M_mrg, vC_te, vA_te)

            row = {
                "A_transfer": metrics(y_te, fit_apply(vC_tr, y_tr, vC_te), k5),
                "D_transfer_indomain": metrics(
                    y_te, fit_apply(vA_tr, y_tr, apply_map(M_src, vC_te)), k5
                ),
                "D_transfer_merged": metrics(
                    y_te, fit_apply(vA_tr, y_tr, apply_map(M_mrg, vC_te)), k5
                ),
                "E_transfer_oracle": metrics(y_te, fit_apply(vA_tr, y_tr, vA_te), k5),
                "A_within": metrics(y_te, probe_oof_same(vC_te, y_te, groups_te), k5),
                "E_within_oracle": metrics(y_te, probe_oof_same(vA_te, y_te, groups_te), k5),
                "random": metrics(y_te, rng.standard_normal(len(y_te)), k5),
            }
            dres["layers"][L] = row
            print(
                f"  L{L:02d} A_tr {row['A_transfer']['pr_auc']:.3f} "
                f"D_tr_ind {row['D_transfer_indomain']['pr_auc']:.3f} "
                f"D_tr_mrg {row['D_transfer_merged']['pr_auc']:.3f} "
                f"E_tr {row['E_transfer_oracle']['pr_auc']:.3f} "
                f"| A_within {row['A_within']['pr_auc']:.3f} "
                f"E_within {row['E_within_oracle']['pr_auc']:.3f} "
                f"| R2 src {dres['map_r2']['indomain_srcfam'][L]:.3f} "
                f"mrg {dres['map_r2']['merged'][L]:.3f}  [{time.time() - t0:.0f}s]",
                flush=True,
            )
        results["directions"][key] = dres

    results["_meta"] = {
        "dv": "compliance (StrongREJECT-style, evil aligned dimension)",
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "revision": "a09a35458c702b33eeacc393d103063234e8bc28",
        "v_C": "context_end (last prompt token)",
        "v_A": "t1 (answer-span)",
        "layers": layers,
        "prespecified_layer": PRESPEC_LAYER,
        "probe_C": PROBE_C,
        "ridge_lambda": RIDGE_LAMBDA,
        "hidden_dim": HIDDEN,
        "n_benign_map_rows": N_MAP_FIT_BENIGN,
        "families_used": usable,
        "n_directions": len(usable) * (len(usable) - 1),
        "power_caveat": (
            "with 2 usable families there is ONE transfer pair (two directions) and a "
            "direction-level difference is not separable from that pair's "
            "idiosyncrasy; with 3 families there are 6 ordered directions, but any "
            "family flagged thin_target (<= "
            f"{THIN_TARGET_POS_MAX} test positives at the 5% base rate) yields a very "
            "noisy PR-AUC as a TARGET and its as-target rows are directional only"
        ),
        "train_size_note": (
            "transfer probes train on the FULL source family; within-family "
            "references train on 4/5 of the (5%-base) target set -- n_train "
            "differs and is reported per direction"
        ),
        "excluded_family": "evil_toxicchat (0 parsed compliance scores; schema gap)",
    }
    out = DEST / f"transfer_results{args.out_suffix}.json"
    out.write_text(json.dumps(results, indent=1))
    print(f"\n[done] wrote {out}  ({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
