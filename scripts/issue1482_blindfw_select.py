"""Build blinded best/worst-predicted SAE-feature selections at full width (task #1482).

Re-runs the 08-02 blinded qualitative digest against the REAL full-width
dense-context -> SAE answer-feature target
(``issue1482_densesae_fullwidth/perfeature/ridge__mean_perfeature.npz``)
instead of the panel/pre-grid target the original assembly used.

Descriptions come from the #1773 FULL-DICTIONARY release (5 packed shards,
126,110 rows) UNION the 1,692-row recovery, recovery winning on collision.
An earlier run of this digest used ``eval_results/issue_1773/labels/
descriptions.jsonl`` -- a 16,288-row PANEL subset -- which restricted every
selection to a top-activity slice; that run is preserved under
``superseded_panel_descriptions/`` and is NOT the full-width result.

Three selections, all drawn from the same described universe:

A ``raw``       top/bottom-100 by R^2, no gate. The literal analogue of the
                panel read. Its bottom is expected to be numerically
                degenerate -- that degeneracy IS the reported result.
B ``gated``     top/bottom-100 after the MEASURED split-half reliability gate
                (activity deciles clearing r_full >= 0.80). The headline.
C ``matched``   top/bottom-10 by R^2 *within* each ss_tot decile of the gated
                set, so both groups carry identical variance composition.
                Isolates predictability from feature variance.

The A/B group labels handed to the judge are randomized; keys are written to
disk and are not consulted until the verdicts are recorded.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import numpy as np  # noqa: E402  (env must load before heavy imports)

REPO = Path(__file__).resolve().parent.parent
DESC_FULLDICT = Path("/mnt/eps-data/thomasjiralerspong/issue1773_fulldict/labels_upload")
DESC_RECOVERED = REPO / "eval_results/issue_1773/recovery_1934/descriptions_recovered.jsonl"
PANEL_ASSEMBLY = REPO / "eval_results/issue_1482/result2_assembly/top_bottom100_descriptions.json"
OUT_DIR = REPO / "eval_results/issue_1482/result2_assembly_fullwidth"
SUPERSEDED = OUT_DIR / "superseded_panel_descriptions/selections.json"
RELIABILITY_JSON = REPO / "eval_results/issue_1482/r2_reliability/reliability.json"
RELIABILITY_NPZ = (
    REPO / "data/issue_1482/blindfw_dl/issue1482_densesae_fullwidth/"
    "r2_reliability/r2_halves_perfeature.npz"
)

R_FULL_FLOOR = 0.80  # conventional reliability floor for a measurement instrument
SEED = 14822026003  # fresh stream: the panel-subset run's draws must not be reused
N_SELECT = 100
N_STRATA = 10


def load_descriptions() -> tuple[dict[int, str], dict]:
    """#1773 full dictionary (packed shards) UNION recovery; recovery wins on collision."""
    manifest = json.loads((DESC_FULLDICT / "descriptions.manifest.json").read_text())
    base: dict[int, str] = {}
    for shard in sorted(DESC_FULLDICT.glob("descriptions.shard*.jsonl")):
        with shard.open() as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                base[int(obj["feat_id"])] = (obj.get("description") or "").strip()
    rec: dict[int, str] = {}
    with DESC_RECOVERED.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            rec[int(obj["feat_id"])] = (obj.get("description") or "").strip()

    if len(base) != manifest["n_rows"]:
        raise AssertionError(f"shard rows {len(base)} != manifest n_rows {manifest['n_rows']}")

    merged = dict(base)
    merged.update(rec)
    prov = {
        "n_base_fulldict": len(base),
        "n_recovered": len(rec),
        "n_collisions": len(set(base) & set(rec)),
        "n_merged": len(merged),
        "manifest_n_rows": manifest["n_rows"],
        "manifest_sha256": manifest["source_sha256"],
    }
    return merged, prov


def percentile_of(sorted_vals: np.ndarray, v: float) -> float:
    return float(np.searchsorted(sorted_vals, v) / len(sorted_vals) * 100.0)


def degeneracy_table(r2: np.ndarray, sst: np.ndarray) -> list[dict]:
    """R^2 degeneracy by ss_tot decile -- documents the selection trap."""
    order = np.argsort(sst)
    n = len(sst)
    rows = []
    for d in range(10):
        idx = order[d * n // 10 : (d + 1) * n // 10]
        rr, tt = r2[idx], sst[idx]
        rows.append(
            {
                "decile": d + 1,
                "n": int(len(idx)),
                "ss_tot_min": float(tt.min()),
                "ss_tot_max": float(tt.max()),
                "median_r2": float(np.median(rr)),
                "frac_r2_below_neg1": float((rr < -1).mean()),
                "min_r2": float(rr.min()),
            }
        )
    return rows


def group_stats(r2: np.ndarray, sst: np.ndarray, sel: np.ndarray, ssort: np.ndarray) -> dict:
    pcts = [percentile_of(ssort, v) for v in sst[sel]]
    return {
        "n": int(len(sel)),
        "r2_min": float(r2[sel].min()),
        "r2_max": float(r2[sel].max()),
        "r2_median": float(np.median(r2[sel])),
        "ss_tot_pct_median": float(np.median(pcts)),
        "ss_tot_pct_p10": float(np.percentile(pcts, 10)),
        "ss_tot_pct_p90": float(np.percentile(pcts, 90)),
    }


def entries(fid, r2, sst, sel, ssort, desc) -> list[dict]:
    return [
        {
            "feat_id": int(fid[i]),
            "r2": round(float(r2[i]), 5),
            "ss_tot": float(sst[i]),
            "ss_tot_pct": round(percentile_of(ssort, sst[i]), 2),
            "description": desc[int(fid[i])],
        }
        for i in sel
    ]


def reliability_gate() -> dict:
    """MEASURED split-half reliability gate: activity deciles clearing R_FULL_FLOOR."""
    rel = json.loads(RELIABILITY_JSON.read_text())
    ok = [d for d in rel["deciles"] if d["r_full_spearman_brown"] >= R_FULL_FLOOR]
    rz = np.load(RELIABILITY_NPZ)
    return {
        "kind": "measured_split_half_reliability",
        "source": "eval_results/issue_1482/r2_reliability/reliability.json",
        "rule": f"usable AND activity >= min activity of deciles with r_full >= {R_FULL_FLOOR}",
        "activity_floor": min(d["activity_min"] for d in ok),
        "deciles_kept": [d["decile"] for d in ok],
        "r_full_of_kept_deciles": [round(d["r_full_spearman_brown"], 4) for d in ok],
        "pooled_r_full": rel["pooled"]["r_full_spearman_brown"],
        "pooled_n_features": rel["pooled"]["n_features"],
        "_activity": {
            int(f): float(a) for f, a in zip(rz["feat_ids"], rz["activity"], strict=True)
        },
        "_usable": {int(f) for f, u in zip(rz["feat_ids"], rz["usable"], strict=True) if u},
    }


def build_selections(npz_path: Path, desc: dict[int, str], prov: dict) -> dict:
    z = np.load(npz_path)
    fid_all = z["feat_ids"]
    r2_all = z["r2"].astype(np.float64)
    sst_all = z["ss_tot"]
    scored_all = z["scored"]

    sc = np.flatnonzero(scored_all)
    ssort = np.sort(sst_all[sc])
    deg = degeneracy_table(r2_all[sc], sst_all[sc])

    keep = np.array([i for i in sc if desc.get(int(fid_all[i]), "")], dtype=np.int64)
    fid, r2, sst = fid_all[keep], r2_all[keep], sst_all[keep]

    coverage = {
        "n_columns_total": int(len(fid_all)),
        "n_scored": int(len(sc)),
        "n_zero_variance_unscored": int(len(fid_all) - len(sc)),
        "n_described_nonempty": int(len([k for k, v in desc.items() if v and k >= 0])),
        "n_universe_described_and_scored": int(len(keep)),
        "frac_scored_with_description": round(len(keep) / len(sc), 5),
        "descriptions_provenance": prov,
    }

    gate = reliability_gate()
    act, usable = gate.pop("_activity"), gate.pop("_usable")
    gmask = np.array(
        [int(f) in usable and act.get(int(f), -1.0) >= gate["activity_floor"] for f in fid]
    )
    gidx = np.flatnonzero(gmask)
    gate["n_universe_after_gate"] = int(len(gidx))

    order = np.argsort(r2)
    sel_a = {"bottom": order[:N_SELECT], "top": order[-N_SELECT:][::-1]}

    gorder = gidx[np.argsort(r2[gidx])]
    sel_b = {"bottom": gorder[:N_SELECT], "top": gorder[-N_SELECT:][::-1]}

    per = N_SELECT // N_STRATA
    by_sst = gidx[np.argsort(sst[gidx])]
    m = len(by_sst)
    c_bot, c_top = [], []
    for s in range(N_STRATA):
        stratum = by_sst[s * m // N_STRATA : (s + 1) * m // N_STRATA]
        so = stratum[np.argsort(r2[stratum])]
        c_bot.extend(so[:per].tolist())
        c_top.extend(so[-per:][::-1].tolist())
    sel_c = {"bottom": np.array(c_bot, dtype=np.int64), "top": np.array(c_top, dtype=np.int64)}

    out = {
        "coverage": coverage,
        "reliability_gate": gate,
        "ss_tot_degeneracy_table": deg,
        "selections": {},
    }
    for name, sel in (("A_raw", sel_a), ("B_gated", sel_b), ("C_matched", sel_c)):
        out["selections"][name] = {
            "top": entries(fid, r2, sst, sel["top"], ssort, desc),
            "bottom": entries(fid, r2, sst, sel["bottom"], ssort, desc),
            "stats": {
                "top": group_stats(r2, sst, sel["top"], ssort),
                "bottom": group_stats(r2, sst, sel["bottom"], ssort),
            },
        }
    return out


def overlaps(sels: dict) -> dict:
    def ids(blk, side):
        return {e["feat_id"] for e in blk[side]}

    res, names = {}, list(sels)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            for side in ("top", "bottom"):
                res[f"{a}_vs_{b}__{side}"] = len(ids(sels[a], side) & ids(sels[b], side))

    if PANEL_ASSEMBLY.exists():
        panel = json.loads(PANEL_ASSEMBLY.read_text())
        p = {
            "top": {e["feat_id"] for e in panel["top100_by_r2"]},
            "bottom": {e["feat_id"] for e in panel["bottom100_by_r2"]},
        }
        for name in names:
            for side in ("top", "bottom"):
                res[f"panel_vs_{name}__{side}"] = len(p[side] & ids(sels[name], side))

    if SUPERSEDED.exists():
        old = json.loads(SUPERSEDED.read_text())["selections"]
        for name in names:
            if name in old:
                for side in ("top", "bottom"):
                    res[f"panelsubset_vs_{name}__{side}"] = len(
                        ids(old[name], side) & ids(sels[name], side)
                    )
    return res


def write_blinded(sel: dict, name: str, out_dir: Path, rng: random.Random) -> dict:
    """Emit descriptions-only Group A / Group B files + the unread key."""
    flip = rng.random() < 0.5
    mapping = {"A": "bottom", "B": "top"} if flip else {"A": "top", "B": "bottom"}

    payload = {}
    for group, side in mapping.items():
        descs = [e["description"] for e in sel[side]]
        rng.shuffle(descs)
        payload[group] = descs

    lines = [
        f"# Blinded description groups ({name})",
        "",
        "Two groups of autointerp feature descriptions. Read them and report the",
        "common thread that distinguishes Group A from Group B, if any.",
        "",
    ]
    for group in ("A", "B"):
        lines += [f"## Group {group}", ""]
        lines += [f"{i}. {d}" for i, d in enumerate(payload[group], 1)]
        lines.append("")
    (out_dir / f"blinded_{name}.md").write_text("\n".join(lines))
    (out_dir / f"key_{name}.json").write_text(
        json.dumps({"selection": name, "group_to_side": mapping, "seed": SEED}, indent=1) + "\n"
    )
    return {"blinded_file": f"blinded_{name}.md", "n_per_group": len(payload["A"])}


def build_meta(built: dict) -> dict:
    return {
        "source": built["source"],
        "coverage": built["coverage"],
        "reliability_gate": built["reliability_gate"],
        "ss_tot_degeneracy_table": built["ss_tot_degeneracy_table"],
        "overlaps": built["overlaps"],
        "group_stats": {n: s["stats"] for n, s in built["selections"].items()},
        "selection_designs": {
            "A_raw": "top/bottom-100 by R^2 over the described universe, no gate.",
            "B_gated": "top/bottom-100 after the MEASURED split-half reliability gate. HEADLINE.",
            "C_matched": "top/bottom-10 by R^2 within each of 10 ss_tot deciles of the gated "
            "set, so both groups carry identical variance composition.",
        },
        "caveats": [
            "Descriptions are search-index-only at 0.322 neighbour discrimination (#1773). "
            "This digest is a HYPOTHESIS GENERATOR, never evidence.",
            "SUPERSEDES an earlier run that used eval_results/issue_1773/labels/"
            "descriptions.jsonl -- a 16,288-row PANEL subset -- which confined every "
            "selection to a top-activity slice (median ss_tot percentile 82.5) and so was "
            "NOT a full-width read. That run is preserved under "
            "superseded_panel_descriptions/. The panelsubset_vs_* overlaps measure how "
            "much that bias moved the selections.",
            "Selection A's bottom-100 is numerically degenerate by construction and is "
            "reported to document that degeneracy, not as an interpretable 'hard' set.",
            "Selection B's groups can still differ in ss_tot even under the reliability "
            "gate; selection C removes that by construction and is the read to trust "
            "where the two disagree.",
            "The MLP arm is emitted for selection/overlap comparison only; blinded judging "
            "was run on the ridge arm, which is the headline.",
        ],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="ridge", choices=["ridge", "mlp"])
    ap.add_argument(
        "--npz",
        default="data/issue_1482/blindfw_dl/issue1482_densesae_fullwidth/"
        "perfeature/{arm}__mean_perfeature.npz",
    )
    args = ap.parse_args()

    desc, prov = load_descriptions()
    built = build_selections(REPO / args.npz.format(arm=args.arm), desc, prov)
    built["overlaps"] = overlaps(built["selections"])
    built["source"] = {
        "target_npz": f"issue1482_densesae_fullwidth/perfeature/{args.arm}__mean_perfeature.npz",
        "arm": args.arm,
        "descriptions": [
            str(DESC_FULLDICT / "descriptions.shard{000..004}.jsonl"),
            "eval_results/issue_1773/recovery_1934/descriptions_recovered.jsonl "
            "(wins on collision)",
        ],
        "reliability": "eval_results/issue_1482/r2_reliability/reliability.json",
        "seed": SEED,
    }

    out_dir = OUT_DIR if args.arm == "ridge" else OUT_DIR / "mlp"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "selections.json").write_text(json.dumps(built, indent=1) + "\n")

    if args.arm == "ridge":
        rng = random.Random(SEED)
        blind = {
            n: write_blinded(built["selections"][n], n, out_dir, rng) for n in built["selections"]
        }
        (out_dir / "blinding_manifest.json").write_text(json.dumps(blind, indent=1) + "\n")
        (out_dir / "meta.json").write_text(json.dumps(build_meta(built), indent=1) + "\n")

    print(
        json.dumps(
            {
                "coverage": built["coverage"],
                "gate": built["reliability_gate"],
                "overlaps": built["overlaps"],
            },
            indent=1,
        )
    )
    for n, s in built["selections"].items():
        print(n, "TOP", s["stats"]["top"], "\n ", n, "BOT", s["stats"]["bottom"])


if __name__ == "__main__":
    main()
