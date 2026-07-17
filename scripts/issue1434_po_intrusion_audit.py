"""#1434 po round — CJK-intrusion audit of the judged pools the regime lattice rests on.

Pure counting (analyzer Step 3.7): per (arm, read-context) cell, join the panel
judge_raw per-item mean scores (mean of valid draws > 50 = positive; invalid
draws dropped, never coerced — the worker's own convention) with a CJK-script
regex over the SAME completions, then recount the pooled regime contrast
D = po - con under (a) zeroed-intrusion (intruded rows scored non-positive in
BOTH arms) and (b) excluded-intrusion (intruded rows removed from numerator +
denominator). Alignment is gated: the recomputed per-cell judged rate must
match the committed `regime_contrast.json` cell k/n exactly, else the join
aborts loud. Also audits the 4 po Tier-2 verdict pools (install lattice).

Output: eval_results/issue_1434/writing-style-positive-only-regime/po_intrusion_audit.json
Only aggregate counts are printed; no completion text enters any log.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import issue1434_cells as cells  # noqa: E402

DATA_REPO = "superkaiba1/explore-persona-space-data"
PREFIX = "issue1434_writingstyle"
LOCAL = REPO_ROOT / "data" / "issue_1434" / "hf_dl" / "po_scan"
ROUND_DIR = REPO_ROOT / "eval_results" / "issue_1434" / "writing-style-positive-only-regime"
CJK_RE = re.compile(r"[一-鿿㐀-䶿豈-﫿぀-ヿ가-힯]")

VERDICT_ARMS = {  # cell -> (po run id, con run id)
    "pers": ("ws-po-pers-lr1e5", "ws-pers-lr1e5"),
    "bare": ("ws-po-bare-lr1e5", "ws-bare-lr1e5"),
    "conv": ("ws-po-conv-lr3e5", "ws-conv-lr1e5"),
    "icl": ("ws-po-icl-lr1e5", "ws-icl-lr1e5"),
}
PO_CELL = {"pers": "ws-po-pers", "bare": "ws-po-bare", "conv": "ws-po-conv", "icl": "ws-po-icl"}


def dl(path: str) -> Path:
    return Path(hf_hub_download(DATA_REPO, path, repo_type="dataset", local_dir=str(LOCAL)))


def item_scores(judge_path: Path, idmap_path: Path | None) -> dict[str, float]:
    """Per-item mean of valid draw scores from a judge_raw file (drop invalid)."""
    raw = json.loads(judge_path.read_text())
    idmap = json.loads(idmap_path.read_text()) if idmap_path else {}
    per_item: dict[str, list[float]] = {}
    for key, rec in raw["all_scores"].items():
        iid = key.rsplit("__", 2)[0]
        iid = idmap.get(iid, iid)
        s = rec.get("score")
        if isinstance(s, (int, float)) and 0 <= s <= 100:
            per_item.setdefault(iid, []).append(float(s))
    return {iid: sum(v) / len(v) for iid, v in per_item.items() if v}


def flat_completions(comp_path: Path) -> list[str]:
    d = json.loads(comp_path.read_text())
    comps = d["completions"]
    if comps and isinstance(comps[0], list):
        return [c for group in comps for c in group]
    return list(comps)


def audit_cell(tag: str, comp_path: str) -> dict:
    """One judged pool: n, k_pos, intruded, fired-overlap, zeroed/excluded recounts."""
    jp = dl(f"{PREFIX}/judge/pv/judge_raw_pv_{tag}.json")
    imp = None
    try:
        imp = dl(f"{PREFIX}/judge/pv/idmap_{tag}.json")
    except Exception:
        imp = None
    scores = item_scores(jp, imp)
    comps = flat_completions(dl(comp_path))
    # item ids are q-major: {tag}-q{qi:03d}-c{ci:03d}; infer comps-per-q from ids
    ids = sorted(scores.keys())
    n_q = 1 + max(int(i.split("-q")[-1].split("-c")[0]) for i in ids)
    n_c = len(comps) // n_q
    rows = []
    for qi in range(n_q):
        for ci in range(n_c):
            iid = f"{tag}-q{qi:03d}-c{ci:03d}"
            text = comps[qi * n_c + ci]
            if iid in scores:
                rows.append((scores[iid] > 50, bool(CJK_RE.search(text))))
    n = len(rows)
    k = sum(1 for pos, _ in rows if pos)
    n_intr = sum(1 for _, intr in rows if intr)
    k_intr_fired = sum(1 for pos, intr in rows if pos and intr)
    return {
        "n_scored": n,
        "k_positive": k,
        "n_intruded": n_intr,
        "fired_and_intruded": k_intr_fired,
        "k_zeroed": k - k_intr_fired,
        "n_excluded": n - n_intr,
        "k_excluded": k - k_intr_fired,
    }


def pooled(cells_list: list[dict]) -> tuple[int, int, int, int]:
    n = sum(c["n_scored"] for c in cells_list)
    k = sum(c["k_positive"] for c in cells_list)
    kz = sum(c["k_zeroed"] for c in cells_list)
    ne = sum(c["n_excluded"] for c in cells_list)
    ke = sum(c["k_excluded"] for c in cells_list)
    return n, k, kz, ne, ke  # type: ignore[return-value]


def lattice(d: float, lo: float, hi: float) -> str:
    if lo > 0:
        return "Broader-leakage"
    if hi < 0:
        return "Narrower-leakage"
    return "Indistinguishable"


def main() -> None:
    rc = json.loads((ROUND_DIR / "regime_contrast.json").read_text())
    out: dict = {"cells": [], "contexts": {}, "tier2_po": {}}
    for short, (po_run, con_run) in VERDICT_ARMS.items():
        ctx_key = PO_CELL[short]
        entry = rc["contexts"][ctx_key]
        src = entry["source_ctx"]
        read_ctxs = entry["pooled"]["po"]["contexts"]
        po_cells, con_cells = [], []
        for ctx in read_ctxs:
            po_a = audit_cell(
                f"pn-{po_run}-{ctx}",
                f"{PREFIX}/raw_completions/po/panel/{po_run}/completions__trained__{ctx}.json",
            )
            con_a = audit_cell(
                f"pn-{con_run}-{ctx}",
                f"{PREFIX}/raw_completions/panel/{con_run}/completions__trained__{ctx}.json",
            )
            # alignment gate vs committed per-cell aggregates
            match = [
                c for c in rc["cells"] if c["training_cell"] == ctx_key and c["read_ctx"] == ctx
            ]
            assert match, f"no committed cell for {ctx_key}@{ctx}"
            m = match[0]
            assert (po_a["k_positive"], po_a["n_scored"]) == (
                m["po"]["k_positive"],
                m["po"]["n_scored"],
            ), f"po join misaligned at {ctx_key}@{ctx}: {po_a} vs {m['po']}"
            assert (con_a["k_positive"], con_a["n_scored"]) == (
                m["con"]["k_positive"],
                m["con"]["n_scored"],
            ), f"con join misaligned at {ctx_key}@{ctx}: {con_a} vs {m['con']}"
            out["cells"].append(
                {"training_cell": ctx_key, "read_ctx": ctx, "po": po_a, "con": con_a}
            )
            po_cells.append(po_a)
            con_cells.append(con_a)
        # pooled recounts
        pn, pk, pkz, pne, pke = pooled(po_cells)
        cn, ck, ckz, cne, cke = pooled(con_cells)
        res = {"source_ctx": src, "po_run": po_run, "con_run": con_run}
        for name, (k1, n1, k2, n2) in {
            "original": (pk, pn, ck, cn),
            "zeroed": (pkz, pn, ckz, cn),
            "excluded": (pke, pne, cke, cne),
        }.items():
            d = k1 / n1 - k2 / n2
            lo, hi = cells.newcombe(k1, n1, k2, n2)
            res[name] = {
                "po_rate": round(k1 / n1, 4),
                "con_rate": round(k2 / n2, 4),
                "D": round(d, 4),
                "newcombe_95": [round(lo, 4), round(hi, 4)],
                "lattice": lattice(d, lo, hi),
            }
        res["po_intruded_frac"] = round(sum(c["n_intruded"] for c in po_cells) / pn, 4)
        res["con_intruded_frac"] = round(sum(c["n_intruded"] for c in con_cells) / cn, 4)
        out["contexts"][ctx_key] = res
        # po Tier-2 verdict pool (install-lattice instrument): tag = t2-trained-<run_id>
        t2 = audit_cell(
            f"t2-trained-{po_run}",
            f"{PREFIX}/raw_completions/po/tier2/{po_run}/completions__trained__{src}.json",
        )
        out["tier2_po"][ctx_key] = t2
    (ROUND_DIR / "po_intrusion_audit.json").write_text(json.dumps(out, indent=1))
    for ctx_key, r in out["contexts"].items():
        print(
            ctx_key,
            "orig",
            r["original"]["D"],
            r["original"]["lattice"],
            "| zeroed",
            r["zeroed"]["D"],
            r["zeroed"]["lattice"],
            "| excl",
            r["excluded"]["D"],
            r["excluded"]["lattice"],
            "| intr po/con",
            r["po_intruded_frac"],
            r["con_intruded_frac"],
        )
    print("tier2:", json.dumps(out["tier2_po"]))


if __name__ == "__main__":
    main()
