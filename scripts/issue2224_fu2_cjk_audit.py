"""Language-intrusion (CJK) audit of the seed-137 judged pools (issue #2224 fold).

Pure counting — no generation text is printed. Per deciding cell: the fraction
of judged generations containing CJK characters, the cell's graded mean under
the as-judged / intrusion-excluded / intrusion-zeroed conventions, and the 18
deciding paired contrasts' response-level deltas recomputed under exclusion,
with direction agreement against the committed seed-42 deltas.

Run from the issue-2224 worktree root:
    uv run python scripts/issue2224_fu2_cjk_audit.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

CJK = re.compile(r"[一-鿿㐀-䶿豈-﫿぀-ヿ가-힯]")

GEN_ROOT = Path("data/issue_2224/screening_ft_seed137/postft_eval")
SCORES_ROOT = Path("eval_results/issue_2224/followup_r2/selection_finetune_seed137")
COMPARISON = Path("eval_results/issue_2224/followup_r2/seed137_comparison.json")
OUT = Path("eval_results/issue_2224/followup_r2/cjk_audit_seed137.json")


def _cell_flags(cell: str) -> dict[str, bool]:
    """{item key 'qid-gN': contains-CJK} for one cell's generations."""
    flags: dict[str, bool] = {}
    with open(GEN_ROOT / cell / "generations.jsonl", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            key = f"{row['qid']}-g{row['draw']}"
            flags[key] = bool(CJK.search(row.get("response") or ""))
    return flags


def _scores(cell: str) -> dict[str, float]:
    payload = json.loads((SCORES_ROOT / cell / "trait_scores.json").read_text())
    return {
        k: float(v)
        for k, v in payload["trait_expression"]["per_item_scores"].items()
        if v is not None
    }


def _mean(vals: list[float]) -> float | None:
    return sum(vals) / len(vals) if vals else None


def main() -> None:
    comp = json.loads(COMPARISON.read_text())
    cells = sorted(comp["meta"]["deciding_cells"])
    per_cell: dict[str, dict] = {}
    flags_by_cell: dict[str, dict[str, bool]] = {}
    scores_by_cell: dict[str, dict[str, float]] = {}
    for cell in cells:
        flags = _cell_flags(cell)
        scores = _scores(cell)
        flags_by_cell[cell] = flags
        scores_by_cell[cell] = scores
        intruded_keys = {k for k, v in flags.items() if v}
        scored = list(scores.items())
        per_cell[cell] = {
            "n_gens": len(flags),
            "n_intruded": len(intruded_keys),
            "intrusion_rate": round(len(intruded_keys) / max(1, len(flags)), 4),
            "graded_mean_as_judged": round(_mean([v for _, v in scored]), 4),
            "graded_mean_excluded": round(
                _mean([v for k, v in scored if k not in intruded_keys]) or 0.0, 4
            ),
            "graded_mean_zeroed": round(
                _mean([0.0 if k in intruded_keys else v for k, v in scored]), 4
            ),
        }

    contrasts_out = []
    n_agree_excl = 0
    for row in (
        comp["contrasts"].values() if isinstance(comp["contrasts"], dict) else comp["contrasts"]
    ):
        cc = row["value"] if "value" in row else row
        a, b = cc["cell_a"], cc["cell_b"]
        sa, sb = scores_by_cell[a], scores_by_cell[b]
        fa, fb = flags_by_cell[a], flags_by_cell[b]
        shared = sorted(set(sa) & set(sb))
        deltas_all = [sa[k] - sb[k] for k in shared]
        deltas_excl = [sa[k] - sb[k] for k in shared if not (fa.get(k) or fb.get(k))]
        d_all = _mean(deltas_all)
        d_excl = _mean(deltas_excl)
        s42 = cc["seed42"]["response_level"]["mean"]

        def _sign(x: float) -> int:
            return (x > 0) - (x < 0)

        agree_excl = _sign(d_excl) == _sign(s42) or (abs(d_excl) < 1e-9 and abs(s42) < 1e-9)
        n_agree_excl += int(agree_excl)
        contrasts_out.append(
            {
                "contrast": cc["contrast"],
                "corpus": cc["corpus"],
                "trait": cc["trait"],
                "n_paired": len(deltas_all),
                "n_paired_excluded": len(deltas_excl),
                "delta_as_judged": round(d_all, 4),
                "delta_intrusion_excluded": round(d_excl, 4),
                "seed42_delta": s42,
                "direction_agree_vs_seed42_excluded": agree_excl,
            }
        )

    total_gens = sum(c["n_gens"] for c in per_cell.values())
    total_intruded = sum(c["n_intruded"] for c in per_cell.values())
    out = {
        "meta": {
            "script": "issue2224_fu2_cjk_audit",
            "regex": "CJK unified+ext-A / compat ideographs / kana / hangul",
            "note": (
                "pure-count audit of the seed-137 judged pools; the eval panel is drawn from "
                "real prompt banks that contain CJK questions, so intrusion is partly panel-"
                "driven, mirroring the parent run's per-arm audit"
            ),
        },
        "totals": {
            "n_gens": total_gens,
            "n_intruded": total_intruded,
            "rate": round(total_intruded / total_gens, 4),
        },
        "per_cell": per_cell,
        "contrasts": contrasts_out,
        "summary": {
            "n_contrasts": len(contrasts_out),
            "n_direction_agree_vs_seed42_after_exclusion": n_agree_excl,
        },
    }
    OUT.write_text(json.dumps(out, indent=2) + "\n")
    print(
        f"totals: {total_intruded}/{total_gens} intruded ({100 * total_intruded / total_gens:.1f}%)"
    )
    print(f"direction agreement vs seed 42 after exclusion: {n_agree_excl}/{len(contrasts_out)}")
    for c in contrasts_out:
        print(
            f"{c['corpus']}/{c['trait']} {c['contrast']}: as-judged {c['delta_as_judged']:+.3f} "
            f"excluded {c['delta_intrusion_excluded']:+.3f} (s42 {c['seed42_delta']:+.3f})"
        )


if __name__ == "__main__":
    main()
